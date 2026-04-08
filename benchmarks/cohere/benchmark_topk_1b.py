#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark top-K vector search on the FineWeb-Edu ~1B dataset on OBS.

Generates ground truth via flat search, then runs ANN (IVF_PQ) with various
nprobes and measures recall@K and per-query latency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def validate_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got value={value}")
    return value


def parse_int_list(name: str, raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"{name} contains empty item")
        values.append(validate_positive_int(name, int(item)))
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def build_storage_options(
    *,
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
) -> dict[str, str]:
    return {
        "endpoint": endpoint,
        "access_key_id": access_key,
        "secret_access_key": secret_key,
        "region": region,
        "virtual_hosted_style_request": "true",
        "use_opendal": "true",
        "allow_http": "true",
    }


def interpolate_percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    w = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * w


def summarize_latency(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    return {
        "mean": sum(s) / len(s),
        "p50": interpolate_percentile(s, 0.50),
        "p95": interpolate_percentile(s, 0.95),
        "p99": interpolate_percentile(s, 0.99),
        "max": s[-1],
    }


def compute_recall_at_k(
    query_results: list[np.ndarray],
    ground_truth: list[np.ndarray],
    k: int,
) -> float:
    recalls = []
    for result_ids, truth_ids in zip(query_results, ground_truth):
        truth_set = set(truth_ids[:k].tolist())
        overlap = sum(1 for rid in result_ids[:k].tolist() if rid in truth_set)
        recalls.append(overlap / k)
    return sum(recalls) / len(recalls)


def sample_query_vectors(
    dataset,
    *,
    query_count: int,
    vector_column: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample query vectors and their row IDs from the dataset."""
    print(f"Sampling {query_count} query vectors from dataset...")
    sample_table = dataset.sample(query_count, columns=[vector_column], seed=seed)
    vectors = np.stack(sample_table[vector_column].to_pylist())
    return vectors


def generate_ground_truth(
    dataset,
    query_vectors: np.ndarray,
    *,
    vector_column: str,
    top_k: int,
    batch_size: int = 100,
) -> np.ndarray:
    """Generate ground truth via flat brute-force search.

    Returns shape [num_queries, top_k] with row IDs.
    """
    num_queries = len(query_vectors)
    print(f"Generating ground truth: {num_queries} queries, top_k={top_k}")
    all_gt_ids = []

    for i in range(0, num_queries, batch_size):
        batch_end = min(i + batch_size, num_queries)
        batch_queries = query_vectors[i:batch_end]
        batch_gt = []

        for q in batch_queries:
            start = time.perf_counter()
            result = dataset.to_table(
                columns=["_rowid"],
                nearest={
                    "column": vector_column,
                    "q": q.tolist(),
                    "k": top_k,
                    "use_index": False,
                },
            )
            ids = result["_rowid"].to_numpy()
            batch_gt.append(ids)
            elapsed = time.perf_counter() - start
            if i == 0 and len(batch_gt) <= 2:
                print(f"  Query 0 flat search: {elapsed * 1000:.1f}ms, rows={len(ids)}")

        all_gt_ids.extend(batch_gt)
        done = batch_end
        if done % 10 == 0 or done == num_queries:
            print(f"  Ground truth progress: {done}/{num_queries}")

    # Pad to uniform length if needed
    max_len = max(len(gt) for gt in all_gt_ids)
    padded = np.zeros((num_queries, max_len), dtype=np.int64)
    for i, gt in enumerate(all_gt_ids):
        padded[i, : len(gt)] = gt
    return padded[:, :top_k]


def try_torch_ground_truth(
    dataset,
    query_vectors: np.ndarray,
    *,
    vector_column: str,
    top_k: int,
) -> np.ndarray | None:
    """Try using lance.torch for GPU-accelerated ground truth."""
    try:
        from lance.torch.bench_utils import ground_truth

        print("Using lance.torch for GPU-accelerated ground truth...")
        gt = ground_truth(
            dataset,
            vector_column,
            query_vectors,
            metric_type="cosine",
            k=top_k,
            batch_size=10240,
        )
        return gt.numpy()
    except (ImportError, RuntimeError) as e:
        print(f"lance.torch ground truth not available: {e}")
        return None


def run_ann_experiment(
    dataset,
    *,
    experiment_name: str,
    query_vectors: np.ndarray,
    top_k: int,
    warmup_queries: int,
    vector_column: str,
    metric: str,
    nprobes: int,
    refine_factor: int,
) -> dict:
    latency_samples_ms = []
    result_ids_per_query = []
    rows_returned = []

    for qi, qvec in enumerate(query_vectors):
        start = time.perf_counter()
        result = dataset.to_table(
            columns=["_rowid", "_distance"],
            nearest={
                "column": vector_column,
                "q": qvec.tolist(),
                "k": top_k,
                "use_index": True,
                "metric": metric,
                "nprobes": nprobes,
                "refine_factor": refine_factor,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if qi >= warmup_queries:
            latency_samples_ms.append(elapsed_ms)
            rows_returned.append(result.num_rows)
            result_ids_per_query.append(result["_rowid"].to_numpy())

    return {
        "name": experiment_name,
        "nprobes": nprobes,
        "refine_factor": refine_factor,
        "latency_ms": summarize_latency(latency_samples_ms),
        "avg_rows_returned": sum(rows_returned) / len(rows_returned) if rows_returned else 0,
        "query_count": len(query_vectors) - warmup_queries,
        "result_ids": result_ids_per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark top-K search on FineWeb-Edu ~1B dataset on OBS."
    )
    parser.add_argument(
        "--dataset-uri",
        default="s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance",
    )
    parser.add_argument("--endpoint", default="https://obs.ap-southeast-1.myhuaweicloud.com")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--access-key", default="")
    parser.add_argument("--secret-key", default="")
    parser.add_argument("--vector-column", default="vector")
    parser.add_argument("--metric", default="cosine", choices=["L2", "cosine", "dot"])
    parser.add_argument("--top-k", type=int, default=10000)
    parser.add_argument("--recall-k", type=int, default=None, help="K for recall computation. Defaults to top-k.")
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--nprobes", default="32,64,128,256")
    parser.add_argument("--refine-factor", default="5,10,20")
    parser.add_argument("--skip-flat", action="store_true", help="Skip flat ground truth generation")
    parser.add_argument("--ground-truth-path", default=None, help="Path to load/save ground truth (npz)")
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument(
        "--result-path",
        default="/data/work/lance-bench-results/fineweb-edu-1b-topk.json",
    )
    parser.add_argument("--run-label", default="fineweb-edu-1b-topk10k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_positive_int("top_k", args.top_k)
    validate_positive_int("query_count", args.query_count)

    recall_k = args.recall_k or args.top_k

    access_key = args.access_key
    secret_key = args.secret_key
    if not access_key or not secret_key:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        raise ValueError("access-key and secret-key are required")

    storage_opts = build_storage_options(
        endpoint=args.endpoint,
        access_key=access_key,
        secret_key=secret_key,
        region=args.region,
    )

    print(f"Opening dataset: {args.dataset_uri}")
    dataset = lance.dataset(args.dataset_uri, storage_options=storage_opts)
    num_rows = dataset.count_rows()
    print(f"Dataset rows: {num_rows:,}")

    # Check index exists
    indices = dataset.list_indices()
    if not indices:
        print("WARNING: No vector index found. Run build_index_1b.py first.")
    else:
        print(f"Indices: {[(idx.name, idx.type) for idx in indices]}")

    # Sample query vectors
    query_vectors = sample_query_vectors(
        dataset,
        query_count=args.query_count,
        vector_column=args.vector_column,
        seed=args.query_seed,
    )
    print(f"Query vectors shape: {query_vectors.shape}")

    # Ground truth
    ground_truth = None
    if args.ground_truth_path and Path(args.ground_truth_path).exists():
        print(f"Loading ground truth from {args.ground_truth_path}")
        gt_data = np.load(args.ground_truth_path)
        ground_truth = gt_data["ground_truth"]
        print(f"Ground truth shape: {ground_truth.shape}")

    if ground_truth is None and not args.skip_flat:
        # Try torch first, fall back to flat search
        ground_truth = try_torch_ground_truth(
            dataset,
            query_vectors,
            vector_column=args.vector_column,
            top_k=recall_k,
        )
        if ground_truth is None:
            ground_truth = generate_ground_truth(
                dataset,
                query_vectors,
                vector_column=args.vector_column,
                top_k=recall_k,
            )
        print(f"Ground truth shape: {ground_truth.shape}")

        if args.ground_truth_path:
            print(f"Saving ground truth to {args.ground_truth_path}")
            Path(args.ground_truth_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.ground_truth_path, ground_truth=ground_truth)

    # Run ANN experiments
    nprobes_list = parse_int_list("nprobes", args.nprobes)
    refine_factors = parse_int_list("refine_factor", args.refine_factor)

    experiments = []
    for nprobes in nprobes_list:
        for rf in refine_factors:
            name = f"ivf_pq_np{nprobes}_rf{rf}"
            print(f"\nRunning {name}...")
            result = run_ann_experiment(
                dataset,
                experiment_name=name,
                query_vectors=query_vectors,
                top_k=args.top_k,
                warmup_queries=args.warmup_queries,
                vector_column=args.vector_column,
                metric=args.metric,
                nprobes=nprobes,
                refine_factor=rf,
            )

            # Compute recall
            if ground_truth is not None:
                recall = compute_recall_at_k(
                    result["result_ids"],
                    ground_truth[args.warmup_queries:],
                    recall_k,
                )
                result[f"recall_at_{recall_k}"] = recall
                print(f"  recall@{recall_k}: {recall:.4f}")

            print(f"  latency: {result['latency_ms']['mean']:.1f}ms mean, "
                  f"{result['latency_ms']['p99']:.1f}ms p99")
            del result["result_ids"]  # Don't include raw IDs in output
            experiments.append(result)

    # Build result document
    result_doc = {
        "run_label": args.run_label,
        "host": socket.gethostname(),
        "dataset": {
            "uri": args.dataset_uri,
            "rows": num_rows,
            "vector_column": args.vector_column,
            "metric": args.metric,
        },
        "query": {
            "count": args.query_count,
            "top_k": args.top_k,
            "recall_k": recall_k,
            "warmup_queries": args.warmup_queries,
            "seed": args.query_seed,
        },
        "experiments": experiments,
    }

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result_doc, indent=2, sort_keys=True) + "\n")
    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
