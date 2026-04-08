#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Distributed query merge demo: compare Approach A vs B for vector search.

Approach A (Lance native): single dataset with multi-segment IVF-PQ index.
    Query: dataset.to_table(nearest={...}) — Lance internally merges.

Approach B (Manual sharding): N independent datasets, each with its own index.
    Query: N parallel processes → merge partial top-K → global top-K.

Usage:
    # Full comparison on 1M test dataset
    python benchmarks/cohere/distributed_query_demo.py \
        --source-uri s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-raw-1m.lance \
        --num-shards 4,8 \
        --top-k 1000 \
        --query-count 50

    # Quick test locally
    python benchmarks/cohere/distributed_query_demo.py \
        --source-uri /tmp/test.lance \
        --num-shards 2 \
        --top-k 100 \
        --query-count 10 \
        --num-partitions 16

    # Memory vs Disk benchmark (local NVMe shards with RQ index)
    python benchmarks/cohere/distributed_query_demo.py \
        --bench-memory-vs-disk \
        --shard-dir /data/work/lance-shards-rq \
        --query-count 20 --warmup 3

    # Memory vs Disk benchmark (needs sudo for drop_caches)
    sudo python benchmarks/cohere/distributed_query_demo.py \
        --bench-memory-vs-disk \
        --shard-dir /data/work/lance-shards-rq \
        --query-count 20 --warmup 3
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


# ── Helpers ──────────────────────────────────────────────────────────


def build_storage_options(args) -> dict[str, str]:
    opts = {
        "endpoint": args.endpoint,
        "region": args.region,
        "virtual_hosted_style_request": "true",
        "allow_http": "true",
    }
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        opts["access_key_id"] = args.access_key
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        opts["secret_access_key"] = args.secret_key
    return opts


def interpolate_percentile(sorted_values: list[float], p: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    w = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * w


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
    results: list[np.ndarray],
    ground_truth: list[np.ndarray],
    k: int,
) -> float:
    recalls = []
    for res_ids, gt_ids in zip(results, ground_truth):
        gt_set = set(gt_ids[:k].tolist())
        overlap = sum(1 for rid in res_ids[:k].tolist() if rid in gt_set)
        recalls.append(overlap / k)
    return sum(recalls) / len(recalls)


def drop_page_cache():
    """Drop Linux page cache (requires root)."""
    subprocess.run(
        ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=True,
    )


def warm_page_cache(shard_dir: str, num_shards: int):
    """Read all files under shard dirs to populate page cache."""
    for i in range(num_shards):
        shard_path = Path(shard_dir) / f"shard-{i}.lance"
        if not shard_path.exists():
            continue
        for fpath in shard_path.rglob("*"):
            if fpath.is_file() and fpath.stat().st_size > 0:
                try:
                    with open(fpath, "rb") as f:
                        while f.read(64 * 1024 * 1024):
                            pass
                except (OSError, PermissionError):
                    pass


# ── Approach A: Single dataset, multi-segment index ──────────────────


def approach_a_build_index(
    dataset,
    *,
    column: str,
    metric: str,
    num_partitions: int,
    num_sub_vectors: int,
    num_bits: int,
    num_workers: int,
) -> float:
    """Build IVF-PQ index with shared centroids (distributed build)."""
    from lance.indices import IndicesBuilder

    print(f"[A] Building distributed index ({num_workers} workers)...")
    t0 = time.perf_counter()

    builder = IndicesBuilder(dataset, column)
    pre = builder.prepare_global_ivf_pq(
        num_partitions=num_partitions,
        num_subvectors=num_sub_vectors,
        distance_type=metric,
        sample_rate=256,
        max_iters=50,
    )

    frags = dataset.get_fragments()
    frag_ids = [f.fragment_id for f in frags]
    chunk_size = math.ceil(len(frag_ids) / num_workers)
    groups = [frag_ids[i : i + chunk_size] for i in range(0, len(frag_ids), chunk_size)]

    segments = []
    for g in groups:
        if not g:
            continue
        seg = dataset.create_index_uncommitted(
            column=column,
            index_type="IVF_PQ",
            fragment_ids=g,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            num_bits=num_bits,
            ivf_centroids=pre["ivf_centroids"],
            pq_codebook=pre["pq_codebook"],
        )
        segments.append(seg)

    merged = dataset.create_index_segment_builder().with_segments(segments).build_all()
    dataset.commit_existing_index_segments("vector_idx", column, merged)

    elapsed = time.perf_counter() - t0
    print(f"[A] Index built in {elapsed:.1f}s ({len(segments)} segments)")
    return elapsed


def approach_a_query(
    dataset,
    query_vectors: np.ndarray,
    *,
    column: str,
    metric: str,
    top_k: int,
    nprobes: int,
    warmup: int,
    refine_factor: int,
) -> dict:
    """Query Approach A: single call, Lance merges internally."""
    latency_ms = []
    result_ids = []

    for qi, qvec in enumerate(query_vectors):
        start = time.perf_counter()
        result = dataset.to_table(
            columns=["_rowid", "_distance"],
            nearest={
                "column": column,
                "q": qvec.tolist(),
                "k": top_k,
                "use_index": True,
                "metric": metric,
                "nprobes": nprobes,
                "refine_factor": refine_factor,
            },
        )
        elapsed = (time.perf_counter() - start) * 1000.0

        if qi >= warmup:
            latency_ms.append(elapsed)
            result_ids.append(result["_rowid"].to_numpy())

    return {
        "approach": "A",
        "latency_ms": summarize_latency(latency_ms),
        "result_ids": result_ids,
    }


# ── Approach B: Sharded datasets, parallel query + merge ─────────────


def _query_shard_worker(args_tuple):
    """Worker function for Approach B — queries a single shard."""
    (
        shard_uri,
        storage_opts,
        query_vectors,
        column,
        metric,
        top_k,
        nprobes,
        refine_factor,
    ) = args_tuple

    # Each worker opens its own dataset
    ds = lance.dataset(shard_uri, storage_options=storage_opts)
    partial_results = []

    for qvec in query_vectors:
        result = ds.to_table(
            columns=["_rowid", "_distance"],
            nearest={
                "column": column,
                "q": qvec.tolist(),
                "k": top_k,
                "use_index": True,
                "metric": metric,
                "nprobes": nprobes,
                "refine_factor": refine_factor,
            },
        )
        rowids = result["_rowid"].to_numpy()
        dists = result["_distance"].to_numpy()
        partial_results.append(np.column_stack([dists, rowids]))

    return partial_results


def merge_top_k(
    partial_results_per_shard: list[list[np.ndarray]],
    top_k: int,
) -> list[np.ndarray]:
    """Merge N shards' partial top-K into global top-K.

    Each element: shape [local_k, 2] where col 0 = distance, col 1 = row_id.
    For Approach B with separate datasets, row_ids need a global offset.
    """
    num_queries = len(partial_results_per_shard[0])
    global_results = []

    for qi in range(num_queries):
        all_candidates = []
        for shard_results in partial_results_per_shard:
            all_candidates.append(shard_results[qi])
        stacked = np.vstack(all_candidates)
        # Sort by distance (ascending = closer = better)
        sorted_idx = np.argsort(stacked[:, 0])
        global_results.append(stacked[sorted_idx[:top_k], 1].astype(np.int64))

    return global_results


def approach_b_setup(
    source_ds,
    *,
    work_dir: str,
    num_shards: int,
    column: str,
    metric: str,
    num_partitions: int,
    num_sub_vectors: int,
    num_bits: int,
    storage_opts: dict | None = None,
    is_s3: bool = False,
) -> tuple[list[str], float]:
    """Split source dataset into N shards and build index on each."""
    print(f"[B] Setting up {num_shards} shards...")
    t0 = time.perf_counter()

    total_rows = source_ds.count_rows()
    rows_per_shard = math.ceil(total_rows / num_shards)
    shard_uris = []

    for i in range(num_shards):
        start_row = i * rows_per_shard
        end_row = min(start_row + rows_per_shard, total_rows)
        if start_row >= total_rows:
            break

        if is_s3:
            shard_uri = f"{work_dir}/shard-{i}.lance"
        else:
            shard_uri = str(Path(work_dir) / f"shard-{i}.lance")
            if Path(shard_uri).exists():
                shutil.rmtree(shard_uri)

        print(f"  Shard {i}: rows {start_row}-{end_row}...")
        shard_table = source_ds.to_table(
            columns=[column],
            offset=start_row,
            limit=end_row - start_row,
        )
        # Include id column if present
        if "id" in source_ds.schema.names:
            id_table = source_ds.to_table(
                columns=["id"],
                offset=start_row,
                limit=end_row - start_row,
            )
            shard_table = pa.table(
                {"id": id_table["id"], column: shard_table[column]}
            )

        shard_ds = lance.write_dataset(
            shard_table, shard_uri, mode="overwrite", storage_options=storage_opts
        )

        # Build index on this shard
        shard_num_partitions = max(num_partitions // num_shards, 4)
        shard_ds.create_index(
            column=column,
            index_type="IVF_PQ",
            metric=metric,
            num_partitions=shard_num_partitions,
            num_sub_vectors=num_sub_vectors,
            num_bits=num_bits,
        )
        shard_uris.append(shard_uri)
        print(f"    Done: {shard_ds.count_rows()} rows, index built")

    elapsed = time.perf_counter() - t0
    print(f"[B] Setup complete in {elapsed:.1f}s ({len(shard_uris)} shards)")
    return shard_uris, elapsed


def approach_b_query(
    shard_uris: list[str],
    query_vectors: np.ndarray,
    *,
    storage_opts: dict | None,
    column: str,
    metric: str,
    top_k: int,
    nprobes: int,
    warmup: int,
    refine_factor: int,
    max_workers: int | None = None,
) -> dict:
    """Query Approach B: parallel shard queries + merge."""
    num_shards = len(shard_uris)
    workers = max_workers or min(num_shards, os.cpu_count() or 4)
    k_per_shard = top_k  # Each shard returns top-K, merge takes global top-K

    # Warmup run (sequential to avoid interference)
    for uri in shard_uris[:1]:
        ds = lance.dataset(uri, storage_options=storage_opts)
        ds.to_table(
            columns=["_rowid", "_distance"],
            nearest={
                "column": column,
                "q": query_vectors[0].tolist(),
                "k": top_k,
                "use_index": True,
                "metric": metric,
                "nprobes": nprobes,
                "refine_factor": refine_factor,
            },
        )

    # Timed runs
    latency_ms = []
    num_queries = len(query_vectors)

    for qi in range(warmup, num_queries):
        qvec = query_vectors[qi]
        start = time.perf_counter()

        # Parallel query across shards
        args_list = [
            (
                uri,
                storage_opts,
                [qvec],
                column,
                metric,
                k_per_shard,
                nprobes,
                refine_factor,
            )
            for uri in shard_uris
        ]

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_query_shard_worker, a) for a in args_list]
            partial_results = [f.result() for f in as_completed(futures)]

        # Reorder by shard index (as_completed may be out of order)
        # partial_results[i] is list of [query_results] from one shard
        merged_ids = merge_top_k(partial_results, top_k)

        elapsed = (time.perf_counter() - start) * 1000.0
        latency_ms.append(elapsed)

    return {
        "approach": "B",
        "num_shards": num_shards,
        "workers": workers,
        "latency_ms": summarize_latency(latency_ms),
        "merge_overhead_note": "ProcessPool startup included in timing",
    }


# ── Ground Truth ─────────────────────────────────────────────────────


def generate_ground_truth(
    dataset,
    query_vectors: np.ndarray,
    *,
    column: str,
    top_k: int,
    metric: str,
) -> np.ndarray:
    """Flat brute-force ground truth."""
    print(f"Generating ground truth ({len(query_vectors)} queries, top_k={top_k})...")
    all_gt = []
    for i, qvec in enumerate(query_vectors):
        result = dataset.to_table(
            columns=["_rowid"],
            nearest={
                "column": column,
                "q": qvec.tolist(),
                "k": top_k,
                "use_index": False,
                "metric": metric,
            },
        )
        all_gt.append(result["_rowid"].to_numpy())
        if (i + 1) % 10 == 0:
            print(f"  GT progress: {i + 1}/{len(query_vectors)}")
    return np.stack(all_gt)


# ── Memory vs Disk Benchmark ───────────────────────────────────────


def bench_memory_vs_disk(args) -> None:
    """Benchmark Plan B: page cache hot (all in RAM) vs cold (all from NVMe)."""
    shard_dir = args.shard_dir
    num_shards = args.num_shards_mem_disk or 4
    column = args.column
    metric = args.metric
    top_k = args.top_k
    nprobes = args.nprobes
    refine_factor = args.refine_factor
    query_count = args.query_count
    warmup = args.warmup
    seed = args.seed
    max_workers = args.max_workers

    # Validate shards
    shard_uris = [
        str(Path(shard_dir) / f"shard-{i}.lance")
        for i in range(num_shards)
    ]
    for uri in shard_uris:
        if not Path(uri).exists():
            print(f"ERROR: Shard not found: {uri}")
            sys.exit(1)

    # Print shard info
    for i, uri in enumerate(shard_uris):
        ds = lance.dataset(uri)
        indices = ds.list_indices()
        idx_types = [idx["type"] if isinstance(idx, dict) else idx.type for idx in indices]
        print(f"  Shard {i}: {ds.count_rows():,} rows, indices={idx_types}")
        if not indices:
            print(f"  WARNING: Shard {i} has no index! Queries will be slow.")

    # Sample query vectors
    print(f"\nSampling {query_count} query vectors (seed={seed})...")
    rng = np.random.default_rng(seed)
    ds0 = lance.dataset(shard_uris[0])
    total_rows = ds0.count_rows()
    sample_indices = rng.choice(total_rows, size=query_count, replace=False)
    query_table = ds0.take(sample_indices, columns=[column])
    query_vectors = np.stack(query_table[column].to_pylist())
    print(f"  Query shape: {query_vectors.shape}")

    results = {}

    # Test 1: Page Cache Hot (all in RAM)
    if not args.skip_memory:
        print("\n" + "=" * 60)
        print("TEST 1: Page Cache Hot (all in RAM)")
        print("=" * 60)

        print("Warming page cache (reading all shard files)...")
        t0 = time.perf_counter()
        warm_page_cache(shard_dir, num_shards)
        warm_elapsed = time.perf_counter() - t0
        print(f"  Page cache warmed in {warm_elapsed:.1f}s")

        # Warmup queries
        print(f"Running {warmup} warmup queries...")
        for qi in range(warmup):
            approach_b_query(
                shard_uris, query_vectors[qi:qi+1],
                storage_opts=None,
                column=column, metric=metric,
                top_k=top_k, nprobes=nprobes,
                warmup=0, refine_factor=refine_factor,
                max_workers=max_workers,
            )

        # Timed queries
        num_timed = query_count - warmup
        print(f"Running {num_timed} timed queries...")
        latency_ms = []
        for qi in range(warmup, query_count):
            _, elapsed = _bench_single_query(
                shard_uris, query_vectors[qi],
                column=column, metric=metric,
                top_k=top_k, nprobes=nprobes,
                refine_factor=refine_factor,
                max_workers=max_workers,
            )
            latency_ms.append(elapsed)
            if (qi - warmup + 1) % 5 == 0:
                print(f"  Query {qi + 1}/{query_count}: {elapsed:.0f}ms")

        mem_summary = summarize_latency(latency_ms)
        results["memory"] = {
            "label": "Page Cache Hot (all in RAM)",
            "latency_ms": mem_summary,
            "page_cache_warm_s": warm_elapsed,
            "query_count": len(latency_ms),
        }
        print(f"\n  Memory Results:")
        print(f"    Mean: {mem_summary['mean']:.1f}ms")
        print(f"    P50:  {mem_summary['p50']:.1f}ms")
        print(f"    P95:  {mem_summary['p95']:.1f}ms")
        print(f"    P99:  {mem_summary['p99']:.1f}ms")
        print(f"    Max:  {mem_summary['max']:.1f}ms")

    # Test 2: Page Cache Cold (all from NVMe, drop_caches per query)
    if not args.skip_disk:
        print("\n" + "=" * 60)
        print("TEST 2: Page Cache Cold (drop_caches per query)")
        print("=" * 60)

        try:
            drop_page_cache()
            print("  drop_caches OK")
        except (subprocess.CalledProcessError, PermissionError) as e:
            print(f"  ERROR: Cannot drop page cache (need root): {e}")
            print(f"  Run with: sudo python {sys.argv[0]} ...")
            sys.exit(1)

        # Warmup with drop_caches
        print(f"Running {warmup} warmup queries (each with drop_caches)...")
        for qi in range(warmup):
            drop_page_cache()
            _bench_single_query(
                shard_uris, query_vectors[qi],
                column=column, metric=metric,
                top_k=top_k, nprobes=nprobes,
                refine_factor=refine_factor,
                max_workers=max_workers,
            )

        # Timed queries with drop_caches
        num_timed = query_count - warmup
        print(f"Running {num_timed} timed queries (each with drop_caches)...")
        latency_ms = []
        for qi in range(warmup, query_count):
            drop_page_cache()
            _, elapsed = _bench_single_query(
                shard_uris, query_vectors[qi],
                column=column, metric=metric,
                top_k=top_k, nprobes=nprobes,
                refine_factor=refine_factor,
                max_workers=max_workers,
            )
            latency_ms.append(elapsed)
            if (qi - warmup + 1) % 5 == 0:
                print(f"  Query {qi + 1}/{query_count}: {elapsed:.0f}ms")

        disk_summary = summarize_latency(latency_ms)
        results["disk"] = {
            "label": "Page Cache Cold (all from NVMe)",
            "latency_ms": disk_summary,
            "query_count": len(latency_ms),
        }
        print(f"\n  Disk Results:")
        print(f"    Mean: {disk_summary['mean']:.1f}ms")
        print(f"    P50:  {disk_summary['p50']:.1f}ms")
        print(f"    P95:  {disk_summary['p95']:.1f}ms")
        print(f"    P99:  {disk_summary['p99']:.1f}ms")
        print(f"    Max:  {disk_summary['max']:.1f}ms")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'':>4} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10} {'Max(ms)':>10}")
    print("-" * 70)

    for key in ["memory", "disk"]:
        if key in results:
            r = results[key]["latency_ms"]
            label = results[key]["label"]
            print(f"  {label:<25} {r['mean']:>10.1f} {r['p50']:>10.1f} {r['p95']:>10.1f} {r['p99']:>10.1f} {r['max']:>10.1f}")

    if "memory" in results and "disk" in results:
        mem_mean = results["memory"]["latency_ms"]["mean"]
        disk_mean = results["disk"]["latency_ms"]["mean"]
        ratio = disk_mean / mem_mean if mem_mean > 0 else float("inf")
        print(f"\n  Disk/Memory latency ratio: {ratio:.2f}x")
        print(f"  Disk overhead: {disk_mean - mem_mean:.1f}ms")

    # Save results
    result_doc = {
        "host": socket.gethostname(),
        "benchmark": "memory_vs_disk",
        "config": {
            "shard_dir": shard_dir,
            "num_shards": num_shards,
            "metric": metric,
            "top_k": top_k,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
            "query_count": query_count,
            "warmup": warmup,
        },
        "results": results,
    }

    result_path = args.result_path or str(Path(shard_dir) / "bench_memory_vs_disk.json")
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result_path).write_text(json.dumps(result_doc, indent=2) + "\n")
    print(f"\nResults saved to {result_path}")


def _bench_single_query(
    shard_uris: list[str],
    qvec: np.ndarray,
    *,
    column: str,
    metric: str,
    top_k: int,
    nprobes: int,
    refine_factor: int,
    max_workers: int | None = None,
) -> tuple[np.ndarray, float]:
    """Run a single distributed query, return (merged_ids, latency_ms)."""
    workers = max_workers or min(len(shard_uris), os.cpu_count() or 4)
    start = time.perf_counter()

    args_list = [
        (uri, None, [qvec], column, metric, top_k, nprobes, refine_factor)
        for uri in shard_uris
    ]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_query_shard_worker, a) for a in args_list]
        partial_results = [f.result() for f in as_completed(futures)]

    merged = merge_top_k(partial_results, top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return merged, elapsed_ms


# ── Main ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare distributed query approaches A vs B."
    )
    p.add_argument(
        "--source-uri",
        default="s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-raw-1m.lance",
    )
    p.add_argument("--work-dir", default="/tmp/lance-dist-query-demo")
    p.add_argument("--endpoint", default="https://obs.ap-southeast-1.myhuaweicloud.com")
    p.add_argument("--region", default="ap-southeast-1")
    p.add_argument("--access-key", default="")
    p.add_argument("--secret-key", default="")
    p.add_argument("--column", default="vector")
    p.add_argument("--metric", default="cosine", choices=["L2", "cosine", "dot"])
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--query-count", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nprobes", type=int, default=32)
    p.add_argument("--refine-factor", type=int, default=1)
    p.add_argument("--num-partitions", type=int, default=None)
    p.add_argument("--num-sub-vectors", type=int, default=64)
    p.add_argument("--num-bits", type=int, default=8)
    p.add_argument("--num-shards", default="2,4,8")
    p.add_argument("--skip-setup", action="store_true")
    p.add_argument("--result-path", default=None)
    p.add_argument("--skip-ground-truth", action="store_true")

    # Memory vs Disk benchmark
    p.add_argument("--bench-memory-vs-disk", action="store_true",
                   help="Run memory vs disk benchmark instead of A/B comparison")
    p.add_argument("--shard-dir", default=None,
                   help="Directory containing shard-{0..N}.lance (for --bench-memory-vs-disk)")
    p.add_argument("--num-shards-mem-disk", type=int, default=None,
                   help="Number of shards for memory vs disk benchmark")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--skip-memory", action="store_true", help="Skip memory test")
    p.add_argument("--skip-disk", action="store_true", help="Skip disk test")

    return p.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    # Memory vs Disk benchmark mode
    if args.bench_memory_vs_disk:
        if not args.shard_dir:
            print("ERROR: --shard-dir required with --bench-memory-vs-disk")
            sys.exit(1)
        bench_memory_vs_disk(args)
        return

    access_key = args.access_key or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = args.secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    storage_opts = None
    is_s3 = args.source_uri.startswith("s3://")

    if is_s3:
        if not access_key or not secret_key:
            raise ValueError("access-key/secret-key required for S3")
        storage_opts = build_storage_options(args)

    print(f"Opening source: {args.source_uri}")
    source_ds = lance.dataset(args.source_uri, storage_options=storage_opts)
    num_rows = source_ds.count_rows()
    print(f"Rows: {num_rows:,}")

    num_partitions = args.num_partitions or max(
        int(math.sqrt(num_rows)), 16
    )
    shard_counts = parse_int_list(args.num_shards)
    print(f"num_partitions={num_partitions}, shard_counts={shard_counts}")

    # Sample query vectors
    print(f"Sampling {args.query_count} queries (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(num_rows, size=min(args.query_count, num_rows), replace=False)
    query_table = source_ds.take(sample_indices, columns=[args.column])
    query_vectors = np.stack(query_table[args.column].to_pylist())
    print(f"Query shape: {query_vectors.shape}")

    # Ground truth
    ground_truth = None
    if not args.skip_ground_truth:
        ground_truth = generate_ground_truth(
            source_ds, query_vectors, column=args.column,
            top_k=args.top_k, metric=args.metric,
        )

    # ── Approach A ──
    work_dir = args.work_dir
    approach_a_uri = str(Path(work_dir) / "approach_a.lance") if not is_s3 else f"{work_dir}/approach_a.lance"

    if not args.skip_setup:
        if not is_s3 and Path(approach_a_uri).exists():
            shutil.rmtree(approach_a_uri)
        print("\n=== Approach A: Setup ===")
        t0 = time.perf_counter()
        a_ds = lance.write_dataset(
            source_ds.to_table(),
            approach_a_uri,
            mode="overwrite",
            max_rows_per_file=num_rows // 4,  # Ensure 4+ fragments for distributed build
            storage_options=storage_opts,
        )
        write_time = time.perf_counter() - t0
        print(f"Written in {write_time:.1f}s, fragments={len(a_ds.get_fragments())}")

        a_index_time = approach_a_build_index(
            a_ds,
            column=args.column,
            metric=args.metric,
            num_partitions=num_partitions,
            num_sub_vectors=args.num_sub_vectors,
            num_bits=args.num_bits,
            num_workers=4,
        )
    else:
        a_ds = lance.dataset(approach_a_uri, storage_options=storage_opts)
        a_index_time = 0

    print("\n=== Approach A: Query ===")
    a_result = approach_a_query(
        a_ds, query_vectors,
        column=args.column, metric=args.metric,
        top_k=args.top_k, nprobes=args.nprobes,
        warmup=args.warmup, refine_factor=args.refine_factor,
    )
    if ground_truth is not None:
        a_result[f"recall_at_{args.top_k}"] = compute_recall_at_k(
            a_result["result_ids"], ground_truth[args.warmup:], args.top_k,
        )
        print(f"  recall@{args.top_k}: {a_result[f'recall_at_{args.top_k}']:.4f}")
    print(f"  latency: mean={a_result['latency_ms']['mean']:.1f}ms, p99={a_result['latency_ms']['p99']:.1f}ms")
    del a_result["result_ids"]

    # ── Approach B: for each shard count ──
    b_results = []
    for num_shards in shard_counts:
        print(f"\n=== Approach B: {num_shards} shards ===")
        shard_dir = f"{work_dir}/approach_b_{num_shards}shards"

        if not args.skip_setup:
            shard_uris, setup_time = approach_b_setup(
                source_ds,
                work_dir=shard_dir,
                num_shards=num_shards,
                column=args.column,
                metric=args.metric,
                num_partitions=num_partitions,
                num_sub_vectors=args.num_sub_vectors,
                num_bits=args.num_bits,
                storage_opts=storage_opts if is_s3 else None,
                is_s3=is_s3,
            )
        else:
            shard_uris = [
                f"{shard_dir}/shard-{i}.lance" for i in range(num_shards)
            ]

        b_result = approach_b_query(
            shard_uris, query_vectors,
            storage_opts=storage_opts,
            column=args.column, metric=args.metric,
            top_k=args.top_k, nprobes=args.nprobes,
            warmup=args.warmup, refine_factor=args.refine_factor,
        )
        if ground_truth is not None:
            # Note: Approach B row_ids are shard-local, not comparable to source dataset
            # For fair comparison, we'd need to track global IDs. Mark as N/A for now.
            b_result["recall_note"] = "Row IDs are shard-local; direct recall comparison requires ID mapping"
        b_results.append(b_result)
        print(f"  latency: mean={b_result['latency_ms']['mean']:.1f}ms, p99={b_result['latency_ms']['p99']:.1f}ms")

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"{'Approach':<25} {'Mean(ms)':<12} {'P50(ms)':<12} {'P99(ms)':<12} {'Recall':<10}")
    print("-" * 70)
    recall_key = f"recall_at_{args.top_k}"
    print(
        f"{'A (native multi-seg)':<25} "
        f"{a_result['latency_ms']['mean']:<12.1f} "
        f"{a_result['latency_ms']['p50']:<12.1f} "
        f"{a_result['latency_ms']['p99']:<12.1f} "
        f"{a_result.get(recall_key, 'N/A')}"
    )
    for br in b_results:
        label = f"B ({br['num_shards']} shards, {br['workers']}w)"
        print(
            f"{label:<25} "
            f"{br['latency_ms']['mean']:<12.1f} "
            f"{br['latency_ms']['p50']:<12.1f} "
            f"{br['latency_ms']['p99']:<12.1f} "
            f"{'N/A':<10}"
        )
    print("=" * 70)

    # Save results
    result_doc = {
        "host": socket.gethostname(),
        "dataset": {
            "uri": args.source_uri,
            "rows": num_rows,
            "column": args.column,
            "metric": args.metric,
            "num_partitions": num_partitions,
        },
        "query": {
            "count": args.query_count,
            "top_k": args.top_k,
            "nprobes": args.nprobes,
            "refine_factor": args.refine_factor,
            "warmup": args.warmup,
        },
        "approach_a": a_result,
        "approach_b": b_results,
        "index_build_time_a_s": a_index_time,
    }

    if args.result_path:
        Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_path).write_text(json.dumps(result_doc, indent=2, sort_keys=True) + "\n")
        print(f"\nResults saved to {args.result_path}")


if __name__ == "__main__":
    main()
