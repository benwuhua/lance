#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Unified benchmark for distributed vector search.

Supports:
  - Plan A (single dataset) vs Plan B (sharded parallel)
  - IVF_PQ vs IVF_RQ index types
  - OBS (S3), SSD (local NVMe cold), DRAM (local NVMe hot) storage

Subcommands:
  setup   - Build shards with index from source dataset
  query   - Run benchmark with specified configuration
  compare - Run multiple configurations and output comparison

Usage:
  # Setup: build 4 shards with RQ index
  python benchmarks/cohere/bench.py setup \\
      --source-uri s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance \\
      --shard-dir /data/work/shards --index-type RQ --num-shards 4

  # Query Plan B, RQ, DRAM (page cache hot)
  python benchmarks/cohere/bench.py query \\
      --plan B --storage dram --index-type RQ \\
      --shard-dir /data/work/shards \\
      --nprobes 1024 --refine-factor 1 --top-k 10000

  # Query Plan B, RQ, SSD (needs root for drop_caches)
  sudo python benchmarks/cohere/bench.py query \\
      --plan B --storage ssd --index-type RQ \\
      --shard-dir /data/work/shards \\
      --nprobes 1024 --refine-factor 1 --top-k 10000

  # Query Plan A on OBS
  python benchmarks/cohere/bench.py query \\
      --plan A --storage obs --index-type PQ \\
      --dataset-uri s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance \\
      --nprobes 2048 --refine-factor 10 --top-k 10000

  # Compare across multiple configurations
  python benchmarks/cohere/bench.py compare \\
      --plan B --index-type RQ --shard-dir /data/work/shards \\
      --storage-list dram,ssd \\
      --nprobes-list 512,1024 --rf-list 1,2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import lance
import numpy as np


# ── Utilities ────────────────────────────────────────────────────────


def interpolate_percentile(sorted_values: list[float], p: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lo, hi = math.floor(rank), math.ceil(rank)
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
        "min": s[0],
    }


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def compute_recall_at_k(
    ann_ids: list[np.ndarray],
    gt_ids: list[np.ndarray],
    k: int,
) -> float:
    """Recall@K: fraction of ground truth top-K found in ANN top-K."""
    recalls = []
    for ann, gt in zip(ann_ids, gt_ids):
        gt_set = set(gt[:k].tolist())
        overlap = sum(1 for rid in ann[:k].tolist() if rid in gt_set)
        recalls.append(overlap / k)
    return sum(recalls) / len(recalls)


def compute_exact_distances(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Compute exact distances between a query vector and candidate vectors."""
    if metric == "cosine":
        dots = vectors @ query_vec
        q_norm = np.linalg.norm(query_vec)
        v_norms = np.linalg.norm(vectors, axis=1)
        cos_sim = dots / (q_norm * v_norms + 1e-10)
        return 1.0 - cos_sim
    elif metric == "L2":
        diff = vectors - query_vec
        return np.sum(diff * diff, axis=1)
    elif metric == "dot":
        return -(vectors @ query_vec)
    else:
        raise ValueError(f"Unsupported metric for exact rerank: {metric}")


def generate_ground_truth(
    dataset,
    query_vectors: np.ndarray,
    *,
    column: str,
    top_k: int,
    metric: str,
) -> list[np.ndarray]:
    """Flat brute-force ground truth (no index). Returns list of _rowid arrays."""
    print(f"Generating ground truth ({len(query_vectors)} queries, k={top_k})...")
    all_ids = []
    for i, qvec in enumerate(query_vectors):
        result = dataset.to_table(
            columns=["_rowid"],
            nearest={
                "column": column,
                "q": qvec.tolist() if isinstance(qvec, np.ndarray) else qvec,
                "k": top_k,
                "use_index": False,
                "metric": metric,
            },
        )
        all_ids.append(result["_rowid"].to_numpy())
        if (i + 1) % 5 == 0:
            print(f"  GT progress: {i + 1}/{len(query_vectors)}")
    return all_ids


def get_index_type(ds) -> str:
    """Get index type string. Compatible with lance 4.0.0 (dict) and older (object)."""
    indices = ds.list_indices()
    if not indices:
        return "none"
    idx = indices[0]
    if isinstance(idx, dict):
        return idx.get("type", "unknown")
    return getattr(idx, "type", "unknown")


def build_storage_options(args) -> dict[str, str] | None:
    """Build OBS storage options from args and env vars."""
    if not getattr(args, "endpoint", None):
        return None

    ak = getattr(args, "access_key", "") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    sk = getattr(args, "secret_key", "") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not ak or not sk:
        return None

    return {
        "endpoint": args.endpoint,
        "access_key_id": ak,
        "secret_access_key": sk,
        "region": getattr(args, "region", "ap-southeast-1"),
        "virtual_hosted_style_request": "true",
        "allow_http": "true",
        "use_opendal": "true",
    }


def make_nearest_kwargs(
    *,
    column: str,
    qvec,
    top_k: int,
    metric: str,
    nprobes: int,
    refine_factor: int,
) -> dict:
    """Build the `nearest` dict for dataset.to_table()."""
    return {
        "column": column,
        "q": qvec.tolist() if isinstance(qvec, np.ndarray) else qvec,
        "k": top_k,
        "use_index": True,
        "metric": metric,
        "nprobes": nprobes,
        "refine_factor": refine_factor,
    }


# ── Page Cache Management ───────────────────────────────────────────


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


def warm_page_cache_single(path: str):
    """Read all files under a single dataset directory."""
    for fpath in Path(path).rglob("*"):
        if fpath.is_file() and fpath.stat().st_size > 0:
            try:
                with open(fpath, "rb") as f:
                    while f.read(64 * 1024 * 1024):
                        pass
            except (OSError, PermissionError):
                pass


# ── Query: Plan A (single dataset) ──────────────────────────────────


def query_plan_a(
    dataset,
    query_vectors: np.ndarray,
    *,
    column: str,
    metric: str,
    top_k: int,
    nprobes: int,
    refine_factor: int,
    warmup: int,
    drop_between: bool = False,
) -> dict:
    """Plan A: single dataset, single call, Lance merges internally."""
    latency_ms = []

    for qi, qvec in enumerate(query_vectors):
        if drop_between and qi >= warmup:
            drop_page_cache()

        start = time.perf_counter()
        dataset.to_table(
            columns=["_distance"],
            nearest=make_nearest_kwargs(
                column=column, qvec=qvec, top_k=top_k,
                metric=metric, nprobes=nprobes, refine_factor=refine_factor,
            ),
        )
        elapsed = (time.perf_counter() - start) * 1000.0

        if qi >= warmup:
            latency_ms.append(elapsed)

    return {
        "plan": "A",
        "latency_ms": summarize_latency(latency_ms),
        "query_count": len(latency_ms),
    }


# ── Query: Plan B (sharded, serial) ─────────────────────────────────


def _rowid_to_position(dataset, rowids: np.ndarray) -> np.ndarray:
    """Convert _rowid values (fragment_id<<32|offset) to 0-based positional indices.

    _rowid is lance's internal row address: (fragment_id: u32 << 32) | (row_offset: u32).
    take() expects 0-based positional indices, so we need this conversion.
    """
    fragments = list(dataset.get_fragments())
    frag_base = {}
    cumulative = 0
    for frag in fragments:
        frag_base[frag.fragment_id] = cumulative
        cumulative += frag.count_rows()

    original_shape = rowids.shape
    rowids_u64 = rowids.astype(np.uint64).ravel()
    frag_ids = (rowids_u64 >> 32).astype(np.int32)
    offsets = (rowids_u64 & 0xFFFFFFFF).astype(np.int64)

    # Vectorized lookup: build array indexed by fragment_id
    if len(frag_base) > 0:
        max_frag_id = max(frag_base.keys())
        lookup = np.zeros(max_frag_id + 1, dtype=np.int64)
        for fid, base in frag_base.items():
            lookup[fid] = base
        bases = lookup[frag_ids]
    else:
        bases = np.zeros(len(frag_ids), dtype=np.int64)

    return (bases + offsets).reshape(original_shape)


def _query_single_shard(
    ds,
    qvec,
    *,
    column: str,
    metric: str,
    top_k: int,
    nprobes: int,
    refine_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query a single shard, return (rowids, distances)."""
    result = ds.to_table(
        columns=["_rowid", "_distance"],
        nearest=make_nearest_kwargs(
            column=column, qvec=qvec, top_k=top_k,
            metric=metric, nprobes=nprobes, refine_factor=refine_factor,
        ),
    )
    return result["_rowid"].to_numpy(), result["_distance"].to_numpy()


def query_plan_b(
    datasets: list,
    query_vectors: np.ndarray,
    *,
    column: str,
    metric: str,
    top_k: int,
    nprobes: int,
    refine_factor: int,
    warmup: int,
    drop_between: bool = False,
    max_workers: int | None = None,
    rerank_factor: int = 0,
) -> dict:
    """Plan B: parallel shard queries via ThreadPoolExecutor + merge top-K.

    Uses threads (not processes) because lance releases GIL during Rust IO.

    When rerank_factor > 0, after approximate merge, fetches original vectors
    for top-K*rerank_factor candidates and computes exact distances for re-ranking.
    This produces accurate distances and higher recall without increasing refine_factor.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    latency_ms = []
    ann_ids_per_query = []  # for recall computation
    num_shards = len(datasets)
    workers = max_workers or num_shards

    for qi, qvec in enumerate(query_vectors):
        if drop_between and qi >= warmup:
            drop_page_cache()

        start = time.perf_counter()

        # Query all shards in parallel via threads (no vectors, fast)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _query_single_shard, ds, qvec,
                    column=column, metric=metric, top_k=top_k,
                    nprobes=nprobes, refine_factor=refine_factor,
                ): i
                for i, ds in enumerate(datasets)
            }
            partial_results = [None] * num_shards
            for future in as_completed(futures):
                idx = futures[future]
                partial_results[idx] = future.result()

        # Merge: concat all (rowid, distance) pairs with shard origin
        all_rowids = np.concatenate([r[0] for r in partial_results])
        all_dists = np.concatenate([r[1] for r in partial_results])
        all_shard_ids = np.concatenate(
            [np.full(len(r[0]), i, dtype=np.int32) for i, r in enumerate(partial_results)]
        )

        if rerank_factor > 0:
            # Take more candidates for exact rerank
            n_candidates = min(top_k * rerank_factor, len(all_rowids))
            sorted_idx = np.argsort(all_dists)[:n_candidates]
            cand_rowids = all_rowids[sorted_idx]
            cand_shard_ids = all_shard_ids[sorted_idx]

            # Fetch original vectors per shard in parallel (only for rerank candidates)
            # Convert _rowid → positional index, then take()
            exact_dists = np.empty(n_candidates, dtype=np.float64)

            def _fetch_and_score(shard_id):
                mask = cand_shard_ids == shard_id
                if not np.any(mask):
                    return shard_id, mask, np.array([]), np.array([])
                rowids = cand_rowids[mask]
                positions = _rowid_to_position(datasets[shard_id], rowids)
                vec_table = datasets[shard_id].take(
                    positions.tolist(), columns=[column]
                )
                vecs = np.stack(vec_table[column].to_pylist())
                dists = compute_exact_distances(qvec, vecs, metric)
                return shard_id, mask, vecs, dists

            with ThreadPoolExecutor(max_workers=num_shards) as pool:
                for shard_id, mask, vecs, dists in pool.map(
                    _fetch_and_score, range(num_shards)
                ):
                    if len(dists) > 0:
                        exact_dists[mask] = dists

            # Final sort by exact distance
            final_idx = np.argsort(exact_dists)[:top_k]
            merged_rowids = cand_rowids[final_idx]
            merged_dists = exact_dists[final_idx]
        else:
            sorted_idx = np.argsort(all_dists)[:top_k]
            merged_rowids = all_rowids[sorted_idx]
            merged_dists = all_dists[sorted_idx]

        elapsed = (time.perf_counter() - start) * 1000.0

        if qi >= warmup:
            latency_ms.append(elapsed)
            ann_ids_per_query.append(merged_rowids)

    result = {
        "plan": "B",
        "num_shards": num_shards,
        "workers": workers,
        "latency_ms": summarize_latency(latency_ms),
        "query_count": len(latency_ms),
        "ann_ids": ann_ids_per_query,
    }
    if rerank_factor > 0:
        result["rerank_factor"] = rerank_factor
    return result


# ── Open datasets ────────────────────────────────────────────────────


def open_datasets(
    *,
    plan: str,
    shard_dir: str | None = None,
    dataset_uri: str | None = None,
    storage_opts: dict | None = None,
    num_shards: int = 4,
) -> list:
    """Open dataset(s) for querying."""
    if plan == "A":
        if not dataset_uri:
            raise ValueError("--dataset-uri required for Plan A")
        print(f"Opening dataset: {dataset_uri}")
        ds = lance.dataset(dataset_uri, storage_options=storage_opts)
        print(f"  Rows: {ds.count_rows():,}, index={get_index_type(ds)}")
        return [ds]
    else:
        # Plan B: open shards
        if not shard_dir:
            raise ValueError("--shard-dir required for Plan B")
        datasets = []
        for i in range(num_shards):
            uri = f"{shard_dir.rstrip('/')}/shard-{i}.lance"
            if not uri.startswith("s3://") and not Path(uri).exists():
                print(f"ERROR: Shard not found: {uri}")
                sys.exit(1)
            ds = lance.dataset(uri, storage_options=storage_opts)
            idx_type = get_index_type(ds)
            print(f"  Shard {i}: {ds.count_rows():,} rows, index={idx_type}")
            datasets.append(ds)
        return datasets


# ── Setup: build shards ─────────────────────────────────────────────


def cmd_setup(args):
    """Build N shards with index from source dataset."""
    import shutil

    source_uri = args.source_uri
    shard_dir = args.shard_dir
    num_shards = args.num_shards
    index_type = args.index_type
    column = args.column
    metric = args.metric
    num_partitions = args.num_partitions

    is_s3 = source_uri.startswith("s3://")
    storage_opts = build_storage_options(args) if is_s3 else None

    print(f"[Setup] Source: {source_uri}")
    print(f"[Setup] Output: {shard_dir}")
    print(f"[Setup] Shards: {num_shards}, Index: {index_type}")

    source_ds = lance.dataset(source_uri, storage_options=storage_opts)
    total_rows = source_ds.count_rows()
    rows_per_shard = math.ceil(total_rows / num_shards)
    shard_partitions = max(num_partitions // num_shards, 4)

    print(f"  Total rows: {total_rows:,}")
    print(f"  Rows/shard: {rows_per_shard:,}")
    print(f"  Partitions/shard: {shard_partitions}")

    if not is_s3:
        Path(shard_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_shards):
        shard_uri = str(Path(shard_dir) / f"shard-{i}.lance")
        start_row = i * rows_per_shard
        end_row = min(start_row + rows_per_shard, total_rows)
        if start_row >= total_rows:
            break

        # Skip if already exists with index
        if not is_s3 and Path(shard_uri).exists():
            existing = lance.dataset(shard_uri)
            if existing.count_rows() == end_row - start_row and existing.list_indices():
                print(f"  Shard {i} already exists with index, skipping")
                continue
            shutil.rmtree(shard_uri)

        print(f"  Shard {i}: rows {start_row}-{end_row}...")

        # Read data from source
        shard_table = source_ds.to_table(
            columns=[column],
            offset=start_row,
            limit=end_row - start_row,
        )

        # Write shard
        shard_ds = lance.write_dataset(shard_table, shard_uri, mode="overwrite")

        # Build index
        print(f"  Shard {i}: building {index_type} index (partitions={shard_partitions})...")
        create_kwargs = {
            "column": column,
            "index_type": index_type,
            "metric": metric,
            "num_partitions": shard_partitions,
        }
        if index_type == "IVF_PQ":
            create_kwargs["num_sub_vectors"] = args.num_sub_vectors
            create_kwargs["num_bits"] = args.num_bits

        shard_ds.create_index(**create_kwargs)
        idx_type = get_index_type(shard_ds)
        print(f"  Shard {i}: done ({shard_ds.count_rows()} rows, index={idx_type})")

    print(f"\n[Setup] Complete. {num_shards} shards in {shard_dir}")


# ── Query command ────────────────────────────────────────────────────


def cmd_query(args):
    """Run a single benchmark query configuration."""
    plan = args.plan
    storage = args.storage
    column = args.column
    metric = args.metric
    top_k = args.top_k
    nprobes = args.nprobes
    refine_factor = args.refine_factor
    query_count = args.query_count
    warmup = args.warmup
    seed = args.seed

    is_obs = storage == "obs"
    drop_between = storage == "ssd"
    storage_opts = build_storage_options(args) if is_obs else None

    # Resolve shard_dir and dataset_uri based on plan and storage
    shard_dir = getattr(args, "shard_dir", None)
    dataset_uri = getattr(args, "dataset_uri", None)

    # Prepare storage (warm cache for DRAM, initial drop for SSD)
    no_warm = getattr(args, "no_warm", False)
    if storage == "dram" and shard_dir and not no_warm:
        print(f"Warming page cache for {shard_dir}...")
        t0 = time.perf_counter()
        warm_page_cache(shard_dir, args.num_shards)
        print(f"  Page cache warmed in {time.perf_counter() - t0:.1f}s")
    elif storage == "dram" and dataset_uri and not dataset_uri.startswith("s3://") and not no_warm:
        print(f"Warming page cache for {dataset_uri}...")
        t0 = time.perf_counter()
        warm_page_cache_single(dataset_uri)
        print(f"  Page cache warmed in {time.perf_counter() - t0:.1f}s")
    elif storage == "ssd":
        try:
            drop_page_cache()
            print("  Initial drop_caches OK")
        except (subprocess.CalledProcessError, PermissionError) as e:
            print(f"  ERROR: Cannot drop page cache (need root): {e}")
            print(f"  Run with: sudo python {sys.argv[0]} ...")
            sys.exit(1)

    # Open datasets
    datasets = open_datasets(
        plan=plan,
        shard_dir=shard_dir,
        dataset_uri=dataset_uri,
        storage_opts=storage_opts,
        num_shards=args.num_shards,
    )

    # Sample query vectors from first dataset
    ds0 = datasets[0]
    print(f"\nSampling {query_count} query vectors (seed={seed})...")
    rng = np.random.default_rng(seed)
    total_rows = ds0.count_rows()
    sample_indices = rng.choice(total_rows, size=min(query_count, total_rows), replace=False)
    query_table = ds0.take(sample_indices, columns=[column])
    query_vectors = np.stack(query_table[column].to_pylist())
    print(f"  Query shape: {query_vectors.shape}")

    # Warmup queries (with per-query drop for SSD)
    print(f"Running {warmup} warmup queries...")
    for qi in range(warmup):
        if drop_between:
            drop_page_cache()
        qvec = query_vectors[qi]
        for ds in datasets:
            ds.to_table(
                columns=["_distance"],
                nearest=make_nearest_kwargs(
                    column=column, qvec=qvec, top_k=top_k,
                    metric=metric, nprobes=nprobes, refine_factor=refine_factor,
                ),
            )

    # Timed queries
    num_timed = query_count - warmup
    print(f"Running {num_timed} timed queries"
          f"{' (drop_caches before each)' if drop_between else ''}...")

    query_fn = query_plan_a if plan == "A" else query_plan_b
    rerank_factor = getattr(args, "rerank_factor", 0)
    query_args = dict(
        column=column, metric=metric,
        top_k=top_k, nprobes=nprobes,
        refine_factor=refine_factor, warmup=warmup,
        drop_between=drop_between,
        rerank_factor=rerank_factor if plan == "B" else 0,
    )

    if plan == "A":
        result = query_fn(datasets[0], query_vectors, **query_args)
    else:
        result = query_fn(datasets, query_vectors, **query_args)

    result["storage"] = storage
    result["index_type"] = getattr(args, "index_type", get_index_type(ds0))
    result["nprobes"] = nprobes
    result["refine_factor"] = refine_factor
    result["top_k"] = top_k

    # Compute recall if requested
    recall_k = getattr(args, "recall_k", None)
    gt_path = getattr(args, "gt_path", None)
    if (recall_k or gt_path) and plan == "B" and not is_obs:
        if gt_path:
            print(f"\nLoading ground truth from {gt_path}...")
            gt_data = np.load(gt_path)
            gt_ids = [gt_data[k] for k in sorted(gt_data.files)]
            recall_k = len(gt_ids[0])
            print(f"  GT: {len(gt_ids)} queries, k={recall_k} (positional indices)")
        else:
            print(f"\nComputing recall@{recall_k} against flat search on shard-0...")
            gt_ids = generate_ground_truth(
                datasets[0], query_vectors[warmup:],
                column=column, top_k=recall_k, metric=metric,
            )
        # Re-run ANN on shard-0 only to get per-shard results
        ann_ids_shard0 = []
        for qi in range(warmup, query_count):
            result_s0 = datasets[0].to_table(
                columns=["_rowid"],
                nearest=make_nearest_kwargs(
                    column=column, qvec=query_vectors[qi], top_k=recall_k,
                    metric=metric, nprobes=nprobes, refine_factor=refine_factor,
                ),
            )
            ann_rowids = result_s0["_rowid"].to_numpy()
            if gt_path:
                # GT uses positional indices; convert ANN _rowid to positional
                ann_ids_shard0.append(_rowid_to_position(datasets[0], ann_rowids))
            else:
                ann_ids_shard0.append(ann_rowids)

        recall = compute_recall_at_k(ann_ids_shard0, gt_ids, recall_k)
        result[f"recall_at_{recall_k}"] = recall
        print(f"  recall@{recall_k}: {recall:.4f} (shard-0 only)")

    # Print results
    lat = result["latency_ms"]
    label = f"{plan}-{result['index_type']}-{storage}-np{nprobes}-rf{refine_factor}"
    print(f"\n  Results ({label}):")
    print(f"    Mean: {lat['mean']:.1f}ms")
    print(f"    P50:  {lat['p50']:.1f}ms")
    print(f"    P95:  {lat['p95']:.1f}ms")
    print(f"    P99:  {lat['p99']:.1f}ms")
    print(f"    Max:  {lat['max']:.1f}ms")

    # Save results (strip non-serializable fields like numpy arrays)
    save_result = {k: v for k, v in result.items() if k != "ann_ids"}
    result_doc = {
        "host": socket.gethostname(),
        "benchmark": "distributed_vector_search",
        "config": {
            "plan": plan,
            "storage": storage,
            "index_type": result["index_type"],
            "num_shards": args.num_shards if plan == "B" else 1,
            "metric": metric,
            "top_k": top_k,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
            "query_count": num_timed,
            "warmup": warmup,
            "seed": seed,
        },
        "result": save_result,
    }

    result_path = args.result_path
    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(json.dumps(result_doc, indent=2) + "\n")
        print(f"\n  Results saved to {result_path}")

    return result


# ── Compare command ──────────────────────────────────────────────────


def cmd_compare(args):
    """Run multiple configurations and output comparison table."""
    storage_list = (
        [s.strip() for s in args.storage_list.split(",") if s.strip()]
        if args.storage_list
        else [args.storage]
    )
    nprobes_list = parse_int_list(args.nprobes_list) if args.nprobes_list else [args.nprobes]
    rf_list = parse_int_list(args.rf_list) if args.rf_list else [args.refine_factor]

    all_results = []

    for storage in storage_list:
        for nprobes in nprobes_list:
            for rf in rf_list:
                label = f"{args.plan}-{args.index_type}-{storage}-np{nprobes}-rf{rf}"
                print(f"\n{'=' * 60}")
                print(f"Running: {label}")
                print(f"{'=' * 60}")

                # Override args for this config
                args.storage = storage
                args.nprobes = nprobes
                args.refine_factor = rf

                try:
                    result = cmd_query(args)
                    result["label"] = label
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results.append({"label": label, "error": str(e)})

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Label':<45} {'Mean(ms)':>10} {'P50(ms)':>10} {'P99(ms)':>10} {'Max(ms)':>10}")
    print("-" * 90)

    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<45} {'ERROR':>10}")
            continue
        lat = r["latency_ms"]
        print(
            f"{r['label']:<45} "
            f"{lat['mean']:>10.1f} "
            f"{lat['p50']:>10.1f} "
            f"{lat['p99']:>10.1f} "
            f"{lat['max']:>10.1f}"
        )

    print("=" * 90)

    # Compute ratios if we have DRAM + SSD results
    dram_results = [r for r in all_results if "latency_ms" in r and r.get("storage") == "dram"]
    ssd_results = [r for r in all_results if "latency_ms" in r and r.get("storage") == "ssd"]
    if dram_results and ssd_results:
        dram_mean = np.mean([r["latency_ms"]["mean"] for r in dram_results])
        ssd_mean = np.mean([r["latency_ms"]["mean"] for r in ssd_results])
        ratio = ssd_mean / dram_mean if dram_mean > 0 else float("inf")
        print(f"\n  SSD/DRAM latency ratio: {ratio:.2f}x")
        print(f"  SSD overhead: {ssd_mean - dram_mean:.1f}ms")

    # Save comparison
    result_doc = {
        "host": socket.gethostname(),
        "benchmark": "distributed_vector_search_comparison",
        "results": all_results,
    }

    result_path = args.result_path
    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(json.dumps(result_doc, indent=2) + "\n")
        print(f"\nComparison saved to {result_path}")


# ── Argument Parsing ─────────────────────────────────────────────────


def add_common_args(p: argparse.ArgumentParser):
    """Arguments shared across all subcommands."""
    p.add_argument("--column", default="vector", help="Vector column name")
    p.add_argument("--metric", default="cosine", choices=["L2", "cosine", "dot"])
    p.add_argument("--index-type", default="IVF_RQ", choices=["IVF_PQ", "IVF_RQ"])

    # OBS credentials
    p.add_argument("--endpoint", default="https://obs.ap-southeast-1.myhuaweicloud.com")
    p.add_argument("--region", default="ap-southeast-1")
    p.add_argument("--access-key", default="")
    p.add_argument("--secret-key", default="")

    # PQ-specific
    p.add_argument("--num-sub-vectors", type=int, default=64)
    p.add_argument("--num-bits", type=int, default=8)


def add_query_args(p: argparse.ArgumentParser):
    """Arguments for query/compare subcommands."""
    p.add_argument("--plan", required=True, choices=["A", "B"],
                   help="Plan A (single dataset) or Plan B (sharded)")
    p.add_argument("--storage", required=True, choices=["obs", "ssd", "dram"],
                   help="obs=S3, ssd=NVMe cold, dram=NVMe hot")
    p.add_argument("--nprobes", type=int, default=1024)
    p.add_argument("--refine-factor", type=int, default=1)
    p.add_argument("--top-k", type=int, default=10000)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--query-count", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-path", default=None)
    p.add_argument("--recall-k", type=int, default=None,
                   help="Compute recall@K against flat search (default: same as top-k)")
    p.add_argument("--rerank-factor", type=int, default=0,
                   help="Exact rerank factor: fetch original vectors for top-K*N candidates "
                        "and compute exact distances for re-ranking (0=disabled)")
    p.add_argument("--no-warm", action="store_true",
                   help="Skip page cache warming for DRAM runs")
    p.add_argument("--gt-path", default=None,
                   help="Pre-computed GT .npz with positional indices (skips GT generation, "
                        "enables cross-dataset recall e.g. PCA vs original vectors)")

    # Data source (at least one required depending on plan)
    p.add_argument("--dataset-uri", default=None,
                   help="Dataset URI for Plan A (required for Plan A)")
    p.add_argument("--shard-dir", default=None,
                   help="Directory with shard-{0..N}.lance (required for Plan B)")

    add_common_args(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified benchmark for distributed vector search"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = sub.add_parser("setup", help="Build shards with index from source dataset")
    p_setup.add_argument("--source-uri", required=True,
                         help="Source dataset URI (local path or s3://)")
    p_setup.add_argument("--shard-dir", required=True,
                         help="Output directory for shards")
    p_setup.add_argument("--num-shards", type=int, default=4)
    p_setup.add_argument("--num-partitions", type=int, default=4502,
                         help="Total IVF partitions (divided across shards)")
    add_common_args(p_setup)

    # query
    p_query = sub.add_parser("query", help="Run single benchmark configuration")
    add_query_args(p_query)

    # compare
    p_compare = sub.add_parser("compare", help="Run multiple configurations and compare")
    add_query_args(p_compare)
    p_compare.add_argument("--storage-list", default=None,
                           help="Comma-separated storage backends (e.g. dram,ssd)")
    p_compare.add_argument("--nprobes-list", default=None,
                           help="Comma-separated nprobes values")
    p_compare.add_argument("--rf-list", default=None,
                           help="Comma-separated refine factor values")

    return parser


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
