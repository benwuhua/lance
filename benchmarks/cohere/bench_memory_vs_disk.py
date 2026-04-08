#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark: 全内存 vs 全磁盘 for distributed Plan B with IVF-RQ index.

Compares query latency when:
  - 全内存: page cache hot (all auxiliary.idx + index files cached in RAM)
  - 全磁盘: page cache cold (drop_caches before each query, every read hits NVMe)

Both use local NVMe storage (no OBS network I/O).

Usage (on ECS with root access):
    # Build 4 shards with RQ index first (one-time setup)
    python benchmarks/cohere/bench_memory_vs_disk.py \\
        --shard-dir /data/work/lance-shards-rq \\
        --source-uri s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance \\
        --setup

    # Run benchmark (needs sudo for drop_caches)
    sudo python benchmarks/cohere/bench_memory_vs_disk.py \\
        --shard-dir /data/work/lance-shards-rq \\
        --query-count 20 --warmup 3

If shards already exist with RQ index, skip --setup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


# ── Helpers ──────────────────────────────────────────────────────────


def summarize_latency(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    n = len(s)
    return {
        "mean": sum(s) / n,
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)] if n >= 20 else s[-1],
        "p99": s[int(n * 0.99)] if n >= 100 else s[-1],
        "max": s[-1],
        "min": s[0],
    }


def drop_page_cache():
    """Drop Linux page cache (requires root)."""
    subprocess.run(
        ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=True,
    )


def warm_page_cache(shard_dir: str, num_shards: int):
    """Read all files under shard dirs to populate page cache."""
    import mmap

    for i in range(num_shards):
        shard_path = Path(shard_dir) / f"shard-{i}.lance"
        if not shard_path.exists():
            continue
        for fpath in shard_path.rglob("*"):
            if fpath.is_file() and fpath.stat().st_size > 0:
                try:
                    with open(fpath, "rb") as f:
                        # Read in chunks to avoid huge allocations
                        while f.read(64 * 1024 * 1024):
                            pass
                except (OSError, PermissionError):
                    pass


# ── Shard Query Worker ───────────────────────────────────────────────


def _query_shard_worker(args_tuple):
    """Worker: query a single shard, return (distance, rowid) pairs."""
    (shard_uri, qvec, column, metric, k, nprobes, refine_factor) = args_tuple
    ds = lance.dataset(shard_uri)
    result = ds.to_table(
        columns=["_rowid", "_distance"],
        nearest={
            "column": column,
            "q": qvec.tolist(),
            "k": k,
            "use_index": True,
            "metric": metric,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
        },
    )
    rowids = result["_rowid"].to_numpy()
    dists = result["_distance"].to_numpy()
    return np.column_stack([dists, rowids])


def merge_top_k(
    partial_results: list[np.ndarray],
    top_k: int,
) -> np.ndarray:
    """Merge N shards' partial top-K into global top-K."""
    stacked = np.vstack(partial_results)
    sorted_idx = np.argsort(stacked[:, 0])
    return stacked[sorted_idx[:top_k], 1].astype(np.int64)


def query_distributed(
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
        (uri, qvec, column, metric, top_k, nprobes, refine_factor)
        for uri in shard_uris
    ]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_query_shard_worker, a) for a in args_list]
        partial_results = [f.result() for f in as_completed(futures)]

    merged = merge_top_k(partial_results, top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return merged, elapsed_ms


# ── Setup: Build RQ Shards ──────────────────────────────────────────


def setup_rq_shards(
    source_uri: str,
    shard_dir: str,
    *,
    num_shards: int,
    column: str,
    metric: str,
    num_partitions: int,
    storage_opts: dict | None,
):
    """Split source dataset into N shards, build IVF-RQ index on each."""
    import shutil

    print(f"[Setup] Building {num_shards} RQ shards from {source_uri}")
    print(f"[Setup] Output: {shard_dir}")

    source_ds = lance.dataset(source_uri, storage_options=storage_opts)
    total_rows = source_ds.count_rows()
    rows_per_shard = math.ceil(total_rows / num_shards)
    shard_partitions = max(num_partitions // num_shards, 4)

    print(f"  Total rows: {total_rows:,}")
    print(f"  Rows/shard: {rows_per_shard:,}")
    print(f"  Partitions/shard: {shard_partitions}")

    Path(shard_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_shards):
        shard_uri = str(Path(shard_dir) / f"shard-{i}.lance")
        start_row = i * rows_per_shard
        end_row = min(start_row + rows_per_shard, total_rows)
        if start_row >= total_rows:
            break

        if Path(shard_uri).exists():
            existing = lance.dataset(shard_uri)
            if existing.count_rows() == end_row - start_row and existing.list_indices():
                print(f"  Shard {i} already exists with index, skipping")
                continue
            shutil.rmtree(shard_uri)

        print(f"  Shard {i}: rows {start_row}-{end_row}...")
        shard_table = source_ds.to_table(
            columns=[column],
            offset=start_row,
            limit=end_row - start_row,
        )
        shard_ds = lance.write_dataset(shard_table, shard_uri, mode="overwrite")

        # Build IVF-RQ (RaBitQ / Binary Quantization) index
        print(f"  Shard {i}: building IVF_RQ index (partitions={shard_partitions})...")
        shard_ds.create_index(
            column=column,
            index_type="IVF_RQ",
            metric=metric,
            num_partitions=shard_partitions,
        )
        print(f"  Shard {i}: done ({shard_ds.count_rows()} rows, index built)")


# ── Main Benchmark ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark: 全内存 vs 全磁盘 with IVF-RQ")
    p.add_argument("--shard-dir", required=True, help="Directory containing shard-{0..N}.lance")
    p.add_argument("--source-uri", default=None, help="Source dataset URI (for --setup)")
    p.add_argument("--endpoint", default="https://obs.ap-southeast-1.myhuaweicloud.com")
    p.add_argument("--region", default="ap-southeast-1")
    p.add_argument("--access-key", default="")
    p.add_argument("--secret-key", default="")
    p.add_argument("--column", default="vector")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--num-partitions", type=int, default=4502)
    p.add_argument("--top-k", type=int, default=10000)
    p.add_argument("--nprobes", type=int, default=1024)
    p.add_argument("--refine-factor", type=int, default=1)
    p.add_argument("--query-count", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-path", default=None)
    p.add_argument("--setup", action="store_true", help="Build RQ shards from source")
    p.add_argument("--skip-memory", action="store_true", help="Skip 全内存 test")
    p.add_argument("--skip-disk", action="store_true", help="Skip 全磁盘 test")
    return p.parse_args()


def main():
    args = parse_args()

    storage_opts = None
    if args.source_uri and args.source_uri.startswith("s3://"):
        ak = args.access_key or os.environ.get("AWS_ACCESS_KEY_ID", "")
        sk = args.secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if not ak or not sk:
            print("WARNING: No OBS credentials, S3 source won't work")
        else:
            storage_opts = {
                "endpoint": args.endpoint,
                "access_key_id": ak,
                "secret_access_key": sk,
                "region": args.region,
                "virtual_hosted_style_request": "true",
                "use_opendal": "true",
                "allow_http": "true",
            }

    # ── Setup phase ──
    if args.setup:
        if not args.source_uri:
            print("ERROR: --source-uri required with --setup")
            sys.exit(1)
        setup_rq_shards(
            args.source_uri,
            args.shard_dir,
            num_shards=args.num_shards,
            column=args.column,
            metric=args.metric,
            num_partitions=args.num_partitions,
            storage_opts=storage_opts,
        )
        print("[Setup] Done.\n")

    # ── Validate shards exist ──
    shard_uris = [
        str(Path(args.shard_dir) / f"shard-{i}.lance")
        for i in range(args.num_shards)
    ]
    for uri in shard_uris:
        if not Path(uri).exists():
            print(f"ERROR: Shard not found: {uri}")
            sys.exit(1)

    # Print shard info
    for i, uri in enumerate(shard_uris):
        ds = lance.dataset(uri)
        indices = ds.list_indices()
        idx_types = [idx.type for idx in indices]
        print(f"  Shard {i}: {ds.count_rows():,} rows, indices={idx_types}")
        if not indices:
            print(f"  WARNING: Shard {i} has no index! Queries will be slow.")

    # ── Sample query vectors ──
    print(f"\nSampling {args.query_count} query vectors (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    ds0 = lance.dataset(shard_uris[0])
    total_rows = ds0.count_rows()
    sample_indices = rng.choice(total_rows, size=args.query_count, replace=False)
    query_table = ds0.take(sample_indices, columns=[args.column])
    query_vectors = np.stack(query_table[args.column].to_pylist())
    print(f"  Query shape: {query_vectors.shape}")

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # Test 1: 全内存 (Page Cache Hot)
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_memory:
        print("\n" + "=" * 60)
        print("TEST 1: 全内存 (Page Cache Hot)")
        print("=" * 60)

        print("Warming page cache (reading all shard files)...")
        t0 = time.perf_counter()
        warm_page_cache(args.shard_dir, args.num_shards)
        warm_elapsed = time.perf_counter() - t0
        print(f"  Page cache warmed in {warm_elapsed:.1f}s")

        # Warmup queries
        print(f"Running {args.warmup} warmup queries...")
        for qi in range(args.warmup):
            query_distributed(
                shard_uris, query_vectors[qi],
                column=args.column, metric=args.metric,
                top_k=args.top_k, nprobes=args.nprobes,
                refine_factor=args.refine_factor,
                max_workers=args.max_workers,
            )

        # Timed queries
        print(f"Running {args.query_count - args.warmup} timed queries...")
        latency_ms = []
        for qi in range(args.warmup, args.query_count):
            _, elapsed = query_distributed(
                shard_uris, query_vectors[qi],
                column=args.column, metric=args.metric,
                top_k=args.top_k, nprobes=args.nprobes,
                refine_factor=args.refine_factor,
                max_workers=args.max_workers,
            )
            latency_ms.append(elapsed)
            if (qi - args.warmup + 1) % 5 == 0:
                print(f"  Query {qi + 1}/{args.query_count}: {elapsed:.0f}ms")

        mem_summary = summarize_latency(latency_ms)
        results["memory"] = {
            "label": "全内存 (Page Cache Hot)",
            "latency_ms": mem_summary,
            "page_cache_warm_s": warm_elapsed,
            "query_count": len(latency_ms),
        }
        print(f"\n  全内存 Results:")
        print(f"    Mean: {mem_summary['mean']:.1f}ms")
        print(f"    P50:  {mem_summary['p50']:.1f}ms")
        print(f"    P95:  {mem_summary['p95']:.1f}ms")
        print(f"    P99:  {mem_summary['p99']:.1f}ms")
        print(f"    Max:  {mem_summary['max']:.1f}ms")

    # ═══════════════════════════════════════════════════════════════
    # Test 2: 全磁盘 (Page Cache Cold — drop before each query)
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_disk:
        print("\n" + "=" * 60)
        print("TEST 2: 全磁盘 (Page Cache Cold — drop_caches per query)")
        print("=" * 60)

        # Check we can drop caches
        try:
            drop_page_cache()
            print("  drop_caches OK")
        except (subprocess.CalledProcessError, PermissionError) as e:
            print(f"  ERROR: Cannot drop page cache (need root): {e}")
            print(f"  Run with: sudo python {sys.argv[0]} ...")
            sys.exit(1)

        # Warmup with drop_caches
        print(f"Running {args.warmup} warmup queries (each with drop_caches)...")
        for qi in range(args.warmup):
            drop_page_cache()
            query_distributed(
                shard_uris, query_vectors[qi],
                column=args.column, metric=args.metric,
                top_k=args.top_k, nprobes=args.nprobes,
                refine_factor=args.refine_factor,
                max_workers=args.max_workers,
            )

        # Timed queries with drop_caches
        print(f"Running {args.query_count - args.warmup} timed queries (each with drop_caches)...")
        latency_ms = []
        for qi in range(args.warmup, args.query_count):
            drop_page_cache()
            _, elapsed = query_distributed(
                shard_uris, query_vectors[qi],
                column=args.column, metric=args.metric,
                top_k=args.top_k, nprobes=args.nprobes,
                refine_factor=args.refine_factor,
                max_workers=args.max_workers,
            )
            latency_ms.append(elapsed)
            if (qi - args.warmup + 1) % 5 == 0:
                print(f"  Query {qi + 1}/{args.query_count}: {elapsed:.0f}ms")

        disk_summary = summarize_latency(latency_ms)
        results["disk"] = {
            "label": "全磁盘 (Page Cache Cold)",
            "latency_ms": disk_summary,
            "query_count": len(latency_ms),
        }
        print(f"\n  全磁盘 Results:")
        print(f"    Mean: {disk_summary['mean']:.1f}ms")
        print(f"    P50:  {disk_summary['p50']:.1f}ms")
        print(f"    P95:  {disk_summary['p95']:.1f}ms")
        print(f"    P99:  {disk_summary['p99']:.1f}ms")
        print(f"    Max:  {disk_summary['max']:.1f}ms")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
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
        print(f"\n  磁盘/内存 延迟比: {ratio:.2f}×")
        print(f"  磁盘额外开销: {disk_mean - mem_mean:.1f}ms")

    # Save results
    import socket

    result_doc = {
        "host": socket.gethostname(),
        "benchmark": "memory_vs_disk",
        "config": {
            "shard_dir": args.shard_dir,
            "num_shards": args.num_shards,
            "index_type": "IVF_RQ",
            "metric": args.metric,
            "top_k": args.top_k,
            "nprobes": args.nprobes,
            "refine_factor": args.refine_factor,
            "query_count": args.query_count,
            "warmup": args.warmup,
            "num_partitions": args.num_partitions,
        },
        "results": results,
    }

    if args.result_path:
        Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_path).write_text(json.dumps(result_doc, indent=2) + "\n")
        print(f"\nResults saved to {args.result_path}")
    else:
        # Default result path
        default_path = Path(args.shard_dir) / "bench_memory_vs_disk.json"
        default_path.write_text(json.dumps(result_doc, indent=2) + "\n")
        print(f"\nResults saved to {default_path}")


if __name__ == "__main__":
    main()
