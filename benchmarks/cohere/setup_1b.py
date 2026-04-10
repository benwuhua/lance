#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Build ~1B row vector dataset by duplicating existing source data.

Reads from a source lance dataset in batches, cycles through the data
to fill N shards with multiplied rows, then builds IVF_RQ index on each.

Usage:
  python benchmarks/cohere/setup_1b.py \
      --source-uri /data/work/tmp/fineweb-edu-1b-local.lance \
      --shard-dir /data/work/tmp/lance-shards-1b \
      --num-shards 5 --multiplier 3 --num-partitions 13107

  # Resume after interruption (skips completed shards automatically)
  python benchmarks/cohere/setup_1b.py \
      --source-uri /data/work/tmp/fineweb-edu-1b-local.lance \
      --shard-dir /data/work/tmp/lance-shards-1b

  # Only build index for existing shards (skip data write)
  python benchmarks/cohere/setup_1b.py \
      --shard-dir /data/work/tmp/lance-shards-1b \
      --skip-write
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ~1B row vector dataset from duplicated source data"
    )
    p.add_argument("--source-uri", default="/data/work/tmp/fineweb-edu-1b-local.lance",
                    help="Source lance dataset URI")
    p.add_argument("--shard-dir", default="/data/work/tmp/lance-shards-1b",
                    help="Output directory for shards")
    p.add_argument("--num-shards", type=int, default=5,
                    help="Number of shards to create")
    p.add_argument("--multiplier", type=int, default=3,
                    help="How many times to cycle through source data")
    p.add_argument("--num-partitions", type=int, default=13107,
                    help="IVF partitions per shard (nlist)")
    p.add_argument("--column", default="vector",
                    help="Vector column name")
    p.add_argument("--metric", default="cosine",
                    help="Distance metric")
    p.add_argument("--batch-size", type=int, default=1_000_000,
                    help="Rows per write batch")
    p.add_argument("--skip-write", action="store_true",
                    help="Skip data writing, only build indexes for existing shards")
    p.add_argument("--skip-index", action="store_true",
                    help="Skip index building")
    p.add_argument("--noise-std", type=float, default=0.01,
                    help="Gaussian noise std to add to duplicate vectors (0=exact copy)")
    p.add_argument("--shard-id", type=int, default=None,
                    help="Only build this specific shard (0-indexed). Default: build all.")
    return p.parse_args()


def shard_is_complete(shard_uri: str, expected_rows: int) -> bool:
    """Check if a shard has the expected rows and an index."""
    if not Path(shard_uri).exists():
        return False
    try:
        ds = lance.dataset(shard_uri)
        if ds.count_rows() != expected_rows:
            return False
        return len(ds.list_indices()) > 0
    except Exception:
        return False


def write_shard(
    *,
    source_uri: str,
    shard_uri: str,
    total_source: int,
    rows_per_shard: int,
    column: str,
    batch_size: int,
    noise_std: float = 0.01,
) -> None:
    """Write one shard by cycling through source data with optional noise."""
    source = lance.dataset(source_uri)
    rng = np.random.default_rng(42)
    offset = 0
    written = 0
    cycle = 0  # track how many full cycles through source

    while written < rows_per_shard:
        remaining = rows_per_shard - written
        available = total_source - offset
        read_size = min(batch_size, remaining, available)

        if read_size <= 0:
            offset = 0
            cycle += 1
            continue

        t0 = time.perf_counter()
        table = source.take(range(offset, offset + read_size), columns=[column])
        take_time = time.perf_counter() - t0

        # Add noise to duplicate copies (cycle > 0) to make vectors unique
        if cycle > 0 and noise_std > 0:
            vecs = np.stack(table[column].to_pylist())
            noise = rng.normal(0, noise_std, vecs.shape).astype(np.float32)
            vecs = vecs + noise
            # Re-normalize for cosine similarity
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms
            # Replace column in table
            table = table.set_column(
                table.schema.get_field_index(column),
                column,
                pa.FixedSizeListArray.from_arrays(vecs.flatten(), list_size=vecs.shape[1]),
            )

        mode = "create" if written == 0 else "append"
        t1 = time.perf_counter()
        lance.write_dataset(table, shard_uri, mode=mode)
        write_time = time.perf_counter() - t1

        written += read_size
        offset = (offset + read_size) % total_source

        pct = written / rows_per_shard * 100
        print(
            f"    {written:>12,} / {rows_per_shard:,} rows ({pct:5.1f}%) "
            f"take={take_time:.1f}s write={write_time:.1f}s",
            flush=True,
        )


def build_index(
    *,
    shard_uri: str,
    column: str,
    metric: str,
    num_partitions: int,
    index_type: str = "IVF_RQ",
) -> None:
    """Build vector index on a shard."""
    ds = lance.dataset(shard_uri)
    print(f"  Building {index_type} index (partitions={num_partitions})...", flush=True)
    t0 = time.perf_counter()
    ds.create_index(
        column=column,
        index_type=index_type,
        metric=metric,
        num_partitions=num_partitions,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Index built in {elapsed:.0f}s ({elapsed / 60:.1f}min)", flush=True)


def main() -> None:
    args = parse_args()

    total_source = 0
    if not args.skip_write:
        source = lance.dataset(args.source_uri)
        total_source = source.count_rows()
        rows_per_shard = (total_source * args.multiplier) // args.num_shards
        print(f"Source: {args.source_uri}")
        print(f"  Source rows: {total_source:,}")
        print(f"  Target: {args.num_shards} shards × {rows_per_shard:,} rows = "
              f"{rows_per_shard * args.num_shards:,} total")
        print(f"  Partitions/shard: {args.num_partitions:,}")
        print(f"  Density: {rows_per_shard / args.num_partitions:,.0f} rows/partition")
        print()
    else:
        # Infer rows_per_shard from existing shards
        rows_per_shard = 0
        for i in range(args.num_shards):
            uri = str(Path(args.shard_dir) / f"shard-{i}.lance")
            if Path(uri).exists():
                ds = lance.dataset(uri)
                rows_per_shard = ds.count_rows()
                break
        print(f"Skip-write mode. Rows/shard: {rows_per_shard:,}")
        print()

    os.makedirs(args.shard_dir, exist_ok=True)

    shard_ids = [args.shard_id] if args.shard_id is not None else list(range(args.num_shards))

    for shard_id in shard_ids:
        shard_uri = str(Path(args.shard_dir) / f"shard-{shard_id}.lance")
        print(f"{'=' * 60}")
        print(f"  Shard {shard_id}: {shard_uri}")
        print(f"{'=' * 60}")

        # Check if already complete
        if shard_is_complete(shard_uri, rows_per_shard):
            print(f"  Already complete, skipping")
            continue

        # Write data
        if not args.skip_write:
            # Remove incomplete shard
            if Path(shard_uri).exists():
                import shutil
                print(f"  Removing incomplete shard...")
                shutil.rmtree(shard_uri)

            write_shard(
                source_uri=args.source_uri,
                shard_uri=shard_uri,
                total_source=total_source,
                rows_per_shard=rows_per_shard,
                column=args.column,
                batch_size=args.batch_size,
            )

        # Build index
        if not args.skip_index:
            if not Path(shard_uri).exists():
                print(f"  ERROR: shard does not exist, cannot build index")
                continue
            ds = lance.dataset(shard_uri)
            if len(ds.list_indices()) > 0:
                print(f"  Index already exists, skipping")
            else:
                build_index(
                    shard_uri=shard_uri,
                    column=args.column,
                    metric=args.metric,
                    num_partitions=args.num_partitions,
                )

        # Verify
        final_ds = lance.dataset(shard_uri)
        final_rows = final_ds.count_rows()
        has_index = len(final_ds.list_indices()) > 0
        print(f"  Done: {final_rows:,} rows, index={'yes' if has_index else 'no'}")
        print()

    # Summary
    print(f"{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    total = 0
    for i in range(args.num_shards):
        uri = str(Path(args.shard_dir) / f"shard-{i}.lance")
        if Path(uri).exists():
            ds = lance.dataset(uri)
            rows = ds.count_rows()
            idx = "IVF_RQ" if ds.list_indices() else "none"
            print(f"  shard-{i}: {rows:,} rows, index={idx}")
            total += rows
        else:
            print(f"  shard-{i}: MISSING")
    print(f"  Total: {total:,} rows")


if __name__ == "__main__":
    main()
