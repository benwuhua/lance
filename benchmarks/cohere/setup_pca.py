#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Setup PCA-reduced shards for IVF_RQ benchmarking.

Reduces vector dimensionality via TruncatedSVD (no centering — preserves
cosine structure better than PCA which subtracts the mean).

Pipeline:
  1. Train TruncatedSVD on sample from shard-0
  2. Transform all shards: v' = SVD(v), then L2-normalize
  3. Build IVF_RQ index on reduced-dim vectors
  4. Convert existing GT (_rowid) to positional indices for cross-dataset recall

Usage:
  # Build PCA-512 shards from 1B dataset
  python benchmarks/cohere/setup_pca.py \
      --source-dir /data/work/tmp/s1b \
      --output-dir /data/work/tmp/s1b-pca512 \
      --pca-dim 512 --num-shards 5 --num-partitions 13107

  # Process only shard 2 (parallel mode)
  python benchmarks/cohere/setup_pca.py \
      --source-dir /data/work/tmp/s1b \
      --output-dir /data/work/tmp/s1b-pca512 \
      --pca-dim 512 --only-shard 2 --num-partitions 13107

  # Convert existing GT to positional indices
  python benchmarks/cohere/setup_pca.py \
      --convert-gt /data/work/tmp/gt_1b_shard0_top10k_5q_seed42.npz \
      --source-dir /data/work/tmp/s1b \
      --output-dir /data/work/tmp/s1b-pca512
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


# ── Utilities ────────────────────────────────────────────────────────


def _rowid_to_position(dataset, rowids: np.ndarray) -> np.ndarray:
    """Convert _rowid values to 0-based positional indices.

    Lance _rowid = (fragment_id: u32 << 32) | (row_offset: u32).
    We convert to 0-based positional index by adding fragment base offset.
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


def get_index_type(ds) -> str:
    indices = ds.list_indices()
    if not indices:
        return "none"
    idx = indices[0]
    if isinstance(idx, dict):
        return idx.get("type", "unknown")
    return getattr(idx, "type", "unknown")


def vectors_to_numpy(table, column: str) -> np.ndarray:
    """Extract vector column from lance table as numpy array."""
    col = table.column(column)
    if hasattr(col, "combine_chunks"):
        arr = col.combine_chunks()
    else:
        arr = col
    flat = arr.values.to_numpy()
    list_size = arr.type.list_size
    return flat.reshape(-1, list_size)


# ── SVD Training ────────────────────────────────────────────────────


def train_svd(
    source_dir: str,
    shard_id: int,
    sample_size: int,
    pca_dim: int,
    column: str,
) -> object:
    """Train TruncatedSVD on a sample from specified shard."""
    from sklearn.decomposition import TruncatedSVD

    shard_path = f"{source_dir}/shard-{shard_id}.lance"
    ds = lance.dataset(shard_path)
    total = ds.count_rows()
    sample_n = min(sample_size, total)

    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=sample_n, replace=False)
    table = ds.take(indices.tolist(), columns=[column])
    vectors = vectors_to_numpy(table, column).astype(np.float32)

    print(f"Training TruncatedSVD(n_components={pca_dim}) on {sample_n:,} vectors (dim={vectors.shape[1]})...")
    t0 = time.time()
    svd = TruncatedSVD(n_components=pca_dim, random_state=42)
    svd.fit(vectors)
    elapsed = time.time() - t0

    cumvar = np.cumsum(svd.explained_variance_ratio_)
    print(f"  Trained in {elapsed:.1f}s")
    print(f"  Explained variance: {cumvar[-1]:.4f} ({pca_dim} components)")
    for threshold in [0.90, 0.95, 0.99]:
        n_needed = int(np.searchsorted(cumvar, threshold) + 1)
        print(f"  {threshold:.0%} variance at {n_needed} components")

    return svd


# ── Shard Transform ─────────────────────────────────────────────────


def transform_shard(
    source_path: str,
    output_path: str,
    components: np.ndarray,
    column: str,
    batch_size: int,
) -> None:
    """Transform vectors in a shard using SVD components, write to new location.

    components: (pca_dim, orig_dim) from TruncatedSVD
    Output vectors are L2-normalized for cosine metric.
    """
    source = lance.dataset(source_path)
    total = source.count_rows()
    pca_dim = components.shape[0]

    written = 0
    mode = "create"

    while written < total:
        read_size = min(batch_size, total - written)
        table = source.to_table(columns=[column], offset=written, limit=read_size)
        vectors = vectors_to_numpy(table, column).astype(np.float32)

        # Transform: v' = v @ components.T
        transformed = vectors @ components.T

        # L2 normalize (for cosine metric)
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        transformed = transformed / np.maximum(norms, 1e-10)
        transformed = transformed.astype(np.float32)

        # Build pyarrow table with FixedSizeListArray
        flat = pa.array(transformed.ravel(), type=pa.float32())
        vec_array = pa.FixedSizeListArray.from_arrays(flat, list_size=pca_dim)
        new_table = pa.table({column: vec_array})

        lance.write_dataset(new_table, output_path, mode=mode)
        written += read_size
        mode = "append"
        print(f"    {written:,}/{total:,} rows written")


# ── GT Conversion ───────────────────────────────────────────────────


def convert_gt_to_positional(
    gt_path: str,
    original_shard_path: str,
    output_path: str,
) -> None:
    """Convert _rowid GT to positional indices for cross-dataset recall."""
    print(f"\nConverting GT to positional indices...")
    ds = lance.dataset(original_shard_path)
    gt_data = np.load(gt_path)
    gt_pos = []
    for k in sorted(gt_data.files):
        arr = gt_data[k]
        positions = _rowid_to_position(ds, arr)
        # Handle 2D arrays: split into per-query rows
        if positions.ndim == 2:
            gt_pos.extend(positions[i] for i in range(positions.shape[0]))
        else:
            gt_pos.append(positions)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, *gt_pos)
    n_queries = len(gt_pos)
    top_k = len(gt_pos[0]) if gt_pos else 0
    print(f"  Positional GT saved to {output_path} ({n_queries} queries, k={top_k})")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Setup PCA-reduced shards for benchmarking")
    parser.add_argument("--source-dir", required=True, help="Source shard directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--pca-dim", type=int, default=512, help="Target dimension")
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--num-partitions", type=int, default=13107, help="IVF partitions per shard")
    parser.add_argument("--column", default="vector")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--sample-size", type=int, default=500_000, help="Vectors for SVD training")
    parser.add_argument("--batch-size", type=int, default=2_000_000, help="Rows per write batch")
    parser.add_argument("--train-shard", type=int, default=0, help="Shard to sample for SVD training")
    parser.add_argument("--index-type", default="IVF_RQ", choices=["IVF_RQ", "IVF_SQ"],
                        help="Index type to build (default: IVF_RQ)")
    parser.add_argument("--only-shard", type=int, default=None,
                        help="Process only this single shard ID (skips SVD training)")
    parser.add_argument("--convert-gt", default=None,
                        help="Existing GT .npz to convert to positional indices")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # GT conversion mode
    if args.convert_gt:
        original_shard = f"{args.source_dir}/shard-0.lance"
        gt_output = f"{args.output_dir}/gt_positional.npz"
        convert_gt_to_positional(args.convert_gt, original_shard, gt_output)
        return

    # Step 1: Load or train SVD
    pca_model_path = f"{args.output_dir}/svd_model.pkl"

    if args.only_shard is not None:
        # Single shard mode: load existing SVD model
        if not Path(pca_model_path).exists():
            print(f"ERROR: SVD model not found at {pca_model_path}")
            print(f"Train first without --only-shard")
            sys.exit(1)
        with open(pca_model_path, "rb") as f:
            svd = pickle.load(f)
        components = svd.components_.astype(np.float32)

        # Process single shard
        i = args.only_shard
        source_path = f"{args.source_dir}/shard-{i}.lance"
        output_path = f"{args.output_dir}/shard-{i}.lance"
        source_rows = lance.dataset(source_path).count_rows()

        if Path(output_path).exists():
            existing = lance.dataset(output_path)
            if existing.count_rows() == source_rows and existing.list_indices():
                print(f"Shard {i}: already complete, skipping")
                return

        print(f"\nShard {i}: transforming {source_rows:,} rows...")
        t0 = time.time()
        transform_shard(source_path, output_path, components, args.column, args.batch_size)
        print(f"  Transform done in {time.time() - t0:.0f}s")

        print(f"Shard {i}: building {args.index_type} index ({args.num_partitions:,} partitions)...")
        shard_ds = lance.dataset(output_path)
        t0 = time.time()
        shard_ds.create_index(
            column=args.column,
            index_type=args.index_type,
            metric=args.metric,
            num_partitions=args.num_partitions,
        )
        elapsed = time.time() - t0
        idx_type = get_index_type(shard_ds)
        print(f"  Index built in {elapsed:.0f}s ({shard_ds.count_rows():,} rows, index={idx_type})")
        print(f"\nAll done!")
        return

    # Full mode: train SVD + transform all shards
    if Path(pca_model_path).exists():
        print(f"\nLoading existing SVD model from {pca_model_path}...")
        with open(pca_model_path, "rb") as f:
            svd = pickle.load(f)
    else:
        svd = train_svd(args.source_dir, args.train_shard, args.sample_size, args.pca_dim, args.column)
        with open(pca_model_path, "wb") as f:
            pickle.dump(svd, f)
        print(f"\nSVD model saved to {pca_model_path}")

    components = svd.components_.astype(np.float32)

    # Step 2: Transform and index each shard
    for i in range(args.num_shards):
        source_path = f"{args.source_dir}/shard-{i}.lance"
        output_path = f"{args.output_dir}/shard-{i}.lance"

        source_rows = lance.dataset(source_path).count_rows()

        # Skip if already complete
        if Path(output_path).exists():
            existing = lance.dataset(output_path)
            if existing.count_rows() == source_rows and existing.list_indices():
                print(f"\nShard {i}: already complete, skipping")
                continue

        print(f"\nShard {i}: transforming {source_rows:,} rows...")
        t0 = time.time()
        transform_shard(source_path, output_path, components, args.column, args.batch_size)
        print(f"  Transform done in {time.time() - t0:.0f}s")

        # Build index
        print(f"Shard {i}: building {args.index_type} index ({args.num_partitions:,} partitions)...")
        shard_ds = lance.dataset(output_path)
        t0 = time.time()
        shard_ds.create_index(
            column=args.column,
            index_type=args.index_type,
            metric=args.metric,
            num_partitions=args.num_partitions,
        )
        elapsed = time.time() - t0
        idx_type = get_index_type(shard_ds)
        print(f"  Index built in {elapsed:.0f}s ({shard_ds.count_rows():,} rows, index={idx_type})")

    # Step 3: Convert existing GT if available
    default_gt = "/data/work/tmp/gt_1b_shard0_top10k_5q_seed42.npz"
    if Path(default_gt).exists():
        gt_output = f"{args.output_dir}/gt_positional.npz"
        if not Path(gt_output).exists():
            convert_gt_to_positional(default_gt, f"{args.source_dir}/shard-0.lance", gt_output)

    print(f"\nAll done! PCA shards in {args.output_dir}")


if __name__ == "__main__":
    main()
