#!/usr/bin/env python3
"""Convert .npy embedding files to .fbin/.ibin format for HNSW benchmarking.

Reads float16 vectors from .npy files, concatenates them, and writes:
- base.fbin:  base vectors in big-ann-benchmarks format
- query.fbin: query vectors in big-ann-benchmarks format
- gt.ibin:    ground truth top-K neighbors (L2 distance)

File formats (all little-endian):
  .fbin: [n:u32][dim:u32][f32 data row-major, n * dim floats]
  .ibin: [n:u32][k:u32][i32 data row-major, n * k ints]
"""

import argparse
import glob
import os
import struct
import time

import numpy as np


def load_npy_files(data_dir: str) -> np.ndarray:
    """Load and concatenate all .npy files from data_dir, convert to float32."""
    pattern = os.path.join(data_dir, "*.npy")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    print(f"Found {len(paths)} .npy files")
    arrays = []
    for i, path in enumerate(paths):
        arr = np.load(path)
        print(f"  [{i+1}/{len(paths)}] {os.path.basename(path)}: shape={arr.shape}, dtype={arr.dtype}")
        arrays.append(arr)

    concatenated = np.concatenate(arrays, axis=0).astype(np.float32)
    print(f"Total vectors: {concatenated.shape[0]}, dim: {concatenated.shape[1]}")
    return concatenated


def write_fbin(path: str, data: np.ndarray) -> None:
    """Write array to .fbin format: [n:u32][dim:u32][f32 data]."""
    n, dim = data.shape
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, dim))
        f.write(data.tobytes())
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Wrote {path}: {n} vectors, dim={dim}, {size_mb:.1f} MB")


def write_ibin(path: str, data: np.ndarray) -> None:
    """Write array to .ibin format: [n:u32][k:u32][i32 data]."""
    n, k = data.shape
    assert data.dtype == np.int32, f"Expected int32, got {data.dtype}"
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, k))
        f.write(data.tobytes())
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Wrote {path}: {n} queries, k={k}, {size_mb:.1f} MB")


def compute_ground_truth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    batch_size: int = 10,
) -> np.ndarray:
    """Compute brute-force top-K nearest neighbors using L2 distance.

    For each query, computes ||q - b||^2 = ||q||^2 - 2*q.b + ||b||^2
    and returns the indices of the k smallest distances.

    Args:
        base:    (N, D) float32 base vectors.
        queries: (Q, D) float32 query vectors.
        k:       Number of nearest neighbors to find.
        batch_size: Number of queries to process at once.

    Returns:
        (Q, K) int32 array of neighbor indices.
    """
    n = base.shape[0]
    dim = base.shape[1]
    num_queries = queries.shape[0]
    all_indices = np.empty((num_queries, k), dtype=np.int32)

    # Precompute base norms: ||b||^2 for each base vector.
    base_sq = np.sum(base ** 2, axis=1)  # (N,)

    print(f"Computing ground truth: {num_queries} queries x {n} base vectors, k={k}")
    t0 = time.time()

    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        batch = queries[start:end]  # (B, D)

        # L2^2 = ||b||^2 - 2*q.b + ||q||^2
        # We only need argsort, so ||q||^2 is constant per query and can be omitted.
        query_sq = np.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
        dots = batch @ base.T  # (B, N)
        dists = base_sq[np.newaxis, :] - 2 * dots + query_sq  # (B, N)

        # argpartition is O(N) per row vs O(N log N) for full argsort.
        # Then sort only the top-K subset.
        topk_idx = np.argpartition(dists, k, axis=1)[:, :k]  # (B, K)

        # Sort the top-K by actual distance for correct ordering.
        rows = np.arange(end - start)[:, None]
        topk_dists = dists[rows, topk_idx]
        sorted_order = np.argsort(topk_dists, axis=1)
        topk_idx = np.take_along_axis(topk_idx, sorted_order, axis=1)

        all_indices[start:end] = topk_idx

        done = end
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else float("inf")
        eta = (num_queries - done) / rate if rate > 0 else 0
        print(
            f"  {done}/{num_queries} queries done "
            f"({done / num_queries * 100:.1f}%), "
            f"elapsed {elapsed:.1f}s, ETA {eta:.0f}s"
        )

    total_time = time.time() - t0
    print(f"Ground truth computed in {total_time:.1f}s")
    return all_indices


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npy embeddings to .fbin/.ibin for HNSW benchmarking"
    )
    parser.add_argument(
        "--data-dir",
        default="/data/work/tmp/raw-cache-1m/emb/",
        help="Directory containing .npy embedding files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/work/tmp/cohere1m-fbin/",
        help="Output directory for .fbin/.ibin files (default: %(default)s)",
    )
    parser.add_argument(
        "--num-base",
        type=int,
        default=1_000_000,
        help="Number of base vectors to use (default: %(default)s)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of query vectors (taken from end of base) (default: %(default)s)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of ground truth nearest neighbors (default: %(default)s)",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=10,
        help="Batch size for ground truth computation (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Cohere 1M .fbin/.ibin preparation")
    print("=" * 60)
    print(f"Data dir:      {args.data_dir}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Num base:      {args.num_base:,}")
    print(f"Num queries:   {args.num_queries}")
    print(f"Top-K:         {args.k}")
    print()

    # Load all .npy files.
    t0 = time.time()
    all_vectors = load_npy_files(args.data_dir)
    total_loaded = all_vectors.shape[0]
    dim = all_vectors.shape[1]
    print(f"Loaded {total_loaded:,} vectors in {time.time() - t0:.1f}s")
    print()

    # Take first num_base vectors.
    if total_loaded < args.num_base:
        raise ValueError(
            f"Requested {args.num_base:,} base vectors but only "
            f"{total_loaded:,} loaded"
        )
    base = all_vectors[: args.num_base].copy()

    # Take last num_queries vectors from base as queries.
    # They stay in base — for benchmarking this is fine.
    if args.num_queries > args.num_base:
        raise ValueError(
            f"Requested {args.num_queries} queries but only "
            f"{args.num_base:,} base vectors"
        )
    queries = base[-args.num_queries :].copy()

    print(f"Base:    {base.shape[0]:,} vectors x {dim}D")
    print(f"Queries: {queries.shape[0]} vectors x {dim}D (last {args.num_queries} of base)")
    print()

    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=True)

    # Write base.fbin.
    write_fbin(os.path.join(args.output_dir, "base.fbin"), base)

    # Write query.fbin.
    write_fbin(os.path.join(args.output_dir, "query.fbin"), queries)

    # Compute and write ground truth.
    gt = compute_ground_truth(base, queries, args.k, batch_size=args.query_batch_size)
    write_ibin(os.path.join(args.output_dir, "gt.ibin"), gt)

    # Summary.
    print()
    print("=" * 60)
    print("Done!")
    print(f"Output: {args.output_dir}")
    print(f"  base.fbin:  {base.shape[0]:,} vectors")
    print(f"  query.fbin: {queries.shape[0]} vectors")
    print(f"  gt.ibin:    {queries.shape[0]} queries x top-{args.k}")
    print("=" * 60)


if __name__ == "__main__":
    main()
