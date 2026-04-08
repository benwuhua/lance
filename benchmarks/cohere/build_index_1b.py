#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Build IVF-PQ index on the FineWeb-Edu ~1B vector dataset stored on OBS."""

from __future__ import annotations

import argparse
import math
import time

import lance


def validate_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got value={value}")
    return value


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


def compute_default_num_partitions(num_rows: int) -> int:
    return min(num_rows // 4000, round(math.sqrt(num_rows)))


def check_existing_index(dataset) -> bool:
    indices = dataset.list_indices()
    return any(idx.type == "IVF_PQ" for idx in indices)


def build_index_simple(
    dataset,
    *,
    column: str,
    metric: str,
    num_partitions: int,
    num_sub_vectors: int,
    num_bits: int,
    replace: bool,
) -> float:
    print(
        f"Building IVF_PQ index: "
        f"num_partitions={num_partitions}, "
        f"num_sub_vectors={num_sub_vectors}, "
        f"num_bits={num_bits}, "
        f"metric={metric}"
    )
    start = time.perf_counter()
    dataset.create_index(
        column=column,
        index_type="IVF_PQ",
        metric=metric,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        num_bits=num_bits,
        replace=replace,
    )
    elapsed = time.perf_counter() - start
    return elapsed


def build_index_distributed(
    dataset,
    *,
    column: str,
    metric: str,
    num_partitions: int,
    num_sub_vectors: int,
    num_bits: int,
    num_workers: int,
    replace: bool,
) -> float:
    from lance.indices import IndicesBuilder

    print(
        f"Building IVF_PQ index (distributed, {num_workers} workers): "
        f"num_partitions={num_partitions}, "
        f"num_sub_vectors={num_sub_vectors}, "
        f"num_bits={num_bits}, "
        f"metric={metric}"
    )

    total_start = time.perf_counter()

    # Step 1: Global training (centroids + codebook)
    print("Step 1/3: Training IVF centroids and PQ codebook...")
    builder = IndicesBuilder(dataset, column)
    train_start = time.perf_counter()
    pre = builder.prepare_global_ivf_pq(
        num_partitions=num_partitions,
        num_subvectors=num_sub_vectors,
        distance_type=metric,
        sample_rate=256,
        max_iters=50,
    )
    train_elapsed = time.perf_counter() - train_start
    print(f"  Training done in {train_elapsed:.1f}s")

    # Step 2: Build index segments per fragment group
    print("Step 2/3: Building index segments...")
    frags = dataset.get_fragments()
    frag_ids = [f.fragment_id for f in frags]
    chunk_size = math.ceil(len(frag_ids) / num_workers)
    groups = [
        frag_ids[i : i + chunk_size] for i in range(0, len(frag_ids), chunk_size)
    ]
    print(f"  {len(frags)} fragments split into {len(groups)} groups")

    segments = []
    for group_idx, group in enumerate(groups):
        seg_start = time.perf_counter()
        print(f"  Group {group_idx + 1}/{len(groups)}: {len(group)} fragments...")
        seg = dataset.create_index_uncommitted(
            column=column,
            index_type="IVF_PQ",
            fragment_ids=group,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            num_bits=num_bits,
            ivf_centroids=pre["ivf_centroids"],
            pq_codebook=pre["pq_codebook"],
        )
        segments.append(seg)
        seg_elapsed = time.perf_counter() - seg_start
        print(f"    Done in {seg_elapsed:.1f}s")

    # Step 3: Merge and commit
    print("Step 3/3: Merging and committing index...")
    commit_start = time.perf_counter()
    merged = (
        dataset.create_index_segment_builder()
        .with_segments(segments)
        .build_all()
    )
    dataset.commit_existing_index_segments("vector_idx", column, merged)
    commit_elapsed = time.perf_counter() - commit_start
    print(f"  Committed in {commit_elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    return total_elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build IVF-PQ index on FineWeb-Edu ~1B vector dataset on OBS."
    )
    parser.add_argument(
        "--dataset-uri",
        default="s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance",
    )
    parser.add_argument("--endpoint", default="https://obs.ap-southeast-1.myhuaweicloud.com")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--access-key", default="")
    parser.add_argument("--secret-key", default="")
    parser.add_argument("--column", default="vector")
    parser.add_argument("--metric", default="cosine", choices=["L2", "cosine", "dot"])
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--num-sub-vectors", type=int, default=64)
    parser.add_argument("--num-bits", type=int, default=8, choices=[4, 8])
    parser.add_argument(
        "--mode",
        default="simple",
        choices=["simple", "distributed"],
        help="simple: single create_index() call. distributed: split fragments across workers.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    access_key = args.access_key
    secret_key = args.secret_key
    if not access_key or not secret_key:
        import os

        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        raise ValueError("access-key and secret-key are required (via args or env vars)")

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

    num_partitions = args.num_partitions
    if num_partitions is None:
        num_partitions = compute_default_num_partitions(num_rows)
    print(f"num_partitions: {num_partitions:,}")

    if check_existing_index(dataset):
        if args.replace:
            print("Existing IVF_PQ index found, replacing...")
        else:
            print("Existing IVF_PQ index found, skipping build. Use --replace to rebuild.")
            return

    if args.mode == "simple":
        elapsed = build_index_simple(
            dataset,
            column=args.column,
            metric=args.metric,
            num_partitions=num_partitions,
            num_sub_vectors=args.num_sub_vectors,
            num_bits=args.num_bits,
            replace=args.replace,
        )
    else:
        elapsed = build_index_distributed(
            dataset,
            column=args.column,
            metric=args.metric,
            num_partitions=num_partitions,
            num_sub_vectors=args.num_sub_vectors,
            num_bits=args.num_bits,
            num_workers=args.num_workers,
            replace=args.replace,
        )

    print(f"\nIndex build complete in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    # Verify
    indices = dataset.list_indices()
    print(f"Indices: {[idx.name for idx in indices]}")


if __name__ == "__main__":
    main()
