#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ingest_fineweb_edu import (
    DEFAULT_OBS_RAW_PREFIX,
    DEFAULT_REPO_ID,
    build_obs_raw_key,
    download_embedding_shard,
    filter_embedding_shards,
    slice_shard_paths,
    validate_positive_int,
)


def create_obs_client(*, endpoint: str, access_key: str, secret_key: str) -> ObsClient:
    from obs import ObsClient

    return ObsClient(
        access_key_id=access_key,
        secret_access_key=secret_key,
        server=endpoint,
    )


def obs_object_exists(*, client: ObsClient, bucket: str, object_key: str) -> bool:
    response = client.getObjectMetadata(bucket, object_key)
    if response.status == 200:
        return True
    if response.status == 404:
        return False
    raise RuntimeError(
        f"head failed: bucket={bucket} key={object_key} status={response.status}"
    )


def upload_local_shard(
    *,
    client: ObsClient,
    bucket: str,
    object_key: str,
    local_path: str,
) -> None:
    response = client.putFile(bucket, object_key, local_path)
    if response.status >= 300:
        raise RuntimeError(
            f"upload failed: path={local_path} key={object_key} status={response.status}"
        )


def sync_embedding_shard(
    *,
    repo_id: str,
    shard_path: str,
    cache_dir: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    raw_prefix: str,
    skip_existing: bool,
) -> dict[str, str | bool]:
    local_path = download_embedding_shard(
        repo_id=repo_id,
        shard_path=shard_path,
        cache_dir=cache_dir,
    )
    object_key = build_obs_raw_key(local_path.name, raw_prefix=raw_prefix)

    client = create_obs_client(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
    )
    try:
        if skip_existing and obs_object_exists(
            client=client,
            bucket=bucket,
            object_key=object_key,
        ):
            return {
                "shard_path": shard_path,
                "local_path": str(local_path),
                "object_key": object_key,
                "uploaded": False,
            }
        upload_local_shard(
            client=client,
            bucket=bucket,
            object_key=object_key,
            local_path=str(local_path),
        )
        return {
            "shard_path": shard_path,
            "local_path": str(local_path),
            "object_key": object_key,
            "uploaded": True,
        }
    finally:
        if hasattr(client, "close"):
            client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Cohere fineweb-edu embedding shards from Hugging Face and sync them to OBS raw."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--raw-prefix", default=DEFAULT_OBS_RAW_PREFIX)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    args = parser.parse_args()
    validate_positive_int("workers", args.workers)
    if args.max_shards is not None:
        validate_positive_int("max_shards", args.max_shards)
    if args.start_shard < 0:
        raise ValueError(f"start_shard must be >= 0, got value={args.start_shard}")
    return args


def main() -> None:
    from huggingface_hub import HfApi

    args = parse_args()
    api = HfApi()
    shard_paths = slice_shard_paths(
        filter_embedding_shards(api.list_repo_files(args.repo_id, repo_type="dataset")),
        start_shard=args.start_shard,
        max_shards=args.max_shards,
    )
    if not shard_paths:
        raise ValueError(
            "no embedding shards selected after slicing, "
            f"start_shard={args.start_shard} max_shards={args.max_shards}"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                sync_embedding_shard,
                repo_id=args.repo_id,
                shard_path=shard_path,
                cache_dir=args.cache_dir,
                bucket=args.bucket,
                endpoint=args.endpoint,
                access_key=args.access_key,
                secret_key=args.secret_key,
                raw_prefix=args.raw_prefix,
                skip_existing=args.skip_existing,
            ): shard_path
            for shard_path in shard_paths
        }
        for future in as_completed(futures):
            result = future.result()
            action = "uploaded" if result["uploaded"] else "skipped"
            print(
                f"{action} shard={result['shard_path']} "
                f"key={result['object_key']} local_path={result['local_path']}"
            )


if __name__ == "__main__":
    main()
