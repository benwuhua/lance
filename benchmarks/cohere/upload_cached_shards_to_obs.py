#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
from pathlib import Path

from obs import ObsClient

from ingest_fineweb_edu import DEFAULT_OBS_RAW_PREFIX, build_obs_raw_key, list_cached_embedding_shards


def validate_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got value={value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload cached Cohere fineweb-edu embedding shards to OBS raw prefix."
    )
    parser.add_argument("--cache-dir", required=True, help="Embedding cache root containing emb/")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--raw-prefix", default=DEFAULT_OBS_RAW_PREFIX)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    if args.max_files is not None:
        validate_positive_int("max_files", args.max_files)
    return args


def main() -> None:
    args = parse_args()
    shard_paths = list_cached_embedding_shards(args.cache_dir)
    if args.max_files is not None:
        shard_paths = shard_paths[: args.max_files]
    if not shard_paths:
        raise ValueError(f"no complete cached shards found under cache_dir={args.cache_dir}")

    client = ObsClient(
        access_key_id=args.access_key,
        secret_access_key=args.secret_key,
        server=args.endpoint,
    )
    try:
        for shard_path in shard_paths:
            object_key = build_obs_raw_key(shard_path.name, raw_prefix=args.raw_prefix)
            response = client.putFile(args.bucket, object_key, str(shard_path))
            if response.status >= 300:
                raise RuntimeError(
                    f"upload failed: shard={shard_path} key={object_key} status={response.status}"
                )
            print(
                "uploaded "
                f"file={shard_path.name} size={shard_path.stat().st_size} key={object_key}"
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
