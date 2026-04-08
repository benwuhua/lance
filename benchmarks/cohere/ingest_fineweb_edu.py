#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import pyarrow as pa


DEFAULT_REPO_ID = "Cohere/fineweb-edu-emb"
DEFAULT_VECTOR_DIM = 1024
DEFAULT_OBS_RAW_PREFIX = "fineweb-edu-emb-raw/emb"


def validate_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got value={value}")
    return value


def filter_embedding_shards(paths: Iterable[str]) -> list[str]:
    return sorted(
        path for path in paths if path.startswith("emb/") and path.endswith(".npy")
    )


def build_obs_raw_key(filename: str, *, raw_prefix: str = DEFAULT_OBS_RAW_PREFIX) -> str:
    return f"{raw_prefix.rstrip('/')}/{filename}"


def list_cached_embedding_shards(cache_dir: str | Path) -> list[Path]:
    emb_dir = Path(cache_dir) / "emb"
    if not emb_dir.exists():
        return []
    return sorted(
        path
        for path in emb_dir.iterdir()
        if path.is_file() and path.suffix == ".npy" and not path.name.endswith(".incomplete")
    )


def slice_shard_paths(
    shard_paths: list[str], *, start_shard: int = 0, max_shards: int | None = None
) -> list[str]:
    if start_shard < 0:
        raise ValueError(f"start_shard must be >= 0, got value={start_shard}")
    sliced_paths = shard_paths[start_shard:]
    if max_shards is not None:
        validate_positive_int("max_shards", max_shards)
        sliced_paths = sliced_paths[:max_shards]
    return sliced_paths


def build_schema(vector_dim: int) -> pa.Schema:
    validate_positive_int("vector_dim", vector_dim)
    return pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def build_hf_download_url(repo_id: str, shard_path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{shard_path}"


def download_public_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    temporary_path = destination.with_suffix(destination.suffix + ".incomplete")
    request = Request(url, headers={"User-Agent": "lance-fineweb-ingest/1.0"})
    with urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temporary_path.replace(destination)
    return destination


def download_embedding_shard(
    *, repo_id: str, shard_path: str, cache_dir: str | None = None
) -> Path:
    cache_root = Path(cache_dir or "~/.cache/lance-fineweb").expanduser()
    local_path = cache_root / repo_id.replace("/", "__") / shard_path
    return download_public_file(build_hf_download_url(repo_id, shard_path), local_path)


def download_obs_raw_shard(
    *,
    bucket: str,
    shard_path: str,
    cache_dir: str | None,
    endpoint: str,
    access_key: str,
    secret_key: str,
    raw_prefix: str = DEFAULT_OBS_RAW_PREFIX,
) -> Path:
    from obs import ObsClient

    cache_root = Path(cache_dir or "~/.cache/lance-fineweb-raw").expanduser()
    local_path = cache_root / shard_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path

    temporary_path = local_path.with_suffix(local_path.suffix + ".incomplete")
    object_key = build_obs_raw_key(Path(shard_path).name, raw_prefix=raw_prefix)
    client = ObsClient(
        access_key_id=access_key,
        secret_access_key=secret_key,
        server=endpoint,
    )
    try:
        response = client.getObject(bucket, object_key, downloadPath=str(temporary_path))
        if response.status >= 300:
            raise RuntimeError(
                "OBS raw shard download failed: "
                f"bucket={bucket} key={object_key} status={response.status}"
            )
        temporary_path.replace(local_path)
        return local_path
    finally:
        client.close()


def resolve_shard_path(
    *,
    repo_id: str,
    shard_path: str,
    cache_dir: str | None,
    raw_bucket: str | None,
    raw_endpoint: str | None,
    raw_access_key: str | None,
    raw_secret_key: str | None,
    raw_prefix: str = DEFAULT_OBS_RAW_PREFIX,
) -> Path:
    if all([raw_bucket, raw_endpoint, raw_access_key, raw_secret_key]):
        try:
            return download_obs_raw_shard(
                bucket=raw_bucket,
                shard_path=shard_path,
                cache_dir=cache_dir,
                endpoint=raw_endpoint,
                access_key=raw_access_key,
                secret_key=raw_secret_key,
                raw_prefix=raw_prefix,
            )
        except Exception as error:
            print(
                "falling back to Hugging Face shard download "
                f"for shard={shard_path} after OBS raw download failed: {error}"
            )
    return download_embedding_shard(
        repo_id=repo_id,
        shard_path=shard_path,
        cache_dir=cache_dir,
    )


def compute_next_row_id(
    *,
    repo_id: str,
    shard_paths: list[str],
    start_shard: int,
    cache_dir: str | None,
    raw_bucket: str | None,
    raw_endpoint: str | None,
    raw_access_key: str | None,
    raw_secret_key: str | None,
    raw_prefix: str = DEFAULT_OBS_RAW_PREFIX,
) -> int:
    if start_shard <= 0:
        return 0

    next_row_id = 0
    for shard_path in shard_paths[:start_shard]:
        local_path = resolve_shard_path(
            repo_id=repo_id,
            shard_path=shard_path,
            cache_dir=cache_dir,
            raw_bucket=raw_bucket,
            raw_endpoint=raw_endpoint,
            raw_access_key=raw_access_key,
            raw_secret_key=raw_secret_key,
            raw_prefix=raw_prefix,
        )
        vectors = np.load(local_path, mmap_mode="r")
        if vectors.ndim != 2:
            raise ValueError(
                f"embedding shard must be 2D, got shard={shard_path}, shape={vectors.shape}"
            )
        next_row_id += vectors.shape[0]
    return next_row_id


def create_storage_options(args: argparse.Namespace) -> dict[str, str] | None:
    access_key = args.access_key or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = args.secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint = args.endpoint or os.getenv("AWS_ENDPOINT")
    region = args.region or os.getenv("AWS_REGION")
    virtual_hosted = (
        args.virtual_hosted_style_request
        or os.getenv("AWS_VIRTUAL_HOSTED_STYLE_REQUEST")
    )
    use_opendal = args.use_opendal or os.getenv("USE_OPENDAL")

    options = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "aws_endpoint": endpoint,
        "region": region,
        "virtual_hosted_style_request": virtual_hosted,
        "use_opendal": use_opendal,
    }
    filtered = {key: value for key, value in options.items() if value}
    return filtered or None


def build_record_batch(
    vectors: np.ndarray, *, start_id: int, vector_dim: int
) -> pa.RecordBatch:
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")
    if vectors.shape[1] != vector_dim:
        raise ValueError(
            "vector dimension mismatch, "
            f"got vector_dim={vectors.shape[1]}, expected={vector_dim}"
        )
    ids = pa.array(range(start_id, start_id + vectors.shape[0]), type=pa.int64())
    flattened = pa.array(vectors.reshape(-1), type=pa.float32())
    vector_array = pa.FixedSizeListArray.from_arrays(flattened, list_size=vector_dim)
    return pa.RecordBatch.from_arrays([ids, vector_array], names=["id", "vector"])


def iter_embedding_batches(
    *,
    repo_id: str,
    shard_paths: list[str],
    max_rows: int,
    batch_rows: int,
    vector_dim: int,
    cache_dir: str | None = None,
    raw_bucket: str | None = None,
    raw_endpoint: str | None = None,
    raw_access_key: str | None = None,
    raw_secret_key: str | None = None,
    raw_prefix: str = DEFAULT_OBS_RAW_PREFIX,
    starting_row_id: int = 0,
):
    validate_positive_int("max_rows", max_rows)
    validate_positive_int("batch_rows", batch_rows)
    validate_positive_int("vector_dim", vector_dim)

    total_rows_written = 0
    next_row_id = starting_row_id

    for shard_index, shard_path in enumerate(shard_paths):
        if total_rows_written >= max_rows:
            break

        local_path = resolve_shard_path(
            repo_id=repo_id,
            shard_path=shard_path,
            cache_dir=cache_dir,
            raw_bucket=raw_bucket,
            raw_endpoint=raw_endpoint,
            raw_access_key=raw_access_key,
            raw_secret_key=raw_secret_key,
            raw_prefix=raw_prefix,
        )
        vectors = np.load(local_path, mmap_mode="r")
        if vectors.ndim != 2:
            raise ValueError(
                f"embedding shard must be 2D, got shard={shard_path}, shape={vectors.shape}"
            )
        if vectors.shape[1] != vector_dim:
            raise ValueError(
                "embedding shard has unexpected dimension, "
                f"got shard={shard_path}, shape={vectors.shape}, expected_dim={vector_dim}"
            )

        rows_remaining = max_rows - total_rows_written
        shard_rows = min(vectors.shape[0], rows_remaining)
        print(
            "processing shard "
            f"index={shard_index} path={shard_path} shard_rows={shard_rows} "
            f"total_rows_written={total_rows_written}"
        )

        for start in range(0, shard_rows, batch_rows):
            stop = min(start + batch_rows, shard_rows)
            batch_vectors = np.asarray(vectors[start:stop], dtype=np.float32)
            yield build_record_batch(
                batch_vectors,
                start_id=next_row_id,
                vector_dim=vector_dim,
            )
            batch_size = stop - start
            next_row_id += batch_size
            total_rows_written += batch_size


def resolve_output_path(output_uri: str) -> Path | None:
    if "://" in output_uri:
        return None
    return Path(output_uri)


def write_dataset(args: argparse.Namespace) -> None:
    import lance
    from huggingface_hub import HfApi

    storage_options = create_storage_options(args)
    output_path = resolve_output_path(args.output)
    if output_path is not None and output_path.exists() and args.mode == "overwrite":
        shutil.rmtree(output_path)

    api = HfApi()
    all_shard_paths = filter_embedding_shards(
        api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    )
    shard_paths = slice_shard_paths(
        all_shard_paths,
        start_shard=args.start_shard,
        max_shards=args.max_shards,
    )
    if not shard_paths:
        raise ValueError(f"no embedding shards found for repo_id={args.repo_id}")

    starting_row_id = args.starting_row_id
    if starting_row_id is None:
        starting_row_id = compute_next_row_id(
            repo_id=args.repo_id,
            shard_paths=all_shard_paths,
            start_shard=args.start_shard,
            cache_dir=args.cache_dir,
            raw_bucket=args.raw_bucket,
            raw_endpoint=args.raw_endpoint,
            raw_access_key=args.raw_access_key,
            raw_secret_key=args.raw_secret_key,
            raw_prefix=args.raw_prefix,
        )

    schema = build_schema(args.vector_dim)
    dataset = lance.write_dataset(
        iter_embedding_batches(
            repo_id=args.repo_id,
            shard_paths=shard_paths,
            max_rows=args.max_rows,
            batch_rows=args.batch_rows,
            vector_dim=args.vector_dim,
            cache_dir=args.cache_dir,
            raw_bucket=args.raw_bucket,
            raw_endpoint=args.raw_endpoint,
            raw_access_key=args.raw_access_key,
            raw_secret_key=args.raw_secret_key,
            raw_prefix=args.raw_prefix,
            starting_row_id=starting_row_id,
        ),
        args.output,
        schema=schema,
        mode=args.mode,
        max_rows_per_group=args.max_rows_per_group,
        max_rows_per_file=args.max_rows_per_file,
        storage_options=storage_options,
    )
    print(
        "write complete "
        f"uri={dataset.uri} rows={dataset.count_rows()} vector_dim={args.vector_dim}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Cohere fineweb-edu embeddings into a Lance dataset."
    )
    parser.add_argument("--output", required=True, help="Destination Lance dataset URI")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--batch-rows", type=int, default=16_384)
    parser.add_argument("--max-rows-per-group", type=int, default=8_192)
    parser.add_argument("--max-rows-per-file", type=int, default=1_000_000)
    parser.add_argument("--vector-dim", type=int, default=DEFAULT_VECTOR_DIM)
    parser.add_argument("--mode", choices=["create", "overwrite", "append"], default="create")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--starting-row-id", type=int)
    parser.add_argument("--access-key")
    parser.add_argument("--secret-key")
    parser.add_argument("--endpoint")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--virtual-hosted-style-request")
    parser.add_argument("--use-opendal")
    parser.add_argument("--raw-bucket")
    parser.add_argument("--raw-endpoint")
    parser.add_argument("--raw-access-key")
    parser.add_argument("--raw-secret-key")
    parser.add_argument("--raw-prefix", default=DEFAULT_OBS_RAW_PREFIX)

    args = parser.parse_args()
    validate_positive_int("max_rows", args.max_rows)
    validate_positive_int("batch_rows", args.batch_rows)
    validate_positive_int("max_rows_per_group", args.max_rows_per_group)
    validate_positive_int("max_rows_per_file", args.max_rows_per_file)
    validate_positive_int("vector_dim", args.vector_dim)
    if args.max_shards is not None:
        validate_positive_int("max_shards", args.max_shards)
    if args.start_shard < 0:
        raise ValueError(f"start_shard must be >= 0, got value={args.start_shard}")
    if args.starting_row_id is not None and args.starting_row_id < 0:
        raise ValueError(
            f"starting_row_id must be >= 0, got value={args.starting_row_id}"
        )
    return args


def main() -> None:
    write_dataset(parse_args())


if __name__ == "__main__":
    main()
