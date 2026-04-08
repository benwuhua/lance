# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def load_ingest_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "cohere"
        / "ingest_fineweb_edu.py"
    )
    spec = spec_from_file_location("ingest_fineweb_edu", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_sync_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "cohere"
        / "sync_hf_shards_to_obs_raw.py"
    )
    spec = spec_from_file_location("sync_hf_shards_to_obs_raw", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filter_embedding_shards_sorts_npy_paths():
    module = load_ingest_module()

    shard_paths = module.filter_embedding_shards(
        [
            "README.md",
            "emb/b.npy",
            "emb/a.npy",
            "corpus/a.jsonl.zst",
            "emb/ignore.txt",
        ]
    )

    assert shard_paths == ["emb/a.npy", "emb/b.npy"]


def test_build_record_batch_creates_sequential_ids():
    module = load_ingest_module()

    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    batch = module.build_record_batch(vectors, start_id=10, vector_dim=2)

    assert batch.num_rows == 2
    assert batch["id"].to_pylist() == [10, 11]
    assert batch["vector"].to_pylist() == [[1.0, 2.0], [3.0, 4.0]]


def test_build_hf_download_url_uses_dataset_resolve_path():
    module = load_ingest_module()

    assert module.build_hf_download_url(
        "Cohere/fineweb-edu-emb",
        "emb/example.npy",
    ) == (
        "https://huggingface.co/datasets/Cohere/fineweb-edu-emb/resolve/main/"
        "emb/example.npy"
    )


def test_build_obs_raw_key_uses_fixed_prefix():
    module = load_ingest_module()

    assert (
        module.build_obs_raw_key("new_CC-MAIN-2013-20-train-00000-of-00014.npy")
        == "fineweb-edu-emb-raw/emb/new_CC-MAIN-2013-20-train-00000-of-00014.npy"
    )


def test_list_cached_embedding_shards_ignores_incomplete_files(tmp_path):
    module = load_ingest_module()

    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    (emb_dir / "b.npy").write_bytes(b"b")
    (emb_dir / "a.npy").write_bytes(b"a")
    (emb_dir / "c.npy.incomplete").write_bytes(b"c")
    (emb_dir / "note.txt").write_text("ignore")

    assert module.list_cached_embedding_shards(tmp_path) == [
        emb_dir / "a.npy",
        emb_dir / "b.npy",
    ]


def test_iter_embedding_batches_truncates_at_max_rows(tmp_path, monkeypatch):
    module = load_ingest_module()

    shard1 = tmp_path / "shard1.npy"
    shard2 = tmp_path / "shard2.npy"
    np.save(shard1, np.arange(12, dtype=np.float32).reshape(3, 4))
    np.save(shard2, np.arange(12, 32, dtype=np.float32).reshape(5, 4))

    file_map = {
        "emb/001.npy": str(shard1),
        "emb/002.npy": str(shard2),
    }

    def fake_download(*, repo_id, shard_path, cache_dir=None):
        assert repo_id == "Cohere/fineweb-edu-emb"
        assert cache_dir == "/tmp/hf-cache"
        return Path(file_map[shard_path])

    monkeypatch.setattr(module, "download_embedding_shard", fake_download)

    batches = list(
        module.iter_embedding_batches(
            repo_id="Cohere/fineweb-edu-emb",
            shard_paths=["emb/001.npy", "emb/002.npy"],
            max_rows=6,
            batch_rows=2,
            vector_dim=4,
            cache_dir="/tmp/hf-cache",
        )
    )

    assert [batch.num_rows for batch in batches] == [2, 1, 2, 1]
    assert batches[0]["id"].to_pylist() == [0, 1]
    assert batches[-1]["id"].to_pylist() == [5]


def test_resolve_shard_path_prefers_obs_raw(tmp_path, monkeypatch):
    module = load_ingest_module()

    cached_path = tmp_path / "emb" / "001.npy"
    cached_path.parent.mkdir()
    cached_path.write_bytes(b"obs")

    calls = []

    def fake_download_obs_raw_shard(**kwargs):
        calls.append(("obs", kwargs))
        return cached_path

    def fake_download_embedding_shard(**kwargs):
        calls.append(("hf", kwargs))
        return cached_path

    monkeypatch.setattr(module, "download_obs_raw_shard", fake_download_obs_raw_shard)
    monkeypatch.setattr(module, "download_embedding_shard", fake_download_embedding_shard)

    resolved = module.resolve_shard_path(
        repo_id="Cohere/fineweb-edu-emb",
        shard_path="emb/001.npy",
        cache_dir=str(tmp_path),
        raw_bucket="knowledgebase-5f43",
        raw_endpoint="https://obs.ap-southeast-1.myhuaweicloud.com",
        raw_access_key="ak",
        raw_secret_key="sk",
        raw_prefix="fineweb-edu-emb-raw/emb",
    )

    assert resolved == cached_path
    assert calls == [
        (
            "obs",
            {
                "bucket": "knowledgebase-5f43",
                "shard_path": "emb/001.npy",
                "cache_dir": str(tmp_path),
                "endpoint": "https://obs.ap-southeast-1.myhuaweicloud.com",
                "access_key": "ak",
                "secret_key": "sk",
                "raw_prefix": "fineweb-edu-emb-raw/emb",
            },
        )
    ]


def test_resolve_shard_path_falls_back_to_huggingface(tmp_path, monkeypatch):
    module = load_ingest_module()

    cached_path = tmp_path / "emb" / "001.npy"
    cached_path.parent.mkdir()
    cached_path.write_bytes(b"hf")

    calls = []

    def fake_download_embedding_shard(**kwargs):
        calls.append(kwargs)
        return cached_path

    monkeypatch.setattr(module, "download_embedding_shard", fake_download_embedding_shard)

    resolved = module.resolve_shard_path(
        repo_id="Cohere/fineweb-edu-emb",
        shard_path="emb/001.npy",
        cache_dir=str(tmp_path),
        raw_bucket=None,
        raw_endpoint=None,
        raw_access_key=None,
        raw_secret_key=None,
        raw_prefix="fineweb-edu-emb-raw/emb",
    )

    assert resolved == cached_path
    assert calls == [
        {
            "repo_id": "Cohere/fineweb-edu-emb",
            "shard_path": "emb/001.npy",
            "cache_dir": str(tmp_path),
        }
    ]


def test_resolve_shard_path_falls_back_to_huggingface_when_obs_raw_download_fails(
    tmp_path, monkeypatch
):
    module = load_ingest_module()

    cached_path = tmp_path / "emb" / "001.npy"
    cached_path.parent.mkdir()
    cached_path.write_bytes(b"hf")

    calls = []

    def fake_download_obs_raw_shard(**kwargs):
        calls.append(("obs", kwargs))
        raise RuntimeError("OBS raw shard download failed: status=404")

    def fake_download_embedding_shard(**kwargs):
        calls.append(("hf", kwargs))
        return cached_path

    monkeypatch.setattr(module, "download_obs_raw_shard", fake_download_obs_raw_shard)
    monkeypatch.setattr(module, "download_embedding_shard", fake_download_embedding_shard)

    resolved = module.resolve_shard_path(
        repo_id="Cohere/fineweb-edu-emb",
        shard_path="emb/001.npy",
        cache_dir=str(tmp_path),
        raw_bucket="knowledgebase-5f43",
        raw_endpoint="https://obs.ap-southeast-1.myhuaweicloud.com",
        raw_access_key="ak",
        raw_secret_key="sk",
        raw_prefix="fineweb-edu-emb-raw/emb",
    )

    assert resolved == cached_path
    assert calls == [
        (
            "obs",
            {
                "bucket": "knowledgebase-5f43",
                "shard_path": "emb/001.npy",
                "cache_dir": str(tmp_path),
                "endpoint": "https://obs.ap-southeast-1.myhuaweicloud.com",
                "access_key": "ak",
                "secret_key": "sk",
                "raw_prefix": "fineweb-edu-emb-raw/emb",
            },
        ),
        (
            "hf",
            {
                "repo_id": "Cohere/fineweb-edu-emb",
                "shard_path": "emb/001.npy",
                "cache_dir": str(tmp_path),
            },
        ),
    ]


def test_slice_shard_paths_applies_start_and_limit():
    module = load_ingest_module()

    shard_paths = [f"emb/{i:03d}.npy" for i in range(5)]

    assert module.slice_shard_paths(shard_paths, start_shard=2, max_shards=2) == [
        "emb/002.npy",
        "emb/003.npy",
    ]


def test_compute_next_row_id_counts_rows_before_start_shard(tmp_path, monkeypatch):
    module = load_ingest_module()

    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    third = tmp_path / "third.npy"
    np.save(first, np.zeros((3, 2), dtype=np.float32))
    np.save(second, np.zeros((5, 2), dtype=np.float32))
    np.save(third, np.zeros((7, 2), dtype=np.float32))

    file_map = {
        "emb/000.npy": first,
        "emb/001.npy": second,
        "emb/002.npy": third,
    }

    def fake_resolve(**kwargs):
        return file_map[kwargs["shard_path"]]

    monkeypatch.setattr(module, "resolve_shard_path", fake_resolve)

    assert (
        module.compute_next_row_id(
            repo_id="Cohere/fineweb-edu-emb",
            shard_paths=["emb/000.npy", "emb/001.npy", "emb/002.npy"],
            start_shard=2,
            cache_dir=str(tmp_path),
            raw_bucket="bucket",
            raw_endpoint="endpoint",
            raw_access_key="ak",
            raw_secret_key="sk",
            raw_prefix="fineweb-edu-emb-raw/emb",
        )
        == 8
    )


def test_create_storage_options_prefers_cli_values(monkeypatch):
    module = load_ingest_module()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-sk")
    monkeypatch.setenv("AWS_ENDPOINT", "env-endpoint")
    monkeypatch.setenv("AWS_REGION", "env-region")
    monkeypatch.setenv("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "env-vhost")
    monkeypatch.setenv("USE_OPENDAL", "env-opendal")

    args = module.argparse.Namespace(
        access_key="cli-ak",
        secret_key="cli-sk",
        endpoint="cli-endpoint",
        region="cli-region",
        virtual_hosted_style_request="cli-vhost",
        use_opendal="cli-opendal",
    )

    assert module.create_storage_options(args) == {
        "aws_access_key_id": "cli-ak",
        "aws_secret_access_key": "cli-sk",
        "aws_endpoint": "cli-endpoint",
        "region": "cli-region",
        "virtual_hosted_style_request": "cli-vhost",
        "use_opendal": "cli-opendal",
    }


def test_build_record_batch_rejects_wrong_vector_dim():
    module = load_ingest_module()

    with pytest.raises(ValueError, match="vector dimension mismatch"):
        module.build_record_batch(
            np.array([[1.0, 2.0]], dtype=np.float32),
            start_id=0,
            vector_dim=3,
        )


def test_sync_embedding_shard_skips_existing_raw_object(tmp_path, monkeypatch):
    module = load_sync_module()

    local_shard = tmp_path / "emb" / "001.npy"
    local_shard.parent.mkdir()
    local_shard.write_bytes(b"cached")

    calls = []

    def fake_download_embedding_shard(**kwargs):
        calls.append(("download", kwargs))
        return local_shard

    def fake_create_obs_client(*, endpoint, access_key, secret_key):
        calls.append(
            (
                "client",
                {
                    "endpoint": endpoint,
                    "access_key": access_key,
                    "secret_key": secret_key,
                },
            )
        )
        return object()

    def fake_obs_object_exists(*, client, bucket, object_key):
        calls.append(("exists", {"bucket": bucket, "object_key": object_key}))
        return True

    def fake_upload_local_shard(**kwargs):
        calls.append(("upload", kwargs))

    monkeypatch.setattr(module, "download_embedding_shard", fake_download_embedding_shard)
    monkeypatch.setattr(module, "create_obs_client", fake_create_obs_client)
    monkeypatch.setattr(module, "obs_object_exists", fake_obs_object_exists)
    monkeypatch.setattr(module, "upload_local_shard", fake_upload_local_shard)

    result = module.sync_embedding_shard(
        repo_id="Cohere/fineweb-edu-emb",
        shard_path="emb/001.npy",
        cache_dir=str(tmp_path),
        bucket="knowledgebase-5f43",
        endpoint="https://obs.ap-southeast-1.myhuaweicloud.com",
        access_key="ak",
        secret_key="sk",
        raw_prefix="fineweb-edu-emb-raw/emb",
        skip_existing=True,
    )

    assert result == {
        "shard_path": "emb/001.npy",
        "local_path": str(local_shard),
        "object_key": "fineweb-edu-emb-raw/emb/001.npy",
        "uploaded": False,
    }
    assert [call[0] for call in calls] == ["download", "client", "exists"]


def test_sync_embedding_shard_uploads_missing_raw_object(tmp_path, monkeypatch):
    module = load_sync_module()

    local_shard = tmp_path / "emb" / "001.npy"
    local_shard.parent.mkdir()
    local_shard.write_bytes(b"cached")

    calls = []

    def fake_download_embedding_shard(**kwargs):
        calls.append(("download", kwargs))
        return local_shard

    def fake_create_obs_client(*, endpoint, access_key, secret_key):
        calls.append(
            (
                "client",
                {
                    "endpoint": endpoint,
                    "access_key": access_key,
                    "secret_key": secret_key,
                },
            )
        )
        return object()

    def fake_obs_object_exists(*, client, bucket, object_key):
        calls.append(("exists", {"bucket": bucket, "object_key": object_key}))
        return False

    def fake_upload_local_shard(**kwargs):
        calls.append(("upload", kwargs))

    monkeypatch.setattr(module, "download_embedding_shard", fake_download_embedding_shard)
    monkeypatch.setattr(module, "create_obs_client", fake_create_obs_client)
    monkeypatch.setattr(module, "obs_object_exists", fake_obs_object_exists)
    monkeypatch.setattr(module, "upload_local_shard", fake_upload_local_shard)

    result = module.sync_embedding_shard(
        repo_id="Cohere/fineweb-edu-emb",
        shard_path="emb/001.npy",
        cache_dir=str(tmp_path),
        bucket="knowledgebase-5f43",
        endpoint="https://obs.ap-southeast-1.myhuaweicloud.com",
        access_key="ak",
        secret_key="sk",
        raw_prefix="fineweb-edu-emb-raw/emb",
        skip_existing=True,
    )

    assert result == {
        "shard_path": "emb/001.npy",
        "local_path": str(local_shard),
        "object_key": "fineweb-edu-emb-raw/emb/001.npy",
        "uploaded": True,
    }
    assert [call[0] for call in calls] == ["download", "client", "exists", "upload"]
