# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq


def load_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "cohere"
        / "benchmark_topk_latency.py"
    )
    spec = spec_from_file_location("benchmark_topk_latency", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("8", [8]),
        ("8,32,128", [8, 32, 128]),
        ("8, 32 ,128", [8, 32, 128]),
    ],
)
def test_parse_nprobes(raw_value, expected):
    module = load_benchmark_module()

    assert module.parse_nprobes(raw_value) == expected


@pytest.mark.parametrize("raw_value", ["", "0", "8,0", "a,8"])
def test_parse_nprobes_rejects_invalid_values(raw_value):
    module = load_benchmark_module()

    with pytest.raises(ValueError):
        module.parse_nprobes(raw_value)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2", [2]),
        ("2,5,10", [2, 5, 10]),
    ],
)
def test_parse_refine_factors(raw_value, expected):
    module = load_benchmark_module()

    assert module.parse_refine_factors(raw_value) == expected


def test_validate_positive_int_rejects_zero():
    module = load_benchmark_module()

    with pytest.raises(ValueError, match="top_k"):
        module.validate_positive_int("top_k", 0)


def test_summarize_latency_samples_returns_expected_percentiles():
    module = load_benchmark_module()

    summary = module.summarize_latency_samples([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["mean"] == pytest.approx(3.0)
    assert summary["p50"] == pytest.approx(3.0)
    assert summary["p95"] == pytest.approx(4.8)
    assert summary["p99"] == pytest.approx(4.96)
    assert summary["max"] == pytest.approx(5.0)


def test_validate_source_files_requires_expected_parquet_inputs(tmp_path):
    module = load_benchmark_module()

    dataset_dir = tmp_path / "cohere_medium_1m"
    dataset_dir.mkdir()
    (dataset_dir / "shuffle_train.parquet").touch()
    (dataset_dir / "test.parquet").touch()

    with pytest.raises(FileNotFoundError, match="neighbors.parquet"):
        module.validate_source_files(dataset_dir)


def test_validate_top_k_rejects_values_larger_than_dataset_rows():
    module = load_benchmark_module()

    with pytest.raises(ValueError, match="top_k"):
        module.validate_top_k_against_rows(top_k=6, dataset_rows=5)


def test_inspect_source_dataset_reports_rows_and_query_vectors(tmp_path):
    module = load_benchmark_module()

    dataset_dir = tmp_path / "cohere_medium_1m"
    dataset_dir.mkdir()

    train_table = pa.table(
        {
            "id": [1, 2, 3],
            "emb": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        }
    )
    test_table = pa.table(
        {
            "id": [11, 12],
            "emb": [[0.1, 0.2], [0.3, 0.4]],
        }
    )
    neighbors_table = pa.table({"neighbors_id": [[1, 2], [2, 3]]})

    pq.write_table(train_table, dataset_dir / "shuffle_train.parquet")
    pq.write_table(test_table, dataset_dir / "test.parquet")
    pq.write_table(neighbors_table, dataset_dir / "neighbors.parquet")

    source_info = module.inspect_source_dataset(dataset_dir, query_count=2)

    assert source_info["dataset_rows"] == 3
    assert source_info["vector_field"] == "emb"
    assert source_info["query_ids"] == [11, 12]
    assert source_info["query_vectors"] == [[0.1, 0.2], [0.3, 0.4]]


class FakeDataset:
    def __init__(self, rows_per_query):
        self.rows_per_query = rows_per_query
        self.calls = []

    def to_table(self, **kwargs):
        self.calls.append(kwargs)
        row_count = self.rows_per_query[len(self.calls) - 1]
        return pa.table({"id": list(range(row_count))})


def test_run_query_experiment_excludes_warmup_queries(monkeypatch):
    module = load_benchmark_module()
    dataset = FakeDataset([2, 3, 3])

    perf_counter_values = iter([0.0, 1.0, 10.0, 14.0, 20.0, 29.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(perf_counter_values))

    result = module.run_query_experiment(
        dataset,
        experiment_name="flat",
        query_vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        top_k=100,
        warmup_queries=1,
        use_index=False,
    )

    assert result["name"] == "flat"
    assert result["use_index"] is False
    assert result["latency_ms"]["mean"] == pytest.approx(6_500.0)
    assert result["average_rows_returned"] == pytest.approx(3.0)
    assert len(dataset.calls) == 3
    assert dataset.calls[0]["columns"] == ["id", "_distance"]


def test_run_query_experiment_records_ann_metadata(monkeypatch):
    module = load_benchmark_module()
    dataset = FakeDataset([4, 4])

    perf_counter_values = iter([0.0, 0.5, 1.0, 2.5])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(perf_counter_values))

    result = module.run_query_experiment(
        dataset,
        experiment_name="ivf_pq_np32",
        query_vectors=[[0.1, 0.2], [0.3, 0.4]],
        top_k=100,
        warmup_queries=0,
        use_index=True,
        index_type="IVF_PQ",
        index_params={"nprobes": 32, "refine_factor": 10},
    )

    assert result["name"] == "ivf_pq_np32"
    assert result["use_index"] is True
    assert result["index_type"] == "IVF_PQ"
    assert result["index_params"] == {"nprobes": 32, "refine_factor": 10}
    assert result["latency_ms"]["p50"] == pytest.approx(1_000.0)


def test_build_ann_experiment_name_for_ivf_rq():
    module = load_benchmark_module()

    assert (
        module.build_ann_experiment_name("IVF_RQ", nprobes=256, refine_factor=2)
        == "ivf_rq_np256_rf2"
    )


def test_build_result_document_and_write_json(tmp_path):
    module = load_benchmark_module()

    result_document = module.build_result_document(
        run_label="cohere1m-topk100k",
        host_name="x86-host",
        dataset_metadata={
            "name": "cohere_medium_1m",
            "source_dir": "/data/work/vectordb-bench/dataset/cohere/cohere_medium_1m",
            "lance_uri": "/data/work/lance-cohere1m/cohere1m.lance",
            "dataset_rows": 1_000_000,
            "dim": 768,
            "metric": "cosine",
        },
        query_metadata={"query_count": 100, "top_k": 100_000, "warmup_queries": 5},
        experiments=[{"name": "flat", "use_index": False, "latency_ms": {"mean": 1.0}}],
    )

    result_path = tmp_path / "result.json"
    module.write_result_json(result_document, result_path)

    written = result_path.read_text()
    assert '"run_label": "cohere1m-topk100k"' in written
    assert '"host": "x86-host"' in written
    assert '"query_count": 100' in written


def test_compute_recall_at_k_uses_ground_truth_prefix():
    module = load_benchmark_module()

    recall = module.compute_recall_at_k(
        query_results=[[10, 20, 30], [50, 60, 70]],
        ground_truth=[[30, 20, 99], [80, 70, 60]],
        k=2,
    )

    assert recall == pytest.approx(0.25)


def test_finalize_experiment_uses_post_warmup_ground_truth():
    module = load_benchmark_module()

    finalized = module.finalize_experiment(
        {
            "name": "ivf_pq_np8",
            "query_result_ids": [[10, 20], [30, 40]],
        },
        ground_truth=[[0, 1], [20, 10], [50, 30]],
        recall_k=2,
        warmup_queries=1,
    )

    assert finalized["recall_at_2"] == pytest.approx(0.75)


def test_to_fixed_size_vector_array_converts_list_vectors():
    module = load_benchmark_module()

    vector_array = pa.array([[1.0, 2.0], [3.0, 4.0]])
    fixed_size_array = module.to_fixed_size_vector_array(vector_array)

    assert fixed_size_array.type.list_size == 2
    assert fixed_size_array.to_pylist() == [[1.0, 2.0], [3.0, 4.0]]


def test_to_fixed_size_vector_array_accepts_chunked_array():
    module = load_benchmark_module()

    vector_column = pa.chunked_array([pa.array([[1.0, 2.0]]), pa.array([[3.0, 4.0]])])
    fixed_size_array = module.to_fixed_size_vector_array(vector_column)

    assert fixed_size_array.type.list_size == 2
    assert fixed_size_array.to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
