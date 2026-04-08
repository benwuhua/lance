#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def validate_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got value={value}")
    return value


def parse_positive_int_list(name: str, raw_value: str) -> list[int]:
    if not raw_value.strip():
        raise ValueError(f"{name} must contain at least one integer value")

    parsed_values = []
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"{name} contains an empty item: raw_value={raw_value!r}")
        try:
            parsed_value = int(item)
        except ValueError as exc:
            raise ValueError(
                f"{name} must contain only integers, got item={item!r}"
            ) from exc
        parsed_values.append(validate_positive_int(name, parsed_value))
    return parsed_values


def parse_nprobes(raw_value: str) -> list[int]:
    return parse_positive_int_list("nprobes", raw_value)


def parse_refine_factors(raw_value: str) -> list[int]:
    return parse_positive_int_list("refine_factor", raw_value)


def build_ann_experiment_name(index_type: str, *, nprobes: int, refine_factor: int) -> str:
    return f"{index_type.lower()}_np{nprobes}_rf{refine_factor}"


def interpolate_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("latency samples must not be empty")

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    weight = rank - lower_index
    return sorted_values[lower_index] + (
        sorted_values[upper_index] - sorted_values[lower_index]
    ) * weight


def summarize_latency_samples(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("latency samples must not be empty")

    sorted_samples = sorted(samples_ms)
    sample_count = len(sorted_samples)
    return {
        "mean": sum(sorted_samples) / sample_count,
        "p50": interpolate_percentile(sorted_samples, 0.50),
        "p95": interpolate_percentile(sorted_samples, 0.95),
        "p99": interpolate_percentile(sorted_samples, 0.99),
        "max": sorted_samples[-1],
    }


def validate_source_files(dataset_dir: str | Path) -> dict[str, Path]:
    dataset_path = Path(dataset_dir)
    required_files = {
        "train": dataset_path / "shuffle_train.parquet",
        "test": dataset_path / "test.parquet",
        "neighbors": dataset_path / "neighbors.parquet",
    }

    for path in required_files.values():
        if not path.exists():
            raise FileNotFoundError(f"required dataset file is missing: path={path}")

    return required_files


def validate_top_k_against_rows(*, top_k: int, dataset_rows: int) -> int:
    validate_positive_int("top_k", top_k)
    validate_positive_int("dataset_rows", dataset_rows)
    if top_k > dataset_rows:
        raise ValueError(
            f"top_k must be <= dataset_rows, got top_k={top_k}, dataset_rows={dataset_rows}"
        )
    return top_k


def inspect_source_dataset(dataset_dir: str | Path, *, query_count: int) -> dict[str, object]:
    validate_positive_int("query_count", query_count)
    source_files = validate_source_files(dataset_dir)

    train_table = pq.read_table(source_files["train"], columns=["id", "emb"])
    test_table = pq.read_table(source_files["test"], columns=["id", "emb"])

    dataset_rows = train_table.num_rows
    validate_top_k_against_rows(top_k=1, dataset_rows=dataset_rows)

    if query_count > test_table.num_rows:
        raise ValueError(
            f"query_count must be <= available test rows, got query_count={query_count}, "
            f"test_rows={test_table.num_rows}"
        )

    query_slice = test_table.slice(0, query_count)
    return {
        "dataset_rows": dataset_rows,
        "id_field": "id",
        "vector_field": "emb",
        "neighbors_field": "neighbors_id",
        "query_ids": query_slice["id"].to_pylist(),
        "query_vectors": query_slice["emb"].to_pylist(),
    }


def to_fixed_size_vector_array(vector_array: pa.Array | pa.ChunkedArray) -> pa.FixedSizeListArray:
    if isinstance(vector_array, pa.ChunkedArray):
        vector_array = vector_array.combine_chunks()

    offsets = vector_array.offsets.to_numpy(zero_copy_only=False)
    list_sizes = offsets[1:] - offsets[:-1]
    if len(list_sizes) == 0:
        raise ValueError("vector column must not be empty")

    list_size = int(list_sizes[0])
    if list_size < 1:
        raise ValueError(f"vector dimension must be >= 1, got dimension={list_size}")
    if not (list_sizes == list_size).all():
        raise ValueError("vector column must have a fixed list size for every row")

    return pa.FixedSizeListArray.from_arrays(vector_array.values, list_size=list_size)


def run_query_experiment(
    dataset,
    *,
    experiment_name: str,
    query_vectors: list[list[float]],
    top_k: int,
    warmup_queries: int,
    use_index: bool,
    index_type: str | None = None,
    index_params: dict[str, object] | None = None,
) -> dict[str, object]:
    validate_positive_int("top_k", top_k)
    if warmup_queries < 0:
        raise ValueError(
            f"warmup_queries must be >= 0, got warmup_queries={warmup_queries}"
        )

    latency_samples_ms = []
    rows_returned = []
    query_result_ids = []
    for query_index, query_vector in enumerate(query_vectors):
        start = time.perf_counter()
        result_table = dataset.to_table(
            columns=["id", "_distance"],
            nearest={
                "column": "vector",
                "q": query_vector,
                "k": top_k,
                "use_index": use_index,
                **(index_params or {}),
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if query_index >= warmup_queries:
            latency_samples_ms.append(elapsed_ms)
            rows_returned.append(result_table.num_rows)
            if "id" in result_table.column_names:
                query_result_ids.append(result_table["id"].to_pylist())

    if not latency_samples_ms:
        raise ValueError(
            "timed query set is empty after warmup, "
            f"got query_count={len(query_vectors)}, warmup_queries={warmup_queries}"
        )

    result = {
        "name": experiment_name,
        "use_index": use_index,
        "latency_ms": summarize_latency_samples(latency_samples_ms),
        "average_rows_returned": sum(rows_returned) / len(rows_returned),
        "query_result_ids": query_result_ids,
    }
    if index_type is not None:
        result["index_type"] = index_type
    if index_params is not None:
        result["index_params"] = index_params
    return result


def build_result_document(
    *,
    run_label: str,
    host_name: str,
    dataset_metadata: dict[str, object],
    query_metadata: dict[str, object],
    experiments: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "run_label": run_label,
        "host": host_name,
        "dataset": dataset_metadata,
        "query": query_metadata,
        "experiments": experiments,
    }


def write_result_json(result_document: dict[str, object], result_path: str | Path) -> None:
    output_path = Path(result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_document, indent=2, sort_keys=True) + "\n")


def compute_recall_at_k(
    *,
    query_results: list[list[int]],
    ground_truth: list[list[int]],
    k: int,
) -> float:
    validate_positive_int("k", k)
    if len(query_results) != len(ground_truth):
        raise ValueError(
            "query_results and ground_truth must have the same length, "
            f"got query_results={len(query_results)}, ground_truth={len(ground_truth)}"
        )

    recalls = []
    for result_ids, truth_ids in zip(query_results, ground_truth):
        truth_prefix = set(truth_ids[:k])
        if not truth_prefix:
            raise ValueError("ground truth prefix must not be empty")
        result_prefix = result_ids[:k]
        overlap = sum(1 for result_id in result_prefix if result_id in truth_prefix)
        recalls.append(overlap / k)
    return sum(recalls) / len(recalls)


def load_ground_truth(dataset_dir: str | Path, *, query_count: int) -> list[list[int]]:
    source_files = validate_source_files(dataset_dir)
    neighbors_table = pq.read_table(source_files["neighbors"], columns=["neighbors_id"])
    if query_count > neighbors_table.num_rows:
        raise ValueError(
            "query_count must be <= available neighbors rows, "
            f"got query_count={query_count}, neighbors_rows={neighbors_table.num_rows}"
        )
    return neighbors_table.slice(0, query_count)["neighbors_id"].to_pylist()


def import_lance():
    import lance

    return lance


def materialize_lance_dataset(
    *,
    dataset_dir: str | Path,
    lance_uri: str | Path,
) -> tuple[object, float]:
    lance = import_lance()
    dataset_path = Path(lance_uri)

    if dataset_path.exists():
        return lance.dataset(dataset_path), 0.0

    source_info = inspect_source_dataset(dataset_dir, query_count=1)
    source_files = validate_source_files(dataset_dir)
    train_table = pq.read_table(
        source_files["train"],
        columns=[source_info["id_field"], source_info["vector_field"]],
    )
    lance_table = pa.table(
        {
            "id": train_table[source_info["id_field"]],
            "vector": to_fixed_size_vector_array(train_table[source_info["vector_field"]]),
        }
    )

    start = time.perf_counter()
    dataset = lance.write_dataset(lance_table, dataset_path, mode="overwrite")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return dataset, elapsed_ms


def ensure_vector_index(
    dataset,
    *,
    index_type: str,
    num_partitions: int,
    num_bits: int,
    metric_type: str,
    num_sub_vectors: int | None = None,
) -> float:
    existing_indices = dataset.describe_indices()
    if existing_indices:
        return 0.0

    start = time.perf_counter()
    create_index_kwargs = {
        "column": "vector",
        "index_type": index_type,
        "metric": metric_type,
        "num_partitions": num_partitions,
        "num_bits": num_bits,
    }
    if index_type == "IVF_PQ":
        if num_sub_vectors is None:
            raise ValueError("num_sub_vectors is required for IVF_PQ")
        create_index_kwargs["num_sub_vectors"] = num_sub_vectors
    dataset.create_index(
        **create_index_kwargs,
    )
    return (time.perf_counter() - start) * 1000.0


def finalize_experiment(
    experiment: dict[str, object],
    *,
    ground_truth: list[list[int]] | None = None,
    recall_k: int | None = None,
    warmup_queries: int = 0,
) -> dict[str, object]:
    finalized = dict(experiment)
    query_result_ids = finalized.pop("query_result_ids", [])
    if ground_truth is not None and recall_k is not None:
        finalized[f"recall_at_{recall_k}"] = compute_recall_at_k(
            query_results=query_result_ids,
            ground_truth=ground_truth[warmup_queries:],
            k=recall_k,
        )
    return finalized


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    source_info = inspect_source_dataset(args.dataset_dir, query_count=args.query_count)
    validate_top_k_against_rows(top_k=args.top_k, dataset_rows=source_info["dataset_rows"])
    ground_truth = load_ground_truth(args.dataset_dir, query_count=args.query_count)

    dataset, dataset_build_ms = materialize_lance_dataset(
        dataset_dir=args.dataset_dir,
        lance_uri=args.lance_uri,
    )
    index_build_ms = ensure_vector_index(
        dataset,
        index_type=args.index_type,
        num_partitions=args.num_partitions,
        num_bits=args.num_bits,
        metric_type=args.metric,
        num_sub_vectors=args.num_sub_vectors if args.index_type == "IVF_PQ" else None,
    )

    experiments = []
    if not args.skip_flat:
        flat_result = run_query_experiment(
            dataset,
            experiment_name="flat",
            query_vectors=source_info["query_vectors"],
            top_k=args.top_k,
            warmup_queries=args.warmup_queries,
            use_index=False,
        )
        experiments.append(finalize_experiment(flat_result))

    for nprobes in parse_nprobes(args.nprobes):
        for refine_factor in parse_refine_factors(args.refine_factor):
            ann_result = run_query_experiment(
                dataset,
                experiment_name=build_ann_experiment_name(
                    args.index_type, nprobes=nprobes, refine_factor=refine_factor
                ),
                query_vectors=source_info["query_vectors"],
                top_k=args.top_k,
                warmup_queries=args.warmup_queries,
                use_index=True,
                index_type=args.index_type,
                index_params={
                    "metric": args.metric,
                    "nprobes": nprobes,
                    "refine_factor": refine_factor,
                },
            )
            experiments.append(
                finalize_experiment(
                    ann_result,
                    ground_truth=ground_truth,
                    recall_k=min(args.recall_k, args.top_k, len(ground_truth[0])),
                    warmup_queries=args.warmup_queries,
                )
            )

    return build_result_document(
        run_label=args.run_label,
        host_name=socket.gethostname(),
        dataset_metadata={
            "name": "cohere_medium_1m",
            "source_dir": str(args.dataset_dir),
            "lance_uri": str(args.lance_uri),
            "dataset_rows": source_info["dataset_rows"],
            "dim": args.dim,
            "metric": args.metric,
            "dataset_build_ms": dataset_build_ms,
            "index_build_ms": index_build_ms,
        },
        query_metadata={
            "query_count": args.query_count,
            "top_k": args.top_k,
            "warmup_queries": args.warmup_queries,
        },
        experiments=experiments,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark large-topK search on Cohere 1M")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/work/vectordb-bench/dataset/cohere/cohere_medium_1m"),
    )
    parser.add_argument(
        "--lance-uri",
        type=Path,
        default=Path("/data/work/lance-cohere1m/cohere1m.lance"),
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("/data/work/lance-bench-results/cohere1m-topk100k.json"),
    )
    parser.add_argument("--run-label", default="cohere1m-topk100k")
    parser.add_argument("--index-type", choices=["IVF_PQ", "IVF_RQ"], default="IVF_PQ")
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=100_000)
    parser.add_argument("--skip-flat", action="store_true")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--nprobes", default="8,32,128")
    parser.add_argument("--refine-factor", default="1")
    parser.add_argument("--num-partitions", type=int, default=256)
    parser.add_argument("--num-sub-vectors", type=int, default=96)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--recall-k", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_positive_int("query_count", args.query_count)
    validate_positive_int("warmup_queries", args.warmup_queries)
    validate_positive_int("top_k", args.top_k)
    validate_positive_int("num_partitions", args.num_partitions)
    validate_positive_int("num_sub_vectors", args.num_sub_vectors)
    validate_positive_int("num_bits", args.num_bits)
    validate_positive_int("recall_k", args.recall_k)

    result_document = run_benchmark(args)
    write_result_json(result_document, args.result_path)


if __name__ == "__main__":
    main()
