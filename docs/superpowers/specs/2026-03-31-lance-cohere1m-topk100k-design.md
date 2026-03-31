# Lance Cohere 1M TopK=100K Latency Design

**Goal:** Add a reusable benchmark workflow in the Lance repo to measure large-topK vector search latency on a remote x86 server using the existing Cohere 1M dataset already staged under `/data/work`.

**Non-goals:**
- Build a new general-purpose benchmark framework.
- Reformat or replace the existing `benchmarks/` structure.
- Add broad parameter sweeps or multiple datasets in the first slice.

## Context

The target environment already has:

- a remote x86 host accessible as `hannsdb-x86`
- the Cohere 1M dataset staged at `/data/work/vectordb-bench/dataset/cohere/cohere_medium_1m`
- an existing benchmark workflow style in HannsDB that favors:
  - a thin Python entrypoint for the actual measurement logic
  - a thin shell wrapper for remote execution, env vars, and result paths

The dataset shape is compatible with a thin Lance-side adapter:

- `shuffle_train.parquet`
- `test.parquet`
- `neighbors.parquet`
- `scalar_labels.parquet`

The existing VectorDBBench dataset definition confirms:

- case: `Performance768D1M`
- dimension: `768`
- metric: `cosine`

## User-Approved Scope

The benchmark should:

- run on the existing x86 environment under `/data/work`
- reuse the staged Cohere 1M parquet dataset
- avoid introducing a new benchmark framework
- compare both:
  - flat brute-force search as a baseline
  - ANN search as the primary path
- focus on large-topK latency with `topK=100000`

## Recommended Approach

Implement a very small, self-contained benchmark entry under `benchmarks/` and a shell wrapper for x86 execution.

This is intentionally not integrated into a broader benchmark harness. The main objective is reproducible measurement with minimal glue and minimal repo disruption.

## Architecture

There are two layers:

1. A Python benchmark script inside the Lance repo.
2. A shell wrapper that runs it in the x86 environment with absolute paths and stable output locations.

The Python script is responsible for:

- reading the staged parquet dataset
- materializing a Lance dataset if it does not already exist
- creating an ANN index if it does not already exist
- running query measurements for both flat and ANN modes
- writing a structured JSON result

The shell wrapper is responsible for:

- selecting paths under `/data/work`
- activating the appropriate Python environment
- passing env vars and CLI args
- making repeatable remote execution easy

## Data Flow

### Input data

Source directory:

- `/data/work/vectordb-bench/dataset/cohere/cohere_medium_1m`

Required files:

- `shuffle_train.parquet`
- `test.parquet`
- `neighbors.parquet`

The initial implementation assumes these files exist and errors clearly if they do not.

### Lance dataset materialization

The benchmark script creates a Lance dataset at a stable x86-local path, for example:

- `/data/work/lance-cohere1m/cohere1m.lance`

If this dataset already exists, it is reused.

The materialized dataset should contain:

- an id column
- a vector column using the embedding field from the parquet source

The exact source field names should be resolved from the Cohere parquet schema in code and validated early with descriptive errors.

### Query input

Queries come from `test.parquet`.

The benchmark uses the first `QUERY_COUNT` queries, where the default is `100`.

Warmup queries are run before timed measurements and are excluded from reported latency statistics.

## Query Modes

### Flat baseline

Flat baseline uses:

- `use_index=False`
- `topK=100000`

Purpose:

- quantify the cost of exact search plus large result materialization
- provide a direct comparison point for ANN runs

### ANN primary path

The first slice uses `IVF_PQ` only.

Initial fixed index configuration:

- `metric_type=cosine`
- `num_partitions=256`
- `num_sub_vectors=96`
- `num_bits=8`

Initial query-time sweep:

- `nprobes=8`
- `nprobes=32`
- `nprobes=128`

Purpose:

- measure how latency scales as the ANN path becomes more exhaustive
- avoid conflating the first result set with a large configuration sweep

### Refine behavior

`refine_factor` is intentionally excluded from the first slice.

Reason:

- with `topK=100000`, refine can amplify random access and result materialization costs
- the first useful answer is whether large-topK latency is dominated by flat work or by ANN candidate expansion

If ANN recall is clearly insufficient in the first results, a follow-up slice can add a single `refine_factor=2` measurement.

## Measurement Method

Measurements are taken per query, not per batch.

This is important because large-topK workloads tend to have meaningful tail latency, and batching can hide that.

Per experiment:

1. Run a small warmup.
2. Time each query independently.
3. Record the number of rows returned.
4. Compute summary statistics after the run.

Reported latency fields:

- `mean`
- `p50`
- `p95`
- `p99`
- `max`

## Result Validation

### Functional validation

Every query run must verify:

- the query completes successfully
- the returned table is non-empty
- the returned row count is recorded

### ANN quality validation

The benchmark should compute a lightweight recall signal using `neighbors.parquet`.

The initial quality metric is:

- `recall@100`

This avoids depending on a `recall@100000` ground-truth interpretation while still ensuring the ANN path is not silently broken.

Flat search acts as the correctness baseline for behavior, but recall should be computed against the provided ground truth where possible.

## Output

Each run writes one JSON file to a stable directory, for example:

- `/data/work/lance-bench-results`

The JSON must include:

- run label
- host name
- source dataset path
- Lance dataset path
- dataset row count
- vector dimension
- metric
- query count
- warmup count
- topK
- experiment list

Each experiment entry must include:

- experiment name
- whether index was used
- index type if applicable
- index parameters if applicable
- latency summary
- average returned rows
- recall metric if applicable

If a run fails after startup, the JSON should include an `error` section when practical, otherwise the script should fail loudly with context-rich stderr.

## Error Handling

The benchmark should reject invalid states early:

- missing dataset files
- `topK > dataset_rows`
- invalid `query_count`
- invalid `nprobes`
- unexpected vector column shape or type

Errors must include the relevant variable names and values.

## File Layout

Proposed files:

- `benchmarks/cohere/benchmark_topk_latency.py`
- `benchmarks/cohere/run_cohere1m_topk_latency_x86.sh`

No additional framework-level refactor is part of this slice.

## Testing Strategy

This change still needs tests because the repo requires tests for features and bugfixes.

The tests should focus on the benchmark adapter logic rather than full 1M execution:

- path validation
- result summarization
- parameter parsing and validation
- small local smoke path using tiny synthetic vectors or a tiny temporary Lance dataset

The real x86 run is the environment verification, not the only verification.

## Risks

### Risk: large-topK dominates output materialization

This is expected and is part of what the benchmark is trying to capture.

Mitigation:

- always compare ANN against flat
- report returned row counts

### Risk: first run mixes build cost and query cost

Mitigation:

- separate dataset creation/index creation timing from query timing
- reuse existing dataset/index on subsequent runs

### Risk: source parquet schema differs from expectations

Mitigation:

- validate source columns explicitly and fail with descriptive errors

## Success Criteria

The first slice is successful if:

- a single command on `hannsdb-x86` can run the benchmark
- the benchmark reuses `/data/work` data without downloading
- flat and ANN runs both complete for `topK=100000`
- the output JSON contains stable latency summaries
- the workflow is easy to rerun for additional measurements
