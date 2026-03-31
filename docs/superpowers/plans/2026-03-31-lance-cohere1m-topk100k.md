# Lance Cohere 1M TopK=100K Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable, x86-friendly Lance benchmark workflow that measures Cohere 1M large-topK vector search latency for flat and ANN search using the staged `/data/work` parquet dataset.

**Architecture:** A small Python benchmark script handles dataset adaptation, optional index creation, query timing, and JSON output. A small shell wrapper provides stable x86 execution with absolute `/data/work` paths and environment-variable configuration.

**Tech Stack:** Python, Lance Python API, PyArrow/Parquet, shell, existing `benchmarks/` layout

---

## File Map

- Create: `benchmarks/cohere/benchmark_topk_latency.py`
- Create: `benchmarks/cohere/run_cohere1m_topk_latency_x86.sh`
- Create: `python/python/tests/benchmarks/test_cohere_topk_latency.py`
- Modify: `benchmarks/cohere/README.md` if needed to document usage

## Chunk 1: Benchmark Script Skeleton

### Task 1: Create the benchmark module skeleton

**Files:**
- Create: `benchmarks/cohere/benchmark_topk_latency.py`
- Test: `python/python/tests/benchmarks/test_cohere_topk_latency.py`

- [ ] **Step 1: Write the failing test for argument validation and stat summarization**

Add tests that exercise small pure functions such as:
- parsing comma-separated `nprobes`
- validating `top_k`, `query_count`, and `warmup_queries`
- summarizing a list of latency samples into `mean/p50/p95/p99/max`

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -v`
Expected: FAIL because the new module and helpers do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:
- a CLI parser
- helper functions for validation
- helper functions for percentile/stat calculation
- result dataclass or dict builder for experiment summaries

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -v`
Expected: PASS for the pure helper tests.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py
git commit -m "feat: add benchmark helpers for cohere topk latency"
```

## Chunk 2: Source Dataset Adapter

### Task 2: Add parquet source validation and query loading

**Files:**
- Modify: `benchmarks/cohere/benchmark_topk_latency.py`
- Modify: `python/python/tests/benchmarks/test_cohere_topk_latency.py`

- [ ] **Step 1: Write the failing tests for source file checks**

Add tests for:
- required source files are checked
- missing file errors include the absolute path
- invalid `top_k > dataset_rows` is rejected

- [ ] **Step 2: Run the targeted tests**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "source or validate" -v`
Expected: FAIL because source validation and row-count validation are not implemented.

- [ ] **Step 3: Implement source dataset inspection**

Add logic that:
- validates `shuffle_train.parquet`, `test.parquet`, and `neighbors.parquet`
- loads query vectors from `test.parquet`
- validates row counts and query limits early

Keep the schema handling explicit and descriptive so failures show the missing field names and file paths.

- [ ] **Step 4: Run the tests again**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "source or validate" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py
git commit -m "feat: validate cohere parquet benchmark inputs"
```

## Chunk 3: Lance Dataset Materialization

### Task 3: Add reusable Lance dataset creation or reuse

**Files:**
- Modify: `benchmarks/cohere/benchmark_topk_latency.py`
- Modify: `python/python/tests/benchmarks/test_cohere_topk_latency.py`

- [ ] **Step 1: Write a failing smoke test for dataset reuse**

Add a small test using a tiny temporary dataset that verifies:
- the helper creates a Lance dataset when missing
- a second call reuses the existing Lance dataset path

- [ ] **Step 2: Run the targeted test**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "materialize or reuse" -v`
Expected: FAIL because Lance dataset materialization is not implemented.

- [ ] **Step 3: Implement dataset materialization**

Add logic that:
- reads the parquet training data
- writes a Lance dataset to the configured URI
- reopens and reuses the dataset if it already exists
- records creation timing separately from query timing

Use names that clearly reflect whether a path is source parquet or output Lance dataset.

- [ ] **Step 4: Re-run the targeted test**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "materialize or reuse" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py
git commit -m "feat: add cohere parquet to lance dataset adapter"
```

## Chunk 4: IVF_PQ Index Management

### Task 4: Add configurable IVF_PQ creation and reuse

**Files:**
- Modify: `benchmarks/cohere/benchmark_topk_latency.py`
- Modify: `python/python/tests/benchmarks/test_cohere_topk_latency.py`

- [ ] **Step 1: Write the failing test for index parameter validation**

Add tests for:
- invalid `nprobes` values
- invalid partition or PQ params
- experiment config objects that serialize the expected index metadata

- [ ] **Step 2: Run the targeted tests**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "index or nprobes" -v`
Expected: FAIL because index config validation is incomplete.

- [ ] **Step 3: Implement IVF_PQ management**

Add logic that:
- creates the `IVF_PQ` index when needed
- reuses the existing index when possible
- records index build timing separately
- stores the exact index parameters in the output JSON

Initial fixed defaults:
- `metric_type=cosine`
- `num_partitions=256`
- `num_sub_vectors=96`
- `num_bits=8`

- [ ] **Step 4: Re-run the targeted tests**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "index or nprobes" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py
git commit -m "feat: add ivf pq setup for cohere latency benchmark"
```

## Chunk 5: Flat and ANN Query Timing

### Task 5: Implement per-query timing for flat baseline

**Files:**
- Modify: `benchmarks/cohere/benchmark_topk_latency.py`
- Modify: `python/python/tests/benchmarks/test_cohere_topk_latency.py`

- [ ] **Step 1: Write the failing smoke test for flat experiment execution**

Add a tiny test dataset and verify:
- warmup queries are excluded from timing stats
- each query produces a non-empty result
- the returned result includes latency summary and average returned rows

- [ ] **Step 2: Run the targeted test**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "flat experiment" -v`
Expected: FAIL because experiment execution is not implemented.

- [ ] **Step 3: Implement flat per-query timing**

Use `use_index=False` and time each query individually.

Record:
- latency samples
- average returned rows
- experiment metadata

- [ ] **Step 4: Re-run the targeted test**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "flat experiment" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py
git commit -m "feat: add flat cohere topk latency measurement"
```

### Task 6: Implement ANN per-query timing for `nprobes=8,32,128`

**Files:**
- Modify: `benchmarks/cohere/benchmark_topk_latency.py`
- Modify: `python/python/tests/benchmarks/test_cohere_topk_latency.py`

- [ ] **Step 1: Write the failing smoke test for ANN experiment execution**

Add a tiny indexed dataset test that verifies:
- ANN experiments run with explicit `nprobes`
- the experiment result includes the right `index_type` and index params
- latency summaries are emitted for each `nprobes` value

- [ ] **Step 2: Run the targeted test**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "ann experiment" -v`
Expected: FAIL because ANN timing is not implemented.

- [ ] **Step 3: Implement ANN per-query timing**

Use:
- `use_index=True`
- `nprobes` sweep over `8,32,128`
- no `refine_factor` in the first version

Return one experiment object per `nprobes` value.

- [ ] **Step 4: Re-run the targeted test**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "ann experiment" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py
git commit -m "feat: add ann cohere topk latency measurement"
```

## Chunk 6: Recall and Result JSON

### Task 7: Add lightweight recall and final JSON output

**Files:**
- Modify: `benchmarks/cohere/benchmark_topk_latency.py`
- Modify: `python/python/tests/benchmarks/test_cohere_topk_latency.py`

- [ ] **Step 1: Write failing tests for JSON result shape**

Add tests that verify:
- top-level metadata fields exist
- experiment entries include latency summary
- ANN entries include recall metadata when ground truth is available

- [ ] **Step 2: Run the targeted tests**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "json or recall" -v`
Expected: FAIL because JSON assembly and recall wiring are incomplete.

- [ ] **Step 3: Implement result assembly**

Add:
- top-level run metadata
- per-experiment metadata
- optional `recall_at_100`
- JSON write to the configured result path

Make the output stable and easy to diff between runs.

- [ ] **Step 4: Re-run the targeted tests**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -k "json or recall" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py
git commit -m "feat: write structured cohere topk benchmark results"
```

## Chunk 7: x86 Wrapper and Documentation

### Task 8: Add the x86 shell wrapper

**Files:**
- Create: `benchmarks/cohere/run_cohere1m_topk_latency_x86.sh`
- Modify: `benchmarks/cohere/README.md`

- [ ] **Step 1: Write the wrapper with explicit `/data/work` defaults**

Include env-configurable defaults for:
- source dataset directory
- Lance output URI
- result directory
- query count
- topK
- warmup queries
- nprobes

- [ ] **Step 2: Add usage documentation**

Document:
- required remote files
- example command
- where results are written

- [ ] **Step 3: Run a shell syntax check**

Run: `bash -n benchmarks/cohere/run_cohere1m_topk_latency_x86.sh`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/cohere/run_cohere1m_topk_latency_x86.sh benchmarks/cohere/README.md
git commit -m "feat: add x86 wrapper for cohere topk latency benchmark"
```

## Chunk 8: Verification

### Task 9: Run local verification for the new benchmark helpers

**Files:**
- No code changes unless failures require fixes

- [ ] **Step 1: Run the benchmark test file**

Run: `pytest python/python/tests/benchmarks/test_cohere_topk_latency.py -v`
Expected: PASS.

- [ ] **Step 2: Run formatting if needed**

Run: `python -m compileall benchmarks/cohere/benchmark_topk_latency.py`
Expected: successful bytecode compilation.

- [ ] **Step 3: Perform a tiny smoke invocation locally**

Run the benchmark script against a tiny temporary synthetic dataset or a small temporary parquet fixture used by the tests.

Expected:
- JSON output is created
- flat and ANN experiment entries are both present

- [ ] **Step 4: Commit any required verification fixes**

```bash
git add benchmarks/cohere/benchmark_topk_latency.py python/python/tests/benchmarks/test_cohere_topk_latency.py benchmarks/cohere/run_cohere1m_topk_latency_x86.sh benchmarks/cohere/README.md
git commit -m "test: verify cohere topk latency benchmark workflow"
```

## Chunk 9: Remote x86 Validation

### Task 10: Run the benchmark on `hannsdb-x86`

**Files:**
- No required file changes unless remote issues expose a real bug

- [ ] **Step 1: Sync the Lance repo to the remote machine**

Use the established remote sync approach already used for HannsDB, adapted for the Lance repo path on x86.

- [ ] **Step 2: Run the wrapper on x86**

Example:

```bash
QUERY_COUNT=100 TOPK=100000 bash benchmarks/cohere/run_cohere1m_topk_latency_x86.sh
```

Expected:
- the benchmark reuses `/data/work/vectordb-bench/dataset/cohere/cohere_medium_1m`
- it creates or reuses the Lance dataset
- it emits one JSON result file under the configured result directory

- [ ] **Step 3: Inspect the JSON result**

Verify:
- one flat experiment exists
- three ANN experiments exist for `8,32,128`
- latency summaries are populated

- [ ] **Step 4: If needed, fix and re-run**

Any remote-only bug should get:
- a local regression test where practical
- the smallest possible code fix

---

Plan complete and saved to `docs/superpowers/plans/2026-03-31-lance-cohere1m-topk100k.md`. Ready to execute?
