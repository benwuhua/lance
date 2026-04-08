# Cohere 1M Large TopK Benchmark

This benchmark measures large-topK vector search latency on the staged Cohere 1M dataset.

## Dataset

The default dataset location matches the existing x86 benchmark environment:

```bash
/data/work/vectordb-bench/dataset/cohere/cohere_medium_1m
```

Expected files:

- `shuffle_train.parquet`
- `test.parquet`
- `neighbors.parquet`

## Default Benchmark

The benchmark compares:

- flat search with `use_index=False`
- `IVF_PQ` search with `nprobes=8,32,128`

Default query settings:

- `topK=100000`
- `query_count=100`
- `warmup_queries=5`

## Run On x86

```bash
bash benchmarks/cohere/run_cohere1m_topk_latency_x86.sh
```

Useful overrides:

```bash
QUERY_COUNT=1000 TOPK=100000 NPROBES=16,64,128 bash benchmarks/cohere/run_cohere1m_topk_latency_x86.sh
```

## Output

By default the wrapper writes results to:

```bash
/data/work/lance-bench-results/cohere1m-topk100k.json
```

The JSON includes:

- dataset metadata
- query metadata
- flat latency summary
- ANN latency summaries
- `recall_at_100` for ANN runs

## FineWeb Edu 1B

The staged FineWeb Edu flow supports two layers:

- raw embedding shards in OBS under `fineweb-edu-emb-raw/emb/`
- Lance datasets built from those raw shards

To build the dataset in resumable batches, use:

```bash
bash benchmarks/cohere/run_fineweb_edu_1b_batches.sh
```

Useful overrides:

```bash
START_SHARD=0 MAX_SHARDS=100 MODE=overwrite bash benchmarks/cohere/run_fineweb_edu_1b_batches.sh
START_SHARD=100 MAX_SHARDS=100 MODE=append bash benchmarks/cohere/run_fineweb_edu_1b_batches.sh
```

Each batch writes a separate log under:

```bash
/data/work/logs/fineweb-edu-1b-batch-<START_SHARD>.log
```
