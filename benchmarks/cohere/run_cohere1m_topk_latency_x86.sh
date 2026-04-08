#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${DATASET_DIR:-/data/work/vectordb-bench/dataset/cohere/cohere_medium_1m}"
LANCE_URI="${LANCE_URI:-/data/work/lance-cohere1m/cohere1m.lance}"
RESULT_PATH="${RESULT_PATH:-/data/work/lance-bench-results/cohere1m-topk100k.json}"
RUN_LABEL="${RUN_LABEL:-cohere1m-topk100k}"
INDEX_TYPE="${INDEX_TYPE:-IVF_PQ}"
QUERY_COUNT="${QUERY_COUNT:-100}"
WARMUP_QUERIES="${WARMUP_QUERIES:-5}"
TOPK="${TOPK:-100000}"
SKIP_FLAT="${SKIP_FLAT:-0}"
METRIC="${METRIC:-cosine}"
DIM="${DIM:-768}"
NPROBES="${NPROBES:-8,32,128}"
REFINE_FACTOR="${REFINE_FACTOR:-1}"
NUM_PARTITIONS="${NUM_PARTITIONS:-256}"
NUM_SUB_VECTORS="${NUM_SUB_VECTORS:-96}"
NUM_BITS="${NUM_BITS:-8}"
RECALL_K="${RECALL_K:-100}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/benchmarks/cohere/benchmark_topk_latency.py"
  --dataset-dir "$DATASET_DIR" \
  --lance-uri "$LANCE_URI" \
  --result-path "$RESULT_PATH" \
  --run-label "$RUN_LABEL" \
  --index-type "$INDEX_TYPE" \
  --query-count "$QUERY_COUNT" \
  --warmup-queries "$WARMUP_QUERIES" \
  --top-k "$TOPK" \
  --metric "$METRIC" \
  --dim "$DIM" \
  --nprobes "$NPROBES" \
  --refine-factor "$REFINE_FACTOR" \
  --num-partitions "$NUM_PARTITIONS" \
  --num-sub-vectors "$NUM_SUB_VECTORS" \
  --num-bits "$NUM_BITS" \
  --recall-k "$RECALL_K"
)

if [[ "$SKIP_FLAT" == "1" ]]; then
  CMD+=(--skip-flat)
fi

"${CMD[@]}"

echo "result file: $RESULT_PATH"
