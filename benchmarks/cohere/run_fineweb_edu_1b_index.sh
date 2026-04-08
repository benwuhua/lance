#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build IVF-PQ index on FineWeb-Edu ~1B dataset on OBS, then run top-K benchmark.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/work/venvs/lance-benchmark/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_URI="${DATASET_URI:-s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance}"
ENDPOINT="${ENDPOINT:-https://obs.ap-southeast-1.myhuaweicloud.com}"
REGION="${REGION:-ap-southeast-1}"
ACCESS_KEY="${ACCESS_KEY:-${AWS_ACCESS_KEY_ID:-}}"
SECRET_KEY="${SECRET_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
LOG_DIR="${LOG_DIR:-/data/work/logs}"
RESULT_DIR="${RESULT_DIR:-/data/work/lance-bench-results}"

# Index parameters
NUM_PARTITIONS="${NUM_PARTITIONS:-}"
NUM_SUB_VECTORS="${NUM_SUB_VECTORS:-64}"
NUM_BITS="${NUM_BITS:-8}"
METRIC="${METRIC:-cosine}"
MODE="${MODE:-simple}"  # simple or distributed
REPLACE="${REPLACE:-}"

# Benchmark parameters
TOP_K="${TOP_K:-10000}"
QUERY_COUNT="${QUERY_COUNT:-100}"
WARMUP="${WARMUP:-5}"
NPROBES="${NPROBES:-32,64,128,256}"
REFINE_FACTOR="${REFINE_FACTOR:-5,10,20}"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

if [[ -z "${ACCESS_KEY}" || -z "${SECRET_KEY}" ]]; then
  echo "ACCESS_KEY/SECRET_KEY are required" >&2
  exit 1
fi

INDEX_ARGS=(
  --dataset-uri "${DATASET_URI}"
  --endpoint "${ENDPOINT}"
  --region "${REGION}"
  --access-key "${ACCESS_KEY}"
  --secret-key "${SECRET_KEY}"
  --metric "${METRIC}"
  --num-sub-vectors "${NUM_SUB_VECTORS}"
  --num-bits "${NUM_BITS}"
  --mode "${MODE}"
)
[[ -n "${NUM_PARTITIONS}" ]] && INDEX_ARGS+=(--num-partitions "${NUM_PARTITIONS}")
[[ -n "${REPLACE}" ]] && INDEX_ARGS+=(--replace)

echo "=== Building index ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_index_1b.py" "${INDEX_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/fineweb-edu-1b-build-index.log"

echo ""
echo "=== Running top-K benchmark ==="
GT_PATH="${RESULT_DIR}/fineweb-edu-1b-gt-top${TOP_K}.npz"

"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_topk_1b.py" \
  --dataset-uri "${DATASET_URI}" \
  --endpoint "${ENDPOINT}" \
  --region "${REGION}" \
  --access-key "${ACCESS_KEY}" \
  --secret-key "${SECRET_KEY}" \
  --metric "${METRIC}" \
  --top-k "${TOP_K}" \
  --query-count "${QUERY_COUNT}" \
  --warmup-queries "${WARMUP}" \
  --nprobes "${NPROBES}" \
  --refine-factor "${REFINE_FACTOR}" \
  --ground-truth-path "${GT_PATH}" \
  --result-path "${RESULT_DIR}/fineweb-edu-1b-topk.json" \
  2>&1 | tee "${LOG_DIR}/fineweb-edu-1b-benchmark.log"

echo ""
echo "Done. Results at ${RESULT_DIR}/fineweb-edu-1b-topk.json"
