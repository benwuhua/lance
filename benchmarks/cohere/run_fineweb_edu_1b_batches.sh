#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/work/venvs/lance-benchmark/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INGEST_SCRIPT="${SCRIPT_DIR}/ingest_fineweb_edu.py"

OUTPUT_URI="${OUTPUT_URI:-s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance}"
RAW_BUCKET="${RAW_BUCKET:-knowledgebase-5f43}"
RAW_ENDPOINT="${RAW_ENDPOINT:-https://obs.ap-southeast-1.myhuaweicloud.com}"
RAW_PREFIX="${RAW_PREFIX:-fineweb-edu-emb-raw/emb}"

ACCESS_KEY="${ACCESS_KEY:-${AWS_ACCESS_KEY_ID:-}}"
SECRET_KEY="${SECRET_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
ENDPOINT="${ENDPOINT:-https://obs.ap-southeast-1.myhuaweicloud.com}"
REGION="${REGION:-ap-southeast-1}"
USE_OPENDAL="${USE_OPENDAL:-true}"
VIRTUAL_HOSTED_STYLE_REQUEST="${VIRTUAL_HOSTED_STYLE_REQUEST:-true}"

START_SHARD="${START_SHARD:-0}"
MAX_SHARDS="${MAX_SHARDS:-100}"
MAX_ROWS="${MAX_ROWS:-1000000000}"
BATCH_ROWS="${BATCH_ROWS:-16384}"
MAX_ROWS_PER_GROUP="${MAX_ROWS_PER_GROUP:-8192}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-1000000}"
CACHE_DIR="${CACHE_DIR:-/data/work/tmp/fineweb-edu-batches/shard-${START_SHARD}}"
LOG_DIR="${LOG_DIR:-/data/work/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/fineweb-edu-1b-batch-${START_SHARD}.log}"
MODE="${MODE:-append}"

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

if [[ -z "${ACCESS_KEY}" || -z "${SECRET_KEY}" ]]; then
  echo "ACCESS_KEY/SECRET_KEY are required" >&2
  exit 1
fi

if [[ "${START_SHARD}" == "0" && "${MODE}" == "append" ]]; then
  MODE="overwrite"
fi

{
  echo "starting batch"
  echo "output_uri=${OUTPUT_URI}"
  echo "start_shard=${START_SHARD}"
  echo "max_shards=${MAX_SHARDS}"
  echo "mode=${MODE}"
  echo "cache_dir=${CACHE_DIR}"
  "${PYTHON_BIN}" "${INGEST_SCRIPT}" \
    --output "${OUTPUT_URI}" \
    --mode "${MODE}" \
    --max-rows "${MAX_ROWS}" \
    --start-shard "${START_SHARD}" \
    --max-shards "${MAX_SHARDS}" \
    --batch-rows "${BATCH_ROWS}" \
    --max-rows-per-group "${MAX_ROWS_PER_GROUP}" \
    --max-rows-per-file "${MAX_ROWS_PER_FILE}" \
    --cache-dir "${CACHE_DIR}" \
    --access-key "${ACCESS_KEY}" \
    --secret-key "${SECRET_KEY}" \
    --endpoint "${ENDPOINT}" \
    --region "${REGION}" \
    --virtual-hosted-style-request "${VIRTUAL_HOSTED_STYLE_REQUEST}" \
    --use-opendal "${USE_OPENDAL}" \
    --raw-bucket "${RAW_BUCKET}" \
    --raw-endpoint "${RAW_ENDPOINT}" \
    --raw-access-key "${ACCESS_KEY}" \
    --raw-secret-key "${SECRET_KEY}" \
    --raw-prefix "${RAW_PREFIX}"
  echo "batch complete"
} 2>&1 | tee -a "${LOG_FILE}"
