#!/bin/bash
# test_pca512_sq.sh - PCA-512 + IVF_SQ DRAM + OBS sweep
set -e

cd /data/work/lance
source /data/work/venvs/lance-benchmark/bin/activate
export TMPDIR=/data/work/tmp/lance-tmp
export LANCE_IO_THREADS=128

SQ_DIR=/data/work/tmp/s1b-pca512-sq
GT=/data/work/tmp/s1b-pca512/gt_positional_v2.npz
RESULT_DIR=/data/work/tmp/pca512-sq-results
BENCH="benchmarks/cohere/bench.py"

mkdir -p $RESULT_DIR

echo "================================================================"
echo "  PCA-512 + IVF_SQ DRAM + OBS Sweep"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo "SQ shards: $SQ_DIR"
echo "GT: $GT"

# === Part 1: DRAM sweep ===
echo ""
echo "=== DRAM Pareto Sweep ==="

run_dram() {
    local np=$1 rf=$2
    local label="pca512-sq-dram-np${np}-rf${rf}"
    echo ""
    echo "  Running: $label"
    python -u $BENCH query \
        --plan B --storage dram --index-type IVF_SQ \
        --shard-dir $SQ_DIR \
        --nprobes $np --refine-factor $rf --top-k 10000 \
        --num-shards 5 --query-count 8 --warmup 3 --seed 42 \
        --column vector --metric cosine \
        --gt-path $GT \
        --result-path $RESULT_DIR/${label}.json \
        --no-warm
}

run_dram 128  1
run_dram 256  1
run_dram 512  1
run_dram 1024 1
run_dram 128  2
run_dram 256  2
run_dram 512  2
run_dram 1024 2
run_dram 1024 3
run_dram 1024 5

echo ""
echo "=== DRAM sweep complete ==="

# === Part 2: OBS sweep ===
echo ""
echo "=== OBS Sweep ==="

# Clear AWS env vars to avoid duplicate field error
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_ACCESS_KEY
unset AWS_SECRET_KEY

OBS_SHARD_DIR=s3://knowledgebase-5f43/pca512-sq-1b

run_obs() {
    local np=$1 rf=$2
    local label="pca512-sq-obs-np${np}-rf${rf}"
    echo ""
    echo "  Running: $label"
    python -u $BENCH query \
        --plan B --storage obs --index-type IVF_SQ \
        --shard-dir $OBS_SHARD_DIR \
        --nprobes $np --refine-factor $rf --top-k 10000 \
        --num-shards 5 --query-count 8 --warmup 3 --seed 42 \
        --column vector --metric cosine \
        --access-key "0JIMBMDMUBP8UUPHWJ44" \
        --secret-key "J2V4tESvsdNoKUkkrdDJHpQvt9VwqV6dUUcKnwps" \
        --result-path $RESULT_DIR/${label}.json
}

run_obs 128  1
run_obs 256  1
run_obs 512  1
run_obs 1024 1
run_obs 128  2
run_obs 256  2
run_obs 512  2
run_obs 1024 2

echo ""
echo "================================================================"
echo "  ALL TESTS COMPLETE - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo "Results in: $RESULT_DIR/"
ls -la $RESULT_DIR/
