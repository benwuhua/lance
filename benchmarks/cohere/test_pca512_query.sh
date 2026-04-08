#!/bin/bash
# test_pca512_query.sh - PCA-512 DRAM benchmark sweep (run after shard-2 is done)
# Usage: nohup bash /data/work/tmp/test_pca512_query.sh > /data/work/tmp/pca512_query.log 2>&1 &
set -e

cd /data/work/lance
source /data/work/venvs/lance-benchmark/bin/activate
export TMPDIR=/data/work/tmp/lance-tmp
export LANCE_IO_THREADS=128

PCA_DIR=/data/work/tmp/s1b-pca512
RESULT_DIR=/data/work/tmp/pca512-results
BENCH="benchmarks/cohere/bench.py"
GT_PATH=$PCA_DIR/gt_positional.npz

mkdir -p $RESULT_DIR

echo "================================================================"
echo "  PCA-512 DRAM Benchmark Sweep"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# Verify all shards ready
for i in 0 1 2 3 4; do
    rows=$(python3 -c "import lance; print(lance.dataset('$PCA_DIR/shard-${i}.lance').count_rows())")
    has_idx=$(python3 -c "import lance; print(bool(lance.dataset('$PCA_DIR/shard-${i}.lance').list_indices()))")
    echo "  shard-$i: $rows rows, index=$has_idx"
    if [ "$has_idx" = "False" ]; then
        echo "ERROR: shard-$i missing index!"
        exit 1
    fi
done

echo ""
echo "GT: $GT_PATH"
python3 -c "
import numpy as np
gt = np.load('$GT_PATH')
print(f'  {len(gt.files)} queries, k={np.load(\"$GT_PATH\")[gt.files[0]].shape[-1]}')
"

# Warm page cache
echo ""
echo "=== Warming page cache ==="
python3 -u -c "
import sys; sys.path.insert(0, '/data/work/lance')
from benchmarks.cohere.bench import warm_page_cache
import time; t0 = time.time()
warm_page_cache('$PCA_DIR', 5)
print(f'Warm done in {time.time()-t0:.0f}s')
"

# Run sweep
run_query() {
    local np=$1 rf=$2
    local label="pca512-dram-np${np}-rf${rf}"
    echo ""
    echo "  Running: $label"
    python -u $BENCH query \
        --plan B --storage dram --index-type IVF_RQ \
        --shard-dir $PCA_DIR \
        --nprobes $np --refine-factor $rf --top-k 10000 \
        --num-shards 5 --query-count 8 --warmup 3 --seed 42 \
        --column vector --metric cosine \
        --gt-path $GT_PATH \
        --result-path $RESULT_DIR/${label}.json \
        --no-warm
}

echo ""
echo "=== DRAM Pareto Sweep ==="
run_query 128  1
run_query 256  1
run_query 512  1
run_query 1024 1
run_query 128  2
run_query 256  2
run_query 512  2
run_query 1024 2
run_query 1024 3
run_query 1024 5

echo ""
echo "================================================================"
echo "  ALL TESTS COMPLETE - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo "Results in: $RESULT_DIR/"
ls -la $RESULT_DIR/
