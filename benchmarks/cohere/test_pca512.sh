#!/bin/bash
# test_pca512.sh - PCA-512 + RaBitQ benchmark on 1B dataset
# Usage: nohup bash /data/work/tmp/test_pca512.sh > /data/work/tmp/test_pca512.log 2>&1 &
set -e

cd /data/work/lance
source /data/work/venvs/lance-benchmark/bin/activate
export TMPDIR=/data/work/tmp/lance-tmp
export LANCE_IO_THREADS=128

SOURCE_DIR=/data/work/tmp/s1b
PCA_DIR=/data/work/tmp/s1b-pca512
RESULT_DIR=/data/work/tmp/pca512-results
BENCH="benchmarks/cohere/bench.py"

mkdir -p $RESULT_DIR

echo "================================================================"
echo "  PCA-512 + RaBitQ Benchmark (1B rows, 5 shards)"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# Step 1: Install sklearn if needed
python3 -c "from sklearn.decomposition import TruncatedSVD" 2>/dev/null || {
    echo "Installing scikit-learn..."
    pip install scikit-learn -q
}

# Step 2: Build PCA shards + convert GT
echo ""
echo "=== Building PCA-512 shards ==="
python -u benchmarks/cohere/setup_pca.py \
    --source-dir $SOURCE_DIR \
    --output-dir $PCA_DIR \
    --pca-dim 512 --num-shards 5 --num-partitions 13107 \
    --sample-size 500000 --batch-size 2000000

echo ""
echo "=== Converting GT to positional indices ==="
python -u benchmarks/cohere/setup_pca.py \
    --convert-gt /data/work/tmp/gt_1b_shard0_top10k_5q_seed42.npz \
    --source-dir $SOURCE_DIR \
    --output-dir $PCA_DIR

GT_PATH=$PCA_DIR/gt_positional.npz

# Step 3: Warm page cache for PCA shards
echo ""
echo "=== Warming page cache for PCA shards ==="
python3 -u -c "
import sys; sys.path.insert(0, '/data/work/lance')
from benchmarks.cohere.bench import warm_page_cache
import time; t0 = time.time()
warm_page_cache('$PCA_DIR', 5)
print(f'Warm done in {time.time()-t0:.0f}s')
"

# Step 4: Run DRAM sweep with recall
echo ""
echo "=== DRAM Pareto Sweep (PCA-512) ==="

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

# Sweep: same configs as baseline for comparison
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
