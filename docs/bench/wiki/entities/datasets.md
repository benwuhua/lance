---
tags: [dataset, entity]
date: 2026-04-08
---

# Datasets

## 1B FineWeb-Edu (Primary)

- **Rows**: 972,996,765 (5 shards x 194,599,353)
- **Dimensions**: 1024 (float32, L2-normalized)
- **Source**: FineWeb-Edu embeddings, replicated 3x from 324M source + Gaussian noise (sigma=0.01) + L2 re-normalize
- **Metric**: cosine
- **Local**: `/data/work/tmp/s1b/shard-{0-4}.lance`
- **OBS**: `s3://knowledgebase-5f43/fineweb-edu-1b-rq-shard4/shard-{0-4}.lance/`
- **Index**: IVF_RQ, 13,107 partitions/shard, density ~14,800 vecs/partition
- **Build script**: `benchmarks/cohere/setup_1b.py`
- **Status**: Complete, all 5 shards indexed

### Index Parameters (derived from first principles)

```
nlist_total = min(N/1000, 65536) = 65,536
nlist/shard = 65,536 / 5 = 13,107
density = 194M / 13,107 = ~14,800 vectors/partition
nprobe range = 1-5% x 13,107 = 131-655
```

## PCA-512 (Dimensionality Reduced)

- **Rows**: 972,996,765 (5 shards x 194,599,353, same as 1B)
- **Dimensions**: 512 (float32, L2-normalized after SVD transform)
- **Method**: TruncatedSVD(512) trained on 500K sample from shard-0
- **Explained variance**: 96.63%
- **Local**: `/data/work/tmp/s1b-pca512/shard-{0-4}.lance`
- **OBS**: `s3://knowledgebase-5f43/pca512-1b/`
- **Index**: IVF_RQ, 13,107 partitions/shard (same as baseline)
- **Build script**: `benchmarks/cohere/setup_pca.py`
- **SVD model**: `/data/work/tmp/s1b-pca512/svd_model.pkl`
- **Status**: Complete, all 5 shards indexed and on OBS

## PCA-512 + IVF_SQ (Scalar Quantization)

- **Rows**: Same as PCA-512 (copies of the same vectors)
- **Dimensions**: 512 (float32, L2-normalized)
- **Index**: IVF_SQ, 13,107 partitions/shard, 8-bit scalar quantization
- **Shard size**: 466GB/shard (SQ codes ~100GB + raw vectors ~386GB)
- **Local**: `/data/work/tmp/s1b-pca512-sq/shard-{0-4}.lance`
- **OBS**: `s3://knowledgebase-5f43/pca512-sq-1b/`
- **Status**: Complete, tested — **SQ is slower than RQ everywhere** (see [pca512-sq experiment](../experiments/pca512-sq.md))

## 324M FineWeb-Edu (Legacy)

- **Rows**: ~324M (4 shards x ~81M)
- **Dimensions**: 1024 (float32, L2-normalized)
- **Source**: FineWeb-Edu embeddings (original, no replication)
- **Metric**: cosine
- **Local**: `/data/work/tmp/lance-shards-rq/shard-{0-3}.lance`
- **Index**: IVF_RQ, 1,125 partitions/shard
- **Status**: Legacy, kept for historical comparison

## Ground Truth

| File | Queries | K | Source | Notes |
|------|---------|---|--------|-------|
| `/data/work/tmp/gt_1b_shard0_top10k_5q_seed42.npz` | 5 | 10,000 | Original 1024-dim shard-0 | Flat search, seed=42 |
| `/data/work/tmp/s1b-pca512/gt_positional_v2.npz` | 5 | 10,000 | Original 1024-dim shard-0 | Positional indices, matched to benchmark queries |

**Critical**: GT v2 uses `rng.choice(n, 8)` + take indices[3:] (warmup=3) to match benchmark sampling exactly.
See [dimensionality-reduction](../concepts/dimensionality-reduction.md) for cross-dataset recall methodology.
