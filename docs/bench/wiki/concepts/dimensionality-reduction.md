---
tags: [concept, pca, svd]
date: 2026-04-08
---

# Dimensionality Reduction for Vector Search

## Why Reduce Dimensions?

For OBS (cloud S3) deployment, network transfer is the bottleneck:
- 1024-dim vector = 1024 x 4B = 4,096 bytes per vector
- 512-dim vector = 512 x 4B = 2,048 bytes per vector
- At rf=2, fetching 20,000 vectors: 80MB vs 40MB transfer

Reducing dimensions halves network transfer, potentially halving OBS latency.

## Method: TruncatedSVD

We use **TruncatedSVD** (not PCA) because:
- No centering step — preserves cosine structure better
- PCA subtracts the global mean, which distorts angular relationships
- TruncatedSVD finds the top-K singular vectors directly

### Pipeline
1. Train SVD on sample from shard-0 (500K vectors)
2. Transform all vectors: `v' = v @ components.T` (components shape: 512x1024)
3. L2-normalize transformed vectors (required for cosine metric)
4. Build IVF_RQ index on reduced-dim vectors
5. At query time: transform query vector the same way, then search

### Variance Retention
- 512 components from 1024: **96.63%** explained variance
- This is a hard ceiling on achievable recall (see [pca512 experiment](../experiments/pca512.md))

## Cross-Dataset Recall Methodology

Comparing ANN results from reduced-dim index against GT from original vectors:

1. **GT generation**: Flat search on **original 1024-dim** shard-0
2. **Index to positional**: Convert GT `_rowid` to 0-based positional indices
3. **ANN query**: Search PCA-512 shard-0, get `_rowid` results
4. **Index to positional**: Convert ANN `_rowid` to positional indices
5. **Recall**: `|ANN_positional ∩ GT_positional| / K`

This works because PCA-512 shards preserve the same row order as original shards.
The positional index is row-order-independent (survives SVD transformation).

### GT Query Alignment

**Critical**: GT queries must use identical sampling as the benchmark:
```python
rng = np.random.default_rng(42)
indices = rng.choice(total, size=8, replace=False)  # 8 queries
query_indices = indices[3:]  # After warmup=3
```

Using `rng.choice(n, 5)` instead of `rng.choice(n, 8)` produces completely different indices,
even with the same seed. This caused recall=0 in initial PCA-512 tests.

## Dimension Trade-offs

| Dim | Variance | Recall Ceiling | Vector Size | OBS Advantage |
|-----|----------|---------------|-------------|---------------|
| 1024 | 100% | 1.0 | 4KB | Baseline |
| 512 | 96.6% | ~0.97 | 2KB | 3-7% faster |
| 256 | ~92% est. | ~0.93 est. | 1KB | ~12% faster est. |
| 128 | ~85% est. | ~0.85 est. | 0.5KB | ~20% faster est. |

Lower dimensions → more speed but lower recall ceiling. The 512-dim sweet spot gives 3-7% OBS speedup with recall up to ~0.97.

## Related

- [PCA-512 experiment results](../experiments/pca512.md)
- [Setup script](../../benchmarks/cohere/setup_pca.py)
- [Datasets](../entities/datasets.md) — PCA-512 dataset details
