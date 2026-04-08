---
tags: [index, entity]
date: 2026-04-08
---

# Index Configurations

## IVF_RQ (Primary)

**Residual Quantization** — the index type used in all production benchmarks.

### What it is
- IVF (Inverted File) partitions vectors into Voronoi cells
- RQ (Residual Quantization) compresses each vector into a multi-level code
- At query time: probe N partitions, compute approximate distances via RQ codes, optionally refine with original vectors

### Our Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_partitions | 13,107/shard | N/1000 total, divided by 5 shards |
| metric | cosine | FineWeb-Edu embeddings are L2-normalized |
| codebook levels | 4 (default) | Lance RQ default |
| bits per level | 8 (default) | 256 centroids per level |

### Key Properties
- **Compressed size**: ~132GB for 1B vectors (1024-dim), fits in 493GB RAM
- **Refine factor (rf)**: Each unit costs ~330-430ms CPU time (independent of storage)
- **Recall ceiling**: rf=1 max ~0.86, rf=2 max ~0.98, rf=3 max ~0.99+
- **Distance computation**: RQ codes only (fast), then original vectors for refinement

### How refine_factor works
1. Probe N partitions, collect candidates
2. For rf=R: take top-K*R candidates by approximate distance
3. Re-rank with original vectors, return top-K
4. Each additional rf unit is CPU-bound (~350ms), not IO-bound

See [metrics](../concepts/metrics.md) for recall measurement methodology.

## IVF_PQ (Alternative)

**Product Quantization** — alternative index type, tested but not used in production.

### What it is
- Same IVF partitioning as RQ
- PQ splits each vector into sub-vectors and quantizes each independently
- Parameters: `num_sub_vectors=64`, `num_bits=8`

### Status
- Tested in early benchmarks, lower recall at same compression ratio
- Not used in current experiments
- See [optimization-roadmap](../roadmap/optimization-roadmap.md) for PQ FastScan potential

## Index Size Reference

| Dataset | Dim | Partitions | Index Size | Rows/Shard |
|---------|-----|------------|------------|------------|
| 1B baseline | 1024 | 13,107 | ~132GB | 194M |
| PCA-512 | 512 | 13,107 | ~66GB | 194M |
| 324M | 1024 | 1,125 | ~44GB | 81M |
