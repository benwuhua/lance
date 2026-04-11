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
| PCA-512 + SQ | 512 | 13,107 | ~100GB/shard (SQ codes) + 386GB/shard (vectors) | 194M |
| 324M | 1024 | 1,125 | ~44GB | 81M |

## IVF_USQ (Experimental)

**Ultra-Sparse Quantization** — 4-bit quantization via Hanns library.

### What it is
- Same IVF partitioning as RQ
- USQ applies random rotation → normalize → 4-bit quantize
- Uses Hanns approximate scoring for distance computation
- Gated behind `#[cfg(feature = "hanns")]`

### Our Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_partitions | 13,107/shard | Same as RQ/SQ for fair comparison |
| metric | cosine | Same data |
| num_bits | 4 | USQ default (half of SQ's 8-bit) |
| code size | ~336B/vector | 256B packed + 64B signs + 16B meta |

### Key Properties
- **Build time**: ~20 min/shard (194M × 512-dim)
- **Recall ceiling**: ~0.97 (same as RQ)
- **Speed**: 2-6% faster than RQ at np<=256, 7-31% slower at np>=1024 (cache pollution from 63GB/shard index)
- **Trade-off**: Lower per-vector accuracy (4-bit) but much faster distance computation

