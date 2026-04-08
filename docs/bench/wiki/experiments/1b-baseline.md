---
tags: [experiment, baseline, 1b]
date: 2026-04-08
---

# 1B Baseline Experiment

## Goal

Establish performance baseline for 1B-scale (972M rows) IVF_RQ vector search across three storage tiers (DRAM, SSD, OBS) with recall measurements.

## Setup

- **Dataset**: 1B FineWeb-Edu, 5 shards x 194M rows, 1024-dim, cosine
- **Index**: IVF_RQ, 13,107 partitions/shard
- **Queries**: 5 timed (after 3 warmup), seed=42
- **GT**: Flat search on shard-0, k=10,000
- **Scripts**: `benchmarks/cohere/bench.py query --plan B`
- **Run scripts**: `benchmarks/cohere/test_pca512_query.sh` (DRAM), custom OBS sweep

## Results

### DRAM (Partial Page Cache)

| nprobe | rf | Recall | Mean (ms) | P50 (ms) | P99 (ms) |
|--------|-----|--------|-----------|----------|----------|
| 128 | 1 | 0.8300 | 602 | — | — |
| 256 | 1 | 0.8450 | 712 | — | — |
| 1024 | 1 | 0.8607 | 784 | — | — |
| 128 | 2 | 0.9213 | 956 | — | — |
| 256 | 2 | 0.9497 | 1076 | — | — |
| 1024 | 2 | 0.9825 | 1111 | — | — |
| 1024 | 3 | 0.9932 | 1543 | — | — |
| 1024 | 5 | 0.9950 | 2517 | — | — |

### SSD (Cold Cache)

| nprobe | rf | Recall | Mean (ms) | P50 (ms) | P99 (ms) |
|--------|-----|--------|-----------|----------|----------|
| 128 | 1 | 0.8300 | 614 | — | — |
| 256 | 1 | 0.8450 | 727 | — | — |
| 1024 | 1 | 0.8607 | 936 | — | — |
| 128 | 2 | 0.9213 | 948 | — | — |
| 256 | 2 | 0.9497 | 1075 | — | — |
| 1024 | 2 | 0.9825 | 1298 | — | — |
| 1024 | 3 | 0.9932 | 1715 | — | — |
| 1024 | 5 | 0.9950 | 2656 | — | — |

### OBS

| nprobe | rf | Mean (ms) |
|--------|-----|-----------|
| 64 | 1 | 10,056 |
| 128 | 1 | 10,442 |
| 512 | 1 | 11,508 |
| 64 | 2 | 19,772 |
| 256 | 2 | 20,806 |
| 512 | 2 | 21,006 |

## Findings

1. **CPU is the bottleneck**: rf unit cost is ~330-430ms, identical for DRAM and SSD. The algorithm needs multi-bit RQ or SQ8 rerank to break this ceiling.

2. **DRAM-SSD gap is small**: Only 12ms at np=128, 152ms at np=1024. Index data (~132GB) fits in RAM, so SSD overhead is just the residual IO for original vectors.

3. **OBS is network-bound**: ~10s base latency dominated by network transfer. rf doubles the transfer (downloading original vectors).

4. **Recommended configs**:
   - recall >= 0.90: np=128 rf=2 → 956ms (DRAM)
   - recall >= 0.95: np=1024 rf=2 → 1111ms (DRAM)
   - recall >= 0.99: np=1024 rf=3 → 1543ms (DRAM)

5. **Full-DRAM estimate**: Current setup already has index fully in RAM. Full-DRAM would save only ~9% (1400ms vs 1543ms at np=1024 rf=3).

## Cross-refs

- [Storage backends](../entities/storage.md) — detailed per-tier analysis
- [Metrics](../concepts/metrics.md) — recall and latency definitions
- [PCA-512](pca512.md) — dimensionality reduction comparison
- [Optimization roadmap](../roadmap/optimization-roadmap.md) — how to improve
