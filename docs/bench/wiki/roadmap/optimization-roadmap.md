---
tags: [roadmap, optimization]
date: 2026-04-08
---

# Optimization Roadmap

## Current Performance

Best achievable with current Lance IVF_RQ on 1B dataset:

| Target | Config | DRAM | SSD | OBS |
|--------|--------|------|-----|-----|
| recall >= 0.90 | np=128 rf=2 | 956ms | 948ms | ~20s |
| recall >= 0.95 | np=1024 rf=2 | 1111ms | 1298ms | ~21s |
| recall >= 0.99 | np=1024 rf=3 | 1543ms | 1715ms | ~40s |

## Optimization Priorities

### P0: Multi-bit RQ (Expected: 2-3x latency improvement at high recall)

**Problem**: Current 8-bit RQ (256 centroids/level) is over-precise for initial scan. Distance computation over many centroids is slow.

**Solution**: Configurable bit width (4-6 bits) per RQ level. Fewer centroids = faster lookup, more levels for same quality.

**Expected impact**:
- rf=1 could achieve 0.95 recall (vs current rf=2)
- Saves one rf unit = ~350ms per query
- Combined with SQ8 rerank, could reach 0.99 with rf=1 + SQ8

**Effort**: Medium (Rust core changes to RQ codebook generation)

### P1: SQ8 Two-Stage Rerank (Expected: 10x faster refinement)

**Problem**: Current refinement reads original (float32) vectors. For top-K=10,000 with rf=3, that's 30,000 random vector reads.

**Solution**: Store SQ8 (uint8 per dimension) compressed vectors alongside RQ codes. Use SQ8 for refinement instead of original vectors.

**Expected impact**:
- SQ8 distance: ~10x faster than float32 (uint8 arithmetic, cache-friendly)
- SQ8 recall degradation: ~0.1% vs original (negligible)
- Storage: +25% index size (1 byte/dim vs 4 bytes/dim for original)

**Why external take() doesn't work**: See [exact-rerank experiment](../experiments/exact-rerank.md). Random access to columnar storage is ~0.28ms/candidate. Must be stored within the index.

**Effort**: Medium-High (index storage format change + distance computation path)

### P2: PQ FastScan (Expected: 4-10x PQ throughput)

**Problem**: Current PQ distance computation is not SIMD-optimized.

**Solution**: Implement block-based PQ scan (bbs=32) with AVX-512 VNNI `dpbusd` instructions.

**Expected impact**: 4-10x PQ distance computation throughput.

**Effort**: High (platform-specific SIMD optimization)

### P3: Cross-Query Partition Cache (Expected: 50-80% IO reduction)

**Problem**: Each query re-reads partition centroids and metadata from storage.

**Solution**: LRU cache of partition metadata across queries, keyed by (dataset, partition_id).

**Expected impact**:
- DRAM: Minimal (already cached by page cache)
- SSD: 50-80% IO reduction for repeated queries
- OBS: 50-80% network transfer reduction (most impactful)

**Effort**: Low-Medium (add caching layer to index reader)

## Optimization Impact Projection

If all optimizations are implemented:

| Config | Current | Projected | Improvement |
|--------|---------|-----------|-------------|
| recall 0.95 (DRAM) | 1111ms | ~300ms | 3.7x |
| recall 0.99 (DRAM) | 1543ms | ~500ms | 3.1x |
| recall 0.95 (OBS) | ~21s | ~8s | 2.6x |
| recall 0.99 (OBS) | ~40s | ~15s | 2.7x |

## Quick Wins (No Code Changes)

These optimizations are available without modifying Lance:

1. **Increase nprobe, decrease rf**: np=1024 rf=2 gives 0.9825 recall @ 1111ms. Better than np=256 rf=2 (0.9497 @ 1076ms) for only 35ms more.

2. **PCA-512 for OBS**: 3-7% OBS latency reduction when recall <= 0.95 is acceptable. See [PCA-512 experiment](../experiments/pca512.md).

3. **Warm page cache properly**: Current warm_page_cache reads everything, polluting cache. Only warm index files (~132GB) for better cache utilization.

## Cross-refs

- [Knowhere gap analysis](knowhere-gap-analysis.md) — detailed comparison with Knowhere
- [1B baseline](../experiments/1b-baseline.md) — current performance numbers
- [Exact rerank](../experiments/exact-rerank.md) — why external rerank failed
