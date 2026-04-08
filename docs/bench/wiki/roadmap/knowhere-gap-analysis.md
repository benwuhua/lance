---
tags: [roadmap, analysis, knowhere]
date: 2026-04-08
---

# Lance vs Knowhere Index Comparison

## Current State

Lance IVF_RQ achieves competitive recall but has significant latency gaps vs Knowhere (Milvus/Zilliz's index engine) for high-recall scenarios.

### Performance Gap (at recall ~0.99)

| System | Config | Latency | Key Difference |
|--------|--------|---------|---------------|
| Knowhere | nprobe=16, rf=1 | ~50ms | Multi-bit RQ (5-bit) |
| Lance | np=1024 rf=3 | 1543ms | 8-bit RQ, rf=3 for recall |

**30x latency gap** at comparable recall levels.

## Root Cause Analysis

### Why Knowhere is faster

1. **Multi-bit RQ**: Knowhere uses 5-bit (32 centroids) or 6-bit (64 centroids) per RQ level vs Lance's 8-bit (256 centroids). Fewer centroids = faster distance computation but needs more levels for same quality.

2. **SQ8 two-stage rerank**: Instead of using original vectors for refinement, Knowhere stores SQ8 (scalar quantization to 8-bit) compressed vectors alongside RQ codes. SQ8 distance is nearly as accurate as original but much faster to compute.

3. **PQ FastScan**: For the initial scan, Knowhere uses optimized PQ distance computation with:
   - Block size 32 (bbs=32) for cache-friendly access
   - AVX-512 VNNI `dpbusd` instruction for 4x throughput
   - Pre-computed lookup tables per sub-vector

4. **Partition cache**: Knowhere caches partition centroids/metadata across queries. Lance currently reloads per-query.

## Prioritized Improvements

### P0: Multi-bit RQ
- **Impact**: rf=1 could achieve 0.95 recall (vs current rf=2 needed)
- **Savings**: ~330-430ms per eliminated rf unit
- **Approach**: Support 4-6 bit RQ levels with configurable bit width
- **Trade-off**: More levels needed, but distance computation is faster per level

### P1: SQ8 Two-Stage Rerank
- **Impact**: Replace rf (original vectors) with SQ8 rerank
- **Savings**: SQ8 distance ~10x faster than original vector distance
- **Approach**: Store SQ8-compressed copy of vectors in index metadata
- **Trade-off**: Extra storage (~25% index size increase), but eliminates random IO

### P2: PQ FastScan
- **Impact**: 4-10x PQ distance throughput
- **Approach**: Implement bbs=32 blocking + AVX-512 VNNI path
- **Trade-off**: Significant implementation effort, x86-only optimization

### P3: Cross-Query Partition Cache
- **Impact**: 50-80% IO reduction for repeated queries
- **Approach**: LRU cache of partition centroids/metadata across queries
- **Trade-off**: Memory usage, cache invalidation complexity
- **Note**: Most impactful for OBS where IO is expensive

## Related Documents

- Detailed analysis: `docs/bench/1b/lance-vs-knowhere-index-comparison.md` (if exists on ECS)
- [1B baseline results](../experiments/1b-baseline.md)
- [Exact rerank experiment](../experiments/exact-rerank.md) — why external take() doesn't work
