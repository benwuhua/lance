---
tags: [concept, metrics]
date: 2026-04-08
---

# Performance Metrics

## Recall@K

The primary quality metric. Measures what fraction of the true top-K results are found by ANN search.

### Definition
```
recall@K = |ANN_top_K ∩ GT_top_K| / K
```

### Our Measurement
- **K = 10,000** (top-K search target)
- **GT source**: Flat brute-force search (`use_index=False`) on original 1024-dim vectors
- **Per-shard measurement**: Recall is measured on shard-0 only, not across merged results
- **Query count**: 5 timed queries (after 3 warmup), seed=42

### Cross-Dataset Recall (PCA-512)
When comparing PCA-512 results against GT from original vectors:
1. GT is computed on **original 1024-dim** shard-0 (flat search)
2. GT `_rowid` values are converted to **positional indices** (0-based row numbers)
3. ANN results from PCA-512 shard-0 are also converted to positional indices
4. Recall = overlap of positional indices

This works because PCA-512 shards preserve the same row order as original shards.

See also: [dimensionality-reduction](dimensionality-reduction.md), [datasets](../entities/datasets.md)

## Latency

### What we measure
- **End-to-end query latency**: from query submission to merged results returned
- **Includes**: index probe + distance computation + refinement + merge
- **Warmup**: 3 queries discarded before timing begins
- **Reported**: mean, p50, p95, p99, max across 5 timed queries

### Latency Components (Plan B, 5 shards)

```
Total = max(shard_query_times) + merge_cost
shard_query = partition_probe + distance_compute + refine
merge = sort(all_results) + take_top_K  (~1ms, negligible)
```

### Latency by Storage Tier
See [storage](../entities/storage.md) for detailed breakdowns.

## Key Parameters

### nprobe (N)
- Number of IVF partitions to probe per query
- Higher = better recall, higher latency
- Range tested: 64-1024 (out of 13,107 partitions)
- 1-5% of partitions is the useful range for our density (~14,800 vecs/partition)

### refine_factor (rf)
- Number of candidates to re-rank with original vectors
- rf=R means take top-K*R candidates by approximate distance, re-rank, return top-K
- Each additional rf unit costs ~330-430ms (CPU-bound, same for DRAM/SSD)
- rf=1: no refinement (approximate distances only)
- rf=2: 2x candidates re-ranked (big recall jump: ~0.86 → ~0.98)
- rf=3-5: diminishing returns for recall, linear cost increase

### top-K
- Number of nearest neighbors to return
- We use K=10,000 (large-scale top-K for retrieval augmentation)
- Large K makes refinement more expensive (more candidates to re-rank)

## Pareto Frontier

The optimal trade-off curve between recall and latency. Key operating points for 1B DRAM:

| Target Recall | Config | Latency |
|--------------|--------|---------|
| >= 0.90 | np=128 rf=2 | 956ms |
| >= 0.95 | np=1024 rf=2 | 1111ms |
| >= 0.99 | np=1024 rf=3 | 1543ms |
