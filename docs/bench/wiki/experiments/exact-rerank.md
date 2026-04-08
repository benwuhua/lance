---
tags: [experiment, rerank, abandoned]
date: 2026-04-08
---

# Exact Rerank Experiment (Abandoned)

## Goal

After Plan B merge, fetch original vectors for top-K*rerank_factor candidates and compute exact distances for re-ranking. This would produce accurate distances and higher recall without increasing refine_factor.

## Setup

- **Method**: After approximate merge across shards, take top-K*N candidates, fetch original vectors via `take()`, compute exact cosine distances, re-sort
- **Implementation**: Added to `benchmarks/cohere/bench.py` as `--rerank-factor` parameter
- **Parallelism**: Fetch vectors per shard in parallel using ThreadPoolExecutor

## Results

**Not completed** — abandoned after analysis revealed fundamental bottleneck.

### Estimated Cost
- `take()` random access: ~0.28ms/candidate
- At rerank_factor=2 (20,000 candidates): 20,000 x 0.28ms = **6,479ms**
- vs rf=2 (refine_factor in index): **1,111ms**

## Findings

1. **External take() is too slow**: Columnar storage is optimized for sequential reads, not random access. Each `take()` call requires seeking to different fragments and rows.

2. **5x slower than built-in refinement**: Index-internal refinement reads vectors sequentially during the probe phase. External take() pays random access penalty.

3. **The right solution is index-internal**: SQ8 storage within the index (like Knowhere's approach) would enable fast re-ranking without random access to original data.

## Why This Matters

This experiment confirmed that the refinement bottleneck is architectural, not algorithmic:
- **Current**: refine_factor reads original vectors sequentially during index probe (fast)
- **External rerank**: take() reads original vectors randomly (slow)
- **Needed**: SQ8-compressed vectors stored alongside RQ codes in the index (fast + accurate)

See [optimization-roadmap](../roadmap/optimization-roadmap.md) for the SQ8 two-stage rerank proposal.

## Code

The implementation remains in `benchmarks/cohere/bench.py`:
- `--rerank-factor N` parameter
- `_rowid_to_position()` for converting _rowid to positional indices
- `_fetch_and_score()` for parallel vector fetch and exact distance computation
- Integrated into `query_plan_b()`

## Cross-refs

- [Top-K architecture](../concepts/topk-architecture.md) — Plan B merge details
- [Optimization roadmap](../roadmap/optimization-roadmap.md) — SQ8 rerank proposal
- [Knowhere gap analysis](../roadmap/knowhere-gap-analysis.md)
