---
tags: [concept, architecture]
date: 2026-04-08
---

# Distributed Top-K Search Architecture

## Plan A: Single Dataset

Single Lance dataset, single `to_table(nearest={...})` call. Lance handles partition probing and merging internally.

```
Query → Lance Dataset → [IVF probe → distance compute → refine] → Results
```

- Simple but limited to single machine
- Lance merges partitions internally
- Not used for production benchmarks (sharding needed for 1B scale)

## Plan B: Sharded Parallel (Production)

Split dataset into N shards, query each in parallel, merge results client-side.

```
Query ─┬→ Shard 0 → [probe + compute + refine] → partial_0 ─┐
       ├→ Shard 1 → [probe + compute + refine] → partial_1 ─┤
       ├→ Shard 2 → [probe + compute + refine] → partial_2 ─┼→ Merge → Top-K
       ├→ Shard 3 → [probe + compute + refine] → partial_3 ─┤
       └→ Shard 4 → [probe + compute + refine] → partial_4 ─┘
```

### Implementation Details

- **Parallelism**: `ThreadPoolExecutor` (not processes) — Lance releases GIL during Rust IO
- **Workers**: 5 (one per shard, IO-bound on OBS, CPU-bound on DRAM)
- **Merge**: Concatenate all (rowid, distance) pairs, sort by distance, take top-K
- **Merge cost**: ~1ms (negligible vs query time)

### Merge Algorithm
```python
# Each shard returns top-K results (rowid, distance)
all_results = concat(shard_0_results, ..., shard_4_results)  # 5*K pairs
sorted_results = sort(all_results, by=distance)[:top_k]       # Take top-K
```

### Exact Rerank (Abandoned)
An optional post-merge rerank step was implemented but abandoned:
1. After approximate merge, take top-K*rerank_factor candidates
2. Fetch original vectors via `take()` for each candidate
3. Compute exact distances
4. Re-sort by exact distance

**Why abandoned**: `take()` random access to columnar storage is ~0.28ms/candidate, making it far slower than increasing `refine_factor`. The bottleneck is IO pattern, not computation. See [exact-rerank experiment](../experiments/exact-rerank.md).

### _rowid vs Positional Indices

Lance uses `_rowid = (fragment_id: u32 << 32) | (row_offset: u32)` internally.
For cross-dataset comparison (PCA vs original), we convert to 0-based positional indices:
```python
frag_base[frag.fragment_id] = cumulative_offset  # Built from fragment list
position = frag_base[frag_id] + row_offset
```

This vectorized conversion handles 2D arrays efficiently.

## Benchmark Tool

`benchmarks/cohere/bench.py` — unified benchmark script.

### Subcommands
- `setup` — Build shards with index from source dataset
- `query` — Run single benchmark configuration
- `compare` — Run multiple configurations and output comparison

### Key Parameters
```
--plan B          # Sharded parallel
--storage dram    # DRAM / SSD / OBS
--shard-dir PATH  # Shard directory or s3:// URI
--nprobes N       # Partitions to probe
--refine-factor N # Refinement multiplier
--top-k 10000     # Number of results
--num-shards 5    # Number of shards
--query-count 8   # Total queries (incl warmup)
--warmup 3        # Warmup queries to discard
--seed 42         # RNG seed for reproducibility
--gt-path PATH    # Pre-computed ground truth .npz
```
