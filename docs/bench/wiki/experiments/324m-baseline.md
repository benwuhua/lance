---
tags: [experiment, legacy, 324m]
date: 2026-04-08
---

# 324M Baseline (Legacy)

## Goal

Initial benchmark of distributed top-K search on the FineWeb-Edu 324M dataset. Established early performance baselines and Pareto curves.

## Setup

- **Dataset**: FineWeb-Edu 324M rows, 4 shards x ~81M, 1024-dim, cosine
- **Index**: IVF_RQ, 1,125 partitions/shard
- **Scripts**: `benchmarks/cohere/pareto_sweep.py`
- **Results**: `docs/bench/pareto-results.json`

## Key Results (Summary)

Results stored in `docs/bench/pareto-results.json`. This experiment established:
- The basic Pareto curve shape for IVF_RQ
- The rf unit cost (~350ms) that holds across datasets
- The DRAM/SSD/OBS latency hierarchy

## Status

Superseded by the [1B baseline](1b-baseline.md). The 324M dataset and results are kept for historical reference but are no longer actively used.

## Cross-refs

- [1B baseline](1b-baseline.md) — current primary baseline
- [Pareto sweep script](../../benchmarks/cohere/pareto_sweep.py)
- [Datasets](../entities/datasets.md) — 324M dataset details
