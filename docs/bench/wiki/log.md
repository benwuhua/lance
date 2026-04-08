# Wiki Log

Chronological record of wiki activity.

## [2026-04-08] ingest | 1B Baseline Results (DRAM + SSD + OBS)
- Source: benchmark results from ECS
- Pages created: [1b-baseline](experiments/1b-baseline.md)
- Pages updated: [datasets](entities/datasets.md), [indexes](entities/indexes.md), [storage](entities/storage.md)
- Key data: 14 configs across DRAM/SSD, 6 OBS configs

## [2026-04-08] ingest | PCA-512 Experiment Results
- Source: benchmark results from ECS (`/data/work/tmp/pca512-results-v2/`)
- Pages created: [pca512](experiments/pca512.md), [dimensionality-reduction](concepts/dimensionality-reduction.md)
- Key findings: recall ceiling ~0.97, OBS 3-7% faster, DRAM low-nprobe 10% faster
- GT alignment fix: regenerated with `rng.choice(n, 8)` + indices[3:] to match benchmark

## [2026-04-08] ingest | Exact Rerank Analysis (Abandoned)
- Source: analysis of take() performance on columnar storage
- Pages created: [exact-rerank](experiments/exact-rerank.md)
- Key finding: take() at 0.28ms/candidate is 5x slower than built-in refinement
- Conclusion: need SQ8 storage within index, not external take()

## [2026-04-08] ingest | Knowhere Gap Analysis + Optimization Roadmap
- Source: analysis of Knowhere/Milvus index architecture
- Pages created: [knowhere-gap-analysis](roadmap/knowhere-gap-analysis.md), [optimization-roadmap](roadmap/optimization-roadmap.md)
- Key gaps: multi-bit RQ (P0), SQ8 rerank (P1), PQ FastScan (P2), partition cache (P3)

## [2026-04-08] ingest | Wiki Initialization
- Source: project knowledge from memory files + codebase
- Pages created: all entity, concept, experiment, roadmap pages
- Pages created: [SCHEMA](SCHEMA.md), [index](index.md), [log](log.md) (this file)
- Total pages: 15

## [2026-04-06] ingest | 1B Dataset Build Complete
- Source: ECS dataset creation
- 5 shards x 194M rows, IVF_RQ 13,107 partitions each
- Data replicated 3x from 324M source with Gaussian noise

## [2026-04-06] ingest | 324M Baseline Results (Legacy)
- Source: pareto_sweep.py results
- Pages created: [324m-baseline](experiments/324m-baseline.md)
- Results stored in `docs/bench/pareto-results.json`
