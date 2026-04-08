# Wiki Index

Last updated: 2026-04-08

## Entities

### [Datasets](entities/datasets.md)
1B FineWeb-Edu (1024-dim), PCA-512 (512-dim), 324M legacy, ground truth files.

### [Indexes](entities/indexes.md)
IVF_RQ configuration (13,107 partitions, 4-level RQ), IVF_PQ alternative, index size reference.

### [Storage](entities/storage.md)
DRAM (page cache hot), SSD (cold cache), OBS (Huawei S3). Latency models and cost breakdown.

### [ECS Environment](entities/ecs.md)
Host ecs-hk-1b: 493GB RAM, 12TB NVMe, OBS credentials, data paths.

## Concepts

### [Performance Metrics](concepts/metrics.md)
Recall@K definition, cross-dataset recall methodology, latency measurement, nprobe/refine-factor parameters, Pareto frontier.

### [Top-K Architecture](concepts/topk-architecture.md)
Plan A (single dataset) vs Plan B (sharded parallel). ThreadPoolExecutor, merge algorithm, _rowid conversion, exact rerank (abandoned).

### [Dimensionality Reduction](concepts/dimensionality-reduction.md)
TruncatedSVD method, variance retention, cross-dataset recall methodology, dimension trade-offs.

## Experiments

### [1B Baseline](experiments/1b-baseline.md)
Primary benchmark: 972M rows, 1024-dim, IVF_RQ. DRAM/SSD/OBS results across 14 configs. Key finding: CPU is bottleneck at high recall.

### [PCA-512](experiments/pca512.md)
TruncatedSVD 1024→512 dim reduction. Recall ceiling ~0.97, DRAM 10% faster at low nprobe, OBS 3-7% faster. Best for OBS with recall <= 0.95.

### [PCA-512 + IVF_SQ](experiments/pca512-sq.md)
IVF_SQ (8-bit scalar quantization) on PCA-512. **Slower than RQ everywhere** — SQ stores codes on top of original vectors (466GB vs 386GB/shard). OBS 10-100% slower. SQ needs to replace (not supplement) float32 vectors to be useful.

### [Exact Rerank](experiments/exact-rerank.md) (Abandoned)
Post-merge exact rerank via take(). Abandoned: random access to columnar storage 5x slower than built-in refinement.

### [324M Baseline](experiments/324m-baseline.md) (Legacy)
Initial 324M benchmark. Superseded by 1B baseline. Results in `pareto-results.json`.

## Roadmap

### [Knowhere Gap Analysis](roadmap/knowhere-gap-analysis.md)
Lance vs Knowhere: 30x latency gap at high recall. Root causes: multi-bit RQ, SQ8 rerank, PQ FastScan, partition cache.

### [Optimization Roadmap](roadmap/optimization-roadmap.md)
P0 Multi-bit RQ → P1 SQ8 rerank → P2 PQ FastScan → P3 Partition cache. Projected 3x latency improvement.
