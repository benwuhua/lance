---
tags: [storage, entity]
date: 2026-04-08
---

# Storage Backends

Three storage tiers benchmarked for distributed vector search.

## DRAM (NVMe Hot)

- **What**: Local NVMe RAID0 (12TB), data fully in Linux page cache
- **When**: Data warm in RAM from prior reads or explicit `warm_page_cache()`
- **Latency profile**: IO is negligible, dominated by CPU (RQ distance computation)
- **Preparation**: `warm_page_cache()` reads all shard files into page cache (~40 min for 1B)
- **Reality check**: 1B dataset is 3.8 TiB total, exceeds 493GB RAM, so page cache is partial
  - Index data (~132GB) fits entirely in RAM
  - Raw vectors partially cached (warm_page_cache pollutes cache by reading everything)
- **Benchmark flag**: `--storage dram`

### DRAM latency model
```
Total = probe_cost(nprobe) + rf_cost(refine_factor * K)
probe_cost ~ 200-600ms (CPU + small IO for index)
rf_cost ~ 330-430ms per rf unit (pure CPU)
```

## SSD (NVMe Cold)

- **What**: Same local NVMe, but page cache dropped before each query
- **When**: Simulates cold-start or data-larger-than-RAM scenario
- **Preparation**: `drop_caches` before each query (requires root: `echo 3 > /proc/sys/vm/drop_caches`)
- **Latency profile**: Small IO overhead on top of DRAM baseline
- **Benchmark flag**: `--storage ssd`

### SSD overhead analysis
| nprobe | DRAM (ms) | SSD (ms) | Overhead (ms) | Reason |
|--------|-----------|----------|---------------|--------|
| 128 | 602 | 614 | +12 | Small working set, mostly in index |
| 1024 | 784 | 936 | +152 | Larger probe reads more pages |
| 1024 rf=3 | 1543 | 1715 | +172 | rf=3 reads more original vectors |

**Key insight**: SSD overhead is proportional to IO working set, not rf. rf is CPU-bound.

## OBS (Huawei Cloud S3)

- **What**: Huawei OBS (S3-compatible) in ap-southeast-1
- **When**: Cloud deployment, data not local
- **Endpoint**: `https://obs.ap-southeast-1.myhuaweicloud.com`
- **Bucket**: `knowledgebase-5f43`
- **Upload tool**: `/usr/local/bin/obsutil` (5.7.9) — use `obsutil sync` (not `cp`, which requires interactive confirmation)
- **Benchmark flag**: `--storage obs`

### OBS latency model
```
Total = network_transfer_cost + CPU_cost
network_transfer ~ 8-10s base (index metadata + partition lists)
rf adds ~10s per unit (downloading original vectors over network)
```

### OBS latency reference (1B, 1024-dim)
| Config | Mean (ms) | Notes |
|--------|-----------|-------|
| np=128 rf=1 | 10,056 | Baseline OBS |
| np=512 rf=1 | 11,508 | More partitions to read |
| np=256 rf=2 | 20,806 | rf=2 doubles vector downloads |
| np=512 rf=2 | 21,006 | rf=2 + high nprobe |

### PCA-512 OBS advantage
PCA-512 vectors are half the size (512x4B vs 1024x4B), reducing network transfer:
- rf=1: ~7% faster (9,382ms vs 10,056ms at np=128)
- rf=2: ~5% faster (18,729ms vs 19,772ms at np=128)

See [pca512 experiment](../experiments/pca512.md) for full comparison.

## Storage Cost Summary

| Tier | np=128 rf=1 | np=1024 rf=3 | Best for |
|------|-------------|--------------|----------|
| DRAM | 602ms | 1543ms | Lowest latency |
| SSD | 614ms | 1715ms | Data > RAM |
| OBS | 10,056ms | ~40,000ms est. | Cloud deployment |
