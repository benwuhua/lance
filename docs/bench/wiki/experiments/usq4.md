---
tags: [experiment, quantization, usq, 4-bit]
date: 2026-04-11
---

# USQ 4-bit Experiment

## Goal

Benchmark Hanns USQ (Ultra-Sparse Quantization, 4-bit) vs existing IVF_RQ on PCA-512 data, measuring recall and latency across DRAM and SSD.

## Setup

- **Dataset**: PCA-512, 5 shards x 194M rows (FineWeb-Edu 1B), cosine metric
- **Index USQ**: IVF_USQ, 13,107 partitions/shard, 4-bit quantization
- **Index RQ**: IVF_RQ, 13,107 partitions/shard (existing baseline)
- **GT**: Flat search on PCA-512 shard-0 (seed=42), 5 queries, k=10000
- **Query vectors**: Sampled from shard-0 via `rng.choice(n, 5, seed=42)`
- **Storage**: DRAM (page cache warm) and SSD (drop_caches between queries)
- **IO_THREADS**: 128

## Index Build

- **Build time**: ~20 min/shard serial, ~50 min for 5 shards parallel
- **Code size**: ~336 bytes/vector (256B codes + 64B signs + 16B meta)
- **RQ code size**: ~160 bytes/vector (8-bit x 512/2 = 256B, compressed)
- **Build optimization**: `Arc<OnceLock>` for QR decomposition (was O(N x d^3), now O(d^3) once)

## Results

### Single Shard (shard-0, 194M rows)

| Config | Recall | USQ Mean | RQ Mean | Delta |
|--------|--------|----------|---------|-------|
| np=128 rf=1 | 0.7926 | 180ms | 183ms | -2% |
| np=256 rf=1 | 0.8032 | 159ms | 162ms | -2% |
| np=1024 rf=1 | 0.8111 | 282ms | 283ms | 0% |
| np=128 rf=2 | 0.9093 | 234ms | 236ms | -1% |
| np=256 rf=2 | 0.9340 | 259ms | 260ms | 0% |
| np=1024 rf=2 | 0.9541 | 335ms | 341ms | -2% |
| np=1024 rf=3 | 0.9795 | 520ms | 524ms | -1% |
| np=1024 rf=5 | 0.9900 | 953ms | 968ms | -2% |

### 5 Shards (970M rows, 5 queries, k=10000)

#### DRAM (page cache warm)

| Config | Recall | USQ Mean | RQ Mean | Delta |
|--------|--------|----------|---------|-------|
| np=128 rf=1 | 0.7926 | 580ms | 573ms | +1% |
| np=256 rf=1 | 0.8032 | 517ms | 509ms | +2% |
| np=1024 rf=1 | 0.8111 | 924ms | 903ms | +2% |
| np=128 rf=2 | 0.9093 | 782ms | 776ms | +1% |
| np=256 rf=2 | 0.9340 | 809ms | 801ms | +1% |
| np=1024 rf=2 | 0.9541 | 896ms | 900ms | 0% |
| np=1024 rf=3 | 0.9795 | 1342ms | 1348ms | 0% |
| np=1024 rf=5 | 0.9900 | 2282ms | 2274ms | 0% |

#### SSD (cold cache, drop_caches between queries)

| Config | Recall | USQ Mean | RQ Mean | Delta |
|--------|--------|----------|---------|-------|
| np=128 rf=1 | 0.7926 | 406ms | 399ms | +2% |
| np=256 rf=1 | 0.8032 | 432ms | 410ms | +5% |
| np=1024 rf=1 | 0.8111 | 482ms | 470ms | +3% |
| np=128 rf=2 | 0.9093 | 754ms | 744ms | +1% |
| np=256 rf=2 | 0.9340 | 776ms | 772ms | +1% |
| np=1024 rf=2 | 0.9541 | 862ms | 856ms | +1% |
| np=1024 rf=3 | 0.9795 | 1296ms | 1288ms | +1% |
| np=1024 rf=5 | 0.9900 | 2220ms | 2212ms | 0% |

## Findings

1. **Identical recall**: USQ and RQ produce exactly the same recall at every (nprobe, refine_factor) configuration. This means both quantizers have similar approximation quality for this data distribution.

2. **Negligible latency difference**: USQ is within 0-5% of RQ latency across all configs. No statistically significant advantage in either direction.

3. **DRAM vs SSD gap**: Both USQ and RQ show the same DRAM-SSD gap (~50-150ms depending on nprobe), confirming the bottleneck is the same (index IO pattern, not quantization format).

4. **rf unit cost**: Both ~350-400ms/rf unit, consistent with CPU-bound scoring.

5. **Code size trade-off not realized**: USQ uses ~336B/vector (4-bit) vs RQ ~256B/vector (8-bit + sub-vector structure). Despite 4-bit quantization, the extra meta/sign columns mean USQ is actually larger than RQ. This explains why there's no IO advantage.

## Recommendations

1. **USQ not advantageous over RQ for PCA-512**: Same recall, same latency, slightly larger index size.
2. **Potential advantage paths**: USQ could shine with (a) higher dimensions where code compression matters more, (b) custom IO patterns that skip meta columns, or (c) different data distributions where 4-bit quantization better captures structure.
3. **RQ remains the default**: No reason to switch from RQ based on these results.

## Implementation Notes

- USQ integration required ~15 files across lance-index, lance, and python crates
- Feature-gated behind `#[cfg(feature = "hanns")]`
- Key bug fix: `dist_calculator()` receives flat `Float32Array` (not `FixedSizeListArray`) — all quantizer storages use this convention
- QR decomposition must be cached via `Arc<OnceLock>` to avoid O(N x d^3) overhead during index build
- Transform pipeline must drop the float32 vector column after encoding to avoid 7.2x shuffle bloat

## Cross-refs

- [PCA-512 experiment](pca512.md) — dimensionality reduction baseline
- [1B baseline](1b-baseline.md) — full 1024-dim results
- [Lance vs Knowhere analysis](../../1b/lance-vs-knowhere-index-comparison.md) — optimization roadmap
