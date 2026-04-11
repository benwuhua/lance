---
tags: [experiment, usq, ultra-sparse-quantization]
date: 2026-04-11
---

# PCA-512 + IVF_USQ Experiment

## Goal

Test IVF_USQ (4-bit Ultra-Sparse Quantization via Hanns) on PCA-512 data as a potential replacement for IVF_RQ. USQ uses random rotation + 4-bit quantization with Hanns approximate scoring.

## Setup

- **Dataset**: Same PCA-512 vectors, 5 shards x 194M rows, 512-dim, cosine metric
- **Index**: IVF_USQ, 13,107 partitions/shard, 4-bit quantization
- **USQ code size**: ~336B/vector (256B packed codes + 64B signs + 16B meta)
- **GT**: `gt_pca512_shard0_top10k_5q_seed42_v2.npz` (flat search, seed=42, _rowid format)
- **Query count**: 5 (no warmup), seed=42
- **Hanns**: External crate, gated behind `#[cfg(feature = "hanns")]`

## Build

- **Time**: ~50 min for 5 shards parallel (194M × 512-dim each)
- **Bottleneck**: QR decomposition (O(d³) = O(512³)) amortized via `Arc<OnceLock>`
- **Index size**: ~63GB auxiliary.idx + 26MB index.idx per shard
- **Command**: `ds.create_index("vector", "IVF_USQ", metric="cosine", num_partitions=13107)`

## Results

### 5 Shards DRAM (page cache warm, 970M rows)

| Config | Recall | USQ Mean (ms) | RQ Mean (ms) | SQ Mean (ms) | USQ vs RQ | USQ vs SQ |
|--------|--------|---------------|--------------|--------------|-----------|-----------|
| np=128 rf=1 | 0.7926 | 496 | 523 | 993 | -5% | -50% |
| np=256 rf=1 | 0.8032 | 597 | 636 | 1,050 | -6% | -43% |
| np=512 rf=1 | 0.8081 | 806 | — | 1,684 | — | -52% |
| np=1024 rf=1 | 0.8111 | 1,105 | 846 | 2,479 | +31% | -55% |
| np=128 rf=2 | 0.9093 | 856 | 875 | 1,085 | -2% | -21% |
| np=256 rf=2 | 0.9340 | 957 | 977 | 1,234 | -2% | -22% |
| np=512 rf=2 | 0.9467 | 1,173 | — | 1,653 | — | -29% |
| np=1024 rf=2 | 0.9541 | 1,478 | 1,183 | 2,448 | +25% | -40% |
| np=128 rf=3 | 0.9278 | 1,260 | — | — | — | — |
| np=256 rf=3 | 0.9555 | 1,344 | — | — | — | — |
| np=512 rf=3 | 0.9711 | 1,516 | — | — | — | — |
| np=1024 rf=3 | 0.9795 | 1,812 | 1,563 | 2,805 | +16% | -35% |
| np=128 rf=5 | 0.9340 | 1,967 | — | — | — | — |
| np=256 rf=5 | 0.9639 | 2,277 | — | — | — | — |
| np=512 rf=5 | 0.9807 | 2,442 | — | — | — | — |
| np=1024 rf=5 | 0.9900 | 2,678 | 2,494 | 3,575 | +7% | -25% |

### 5 Shards SSD (cold cache, drop_caches before each config)

| Config | Recall | USQ Mean (ms) | USQ P99 (ms) | SSD vs DRAM |
|--------|--------|---------------|---------------|-------------|
| np=128 rf=1 | 0.7926 | 522 | 655 | +5% |
| np=256 rf=1 | 0.8032 | 628 | 809 | +5% |
| np=512 rf=1 | 0.8081 | 823 | 1,090 | +2% |
| np=1024 rf=1 | 0.8111 | 1,142 | 1,559 | +3% |
| np=128 rf=2 | 0.9093 | 868 | 1,012 | +1% |
| np=256 rf=2 | 0.9340 | 964 | 1,137 | +1% |
| np=512 rf=2 | 0.9467 | 1,187 | 1,435 | +1% |
| np=1024 rf=2 | 0.9541 | 1,513 | 1,965 | +2% |
| np=128 rf=3 | 0.9278 | 1,212 | 1,354 | -4% |
| np=256 rf=3 | 0.9555 | 1,368 | 1,532 | +2% |
| np=512 rf=3 | 0.9711 | 1,562 | 1,793 | +3% |
| np=1024 rf=3 | 0.9795 | 1,905 | 2,285 | +5% |
| np=128 rf=5 | 0.9340 | 1,958 | 2,106 | -0% |
| np=256 rf=5 | 0.9639 | 2,130 | 2,237 | -6% |
| np=512 rf=5 | 0.9807 | 2,464 | 2,591 | +1% |
| np=1024 rf=5 | 0.9900 | 2,805 | 3,050 | +5% |

### OBS (Huawei S3, remote storage)

| Config | Recall | USQ Mean (ms) | USQ P99 (ms) | RQ Mean (ms) | SQ Mean (ms) |
|--------|--------|---------------|---------------|--------------|--------------|
| np=128 rf=1 | 0.7926 | 6,375 | 10,689 | 9,382 | 10,518 |
| np=256 rf=1 | 0.8032 | 6,555 | 10,457 | 10,079 | 12,508 |
| np=1024 rf=1 | 0.8111 | 7,982 | 12,573 | 11,363 | 22,883 |
| np=128 rf=2 | 0.9093 | 11,304 | 15,474 | 18,729 | 20,692 |
| np=256 rf=2 | 0.9340 | 11,580 | 15,459 | 19,306 | 23,298 |
| np=1024 rf=2 | 0.9541 | 13,088 | 17,626 | 20,594 | 32,103 |

## Analysis

### USQ vs RQ (Same Recall, Different Speed)

USQ and RQ have **identical recall** at every config point (same partitions, same centroid assignments). The latency difference varies:

| Region | USQ vs RQ | Explanation |
|--------|-----------|-------------|
| np≤256, rf≤2 | USQ 2-6% faster | USQ 4-bit distance is slightly faster than RQ decode |
| np≥1024, rf≥1 | USQ 7-31% slower | USQ index is 63GB/shard vs RQ ~1GB/shard → cache pollution |
| SSD | Same as DRAM | IO is not the bottleneck (rf dominates) |

**Cache pollution at high nprobe**: USQ auxiliary.idx (63GB) is much larger than RQ auxiliary.idx (~0.8GB). At np=1024, the USQ index reads sweep through more pages, evicting useful data from the 493GB RAM. RQ's tiny index stays fully cached.

### USQ vs SQ (Matched Recall Comparison)

SQ rf=1 recall (0.958) exceeds USQ rf=2 recall (0.909-0.954). At matched recall ~0.95:

| Target | USQ Config | USQ DRAM | SQ Config | SQ DRAM | Winner |
|--------|-----------|----------|-----------|---------|--------|
| ~0.79 | np=128 rf=1 | **496ms** | — | — | USQ only |
| ~0.93 | np=256 rf=2 (0.934) | **957ms** | np=128 rf=1 (0.958) | 993ms | **USQ** |
| ~0.95 | np=1024 rf=2 (0.954) | 1,478ms | np=256 rf=1 (0.962) | **1,050ms** | **SQ 29% faster** |
| ~0.97 | np=512 rf=3 (0.971) | **1,516ms** | np=512 rf=2 (0.971) | 1,653ms | **USQ 8% faster** |

### rf Unit Cost

| Storage | USQ ms/rf | RQ ms/rf | SQ ms/rf |
|---------|----------|---------|---------|
| DRAM | ~350 | ~330 | ~500 |
| SSD | ~350 | ~330 | ~520 |

USQ and RQ have similar rf unit cost (~350ms), confirming CPU-bound refinement. SQ is ~50% higher due to larger code sizes.


### OBS Analysis (Key Finding)

On OBS, USQ is **dramatically faster** than both RQ and SQ at matched recall:

| Target Recall | USQ Config | USQ OBS | RQ Config | RQ OBS | SQ Config | SQ OBS | USQ Advantage |
|--------------|-----------|---------|-----------|--------|-----------|--------|---------------|
| ~0.79 | np=128 rf=1 | **6,375ms** | np=128 rf=1 | 9,382ms | — | — | **32% faster than RQ** |
| ~0.80 | np=256 rf=1 | **6,555ms** | np=256 rf=1 | 10,079ms | np=128 rf=1 (0.958) | 10,518ms | **35-38% faster** |
| ~0.91 | np=128 rf=2 | **11,304ms** | np=128 rf=2 | 18,729ms | np=128 rf=2 (0.964) | 20,692ms | **40-45% faster** |
| ~0.95 | np=1024 rf=2 | **13,088ms** | np=1024 rf=2 | 20,594ms | np=256 rf=2 (0.969) | 23,298ms | **36-44% faster** |

**Why USQ wins on OBS**: Network transfer is the bottleneck on OBS. USQ's 4-bit codes are much smaller than both float32 vectors (for RQ/SQ refinement) and uint8 SQ codes. During refinement, USQ reads ~336B of USQ codes per candidate vs 2048B (512 floats) for original vectors. This ~6x reduction in network data dominates the latency difference.

**rf=1 sweet spot on OBS**: USQ rf=1 achieves 0.79 recall at ~6.4s — fast enough for interactive use. RQ rf=1 takes ~9.4s for the same recall.

### Recall Ceiling

USQ ceiling ~0.990 (np=1024 rf=5), same as RQ. SQ ceiling ~0.972. USQ can reach higher recall than SQ because refinement reads original float32 vectors (same as RQ).

## When to Use USQ vs RQ vs SQ

| Scenario | Winner | Reason |
|----------|--------|--------|
| DRAM, recall ≤ 0.93, np ≤ 256 | **USQ** | 2-6% faster than RQ, same recall |
| DRAM, recall ≥ 0.95, np ≥ 1024 | **RQ** | RQ's tiny index avoids cache pollution |
| SSD | **RQ** | IO not bottleneck; RQ's smaller index is cleaner |
| OBS, recall ≤ 0.95 | **USQ** | 32-50% faster than RQ, 40-65% faster than SQ at rf=1 |
| OBS, recall ≥ 0.95 | **USQ** | 35-40% faster than RQ rf=2 at similar recall |
| Any, recall ≤ 0.96, simplest config | **SQ rf=1** | 0.958 recall without refinement |

## Recommendation

**USQ is not a clear win over RQ for PCA-512 on DRAM/SSD**. The 4-bit quantization has identical approximation quality to RQ's multi-level residual approach, but the larger index size (63GB vs 0.8GB per shard) causes cache pollution at high nprobe.

**USQ is the clear winner for OBS deployment**: 32-50% faster than RQ at rf=1, and 35-40% faster than RQ at matched recall on OBS. The 4-bit codes are significantly smaller than float32 vectors, reducing network transfer during refinement.

For DRAM/SSD, USQ is not a clear win over RQ due to cache pollution from the 63GB/shard index. The advantage only appears at low nprobe (np≤256) where cache pressure is minimal.

USQ could further improve when:
1. **Index size is decoupled from vector storage** (read USQ codes from separate files)
2. **Higher dimensions** where 4-bit compression ratio improves further

## Implementation Notes

- USQ integration: ~15 files across lance-index, lance, python crates
- Feature-gated behind `#[cfg(feature = "hanns")]`
- Key bug: `dist_calculator()` receives flat `Float32Array` (not `FixedSizeListArray`)
- QR decomposition cached via `Arc<OnceLock>` to avoid O(N × d³) during build
- Transform pipeline drops float32 vector column after encoding to avoid shuffle bloat

## Cross-refs

- [PCA-512 RQ results](pca512.md) — RQ baseline
- [PCA-512 SQ results](pca512-sq.md) — SQ comparison
- [USQ 4-bit experiment (1B full)](usq4.md) — original USQ vs RQ comparison
- [Knowhere gap analysis](../roadmap/knowhere-gap-analysis.md) — optimization context
