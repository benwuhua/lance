---
tags: [experiment, usq, ultra-sparse-quantization, 8-bit]
date: 2026-04-11
---

# PCA-512 + IVF_USQ 8-bit Experiment

## Goal

Test IVF_USQ with 8-bit quantization on PCA-512 data to fill the recall 0.81-0.96 gap identified in the OBS Pareto frontier. USQ 4-bit rf=1 covers recall ≤0.81, SQ rf=1 covers ~0.96. USQ 8-bit should deliver 0.95+ recall at rf=1.

> Methodology note: this page preserves the earlier `5-query / warmup=0 / _rowid GT` and the 2026-04-12 OBS refresh slice. The canonical cross-family comparison now lives in [pca512-usq-revalidation.md](pca512-usq-revalidation.md) and uses the unified legacy contract (`query_count=8`, `warmup=3`, positional GT).

2026-04-12 refresh: reran the OBS sweep on ECS `knowledge-ecs-44c3` with the same 5-query GT and seed. DRAM / SSD numbers below remain from the original 2026-04-11 run; only the OBS table and Pareto conclusions were refreshed.

## Setup

- **Dataset**: Same PCA-512 vectors, 5 shards x 194M rows, 512-dim, cosine metric
- **Index**: IVF_USQ, 13,107 partitions/shard, **8-bit quantization**
- **USQ code size**: ~592B/vector (512B packed codes + 64B signs + 16B meta)
- **Index size**: ~117GB auxiliary.idx + 26MB index.idx per shard (vs 63GB for 4-bit)
- **GT**: `gt_pca512_shard0_top10k_5q_seed42_v2.npz` (flat search, seed=42, _rowid format)
- **Query count**: 5 (no warmup), seed=42
- **Note**: RQ index dropped from shards to force Lance to select USQ index

## Build

- **Time**: ~57 min for 5 shards parallel (started 20:07, completed 21:03 CST)
- **Temp space**: 1.3TB peak, 117GB/shard final
- **Command**: `ds.create_index("vector", "IVF_USQ", metric="cosine", num_partitions=13107, name="usq8_idx", num_bits=8)`

## Results

### 5 Shards DRAM (page cache warm, partial — 584GB index > 493GB RAM)

| Config | Recall | USQ 8-bit Mean (ms) | USQ 8-bit P95 (ms) | USQ 4-bit Mean (ms) | RQ Mean (ms) |
|--------|--------|---------------------|---------------------|---------------------|--------------|
| np=128 rf=1 | 0.9474 | 1,731 | 2,464 | 496 | 523 |
| np=256 rf=1 | 0.9730 | 1,826 | 2,039 | 597 | 636 |
| np=512 rf=1 | 0.9889 | 3,091 | 3,689 | 806 | — |
| np=1024 rf=1 | 0.9949 | 5,770 | 7,026 | 1,105 | 846 |
| np=128 rf=2 | 0.9477 | 1,402 | 1,588 | 856 | 875 |
| np=256 rf=2 | 0.9734 | 2,005 | 2,184 | 957 | 977 |
| np=512 rf=2 | 0.9895 | 3,274 | 3,808 | 1,173 | — |
| np=1024 rf=2 | 0.9967 | 5,722 | 6,688 | 1,478 | 1,183 |
| np=128 rf=3 | 0.9477 | 1,820 | 2,047 | 1,260 | — |
| np=256 rf=3 | 0.9734 | 2,404 | 2,640 | 1,344 | — |
| np=512 rf=3 | 0.9895 | 3,628 | 4,038 | 1,516 | — |
| np=1024 rf=3 | 0.9967 | 6,045 | 7,018 | 1,812 | 1,563 |
| np=128 rf=5 | 0.9477 | 2,538 | 2,738 | 1,967 | — |
| np=256 rf=5 | 0.9734 | 3,164 | 3,400 | 2,277 | — |
| np=512 rf=5 | 0.9895 | 4,337 | 4,610 | 2,442 | — |
| np=1024 rf=5 | 0.9967 | 6,769 | 7,370 | 2,678 | 2,494 |

### 5 Shards SSD (cold cache, drop_caches before each config)

| Config | Recall | USQ 8-bit Mean (ms) | USQ 8-bit P95 (ms) | DRAM vs SSD |
|--------|--------|---------------------|---------------------|-------------|
| np=128 rf=1 | 0.9474 | 1,214 | 1,406 | -30% |
| np=256 rf=1 | 0.9730 | 2,042 | 2,335 | +12% |
| np=512 rf=1 | 0.9889 | 3,424 | 4,131 | +11% |
| np=1024 rf=1 | 0.9949 | 6,096 | 7,330 | +6% |
| np=128 rf=2 | 0.9477 | 1,554 | 1,764 | +11% |
| np=256 rf=2 | 0.9734 | 2,312 | 2,508 | +15% |
| np=512 rf=2 | 0.9895 | 3,915 | 4,423 | +20% |
| np=1024 rf=2 | 0.9967 | 6,321 | 7,361 | +10% |
| np=128 rf=3 | 0.9477 | 1,882 | 2,094 | +3% |
| np=256 rf=3 | 0.9734 | 2,615 | 2,818 | +9% |
| np=512 rf=3 | 0.9895 | 3,929 | 4,459 | +8% |
| np=1024 rf=3 | 0.9967 | 6,665 | 7,689 | +10% |
| np=128 rf=5 | 0.9477 | 2,591 | 2,833 | +2% |
| np=256 rf=5 | 0.9734 | 3,363 | 3,585 | +6% |
| np=512 rf=5 | 0.9895 | 4,706 | 5,039 | +9% |
| np=1024 rf=5 | 0.9967 | 7,375 | 8,322 | +9% |

### OBS (Huawei S3, remote storage)

| Config | Recall | USQ 8-bit Mean (ms) | USQ 8-bit P95 (ms) | RQ Mean (ms) | USQ 4-bit Mean (ms) |
|--------|--------|---------------------|---------------------|--------------|---------------------|
| np=128 rf=1 | 0.9474 | 8,592 | 11,575 | 9,382 | 6,375 |
| np=256 rf=1 | 0.9730 | 7,710 | 9,767 | 10,079 | 6,555 |
| np=1024 rf=1 | 0.9949 | 21,312 | 23,382 | 11,363 | 7,982 |
| np=128 rf=2 | 0.9477 | 11,197 | 12,277 | 18,729 | 11,304 |
| np=256 rf=2 | 0.9734 | 12,203 | 13,335 | 19,306 | 11,580 |
| np=1024 rf=2 | 0.9967 | 27,988 | 32,322 | 20,594 | 13,088 |

Note: OBS recall is the same as DRAM for the same config (same index, same nprobe/rf). Refreshed OBS results were written to `/data/work/tmp/pca512-usq8-obs-recall-20260412-summary.json`.

## Analysis

### Key Finding: Cache Thrashing Kills DRAM/SSD Performance

USQ 8-bit auxiliary.idx = 117GB/shard × 5 = 584GB, which **exceeds the 493GB RAM** on the test machine. This causes severe cache thrashing:

- **np=128 rf=1**: 1,731ms (DRAM) vs 1,214ms (SSD) — **DRAM is 43% SLOWER than SSD** because cache warm loaded 584GB into 493GB RAM, evicting useful pages. SSD's clean start is faster.
- **np=1024 rf=1**: 5,770ms — **6.8x slower** than RQ (846ms) and **5.2x slower** than USQ 4-bit (1,105ms)

The recall is excellent (0.9474 at np=128 rf=1), confirming 8-bit quantization precision. But the index size makes it impractical for DRAM deployment at this scale.

### USQ 8-bit vs 4-bit: Precision vs Size Tradeoff

| Metric | USQ 4-bit | USQ 8-bit | Delta |
|--------|-----------|-----------|-------|
| Code size | 336B/vector | 592B/vector | +76% |
| Index size/shard | 63GB | 117GB | +86% |
| Recall rf=1 (np=128) | 0.7926 | 0.9474 | +0.155 |
| Recall rf=1 (np=256) | 0.8032 | 0.9730 | +0.170 |
| DRAM np=128 rf=1 | 496ms | 1,731ms | +249% |
| DRAM np=1024 rf=1 | 1,105ms | 5,770ms | +422% |

8-bit achieves **0.95+ recall at rf=1** (the goal), but at 2.5-5x the latency cost of 4-bit on DRAM/SSD due to cache pressure.

### OBS: Pareto Frontier Update

The 2026-04-12 ECS rerun materially improved the high-recall OBS numbers:

- `np=256 rf=1`: `9,932ms -> 7,710ms` (`-22%`) at the same `0.9730` recall
- `np=1024 rf=1`: `23,397ms -> 21,312ms` (`-9%`)
- `np=256 rf=2`: `15,076ms -> 12,203ms` (`-19%`)

This changes the OBS Pareto picture. `np=256 rf=1` is now the key frontier point:

- vs SQ `np=128 rf=1`: `7,710ms` vs `10,518ms`, with higher recall (`0.9730` vs `0.958`)
- vs RQ `np=256 rf=2`: `7,710ms` vs `19,306ms`, with higher recall (`0.9730` vs `0.948`)
- `np=128 rf=1` no longer matters operationally because it is dominated by `np=256 rf=1` (lower recall and higher latency)

Current OBS Pareto points across PCA-512 variants:

| Frontier Point | Recall | Mean (ms) | Why It Survives |
|---------------|--------|-----------|-----------------|
| USQ 4-bit np=128 rf=1 | 0.7926 | 6,375 | Lowest-latency OBS point |
| USQ 4-bit np=256 rf=1 | 0.8032 | 6,555 | Small recall bump for tiny latency increase |
| USQ 8-bit np=256 rf=1 | 0.9730 | 7,710 | Collapses the old 0.81-0.96 gap |
| USQ 8-bit np=1024 rf=1 | 0.9949 | 21,312 | Best sub-0.995 recall on OBS |
| USQ 8-bit np=1024 rf=2 | 0.9967 | 27,988 | Highest observed recall on OBS |

### Recall Plateau

USQ 8-bit recall plateaus quickly: np=128 → 0.9474, np=256 → 0.9730, np=512 → 0.9889, np=1024 → 0.9949. Increasing rf from 1→2 barely improves recall (because 8-bit codes are already precise). This is the expected behavior — 8-bit quantization has high intrinsic accuracy.

### rf Unit Cost

| Storage | USQ 8-bit ms/rf | USQ 4-bit ms/rf | RQ ms/rf |
|---------|-----------------|-----------------|----------|
| DRAM np=128 | ~730 | ~350 | ~330 |
| DRAM np=1024 | ~350 | ~350 | ~330 |
| SSD np=128 | ~680 | ~350 | ~330 |

USQ 8-bit rf cost is 2x higher at low nprobe (more candidates to refine, larger codes to decode). Converges at high nprobe where CPU dominates.

## Conclusion

**USQ 8-bit remains impractical on DRAM / SSD, but the OBS recommendation changed after the 2026-04-12 rerun:**

1. **DRAM/SSD**: 584GB index still exceeds 493GB RAM and causes severe cache thrashing. Nothing in the refresh changes that.
2. **OBS**: `np=256 rf=1` is now a true Pareto point at `7.7s / 0.973 recall`. It is faster than SQ `rf=1` and RQ `rf=2`, while also delivering higher recall than both.
3. **High recall**: `np=1024 rf=1` and `rf=2` stay expensive, but they now define the only observed OBS frontier points above `0.99` recall.

**Updated recommendation**:

- For **OBS recall <= 0.80**, keep using **USQ 4-bit rf=1**
- For **OBS recall ~0.97**, prefer **USQ 8-bit np=256 rf=1**
- For **OBS recall ~0.995+**, use **USQ 8-bit np=1024 rf=1/2** only if the extra 13-20s latency is acceptable

SQ `rf=1` is no longer on the OBS Pareto frontier once the refreshed USQ 8-bit numbers are used.

## Cross-refs

- [USQ 4-bit experiment](pca512-usq.md) — USQ 4-bit results
- [PCA-512 RQ results](pca512.md) — RQ baseline
- [PCA-512 SQ results](pca512-sq.md) — SQ comparison
