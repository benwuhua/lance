---
tags: [experiment, sq, scalar-quantization]
date: 2026-04-08
---

# PCA-512 + IVF_SQ Experiment

## Goal

Test whether IVF_SQ (Scalar Quantization, 8-bit per dimension) on PCA-512 data can reduce OBS latency by reducing index size and network transfer.

## Setup

- **Dataset**: Same PCA-512 vectors as [pca512](pca512.md), 5 shards x 194M rows, 512-dim
- **Index**: IVF_SQ, 13,107 partitions/shard (same as RQ), 8-bit scalar quantization
- **SQ shard size**: 466GB/shard (vs RQ 386GB/shard — SQ index ~100GB/shard on top of raw vectors)
- **GT**: Same `gt_positional_v2.npz` (positional indices valid for both RQ and SQ)
- **Local**: `/data/work/tmp/s1b-pca512-sq/`
- **OBS**: `s3://knowledgebase-5f43/pca512-sq-1b/`

## Results

### DRAM

| Config | SQ Recall | SQ Mean (ms) | RQ Recall | RQ Mean (ms) |
|--------|-----------|-------------|-----------|-------------|
| np=128 rf=1 | 0.958 | 993 | 0.790 | 523 |
| np=256 rf=1 | 0.962 | 1050 | 0.793 | 636 |
| np=512 rf=1 | 0.964 | 1684 | — | — |
| np=1024 rf=1 | 0.965 | 2479 | 0.794 | 846 |
| np=128 rf=2 | 0.964 | 1085 | 0.942 | 875 |
| np=256 rf=2 | 0.969 | 1234 | 0.948 | 977 |
| np=512 rf=2 | 0.971 | 1653 | — | — |
| np=1024 rf=2 | 0.972 | 2448 | 0.953 | 1183 |
| np=1024 rf=3 | 0.972 | 2805 | 0.967 | 1563 |
| np=1024 rf=5 | 0.972 | 3575 | 0.971 | 2494 |

### OBS

| Config | SQ Mean (ms) | RQ Mean (ms) | Delta |
|--------|-------------|-------------|-------|
| np=128 rf=1 | 10,518 | 9,382 | +12% |
| np=256 rf=1 | 12,508 | 10,079 | +24% |
| np=1024 rf=1 | 22,883 | 11,363 | +101% |
| np=128 rf=2 | 20,692 | 18,729 | +10% |
| np=256 rf=2 | 23,298 | 19,306 | +21% |
| np=1024 rf=2 | 32,103 | 20,594 | +56% |

## Findings

### Matched-Recall Comparison (OBS, the key insight)

SQ rf=1 recall (0.958-0.965) already exceeds RQ rf=2 recall (0.942-0.953).
On OBS, **SQ rf=1 is 35-44% faster** than RQ rf=2 at the same recall level:

| Target Recall | SQ Config | SQ OBS | RQ Config | RQ OBS | SQ Advantage |
|--------------|-----------|--------|-----------|--------|-------------|
| ~0.95 | np=128 rf=1 (0.958) | **10,518ms** | np=128 rf=2 (0.942) | 18,729ms | **44% faster** |
| ~0.96 | np=256 rf=1 (0.962) | **12,508ms** | np=256 rf=2 (0.948) | 19,306ms | **35% faster** |
| ~0.97 | np=1024 rf=2 (0.972) | 32,103ms | np=1024 rf=2 (0.953) | 20,594ms | -56% (SQ worse) |

**Why**: SQ 8-bit quantization is more accurate than RQ 1-bit residual, so SQ rf=1 doesn't need refinement. On OBS, rf=2 downloads 20,000 original float32 vectors (~10s). SQ rf=1 skips this entirely — only reads SQ uint8 codes.

### Matched-rf Comparison (less favorable for SQ)

At same rf, SQ is slower because the SQ index is larger:
- SQ index: 466GB/shard (SQ codes ~100GB + raw vectors ~386GB)
- RQ index: 386GB/shard (RQ codes ~0.8GB + raw vectors ~386GB)
- Larger index → more S3 GET requests → more RTT overhead

### DRAM

SQ is 2-3x slower at matched rf because the 100GB SQ index per shard exceeds page cache (493GB total RAM, 5×100GB = 500GB SQ codes alone). But at matched recall, SQ rf=1 (993ms at 0.958) vs RQ rf=2 (875ms at 0.942) — SQ is only slightly slower on DRAM but has higher recall.

### Recall Ceiling

SQ recall plateaus at 0.972 at rf≥2. This is a hard ceiling from the 8-bit quantization error. RQ continues improving past 0.97 with higher rf (RQ np=1024 rf=3 = 0.967, rf=5 = 0.971).

## When to Use SQ vs RQ

| Scenario | Winner | Reason |
|----------|--------|--------|
| OBS, recall ≤ 0.96 | **SQ** | rf=1 sufficient, saves ~10s of vector download |
| OBS, recall > 0.97 | **RQ** | SQ can't reach 0.99+, RQ with rf=3+ can |
| DRAM | **RQ** | RQ index fits in page cache, SQ doesn't |
| SSD | **RQ** | Same reason as DRAM |

## Recommendation

IVF_SQ in its current form is **not useful** for this benchmark. The right approach (from [optimization-roadmap](../roadmap/optimization-roadmap.md) P1) is to implement **SQ8 two-stage rerank within IVF_RQ**:
1. Use RQ codes for initial scan (compact, fast)
2. Use SQ8 codes for refinement (replaces float32 vector reads)
3. Store SQ8 alongside RQ codes in the index file

This would eliminate the need to read original vectors for refinement, reducing both IO and computation.

## Cross-refs

- [PCA-512 RQ results](pca512.md) — the comparison baseline
- [Optimization roadmap](../roadmap/optimization-roadmap.md) — SQ8 two-stage rerank proposal
- [Knowhere gap analysis](../roadmap/knowhere-gap-analysis.md) — how Knowhere uses SQ8
