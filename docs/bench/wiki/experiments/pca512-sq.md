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

1. **SQ is slower everywhere — both DRAM and OBS.** This is the opposite of what we expected.

2. **SQ index is larger, not smaller**: 466GB/shard (SQ) vs 386GB/shard (RQ). SQ stores uint8 codes for every dimension (512B/vector = 100GB/shard) ON TOP of the original float32 vectors. RQ stores only ~4B/vector of residual codes. **SQ does not replace the original vectors** — refine_factor still reads them.

3. **Recall ceiling at 0.972**: SQ quantization error plateaus at rf≥2. The 8-bit per-dimension quantization is more accurate than RQ at low rf (0.958 vs 0.790 at rf=1), but hits its ceiling earlier.

4. **DRAM 2-3x slower**: SQ index (100GB/shard) exceeds page cache capacity. The system is IO-bound reading SQ codes + original vectors from disk.

5. **OBS 10-100% slower**: Larger index = more S3 GET requests = more RTT. np=1024 rf=1 went from 11.4s to 22.9s because the SQ index is 2.3TB total (vs ~66GB for RQ).

6. **Root cause**: Lance's IVF_SQ stores SQ codes alongside original float32 vectors. The SQ codes are NOT used as a replacement for original vectors during refinement. refine_factor still reads the full float32 vectors. So the total data read = SQ codes (for initial scan) + float32 vectors (for refinement), which is MORE than RQ (which only reads RQ codes + float32 vectors for refinement).

## Why SQ8 Should Be Better (In Theory)

SQ8's advantage is when used as a **fast rerank layer WITHIN the index**, replacing the need to read original float32 vectors:
- SQ8 distance: uint8 arithmetic, ~10x faster than float32
- SQ8 storage: 512 bytes/vector, not 2KB (float32) or 4KB (1024-dim float32)

But Lance's current implementation doesn't do this. It stores SQ codes AND original vectors, using SQ only for the initial scan.

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
