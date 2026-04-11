# 1B Vector Search Benchmark Report

> Dataset: 972M rows, FineWeb-Edu embeddings, 1024-dim, cosine
> Date: 2026-04-09
> Environment: Huawei Cloud ECS (493GB RAM, 12TB NVMe, OBS S3)

---

## 1. Architecture

### 1.1 Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Data Preparation Pipeline                              │
│                                                                                 │
│  FineWeb-Edu ──┐                                                                │
│  (324M rows)   │  replicate 3×  +  Gaussian noise (σ=0.01)  +  L2 normalize    │
│  1024-dim      ├──────────────────────────────────────────────────────────┐      │
│  float32       │                                                         │      │
│                └─────────────────────┐                                   │      │
│                                      ▼                                   ▼      │
│                           ┌──────────────────┐                  ┌──────────────┐│
│                           │   1B Baseline     │                  │   PCA-512    ││
│                           │   972M × 1024-d   │    SVD(512)     │  972M × 512d ││
│                           │                   │◄─────────────────│              ││
│                           │   5 × 194M shards │                  │ 5 × 194M     ││
│                           └────────┬─────────┘                  └──────┬───────┘│
│                                    │                                   │        │
│                         build_index │                     build_index  │        │
│                           IVF_RQ    │                        ┌─────────┤        │
│                           13,107 pt │                        │         │        │
│                                    ▼                        ▼         ▼        │
│                           ┌──────────────────┐    ┌──────────────┐ ┌─────────┐ │
│                           │   IVF_RQ          │    │   IVF_RQ     │ │ IVF_SQ  │ │
│                           │   5×386GB/shard   │    │   5×77GB/sh  │ │ 5×466GB │ │
│                           │   RQ codes ~0.8GB │    │   RQ ~0.4GB  │ │ SQ~100GB│ │
│                           │   + raw vectors   │    │   + raw vecs │ │ +raw vec│ │
│                           └──────────────────┘    └──────────────┘ └─────────┘ │
│                                                                   │         │
│                                              ┌──────────────────┐ │         │
│                                              │ IVF_USQ (4-bit)  │ │         │
│                                              │ 5×256GB/shard    │ │         │
│                                              │ USQ codes ~63GB  │ │         │
│                                              │ + raw vectors    │ │         │
│                                              └──────────────────┘ │         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Query Path (Plan B: Sharded Parallel)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         Query Execution Flow                                   │
│                                                                               │
│  Query Vector ─────────┐                                                      │
│  (512-dim, cosine)     │                                                      │
│                        ▼                                                      │
│              ┌──── ThreadPoolExecutor (5 workers) ────┐                        │
│              │                                         │                        │
│              │  ┌─────────┐  ┌─────────┐       ┌─────┐│                        │
│              │  │ Worker0 │  │ Worker1 │  ...  │ W4  ││                        │
│              │  └────┬────┘  └────┬────┘       └──┬──┘│                        │
│              └───────┼────────────┼───────────────┼───┘                        │
│                      ▼            ▼               ▼                            │
│              ┌─────────────────────────────────────────┐                       │
│              │         Per-Shard Search Pipeline        │                       │
│              │                                           │                       │
│              │  ┌─────────────┐    ┌──────────────────┐  │                       │
│              │  │ 1. IVF Probe │    │  Storage Backend │  │                       │
│              │  │  nprobe=N    │───►│  DRAM / SSD / S3 │  │                       │
│              │  │  of 13,107   │    └──────────────────┘  │                       │
│              │  └──────┬──────┘                          │                       │
│              │         │ ~14,800 candidates/partition     │                       │
│              │         ▼                                   │                       │
│              │  ┌─────────────┐    ┌──────────────────┐  │                       │
│              │  │ 2. Distance  │    │  RQ codes (0.8GB)│  │                       │
│              │  │  Compute     │◄───│  or SQ codes(100G)│ │                       │
│              │  │  approximate │    └──────────────────┘  │                       │
│              │  └──────┬──────┘                          │                       │
│              │         │ top-K×rf candidates              │                       │
│              │         ▼                                   │                       │
│              │  ┌─────────────┐    ┌──────────────────┐  │                       │
│              │  │ 3. Refine    │    │  Original vectors│  │                       │
│              │  │  (rf > 1)    │───►│  float32 (386GB) │  │                       │
│              │  │  re-rank     │    │  skip if SQ rf=1 │  │                       │
│              │  └──────┬──────┘    └──────────────────┘  │                       │
│              │         │ top-K results per shard           │                       │
│              └─────────┼──────────────────────────────────┘                       │
│                        │                                                       │
│                        ▼                                                       │
│              ┌──────────────────┐                                               │
│              │    Merge          │                                               │
│              │    5×10K results  │                                               │
│              │    sort by dist   │                                               │
│              │    take top-10K   │                                               │
│              │    (~1ms)         │                                               │
│              └────────┬─────────┘                                               │
│                       ▼                                                        │
│               Final Top-10K                                                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Index Internal Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IVF Index Layout (per shard, 194M rows)                   │
│                                                                             │
│  Centroid Index (IVF)                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │ 13,107 centroids (512-dim float32)  │  ← used to find top-nprobe        │
│  │ ~26MB                               │    partitions per query            │
│  └─────────────────────────────────────┘                                    │
│          │                                                                  │
│          │  partition assignment (Voronoi cell)                              │
│          ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Partition (×13,107)                               │    │
│  │  ~14,800 vectors each                                               │    │
│  │                                                                     │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │    │
│  │  │ IVF_RQ            │  │ IVF_RQ            │  │ IVF_SQ           │  │    │
│  │  │                  │  │ (PCA-512)        │  │ (PCA-512)        │  │    │
│  │  │ RQ codes:        │  │ RQ codes:        │  │ SQ codes:        │  │    │
│  │  │ 4 levels × 1byte │  │ 4 levels × 1byte │  │ 512 bytes/vec    │  │    │
│  │  │ = 4B / vector    │  │ = 4B / vector    │  │ = 512B / vector  │  │    │
│  │  │                  │  │                  │  │                  │  │    │
│  │  │ Total: ~0.8GB    │  │ Total: ~0.4GB    │  │ Total: ~100GB    │  │    │
│  │  │                  │  │                  │  │                  │  │    │
│  │  │ rf=1: 0.79 recall│  │ rf=1: 0.79 recall│  │ rf=1: 0.96 recall│  │    │
│  │  │ rf=2: 0.98 recall│  │ rf=2: 0.95 recall│  │ rf=2: 0.97 recall│  │    │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │    │
│  │  ┌──────────────────┐                                              │    │
│  │  │ IVF_USQ (4-bit)  │  ← NEW: Hanns 4-bit quantization           │    │
│  │  │ USQ codes:       │                                              │    │
│  │  │ 256B packed      │                                              │    │
│  │  │ + 64B signs      │                                              │    │
│  │  │ + 16B meta       │                                              │    │
│  │  │ = 336B / vector  │                                              │    │
│  │  │                  │                                              │    │
│  │  │ Total: ~63GB     │  (large! but 6x smaller than float32)       │    │
│  │  │                  │                                              │    │
│  │  │ rf=1: 0.79 recall│  (same as RQ)                               │    │
│  │  │ rf=2: 0.95 recall│                                              │    │
│  │  │ rf=5: 0.99 recall│  (higher ceiling than RQ 0.97)              │    │
│  │  └──────────────────┘                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│          │                                                                  │
│          │  refine_factor > 1: read original vectors                        │
│          ▼                                                                  │
│  ┌─────────────────────────────────────┐                                    │
│  │ Raw vectors (float32)               │  ← columnar storage (.lance)       │
│  │ 1024-dim or 512-dim                 │    random access via take()        │
│  │ ~386GB / shard (1024-dim)           │    IO bottleneck on OBS            │
│  │ ~193GB / shard (512-dim)            │                                    │
│  └─────────────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Storage Tier Latency Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Query Latency Breakdown                              │
│                                                                         │
│  DRAM (page cache hot)           SSD (cold cache)     OBS (S3)         │
│  ┌────────────────────────┐      ┌─────────────────┐  ┌──────────────┐ │
│  │████                    │      │██████            │  │██████████████│ │
│  │████ CPU (rf)           │      │██████ CPU        │  │██████████████│ │
│  │████ ~350ms/rf          │      │██████ ~350ms/rf  │  │██████████████│ │
│  │████                    │      │██████            │  │████████████  │ │
│  │████                    │      │██████ IO         │  │██████████████│ │
│  │████ IO: ~50-200ms      │      │██████ ~100-500ms │  │██████████████│ │
│  │████ (cache hit)        │      │██████ (NVMe)     │  │████████ S3   │ │
│  │████                    │      │██████            │  │████████ 8-10s │ │
│  │████                    │      │██████            │  │████████ /rf   │ │
│  └────────────────────────┘      └─────────────────┘  └──────────────┘ │
│                                                                         │
│  Bottleneck: CPU (rf)            CPU (rf) + IO       IO (S3 RTT)        │
│  rf unit cost: ~350ms            ~350ms + IO          ~8-10s /rf        │
│  rf=1 → 0.5-1.7s                0.5-3.0s             8-23s             │
│  rf=2 → 0.9-2.5s                0.9-3.3s             17-32s            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.5 IVF Index Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_partitions | 13,107/shard | min(N/1000, 65536)/5 |
| density | ~14,800 vecs/partition | 194M / 13,107 |
| nprobe range | 128-1024 | 1-8% of partitions |
| refine_factor | 1-5 | re-rank top-K×rf with original vectors |

### 1.6 Benchmark Tool

```
benchmarks/cohere/bench.py query \
    --plan B --storage {dram,ssd,obs} \
    --index-type {IVF_RQ,IVF_SQ,IVF_USQ} \
    --shard-dir PATH --nprobes N --refine-factor N \
    --top-k 10000 --num-shards 5 --query-count 8 --warmup 3
```

---

## 2. Index Variants Tested

### 2.1 IVF_RQ (Residual Quantization)

- 4-level residual quantization, 8-bit per level (256 centroids)
- Compressed codes: ~0.8GB/shard (very compact)
- Total shard: ~386GB (RQ codes + raw float32 vectors)
- **Strengths**: Compact codes, recall scales with rf (up to 0.995)
- **Weaknesses**: rf=1 recall only ~0.79 (1-bit residual too coarse), rf>1 reads original float32 vectors

### 2.2 IVF_SQ (Scalar Quantization)

- 8-bit per dimension (float32 → uint8, 4x compression per vector)
- SQ codes: ~100GB/shard (on top of raw vectors)
- Total shard: ~466GB (SQ codes + raw float32 vectors)
- **Strengths**: SQ distance very accurate, rf=1 recall 0.958 (vs RQ 0.790)
- **Weaknesses**: Index exceeds page cache, recall ceiling at 0.972

### 2.3 IVF_USQ (Ultra-Sparse Quantization, 4-bit via Hanns)

- Random rotation → normalize → 4-bit quantize with Hanns approximate scoring
- USQ codes: ~336B/vector (256B packed codes + 64B signs + 16B meta)
- Index auxiliary: ~63GB/shard (much larger than RQ ~0.8GB)
- Total shard: ~256GB/shard (USQ codes ~63GB + raw vectors ~193GB)
- **Strengths**: 4-bit codes ~6x smaller than float32 vectors → massive OBS advantage; rf unit cost same as RQ (~350ms)
- **Weaknesses**: 63GB/shard index causes cache pollution on DRAM at high nprobe; recall identical to RQ (no quality advantage)
- Feature-gated behind `#[cfg(feature = "hanns")]`

### 2.4 Dimensionality Reduction: PCA-512

- TruncatedSVD 1024→512 on 500K sample, explained variance 96.63%
- Applied to RQ, SQ, and USQ variants
- ~2x smaller index, ~2x faster distance computation

---

## 3. Sweep Results

### 3.1 1B Baseline (1024-dim, IVF_RQ)

Index: 5 shards × 386GB = 1.93TB total. Recall measured on shard-0 against flat search GT.

#### DRAM (partial page cache)

| nprobe | rf | Recall | Mean (ms) | P99 (ms) |
|--------|-----|--------|-----------|----------|
| 64 | 1 | 0.79* | 616 | 860 |
| 64 | 2 | — | 840 | 975 |
| 128 | 1 | 0.830 | 626 | 880 |
| 128 | 2 | 0.921 | 904 | 1,011 |
| 256 | 1 | 0.845 | 656 | 786 |
| 256 | 2 | 0.950 | 968 | 1,063 |
| 512 | 1 | — | 985 | 1,047 |
| 512 | 2 | — | 1,262 | 1,338 |
| 1024 | 1 | 0.861 | 1,740 | 1,843 |
| 1024 | 2 | 0.983 | 2,150 | 2,218 |
| 1024 | 3 | 0.993 | 2,513 | 2,631 |
| 1024 | 5 | 0.995 | 3,468 | 3,786 |

*Recall for np<128 estimated from GT subset. Some configs missing recall (ran before GT integration).

#### SSD (cold cache)

| nprobe | rf | Mean (ms) | P99 (ms) | SSD vs DRAM |
|--------|-----|-----------|----------|-------------|
| 64 | 1 | 631 | 888 | +2% |
| 64 | 2 | 889 | 1,054 | +6% |
| 128 | 1 | 677 | 979 | +8% |
| 128 | 2 | 928 | 1,106 | +3% |
| 256 | 1 | 754 | 914 | +15% |
| 256 | 2 | 1,058 | 1,193 | +9% |
| 512 | 1 | 1,168 | 1,231 | +19% |
| 512 | 2 | 1,497 | 1,529 | +19% |
| 1024 | 1 | 2,362 | 2,534 | +36% |
| 1024 | 2 | 2,636 | 2,738 | +23% |
| 1024 | 3 | 2,983 | 3,094 | +19% |
| 1024 | 5 | 3,986 | 4,250 | +15% |

#### OBS (S3)

| nprobe | rf | Mean (ms) | P99 (ms) |
|--------|-----|-----------|----------|
| 64 | 1 | 8,218 | 8,648 |
| 64 | 2 | 17,051 | 20,292 |
| 128 | 1 | 8,403 | 8,946 |
| 128 | 2 | 25,964 | 30,282 |
| 256 | 1 | 8,909 | 9,481 |
| 256 | 2 | 16,395 | 16,955 |
| 512 | 1 | 11,195 | 11,762 |
| 512 | 2 | 19,041 | 21,868 |
| 1024 | 1 | 17,588 | 18,759 |
| 1024 | 2 | 24,984 | 25,744 |

---

### 3.2 PCA-512 (512-dim, IVF_RQ)

Index: 5 shards × ~77GB = 385GB total. 2x smaller index + 2x faster distance computation.

#### DRAM

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | vs 1B Baseline |
|--------|-----|--------|-----------|----------|----------------|
| 128 | 1 | 0.790 | 523 | 537 | **16% faster** |
| 128 | 2 | 0.942 | 875 | 888 | 3% faster |
| 256 | 1 | 0.793 | 636 | 663 | 3% faster |
| 256 | 2 | 0.948 | 977 | 1,006 | same |
| 512 | 1 | 0.794 | 756 | 856 | 23% faster |
| 512 | 2 | 0.951 | 1,091 | 1,167 | 14% faster |
| 1024 | 1 | 0.794 | 846 | 1,035 | **51% faster** |
| 1024 | 2 | 0.953 | 1,183 | 1,315 | **45% faster** |
| 1024 | 3 | 0.967 | 1,563 | 1,694 | **38% faster** |
| 1024 | 5 | 0.971 | 2,494 | 2,609 | 28% faster |

#### OBS

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | vs 1B Baseline OBS |
|--------|-----|--------|-----------|----------|---------------------|
| 128 | 1 | 0.790 | 9,382 | 9,510 | same |
| 128 | 2 | 0.942 | 18,729 | 18,995 | 28% faster |
| 256 | 1 | 0.793 | 10,079 | 10,288 | same |
| 256 | 2 | 0.948 | 19,306 | 19,568 | same |
| 512 | 1 | 0.794 | 10,698 | 11,287 | same |
| 512 | 2 | 0.951 | 20,010 | 20,430 | same |
| 1024 | 1 | 0.794 | 11,363 | 12,046 | **35% faster** |
| 1024 | 2 | 0.953 | 20,594 | 21,375 | 17% faster |

**PCA-512 conclusion**: DRAM improvement significant at high nprobe (up to 51%). OBS improvement only at high nprobe + rf=1 (up to 35%). Recall ceiling ~0.97 from dimension loss.

---

### 3.3 PCA-512 + IVF_SQ (512-dim, IVF_SQ)

Index: 5 shards × 466GB = 2.33TB total (SQ codes ~100GB + raw vectors ~386GB per shard).

#### DRAM

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | RQ DRAM Mean | SQ vs RQ |
|--------|-----|--------|-----------|----------|-------------|----------|
| 128 | 1 | **0.958** | 993 | 1,102 | 523 | +90% |
| 256 | 1 | **0.962** | 1,050 | 1,289 | 636 | +65% |
| 512 | 1 | **0.964** | 1,684 | 2,129 | 756 | +123% |
| 1024 | 1 | **0.965** | 2,479 | 3,020 | 846 | +193% |
| 128 | 2 | **0.964** | 1,085 | 1,215 | 875 | +24% |
| 256 | 2 | **0.969** | 1,234 | 1,497 | 977 | +26% |
| 512 | 2 | **0.971** | 1,653 | 1,883 | 1,091 | +52% |
| 1024 | 2 | **0.972** | 2,448 | 2,815 | 1,183 | +107% |
| 1024 | 3 | 0.972 | 2,805 | 3,411 | 1,563 | +80% |
| 1024 | 5 | 0.972 | 3,575 | 4,415 | 2,494 | +43% |

#### SSD (cold cache)

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | SQ DRAM | SSD vs DRAM |
|--------|-----|--------|-----------|----------|---------|-------------|
| 128 | 1 | 0.958 | 949 | 1,001 | 993 | -5% |
| 256 | 1 | 0.962 | 1,275 | 1,442 | 1,050 | +21% |
| 512 | 1 | 0.964 | 1,813 | 2,136 | 1,684 | +8% |
| 1024 | 1 | 0.965 | 3,027 | 3,483 | 2,479 | +22% |
| 128 | 2 | 0.964 | 1,190 | 1,340 | 1,085 | +10% |
| 256 | 2 | 0.969 | 1,477 | 1,746 | 1,234 | +20% |
| 512 | 2 | 0.971 | 2,005 | 2,358 | 1,653 | +21% |
| 1024 | 2 | 0.972 | 3,251 | 3,951 | 2,448 | +33% |
| 1024 | 3 | 0.972 | 3,565 | 4,352 | 2,805 | +27% |
| 1024 | 5 | 0.972 | 4,313 | 5,216 | 3,575 | +21% |

#### OBS

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | RQ OBS | SQ vs RQ |
|--------|-----|--------|-----------|----------|--------|----------|
| 128 | 1 | 0.958 | 10,518 | 11,031 | 9,382 | +12% |
| 256 | 1 | 0.962 | 12,508 | 13,935 | 10,079 | +24% |
| 1024 | 1 | 0.965 | 22,883 | 25,807 | 11,363 | +101% |
| 128 | 2 | 0.964 | 20,692 | 24,037 | 18,729 | +10% |
| 256 | 2 | 0.969 | 23,298 | 26,854 | 19,306 | +21% |
| 1024 | 2 | 0.972 | 32,103 | 34,735 | 20,594 | +56% |

---

### 3.4 PCA-512 + IVF_USQ (512-dim, IVF_USQ)

Index: 5 shards × ~256GB = 1.28TB total. USQ codes ~63GB/shard on top of raw vectors. Uses same PCA-512 data as RQ/SQ.

#### DRAM

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | RQ Mean | SQ Mean | USQ vs RQ |
|--------|-----|--------|-----------|----------|---------|---------|-----------|
| 128 | 1 | 0.793 | **496** | 612 | 523 | 993 | -5% |
| 256 | 1 | 0.803 | **597** | 747 | 636 | 1,050 | -6% |
| 512 | 1 | 0.808 | **806** | 1,037 | — | 1,684 | — |
| 1024 | 1 | 0.811 | 1,105 | 1,507 | **846** | 2,479 | +31% |
| 128 | 2 | 0.909 | **856** | 980 | 875 | 1,085 | -2% |
| 256 | 2 | 0.934 | **957** | 1,110 | 977 | 1,234 | -2% |
| 512 | 2 | 0.947 | **1,173** | 1,398 | — | 1,653 | — |
| 1024 | 2 | 0.954 | 1,478 | 1,975 | **1,183** | 2,448 | +25% |
| 1024 | 3 | 0.980 | 1,812 | 2,211 | **1,563** | 2,805 | +16% |
| 1024 | 5 | 0.990 | 2,678 | 2,941 | **2,494** | 3,575 | +7% |

> USQ faster than RQ at np≤256 (2-6%) due to 4-bit distance speed. USQ slower at np≥1024 (7-31%) because 63GB auxiliary.idx causes cache pollution in 493GB RAM.

#### SSD (cold cache)

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | SSD vs DRAM |
|--------|-----|--------|-----------|----------|-------------|
| 128 | 1 | 0.793 | 522 | 655 | +5% |
| 256 | 1 | 0.803 | 628 | 809 | +5% |
| 512 | 1 | 0.808 | 823 | 1,090 | +2% |
| 1024 | 1 | 0.811 | 1,142 | 1,559 | +3% |
| 128 | 2 | 0.909 | 868 | 1,012 | +1% |
| 256 | 2 | 0.934 | 964 | 1,137 | +1% |
| 512 | 2 | 0.947 | 1,187 | 1,435 | +1% |
| 1024 | 2 | 0.954 | 1,513 | 1,965 | +2% |
| 1024 | 3 | 0.980 | 1,905 | 2,285 | +5% |
| 1024 | 5 | 0.990 | 2,805 | 3,050 | +5% |

> SSD overhead minimal (+1-5%). Same pattern as DRAM — CPU-bound, not IO-bound.

#### OBS

| nprobe | rf | Recall | Mean (ms) | P99 (ms) | RQ OBS | SQ OBS | USQ vs RQ | USQ vs SQ |
|--------|-----|--------|-----------|----------|--------|--------|-----------|-----------|
| 128 | 1 | 0.793 | **6,375** | 10,689 | 9,382 | 10,518 | **-32%** | **-39%** |
| 256 | 1 | 0.803 | **6,555** | 10,457 | 10,079 | 12,508 | **-35%** | **-48%** |
| 1024 | 1 | 0.811 | **7,982** | 12,573 | 11,363 | 22,883 | **-30%** | **-65%** |
| 128 | 2 | 0.909 | **11,304** | 15,474 | 18,729 | 20,692 | **-40%** | **-45%** |
| 256 | 2 | 0.934 | **11,580** | 15,459 | 19,306 | 23,298 | **-40%** | **-50%** |
| 1024 | 2 | 0.954 | **13,088** | 17,626 | 20,594 | 32,103 | **-36%** | **-59%** |

> **USQ dominates OBS**: 30-40% faster than RQ, 39-65% faster than SQ. Network is the bottleneck and USQ's 4-bit codes (~336B/vector) are ~6x smaller than float32 vectors (2048B/vector), dramatically reducing S3 download time.

---

## 4. Cross-Index Comparison

### 4.1 Pareto Frontier: DRAM (best latency at each recall level)

| Recall | Config | Mean (ms) | Notes |
|--------|--------|-----------|-------|
| 0.79 | 1B np=128 rf=1 | **626** | 1024-dim baseline |
| 0.83 | 1B np=128 rf=1 | 626 | |
| 0.91 | USQ np=128 rf=2 | **856** | New: beats 1B by 5% |
| 0.93 | USQ np=256 rf=2 | **957** | New: beats RQ 977ms |
| **0.958** | **SQ np=128 rf=1** | **993** | Best for recall≤0.96 |
| 0.98 | 1B np=1024 rf=2 | 2,150 | |
| 0.99 | 1B np=1024 rf=3 | 2,513 | |

> Change: USQ occupies recall 0.91-0.95 region, squeezing RQ out. SQ rf=1 still dominates recall 0.96.

### 4.2 Pareto Frontier: OBS (best latency at each recall level)

| Recall | Config | Mean (ms) | Notes |
|--------|--------|-----------|-------|
| 0.79 | USQ np=128 rf=1 | **6,375** | New: 32% faster than RQ |
| 0.80 | USQ np=256 rf=1 | **6,555** | New: 35% faster than RQ |
| 0.81 | USQ np=1024 rf=1 | **7,982** | New: 30% faster than RQ |
| **0.91** | **USQ np=128 rf=2** | **11,304** | New: 40% faster than RQ rf=2 |
| **0.934** | **USQ np=256 rf=2** | **11,580** | New: 40% faster than RQ rf=2 |
| **0.958** | **SQ np=128 rf=1** | **10,518** | Still best for recall 0.96 |
| 0.954 | USQ np=1024 rf=2 | **13,088** | New: 36% faster than RQ |

> Change: USQ completely reshapes the OBS Pareto frontier. All recall levels below 0.96 are now USQ territory. SQ rf=1 retains its niche at recall=0.958 but is only 8% faster than USQ np=256 rf=2 (10.5s vs 11.6s) with only marginally higher recall.

### 4.3 Matched-Recall OBS Comparison (USQ changes the picture)

USQ's 4-bit codes reduce OBS network transfer by ~6x vs float32 vectors. This transforms the cost-benefit tradeoff:

| Target Recall | Best Config | OBS Latency | Previous Best | Previous Latency | Improvement |
|--------------|-------------|-------------|---------------|-----------------|-------------|
| ~0.79 | **USQ np=128 rf=1** | **6,375ms** | RQ np=128 rf=1 | 9,382ms | **32% faster** |
| ~0.91 | **USQ np=128 rf=2** | **11,304ms** | RQ np=128 rf=2 | 18,729ms | **40% faster** |
| ~0.95 | **USQ np=1024 rf=2** | **13,088ms** | SQ np=128 rf=1 (0.958) | 10,518ms | SQ 20% faster but lower recall |
| ~0.96 | SQ np=128 rf=1 | **10,518ms** | — | — | SQ niche retained |

**Why USQ wins on OBS**: Network is the bottleneck. USQ reads ~336B of 4-bit codes per candidate during refinement, vs 2048B (512×float32) for original vectors. This ~6x reduction in S3 download volume dominates latency. RQ and SQ refinement both download float32 vectors — USQ avoids this by using USQ codes for distance approximation even during refinement.

### 4.4 Pareto Shift Summary

**DRAM**: USQ shifts the frontier at recall 0.79-0.91 (was 1B-RQ territory). SQ retains recall 0.96+ niche.

```
Recall  0.79    0.83    0.91    0.94    0.96    0.98    0.99
Before: [1B-RQ] [1B-RQ] [------] [RQ   ] [SQ   ] [1B   ] [1B   ]
After:  [USQ  ] [1B-RQ] [USQ  ] [RQ   ] [SQ   ] [USQ  ] [1B   ]
                                         ↑ SQ still dominates recall≤0.96
```

**OBS**: USQ completely replaces RQ at recall ≤0.81. But SQ rf=1 (10.5s, recall 0.958) still dominates the recall 0.91-0.96 gap — USQ rf=2 (11.3s, recall 0.909) is slower AND lower recall than SQ rf=1.

```
Recall  0.79    0.81           0.958             0.97
Before: [RQ   ] [RQ  .........] [SQ rf=1        ] [RQ rf=2 ...]
After:  [USQ  ] [USQ .........] [SQ rf=1        ] [SQ rf=2   ]
        ↑ 32% faster           ↑ SQ still king     ↑ SQ dominates
```

**Key insight**: On OBS, USQ rf=1 is the fastest option for recall ≤0.81, but **SQ rf=1** remains the best for recall 0.96. There is no index that fills the 0.81-0.96 recall gap efficiently on OBS — this is where multi-bit RQ or higher-bit USQ would help.

---

## 5. Analysis

### 5.1 Storage Tier Breakdown

| Component | DRAM | SSD | OBS |
|-----------|------|-----|-----|
| Index scan (RQ/SQ codes) | ~100ms | ~100-500ms | ~8-10s (S3 GET RTT) |
| Refinement (float32 vectors) | ~350ms/rf | ~350ms/rf | ~8-10s/rf (S3 download) |
| Distance computation | ~100-500ms | ~100-500ms | ~100-500ms |
| rf=1 total | ~0.5-1.7s | ~0.5-3.0s | ~8-23s |
| rf=2 total | ~0.9-2.5s | ~0.9-3.3s | ~17-32s |

**Key insight**: On DRAM/SSD, rf cost is CPU-bound (~350ms/unit). On OBS, rf cost is IO-bound (~8-10s/unit for vector download).

### 5.2 SQ Recall Ceiling

SQ recall plateaus at **0.972** regardless of rf>2. This is a hard ceiling from 8-bit quantization error. RQ continues improving past 0.97 with higher rf.

| Index | rf=1 max | rf=2 max | rf=3 max | rf=5 max |
|-------|----------|----------|----------|----------|
| IVF_RQ (PCA-512) | 0.794 | 0.953 | 0.967 | 0.971 |
| IVF_USQ (PCA-512) | 0.811 | 0.954 | 0.980 | 0.990 |
| IVF_SQ (PCA-512) | **0.965** | **0.972** | 0.972 | 0.972 |
| IVF_RQ (1B 1024-dim) | 0.861 | 0.983 | 0.993 | 0.995 |

> USQ recall ceiling (0.990) is higher than RQ (0.971) because USQ refinement uses USQ 4-bit codes (higher quality approximation) rather than RQ multi-level residual codes. But both RQ and USQ use the same IVF partitioning, so recall at rf=1 is nearly identical.

### 5.3 Why SQ is Slower at Matched rf

SQ index is larger than RQ index:
- SQ: 466GB/shard (SQ codes ~100GB + raw vectors ~386GB)
- RQ: 386GB/shard (RQ codes ~0.8GB + raw vectors ~386GB)
- USQ: ~256GB/shard (USQ codes ~63GB + raw vectors ~193GB)
- Larger index → more S3 GET requests → more RTT overhead on OBS
- On DRAM, 500GB SQ codes exceed 493GB page cache

### 5.4 USQ: Cache Pollution vs Network Advantage

| Storage | USQ vs RQ | Root Cause |
|---------|-----------|------------|
| DRAM, np≤256 | 2-6% faster | 4-bit distance slightly faster than RQ decode |
| DRAM, np≥1024 | 7-31% slower | 63GB auxiliary.idx evicts useful pages from 493GB RAM |
| SSD | Same as DRAM | IO not bottleneck, CPU-bound |
| **OBS** | **30-40% faster** | Network bottleneck, 4-bit codes ~6x smaller than float32 vectors |

This tradeoff is fundamental: USQ's compact codes are a disadvantage when IO is free (DRAM) but a massive advantage when IO is expensive (OBS).

---

## 6. Recommendations by Scenario

| Scenario | Recommended Config | Latency | Recall | Notes |
|----------|-------------------|---------|--------|-------|
| **DRAM, recall ≤ 0.95** | USQ np=256 rf=2 | **957ms** | 0.934 | 2% faster than RQ, same data |
| **DRAM, recall ≤ 0.96** | SQ np=128 rf=1 | 993ms | 0.958 | Single pass, simplest |
| **DRAM, recall ≥ 0.99** | 1B np=1024 rf=3 | 2,513ms | 0.993 | Need full 1024-dim for 0.99+ |
| **SSD, recall ≤ 0.95** | USQ np=256 rf=2 | **964ms** | 0.934 | Same as DRAM, IO minimal |
| **OBS, recall ≤ 0.81** | **USQ np=1024 rf=1** | **7,982ms** | 0.811 | 30% faster than RQ |
| **OBS, recall ~0.93** | **USQ np=256 rf=2** | **11,580ms** | 0.934 | 40% faster than RQ rf=2 |
| **OBS, recall ~0.96** | SQ np=128 rf=1 | 10,518ms | 0.958 | SQ niche, but USQ np=256 rf=2 close |
| **OBS, recall ~0.95** | **USQ np=1024 rf=2** | **13,088ms** | 0.954 | 36% faster than RQ, beats SQ at np≥1024 |

### IO Threads

LANCE_IO_THREADS=512 vs 128 showed **<1% improvement**. The bottleneck is per-query IO scheduling, not global thread pool.

---

## 7. Optimization Roadmap

Current best (DRAM): 2,513ms for recall 0.993. Knowhere achieves ~50ms at similar recall. 30x gap.

| Priority | Optimization | Expected Impact | Status |
|----------|-------------|----------------|--------|
| P0 | Multi-bit RQ (4-6 bit levels) | rf=1 → 0.95 recall, save ~350ms/rf | Not started |
| P1 | SQ8 two-stage rerank (within IVF_RQ) | Replace float32 reads, 10x faster refinement | Not started |
| P2 | PQ FastScan (bbs=32, AVX-512 VNNI) | 4-10x PQ distance throughput | Not started |
| P3 | Cross-query partition cache (LRU) | 50-80% IO reduction on OBS | Not started |

**Projected** if all optimizations implemented: recall 0.99 in ~500ms (DRAM), ~15s (OBS).

---

## 8. Data Paths (ECS)

| Item | Path |
|------|------|
| 1B shards (1024-dim) | `/data/work/tmp/s1b/shard-{0-4}.lance` |
| PCA-512 shards (RQ+USQ) | `/data/work/tmp/s1b-pca512/shard-{0-4}.lance` |
| SQ shards | `/data/work/tmp/s1b-pca512-sq/shard-{0-4}.lance` |
| 1B results | `/data/work/tmp/pareto-results/` |
| PCA-512 results | `/data/work/tmp/pca512-results-v2/` |
| SQ results | `/data/work/tmp/pca512-sq-results/` |
| USQ results | `/tmp/usq_dram_np*_rf*.json`, `/tmp/usq_ssd_np*_rf*.json`, `/tmp/usq_obs_np*_rf*.json` |
| GT (_rowid format) | `/data/work/tmp/gt_pca512_shard0_top10k_5q_seed42_v2.npz` |
| GT (positional) | `/data/work/tmp/s1b-pca512/gt_positional_v2.npz` |
| OBS (1B) | `s3://knowledgebase-5f43/fineweb-edu-1b-rq-shard4/` |
| OBS (PCA-512, RQ+USQ) | `s3://knowledgebase-5f43/pca512-1b/` |
| OBS (SQ) | `s3://knowledgebase-5f43/pca512-sq-1b/` |
