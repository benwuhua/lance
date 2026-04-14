IR# 1B Vector Search Benchmark Report

> Dataset: 972M rows, FineWeb-Edu embeddings, 1024-dim, cosine
> Date: 2026-04-14
> Environment: Huawei Cloud ECS (493GB RAM, 12TB NVMe, OBS S3)

---

## 1. Architecture

### 1.0 Canonical Benchmark Contract

Unless a subsection explicitly says otherwise, the cross-family conclusions in this report refer to the revalidated PCA-512 contract:

- `query_count=8`
- `warmup=3`
- `timed_queries=5`
- `seed=42`
- local recall on shard-0 against positional GT
- 5-shard end-to-end latency for `dram`, `ssd`, and `obs`
- `USQ4` and `USQ8` treated as separate families, never merged into generic `USQ`

Two older experiment slices still appear later in the report for context:

- `USQ4`: `5-query / warmup=0 / _rowid GT`
- `USQ8`: `5-query / warmup=0 / _rowid GT` plus the 2026-04-12 OBS refresh

Those slices are no longer the source-of-truth for cross-family recommendations.

### 1.1 Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Data Preparation Pipeline                              │
│                                                                                 │
│  FineWeb-Edu benchmark corpus ──┐                                         │      │
│  (972M rows)                    │                                         │      │
│  1024-dim                       ├───────────────────────────────────────┐ │      │
│  normalized vectors             │                                       │ │      │
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
│                           │   5×412GB/shard   │    │  5×207GB/sh  │ │ 5×293G  │ │
│                           │   search ~26GB    │    │ search ~14GB │ │search100G│ │
│                           │   + stored vecs   │    │ +stored vecs │ │ +stored │ │
│                           └──────────────────┘    └──────┬───────┘ └────┬────┘ │
│                                                          │               │      │
│                                       ┌──────────────────┘               │      │
│                                       ▼                                  ▼      │
│                           ┌──────────────────┐                  ┌────────────────┐
│                           │ IVF_USQ4          │                  │ IVF_USQ8       │
│                           │ 5×256GB/shard     │                  │ 5×310GB/shard  │
│                           │ search ~63GB/sh   │                  │ search ~117GB  │
│                           │ + stored vecs     │                  │ + stored vecs  │
│                           └──────────────────┘                  └────────────────┘
└─────────────────────────────────────────────────────────────────────────────────┘
```

The 972M benchmark corpus was synthesized earlier from a 324M source corpus via 3x replication plus small Gaussian noise. The main report uses only the final 972M benchmark corpus numbers.

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
│              ┌────────────────────────────────────────────────────────────┐      │
│              │                   Per-Shard Search                         │      │
│              │                                                            │      │
│              │  1) IVF probe picks multiple partitions                    │      │
│              │                                                            │      │
│              │     [P17] [P204] [P991] ... [P8192]                        │      │
│              │        │      │      │           │                          │      │
│              │        └──────┴──────┴───── ... ─┘                          │      │
│              │                nprobe selected partitions                   │      │
│              │                                                            │      │
│              │  2) Scan search payload inside every selected partition     │      │
│              │                                                            │      │
│              │     P17:   v1 v2 v3 v4 ... v14800                          │      │
│              │     P204:  v1 v2 v3 v4 ... v14800                          │      │
│              │     P991:  v1 v2 v3 v4 ... v14800                          │      │
│              │                                                            │      │
│              │     approximate scores over RQ / SQ / USQ search payload   │      │
│              │                                                            │      │
│              │  3) Keep top-K × refine_factor candidates                  │      │
│              │                                                            │      │
│              │     c42  c105  c933  c1401  c8128  ...                     │      │
│              │                                                            │      │
│              │  4) Refine stage reads many candidate vectors, not one      │      │
│              │                                                            │      │
│              │     vec[c42] vec[c105] vec[c933] vec[c1401] ...            │      │
│              │                                                            │      │
│              │  5) Emit shard-local top-K                                 │      │
│              └─────────┬──────────────────────────────────────────────────┘      │
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
│                 IVF Layout: Per-Query Work Inside One Shard                 │
│                                                                             │
│  IVF centroids                                                              │
│  ┌─────────────────────────────────────┐                                    │
│  │ 13,107 partitions / shard           │                                    │
│  │ query probes nprobe of them         │                                    │
│  └─────────────────────────────────────┘                                    │
│                    │                                                        │
│                    ▼                                                        │
│  selected partitions                                                        │
│  ┌─────────────────────────────────────┐                                    │
│  │ P17   P204   P991   ...   P8192     │                                    │
│  │ each partition ~14,800 vectors      │                                    │
│  └─────────────────────────────────────┘                                    │
│                    │                                                        │
│                    ▼                                                        │
│  approximate scan over search payload                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │ RQ:   72B/vector on PCA-512         │                                    │
│  │ SQ:   512B/vector                   │                                    │
│  │ USQ4: 336B/vector                   │                                    │
│  │ USQ8: 592B/vector                   │                                    │
│  └─────────────────────────────────────┘                                    │
│                    │                                                        │
│                    ▼                                                        │
│  refine stage (if rerank is enabled)                                        │
│  ┌─────────────────────────────────────┐                                    │
│  │ reread top-K × refine_factor        │                                    │
│  │ candidate vectors from storage      │                                    │
│  └─────────────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Per-query work is controlled by `nprobe` and `refine_factor`, not by the full `/shard` total:

| `nprobe`   | partitions touched / shard   | approximate candidates / shard   |
| ---------- | ---------------------------- | -------------------------------- |
| `128`      | `128`                        | `~1.89M`                         |
| `256`      | `256`                        | `~3.79M`                         |
| `1024`     | `1024`                       | `~15.16M`                        |

Approximate-search bytes touched per shard on PCA-512:

| Family   | bytes / vector in search payload   | `nprobe=128`   | `nprobe=256`   | `nprobe=1024`   | I/O mode                           |
| -------- | ---------------------------------- | -------------- | -------------- | --------------- | -------------------------------- |
| `RQ`     | `72B`                              | `~130MB`       | `~260MB`       | `~1.04GB`       | partition-local sequential scan; multi-partition scattered |
| `SQ`     | `512B`                             | `~925MB`       | `~1.85GB`      | `~7.40GB`       | partition-local sequential scan; multi-partition scattered |
| `USQ4`   | `336B`                             | `~607MB`       | `~1.21GB`      | `~4.86GB`       | partition-local sequential scan; multi-partition scattered |
| `USQ8`   | `592B`                             | `~1.07GB`      | `~2.14GB`      | `~8.56GB`       | partition-local sequential scan; multi-partition scattered |

Refine-stage vector rereads per shard:

| `refine_factor`   | vectors reread / shard         | logical vector payload on PCA-512 (`512 × f32`)          | I/O mode                   |
| ----------------- | ------------------------ | -------------------------------------------------- | ------------------------ |
| `1`               | `10,000`                 | `~20MB`                                            | random gather over candidate ids |
| `2`               | `20,000`                 | `~41MB`                                            | random gather over candidate ids |
| `5`               | `50,000`                 | `~102MB`                                           | random gather over candidate ids |

Interpretation:

- Approximate search is not a single giant sequential scan over the shard. It is a sequence of partition-local scans.
- Refine is not partition-local scanning. It is a gather over the candidate vector ids that survived approximate scoring.
- On OBS, the refine gather is typically the more random stage, while the approximate stage is dominated by the total bytes scanned across many selected partitions.

This is the first-principles comparison that matters for latency:

- `nprobe` controls how many partitions and candidates the approximate stage scans.
- `bytes/vector` controls how expensive each scanned candidate is.
- `refine_factor` controls how many original vectors get reread from storage.

### 1.4 Storage Tier Latency Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Query Latency Breakdown                              │
│                                                                         │
│  DRAM (page cache hot)           SSD (cold cache)     OBS (S3 / OBS)   │
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
│  rf=1 → 0.5-2.4s                0.5-3.3s             5.7-21.2s         │
│  rf=2 → 0.9-2.5s                0.9-3.3s             10.5-17.6s        │
└─────────────────────────────────────────────────────────────────────────┘
```

Revalidated storage-tier shape:

- `DRAM`: `RQ` frontier at low recall, `SQ` frontier at `0.957+`, `USQ8` only a narrow niche
- `SSD`: same as DRAM with modest cold-cache overhead
- `OBS`: frontier shifts entirely to Hanns-backed `USQ4` and `USQ8`

Footprint convention used below:

- `search payload`: compressed search codes or quantized auxiliary payload used during approximate distance evaluation
- `stored vector payload`: the underlying vector column stored in each shard
- `total shard footprint`: `search payload + stored vector payload`, plus only small metadata overhead
- All family comparisons below use the same three fields. Do not compare `search payload` against `total shard footprint` as if they were the same metric.

### 1.5 IVF Index Parameters

| Parameter      | Value                  | Rationale                              |
| -------------- | ---------------------- | -------------------------------------- |
| num_partitions | 13,107/shard           | min(N/1000, 65536)/5                   |
| density        | ~14,800 vecs/partition | 194M / 13,107                          |
| nprobe range   | 128-1024               | 1-8% of partitions                     |
| refine_factor  | 1-5                    | re-rank top-K×rf with original vectors |

### 1.6 Benchmark Tool

```
benchmarks/cohere/bench.py query \
    --plan B --storage {dram,ssd,obs} \
    --index-type {IVF_RQ,IVF_SQ,IVF_USQ} \
    --shard-dir PATH --nprobes N --refine-factor N \
    --top-k 10000 --num-shards 5 --query-count 8 --warmup 3
```

Notes:

- `bench.py` does not expose `index_name`, so one live dataset path can only serve one benchmark family at a time.
- `bench.py` computes recall in-band only for non-OBS paths; OBS recall in the canonical matrix is copied from the matching local DRAM config.

---

## 2. Index Variants Tested

### 2.0 Canonical Footprint Table

All four families are compared with the same three fields:

| Family                   | Search payload / shard    | Search payload implementation                           | Stored vector payload / shard   | Total shard footprint / shard   |
| ------------------------ | ------------------------- | ------------------------------------------------------- | ------------------------------- | ------------------------------- |
| `RQ` (1024-dim baseline) | `~26GB`                   | 1-bit RQ codes + `add/scale` factors (`136B/vector`)    | `~386GB`                        | `~412GB`                        |
| `RQ` (PCA-512)           | `~14GB`                   | 1-bit RQ codes + `add/scale` factors (`72B/vector`)     | `~193GB`                        | `~207GB`                        |
| `SQ` (PCA-512)           | `~100GB`                  | SQ code stream (`512B/vector`)                          | `~193GB`                        | `~293GB`                        |
| `USQ4` (PCA-512)         | `~63GB`                   | Hanns search payload in `auxiliary.idx` (`336B/vector`) | `~193GB`                        | `~256GB`                        |
| `USQ8` (PCA-512)         | `~117GB`                  | Hanns search payload in `auxiliary.idx` (`592B/vector`) | `~193GB`                        | `~310GB`                        |

Read this table literally:

- `RQ`, `SQ`, and `USQ` all have a `search payload`.
- The only thing that differs is the implementation of that payload: `RQ` code stream, `SQ` code stream, or Hanns payload stored in `auxiliary.idx`.
- The `GB` figures above are always `GB per shard`.
- For `RQ`, the payload is not just packed 1-bit codes; it also carries two `f32` factors per vector, which is why it lands around `72B/vector` on PCA-512 instead of `64B/vector`.

### 2.1 IVF_RQ (Residual Quantization)

- 1-bit residual quantization with per-vector `add/scale` correction factors
- Search payload: ~26GB/shard on the 1024-dim baseline, ~14GB/shard on PCA-512
- Stored vector payload: ~386GB/shard on the 1024-dim baseline
- Total shard footprint: ~412GB/shard on the 1024-dim baseline, ~207GB/shard on PCA-512
- Search payload implementation: packed 1-bit RQ codes plus two `f32` factors (`136B/vector` at 1024-dim, `72B/vector` at PCA-512)
- **Strengths**: Compact codes, recall scales with rf (up to 0.995)
- **Weaknesses**: rf=1 recall only ~0.79 (1-bit residual too coarse), rf>1 reads stored vectors

### 2.2 IVF_SQ (Scalar Quantization)

- 8-bit per dimension (float32 → uint8, 4x compression per vector)
- Search payload: ~100GB/shard
- Stored vector payload: ~193GB/shard on PCA-512
- Total shard footprint: ~293GB/shard
- Search payload implementation: SQ code stream (`512B/vector`)
- **Strengths**: SQ distance very accurate, rf=1 recall 0.958 (vs RQ 0.790)
- **Weaknesses**: Index exceeds page cache, recall ceiling at 0.972

### 2.3 IVF_USQ4 (Ultra-Sparse Quantization, 4-bit via Hanns)

- Random rotation → normalize → 4-bit quantize with Hanns approximate scoring
- Search payload implementation: Hanns payload in `auxiliary.idx` (`336B/vector`)
- Search payload: ~63GB/shard
- Stored vector payload: ~193GB/shard on PCA-512
- Total shard footprint: ~256GB/shard
- **Strengths**: 4-bit codes ~6x smaller than stored PCA vectors → massive OBS advantage; rf unit cost same as RQ (~350ms)
- **Weaknesses**: 63GB/shard index causes cache pollution on DRAM at high nprobe; recall identical to RQ (no quality advantage)
- Feature-gated behind `#[cfg(feature = "hanns")]`
- This subsection is specifically the 4-bit variant. Revalidated cross-family comparisons in Section 4 distinguish `USQ4` from `USQ8` explicitly.

### 2.4 IVF_USQ8 (Ultra-Sparse Quantization, 8-bit via Hanns)

- Same rotation / normalization pipeline as `USQ4`, but with 8-bit quantization
- Search payload implementation: Hanns payload in `auxiliary.idx` (`592B/vector`)
- Search payload: ~117GB/shard
- Stored vector payload: ~193GB/shard on PCA-512
- Total shard footprint: ~310GB/shard
- **Strengths**: Higher intrinsic recall than `USQ4`, especially on OBS at `rf=1`
- **Weaknesses**: Larger search payload increases DRAM/SSD cache pressure and raises build cost
- In the revalidated matrix, `USQ8` is mostly an OBS-focused family

### 2.5 Dimensionality Reduction: PCA-512

- TruncatedSVD 1024→512 on 500K sample, explained variance 96.63%
- Applied to `RQ`, `SQ`, `USQ4`, and `USQ8`
- ~2x smaller index, ~2x faster distance computation

---

## 3. Sweep Results

### 3.1 1B Baseline (1024-dim, IVF_RQ)

Index: 5 shards × ~412GB = ~2.06TB total. Recall measured on shard-0 against flat search GT.

#### DRAM (partial page cache)

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   |
| -------- | ----- | -------- | ----------- | ---------- |
| 64       | 1     | 0.79*    | 616         | 860        |
| 64       | 2     | —        | 840         | 975        |
| 128      | 1     | 0.830    | 626         | 880        |
| 128      | 2     | 0.921    | 904         | 1,011      |
| 256      | 1     | 0.845    | 656         | 786        |
| 256      | 2     | 0.950    | 968         | 1,063      |
| 512      | 1     | —        | 985         | 1,047      |
| 512      | 2     | —        | 1,262       | 1,338      |
| 1024     | 1     | 0.861    | 1,740       | 1,843      |
| 1024     | 2     | 0.983    | 2,150       | 2,218      |
| 1024     | 3     | 0.993    | 2,513       | 2,631      |
| 1024     | 5     | 0.995    | 3,468       | 3,786      |

*Recall for np<128 estimated from GT subset. Some configs missing recall (ran before GT integration).

#### SSD (cold cache)

| nprobe   | rf    | Mean (ms)   | P99 (ms)   | SSD vs DRAM   |
| -------- | ----- | ----------- | ---------- | ------------- |
| 64       | 1     | 631         | 888        | +2%           |
| 64       | 2     | 889         | 1,054      | +6%           |
| 128      | 1     | 677         | 979        | +8%           |
| 128      | 2     | 928         | 1,106      | +3%           |
| 256      | 1     | 754         | 914        | +15%          |
| 256      | 2     | 1,058       | 1,193      | +9%           |
| 512      | 1     | 1,168       | 1,231      | +19%          |
| 512      | 2     | 1,497       | 1,529      | +19%          |
| 1024     | 1     | 2,362       | 2,534      | +36%          |
| 1024     | 2     | 2,636       | 2,738      | +23%          |
| 1024     | 3     | 2,983       | 3,094      | +19%          |
| 1024     | 5     | 3,986       | 4,250      | +15%          |

#### OBS (S3)

| nprobe   | rf    | Mean (ms)   | P99 (ms)   |
| -------- | ----- | ----------- | ---------- |
| 64       | 1     | 8,218       | 8,648      |
| 64       | 2     | 17,051      | 20,292     |
| 128      | 1     | 8,403       | 8,946      |
| 128      | 2     | 25,964      | 30,282     |
| 256      | 1     | 8,909       | 9,481      |
| 256      | 2     | 16,395      | 16,955     |
| 512      | 1     | 11,195      | 11,762     |
| 512      | 2     | 19,041      | 21,868     |
| 1024     | 1     | 17,588      | 18,759     |
| 1024     | 2     | 24,984      | 25,744     |

---

### 3.2 PCA-512 (512-dim, IVF_RQ)

Index: 5 shards × ~207GB = ~1.04TB total. `RQ` search payload is ~14GB/shard on PCA-512, while stored PCA vectors contribute ~193GB/shard.

#### DRAM

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   | vs 1B Baseline   |
| -------- | ----- | -------- | ----------- | ---------- | ---------------- |
| 128      | 1     | 0.790    | 523         | 537        | **16% faster**   |
| 128      | 2     | 0.942    | 875         | 888        | 3% faster        |
| 256      | 1     | 0.793    | 636         | 663        | 3% faster        |
| 256      | 2     | 0.948    | 977         | 1,006      | same             |
| 512      | 1     | 0.794    | 756         | 856        | 23% faster       |
| 512      | 2     | 0.951    | 1,091       | 1,167      | 14% faster       |
| 1024     | 1     | 0.794    | 846         | 1,035      | **51% faster**   |
| 1024     | 2     | 0.953    | 1,183       | 1,315      | **45% faster**   |
| 1024     | 3     | 0.967    | 1,563       | 1,694      | **38% faster**   |
| 1024     | 5     | 0.971    | 2,494       | 2,609      | 28% faster       |

#### OBS

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   | vs 1B Baseline OBS    |
| -------- | ----- | -------- | ----------- | ---------- | --------------------- |
| 128      | 1     | 0.790    | 9,382       | 9,510      | same                  |
| 128      | 2     | 0.942    | 18,729      | 18,995     | 28% faster            |
| 256      | 1     | 0.793    | 10,079      | 10,288     | same                  |
| 256      | 2     | 0.948    | 19,306      | 19,568     | same                  |
| 512      | 1     | 0.794    | 10,698      | 11,287     | same                  |
| 512      | 2     | 0.951    | 20,010      | 20,430     | same                  |
| 1024     | 1     | 0.794    | 11,363      | 12,046     | **35% faster**        |
| 1024     | 2     | 0.953    | 20,594      | 21,375     | 17% faster            |

**PCA-512 conclusion**: DRAM improvement significant at high nprobe (up to 51%). OBS improvement only at high nprobe + rf=1 (up to 35%). Recall ceiling ~0.97 from dimension loss.

---

### 3.3 PCA-512 + IVF_SQ (512-dim, IVF_SQ)

Index: 5 shards × ~293GB = ~1.47TB total (SQ search payload ~100GB + stored PCA vectors ~193GB per shard).

#### DRAM

| nprobe   | rf    | Recall    | Mean (ms)   | P99 (ms)   | RQ DRAM Mean  | SQ vs RQ   |
| -------- | ----- | --------- | ----------- | ---------- | ------------- | ---------- |
| 128      | 1     | **0.958** | 993         | 1,102      | 523           | +90%       |
| 256      | 1     | **0.962** | 1,050       | 1,289      | 636           | +65%       |
| 512      | 1     | **0.964** | 1,684       | 2,129      | 756           | +123%      |
| 1024     | 1     | **0.965** | 2,479       | 3,020      | 846           | +193%      |
| 128      | 2     | **0.964** | 1,085       | 1,215      | 875           | +24%       |
| 256      | 2     | **0.969** | 1,234       | 1,497      | 977           | +26%       |
| 512      | 2     | **0.971** | 1,653       | 1,883      | 1,091         | +52%       |
| 1024     | 2     | **0.972** | 2,448       | 2,815      | 1,183         | +107%      |
| 1024     | 3     | 0.972     | 2,805       | 3,411      | 1,563         | +80%       |
| 1024     | 5     | 0.972     | 3,575       | 4,415      | 2,494         | +43%       |

#### SSD (cold cache)

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   | SQ DRAM   | SSD vs DRAM   |
| -------- | ----- | -------- | ----------- | ---------- | --------- | ------------- |
| 128      | 1     | 0.958    | 949         | 1,001      | 993       | -5%           |
| 256      | 1     | 0.962    | 1,275       | 1,442      | 1,050     | +21%          |
| 512      | 1     | 0.964    | 1,813       | 2,136      | 1,684     | +8%           |
| 1024     | 1     | 0.965    | 3,027       | 3,483      | 2,479     | +22%          |
| 128      | 2     | 0.964    | 1,190       | 1,340      | 1,085     | +10%          |
| 256      | 2     | 0.969    | 1,477       | 1,746      | 1,234     | +20%          |
| 512      | 2     | 0.971    | 2,005       | 2,358      | 1,653     | +21%          |
| 1024     | 2     | 0.972    | 3,251       | 3,951      | 2,448     | +33%          |
| 1024     | 3     | 0.972    | 3,565       | 4,352      | 2,805     | +27%          |
| 1024     | 5     | 0.972    | 4,313       | 5,216      | 3,575     | +21%          |

#### OBS

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   | RQ OBS   | SQ vs RQ   |
| -------- | ----- | -------- | ----------- | ---------- | -------- | ---------- |
| 128      | 1     | 0.958    | 10,518      | 11,031     | 9,382    | +12%       |
| 256      | 1     | 0.962    | 12,508      | 13,935     | 10,079   | +24%       |
| 1024     | 1     | 0.965    | 22,883      | 25,807     | 11,363   | +101%      |
| 128      | 2     | 0.964    | 20,692      | 24,037     | 18,729   | +10%       |
| 256      | 2     | 0.969    | 23,298      | 26,854     | 19,306   | +21%       |
| 1024     | 2     | 0.972    | 32,103      | 34,735     | 20,594   | +56%       |

---

### 3.4 PCA-512 + IVF_USQ4 (512-dim, IVF_USQ, exploratory slice)

Index: 5 shards × ~256GB = 1.28TB total. Search payload ~63GB/shard plus stored PCA vectors ~193GB/shard. Uses the same PCA-512 dataset as `RQ`, `SQ`, and `USQ8`.

> This section preserves the earlier `5-query / warmup=0 / _rowid GT` exploratory slice for `USQ4`. The canonical cross-family comparison now lives in Section 4 under the unified legacy contract (`query_count=8`, `warmup=3`, positional GT).

#### DRAM

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   | RQ Mean   | SQ Mean   | USQ vs RQ   |
| -------- | ----- | -------- | ----------- | ---------- | --------- | --------- | ----------- |
| 128      | 1     | 0.793    | **496**     | 612        | 523       | 993       | -5%         |
| 256      | 1     | 0.803    | **597**     | 747        | 636       | 1,050     | -6%         |
| 512      | 1     | 0.808    | **806**     | 1,037      | —         | 1,684     | —           |
| 1024     | 1     | 0.811    | 1,105       | 1,507      | **846**   | 2,479     | +31%        |
| 128      | 2     | 0.909    | **856**     | 980        | 875       | 1,085     | -2%         |
| 256      | 2     | 0.934    | **957**     | 1,110      | 977       | 1,234     | -2%         |
| 512      | 2     | 0.947    | **1,173**   | 1,398      | —         | 1,653     | —           |
| 1024     | 2     | 0.954    | 1,478       | 1,975      | **1,183** | 2,448     | +25%        |
| 1024     | 3     | 0.980    | 1,812       | 2,211      | **1,563** | 2,805     | +16%        |
| 1024     | 5     | 0.990    | 2,678       | 2,941      | **2,494** | 3,575     | +7%         |

> USQ faster than RQ at np≤256 (2-6%) due to 4-bit distance speed. USQ slower at np≥1024 (7-31%) because the `~63GB/shard` search payload causes cache pollution in 493GB RAM.

#### SSD (cold cache)

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   | SSD vs DRAM   |
| -------- | ----- | -------- | ----------- | ---------- | ------------- |
| 128      | 1     | 0.793    | 522         | 655        | +5%           |
| 256      | 1     | 0.803    | 628         | 809        | +5%           |
| 512      | 1     | 0.808    | 823         | 1,090      | +2%           |
| 1024     | 1     | 0.811    | 1,142       | 1,559      | +3%           |
| 128      | 2     | 0.909    | 868         | 1,012      | +1%           |
| 256      | 2     | 0.934    | 964         | 1,137      | +1%           |
| 512      | 2     | 0.947    | 1,187       | 1,435      | +1%           |
| 1024     | 2     | 0.954    | 1,513       | 1,965      | +2%           |
| 1024     | 3     | 0.980    | 1,905       | 2,285      | +5%           |
| 1024     | 5     | 0.990    | 2,805       | 3,050      | +5%           |

> SSD overhead minimal (+1-5%). Same pattern as DRAM — CPU-bound, not IO-bound.

#### OBS

| nprobe   | rf    | Recall   | Mean (ms)   | P99 (ms)   | RQ OBS   | SQ OBS   | USQ vs RQ   | USQ vs SQ   |
| -------- | ----- | -------- | ----------- | ---------- | -------- | -------- | ----------- | ----------- |
| 128      | 1     | 0.793    | **6,375**   | 10,689     | 9,382    | 10,518   | **-32%**    | **-39%**    |
| 256      | 1     | 0.803    | **6,555**   | 10,457     | 10,079   | 12,508   | **-35%**    | **-48%**    |
| 1024     | 1     | 0.811    | **7,982**   | 12,573     | 11,363   | 22,883   | **-30%**    | **-65%**    |
| 128      | 2     | 0.909    | **11,304**  | 15,474     | 18,729   | 20,692   | **-40%**    | **-45%**    |
| 256      | 2     | 0.934    | **11,580**  | 15,459     | 19,306   | 23,298   | **-40%**    | **-50%**    |
| 1024     | 2     | 0.954    | **13,088**  | 17,626     | 20,594   | 32,103   | **-36%**    | **-59%**    |

> **USQ dominates OBS**: 30-40% faster than RQ, 39-65% faster than SQ. Network is the bottleneck and `USQ4` search payload (~336B/vector) is about 6x smaller than the stored 512-dim vector payload (~1024B/vector), dramatically reducing S3 download time.

---

### 3.5 PCA-512 + IVF_USQ8 (512-dim, IVF_USQ8, historical slice)

Index: 5 shards × ~310GB = ~1.55TB total. Search payload ~117GB/shard plus stored PCA vectors ~193GB/shard.

> This section preserves the earlier `5-query / warmup=0 / _rowid GT` `USQ8` slice and the 2026-04-12 OBS refresh that was recorded in [pca512-usq8.md](wiki/experiments/pca512-usq8.md). It is kept here so Chapter 3 includes all four families. The canonical cross-family comparison still lives in Section 4 under the unified legacy contract (`query_count=8`, `warmup=3`, positional GT).

#### DRAM

| Config         | Recall   | Mean (ms)   | P95 (ms)   | vs `USQ4` DRAM   |
| -------------- | -------- | ----------- | ---------- | ---------------- |
| `np=128 rf=1`  | 0.9474   | 1,731       | 2,464      | +249%            |
| `np=256 rf=1`  | 0.9730   | 1,826       | 2,039      | +206%            |
| `np=512 rf=1`  | 0.9889   | 3,091       | 3,689      | +284%            |
| `np=1024 rf=1` | 0.9949   | 5,770       | 7,026      | +422%            |
| `np=128 rf=2`  | 0.9477   | 1,402       | 1,588      | +64%             |
| `np=256 rf=2`  | 0.9734   | 2,005       | 2,184      | +110%            |
| `np=512 rf=2`  | 0.9895   | 3,274       | 3,808      | +179%            |
| `np=1024 rf=2` | 0.9967   | 5,722       | 6,688      | +287%            |

#### SSD

| Config         | Recall   | Mean (ms)   | P95 (ms)   | SSD vs DRAM   |
| -------------- | -------- | ----------- | ---------- | ------------- |
| `np=128 rf=1`  | 0.9474   | 1,214       | 1,406      | -30%          |
| `np=256 rf=1`  | 0.9730   | 2,042       | 2,335      | +12%          |
| `np=512 rf=1`  | 0.9889   | 3,424       | 4,131      | +11%          |
| `np=1024 rf=1` | 0.9949   | 6,096       | 7,330      | +6%           |
| `np=128 rf=2`  | 0.9477   | 1,554       | 1,764      | +11%          |
| `np=256 rf=2`  | 0.9734   | 2,312       | 2,508      | +15%          |
| `np=512 rf=2`  | 0.9895   | 3,915       | 4,423      | +20%          |
| `np=1024 rf=2` | 0.9967   | 6,321       | 7,361      | +10%          |

#### OBS

| Config         | Recall   | Mean (ms)   | P95 (ms)   | vs `RQ` OBS   | vs `USQ4` OBS    |
| -------------- | -------- | ----------- | ---------- | ------------- | ---------------- |
| `np=128 rf=1`  | 0.9474   | 8,592       | 11,575     | -8%           | +35%             |
| `np=256 rf=1`  | 0.9730   | 7,710       | 9,767      | -24%          | +18%             |
| `np=1024 rf=1` | 0.9949   | 21,312      | 23,382     | +88%          | +167%            |
| `np=128 rf=2`  | 0.9477   | 11,197      | 12,277     | -40%          | -1%              |
| `np=256 rf=2`  | 0.9734   | 12,203      | 13,335     | -37%          | +5%              |
| `np=1024 rf=2` | 0.9967   | 27,988      | 32,322     | +36%          | +114%            |

Takeaway:

- This slice is the historical reason `USQ8` entered the report: it opened a high-recall OBS region that `RQ`, `SQ`, and `USQ4` did not cover at the time.
- The same slice also showed why `USQ8` never became a clean local default: `~117GB/shard` search payload causes much more cache pressure than `USQ4`.
- Section 4 keeps the revalidated contract and should be treated as the source-of-truth for current recommendations.

---

## 4. Revalidated Cross-Family Comparison

The canonical cross-family comparison now uses a single explicit contract:

- `query_count=8`
- `warmup=3`
- `timed_queries=5`
- `seed=42`
- recall on shard-0 against positional GT:
  `/data/work/tmp/s1b-pca512/gt_positional_v2.npz`
- latency as 5-shard end-to-end search
- OBS recall copied from the matching local DRAM config, because `bench.py` does not compute recall in-band for `--storage obs`

The earlier `5-query / warmup=0 / _rowid GT` `USQ4` and `USQ8` slices remain useful as exploratory experiments, but they are no longer the source-of-truth for cross-family conclusions.

### 4.1 Build Evidence Snapshot

| Family   | Build Evidence                                                                                                   | Footprint                                            | Status                                                                           |
| -------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------- |
| `RQ`     | Historical shard logs show transform `4604-5534s` and index build `605-1028s` per shard                          | `search ~14GB + stored ~193GB = total ~207GB/shard`  | Query matrix revalidated; full 5-shard build wall time still needs normalization |
| `SQ`     | Query matrix revalidated, but `build_sq2.log` includes `No space left on device` and only a partial clean record | `search ~100GB + stored ~193GB = total ~293GB/shard` | Build timing not yet canonical                                                   |
| `USQ4`   | `~50 min` for 5 shards parallel                                                                                  | `search ~63GB + stored ~193GB = total ~256GB/shard`  | Clean enough for query-side recommendations                                      |
| `USQ8`   | `~57 min` for 5 shards parallel, `~1.3TB` peak temp                                                              | `search ~117GB + stored ~193GB = total ~310GB/shard` | Clean enough for query-side recommendations                                      |

### 4.2 DRAM Pareto Frontier

| Recall   | Config             | Mean (ms)   | Why It Survives                          |
| -------- | ------------------ | ----------- | ---------------------------------------- |
| 0.78944  | `RQ np=128 rf=1`   | **534**     | Lowest-latency point                     |
| 0.79064  | `RQ np=256 rf=1`   | **622**     | Small recall bump for modest latency     |
| 0.79190  | `RQ np=1024 rf=1`  | **799**     | Best `RQ rf=1` ceiling                   |
| 0.94134  | `RQ np=128 rf=2`   | **872**     | Cheapest jump into the `0.94` band       |
| 0.95748  | `SQ np=128 rf=1`   | **947**     | First `0.95+` point                      |
| 0.95956  | `USQ8 np=128 rf=1` | **1,021**   | Slight recall bump over `SQ np=128 rf=1` |
| 0.96196  | `SQ np=256 rf=1`   | **1,083**   | Better recall with no rerank             |
| 0.96370  | `SQ np=128 rf=2`   | **1,093**   | Small recall gain over `SQ np=256 rf=1`  |
| 0.96854  | `SQ np=256 rf=2`   | **1,269**   | Strongest sub-0.97 local point           |
| 0.97230  | `SQ np=1024 rf=2`  | **2,428**   | Highest revalidated local recall         |

Takeaway:

- Under the unified contract, the DRAM frontier splits cleanly by family:
  `RQ` owns the cheapest `0.79-0.94` region,
  then `SQ` owns almost the entire `0.957-0.972` region.
- `USQ4` disappears completely from the DRAM frontier.
- `USQ8` survives only as one narrow niche point at `0.95956 / 1.02s`.
- That single `USQ8` point is not enough to justify `~117GB/shard` search payload or the higher build cost for local-only deployment.

### 4.3 SSD Pareto Frontier

| Recall   | Config             | Mean (ms)   | Why It Survives                          |
| -------- | ------------------ | ----------- | ---------------------------------------- |
| 0.78944  | `RQ np=128 rf=1`   | **527**     | Lowest-latency point                     |
| 0.79064  | `RQ np=256 rf=1`   | **635**     | Small recall bump                        |
| 0.79190  | `RQ np=1024 rf=1`  | **766**     | Best `RQ rf=1` ceiling                   |
| 0.94134  | `RQ np=128 rf=2`   | **882**     | Cheapest jump into the `0.94` band       |
| 0.94630  | `RQ np=256 rf=2`   | **962**     | Best local `~0.95` point                 |
| 0.95748  | `SQ np=128 rf=1`   | **972**     | First `0.95+` point                      |
| 0.95956  | `USQ8 np=128 rf=1` | **1,152**   | Slight recall bump over `SQ np=128 rf=1` |
| 0.96370  | `SQ np=128 rf=2`   | **1,205**   | Best `~0.964` point                      |
| 0.96854  | `SQ np=256 rf=2`   | **1,465**   | Strongest sub-0.97 SSD point             |
| 0.97230  | `SQ np=1024 rf=2`  | **3,256**   | Highest revalidated SSD recall           |

Takeaway:

- SSD tells the same story as DRAM with a modest cold-cache penalty:
  `RQ` first, then `SQ`, with `USQ4` absent from the frontier.
- `USQ8` again survives only as a narrow `~0.96` niche, not as a broadly dominant local strategy.
- In practice, nothing in the revalidated SSD matrix argues for building `USQ` if the target tier is local storage only.

### 4.4 OBS Pareto Frontier

| Recall   | Config              | Mean (ms)   | Why It Survives                                 |
| -------- | ------------------- | ----------- | ----------------------------------------------- |
| 0.95060  | `USQ4 np=128 rf=1`  | **5,690**   | Lowest-latency revalidated OBS point            |
| 0.95440  | `USQ4 np=256 rf=1`  | **6,824**   | Small recall bump for modest latency            |
| 0.95956  | `USQ8 np=128 rf=1`  | **7,136**   | First `~0.96` point                             |
| 0.96736  | `USQ8 np=256 rf=1`  | **8,823**   | Best `~0.967` OBS point                         |
| 0.97182  | `USQ4 np=1024 rf=2` | **17,583**  | Best latency near `0.972` recall                |
| 0.97202  | `USQ8 np=1024 rf=1` | **21,190**  | Slightly higher recall than `USQ4 np=1024 rf=2` |

Takeaway:

- Under the revalidated legacy contract, every observed `RQ` and `SQ` OBS point is dominated.
- `USQ4` owns the lowest-latency OBS region through roughly `0.955` recall.
- `USQ8` takes over above `~0.96`; this is the canonical replacement for the older mixed `generic USQ` wording.
- The practical high-recall OBS choice is now a tradeoff:
  `USQ4 np=1024 rf=2` is faster at `0.97182`,
  while `USQ8 np=1024 rf=1` buys only `+0.00020` recall for `+3.6s`.
- So the revalidated OBS story is not "always use `USQ8`". It is:
  `USQ4` for the lowest-latency points,
  `USQ8` for the `0.96+` band and the absolute recall ceiling.

### 4.5 What Revalidation Changed

- The older report mixed two methodologies:
  exploratory `5-query / warmup=0 / _rowid GT` slices for `USQ4` and `USQ8`,
  plus the legacy positional-GT path for `RQ` and `SQ`.
- Once every family is read through the same legacy contract, the local story changes materially:
  `RQ` then `SQ`,
  not `USQ4`.
- The OBS story also becomes more precise:
  `USQ4` owns the lowest-latency `0.95-0.955` region,
  `USQ8` owns the `0.96+` region,
  and the practical `~0.972` choice is still a tradeoff rather than a pure `USQ8` win.
- The old "generic `USQ`" wording should therefore be treated as obsolete in this report.

---

## 5. Analysis

### 5.1 First-Principles Readout

Each query has two different I/O stages:

1. **Approximate scan**
   Read `nprobe` touched partitions per shard, stream their search payload, and score all touched candidates.
2. **Refine gather**
   Re-read `top_k * refine_factor` candidate vectors per shard and compute exact similarity on stored vectors.

The right first-principles question is therefore not "which family is smaller overall?", but:

- how many partition ranges must be touched,
- how many bytes must be scanned in the approximate stage,
- how many random candidate-vector gathers must be issued in the refine stage.

#### 5.1.1 Approximate-Stage Operation Count Across 5 Shards

| `nprobe` | Partition ranges touched | Approximate candidates touched |
| -------- | ------------------------ | ------------------------------ |
| `128`    | `640`                    | `~9.45M`                       |
| `256`    | `1,280`                  | `~18.95M`                      |
| `1024`   | `5,120`                  | `~75.80M`                      |

These counts are family-independent. `RQ`, `SQ`, `USQ4`, and `USQ8` touch the same number of partition ranges and the same number of approximate candidates at a given `nprobe`. The family difference is the **bytes per touched candidate**.

#### 5.1.2 Approximate-Stage Bytes Touched Across 5 Shards On PCA-512

| Family | Bytes / candidate | `np=128` | `np=256` | `np=1024` | vs `RQ` |
| ------ | ----------------- | -------- | -------- | --------- | ------- |
| `RQ`   | `72B`             | `~650MB` | `~1.30GB` | `~5.20GB` | `1.0x` |
| `SQ`   | `512B`            | `~4.63GB` | `~9.25GB` | `~37.0GB` | `7.1x` |
| `USQ4` | `336B`            | `~3.04GB` | `~6.05GB` | `~24.3GB` | `4.7x` |
| `USQ8` | `592B`            | `~5.35GB` | `~10.7GB` | `~42.8GB` | `8.2x` |

This table is the cleanest explanation for the local story:

- `RQ` touches the fewest scan bytes by a wide margin.
- `USQ4` reduces scan bytes relative to `SQ`, but is still `~4.7x` heavier than `RQ`.
- `USQ8` is the heaviest family in approximate-stage bandwidth terms.

In other words, the approximate stage is primarily a **bandwidth problem**, not a point-look-up problem. The number of partition reads grows with `nprobe`, but the dominant variable across families is scanned bytes.

#### 5.1.3 Refine-Stage Logical Work Across 5 Shards

| `refine_factor` | Candidate gathers | Logical vector bytes reread on PCA-512 | I/O mode        |
| --------------- | ----------------- | -------------------------------------- | --------------- |
| `1`             | `50,000`          | `~100MB`                               | random gather   |
| `2`             | `100,000`         | `~205MB`                               | random gather   |
| `5`             | `250,000`         | `~510MB`                               | random gather   |

The refine stage is therefore the opposite of approximate scan:

- total bytes are much smaller than the approximate stage,
- but the access pattern is much worse,
- and `refine_factor` scales the number of logical gathers linearly.

That distinction matters by storage tier:

| Storage | Approximate-stage pressure           | Refine-stage pressure           | Net effect on frontier                                  |
| ------- | ------------------------------------ | ------------------------------- | ------------------------------------------------------- |
| `DRAM`  | scan bandwidth + cache pressure + CPU  | cheap random gather               | smallest scan payload wins unless quality is too weak      |
| `SSD`   | scan bandwidth + cold-cache NVMe reads | still manageable random gather   | similar shape to `DRAM`, with slightly larger scan penalty |
| `OBS`   | remote range-read bandwidth            | remote random gather amplification | families that stay at low `rf` win disproportionately    |

#### 5.1.4 Budget Proxies Implied By Measured Latency

The report does not have stage-isolated tracing, so the following numbers are **blended proxies**, not direct instrumentation:

- **scan-bandwidth proxy** = approximate-stage bytes / end-to-end mean latency
- **logical-gather-rate proxy** = refine gathers / end-to-end mean latency

They are still useful because they show which stage the storage tier can absorb more easily.

| Anchor config     | Tier   | Mean latency | Approx bytes | Refine gathers | Scan-bandwidth proxy | Logical-gather-rate proxy |
| ----------------- | ------ | ------------ | ------------ | -------------- | -------------------- | ------------------------- |
| `RQ np=128 rf=2`  | `DRAM` | `872ms`      | `~650MB`     | `100,000`      | `~0.75 GB/s`         | `~115K gathers/s`         |
| `SQ np=128 rf=1`  | `DRAM` | `947ms`      | `~4.63GB`    | `50,000`       | `~4.89 GB/s`         | `~53K gathers/s`          |
| `RQ np=128 rf=2`  | `OBS`  | `18,729ms`   | `~650MB`     | `100,000`      | `~0.035 GB/s`        | `~5.3K gathers/s`         |
| `USQ4 np=128 rf=1` | `OBS`  | `5,690ms`    | `~3.04GB`    | `50,000`       | `~0.53 GB/s`         | `~8.8K gathers/s`         |
| `USQ8 np=128 rf=1` | `OBS`  | `7,136ms`    | `~5.35GB`    | `50,000`       | `~0.75 GB/s`         | `~7.0K gathers/s`         |

The proxy table sharpens the tier-level story:

- Local tiers can sustain multi-GB/s aggregate scan rates, so the main fight is **which family minimizes scan bytes while keeping recall high enough**.
- OBS can still move large streamed ranges, but logical gather rate collapses by roughly an order of magnitude relative to local tiers.
- That is exactly why `RQ` can remain excellent locally yet fall behind remotely: its cheap scan budget is not enough to compensate for the extra refine amplification.

#### 5.1.5 What `analyze_plan()` Actually Shows

The benchmark now also has a lightweight stage probe using `Scanner.analyze_plan()` on one representative query. This is still not full physical I/O tracing, but it does expose the runtime shape of the vector-search plan:

- `ANNSubIndex` / `ANNIvfPartition` for the approximate index search
- `Take` for candidate materialization
- `KNNVectorDistance` for exact-distance refinement over the materialized candidates

Observed single-shard node-family aggregates (`seed=42`, representative post-warmup query, `top_k=10000`):

| Case               | Tier    | Approximate node-family elapsed sum (`ANNSubIndex` + `ANNIvfPartition`) | Refine node-family elapsed sum (`KNNVectorDistance` + `Take`) | Key raw nodes |
| ------------------ | ------- | ------------------------------------------------------------------------ | ------------------------------------------------------------- | ------------- |
| `SQ np=128 rf=1`   | `local` | `470.4ms`                                                                | `1133.5ms`                                                    | `ANNSubIndex=468.9ms`, `Take=470.6ms`, `KNNVectorDistance=662.8ms` |
| `USQ4 np=128 rf=1` | `local` | `256.3ms`                                                                | `741.5ms`                                                     | `ANNSubIndex=254.9ms`, `Take=256.8ms`, `KNNVectorDistance=484.7ms` |
| `USQ8 np=128 rf=1` | `OBS`   | `790.5ms`                                                                | `10570.9ms`                                                   | `ANNSubIndex=788.5ms`, `Take=790.7ms`, `KNNVectorDistance=9780.2ms` |

Coverage note:

- this stage-sample table currently covers `SQ`, `USQ4`, and `USQ8`;
- there is no live `PCA-512 RQ` dataset path left on ECS today, so the canonical `RQ` family is still missing a same-contract `analyze_plan()` sample;
- the only reachable live `RQ` plan sample is from the `1024D` baseline dataset (`/data/work/tmp/s1b/shard-0.lance`), which is intentionally excluded from this `PCA-512` table.

These samples add two useful facts on top of the earlier proxy analysis:

- For local `PCA-512`, `USQ4` is indeed lighter than `SQ` in both the approximate node (`ANNSubIndex`) and the upper refinement nodes (`Take` + `KNNVectorDistance`).
- On OBS, the approximate node is not the dominant cost even for `USQ8`; the upper refinement path grows to multi-second scale. That is consistent with the earlier claim that remote candidate materialization and exact-distance refinement are much more expensive than local rerank.

Important limitation:

- these aggregates are **not** mutually-exclusive stage wall-clock timings; `analyze_plan()`
  prints subtree elapsed ranges, so summing parent and child nodes (`ANNSubIndex` + `ANNIvfPartition`,
  `KNNVectorDistance` + `Take`) double-counts time;
- these plan samples do **not** yet expose trustworthy physical `bytes_read / iops / requests` for vector search;
- the sampled plans did not emit standalone `LanceRead` nodes, so the current `read_stage` summary stays zero;
- use the `analyze_plan()` samples as **node-family elapsed evidence**, not as final physical I/O accounting
  or precise phase-exclusive latency splits.

### 5.2 Family Summary Under The Canonical Matrix

| Family | Approximate-stage bytes / candidate | What it buys                                        | Local role                          | OBS role                               |
| ------ | ----------------------------------- | --------------------------------------------------- | ----------------------------------- | -------------------------------------- |
| `RQ`   | `72B`                               | smallest scan budget, but needs more rerank to reach high recall   | best low/medium-recall local family   | weak, because `rf>1` becomes expensive remotely |
| `SQ`   | `512B`                              | much larger scan budget, but very strong `rf=1` quality            | best local high-recall family         | workable, but scan budget is heavy on OBS      |
| `USQ4` | `336B`                              | lower scan budget than `SQ`, stronger `rf=1` quality than `RQ`     | not good enough to beat `RQ/SQ` locally | best low-latency OBS family                 |
| `USQ8` | `592B`                              | strongest `rf=1` quality, highest approximate-stage byte cost      | narrow local niche only               | best OBS family for `0.96+` and max recall    |

This reduces the cross-family interpretation to one sentence:

- local tiers reward the smallest approximate-stage byte budget that still reaches the target recall,
- remote tiers reward the family that avoids refine amplification, even if its approximate-stage scan budget is larger.

### 5.3 SQ Ceiling And USQ Split

`SQ` still plateaus at `0.97230` under the revalidated local matrix. That keeps one older conclusion intact: `SQ` remains the best local high-recall family up to its scalar-quantization ceiling.

The more important update is how to read `RQ`, `USQ4`, and `USQ8` through stage budgets instead of generic family labels.

#### 5.3.1 Why `RQ` Wins Locally But Loses On OBS

| Config             | Recall  | Approx bytes across 5 shards | Refine gathers | Mean latency                 |
| ------------------ | ------- | ---------------------------- | -------------- | ---------------------------- |
| `RQ np=128 rf=2`   | `0.941` | `~650MB`  | `100,000` | `872ms` local / `18,729ms` OBS  |
| `SQ np=128 rf=1`   | `0.957` | `~4.63GB` | `50,000`  | `947ms` local / `10,518ms` OBS  |
| `USQ8 np=128 rf=1` | `0.960` | `~5.35GB` | `50,000`  | `1,021ms` local / `7,136ms` OBS |

This is the cleanest first-principles explanation for the tier split:

- On local storage, `RQ` wins because its scan budget is tiny. Doubling refine gathers from `50,000` to `100,000` is still cheap enough.
- On OBS, the same `rf=2` move is much more expensive because refine is a remote random-gather stage.
- So `RQ` loses remotely not because its scan bytes are too large, but because it needs more gather amplification to enter the same recall band.

Method note:

- unlike `SQ`, `USQ4`, and `USQ8`, this `RQ` explanation is still budget-driven rather than node-timing-driven;
- the live `PCA-512 RQ` path was no longer available when `analyze_plan()` probes were added, so the `RQ` row above is supported by the canonical latency matrix plus the scan-byte / gather-count model, not by a same-contract `ANNSubIndex` / `Take` / `KNNVectorDistance` sample.

#### 5.3.2 Why `USQ4` Wins The Low-Latency OBS Region

`USQ4` is the best compromise between the two stage budgets:

- compared with `SQ`, it cuts approximate-stage bytes from `~4.63GB` to `~3.04GB` at `np=128`,
- compared with `RQ`, it keeps `rf=1` at useful recall instead of paying `rf=2`,
- that combination is exactly what the OBS frontier rewards.

That is why the first strong OBS production point is:

- `USQ4 np=128 rf=1` -> `0.95060` recall, `5,690ms`

not `RQ` and not `SQ`.

#### 5.3.3 Why `USQ8` Is Not A Strict Upgrade Over `USQ4`

| Config              | Recall    | Approx bytes across 5 shards | Refine gathers | Mean latency |
| ------------------- | --------- | ---------------------------- | -------------- | ------------ |
| `USQ4 np=1024 rf=2` | `0.97182` | `~24.3GB`                    | `100,000`      | `17,583ms`   |
| `USQ8 np=1024 rf=1` | `0.97202` | `~42.8GB`                    | `50,000`       | `21,190ms`   |

Near `~0.972` recall on OBS:

- `USQ8` halves logical gathers from `100,000` to `50,000`,
- but it also adds roughly `+18.5GB` of approximate-stage bytes per query across 5 shards,
- and that extra remote scan bandwidth is still expensive enough that `USQ4 np=1024 rf=2` remains faster.

So the `USQ4` / `USQ8` split is now precise:

- `USQ4` is the latency-first OBS choice,
- `USQ8` is the recall-first OBS choice,
- and the boundary between them is governed by scan-bandwidth cost versus refine-gather amplification.

### 5.4 Build-Side Confidence

Query-side conclusions are now on solid ground. Build-side evidence is still asymmetric:

- `USQ4` and `USQ8` have clean enough build notes for planning
- `RQ` has strong per-shard historical logs, but not a single normalized wall-clock summary
- `SQ` still lacks a clean canonical build record because the earlier build logs include out-of-space retries

That means the current report can recommend query configs with confidence, but a truly final build-cost comparison still needs one more normalization pass for `RQ` and `SQ`.

---

## 6. Recommendations By Scenario

The practical default choices are now straightforward:

- If the target tier is `DRAM` or `SSD`, build `RQ` for low/medium recall and `SQ` for high recall.
- If the target tier is `OBS`, build `USQ4` first; add `USQ8` only when the target is `0.96+` recall.
- Do not build a generic `USQ` local deployment just because it looked good in the older mixed report.

| Scenario                                  | Recommended Config  | Latency      | Recall   | Why This Is The Default                     |
| ----------------------------------------- | ------------------- | ------------ | -------- | ------------------------------------------- |
| `DRAM`, cheapest acceptable result        | `RQ np=128 rf=1`    | **534ms**    | 0.789    | lowest local latency                        |
| `DRAM`, around `0.94` recall              | `RQ np=128 rf=2`    | **872ms**    | 0.941    | cheapest jump into the `0.94` band          |
| `DRAM`, around `0.96` recall              | `SQ np=128 rf=1`    | **947ms**    | 0.957    | first `0.95+` local frontier point          |
| `DRAM`, best revalidated local ceiling    | `SQ np=1024 rf=2`   | **2,428ms**  | 0.972    | highest local recall observed               |
| `SSD`, around `0.95` recall               | `RQ np=256 rf=2`    | **962ms**    | 0.946    | best SSD point near `0.95`                  |
| `SSD`, around `0.96` recall               | `SQ np=128 rf=1`    | **972ms**    | 0.957    | first `0.95+` SSD frontier point            |
| `OBS`, lowest-latency production point    | `USQ4 np=128 rf=1`  | **5,690ms**  | 0.951    | fastest revalidated OBS point               |
| `OBS`, slightly higher low-latency recall | `USQ4 np=256 rf=1`  | **6,824ms**  | 0.954    | best `~0.954` OBS point                     |
| `OBS`, first `0.96+` point                | `USQ8 np=128 rf=1`  | **7,136ms**  | 0.960    | cheapest way into the `0.96+` band          |
| `OBS`, around `0.967` recall              | `USQ8 np=256 rf=1`  | **8,823ms**  | 0.967    | best mid/high-recall OBS point              |
| `OBS`, near `0.972` with better latency   | `USQ4 np=1024 rf=2` | **17,583ms** | 0.972    | faster than the closest `USQ8` point        |
| `OBS`, absolute recall first              | `USQ8 np=1024 rf=1` | **21,190ms** | 0.972    | only if the extra `+0.00020` recall matters |

### Artifact Paths

| Item                        | Path                                                     |
| --------------------------- | -------------------------------------------------------- |
| Revalidated ledger          | `docs/bench/wiki/experiments/pca512-usq-revalidation.md` |
| `RQ` legacy summary         | `/data/work/tmp/pca512-rq-legacy-summary.json`           |
| `SQ` legacy summary         | `/data/work/tmp/pca512-sq-legacy-summary.json`           |
| `USQ4` local legacy summary | `/data/work/tmp/pca512-usq4-legacy-summary.json`         |
| `USQ4` OBS legacy summary   | `/data/work/tmp/pca512-usq4-obs-legacy-summary.json`     |
| `USQ8` legacy summary       | `/data/work/tmp/pca512-usq8-legacy-summary.json`         |
| Positional GT               | `/data/work/tmp/s1b-pca512/gt_positional_v2.npz`         |
| OBS prefix (`USQ4`)         | `s3://knowledgebase-5f43/pca512-usq4-1b/`                |
| OBS prefix (`USQ8`)         | `s3://knowledgebase-5f43/pca512-1b/`                     |

---

## 7. Optimization Roadmap

Current best (DRAM): 2,513ms for recall 0.993. Knowhere achieves ~50ms at similar recall. 30x gap.

| Priority   | Optimization                         | Expected Impact                              | Status      |
| ---------- | ------------------------------------ | -------------------------------------------- | ----------- |
| P0         | Multi-bit RQ (4-6 bit levels)        | rf=1 → 0.95 recall, save ~350ms/rf           | Not started |
| P1         | SQ8 two-stage rerank (within IVF_RQ) | Replace float32 reads, 10x faster refinement | Not started |
| P2         | PQ FastScan (bbs=32, AVX-512 VNNI)   | 4-10x PQ distance throughput                 | Not started |
| P3         | Cross-query partition cache (LRU)    | 50-80% IO reduction on OBS                   | Not started |

**Projected** if all optimizations implemented: recall 0.99 in ~500ms (DRAM), ~15s (OBS).

---

## 8. Data Paths (ECS)

| Item                                                        | Path                                                      |
| ----------------------------------------------------------- | --------------------------------------------------------- |
| 1B shards (1024-dim)                                        | `/data/work/tmp/s1b/shard-{0-4}.lance`                    |
| PCA-512 shards (recyclable local path, current live `USQ4`) | `/data/work/tmp/s1b-pca512/shard-{0-4}.lance`             |
| SQ shards                                                   | `/data/work/tmp/s1b-pca512-sq/shard-{0-4}.lance`          |
| 1B results                                                  | `/data/work/tmp/pareto-results/`                          |
| PCA-512 results                                             | `/data/work/tmp/pca512-results-v2/`                       |
| SQ results                                                  | `/data/work/tmp/pca512-sq-results/`                       |
| `RQ` revalidated summary                                    | `/data/work/tmp/pca512-rq-legacy-summary.json`            |
| `SQ` revalidated summary                                    | `/data/work/tmp/pca512-sq-legacy-summary.json`            |
| `USQ4` revalidated local summary                            | `/data/work/tmp/pca512-usq4-legacy-summary.json`          |
| `USQ4` revalidated OBS summary                              | `/data/work/tmp/pca512-usq4-obs-legacy-summary.json`      |
| `USQ8` revalidated summary                                  | `/data/work/tmp/pca512-usq8-legacy-summary.json`          |
| GT (_rowid format)                                          | `/data/work/tmp/gt_pca512_shard0_top10k_5q_seed42_v2.npz` |
| GT (positional)                                             | `/data/work/tmp/s1b-pca512/gt_positional_v2.npz`          |
| OBS (1B)                                                    | `s3://knowledgebase-5f43/fineweb-edu-1b-rq-shard4/`       |
| OBS (PCA-512, `USQ8`)                                       | `s3://knowledgebase-5f43/pca512-1b/`                      |
| OBS (PCA-512, `USQ4`)                                       | `s3://knowledgebase-5f43/pca512-usq4-1b/`                 |
| OBS (SQ)                                                    | `s3://knowledgebase-5f43/pca512-sq-1b/`                   |
