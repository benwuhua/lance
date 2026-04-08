---
tags: [environment, entity]
date: 2026-04-08
---

# ECS Environment

## Host: ecs-hk-1b

| Spec | Value |
|------|-------|
| OS | Linux x86_64 |
| RAM | 493 GB |
| Disk | 12 TB NVMe RAID0 |
| Access | SSH (root) |
| Python venv | `/data/work/venvs/lance-benchmark/` |
| Lance version | 4.0.0 |
| Lance source | `/data/work/lance/` |

## Environment Variables

```bash
export TMPDIR=/data/work/tmp/lance-tmp    # System /tmp is only 40GB
export LANCE_IO_THREADS=128
```

## OBS Credentials

```
Endpoint: https://obs.ap-southeast-1.myhuaweicloud.com
Region: ap-southeast-1
AK: 0JIMBMDMUBP8UUPHWJ44
SK: J2V4tESvsdNoKUkkrdDJHpQvt9VwqV6dUUcKnwps
Bucket: knowledgebase-5f43
```

## OBS Upload Commands

```bash
# Upload a shard (use sync, not cp — cp requires interactive confirmation)
/usr/local/bin/obsutil sync /data/work/tmp/s1b/shard-0.lance/ \
    obs://knowledgebase-5f43/fineweb-edu-1b-rq-shard4/shard-0.lance/ -j 10

# List OBS contents
/usr/local/bin/obsutil ls obs://knowledgebase-5f43/ -s -limit 100
```

## Key Data Paths

| Data | Path |
|------|------|
| 1B baseline shards | `/data/work/tmp/s1b/shard-{0-4}.lance` |
| PCA-512 shards | `/data/work/tmp/s1b-pca512/shard-{0-4}.lance` |
| 324M legacy shards | `/data/work/tmp/lance-shards-rq/shard-{0-3}.lance` |
| 1B ground truth | `/data/work/tmp/gt_1b_shard0_top10k_5q_seed42.npz` |
| PCA-512 GT v2 | `/data/work/tmp/s1b-pca512/gt_positional_v2.npz` |
| 1B benchmark results | `/data/work/tmp/lance-benchmark-results/` |
| PCA-512 DRAM results | `/data/work/tmp/pca512-results-v2/` |
| Temp dir | `/data/work/tmp/lance-tmp/` |
| Source dataset | `s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-1b.lance` |

## OBS Data Layout

```
s3://knowledgebase-5f43/
├── fineweb-edu/fineweb-edu-1b.lance          # Source dataset (1B rows)
├── fineweb-edu-1b-rq-shard4/
│   └── shard-{0-4}.lance/                     # 1B baseline shards (1024-dim)
├── pca512-1b/
│   └── shard-{0-4}.lance/                     # PCA-512 shards (512-dim)
└── fineweb-edu-rq-shard4/                     # Legacy 324M (old)
```

## Known Issues

- **SSH disconnects**: Long-running SSH sessions may disconnect. Always use `nohup` for benchmark runs.
- **Page cache warming**: Takes ~40 min for 1B, reads all files sequentially. Warms index + raw vectors.
- **tmp space**: System `/tmp` is only 40GB. Always set `TMPDIR=/data/work/tmp/lance-tmp`.
- **obsutil cp vs sync**: `obsutil cp` requires interactive confirmation per file. Use `obsutil sync` instead.
