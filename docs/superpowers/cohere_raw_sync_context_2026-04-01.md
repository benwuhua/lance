# Cohere 1B Pipeline Context Handoff (2026-04-01)

## Goal
- Benchmark/query path for large-scale Cohere/FineWeb vectors in object storage.
- Current focus: switch from direct HuggingFace sequential ingest to **parallel raw shard sync to OBS**, then ingest from OBS raw.

## Environment
- Remote ECS: `ecs-119-8-59-254`
- Spec: 64 vCPU, 493 GiB RAM
- Work root: `/data/work`
- Repo on remote: `/data/work/lance` (branched from local `.worktree/lance-cohere1m-topk100k`)
- Logs dir: `/data/work/logs`
- Current sync cache root: `/data/work/tmp/fineweb-edu-raw-sync-batch-5`
- Object bucket: `knowledgebase-5f43`
- OBS endpoint: `https://obs.ap-southeast-1.myhuaweicloud.com`
- Region: `ap-southeast-1`

## Current code state (local)
- Added/used script: `benchmarks/cohere/sync_hf_shards_to_obs_raw.py`
  - Parallel sync workers from HF shard list to OBS raw path
  - Supports `--start-shard`, `--max-shards`, `--workers`, `--skip-existing`
- Data ingest uses OBS-first path via `benchmarks/cohere/ingest_fineweb_edu.py`
  - Supports shard slicing/batching and raw fallback logic
  - Prefer OBS raw if available, fallback to HF when missing

## Test status before handoff
- `python/.venv/bin/python -m pytest /Users/ryan/Code/vectorDB/lance/.worktree/lance-cohere1m-topk100k/python/python/tests/test_fineweb_edu_ingest.py -q`
  - Passed: 15
- `python/.venv/bin/python -m pytest /Users/ryan/Code/vectorDB/lance/.worktree/lance-cohere1m-topk100k/python/python/tests/test_cohere_topk_latency.py -q`
  - Passed: 22

## Active remote task (as checked now)
- Process: `71422`
- Command:
  - `python .../sync_hf_shards_to_obs_raw.py --cache-dir /data/work/tmp/fineweb-edu-raw-sync-batch-5 --bucket knowledgebase-5f43 --endpoint https://obs.ap-southeast-1.myhuaweicloud.com --access-key ... --secret-key ... --start-shard 5 --max-shards 8 --workers 4 --skip-existing`
- Runtime: `31m06s+` and still alive.
- Log file exists but is still size `0` bytes (`/data/work/logs/fineweb-edu-raw-sync-batch-5.out`), so progress is read from cache-file states.

## Local cache progress (shard batch `5..12`)
- Completed `.npy` files: 5
  - `new_CC-MAIN-2013-20-train-00005-of-00014.npy`
  - `00006`
  - `00007`
  - `00008`
  - `00009`
- Incomplete `.npy.incomplete`: 3
  - `00010`
  - `00011`
  - `00012`
- Approx batch completion: `5/8 = 62.5%` by file state.

## Interpretation
- Task is progressing and not hung.
- Major issue previously encountered (cache contention from concurrent tasks) has been avoided by using independent cache dir.
- Bottleneck now is remaining shard downloads/writes (expected while `.npy.incomplete` are present).
- Log silence (`0` bytes) is not an error by itself.

## Last known important paths
- Sync job script: `[sync_hf_shards_to_obs_raw.py](/Users/ryan/Code/vectorDB/lance/.worktree/lance-cohere1m-topk100k/benchmarks/cohere/sync_hf_shards_to_obs_raw.py)`
- Ingest script: `[ingest_fineweb_edu.py](/Users/ryan/Code/vectorDB/lance/.worktree/lance-cohere1m-topk100k/benchmarks/cohere/ingest_fineweb_edu.py)`
- Existing 1M datasets:
  - `s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-raw-1m.lance`
  - `s3://knowledgebase-5f43/fineweb-edu/fineweb-edu-raw-1m-ray.lance`

## Next action for next model
1. Keep watch on PID `71422` until shards `00010-00012` finish.
2. Verify files are uploaded into OBS raw path:
   - `fineweb-edu-emb-raw/emb/`
3. Start next batch window (e.g. `--start-shard 13 --max-shards ...`) once current batch ends.
4. Only when enough raw data exists, run `ingest_fineweb_edu.py` with append mode from OBS raw.
