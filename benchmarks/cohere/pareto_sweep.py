#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Pareto-optimal retrieval configuration sweep for DRAM, SSD, OBS.

Runs bench.py with multiple (nprobes, refine_factor) combinations across
storage media, computes recall against persisted ground truth, and
identifies Pareto-optimal configurations.

Usage:
  # Full sweep (DRAM + SSD + OBS)
  python benchmarks/cohere/pareto_sweep.py \
      --media dram,ssd,obs \
      --nprobes-list 64,128,256,512,1024 \
      --rf-list 1,2,3,5 \
      --shard-dir /data/work/tmp/lance-shards-rq \
      --gt-path /data/work/tmp/gt_shard0_top10k_5q_seed42.npz \
      --result-dir /data/work/tmp/pareto-results

  # DRAM only (fast, for development)
  python benchmarks/cohere/pareto_sweep.py \
      --media dram \
      --nprobes-list 128,256,512 \
      --rf-list 1,2,3 \
      --shard-dir /data/work/tmp/lance-shards-rq \
      --gt-path /data/work/tmp/gt_shard0_top10k_5q_seed42.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ── Configuration ────────────────────────────────────────────────────

BENCH_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bench.py"
)
OBS_SHARD_URI = "s3://knowledgebase-5f43/fineweb-edu-1b-rq-shard4"

# Fixed benchmark parameters
TOP_K = 10000
NUM_SHARDS = 5
QUERY_COUNT = 8   # 3 warmup + 5 timed (matching GT's 5 queries)
WARMUP = 3
SEED = 42
COLUMN = "vector"
METRIC = "cosine"
INDEX_TYPE = "IVF_RQ"


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_media_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


# ── Run bench.py for one configuration ──────────────────────────────

def run_single_config(
    *,
    storage: str,
    nprobes: int,
    refine_factor: int,
    shard_dir: str,
    result_path: str,
) -> dict:
    """Run bench.py query for a single (nprobes, rf, storage) config."""
    # For OBS, use S3 shard dir instead of local
    effective_shard_dir = shard_dir
    if storage == "obs":
        effective_shard_dir = OBS_SHARD_URI

    cmd = [
        sys.executable, "-u", BENCH_SCRIPT, "query",
        "--plan", "B",
        "--storage", storage,
        "--index-type", INDEX_TYPE,
        "--shard-dir", effective_shard_dir,
        "--nprobes", str(nprobes),
        "--refine-factor", str(refine_factor),
        "--top-k", str(TOP_K),
        "--num-shards", str(NUM_SHARDS),
        "--query-count", str(QUERY_COUNT),
        "--warmup", str(WARMUP),
        "--seed", str(SEED),
        "--column", COLUMN,
        "--metric", METRIC,
        "--result-path", result_path,
        "--no-warm",
    ]
    # For OBS: clear AWS env vars to avoid duplicate field error in lance
    # Credentials are passed via --access-key/--secret-key CLI args
    if storage == "obs":
        cmd.extend([
            "--access-key", os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "--secret-key", os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        ])

    print(f"\n  Running: {storage} np={nprobes} rf={refine_factor}")
    t0 = time.perf_counter()
    # Clear AWS env vars for OBS subprocess to avoid duplicate field conflict
    env_override = None
    if storage == "obs":
        env_override = {k: v for k, v in os.environ.items()
                        if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                                      "AWS_ACCESS_KEY", "AWS_SECRET_KEY")}
    result = subprocess.run(cmd, capture_output=False, text=True,
                           env=env_override or None)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s)")
        return {"storage": storage, "nprobes": nprobes,
                "refine_factor": refine_factor, "error": "non-zero exit"}

    # Read result JSON
    if Path(result_path).exists():
        with open(result_path) as f:
            data = json.load(f)
        # Flatten: bench.py nests results under "result" key
        if "result" in data and isinstance(data["result"], dict):
            data.update(data["result"])
        if "config" in data and isinstance(data["config"], dict):
            data.update(data["config"])
        data["wall_time_s"] = elapsed
        return data

    return {"storage": storage, "nprobes": nprobes,
            "refine_factor": refine_factor, "error": "no result file"}


# ── Recall computation ──────────────────────────────────────────────

def compute_recall_from_gt(
    *,
    gt_ids: np.ndarray,          # (num_queries, k) uint64
    shard_dir: str,
    nprobes: int,
    refine_factor: int,
    query_vectors: np.ndarray,   # (num_queries, dim) float32
) -> float:
    """Compute recall@K for shard-0 ANN results against GT."""
    import lance

    k = gt_ids.shape[1]
    shard_uri = f"{shard_dir.rstrip('/')}/shard-0.lance"
    ds = lance.dataset(shard_uri)

    recalls = []
    for i, qvec in enumerate(query_vectors):
        result = ds.to_table(
            columns=["_rowid"],
            nearest={
                "column": COLUMN,
                "q": qvec.tolist(),
                "k": k,
                "use_index": True,
                "metric": METRIC,
                "nprobes": nprobes,
                "refine_factor": refine_factor,
            },
        )
        ann_ids = set(result["_rowid"].to_numpy().tolist())
        gt_set = set(gt_ids[i].tolist())
        overlap = len(ann_ids & gt_set)
        recalls.append(overlap / k)

    return sum(recalls) / len(recalls)


# ── Media-specific setup ────────────────────────────────────────────

def setup_dram(shard_dir: str):
    """Warm page cache for DRAM runs."""
    print("\n[Setup] Warming page cache for DRAM...")
    # Import from sibling bench.py (may not be on sys.path when run directly)
    sys.path.insert(0, os.path.dirname(BENCH_SCRIPT))
    from bench import warm_page_cache
    t0 = time.perf_counter()
    warm_page_cache(shard_dir, NUM_SHARDS)
    print(f"  Page cache warmed in {time.perf_counter() - t0:.0f}s")


def setup_ssd():
    """Drop page cache for SSD runs (needs root)."""
    print("\n[Setup] Dropping page cache for SSD...")
    subprocess.run(
        ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=True,
    )
    print("  Page cache dropped OK")


# ── Sweep execution ─────────────────────────────────────────────────

def sweep_medium(
    *,
    medium: str,
    nprobes_list: list[int],
    rf_list: list[int],
    shard_dir: str,
    result_dir: str,
) -> list[dict]:
    """Run all (nprobes, rf) configs for one storage medium."""
    results = []

    for nprobes in nprobes_list:
        for rf in rf_list:
            # OBS: skip rf > 5 (extremely slow on S3)
            if medium == "obs" and rf > 5:
                print(f"  Skipping OBS np={nprobes} rf={rf} (rf>2 impractical on S3)")
                continue

            label = f"{medium}-np{nprobes}-rf{rf}"
            result_path = os.path.join(result_dir, f"{label}.json")

            result = run_single_config(
                storage=medium,
                nprobes=nprobes,
                refine_factor=rf,
                shard_dir=shard_dir,
                result_path=result_path,
            )
            result["label"] = label
            results.append(result)

            # Drop caches between SSD configs
            if medium == "ssd":
                try:
                    setup_ssd()
                except (subprocess.CalledProcessError, PermissionError):
                    print("  WARNING: Could not drop caches between configs")

    return results


# ── Pareto analysis ─────────────────────────────────────────────────

def find_pareto_front(
    results: list[dict],
    *,
    objective_diminishing: str = "latency_ms.mean",  # minimize
    objective_improving: str = "recall",              # maximize
) -> list[dict]:
    """Find Pareto-optimal points: no other point is both faster AND higher recall."""
    # Filter out errors
    valid = [r for r in results if "error" not in r and "latency_ms" in r]
    if not valid:
        return []

    # Sort by recall descending (dominant objective)
    valid.sort(key=lambda r: -r.get(objective_improving, 0))

    pareto = []
    best_latency = float("inf")
    for r in valid:
        lat = _get_nested(r, objective_diminishing)
        rec = r.get(objective_improving, 0)
        if lat < best_latency:
            pareto.append(r)
            best_latency = lat

    return pareto


def _get_nested(d: dict, key: str):
    """Get nested dict value like 'latency_ms.mean'."""
    keys = key.split(".")
    val = d
    for k in keys:
        val = val[k] if isinstance(val, dict) else val
    return val


def recommend_configs(
    all_results: dict[str, list[dict]],
    recall_targets: list[float] = None,
) -> dict[str, dict[float, dict]]:
    """For each medium and recall target, find the config with lowest latency."""
    if recall_targets is None:
        recall_targets = [0.90, 0.95, 0.99]

    recommendations = {}

    for medium, results in all_results.items():
        valid = [r for r in results if "error" not in r and "latency_ms" in r]
        valid.sort(key=lambda r: _get_nested(r, "latency_ms.mean"))
        medium_rec = {}

        for target in recall_targets:
            best = None
            for r in valid:
                if r.get("recall", 0) >= target:
                    best = r
                    break
            if best:
                medium_rec[target] = {
                    "nprobes": best["nprobes"] if "nprobes" in best else best.get("config", {}).get("nprobes"),
                    "refine_factor": best["refine_factor"] if "refine_factor" in best else best.get("config", {}).get("refine_factor"),
                    "recall": best.get("recall", 0),
                    "mean_latency_ms": _get_nested(best, "latency_ms.mean"),
                }
            else:
                medium_rec[target] = None

        recommendations[medium] = medium_rec

    return recommendations


# ── Report generation ───────────────────────────────────────────────

def print_sweep_table(all_results: dict[str, list[dict]]):
    """Print summary table for all sweep results."""
    print("\n" + "=" * 100)
    print(f"{'Label':<30} {'Mean(ms)':>10} {'P50(ms)':>10} {'P99(ms)':>10} {'Recall':>8}")
    print("-" * 100)

    for medium in ["dram", "ssd", "obs"]:
        results = all_results.get(medium, [])
        valid = [r for r in results if "error" not in r and "latency_ms" in r]
        valid.sort(key=lambda r: _get_nested(r, "latency_ms.mean"))

        if valid:
            print(f"\n  [{medium.upper()}]")

        for r in valid:
            lat = r["latency_ms"] if isinstance(r.get("latency_ms"), dict) else r.get("latency_ms", {})
            if isinstance(lat, dict):
                mean_l = lat.get("mean", 0)
                p50 = lat.get("p50", 0)
                p99 = lat.get("p99", 0)
            else:
                mean_l = p50 = p99 = 0

            recall = r.get("recall", 0)
            label = r.get("label", "")
            print(
                f"  {label:<28} "
                f"{mean_l:>10.1f} "
                f"{p50:>10.1f} "
                f"{p99:>10.1f} "
                f"{recall:>8.4f}"
            )

    print("=" * 100)


def print_recommendations(recommendations: dict):
    """Print recommended configs per recall target."""
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 80)

    for medium in ["dram", "ssd", "obs"]:
        rec = recommendations.get(medium, {})
        print(f"\n  [{medium.upper()}]")
        for target in sorted(rec.keys()):
            cfg = rec[target]
            if cfg is None:
                print(f"    recall >= {target}: NOT ACHIEVABLE")
            else:
                print(
                    f"    recall >= {target}: np={cfg['nprobes']}, rf={cfg['refine_factor']} "
                    f"-> recall={cfg['recall']:.4f}, latency={cfg['mean_latency_ms']:.0f}ms"
                )

    print("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pareto sweep for optimal retrieval configuration"
    )
    parser.add_argument("--media", default="dram,ssd,obs",
                        help="Comma-separated media to sweep")
    parser.add_argument("--nprobes-list", default="64,128,256,512,1024",
                        help="Comma-separated nprobes values")
    parser.add_argument("--rf-list", default="1,2,3,5",
                        help="Comma-separated refine factor values")
    parser.add_argument("--shard-dir", required=True,
                        help="Local shard directory")
    parser.add_argument("--gt-path", required=True,
                        help="Path to ground truth .npz file")
    parser.add_argument("--result-dir", default="/data/work/tmp/pareto-results",
                        help="Directory for result JSON files")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip running benchmarks, only analyze existing results")
    parser.add_argument("--skip-recall", action="store_true",
                        help="Skip recall computation (use saved results only)")
    parser.add_argument("--no-warm", action="store_true",
                        help="Skip page cache warming (if already warm)")

    args = parser.parse_args()

    media_list = parse_media_list(args.media)
    nprobes_list = parse_int_list(args.nprobes_list)
    rf_list = parse_int_list(args.rf_list)

    os.makedirs(args.result_dir, exist_ok=True)

    # Load ground truth
    print(f"Loading ground truth from {args.gt_path}...")
    gt_data = np.load(args.gt_path)
    gt_ids = gt_data["gt_ids"]
    print(f"  GT shape: {gt_ids.shape} ({gt_ids.shape[0]} queries, k={gt_ids.shape[1]})")

    # Load query vectors (re-generate with same seed)
    import lance
    ds0 = lance.dataset(f"{args.shard_dir.rstrip('/')}/shard-0.lance")
    rng = np.random.default_rng(SEED)
    total_rows = ds0.count_rows()
    query_indices = rng.choice(total_rows, size=min(QUERY_COUNT - WARMUP, total_rows), replace=False)
    query_table = ds0.take(query_indices, columns=[COLUMN])
    query_vectors = np.stack(query_table[COLUMN].to_pylist())
    print(f"  Query vectors: {query_vectors.shape}")

    # ── Run sweeps ──
    all_results = {}

    if not args.skip_sweep:
        # Setup media in correct order (DRAM first to warm cache, then SSD cold, then OBS)
        for medium in media_list:
            print(f"\n{'=' * 60}")
            print(f"  Sweeping {medium.upper()}")
            print(f"{'=' * 60}")

            if medium == "dram" and not args.no_warm:
                setup_dram(args.shard_dir)
            elif medium == "ssd":
                setup_ssd()

            results = sweep_medium(
                medium=medium,
                nprobes_list=nprobes_list,
                rf_list=rf_list,
                shard_dir=args.shard_dir,
                result_dir=args.result_dir,
            )
            all_results[medium] = results
    else:
        # Load existing results
        print("\nLoading existing results...")
        for medium in media_list:
            results = []
            for nprobes in nprobes_list:
                for rf in rf_list:
                    path = os.path.join(args.result_dir, f"{medium}-np{nprobes}-rf{rf}.json")
                    if Path(path).exists():
                        with open(path) as f:
                            data = json.load(f)
                        # Flatten: bench.py nests results under "result" key
                        if "result" in data and isinstance(data["result"], dict):
                            data.update(data["result"])
                        if "config" in data and isinstance(data["config"], dict):
                            data.update(data["config"])
                        data["label"] = f"{medium}-np{nprobes}-rf{rf}"
                        results.append(data)
            all_results[medium] = results

    # ── Compute recall ──
    if not args.skip_recall:
        print(f"\n{'=' * 60}")
        print("  Computing recall against ground truth")
        print(f"{'=' * 60}")

        for medium, results in all_results.items():
            if medium == "obs":
                print(f"  Skipping recall for OBS (no local GT comparison)")
                continue
            for r in results:
                if "error" in r:
                    continue
                # Extract nprobes and rf from config or result
                cfg = r.get("config", r)
                np_val = cfg.get("nprobes", r.get("nprobes"))
                rf_val = cfg.get("refine_factor", r.get("refine_factor"))

                if np_val is None or rf_val is None:
                    continue

                print(f"  recall(shard-0): {medium} np={np_val} rf={rf_val}...", end=" ", flush=True)
                recall = compute_recall_from_gt(
                    gt_ids=gt_ids,
                    shard_dir=args.shard_dir,
                    nprobes=np_val,
                    refine_factor=rf_val,
                    query_vectors=query_vectors,
                )
                r["recall"] = recall
                print(f"{recall:.4f}")

    # ── Analysis ──
    print_sweep_table(all_results)

    recommendations = recommend_configs(all_results)
    print_recommendations(recommendations)

    # Find Pareto fronts per medium
    for medium in media_list:
        pareto = find_pareto_front(all_results.get(medium, []))
        if pareto:
            print(f"\n  Pareto front ({medium.upper()}):")
            for p in pareto:
                lat = _get_nested(p, "latency_ms.mean")
                rec = p.get("recall", 0)
                cfg = p.get("config", p)
                np_val = cfg.get("nprobes", p.get("nprobes"))
                rf_val = cfg.get("refine_factor", p.get("refine_factor"))
                print(f"    np={np_val} rf={rf_val} -> recall={rec:.4f}, latency={lat:.0f}ms")

    # Save full results
    output_path = os.path.join(args.result_dir, "sweep_summary.json")
    with open(output_path, "w") as f:
        # Strip non-serializable fields
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if k != "ann_ids"}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump({
            "recommendations": clean(recommendations),
            "all_results": clean(all_results),
        }, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
