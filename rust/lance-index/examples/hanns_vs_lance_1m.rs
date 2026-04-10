// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark comparing Hanns HNSW vs Lance HNSW on Cohere 1M dataset.
//!
//! Reads big-ann-benchmarks format files (.fbin / .ibin) from a data directory
//! and compares build time, search latency, QPS, and recall across ef sweeps.
//!
//! Data format (.fbin):
//!   [4 bytes: n (u32 LE)] [4 bytes: dim (u32 LE)] [n * dim * 4 bytes: f32 row-major]
//!
//! Data format (.ibin):
//!   [4 bytes: n (u32 LE)] [4 bytes: k (u32 LE)] [n * k * 4 bytes: i32 row-major]
//!
//! Run with:
//!   cargo run --example hanns_vs_lance_1m --features hanns --release -- <data_dir>
//!
//! where <data_dir> contains: base.fbin, query.fbin, gt.ibin

#[cfg(feature = "hanns")]
use std::error::Error;
#[cfg(feature = "hanns")]
use std::fs::File;
#[cfg(feature = "hanns")]
use std::io::Read;
#[cfg(feature = "hanns")]
use std::sync::Arc;
#[cfg(feature = "hanns")]
use std::time::Instant;

#[cfg(feature = "hanns")]
use arrow_array::{FixedSizeListArray, Float32Array};
#[cfg(feature = "hanns")]
use lance_arrow::FixedSizeListArrayExt;
#[cfg(feature = "hanns")]
use lance_index::vector::{
    flat::storage::FlatFloatStorage,
    hnsw::builder::{HNSW, HnswBuildParams, HnswQueryParams},
    v3::subindex::IvfSubIndex,
};
#[cfg(feature = "hanns")]
use lance_linalg::distance::DistanceType;

#[cfg(feature = "hanns")]
const TOP_K: usize = 10;
#[cfg(feature = "hanns")]
const HNSW_M: usize = 16;
#[cfg(feature = "hanns")]
const EF_CONSTRUCTION: usize = 200;
#[cfg(feature = "hanns")]
const EF_SWEEP: [usize; 5] = [50, 100, 200, 400, 800];

#[cfg(feature = "hanns")]
fn read_fbin(path: &std::path::Path) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let mut buf = Vec::new();
    let mut f = File::open(path)?;
    f.read_to_end(&mut buf)?;
    if buf.len() < 8 {
        return Err(format!("file too small: {}", path.display()).into());
    }
    let n = u32::from_le_bytes(buf[0..4].try_into()?) as usize;
    let dim = u32::from_le_bytes(buf[4..8].try_into()?) as usize;
    let expected = 8 + n * dim * 4;
    if buf.len() < expected {
        return Err(format!(
            "fbin size mismatch: expected {} bytes, got {} ({})",
            expected,
            buf.len(),
            path.display()
        )
        .into());
    }
    // Reinterpret the raw bytes as native-endian f32 (LE on all targets we care about).
    let data = buf[8..8 + n * dim * 4]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok((data, n, dim))
}

#[cfg(feature = "hanns")]
fn read_ibin(path: &std::path::Path) -> Result<(Vec<i32>, usize, usize), Box<dyn Error>> {
    let mut buf = Vec::new();
    let mut f = File::open(path)?;
    f.read_to_end(&mut buf)?;
    if buf.len() < 8 {
        return Err(format!("file too small: {}", path.display()).into());
    }
    let n = u32::from_le_bytes(buf[0..4].try_into()?) as usize;
    let k = u32::from_le_bytes(buf[4..8].try_into()?) as usize;
    let expected = 8 + n * k * 4;
    if buf.len() < expected {
        return Err(format!(
            "ibin size mismatch: expected {} bytes, got {} ({})",
            expected,
            buf.len(),
            path.display()
        )
        .into());
    }
    let data = buf[8..8 + n * k * 4]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok((data, n, k))
}

#[cfg(feature = "hanns")]
fn compute_recall(result_ids: &[usize], gt_ids: &[i32], gt_k: usize) -> f32 {
    let k = result_ids.len().min(gt_k);
    let gt_set: std::collections::HashSet<i32> =
        gt_ids[..k].iter().copied().collect();
    let hits = result_ids
        .iter()
        .filter(|id| gt_set.contains(&(**id as i32)))
        .count();
    hits as f32 / k as f32
}

#[cfg(feature = "hanns")]
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <data_dir>", args[0]);
        eprintln!("  data_dir must contain: base.fbin, query.fbin, gt.ibin");
        std::process::exit(1);
    }
    let data_dir = std::path::Path::new(&args[1]);

    let base_path = data_dir.join("base.fbin");
    let query_path = data_dir.join("query.fbin");
    let gt_path = data_dir.join("gt.ibin");

    // Load data.
    println!("Loading base vectors from {}...", base_path.display());
    let (base, n, dim) = read_fbin(&base_path)?;
    println!("  {} vectors, dim={}", n, dim);

    println!("Loading query vectors from {}...", query_path.display());
    let (query, nq, qdim) = read_fbin(&query_path)?;
    assert_eq!(qdim, dim, "query dim mismatch: {} vs {}", qdim, dim);
    let num_queries = nq.min(100);
    println!("  {} queries (using {})", nq, num_queries);

    println!("Loading ground truth from {}...", gt_path.display());
    let (gt_data, gt_nq, gt_k) = read_ibin(&gt_path)?;
    println!("  {} queries, k={}", gt_nq, gt_k);

    // ====================================================================
    // Build Hanns HNSW
    // ====================================================================
    println!(
        "Building Hanns HNSW (M={}, ef_construction={})...",
        HNSW_M, EF_CONSTRUCTION
    );
    let mut config =
        hanns::IndexConfig::new(hanns::IndexType::Hnsw, hanns::MetricType::L2, dim);
    config.params.m = Some(HNSW_M);
    config.params.ef_construction = Some(EF_CONSTRUCTION);
    config.params.ef_search = Some(EF_SWEEP[0]);
    config.params.ml = Some(1.0 / (HNSW_M as f32).ln());
    // Disable adaptive ef so configured ef_search is honored exactly.
    config.params.hnsw_adaptive_k = Some(0.0);

    let mut hanns_index = hanns::HnswIndex::new(&config)?;
    let t0 = Instant::now();
    hanns_index.train(&base)?;
    hanns_index.add_parallel(&base, None, None)?;
    let hanns_build_s = t0.elapsed().as_secs_f64();
    println!("  Hanns build: {:.2}s", hanns_build_s);

    // ====================================================================
    // Build Lance HNSW
    // ====================================================================
    println!(
        "Building Lance HNSW (M={}, ef_construction={})...",
        HNSW_M, EF_CONSTRUCTION
    );
    let fsl = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base.clone()),
        dim as i32,
    )
    .unwrap();
    let vectors = Arc::new(FlatFloatStorage::new(fsl, DistanceType::L2));

    let build_params = HnswBuildParams {
        max_level: 7,
        m: HNSW_M,
        ef_construction: EF_CONSTRUCTION,
        prefetch_distance: Some(2),
    };

    let t0 = Instant::now();
    let lance_hnsw = HNSW::index_vectors(vectors.as_ref(), build_params)?;
    let lance_build_s = t0.elapsed().as_secs_f64();
    println!("  Lance build: {:.2}s", lance_build_s);

    // ====================================================================
    // Search sweep
    // ====================================================================
    println!();
    println!(
        "================================================================="
    );
    println!(
        "  Cohere 1M HNSW Benchmark: {} x {}d, K={}, M={}",
        n, dim, TOP_K, HNSW_M
    );
    println!(
        "================================================================="
    );
    println!(
        "  Build: Hanns {:.2}s (parallel), Lance {:.2}s",
        hanns_build_s, lance_build_s
    );
    println!();
    println!(
        "{:>4}  {:>13} {:>11} {:>14} {:>10} {:>7}",
        "ef", "Hanns recall", "Hanns QPS", "Lance recall", "Lance QPS", "Ratio"
    );
    println!(
        "----  ------------  ----------  -------------  ----------  -----"
    );

    let mut search_ratios: Vec<f64> = Vec::new();

    for &ef in &EF_SWEEP {
        // --- Hanns search ---
        hanns_index.set_ef_search(ef);
        let mut hanns_recall_sum = 0.0f32;
        let mut hanns_total_us: u128 = 0;

        for qi in 0..num_queries {
            let q_start = qi * dim;
            let query_slice = &query[q_start..q_start + dim];
            let gt_start = qi * gt_k;

            let req = hanns::SearchRequest {
                top_k: TOP_K,
                nprobe: ef,
                filter: None,
                params: None,
                radius: None,
            };

            let t1 = Instant::now();
            let result = hanns_index.search(query_slice, &req)?;
            hanns_total_us += t1.elapsed().as_micros();

            let ids: Vec<usize> = result
                .ids
                .iter()
                .filter(|&&id| id >= 0)
                .map(|&id| id as usize)
                .collect();
            hanns_recall_sum += compute_recall(&ids, &gt_data[gt_start..gt_start + gt_k], gt_k);
        }

        let hanns_mean_us = hanns_total_us / num_queries as u128;
        let hanns_qps = 1_000_000.0 / hanns_mean_us as f64;
        let hanns_recall = hanns_recall_sum / num_queries as f32;

        // --- Lance search ---
        let mut lance_recall_sum = 0.0f32;
        let mut lance_total_us: u128 = 0;

        for qi in 0..num_queries {
            let q_start = qi * dim;
            let gt_start = qi * gt_k;

            let query_value = Float32Array::from(query[q_start..q_start + dim].to_vec());
            let query_fsl = FixedSizeListArray::try_new_from_values(query_value, dim as i32)
                .unwrap();
            let q = query_fsl.value(0);

            let query_params = HnswQueryParams {
                ef,
                lower_bound: None,
                upper_bound: None,
                dist_q_c: 0.0,
            };

            let t1 = Instant::now();
            let results = lance_hnsw.search_basic(q, TOP_K, &query_params, None, vectors.as_ref())?;
            lance_total_us += t1.elapsed().as_micros();

            let ids: Vec<usize> = results.iter().map(|node| node.id as usize).collect();
            lance_recall_sum += compute_recall(&ids, &gt_data[gt_start..gt_start + gt_k], gt_k);
        }

        let lance_mean_us = lance_total_us / num_queries as u128;
        let lance_qps = 1_000_000.0 / lance_mean_us as f64;
        let lance_recall = lance_recall_sum / num_queries as f32;

        let ratio = lance_qps / hanns_qps;
        search_ratios.push(ratio);

        println!(
            "{:>4}       {:>.4}     {:>7.0}        {:>.4}     {:>7.0}   {:>.3}",
            ef, hanns_recall, hanns_qps, lance_recall, lance_qps, ratio
        );
    }

    // Geometric mean of search ratios.
    let geo_mean = search_ratios
        .iter()
        .fold(1.0f64, |acc, r| acc * r)
        .powf(1.0 / search_ratios.len() as f64);

    println!(
        "================================================================="
    );
    println!(
        "  Search ratio (Lance/Hanns):  {:.2} (geometric mean)",
        geo_mean
    );
    println!();

    Ok(())
}

#[cfg(not(feature = "hanns"))]
fn main() {
    eprintln!("This example requires the 'hanns' feature flag.");
    eprintln!(
        "Run with: cargo run --example hanns_vs_lance_1m --features hanns --release -- <data_dir>"
    );
}
