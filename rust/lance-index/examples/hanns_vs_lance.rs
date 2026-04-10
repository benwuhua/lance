// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark comparing Hanns HNSW vs Lance HNSW performance.
//!
//! Generates random vectors, builds both indices, runs queries, and reports
//! build time, search latency, QPS, and recall against brute-force ground truth.
//!
//! Run with:
//!   cargo run --example hanns_vs_lance --features hanns

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{FixedSizeListArray, Float32Array, types::Float32Type};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::{
    flat::storage::FlatFloatStorage,
    hnsw::builder::{HNSW, HnswBuildParams, HnswQueryParams},
    v3::subindex::IvfSubIndex,
};
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIMENSION: usize = 128;
const NUM_VECTORS: usize = 10_000;
const NUM_QUERIES: usize = 100;
const K: usize = 10;
const SEED_DATA: [u8; 32] = [42; 32];
const SEED_QUERY: u64 = 12345;
const HANNS_M: usize = 16;
const HANNS_EF_CONSTRUCTION: usize = 200;
const SEARCH_EF: usize = 50;

/// Generate a flat f32 vector for Hanns (row-major).
fn generate_flat_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Brute-force top-K search returning (index, squared-L2 distance) pairs sorted by distance.
fn brute_force_topk(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = data.len() / dim;
    let mut distances: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let start = i * dim;
            let mut sum = 0.0f32;
            for j in 0..dim {
                let diff = data[start + j] - query[j];
                sum += diff * diff;
            }
            (i, sum)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

/// Compute recall: fraction of ground-truth top-K IDs found in the approximate results.
fn compute_recall(ground_truth: &[(usize, f32)], approx_ids: &[usize]) -> f32 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let gt_ids: std::collections::HashSet<usize> =
        ground_truth.iter().map(|(id, _)| *id).collect();
    let hits = approx_ids.iter().filter(|id| gt_ids.contains(id)).count();
    hits as f32 / gt_ids.len() as f32
}

struct BenchmarkResult {
    name: &'static str,
    build_ms: u128,
    mean_search_us: u128,
    qps: f64,
    recall: f32,
}

fn format_number(n: f64) -> String {
    if n >= 1_000_000.0 {
        format!("{:.1}M", n / 1_000_000.0)
    } else if n >= 1_000.0 {
        format!("{:.1}K", n / 1_000.0)
    } else {
        format!("{:.1}", n)
    }
}

fn print_results(lance: &BenchmarkResult, hanns: &BenchmarkResult) {
    println!();
    println!(
        "================================================================="
    );
    println!(
        "  HNSW Benchmark: {} vectors x {}d, K={}, ef={}",
        format_number(NUM_VECTORS as f64),
        DIMENSION,
        K,
        SEARCH_EF,
    );
    println!(
        "================================================================="
    );
    println!(
        "{:<20} {:>12} {:>16} {:>10} {:>8}",
        "Index", "Build (ms)", "Mean Query (us)", "QPS", "Recall"
    );
    println!(
        "-----------------------------------------------------------------"
    );

    for result in [lance, hanns] {
        println!(
            "{:<20} {:>12} {:>16} {:>10} {:>8.4}",
            result.name,
            result.build_ms,
            result.mean_search_us,
            format_number(result.qps),
            result.recall,
        );
    }
    println!(
        "================================================================="
    );

    let build_ratio = lance.build_ms as f64 / hanns.build_ms.max(1) as f64;
    let search_ratio = lance.mean_search_us as f64 / hanns.mean_search_us.max(1) as f64;
    println!(
        "  Build ratio (Lance/Hanns):   {:.2}x",
        build_ratio,
    );
    println!(
        "  Search ratio (Lance/Hanns):  {:.2}x",
        search_ratio,
    );
    println!();
}

#[cfg(feature = "hanns")]
fn main() {
    // --- Generate data vectors ---
    let data = generate_random_array_with_seed::<Float32Type>(
        NUM_VECTORS * DIMENSION,
        SEED_DATA,
    );
    let fsl = FixedSizeListArray::try_new_from_values(data.clone(), DIMENSION as i32).unwrap();

    // Flat f32 for Hanns and brute-force.
    let data_flat: Vec<f32> = data.values().to_vec();

    // --- Generate query vectors (different seed) ---
    let query_flat = generate_flat_vectors(NUM_QUERIES, DIMENSION, SEED_QUERY);
    let query_fsl_values = Float32Array::from(query_flat.clone());
    let query_fsl =
        FixedSizeListArray::try_new_from_values(query_fsl_values, DIMENSION as i32).unwrap();

    // --- Compute brute-force ground truth for all queries ---
    let ground_truth: Vec<Vec<(usize, f32)>> = (0..NUM_QUERIES)
        .map(|i| {
            let q_start = i * DIMENSION;
            brute_force_topk(&data_flat, DIMENSION, &query_flat[q_start..q_start + DIMENSION], K)
        })
        .collect();

    // ====================================================================
    // Lance HNSW
    // ====================================================================
    println!(
        "Building Lance HNSW (M={}, ef_construction={})...",
        HANNS_M, HANNS_EF_CONSTRUCTION
    );
    let vectors = Arc::new(FlatFloatStorage::new(fsl.clone(), DistanceType::L2));

    let build_params = HnswBuildParams {
        max_level: 7,
        m: HANNS_M,
        ef_construction: HANNS_EF_CONSTRUCTION,
        prefetch_distance: Some(2),
    };

    let t0 = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let lance_hnsw = pool
        .install(|| HNSW::index_vectors(vectors.as_ref(), build_params))
        .unwrap();
    let lance_build_ms = t0.elapsed().as_millis();
    println!("  Lance build: {}ms", lance_build_ms);

    // Search Lance HNSW
    let lance_query_params = HnswQueryParams {
        ef: SEARCH_EF,
        lower_bound: None,
        upper_bound: None,
        dist_q_c: 0.0,
    };

    let mut lance_search_times_us: Vec<u128> = Vec::with_capacity(NUM_QUERIES);
    let mut lance_recall_sum = 0.0f32;

    for i in 0..NUM_QUERIES {
        let query = query_fsl.value(i);
        let t1 = Instant::now();
        let results = lance_hnsw
            .search_basic(query, K, &lance_query_params, None, vectors.as_ref())
            .unwrap();
        lance_search_times_us.push(t1.elapsed().as_micros());

        let ids: Vec<usize> = results.iter().map(|node| node.id as usize).collect();
        lance_recall_sum += compute_recall(&ground_truth[i], &ids);
    }

    let lance_mean_us =
        lance_search_times_us.iter().sum::<u128>() / lance_search_times_us.len() as u128;
    let lance_qps = 1_000_000.0 / lance_mean_us as f64;
    let lance_recall = lance_recall_sum / NUM_QUERIES as f32;

    // ====================================================================
    // Hanns HNSW
    // ====================================================================
    println!(
        "Building Hanns HNSW (M={}, ef_construction={})...",
        HANNS_M, HANNS_EF_CONSTRUCTION
    );

    let mut config =
        hanns::IndexConfig::new(hanns::IndexType::Hnsw, hanns::MetricType::L2, DIMENSION);
    config.params.m = Some(HANNS_M);
    config.params.ef_construction = Some(HANNS_EF_CONSTRUCTION);
    // Use the Hanns default ef_search for building, then override per-query via nprobe.
    config.params.ef_search = Some(SEARCH_EF);
    // Disable adaptive ef so the configured ef_search is honored exactly.
    config.params.hnsw_adaptive_k = Some(0.0);

    let mut hanns_index = hanns::HnswIndex::new(&config)
        .expect("Hanns HnswIndex::new failed");

    let t0 = Instant::now();
    hanns_index
        .train(&data_flat)
        .expect("Hanns HnswIndex::train failed");
    hanns_index
        .add(&data_flat, None)
        .expect("Hanns HnswIndex::add failed");
    let hanns_build_ms = t0.elapsed().as_millis();
    println!("  Hanns build: {}ms", hanns_build_ms);

    // Search Hanns HNSW
    let mut hanns_search_times_us: Vec<u128> = Vec::with_capacity(NUM_QUERIES);
    let mut hanns_recall_sum = 0.0f32;

    for i in 0..NUM_QUERIES {
        let q_start = i * DIMENSION;
        let query = &query_flat[q_start..q_start + DIMENSION];

        let req = hanns::SearchRequest {
            top_k: K,
            nprobe: SEARCH_EF, // nprobe is reused as ef_search override for HNSW.
            filter: None,
            params: None,
            radius: None,
        };

        let t1 = Instant::now();
        let result = hanns_index
            .search(query, &req)
            .expect("Hanns HnswIndex::search failed");
        hanns_search_times_us.push(t1.elapsed().as_micros());

        let ids: Vec<usize> = result
            .ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as usize)
            .collect();
        hanns_recall_sum += compute_recall(&ground_truth[i], &ids);
    }

    let hanns_mean_us =
        hanns_search_times_us.iter().sum::<u128>() / hanns_search_times_us.len() as u128;
    let hanns_qps = 1_000_000.0 / hanns_mean_us as f64;
    let hanns_recall = hanns_recall_sum / NUM_QUERIES as f32;

    // ====================================================================
    // Report
    // ====================================================================
    let lance_result = BenchmarkResult {
        name: "Lance HNSW",
        build_ms: lance_build_ms,
        mean_search_us: lance_mean_us,
        qps: lance_qps,
        recall: lance_recall,
    };
    let hanns_result = BenchmarkResult {
        name: "Hanns HNSW",
        build_ms: hanns_build_ms,
        mean_search_us: hanns_mean_us,
        qps: hanns_qps,
        recall: hanns_recall,
    };

    print_results(&lance_result, &hanns_result);
}

#[cfg(not(feature = "hanns"))]
fn main() {
    eprintln!("This example requires the 'hanns' feature flag.");
    eprintln!("Run with: cargo run --example hanns_vs_lance --features hanns");
}
