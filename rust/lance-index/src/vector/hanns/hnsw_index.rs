// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Hanns HNSW index as an IvfSubIndex for Lance IVF.

use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use arrow::array::AsArray;
use arrow_array::{ArrayRef, BinaryArray, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use deepsize::{Context, DeepSizeOf};
use lance_core::{Error, ROW_ID_FIELD, Result};
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use crate::metrics::MetricsCollector;
use crate::prefilter::PreFilter;
use crate::vector::storage::VectorStore;
use crate::vector::v3::subindex::IvfSubIndex;
use crate::vector::{DIST_COL, Query};

const HANNS_HNSW_TYPE: &str = "HANNS_HNSW";
const HANNS_HNSW_METADATA_KEY: &str = "lance:hanns_hnsw";
const GRAPH_DATA_COLUMN: &str = "graph_data";

static ANN_SEARCH_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ])
    .into()
});

/// Build parameters for Hanns HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct HannsHnswBuildParams {
    /// Number of connections per node (HNSW M parameter).
    pub m: usize,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Level multiplier (ml). Defaults to 1/ln(M) when None.
    pub ml: Option<f32>,
}

impl Default for HannsHnswBuildParams {
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 400,
            ml: None,
        }
    }
}

impl HannsHnswBuildParams {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m,
            ef_construction,
            ml: None,
        }
    }
}

/// Query parameters for Hanns HNSW search.
#[derive(Debug, Clone, Copy, DeepSizeOf)]
pub struct HannsHnswQueryParams {
    /// HNSW ef_search parameter controlling search beam width.
    pub ef: usize,
}

impl From<&Query> for HannsHnswQueryParams {
    fn from(query: &Query) -> Self {
        // Use ef from query if set, otherwise default to a reasonable value based on k.
        let ef = query.ef.unwrap_or_else(|| (query.k * 2).max(400));
        Self { ef }
    }
}

/// Internal metadata stored alongside the serialized graph bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HannsMetadata {
    /// Vector dimensionality.
    dim: usize,
    /// The metric type used for distance computation.
    metric_type: String,
    /// Original build parameters (for rebuilding during remap).
    build_params: HannsHnswBuildParams,
    /// Mapping from Hanns internal node indices to Lance row IDs.
    /// Stored as pairs to keep serialization compact.
    node_to_rowid: Vec<(u32, u64)>,
}

struct HannsRuntimeCache {
    index: OnceLock<hanns::HnswIndex>,
    init_lock: Mutex<()>,
    initialized_size_estimate: usize,
}

impl HannsRuntimeCache {
    fn new(serialized_size: usize) -> Self {
        Self {
            index: OnceLock::new(),
            init_lock: Mutex::new(()),
            initialized_size_estimate: serialized_size,
        }
    }

    fn from_index(index: hanns::HnswIndex, serialized_size: usize) -> Self {
        Self {
            index: OnceLock::from(index),
            init_lock: Mutex::new(()),
            initialized_size_estimate: serialized_size,
        }
    }

    fn get(&self) -> Option<&hanns::HnswIndex> {
        self.index.get()
    }

    fn get_or_init_with<F>(&self, init: F) -> Result<&hanns::HnswIndex>
    where
        F: FnOnce() -> Result<hanns::HnswIndex>,
    {
        if let Some(index) = self.index.get() {
            return Ok(index);
        }

        let _guard = self
            .init_lock
            .lock()
            .map_err(|_| Error::index("Hanns HNSW runtime cache init lock was poisoned"))?;
        if let Some(index) = self.index.get() {
            return Ok(index);
        }

        let index = init()?;
        self.index
            .set(index)
            .map_err(|_| Error::index("Hanns HNSW runtime cache was initialized unexpectedly"))?;
        self.index.get().ok_or_else(|| {
            Error::index("Hanns HnswIndex runtime cache was not initialized after deserialization")
        })
    }
}

impl DeepSizeOf for HannsRuntimeCache {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        if self.index.get().is_some() {
            // Hanns does not expose deep-size accounting for its runtime graph,
            // so use the persisted graph bytes as a conservative retained-size estimate.
            self.initialized_size_estimate
        } else {
            0
        }
    }
}

/// Hanns HNSW index wrapper implementing IvfSubIndex.
///
/// This wraps the Hanns HNSW implementation as a sub-index for Lance IVF.
/// The index is serialized as opaque bytes and stored in a single BinaryArray column.
#[derive(Clone)]
pub struct HannsHnswIndex {
    /// Serialized graph bytes (kept for deep_size_of and remap).
    serialized: Vec<u8>,
    /// Runtime Hanns graph cache shared by clones.
    runtime_cache: Arc<HannsRuntimeCache>,
    /// Vector dimension.
    dim: usize,
    /// Distance type used for this index.
    metric_type: DistanceType,
    /// Build parameters (preserved for remap).
    build_params: HannsHnswBuildParams,
    /// Mapping from Hanns node index to Lance row ID.
    node_to_rowid: Vec<u64>,
}

impl DeepSizeOf for HannsHnswIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.serialized.deep_size_of_children(context)
            + self.runtime_cache.deep_size_of_children(context)
            + self.dim.deep_size_of_children(context)
            + self.node_to_rowid.deep_size_of_children(context)
    }
}

impl Debug for HannsHnswIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HannsHnswIndex(dim={}, num_vectors={}, metric={:?})",
            self.dim,
            self.node_to_rowid.len(),
            self.metric_type,
        )
    }
}

/// Convert Lance DistanceType to Hanns MetricType.
fn lance_to_hanns_metric(dt: DistanceType) -> hanns::MetricType {
    match dt {
        DistanceType::L2 => hanns::MetricType::L2,
        DistanceType::Cosine => hanns::MetricType::Cosine,
        DistanceType::Dot => hanns::MetricType::Ip,
        _ => hanns::MetricType::L2,
    }
}

/// Extract raw f32 vectors from a FlatFloatStorage-backed VectorStore.
///
/// Returns a flat &[f32] in row-major layout (dim * num_vectors elements).
fn extract_flat_vectors(storage: &impl VectorStore) -> Result<(Vec<f32>, usize)> {
    let batches: Vec<RecordBatch> = storage.to_batches()?.collect();
    if batches.is_empty() {
        return Ok((Vec::new(), 0));
    }

    // Look for the "flat" column (FlatQuantizer storage column name).
    // The VectorStore batches should contain a FixedSizeList<f32> column named "flat".
    let mut all_vectors = Vec::new();
    let mut dim = 0usize;

    for batch in &batches {
        let vectors_col = batch
            .column_by_name("flat")
            .ok_or_else(|| Error::index("flat column not found in VectorStore batch; Hanns HNSW requires FlatQuantizer storage"))?;
        let fsl = vectors_col.as_fixed_size_list();
        if dim == 0 {
            dim = fsl.value_length() as usize;
        }
        let flat_values = fsl.values().as_primitive::<arrow::datatypes::Float32Type>();
        all_vectors.extend_from_slice(flat_values.values());
    }

    Ok((all_vectors, dim))
}

impl HannsHnswIndex {
    #[cfg(test)]
    fn is_runtime_cache_initialized(&self) -> bool {
        self.runtime_cache.get().is_some()
    }

    #[cfg(test)]
    fn runtime_cache_ptr(&self) -> Option<*const hanns::HnswIndex> {
        self.runtime_cache
            .get()
            .map(|index| index as *const hanns::HnswIndex)
    }

    /// Build a Hanns HNSW index from raw f32 vectors.
    fn build_from_vectors(
        vectors: &[f32],
        dim: usize,
        metric_type: DistanceType,
        params: &HannsHnswBuildParams,
    ) -> Result<Self> {
        let hanns_metric = lance_to_hanns_metric(metric_type);

        let mut config = hanns::IndexConfig::new(hanns::IndexType::Hnsw, hanns_metric, dim);
        config.params.m = Some(params.m);
        config.params.ef_construction = Some(params.ef_construction);
        config.params.ml = params.ml;

        let mut index = hanns::HnswIndex::new(&config)
            .map_err(|e| Error::index(format!("Hanns HnswIndex::new failed: {}", e)))?;

        // Train the index (required before add_parallel).
        index
            .train(vectors)
            .map_err(|e| Error::index(format!("Hanns HnswIndex::train failed: {}", e)))?;

        // Build with sequential IDs (0..n), so Hanns node index == ID.
        index
            .add_parallel(vectors, None, None)
            .map_err(|e| Error::index(format!("Hanns HnswIndex::add_parallel failed: {}", e)))?;

        let serialized = index.serialize_to_bytes().map_err(|e| {
            Error::index(format!("Hanns HnswIndex::serialize_to_bytes failed: {}", e))
        })?;
        let serialized_size = serialized.len();

        Ok(Self {
            serialized,
            runtime_cache: Arc::new(HannsRuntimeCache::from_index(index, serialized_size)),
            dim,
            metric_type,
            build_params: params.clone(),
            node_to_rowid: Vec::new(), // will be set by caller
        })
    }

    fn runtime_index(&self) -> Result<&hanns::HnswIndex> {
        if let Some(index) = self.runtime_cache.get() {
            return Ok(index);
        }

        self.runtime_cache.get_or_init_with(|| {
            hanns::HnswIndex::deserialize_from_bytes(&self.serialized)
                .map_err(|e| Error::index(format!("Hanns HnswIndex::deserialize failed: {}", e)))
        })
    }

    /// Search using the Hanns index, returning up to k results.
    fn search_hanns(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(usize, f32)>> {
        let index = self.runtime_index()?;

        let req = hanns::SearchRequest {
            top_k: k,
            nprobe: ef, // nprobe is reused as ef_search for HNSW.
            filter: None,
            params: None,
            radius: None,
        };

        let result = index
            .search(query, &req)
            .map_err(|e| Error::index(format!("Hanns HnswIndex::search failed: {}", e)))?;

        let mut results = Vec::with_capacity(k);
        for i in 0..result.ids.len() {
            let id = result.ids[i];
            if id < 0 {
                continue;
            }
            let dist = result.distances[i];
            results.push((id as usize, dist));
        }

        Ok(results)
    }

    /// Load the index from serialized bytes and metadata.
    fn load_from_parts(serialized: Vec<u8>, metadata: &HannsMetadata) -> Result<Self> {
        let serialized_size = serialized.len();
        let metric_type = match metadata.metric_type.as_str() {
            "l2" => DistanceType::L2,
            "cosine" => DistanceType::Cosine,
            "dot" => DistanceType::Dot,
            other => {
                return Err(Error::index(format!(
                    "unknown metric type in Hanns metadata: {}",
                    other
                )));
            }
        };

        let node_to_rowid = {
            let mut mapping = vec![0u64; metadata.node_to_rowid.len()];
            for (node_idx, rowid) in &metadata.node_to_rowid {
                if *node_idx as usize >= mapping.len() {
                    return Err(Error::index(format!(
                        "node_to_rowid mapping out of bounds: node_idx={} but len={}",
                        node_idx,
                        mapping.len()
                    )));
                }
                mapping[*node_idx as usize] = *rowid;
            }
            mapping
        };

        Ok(Self {
            serialized,
            runtime_cache: Arc::new(HannsRuntimeCache::new(serialized_size)),
            dim: metadata.dim,
            metric_type,
            build_params: metadata.build_params.clone(),
            node_to_rowid,
        })
    }
}

impl IvfSubIndex for HannsHnswIndex {
    type QueryParams = HannsHnswQueryParams;
    type BuildParams = HannsHnswBuildParams;

    fn load(data: RecordBatch) -> Result<Self>
    where
        Self: Sized,
    {
        if data.num_rows() == 0 {
            return Ok(Self {
                serialized: Vec::new(),
                runtime_cache: Arc::new(HannsRuntimeCache::new(0)),
                dim: 0,
                metric_type: DistanceType::L2,
                build_params: HannsHnswBuildParams::default(),
                node_to_rowid: Vec::new(),
            });
        }

        // Read metadata from schema.
        let metadata_json = data
            .schema_ref()
            .metadata()
            .get(HANNS_HNSW_METADATA_KEY)
            .ok_or_else(|| {
                Error::index(format!(
                    "{} not found in schema metadata",
                    HANNS_HNSW_METADATA_KEY
                ))
            })?;
        let metadata: HannsMetadata = serde_json::from_str(metadata_json).map_err(|e| {
            Error::index(format!(
                "Failed to parse Hanns metadata: {}, json: {}",
                e, metadata_json
            ))
        })?;

        // Read serialized graph bytes from the first row of the BinaryArray column.
        let graph_col = data.column_by_name(GRAPH_DATA_COLUMN).ok_or_else(|| {
            Error::index(format!("column {} not found in batch", GRAPH_DATA_COLUMN))
        })?;
        let binary_array = graph_col.as_binary::<i32>();
        let serialized = binary_array.value(0).to_vec();

        Self::load_from_parts(serialized, &metadata)
    }

    fn name() -> &'static str {
        HANNS_HNSW_TYPE
    }

    fn metadata_key() -> &'static str {
        HANNS_HNSW_METADATA_KEY
    }

    fn schema() -> arrow_schema::SchemaRef {
        Schema::new(vec![Field::new(GRAPH_DATA_COLUMN, DataType::Binary, false)]).into()
    }

    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        _storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let schema = ANN_SEARCH_SCHEMA.clone();

        if self.node_to_rowid.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }

        // Convert query to flat f32 slice.
        let query_arr = query.as_primitive::<arrow::datatypes::Float32Type>();
        let query_slice = query_arr.values();

        // Determine effective ef_search.
        let ef = params.ef.max(k);

        let results = self.search_hanns(query_slice, k, ef)?;

        // Apply prefilter: map Hanns node IDs to Lance row IDs and filter.
        let mut row_ids = Vec::with_capacity(results.len());
        let mut distances = Vec::with_capacity(results.len());

        if prefilter.is_empty() {
            for (node_idx, dist) in &results {
                let rowid = self
                    .node_to_rowid
                    .get(*node_idx)
                    .copied()
                    .unwrap_or(*node_idx as u64);
                row_ids.push(rowid);
                distances.push(*dist);
            }
        } else {
            let mask = prefilter.mask();
            for (node_idx, dist) in &results {
                let rowid = self
                    .node_to_rowid
                    .get(*node_idx)
                    .copied()
                    .unwrap_or(*node_idx as u64);
                if mask.selected(rowid) {
                    row_ids.push(rowid);
                    distances.push(*dist);
                }
            }
        }

        metrics.record_comparisons(self.node_to_rowid.len());

        let row_ids = Arc::new(UInt64Array::from(row_ids));
        let distances = Arc::new(Float32Array::from(distances));

        Ok(RecordBatch::try_new(schema, vec![distances, row_ids])?)
    }

    fn index_vectors(storage: &impl VectorStore, params: Self::BuildParams) -> Result<Self>
    where
        Self: Sized,
    {
        if storage.is_empty() {
            return Ok(Self {
                serialized: Vec::new(),
                runtime_cache: Arc::new(HannsRuntimeCache::new(0)),
                dim: 0,
                metric_type: storage.distance_type(),
                build_params: params,
                node_to_rowid: Vec::new(),
            });
        }

        let (vectors, dim) = extract_flat_vectors(storage)?;
        let num_vectors = storage.len();

        log::debug!(
            "Building Hanns HNSW: num={}, dim={}, m={}, ef_construction={}, metric={:?}",
            num_vectors,
            dim,
            params.m,
            params.ef_construction,
            storage.distance_type(),
        );

        let metric_type = storage.distance_type();
        let mut index = Self::build_from_vectors(&vectors, dim, metric_type, &params)?;

        // Collect row ID mapping: node i -> storage.row_id(i).
        index.node_to_rowid = (0..num_vectors).map(|i| storage.row_id(i as u32)).collect();

        Ok(index)
    }

    fn remap(&self, _mapping: &HashMap<u64, Option<u64>>, store: &impl VectorStore) -> Result<Self>
    where
        Self: Sized,
    {
        // Rebuild the index from the remapped storage, since Hanns HNSW
        // graph structure depends on vector positions.
        Self::index_vectors(store, self.build_params.clone())
    }

    fn to_batch(&self) -> Result<RecordBatch> {
        // Build metadata.
        let node_to_rowid: Vec<(u32, u64)> = self
            .node_to_rowid
            .iter()
            .enumerate()
            .map(|(i, &rowid)| (i as u32, rowid))
            .collect();

        let metric_str = match self.metric_type {
            DistanceType::L2 => "l2",
            DistanceType::Cosine => "cosine",
            DistanceType::Dot => "dot",
            other => {
                return Err(Error::index(format!(
                    "unsupported metric type for Hanns HNSW: {:?}",
                    other
                )));
            }
        };

        let metadata = HannsMetadata {
            dim: self.dim,
            metric_type: metric_str.to_string(),
            build_params: self.build_params.clone(),
            node_to_rowid,
        };

        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| Error::index(format!("Failed to serialize Hanns metadata: {}", e)))?;

        let schema = Self::schema()
            .as_ref()
            .clone()
            .with_metadata(HashMap::from_iter(vec![(
                HANNS_HNSW_METADATA_KEY.to_string(),
                metadata_json,
            )]));

        // Store serialized bytes as a single-row BinaryArray.
        let binary_array = BinaryArray::from(vec![self.serialized.as_slice()]);

        Ok(RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(binary_array)],
        )?)
    }
}

#[cfg(test)]
#[cfg(feature = "hanns")]
mod tests {
    use super::*;
    use lance_linalg::distance::DistanceType;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    /// Build a HannsHnswIndex from raw vectors using the private method
    /// via the IvfSubIndex::index_vectors path would require a VectorStore,
    /// so we call build_from_vectors directly by constructing via the internal API.
    fn build_test_index(
        vectors: &[f32],
        dim: usize,
        metric_type: DistanceType,
        params: &HannsHnswBuildParams,
        num_vectors: usize,
    ) -> HannsHnswIndex {
        let mut index =
            HannsHnswIndex::build_from_vectors(vectors, dim, metric_type, params).unwrap();
        // Set identity mapping: node i -> row id i.
        index.node_to_rowid = (0..num_vectors).map(|i| i as u64).collect();
        index
    }

    #[test]
    fn test_hanns_build_and_search() {
        let dim = 128;
        let num_vectors = 1000;
        let k = 10;
        let vectors = random_vectors(num_vectors, dim, 42);

        let params = HannsHnswBuildParams::new(16, 200);
        let index = build_test_index(&vectors, dim, DistanceType::L2, &params, num_vectors);

        // Use the first vector as query.
        let query = &vectors[0..dim];
        let results = index.search_hanns(query, k, 400).unwrap();

        assert!(!results.is_empty(), "search should return results");
        assert!(
            results.len() <= k,
            "results should not exceed k={}; got {}",
            k,
            results.len()
        );

        // Results should be sorted by ascending distance.
        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "results should be sorted by distance: {} <= {}",
                window[0].1,
                window[1].1
            );
        }

        // The first result should have the smallest distance.
        let min_dist = results
            .iter()
            .map(|(_, d)| *d)
            .fold(f32::INFINITY, f32::min);
        assert!(
            (results[0].1 - min_dist).abs() < 1e-6,
            "first result should have the smallest distance: first={} min={}",
            results[0].1,
            min_dist
        );

        // All returned IDs should be valid (within the index range).
        for (id, _) in &results {
            assert!(
                *id < num_vectors,
                "result ID {} should be < num_vectors {}",
                id,
                num_vectors
            );
        }

        // All distances should be non-negative for L2.
        for (_, dist) in &results {
            assert!(
                *dist >= 0.0,
                "L2 distance should be non-negative, got {}",
                dist
            );
        }
    }

    #[test]
    fn test_hanns_serialize_roundtrip() {
        let dim = 32;
        let num_vectors = 100;
        let vectors = random_vectors(num_vectors, dim, 123);

        let params = HannsHnswBuildParams::new(16, 200);
        let original = build_test_index(&vectors, dim, DistanceType::L2, &params, num_vectors);

        // Serialize.
        let batch = original.to_batch().unwrap();
        assert_eq!(batch.num_rows(), 1, "to_batch should produce 1 row");
        assert!(
            batch.column_by_name(GRAPH_DATA_COLUMN).is_some(),
            "batch should contain graph_data column"
        );
        assert!(
            batch
                .schema_ref()
                .metadata()
                .contains_key(HANNS_HNSW_METADATA_KEY),
            "batch schema should contain Hanns metadata key"
        );

        // Deserialize.
        let loaded = HannsHnswIndex::load(batch).unwrap();

        // Search with both and compare results.
        let query = &vectors[0..dim];
        let original_results = original.search_hanns(query, 5, 400).unwrap();
        let loaded_results = loaded.search_hanns(query, 5, 400).unwrap();

        assert_eq!(
            original_results.len(),
            loaded_results.len(),
            "loaded index should return same number of results"
        );
        for (orig, loaded) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(orig.0, loaded.0, "node IDs should match after roundtrip");
            assert!(
                (orig.1 - loaded.1).abs() < 1e-6,
                "distances should match after roundtrip: {} vs {}",
                orig.1,
                loaded.1
            );
        }
    }

    #[test]
    fn test_hanns_runtime_cache() {
        let dim = 32;
        let num_vectors = 100;
        let vectors = random_vectors(num_vectors, dim, 456);

        let params = HannsHnswBuildParams::new(16, 200);
        let original = build_test_index(&vectors, dim, DistanceType::L2, &params, num_vectors);
        assert!(
            original.is_runtime_cache_initialized(),
            "newly built index should seed the runtime cache"
        );
        let loaded = HannsHnswIndex::load(original.to_batch().unwrap()).unwrap();
        let query = &vectors[0..dim];

        assert!(
            !loaded.is_runtime_cache_initialized(),
            "loaded index should start with an empty runtime cache"
        );

        let first_results = loaded.search_hanns(query, 5, 400).unwrap();
        assert!(
            loaded.is_runtime_cache_initialized(),
            "loaded index should initialize its runtime cache after first search"
        );
        let first_cache_ptr = loaded
            .runtime_cache_ptr()
            .expect("runtime cache should be initialized after first search");

        let second_results = loaded.search_hanns(query, 5, 400).unwrap();
        let second_cache_ptr = loaded
            .runtime_cache_ptr()
            .expect("runtime cache should remain initialized after second search");

        assert_eq!(
            first_cache_ptr, second_cache_ptr,
            "second search should reuse the cached Hanns runtime"
        );
        assert_eq!(
            first_results, second_results,
            "caching should not change Hanns search results"
        );
    }

    #[test]
    fn test_hanns_runtime_cache_shared_by_clone() {
        let dim = 32;
        let num_vectors = 100;
        let vectors = random_vectors(num_vectors, dim, 789);

        let params = HannsHnswBuildParams::new(16, 200);
        let original = build_test_index(&vectors, dim, DistanceType::L2, &params, num_vectors);
        let loaded = HannsHnswIndex::load(original.to_batch().unwrap()).unwrap();
        let cloned = loaded.clone();
        let query = &vectors[0..dim];

        assert!(
            !loaded.is_runtime_cache_initialized(),
            "loaded index should start with an empty runtime cache"
        );
        assert!(
            !cloned.is_runtime_cache_initialized(),
            "clone should observe the same empty runtime cache before search"
        );

        cloned.search_hanns(query, 5, 400).unwrap();

        assert!(
            loaded.is_runtime_cache_initialized(),
            "original loaded index should observe cache initialized by clone"
        );
        assert!(
            cloned.is_runtime_cache_initialized(),
            "searched clone should observe initialized cache"
        );
        assert_eq!(
            loaded.runtime_cache_ptr(),
            cloned.runtime_cache_ptr(),
            "clone and original should share the same cached Hanns runtime"
        );
    }

    #[test]
    fn test_hanns_runtime_cache_shared_by_concurrent_first_searches() {
        let dim = 32;
        let num_vectors = 100;
        let vectors = random_vectors(num_vectors, dim, 131415);

        let params = HannsHnswBuildParams::new(16, 200);
        let original = build_test_index(&vectors, dim, DistanceType::L2, &params, num_vectors);
        let loaded = Arc::new(HannsHnswIndex::load(original.to_batch().unwrap()).unwrap());
        let query = Arc::new(vectors[0..dim].to_vec());
        let thread_count = 8;
        let barrier = Arc::new(std::sync::Barrier::new(thread_count));

        assert!(
            !loaded.is_runtime_cache_initialized(),
            "loaded index should start with an empty runtime cache"
        );

        let mut handles = Vec::with_capacity(thread_count);
        for _ in 0..thread_count {
            let loaded = Arc::clone(&loaded);
            let query = Arc::clone(&query);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let results = loaded.search_hanns(&query, 5, 400).unwrap();
                assert!(
                    !results.is_empty(),
                    "concurrent first search should return results"
                );
                loaded
                    .runtime_cache_ptr()
                    .expect("runtime cache should be initialized after search")
                    as usize
            }));
        }

        let mut cache_ptrs = Vec::with_capacity(thread_count);
        for handle in handles {
            cache_ptrs.push(handle.join().expect("search thread should not panic"));
        }

        assert!(
            loaded.is_runtime_cache_initialized(),
            "concurrent first searches should initialize runtime cache"
        );
        for ptr in &cache_ptrs[1..] {
            assert_eq!(
                cache_ptrs[0], *ptr,
                "all concurrent searches should observe the same cached Hanns runtime"
            );
        }
    }

    #[test]
    fn test_hanns_runtime_cache_deep_size() {
        let dim = 32;
        let num_vectors = 100;
        let vectors = random_vectors(num_vectors, dim, 101112);

        let params = HannsHnswBuildParams::new(16, 200);
        let original = build_test_index(&vectors, dim, DistanceType::L2, &params, num_vectors);
        let loaded = HannsHnswIndex::load(original.to_batch().unwrap()).unwrap();
        let query = &vectors[0..dim];

        assert!(
            !loaded.is_runtime_cache_initialized(),
            "loaded index should start with an empty runtime cache"
        );

        let before_search_size = loaded.deep_size_of();
        loaded.search_hanns(query, 5, 400).unwrap();
        let after_search_size = loaded.deep_size_of();

        assert!(
            after_search_size >= before_search_size + loaded.serialized.len(),
            "deep size should increase by at least serialized graph bytes after runtime cache initializes: before={}, after={}, serialized={}",
            before_search_size,
            after_search_size,
            loaded.serialized.len()
        );
    }

    #[test]
    fn test_hanns_build_params_default() {
        let params = HannsHnswBuildParams::default();
        assert_eq!(params.m, 32, "default m should be 32");
        assert_eq!(
            params.ef_construction, 400,
            "default ef_construction should be 400"
        );
        assert!(
            params.ml.is_none(),
            "default ml should be None (auto 1/ln(M))"
        );
    }

    #[test]
    fn test_hanns_query_params_from_query() {
        // With ef explicitly set.
        let query_with_ef = Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0; 4])),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: Some(200),
            refine_factor: None,
            metric_type: None,
            use_index: true,
            dist_q_c: 0.0,
        };
        let params = HannsHnswQueryParams::from(&query_with_ef);
        assert_eq!(params.ef, 200, "ef should come from query when set");

        // With ef unset — should fall back to max(k*2, 400).
        let query_without_ef = Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0; 4])),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: None,
            use_index: true,
            dist_q_c: 0.0,
        };
        let params = HannsHnswQueryParams::from(&query_without_ef);
        // k=10, so fallback = max(10*2, 400) = 400.
        assert_eq!(
            params.ef, 400,
            "ef should fall back to max(k*2, 400) when unset"
        );

        // With large k — fallback = max(k*2, 400) = k*2.
        let query_large_k = Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0; 4])),
            k: 500,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: None,
            use_index: true,
            dist_q_c: 0.0,
        };
        let params = HannsHnswQueryParams::from(&query_large_k);
        assert_eq!(
            params.ef, 1000,
            "ef should be k*2=1000 when k is large enough"
        );
    }
}
