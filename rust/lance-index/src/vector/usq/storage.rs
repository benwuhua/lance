// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, OnceLock};

use arrow::array::AsArray;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array, UInt8Array};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result, ROW_ID};
use lance_file::previous::reader::FileReader as PreviousFileReader;
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::usq::{USQ_CODE_COLUMN, USQ_META_COLUMN, USQ_SIGN_COLUMN, USQ_METADATA_KEY};

#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct UsqQuantizationMetadata {
    pub dim: u32,
    pub num_bits: u8,
    pub rotation_seed: u64,
}

#[async_trait]
impl QuantizerMetadata for UsqQuantizationMetadata {
    fn buffer_index(&self) -> Option<u32> {
        None
    }

    fn parse_buffer(&mut self, _bytes: bytes::Bytes) -> Result<()> {
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<bytes::Bytes>> {
        Ok(None)
    }

    async fn load(reader: &PreviousFileReader) -> Result<Self> {
        let metadata_str = reader
            .schema()
            .metadata
            .get(USQ_METADATA_KEY)
            .ok_or_else(|| {
                Error::index(format!(
                    "Reading USQ metadata: key {} not found",
                    USQ_METADATA_KEY
                ))
            })?;
        serde_json::from_str(metadata_str)
            .map_err(|e| Error::index(format!("Failed to parse USQ metadata: {}", e)))
    }
}

#[derive(Debug)]
pub struct USQStorage {
    metadata: UsqQuantizationMetadata,
    batch: RecordBatch,
    distance_type: DistanceType,

    // Helper fields extracted from batch
    row_ids: UInt64Array,
    codes: FixedSizeListArray,
    signs: FixedSizeListArray,
    meta: FixedSizeListArray,

    /// Cached quantizer — QR decomposition (O(d³)) runs once per partition load,
    /// not once per query call to dist_calculator().
    cached_quantizer: OnceLock<hanns::quantization::usq::UsqQuantizer>,
}

impl Clone for USQStorage {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            batch: self.batch.clone(),
            distance_type: self.distance_type,
            row_ids: self.row_ids.clone(),
            codes: self.codes.clone(),
            signs: self.signs.clone(),
            meta: self.meta.clone(),
            // Clone starts with empty lock; quantizer will be re-initialized on first use.
            cached_quantizer: OnceLock::new(),
        }
    }
}

impl DeepSizeOf for USQStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.batch.get_array_memory_size()
    }
}

#[async_trait]
impl QuantizerStorage for USQStorage {
    type Metadata = UsqQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        _frag_reuse_index: Option<Arc<crate::frag_reuse::FragReuseIndex>>,
    ) -> Result<Self> {
        let row_ids = batch[ROW_ID]
            .as_primitive::<arrow::datatypes::UInt64Type>()
            .clone();
        let codes = batch[USQ_CODE_COLUMN].as_fixed_size_list().clone();
        let signs = batch[USQ_SIGN_COLUMN].as_fixed_size_list().clone();
        let meta = batch[USQ_META_COLUMN].as_fixed_size_list().clone();

        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            codes,
            signs,
            meta,
            cached_quantizer: OnceLock::new(),
        })
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        reader: &PreviousFileReader,
        range: std::ops::Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
        frag_reuse_index: Option<Arc<crate::frag_reuse::FragReuseIndex>>,
    ) -> Result<Self> {
        let schema = reader.schema();
        let batch = reader.read_range(range, schema).await?;
        Self::try_from_batch(batch, metadata, distance_type, frag_reuse_index)
    }
}

/// Distance calculator using USQ approximate scoring via Hanns.
pub struct USQDistCalculator<'a> {
    query_norm_sq: f32,
    query_state: hanns::quantization::usq::UsqQueryState,
    quantizer: &'a hanns::quantization::usq::UsqQuantizer,
    codes: &'a [u8],
    code_bytes: usize,
    meta: &'a FixedSizeListArray,
    num_vectors: usize,
}

impl DistCalculator for USQDistCalculator<'_> {
    fn distance(&self, id: u32) -> f32 {
        let id = id as usize;
        if id >= self.num_vectors {
            return f32::MAX;
        }

        let stored_codes = &self.codes[id * self.code_bytes..(id + 1) * self.code_bytes];

        let meta_vals = self.meta.value(id);
        let meta_slice = meta_vals
            .as_primitive::<arrow::datatypes::Float32Type>()
            .values();
        let stored_norm = meta_slice[0];
        let stored_vmax = meta_slice[2];
        let stored_qq = meta_slice[3];

        let score = self
            .quantizer
            .score_with_meta(&self.query_state, stored_norm, stored_vmax, stored_qq, stored_codes);

        // L2 distance: ||q||² + ||x||² - 2 * <q, x>
        self.query_norm_sq + meta_slice[1] - 2.0 * score
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        (0..self.num_vectors)
            .map(|id| self.distance(id as u32))
            .collect()
    }
}

impl VectorStore for USQStorage {
    type DistanceCalculator<'a> = USQDistCalculator<'a>;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send> {
        Ok(std::iter::once(self.batch.clone()))
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.value(id as usize)
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.values().iter()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn append_batch(&self, _batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        Err(Error::index("USQ does not support append_batch"))
    }

    fn dist_calculator(&self, query: ArrayRef, _dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        // Query is always a flat Float32Array, not FixedSizeListArray (see IVF search pipeline).
        let query_vals = query
            .as_primitive::<arrow::datatypes::Float32Type>()
            .values();

        // Get (or lazily init) the partition-level quantizer — QR runs at most once
        // per USQStorage lifetime, not once per query.
        let quantizer = self.cached_quantizer.get_or_init(|| {
            let usq_config =
                hanns::quantization::usq::UsqConfig::new(
                    self.metadata.dim as usize,
                    self.metadata.num_bits,
                )
                .expect("USQ config creation failed")
                .with_seed(self.metadata.rotation_seed);
            hanns::quantization::usq::UsqQuantizer::new(usq_config)
        });
        let query_state = quantizer.precompute_query_state(query_vals);
        let query_norm_sq: f32 = query_vals.iter().map(|v| v * v).sum();

        let codes_flat = self
            .codes
            .values()
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .values();

        USQDistCalculator {
            query_norm_sq,
            query_state,
            quantizer,
            codes: codes_flat,
            code_bytes: self.codes.value_length() as usize,
            meta: &self.meta,
            num_vectors: self.codes.values().len() / self.codes.value_length() as usize,
        }
    }

    fn dist_calculator_from_id(&self, _id: u32) -> Self::DistanceCalculator<'_> {
        unimplemented!("USQ does not support dist_calculator_from_id")
    }
}
