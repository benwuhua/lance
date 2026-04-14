// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, OnceLock};

use arrow::array::AsArray;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array};
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

impl std::fmt::Debug for USQStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("USQStorage")
            .field("metadata", &self.metadata)
            .field("batch", &self.batch)
            .field("distance_type", &self.distance_type)
            .field("row_ids", &self.row_ids)
            .field("codes", &self.codes)
            .field("signs", &self.signs)
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
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

fn empty_signs(num_rows: usize) -> FixedSizeListArray {
    FixedSizeListArray::new_null(
        Arc::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::UInt8,
            true,
        )),
        0,
        num_rows,
    )
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
        let signs = batch
            .column_by_name(USQ_SIGN_COLUMN)
            .map(|arr| arr.as_fixed_size_list().clone())
            .unwrap_or_else(|| empty_signs(batch.num_rows()));
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
        let batch = match schema.project_preserve_system_columns(&[
            ROW_ID,
            USQ_CODE_COLUMN,
            USQ_META_COLUMN,
        ]) {
            Ok(projected_schema) => reader.read_range(range, &projected_schema).await?,
            Err(_) => reader.read_range(range, schema).await?,
        };
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, UInt8Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::datatypes::Schema;
    use lance_core::ROW_ID;
    use lance_io::object_store::ObjectStore;
    use lance_table::format::SelfDescribingFileReader;
    use lance_table::io::manifest::ManifestDescribing;
    use object_store::path::Path;

    use crate::vector::CENTROID_DIST_COLUMN;
    use crate::vector::quantizer::QuantizerStorage;
    use crate::vector::storage::VectorStore;
    use crate::vector::usq::builder::USQuantizer;

    use super::*;

    fn build_signless_storage(
        raw_vectors: &[Vec<f32>],
        num_bits: u8,
    ) -> (USQStorage, Vec<(f32, f32, f32, f32)>, Vec<u64>, usize) {
        let dim = raw_vectors.first().map(|v| v.len()).unwrap_or_default();
        let quantizer = USQuantizer::new(dim, num_bits, 42);
        let mut packed_bits = Vec::new();
        let mut meta_values = Vec::new();
        let mut encoded_rows = Vec::new();

        for vector in raw_vectors {
            let encoded = quantizer
                .transform(
                    &FixedSizeListArray::try_new_from_values(
                        Float32Array::from(vector.clone()),
                        dim as i32,
                    )
                    .unwrap(),
                )
                .unwrap();
            packed_bits.extend_from_slice(&encoded.packed_bits);
            meta_values.extend(
                encoded
                    .norms
                    .iter()
                    .zip(encoded.norms_sq.iter())
                    .zip(encoded.vmaxs.iter())
                    .zip(encoded.quant_qualities.iter())
                    .flat_map(|(((n, ns), vm), qq)| [*n, *ns, *vm, *qq]),
            );
            encoded_rows.push((
                encoded.norms[0],
                encoded.norms_sq[0],
                encoded.vmaxs[0],
                encoded.quant_qualities[0],
            ));
        }

        let code_bytes = quantizer.code_bytes();
        let row_ids: Vec<u64> = (0..raw_vectors.len()).map(|idx| (idx + 1) as u64 * 10).collect();
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new(ROW_ID, DataType::UInt64, false),
                Field::new(
                    USQ_CODE_COLUMN,
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::UInt8, true)),
                        code_bytes as i32,
                    ),
                    true,
                ),
                Field::new(
                    USQ_META_COLUMN,
                    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                    true,
                ),
            ])),
            vec![
                Arc::new(UInt64Array::from(row_ids.clone())),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(packed_bits),
                        code_bytes as i32,
                    )
                    .unwrap(),
                ),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(meta_values), 4)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

        let metadata = UsqQuantizationMetadata {
            dim: dim as u32,
            num_bits,
            rotation_seed: 42,
        };
        let storage = <USQStorage as QuantizerStorage>::try_from_batch(
            batch,
            &metadata,
            DistanceType::L2,
            None,
        )
        .unwrap();

        (storage, encoded_rows, row_ids, code_bytes)
    }

    #[tokio::test]
    async fn test_load_partition_projects_search_columns() {
        let object_store = ObjectStore::memory();
        let path = Path::from("/usq_storage_projection");
        let arrow_schema = ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(CENTROID_DIST_COLUMN, DataType::Float32, true),
            Field::new(
                USQ_CODE_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 4),
                true,
            ),
            Field::new(
                USQ_SIGN_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 1),
                true,
            ),
            Field::new(
                USQ_META_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                true,
            ),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let mut writer =
            lance_file::previous::writer::FileWriter::<ManifestDescribing>::try_new(
            &object_store,
            &path,
            schema,
            &Default::default(),
        )
        .await
        .unwrap();
        let row_ids = Arc::new(UInt64Array::from(vec![10_u64, 20_u64]));
        let centroid_dists = Arc::new(Float32Array::from(vec![0.25_f32, 0.5_f32]));
        let codes = Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(vec![1_u8, 2, 3, 4, 5, 6, 7, 8]),
            4,
        )
        .unwrap());
        let signs = Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(vec![0_u8, 1_u8]),
            1,
        )
        .unwrap());
        let meta = Arc::new(FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 16.0_f32, 5.0_f32, 6.0_f32,
            ]),
            4,
        )
        .unwrap());
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![row_ids, centroid_dists, codes, signs, meta],
        )
        .unwrap();
        writer.write(&[batch]).await.unwrap();
        writer.finish().await.unwrap();

        let reader = PreviousFileReader::try_new_self_described(&object_store, &path, None)
            .await
            .unwrap();
        let metadata = UsqQuantizationMetadata {
            dim: 8,
            num_bits: 4,
            rotation_seed: 42,
        };

        let storage =
            USQStorage::load_partition(&reader, 0..2, DistanceType::L2, &metadata, None)
                .await
                .unwrap();

        assert_eq!(storage.batch.num_columns(), 3);
        assert!(storage.batch.column_by_name(ROW_ID).is_some());
        assert!(storage.batch.column_by_name(USQ_CODE_COLUMN).is_some());
        assert!(storage.batch.column_by_name(USQ_META_COLUMN).is_some());
        assert!(storage.batch.column_by_name(USQ_SIGN_COLUMN).is_none());
        assert!(storage.batch.column_by_name(CENTROID_DIST_COLUMN).is_none());
    }

    #[test]
    fn try_from_batch_scores_identically_with_or_without_legacy_signs() {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                0.1,
            ]),
            8,
        )
        .unwrap();
        let quantizer = USQuantizer::new(8, 4, 42);
        let encoded = quantizer.transform(&vectors).unwrap();

        let code_bytes = quantizer.code_bytes() as i32;
        let row_ids = Arc::new(UInt64Array::from(vec![10_u64, 20_u64]));
        let codes = Arc::new(
            FixedSizeListArray::try_new_from_values(
                UInt8Array::from(encoded.packed_bits),
                code_bytes,
            )
            .unwrap(),
        );
        let meta_values: Vec<f32> = encoded
            .norms
            .iter()
            .zip(encoded.norms_sq.iter())
            .zip(encoded.vmaxs.iter())
            .zip(encoded.quant_qualities.iter())
            .flat_map(|(((n, ns), vm), qq)| [*n, *ns, *vm, *qq])
            .collect();
        let meta = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(meta_values), 4).unwrap(),
        );
        let legacy_signs = Arc::new(
            FixedSizeListArray::try_new_from_values(
                UInt8Array::from(vec![0_u8; vectors.len() * (quantizer.padded_dim() / 8)]),
                (quantizer.padded_dim() / 8) as i32,
            )
            .unwrap(),
        );

        let schema_without_signs = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                USQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    code_bytes,
                ),
                true,
            ),
            Field::new(
                USQ_META_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                true,
            ),
        ]));
        let schema_with_signs = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                USQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    code_bytes,
                ),
                true,
            ),
            Field::new(
                USQ_SIGN_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    (quantizer.padded_dim() / 8) as i32,
                ),
                true,
            ),
            Field::new(
                USQ_META_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                true,
            ),
        ]));

        let signless_batch = RecordBatch::try_new(
            schema_without_signs,
            vec![row_ids.clone(), codes.clone(), meta.clone()],
        )
        .unwrap();
        let legacy_batch =
            RecordBatch::try_new(schema_with_signs, vec![row_ids, codes, legacy_signs, meta])
                .unwrap();

        let metadata = UsqQuantizationMetadata {
            dim: 8,
            num_bits: 4,
            rotation_seed: 42,
        };
        let signless_storage =
            <USQStorage as QuantizerStorage>::try_from_batch(
                signless_batch,
                &metadata,
                DistanceType::L2,
                None,
            )
            .unwrap();
        let legacy_storage =
            <USQStorage as QuantizerStorage>::try_from_batch(
                legacy_batch,
                &metadata,
                DistanceType::L2,
                None,
            )
            .unwrap();

        let query = Arc::new(Float32Array::from(vec![
            0.15_f32, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85,
        ]));
        let signless_calc = signless_storage.dist_calculator(query.clone(), 0.0);
        let legacy_calc = legacy_storage.dist_calculator(query, 0.0);

        for id in 0..vectors.len() as u32 {
            let signless_distance = signless_calc.distance(id);
            let legacy_distance = legacy_calc.distance(id);
            assert!(
                (signless_distance - legacy_distance).abs() < 1e-5,
                "distance mismatch at row {id}: signless={} legacy={}",
                signless_distance,
                legacy_distance
            );
        }
    }

    #[test]
    fn signless_storage_preserves_hanns_score_ordering() {
        let raw_vectors = vec![
            vec![0.10_f32, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            vec![0.82_f32, 0.72, 0.62, 0.52, 0.42, 0.32, 0.22, 0.12],
            vec![0.30_f32, 0.10, 0.60, 0.20, 0.90, 0.40, 0.80, 0.50],
            vec![0.95_f32, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
        ];
        let (storage, encoded_rows, _row_ids, code_bytes) = build_signless_storage(&raw_vectors, 4);

        let query = vec![0.18_f32, 0.28, 0.38, 0.48, 0.58, 0.68, 0.78, 0.88];
        let query_arr = Arc::new(Float32Array::from(query.clone()));
        let storage_calc = storage.dist_calculator(query_arr, 0.0);

        let hanns_config = hanns::quantization::usq::UsqConfig::new(8, 4)
            .unwrap()
            .with_seed(42);
        let hanns_quantizer = hanns::quantization::usq::UsqQuantizer::new(hanns_config);
        let query_state = hanns_quantizer.precompute_query_state(&query);
        let query_norm_sq: f32 = query.iter().map(|v| v * v).sum();

        let mut expected_order: Vec<(usize, f32)> = raw_vectors
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                let (norm, norm_sq, vmax, qq) = encoded_rows[idx];
                let row_codes =
                    &storage.codes.values().as_primitive::<arrow::datatypes::UInt8Type>().values()
                        [idx * code_bytes..(idx + 1) * code_bytes];
                let score =
                    hanns_quantizer.score_with_meta(&query_state, norm, vmax, qq, row_codes);
                let distance = query_norm_sq + norm_sq - 2.0 * score;
                (idx, distance)
            })
            .collect();
        expected_order.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut storage_order: Vec<(usize, f32)> = (0..raw_vectors.len())
            .map(|idx| (idx, storage_calc.distance(idx as u32)))
            .collect();
        storage_order.sort_by(|a, b| a.1.total_cmp(&b.1));

        assert_eq!(
            storage_order
                .iter()
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>(),
            expected_order
                .iter()
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn signless_storage_preserves_topk_against_hanns_reference() {
        let raw_vectors = vec![
            vec![0.05_f32, 0.12, 0.18, 0.21, 0.29, 0.33, 0.41, 0.52],
            vec![0.90_f32, 0.82, 0.72, 0.61, 0.52, 0.40, 0.28, 0.14],
            vec![0.21_f32, 0.45, 0.11, 0.67, 0.30, 0.74, 0.28, 0.61],
            vec![0.88_f32, 0.10, 0.22, 0.35, 0.49, 0.58, 0.69, 0.80],
            vec![0.17_f32, 0.24, 0.39, 0.43, 0.57, 0.68, 0.79, 0.91],
            vec![0.63_f32, 0.55, 0.47, 0.38, 0.26, 0.19, 0.08, 0.03],
        ];
        let (storage, encoded_rows, row_ids, code_bytes) = build_signless_storage(&raw_vectors, 4);

        let queries = vec![
            vec![0.10_f32, 0.18, 0.25, 0.30, 0.41, 0.49, 0.58, 0.67],
            vec![0.91_f32, 0.79, 0.71, 0.59, 0.48, 0.37, 0.25, 0.11],
            vec![0.26_f32, 0.41, 0.16, 0.59, 0.34, 0.69, 0.31, 0.55],
        ];
        let top_k = 3;
        let hanns_quantizer = hanns::quantization::usq::UsqQuantizer::new(
            hanns::quantization::usq::UsqConfig::new(8, 4)
                .unwrap()
                .with_seed(42),
        );

        for query in queries {
            let storage_calc =
                storage.dist_calculator(Arc::new(Float32Array::from(query.clone())), 0.0);
            let query_state = hanns_quantizer.precompute_query_state(&query);
            let query_norm_sq: f32 = query.iter().map(|v| v * v).sum();

            let mut expected_topk: Vec<(u64, f32)> = row_ids
                .iter()
                .enumerate()
                .map(|(idx, row_id)| {
                    let (norm, norm_sq, vmax, qq) = encoded_rows[idx];
                    let row_codes =
                        &storage.codes.values().as_primitive::<arrow::datatypes::UInt8Type>().values()
                            [idx * code_bytes..(idx + 1) * code_bytes];
                    let score =
                        hanns_quantizer.score_with_meta(&query_state, norm, vmax, qq, row_codes);
                    let distance = query_norm_sq + norm_sq - 2.0 * score;
                    (*row_id, distance)
                })
                .collect();
            expected_topk.sort_by(|a, b| a.1.total_cmp(&b.1));
            expected_topk.truncate(top_k);

            let mut storage_topk: Vec<(u64, f32)> = row_ids
                .iter()
                .enumerate()
                .map(|(idx, row_id)| (*row_id, storage_calc.distance(idx as u32)))
                .collect();
            storage_topk.sort_by(|a, b| a.1.total_cmp(&b.1));
            storage_topk.truncate(top_k);

            assert_eq!(
                storage_topk
                    .iter()
                    .map(|(row_id, _)| row_id)
                    .collect::<Vec<_>>(),
                expected_topk
                    .iter()
                    .map(|(row_id, _)| row_id)
                    .collect::<Vec<_>>()
            );
        }
    }
}
