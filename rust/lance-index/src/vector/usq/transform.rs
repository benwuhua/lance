// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! USQ transformer for the IVF build pipeline.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{ArrayRef, FixedSizeListArray, UInt8Array};
use arrow_schema::Field;
use lance_arrow::*;
use lance_core::{Error, Result};

use crate::vector::transform::Transformer;
use crate::vector::usq::builder::USQuantizer;
use crate::vector::usq::{USQ_CODE_COLUMN, USQ_META_COLUMN, USQ_SIGN_COLUMN};

/// Transformer that quantizes vectors using USQ (4-bit).
#[derive(Debug)]
pub struct USQTransformer {
    quantizer: USQuantizer,
}

impl USQTransformer {
    pub fn new(quantizer: USQuantizer) -> Self {
        Self { quantizer }
    }
}

impl Transformer for USQTransformer {
    fn transform(&self, batch: &arrow_array::RecordBatch) -> Result<arrow_array::RecordBatch> {
        let schema = batch.schema();

        // Find the vectors column (last FixedSizeList<f32>) — identify by index so we can drop it.
        let (vec_col_idx, _) = batch
            .columns()
            .iter()
            .enumerate()
            .rev()
            .find(|(_, col)| {
                col.as_fixed_size_list_opt()
                    .is_some_and(|fsl| matches!(fsl.value_type(), arrow_schema::DataType::Float32))
            })
            .ok_or_else(|| Error::index("USQ: no float32 vector column found"))?;
        let vec_col_name = schema.field(vec_col_idx).name().clone();
        let vectors_col = batch.column(vec_col_idx);

        let mut arrays: Vec<(String, ArrayRef)> = Vec::with_capacity(batch.num_columns() + 2);

        // Keep existing columns, dropping the input vector column and any stale USQ columns.
        for i in 0..batch.num_columns() {
            let field = schema.field(i);
            if field.name() == USQ_CODE_COLUMN
                || field.name() == USQ_SIGN_COLUMN
                || field.name() == USQ_META_COLUMN
                || field.name() == &vec_col_name
            {
                continue;
            }
            arrays.push((field.name().clone(), batch.column(i).clone()));
        }

        let fsl = vectors_col.as_fixed_size_list();

        let encoded = self.quantizer.transform(fsl)?;

        let code_bytes = self.quantizer.code_bytes();

        // Add USQ code column
        let codes = UInt8Array::from(encoded.packed_bits);
        let code_fsl = FixedSizeListArray::try_new_from_values(codes, code_bytes as i32)?;
        arrays.push((USQ_CODE_COLUMN.to_string(), Arc::new(code_fsl)));

        // Add USQ meta column (norm, norm_sq, vmax, quant_quality as f32)
        let meta_values: Vec<f32> = encoded
            .norms
            .iter()
            .zip(encoded.norms_sq.iter())
            .zip(encoded.vmaxs.iter())
            .zip(encoded.quant_qualities.iter())
            .flat_map(|(((n, ns), vm), qq)| [*n, *ns, *vm, *qq])
            .collect();
        let meta_arr = arrow_array::Float32Array::from(meta_values);
        let meta_fsl = FixedSizeListArray::try_new_from_values(meta_arr, 4)?;
        arrays.push((USQ_META_COLUMN.to_string(), Arc::new(meta_fsl)));

        let fields: Vec<Field> = arrays
            .iter()
            .map(|(name, arr)| Field::new(name.clone(), arr.data_type().clone(), true))
            .collect();

        let schema = arrow_schema::Schema::new(fields);
        let columns: Vec<ArrayRef> = arrays.into_iter().map(|(_, arr)| arr).collect();

        Ok(arrow_array::RecordBatch::try_new(
            Arc::new(schema),
            columns,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt64Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::ROW_ID;

    use crate::vector::usq::USQ_SIGN_COLUMN;

    #[test]
    fn transform_emits_signless_usq_columns() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                USQ_SIGN_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 8),
                true,
            ),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 8),
                true,
            ),
        ]));

        let row_ids = Arc::new(UInt64Array::from(vec![1_u64, 2_u64]));
        let stale_signs = Arc::new(
            FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![0_u8; 16]), 8).unwrap(),
        );
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(vec![
                    0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                    0.1,
                ]),
                8,
            )
            .unwrap(),
        );

        let batch = RecordBatch::try_new(schema, vec![row_ids, stale_signs, vectors]).unwrap();
        let transformer = USQTransformer::new(USQuantizer::new(8, 4, 42));

        let transformed = transformer.transform(&batch).unwrap();

        assert!(transformed.column_by_name(ROW_ID).is_some());
        assert!(transformed.column_by_name(USQ_CODE_COLUMN).is_some());
        assert!(transformed.column_by_name(USQ_META_COLUMN).is_some());
        assert!(transformed.column_by_name(USQ_SIGN_COLUMN).is_none());
        assert!(transformed.column_by_name("vector").is_none());
    }
}
