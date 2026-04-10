// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! USQ transformer for the IVF build pipeline.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{ArrayRef, Float32Array, FixedSizeListArray, UInt8Array};
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
        let mut arrays: Vec<(String, ArrayRef)> = Vec::with_capacity(batch.num_columns() + 2);

        // Keep existing columns
        let schema = batch.schema();
        for i in 0..batch.num_columns() {
            let field = schema.field(i);
            if field.name() == USQ_CODE_COLUMN
                || field.name() == USQ_SIGN_COLUMN
                || field.name() == USQ_META_COLUMN
            {
                continue;
            }
            arrays.push((field.name().clone(), batch.column(i).clone()));
        }

        // Find the vectors column (last FixedSizeList<f32>)
        let vectors_col = batch
            .columns()
            .iter()
            .rev()
            .find(|col| {
                col.as_fixed_size_list_opt().map_or(false, |fsl| {
                    matches!(fsl.value_type(), arrow_schema::DataType::Float32)
                })
            })
            .ok_or_else(|| Error::index("USQ: no float32 vector column found"))?;

        let fsl = vectors_col.as_fixed_size_list();

        let encoded = self.quantizer.transform(fsl)?;

        let code_bytes = self.quantizer.code_bytes();
        let sign_bytes = self.quantizer.sign_bytes();

        // Add USQ code column
        let codes = UInt8Array::from(encoded.packed_bits);
        let code_fsl = FixedSizeListArray::try_new_from_values(codes, code_bytes as i32)?;
        arrays.push((USQ_CODE_COLUMN.to_string(), Arc::new(code_fsl)));

        // Add USQ sign column
        let signs = UInt8Array::from(encoded.sign_bits);
        let sign_fsl = FixedSizeListArray::try_new_from_values(signs, sign_bytes as i32)?;
        arrays.push((USQ_SIGN_COLUMN.to_string(), Arc::new(sign_fsl)));

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

        Ok(arrow_array::RecordBatch::try_new(Arc::new(schema), columns)?)
    }
}
