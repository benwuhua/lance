// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, UInt8Array};
use arrow_schema::{DataType, Field};
use deepsize::DeepSizeOf;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use rayon::prelude::*;

use crate::vector::quantizer::{Quantization, Quantizer, QuantizerBuildParams};
use crate::vector::usq::storage::{USQStorage, UsqQuantizationMetadata};
use crate::vector::usq::{USQBuildParams, USQ_CODE_COLUMN, USQ_META_COLUMN, USQ_METADATA_KEY, USQ_SIGN_COLUMN};

#[derive(Debug, Clone, DeepSizeOf)]
pub struct USQuantizer {
    dim: usize,
    num_bits: u8,
    rotation_seed: u64,
}

impl USQuantizer {
    pub fn new(dim: usize, num_bits: u8, rotation_seed: u64) -> Self {
        Self {
            dim,
            num_bits,
            rotation_seed,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn num_bits(&self) -> u8 {
        self.num_bits
    }

    /// Padded dimension (round up to multiple of 64).
    pub fn padded_dim(&self) -> usize {
        (self.dim + 63) / 64 * 64
    }

    /// Code bytes per vector for packed bits.
    pub fn code_bytes(&self) -> usize {
        self.padded_dim() * self.num_bits as usize / 8
    }

    /// Sign bytes per vector.
    pub fn sign_bytes(&self) -> usize {
        self.padded_dim() / 8
    }

    pub fn transform(&self, vectors: &FixedSizeListArray) -> Result<EncodedBatch> {
        let n = vectors.len();
        let dim = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<arrow::datatypes::Float32Type>();

        let code_bytes = self.code_bytes();
        let sign_bytes = self.sign_bytes();

        let mut packed_bits = vec![0u8; n * code_bytes];
        let mut sign_bits = vec![0u8; n * sign_bytes];
        let mut norms = Vec::with_capacity(n);
        let mut norms_sq = Vec::with_capacity(n);
        let mut vmaxs = Vec::with_capacity(n);
        let mut quant_qualities = Vec::with_capacity(n);

        let usq_config =
            hanns::quantization::usq::UsqConfig::new(dim, self.num_bits)
                .map_err(|e| Error::index(format!("USQ config error: {}", e)))?
                .with_seed(self.rotation_seed);

        // Encode vectors in parallel using rayon
        let results: Vec<Result<EncodedVector>> = values
            .values()
            .par_chunks_exact(dim)
            .map(|vector| {
                let quantizer = hanns::quantization::usq::UsqQuantizer::new(usq_config.clone());
                let encoded = quantizer.encode(vector);
                Ok(EncodedVector {
                    packed_bits: encoded.packed_bits.clone(),
                    sign_bits: encoded.sign_bits.clone(),
                    norm: encoded.norm,
                    norm_sq: encoded.norm_sq,
                    vmax: encoded.vmax,
                    quant_quality: encoded.quant_quality,
                })
            })
            .collect();

        for (i, result) in results.into_iter().enumerate() {
            let encoded = result?;
            packed_bits[i * code_bytes..(i + 1) * code_bytes].copy_from_slice(&encoded.packed_bits);
            sign_bits[i * sign_bytes..(i + 1) * sign_bytes].copy_from_slice(&encoded.sign_bits);
            norms.push(encoded.norm);
            norms_sq.push(encoded.norm_sq);
            vmaxs.push(encoded.vmax);
            quant_qualities.push(encoded.quant_quality);
        }

        Ok(EncodedBatch {
            packed_bits,
            sign_bits,
            norms,
            norms_sq,
            vmaxs,
            quant_qualities,
        })
    }
}

pub(crate) struct EncodedVector {
    pub packed_bits: Vec<u8>,
    pub sign_bits: Vec<u8>,
    pub norm: f32,
    pub norm_sq: f32,
    pub vmax: f32,
    pub quant_quality: f32,
}

pub(crate) struct EncodedBatch {
    pub packed_bits: Vec<u8>,
    pub sign_bits: Vec<u8>,
    pub norms: Vec<f32>,
    pub norms_sq: Vec<f32>,
    pub vmaxs: Vec<f32>,
    pub quant_qualities: Vec<f32>,
}

impl Quantization for USQuantizer {
    type BuildParams = USQBuildParams;
    type Metadata = UsqQuantizationMetadata;
    type Storage = USQStorage;

    fn build(
        data: &dyn Array,
        _: lance_linalg::distance::DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        let fsl = data.as_fixed_size_list();
        let dim = fsl.value_length() as usize;
        if fsl.value_type() != DataType::Float32 {
            return Err(Error::invalid_input(format!(
                "USQ only supports Float32, got {:?}",
                fsl.value_type()
            )));
        }
        Ok(Self::new(dim, params.num_bits, params.rotation_seed))
    }

    fn retrain(&mut self, _data: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.code_bytes()
    }

    fn column(&self) -> &'static str {
        USQ_CODE_COLUMN
    }

    fn use_residual(_: lance_linalg::distance::DistanceType) -> bool {
        true
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        let fsl = vectors
            .as_fixed_size_list_opt()
            .ok_or_else(|| Error::index("USQ expects FixedSizeList input"))?;

        let batch = self.transform(fsl)?;
        let code_bytes = self.code_bytes();
        let n = fsl.len();

        let codes = UInt8Array::from(batch.packed_bits);
        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            codes,
            code_bytes as i32,
        )?))
    }

    fn metadata_key() -> &'static str {
        USQ_METADATA_KEY
    }

    fn quantization_type() -> crate::vector::quantizer::QuantizationType {
        crate::vector::quantizer::QuantizationType::Usq
    }

    fn metadata(
        &self,
        _args: Option<crate::vector::quantizer::QuantizationMetadata>,
    ) -> Self::Metadata {
        UsqQuantizationMetadata {
            dim: self.dim as u32,
            num_bits: self.num_bits,
            rotation_seed: self.rotation_seed,
        }
    }

    fn from_metadata(
        metadata: &Self::Metadata,
        _: lance_linalg::distance::DistanceType,
    ) -> Result<Quantizer> {
        Ok(Quantizer::Usq(Self::new(
            metadata.dim as usize,
            metadata.num_bits,
            metadata.rotation_seed,
        )))
    }

    fn field(&self) -> Field {
        Field::new(
            USQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.code_bytes() as i32,
            ),
            true,
        )
    }

    fn extra_fields(&self) -> Vec<Field> {
        vec![
            Field::new(
                USQ_SIGN_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    self.sign_bytes() as i32,
                ),
                true,
            ),
            Field::new(
                USQ_META_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    4, // norm, norm_sq, vmax, quant_quality
                ),
                true,
            ),
        ]
    }
}

impl TryFrom<Quantizer> for USQuantizer {
    type Error = Error;

    fn try_from(quantizer: Quantizer) -> Result<Self> {
        match quantizer {
            Quantizer::Usq(q) => Ok(q),
            _ => Err(Error::invalid_input(
                "Cannot convert non-USQuantizer to USQuantizer",
            )),
        }
    }
}

impl From<USQuantizer> for Quantizer {
    fn from(q: USQuantizer) -> Self {
        Self::Usq(q)
    }
}
