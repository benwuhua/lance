// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ultra-Sparse Quantization (USQ) backed by Hanns.

use crate::vector::quantizer::QuantizerBuildParams;
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod storage;
pub mod transform;

pub const USQ_CODE_COLUMN: &str = "__usq_code";
pub const USQ_SIGN_COLUMN: &str = "__usq_sign";
pub const USQ_META_COLUMN: &str = "__usq_meta";
pub const USQ_METADATA_KEY: &str = "lance:usq";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum USQRotationType {
    #[default]
    Random,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct USQBuildParams {
    pub num_bits: u8,
    pub rotation_seed: u64,
}

impl USQBuildParams {
    pub fn new(num_bits: u8) -> Self {
        Self {
            num_bits,
            rotation_seed: 42,
        }
    }

    pub fn with_seed(num_bits: u8, rotation_seed: u64) -> Self {
        Self {
            num_bits,
            rotation_seed,
        }
    }
}

impl Default for USQBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 4,
            rotation_seed: 42,
        }
    }
}

impl QuantizerBuildParams for USQBuildParams {
    fn sample_size(&self) -> usize {
        0
    }
}
