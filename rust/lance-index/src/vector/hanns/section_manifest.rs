// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Hanns sectioned snapshot manifest parsing for Lance adapter planning.

use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Load modes advertised by Hanns sectioned snapshot manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HannsLoadMode {
    /// Materialize the snapshot into process memory before search.
    OwnedMemory,
    /// Use memory-mapped section access.
    Mmap,
    /// Use page-cache backed section access.
    PageCache,
    /// Lazily open sections as search needs them.
    Lazy,
}

/// Payload feature flags advertised by Hanns sectioned snapshot manifests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HannsManifestFeatures {
    /// Snapshot contains raw vector payloads.
    #[serde(default)]
    pub raw_vectors: bool,
    /// Snapshot contains graph adjacency or graph-level routing payloads.
    #[serde(default)]
    pub graph_payload: bool,
    /// Snapshot contains quantizer/codebook metadata or quantized codes.
    #[serde(default)]
    pub quantized_payload: bool,
    /// Snapshot contains compressed vector codes used for search.
    #[serde(default)]
    pub compressed_vectors: bool,
}

/// Minimal Hanns sectioned snapshot manifest shape needed by Lance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HannsSectionManifest {
    /// Manifest schema version.
    pub version: u32,
    /// ANN family, for example `hnsw`, `ivf`, or `disk_ann`.
    pub family: String,
    /// Concrete Hanns sectioned snapshot variant.
    pub variant: String,
    /// Vector dimension.
    pub dim: usize,
    /// Metric name.
    pub metric: String,
    /// Logical vector count.
    pub count: usize,
    /// Load modes supported by the artifact.
    #[serde(default = "default_load_modes")]
    pub supported_load_modes: Vec<HannsLoadMode>,
    /// Storage planning feature flags.
    #[serde(default)]
    pub features: HannsManifestFeatures,
}

/// Storage planning hints derived from Hanns manifest capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HannsSectionStoragePlan {
    /// True when the artifact can be materialized in memory.
    pub can_load_owned_memory: bool,
    /// True when raw vectors are present.
    pub has_raw_vectors: bool,
    /// True when graph adjacency/routing payloads are present.
    pub has_graph_payload: bool,
    /// True when quantized payloads are present.
    pub has_quantized_payload: bool,
    /// True when compressed vector payloads are present.
    pub has_compressed_vectors: bool,
}

impl HannsSectionManifest {
    /// Parse a Hanns sectioned snapshot manifest JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|error| Error::index(format!("failed to parse Hanns manifest JSON: {error}")))
    }

    /// Return true when the manifest advertises support for `mode`.
    pub fn supports_load_mode(&self, mode: HannsLoadMode) -> bool {
        self.supported_load_modes.contains(&mode)
    }

    /// Convert manifest capabilities into a compact storage planning summary.
    pub fn storage_plan(&self) -> HannsSectionStoragePlan {
        HannsSectionStoragePlan {
            can_load_owned_memory: self.supports_load_mode(HannsLoadMode::OwnedMemory),
            has_raw_vectors: self.features.raw_vectors,
            has_graph_payload: self.features.graph_payload,
            has_quantized_payload: self.features.quantized_payload,
            has_compressed_vectors: self.features.compressed_vectors,
        }
    }

    /// Validate that this artifact can be consumed through Lance's current owned-memory adapter path.
    pub fn validate_for_lance_owned_memory(&self) -> Result<()> {
        if self.version != 1 {
            return Err(Error::index(format!(
                "unsupported Hanns manifest version {}; expected 1",
                self.version
            )));
        }
        if self.dim == 0 {
            return Err(Error::index(
                "Hanns manifest dimension must be greater than 0",
            ));
        }
        if !self.supports_load_mode(HannsLoadMode::OwnedMemory) {
            return Err(Error::index(format!(
                "Hanns manifest variant {} does not support owned-memory load",
                self.variant
            )));
        }
        Ok(())
    }
}

fn default_load_modes() -> Vec<HannsLoadMode> {
    vec![HannsLoadMode::OwnedMemory]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hanns_manifest_capabilities() {
        let json = r#"{
            "version": 1,
            "family": "disk_ann",
            "variant": "pqflash_sections_v1",
            "dim": 128,
            "metric": "l2",
            "count": 1000,
            "supported_load_modes": ["owned_memory"],
            "features": {
                "raw_vectors": true,
                "graph_payload": true,
                "quantized_payload": true,
                "compressed_vectors": true
            },
            "sections": []
        }"#;

        let manifest = HannsSectionManifest::from_json(json).expect("manifest should parse");
        let plan = manifest.storage_plan();

        assert_eq!(manifest.variant, "pqflash_sections_v1");
        assert!(plan.can_load_owned_memory);
        assert!(plan.has_raw_vectors);
        assert!(plan.has_graph_payload);
        assert!(plan.has_quantized_payload);
        assert!(plan.has_compressed_vectors);
        manifest
            .validate_for_lance_owned_memory()
            .expect("owned-memory manifest should validate");
    }

    #[test]
    fn defaults_legacy_hanns_manifest_capabilities() {
        let json = r#"{
            "version": 1,
            "family": "ivf",
            "variant": "ivf_flat_sections_v1",
            "dim": 4,
            "metric": "l2",
            "count": 3,
            "sections": []
        }"#;

        let manifest = HannsSectionManifest::from_json(json).expect("manifest should parse");
        let plan = manifest.storage_plan();

        assert_eq!(
            manifest.supported_load_modes,
            vec![HannsLoadMode::OwnedMemory]
        );
        assert!(plan.can_load_owned_memory);
        assert!(!plan.has_raw_vectors);
        assert!(!plan.has_graph_payload);
        assert!(!plan.has_quantized_payload);
        assert!(!plan.has_compressed_vectors);
    }

    #[test]
    fn rejects_manifest_without_owned_memory_support() {
        let json = r#"{
            "version": 1,
            "family": "disk_ann",
            "variant": "future_page_cache",
            "dim": 16,
            "metric": "l2",
            "count": 4,
            "supported_load_modes": ["page_cache"],
            "sections": []
        }"#;

        let manifest = HannsSectionManifest::from_json(json).expect("manifest should parse");
        let error = manifest
            .validate_for_lance_owned_memory()
            .expect_err("owned memory should be rejected");

        assert!(
            error.to_string().contains("owned-memory"),
            "unexpected error: {error}"
        );
    }
}
