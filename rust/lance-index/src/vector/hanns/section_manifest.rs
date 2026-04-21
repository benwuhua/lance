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

/// One named payload section in a Hanns sectioned snapshot artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HannsSectionDescriptor {
    /// Stable section name used by the Hanns snapshot loader.
    pub name: String,
    /// Section payload length in bytes.
    pub len: u64,
    /// Optional payload checksum advertised by Hanns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
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
    /// Named payload sections in the artifact.
    #[serde(default)]
    pub sections: Vec<HannsSectionDescriptor>,
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
    /// Number of section descriptors advertised by the manifest.
    pub section_count: usize,
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
            section_count: self.sections.len(),
        }
    }

    /// Return the section descriptor with `name`, if present.
    pub fn section(&self, name: &str) -> Option<&HannsSectionDescriptor> {
        self.sections.iter().find(|section| section.name == name)
    }

    /// Return true when the manifest advertises a section with `name`.
    pub fn has_section(&self, name: &str) -> bool {
        self.section(name).is_some()
    }

    /// Validate that all `required_sections` are advertised by this manifest.
    pub fn validate_sections_present(&self, required_sections: &[&str]) -> Result<()> {
        for section in required_sections {
            if !self.has_section(section) {
                return Err(Error::index(format!(
                    "Hanns manifest variant {} is missing required section {}",
                    self.variant, section
                )));
            }
        }
        Ok(())
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
            "sections": [
                {"name": "pqflash.meta.json", "len": 128},
                {"name": "pqflash.node_pq_codes.u8", "len": 4096, "checksum": "sha256:abc"}
            ]
        }"#;

        let manifest = HannsSectionManifest::from_json(json).expect("manifest should parse");
        let plan = manifest.storage_plan();

        assert_eq!(manifest.variant, "pqflash_sections_v1");
        assert!(plan.can_load_owned_memory);
        assert!(plan.has_raw_vectors);
        assert!(plan.has_graph_payload);
        assert!(plan.has_quantized_payload);
        assert!(plan.has_compressed_vectors);
        assert_eq!(plan.section_count, 2);
        assert_eq!(
            manifest
                .section("pqflash.node_pq_codes.u8")
                .and_then(|section| section.checksum.as_deref()),
            Some("sha256:abc")
        );
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
        assert_eq!(plan.section_count, 0);
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

    #[test]
    fn validates_required_sections_present() {
        let json = r#"{
            "version": 1,
            "family": "hnsw",
            "variant": "hnsw_sections_v1",
            "dim": 8,
            "metric": "l2",
            "count": 2,
            "sections": [
                {"name": "hnsw.meta.json", "len": 48},
                {"name": "hnsw.vectors.f32", "len": 64}
            ]
        }"#;

        let manifest = HannsSectionManifest::from_json(json).expect("manifest should parse");
        manifest
            .validate_sections_present(&["hnsw.meta.json", "hnsw.vectors.f32"])
            .expect("required sections should validate");

        let error = manifest
            .validate_sections_present(&["hnsw.neighbors.ids.i64"])
            .expect_err("missing section should be rejected");
        assert!(
            error.to_string().contains("hnsw.neighbors.ids.i64"),
            "unexpected error: {error}"
        );
    }
}
