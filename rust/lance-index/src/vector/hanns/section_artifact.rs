// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory Hanns sectioned artifact bridge for Lance-owned storage bytes.

use std::collections::BTreeMap;

use lance_core::{Error, Result};

use crate::vector::hanns::section_manifest::HannsSectionManifest;

/// A validated Hanns sectioned artifact assembled from Lance-owned section bytes.
#[derive(Debug, Clone)]
pub struct HannsSectionArtifact {
    manifest: HannsSectionManifest,
    sections: BTreeMap<String, Vec<u8>>,
}

impl HannsSectionArtifact {
    /// Build an artifact from a parsed Hanns manifest and named section bytes.
    pub fn new(
        manifest: HannsSectionManifest,
        sections: BTreeMap<String, Vec<u8>>,
    ) -> Result<Self> {
        validate_manifest_sections(&manifest, &sections)?;
        Ok(Self { manifest, sections })
    }

    /// Build an artifact from an iterator of named section byte buffers.
    pub fn from_sections(
        manifest: HannsSectionManifest,
        sections: impl IntoIterator<Item = (String, Vec<u8>)>,
    ) -> Result<Self> {
        Self::new(manifest, sections.into_iter().collect())
    }

    /// Return the Lance-side parsed manifest.
    pub fn manifest(&self) -> &HannsSectionManifest {
        &self.manifest
    }

    /// Return the serialized payload for a named section.
    pub fn read_section_bytes(&self, name: &str) -> Result<&[u8]> {
        self.sections
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::index(format!("missing Hanns section: {name}")))
    }

    /// Return the byte length of a named section.
    pub fn section_len(&self, name: &str) -> Result<u64> {
        self.manifest
            .section(name)
            .map(|section| section.len)
            .ok_or_else(|| Error::index(format!("missing Hanns section descriptor: {name}")))
    }

    /// Return a byte range from a named section.
    pub fn read_range_bytes(&self, name: &str, offset: u64, len: usize) -> Result<&[u8]> {
        let section = self.read_section_bytes(name)?;
        let offset = usize::try_from(offset)
            .map_err(|_| Error::index(format!("section range offset overflows usize: {name}")))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::index(format!("section range overflows usize: {name}")))?;
        section
            .get(offset..end)
            .ok_or_else(|| Error::index(format!("section range out of bounds: {name}")))
    }

    /// Consume this artifact and return the section byte map.
    pub fn into_sections(self) -> BTreeMap<String, Vec<u8>> {
        self.sections
    }
}

fn validate_manifest_sections(
    manifest: &HannsSectionManifest,
    sections: &BTreeMap<String, Vec<u8>>,
) -> Result<()> {
    for descriptor in &manifest.sections {
        let bytes = sections.get(&descriptor.name).ok_or_else(|| {
            Error::index(format!(
                "Hanns manifest variant {} is missing section bytes for {}",
                manifest.variant, descriptor.name
            ))
        })?;
        if bytes.len() as u64 != descriptor.len {
            return Err(Error::index(format!(
                "Hanns section {} length mismatch: manifest={} actual={}",
                descriptor.name,
                descriptor.len,
                bytes.len()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> HannsSectionManifest {
        HannsSectionManifest::from_json(
            r#"{
                "version": 1,
                "family": "hnsw",
                "variant": "hnsw_sections_v1",
                "dim": 4,
                "metric": "l2",
                "count": 2,
                "supported_load_modes": ["owned_memory"],
                "features": {
                    "raw_vectors": true,
                    "graph_payload": true
                },
                "sections": [
                    {"name": "hnsw.meta.json", "len": 2},
                    {"name": "hnsw.vectors.f32", "len": 4}
                ]
            }"#,
        )
        .expect("manifest should parse")
    }

    #[test]
    fn validates_manifest_sections_and_reads_ranges() {
        let artifact = HannsSectionArtifact::from_sections(
            manifest(),
            [
                ("hnsw.meta.json".to_string(), vec![1, 2]),
                ("hnsw.vectors.f32".to_string(), vec![3, 4, 5, 6]),
            ],
        )
        .expect("artifact should validate");

        assert_eq!(artifact.section_len("hnsw.vectors.f32").unwrap(), 4);
        assert_eq!(
            artifact.read_range_bytes("hnsw.vectors.f32", 1, 2).unwrap(),
            &[4, 5]
        );
        assert_eq!(artifact.manifest().variant, "hnsw_sections_v1");
    }

    #[test]
    fn rejects_missing_section_bytes() {
        let error = HannsSectionArtifact::from_sections(
            manifest(),
            [("hnsw.meta.json".to_string(), vec![1, 2])],
        )
        .expect_err("missing section should be rejected");

        assert!(
            error.to_string().contains("hnsw.vectors.f32"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_section_length_mismatch() {
        let error = HannsSectionArtifact::from_sections(
            manifest(),
            [
                ("hnsw.meta.json".to_string(), vec![1, 2]),
                ("hnsw.vectors.f32".to_string(), vec![3]),
            ],
        )
        .expect_err("length mismatch should be rejected");

        assert!(
            error.to_string().contains("length mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn consumes_section_map_after_validation() {
        let artifact = HannsSectionArtifact::from_sections(
            manifest(),
            [
                ("hnsw.meta.json".to_string(), vec![1, 2]),
                ("hnsw.vectors.f32".to_string(), vec![3, 4, 5, 6]),
            ],
        )
        .expect("artifact should validate");

        let sections = artifact.into_sections();
        assert_eq!(sections["hnsw.meta.json"], vec![1, 2]);
        assert_eq!(sections["hnsw.vectors.f32"], vec![3, 4, 5, 6]);
    }
}
