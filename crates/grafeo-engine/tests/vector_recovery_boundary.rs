//! Reject unsealed foreign Vector images without modifying the source file.
//!
//! Current Catalog/LPG4 bytes remain unchanged while metadata is removed,
//! downgraded, or kept with its original recovery-image seal. A valid foreign
//! exact payload cannot be treated as a cache hint or blessed by writable open.

#![cfg(all(
    feature = "lpg",
    feature = "wal",
    feature = "grafeo-file",
    feature = "vector-index"
))]

use std::path::Path;

use grafeo_common::storage::SectionType;
use grafeo_common::types::{
    AuthoritativeFormat, ModelFormatVersion, StoreId, Value, WorldMetadataSectionV1,
    WorldMetadataSectionV2,
};
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_engine::{Config, GrafeoDB, GraphModel};
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

const INDEX_LABEL: &str = "Doc";
const INDEX_PROPERTY: &str = "embedding";
const CURRENT_CATALOG_VERSION: u8 = 7;
const CURRENT_LPG_VERSION: u8 = 4;
const CURRENT_VECTOR_VERSION: u8 = 4;

#[derive(Clone, Debug)]
struct RawSection {
    section_type: SectionType,
    version: u8,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct RawImage {
    epoch: u64,
    transaction_id: u64,
    node_count: u64,
    edge_count: u64,
    sections: Vec<RawSection>,
}

impl RawImage {
    fn section(&self, section_type: SectionType) -> &RawSection {
        self.sections
            .iter()
            .find(|section| section.section_type == section_type)
            .unwrap_or_else(|| panic!("missing {section_type:?} section"))
    }

    fn section_mut(&mut self, section_type: SectionType) -> &mut RawSection {
        self.sections
            .iter_mut()
            .find(|section| section.section_type == section_type)
            .unwrap_or_else(|| panic!("missing {section_type:?} section"))
    }

    fn remove(&mut self, section_type: SectionType) {
        let before = self.sections.len();
        self.sections
            .retain(|section| section.section_type != section_type);
        assert_eq!(
            self.sections.len() + 1,
            before,
            "expected exactly one {section_type:?} section"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum MetadataSeal {
    Absent,
    V1,
    CurrentV2,
}

impl MetadataSeal {
    fn rejection_diagnostic(self) -> &'static str {
        match self {
            Self::Absent => "requires WorldMetadata",
            Self::V1 => "requires WorldMetadata v2",
            Self::CurrentV2 => "recovery image digest mismatch",
        }
    }
}

fn read_image(path: &Path) -> RawImage {
    let manager = GrafeoFileManager::open_read_only(path).expect("open raw container image");
    let header = manager.active_header();
    let directory = manager
        .read_section_directory()
        .expect("read raw section directory")
        .expect("active raw section directory");
    let sections = directory
        .entries()
        .iter()
        .map(|entry| RawSection {
            section_type: entry.section_type,
            version: entry.version,
            bytes: manager
                .read_section_data(entry)
                .expect("read CRC-valid section bytes"),
        })
        .collect();
    manager.close().expect("close raw container image");

    RawImage {
        epoch: header.epoch,
        transaction_id: header.transaction_id,
        node_count: header.node_count,
        edge_count: header.edge_count,
        sections,
    }
}

fn publish_image(path: &Path, image: &RawImage) {
    let manager = GrafeoFileManager::open(path).expect("open hostile image for fixture write");
    let sections: Vec<_> = image
        .sections
        .iter()
        .map(|section| SectionWrite::new(section.section_type, section.version, &section.bytes))
        .collect();
    manager
        .write_versioned_sections(
            &sections,
            image.epoch,
            image.transaction_id,
            image.node_count,
            image.edge_count,
        )
        .expect("write hostile image with fresh directory checksums");
    manager.close().expect("close hostile fixture image");
}

fn save_empty_three_dimensional_index(path: &Path) -> StoreId {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))
        .expect("create current three-dimensional source world");
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some(INDEX_LABEL.into()),
        property: INDEX_PROPERTY.into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(3),
            metric: Some("cosine".into()),
            m: Some(8),
            ef_construction: Some(64),
            ef: None,
            quantization: None,
        },
    })
    .expect("create empty three-dimensional descriptor");
    let store_id = db.store_id();
    db.save(path)
        .expect("save current three-dimensional source world");
    db.close()
        .expect("close current three-dimensional source world");
    store_id
}

fn save_foreign_two_dimensional_index(path: &Path) -> RawSection {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))
        .expect("create foreign two-dimensional world");
    let foreign = db.create_node_with_props(
        &[INDEX_LABEL],
        [(INDEX_PROPERTY, Value::Vector(vec![1.0, 0.0].into()))],
    );
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: Some(INDEX_LABEL.into()),
        property: INDEX_PROPERTY.into(),
        kind: grafeo_engine::IndexCreateKind::Vector {
            dimensions: Some(2),
            metric: Some("cosine".into()),
            m: Some(8),
            ef_construction: Some(64),
            ef: None,
            quantization: None,
        },
    })
    .expect("create foreign two-dimensional index");
    db.save(path)
        .expect("save valid foreign two-dimensional world");
    db.close()
        .expect("close valid foreign two-dimensional world");

    let reopened =
        GrafeoDB::open_read_only(path).expect("verify foreign world is independently valid");
    assert_eq!(
        reopened
            .vector_search(INDEX_LABEL, INDEX_PROPERTY, &[1.0, 0.0], 1, None, None,)
            .expect("search valid foreign two-dimensional index")[0]
            .0,
        foreign
    );
    reopened.close().expect("close verified foreign world");

    let image = read_image(path);
    let section = image.section(SectionType::VectorStore).clone();
    assert_eq!(section.version, CURRENT_VECTOR_VERSION);
    section
}

fn authoritative_components(sections: &[RawSection]) -> Vec<(ModelFormatVersion, &[u8])> {
    let mut components = Vec::new();
    for section in sections {
        let format = match section.section_type {
            SectionType::Catalog => Some(AuthoritativeFormat::Catalog),
            SectionType::Cdc => Some(AuthoritativeFormat::Cdc),
            SectionType::LpgStore => Some(AuthoritativeFormat::Lpg),
            SectionType::CompactStore => Some(AuthoritativeFormat::Compact),
            SectionType::OverlayDeletions => Some(AuthoritativeFormat::OverlayDeletions),
            SectionType::WorldMetadata
            | SectionType::VectorStore
            | SectionType::TextIndex
            | SectionType::RdfStore
            | SectionType::RdfRing
            | SectionType::PropertyIndex => None,
            other => panic!("unexpected section in LPG rejection fixture: {other:?}"),
        };
        if let Some(format) = format {
            components.push((
                ModelFormatVersion::new(format, u16::from(section.version))
                    .expect("valid authoritative format version"),
                section.bytes.as_slice(),
            ));
        }
    }
    components
}

fn make_unsealed_foreign_image(
    image: &mut RawImage,
    foreign_vector: &RawSection,
    metadata: MetadataSeal,
) {
    let current_metadata = {
        let section = image.section(SectionType::WorldMetadata);
        assert_eq!(section.version, WorldMetadataSectionV2::SECTION_VERSION);
        WorldMetadataSectionV2::decode(&section.bytes).expect("decode current source WorldMetadata")
    };

    let catalog = image.section(SectionType::Catalog);
    assert_eq!(catalog.version, CURRENT_CATALOG_VERSION);
    assert_eq!(catalog.bytes.first(), Some(&CURRENT_CATALOG_VERSION));

    let lpg = image.section(SectionType::LpgStore);
    assert_eq!(lpg.version, CURRENT_LPG_VERSION);
    assert!(lpg.bytes.starts_with(b"LPG4"));
    // Fresh images must use the current LPG4 envelope wire version.
    assert_eq!(lpg.bytes.get(4), Some(&3));

    assert_eq!(foreign_vector.version, CURRENT_VECTOR_VERSION);
    let vector = image.section_mut(SectionType::VectorStore);
    assert_eq!(vector.version, CURRENT_VECTOR_VERSION);
    assert_ne!(
        vector.bytes, foreign_vector.bytes,
        "the hostile VectorStore must come from a distinct valid world"
    );
    vector.bytes.clone_from(&foreign_vector.bytes);

    // Auxiliary bytes are outside the authoritative cut. The unchanged
    // authoritative seal must still verify after the foreign-vector swap.
    assert!(
        current_metadata
            .cut()
            .verify_components(&authoritative_components(&image.sections))
            .is_ok()
    );

    match metadata {
        MetadataSeal::Absent => image.remove(SectionType::WorldMetadata),
        MetadataSeal::V1 => {
            let encoded = {
                let components = authoritative_components(&image.sections);
                let descriptor = current_metadata.cut().descriptor().clone();
                WorldMetadataSectionV1::seal(descriptor, &components)
                    .expect("seal current authoritative components")
                    .encode()
                    .expect("encode WorldMetadata v1")
            };
            let world = image.section_mut(SectionType::WorldMetadata);
            world.version = WorldMetadataSectionV1::SECTION_VERSION;
            world.bytes = encoded;
        }
        MetadataSeal::CurrentV2 => {
            // Keep the authentic source seal: it must reject the foreign
            // auxiliary image even though all authoritative bytes match.
            assert_eq!(
                image.section(SectionType::WorldMetadata).version,
                WorldMetadataSectionV2::SECTION_VERSION
            );
        }
    }
}

fn assert_rejected_without_rewrite(path: &Path, metadata: MetadataSeal) {
    let before = std::fs::read(path).expect("read hostile image before rejection");
    for read_only in [true, false] {
        let result = if read_only {
            GrafeoDB::open_read_only(path)
        } else {
            GrafeoDB::open(path)
        };
        let error = result
            .err()
            .expect("unsealed Vector image must be rejected");
        assert!(
            matches!(error, Error::Storage(StorageError::Corruption(_))),
            "{error}"
        );
        assert!(
            error.to_string().contains(metadata.rejection_diagnostic()),
            "{error}"
        );
        assert_eq!(
            std::fs::read(path).expect("read hostile image after rejection"),
            before,
            "rejected read_only={read_only} recovery must not rewrite the source"
        );
    }
}

fn qualify_unsealed_rejection(metadata: MetadataSeal) {
    let directory = tempfile::tempdir().expect("temporary rejection directory");
    let source = directory.path().join("three-dimensional-current.grafeo");
    let foreign = directory.path().join("two-dimensional-foreign.grafeo");
    let target = directory.path().join("foreign-vector.grafeo");

    let source_store_id = save_empty_three_dimensional_index(&source);
    let foreign_vector = save_foreign_two_dimensional_index(&foreign);
    std::fs::copy(&source, &target).expect("copy source before metadata rewrite");

    let mut image = read_image(&target);
    make_unsealed_foreign_image(&mut image, &foreign_vector, metadata);
    publish_image(&target, &image);

    let candidate = read_image(&target);
    assert_eq!(
        candidate.section(SectionType::Catalog).version,
        CURRENT_CATALOG_VERSION
    );
    assert_eq!(
        candidate.section(SectionType::LpgStore).version,
        CURRENT_LPG_VERSION
    );
    assert_eq!(
        candidate.section(SectionType::VectorStore).bytes,
        foreign_vector.bytes
    );
    match metadata {
        MetadataSeal::Absent => assert!(
            candidate
                .sections
                .iter()
                .all(|section| section.section_type != SectionType::WorldMetadata)
        ),
        MetadataSeal::V1 => {
            let world = candidate.section(SectionType::WorldMetadata);
            assert_eq!(world.version, WorldMetadataSectionV1::SECTION_VERSION);
            let metadata = WorldMetadataSectionV1::decode(&world.bytes)
                .expect("decode WorldMetadata v1 over current authoritative bytes");
            assert_eq!(metadata.cut().store_id(), source_store_id);
        }
        MetadataSeal::CurrentV2 => assert_eq!(
            candidate.section(SectionType::WorldMetadata).version,
            WorldMetadataSectionV2::SECTION_VERSION
        ),
    }

    assert_rejected_without_rewrite(&target, metadata);
}

#[test]
fn absent_metadata_rejects_foreign_vector_v4_without_rewrite() {
    qualify_unsealed_rejection(MetadataSeal::Absent);
}

#[test]
fn world_metadata_v1_rejects_foreign_vector_v4_without_rewrite() {
    qualify_unsealed_rejection(MetadataSeal::V1);
}

#[test]
fn current_world_metadata_rejects_foreign_vector_v4_without_rewrite() {
    qualify_unsealed_rejection(MetadataSeal::CurrentV2);
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_current_lpg_rejects_foreign_vector_v4_without_rewrite() {
    let directory = tempfile::tempdir().expect("temporary compact rejection directory");
    let source = directory
        .path()
        .join("compact-three-dimensional-current.grafeo");
    let foreign = directory
        .path()
        .join("compact-two-dimensional-foreign.grafeo");
    let target = directory.path().join("compact-foreign-vector.grafeo");

    {
        let mut db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))
            .expect("create compact source world");
        db.create_node_with_props(
            &[INDEX_LABEL],
            [
                (INDEX_PROPERTY, Value::Vector(vec![1.0, 0.0, 0.0].into())),
                ("name", Value::from("nearest")),
            ],
        );
        db.create_node_with_props(
            &[INDEX_LABEL],
            [
                (INDEX_PROPERTY, Value::Vector(vec![0.0, 1.0, 0.0].into())),
                ("name", Value::from("other")),
            ],
        );
        db.compact()
            .expect("move authoritative vector properties into compact base");
        assert_eq!(
            grafeo_engine::database::testing::root_lpg_store(&db).node_count(),
            0,
            "the mutable overlay must be empty after compact"
        );
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some(INDEX_LABEL.into()),
            property: INDEX_PROPERTY.into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create index over the tier-merged compact graph");
        db.save(&source).expect("save current compact source world");
        db.close().expect("close current compact source world");
    }
    let foreign_vector = save_foreign_two_dimensional_index(&foreign);
    std::fs::copy(&source, &target).expect("copy compact source before metadata rewrite");

    let image = read_image(&target);
    assert!(
        image
            .sections
            .iter()
            .any(|section| section.section_type == SectionType::CompactStore),
        "the source must place authoritative vector properties in a compact base"
    );
    for metadata in [MetadataSeal::V1, MetadataSeal::CurrentV2] {
        let mut candidate = image.clone();
        make_unsealed_foreign_image(&mut candidate, &foreign_vector, metadata);
        publish_image(&target, &candidate);

        assert_rejected_without_rewrite(&target, metadata);
    }
}
