//! Hostile qualification for portable identity and integrity-sealed container cuts.

#![cfg(all(
    feature = "grafeo-file",
    feature = "lpg",
    feature = "triple-store",
    feature = "wal"
))]

use std::path::Path;

use grafeo_common::storage::SectionType;
use grafeo_common::types::{
    AuthoritativeFormat, ModelFormatVersion, RecoveryImageComponent, RecoveryImageCoordinatesV1,
    StoreId, WorldCut, WorldCutDescriptor, WorldMetadataSectionV2,
};
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_engine::{Config, GrafeoDB, GraphModel};
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

struct RawSection {
    section_type: SectionType,
    version: u8,
    bytes: Vec<u8>,
}

fn persistent(path: &Path, model: GraphModel) -> Config {
    Config::persistent(path).with_graph_model(model)
}

fn create_lpg(path: &Path) -> StoreId {
    let db = GrafeoDB::with_config(persistent(path, GraphModel::Lpg)).unwrap();
    db.session().execute("INSERT (:World {id: 1})").unwrap();
    let store_id = db.store_id();
    db.close().unwrap();
    store_id
}

fn read_raw(path: &Path) -> (u64, u64, u64, u64, Vec<RawSection>) {
    let manager = GrafeoFileManager::open_read_only(path).unwrap();
    let header = manager.active_header();
    let directory = manager
        .read_section_directory()
        .unwrap()
        .expect("section directory");
    let sections = directory
        .entries()
        .iter()
        .map(|entry| RawSection {
            section_type: entry.section_type,
            version: entry.version,
            bytes: manager.read_section_data(entry).unwrap(),
        })
        .collect();
    manager.close().unwrap();
    (
        header.epoch,
        header.transaction_id,
        header.node_count,
        header.edge_count,
        sections,
    )
}

fn rewrite(path: &Path, coordinates: (u64, u64, u64, u64), sections: &[RawSection]) {
    let manager = GrafeoFileManager::open(path).unwrap();
    let writes: Vec<_> = sections
        .iter()
        .map(|section| SectionWrite::new(section.section_type, section.version, &section.bytes))
        .collect();
    manager
        .write_versioned_sections(
            &writes,
            coordinates.0,
            coordinates.1,
            coordinates.2,
            coordinates.3,
        )
        .unwrap();
    manager.close().unwrap();
}

fn reseal_world_metadata(sections: &mut [RawSection], coordinates: (u64, u64, u64, u64)) {
    let metadata_index = sections
        .iter()
        .position(|section| section.section_type == SectionType::WorldMetadata)
        .expect("WorldMetadata section");
    assert_eq!(
        sections[metadata_index].version,
        WorldMetadataSectionV2::SECTION_VERSION
    );
    let prior = WorldMetadataSectionV2::decode(&sections[metadata_index].bytes).unwrap();
    let encoded = {
        let mut components = Vec::new();
        for section in sections.iter() {
            let version = u16::from(section.version);
            let mut push = |format| {
                components.push((
                    ModelFormatVersion::new(format, version).unwrap(),
                    section.bytes.as_slice(),
                ));
            };
            match section.section_type {
                SectionType::Catalog => push(AuthoritativeFormat::Catalog),
                SectionType::Cdc => push(AuthoritativeFormat::Cdc),
                SectionType::LpgStore => push(AuthoritativeFormat::Lpg),
                SectionType::RdfStore => {
                    push(AuthoritativeFormat::Rdf);
                    let (history_version, history_bytes) =
                        grafeo_core::graph::rdf::section::canonical_history_component(
                            section.bytes.as_slice(),
                        )
                        .unwrap();
                    components.push((
                        ModelFormatVersion::new(AuthoritativeFormat::RdfHistory, history_version)
                            .unwrap(),
                        history_bytes,
                    ));
                }
                SectionType::CompactStore => push(AuthoritativeFormat::Compact),
                SectionType::OverlayDeletions => push(AuthoritativeFormat::OverlayDeletions),
                SectionType::WorldMetadata
                | SectionType::VectorStore
                | SectionType::TextIndex
                | SectionType::RdfRing
                | SectionType::PropertyIndex => {}
                other => panic!("unexpected section in test image: {other:?}"),
            }
        }
        let descriptor = prior.cut().descriptor();
        let descriptor = WorldCutDescriptor::new(
            descriptor.store_id(),
            descriptor.epoch(),
            descriptor.graph_model(),
            components.iter().map(|(format, _)| *format).collect(),
            descriptor.schema().clone(),
            descriptor.projections().to_vec(),
            descriptor.history(),
        )
        .unwrap();
        let cut = WorldCut::seal_components(descriptor, &components).unwrap();
        let mut recovery: Vec<_> = sections
            .iter()
            .filter(|section| section.section_type != SectionType::WorldMetadata)
            .map(|section| {
                RecoveryImageComponent::new(
                    section.section_type as u32,
                    u16::from(section.version),
                    &section.bytes,
                )
                .unwrap()
            })
            .collect();
        let recovery_coordinates = RecoveryImageCoordinatesV1::new(
            coordinates.0,
            coordinates.1,
            cut.descriptor().graph_model(),
            coordinates.2,
            coordinates.3,
        );
        recovery.push(recovery_coordinates.component());
        WorldMetadataSectionV2::seal(cut, &recovery)
            .unwrap()
            .encode()
            .unwrap()
    };
    sections[metadata_index].bytes = encoded;
}

fn assert_container_corruption(path: &Path) {
    let error = match GrafeoDB::with_config(Config::persistent(path)) {
        Ok(db) => {
            let _ = db.close();
            panic!("corrupt container unexpectedly opened")
        }
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::Storage(StorageError::Corruption(_))),
        "expected structured container corruption, got {error:?}"
    );
}

#[test]
fn lpg_container_persists_identity_and_exact_world_sections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lpg-world.grafeo");
    let expected = create_lpg(&path);

    let (_, _, _, _, sections) = read_raw(&path);
    assert!(
        sections.iter().any(|section| {
            section.section_type == SectionType::Catalog && section.version == 7
        })
    );
    assert!(
        sections.iter().any(|section| {
            section.section_type == SectionType::LpgStore && section.version == 4
        })
    );
    assert!(sections.iter().any(|section| {
        section.section_type == SectionType::WorldMetadata
            && section.version == WorldMetadataSectionV2::SECTION_VERSION
    }));
    assert!(
        !sections
            .iter()
            .any(|section| section.section_type == SectionType::RdfStore)
    );

    let reopened = GrafeoDB::with_config(persistent(&path, GraphModel::Lpg)).unwrap();
    assert_eq!(reopened.store_id(), expected);
    assert_eq!(reopened.node_count(), 1);
    reopened.close().unwrap();
}

#[test]
fn empty_rdf_container_persists_its_handle_namespace_without_fake_lpg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty-rdf-world.grafeo");
    let expected = {
        let db = GrafeoDB::with_config(persistent(&path, GraphModel::Rdf)).unwrap();
        let store_id = db.store_id();
        db.close().unwrap();
        store_id
    };

    let (_, _, _, _, sections) = read_raw(&path);
    assert!(
        sections.iter().any(|section| {
            section.section_type == SectionType::Catalog && section.version == 2
        })
    );
    assert!(
        sections.iter().any(|section| {
            section.section_type == SectionType::RdfStore && section.version == 6
        })
    );
    assert!(
        sections
            .iter()
            .any(|section| section.section_type == SectionType::WorldMetadata)
    );
    assert!(
        !sections
            .iter()
            .any(|section| section.section_type == SectionType::LpgStore)
    );

    let reopened = GrafeoDB::with_config(persistent(&path, GraphModel::Rdf)).unwrap();
    assert_eq!(reopened.store_id(), expected);
    reopened.close().unwrap();
}

#[test]
fn copied_container_preserves_logical_store_identity() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.grafeo");
    let copy = dir.path().join("copy.grafeo");
    let expected = create_lpg(&source);
    std::fs::copy(&source, &copy).unwrap();

    let reopened = GrafeoDB::with_config(persistent(&copy, GraphModel::Lpg)).unwrap();
    assert_eq!(reopened.store_id(), expected);
    assert_eq!(reopened.node_count(), 1);
    reopened.close().unwrap();
}

#[test]
fn uncompacted_save_publishes_a_complete_versioned_world_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("saved.grafeo");
    let source =
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    source.session().execute("INSERT (:World {id: 1})").unwrap();
    let expected = source.store_id();

    source.save(&path).unwrap();

    let (_, _, nodes, edges, sections) = read_raw(&path);
    assert_eq!((nodes, edges), (1, 0));
    assert!(
        sections.iter().any(|section| {
            section.section_type == SectionType::Catalog && section.version == 7
        })
    );
    assert!(
        sections.iter().any(|section| {
            section.section_type == SectionType::LpgStore && section.version == 4
        })
    );
    assert!(sections.iter().any(|section| {
        section.section_type == SectionType::WorldMetadata
            && section.version == WorldMetadataSectionV2::SECTION_VERSION
    }));
    let reopened = GrafeoDB::with_config(persistent(&path, GraphModel::Lpg)).unwrap();
    assert_eq!(reopened.store_id(), expected);
    assert_eq!(reopened.node_count(), 1);
    reopened.close().unwrap();
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_checkpoint_header_counts_the_complete_tier_merged_graph() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compact-header.grafeo");
    let mut db = GrafeoDB::with_config(persistent(&path, GraphModel::Lpg)).unwrap();
    db.session()
        .execute("INSERT (:A)-[:LINKS_TO]->(:B)")
        .unwrap();
    db.compact().unwrap();

    db.wal_checkpoint().unwrap();
    // Inspect this explicit checkpoint while the writer retains its lock.
    // Closing first could checkpoint again and mask incorrect header counts.
    let checkpoint_image = dir.path().join("checkpoint-image.grafeo");
    std::fs::copy(&path, &checkpoint_image).unwrap();
    let (_, _, nodes, edges, sections) = read_raw(&checkpoint_image);
    assert_eq!((nodes, edges), (2, 1));
    assert!(
        sections
            .iter()
            .any(|section| section.section_type == SectionType::CompactStore)
    );
    db.close().unwrap();

    let reopened = GrafeoDB::with_config(persistent(&path, GraphModel::Lpg)).unwrap();
    assert_eq!((reopened.node_count(), reopened.edge_count()), (2, 1));
    reopened.close().unwrap();
}

#[test]
fn authoritative_tamper_with_fresh_crc_fails_manifest_verification() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tampered.grafeo");
    create_lpg(&path);
    let (epoch, transaction_id, nodes, edges, mut sections) = read_raw(&path);
    let lpg = sections
        .iter_mut()
        .find(|section| section.section_type == SectionType::LpgStore)
        .unwrap();
    let last = lpg.bytes.last_mut().expect("non-empty LPG section");
    *last ^= 0x80;
    rewrite(&path, (epoch, transaction_id, nodes, edges), &sections);

    assert_container_corruption(&path);
}

#[test]
fn header_epoch_and_model_cannot_disagree_with_world_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("coordinates.grafeo");
    create_lpg(&source);
    let (epoch, transaction_id, nodes, edges, sections) = read_raw(&source);

    rewrite(
        &source,
        (epoch + 1, transaction_id, nodes, edges),
        &sections,
    );
    assert_container_corruption(&source);

    let wrong_model = dir.path().join("wrong-model.grafeo");
    let manager =
        GrafeoFileManager::create_with_graph_model(&wrong_model, GraphModel::Both.as_u8()).unwrap();
    let writes: Vec<_> = sections
        .iter()
        .map(|section| SectionWrite::new(section.section_type, section.version, &section.bytes))
        .collect();
    manager
        .write_versioned_sections(&writes, epoch, transaction_id, nodes, edges)
        .unwrap();
    manager.close().unwrap();
    assert_container_corruption(&wrong_model);
}

#[test]
fn self_consistent_manifest_cannot_claim_an_unsupported_section_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsupported-version.grafeo");
    create_lpg(&path);
    let (epoch, transaction_id, nodes, edges, mut sections) = read_raw(&path);
    sections
        .iter_mut()
        .find(|section| section.section_type == SectionType::Catalog)
        .unwrap()
        .version = 99;
    reseal_world_metadata(&mut sections, (epoch, transaction_id, nodes, edges));
    rewrite(&path, (epoch, transaction_id, nodes, edges), &sections);

    assert_container_corruption(&path);
}

#[test]
fn exact_container_cannot_be_downgraded_by_removing_world_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.grafeo");
    create_lpg(&path);
    let (epoch, transaction_id, nodes, edges, mut sections) = read_raw(&path);
    sections.retain(|section| section.section_type != SectionType::WorldMetadata);
    rewrite(&path, (epoch, transaction_id, nodes, edges), &sections);

    assert_container_corruption(&path);
}
