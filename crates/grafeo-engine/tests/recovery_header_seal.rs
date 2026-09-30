//! Hostile qualification for recovery-critical container header coordinates.
//!
//! A valid section directory and per-section CRCs do not prove that the active
//! database header belongs to those section bytes. This test republishes the
//! exact image under a forged, high WAL replay floor. WorldMetadata v2 must
//! reject that splice before recovery can rewrite the container.

#![cfg(all(feature = "grafeo-file", feature = "lpg", feature = "wal"))]

use std::path::Path;

use grafeo_common::storage::SectionType;
use grafeo_common::types::{NodeId, WorldMetadataSectionV2};
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_engine::{Config, GrafeoDB, GraphModel};
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

#[derive(Debug, PartialEq, Eq)]
struct RawSection {
    section_type: SectionType,
    version: u8,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct RawImage {
    epoch: u64,
    transaction_id: u64,
    node_count: u64,
    edge_count: u64,
    sections: Vec<RawSection>,
}

fn read_image(path: &Path) -> RawImage {
    let manager = GrafeoFileManager::open_read_only(path).expect("open raw container");
    let header = manager.active_header();
    let directory = manager
        .read_section_directory()
        .expect("read CRC-valid directory")
        .expect("active section directory");
    let sections = directory
        .entries()
        .iter()
        .map(|entry| RawSection {
            section_type: entry.section_type,
            version: entry.version,
            bytes: manager
                .read_section_data(entry)
                .expect("read CRC-valid section"),
        })
        .collect();
    manager.close().expect("close raw container");

    RawImage {
        epoch: header.epoch,
        transaction_id: header.transaction_id,
        node_count: header.node_count,
        edge_count: header.edge_count,
        sections,
    }
}

fn republish_with_transaction_id(path: &Path, image: &RawImage, transaction_id: u64) {
    let manager = GrafeoFileManager::open(path).expect("open hostile rewrite target");
    let writes: Vec<_> = image
        .sections
        .iter()
        .map(|section| SectionWrite::new(section.section_type, section.version, &section.bytes))
        .collect();
    manager
        .write_versioned_sections(
            &writes,
            image.epoch,
            transaction_id,
            image.node_count,
            image.edge_count,
        )
        .expect("publish CRC-valid image under forged WAL floor");
    manager.close().expect("close hostile rewrite target");
}

#[test]
fn v2_rejects_crc_valid_high_same_epoch_wal_floor_before_rewrite() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("forged-recovery-header.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Lpg))
        .expect("create persistent LPG world");
    let node = db.create_node_with_props(&["World"], [("id", 1_i64)]);
    assert_ne!(node, NodeId::INVALID);
    db.close().expect("checkpoint source world");

    let original = read_image(&path);
    assert!(original.transaction_id > 0);
    assert!(original.sections.iter().any(|section| {
        section.section_type == SectionType::WorldMetadata
            && section.version == WorldMetadataSectionV2::SECTION_VERSION
    }));

    let forged_transaction_id = original
        .transaction_id
        .checked_add(1_000_000)
        .expect("fixture transaction id has headroom");
    republish_with_transaction_id(&path, &original, forged_transaction_id);

    let forged = read_image(&path);
    assert_eq!(forged.epoch, original.epoch, "epoch must remain unchanged");
    assert_eq!(forged.transaction_id, forged_transaction_id);
    assert_eq!(forged.node_count, original.node_count);
    assert_eq!(forged.edge_count, original.edge_count);
    assert_eq!(
        forged.sections, original.sections,
        "the hostile publication must preserve every section version and byte"
    );

    let before_failed_open = std::fs::read(&path).expect("read hostile image before recovery");
    let error = match GrafeoDB::open(&path) {
        Ok(db) => {
            let _ = db.close();
            panic!("container with a forged WAL replay floor unexpectedly opened")
        }
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::Storage(StorageError::Corruption(_))),
        "expected structured container corruption, got {error:?}"
    );
    assert!(
        error.to_string().contains("recovery image digest mismatch"),
        "the v2 header-coordinate seal must reject before WAL recovery: {error}"
    );
    assert_eq!(
        std::fs::read(&path).expect("read hostile image after rejected recovery"),
        before_failed_open,
        "failed recovery must not rewrite the rejected container"
    );
}
