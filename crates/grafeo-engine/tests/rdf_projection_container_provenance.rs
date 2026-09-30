//! Hostile current-container qualification for RDF→LPG row provenance.

#![cfg(all(
    feature = "grafeo-file",
    feature = "lpg",
    feature = "sparql",
    feature = "triple-store",
    feature = "wal"
))]

use std::path::Path;
use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::storage::{Section, SectionType};
use grafeo_common::types::{
    AuthoritativeFormat, EdgeId, EpochId, GraphModelTag, ModelFormatVersion, NodeId,
    RecoveryImageComponent, RecoveryImageCoordinatesV1, Value, WorldCut, WorldMetadataSectionV2,
};
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection};
use grafeo_core::graph::rdf::{RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY};
use grafeo_engine::{Config, GrafeoDB, GraphModel};
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

const PERSON: &str = "http://ex.org/Person";
const ALIX: &str = "http://ex.org/alix";
const FORGED: &str = "http://ex.org/forged";

struct RawSection {
    section_type: SectionType,
    version: u8,
    bytes: Vec<u8>,
}

fn both_without_wal(path: &Path) -> Config {
    let mut config = Config::persistent(path).with_graph_model(GraphModel::Both);
    config.wal_enabled = false;
    config
}

fn create_projection_container(path: &Path) -> String {
    let db = GrafeoDB::with_config(both_without_wal(path)).unwrap();
    db.execute_sparql(&format!("INSERT DATA {{ <{ALIX}> a <{PERSON}> . }}"))
        .unwrap();
    let projection_id = db.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    assert_eq!(db.rebuild_rdf_lpg_projection(projection_id).unwrap(), 1);
    let owner = db.rdf_lpg_projection(projection_id).unwrap().owner_marker();
    db.close().unwrap();
    owner
}

fn read_raw(path: &Path) -> (u64, u64, Vec<RawSection>) {
    let manager = GrafeoFileManager::open_read_only(path).unwrap();
    let header = manager.active_header();
    let directory = manager
        .read_section_directory()
        .unwrap()
        .expect("current section directory");
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
    (header.epoch, header.transaction_id, sections)
}

fn authoritative_components(sections: &[RawSection]) -> Vec<(ModelFormatVersion, &[u8])> {
    let mut components = Vec::new();
    for section in sections {
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
            SectionType::OverlayDeletions => {
                push(AuthoritativeFormat::OverlayDeletions);
            }
            SectionType::WorldMetadata
            | SectionType::VectorStore
            | SectionType::TextIndex
            | SectionType::RdfRing
            | SectionType::PropertyIndex => {}
            other => panic!("unexpected section in hostile fixture: {other:?}"),
        }
    }
    components
}

fn forge_lpg_section(path: &Path, forged: &Arc<LpgStore>, node_count: u64, edge_count: u64) {
    let (epoch, transaction_id, mut sections) = read_raw(path);
    let cut = EpochId::new(epoch);
    assert!(
        forged.current_epoch() <= cut,
        "synthetic history must fit the actual container cut"
    );
    // Advance the capture frontier only; retain all deliberately malformed
    // provenance histories. Do not change the authentic RDF/catalog epochs.
    forged.sync_epoch(cut);
    let section = LpgStoreSection::new(Arc::clone(forged));
    let forged_lpg = section.serialize().unwrap();
    assert_eq!(section.version(), 4);
    assert!(forged_lpg.starts_with(b"LPG4"));
    assert_eq!(forged_lpg.get(4), Some(&3));
    let expected_checksum = u32::from_le_bytes(forged_lpg[16..20].try_into().unwrap());
    assert_eq!(crc32fast::hash(&forged_lpg[24..]), expected_checksum);
    let metadata_index = sections
        .iter()
        .position(|section| section.section_type == SectionType::WorldMetadata)
        .expect("WorldMetadata section");
    assert_eq!(
        sections[metadata_index].version,
        WorldMetadataSectionV2::SECTION_VERSION
    );
    let descriptor = WorldMetadataSectionV2::decode(&sections[metadata_index].bytes)
        .unwrap()
        .cut()
        .descriptor()
        .clone();
    assert_eq!(descriptor.projections().len(), 1);
    assert_eq!(descriptor.epoch(), cut);

    let lpg = sections
        .iter_mut()
        .find(|section| section.section_type == SectionType::LpgStore)
        .expect("LPG section");
    lpg.version = section.version();
    lpg.bytes = forged_lpg;

    let resealed = {
        let components = authoritative_components(&sections);
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
        let coordinates = RecoveryImageCoordinatesV1::new(
            epoch,
            transaction_id,
            GraphModelTag::Both,
            node_count,
            edge_count,
        );
        recovery.push(coordinates.component());
        WorldMetadataSectionV2::seal(cut, &recovery)
            .unwrap()
            .encode()
            .unwrap()
    };
    sections[metadata_index].bytes = resealed;

    let manager = GrafeoFileManager::open(path).unwrap();
    let writes: Vec<_> = sections
        .iter()
        .map(|section| SectionWrite::new(section.section_type, section.version, &section.bytes))
        .collect();
    manager
        .write_versioned_sections(&writes, epoch, transaction_id, node_count, edge_count)
        .unwrap();
    manager.close().unwrap();
}

fn standalone_store(path: &Path) -> Arc<LpgStore> {
    let store = Arc::new(LpgStore::new().unwrap());
    store.sync_epoch(EpochId::new(read_raw(path).0));
    store
}

fn insert_projection_row(store: &LpgStore, node: NodeId, owner: &str, iri: &str) {
    let row_epoch = store.current_epoch();
    store.create_node_with_id(node, &["Person"]).unwrap();
    store.set_node_property_at_epoch(
        node,
        RDF_LPG_PROJECTION_OWNER_PROPERTY,
        Value::from(owner),
        row_epoch,
    );
    store.set_node_property_at_epoch(
        node,
        RDF_LPG_PROJECTION_IRI_PROPERTY,
        Value::from(iri),
        row_epoch,
    );
}

fn restore_projection_row_exact(
    store: &LpgStore,
    node: NodeId,
    owner: &str,
    iri: &str,
    created: EpochId,
    marker_epoch: EpochId,
) {
    store
        .restore_node_history_exact(
            node,
            &[(created, None)],
            &[(created, vec![ArcStr::from("Person")])],
        )
        .unwrap();
    store.set_node_property_at_epoch(
        node,
        RDF_LPG_PROJECTION_OWNER_PROPERTY,
        Value::from(owner),
        marker_epoch,
    );
    store.set_node_property_at_epoch(
        node,
        RDF_LPG_PROJECTION_IRI_PROPERTY,
        Value::from(iri),
        marker_epoch,
    );
    store.sync_epoch(created.max(marker_epoch));
}

fn assert_projection_corruption(path: &Path, expected: &str) {
    let error = match GrafeoDB::with_config(both_without_wal(path)) {
        Ok(db) => {
            let _ = db.close();
            panic!("forged projection container unexpectedly opened")
        }
        Err(error) => error,
    };
    assert!(
        matches!(
            error,
            Error::Storage(StorageError::Corruption(_)) | Error::Serialization(_)
        ),
        "expected structured projection/container corruption, got {error:?}"
    );
    assert!(
        error.to_string().contains(expected),
        "diagnostic must reach {expected:?}, got: {error}"
    );
}

#[test]
fn valid_current_projection_container_still_opens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid-projection.grafeo");
    create_projection_container(&path);

    let reopened = GrafeoDB::with_config(both_without_wal(&path)).unwrap();
    assert_eq!(reopened.node_count(), 1);
    reopened.close().unwrap();
}

#[test]
fn resealed_canonical_projection_rows_still_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resealed-valid-projection.grafeo");
    let owner = create_projection_container(&path);
    let replacement = standalone_store(&path);
    insert_projection_row(&replacement, NodeId::new(1), &owner, ALIX);
    forge_lpg_section(&path, &replacement, 1, 0);

    let reopened = GrafeoDB::with_config(both_without_wal(&path)).unwrap();
    assert_eq!(reopened.node_count(), 1);
    let row = grafeo_engine::database::testing::root_lpg_store(&reopened)
        .get_node(NodeId::new(1))
        .unwrap();
    assert_eq!(
        row.get_property(RDF_LPG_PROJECTION_IRI_PROPERTY),
        Some(&Value::from(ALIX))
    );
    reopened.close().unwrap();
}

#[test]
fn resealed_container_cannot_claim_a_receipt_without_its_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing-rows.grafeo");
    create_projection_container(&path);
    forge_lpg_section(&path, &standalone_store(&path), 0, 0);

    assert_projection_corruption(&path, "status records 1 rows but the captured target has 0");
}

#[test]
fn resealed_container_rejects_missing_and_unknown_owner_markers() {
    for mode in ["missing", "unknown"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("{mode}-owner.grafeo"));
        let owner = create_projection_container(&path);
        let forged = standalone_store(&path);
        insert_projection_row(&forged, NodeId::new(1), &owner, ALIX);
        forged
            .create_node_with_id(NodeId::new(2), &["Person"])
            .unwrap();
        forged.set_node_property_at_epoch(
            NodeId::new(2),
            RDF_LPG_PROJECTION_IRI_PROPERTY,
            Value::from(FORGED),
            forged.current_epoch(),
        );
        if mode == "unknown" {
            forged.set_node_property_at_epoch(
                NodeId::new(2),
                RDF_LPG_PROJECTION_OWNER_PROPERTY,
                Value::from("not-a-known-projection"),
                forged.current_epoch(),
            );
        }
        forge_lpg_section(&path, &forged, 2, 0);

        assert_projection_corruption(
            &path,
            if mode == "missing" {
                "reserved source IRI has no ownership marker"
            } else {
                "references unknown owner"
            },
        );
    }
}

#[test]
fn resealed_container_rejects_edges_incident_to_projection_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("incident-edge.grafeo");
    let owner = create_projection_container(&path);
    let forged = standalone_store(&path);
    insert_projection_row(&forged, NodeId::new(1), &owner, ALIX);
    forged
        .create_node_with_id(NodeId::new(2), &["Ordinary"])
        .unwrap();
    forged
        .create_edge_with_id(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "FORGED_EDGE",
        )
        .unwrap();
    forge_lpg_section(&path, &forged, 2, 1);

    assert_projection_corruption(&path, "is incident to projection-owned node history");
}

#[test]
fn resealed_lpg4_rejects_incomplete_closed_projection_metadata_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed-projection-history.grafeo");
    let owner = create_projection_container(&path);
    let forged = Arc::new(LpgStore::new().unwrap());
    restore_projection_row_exact(
        &forged,
        NodeId::new(1),
        &owner,
        ALIX,
        EpochId::INITIAL,
        EpochId::INITIAL,
    );
    forged
        .restore_node_history_exact(
            NodeId::new(2),
            &[(EpochId::INITIAL, Some(EpochId::new(1)))],
            &[(EpochId::INITIAL, vec![ArcStr::from("Person")])],
        )
        .unwrap();
    forged.set_node_property_at_epoch(
        NodeId::new(2),
        RDF_LPG_PROJECTION_OWNER_PROPERTY,
        Value::from(owner),
        EpochId::INITIAL,
    );
    forged.sync_epoch(EpochId::new(1));
    assert!(forged.get_node(NodeId::new(2)).is_none());
    forge_lpg_section(&path, &forged, 1, 0);

    assert_projection_corruption(
        &path,
        "properties exist outside the projection metadata plane",
    );
}

#[test]
fn resealed_lpg4_rejects_noncanonical_projection_marker_history_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("projection-marker-history.grafeo");
    let owner = create_projection_container(&path);
    let forged = Arc::new(LpgStore::new().unwrap());
    restore_projection_row_exact(
        &forged,
        NodeId::new(1),
        &owner,
        ALIX,
        EpochId::INITIAL,
        EpochId::new(1),
    );
    forge_lpg_section(&path, &forged, 1, 0);

    assert_projection_corruption(&path, "not structural creation");
}

#[test]
fn resealed_lpg4_rejects_closed_incident_edge_history_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed-incident-edge.grafeo");
    let owner = create_projection_container(&path);
    let forged = Arc::new(LpgStore::new().unwrap());
    restore_projection_row_exact(
        &forged,
        NodeId::new(1),
        &owner,
        ALIX,
        EpochId::INITIAL,
        EpochId::INITIAL,
    );
    forged
        .restore_node_history_exact(
            NodeId::new(2),
            &[(EpochId::INITIAL, None)],
            &[(EpochId::INITIAL, vec![ArcStr::from("Ordinary")])],
        )
        .unwrap();
    forged
        .restore_edge_history_exact(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "HIDDEN_EDGE",
            &[(EpochId::INITIAL, Some(EpochId::new(1)))],
        )
        .unwrap();
    forged.sync_epoch(EpochId::new(1));
    assert!(forged.get_edge(EdgeId::new(1)).is_none());
    forge_lpg_section(&path, &forged, 2, 0);

    assert_projection_corruption(&path, "is incident to projection-owned node history");
}

#[test]
fn resealed_lpg4_rejects_closed_named_graph_marker_history_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed-named-marker-history.grafeo");
    let owner = create_projection_container(&path);
    let forged = Arc::new(LpgStore::new().unwrap());
    restore_projection_row_exact(
        &forged,
        NodeId::new(1),
        &owner,
        ALIX,
        EpochId::INITIAL,
        EpochId::INITIAL,
    );
    forged.create_graph("hidden").unwrap();
    let named = forged.graph("hidden").unwrap();
    named
        .restore_node_history_exact(
            NodeId::new(1),
            &[(EpochId::INITIAL, Some(EpochId::new(1)))],
            &[(EpochId::INITIAL, vec![ArcStr::from("Person")])],
        )
        .unwrap();
    named.set_node_property_at_epoch(
        NodeId::new(1),
        RDF_LPG_PROJECTION_OWNER_PROPERTY,
        Value::from(owner),
        EpochId::INITIAL,
    );
    named.sync_epoch(EpochId::new(1));
    forged.sync_epoch(EpochId::new(1));
    assert!(named.get_node(NodeId::new(1)).is_none());
    forge_lpg_section(&path, &forged, 1, 0);

    assert_projection_corruption(
        &path,
        "projection ownership is confined to the default graph",
    );
}
