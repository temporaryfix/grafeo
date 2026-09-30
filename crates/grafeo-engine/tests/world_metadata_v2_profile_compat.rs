//! Qualification for the retained section-only WorldMetadata v2 profile.
//!
//! Sections authenticate exact current Catalog7/Text5/Vector4 bytes, but not
//! recovery-header coordinates. An existing WAL cannot bind to that profile.
//! The existing empty-tail checkpoint path seals authoritative restored state;
//! it must preserve exact owner/index bytes and reject later header splices.
//! Upstream compatibility for this profile requires separate review.

#![cfg(all(
    feature = "grafeo-file",
    feature = "lpg",
    feature = "wal",
    feature = "vector-index"
))]

use std::path::{Path, PathBuf};

use grafeo_common::storage::{Section, SectionType};
use grafeo_common::types::{
    GraphModelTag, GraphPath, NodeId, RecoveryImageComponent, RecoveryImageCoordinatesV1,
    TransactionId, Value, WorldMetadataSectionV2,
};
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_core::graph::lpg::PhysicalIndexKey;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};
use grafeo_storage::wal::{
    DurabilityMode as WalDurabilityMode, LpgMutationOp, LpgWal, WalConfig, WalRecord,
};

const CURRENT_CATALOG_VERSION: u8 = 7;
const CURRENT_VECTOR_VERSION: u8 = 4;
#[cfg(feature = "text-index")]
const CURRENT_TEXT_VERSION: u8 = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
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
}

fn sidecar_wal_dir(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

fn read_image(path: &Path) -> RawImage {
    let manager = GrafeoFileManager::open_read_only(path).expect("open raw container image");
    assert_eq!(
        manager.graph_model_tag(),
        GraphModelTag::Lpg.as_u8(),
        "the compatibility fixture must remain an LPG container"
    );
    let header = manager.active_header();
    let directory = manager
        .read_section_directory()
        .expect("read CRC-valid section directory")
        .expect("active section directory");
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
    let manager = GrafeoFileManager::open(path).expect("open raw publication target");
    let writes: Vec<_> = image
        .sections
        .iter()
        .map(|section| SectionWrite::new(section.section_type, section.version, &section.bytes))
        .collect();
    manager
        .write_versioned_sections(
            &writes,
            image.epoch,
            image.transaction_id,
            image.node_count,
            image.edge_count,
        )
        .expect("publish image with a fresh CRC-valid directory");
    manager.close().expect("close raw publication target");
}

fn recovery_section_components(image: &RawImage) -> Vec<RecoveryImageComponent<'_>> {
    image
        .sections
        .iter()
        .filter(|section| section.section_type != SectionType::WorldMetadata)
        .map(|section| {
            RecoveryImageComponent::new(
                section.section_type as u32,
                u16::from(section.version),
                &section.bytes,
            )
            .expect("valid recovery-image component")
        })
        .collect()
}

fn metadata(image: &RawImage) -> WorldMetadataSectionV2 {
    let section = image.section(SectionType::WorldMetadata);
    assert_eq!(section.version, WorldMetadataSectionV2::SECTION_VERSION);
    WorldMetadataSectionV2::decode(&section.bytes).expect("decode WorldMetadata v2")
}

fn replace_with_frozen_section_only_profile(image: &mut RawImage) {
    let current = metadata(image);
    let frozen = {
        let components = recovery_section_components(image);
        WorldMetadataSectionV2::seal(current.cut().clone(), &components)
            .expect("seal frozen section-only WorldMetadata v2 profile")
    };
    assert_ne!(
        frozen.recovery_image_digest(),
        current.recovery_image_digest(),
        "removing the synthetic coordinate component must change only the physical seal"
    );
    assert_eq!(
        frozen.cut(),
        current.cut(),
        "the compatibility profile must preserve the exact logical WorldCut"
    );
    image.section_mut(SectionType::WorldMetadata).bytes = frozen
        .encode()
        .expect("encode frozen section-only WorldMetadata v2 profile");
}

fn assert_section_only_profile(image: &RawImage) {
    let metadata = metadata(image);
    let section_components = recovery_section_components(image);
    metadata
        .verify_recovery_components(&section_components)
        .expect("frozen v2 profile must authenticate every physical section");

    let coordinates = RecoveryImageCoordinatesV1::new(
        image.epoch,
        image.transaction_id,
        GraphModelTag::Lpg,
        image.node_count,
        image.edge_count,
    );
    let mut current_components = section_components;
    current_components.push(coordinates.component());
    metadata
        .verify_recovery_components(&current_components)
        .expect_err("frozen v2 profile must not claim to seal header coordinates");
}

fn assert_current_coordinate_profile(image: &RawImage) {
    let metadata = metadata(image);
    let section_components = recovery_section_components(image);
    metadata
        .verify_recovery_components(&section_components)
        .expect_err("the current writer must not emit the frozen section-only profile");

    let coordinates = RecoveryImageCoordinatesV1::new(
        image.epoch,
        image.transaction_id,
        GraphModelTag::Lpg,
        image.node_count,
        image.edge_count,
    );
    let mut current_components = section_components;
    current_components.push(coordinates.component());
    metadata
        .verify_recovery_components(&current_components)
        .expect("current v2 profile must seal the exact active-header coordinates");
}

fn exact_vector_image(db: &GrafeoDB) -> Vec<u8> {
    grafeo_core::index::vector::VectorStoreSection::from_views(
        grafeo_engine::database::testing::root_lpg_store(db)
            .vector_index_entries()
            .into_iter()
            .map(|(_, view)| {
                (
                    PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
                    view,
                )
            })
            .collect(),
    )
    .serialize()
    .expect("serialize exact in-memory vector image")
}

#[cfg(feature = "text-index")]
fn exact_text_image(db: &GrafeoDB) -> Vec<u8> {
    grafeo_core::index::text::TextIndexSection::from_views(
        grafeo_engine::database::testing::root_lpg_store(db)
            .text_index_entries()
            .into_iter()
            .map(|(_, view)| {
                (
                    PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                    view,
                )
            })
            .collect(),
    )
    .serialize()
    .expect("serialize exact in-memory text image")
}

fn assert_world_and_exact_indexes(
    db: &GrafeoDB,
    nearest: NodeId,
    expected_vector: &[u8],
    #[cfg(feature = "text-index")] expected_text: &[u8],
) {
    assert_eq!(db.node_count(), 2, "authoritative node count changed");
    assert_eq!(db.edge_count(), 1, "authoritative edge count changed");
    assert!(
        db.get_node(nearest).is_some(),
        "nearest document disappeared"
    );

    let index = grafeo_engine::database::testing::root_lpg_store(db)
        .get_vector_index("Doc", "embedding")
        .expect("Catalog v7 vector descriptor must be restored");
    assert_eq!(index.config().dimensions, 3);
    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 2, None, None)
        .expect("search exact restored vector topology");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, nearest);
    assert_eq!(
        exact_vector_image(db),
        expected_vector,
        "read-only section verification must retain exact authenticated vector state"
    );

    #[cfg(feature = "text-index")]
    {
        let results = db
            .text_search("Doc", "body", "compatibility", 10)
            .expect("search exact restored text postings");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, nearest);
        assert_eq!(
            exact_text_image(db),
            expected_text,
            "read-only section verification must retain exact authenticated text state"
        );
    }
}

fn append_epochless_legacy_commit(path: &Path, transaction_id: TransactionId) {
    let wal = LpgWal::with_config(
        sidecar_wal_dir(path),
        WalConfig {
            durability: WalDurabilityMode::Sync,
            ..WalConfig::default()
        },
    )
    .expect("create hostile sidecar WAL");
    wal.log(&WalRecord::lpg(
        transaction_id,
        GraphPath::root(),
        LpgMutationOp::CreateNode {
            id: NodeId::new(90_001),
            labels: vec!["MustNotBeSilentlyDiscarded".to_string()],
        },
    ))
    .expect("append legacy LPG mutation");
    wal.log(&WalRecord::TransactionCommit { transaction_id })
        .expect("append epoch-less legacy commit marker");
    wal.sync().expect("sync hostile sidecar WAL");
    wal.close().expect("close hostile sidecar WAL");
}

#[test]
fn section_only_v2_preserves_exact_indexes_ignores_header_coordinates_and_upgrades()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir().expect("temporary compatibility directory");
    let path = directory.path().join("section-only-v2.grafeo");
    let wal_probe = directory.path().join("section-only-v2-wal-probe.grafeo");
    let splice_probe = directory.path().join("coordinate-splice.grafeo");

    let nearest;
    {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))?;
        nearest = db.create_node_with_props(
            &["Doc"],
            [
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into())),
                ("body", Value::from("compatibility profile needle")),
            ],
        );
        let other = db.create_node_with_props(
            &["Doc"],
            [
                ("embedding", Value::Vector(vec![0.0, 1.0, 0.0].into())),
                ("body", Value::from("unrelated document")),
            ],
        );
        assert!(nearest.is_valid() && other.is_valid());
        assert!(db.create_edge(nearest, other, "LINKS").is_valid());
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create exact vector index");
        #[cfg(feature = "text-index")]
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create exact text index");
        db.save(&path)?;
        db.close().expect("close current source image");
    }

    let mut image = read_image(&path);
    assert_eq!(
        image.section(SectionType::Catalog).version,
        CURRENT_CATALOG_VERSION
    );
    assert_eq!(
        image.section(SectionType::VectorStore).version,
        CURRENT_VECTOR_VERSION
    );
    #[cfg(feature = "text-index")]
    assert_eq!(
        image.section(SectionType::TextIndex).version,
        CURRENT_TEXT_VERSION
    );
    assert_current_coordinate_profile(&image);

    let expected_catalog = image.section(SectionType::Catalog).bytes.clone();
    let expected_vector = image.section(SectionType::VectorStore).bytes.clone();
    #[cfg(feature = "text-index")]
    let expected_text = image.section(SectionType::TextIndex).bytes.clone();

    // Reconstruct the exact profile emitted by the first WorldMetadata v2
    // writer: same logical cut, every non-metadata section, no coordinates.
    replace_with_frozen_section_only_profile(&mut image);
    assert_section_only_profile(&image);
    publish_image(&path, &image);

    let same_header = read_image(&path);
    assert_eq!(same_header.transaction_id, image.transaction_id);
    assert_eq!(same_header.node_count, image.node_count);
    assert_eq!(same_header.edge_count, image.edge_count);
    assert_section_only_profile(&same_header);
    let read_only = GrafeoDB::open_read_only(&path).expect("read frozen section-only v2 profile");
    assert_world_and_exact_indexes(
        &read_only,
        nearest,
        &expected_vector,
        #[cfg(feature = "text-index")]
        &expected_text,
    );
    read_only
        .close()
        .expect("close read-only compatibility image");

    // The frozen profile did not seal these values. Publishing its exact
    // section inventory beneath different transaction/cardinality coordinates
    // must still select the compatibility reader; authoritative bytes, not
    // cached header counts, determine the recovered world.
    image.transaction_id = image
        .transaction_id
        .checked_add(1_000_000)
        .expect("fixture transaction id has headroom");
    image.node_count = image
        .node_count
        .checked_add(101)
        .expect("fixture node count has headroom");
    image.edge_count = image
        .edge_count
        .checked_add(103)
        .expect("fixture edge count has headroom");
    publish_image(&path, &image);
    let changed_header = read_image(&path);
    assert_section_only_profile(&changed_header);
    assert_eq!(changed_header.transaction_id, image.transaction_id);
    assert_eq!(changed_header.node_count, image.node_count);
    assert_eq!(changed_header.edge_count, image.edge_count);

    let read_only = GrafeoDB::open_read_only(&path)
        .expect("section-only v2 must remain readable under unsealed coordinates");
    assert_world_and_exact_indexes(
        &read_only,
        nearest,
        &expected_vector,
        #[cfg(feature = "text-index")]
        &expected_text,
    );
    read_only
        .close()
        .expect("close changed-header compatibility image");

    // A predecessor container cannot authorize a WAL tail under current-only
    // recovery, even before the unauthenticated transaction coordinate is used. Its TID is
    // deliberately below the forged header value: trusting that value would
    // silently discard the group and make this open succeed.
    std::fs::copy(&path, &wal_probe).expect("copy compatibility image for WAL probe");
    let legacy_transaction_id = TransactionId::new(
        same_header
            .transaction_id
            .checked_add(1)
            .expect("source transaction id has headroom"),
    );
    assert!(legacy_transaction_id.as_u64() < changed_header.transaction_id);
    append_epochless_legacy_commit(&wal_probe, legacy_transaction_id);
    let before_rejected_open = std::fs::read(&wal_probe).expect("read WAL probe container");
    let error = match GrafeoDB::open(&wal_probe) {
        Ok(db) => {
            let _ = db.close();
            panic!("unsealed header TID silently classified an epoch-less WAL group")
        }
        Err(error) => error,
    };
    assert!(matches!(error, Error::Serialization(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("cannot bind WAL to an unauthenticated container image"),
        "recovery must reject predecessor base/tail authority before replay: {error}"
    );
    assert_eq!(
        std::fs::read(&wal_probe).expect("reread rejected WAL probe container"),
        before_rejected_open,
        "failed recovery must not rewrite the compatibility container"
    );

    // A writable close is an explicit checkpoint. It must preserve the exact
    // Catalog/index generation, repair cached counts from authoritative data,
    // and upgrade WorldMetadata to the coordinate-sealed v2 profile.
    let writable = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )?;
    assert_world_and_exact_indexes(
        &writable,
        nearest,
        &expected_vector,
        #[cfg(feature = "text-index")]
        &expected_text,
    );
    writable.close().expect("upgrade frozen v2 profile");

    let upgraded = read_image(&path);
    assert_eq!(upgraded.node_count, 2);
    assert_eq!(upgraded.edge_count, 1);
    assert_eq!(
        upgraded.section(SectionType::Catalog).bytes,
        expected_catalog,
        "profile upgrade must preserve exact Catalog v7 state"
    );
    assert_eq!(
        upgraded.section(SectionType::VectorStore).bytes,
        expected_vector,
        "profile upgrade must preserve exact vector state"
    );
    #[cfg(feature = "text-index")]
    assert_eq!(
        upgraded.section(SectionType::TextIndex).bytes,
        expected_text,
        "profile upgrade must preserve exact text state"
    );
    assert_current_coordinate_profile(&upgraded);

    // Once upgraded, replaying the same CRC-valid section inventory beneath a
    // different active-header transaction coordinate is a hostile splice.
    std::fs::copy(&path, &splice_probe).expect("copy upgraded image for splice probe");
    let mut spliced = upgraded.clone();
    spliced.transaction_id = spliced
        .transaction_id
        .checked_add(1_000_000)
        .expect("upgraded transaction id has headroom");
    publish_image(&splice_probe, &spliced);
    let published_splice = read_image(&splice_probe);
    assert_eq!(published_splice.sections, upgraded.sections);
    assert_eq!(published_splice.transaction_id, spliced.transaction_id);

    let before_rejected_splice =
        std::fs::read(&splice_probe).expect("read coordinate splice before open");
    let error = match GrafeoDB::open(&splice_probe) {
        Ok(db) => {
            let _ = db.close();
            panic!("coordinate-sealed v2 image accepted a foreign WAL floor")
        }
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::Storage(StorageError::Corruption(_))),
        "expected structured container corruption, got {error:?}"
    );
    assert!(
        error.to_string().contains("recovery image digest mismatch"),
        "the current coordinate seal must reject before model/WAL recovery: {error}"
    );
    assert_eq!(
        std::fs::read(&splice_probe).expect("reread rejected coordinate splice"),
        before_rejected_splice,
        "failed recovery must not rewrite the rejected coordinate splice"
    );
    Ok(())
}
