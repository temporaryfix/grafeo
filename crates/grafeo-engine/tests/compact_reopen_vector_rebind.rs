//! Recovery qualification for topology-only vector indexes over a compact base.
//!
//! A restored HNSW section owns topology, not duplicate vector payloads. When
//! its LPG overlay promotes and re-embeds a cold node, index maintenance must
//! resolve every other candidate through the exact CompactStore generation.

#![cfg(all(
    feature = "lpg",
    feature = "wal",
    feature = "grafeo-file",
    feature = "compact-store",
    feature = "vector-index"
))]

use std::path::Path;

use grafeo_common::storage::SectionType;
use grafeo_common::types::{NodeId, Value, WorldMetadataSectionV2};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::file::GrafeoFileManager;

fn persistent_sync(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent LPG database")
}

fn assert_recovery_sealed_v2(path: &Path) {
    let manager = GrafeoFileManager::open_read_only(path).expect("open persisted container");
    let directory = manager
        .read_section_directory()
        .expect("read persisted section directory")
        .expect("active persisted section directory");
    let metadata = directory
        .entries()
        .iter()
        .find(|entry| entry.section_type == SectionType::WorldMetadata)
        .expect("current containers carry WorldMetadata");
    assert_eq!(metadata.version, WorldMetadataSectionV2::SECTION_VERSION);
    manager.close().expect("close persisted container");
}

fn assert_complete_nearest_neighbour_cut(db: &GrafeoDB, expected: &[NodeId], moved: NodeId) {
    let nearest = db
        .vector_search(
            "Document",
            "embedding",
            &[-1_000.0, -1_000.0],
            1,
            None,
            None,
        )
        .expect("search the restored physical vector index");
    assert_eq!(
        nearest.first().map(|(id, _)| *id),
        Some(moved),
        "the re-embedded cold identity must reconnect at its new nearest-neighbour position"
    );

    let mut all_hits: Vec<_> = db
        .vector_search(
            "Document",
            "embedding",
            &[-1_000.0, -1_000.0],
            expected.len(),
            None,
            None,
        )
        .expect("search the complete restored topology")
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    all_hits.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    assert_eq!(
        all_hits, expected,
        "a base-node update must not disconnect any identity from the restored HNSW topology"
    );
}

#[test]
fn sealed_compact_reopen_binds_vector_cold_base_before_session_update() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let path = directory.path().join("compact-vector-rebind.grafeo");
    let mut ids = Vec::new();

    {
        let mut db = persistent_sync(&path);
        let mut session = db.session();
        session
            .begin_transaction()
            .expect("begin fixture transaction");
        for offset in 0..96_u64 {
            ids.push(
                session
                    .create_node_with_props(
                        &["Document"],
                        [("embedding", Value::Vector(vec![offset as f32, 0.0].into()))],
                    )
                    .expect("create vector fixture node"),
            );
        }
        session.commit().expect("commit vector fixture");
        drop(session);

        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Document".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(2),
                metric: Some("euclidean".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create topology-only HNSW index");
        db.compact().expect("compact authoritative LPG rows");
        db.close().expect("checkpoint compact vector database");
    }
    assert_recovery_sealed_v2(&path);

    let reopened = GrafeoDB::open(&path).expect("reopen sealed compact vector database");
    let entry_point = grafeo_engine::database::testing::root_lpg_store(&reopened)
        .get_vector_index("Document", "embedding")
        .expect("restored vector index")
        .snapshot_topology()
        .0;
    let moved = ids
        .iter()
        .rev()
        .copied()
        .find(|id| Some(*id) != entry_point)
        .expect("fixture has a non-entry-point cold node");
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&reopened)
            .get_node(moved)
            .is_none(),
        "the update target must begin resident only in the compact base"
    );

    {
        let mut session = reopened.session();
        session.begin_transaction().expect("begin cold-node update");
        session
            .set_node_property(
                moved,
                "embedding",
                Value::Vector(vec![-1_000.0, -1_000.0].into()),
            )
            .expect("re-embed compact-base node through Session");
        session.commit().expect("commit cold-node update");
    }

    let (_, _, topology) = grafeo_engine::database::testing::root_lpg_store(&reopened)
        .get_vector_index("Document", "embedding")
        .expect("updated vector index")
        .snapshot_topology();
    let moved_layers = topology
        .iter()
        .find_map(|(id, layers)| (*id == moved).then_some(layers))
        .expect("updated identity remains in HNSW topology");
    assert!(
        moved_layers
            .first()
            .is_some_and(|neighbors| neighbors.iter().any(|id| *id != moved)),
        "re-embedding must resolve cold candidates and publish a real layer-zero neighbour"
    );
    assert_complete_nearest_neighbour_cut(&reopened, &ids, moved);

    reopened.close().expect("checkpoint promoted vector node");
    assert_recovery_sealed_v2(&path);

    let reopened_again = GrafeoDB::open(&path).expect("reopen promoted compact vector database");
    assert_complete_nearest_neighbour_cut(&reopened_again, &ids, moved);
    reopened_again
        .close()
        .expect("close qualification database");
}
