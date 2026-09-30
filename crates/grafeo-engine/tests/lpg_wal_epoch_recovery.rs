//! Exact LPG transaction-time recovery from durable WAL commit markers.

#![cfg(all(feature = "lpg", feature = "wal"))]

use grafeo_common::types::{
    EpochId, GraphPath, HistoryCompleteness, NodeId, StoreId, TransactionId, Value,
    WorldIdentityMetadataV1,
};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::wal::{
    DurabilityMode as WalDurabilityMode, LpgMutationOp, LpgWal, WalConfig, WalRecord,
};

fn open_directory_database(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open WAL-directory LPG database")
}

fn write_wal(path: &std::path::Path, records: impl IntoIterator<Item = WalRecord>) {
    let wal = LpgWal::with_config(
        path.join("wal"),
        WalConfig {
            durability: WalDurabilityMode::Sync,
            ..WalConfig::default()
        },
    )
    .expect("create test WAL");
    let metadata = WorldIdentityMetadataV1::new(
        StoreId::generate().expect("test store identity"),
        HistoryCompleteness::Complete,
    )
    .expect("current identity metadata");
    wal.log(&WalRecord::StoreIdentityMeta { metadata })
        .expect("append identity");
    wal.log(&WalRecord::GraphModelMeta {
        model: GraphModel::Lpg.as_u8(),
    })
    .expect("append model");
    for record in records {
        wal.log(&record).expect("append test WAL record");
    }
    wal.sync().expect("sync test WAL");
}

fn property_at(
    store: &grafeo_core::graph::lpg::LpgStore,
    node: NodeId,
    epoch: u64,
    key: &str,
) -> Option<Value> {
    store
        .get_node_at_epoch(node, EpochId::new(epoch))
        .and_then(|node| node.get_property(key).cloned())
}

#[test]
fn default_graph_recovery_preserves_commit_epochs_and_reserved_gaps() {
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("default-epoch-db");
    let node = NodeId::new(41);
    let create_tx = TransactionId::new(2);
    let update_tx = TransactionId::new(3);

    write_wal(
        &path,
        [
            WalRecord::lpg(
                create_tx,
                GraphPath::root(),
                LpgMutationOp::CreateNode {
                    id: node,
                    labels: vec!["Tracked".to_string()],
                },
            ),
            WalRecord::lpg(
                create_tx,
                GraphPath::root(),
                LpgMutationOp::SetNodeProperty {
                    id: node,
                    key: "state".to_string(),
                    value: Value::from("created"),
                },
            ),
            WalRecord::Committed {
                transaction_id: create_tx,
                epoch: EpochId::new(2),
            },
            // Epochs 3 and 4 model reservations whose durable marker failed.
            WalRecord::lpg(
                update_tx,
                GraphPath::root(),
                LpgMutationOp::SetNodeProperty {
                    id: node,
                    key: "state".to_string(),
                    value: Value::from("updated"),
                },
            ),
            WalRecord::Committed {
                transaction_id: update_tx,
                epoch: EpochId::new(5),
            },
        ],
    );

    let db = open_directory_database(&path);
    assert_eq!(db.current_epoch(), EpochId::new(5));
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&db).current_epoch(),
        EpochId::new(5)
    );
    assert_eq!(
        property_at(
            grafeo_engine::database::testing::root_lpg_store(&db),
            node,
            1,
            "state"
        ),
        None
    );
    assert_eq!(
        property_at(
            grafeo_engine::database::testing::root_lpg_store(&db),
            node,
            2,
            "state"
        ),
        Some(Value::from("created"))
    );
    assert_eq!(
        property_at(
            grafeo_engine::database::testing::root_lpg_store(&db),
            node,
            4,
            "state"
        ),
        Some(Value::from("created")),
        "reserved epoch gaps must not move the update backward"
    );
    assert_eq!(
        property_at(
            grafeo_engine::database::testing::root_lpg_store(&db),
            node,
            5,
            "state"
        ),
        Some(Value::from("updated"))
    );
}

#[test]
fn named_graph_recovery_stamps_the_exact_path_and_reserved_gaps() {
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("named-epoch-db");
    let graph = "archive";
    let graph_path = GraphPath::from_components(&[graph]).expect("checked literal graph path");
    let node = NodeId::new(73);
    let create_tx = TransactionId::new(2);
    let update_tx = TransactionId::new(3);

    write_wal(
        &path,
        [
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: graph_path.clone(),
                transaction_id: create_tx,
            },
            WalRecord::lpg(
                create_tx,
                graph_path.clone(),
                LpgMutationOp::CreateNode {
                    id: node,
                    labels: vec!["Archived".to_string()],
                },
            ),
            WalRecord::lpg(
                create_tx,
                graph_path.clone(),
                LpgMutationOp::SetNodeProperty {
                    id: node,
                    key: "revision".to_string(),
                    value: Value::Int64(1),
                },
            ),
            WalRecord::Committed {
                transaction_id: create_tx,
                epoch: EpochId::new(7),
            },
            // Epochs 8-10 are intentionally absent durable reservations.
            WalRecord::lpg(
                update_tx,
                graph_path,
                LpgMutationOp::SetNodeProperty {
                    id: node,
                    key: "revision".to_string(),
                    value: Value::Int64(2),
                },
            ),
            WalRecord::Committed {
                transaction_id: update_tx,
                epoch: EpochId::new(11),
            },
        ],
    );

    let db = open_directory_database(&path);
    let named = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph(graph)
        .expect("recover named graph");
    assert_eq!(db.current_epoch(), EpochId::new(11));
    assert_eq!(named.current_epoch(), EpochId::new(11));
    assert_eq!(property_at(&named, node, 6, "revision"), None);
    assert_eq!(
        property_at(&named, node, 7, "revision"),
        Some(Value::Int64(1))
    );
    assert_eq!(
        property_at(&named, node, 10, "revision"),
        Some(Value::Int64(1)),
        "named-graph update must not collapse into a reserved gap"
    );
    assert_eq!(
        property_at(&named, node, 11, "revision"),
        Some(Value::Int64(2))
    );
}
