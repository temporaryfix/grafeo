//! Exact default-vs-empty-named LPG recovery from a post-checkpoint WAL tail.
//!
//! Current GraphPath coordinates keep [] and [""] distinct through the durable
//! wire and recovery dispatch. No predecessor graph-tagged record is accepted.

#![cfg(all(feature = "lpg", feature = "wal", feature = "grafeo-file"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, GraphPath, NodeId, PropertyKey, TransactionId, Value};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::wal::{
    DurabilityMode as WalDurabilityMode, LpgMutationOp, LpgWal, WalConfig, WalRecord,
};

fn sidecar_wal_dir(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

fn copy_tree(source: &Path, target: &Path) {
    std::fs::create_dir_all(target).expect("create copied WAL directory");
    for entry in std::fs::read_dir(source).expect("read source WAL directory") {
        let entry = entry.expect("read source WAL entry");
        let destination = target.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &destination);
        } else {
            std::fs::copy(entry.path(), destination).expect("copy WAL file");
        }
    }
}

fn snapshot_process_image(source: &Path, target: &Path) {
    std::fs::copy(source, target).expect("copy checkpoint container");
    let source_wal = sidecar_wal_dir(source);
    if source_wal.exists() {
        copy_tree(&source_wal, &sidecar_wal_dir(target));
    }
}

fn open_sync(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent LPG database")
}

fn append_committed(wal: &LpgWal, transaction_id: TransactionId, epoch: EpochId) {
    wal.log(&WalRecord::Committed {
        transaction_id,
        epoch,
    })
    .expect("append durable commit marker");
}

#[test]
fn post_checkpoint_wal_keeps_default_and_empty_named_graphs_distinct() {
    let temp = tempfile::TempDir::new().expect("temporary directory");
    let live_path = temp.path().join("live.grafeo");
    let crash_image = temp.path().join("crash-image.grafeo");

    let db = open_sync(&live_path);
    let checkpoint_node = db.create_node(&["CheckpointDefault"]);
    assert!(checkpoint_node.is_valid());
    db.wal_checkpoint().expect("establish durable checkpoint");
    let checkpoint_epoch = db.current_epoch();

    // Copy the exact durable process image before closing the live handle. The
    // records below are therefore necessarily a WAL tail after the checkpoint,
    // never data accidentally supplied by a later clean close snapshot.
    snapshot_process_image(&live_path, &crash_image);
    db.close()
        .expect("close original after capturing its image");
    drop(db);

    let wal = LpgWal::with_config(
        sidecar_wal_dir(&crash_image),
        WalConfig {
            durability: WalDurabilityMode::Sync,
            ..WalConfig::default()
        },
    )
    .expect("open copied sidecar WAL");

    let shared_id = NodeId::new(70_001);
    let default_tx = TransactionId::new(80_001);
    wal.log(&WalRecord::lpg(
        default_tx,
        GraphPath::root(),
        LpgMutationOp::CreateNode {
            id: shared_id,
            labels: vec!["DefaultAfterCheckpoint".to_string()],
        },
    ))
    .expect("append exact default-graph mutation");
    wal.log(&WalRecord::lpg(
        default_tx,
        GraphPath::root(),
        LpgMutationOp::SetNodeProperty {
            id: shared_id,
            key: "scope".to_string(),
            value: Value::from("default"),
        },
    ))
    .expect("append default-graph property");
    append_committed(
        &wal,
        default_tx,
        EpochId::new(checkpoint_epoch.as_u64() + 1),
    );

    // An independent root-only row must never leak into the empty-name child.
    let root_only_id = NodeId::new(70_002);
    let root_only_tx = TransactionId::new(80_002);
    wal.log(&WalRecord::lpg(
        root_only_tx,
        GraphPath::root(),
        LpgMutationOp::CreateNode {
            id: root_only_id,
            labels: vec!["RootOnly".to_string()],
        },
    ))
    .expect("append current root-only mutation");
    append_committed(
        &wal,
        root_only_tx,
        EpochId::new(checkpoint_epoch.as_u64() + 2),
    );

    let empty_named_tx = TransactionId::new(80_003);
    let empty_path = GraphPath::from_components(&[""]).expect("checked empty-name child path");
    wal.log(&WalRecord::CreateLpgGraph {
        incarnation: grafeo_common::types::GraphIncarnationId::new(1),
        graph: empty_path.clone(),
        transaction_id: empty_named_tx,
    })
    .expect("append empty named-graph lifecycle record");
    wal.log(&WalRecord::lpg(
        empty_named_tx,
        empty_path.clone(),
        LpgMutationOp::CreateNode {
            id: shared_id,
            labels: vec!["EmptyNamedAfterCheckpoint".to_string()],
        },
    ))
    .expect("append exact empty-named-graph mutation");
    wal.log(&WalRecord::lpg(
        empty_named_tx,
        empty_path,
        LpgMutationOp::SetNodeProperty {
            id: shared_id,
            key: "scope".to_string(),
            value: Value::from("empty-named"),
        },
    ))
    .expect("append empty-named-graph property");
    append_committed(
        &wal,
        empty_named_tx,
        EpochId::new(checkpoint_epoch.as_u64() + 3),
    );
    wal.sync().expect("sync complete post-checkpoint WAL tail");
    drop(wal);

    let recovered = open_sync(&crash_image);
    let scope_key = PropertyKey::new("scope");
    let default = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .get_node(shared_id)
        .expect("recover default-graph node");
    assert!(default.has_label("DefaultAfterCheckpoint"));
    assert!(!default.has_label("EmptyNamedAfterCheckpoint"));
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_node_property(shared_id, &scope_key),
        Some(Value::from("default"))
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_node(checkpoint_node)
            .is_some(),
        "the checkpoint prefix must remain intact"
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_node(root_only_id)
            .is_some_and(|node| node.has_label("RootOnly")),
        "the independent root row must retain its exact coordinate"
    );

    let empty_named = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .graph("")
        .expect("recover legal empty named graph");
    let named_node = empty_named
        .get_node(shared_id)
        .expect("recover empty-named-graph node");
    assert!(named_node.has_label("EmptyNamedAfterCheckpoint"));
    assert!(!named_node.has_label("DefaultAfterCheckpoint"));
    assert_eq!(
        empty_named.get_node_property(shared_id, &scope_key),
        Some(Value::from("empty-named"))
    );
    assert!(
        empty_named.get_node(root_only_id).is_none(),
        "the root coordinate must not leak into the empty-name child"
    );
    assert_eq!(
        recovered.current_epoch(),
        checkpoint_epoch.next().next().next()
    );
    recovered.close().expect("close recovered image");
}
