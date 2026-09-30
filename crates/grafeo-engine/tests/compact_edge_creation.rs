//! Direct edge construction must preserve tier routing and publication boundaries.

#![cfg(all(feature = "lpg", feature = "compact-store"))]
#![allow(missing_docs)]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::{Config, GrafeoDB};

fn database() -> GrafeoDB {
    let config = Config::in_memory();
    #[cfg(feature = "cdc")]
    let config = config.with_cdc();
    GrafeoDB::with_config(config).expect("database")
}

#[test]
fn named_edge_creation_without_cdc_dependency_keeps_default_graph_unchanged() {
    let db = GrafeoDB::new_in_memory();
    assert!(
        db.create_graph("https://example.test/named")
            .expect("graph")
    );
    let default = db.create_node(&["Default"]);
    let session = db.session();
    session
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["https://example.test/named"])
                .expect("literal graph path"),
        )
        .expect("select existing graph");
    let src = session.create_node(&["Source"]);
    let dst = session.create_node(&["Destination"]);
    let edge = session.create_edge(src, dst, "NAMED");
    let value = session.get_edge(edge).expect("named edge");
    assert_eq!((value.src, value.dst), (src, dst));
    assert_eq!(value.edge_type.as_str(), "NAMED");
    assert_eq!(db.edge_count(), 0);
    assert!(db.get_node(default).is_some());
}

#[cfg(feature = "cdc")]
#[test]
fn bare_and_empty_property_edge_creates_keep_absent_after_images() {
    use grafeo_common::types::EpochId;
    use grafeo_engine::cdc::{ChangeKind, EntityId};
    let mut db = database();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.compact().expect("compact");
    let bare = db.session().create_edge(a, b, "BARE");
    let empty = db
        .session()
        .create_edge_with_props(a, b, "EMPTY", [])
        .expect("empty props");
    assert!(bare.is_valid());
    assert!(empty.is_valid());
    let events: Vec<_> = db
        .fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
        .expect("events")
        .into_iter()
        .filter(|event| matches!(event.entity_id, EntityId::Edge(_)))
        .collect();
    assert_eq!(events.len(), 2);
    assert!(
        events
            .iter()
            .all(|event| event.kind == ChangeKind::Create && event.after.is_none())
    );
}

#[cfg(all(feature = "cdc", feature = "triple-store", feature = "sparql"))]
#[test]
fn mixed_model_session_stages_compound_edge_and_triple_in_one_transaction() {
    use grafeo_common::types::EpochId;
    use grafeo_engine::{GraphModel, cdc::EntityId};
    let mut db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Both)
            .with_cdc(),
    )
    .expect("mixed database");
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.compact().expect("compact mixed LPG");
    let mut session = db.session();
    session.begin_transaction().expect("begin mixed");
    let edge = session
        .create_edge_with_props(a, b, "MIXED", [("value", Value::Int64(4))])
        .expect("edge");
    assert!(edge.is_valid());
    session
        .execute_sparql("INSERT DATA { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o> }")
        .expect("triple");
    assert!(db.get_edge(edge).is_none());
    session.commit().expect("commit both models");
    let edge_event = db
        .fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
        .expect("events")
        .into_iter()
        .find(|event| event.entity_id == EntityId::Edge(edge))
        .expect("edge event");
    assert_eq!(
        edge_event.after.expect("full image").get("value"),
        Some(&Value::Int64(4))
    );
    assert_eq!(
        db.session()
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> <http://ex.org/o> }")
            .expect("committed triple")
            .row_count(),
        1
    );
}

#[test]
fn direct_edges_connect_cold_and_new_endpoints_and_survive_recompact() {
    let mut db = database();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let before = db.current_epoch();
    db.compact().expect("compact");
    let fresh = db.create_node(&["New"]);
    let session = db.session();
    let cold = session.create_edge(a, b, "COLD");
    assert!(cold.is_valid(), "cold endpoints must use the merged writer");
    let outgoing = session
        .create_edge_with_props(a, fresh, "OUT", [("weight", Value::Int64(7))])
        .expect("cold to new");
    let incoming = session
        .create_edge_with_props(fresh, b, "IN", [("weight", Value::Int64(8))])
        .expect("new to cold");
    assert!(incoming.is_valid());
    drop(session);
    for (id, src, dst) in [(cold, a, b), (outgoing, a, fresh), (incoming, fresh, b)] {
        let edge = db.get_edge(id).expect("created edge");
        assert_eq!((edge.src, edge.dst), (src, dst));
        assert!(db.get_edge_at_epoch(id, before).is_none());
    }
    db.compact().expect("recompact created edges");
    assert_eq!(db.edge_count(), 3);
    assert_eq!(
        db.get_edge(outgoing)
            .expect("recompacted edge")
            .get_property("weight"),
        Some(&Value::Int64(7))
    );
}

#[test]
fn invalid_second_endpoint_never_promotes_the_valid_cold_endpoint() {
    let mut db = database();
    let cold = db.create_node(&["Cold"]);
    db.compact().expect("compact");
    let layered = db.layered_store().expect("layered");
    assert_eq!(layered.overlay_node_count(), 0);
    for missing in [NodeId::new(999), NodeId::INVALID] {
        assert!(
            db.session()
                .create_edge_with_props(cold, missing, "REJECTED", [("x", Value::Int64(1))])
                .is_err(),
            "invalid endpoint must be a checked rejection"
        );
        assert_eq!(layered.overlay_node_count(), 0);
        assert_eq!(layered.overlay_edge_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }
}

#[test]
fn closed_and_foreign_pending_endpoints_are_not_physical_existence() {
    let mut db = database();
    let live = db.create_node(&["Live"]);
    let closed = db.create_node(&["Closed"]);
    assert!(db.delete_node(closed));
    db.compact().expect("compact retained closed node");
    assert!(
        db.session()
            .create_edge_with_props(live, closed, "CLOSED", [])
            .is_err()
    );
    let mut owner = db.session();
    owner.begin_transaction().expect("owner begins");
    let pending = owner.create_node(&["Pending"]);
    assert!(pending.is_valid());
    assert!(
        db.session()
            .create_edge_with_props(live, pending, "FOREIGN", [])
            .is_err()
    );
    let own = owner
        .create_edge_with_props(live, pending, "OWN", [])
        .expect("own pending endpoint is visible");
    assert!(owner.get_edge(own).is_some());
    assert!(db.get_edge(own).is_none());
    owner.rollback().expect("rollback owner");
    assert_eq!(db.edge_count(), 0);
}

#[test]
fn explicit_and_disabled_auto_commit_keep_cold_edges_private_until_commit() {
    for explicit in [true, false] {
        let mut db = database();
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        db.compact().expect("compact");
        let reader = db.session();
        let mut writer = db.session();
        if explicit {
            writer.begin_transaction().expect("begin");
        } else {
            writer.set_auto_commit(false);
        }
        let edge = writer
            .create_edge_with_props(a, b, "PENDING", [("x", Value::Int64(3))])
            .expect("create pending edge");
        assert!(writer.in_transaction());
        assert!(writer.get_edge(edge).is_some());
        assert!(reader.get_edge(edge).is_none());
        writer.commit().expect("commit");
        assert!(db.get_edge(edge).is_some());
    }
}

#[test]
fn read_only_session_cannot_create_edges_through_retained_writer_handles() {
    let mut db = database();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.compact().expect("compact");
    let reader = db.session_with_role(grafeo_engine::auth::Role::ReadOnly);
    assert!(reader.create_edge_with_props(a, b, "DENIED", []).is_err());
    assert!(!reader.create_edge(a, b, "DENIED").is_valid());
    assert_eq!(db.layered_store().expect("layered").overlay_node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

#[cfg(feature = "cdc")]
#[test]
fn compound_create_is_one_full_event_followed_by_an_independent_update() {
    check_compound_create(false);
}

#[cfg(feature = "cdc")]
#[test]
fn named_compound_create_is_one_full_event_followed_by_an_independent_update() {
    check_compound_create(true);
}

#[cfg(feature = "cdc")]
fn check_compound_create(named: bool) {
    use grafeo_common::types::EpochId;
    use grafeo_engine::cdc::{ChangeKind, EntityId};

    {
        let mut db = database();
        let graph = "https://example.test/graph";
        if named {
            assert!(db.create_graph(graph).expect("named graph"));
        }
        let session = db.session();
        if named {
            session
                .use_graph_path(
                    &grafeo_common::types::GraphPath::from_components(&[graph])
                        .expect("literal graph path"),
                )
                .expect("select existing graph");
        }
        let a = session.create_node(&["A"]);
        let b = session.create_node(&["B"]);
        drop(session);
        if !named {
            db.compact().expect("compact default graph");
        }
        let mut session = db.session();
        if named {
            session
                .use_graph_path(
                    &grafeo_common::types::GraphPath::from_components(&[graph])
                        .expect("literal graph path"),
                )
                .expect("select existing graph");
        }
        session
            .begin_transaction()
            .expect("begin compound transaction");
        let edge = session
            .create_edge_with_props(
                a,
                b,
                "COMPLETE",
                [
                    ("x", Value::Int64(1)),
                    ("x", Value::Int64(2)),
                    ("y", Value::Int64(4)),
                ],
            )
            .expect("compound create");
        assert!(edge.is_valid());
        assert!(session.get_edge(edge).is_some());
        session
            .set_edge_property(edge, "x", Value::Int64(3))
            .expect("independent SET");
        assert!(
            db.fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
                .expect("uncommitted events")
                .iter()
                .all(|event| event.entity_id != EntityId::Edge(edge))
        );
        session.commit().expect("commit compound transaction");
        let events: Vec<_> = db
            .fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
            .expect("committed events")
            .into_iter()
            .filter(|event| event.entity_id == EntityId::Edge(edge))
            .collect();
        assert_eq!(
            events.len(),
            2,
            "one compound Create and one ordinary Update"
        );
        assert_eq!(events[0].kind, ChangeKind::Create);
        assert_eq!(events[0].edge_type.as_deref(), Some("COMPLETE"));
        assert_eq!(
            (events[0].src_id, events[0].dst_id),
            (Some(a.as_u64()), Some(b.as_u64()))
        );
        let expected_graph = if named {
            grafeo_common::types::GraphPath::from_components(&[graph]).expect("literal graph path")
        } else {
            grafeo_common::types::GraphPath::root()
        };
        assert_eq!(events[0].graph_path(), Some(&expected_graph));
        assert_eq!(events[0].triple_graph, None);
        assert_eq!(
            events[0]
                .after
                .as_ref()
                .expect("complete create properties")
                .get("x"),
            Some(&Value::Int64(2))
        );
        assert_eq!(
            events[0]
                .after
                .as_ref()
                .expect("complete create properties")
                .get("y"),
            Some(&Value::Int64(4))
        );
        assert_eq!(events[1].kind, ChangeKind::Update);
        assert_eq!(
            events[1].before.as_ref().expect("update before").get("x"),
            Some(&Value::Int64(2))
        );
        assert_eq!(
            events[1].after.as_ref().expect("update after").get("x"),
            Some(&Value::Int64(3))
        );
        if named {
            assert_eq!(db.edge_count(), 0);
        }
    }
}

#[cfg(feature = "cdc")]
#[test]
fn savepoint_and_transaction_rollback_discard_complete_edge_events() {
    use grafeo_common::types::EpochId;
    use grafeo_engine::cdc::EntityId;
    let mut db = database();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.compact().expect("compact");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session.savepoint("before_edge").expect("savepoint");
    let first = session
        .create_edge_with_props(a, b, "SAVEPOINT", [("x", Value::Int64(1))])
        .expect("first edge");
    assert!(first.is_valid());
    assert!(session.get_edge(first).is_some());
    session
        .rollback_to_savepoint("before_edge")
        .expect("rollback savepoint");
    assert!(session.get_edge(first).is_none());
    let second = session
        .create_edge_with_props(a, b, "ROLLBACK", [("x", Value::Int64(2))])
        .expect("second edge");
    assert!(second.is_valid());
    assert!(session.get_edge(second).is_some());
    session.rollback().expect("rollback transaction");
    assert!(db.get_edge(second).is_none());
    assert!(
        db.fixture_changes(EpochId::INITIAL..=EpochId::new(u64::MAX))
            .expect("events")
            .iter()
            .all(|event| !matches!(event.entity_id, EntityId::Edge(_)))
    );
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn wal_recovery_contains_one_creation_and_each_property_operation() {
    use grafeo_engine::DurabilityMode;
    use grafeo_storage::wal::{LpgMutationOp, WalRecord, WalRecovery};
    let temp = tempfile::tempdir().expect("directory");
    let path = temp.path().join("database.grafeo");
    let config = Config::persistent(&path).with_wal_durability(DurabilityMode::Sync);
    #[cfg(feature = "cdc")]
    let config = config.with_cdc();
    let edge;
    {
        let mut db = GrafeoDB::with_config(config.clone()).expect("persistent database");
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        db.compact().expect("compact persistent endpoints");
        assert_eq!(db.layered_store().expect("layered").overlay_node_count(), 0);
        edge = db
            .session()
            .create_edge_with_props(
                a,
                b,
                "ONCE",
                [("x", Value::Int64(7)), ("y", Value::Int64(9))],
            )
            .expect("edge");
        assert!(
            db.session()
                .create_edge_with_props(a, NodeId::new(999), "REJECTED", [])
                .is_err()
        );
        db.wal().expect("WAL").sync().expect("sync");
        let inspection = tempfile::tempdir().expect("inspection directory");
        let inspection_wal = inspection.path().join("wal");
        std::fs::create_dir(&inspection_wal).unwrap();
        for entry in std::fs::read_dir(path.with_extension("grafeo.wal")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), inspection_wal.join(entry.file_name())).unwrap();
        }
        let records = WalRecovery::new(&inspection_wal)
            .unwrap()
            .recover()
            .expect("recover records");
        let creates: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::LpgMutation {
                    graph,
                    op: LpgMutationOp::CreateEdge { id, edge_type, .. },
                    ..
                } => Some((graph, *id, edge_type)),
                _ => None,
            })
            .collect();
        assert_eq!(creates.len(), 1);
        assert_eq!(
            creates[0],
            (
                &grafeo_common::types::GraphPath::root(),
                edge,
                &"ONCE".to_string()
            )
        );
        assert_eq!(records.iter().filter(|record| matches!(record,
            WalRecord::LpgMutation { op: LpgMutationOp::SetEdgeProperty { id, .. }, .. } if *id == edge)).count(), 2);
        db.close().expect("close");
    }
    let reopened = GrafeoDB::with_config(config).expect("reopen");
    let restored = reopened.get_edge(edge).expect("durable edge");
    assert_eq!(restored.get_property("x"), Some(&Value::Int64(7)));
    assert_eq!(restored.get_property("y"), Some(&Value::Int64(9)));
}

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "cdc",
    feature = "testing-crash-injection"
))]
#[test]
fn failed_edge_wal_append_never_stages_a_compound_create() {
    use grafeo_common::testing::wal_failure::{
        disable_mutation_log_failure, enable_mutation_log_failure_once,
    };
    let temp = tempfile::tempdir().expect("directory");
    let db =
        GrafeoDB::with_config(Config::persistent(temp.path().join("failure.grafeo")).with_cdc())
            .expect("database");
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let events_before = db.memory_usage().cdc.event_count;
    enable_mutation_log_failure_once();
    let result = db
        .session()
        .create_edge_with_props(a, b, "FAIL", [("x", Value::Int64(1))]);
    disable_mutation_log_failure();
    assert!(result.is_err());
    assert!(db.is_durability_poisoned());
    assert_eq!(db.edge_count(), 0);
    assert_eq!(db.memory_usage().cdc.event_count, events_before);
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;
