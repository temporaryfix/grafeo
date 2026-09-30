//! Direct node deletion is one atomic DETACH transaction.
//!
//! The node and every incoming, outgoing, or self-loop edge must share the
//! same transaction/WAL outcome. Concurrent edge creation must never publish
//! an edge whose endpoint was deleted.

#![cfg(feature = "lpg")]

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

#[derive(Clone, Copy)]
struct DetachFixture {
    victim: NodeId,
    incoming: EdgeId,
    outgoing: EdgeId,
    self_loop: EdgeId,
    survivor: EdgeId,
}

fn seed_detach_fixture(db: &GrafeoDB) -> DetachFixture {
    let victim = db.create_node(&["Victim"]);
    let source = db.create_node(&["Source"]);
    let target = db.create_node(&["Target"]);
    let incoming = db.create_edge(source, victim, "INCOMING");
    let outgoing = db.create_edge(victim, target, "OUTGOING");
    let self_loop = db.create_edge(victim, victim, "SELF");
    let survivor = db.create_edge(source, target, "SURVIVES");
    DetachFixture {
        victim,
        incoming,
        outgoing,
        self_loop,
        survivor,
    }
}

fn assert_fixture_intact(db: &GrafeoDB, fixture: DetachFixture) {
    assert!(db.get_node(fixture.victim).is_some(), "victim must exist");
    assert!(
        db.get_edge(fixture.incoming).is_some(),
        "incoming edge must exist"
    );
    assert!(
        db.get_edge(fixture.outgoing).is_some(),
        "outgoing edge must exist"
    );
    assert!(
        db.get_edge(fixture.self_loop).is_some(),
        "self-loop must exist"
    );
    assert!(
        db.get_edge(fixture.survivor).is_some(),
        "unrelated edge must exist"
    );
}

fn assert_fixture_detached(db: &GrafeoDB, fixture: DetachFixture) {
    assert!(db.get_node(fixture.victim).is_none(), "victim must be gone");
    assert!(
        db.get_edge(fixture.incoming).is_none(),
        "incoming edge must be detached"
    );
    assert!(
        db.get_edge(fixture.outgoing).is_none(),
        "outgoing edge must be detached"
    );
    assert!(
        db.get_edge(fixture.self_loop).is_none(),
        "self-loop must be detached exactly once"
    );
    assert!(
        db.get_edge(fixture.survivor).is_some(),
        "unrelated edge must survive"
    );
}

#[test]
fn database_delete_node_detaches_incoming_outgoing_and_self_loop_edges() {
    let db = GrafeoDB::new_in_memory();
    let fixture = seed_detach_fixture(&db);

    assert!(db.delete_node(fixture.victim));

    assert_fixture_detached(&db, fixture);
    assert_eq!(db.node_count(), 2);
    assert_eq!(db.edge_count(), 1);
}

#[test]
fn explicit_rollback_restores_the_node_and_every_incident_edge() {
    let db = GrafeoDB::new_in_memory();
    let fixture = seed_detach_fixture(&db);
    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("begin detach transaction");

    assert!(writer.delete_node(fixture.victim));
    assert!(
        writer.get_node(fixture.victim).is_none(),
        "writer reads its node delete"
    );
    assert!(
        writer.get_edge(fixture.incoming).is_none(),
        "writer reads its incoming-edge delete"
    );
    assert!(
        writer.get_edge(fixture.outgoing).is_none(),
        "writer reads its outgoing-edge delete"
    );
    assert!(
        writer.get_edge(fixture.self_loop).is_none(),
        "writer reads its self-loop delete"
    );
    assert_fixture_intact(&db, fixture);

    writer.rollback().expect("rollback detach transaction");
    assert_fixture_intact(&db, fixture);
}

#[derive(Clone, Copy)]
struct ZeroWidthFixture {
    anchor: NodeId,
    ephemeral: NodeId,
    edge: EdgeId,
    committed: EpochId,
    surviving_nodes: usize,
}

fn commit_zero_width_fixture(db: &GrafeoDB) -> ZeroWidthFixture {
    let surviving_nodes = db.node_count() + 1;
    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("begin ephemeral transaction");
    let anchor = writer.create_node(&["Anchor"]);
    let ephemeral = writer
        .create_node_with_props(&["Draft"], [("title", Value::from("ephemeral"))])
        .expect("create ephemeral node");
    assert!(writer.add_node_label(ephemeral, "Reviewed"));
    let edge = writer
        .create_edge_with_props(
            ephemeral,
            anchor,
            "TEMPORARY",
            [("weight", Value::Int64(7))],
        )
        .expect("create incident edge");

    assert!(
        writer.delete_node(ephemeral),
        "DETACH DELETE must see a node created by this transaction"
    );
    assert!(writer.get_node(ephemeral).is_none());
    assert!(writer.get_edge(edge).is_none());
    let committed = writer.commit().expect("commit zero-width lifetime");
    ZeroWidthFixture {
        anchor,
        ephemeral,
        edge,
        committed,
        surviving_nodes,
    }
}

fn assert_zero_width_history(db: &GrafeoDB, fixture: ZeroWidthFixture) {
    let ZeroWidthFixture {
        anchor,
        ephemeral,
        edge,
        committed,
        surviving_nodes,
    } = fixture;
    assert!(db.get_node(ephemeral).is_none());
    assert!(db.get_edge(edge).is_none());
    assert!(db.get_node_at_epoch(ephemeral, committed).is_none());
    assert!(db.get_edge_at_epoch(edge, committed).is_none());
    assert_eq!(
        db.get_node_history(ephemeral)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect::<Vec<_>>(),
        vec![(committed, Some(committed))]
    );
    assert_eq!(
        db.get_edge_history(edge)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect::<Vec<_>>(),
        vec![(committed, Some(committed))]
    );
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(db)
            .node_property_history_for_key(ephemeral, "title"),
        vec![
            (committed, Value::from("ephemeral")),
            (committed, Value::Null),
        ]
    );
    assert_eq!(
        db.get_node_property_history(ephemeral, "title"),
        vec![
            (committed, Value::from("ephemeral")),
            (committed, Value::Null),
        ]
    );
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(db).edge_property_history(edge),
        vec![(
            "weight".into(),
            vec![(committed, Value::Int64(7)), (committed, Value::Null)],
        )]
    );
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(db)
            .node_label_history(ephemeral)
            .into_iter()
            .map(|(epoch, labels)| {
                (
                    epoch,
                    labels
                        .into_iter()
                        .map(|label| label.to_string())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (committed, vec![String::from("Draft")]),
            (
                committed,
                vec![String::from("Draft"), String::from("Reviewed")],
            ),
        ]
    );
    assert!(db.get_node(anchor).is_some());
    assert_eq!(db.node_count(), surviving_nodes);
    assert_eq!(db.edge_count(), 0);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(db)
            .nodes_by_label("Draft")
            .is_empty()
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(db)
            .nodes_by_label("Reviewed")
            .is_empty()
    );
}

#[test]
fn same_transaction_create_mutate_detach_commits_exact_zero_width_histories() {
    let db = GrafeoDB::new_in_memory();
    let fixture = commit_zero_width_fixture(&db);
    assert_zero_width_history(&db, fixture);
}

#[cfg(feature = "compact-store")]
#[test]
fn zero_width_commit_on_layered_store_survives_compact_and_portable_copy() {
    let mut db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_gc_interval(0))
        .expect("open retained-history database");
    let cold = db.create_node(&["Cold"]);
    db.compact().expect("install managed Layered store");
    let fixture = commit_zero_width_fixture(&db);
    assert_zero_width_history(&db, fixture);

    for _ in 0..2 {
        db.compact().expect("recompact retained zero-width history");
        assert!(db.get_node(cold).is_some());
        assert!(
            db.get_node_at_epoch(fixture.ephemeral, fixture.committed)
                .is_none()
        );
        assert!(
            db.get_edge_at_epoch(fixture.edge, fixture.committed)
                .is_none()
        );
        assert_eq!(
            db.get_node_property_history(fixture.ephemeral, "title"),
            vec![
                (fixture.committed, Value::from("ephemeral")),
                (fixture.committed, Value::Null),
            ]
        );
        assert_eq!(
            db.get_edge_history(fixture.edge)
                .into_iter()
                .map(|(created, deleted, _)| (created, deleted))
                .collect::<Vec<_>>(),
            vec![(fixture.committed, Some(fixture.committed))]
        );
        // Read complete edge/label histories from an exact portable copy, not
        // the original database's overlay-only raw handle after compaction.
        let copied = db.to_memory().expect("copy complete retained history");
        assert_zero_width_history(&copied, fixture);
    }
}

#[cfg(feature = "compact-store")]
#[test]
fn zero_width_propertyless_edge_survives_recompact_and_portable_copy() {
    let mut db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_gc_interval(0))
        .expect("open retained-history database");
    let src = db.create_node(&["Node"]);
    let dst = db.create_node(&["Node"]);
    let survivor = db.create_edge(src, dst, "R");
    assert!(survivor.is_valid());
    db.compact()
        .expect("install existing cold relationship table");

    let mut writer = db.session();
    writer.begin_transaction().expect("begin zero-width edge");
    let ephemeral = writer.create_edge(src, dst, "R");
    assert!(ephemeral.is_valid());
    assert!(writer.delete_edge(ephemeral));
    let committed = writer.commit().expect("commit zero-width edge");
    drop(writer);

    for _ in 0..2 {
        db.compact().expect("recompact propertyless history");
        let copied = db.to_memory().expect("copy propertyless history");
        for store in [&db, &copied] {
            assert_eq!(store.node_count(), 2);
            assert_eq!(store.edge_count(), 1);
            assert!(store.get_edge(survivor).is_some());
            assert!(store.get_edge(ephemeral).is_none());
            assert!(store.get_edge_at_epoch(ephemeral, committed).is_none());
            let history = store.get_edge_history(ephemeral);
            assert_eq!(history.len(), 1);
            let (created, deleted, edge) = &history[0];
            assert_eq!(*created, committed);
            assert_eq!(*deleted, Some(committed));
            assert_eq!(edge.src, src);
            assert_eq!(edge.dst, dst);
            assert_eq!(edge.edge_type.as_str(), "R");
            assert!(edge.properties.is_empty());
        }
    }
}

#[test]
fn full_rollback_discards_same_transaction_create_and_detach() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("begin ephemeral transaction");
    let anchor = writer.create_node(&["Anchor"]);
    let ephemeral = writer.create_node(&["Draft"]);
    let edge = writer.create_edge(ephemeral, anchor, "TEMPORARY");
    assert!(writer.delete_node(ephemeral));

    writer.rollback().expect("roll back ephemeral transaction");

    assert!(db.get_node(anchor).is_none());
    assert!(db.get_node(ephemeral).is_none());
    assert!(db.get_edge(edge).is_none());
    assert!(db.get_node_history(ephemeral).is_empty());
    assert!(db.get_edge_history(edge).is_empty());
}

#[test]
fn savepoint_before_create_discards_same_transaction_create_and_detach() {
    let db = GrafeoDB::new_in_memory();
    let anchor = db.create_node(&["Anchor"]);
    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("begin savepoint transaction");
    writer.savepoint("before-create").expect("create savepoint");
    let ephemeral = writer.create_node(&["Draft"]);
    let edge = writer.create_edge(ephemeral, anchor, "TEMPORARY");
    assert!(writer.delete_node(ephemeral));

    writer
        .rollback_to_savepoint("before-create")
        .expect("roll back create and detach");
    writer.commit().expect("commit savepoint rollback");

    assert!(db.get_node(anchor).is_some());
    assert!(db.get_node(ephemeral).is_none());
    assert!(db.get_edge(edge).is_none());
    assert!(db.get_node_history(ephemeral).is_empty());
    assert!(db.get_edge_history(edge).is_empty());
}

#[test]
fn savepoint_before_detach_restores_same_transaction_node_and_edge() {
    let db = GrafeoDB::new_in_memory();
    let anchor = db.create_node(&["Anchor"]);
    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("begin savepoint transaction");
    let ephemeral = writer
        .create_node_with_props(&["Draft"], [("title", Value::from("retained"))])
        .expect("create retained node");
    let edge = writer
        .create_edge_with_props(
            ephemeral,
            anchor,
            "TEMPORARY",
            [("weight", Value::Int64(9))],
        )
        .expect("create retained edge");
    writer.savepoint("before-detach").expect("create savepoint");
    assert!(writer.delete_node(ephemeral));

    writer
        .rollback_to_savepoint("before-detach")
        .expect("roll back detach");
    assert!(writer.get_node(ephemeral).is_some());
    assert!(writer.get_edge(edge).is_some());
    writer.commit().expect("commit restored entities");

    assert_eq!(
        db.get_node(ephemeral)
            .and_then(|node| node.get_property("title").cloned()),
        Some(Value::from("retained"))
    );
    assert_eq!(
        db.get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(9))
    );
    assert_eq!(db.get_node_history(ephemeral)[0].1, None);
    assert_eq!(db.get_edge_history(edge)[0].1, None);
}

#[cfg(feature = "testing-statement-injection")]
#[test]
fn injected_commit_failure_rolls_back_the_whole_detach() {
    use grafeo_common::testing::statement_failure::with_commit_failure;

    let db = GrafeoDB::new_in_memory();
    let fixture = seed_detach_fixture(&db);

    with_commit_failure(|| {
        let mut writer = db.session();
        writer
            .begin_transaction()
            .expect("begin detach transaction");
        assert!(writer.delete_node(fixture.victim));
        assert!(writer.commit().is_err(), "commit injection must fire");
    });

    assert_fixture_intact(&db, fixture);
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_base_delete_node_detaches_cold_incident_edges() {
    let mut db = GrafeoDB::new_in_memory();
    let fixture = seed_detach_fixture(&db);
    db.compact().expect("compact fixture into cold base");

    assert!(db.delete_node(fixture.victim));

    assert_fixture_detached(&db, fixture);
    assert_eq!(db.edge_count(), 1);
}

#[test]
fn delete_commit_wins_then_concurrent_edge_commit_is_rejected() {
    let db = GrafeoDB::new_in_memory();
    let victim = db.create_node(&["Victim"]);
    let other = db.create_node(&["Other"]);
    let mut creator = db.session();
    let mut deleter = db.session();
    creator.begin_transaction().expect("begin edge creator");
    deleter.begin_transaction().expect("begin node deleter");
    let edge = creator.create_edge(victim, other, "RACES");
    assert!(edge.is_valid());
    assert!(deleter.delete_node(victim));

    deleter.commit().expect("node delete publishes first");
    assert!(
        creator.commit().is_err(),
        "edge commit must reject its deleted endpoint"
    );

    assert!(db.get_node(victim).is_none());
    assert!(db.get_edge(edge).is_none());
}

#[test]
fn edge_commit_wins_then_concurrent_stale_detach_is_rejected() {
    let db = GrafeoDB::new_in_memory();
    let victim = db.create_node(&["Victim"]);
    let other = db.create_node(&["Other"]);
    let mut creator = db.session();
    let mut deleter = db.session();
    creator.begin_transaction().expect("begin edge creator");
    deleter.begin_transaction().expect("begin node deleter");
    let edge = creator.create_edge(victim, other, "RACES");
    assert!(edge.is_valid());
    assert!(deleter.delete_node(victim));

    creator.commit().expect("edge publishes first");
    assert!(
        deleter.commit().is_err(),
        "stale detach must reject a newly committed incident edge"
    );

    assert!(db.get_node(victim).is_some());
    assert!(db.get_edge(edge).is_some());
}

#[cfg(feature = "cdc")]
#[test]
fn database_detach_emits_exactly_one_cdc_delete_per_removed_entity() {
    use grafeo_common::types::Value;
    use grafeo_engine::Config;
    use grafeo_engine::cdc::{ChangeKind, EntityId};

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).expect("open CDC database");
    let fixture = seed_detach_fixture(&db);
    db.set_node_property(fixture.victim, "phase", Value::from("ready"))
        .expect("set node property");
    db.set_edge_property(fixture.survivor, "weight", Value::Int64(7))
        .expect("set edge property");

    assert!(db.delete_node(fixture.victim));

    let change_count = |entity, kind| {
        db.fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(entity))
            .expect("read CDC history")
            .into_iter()
            .filter(|event| event.kind == kind)
            .count()
    };
    assert_eq!(
        change_count(EntityId::Node(fixture.victim), ChangeKind::Update),
        1
    );
    assert_eq!(
        change_count(EntityId::Edge(fixture.survivor), ChangeKind::Update),
        1
    );
    assert_eq!(
        change_count(EntityId::Node(fixture.victim), ChangeKind::Delete),
        1
    );
    assert_eq!(
        change_count(EntityId::Edge(fixture.incoming), ChangeKind::Delete),
        1
    );
    assert_eq!(
        change_count(EntityId::Edge(fixture.outgoing), ChangeKind::Delete),
        1
    );
    assert_eq!(
        change_count(EntityId::Edge(fixture.self_loop), ChangeKind::Delete),
        1
    );
    assert_eq!(
        change_count(EntityId::Edge(fixture.survivor), ChangeKind::Delete),
        0
    );

    assert!(db.delete_edge(fixture.survivor));
    assert_eq!(
        change_count(EntityId::Edge(fixture.survivor), ChangeKind::Delete),
        1
    );
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod durability {
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    #[cfg(feature = "testing-crash-injection")]
    use super::assert_fixture_intact;
    use super::{
        assert_fixture_detached, assert_zero_width_history, commit_zero_width_fixture,
        seed_detach_fixture,
    };

    fn open_wal_directory(path: &std::path::Path) -> GrafeoDB {
        let config = Config::persistent(path).with_storage_format(StorageFormat::WalDirectory);
        GrafeoDB::with_config(config).expect("open WAL-directory database")
    }

    #[test]
    fn zero_width_commit_survives_wal_reopen_with_exact_histories() {
        let dir = tempfile::tempdir().expect("temporary database directory");
        let path = dir.path().join("zero-width-wal");
        let fixture;
        {
            let db = open_wal_directory(&path);
            fixture = commit_zero_width_fixture(&db);
            assert_zero_width_history(&db, fixture);
            db.close().expect("close committed database");
        }
        let reopened = open_wal_directory(&path);
        assert_zero_width_history(&reopened, fixture);
    }

    #[test]
    fn zero_width_commit_survives_container_save_and_reopen() {
        let dir = tempfile::tempdir().expect("temporary database directory");
        let path = dir.path().join("zero-width.grafeo");
        let db = GrafeoDB::new_in_memory();
        let fixture = commit_zero_width_fixture(&db);
        assert_zero_width_history(&db, fixture);
        db.save(&path).expect("save complete container history");
        let reopened = GrafeoDB::open(&path).expect("reopen complete container history");
        assert_zero_width_history(&reopened, fixture);
    }

    #[test]
    fn committed_detach_survives_wal_reopen_without_resurrecting_edges() {
        let dir = tempfile::tempdir().expect("temporary database directory");
        let path = dir.path().join("detach-reopen");
        let fixture;
        {
            let db = open_wal_directory(&path);
            fixture = seed_detach_fixture(&db);
            assert!(db.delete_node(fixture.victim));
            assert_fixture_detached(&db, fixture);
            db.close().expect("close committed database");
        }

        let reopened = open_wal_directory(&path);
        assert_fixture_detached(&reopened, fixture);
        assert_eq!(reopened.node_count(), 2);
        assert_eq!(reopened.edge_count(), 1);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn detach_mutation_wal_failure_rolls_back_and_reopen_keeps_original_graph() {
        use grafeo_common::testing::wal_failure::{
            disable_mutation_log_failure, enable_mutation_log_failure_once,
        };

        let dir = tempfile::tempdir().expect("temporary database directory");
        let path = dir.path().join("detach-wal-failure");
        let fixture;
        {
            let db = open_wal_directory(&path);
            fixture = seed_detach_fixture(&db);
            enable_mutation_log_failure_once();
            let deleted = db.delete_node(fixture.victim);
            disable_mutation_log_failure();

            assert!(!deleted, "WAL mutation failure must fail the detach");
            assert!(
                db.is_durability_poisoned(),
                "ambiguous WAL mutation state must poison the database"
            );
            assert_fixture_intact(&db, fixture);
            // Drop deliberately cannot checkpoint a poisoned database. Reopen
            // resolves the durable outcome from the WAL.
        }

        let reopened = open_wal_directory(&path);
        assert_fixture_intact(&reopened, fixture);
        assert_eq!(reopened.node_count(), 3);
        assert_eq!(reopened.edge_count(), 4);
    }

    /// A direct CRUD method has only a boolean failure channel. If its WAL
    /// framing fails inside a caller-owned transaction, it must fail closed by
    /// aborting that transaction rather than leave a proper prefix of the
    /// incident-edge deletes staged in memory.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn explicit_transaction_wal_failure_aborts_partial_detach() {
        use grafeo_common::testing::wal_failure::{
            disable_mutation_log_failure, enable_mutation_log_failure_once,
        };

        let dir = tempfile::tempdir().expect("temporary database directory");
        let path = dir.path().join("explicit-detach-wal-failure");
        let db = open_wal_directory(&path);
        let fixture = seed_detach_fixture(&db);
        let mut writer = db.session();
        writer
            .begin_transaction()
            .expect("begin explicit transaction");

        enable_mutation_log_failure_once();
        let deleted = writer.delete_node(fixture.victim);
        disable_mutation_log_failure();

        assert!(!deleted, "WAL mutation failure must fail the detach");
        assert!(
            !writer.in_transaction(),
            "a boolean-returning detach cannot leave a failed partial statement committable"
        );
        assert!(db.is_durability_poisoned());
        assert_fixture_intact(&db, fixture);
    }
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;
