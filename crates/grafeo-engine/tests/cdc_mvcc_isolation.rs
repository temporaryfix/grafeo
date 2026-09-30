//! MVCC isolation probes on **persistent, CDC-wrapped** stores.
//!
//! Mirrors `wal_mvcc_isolation` but with CDC enabled, so the session write
//! store is `CdcGraphStore` → `WalGraphStore` → `LpgStore`. Cypher mutations
//! route through `active_write_store()` = the outer `CdcGraphStore`; if it does
//! not override `*_buffered`, the trait default chains to write-through and
//! bypasses the WAL wrapper's buffered isolation entirely.
//!
//! These probes assert that an uncommitted Cypher `SET` is invisible to other
//! sessions even with CDC enabled, and that no CDC `Update` event reaches the
//! committed `CdcLog` until the transaction commits (it stays in the per-tx
//! pending buffer, flushed at commit / discarded on rollback).

#![cfg(all(feature = "lpg", feature = "wal", feature = "gql", feature = "cdc"))]
#![allow(missing_docs)]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::cdc::ChangeKind;
use grafeo_engine::{Config, GrafeoDB};

fn open_persistent_cdc(path: &std::path::Path) -> GrafeoDB {
    let db =
        GrafeoDB::with_config(Config::persistent(path).with_cdc()).expect("open persistent cdc db");
    assert!(
        db.is_persistent(),
        "probe requires a persistent WAL+CDC-wrapped session store"
    );
    db
}

/// Resolve the `Ann` node id via a committed read.
fn ann_id(db: &GrafeoDB) -> NodeId {
    let s = db.session();
    let r = s
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN id(p)")
        .expect("id lookup");
    match &r.rows()[0][0] {
        Value::Int64(id) => NodeId::new(u64::try_from(*id).expect("node id is non-negative")),
        other => panic!("expected Int64 node id, got {other:?}"),
    }
}

/// Cypher `SET` inside an uncommitted transaction must be invisible to other
/// sessions on a persistent CDC-wrapped store, and must not emit a committed
/// CDC `Update` event until commit.
#[test]
fn persistent_cdc_cypher_uncommitted_set_invisible_to_others() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cdc_mvcc.grafeo");

    let db = open_persistent_cdc(&path);
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .expect("seed");
    let id = ann_id(&db);
    // Baseline: the committed seed CREATE emits its own CDC events (incl. property
    // Updates). We assert the *uncommitted* SET adds no further committed event.
    let updates_baseline = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(id))
        .expect("history")
        .into_iter()
        .filter(|e| e.kind == ChangeKind::Update)
        .count();

    writer.begin_transaction().expect("begin");
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .expect("uncommitted set");

    // Read-your-writes on the writer.
    let own = writer
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .expect("writer read");
    assert_eq!(
        own.rows()[0][0].clone(),
        Value::Int64(99),
        "writer must see its own buffered SET"
    );

    // Another session on the same DB must still see the committed value (30).
    let reader = db.session();
    let seen = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .expect("reader read")
        .rows()[0][0]
        .clone();

    // No committed CDC Update event for the uncommitted SET yet (only the
    // committed Create from the seed should be in the log).
    let updates_before = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(id))
        .expect("history")
        .into_iter()
        .filter(|e| e.kind == ChangeKind::Update)
        .count();

    writer.rollback().expect("rollback");

    assert_eq!(
        seen,
        Value::Int64(30),
        "uncommitted Cypher SET must not be visible to other sessions on a \
         CDC-wrapped persistent store (CdcGraphStore must buffer its `*_buffered` \
         writes into the inner overlay, not write through to the committed column)"
    );
    assert_eq!(
        updates_before, updates_baseline,
        "uncommitted Cypher SET must not add a committed CDC Update event before commit"
    );

    // After rollback the committed value is restored.
    let after = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .expect("post-rollback read");
    assert_eq!(after.rows()[0][0].clone(), Value::Int64(30));

    db.close().expect("close");
}

/// Control: session-direct `set_node_property` bypasses `CdcGraphStore` (it goes
/// through `active_lpg_store()`), so it buffers correctly even pre-fix —
/// isolating the gap to the Cypher / `active_write_store()` route only.
#[test]
fn persistent_cdc_session_direct_set_still_isolated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cdc_mvcc_direct.grafeo");

    let db = open_persistent_cdc(&path);
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .expect("seed");
    let id = ann_id(&db);

    writer.begin_transaction().expect("begin");
    writer
        .set_node_property(id, "age", Value::Int64(99))
        .expect("session-direct set");

    let reader = db.session();
    let seen = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .expect("read")
        .rows()[0][0]
        .clone();

    writer.rollback().expect("rollback");
    assert_eq!(
        seen,
        Value::Int64(30),
        "session-direct set_node_property must buffer via active_lpg_store even on CDC"
    );

    db.close().expect("close");
}

/// After commit, the CDC `Update` event is flushed to the committed log,
/// confirming events are deferred to commit (not lost) by the buffered path.
#[test]
fn persistent_cdc_committed_set_emits_event() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cdc_mvcc_commit.grafeo");

    let db = open_persistent_cdc(&path);
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .expect("seed");
    let id = ann_id(&db);

    writer.begin_transaction().expect("begin");
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .expect("set");
    writer.commit().expect("commit");

    let updates: usize = db
        .fixture_changes(grafeo_engine::cdc::EntityHistoryQuery::new(id))
        .expect("history")
        .into_iter()
        .filter(|e| e.kind == ChangeKind::Update)
        .count();
    assert!(
        updates >= 1,
        "committed Cypher SET must emit a CDC Update event"
    );

    db.close().expect("close");
}

#[path = "support/cdc_pages.rs"]
mod cdc_pages;
use cdc_pages::CdcFixtureChanges;
