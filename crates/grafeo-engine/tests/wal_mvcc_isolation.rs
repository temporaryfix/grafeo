//! MVCC isolation probes on **persistent WAL-wrapped** stores.
//!
//! In-memory `mvcc_isolation` tests exercise bare `LpgStore` (no `WalGraphStore`).
//! On persistent DBs the session wraps the write store in [`WalGraphStore`], and
//! Cypher mutations route through `active_write_store()` → trait-default
//! write-through for `*_buffered` (Option B; see `wal_store.rs` TODO(unified-mvcc)).
//!
//! These probes assert the invariant: an uncommitted Cypher `SET` must be
//! invisible to other sessions even with WAL enabled. This holds because
//! `WalGraphStore` overrides the `*_buffered` writes to delegate to the inner
//! store's overlay delta (instead of trait-default write-through), while still
//! logging each mutation to the WAL immediately.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test wal_mvcc_isolation
//! ```

#![cfg(all(feature = "lpg", feature = "wal", feature = "gql"))]
#![allow(missing_docs)]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// Opens a persistent DB with WAL enabled (default for `Config::persistent`).
fn open_persistent(path: &std::path::Path) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::persistent(path)).expect("open persistent db");
    assert!(
        db.is_persistent(),
        "probe requires a persistent WAL-wrapped session store"
    );
    db
}

/// Mirrors `mvcc_isolation::uncommitted_set_is_invisible_to_other_sessions` on a
/// persistent WAL-backed database using the Cypher query path (not session-direct
/// `set_node_property`, which bypasses `WalGraphStore` via `active_lpg_store()`).
#[test]
fn persistent_wal_cypher_uncommitted_set_invisible_to_others() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("wal_mvcc.grafeo");

    let db = open_persistent(&path);
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .expect("seed");

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
    let r = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .expect("reader read");
    let seen = r.rows()[0][0].clone();

    writer.rollback().expect("rollback");

    assert_eq!(
        seen,
        Value::Int64(30),
        "uncommitted Cypher SET must not be visible to other sessions on a \
         WAL-wrapped persistent store (WalGraphStore must buffer its `*_buffered` \
         writes into the inner overlay, not write through to the committed column)"
    );

    let after = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .expect("post-rollback read");
    assert_eq!(
        after.rows()[0][0].clone(),
        Value::Int64(30),
        "rollback must restore committed age=30"
    );

    db.close().expect("close");
}

/// Control: the in-memory equivalent of the probe above must pass (bare LpgStore).
#[test]
fn in_memory_cypher_uncommitted_set_invisible_to_others_control() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .expect("seed");

    writer.begin_transaction().expect("begin");
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .expect("set");

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
        "in-memory control must stay isolated"
    );
}

/// Session-direct API bypasses `WalGraphStore` (`active_lpg_store()`). This probe
/// documents that the bypass path **does** buffer correctly even on persistent WAL —
/// isolating the gap to the Cypher / `active_write_store()` route only.
#[test]
fn persistent_wal_session_direct_set_still_isolated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("wal_mvcc_direct.grafeo");

    let db = open_persistent(&path);
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .expect("seed");

    // Resolve node id via a read (committed).
    let id = {
        let r = writer
            .execute("MATCH (p:Person {name: 'Ann'}) RETURN id(p)")
            .expect("id lookup");
        match &r.rows()[0][0] {
            Value::Int64(id) => grafeo_common::types::NodeId::new(
                u64::try_from(*id).expect("node id is non-negative"),
            ),
            other => panic!("expected Int64 node id, got {other:?}"),
        }
    };

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
        "session-direct set_node_property must buffer via active_lpg_store even on WAL"
    );

    db.close().expect("close");
}
