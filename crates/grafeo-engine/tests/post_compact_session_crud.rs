//! Regression tests: the **session** direct CRUD API (`session.get_node`,
//! `session.set_node_property`, `session.delete_node`, …) must keep working on
//! base-resident entities after [`GrafeoDB::compact`].
//!
//! Same root cause as the DB-level surface (`post_compact_imperative_crud`): the
//! session's direct lookup/mutation helpers routed through `active_lpg_store()` —
//! the raw overlay — instead of the tier-merged read view / the layered
//! `GraphStoreMut`. After compact a base entity is invisible to the overlay, so
//! reads returned `None` and writes were silently dropped. GQL (`execute`) was
//! unaffected because it already routes through `active_write_store()`.
//!
//! Setup runs through the (already-fixed) `db.*` API; the assertions exercise the
//! `session.*` direct API.
//!
//! ```bash
//! cargo nextest run -p grafeo-engine --features "compact-store lpg gql" \
//!     --test post_compact_session_crud
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn session_get_node_on_base_node_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    db.compact().expect("compact");

    let s = db.session();
    let n = s
        .get_node(a)
        .expect("session.get_node must read a base node after compact");
    assert!(n.labels.iter().any(|l| l.as_str() == "A"));
    assert_eq!(s.get_node_property(a, "v"), Some(Value::Int64(1)));
}

#[test]
fn session_get_edge_on_base_edge_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let e = db.create_edge(a, b, "T");
    db.compact().expect("compact");

    let s = db.session();
    let edge = s
        .get_edge(e)
        .expect("session.get_edge must read a base edge after compact");
    assert_eq!(edge.edge_type.as_str(), "T");
}

#[test]
fn session_direct_traversal_keeps_a_snapshot_of_compacted_topology() {
    let mut db = GrafeoDB::new_in_memory();
    let source = db.create_node(&["Source"]);
    let target = db.create_node(&["Target"]);
    let edge = db.create_edge(source, target, "BASE");
    db.compact().expect("compact");

    let mut reader = db.session();
    reader.begin_transaction().expect("begin reader");
    assert_eq!(reader.get_neighbors_outgoing(source), vec![(target, edge)]);

    let mut writer = db.session();
    writer.begin_transaction().expect("begin writer");
    assert!(writer.delete_edge(edge), "delete compacted base edge");
    writer.commit().expect("commit writer");

    assert!(
        reader.get_edge(edge).is_some(),
        "point reads retain a base edge deleted after BEGIN"
    );
    assert_eq!(
        reader.get_neighbors_outgoing(source),
        vec![(target, edge)],
        "outgoing adjacency retains the transaction snapshot"
    );
    assert_eq!(
        reader.get_neighbors_incoming(target),
        vec![(source, edge)],
        "incoming adjacency retains the transaction snapshot"
    );
    assert_eq!(
        reader.get_neighbors_outgoing_by_type(source, "BASE"),
        vec![(target, edge)]
    );
    assert_eq!(reader.get_degree(source), (1, 0));

    reader.rollback().expect("rollback reader");
    assert!(
        db.session().get_neighbors_outgoing(source).is_empty(),
        "a fresh snapshot observes the committed deletion"
    );
}

#[test]
fn session_set_property_on_base_node_persists_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    db.compact().expect("compact");

    let s = db.session();
    s.set_node_property(a, "v", Value::Int64(2))
        .expect("set_node_property");

    // Visible through the session direct read …
    assert_eq!(s.get_node_property(a, "v"), Some(Value::Int64(2)));
    // … and through the DB read path / GQL.
    assert_eq!(
        db.get_node(a).and_then(|n| n
            .properties
            .get(&grafeo_common::types::PropertyKey::new("v"))
            .cloned()),
        Some(Value::Int64(2))
    );
}

#[test]
fn session_delete_base_node_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["A"]);
    db.compact().expect("compact");

    let s = db.session();
    assert!(s.delete_node(a), "session.delete_node on base node");
    assert!(s.get_node(a).is_none(), "deleted base node not readable");
    assert!(s.get_node(b).is_some(), "sibling base node survives");
    // The DB read path agrees.
    assert!(db.get_node(a).is_none());
}

#[test]
fn session_set_property_on_base_node_in_explicit_transaction() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    db.compact().expect("compact");

    // The buffered (transactional) write path promotes the base-resident node,
    // the direct read observes its own pending write (read-your-writes — see
    // `session_read_your_writes` for the general guarantee), and the write
    // survives commit.
    let mut s = db.session();
    s.begin_transaction().expect("begin");
    s.set_node_property(a, "v", Value::Int64(2))
        .expect("set in txn");
    assert_eq!(
        s.get_node_property(a, "v"),
        Some(Value::Int64(2)),
        "read-your-writes within the transaction"
    );
    s.commit().expect("commit");

    // Persisted after commit, via the DB read path and a fresh session.
    assert_eq!(
        db.get_node(a).and_then(|n| n
            .properties
            .get(&grafeo_common::types::PropertyKey::new("v"))
            .cloned()),
        Some(Value::Int64(2)),
        "transactional write to a base node must persist after commit"
    );
    assert_eq!(
        db.session().get_node_property(a, "v"),
        Some(Value::Int64(2))
    );
}
