//! Regression: the direct session point-read APIs (`get_node`,
//! `get_node_property`, `get_edge`, …) must observe the session's own pending
//! writes inside an explicit transaction (read-your-writes).
//!
//! Pre-fix: a direct mutation inside a transaction is *buffered* (staged in the
//! per-transaction write-set), but the direct point reads went through
//! `get_node_versioned`, which read the committed version chain and never
//! consulted the buffer — so they returned the last committed value
//! mid-transaction. GQL reads (which apply the buffer during execution) were
//! correct, and the write always persisted after commit; only the direct
//! read-your-writes was stale.

#![cfg(feature = "lpg")]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::{GrafeoDB, transaction::IsolationLevel};

#[test]
fn direct_get_node_property_reads_own_uncommitted_write() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");

    let mut s = db.session();
    s.begin_transaction().expect("begin");
    s.set_node_property(a, "v", Value::Int64(2)).expect("set");

    assert_eq!(
        s.get_node_property(a, "v"),
        Some(Value::Int64(2)),
        "read-your-writes: the direct point read must see the pending write"
    );

    s.commit().expect("commit");
    assert_eq!(
        db.session().get_node_property(a, "v"),
        Some(Value::Int64(2))
    );
}

#[test]
fn direct_get_node_reads_own_uncommitted_new_property() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);

    let mut s = db.session();
    s.begin_transaction().expect("begin");
    s.set_node_property(a, "name", Value::String("alix".into()))
        .expect("set");

    let node = s.get_node(a).expect("node visible to its own txn");
    assert_eq!(
        node.properties
            .get(&grafeo_common::types::PropertyKey::new("name"))
            .cloned(),
        Some(Value::String("alix".into())),
        "read-your-writes via get_node"
    );

    s.commit().expect("commit");
}

#[test]
fn direct_get_edge_reads_own_uncommitted_write() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let e = db.create_edge(a, b, "T");
    db.set_edge_property(e, "w", Value::Int64(1))
        .expect("set edge property");

    let mut s = db.session();
    s.begin_transaction().expect("begin");
    s.set_edge_property(e, "w", Value::Int64(2)).expect("set");

    let edge = s.get_edge(e).expect("edge visible to its own txn");
    assert_eq!(
        edge.properties
            .get(&grafeo_common::types::PropertyKey::new("w"))
            .cloned(),
        Some(Value::Int64(2)),
        "read-your-writes via get_edge"
    );

    s.commit().expect("commit");
}

#[test]
fn direct_get_nodes_batch_reads_own_uncommitted_write() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");

    let mut s = db.session();
    s.begin_transaction().expect("begin");
    s.set_node_property(a, "v", Value::Int64(2)).expect("set");

    let batch = s.get_nodes_batch(&[a]);
    assert_eq!(
        batch[0].as_ref().and_then(|n| n
            .properties
            .get(&grafeo_common::types::PropertyKey::new("v"))
            .cloned()),
        Some(Value::Int64(2)),
        "read-your-writes via get_nodes_batch"
    );

    s.commit().expect("commit");
}

#[test]
fn direct_traversal_reads_hold_the_transaction_snapshot() {
    let db = GrafeoDB::new_in_memory();
    let setup = db.session();
    let source = setup.create_node(&["Source"]);
    let original_target = setup
        .create_node_with_props(&["Target"], [("version", Value::Int64(1))])
        .expect("create original target");
    let late_target = setup.create_node(&["LateTarget"]);
    let original = setup.create_edge(source, original_target, "ORIGINAL");

    let mut reader = db.session();
    reader.begin_transaction().expect("begin reader snapshot");

    assert_eq!(
        reader.get_neighbors_outgoing(source),
        vec![(original_target, original)]
    );
    assert_eq!(
        reader.get_neighbors_incoming(original_target),
        vec![(source, original)]
    );
    assert_eq!(reader.get_degree(source), (1, 0));

    // Publish an insertion, a deletion, and a property update after the
    // reader's snapshot. Every direct API below must continue to observe the
    // same pre-commit cut: no late edge, the deleted edge still present, and
    // the old node materialization.
    let mut writer = db.session();
    writer.begin_transaction().expect("begin writer");
    let late = writer.create_edge(source, late_target, "LATE");
    assert!(late.is_valid(), "writer must create the late edge");
    assert!(
        writer.delete_edge(original),
        "writer must delete original edge"
    );
    writer
        .set_node_property(original_target, "version", Value::Int64(2))
        .expect("update target");
    writer.commit().expect("commit writer");

    assert!(
        reader.get_edge(original).is_some(),
        "an edge deleted after the snapshot remains visible"
    );
    assert!(
        reader.get_edge(late).is_none(),
        "an edge inserted after the snapshot stays invisible"
    );
    assert_eq!(
        reader.get_neighbors_outgoing(source),
        vec![(original_target, original)],
        "outgoing traversal must be a repeatable snapshot read"
    );
    assert_eq!(
        reader.get_neighbors_incoming(original_target),
        vec![(source, original)],
        "incoming traversal must retain an edge deleted after BEGIN"
    );
    assert_eq!(
        reader.get_neighbors_outgoing_by_type(source, "ORIGINAL"),
        vec![(original_target, original)]
    );
    assert!(
        reader
            .get_neighbors_outgoing_by_type(source, "LATE")
            .is_empty()
    );
    assert_eq!(reader.get_degree(source), (1, 0));
    assert_eq!(reader.get_degree(original_target), (0, 1));

    let batch = reader.get_nodes_batch(&[original_target, late_target]);
    assert_eq!(
        batch[0]
            .as_ref()
            .and_then(|node| node.properties.get(&PropertyKey::new("version"))),
        Some(&Value::Int64(1)),
        "batch point reads must use the same transaction snapshot"
    );

    reader.rollback().expect("rollback read-only snapshot");
    let current = db.session();
    assert_eq!(
        current.get_neighbors_outgoing(source),
        vec![(late_target, late)],
        "a new session sees the writer's committed topology"
    );
    assert_eq!(
        current.get_node_property(original_target, "version"),
        Some(Value::Int64(2))
    );
}

#[test]
fn direct_traversal_reads_own_edge_creates_and_deletes() {
    let db = GrafeoDB::new_in_memory();
    let setup = db.session();
    let source = setup.create_node(&["Source"]);
    let old_target = setup.create_node(&["OldTarget"]);
    let own_target = setup.create_node(&["OwnTarget"]);
    let old = setup.create_edge(source, old_target, "OLD");

    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let own = session.create_edge(source, own_target, "OWN");
    assert!(session.delete_edge(old));

    assert_eq!(
        session.get_neighbors_outgoing(source),
        vec![(own_target, own)],
        "adjacency must overlay the transaction's pending create and delete"
    );
    assert_eq!(
        session.get_neighbors_outgoing_by_type(source, "OWN"),
        vec![(own_target, own)]
    );
    assert_eq!(session.get_degree(source), (1, 0));

    session.rollback().expect("rollback");
    assert_eq!(
        db.session().get_neighbors_outgoing(source),
        vec![(old_target, old)],
        "rolling back restores the original topology"
    );
}

#[test]
fn serializable_trackers_follow_a_named_graph_selected_after_begin() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.create_graph("accounts").expect("create named graph"));

    let setup = db.session();
    setup
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["accounts"])
                .expect("literal graph path"),
        )
        .expect("select existing graph");
    let account_a = setup
        .create_node_with_props(&["Account"], [("balance", Value::Int64(100))])
        .expect("create account A");
    let account_b = setup
        .create_node_with_props(&["Account"], [("balance", Value::Int64(100))])
        .expect("create account B");

    // Deliberately begin on the default graph, then switch. The named graph
    // must receive this transaction's SSI read/write tracker bridges when it
    // becomes part of the transaction.
    let mut s1 = db.session();
    let mut s2 = db.session();
    s1.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin s1");
    s2.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin s2");
    s1.use_graph_path(
        &grafeo_common::types::GraphPath::from_components(&["accounts"])
            .expect("literal graph path"),
    )
    .expect("select existing graph");
    s2.use_graph_path(
        &grafeo_common::types::GraphPath::from_components(&["accounts"])
            .expect("literal graph path"),
    )
    .expect("select existing graph");

    for session in [&s1, &s2] {
        assert_eq!(
            session.get_node_property(account_a, "balance"),
            Some(Value::Int64(100))
        );
        assert_eq!(
            session.get_node_property(account_b, "balance"),
            Some(Value::Int64(100))
        );
    }

    s1.set_node_property(account_a, "balance", Value::Int64(-100))
        .expect("s1 write A");
    s2.set_node_property(account_b, "balance", Value::Int64(-100))
        .expect("s2 write B");

    s1.commit().expect("first committer succeeds");
    let second = s2.commit();
    assert!(
        second.is_err(),
        "SSI must reject named-graph write skew after a post-BEGIN graph switch"
    );
    assert!(
        second
            .unwrap_err()
            .to_string()
            .contains("Serialization failure"),
        "the second committer must fail specifically at Serializable validation"
    );
}

// The label branch of the delta overlay: a buffered label add (via GQL `SET n:L`
// in a transaction) must be visible to a direct `get_node` read in that txn.
#[cfg(feature = "gql")]
#[test]
fn direct_get_node_reads_own_uncommitted_label() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);

    let mut s = db.session();
    s.begin_transaction().expect("begin");
    s.execute("MATCH (n:A) SET n:B").expect("gql set label");

    let node = s.get_node(a).expect("node visible to its own txn");
    assert!(
        node.labels.iter().any(|l| l.as_str() == "B"),
        "read-your-writes for a buffered label add: {:?}",
        node.labels
    );

    s.commit().expect("commit");
}
