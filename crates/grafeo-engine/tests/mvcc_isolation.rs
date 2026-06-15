//! Snapshot isolation for transactional property writes and deletes
//! (unified-MVCC increment 1).
#![cfg(feature = "lpg")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn uncommitted_set_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .unwrap();

    // Read-your-writes: the writer sees 99.
    let own = writer
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    assert_eq!(
        own.rows()[0][0].clone(),
        Value::Int64(99),
        "writer must see its own write"
    );

    // Another session must still see the committed value (30).
    let reader = db.session();
    let r = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    let seen = r.rows()[0][0].clone();

    writer.rollback().unwrap();
    assert_eq!(
        seen,
        Value::Int64(30),
        "uncommitted SET must not be visible to other sessions"
    );

    // After rollback the committed value is unchanged.
    let after = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    assert_eq!(
        after.rows()[0][0].clone(),
        Value::Int64(30),
        "rollback restores committed value"
    );
}

#[test]
fn committed_set_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann', age: 30})")
        .unwrap();
    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.age")
        .unwrap();
    assert_eq!(
        r.rows()[0][0].clone(),
        Value::Int64(99),
        "committed SET must be visible"
    );
}

#[test]
fn uncommitted_delete_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) DETACH DELETE p")
        .unwrap();

    // Writer no longer sees Ann (read-your-writes for delete).
    let own = writer
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        own.row_count(),
        0,
        "writer must not see its own deleted node"
    );

    // Another session must still see Ann.
    let reader = db.session();
    let during = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name")
        .unwrap();
    let visible_during = during.row_count();

    writer.rollback().unwrap();
    let after = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name")
        .unwrap();

    assert_eq!(
        visible_during, 1,
        "uncommitted delete must not be visible to other sessions"
    );
    assert_eq!(
        after.row_count(),
        1,
        "node must exist after rollback of delete"
    );
}

#[test]
fn committed_delete_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) DETACH DELETE p")
        .unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p.name")
        .unwrap();
    assert_eq!(r.row_count(), 0, "committed delete must be visible");
}

#[test]
fn writer_sees_own_set_via_filter_not_just_projection() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.execute("CREATE (:Person {name: 'Ann', age: 30})")
        .unwrap();
    w.begin_transaction().unwrap();
    w.execute("MATCH (p:Person {name: 'Ann'}) SET p.age = 99")
        .unwrap();
    let r = w
        .execute("MATCH (p:Person) WHERE p.age = 99 RETURN p.name")
        .unwrap();
    assert_eq!(
        r.row_count(),
        1,
        "writer must match its own uncommitted SET in WHERE"
    );
    w.rollback().unwrap();
}

/// Uncommitted DETACH DELETE of a node WITH edges must not tombstone edges for
/// other readers (edge-adjacency isolation).
///
/// Currently `delete_node_edges` uses `TransactionId::SYSTEM` with eager
/// `batch_mark_deleted`, so the adjacency tombstone is immediately visible to
/// other sessions regardless of the deleting transaction's commit status.
/// Fixing this requires threading `transaction_id` into `delete_node_edges` and
/// deferring adjacency removal — left as a follow-up (unified-mvcc increment 2).
///
/// TODO(unified-mvcc): defer adjacency tombstones for transactional DETACH.
#[test]
#[ignore = "known bug: delete_node_edges uses eager adjacency tombstones (TransactionId::SYSTEM); adjacency isolation deferred to unified-mvcc increment 2"]
fn uncommitted_detach_delete_edges_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann'})-[:KNOWS]->(:Person {name: 'Bob'})")
        .unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) DETACH DELETE p")
        .unwrap();

    // Another session must still see Ann's edges (delete not committed).
    let reader = db.session();
    let r = reader
        .execute("MATCH (:Person {name: 'Bob'})<-[:KNOWS]-(p) RETURN p.name")
        .unwrap();

    writer.rollback().unwrap();

    assert_eq!(
        r.row_count(),
        1,
        "uncommitted DETACH DELETE must not tombstone edges for other sessions"
    );
}
