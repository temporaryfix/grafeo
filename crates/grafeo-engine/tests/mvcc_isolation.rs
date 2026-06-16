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

#[test]
fn uncommitted_label_add_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p:Secret")
        .unwrap();

    // Writer sees its own label (read-your-writes), via has-label and via scan.
    let own = writer.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    assert_eq!(
        own.row_count(),
        1,
        "writer must see its own uncommitted label"
    );

    // Other session must NOT see the :Secret label.
    let reader = db.session();
    let scan = reader.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    let lbls = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN labels(p) AS l")
        .unwrap();
    let seen_scan = scan.row_count();
    let seen_labels = format!("{:?}", lbls.rows()[0][0]);

    writer.rollback().unwrap();
    assert_eq!(
        seen_scan, 0,
        "uncommitted label must not be visible via scan to other sessions"
    );
    assert!(
        !seen_labels.contains("Secret"),
        "uncommitted label must not appear in labels(p) for other sessions: {seen_labels}"
    );

    // After rollback the writer's tx label is gone everywhere.
    let after = reader.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    assert_eq!(after.row_count(), 0, "rolled-back label must not exist");
}

#[test]
fn committed_label_add_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.execute("CREATE (:Person {name: 'Ann'})").unwrap();
    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) SET p:Secret")
        .unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader.execute("MATCH (p:Secret) RETURN p.name").unwrap();
    assert_eq!(
        r.row_count(),
        1,
        "committed label must be visible to other sessions"
    );
}

#[test]
fn uncommitted_label_remove_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person:Vip {name: 'Ann'})")
        .unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) REMOVE p:Vip")
        .unwrap();

    // Other session must still see :Vip (remove not committed).
    let reader = db.session();
    let during = reader.execute("MATCH (p:Vip) RETURN p.name").unwrap();
    let seen_during = during.row_count();

    writer.rollback().unwrap();
    let after = reader.execute("MATCH (p:Vip) RETURN p.name").unwrap();

    assert_eq!(
        seen_during, 1,
        "uncommitted label-remove must not be visible to other sessions"
    );
    assert_eq!(after.row_count(), 1, ":Vip must be restored after rollback");
}

/// Uncommitted DETACH DELETE must not tombstone incident edges for other readers
/// (edge-adjacency isolation, unified-MVCC increment 2b).
///
/// The DETACH operator threads `transaction_id` into edge deletion and stamps
/// `deleted_epoch = PENDING`, deferring the adjacency tombstone to commit. This
/// test is a regression guard: an uncommitted DETACH DELETE must leave the
/// incident edge visible to concurrent readers.
#[test]
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

#[test]
fn uncommitted_edge_delete_is_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:N {id: 1})-[:R {w: 5}]->(:N {id: 2})")
        .unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) DELETE r")
        .unwrap();

    // Writer no longer traverses the edge (read-your-writes).
    let own = writer
        .execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id")
        .unwrap();
    assert_eq!(
        own.row_count(),
        0,
        "writer must not see its own deleted edge"
    );

    // Other session must still traverse a->b.
    let reader = db.session();
    let during = reader
        .execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id")
        .unwrap();
    let visible_during = during.row_count();

    writer.rollback().unwrap();
    let after = reader
        .execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id")
        .unwrap();

    assert_eq!(
        visible_during, 1,
        "uncommitted edge delete must not be visible to other sessions"
    );
    assert_eq!(
        after.row_count(),
        1,
        "edge must exist after rollback of delete"
    );
}

#[test]
fn committed_edge_delete_is_visible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:N {id: 1})-[:R]->(:N {id: 2})")
        .unwrap();
    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) DELETE r")
        .unwrap();
    writer.commit().unwrap();

    let reader = db.session();
    let r = reader
        .execute("MATCH (:N {id: 1})-[r:R]->(b) RETURN b.id")
        .unwrap();
    assert_eq!(r.row_count(), 0, "committed edge delete must be visible");
}

/// A committed transactional DETACH DELETE must remove BOTH the node and its
/// incident edges for every other session (MVCC increment 2b). This locks the
/// node-finalize + edge-finalize interaction on commit: the DETACH operator
/// deletes the incident edge transactionally (PENDING) before the node, and
/// commit must finalize both so a fresh reader sees neither.
#[test]
fn committed_detach_delete_edges_invisible_to_other_sessions() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:Person {name: 'Ann'})-[:KNOWS]->(:Person {name: 'Bob'})")
        .unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (p:Person {name: 'Ann'}) DETACH DELETE p")
        .unwrap();
    writer.commit().unwrap();

    // A fresh reader must see neither the edge nor Ann.
    let reader = db.session();
    let edge = reader
        .execute("MATCH (:Person {name: 'Bob'})<-[:KNOWS]-(p) RETURN p.name")
        .unwrap();
    assert_eq!(
        edge.row_count(),
        0,
        "committed DETACH DELETE must remove the incident edge for other sessions"
    );
    let node = reader
        .execute("MATCH (p:Person {name: 'Ann'}) RETURN p")
        .unwrap();
    assert_eq!(
        node.row_count(),
        0,
        "committed DETACH DELETE must remove the node for other sessions"
    );
}

/// Task 3b: edge-delete isolation for EXISTS/COUNT subquery fast paths.
///
/// `edge_matches` in the filter evaluator post-filters adjacency candidates for
/// both `ExistsSubquery` (`WHERE EXISTS { (a)-[:R]->() }`) and `CountSubquery`
/// (`RETURN COUNT { (a)-[:R]->() }`).  Before the fix it did NOT check edge
/// visibility, so a writer that deleted an edge would still observe it through
/// these paths (read-your-writes violation).
#[test]
fn deleted_edge_invisible_to_writer_via_exists_subquery() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:N {id: 1})-[:R]->(:N {id: 2})")
        .unwrap();

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) DELETE r")
        .unwrap();

    // Writer must NOT see the deleted edge via EXISTS { ... } (read-your-writes).
    let own_exists = writer
        .execute("MATCH (a:N {id: 1}) WHERE EXISTS { (a)-[:R]->() } RETURN a.id")
        .unwrap();
    assert_eq!(
        own_exists.row_count(),
        0,
        "writer must not see its own pending-deleted edge via EXISTS subquery"
    );

    // Writer must NOT see the deleted edge via COUNT { ... } (read-your-writes).
    let own_count = writer
        .execute("MATCH (a:N {id: 1}) RETURN COUNT { (a)-[:R]->() } AS c")
        .unwrap();
    let count_val = own_count.rows()[0][0].clone();
    assert_eq!(
        count_val,
        Value::Int64(0),
        "writer must not see its own pending-deleted edge via COUNT subquery"
    );

    // Another session (no tx) must still see the edge (isolation for others).
    let reader = db.session();
    let reader_exists = reader
        .execute("MATCH (a:N {id: 1}) WHERE EXISTS { (a)-[:R]->() } RETURN a.id")
        .unwrap();
    assert_eq!(
        reader_exists.row_count(),
        1,
        "uncommitted edge delete must not be visible to other sessions via EXISTS"
    );

    let reader_count = reader
        .execute("MATCH (a:N {id: 1}) RETURN COUNT { (a)-[:R]->() } AS c")
        .unwrap();
    let reader_count_val = reader_count.rows()[0][0].clone();
    assert_eq!(
        reader_count_val,
        Value::Int64(1),
        "uncommitted edge delete must not be visible to other sessions via COUNT"
    );

    writer.rollback().unwrap();
}

/// MERGE must not match the writer's own just-deleted edge (MVCC increment 2b).
///
/// `find_matching_edge` resolves adjacency candidates; under the deferred
/// edge-delete model the candidate index still returns a PENDING-deleted edge,
/// so MERGE must post-filter through tx-aware visibility. Otherwise a MERGE
/// issued AFTER deleting that edge in the same tx matches the dead edge, creates
/// nothing, and at commit the delete finalizes — leaving the edge MERGE was
/// supposed to guarantee gone (read-your-writes + MERGE-contract violation).
#[test]
fn merge_does_not_match_writer_own_deleted_edge() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer
        .execute("CREATE (:N {id: 1})-[:R]->(:N {id: 2})")
        .unwrap();

    writer.begin_transaction().unwrap();
    // Delete the edge within the tx.
    writer
        .execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) DELETE r")
        .unwrap();
    // MERGE the same edge: it must NOT match the just-deleted candidate, so it
    // creates a fresh one (read-your-writes).
    writer
        .execute("MATCH (a:N {id: 1}), (b:N {id: 2}) MERGE (a)-[:R]->(b)")
        .unwrap();

    // Within the tx the writer must now see exactly one such edge (the MERGE
    // re-created it because the deleted edge did not match).
    let own = writer
        .execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) RETURN count(r) AS cnt")
        .unwrap();
    assert_eq!(
        own.rows()[0][0].clone(),
        Value::Int64(1),
        "writer must see exactly one edge: MERGE must not match the deleted edge"
    );

    writer.commit().unwrap();

    // A fresh reader must see exactly one edge persist: the original delete
    // finalized AND the MERGE-created edge committed.
    let reader = db.session();
    let after = reader
        .execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) RETURN count(r) AS cnt")
        .unwrap();
    assert_eq!(
        after.rows()[0][0].clone(),
        Value::Int64(1),
        "exactly one edge must persist after commit (deleted finalized, merged kept)"
    );
}

/// Re-deleting an edge already deleted by the same tx must not double-count
/// (MVCC increment 2b). A self-loop `DETACH DELETE` walks the node's outgoing
/// AND incoming adjacency with no self-loop dedup, so the loop edge is deleted
/// twice by the same tx. The second delete must be an idempotent no-op so the
/// pending set carries the edge once and finalize decrements counts once.
///
/// NOTE: this is an end-to-end path guard. The raw `live_edge_count` underflow
/// from the duplicate decrement is masked at the query layer (`edge_count()`
/// re-scans the version chains; the statistics counter is clamped with
/// `.max(0)`), so the authoritative count-integrity RED witness lives at the
/// store level (`edge_delete_transactional_is_idempotent_per_tx`). This test
/// still locks the self-loop DETACH path end-to-end (no panic / mis-traversal).
#[test]
fn self_loop_detach_delete_counts_once() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    // Self-loop: an edge from a node back to itself.
    writer.execute("CREATE (a:N {id: 1})-[:R]->(a)").unwrap();

    // Sanity: exactly one edge exists.
    let before = writer
        .execute("MATCH ()-[r:R]->() RETURN count(r) AS cnt")
        .unwrap();
    assert_eq!(before.rows()[0][0].clone(), Value::Int64(1));

    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (a:N {id: 1}) DETACH DELETE a")
        .unwrap();
    writer.commit().unwrap();

    // After commit the edge count must be exactly 0 (not under-counted / negative
    // from a double decrement).
    let reader = db.session();
    let after = reader
        .execute("MATCH ()-[r:R]->() RETURN count(r) AS cnt")
        .unwrap();
    assert_eq!(
        after.rows()[0][0].clone(),
        Value::Int64(0),
        "self-loop DETACH DELETE must leave exactly zero edges (no double-count)"
    );
}

/// MVCC 2c probe 1: two MERGEs of the same node key in one tx must create exactly one node
/// when a property index exists (triggering the index-based candidate path).
///
/// `find_matching_node` uses the committed property index when `has_property_index` is true.
/// The node created by the first MERGE is written via `set_node_property_buffered` (delta
/// only), so the committed property index is NOT updated → the second MERGE's index lookup
/// finds no candidate → duplicate is created. RED until `find_matching_node` also scans
/// the per-tx delta when the index path is taken.
#[test]
fn merge_twice_in_tx_creates_one_node() {
    let db = GrafeoDB::new_in_memory();
    // Create the property index before any writes so find_matching_node uses the
    // index-based candidate path (has_property_index("k") == true).
    db.create_property_index("k");
    let mut w = db.session();
    w.begin_transaction().unwrap();
    w.execute("MERGE (n:Thing {k: 1})").unwrap();
    w.execute("MERGE (n:Thing {k: 1})").unwrap();
    let own = w.execute("MATCH (n:Thing {k: 1}) RETURN n").unwrap();
    assert_eq!(
        own.row_count(),
        1,
        "two MERGEs of the same key in one tx must create exactly one node"
    );
    w.commit().unwrap();
    let reader = db.session();
    let after = reader.execute("MATCH (n:Thing {k: 1}) RETURN n").unwrap();
    assert_eq!(after.row_count(), 1, "exactly one node after commit");
}

/// MVCC 2c probe 2: MERGE must match a node CREATEd earlier in the same tx when a
/// property index exists (triggering the index-based candidate path).
///
/// `find_matching_node` uses `find_nodes_by_properties` via the committed index when
/// `has_property_index` returns true. A node CREATEd in the same tx writes properties
/// via `set_node_property_buffered` (delta only) — NOT into the committed index — so the
/// index lookup misses it → MERGE duplicates the node. RED until the index path also
/// consults the per-tx delta.
#[test]
fn merge_matches_same_tx_created_node() {
    let db = GrafeoDB::new_in_memory();
    // Create the property index so find_matching_node takes the index path.
    db.create_property_index("k");
    let mut w = db.session();
    w.begin_transaction().unwrap();
    w.execute("CREATE (n:Thing {k: 1})").unwrap();
    w.execute("MERGE (m:Thing {k: 1})").unwrap();
    let own = w.execute("MATCH (n:Thing {k: 1}) RETURN n").unwrap();
    assert_eq!(
        own.row_count(),
        1,
        "MERGE must match the same-tx CREATEd node, not duplicate it"
    );
    w.commit().unwrap();
}

/// MVCC 2c probe 3: MERGE on an edge property must see the value SET earlier in
/// the same tx (read-your-writes), not the committed (old) value.
///
/// `find_matching_edge` reads committed edge properties rather than the per-tx
/// delta → MERGE on the tx-visible updated value finds no match → duplicate edge.
/// RED until `find_matching_edge` reads through the tx delta.
#[test]
fn merge_edge_matches_same_tx_set_property() {
    let db = GrafeoDB::new_in_memory();
    let mut w = db.session();
    w.execute("CREATE (:N {id: 1})-[:R {w: 1}]->(:N {id: 2})")
        .unwrap();
    w.begin_transaction().unwrap();
    w.execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) SET r.w = 5")
        .unwrap();
    // MERGE on the tx-visible value (5) must MATCH the existing edge, not create a second.
    w.execute("MATCH (a:N {id: 1}), (b:N {id: 2}) MERGE (a)-[r:R {w: 5}]->(b)")
        .unwrap();
    let own = w
        .execute("MATCH (:N {id: 1})-[r:R]->(:N {id: 2}) RETURN r")
        .unwrap();
    assert_eq!(
        own.row_count(),
        1,
        "MERGE must match the edge whose prop was SET earlier in the tx, not create a duplicate"
    );
    w.commit().unwrap();
}
