//! Write-set completeness invariant: store-derived complete write-set.
//!
//! Increment 2e (Part E) ensures that every entity touched by a transaction
//! is present in the manager's write-set **before** commit-time validation
//! runs.  Specifically, the write-set is completed from the store's commit
//! chokepoints (creates ∪ deletes ∪ edge-deletes ∪ overlay-touched) right
//! after `touched_graphs` is snapshotted and before `commit` is called.
//!
//! This file tests the invariant: after a transaction that (a) CREATEs a
//! node, (b) SETs a property on a pre-existing node, (c) SETs a label on a
//! pre-existing node, and (d) DELETEs a pre-existing node, the committed
//! write-set must contain all four entities.
//!
//! The write-set derivation is not isolation-gated: it runs for every
//! committed transaction.  We use the default SI session so the test is
//! always active regardless of the Serializable feature flag.

#![cfg(feature = "lpg")]

use grafeo_common::types::NodeId;
use grafeo_engine::{
    GrafeoDB,
    transaction::{EntityId, IsolationLevel},
};

// ============================================================================
// Helper
// ============================================================================

fn assert_node_in_ws(ws: &std::collections::HashSet<EntityId>, id: NodeId, label: &str) {
    assert!(
        ws.contains(&EntityId::Node(id)),
        "write-set must contain {} (id={:?}), but it contains {:?}",
        label,
        id,
        ws,
    );
}

// ============================================================================
// Core invariant test
// ============================================================================

/// After a transaction that touches four entity kinds, the committed write-set
/// (read back from the TransactionManager) must contain all four.
///
/// Breakdown of the four "slots" coming from different chokepoints:
///  - `created`:      node created inside the tx  → pending_tx_creates
///  - `prop_target`:  node whose property was SET  → tx_property_overlay (node_props)
///  - `label_target`: node whose label was SET     → tx_property_overlay (node_labels)
///  - `deleted`:      node that was DETACH DELETE'd → pending_tx_deletes
///
/// We use the same node for the prop+label SET to keep the test simple;
/// the write-set deduplicates so this is fine.
#[test]
fn write_set_complete_from_chokepoints() {
    let db = GrafeoDB::new_in_memory();

    // ── Setup: create pre-existing nodes outside any transaction ────────────
    let setup = db.session();

    // Node whose property and label will be SET inside the tx.
    let prop_label_target = setup
        .create_node_with_props(&["Target"], [("x", grafeo_common::types::Value::Int64(1))])
        .expect("create prop_label_target");

    // Node that will be DELETEd inside the tx.
    let deleted = setup.create_node(&["ToDelete"]);

    drop(setup); // auto-commits (session auto-commit)

    // ── Transaction: four chokepoints ───────────────────────────────────────
    let mut session = db.session();
    session.begin_transaction().unwrap();

    // Capture the tx id while the transaction is still active.
    let tid = session
        .active_transaction_id()
        .expect("transaction must be active");

    // (a) CREATE a new node inside the tx  → pending_tx_creates
    let created = session.create_node(&["Fresh"]);

    // (b) SET a property on pre-existing node  → tx_property_overlay node_props
    session
        .set_node_property(
            prop_label_target,
            "x",
            grafeo_common::types::Value::Int64(42),
        )
        .unwrap();

    // (c) SET a label on the same pre-existing node  → tx_property_overlay node_labels
    // Use execute() since there is no direct add_label API on Session.
    session
        .execute(&format!(
            "MATCH (n) WHERE id(n) = {} SET n:ExtraLabel",
            prop_label_target.as_u64()
        ))
        .unwrap();

    // (d) DELETE the pre-existing node  → pending_tx_deletes
    session.delete_node(deleted);

    // Commit — the derivation block runs before validation.
    session.commit().unwrap();

    // ── Inspect the write-set after commit ──────────────────────────────────
    // The TransactionManager preserves the entry (state = Committed) until GC.
    let ws = session
        .transaction_manager_ref()
        .get_write_set(tid)
        .expect("write-set must be readable after commit");

    // All entities must be present.
    assert_node_in_ws(&ws, created, "created node (pending_tx_creates)");
    assert_node_in_ws(&ws, prop_label_target, "property/label-SET node (overlay)");
    assert_node_in_ws(&ws, deleted, "deleted node (pending_tx_deletes)");

    // Sanity: the write-set is non-empty.
    assert!(
        !ws.is_empty(),
        "write-set must be non-empty after a write transaction"
    );
}

// ============================================================================
// Edge-delete chokepoint
// ============================================================================

/// Validates that a transactionally deleted edge also lands in the write-set
/// via the `pending_tx_edge_deletes` chokepoint.
#[test]
fn write_set_includes_deleted_edge() {
    let db = GrafeoDB::new_in_memory();

    // Create two nodes and an edge outside any transaction.
    let setup = db.session();
    let a = setup.create_node(&["A"]);
    let b = setup.create_node(&["B"]);
    let eid = setup.create_edge(a, b, "REL");
    drop(setup);

    // Delete the edge inside a transaction.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let tid = session.active_transaction_id().unwrap();

    session.delete_edge(eid);

    session.commit().unwrap();

    let ws = session
        .transaction_manager_ref()
        .get_write_set(tid)
        .expect("write-set must be readable after commit");

    assert!(
        ws.contains(&EntityId::Edge(eid)),
        "write-set must contain the deleted edge {:?}, got {:?}",
        eid,
        ws
    );
}

// ============================================================================
// Edge-create chokepoint
// ============================================================================

/// Validates that an edge created inside a transaction lands in the write-set
/// via the `pending_tx_creates` (edge) chokepoint.
#[test]
fn write_set_includes_created_edge() {
    let db = GrafeoDB::new_in_memory();

    // Create endpoints outside transaction.
    let setup = db.session();
    let a = setup.create_node(&["A"]);
    let b = setup.create_node(&["B"]);
    drop(setup);

    // Create the edge inside a transaction.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let tid = session.active_transaction_id().unwrap();

    let eid = session.create_edge(a, b, "REL");

    session.commit().unwrap();

    let ws = session
        .transaction_manager_ref()
        .get_write_set(tid)
        .expect("write-set must be readable after commit");

    assert!(
        ws.contains(&EntityId::Edge(eid)),
        "write-set must contain the created edge {:?}, got {:?}",
        eid,
        ws
    );
}

// ============================================================================
// GE2: coarse Label/RelType fan-out — phantom write chokepoints
// ============================================================================

/// `create_node` inside a transaction fans out a coarse `EntityId::Label(_)` write
/// for each label assigned to the new node.
///
/// This is the load-bearing chokepoint for GE3's phantom detection: an escalated
/// `Label(L)` reader must form an rw-antidependency with any writer that creates
/// a node carrying that label.
#[test]
fn create_label_node_write_set_includes_label() {
    let db = GrafeoDB::new_in_memory();

    let mut session = db.session();
    session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    let tid = session.active_transaction_id().unwrap();

    // Create a node with label "Foo" inside the transaction.
    let _nid = session.create_node(&["Foo"]);

    session.commit().unwrap();

    let ws = session
        .transaction_manager_ref()
        .get_write_set(tid)
        .expect("write-set must be readable after commit");

    // The write-set must contain at least one EntityId::Label(_) entry — the
    // coarse phantom write recorded by create_node_versioned via record_coarse_node_write.
    let label_entries: Vec<_> = ws
        .iter()
        .filter(|e| matches!(e, EntityId::Label(_)))
        .collect();
    assert!(
        !label_entries.is_empty(),
        "write-set must contain EntityId::Label(_) after create_node, but got {:?}",
        ws
    );
}

/// `SET n:L` (add_label_buffered path) inside a transaction also fans out a coarse
/// `EntityId::Label(_)` write for the newly added label.
#[test]
fn add_label_write_set_includes_label() {
    let db = GrafeoDB::new_in_memory();

    // Create a node outside any transaction.
    let setup = db.session();
    let nid = setup.create_node(&["Base"]);
    drop(setup);

    let mut session = db.session();
    session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    let tid = session.active_transaction_id().unwrap();

    // Add a label to the pre-existing node inside the tx.
    session
        .execute(&format!(
            "MATCH (n) WHERE id(n) = {} SET n:NewLabel",
            nid.as_u64()
        ))
        .unwrap();

    session.commit().unwrap();

    let ws = session
        .transaction_manager_ref()
        .get_write_set(tid)
        .expect("write-set must be readable after commit");

    let label_entries: Vec<_> = ws
        .iter()
        .filter(|e| matches!(e, EntityId::Label(_)))
        .collect();
    assert!(
        !label_entries.is_empty(),
        "write-set must contain EntityId::Label(_) after SET n:L, but got {:?}",
        ws
    );
}

/// `create_edge` inside a transaction fans out a coarse `EntityId::RelType(_)` write
/// for the relationship type of the new edge.
///
/// This is the load-bearing chokepoint for GE3's phantom detection on the edge side:
/// an escalated `RelType(T)` reader must form an rw-antidependency with any writer
/// that creates an edge of that type.
#[test]
fn create_edge_write_set_includes_rel_type() {
    let db = GrafeoDB::new_in_memory();

    // Create endpoint nodes outside any transaction.
    let setup = db.session();
    let a = setup.create_node(&["A"]);
    let b = setup.create_node(&["B"]);
    drop(setup);

    let mut session = db.session();
    session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    let tid = session.active_transaction_id().unwrap();

    // Create an edge of type "KNOWS" inside the tx.
    let _eid = session.create_edge(a, b, "KNOWS");

    session.commit().unwrap();

    let ws = session
        .transaction_manager_ref()
        .get_write_set(tid)
        .expect("write-set must be readable after commit");

    let rel_type_entries: Vec<_> = ws
        .iter()
        .filter(|e| matches!(e, EntityId::RelType(_)))
        .collect();
    assert!(
        !rel_type_entries.is_empty(),
        "write-set must contain EntityId::RelType(_) after create_edge, but got {:?}",
        ws
    );
}
