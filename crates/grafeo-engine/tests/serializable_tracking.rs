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
use grafeo_engine::{GrafeoDB, transaction::EntityId};

// ============================================================================
// Helper
// ============================================================================

fn assert_node_in_ws(
    ws: &std::collections::HashSet<EntityId>,
    id: NodeId,
    label: &str,
) {
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
        .set_node_property(prop_label_target, "x", grafeo_common::types::Value::Int64(42))
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
