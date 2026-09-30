//! Public as-of graph API (A2): Application-facing neighbors/edges/scrub without
//! CompactStore internals.
//!
//! After `compact()` (temporal merge): `PENDING` == current CSR; a closed
//! edge is absent from current neighbors but visible as-of before delete.

#![cfg(all(feature = "compact-store", feature = "lpg"))]

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_core::graph::Direction;
use grafeo_engine::GrafeoDB;

fn bump_epoch(db: &GrafeoDB) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.commit().unwrap();
}

fn sort_ids(mut ids: Vec<NodeId>) -> Vec<NodeId> {
    ids.sort_unstable();
    ids
}

fn edge_ids(edges: &[grafeo_core::graph::lpg::Edge]) -> Vec<EdgeId> {
    let mut ids: Vec<EdgeId> = edges.iter().map(|e| e.id).collect();
    ids.sort_unstable();
    ids
}

fn node_ids(nodes: &[grafeo_core::graph::lpg::Node]) -> Vec<NodeId> {
    let mut ids: Vec<NodeId> = nodes.iter().map(|node| node.id).collect();
    ids.sort_unstable();
    ids
}

fn node_lifetimes(db: &GrafeoDB, id: NodeId) -> Vec<(EpochId, Option<EpochId>)> {
    db.get_node_history(id)
        .into_iter()
        .map(|(created, deleted, _)| (created, deleted))
        .collect()
}

fn edge_lifetimes(db: &GrafeoDB, id: EdgeId) -> Vec<(EpochId, Option<EpochId>)> {
    db.get_edge_history(id)
        .into_iter()
        .map(|(created, deleted, _)| (created, deleted))
        .collect()
}

/// Current scans deliberately omit closed identities, but an historical cut
/// must enumerate every identity whose lifetime contains that cut even before
/// a compact base exists.
#[test]
fn live_asof_enumeration_includes_entities_deleted_after_the_cut() {
    let db = GrafeoDB::new_in_memory();
    let source = db.create_node(&["Source"]);
    let target = db.create_node(&["Target"]);
    let edge = db.create_edge(source, target, "LINKS");
    let visible = db.current_epoch();

    assert!(db.delete_node(source), "delete source and incident edge");
    assert_eq!(node_ids(&db.nodes_at_epoch(visible)), vec![source, target]);
    assert_eq!(edge_ids(&db.edges_at_epoch(visible)), vec![edge]);
    assert!(db.get_node(source).is_none());
    assert!(db.get_edge(edge).is_none());
}

/// Compaction may change the physical tier, never the public temporal answer:
/// structural lives and complete property logs remain available for identities
/// that are no longer part of the current graph.
#[test]
fn public_entity_and_property_history_survives_temporal_compaction() {
    let mut db = GrafeoDB::new_in_memory();
    let source =
        db.create_node_with_props(&["Source"], [("phase", Value::String("created".into()))]);
    let target = db.create_node(&["Target"]);
    let edge = db.create_edge_with_props(source, target, "LINKS", [("weight", Value::Int64(1))]);

    db.set_node_property(source, "phase", Value::String("updated".into()))
        .expect("set node property");
    assert!(db.remove_node_property(source, "phase"));
    db.set_edge_property(edge, "weight", Value::Int64(2))
        .expect("set edge property");
    let visible = db.current_epoch();

    assert!(db.delete_node(source), "delete source and incident edge");
    let expected_node_lifetimes = node_lifetimes(&db, source);
    let expected_edge_lifetimes = edge_lifetimes(&db, edge);
    let expected_phase_history = db.get_node_property_history(source, "phase");
    let expected_all_properties = db.get_all_node_property_history(source);
    assert!(!expected_node_lifetimes.is_empty());
    assert!(!expected_edge_lifetimes.is_empty());
    assert_eq!(
        expected_phase_history.last().map(|(_, value)| value),
        Some(&Value::Null),
        "property removal is retained as a temporal tombstone"
    );

    db.compact().expect("temporal compact");

    assert_eq!(node_ids(&db.nodes_at_epoch(visible)), vec![source, target]);
    assert_eq!(edge_ids(&db.edges_at_epoch(visible)), vec![edge]);
    assert_eq!(node_lifetimes(&db, source), expected_node_lifetimes);
    assert_eq!(edge_lifetimes(&db, edge), expected_edge_lifetimes);
    assert_eq!(
        db.get_node_property_history(source, "phase"),
        expected_phase_history
    );
    assert_eq!(
        db.get_all_node_property_history(source),
        expected_all_properties
    );
}

/// Delete-then-compact: packed closed life is as-of only (merge fixture).
#[test]
fn neighbors_at_epoch_pending_equals_current_and_closed_absent_after_delete() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let c = db.create_node(&["Person"]);
    db.set_node_property(a, "name", Value::from("A"))
        .expect("set node property");
    db.set_node_property(b, "name", Value::from("B"))
        .expect("set node property");
    db.set_node_property(c, "name", Value::from("C"))
        .expect("set node property");

    let live = db.create_edge_with_props(a, b, "KNOWS", [("w", Value::Int64(1))]);
    let dead = db.create_edge_with_props(a, c, "KNOWS", [("w", Value::Int64(2))]);
    let e_open = db.current_epoch();

    bump_epoch(&db);
    assert!(db.delete_edge(dead));
    let e_del = db.current_epoch();
    bump_epoch(&db);

    db.compact().expect("compact");

    let current = sort_ids(db.graph_store().neighbors(a, Direction::Outgoing));
    let pending = sort_ids(db.neighbors_at_epoch(a, Direction::Outgoing, EpochId::PENDING));
    assert_eq!(
        pending, current,
        "PENDING == current CSR (open prefix + overlay)"
    );
    assert_eq!(pending, vec![b], "deleted edge not in current neighbors");

    let mut before = db.neighbors_at_epoch(a, Direction::Outgoing, e_open);
    before.sort_unstable();
    let mut expect_open = vec![b, c];
    expect_open.sort_unstable();
    assert_eq!(
        before, expect_open,
        "as-of before delete includes the closed edge"
    );

    assert_eq!(
        db.neighbors_at_epoch(a, Direction::Outgoing, e_del),
        vec![b],
        "closed edge absent at/after delete epoch"
    );

    assert_eq!(edge_ids(&db.edges_at_epoch(EpochId::PENDING)), vec![live]);
    let mut open_edges = edge_ids(&db.edges_at_epoch(e_open));
    open_edges.sort_unstable();
    let mut expect_edges = vec![live, dead];
    expect_edges.sort_unstable();
    assert_eq!(open_edges, expect_edges);
    assert_eq!(edge_ids(&db.edges_at_epoch(e_del)), vec![live]);

    let nodes = db.nodes_at_epoch(e_open);
    assert_eq!(nodes.len(), 3);

    let scrub = db.scrub_at_epoch(e_open);
    assert!(!scrub.nodes.is_empty(), "node frames present after compact");
    let mut scrub_edge_ids: Vec<EdgeId> = scrub
        .edges
        .iter()
        .flat_map(|f| f.edge_ids.iter().copied())
        .collect();
    scrub_edge_ids.sort_unstable();
    assert_eq!(
        scrub_edge_ids, expect_edges,
        "scrub_at_epoch includes edges, not only nodes"
    );

    let current_scrub = db.scrub_at_epoch(EpochId::PENDING);
    let mut current_scrub_ids: Vec<EdgeId> = current_scrub
        .edges
        .iter()
        .flat_map(|f| f.edge_ids.iter().copied())
        .collect();
    current_scrub_ids.sort_unstable();
    assert_eq!(current_scrub_ids, vec![live]);
}

/// Compact first, then overlay-delete without recompact: as-of must still
/// hide the dest at the delete epoch (cold CSR is still open; tombstone is hot).
#[test]
fn neighbors_at_epoch_overlay_delete_without_recompact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let c = db.create_node(&["Person"]);
    let _live = db.create_edge_with_props(a, b, "KNOWS", [("w", Value::Int64(1))]);
    let dead = db.create_edge_with_props(a, c, "KNOWS", [("w", Value::Int64(2))]);
    let e_open = db.current_epoch();

    db.compact().expect("compact");

    bump_epoch(&db);
    assert!(db.delete_edge(dead));
    let e_del = db.current_epoch();
    bump_epoch(&db);

    let pending = sort_ids(db.neighbors_at_epoch(a, Direction::Outgoing, EpochId::PENDING));
    let current = sort_ids(db.graph_store().neighbors(a, Direction::Outgoing));
    assert_eq!(pending, current);
    assert_eq!(pending, vec![b], "deleted dest gone from current/PENDING");

    assert_eq!(
        sort_ids(db.neighbors_at_epoch(a, Direction::Outgoing, e_del)),
        pending,
        "neighbors_at_epoch(e_del) == PENDING after overlay delete, no recompact"
    );

    let mut before = db.neighbors_at_epoch(a, Direction::Outgoing, e_open);
    before.sort_unstable();
    let mut expect_open = vec![b, c];
    expect_open.sort_unstable();
    assert_eq!(
        before, expect_open,
        "as-of before delete still includes the closed dest"
    );
}
