//! Compact-base allocator continuity across an empty-overlay reopen.

#![cfg(all(
    feature = "compact-store",
    feature = "grafeo-file",
    feature = "lpg",
    feature = "wal"
))]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB};

fn bump_epoch(db: &GrafeoDB) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.commit().unwrap();
}

#[test]
fn empty_overlay_reopen_allocates_above_live_nodes_and_closed_edges() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("compact_allocator.grafeo");

    let (old_src, old_dst, closed_edge, edge_open, edge_deleted) = {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let old_src = db.create_node_with_props(&["Person"], [("name", Value::from("old-source"))]);
        let old_dst =
            db.create_node_with_props(&["Person"], [("name", Value::from("old-destination"))]);
        let closed_edge = db.create_edge(old_src, old_dst, "OLD_EDGE");
        let edge_open = db.current_epoch();

        bump_epoch(&db);
        assert!(db.delete_edge(closed_edge));
        let edge_deleted = db.current_epoch();
        bump_epoch(&db);

        db.compact().expect("compact temporal fixture");
        let layered = db.layered_store().expect("compact installs layered store");
        assert_eq!(layered.overlay_node_count(), 0);
        assert_eq!(layered.overlay_edge_count(), 0);
        assert!(
            layered
                .base_store()
                .closed_edge_ids()
                .contains(&closed_edge),
            "the deleted edge id must survive as a cold temporal identity"
        );
        assert!(db.get_edge(closed_edge).is_none());
        assert!(db.get_edge_at_epoch(closed_edge, edge_open).is_some());
        assert!(db.get_edge_at_epoch(closed_edge, edge_deleted).is_none());

        db.close().expect("checkpoint compact database");
        (old_src, old_dst, closed_edge, edge_open, edge_deleted)
    };

    let db = GrafeoDB::open(&path).expect("reopen compact database");
    let layered = db.layered_store().expect("reopen restores layered store");
    assert_eq!(
        layered.overlay_node_count(),
        0,
        "the reopen fixture must exercise a genuinely empty overlay"
    );
    assert_eq!(layered.overlay_edge_count(), 0);

    let old_name = PropertyKey::new("name");
    let old_edge_before_create = db
        .get_edge_at_epoch(closed_edge, edge_open)
        .expect("closed edge remains visible before its delete epoch");
    assert_eq!(old_edge_before_create.src, old_src);
    assert_eq!(old_edge_before_create.dst, old_dst);

    let fresh_node =
        db.create_node_with_props(&["Person"], [("name", Value::from("fresh-after-reopen"))]);
    let fresh_edge = db.create_edge(old_src, fresh_node, "FRESH_EDGE");

    assert!(fresh_node.is_valid());
    assert!(fresh_edge.is_valid());
    assert!(
        fresh_node.as_u64() > old_src.as_u64().max(old_dst.as_u64()),
        "new node id {fresh_node:?} must be above every cold node identity"
    );
    assert!(
        fresh_edge.as_u64() > closed_edge.as_u64(),
        "new edge id {fresh_edge:?} must be above the retained closed id {closed_edge:?}"
    );

    assert_eq!(
        db.get_node(old_src)
            .and_then(|node| node.properties.get(&old_name).cloned()),
        Some(Value::from("old-source")),
        "allocating a fresh node must not shadow the old cold node"
    );
    assert!(
        db.get_edge(closed_edge).is_none(),
        "the old closed edge must remain absent from the current graph"
    );
    let old_edge_after_create = db
        .get_edge_at_epoch(closed_edge, edge_open)
        .expect("fresh allocation must not shadow closed-edge history");
    assert_eq!(old_edge_after_create.src, old_src);
    assert_eq!(old_edge_after_create.dst, old_dst);
    assert!(db.get_edge_at_epoch(closed_edge, edge_deleted).is_none());

    let current_fresh = db.get_edge(fresh_edge).expect("fresh edge is current");
    assert_eq!(current_fresh.src, old_src);
    assert_eq!(current_fresh.dst, fresh_node);
}
