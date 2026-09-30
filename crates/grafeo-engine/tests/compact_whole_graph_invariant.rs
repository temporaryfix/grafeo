//! Invariant: every whole-graph surface of `GrafeoDB` must report the *tier-merged*
//! graph after `compact()`, not the (empty) raw overlay.
//!
//! `compact()` folds all data into the read-only columnar base and installs a
//! fresh empty overlay. Any whole-graph read/serialize that consults only the
//! overlay therefore sees an EMPTY graph — silent data loss. This locks the
//! contract for `node_count`/`edge_count`, `to_memory`, `export_snapshot`/`import`,
//! and `save`/reopen, with GQL (always tier-merged) as the oracle.

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::{EdgeId, EpochId, EpochInterval, NodeId, Value};
use grafeo_engine::GrafeoDB;

fn gql_count(db: &GrafeoDB, q: &str) -> i64 {
    match &db.session().execute(q).unwrap().rows()[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64, got {other:?}"),
    }
}

fn assert_exact_edge_history(
    db: &GrafeoDB,
    edge: EdgeId,
    endpoints: (NodeId, NodeId),
    lifetime: (EpochId, Option<EpochId>),
) {
    let history = db.graph_store().get_edge_history(edge);
    assert_eq!(history.len(), 1);
    assert_eq!((history[0].0, history[0].1), lifetime);
    assert_eq!(history[0].2.id, edge);
    assert_eq!((history[0].2.src, history[0].2.dst), endpoints);
    assert_eq!(history[0].2.edge_type.as_str(), "T");
    assert!(db.get_edge_at_epoch(edge, lifetime.0).is_some());
    assert_eq!(db.get_edge(edge).is_some(), lifetime.1.is_none());
    if let Some(deleted) = lifetime.1 {
        assert!(db.get_edge_at_epoch(edge, deleted).is_none());
    }
}

#[test]
fn whole_graph_surfaces_survive_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.set_node_property(a, "v", Value::Int64(1))
        .expect("set node property");
    let edge = db.create_edge(a, b, "T");
    assert!(edge.is_valid());
    db.compact().expect("compact");

    // Counts.
    assert_eq!(db.node_count(), 2, "node_count after compact");
    assert_eq!(db.edge_count(), 1, "edge_count after compact");

    // GQL oracle agrees.
    assert_eq!(gql_count(&db, "MATCH (n) RETURN count(n)"), 2);
    assert_eq!(gql_count(&db, "MATCH ()-[r]->() RETURN count(r)"), 1);

    // to_memory() round-trip preserves the whole graph.
    let mem = db.to_memory().expect("to_memory");
    assert_eq!(mem.node_count(), 2, "to_memory preserves nodes");
    assert_eq!(mem.edge_count(), 1, "to_memory preserves edges");
    assert_eq!(
        gql_count(&mem, "MATCH (n:A) WHERE n.v = 1 RETURN count(n)"),
        1
    );

    // export_snapshot() -> import_snapshot() round-trip.
    let bytes = db.export_snapshot().expect("export_snapshot");
    let imp = GrafeoDB::import_snapshot(&bytes).expect("import_snapshot");
    assert_eq!(imp.node_count(), 2, "export/import preserves nodes");
    assert_eq!(imp.edge_count(), 1, "export/import preserves edges");
    assert_eq!(
        gql_count(&imp, "MATCH (a:A)-[:T]->(b:B) RETURN count(*)"),
        1
    );
}

#[test]
fn compact_sparse_and_closed_edge_histories_survive_copy_and_recompact() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let retired = db.create_edge(a, b, "T");
    assert!(retired.is_valid());
    let retired_created = db.current_epoch();
    assert!(db.delete_edge(retired));
    let retired_deleted = db.current_epoch();
    let edge = db.create_edge(a, b, "T");
    assert!(edge.is_valid());
    assert_ne!(edge, retired);
    let edge_created = db.current_epoch();
    db.compact().expect("compact");

    let assert_world = |copy: &GrafeoDB| {
        assert_eq!(copy.current_epoch(), edge_created);
        assert_eq!(copy.node_count(), 2);
        assert_eq!(copy.edge_count(), 1);
        assert_exact_edge_history(
            copy,
            retired,
            (a, b),
            (retired_created, Some(retired_deleted)),
        );
        assert_exact_edge_history(copy, edge, (a, b), (edge_created, None));
        assert_eq!(
            gql_count(copy, "MATCH (a:A)-[:T]->(b:B) RETURN count(*)"),
            1
        );
    };
    for generation in 0..2 {
        assert_world(&db);
        let base = db.layered_store().unwrap().base_store();
        assert_eq!(base.live_original_edge_ids(), vec![edge]);
        assert_eq!(
            base.structural_edge_rows(),
            vec![
                (
                    retired,
                    EpochInterval::closed(retired_created, retired_deleted)
                ),
                (edge, EpochInterval::open(edge_created)),
            ]
        );
        assert_world(
            &db.to_memory()
                .expect("fork preserves sparse and closed identities"),
        );
        let bytes = db.export_snapshot().expect("export exact history");
        assert_world(&GrafeoDB::import_snapshot(&bytes).expect("import exact history"));
        if generation == 0 {
            db.compact().expect("recompact preserves exact history");
        }
    }
}
