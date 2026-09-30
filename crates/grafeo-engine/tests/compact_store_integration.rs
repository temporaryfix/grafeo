//! Integration tests for CompactStore through GrafeoDB::with_read_store() + GQL.
//!
//! Requires features: `compact-store` + `gql` (default) for query execution.
//!
//! Validates that CompactStore works end-to-end as an external read-only store:
//! queries are planned and executed against CompactStore data through the same
//! session interface as LpgStore.

#![cfg(feature = "compact-store")]

use std::sync::Arc;

use grafeo_core::graph::compact::CompactStoreBuilder;
use grafeo_core::graph::traits::GraphStoreSearch;
use grafeo_engine::{Config, GrafeoDB};

/// Build a CompactStore with test data and wrap it in GrafeoDB::with_store().
fn build_test_db() -> GrafeoDB {
    let scores: Vec<u64> = (0..10).map(|i| (i % 5) + 1).collect();
    let names: Vec<&str> = vec![
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    ];

    let ratings: Vec<u64> = (0..50).map(|i| (i % 5) + 1).collect();

    // Each of 50 activities links to one of 10 items.
    let activity_to_item: Vec<(u32, u32)> = (0..50).map(|i| (i, i % 10)).collect();

    let store = CompactStoreBuilder::new()
        .node_table("Item", |t| {
            t.column_bitpacked("score", &scores, 4)
                .column_dict("name", &names)
        })
        .node_table("Activity", |t| t.column_bitpacked("rating", &ratings, 4))
        .rel_table("ACTIVITY_ON", "Activity", "Item", |r| {
            r.edges(activity_to_item).backward(true)
        })
        .build()
        .expect("CompactStore build failed");

    GrafeoDB::with_read_store(
        Arc::new(store) as Arc<dyn GraphStoreSearch>,
        Config::default(),
    )
    .expect("GrafeoDB::with_read_store failed")
}

// ── Basic scan queries ──────────────────────────────────────────

#[test]
fn match_all_items() {
    let db = build_test_db();
    let session = db.session();
    let result = session.execute("MATCH (n:Item) RETURN n").unwrap();
    assert_eq!(result.rows().len(), 10);
}

#[test]
fn match_all_activities() {
    let db = build_test_db();
    let session = db.session();
    let result = session.execute("MATCH (n:Activity) RETURN n").unwrap();
    assert_eq!(result.rows().len(), 50);
}

// ── Property access ──────────────────────────────────────────────

#[test]
fn return_property() {
    let db = build_test_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 10);
    // Verify we get string values back
    let first_name = &result.rows()[0][0];
    assert!(
        matches!(first_name, grafeo_common::types::Value::String(_)),
        "Expected string property, got {:?}",
        first_name
    );
}

// ── Edge traversal ───────────────────────────────────────────────

#[test]
fn traverse_outgoing() {
    let db = build_test_db();
    let session = db.session();
    let result = session
        .execute("MATCH (a:Activity)-[:ACTIVITY_ON]->(i:Item) RETURN a, i")
        .unwrap();
    // 50 activities, each with one ACTIVITY_ON edge
    assert_eq!(result.rows().len(), 50);
}

#[test]
fn traverse_incoming() {
    let db = build_test_db();
    let session = db.session();
    let result = session
        .execute("MATCH (i:Item)<-[:ACTIVITY_ON]-(a:Activity) RETURN i, a")
        .unwrap();
    // Same 50 edges, traversed from the other direction
    assert_eq!(result.rows().len(), 50);
}

// ── Aggregation ──────────────────────────────────────────────────

#[test]
fn count_per_label() {
    let db = build_test_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN count(n) AS cnt")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], grafeo_common::types::Value::Int64(10));
}

// ── Read-only enforcement ───────────────────────────────────────

#[test]
fn create_rejected_on_read_only_store() {
    let db = build_test_db();
    let session = db.session();

    // Mutations should be rejected on a read-only store
    let result = session.execute("CREATE (:Item {name: 'lambda'})");
    assert!(result.is_err(), "CREATE should fail on a read-only store");
}

// ── GrafeoDB::compact() ────────────────────────────────────────

#[test]
fn compact_is_repeatable_preserving_cold_and_hot_history() {
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_core::graph::traits::GraphStore;

    let mut db = GrafeoDB::new_in_memory();
    let cold = db.create_node(&["Cold"]);
    let untouched = db.create_node(&["Untouched"]);
    db.set_node_property(cold, "value", Value::Int64(1))
        .unwrap();
    let first_epoch = db.current_epoch();
    db.compact().expect("initial compaction");
    let generation_view = db.layered_store().unwrap();
    assert!(db.create_projection(
        "cold-view",
        grafeo_core::graph::ProjectionSpec::new().with_node_labels(["Cold"]),
    ));

    let hot = db.create_node(&["Hot"]);
    let edge = db.create_edge(cold, hot, "CONNECTS");
    db.set_node_property(cold, "value", Value::Int64(2))
        .unwrap();
    let second_epoch = db.current_epoch();
    for _ in 0..2 {
        db.compact()
            .expect("compaction is repeatable, even with an empty overlay");
        assert_eq!((db.node_count(), db.edge_count()), (3, 1));
        assert!(db.get_node(untouched).is_some());
        assert_eq!(generation_view.base_store().node_count(), 3);
        assert_eq!(db.projection("cold-view").unwrap().node_count(), 1);
        assert_eq!(db.current_epoch(), second_epoch);
        assert_eq!(db.get_edge(edge).unwrap().dst, hot);
        for (epoch, value) in [(first_epoch, 1), (second_epoch, 2)] {
            assert_eq!(
                db.get_node_at_epoch(cold, epoch)
                    .unwrap()
                    .properties
                    .get(&PropertyKey::new("value")),
                Some(&Value::Int64(value))
            );
        }
    }
    db.set_node_property(hot, "value", Value::Int64(3)).unwrap();
    assert_eq!(
        db.get_node(hot)
            .unwrap()
            .properties
            .get(&PropertyKey::new("value")),
        Some(&Value::Int64(3))
    );
}

#[test]
fn compact_rejects_a_live_session_that_would_retain_the_old_store() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:BeforeCompact)").unwrap();
    let session = db.session();

    let error = db
        .compact()
        .expect_err("compaction must not strand a live Session on the old overlay");
    assert!(error.to_string().contains("drop 1 live Session"));
    assert_eq!(
        session
            .execute("MATCH (n:BeforeCompact) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        grafeo_common::types::Value::Int64(1),
        "a rejected compaction must leave the original store untouched"
    );

    drop(session);
    db.compact().expect("retry after dropping the stale handle");
}

#[test]
fn recompact_rejects_live_sessions_and_preserves_overlay_data() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Base)").unwrap();
    db.compact().unwrap();
    db.execute("INSERT (:Overlay)").unwrap();
    let session = db.session();

    assert!(
        db.compact().is_err(),
        "recompaction must not replace the overlay held by a live Session"
    );
    assert_eq!(
        session.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0],
        grafeo_common::types::Value::Int64(2)
    );

    drop(session);
    db.compact().expect("retry after dropping Session");
    assert_eq!(db.node_count(), 2);
}

#[test]
fn compact_reads_survive() {
    let mut db = GrafeoDB::new_in_memory();

    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();
    db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
    db.execute(
        "MATCH (p:Person {name: 'Alix'}), (c:City {name: 'Amsterdam'}) \
         INSERT (p)-[:LIVES_IN]->(c)",
    )
    .unwrap();
    db.execute(
        "MATCH (p:Person {name: 'Gus'}), (c:City {name: 'Amsterdam'}) \
         INSERT (p)-[:LIVES_IN]->(c)",
    )
    .unwrap();

    db.compact().unwrap();

    // Verify read queries still work.
    let session = db.session();
    let persons = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(persons.rows().len(), 2);

    let cities = session.execute("MATCH (c:City) RETURN c.name").unwrap();
    assert_eq!(cities.rows().len(), 1);

    // Verify edge traversal.
    let edges = session
        .execute("MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.name, c.name")
        .unwrap();
    assert_eq!(edges.rows().len(), 2);
}

#[test]
fn compact_then_write() {
    let mut db = GrafeoDB::new_in_memory();

    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    db.compact().unwrap();

    // Writes should succeed on the layered store.
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows().len(), 3);

    let names: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|row| row[0].as_str().map(|s| s.to_string()))
        .collect();
    assert_eq!(names, vec!["Alix", "Gus", "Vincent"]);
}

#[test]
fn compact_preserves_bool_and_string_properties() {
    let mut db = GrafeoDB::new_in_memory();

    db.execute("INSERT (:Item {name: 'alpha', active: true})")
        .unwrap();
    db.execute("INSERT (:Item {name: 'beta', active: false})")
        .unwrap();

    db.compact().unwrap();

    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.name, n.active ORDER BY n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);

    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::String(arcstr::literal!("alpha"))
    );
    assert_eq!(result.rows()[0][1], grafeo_common::types::Value::Bool(true));

    assert_eq!(
        result.rows()[1][0],
        grafeo_common::types::Value::String(arcstr::literal!("beta"))
    );
    assert_eq!(
        result.rows()[1][1],
        grafeo_common::types::Value::Bool(false)
    );
}

#[test]
fn compact_empty_database() {
    let mut db = GrafeoDB::new_in_memory();
    db.compact().unwrap();

    let session = db.session();
    let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
    assert_eq!(result.rows()[0][0], grafeo_common::types::Value::Int64(0));

    // Write to empty compacted database.
    session.execute("INSERT (:Node {val: 1})").unwrap();
    let after = session.execute("MATCH (n) RETURN count(n)").unwrap();
    assert_eq!(after.rows()[0][0], grafeo_common::types::Value::Int64(1));
}

#[test]
fn recompact_merges_overlay() {
    let mut db = GrafeoDB::new_in_memory();

    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.compact().unwrap();

    // Write to the overlay.
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

    // Recompact: merge overlay into base.
    db.compact().unwrap();

    // All data should be in the merged base now.
    let session = db.session();
    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows().len(), 3);

    // Continue writing after recompact.
    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    let after = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(after.rows()[0][0], grafeo_common::types::Value::Int64(4));
}

/// Temporal recompaction: `recompact()` folds the overlay's committed history
/// into a **temporal** cold base, so as-of reads survive compaction.
///
/// Writes a node property across two committed epochs (V1 at e1, V2 at e2),
/// recompacts, and asserts the cold base preserved both versions: an as-of read
/// at e1 returns V1 and at e2 returns V2. Before temporal recompaction the base
/// was all-open, so an as-of read at e1 would have returned V2 (the current
/// value) — that is the bug this proves fixed. Also checks that current reads
/// and node/edge counts are unchanged across `recompact()`.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn recompact_preserves_temporal_history_in_cold_base() {
    use grafeo_common::types::Value;
    use grafeo_core::graph::traits::GraphStore;

    let mut db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Create the node and set its first value (V1) in a committed transaction.
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session.commit().unwrap();
    drop(session);

    // Compact to the columnar base (first compact is already temporal).
    db.compact().unwrap();
    let e1 = db.current_epoch();

    // Resolve the node's id (preserved across compaction).
    let nodes = db.graph_store().nodes_by_label("Person");
    assert_eq!(nodes.len(), 1);
    let node = nodes[0];

    // Update the property to V2 in a later committed transaction (new epoch).
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let e2 = db.current_epoch();
    assert!(
        e2.as_u64() > e1.as_u64(),
        "epoch must advance between writes"
    );

    // Counts before recompaction.
    let nodes_before = db.graph_store().node_count();
    let edges_before = db.graph_store().edge_count();

    // Recompact: fold overlay history into a temporal cold base.
    db.compact().unwrap();

    // The store routes as-of reads through the (now temporal) cold base.
    let store = db.graph_store();
    let age = grafeo_common::types::PropertyKey::new("age");

    // Cold base preserved history: V1 at e1, V2 at e2.
    let at_e1 = store
        .get_node_at_epoch(node, e1)
        .and_then(|n| n.properties.get(&age).cloned());
    assert_eq!(
        at_e1,
        Some(Value::Int64(30)),
        "as-of read at e1 must return V1 (30) from the temporal cold base"
    );
    let at_e2 = store
        .get_node_at_epoch(node, e2)
        .and_then(|n| n.properties.get(&age).cloned());
    assert_eq!(
        at_e2,
        Some(Value::Int64(31)),
        "as-of read at e2 must return V2 (31)"
    );

    // Current read is the latest value, and counts are unchanged.
    assert_eq!(store.get_node_property(node, &age), Some(Value::Int64(31)));
    assert_eq!(store.node_count(), nodes_before);
    assert_eq!(store.edge_count(), edges_before);

    // The cold base alone (overlay reset) serves the node's current state.
    let base = db.layered_store().unwrap().base_store();
    assert_eq!(base.get_node_property(node, &age), Some(Value::Int64(31)));
    assert_eq!(
        base.get_node_property_at_epoch(node, &age, e1),
        Some(Value::Int64(30)),
        "the temporal base itself holds the historical value at e1"
    );

    // Writes still work after recompaction.
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let after = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(after.rows()[0][0], Value::Int64(2));
}

#[test]
fn named_graphs_survive_compact_and_recompact() {
    let mut db = GrafeoDB::new_in_memory();

    assert!(db.create_graph("europe").unwrap());
    assert!(db.create_graph("asia").unwrap());

    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.compact().unwrap();

    let mut names = db.list_graphs();
    names.sort();
    assert_eq!(names, vec!["asia".to_string(), "europe".to_string()]);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("europe")
            .is_some()
    );
    db.set_current_graph(Some("asia")).unwrap();
    db.set_current_graph(None).unwrap();

    db.compact().unwrap();

    let mut names = db.list_graphs();
    names.sort();
    assert_eq!(names, vec!["asia".to_string(), "europe".to_string()]);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("europe")
            .is_some()
    );

    assert!(db.drop_graph("europe").expect("drop graph"));
    assert_eq!(db.list_graphs(), vec!["asia".to_string()]);
}

/// Change 2: the **first** `compact()` is now temporal — it folds the source
/// store's committed history into the cold base instead of dropping it.
///
/// Writes V1 (age 30) at e1 and V2 (age 31) at e2, then calls `compact()`
/// *once* (NO `recompact()`). Before this change the first base was all-open,
/// so an as-of read at e1 returned the current value (31); now the temporal
/// cold base preserves the history and an as-of read at e1 returns 30. The
/// read goes straight through `db.get_node_at_epoch` (Change 3 routing).
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn first_compact_preserves_temporal_history() {
    use grafeo_common::types::{PropertyKey, Value};

    let mut db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let e1 = db.current_epoch();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let e2 = db.current_epoch();
    assert!(
        e2.as_u64() > e1.as_u64(),
        "epoch must advance between writes"
    );

    let node = db.graph_store().nodes_by_label("Person")[0];

    // ONLY compact() — no recompact. The first base must already be temporal.
    db.compact().unwrap();

    let age = PropertyKey::new("age");
    let at_e1 = db
        .get_node_at_epoch(node, e1)
        .and_then(|n| n.properties.get(&age).cloned());
    assert_eq!(
        at_e1,
        Some(Value::Int64(30)),
        "first compact() must preserve V1 (30) at e1 in the temporal cold base"
    );
    let at_e2 = db
        .get_node_at_epoch(node, e2)
        .and_then(|n| n.properties.get(&age).cloned());
    assert_eq!(
        at_e2,
        Some(Value::Int64(31)),
        "as-of read at e2 must return V2"
    );

    // The cold base itself (overlay was reset by the temporal fold) holds it.
    let base = db.layered_store().unwrap().base_store();
    assert_eq!(
        base.get_node_property_at_epoch(node, &age, e1),
        Some(Value::Int64(30)),
        "temporal cold base built by the first compact() holds history at e1"
    );

    // Writes still work after the first (temporal) compaction.
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let after = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(after.rows()[0][0], Value::Int64(2));
}

/// The whole-state `scrub_at_epoch` API reflects cold-base history: after a
/// temporal `compact()`, the columnar scrub at an earlier epoch holds that
/// epoch's value, not the latest.
#[test]
#[cfg(all(feature = "compact-store", feature = "lpg"))]
fn scrub_at_epoch_reflects_cold_base_history() {
    use grafeo_common::types::{PropertyKey, Value};

    let mut db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let e1 = db.current_epoch();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let e2 = db.current_epoch();

    let node = db.graph_store().nodes_by_label("Person")[0];
    db.compact().unwrap(); // temporal cold base

    let age = PropertyKey::new("age");
    let scrub_val = |epoch| -> Option<Value> {
        db.scrub_at_epoch(epoch).nodes.iter().find_map(|f| {
            f.node_ids
                .iter()
                .position(|n| *n == node)
                .and_then(|i| f.columns.get(&age).and_then(|c| c[i].clone()))
        })
    };
    assert_eq!(
        scrub_val(e1),
        Some(Value::Int64(30)),
        "whole-state scrub at e1 must reflect history (V1=30)"
    );
    assert_eq!(
        scrub_val(e2),
        Some(Value::Int64(31)),
        "whole-state scrub at e2 must reflect V2=31"
    );
}

/// Change 3: the convenience `db.get_node_at_epoch()` routes through the
/// `LayeredStore`, so it sees history folded into the cold base across a
/// `compact()` + writes + `recompact()` cycle.
///
/// Before this change the method read only `lpg_store()` (the overlay), which
/// after `recompact()` is empty for the cold node — so it returned the latest
/// value (or nothing), missing the cold history. Now it returns V1 at e1.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn db_get_node_at_epoch_sees_cold_base_history() {
    use grafeo_common::types::{PropertyKey, Value};

    let mut db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session.commit().unwrap();
    drop(session);

    db.compact().unwrap();
    let e1 = db.current_epoch();
    let node = db.graph_store().nodes_by_label("Person")[0];

    // Write V2 to the overlay in a later epoch, then recompact: now BOTH
    // versions live in the cold base and the overlay is reset.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let e2 = db.current_epoch();

    db.compact().unwrap();

    // The convenience method now combines cold-base-as-of + overlay.
    let age = PropertyKey::new("age");
    let at_e1 = db
        .get_node_at_epoch(node, e1)
        .and_then(|n| n.properties.get(&age).cloned());
    assert_eq!(
        at_e1,
        Some(Value::Int64(30)),
        "db.get_node_at_epoch must return cold-base history (V1=30) at e1"
    );
    let at_e2 = db
        .get_node_at_epoch(node, e2)
        .and_then(|n| n.properties.get(&age).cloned());
    assert_eq!(
        at_e2,
        Some(Value::Int64(31)),
        "db.get_node_at_epoch at e2 -> V2"
    );
}

/// Change 1: `compact_if_needed()` self-maintains the cold tier once the overlay
/// exceeds the configured threshold, preserving history.
///
/// With the threshold set low, compact() then write enough committed nodes to
/// exceed it. `compact_if_needed()` returns `true`, the overlay shrinks (its
/// committed history folded into the cold base), and an as-of read of the
/// historical value still works through the (now temporal) base.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn compact_if_needed_triggers_above_threshold_and_preserves_history() {
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_engine::Config;

    let mut db =
        GrafeoDB::with_config(Config::in_memory().with_compaction_overlay_threshold(3)).unwrap();

    // Seed one node and compact so we are in layered mode with a cold base.
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    db.compact().unwrap();
    let e1 = db.current_epoch();
    let alix = db.graph_store().nodes_by_label("Person")[0];

    // Below threshold: no recompaction yet.
    assert!(
        !db.compact_if_needed().unwrap(),
        "overlay below threshold must not recompact"
    );

    // Commit enough writes to the overlay to exceed the threshold (3).
    for name in ["Gus", "Vincent", "Jules", "Mia"] {
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }
    let overlay_before = {
        let view = db.layered_store().unwrap();
        view.overlay_node_count() + view.overlay_edge_count()
    };
    assert!(
        overlay_before > 3,
        "overlay must exceed threshold before trigger"
    );

    // Trigger fires: returns true and folds the overlay into the cold base.
    assert!(
        db.compact_if_needed().unwrap(),
        "overlay above threshold must recompact"
    );
    let overlay_after = {
        let view = db.layered_store().unwrap();
        view.overlay_node_count() + view.overlay_edge_count()
    };
    assert!(
        overlay_after < overlay_before,
        "overlay must shrink after recompaction ({overlay_before} -> {overlay_after})"
    );

    // History preserved + reads correct: all 5 people visible, Alix's old age
    // still readable at e1 from the temporal cold base.
    let count = db
        .execute("MATCH (p:Person) RETURN count(p)")
        .unwrap()
        .rows()[0][0]
        .clone();
    assert_eq!(count, Value::Int64(5));

    let age = PropertyKey::new("age");
    let at_e1 = db
        .get_node_at_epoch(alix, e1)
        .and_then(|n| n.properties.get(&age).cloned());
    assert_eq!(
        at_e1,
        Some(Value::Int64(30)),
        "history preserved across auto-recompact"
    );

    // Idempotent right after a fold: overlay is back below threshold.
    assert!(
        !db.compact_if_needed().unwrap(),
        "freshly-recompacted overlay is below threshold"
    );
}
