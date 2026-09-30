//! Transaction and named-graph lifecycle guarantees for virtual LPG projections.
//!
//! These tests cover `CREATE PROJECTION` / `DROP PROJECTION`, whose registry
//! entries are lightweight virtual graph views. They deliberately do not
//! exercise the durable RDF→LPG projection and reconciliation subsystem.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::sync::Arc;

use grafeo_engine::GrafeoDB;

#[test]
fn create_projection_rollback_never_publishes_registry_entry() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("START TRANSACTION").unwrap();
    session
        .execute("CREATE PROJECTION rolled_back LABELS (Person)")
        .unwrap();

    assert!(
        db.projection("rolled_back").is_none(),
        "an uncommitted projection must remain session-private"
    );

    session.execute("ROLLBACK").unwrap();
    assert!(
        db.projection("rolled_back").is_none(),
        "rollback must discard the staged projection"
    );
}

#[test]
fn drop_projection_rollback_preserves_committed_registry_entry() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE PROJECTION kept LABELS (Person)")
        .unwrap();
    let original = db.projection("kept").expect("committed projection");

    session.execute("START TRANSACTION").unwrap();
    session.execute("DROP PROJECTION kept").unwrap();

    assert!(
        db.projection("kept").is_some(),
        "an uncommitted drop must not hide the committed registry entry"
    );

    session.execute("ROLLBACK").unwrap();
    let restored = db
        .projection("kept")
        .expect("rollback must preserve the committed projection");
    assert!(
        Arc::ptr_eq(&original, &restored),
        "rollback must restore the exact committed projection handle"
    );
}

#[test]
fn projection_create_savepoint_restores_the_prior_post_image() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("START TRANSACTION").unwrap();
    session
        .execute("CREATE PROJECTION retained LABELS (Person)")
        .unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session
        .execute("CREATE PROJECTION discarded LABELS (City)")
        .unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("COMMIT").unwrap();

    assert!(
        db.projection("retained").is_some(),
        "the projection staged before the savepoint must commit"
    );
    assert!(
        db.projection("discarded").is_none(),
        "rollback to savepoint must discard later projection creation"
    );
}

#[test]
fn projection_drop_savepoint_restores_the_exact_entry() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE PROJECTION retained LABELS (Person)")
        .unwrap();
    let original = db.projection("retained").expect("committed projection");

    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session.execute("DROP PROJECTION retained").unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("COMMIT").unwrap();

    let restored = db
        .projection("retained")
        .expect("savepoint rollback must restore the projection");
    assert!(
        Arc::ptr_eq(&original, &restored),
        "savepoint rollback must restore the exact registry entry"
    );
}

#[test]
fn failed_commit_does_not_publish_default_graph_projection() {
    let db = GrafeoDB::new_in_memory();
    db.create_graph("doomed").unwrap();
    let writer = db.session();

    writer.execute("START TRANSACTION").unwrap();
    writer
        .execute("CREATE PROJECTION must_abort LABELS (Root)")
        .unwrap();
    writer.execute("USE GRAPH doomed").unwrap();
    writer.execute("INSERT (:Pinned {value: 1})").unwrap();

    assert!(
        db.drop_graph("doomed").expect("drop graph"),
        "the concurrent graph drop must win the lifecycle race"
    );
    writer
        .execute("COMMIT")
        .expect_err("the writer pinned to the dropped graph must fail commit");

    assert!(
        db.projection("must_abort").is_none(),
        "a failed transaction must not leak its unrelated default-graph projection"
    );
}

#[test]
fn drop_graph_cascades_its_projection_registry_entries() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE GRAPH source").unwrap();
    session.execute("USE GRAPH source").unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session
        .execute("CREATE PROJECTION source_people LABELS (Person)")
        .unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();

    session.execute("DROP GRAPH source").unwrap();

    assert!(
        db.projection("source_people").is_none(),
        "dropping a graph must unregister projections owned by that graph incarnation"
    );
}

#[test]
fn held_projection_handle_remains_a_readable_detached_snapshot_after_drop() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE GRAPH source").unwrap();
    session.execute("USE GRAPH source").unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session
        .execute("CREATE PROJECTION source_people LABELS (Person)")
        .unwrap();
    let held = db
        .projection("source_people")
        .expect("published projection handle");
    assert_eq!(held.node_count(), 1);
    session.execute("SESSION RESET GRAPH").unwrap();

    session.execute("DROP GRAPH source").unwrap();

    assert!(db.projection("source_people").is_none());
    assert_eq!(
        held.node_count(),
        1,
        "an acquired Arc must keep its exact detached source incarnation readable"
    );
}

#[test]
fn drop_recreate_preserves_same_named_projection_on_replacement_incarnation() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE GRAPH replaceable").unwrap();
    session.execute("USE GRAPH replaceable").unwrap();
    session.execute("INSERT (:Original {value: 1})").unwrap();
    session
        .execute("CREATE PROJECTION current_view LABELS (Original)")
        .unwrap();
    let original = db
        .projection("current_view")
        .expect("projection on original graph incarnation");
    session.execute("SESSION RESET GRAPH").unwrap();

    session.execute("START TRANSACTION").unwrap();
    session.execute("DROP GRAPH replaceable").unwrap();
    session.execute("CREATE GRAPH replaceable").unwrap();
    session.execute("USE GRAPH replaceable").unwrap();
    session.execute("INSERT (:Replacement {value: 2})").unwrap();
    session
        .execute("CREATE PROJECTION current_view LABELS (Replacement)")
        .expect("the staged cascade must make the projection name reusable");
    session.execute("COMMIT").unwrap();

    let replacement = db
        .projection("current_view")
        .expect("replacement projection must survive the old graph's cascade");
    assert!(
        !Arc::ptr_eq(&original, &replacement),
        "the registry must publish a new projection handle"
    );
    assert_eq!(replacement.nodes_by_label_count("Original"), 0);
    assert_eq!(replacement.nodes_by_label_count("Replacement"), 1);
    assert_eq!(original.nodes_by_label_count("Original"), 1);
    assert_eq!(original.nodes_by_label_count("Replacement"), 0);
}

#[test]
fn create_projection_rejects_a_missing_active_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let missing = grafeo_common::types::GraphPath::from_components(&["missing"]).unwrap();
    session
        .use_graph_path(&missing)
        .expect_err("missing native target is rejected");
    assert!(session.current_graph_path().components().is_empty());
    assert!(db.create_graph("missing").unwrap());
    session.use_graph_path(&missing).unwrap();
    assert!(db.drop_graph("missing").expect("drop graph"));

    session
        .execute("CREATE PROJECTION orphan LABELS (Person)")
        .expect_err("projection creation must require a registered source graph");

    assert!(
        db.projection("orphan").is_none(),
        "a missing graph must never produce a projection over a fallback store"
    );
}

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]
#[test]
fn wal_commit_append_failure_never_publishes_virtual_projection_registry_state() {
    use grafeo_common::testing::wal_failure::{
        disable_commit_log_failure, enable_commit_log_failure_once,
    };
    use grafeo_engine::{Config, DurabilityMode, GraphModel};

    let directory = tempfile::TempDir::new().unwrap();
    let path = directory
        .path()
        .join("virtual-projection-commit-failure.grafeo");
    {
        let db = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Lpg)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        let session = db.session();
        session.execute("START TRANSACTION").unwrap();
        session
            .execute("CREATE PROJECTION must_not_publish LABELS (Person)")
            .unwrap();

        enable_commit_log_failure_once();
        let error = session
            .execute("COMMIT")
            .expect_err("the commit-frame append must fail");
        disable_commit_log_failure();

        assert!(error.to_string().contains("outcome is unknown"), "{error}");
        assert!(db.is_durability_poisoned());
        assert!(
            db.projection("must_not_publish").is_none(),
            "a failed durable boundary must not leak the prepared runtime registry post-image"
        );
    }

    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    assert!(
        reopened.projection("must_not_publish").is_none(),
        "process-local virtual registrations are not manufactured by recovery"
    );
}
