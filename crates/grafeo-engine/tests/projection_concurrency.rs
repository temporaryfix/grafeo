//! Serialization and schema-ownership contracts for virtual LPG projections.
//!
//! These tests deliberately cover only the runtime `CREATE PROJECTION` /
//! `DROP PROJECTION` registry. The durable RDF→LPG projection registry has
//! separate receipt and reconciliation guarantees.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::sync::{Arc, Barrier};
use std::thread;

use grafeo_common::utils::error::{Error, ErrorCode, TransactionError};
use grafeo_core::graph::{GraphStoreSearch, ProjectionSpec, lpg::LpgStore};
#[cfg(feature = "triple-store")]
use grafeo_engine::GraphModel;
use grafeo_engine::transaction::IsolationLevel;
use grafeo_engine::{Config, GrafeoDB, Session};

fn shown_projection_names(session: &Session) -> Vec<String> {
    let result = session
        .execute("SHOW PROJECTIONS")
        .expect("SHOW PROJECTIONS must succeed");
    assert_eq!(result.columns, vec!["name"]);
    result
        .rows()
        .iter()
        .map(|row| {
            row[0]
                .as_str()
                .expect("projection names must be strings")
                .to_owned()
        })
        .collect()
}

#[test]
fn concurrent_same_name_creates_stage_twice_then_have_one_commit_winner() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let start = Arc::new(Barrier::new(2));
    let staged = Arc::new(Barrier::new(2));

    let workers: Vec<_> = (0..2)
        .map(|_| {
            let db = Arc::clone(&db);
            let start = Arc::clone(&start);
            let staged = Arc::clone(&staged);
            thread::spawn(move || {
                let session = db.session();
                session.execute("START TRANSACTION").unwrap();
                start.wait();
                let create = session
                    .execute("CREATE PROJECTION contested LABELS (Person)")
                    .map(|_| ())
                    .map_err(|error| error.to_string());
                // Reach this rendezvous even on an unexpected CREATE error so
                // a regression fails rather than deadlocking the test.
                staged.wait();
                let commit = session
                    .execute("COMMIT")
                    .map(|_| ())
                    .map_err(|error| error.to_string());
                (create, commit)
            })
        })
        .collect();

    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("projection worker panicked"))
        .collect();
    let creates = outcomes.iter().filter(|(create, _)| create.is_ok()).count();
    let commits = outcomes.iter().filter(|(_, commit)| commit.is_ok()).count();

    assert_eq!(
        creates, 2,
        "both snapshots must be allowed to stage the absent name: {outcomes:?}"
    );
    assert_eq!(
        commits, 1,
        "publication CAS must choose exactly one commit winner: {outcomes:?}"
    );
    assert_eq!(db.list_projections(), vec!["contested"]);
}

#[test]
fn projection_commit_before_graph_drop_is_cascaded_by_the_drop_commit() {
    let db = GrafeoDB::new_in_memory();
    db.create_graph("source").unwrap();

    let creator = db.session();
    creator.execute("USE GRAPH source").unwrap();
    creator.execute("START TRANSACTION").unwrap();
    creator
        .execute("CREATE PROJECTION source_people LABELS (Person)")
        .unwrap();

    let dropper = db.session();
    dropper.execute("START TRANSACTION").unwrap();
    dropper.execute("DROP GRAPH source").unwrap();

    creator.execute("COMMIT").unwrap();
    assert!(
        db.projection("source_people").is_some(),
        "the earlier projection commit must publish before the drop serializes"
    );

    dropper.execute("COMMIT").unwrap();
    assert!(
        db.projection("source_people").is_none(),
        "the later graph-drop publication must cascade the exact source projection"
    );
    assert!(!db.list_graphs().iter().any(|name| name == "source"));
}

#[test]
fn graph_drop_commit_invalidates_projection_pinned_to_old_incarnation() {
    let db = GrafeoDB::new_in_memory();
    db.create_graph("source").unwrap();

    let creator = db.session();
    creator.execute("USE GRAPH source").unwrap();
    creator.execute("START TRANSACTION").unwrap();
    creator
        .execute("CREATE PROJECTION source_people LABELS (Person)")
        .unwrap();

    let dropper = db.session();
    dropper.execute("START TRANSACTION").unwrap();
    dropper.execute("DROP GRAPH source").unwrap();
    dropper.execute("COMMIT").unwrap();

    // Reusing the key must not make a projection pinned to the removed Arc
    // valid again: graph identity is the exact incarnation, not just its name.
    db.create_graph("source").unwrap();
    let replacement = db.session();
    replacement.execute("USE GRAPH source").unwrap();
    replacement.execute("INSERT (:Replacement)").unwrap();

    creator
        .execute("COMMIT")
        .expect_err("the old-incarnation projection must lose the lifecycle race");
    assert!(
        db.projection("source_people").is_none(),
        "a failed old-incarnation commit must not attach to the replacement graph"
    );
}

#[test]
fn default_graph_projection_keeps_owning_schema_nonempty() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA analytics").unwrap();
    session.execute("SESSION SET SCHEMA analytics").unwrap();
    session
        .execute("CREATE PROJECTION analytics_people LABELS (Person)")
        .unwrap();
    session.execute("SESSION RESET SCHEMA").unwrap();

    let error = session
        .execute("DROP SCHEMA analytics")
        .expect_err("a projection owned by the schema default graph must enforce RESTRICT");
    assert!(
        error.to_string().contains("not empty"),
        "DROP SCHEMA should report its nonempty-schema contract: {error}"
    );
    assert!(
        db.projection("analytics_people").is_some(),
        "RESTRICT must preserve, not silently cascade, the projection"
    );

    session.execute("DROP PROJECTION analytics_people").unwrap();
    session.execute("DROP SCHEMA analytics").unwrap();
}

#[test]
fn named_graph_projection_survives_rejected_schema_drop() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA analytics").unwrap();
    session.execute("SESSION SET SCHEMA analytics").unwrap();
    session.execute("CREATE GRAPH source").unwrap();
    session.execute("USE GRAPH source").unwrap();
    session
        .execute("CREATE PROJECTION named_people LABELS (Person)")
        .unwrap();
    session.execute("SESSION RESET ALL").unwrap();

    session
        .execute("DROP SCHEMA analytics")
        .expect_err("the schema is nonempty and DROP SCHEMA is RESTRICT");
    assert!(
        db.projection("named_people").is_some(),
        "a rejected schema drop must not cascade a named-graph projection"
    );

    session.execute("DROP PROJECTION named_people").unwrap();
    session.execute("SESSION SET SCHEMA analytics").unwrap();
    session.execute("DROP GRAPH source").unwrap();
    session.execute("SESSION RESET SCHEMA").unwrap();
    session.execute("DROP SCHEMA analytics").unwrap();
}

#[test]
fn show_projections_reads_the_calling_sessions_staged_post_image_only() {
    let db = GrafeoDB::new_in_memory();
    let seed = db.session();
    seed.execute("CREATE PROJECTION committed LABELS (Person)")
        .unwrap();

    let writer = db.session();
    writer.execute("START TRANSACTION").unwrap();
    writer.execute("DROP PROJECTION committed").unwrap();
    writer
        .execute("CREATE PROJECTION staged LABELS (City)")
        .unwrap();

    assert_eq!(
        shown_projection_names(&writer),
        vec!["staged"],
        "SHOW must apply this session's staged create/drop post-image"
    );
    assert_eq!(
        shown_projection_names(&db.session()),
        vec!["committed"],
        "another session must still see the committed registry snapshot"
    );

    writer.execute("ROLLBACK").unwrap();
    assert_eq!(shown_projection_names(&writer), vec!["committed"]);
}

#[test]
fn snapshot_transaction_repeats_its_projection_registry_cut() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.create_projection("initial", ProjectionSpec::new()));

    let mut reader = db.session();
    reader
        .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(shown_projection_names(&reader), vec!["initial"]);

    assert!(db.create_projection("later", ProjectionSpec::new()));
    assert_eq!(
        shown_projection_names(&reader),
        vec!["initial"],
        "Snapshot Isolation must not expose projection DDL committed after BEGIN"
    );
    reader.commit().unwrap();

    assert_eq!(
        shown_projection_names(&db.session()),
        vec!["initial", "later"]
    );
}

#[test]
fn read_committed_projection_show_advances_between_statements() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.create_projection("initial", ProjectionSpec::new()));

    let mut reader = db.session();
    reader
        .begin_transaction_with_isolation(IsolationLevel::ReadCommitted)
        .unwrap();
    assert_eq!(shown_projection_names(&reader), vec!["initial"]);
    assert!(db.create_projection("later", ProjectionSpec::new()));
    assert_eq!(
        shown_projection_names(&reader),
        vec!["initial", "later"],
        "Read Committed must resolve each projection-registry statement anew"
    );
    reader.rollback().unwrap();
}

#[test]
fn serializable_projection_predicate_detects_concurrent_registry_change() {
    let db = GrafeoDB::new_in_memory();
    let mut reader = db.session();
    reader
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    assert!(shown_projection_names(&reader).is_empty());
    reader.execute("INSERT (:ObservedEmptyRegistry)").unwrap();

    assert!(db.create_projection("phantom", ProjectionSpec::new()));
    let error = reader
        .commit()
        .expect_err("Serializable whole-registry predicate must reject a projection phantom");
    assert_eq!(
        error.error_code(),
        ErrorCode::TransactionSerialization,
        "projection phantoms are Serializable predicate conflicts, not write-write conflicts"
    );
    assert!(
        matches!(
            &error,
            Error::Transaction(TransactionError::SerializationFailure(message))
                if message.contains("projection registry changed")
        ),
        "unexpected Serializable failure: {error}"
    );
    assert_eq!(
        db.node_count(),
        0,
        "the conflicting transaction's unrelated LPG write must roll back atomically"
    );
}

#[test]
fn direct_projection_api_remains_usable_with_an_external_read_store() {
    let store = Arc::new(LpgStore::new().unwrap());
    store.create_node(&["Person"]);
    let db = GrafeoDB::with_read_store(
        Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
        Config::in_memory(),
    )
    .unwrap();
    db.set_current_graph(Some("opaque/external"))
        .expect("external stores retain opaque graph selectors");

    assert!(db.create_projection(
        "external_people",
        ProjectionSpec::new().with_node_labels(["Person"]),
    ));
    assert_eq!(
        db.projection("external_people")
            .expect("external projection")
            .node_count(),
        1
    );
    assert!(db.drop_projection("external_people"));
}

#[cfg(feature = "triple-store")]
#[test]
fn lpg_graph_drop_never_cascades_the_durable_rdf_projection_registry() {
    const SOURCE: &str = "source";
    const PERSON: &str = "http://example.com/Person";

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
    db.execute_sparql(&format!(
        "INSERT DATA {{ GRAPH <{SOURCE}> {{ <http://example.com/alix> a <{PERSON}> . }} }}"
    ))
    .expect("insert a row into the same-named RDF source graph");
    let durable_id = db
        .declare_named_rdf_lpg_projection(SOURCE, PERSON, "Person")
        .expect("declare durable named-graph RDF→LPG mapping");
    assert_eq!(
        db.rebuild_rdf_lpg_projection(durable_id)
            .expect("publish a durable RDF→LPG generation"),
        1
    );
    let durable_before = db
        .rdf_lpg_projection(durable_id)
        .expect("durable mapping must be registered");
    assert_eq!(durable_before.source_graph(), Some(SOURCE));
    assert_eq!(durable_before.generation(), 1);
    assert_eq!(durable_before.row_count(), 1);
    let receipt_before = durable_before
        .receipt()
        .expect("a successful rebuild must publish a receipt")
        .clone();
    assert_eq!(receipt_before.source_graph().name(), Some(SOURCE));
    assert_eq!(receipt_before.generation(), 1);
    assert_eq!(receipt_before.row_count(), 1);

    let session = db.session();
    session.execute(&format!("CREATE GRAPH {SOURCE}")).unwrap();
    session.execute(&format!("USE GRAPH {SOURCE}")).unwrap();
    session
        .execute("CREATE PROJECTION runtime_people LABELS (Person)")
        .unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();
    session.execute(&format!("DROP GRAPH {SOURCE}")).unwrap();

    assert!(db.projection("runtime_people").is_none());
    let durable_after = db
        .rdf_lpg_projection(durable_id)
        .expect("LPG graph cascade must not touch durable RDF→LPG metadata");
    assert_eq!(
        durable_after.mapping_digest(),
        durable_before.mapping_digest(),
        "the durable mapping definition must remain byte-identical"
    );
    assert_eq!(durable_after.generation(), durable_before.generation());
    assert_eq!(durable_after.row_count(), durable_before.row_count());
    assert_eq!(
        durable_after.receipt(),
        Some(&receipt_before),
        "the exact published receipt must survive a same-named LPG graph drop"
    );
}
