//! Time-travel query tests (0.5.14).
//!
//! Validates epoch-based time-travel: querying historical state,
//! session viewing_epoch override, and version history APIs.
//!
//! Note: epochs only advance when explicit transactions are committed.
//! The MVCC assigns `created_epoch = tx.start_epoch`, so to query "before"
//! a creation we need the creation to happen at epoch > 0 by committing
//! a preliminary transaction first.

#![cfg(feature = "lpg")]

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;

#[cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]
use grafeo_engine::{Config, DurabilityMode, GraphModel};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn setup_db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

/// Advances the epoch by committing an empty transaction.
fn bump_epoch(session: &mut grafeo_engine::Session) {
    session.begin_transaction().unwrap();
    session.commit().unwrap();
}

// ---------------------------------------------------------------------------
// execute_at_epoch
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "gql")]
fn test_execute_at_epoch_sees_old_state() {
    let db = setup_db();
    let mut session = db.session();

    // Create a person inside a transaction
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session.commit().unwrap();
    let epoch_after_insert = db.current_epoch();

    // Update the person in a new transaction
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    session.commit().unwrap();

    // Current state should be 31
    let result = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(31));

    // Time-travel to the epoch after insert should see the node.
    // Note: properties are not versioned per-epoch in the current architecture,
    // so time-travel applies to node/edge existence, not property values.
    let result = session
        .execute_at_epoch(
            "MATCH (p:Person {name: 'Alix'}) RETURN p.age",
            epoch_after_insert,
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
#[cfg(feature = "gql")]
fn test_execute_at_epoch_before_creation_returns_empty() {
    let db = setup_db();
    let mut session = db.session();

    // Capture epoch before any data exists
    let epoch_before = db.current_epoch();

    // Advance epoch so the INSERT starts at a higher epoch
    bump_epoch(&mut session);

    // Insert inside a transaction (node created at epoch 1)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.commit().unwrap();

    // Verify node exists at current epoch
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // At epoch before creation, the node should not exist
    let result = session
        .execute_at_epoch("MATCH (p:Person) RETURN p.name", epoch_before)
        .unwrap();
    assert!(
        result.rows().is_empty(),
        "Node should not be visible before its creation epoch"
    );
}

#[test]
#[cfg(feature = "gql")]
fn execute_at_epoch_rejects_mutation_without_residue() {
    let db = setup_db();
    let session = db.session();
    let epoch = db.current_epoch();

    let error = session
        .execute_at_epoch("INSERT (:Person {name: 'Escape'})", epoch)
        .expect_err("a historical view must be read-only");

    assert!(
        error.to_string().contains("historical view is read-only"),
        "unexpected error: {error}"
    );
    assert_eq!(db.node_count(), 0, "the rejected write must leave no node");
    assert!(
        !session.in_transaction(),
        "the rejected write must not open an implicit transaction"
    );
    assert_eq!(
        db.current_epoch(),
        epoch,
        "the rejected write must not publish an epoch"
    );
}

#[test]
#[cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]
fn rejected_historical_mutation_is_absent_after_sync_wal_reopen() {
    let directory = tempfile::TempDir::new().expect("create temporary database directory");
    let path = directory.path().join("historical-rejection.grafeo");
    let committed_epoch;

    {
        let db = GrafeoDB::with_config(
            Config::persistent(&path)
                .with_graph_model(GraphModel::Lpg)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .expect("open persistent database");
        let session = db.session();
        committed_epoch = db.current_epoch();
        let wal_records_before = db.wal().expect("sync WAL must be present").record_count();

        session
            .execute_at_epoch("INSERT (:Escape)", committed_epoch)
            .expect_err("historical mutation must be rejected before WAL framing");

        assert_eq!(
            db.wal()
                .expect("sync WAL must remain present")
                .record_count(),
            wal_records_before,
            "rejection must happen before BEGIN or ABORT reaches the WAL"
        );
        assert_eq!(db.current_epoch(), committed_epoch);
        assert_eq!(db.node_count(), 0);
        assert!(!session.in_transaction());
        db.close().expect("close persistent database");
    }

    let reopened = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("reopen persistent database");
    assert_eq!(reopened.current_epoch(), committed_epoch);
    assert_eq!(reopened.node_count(), 0);
    assert_eq!(
        reopened
            .execute("MATCH (n:Escape) RETURN count(n)")
            .expect("query reopened database")
            .rows()[0][0]
            .as_int64(),
        Some(0),
        "recovery must not materialize the rejected mutation"
    );
}

#[test]
#[cfg(feature = "gql")]
fn execute_at_epoch_with_params_uses_the_requested_cut() {
    let db = setup_db();
    let mut session = db.session();
    let before_creation = db.current_epoch();
    bump_epoch(&mut session);
    session
        .execute("INSERT (:Person {name: 'Future'})")
        .unwrap();

    let params =
        std::collections::HashMap::from([("name".to_string(), Value::String("Future".into()))]);
    let result = session
        .execute_at_epoch_with_params(
            "MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            before_creation,
            Some(params),
        )
        .expect("parameterized historical query");

    assert!(
        result.rows().is_empty(),
        "the parameterized path must not silently fall back to the current epoch"
    );
}

#[test]
#[cfg(feature = "gql")]
fn execute_at_epoch_rejects_an_active_transaction_without_disturbing_it() {
    let db = setup_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Pending'})")
        .unwrap();

    let error = session
        .execute_at_epoch(
            "MATCH (p:Person {name: 'Pending'}) RETURN p.name",
            db.current_epoch(),
        )
        .expect_err("an explicit historical cut cannot replace a transaction snapshot");
    assert!(
        error.to_string().contains("active transaction"),
        "unexpected error: {error}"
    );
    assert!(
        session.in_transaction(),
        "the transaction must remain active"
    );

    let result = session
        .execute("MATCH (p:Person {name: 'Pending'}) RETURN p.name")
        .expect("the original transaction context must remain usable");
    assert_eq!(result.rows().len(), 1);
    session.rollback().unwrap();
    assert_eq!(db.node_count(), 0);
}

#[test]
#[cfg(feature = "gql")]
fn execute_at_epoch_rejects_graph_lifecycle_commands() {
    let db = setup_db();
    let session = db.session();

    let error = session
        .execute_at_epoch("CREATE GRAPH escaped", db.current_epoch())
        .expect_err("historical execution must accept read queries only");
    assert!(
        error
            .to_string()
            .contains("historical execution accepts read queries only"),
        "unexpected error: {error}"
    );
    assert!(
        !db.list_graphs().iter().any(|name| name == "escaped"),
        "the rejected lifecycle command must leave no graph"
    );
}

#[test]
#[cfg(feature = "gql")]
fn historical_execution_rejects_unclassified_procedure_effects() {
    let db = setup_db();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE plant() RETURNS (n NODE) \
             AS { INSERT (n:Secret) RETURN n }",
        )
        .expect("create mutating procedure fixture");
    session
        .execute("CALL plant()")
        .expect("warm a current procedure call");

    let error = session
        .execute_at_epoch("CALL plant()", db.current_epoch())
        .expect_err("procedure effects are not yet safe in a historical view");
    assert!(
        error
            .to_string()
            .contains("procedure calls are not qualified for historical execution"),
        "unexpected error: {error}"
    );
    let result = session
        .execute("MATCH (n:Secret) RETURN n")
        .expect("check for escaped mutation");
    assert_eq!(
        result.row_count(),
        1,
        "historical CALL must not repeat the warmed mutation"
    );
}

#[test]
#[cfg(feature = "gql")]
fn pending_epoch_means_current_committed_cut_not_uncommitted_state() {
    let db = setup_db();
    let mut writer = db.session();
    let reader = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Person {name: 'Uncommitted'})")
        .unwrap();

    let result = reader
        .execute_at_epoch(
            "MATCH (p:Person {name: 'Uncommitted'}) RETURN p.name",
            EpochId::PENDING,
        )
        .expect("PENDING must normalize to the committed publication cut");
    assert!(
        result.rows().is_empty(),
        "an internal PENDING sentinel must not expose another transaction"
    );

    writer.rollback().unwrap();
}

#[test]
#[cfg(all(feature = "gql", feature = "cdc"))]
fn rejected_historical_mutation_emits_no_cdc_event() {
    use grafeo_engine::Config;

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).expect("open CDC database");
    let session = db.session();
    let epoch = db.current_epoch();
    let before = db
        .fixture_changes(EpochId::INITIAL..=EpochId::PENDING)
        .expect("read initial CDC log");
    session.set_viewing_epoch(epoch);

    session
        .execute("INSERT (:Person {name: 'Escape'})")
        .expect_err("historical mutation must fail before CDC staging");

    let after = db
        .fixture_changes(EpochId::INITIAL..=EpochId::PENDING)
        .expect("read final CDC log");
    assert_eq!(after.len(), before.len());
    assert_eq!(db.current_epoch(), epoch);
    assert!(!session.in_transaction());
    assert_eq!(db.node_count(), 0);
}

#[test]
fn test_deleted_node_history_preserves_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Insert in a transaction (Vincent created at epoch 0)
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    session.commit().unwrap();

    // Bump epoch to separate insert from delete
    bump_epoch(&mut session);

    // Delete in a separate transaction
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Vincent'}) DELETE p")
        .unwrap();
    session.commit().unwrap();

    // After deletion, node should not be visible at current epoch
    let result = session
        .execute("MATCH (p:Person {name: 'Vincent'}) RETURN p")
        .unwrap();
    assert!(result.rows().is_empty());

    // The version history API still tracks the deletion metadata.
    // Note: MATCH-based time-travel on deleted nodes is not supported
    // because deletion removes labels, properties, and index entries.
    // Use get_node_at_epoch() or get_node_history() for deleted nodes.
}

#[test]
fn test_get_node_at_epoch_deleted_node() {
    let db = setup_db();
    let mut session = db.session();

    // Bump epoch so the node is created at epoch > 0
    bump_epoch(&mut session);

    let id = db.create_node(&["Person"]);
    let epoch_after_insert = db.current_epoch();

    // Bump epoch before delete
    bump_epoch(&mut session);

    db.delete_node(id);

    // Node should be visible at the epoch after insert (before deletion)
    let node = db.get_node_at_epoch(id, epoch_after_insert);
    assert!(node.is_some());

    // Node should not be visible at the current epoch (after deletion)
    let node = db.get_node_at_epoch(id, db.current_epoch());
    assert!(node.is_none());

    // Node should not be visible at epoch 0 (before creation)
    let node = db.get_node_at_epoch(id, EpochId::new(0));
    assert!(node.is_none());
}

// ---------------------------------------------------------------------------
// Session viewing_epoch override
// ---------------------------------------------------------------------------

#[test]
fn test_session_set_viewing_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Insert Eve in a transaction (Eve created at epoch 0)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Eve'})").unwrap();
    session.commit().unwrap();
    let epoch_after_eve = db.current_epoch();

    // Bump epoch to separate Eve from Frank
    bump_epoch(&mut session);

    // Insert Frank in a separate transaction (Frank created at epoch 2)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Frank'})").unwrap();
    session.commit().unwrap();

    // Without override: 2 nodes
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 2);

    // Set viewing epoch to after Eve but before Frank
    session.set_viewing_epoch(epoch_after_eve);
    assert_eq!(session.viewing_epoch(), Some(epoch_after_eve));

    // With override: only Eve should be visible
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Clear override
    session.clear_viewing_epoch();
    assert_eq!(session.viewing_epoch(), None);

    // Back to normal: 2 nodes
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 2);
}

#[test]
#[cfg(feature = "gql")]
fn persistent_viewing_epoch_rejects_mutation_until_cleared() {
    let db = setup_db();
    let session = db.session();
    let epoch = db.current_epoch();
    session.set_viewing_epoch(epoch);

    let error = session
        .execute("INSERT (:Person {name: 'Escape'})")
        .expect_err("a persistent historical view must be read-only");
    assert!(
        error.to_string().contains("historical view is read-only"),
        "unexpected error: {error}"
    );
    assert_eq!(db.node_count(), 0, "the rejected write must leave no node");
    assert!(
        !session.in_transaction(),
        "the rejected write must not open an implicit transaction"
    );
    assert_eq!(
        db.current_epoch(),
        epoch,
        "the rejected write must not publish an epoch"
    );

    session.clear_viewing_epoch();
    session
        .execute("INSERT (:Person {name: 'Allowed'})")
        .expect("clearing the historical view must restore writes");
    assert_eq!(db.node_count(), 1);
}

#[test]
fn persistent_viewing_epoch_rejects_direct_mutation_without_residue() {
    let db = setup_db();
    let session = db.session();
    let epoch = db.current_epoch();
    session.set_viewing_epoch(epoch);

    let error = session
        .create_node_with_props(&["Person"], [("name", Value::String("Escape".into()))])
        .expect_err("direct CRUD must honor the historical read-only fence");
    assert!(
        error.to_string().contains("historical view is read-only"),
        "unexpected error: {error}"
    );
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.current_epoch(), epoch);
    assert!(!session.in_transaction());
}

#[test]
#[cfg(feature = "gql")]
fn persistent_override_cannot_replace_a_transaction_snapshot() {
    let db = setup_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Pending'})")
        .unwrap();
    session.set_viewing_epoch(db.current_epoch());

    let result = session
        .execute("MATCH (p:Person {name: 'Pending'}) RETURN p.name")
        .expect("the transaction must retain its BEGIN context");
    assert_eq!(
        result.rows().len(),
        1,
        "the persistent override must not discard read-your-writes"
    );

    let error = session
        .execute("INSERT (:Person {name: 'Rejected'})")
        .expect_err("historical mode must reject new writes inside a transaction");
    assert!(
        error.to_string().contains("historical view is read-only"),
        "unexpected error: {error}"
    );

    session.clear_viewing_epoch();
    session.rollback().unwrap();
    assert_eq!(db.node_count(), 0, "rollback must discard the pending node");
}

#[test]
#[cfg(feature = "gql")]
fn transaction_begin_cut_temporarily_supersedes_persistent_override() {
    let db = setup_db();
    let mut session = db.session();
    let before_creation = db.current_epoch();
    bump_epoch(&mut session);
    session
        .execute("INSERT (:Person {name: 'Committed'})")
        .unwrap();
    session.set_viewing_epoch(before_creation);

    let historical = session
        .execute("MATCH (p:Person {name: 'Committed'}) RETURN p.name")
        .unwrap();
    assert!(historical.rows().is_empty());

    session.begin_transaction().unwrap();
    let transactional = session
        .execute("MATCH (p:Person {name: 'Committed'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        transactional.rows().len(),
        1,
        "an active transaction must retain its own immutable BEGIN cut"
    );
    session.rollback().unwrap();

    let historical_again = session
        .execute("MATCH (p:Person {name: 'Committed'}) RETURN p.name")
        .unwrap();
    assert!(
        historical_again.rows().is_empty(),
        "the persistent override must resume after transaction closure"
    );
}

#[test]
#[cfg(all(feature = "gql", feature = "triple-store"))]
fn lpg_viewing_epoch_does_not_disable_rdf_mutation_in_both_mode() {
    use grafeo_core::graph::rdf::{Quad, Term, Triple};
    use grafeo_engine::{Config, GraphModel};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("open dual-model database");
    let session = db.session();
    session.set_viewing_epoch(db.current_epoch());
    let quad = Quad::new(Triple::new(
        Term::iri("urn:subject"),
        Term::iri("urn:predicate"),
        Term::literal("object"),
    ));

    assert_eq!(
        session
            .insert_rdf_quads([quad.clone()])
            .expect("LPG viewing_epoch must not redefine RDF temporal semantics"),
        1
    );
    assert!(
        session
            .try_contains_rdf_quad(&quad)
            .expect("read inserted RDF quad")
    );
}

#[test]
fn test_session_reset_clears_viewing_epoch() {
    let db = setup_db();
    let session = db.session();

    session.set_viewing_epoch(EpochId::new(1));
    assert!(session.viewing_epoch().is_some());

    session.reset_session().unwrap();
    assert!(session.viewing_epoch().is_none());
}

// ---------------------------------------------------------------------------
// GQL session parameter
// ---------------------------------------------------------------------------

#[test]
fn test_gql_session_set_viewing_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Insert Grace in a transaction (Grace created at epoch 0)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Grace'})").unwrap();
    session.commit().unwrap();
    let epoch = db.current_epoch();

    // Bump epoch to separate Grace from Hank
    bump_epoch(&mut session);

    // Insert Hank in a separate transaction (Hank created at epoch 2)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Hank'})").unwrap();
    session.commit().unwrap();

    // Set viewing_epoch via GQL session parameter
    let set_result = session
        .execute(&format!(
            "SESSION SET PARAMETER viewing_epoch = {}",
            epoch.as_u64()
        ))
        .unwrap();
    assert!(set_result.status_message.is_some());

    // Only Grace should be visible
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Reset session
    session.execute("SESSION RESET").unwrap();

    // Both should be visible again
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 2);
}

// ---------------------------------------------------------------------------
// Point lookups at epoch
// ---------------------------------------------------------------------------

#[test]
fn test_get_node_at_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Advance epoch first so nodes are created at epoch > 0
    bump_epoch(&mut session);

    let id = db.create_node(&["Person"]);
    db.set_node_property(id, "name", Value::String("Alix".into()))
        .expect("set node property");
    let epoch_after = db.current_epoch();

    // Node should exist at this epoch
    let node = db.get_node_at_epoch(id, epoch_after);
    assert!(node.is_some());

    // Node should not exist at epoch 0 (before it was created)
    let node = db.get_node_at_epoch(id, EpochId::new(0));
    assert!(node.is_none());
}

#[test]
fn test_get_edge_at_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Advance epoch first
    bump_epoch(&mut session);

    let src = db.create_node(&["Person"]);
    let dst = db.create_node(&["Person"]);
    let edge_id = db.create_edge(src, dst, "KNOWS");
    let epoch_after = db.current_epoch();

    let edge = db.get_edge_at_epoch(edge_id, epoch_after);
    assert!(edge.is_some());

    let edge = db.get_edge_at_epoch(edge_id, EpochId::new(0));
    assert!(edge.is_none());
}

// ---------------------------------------------------------------------------
// Version history
// ---------------------------------------------------------------------------

#[test]
fn test_node_history_single_version() {
    let db = setup_db();

    let id = db.create_node(&["Person"]);
    db.set_node_property(id, "name", Value::String("Alix".into()))
        .expect("set node property");

    let history = db.get_node_history(id);
    assert_eq!(history.len(), 1);

    let (created_epoch, deleted_epoch, node) = &history[0];
    let _ = created_epoch.as_u64();
    assert!(deleted_epoch.is_none());
    assert!(node.has_label("Person"));
}

#[test]
fn test_node_history_deleted() {
    let db = setup_db();

    let id = db.create_node(&["Person"]);
    db.delete_node(id);

    let history = db.get_node_history(id);
    // Version chain should still exist with a deleted marker
    assert!(!history.is_empty());
    let (_, deleted_epoch, _) = &history[0];
    assert!(deleted_epoch.is_some());
}

#[test]
fn test_node_history_nonexistent() {
    let db = setup_db();
    use grafeo_common::types::NodeId;

    let history = db.get_node_history(NodeId::new(9999));
    assert!(history.is_empty());
}

#[test]
fn test_edge_history_single_version() {
    let db = setup_db();

    let src = db.create_node(&["Person"]);
    let dst = db.create_node(&["Person"]);
    let edge_id = db.create_edge(src, dst, "KNOWS");

    let history = db.get_edge_history(edge_id);
    assert_eq!(history.len(), 1);

    let (created_epoch, deleted_epoch, edge) = &history[0];
    let _ = created_epoch.as_u64();
    assert!(deleted_epoch.is_none());
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
    assert_eq!(edge.src, src);
    assert_eq!(edge.dst, dst);
}

#[test]
fn test_current_epoch_increments() {
    let db = setup_db();
    let mut session = db.session();

    let e1 = db.current_epoch();

    // Epoch only advances with explicit transaction commit
    bump_epoch(&mut session);

    let e2 = db.current_epoch();
    assert!(e2.as_u64() > e1.as_u64());
}

// ---------------------------------------------------------------------------
// ValidityTs (unit tests are in the type itself, but add a smoke test)
// ---------------------------------------------------------------------------

#[test]
fn test_validity_ts_smoke() {
    use grafeo_common::types::ValidityTs;

    let ts1 = ValidityTs::new(100);
    let ts2 = ValidityTs::new(200);

    // Reverse ordering
    assert!(ts2 < ts1);

    // Key roundtrip
    let key = ValidityTs::versioned_key(42, ts1);
    let (id, decoded) = ValidityTs::from_versioned_key(&key);
    assert_eq!(id, 42);
    assert_eq!(decoded, ts1);
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;
