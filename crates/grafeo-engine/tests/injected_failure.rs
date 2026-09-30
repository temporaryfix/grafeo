//! Rollback atomicity tests using the `testing-statement-injection` hook.
//!
//! These tests exercise the rollback contract independently of any specific
//! engine failure path (constraint violations, parse errors, etc.). The
//! injection hook forces `Session::execute` or `Session::commit` to return
//! `Err` at a chosen call count, letting us assert:
//!
//!   * a mid-tx statement failure leaves prior writes undone after ROLLBACK,
//!   * a commit-time failure leaves all writes undone,
//!   * both properties hold across schema boundaries.
//!
//! ```bash
//! cargo test -p grafeo-engine --features lpg,gql,wal,cypher,testing-statement-injection \
//!     --test injected_failure
//! ```

#![cfg(feature = "testing-statement-injection")]

use grafeo_common::testing::statement_failure::{
    with_commit_failure, with_statement_failure_after,
};
use grafeo_engine::GrafeoDB;

#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]
use grafeo_engine::{Config, DurabilityMode, GraphModel};

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

#[test]
fn injected_statement_failure_rolls_back_prior_writes_single_schema() {
    let db = db();
    let session = db.session();

    // The injection counter ticks for every `Session::execute` call, including
    // DDL and session commands, so arm it to fail *after* the three writes.
    // Counter layout: START TRANSACTION (1), INSERT (2), INSERT (3),
    // INSERT (4) <- should fail.
    with_statement_failure_after(4, || {
        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Item {v: 1})").unwrap();
        session.execute("INSERT (:Item {v: 2})").unwrap();
        let failing = session.execute("INSERT (:Item {v: 3})");
        assert!(
            failing.is_err(),
            "4th execute should be injected as failure"
        );
    });

    // Transaction is still active after the injected failure; rollback explicitly.
    session.execute("ROLLBACK").unwrap();

    let remaining = session.execute("MATCH (n:Item) RETURN n").unwrap();
    assert_eq!(
        remaining.row_count(),
        0,
        "all in-transaction writes must be undone on ROLLBACK"
    );
}

#[test]
fn injected_statement_failure_rolls_back_cross_schema_writes() {
    let db = db();
    let session = db.session();

    session.execute("CREATE SCHEMA alpha").unwrap();
    session.execute("CREATE SCHEMA beta").unwrap();

    // Counter: START TX (1), SET SCHEMA alpha (2), INSERT alpha (3),
    // SET SCHEMA beta (4), INSERT beta (5), SET SCHEMA (6) -> fail.
    with_statement_failure_after(6, || {
        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET SCHEMA alpha").unwrap();
        session.execute("INSERT (:Row {owner: 'alpha'})").unwrap();
        session.execute("SESSION SET SCHEMA beta").unwrap();
        session.execute("INSERT (:Row {owner: 'beta'})").unwrap();
        let failing = session.execute("SESSION RESET SCHEMA");
        assert!(
            failing.is_err(),
            "6th execute should be injected as failure"
        );
    });

    session.execute("ROLLBACK").unwrap();

    session.execute("SESSION SET SCHEMA alpha").unwrap();
    let alpha = session.execute("MATCH (n:Row) RETURN n").unwrap();
    assert_eq!(
        alpha.row_count(),
        0,
        "alpha writes must be undone by cross-schema rollback after injected failure"
    );

    session.execute("SESSION SET SCHEMA beta").unwrap();
    let beta = session.execute("MATCH (n:Row) RETURN n").unwrap();
    assert_eq!(
        beta.row_count(),
        0,
        "beta writes must be undone by cross-schema rollback after injected failure"
    );
}

#[test]
fn injected_commit_failure_rolls_back_prior_writes() {
    let db = db();
    let session = db.session();

    with_commit_failure(|| {
        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Item {v: 1})").unwrap();
        session.execute("INSERT (:Item {v: 2})").unwrap();
        let commit_result = session.execute("COMMIT");
        assert!(commit_result.is_err(), "injected commit failure expected");
    });

    // After the injected commit failure, the transaction's writes must not be
    // visible. A follow-up query should see zero `Item` nodes.
    let remaining = session.execute("MATCH (n:Item) RETURN n").unwrap();
    assert_eq!(
        remaining.row_count(),
        0,
        "injected commit failure must leave no writes visible"
    );
}

#[test]
fn nested_injected_commit_failure_aborts_every_frame_and_resets_depth() {
    let db = db();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    assert!(session.create_node(&["AbortedOuter"]).is_valid());
    session.begin_transaction().unwrap();
    assert!(session.create_node(&["AbortedNestedOne"]).is_valid());
    session.begin_transaction().unwrap();
    assert!(session.create_node(&["AbortedNestedTwo"]).is_valid());

    let error =
        with_commit_failure(|| session.commit()).expect_err("the injected nested commit must fail");
    assert!(
        error.to_string().contains("injected commit failure"),
        "unexpected commit error: {error}"
    );
    assert!(
        !session.in_transaction(),
        "commit failure must abort the outer transaction, not only the innermost savepoint"
    );
    assert_eq!(
        db.node_count(),
        0,
        "writes from every nesting depth must be discarded"
    );
    assert!(
        !db.is_durability_poisoned(),
        "a successful pre-prepare rollback must leave durability healthy"
    );

    // A single commit must finish this new outer transaction. If the failed
    // transaction left stale nesting depth, this would only release a phantom
    // nested frame (or fail while looking for its internal savepoint).
    session.begin_transaction().unwrap();
    assert!(session.create_node(&["Survivor"]).is_valid());
    session.commit().expect("a later transaction must commit");
    assert!(
        !session.in_transaction(),
        "the later transaction must not inherit stale nesting depth"
    );
    assert_eq!(db.node_count(), 1);
}

#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]
fn sidecar_wal_dir(path: &std::path::Path) -> std::path::PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    std::path::PathBuf::from(sidecar)
}

#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]
fn copy_tree(source: &std::path::Path, destination: &std::path::Path) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let destination_entry = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &destination_entry);
        } else {
            std::fs::copy(entry.path(), destination_entry).unwrap();
        }
    }
}

#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]
fn copy_live_database(source: &std::path::Path, destination: &std::path::Path) {
    std::fs::copy(source, destination).unwrap();
    let source_wal = sidecar_wal_dir(source);
    if source_wal.exists() {
        copy_tree(&source_wal, &sidecar_wal_dir(destination));
    }
}

#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]
fn persistent_both_sync(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent dual-model database")
}

#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]
#[test]
fn nested_injected_commit_failure_aborts_both_models_and_cannot_recover() {
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("nested-injected-failure.grafeo");
    let crash_copy = directory
        .path()
        .join("nested-injected-failure-crash-copy.grafeo");
    let db = persistent_both_sync(&path);
    let mut session = db.session();

    session.begin_transaction().unwrap();
    assert!(session.create_node(&["AbortedOuter"]).is_valid());
    session
        .execute_sparql(
            r#"INSERT DATA {
                <http://example.com/aborted-outer> <http://example.com/p> "outer" .
            }"#,
        )
        .unwrap();

    session.begin_transaction().unwrap();
    assert!(session.create_node(&["AbortedNested"]).is_valid());
    session
        .execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://example.com/aborted-graph> {
                    <http://example.com/aborted-nested> <http://example.com/p> "nested" .
                }
            }"#,
        )
        .unwrap();

    with_commit_failure(|| session.commit())
        .expect_err("the injected nested commit must abort the mixed transaction");
    assert!(!session.in_transaction());
    assert_eq!(db.node_count(), 0, "all pending LPG nodes must be gone");
    assert_eq!(
        session
            .execute_sparql("SELECT ?s WHERE { GRAPH ?g { ?s ?p ?o } }")
            .unwrap()
            .row_count(),
        0,
        "all pending RDF quads must be gone"
    );
    assert!(
        db.rdf_store()
            .graph("http://example.com/aborted-graph")
            .is_none(),
        "the failed transaction must not retain named-graph identity"
    );

    session.begin_transaction().unwrap();
    assert!(session.create_node(&["Survivor"]).is_valid());
    session
        .execute_sparql(
            r#"INSERT DATA {
                <http://example.com/survivor> <http://example.com/p> "live" .
            }"#,
        )
        .unwrap();
    session
        .commit()
        .expect("a later mixed transaction must commit");
    assert!(!session.in_transaction());
    assert_eq!(db.node_count(), 1);
    assert_eq!(
        session
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap()
            .row_count(),
        1
    );

    db.wal().expect("persistent database WAL").sync().unwrap();
    copy_live_database(&path, &crash_copy);
    std::mem::forget(session);
    std::mem::forget(db);

    let recovered = persistent_both_sync(&crash_copy);
    assert_eq!(
        recovered.node_count(),
        1,
        "WAL recovery must keep only the later committed LPG node"
    );
    assert_eq!(
        recovered
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap()
            .row_count(),
        1,
        "WAL recovery must keep only the later committed RDF triple"
    );
    assert!(
        recovered
            .rdf_store()
            .graph("http://example.com/aborted-graph")
            .is_none(),
        "WAL recovery must not resurrect aborted named-graph state"
    );
}
