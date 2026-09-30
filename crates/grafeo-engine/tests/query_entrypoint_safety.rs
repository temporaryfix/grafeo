//! Regression gates for Session query-boundary classification.

#![cfg(all(feature = "gql", feature = "lpg"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// A keyword substring is not evidence that a logical plan mutates.
///
/// The former text heuristic classified the `InsertLog` label as an INSERT.
/// With auto-commit disabled that silently opened a transaction for this read
/// and left it active after the result was returned.
#[test]
fn parameterized_read_with_mutation_keyword_does_not_open_transaction() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:InsertLog {name: 'read only'})")
        .expect("seed row");

    let mut session = db.session();
    session.set_auto_commit(false);
    let result = session
        .execute_with_params("MATCH (n:InsertLog) RETURN n.name", HashMap::new())
        .expect("parameterized read");
    assert_eq!(result.row_count(), 1);

    // A genuine read must not leave an implicit transaction behind.
    session
        .begin_transaction()
        .expect("read was classified from its plan, not a keyword substring");
    session.rollback().expect("cleanup explicit transaction");
}

/// Parameterized execution must apply the same read-only transaction rule as
/// the non-parameterized query path.
#[test]
fn parameterized_mutation_is_rejected_in_read_only_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session
        .execute("START TRANSACTION READ ONLY")
        .expect("start read-only transaction");

    let error = session
        .execute_with_params(
            "INSERT (:Person {name: $name})",
            HashMap::from([("name".to_string(), "Alix".into())]),
        )
        .expect_err("read-only transaction must reject a parameterized write");
    assert!(
        error.to_string().to_ascii_lowercase().contains("read-only"),
        "unexpected error: {error}"
    );

    session.rollback().expect("cleanup read-only transaction");
}

/// Sessions are lightweight views of database state, not independent owners
/// that may keep operating after terminal close.
#[test]
fn existing_session_rejects_queries_and_transactions_after_close() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    db.close().expect("close database");

    let query_error = session
        .execute("MATCH (n) RETURN n")
        .expect_err("query through a pre-existing session must reject closed database");
    assert!(
        query_error
            .to_string()
            .to_ascii_lowercase()
            .contains("closed"),
        "unexpected query error: {query_error}"
    );

    let transaction_error = session
        .begin_transaction()
        .expect_err("transaction begin must reject closed database");
    assert!(
        transaction_error
            .to_string()
            .to_ascii_lowercase()
            .contains("closed"),
        "unexpected transaction error: {transaction_error}"
    );
}

/// Standalone DDL remains atomic; explicit transactions own private metadata
/// and SHOW sees that metadata without publishing it.
#[test]
fn catalog_ddl_is_atomic_standalone_and_savepoint_owned_in_transactions() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.set_auto_commit(false);

    session
        .execute("CREATE NODE TYPE Standalone (name STRING)")
        .expect("standalone catalog DDL");
    session
        .begin_transaction()
        .expect("standalone catalog DDL must not leave an implicit transaction");
    session.savepoint("before_ddl").expect("create savepoint");

    session
        .execute("CREATE NODE TYPE MustNotLeak (name STRING)")
        .expect("catalog DDL must be private to the user transaction");
    assert_eq!(
        db.session()
            .execute("SHOW NODE TYPES")
            .expect("committed catalog")
            .row_count(),
        1
    );

    // SHOW remains a read-only operation and is legal at the savepoint cut.
    let types = session
        .execute("SHOW NODE TYPES")
        .expect("SHOW must remain read-only inside a transaction");
    assert_eq!(types.row_count(), 2);

    session
        .rollback_to_savepoint("before_ddl")
        .expect("rollback to savepoint");
    session.rollback().expect("cleanup transaction");

    let types = session
        .execute("SHOW NODE TYPES")
        .expect("show final types");
    assert_eq!(
        types.row_count(),
        1,
        "savepoint-discarded DDL must not reach the catalog"
    );
    assert_eq!(types.rows()[0][0], Value::from("Standalone"));
}
