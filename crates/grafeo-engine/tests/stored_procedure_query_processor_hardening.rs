//! Public `QueryProcessor` stored-procedure authority boundary.
//!
//! Supplying MVCC coordinates is not proof that a caller owns Session
//! authorization, transaction publication, rollback, or durability framing.

#![cfg(all(feature = "lpg", feature = "gql"))]
#![allow(missing_docs)]

use std::sync::Arc;

use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_engine::GrafeoDB;
use grafeo_engine::catalog::{Catalog, ProcedureDefinition};
use grafeo_engine::query::{QueryLanguage, QueryProcessor};
use grafeo_engine::transaction::TransactionManager;

#[test]
fn catalog_call_preserves_mutations_through_the_session_boundary() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE plant(name STRING) RETURNS (name STRING) AS { \
             INSERT (n:Framed {name: $name}) RETURN n.name AS body_local }",
        )
        .unwrap();
    let initial_epoch = db.current_epoch();
    let result = session
        .execute_with_params(
            "CALL plant($value) YIELD name",
            [("value".to_string(), Value::String("durable".into()))].into(),
        )
        .unwrap();

    // RETURNS supplies public names positionally, not by matching body aliases.
    assert_eq!(result.columns, ["name"]);
    assert_eq!(result.rows(), &[vec![Value::String("durable".into())]]);
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.current_epoch(), initial_epoch.next());
    assert!(!session.in_transaction());
}

fn rejected_catalog_yield_leaves_no_write(query: &str) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE plant() RETURNS (name STRING) AS { \
             INSERT (n:MustRollback {name: 'unpublished'}) RETURN n.name AS actual }",
        )
        .unwrap();
    let initial_epoch = db.current_epoch();
    let error = session
        .execute_with_params(query, Default::default())
        .unwrap_err();
    assert!(
        matches!(&error, Error::Query(query)
            if query.kind == grafeo_common::utils::error::QueryErrorKind::Semantic
                && query.message.contains("no declared output 'missing'")),
        "{error:?}"
    );
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.current_epoch(), initial_epoch);
    assert!(!session.in_transaction());
}

#[test]
fn catalog_call_rejects_missing_yield_before_commit() {
    rejected_catalog_yield_leaves_no_write("CALL plant() YIELD missing");
}

#[test]
fn catalog_call_rejects_partial_yield_before_commit() {
    rejected_catalog_yield_leaves_no_write("CALL plant() YIELD name, missing");
}

#[test]
fn detached_processor_builtin_call_uses_the_wrapped_store_cut() {
    let store = Arc::new(LpgStore::new().expect("create advanced LPG store"));
    store.set_epoch(EpochId::new(9));
    let node = store.create_node(&["AdvancedBeforeProcessor"]);
    assert!(node.is_valid());

    let processor = QueryProcessor::for_lpg(store);
    let result = processor
        .process("CALL grafeo.labels()", QueryLanguage::Gql, None)
        .expect("detached processor must read the wrapped store's committed cut");

    assert!(
        result
            .rows()
            .iter()
            .any(|row| row.first().and_then(Value::as_str) == Some("AdvancedBeforeProcessor"))
    );
}

#[test]
fn caller_supplied_transaction_context_does_not_authorize_catalog_writes() {
    let store = Arc::new(LpgStore::new().expect("create LPG store"));
    let transaction_manager = Arc::new(TransactionManager::new());
    let catalog = Arc::new(Catalog::new());
    catalog
        .register_procedure(ProcedureDefinition {
            name: "unguarded_plant".to_string(),
            params: Vec::new(),
            returns: vec![("name".to_string(), "STRING".to_string())],
            body: "INSERT (n:Escaped {name: 'unframed'}) RETURN n.name AS name".to_string(),
        })
        .expect("register a write-capable catalog procedure");

    // Both values come from public QueryProcessor/TransactionManager APIs, but
    // they carry no Session authorization or commit/rollback ownership.
    let transaction_id = transaction_manager.begin();
    let viewing_epoch = transaction_manager.current_epoch();
    let processor = QueryProcessor::for_lpg_with_transaction(
        Arc::clone(&store),
        Arc::clone(&transaction_manager),
    )
    .with_catalog(catalog)
    .with_transaction_context(viewing_epoch, transaction_id);

    let error = processor
        .process("CALL unguarded_plant()", QueryLanguage::Gql, None)
        .expect_err("caller-supplied MVCC context must not authorize a catalog write");

    assert!(
        matches!(
            &error,
            Error::Transaction(TransactionError::InvalidState(message))
                if message.contains("Session-owned transaction")
        ),
        "expected a structured Session-ownership rejection, got: {error:?}"
    );
    assert!(
        store.all_node_ids().is_empty(),
        "the procedure body must be rejected before even a pending node is allocated"
    );
    assert_eq!(transaction_manager.current_epoch(), viewing_epoch);

    transaction_manager
        .abort(transaction_id)
        .expect("release the caller-owned transaction fixture");
}
