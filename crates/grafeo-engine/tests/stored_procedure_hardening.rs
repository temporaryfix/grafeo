//! Security and execution-boundary contracts for catalog stored procedures.
//!
//! These tests intentionally exercise the public Session API. Catalog procedure
//! bodies must inherit the caller's authorization and transaction boundary; a
//! nested planner must never become a second, less-governed execution path.

#![cfg(all(feature = "lpg", feature = "gql"))]
#![allow(missing_docs)]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind, TransactionError};
use grafeo_engine::auth::Role;
use grafeo_engine::{GrafeoDB, IsolationLevel, Session};

fn call_string_column(session: &Session, query: &str) -> Vec<String> {
    session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query} failed: {error:?}"))
        .rows()
        .iter()
        .map(|row| {
            row.first()
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("{query} returned a non-string row: {row:?}"))
                .to_string()
        })
        .collect()
}

fn semantic_contract_message(error: &Error) -> Option<&str> {
    match error {
        Error::Query(query) if query.kind == QueryErrorKind::Semantic => Some(&query.message),
        Error::Context { source, .. } => semantic_contract_message(source),
        _ => None,
    }
}

fn contains_type_mismatch(error: &Error, expected: &str, found: &str) -> bool {
    match error {
        Error::TypeMismatch {
            expected: actual_expected,
            found: actual_found,
        } => actual_expected == expected && actual_found == found,
        Error::Context { source, .. } => contains_type_mismatch(source, expected, found),
        _ => false,
    }
}

#[test]
fn readonly_identity_can_call_readonly_builtin() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("seed visible label");
    let epoch = db.current_epoch();

    let reader = db.session_with_role(Role::ReadOnly);
    let result = reader
        .execute("CALL grafeo.labels()")
        .expect("read hardening must preserve read-only builtins");

    assert!(
        result
            .rows()
            .iter()
            .any(|row| row.first().and_then(Value::as_str) == Some("Person")),
        "labels() must expose the seeded label to an authorized reader: {:?}",
        result.rows()
    );
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.current_epoch(), epoch);
    assert!(!reader.in_transaction());
}

#[test]
fn readonly_identity_can_call_readonly_catalog_procedure() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("seed procedure input");
    admin
        .execute(
            "CREATE PROCEDURE list_people(name STRING) RETURNS (name STRING) AS { \
             MATCH (p:Person) WHERE p.name = $name RETURN p.name AS name }",
        )
        .expect("create read-only catalog procedure");
    let epoch = db.current_epoch();

    let reader = db.session_with_role(Role::ReadOnly);
    let result = reader
        .execute("CALL list_people('Alix')")
        .expect("an authorized reader may call a read-only catalog procedure");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Alix"));
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.current_epoch(), epoch);
    assert!(!reader.in_transaction());
}

#[test]
fn write_catalog_procedure_requires_write_authority_before_body() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin
        .execute(
            "CREATE PROCEDURE plant() RETURNS (name STRING) AS { \
             INSERT (n:Secret {name: 'escape'}) RETURN n.name AS name }",
        )
        .expect("create mutating catalog procedure");
    let epoch = db.current_epoch();

    let reader = db.session_with_role(Role::ReadOnly);
    let error = reader
        .execute("CALL plant()")
        .expect_err("write effects must be authorized before the procedure body runs");

    assert!(
        matches!(
            &error,
            Error::Query(query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("permission denied")
        ),
        "expected structured permission denial, got: {error:?}"
    );
    assert_eq!(db.node_count(), 0, "the denied body must not create a node");
    assert_eq!(db.current_epoch(), epoch);
    assert!(!reader.in_transaction());
}

#[test]
fn write_catalog_procedure_fails_closed_in_readonly_transaction() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin
        .execute(
            "CREATE PROCEDURE plant() RETURNS (name STRING) AS { \
             INSERT (n:Secret {name: 'escape'}) RETURN n.name AS name }",
        )
        .expect("create mutating catalog procedure");
    let epoch = db.current_epoch();

    let mut session = db.session();
    session
        .execute("START TRANSACTION READ ONLY")
        .expect("start read-only transaction");
    let error = session
        .execute("CALL plant()")
        .expect_err("a write procedure must honor the active read-only transaction");

    assert!(
        matches!(&error, Error::Transaction(TransactionError::ReadOnly)),
        "expected the standard read-only transaction error, got: {error:?}"
    );
    assert_eq!(db.node_count(), 0, "the rejected body must leave no node");
    assert_eq!(db.current_epoch(), epoch);
    assert!(
        session.in_transaction(),
        "a rejected statement must not abort the caller's transaction"
    );
    let count = session
        .execute("MATCH (n:Secret) RETURN count(n)")
        .expect("the read-only transaction must remain usable");
    assert_eq!(count.rows()[0][0].as_int64(), Some(0));
    session.rollback().expect("clean up read-only transaction");
}

#[test]
fn cached_write_catalog_procedure_cannot_cross_session_authority() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin
        .execute(
            "CREATE PROCEDURE cached_plant() RETURNS (name STRING) AS { \
             INSERT (n:Secret {name: 'cached'}) RETURN n.name AS name }",
        )
        .expect("create mutating catalog procedure");

    admin
        .execute("CALL cached_plant()")
        .expect("authorized call primes any reusable plan path");
    assert_eq!(db.node_count(), 1);

    let reader = db.session_with_role(Role::ReadOnly);
    let error = reader
        .execute("CALL cached_plant()")
        .expect_err("a cached admin plan must not retain write authority for a reader");

    assert!(
        matches!(
            &error,
            Error::Query(query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("permission denied")
        ),
        "expected structured permission denial on the cached path, got: {error:?}"
    );
    assert_eq!(
        db.node_count(),
        1,
        "the reader must not execute the writable store captured by another session"
    );
    assert!(!reader.in_transaction());
}

#[test]
fn catalog_procedure_parameters_are_values_not_gql_source_text() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE echo(payload STRING) \
             RETURNS (value STRING, marker STRING) AS { \
             RETURN $payload AS value, '$payload' AS marker }",
        )
        .expect("create typed echo procedure");

    let payload = "x') INSERT (:Injected) //".to_string();
    let result = session
        .execute_with_params(
            "CALL echo($argument)",
            HashMap::from([(
                "argument".to_string(),
                Value::String(payload.clone().into()),
            )]),
        )
        .expect("typed arguments must be bound as values, never interpolated into source text");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some(payload.as_str()));
    assert_eq!(
        result.rows()[0][1].as_str(),
        Some("$payload"),
        "a parameter-looking token inside a string literal is ordinary data"
    );
    assert_eq!(
        db.node_count(),
        0,
        "the parameter payload must never become executable GQL"
    );
    assert!(!session.in_transaction());
}

#[test]
fn parameterized_write_procedure_inherits_query_resources() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE parameterized_plant(name STRING) RETURNS (name STRING) AS { \
             INSERT (n:ParameterizedPlant {name: $name}) RETURN n.name AS name }",
        )
        .expect("create parameterized write procedure");

    let result = session
        .execute_with_params(
            "CALL parameterized_plant($name)",
            HashMap::from([("name".to_string(), Value::String("resource-owned".into()))]),
        )
        .expect("parameterized write procedure must receive the query resource context");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("resource-owned"));
    assert_eq!(db.node_count(), 1);
    assert!(!session.in_transaction());
}

#[test]
fn profiled_write_procedure_inherits_query_resources() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE profiled_plant() RETURNS (name STRING) AS { \
             INSERT (n:ProfiledPlant {name: 'profiled'}) RETURN n.name AS name }",
        )
        .expect("create profiled write procedure");

    let profile = session
        .execute("PROFILE CALL profiled_plant()")
        .expect("PROFILE must forward resources through every profiling wrapper");

    assert_eq!(profile.row_count(), 1, "PROFILE returns its report row");
    assert_eq!(db.node_count(), 1);
    assert!(!session.in_transaction());
}

#[test]
fn nested_write_catalog_procedure_requires_transitive_write_authority() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin
        .execute(
            "CREATE PROCEDURE nested_leaf_write() RETURNS (name STRING) AS { \
             INSERT (n:NestedSecret {name: 'denied'}) RETURN n.name AS name }",
        )
        .expect("create nested mutating leaf procedure");
    admin
        .execute(
            "CREATE PROCEDURE nested_write() RETURNS (name STRING) AS { \
             CALL nested_leaf_write() }",
        )
        .expect("create read-looking wrapper around mutating procedure");
    let epoch = db.current_epoch();

    let reader = db.session_with_role(Role::ReadOnly);
    let error = reader
        .execute("CALL nested_write()")
        .expect_err("transitive write effects must require write authority");

    assert!(
        matches!(
            &error,
            Error::Query(query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("permission denied")
        ),
        "expected structured permission denial for nested write, got: {error:?}"
    );
    assert_eq!(db.node_count(), 0, "the nested write must not execute");
    assert_eq!(
        db.current_epoch(),
        epoch,
        "denial must not publish an epoch"
    );
    assert!(!reader.in_transaction());
}

#[test]
fn nested_write_catalog_procedure_auto_commits_one_visible_epoch() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE nested_leaf_write() RETURNS (name STRING) AS { \
             INSERT (n:NestedSecret {name: 'committed'}) RETURN n.name AS name }",
        )
        .expect("create nested mutating leaf procedure");
    session
        .execute(
            "CREATE PROCEDURE nested_write() RETURNS (name STRING) AS { \
             CALL nested_leaf_write() }",
        )
        .expect("create wrapper around mutating procedure");
    let epoch = db.current_epoch();

    let result = session
        .execute("CALL nested_write()")
        .expect("authorized nested write must execute in one auto-commit transaction");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("committed"));
    assert_eq!(
        db.current_epoch(),
        epoch.next(),
        "the outer CALL must publish exactly one commit epoch"
    );
    assert!(!session.in_transaction());

    let observer = db.session();
    let visible = observer
        .execute("MATCH (n:NestedSecret) RETURN n.name AS name")
        .expect("a separate session must observe the committed nested write");
    assert_eq!(visible.row_count(), 1);
    assert_eq!(visible.rows()[0][0].as_str(), Some("committed"));
}

#[test]
fn nested_write_catalog_procedure_uses_explicit_transaction_and_rolls_back() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin
        .execute(
            "CREATE PROCEDURE nested_leaf_write() RETURNS (name STRING) AS { \
             INSERT (n:NestedSecret {name: 'temporary'}) RETURN n.name AS name }",
        )
        .expect("create nested mutating leaf procedure");
    admin
        .execute(
            "CREATE PROCEDURE nested_write() RETURNS (name STRING) AS { \
             CALL nested_leaf_write() }",
        )
        .expect("create wrapper around mutating procedure");
    let epoch = db.current_epoch();

    let mut writer = db.session();
    writer
        .execute("START TRANSACTION")
        .expect("start caller-owned transaction");
    writer
        .execute("CALL nested_write()")
        .expect("nested write must join the caller's transaction");

    let own_view = writer
        .execute("MATCH (n:NestedSecret) RETURN n.name AS name")
        .expect("the caller must read its nested pending write");
    assert_eq!(own_view.row_count(), 1);
    assert_eq!(own_view.rows()[0][0].as_str(), Some("temporary"));
    assert!(writer.in_transaction());
    assert_eq!(
        db.current_epoch(),
        epoch,
        "an explicit transaction must not publish during CALL"
    );

    let observer = db.session();
    let outside_view = observer
        .execute("MATCH (n:NestedSecret) RETURN count(n)")
        .expect("an observer must remain usable while the write is pending");
    assert_eq!(outside_view.rows()[0][0].as_int64(), Some(0));

    writer
        .rollback()
        .expect("roll back caller-owned transaction");

    assert!(!writer.in_transaction());
    assert_eq!(db.current_epoch(), epoch, "rollback must publish no epoch");
    assert_eq!(db.node_count(), 0, "rollback must discard the nested write");
    let after_rollback = db
        .session()
        .execute("MATCH (n:NestedSecret) RETURN count(n)")
        .expect("the rolled-back write must remain invisible");
    assert_eq!(after_rollback.rows()[0][0].as_int64(), Some(0));
}

#[test]
fn direct_catalog_procedure_recursion_is_a_structured_semantic_error() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE direct_cycle() RETURNS (value STRING) AS { \
             CALL direct_cycle() }",
        )
        .expect("create directly recursive procedure fixture");

    let error = session
        .execute("CALL direct_cycle()")
        .expect_err("direct recursion must be rejected before execution");

    assert!(
        matches!(
            &error,
            Error::Query(query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("procedure call cycle detected")
                    && query.message.contains("direct_cycle -> direct_cycle")
        ),
        "expected structured direct-cycle rejection, got: {error:?}"
    );
    assert!(!session.in_transaction());
}

#[test]
fn indirect_catalog_procedure_recursion_is_a_structured_semantic_error() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE cycle_left() RETURNS (value STRING) AS { \
             CALL cycle_right() }",
        )
        .expect("create first half of indirect recursion fixture");
    session
        .execute(
            "CREATE PROCEDURE cycle_right() RETURNS (value STRING) AS { \
             CALL cycle_left() }",
        )
        .expect("create second half of indirect recursion fixture");

    let error = session
        .execute("CALL cycle_left()")
        .expect_err("indirect recursion must be rejected before execution");

    assert!(
        matches!(
            &error,
            Error::Query(query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("procedure call cycle detected")
                    && query
                        .message
                        .contains("cycle_left -> cycle_right -> cycle_left")
        ),
        "expected structured indirect-cycle rejection, got: {error:?}"
    );
    assert!(!session.in_transaction());
}

#[test]
fn unknown_catalog_call_is_a_structured_semantic_error() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let epoch = db.current_epoch();

    let error = session
        .execute("CALL procedure_that_does_not_exist()")
        .expect_err("an unknown CALL must fail during semantic qualification");

    assert!(
        matches!(
            &error,
            Error::Query(query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("unknown procedure")
                    && query.message.contains("procedure_that_does_not_exist")
        ),
        "expected a structured unknown-procedure error, got: {error:?}"
    );
    assert_eq!(db.current_epoch(), epoch);
    assert!(!session.in_transaction());

    let listing = session
        .execute("CALL GRAFEO.procedures()")
        .expect("builtin namespace matching is case-insensitive");
    assert!(listing.row_count() > 0);
}

#[test]
fn trusted_builtin_namespace_cannot_be_shadowed_by_a_catalog_leaf_name() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("seed a label visible to the trusted builtin");
    session
        .execute(
            "CREATE PROCEDURE labels() RETURNS (label STRING) AS { \
             RETURN 'catalog-labels' AS label }",
        )
        .expect("create a catalog procedure whose leaf name matches a builtin");
    let epoch = db.current_epoch();

    let unqualified = session
        .execute("CALL labels()")
        .expect("an unqualified call may intentionally resolve through the catalog");
    assert_eq!(unqualified.rows(), &[vec![Value::from("catalog-labels")]]);

    let qualified = session
        .execute("CALL grafeo.labels()")
        .expect("the trusted namespace must resolve the builtin directly");
    assert!(
        qualified
            .rows()
            .iter()
            .any(|row| row.first().and_then(Value::as_str) == Some("Person")),
        "grafeo.labels() must execute the builtin, got: {:?}",
        qualified.rows()
    );
    assert!(
        qualified
            .rows()
            .iter()
            .all(|row| row.first().and_then(Value::as_str) != Some("catalog-labels")),
        "a catalog leaf name must not shadow the trusted grafeo namespace"
    );
    assert_eq!(db.current_epoch(), epoch);
    assert!(!session.in_transaction());
}

#[test]
fn metadata_builtins_remain_pinned_to_snapshot_isolation_begin_cut() {
    let db = GrafeoDB::new_in_memory();
    let begin_epoch = db.current_epoch();
    let mut snapshot = db.session();
    snapshot
        .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
        .expect("begin explicit Snapshot Isolation transaction");
    assert!(snapshot.in_transaction());

    let publisher = db.session();
    publisher
        .execute(
            "INSERT (:FutureNode {future_node_key: 'node'}) \
             -[:FUTURE_REL {future_edge_key: 'edge'}]->(:FutureNode)",
        )
        .expect("publish graph data carrying new metadata");
    let published_epoch = db.current_epoch();
    assert_eq!(published_epoch, begin_epoch.next());
    assert_eq!(db.node_count(), 2);
    assert_eq!(db.edge_count(), 1);
    assert!(!publisher.in_transaction());

    let published_labels = call_string_column(&publisher, "CALL grafeo.labels()");
    let published_relationships = call_string_column(&publisher, "CALL grafeo.relationshipTypes()");
    let published_properties = call_string_column(&publisher, "CALL grafeo.propertyKeys()");
    assert!(published_labels.iter().any(|value| value == "FutureNode"));
    assert!(
        published_relationships
            .iter()
            .any(|value| value == "FUTURE_REL")
    );
    assert!(
        published_properties
            .iter()
            .any(|value| value == "future_node_key")
    );
    assert!(
        published_properties
            .iter()
            .any(|value| value == "future_edge_key")
    );

    let snapshot_labels = call_string_column(&snapshot, "CALL grafeo.labels()");
    assert!(snapshot.in_transaction());
    let snapshot_relationships = call_string_column(&snapshot, "CALL grafeo.relationshipTypes()");
    assert!(snapshot.in_transaction());
    let snapshot_properties = call_string_column(&snapshot, "CALL grafeo.propertyKeys()");
    assert!(snapshot.in_transaction());

    assert!(
        snapshot_labels.iter().all(|value| value != "FutureNode"),
        "labels() drifted beyond the begin cut: {snapshot_labels:?}"
    );
    assert!(
        snapshot_relationships
            .iter()
            .all(|value| value != "FUTURE_REL"),
        "relationshipTypes() drifted beyond the begin cut: {snapshot_relationships:?}"
    );
    assert!(
        snapshot_properties
            .iter()
            .all(|value| value != "future_node_key" && value != "future_edge_key"),
        "propertyKeys() drifted beyond the begin cut: {snapshot_properties:?}"
    );
    assert_eq!(db.current_epoch(), published_epoch);

    snapshot
        .rollback()
        .expect("end the pinned read transaction");
    assert!(!snapshot.in_transaction());
    assert_eq!(db.current_epoch(), published_epoch);
}

#[test]
fn metadata_builtins_read_pending_transaction_writes_then_rollback_cleanly() {
    let db = GrafeoDB::new_in_memory();
    let epoch = db.current_epoch();
    let mut writer = db.session();
    writer
        .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
        .expect("begin explicit Snapshot Isolation transaction");
    writer
        .execute(
            "INSERT (:PendingNode {pending_node_key: 'node'}) \
             -[:PENDING_REL {pending_edge_key: 'edge'}]->(:PendingNode)",
        )
        .expect("stage node and edge metadata in the transaction");
    assert!(writer.in_transaction());
    assert_eq!(db.current_epoch(), epoch, "pending write must not publish");

    let own_labels = call_string_column(&writer, "CALL grafeo.labels()");
    let own_relationships = call_string_column(&writer, "CALL grafeo.relationshipTypes()");
    let own_properties = call_string_column(&writer, "CALL grafeo.propertyKeys()");
    assert!(own_labels.iter().any(|value| value == "PendingNode"));
    assert!(own_relationships.iter().any(|value| value == "PENDING_REL"));
    assert!(
        own_properties
            .iter()
            .any(|value| value == "pending_node_key")
    );
    assert!(
        own_properties
            .iter()
            .any(|value| value == "pending_edge_key")
    );
    assert!(writer.in_transaction());
    assert_eq!(db.current_epoch(), epoch);

    let observer = db.session();
    assert!(
        call_string_column(&observer, "CALL grafeo.labels()")
            .iter()
            .all(|value| value != "PendingNode")
    );
    assert!(
        call_string_column(&observer, "CALL grafeo.relationshipTypes()")
            .iter()
            .all(|value| value != "PENDING_REL")
    );
    assert!(
        call_string_column(&observer, "CALL grafeo.propertyKeys()")
            .iter()
            .all(|value| value != "pending_node_key" && value != "pending_edge_key")
    );
    assert!(!observer.in_transaction());
    assert!(writer.in_transaction());

    writer.rollback().expect("discard pending metadata fixture");
    assert!(!writer.in_transaction());
    assert_eq!(db.current_epoch(), epoch, "rollback must publish no epoch");
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);

    assert!(
        call_string_column(&observer, "CALL grafeo.labels()")
            .iter()
            .all(|value| value != "PendingNode")
    );
    assert!(
        call_string_column(&observer, "CALL grafeo.relationshipTypes()")
            .iter()
            .all(|value| value != "PENDING_REL")
    );
    assert!(
        call_string_column(&observer, "CALL grafeo.propertyKeys()")
            .iter()
            .all(|value| value != "pending_node_key" && value != "pending_edge_key")
    );
    assert!(!observer.in_transaction());
}

#[test]
fn explain_mutation_in_catalog_body_is_rejected_before_transaction_framing() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE explain_plant() RETURNS (plan STRING) AS { \
             EXPLAIN INSERT (n:ExplainedResidue {marker: 'forbidden'}) RETURN n }",
        )
        .expect("create hostile EXPLAIN mutation fixture");
    let epoch = db.current_epoch();

    let error = session
        .execute("CALL explain_plant()")
        .expect_err("procedure modifiers must be rejected before nested planning");

    assert!(
        matches!(
            &error,
            Error::Query(query)
                if query.kind == QueryErrorKind::Semantic
                    && query.message.contains("EXPLAIN")
        ),
        "expected structured EXPLAIN rejection, got: {error:?}"
    );
    assert_eq!(
        db.node_count(),
        0,
        "the explained INSERT must leave no residue"
    );
    assert_eq!(db.edge_count(), 0);
    assert_eq!(db.current_epoch(), epoch, "rejection must publish no epoch");
    assert!(!session.in_transaction());
    let residue = session
        .execute("MATCH (n:ExplainedResidue) RETURN count(n)")
        .expect("session must remain usable after semantic rejection");
    assert_eq!(residue.rows()[0][0].as_int64(), Some(0));
}

#[test]
fn wrong_catalog_argument_type_is_semantic_before_body_execution() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE typed_plant(quantity INTEGER) RETURNS (name STRING) AS { \
             INSERT (n:WrongArgumentResidue {name: 'must-not-exist'}) \
             RETURN n.name AS name }",
        )
        .expect("create typed mutating procedure fixture");
    let epoch = db.current_epoch();

    let error = session
        .execute("CALL typed_plant('not-an-integer')")
        .expect_err("a wrongly typed argument must fail before the body runs");

    let message = semantic_contract_message(&error)
        .unwrap_or_else(|| panic!("expected structured Semantic error, got: {error:?}"));
    assert!(
        message.contains("procedure 'typed_plant' argument 'quantity' expects INT64, found STRING"),
        "missing precise argument contract message: {message}"
    );
    assert_eq!(db.node_count(), 0, "argument failure must leave no residue");
    assert_eq!(
        db.current_epoch(),
        epoch,
        "argument failure must publish no epoch"
    );
    assert!(!session.in_transaction());
}

#[test]
fn wrong_catalog_argument_count_is_semantic_not_internal() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE counted_echo(value STRING) RETURNS (value STRING) AS { \
             RETURN $value AS value }",
        )
        .expect("create one-argument procedure fixture");
    let epoch = db.current_epoch();

    let error = session
        .execute("CALL counted_echo()")
        .expect_err("wrong argument count must be a user-facing contract error");

    let message = semantic_contract_message(&error)
        .unwrap_or_else(|| panic!("expected structured Semantic error, got: {error:?}"));
    assert!(
        message.contains("procedure 'counted_echo' expects 1 arguments, got 0"),
        "missing precise argument-count contract message: {message}"
    );
    assert_eq!(db.current_epoch(), epoch);
    assert!(!session.in_transaction());
}

#[test]
fn unsupported_catalog_declared_type_is_semantic_before_execution() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE unsupported_contract() RETURNS (value UUID) AS { \
             INSERT (n:UnsupportedTypeResidue) RETURN 'value' AS value }",
        )
        .expect("catalog DDL stores the declaration for call-time qualification");
    let epoch = db.current_epoch();

    let error = session
        .execute("CALL unsupported_contract()")
        .expect_err("unsupported declared types must fail before body execution");

    let message = semantic_contract_message(&error)
        .unwrap_or_else(|| panic!("expected structured Semantic error, got: {error:?}"));
    assert!(
        message.contains(
            "procedure 'unsupported_contract' return 'value' uses unsupported type 'UUID'"
        ),
        "missing precise unsupported-type contract message: {message}"
    );
    assert_eq!(
        db.node_count(),
        0,
        "unsupported signature must be rejected before execution"
    );
    assert_eq!(db.current_epoch(), epoch);
    assert!(!session.in_transaction());
}

#[test]
fn wrong_catalog_return_type_rolls_back_auto_commit_write() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE wrong_return_type() RETURNS (name STRING) AS { \
             INSERT (n:WrongReturnResidue {name: 'temporary'}) RETURN 7 AS name }",
        )
        .expect("create mutating wrong-return-type fixture");
    let epoch = db.current_epoch();

    let error = session
        .execute("CALL wrong_return_type()")
        .expect_err("runtime return contract must reject the body result");

    assert!(
        contains_type_mismatch(&error, "STRING", "INT64"),
        "expected structured STRING/INT64 type mismatch, got: {error:?}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains(
            "procedure 'wrong_return_type' output 'name' violates its declared return type"
        ),
        "missing precise return contract context: {rendered}"
    );
    assert_eq!(
        db.node_count(),
        0,
        "the failed output contract must roll back the body write"
    );
    assert_eq!(
        db.current_epoch(),
        epoch,
        "failed auto-commit must publish no epoch"
    );
    assert!(!session.in_transaction());
    let residue = session
        .execute("MATCH (n:WrongReturnResidue) RETURN count(n)")
        .expect("session must remain usable after output contract rollback");
    assert_eq!(residue.rows()[0][0].as_int64(), Some(0));
}

#[test]
fn wrong_catalog_return_type_rolls_back_only_its_explicit_transaction_statement() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session
        .execute(
            "CREATE PROCEDURE wrong_return_in_tx() RETURNS (name STRING) AS { \
             INSERT (n:WrongReturnResidue {name: 'temporary'}) RETURN 7 AS name }",
        )
        .expect("create explicit-transaction wrong-return fixture");
    let epoch = db.current_epoch();

    session
        .begin_transaction()
        .expect("begin caller-owned transaction");
    session
        .execute("INSERT (:BeforeFailure {name: 'preserved'})")
        .expect("stage valid work before the failing CALL");
    let error = session
        .execute("CALL wrong_return_in_tx()")
        .expect_err("the return contract must fail after staging its body write");

    assert!(contains_type_mismatch(&error, "STRING", "INT64"));
    assert!(session.in_transaction());
    assert_eq!(db.current_epoch(), epoch);
    let own_before = session
        .execute("MATCH (before:BeforeFailure) RETURN count(before)")
        .expect("the caller transaction must remain usable");
    let own_residue = session
        .execute("MATCH (residue:WrongReturnResidue) RETURN count(residue)")
        .expect("the caller transaction must hide rolled-back residue");
    assert_eq!(own_before.rows()[0][0].as_int64(), Some(1));
    assert_eq!(own_residue.rows()[0][0].as_int64(), Some(0));

    let commit_epoch = session
        .commit()
        .expect("earlier valid work remains committable");
    assert_eq!(commit_epoch, epoch.next());
    assert!(!session.in_transaction());
    let observer = db.session();
    let visible_before = observer
        .execute("MATCH (before:BeforeFailure) RETURN count(before)")
        .expect("observe the committed statement boundary");
    let visible_residue = observer
        .execute("MATCH (residue:WrongReturnResidue) RETURN count(residue)")
        .expect("rolled-back statement residue must remain absent");
    assert_eq!(visible_before.rows()[0][0].as_int64(), Some(1));
    assert_eq!(visible_residue.rows()[0][0].as_int64(), Some(0));
}

#[test]
fn mixed_mutation_and_failing_readonly_call_is_one_atomic_statement() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session
        .execute(
            "CREATE PROCEDURE wrong_readonly_return() RETURNS (name STRING) AS { \
             RETURN 7 AS name }",
        )
        .expect("create read-only wrong-return fixture");

    session
        .begin_transaction()
        .expect("begin caller transaction");
    session
        .execute("INSERT (:BeforeMixedFailure)")
        .expect("stage valid work before mixed statement");
    let error = session
        .execute(
            "INSERT (:MixedCallResidue) \
             CALL wrong_readonly_return() YIELD name RETURN name",
        )
        .expect_err("late read-only CALL failure must reject the mixed statement");

    assert!(contains_type_mismatch(&error, "STRING", "INT64"));
    assert!(session.in_transaction());
    let own_before = session
        .execute("MATCH (:BeforeMixedFailure) RETURN count(*)")
        .expect("earlier statement remains visible");
    let own_residue = session
        .execute("MATCH (:MixedCallResidue) RETURN count(*)")
        .expect("failed mixed statement leaves no residue");
    assert_eq!(own_before.rows()[0][0].as_int64(), Some(1));
    assert_eq!(own_residue.rows()[0][0].as_int64(), Some(0));

    session.commit().expect("commit only the earlier statement");
    let observer = db.session();
    let visible_before = observer
        .execute("MATCH (:BeforeMixedFailure) RETURN count(*)")
        .expect("earlier statement committed");
    let visible_residue = observer
        .execute("MATCH (:MixedCallResidue) RETURN count(*)")
        .expect("mixed residue remains absent after commit");
    assert_eq!(visible_before.rows()[0][0].as_int64(), Some(1));
    assert_eq!(visible_residue.rows()[0][0].as_int64(), Some(0));
}

#[test]
fn ordinary_multirow_mutation_error_is_statement_atomic_in_explicit_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session
        .execute("CREATE NODE TYPE AtomicRow (id INTEGER NOT NULL)")
        .expect("create late-validation fixture");

    session
        .begin_transaction()
        .expect("begin caller transaction");
    session
        .execute("INSERT (:BeforeOrdinaryFailure)")
        .expect("stage earlier valid statement");
    session
        .execute(
            "UNWIND [{id: 1}, {}] AS row \
             INSERT (:AtomicRow {id: row.id}) RETURN row.id",
        )
        .expect_err("the second row must fail its NOT NULL constraint");

    assert!(session.in_transaction());
    let own_before = session
        .execute("MATCH (:BeforeOrdinaryFailure) RETURN count(*)")
        .expect("earlier statement remains visible");
    let own_rows = session
        .execute("MATCH (:AtomicRow) RETURN count(*)")
        .expect("partial rows are absent after statement rollback");
    assert_eq!(own_before.rows()[0][0].as_int64(), Some(1));
    assert_eq!(own_rows.rows()[0][0].as_int64(), Some(0));

    session.commit().expect("commit only the earlier statement");
    let observer = db.session();
    let visible_before = observer
        .execute("MATCH (:BeforeOrdinaryFailure) RETURN count(*)")
        .expect("earlier statement committed");
    let visible_rows = observer
        .execute("MATCH (:AtomicRow) RETURN count(*)")
        .expect("failed statement remains absent");
    assert_eq!(visible_before.rows()[0][0].as_int64(), Some(1));
    assert_eq!(visible_rows.rows()[0][0].as_int64(), Some(0));
}

#[test]
fn direct_insert_return_suffix_preserves_exact_string_value() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let epoch = db.current_epoch();

    let result = session
        .execute("INSERT (n:DirectInsertReturn {name: 'Alix'}) RETURN n.name AS name")
        .expect("INSERT RETURN suffix must be parsed and executed");

    assert_eq!(result.rows(), &[vec![Value::String("Alix".into())]]);
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.current_epoch(), epoch.next());
    assert!(!session.in_transaction());
}

#[test]
fn mutation_only_insert_and_create_preserve_legacy_entity_results() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for statement in [
        "INSERT (inserted:LegacyInsert {name: 'inserted'});",
        "CREATE (created:LegacyCreate {name: 'created'});",
    ] {
        let result = session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement} failed: {error:?}"));
        assert_eq!(result.row_count(), 1, "{statement}");
        assert_eq!(result.rows()[0].len(), 1, "{statement}");
        assert!(
            matches!(&result.rows()[0][0], Value::Map(map)
                if map.contains_key(&grafeo_common::types::PropertyKey::new("_id"))),
            "mutation-only entity result changed for {statement}: {:?}",
            result.rows()
        );
    }
    assert_eq!(db.node_count(), 2);
    assert!(!session.in_transaction());
}

#[test]
fn finish_drains_mutations_without_emitting_rows() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let initial_epoch = db.current_epoch();

    let inserted = session
        .execute("INSERT (:FinishMutation {name: 'drained'}) FINISH")
        .expect("FINISH must execute its insertion child");
    assert_eq!(inserted.row_count(), 0);
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.current_epoch(), initial_epoch.next());

    let deleted = session
        .execute("MATCH (n:FinishMutation) DELETE n FINISH")
        .expect("FINISH must execute its deletion child");
    assert_eq!(deleted.row_count(), 0);
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.current_epoch(), initial_epoch.next().next());
    assert!(!session.in_transaction());
}

#[test]
fn void_catalog_procedure_finish_drains_and_exposes_no_columns() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE void_finish() RETURNS () AS { \
             INSERT (:VoidFinish) FINISH }",
        )
        .expect("create a void mutating procedure");
    let initial_epoch = db.current_epoch();

    let result = session
        .execute("CALL void_finish()")
        .expect("FINISH is a zero-column body that must still drain its mutation");

    assert_eq!(result.row_count(), 0);
    assert!(result.columns.is_empty());
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.current_epoch(), initial_epoch.next());
    assert!(!session.in_transaction());
}

#[test]
fn outer_limit_cannot_truncate_a_write_capable_procedure() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE fill_beyond_chunk() RETURNS (ordinal INTEGER) AS { \
             UNWIND range(1, 1025) AS i \
             INSERT (:EagerLimitWrite {ordinal: i}) \
             RETURN i AS ordinal }",
        )
        .expect("create a multi-chunk mutating procedure");
    let initial_epoch = db.current_epoch();

    let result = session
        .execute("CALL fill_beyond_chunk() YIELD ordinal RETURN ordinal LIMIT 1")
        .expect("the invocation boundary must finish before exposing its first row");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0].as_int64(), Some(1));
    assert_eq!(db.node_count(), 1025);
    assert_eq!(db.current_epoch(), initial_epoch.next());
    assert!(!session.in_transaction());
}

#[test]
fn outer_limit_zero_still_completes_a_write_capable_procedure() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE zero_visible_rows() RETURNS (ordinal INTEGER) AS { \
             UNWIND range(1, 3) AS i \
             INSERT (:ZeroLimitWrite {ordinal: i}) \
             RETURN i AS ordinal }",
        )
        .expect("create a mutating procedure for LIMIT 0");
    let initial_epoch = db.current_epoch();

    let result = session
        .execute("CALL zero_visible_rows() YIELD ordinal RETURN ordinal LIMIT 0")
        .expect("LIMIT 0 hides rows, not the procedure invocation");

    assert_eq!(result.row_count(), 0);
    assert_eq!(db.node_count(), 3);
    assert_eq!(db.current_epoch(), initial_epoch.next());
    assert!(!session.in_transaction());
}

#[test]
fn direct_mutation_limit_zero_still_completes_the_write() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let initial_epoch = db.current_epoch();

    let result = session
        .execute(
            "INSERT (n:DirectZeroLimitWrite {name: 'committed'}) \
             RETURN n.name AS name LIMIT 0",
        )
        .expect("LIMIT 0 hides the mutation result, not the mutation");

    assert_eq!(result.row_count(), 0);
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.current_epoch(), initial_epoch.next());
    assert!(!session.in_transaction());
}

#[test]
fn exhaustive_limit_preserves_apply_invocation_multiplicity() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE apply_write() RETURNS (marker INTEGER) AS { \
             INSERT (:ApplyLimitWrite) RETURN 1 AS marker }",
        )
        .expect("create an Apply write procedure");
    let initial_epoch = db.current_epoch();

    let result = session
        .execute(
            "UNWIND range(1, 1025) AS outer \
             CALL apply_write() \
             RETURN outer LIMIT 1",
        )
        .expect("the outer limit must exhaust every semantic Apply invocation");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0].as_int64(), Some(1));
    assert_eq!(db.node_count(), 1025);
    assert_eq!(db.current_epoch(), initial_epoch.next());
    assert!(!session.in_transaction());
}

#[test]
fn exhaustive_limit_zero_preserves_apply_invocation_multiplicity() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE apply_write_zero() RETURNS (marker INTEGER) AS { \
             INSERT (:ApplyZeroLimitWrite) RETURN 1 AS marker }",
        )
        .expect("create an Apply write procedure for LIMIT 0");
    let initial_epoch = db.current_epoch();

    let result = session
        .execute(
            "UNWIND range(1, 1025) AS outer \
             CALL apply_write_zero() \
             RETURN outer LIMIT 0",
        )
        .expect("LIMIT 0 must still execute every reachable Apply invocation");

    assert_eq!(result.row_count(), 0);
    assert_eq!(db.node_count(), 1025);
    assert_eq!(db.current_epoch(), initial_epoch.next());
    assert!(!session.in_transaction());
}

#[test]
fn outer_limit_cannot_hide_a_late_procedure_contract_failure() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE late_contract() RETURNS (value STRING) AS { \
             UNWIND range(1, 1025) AS i \
             INSERT (:LateContractResidue {ordinal: i}) \
             RETURN CASE WHEN i = 1025 THEN 7 ELSE 'ok' END AS value }",
        )
        .expect("create a late-contract-failure fixture");
    let initial_epoch = db.current_epoch();

    let error = session
        .execute("CALL late_contract() YIELD value RETURN value LIMIT 1")
        .expect_err("LIMIT must not commit a prefix or hide the final invalid row");

    assert!(
        contains_type_mismatch(&error, "STRING", "INT64"),
        "expected the late STRING/INT64 mismatch, got: {error:?}"
    );
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.current_epoch(), initial_epoch);
    let residue = session
        .execute("MATCH (:LateContractResidue) RETURN count(*)")
        .expect("the complete failed procedure invocation must roll back");
    assert_eq!(residue.rows()[0][0].as_int64(), Some(0));
    assert!(!session.in_transaction());
}

#[test]
fn outer_limit_zero_cannot_hide_a_procedure_contract_failure() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE invalid_hidden_row() RETURNS (value STRING) AS { \
             INSERT (:ZeroLimitContractResidue) RETURN 7 AS value }",
        )
        .expect("create a LIMIT 0 contract-failure fixture");
    let initial_epoch = db.current_epoch();

    let error = session
        .execute("CALL invalid_hidden_row() YIELD value RETURN value LIMIT 0")
        .expect_err("LIMIT 0 must drain validation and roll the statement back");

    assert!(
        contains_type_mismatch(&error, "STRING", "INT64"),
        "expected the hidden STRING/INT64 mismatch, got: {error:?}"
    );
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.current_epoch(), initial_epoch);
    assert!(!session.in_transaction());
}

#[cfg(all(feature = "wal", feature = "cdc"))]
#[test]
fn failed_catalog_statement_has_no_wal_cdc_or_reopen_residue() {
    use std::path::Path;

    use grafeo_common::types::EpochId;
    use grafeo_engine::cdc::ChangeKind;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, DurabilityMode, GraphModel};
    use grafeo_storage::wal::{LpgMutationOp, WalRecord, WalRecovery};

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(
            Config::persistent(path)
                .with_graph_model(GraphModel::Lpg)
                .with_storage_format(StorageFormat::WalDirectory)
                .with_wal_durability(DurabilityMode::Sync)
                .with_cdc(),
        )
        .expect("open synchronous WAL procedure database")
    }

    let directory = tempfile::tempdir().expect("create procedure WAL directory");
    let path = directory.path().join("procedure-statement-atomicity");
    let committed_epoch;

    {
        let db = open(&path);
        let mut session = db.session();
        session
            .execute(
                "CREATE PROCEDURE wrong_return_durable() RETURNS (name STRING) AS { \
                 INSERT (:WrongReturnResidue {failed_only_key: 'temporary'}) \
                 RETURN 7 AS name }",
            )
            .expect("create durable wrong-return fixture");
        let starting_epoch = db.current_epoch();

        session
            .begin_transaction()
            .expect("begin caller-owned durable transaction");
        session
            .execute("INSERT (:BeforeFailure {kept_only_key: 'preserved'})")
            .expect("stage valid durable work before failing CALL");
        let records_before_call = db.wal().expect("WAL is configured").record_count();
        let error = session
            .execute("CALL wrong_return_durable()")
            .expect_err("runtime return contract must reject the body result");

        assert!(contains_type_mismatch(&error, "STRING", "INT64"));
        assert!(
            db.wal().expect("WAL is configured").record_count() > records_before_call,
            "the failing body must really execute inside a WAL-framed savepoint"
        );
        assert!(session.in_transaction());
        assert_eq!(db.current_epoch(), starting_epoch);
        assert!(
            db.fixture_changes(EpochId::INITIAL..=EpochId::PENDING)
                .expect("read pre-commit CDC")
                .is_empty(),
            "neither valid nor rolled-back transaction writes may leak to CDC before commit"
        );
        let own_before = session
            .execute("MATCH (:BeforeFailure) RETURN count(*)")
            .expect("valid pre-failure work remains visible to its transaction");
        let own_residue = session
            .execute("MATCH (:WrongReturnResidue) RETURN count(*)")
            .expect("rolled-back procedure residue remains queryable as absent");
        assert_eq!(own_before.rows()[0][0].as_int64(), Some(1));
        assert_eq!(own_residue.rows()[0][0].as_int64(), Some(0));

        committed_epoch = session
            .commit()
            .expect("commit only the valid pre-failure statement");
        assert_eq!(committed_epoch, starting_epoch.next());
        let changes = db
            .fixture_changes(committed_epoch..=committed_epoch)
            .expect("read committed procedure CDC boundary");
        assert_eq!(
            changes.len(),
            2,
            "CDC must contain exactly the valid create and property update: {changes:?}"
        );
        let create = changes
            .iter()
            .find(|change| change.kind == ChangeKind::Create)
            .expect("valid node create is present");
        let labels = create
            .labels
            .as_ref()
            .expect("node create CDC carries labels");
        assert!(labels.iter().any(|label| label == "BeforeFailure"));
        assert!(!labels.iter().any(|label| label == "WrongReturnResidue"));
        let update = changes
            .iter()
            .find(|change| change.kind == ChangeKind::Update)
            .expect("valid property update is present");
        assert_eq!(
            update
                .after
                .as_ref()
                .and_then(|properties| properties.get("kept_only_key"))
                .and_then(Value::as_str),
            Some("preserved")
        );
        assert!(changes.iter().all(|change| {
            change
                .labels
                .as_ref()
                .is_none_or(|labels| labels.iter().all(|label| label != "WrongReturnResidue"))
                && change
                    .after
                    .as_ref()
                    .is_none_or(|properties| !properties.contains_key("failed_only_key"))
        }));

        db.wal()
            .expect("WAL is configured")
            .sync()
            .expect("sync WAL");
        let inspection = tempfile::tempdir().expect("inspection directory");
        let inspection_wal = inspection.path().join("wal");
        std::fs::create_dir(&inspection_wal).unwrap();
        for entry in std::fs::read_dir(path.join("wal")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), inspection_wal.join(entry.file_name())).unwrap();
        }
        let recovered = WalRecovery::new(&inspection_wal)
            .unwrap()
            .recover()
            .expect("recover the committed logical WAL stream");
        let operations: Vec<_> = recovered
            .iter()
            .filter_map(|record| match record {
                WalRecord::LpgMutation { op, .. } => Some(op),
                _ => None,
            })
            .collect();
        assert_eq!(
            operations.len(),
            3,
            "recovery must retain exactly the valid node create, property write and label image: {operations:?}"
        );
        assert!(operations.iter().any(|operation| matches!(
            operation,
            LpgMutationOp::CreateNode { labels, .. }
                if labels.iter().any(|label| label == "BeforeFailure")
        )));
        assert!(operations.iter().any(|operation| matches!(
            operation,
            LpgMutationOp::SetNodeProperty { key, value, .. }
                if key == "kept_only_key" && value.as_str() == Some("preserved")
        )));
        assert!(operations.iter().any(|operation| matches!(
            operation,
            LpgMutationOp::NodeLabelImages { birth: true, images, .. }
                if images.len() == 1 && images[0].as_slice() == ["BeforeFailure"]
        )));
        assert!(operations.iter().all(|operation| match operation {
            LpgMutationOp::CreateNode { labels, .. } => {
                labels.iter().all(|label| label != "WrongReturnResidue")
            }
            LpgMutationOp::SetNodeProperty { key, .. } => key != "failed_only_key",
            _ => true,
        }));

        drop(session);
        db.close().expect("close durable procedure database");
    }

    let reopened = open(&path);
    assert_eq!(reopened.current_epoch(), committed_epoch);
    let observer = reopened.session();
    let visible_before = observer
        .execute("MATCH (n:BeforeFailure) RETURN n.kept_only_key")
        .expect("reopen valid statement");
    let visible_residue = observer
        .execute("MATCH (:WrongReturnResidue) RETURN count(*)")
        .expect("reopen without failed procedure residue");
    assert_eq!(visible_before.rows()[0][0].as_str(), Some("preserved"));
    assert_eq!(visible_residue.rows()[0][0].as_int64(), Some(0));
    drop(observer);
    reopened.close().expect("close reopened procedure database");
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;
