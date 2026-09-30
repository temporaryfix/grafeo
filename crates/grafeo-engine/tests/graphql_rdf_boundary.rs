//! GraphQL-RDF Session boundary regressions.

#![cfg(all(feature = "triple-store", feature = "graphql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, GraphModel, Term, Triple};

fn rdf_db_with_person() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
        .expect("open RDF database");
    db.batch_insert_rdf([
        Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://example.org/Person"),
        ),
        Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/name"),
            Term::literal("Alix"),
        ),
    ])
    .expect("seed RDF person");
    db
}

fn rdf_db_with_people() -> GrafeoDB {
    let db = rdf_db_with_person();
    db.batch_insert_rdf([
        Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/city"),
            Term::literal("Amsterdam"),
        ),
        Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://example.org/Person"),
        ),
        Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://example.org/name"),
            Term::literal("Gus"),
        ),
        Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://example.org/city"),
            Term::literal("Berlin"),
        ),
        Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/friend"),
            Term::iri("http://example.org/gus"),
        ),
        // Same visible string, different RDF term: this must not join to Gus.
        Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/friend"),
            Term::literal("http://example.org/gus"),
        ),
    ])
    .expect("seed distinct subjects and term identities");
    db
}

#[test]
fn graphql_rdf_scalar_filters_and_fragments_join_the_same_subject() {
    let db = rdf_db_with_people();
    let session = db.session();
    let result = session
        .execute_graphql_rdf("query { person { name city } }")
        .expect("join scalar fields");
    assert_eq!(result.columns, ["name", "city"]);
    assert_eq!(result.row_count(), 2);
    for row in [
        vec![Value::from("Alix"), Value::from("Amsterdam")],
        vec![Value::from("Gus"), Value::from("Berlin")],
    ] {
        assert!(result.rows().contains(&row));
    }
    for query in [
        "query { person(name: \"Alix\") { name ... on Person { city } } }",
        "query { person(name: \"Alix\") { name ...City } } fragment City on Person { city }",
    ] {
        let result = session.execute_graphql_rdf(query).expect("join fragment");
        assert_eq!(
            result.rows(),
            &[vec![Value::from("Alix"), Value::from("Amsterdam")]]
        );
    }
}

#[test]
fn graphql_rdf_nested_joins_use_term_identity_not_visible_string_equality() {
    let db = rdf_db_with_people();
    let result = db
        .session()
        .execute_graphql_rdf(
            "query { person(name: \"Alix\") { friend(name: \"Gus\") { name city } } }",
        )
        .expect("join nested bound subjects");
    assert_eq!(
        result.rows(),
        &[vec![Value::from("Gus"), Value::from("Berlin")]]
    );
}

#[test]
fn graphql_rdf_reads_committed_rdf_without_opening_a_transaction() {
    let db = rdf_db_with_person();
    let mut session = db.session();
    session.set_auto_commit(false);

    let result = session
        .execute_graphql_rdf("query { person { name } }")
        .expect("GraphQL-RDF read");
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));

    session
        .begin_transaction()
        .expect("a read-only GraphQL-RDF plan must not open an implicit transaction");
    session.rollback().expect("cleanup transaction");
}

#[test]
fn graphql_rdf_parameters_are_substituted_before_planning() {
    let db = rdf_db_with_person();
    let session = db.session();
    let result = session
        .execute_graphql_rdf_with_params(
            "query Person($wanted: String!) { person(name: $wanted) { name } }",
            HashMap::from([("wanted".to_string(), Value::from("Alix"))]),
        )
        .expect("parameterized GraphQL-RDF read");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));
}

#[test]
fn graphql_rdf_variable_defaults_are_applied_without_parameter_map() {
    let db = rdf_db_with_person();
    let session = db.session();
    let result = session
        .execute_graphql_rdf(
            "query Person($wanted: String = \"Alix\") { person(name: $wanted) { name } }",
        )
        .expect("GraphQL-RDF read with variable default");

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::from("Alix"));
}
