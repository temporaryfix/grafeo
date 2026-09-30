//! Native RDF Serializable-isolation acceptance tests.
//!
//! These tests deliberately use a conservative dataset predicate: an RDF
//! reader must conflict with a concurrent RDF writer even when the two
//! statements differ. That is the minimum sound anti-phantom contract; finer
//! graph/predicate keys may improve concurrency without weakening it.

#![cfg(feature = "triple-store")]

#[cfg(all(feature = "lpg", feature = "sparql"))]
use grafeo_common::types::Value;
use grafeo_core::graph::rdf::{Quad, Term, Triple};
use grafeo_engine::{Config, GrafeoDB, GraphModel, transaction::IsolationLevel};

fn quad(subject: &str) -> Quad {
    Quad::new(Triple::new(
        Term::iri(subject),
        Term::iri("http://example.org/onCall"),
        Term::literal("true"),
    ))
}

#[test]
fn rdf_only_profile_exposes_real_serializable_transactions() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
        .expect("construct RDF-only database");
    let statement = quad("http://example.org/alix");

    let mut session = db.session();
    session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("RDF-only Session must expose opt-in Serializable isolation");
    assert!(
        !session
            .try_contains_rdf_quad(&statement)
            .expect("Serializable RDF membership read"),
        "fresh database must not contain the statement"
    );
    session
        .insert_rdf_quads([statement.clone()])
        .expect("Serializable RDF write");
    assert!(
        session
            .try_contains_rdf_quad(&statement)
            .expect("Serializable RDF read-your-writes"),
        "Serializable RDF transaction must read its own pending write"
    );
    session.commit().expect("commit Serializable RDF write");
    assert!(db.contains_rdf_quad(&statement));
}

#[cfg(feature = "sparql")]
#[test]
fn rdf_write_skew_is_prevented_under_serializable() {
    use grafeo_common::utils::error::{Error, TransactionError};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
        .expect("construct RDF-only database");
    let alix = quad("http://example.org/alix");
    let gus = quad("http://example.org/gus");
    db.insert_rdf_quads([alix.clone(), gus.clone()])
        .expect("seed on-call statements");

    let mut first = db.session();
    first
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin first Serializable transaction");
    let mut second = db.session();
    second
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin second Serializable transaction");

    for session in [&first, &second] {
        assert!(
            session
                .try_contains_rdf_quad(&alix)
                .expect("read Alix predicate")
        );
        assert!(
            session
                .try_contains_rdf_quad(&gus)
                .expect("read Gus predicate")
        );
    }

    first
        .execute_sparql(
            "DELETE DATA { <http://example.org/alix> <http://example.org/onCall> \"true\" . }",
        )
        .expect("first stages removal");
    second
        .execute_sparql(
            "DELETE DATA { <http://example.org/gus> <http://example.org/onCall> \"true\" . }",
        )
        .expect("second stages removal");

    first.commit().expect("first committer succeeds");
    let error = second
        .commit()
        .expect_err("Serializable commit must abort the second side of RDF write skew");
    assert!(
        matches!(
            error,
            Error::Transaction(
                TransactionError::WriteConflict(_) | TransactionError::SerializationFailure(_)
            )
        ),
        "expected a typed transaction conflict, got: {error}"
    );
    assert!(!second.in_transaction());
    assert!(!db.contains_rdf_quad(&alix));

    assert!(
        db.contains_rdf_quad(&gus),
        "the aborted removal must roll back so at least one on-call statement remains"
    );
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { ?s <http://example.org/onCall> \"true\" }")
            .expect("read committed survivors")
            .row_count(),
        1
    );
}

/// Serializable is one database-wide contract, not two unrelated model-local
/// mechanisms. A cycle whose two write legs cross the LPG/RDF boundary must
/// therefore be rejected just like an LPG-only or RDF-only write skew.
#[cfg(all(feature = "lpg", feature = "sparql"))]
#[test]
fn mixed_model_write_skew_is_prevented_under_serializable() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("construct dual-model database");
    let lpg_guard = db.create_node_with_props(&["Guard"], [("enabled", true)]);
    assert!(lpg_guard.is_valid(), "seed LPG guard node");
    let rdf_guard = quad("http://example.org/rdf-guard");
    db.insert_rdf_quads([rdf_guard.clone()])
        .expect("seed RDF guard statement");

    let mut lpg_writer = db.session();
    lpg_writer
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin LPG-writing transaction");
    let mut rdf_writer = db.session();
    rdf_writer
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin RDF-writing transaction");

    for session in [&lpg_writer, &rdf_writer] {
        assert_eq!(
            session.get_node_property(lpg_guard, "enabled"),
            Some(Value::Bool(true)),
            "both transactions must read the LPG side of the invariant"
        );
        assert!(
            session
                .try_contains_rdf_quad(&rdf_guard)
                .expect("read RDF side of invariant"),
            "both transactions must read the RDF side of the invariant"
        );
    }

    lpg_writer
        .set_node_property(lpg_guard, "enabled", Value::Bool(false))
        .expect("stage LPG guard removal");
    rdf_writer
        .execute_sparql(
            "DELETE DATA { <http://example.org/rdf-guard> <http://example.org/onCall> \"true\" . }",
        )
        .expect("stage RDF guard removal");

    lpg_writer.commit().expect("first committer succeeds");
    let error = rdf_writer
        .commit()
        .expect_err("cross-model SSI cycle must abort the second committer");
    assert!(
        error.to_string().contains("Serialization failure"),
        "expected a serialization failure, got: {error}"
    );

    assert!(
        db.contains_rdf_quad(&rdf_guard),
        "the aborted RDF write must roll back, preserving the cross-model invariant"
    );
}
