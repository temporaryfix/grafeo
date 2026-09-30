//! Direct typed-RDF APIs must enforce exact, preflighted graph capabilities.

#![cfg(all(feature = "triple-store", feature = "sparql"))]

use grafeo_core::graph::rdf::{Quad, Term, Triple};
use grafeo_engine::auth::{Grant, Identity, RdfGraphGrant, Role};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
        .expect("create RDF database")
}

fn triple(subject: &str) -> Triple {
    Triple::new(
        Term::iri(subject),
        Term::iri("http://example.org/p"),
        Term::literal("value"),
    )
}

fn assert_denied<T>(result: grafeo_common::utils::error::Result<T>) {
    let Err(error) = result else {
        panic!("operation must be denied");
    };
    assert!(
        error.to_string().contains("permission denied"),
        "unexpected authorization error: {error}"
    );
}

#[test]
fn rdf_grants_are_typed_case_sensitive_and_separate_from_lpg_grants() {
    let db = rdf_db();

    let default_quad = Quad::new(triple("http://example.org/default-subject"));
    let named_default = Quad::named(
        triple("http://example.org/named-default-subject"),
        "default",
    );
    let lower = Quad::named(
        triple("http://example.org/lower-subject"),
        "http://example.org/claims",
    );
    let upper = Quad::named(
        triple("http://example.org/upper-subject"),
        "http://example.org/Claims",
    );

    let default_writer = db.session_with_identity(
        Identity::new("default-writer", [Role::ReadWrite])
            .with_rdf_grants([RdfGraphGrant::default_graph(Role::ReadWrite)]),
    );
    assert_eq!(
        default_writer
            .insert_rdf_batch([default_quad.triple().clone()])
            .unwrap(),
        1
    );
    assert_denied(default_writer.insert_rdf_quads([named_default.clone()]));

    let named_default_writer = db.session_with_identity(
        Identity::new("named-default-writer", [Role::ReadWrite])
            .with_rdf_grants([RdfGraphGrant::named("default", Role::ReadWrite)]),
    );
    assert_eq!(
        named_default_writer
            .insert_rdf_quads([named_default.clone()])
            .unwrap(),
        1
    );
    assert_denied(
        named_default_writer.insert_rdf_batch([triple("http://example.org/denied-default")]),
    );

    let exact_writer = db.session_with_identity(
        Identity::new("exact-writer", [Role::ReadWrite]).with_rdf_grants([RdfGraphGrant::named(
            "http://example.org/claims",
            Role::ReadWrite,
        )]),
    );
    assert_denied(exact_writer.insert_rdf_quads([upper.clone()]));
    assert_eq!(exact_writer.insert_rdf_quads([lower.clone()]).unwrap(), 1);

    let lpg_grant_only = db.session_with_identity(
        Identity::new("lpg-only", [Role::ReadWrite]).with_grants([Grant::new(
            grafeo_common::types::GraphPath::from_components(&["http://example.org/claims"])
                .expect("literal LPG graph path"),
            Role::ReadWrite,
        )]),
    );
    assert_denied(lpg_grant_only.insert_rdf_quads([Quad::named(
        triple("http://example.org/lpg-grant-must-not-authorize-rdf"),
        "http://example.org/claims",
    )]));

    assert!(db.contains_rdf_quad(&default_quad));
    assert!(db.contains_rdf_quad(&named_default));
    assert!(db.contains_rdf_quad(&lower));
    assert!(!db.contains_rdf_quad(&upper));
}

#[test]
fn rdf_exact_reads_enforce_grants_and_infallible_contains_fails_closed() {
    let db = rdf_db();
    let default = Quad::new(triple("http://example.org/read-default"));
    let named_default = Quad::named(triple("http://example.org/read-named-default"), "default");
    let lower = Quad::named(
        triple("http://example.org/read-lower"),
        "http://example.org/claims",
    );
    let upper = Quad::named(
        triple("http://example.org/read-upper"),
        "http://example.org/Claims",
    );
    let admin = db.session();
    assert_eq!(
        admin
            .insert_rdf_quads([
                default.clone(),
                named_default.clone(),
                lower.clone(),
                upper.clone(),
            ])
            .unwrap(),
        4
    );

    let exact_reader = db.session_with_identity(
        Identity::new("exact-reader", [Role::ReadOnly]).with_rdf_grants([RdfGraphGrant::named(
            "http://example.org/claims",
            Role::ReadOnly,
        )]),
    );
    assert!(exact_reader.try_contains_rdf_quad(&lower).unwrap());
    assert_denied(exact_reader.try_contains_rdf_quad(&upper));
    assert!(
        !exact_reader.contains_rdf_quad(&upper),
        "the compatibility contains API must map authorization errors to false"
    );
    assert_denied(exact_reader.try_contains_rdf_quad(&default));

    let default_reader = db.session_with_identity(
        Identity::new("default-reader", [Role::ReadOnly])
            .with_rdf_grants([RdfGraphGrant::default_graph(Role::ReadOnly)]),
    );
    assert!(default_reader.try_contains_rdf_quad(&default).unwrap());
    assert_denied(default_reader.try_contains_rdf_quad(&named_default));

    let named_default_reader = db.session_with_identity(
        Identity::new("named-default-reader", [Role::ReadOnly])
            .with_rdf_grants([RdfGraphGrant::named("default", Role::ReadOnly)]),
    );
    assert!(
        named_default_reader
            .try_contains_rdf_quad(&named_default)
            .unwrap()
    );
    assert_denied(named_default_reader.try_contains_rdf_quad(&default));
}

#[test]
fn rdf_quad_write_preflights_every_target_before_open_transaction_mutation() {
    let db = rdf_db();
    let allowed = Quad::named(
        triple("http://example.org/preflight-allowed"),
        "http://example.org/allowed",
    );
    let forbidden = Quad::named(
        triple("http://example.org/preflight-forbidden"),
        "http://example.org/forbidden",
    );
    let default = Quad::new(triple("http://example.org/preflight-default"));

    let mut writer = db.session_with_identity(
        Identity::new("preflight-writer", [Role::ReadWrite]).with_rdf_grants([
            RdfGraphGrant::named("http://example.org/allowed", Role::ReadWrite),
            RdfGraphGrant::default_graph(Role::ReadWrite),
        ]),
    );
    writer.begin_transaction().unwrap();
    assert_denied(writer.insert_rdf_quads([
        allowed.clone(),
        allowed.clone(),
        default.clone(),
        forbidden,
    ]));
    assert!(
        !writer.try_contains_rdf_quad(&allowed).unwrap(),
        "a denied batch must not leave an allowed prefix pending"
    );
    assert!(
        !writer.try_contains_rdf_quad(&default).unwrap(),
        "a denied batch must not leave a default-graph prefix pending"
    );
    writer.commit().unwrap();

    assert!(!db.contains_rdf_quad(&allowed));
    assert!(!db.contains_rdf_quad(&default));
    db.execute_sparql("CREATE GRAPH <http://example.org/allowed>")
        .expect("preflight denial must not create the allowed named graph");
}
