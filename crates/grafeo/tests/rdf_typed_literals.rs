//! Typed RDF literals: exact BGP matching, not numeric/temporal coercion.
//!
//! ```text
//! cargo test -p grafeo --no-default-features --features rdf \
//!   --test rdf_typed_literals -- --test-threads=1
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo::{Config, GrafeoDB, GraphModel, Quad, Term, Triple};

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).expect("rdf db")
}

fn t(s: &str, o: Term) -> Triple {
    Triple::new(Term::iri(s), Term::iri("http://ex.org/p"), o)
}

fn hits(db: &GrafeoDB, q: &str) -> usize {
    db.execute_sparql(q).unwrap().row_count()
}

/// Quad-inserted `"001"^^xsd:integer` is visible to SPARQL with that lexical form.
#[test]
fn quad_xsd_integer_001_exact_sparql() {
    let db = rdf_db();
    db.insert_rdf_quads([Quad::new(t(
        "http://ex.org/s",
        Term::typed_literal("001", "http://www.w3.org/2001/XMLSchema#integer"),
    ))])
    .unwrap();
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> "001"^^<http://www.w3.org/2001/XMLSchema#integer> }"#,
        ),
        1,
        "lexical 001^^integer must match SELECT"
    );
    assert_eq!(
        hits(
            &db,
            r#"ASK { <http://ex.org/s> <http://ex.org/p> "001"^^<http://www.w3.org/2001/XMLSchema#integer> }"#,
        ),
        1,
        "lexical 001^^integer must match ASK"
    );
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> "1"^^<http://www.w3.org/2001/XMLSchema#integer> }"#,
        ),
        0,
        "canonical 1^^integer must not match 001"
    );
}

/// Opaque custom datatype is exact-match only.
#[test]
fn quad_opaque_datatype_exact_sparql() {
    let db = rdf_db();
    db.insert_rdf_quads([Quad::new(t(
        "http://ex.org/s",
        Term::typed_literal("foo", "http://ex.org/dt"),
    ))])
    .unwrap();
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> "foo"^^<http://ex.org/dt> }"#,
        ),
        1
    );
    assert_eq!(
        hits(
            &db,
            r#"ASK { <http://ex.org/s> <http://ex.org/p> "foo"^^<http://ex.org/dt> }"#
        ),
        1
    );
    assert_eq!(
        hits(&db, r#"SELECT ?s WHERE { ?s <http://ex.org/p> "foo" }"#),
        0,
        "plain string must not match typed opaque"
    );
}

/// Decimal lexical form is preserved (`1.10` ≠ `1.1`).
#[test]
fn quad_decimal_lexical_exact() {
    let db = rdf_db();
    db.insert_rdf_quads([Quad::new(t(
        "http://ex.org/s",
        Term::typed_literal("1.10", "http://www.w3.org/2001/XMLSchema#decimal"),
    ))])
    .unwrap();
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> "1.10"^^<http://www.w3.org/2001/XMLSchema#decimal> }"#,
        ),
        1
    );
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> "1.1"^^<http://www.w3.org/2001/XMLSchema#decimal> }"#,
        ),
        0
    );
}

/// date / time / dateTime lexical forms are exact in BGPs.
#[test]
fn quad_date_datetime_exact() {
    let db = rdf_db();
    db.insert_rdf_quads([
        Quad::new(t(
            "http://ex.org/d",
            Term::typed_literal("2020-01-01", "http://www.w3.org/2001/XMLSchema#date"),
        )),
        Quad::new(t(
            "http://ex.org/tm",
            Term::typed_literal("12:00:00", "http://www.w3.org/2001/XMLSchema#time"),
        )),
        Quad::new(t(
            "http://ex.org/dt",
            Term::typed_literal(
                "2020-01-01T00:00:00Z",
                "http://www.w3.org/2001/XMLSchema#dateTime",
            ),
        )),
    ])
    .unwrap();
    assert_eq!(
        hits(
            &db,
            r#"ASK { <http://ex.org/d> <http://ex.org/p> "2020-01-01"^^<http://www.w3.org/2001/XMLSchema#date> }"#,
        ),
        1
    );
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> "12:00:00"^^<http://www.w3.org/2001/XMLSchema#time> }"#,
        ),
        1
    );
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> "2020-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> }"#,
        ),
        1
    );
}

/// SPARQL INSERT DATA typed literal is exactly in the store.
#[test]
fn sparql_insert_typed_integer_exact_membership() {
    let db = rdf_db();
    db.execute_sparql(
        r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "001"^^<http://www.w3.org/2001/XMLSchema#integer> }"#,
    )
    .unwrap();
    let found = db
        .rdf_store()
        .find(&grafeo_core::graph::rdf::TriplePattern {
            subject: Some(Term::iri("http://ex.org/s")),
            predicate: Some(Term::iri("http://ex.org/p")),
            object: Some(Term::typed_literal(
                "001",
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        });
    assert_eq!(
        found.len(),
        1,
        "SPARQL INSERT must store lexical 001^^integer, not coerced 1"
    );
    assert_eq!(
        hits(
            &db,
            r#"ASK { <http://ex.org/s> <http://ex.org/p> "001"^^<http://www.w3.org/2001/XMLSchema#integer> }"#,
        ),
        1,
        "SPARQL INSERT then ASK exact membership"
    );
}

/// Exact read-only membership is lexical+datatype identity, not numeric coercion.
#[test]
fn contains_rdf_quad_exact_typed_membership() {
    let db = rdf_db();
    let exact = Quad::new(t(
        "http://ex.org/s",
        Term::typed_literal("001", "http://www.w3.org/2001/XMLSchema#integer"),
    ));
    let canonical = Quad::new(t(
        "http://ex.org/s",
        Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
    ));
    db.insert_rdf_quads([exact.clone()]).unwrap();
    assert!(
        db.contains_rdf_quad(&exact),
        "committed typed quad must be visible to the membership API"
    );
    assert!(
        !db.contains_rdf_quad(&canonical),
        "canonical 1^^integer must not match stored 001"
    );
    let session = db.session();
    assert!(session.contains_rdf_quad(&exact));
    assert!(!session.contains_rdf_quad(&canonical));
}

/// FILTER still compares integers in the numeric value space.
#[test]
fn filter_still_coerces_integer() {
    let db = rdf_db();
    db.insert_rdf_quads([Quad::new(t(
        "http://ex.org/s",
        Term::typed_literal("001", "http://www.w3.org/2001/XMLSchema#integer"),
    ))])
    .unwrap();
    assert_eq!(
        hits(
            &db,
            r#"SELECT ?s WHERE { ?s <http://ex.org/p> ?o FILTER(?o > 0) }"#,
        ),
        1,
        "FILTER numeric comparison must still see 001 as integer"
    );
}
