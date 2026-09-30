//! Facade `native` compiles LPG+RDF storage without query-language parsers.
//!
//! ```text
//! cargo test -p grafeo --no-default-features --features native \
//!   --test native_no_parsers -- --test-threads=1
//! ```

#![cfg(feature = "native")]

use grafeo::{Config, GrafeoDB, GraphModel, Quad, Term, Triple};

#[test]
fn facade_native_profile_excludes_parsers() {
    const {
        let all_languages = cfg!(feature = "gql")
            && cfg!(feature = "cypher")
            && cfg!(feature = "sparql")
            && cfg!(feature = "gremlin")
            && cfg!(feature = "graphql")
            && cfg!(feature = "sql-pgq");
        let no_languages = !cfg!(feature = "gql")
            && !cfg!(feature = "cypher")
            && !cfg!(feature = "sparql")
            && !cfg!(feature = "gremlin")
            && !cfg!(feature = "graphql")
            && !cfg!(feature = "sql-pgq");
        assert!(
            all_languages || no_languages,
            "a query-language feature leaked into this facade native_no_parsers build; \
             compile with --no-default-features --features native"
        );
    }
}

#[test]
fn facade_native_both_quads_without_sparql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("both");
    let _ = db.create_node(&["Thing"]);
    let quad = Quad::named(
        Triple::new(
            Term::iri("http://ex.org/s"),
            Term::iri("http://ex.org/p"),
            Term::literal("v"),
        ),
        "http://ex.org/g",
    );
    let (n, _) = db.insert_rdf_quads([quad.clone()]).expect("insert");
    assert_eq!(n, 1);
    assert!(db.contains_rdf_quad(&quad));
}
