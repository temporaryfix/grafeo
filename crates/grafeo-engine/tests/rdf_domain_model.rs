//! RDF domain model: organisation, interval participation, and separate claims.
//!
//! Valid-time is literals + SPARQL FILTER, not LPG `execute_at_epoch`.
//! Claims live in a named graph so a default-graph role query cannot see them.

#![cfg(all(feature = "triple-store", feature = "sparql"))]

use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PREFIX: &str = r#"
PREFIX k: <https://example.org/ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"#;

fn both_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both))
        .expect("open in-memory dual-model db")
}

#[test]
fn rdf_sparql_org_role_interval_and_claim_graph() {
    let db = both_db();
    db.execute_sparql(&format!(
        r#"
        {PREFIX}
        INSERT DATA {{
          <https://example.org/org/example> a k:Organisation ;
            k:name "Example Manufacturing" .
          <https://example.org/part/example-vehicle> a k:Participation ;
            k:bearer <https://example.org/org/example> ;
            k:role k:Prime ;
            k:instrument <https://example.org/instrument/vehicle> ;
            k:start "2010-01-01" ;
            k:end "2020-01-01" .
        }}
        "#
    ))
    .expect("insert world");

    db.execute_sparql(&format!(
        r#"
        {PREFIX}
        INSERT DATA {{
          GRAPH <https://example.org/graphs/claims> {{
            <https://example.org/claim/award-1> a k:SourceClaim ;
              k:about <https://example.org/org/example> ;
              k:text "Still prime in 2024." .
          }}
        }}
        "#
    ))
    .expect("insert claim graph");

    let inside = db
        .execute_sparql(&format!(
            r#"
            {PREFIX}
            SELECT ?name WHERE {{
              ?p a k:Participation ;
                 k:bearer ?org ;
                 k:start ?start ;
                 k:end ?end .
              ?org k:name ?name .
              FILTER (?start <= "2015-06-01" && ?end > "2015-06-01")
            }}
            "#
        ))
        .expect("role at T-inside");
    assert_eq!(inside.row_count(), 1, "2015 is inside [2010, 2020)");

    let outside = db
        .execute_sparql(&format!(
            r#"
            {PREFIX}
            SELECT ?name WHERE {{
              ?p a k:Participation ;
                 k:bearer ?org ;
                 k:start ?start ;
                 k:end ?end .
              ?org k:name ?name .
              FILTER (?start <= "2024-01-01" && ?end > "2024-01-01")
            }}
            "#
        ))
        .expect("role at T-outside");
    assert_eq!(
        outside.row_count(),
        0,
        "2024 is outside the interval; a claim must not invent a role"
    );

    let leaked = db
        .execute_sparql(&format!(
            r#"
            {PREFIX}
            SELECT ?c WHERE {{
              ?c a k:SourceClaim .
            }}
            "#
        ))
        .expect("default graph must not see claims");
    assert_eq!(
        leaked.row_count(),
        0,
        "claim triples live in GRAPH claims, not the default world"
    );

    let claimed = db
        .execute_sparql(&format!(
            r#"
            {PREFIX}
            SELECT ?text WHERE {{
              GRAPH <https://example.org/graphs/claims> {{
                ?c a k:SourceClaim ; k:text ?text .
              }}
            }}
            "#
        ))
        .expect("read claim graph");
    assert_eq!(claimed.row_count(), 1);
}
