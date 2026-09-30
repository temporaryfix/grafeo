//! Canonical RDF graph membership keeps one exact statement lifetime.
#![cfg(all(feature = "triple-store", feature = "sparql"))]
#![allow(missing_docs)]

use grafeo_engine::{Config, GrafeoDB, GraphModel, Quad, Term, Triple};

fn alias(language: &str) -> Triple {
    Triple::new(
        Term::iri("urn:s"),
        Term::iri("urn:p"),
        Term::lang_literal("hello", language),
    )
}

#[test]
fn canonical_aliases_preserve_live_handles_valid_time_and_historical_spellings() {
    for model in [GraphModel::Rdf, GraphModel::Both]
        .into_iter()
        .filter(|model| cfg!(feature = "lpg") || *model == GraphModel::Rdf)
    {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(model)).unwrap();
        let session = db.session();
        session.set_rdf_valid_time_tai_ns(100, 200).unwrap();
        let upper = alias("EN");
        let lower = alias("en");
        let original = [
            Quad::new(upper.clone()),
            Quad::named(upper.clone(), "urn:g"),
        ];
        let aliases = [
            Quad::new(lower.clone()),
            Quad::named(lower.clone(), "urn:g"),
        ];
        assert_eq!(session.insert_rdf_quads(original.clone()).unwrap(), 2);
        let inserted = db.current_epoch();
        let before = db.rdf_history_cut(inserted).unwrap();
        assert_eq!(before.quads.len(), 2);
        assert_eq!(session.insert_rdf_quads(aliases.clone()).unwrap(), 0);
        #[cfg(feature = "ring-index")]
        db.rdf_store().rebuild_ring();
        assert_eq!(
            db.execute_sparql(r#"SELECT ?s WHERE { ?s <urn:p> "hello"@en }"#)
                .unwrap()
                .row_count(),
            1
        );

        for quad in &aliases {
            assert!(session.try_contains_rdf_quad(quad).unwrap());
            let version = before
                .quads
                .iter()
                .find(|v| v.quad.graph() == quad.graph())
                .unwrap();
            assert_eq!(
                db.rdf_statement_handle(quad, version.graph_incarnation)
                    .unwrap(),
                version.statement
            );
        }
        assert_eq!(
            db.execute_sparql("SELECT ?o WHERE { <urn:s> <urn:p> ?o }")
                .unwrap()
                .row_count(),
            1
        );
        db.execute_sparql(r#"DELETE DATA { <urn:s> <urn:p> "hello"@en . GRAPH <urn:g> { <urn:s> <urn:p> "hello"@en } }"#).unwrap();
        assert!(
            db.rdf_history_cut(db.current_epoch())
                .unwrap()
                .quads
                .is_empty()
        );
        assert_eq!(session.insert_rdf_quads(aliases).unwrap(), 2);
        let after = db.rdf_history_cut(db.current_epoch()).unwrap();
        assert!(after.quads.iter().all(|v| v.quad.triple() == &lower));
        assert_eq!(db.rdf_history_cut(inserted).unwrap(), before);
        let history = db.rdf_dataset_history().unwrap();
        assert_eq!(history.quad_versions().len(), 4);
        assert!(history.quad_versions().iter().all(|v| v.valid().is_some()));
        #[cfg(feature = "ring-index")]
        {
            db.rdf_store().rebuild_ring();
            assert_eq!(
                db.execute_sparql("SELECT ?o WHERE { <urn:s> <urn:p> ?o }")
                    .unwrap()
                    .row_count(),
                1
            );
        }
        #[cfg(feature = "lpg")]
        {
            let bytes = db.export_snapshot().unwrap();
            let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
            assert_eq!(restored.rdf_history_cut(inserted).unwrap(), before);
            assert_eq!(
                restored.rdf_history_cut(restored.current_epoch()).unwrap(),
                after
            );
        }
    }
}

#[test]
fn pending_aliases_deduplicate_and_delete_the_visible_representative() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    assert_eq!(
        session
            .insert_rdf_batch([alias("EN"), alias("en")])
            .unwrap(),
        1
    );
    assert_eq!(
        session
            .execute_sparql("SELECT ?o WHERE { <urn:s> <urn:p> ?o }")
            .unwrap()
            .row_count(),
        1
    );
    session
        .execute_sparql(r#"DELETE DATA { <urn:s> <urn:p> "hello"@en }"#)
        .unwrap();
    assert!(
        !session
            .try_contains_rdf_quad(&Quad::new(alias("EN")))
            .unwrap()
    );
    session.commit().unwrap();
    assert!(db.rdf_store().is_empty());
    assert!(db.rdf_dataset_history().unwrap().quad_versions().is_empty());
}

#[test]
fn concurrent_alias_inserts_have_one_committed_winner() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first.insert_rdf_batch([alias("EN")]).unwrap();
    second.insert_rdf_batch([alias("en")]).unwrap();
    first.commit().unwrap();
    assert!(
        second.commit().is_err(),
        "the alias conflict must be rejected before its durable commit marker"
    );
    assert_eq!(
        db.rdf_store().triples().as_slice(),
        &[std::sync::Arc::new(alias("EN"))]
    );
    assert_eq!(db.rdf_dataset_history().unwrap().quad_versions().len(), 1);
}

#[test]
fn datetime_components_keep_exact_operands_across_public_expression_paths() {
    use grafeo_common::types::Value;

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let literal = "\"2024-06-15T10:30:45.1234567890123456789+02:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime>";
    db.execute_sparql(&format!(
        "INSERT DATA {{ <urn:date-subject> <urn:date> {literal} }}"
    ))
    .unwrap();
    let typed = |lexical: &str, datatype: &str| Value::RdfLiteral {
        lexical: lexical.into(),
        datatype: Some(datatype.into()),
        language: None,
    };
    let expected = vec![vec![
        Value::Int64(2024),
        Value::Int64(6),
        Value::Int64(15),
        Value::Int64(10),
        Value::Int64(30),
        typed(
            "45.1234567890123456789",
            "http://www.w3.org/2001/XMLSchema#decimal",
        ),
        typed("PT2H", "http://www.w3.org/2001/XMLSchema#dayTimeDuration"),
        Value::from("+02:00"),
    ]];
    for source in [
        "<urn:date-subject> <urn:date> ?d".to_string(),
        format!("VALUES ?d {{ {literal} }}"),
        format!("BIND({literal} AS ?d)"),
        format!("{{ SELECT DISTINCT ?d WHERE {{ VALUES ?d {{ {literal} {literal} }} }} }}"),
    ] {
        let result = db
            .execute_sparql(&format!(
                "SELECT (YEAR(?d) AS ?y) (MONTH(?d) AS ?m) (DAY(?d) AS ?day) \
             (HOURS(?d) AS ?h) (MINUTES(?d) AS ?min) (SECONDS(?d) AS ?sec) \
             (TIMEZONE(?d) AS ?zone) (TZ(?d) AS ?tz) WHERE {{ {source} }}"
            ))
            .unwrap();
        assert_eq!(result.rows(), expected, "{source}");
    }
    let filtered = db.execute_sparql(
        "SELECT ?s WHERE { ?s <urn:date> ?d FILTER(SECONDS(?d) > 45.1234567890123456788 && SECONDS(?d) < 45.1234567890123456790) }"
    ).unwrap();
    assert_eq!(filtered.rows(), vec![vec![Value::from("urn:date-subject")]]);
    db.execute_sparql(
        "INSERT { ?s <urn:year> ?year } WHERE { ?s <urn:date> ?d BIND(YEAR(?d) AS ?year) FILTER(MONTH(?d) = 6) }"
    ).unwrap();
    assert!(db.rdf_store().contains(&Triple::new(
        Term::iri("urn:date-subject"),
        Term::iri("urn:year"),
        Term::typed_literal("2024", "http://www.w3.org/2001/XMLSchema#integer"),
    )));
}

#[test]
fn datetime_components_preserve_result_datatypes_large_years_and_lexical_offsets() {
    use grafeo_common::types::Value;

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let result = db.execute_sparql(
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
         SELECT (YEAR(?d) AS ?y) (STR(SECONDS(?d)) AS ?sec) \
         (DATATYPE(SECONDS(?d)) AS ?seconds_type) (DATATYPE(TIMEZONE(?d)) AS ?zone_type) \
         (TZ(?d) AS ?tz) WHERE { VALUES ?d { \
         \"123456789012345678901234-12-31T23:59:59.123456789012345678901234+00:00\"^^xsd:dateTime } }"
    ).unwrap();
    assert_eq!(
        result.rows(),
        vec![vec![
            Value::RdfLiteral {
                lexical: "123456789012345678901234".into(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
                language: None,
            },
            Value::from("59.123456789012345678901234"),
            Value::from("http://www.w3.org/2001/XMLSchema#decimal"),
            Value::from("http://www.w3.org/2001/XMLSchema#dayTimeDuration"),
            Value::from("+00:00"),
        ]]
    );
    for (lexical, expected_year, timezone) in [
        ("2024-12-31T24:00:00Z", 2025, "Z"),
        ("-0001-06-15T12:00:00-00:00", -1, "-00:00"),
        ("2024-06-15T12:00:00", 2024, ""),
    ] {
        let result = db.execute_sparql(&format!(
            "SELECT (YEAR(?d) AS ?y) (TZ(?d) AS ?tz) WHERE {{ BIND(\"{lexical}\"^^<http://www.w3.org/2001/XMLSchema#dateTime> AS ?d) }}"
        )).unwrap();
        assert_eq!(
            result.rows(),
            vec![vec![Value::Int64(expected_year), Value::from(timezone)]]
        );
    }
}

#[test]
fn datetime_components_reject_invalid_terms_and_distinguish_absent_timezone() {
    use grafeo_common::types::Value;

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    for operand in [
        "\"2024-02-30T12:00:00Z\"^^<http://www.w3.org/2001/XMLSchema#dateTime>",
        "\"2024-06-15T12:00:00+14:01\"^^<http://www.w3.org/2001/XMLSchema#dateTime>",
        "\"2024-06-15T12:00:00Z\"",
        "\"2024-06-15T12:00:00Z\"@en",
        "<urn:not-a-date>",
    ] {
        let result = db.execute_sparql(&format!(
            "SELECT (YEAR(?d) AS ?y) (SECONDS(?d) AS ?sec) (TIMEZONE(?d) AS ?zone) (TZ(?d) AS ?tz) WHERE {{ VALUES ?d {{ {operand} }} }}"
        )).unwrap();
        assert_eq!(result.rows(), vec![vec![Value::Null; 4]], "{operand}");
    }
    let result = db
        .execute_sparql(
            "SELECT (TIMEZONE(?d) AS ?zone) (TZ(?d) AS ?tz) WHERE { \
         VALUES ?d { \"2024-06-15T12:00:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime> } }",
        )
        .unwrap();
    assert_eq!(result.rows(), vec![vec![Value::Null, Value::from("")]]);
}
