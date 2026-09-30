//! Public ordinary-DISTINCT identity and earliest-witness controls.
//! Aggregate DISTINCT has separate resumable-state acceptance.

#[cfg(all(feature = "lpg", feature = "gql"))]
mod native {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;
    use grafeo_engine::database::QueryResult;
    use grafeo_engine::query::executor::{ExecutionOptions, ResultLimits};
    use std::collections::HashMap;

    fn execute_both(values: Vec<Value>) -> [QueryResult; 2] {
        let db = GrafeoDB::new_in_memory();
        let query = "UNWIND $values AS value RETURN DISTINCT value";
        let params = HashMap::from([("values".to_owned(), Value::List(values.into()))]);
        let eager = db.execute_with_params(query, params.clone()).unwrap();
        let streamed = db
            .stream_with_options(query, params, ExecutionOptions::default())
            .unwrap()
            .collect(ResultLimits::default())
            .unwrap();
        [eager, streamed]
    }

    #[test]
    fn integer_and_float_bit_pattern_are_distinct_in_both_routes() {
        // 1.0f64.to_bits() == 4607182418800017408: the old pull key
        // stored both representations in the same Int64 variant.
        let integer = Value::Int64(4_607_182_418_800_017_408);
        let float = Value::Float64(1.0);
        for result in execute_both(vec![
            integer.clone(),
            float.clone(),
            integer.clone(),
            float.clone(),
        ]) {
            assert_eq!(result.rows(), &[vec![integer.clone()], vec![float.clone()]]);
        }
    }

    #[test]
    fn duplicates_across_input_and_output_chunks_keep_first_encounter_order() {
        // More than both the RDF/native 1024-row input and 2048-row output
        // defaults, with later duplicates in a different order. The witness
        // order is deliberately neither ascending nor descending key order.
        let first: Vec<_> = (0..2053)
            .map(|index| Value::Int64((index * 37) % 2053))
            .collect();
        let mut input = first.clone();
        input.extend(first.iter().rev().cloned());
        input.push(Value::Int64(9001));
        input.push(first[0].clone());
        let mut expected: Vec<_> = first.into_iter().map(|value| vec![value]).collect();
        expected.push(vec![Value::Int64(9001)]);
        for result in execute_both(input) {
            assert_eq!(result.rows(), expected.as_slice());
        }
    }

    #[test]
    fn null_empty_list_and_empty_map_are_three_distinct_keys() {
        let list = Value::List(Vec::new().into());
        let map = Value::Map(std::collections::BTreeMap::new().into());
        for result in execute_both(vec![
            Value::Null,
            list.clone(),
            map.clone(),
            Value::Null,
            map.clone(),
            list.clone(),
        ]) {
            assert_eq!(
                result.rows(),
                &[vec![Value::Null], vec![list.clone()], vec![map.clone()]]
            );
        }
    }
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
mod rdf {
    use grafeo_common::types::Value;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};
    use std::fmt::Write;

    fn database() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    #[test]
    fn same_spelled_literal_and_iri_keep_identity_and_first_witness_order() {
        let db = database();
        let result = db
            .execute_sparql(
                r#"SELECT ?term (ISIRI(?term) AS ?iri) WHERE {
                   { SELECT DISTINCT ?term WHERE {
                       VALUES ?term { "urn:x" <urn:x> "urn:x" <urn:x> }
                   } }
               }"#,
            )
            .unwrap();
        // Public strings collide; the outer ISIRI observes the retained
        // hidden identity companion after DISTINCT and a subquery boundary.
        assert_eq!(
            result.rows(),
            &[
                vec![Value::from("urn:x"), Value::Bool(false)],
                vec![Value::from("urn:x"), Value::Bool(true)],
            ]
        );
    }

    #[test]
    fn canonical_language_and_string_terms_deduplicate_without_losing_companions() {
        let db = database();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
               SELECT (STR(?term) AS ?lexical) (LANG(?term) AS ?language)
                      (DATATYPE(?term) AS ?datatype) WHERE {
                   { SELECT DISTINCT ?term WHERE {
                       VALUES ?term {
                           "colour"@EN "plain"^^xsd:string
                           "colour"@en "plain" "colour"@EN
                       }
                   } }
               }"#,
            )
            .unwrap();
        assert_eq!(
            result.rows(),
            &[
                vec![
                    Value::from("colour"),
                    // Term::same_identity ignores language-tag case, while
                    // Literal::language preserves the source spelling. DISTINCT
                    // retains the earliest exact witness, whose tag is `EN`.
                    Value::from("EN"),
                    Value::from("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString")
                ],
                vec![
                    Value::from("plain"),
                    Value::from(""),
                    Value::from("http://www.w3.org/2001/XMLSchema#string")
                ],
            ]
        );
    }

    #[test]
    fn equal_numeric_values_keep_distinct_rdf_lexical_witnesses() {
        let db = database();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
               SELECT (STR(?term) AS ?lexical) (DATATYPE(?term) AS ?datatype) WHERE {
                   { SELECT DISTINCT ?term WHERE {
                       VALUES ?term { "01"^^xsd:integer "1"^^xsd:integer "01"^^xsd:integer }
                   } }
               }"#,
            )
            .unwrap();
        let datatype = Value::from("http://www.w3.org/2001/XMLSchema#integer");
        assert_eq!(
            result.rows(),
            &[
                vec![Value::from("01"), datatype.clone()],
                vec![Value::from("1"), datatype],
            ]
        );
    }

    #[test]
    fn rdf_duplicates_across_chunks_keep_earliest_identity_order() {
        let db = database();
        let mut query = String::from(
            "SELECT ?term (ISIRI(?term) AS ?iri) WHERE { { SELECT DISTINCT ?term WHERE { VALUES ?term { ",
        );
        let mut expected = Vec::new();
        for index in 0..2053 {
            let value = (index * 37) % 2053;
            write!(query, "<urn:key:{value}> ").unwrap();
            expected.push(vec![
                Value::from(format!("urn:key:{value}")),
                Value::Bool(true),
            ]);
        }
        for index in (0..2053).rev() {
            let value = (index * 37) % 2053;
            write!(query, "<urn:key:{value}> ").unwrap();
        }
        query.push_str("\"urn:key:0\" <urn:key:0> \"urn:key:0\" } } } }");
        expected.push(vec![Value::from("urn:key:0"), Value::Bool(false)]);
        let result = db.execute_sparql(&query).unwrap();
        assert_eq!(result.rows(), expected.as_slice());
    }

    #[cfg(feature = "spill")]
    #[test]
    fn rdf_pressure_spill_preserves_exact_companions_and_first_witnesses() {
        use grafeo_common::utils::error::ErrorCode;

        let directory = tempfile::tempdir().unwrap();
        let config = Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(2 << 20)
            .with_spill_path(directory.path());
        let mut query = String::from(
            r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT (STR(?term) AS ?lexical) (ISIRI(?term) AS ?iri)
                   (LANG(?term) AS ?language) (DATATYPE(?term) AS ?datatype)
            WHERE { { SELECT DISTINCT ?term WHERE { VALUES ?term {
                "urn:x" <urn:x> "colour"@EN "plain"^^xsd:string
                "01"^^xsd:integer "1"^^xsd:integer UNDEF "#,
        );
        for index in 0..4096 {
            write!(query, "<urn:pressure:{}> ", (index * 37) % 4096).unwrap();
        }
        query.push_str(
            r#""urn:x" <urn:x> "colour"@en "plain"
                "01"^^xsd:integer "1"^^xsd:integer UNDEF } } } }"#,
        );
        let expected = database().execute_sparql(&query).unwrap();
        assert_eq!(expected.row_count(), 4103);

        // The only blocking operator is DISTINCT. Denying its first framed
        // record proves that this actual caller crosses the spill boundary.
        let configurations = vec![config];
        #[cfg(feature = "encryption")]
        let configurations = {
            let mut configurations = configurations;
            let mut encrypted = configurations[0].clone();
            encrypted.encryption = Some(grafeo_engine::config::EncryptionConfig {
                key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new(
                    [0x2a; 32],
                )),
            });
            configurations.push(encrypted);
            configurations
        };
        for config in configurations {
            let denied =
                GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0)).unwrap();
            let error = denied
                .execute_sparql(&query)
                .expect_err("configured DISTINCT disk quota must deny this pressured caller");
            assert_eq!(error.error_code(), ErrorCode::StorageFull);
            drop(denied);

            let spilled = GrafeoDB::with_config(config).unwrap();
            let actual = spilled.execute_sparql(&query).unwrap();
            assert_eq!(actual.rows(), expected.rows());
            assert_eq!(actual.rows()[2][2], Value::from("EN"));
            assert_eq!(actual.rows()[4][0], Value::from("01"));
        }
    }

    #[test]
    fn unbound_and_zero_column_mappings_each_deduplicate_once() {
        let db = database();
        let unbound = db
            .execute_sparql("SELECT DISTINCT ?x WHERE { VALUES ?x { UNDEF UNDEF UNDEF } }")
            .unwrap();
        assert_eq!(unbound.rows(), &[vec![Value::Null]]);
        let zero_columns = db
            .execute_sparql("SELECT DISTINCT * WHERE { VALUES () { () () () } }")
            .unwrap();
        assert!(zero_columns.columns.is_empty());
        assert_eq!(zero_columns.rows(), &[Vec::<Value>::new()]);
        let empty = db
            .execute_sparql("SELECT DISTINCT * WHERE { VALUES () { } }")
            .unwrap();
        assert!(empty.columns.is_empty());
        assert_eq!(empty.row_count(), 0);
    }
}
