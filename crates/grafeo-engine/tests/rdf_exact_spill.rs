//! Public RDF ORDER BY pressure and exact companion-order witnesses.
#![cfg(all(feature = "triple-store", feature = "sparql"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::ErrorCode;
use grafeo_engine::{Config, GrafeoDB, GraphModel};
use std::fmt::Write;

fn order_query() -> String {
    let mut query = String::from("SELECT ?key ?ordinal WHERE { VALUES (?key ?ordinal) { ");
    for key in 0..4096 {
        write!(query, "(\"k{key:04}\" 0) (\"k{key:04}\" 1) ").unwrap();
    }
    query.push_str("} } ORDER BY ASC(?key) DESC(?ordinal)");
    query
}

#[cfg(feature = "spill")]
fn expected_rows() -> Vec<Vec<Value>> {
    (0..4096)
        .flat_map(|key| {
            [
                vec![Value::from(format!("k{key:04}")), Value::Int64(1)],
                vec![Value::from(format!("k{key:04}")), Value::Int64(0)],
            ]
        })
        .collect()
}

#[test]
fn rdf_order_by_resident_admission_is_structured() {
    let database = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(2 << 20),
    )
    .unwrap();
    let error = database
        .execute_sparql(&order_query())
        .expect_err("RDF ORDER BY must exceed the resident budget");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
}

#[cfg(feature = "spill")]
#[test]
fn rdf_order_by_spill_matches_exact_plaintext_and_encrypted_results() {
    let query = order_query();
    let expected = expected_rows();
    let baseline = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
        .unwrap()
        .execute_sparql(&query)
        .unwrap();
    assert_eq!(baseline.rows(), expected.as_slice());

    let directory = tempfile::tempdir().unwrap();
    let base = Config::in_memory()
        .with_graph_model(GraphModel::Rdf)
        .with_memory_limit(2 << 20)
        .with_spill_path(directory.path());
    let denied = GrafeoDB::with_config(base.clone().with_max_query_spill_bytes(0)).unwrap();
    let error = denied
        .execute_sparql(&query)
        .expect_err("RDF ORDER BY must cross the configured spill quota");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
    assert!(
        error.to_string().contains("spill disk quota exceeded"),
        "{error}"
    );
    drop(denied);

    let configurations = vec![base.clone()];
    #[cfg(feature = "encryption")]
    let configurations = {
        let mut configurations = configurations;
        let mut encrypted = base;
        encrypted.encryption = Some(grafeo_engine::config::EncryptionConfig {
            key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new([0x2a; 32])),
        });
        configurations.push(encrypted);
        configurations
    };
    for config in configurations {
        let database = GrafeoDB::with_config(config).unwrap();
        let result = database.execute_sparql(&query).unwrap();
        assert_eq!(result.rows(), expected.as_slice());
    }
}

#[test]
fn rdf_order_by_rejects_one_oversized_value_without_partial_result() {
    let value = "x".repeat((2 << 20) + 1024);
    let query = format!("SELECT ?key WHERE {{ VALUES ?key {{ \"{value}\" }} }} ORDER BY ?key");

    let resident = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(2 << 20),
    )
    .unwrap();
    let error = resident
        .execute_sparql(&query)
        .expect_err("oversized resident RDF sort must fail before output");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);

    #[cfg(feature = "spill")]
    {
        let directory = tempfile::tempdir().unwrap();
        let configured = GrafeoDB::with_config(
            Config::in_memory()
                .with_graph_model(GraphModel::Rdf)
                .with_memory_limit(2 << 20)
                .with_spill_path(directory.path())
                .with_max_query_spill_bytes(64 << 20),
        )
        .unwrap();
        let error = configured
            .execute_sparql(&query)
            .expect_err("one oversized row must not be admitted to spill");
        assert_eq!(error.error_code(), ErrorCode::StorageFull);
        let namespace = directory
            .path()
            .join(format!("grafeo-store-{}", configured.store_id()));
        let entries: Vec<_> = std::fs::read_dir(namespace)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            entries
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            [
                ".grafeo-spill-quota",
                ".grafeo-spill-quota.lock",
                ".grafeo-spill-root"
            ]
            .map(std::ffi::OsString::from)
            .into_iter()
            .collect()
        );
    }
}

fn many_groups_fixture() -> (String, Vec<Vec<Value>>) {
    let mut groups = String::from(
        "SELECT ?group (SUM(?value) AS ?sum) (AVG(?value) AS ?avg) (COUNT(?value) AS ?count) WHERE { VALUES (?group ?value) { ",
    );
    for group in 0..1024 {
        for value in 1..=4 {
            write!(groups, "(\"g{group:04}\" {value}) ").unwrap();
        }
    }
    groups.push_str("} } GROUP BY ?group");
    let expected_groups: Vec<_> = (0..1024)
        .map(|group| {
            vec![
                Value::from(format!("g{group:04}")),
                Value::Int64(10),
                Value::RdfLiteral {
                    lexical: "2.5".into(),
                    language: None,
                    datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
                },
                Value::Int64(4),
            ]
        })
        .collect();
    (groups, expected_groups)
}

fn hot_distinct_fixture() -> (String, Vec<Vec<Value>>) {
    let mut distinct = String::from(
        "SELECT (COUNT(?value) AS ?all_count) (COUNT(DISTINCT ?value) AS ?count) (SUM(DISTINCT ?value) AS ?sum) (SUM(?value) AS ?all_sum) WHERE { VALUES ?value { ",
    );
    for value in 0..4096 {
        write!(distinct, "{value} {value} ").unwrap();
    }
    distinct.push_str("} }");
    (
        distinct,
        vec![vec![
            Value::Int64(8192),
            Value::Int64(4096),
            Value::Int64(8_386_560),
            Value::Int64(16_773_120),
        ]],
    )
}

#[cfg(feature = "spill")]
fn typed_aggregate_fixture() -> (String, Vec<Vec<Value>>) {
    let mut query = String::from(
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> SELECT ?kind ?group (SUM(?value) AS ?sum) (AVG(?value) AS ?avg) (SAMPLE(?ordinal) AS ?sample) (GROUP_CONCAT(STR(?ordinal); SEPARATOR=\"|\") AS ?ordinals) WHERE { VALUES (?kind ?group ?value ?ordinal) { ",
    );
    for group in 0..16 {
        write!(query, "(\"cancel\" \"c{group:03}\" \"10000000000000000\"^^xsd:double 0) (\"cancel\" \"c{group:03}\" \"1\"^^xsd:double 1) (\"cancel\" \"c{group:03}\" \"-10000000000000000\"^^xsd:double 2) ").unwrap();
    }
    for group in 0..512 {
        write!(query, "(\"decimal\" \"d{group:03}\" \"0.1\"^^xsd:decimal 0) (\"decimal\" \"d{group:03}\" \"0.2\"^^xsd:decimal 1) ").unwrap();
    }
    for group in 0..512 {
        write!(query, "(\"float\" \"f{group:03}\" \"1.5\"^^xsd:float 0) (\"float\" \"f{group:03}\" \"2.5\"^^xsd:float 1) ").unwrap();
    }
    for group in 0..512 {
        write!(query, "(\"integer\" \"i{group:03}\" \"18446744073709551615\"^^xsd:unsignedLong 0) (\"integer\" \"i{group:03}\" 1 1) ").unwrap();
    }
    query.push_str("} } GROUP BY ?kind ?group");
    let integer_sum = Value::RdfLiteral {
        lexical: "18446744073709551616".into(),
        language: None,
        datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
    };
    let integer_avg = Value::RdfLiteral {
        lexical: "9223372036854775808.0".into(),
        language: None,
        datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
    };
    let decimal_sum = Value::RdfLiteral {
        lexical: "0.3".into(),
        language: None,
        datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
    };
    let decimal_avg = Value::RdfLiteral {
        lexical: "0.15".into(),
        language: None,
        datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
    };
    let expected = ["cancel", "decimal", "float", "integer"]
        .iter()
        .flat_map(|kind| {
            let decimal_sum = decimal_sum.clone();
            let decimal_avg = decimal_avg.clone();
            let integer_sum = integer_sum.clone();
            let integer_avg = integer_avg.clone();
            let limit = if *kind == "cancel" { 16 } else { 512 };
            (0..limit).map(move |group| match *kind {
                "cancel" => vec![
                    Value::from("cancel"),
                    Value::from(format!("c{group:03}")),
                    Value::Float64(0.0),
                    Value::Float64(0.0),
                    Value::Int64(0),
                    Value::from("0|1|2"),
                ],
                "decimal" => vec![
                    Value::from("decimal"),
                    Value::from(format!("d{group:03}")),
                    decimal_sum.clone(),
                    decimal_avg.clone(),
                    Value::Int64(0),
                    Value::from("0|1"),
                ],
                "float" => vec![
                    Value::from("float"),
                    Value::from(format!("f{group:03}")),
                    Value::RdfLiteral {
                        lexical: "4".into(),
                        language: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#float".into()),
                    },
                    Value::RdfLiteral {
                        lexical: "2".into(),
                        language: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#float".into()),
                    },
                    Value::Int64(0),
                    Value::from("0|1"),
                ],
                "integer" => vec![
                    Value::from("integer"),
                    Value::from(format!("i{group:03}")),
                    integer_sum.clone(),
                    integer_avg.clone(),
                    Value::Int64(0),
                    Value::from("0|1"),
                ],
                _ => unreachable!(),
            })
        })
        .collect();
    (query, expected)
}

#[cfg(feature = "spill")]
#[test]
fn rdf_aggregate_spill_preserves_typed_numeric_and_encounter_order() {
    let (query, expected) = typed_aggregate_fixture();
    let reference = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(64 << 20),
    )
    .unwrap();
    assert_exact_rows(reference.execute_sparql(&query).unwrap().rows(), &expected);
    assert_pressure_semantics_with_budget(&query, &expected, 4 << 20);
}

#[test]
fn rdf_aggregate_many_groups_resident_denial_is_structured() {
    let (query, expected) = many_groups_fixture();
    let reference = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(64 << 20),
    )
    .unwrap();
    assert_exact_rows(reference.execute_sparql(&query).unwrap().rows(), &expected);
    let resident = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(2 << 20),
    )
    .unwrap();
    let error = resident
        .execute_sparql(&query)
        .expect_err("many-group aggregate must exceed resident budget");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
}

#[test]
fn rdf_aggregate_hot_distinct_resident_denial_is_structured() {
    let (query, expected) = hot_distinct_fixture();
    let reference = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(64 << 20),
    )
    .unwrap();
    assert_exact_rows(reference.execute_sparql(&query).unwrap().rows(), &expected);
    let resident = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(2 << 20),
    )
    .unwrap();
    let error = resident
        .execute_sparql(&query)
        .expect_err("hot distinct aggregate must exceed resident budget");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
}

#[cfg(feature = "spill")]
#[test]
fn rdf_aggregate_spill_preserves_many_groups() {
    let (query, expected) = many_groups_fixture();
    assert_pressure_semantics(&query, &expected);
}

#[cfg(feature = "spill")]
#[test]
fn rdf_aggregate_spill_preserves_hot_distinct_membership() {
    let (query, expected) = hot_distinct_fixture();
    assert_pressure_semantics(&query, &expected);
}

#[test]
fn rdf_group_concat_oversized_final_scalar_is_denied_without_prefix() {
    let piece = "x".repeat(1024);
    let mut query = String::from(
        "SELECT (GROUP_CONCAT(?value; SEPARATOR=\"\") AS ?joined) WHERE { VALUES ?value { ",
    );
    for _ in 0..2049 {
        write!(query, "\"{piece}\" ").unwrap();
    }
    query.push_str("} }");

    let resident = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(2 << 20),
    )
    .unwrap();
    let error = resident
        .execute_sparql(&query)
        .expect_err("oversized GROUP_CONCAT must fail resident admission");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
    #[cfg(feature = "spill")]
    {
        let directory = tempfile::tempdir().unwrap();
        let configured = GrafeoDB::with_config(
            Config::in_memory()
                .with_graph_model(GraphModel::Rdf)
                .with_memory_limit(2 << 20)
                .with_spill_path(directory.path())
                .with_max_query_spill_bytes(64 << 20),
        )
        .unwrap();
        let error = configured
            .execute_sparql(&query)
            .expect_err("oversized GROUP_CONCAT must fail final scalar admission");
        assert_eq!(error.error_code(), ErrorCode::StorageFull);
        let namespace = directory
            .path()
            .join(format!("grafeo-store-{}", configured.store_id()));
        let entries: Vec<_> = std::fs::read_dir(namespace)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            entries
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            [
                ".grafeo-spill-quota",
                ".grafeo-spill-quota.lock",
                ".grafeo-spill-root"
            ]
            .map(std::ffi::OsString::from)
            .into_iter()
            .collect()
        );
    }
}

fn assert_exact_rows(actual: &[Vec<Value>], expected: &[Vec<Value>]) {
    assert_eq!(actual.len(), expected.len(), "row count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "row {index}");
    }
}

#[cfg(feature = "spill")]
fn assert_pressure_semantics(query: &str, expected: &[Vec<Value>]) {
    assert_pressure_semantics_with_budget(query, expected, 2 << 20);
}

#[cfg(feature = "spill")]
fn assert_pressure_semantics_with_budget(query: &str, expected: &[Vec<Value>], budget: usize) {
    let baseline = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
        .unwrap()
        .execute_sparql(query)
        .unwrap();
    assert_exact_rows(baseline.rows(), expected);
    assert_eq!(
        baseline.columns.len(),
        expected[0].len(),
        "hidden key leaked"
    );
    let directory = tempfile::tempdir().unwrap();
    let config = Config::in_memory()
        .with_graph_model(GraphModel::Rdf)
        .with_memory_limit(budget)
        .with_spill_path(directory.path());
    let denied = GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0)).unwrap();
    let error = denied
        .execute_sparql(query)
        .expect_err("semantic fixture must spill");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
    assert!(
        error.to_string().contains("spill disk quota exceeded"),
        "{error}"
    );
    drop(denied);
    let configurations = vec![config.clone()];
    #[cfg(feature = "encryption")]
    let configurations = {
        let mut configurations = configurations;
        let mut encrypted = config;
        encrypted.encryption = Some(grafeo_engine::config::EncryptionConfig {
            key_chain: std::sync::Arc::new(grafeo_common::encryption::KeyChain::new([0x2a; 32])),
        });
        configurations.push(encrypted);
        configurations
    };
    for config in configurations {
        let database = GrafeoDB::with_config(config).unwrap();
        for _ in 0..2 {
            let result = database.execute_sparql(query).unwrap();
            assert_exact_rows(result.rows(), expected);
            assert_eq!(result.columns.len(), expected[0].len());
        }
    }
}

#[cfg(feature = "spill")]
#[test]
fn rdf_spill_exact_categories_numeric_dates_hidden_keys_and_directions() {
    // Ordered labels make the assertions independent of the baseline. The
    // visible label and hidden key deliberately have different sort orders.
    let cases = [
        ("u", "UNDEF"),
        ("i", "<urn:a>"),
        ("d", "\"1.25\"^^xsd:decimal"),
        ("f", "\"2\"^^xsd:float"),
        ("n", "\"3\"^^xsd:double"),
        ("b", "\"9007199254740992\"^^xsd:double"),
        ("a", "\"9007199254740993\"^^xsd:integer"),
        ("l", "\"colour\"@EN"),
        ("s", "\"urn:a\""),
        ("t", "\"1999-12-31Z\"^^xsd:date"),
        ("r", "\"2000-01-01Z\"^^xsd:date"),
    ];
    for descending in [false, true] {
        let mut query = String::from(
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> SELECT ?label ?ordinal (ISIRI(?key) AS ?iri) (LANG(?key) AS ?language) (DATATYPE(?key) AS ?datatype) (STR(?key) AS ?lexical) WHERE { { SELECT ?label ?key ?ordinal WHERE { VALUES (?label ?key ?ordinal) { ",
        );
        for ordinal in (0..1024).rev() {
            for (label, key) in cases.iter().rev() {
                write!(query, "(\"{label}\" {key} {ordinal}) ").unwrap();
            }
        }
        query.push_str("} } ORDER BY ASC(?ordinal) ");
        query.push_str(if descending { "DESC" } else { "ASC" });
        query.push_str("(?key) LIMIT 22 } }");
        let mut labels: Vec<_> = cases.iter().map(|(label, _)| *label).collect();
        if descending {
            labels.reverse();
        }
        let expected: Vec<_> = (0..2)
            .flat_map(|ordinal| {
                labels.iter().map(move |label| {
                    let (iri, language, datatype, lexical) = match *label {
                        "u" => (Value::Null, Value::Null, Value::Null, Value::Null),
                        "i" => (
                            Value::Bool(true),
                            Value::Null,
                            Value::Null,
                            Value::from("urn:a"),
                        ),
                        "d" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#decimal"),
                            Value::from("1.25"),
                        ),
                        "f" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#float"),
                            Value::from("2"),
                        ),
                        "n" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#double"),
                            Value::from("3"),
                        ),
                        "b" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#double"),
                            Value::from("9007199254740992"),
                        ),
                        "a" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#integer"),
                            Value::from("9007199254740993"),
                        ),
                        "l" => (
                            Value::Bool(false),
                            Value::from("EN"),
                            Value::from("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"),
                            Value::from("colour"),
                        ),
                        "s" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#string"),
                            Value::from("urn:a"),
                        ),
                        "t" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#date"),
                            Value::from("1999-12-31Z"),
                        ),
                        "r" => (
                            Value::Bool(false),
                            Value::from(""),
                            Value::from("http://www.w3.org/2001/XMLSchema#date"),
                            Value::from("2000-01-01Z"),
                        ),
                        _ => unreachable!(),
                    };
                    vec![
                        Value::from(*label),
                        Value::Int64(ordinal),
                        iri,
                        language,
                        datatype,
                        lexical,
                    ]
                })
            })
            .collect();
        // Full companion rows need more headroom than plain values. Use an
        // unambiguously insufficient budget for denial, then prove that the
        // positive budget still crosses the actual spill boundary.
        let directory = tempfile::tempdir().unwrap();
        let constrained = GrafeoDB::with_config(
            Config::in_memory()
                .with_graph_model(GraphModel::Rdf)
                .with_memory_limit(1 << 20)
                .with_spill_path(directory.path()),
        )
        .unwrap();
        let error = constrained
            .execute_sparql(&query)
            .expect_err("full companion rows exceed this smaller resident budget");
        assert_eq!(error.error_code(), ErrorCode::StorageFull);
        assert!(
            error.to_string().contains("resident-memory limit exceeded"),
            "{error}"
        );
        drop(constrained);
        assert_pressure_semantics_with_budget(&query, &expected, 3 << 20);
    }
}

#[cfg(feature = "spill")]
#[test]
fn rdf_spill_expression_error_is_unbound_and_equal_keys_are_stable() {
    for descending in [false, true] {
        let mut query = String::from("SELECT ?ordinal WHERE { VALUES (?denominator ?ordinal) { ");
        // Equal keys span multiple input chunks and runs. Expression errors
        // sort at the same low end as UNDEF, preserving input order for ties.
        for ordinal in 0..8192 {
            write!(
                query,
                "({} {ordinal}) ",
                if ordinal % 2 == 0 { "0" } else { "1" }
            )
            .unwrap();
        }
        write!(
            query,
            "}} }} ORDER BY {}(1 / ?denominator) OFFSET 2040 LIMIT 32",
            if descending { "DESC" } else { "ASC" }
        )
        .unwrap();
        let first = i64::from(descending);
        let expected: Vec<_> = (0..32)
            .map(|index| vec![Value::Int64(first + (index + 2040) * 2)])
            .collect();
        assert_pressure_semantics(&query, &expected);
    }
}

#[cfg(feature = "spill")]
#[test]
fn rdf_spill_projected_volatile_key_preserves_visible_values() {
    let mut query = String::from("SELECT ?random ?ordinal WHERE { VALUES ?ordinal { ");
    for ordinal in 0..8192 {
        write!(query, "{ordinal} ").unwrap();
    }
    query.push_str("} BIND(RAND() AS ?random) } ORDER BY ?random LIMIT 64");
    let directory = tempfile::tempdir().unwrap();
    let config = Config::in_memory()
        .with_graph_model(GraphModel::Rdf)
        .with_memory_limit(2 << 20)
        .with_spill_path(directory.path());
    let denied = GrafeoDB::with_config(config.clone().with_max_query_spill_bytes(0)).unwrap();
    let error = denied
        .execute_sparql(&query)
        .expect_err("volatile sort must spill");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
    assert!(
        error.to_string().contains("spill disk quota exceeded"),
        "{error}"
    );
    drop(denied);
    let database = GrafeoDB::with_config(config).unwrap();
    let result = database.execute_sparql(&query).unwrap();
    assert_eq!(result.columns, ["random", "ordinal"]);
    assert_eq!(result.rows().len(), 64);
    let mut previous = 0.0;
    for row in result.rows() {
        let Value::Float64(random) = row[0] else {
            panic!("RAND must remain numeric")
        };
        assert!((0.0..1.0).contains(&random));
        assert!(
            random >= previous,
            "visible RAND values must remain sorted after sorting"
        );
        previous = random;
        assert!(matches!(row[1], Value::Int64(0..=8191)));
    }
}

#[test]
fn rdf_unbound_term_observables_remain_expression_errors() {
    let database =
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let result = database.execute_sparql("SELECT (ISIRI(?x) AS ?iri) (LANG(?x) AS ?lang) (DATATYPE(?x) AS ?dt) (STR(?x) AS ?s) WHERE { VALUES ?x { UNDEF } }").unwrap();
    assert_eq!(
        result.rows(),
        &[vec![Value::Null, Value::Null, Value::Null, Value::Null]]
    );
}

#[cfg(feature = "spill")]
#[test]
fn rdf_aggregate_params_and_profile_preserve_forced_spill_contract() {
    use grafeo_core::graph::rdf::{Term, Triple};
    use std::collections::HashMap;

    let (_, expected) = many_groups_fixture();
    // Retain the same 1024 groups and four operands per group, with a compact
    // physical plan. Thousands of inline VALUES rows expand into BIND/UNION
    // nodes whose full PROFILE diagnostic legitimately exceeds this budget.
    let query = "SELECT ?group (SUM(?value) AS ?sum) (AVG(?value) AS ?avg) (COUNT(?value) AS ?count) WHERE { ?row <urn:fixture:group> ?group . VALUES ?value { 1 2 3 4 } } GROUP BY ?group".to_string();
    let fixture = || {
        (0..1024).map(|group| {
            Triple::new(
                Term::iri(format!("urn:fixture:row:{group}")),
                Term::iri("urn:fixture:group"),
                Term::literal(format!("g{group:04}")),
            )
        })
    };
    let profiled = format!("EXPLAIN ANALYZE {query}");
    // SPARQL parameters currently select the substitution entry point but do
    // not introduce parameter expressions into the translated logical plan.
    let params = || HashMap::from([("route_probe".to_owned(), Value::Int64(1))]);
    for denied in [true, false] {
        let directory = tempfile::tempdir().unwrap();
        let mut config = Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            // Only disk quota changes between denial and success: every route
            // must execute the same fixture under forced-spill pressure.
            .with_memory_limit(2 << 20)
            .with_spill_path(directory.path());
        if denied {
            config = config.with_max_query_spill_bytes(0);
        }
        for profile in [false, true] {
            let text = if profile {
                profiled.as_str()
            } else {
                query.as_str()
            };
            for database_route in [false, true] {
                let database = GrafeoDB::with_config(config.clone()).unwrap();
                assert_eq!(database.batch_insert_rdf(fixture()).unwrap(), 1024);
                let session = database.session();
                let resident_before = database.buffer_manager().allocated();
                let assert_cleanup = || {
                    assert_eq!(database.buffer_manager().allocated(), resident_before);
                    let namespace = directory
                        .path()
                        .join(format!("grafeo-store-{}", database.store_id()));
                    let entries: Vec<_> = std::fs::read_dir(namespace)
                        .unwrap()
                        .map(|entry| entry.unwrap().file_name())
                        .collect();
                    assert_eq!(
                        entries
                            .into_iter()
                            .collect::<std::collections::BTreeSet<_>>(),
                        [
                            ".grafeo-spill-quota",
                            ".grafeo-spill-quota.lock",
                            ".grafeo-spill-root"
                        ]
                        .map(std::ffi::OsString::from)
                        .into_iter()
                        .collect()
                    );
                };
                let result = if database_route {
                    database.execute_language(text, "sparql", Some(params()))
                } else {
                    session.execute_sparql_with_params(text, params())
                };
                if denied {
                    let error = result.expect_err("every route must reach spill quota denial");
                    assert_eq!(error.error_code(), ErrorCode::StorageFull);
                    assert!(
                        error.to_string().contains("spill disk quota exceeded"),
                        "{error}"
                    );
                } else {
                    let result = result.unwrap_or_else(|error| {
                        panic!(
                            "forced-spill success failed: profile={profile}, database_route={database_route}: {error}"
                        )
                    });
                    if profile {
                        assert_eq!(result.columns, ["profile"]);
                        let [row] = result.rows() else {
                            panic!("PROFILE must return one diagnostic row")
                        };
                        let [Value::String(text)] = row.as_slice() else {
                            panic!("PROFILE must return text")
                        };
                        assert!(text.split_whitespace().any(|field| field == "rows=1024"));
                        assert!(text.split_whitespace().any(|field| field == "rows=4096"));
                        // Query-wide counters are repeated for physical nodes;
                        // inspect the root snapshot without summing children.
                        let snapshot = text
                            .lines()
                            .find_map(|line| line.trim().strip_prefix("query-wide "))
                            .expect("PROFILE must expose query resource counters");
                        let counter = |name: &str| {
                            snapshot
                                .split_whitespace()
                                .filter_map(|field| field.split_once('='))
                                .find_map(|(key, value)| (key == name).then_some(value))
                                .unwrap_or_else(|| panic!("missing {name}: {text}"))
                                .parse::<u64>()
                                .unwrap_or_else(|error| panic!("invalid {name}: {error}: {text}"))
                        };
                        assert!(counter("resident_peak_bytes") > 0, "{text}");
                        assert!(counter("spilled_bytes_total") > 0, "{text}");
                        assert!(counter("spill_runs_total") > 0, "{text}");
                        assert_eq!(counter("spill_partitions_total"), 0, "{text}");
                        let physical = text
                            .lines()
                            .find(|line| line.contains("spill_physical_reserved_bytes="))
                            .expect("physical PROFILE history");
                        assert!(
                            physical.contains("spill_physical_reserved_bytes=0 "),
                            "{text}"
                        );
                        assert!(physical.contains("spill_cleanup_debt_bytes=0 "), "{text}");
                        assert!(
                            physical.contains("spill_observed_file_bytes_at_publication=0 "),
                            "{text}"
                        );
                        assert!(physical.contains("spill_cleanup_failed=false"), "{text}");
                        assert!(!physical.contains("spill_physical_peak_bytes=0 "), "{text}");
                        assert!(
                            !physical.contains("spill_observed_file_peak_bytes=0 "),
                            "{text}"
                        );
                        #[cfg(not(target_arch = "wasm32"))]
                        assert!(counter("merge_time_ns") > 0, "{text}");
                        assert!(result.execution_time_ms.is_some());
                    } else {
                        assert_exact_rows(result.rows(), &expected);
                    }
                }
                assert_cleanup();

                // Reuse the actual Session after its failure/profile call, and
                // the same database entry point when that route was exercised.
                let follow_up = "SELECT (COUNT(*) AS ?count) WHERE { VALUES ?x { 1 } }";
                let result = session.execute_sparql(follow_up).unwrap();
                assert_exact_rows(result.rows(), &[vec![Value::Int64(1)]]);
                drop(result);
                assert_cleanup();
                if database_route {
                    let result = database
                        .execute_language(follow_up, "sparql", Some(params()))
                        .unwrap();
                    assert_exact_rows(result.rows(), &[vec![Value::Int64(1)]]);
                    drop(result);
                    assert_cleanup();
                }
            }
        }
    }
}

#[cfg(feature = "spill")]
#[test]
fn rdf_large_values_profile_diagnostic_denial_releases_resources() {
    let (query, expected) = many_groups_fixture();
    let directory = tempfile::tempdir().unwrap();
    let database = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(2 << 20)
            .with_spill_path(directory.path()),
    )
    .unwrap();
    let session = database.session();
    let resident_before = database.buffer_manager().allocated();
    let result = session.execute_sparql(&query).unwrap();
    assert_exact_rows(result.rows(), &expected);
    drop(result);

    // The query fits through spill, but formatting the thousands of physical
    // VALUES/BIND nodes into a complete diagnostic needs its own admission.
    let error = session
        .execute_sparql(&format!("EXPLAIN ANALYZE {query}"))
        .expect_err("oversized PROFILE diagnostic must respect the resident limit");
    assert_eq!(error.error_code(), ErrorCode::StorageFull);
    assert!(
        error.to_string().contains("resident-memory limit exceeded"),
        "{error}"
    );
    drop(error);
    assert_eq!(database.buffer_manager().allocated(), resident_before);
    let namespace = directory
        .path()
        .join(format!("grafeo-store-{}", database.store_id()));
    let entries: Vec<_> = std::fs::read_dir(namespace)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(
        entries
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>(),
        [
            ".grafeo-spill-quota",
            ".grafeo-spill-quota.lock",
            ".grafeo-spill-root"
        ]
        .map(std::ffi::OsString::from)
        .into_iter()
        .collect()
    );

    let result = session
        .execute_sparql("SELECT (COUNT(*) AS ?count) WHERE { VALUES ?x { 1 } }")
        .unwrap();
    assert_exact_rows(result.rows(), &[vec![Value::Int64(1)]]);
    drop(result);
    assert_eq!(database.buffer_manager().allocated(), resident_before);
}

// Linux /proc reports an actual process high-water mark, including allocations
// outside the query grants. Other platforms need their own measurement witness;
// this Linux-only test is not evidence of their process-memory acceptance.
#[cfg(all(feature = "spill", target_os = "linux"))]
#[test]
fn rdf_aggregate_process_peak_linux() {
    use grafeo_core::graph::rdf::{Term, Triple};

    const CHILD_SCALE: &str = "GRAFEO_RDF_PEAK_CHILD_SCALE";
    const CHILD_DENIED: &str = "GRAFEO_RDF_PEAK_CHILD_DENIED";
    const BUDGET: usize = 2 << 20;
    // Initial calibration ceiling: a fixed 16 query budgets above the larger
    // runtime/store high-water baseline, independent of input cardinality.
    // This includes planning, store wrappers, allocator retention, eager output
    // and diagnostics. It is deliberately NOT an operator-grant bound: RSS
    // subtraction cannot attribute ownership or account for allocator reuse.
    // Pair it with the independently enforced grant peak below. Tightening this
    // process headroom needs fresh debug/release calibration, not a larger
    // query budget or a ceiling that grows with the measured input.
    const PROCESS_HEADROOM: usize = 32 << 20;
    const GROUPS: usize = 1024;

    let Some(scale) = std::env::var_os(CHILD_SCALE) else {
        for scale in [1, 2, 4] {
            for denied in [false, true] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "rdf_aggregate_process_peak_linux",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env(CHILD_SCALE, scale.to_string())
                    .env(CHILD_DENIED, if denied { "1" } else { "0" })
                    .output()
                    .unwrap();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(
                    output.status.success(),
                    "resource child scale={scale}, denied={denied}: {}\n{stdout}\n{stderr}",
                    output.status
                );
                assert!(stdout.contains("rdf process peak:"), "{stdout}\n{stderr}");
                print!("{stdout}");
            }
        }
        return;
    };
    let scale: usize = scale.to_str().unwrap().parse().unwrap();
    assert!([1, 2, 4].contains(&scale));
    let denied = match std::env::var(CHILD_DENIED).unwrap().as_str() {
        "0" => false,
        "1" => true,
        other => panic!("invalid child quota mode: {other}"),
    };
    let process_memory = || {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let bytes = |name: &str| {
            let line = status
                .lines()
                .find_map(|line| line.strip_prefix(name))
                .unwrap_or_else(|| panic!("missing Linux process counter {name}"));
            let mut fields = line.split_whitespace();
            let kib: usize = fields.next().unwrap().parse().unwrap();
            assert_eq!(fields.next(), Some("kB"));
            kib.checked_mul(1024).unwrap()
        };
        (bytes("VmRSS:"), bytes("VmHWM:"))
    };
    let runtime_baseline = process_memory();
    let directory = tempfile::tempdir().unwrap();
    let mut config = Config::in_memory()
        .with_graph_model(GraphModel::Rdf)
        .with_memory_limit(BUDGET)
        .with_spill_path(directory.path());
    if denied {
        config = config.with_max_query_spill_bytes(0);
    }
    let database = GrafeoDB::with_config(config).unwrap();
    assert_eq!(
        database
            .batch_insert_rdf((0..GROUPS).map(|group| {
                Triple::new(
                    Term::iri(format!("urn:peak:row:{group}")),
                    Term::iri("urn:peak:group"),
                    Term::literal(format!("g{group:04}")),
                )
            }))
            .unwrap(),
        GROUPS
    );
    let session = database.session();
    let resident_before = database.buffer_manager().allocated();
    let store_baseline = process_memory();
    let ceiling = runtime_baseline
        .1
        .max(store_baseline.1)
        .checked_add(PROCESS_HEADROOM)
        .unwrap();
    let assert_cleanup = || {
        assert_eq!(database.buffer_manager().allocated(), resident_before);
        let namespace = directory
            .path()
            .join(format!("grafeo-store-{}", database.store_id()));
        let entries: Vec<_> = std::fs::read_dir(namespace)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            entries
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            [
                ".grafeo-spill-quota",
                ".grafeo-spill-quota.lock",
                ".grafeo-spill-root"
            ]
            .map(std::ffi::OsString::from)
            .into_iter()
            .collect()
        );
    };

    // N/2N/4N operands, fixed stored graph and fixed output cardinality. The
    // VALUES plan has at most 16 constants, avoiding a large diagnostic tree.
    let values = 4 * scale;
    let mut query = String::from(
        "SELECT ?group (SUM(?value) AS ?sum) (AVG(?value) AS ?avg) (COUNT(?value) AS ?count) WHERE { ?row <urn:peak:group> ?group . VALUES ?value { ",
    );
    for value in 1..=values {
        write!(query, "{value} ").unwrap();
    }
    query.push_str("} } GROUP BY ?group");
    let result = session.execute_sparql(&query);
    let sampled_resources = if denied {
        let error = result.expect_err("every input size must require spill");
        assert_eq!(error.error_code(), ErrorCode::StorageFull);
        assert!(
            error.to_string().contains("spill disk quota exceeded"),
            "{error}"
        );
        drop(error);
        None
    } else {
        let result = result.unwrap();
        let expected: Vec<_> = (0..GROUPS)
            .map(|group| {
                vec![
                    Value::from(format!("g{group:04}")),
                    Value::Int64(i64::try_from(values * (values + 1) / 2).unwrap()),
                    Value::RdfLiteral {
                        lexical: format!("{}.5", values / 2).into(),
                        language: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
                    },
                    Value::Int64(i64::try_from(values).unwrap()),
                ]
            })
            .collect();
        assert_exact_rows(result.rows(), &expected);
        drop(result);
        drop(expected);
        assert_cleanup();

        let result = session
            .execute_sparql(&format!("EXPLAIN ANALYZE {query}"))
            .unwrap();
        let [row] = result.rows() else {
            panic!("expected one PROFILE row")
        };
        let [Value::String(text)] = row.as_slice() else {
            panic!("expected PROFILE text")
        };
        let snapshot = text
            .lines()
            .find_map(|line| line.trim().strip_prefix("query-wide "))
            .expect("query-wide resource snapshot");
        let counter = |name: &str| {
            snapshot
                .split_whitespace()
                .filter_map(|field| field.split_once('='))
                .find_map(|(key, value)| (key == name).then_some(value))
                .unwrap_or_else(|| panic!("missing {name}: {text}"))
                .parse::<usize>()
                .unwrap()
        };
        let counters = (
            counter("resident_peak_bytes"),
            counter("spilled_bytes_total"),
            counter("spill_runs_total"),
        );
        assert!((1..=BUDGET).contains(&counters.0), "{text}");
        assert!(counters.1 > 0, "{text}");
        assert!(counters.2 > 0, "{text}");
        assert!(
            text.split_whitespace()
                .any(|field| field == format!("rows={GROUPS}"))
        );
        assert!(
            text.split_whitespace()
                .any(|field| field == format!("rows={}", GROUPS * values))
        );
        drop(result);
        Some(counters)
    };
    assert_cleanup();
    let follow_up = session
        .execute_sparql("SELECT (COUNT(*) AS ?count) WHERE { VALUES ?x { 1 } }")
        .unwrap();
    assert_exact_rows(follow_up.rows(), &[vec![Value::Int64(1)]]);
    drop(follow_up);
    assert_cleanup();
    let final_memory = process_memory();
    // These are the successful PROFILE query's sampled counters. A denied
    // ordinary query has no diagnostic result; do not fabricate zero samples.
    let (resident_peak, spilled_bytes, spill_runs) = sampled_resources.map_or_else(
        || {
            (
                "unavailable".to_owned(),
                "unavailable".to_owned(),
                "unavailable".to_owned(),
            )
        },
        |(resident, bytes, runs)| (resident.to_string(), bytes.to_string(), runs.to_string()),
    );
    println!(
        "rdf process peak: operands={} denied={denied} runtime_rss={} runtime_hwm={} store_rss={} store_hwm={} final_rss={} peak={} ceiling={ceiling} profile_resident_peak_bytes={resident_peak} profile_spilled_bytes_total={spilled_bytes} profile_spill_runs_total={spill_runs}",
        GROUPS * values,
        runtime_baseline.0,
        runtime_baseline.1,
        store_baseline.0,
        store_baseline.1,
        final_memory.0,
        final_memory.1,
    );
    assert!(final_memory.1 > 0);
    assert!(
        final_memory.1 <= ceiling,
        "process peak {} exceeds fixed ceiling {ceiling}",
        final_memory.1
    );
}

#[cfg(all(feature = "gql", feature = "lpg"))]
#[test]
fn rdf_stream_route_explicitly_rejects_unqualified_sparql_language() {
    use grafeo_engine::query::ExecutionOptions;
    use std::collections::HashMap;

    let database =
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let query = "SELECT (COUNT(*) AS ?count) WHERE { VALUES ?x { 1 } }";
    let options = || ExecutionOptions {
        language: Some("sparql".to_owned()),
        ..ExecutionOptions::default()
    };
    let error = database
        .stream_with_options(query, HashMap::new(), options())
        .expect_err("SPARQL cursor is not qualified");
    assert!(
        error.to_string().contains("this read cursor supports GQL"),
        "{error}"
    );
    let error = database
        .session()
        .stream_with_options(query, HashMap::new(), options())
        .err()
        .expect("Session must reject the same unsupported language");
    assert!(
        error.to_string().contains("this read cursor supports GQL"),
        "{error}"
    );
}
