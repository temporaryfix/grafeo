//! Integration tests for SPARQL expressions in projection and GROUP BY.
//!
//! Covers two areas:
//!   1. **Projection functions**: STR(), STRLEN() used in SELECT (not just FILTER).
//!   2. **GROUP BY with expressions**: STR() inside GROUP BY combined with COUNT.
// Test values are small known constants
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
//!
//! These tests verify the full pipeline: SPARQL translator projection handling,
//! RDF planner expression pre-projection, and physical execution.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_aggregate_expressions
//! ```

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod sparql_aggregate_expression_tests {
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::{Error, QueryErrorKind};
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    fn insert_sample_triples(db: &GrafeoDB) {
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/name> "Alix" .
                <http://ex.org/alix> <http://ex.org/age>  "30" .
                <http://ex.org/gus>  <http://ex.org/name> "Gus" .
                <http://ex.org/gus>  <http://ex.org/age>  "25" .
            }"#,
        )
        .unwrap();
    }

    fn assert_semantic_error(db: &GrafeoDB, query: &str, expected: &str) {
        let error = match db.execute_sparql(query) {
            Ok(result) => panic!("query unexpectedly executed `{query}`: {result:?}"),
            Err(error) => error,
        };
        assert!(
            matches!(
                &error,
                Error::Query(query_error) if query_error.kind == QueryErrorKind::Semantic
            ),
            "expected a semantic query error for `{query}`: {error:?}",
        );
        assert!(
            error.to_string().contains(expected),
            "expected `{expected}` in error for `{query}`, got: {error}",
        );
    }

    fn assert_permanent_aggregate_placement_error(
        db: &GrafeoDB,
        query: &str,
        expected_clause: &str,
    ) {
        let error = match db.execute_sparql(query) {
            Ok(result) => panic!("query unexpectedly executed `{query}`: {result:?}"),
            Err(error) => error,
        };
        assert!(
            matches!(
                &error,
                Error::Query(query_error) if query_error.kind == QueryErrorKind::Semantic
            ),
            "expected a semantic query error for `{query}`: {error:?}",
        );
        let message = error.to_string();
        assert!(
            message.contains(expected_clause),
            "expected clause-specific `{expected_clause}` error for `{query}`, got: {message}",
        );
        assert!(
            !message.contains("aggregate hoisting"),
            "permanent illegal-placement errors must not mention the temporary hoisting gate: {message}",
        );
    }

    /// A projected alias for the exact GROUP BY expression is a real output,
    /// not an implementation-specific aggregate key column.
    #[test]
    fn sparql_group_by_str_with_count() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result = db
            .execute_sparql(
                "SELECT (STR(?s) AS ?subject) (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY (STR(?s)) ORDER BY ?subject",
            )
            .expect("a projected GROUP BY expression alias remains available after aggregation");

        assert_eq!(result.columns, ["subject", "cnt"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("http://ex.org/alix".into()), Value::Int64(2),],
                vec![Value::String("http://ex.org/gus".into()), Value::Int64(2),],
            ]
        );
    }

    /// ORDER BY ASC(STR(?s)): expression-based sorting should work.
    #[test]
    fn sparql_order_by_str() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let qr = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/name> ?o } ORDER BY ASC(STR(?s))")
            .unwrap();
        assert_eq!(
            qr.row_count(),
            2,
            "ORDER BY ASC(STR(?s)) should return 2 rows"
        );
    }

    /// Both GROUP BY and ORDER BY consume the projected expression alias.
    #[test]
    fn sparql_group_by_and_order_by_both_complex() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result = db
            .execute_sparql(
                "SELECT (STR(?s) AS ?subject) (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY (STR(?s)) ORDER BY DESC(?subject)",
            )
            .expect("ORDER BY resolves the materialized grouped SELECT alias");

        assert_eq!(result.columns, ["subject", "cnt"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("http://ex.org/gus".into()), Value::Int64(2),],
                vec![Value::String("http://ex.org/alix".into()), Value::Int64(2),],
            ]
        );
    }

    #[test]
    fn aggregate_select_materializes_group_derived_aliases_before_order() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (?group AS ?renamed)
                          ((?group + 10) AS ?computed)
                          (COUNT(*) AS ?count)
                   WHERE { VALUES ?group { 2 1 } }
                   GROUP BY ?group
                   ORDER BY ?computed"#,
            )
            .expect("group-derived SELECT aliases are extended before ORDER BY");

        assert_eq!(result.columns, ["renamed", "computed", "count"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::Int64(1), Value::Int64(11), Value::Int64(1)],
                vec![Value::Int64(2), Value::Int64(12), Value::Int64(1)],
            ]
        );
    }

    #[test]
    fn aggregate_order_by_unaliased_group_expression_uses_group_key() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?count)
                   WHERE { VALUES ?value { "b" "a" "a" } }
                   GROUP BY (STR(?value))
                   ORDER BY (STR(?value))
                   LIMIT 1"#,
            )
            .expect("ORDER BY can reuse an unaliased GROUP BY expression after aggregation");

        assert_eq!(result.columns, ["count"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(2)]]);
    }

    #[test]
    fn aggregate_having_unaliased_group_expression_uses_group_key() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?count)
                   WHERE { VALUES ?value { "b" "a" "a" } }
                   GROUP BY (STR(?value))
                   HAVING (STR(?value) = "a")"#,
            )
            .expect("HAVING can reuse an unaliased GROUP BY expression after aggregation");

        assert_eq!(result.columns, ["count"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(2)]]);
    }

    #[test]
    fn aggregate_group_keys_preserve_rdf_term_identity() {
        let db = rdf_db();
        for group_expression in ["?value", "IF(true, ?value, ?value)"] {
            let query = format!(
                r#"SELECT ?group (COUNT(*) AS ?count)
                   WHERE {{ VALUES ?value {{ <urn:same> "urn:same" }} }}
                   GROUP BY ({group_expression} AS ?group)"#,
            );
            let result = db
                .execute_sparql(&query)
                .expect("group keys distinguish RDF term identity, not host values");

            assert_eq!(result.columns, ["group", "count"]);
            assert_eq!(result.row_count(), 2, "query: {query}");
            assert!(result.rows().iter().all(|row| row[1] == Value::Int64(1)));
        }
    }

    #[test]
    fn parenthesized_variable_grouping_preserves_the_public_variable() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?value (COUNT(*) AS ?count)
                   WHERE { VALUES ?value { 2 1 1 } }
                   GROUP BY (?value)
                   ORDER BY ?value"#,
            )
            .expect("GROUP BY (?value) is equivalent to GROUP BY ?value");

        assert_eq!(result.columns, ["value", "count"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::Int64(1), Value::Int64(2)],
                vec![Value::Int64(2), Value::Int64(1)],
            ]
        );
    }

    #[test]
    fn grouped_wildcard_projects_only_unique_named_group_outputs() {
        let db = rdf_db();

        let anonymous = db
            .execute_sparql(
                r#"SELECT DISTINCT * WHERE {
                       {
                           SELECT * WHERE { VALUES ?x { "a" "b" } }
                           GROUP BY (STR(?x))
                       }
                   }"#,
            )
            .expect("anonymous group keys do not escape a wildcard subselect");
        assert!(anonymous.columns.is_empty());
        assert_eq!(anonymous.row_count(), 1, "DISTINCT sees one empty mapping");

        let repeated = db
            .execute_sparql("SELECT * WHERE { VALUES ?x { 1 2 } } GROUP BY ?x ?x")
            .expect("repeated named group keys have one public wildcard output");
        assert_eq!(repeated.columns, ["x"]);
        assert_eq!(repeated.row_count(), 2);
    }

    #[test]
    fn empty_explicit_group_does_not_synthesize_a_join_mapping() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?x WHERE {
                       { SELECT ?x WHERE { ?s <urn:missing> ?x } GROUP BY ?x }
                       VALUES ?x { <urn:a> }
                   }"#,
            )
            .expect("an empty grouped subquery contributes no join mappings");

        assert!(result.rows().is_empty());
    }

    #[test]
    fn direct_variable_grouping_preserves_native_vector_values() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                "SELECT ?v (COUNT(*) AS ?count) WHERE { BIND(VECTOR(1, 2) AS ?v) } GROUP BY ?v",
            )
            .expect("native extension values remain legal direct grouping keys");

        assert_eq!(result.columns, ["v", "count"]);
        assert_eq!(
            result.rows(),
            &[vec![Value::Vector(vec![1.0, 2.0].into()), Value::Int64(1)]]
        );
    }

    #[test]
    fn expression_grouping_preserves_rdf_or_native_bind_identity() {
        let db = rdf_db();
        let cases = [
            (
                "aliased variable expression",
                r#"SELECT ?g (COUNT(*) AS ?count)
                   WHERE { BIND(VECTOR(1) AS ?v) }
                   GROUP BY (?v AS ?g)"#,
                r#"SELECT ?g (COUNT(*) AS ?count)
                   WHERE { VALUES ?v { <urn:x> "urn:x" } }
                   GROUP BY (?v AS ?g)"#,
            ),
            (
                "aliased computed expression",
                r#"SELECT ?g (COUNT(*) AS ?count)
                   WHERE { BIND(VECTOR(1) AS ?v) }
                   GROUP BY (COALESCE(?v, VECTOR(0)) AS ?g)"#,
                r#"SELECT ?g (COUNT(*) AS ?count)
                   WHERE { VALUES ?v { <urn:x> "urn:x" } }
                   GROUP BY (COALESCE(?v, <urn:missing>) AS ?g)"#,
            ),
            (
                "unparenthesized built-in expression",
                r#"SELECT (COALESCE(?v, VECTOR(0)) AS ?g) (COUNT(*) AS ?count)
                   WHERE { BIND(VECTOR(1) AS ?v) }
                   GROUP BY COALESCE(?v, VECTOR(0))"#,
                r#"SELECT (COALESCE(?v, <urn:missing>) AS ?g) (COUNT(*) AS ?count)
                   WHERE { VALUES ?v { <urn:x> "urn:x" } }
                   GROUP BY COALESCE(?v, <urn:missing>)"#,
            ),
        ];
        let mut failures = Vec::new();

        for (label, native_query, rdf_query) in cases {
            match db.execute_sparql(native_query) {
                Ok(result) => assert_eq!(
                    result.rows(),
                    &[vec![Value::Vector(vec![1.0].into()), Value::Int64(1)]],
                    "native case: {label}"
                ),
                Err(error) => failures.push(format!("native {label}: {error}")),
            }

            match db.execute_sparql(rdf_query) {
                Ok(result) => {
                    assert_eq!(result.columns, ["g", "count"], "RDF case: {label}");
                    assert_eq!(result.row_count(), 2, "RDF case: {label}");
                    assert!(
                        result.rows().iter().all(|row| {
                            row == &vec![Value::String("urn:x".into()), Value::Int64(1)]
                        }),
                        "same-spelled IRI and literal must remain separate: {label}: {:?}",
                        result.rows()
                    );
                }
                Err(error) => failures.push(format!("RDF {label}: {error}")),
            }
        }

        assert!(
            failures.is_empty(),
            "GROUP expression identity failures:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn native_group_identity_survives_regroup_and_compatibility_join() {
        let db = rdf_db();
        let regrouped = db
            .execute_sparql(
                r#"SELECT ?g (COUNT(*) AS ?count) WHERE {
                       {
                           SELECT ?g WHERE { VALUES ?k { 1 2 } }
                           GROUP BY (VECTOR(?k) AS ?g)
                       }
                   }
                   GROUP BY ?g"#,
            )
            .expect("native group identities compose across an aggregate boundary");
        assert_eq!(regrouped.row_count(), 2);
        assert!(
            regrouped
                .rows()
                .iter()
                .all(|row| row.get(1) == Some(&Value::Int64(1)))
        );

        let joined = db
            .execute_sparql(
                r#"SELECT ?g WHERE {
                       {
                           SELECT ?g WHERE { VALUES ?left { 1 2 } }
                           GROUP BY (VECTOR(?left) AS ?g)
                       }
                       {
                           SELECT ?g WHERE { VALUES ?right { 2 3 } }
                           GROUP BY (VECTOR(?right) AS ?g)
                       }
                   }"#,
            )
            .expect("native grouped aliases retain compatibility identity");
        assert_eq!(joined.rows(), &[vec![Value::Vector(vec![2.0].into())]]);
    }

    /// Catches treating a UNION-visible group-helper column as proof that
    /// every branch row has a usable DISTINCT key.
    #[test]
    fn distinct_normalizes_sparse_group_helpers_rowwise_after_union() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT DISTINCT ?value WHERE {
                       {
                           SELECT ?value WHERE { VALUES ?value { <urn:grouped> } }
                           GROUP BY ?value
                       }
                       UNION
                       { VALUES ?value { <urn:ordinary-a> <urn:ordinary-b> } }
                   }"#,
            )
            .expect("DISTINCT rebuilds missing row-level group provenance after UNION padding");

        assert_eq!(result.columns, ["value"]);
        assert_eq!(result.row_count(), 3);
        for value in ["urn:grouped", "urn:ordinary-a", "urn:ordinary-b"] {
            assert!(
                result.rows().contains(&vec![Value::String(value.into())]),
                "missing distinct RDF term {value}: {:?}",
                result.rows()
            );
        }
    }

    /// Catches treating a UNION-visible group-helper column as proof that
    /// every branch row has a usable GROUP BY key.
    #[test]
    fn grouping_normalizes_sparse_group_helpers_rowwise_after_union() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?value (COUNT(*) AS ?count) WHERE {
                       {
                           SELECT ?value WHERE { VALUES ?value { <urn:grouped> } }
                           GROUP BY ?value
                       }
                       UNION
                       { VALUES ?value { <urn:ordinary-a> <urn:ordinary-b> } }
                   }
                   GROUP BY ?value"#,
            )
            .expect("GROUP BY rebuilds missing row-level group provenance after UNION padding");

        assert_eq!(result.columns, ["value", "count"]);
        assert_eq!(result.row_count(), 3);
        assert!(
            result
                .rows()
                .iter()
                .all(|row| row.get(1) == Some(&Value::Int64(1))),
            "ordinary UNION rows must not collapse under a shared NULL helper: {:?}",
            result.rows()
        );
    }

    /// Catches dropping the only native identity helper when a compatibility
    /// join coalesces a grouped binding through an unbound peer.
    #[test]
    fn coalesced_native_group_identity_survives_a_later_compatibility_join() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?g WHERE {
                       {
                           SELECT ?g WHERE { VALUES ?left { 1 } }
                           GROUP BY (VECTOR(?left) AS ?g)
                       }
                       VALUES ?g { UNDEF }
                       {
                           SELECT ?g WHERE { VALUES ?right { 1 } }
                           GROUP BY (VECTOR(?right) AS ?g)
                       }
                   }"#,
            )
            .expect("coalescing through UNDEF preserves native compatibility provenance");

        assert_eq!(result.columns, ["g"]);
        assert_eq!(result.rows(), &[vec![Value::Vector(vec![1.0].into())]]);
    }

    /// Catches evaluating a computed SELECT DISTINCT alias without carrying
    /// its canonical RDF-or-native identity across the final projection.
    #[test]
    fn computed_distinct_if_preserves_rdf_term_identity() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT DISTINCT (IF(?branch = 0, <urn:x>, "urn:x") AS ?term)
                   WHERE { VALUES ?branch { 0 1 } }"#,
            )
            .expect("computed DISTINCT aliases retain RDF term identity");

        assert_eq!(result.columns, ["term"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("urn:x".into())],
                vec![Value::String("urn:x".into())],
            ]
        );
    }

    /// Catches falling back to the generic formatted DISTINCT key for a
    /// complex native value that prints exactly like an ordinary string.
    #[test]
    fn computed_distinct_keeps_native_vector_separate_from_lookalike_string() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?v WHERE {
                       {
                           SELECT DISTINCT
                                  (IF(?left = 0, "Vector([1.0])", VECTOR(1)) AS ?v)
                           WHERE { VALUES ?left { 0 1 } }
                       }
                       {
                           SELECT DISTINCT
                                  (IF(?right = 0, "Vector([1.0])", VECTOR(1)) AS ?v)
                           WHERE { VALUES ?right { 0 1 } }
                       }
                   }"#,
            )
            .expect("DISTINCT outputs retain discriminated compatibility identity");

        assert_eq!(result.columns, ["v"]);
        assert_eq!(result.row_count(), 2);
        assert!(
            result
                .rows()
                .contains(&vec![Value::String("Vector([1.0])".into())])
        );
        assert!(
            result
                .rows()
                .contains(&vec![Value::Vector(vec![1.0].into())])
        );
    }

    /// Catches excluding outer DISTINCT when deciding whether a direct SAMPLE
    /// alias must retain RDF term identity across the aggregate boundary.
    #[test]
    fn distinct_sample_alias_preserves_rdf_term_identity() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT DISTINCT (SAMPLE(?value) AS ?sample) WHERE {
                       VALUES (?group ?value) { (1 <urn:x>) (2 "urn:x") }
                   }
                   GROUP BY ?group"#,
            )
            .expect("SAMPLE output identity remains available to outer DISTINCT");

        assert_eq!(result.columns, ["sample"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("urn:x".into())],
                vec![Value::String("urn:x".into())],
            ]
        );
    }

    /// Controls against replacing SAMPLE preparation with RDF-only tagging,
    /// which would reject a legal native extension result.
    #[test]
    fn distinct_sample_alias_preserves_native_vector_values() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT DISTINCT (SAMPLE(VECTOR(?group)) AS ?sample)
                   WHERE { VALUES ?group { 1 2 } }
                   GROUP BY ?group"#,
            )
            .expect("native SAMPLE results remain legal under DISTINCT");

        assert_eq!(result.columns, ["sample"]);
        assert_eq!(result.row_count(), 2);
        assert!(
            result
                .rows()
                .contains(&vec![Value::Vector(vec![1.0].into())])
        );
        assert!(
            result
                .rows()
                .contains(&vec![Value::Vector(vec![2.0].into())])
        );
    }

    #[test]
    fn redundant_parentheses_do_not_change_group_expression_identity() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (STR((?x)) AS ?key) (COUNT(*) AS ?count)
                   WHERE { VALUES ?x { "a" } }
                   GROUP BY (STR(?x))"#,
            )
            .expect("transparent parentheses preserve grouped-expression reuse");

        assert_eq!(result.columns, ["key", "count"]);
        assert_eq!(
            result.rows(),
            &[vec![Value::String("a".into()), Value::Int64(1)]]
        );
    }

    #[test]
    fn grouped_projection_aliases_are_sequentially_available() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (?group AS ?first)
                          ((?first + 10) AS ?second)
                          (COUNT(*) AS ?count)
                          ((?count + 1) AS ?next)
                   WHERE { VALUES ?group { 2 1 1 } }
                   GROUP BY ?group
                   ORDER BY ?second"#,
            )
            .expect("later grouped SELECT expressions can consume earlier aliases");

        assert_eq!(result.columns, ["first", "second", "count", "next"]);
        assert_eq!(
            result.rows(),
            &[
                vec![
                    Value::Int64(1),
                    Value::Int64(11),
                    Value::Int64(2),
                    Value::Int64(3),
                ],
                vec![
                    Value::Int64(2),
                    Value::Int64(12),
                    Value::Int64(1),
                    Value::Int64(2),
                ],
            ]
        );
    }

    #[test]
    fn grouped_projection_aliases_reject_backward_references() {
        let db = rdf_db();
        assert_semantic_error(
            &db,
            r#"SELECT ((?sum + 1) AS ?before) (SUM(?value) AS ?sum)
               WHERE { VALUES ?value { 1 2 } }"#,
            "not grouped or aggregated",
        );
    }

    #[test]
    fn select_and_group_assignment_targets_must_be_fresh() {
        let db = rdf_db();
        for query in [
            "SELECT (COUNT(*) AS ?value) WHERE { VALUES ?value { 1 } }",
            "SELECT (1 AS ?duplicate) (2 AS ?duplicate) WHERE {}",
            "SELECT ?existing (COUNT(*) AS ?count) WHERE { VALUES ?existing { 1 } } GROUP BY (?existing AS ?existing)",
        ] {
            assert_semantic_error(&db, query, "already in scope");
        }
    }

    #[test]
    fn aggregate_modifiers_reject_ungrouped_variables() {
        let db = rdf_db();
        for query in [
            r#"SELECT ?group (COUNT(*) AS ?count)
               WHERE { VALUES (?group ?other) { (1 "a") (1 "b") } }
               GROUP BY ?group
               HAVING (?other = "a")"#,
            r#"SELECT ?group (COUNT(*) AS ?count)
               WHERE { VALUES (?group ?other) { (1 "a") (1 "b") } }
               GROUP BY ?group
               ORDER BY ?other"#,
        ] {
            assert_semantic_error(&db, query, "not grouped or aggregated");
        }
    }

    #[test]
    fn aggregate_select_rejects_ungrouped_projection_variables() {
        let db = rdf_db();
        let error = db
            .execute_sparql("SELECT ?value (COUNT(*) AS ?count) WHERE { VALUES ?value { 1 2 } }")
            .expect_err("a nonaggregate variable cannot escape a global aggregate");

        assert!(
            matches!(
                &error,
                Error::Query(query_error) if query_error.kind == QueryErrorKind::Semantic
            ),
            "ungrouped aggregate projection must be a semantic query error: {error:?}",
        );
        assert!(
            error.to_string().contains("not grouped"),
            "unexpected ungrouped projection error: {error}",
        );
    }

    #[test]
    fn global_aggregate_allows_constant_projection_expressions() {
        let db = rdf_db();
        let result = db
            .execute_sparql("SELECT (1 AS ?one) (COUNT(*) AS ?count) WHERE {}")
            .expect("constant expressions are legal beside a global aggregate");

        assert_eq!(result.columns, ["one", "count"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(1), Value::Int64(1)]]);
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_nested_operands() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            "SELECT (COUNT(SUM(?value)) AS ?count) WHERE { VALUES ?value { 1 2 } }",
            "nested aggregate",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_nested_having_operands() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT (COUNT(*) AS ?count)
               WHERE { VALUES ?value { 1 2 } }
               HAVING (COUNT(SUM(?value)) > 0)"#,
            "nested aggregate",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_nested_order_operands() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT (COUNT(*) AS ?count)
               WHERE { VALUES ?value { 1 2 } }
               ORDER BY (COUNT(SUM(?value)))"#,
            "nested aggregate",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_filter() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            "SELECT (COUNT(*) AS ?count) WHERE { VALUES ?value { 1 2 } FILTER(SUM(?value) > 0) }",
            "FILTER",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_bind() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            "SELECT (COUNT(*) AS ?count) WHERE { VALUES ?value { 1 2 } BIND(SUM(?value) AS ?sum) }",
            "BIND",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_exists_filter() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT ?subject
               WHERE {
                 VALUES ?subject { 1 }
                 FILTER(EXISTS {
                   VALUES ?value { 1 2 }
                   FILTER(SUM(?value) > 0)
                 })
               }"#,
            "FILTER",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_select_exists_bind() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT (EXISTS {
                 VALUES ?value { 1 2 }
                 BIND(SUM(?value) AS ?sum)
               } AS ?seen)
               WHERE {}"#,
            "BIND",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_having_not_exists_filter() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT (COUNT(*) AS ?count)
               WHERE {}
               HAVING (NOT EXISTS {
                 VALUES ?value { 1 2 }
                 FILTER(SUM(?value) > 0)
               })"#,
            "FILTER",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_order_not_exists_bind() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT (COUNT(*) AS ?count)
               WHERE {}
               ORDER BY (NOT EXISTS {
                 VALUES ?value { 1 2 }
                 BIND(SUM(?value) AS ?sum)
               })"#,
            "BIND",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_group_by() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            "SELECT (COUNT(*) AS ?count) WHERE { VALUES ?value { 1 2 } } GROUP BY (SUM(?value))",
            "GROUP BY",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_group_by_exists_bind() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT ?group
               WHERE { VALUES ?group { 1 } }
               GROUP BY ?group (EXISTS {
                 VALUES ?value { 1 2 }
                 BIND(SUM(?value) AS ?sum)
               })"#,
            "BIND",
        );
    }

    #[test]
    fn aggregate_placement_remains_illegal_in_group_by_not_exists_filter() {
        let db = rdf_db();
        assert_permanent_aggregate_placement_error(
            &db,
            r#"SELECT ?group
               WHERE { VALUES ?group { 1 } }
               GROUP BY ?group (NOT EXISTS {
                 VALUES ?value { 1 2 }
                 FILTER(SUM(?value) > 0)
               })"#,
            "FILTER",
        );
    }

    #[test]
    fn aggregate_hoist_scalar_projection_expressions_execute() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT (((COUNT(*)) + 1) AS ?n)
                          (sameTerm(SUM(?value) + 1, "4"^^xsd:integer) AS ?sum_ok)
                          (sameTerm(AVG(?value) + 1, "2.5"^^xsd:decimal) AS ?average_ok)
                          (IF(COUNT(*) = 2, true, false) AS ?if_ok)
                          (sameTerm(COALESCE(AVG(?value), 0), "1.5"^^xsd:decimal) AS ?coalesce_ok)
                   WHERE { VALUES ?value { 1 2 } }"#,
            )
            .expect("scalar expressions can consume aggregates from an implicit global group");

        assert_eq!(
            result.columns,
            ["n", "sum_ok", "average_ok", "if_ok", "coalesce_ok"]
        );
        assert_eq!(
            result.rows(),
            &[vec![
                Value::Int64(3),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
            ]]
        );
    }

    #[test]
    fn aggregate_hoist_having_only_creates_a_sealed_global_group() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (1 AS ?one)
                   WHERE { VALUES ?value { 1 2 } }
                   HAVING (SUM(?value) > 2)"#,
            )
            .expect("a HAVING-only aggregate creates an implicit global group");

        assert_eq!(result.columns, ["one"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(1)]]);
    }

    #[test]
    fn aggregate_hoist_having_only_still_rejects_ungrouped_projection() {
        let db = rdf_db();
        assert_semantic_error(
            &db,
            r#"SELECT ?value
               WHERE { VALUES ?value { 1 2 } }
               HAVING (SUM(?value) > 0)"#,
            "not grouped",
        );
    }

    #[test]
    fn aggregate_hoist_order_only_creates_a_sealed_global_group() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (1 AS ?one)
                   WHERE { VALUES ?value { 1 2 } }
                   ORDER BY (SUM(?value))"#,
            )
            .expect("an ORDER-only aggregate creates an implicit global group");

        assert_eq!(result.columns, ["one"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(1)]]);
    }

    #[test]
    fn aggregate_hoist_grouped_having_and_order_use_distinct_hidden_aggregates() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?group
                   WHERE { VALUES (?group ?value) { ("a" 1) ("a" 3) ("b" 5) } }
                   GROUP BY ?group
                   HAVING (COUNT(*) > 0)
                   ORDER BY DESC(SUM(?value))"#,
            )
            .expect("HAVING and ORDER BY can consume distinct modifier-only aggregates");

        assert_eq!(result.columns, ["group"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("b".into())],
                vec![Value::String("a".into())],
            ]
        );
    }

    #[test]
    fn aggregate_hoist_reuses_one_aggregate_across_select_having_and_order() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?group (SUM(?value) AS ?total)
                   WHERE { VALUES (?group ?value) { ("a" 1) ("a" 3) ("b" 5) } }
                   GROUP BY ?group
                   HAVING (SUM(?value) >= 4)
                   ORDER BY DESC(SUM(?value))"#,
            )
            .expect("SELECT, HAVING, and ORDER BY reuse one aggregate result");

        assert_eq!(result.columns, ["group", "total"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("b".into()), Value::Int64(5)],
                vec![Value::String("a".into()), Value::Int64(4)],
            ]
        );
    }

    #[test]
    fn aggregate_hoist_duplicate_direct_aliases_have_identical_values() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (SUM(?value) AS ?first) (SUM(?value) AS ?second)
                   WHERE { VALUES ?value { 1 2 } }"#,
            )
            .expect("duplicate direct aliases reuse one aggregate value");

        assert_eq!(result.columns, ["first", "second"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(3), Value::Int64(3)]]);
    }

    #[test]
    fn aggregate_hoist_having_rejects_select_aggregate_alias() {
        let db = rdf_db();
        assert_semantic_error(
            &db,
            r#"SELECT (SUM(?value) AS ?first) (SUM(?value) AS ?second)
               WHERE { VALUES ?value { 1 2 } }
               HAVING (?second > 0)"#,
            "not grouped or aggregated",
        );
    }

    #[test]
    fn aggregate_hoist_having_rejects_select_aggregate_alias_in_exact_predicate() {
        let db = rdf_db();
        assert_semantic_error(
            &db,
            r#"SELECT (SAMPLE(?value) AS ?canonical)
                      (SAMPLE(?value) AS ?copied)
               WHERE { VALUES ?value { <urn:x> } }
               HAVING (sameTerm(?copied, <urn:x>))"#,
            "not grouped or aggregated",
        );
    }

    #[test]
    fn aggregate_hoist_duplicate_alias_preserves_exact_rdf_identity() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?same
                   WHERE {
                     {
                       SELECT (SAMPLE(?value) AS ?canonical)
                              (SAMPLE(?value) AS ?copied)
                       WHERE { VALUES ?value { <urn:x> } }
                     }
                     BIND(sameTerm(?copied, <urn:x>) AS ?same)
                   }"#,
            )
            .expect("a duplicate aggregate alias copies canonical RDF identity");

        assert_eq!(result.columns, ["same"]);
        assert_eq!(result.rows(), &[vec![Value::Bool(true)]]);
    }

    #[test]
    fn aggregate_hoist_wildcard_and_zero_column_boundaries_are_sealed() {
        let db = rdf_db();
        let wildcard_subselect = db
            .execute_sparql(
                r#"SELECT *
                   WHERE {
                     {
                       SELECT *
                       WHERE { VALUES ?value { 1 2 } }
                       HAVING (COUNT(*) > 0)
                     }
                     BIND(1 AS ?public)
                   }"#,
            )
            .expect("a wildcard subselect seals its hidden aggregate columns");

        assert_eq!(wildcard_subselect.columns, ["public"]);
        assert_eq!(wildcard_subselect.rows(), &[vec![Value::Int64(1)]]);

        let zero_column = db
            .execute_sparql(
                r#"SELECT *
                   WHERE { VALUES ?value { 1 2 } }
                   HAVING (COUNT(*) > 0)"#,
            )
            .expect("a zero-column grouped projection seals all helpers");

        assert!(zero_column.columns.is_empty());
        assert_eq!(zero_column.rows(), &[Vec::<Value>::new()]);
    }

    #[test]
    fn aggregate_hoist_implicit_group_wildcard_subselect_exports_no_where_names() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (7 AS ?value)
                   WHERE {
                     {
                       SELECT *
                       WHERE { VALUES ?value { 1 2 } }
                       HAVING (COUNT(*) > 0)
                     }
                   }"#,
            )
            .expect("an outer alias can reuse a WHERE name erased by implicit grouping");

        assert_eq!(result.columns, ["value"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(7)]]);
    }

    #[test]
    fn aggregate_hoist_nested_and_modifier_consumers_preserve_exact_term_identity() {
        let db = rdf_db();
        let typed = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT (sameTerm(SAMPLE(?value), "7"^^xsd:unsignedByte) AS ?same_type)
                          (sameTerm(SAMPLE(?value), "7"^^xsd:integer) AS ?wrong_type)
                   WHERE { VALUES ?value { "7"^^xsd:unsignedByte } }
                   HAVING (sameTerm(SAMPLE(?value), "7"^^xsd:unsignedByte))
                   ORDER BY (SAMPLE(?value))"#,
            )
            .expect("nested and modifier consumers retain a typed numeric SAMPLE term");

        assert_eq!(typed.columns, ["same_type", "wrong_type"]);
        assert_eq!(
            typed.rows(),
            &[vec![Value::Bool(true), Value::Bool(false)]],
            "numeric value equality must not erase the sampled RDF datatype"
        );

        let language_and_blank = db
            .execute_sparql(
                r#"SELECT DISTINCT ?kind
                          (sameTerm(SAMPLE(?value), ?expected) AS ?same)
                   WHERE {
                     {
                       VALUES (?kind ?value ?expected) {
                         ("language" "colour"@EN "colour"@en)
                       }
                     }
                     UNION {
                       BIND("blank" AS ?kind)
                       BIND(BNODE("qualification") AS ?value)
                       BIND(?value AS ?expected)
                     }
                   }
                   GROUP BY ?kind ?expected
                   HAVING (sameTerm(SAMPLE(?value), ?expected))
                   ORDER BY (SAMPLE(?value))"#,
            )
            .expect("language and blank-node identity survives grouped aggregate reuse");

        assert_eq!(language_and_blank.columns, ["kind", "same"]);
        assert_eq!(language_and_blank.row_count(), 2);
        assert!(
            language_and_blank
                .rows()
                .contains(&vec![Value::String("language".into()), Value::Bool(true),])
        );
        assert!(
            language_and_blank
                .rows()
                .contains(&vec![Value::String("blank".into()), Value::Bool(true),])
        );

        let distinct_collision = db
            .execute_sparql(
                r#"SELECT DISTINCT
                          (COALESCE(SAMPLE(?value), <urn:missing>) AS ?term)
                   WHERE {
                     VALUES (?group ?value) {
                       (1 <urn:x>)
                       (2 "urn:x")
                     }
                   }
                   GROUP BY ?group
                   HAVING (sameTerm(SAMPLE(?value), SAMPLE(?value)))
                   ORDER BY (SAMPLE(?value))"#,
            )
            .expect("DISTINCT sees exact identity from a nested, modifier-reused SAMPLE");

        assert_eq!(distinct_collision.columns, ["term"]);
        assert_eq!(
            distinct_collision.rows(),
            &[
                vec![Value::String("urn:x".into())],
                vec![Value::String("urn:x".into())],
            ],
            "same-spelled IRI and literal results remain distinct after scalar hoisting"
        );
    }

    #[test]
    fn aggregate_hoist_order_by_reused_sample_preserves_rdf_term_categories() {
        let db = rdf_db();
        let aggregate_order = db
            .execute_sparql(
                r#"SELECT ?kind (SAMPLE(?term) AS ?sample)
                   WHERE {
                     VALUES (?kind ?term) {
                       ("literal" "urn:x")
                       ("iri" <urn:x>)
                     }
                   }
                   GROUP BY ?kind
                   ORDER BY (SAMPLE(?term))"#,
            )
            .expect("ORDER BY consumes the exact RDF category of a reused SAMPLE");

        let alias_order = db
            .execute_sparql(
                r#"SELECT ?kind (SAMPLE(?term) AS ?sample)
                   WHERE {
                     VALUES (?kind ?term) {
                       ("literal" "urn:x")
                       ("iri" <urn:x>)
                     }
                   }
                   GROUP BY ?kind
                   ORDER BY ?sample"#,
            )
            .expect("ORDER BY a direct aggregate alias retains exact RDF category");

        let duplicate_alias_order = db
            .execute_sparql(
                r#"SELECT ?kind
                          (SAMPLE(?term) AS ?first)
                          (SAMPLE(?term) AS ?second)
                   WHERE {
                     VALUES (?kind ?term) {
                       ("literal" "urn:x")
                       ("iri" <urn:x>)
                     }
                   }
                   GROUP BY ?kind
                   ORDER BY ?second"#,
            )
            .expect("ORDER BY a duplicate aggregate alias retains canonical RDF category");

        let expected = [
            vec![Value::String("iri".into()), Value::String("urn:x".into())],
            vec![
                Value::String("literal".into()),
                Value::String("urn:x".into()),
            ],
        ];
        assert_eq!(aggregate_order.columns, ["kind", "sample"]);
        assert_eq!(
            aggregate_order.rows(),
            expected.as_slice(),
            "exact RDF ORDER BY ranks the later IRI input before the same-spelled literal"
        );
        assert_eq!(alias_order.columns, ["kind", "sample"]);
        assert_eq!(alias_order.rows(), expected.as_slice());
        assert_eq!(duplicate_alias_order.columns, ["kind", "first", "second"]);
        assert_eq!(
            duplicate_alias_order.rows(),
            &[
                vec![
                    Value::String("iri".into()),
                    Value::String("urn:x".into()),
                    Value::String("urn:x".into()),
                ],
                vec![
                    Value::String("literal".into()),
                    Value::String("urn:x".into()),
                    Value::String("urn:x".into()),
                ],
            ]
        );
    }

    #[test]
    fn aggregate_hoist_order_by_computed_alias_preserves_rdf_term_categories() {
        let db = rdf_db();
        let aggregate_order = db
            .execute_sparql(
                r#"SELECT ?kind
                          (COALESCE(SAMPLE(?term), <urn:missing>) AS ?sample)
                   WHERE {
                     VALUES (?kind ?term) {
                       ("literal" "urn:x")
                       ("iri" <urn:x>)
                     }
                   }
                   GROUP BY ?kind
                   ORDER BY (COALESCE(SAMPLE(?term), <urn:missing>))"#,
            )
            .expect("ORDER BY a computed aggregate expression retains exact RDF category");
        let alias_order = db
            .execute_sparql(
                r#"SELECT ?kind
                          (COALESCE(SAMPLE(?term), <urn:missing>) AS ?sample)
                   WHERE {
                     VALUES (?kind ?term) {
                       ("literal" "urn:x")
                       ("iri" <urn:x>)
                     }
                   }
                   GROUP BY ?kind
                   ORDER BY ?sample"#,
            )
            .expect("ORDER BY a computed aggregate alias retains exact RDF category");
        let transitive_alias_order = db
            .execute_sparql(
                r#"SELECT ?kind
                          (SAMPLE(?term) AS ?first)
                          (COALESCE(?first, <urn:missing>) AS ?second)
                   WHERE {
                     VALUES (?kind ?term) {
                       ("literal" "urn:x")
                       ("iri" <urn:x>)
                     }
                   }
                   GROUP BY ?kind
                   ORDER BY ?second"#,
            )
            .expect("ORDER BY a transitive computed aggregate alias retains exact RDF category");

        let expected = [
            vec![Value::String("iri".into()), Value::String("urn:x".into())],
            vec![
                Value::String("literal".into()),
                Value::String("urn:x".into()),
            ],
        ];
        assert_eq!(aggregate_order.columns, ["kind", "sample"]);
        assert_eq!(aggregate_order.rows(), expected.as_slice());
        assert_eq!(alias_order.columns, ["kind", "sample"]);
        assert_eq!(alias_order.rows(), expected.as_slice());
        assert_eq!(transitive_alias_order.columns, ["kind", "first", "second"]);
        assert_eq!(
            transitive_alias_order.rows(),
            &[
                vec![
                    Value::String("iri".into()),
                    Value::String("urn:x".into()),
                    Value::String("urn:x".into()),
                ],
                vec![
                    Value::String("literal".into()),
                    Value::String("urn:x".into()),
                    Value::String("urn:x".into()),
                ],
            ]
        );
    }

    #[test]
    fn aggregate_alias_subselect_preserves_identity_for_outer_group_and_distinct() {
        let db = rdf_db();
        let ordinary_control = db
            .execute_sparql(
                r#"SELECT ?sample (COUNT(*) AS ?n)
                   WHERE {
                     {
                       SELECT (?term AS ?sample)
                       WHERE { VALUES ?term { "urn:x" <urn:x> } }
                     }
                   }
                   GROUP BY ?sample"#,
            )
            .expect("an ordinary subselect alias exports RDF identity to outer grouping");
        assert_eq!(ordinary_control.row_count(), 2);
        assert!(
            ordinary_control
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into()), Value::Int64(1)])
        );

        let native_control = db
            .execute_sparql(
                r#"SELECT ?sample (COUNT(*) AS ?n)
                   WHERE {
                     {
                       SELECT (SAMPLE(VECTOR(?component)) AS ?sample)
                       WHERE {
                         VALUES (?kind ?component) { ("a" 1) ("b" 2) }
                       }
                       GROUP BY ?kind
                     }
                   }
                   GROUP BY ?sample"#,
            )
            .expect("native aggregate aliases remain native across outer grouping");
        assert_eq!(native_control.row_count(), 2);
        assert!(
            native_control
                .rows()
                .iter()
                .all(|row| row.get(1) == Some(&Value::Int64(1)))
        );

        let bracketed_group_control = db
            .execute_sparql(
                r#"SELECT ?sample (COUNT(*) AS ?n)
                   WHERE {
                     {
                       SELECT (SAMPLE(?term) AS ?sample)
                       WHERE {
                         VALUES (?kind ?term) {
                           ("literal" "urn:x")
                           ("iri" <urn:x>)
                         }
                       }
                       GROUP BY ?kind
                     }
                   }
                   GROUP BY (?sample)"#,
            )
            .expect("transparent GROUP BY syntax preserves exported aggregate identity");
        assert_eq!(bracketed_group_control.row_count(), 2);
        assert!(
            bracketed_group_control
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into()), Value::Int64(1)])
        );

        let grouped = db
            .execute_sparql(
                r#"SELECT ?sample (COUNT(*) AS ?n)
                   WHERE {
                     {
                       SELECT (SAMPLE(?term) AS ?sample)
                       WHERE {
                         VALUES (?kind ?term) {
                           ("literal" "urn:x")
                           ("iri" <urn:x>)
                         }
                       }
                       GROUP BY ?kind
                     }
                   }
                   GROUP BY ?sample"#,
            )
            .expect("an aggregate alias exports RDF identity to outer grouping");
        assert_eq!(grouped.columns, ["sample", "n"]);
        assert_eq!(grouped.row_count(), 2);
        assert!(
            grouped
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into()), Value::Int64(1)])
        );

        let distinct = db
            .execute_sparql(
                r#"SELECT DISTINCT ?sample
                   WHERE {
                     {
                       SELECT (SAMPLE(?term) AS ?sample)
                       WHERE {
                         VALUES (?kind ?term) {
                           ("literal" "urn:x")
                           ("iri" <urn:x>)
                         }
                       }
                       GROUP BY ?kind
                     }
                   }"#,
            )
            .expect("an aggregate alias exports RDF identity to outer DISTINCT");
        assert_eq!(distinct.columns, ["sample"]);
        assert_eq!(distinct.row_count(), 2);
        assert!(
            distinct
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into())])
        );
    }

    #[test]
    fn computed_aggregate_alias_subselect_preserves_identity_for_outer_group() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?sample (COUNT(*) AS ?n)
                   WHERE {
                     {
                       SELECT (COALESCE(SAMPLE(?term), <urn:missing>) AS ?sample)
                       WHERE {
                         VALUES (?kind ?term) {
                           ("literal" "urn:x")
                           ("iri" <urn:x>)
                         }
                       }
                       GROUP BY ?kind
                     }
                   }
                   GROUP BY ?sample"#,
            )
            .expect("a computed aggregate alias exports RDF identity to outer grouping");

        assert_eq!(result.columns, ["sample", "n"]);
        assert_eq!(result.row_count(), 2);
        assert!(
            result
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into()), Value::Int64(1)])
        );
    }

    #[test]
    fn rdf_or_native_demand_flows_backward_through_pattern_binds() {
        let db = rdf_db();
        let ordinary_bind_group = db
            .execute_sparql(
                r#"SELECT ?derived (COUNT(*) AS ?n)
                   WHERE {
                     VALUES ?term { "urn:x" <urn:x> }
                     BIND(COALESCE(?term, <urn:missing>) AS ?derived)
                   }
                   GROUP BY ?derived"#,
            )
            .expect("a demanded ordinary BIND retains RDF identity for grouping");
        assert_eq!(ordinary_bind_group.columns, ["derived", "n"]);
        assert_eq!(ordinary_bind_group.row_count(), 2);
        assert!(
            ordinary_bind_group
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into()), Value::Int64(1)])
        );

        let simple_bind_group = db
            .execute_sparql(
                r#"SELECT ?derived (COUNT(*) AS ?n)
                   WHERE {
                     {
                       SELECT (SAMPLE(?term) AS ?sample)
                       WHERE {
                         VALUES (?kind ?term) {
                           ("literal" "urn:x")
                           ("iri" <urn:x>)
                         }
                       }
                       GROUP BY ?kind
                     }
                     BIND(?sample AS ?derived)
                   }
                   GROUP BY ?derived"#,
            )
            .expect("a demanded simple BIND retains aggregate identity for grouping");
        assert_eq!(simple_bind_group.columns, ["derived", "n"]);
        assert_eq!(simple_bind_group.row_count(), 2);
        assert!(
            simple_bind_group
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into()), Value::Int64(1)])
        );

        let computed_bind_distinct = db
            .execute_sparql(
                r#"SELECT DISTINCT ?derived
                   WHERE {
                     {
                       SELECT (SAMPLE(?term) AS ?sample)
                       WHERE {
                         VALUES (?kind ?term) {
                           ("literal" "urn:x")
                           ("iri" <urn:x>)
                         }
                       }
                       GROUP BY ?kind
                     }
                     BIND(COALESCE(?sample, <urn:missing>) AS ?derived)
                   }"#,
            )
            .expect("a demanded computed BIND retains aggregate identity for DISTINCT");
        assert_eq!(computed_bind_distinct.columns, ["derived"]);
        assert_eq!(computed_bind_distinct.row_count(), 2);
        assert!(
            computed_bind_distinct
                .rows()
                .iter()
                .all(|row| row == &vec![Value::String("urn:x".into())])
        );

        let chained_bind_order = db
            .execute_sparql(
                r#"SELECT ?kind ?second
                   WHERE {
                     {
                       SELECT ?kind (SAMPLE(?term) AS ?sample)
                       WHERE {
                         VALUES (?kind ?term) {
                           ("literal" "urn:x")
                           ("iri" <urn:x>)
                         }
                       }
                       GROUP BY ?kind
                     }
                     BIND(?sample AS ?first)
                     BIND(COALESCE(?first, <urn:missing>) AS ?second)
                   }
                   ORDER BY ?second"#,
            )
            .expect("ORDER demand flows through two sequential BIND aliases");
        assert_eq!(chained_bind_order.columns, ["kind", "second"]);
        assert_eq!(
            chained_bind_order.rows(),
            &[
                vec![Value::String("iri".into()), Value::String("urn:x".into())],
                vec![
                    Value::String("literal".into()),
                    Value::String("urn:x".into()),
                ],
            ]
        );
    }

    #[test]
    fn bracketed_group_by_preserves_native_copied_aggregate_alias_identity() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?sample (COUNT(*) AS ?n)
                   WHERE {
                     {
                       SELECT (SAMPLE(
                                 IF(
                                   ?kind = "string",
                                   "Vector([1.0])",
                                   VECTOR(1)
                                 )
                               ) AS ?source)
                              (?source AS ?sample)
                       WHERE { VALUES ?kind { "string" "vector" } }
                       GROUP BY ?kind
                     }
                   }
                   GROUP BY (?sample)"#,
            )
            .expect("bracketed GROUP BY retains a copied aggregate's native identity");

        assert_eq!(result.columns, ["sample", "n"]);
        assert_eq!(result.row_count(), 2);
        assert!(result.rows().contains(&vec![
            Value::String("Vector([1.0])".into()),
            Value::Int64(1)
        ]));
        assert!(
            result
                .rows()
                .contains(&vec![Value::Vector(vec![1.0].into()), Value::Int64(1)])
        );
    }

    #[test]
    fn aggregate_hoist_nested_consumers_preserve_unbound_and_error_semantics() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT DISTINCT
                          (sameTerm(
                            COALESCE(SUM(?usable), <urn:sum-error>),
                            <urn:sum-error>
                          ) AS ?sum_fell_back)
                          (sameTerm(
                            COALESCE(SAMPLE(?missing), <urn:missing>),
                            <urn:missing>
                          ) AS ?sample_fell_back)
                   WHERE {
                     VALUES (?numerator ?denominator) {
                       (2 1)
                       (99 0)
                       (UNDEF 1)
                     }
                     BIND(
                       IF(?denominator = 0, <urn:not-a-number>, ?numerator)
                       AS ?usable
                     )
                   }
                   HAVING (
                     sameTerm(
                       COALESCE(SUM(?usable), <urn:sum-error>),
                       <urn:sum-error>
                     ) &&
                     sameTerm(
                       COALESCE(SAMPLE(?missing), <urn:missing>),
                       <urn:missing>
                     )
                   )
                   ORDER BY (SAMPLE(?missing))"#,
            )
            .expect("nested aggregate consumers preserve SPARQL error and unbound semantics");

        assert_eq!(result.columns, ["sum_fell_back", "sample_fell_back"]);
        assert_eq!(result.rows(), &[vec![Value::Bool(true), Value::Bool(true)]]);
    }

    #[test]
    fn aggregate_hoist_nested_modifier_reuse_preserves_native_vector_sample() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT DISTINCT ?group
                          (IF(
                            COUNT(*) > 0,
                            SAMPLE(VECTOR(?component)),
                            VECTOR(0)
                          ) AS ?sample)
                   WHERE { VALUES (?group ?component) { ("a" 2) ("b" 1) } }
                   GROUP BY ?group
                   HAVING (COUNT(*) > 0)
                   ORDER BY (SAMPLE(VECTOR(?component)))"#,
            )
            .expect("native VECTOR SAMPLE survives nested SELECT and modifier consumers");

        assert_eq!(result.columns, ["group", "sample"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("b".into()), Value::Vector(vec![1.0].into()),],
                vec![Value::String("a".into()), Value::Vector(vec![2.0].into()),],
            ]
        );
    }

    #[test]
    fn aggregate_hoist_volatile_reuse_observes_one_stored_term() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (SAMPLE(UUID()) AS ?sample)
                          (IRI(STR(SAMPLE(UUID()))) AS ?copy)
                          (sameTerm(
                            SAMPLE(UUID()),
                            IRI(STR(SAMPLE(UUID())))
                          ) AS ?same)
                   WHERE { VALUES ?row { 1 } }
                   HAVING (sameTerm(
                     SAMPLE(UUID()),
                     IRI(STR(SAMPLE(UUID())))
                   ))
                   ORDER BY (SAMPLE(UUID()))"#,
            )
            .expect("SELECT, HAVING, and ORDER observe one stored volatile aggregate result");

        assert_eq!(result.columns, ["sample", "copy", "same"]);
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], result.rows()[0][1]);
        assert_eq!(result.rows()[0][2], Value::Bool(true));
    }

    #[test]
    fn aggregate_hoist_subselect_hidden_state_cannot_capture_outer_variable() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?value ?outer_same WHERE {
                     VALUES ?value { <urn:outer> }
                     {
                       SELECT *
                       WHERE { VALUES ?value { <urn:inner> } }
                       HAVING (sameTerm(SAMPLE(?value), <urn:inner>))
                     }
                     BIND(sameTerm(?value, <urn:outer>) AS ?outer_same)
                   }"#,
            )
            .expect("a zero-column subselect seals hidden helpers and its local ?value");

        assert_eq!(result.columns, ["value", "outer_same"]);
        assert_eq!(
            result.rows(),
            &[vec![Value::String("urn:outer".into()), Value::Bool(true)]]
        );
    }

    /// ORDER BY DESC(STR(?s)): descending with a function expression.
    #[test]
    fn sparql_order_by_desc_str() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let qr = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/name> ?o } ORDER BY DESC(STR(?s))")
            .unwrap();
        assert_eq!(
            qr.row_count(),
            2,
            "ORDER BY DESC(STR(?s)) should return 2 rows"
        );
    }

    // ---------------------------------------------------------------
    // Area 1: SPARQL translator projection with function expressions
    // ---------------------------------------------------------------

    /// STR() in SELECT projection: exercises the translate_projection path
    /// where a FunctionCall expression appears as a projected column.
    #[test]
    fn test_sparql_str_in_projection() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result =
            db.execute_sparql("SELECT (STR(?s) AS ?name) WHERE { ?s <http://ex.org/name> ?o }");

        let qr = result.unwrap();
        assert_eq!(
            qr.row_count(),
            2,
            "STR(?s) projection should return 2 rows (one per subject with name)"
        );
        // The projected alias should appear in the column list
        assert!(
            qr.columns.contains(&"name".to_string()),
            "Result should contain column 'name', got: {:?}",
            qr.columns
        );
        // Every projected value should be a non-empty string (the IRI serialised via STR)
        for row in qr.iter() {
            let val = &row[qr.columns.iter().position(|c| c == "name").unwrap()];
            let s = val.to_string();
            assert!(
                !s.is_empty(),
                "STR(?s) should produce a non-empty string, got: {val:?}"
            );
        }
    }

    /// STRLEN() in SELECT projection: exercises the translate_projection path
    /// where a FunctionCall expression appears as a projected column alongside
    /// a plain variable.
    ///
    /// STRLEN() in RDF projection is evaluated via RdfProjectOperator, which
    /// delegates to RdfExpressionPredicate for full SPARQL function support.
    #[test]
    fn test_sparql_strlen_in_projection() {
        let db = rdf_db();
        insert_sample_triples(&db);

        let result = db.execute_sparql(
            "SELECT ?name (STRLEN(?name) AS ?len) WHERE { ?s <http://ex.org/name> ?name }",
        );

        let qr = result.unwrap();
        assert_eq!(qr.row_count(), 2, "STRLEN projection should return 2 rows");
        assert!(
            qr.columns.contains(&"len".to_string()),
            "Result should contain column 'len', got: {:?}",
            qr.columns
        );

        let name_idx = qr.columns.iter().position(|c| c == "name").unwrap();
        let len_idx = qr.columns.iter().position(|c| c == "len").unwrap();

        for row in qr.iter() {
            let name_val = &row[name_idx];
            let len_val = &row[len_idx];

            // Extract the actual string content (without Display quotes)
            let name_content = match name_val {
                grafeo_common::types::Value::String(s) => s.as_str(),
                other => panic!("Expected String for ?name, got: {other:?}"),
            };
            let expected_len = name_content.len() as i64;

            match len_val {
                grafeo_common::types::Value::Int64(n) => {
                    assert_eq!(
                        *n, expected_len,
                        "STRLEN(\"{name_content}\") should be {expected_len}, got {n}"
                    );
                }
                grafeo_common::types::Value::Float64(f) => {
                    assert_eq!(
                        *f as i64, expected_len,
                        "STRLEN(\"{name_content}\") should be {expected_len}, got {f}"
                    );
                }
                grafeo_common::types::Value::String(s) => {
                    let parsed: i64 = s.parse().unwrap_or(-1);
                    assert_eq!(
                        parsed, expected_len,
                        "STRLEN(\"{name_content}\") should be {expected_len}, got \"{s}\""
                    );
                }
                grafeo_common::types::Value::Null => {
                    panic!("STRLEN should not return Null now that RdfProjectOperator is used");
                }
                other => {
                    panic!("STRLEN should return a numeric or Null value, got: {other:?}");
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Area 2: RDF planner GROUP BY with expression pre-projection
    // ---------------------------------------------------------------

    /// GROUP BY ?s with STR(?s) in projection and COUNT(*) aggregate.
    /// The final schema and values must follow SELECT, not leak the raw key.
    #[test]
    fn test_sparql_group_by_with_str() {
        let db = rdf_db();
        insert_sample_triples(&db);

        // Each subject has 2 triples (name + age), so grouping by ?s
        // with COUNT(*) should yield 2 per group.
        let result = db.execute_sparql(
            "SELECT (STR(?s) AS ?name) (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY ?s",
        );

        let qr = result.expect("a legal group-derived projection expression is materialized");
        assert_eq!(qr.columns, ["name", "cnt"]);

        let mut rows = qr.rows().to_vec();
        rows.sort_by(|left, right| left[0].to_string().cmp(&right[0].to_string()));
        assert_eq!(
            rows,
            vec![
                vec![Value::String("http://ex.org/alix".into()), Value::Int64(2),],
                vec![Value::String("http://ex.org/gus".into()), Value::Int64(2),],
            ],
            "the raw ?s grouping column must not leak in place of ?name",
        );
    }

    // ---------------------------------------------------------------
    // Area 3: SPARQL dateTime functions on typed literals
    // ---------------------------------------------------------------

    fn insert_datetime_triples(db: &GrafeoDB) {
        db.execute_sparql(
            r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            INSERT DATA {
                <http://ex.org/event1> <http://ex.org/date> "2024-06-15T14:30:45+05:30"^^xsd:dateTime .
                <http://ex.org/event2> <http://ex.org/date> "2024-12-25T08:00:00-08:00"^^xsd:dateTime .
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn sparql_year_month_day_from_zoned_datetime() {
        let db = rdf_db();
        insert_datetime_triples(&db);

        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?y ?m ?d WHERE {
                    <http://ex.org/event1> <http://ex.org/date> ?dt .
                    BIND(YEAR(?dt) AS ?y)
                    BIND(MONTH(?dt) AS ?m)
                    BIND(DAY(?dt) AS ?d)
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn sparql_hours_minutes_seconds_from_zoned_datetime() {
        let db = rdf_db();
        insert_datetime_triples(&db);

        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?h ?min ?sec WHERE {
                    <http://ex.org/event1> <http://ex.org/date> ?dt .
                    BIND(HOURS(?dt) AS ?h)
                    BIND(MINUTES(?dt) AS ?min)
                    BIND(SECONDS(?dt) AS ?sec)
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn sparql_timezone_and_tz_from_zoned_datetime() {
        let db = rdf_db();
        insert_datetime_triples(&db);

        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?tz WHERE {
                    <http://ex.org/event2> <http://ex.org/date> ?dt .
                    BIND(TZ(?dt) AS ?tz)
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    // ---------------------------------------------------------------
    // Area 4: SPARQL LANG() and LANGMATCHES()
    // ---------------------------------------------------------------

    fn insert_language_tagged_triples(db: &GrafeoDB) {
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://www.w3.org/2000/01/rdf-schema#label> "Alix"@en .
                <http://ex.org/alix> <http://www.w3.org/2000/01/rdf-schema#label> "Alix"@fr .
                <http://ex.org/gus>  <http://www.w3.org/2000/01/rdf-schema#label> "Gus"@en-US .
                <http://ex.org/item> <http://www.w3.org/2000/01/rdf-schema#label> "Plain" .
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn sparql_langmatches_exact() {
        let db = rdf_db();
        insert_language_tagged_triples(&db);

        let result = db
            .execute_sparql(
                r#"SELECT ?label WHERE {
                    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                    FILTER(LANGMATCHES(LANG(?label), "en"))
                }"#,
            )
            .unwrap();
        // Should match "en" and "en-US" (prefix match)
        assert!(
            result.row_count() >= 2,
            "LANGMATCHES should match 'en' and 'en-US', got {} rows",
            result.row_count()
        );
    }

    #[test]
    fn sparql_langmatches_wildcard() {
        let db = rdf_db();
        insert_language_tagged_triples(&db);

        let result = db
            .execute_sparql(
                r#"SELECT ?label WHERE {
                    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                    FILTER(LANGMATCHES(LANG(?label), "*"))
                }"#,
            )
            .unwrap();
        // Should match all language-tagged literals (en, fr, en-US) but NOT "Plain"
        assert!(
            result.row_count() >= 3,
            "LANGMATCHES(*) should match all tagged literals, got {} rows",
            result.row_count()
        );
    }

    #[test]
    fn sparql_query_without_lang_columns() {
        // Verify strip_internal_columns works: __lang_ columns should not appear in results
        let db = rdf_db();
        insert_language_tagged_triples(&db);

        let result = db
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                }"#,
            )
            .unwrap();
        for col in &result.columns {
            assert!(
                !col.starts_with("__lang_"),
                "Internal __lang_ column should be stripped from results, found: {col}"
            );
        }
    }
}
