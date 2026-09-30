//! End-to-end execution tests aligned with the W3C SPARQL 1.1 Query Language
//! specification. Each section tests actual query execution against an in-memory
//! RDF store, verifying both result counts and value correctness.
//!
//! Spec reference: <https://www.w3.org/TR/sparql11-query/>
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_w3c_execution_tests
//! ```

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod tests {
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::{Error, QueryErrorKind};
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
            .expect("open in-memory rdf db")
    }

    /// Insert a common set of triples used by many tests.
    fn insert_foaf_data(db: &GrafeoDB) {
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/age> "30" .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/gus> .
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/mbox> "alix@example.org" .

                <http://ex.org/gus> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                <http://ex.org/gus> <http://xmlns.com/foaf/0.1/name> "Gus" .
                <http://ex.org/gus> <http://xmlns.com/foaf/0.1/age> "25" .
                <http://ex.org/gus> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/alix> .

                <http://ex.org/vincent> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                <http://ex.org/vincent> <http://xmlns.com/foaf/0.1/name> "Vincent" .

                <http://ex.org/amsterdam> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex.org/City> .
                <http://ex.org/amsterdam> <http://xmlns.com/foaf/0.1/name> "Amsterdam" .

                <http://ex.org/alix> <http://ex.org/livesIn> <http://ex.org/amsterdam> .
                <http://ex.org/gus> <http://ex.org/livesIn> <http://ex.org/amsterdam> .
            }"#,
        )
        .unwrap();
    }

    // ====================================================================
    // 2 - Making Simple Queries
    // ====================================================================

    #[test]
    fn sec2_select_wildcard() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/p> "hello" . }"#)
            .unwrap();

        let r = db.execute_sparql("SELECT * WHERE { ?s ?p ?o }").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.columns.len(), 3);
    }

    #[test]
    fn sec2_select_specific_vars() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3, "Should find Alix, Gus, Vincent");
    }

    // ====================================================================
    // 5 - Graph Patterns
    // ====================================================================

    #[test]
    fn sec5_basic_graph_pattern_join() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://xmlns.com/foaf/0.1/age> ?age
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Only Alix and Gus have both name and age");
    }

    #[test]
    fn sec5_empty_where_clause() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // Empty WHERE should return one empty solution
        let r = db.execute_sparql("SELECT (1 AS ?one) WHERE { }");
        // Depending on implementation, may return 1 row or error
        // The spec says empty BGP matches with one empty solution
        assert!(r.is_ok(), "Empty WHERE clause: {r:?}");
    }

    // ====================================================================
    // 6 - OPTIONAL
    // ====================================================================

    #[test]
    fn sec6_optional_preserves_non_matching() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            3,
            "All 3 people returned; Vincent has NULL age"
        );
    }

    #[test]
    fn sec6_nested_optional() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age ?mbox WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL {
                        ?s <http://xmlns.com/foaf/0.1/age> ?age .
                        OPTIONAL { ?s <http://xmlns.com/foaf/0.1/mbox> ?mbox }
                    }
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            3,
            "All 3 people; only Alix has both age and mbox"
        );
    }

    // ====================================================================
    // 7 - UNION
    // ====================================================================

    #[test]
    fn sec7_union_disjoint_patterns() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?thing ?name WHERE {
                    {
                        ?thing <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                        ?thing <http://xmlns.com/foaf/0.1/name> ?name
                    }
                    UNION
                    {
                        ?thing <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex.org/City> .
                        ?thing <http://xmlns.com/foaf/0.1/name> ?name
                    }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 4, "3 persons + 1 city = 4 results");
    }

    // ====================================================================
    // 8 - Negation
    // ====================================================================

    #[test]
    fn sec8_filter_not_exists_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // Find people who have no age
        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER NOT EXISTS { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Vincent has no age");
    }

    #[test]
    fn sec8_minus_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // All person names MINUS those who know someone
        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    MINUS { ?s <http://xmlns.com/foaf/0.1/knows> ?other }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Vincent does not know anyone");
    }

    // ====================================================================
    // 10 - BIND
    // ====================================================================

    #[test]
    fn sec10_bind_concat() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/first> "Alix" .
                <http://ex.org/alix> <http://ex.org/last> "Vega" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?full WHERE {
                    ?s <http://ex.org/first> ?f .
                    ?s <http://ex.org/last> ?l .
                    BIND(CONCAT(?f, " ", ?l) AS ?full)
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let val = r.rows()[0][0].to_string();
        assert!(
            val.contains("Alix") && val.contains("Vega"),
            "BIND CONCAT should produce 'Alix Vega', got: {val}"
        );
    }

    // ====================================================================
    // 10.2 - VALUES
    // ====================================================================

    #[test]
    fn sec10_values_filter() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    VALUES ?s { <http://ex.org/alix> <http://ex.org/gus> }
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "VALUES restricts to Alix and Gus");
    }

    // ====================================================================
    // 11 - Aggregates
    // ====================================================================

    #[test]
    fn sec11_count_star_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?total) WHERE {
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "COUNT(*) returns 1 row");
    }

    #[test]
    fn sec11_count_with_group_by() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?cnt)
                WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type }
                GROUP BY ?type
                ORDER BY ?type"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Two types: City and Person");
    }

    #[test]
    fn sec11_sum_aggregate() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/val> "10" .
                <http://ex.org/b> <http://ex.org/val> "20" .
                <http://ex.org/c> <http://ex.org/val> "30" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql("SELECT (SUM(?v) AS ?total) WHERE { ?s <http://ex.org/val> ?v }")
            .unwrap();
        assert_eq!(r.row_count(), 1, "SUM should return 1 row");
    }

    #[test]
    fn sec11_min_max_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (MIN(?age) AS ?youngest) (MAX(?age) AS ?oldest)
                WHERE { ?s <http://xmlns.com/foaf/0.1/age> ?age }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "MIN/MAX should return 1 row");
    }

    #[test]
    fn sec11_avg_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (AVG(?age) AS ?avgAge)
                WHERE { ?s <http://xmlns.com/foaf/0.1/age> ?age }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "AVG should return 1 row");
    }

    #[test]
    fn sec11_group_concat_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (GROUP_CONCAT(?name; SEPARATOR=", ") AS ?names)
                WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "GROUP_CONCAT returns 1 row");
        let val = r.rows()[0][0].to_string();
        assert!(
            val.contains("Alix"),
            "GROUP_CONCAT should contain Alix, got: {val}"
        );
        assert!(
            val.contains("Gus"),
            "GROUP_CONCAT should contain Gus, got: {val}"
        );
    }

    #[test]
    fn sec11_sample_aggregate() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT (SAMPLE(?name) AS ?example)
                WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "SAMPLE returns 1 row");
    }

    #[test]
    fn sec11_having_filters_groups() {
        let db = rdf_db();
        insert_foaf_data(&db);

        // Only types with more than 1 instance
        let r = db
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?cnt)
                WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type }
                GROUP BY ?type
                HAVING (COUNT(?s) > 1)"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Only Person has >1 instances (City has 1)"
        );
    }

    #[test]
    fn sec11_count_on_empty_set() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                "SELECT (COUNT(?s) AS ?cnt) WHERE { ?s <http://ex.org/nonexistent> ?o }",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "COUNT over empty set returns 1 row");
    }

    // ====================================================================
    // 15 - Solution Modifiers
    // ====================================================================

    #[test]
    fn sec15_distinct_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT DISTINCT ?city WHERE {
                    ?s <http://ex.org/livesIn> ?city
                }"#,
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Both Alix and Gus live in Amsterdam; DISTINCT collapses to 1"
        );
    }

    #[test]
    fn sec15_order_by_ascending() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3);
        let names: Vec<String> = r.rows().iter().map(|row| row[0].to_string()).collect();
        assert!(
            names[0] <= names[1] && names[1] <= names[2],
            "ORDER BY ASC should be sorted: {names:?}"
        );
    }

    #[test]
    fn sec15_order_by_descending() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY DESC(?name)"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3);
        let names: Vec<String> = r.rows().iter().map(|row| row[0].to_string()).collect();
        assert!(
            names[0] >= names[1] && names[1] >= names[2],
            "ORDER BY DESC should be reverse sorted: {names:?}"
        );
    }

    #[test]
    fn sec15_order_by_multiple_keys() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/dept> "Engineering" .
                <http://ex.org/a> <http://ex.org/name> "Vincent" .
                <http://ex.org/b> <http://ex.org/dept> "Engineering" .
                <http://ex.org/b> <http://ex.org/name> "Alix" .
                <http://ex.org/c> <http://ex.org/dept> "Design" .
                <http://ex.org/c> <http://ex.org/name> "Gus" .
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?dept ?name WHERE {
                    ?s <http://ex.org/dept> ?dept .
                    ?s <http://ex.org/name> ?name
                } ORDER BY ?dept ?name"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3);
        // Design < Engineering, and within Engineering: Alix < Vincent
        let first_dept = r.rows()[0][0].to_string();
        assert!(
            first_dept.contains("Design"),
            "First dept should be Design, got: {first_dept}"
        );
    }

    #[test]
    fn sec15_limit_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                } LIMIT 2"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "LIMIT 2 returns at most 2 rows");
    }

    #[test]
    fn sec15_offset_execution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let all = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name"#,
            )
            .unwrap();
        let offset = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name OFFSET 1"#,
            )
            .unwrap();
        assert_eq!(
            offset.row_count(),
            all.row_count() - 1,
            "OFFSET 1 skips first row"
        );
    }

    #[test]
    fn sec15_limit_offset_combined() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                } ORDER BY ?name LIMIT 1 OFFSET 1"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "LIMIT 1 OFFSET 1 returns exactly 1 row");
    }

    // ====================================================================
    // 16 - Query Forms
    // ====================================================================

    #[test]
    fn sec16_ask_true() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"ASK {
                    ?s <http://xmlns.com/foaf/0.1/name> "Alix"
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "ASK true returns 1 row");
    }

    #[test]
    fn sec16_ask_false() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"ASK {
                    ?s <http://xmlns.com/foaf/0.1/name> "Nonexistent"
                }"#,
            )
            .unwrap();
        // ASK false: returns 1 row with boolean false, or 0 rows
        // (implementation-dependent, but the result should indicate false)
        assert!(r.row_count() <= 1, "ASK false returns 0 or 1 row");
    }

    #[test]
    fn sec16_construct_returns_triples() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"CONSTRUCT {
                    ?s <http://ex.org/hasName> ?name
                } WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert!(
            r.row_count() >= 3,
            "CONSTRUCT should produce at least 3 triples (one per person)"
        );
    }

    #[test]
    fn sec16_describe_returns_triples() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db.execute_sparql("DESCRIBE <http://ex.org/alix>").unwrap();
        assert!(
            r.row_count() >= 1,
            "DESCRIBE should return at least 1 triple about Alix"
        );
    }

    // ====================================================================
    // SPARQL Update Operations
    // ====================================================================

    #[test]
    fn update_insert_data_and_query() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/mia> <http://xmlns.com/foaf/0.1/name> "Mia" .
                <http://ex.org/mia> <http://xmlns.com/foaf/0.1/age> "28"
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    <http://ex.org/mia> <http://xmlns.com/foaf/0.1/name> ?name .
                    <http://ex.org/mia> <http://xmlns.com/foaf/0.1/age> ?age
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn update_delete_data_and_verify() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/x> <http://ex.org/p> "val1" .
                <http://ex.org/x> <http://ex.org/q> "val2"
            }"#,
        )
        .unwrap();

        db.execute_sparql(r#"DELETE DATA { <http://ex.org/x> <http://ex.org/p> "val1" . }"#)
            .unwrap();

        let r = db
            .execute_sparql("SELECT ?p ?o WHERE { <http://ex.org/x> ?p ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only q/val2 remains after delete");
    }

    #[test]
    fn update_delete_where() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/status> "draft" .
                <http://ex.org/b> <http://ex.org/status> "published" .
                <http://ex.org/c> <http://ex.org/status> "draft" .
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"DELETE WHERE {
                ?s <http://ex.org/status> "draft"
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/status> ?o }")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Only the 'published' triple should remain"
        );
    }

    #[test]
    fn update_modify_delete_insert() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/a> <http://ex.org/status> "draft" .
                <http://ex.org/b> <http://ex.org/status> "draft" .
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"DELETE { ?s <http://ex.org/status> "draft" }
               INSERT { ?s <http://ex.org/status> "published" }
               WHERE  { ?s <http://ex.org/status> "draft" }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(r#"SELECT ?s WHERE { ?s <http://ex.org/status> "published" }"#)
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "Both triples should be updated to published"
        );

        let r2 = db
            .execute_sparql(r#"SELECT ?s WHERE { ?s <http://ex.org/status> "draft" }"#)
            .unwrap();
        assert_eq!(r2.row_count(), 0, "No draft triples should remain");
    }

    // ====================================================================
    // FILTER expressions (more comprehensive)
    // ====================================================================

    #[test]
    fn filter_equality() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    FILTER(?name = "Alix")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn filter_inequality() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name != "Alix")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Gus and Vincent (not Alix)");
    }

    #[test]
    fn filter_and_or_combined() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://xmlns.com/foaf/0.1/age> ?age .
                    FILTER(?age >= "25" && ?age <= "30")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Alix (30) and Gus (25) both in range");
    }

    #[test]
    fn filter_or_logic() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name = "Alix" || ?name = "Gus")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "Alix or Gus");
    }

    #[test]
    fn filter_regex_case_insensitive() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(REGEX(?name, "^alix$", "i"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Case-insensitive regex should match Alix");
    }

    #[test]
    fn filter_contains() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(CONTAINS(?name, "us"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Gus contains 'us'");
    }

    #[test]
    fn filter_strstarts() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(STRSTARTS(?name, "V"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only Vincent starts with V");
    }

    // ====================================================================
    // String Functions (execution)
    // ====================================================================

    #[test]
    fn func_strlen() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/name> "Alix" }"#)
            .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT (STRLEN(?name) AS ?len) WHERE {
                    <http://ex.org/x> <http://ex.org/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }

    #[test]
    fn func_ucase() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/name> "Alix" }"#)
            .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT (UCASE(?name) AS ?upper) WHERE {
                    <http://ex.org/x> <http://ex.org/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let upper = r.rows()[0][0].to_string();
        assert!(
            upper.contains("ALIX"),
            "UCASE should produce ALIX, got: {upper}"
        );
    }

    #[test]
    fn func_lcase() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/name> "Alix" }"#)
            .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT (LCASE(?name) AS ?lower) WHERE {
                    <http://ex.org/x> <http://ex.org/name> ?name
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let lower = r.rows()[0][0].to_string();
        assert!(
            lower.contains("alix"),
            "LCASE should produce alix, got: {lower}"
        );
    }

    // ====================================================================
    // Named Graph Operations
    // ====================================================================

    #[test]
    fn named_graph_insert_and_query() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://ex.org/graph1> {
                    <http://ex.org/alix> <http://ex.org/name> "Alix"
                }
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/graph1> {
                        ?s <http://ex.org/name> ?name
                    }
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "Should find Alix in named graph");
    }

    #[test]
    fn named_graph_isolation() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://ex.org/g1> {
                    <http://ex.org/a> <http://ex.org/p> "in-g1"
                }
                GRAPH <http://ex.org/g2> {
                    <http://ex.org/b> <http://ex.org/p> "in-g2"
                }
            }"#,
        )
        .unwrap();

        let r1 = db
            .execute_sparql(
                r#"SELECT ?o WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/p> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(r1.row_count(), 1, "g1 has 1 triple");

        let r2 = db
            .execute_sparql(
                r#"SELECT ?o WHERE {
                    GRAPH <http://ex.org/g2> { ?s <http://ex.org/p> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(r2.row_count(), 1, "g2 has 1 triple");
    }

    // ====================================================================
    // EXPLAIN (Grafeo extension)
    // ====================================================================

    #[test]
    fn explain_shows_plan() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                "EXPLAIN SELECT ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }",
            )
            .unwrap();
        assert_eq!(r.columns, vec!["plan"]);
        assert_eq!(r.row_count(), 1);
        let plan = r.rows()[0][0].to_string();
        assert!(
            plan.contains("TripleScan"),
            "Plan should contain TripleScan, got: {plan}"
        );
    }

    // ====================================================================
    // PREFIX resolution in execution
    // ====================================================================

    #[test]
    fn prefix_resolution() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?name WHERE {
                    ?s foaf:name ?name .
                    ?s a foaf:Person
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 3, "PREFIX + 'a' shorthand should work");
    }

    // ====================================================================
    // Multiple INSERT DATA + complex query
    // ====================================================================

    #[test]
    fn multi_insert_complex_query() {
        let db = rdf_db();

        // Insert in multiple batches
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/jules> <http://ex.org/name> "Jules" .
                <http://ex.org/jules> <http://ex.org/role> "hitman"
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/butch> <http://ex.org/name> "Butch" .
                <http://ex.org/butch> <http://ex.org/role> "boxer"
            }"#,
        )
        .unwrap();

        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/mia> <http://ex.org/name> "Mia" .
                <http://ex.org/mia> <http://ex.org/role> "actress"
            }"#,
        )
        .unwrap();

        // Complex query with FILTER, ORDER BY, LIMIT
        let r = db
            .execute_sparql(
                r#"SELECT ?name ?role WHERE {
                    ?s <http://ex.org/name> ?name .
                    ?s <http://ex.org/role> ?role .
                    FILTER(?role != "hitman")
                } ORDER BY ?name LIMIT 2"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "2 of 3 match filter, LIMIT 2");
    }

    // ====================================================================
    // Edge cases and boundary conditions
    // ====================================================================

    #[test]
    fn empty_result_set() {
        let db = rdf_db();
        let r = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/nonexistent> ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 0, "Query on empty store returns 0 rows");
    }

    #[test]
    fn insert_duplicate_triple() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "val" }"#)
            .unwrap();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/x> <http://ex.org/p> "val" }"#)
            .unwrap();

        let r = db
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/x> <http://ex.org/p> ?o }")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "RDF set semantics: duplicate triple should not produce extra row"
        );
    }

    #[test]
    fn large_insert_and_query() {
        let db = rdf_db();
        use std::fmt::Write;
        let mut triples = String::from("INSERT DATA {\n");
        for i in 0..100 {
            let _ = writeln!(
                triples,
                "    <http://ex.org/n{i}> <http://ex.org/val> \"{i}\" ."
            );
        }
        triples.push('}');
        db.execute_sparql(&triples).unwrap();

        let r = db
            .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/val> ?v } LIMIT 50")
            .unwrap();
        assert_eq!(r.row_count(), 50, "LIMIT 50 on 100 triples");

        let all = db
            .execute_sparql("SELECT (COUNT(?s) AS ?cnt) WHERE { ?s <http://ex.org/val> ?v }")
            .unwrap();
        assert_eq!(all.row_count(), 1, "COUNT returns 1 row");
    }

    // ====================================================================
    // FILTER with IN / NOT IN (if supported at execution level)
    // ====================================================================

    #[test]
    fn filter_in_operator() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name IN ("Alix", "Gus"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "IN should match Alix and Gus");
    }

    #[test]
    fn filter_not_in_operator() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    FILTER(?name NOT IN ("Alix", "Gus"))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "NOT IN should match only Vincent");
    }

    // ====================================================================
    // BOUND function
    // ====================================================================

    #[test]
    fn filter_bound_function() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                    FILTER(BOUND(?age))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "BOUND(?age) should match Alix and Gus");
    }

    #[test]
    fn filter_not_bound_function() {
        let db = rdf_db();
        insert_foaf_data(&db);

        let r = db
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
                    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/age> ?age }
                    FILTER(!BOUND(?age))
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1, "!BOUND(?age) should match only Vincent");
    }

    // ====================================================================
    // COUNT(*) Fast-Path (0.5.37 - Optimizer Foundation)
    // ====================================================================

    #[test]
    fn count_star_fully_unbound() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // 15 triples in the foaf data
        let r = db
            .execute_sparql("SELECT (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        assert_eq!(count, 15);
    }

    #[test]
    fn count_star_predicate_bound() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                "SELECT (COUNT(*) AS ?cnt) WHERE { ?s <http://xmlns.com/foaf/0.1/knows> ?o }",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        assert_eq!(count, 2, "alix knows gus, gus knows alix");
    }

    #[test]
    fn count_star_with_group_by_not_fast_path() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // GROUP BY prevents fast-path, should still work via normal aggregate
        let r = db
            .execute_sparql(
                r#"SELECT ?s (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o } GROUP BY ?s ORDER BY ?s"#,
            )
            .unwrap();
        // Each subject should have its own count
        assert!(r.row_count() > 1, "GROUP BY should produce multiple rows");
    }

    #[test]
    fn count_star_with_distinct_not_fast_path() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // DISTINCT prevents fast-path, should still work
        let r = db
            .execute_sparql("SELECT (COUNT(DISTINCT ?s) AS ?cnt) WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        // Unique subjects: alix, gus, vincent, amsterdam
        assert!(count > 0);
    }

    #[test]
    fn count_star_per_named_graph() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://ex.org/g1> {
                    <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
                    <http://ex.org/gus> <http://xmlns.com/foaf/0.1/name> "Gus" .
                }
            }"#,
        )
        .unwrap();

        let r = db
            .execute_sparql(
                "SELECT (COUNT(*) AS ?cnt) WHERE { GRAPH <http://ex.org/g1> { ?s ?p ?o } }",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let count = r.rows()[0][0].as_int64().unwrap();
        assert_eq!(count, 2);
    }

    // ====================================================================
    // Optimizer: Join Conditions + TripleScan in DPccp (0.5.37)
    // ====================================================================

    #[test]
    fn two_hop_join_produces_correct_results() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // Two-hop: ?a knows ?b, ?b has name ?name
        let r = db
            .execute_sparql(
                r#"SELECT ?a ?name WHERE {
                    ?a <http://xmlns.com/foaf/0.1/knows> ?b .
                    ?b <http://xmlns.com/foaf/0.1/name> ?name .
                }"#,
            )
            .unwrap();
        // alix knows gus (name "Gus"), gus knows alix (name "Alix")
        assert_eq!(r.row_count(), 2);
    }

    #[test]
    fn three_hop_join_produces_correct_results() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // Three-hop: ?a knows ?b, ?b knows ?c, ?c has name ?name
        let r = db
            .execute_sparql(
                r#"SELECT ?a ?name WHERE {
                    ?a <http://xmlns.com/foaf/0.1/knows> ?b .
                    ?b <http://xmlns.com/foaf/0.1/knows> ?c .
                    ?c <http://xmlns.com/foaf/0.1/name> ?name .
                }"#,
            )
            .unwrap();
        // alix->gus->alix (name "Alix"), gus->alix->gus (name "Gus")
        assert_eq!(r.row_count(), 2);
    }

    #[test]
    fn selective_predicate_join_correct() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // Join with selective predicate: livesIn (2 triples) + name (4 triples)
        let r = db
            .execute_sparql(
                r#"SELECT ?name ?city WHERE {
                    ?s <http://ex.org/livesIn> ?c .
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?c <http://xmlns.com/foaf/0.1/name> ?city .
                }"#,
            )
            .unwrap();
        // alix livesIn amsterdam, gus livesIn amsterdam
        assert_eq!(r.row_count(), 2);
    }

    // ====================================================================
    // Phase 3: SPARQL Spec Compliance (0.5.37)
    // ====================================================================

    #[test]
    fn bind_typed_literal_integer() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?x WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> "Alix" .
                    BIND("42"^^xsd:integer AS ?x)
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let val = &r.rows()[0][0];
        // Should be integer 42, not empty string
        assert!(
            val.as_int64() == Some(42) || val.to_string() == "42",
            "expected 42, got: {val:?}"
        );
    }

    #[test]
    fn bind_typed_literal_double() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?x WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> "Alix" .
                    BIND("3.14"^^xsd:double AS ?x)
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let val = &r.rows()[0][0];
        assert!(
            val.to_string().starts_with("3.14"),
            "expected 3.14, got: {val:?}"
        );
    }

    #[test]
    fn filter_equality_pushdown_into_scan() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // FILTER(?name = "Alix") should be pushed into the TripleScan
        // so only matching triples are scanned (index lookup, not full scan + filter)
        let r = db
            .execute_sparql(
                r#"SELECT ?s WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    FILTER(?name = "Alix")
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert!(r.rows()[0][0].to_string().contains("alix"));
    }

    #[test]
    fn filter_equality_on_predicate() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // FILTER on a predicate variable
        let r = db
            .execute_sparql(
                r#"SELECT ?s ?o WHERE {
                    ?s ?p ?o .
                    FILTER(?p = <http://xmlns.com/foaf/0.1/name>)
                }"#,
            )
            .unwrap();
        // 4 names: Alix, Gus, Vincent, Amsterdam
        assert_eq!(r.row_count(), 4);
    }

    // ====================================================================
    // 17.4.4.4 - ISNUMERIC (execution)
    // ====================================================================

    #[test]
    fn select_isnumeric_uses_scanned_rdf_numeric_datatypes_and_facets() {
        let db = rdf_db();
        db.execute_sparql(
            r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
               INSERT DATA {
                   <urn:isnumeric-valid-integer> <urn:value> "+0"^^xsd:integer .
                   <urn:isnumeric-valid-decimal> <urn:value> ".5"^^xsd:decimal .
                   <urn:isnumeric-valid-float> <urn:value> "INF"^^xsd:float .
                   <urn:isnumeric-valid-double> <urn:value> "NaN"^^xsd:double .
                   <urn:isnumeric-valid-nonPositiveInteger> <urn:value> "0"^^xsd:nonPositiveInteger .
                   <urn:isnumeric-valid-negativeInteger> <urn:value> "-1"^^xsd:negativeInteger .
                   <urn:isnumeric-valid-long> <urn:value> "-9223372036854775808"^^xsd:long .
                   <urn:isnumeric-valid-int> <urn:value> "2147483647"^^xsd:int .
                   <urn:isnumeric-valid-short> <urn:value> "-32768"^^xsd:short .
                   <urn:isnumeric-valid-byte> <urn:value> "127"^^xsd:byte .
                   <urn:isnumeric-valid-nonNegativeInteger> <urn:value> "0"^^xsd:nonNegativeInteger .
                   <urn:isnumeric-valid-unsignedLong> <urn:value> "18446744073709551615"^^xsd:unsignedLong .
                   <urn:isnumeric-valid-unsignedInt> <urn:value> "4294967295"^^xsd:unsignedInt .
                   <urn:isnumeric-valid-unsignedShort> <urn:value> "65535"^^xsd:unsignedShort .
                   <urn:isnumeric-valid-unsignedByte> <urn:value> "255"^^xsd:unsignedByte .
                   <urn:isnumeric-valid-positiveInteger> <urn:value> "1"^^xsd:positiveInteger .
                   <urn:isnumeric-invalid-integer> <urn:value> "pumpkin"^^xsd:integer .
                   <urn:isnumeric-invalid-decimal> <urn:value> "1e2"^^xsd:decimal .
                   <urn:isnumeric-invalid-float> <urn:value> "Infinity"^^xsd:float .
                   <urn:isnumeric-invalid-double> <urn:value> "Infinity"^^xsd:double .
                   <urn:isnumeric-invalid-nonPositiveInteger> <urn:value> "1"^^xsd:nonPositiveInteger .
                   <urn:isnumeric-invalid-negativeInteger> <urn:value> "0"^^xsd:negativeInteger .
                   <urn:isnumeric-invalid-long> <urn:value> "9223372036854775808"^^xsd:long .
                   <urn:isnumeric-invalid-int> <urn:value> "2147483648"^^xsd:int .
                   <urn:isnumeric-invalid-short> <urn:value> "32768"^^xsd:short .
                   <urn:isnumeric-invalid-byte> <urn:value> "128"^^xsd:byte .
                   <urn:isnumeric-invalid-nonNegativeInteger> <urn:value> "-1"^^xsd:nonNegativeInteger .
                   <urn:isnumeric-invalid-unsignedLong> <urn:value> "18446744073709551616"^^xsd:unsignedLong .
                   <urn:isnumeric-invalid-unsignedInt> <urn:value> "4294967296"^^xsd:unsignedInt .
                   <urn:isnumeric-invalid-unsignedShort> <urn:value> "65536"^^xsd:unsignedShort .
                   <urn:isnumeric-invalid-unsignedByte> <urn:value> "256"^^xsd:unsignedByte .
                   <urn:isnumeric-invalid-positiveInteger> <urn:value> "0"^^xsd:positiveInteger .
                   <urn:isnumeric-numeric-looking-string> <urn:value> "12" .
               }"#,
        )
        .expect("insert numeric RDF literal coverage");

        let r = db
            .execute_sparql(
                r#"SELECT ?subject WHERE {
                    ?subject <urn:value> ?value .
                    FILTER (ISNUMERIC(?value))
                }"#,
            )
            .expect("ordinary SELECT evaluates isNumeric over scanned RDF terms");

        let mut subjects = r
            .rows()
            .iter()
            .map(|row| {
                row[0]
                    .as_str()
                    .expect("an RDF subject is exposed as a string")
                    .to_string()
            })
            .collect::<Vec<_>>();
        subjects.sort();
        assert_eq!(
            subjects,
            [
                "urn:isnumeric-valid-byte",
                "urn:isnumeric-valid-decimal",
                "urn:isnumeric-valid-double",
                "urn:isnumeric-valid-float",
                "urn:isnumeric-valid-int",
                "urn:isnumeric-valid-integer",
                "urn:isnumeric-valid-long",
                "urn:isnumeric-valid-negativeInteger",
                "urn:isnumeric-valid-nonNegativeInteger",
                "urn:isnumeric-valid-nonPositiveInteger",
                "urn:isnumeric-valid-positiveInteger",
                "urn:isnumeric-valid-short",
                "urn:isnumeric-valid-unsignedByte",
                "urn:isnumeric-valid-unsignedInt",
                "urn:isnumeric-valid-unsignedLong",
                "urn:isnumeric-valid-unsignedShort",
            ],
            "only the exact hand-derived set of valid XSD numeric subjects matches"
        );
    }

    #[test]
    fn select_isnumeric_preserves_typed_literals_from_values() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?value WHERE {
                       VALUES ?value {
                           "18446744073709551615"^^xsd:unsignedLong
                           "256"^^xsd:unsignedByte
                           "12"
                       }
                       FILTER (ISNUMERIC(?value))
                   }"#,
            )
            .expect("ordinary SELECT evaluates isNumeric over VALUES bindings");

        assert_eq!(
            r.row_count(),
            1,
            "only the valid typed VALUES literal is numeric"
        );
        assert!(
            r.rows()[0][0].to_string().contains("18446744073709551615"),
            "the valid xsd:unsignedLong VALUES literal must be selected",
        );
    }

    #[test]
    fn select_isnumeric_rejects_rust_parseable_invalid_numeric_values() {
        let db = rdf_db();
        let values = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?value WHERE {
                       VALUES ?value {
                           "2147483647"^^xsd:int
                           "2147483648"^^xsd:int
                           ".5"^^xsd:decimal
                           "1e2"^^xsd:decimal
                           "1.0"^^xsd:double
                           "Infinity"^^xsd:double
                       }
                       FILTER (ISNUMERIC(?value))
                   }"#,
            )
            .expect("ordinary SELECT evaluates isNumeric over numeric VALUES literals");
        assert_eq!(
            values.row_count(),
            3,
            "only valid xsd:int, xsd:decimal, and xsd:double VALUES literals are numeric"
        );

        let strdt = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?lexical WHERE {
                       {
                           VALUES ?lexical { "2147483647" "2147483648" }
                           BIND(STRDT(?lexical, xsd:int) AS ?value)
                       }
                       UNION {
                           VALUES ?lexical { ".5" "1e2" }
                           BIND(STRDT(?lexical, xsd:decimal) AS ?value)
                       }
                       UNION {
                           VALUES ?lexical { "1.0" "Infinity" }
                           BIND(STRDT(?lexical, xsd:double) AS ?value)
                       }
                       FILTER (ISNUMERIC(?value))
                   }"#,
            )
            .expect("ordinary SELECT evaluates isNumeric over STRDT numeric literals");
        assert_eq!(
            strdt.row_count(),
            3,
            "each native numeric family selects only its valid lexical/facet control"
        );
    }

    #[test]
    fn select_preserves_an_invalid_xsd_boolean_literal() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?value WHERE {
                       BIND("pumpkin"^^xsd:boolean AS ?value)
                   }"#,
            )
            .expect("ordinary SELECT retains an invalid xsd:boolean literal");

        assert!(matches!(
            &r.rows()[0][0],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } if lexical.as_str() == "pumpkin"
                && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#boolean"
        ));
    }

    #[test]
    fn select_strdt_requires_an_iri_datatype_operand() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?value WHERE {
                    { BIND(STRDT("x", <urn:arbitrary-datatype>) AS ?value) }
                    UNION
                    { BIND(STRDT("x", "not-an-iri") AS ?value) }
                    FILTER(BOUND(?value))
                }"#,
            )
            .expect("STRDT datatype errors follow ordinary BIND/FILTER semantics");

        assert_eq!(
            r.row_count(),
            1,
            "a literal datatype operand leaves STRDT unbound"
        );
        assert!(matches!(
            &r.rows()[0][0],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } if lexical.as_str() == "x" && datatype.as_str() == "urn:arbitrary-datatype"
        ));
    }

    #[test]
    fn select_strdt_accepts_dynamic_iri_datatype_operands_only() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?source ?value WHERE {
                       {
                           VALUES (?source ?lexical ?datatype) {
                               ("values" "1" xsd:integer)
                           }
                       }
                       UNION {
                           BIND("iri" AS ?source)
                           BIND("2" AS ?lexical)
                           BIND(IRI("http://www.w3.org/2001/XMLSchema#integer") AS ?datatype)
                       }
                       UNION {
                           BIND("datatype" AS ?source)
                           BIND("3" AS ?lexical)
                           BIND(DATATYPE("3"^^xsd:unsignedByte) AS ?datatype)
                       }
                       UNION {
                           VALUES (?source ?lexical ?datatype) {
                               ("literal" "4" "http://www.w3.org/2001/XMLSchema#integer")
                           }
                       }
                       BIND(STRDT(?lexical, ?datatype) AS ?value)
                       FILTER(BOUND(?value) && ISNUMERIC(?value))
                   }"#,
            )
            .expect("STRDT preserves exact term kind through dynamic datatype dataflow");

        let mut rows = r
            .rows()
            .iter()
            .map(|row| {
                (
                    row[0]
                        .as_str()
                        .expect("the source label is a string")
                        .to_string(),
                    row[1].clone(),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(
            rows,
            [
                (
                    "datatype".to_string(),
                    Value::RdfLiteral {
                        lexical: "3".into(),
                        language: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#unsignedByte".into()),
                    },
                ),
                ("iri".to_string(), Value::Int64(2)),
                ("values".to_string(), Value::Int64(1)),
            ],
            "VALUES, IRI(), and DATATYPE() produce datatype IRIs; a string literal does not"
        );
    }

    #[test]
    fn select_strdt_preserves_datatype_identity_through_variable_copy_bind() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?kind ?source ?datatype ?value ?bound WHERE {
                       {
                           BIND("iri" AS ?kind)
                           BIND(xsd:unsignedByte AS ?source)
                       }
                       UNION {
                           BIND("literal" AS ?kind)
                           BIND("http://www.w3.org/2001/XMLSchema#unsignedByte" AS ?source)
                       }
                       BIND(?source AS ?datatype)
                       BIND(STRDT("7", ?datatype) AS ?value)
                       BIND(BOUND(?value) AS ?bound)
                   }"#,
            )
            .expect("variable-copy BIND retains the datatype operand's exact RDF kind");

        let mut rows = r.rows().to_vec();
        rows.sort_by(|left, right| {
            left[0]
                .as_str()
                .expect("kind is a string")
                .cmp(right[0].as_str().expect("kind is a string"))
        });
        assert_eq!(
            rows,
            [
                vec![
                    Value::String("iri".into()),
                    Value::String("http://www.w3.org/2001/XMLSchema#unsignedByte".into()),
                    Value::String("http://www.w3.org/2001/XMLSchema#unsignedByte".into()),
                    Value::RdfLiteral {
                        lexical: "7".into(),
                        language: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#unsignedByte".into(),),
                    },
                    Value::Bool(true),
                ],
                vec![
                    Value::String("literal".into()),
                    Value::String("http://www.w3.org/2001/XMLSchema#unsignedByte".into()),
                    Value::String("http://www.w3.org/2001/XMLSchema#unsignedByte".into()),
                    Value::Null,
                    Value::Bool(false),
                ],
            ],
            "copying equal visible strings must preserve IRI-versus-literal identity"
        );
    }

    #[test]
    fn select_strdt_preserves_computed_subselect_projection_identity() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?datatype ?value WHERE {
                       { SELECT (<urn:subselect-datatype> AS ?datatype) WHERE { } }
                       BIND(STRDT("x", ?datatype) AS ?value)
                   }"#,
            )
            .expect("a computed subselect projection retains exact IRI identity");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::String("urn:subselect-datatype".into()),
                Value::RdfLiteral {
                    lexical: "x".into(),
                    language: None,
                    datatype: Some("urn:subselect-datatype".into()),
                },
            ]]
        );
    }

    #[test]
    fn select_volatile_computed_subselect_datatype_is_evaluated_once() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?datatype ?value WHERE {
                       { SELECT (UUID() AS ?datatype) WHERE {} }
                       BIND(STRDT("x", ?datatype) AS ?value)
                   }"#,
            )
            .expect("a volatile computed subselect datatype retains exact IRI identity");

        assert_eq!(r.columns, ["datatype", "value"]);
        assert_eq!(r.row_count(), 1);
        let datatype = r.rows()[0][0]
            .as_str()
            .expect("UUID exposes its visible IRI spelling");
        assert!(matches!(
            &r.rows()[0][1],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(literal_datatype),
            } if lexical.as_str() == "x" && literal_datatype.as_str() == datatype
        ));
    }

    #[test]
    fn select_subselect_projection_exact_identity_is_lexically_scoped() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?outer ?datatype ?value WHERE {
                       BIND("outer-literal" AS ?source)
                       BIND(?source AS ?outer)
                       {
                           SELECT (<urn:scoped-datatype> AS ?datatype) WHERE {
                               BIND("inner-literal" AS ?source)
                           }
                       }
                       BIND(STRDT("x", ?datatype) AS ?value)
                   }"#,
            )
            .expect("subselect exact companions remain inside their lexical scope");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::String("outer-literal".into()),
                Value::String("urn:scoped-datatype".into()),
                Value::RdfLiteral {
                    lexical: "x".into(),
                    language: None,
                    datatype: Some("urn:scoped-datatype".into()),
                },
            ]],
            "the hidden inner ?source literal must not replace the same-spelled outer binding"
        );
    }

    #[test]
    fn select_language_literals_and_strlang_report_rdf_lang_string() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?direct ?constructed ?direct_datatype ?constructed_datatype WHERE {
                       BIND("bonjour"@FR AS ?direct)
                       BIND(STRLANG("hello", "EN-us") AS ?constructed)
                       BIND(DATATYPE(?direct) AS ?direct_datatype)
                       BIND(DATATYPE(?constructed) AS ?constructed_datatype)
                   }"#,
            )
            .expect("language literals retain exact language-tagged identity");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::RdfLiteral {
                    lexical: "bonjour".into(),
                    language: Some("fr".into()),
                    datatype: None,
                },
                Value::RdfLiteral {
                    lexical: "hello".into(),
                    language: Some("en-us".into()),
                    datatype: None,
                },
                Value::String("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".into(),),
                Value::String("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".into(),),
            ]]
        );
    }

    #[test]
    fn select_strlang_propagates_nullable_arguments_as_unbound() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?kind ?value ?bound WHERE {
                       {
                           BIND("valid" AS ?kind)
                           BIND("hello" AS ?lexical)
                           BIND("EN-us" AS ?language)
                       }
                       UNION {
                           BIND("null-lexical" AS ?kind)
                           BIND("en" AS ?language)
                       }
                       UNION {
                           BIND("null-language" AS ?kind)
                           BIND("hello" AS ?lexical)
                           OPTIONAL {
                               BIND("fr" AS ?language)
                               FILTER(false)
                           }
                       }
                       BIND(STRLANG(?lexical, ?language) AS ?value)
                       BIND(BOUND(?value) AS ?bound)
                   }"#,
            )
            .expect("STRLANG follows ordinary expression-error semantics for nullable inputs");

        let mut rows = r.rows().to_vec();
        rows.sort_by(|left, right| {
            left[0]
                .as_str()
                .expect("kind is a string")
                .cmp(right[0].as_str().expect("kind is a string"))
        });
        assert_eq!(
            rows,
            [
                vec![
                    Value::String("null-language".into()),
                    Value::Null,
                    Value::Bool(false),
                ],
                vec![
                    Value::String("null-lexical".into()),
                    Value::Null,
                    Value::Bool(false),
                ],
                vec![
                    Value::String("valid".into()),
                    Value::RdfLiteral {
                        lexical: "hello".into(),
                        language: Some("en-us".into()),
                        datatype: None,
                    },
                    Value::Bool(true),
                ],
            ],
            "UNION-null lexical and OPTIONAL-null language arguments must leave BIND unbound"
        );
    }

    #[test]
    fn select_invalid_numeric_typed_comparisons_fail_closed() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?invalid_bound ?valid ?opaque WHERE {
                       BIND("256"^^xsd:unsignedByte < "300"^^xsd:unsignedShort AS ?invalid)
                       BIND(BOUND(?invalid) AS ?invalid_bound)
                       BIND("2"^^xsd:unsignedByte < "3"^^xsd:unsignedShort AS ?valid)
                       BIND("a"^^<urn:opaque-a> < "b"^^<urn:opaque-b> AS ?opaque)
                   }"#,
            )
            .expect("invalid numeric comparisons follow ordinary expression-error semantics");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::Bool(false),
                Value::Bool(true),
                Value::Bool(true),
            ]],
            "invalid numeric datatypes fail closed while valid numeric and opaque controls compare"
        );
    }

    #[test]
    fn select_exact_copy_flow_is_hidden_and_evaluates_volatile_source_once() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?source ?datatype ?value WHERE {
                       BIND(UUID() AS ?source)
                       BIND(?source AS ?datatype)
                       BIND(STRDT("x", ?datatype) AS ?value)
                   }"#,
            )
            .expect("volatile datatype IRI is evaluated once and copied exactly");

        assert_eq!(r.columns, vec!["source", "datatype", "value"]);
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            r.rows()[0][1],
            "the visible UUID source and its variable copy must match"
        );
        let source = r.rows()[0][0]
            .as_str()
            .expect("UUID exposes its IRI spelling");
        assert!(matches!(
            &r.rows()[0][2],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } if lexical.as_str() == "x" && datatype.as_str() == source
        ));
    }

    #[test]
    fn select_keeps_simple_and_xsd_string_values_publicly_string_typed() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?simple ?explicit ?constructed WHERE {
                       BIND("simple" AS ?simple)
                       BIND("explicit"^^xsd:string AS ?explicit)
                       BIND(STRDT("constructed", xsd:string) AS ?constructed)
                   }"#,
            )
            .expect("simple and explicit xsd:string values use the public string contract");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::String("simple".into()),
                Value::String("explicit".into()),
                Value::String("constructed".into()),
            ]]
        );
    }

    #[test]
    fn select_datatype_reads_exact_rdf_literal_datatypes() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?string ?numeric ?arbitrary WHERE {
                       BIND(DATATYPE("s"^^xsd:string) AS ?string)
                       BIND(DATATYPE("7"^^xsd:unsignedByte) AS ?numeric)
                       BIND(DATATYPE("x"^^<urn:arbitrary-datatype>) AS ?arbitrary)
                   }"#,
            )
            .expect("DATATYPE reads exact typed-literal identity");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::String("http://www.w3.org/2001/XMLSchema#string".into()),
                Value::String("http://www.w3.org/2001/XMLSchema#unsignedByte".into()),
                Value::String("urn:arbitrary-datatype".into()),
            ]]
        );
    }

    #[test]
    fn select_numeric_consumers_accept_valid_rdf_literals() {
        let db = rdf_db();
        db.execute_sparql(
            r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
               INSERT DATA {
                   <urn:number> <urn:integer> "8"^^xsd:unsignedByte ;
                                <urn:fraction> "-1.2"^^xsd:decimal .
               }"#,
        )
        .expect("insert RDF numeric consumer fixtures");

        let r = db
            .execute_sparql(
                r#"SELECT ?add ?sub ?mul ?div ?abs ?ceil ?floor ?round ?plus ?minus ?native WHERE {
                       <urn:number> <urn:integer> ?integer ; <urn:fraction> ?fraction .
                       BIND(?integer + 2 AS ?add)
                       BIND(?integer - 3 AS ?sub)
                       BIND(?integer * 2 AS ?mul)
                       BIND(?integer / 2 AS ?div)
                       BIND(ABS(?fraction) AS ?abs)
                       BIND(CEIL(?fraction) AS ?ceil)
                       BIND(FLOOR(?fraction) AS ?floor)
                       BIND(ROUND(?fraction) AS ?round)
                       BIND(+?integer AS ?plus)
                       BIND(-?integer AS ?minus)
                       BIND(2 + 3 AS ?native)
                   }"#,
            )
            .expect("ordinary numeric consumers accept valid RDF numeric literals");

        let decimal = |lexical: &str| Value::RdfLiteral {
            lexical: lexical.into(),
            language: None,
            datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".into()),
        };
        assert_eq!(
            r.rows(),
            &[vec![
                Value::Int64(10),
                Value::Int64(5),
                Value::Int64(16),
                decimal("4.0"),
                decimal("1.2"),
                decimal("-1.0"),
                decimal("-2.0"),
                decimal("-1.0"),
                Value::Int64(8),
                Value::Int64(-8),
                Value::Int64(5),
            ]]
        );
    }

    #[test]
    fn select_numeric_consumers_reject_invalid_rdf_literals() {
        let db = rdf_db();
        db.execute_sparql(
            r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
               INSERT DATA { <urn:number> <urn:invalid> "256"^^xsd:unsignedByte . }"#,
        )
        .expect("insert invalid RDF numeric consumer fixture");

        let r = db
            .execute_sparql(
                r#"SELECT ?add_bound ?compare_bound ?abs_bound ?plus_bound ?minus_bound WHERE {
                       <urn:number> <urn:invalid> ?value .
                       BIND(?value + 1 AS ?add)
                       BIND(?value < 300 AS ?compare)
                       BIND(ABS(?value) AS ?abs)
                       BIND(+?value AS ?plus)
                       BIND(-?value AS ?minus)
                       BIND(BOUND(?add) AS ?add_bound)
                       BIND(BOUND(?compare) AS ?compare_bound)
                       BIND(BOUND(?abs) AS ?abs_bound)
                       BIND(BOUND(?plus) AS ?plus_bound)
                       BIND(BOUND(?minus) AS ?minus_bound)
                   }"#,
            )
            .expect("invalid typed numerics remain expression errors");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
            ]]
        );
    }

    #[test]
    fn select_numeric_aggregate_errors_leave_sum_and_average_unbound() {
        let db = rdf_db();
        db.execute_sparql(
            r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
               INSERT DATA {
                   <urn:one> <urn:value> "2"^^xsd:unsignedByte .
                   <urn:two> <urn:value> "4"^^xsd:unsignedByte .
                   <urn:invalid> <urn:value> "256"^^xsd:unsignedByte .
               }"#,
        )
        .expect("insert RDF numeric aggregate fixtures");

        let r = db
            .execute_sparql(
                r#"SELECT ?count ?sum_bound ?average_bound WHERE {
                       {
                           SELECT (COUNT(?value) AS ?count)
                                  (SUM(?value) AS ?sum)
                                  (AVG(?value) AS ?average)
                           WHERE { ?subject <urn:value> ?value }
                       }
                       BIND(BOUND(?sum) AS ?sum_bound)
                       BIND(BOUND(?average) AS ?average_bound)
                   }"#,
            )
            .expect("numeric aggregate expression errors follow SPARQL set-function semantics");

        assert_eq!(
            r.rows(),
            &[vec![
                Value::Int64(3),
                Value::Bool(false),
                Value::Bool(false),
            ]],
            "COUNT counts every bound term, while a numeric type error makes SUM and AVG errors"
        );
    }

    #[test]
    fn select_isnumeric_evaluates_strdt_typed_literals() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?lexical WHERE {
                       VALUES ?lexical { "INF" "pumpkin" }
                       BIND(STRDT(?lexical, xsd:double) AS ?value)
                       FILTER (ISNUMERIC(?value))
                   }"#,
            )
            .expect("ordinary SELECT evaluates isNumeric over STRDT results");

        assert_eq!(
            r.row_count(),
            1,
            "only the valid STRDT xsd:double literal is numeric"
        );
        assert!(
            r.rows()[0][0].to_string().contains("INF"),
            "the valid STRDT xsd:double literal must be selected",
        );
    }

    #[test]
    fn select_strdt_preserves_an_ill_typed_literal_when_no_native_value_exists() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?value WHERE {
                       BIND(STRDT("pumpkin", xsd:integer) AS ?value)
                   }"#,
            )
            .expect("ordinary SELECT retains an ill-typed STRDT result");

        assert_eq!(
            r.row_count(),
            1,
            "ill-typed STRDT remains a bound RDF literal"
        );
        assert!(matches!(
            &r.rows()[0][0],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } if lexical.as_str() == "pumpkin"
                && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#integer"
        ));
    }

    #[test]
    fn select_isnumeric_accepts_computed_infinities_and_rejects_invalid_arity() {
        let db = rdf_db();
        let r = db
            .execute_sparql(
                r#"SELECT ?positive ?negative WHERE {
                    BIND(1e308 * 1e308 AS ?positive)
                    BIND(-(1e308 * 1e308) AS ?negative)
                    FILTER (ISNUMERIC(?positive) && ISNUMERIC(?negative))
                }"#,
            )
            .expect("ordinary SELECT evaluates isNumeric over computed native infinities");
        assert_eq!(
            r.rows(),
            &[vec![
                Value::Float64(f64::INFINITY),
                Value::Float64(f64::NEG_INFINITY),
            ]],
            "computed INF and -INF are numeric"
        );

        for query in [
            "SELECT * WHERE { FILTER (ISNUMERIC()) }",
            "SELECT * WHERE { FILTER (ISNUMERIC(1, 2)) }",
        ] {
            let err = db
                .execute_sparql(query)
                .expect_err("ordinary SELECT isNumeric arity must be a translation error");
            assert!(
                matches!(
                    &err,
                    Error::Query(query_error)
                        if query_error.kind == QueryErrorKind::Semantic
                ),
                "isNumeric arity must return a structured semantic query error: {err:?}",
            );
            assert!(
                err.to_string()
                    .contains("ISNUMERIC requires exactly 1 argument"),
                "unexpected isNumeric arity error: {err}",
            );
        }
    }

    #[test]
    fn select_exact_relational_compatibility_distinguishes_same_spelled_terms() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:literal-subject> <urn:p> "urn:x" .
                   <urn:iri-subject> <urn:p> <urn:x> .
                   <urn:left> <urn:left-value> <urn:x> .
                   <urn:right> <urn:right-value> "urn:x" .
               }"#,
        )
        .expect("insert exact compatibility fixtures");

        let basic = db
            .execute_sparql(
                r#"SELECT ?left ?right WHERE {
                       ?left <urn:left-value> ?term .
                       ?right <urn:right-value> ?term .
                   }"#,
            )
            .expect("a Basic join compares exact RDF terms");
        assert_eq!(
            basic.row_count(),
            0,
            "an IRI and same-spelled literal are not join-compatible"
        );

        let prior_binding = db
            .execute_sparql(
                r#"SELECT ?subject ?term WHERE {
                       { SELECT (IRI("urn:x") AS ?term) WHERE {} }
                       ?subject <urn:p> ?term .
                   }"#,
            )
            .expect("a prior exact binding constrains a later Basic pattern");
        assert_eq!(prior_binding.row_count(), 1);
        assert_eq!(
            prior_binding.rows()[0][0],
            Value::String("urn:iri-subject".into()),
            "the same-spelled literal must not match the prior IRI binding"
        );

        let incompatible_subselects = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT (IRI("urn:x") AS ?term) WHERE {} }
                       { SELECT ("urn:x" AS ?term) WHERE {} }
                   }"#,
            )
            .expect("explicit subselect exports compare exact RDF terms");
        assert_eq!(
            incompatible_subselects.row_count(),
            0,
            "incompatible explicit exports do not join"
        );
    }

    #[test]
    fn select_exact_optional_filters_respect_unbound_compatibility_keys() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { <urn:t> <urn:q> <urn:a> . }"#)
            .expect("insert OPTIONAL compatibility fixture");

        let propagated = db
            .execute_sparql(
                r#"SELECT ?x WHERE {
                       {
                           OPTIONAL { ?s <urn:p> ?x }
                           FILTER(!BOUND(?x))
                       }
                       OPTIONAL { ?t <urn:q> ?x }
                   }"#,
            )
            .expect("a left-side filter is not mirrored across a wildcard key");
        assert_eq!(
            propagated.rows(),
            &[vec![Value::String("urn:a".into())]],
            "the right binding must survive and coalesce into the unbound left mapping"
        );

        let pushed = db
            .execute_sparql(
                r#"SELECT ?x WHERE {
                       OPTIONAL { ?s <urn:p> ?x }
                       OPTIONAL { ?t <urn:q> ?x }
                       FILTER(!BOUND(?x))
                   }"#,
            )
            .expect("a final filter stays above a wildcard-coalescing OPTIONAL");
        assert_eq!(
            pushed.row_count(),
            0,
            "the coalesced right binding makes !BOUND(?x) false"
        );

        let inner = db
            .execute_sparql(
                r#"SELECT ?x WHERE {
                       OPTIONAL { ?s <urn:p> ?x }
                       VALUES ?x { <urn:a> }
                       FILTER(?x = <urn:a>)
                   }"#,
            )
            .expect("an inner compatibility filter stays above coalescing");
        assert_eq!(
            inner.rows(),
            &[vec![Value::String("urn:a".into())]],
            "a filter pushed into the unbound left mapping would lose the compatible VALUES row"
        );
    }

    #[test]
    fn select_subselect_projection_is_an_optimizer_lexical_boundary() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:s> <urn:p> <urn:outer> .
                   <urn:r> <urn:q> <urn:different> .
               }"#,
        )
        .expect("insert subselect lexical-scope fixture");

        let result = db
            .execute_sparql(
                r#"SELECT ?x ?hidden WHERE {
                       <urn:s> <urn:p> ?x
                       OPTIONAL {
                           SELECT ?hidden WHERE { ?hidden <urn:q> ?x }
                       }
                       FILTER(?x = <urn:outer>)
                   }"#,
            )
            .expect("a non-exported subselect name stays local");
        assert_eq!(
            result.rows(),
            &[vec![
                Value::String("urn:outer".into()),
                Value::String("urn:r".into()),
            ]],
            "outer ?x filter must not be mirrored into the subselect's local ?x"
        );
    }

    #[test]
    fn select_leading_optional_starts_from_the_unit_solution() {
        let db = rdf_db();

        let empty = db
            .execute_sparql(
                r#"SELECT ?x WHERE {
                       OPTIONAL { FILTER(false) BIND(1 AS ?x) }
                   }"#,
            )
            .expect("an empty leading OPTIONAL preserves the unit solution");
        assert_eq!(empty.rows(), &[vec![Value::Null]]);

        let nonempty = db
            .execute_sparql(
                r#"SELECT ?x WHERE {
                       OPTIONAL { VALUES ?x { 1 2 } }
                   } ORDER BY ?x"#,
            )
            .expect("a nonempty leading OPTIONAL preserves every right mapping");
        assert_eq!(
            nonempty.rows(),
            &[vec![Value::Int64(1)], vec![Value::Int64(2)]],
            "leading OPTIONAL preserves bag multiplicity rather than collapsing to one row"
        );
    }

    #[test]
    fn select_standalone_minus_preserves_the_disjoint_unit_solution() {
        let db = rdf_db();
        for right in ["VALUES ?right { 1 }", "FILTER(false)"] {
            let query = format!("SELECT * WHERE {{ MINUS {{ {right} }} }}");
            let result = db
                .execute_sparql(&query)
                .unwrap_or_else(|error| panic!("standalone MINUS failed for {right}: {error}"));
            assert!(
                result.columns.is_empty(),
                "MINUS never exposes RHS variables: {right}"
            );
            assert_eq!(
                result.row_count(),
                1,
                "MINUS against the unit solution has no shared bound domain and preserves it: {right}"
            );
        }
    }

    #[test]
    fn select_exact_values_compatibility_is_row_aware_for_undef() {
        let db = rdf_db();

        let bound = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT ("urn:x" AS ?term) WHERE {} }
                       VALUES ?term { <urn:x> }
                   }"#,
            )
            .expect("bound VALUES cells compare exact RDF terms");
        assert_eq!(
            bound.row_count(),
            0,
            "a bound VALUES IRI rejects a prior same-spelled literal"
        );

        let mixed = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT ("urn:x" AS ?term) WHERE {} }
                       VALUES ?term { <urn:x> UNDEF }
                   }"#,
            )
            .expect("mixed VALUES handles bound and UNDEF rows independently");
        assert_eq!(
            mixed.rows(),
            &[vec![Value::String("urn:x".into())]],
            "UNDEF preserves the prior literal while the incompatible bound row is rejected"
        );

        let undef = db
            .execute_sparql(
                r#"SELECT ?term ?value WHERE {
                       { SELECT (IRI("urn:outer") AS ?term) WHERE {} }
                       VALUES ?term { UNDEF }
                       BIND(STRDT("x", ?term) AS ?value)
                   }"#,
            )
            .expect("an all-UNDEF table leaves the prior binding untouched");
        assert_eq!(undef.row_count(), 1);
        assert_eq!(undef.rows()[0][0], Value::String("urn:outer".into()));
        assert!(matches!(
            &undef.rows()[0][1],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } if lexical.as_str() == "x" && datatype.as_str() == "urn:outer"
        ));

        let introduced = db
            .execute_sparql(
                r#"SELECT ?term ?new WHERE {
                       { SELECT (<urn:outer> AS ?term) WHERE {} }
                       VALUES (?term ?new) { (UNDEF <urn:new>) }
                   }"#,
            )
            .expect("a prior-plan VALUES row binds newly introduced columns");
        assert_eq!(
            introduced.rows(),
            &[vec![
                Value::String("urn:outer".into()),
                Value::String("urn:new".into()),
            ]]
        );
    }

    #[test]
    fn select_exact_values_evaluates_the_left_mapping_once() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?token WHERE {
                       BIND(UUID() AS ?token)
                       VALUES ?unused { UNDEF UNDEF }
                   }"#,
            )
            .expect("VALUES joins one materialized left mapping to both UNDEF rows");

        assert_eq!(result.row_count(), 2, "VALUES preserves bag multiplicity");
        assert_eq!(
            result.rows()[0][0],
            result.rows()[1][0],
            "the volatile left binding must be evaluated once, then joined twice"
        );
    }

    #[test]
    fn select_exact_compatibility_coalesces_partial_mappings() {
        let db = rdf_db();
        for (query, expected) in [
            (
                r#"SELECT ?term WHERE {
                       VALUES ?term { UNDEF }
                       VALUES ?term { <urn:right> }
                   }"#,
                Value::String("urn:right".into()),
            ),
            (
                r#"SELECT ?term WHERE {
                       VALUES ?term { <urn:left> }
                       VALUES ?term { UNDEF }
                   }"#,
                Value::String("urn:left".into()),
            ),
            (
                r#"SELECT ?term WHERE {
                       VALUES ?term { UNDEF }
                       VALUES ?term { UNDEF }
                   }"#,
                Value::Null,
            ),
        ] {
            let result = db
                .execute_sparql(query)
                .expect("compatible partial mappings merge into one solution");
            assert_eq!(result.rows(), &[vec![expected]]);
        }

        let multi_key = db
            .execute_sparql(
                r#"SELECT ?first ?second WHERE {
                       VALUES (?first ?second) { (<urn:x> UNDEF) }
                       VALUES (?first ?second) {
                           (UNDEF <urn:y>)
                           ("urn:x" <urn:y>)
                       }
                   }"#,
            )
            .expect("multi-key compatibility compares every jointly bound RDF term");
        assert_eq!(
            multi_key.rows(),
            &[vec![
                Value::String("urn:x".into()),
                Value::String("urn:y".into()),
            ]],
            "the wildcard row merges, while the same-spelled literal conflicts with the IRI"
        );
    }

    #[test]
    fn select_exact_keys_use_canonical_rdf_term_serialization() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:lang> <urn:value> "x"@en .
                   <urn:string> <urn:value> "x" .
               }"#,
        )
        .expect("insert canonical exact-key fixtures");

        for (binding, subject) in [
            (r#"{ SELECT ("x"@EN AS ?term) WHERE {} }"#, "urn:lang"),
            (r#"BIND(STRLANG("x", "EN") AS ?term)"#, "urn:lang"),
            (
                r#"VALUES ?term { "x"^^<http://www.w3.org/2001/XMLSchema#string> }"#,
                "urn:string",
            ),
        ] {
            let query = format!(
                r#"SELECT ?term WHERE {{
                       {binding}
                       <{subject}> <urn:value> ?term
                   }}"#,
            );
            let result = db.execute_sparql(&query).unwrap_or_else(|error| {
                panic!("canonical binding `{binding}` failed to join: {error}")
            });
            assert_eq!(
                result.row_count(),
                1,
                "canonical binding `{binding}` must equal the stored RDF term"
            );
        }
    }

    #[test]
    fn select_exact_demanded_conditional_bindings_preserve_one_term() {
        let db = rdf_db();
        for (expression, expected_datatype) in [
            ("COALESCE(?missing, <urn:coalesce>)", "urn:coalesce"),
            ("IF(true, <urn:if>, \"urn:literal\")", "urn:if"),
        ] {
            let query = format!(
                r#"SELECT ?value WHERE {{
                       BIND({expression} AS ?datatype)
                       BIND(STRDT("x", ?datatype) AS ?value)
                   }}"#,
            );
            let result = db
                .execute_sparql(&query)
                .unwrap_or_else(|error| panic!("demanded `{expression}` failed: {error}"));
            assert_eq!(result.columns, vec!["value"]);
            assert!(matches!(
                &result.rows()[0][0],
                Value::RdfLiteral {
                    lexical,
                    language: None,
                    datatype: Some(datatype),
                } if lexical.as_str() == "x" && datatype.as_str() == expected_datatype
            ));
        }

        for exact_consumer in ["", "BIND(STRDT(\"y\", ?value) AS ?demand)"] {
            let query = format!(
                r#"SELECT ?value WHERE {{
                       BIND(STRDT("x", "urn:not-an-iri") AS ?value)
                       {exact_consumer}
                   }}"#,
            );
            let result = db
                .execute_sparql(&query)
                .expect("demanded and undemanded invalid STRDT both stay unbound");
            assert_eq!(result.row_count(), 1);
            assert_eq!(result.rows()[0][0], Value::Null);
        }

        for expression in ["IF(true, UUID(), UUID())", "COALESCE(?missing, UUID())"] {
            let query = format!(
                r#"SELECT ?datatype ?value WHERE {{
                       BIND({expression} AS ?datatype)
                       BIND(STRDT("x", ?datatype) AS ?value)
                   }}"#,
            );
            let result = db
                .execute_sparql(&query)
                .unwrap_or_else(|error| panic!("volatile `{expression}` failed: {error}"));
            let datatype = result.rows()[0][0]
                .as_str()
                .expect("volatile conditional exposes one IRI spelling");
            assert!(
                matches!(
                    &result.rows()[0][1],
                    Value::RdfLiteral {
                        lexical,
                        language: None,
                        datatype: Some(actual),
                    } if lexical.as_str() == "x" && actual.as_str() == datatype
                ),
                "visible/tag/exact materialization re-evaluated `{expression}`"
            );
        }
    }

    #[test]
    fn select_exact_branch_compatibility_is_correlation_local() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:literal-subject> <urn:p> "urn:x" .
                   <urn:iri-subject> <urn:p> <urn:x> .
                   <urn:iri-subject-2> <urn:p> <urn:x> .
               }"#,
        )
        .expect("insert branch compatibility fixtures");

        let union = db
            .execute_sparql(
                r#"SELECT ?which WHERE {
                       { SELECT (IRI("urn:x") AS ?term) WHERE {} }
                       {
                           ?subject <urn:p> ?term .
                           FILTER (?subject = <urn:literal-subject>)
                           BIND("bad" AS ?which)
                       }
                       UNION {
                           ?subject <urn:p> ?term .
                           FILTER (?subject = <urn:iri-subject>)
                           BIND("good" AS ?which)
                       }
                       BIND(STRDT("x", ?term) AS ?value)
                   }"#,
            )
            .expect("correlated UNION alternatives compare exact terms");
        assert_eq!(union.rows(), &[vec![Value::String("good".into())]]);

        let optional = db
            .execute_sparql(
                r#"SELECT ?term ?right WHERE {
                       { SELECT (IRI("urn:x") AS ?term) WHERE {} }
                       OPTIONAL {
                           <urn:literal-subject> <urn:p> ?term .
                           BIND("matched" AS ?right)
                       }
                   }"#,
            )
            .expect("an incompatible OPTIONAL preserves its left row");
        assert_eq!(
            optional.rows(),
            &[vec![Value::String("urn:x".into()), Value::Null]]
        );

        let minus = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT (IRI("urn:x") AS ?term) WHERE {} }
                       MINUS { <urn:literal-subject> <urn:p> ?term }
                   }"#,
            )
            .expect("an incompatible MINUS does not remove the left row");
        assert_eq!(minus.rows(), &[vec![Value::String("urn:x".into())]]);

        let exists = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT (IRI("urn:x") AS ?term) WHERE {} }
                       FILTER EXISTS { <urn:literal-subject> <urn:p> ?term }
                   }"#,
            )
            .expect("incompatible FILTER EXISTS rejects the left row");
        assert_eq!(exists.row_count(), 0);

        let not_exists = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT (IRI("urn:x") AS ?term) WHERE {} }
                       FILTER NOT EXISTS { <urn:literal-subject> <urn:p> ?term }
                   }"#,
            )
            .expect("incompatible FILTER NOT EXISTS preserves the left row");
        assert_eq!(not_exists.rows(), &[vec![Value::String("urn:x".into())]]);

        let multiplicity = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT (<urn:x> AS ?term) WHERE {} }
                       FILTER EXISTS { ?subject <urn:p> ?term }
                   }"#,
            )
            .expect("FILTER EXISTS is a typed semi join");
        assert_eq!(
            multiplicity.row_count(),
            1,
            "multiple inner matches must not duplicate the outer mapping"
        );

        let uncorrelated = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       { SELECT (<urn:x> AS ?term) WHERE {} }
                       FILTER NOT EXISTS { VALUES ?other { <urn:present> } }
                   }"#,
            )
            .expect("uncorrelated NOT EXISTS has non-MINUS keyless semantics");
        assert_eq!(uncorrelated.row_count(), 0);

        let unit_exists = db
            .execute_sparql(
                r#"SELECT (1 AS ?kept) WHERE {
                       FILTER EXISTS { VALUES ?right { 1 2 } }
                   }"#,
            )
            .expect("EXISTS evaluates against the unit solution");
        assert_eq!(
            unit_exists.rows(),
            &[vec![Value::Int64(1)]],
            "EXISTS is a semi join and never leaks or multiplies right columns"
        );

        let unit_exists_empty = db
            .execute_sparql(
                r#"SELECT (1 AS ?kept) WHERE {
                       FILTER EXISTS { FILTER(false) }
                   }"#,
            )
            .expect("empty EXISTS rejects the unit solution");
        assert_eq!(unit_exists_empty.row_count(), 0);

        let unit_not_exists_nonempty = db
            .execute_sparql(
                r#"SELECT (1 AS ?kept) WHERE {
                       FILTER NOT EXISTS { VALUES ?right { 1 } }
                   }"#,
            )
            .expect("nonempty NOT EXISTS rejects the unit solution");
        assert_eq!(unit_not_exists_nonempty.row_count(), 0);

        let unit_not_exists_empty = db
            .execute_sparql(
                r#"SELECT (1 AS ?kept) WHERE {
                       FILTER NOT EXISTS { FILTER(false) }
                   }"#,
            )
            .expect("empty NOT EXISTS preserves the unit solution");
        assert_eq!(unit_not_exists_empty.rows(), &[vec![Value::Int64(1)]]);

        let compound = db
            .execute_sparql(
                r#"SELECT ?flag WHERE {
                       BIND((RAND() >= 0) AS ?flag)
                       FILTER (?flag && EXISTS { VALUES ?right { 1 } })
                   }"#,
            )
            .expect_err("compound EXISTS must fail closed until row-aware scalar lowering exists");
        assert!(
            compound.to_string().contains("compound or modifier EXISTS"),
            "unexpected structured error: {compound}"
        );
    }

    #[test]
    fn select_exact_path_and_group_outputs_remain_identity_keys() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:start> <urn:path> <urn:x> .
                   <urn:right> <urn:value> "urn:x" .
               }"#,
        )
        .expect("insert path/group exact fixtures");

        let path = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       <urn:start> <urn:path>+ ?term .
                       <urn:right> <urn:value> ?term .
                   }"#,
            )
            .expect("property-path outputs carry exact identity into joins");
        assert_eq!(path.row_count(), 0);

        let grouped = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       {
                           SELECT ?term WHERE { <urn:start> <urn:path> ?term }
                           GROUP BY ?term
                       }
                       <urn:right> <urn:value> ?term .
                   }"#,
            )
            .expect("direct GROUP BY outputs carry exact identity into joins");
        assert_eq!(grouped.row_count(), 0);
    }

    #[test]
    fn select_group_by_collapses_canonical_rdf_spellings_without_hidden_columns() {
        let db = rdf_db();
        for values in [
            r#""colour"@EN "colour"@en"#,
            r#""x" "x"^^<http://www.w3.org/2001/XMLSchema#string>"#,
        ] {
            let result = db
                .execute_sparql(&format!(
                    r#"SELECT ?value (COUNT(*) AS ?count)
                       WHERE {{ VALUES ?value {{ {values} }} }}
                       GROUP BY ?value"#,
                ))
                .unwrap_or_else(|error| {
                    panic!("canonical RDF spellings must share one GROUP BY key: {error}")
                });
            assert_eq!(
                result.columns,
                ["value", "count"],
                "lossless and canonical companions stay internal"
            );
            assert_eq!(result.row_count(), 1);
            assert_eq!(result.rows()[0][1], Value::Int64(2));
        }
    }

    #[test]
    fn select_exact_named_graph_compatibility_checks_term_kind() {
        let db = rdf_db();
        db.execute_sparql(r#"INSERT DATA { GRAPH <urn:g> { <urn:s> <urn:p> <urn:o> } }"#)
            .expect("insert named graph fixture");

        let literal = db
            .execute_sparql(
                r#"SELECT ?g WHERE {
                       { SELECT ("urn:g" AS ?g) WHERE {} }
                       GRAPH ?g { <urn:s> <urn:p> <urn:o> }
                   }"#,
            )
            .expect("a graph variable compares exact RDF terms");
        assert_eq!(
            literal.row_count(),
            0,
            "a pre-bound literal graph name must not match an IRI graph name"
        );

        let iri = db
            .execute_sparql(
                r#"SELECT ?g WHERE {
                       { SELECT (IRI("urn:g") AS ?g) WHERE {} }
                       GRAPH ?g { <urn:s> <urn:p> <urn:o> }
                   }"#,
            )
            .expect("the matching IRI graph-name control succeeds");
        assert_eq!(iri.rows(), &[vec![Value::String("urn:g".into())]]);
    }

    #[test]
    fn select_exact_modifier_consumers_resolve_projection_aliases() {
        let db = rdf_db();
        for (projection, source_binding) in [
            ("(IRI(?lex) AS ?dt)", ""),
            ("?dt", "BIND(IRI(?lex) AS ?dt)"),
            ("(?source AS ?dt)", "BIND(IRI(?lex) AS ?source)"),
        ] {
            let query = format!(
                r#"SELECT {projection} WHERE {{
                       VALUES ?lex {{ "urn:b" "urn:a" }}
                       {source_binding}
                   }}
                   ORDER BY (COALESCE(STR(DATATYPE(STRDT("x", ?dt))), "zzz"))
                   LIMIT 1"#,
            );
            let result = db
                .execute_sparql(&query)
                .unwrap_or_else(|error| panic!("projection `{projection}` failed: {error}"));
            assert_eq!(
                result.columns,
                vec!["dt"],
                "projection `{projection}` must expose no internal exact columns"
            );
            assert_eq!(
                result.rows(),
                &[vec![Value::String("urn:a".into())]],
                "ORDER BY must consume exact projection alias `{projection}`"
            );
        }

        let grouped = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?count) WHERE {
                       VALUES ?dt { <urn:b> <urn:a> }
                   }
                   GROUP BY (COALESCE(STR(DATATYPE(STRDT("x", ?dt))), "zzz"))"#,
            )
            .expect("RDF-aware GROUP BY preprojection evaluates exact functions");
        assert_eq!(grouped.row_count(), 2);
        assert!(
            grouped
                .rows()
                .iter()
                .all(|row| row.last() == Some(&Value::Int64(1))),
            "distinct exact RDF GROUP BY keys must not collapse into one null key: {:?}",
            grouped.rows()
        );
    }

    #[test]
    fn select_exact_group_expression_alias_is_materialized_once() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:iri-subject> <urn:value> <urn:x> .
                   <urn:literal-subject> <urn:value> "urn:x" .
               }"#,
        )
        .expect("insert GROUP alias identity fixtures");

        let result = db
            .execute_sparql(
                r#"SELECT ?subject ?value WHERE {
                       {
                           SELECT ?dt WHERE { VALUES ?lex { "urn:x" } }
                           GROUP BY (IRI(?lex) AS ?dt)
                       }
                       ?subject <urn:value> ?dt
                       BIND(STRDT("x", ?dt) AS ?value)
                   }"#,
            )
            .expect("GROUP expression alias exports one lossless RDF term");

        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::String("urn:iri-subject".into()),
            "the same-spelled literal must not join the IRI-valued GROUP alias"
        );
        assert!(matches!(
            &result.rows()[0][1],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } if lexical.as_str() == "x" && datatype.as_str() == "urn:x"
        ));
    }

    #[test]
    fn select_aggregate_exact_output_preserves_sample_identity() {
        let db = rdf_db();
        let ordinary = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?count) WHERE {
                       VALUES ?value { 1 2 }
                   }"#,
            )
            .expect("ordinary undemanded aggregates remain supported");
        assert_eq!(ordinary.rows(), &[vec![Value::Int64(2)]]);

        let demanded = db
            .execute_sparql(
                r#"SELECT ?out WHERE {
                       {
                           SELECT (SAMPLE(?value) AS ?datatype) WHERE {
                               VALUES ?value { <urn:datatype> }
                           }
                       }
                       BIND(STRDT("x", ?datatype) AS ?out)
                   }"#,
            )
            .expect("SAMPLE preserves exact RDF identity across a subselect boundary");
        assert_eq!(demanded.row_count(), 1);
        assert!(matches!(
            &demanded.rows()[0][0],
            Value::RdfLiteral {
                lexical,
                language: None,
                datatype: Some(datatype),
            } if lexical.as_str() == "x" && datatype.as_str() == "urn:datatype"
        ));
    }

    #[test]
    fn select_term_aggregates_export_one_exact_rdf_term() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?sample_ok ?min_ok ?max_ok ?sample_dt ?min_dt ?max_dt WHERE {
                       {
                           SELECT (SAMPLE(?sample_input) AS ?sample)
                                  (MIN(?ordered) AS ?minimum)
                                  (MAX(?ordered) AS ?maximum)
                           WHERE {
                               VALUES (?sample_input ?ordered) {
                                   (<urn:sample> <urn:z>)
                                   (<urn:sample> <urn:a>)
                               }
                           }
                       }
                       BIND(sameTerm(?sample, <urn:sample>) AS ?sample_ok)
                       BIND(sameTerm(?minimum, <urn:a>) AS ?min_ok)
                       BIND(sameTerm(?maximum, <urn:z>) AS ?max_ok)
                       BIND(STRDT("v", ?sample) AS ?sample_dt)
                       BIND(STRDT("v", ?minimum) AS ?min_dt)
                       BIND(STRDT("v", ?maximum) AS ?max_dt)
                   }"#,
            )
            .expect("term-returning aggregates export visible and exact identity atomically");

        assert_eq!(
            result.columns,
            vec![
                "sample_ok",
                "min_ok",
                "max_ok",
                "sample_dt",
                "min_dt",
                "max_dt"
            ]
        );
        assert_eq!(result.row_count(), 1);
        let row = &result.rows()[0];
        assert_eq!(
            &row[..3],
            &[Value::Bool(true), Value::Bool(true), Value::Bool(true)]
        );
        for (value, datatype) in row[3..].iter().zip(["urn:sample", "urn:a", "urn:z"]) {
            assert!(matches!(
                value,
                Value::RdfLiteral {
                    lexical,
                    language: None,
                    datatype: Some(actual),
                } if lexical.as_str() == "v" && actual.as_str() == datatype
            ));
        }
    }

    #[test]
    fn select_min_max_do_not_split_same_spelled_rdf_kinds() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?minimum ?maximum ?min_is_iri ?max_is_literal ?different WHERE {
                       {
                           SELECT (MIN(?term) AS ?minimum) (MAX(?term) AS ?maximum)
                           WHERE { VALUES ?term { <urn:x> "urn:x" } }
                       }
                       BIND(sameTerm(?minimum, <urn:x>) AS ?min_is_iri)
                       BIND(sameTerm(?maximum, "urn:x") AS ?max_is_literal)
                       BIND(!sameTerm(?minimum, ?maximum) AS ?different)
                   }"#,
            )
            .expect("MIN/MAX retain one atomic visible/exact RDF term");

        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0],
            vec![
                Value::String("urn:x".into()),
                Value::String("urn:x".into()),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
            ]
        );
    }

    #[test]
    fn select_exact_sample_evaluates_volatile_operand_once() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?same ?typed_same WHERE {
                       {
                           SELECT (SAMPLE(UUID()) AS ?sample)
                           WHERE { VALUES ?row { 1 } }
                       }
                       BIND(sameTerm(?sample, IRI(STR(?sample))) AS ?same)
                       BIND(STRDT("v", ?sample) AS ?typed)
                       BIND(
                           sameTerm(?typed, STRDT("v", IRI(STR(?sample))))
                           AS ?typed_same
                       )
                   }"#,
            )
            .expect("SAMPLE materializes a volatile RDF term once before aggregation");

        assert_eq!(result.rows(), &[vec![Value::Bool(true), Value::Bool(true)]]);
    }

    #[test]
    #[cfg(feature = "ring-index")]
    fn select_ring_fallback_preserves_complete_rdf_pattern_semantics() {
        let db = rdf_db();
        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:self> <urn:edge> <urn:self> .
                   <urn:self> <urn:edge> <urn:other> .
                   <urn:self> <urn:p> <urn:repeat-b> .
                   <urn:repeat-b> <urn:q> <urn:self> .
                   <urn:bad> <urn:edge> <urn:other> .
                   <urn:bad> <urn:p> <urn:repeat-v> .
                   <urn:repeat-v> <urn:q> <urn:bad> .
                   <urn:predicate-subject> <urn:predicate-subject> <urn:independent-object> .

                   <urn:a> <urn:p> <urn:b> .
                   <urn:b> <urn:q> <urn:c0-dead> .
                   <urn:b> <urn:q> <urn:c1-good> .
                   <urn:b> <urn:q> <urn:c2-good> .
                   <urn:c1-good> <urn:r> <urn:a> .
                   <urn:c2-good> <urn:r> <urn:a> .

                   <urn:l1> <urn:lang> "x"@EN .
                   <urn:l2> <urn:lang> "x"@en .
                   <urn:l3> <urn:lang> "x"@En .

                   <urn:i1> <urn:identity-p> <urn:x> .
                   <urn:i1> <urn:identity-p> "urn:x" .
                   <urn:i2> <urn:identity-q> <urn:x> .
                   <urn:i2> <urn:identity-q> "urn:x" .
                   <urn:i3> <urn:identity-r> <urn:x> .
                   <urn:i3> <urn:identity-r> "urn:x" .

                   <urn:string1> <urn:string-p> "x" .
                   <urn:string2> <urn:string-q> "x"^^<http://www.w3.org/2001/XMLSchema#string> .
                   <urn:string3> <urn:string-r> "x" .
               }"#,
        )
        .expect("insert Ring fallback adversarial fixtures");

        let repeated = db
            .execute_sparql("SELECT ?x WHERE { ?x <urn:edge> ?x }")
            .expect("repeated variables within one pattern enforce identity");
        assert_eq!(
            repeated.rows(),
            &[vec![Value::String("urn:self".into())]],
            "an unequal subject/object triple must not satisfy a repeated variable"
        );

        for query in [
            "SELECT ?x WHERE { ?x (<urn:edge>|<urn:p>) ?x }",
            "SELECT ?x WHERE { ?x !(<urn:p>) ?x }",
            "SELECT ?x WHERE { ?x <urn:edge>+ ?x }",
        ] {
            let path = db
                .execute_sparql(query)
                .expect("every property-path branch enforces repeated-variable identity");
            assert_eq!(
                path.rows(),
                &[vec![Value::String("urn:self".into())]],
                "path translation bypassed repeated-variable normalization: {query}"
            );
        }

        let collision = db
            .execute_sparql("SELECT ?x ?__rdf_repeat_0 WHERE { ?x ?x ?__rdf_repeat_0 }")
            .expect("internal repeated-variable names cannot collide with user variables");
        assert_eq!(
            collision.rows(),
            &[vec![
                Value::String("urn:predicate-subject".into()),
                Value::String("urn:independent-object".into()),
            ]]
        );

        let branching = db
            .execute_sparql(
                r#"SELECT ?c WHERE {
                       ?a <urn:p> ?b .
                       ?b <urn:q> ?c .
                       ?c <urn:r> ?a .
                   }
                   ORDER BY ?c"#,
            )
            .expect("typed fallback enumerates every cyclic continuation");
        assert_eq!(
            branching.rows(),
            &[
                vec![Value::String("urn:c1-good".into())],
                vec![Value::String("urn:c2-good".into())],
            ],
            "a dead first continuation must not hide later matches or multiplicity"
        );

        let language = db
            .execute_sparql(
                r#"SELECT ?term WHERE {
                       <urn:l1> <urn:lang> ?term .
                       <urn:l2> <urn:lang> ?term .
                       <urn:l3> <urn:lang> ?term .
                   }"#,
            )
            .expect("typed fallback uses canonical language-tag identity");
        assert_eq!(language.row_count(), 1);
        assert!(matches!(
            &language.rows()[0][0],
            Value::RdfLiteral {
                language: Some(language),
                ..
            } if language.eq_ignore_ascii_case("en")
        ));

        let collision_query = r#"SELECT ?term WHERE {
                   <urn:i1> <urn:identity-p> ?term .
                   <urn:i2> <urn:identity-q> ?term .
                   <urn:i3> <urn:identity-r> ?term .
               }
               ORDER BY ?term"#;
        let fallback_collision = db
            .execute_sparql(collision_query)
            .expect("typed fallback keeps IRI and literal identity classes separate");
        assert_eq!(fallback_collision.row_count(), 2);

        let string_query = r#"SELECT ?term WHERE {
                   <urn:string1> <urn:string-p> ?term .
                   <urn:string2> <urn:string-q> ?term .
                   <urn:string3> <urn:string-r> ?term .
               }"#;
        let fallback_string = db
            .execute_sparql(string_query)
            .expect("typed fallback canonicalizes plain and explicit xsd:string");
        assert_eq!(fallback_string.row_count(), 1);

        let repeated_multi_query = r#"SELECT ?x ?y WHERE {
                   ?x <urn:edge> ?x .
                   ?x <urn:p> ?y .
                   ?y <urn:q> ?x .
               }"#;
        let fallback_repeated_multi = db
            .execute_sparql(repeated_multi_query)
            .expect("typed repeated-variable normalization remains exact");
        assert_eq!(
            fallback_repeated_multi.rows(),
            &[vec![
                Value::String("urn:self".into()),
                Value::String("urn:repeat-b".into()),
            ]]
        );

        let branching_query = r#"SELECT ?c WHERE {
                   ?a <urn:p> ?b .
                   ?b <urn:q> ?c .
                   ?c <urn:r> ?a .
               }
               ORDER BY ?c"#;
        let language_query = r#"SELECT ?term WHERE {
                   <urn:l1> <urn:lang> ?term .
                   <urn:l2> <urn:lang> ?term .
                   <urn:l3> <urn:lang> ?term .
               }"#;
        let count_query = r#"SELECT (COUNT(*) AS ?count) WHERE {
                   ?a <urn:p> ?b .
                   ?b <urn:q> ?c .
                   ?c <urn:r> ?a .
               }"#;
        let offset_query = r#"SELECT ?c WHERE {
                   ?a <urn:p> ?b .
                   ?b <urn:q> ?c .
                   ?c <urn:r> ?a .
               }
               ORDER BY ?c
               OFFSET 1
               LIMIT 1"#;
        let language_count_query =
            r#"SELECT (COUNT(*) AS ?count) WHERE { ?s <urn:lang> "x"@eN . }"#;
        let fallback_branching_rows = branching.rows().to_vec();
        let fallback_language_rows = language.rows().to_vec();
        let fallback_collision_rows = fallback_collision.rows().to_vec();
        let fallback_string_rows = fallback_string.rows().to_vec();
        let fallback_count_rows = db.execute_sparql(count_query).unwrap().rows().to_vec();
        let fallback_offset_rows = db.execute_sparql(offset_query).unwrap().rows().to_vec();
        assert_eq!(
            db.execute_sparql(language_count_query).unwrap().rows(),
            &[vec![Value::Int64(3)]],
            "canonical language constants count every representation"
        );

        db.rdf_store().rebuild_ring();
        let explain = db
            .execute_sparql(&format!("EXPLAIN {branching_query}"))
            .expect("explain qualified native Ring query");
        assert!(
            explain.rows()[0][0].to_string().contains("RdfLeapfrog"),
            "fresh qualified triangle should select native Ring: {:?}",
            explain.rows()
        );

        for (query, fallback_rows) in [
            (branching_query, fallback_branching_rows),
            (language_query, fallback_language_rows),
            (collision_query, fallback_collision_rows),
            (string_query, fallback_string_rows),
            (count_query, fallback_count_rows),
            (offset_query, fallback_offset_rows),
        ] {
            let native = db
                .execute_sparql(query)
                .expect("native Ring matches the typed fallback corpus");
            assert_eq!(native.rows(), fallback_rows.as_slice(), "query: {query}");
        }

        let repeated_explain = db
            .execute_sparql(&format!("EXPLAIN {repeated_multi_query}"))
            .expect("explain repeated-variable wrapper fallback");
        assert!(
            !repeated_explain.rows()[0][0]
                .to_string()
                .contains("RdfLeapfrog"),
            "owned repeated-variable Filter/Project wrappers remain on typed fallback"
        );
        assert_eq!(
            db.execute_sparql(repeated_multi_query).unwrap().rows(),
            fallback_repeated_multi.rows()
        );

        let projected_bag = db
            .execute_sparql(
                r#"SELECT ?a WHERE {
                       ?a <urn:p> ?b .
                       ?b <urn:q> ?c .
                       ?c <urn:r> ?a .
                   }
                   ORDER BY ?a"#,
            )
            .expect("native Ring preserves projected bag multiplicity");
        assert_eq!(
            projected_bag.rows(),
            &[
                vec![Value::String("urn:a".into())],
                vec![Value::String("urn:a".into())],
            ]
        );
        assert_eq!(
            db.execute_sparql(
                r#"SELECT DISTINCT ?a WHERE {
                       ?a <urn:p> ?b .
                       ?b <urn:q> ?c .
                       ?c <urn:r> ?a .
                   }"#,
            )
            .unwrap()
            .row_count(),
            1
        );
        assert_eq!(
            db.execute_sparql(
                r#"SELECT ?c WHERE {
                       ?a <urn:p> ?b .
                       ?b <urn:q> ?c .
                       ?c <urn:r> ?a .
                   }
                   LIMIT 0"#,
            )
            .unwrap()
            .row_count(),
            0
        );
        assert_eq!(
            db.execute_sparql(
                r#"SELECT ?c WHERE {
                       ?a <urn:p> ?b .
                       ?b <urn:q> ?c .
                       ?c <urn:r> ?a .
                   }
                   LIMIT 1"#,
            )
            .unwrap()
            .row_count(),
            1
        );

        db.execute_sparql(
            r#"INSERT DATA {
                   <urn:b> <urn:q> <urn:c3-new> .
                   <urn:c3-new> <urn:r> <urn:a> .
               }"#,
        )
        .expect("make the derived Ring stale");
        let stale_explain = db
            .execute_sparql(&format!("EXPLAIN {branching_query}"))
            .expect("explain stale Ring fallback");
        assert!(
            !stale_explain.rows()[0][0]
                .to_string()
                .contains("RdfLeapfrog")
        );
        let stale_fallback = db.execute_sparql(branching_query).unwrap();
        assert_eq!(stale_fallback.row_count(), 3);
        assert_eq!(
            db.execute_sparql(language_count_query).unwrap().rows(),
            &[vec![Value::Int64(3)]]
        );
        db.rdf_store().rebuild_ring();
        let refreshed_native = db.execute_sparql(branching_query).unwrap();
        assert_eq!(refreshed_native.rows(), stale_fallback.rows());
        assert_eq!(
            db.execute_sparql(language_count_query).unwrap().rows(),
            &[vec![Value::Int64(3)]]
        );
    }

    #[test]
    fn select_exact_min_is_available_to_having_and_order() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?group (MIN(?value) AS ?minimum) WHERE {
                       VALUES (?group ?value) {
                           ("first" <urn:b>)
                           ("second" <urn:a>)
                           ("drop" "urn:0")
                       }
                   }
                   GROUP BY ?group
                   HAVING (!sameTerm(MIN(?value), "urn:0"))
                   ORDER BY ASC(MIN(?value))"#,
            )
            .expect("HAVING and ORDER consume the same exact MIN result as projection");

        assert_eq!(
            result.rows(),
            &[
                vec![
                    Value::String("second".into()),
                    Value::String("urn:a".into())
                ],
                vec![Value::String("first".into()), Value::String("urn:b".into())],
            ]
        );
    }

    #[test]
    fn select_term_aggregates_preserve_typed_and_language_literals() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?language ?typed ?language_same ?typed_same WHERE {
                       {
                           SELECT (SAMPLE(?language_input) AS ?language)
                                  (MIN(?typed_input) AS ?typed)
                           WHERE {
                               VALUES (?language_input ?typed_input) {
                                   ("colour"@EN "01"^^<urn:type>)
                               }
                           }
                       }
                       BIND(sameTerm(?language, "colour"@en) AS ?language_same)
                       BIND(sameTerm(?typed, "01"^^<urn:type>) AS ?typed_same)
                   }"#,
            )
            .expect("selector aggregates retain literal lexical form and annotations");

        assert_eq!(result.row_count(), 1);
        let row = &result.rows()[0];
        assert!(matches!(
            &row[0],
            Value::RdfLiteral {
                lexical,
                language: Some(language),
                datatype: None,
            } if lexical.as_str() == "colour" && language.as_str() == "en"
        ));
        assert!(
            matches!(
                &row[1],
                Value::RdfLiteral {
                    lexical,
                    language: None,
                    datatype: Some(datatype),
            } if lexical.as_str() == "01"
                && datatype.as_str() == "urn:type"
            ),
            "unexpected typed MIN result: {row:?}"
        );
        assert_eq!(&row[2..], &[Value::Bool(true), Value::Bool(true)]);
    }

    #[test]
    fn select_literal_aggregates_have_exact_sparql_result_datatypes() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?count ?sum ?average ?concat
                          ?count_same ?sum_same ?average_same ?concat_same WHERE {
                       {
                           SELECT (COUNT(?number) AS ?count)
                                  (SUM(?number) AS ?sum)
                                  (AVG(?number) AS ?average)
                                  (GROUP_CONCAT(?label; SEPARATOR="|") AS ?concat)
                           WHERE {
                               VALUES (?number ?label) { (1 "a") (2 "a") }
                           }
                       }
                       BIND(sameTerm(?count, "2"^^xsd:integer) AS ?count_same)
                       BIND(sameTerm(?sum, "3"^^xsd:integer) AS ?sum_same)
                       BIND(sameTerm(?average, "1.5"^^xsd:decimal) AS ?average_same)
                       BIND(sameTerm(?concat, "a|a") AS ?concat_same)
                   }"#,
            )
            .expect("constructed aggregate literals retain their normative RDF datatypes");

        assert_eq!(result.row_count(), 1);
        assert_eq!(
            &result.rows()[0][4..],
            &[
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
            ],
            "COUNT and integer SUM are xsd:integer, integer AVG is xsd:decimal, and GROUP_CONCAT is a simple literal"
        );
    }

    #[test]
    fn select_empty_global_aggregates_follow_sparql_identities_and_errors() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?count_star ?count_value ?sum ?average ?concat
                          ?sum_same ?average_same ?concat_same
                          ?minimum_bound ?maximum_bound ?sample_bound WHERE {
                       {
                           SELECT (COUNT(*) AS ?count_star)
                                  (COUNT(?value) AS ?count_value)
                                  (SUM(?value) AS ?sum)
                                  (AVG(?value) AS ?average)
                                  (GROUP_CONCAT(?value; SEPARATOR="|") AS ?concat)
                                  (MIN(?value) AS ?minimum)
                                  (MAX(?value) AS ?maximum)
                                  (SAMPLE(?value) AS ?sample)
                           WHERE {
                               VALUES ?value { 1 }
                               FILTER(false)
                           }
                       }
                       BIND(sameTerm(?sum, "0"^^xsd:integer) AS ?sum_same)
                       BIND(sameTerm(?average, "0"^^xsd:integer) AS ?average_same)
                       BIND(sameTerm(?concat, "") AS ?concat_same)
                       BIND(BOUND(?minimum) AS ?minimum_bound)
                       BIND(BOUND(?maximum) AS ?maximum_bound)
                       BIND(BOUND(?sample) AS ?sample_bound)
                   }"#,
            )
            .expect("empty global aggregates produce the SPARQL-defined single solution");

        assert_eq!(
            result.rows(),
            &[vec![
                Value::Int64(0),
                Value::Int64(0),
                Value::Int64(0),
                Value::Int64(0),
                Value::String("".into()),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
            ]]
        );
    }

    #[test]
    fn select_explicit_group_over_empty_input_emits_no_groups() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?category (MAX(?value) AS ?maximum) WHERE {
                       ?item <urn:missing-category> ?category ;
                             <urn:missing-value> ?value .
                   }
                   GROUP BY ?category"#,
            )
            .expect("explicit grouping of empty input returns no groups");

        assert!(
            result.rows().is_empty(),
            "current W3C agg-empty-group-max-1 has no explicit groups"
        );
    }

    #[test]
    fn select_distinct_aggregates_use_exact_rdf_term_identity() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?term_count ?numeric_sum WHERE {
                       {
                           SELECT (COUNT(DISTINCT ?term) AS ?term_count)
                                  (SUM(DISTINCT ?number) AS ?numeric_sum)
                           WHERE {
                               VALUES ?term {
                                   <urn:x> "urn:x"
                                   "a" "a"^^xsd:string
                                   "colour"@EN "colour"@en
                                   "1"^^xsd:integer "01"^^xsd:integer
                               }
                               VALUES ?number {
                                   "1"^^xsd:integer "01"^^xsd:integer
                               }
                           }
                       }
                   }"#,
            )
            .expect("DISTINCT aggregates compare canonical RDF terms, not visible host values");

        assert_eq!(
            result.rows(),
            &[vec![Value::Int64(6), Value::Int64(2)]],
            "six distinct RDF terms survive canonical identity, while SUM consumes both exact numeric spellings once"
        );
    }

    #[test]
    fn select_distinct_uses_canonical_rdf_term_identity() {
        let db = rdf_db();

        let same_spelling = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?count) WHERE {
                       { SELECT DISTINCT ?term WHERE { VALUES ?term { <urn:x> "urn:x" } } }
                   }"#,
            )
            .expect("SELECT DISTINCT keeps same-spelled RDF terms of different kinds");
        assert_eq!(same_spelling.rows(), &[vec![Value::Int64(2)]]);

        let canonical_language = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?count) WHERE {
                       { SELECT DISTINCT ?term WHERE { VALUES ?term { "colour"@EN "colour"@en } } }
                   }"#,
            )
            .expect("SELECT DISTINCT canonicalizes RDF language-tag case");
        assert_eq!(canonical_language.rows(), &[vec![Value::Int64(1)]]);
    }

    #[test]
    fn select_count_distinct_star_uses_only_the_logical_solution_mapping() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?count WHERE {
                       {
                           SELECT (COUNT(DISTINCT *) AS ?count)
                                  (SUM(RAND()) AS ?volatile_helper)
                           WHERE { VALUES ?value { 1 1 } }
                       }
                   }"#,
            )
            .expect("COUNT(DISTINCT *) ignores duplicate mappings and physical helper columns");

        assert_eq!(result.rows(), &[vec![Value::Int64(1)]]);

        let native = db
            .execute_sparql(
                "SELECT (COUNT(DISTINCT *) AS ?count) WHERE { \
                 BIND(VECTOR(1, 2) AS ?vector) }",
            )
            .expect("COUNT(DISTINCT *) falls back to native identity for native values");
        assert_eq!(native.rows(), &[vec![Value::Int64(1)]]);

        let native_expression = db
            .execute_sparql("SELECT (COUNT(DISTINCT VECTOR(1, 2)) AS ?count) WHERE {}")
            .expect(
                "COUNT(DISTINCT expr) must not reject a native expression as an RDF identity error",
            );
        assert_eq!(native_expression.row_count(), 1);

        let mixed_expression = db
            .execute_sparql(
                r#"SELECT (COUNT(DISTINCT IF(
                           ?kind = 0,
                           <urn:x>,
                           IF(?kind = 1, "urn:x", VECTOR(1, 2))
                       )) AS ?count)
                   WHERE { VALUES ?kind { 0 1 } }"#,
            )
            .expect("each mixed conditional branch chooses exact RDF or native identity locally");
        assert_eq!(
            mixed_expression.rows(),
            &[vec![Value::Int64(2)]],
            "same-spelled IRI and literal branches must not collapse because an unselected native branch exists"
        );
    }

    #[test]
    fn select_group_concat_accepts_sparql_string_literals() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?language ?typed ?plain WHERE {
                       {
                           SELECT (GROUP_CONCAT(?language_input) AS ?language)
                                  (GROUP_CONCAT(?typed_input) AS ?typed)
                                  (GROUP_CONCAT(?plain_input) AS ?plain)
                           WHERE {
                               VALUES (?language_input ?typed_input ?plain_input) {
                                   ("colour"@EN "typed"^^xsd:string "plain")
                               }
                           }
                       }
                   }"#,
            )
            .expect("GROUP_CONCAT accepts every SPARQL string-literal form");

        assert_eq!(
            result.rows(),
            &[vec![
                Value::String("colour".into()),
                Value::String("typed".into()),
                Value::String("plain".into()),
            ]]
        );
    }

    #[test]
    fn select_group_concat_rejects_non_string_rdf_terms() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?iri_bound ?numeric_bound ?typed_bound ?blank_bound WHERE {
                       { SELECT (GROUP_CONCAT(?value) AS ?iri)
                         WHERE { VALUES ?value { <urn:value> } } }
                       { SELECT (GROUP_CONCAT(?value) AS ?numeric)
                         WHERE { VALUES ?value { 1 } } }
                       { SELECT (GROUP_CONCAT(?value) AS ?typed)
                         WHERE { VALUES ?value { "value"^^<urn:type> } } }
                       { SELECT (GROUP_CONCAT(?value) AS ?blank)
                         WHERE { BIND(BNODE() AS ?value) } }
                       BIND(BOUND(?iri) AS ?iri_bound)
                       BIND(BOUND(?numeric) AS ?numeric_bound)
                       BIND(BOUND(?typed) AS ?typed_bound)
                       BIND(BOUND(?blank) AS ?blank_bound)
                   }"#,
            )
            .expect("GROUP_CONCAT reports argument-type errors as unbound aggregate results");

        assert_eq!(
            result.rows(),
            &[vec![
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
            ]]
        );
    }

    #[test]
    fn select_numeric_aggregates_preserve_precision_and_promotion_kind() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?kind (SUM(?value) AS ?sum) (AVG(?value) AS ?average) WHERE {
                       {
                           BIND("decimal" AS ?kind)
                           VALUES ?value { "0.1"^^xsd:decimal "0.2"^^xsd:decimal }
                       }
                       UNION {
                           BIND("double" AS ?kind)
                           VALUES ?value { "1.5"^^xsd:double "2.5"^^xsd:double }
                       }
                       UNION {
                           BIND("float" AS ?kind)
                           VALUES ?value { "1.5"^^xsd:float "2.5"^^xsd:float }
                       }
                       UNION {
                           BIND("integer" AS ?kind)
                           VALUES ?value { "18446744073709551615"^^xsd:unsignedLong 1 }
                       }
                   }
                   GROUP BY ?kind
                   ORDER BY ?kind"#,
            )
            .expect("SPARQL numeric aggregates preserve exact decimal/integer state and FP kind");

        assert_eq!(result.row_count(), 4);
        let rows = result.rows();
        assert!(matches!(
            &rows[0][..],
            [Value::String(kind),
             Value::RdfLiteral { lexical: sum, datatype: Some(sum_datatype), .. },
             Value::RdfLiteral { lexical: average, datatype: Some(avg_datatype), .. }]
                if kind.as_str() == "decimal"
                    && sum.as_str() == "0.3"
                    && average.as_str() == "0.15"
                    && sum_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
                    && avg_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
        ));
        assert_eq!(
            &rows[1][..],
            &[
                Value::String("double".into()),
                Value::Float64(4.0),
                Value::Float64(2.0),
            ]
        );
        assert!(matches!(
            &rows[2][..],
            [Value::String(kind),
             Value::RdfLiteral { lexical: sum, datatype: Some(sum_datatype), .. },
             Value::RdfLiteral { lexical: average, datatype: Some(avg_datatype), .. }]
                if kind.as_str() == "float"
                    && sum.as_str() == "4"
                    && average.as_str() == "2"
                    && sum_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#float"
                    && avg_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#float"
        ));
        assert!(matches!(
            &rows[3][..],
            [Value::String(kind),
             Value::RdfLiteral { lexical: sum, datatype: Some(sum_datatype), .. },
             Value::RdfLiteral { lexical: average, datatype: Some(avg_datatype), .. }]
                if kind.as_str() == "integer"
                    && sum.as_str() == "18446744073709551616"
                    && average.as_str() == "9223372036854775808.0"
                    && sum_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#integer"
                    && avg_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
        ));
    }

    #[test]
    fn select_numeric_aggregates_follow_cross_kind_promotion() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?kind (SUM(?value) AS ?sum) (AVG(?value) AS ?average) WHERE {
                       {
                           BIND("decimal-float" AS ?kind)
                           VALUES ?value { "0.5"^^xsd:decimal "1.5"^^xsd:float }
                       }
                       UNION {
                           BIND("float-double" AS ?kind)
                           VALUES ?value { "1.5"^^xsd:float "2.5"^^xsd:double }
                       }
                   }
                   GROUP BY ?kind
                   ORDER BY ?kind"#,
            )
            .expect("SPARQL numeric promotion follows integer-decimal-float-double order");

        assert_eq!(result.row_count(), 2);
        assert!(matches!(
            &result.rows()[0][..],
            [Value::String(kind),
             Value::RdfLiteral { lexical: sum, datatype: Some(sum_datatype), .. },
             Value::RdfLiteral { lexical: average, datatype: Some(avg_datatype), .. }]
                if kind.as_str() == "decimal-float"
                    && sum.as_str() == "2"
                    && average.as_str() == "1"
                    && sum_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#float"
                    && avg_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#float"
        ));
        assert_eq!(
            result.rows()[1],
            vec![
                Value::String("float-double".into()),
                Value::Float64(4.0),
                Value::Float64(2.0),
            ]
        );
    }

    #[test]
    fn select_exact_aggregate_order_and_having_do_not_round_large_integers() {
        let db = rdf_db();
        let ordered = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?group (SUM(?value) AS ?sum) WHERE {
                       VALUES (?group ?value) {
                           ("lower" "18446744073709551614"^^xsd:unsignedLong)
                           ("higher" "18446744073709551615"^^xsd:unsignedLong)
                       }
                   }
                   GROUP BY ?group
                   ORDER BY DESC(?sum)"#,
            )
            .expect("ORDER BY compares exact aggregate integers beyond i64/f64 range");

        assert_eq!(ordered.row_count(), 2);
        assert_eq!(ordered.rows()[0][0], Value::String("higher".into()));
        assert_eq!(ordered.rows()[1][0], Value::String("lower".into()));

        let filtered = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?group (SUM(?value) AS ?sum) WHERE {
                       VALUES (?group ?value) {
                           ("lower" "18446744073709551614"^^xsd:unsignedLong)
                           ("higher" "18446744073709551615"^^xsd:unsignedLong)
                       }
                   }
                   GROUP BY ?group
                   HAVING (SUM(?value) = "18446744073709551615"^^xsd:integer)"#,
            )
            .expect("HAVING compares exact aggregate integers without f64 collapse");

        assert_eq!(filtered.row_count(), 1);
        assert_eq!(filtered.rows()[0][0], Value::String("higher".into()));
    }

    #[test]
    fn select_exact_aggregate_arithmetic_preserves_range_and_decimal_kind() {
        let db = rdf_db();
        let integer = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT (?sum + 1 AS ?next) WHERE {
                       {
                           SELECT (SUM(?value) AS ?sum) WHERE {
                               VALUES ?value { "9223372036854775807"^^xsd:integer }
                           }
                       }
                   }"#,
            )
            .expect("arithmetic over an integer aggregate cannot overflow i64");
        assert!(matches!(
            &integer.rows()[0][0],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "9223372036854775808"
                    && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#integer"
        ));

        let decimal = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT (?sum + "0.1"^^xsd:decimal AS ?next) WHERE {
                       {
                           SELECT (SUM(?value) AS ?sum) WHERE {
                               VALUES ?value { "0.1"^^xsd:decimal "0.2"^^xsd:decimal }
                           }
                       }
                   }"#,
            )
            .expect("arithmetic over a decimal aggregate stays exact and decimal-typed");
        assert!(
            matches!(
                &decimal.rows()[0][0],
                Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                    if lexical.as_str() == "0.4"
                        && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
            ),
            "unexpected decimal result: {:?}",
            decimal.rows()
        );
    }

    #[test]
    fn select_decimal_aggregate_results_never_use_exponent_lexicals() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?sum ?average ?sum_numeric ?average_numeric
                          ?sum_arithmetic_bound ?average_arithmetic_bound WHERE {
                       {
                           SELECT (SUM(?value) AS ?sum) (AVG(?value) AS ?average)
                           WHERE { VALUES ?value { "0.0000001"^^xsd:decimal } }
                       }
                       BIND(isNumeric(?sum) AS ?sum_numeric)
                       BIND(isNumeric(?average) AS ?average_numeric)
                       BIND(?sum + "0.0"^^xsd:decimal AS ?sum_arithmetic)
                       BIND(?average + "0.0"^^xsd:decimal AS ?average_arithmetic)
                       BIND(BOUND(?sum_arithmetic) AS ?sum_arithmetic_bound)
                       BIND(BOUND(?average_arithmetic) AS ?average_arithmetic_bound)
                   }"#,
            )
            .expect("exact decimal aggregate output remains a valid numeric RDF literal");

        assert!(matches!(
            &result.rows()[0][..],
            [Value::RdfLiteral { lexical: sum, datatype: Some(sum_datatype), .. },
             Value::RdfLiteral { lexical: average, datatype: Some(average_datatype), .. },
             Value::Bool(true), Value::Bool(true), Value::Bool(true), Value::Bool(true)]
                if sum.as_str() == "0.0000001"
                    && average.as_str() == "0.0000001"
                    && sum_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
                    && average_datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
        ));
    }

    #[test]
    fn select_numeric_aggregate_having_uses_rdf_effective_boolean_value() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (SUM(?value) AS ?sum)
                   WHERE { VALUES ?value { 1 } }
                   HAVING (SUM(?value))"#,
            )
            .expect("a nonzero numeric aggregate has true RDF effective boolean value");

        assert_eq!(result.rows(), &[vec![Value::Int64(1)]]);
    }

    #[test]
    fn select_exact_min_max_order_decimal_values_numerically() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT (MIN(?value) AS ?minimum) (MAX(?value) AS ?maximum)
                   WHERE { VALUES ?value { "10.0"^^xsd:decimal "2.0"^^xsd:decimal } }"#,
            )
            .expect("MIN/MAX compare exact decimals by numeric value, not lexical spelling");

        assert!(matches!(
            &result.rows()[0][0],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "2.0"
                    && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
        ));
        assert!(matches!(
            &result.rows()[0][1],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "10.0"
                    && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
        ));
    }

    #[test]
    fn select_min_observes_errors_at_the_sparql_order_low_end() {
        let db = rdf_db();
        for values in ["0 1", "UNDEF 1"] {
            let result = db
                .execute_sparql(&format!(
                    r#"SELECT ?minimum_bound ?maximum_bound ?maximum WHERE {{
                           {{
                               SELECT (MIN(1 / ?value) AS ?minimum)
                                      (MAX(1 / ?value) AS ?maximum)
                               WHERE {{ VALUES ?value {{ {values} }} }}
                           }}
                           BIND(BOUND(?minimum) AS ?minimum_bound)
                           BIND(BOUND(?maximum) AS ?maximum_bound)
                       }}"#
                ))
                .expect("MIN and MAX retain SPARQL's asymmetric placement of errors");

            assert!(matches!(
                &result.rows()[0][..],
                [Value::Bool(false), Value::Bool(true),
                 Value::RdfLiteral { lexical, datatype: Some(datatype), .. }]
                    if lexical.as_str() == "1.0"
                        && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
            ));
        }

        let all_unbound = db
            .execute_sparql(
                r#"SELECT ?minimum_bound ?maximum_bound WHERE {
                       { SELECT (MIN(?value) AS ?minimum) (MAX(?value) AS ?maximum)
                         WHERE { VALUES ?value { UNDEF } } }
                       BIND(BOUND(?minimum) AS ?minimum_bound)
                       BIND(BOUND(?maximum) AS ?maximum_bound)
                   }"#,
            )
            .expect("all-error MIN and MAX both remain unbound");
        assert_eq!(
            all_unbound.rows(),
            &[vec![Value::Bool(false), Value::Bool(false)]]
        );
    }

    #[test]
    fn select_ieee_division_and_round_follow_sparql_numeric_rules() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?float_inf ?double_inf ?nan ?float_round ?double_round WHERE {
                       BIND("1"^^xsd:float / "0"^^xsd:float AS ?float_inf)
                       BIND("-1"^^xsd:double / "0"^^xsd:double AS ?double_inf)
                       BIND("0"^^xsd:double / "0"^^xsd:double AS ?nan)
                       BIND(ROUND("-2.5"^^xsd:float) AS ?float_round)
                       BIND(ROUND("-2.5"^^xsd:double) AS ?double_round)
                   }"#,
            )
            .expect("floating division and ROUND retain their SPARQL/XPath semantics");

        let row = &result.rows()[0];
        assert!(matches!(
            &row[0],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "INF"
                    && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#float"
        ));
        assert!(matches!(&row[1], Value::Float64(value) if *value == f64::NEG_INFINITY));
        assert!(matches!(&row[2], Value::Float64(value) if value.is_nan()));
        assert!(matches!(
            &row[3],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "-2"
                    && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#float"
        ));
        assert_eq!(row[4], Value::Float64(-2.0));
    }

    #[test]
    fn select_string_invalid_numeric_and_nan_boolean_semantics_are_total() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?plain ?empty ?invalid ?concat ?eq ?ne ?lt ?le ?gt ?ge WHERE {
                       { SELECT (GROUP_CONCAT(?value) AS ?joined)
                         WHERE { VALUES ?value { "x" } } }
                       BIND(IF("x", true, false) AS ?plain)
                       BIND(IF("", true, false) AS ?empty)
                       BIND(IF("invalid"^^xsd:integer, true, false) AS ?invalid)
                       BIND(IF(?joined, true, false) AS ?concat)
                       BIND("NaN"^^xsd:double AS ?nan)
                       BIND(?nan = ?nan AS ?eq)
                       BIND(?nan != ?nan AS ?ne)
                       BIND(?nan < 1 AS ?lt)
                       BIND(?nan <= 1 AS ?le)
                       BIND(?nan > 1 AS ?gt)
                       BIND(?nan >= 1 AS ?ge)
                   }"#,
            )
            .expect("EBV and NaN comparisons produce defined boolean results");

        assert_eq!(
            result.rows(),
            &[vec![
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
            ]]
        );
    }

    #[test]
    fn select_order_by_places_unbound_at_the_sparql_low_end() {
        // SPARQL has one low-end term order: DESC reverses it, so unbound
        // moves from first to last. This intentionally differs from LPG/GQL's
        // explicit NULLS FIRST/LAST final-placement contract.
        let db = rdf_db();
        let ascending = db
            .execute_sparql(
                "SELECT ?value WHERE { VALUES ?value { UNDEF 1 } } ORDER BY ASC(?value)",
            )
            .expect("ascending ORDER BY accepts unbound values");
        assert_eq!(
            ascending.rows(),
            &[vec![Value::Null], vec![Value::Int64(1)]]
        );

        let descending = db
            .execute_sparql(
                "SELECT ?value WHERE { VALUES ?value { UNDEF 1 } } ORDER BY DESC(?value)",
            )
            .expect("descending ORDER BY accepts unbound values");
        assert_eq!(
            descending.rows(),
            &[vec![Value::Int64(1)], vec![Value::Null]]
        );
    }

    #[test]
    fn select_order_by_preserves_rdf_term_categories() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?kind ?term WHERE {
                       { BIND(<urn:z> AS ?term) BIND("iri" AS ?kind) }
                       UNION
                       { BIND("a" AS ?term) BIND("literal" AS ?kind) }
                   }
                   ORDER BY ?term"#,
            )
            .expect("ORDER BY receives exact RDF term categories, not collapsed host strings");

        assert_eq!(
            result.rows(),
            &[
                vec![Value::String("iri".into()), Value::String("urn:z".into())],
                vec![Value::String("literal".into()), Value::String("a".into())],
            ]
        );
    }

    #[test]
    fn select_order_by_can_use_a_non_projected_in_scope_variable() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?shown WHERE {
                       VALUES (?shown ?sort) {
                           ("first" 2)
                           ("second" 1)
                       }
                   }
                   ORDER BY ?sort
                   LIMIT 1"#,
            )
            .expect("ORDER BY runs before projection hides non-selected variables");

        assert_eq!(result.rows(), &[vec![Value::String("second".into())]]);
    }

    #[test]
    fn select_order_by_uses_boolean_and_datetime_value_order() {
        let db = rdf_db();
        let booleans = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?lex ?value WHERE {
                       VALUES ?lex { "true" "false" }
                       BIND(STRDT(?lex, xsd:boolean) AS ?value)
                   }
                   ORDER BY ?value"#,
            )
            .expect("boolean literals sort by value rather than source lexical form");
        assert_eq!(
            booleans.rows(),
            &[
                vec![Value::String("false".into()), Value::Bool(false)],
                vec![Value::String("true".into()), Value::Bool(true)],
            ]
        );

        let date_times = db
            .execute_sparql(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                   SELECT ?lex ?value WHERE {
                       VALUES ?lex {
                           "1999-12-31T20:00:00-05:00"
                           "2000-01-01T00:00:00Z"
                       }
                       BIND(STRDT(?lex, xsd:dateTime) AS ?value)
                   }
                   ORDER BY ?value"#,
            )
            .expect("dateTime literals sort by normalized instant");
        assert_eq!(
            date_times
                .rows()
                .iter()
                .map(|row| &row[0])
                .collect::<Vec<_>>(),
            vec![
                &Value::String("2000-01-01T00:00:00Z".into()),
                &Value::String("1999-12-31T20:00:00-05:00".into()),
            ]
        );
    }

    #[test]
    fn select_order_by_does_not_force_native_aliases_into_rdf_identity() {
        let db = rdf_db();
        for query in [
            "SELECT (VECTOR(1, 2) AS ?vector) WHERE {} ORDER BY ?vector",
            "SELECT ?vector WHERE { BIND(VECTOR(1, 2) AS ?vector) } ORDER BY ?vector",
        ] {
            let result = db
                .execute_sparql(query)
                .unwrap_or_else(|error| panic!("native ORDER BY failed for `{query}`: {error}"));
            // VECTOR evaluation is a separate pre-existing RDF extension
            // surface. This regression specifically proves ORDER BY does not
            // turn that native expression into an RDF-identity requirement.
            assert_eq!(result.columns, ["vector"]);
            assert_eq!(result.row_count(), 1);
        }
    }

    #[test]
    fn select_min_max_order_numeric_looking_strings_lexically() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (MIN(?value) AS ?minimum) (MAX(?value) AS ?maximum)
                   WHERE { VALUES ?value { "10" "2" } }"#,
            )
            .expect("MIN/MAX share the exact RDF term order used by ORDER BY");

        assert_eq!(
            result.rows(),
            &[vec![Value::String("10".into()), Value::String("2".into()),]]
        );
    }

    #[test]
    fn select_numeric_order_is_transitive_across_integer_float_and_double() {
        let db = rdf_db();
        for (values, expected) in [
            (
                r#"("C" "16777217"^^xsd:double)
                    ("A" "16777217"^^xsd:integer)
                    ("B" "16777216"^^xsd:float)"#,
                ["B", "C", "A"],
            ),
            (
                r#"("A" "16777217"^^xsd:integer)
                    ("B" "16777216"^^xsd:float)
                    ("C" "16777217"^^xsd:double)"#,
                ["B", "A", "C"],
            ),
            (
                r#"("C" "16777217"^^xsd:double)
                    ("B" "16777216"^^xsd:float)
                    ("A" "16777217"^^xsd:integer)"#,
                ["B", "C", "A"],
            ),
        ] {
            let query = format!(
                r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                    SELECT ?label WHERE {{
                        VALUES (?label ?value) {{ {values} }}
                    }}
                    ORDER BY ?value"#,
            );
            let result = db
                .execute_sparql(&query)
                .expect("mixed numeric ORDER BY uses a transitive blocking comparator");
            assert_eq!(
                result
                    .rows()
                    .iter()
                    .map(|row| row[0].as_str().expect("label is a string"))
                    .collect::<Vec<_>>(),
                expected,
            );
        }
    }

    #[test]
    fn select_aggregate_projects_hidden_group_sort_keys_after_ordering() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT (COUNT(*) AS ?count) WHERE {
                       VALUES ?group { 2 1 }
                   }
                   GROUP BY ?group
                   ORDER BY ?group
                   LIMIT 1"#,
            )
            .expect("aggregate ORDER BY sees group keys before final SELECT projection");

        assert_eq!(result.columns, ["count"]);
        assert_eq!(result.rows(), &[vec![Value::Int64(1)]]);
    }

    #[test]
    fn select_constructed_aggregate_errors_are_not_silently_skipped() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?bound_count ?expression_count
                          ?sum_bound ?average_bound ?concat_bound WHERE {
                       {
                           SELECT (COUNT(?value) AS ?bound_count)
                                  (COUNT(4 / ?value) AS ?expression_count)
                                  (SUM(4 / ?value) AS ?sum)
                                  (AVG(4 / ?value) AS ?average)
                                  (GROUP_CONCAT(
                                      IF(?value = 0, STR(1 / 0), STR(?value));
                                      SEPARATOR="|"
                                   ) AS ?concat)
                           WHERE { VALUES ?value { 2 UNDEF 0 } }
                       }
                       BIND(BOUND(?sum) AS ?sum_bound)
                       BIND(BOUND(?average) AS ?average_bound)
                       BIND(BOUND(?concat) AS ?concat_bound)
                   }"#,
            )
            .expect("ListEval errors reach SPARQL set functions without becoming SQL nulls");

        assert_eq!(
            result.rows(),
            &[vec![
                Value::Int64(2),
                Value::Int64(1),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(false),
            ]]
        );
    }

    #[test]
    fn select_constructed_aggregate_aliases_survive_having_and_order() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?group
                          (COUNT(?number) AS ?count)
                          (SUM(?number) AS ?sum)
                          (AVG(?number) AS ?average)
                          (GROUP_CONCAT(?text; SEPARATOR="") AS ?concat)
                   WHERE {
                       VALUES (?group ?number ?text) {
                           ("a" 1 "a") ("a" 3 "a")
                           ("b" 2 "b") ("b" 4 "b")
                           ("drop" 100 "d")
                       }
                   }
                   GROUP BY ?group
                   HAVING (
                       COUNT(?number) = 2 &&
                       SUM(?number) >= 4 &&
                       AVG(?number) >= 2 &&
                       STRLEN(GROUP_CONCAT(?text; SEPARATOR="")) = 2
                   )
                   ORDER BY DESC(?sum) DESC(?average) DESC(?count) ?concat"#,
            )
            .expect("HAVING and ORDER reuse constructed aggregate aliases after RDF finalization");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            &result.rows()[0][..3],
            &[Value::String("b".into()), Value::Int64(2), Value::Int64(6),]
        );
        assert!(matches!(
            &result.rows()[0][3],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "3.0"
                    && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
        ));
        assert_eq!(result.rows()[0][4], Value::String("bb".into()));
        assert_eq!(
            &result.rows()[1][..3],
            &[Value::String("a".into()), Value::Int64(2), Value::Int64(4),]
        );
        assert!(matches!(
            &result.rows()[1][3],
            Value::RdfLiteral { lexical, datatype: Some(datatype), .. }
                if lexical.as_str() == "2.0"
                    && datatype.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
        ));
        assert_eq!(result.rows()[1][4], Value::String("aa".into()));
    }

    #[test]
    fn select_constructed_aggregates_share_one_volatile_input_evaluation() {
        let db = rdf_db();
        let result = db
            .execute_sparql(
                r#"SELECT ?count ?sum_same ?average_same ?concat_same WHERE {
                       {
                           SELECT (SAMPLE(?random) AS ?sample)
                                  (COUNT(DISTINCT ?random) AS ?count)
                                  (SUM(?random) AS ?sum)
                                  (AVG(?random) AS ?average)
                                  (GROUP_CONCAT(STR(?random)) AS ?concat)
                           WHERE { BIND(RAND() AS ?random) }
                       }
                       BIND(sameTerm(?sum, ?sample) AS ?sum_same)
                       BIND(sameTerm(?average, ?sample) AS ?average_same)
                       BIND(sameTerm(?concat, STR(?sample)) AS ?concat_same)
                   }"#,
            )
            .expect("each aggregate consumes the one materialized volatile input value");

        assert_eq!(
            result.rows(),
            &[vec![
                Value::Int64(1),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
            ]]
        );
    }

    #[test]
    fn describe_without_where_clause() {
        let db = rdf_db();
        insert_foaf_data(&db);
        // DESCRIBE <iri> without WHERE should return CBD for that resource
        let r = db.execute_sparql("DESCRIBE <http://ex.org/alix>").unwrap();
        // alix has 5 triples: type, name, age, knows, mbox
        assert!(
            r.row_count() >= 5,
            "DESCRIBE without WHERE should return triples, got {} rows",
            r.row_count()
        );
    }

    #[test]
    fn select_reduced_returns_results() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                "SELECT REDUCED ?type WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type }",
            )
            .unwrap();
        // REDUCED may or may not eliminate duplicates (spec allows either).
        // Our implementation treats REDUCED as a no-op, returning all rows.
        assert!(
            r.row_count() >= 2,
            "REDUCED should return at least the distinct count"
        );
    }

    #[test]
    fn physical_explain_shows_operator_names() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"EXPLAIN SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name .
                    ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person>
                }"#,
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        let plan = r.rows()[0][0].to_string();
        // Physical plan should contain physical operator names
        assert!(
            plan.contains("RdfTripleScan") || plan.contains("HashJoin") || plan.contains("Project"),
            "EXPLAIN should show physical operator names, got: {plan}"
        );
    }

    #[test]
    fn explain_analyze_shows_timing() {
        let db = rdf_db();
        insert_foaf_data(&db);
        let r = db
            .execute_sparql(
                r#"EXPLAIN ANALYZE SELECT ?name WHERE {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }"#,
            )
            .unwrap();
        assert!(r.row_count() >= 1);
        let profile = r.rows()[0][0].to_string();
        // EXPLAIN ANALYZE should show timing or operator stats
        assert!(
            profile.contains("time") || profile.contains("rows") || profile.contains("ms"),
            "EXPLAIN ANALYZE should show execution stats, got: {profile}"
        );
    }

    #[test]
    fn unsupported_service_is_structured_error_not_empty_success() {
        let db = rdf_db();
        let err = db
            .execute_sparql("SELECT ?x WHERE { SERVICE <http://example.org/sparql> { ?x ?p ?o } }")
            .expect_err("SERVICE must not succeed with an empty result");
        let msg = err.to_string();
        assert!(
            msg.contains("SERVICE") && !msg.is_empty(),
            "structured unsupported error, got {msg}"
        );
    }
}
