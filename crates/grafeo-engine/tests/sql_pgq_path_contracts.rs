//! SQL/PGQ path bindings must exist where predicates consume them.
#![cfg(all(feature = "lpg", feature = "gql", feature = "sql-pgq"))]

mod support;

use grafeo_common::types::Value;
use grafeo_engine::database::QueryResult;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn path_rows(result: &QueryResult) -> Vec<(String, i64, String)> {
    let mut rows: Vec<_> = result
        .rows()
        .iter()
        .map(|row| {
            (
                row[0].as_str().unwrap().to_owned(),
                row[1].as_int64().unwrap(),
                row[2].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    rows.sort();
    rows
}

/// Enumerate at most two hops directly from the shared fixture's edge table.
/// Iterating both edge rows preserves parallel-edge multiplicity.
fn expected_rows(edge_type: &str, min: i64, max: i64) -> Vec<(String, i64, String)> {
    let mut rows = Vec::new();
    for &(source, middle, kind) in support::EDGES {
        if kind != edge_type {
            continue;
        }
        if min <= 1 {
            rows.push((source.to_owned(), 1, middle.to_owned()));
        }
        if max == 2 {
            for &(next_source, target, next_kind) in support::EDGES {
                if next_source == middle && next_kind == edge_type {
                    rows.push((source.to_owned(), 2, target.to_owned()));
                }
            }
        }
    }
    rows.sort();
    rows
}

#[test]
fn sql_pgq_anonymous_endpoints_are_distinct() -> TestResult {
    let db = support::adversarial_path_graph();
    assert_eq!(
        db.execute_sql(
            "SELECT COUNT(*) AS count FROM GRAPH_TABLE (
             MATCH (:Node)-[]->(:Node) COLUMNS (1 AS one))",
        )?
        .rows(),
        vec![vec![Value::Int64(14)]]
    );
    // Reserve the named target before assigning a name to the anonymous source.
    for pattern in [
        "(:Node)-[:OTHER]->(_anon_0:Node)",
        "(_anon_0:Node)-[:OTHER]->(:Node)",
    ] {
        assert_eq!(
            db.execute_sql(&format!(
                "SELECT COUNT(*) AS count FROM GRAPH_TABLE (
                 MATCH {pattern} COLUMNS (_anon_0.id AS id))"
            ))?
            .rows(),
            vec![vec![Value::Int64(2)]],
            "{pattern}"
        );
    }
    Ok(())
}

#[test]
fn sql_pgq_path_length_filters_wait_for_expand() -> TestResult {
    let db = support::adversarial_path_graph();
    for (kind, predicate, min, max) in [
        ("REL", "LENGTH(p) = 2", 2, 2),
        ("REL", "LENGTH(p) >= 1", 1, 2),
        ("OTHER", "LENGTH(p) = 1", 1, 1),
    ] {
        let result = db.execute_sql(&format!(
            "SELECT * FROM GRAPH_TABLE (
             MATCH (src:Node)-[p:{kind}*1..2]->(dst:Node)
             WHERE {predicate}
             COLUMNS (src.id AS source, LENGTH(p) AS distance, dst.id AS target))"
        ))?;
        assert_eq!(path_rows(&result), expected_rows(kind, min, max));
    }
    let outer_filter = db.execute_sql(
        "SELECT * FROM GRAPH_TABLE (
         MATCH (src:Node)-[p:OTHER*1..2]->(dst:Node)
         COLUMNS (src.id AS source, LENGTH(p) AS distance, dst.id AS target))
         WHERE distance = 1",
    )?;
    assert_eq!(path_rows(&outer_filter), expected_rows("OTHER", 1, 1));
    Ok(())
}

#[test]
fn sql_pgq_path_nodes_edges_filters_keep_parallel_identity() -> TestResult {
    let db = support::adversarial_path_graph();
    let result = db.execute_sql(
        "SELECT * FROM GRAPH_TABLE (
         MATCH (src:Node)-[p:OTHER*1..2]->(dst:Node)
         WHERE src.id = 's' AND dst.id = 'x'
           AND NODES(p) IS NOT NULL AND EDGES(p) IS NOT NULL
         COLUMNS (NODES(p) AS nodes, EDGES(p) AS edges))",
    )?;
    let rows = result.rows();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row[0].as_list().unwrap().len(), 2);
        assert_eq!(row[1].as_list().unwrap().len(), 1);
    }
    assert_eq!(rows[0][0], rows[1][0]);
    assert_ne!(rows[0][1], rows[1][1]);
    Ok(())
}

#[test]
fn sql_pgq_path_filter_preserves_reader_snapshot() -> TestResult {
    let db = support::adversarial_path_graph();
    let mut reader = db.session();
    reader.begin_transaction()?;
    let query = "SELECT * FROM GRAPH_TABLE (
                 MATCH (src:Node)-[p:OTHER*1..2]->(dst:Node)
                 WHERE LENGTH(p) = 1
                 COLUMNS (src.id AS source, LENGTH(p) AS distance, dst.id AS target))";
    let expected = expected_rows("OTHER", 1, 1);
    assert_eq!(path_rows(&reader.execute_sql(query)?), expected);
    db.execute("MATCH (:Node)-[r:OTHER]->(:Node) DELETE r")?;
    assert_eq!(path_rows(&reader.execute_sql(query)?), expected);
    assert!(db.execute_sql(query)?.rows().is_empty());
    reader.rollback()?;
    Ok(())
}
