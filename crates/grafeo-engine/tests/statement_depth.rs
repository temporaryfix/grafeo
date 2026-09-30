//! Statement depth limits: a long flat statement is rejected, never aborts.
//!
//! Translation, binding, optimization and planning recurse once per operator
//! and nested expression, so a long `CREATE` pattern list or `OR` chain once
//! overflowed a 2 MiB thread and aborted the process. Every test runs on a
//! 2 MiB thread. Statements at the limit must still run there: that is what
//! makes the limit safe, and a change that grows planner stack frames fails
//! these tests before it can abort a caller.

use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::plan_depth::MAX_PLAN_DEPTH;

/// The limit parsers put on one left-associative operator chain.
const MAX_CHAIN_LENGTH: usize = 512;

/// Size of a statement far past every limit.
const OVERSIZED: usize = 50_000;

fn on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(f)
        .expect("spawn test thread")
        .join()
        .expect("statement aborted the thread instead of returning")
}

fn create_patterns(count: usize) -> String {
    let patterns: Vec<String> = (0..count).map(|i| format!("(:P {{id: {i}}})")).collect();
    format!("CREATE {}", patterns.join(", "))
}

fn match_patterns(count: usize) -> String {
    let patterns: Vec<String> = (0..count).map(|i| format!("(a{i})")).collect();
    format!("MATCH {} RETURN 1", patterns.join(", "))
}

fn or_chain(terms: usize) -> String {
    format!(
        "MATCH (n) WHERE {}false RETURN n",
        "n.a = 1 OR ".repeat(terms - 1)
    )
}

fn assert_rejected(query: &str, result: grafeo_common::utils::error::Result<()>) {
    match result {
        Err(Error::Query(error))
            if matches!(
                error.kind,
                QueryErrorKind::Semantic | QueryErrorKind::Syntax
            ) && (error.message.contains("nests more than")
                || error.message.contains("chain longer than")
                || error.message.contains("chained operators")
                || error.message.contains("nesting depth")) => {}
        other => panic!(
            "{}… must be rejected by a depth limit, got {other:?}",
            &query[..query.len().min(60)]
        ),
    }
}

#[test]
fn statements_at_the_limits_run_on_a_small_stack() {
    on_small_stack(|| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // A CREATE list plans one operator per pattern plus a few above them.
        session
            .execute(&create_patterns(MAX_PLAN_DEPTH - 8))
            .expect("CREATE at the depth limit");
        // On an empty graph: over nodes, this many patterns is a cartesian product.
        GrafeoDB::new_in_memory()
            .session()
            .execute(&match_patterns(MAX_PLAN_DEPTH - 8))
            .expect("MATCH list at the depth limit");
        session
            .execute(&or_chain(MAX_CHAIN_LENGTH))
            .expect("OR chain at the chain limit");
    });
}

#[test]
fn statements_just_past_the_limits_are_rejected() {
    on_small_stack(|| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for query in [
            create_patterns(MAX_PLAN_DEPTH + 1),
            match_patterns(MAX_PLAN_DEPTH + 1),
            or_chain(MAX_CHAIN_LENGTH + 2),
        ] {
            assert_rejected(&query, session.execute(&query).map(drop));
        }
    });
}

#[test]
fn oversized_gql_statements_are_rejected_not_aborted() {
    on_small_stack(|| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for query in [
            create_patterns(OVERSIZED),
            match_patterns(OVERSIZED),
            or_chain(OVERSIZED),
            format!("RETURN {}1 AS x", "1 + ".repeat(OVERSIZED)),
            format!("RETURN {}true AS x", "NOT ".repeat(OVERSIZED)),
            format!("RETURN {}1 AS x", "- ".repeat(OVERSIZED)),
            format!("MATCH (a) {} RETURN a", "WITH a ".repeat(OVERSIZED)),
            // Chains nested in parentheses: every chain is within its own cap.
            (0..127).fold(String::from("1"), |inner, _| {
                format!("({inner}{})", " + 1".repeat(MAX_CHAIN_LENGTH - 1))
            }),
        ] {
            let query = if query.starts_with('(') {
                format!("RETURN {query} AS x")
            } else {
                query
            };
            assert_rejected(&query, session.execute(&query).map(drop));
        }
    });
}

#[cfg(feature = "cypher")]
#[test]
fn oversized_cypher_statements_are_rejected_not_aborted() {
    on_small_stack(|| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for query in [
            create_patterns(OVERSIZED),
            or_chain(OVERSIZED),
            format!("RETURN {}1 AS x", "2 ^ ".repeat(OVERSIZED)),
            format!("MATCH (a) {} RETURN a", "WITH a ".repeat(OVERSIZED)),
        ] {
            assert_rejected(&query, session.execute_cypher(&query).map(drop));
        }
    });
}

#[cfg(feature = "sql-pgq")]
#[test]
fn oversized_sql_pgq_statements_are_rejected_not_aborted() {
    on_small_stack(|| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for query in [
            format!(
                "SELECT id FROM GRAPH_TABLE (MATCH (n) WHERE {}false COLUMNS (n.id AS id))",
                "n.a = 1 OR ".repeat(OVERSIZED)
            ),
            format!(
                "SELECT {}1 AS x FROM GRAPH_TABLE (MATCH (n) COLUMNS (n.id AS id))",
                "NOT ".repeat(OVERSIZED)
            ),
        ] {
            assert_rejected(&query, session.execute_sql(&query).map(drop));
        }
    });
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn oversized_sparql_statements_are_rejected_not_aborted() {
    use grafeo_engine::Config;
    use grafeo_engine::config::GraphModel;

    on_small_stack(|| {
        // This is a depth/stack control, including a large valid bulk statement.
        // Deadline enforcement is covered separately by the timeout controls.
        let db = GrafeoDB::with_config(
            Config::in_memory()
                .with_graph_model(GraphModel::Both)
                .without_query_timeout(),
        )
        .expect("RDF database");
        let session = db.session();
        let triples: Vec<String> = (0..OVERSIZED)
            .map(|i| format!("?s{i} <http://p> ?s{} .", i + 1))
            .collect();
        let groups: Vec<String> = (0..OVERSIZED)
            .map(|i| format!("{{ ?s{i} <http://p> ?o }}"))
            .collect();
        for query in [
            format!("SELECT ?s0 WHERE {{ {} }}", triples.join(" ")),
            format!("SELECT ?s0 WHERE {{ {} }}", groups.join(" ")),
            format!(
                "SELECT ?s WHERE {{ ?s ?p ?o FILTER({}false) }}",
                "?o = 1 || ".repeat(OVERSIZED)
            ),
        ] {
            assert_rejected(&query, session.execute_sparql(&query).map(drop));
        }
        // Bulk data is not a join chain and stays unlimited.
        let data: Vec<String> = (0..OVERSIZED)
            .map(|i| format!("<http://s{i}> <http://p> {i} ."))
            .collect();
        session
            .execute_sparql(&format!("INSERT DATA {{ {} }}", data.join(" ")))
            .expect("bulk INSERT DATA");
    });
}
