//! Integration tests for the heap-based top-K rewrite added in
//! `query/planner/lpg/project.rs::try_heap_topk_rewrite`.
//!
//! These tests exercise end-to-end via `session.execute()`, asserting result
//! correctness on the cases the rewrite fires for and the cases that should
//! fall through.
//!
//! Direct string-match verification that the heap rewrite fired isn't
//! possible from outside the engine: EXPLAIN walks the logical tree (which
//! the rewrite leaves unchanged — the fusion is physical-only, see the
//! `try_topk_rewrite` doc), and PROFILE gates the rewrite off so its output
//! never names TopK either. PROFILE is still useful for the *negative*
//! direction (test 18 confirms the gate works by asserting Sort + Limit
//! both run with timings); silent fall-through under non-PROFILE mode is
//! the regression class caught by the §2.9 e2e benchmark in `benches/topk_e2e.rs`.

#![cfg(feature = "lpg")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Inserts `n` `:Item` nodes with property `r` set to a deterministic
/// pseudo-random `Int64` derived from the index. Returns the DB.
fn seed_items(n: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for i in 0..n {
        let r = ((i as u64).wrapping_mul(2_654_435_761) % 1_000_000) as i64;
        session
            .execute(&format!("INSERT (:Item {{id: {i}, r: {r}}})"))
            .unwrap();
    }
    db
}

/// Returns the EXPLAIN plan text for `query` against `db`.
fn explain(db: &GrafeoDB, query: &str) -> String {
    let session = db.session();
    let result = session
        .execute(&format!("EXPLAIN {query}"))
        .expect("EXPLAIN should not fail");
    match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("EXPLAIN should return String, got {other:?}"),
    }
}

/// Returns the PROFILE plan text for `query` against `db`.
fn profile(db: &GrafeoDB, query: &str) -> String {
    let session = db.session();
    let result = session
        .execute(&format!("PROFILE {query}"))
        .expect("PROFILE should not panic");
    match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("PROFILE should return String, got {other:?}"),
    }
}

// Task 23
#[cfg(feature = "vector-index")]
#[test]
fn cypher_vector_topk_still_fires_first() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // Embeddings spread by angle so cosine similarity to [1,0,0] is monotone:
    // id=0 → [1,9,0] (mostly y-axis, low sim), id=9 → [10,0,0] (x-axis, sim=1).
    for i in 0..10 {
        let x = (i + 1) as i64;
        let y = (9 - i) as i64;
        session
            .execute(&format!(
                "INSERT (:Doc {{id: {i}, embedding: [{x}.0, {y}.0, 0.0]}})"
            ))
            .unwrap();
    }
    db.create_vector_index(
        "Doc",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();

    let result = session
        .execute(
            "MATCH (d:Doc) RETURN d.id \
             ORDER BY cosine_similarity(d.embedding, [1.0, 0.0, 0.0]) DESC LIMIT 3",
        )
        .unwrap();
    assert_eq!(result.row_count(), 3);

    // id=9 is closest to [1,0,0] in cosine similarity (embedding [10,0,0]).
    let top_id = match &result.rows()[0][0] {
        Value::Int64(i) => *i,
        other => panic!("expected Int64 id, got {other:?}"),
    };
    assert_eq!(top_id, 9, "id=9 has embedding closest to [1,0,0]");
}

// Task 22
#[test]
fn cypher_order_by_after_optional_match_uses_topk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // 10 :A nodes, half with an outgoing :R to a :B
    for i in 0..10 {
        session
            .execute(&format!("INSERT (:A {{id: {i}, r: {i}}})"))
            .unwrap();
        if i % 2 == 0 {
            session
                .execute(&format!(
                    "MATCH (a:A {{id: {i}}}) INSERT (a)-[:R]->(:B {{tag: {i}}})"
                ))
                .unwrap();
        }
    }

    let result = session
        .execute("MATCH (a:A) OPTIONAL MATCH (a)-[:R]->(b:B) RETURN a.id, b.tag ORDER BY a.r DESC LIMIT 3")
        .unwrap();
    assert_eq!(result.row_count(), 3);
    // Top 3 by a.r DESC: ids 9, 8, 7 (9 has no :B → b.tag is Null).
    assert_eq!(result.rows()[0][0], Value::Int64(9));
    assert_eq!(result.rows()[1][0], Value::Int64(8));
    assert_eq!(result.rows()[2][0], Value::Int64(7));
}

// Task 21
#[test]
fn cypher_order_by_with_filter_uses_topk() {
    let db = seed_items(100);
    let session = db.session();

    // WHERE filter sits below Sort; the rewrite plans sort.input as-is and
    // wraps with TopK, so the filter still pushes via the existing path.
    let result = session
        .execute("MATCH (n:Item) WHERE n.r > 100000 RETURN n.r ORDER BY n.r DESC LIMIT 5")
        .unwrap();
    assert!(result.row_count() <= 5);
    for row in result.rows() {
        if let Value::Int64(r) = &row[0] {
            assert!(*r > 100_000, "filter should be honoured: r={r}");
        } else {
            panic!("expected Int64, got {:?}", row[0]);
        }
    }
}

// Task 20
#[test]
fn cypher_order_by_aggregate_alias_falls_through() {
    let db = seed_items(20);
    // ORDER BY uses the aggregate alias `c`. plan_sort needs the augmenting
    // projection path; the heap rewrite must defer.
    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.id, count(*) AS c ORDER BY c DESC LIMIT 5")
        .unwrap();
    // 20 distinct ids → 20 groups of count 1; LIMIT picks 5.
    assert!(result.row_count() <= 5);
}

// Task 19
#[test]
fn cypher_skip_limit_falls_through() {
    let db = seed_items(20);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.r ORDER BY n.r DESC SKIP 5 LIMIT 5")
        .unwrap();
    assert_eq!(result.row_count(), 5);

    // The plan should be Limit ← Skip ← Sort (separate operators); the
    // rewrite only fires when limit.input is Sort directly.
    let plan = explain(&db, "MATCH (n:Item) RETURN n.r ORDER BY n.r DESC SKIP 5 LIMIT 5");
    assert!(
        plan.contains("Skip"),
        "Plan should contain Skip operator:\n{plan}"
    );
}

// Task 18
#[test]
fn cypher_order_by_limit_unfused_under_profile() {
    let db = seed_items(20);
    let plan = profile(&db, "MATCH (n:Item) RETURN n.r ORDER BY n.r DESC LIMIT 5");

    // PROFILE should not panic, and the unfused path should run — both
    // Sort and Limit operators visible.
    assert!(
        plan.contains("Sort"),
        "PROFILE under heap-rewrite-disabled should show Sort:\n{plan}"
    );
    assert!(
        plan.contains("Limit"),
        "PROFILE under heap-rewrite-disabled should show Limit:\n{plan}"
    );
    assert!(
        plan.contains("rows="),
        "PROFILE should report row counts:\n{plan}"
    );
    // Defensive: confirm the rewrite did NOT fire under PROFILE.
    assert!(
        !plan.contains("TopK"),
        "PROFILE must not show TopK — rewrite should be gated by !profiling:\n{plan}"
    );
}

// Task 17
#[test]
fn cypher_order_by_limit_uses_topk() {
    let db = seed_items(100);
    let session = db.session();

    let result = session
        .execute("MATCH (n:Item) RETURN n.r ORDER BY n.r DESC LIMIT 5")
        .unwrap();

    assert_eq!(result.row_count(), 5);

    // Returned values should be the 5 highest `r` in DESC order.
    // Compute expected by replaying the seed formula and sorting.
    let mut all: Vec<i64> = (0..100u64)
        .map(|i| (i.wrapping_mul(2_654_435_761) % 1_000_000) as i64)
        .collect();
    all.sort_unstable_by(|a, b| b.cmp(a));
    let expected_top5: Vec<Value> = all.iter().take(5).map(|&v| Value::Int64(v)).collect();
    let actual_top5: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(actual_top5, expected_top5);
}
