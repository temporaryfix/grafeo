//! Integration tests for the heap-based top-K rewrite added in
//! `query/planner/lpg/project.rs::try_heap_topk_rewrite`.
//!
//! These tests exercise end-to-end via `session.execute()`, asserting both
//! result correctness and (via EXPLAIN/PROFILE) that the rewrite fired or
//! fell through as expected.

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
