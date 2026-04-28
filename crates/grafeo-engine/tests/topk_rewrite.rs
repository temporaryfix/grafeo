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
#[allow(dead_code)] // used by tests added in subsequent commits
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

// Tests follow in subsequent commits (Tasks 17-23).
