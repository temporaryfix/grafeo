//! Regression: `scrub_at_epoch(EpochId::PENDING)` (the natural "latest" sentinel)
//! must return current values, not blank everything.
//!
//! Temporal columns use a half-open validity interval `[from, to)`; a current
//! (open) value has `to == PENDING == u64::MAX`, so a naive `contains(PENDING)`
//! is `MAX < MAX` = false and every current value reads as `None`. PENDING must
//! be treated as "latest" in the as-of lookup (mirroring `get_node_at_epoch`).

#![cfg(all(feature = "compact-store", feature = "lpg"))]

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;

fn non_null(scrub: &grafeo_core::graph::compact::GraphScrub) -> usize {
    scrub
        .nodes
        .iter()
        .map(|f| {
            f.columns
                .values()
                .map(|c| c.iter().filter(|v| v.is_some()).count())
                .sum::<usize>()
        })
        .sum()
}

#[test]
fn scrub_at_pending_returns_current_values() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "v", Value::Int64(7))
        .expect("set node property");
    db.set_node_property(a, "w", Value::Int64(9))
        .expect("set node property");
    db.compact().expect("compact");

    let at_current = non_null(&db.scrub_at_epoch(db.current_epoch()));
    let at_pending = non_null(&db.scrub_at_epoch(EpochId::PENDING));

    assert!(at_current >= 2, "sanity: current scrub sees the two values");
    assert_eq!(
        at_pending, at_current,
        "scrub_at_epoch(PENDING) must equal the current scrub, not blank values"
    );
}
