//! Connected engine maintenance controls; bounded transaction batches reach
//! the actual compact table limit without exhausting one tiered epoch arena.

use super::GrafeoDB;
use crate::Config;
use grafeo_common::types::{NodeId, Value};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn rejected_first_compact_retains_native_source_and_running_timer_then_retries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed-first-compact.grafeo");
    let mut db = GrafeoDB::with_config(
        Config::persistent(&path).with_checkpoint_interval(Duration::from_hours(1)),
    )
    .unwrap();
    let native = Arc::clone(db.store_arc());
    let timer_thread = super::checkpoint_timer::tests::running_timer_thread(
        db.checkpoint_timer.lock().as_ref().unwrap(),
    );
    // IDs encode 15-bit table numbers: 32768 groups are representable, and
    // the next distinct label group must reject actual fold preparation.
    let mut excess = NodeId::INVALID;
    for start in (0..=32_768).step_by(4096) {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        for index in start..(start + 4096).min(32_769) {
            excess = session.create_node(&[&format!("Table{index}")]);
            assert!(excess.is_valid(), "seed table {index}");
        }
        session.commit().unwrap();
    }
    let epoch = db.current_epoch();
    let next_id = native.next_node_id();
    let error = db
        .compact()
        .expect_err("32769 live tables exceed compact IDs");
    assert!(error.to_string().contains("32769"), "{error}");
    assert!(error.to_string().contains("table"), "{error}");
    assert!(db.layered_store.is_none());
    assert!(db.external_read_store.is_none());
    assert!(db.external_write_store.is_none());
    assert!(Arc::ptr_eq(db.store_arc(), &native));
    assert_eq!(native.next_node_id(), next_id);
    assert_eq!(db.current_epoch(), epoch);
    assert_eq!(db.node_count(), 32_769);
    assert_eq!(
        super::checkpoint_timer::tests::running_timer_thread(
            db.checkpoint_timer.lock().as_ref().unwrap(),
        ),
        timer_thread,
        "failed preparation must neither stop nor replace the native timer"
    );

    // A committed deletion removes the excess *current* table. Its retained
    // structural history remains a sidecar and does not need a packed table.
    assert!(db.delete_node(excess));
    let first = NodeId::new(0);
    db.set_node_property(first, "after_failure", Value::from("still writable"))
        .unwrap();
    let history = db.get_node_property_history(first, "after_failure");
    db.compact()
        .expect("rejected source remains eligible for initial conversion");
    assert!(db.layered_store.is_some());
    assert!(db.checkpoint_timer.lock().is_none());
    assert_eq!(db.node_count(), 32_768);
    assert!(db.get_node(excess).is_none());
    assert_eq!(
        db.get_node_property_history(first, "after_failure"),
        history
    );
    db.set_node_property(first, "after_failure", Value::from("layered writable"))
        .unwrap();
    assert_eq!(
        db.get_node_property_at_epoch(first, "after_failure", db.current_epoch()),
        Some(Value::from("layered writable"))
    );
}
