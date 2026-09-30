//! Executable Rust examples from the repeatable compact-store guide.
//! Keep each function body identical to its corresponding guide example.

#![cfg(all(feature = "lpg", feature = "gql", feature = "compact-store"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::{Config, GrafeoDB};

#[test]
fn guide_compact_write_compact_preserves_recorded_history() -> Result<()> {
    let mut db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    let (id, recorded) = {
        let mut session = db.session();
        session.begin_transaction()?;
        let id = session.create_node_with_props(
            &["Person", "Researcher"],
            [("name", Value::from("Alix")), ("age", Value::Int64(30))],
        )?;
        let recorded = session.commit()?;
        (id, recorded)
    }; // Drop the Session before maintenance.

    db.compact()?;
    db.set_node_property(id, "age", Value::Int64(31))?;
    db.compact()?;

    assert_eq!(
        db.get_node_property_at_epoch(id, "age", recorded),
        Some(Value::Int64(30)),
    );
    assert_eq!(
        db.get_node_property_at_epoch(id, "age", db.current_epoch()),
        Some(Value::Int64(31)),
    );
    Ok(())
}

#[test]
fn guide_compact_if_needed_uses_strict_native_and_overlay_threshold() -> Result<()> {
    let config = Config::in_memory().with_compaction_overlay_threshold(1);
    let mut db = GrafeoDB::with_config(config)?;

    let first = db
        .session()
        .create_node_with_props(&["Item"], [("value", Value::Int64(1))])?;
    assert!(!db.compact_if_needed()?); // One node equals the threshold.

    let second = db
        .session()
        .create_node_with_props(&["Item"], [("value", Value::Int64(2))])?;
    assert!(db.compact_if_needed()?); // Two native nodes: first conversion.

    db.set_node_property(first, "value", Value::Int64(3))?;
    assert!(!db.compact_if_needed()?); // One promoted node in the overlay.
    db.set_node_property(second, "value", Value::Int64(4))?;
    assert!(db.compact_if_needed()?); // Fold the two overlay nodes.
    Ok(())
}

#[test]
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn compact_preserves_explicit_native_read_only_admission() -> Result<()> {
    use grafeo_common::utils::error::{Error, TransactionError};
    use grafeo_engine::AccessMode;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("read-only-compaction.grafeo");
    GrafeoDB::open(&path)?.close()?;
    let before = std::fs::read(&path)?;
    let config = Config::persistent(&path).with_access_mode(AccessMode::ReadOnly);
    let mut db = GrafeoDB::with_config(config)?;
    let epoch = db.current_epoch();
    assert!(db.is_read_only());
    assert!(matches!(
        db.compact(),
        Err(Error::Transaction(TransactionError::ReadOnly))
    ));
    assert!(db.is_read_only());
    assert!(db.layered_store().is_none());
    assert_eq!(db.current_epoch(), epoch);
    assert_eq!((db.node_count(), db.edge_count()), (0, 0));
    assert!(matches!(
        db.session()
            .create_node_with_props(&["Rejected"], [("value", Value::Int64(1))]),
        Err(Error::Transaction(TransactionError::ReadOnly))
    ));
    assert_eq!(db.node_count(), 0);
    db.close()?;
    assert_eq!(std::fs::read(&path)?, before);
    Ok(())
}

#[test]
fn explicit_external_snapshot_conversion_is_repeatable_and_writable() -> Result<()> {
    use std::sync::Arc;

    use grafeo_core::graph::compact::CompactStoreBuilder;
    use grafeo_core::graph::traits::{GraphStore, GraphStoreSearch};

    let source = Arc::new(
        CompactStoreBuilder::new()
            .node_table("Item", |table| table.column_bitpacked("value", &[1], 1))
            .build()
            .expect("build external current-state snapshot"),
    );
    let id = source.node_ids()[0];
    let read_store = Arc::clone(&source) as Arc<dyn GraphStoreSearch>;
    let config = Config::in_memory().with_compaction_overlay_threshold(0);
    let mut db = GrafeoDB::with_read_store(read_store, config)?;
    assert!(db.is_read_only());
    assert!(
        !db.compact_if_needed()?,
        "policy must not adopt an external reader"
    );
    assert!(db.layered_store().is_none());

    db.compact()?;
    assert!(!db.is_read_only());
    let layered = db.layered_store().expect("installed layered owner");
    assert_eq!(
        db.get_node_property_at_epoch(id, "value", db.current_epoch()),
        Some(Value::Int64(1))
    );
    db.set_node_property(id, "value", Value::Int64(2))?;
    let changed = db.current_epoch();
    db.compact()?;
    assert!(Arc::ptr_eq(
        &layered.base_store(),
        &db.layered_store()
            .expect("retained layered owner")
            .base_store(),
    ));
    assert_eq!(
        db.get_node_property_at_epoch(id, "value", changed),
        Some(Value::Int64(2))
    );
    db.set_node_property(id, "value", Value::Int64(3))?;
    assert_eq!(
        db.get_node_property_at_epoch(id, "value", db.current_epoch()),
        Some(Value::Int64(3))
    );
    assert_eq!(
        source.get_node_property(id, &grafeo_common::types::PropertyKey::new("value")),
        Some(Value::Int64(1)),
        "conversion must not rewrite the caller's external snapshot"
    );
    Ok(())
}
