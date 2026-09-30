//! Tests for the diagnostic content of open_multi error messages —
//! the original snapshot_multi.rs tests assert only that rejection
//! happens; these assert that the operator can actually diagnose
//! WHY without reaching for a debugger.

#![cfg(feature = "lpg")]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn snapshot_with_node_42(label: &str, id_prop: &str) -> TestResult<Vec<u8>> {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))?;
    let session = db.session();
    // Keep the diagnostic's exact NodeId without encoding an obsolete wire
    // layout. Only the final node enters the public extracted snapshot.
    for _ in 0..42 {
        session.create_node_with_props(&[], std::iter::empty::<(&str, Value)>())?;
    }
    let id = session.create_node_with_props(&[label], [("id", Value::from(id_prop))])?;
    assert_eq!(id, NodeId::new(42));
    let shard = db.extract_subgraph(&[id])?;
    assert_eq!(shard.node_count(), 1);
    let bytes = shard.export_snapshot()?;
    assert_eq!(grafeo_engine::snapshot_info(&bytes)?.version, 12);
    Ok(bytes)
}

#[test]
fn duplicate_node_error_names_both_snapshots_and_their_labels() -> TestResult {
    let a = snapshot_with_node_42("UniversalConcept", "concept:bitter")?;
    let b = snapshot_with_node_42("NicheDescriptor", "tea:bitter")?;

    match GrafeoDB::open_multi([a.as_slice(), b.as_slice()]) {
        Ok(_) => panic!("collision must be rejected"),
        Err(e) => {
            let message = e.to_string();
            // Diagnostic must surface BOTH sides — snapshot index,
            // labels, and the external id property.
            assert!(message.contains("42"), "must name the NodeId: {message}");
            assert!(
                message.contains("UniversalConcept"),
                "must name the prior side's label: {message}"
            );
            assert!(
                message.contains("NicheDescriptor"),
                "must name the new side's label: {message}"
            );
            assert!(
                message.contains("concept:bitter"),
                "must name the prior id property: {message}"
            );
            assert!(
                message.contains("tea:bitter"),
                "must name the new id property: {message}"
            );
        }
    }
    Ok(())
}

#[test]
fn schema_conflict_under_union_policy_names_the_type() {
    let db_a = GrafeoDB::new_in_memory();
    db_a.session()
        .execute("CREATE NODE TYPE Person (name STRING)")
        .expect("ddl a");
    let bytes_a = db_a.export_snapshot().expect("export a");

    let db_b = GrafeoDB::new_in_memory();
    db_b.session()
        .execute("CREATE NODE TYPE Person (age INTEGER)")
        .expect("ddl b");
    let bytes_b = db_b.export_snapshot().expect("export b");

    match GrafeoDB::open_multi([bytes_a.as_slice(), bytes_b.as_slice()]) {
        Ok(_) => panic!("same-name-different-shape must reject under union policy"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("Person"),
                "conflict must name the offending type: {message}"
            );
            assert!(
                message.contains("NodeType"),
                "conflict must name the catalog kind: {message}"
            );
            assert!(
                message.contains("redefines") || message.contains("differs"),
                "conflict must signal a redefinition: {message}"
            );
        }
    }
}
