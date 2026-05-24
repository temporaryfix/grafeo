//! Tests for the diagnostic content of open_multi error messages —
//! the original snapshot_multi.rs tests assert only that rejection
//! happens; these assert that the operator can actually diagnose
//! WHY without reaching for a debugger.

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

// TestSnapshot scaffolding — duplicated rather than shared because
// integration tests can't share private helpers across files without
// a `tests/common/mod.rs` arrangement that's heavier than the
// duplication.
#[derive(serde::Serialize, serde::Deserialize)]
struct TestSnapshot {
    version: u8,
    nodes: Vec<TestNode>,
    edges: Vec<TestEdge>,
    named_graphs: Vec<()>,
    rdf_triples: Vec<()>,
    rdf_named_graphs: Vec<()>,
    schema: TestSnapshotSchema,
    indexes: TestSnapshotIndexes,
    epoch: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct TestSnapshotSchema {
    node_types: Vec<()>,
    edge_types: Vec<()>,
    graph_types: Vec<()>,
    procedures: Vec<()>,
    schemas: Vec<()>,
    graph_type_bindings: Vec<(String, String)>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct TestSnapshotIndexes {
    property_indexes: Vec<()>,
    vector_indexes: Vec<()>,
    text_indexes: Vec<()>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestNode {
    id: NodeId,
    labels: Vec<String>,
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

fn encode(nodes: Vec<TestNode>, edges: Vec<TestEdge>) -> Vec<u8> {
    let snap = TestSnapshot {
        // Must match SNAPSHOT_VERSION in persistence.rs.
        version: 4,
        nodes,
        edges,
        named_graphs: vec![],
        rdf_triples: vec![],
        rdf_named_graphs: vec![],
        schema: TestSnapshotSchema::default(),
        indexes: TestSnapshotIndexes::default(),
        epoch: 0,
    };
    bincode::serde::encode_to_vec(&snap, bincode::config::standard()).unwrap()
}

fn node_with_id_prop(id: u64, label: &str, id_prop: &str) -> TestNode {
    TestNode {
        id: NodeId::new(id),
        labels: vec![label.to_string()],
        properties: vec![(
            "id".to_string(),
            vec![(EpochId::new(0), Value::String(id_prop.into()))],
        )],
    }
}

#[test]
fn duplicate_node_error_names_both_snapshots_and_their_labels() {
    let a = encode(
        vec![node_with_id_prop(42, "UniversalConcept", "concept:bitter")],
        vec![],
    );
    let b = encode(
        vec![node_with_id_prop(42, "NicheDescriptor", "tea:bitter")],
        vec![],
    );

    match GrafeoDB::open_multi(&[a.as_slice(), b.as_slice()]) {
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
}
