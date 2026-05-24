//! Integration tests for `GrafeoDB::open_multi` — loading multiple
//! snapshot blobs into one in-memory database.
//!
//! Mirrors the `TestSnapshot` pattern from `snapshot.rs` because the
//! real `Snapshot` struct is crate-private; we emit byte payloads via
//! a parallel struct whose bincode encoding is identical to the real
//! one, so the engine accepts them as legitimate snapshots.

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

// --------------------------------------------------------------------
// TestSnapshot mirror — lets us craft snapshot bytes with hand-picked
// NodeIds / EdgeIds for the cross-snapshot conflict tests. Bincode
// encoding is identical to the real `Snapshot` struct in
// persistence.rs because the field shapes match exactly.
// --------------------------------------------------------------------

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
    // Field shapes here must match `SnapshotSchema` in persistence.rs
    // for bincode encoding to round-trip cleanly. `Vec<()>` is safe for
    // the inner sequences as long as we keep them empty (an empty
    // Vec<T> encodes as a length-0 prefix regardless of T); the
    // `graph_type_bindings` field uses the real tuple type because
    // we may legitimately want to seed it in future tests.
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

fn encode_snapshot(nodes: Vec<TestNode>, edges: Vec<TestEdge>) -> Vec<u8> {
    let snap = TestSnapshot {
        // Must match `SNAPSHOT_VERSION` in
        // crates/grafeo-engine/src/database/persistence.rs (crate-private,
        // so it can't be imported here). If that constant bumps, update
        // this literal in the same change.
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

fn node(id: u64, label: &str) -> TestNode {
    TestNode {
        id: NodeId::new(id),
        labels: vec![label.to_string()],
        properties: vec![],
    }
}

#[test]
fn open_multi_rejects_duplicate_node_id_across_snapshots() {
    let a = encode_snapshot(vec![node(1, "Person")], vec![]);
    let b = encode_snapshot(vec![node(1, "Animal")], vec![]);

    let result = GrafeoDB::open_multi(&[a.as_slice(), b.as_slice()]);

    match result {
        Ok(_) => panic!("must reject duplicate NodeId"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("duplicate") && message.contains("node"),
                "error must name the conflict; got: {message}"
            );
        }
    }
}

#[test]
fn open_multi_with_single_snapshot_matches_import_snapshot() {
    // Build a small graph, export it, and confirm `open_multi(&[bytes])`
    // produces a database equivalent to `import_snapshot(bytes)`.
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", "Alix".into());
    let gus = db.create_node(&["Person"]);
    db.set_node_property(gus, "name", "Gus".into());
    db.create_edge(alix, gus, "KNOWS");

    let bytes = db.export_snapshot().expect("export");
    let via_import = GrafeoDB::import_snapshot(&bytes).expect("import_snapshot");
    let via_multi = GrafeoDB::open_multi(&[bytes.as_slice()]).expect("open_multi");

    assert_eq!(via_import.node_count(), via_multi.node_count());
    assert_eq!(via_import.edge_count(), via_multi.edge_count());

    let result = via_multi
        .session()
        .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
        .expect("query");
    assert_eq!(result.rows().len(), 1);

    let row = &result.rows()[0];
    assert_eq!(row[0], Value::String("Alix".into()));
    assert_eq!(row[1], Value::String("Gus".into()));
}
