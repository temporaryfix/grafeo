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

fn edge(id: u64, src: u64, dst: u64, edge_type: &str) -> TestEdge {
    TestEdge {
        id: EdgeId::new(id),
        src: NodeId::new(src),
        dst: NodeId::new(dst),
        edge_type: edge_type.to_string(),
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

#[test]
fn open_multi_resolves_edges_whose_endpoint_lives_in_another_snapshot() {
    // Snapshot A — shared chunk: one UniversalConcept node.
    let a = encode_snapshot(vec![node(10, "UniversalConcept")], vec![]);

    // Snapshot B — niche chunk: one NicheDescriptor node and an edge
    // pointing at the UniversalConcept node that only exists in A.
    let b = encode_snapshot(
        vec![node(20, "NicheDescriptor")],
        vec![edge(100, 20, 10, "MAPS_TO_CONCEPT")],
    );

    let db = GrafeoDB::open_multi(&[a.as_slice(), b.as_slice()])
        .expect("merge two disjoint snapshots with cross-snapshot edge");

    assert_eq!(db.node_count(), 2);
    assert_eq!(db.edge_count(), 1);

    let result = db
        .session()
        .execute(
            "MATCH (n:NicheDescriptor)-[:MAPS_TO_CONCEPT]->(c:UniversalConcept) RETURN count(*)",
        )
        .expect("cypher");
    assert_eq!(result.rows().len(), 1);
    let row = &result.rows()[0];
    let count = match &row[0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64 count, got {other:?}"),
    };
    assert_eq!(count, 1, "MAPS_TO_CONCEPT must resolve across snapshots");
}

#[test]
fn open_multi_rejects_dangling_edge_endpoint() {
    // Snapshot B carries an edge whose `dst` (NodeId 99) is not
    // present in any snapshot. open_multi must reject; otherwise the
    // edge silently becomes orphaned at load time.
    let a = encode_snapshot(vec![node(10, "UniversalConcept")], vec![]);
    let b = encode_snapshot(
        vec![node(20, "NicheDescriptor")],
        vec![edge(100, 20, 99, "MAPS_TO_CONCEPT")],
    );

    let result = GrafeoDB::open_multi(&[a.as_slice(), b.as_slice()]);
    match result {
        Ok(_) => panic!("must reject dangling endpoint"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("99") && message.contains("non-existent"),
                "error must name the missing endpoint; got: {message}"
            );
        }
    }
}

#[test]
fn open_multi_rejects_duplicate_edge_id_across_snapshots() {
    let a = encode_snapshot(
        vec![node(1, "Person"), node(2, "Person")],
        vec![edge(100, 1, 2, "KNOWS")],
    );
    let b = encode_snapshot(
        vec![node(3, "Person"), node(4, "Person")],
        vec![edge(100, 3, 4, "KNOWS")],
    );

    let result = GrafeoDB::open_multi(&[a.as_slice(), b.as_slice()]);

    match result {
        Ok(_) => panic!("must reject duplicate EdgeId"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("duplicate") && message.contains("edge"),
                "error must name the conflict; got: {message}"
            );
        }
    }
}

#[test]
fn open_multi_rejects_divergent_schemas() {
    // Two real databases with different DDL — same node type label but
    // different declared property names — must reject as schema mismatch.
    // No nodes are inserted so no NodeId collision masks the schema check.
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

    // Defensive: confirm the test actually exercises a schema difference.
    // If DDL parsing changes were to collapse both forms to the same
    // schema bytes, this test would silently become a false green.
    assert_ne!(
        bytes_a, bytes_b,
        "test setup: divergent-DDL snapshots must produce different bytes"
    );

    let result = GrafeoDB::open_multi(&[bytes_a.as_slice(), bytes_b.as_slice()]);
    match result {
        Ok(_) => panic!("must reject schema mismatch"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("schema"),
                "error must mention schema; got: {message}"
            );
        }
    }
}

#[test]
fn open_multi_accepts_matching_schemas() {
    // Two databases with identical DDL — even though catalog iteration
    // order is HashMap-dependent — must merge cleanly.
    // No nodes are inserted so no NodeId collision masks the schema check.
    let make_db = || {
        let db = GrafeoDB::new_in_memory();
        db.session()
            .execute("CREATE NODE TYPE Person (name STRING)")
            .expect("ddl");
        db
    };

    let db_a = make_db();
    let bytes_a = db_a.export_snapshot().expect("export a");

    let db_b = make_db();
    let bytes_b = db_b.export_snapshot().expect("export b");

    let merged = GrafeoDB::open_multi(&[bytes_a.as_slice(), bytes_b.as_slice()])
        .expect("matching schemas must merge");

    // Sanity: the merged database should still know about the Person
    // node type — confirms schema survived the merge, not just that
    // `open_multi` returned Ok.
    let result = merged
        .session()
        .execute("MATCH (n:Person) RETURN count(*)")
        .expect("Cypher must compile against the merged schema");
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn open_multi_rejects_empty_input() {
    let result = GrafeoDB::open_multi(&[]);
    match result {
        Ok(_) => panic!("must reject empty snapshot list"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("at least one"),
                "error must explain why; got: {message}"
            );
        }
    }
}

#[test]
fn open_multi_accepts_named_graphs_from_single_snapshot() {
    let db = GrafeoDB::new_in_memory();
    db.create_graph("g1").expect("create_graph");
    let bytes = db.export_snapshot().expect("export");

    // Pair the named-graph snapshot with a plain one — should still load.
    let plain = GrafeoDB::new_in_memory();
    plain.create_node(&["Marker"]);
    let plain_bytes = plain.export_snapshot().expect("export plain");

    let merged = GrafeoDB::open_multi(&[bytes.as_slice(), plain_bytes.as_slice()])
        .expect("named graph in only one snapshot is fine");
    let names = merged.list_graphs();
    assert!(
        names.iter().any(|n| n == "g1"),
        "named graph must be restored; got names: {names:?}"
    );
}

#[test]
fn open_multi_unions_property_indexes_across_snapshots() {
    let db_a = GrafeoDB::new_in_memory();
    db_a.create_property_index("id");
    let bytes_a = db_a.export_snapshot().expect("export a");

    let db_b = GrafeoDB::new_in_memory();
    db_b.create_property_index("slug");
    let bytes_b = db_b.export_snapshot().expect("export b");

    let merged = GrafeoDB::open_multi(&[bytes_a.as_slice(), bytes_b.as_slice()]).expect("merge");

    assert!(merged.has_property_index("id"), "id index must be restored");
    assert!(
        merged.has_property_index("slug"),
        "slug index must be restored"
    );
}

#[cfg(feature = "temporal")]
#[test]
fn open_multi_restores_max_epoch_across_snapshots() {
    use grafeo_common::types::EpochId;

    // Two empty-by-design snapshot blobs whose ONLY meaningful
    // difference is the epoch field, so the test isolates that path
    // from any node/edge restoration noise.
    let mk = |epoch: u64| {
        let snap = TestSnapshot {
            version: 4,
            nodes: vec![],
            edges: vec![],
            named_graphs: vec![],
            rdf_triples: vec![],
            rdf_named_graphs: vec![],
            schema: TestSnapshotSchema::default(),
            indexes: TestSnapshotIndexes::default(),
            epoch,
        };
        bincode::serde::encode_to_vec(&snap, bincode::config::standard()).unwrap()
    };

    let low = mk(5);
    let high = mk(42);
    let merged = GrafeoDB::open_multi(&[low.as_slice(), high.as_slice()]).unwrap();

    // The merged DB should sit at the higher epoch; otherwise a later
    // write would clobber high-epoch property history from `high`.
    assert_eq!(
        merged.current_epoch(),
        EpochId::new(42),
        "open_multi must restore epoch as max across snapshots"
    );
}
