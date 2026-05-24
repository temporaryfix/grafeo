//! Integration tests for `GrafeoDB::open_multi` — loading multiple
//! snapshot blobs into one in-memory database.
//!
//! Mirrors the `TestSnapshot` pattern from `snapshot.rs` because the
//! real `Snapshot` struct is crate-private; we emit byte payloads via
//! a parallel struct whose bincode encoding is identical to the real
//! one, so the engine accepts them as legitimate snapshots.

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

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
