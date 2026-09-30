//! Integration tests for the single-file `.grafeo` database format.

#![cfg(feature = "lpg")]
#![cfg(feature = "grafeo-file")]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// Helper: extract string values from column 0 of query result rows.
fn extract_strings(rows: &[Vec<Value>]) -> Vec<String> {
    let mut names: Vec<String> = rows
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    names.sort();
    names
}

/// Helper: compute sidecar WAL path for a .grafeo file.
fn sidecar_wal_path(db_path: &std::path::Path) -> std::path::PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

// =========================================================================
// Basic create, open, and reopen
// =========================================================================

#[test]
fn create_new_grafeo_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);

    // File should exist
    assert!(path.exists());
    assert!(path.is_file());

    db.close().unwrap();
}

#[test]
fn insert_close_reopen_persists_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("persist.grafeo");

    // Create and populate
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);
        db.close().unwrap();
    }

    // Sidecar WAL should be gone after close
    assert!(
        !sidecar_wal_path(&path).exists(),
        "sidecar WAL should be removed after close"
    );

    // Reopen and verify
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);

        // Verify data is queryable
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix", "Gus"]);
        db.close().unwrap();
    }
}

#[test]
#[cfg(feature = "wal")]
fn save_as_grafeo_file_from_in_memory() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("exported.grafeo");

    // Create in-memory DB and populate
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    assert_eq!(db.node_count(), 2);

    // Save as .grafeo file
    db.save(&path).unwrap();

    // Open the file and verify
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 2);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(extract_strings(result.rows()), vec!["Amsterdam", "Berlin"]);
    db2.close().unwrap();
}

#[cfg(all(feature = "wal", feature = "gql"))]
#[test]
fn monolithic_save_restores_authoritative_catalog_and_graph_scoped_indexes() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("catalog-and-indexes.grafeo");
    let source = GrafeoDB::new_in_memory();
    let session = source.session();
    session
        .execute("INSERT (:Person {email: 'first@example.com'})")
        .unwrap();
    session
        .execute("CREATE CONSTRAINT unique_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    session
        .execute("CREATE INDEX idx_default_email FOR (n:Person) ON (n.email)")
        .unwrap();
    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("USE GRAPH analytics").unwrap();
    session.execute("INSERT (:Event {code: 'launch'})").unwrap();
    session
        .execute("CREATE INDEX idx_analytics_code FOR (n:Event) ON (n.code)")
        .unwrap();
    session.execute("USE GRAPH DEFAULT").unwrap();
    drop(session);

    // An un-compacted in-memory source takes the monolithic snapshot branch,
    // not the section/container checkpoint path.
    source.save(&path).unwrap();

    let reopened = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = reopened.session();
    assert_eq!(
        extract_strings(session.execute("SHOW CONSTRAINTS").unwrap().rows()),
        vec!["unique_email"]
    );
    assert_eq!(
        extract_strings(session.execute("SHOW INDEXES").unwrap().rows()),
        vec!["idx_analytics_code", "idx_default_email"]
    );
    assert!(reopened.has_property_index("email"));
    assert!(
        session
            .execute("INSERT (:Person {email: 'first@example.com'})")
            .is_err(),
        "the named unique constraint must still enforce after reopen"
    );

    session.execute("USE GRAPH analytics").unwrap();
    assert_eq!(
        session
            .execute("MATCH (n:Event {code: 'launch'}) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(1)
    );
    session.execute("DROP INDEX idx_analytics_code").unwrap();
    assert_eq!(
        extract_strings(session.execute("SHOW INDEXES").unwrap().rows()),
        vec!["idx_default_email"]
    );

    session.execute("USE GRAPH DEFAULT").unwrap();
    assert!(
        reopened.has_property_index("email"),
        "dropping the named-graph index must not remove the default registry"
    );
    session.execute("DROP INDEX idx_default_email").unwrap();
    assert!(!reopened.has_property_index("email"));
    session.execute("DROP CONSTRAINT unique_email").unwrap();
    session
        .execute("INSERT (:Person {email: 'first@example.com'})")
        .expect("dropping the restored constraint removes enforcement");
    drop(session);
    reopened.close().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn monolithic_save_capture_failure_leaves_no_destination() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("must-not-exist.grafeo");
    let db = GrafeoDB::new_in_memory();
    let mut transaction = db.session();
    transaction.begin_transaction().unwrap();

    let error = db.save(&path).unwrap_err().to_string();
    assert!(error.contains("quiescent committed cut"), "{error}");
    assert!(
        std::fs::symlink_metadata(&path).is_err(),
        "capture failure must not leave an empty or partial destination"
    );
    transaction.rollback().unwrap();
}

#[cfg(feature = "wal")]
#[test]
fn monolithic_save_never_clobbers_an_existing_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("owned.grafeo");
    {
        let existing = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        existing.session().execute("INSERT (:Existing)").unwrap();
        existing.close().unwrap();
    }

    let source = GrafeoDB::new_in_memory();
    source.session().execute("INSERT (:Replacement)").unwrap();
    let error = source.save(&path).unwrap_err().to_string();
    assert!(error.contains("existing destination"), "{error}");

    let reopened = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(
        reopened
            .session()
            .execute("MATCH (n:Existing) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(1)
    );
    assert_eq!(
        reopened
            .session()
            .execute("MATCH (n:Replacement) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(0)
    );
    reopened.close().unwrap();
}

#[test]
fn wal_checkpoint_writes_to_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("checkpoint.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    // Checkpoint should write snapshot to file
    db.wal_checkpoint().unwrap();

    // Verify the file manager has a non-empty header
    let fm = db.file_manager().expect("should have file manager");
    let header = fm.active_header();
    // v2 format uses section directory (snapshot_length = 0),
    // v1 used snapshot_length > 0. Either is valid.
    assert!(header.iteration > 0, "checkpoint should have been written");
    assert_eq!(header.node_count, 1);
    assert_eq!(header.edge_count, 0);

    db.close().unwrap();
}

#[test]
fn multiple_checkpoints_alternate_headers() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("multi.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    db.wal_checkpoint().unwrap();

    let fm = db.file_manager().unwrap();
    assert_eq!(fm.active_header().iteration, 1);

    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(fm.active_header().iteration, 2);
    assert_eq!(fm.active_header().node_count, 2);

    db.close().unwrap();

    // Reopen and verify both nodes are there
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 2);
    db2.close().unwrap();
}

#[test]
#[cfg(feature = "wal")]
fn auto_detect_does_not_use_grafeo_file_for_directory_path() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test_legacy");

    // Without .grafeo extension, should use WAL directory format
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

    #[cfg(feature = "grafeo-file")]
    assert!(
        db.file_manager().is_none(),
        "directory path should not use single-file format"
    );

    let session = db.session();
    session.execute("INSERT (:Person {name: 'Butch'})").unwrap();
    db.close().unwrap();

    // Path should be a directory (WAL format)
    assert!(path.is_dir());
}

#[test]
fn info_reports_persistence() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("info.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let info = db.info();
    assert!(info.is_persistent);
    assert!(info.path.is_some());
    db.close().unwrap();
}

// =========================================================================
// Checkpoint merging: data inserted between checkpoints is preserved
// =========================================================================

#[test]
fn checkpoint_merges_incremental_writes() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("merge.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    // Batch 1: insert 3 nodes, checkpoint
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(db.file_manager().unwrap().active_header().node_count, 3);

    // Batch 2: insert 2 more nodes, modify one, checkpoint again
    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(db.file_manager().unwrap().active_header().node_count, 5);

    // Batch 3: delete a node, add an edge, checkpoint
    session
        .execute("MATCH (p:Person {name: 'Gus'}) DELETE p")
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Jules'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();
    db.wal_checkpoint().unwrap();

    let header = db.file_manager().unwrap().active_header();
    assert_eq!(header.node_count, 4);
    assert_eq!(header.edge_count, 1);
    assert_eq!(header.iteration, 3); // 3 checkpoints = iteration 3

    db.close().unwrap();

    // Reopen and verify final state
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 4);
    assert_eq!(db2.edge_count(), 1);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(
        extract_strings(result.rows()),
        vec!["Alix", "Jules", "Mia", "Vincent"]
    );

    // Verify the property survived
    let result = session2
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(31));

    db2.close().unwrap();
}

#[test]
fn writes_after_checkpoint_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("post_checkpoint.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    db.wal_checkpoint().unwrap();

    // Write MORE data after checkpoint, then close (without explicit checkpoint)
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    session.execute("INSERT (:City {name: 'Prague'})").unwrap();
    db.close().unwrap(); // close() does its own checkpoint

    // Reopen: all 3 cities should be present
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 3);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(
        extract_strings(result.rows()),
        vec!["Amsterdam", "Berlin", "Prague"]
    );
    db2.close().unwrap();
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn empty_database_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("empty.grafeo");

    // Create, close immediately, reopen
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.close().unwrap();
    }
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
        db.close().unwrap();
    }
}

#[test]
fn multiple_reopen_cycles() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("cycles.grafeo");

    // Cycle 1: create with data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 2: add more data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 1);
        db.session()
            .execute("INSERT (:Person {name: 'Gus'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 3: add more data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 2);
        db.session()
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 4: verify all data present
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 3);
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(
            extract_strings(result.rows()),
            vec!["Alix", "Gus", "Vincent"]
        );
        db.close().unwrap();
    }
}

#[test]
fn large_property_values_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("large_props.grafeo");

    let big_string = "x".repeat(100_000);

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute_with_params(
                "INSERT (:Doc {content: $text})",
                [("text".to_string(), Value::String(big_string.clone().into()))]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session.execute("MATCH (d:Doc) RETURN d.content").unwrap();
        match &result.rows()[0][0] {
            Value::String(s) => assert_eq!(s.len(), 100_000),
            other => panic!("expected String, got {other:?}"),
        }
        db.close().unwrap();
    }
}

#[test]
fn diverse_property_types_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute(
                "INSERT (:Thing { \
                    str_val: 'hello', \
                    int_val: 42, \
                    float_val: 3.14, \
                    bool_val: true, \
                    list_val: [1, 2, 3] \
                 })",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:Thing) RETURN t.str_val, t.int_val, t.float_val, t.bool_val, t.list_val",
            )
            .unwrap();
        let row = &result.rows()[0];
        assert_eq!(row[0], Value::String("hello".into()));
        assert_eq!(row[1], Value::Int64(42));
        assert!(matches!(row[2], Value::Float64(_)));
        assert_eq!(row[3], Value::Bool(true));
        assert!(matches!(row[4], Value::List(_)));
        db.close().unwrap();
    }
}

#[test]
fn open_nonexistent_creates_new() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("new.grafeo");

    assert!(!path.exists());
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert!(path.exists());
    assert_eq!(db.node_count(), 0);
    db.close().unwrap();
}

#[test]
#[cfg(feature = "wal")]
fn file_grows_and_preserves_deleted_history_across_checkpoint() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("size.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let fm = db.file_manager().unwrap();
    let initial_size = fm.file_size().unwrap();

    // Add substantial data
    let session = db.session();
    for i in 0..100 {
        session
            .execute(&format!(
                "INSERT (:Node {{idx: {i}, data: '{}'}})",
                "x".repeat(1000)
            ))
            .unwrap();
    }
    db.wal_checkpoint().unwrap();
    let large_size = fm.file_size().unwrap();
    assert!(large_size > initial_size, "file should grow with data");
    let before_delete = db.current_epoch();
    let deleted = db
        .iter_nodes()
        .find(|node| node.get_property("idx") == Some(&grafeo_common::types::Value::Int64(99)))
        .unwrap();
    let deleted_id = deleted.id;

    // Delete most data
    session
        .execute("MATCH (n:Node) WHERE n.idx > 5 DELETE n")
        .unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(db.node_count(), 6);
    assert!(db.get_node(deleted_id).is_none());
    let historical = db.get_node_at_epoch(deleted_id, before_delete).unwrap();
    assert_eq!(historical.id, deleted_id);
    assert_eq!(
        historical.get_property("data"),
        deleted.get_property("data")
    );
    db.close().unwrap();
    let reopened = GrafeoDB::open(&path).unwrap();
    assert_eq!(reopened.node_count(), 6);
    assert!(reopened.get_node(deleted_id).is_none());
    let historical = reopened
        .get_node_at_epoch(deleted_id, before_delete)
        .unwrap();
    assert_eq!(historical.id, deleted_id);
    assert_eq!(historical.get_property("idx"), deleted.get_property("idx"));
    assert_eq!(
        historical.get_property("data"),
        deleted.get_property("data")
    );
    reopened.close().unwrap();
}

#[test]
#[cfg(feature = "wal")]
fn sidecar_wal_exists_during_operation() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_lifecycle.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Sidecar WAL should exist while DB is open
    let wal = sidecar_wal_path(&path);
    assert!(wal.exists(), "sidecar WAL should exist during operation");
    assert!(wal.is_dir(), "sidecar WAL should be a directory");

    db.close().unwrap();

    // After close, sidecar should be cleaned up
    assert!(!wal.exists(), "sidecar WAL should be removed after close");
}

#[test]
fn checkpoint_idempotent() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("idempotent.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Multiple checkpoints without intervening writes should be safe
    db.wal_checkpoint().unwrap();
    let iter1 = db.file_manager().unwrap().active_header().iteration;

    db.wal_checkpoint().unwrap();
    let iter2 = db.file_manager().unwrap().active_header().iteration;

    db.wal_checkpoint().unwrap();
    let iter3 = db.file_manager().unwrap().active_header().iteration;

    // Each checkpoint increments the iteration counter
    assert_eq!(iter2, iter1 + 1);
    assert_eq!(iter3, iter2 + 1);

    // Data is still consistent
    assert_eq!(db.node_count(), 1);

    db.close().unwrap();
}

#[test]
fn concurrent_sessions_before_close() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sessions.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

        // Multiple sessions writing data
        let s1 = db.session();
        let s2 = db.session();

        s1.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s2.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s1.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        assert_eq!(db.node_count(), 3);
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 3);
        db.close().unwrap();
    }
}

#[test]
fn named_graphs_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_graphs.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("CREATE GRAPH social").unwrap();
        session.execute("USE GRAPH social").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("USE GRAPH DEFAULT").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        // Default graph should have Gus
        let session = db.session();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Gus"]);

        // Social graph should have Alix
        session.execute("USE GRAPH social").unwrap();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);
        db.close().unwrap();
    }
}

#[test]
fn edges_with_properties_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("edge_props.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS {since: 2020, strength: 0.95}]->(b)",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session
            .execute("MATCH ()-[e:KNOWS]->() RETURN e.since, e.strength")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(2020));
        assert!(matches!(result.rows()[0][1], Value::Float64(_)));
        db.close().unwrap();
    }
}

// =========================================================================
// Corruption and validation
// =========================================================================

#[test]
fn corrupt_snapshot_detected_on_open() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("corrupt.grafeo");

    // Write valid data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    // Corrupt the post-header region at offset 12288 (0x3000). In v1 files
    // this is the snapshot blob; in v2 files it is the section directory.
    // Either way, the bytes here are integrity-checked at open and the
    // corruption must be surfaced rather than masked.
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(12288)).unwrap();
        file.write_all(b"CORRUPTED DATA HERE!!!").unwrap();
    }

    let result = GrafeoDB::with_config(Config::persistent(&path));
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    let signals_corruption = err_msg.contains("checksum")
        || err_msg.contains("section directory")
        || err_msg.contains("failed to parse");
    assert!(
        signals_corruption,
        "expected a corruption-related error, got: {err_msg}"
    );
}

#[test]
fn validate_reports_clean_state() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("valid.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    let validation = db.validate();
    assert!(validation.errors.is_empty(), "should have no errors");
    db.close().unwrap();
}

// =========================================================================
// WAL status and detailed stats
// =========================================================================

#[test]
#[cfg(feature = "wal")]
fn wal_status_reflects_single_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_status.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let status = db.wal_status().unwrap();
    assert!(status.enabled);

    db.close().unwrap();
}

#[test]
fn detailed_stats_with_grafeo_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("stats.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    for i in 0..10 {
        session
            .execute(&format!("INSERT (:Node {{idx: {i}}})"))
            .unwrap();
    }

    let stats = db.detailed_stats();
    assert_eq!(stats.node_count, 10);
    assert_eq!(stats.edge_count, 0);

    db.close().unwrap();
}

// =========================================================================
// File locking
// =========================================================================

#[test]
#[cfg(feature = "wal")]
fn second_open_of_same_file_is_rejected() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("locked.grafeo");

    let db1 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db1.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Second open should fail because the file is locked
    let result = GrafeoDB::open(&path);
    assert!(result.is_err(), "second open should fail due to file lock");

    db1.close().unwrap();

    // After close, open should succeed
    let db2 = GrafeoDB::open(&path).unwrap();
    assert_eq!(db2.node_count(), 1);
    db2.close().unwrap();
}

#[test]
#[cfg(feature = "wal")]
fn lock_released_on_drop() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("drop_lock.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session().execute("INSERT (:X {v: 1})").unwrap();
        // Drop without explicit close: lock should still be released
    }

    // Should be able to open after drop
    let db2 = GrafeoDB::open(&path).unwrap();
    // Data may or may not persist (no explicit close/checkpoint), but open should succeed
    db2.close().unwrap();
}

// =========================================================================
// Schema (DDL) persistence
// =========================================================================

#[test]
#[cfg(feature = "wal")]
fn node_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("schema.grafeo");

    // Create DB and define node types
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
            .unwrap();
        session
            .execute("CREATE NODE TYPE Company (name STRING NOT NULL)")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        db.close().unwrap();
    }

    // Reopen and verify types survived
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        // Verify data
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);

        // Verify node type definitions survived via SHOW NODE TYPES
        let result = session.execute("SHOW NODE TYPES").unwrap();
        let type_names = extract_strings(result.rows());
        assert!(
            type_names.contains(&"Person".to_string()),
            "Person type missing: {type_names:?}"
        );
        assert!(
            type_names.contains(&"Company".to_string()),
            "Company type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
#[cfg(feature = "wal")]
fn edge_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("edge_types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE EDGE TYPE KNOWS (since INT64)")
            .unwrap();
        session
            .execute("CREATE EDGE TYPE WORKS_AT (role STRING)")
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW EDGE TYPES").unwrap();
        let type_names = extract_strings(result.rows());
        assert!(
            type_names.contains(&"KNOWS".to_string()),
            "KNOWS type missing: {type_names:?}"
        );
        assert!(
            type_names.contains(&"WORKS_AT".to_string()),
            "WORKS_AT type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
#[cfg(feature = "wal")]
fn graph_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("graph_types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING)")
            .unwrap();
        session
            .execute("CREATE EDGE TYPE KNOWS (since INT64)")
            .unwrap();
        session
            .execute(
                "CREATE GRAPH TYPE SocialGraph (\
                 NODE TYPE Person (name STRING),\
                 EDGE TYPE KNOWS (since INT64)\
                 )",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW GRAPH TYPES").unwrap();
        let type_names = extract_strings(result.rows());
        assert!(
            type_names.contains(&"SocialGraph".to_string()),
            "SocialGraph type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
fn schema_survives_export_import_roundtrip() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();

    // Export and import
    let snapshot = db.export_snapshot().unwrap();
    let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session2 = db2.session();

    // Verify data
    let result = session2.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(extract_strings(result.rows()), vec!["Alix"]);

    // Verify schema
    let result = session2.execute("SHOW NODE TYPES").unwrap();
    let type_names = extract_strings(result.rows());
    assert!(
        type_names.contains(&"Person".to_string()),
        "Person type missing after import: {type_names:?}"
    );
}

#[test]
#[cfg(feature = "algos")]
fn stored_procedures_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("procedures.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute(
                "CREATE PROCEDURE get_people() RETURNS (name STRING) \
                 AS { MATCH (p:Person) RETURN p.name AS name }",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        // Insert data so the procedure has something to return
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        // CALL the procedure to verify it survived the roundtrip
        let result = session.execute("CALL get_people()").unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);
        db.close().unwrap();
    }
}

#[test]
#[cfg(feature = "wal")]
fn schema_with_data_across_multiple_cycles() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("schema_cycles.grafeo");

    // Cycle 1: Create type + insert
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
            .unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    // Cycle 2: Add another type + more data
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        // Verify first type survived
        let result = session.execute("SHOW NODE TYPES").unwrap();
        assert!(extract_strings(result.rows()).contains(&"Person".to_string()));

        session
            .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
            .unwrap();
        session
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 3: Verify everything
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW NODE TYPES").unwrap();
        let types = extract_strings(result.rows());
        assert!(types.contains(&"Person".to_string()), "Person missing");
        assert!(types.contains(&"City".to_string()), "City missing");

        assert_eq!(db.node_count(), 2);
        db.close().unwrap();
    }
}

// =========================================================================
// WAL-disabled single-file mode (issue #185)
// =========================================================================

/// Verifies the bug fixed in #185: file manager was previously gated behind
/// `wal_enabled`, so opening with WAL disabled + SingleFile produced no output.
/// With the fix, checkpoint-on-close persists the snapshot correctly.
#[test]
fn wal_disabled_single_file_persists_on_close() {
    use grafeo_engine::config::StorageFormat;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("no_wal.grafeo");

    {
        let mut config = Config::persistent(&path).with_storage_format(StorageFormat::SingleFile);
        config.wal_enabled = false;
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        assert_eq!(db.node_count(), 2);

        // No sidecar WAL should exist: WAL is disabled
        assert!(
            !sidecar_wal_path(&path).exists(),
            "no sidecar WAL should be created when wal_enabled: false"
        );

        db.close().unwrap();
    }

    // Sidecar WAL should not exist (was never created)
    assert!(
        !sidecar_wal_path(&path).exists(),
        "sidecar WAL should not exist after close with wal_enabled: false"
    );

    // File should exist and contain the checkpointed data
    assert!(path.exists() && path.is_file());

    {
        let mut config = Config::persistent(&path).with_storage_format(StorageFormat::SingleFile);
        config.wal_enabled = false;
        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(
            db.node_count(),
            2,
            "data must survive close-reopen with WAL disabled"
        );

        let session = db.session();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);
        db.close().unwrap();
    }
}

// =========================================================================
// Drop: implicit close persists data and cleans up sidecar WAL
// =========================================================================

/// Dropping a GrafeoDB without explicitly calling close() still persists all
/// data and cleans up the sidecar WAL, because GrafeoDB::Drop calls close().
/// This verifies that implicit close (via drop) is as safe as explicit close.
#[test]
fn drop_persists_data_and_removes_sidecar_wal() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("drop_implicit_close.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
        // Drop triggers close(), which checkpoints and removes sidecar WAL
        drop(db);
    }

    // Sidecar WAL must be cleaned up (drop triggers close())
    assert!(
        !sidecar_wal_path(&path).exists(),
        "sidecar WAL should be removed after implicit close via drop"
    );

    // File must contain the checkpointed data
    assert!(path.exists() && path.is_file());

    // Reopen: both nodes must be present
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(
            db.node_count(),
            2,
            "both nodes must survive implicit close via drop"
        );
        let result = db
            .session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Jules", "Vincent"]);
        db.close().unwrap();
    }
}

// =========================================================================
// Concurrent read-only access
// =========================================================================

/// Two open_read_only handles on the same file can coexist (shared locks).
#[test]
fn two_concurrent_read_only_opens() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("shared_ro.grafeo");

    // Writer: create, populate, close (releases exclusive lock)
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session()
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        db.close().unwrap();
    }

    // Both read-only handles must open and read successfully
    let ro1 = GrafeoDB::open_read_only(&path).unwrap();
    let ro2 = GrafeoDB::open_read_only(&path).unwrap();

    assert_eq!(ro1.node_count(), 1);
    assert_eq!(ro2.node_count(), 1);

    let r1 = ro1
        .session()
        .execute("MATCH (c:City) RETURN c.name")
        .unwrap();
    let r2 = ro2
        .session()
        .execute("MATCH (c:City) RETURN c.name")
        .unwrap();
    assert_eq!(extract_strings(r1.rows()), vec!["Amsterdam"]);
    assert_eq!(extract_strings(r2.rows()), vec!["Amsterdam"]);

    ro1.close().unwrap();
    ro2.close().unwrap();
}

/// open_read_only must fail while a writer holds the exclusive lock.
#[test]
fn read_only_blocked_while_writer_holds_lock() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("writer_lock.grafeo");

    // Create the file first (writer creates it)
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Mia'})")
            .unwrap();
        db.close().unwrap();
    }

    // Writer holds exclusive lock
    let writer = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

    // Read-only open must fail (cannot acquire shared lock while exclusive held)
    let result = GrafeoDB::open_read_only(&path);
    assert!(
        result.is_err(),
        "open_read_only must fail while writer holds exclusive lock"
    );

    writer.close().unwrap();

    // After writer releases, read-only must succeed
    let ro = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(ro.node_count(), 1);
    ro.close().unwrap();
}

// =========================================================================
// WAL and checkpoint interaction
// =========================================================================

/// Data written to the WAL after the last checkpoint is recovered on reopen.
/// This covers the case where the process exits between a checkpoint and
/// the subsequent close (e.g. a crash or ungraceful shutdown).
#[test]
#[cfg(feature = "wal")]
fn wal_data_after_checkpoint_survives_drop() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_after_checkpoint.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();

        // First batch: checkpointed to snapshot
        session.execute("INSERT (:Person {name: 'Butch'})").unwrap();
        db.wal_checkpoint().unwrap();

        let fm = db.file_manager().unwrap();
        assert_eq!(
            fm.active_header().node_count,
            1,
            "checkpoint must capture first node"
        );

        // Second batch: in WAL only, not yet in snapshot
        session
            .execute("INSERT (:Person {name: 'Shosanna'})")
            .unwrap();

        // Drop without close: second node is only in sidecar WAL
        drop(db);
    }

    // Reopen: both nodes must be present (first from snapshot, second from WAL)
    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.node_count(), 2);
        let result = db
            .session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Butch", "Shosanna"]);
        db.close().unwrap();
    }
}

// ── Read-only checkpoint guard tests ─────────────────────────────

/// Regression: wal_checkpoint() on a read-only database should be a no-op,
/// not an error. Read-only databases have no WAL and the on-disk file is
/// already a valid snapshot.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn wal_checkpoint_on_read_only_is_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro_checkpoint.grafeo");

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::open_read_only(&path).unwrap();
    // Should succeed (no-op), not fail with "read-only mode" error.
    db.wal_checkpoint()
        .expect("wal_checkpoint on read-only should be a no-op");
    assert_eq!(db.node_count(), 1);
}

/// Regression: save() from a read-only database should succeed.
/// save() creates a NEW file at the target path, reading from self.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn save_from_read_only_database() {
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("ro_save_src.grafeo");
    let dest_path = dir.path().join("ro_save_dest.grafeo");

    {
        let db = GrafeoDB::open(&src_path).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::open_read_only(&src_path).unwrap();
    let source_before = std::fs::read(&src_path).unwrap();
    let sidecar = dir.path().join("ro_save_src.grafeo.wal");
    assert!(!sidecar.exists());
    db.save(&dest_path)
        .expect("save from read-only should succeed");
    assert_eq!(std::fs::read(&src_path).unwrap(), source_before);
    assert!(!sidecar.exists());
    assert!(db.execute("INSERT (:Forbidden)").is_err());

    let restored = GrafeoDB::open(&dest_path).unwrap();
    assert_eq!(restored.node_count(), 2);
    restored.close().unwrap();
}

// =========================================================================
// Layered overlay deletion durability (#323 follow-up)
// =========================================================================

#[cfg(feature = "compact-store")]
fn assert_compact_section_reopen_rejected(
    section_type: grafeo_common::storage::SectionType,
    rewrite: impl Fn(&mut u8, &mut Vec<u8>, grafeo_common::types::NodeId, grafeo_common::types::EdgeId),
    reseal_world: bool,
    expected_error: &str,
) {
    use grafeo_common::storage::{Section, SectionType};
    use grafeo_common::utils::error::{Error, StorageError};
    use grafeo_core::graph::compact::deletions_section::OverlayDeletionsSection;
    use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

    for read_only in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay-deletions-version.grafeo");
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let deleted_node = db.create_node(&["Deleted"]);
        let survivor = db.create_node(&["Survivor"]);
        let deleted_edge = db.create_edge(deleted_node, survivor, "LINK");
        db.compact().unwrap();
        assert!(db.delete_edge(deleted_edge));
        assert!(db.delete_node(deleted_node));
        db.close().unwrap();
        drop(db);

        // Use the container writer to retain header coordinates and renew
        // section/directory checksums, as in the other hostile-image tests.
        let file = GrafeoFileManager::open(&path).unwrap();
        let header = file.active_header();
        let directory = file.read_section_directory().unwrap().unwrap();
        let mut sections: Vec<_> = directory
            .entries()
            .iter()
            .map(|entry| {
                (
                    entry.section_type,
                    entry.version,
                    file.read_section_data(entry).unwrap(),
                )
            })
            .collect();
        let compact = sections
            .iter()
            .find(|entry| entry.0 == SectionType::CompactStore)
            .expect("fixture must retain its compact base");
        assert_eq!(compact.1, 9);
        assert!(compact.2.starts_with(b"GCST\x09"));
        let deletion = sections
            .iter()
            .find(|entry| entry.0 == SectionType::OverlayDeletions)
            .expect("fixture must persist its base-node and base-edge deletions");
        assert_eq!(deletion.1, 2);
        assert!(deletion.2.starts_with(b"GODL\x02"));
        let mut current = OverlayDeletionsSection::empty();
        current.deserialize(&deletion.2).unwrap();
        assert_eq!(current.deleted_node_ids(), vec![deleted_node]);
        assert_eq!(current.deleted_edge_ids(), vec![deleted_edge]);
        let target = sections
            .iter_mut()
            .find(|entry| entry.0 == section_type)
            .unwrap();
        rewrite(&mut target.1, &mut target.2, deleted_node, deleted_edge);
        let crc_offset = target.2.len() - 4;
        let crc = crc32fast::hash(&target.2[..crc_offset]);
        target.2[crc_offset..].copy_from_slice(&crc.to_le_bytes());
        if reseal_world {
            reseal_current_compact_sections(&file, &mut sections);
        }
        let writes: Vec<_> = sections
            .iter()
            .map(|entry| SectionWrite::new(entry.0, entry.1, &entry.2))
            .collect();
        file.write_versioned_sections(
            &writes,
            header.epoch,
            header.transaction_id,
            header.node_count,
            header.edge_count,
        )
        .unwrap();
        // Explicitly verify both directory and section CRCs before asking the
        // engine to reject the format. Version/framing controls retain the
        // source world seal and must fail before its digest; unknown flags
        // carry a fresh world seal so they reach the current decoder.
        let directory = file.read_section_directory().unwrap().unwrap();
        let entry = directory
            .entries()
            .iter()
            .find(|entry| entry.section_type == section_type)
            .unwrap();
        let expected = sections
            .iter()
            .find(|entry| entry.0 == section_type)
            .unwrap();
        assert_eq!(entry.version, expected.1);
        assert_eq!(file.read_section_data(entry).unwrap(), expected.2);
        file.close().unwrap();
        drop(file);

        let before = std::fs::read(&path).unwrap();
        let config = if read_only {
            Config::read_only(&path)
        } else {
            Config::persistent(&path)
        };
        let error = GrafeoDB::with_config(config)
            .err()
            .expect("unsupported compact section must not reopen");
        assert!(
            if reseal_world {
                matches!(&error, Error::Internal(message) if message.contains(expected_error))
            } else {
                matches!(
                    &error,
                    Error::Storage(StorageError::Corruption(message))
                        if message.contains(expected_error)
                )
            },
            "read_only={read_only}: expected exact format refusal, got {error}"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "rejected open must leave the container unchanged (read_only={read_only})"
        );
    }
}

#[cfg(feature = "compact-store")]
fn reseal_current_compact_sections(
    file: &grafeo_storage::file::GrafeoFileManager,
    sections: &mut [(grafeo_common::storage::SectionType, u8, Vec<u8>)],
) {
    use grafeo_common::storage::SectionType;
    use grafeo_common::types::{
        AuthoritativeFormat, GraphModelTag, RecoveryImageComponent, RecoveryImageCoordinatesV1,
        WorldCut, WorldMetadataSectionV2,
    };

    let header = file.active_header();
    let metadata_index = sections
        .iter()
        .position(|entry| entry.0 == SectionType::WorldMetadata)
        .unwrap();
    let metadata = WorldMetadataSectionV2::decode(&sections[metadata_index].2).unwrap();
    let descriptor = metadata.cut().descriptor().clone();
    let logical: Vec<_> = descriptor
        .formats()
        .iter()
        .map(|format| {
            let kind = match format.format() {
                AuthoritativeFormat::Lpg => SectionType::LpgStore,
                AuthoritativeFormat::Catalog => SectionType::Catalog,
                AuthoritativeFormat::Cdc => SectionType::Cdc,
                AuthoritativeFormat::Compact => SectionType::CompactStore,
                AuthoritativeFormat::OverlayDeletions => SectionType::OverlayDeletions,
                other => panic!("unexpected format in compact fixture: {other:?}"),
            };
            let entry = sections.iter().find(|entry| entry.0 == kind).unwrap();
            assert_eq!(u16::from(entry.1), format.version());
            (*format, entry.2.as_slice())
        })
        .collect();
    let cut = WorldCut::seal_components(descriptor.clone(), &logical).unwrap();
    let coordinates = RecoveryImageCoordinatesV1::new(
        header.epoch,
        header.transaction_id,
        GraphModelTag::Lpg,
        header.node_count,
        header.edge_count,
    );
    let mut physical: Vec<_> = sections
        .iter()
        .filter(|entry| entry.0 != SectionType::WorldMetadata)
        .map(|entry| {
            RecoveryImageComponent::new(entry.0 as u32, u16::from(entry.1), &entry.2).unwrap()
        })
        .collect();
    physical.push(coordinates.component());
    sections[metadata_index].2 = WorldMetadataSectionV2::seal(cut, &physical)
        .unwrap()
        .encode()
        .unwrap();
}

#[cfg(feature = "compact-store")]
#[test]
fn overlay_deletions_v1_directory_and_payload_are_rejected_on_reopen() {
    assert_compact_section_reopen_rejected(
        grafeo_common::storage::SectionType::OverlayDeletions,
        |version, bytes, node, edge| {
            // Authentic v1 layout: reserved header bytes, node count and IDs,
            // then edge count and IDs. It carries no deletion epochs.
            let mut predecessor = b"GODL\x01\0\0\0".to_vec();
            predecessor.extend_from_slice(&1_u64.to_le_bytes());
            predecessor.extend_from_slice(&node.0.to_le_bytes());
            predecessor.extend_from_slice(&1_u64.to_le_bytes());
            predecessor.extend_from_slice(&edge.0.to_le_bytes());
            predecessor.extend_from_slice(&[0; 4]); // Replaced with a valid CRC.
            *version = 1;
            *bytes = predecessor;
        },
        false,
        "unsupported OverlayDeletions section directory version 1",
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn overlay_deletions_current_directory_rejects_altered_payload_header_on_reopen() {
    assert_compact_section_reopen_rejected(
        grafeo_common::storage::SectionType::OverlayDeletions,
        |version, bytes, _, _| {
            assert_eq!(*version, 2);
            // Keep all current `(id, delete_epoch)` records and the current
            // directory version, but claim v1 in the sealed payload header.
            bytes[4] = 1;
        },
        false,
        "OverlayDeletions directory version 2 disagrees with its payload framing",
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_store_predecessor_directories_and_payloads_are_rejected_on_reopen() {
    let fixtures: [&[u8]; 8] = [
        include_bytes!("fixtures/gcst/rejected_gcst_v1.bin"),
        include_bytes!("fixtures/gcst/rejected_gcst_v2.bin"),
        include_bytes!("fixtures/gcst/rejected_gcst_v3.bin"),
        include_bytes!("fixtures/gcst/rejected_gcst_v4.bin"),
        include_bytes!("fixtures/gcst/rejected_gcst_v5.bin"),
        include_bytes!("fixtures/gcst/rejected_gcst_v6.bin"),
        include_bytes!("fixtures/gcst/rejected_gcst_v7.bin"),
        include_bytes!("fixtures/gcst/rejected_gcst_v8.bin"),
    ];
    for bytes in fixtures {
        let version = bytes[4];
        assert_compact_section_reopen_rejected(
            grafeo_common::storage::SectionType::CompactStore,
            |directory_version, payload, _, _| {
                *directory_version = version;
                *payload = bytes.to_vec();
            },
            false,
            &format!("unsupported CompactStore section directory version {version}"),
        );
    }
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_store_current_directory_rejects_altered_payload_header_on_reopen() {
    assert_compact_section_reopen_rejected(
        grafeo_common::storage::SectionType::CompactStore,
        |version, bytes, _, _| {
            assert_eq!(*version, 9);
            bytes[4] = 8;
        },
        false,
        "CompactStore directory version 9 disagrees with its payload framing",
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_store_unknown_flags_are_rejected_after_world_verification_on_reopen() {
    for flag in [0x04, 0x80] {
        assert_compact_section_reopen_rejected(
            grafeo_common::storage::SectionType::CompactStore,
            |version, bytes, _, _| {
                assert_eq!(*version, 9);
                bytes[5] |= flag;
            },
            true,
            "unsupported CompactStore flags",
        );
    }
}

/// Regression test: when a database has been compacted (so deletes go
/// through the LayeredStore's `deleted_from_base_*` sets rather than
/// directly modifying an LpgStore), those deletions must survive a
/// close/reopen cycle. Before the OverlayDeletions section landed, the
/// deletion sets were in-memory only and previously-deleted base nodes
/// silently reappeared on reopen.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn deleted_base_nodes_stay_deleted_across_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("layered_delete_persist.grafeo");

    {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        // Drop the session so its lock is released before compact() takes
        // the write lock on the store.
        drop(session);

        // Compact: pushes the three nodes into the columnar base and
        // installs a LayeredStore. Subsequent deletes go through
        // `LayeredStore::delete_node` and write to
        // `deleted_from_base_nodes`, which is exactly the path we need
        // to durably persist.
        db.compact().expect("compact should succeed");

        let session = db.session();
        session
            .execute("MATCH (p:Person {name: 'Gus'}) DELETE p")
            .unwrap();
        drop(session);

        // Verify the delete is visible in the running database.
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix", "Vincent"]);
        drop(session);

        db.close().unwrap();
    }

    // Reopen and confirm Gus did not reappear.
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(
        extract_strings(result.rows()),
        vec!["Alix", "Vincent"],
        "previously-deleted base node must stay deleted across reopen"
    );
    drop(session);
    db.close().unwrap();
}

/// Companion test for edge deletion through the LayeredStore.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn deleted_base_edges_stay_deleted_across_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("layered_edge_delete_persist.grafeo");

    {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();
        drop(session);

        db.compact().expect("compact should succeed");

        let session = db.session();
        session
            .execute("MATCH (:Person)-[r:KNOWS]->(:Person) DELETE r")
            .unwrap();
        let result = session
            .execute("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(0));
        drop(session);

        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    let result = session
        .execute("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(0),
        "previously-deleted base edge must stay deleted across reopen"
    );
    drop(session);
    db.close().unwrap();
}
