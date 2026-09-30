//! Integration tests for operational WAL directories and exact saved containers.
//!
//! Proves that data survives a close/reopen cycle in operational WAL directories,
//! including after rotation, and that saved containers recover subsequent tails.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test wal_directory
//! ```

#![cfg(feature = "lpg")]

#[cfg(feature = "wal")]
mod wal_directory {
    use grafeo_common::types::Value;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, DurabilityMode, GrafeoDB};

    #[cfg(feature = "lpg")]
    #[test]
    fn exact_save_auto_detects_both_suffixes_and_recovers_later_wal_tail() {
        use grafeo_common::types::GraphPath;
        use grafeo_engine::{CreateIndexRequest, IndexCreateKind};
        use std::path::{Path, PathBuf};

        fn sidecar(path: &Path) -> PathBuf {
            let mut name = path.as_os_str().to_owned();
            name.push(".wal");
            PathBuf::from(name)
        }

        fn copy_tree(source: &Path, destination: &Path) {
            std::fs::create_dir(destination).unwrap();
            for entry in std::fs::read_dir(source).unwrap() {
                let entry = entry.unwrap();
                let target = destination.join(entry.file_name());
                if entry.file_type().unwrap().is_dir() {
                    copy_tree(&entry.path(), &target);
                } else {
                    std::fs::copy(entry.path(), target).unwrap();
                }
            }
        }

        let source = GrafeoDB::new_in_memory();
        let first = source
            .create_node_with_props(&["Doc"], [("body", Value::from("exact retained document"))]);
        let second = source.create_node(&["Peer"]);
        let edge = source.create_edge(first, second, "LINKS");
        source
            .set_node_property(first, "state", Value::from("before"))
            .unwrap();
        let historical_epoch = source.current_epoch();
        source
            .set_node_property(first, "state", Value::from("saved"))
            .unwrap();
        let burned = source.create_node(&["Burned"]);
        assert!(source.delete_node(burned));
        let retired_owner = source
            .create_index(CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some("retired".into()),
                label: None,
                property: "retired".into(),
                kind: IndexCreateKind::Property,
            })
            .unwrap();
        assert!(source.drop_index(retired_owner).unwrap());
        let owner = source
            .create_index(CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some("state_owner".into()),
                label: None,
                property: "state".into(),
                kind: IndexCreateKind::Property,
            })
            .unwrap();
        assert!(owner.as_u32() > retired_owner.as_u32());
        #[cfg(feature = "text-index")]
        source
            .create_index(CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some("text_owner".into()),
                label: Some("Doc".into()),
                property: "body".into(),
                kind: IndexCreateKind::Text {
                    min_token_length: None,
                },
            })
            .unwrap();
        #[cfg(feature = "vector-index")]
        {
            source
                .set_node_property(first, "embedding", Value::Vector(vec![1.0, 0.0].into()))
                .unwrap();
            source
                .create_index(CreateIndexRequest {
                    graph: GraphPath::root(),
                    name: Some("vector_owner".into()),
                    label: Some("Doc".into()),
                    property: "embedding".into(),
                    kind: IndexCreateKind::Vector {
                        dimensions: Some(2),
                        metric: Some("cosine".into()),
                        m: None,
                        ef_construction: None,
                        ef: None,
                        quantization: None,
                    },
                })
                .unwrap();
        }
        let expected_snapshot = source.export_snapshot().unwrap();
        let expected_cut = source.world_cut().unwrap();
        let temp = tempfile::tempdir().unwrap();
        for name in ["saved", "saved.grafeo"] {
            let path = temp.path().join(name);
            source.save(&path).unwrap();
            assert!(path.is_file());
            assert!(
                !sidecar(&path).exists(),
                "save itself must not synthesize a WAL"
            );
            let reopened = GrafeoDB::open(&path).unwrap();
            assert_eq!(reopened.export_snapshot().unwrap(), expected_snapshot);
            assert_eq!(reopened.world_cut().unwrap(), expected_cut);
            assert_eq!(reopened.store_id(), source.store_id());
            assert_eq!(
                reopened.get_node_property_at_epoch(first, "state", historical_epoch),
                Some(Value::from("before"))
            );
            assert_eq!(reopened.get_edge(edge).unwrap().dst, second);
            assert!(reopened.has_property_index("state"));
            let saved_bytes = std::fs::read(&path).unwrap();
            assert!(source.save(&path).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), saved_bytes);

            reopened
                .set_node_property(first, "state", Value::from("tail"))
                .unwrap();
            let tail_node = reopened.create_node(&["Later"]);
            assert!(tail_node.as_u64() > burned.as_u64());
            let tail_edge = reopened.create_edge(second, tail_node, "LATER");
            let later_owner = reopened
                .create_index(CreateIndexRequest {
                    graph: GraphPath::root(),
                    name: Some("later_owner".into()),
                    label: None,
                    property: "later".into(),
                    kind: IndexCreateKind::Property,
                })
                .unwrap();
            assert!(later_owner.as_u32() > owner.as_u32());
            reopened.wal().unwrap().flush().unwrap();
            assert_eq!(
                std::fs::read(&path).unwrap(),
                saved_bytes,
                "tail must not be checkpointed yet"
            );
            let expected_tail = reopened.export_snapshot().unwrap();
            let expected_tail_cut = reopened.world_cut().unwrap();
            let crash = temp.path().join(format!("crash-{name}"));
            std::fs::copy(&path, &crash).unwrap();
            copy_tree(&sidecar(&path), &sidecar(&crash));
            reopened.close().unwrap();
            let recovered = GrafeoDB::open(&crash).unwrap();
            assert_eq!(recovered.export_snapshot().unwrap(), expected_tail);
            assert_eq!(recovered.world_cut().unwrap(), expected_tail_cut);
            assert_eq!(
                recovered.get_node(first).unwrap().get_property("state"),
                Some(&Value::from("tail"))
            );
            assert_eq!(recovered.get_edge(tail_edge).unwrap().dst, tail_node);
            assert!(
                recovered.drop_index(later_owner).unwrap(),
                "tail restored exact owner ID"
            );
            recovered.close().unwrap();
        }
        assert_eq!(source.export_snapshot().unwrap(), expected_snapshot);
        assert_eq!(source.world_cut().unwrap(), expected_cut);
        source.close().unwrap();
    }

    /// Basic roundtrip: create nodes and edges, close, reopen, verify.
    #[test]
    fn roundtrip_no_rotation() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            let a = db.create_node(&["Person"]);
            db.set_node_property(a, "name", Value::String("Alix".into()))
                .expect("set node property");
            let b = db.create_node(&["Person"]);
            db.set_node_property(b, "name", Value::String("Gus".into()))
                .expect("set node property");
            db.create_edge(a, b, "KNOWS");
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("reopen");
            assert_eq!(db.node_count(), 2, "expected 2 nodes after reopen");
            assert_eq!(db.edge_count(), 1, "expected 1 edge after reopen");
            db.close().expect("close");
        }
    }

    /// Proves the checkpoint.meta bug: data written before WAL rotation
    /// must survive close/reopen. Without the fix, recovery skips old WAL
    /// files because checkpoint.meta records the current sequence number.
    #[test]
    fn roundtrip_after_wal_rotation() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("rotdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");

            // Write nodes BEFORE rotation (these go into wal_0.log)
            let a = db.create_node(&["Person"]);
            db.set_node_property(a, "name", Value::String("Alix".into()))
                .expect("set node property");
            let b = db.create_node(&["Person"]);
            db.set_node_property(b, "name", Value::String("Gus".into()))
                .expect("set node property");
            db.create_edge(a, b, "KNOWS");

            // Force WAL rotation so current_sequence advances
            let wal = db.wal().expect("WAL should be present");
            wal.rotate().expect("rotate WAL");

            // Write more nodes AFTER rotation (these go into wal_1.log)
            let c = db.create_node(&["Person"]);
            db.set_node_property(c, "name", Value::String("Vincent".into()))
                .expect("set node property");

            db.close().expect("close");
        }

        // Reopen and verify ALL data is present
        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("reopen");

            assert_eq!(
                db.node_count(),
                3,
                "expected 3 nodes after reopen (2 before rotation + 1 after)"
            );
            assert_eq!(
                db.edge_count(),
                1,
                "expected 1 edge after reopen (created before rotation)"
            );

            // Verify the node created before rotation is queryable
            let session = db.session();
            let result = session
                .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.name")
                .unwrap();
            assert_eq!(
                result.rows().len(),
                1,
                "node created before WAL rotation should be recoverable"
            );

            db.close().expect("close");
        }
    }

    /// Closing a directory-format database must never manufacture a commit
    /// marker for the last assigned transaction. A transaction rejected by
    /// conflict validation has mutation frames in the WAL but no durable
    /// commit and must remain aborted after reopen.
    #[test]
    fn close_does_not_commit_the_last_rejected_transaction() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("rejected_last_tx");
        let node;
        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            node = db.create_node(&["Counter"]);
            db.set_node_property(node, "value", Value::Int64(0))
                .expect("set node property");

            let mut winner = db.session();
            let mut rejected = db.session();
            winner.begin_transaction().unwrap();
            rejected.begin_transaction().unwrap();
            winner
                .set_node_property(node, "value", Value::Int64(1))
                .expect("buffer winner write");
            rejected
                .set_node_property(node, "value", Value::Int64(2))
                .expect("buffer competing write");
            winner.commit().expect("first writer commits");
            assert!(
                rejected.commit().is_err(),
                "second writer must fail first-writer-wins validation"
            );

            db.close().expect("close after rejected transaction");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(
            db.get_node(node).unwrap().get_property("value"),
            Some(&Value::Int64(1)),
            "close must not authenticate the rejected transaction's WAL tail"
        );
    }

    /// Data accumulates correctly across multiple close/reopen cycles,
    /// including a WAL rotation mid-way through.
    #[test]
    fn multiple_reopen_cycles() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("cycles");
        let open = || {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            GrafeoDB::with_config(config).expect("open")
        };

        // Cycle 1: seed 5 nodes
        {
            let db = open();
            for i in 0..5 {
                db.create_node_with_props(
                    &["Batch"],
                    [("cycle", Value::Int64(1)), ("idx", Value::Int64(i))],
                );
            }
            db.close().expect("close");
        }

        // Cycle 2: verify cycle-1 data, add more, force WAL rotation
        {
            let db = open();
            assert_eq!(db.node_count(), 5);
            for i in 0..10 {
                db.create_node_with_props(
                    &["Batch"],
                    [("cycle", Value::Int64(2)), ("idx", Value::Int64(i))],
                );
            }
            db.wal().expect("wal").rotate().expect("rotate");
            for i in 0..3 {
                db.create_node_with_props(
                    &["Batch"],
                    [("cycle", Value::Int64(2)), ("idx", Value::Int64(10 + i))],
                );
            }
            db.close().expect("close");
        }

        // Cycle 3: all 18 nodes from both cycles must be present
        {
            let db = open();
            assert_eq!(db.node_count(), 18, "5 + 13 nodes across two cycles");

            let session = db.session();
            let result = session
                .execute("MATCH (n:Batch) WHERE n.cycle = 1 RETURN count(n)")
                .unwrap();
            assert_eq!(result.rows()[0][0], Value::Int64(5), "cycle-1 nodes intact");

            let result = session
                .execute("MATCH (n:Batch) WHERE n.cycle = 2 RETURN count(n)")
                .unwrap();
            assert_eq!(
                result.rows()[0][0],
                Value::Int64(13),
                "cycle-2 nodes intact"
            );
            db.close().expect("close");
        }
    }

    /// Dropping the database without calling `close()` must still persist data.
    /// The `Drop` impl calls `close()` internally.
    #[test]
    fn drop_without_explicit_close() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("dropdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open");
            let n = db.create_node(&["Person"]);
            db.set_node_property(n, "name", Value::String("Alix".into()))
                .expect("set node property");
            // intentionally no db.close(), Drop handles it
        }

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("reopen");
            assert_eq!(db.node_count(), 1, "node should survive implicit Drop");

            let session = db.session();
            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert_eq!(
                result.rows()[0][0],
                Value::String("Alix".into()),
                "property should survive implicit Drop"
            );
        }
    }

    /// Regression test for GrafeoDB/grafeo#221 (WAL deadlock on second batch).
    ///
    /// Direct CRUD calls on a persistent database with WAL would deadlock on
    /// the second batch of writes because `sync_all()` was called while holding
    /// the `active_log` mutex, and Batch mode triggers sync when >100ms have
    /// elapsed since the last sync (i.e., the first write of the second batch).
    #[test]
    fn second_batch_crud_does_not_deadlock() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("deadlock_test");

        let config = Config::persistent(&path)
            .with_storage_format(StorageFormat::WalDirectory)
            .with_wal_durability(DurabilityMode::Batch {
                max_delay_ms: 100,
                max_records: 1000,
            });
        let db = GrafeoDB::with_config(config).expect("open");

        // First batch: create nodes with properties
        for i in 0..10 {
            let id = db.create_node(&["Person"]);
            db.set_node_property(id, "name", Value::from(format!("Node{i}")))
                .expect("set node property");
            db.set_node_property(id, "index", Value::Int64(i))
                .expect("set node property");
        }

        // Sleep long enough to trigger the explicitly configured Batch threshold.
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Second batch: this would deadlock before the fix because the first
        // write_frame triggers sync_all() while holding active_log.
        for i in 10..20 {
            let id = db.create_node(&["Person"]);
            db.set_node_property(id, "name", Value::from(format!("Node{i}")))
                .expect("set node property");
            db.set_node_property(id, "index", Value::Int64(i))
                .expect("set node property");
        }

        assert_eq!(db.node_count(), 20);
        db.close().expect("close");

        // Verify data survives reopen
        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(db.node_count(), 20, "all nodes should survive reopen");
        db.close().expect("close");
    }

    /// Verify that no checkpoint.meta file is written for directory format.
    /// Its presence would cause recovery to skip WAL files.
    #[test]
    fn no_checkpoint_meta_written() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("nockpt");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open");
            db.create_node(&["Test"]);
            db.close().expect("close");
        }

        let wal_dir = path.join("wal");
        let checkpoint_meta = wal_dir.join("checkpoint.meta");
        assert!(
            !checkpoint_meta.exists(),
            "checkpoint.meta should NOT exist for directory-format WAL"
        );
    }

    // ========================================================================
    // Regression tests for GrafeoDB/grafeo#252: WAL replay on reopen
    // ========================================================================

    /// Exact reproduction of the #252 bug: data written through sessions/queries
    /// (not the CRUD API) must survive close/reopen. The original bug was that
    /// server-style usage (query API, no explicit save()) lost all data.
    #[test]
    fn query_mutations_persist_directory_format() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("query_dir");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();
            session
                .execute("INSERT (:Person {name: 'Alix', age: 30})")
                .expect("insert Alix");
            session
                .execute("INSERT (:Person {name: 'Gus', age: 25})")
                .expect("insert Gus");
            session
                .execute(
                    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                     INSERT (a)-[:KNOWS {since: 2020}]->(b)",
                )
                .expect("insert edge");
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("reopen");
            assert_eq!(db.node_count(), 2, "query-created nodes should persist");
            assert_eq!(db.edge_count(), 1, "query-created edge should persist");

            let session = db.session();
            let result = session
                .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(
                result.rows()[0][0],
                Value::Int64(30),
                "property should survive WAL replay"
            );

            let result = session
                .execute("MATCH ()-[e:KNOWS]->() RETURN e.since")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(
                result.rows()[0][0],
                Value::Int64(2020),
                "edge property should survive WAL replay"
            );
            db.close().expect("close");
        }
    }

    /// Edge and property operations (set, remove, delete) roundtrip through
    /// the WAL directory format.
    #[test]
    fn edge_property_and_delete_roundtrip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("edgeprops");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open");

            let a = db.create_node(&["Person"]);
            db.set_node_property(a, "name", Value::String("Alix".into()))
                .expect("set node property");
            db.set_node_property(a, "temp", Value::String("remove_me".into()))
                .expect("set node property");
            db.remove_node_property(a, "temp");

            let b = db.create_node(&["Person"]);
            db.set_node_property(b, "name", Value::String("Gus".into()))
                .expect("set node property");

            let c = db.create_node(&["Person"]);
            db.set_node_property(c, "name", Value::String("Vincent".into()))
                .expect("set node property");

            let e1 = db.create_edge(a, b, "KNOWS");
            db.set_edge_property(e1, "weight", Value::Float64(0.8))
                .expect("set edge property");

            let e2 = db.create_edge(b, c, "KNOWS");

            // Delete the edge first, then the node (CRUD API does not detach)
            db.delete_edge(e2);
            db.delete_node(c);

            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("reopen");

            assert_eq!(db.node_count(), 2, "Vincent should be deleted");
            assert_eq!(db.edge_count(), 1, "only KNOWS(Alix->Gus) should remain");

            let session = db.session();

            // Verify removed property is gone
            let result = session
                .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.temp")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(
                result.rows()[0][0],
                Value::Null,
                "removed property should stay removed"
            );

            // Verify edge property survived
            let result = session
                .execute("MATCH ()-[e:KNOWS]->() RETURN e.weight")
                .unwrap();
            assert_eq!(result.rows().len(), 1);
            assert_eq!(
                result.rows()[0][0],
                Value::Float64(0.8),
                "edge property should persist"
            );

            db.close().expect("close");
        }
    }

    /// Named graph operations persist through WAL directory format recovery.
    #[test]
    fn named_graph_persists_directory_format() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("named_graph_dir");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();

            session
                .execute("CREATE GRAPH social")
                .expect("create graph");
            session.execute("USE GRAPH social").expect("use graph");
            session
                .execute("INSERT (:Friend {name: 'Alix'})")
                .expect("insert in named graph");
            session
                .execute("INSERT (:Friend {name: 'Gus'})")
                .expect("insert in named graph");

            // Also insert into default graph
            session.execute("USE GRAPH DEFAULT").expect("use default");
            session
                .execute("INSERT (:Root {tag: 'default'})")
                .expect("insert in default");

            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("reopen");
            let session = db.session();

            // Default graph should have its node
            let result = session.execute("MATCH (n:Root) RETURN n.tag").unwrap();
            assert_eq!(result.rows().len(), 1, "default graph node should persist");

            // Named graph should have its nodes
            session.execute("USE GRAPH social").expect("use graph");
            let result = session
                .execute("MATCH (n:Friend) RETURN n.name ORDER BY n.name")
                .unwrap();
            assert_eq!(
                result.rows().len(),
                2,
                "named graph nodes should persist across restart"
            );

            db.close().expect("close");
        }
    }

    /// Auto-detect format: when using `Config::persistent(path)` without
    /// explicit `StorageFormat`, the database should still recover WAL data.
    /// This matches the exact scenario from the #252 bug report where the
    /// server uses default config.
    #[test]
    fn auto_detect_format_recovery() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("autodetect");

        // Use a directory path (no .grafeo extension) with default config,
        // which should auto-select WAL directory format.
        {
            let config = Config::persistent(&path);
            let db = GrafeoDB::with_config(config).expect("open");
            let session = db.session();
            session
                .execute("INSERT (:Person {name: 'Alix'})")
                .expect("insert");
            session
                .execute("INSERT (:Person {name: 'Gus'})")
                .expect("insert");

            // Explicitly do NOT call save(), matching the server scenario
            db.close().expect("close");
        }

        {
            let config = Config::persistent(&path);
            let db = GrafeoDB::with_config(config).expect("reopen");
            assert_eq!(
                db.node_count(),
                2,
                "auto-detected format should replay WAL on reopen"
            );

            let session = db.session();
            let result = session
                .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
                .unwrap();
            assert_eq!(result.rows().len(), 2);
            db.close().expect("close");
        }
    }
}
