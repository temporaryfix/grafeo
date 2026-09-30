//! Durable Session feed acceptance through the authoritative directory WAL.
#![cfg(all(feature = "wal", feature = "cdc", feature = "lpg", feature = "gql"))]
#![allow(missing_docs)]

use grafeo_common::types::{EpochId, GraphIncarnationId, GraphPath, Value};
use grafeo_engine::cdc::ChangeEvent;
use grafeo_engine::config::{DurabilityMode, StorageFormat};
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::{WalRecord, WalRecovery};

fn config(path: &std::path::Path) -> Config {
    Config::persistent(path)
        .with_cdc()
        .with_storage_format(StorageFormat::WalDirectory)
        .with_wal_durability(DurabilityMode::Sync)
}
fn changes(db: &GrafeoDB) -> Vec<ChangeEvent> {
    let mut cursor = None;
    let mut events = Vec::new();
    loop {
        let page = db
            .session()
            .changes_after(cursor.as_ref(), 2, 64 * 1024 * 1024)
            .unwrap();
        if cursor == Some(page.next) {
            break;
        }
        cursor = Some(page.next);
        events.extend(page.events);
    }
    events
}
fn image(db: &GrafeoDB) -> serde_json::Value {
    serde_json::to_value(changes(db)).unwrap()
}

/// Explicit release qualification control; run once per format/capture mode
/// under the existing source-sealed runner (optionally Valgrind Massif).
#[cfg(feature = "grafeo-file")]
#[test]
#[ignore = "explicit persisted CDC cost/heap qualification"]
fn persisted_cdc_cost_and_footprint_fixture() {
    use std::time::Instant;
    let enabled = match std::env::var("GRAFEO_CDC_COST_CAPTURE").as_deref() {
        Ok("on") => true,
        Ok("off") => false,
        other => panic!("set GRAFEO_CDC_COST_CAPTURE=on|off: {other:?}"),
    };
    let format = match std::env::var("GRAFEO_CDC_COST_FORMAT").as_deref() {
        Ok("directory") => StorageFormat::WalDirectory,
        Ok("container") => StorageFormat::SingleFile,
        other => panic!("set GRAFEO_CDC_COST_FORMAT=directory|container: {other:?}"),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(if format == StorageFormat::SingleFile {
        "cost.grafeo"
    } else {
        "cost"
    });
    let mut cfg = config(&path).with_storage_format(format);
    cfg.cdc_enabled = enabled;
    cfg.cdc_retention.max_epochs = None;
    cfg.cdc_retention.max_events = Some(64);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    let id = db
        .session()
        .create_node_with_props(&["Cost"], [("payload", Value::from("initial"))])
        .unwrap();
    let wal_before = db.wal_status().unwrap();
    let mut transaction_ns = Vec::new();
    let mut commit_ns = Vec::new();
    let mut gc_ns = Vec::new();
    let mut final_payload = String::new();
    for batch in 0..32 {
        let transaction_started = Instant::now();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        for row in 0..16 {
            final_payload = format!("{batch:04}:{row:04}:{}", "x".repeat(1014));
            assert_eq!(final_payload.len(), 1024);
            session
                .set_node_property(id, "payload", Value::from(final_payload.clone()))
                .unwrap();
        }
        let started = Instant::now();
        session.commit().unwrap();
        commit_ns.push(started.elapsed().as_nanos());
        transaction_ns.push(transaction_started.elapsed().as_nanos());
        let started = Instant::now();
        db.gc().unwrap();
        gc_ns.push(started.elapsed().as_nanos());
    }
    let wal_after = db.wal_status().unwrap();
    let retained = db.changes_after(None, 64, 1024 * 1024).unwrap();
    assert_eq!(retained.events.len(), if enabled { 64 } else { 0 });
    let structural = db.memory_usage().cdc;
    let retained_serialized_bytes =
        bincode::serde::encode_to_vec(&retained.events, bincode::config::standard())
            .unwrap()
            .len();
    let expected = serde_json::to_value(&retained).unwrap();
    let cursor = retained.next;
    drop(retained);
    db.close().unwrap();
    drop(db);
    let mut restart_ns = Vec::new();
    for _ in 0..12 {
        let started = Instant::now();
        let reopened = GrafeoDB::with_config(cfg.clone()).unwrap();
        restart_ns.push(started.elapsed().as_nanos());
        assert_eq!(reopened.node_count(), 1);
        assert_eq!(
            reopened.get_node(id).unwrap().get_property("payload"),
            Some(&Value::from(final_payload.clone()))
        );
        assert_eq!(
            serde_json::to_value(reopened.changes_after(None, 64, 1024 * 1024).unwrap()).unwrap(),
            expected
        );
        assert!(
            reopened
                .changes_after(Some(&cursor), 1, 4096)
                .unwrap()
                .events
                .is_empty()
        );
        reopened.close().unwrap();
    }
    println!(
        "CDC_COST {}",
        serde_json::json!({
            "capture": enabled, "format": if format == StorageFormat::SingleFile { "container" } else { "directory" },
            "batches": 32, "updates_per_batch": 16, "payload_bytes": 1024,
            "retained_events": structural.event_count, "cdc_structural_bytes": structural.total_bytes,
            "retained_serialized_bytes": retained_serialized_bytes,
            "wal_added_bytes": wal_after.size_bytes - wal_before.size_bytes,
            "wal_added_records": wal_after.record_count - wal_before.record_count,
            "transaction_ns": transaction_ns, "commit_ns": commit_ns, "gc_ns": gc_ns, "restart_ns": restart_ns,
            "timing_scope": "observations only; Sync I/O and shared-host scheduling included; no calibrated regression claim"
        })
    );
}

#[cfg(all(
    feature = "grafeo-file",
    feature = "triple-store",
    feature = "sparql",
    feature = "testing-statement-injection"
))]
#[test]
fn mixed_publication_pages_retention_and_checkpoint_share_recoverable_cuts() {
    use grafeo_common::utils::error::ErrorCode;
    use grafeo_engine::GraphModel;
    use std::sync::Arc;
    use std::time::Duration;

    fn append(db: &GrafeoDB, round: i64) {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        for _ in 0..2 {
            session
                .create_node_with_props(&["Concurrent"], [("round", Value::Int64(round))])
                .unwrap();
        }
        session
            .execute_sparql(&format!(
                "INSERT DATA {{ <urn:round:{round}> <urn:p> <urn:o> }}"
            ))
            .unwrap();
        session.commit().unwrap();
    }
    for format in [StorageFormat::WalDirectory, StorageFormat::SingleFile] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(if format == StorageFormat::SingleFile {
            "concurrent.grafeo"
        } else {
            "concurrent"
        });
        let mut cfg = config(&path)
            .with_storage_format(format)
            .with_graph_model(GraphModel::Both);
        cfg.cdc_retention.max_epochs = None;
        cfg.cdc_retention.max_events = Some(3);
        let db = Arc::new(GrafeoDB::with_config(cfg.clone()).unwrap());
        append(&db, 0);
        let owned = db.changes_after(None, 3, 64 * 1024).unwrap();
        let owned_image = serde_json::to_value(&owned).unwrap();
        let stale = owned.next;
        let mut cursor = stale;
        for round in 1..=8 {
            let pause = db.testing_pause_cdc_before_publication().unwrap();
            let writer_db = Arc::clone(&db);
            let writer = std::thread::spawn(move || append(&writer_db, round));
            let reached = pause.wait_until_reached(Duration::from_secs(10));
            let fenced = db.testing_publication_write_locked();
            let reader_db = Arc::clone(&db);
            let reader = std::thread::spawn(move || {
                reader_db
                    .changes_after(Some(&cursor), 3, 64 * 1024)
                    .unwrap()
            });
            let maintenance_db = Arc::clone(&db);
            let maintenance = std::thread::spawn(move || {
                maintenance_db.gc().unwrap();
                maintenance_db.wal_checkpoint().unwrap();
            });
            pause.release();
            writer.join().unwrap();
            let page = reader.join().unwrap();
            maintenance.join().unwrap();
            drop(pause);
            assert!(
                reached && fenced,
                "mixed writer must hold publication fence at the staged feed cut"
            );
            assert_eq!(page.events.len(), 3);
            assert_eq!(page.next.sequence, cursor.sequence + 3);
            let epoch = page.events[0].epoch;
            assert!(page.events.iter().all(|event| event.epoch == epoch));
            assert_eq!(
                page.events.iter().filter(|e| e.entity_id.is_node()).count(),
                2
            );
            assert_eq!(
                page.events
                    .iter()
                    .filter(|e| e.entity_id.is_triple())
                    .count(),
                1
            );
            assert_eq!(
                image(&db),
                serde_json::to_value(&page.events).unwrap(),
                "GC retains the complete latest mixed epoch"
            );
            cursor = page.next;
            if round > 1 {
                assert_eq!(
                    db.changes_after(Some(&stale), 3, 64 * 1024)
                        .unwrap_err()
                        .error_code(),
                    ErrorCode::CursorEvicted
                );
            }
        }
        assert_eq!(
            serde_json::to_value(&owned).unwrap(),
            owned_image,
            "owned pages survive retention"
        );
        let expected = image(&db);
        assert_eq!(cursor.sequence, 27);
        db.close().unwrap();
        drop(db);
        for _ in 0..2 {
            let reopened = GrafeoDB::with_config(cfg.clone()).unwrap();
            assert_eq!(image(&reopened), expected);
            assert_eq!(reopened.node_count(), 18);
            assert_eq!(
                reopened
                    .execute_sparql("SELECT ?s WHERE { ?s <urn:p> <urn:o> }")
                    .unwrap()
                    .row_count(),
                9
            );
            let eof = reopened.changes_after(Some(&cursor), 3, 64 * 1024).unwrap();
            assert!(eof.events.is_empty());
            assert_eq!(eof.next, cursor);
            assert_eq!(
                reopened
                    .changes_after(Some(&stale), 3, 64 * 1024)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorEvicted
            );
            reopened.close().unwrap();
        }
    }
}

#[cfg(feature = "grafeo-file")]
#[test]
fn node_label_updates_and_delete_keep_native_images_after_recovery() {
    use grafeo_engine::cdc::ChangeKind;
    for format in [StorageFormat::WalDirectory, StorageFormat::SingleFile] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(if format == StorageFormat::SingleFile {
            "labels.grafeo"
        } else {
            "labels"
        });
        let cfg = config(&path).with_storage_format(format);
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        let session = db.session();
        let node = session
            .create_node_with_props(&["Base"], [("value", Value::Int64(7))])
            .unwrap();
        assert!(session.add_node_label(node, "Added"));
        assert!(session.remove_node_label(node, "Added"));
        assert!(session.delete_node(node));
        let events = changes(&db);
        assert_eq!(events.len(), 4);
        assert_eq!(
            events.iter().map(|e| e.kind.clone()).collect::<Vec<_>>(),
            [
                ChangeKind::Create,
                ChangeKind::Update,
                ChangeKind::Update,
                ChangeKind::Delete
            ]
        );
        for (event, expected) in events.iter().zip([
            vec!["Base"],
            vec!["Added", "Base"],
            vec!["Added", "Base"],
            vec!["Base"],
        ]) {
            let mut labels = event.labels.clone().unwrap();
            labels.sort();
            assert_eq!(labels, expected);
        }
        assert_eq!(events[3].before, events[0].after);
        let expected = image(&db);
        drop(session);
        db.close().unwrap();
        for _ in 0..2 {
            let reopened = GrafeoDB::with_config(cfg.clone()).unwrap();
            assert_eq!(image(&reopened), expected);
            assert_eq!(reopened.node_count(), 0);
            reopened.close().unwrap();
        }
    }
}

#[cfg(all(feature = "grafeo-file", feature = "triple-store", feature = "sparql"))]
#[test]
fn rdf_projection_reconciliation_preserves_exact_feed_across_reopens() {
    use grafeo_core::graph::rdf::{
        RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY,
    };
    use grafeo_engine::GraphModel;
    use grafeo_engine::cdc::ChangeKind;

    for format in [StorageFormat::WalDirectory, StorageFormat::SingleFile] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(if format == StorageFormat::SingleFile {
            "projection.grafeo"
        } else {
            "projection"
        });
        let cfg = config(&path)
            .with_storage_format(format)
            .with_graph_model(GraphModel::Both);
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        db.execute_sparql("INSERT DATA { <http://ex.org/alix> a <http://ex.org/Person> . <http://ex.org/gus> a <http://ex.org/Person> . }").unwrap();
        let source = image(&db);
        assert_eq!(source.as_array().unwrap().len(), 2);
        let id = db
            .declare_rdf_lpg_projection("http://ex.org/Person", "Person")
            .unwrap();
        assert_eq!(image(&db), source, "declaration emits no entity events");
        assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 2);
        let initial = changes(&db);
        assert_eq!(initial.len(), 4);
        let marker = db.rdf_lpg_projection(id).unwrap().owner_marker();
        for (event, iri) in initial[2..]
            .iter()
            .zip(["http://ex.org/alix", "http://ex.org/gus"])
        {
            assert!(event.entity_id.is_node());
            assert_eq!(event.kind, ChangeKind::Create);
            assert_eq!(event.labels.as_ref().unwrap(), &["Person"]);
            assert!(event.before.is_none());
            let props = event.after.as_ref().unwrap();
            assert_eq!(props.len(), 2);
            assert_eq!(
                props.get(RDF_LPG_PROJECTION_IRI_PROPERTY),
                Some(&Value::from(iri))
            );
            assert_eq!(
                props.get(RDF_LPG_PROJECTION_OWNER_PROPERTY),
                Some(&Value::from(marker.clone()))
            );
            assert_eq!(event.lpg_graph, Some(GraphPath::root()));
        }
        assert_eq!(initial[2].epoch, initial[3].epoch);
        let cursor = db
            .session()
            .changes_after(None, 4, 64 * 1024 * 1024)
            .unwrap()
            .next;
        assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 2);
        assert_eq!(
            image(&db),
            serde_json::to_value(&initial).unwrap(),
            "unchanged rebuild emits no events"
        );
        db.execute_sparql("DELETE DATA { <http://ex.org/alix> a <http://ex.org/Person> . }")
            .unwrap();
        assert_eq!(db.rebuild_rdf_lpg_projection(id).unwrap(), 1);
        let final_events = changes(&db);
        assert_eq!(final_events.len(), 6);
        assert!(final_events[4].entity_id.is_triple());
        assert_eq!(final_events[4].kind, ChangeKind::Delete);
        assert_eq!(final_events[5].kind, ChangeKind::Delete);
        assert_eq!(final_events[5].entity_id, initial[2].entity_id);
        assert_eq!(final_events[5].before, initial[2].after);
        assert_eq!(final_events[5].labels, initial[2].labels);
        assert!(final_events[5].after.is_none());
        let expected = image(&db);
        let tail = serde_json::to_value(
            db.session()
                .changes_after(Some(&cursor), 2, 64 * 1024 * 1024)
                .unwrap(),
        )
        .unwrap();
        db.close().unwrap();
        for _ in 0..2 {
            let reopened = GrafeoDB::with_config(cfg.clone()).unwrap();
            assert_eq!(image(&reopened), expected);
            assert_eq!(
                serde_json::to_value(
                    reopened
                        .session()
                        .changes_after(Some(&cursor), 2, 64 * 1024 * 1024)
                        .unwrap()
                )
                .unwrap(),
                tail
            );
            assert_eq!(
                reopened
                    .execute("MATCH (n:Person) RETURN n")
                    .unwrap()
                    .row_count(),
                1
            );
            assert_eq!(reopened.rebuild_rdf_lpg_projection(id).unwrap(), 1);
            assert_eq!(image(&reopened), expected);
            reopened.close().unwrap();
        }
    }
}

#[cfg(all(feature = "compact-store", feature = "grafeo-file"))]
#[test]
fn native_mutation_routes_preserve_exact_feed_through_compact_copy_and_reopen() {
    use grafeo_engine::cdc::{ChangeKind, EntityId};
    for format in [StorageFormat::WalDirectory, StorageFormat::SingleFile] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(if format == StorageFormat::SingleFile {
            "routes.grafeo"
        } else {
            "routes"
        });
        let cfg = config(&path).with_storage_format(format);
        let mut db = GrafeoDB::with_config(cfg.clone()).unwrap();
        let direct = db.create_node_with_props(&["Direct"], [("value", Value::Int64(1))]);
        let session = db
            .session()
            .create_node_with_props(&["Session"], [("value", Value::Int64(2))])
            .unwrap();
        db.execute_with_params(
            "INSERT (:Parameterized {value: $value})",
            [("value".into(), Value::Int64(3))].into(),
        )
        .unwrap();
        let vectors = db.batch_create_nodes(
            "VectorBatch",
            "embedding",
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        );
        let properties = db.batch_create_nodes_with_props(
            "PropertyBatch",
            vec![
                [("value".into(), Value::Int64(4))].into(),
                [("value".into(), Value::Int64(5))].into(),
            ],
        );
        assert_eq!(vectors.len(), 2);
        assert_eq!(properties.len(), 2);
        let edge =
            db.create_edge_with_props(direct, session, "LINK", [("weight", Value::Int64(7))]);
        let events = changes(&db);
        assert_eq!(
            events.len(),
            9,
            "seven node creates, one parameterized property update and one edge create"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == ChangeKind::Create)
                .count(),
            8
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == ChangeKind::Update)
                .count(),
            1
        );
        for id in [direct, session]
            .into_iter()
            .chain(vectors)
            .chain(properties)
        {
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event.entity_id == EntityId::Node(id))
                    .count(),
                1
            );
        }
        assert_eq!(
            events
                .iter()
                .filter(|event| event.entity_id == EntityId::Edge(edge))
                .count(),
            1
        );
        let original = image(&db);
        db.execute("CREATE PROJECTION virtual_route LABELS (Direct)")
            .unwrap();
        db.execute("DROP PROJECTION virtual_route").unwrap();
        assert_eq!(
            image(&db),
            original,
            "virtual metadata emits no logical entity event"
        );
        db.compact().unwrap();
        db.compact().unwrap();
        assert_eq!(
            image(&db),
            original,
            "physical rewrites emit no logical event"
        );
        db.execute("CREATE GRAPH copied_route AS COPY OF default")
            .unwrap();
        let copied_path = GraphPath::from_components(&["copied_route"]).unwrap();
        let copied: Vec<_> = changes(&db)
            .into_iter()
            .filter(|event| event.lpg_graph.as_ref() == Some(&copied_path))
            .collect();
        assert_eq!(
            copied.len(),
            8,
            "seven complete node postimages and one edge postimage"
        );
        assert!(
            copied
                .iter()
                .all(|event| event.kind == ChangeKind::Create && event.after.is_some())
        );
        assert!(copied.iter().all(|event| event.epoch == copied[0].epoch));
        assert_eq!(
            copied
                .iter()
                .filter(|event| event.entity_id.is_node())
                .count(),
            7
        );
        assert_eq!(
            copied
                .iter()
                .filter(|event| matches!(event.entity_id, EntityId::Edge(_)))
                .count(),
            1
        );
        let aba_path = GraphPath::from_components(&["aba_route"]).unwrap();
        let mut incarnations = Vec::new();
        for label in ["Old", "New"] {
            if label == "New" {
                db.execute("DROP GRAPH aba_route").unwrap();
            }
            db.execute("CREATE GRAPH aba_route").unwrap();
            let owner = db.session();
            owner.use_graph_path(&aba_path).unwrap();
            let id = owner.create_node(&[label]);
            let event = changes(&db)
                .into_iter()
                .find(|event| {
                    event.lpg_graph.as_ref() == Some(&aba_path)
                        && event.labels.as_ref().is_some_and(|labels| {
                            labels.iter().any(|value| value.as_str() == label)
                        })
                })
                .unwrap();
            assert_eq!(event.entity_id, EntityId::Node(id));
            incarnations.push((id, event.graph_incarnation.unwrap()));
        }
        assert_eq!(
            incarnations[0].0, incarnations[1].0,
            "fixture must exercise local ID reuse"
        );
        assert_ne!(incarnations[0].1, incarnations[1].1);
        assert_eq!(changes(&db).len(), 19);
        let expected = image(&db);
        let cursor = db.changes_after(None, 9, 1024 * 1024).unwrap().next;
        let expected_tail =
            serde_json::to_value(db.changes_after(Some(&cursor), 32, 1024 * 1024).unwrap())
                .unwrap();
        db.close().unwrap();
        drop(db);
        for _ in 0..2 {
            let reopened = GrafeoDB::with_config(cfg.clone()).unwrap();
            assert_eq!(image(&reopened), expected);
            assert_eq!(
                serde_json::to_value(
                    reopened
                        .changes_after(Some(&cursor), 32, 1024 * 1024)
                        .unwrap()
                )
                .unwrap(),
                expected_tail
            );
            let reader = reopened.session();
            for graph in [GraphPath::root(), copied_path.clone()] {
                reader.use_graph_path(&graph).unwrap();
                assert_eq!(reader.execute("MATCH (n) RETURN n").unwrap().row_count(), 7);
                assert_eq!(
                    reader
                        .execute("MATCH ()-[e:LINK]->() RETURN e.weight")
                        .unwrap()
                        .rows(),
                    &[vec![Value::Int64(7)]]
                );
            }
            reader.use_graph_path(&aba_path).unwrap();
            assert_eq!(
                reader
                    .execute("MATCH (n:Old) RETURN n")
                    .unwrap()
                    .row_count(),
                0
            );
            assert_eq!(
                reader
                    .execute("MATCH (n:New) RETURN n")
                    .unwrap()
                    .row_count(),
                1
            );
            drop(reader);
            reopened.close().unwrap();
        }
    }
}

#[test]
fn directory_wal_reopens_the_exact_ordered_feed_twice() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path());
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let a = session
        .create_node_with_props(
            &["Person"],
            [("name", Value::from("a")), ("rank", Value::Int64(1))],
        )
        .unwrap();
    let b = session.create_node(&["Person"]);
    let edge = session
        .create_edge_with_props(a, b, "KNOWS", [("weight", Value::Int64(2))])
        .unwrap();
    session
        .set_node_property(a, "rank", Value::Int64(3))
        .unwrap();
    session.execute("INSERT (:QueryPath {value: 7})").unwrap();
    assert!(changes(&db).is_empty());
    session.savepoint("kept").unwrap();
    session.execute("INSERT (:TruncatedBySavepoint)").unwrap();
    session.rollback_to_savepoint("kept").unwrap();
    let epoch = session.commit().unwrap();
    let expected = image(&db);
    let events = changes(&db);
    assert_eq!(events.len(), 6); // Query INSERT stages Create followed by its property image.
    assert!(events.iter().all(|event| event.epoch == epoch
        && event.graph_incarnation == Some(GraphIncarnationId::DEFAULT_GRAPH)));
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Discarded)").unwrap();
    session.rollback().unwrap();
    assert_eq!(image(&db), expected);
    drop(session);
    let resume = db.changes_after(None, 3, 1024 * 1024).unwrap().next;
    let expected_tail = db.changes_after(Some(&resume), 100, 1024 * 1024).unwrap();
    db.wal_checkpoint().unwrap(); // Directory WAL is never retired by checkpoint.
    db.close().unwrap();
    drop(db);
    for pass in 0..2 {
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        assert_eq!(image(&db), expected, "reopen {pass}");
        let tail = db
            .session()
            .changes_after(Some(&resume), 100, 1024 * 1024)
            .unwrap();
        assert_eq!(tail.next, expected_tail.next);
        assert_eq!(
            serde_json::to_value(tail.events).unwrap(),
            serde_json::to_value(&expected_tail.events).unwrap()
        );
        assert_eq!(
            db.session()
                .execute("MATCH (n:TruncatedBySavepoint) RETURN n")
                .unwrap()
                .row_count(),
            0
        );
        assert_eq!(
            db.get_node(a).unwrap().get_property("rank"),
            Some(&Value::Int64(3))
        );
        assert!(db.get_edge(edge).is_some());
        if pass == 1 {
            let prior = changes(&db)
                .iter()
                .map(|event| event.timestamp)
                .max()
                .unwrap();
            db.session().execute("INSERT (:AfterReopen)").unwrap();
            assert!(changes(&db).last().unwrap().timestamp > prior);
        }
        db.close().unwrap();
    }
}

#[test]
fn graph_aba_retains_source_incarnations_and_literal_paths() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path());
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    let session = db.session();
    let literal = GraphPath::from_components(&["a/b"]).unwrap();
    let a = GraphPath::from_components(&["a"]).unwrap();
    let nested = GraphPath::from_components(&["a", "b"]).unwrap();
    assert!(db.create_graph_path(&literal).unwrap());
    assert!(db.create_graph_path(&a).unwrap());
    assert!(db.create_graph_path(&nested).unwrap());
    for path in [GraphPath::root(), literal.clone(), nested.clone()] {
        session.use_graph_path(&path).unwrap();
        session.execute("INSERT (:First)").unwrap();
    }
    session.use_graph_path(&GraphPath::root()).unwrap();
    assert!(db.drop_graph_path(&literal).unwrap());
    assert!(db.create_graph_path(&literal).unwrap());
    session.use_graph_path(&literal).unwrap();
    session.execute("INSERT (:Second)").unwrap();
    let events = changes(&db);
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].entity_id, events[1].entity_id);
    assert_eq!(events[1].entity_id, events[2].entity_id);
    assert_ne!(events[1].graph_incarnation, events[3].graph_incarnation);
    assert_eq!(events[1].lpg_graph, Some(literal));
    assert_eq!(events[2].lpg_graph, Some(nested));
    let expected = image(&db);
    drop(session);
    db.close().unwrap();
    drop(db);
    for _ in 0..2 {
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        assert_eq!(image(&db), expected);
        db.close().unwrap();
    }
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn mixed_commit_has_one_manifest_and_two_native_batches() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path()).with_graph_model(grafeo_engine::GraphModel::Both);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Mixed)").unwrap();
    session.execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "line\nquote\"" . GRAPH <urn:g> { <urn:s> <urn:p> "01"^^<http://www.w3.org/2001/XMLSchema#integer> . } }"#).unwrap();
    let epoch = session.commit().unwrap();
    let expected = image(&db);
    let events = changes(&db);
    assert_eq!(events.len(), 3);
    assert!(
        events
            .iter()
            .all(|event| event.epoch == epoch && event.graph_incarnation.is_some())
    );
    assert_eq!(
        events[1].graph_incarnation,
        Some(GraphIncarnationId::DEFAULT_GRAPH)
    );
    assert_ne!(
        events[2].graph_incarnation,
        Some(GraphIncarnationId::DEFAULT_GRAPH)
    );
    drop(session);
    db.close().unwrap();
    drop(db);
    {
        let records = WalRecovery::new(dir.path().join("wal"))
            .unwrap()
            .recover()
            .unwrap();
        let batches: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::CdcBatch {
                    model, epoch: e, ..
                } if *e == epoch => Some(*model),
                _ => None,
            })
            .collect();
        assert_eq!(batches, [1, 2]);
        assert_eq!(records.iter().filter(|record| matches!(record, WalRecord::CommittedWithCdc { epoch: e, models: 3, .. } if *e == epoch)).count(), 1);
    }
    for _ in 0..2 {
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        assert_eq!(image(&db), expected);
        assert_eq!(
            db.session()
                .execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
                .unwrap()
                .row_count(),
            1
        );
        db.close().unwrap();
    }
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn durable_cdc_child() {
    let Ok(path) = std::env::var("GRAFEO_CDC_CHILD_PATH") else {
        return;
    };
    let mode = std::env::var("GRAFEO_CDC_CHILD_MODE").unwrap();
    let cfg = config(std::path::Path::new(&path));
    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    let cfg = cfg.with_graph_model(grafeo_engine::GraphModel::Both);
    let db = GrafeoDB::with_config(cfg).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:CrashWitness)").unwrap();
    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    session
        .execute_sparql("INSERT DATA { <urn:crash> <urn:p> <urn:o> . }")
        .unwrap();
    match mode.as_str() {
        "before" | "after" => {
            std::panic::set_hook(Box::new(|_| std::process::exit(87)));
            grafeo_common::testing::crash::enable_crash_named(if mode == "before" {
                "commit:before_marker"
            } else {
                "commit:after_marker_before_publication"
            });
            session.commit().unwrap();
            panic!("crash site not reached");
        }
        "lost_ack" => {
            grafeo_common::testing::wal_failure::enable_commit_ack_failure_once();
            assert!(session.commit().is_err());
            assert!(db.is_durability_poisoned());
            assert!(
                db.fixture_changes(EpochId::INITIAL..=EpochId::PENDING)
                    .is_err()
            );
        }
        "synced" => {
            session.commit().unwrap();
        }
        _ => panic!("unknown child mode"),
    }
    std::process::exit(87);
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn process_crash_resolves_state_and_feed_together() {
    for mode in ["before", "after", "lost_ack", "synced"] {
        let dir = tempfile::tempdir().unwrap();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "durable_cdc_child", "--nocapture"])
            .env("GRAFEO_CDC_CHILD_PATH", dir.path())
            .env("GRAFEO_CDC_CHILD_MODE", mode)
            .output()
            .unwrap();
        assert_eq!(
            child.status.code(),
            Some(87),
            "{}",
            String::from_utf8_lossy(&child.stderr)
        );
        let cfg = config(dir.path());
        #[cfg(all(feature = "triple-store", feature = "sparql"))]
        let cfg = cfg.with_graph_model(grafeo_engine::GraphModel::Both);
        let count = usize::from(mode != "before");
        let mut previous = None;
        for _ in 0..2 {
            let db = GrafeoDB::with_config(cfg.clone()).unwrap();
            assert_eq!(
                db.session()
                    .execute("MATCH (n:CrashWitness) RETURN n")
                    .unwrap()
                    .row_count(),
                count,
                "{mode}"
            );
            let event_count = count;
            #[cfg(all(feature = "triple-store", feature = "sparql"))]
            let event_count = {
                assert_eq!(
                    db.session()
                        .execute_sparql("SELECT ?s WHERE { ?s <urn:p> <urn:o> }")
                        .unwrap()
                        .row_count(),
                    count,
                    "{mode}"
                );
                event_count * 2
            };
            assert_eq!(changes(&db).len(), event_count, "{mode}");
            let current = image(&db);
            if let Some(previous) = previous.replace(current.clone()) {
                assert_eq!(current, previous, "{mode}");
            }
            db.close().unwrap();
        }
    }
}

#[test]
fn oversized_cdc_payload_aborts_state_and_feed_before_commit() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path());
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .create_node_with_props(
            &["TooLarge"],
            [("payload", Value::from("x".repeat(16 * 1024 * 1024)))],
        )
        .unwrap();
    assert!(
        session.commit().is_err(),
        "the bounded feed must fail while rollback is still possible"
    );
    assert_eq!(db.node_count(), 0);
    assert!(changes(&db).is_empty());
    drop(session);
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(cfg).unwrap();
    assert_eq!(db.node_count(), 0);
    assert!(changes(&db).is_empty());
    db.close().unwrap();
}

#[cfg(all(feature = "grafeo-file", feature = "triple-store", feature = "sparql"))]
fn container_config(path: &std::path::Path) -> Config {
    config(path)
        .with_storage_format(StorageFormat::SingleFile)
        .with_graph_model(grafeo_engine::GraphModel::Both)
}

#[cfg(all(feature = "grafeo-file", feature = "triple-store", feature = "sparql"))]
fn populate_retained_cut(db: &GrafeoDB) {
    let session = db.session();
    let literal = GraphPath::from_components(&["a/b"]).unwrap();
    let a = GraphPath::from_components(&["a"]).unwrap();
    let nested = GraphPath::from_components(&["a", "b"]).unwrap();
    for path in [&literal, &a, &nested] {
        db.create_graph_path(path).unwrap();
    }
    for path in [GraphPath::root(), literal.clone(), nested] {
        session.use_graph_path(&path).unwrap();
        session.execute("INSERT (:Retained {value: 1})").unwrap();
    }
    session.use_graph_path(&GraphPath::root()).unwrap();
    db.drop_graph_path(&literal).unwrap();
    db.create_graph_path(&literal).unwrap();
    session.use_graph_path(&literal).unwrap();
    session.execute("INSERT (:Recreated)").unwrap();
    session
        .execute_sparql("INSERT DATA { <urn:s> <urn:p> 1 . GRAPH <urn:g> { <urn:s> <urn:p> 2 . } }")
        .unwrap();
}

#[cfg(all(feature = "grafeo-file", feature = "triple-store", feature = "sparql"))]
#[test]
fn container_snapshot_and_exact_restore_preserve_native_retained_feed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.grafeo");
    let db = GrafeoDB::with_config(container_config(&path)).unwrap();
    populate_retained_cut(&db);
    let expected = image(&db);
    assert_eq!(changes(&db).len(), 9);
    let bytes = db.export_snapshot().unwrap();
    assert_eq!(&bytes[..5], b"\x0cCDC1");
    let imported = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(image(&imported), expected);
    let fork = GrafeoDB::open_multi([&bytes]).unwrap();
    assert_eq!(image(&fork), expected);
    assert_ne!(fork.store_id(), db.store_id());
    let fork_bytes = fork.export_snapshot().unwrap();
    let fork_reopened = GrafeoDB::import_snapshot(&fork_bytes).unwrap();
    assert_eq!(image(&fork_reopened), expected);
    assert_eq!(
        fork_reopened.world_cut().unwrap(),
        fork.world_cut().unwrap()
    );

    // A non-CDC union source must allocate above the retained owner's lifetime
    // floor, including graph IDs retired before the captured cut.
    let extra = GrafeoDB::new_in_memory();
    let extra_path = GraphPath::from_components(&["union-only"]).unwrap();
    extra.create_graph_path(&extra_path).unwrap();
    let extra_session = extra.session();
    extra_session.use_graph_path(&extra_path).unwrap();
    extra_session.execute("INSERT (:UnionOnly)").unwrap();
    let extra_bytes = extra.export_snapshot().unwrap();
    for sources in [[&bytes, &extra_bytes], [&extra_bytes, &bytes]] {
        let union = GrafeoDB::open_multi(sources).unwrap();
        assert_eq!(image(&union), expected);
        let reopened = GrafeoDB::import_snapshot(&union.export_snapshot().unwrap()).unwrap();
        assert_eq!(image(&reopened), expected);
        let session = reopened.session();
        session.use_graph_path(&extra_path).unwrap();
        assert_eq!(
            session
                .execute("MATCH (n:UnionOnly) RETURN n")
                .unwrap()
                .row_count(),
            1
        );
    }
    let error = GrafeoDB::open_multi([&bytes, &bytes])
        .err()
        .expect("ambiguous retained feed union must reject");
    assert!(
        error
            .to_string()
            .contains("at most one authoritative retained CDC feed")
    );
    let target = GrafeoDB::with_config(
        Config::in_memory()
            .with_cdc()
            .with_graph_model(grafeo_engine::GraphModel::Both),
    )
    .unwrap();
    target.session().execute("INSERT (:TargetOnly)").unwrap();
    target.restore_snapshot(&bytes).unwrap();
    assert_eq!(image(&target), expected);
    assert_eq!(
        target
            .session()
            .execute("MATCH (n:TargetOnly) RETURN n")
            .unwrap()
            .row_count(),
        0
    );
    let copied = dir.path().join("copy.grafeo");
    db.save(&copied).unwrap();
    let saved = GrafeoDB::with_config(container_config(&copied)).unwrap();
    assert_eq!(image(&saved), expected);
    assert_eq!(saved.world_cut().unwrap(), db.world_cut().unwrap());
    for event in changes(&saved).iter().filter(|event| event.after.is_some()) {
        assert_eq!(
            event.after.as_ref().unwrap().get("value"),
            Some(&Value::Int64(1))
        );
    }
    saved.close().unwrap();
    db.wal_checkpoint().unwrap();
    db.close().unwrap();
    drop(db);
    for _ in 0..2 {
        let reopened = GrafeoDB::with_config(container_config(&path)).unwrap();
        assert_eq!(image(&reopened), expected);
        reopened.close().unwrap();
    }
}

#[cfg(all(feature = "grafeo-file", feature = "triple-store", feature = "sparql"))]
#[test]
fn retained_checkpoint_child() {
    let Ok(path) = std::env::var("GRAFEO_CDC_CHECKPOINT_CHILD") else {
        return;
    };
    let db = GrafeoDB::with_config(container_config(std::path::Path::new(&path))).unwrap();
    populate_retained_cut(&db);
    db.wal_checkpoint().unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Tail)").unwrap();
    session
        .execute_sparql("INSERT DATA { <urn:tail> <urn:p> 3 }")
        .unwrap();
    session.commit().unwrap();
    std::fs::write(
        format!("{path}.expected"),
        serde_json::to_vec(&image(&db)).unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{path}.snapshot"), db.export_snapshot().unwrap()).unwrap();
    // No destructor/close checkpoint: recover the acknowledged sidecar tail.
    std::process::exit(87);
}

#[cfg(all(feature = "grafeo-file", feature = "triple-store", feature = "sparql"))]
#[test]
fn checkpoint_plus_synced_tail_recovers_identically_twice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checkpoint.grafeo");
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "retained_checkpoint_child", "--nocapture"])
        .env("GRAFEO_CDC_CHECKPOINT_CHILD", &path)
        .output()
        .unwrap();
    assert_eq!(
        child.status.code(),
        Some(87),
        "{}",
        String::from_utf8_lossy(&child.stderr)
    );
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}.expected", path.display())).unwrap())
            .unwrap();
    let expected_snapshot = std::fs::read(format!("{}.snapshot", path.display())).unwrap();
    for _ in 0..2 {
        let db = GrafeoDB::with_config(container_config(&path)).unwrap();
        assert_eq!(image(&db), expected);
        // The whole exact image also proves generation/floor/next-sequence and
        // HLC authority, before a durable public cursor exists to expose them.
        assert_eq!(db.export_snapshot().unwrap(), expected_snapshot);
        assert_eq!(changes(&db).len(), 11);
        assert_eq!(
            db.session()
                .execute("MATCH (n:Tail) RETURN n")
                .unwrap()
                .row_count(),
            1
        );
        db.close().unwrap();
    }
}

#[cfg(feature = "grafeo-file")]
#[test]
fn container_checkpoint_does_not_resurrect_a_fully_pruned_feed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty-window.grafeo");
    let mut cfg = config(&path).with_storage_format(StorageFormat::SingleFile);
    cfg.cdc_retention.max_events = Some(0);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    db.session().execute("INSERT (:RetainedState)").unwrap();
    assert_eq!(changes(&db).len(), 1);
    db.gc().unwrap();
    assert!(changes(&db).is_empty());
    let empty_cursor = db.changes_after(None, 1, 4096).unwrap().next;
    assert_eq!(empty_cursor.sequence, 1);
    db.wal_checkpoint().unwrap();
    let expected = db.export_snapshot().unwrap();
    db.close().unwrap();
    drop(db);
    for _ in 0..2 {
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        assert!(changes(&db).is_empty());
        assert_eq!(db.changes_after(None, 1, 4096).unwrap().next, empty_cursor);
        assert_eq!(
            db.changes_after(Some(&empty_cursor), 1, 4096).unwrap().next,
            empty_cursor
        );
        assert_eq!(db.node_count(), 1);
        assert_eq!(db.export_snapshot().unwrap(), expected);
        db.close().unwrap();
    }
    let db = GrafeoDB::with_config(cfg).unwrap();
    db.session().execute("INSERT (:NewFeedEvent)").unwrap();
    assert_eq!(changes(&db).len(), 1);
    let resumed = db.changes_after(Some(&empty_cursor), 1, 4096).unwrap();
    assert_eq!(resumed.events.len(), 1);
    assert_eq!(resumed.next.sequence, 2);
    db.close().unwrap();
}

#[test]
fn durable_pages_respect_rows_bytes_and_exclusive_positions() {
    use grafeo_common::types::DurableCursor;
    use grafeo_common::utils::error::ErrorCode;
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    for label in ["One", "Two", "Three"] {
        db.session().execute(&format!("INSERT (:{label})")).unwrap();
    }
    let expected = db
        .fixture_changes(EpochId::INITIAL..=EpochId::PENDING)
        .unwrap();
    assert_eq!(expected.len(), 3);
    for size in [1, 2] {
        let mut collected = Vec::new();
        let mut cursor = None;
        loop {
            let page = db.changes_after(cursor.as_ref(), size, 4096).unwrap();
            assert!(page.events.len() <= size);
            assert_eq!(
                page.next.sequence,
                (collected.len() + page.events.len()) as u64
            );
            if page.events.is_empty() {
                assert_eq!(Some(page.next), cursor);
                break;
            }
            cursor = Some(DurableCursor::from_bytes(&page.next.to_bytes()).unwrap());
            collected.extend(page.events);
        }
        assert_eq!(
            serde_json::to_value(collected).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }
    let first_bytes = bincode::serde::encode_to_vec(&expected[0], bincode::config::standard())
        .unwrap()
        .len();
    let one = db.changes_after(None, 10, first_bytes).unwrap();
    assert_eq!(one.events.len(), 1);
    assert_eq!(one.next.sequence, 1);
    assert_eq!(
        db.changes_after(None, 10, first_bytes - 1)
            .unwrap_err()
            .error_code(),
        ErrorCode::StorageFull
    );
    for (rows, bytes) in [(0, 1024), (1, 0)] {
        assert_eq!(
            db.changes_after(None, rows, bytes)
                .unwrap_err()
                .error_code(),
            ErrorCode::InvalidInput
        );
    }
}

#[test]
fn durable_cursor_errors_are_structured_and_future_cuts_cannot_resume() {
    use grafeo_common::types::{DurableCursor, FeedId, StoreId};
    use grafeo_common::utils::error::ErrorCode;
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.session().execute("INSERT (:First)").unwrap();
    let first = db.changes_after(None, 1, 4096).unwrap().next;
    let old_cut = db.export_snapshot().unwrap();
    db.session().execute("INSERT (:Second)").unwrap();
    let future = db.changes_after(Some(&first), 1, 4096).unwrap().next;
    let foreign = DurableCursor::new(
        StoreId::generate().unwrap(),
        first.feed,
        first.generation,
        first.sequence,
        first.epoch,
    )
    .unwrap();
    let wrong_feed = DurableCursor::new(
        first.store_id,
        FeedId::new(2, 0).unwrap(),
        first.generation,
        first.sequence,
        first.epoch,
    )
    .unwrap();
    for cursor in [foreign, wrong_feed] {
        assert_eq!(
            db.changes_after(Some(&cursor), 1, 4096)
                .unwrap_err()
                .error_code(),
            ErrorCode::CursorForeign
        );
    }
    let mut altered = first;
    altered.digest[0] ^= 1;
    for cursor in [
        altered,
        DurableCursor::new(
            first.store_id,
            first.feed,
            first.generation + 1,
            first.sequence,
            first.epoch,
        )
        .unwrap(),
        DurableCursor::new(
            first.store_id,
            first.feed,
            first.generation,
            99,
            first.epoch,
        )
        .unwrap(),
        DurableCursor::new(
            first.store_id,
            first.feed,
            first.generation,
            first.sequence,
            EpochId::INITIAL,
        )
        .unwrap(),
    ] {
        assert_eq!(
            db.changes_after(Some(&cursor), 1, 4096)
                .unwrap_err()
                .error_code(),
            ErrorCode::CursorInvalid
        );
    }
    let restored = GrafeoDB::import_snapshot(&old_cut).unwrap();
    assert!(
        restored
            .changes_after(Some(&first), 1, 4096)
            .unwrap()
            .events
            .is_empty()
    );
    assert_eq!(
        restored
            .changes_after(Some(&future), 1, 4096)
            .unwrap_err()
            .error_code(),
        ErrorCode::CursorInvalid
    );
    let fork = GrafeoDB::import_snapshot_as_fork(&old_cut).unwrap();
    assert_eq!(
        fork.changes_after(Some(&first), 1, 4096)
            .unwrap_err()
            .error_code(),
        ErrorCode::CursorForeign
    );
}

#[cfg(feature = "grafeo-file")]
#[test]
fn checkpointed_retention_preserves_cursor_floor_and_empty_boundary() {
    use grafeo_common::utils::error::ErrorCode;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cursor-floor.grafeo");
    let mut cfg = config(&path).with_storage_format(StorageFormat::SingleFile);
    cfg.cdc_retention.max_epochs = None;
    cfg.cdc_retention.max_events = Some(1);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    db.session().execute("INSERT (:One)").unwrap();
    let evicted = db.changes_after(None, 1, 4096).unwrap().next;
    db.session().execute("INSERT (:Two)").unwrap();
    let boundary = db.changes_after(Some(&evicted), 1, 4096).unwrap().next;
    db.session().execute("INSERT (:Three)").unwrap();
    db.gc().unwrap();
    assert_eq!(
        db.changes_after(Some(&evicted), 1, 4096)
            .unwrap_err()
            .error_code(),
        ErrorCode::CursorEvicted
    );
    let page = db.changes_after(Some(&boundary), 1, 4096).unwrap();
    assert_eq!(page.next.sequence, 3);
    db.wal_checkpoint().unwrap();
    db.close().unwrap();
    drop(db);
    for _ in 0..2 {
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        assert_eq!(
            db.changes_after(Some(&evicted), 1, 4096)
                .unwrap_err()
                .error_code(),
            ErrorCode::CursorEvicted
        );
        assert_eq!(db.changes_after(None, 1, 4096).unwrap().next, page.next);
        assert_eq!(
            db.changes_after(Some(&boundary), 1, 4096).unwrap().next,
            page.next
        );
        assert!(
            db.changes_after(Some(&page.next), 1, 4096)
                .unwrap()
                .events
                .is_empty()
        );
        db.close().unwrap();
    }
}

#[test]
fn directory_retention_preserves_floor_and_native_graph_across_reopen() {
    use grafeo_common::utils::error::ErrorCode;
    for max_events in [0, 1, 2] {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(dir.path());
        cfg.cdc_retention.max_events = Some(max_events);
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        db.session().execute("INSERT (:First)").unwrap();
        let evicted = db.changes_after(None, 1, 4096).unwrap().next;
        // Both events share one epoch: max_events=1 must remove both.
        db.session().execute("INSERT (:Second), (:Third)").unwrap();
        let end = db.changes_after(None, 8, 4096).unwrap().next;
        db.gc().unwrap();
        db.gc().unwrap(); // Repeating without a changed input writes no transition.
        let retained = image(&db);
        assert_eq!(changes(&db).len(), if max_events == 2 { 2 } else { 0 });
        if max_events < 2 {
            assert_eq!(
                db.changes_after(Some(&evicted), 1, 4096)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorEvicted
            );
        } else {
            assert_eq!(
                db.changes_after(Some(&evicted), 8, 4096)
                    .unwrap()
                    .events
                    .len(),
                2
            );
        }
        db.wal_checkpoint().unwrap();
        db.close().unwrap();
        drop(db);
        for _ in 0..2 {
            let db = GrafeoDB::with_config(cfg.clone()).unwrap();
            assert_eq!(image(&db), retained);
            assert_eq!(
                db.session()
                    .execute("MATCH (n) RETURN n")
                    .unwrap()
                    .row_count(),
                3
            );
            assert!(
                db.changes_after(Some(&end), 1, 4096)
                    .unwrap()
                    .events
                    .is_empty()
            );
            if max_events < 2 {
                assert_eq!(
                    db.changes_after(Some(&evicted), 1, 4096)
                        .unwrap_err()
                        .error_code(),
                    ErrorCode::CursorEvicted
                );
            }
            db.close().unwrap();
        }
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        db.session().execute("INSERT (:Later)").unwrap();
        let later = db.changes_after(Some(&end), 1, 4096).unwrap();
        assert_eq!(later.events.len(), 1);
        assert_eq!(later.next.sequence, end.sequence + 1);
        db.gc().unwrap();
        let after = image(&db);
        db.close().unwrap();
        drop(db);
        let db = GrafeoDB::with_config(cfg).unwrap();
        assert_eq!(image(&db), after);
        assert!(
            db.changes_after(Some(&later.next), 1, 4096)
                .unwrap()
                .events
                .is_empty()
        );
        assert_eq!(
            db.session()
                .execute("MATCH (n) RETURN n")
                .unwrap()
                .row_count(),
            4
        );
        db.close().unwrap();
    }
}

#[cfg(all(feature = "grafeo-file", feature = "triple-store", feature = "sparql"))]
#[test]
fn directory_retention_survives_mixed_snapshots_save_and_reopen() {
    use grafeo_common::utils::error::ErrorCode;
    let dir = tempfile::tempdir().unwrap();
    let mut cfg =
        config(&dir.path().join("source")).with_graph_model(grafeo_engine::GraphModel::Both);
    cfg.cdc_retention.max_events = Some(0);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    db.session().execute("INSERT (:First)").unwrap();
    let evicted = db.changes_after(None, 1, 4096).unwrap().next;
    db.session()
        .execute_sparql("INSERT DATA { GRAPH <urn:g> { <urn:s> <urn:p> <urn:o> } }")
        .unwrap();
    let old_image = image(&db);
    let before = db.export_snapshot().unwrap();
    db.gc().unwrap();
    assert!(changes(&db).is_empty());
    db.session().execute("INSERT (:Later)").unwrap();
    let future = db.changes_after(None, 1, 4096).unwrap().next;
    let expected = image(&db);
    let after = db.export_snapshot().unwrap();
    let saved = dir.path().join("saved.grafeo");
    db.save(&saved).unwrap();
    for (bytes, pruned) in [(&before, false), (&after, true)] {
        let restored = GrafeoDB::import_snapshot(bytes).unwrap();
        assert_eq!(
            image(&restored),
            if pruned {
                expected.clone()
            } else {
                old_image.clone()
            }
        );
        if pruned {
            assert_eq!(
                restored
                    .changes_after(Some(&evicted), 1, 4096)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorEvicted
            );
        } else {
            assert_eq!(
                restored
                    .changes_after(Some(&future), 1, 4096)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorInvalid
            );
        }
    }
    db.close().unwrap();
    drop(db);
    let copied = GrafeoDB::with_config(container_config(&saved)).unwrap();
    let backup = dir.path().join("backup");
    let segment = copied.backup_full(&backup).unwrap();
    copied.close().unwrap();
    drop(copied);
    let restored_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup, segment.end_epoch, &restored_path).unwrap();
    for config in [
        cfg,
        container_config(&saved),
        container_config(&restored_path),
    ] {
        for _ in 0..2 {
            let restored = GrafeoDB::with_config(config.clone()).unwrap();
            assert_eq!(image(&restored), expected);
            assert_eq!(
                restored
                    .session()
                    .execute("MATCH (n) RETURN n")
                    .unwrap()
                    .row_count(),
                2
            );
            assert_eq!(
                restored
                    .session()
                    .execute_sparql("SELECT ?s WHERE { GRAPH <urn:g> { ?s <urn:p> <urn:o> } }")
                    .unwrap()
                    .row_count(),
                1
            );
            assert_eq!(
                restored
                    .changes_after(Some(&evicted), 1, 4096)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorEvicted
            );
            assert!(
                restored
                    .changes_after(Some(&future), 1, 4096)
                    .unwrap()
                    .events
                    .is_empty()
            );
            restored.close().unwrap();
        }
    }
}

#[cfg(all(
    feature = "grafeo-file",
    feature = "triple-store",
    feature = "sparql",
    feature = "compact-store"
))]
#[test]
fn mixed_incremental_target_cut_preserves_feed_through_compact_and_subsequent_write() {
    use grafeo_common::utils::error::ErrorCode;
    use grafeo_engine::cdc::{ChangeKind, EntityId};

    fn graph_image(db: &GrafeoDB) -> [Vec<Vec<Value>>; 4] {
        [
            db.execute("MATCH (n) RETURN n.name ORDER BY n.name")
                .unwrap()
                .rows()
                .to_vec(),
            db.execute("MATCH (a)-[e:LINK]->(b) RETURN a.name, b.name, e.weight")
                .unwrap()
                .rows()
                .to_vec(),
            db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } ORDER BY ?s ?p ?o")
                .unwrap()
                .rows()
                .to_vec(),
            db.execute_sparql(
                "SELECT ?g ?s ?p ?o WHERE { GRAPH ?g { ?s ?p ?o } } ORDER BY ?g ?s ?p ?o",
            )
            .unwrap()
            .rows()
            .to_vec(),
        ]
    }

    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.grafeo");
    let backup = dir.path().join("backup");
    let destination = dir.path().join("restored.grafeo");
    let db = GrafeoDB::with_config(container_config(&source)).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let a = session
        .create_node_with_props(&["Retained"], [("name", Value::from("a"))])
        .unwrap();
    session
        .execute_sparql(
            "INSERT DATA { <urn:base> <urn:p> 1 . GRAPH <urn:g> { <urn:base> <urn:p> 2 } }",
        )
        .unwrap();
    session.commit().unwrap();
    assert_eq!(changes(&db).len(), 3);
    let full = db.backup_full(&backup).unwrap();

    session.begin_transaction().unwrap();
    let b = session
        .create_node_with_props(&["Retained"], [("name", Value::from("b"))])
        .unwrap();
    let edge = session
        .create_edge_with_props(a, b, "LINK", [("weight", Value::Int64(7))])
        .unwrap();
    session
        .execute_sparql(
            "INSERT DATA { <urn:target> <urn:p> 3 . GRAPH <urn:g> { <urn:target> <urn:p> 4 } }",
        )
        .unwrap();
    let target_epoch = session.commit().unwrap();
    let target_cut = db.world_cut().unwrap();
    assert_eq!(target_cut.epoch(), target_epoch);
    let target_graph = graph_image(&db);
    assert_eq!(target_graph.each_ref().map(Vec::len), [2, 1, 2, 2]);
    let target_page = db.changes_after(None, 32, 1024 * 1024).unwrap();
    assert_eq!(target_page.events.len(), 7);
    assert_eq!(
        target_page
            .events
            .iter()
            .filter(|event| event.epoch == target_epoch)
            .count(),
        4
    );
    let target_cursor = target_page.next;
    let target_feed = serde_json::to_value(&target_page).unwrap();

    // Both later transactions belong to the same segment as the selected cut.
    session.begin_transaction().unwrap();
    session
        .set_node_property(a, "name", Value::from("later-a"))
        .unwrap();
    session
        .execute_sparql(
            "INSERT DATA { <urn:later> <urn:p> 5 . GRAPH <urn:g> { <urn:later> <urn:p> 6 } }",
        )
        .unwrap();
    session.commit().unwrap();
    let later = session
        .create_node_with_props(&["Later"], [("name", Value::from("later"))])
        .unwrap();
    let future_cursor = db.changes_after(None, 32, 1024 * 1024).unwrap().next;
    assert_eq!(future_cursor.sequence, target_cursor.sequence + 4);
    let increment = db.backup_incremental(&backup).unwrap();
    assert!(full.end_epoch < target_epoch && target_epoch < increment.end_epoch);
    drop(session);
    db.close().unwrap();
    drop(db);

    GrafeoDB::restore_to_epoch(&backup, target_epoch, &destination).unwrap();
    let restored_config = container_config(&destination);
    for reopen in 0..2 {
        let mut restored = GrafeoDB::with_config(restored_config.clone()).unwrap();
        for compacted in [false, true] {
            if compacted {
                restored.compact().unwrap();
            }
            let restored_cut = restored.world_cut().unwrap();
            if reopen == 0 && !compacted {
                assert_eq!(restored_cut, target_cut);
            }
            // Compaction changes authoritative formats and component bytes,
            // hence their state/manifest digests, while preserving this cut's
            // logical identity and provenance. Graph/feed equality follows below.
            restored_cut
                .verify_for_store(target_cut.store_id())
                .unwrap();
            assert_eq!(restored_cut.epoch(), target_epoch);
            let descriptor = restored_cut.descriptor();
            let target_descriptor = target_cut.descriptor();
            assert_eq!(descriptor.graph_model(), target_descriptor.graph_model());
            assert_eq!(descriptor.schema(), target_descriptor.schema());
            assert_eq!(descriptor.projections(), target_descriptor.projections());
            assert_eq!(descriptor.history(), target_descriptor.history());
            assert_eq!(graph_image(&restored), target_graph);
            assert_eq!(
                restored.get_node(a).unwrap().get_property("name"),
                Some(&Value::from("a"))
            );
            assert_eq!(
                restored.get_node(b).unwrap().get_property("name"),
                Some(&Value::from("b"))
            );
            assert_eq!(
                restored.get_edge(edge).unwrap().get_property("weight"),
                Some(&Value::Int64(7))
            );
            assert!(restored.get_node(later).is_none());
            assert_eq!(
                serde_json::to_value(restored.changes_after(None, 32, 1024 * 1024).unwrap())
                    .unwrap(),
                target_feed
            );
            let eof = restored
                .changes_after(Some(&target_cursor), 32, 1024 * 1024)
                .unwrap();
            assert!(eof.events.is_empty());
            assert_eq!(eof.next, target_cursor);
            assert_eq!(
                restored
                    .changes_after(Some(&future_cursor), 32, 1024 * 1024)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorInvalid
            );
        }
        restored.close().unwrap();
    }

    let restored = GrafeoDB::with_config(restored_config.clone()).unwrap();
    let mut writer = restored.session();
    writer.begin_transaction().unwrap();
    let written = writer
        .create_node_with_props(&["AfterRestore"], [("name", Value::from("written"))])
        .unwrap();
    writer
        .execute_sparql("INSERT DATA { <urn:written> <urn:p> 7 }")
        .unwrap();
    let written_epoch = writer.commit().unwrap();
    drop(writer);
    assert!(written.as_u64() > b.as_u64());
    assert!(written_epoch > target_epoch);
    let tail = restored
        .changes_after(Some(&target_cursor), 32, 1024 * 1024)
        .unwrap();
    assert_eq!(tail.events.len(), 2);
    assert_eq!(tail.next.sequence, target_cursor.sequence + 2);
    assert!(
        tail.events
            .iter()
            .all(|event| event.epoch == written_epoch && event.kind == ChangeKind::Create)
    );
    assert_eq!(
        tail.events
            .iter()
            .filter(|event| event.entity_id == EntityId::Node(written))
            .count(),
        1
    );
    assert_eq!(
        tail.events
            .iter()
            .filter(|event| event.entity_id.is_triple())
            .count(),
        1
    );
    let written_graph = graph_image(&restored);
    assert_eq!(written_graph.each_ref().map(Vec::len), [3, 1, 3, 2]);
    assert_eq!(
        written_graph[0],
        [
            vec![Value::from("a")],
            vec![Value::from("b")],
            vec![Value::from("written")]
        ]
    );
    let written_events = changes(&restored);
    assert_eq!(written_events.len(), 9);
    assert_eq!(
        serde_json::to_value(&written_events[..7]).unwrap(),
        serde_json::to_value(&target_page.events).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&written_events[7..]).unwrap(),
        serde_json::to_value(&tail.events).unwrap()
    );
    let written_feed = image(&restored);
    let written_tail = serde_json::to_value(&tail).unwrap();
    restored.close().unwrap();
    drop(restored);
    let reopened = GrafeoDB::with_config(restored_config).unwrap();
    assert_eq!(graph_image(&reopened), written_graph);
    assert_eq!(image(&reopened), written_feed);
    assert_eq!(
        serde_json::to_value(
            reopened
                .changes_after(Some(&target_cursor), 32, 1024 * 1024)
                .unwrap()
        )
        .unwrap(),
        written_tail
    );
    reopened.close().unwrap();
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn directory_retention_crash_child() {
    let Ok(path) = std::env::var("GRAFEO_CDC_RETENTION_CHILD") else {
        return;
    };
    let mode = std::env::var("GRAFEO_CDC_RETENTION_MODE").unwrap();
    let mut cfg = config(std::path::Path::new(&path)).with_wal_durability(DurabilityMode::Batch {
        max_delay_ms: 60_000,
        max_records: 10_000,
    });
    cfg.cdc_retention.max_events = Some(0);
    let db = GrafeoDB::with_config(cfg).unwrap();
    db.session().execute("INSERT (:A), (:B)").unwrap();
    db.wal_checkpoint().unwrap(); // Persist the preimage for the before-record crash.
    std::fs::write(
        format!("{path}.cursor"),
        db.changes_after(None, 1, 4096).unwrap().next.to_bytes(),
    )
    .unwrap();
    if mode == "failed" {
        // A failed native write makes the WAL owner terminal. GC must neither
        // acknowledge retention nor let close persist an unacknowledged floor.
        grafeo_common::testing::wal_failure::enable_mutation_log_failure_once();
        assert!(db.session().execute("INSERT (:MustFail)").is_err());
        assert!(db.gc().is_err());
        assert!(db.is_durability_poisoned());
        std::process::exit(87);
    }
    std::panic::set_hook(Box::new(|_| std::process::exit(87)));
    if mode != "acknowledged" {
        grafeo_common::testing::crash::enable_crash_named(if mode == "before" {
            "cdc_retention:before_record"
        } else {
            "cdc_retention:after_sync_before_prune"
        });
    }
    db.gc().unwrap();
    assert_eq!(mode, "acknowledged", "crash site not reached");
    std::process::exit(87); // No destructor, close, or later fsync.
}

#[test]
fn directory_epoch_retention_and_foreign_floor_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path());
    cfg.cdc_retention.max_epochs = Some(0);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    db.session().execute("INSERT (:Old)").unwrap();
    db.session().execute("INSERT (:Current)").unwrap();
    let epoch = db.current_epoch();
    db.gc().unwrap();
    assert_eq!(changes(&db).len(), 1);
    let retained = image(&db);
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    assert_eq!(image(&db), retained);
    db.close().unwrap();
    drop(db);
    // Structurally valid metadata cannot invent a different native preimage.
    let wal = grafeo_storage::wal::LpgWal::open(dir.path().join("wal")).unwrap();
    wal.log(&WalRecord::CdcRetention {
        epoch,
        generation: 1,
        previous_floor: 2,
        floor: 4,
        next_sequence: 4,
    })
    .unwrap();
    wal.sync().unwrap();
    drop(wal);
    for _ in 0..2 {
        assert!(GrafeoDB::with_config(cfg.clone()).is_err());
    }
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn directory_retention_crash_and_failed_wal_keep_exact_acknowledged_cut() {
    use grafeo_common::{types::DurableCursor, utils::error::ErrorCode};
    for mode in ["before", "after", "acknowledged", "failed"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store");
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "directory_retention_crash_child", "--nocapture"])
            .env("GRAFEO_CDC_RETENTION_CHILD", &path)
            .env("GRAFEO_CDC_RETENTION_MODE", mode)
            .output()
            .unwrap();
        assert_eq!(
            child.status.code(),
            Some(87),
            "{mode}: {}",
            String::from_utf8_lossy(&child.stderr)
        );
        let cursor = DurableCursor::from_bytes(
            &std::fs::read(format!("{}.cursor", path.display())).unwrap(),
        )
        .unwrap();
        for _ in 0..2 {
            let db = GrafeoDB::with_config(config(&path)).unwrap();
            let pruned = matches!(mode, "after" | "acknowledged");
            assert_eq!(changes(&db).len(), if pruned { 0 } else { 2 }, "{mode}");
            if pruned {
                assert_eq!(
                    db.changes_after(Some(&cursor), 1, 4096)
                        .unwrap_err()
                        .error_code(),
                    ErrorCode::CursorEvicted
                );
            } else {
                assert_eq!(
                    db.changes_after(Some(&cursor), 1, 4096)
                        .unwrap()
                        .events
                        .len(),
                    1
                );
            }
            assert_eq!(
                db.session()
                    .execute("MATCH (n) RETURN n")
                    .unwrap()
                    .row_count(),
                2
            );
            db.close().unwrap();
        }
    }
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;

#[test]
fn session_reads_share_feed_when_capture_is_disabled_and_own_pages_after_close() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.session().create_node(&["Captured"]);
    let original = db.changes_after(None, 1, 4096).unwrap();
    db.set_cdc_enabled(false);
    let reader = db.session();
    assert_eq!(
        reader.changes_after(None, 1, 4096).unwrap().next,
        original.next
    );
    reader.create_node(&["Uncaptured"]);
    assert_eq!(
        reader
            .changes_after(Some(&original.next), 1, 4096)
            .unwrap()
            .next,
        original.next
    );
    db.set_cdc_enabled(true);
    db.session().create_node(&["Later"]);
    let owned = reader.changes_after(Some(&original.next), 1, 4096).unwrap();
    assert_eq!(owned.events.len(), 1);
    assert_eq!(owned.next.sequence, original.next.sequence + 1);
    let retained = serde_json::to_value(&owned).unwrap();
    db.close().unwrap();
    assert!(reader.changes_after(Some(&owned.next), 1, 4096).is_err());
    assert!(db.changes_after(Some(&owned.next), 1, 4096).is_err());
    drop(reader);
    drop(db);
    assert_eq!(serde_json::to_value(owned).unwrap(), retained);
}

#[test]
fn authorized_pages_bound_hidden_inspection_without_charging_hidden_payloads() {
    use grafeo_common::utils::error::ErrorCode;
    use grafeo_engine::auth::{Grant, Identity, Role};
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    let writer = db.session();
    writer.execute("CREATE GRAPH hidden").unwrap();
    writer.execute("USE GRAPH hidden").unwrap();
    writer
        .create_node_with_props(&["Secret"], [("payload", Value::from("x".repeat(8192)))])
        .unwrap();
    writer.execute("USE GRAPH default").unwrap();
    writer.create_node(&["Visible"]);
    let all = db.changes_after(None, 100, 1024 * 1024).unwrap();
    assert_eq!(all.events.len(), 2);
    let reader = db.session_with_identity(
        Identity::new("root-reader", [Role::ReadOnly])
            .with_grants([Grant::new(GraphPath::root(), Role::ReadOnly)]),
    );
    let hidden = reader.changes_after(None, 1, 1).unwrap();
    assert!(hidden.events.is_empty());
    assert_eq!(hidden.next.sequence, 1);
    let visible = reader.changes_after(Some(&hidden.next), 1, 4096).unwrap();
    assert_eq!(visible.events.len(), 1);
    assert_eq!(visible.events[0].graph_path(), Some(&GraphPath::root()));
    assert_eq!(visible.next, all.next);
    let eof = reader.changes_after(Some(&visible.next), 1, 1).unwrap();
    assert!(eof.events.is_empty());
    assert_eq!(eof.next, visible.next);
    assert_eq!(
        reader.changes_after(None, 0, 1).unwrap_err().error_code(),
        ErrorCode::InvalidInput
    );
    let other = GrafeoDB::new_in_memory();
    let foreign = other.changes_after(None, 1, 1).unwrap().next;
    assert_eq!(
        reader
            .changes_after(Some(&foreign), 1, 1)
            .unwrap_err()
            .error_code(),
        ErrorCode::CursorForeign
    );
    let denied = db.session_with_identity(Identity::new("denied", []));
    assert!(denied.changes_after(None, 1, 1).is_err());
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn authorized_rdf_pages_advance_past_hidden_named_graphs() {
    use grafeo_engine::auth::{Identity, RdfGraphGrant, Role};
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_cdc()
            .with_graph_model(grafeo_engine::GraphModel::Rdf),
    )
    .unwrap();
    db.session()
        .execute_sparql("INSERT DATA { GRAPH <urn:hidden> { <urn:s> <urn:p> 1 } }")
        .unwrap();
    db.session()
        .execute_sparql("INSERT DATA { <urn:visible> <urn:p> 2 }")
        .unwrap();
    let reader = db.session_with_identity(
        Identity::new("rdf-default", [Role::ReadOnly]).with_rdf_grants([RdfGraphGrant::Default {
            role: Role::ReadOnly,
        }]),
    );
    let hidden = reader.changes_after(None, 1, 1).unwrap();
    assert!(hidden.events.is_empty());
    assert_eq!(hidden.next.sequence, 1);
    let visible = reader.changes_after(Some(&hidden.next), 1, 4096).unwrap();
    assert_eq!(visible.events.len(), 1);
    assert_eq!(
        visible.events[0].triple_subject.as_deref(),
        Some("<urn:visible>")
    );
    assert_eq!(visible.events[0].triple_graph, None);
    assert_eq!(visible.next.sequence, 2);
}

#[test]
fn indexed_entity_pages_seek_candidates_and_share_global_cursor_positions() {
    use grafeo_engine::cdc::EntityHistoryQuery;
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let ids: Vec<_> = (0..4096)
        .map(|_| session.create_node(&["Indexed"]))
        .collect();
    session.commit().unwrap();
    let query = EntityHistoryQuery::new(ids[2048]);
    let page = db.history_after(&query, None, 1, 4096).unwrap();
    assert_eq!(
        page.events.len(),
        1,
        "one inspected candidate must reach the indexed entity"
    );
    assert_eq!(page.events[0].entity_id.as_u64(), ids[2048].as_u64());
    assert_eq!(page.next.sequence, 2049);
    assert_eq!(
        page.events.capacity(),
        1,
        "one-row budget must not reserve four event slots"
    );
    let adjacent = db.changes_after(Some(&page.next), 1, 4096).unwrap();
    assert_eq!(adjacent.events[0].entity_id.as_u64(), ids[2049].as_u64());
    let exhausted = session
        .history_after(&query, Some(&page.next), 1, 1)
        .unwrap();
    assert!(exhausted.events.is_empty());
    assert_eq!(exhausted.next.sequence, 4096);
    assert_eq!(
        session
            .history_after(&query, Some(&exhausted.next), 1, 1)
            .unwrap()
            .next,
        exhausted.next
    );
    assert!(db.history_after(&query, None, 0, 4096).is_err());
    assert!(db.history_after(&query, None, 1, 1).is_err());
}

#[test]
fn indexed_history_filters_graph_epoch_and_permissions_before_payload_copying() {
    use grafeo_engine::auth::{Grant, Identity, Role};
    use grafeo_engine::cdc::{EntityHistoryQuery, HistoryGraph};
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    let writer = db.session();
    writer.execute("CREATE GRAPH hidden").unwrap();
    writer.execute("USE GRAPH hidden").unwrap();
    let hidden = writer
        .create_node_with_props(&["Hidden"], [("payload", Value::from("x".repeat(8192)))])
        .unwrap();
    writer.execute("USE GRAPH default").unwrap();
    let visible = writer.create_node(&["Visible"]);
    assert_eq!(hidden, visible);
    let reader = db.session_with_identity(
        Identity::new("root-only", [Role::ReadOnly])
            .with_grants([Grant::new(GraphPath::root(), Role::ReadOnly)]),
    );
    let mut query = EntityHistoryQuery::new(visible);
    let skipped = reader.history_after(&query, None, 1, 1).unwrap();
    assert!(skipped.events.is_empty());
    assert_eq!(skipped.next.sequence, 1);
    let page = reader
        .history_after(&query, Some(&skipped.next), 1, 4096)
        .unwrap();
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].graph_path(), Some(&GraphPath::root()));
    query.since_epoch = page.events[0].epoch;
    let skipped_epoch = db.history_after(&query, None, 1, 1).unwrap();
    assert_eq!(skipped_epoch.next, skipped.next);
    assert!(skipped_epoch.events.is_empty());
    query.graph = HistoryGraph::Lpg(GraphPath::from_components(&["hidden"]).unwrap());
    assert!(reader.history_after(&query, None, 1, 4096).is_err());
    query.graph = HistoryGraph::Lpg(GraphPath::root());
    let matched = reader
        .history_after(&query, Some(&skipped.next), 1, 4096)
        .unwrap();
    assert_eq!(matched.next, page.next);
    db.close().unwrap();
    assert!(
        reader
            .history_after(&query, Some(&matched.next), 1, 4096)
            .is_err()
    );
    assert_eq!(matched.events.len(), 1);
}

#[test]
fn entity_history_cursor_resumes_exact_updates_after_two_reopens() {
    use grafeo_engine::cdc::EntityHistoryQuery;
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path());
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    let session = db.session();
    let node = session.create_node(&["History"]);
    session
        .set_node_property(node, "n", Value::Int64(1))
        .unwrap();
    session
        .set_node_property(node, "n", Value::Int64(2))
        .unwrap();
    let query = EntityHistoryQuery::new(node);
    let first = session.history_after(&query, None, 1, 4096).unwrap();
    let tail = session
        .history_after(&query, Some(&first.next), 2, 4096)
        .unwrap();
    assert_eq!(tail.events.len(), 2);
    let expected = serde_json::to_value(&tail).unwrap();
    drop(session);
    db.close().unwrap();
    drop(db);
    for _ in 0..2 {
        let db = GrafeoDB::with_config(cfg.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(
                db.session()
                    .history_after(&query, Some(&first.next), 2, 4096)
                    .unwrap()
            )
            .unwrap(),
            expected
        );
        db.close().unwrap();
    }
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn entity_history_rdf_scope_keeps_default_and_named_grants_distinct() {
    use grafeo_engine::auth::{Identity, RdfGraphGrant, Role};
    use grafeo_engine::cdc::{EntityHistoryQuery, HistoryGraph};
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_cdc()
            .with_graph_model(grafeo_engine::GraphModel::Rdf),
    )
    .unwrap();
    db.session()
        .execute_sparql("INSERT DATA { <urn:s> <urn:p> 1 . GRAPH <urn:g> { <urn:s> <urn:p> 1 } }")
        .unwrap();
    let all = db.changes_after(None, 2, 4096).unwrap();
    assert_eq!(all.events.len(), 2);
    let default = all
        .events
        .iter()
        .find(|event| event.triple_graph.is_none())
        .unwrap();
    let named = all
        .events
        .iter()
        .find(|event| event.triple_graph.is_some())
        .unwrap();
    let reader = db.session_with_identity(
        Identity::new("rdf-default", [Role::ReadOnly]).with_rdf_grants([RdfGraphGrant::Default {
            role: Role::ReadOnly,
        }]),
    );
    let mut query = EntityHistoryQuery::new(default.entity_id);
    query.graph = HistoryGraph::Rdf(None);
    let page = reader.history_after(&query, None, 2, 4096).unwrap();
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].triple_graph, None);
    let lpg_reader =
        db.session_with_identity(Identity::new("lpg-root", [Role::ReadOnly]).with_grants([
            grafeo_engine::auth::Grant::new(GraphPath::root(), Role::ReadOnly),
        ]));
    assert!(
        lpg_reader.history_after(&query, None, 2, 4096).is_err(),
        "an LPG root grant cannot read RDF default history"
    );
    query.entity_id = named.entity_id;
    query.graph = HistoryGraph::Rdf(Some("urn:g".into()));
    assert!(reader.history_after(&query, None, 2, 4096).is_err());
    let page = db.history_after(&query, None, 2, 4096).unwrap();
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].triple_graph.as_deref(), Some("urn:g"));
}
