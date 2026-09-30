//! Integration tests for snapshot export/import.

#[cfg(feature = "lpg")]
use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

#[cfg(feature = "lpg")]
type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[test]
#[cfg(feature = "lpg")]
fn export_import_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(restored.node_count(), 0);
    assert_eq!(restored.edge_count(), 0);
}

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
}

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_edges() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", "Alix".into())
        .expect("set node property");
    let gus = db.create_node(&["Person"]);
    db.set_node_property(gus, "name", "Gus".into())
        .expect("set node property");
    db.create_edge(alix, gus, "KNOWS");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);
    assert_eq!(restored.edge_count(), 1);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Item {name: 'Widget', price: 9.99, active: true})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (i:Item) RETURN i.name, i.price, i.active")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn import_rejects_invalid_data() {
    let result = GrafeoDB::import_snapshot(b"not a valid snapshot");
    assert!(result.is_err());
}

#[test]
#[cfg(feature = "lpg")]
fn snapshot_round_trip_schema() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", "Alix".into())
        .expect("set node property");
    let gus = db.create_node(&["Person"]);
    db.set_node_property(gus, "name", "Gus".into())
        .expect("set node property");
    db.create_edge(alix, gus, "KNOWS");

    let schema_before = db.schema();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let schema_after = restored.schema();

    // Both schemas should report the same label/edge info
    let fmt_before = format!("{schema_before:?}");
    let fmt_after = format!("{schema_after:?}");
    assert_eq!(fmt_before, fmt_after);
}

// --- Edge property round-trip ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_edge_properties() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let edge = db.create_edge(a, b, "KNOWS");
    db.set_edge_property(edge, "since", Value::Int64(2020))
        .expect("set edge property");
    db.set_edge_property(edge, "strength", Value::Float64(0.95))
        .expect("set edge property");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH ()-[e:KNOWS]->() RETURN e.since, e.strength")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(2020));
    assert_eq!(result.rows()[0][1], Value::Float64(0.95));
}

// --- Multi-label nodes ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_multiple_labels() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Person", "Employee"]);
    db.create_node(&["Person", "Manager"]);
    db.create_node(&["Animal"]);

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 3);

    let session = restored.session();
    let persons = session.execute("MATCH (p:Person) RETURN p").unwrap();
    assert_eq!(persons.rows().len(), 2);

    let employees = session.execute("MATCH (e:Employee) RETURN e").unwrap();
    assert_eq!(employees.rows().len(), 1);

    let managers = session.execute("MATCH (m:Manager) RETURN m").unwrap();
    assert_eq!(managers.rows().len(), 1);

    let animals = session.execute("MATCH (a:Animal) RETURN a").unwrap();
    assert_eq!(animals.rows().len(), 1);
}

// --- Temporal Value types ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_temporal_values() {
    use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};

    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Temporal"]);

    let date = Date::from_ymd(2025, 6, 15).unwrap();
    let time = Time::from_hms(14, 30, 0).unwrap();
    let timestamp = Timestamp::from_secs(1_700_000_000);
    let duration = Duration::new(1, 15, 3_600_000_000_000); // 1 month, 15 days, 1 hour
    let zoned = ZonedDatetime::from_timestamp_offset(Timestamp::from_secs(1_700_000_000), 3600);

    db.set_node_property(id, "date_val", Value::Date(date))
        .expect("set node property");
    db.set_node_property(id, "time_val", Value::Time(time))
        .expect("set node property");
    db.set_node_property(id, "ts_val", Value::Timestamp(timestamp))
        .expect("set node property");
    db.set_node_property(id, "dur_val", Value::Duration(duration))
        .expect("set node property");
    db.set_node_property(id, "zdt_val", Value::ZonedDatetime(zoned))
        .expect("set node property");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH (t:Temporal) RETURN t.date_val, t.time_val, t.ts_val, t.dur_val, t.zdt_val")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Date(date));
    assert_eq!(result.rows()[0][1], Value::Time(time));
    assert_eq!(result.rows()[0][2], Value::Timestamp(timestamp));
    assert_eq!(result.rows()[0][3], Value::Duration(duration));
    assert_eq!(result.rows()[0][4], Value::ZonedDatetime(zoned));
}

// --- All scalar Value types ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_all_value_types() {
    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Test"]);
    db.set_node_property(id, "str_val", Value::String("hello".into()))
        .expect("set node property");
    db.set_node_property(id, "int_val", Value::Int64(42))
        .expect("set node property");
    db.set_node_property(id, "float_val", Value::Float64(9.81))
        .expect("set node property");
    db.set_node_property(id, "bool_val", Value::Bool(true))
        .expect("set node property");
    db.set_node_property(id, "null_val", Value::Null)
        .expect("set node property");
    db.set_node_property(
        id,
        "bytes_val",
        Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF].into()),
    )
    .expect("set node property");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH (t:Test) RETURN t.str_val, t.int_val, t.float_val, t.bool_val, t.null_val, t.bytes_val")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("hello".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(42));
    assert_eq!(result.rows()[0][2], Value::Float64(9.81));
    assert_eq!(result.rows()[0][3], Value::Bool(true));
    assert_eq!(result.rows()[0][4], Value::Null);
    assert_eq!(
        result.rows()[0][5],
        Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF].into())
    );
}

// --- List and Map values ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_collection_values() {
    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Test"]);
    db.set_node_property(
        id,
        "tags",
        Value::List(vec![Value::String("a".into()), Value::String("b".into())].into()),
    )
    .expect("set node property");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session.execute("MATCH (t:Test) RETURN t.tags").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::List(vec![Value::String("a".into()), Value::String("b".into()),].into())
    );
}

// --- Multiple edge types ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_multiple_edge_types() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let c = db.create_node(&["Company"]);

    db.create_edge(a, b, "KNOWS");
    db.create_edge(a, c, "WORKS_AT");
    db.create_edge(b, c, "WORKS_AT");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 3);
    assert_eq!(restored.edge_count(), 3);

    let session = restored.session();
    let knows = session.execute("MATCH ()-[e:KNOWS]->() RETURN e").unwrap();
    assert_eq!(knows.rows().len(), 1);

    let works = session
        .execute("MATCH ()-[e:WORKS_AT]->() RETURN e")
        .unwrap();
    assert_eq!(works.rows().len(), 2);
}

// --- Nodes with no properties ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_empty_property_nodes() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Empty"]);
    db.create_node(&["Empty"]);

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);

    let session = restored.session();
    let result = session.execute("MATCH (e:Empty) RETURN e").unwrap();
    assert_eq!(result.rows().len(), 2);
}

// --- Moderate dataset ---

#[test]
#[cfg(feature = "lpg")]
fn export_import_moderate_dataset() {
    let db = GrafeoDB::new_in_memory();

    // Create 100 nodes with properties
    let mut ids = Vec::new();
    for i in 0..100 {
        let id = db.create_node(&["Item"]);
        db.set_node_property(id, "index", Value::Int64(i))
            .expect("set node property");
        db.set_node_property(id, "name", Value::String(format!("item_{i}").into()))
            .expect("set node property");
        ids.push(id);
    }

    // Create 50 edges
    for i in 0..50 {
        db.create_edge(ids[i], ids[i + 50], "LINKS_TO");
    }

    let bytes = db.export_snapshot().unwrap();
    assert!(!bytes.is_empty());

    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 100);
    assert_eq!(restored.edge_count(), 50);

    // Verify a sample property survived
    let session = restored.session();
    let result = session
        .execute("MATCH (i:Item) WHERE i.index = 42 RETURN i.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("item_42".into()));
}

// --- Import empty bytes ---

#[test]
fn import_rejects_empty_bytes() {
    let result = GrafeoDB::import_snapshot(&[]);
    assert!(result.is_err());
}

// --- Snapshot version mismatch ---

#[test]
#[cfg(feature = "lpg")]
fn import_rejects_unsupported_version() {
    // Export a valid snapshot, then tamper with the version byte to trigger the
    // "unsupported snapshot version" error path.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let mut bytes = db.export_snapshot().unwrap();
    // The first byte in bincode standard encoding for a struct starting with
    // a u8 field is the version byte itself.
    bytes[0] = 99; // Set to invalid version

    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for tampered snapshot"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("snapshot")
                    || err_msg.contains("unsupported")
                    || err_msg.contains("import"),
                "Expected snapshot error, got: {err_msg}"
            );
        }
    }
}

// --- Double export produces identical bytes ---

#[test]
#[cfg(feature = "lpg")]
fn double_export_is_deterministic() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let bytes1 = db.export_snapshot().unwrap();
    let bytes2 = db.export_snapshot().unwrap();
    assert_eq!(bytes1, bytes2);
}

// --- Edge reference validation ---

/// Builds a malformed current image without relying on recovery insertion:
/// that API deliberately does not create an edge with an absent endpoint.
#[cfg(feature = "lpg")]
fn current_snapshot_with_dangling_endpoint(source_is_missing: bool) -> Vec<u8> {
    let db = GrafeoDB::new_in_memory();
    let source = db.create_node(&["Source"]);
    let destination = db.create_node(&["Destination"]);
    let edge_type = "DANGLING_ENDPOINT_VALIDATION_WITNESS";
    let edge = db.create_edge(source, destination, edge_type);
    let mut bytes = db.export_snapshot().expect("genuine current writer");
    assert_eq!(
        GrafeoDB::import_snapshot(&bytes)
            .expect("valid control image")
            .edge_count(),
        1
    );

    // A tuple has the same current bincode field sequence as SnapshotEdge.
    // Include its retained history to locate exactly one complete edge record,
    // not an incidental copy of the type name in catalog metadata.
    let lifetimes: Vec<_> = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_edge_history(edge)
        .into_iter()
        .map(|(created, deleted, _)| (created, deleted))
        .collect();
    let properties: Vec<(String, Vec<(EpochId, Vec<u8>)>)> = Vec::new();
    let encode_edge = |src, dst| {
        bincode::serde::encode_to_vec(
            (edge, src, dst, edge_type, &lifetimes, &properties),
            bincode::config::standard(),
        )
        .expect("current edge field encoding")
    };
    let original = encode_edge(source, destination);
    let positions: Vec<_> = bytes
        .windows(original.len())
        .enumerate()
        .filter_map(|(offset, window)| (window == original).then_some(offset))
        .collect();
    assert_eq!(
        positions.len(),
        1,
        "fixture must identify exactly one encoded edge"
    );
    let absent = NodeId::new(999);
    let corrupt = if source_is_missing {
        encode_edge(absent, destination)
    } else {
        encode_edge(source, absent)
    };
    let offset = positions[0];
    bytes.splice(offset..offset + original.len(), corrupt);
    let body_len = u64::try_from(bytes.len() - 13).expect("bounded current snapshot");
    bytes[5..13].copy_from_slice(&body_len.to_le_bytes());
    let info = grafeo_engine::snapshot_info(&bytes).expect("well-formed current image");
    assert_eq!((info.node_count, info.edge_count), (2, 1));
    bytes
}

#[test]
#[cfg(feature = "lpg")]
fn import_rejects_dangling_edge_source() {
    let bytes = current_snapshot_with_dangling_endpoint(true);
    let error = GrafeoDB::import_snapshot(&bytes)
        .err()
        .expect("dangling source must be rejected");
    assert!(
        error.to_string().contains("non-existent source node"),
        "{error}"
    );
}

#[test]
#[cfg(feature = "lpg")]
fn import_rejects_dangling_edge_destination() {
    let bytes = current_snapshot_with_dangling_endpoint(false);
    let error = GrafeoDB::import_snapshot(&bytes)
        .err()
        .expect("dangling destination must be rejected");
    assert!(
        error.to_string().contains("non-existent destination node"),
        "{error}"
    );
}

#[cfg(feature = "lpg")]
fn replace_unique_snapshot_record(
    bytes: &mut Vec<u8>,
    original: &[u8],
    replacement: Vec<u8>,
) -> TestResult {
    let mut positions = bytes
        .windows(original.len())
        .enumerate()
        .filter_map(|(offset, window)| (window == original).then_some(offset));
    let offset = positions
        .next()
        .ok_or("current record was not found in writer output")?;
    assert!(
        positions.next().is_none(),
        "fixture must identify exactly one complete record"
    );
    bytes.splice(offset..offset + original.len(), replacement);
    let body_len = u64::try_from(bytes.len() - 13).expect("bounded current snapshot");
    bytes[5..13].copy_from_slice(&body_len.to_le_bytes());
    Ok(())
}

#[test]
#[cfg(feature = "lpg")]
fn import_rejects_duplicate_node_ids() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let first = db.create_node(&["A"]);
    let second = db.create_node(&["DUPLICATE_NODE_ID_VALIDATION_WITNESS"]);
    let mut bytes = db.export_snapshot()?;
    assert_eq!(GrafeoDB::import_snapshot(&bytes)?.node_count(), 2);

    let lifetimes: Vec<_> = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_node_history(second)
        .into_iter()
        .map(|(created, deleted, _)| (created, deleted))
        .collect();
    let label_versions: Vec<_> = grafeo_engine::database::testing::root_lpg_store(&db)
        .node_label_history(second)
        .into_iter()
        .map(|(epoch, labels)| {
            (
                epoch,
                labels
                    .into_iter()
                    .map(|label| label.to_string())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let properties: Vec<(String, Vec<(EpochId, Vec<u8>)>)> = Vec::new();
    // Change only the record ID, retaining the real writer's label and
    // structural history, catalog, identity and allocator state.
    let encode_node = |id| {
        bincode::serde::encode_to_vec(
            (id, &lifetimes, &label_versions, &properties),
            bincode::config::standard(),
        )
    };
    replace_unique_snapshot_record(&mut bytes, &encode_node(second)?, encode_node(first)?)?;
    let error = GrafeoDB::import_snapshot(&bytes)
        .err()
        .ok_or("Expected error for duplicate node ID")?;
    let err = error.to_string();
    assert!(
        err.contains("duplicate node ID"),
        "Expected duplicate node error, got: {err}"
    );
    Ok(())
}

#[test]
#[cfg(feature = "lpg")]
fn import_rejects_duplicate_edge_ids() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let source = db.create_node(&["Source"]);
    let destination = db.create_node(&["Destination"]);
    let edge_type = "DUPLICATE_EDGE_ID_VALIDATION_WITNESS";
    let first = db.create_edge(source, destination, edge_type);
    let second = db.create_edge(source, destination, edge_type);
    let mut bytes = db.export_snapshot()?;
    assert_eq!(GrafeoDB::import_snapshot(&bytes)?.edge_count(), 2);

    let lifetimes: Vec<_> = grafeo_engine::database::testing::root_lpg_store(&db)
        .get_edge_history(second)
        .into_iter()
        .map(|(created, deleted, _)| (created, deleted))
        .collect();
    let properties: Vec<(String, Vec<(EpochId, Vec<u8>)>)> = Vec::new();
    let encode_edge = |id| {
        bincode::serde::encode_to_vec(
            (id, source, destination, edge_type, &lifetimes, &properties),
            bincode::config::standard(),
        )
    };
    replace_unique_snapshot_record(&mut bytes, &encode_edge(second)?, encode_edge(first)?)?;
    let error = GrafeoDB::import_snapshot(&bytes)
        .err()
        .ok_or("Expected error for duplicate edge ID")?;
    let err = error.to_string();
    assert!(
        err.contains("duplicate edge ID"),
        "Expected duplicate edge error, got: {err}"
    );
    Ok(())
}

// =========================================================================
// Named Graph Snapshot Tests
// =========================================================================

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("USE GRAPH analytics").unwrap();
    session
        .execute("INSERT (:KPI {name: 'pageviews', count: 42})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    // Default graph
    assert_eq!(restored.node_count(), 1);
    let session2 = restored.session();
    let result = session2.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

    // Named graph
    session2.execute("USE GRAPH analytics").unwrap();
    let result = session2
        .execute("MATCH (m:KPI) RETURN m.name, m.count")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("pageviews".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(42));
}

#[test]
#[cfg(feature = "lpg")]
fn export_import_preserves_multiple_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE GRAPH alpha").unwrap();
    session.execute("USE GRAPH alpha").unwrap();
    session.execute("INSERT (:Item {name: 'Widget'})").unwrap();

    session.execute("USE GRAPH default").unwrap();
    session.execute("CREATE GRAPH beta").unwrap();
    session.execute("USE GRAPH beta").unwrap();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session2 = restored.session();

    session2.execute("USE GRAPH alpha").unwrap();
    let result = session2.execute("MATCH (i:Item) RETURN i.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Widget".into()));

    session2.execute("USE GRAPH beta").unwrap();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(result.rows()[1][0], Value::String("Berlin".into()));
}

#[test]
#[cfg(feature = "lpg")]
fn restore_snapshot_includes_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("CREATE GRAPH metrics").unwrap();
    session.execute("USE GRAPH metrics").unwrap();
    session.execute("INSERT (:KPI {name: 'clicks'})").unwrap();

    let snapshot = db.export_snapshot().unwrap();

    // Modify: add more data
    session.execute("USE GRAPH default").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Restore
    db.restore_snapshot(&snapshot).unwrap();

    assert_eq!(db.node_count(), 1, "default graph restored to 1 node");

    let session2 = db.session();
    session2.execute("USE GRAPH metrics").unwrap();
    let result = session2.execute("MATCH (m:KPI) RETURN m.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("clicks".into()));
}

#[test]
fn import_v1_snapshot_is_rejected() {
    // Reject the predecessor header before attempting any payload decode.
    let result = GrafeoDB::import_snapshot(&[1]);
    assert!(result.is_err(), "V1 snapshots should be rejected");
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("unsupported snapshot version"),
        "Expected version error, got: {err}"
    );
}

#[test]
#[cfg(feature = "lpg")]
fn to_memory_copies_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("CREATE GRAPH backup").unwrap();
    session.execute("USE GRAPH backup").unwrap();
    session
        .execute("INSERT (:Archive {date: '2025-01-01'})")
        .unwrap();

    let copy = db.to_memory().unwrap();

    // Default graph copied
    assert_eq!(copy.node_count(), 1);

    // Named graph copied
    let session2 = copy.session();
    session2.execute("USE GRAPH backup").unwrap();
    let result = session2.execute("MATCH (a:Archive) RETURN a.date").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("2025-01-01".into()));

    // Independence: mutating original doesn't affect copy
    session
        .execute("INSERT (:Archive {date: '2025-02-01'})")
        .unwrap();
    let result2 = session2.execute("MATCH (a:Archive) RETURN a.date").unwrap();
    assert_eq!(result2.rows().len(), 1, "copy should still have 1 node");
}

// =========================================================================
// RDF Snapshot Tests
// =========================================================================

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod rdf_snapshots {
    use grafeo_common::types::Value;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    #[test]
    fn export_import_preserves_rdf_triples() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();

        let bytes = db.export_snapshot().unwrap();
        let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

        let session2 = restored.session();
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name } ORDER BY ?name")
            .unwrap();
        assert_eq!(result.rows().len(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn export_import_preserves_rdf_named_graphs() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }
                }"#,
            )
            .unwrap();

        let bytes = db.export_snapshot().unwrap();
        let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

        let session2 = restored.session();

        // Default graph
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

        // Named graph
        let result = session2
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn restore_snapshot_includes_rdf_data() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                }"#,
            )
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();

        // Add more data
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();

        // Restore: should go back to just Alix
        db.restore_snapshot(&snapshot).unwrap();

        let session2 = db.session();
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap();
        assert_eq!(result.rows().len(), 1, "restore should revert to snapshot");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn to_memory_copies_rdf_data() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }
                }"#,
            )
            .unwrap();

        let copy = db.to_memory().unwrap();

        let session2 = copy.session();
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap();
        assert_eq!(result.rows().len(), 1, "default RDF graph copied");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

        let result = session2
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(result.rows().len(), 1, "named RDF graph copied");
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));

        // Independence: mutating original doesn't affect copy
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/mia> <http://ex.org/name> "Mia" .
                }"#,
            )
            .unwrap();
        let result = session2
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(result.rows().len(), 1, "copy should be independent");
    }

    #[test]
    fn export_import_preserves_typed_rdf_literals() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
                    <http://ex.org/alix> <http://ex.org/greeting> "Bonjour"@fr .
                }"#,
            )
            .unwrap();

        let bytes = db.export_snapshot().unwrap();
        let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

        let session2 = restored.session();
        let result = session2
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/alix> ?p ?o } ORDER BY ?o")
            .unwrap();
        assert_eq!(
            result.rows().len(),
            2,
            "typed and lang literals should survive"
        );
    }

    fn rdf_history_fixture() -> GrafeoDB {
        use grafeo_core::graph::rdf::{Term, Triple};
        let db = rdf_db();
        db.execute_sparql("CREATE GRAPH <urn:empty>").unwrap();
        db.execute_sparql(
            r#"INSERT DATA {
                <urn:default> <urn:p> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
                GRAPH <urn:versions> { <urn:named> <urn:p> "Bonjour"@fr . }
            }"#,
        )
        .unwrap();
        db.execute_sparql("DROP GRAPH <urn:versions>").unwrap();
        db.execute_sparql("CREATE GRAPH <urn:versions>").unwrap();
        db.execute_sparql(r#"INSERT DATA { GRAPH <urn:versions> { <urn:new> <urn:p> "new" } }"#)
            .unwrap();
        db.insert_rdf_valid_tai_ns(
            [Triple::new(
                Term::iri("urn:valid"),
                Term::iri("urn:p"),
                Term::literal("interval"),
            )],
            -7,
            13,
        )
        .unwrap();
        db
    }

    #[test]
    fn portable_rdf_history_identity_and_fork_are_exact_across_profiles() {
        let source = rdf_history_fixture();
        let history = source.rdf_dataset_history().unwrap();
        let artifact = source.export_snapshot_artifact().unwrap();
        let restored = GrafeoDB::import_snapshot_artifact(&artifact).unwrap();
        let restored_history = restored.rdf_dataset_history().unwrap();
        assert_eq!(restored.world_identity(), source.world_identity());
        assert_eq!(restored.current_epoch(), source.current_epoch());
        assert_eq!(restored.rdf_store_commit_epoch(), source.current_epoch());
        assert_eq!(restored_history.graph_lives(), history.graph_lives());
        assert_eq!(restored_history.quad_versions(), history.quad_versions());
        assert_eq!(
            restored_history.next_graph_incarnation(),
            history.next_graph_incarnation()
        );
        assert_eq!(restored.export_snapshot().unwrap(), artifact.bytes());
        assert_eq!(history.graph_lives().len(), 3);
        assert_eq!(history.quad_versions().len(), 4);

        let fork = source.to_memory().unwrap();
        assert_ne!(fork.store_id(), source.store_id());
        assert_eq!(fork.current_epoch(), source.current_epoch());
        let fork_history = fork.rdf_dataset_history().unwrap();
        assert_eq!(fork_history.graph_lives(), history.graph_lives());
        assert_eq!(
            fork_history.quad_versions().len(),
            history.quad_versions().len()
        );
        for original in history.quad_versions() {
            let copied = fork_history
                .quad_versions()
                .iter()
                .find(|row| {
                    row.quad() == original.quad()
                        && row.tx() == original.tx()
                        && row.graph_incarnation() == original.graph_incarnation()
                })
                .unwrap();
            assert_eq!(copied.valid(), original.valid());
            assert_ne!(copied.statement(), original.statement());
        }
        fork.execute_sparql(r#"INSERT DATA { <urn:fork-only> <urn:p> "independent" }"#)
            .unwrap();
        assert_eq!(source.export_snapshot().unwrap(), artifact.bytes());

        // Qualification exchanges these same current bytes between separately
        // compiled full and RDF-only binaries. Ordinary runs still execute all
        // history, lineage and independent-write assertions above.
        if let Some(path) = std::env::var_os("GRAFEO_RDF_PORTABLE_EXPORT") {
            use std::io::Write;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap()
                .write_all(artifact.bytes())
                .unwrap();
        }
        if let Some(path) = std::env::var_os("GRAFEO_RDF_PORTABLE_IMPORT") {
            let bytes = std::fs::read(path).unwrap();
            let cross_profile = GrafeoDB::import_snapshot(&bytes).unwrap();
            assert_eq!(cross_profile.config().graph_model, GraphModel::Rdf);
            assert_eq!(cross_profile.export_snapshot().unwrap(), bytes);
            let cross_history = cross_profile.rdf_dataset_history().unwrap();
            assert_eq!(cross_history.graph_lives(), history.graph_lives());
            assert_eq!(
                cross_history.quad_versions().len(),
                history.quad_versions().len()
            );
            for original in history.quad_versions() {
                assert!(cross_history.quad_versions().iter().any(|row| row.quad()
                    == original.quad()
                    && row.tx() == original.tx()
                    && row.valid() == original.valid()
                    && row.graph_incarnation() == original.graph_incarnation()));
            }
        }
    }

    #[test]
    fn portable_rdf_restore_rejects_corruption_and_busy_targets_before_replacement() {
        let source = rdf_history_fixture();
        let bytes = source.export_snapshot().unwrap();
        let target = rdf_db();
        for id in 0..10 {
            target
                .execute_sparql(&format!(
                    "INSERT DATA {{ <urn:target:{id}> <urn:p> \"target\" }}"
                ))
                .unwrap();
        }
        assert!(target.current_epoch() > source.current_epoch());
        let before = target.export_snapshot().unwrap();
        let before_epoch = target.current_epoch();
        let mut trailing = bytes.clone();
        trailing.push(0);
        let mut invalid_cdc = bytes.clone();
        let last = invalid_cdc.len() - 1;
        invalid_cdc[last] ^= 0xff;
        for corrupt in [
            &bytes[..bytes.len() - 1],
            trailing.as_slice(),
            invalid_cdc.as_slice(),
            &[99],
        ] {
            assert!(GrafeoDB::import_snapshot(corrupt).is_err());
            assert!(target.restore_snapshot(corrupt).is_err());
            assert_eq!(target.export_snapshot().unwrap(), before);
        }
        if let Some(directory) = std::env::var_os("GRAFEO_RDF_PORTABLE_BAD_IMAGES") {
            for family in ["text", "vector", "projection"] {
                let malformed =
                    std::fs::read(std::path::Path::new(&directory).join(format!("{family}.bin")))
                        .unwrap();
                assert!(GrafeoDB::import_snapshot(&malformed).is_err(), "{family}");
                assert!(target.restore_snapshot(&malformed).is_err(), "{family}");
                assert_eq!(target.export_snapshot().unwrap(), before);
            }
        }
        let mut reader = target.session();
        reader.begin_transaction().unwrap();
        assert!(target.restore_snapshot(&bytes).is_err());
        assert_eq!(target.current_epoch(), before_epoch);
        reader.rollback().unwrap();
        assert_eq!(target.export_snapshot().unwrap(), before);
        let mut writer = source.session();
        writer.begin_transaction().unwrap();
        writer
            .execute_sparql(r#"INSERT DATA { <urn:pending> <urn:p> "pending" }"#)
            .unwrap();
        assert!(source.export_snapshot().is_err());
        writer.rollback().unwrap();
        target.restore_snapshot(&bytes).unwrap();
        assert_eq!(target.export_snapshot().unwrap(), bytes);
        assert_eq!(target.world_identity(), source.world_identity());
        assert_eq!(target.current_epoch(), source.current_epoch());
        assert_eq!(target.rdf_store_commit_epoch(), source.current_epoch());
        reader
            .execute_sparql("SELECT ?s WHERE { ?s <urn:p> ?o }")
            .unwrap();
        target
            .execute_sparql(r#"INSERT DATA { <urn:after-restore> <urn:p> "new" }"#)
            .unwrap();
        assert!(target.current_epoch() > source.current_epoch());
        assert_eq!(source.export_snapshot().unwrap(), bytes);
    }

    #[test]
    fn portable_rdf_artifact_rejects_a_valid_seal_over_false_identity() {
        use grafeo_engine::{SnapshotArtifact, StoreId, WorldCutDescriptor};
        let artifact = rdf_history_fixture().export_snapshot_artifact().unwrap();
        let authentic = artifact.cut().descriptor();
        let forged = WorldCutDescriptor::new(
            StoreId::generate().unwrap(),
            authentic.epoch(),
            authentic.graph_model(),
            authentic.formats().to_vec(),
            authentic.schema().clone(),
            authentic.projections().to_vec(),
            authentic.history(),
        )
        .unwrap();
        let forged = SnapshotArtifact::new(artifact.bytes().to_vec(), forged).unwrap();
        forged.verify().unwrap();
        assert!(GrafeoDB::import_snapshot_artifact(&forged).is_err());
    }

    #[cfg(feature = "cdc")]
    #[test]
    fn portable_rdf_cdc_resumes_after_exact_import_and_restore_but_not_fork() {
        use grafeo_common::utils::error::ErrorCode;
        let source = GrafeoDB::with_config(
            Config::in_memory()
                .with_graph_model(GraphModel::Rdf)
                .with_cdc(),
        )
        .unwrap();
        for subject in ["a", "b", "c"] {
            source
                .execute_sparql(&format!(
                    "INSERT DATA {{ GRAPH <urn:feed> {{ <urn:{subject}> <urn:p> \"value\" }} }}"
                ))
                .unwrap();
        }
        let first = source.changes_after(None, 1, 64 * 1024).unwrap();
        assert_eq!(first.events.len(), 1);
        let expected = source
            .changes_after(Some(&first.next), 4, 64 * 1024)
            .unwrap();
        assert_eq!(expected.events.len(), 2);
        let encode = |events: &[grafeo_engine::cdc::ChangeEvent]| {
            bincode::serde::encode_to_vec(events, bincode::config::standard()).unwrap()
        };
        let bytes = source.export_snapshot().unwrap();
        let imported = GrafeoDB::import_snapshot(&bytes).unwrap();
        let target = rdf_db();
        target.restore_snapshot(&bytes).unwrap();
        for copy in [&imported, &target] {
            let resumed = copy.changes_after(Some(&first.next), 4, 64 * 1024).unwrap();
            assert_eq!(resumed.next, expected.next);
            assert_eq!(encode(&resumed.events), encode(&expected.events));
            assert_eq!(copy.export_snapshot().unwrap(), bytes);
        }
        let fork = source.to_memory().unwrap();
        let error = fork
            .changes_after(Some(&first.next), 4, 64 * 1024)
            .unwrap_err();
        assert_eq!(error.error_code(), ErrorCode::CursorForeign);
        assert_eq!(
            fork.changes_after(None, 4, 64 * 1024).unwrap().events.len(),
            3
        );
    }

    #[cfg(feature = "wal")]
    #[test]
    fn portable_rdf_restore_refuses_a_wal_backed_target_without_mutation() {
        let bytes = rdf_history_fixture().export_snapshot().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("portable-target.grafeo");
        let target =
            GrafeoDB::with_config(Config::persistent(&path).with_graph_model(GraphModel::Rdf))
                .unwrap();
        target
            .execute_sparql(r#"INSERT DATA { <urn:target> <urn:p> "kept" }"#)
            .unwrap();
        let before = target.export_snapshot().unwrap();
        assert!(target.restore_snapshot(&bytes).is_err());
        assert_eq!(target.export_snapshot().unwrap(), before);
        target.close().unwrap();
        let reopened = GrafeoDB::open(&path).unwrap();
        assert_eq!(reopened.export_snapshot().unwrap(), before);
    }
}

// ---------------------------------------------------------------------------
// Snapshot forward-compatibility (T3-04)
// ---------------------------------------------------------------------------

#[test]
fn import_unknown_snapshot_version_returns_clear_error() {
    // Unknown future headers are refused without a fabricated payload.
    let result = GrafeoDB::import_snapshot(&[99]);
    match result {
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("version") || err.contains("unsupported") || err.contains("99"),
                "error should mention version issue, got: {err}"
            );
        }
        Ok(_) => panic!("importing an unknown snapshot version should error"),
    }
}

#[test]
#[cfg(feature = "lpg")]
fn import_truncated_snapshot_returns_error() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Test"]);
    let bytes = db.export_snapshot().unwrap();

    // Truncate to half
    let truncated = &bytes[..bytes.len() / 2];
    let result = GrafeoDB::import_snapshot(truncated);
    assert!(
        result.is_err(),
        "importing a truncated snapshot should error"
    );
}

#[test]
fn import_empty_bytes_returns_error() {
    let result = GrafeoDB::import_snapshot(&[]);
    assert!(result.is_err(), "importing empty bytes should error");
}

// --- Multi-schema round-trip (ISO/IEC 39075 catalog hierarchy) ---

#[test]
#[cfg(feature = "lpg")]
fn snapshot_round_trip_multi_schema_isolation() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();

    s.execute("CREATE SCHEMA reporting").unwrap();
    s.execute("CREATE SCHEMA social").unwrap();

    s.execute("SESSION SET SCHEMA reporting").unwrap();
    s.execute("INSERT (:Report {title: 'Q1'})").unwrap();

    s.execute("SESSION SET SCHEMA social").unwrap();
    s.execute("INSERT (:Friend {name: 'Alix'})").unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let r = restored.session();

    // Schemas survive
    let schemas = r.execute("SHOW SCHEMAS").unwrap();
    assert_eq!(schemas.row_count(), 2, "both schemas should be restored");

    // Cross-schema isolation is preserved
    r.execute("SESSION SET SCHEMA reporting").unwrap();
    let reports = r.execute("MATCH (n:Report) RETURN n").unwrap();
    assert_eq!(reports.row_count(), 1, "reporting should have 1 Report");
    let leaked = r.execute("MATCH (n:Friend) RETURN n").unwrap();
    assert_eq!(
        leaked.row_count(),
        0,
        "social data must not leak into reporting after restore"
    );

    r.execute("SESSION SET SCHEMA social").unwrap();
    let friends = r.execute("MATCH (n:Friend) RETURN n").unwrap();
    assert_eq!(friends.row_count(), 1, "social should have 1 Friend");
    let leaked = r.execute("MATCH (n:Report) RETURN n").unwrap();
    assert_eq!(
        leaked.row_count(),
        0,
        "reporting data must not leak into social after restore"
    );
}

#[test]
#[cfg(feature = "lpg")]
fn snapshot_round_trip_named_graph_within_schema() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();

    s.execute("CREATE SCHEMA reports").unwrap();
    s.execute("SESSION SET SCHEMA reports").unwrap();
    s.execute("CREATE GRAPH quarterly").unwrap();
    s.execute("SESSION SET GRAPH quarterly").unwrap();
    s.execute("INSERT (:Row {q: 1})").unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let r = restored.session();

    r.execute("SESSION SET SCHEMA reports").unwrap();
    let graphs = r.execute("SHOW GRAPHS").unwrap();
    assert_eq!(
        graphs.row_count(),
        1,
        "schema-scoped named graph should survive round-trip"
    );

    r.execute("SESSION SET GRAPH quarterly").unwrap();
    let rows = r.execute("MATCH (n:Row) RETURN n.q").unwrap();
    assert_eq!(
        rows.row_count(),
        1,
        "data in schema-scoped named graph should survive round-trip"
    );
}
