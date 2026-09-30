//! Integration tests for `GrafeoDB::open_multi` — loading multiple
//! snapshot blobs into one in-memory database.
//!
//! Handcrafted conflict fixtures start from a real current writer image, then
//! replace only entity rows and allocator counters through a local wire DTO.
//! This permits cross-shard and invalid endpoints without bypassing current
//! catalog, identity or history admission.

#![cfg(feature = "lpg")]

use grafeo_common::types::{EdgeId, EpochId, GraphPath, NodeId, Value, WorldIdentityMetadataV1};
use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

// --------------------------------------------------------------------
// TestSnapshot mirror — lets us craft snapshot bytes with hand-picked
// NodeIds / EdgeIds for the cross-snapshot conflict tests. This is the actual
// current wire, not a predecessor reader. The writer's exact catalog, index
// and identity payloads are preserved. Property values remain encoded bytes.
// --------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct TestSnapshot {
    version: u8,
    epoch: u64,
    graph_model: u8,
    world_identity: WorldIdentityMetadataV1,
    next_graph_incarnation_id: u64,
    graphs: Vec<TestLpgGraph>,
    catalog_state: Vec<u8>,
    text_indexes: Vec<u8>,
    vector_indexes: Vec<u8>,
    rdf_lpg_projections: Vec<u8>,
    rdf_dataset_history: Vec<u8>,
    cdc_checkpoint: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestLpgGraph {
    #[serde(with = "grafeo_common::types::graph_path_bytes")]
    path: GraphPath,
    incarnation: grafeo_common::types::GraphIncarnationId,
    next_node_id: u64,
    next_edge_id: u64,
    retained_history_floor: u64,
    nodes: Vec<TestNode>,
    edges: Vec<TestEdge>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestNode {
    id: NodeId,
    lifetimes: Vec<(EpochId, Option<EpochId>)>,
    label_versions: Vec<(EpochId, Vec<String>)>,
    properties: Vec<(String, Vec<(EpochId, Vec<u8>)>)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    lifetimes: Vec<(EpochId, Option<EpochId>)>,
    properties: Vec<(String, Vec<(EpochId, Vec<u8>)>)>,
}

fn current_snapshot() -> TestResult<TestSnapshot> {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))?;
    let bytes = db.export_snapshot()?;
    assert_eq!(grafeo_engine::snapshot_info(&bytes)?.version, 12);
    let (snapshot, consumed): (TestSnapshot, usize) =
        bincode::serde::decode_from_slice(&bytes[13..], bincode::config::standard())?;
    assert_eq!(&bytes[..5], b"\x0cCDC1");
    assert_eq!(consumed, bytes.len() - 13);
    assert_eq!(snapshot.version, 12);
    assert_eq!(snapshot.graph_model, 0);
    let [root] = snapshot.graphs.as_slice() else {
        return Err("empty LPG writer must produce exactly one root graph".into());
    };
    assert_eq!(root.path, GraphPath::root());
    assert!(root.nodes.is_empty() && root.edges.is_empty());
    assert_eq!(
        encode_current(&snapshot)?,
        bytes,
        "the local DTO must preserve the complete current writer seed"
    );
    Ok(snapshot)
}

fn encode_current(snapshot: &TestSnapshot) -> TestResult<Vec<u8>> {
    let body = bincode::serde::encode_to_vec(snapshot, bincode::config::standard())?;
    let mut bytes = b"\x0cCDC1".to_vec();
    bytes.extend_from_slice(&u64::try_from(body.len())?.to_le_bytes());
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

fn next_id(ids: impl Iterator<Item = u64>) -> TestResult<u64> {
    match ids.max() {
        Some(maximum) => maximum
            .checked_add(1)
            .ok_or_else(|| "test ID exhausted".into()),
        None => Ok(0),
    }
}

fn encode_snapshot(nodes: Vec<TestNode>, edges: Vec<TestEdge>) -> TestResult<Vec<u8>> {
    let mut snap = current_snapshot()?;
    snap.graphs = vec![TestLpgGraph {
        incarnation: grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
        path: GraphPath::root(),
        next_node_id: next_id(nodes.iter().map(|node| node.id.as_u64()))?,
        next_edge_id: next_id(edges.iter().map(|edge| edge.id.as_u64()))?,
        retained_history_floor: 0,
        nodes,
        edges,
    }];
    encode_current(&snap)
}

fn node(id: u64, label: &str) -> TestNode {
    TestNode {
        id: NodeId::new(id),
        lifetimes: vec![(EpochId::INITIAL, None)],
        label_versions: vec![(EpochId::INITIAL, vec![label.to_string()])],
        properties: vec![],
    }
}

fn edge(id: u64, src: u64, dst: u64, edge_type: &str) -> TestEdge {
    TestEdge {
        id: EdgeId::new(id),
        src: NodeId::new(src),
        dst: NodeId::new(dst),
        edge_type: edge_type.to_string(),
        lifetimes: vec![(EpochId::INITIAL, None)],
        properties: vec![],
    }
}

#[test]
fn open_multi_rejects_duplicate_node_id_across_snapshots() -> TestResult {
    let a = encode_snapshot(vec![node(1, "Person")], vec![])?;
    let b = encode_snapshot(vec![node(1, "Animal")], vec![])?;

    let result = GrafeoDB::open_multi([a.as_slice(), b.as_slice()]);

    match result {
        Ok(_) => panic!("must reject duplicate NodeId"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("duplicate") && message.contains("NodeId"),
                "error must name the conflict; got: {message}"
            );
        }
    }
    Ok(())
}

#[test]
fn open_multi_with_single_snapshot_matches_data_but_forks_identity() {
    // `open_multi` preserves the graph payload but is intentionally a logical
    // fork, while `import_snapshot` is an exact replica.
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", "Alix".into())
        .expect("set node property");
    let gus = db.create_node(&["Person"]);
    db.set_node_property(gus, "name", "Gus".into())
        .expect("set node property");
    db.create_edge(alix, gus, "KNOWS");

    let bytes = db.export_snapshot().expect("export");
    let via_import = GrafeoDB::import_snapshot(&bytes).expect("import_snapshot");
    let via_multi = GrafeoDB::open_multi([bytes.as_slice()]).expect("open_multi");

    assert_eq!(via_import.node_count(), via_multi.node_count());
    assert_eq!(via_import.edge_count(), via_multi.edge_count());
    assert_eq!(via_import.store_id(), db.store_id());
    assert_ne!(via_multi.store_id(), db.store_id());

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
fn open_multi_resolves_edges_whose_endpoint_lives_in_another_snapshot() -> TestResult {
    // Snapshot A — shared chunk: one UniversalConcept node.
    let a = encode_snapshot(vec![node(10, "UniversalConcept")], vec![])?;

    // Snapshot B — niche chunk: one NicheDescriptor node and an edge
    // pointing at the UniversalConcept node that only exists in A.
    let b = encode_snapshot(
        vec![node(20, "NicheDescriptor")],
        vec![edge(100, 20, 10, "MAPS_TO_CONCEPT")],
    )?;

    let db = GrafeoDB::open_multi([a.as_slice(), b.as_slice()])
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
    Ok(())
}

#[test]
fn open_multi_fork_reidentifies_union_without_rejecting_cross_shard_endpoint() {
    let source = GrafeoDB::new_in_memory();
    let concept = source.create_node(&["UniversalConcept"]);
    let niche = source.create_node(&["NicheDescriptor"]);
    source.create_edge(niche, concept, "MAPS_TO_CONCEPT");

    // Source-side ownership puts the edge in `niche_shard` even though its
    // destination node exists only in `concept_shard`.
    let concept_shard = source.extract_subgraph(&[concept]).unwrap();
    let niche_shard = source.extract_subgraph(&[niche]).unwrap();
    let concept_store_id = concept_shard.store_id();
    let niche_store_id = niche_shard.store_id();
    let concept_bytes = concept_shard.export_snapshot().unwrap();
    let niche_bytes = niche_shard.export_snapshot().unwrap();

    let merged = GrafeoDB::open_multi([concept_bytes.as_slice(), niche_bytes.as_slice()])
        .expect("fork normalization must defer endpoint resolution to the snapshot set");

    assert_ne!(merged.store_id(), concept_store_id);
    assert_ne!(merged.store_id(), niche_store_id);
    assert_eq!(merged.node_count(), 2);
    assert_eq!(merged.edge_count(), 1);
}

#[test]
fn open_multi_rejects_dangling_edge_endpoint() -> TestResult {
    // Snapshot B carries an edge whose `dst` (NodeId 99) is not
    // present in any snapshot. open_multi must reject; otherwise the
    // edge silently becomes orphaned at load time.
    let a = encode_snapshot(vec![node(10, "UniversalConcept")], vec![])?;
    let b = encode_snapshot(
        vec![node(20, "NicheDescriptor")],
        vec![edge(100, 20, 99, "MAPS_TO_CONCEPT")],
    )?;

    let result = GrafeoDB::open_multi([a.as_slice(), b.as_slice()]);
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
    Ok(())
}

#[test]
fn open_multi_rejects_duplicate_edge_id_across_snapshots() -> TestResult {
    let a = encode_snapshot(
        vec![node(1, "Person"), node(2, "Person")],
        vec![edge(100, 1, 2, "KNOWS")],
    )?;
    let b = encode_snapshot(
        vec![node(3, "Person"), node(4, "Person")],
        vec![edge(100, 3, 4, "KNOWS")],
    )?;

    let result = GrafeoDB::open_multi([a.as_slice(), b.as_slice()]);

    match result {
        Ok(_) => panic!("must reject duplicate EdgeId"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains("duplicate") && message.contains("EdgeId"),
                "error must name the conflict; got: {message}"
            );
        }
    }
    Ok(())
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

    let result = GrafeoDB::open_multi([bytes_a.as_slice(), bytes_b.as_slice()]);
    match result {
        Ok(_) => panic!("must reject schema mismatch"),
        Err(e) => {
            let message = e.to_string();
            // Under UnionWithConflictCheck (default), same-name-different-shape
            // types are rejected with "redefines NodeType". The old strict-equality
            // path said "schema does not match". Both communicate the same conflict.
            assert!(
                message.contains("redefines") || message.contains("schema"),
                "error must describe the type conflict; got: {message}"
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

    let merged = GrafeoDB::open_multi([bytes_a.as_slice(), bytes_b.as_slice()])
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
    let empty: &[&[u8]] = &[];
    let result = GrafeoDB::open_multi(empty);
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

    let merged = GrafeoDB::open_multi([bytes.as_slice(), plain_bytes.as_slice()])
        .expect("named graph in only one snapshot is fine");
    let names = merged.list_graphs();
    assert!(
        names.iter().any(|n| n == "g1"),
        "named graph must be restored; got names: {names:?}"
    );
}

#[test]
fn open_multi_unions_property_indexes_across_snapshots() -> TestResult {
    let db_a = GrafeoDB::new_in_memory();
    let owner_a = db_a.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "id".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })?;
    assert_eq!(owner_a.as_u32(), 0);
    let bytes_a = db_a.export_snapshot()?;

    let db_b = GrafeoDB::new_in_memory();
    let slug_request = || grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "slug".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    };
    let conflicting_owner = db_b.create_index(slug_request())?;
    assert_eq!(conflicting_owner, owner_a);
    let conflicting_bytes = db_b.export_snapshot()?;
    let error = GrafeoDB::open_multi([bytes_a.as_slice(), conflicting_bytes.as_slice()])
        .err()
        .ok_or("independently allocated ID0 owners must not be renumbered")?;
    assert!(
        matches!(&error, Error::Serialization(message)
            if message == "open_multi: index owner ID 0 conflicts"),
        "{error}"
    );

    // Re-creation consumes the next non-reusable owner ID through the real
    // public lifecycle, so these exact owners are now disjoint.
    assert!(db_b.drop_index(conflicting_owner)?);
    let owner_b = db_b.create_index(slug_request())?;
    assert_eq!(owner_b.as_u32(), 1);
    let bytes_b = db_b.export_snapshot()?;

    let merged = GrafeoDB::open_multi([bytes_a.as_slice(), bytes_b.as_slice()])?;

    assert!(merged.has_property_index("id"), "id index must be restored");
    assert!(
        merged.has_property_index("slug"),
        "slug index must be restored"
    );
    Ok(())
}

#[test]
fn open_multi_unions_disjoint_schemas_by_default() {
    // Snapshot A — shared chunk: declares UniversalConcept type only.
    let db_a = GrafeoDB::new_in_memory();
    db_a.session()
        .execute("CREATE NODE TYPE UniversalConcept (id STRING, label STRING)")
        .expect("ddl a");
    let bytes_a = db_a.export_snapshot().expect("export a");

    // Snapshot B — niche chunk: declares NicheDescriptor type only.
    let db_b = GrafeoDB::new_in_memory();
    db_b.session()
        .execute("CREATE NODE TYPE NicheDescriptor (id STRING, niche STRING)")
        .expect("ddl b");
    let bytes_b = db_b.export_snapshot().expect("export b");

    let merged = GrafeoDB::open_multi([bytes_a.as_slice(), bytes_b.as_slice()])
        .expect("disjoint schemas must merge (default policy is union)");

    // The merged catalog should know about BOTH types — not just
    // the first snapshot's. Probe via the same DDL-redeclaration
    // pattern used in extract_subgraph_carries_schema_and_indexes:
    // a re-declaration of either type should fail because the
    // catalog already contains it after merge.
    let probe_concept = merged
        .session()
        .execute("CREATE NODE TYPE UniversalConcept (id STRING, label STRING)");
    assert!(
        probe_concept.is_err(),
        "merged catalog must carry snapshot A's UniversalConcept type — \
         re-declaration should fail. Got: {probe_concept:?}"
    );

    let probe_descriptor = merged
        .session()
        .execute("CREATE NODE TYPE NicheDescriptor (id STRING, niche STRING)");
    assert!(
        probe_descriptor.is_err(),
        "merged catalog must carry snapshot B's NicheDescriptor type — \
         re-declaration should fail. Got: {probe_descriptor:?}"
    );
}

#[test]
fn open_multi_restores_max_epoch_across_snapshots() -> TestResult {
    // Real empty transactions advance the committed frontier without entity
    // noise. The writer keeps both the snapshot and exact Catalog7 epochs
    // aligned; changing only the outer wire epoch is intentionally invalid.
    let snapshot_at_epoch = |epoch: u64| -> TestResult<Vec<u8>> {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        for expected in 1..=epoch {
            session.begin_transaction()?;
            assert_eq!(session.commit()?, EpochId::new(expected));
        }
        assert_eq!(db.current_epoch(), EpochId::new(epoch));
        assert_eq!((db.node_count(), db.edge_count()), (0, 0));
        Ok(db.export_snapshot()?)
    };
    let low = snapshot_at_epoch(5)?;
    let high = snapshot_at_epoch(42)?;
    let merged = GrafeoDB::open_multi([low.as_slice(), high.as_slice()])?;

    // The merged DB should sit at the higher epoch; otherwise a later
    // write would clobber high-epoch property history from `high`.
    assert_eq!(
        merged.current_epoch(),
        EpochId::new(42),
        "open_multi must restore epoch as max across snapshots"
    );
    Ok(())
}

#[test]
fn snapshot_info_reports_blob_stats_without_loading() {
    use grafeo_engine::snapshot_info;

    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    db.set_node_property(a, "id", Value::String("a".into()))
        .expect("set node property");
    let b = db.create_node(&["B"]);
    db.create_edge(a, b, "R");
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: Default::default(),
        name: None,
        label: None,
        property: "id".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");

    let bytes = db.export_snapshot().expect("export");
    let info = snapshot_info(&bytes).expect("info");

    // SnapshotInfo reports the canonical wire version emitted by this runtime,
    // not the oldest legacy version accepted by the decoder.
    assert_eq!(
        info.version, 12,
        "current canonical snapshot format version"
    );
    assert_eq!(info.node_count, 2);
    assert_eq!(info.edge_count, 1);
    assert_eq!(info.property_index_count, 1);
    assert_eq!(info.named_graph_count, 0);
}

#[test]
fn open_multi_accepts_vec_of_vecs_and_array_of_slices() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["A"]);
    let bytes = db.export_snapshot().expect("export");

    // Vec<Vec<u8>>
    let owned: Vec<Vec<u8>> = vec![bytes.clone()];
    let m1 = GrafeoDB::open_multi(owned).expect("Vec<Vec<u8>> works");
    assert_eq!(m1.node_count(), 1);

    // Array of &[u8]
    let m2 = GrafeoDB::open_multi([bytes.as_slice()]).expect("array of slices works");
    assert_eq!(m2.node_count(), 1);

    // Original &[&[u8]] still works (backwards-compat).
    let m3 = GrafeoDB::open_multi([bytes.as_slice()]).expect("slice of slices works");
    assert_eq!(m3.node_count(), 1);
}

#[test]
#[cfg(feature = "vector-index")]
fn open_multi_rejects_conflicting_vector_index_dimensions() -> TestResult {
    fn blob_with_vector_index(dims: usize) -> TestResult<Vec<u8>> {
        let db = GrafeoDB::new_in_memory();
        let owner = db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(dims),
                metric: None,
                m: None,
                ef_construction: None,
                ef: None,
                quantization: None,
            },
        })?;
        assert_eq!(owner.as_u32(), 0);
        Ok(db.export_snapshot()?)
    }

    // Same (label, property) index, different dimensions across snapshots.
    let a = blob_with_vector_index(4)?;
    let b = blob_with_vector_index(8)?;
    for inputs in [[a.as_slice(), b.as_slice()], [b.as_slice(), a.as_slice()]] {
        let error = GrafeoDB::open_multi(inputs)
            .err()
            .ok_or("conflicting vector index dimensions must be rejected")?;
        assert!(
            matches!(&error, Error::Serialization(message)
                if message == "open_multi: index owner ID 0 conflicts"),
            "error must name the exact conflicting owner, got: {error}"
        );
    }

    // Shared exact state merges; equal configurations alone are insufficient
    // because independently built HNSW images carry different RNG continuations.
    let c = blob_with_vector_index(4)?;
    let d = c.clone();
    GrafeoDB::open_multi([c.as_slice(), d.as_slice()])?;
    let changed = GrafeoDB::import_snapshot(&c)?;
    changed.create_node_with_props(
        &["Doc"],
        [("embedding", Value::Vector(vec![1.0, 0.0, 0.0, 0.0].into()))],
    );
    let divergent = changed.export_snapshot()?;
    let error = GrafeoDB::open_multi([c.as_slice(), divergent.as_slice()])
        .err()
        .ok_or("equal configs with distinct exact vector states must reject")?;
    assert!(error.to_string().contains("byte-identical"), "{error}");
    Ok(())
}

#[cfg(all(feature = "lpg", feature = "text-index", feature = "vector-index"))]
mod exact_owner_overlap {
    use super::{TestResult, TestSnapshot};
    use grafeo_common::storage::Section;
    use grafeo_common::types::{GraphPath, IndexId, NodeId, Value};
    use grafeo_core::graph::lpg::{LpgStoreSection, PhysicalIndexKey, decode_index_key};
    use grafeo_core::index::text::TextIndexSection;
    use grafeo_core::index::vector::VectorStoreSection;
    use grafeo_engine::{CreateIndexRequest, GrafeoDB, IndexCreateKind};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    type ExactImages = BTreeMap<PhysicalIndexKey, Vec<u8>>;

    struct Fixture {
        left: GrafeoDB,
        right: GrafeoDB,
        owners: Vec<(IndexId, PhysicalIndexKey)>,
    }

    fn wire(bytes: &[u8]) -> TestResult<TestSnapshot> {
        let (snapshot, consumed) =
            bincode::serde::decode_from_slice(&bytes[13..], bincode::config::standard())?;
        assert_eq!(&bytes[..5], b"\x0cCDC1");
        assert_eq!(consumed, bytes.len() - 13);
        Ok(snapshot)
    }

    fn exact_images(db: &GrafeoDB) -> TestResult<ExactImages> {
        let mut images = ExactImages::new();
        for (path, graph) in LpgStoreSection::new(Arc::clone(
            grafeo_engine::database::testing::root_lpg_store(db),
        ))
        .capture_graphs()?
        {
            for (key, index) in graph.text_index_entries() {
                let (label, property) = decode_index_key(&key).ok_or("invalid Text fixture key")?;
                let key = PhysicalIndexKey::text(path.clone(), label, property);
                let bytes = TextIndexSection::from_views(vec![(key.clone(), index)]).serialize()?;
                assert!(images.insert(key, bytes).is_none());
            }
            for (key, index) in graph.vector_index_entries() {
                let (label, property) =
                    decode_index_key(&key).ok_or("invalid Vector fixture key")?;
                let key = PhysicalIndexKey::vector(path.clone(), label, property);
                let bytes =
                    VectorStoreSection::from_views(vec![(key.clone(), index)]).serialize()?;
                assert!(images.insert(key, bytes).is_none());
            }
        }
        Ok(images)
    }

    fn create_pair(
        db: &GrafeoDB,
        graph: GraphPath,
        prefix: &str,
    ) -> TestResult<Vec<(IndexId, PhysicalIndexKey)>> {
        let text = db.create_index(CreateIndexRequest {
            graph: graph.clone(),
            name: Some(format!("{prefix}_text")),
            label: Some("Doc".into()),
            property: "body".into(),
            kind: IndexCreateKind::Text {
                min_token_length: None,
            },
        })?;
        let vector = db.create_index(CreateIndexRequest {
            graph: graph.clone(),
            name: Some(format!("{prefix}_vector")),
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(4),
                ef_construction: Some(24),
                ef: None,
                quantization: None,
            },
        })?;
        Ok(vec![
            (text, PhysicalIndexKey::text(graph.clone(), "Doc", "body")),
            (vector, PhysicalIndexKey::vector(graph, "Doc", "embedding")),
        ])
    }

    fn burn_owner(db: &GrafeoDB, expected: u32) -> TestResult {
        let owner = db.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: Some(format!("burned_{expected}")),
            label: None,
            property: format!("burned_{expected}"),
            kind: IndexCreateKind::Property,
        })?;
        assert_eq!(owner.as_u32(), expected);
        assert!(db.drop_index(owner)?);
        Ok(())
    }

    fn add_archive(db: &GrafeoDB, name: &str) -> TestResult<Vec<(IndexId, PhysicalIndexKey)>> {
        assert!(db.create_graph(name)?);
        db.set_current_graph(Some(name))?;
        let first = db.create_node_with_props(
            &["Doc"],
            [
                ("body", Value::from(format!("{name} original history"))),
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into())),
            ],
        );
        let second = db.create_node_with_props(
            &["Doc"],
            [
                ("body", Value::from(format!("{name} second archive"))),
                ("embedding", Value::Vector(vec![0.0, 1.0, 0.0].into())),
            ],
        );
        assert_eq!((first.as_u64(), second.as_u64()), (0, 1));
        db.create_edge(first, second, "NEXT");
        let owners = create_pair(db, GraphPath::from_components(&[name])?, name)?;
        db.set_node_property(
            first,
            "body",
            Value::from(format!("{name} updated history")),
        )?;
        db.set_current_graph(None)?;
        Ok(owners)
    }

    fn fixture() -> TestResult<Fixture> {
        let shared = GrafeoDB::new_in_memory();
        let mut owners = create_pair(&shared, GraphPath::root(), "shared")?;
        let seed = shared.export_snapshot()?;
        let left = GrafeoDB::import_snapshot(&seed)?;
        let right = GrafeoDB::import_snapshot(&seed)?;
        owners.extend(add_archive(&left, "left")?);
        burn_owner(&right, 2)?;
        burn_owner(&right, 3)?;
        owners.extend(add_archive(&right, "right")?);
        burn_owner(&right, 6)?;
        assert_eq!(
            owners
                .iter()
                .map(|(owner, _)| owner.as_u32())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(shared.export_snapshot()?, seed);
        assert_eq!(
            (left.node_count(), right.node_count()),
            (0, 0),
            "shared root has no duplicate rows"
        );
        Ok(Fixture {
            left,
            right,
            owners,
        })
    }

    fn append_continuation(db: &GrafeoDB) -> TestResult<Vec<(GraphPath, NodeId)>> {
        let mut ids = Vec::new();
        for name in [None, Some("left"), Some("right")] {
            db.set_current_graph(name)?;
            let path = name.map_or_else(
                || Ok(GraphPath::root()),
                |name| GraphPath::from_components(&[name]),
            )?;
            for offset in 0..2 {
                let node = db.create_node_with_props(
                    &["Doc"],
                    [
                        ("body", Value::from("continuation exact history")),
                        (
                            "embedding",
                            Value::Vector(vec![0.25, 0.5, 0.75 + offset as f32].into()),
                        ),
                    ],
                );
                assert_eq!(
                    node.as_u64(),
                    if name.is_some() { 2 + offset } else { offset }
                );
                ids.push((path.clone(), node));
            }
        }
        db.set_current_graph(None)?;
        assert_eq!(db.text_search("Doc", "body", "continuation", 10)?.len(), 2);
        assert_eq!(
            db.vector_search("Doc", "embedding", &[0.25, 0.5, 0.75], 1, None, None)?[0].0,
            NodeId::new(0)
        );
        Ok(ids)
    }

    #[test]
    fn open_multi_unions_partial_overlap_exact_text_vector_owners_in_both_orders() -> TestResult {
        let fixture = fixture()?;
        let left = fixture.left.export_snapshot()?;
        let right = fixture.right.export_snapshot()?;
        let source_left = exact_images(&fixture.left)?;
        let source_right = exact_images(&fixture.right)?;
        let mut expected = source_left.clone();
        for (key, bytes) in &source_right {
            if let Some(shared) = expected.get(key) {
                assert_eq!(
                    shared, bytes,
                    "shared A is an exact clone, not a rebuilt index"
                );
            } else {
                expected.insert(key.clone(), bytes.clone());
            }
        }
        assert_eq!(
            (source_left.len(), source_right.len(), expected.len()),
            (4, 4, 6)
        );
        let mut before_order = None;
        let mut after_order = None;
        for inputs in [
            [left.as_slice(), right.as_slice()],
            [right.as_slice(), left.as_slice()],
        ] {
            let merged = GrafeoDB::open_multi(inputs)?;
            assert_ne!(merged.store_id(), fixture.left.store_id());
            assert_eq!(exact_images(&merged)?, expected);
            let merged_wire = wire(&merged.export_snapshot()?)?;
            let before = (merged_wire.text_indexes, merged_wire.vector_indexes);
            if let Some(previous) = &before_order {
                assert_eq!(&before, previous);
            }
            before_order = Some(before);
            let ids = append_continuation(&merged)?;
            let continued_wire = wire(&merged.export_snapshot()?)?;
            let after = (
                ids,
                exact_images(&merged)?,
                continued_wire.text_indexes,
                continued_wire.vector_indexes,
            );
            if let Some(previous) = &after_order {
                assert_eq!(
                    &after, previous,
                    "exact RNG and posting continuation must not depend on input order"
                );
            }
            after_order = Some(after);
            burn_owner(&merged, 7)?;
            for (owner, key) in &fixture.owners {
                assert!(
                    merged.drop_index(*owner)?,
                    "canonical owner ID must survive union"
                );
                assert!(
                    !exact_images(&merged)?.contains_key(key),
                    "owner ID must still address its exact physical key"
                );
            }
            assert!(exact_images(&merged)?.is_empty());
        }
        assert_eq!(fixture.left.export_snapshot()?, left);
        assert_eq!(fixture.right.export_snapshot()?, right);
        assert_eq!(exact_images(&fixture.left)?, source_left);
        assert_eq!(exact_images(&fixture.right)?, source_right);
        Ok(())
    }

    #[test]
    fn open_multi_rejects_divergent_shared_text_and_vector_state_without_touching_inputs()
    -> TestResult {
        for vector in [false, true] {
            let fixture = fixture()?;
            let root = GraphPath::root();
            let changed = if vector {
                PhysicalIndexKey::vector(root.clone(), "Doc", "embedding")
            } else {
                PhysicalIndexKey::text(root.clone(), "Doc", "body")
            };
            let unchanged = if vector {
                PhysicalIndexKey::text(root, "Doc", "body")
            } else {
                PhysicalIndexKey::vector(root, "Doc", "embedding")
            };
            let before = exact_images(&fixture.right)?;
            let property = if vector {
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()))
            } else {
                ("body", Value::from("divergent shared posting"))
            };
            fixture.right.create_node_with_props(&["Doc"], [property]);
            let after = exact_images(&fixture.right)?;
            assert_ne!(before.get(&changed), after.get(&changed));
            assert_eq!(before.get(&unchanged), after.get(&unchanged));
            let left = fixture.left.export_snapshot()?;
            let right = fixture.right.export_snapshot()?;
            for inputs in [
                [left.as_slice(), right.as_slice()],
                [right.as_slice(), left.as_slice()],
            ] {
                let error = GrafeoDB::open_multi(inputs)
                    .err()
                    .ok_or("shared owner exact-state conflict must reject")?;
                assert!(error.to_string().contains("byte-identical"), "{error}");
            }
            assert_eq!(fixture.left.export_snapshot()?, left);
            assert_eq!(fixture.right.export_snapshot()?, right);
            assert_eq!(exact_images(&fixture.right)?, after);
        }
        Ok(())
    }
}
