//! Exact index replacement through the public, populated-target restore path.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "text-index",
    feature = "vector-index"
))]

use super::decode_snapshot_bytes;
use crate::{Config, CreateIndexRequest, GrafeoDB, GraphModel, IndexCreateKind};
use grafeo_common::types::{EpochId, GraphPath, NodeId, TransactionId, Value};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::{PropertyIndexPredicate, PropertyIndexRequest};
use grafeo_core::index::vector::PropertyVectorAccessor;
use std::sync::Arc;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct Witness {
    path: GraphPath,
    node: NodeId,
    deleted: NodeId,
    epoch: EpochId,
    score: u64,
}

fn database() -> TestResult<GrafeoDB> {
    database_for_model(GraphModel::Lpg)
}

fn database_for_model(model: GraphModel) -> TestResult<GrafeoDB> {
    Ok(GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(model)
            .with_gc_interval(0),
    )?)
}

fn graph(db: &GrafeoDB, path: &GraphPath) -> TestResult<Arc<LpgStore>> {
    let mut store = Arc::clone(crate::database::testing::root_lpg_store(db));
    for component in path.components() {
        store = store.graph(component).ok_or("exact graph path is absent")?;
    }
    Ok(store)
}

fn owners(db: &GrafeoDB, path: &GraphPath, prefix: &str, scalar: bool) -> TestResult {
    for (property, label, kind) in [
        ("email", None, IndexCreateKind::Property),
        ("rank", None, IndexCreateKind::BTree),
        (
            "body",
            Some("Doc"),
            IndexCreateKind::Text {
                min_token_length: None,
            },
        ),
        (
            "embedding",
            Some("Doc"),
            IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: scalar.then(|| "scalar".into()),
            },
        ),
    ] {
        db.create_index(CreateIndexRequest {
            graph: path.clone(),
            name: Some(format!("{prefix}-{property}")),
            label: label.map(str::to_owned),
            property: property.into(),
            kind,
        })?;
    }
    Ok(())
}

fn source() -> TestResult<(GrafeoDB, Vec<Witness>)> {
    source_for_model(GraphModel::Lpg)
}

fn source_for_model(model: GraphModel) -> TestResult<(GrafeoDB, Vec<Witness>)> {
    let db = database_for_model(model)?;
    for components in [
        vec!["nested"],
        vec!["nested", ""],
        vec!["nested", "", "deep"],
        vec!["nested//deep"],
        vec![""],
    ] {
        assert!(db.create_graph_path(&GraphPath::from_components(&components)?)?);
    }
    let paths = [
        GraphPath::root(),
        GraphPath::from_components(&["nested", "", "deep"])?,
        GraphPath::from_components(&["nested//deep"])?,
        GraphPath::from_components(&[""])?,
    ];
    let mut witnesses = Vec::new();
    for (position, path) in paths.into_iter().enumerate() {
        let session = db.session();
        session.use_graph_path(&path)?;
        let node = session.create_node_with_props(
            &["Doc"],
            [
                ("email", Value::from("live")),
                ("rank", Value::from(7_i64)),
                ("body", Value::from("initial mature revision")),
                ("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into())),
            ],
        )?;
        let deleted = session.create_node_with_props(
            &["Doc"],
            [
                ("email", Value::from("deleted")),
                ("rank", Value::from(8_i64)),
                ("body", Value::from("retained deleted document")),
                ("embedding", Value::Vector(vec![0.0, 1.0, 0.0].into())),
            ],
        )?;
        owners(&db, &path, &format!("source-{position}"), position == 1)?;
        let store = graph(&db, &path)?;
        let epoch = db.current_epoch();
        let text = store
            .get_text_index("Doc", "body")
            .ok_or("Text owner missing")?;
        let score = text
            .read()
            .score_document_visible(node, "initial", epoch, TransactionId::INVALID, None, false)?
            .ok_or("historical Text score missing")?
            .to_bits();
        session
            .execute("MATCH (n:Doc) WHERE n.email = 'live' SET n.body = 'final mature revision'")?;
        session.set_node_property(node, "rank", Value::from(9_i64))?;
        assert!(session.delete_node(deleted));
        db.transaction_manager
            .with_write_authority(|| store.gc_text_indexes(epoch))?;
        assert_eq!(text.read().retained_from(), epoch);
        let vector = store
            .get_vector_index("Doc", "embedding")
            .ok_or("Vector owner missing")?;
        assert!(!vector.contains(deleted));
        assert!(
            vector
                .snapshot_topology()
                .2
                .iter()
                .any(|(id, _)| *id == deleted)
        );
        witnesses.push(Witness {
            path,
            node,
            deleted,
            epoch,
            score,
        });
    }
    // Consume and retire a final owner ID so exact allocator-floor restoration
    // cannot be faked by assigning contiguous IDs to the surviving owners.
    let retired = db.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("retired-floor".into()),
        label: None,
        property: "retired".into(),
        kind: IndexCreateKind::Property,
    })?;
    assert!(db.drop_index(retired)?);
    assert_eq!(
        db.catalog.index_allocator_high_water(),
        retired.as_u32() + 1
    );
    Ok((db, witnesses))
}

fn populated_target() -> TestResult<GrafeoDB> {
    populated_target_for_model(GraphModel::Lpg)
}

fn populated_target_for_model(model: GraphModel) -> TestResult<GrafeoDB> {
    let db = database_for_model(model)?;
    db.session().execute("INSERT (:Doc {email:'stale', rank:99, body:'stale target document', embedding:[0.0,0.0,1.0]})")?;
    owners(&db, &GraphPath::root(), "target", false)?;
    db.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("target-only".into()),
        label: None,
        property: "obsolete".into(),
        kind: IndexCreateKind::Property,
    })?;
    db.session().execute("CREATE GRAPH target_only")?;
    let path = GraphPath::from_components(&["target_only"])?;
    owners(&db, &path, "target-child", false)?;
    Ok(db)
}

#[test]
fn live_restore_exact_nested_owners_history_and_rng_continue_at_retained_root() -> TestResult {
    let (source, witnesses) = source()?;
    let incoming = source.export_snapshot()?;
    let target = populated_target()?;
    for position in 0..20 {
        let retired = target.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: None,
            label: None,
            property: format!("target-retired-{position}"),
            kind: IndexCreateKind::Property,
        })?;
        assert!(target.drop_index(retired)?);
    }
    assert!(
        target.catalog.index_allocator_high_water() > source.catalog.index_allocator_high_water()
    );
    let retained_root = Arc::clone(crate::database::testing::root_lpg_store(&target));
    let old_text = retained_root
        .get_text_index("Doc", "body")
        .ok_or("old Text missing")?;
    let old_vector = retained_root
        .get_vector_index("Doc", "embedding")
        .ok_or("old Vector missing")?;
    let old_topology = old_vector.snapshot_topology();
    target.restore_snapshot(&incoming)?;
    assert!(Arc::ptr_eq(
        &retained_root,
        crate::database::testing::root_lpg_store(&target)
    ));
    assert_eq!(target.export_snapshot()?, incoming);
    assert_eq!(
        target.catalog.index_allocator_high_water(),
        source.catalog.index_allocator_high_water()
    );
    let mut expected = source.catalog.all_indexes();
    let mut actual = target.catalog.all_indexes();
    expected.sort_unstable_by_key(|owner| owner.id);
    actual.sort_unstable_by_key(|owner| owner.id);
    assert_eq!(actual, expected);
    assert!(!retained_root.has_property_index("obsolete"));
    assert!(retained_root.graph("target_only").is_none());
    assert!(!old_text.read().search("stale", 10).is_empty());
    assert_eq!(old_vector.snapshot_topology(), old_topology);
    for witness in &witnesses {
        let store = graph(&target, &witness.path)?;
        assert!(store.retained_history_floor() <= witness.epoch);
        assert_eq!(
            store.find_nodes_by_property("email", &Value::from("live")),
            vec![witness.node]
        );
        assert!(store.has_property_index("rank"));
        let old_rank = Value::Int64(7);
        let old_rank_eq = store.lookup_nodes_indexed(PropertyIndexRequest {
            property: "rank",
            predicate: PropertyIndexPredicate::Equal(&old_rank),
            epoch: witness.epoch,
            transaction_id: None,
        })?;
        assert_eq!(old_rank_eq, Some(vec![witness.node]));
        let deleted_rank = Value::Int64(8);
        let old_ranks = [old_rank.clone(), deleted_rank.clone()];
        let old_rank_in = store.lookup_nodes_indexed(PropertyIndexRequest {
            property: "rank",
            predicate: PropertyIndexPredicate::In(&old_ranks),
            epoch: witness.epoch,
            transaction_id: None,
        })?;
        assert_eq!(old_rank_in, Some(vec![witness.node, witness.deleted]));
        let old_rank_range = store.lookup_nodes_indexed(PropertyIndexRequest {
            property: "rank",
            predicate: PropertyIndexPredicate::Range {
                min: Some(&old_rank),
                max: Some(&deleted_rank),
                min_inclusive: true,
                max_inclusive: true,
            },
            epoch: witness.epoch,
            transaction_id: None,
        })?;
        assert_eq!(old_rank_range, Some(vec![witness.node, witness.deleted]));
        let current_rank = Value::Int64(9);
        let current_epoch = target.current_epoch();
        let current_rank_eq = store.lookup_nodes_indexed(PropertyIndexRequest {
            property: "rank",
            predicate: PropertyIndexPredicate::Equal(&current_rank),
            epoch: current_epoch,
            transaction_id: None,
        })?;
        assert_eq!(current_rank_eq, Some(vec![witness.node]));
        let current_ranks = [current_rank.clone(), deleted_rank.clone()];
        let current_rank_in = store.lookup_nodes_indexed(PropertyIndexRequest {
            property: "rank",
            predicate: PropertyIndexPredicate::In(&current_ranks),
            epoch: current_epoch,
            transaction_id: None,
        })?;
        assert_eq!(current_rank_in, Some(vec![witness.node]));
        let current_rank_range = store.lookup_nodes_indexed(PropertyIndexRequest {
            property: "rank",
            predicate: PropertyIndexPredicate::Range {
                min: Some(&current_rank),
                max: Some(&current_rank),
                min_inclusive: true,
                max_inclusive: true,
            },
            epoch: current_epoch,
            transaction_id: None,
        })?;
        assert_eq!(current_rank_range, Some(vec![witness.node]));
        let text = store
            .get_text_index("Doc", "body")
            .ok_or("restored Text missing")?;
        assert_eq!(text.read().retained_from(), witness.epoch);
        assert_eq!(
            text.read()
                .score_document_visible(
                    witness.node,
                    "initial",
                    witness.epoch,
                    TransactionId::INVALID,
                    None,
                    false,
                )?
                .map(f64::to_bits),
            Some(witness.score)
        );
        assert_eq!(text.read().score_document(witness.node, "initial"), 0.0);
        let vector = store
            .get_vector_index("Doc", "embedding")
            .ok_or("restored Vector missing")?;
        assert!(!vector.contains(witness.deleted));
        let accessor = PropertyVectorAccessor::new(store.as_ref(), "embedding");
        assert_eq!(
            vector
                .search(&[1.0, 0.0, 0.0], 1, &accessor)
                .first()
                .map(|hit| hit.0),
            Some(witness.node)
        );
    }
    // Identical subsequent commits must consume the restored ID/RNG frontiers
    // and maintain the restored physical bindings, including quantized state.
    for witness in &witnesses {
        for db in [&source, &target] {
            let session = db.session();
            session.use_graph_path(&witness.path)?;
            let node = session.create_node_with_props(
                &["Doc"],
                [
                    ("email", Value::from("continued")),
                    ("body", Value::from("continuation searchable")),
                    ("embedding", Value::Vector(vec![0.0, 0.0, 1.0].into())),
                ],
            )?;
            assert!(node.as_u64() > witness.deleted.as_u64());
            session.set_node_property(witness.node, "body", Value::from("later live revision"))?;
        }
    }
    assert_eq!(target.export_snapshot()?, source.export_snapshot()?);
    assert_eq!(old_vector.snapshot_topology(), old_topology);
    assert!(!old_text.read().search("stale", 10).is_empty());
    Ok(())
}

#[test]
fn live_restore_malformed_exact_index_images_leave_populated_target_unchanged() -> TestResult {
    let (source, _) = source()?;
    let incoming = source.export_snapshot()?;
    let target = populated_target()?;
    let retained_root = Arc::clone(crate::database::testing::root_lpg_store(&target));
    let before = target.export_snapshot()?;
    for corrupt_text in [true, false] {
        let mut snapshot = decode_snapshot_bytes(&incoming)?;
        if corrupt_text {
            snapshot.text_indexes.push(0xff);
        } else {
            snapshot.vector_indexes.push(0xff);
        }
        let malformed = super::encode_snapshot_bytes(&snapshot)?;
        assert!(target.restore_snapshot(&malformed).is_err());
        assert!(Arc::ptr_eq(
            &retained_root,
            crate::database::testing::root_lpg_store(&target)
        ));
        assert_eq!(target.export_snapshot()?, before);
    }
    Ok(())
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
fn seed_admission_rdf(db: &GrafeoDB, token: &str) -> TestResult {
    use crate::ValidTimeInterval;
    let mut session = db.session();
    session.set_rdf_valid_time(Some(ValidTimeInterval::from_tai_nanoseconds(1_000, 2_000)?));
    session.begin_transaction()?;
    session.execute_sparql("CREATE GRAPH <http://admission.test/named>")?;
    session.execute_sparql(&format!(
        "INSERT DATA {{ <http://admission.test/person> a <http://admission.test/Person> . \
         GRAPH <http://admission.test/named> {{ <http://admission.test/row> \
         <http://admission.test/value> '{token}' . }} }}"
    ))?;
    session.commit()?;
    session.set_rdf_valid_time(Some(ValidTimeInterval::from_tai_nanoseconds(3_000, 4_000)?));
    session.begin_transaction()?;
    session.execute_sparql(&format!(
        "DELETE DATA {{ GRAPH <http://admission.test/named> {{ <http://admission.test/row> \
         <http://admission.test/value> '{token}' . }} }}"
    ))?;
    session.execute_sparql(&format!(
        "INSERT DATA {{ GRAPH <http://admission.test/named> {{ <http://admission.test/row> \
         <http://admission.test/value> '{token}-current' . }} }}"
    ))?;
    session.commit()?;
    drop(session);
    let projection = db.declare_rdf_lpg_projection("http://admission.test/Person", "Projected")?;
    assert_eq!(db.rebuild_rdf_lpg_projection(projection)?, 1);
    Ok(())
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn live_restore_late_admission_preserves_both_models_and_retained_handles() -> TestResult {
    use crate::catalog::IndexConfiguration;
    use grafeo_common::types::{MAX_GRAPH_PATH_COMPONENTS, MAX_WORLD_GRAPH_NAME_BYTES};
    use grafeo_common::utils::error::Error;
    use grafeo_core::graph::lpg::LpgStoreSection;
    use grafeo_core::index::text::TextIndexSection;

    let (source, _) = source_for_model(GraphModel::Both)?;
    seed_admission_rdf(&source, "source")?;
    let target = populated_target_for_model(GraphModel::Both)?;
    seed_admission_rdf(&target, "target")?;
    let incoming = source.export_snapshot()?;
    let decoded = decode_snapshot_bytes(&incoming)?;
    let encode = |snapshot: &super::Snapshot| super::encode_snapshot_bytes(snapshot);
    let mut cases = Vec::new();

    let mut bad = decoded.clone();
    bad.graphs
        .last_mut()
        .ok_or("last graph missing")?
        .next_node_id = 0;
    cases.push(("late child allocator", encode(&bad)?, "allocator"));
    let mut bad = decoded.clone();
    let last = bad.graphs.last_mut().ok_or("last graph missing")?;
    assert_eq!(last.path, GraphPath::from_components(&["nested//deep"])?);
    last.nodes
        .push(last.nodes.first().ok_or("late node missing")?.clone());
    cases.push(("late duplicate node", encode(&bad)?, "duplicate"));

    for kind in 0..3 {
        let mut bad = decoded.clone();
        let last = bad.graphs.last_mut().ok_or("last graph missing")?;
        let node = last.nodes.first().ok_or("late node missing")?;
        let edge = super::SnapshotEdge {
            id: grafeo_common::types::EdgeId::new(0),
            src: if kind == 1 { NodeId::new(999) } else { node.id },
            dst: if kind == 2 { NodeId::new(999) } else { node.id },
            edge_type: "Malformed".into(),
            lifetimes: node.lifetimes.clone(),
            properties: Vec::new(),
        };
        last.next_edge_id = 1;
        last.edges.push(edge.clone());
        if kind == 0 {
            last.edges.push(edge);
        }
        let (name, reason) = match kind {
            0 => ("late duplicate edge", "duplicate edge"),
            1 => ("late missing source", "non-existent source"),
            _ => ("late missing destination", "non-existent destination"),
        };
        cases.push((name, encode(&bad)?, reason));
    }

    let owner = source
        .catalog
        .find_index_by_name("source-2-body")
        .ok_or("late Text owner missing")?;
    for mismatch in [false, true] {
        let catalog =
            crate::database::catalog_wire::decode_catalog(&decoded.catalog_state, decoded.epoch)?;
        let definition = catalog.get_index(owner).ok_or("late owner absent")?;
        assert!(catalog.drop_index(owner));
        if mismatch {
            let IndexConfiguration::Text {
                config,
                min_token_length,
            } = definition.configuration
            else {
                return Err("fixture selected non-Text owner".into());
            };
            catalog.create_index(
                Some(&definition.name),
                definition.label,
                definition.property_key,
                definition.key.graph().clone(),
                IndexConfiguration::Text {
                    config,
                    min_token_length: min_token_length + 1,
                },
            )?;
        }
        let mut bad = decoded.clone();
        bad.catalog_state = crate::database::catalog_wire::encode_catalog_read(
            catalog.read().view(),
            decoded.epoch,
        )?;
        cases.push((
            if mismatch {
                "late tokenizer disagreement"
            } else {
                "late missing owner"
            },
            encode(&bad)?,
            if mismatch { "tokenizer" } else { "owner" },
        ));
    }

    let ranges = TextIndexSection::payload_entry_ranges(&decoded.text_indexes)?;
    let (key, range) = ranges.last().ok_or("late Text payload missing")?;
    assert!(!key.graph().components().is_empty());
    let key_size = bincode::serde::encode_to_vec(key, bincode::config::standard())?.len();
    // Current Text5: exact key, two f64 BM25 parameters, tokenizer variant.
    let mut bad = decoded.clone();
    let variant = bad
        .text_indexes
        .get_mut(range.start + key_size + 16)
        .ok_or("tokenizer variant missing")?;
    assert_eq!(*variant, 0);
    *variant = 1;
    cases.push(("late unknown tokenizer", encode(&bad)?, "tokenizer"));

    // Locate the complete uniquely encoded last graph, not an incidental path
    // match in Catalog/Text/Vector bytes. Replace only its first (path) field.
    let last = decoded.graphs.last().ok_or("last graph missing")?;
    let graph_bytes = bincode::serde::encode_to_vec(last, bincode::config::standard())?;
    let offsets: Vec<_> = incoming
        .windows(graph_bytes.len())
        .enumerate()
        .filter_map(|(offset, bytes)| (bytes == graph_bytes).then_some(offset))
        .collect();
    assert_eq!(offsets.len(), 1);
    let offset = offsets[0];
    let max_path_bytes = 4 + MAX_GRAPH_PATH_COMPONENTS * (4 + MAX_WORLD_GRAPH_NAME_BYTES);
    let path = last.path.to_bytes(max_path_bytes)?;
    let path_field = bincode::serde::encode_to_vec(&path, bincode::config::standard())?;
    assert!(graph_bytes.starts_with(&path_field));
    let mut path_cases = Vec::new();
    let mut truncated = path.clone();
    truncated.pop().ok_or("nonempty path expected")?;
    path_cases.push((
        "late truncated path",
        bincode::serde::encode_to_vec(truncated, bincode::config::standard())?,
        "truncated",
    ));
    let mut too_deep = path.clone();
    too_deep[..4].copy_from_slice(&u32::try_from(MAX_GRAPH_PATH_COMPONENTS + 1)?.to_le_bytes());
    path_cases.push((
        "late oversized path depth",
        bincode::serde::encode_to_vec(too_deep, bincode::config::standard())?,
        "components",
    ));
    let mut too_long = path;
    too_long[4..8].copy_from_slice(&u32::try_from(MAX_WORLD_GRAPH_NAME_BYTES + 1)?.to_le_bytes());
    path_cases.push((
        "late oversized component",
        bincode::serde::encode_to_vec(too_long, bincode::config::standard())?,
        "bytes",
    ));
    path_cases.push((
        "late oversized path allocation",
        bincode::serde::encode_to_vec(
            u64::try_from(max_path_bytes + 1)?,
            bincode::config::standard(),
        )?,
        "bounds",
    ));
    for (name, replacement, reason) in path_cases {
        let mut bad = incoming[..offset].to_vec();
        bad.extend_from_slice(&replacement);
        bad.extend_from_slice(&incoming[offset + path_field.len()..]);
        let body_len = u64::try_from(bad.len() - 13)?;
        bad[5..13].copy_from_slice(&body_len.to_le_bytes());
        cases.push((name, bad, reason));
    }
    assert_eq!(cases.len(), 12);

    let before = target.export_snapshot()?;
    let epoch = target.current_epoch();
    let rdf_epoch = target.rdf_store.commit_epoch();
    let owners = target.catalog.all_indexes();
    let graph_arcs = LpgStoreSection::new(Arc::clone(crate::database::testing::root_lpg_store(
        &target,
    )))
    .capture_graphs()?;
    let mut indexes = Vec::new();
    for (path, store) in &graph_arcs {
        indexes.push((
            path.clone(),
            store.current_epoch(),
            [
                store
                    .observe_property_index("email")
                    .ok_or("target Property missing")?,
                store
                    .observe_text_index("Doc", "body")
                    .ok_or("target Text missing")?,
                store
                    .observe_vector_index("Doc", "embedding")
                    .ok_or("target Vector missing")?,
            ],
        ));
    }
    let held_session = target.session();
    let query = "MATCH (n:Doc) RETURN n.email";
    let rows = held_session.execute(query)?.rows;
    assert_eq!(rows, vec![vec![Value::from("stale")]]);
    for (name, bytes, reason) in cases {
        let Err(error) = target.restore_snapshot(&bytes) else {
            return Err(format!("{name}: malformed snapshot was accepted").into());
        };
        assert!(matches!(error, Error::Serialization(_)), "{name}: {error}");
        assert!(
            error.to_string().to_lowercase().contains(reason),
            "{name}: wrong rejection: {error}"
        );
        assert_eq!(
            target.export_snapshot()?,
            before,
            "{name}: exact target changed"
        );
        assert_eq!(
            target.current_epoch(),
            epoch,
            "{name}: published epoch changed"
        );
        assert_eq!(
            target.rdf_store.commit_epoch(),
            rdf_epoch,
            "{name}: RDF epoch changed"
        );
        assert_eq!(
            target.catalog.all_indexes(),
            owners,
            "{name}: owners changed"
        );
        for (path, old) in &graph_arcs {
            assert!(
                Arc::ptr_eq(old, &graph(&target, path)?),
                "{name}: graph incarnation changed"
            );
        }
        for (path, epoch, observations) in &indexes {
            let store = graph(&target, path)?;
            assert_eq!(store.current_epoch(), *epoch, "{name}: child epoch changed");
            for observation in observations {
                target
                    .transaction_manager
                    .with_write_authority(|| store.validate_index_registration(observation))?;
            }
        }
        assert_eq!(
            held_session.execute(query)?.rows,
            rows,
            "{name}: retained read changed"
        );
        assert_eq!(
            source.export_snapshot()?,
            incoming,
            "{name}: source changed"
        );
    }
    target.restore_snapshot(&incoming)?;
    assert_eq!(target.export_snapshot()?, incoming);
    assert!(Arc::ptr_eq(
        &graph_arcs[0].1,
        crate::database::testing::root_lpg_store(&target)
    ));
    assert!(
        crate::database::testing::root_lpg_store(&target)
            .graph("target_only")
            .is_none()
    );
    assert_eq!(
        held_session.execute(query)?.rows,
        vec![vec![Value::from("live")]]
    );
    Ok(())
}

#[test]
fn live_restore_same_epoch_invalidates_retained_named_session_physical_cache() -> TestResult {
    let source = database()?;
    let target = database()?;
    let path = GraphPath::from_components(&["kept"])?;
    for (db, value) in [(&source, "source"), (&target, "target")] {
        let session = db.session();
        session.execute("CREATE GRAPH kept")?;
        session.use_graph_path(&path)?;
        session.execute(&format!("INSERT (:Row {{value:'{value}'}})"))?;
    }
    assert_eq!(source.current_epoch(), target.current_epoch());
    let retained_root = Arc::clone(crate::database::testing::root_lpg_store(&target));
    let retained_session = target.session();
    retained_session.use_graph_path(&path)?;
    let query = "MATCH (n:Row) RETURN n.value";
    for _ in 0..3 {
        assert_eq!(
            retained_session.execute(query)?.rows,
            vec![vec![Value::from("target")]]
        );
    }
    let incoming = source.export_snapshot()?;
    target.restore_snapshot(&incoming)?;
    assert!(Arc::ptr_eq(
        &retained_root,
        crate::database::testing::root_lpg_store(&target)
    ));
    assert_eq!(source.current_epoch(), target.current_epoch());
    assert_eq!(
        retained_session.execute(query)?.rows,
        vec![vec![Value::from("source")]]
    );
    assert_eq!(target.export_snapshot()?, incoming);
    Ok(())
}
