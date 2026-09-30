//! Connected public owner contracts, exercised through the Session commit driver.

use super::*;
use crate::{CreateIndexRequest, GrafeoDB, IndexCreateKind};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

#[test]
#[cfg(all(feature = "gql", feature = "vector-index"))]
fn vector_search_depth_seeded_gql_caller_discriminates_low_override() -> TestResult {
    use grafeo_core::execution::operators::{Operator, VectorScanOperator};
    use grafeo_core::index::vector::{DistanceMetric, HnswIndex, VectorIndexKind};
    use std::collections::HashMap;

    let mut state = 42_u32;
    let mut vectors = Vec::new();
    for _ in 0..320 {
        let mut vector = Vec::new();
        for _ in 0..8 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let component = i32::try_from(state >> 16)? - 32_768;
            vector.push(component as f32 / 32_768.0);
        }
        vectors.push(vector);
    }
    for declared in [1, 128] {
        #[allow(unused_mut)]
        let mut db = GrafeoDB::new_in_memory();
        let session = db.session();
        let mut values: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let mut insertion_order = Vec::new();
        for vector in &vectors[..256] {
            let value: Arc<[f32]> = vector.clone().into();
            let node = session
                .create_node_with_props(&["Doc"], [("emb", Value::Vector(Arc::clone(&value)))])?;
            values.insert(node, value);
            insertion_order.push(node);
        }
        session.execute(&format!("CREATE INDEX seeded_depth FOR (n:Doc) ON (n.emb) USING VECTOR {{dimensions: 8, metric: 'cosine', ef: {declared}}}"))?;
        let physical = crate::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .ok_or("missing declared index")?;
        let config = physical.config().clone();
        assert_eq!(config.ef, declared);
        // Preserve the real GQL-created owner/config while fixing topology RNG.
        let seeded = HnswIndex::with_seed(config, 42);
        let accessor = |node| values.get(&node).cloned();
        for node in insertion_order {
            seeded.insert(node, &values[&node], &accessor);
        }
        db.transaction_manager.with_write_authority(|| {
            crate::database::testing::root_lpg_store(&db).add_vector_index(
                "Doc",
                "emb",
                Arc::new(VectorIndexKind::Hnsw(seeded)),
            );
        });
        drop(session);
        let phases = if cfg!(feature = "compact-store") {
            2
        } else {
            1
        };
        for phase in 0..phases {
            #[cfg(feature = "compact-store")]
            if phase == 1 {
                db.compact()?;
            }
            let mut discriminating_queries = 0;
            for (query_index, query) in vectors[256..].iter().enumerate() {
                let high = db.vector_search("Doc", "emb", query, 1, Some(128), None)?;
                let fixed = db.vector_search("Doc", "emb", query, 1, Some(64), None)?;
                let low = db.vector_search("Doc", "emb", query, 1, Some(1), None)?;
                let expected = if declared == 1 { &low } else { &high };
                assert_eq!(
                    &db.vector_search("Doc", "emb", query, 1, None, None)?,
                    expected
                );
                if low != fixed {
                    discriminating_queries += 1;
                    // Existing GQL ANN procedure requires the physical index;
                    // ordinary ORDER BY is an exact scalar scan, not this caller.
                    // Every planned CALL carries snapshot context, including
                    // auto-commit reads. Its existing visibility-safe search
                    // widens the beam to max(config.ef, 4*k); retain that fence.
                    let call_expected =
                        db.vector_search("Doc", "emb", query, 1, Some(declared.max(4)), None)?;
                    let gql = format!("CALL grafeo.search.vector('Doc', 'emb', {query:?}, 1)");
                    let result = db.session().execute(&gql)?;
                    assert_eq!(
                        result.row_count(),
                        1,
                        "CALL rows declared={declared}, compact phase={phase}, query={query_index}, expected={call_expected:?}"
                    );
                    assert_eq!(
                        result.rows()[0][0],
                        Value::Int64(i64::try_from(call_expected[0].0.0)?),
                        "CALL declared={declared}, compact phase={phase}, query={query_index}, low={low:?}, fixed={fixed:?}, high={high:?}, visible={call_expected:?}"
                    );
                    assert_eq!(
                        result.rows()[0][1],
                        Value::Float64(f64::from(call_expected[0].1)),
                        "CALL distance declared={declared}, compact phase={phase}, query={query_index}"
                    );
                    // Exercise the real physical VectorScan caller on both LPG
                    // and Layered; a hidden fixed ef=64 must differ at depth 1.
                    let mut scan = VectorScanOperator::new(
                        db.graph_store(),
                        Some("Doc".into()),
                        "emb".into(),
                        query.clone(),
                        1,
                        DistanceMetric::Cosine,
                    );
                    let chunk = scan.next()?.ok_or("missing VectorScan result")?;
                    assert_eq!(chunk.row_count(), 1);
                    assert_eq!(
                        chunk.column(0).ok_or("missing node column")?.get_node_id(0),
                        Some(expected[0].0),
                        "VectorScan declared={declared}, compact phase={phase}, query={query_index}, low={low:?}, fixed={fixed:?}, high={high:?}"
                    );
                    assert_eq!(
                        chunk
                            .column(1)
                            .ok_or("missing distance column")?
                            .get_value(0),
                        Some(Value::Float64(f64::from(expected[0].1)))
                    );
                    assert!(scan.next()?.is_none());
                    assert_eq!(
                        crate::database::testing::root_lpg_store(&db)
                            .get_vector_index("Doc", "emb")
                            .ok_or("missing seeded index")?
                            .config()
                            .ef,
                        declared
                    );
                }
            }
            assert!(
                discriminating_queries > 0,
                "seeded corpus must distinguish ef=1 from hardcoded ef=64"
            );
        }
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "gql", feature = "vector-index"))]
fn vector_search_depth_is_committed_owner_configuration_and_zero_is_rejected() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.create_node_with_props(&["Doc"], [("emb", Value::Vector(Arc::from([1.0_f32; 8])))])?;
    session.begin_transaction()?;
    session.execute(
        "CREATE INDEX idx_depth FOR (n:Doc) ON (n.emb) USING VECTOR {dimensions: 8, ef: 128}",
    )?;
    assert!(db.catalog.find_index_by_name("idx_depth").is_none());
    session.commit()?;
    let owner = db
        .catalog
        .find_index_by_name("idx_depth")
        .ok_or("missing committed owner")?;
    let definition = db.catalog.get_index(owner).ok_or("missing definition")?;
    let IndexConfiguration::Vector { config, .. } = &definition.configuration else {
        return Err("wrong index configuration".into());
    };
    assert_eq!(config.ef, 128);
    assert_eq!(
        crate::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .ok_or("missing physical index")?
            .config()
            .ef,
        128
    );
    session.begin_transaction()?;
    session.execute("DROP INDEX idx_depth")?;
    session.rollback()?;
    assert_eq!(db.catalog.get_index(owner), Some(definition.clone()));
    db.rebuild_index(owner)?;
    assert_eq!(db.catalog.get_index(owner), Some(definition.clone()));
    assert_eq!(
        crate::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .ok_or("rebuild lost physical index")?
            .config()
            .ef,
        128
    );
    let before_epoch = db.current_epoch();
    let error = session.execute("CREATE INDEX invalid_depth FOR (n:Other) ON (n.emb) USING VECTOR {dimensions: 8, ef: 0}").expect_err("zero search depth must fail");
    assert!(!error.to_string().is_empty());
    assert!(db.catalog.find_index_by_name("invalid_depth").is_none());
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .get_vector_index("Other", "emb")
            .is_none()
    );
    assert_eq!(db.current_epoch(), before_epoch);
    assert_eq!(db.catalog.get_index(owner), Some(definition));
    Ok(())
}

#[test]
#[cfg(feature = "text-index")]
fn text_tokenizer_native_configuration_survives_rebuild_and_empty_corpus() -> TestResult {
    for minimum in [None, Some(0), Some(1), Some(3), Some(usize::MAX)] {
        let db = GrafeoDB::new_in_memory();
        let node = db
            .session()
            .create_node_with_props(&["Doc"], [("body", Value::from("x ox é archive"))])?;
        let owner = db.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: Some("configured".into()),
            label: Some("Doc".into()),
            property: "body".into(),
            kind: IndexCreateKind::Text {
                min_token_length: minimum,
            },
        })?;
        let resolved = minimum.unwrap_or(2);
        let expected = db.catalog.get_index(owner).ok_or("missing text owner")?;
        assert_eq!(
            expected.configuration,
            IndexConfiguration::Text {
                config: grafeo_core::index::text::BM25Config::default(),
                min_token_length: resolved,
            }
        );
        for rebuild in [false, true] {
            if rebuild {
                db.rebuild_index(owner)?;
            }
            assert_eq!(db.catalog.get_index(owner), Some(expected.clone()));
            let index = crate::database::testing::root_lpg_store(&db)
                .get_text_index("Doc", "body")
                .ok_or("missing text index")?;
            let index = index.read();
            assert!(index.has_simple_tokenizer(resolved));
            for (term, length) in [("x", 1), ("ox", 2), ("é", 2), ("archive", 7)] {
                let ids: Vec<_> = index
                    .search(term, 10)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                assert_eq!(
                    ids,
                    if length >= resolved {
                        vec![node]
                    } else {
                        vec![]
                    }
                );
            }
        }
        #[cfg(feature = "grafeo-file")]
        {
            let copied = GrafeoDB::import_snapshot(&db.export_snapshot()?)?;
            assert_eq!(copied.catalog.get_index(owner), Some(expected));
            assert!(
                crate::database::testing::root_lpg_store(&copied)
                    .get_text_index("Doc", "body")
                    .ok_or("missing copied index")?
                    .read()
                    .has_simple_tokenizer(resolved)
            );
        }
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "text-index", feature = "gql"))]
fn text_tokenizer_gql_configuration_is_transactional_and_rejects_bad_options() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .create_node_with_props(&["Doc"], [("body", Value::from("ox archive"))])?;
    let mut session = db.session();
    let ddl = "CREATE INDEX configured FOR (n:Doc) ON (n.body) USING TEXT {min_token_length: 3}";
    session.begin_transaction()?;
    session.execute(ddl)?;
    assert_eq!(db.catalog.index_count(), 0);
    session.rollback()?;
    assert_eq!(db.catalog.index_count(), 0);
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .get_text_index("Doc", "body")
            .is_none()
    );
    session.begin_transaction()?;
    session.execute(ddl)?;
    session.commit()?;
    let owner = db
        .catalog
        .find_index_by_name("configured")
        .ok_or("missing configured owner")?;
    let definition = db.catalog.get_index(owner).ok_or("missing definition")?;
    assert_eq!(
        definition.configuration,
        IndexConfiguration::Text {
            config: grafeo_core::index::text::BM25Config::default(),
            min_token_length: 3,
        }
    );
    let epoch = db.current_epoch();
    let high_water = db.catalog.index_allocator_high_water();
    for suffix in [
        "USING TEXT {min_token_length: -1}",
        "USING TEXT {min_token_length: '3'}",
        "USING TEXT {min_token_length: 1.5}",
        "USING VECTOR {dimensions: 3, min_token_length: 3}",
    ] {
        assert!(
            session
                .execute(&format!(
                    "CREATE INDEX bad FOR (n:Doc) ON (n.other) {suffix}"
                ))
                .is_err()
        );
        assert_eq!(db.current_epoch(), epoch);
        assert_eq!(db.catalog.index_allocator_high_water(), high_water);
        assert_eq!(db.catalog.get_index(owner), Some(definition.clone()));
        assert_eq!(db.catalog.index_count(), 1);
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "text-index", feature = "grafeo-file"))]
fn text_tokenizer_batched_length_and_document_deltas_preserve_exact_history() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let first = session.create_node_with_props(
        &["Doc"],
        [("body", Value::from("archive archive archive archive"))],
    )?;
    db.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("configured".into()),
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: Some(3),
        },
    })?;
    let initial = db.current_epoch();
    session.begin_transaction()?;
    session.set_node_property(first, "body", Value::from("archive"))?;
    let second = session.create_node_with_props(&["Doc"], [("body", Value::from("companion"))])?;
    session.commit()?; // +1 document, -2 tokens
    let shorter = db.current_epoch();
    let copied = GrafeoDB::import_snapshot(&db.export_snapshot()?)?;
    assert_eq!(copied.export_snapshot()?, db.export_snapshot()?);
    session.begin_transaction()?;
    session.set_node_property(
        first,
        "body",
        Value::from("archive archive archive archive"),
    )?;
    assert!(session.delete_node(second));
    session.commit()?; // -1 document, +2 tokens
    let longer = db.current_epoch();
    let copied = GrafeoDB::import_snapshot(&db.export_snapshot()?)?;
    for database in [&db, &copied] {
        let text = crate::database::testing::root_lpg_store(database)
            .get_text_index("Doc", "body")
            .ok_or("missing Text")?;
        let text = text.read();
        assert!(text.has_simple_tokenizer(3));
        for (epoch, documents, tokens) in [(initial, 1, 4), (shorter, 2, 2), (longer, 1, 4)] {
            assert_eq!(text.doc_count_at(epoch, TransactionId::SYSTEM)?, documents);
            assert_eq!(text.total_length_at(epoch, TransactionId::SYSTEM)?, tokens);
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "gql")]
fn create_then_cascade_drop_does_not_allocate_a_transient_owner() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    assert!(db.create_graph("doomed")?);
    let mut session = db.session();
    session.execute("USE GRAPH doomed")?;
    session.begin_transaction()?;
    session.execute("CREATE INDEX transient FOR (n:Doc) ON (n.value)")?;
    session.execute("DROP GRAPH doomed")?;
    session.commit()?;
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .graph("doomed")
            .is_none()
    );
    assert_eq!(db.catalog.index_count(), 0);
    assert_eq!(db.catalog.index_allocator_high_water(), 0);
    Ok(())
}

#[test]
#[cfg(feature = "text-index")]
fn text_rebuild_rejects_physical_configuration_drift() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let owner = db.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: None,
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    let definition = db.catalog.get_index(owner).ok_or("missing owner")?;
    let physical = crate::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("missing index")?;
    let mut drifted = physical.read().config();
    drifted.k1 += 1.0;
    db.transaction_manager.with_write_authority(|| {
        crate::database::testing::root_lpg_store(&db).add_text_index(
            "Doc",
            "body",
            Arc::new(parking_lot::RwLock::new(
                grafeo_core::index::text::InvertedIndex::new(drifted.clone()),
            )),
        );
    });
    let physical = crate::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("missing replacement index")?;
    assert_eq!(physical.read().config().k1.to_bits(), drifted.k1.to_bits());
    let epoch = db.current_epoch();
    let error = db
        .rebuild_index(owner)
        .expect_err("configuration mismatch must reject");
    assert!(
        error.to_string().contains("configuration differs"),
        "{error}"
    );
    assert_eq!(db.current_epoch(), epoch);
    assert_eq!(db.catalog.get_index(owner), Some(definition));
    assert_eq!(physical.read().config().k1.to_bits(), drifted.k1.to_bits());
    Ok(())
}

fn property_request(graph: GraphPath, name: Option<&str>, property: &str) -> CreateIndexRequest {
    CreateIndexRequest {
        graph,
        name: name.map(str::to_owned),
        label: None,
        property: property.into(),
        kind: IndexCreateKind::Property,
    }
}

#[cfg(feature = "vector-index")]
fn vector_request() -> CreateIndexRequest {
    CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("vector_owner".into()),
        label: Some("Doc".into()),
        property: "emb".into(),
        kind: IndexCreateKind::Vector {
            dimensions: Some(8),
            metric: Some("euclidean".into()),
            m: Some(8),
            ef_construction: Some(37),
            ef: None,
            quantization: Some("scalar".into()),
        },
    }
}

#[test]
fn anonymous_and_explicit_names_are_real_global_owners() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let owner = db.create_index(property_request(GraphPath::root(), None, "value"))?;
    let definition = db
        .catalog
        .get_index(owner)
        .ok_or("missing anonymous owner")?;
    assert_eq!(definition.name, format!("@grafeo-index:{}", owner.as_u32()));
    assert_eq!(db.catalog.find_index_by_name(&definition.name), Some(owner));
    assert!(db.has_property_index("value"));
    let high_water = db.catalog.index_allocator_high_water();
    for name in ["@grafeo-index:", "@grafeo-index:0", "@grafeo-index:future"] {
        assert!(
            db.create_index(property_request(GraphPath::root(), Some(name), "other"))
                .is_err()
        );
    }
    assert_eq!(db.catalog.index_allocator_high_water(), high_water);
    assert!(!db.has_property_index("other"));
    assert!(db.drop_index(owner)?);
    assert!(!db.drop_index(owner)?);
    let replacement = db.create_index(property_request(
        GraphPath::root(),
        Some("shared_name"),
        "value",
    ))?;
    assert!(replacement > owner);
    assert!(db.create_graph("other")?);
    assert!(
        db.create_index(property_request(
            GraphPath::from_components(&["other"])?,
            Some("shared_name"),
            "different"
        ))
        .is_err()
    );
    assert_eq!(
        db.catalog.find_index_by_name("shared_name"),
        Some(replacement)
    );
    assert_eq!(db.catalog.index_count(), 1);
    Ok(())
}

#[test]
#[cfg(feature = "gql")]
fn direct_and_ddl_property_btree_aliases_cannot_create_orphans() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let owner = db.create_index(property_request(GraphPath::root(), None, "value"))?;
    let definition = db.catalog.get_index(owner).ok_or("missing owner")?;
    let high_water = db.catalog.index_allocator_high_water();
    assert!(
        db.create_index(property_request(GraphPath::root(), None, "value"))
            .is_err()
    );
    let mut btree = property_request(GraphPath::root(), Some("btree_alias"), "value");
    btree.kind = IndexCreateKind::BTree;
    assert!(db.create_index(btree).is_err());
    assert!(
        db.execute("CREATE INDEX named_alias FOR (n:DifferentLabel) ON (n.value)")
            .is_err()
    );
    assert_eq!(db.catalog.get_index(owner), Some(definition));
    assert_eq!(db.catalog.find_index_by_name("named_alias"), None);
    assert_eq!(db.catalog.find_index_by_name("btree_alias"), None);
    assert_eq!(db.catalog.index_count(), 1);
    assert_eq!(db.catalog.index_allocator_high_water(), high_water);
    assert!(db.has_property_index("value"));
    Ok(())
}

#[test]
fn root_empty_slash_and_nested_requests_publish_to_distinct_stores() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let root = Arc::clone(crate::database::testing::root_lpg_store(&db));
    // Native construction authority supplies the recursive fixture; every index
    // below is created and dropped through the public committed owner API.
    db.transaction_manager
        .with_write_authority(|| -> TestResult {
            assert!(root.create_graph("")?);
            assert!(root.create_graph("a/b")?);
            assert!(root.create_graph("a")?);
            assert!(root.graph("a").ok_or("missing parent")?.create_graph("b")?);
            Ok(())
        })?;
    let cases = [
        (GraphPath::root(), Arc::clone(&root)),
        (
            GraphPath::from_components(&[""])?,
            root.graph("").ok_or("missing empty child")?,
        ),
        (
            GraphPath::from_components(&["a/b"])?,
            root.graph("a/b").ok_or("missing slash child")?,
        ),
        (
            GraphPath::from_components(&["a", "b"])?,
            root.graph("a")
                .and_then(|a| a.graph("b"))
                .ok_or("missing nested child")?,
        ),
    ];
    let mut owners = Vec::new();
    for (ordinal, (path, store)) in cases.iter().enumerate() {
        let value = Value::from(format!("graph-{ordinal}"));
        let node = db.transaction_manager.with_write_authority(|| {
            let node = store.create_node(&["Doc"]);
            store.set_node_property(node, "value", value.clone());
            node
        });
        let owner = db.create_index(property_request(path.clone(), None, "value"))?;
        assert_eq!(store.current_epoch(), db.current_epoch());
        assert_eq!(db.catalog.index_graph(owner), Some(path.clone()));
        assert_eq!(store.find_nodes_by_property("value", &value), vec![node]);
        assert!(store.has_property_index("value"));
        owners.push(owner);
    }
    assert_eq!(db.catalog.index_count(), 4);
    for (removed, owner) in owners.into_iter().enumerate() {
        assert!(db.drop_index(owner)?);
        for (ordinal, (_, store)) in cases.iter().enumerate() {
            assert_eq!(store.has_property_index("value"), ordinal > removed);
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "gql")]
fn copied_property_index_has_its_own_destination_owner() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE GRAPH source")?;
    session.execute("USE GRAPH source")?;
    session.execute("INSERT (:Doc {value: 'copied'})")?;
    let source_owner = db.create_index(property_request(
        GraphPath::from_components(&["source"])?,
        Some("source_value"),
        "value",
    ))?;
    session.execute("CREATE GRAPH destination AS COPY OF source")?;
    let destination_path = GraphPath::from_components(&["destination"])?;
    let copied = db
        .catalog
        .all_indexes()
        .into_iter()
        .find(|index| index.key.graph() == &destination_path)
        .ok_or("COPY omitted its logical index owner")?;
    assert_ne!(copied.id, source_owner);
    assert_eq!(db.catalog.find_index_by_name(&copied.name), Some(copied.id));
    let source = crate::database::testing::root_lpg_store(&db)
        .graph("source")
        .ok_or("missing source")?;
    let destination = crate::database::testing::root_lpg_store(&db)
        .graph("destination")
        .ok_or("missing destination")?;
    assert!(source.has_property_index("value"));
    assert!(destination.has_property_index("value"));
    assert_eq!(
        destination
            .find_nodes_by_property("value", &Value::from("copied"))
            .len(),
        1
    );
    assert!(db.drop_index(copied.id)?);
    assert!(!destination.has_property_index("value"));
    assert!(source.has_property_index("value"));
    assert_eq!(
        db.catalog
            .get_index(source_owner)
            .ok_or("source owner disappeared")?
            .name,
        "source_value"
    );
    Ok(())
}

#[test]
#[cfg(all(feature = "text-index", feature = "vector-index"))]
fn rebuild_preserves_resolved_owners_and_text_history() -> TestResult {
    use grafeo_core::index::vector::QuantizationType;

    let db = GrafeoDB::with_config(crate::Config::in_memory().with_gc_interval(0))?;
    let node = db.session().create_node_with_props(
        &["Doc"],
        [
            ("body", Value::from("historicaltoken")),
            ("emb", Value::Vector(Arc::from([1.0_f32; 8]))),
        ],
    )?;
    let vector_owner = db.create_index(vector_request())?;
    let text_owner = db.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("text_owner".into()),
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: None,
        },
    })?;
    let vector_before = db
        .catalog
        .get_index(vector_owner)
        .ok_or("missing vector owner")?;
    let text_before = db
        .catalog
        .get_index(text_owner)
        .ok_or("missing text owner")?;
    let high_water = db.catalog.index_allocator_high_water();
    let historic_epoch = db.current_epoch();
    db.set_node_property(node, "body", Value::from("currenttoken"))?;
    let old_text = crate::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("missing text index")?;
    let historic_results = old_text.read().search_visible(
        "historicaltoken",
        10,
        historic_epoch,
        TransactionId::SYSTEM,
        &[],
        &FxHashSet::default(),
    )?;
    assert_eq!(historic_results.len(), 1);
    assert_eq!(historic_results[0].0, node);
    db.rebuild_index(vector_owner)?;
    db.rebuild_index(text_owner)?;
    assert_eq!(
        db.catalog.get_index(vector_owner),
        Some(vector_before.clone())
    );
    assert_eq!(db.catalog.get_index(text_owner), Some(text_before));
    assert_eq!(db.catalog.index_allocator_high_water(), high_water);
    let vector = crate::database::testing::root_lpg_store(&db)
        .get_vector_index("Doc", "emb")
        .ok_or("missing rebuilt vector")?;
    assert_eq!(
        vector_before.configuration,
        IndexConfiguration::Vector {
            config: vector.config().clone(),
            quantization: vector.quantization_type().unwrap_or(QuantizationType::None),
        }
    );
    let text = crate::database::testing::root_lpg_store(&db)
        .get_text_index("Doc", "body")
        .ok_or("missing rebuilt text")?;
    assert_eq!(
        text.read().search_visible(
            "historicaltoken",
            10,
            historic_epoch,
            TransactionId::SYSTEM,
            &[],
            &FxHashSet::default()
        )?,
        historic_results
    );
    let current = db.text_search("Doc", "body", "currenttoken", 10)?;
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].0, node);
    assert!(
        db.text_search("Doc", "body", "historicaltoken", 10)?
            .is_empty()
    );
    Ok(())
}

#[test]
#[cfg(feature = "vector-index")]
fn failed_rebuild_keeps_owner_configuration_and_physical_registration() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let node = db
        .session()
        .create_node_with_props(&["Doc"], [("emb", Value::Vector(Arc::from([1.0_f32; 8])))])?;
    let owner = db.create_index(vector_request())?;
    let definition = db.catalog.get_index(owner).ok_or("missing owner")?;
    let observation = crate::database::testing::root_lpg_store(&db)
        .observe_vector_index("Doc", "emb")
        .ok_or("missing physical registration")?;
    let before = db.vector_search("Doc", "emb", &[1.0; 8], 1, None, None)?;
    assert_eq!(before[0].0, node);
    let epoch = db.current_epoch();
    let high_water = db.catalog.index_allocator_high_water();
    let mut session = db.session();
    session.begin_transaction()?;
    session.rebuild_index_durable(owner)?;
    // The existing logical publication validator rejects this after preparation;
    // a rebuilt physical target must not escape a failed aggregate commit.
    session.pending_graph_type_bindings.lock().insert(
        GraphPath::from_components(&["invalid_binding"])?,
        PendingGraphTypeBinding {
            expected: None,
            replacement: Some("MissingType".into()),
        },
    );
    assert!(session.commit().is_err());
    assert_eq!(db.catalog.get_index(owner), Some(definition));
    assert_eq!(db.catalog.index_allocator_high_water(), high_water);
    assert_eq!(db.current_epoch(), epoch);
    db.transaction_manager.with_write_authority(|| {
        crate::database::testing::root_lpg_store(&db).validate_index_registration(&observation)
    })?;
    assert_eq!(
        db.vector_search("Doc", "emb", &[1.0; 8], 1, None, None)?,
        before
    );
    Ok(())
}

#[test]
fn staged_drop_and_rebuild_reject_recreated_name_aba() -> TestResult {
    for rebuild in [false, true] {
        let db = GrafeoDB::new_in_memory();
        let request = property_request(GraphPath::root(), Some("reused_name"), "value");
        let owner = db.create_index(request.clone())?;
        let mut stale = db.session();
        stale.begin_transaction()?;
        if rebuild {
            stale.rebuild_index_durable(owner)?;
        } else {
            assert!(stale.drop_index_durable(owner)?);
        }
        assert!(db.drop_index(owner)?);
        let replacement = db.create_index(request)?;
        assert!(replacement > owner);
        let definition = db
            .catalog
            .get_index(replacement)
            .ok_or("missing replacement")?;
        let observation = crate::database::testing::root_lpg_store(&db)
            .observe_property_index("value")
            .ok_or("missing replacement registration")?;
        let epoch = db.current_epoch();
        assert!(stale.commit().is_err(), "stale operation rebuild={rebuild}");
        assert_eq!(db.catalog.get_index(owner), None);
        assert_eq!(db.catalog.get_index(replacement), Some(definition));
        assert_eq!(
            db.catalog.find_index_by_name("reused_name"),
            Some(replacement)
        );
        assert_eq!(db.current_epoch(), epoch);
        db.transaction_manager.with_write_authority(|| {
            crate::database::testing::root_lpg_store(&db).validate_index_registration(&observation)
        })?;
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "wal", feature = "vector-index"))]
fn vector_owner_wal_publication_preserves_ids_and_rejects_invalid_requests() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let mut request = vector_request();
    let owner = db.create_index(request.clone())?;
    request.name = None;
    request.property = "second".into();
    let definition = db.catalog.get_index(owner).ok_or("missing owner")?;
    let dir = tempfile::tempdir()?;
    let wal = Arc::new(grafeo_storage::wal::LpgWal::open(dir.path().join("wal"))?);
    let mut session = db.session();
    session.set_wal(Arc::clone(&wal));
    let sequence = wal.record_count();
    let high_water = db.catalog.index_allocator_high_water();
    let second = session.create_index_durable(request.clone())?;
    assert_eq!(second.as_u32(), high_water);
    assert!(wal.record_count() > sequence);
    session.rebuild_index_durable(owner)?;
    assert_eq!(db.catalog.get_index(owner), Some(definition));
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .is_some()
    );
    let sequence = wal.record_count();
    let epoch = db.current_epoch();
    let high_water = db.catalog.index_allocator_high_water();
    assert!(
        session.create_index_durable(request).is_err(),
        "duplicate natural key must reject"
    );
    // A rejected auto-transaction records only its abort, not an owner image.
    assert_eq!(wal.record_count(), sequence + 1);
    assert_eq!(db.current_epoch(), epoch);
    assert_eq!(db.catalog.index_allocator_high_water(), high_water);
    assert!(session.drop_index_durable(owner)?);
    assert!(!session.drop_index_durable(owner)?);
    assert!(session.rebuild_index_durable(owner).is_err());
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "emb")
            .is_none()
    );
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .get_vector_index("Doc", "second")
            .is_some()
    );
    assert!(!session.in_transaction());
    Ok(())
}
