//! Integration tests for `GrafeoDB::extract_subgraph` — produces a new
//! in-memory database containing exactly the requested nodes plus
//! every edge whose SOURCE is in the request set (source-side ownership),
//! with source-allocated NodeIds and EdgeIds preserved.

#![cfg(feature = "lpg")]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

#[test]
fn extract_subgraph_carries_schema_and_indexes() {
    let source = GrafeoDB::new_in_memory();
    source
        .session()
        .execute("CREATE NODE TYPE Concept (id STRING, label STRING)")
        .expect("ddl concept");
    source
        .session()
        .execute("CREATE NODE TYPE NicheDescriptor (id STRING, niche STRING)")
        .expect("ddl niche");
    source
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: None,
            property: "id".into(),
            kind: grafeo_engine::IndexCreateKind::Property,
        })
        .expect("create property index");

    let a = source.create_node(&["Concept"]);
    source
        .set_node_property(a, "id", Value::String("concept:bitter".into()))
        .expect("set node property");

    let target = source.extract_subgraph(&[a]).expect("extract");

    // Schema survived — NicheDescriptor was declared in source but no node
    // with that label is in the extract. If the full catalog was carried,
    // re-declaring the type must fail with "already exists".
    let schema = serde_json::to_string(&target.schema()).expect("schema json");
    assert!(schema.contains("Concept"), "Concept type carried");

    let redeclare = target
        .session()
        .execute("CREATE NODE TYPE NicheDescriptor (id STRING, niche STRING)");
    assert!(
        redeclare.is_err(),
        "NicheDescriptor type carried (full schema, not subset) — \
         re-declaration must fail with type-already-exists"
    );

    // Property index survived.
    assert!(target.has_property_index("id"), "id property index carried");
}

#[test]
fn extract_subgraph_preserves_selected_property_owner_and_allocator_floor() {
    let source = GrafeoDB::new_in_memory();
    let retired = source
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: Some("retiredKey".into()),
            label: None,
            property: "retiredKey".into(),
            kind: grafeo_engine::IndexCreateKind::Property,
        })
        .expect("create retired index");
    assert_eq!(retired.as_u32(), 0);
    assert!(source.drop_index(retired).expect("drop retired index"));

    let owner = source
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: Some("namedId".into()),
            label: None,
            property: "id".into(),
            kind: grafeo_engine::IndexCreateKind::Property,
        })
        .expect("create named id index");
    assert_eq!(owner.as_u32(), 1);

    let a = source.create_node(&["A"]);
    source
        .set_node_property(a, "id", Value::String("shared".into()))
        .expect("set A id");
    let b = source.create_node(&["B"]);
    source
        .set_node_property(b, "id", Value::String("shared".into()))
        .expect("set B id");

    let target = source
        .extract_subgraph(&[a])
        .expect("extract selected node");
    assert_ne!(target.store_id(), source.store_id());
    assert_eq!(
        target.find_nodes_by_property("id", &Value::String("shared".into())),
        vec![a]
    );
    assert!(target.drop_index(owner).expect("drop copied index"));

    let replacement = target
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: Some("newKey".into()),
            label: None,
            property: "newKey".into(),
            kind: grafeo_engine::IndexCreateKind::Property,
        })
        .expect("create replacement index");
    assert_eq!(replacement.as_u32(), owner.as_u32() + 1);
}

#[test]
#[cfg(all(feature = "text-index", feature = "vector-index"))]
fn extract_subgraph_preserves_nondefault_text_vector_indexes_for_selected_nodes() {
    let source = GrafeoDB::new_in_memory();
    let text_owner = source
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: Some("bodySearch".into()),
            label: Some("Doc".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: Some(1),
            },
        })
        .expect("create text index");
    let vector_owner = source
        .create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: Some("embeddingSearch".into()),
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("euclidean".into()),
                m: Some(12),
                ef_construction: Some(32),
                ef: Some(24),
                quantization: None,
            },
        })
        .expect("create vector index");

    let make_doc = |body: &str, embedding: [f32; 3]| {
        let id = source.create_node(&["Doc"]);
        source
            .set_node_property(id, "body", Value::String(body.into()))
            .expect("set body");
        source
            .set_node_property(id, "embedding", Value::Vector(embedding.to_vec().into()))
            .expect("set embedding");
        id
    };
    let a = make_doc("x alpha", [3.0, 4.0, 0.0]);
    let b = make_doc("x beta", [0.0, 0.0, 2.0]);
    let c = make_doc("x outside", [0.0, 3.0, 0.0]);

    let target = source
        .extract_subgraph(&[a, b])
        .expect("extract selected docs");
    assert_ne!(target.store_id(), source.store_id());
    let mut text = target
        .text_search("Doc", "body", "x", 10)
        .expect("target text search");
    text.sort_by_key(|(id, _)| *id);
    assert_eq!(
        text.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![a, b]
    );
    for (_, score) in &text {
        assert!(
            (*score - 0.1823215568).abs() < 1e-9,
            "unexpected BM25 score: {score}"
        );
    }
    assert!(!text.iter().any(|(id, _)| *id == c));

    let vectors = target
        .vector_search("Doc", "embedding", &[0.0, 0.0, 0.0], 2, None, None)
        .expect("target vector search");
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].0, b);
    assert!(
        (vectors[0].1 - 2.0).abs() < 0.01,
        "Euclidean metric was not retained"
    );
    assert!(vectors.iter().all(|(id, _)| *id != c));
    let snapshot = target.export_snapshot().expect("export target");
    let reopened = GrafeoDB::import_snapshot(&snapshot).expect("reopen target snapshot");
    assert_ne!(reopened.store_id(), source.store_id());
    let reopened_text = reopened.text_search("Doc", "body", "x", 10).unwrap();
    assert_eq!(reopened_text.len(), 2);
    for (_, score) in reopened_text {
        assert!(
            (score - 0.1823215568).abs() < 1e-9,
            "unexpected reopened BM25 score: {score}"
        );
    }
    assert_eq!(
        reopened
            .vector_search("Doc", "embedding", &[0.0, 0.0, 0.0], 2, None, None)
            .unwrap()[0]
            .0,
        b
    );
    assert!(
        target
            .drop_index(text_owner)
            .expect("drop copied text owner")
    );
    assert!(
        target
            .drop_index(vector_owner)
            .expect("drop copied vector owner")
    );
}

#[test]
fn extract_subgraph_preserves_node_ids_and_carries_all_outgoing_edges() {
    let source = GrafeoDB::new_in_memory();
    let a = source.create_node(&["Concept"]);
    source
        .set_node_property(a, "id", Value::String("concept:bitter".into()))
        .expect("set node property");
    let b = source.create_node(&["Concept"]);
    source
        .set_node_property(b, "id", Value::String("concept:sweet".into()))
        .expect("set node property");
    let c = source.create_node(&["NicheDescriptor"]);
    source
        .set_node_property(c, "id", Value::String("tea:bitter".into()))
        .expect("set node property");

    // Outgoing edge with dst inside the request set (c → a)
    source.create_edge(c, a, "MAPS_TO_CONCEPT");
    // Outgoing edge with dst OUTSIDE the request set (c → b).
    // Under source-side ownership, this edge is still carried because
    // `c` is in the request set. `b` is not carried, so this edge will
    // have a dangling dst in the extract — that's intentional and
    // resolves cleanly when a sibling extract containing `b` is
    // merged via `open_multi`.
    source.create_edge(c, b, "RELATED");

    let target = source.extract_subgraph(&[c, a]).expect("extract_subgraph");

    // Only the two requested nodes are carried; b is not.
    assert_eq!(target.node_count(), 2);
    // Both of c's outgoing edges are carried, even though c→b has
    // a dst (b) that lives outside the request set.
    assert_eq!(target.edge_count(), 2);

    // The MAPS_TO_CONCEPT edge resolves cleanly — both endpoints
    // are in the extract.
    let result = target
        .session()
        .execute("MATCH (n:NicheDescriptor)-[:MAPS_TO_CONCEPT]->(c:Concept) RETURN c.id")
        .expect("query");
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::String("concept:bitter".into()),
        "MAPS_TO_CONCEPT resolves against the in-set destination"
    );

    // The RELATED edge has a dangling dst (b is not in the extract).
    // A Cypher MATCH that tries to bind the dst won't find a node,
    // so the query returns zero rows. This is the expected behavior
    // for an extract viewed in isolation; sibling-extract merge via
    // open_multi is what restores the dst.
    let dangling = target
        .session()
        .execute("MATCH (n:NicheDescriptor)-[:RELATED]->(d) RETURN d.id")
        .expect("query");
    assert_eq!(
        dangling.rows().len(),
        0,
        "RELATED edge's dst is not in the extract; Cypher MATCH does not bind"
    );
}

#[test]
#[cfg(feature = "gql")]
fn extract_subgraph_preserves_property_history() {
    let source = GrafeoDB::new_in_memory();
    let mut session = source.session();

    // tx1: create the node with score=1.
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Tracked {score: 1})").unwrap();
    session.commit().unwrap();

    // tx2: advance to score=2; capture epoch after this commit.
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Tracked) SET n.score = 2")
        .unwrap();
    session.commit().unwrap();
    let epoch_at_v2 = source.current_epoch();

    // tx3: advance to score=3 (the final / current value).
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Tracked) SET n.score = 3")
        .unwrap();
    session.commit().unwrap();

    // The first (and only) node is NodeId(0).
    let node = NodeId::new(0);
    let target = source.extract_subgraph(&[node]).expect("extract");

    // Current value must be the latest write.
    let current = target
        .session()
        .execute("MATCH (n:Tracked) RETURN n.score")
        .expect("current query");
    assert_eq!(
        current.rows()[0][0],
        Value::Int64(3),
        "latest value preserved after extract"
    );

    // Intermediate epoch must be readable — proves the version chain was
    // copied into the extracted DB, not just the final value.
    let mid = target
        .session()
        .execute_at_epoch("MATCH (n:Tracked) RETURN n.score", epoch_at_v2)
        .expect("epoch query");
    assert_eq!(mid.rows().len(), 1, "node visible at intermediate epoch");
    assert_eq!(
        mid.rows()[0][0],
        Value::Int64(2),
        "intermediate epoch returns score=2"
    );
}

// --------------------------------------------------------------------
// Proptest — extract two disjoint halves of a random graph, merge via
// open_multi, assert the result is observationally equivalent to the
// source.
//
// `extract_subgraph` uses source-side ownership: every edge whose src
// is in the requested node set is carried, including edges whose dst
// lives outside the set. When two disjoint halves of a graph's nodes
// are each extracted, every edge appears in exactly one extract (the
// one containing its src node). `open_multi` merges the two extracts
// and validates endpoints across the union. The guarantee tested here:
//
//   1. Node count round-trips exactly.
//   2. Edge bag round-trips exactly (every edge, not just intra-partition).
//
// This is the headline correctness property for the
// "extract per chunk, merge at load" workflow.
// --------------------------------------------------------------------

use proptest::prelude::*;

#[derive(Debug, Clone)]
struct GraphSpec {
    nodes: Vec<(String, String)>,       // (label, id_prop)
    edges: Vec<(usize, usize, String)>, // (src_index, dst_index, type)
}

fn graph_spec() -> impl Strategy<Value = GraphSpec> {
    // Small graphs only — proptest enumerates ~64 cases by default and
    // each one builds + extracts + merges + queries, so keep it cheap.
    let nodes = prop::collection::vec(
        (prop_oneof![Just("A"), Just("B"), Just("C")], "[a-z]{1,4}")
            .prop_map(|(l, id)| (l.to_string(), id)),
        1..8usize,
    );
    nodes.prop_flat_map(|nodes| {
        let n = nodes.len();
        let edges = prop::collection::vec(
            (
                0..n,
                0..n,
                prop_oneof![Just("R1"), Just("R2"), Just("R3")].prop_map(|s| s.to_string()),
            ),
            0..8usize,
        );
        edges.prop_map(move |edges| GraphSpec {
            nodes: nodes.clone(),
            edges,
        })
    })
}

fn build_graph(spec: &GraphSpec) -> (GrafeoDB, Vec<NodeId>) {
    let db = GrafeoDB::new_in_memory();
    let mut ids = Vec::with_capacity(spec.nodes.len());
    let mut seen_id_props = std::collections::HashSet::new();
    for (label, id_prop) in &spec.nodes {
        // Skip duplicate id_props — the test asserts on count() which
        // would be confounded by two nodes sharing an `id`.
        if !seen_id_props.insert(id_prop.clone()) {
            continue;
        }
        let nid = db.create_node(&[label.as_str()]);
        db.set_node_property(nid, "id", Value::String(id_prop.clone().into()))
            .expect("set node property");
        ids.push(nid);
    }
    let mut seen_edges = std::collections::HashSet::new();
    for (src_idx, dst_idx, edge_type) in &spec.edges {
        let Some(&src) = ids.get(*src_idx) else {
            continue;
        };
        let Some(&dst) = ids.get(*dst_idx) else {
            continue;
        };
        // Dedup (src, dst, type) so we don't double-add parallel edges
        // — Grafeo allows them but proptest will reuse triples and
        // our assertions count distinct (src, dst, type) hops.
        if !seen_edges.insert((src, dst, edge_type.clone())) {
            continue;
        }
        db.create_edge(src, dst, edge_type);
    }
    (db, ids)
}

/// Bag of (src.id, dst.id, type) triples over all edges in a DB.
fn edge_triples_bag(db: &GrafeoDB) -> std::collections::BTreeMap<(String, String, String), usize> {
    let result = db
        .session()
        .execute("MATCH (a)-[r]->(b) RETURN a.id, b.id, type(r)")
        .expect("edge_triples_bag query");
    let mut bag = std::collections::BTreeMap::new();
    for row in result.rows() {
        let Value::String(src_id) = &row[0] else {
            continue;
        };
        let Value::String(dst_id) = &row[1] else {
            continue;
        };
        let Value::String(rel_type) = &row[2] else {
            continue;
        };
        let key = (src_id.to_string(), dst_id.to_string(), rel_type.to_string());
        *bag.entry(key).or_insert(0) += 1;
    }
    bag
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    /// Verifies the round-trip properties of `extract_subgraph` + `open_multi`:
    ///
    /// 1. All nodes survive (node count is exact).
    /// 2. The full edge bag round-trips exactly — including edges that cross
    ///    the partition boundary. Under source-side ownership, every edge is
    ///    carried by the extract that contains its src node, so merging the
    ///    two halves via `open_multi` reproduces the complete source edge set.
    #[test]
    fn extract_then_merge_round_trips(spec in graph_spec()) {
        let (source, ids) = build_graph(&spec);
        // Skip degenerate cases (empty graphs after dedup don't
        // exercise the merge path).
        prop_assume!(!ids.is_empty());

        // Partition: alternating indices into two halves.
        // half_a gets ids[0], ids[2], ids[4], ...
        // half_b gets ids[1], ids[3], ids[5], ...
        let half_a: Vec<NodeId> = ids.iter().step_by(2).copied().collect();
        let half_b: Vec<NodeId> = ids.iter().skip(1).step_by(2).copied().collect();

        // Skip single-node cases — they produce an empty half_b, which
        // doesn't exercise the merge path. proptest will retry with a
        // different seed.
        prop_assume!(!half_b.is_empty());

        let ext_a = source.extract_subgraph(&half_a).expect("extract a");
        let ext_b = source.extract_subgraph(&half_b).expect("extract b");

        let bytes_a = ext_a.export_snapshot().expect("export a");
        let bytes_b = ext_b.export_snapshot().expect("export b");
        drop(ext_a);
        drop(ext_b);

        let merged = GrafeoDB::open_multi([bytes_a.as_slice(), bytes_b.as_slice()])
            .expect("open_multi must accept disjoint sibling extracts");

        prop_assert_eq!(merged.node_count(), source.node_count(),
            "node count must round-trip");

        let source_bag = edge_triples_bag(&source);
        let merged_bag = edge_triples_bag(&merged);
        prop_assert_eq!(merged_bag, source_bag,
            "edge triples must round-trip exactly via extract + open_multi");
    }
}
