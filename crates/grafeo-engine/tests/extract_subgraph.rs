//! Integration tests for `GrafeoDB::extract_subgraph` — produces a new
//! in-memory database containing exactly the requested nodes plus
//! every edge whose endpoints are both in the request set, with
//! source-allocated NodeIds and EdgeIds preserved.

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
    source.create_property_index("id");

    let a = source.create_node(&["Concept"]);
    source.set_node_property(a, "id", Value::String("concept:bitter".into()));

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
fn extract_subgraph_preserves_node_ids_and_includes_interior_edges() {
    let source = GrafeoDB::new_in_memory();
    let a = source.create_node(&["Concept"]);
    source.set_node_property(a, "id", Value::String("concept:bitter".into()));
    let b = source.create_node(&["Concept"]);
    source.set_node_property(b, "id", Value::String("concept:sweet".into()));
    let c = source.create_node(&["NicheDescriptor"]);
    source.set_node_property(c, "id", Value::String("tea:bitter".into()));

    // Interior edge (c → a, both in extract set)
    source.create_edge(c, a, "MAPS_TO_CONCEPT");
    // Boundary edge (c → b, b NOT in extract set; must be excluded)
    source.create_edge(c, b, "RELATED");

    let target = source
        .extract_subgraph(&[c, a])
        .expect("extract_subgraph");

    // NodeIds preserved exactly — caller can union with a sibling extract.
    assert_eq!(target.node_count(), 2);
    assert_eq!(target.edge_count(), 1);

    let result = target
        .session()
        .execute("MATCH (n:NicheDescriptor)-[:MAPS_TO_CONCEPT]->(c:Concept) RETURN c.id")
        .expect("query");
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::String("concept:bitter".into()),
        "interior edge resolves; boundary edge excluded"
    );
}

#[cfg(feature = "temporal")]
#[test]
fn extract_subgraph_preserves_property_history() {
    let source = GrafeoDB::new_in_memory();
    let mut session = source.session();

    // tx1: create the node with score=1.
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Tracked {score: 1})")
        .unwrap();
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
// source for edges that are interior to one partition half.
//
// `extract_subgraph` carries interior edges — those whose src AND dst
// are both in the requested node set. Cross-partition edges (one
// endpoint in each half) are dropped by both extracts. `open_multi`
// unions the two extracted edge bags; it does NOT re-stitch edges that
// neither extract carried. The guarantees tested here are therefore:
//
//   1. Node count round-trips exactly.
//   2. Every intra-partition edge round-trips exactly.
//   3. No spurious edges appear in the merged DB.
//
// This is the headline correctness property for the
// "extract per chunk, merge at load" workflow.
// --------------------------------------------------------------------

use proptest::prelude::*;

#[derive(Debug, Clone)]
struct GraphSpec {
    nodes: Vec<(String, String)>, // (label, id_prop)
    edges: Vec<(usize, usize, String)>, // (src_index, dst_index, type)
}

fn graph_spec() -> impl Strategy<Value = GraphSpec> {
    // Small graphs only — proptest enumerates ~64 cases by default and
    // each one builds + extracts + merges + queries, so keep it cheap.
    let nodes = prop::collection::vec(
        (
            prop_oneof![Just("A"), Just("B"), Just("C")],
            "[a-z]{1,4}",
        )
            .prop_map(|(l, id)| (l.to_string(), id)),
        1..8usize,
    );
    nodes.prop_flat_map(|nodes| {
        let n = nodes.len();
        let edges = prop::collection::vec(
            (
                0..n,
                0..n,
                prop_oneof![Just("R1"), Just("R2"), Just("R3")]
                    .prop_map(|s| s.to_string()),
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
        db.set_node_property(nid, "id", Value::String(id_prop.clone().into()));
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
fn edge_triples_bag(
    db: &GrafeoDB,
) -> std::collections::BTreeMap<(String, String, String), usize> {
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
        let key = (
            src_id.to_string(),
            dst_id.to_string(),
            rel_type.to_string(),
        );
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
    /// 2. All intra-partition edges survive (no interior edges are dropped or
    ///    duplicated by the extract → snapshot → open_multi pipeline).
    /// 3. No spurious edges appear (merged edge count ≤ source edge count).
    ///
    /// Cross-partition edges (those spanning the two halves) are intentionally
    /// dropped by `extract_subgraph`, which only carries interior edges. The
    /// test accounts for this by computing the expected merged bag as the subset
    /// of source edges that are wholly within one partition half.
    #[test]
    fn extract_then_merge_round_trips(spec in graph_spec()) {
        let (source, ids) = build_graph(&spec);
        // Skip degenerate cases (empty graphs after dedup don't
        // exercise the merge path).
        prop_assume!(!ids.is_empty());

        // Partition: alternating indices into two halves.
        // half_a gets ids[0], ids[2], ids[4], ...
        // half_b gets ids[1], ids[3], ids[5], ...
        let set_a: std::collections::HashSet<NodeId> =
            ids.iter().step_by(2).copied().collect();
        let set_b: std::collections::HashSet<NodeId> =
            ids.iter().skip(1).step_by(2).copied().collect();

        // Skip single-node cases — they produce an empty half_b, which
        // doesn't exercise the merge path. proptest will retry with a
        // different seed.
        prop_assume!(!set_b.is_empty());

        let half_a: Vec<NodeId> = set_a.iter().copied().collect();
        let half_b: Vec<NodeId> = set_b.iter().copied().collect();

        let ext_a = source.extract_subgraph(&half_a).expect("extract a");
        let ext_b = source.extract_subgraph(&half_b).expect("extract b");

        let bytes_a = ext_a.export_snapshot().expect("export a");
        let bytes_b = ext_b.export_snapshot().expect("export b");
        drop(ext_a);
        drop(ext_b);

        let merged = GrafeoDB::open_multi(&[bytes_a.as_slice(), bytes_b.as_slice()])
            .expect("open_multi must accept disjoint sibling extracts");

        // Property 1: node count round-trips exactly.
        prop_assert_eq!(
            merged.node_count(),
            source.node_count(),
            "node count must round-trip"
        );

        let merged_bag = edge_triples_bag(&merged);
        let source_bag = edge_triples_bag(&source);

        // Property 3: no spurious edges. Every edge in merged must
        // have been in source.
        for (key, &merged_count) in &merged_bag {
            let source_count = source_bag.get(key).copied().unwrap_or(0);
            prop_assert!(
                merged_count <= source_count,
                "spurious edge in merged DB: {:?} appears {} times in merged but \
                 only {} times in source",
                key, merged_count, source_count
            );
        }

        // Property 2: every intra-partition edge round-trips. Build the
        // expected bag as source edges with both endpoints in the same half.
        let mut expected_bag = std::collections::BTreeMap::<(String, String, String), usize>::new();
        {
            let result = source
                .session()
                .execute("MATCH (a)-[r]->(b) RETURN id(a), a.id, id(b), b.id, type(r)")
                .expect("source edge query with node ids");
            for row in result.rows() {
                // id(a) returns the internal NodeId as an integer.
                let Value::Int64(src_internal) = &row[0] else { continue };
                let Value::String(src_id) = &row[1] else { continue };
                let Value::Int64(dst_internal) = &row[2] else { continue };
                let Value::String(dst_id) = &row[3] else { continue };
                let Value::String(rel_type) = &row[4] else { continue };

                let src_node = NodeId::new(*src_internal as u64);
                let dst_node = NodeId::new(*dst_internal as u64);

                // Include this edge in the expected bag only if both
                // endpoints are in the same partition half.
                let intra = (set_a.contains(&src_node) && set_a.contains(&dst_node))
                    || (set_b.contains(&src_node) && set_b.contains(&dst_node));
                if intra {
                    let key = (
                        src_id.to_string(),
                        dst_id.to_string(),
                        rel_type.to_string(),
                    );
                    *expected_bag.entry(key).or_insert(0) += 1;
                }
            }
        }

        prop_assert_eq!(
            merged_bag,
            expected_bag,
            "intra-partition edge triples must round-trip exactly via \
             extract + open_multi (cross-partition edges are intentionally \
             dropped by extract_subgraph)"
        );
    }
}
