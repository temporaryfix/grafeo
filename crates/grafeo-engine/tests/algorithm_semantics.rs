//! Reference-semantics witnesses for the four named algorithms.
//!
//! Each test encodes a **published** definition of the algorithm whose name the engine exposes, on
//! a fixture whose expected output is hand-derivable and that **separates the correct algorithm
//! from the degenerate variant**. A fixture both variants pass is not evidence (Task 16, Step 1).
//! The published definition each test encodes, so a future reader can check the claim rather than
//! trust it:
//!
//! - `grafeo.louvain` — Blondel, Guillaume, Lambiotte & Lefebvre, "Fast unfolding of communities in
//!   large networks", J. Stat. Mech. P10008 (2008): local moving **and** the community-aggregation
//!   phase, the pair repeated until modularity stops improving. The fixture is the ring of cliques
//!   of Fortunato & Barthélemy, "Resolution limit in community detection", PNAS 104(1):36-41
//!   (2007), whose modularity optimum merges adjacent cliques — a second level that the
//!   local-moving phase alone provably cannot reach.
//! - `grafeo.label_propagation` — LDBC Graphalytics CDLP: **synchronous** updates, i.e. every
//!   vertex in an iteration reads the *previous* iteration's labels, ties broken by smallest label.
//! - `grafeo.clustering_coefficient` — the LDBC Graphalytics directed local clustering metric:
//!   take the unique union of incoming and outgoing neighbours, then count directed edges among
//!   those neighbours over `k(k-1)`. This differs from Fagiolo's total directed coefficient when
//!   reciprocal edges change the center's total degree.
//! - `grafeo.sssp` — source resolution must not be hard-wired to the `name` property: a graph keyed
//!   by any other property must still be usable (plan Task 9, Step 1: key on the internal id with
//!   an optional property override).
//!
//! The original six failures are captured in the Task 9 source-bound RED receipt. Algorithms sit
//! behind the non-default `algos` feature, so run:
//!
//! ```bash
//! cargo nextest run -p grafeo-engine --features algos --test algorithm_semantics --no-capture
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "algos"))]

use std::collections::{BTreeMap, BTreeSet};

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::database::QueryResult;
use grafeo_engine::{GrafeoDB, Session};

// ============================================================================
// Shared helpers
// ============================================================================

/// Creates a node carrying `name` and returns the `(name, id)` pair.
fn node(session: &Session, name: &str) -> (String, NodeId) {
    let id = session
        .create_node_with_props(&["V"], [("name", Value::String(name.into()))])
        .expect("create node");
    (name.to_string(), id)
}

/// Re-keys a procedure result by fixture node name, using the internal ids the fixture handed back.
///
/// Procedure rows carry the internal node id in column 0 (`Value::Int64`), which is opaque; every
/// assertion below is stated over fixture names instead.
fn rows_by_name(result: &QueryResult, nodes: &[(String, NodeId)]) -> BTreeMap<String, Vec<Value>> {
    let by_id: BTreeMap<u64, &str> = nodes
        .iter()
        .map(|(name, id)| (id.as_u64(), name.as_str()))
        .collect();
    let mut out = BTreeMap::new();
    for row in result.rows() {
        let raw = row[0]
            .as_int64()
            .unwrap_or_else(|| panic!("node_id column is not Int64: {:?}", row[0]));
        let id = u64::try_from(raw).expect("node id is non-negative");
        let name = by_id
            .get(&id)
            .unwrap_or_else(|| panic!("procedure returned an unknown node id {id}"));
        out.insert((*name).to_string(), row.clone());
    }
    out
}

/// Groups `name -> community id` into `community id -> sorted members`.
fn communities(assignment: &BTreeMap<String, i64>) -> BTreeMap<i64, Vec<String>> {
    let mut groups: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    for (name, community) in assignment {
        groups.entry(*community).or_default().push(name.clone());
    }
    for members in groups.values_mut() {
        members.sort();
    }
    groups
}

/// Recomputes standard undirected modularity from the ring's fixture edges and returned labels.
fn ring_modularity(assignment: &BTreeMap<String, i64>) -> f64 {
    let mut stats: BTreeMap<i64, (usize, usize)> = BTreeMap::new(); // (internal edges, degree sum)
    let mut edges = Vec::with_capacity(RING_TRIANGLES * 4);
    for triangle in 0..RING_TRIANGLES {
        let a = format!("t{triangle}a");
        let b = format!("t{triangle}b");
        let c = format!("t{triangle}c");
        let next_a = format!("t{}a", (triangle + 1) % RING_TRIANGLES);
        edges.extend([
            (a.clone(), b.clone()),
            (b, c.clone()),
            (c.clone(), a),
            (c, next_a),
        ]);
    }
    for (left, right) in edges {
        let left_community = assignment[&left];
        let right_community = assignment[&right];
        stats.entry(left_community).or_default().1 += 1;
        stats.entry(right_community).or_default().1 += 1;
        if left_community == right_community {
            stats.entry(left_community).or_default().0 += 1;
        }
    }
    let edge_count = (RING_TRIANGLES * 4) as f64;
    stats
        .values()
        .map(|(internal, degree_sum)| {
            *internal as f64 / edge_count - (*degree_sum as f64 / (2.0 * edge_count)).powi(2)
        })
        .sum()
}

/// Reads column `col` of every row as an `i64`, keyed by fixture node name.
fn int_column(
    result: &QueryResult,
    nodes: &[(String, NodeId)],
    col: usize,
) -> BTreeMap<String, i64> {
    rows_by_name(result, nodes)
        .into_iter()
        .map(|(name, row)| {
            let value = row[col]
                .as_int64()
                .unwrap_or_else(|| panic!("column {col} is not Int64: {:?}", row[col]));
            (name, value)
        })
        .collect()
}

// ============================================================================
// 1. louvain: the community-aggregation phase
// ============================================================================

/// Number of triangles in the ring fixture. Chosen because the merge condition derived below needs
/// `m > 2e + 2 = 8` for triangles (`e = 3` internal edges), and an even `m` so the optimum pairs up.
const RING_TRIANGLES: usize = 10;

/// Ring of ten triangles: `t{i}a`–`t{i}b`–`t{i}c` is a K3, and `t{i}c -> t{i+1 mod 10}a` closes the
/// ring. 30 nodes, 40 undirected edges (30 clique edges + 10 bridges).
fn ring_of_triangles(session: &Session) -> Vec<(String, NodeId)> {
    let mut nodes = Vec::with_capacity(RING_TRIANGLES * 3);
    for triangle in 0..RING_TRIANGLES {
        for slot in ['a', 'b', 'c'] {
            nodes.push(node(session, &format!("t{triangle}{slot}")));
        }
    }
    for triangle in 0..RING_TRIANGLES {
        let (a, b, c) = (
            nodes[triangle * 3].1,
            nodes[triangle * 3 + 1].1,
            nodes[triangle * 3 + 2].1,
        );
        session.create_edge(a, b, "LINK");
        session.create_edge(b, c, "LINK");
        session.create_edge(c, a, "LINK");
        let next_a = nodes[((triangle + 1) % RING_TRIANGLES) * 3].1;
        session.create_edge(c, next_a, "LINK");
    }
    nodes
}

/// Full Louvain (Blondel et al. 2008) must resolve the **second** level of this fixture.
///
/// Why the fixture discriminates — all numbers below are for `L = 40` edges, `e = 3` internal edges
/// per triangle, `m = 10` triangles, resolution 1.0, with standard modularity
/// `Q = sum_c [ l_c/L - (d_c/2L)^2 ]`:
///
/// * Partition into the 10 triangles: `Q = e/(e+1) - 1/m = 0.75 - 0.10 = 0.65`.
/// * Partition into 5 pairs of adjacent triangles: `Q = (2e+1)/(2e+2) - 2/m = 0.875 - 0.20 =
///   0.675`. The pair partition wins exactly when `m > 2e + 2`, which is the Fortunato &
///   Barthélemy (2007) resolution limit; `10 > 8` here.
/// * **No single-node move reaches it.** Moving the bridge-carrying node `t{i}c` out of its own
///   triangle into the next one leaves behind a community with 1 internal edge and `d = 5` and
///   creates one with 4 internal edges and `d = 11`, giving
///   `1/40 - (5/80)^2 + 4/40 - (11/80)^2 = 0.1022` against the `2 * 0.065 = 0.13` those two
///   triangles contributed before — a loss of 0.0278. Every other node has no neighbour outside its
///   triangle at all. So a local-moving-only implementation is stuck at 10 communities, and only
///   the aggregation phase (where a whole triangle becomes one super-node, and merging two adjacent
///   super-nodes gains `1/L - (2e+2)^2/(2L^2) = 0.025 - 0.020 = +0.005`) can reach 5. Merging a
///   third triangle into a pair loses 0.015, so the second level stops at pairs and a third level
///   changes nothing.
#[test]
fn louvain_resolves_the_two_level_ring_of_cliques() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let nodes = ring_of_triangles(&session);

    let result = session
        .execute("CALL grafeo.louvain()")
        .expect("CALL grafeo.louvain");
    assert_eq!(
        result.columns,
        vec![
            "node_id".to_string(),
            "community_id".to_string(),
            "modularity".to_string()
        ]
    );
    assert_eq!(result.row_count(), nodes.len(), "one row per node");

    let modularity = result.rows()[0][2]
        .as_float64()
        .expect("modularity column is Float64");
    let assignment = int_column(&result, &nodes, 1);
    let groups = communities(&assignment);

    assert_eq!(
        groups.len(),
        RING_TRIANGLES / 2,
        "full louvain must merge adjacent triangles into {} communities (Q = 0.675); \
         local-moving-only stops at the {RING_TRIANGLES} triangles (Q = 0.650). \
         got {} communities, modularity {modularity}, assignment {groups:?}",
        RING_TRIANGLES / 2,
        groups.len()
    );

    for (community, members) in &groups {
        assert_eq!(
            members.len(),
            6,
            "community {community} must be exactly two whole triangles (6 nodes), got {members:?}"
        );
        let triangles: BTreeSet<&str> =
            members.iter().map(|name| &name[..name.len() - 1]).collect();
        assert_eq!(
            triangles.len(),
            2,
            "community {community} must cover exactly two triangles, got {triangles:?}"
        );
        let mut triangle_ids: Vec<usize> = triangles
            .iter()
            .map(|triangle| triangle[1..].parse().expect("ring triangle id"))
            .collect();
        triangle_ids.sort_unstable();
        let adjacent = (triangle_ids[0] + 1) % RING_TRIANGLES == triangle_ids[1]
            || (triangle_ids[1] + 1) % RING_TRIANGLES == triangle_ids[0];
        assert!(
            adjacent,
            "community {community} must contain adjacent triangles, got {triangle_ids:?}"
        );
    }

    let recomputed = ring_modularity(&assignment);
    assert!(
        (recomputed - 27.0 / 40.0).abs() < 1e-12,
        "fixture Q must be 27/40, got {recomputed}"
    );
    assert!(
        (modularity - recomputed).abs() < 1e-12,
        "reported Q {modularity} != recomputed Q {recomputed}"
    );
}

// ============================================================================
// 2. label_propagation: LDBC CDLP is synchronous
// ============================================================================

/// LDBC Graphalytics CDLP is **synchronous**: in each iteration every vertex adopts the most
/// frequent label among its neighbours *as of the previous iteration*, ties broken by the smallest
/// label. This is the classic bipartite case that separates it from an asynchronous sweep.
///
/// Fixture: the 4-cycle `a -> b -> c -> d -> a`. Initial labels are the vertex order `a=0, b=1,
/// c=2, d=3`; CDLP's neighbourhood is undirected (both in and out edges), so each vertex has
/// exactly two neighbours.
///
/// Synchronous, iteration 1 — every vertex reads the initial labels:
/// `a` sees {b:1, d:3} -> 1; `b` sees {a:0, c:2} -> 0; `c` sees {b:1, d:3} -> 1; `d` sees
/// {a:0, c:2} -> 0. State `{a:1, c:1}`, `{b:0, d:0}` — **two** communities.
/// Iteration 2: `a` sees {b:0, d:0} -> 0, `b` sees {a:1, c:1} -> 1, `c` -> 0, `d` -> 1. The labels
/// swap across the bipartition and keep swapping; the *partition* `{a,c} | {b,d}` is invariant, so
/// the assertions below hold at every iteration count and do not depend on the iteration cap.
///
/// Asynchronous (what an in-place sweep in insertion order does): `a` takes 1 from `b`; then `b`
/// already sees `a`'s new 1 and takes 1; `c` takes 1; `d` takes 1 — one single community, converged
/// in one pass. One community vs two is the observable difference.
#[test]
fn label_propagation_is_synchronous_ldbc_cdlp() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let a = node(&session, "a");
    let b = node(&session, "b");
    let c = node(&session, "c");
    let d = node(&session, "d");
    let nodes = vec![a, b, c, d];
    session.create_edge(nodes[0].1, nodes[1].1, "LINK");
    session.create_edge(nodes[1].1, nodes[2].1, "LINK");
    session.create_edge(nodes[2].1, nodes[3].1, "LINK");
    session.create_edge(nodes[3].1, nodes[0].1, "LINK");

    let result = session
        .execute("CALL grafeo.label_propagation({max_iterations: 2})")
        .expect("CALL grafeo.label_propagation");
    assert_eq!(
        result.columns,
        vec!["node_id".to_string(), "community_id".to_string()]
    );
    assert_eq!(result.row_count(), 4, "one row per node");

    let assignment = int_column(&result, &nodes, 1);
    let groups = communities(&assignment);

    assert_eq!(
        groups.len(),
        2,
        "synchronous CDLP splits the 4-cycle into the two sides of its bipartition; \
         an asynchronous sweep collapses it to one community. got {groups:?}"
    );
    assert_eq!(
        assignment["a"], assignment["c"],
        "synchronous CDLP keeps a and c together: {groups:?}"
    );
    assert_eq!(
        assignment["b"], assignment["d"],
        "synchronous CDLP keeps b and d together: {groups:?}"
    );
    assert_ne!(
        assignment["a"], assignment["b"],
        "synchronous CDLP never merges the two sides of a bipartite cycle: {groups:?}"
    );
}

// ============================================================================
// 3. clustering coefficient: a directed variant must be reachable
// ============================================================================

/// The directed local clustering coefficient must be obtainable from the procedure surface.
///
/// Fixture: the transitive directed triangle `a -> b`, `b -> c`, `a -> c` — the minimal case, a
/// directed triangle with one edge "reversed" relative to the cyclic orientation.
///
/// Undirected (what the engine computes today): every vertex has two neighbours joined by an edge,
/// so `C(v) = 2T / (k(k-1)) = 1.0` for all three.
///
/// Directed LDBC: count *ordered* pairs of distinct neighbours. For `a`,
/// `N(a) = {b, c}`, the two ordered pairs are `(b,c)` — present as `b -> c` — and `(c,b)` — absent.
/// So `C_dir(a) = 1/2`. Identically for `b` (`(a,c)` present as `a -> c`, `(c,a)` absent) and `c`
/// (`(a,b)` present, `(b,a)` absent), so the coefficient is 0.5.
///
/// 0.5 vs 1.0 for every vertex is the observable difference. The `{directed: true}` spelling below
/// is the option Task 9 Step 1 must add; today the parameter is accepted and ignored, so the
/// undirected value comes back.
#[test]
fn clustering_coefficient_exposes_the_directed_variant() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let a = node(&session, "a");
    let b = node(&session, "b");
    let c = node(&session, "c");
    let nodes = vec![a, b, c];
    session.create_edge(nodes[0].1, nodes[1].1, "LINK");
    session.create_edge(nodes[1].1, nodes[2].1, "LINK");
    session.create_edge(nodes[0].1, nodes[2].1, "LINK");

    let undirected = session
        .execute("CALL grafeo.clustering_coefficient()")
        .expect("CALL grafeo.clustering_coefficient");
    let undirected_rows = rows_by_name(&undirected, &nodes);
    for name in ["a", "b", "c"] {
        let value = undirected_rows[name][1]
            .as_float64()
            .expect("coefficient column is Float64");
        assert!(
            (value - 1.0).abs() < 1e-9,
            "baseline: the undirected coefficient of {name} in a triangle is 1.0, got {value}"
        );
    }

    let directed = session
        .execute("CALL grafeo.clustering_coefficient({directed: true})")
        .expect("CALL grafeo.clustering_coefficient({directed: true})");
    let directed_rows = rows_by_name(&directed, &nodes);
    for name in ["a", "b", "c"] {
        let value = directed_rows[name][1]
            .as_float64()
            .expect("coefficient column is Float64");
        assert!(
            (value - 0.5).abs() < 1e-9,
            "the directed local clustering coefficient of {name} in the transitive triangle \
             a->b, b->c, a->c is 1/2 (one of the two ordered neighbour pairs is present); \
             got {value} — the undirected value 1.0 means no directed option exists"
        );
    }
}

#[test]
fn directed_clustering_uses_set_semantics_for_parallel_edges() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let a = node(&session, "a");
    let b = node(&session, "b");
    let c = node(&session, "c");
    for _ in 0..3 {
        session.create_edge(a.1, b.1, "LINK");
        session.create_edge(b.1, c.1, "LINK");
        session.create_edge(a.1, c.1, "LINK");
    }

    let nodes = vec![a, b, c];
    let directed = session
        .execute("CALL grafeo.clustering_coefficient({directed: true})")
        .expect("directed clustering with parallel edges");
    let rows = rows_by_name(&directed, &nodes);
    for name in ["a", "b", "c"] {
        let value = rows[name][1]
            .as_float64()
            .expect("coefficient column is Float64");
        assert!(
            (value - 0.5).abs() < 1e-9,
            "parallel edges must not inflate directed neighbour pairs for {name}, got {value}"
        );
    }
}

#[test]
fn directed_clustering_uses_ldbc_unique_neighbor_union() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let center = node(&session, "center");
    let a = node(&session, "a");
    let b = node(&session, "b");
    let c = node(&session, "c");
    session.create_edge(center.1, a.1, "LINK");
    session.create_edge(center.1, b.1, "LINK");
    session.create_edge(center.1, c.1, "LINK");
    session.create_edge(a.1, center.1, "LINK");
    session.create_edge(a.1, b.1, "LINK");

    let nodes = vec![center, a, b, c];
    let directed = session
        .execute("CALL grafeo.clustering_coefficient({directed: true})")
        .expect("directed clustering with reciprocal center edge");
    let rows = rows_by_name(&directed, &nodes);
    let value = rows["center"][1]
        .as_float64()
        .expect("coefficient column is Float64");
    assert!(
        (value - 1.0 / 6.0).abs() < 1e-9,
        "LDBC uses center's unique in/out-neighbor union {{a,b,c}}: one edge among six ordered pairs, got {value}"
    );
}

// ============================================================================
// 4. sssp: source resolution must not be hard-wired to `name`
// ============================================================================

/// `grafeo.sssp` must resolve a source on a graph whose nodes are keyed by `id`, not `name`.
///
/// This one is a plain bug, not a semantic choice: `SsspAlgorithm::execute` parses the source as an
/// integer and otherwise calls `find_nodes_by_property("name", ...)`, so `sssp('v_0')` raises on an
/// id-keyed graph — exactly the benchmark's graph shape.
///
/// Fixture: `v_0 -> v_1 -> v_2` and `v_0 -> v_3`, nodes carrying only `id`. Unweighted distances
/// from `v_0` are `v_0 = 0, v_1 = 1, v_2 = 2, v_3 = 1`.
///
/// Task 9 Step 1 keys sssp on the internal id "with an optional property override", which leaves
/// the surface spelling open, so the witness accepts either route — the bare call resolving the
/// key property, or an explicit override — and requires that at least one works. The distances are
/// then asserted exactly.
#[test]
fn sssp_resolves_a_source_on_an_id_keyed_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut nodes = Vec::new();
    for index in 0..4 {
        let name = format!("v_{index}");
        let id = session
            .create_node_with_props(&["Vertex"], [("id", Value::String(name.as_str().into()))])
            .expect("create node");
        nodes.push((name, id));
    }
    session.create_edge(nodes[0].1, nodes[1].1, "LINK");
    session.create_edge(nodes[1].1, nodes[2].1, "LINK");
    session.create_edge(nodes[0].1, nodes[3].1, "LINK");

    let spellings = [
        "CALL grafeo.sssp('v_0')",
        "CALL grafeo.sssp({source: 'v_0', key: 'id'})",
    ];
    let mut failures = Vec::new();
    let mut resolved = None;
    for query in spellings {
        match session.execute(query) {
            Ok(result) => {
                resolved = Some((query, result));
                break;
            }
            Err(error) => failures.push(format!("{query} -> {error}")),
        }
    }
    let (query, result) = resolved.unwrap_or_else(|| {
        panic!(
            "sssp cannot resolve a source on a graph keyed by `id`; every spelling failed: {}",
            failures.join(" | ")
        )
    });

    assert_eq!(
        result.columns,
        vec!["node_id".to_string(), "distance".to_string()],
        "{query}"
    );
    let rows = rows_by_name(&result, &nodes);
    for (name, expected) in [("v_0", 0.0), ("v_1", 1.0), ("v_2", 2.0), ("v_3", 1.0)] {
        let row = rows
            .get(name)
            .unwrap_or_else(|| panic!("{query} returned no distance for {name}: {rows:?}"));
        let distance = row[1]
            .as_float64()
            .unwrap_or_else(|| panic!("distance column is not Float64: {:?}", row[1]));
        assert!(
            (distance - expected).abs() < 1e-9,
            "{query}: unweighted distance from v_0 to {name} is {expected}, got {distance}"
        );
    }
}
