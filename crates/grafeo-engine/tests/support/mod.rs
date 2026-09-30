//! Shared fixtures for path tests.
//!
//! Every path defect found on 2026-09-12 hid behind a friendly fixture. The
//! shortest-path spec tests pinned *both* endpoints, so the operator's
//! source-by-target cross product collapsed to a single row and its habit of
//! emitting unreachable targets could not show. The walk-enumeration test used a
//! graph small enough that exponential enumeration looked like correct output.
//!
//! So the adversarial structure lives here once, and path tests use it instead of
//! each building something comfortable:
//!
//! - an **unreachable component**, so "returns unreachable nodes" is visible;
//! - a **diamond**, so `ANY SHORTEST` and `ALL SHORTEST` differ observably;
//! - **two routes of different lengths** to one node, so a hop bound bites;
//! - a **cycle** and a **self-loop**, so walk enumeration diverges from
//!   reachability and pruning has something to prune;
//! - **two edge types**, so edge-type filtering is testable;
//! - **parallel OTHER edges**, so ALL multiplicity differs from destination sets;
//! - **no path back to the start**, so the start node appearing as its own target
//!   is always a defect in this fixture and never a legitimate cycle.
//!
//! Reachability controls leave an endpoint free; separate pinned controls prove
//! route identity, intrinsic predicates and correlated input behavior.

#![allow(dead_code)] // Each test binary compiles this module and uses part of it.

use std::collections::{BTreeSet, VecDeque};

use grafeo_engine::GrafeoDB;

/// The fixture's edges, as `(from, to, type)`. This table **is** the fixture:
/// [`reachable_within`] walks it directly, so a reference result never depends on
/// the engine's own path code being correct.
pub const EDGES: &[(&str, &str, &str)] = &[
    // Diamond: two distinct shortest routes from s to d, both length 2.
    ("s", "a", "REL"),
    ("s", "b", "REL"),
    ("a", "d", "REL"),
    ("b", "d", "REL"),
    // Tail past the diamond.
    ("d", "e", "REL"),
    // Two-cycle beyond the diamond: reachable from s, with no route back to s.
    ("e", "f", "REL"),
    ("f", "e", "REL"),
    // Self-loop on a reachable node: one extra walk at every depth, no extra
    // reachable node.
    ("b", "g", "REL"),
    ("g", "g", "REL"),
    // Long way round to d as well, so d is reachable at two different lengths.
    ("a", "h", "REL"),
    ("h", "d", "REL"),
    // A second edge type, reaching a node that REL cannot.
    ("s", "x", "OTHER"),
    ("s", "x", "OTHER"),
    // Component disconnected from s entirely.
    ("u", "v", "REL"),
];

/// Every node named in [`EDGES`], in sorted order.
pub fn node_ids() -> Vec<&'static str> {
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    for (from, to, _) in EDGES {
        ids.insert(from);
        ids.insert(to);
    }
    ids.into_iter().collect()
}

/// Builds the fixture: every node `:Node {id}`, an index on `id`, and [`EDGES`].
pub fn adversarial_path_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let nodes = node_ids()
        .into_iter()
        .map(|id| format!("(:Node {{id: '{id}'}})"))
        .collect::<Vec<_>>()
        .join(", ");
    session
        .execute(&format!("CREATE {nodes}"))
        .expect("fixture nodes");

    // An index on the property the patterns filter by, so path tests exercise the
    // indexed point lookup that anchors a path pattern rather than a full scan.
    session
        .execute("CREATE INDEX path_id FOR (n:Node) ON (n.id)")
        .expect("fixture index");

    for (from, to, edge_type) in EDGES {
        session
            .execute(&format!(
                "MATCH (a:Node {{id: '{from}'}}), (b:Node {{id: '{to}'}}) \
                 CREATE (a)-[:{edge_type}]->(b)"
            ))
            .expect("fixture edge");
    }

    db
}

/// Reference answer for "which nodes are reachable from `start` in `min..=max`
/// hops over `edge_type`", computed by breadth-first search over [`EDGES`].
///
/// Deliberately independent of the engine: a test comparing the engine against
/// this is comparing it against the fixture's definition, not against another
/// code path that could be wrong in the same way.
pub fn reachable_within(start: &str, min: usize, max: usize, edge_type: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start.to_string(), 0usize));
    seen.insert((start.to_string(), 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= min && depth > 0 {
            found.insert(node.clone());
        }
        if depth == max {
            continue;
        }
        for (from, to, kind) in EDGES {
            if *from == node && *kind == edge_type && seen.insert(((*to).to_string(), depth + 1)) {
                queue.push_back(((*to).to_string(), depth + 1));
            }
        }
    }
    found
}

/// Number of distinct walks of length `min..=max` from `start` over `edge_type`.
///
/// The gap between this and [`reachable_within`] is what a reachability-pruned
/// expand must not enumerate: on this fixture the walk count grows with depth
/// while the reachable set does not.
pub fn walk_count(start: &str, min: usize, max: usize, edge_type: &str) -> usize {
    let mut walks = 0;
    let mut queue = VecDeque::new();
    queue.push_back((start.to_string(), 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= min && depth > 0 {
            walks += 1;
        }
        if depth == max {
            continue;
        }
        for (from, to, kind) in EDGES {
            if *from == node && *kind == edge_type {
                queue.push_back(((*to).to_string(), depth + 1));
            }
        }
    }
    walks
}

/// The `id` values of a result's first column, sorted.
pub fn sorted_ids(result: &grafeo_engine::database::QueryResult) -> Vec<String> {
    let mut ids: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            grafeo_common::types::Value::String(s) => s.to_string(),
            other => panic!("expected a string id, got {other:?}"),
        })
        .collect();
    ids.sort();
    ids
}

/// Shared adversarial graph plus an isolated 2×3×4 parallel-edge triangle.
/// Each starting triangle vertex has 24 closed three-edge walks (72 total).
pub fn adversarial_triangle_graph() -> GrafeoDB {
    let db = adversarial_path_graph();
    let session = db.session();
    session
        .execute("CREATE (:Triangle {id:'ta'}), (:Triangle {id:'tb'}), (:Triangle {id:'tc'})")
        .unwrap();
    for (from, to, multiplicity) in [("ta", "tb", 2), ("tb", "tc", 3), ("tc", "ta", 4)] {
        for _ in 0..multiplicity {
            session.execute(&format!("MATCH (a:Triangle {{id:'{from}'}}), (b:Triangle {{id:'{to}'}}) CREATE (a)-[:TRI]->(b)")).unwrap();
        }
    }
    db
}
