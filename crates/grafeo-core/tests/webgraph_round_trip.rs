//! Round-trip equivalence regression test for the WebGraph adjacency codec.
//!
//! The invariants:
//! 1. After build, every input edge `(src, dst)` (de-duplicated) appears
//!    in `codec.successors(src)`.
//! 2. After a blob round-trip, the same property holds.
//! 3. `out_degree(u)` equals `successors(u).count()` for every `u`.
//!
//! ```bash
//! cargo test -p grafeo-core --test webgraph_round_trip
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test webgraph_round_trip
//! ```

use grafeo_core::codec::{WebGraphBuilder, WebGraphCodec};
use proptest::prelude::*;
use std::collections::BTreeSet;

fn edge_strategy(num_nodes: u64) -> impl Strategy<Value = (u64, u64)> {
    (0u64..num_nodes, 0u64..num_nodes)
}

fn graph_strategy() -> impl Strategy<Value = (u64, Vec<(u64, u64)>)> {
    (1u64..=24).prop_flat_map(|n| {
        let edges = proptest::collection::vec(edge_strategy(n), 0..=80);
        (Just(n), edges)
    })
}

fn check_round_trip(num_nodes: u64, edges: &[(u64, u64)]) {
    // Build expected adjacency: per source, the set of distinct destinations
    // in ascending order.
    let mut expected: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); num_nodes as usize];
    for &(s, d) in edges {
        expected[s as usize].insert(d);
    }

    let mut builder = WebGraphBuilder::new(num_nodes);
    for &(s, d) in edges {
        builder.add_edge(s, d).unwrap();
    }
    let codec = builder.build();

    // Invariant: each node's successors match the de-duped sorted set.
    for u in 0..num_nodes {
        let got: Vec<u64> = codec.successors(u).collect();
        let want: Vec<u64> = expected[u as usize].iter().copied().collect();
        assert_eq!(got, want, "successors of {u} mismatched");
        assert_eq!(codec.out_degree(u), got.len() as u64);
    }

    // Total edges = sum of distinct successors.
    let total: u64 = expected.iter().map(|s| s.len() as u64).sum();
    assert_eq!(codec.num_edges(), total);

    // Blob round-trip.
    let blob = codec.to_bytes();
    let reopened = WebGraphCodec::from_bytes(&blob).expect("from_bytes");
    for u in 0..num_nodes {
        let got: Vec<u64> = reopened.successors(u).collect();
        let want: Vec<u64> = expected[u as usize].iter().copied().collect();
        assert_eq!(got, want, "reopened successors of {u} mismatched");
    }
    assert_eq!(reopened.num_edges(), total);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Every input edge survives encoding and blob round-trip; per-node
    /// successor lists match the de-duped, sorted input.
    #[test]
    fn webgraph_round_trip_arbitrary_small_graphs(
        (num_nodes, edges) in graph_strategy()
    ) {
        check_round_trip(num_nodes, &edges);
    }
}

// ── Fixed regression seeds ───────────────────────────────────────

#[test]
fn webgraph_round_trip_empty_graph() {
    check_round_trip(0, &[]);
}

#[test]
fn webgraph_round_trip_isolated_nodes() {
    check_round_trip(7, &[]);
}

#[test]
fn webgraph_round_trip_self_loops() {
    let edges = [(0u64, 0), (3, 3), (5, 5)];
    check_round_trip(6, &edges);
}

#[test]
fn webgraph_round_trip_dst_below_src() {
    // First-gap is signed; exercise dst < src.
    let edges = [(9u64, 0), (9, 1), (9, 9)];
    check_round_trip(10, &edges);
}

#[test]
fn webgraph_round_trip_dense_node() {
    // One node with edges to every other.
    let edges: Vec<(u64, u64)> = (0u64..20).map(|d| (0u64, d)).collect();
    check_round_trip(20, &edges);
}

#[test]
fn webgraph_round_trip_duplicated_edges() {
    let edges = [(0u64, 1), (0, 1), (0, 1), (0, 2), (1, 0), (1, 0)];
    check_round_trip(3, &edges);
}
