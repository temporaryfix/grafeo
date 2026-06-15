//! Regression: extract_subgraph/remove_orphan_edges must read through the
//! LayeredStore after compact(), not the overlay-only LpgStore.
#![cfg(all(feature = "lpg", feature = "compact-store"))]

use grafeo_engine::GrafeoDB;
use grafeo_engine::GraphStore;

#[test]
fn extract_subgraph_sees_base_tier_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    {
        let session = db.session();
        // Single-statement node+edge insert (confirmed GQL pattern); edges
        // between *existing* nodes would instead use `MATCH ... CREATE`.
        session
            .execute("INSERT (:A {k: 1})-[:T]->(:B {k: 2})")
            .unwrap();
    }
    db.compact().unwrap();

    // The 'A' node now lives in the compacted base tier.
    let layered = db.layered_store().expect("compacted DB has a layered store");
    let a_nodes = layered.nodes_by_label("A");
    assert_eq!(a_nodes.len(), 1);

    // Pre-fix: extract_subgraph reads lpg_store() (overlay only) → base node
    // "does not exist" → Err; its outgoing edge is silently dropped.
    let extract = db
        .extract_subgraph(&a_nodes)
        .expect("base-tier node must be extractable after compact");
    assert_eq!(
        extract.edge_count(),
        1,
        "the base-tier node's outgoing edge must survive the extract"
    );
}
