//! Regression: extract_subgraph/remove_orphan_edges must read through the
//! LayeredStore after compact(), not the overlay-only LpgStore.
#![cfg(all(feature = "lpg", feature = "compact-store"))]

use grafeo_engine::GrafeoDB;

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
    let layered = db
        .layered_store()
        .expect("compacted DB has a layered store");
    let a_nodes = layered.graph_store().nodes_by_label("A");
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

#[test]
#[cfg(feature = "gql")]
fn indexed_subset_after_compact_retains_history_through_snapshot_reopen()
-> Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::types::{GraphPath, Value};
    use grafeo_engine::{CreateIndexRequest, IndexCreateKind};

    let mut source = GrafeoDB::new_in_memory();
    let owner = source.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("scores".into()),
        label: None,
        property: "score".into(),
        kind: IndexCreateKind::Property,
    })?;
    #[cfg(feature = "text-index")]
    source.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("bodies".into()),
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: Some(3),
        },
    })?;
    let before_birth = source.current_epoch();
    let mut session = source.session();
    session.begin_transaction()?;
    session.execute("INSERT (:Doc:Old {key: 'A', score: 10, body: 'needle alpha ox'})")?;
    session.execute("INSERT (:Doc {key: 'B', score: 10, body: 'needle bravo ox'})")?;
    session.execute("INSERT (:Doc {key: 'C', score: 10, body: 'needle outside ox'})")?;
    session.commit()?;
    let birth = source.current_epoch();
    session.begin_transaction()?;
    session.execute("MATCH (n:Doc {key: 'A'}) SET n.score = 20, n.body = 'changed alpha ox'")?;
    session.execute("MATCH (n:Doc {key: 'A'}) REMOVE n:Old")?;
    session.execute("MATCH (n:Doc {key: 'A'}) SET n:New")?;
    session.commit()?;
    drop(session);

    let ids = source.graph_store().nodes_by_label("Doc");
    let identify = |key: &str| {
        ids.iter()
            .copied()
            .find(|&id| {
                source
                    .graph_store()
                    .get_node(id)
                    .and_then(|node| node.get_property("key").cloned())
                    == Some(Value::from(key))
            })
            .ok_or("fixture node missing")
    };
    let a = identify("A")?;
    let b = identify("B")?;
    let c = identify("C")?;
    source.compact()?;
    assert!(source.layered_store().is_some());
    let extracted = source.extract_subgraph(&[a, b])?;
    assert_ne!(extracted.store_id(), source.store_id());
    let reopened = GrafeoDB::import_snapshot(&extracted.export_snapshot()?)?;

    for target in [&extracted, &reopened] {
        assert_eq!(target.node_count(), 2);
        assert!(target.graph_store().get_node(c).is_none());
        assert!(target.has_property_index("score"));
        assert!(
            target.get_node_at_epoch(a, before_birth).is_none(),
            "birth must not move to epoch zero"
        );
        let old = target
            .get_node_at_epoch(a, birth)
            .ok_or("historical A missing")?;
        assert!(old.has_label("Old"));
        assert!(!old.has_label("New"));
        assert_eq!(old.get_property("score"), Some(&Value::Int64(10)));
        let current = target
            .graph_store()
            .get_node(a)
            .ok_or("current A missing")?;
        assert!(current.has_label("New"));
        assert!(!current.has_label("Old"));
        assert_eq!(current.get_property("score"), Some(&Value::Int64(20)));
        let query = "MATCH (n:Doc {score: 10}) RETURN n.key ORDER BY n.key";
        assert_eq!(
            target.session().execute(query)?.rows(),
            &[vec![Value::from("B")]]
        );
        assert_eq!(
            target.session().execute_at_epoch(query, birth)?.rows(),
            &[vec![Value::from("A")], vec![Value::from("B")]],
            "historical property index must retain A/B and exclude source-only C",
        );
        #[cfg(feature = "text-index")]
        {
            let hits = target.text_search("Doc", "body", "needle", 10)?;
            assert_eq!(hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![b]);
            assert!(target.text_search("Doc", "body", "outside", 10)?.is_empty());
            assert!(target.text_search("Doc", "body", "ox", 10)?.is_empty());
        }
    }
    assert!(
        reopened.drop_index(owner)?,
        "canonical owner ID survives both transfers"
    );
    Ok(())
}
