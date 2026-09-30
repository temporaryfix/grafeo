//! Explicit subset-index union rebuilds corpus state without weakening exact union.
#![cfg(all(feature = "lpg", feature = "text-index"))]

use grafeo_common::types::{GraphPath, Value};
use grafeo_engine::{
    CreateIndexRequest, GrafeoDB, IndexCreateKind, IndexMergePolicy, OpenMultiOptions,
};

#[test]
fn extracted_siblings_rebuild_union_scores_and_preserve_transport_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let source = GrafeoDB::new_in_memory();
    let owner = source.create_index(CreateIndexRequest {
        graph: GraphPath::root(),
        name: Some("docText".into()),
        label: Some("Doc".into()),
        property: "body".into(),
        kind: IndexCreateKind::Text {
            min_token_length: Some(3),
        },
    })?;
    let mut ids = Vec::new();
    for body in ["needle alpha ox", "needle bravo ox", "other gamma ox"] {
        let id = source.create_node(&["Doc"]);
        source.set_node_property(id, "body", Value::from(body))?;
        ids.push(id);
    }
    let edge = source.create_edge(ids[0], ids[2], "CROSSES_SUBSET");
    let left = source.extract_subgraph(&ids[..2])?;
    let right = source.extract_subgraph(&ids[2..])?;
    assert_ne!(left.store_id(), source.store_id());
    assert_ne!(right.store_id(), left.store_id());
    assert!(left.graph_store().get_node(ids[2]).is_none());
    assert!(left.graph_store().get_edge(edge).is_some());
    let subset = left.text_search("Doc", "body", "needle", 10)?;
    assert_eq!(subset.len(), 2);
    // Tokenizer excludes "ox": every document has length two. For A/B,
    // N=df=2 and TF normalization is one, so BM25 is ln(1 + 0.5/2.5).
    for (_, score) in &subset {
        assert!((score - 0.182_321_556_793_954_6).abs() < 1e-6);
    }
    let left_bytes = left.export_snapshot()?;
    let right_bytes = right.export_snapshot()?;
    let exact_error = GrafeoDB::open_multi([left_bytes.as_slice(), right_bytes.as_slice()])
        .err()
        .expect("different subset images must fail exact union");
    assert!(
        exact_error.to_string().contains("not byte-identical"),
        "{exact_error}"
    );
    for subset in [&left, &right] {
        subset.create_graph("empty")?;
        subset.create_graph_path(&GraphPath::from_components(&["empty", "nested"])?)?;
    }
    let left_bytes = left.export_snapshot()?;
    let right_bytes = right.export_snapshot()?;
    let union = GrafeoDB::open_multi_with(
        [left_bytes.as_slice(), right_bytes.as_slice()],
        OpenMultiOptions {
            index_policy: IndexMergePolicy::RebuildFromUnion,
            ..OpenMultiOptions::default()
        },
    )?;
    assert_ne!(union.store_id(), source.store_id());
    let mut hits = union.text_search("Doc", "body", "needle", 10)?;
    hits.sort_by_key(|(id, _)| *id);
    assert_eq!(hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(), ids[..2]);
    // Union N=3, df=2, average length=2: ln(1 + 1.5/2.5).
    for (_, score) in &hits {
        assert!(score.is_finite() && (score - 0.470_003_629_245_735_63).abs() < 1e-6);
    }
    assert!(union.text_search("Doc", "body", "ox", 10)?.is_empty());
    let restored_edge = union
        .graph_store()
        .get_edge(edge)
        .expect("transported edge");
    assert_eq!(
        (restored_edge.id, restored_edge.src, restored_edge.dst),
        (edge, ids[0], ids[2])
    );
    assert_eq!(restored_edge.edge_type.as_str(), "CROSSES_SUBSET");
    assert_eq!(union.node_count(), 3);
    assert_eq!(union.edge_count(), 1);
    assert!(union.list_graphs().contains(&"empty".to_owned()));
    // A reopened union remains an exact snapshot of its newly built corpus.
    let reopened = GrafeoDB::import_snapshot(&union.export_snapshot()?)?;
    let mut reopened_hits = reopened.text_search("Doc", "body", "needle", 10)?;
    reopened_hits.sort_by_key(|(id, _)| *id);
    assert_eq!(reopened_hits, hits);
    assert!(reopened.drop_index(owner)?);
    Ok(())
}

#[cfg(feature = "gql")]
fn property_index(graph: GraphPath, name: &str, property: &str) -> CreateIndexRequest {
    CreateIndexRequest {
        graph,
        name: Some(name.into()),
        label: None,
        property: property.into(),
        kind: IndexCreateKind::Property,
    }
}

#[test]
#[cfg(feature = "gql")]
fn extract_subgraph_keeps_named_catalog_owners_but_only_selected_root_topology() {
    let source = GrafeoDB::new_in_memory();
    let named = GraphPath::from_components(&["named"]).expect("named path");
    let nested = GraphPath::from_components(&["named", "child"]).expect("nested path");
    assert!(
        source
            .create_graph_path(&named)
            .expect("create named graph")
    );
    assert!(
        source
            .create_graph_path(&nested)
            .expect("create nested graph")
    );

    let root_owner = source
        .create_index(property_index(GraphPath::root(), "rootOwner", "root_key"))
        .expect("root index");
    let named_owner = source
        .create_index(property_index(named.clone(), "namedOwner", "named_key"))
        .expect("named index");
    let nested_owner = source
        .create_index(property_index(nested.clone(), "nestedOwner", "nested_key"))
        .expect("nested index");

    let root = source.create_node(&["Root"]);
    source
        .set_node_property(root, "root_key", Value::from("selected"))
        .unwrap();
    let session = source.session();
    session.use_graph_path(&named).expect("select named graph");
    let named_node = session.create_node(&["Named"]);
    session
        .set_node_property(named_node, "named_key", Value::String("outside".into()))
        .expect("named property");
    session
        .use_graph_path(&nested)
        .expect("select nested graph");
    let nested_node = session.create_node(&["Nested"]);
    session
        .set_node_property(nested_node, "nested_key", Value::String("outside".into()))
        .expect("nested property");

    let target = source.extract_subgraph(&[root]).expect("extract root node");
    assert_eq!(target.node_count(), 1);
    assert!(
        target.get_node(root).is_some(),
        "selected root node is retained"
    );
    assert_eq!(
        root, named_node,
        "fixture deliberately overlaps graph-local IDs"
    );
    assert_eq!(root, nested_node, "nested ID also overlaps selected root");
    assert!(target.list_graphs().iter().any(|name| name == "named"));

    let target_session = target.session();
    for path in [&named, &nested] {
        target_session
            .use_graph_path(path)
            .expect("select retained empty graph");
        let rows = target_session
            .execute("MATCH (n) RETURN n")
            .expect("query empty extracted graph");
        assert!(rows.rows().is_empty(), "unselected graph topology leaked");
    }

    assert!(target.drop_index(root_owner).expect("drop root owner"));
    assert!(target.drop_index(named_owner).expect("drop named owner"));
    assert!(target.drop_index(nested_owner).expect("drop nested owner"));
}
