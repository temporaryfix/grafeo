//! Integration tests for `GrafeoDB::extract_subgraph` — produces a new
//! in-memory database containing exactly the requested nodes plus
//! every edge whose endpoints are both in the request set, with
//! source-allocated NodeIds and EdgeIds preserved.

use grafeo_common::types::Value;
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
