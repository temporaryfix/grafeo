//! Tests for schema isolation of types (node, edge, graph types).
//!
//! Verifies that SHOW commands and CREATE/DROP/ALTER type commands
//! respect the current schema set via `SESSION SET SCHEMA`.
//!
//! Fixes: <https://github.com/GrafeoDB/grafeo/issues/167>
//!
//! ```bash
//! cargo test -p grafeo-engine --test schema_type_isolation
//! ```

#![cfg(feature = "lpg")]

use grafeo_engine::{GrafeoDB, transaction::IsolationLevel};

// ---------------------------------------------------------------------------
// SHOW GRAPH TYPES (primary bug from issue #167)
// ---------------------------------------------------------------------------

#[test]
fn show_graph_types_respects_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Reproduce issue #167 exactly
    session
        .execute("CREATE SCHEMA IF NOT EXISTS my_schema")
        .unwrap();
    session
        .execute(
            "CREATE GRAPH TYPE IF NOT EXISTS social_network (
                NODE TYPE Person (name STRING NOT NULL, age INTEGER),
                EDGE TYPE KNOWS (since INTEGER)
            )",
        )
        .unwrap();

    // Default schema sees the graph type
    let result = session.execute("SHOW GRAPH TYPES").unwrap();
    assert_eq!(
        result.rows().len(),
        1,
        "default schema should see 1 graph type"
    );

    // Switch to a different schema
    session
        .execute("CREATE SCHEMA IF NOT EXISTS my_schema2")
        .unwrap();
    session.execute("SESSION SET SCHEMA my_schema2").unwrap();

    // my_schema2 should see no graph types
    let result = session.execute("SHOW GRAPH TYPES").unwrap();
    assert_eq!(
        result.rows().len(),
        0,
        "my_schema2 should see 0 graph types (issue #167)"
    );
}

// ---------------------------------------------------------------------------
// SHOW NODE TYPES
// ---------------------------------------------------------------------------

#[test]
fn show_node_types_respects_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE SCHEMA IF NOT EXISTS s1").unwrap();
    session.execute("SESSION SET SCHEMA s1").unwrap();
    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();

    // Visible in s1
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Not visible in default schema
    session.execute("SESSION RESET SCHEMA").unwrap();
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(
        result.rows().len(),
        0,
        "default schema should not see s1 types"
    );
}

// ---------------------------------------------------------------------------
// SHOW EDGE TYPES
// ---------------------------------------------------------------------------

#[test]
fn show_edge_types_respects_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE SCHEMA IF NOT EXISTS s1").unwrap();
    session.execute("SESSION SET SCHEMA s1").unwrap();
    session
        .execute("CREATE EDGE TYPE KNOWS (since INTEGER)")
        .unwrap();

    let result = session.execute("SHOW EDGE TYPES").unwrap();
    assert_eq!(result.rows().len(), 1);

    session.execute("SESSION RESET SCHEMA").unwrap();
    let result = session.execute("SHOW EDGE TYPES").unwrap();
    assert_eq!(
        result.rows().len(),
        0,
        "default schema should not see s1 edge types"
    );
}

// ---------------------------------------------------------------------------
// Type isolation between schemas
// ---------------------------------------------------------------------------

#[test]
fn types_isolated_between_schemas() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE SCHEMA IF NOT EXISTS alpha")
        .unwrap();
    session.execute("CREATE SCHEMA IF NOT EXISTS beta").unwrap();

    // Create same-named type in both schemas
    session.execute("SESSION SET SCHEMA alpha").unwrap();
    session
        .execute("CREATE NODE TYPE Item (color STRING)")
        .unwrap();

    session.execute("SESSION SET SCHEMA beta").unwrap();
    session
        .execute("CREATE NODE TYPE Item (weight FLOAT64)")
        .unwrap();

    // Each schema sees exactly one
    session.execute("SESSION SET SCHEMA alpha").unwrap();
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 1);

    session.execute("SESSION SET SCHEMA beta").unwrap();
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Default schema sees none
    session.execute("SESSION RESET SCHEMA").unwrap();
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 0);
}

// ---------------------------------------------------------------------------
// Default schema types hidden in named schema
// ---------------------------------------------------------------------------

#[test]
fn default_schema_types_hidden_in_named_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create type in default schema
    session
        .execute("CREATE NODE TYPE GlobalType (value STRING)")
        .unwrap();

    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Switch to named schema: default types not visible
    session
        .execute("CREATE SCHEMA IF NOT EXISTS isolated")
        .unwrap();
    session.execute("SESSION SET SCHEMA isolated").unwrap();
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(
        result.rows().len(),
        0,
        "named schema should not see default types"
    );
}

// ---------------------------------------------------------------------------
// DROP type respects schema
// ---------------------------------------------------------------------------

#[test]
fn drop_type_respects_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE SCHEMA IF NOT EXISTS s1").unwrap();
    session.execute("SESSION SET SCHEMA s1").unwrap();
    session
        .execute("CREATE NODE TYPE Temp (val STRING)")
        .unwrap();

    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 1);

    session.execute("DROP NODE TYPE Temp").unwrap();
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 0);
}

// ---------------------------------------------------------------------------
// DROP SCHEMA blocks when types exist
// ---------------------------------------------------------------------------

#[test]
fn drop_schema_blocks_when_types_exist() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE SCHEMA IF NOT EXISTS blocker")
        .unwrap();
    session.execute("SESSION SET SCHEMA blocker").unwrap();
    session
        .execute("CREATE NODE TYPE Pinned (val STRING)")
        .unwrap();

    // Dropping should fail because types exist
    session.execute("SESSION RESET SCHEMA").unwrap();
    let result = session.execute("DROP SCHEMA blocker");
    assert!(result.is_err(), "DROP SCHEMA should fail when types exist");
}

#[test]
fn stale_schema_context_cannot_create_catalog_objects() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    let stale = db.session();

    admin.execute("CREATE SCHEMA vanished").unwrap();
    stale.execute("SESSION SET SCHEMA vanished").unwrap();
    admin.execute("DROP SCHEMA vanished").unwrap();

    let error = stale
        .execute("CREATE NODE TYPE Orphan (value STRING)")
        .expect_err("DDL through a stale schema context must fail");
    assert!(
        error.to_string().contains("vanished") && error.to_string().contains("does not exist"),
        "expected a structured missing-schema error, got: {error}"
    );

    admin.execute("CREATE SCHEMA vanished").unwrap();
    let fresh = db.session();
    fresh.execute("SESSION SET SCHEMA vanished").unwrap();
    assert_eq!(
        fresh.execute("SHOW NODE TYPES").unwrap().row_count(),
        0,
        "a later schema incarnation must not inherit orphan catalog state"
    );
}

#[test]
fn every_schema_relative_catalog_creator_rejects_a_stale_context() {
    for command in [
        "CREATE EDGE TYPE OrphanEdge (value STRING)",
        "CREATE GRAPH TYPE OrphanGraph (NODE TYPE Ghost (value STRING))",
        "CREATE CONSTRAINT orphan_unique FOR (n:Ghost) ON (n.value) UNIQUE",
    ] {
        let db = GrafeoDB::new_in_memory();
        let admin = db.session();
        let stale = db.session();
        admin.execute("CREATE SCHEMA vanished").unwrap();
        stale.execute("SESSION SET SCHEMA vanished").unwrap();
        admin.execute("DROP SCHEMA vanished").unwrap();

        let error = stale
            .execute(command)
            .expect_err("all schema-relative catalog creators must reject stale ownership");
        assert!(
            error.to_string().contains("vanished") && error.to_string().contains("does not exist"),
            "command {command:?} returned the wrong error: {error}"
        );
    }
}

#[test]
fn transactional_create_graph_cannot_outlive_owning_schema() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin.execute("CREATE SCHEMA transient").unwrap();

    let mut creator = db.session();
    creator.execute("SESSION SET SCHEMA transient").unwrap();
    creator.begin_transaction().unwrap();
    creator.execute("CREATE GRAPH orphan").unwrap();

    admin.execute("DROP SCHEMA transient").unwrap();
    let error = creator
        .commit()
        .expect_err("graph publication must revalidate its owning schema");
    assert!(
        error.to_string().contains("transient"),
        "expected an owning-schema conflict, got: {error}"
    );

    admin.execute("CREATE SCHEMA transient").unwrap();
    let fresh = db.session();
    fresh.execute("SESSION SET SCHEMA transient").unwrap();
    assert_eq!(
        fresh.execute("SHOW GRAPHS").unwrap().row_count(),
        0,
        "the replacement schema incarnation must not inherit the aborted graph"
    );
}

#[test]
fn graph_schema_provenance_does_not_depend_on_the_active_graph_touch() {
    for level in [
        IsolationLevel::SnapshotIsolation,
        IsolationLevel::Serializable,
    ] {
        let db = GrafeoDB::new_in_memory();
        let admin = db.session();
        admin.execute("CREATE SCHEMA transient").unwrap();

        let mut creator = db.session();
        creator.execute("SESSION SET SCHEMA transient").unwrap();
        // The native root selection is independent of the current schema:
        // BEGIN must not incidentally pin transient/__default__.
        creator
            .use_graph_path(&grafeo_common::types::GraphPath::root())
            .unwrap();
        creator.begin_transaction_with_isolation(level).unwrap();
        creator.execute("CREATE GRAPH orphan").unwrap();

        admin.execute("DROP SCHEMA transient").unwrap();
        assert_eq!(creator.current_schema().as_deref(), Some("transient"));
        let error = creator
            .commit()
            .expect_err("the exact parent schema incarnation changed");
        assert!(
            matches!(
                error,
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::WriteConflict(_)
                )
            ),
            "expected a lifecycle write conflict under {level:?}, got: {error}"
        );
        assert!(
            !db.list_graphs()
                .iter()
                .any(|graph| graph == "transient/orphan"),
            "an aborted graph must not escape under {level:?}"
        );
    }
}

#[test]
fn graph_schema_provenance_rejects_drop_recreate_aba() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin.execute("CREATE SCHEMA changing").unwrap();
    let original_default = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("changing/__default__")
        .expect("CREATE SCHEMA installs its default partition");

    let mut creator = db.session();
    creator.execute("SESSION SET SCHEMA changing").unwrap();
    // Avoid the older incidental touched-graph check: this regression must
    // depend on the owner token carried by the detached CREATE itself.
    creator
        .use_graph_path(&grafeo_common::types::GraphPath::root())
        .unwrap();
    creator.begin_transaction().unwrap();
    creator.execute("CREATE GRAPH orphan").unwrap();

    admin.execute("DROP SCHEMA changing").unwrap();
    admin.execute("CREATE SCHEMA changing").unwrap();
    let replacement_default = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("changing/__default__")
        .expect("the replacement schema has a default partition");
    assert!(
        !std::sync::Arc::ptr_eq(&original_default, &replacement_default),
        "DROP+CREATE must produce a distinct schema incarnation"
    );

    let error = creator
        .commit()
        .expect_err("a name-equal replacement schema must not satisfy the old owner token");
    assert!(
        matches!(
            error,
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::WriteConflict(_)
            )
        ),
        "expected an incarnation write conflict, got: {error}"
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("changing/__default__")
            .as_ref()
            .is_some_and(|live| std::sync::Arc::ptr_eq(live, &replacement_default)),
        "the rejected transaction must preserve the replacement default partition"
    );
    assert!(
        !db.list_graphs()
            .iter()
            .any(|graph| graph == "changing/orphan"),
        "the graph staged under the old schema incarnation must not publish"
    );
}

#[test]
fn savepoint_restores_exact_graph_owner_token_across_drop_recreate() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin.execute("CREATE SCHEMA versioned").unwrap();

    let mut creator = db.session();
    creator.execute("SESSION SET SCHEMA versioned").unwrap();
    creator
        .use_graph_path(&grafeo_common::types::GraphPath::root())
        .unwrap();
    creator.begin_transaction().unwrap();
    creator.execute("CREATE GRAPH kept").unwrap();
    creator.execute("SAVEPOINT stable_owner").unwrap();
    creator.execute("CREATE GRAPH discarded").unwrap();
    creator
        .execute("ROLLBACK TO SAVEPOINT stable_owner")
        .unwrap();

    admin.execute("DROP SCHEMA versioned").unwrap();
    admin.execute("CREATE SCHEMA versioned").unwrap();
    let replacement_default = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("versioned/__default__")
        .expect("the replacement schema has a default partition");

    let error = creator
        .commit()
        .expect_err("savepoint restore must retain the original schema incarnation token");
    assert!(
        matches!(
            error,
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::WriteConflict(_)
            )
        ),
        "expected a restored-owner write conflict, got: {error}"
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("versioned/__default__")
            .as_ref()
            .is_some_and(|live| std::sync::Arc::ptr_eq(live, &replacement_default)),
        "the rejected transaction must preserve the replacement default partition"
    );
    assert!(
        !db.list_graphs()
            .iter()
            .any(|graph| matches!(graph.as_str(), "versioned/kept" | "versioned/discarded")),
        "neither the retained nor rolled-back detached graph may publish"
    );
}

#[test]
fn transactional_create_graph_requires_registered_owning_schema() {
    let db = GrafeoDB::new_in_memory();
    let mut creator = db.session();
    creator.set_schema("never_registered").unwrap();
    creator.begin_transaction().unwrap();
    let error = creator
        .execute("CREATE GRAPH orphan")
        .expect_err("a schema-qualified graph needs a registered owner at statement time");
    assert!(
        error.to_string().contains("never_registered"),
        "expected an owning-schema conflict, got: {error}"
    );
    creator.rollback().unwrap();

    db.session()
        .execute("CREATE SCHEMA never_registered")
        .unwrap();
    let fresh = db.session();
    fresh
        .execute("SESSION SET SCHEMA never_registered")
        .unwrap();
    assert_eq!(fresh.execute("SHOW GRAPHS").unwrap().row_count(), 0);
}

#[test]
fn drop_schema_rejects_an_owned_index() {
    let db = GrafeoDB::new_in_memory();
    let owner = db.session();
    owner.execute("CREATE SCHEMA indexed").unwrap();
    owner.execute("SESSION SET SCHEMA indexed").unwrap();
    owner
        .execute("CREATE INDEX idx_owned FOR (n:Person) ON (n.name)")
        .unwrap();

    let dropper = db.session();
    let error = dropper
        .execute("DROP SCHEMA indexed")
        .expect_err("an index is a schema-owned object");
    assert!(
        error.to_string().contains("not empty"),
        "expected a non-empty-schema error, got: {error}"
    );

    owner.execute("DROP INDEX idx_owned").unwrap();
    dropper.execute("DROP SCHEMA indexed").unwrap();
}

#[test]
fn drop_graph_atomically_cascades_logical_and_physical_indexes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE GRAPH indexed_graph").unwrap();
    session.execute("SESSION SET GRAPH indexed_graph").unwrap();
    session
        .execute("CREATE INDEX idx_graph FOR (n:Person) ON (n.name)")
        .unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();

    session
        .execute("DROP GRAPH indexed_graph")
        .expect("DROP GRAPH must atomically remove its logical index receipts");
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("indexed_graph")
            .is_none()
    );
    let indexes = session.execute("SHOW INDEXES").unwrap();
    assert!(
        indexes
            .rows()
            .iter()
            .all(|row| row[0].as_str() != Some("idx_graph")),
        "the dropped graph must not leave a logical index receipt"
    );

    assert!(db.create_graph("physical_graph").unwrap());
    db.set_current_graph(Some("physical_graph")).unwrap();
    db.create_index(grafeo_engine::CreateIndexRequest {
        graph: grafeo_common::types::GraphPath::from_components(&["physical_graph"])
            .expect("exact named graph path"),
        name: None,
        label: None,
        property: "key".into(),
        kind: grafeo_engine::IndexCreateKind::Property,
    })
    .expect("create property index");
    db.set_current_graph(None).unwrap();
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("physical_graph")
            .expect("named graph remains present")
            .has_property_index("key")
    );
    assert!(!grafeo_engine::database::testing::root_lpg_store(&db).has_property_index("key"));
    assert!(
        db.drop_graph("physical_graph").expect("drop graph"),
        "DROP GRAPH must cascade an unnamed physical index"
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("physical_graph")
            .is_none()
    );
}

#[test]
fn drop_recreate_drop_preserves_the_original_incarnation_index_cascade() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute("CREATE GRAPH replaceable").unwrap();
    session.execute("SESSION SET GRAPH replaceable").unwrap();
    session
        .execute("CREATE INDEX idx_original FOR (n:Person) ON (n.name)")
        .unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();

    session.begin_transaction().unwrap();
    session.execute("DROP GRAPH replaceable").unwrap();
    session.execute("CREATE GRAPH replaceable").unwrap();
    session.execute("SESSION SET GRAPH replaceable").unwrap();
    session
        .execute("CREATE INDEX idx_replacement FOR (n:Person) ON (n.email)")
        .unwrap();
    session.execute("SESSION RESET GRAPH").unwrap();
    session.execute("DROP GRAPH replaceable").unwrap();
    session
        .commit()
        .expect("cancelling the replacement must retain the original graph's index cascade");

    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("replaceable")
            .is_none()
    );
    let indexes = session.execute("SHOW INDEXES").unwrap();
    assert!(
        indexes.rows().is_empty(),
        "neither graph incarnation may leave a logical index receipt"
    );
}

#[test]
fn staged_index_cannot_publish_after_schema_drop() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin.execute("CREATE SCHEMA fleeting").unwrap();

    let mut indexer = db.session();
    indexer.execute("SESSION SET SCHEMA fleeting").unwrap();
    indexer.begin_transaction().unwrap();
    indexer
        .execute("CREATE INDEX idx_orphan FOR (n:Person) ON (n.name)")
        .unwrap();

    admin.execute("DROP SCHEMA fleeting").unwrap();
    let error = indexer
        .commit()
        .expect_err("index publication must revalidate its target graph incarnation");
    assert!(
        error.to_string().contains("fleeting"),
        "expected a target lifecycle conflict, got: {error}"
    );

    admin.execute("CREATE SCHEMA fleeting").unwrap();
    let fresh = db.session();
    fresh.execute("SESSION SET SCHEMA fleeting").unwrap();
    assert!(
        fresh
            .execute("SHOW INDEXES")
            .unwrap()
            .rows()
            .iter()
            .all(|row| row[0].as_str() != Some("idx_orphan")),
        "the replacement schema incarnation must not inherit the aborted index"
    );
}

#[test]
fn staged_index_requires_registered_target_schema() {
    let db = GrafeoDB::new_in_memory();
    let mut indexer = db.session();
    indexer.set_schema("never_indexed").unwrap();
    indexer.begin_transaction().unwrap();
    let error = indexer
        .execute("CREATE INDEX idx_orphan_direct FOR (n:Person) ON (n.name)")
        .expect_err("an index target must have a registered owner at statement time");
    assert!(
        error.to_string().contains("never_indexed"),
        "expected a target-schema conflict, got: {error}"
    );
    indexer.rollback().unwrap();

    db.session().execute("CREATE SCHEMA never_indexed").unwrap();
    assert!(
        db.session()
            .execute("SHOW INDEXES")
            .unwrap()
            .rows()
            .iter()
            .all(|row| row[0].as_str() != Some("idx_orphan_direct"))
    );
}

#[test]
fn drop_schema_rejects_default_graph_data() {
    let db = GrafeoDB::new_in_memory();
    let owner = db.session();
    owner.execute("CREATE SCHEMA inhabited").unwrap();
    owner.execute("SESSION SET SCHEMA inhabited").unwrap();
    owner.execute("INSERT (:Resident {id: 1})").unwrap();

    let error = db
        .session()
        .execute("DROP SCHEMA inhabited")
        .expect_err("DROP SCHEMA must never discard its default graph's data");
    assert!(
        error.to_string().contains("not empty"),
        "expected a non-empty-schema error, got: {error}"
    );
    assert_eq!(owner.execute("MATCH (n) RETURN n").unwrap().row_count(), 1);
}

#[test]
fn schema_names_are_case_insensitively_unique_and_drop_is_canonical() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA MixedCase").unwrap();

    session
        .execute("CREATE SCHEMA mixedcase")
        .expect_err("case aliases must not create ambiguous namespaces");
    session
        .execute("DROP SCHEMA mIxEdCaSe")
        .expect("DROP must resolve the registered canonical spelling");
    assert_eq!(session.execute("SHOW SCHEMAS").unwrap().row_count(), 0);
}

#[test]
fn parser_free_graphs_preserve_schema_namespace_ownership() {
    let db = GrafeoDB::new_in_memory();
    let admin = db.session();
    admin.execute("CREATE SCHEMA Canonical").unwrap();

    assert!(db.create_graph("canonical/direct").unwrap());
    assert!(
        db.list_graphs()
            .iter()
            .any(|graph| graph == "Canonical/direct")
    );
    admin
        .execute("DROP SCHEMA Canonical")
        .expect_err("a parser-free graph is still owned by its schema");

    assert!(db.drop_graph("cAnOnIcAl/direct").expect("drop graph"));
    admin.execute("DROP SCHEMA Canonical").unwrap();
}

#[test]
fn parser_free_uri_graphs_and_schema_names_cannot_collide() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.create_graph("future/root").unwrap());

    db.session()
        .execute("CREATE SCHEMA future")
        .expect_err("a schema cannot capture an existing root graph namespace");
    assert!(db.list_graphs().iter().any(|graph| graph == "future/root"));
}

#[test]
fn root_constraints_scan_parser_free_slash_graphs() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.create_graph("urn:future/root").unwrap());
    let writer = db.session();
    writer
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["urn:future/root"]).unwrap(),
        )
        .unwrap();
    writer.execute("INSERT (:Item {key: 1})").unwrap();
    writer.execute("INSERT (:Item {key: 1})").unwrap();

    db.session()
        .execute("CREATE CONSTRAINT unique_item FOR (n:Item) ON (n.key) UNIQUE")
        .expect_err("root constraints must scan every root-owned slash graph");
}

#[test]
fn schema_default_partition_cannot_be_dropped_as_a_named_graph() {
    let db = GrafeoDB::new_in_memory();
    db.session().execute("CREATE SCHEMA guarded").unwrap();

    assert!(db.drop_graph("guarded/__default__").is_err());
    assert!(
        !db.create_graph("guarded/__DEFAULT__").unwrap(),
        "case aliases must resolve to the existing reserved partition"
    );
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .graph("guarded/__default__")
            .is_some(),
        "the schema-incarnation token must remain installed"
    );

    let session = db.session();
    session.execute("SESSION SET SCHEMA guarded").unwrap();
    session
        .execute("DROP GRAPH __default__")
        .expect_err("GQL lifecycle DDL must protect the same partition");
    session
        .execute("CREATE GRAPH __DEFAULT__")
        .expect_err("GQL creation must not allocate a case-variant partition");

    assert!(db.create_graph("guarded/path/__default__").unwrap());
    assert!(
        db.drop_graph("guarded/path/__default__")
            .expect("drop graph"),
        "only the exact owning partition is reserved"
    );
}

// ---------------------------------------------------------------------------
// ALTER type respects schema
// ---------------------------------------------------------------------------

#[test]
fn alter_type_respects_schema() -> Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::types::Value;

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE SCHEMA s2")?;
    session.execute("SESSION SET SCHEMA s2")?;
    session.execute("CREATE NODE TYPE Mutable (other INTEGER)")?;
    let other_types = session.execute("SHOW NODE TYPES")?;
    let other_schema = other_types.rows();
    assert_eq!(
        other_schema,
        vec![vec![
            Value::from("Mutable"),
            Value::from("other INT64"),
            Value::from(""),
            Value::from("")
        ]]
    );

    session.execute("CREATE SCHEMA IF NOT EXISTS s1").unwrap();
    session.execute("SESSION SET SCHEMA s1").unwrap();
    session
        .execute("CREATE NODE TYPE Mutable (name STRING)")
        .unwrap();
    session
        .execute("ALTER NODE TYPE Mutable ADD extra STRING")
        .unwrap();

    // Verify the type is still visible and was altered in s1
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(result.rows().len(), 1);
    let altered = vec![vec![
        Value::from("Mutable"),
        Value::from("name STRING, extra STRING"),
        Value::from(""),
        Value::from(""),
    ]];
    assert_eq!(result.rows(), altered);

    // Not visible in default schema (isolation preserved after alter)
    session.execute("SESSION RESET SCHEMA").unwrap();
    let result = session.execute("SHOW NODE TYPES").unwrap();
    assert_eq!(
        result.rows().len(),
        0,
        "altered schema type should not leak to default"
    );
    session.execute("SESSION SET SCHEMA s2")?;
    assert_eq!(session.execute("SHOW NODE TYPES")?.rows(), other_schema);
    session.execute("SESSION SET SCHEMA s1")?;
    assert_eq!(session.execute("SHOW NODE TYPES")?.rows(), altered);
    Ok(())
}

// ---------------------------------------------------------------------------
// CREATE GRAPH TYPED — schema-aware binding (regression from #167 fix)
// ---------------------------------------------------------------------------

#[test]
fn create_graph_typed_respects_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE SCHEMA IF NOT EXISTS s1").unwrap();
    session.execute("SESSION SET SCHEMA s1").unwrap();
    // Pre-declare KNOWS so the bare reference in the graph type body is valid.
    session
        .execute("CREATE EDGE TYPE KNOWS (since INTEGER)")
        .expect("pre-declare KNOWS edge type");
    session
        .execute(
            "CREATE GRAPH TYPE social_network (
                NODE TYPE Person (name STRING NOT NULL),
                EDGE TYPE KNOWS
            )",
        )
        .unwrap();

    // Unqualified TYPED should resolve against the current schema (s1)
    let result = session.execute("CREATE GRAPH IF NOT EXISTS my_social TYPED social_network");
    assert!(
        result.is_ok(),
        "CREATE GRAPH TYPED should succeed when type is in current schema: {result:?}"
    );
}

#[test]
fn create_graph_typed_wrong_schema_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create type in s1
    session.execute("CREATE SCHEMA IF NOT EXISTS s1").unwrap();
    session.execute("SESSION SET SCHEMA s1").unwrap();
    session
        .execute("CREATE GRAPH TYPE org_type (NODE TYPE Dept (name STRING))")
        .unwrap();

    // Switch to s2 — unqualified name should NOT resolve to s1's type
    session.execute("CREATE SCHEMA IF NOT EXISTS s2").unwrap();
    session.execute("SESSION SET SCHEMA s2").unwrap();

    let result = session.execute("CREATE GRAPH IF NOT EXISTS g TYPED org_type");
    assert!(
        result.is_err(),
        "CREATE GRAPH TYPED with unqualified name from wrong schema must fail"
    );
}

// ---------------------------------------------------------------------------
// CREATE GRAPH TYPED — cross-schema qualified references (schema.type syntax)
// ---------------------------------------------------------------------------

#[test]
fn cross_schema_typed_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Define type in s1
    session.execute("CREATE SCHEMA IF NOT EXISTS s1").unwrap();
    session.execute("SESSION SET SCHEMA s1").unwrap();
    session
        .execute(
            "CREATE GRAPH TYPE social_network (
                NODE TYPE Person (name STRING NOT NULL),
                EDGE TYPE KNOWS (since INTEGER)
            )",
        )
        .unwrap();

    // Switch to s2 and reference s1's type with qualified syntax
    session.execute("CREATE SCHEMA IF NOT EXISTS s2").unwrap();
    session.execute("SESSION SET SCHEMA s2").unwrap();

    let result = session.execute("CREATE GRAPH IF NOT EXISTS my_social TYPED s1.social_network");
    assert!(
        result.is_ok(),
        "CREATE GRAPH TYPED with qualified schema.type should succeed: {result:?}"
    );
}

#[test]
fn qualified_type_not_found() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("CREATE GRAPH g TYPED nonexistent.some_type");
    assert!(
        result.is_err(),
        "Qualified reference to nonexistent schema/type must fail"
    );
}

#[test]
fn unqualified_type_no_schema_set() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // No schema set: works just like before schema isolation
    session
        .execute(
            "CREATE GRAPH TYPE flat_type (
                NODE TYPE Item (value INTEGER)
            )",
        )
        .unwrap();

    let result = session.execute("CREATE GRAPH g TYPED flat_type");
    assert!(
        result.is_ok(),
        "Unqualified TYPED with no session schema should resolve in default namespace: {result:?}"
    );
}
