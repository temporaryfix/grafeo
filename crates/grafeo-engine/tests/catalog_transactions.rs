//! Catalog metadata shares the Session transaction, visibility and rollback cut.

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn schema_and_data_are_private_until_commit() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let writer = db.session();
    let reader = db.session();
    writer.execute("START TRANSACTION")?;
    writer.execute("CREATE SCHEMA analytics")?;
    assert!(reader.execute("SESSION SET SCHEMA analytics").is_err());
    assert!(!db.list_graphs().contains(&"analytics/__default__".into()));
    writer.execute("SESSION SET SCHEMA analytics")?;
    writer.execute("INSERT (:Item {value: 42})")?;
    assert_eq!(
        writer.execute("MATCH (n:Item) RETURN n.value")?.rows(),
        &[vec![Value::Int64(42)]]
    );
    writer.execute("COMMIT")?;
    reader.execute("SESSION SET SCHEMA analytics")?;
    assert_eq!(
        reader.execute("MATCH (n:Item) RETURN n.value")?.rows(),
        &[vec![Value::Int64(42)]]
    );
    Ok(())
}

#[test]
fn procedure_and_body_write_rollback_together() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let writer = db.session();
    writer.execute("START TRANSACTION")?;
    writer.execute("CREATE PROCEDURE plant() RETURNS (value INTEGER) AS { INSERT (n:Item {value: 7}) RETURN n.value AS value }")?;
    assert!(db.session().execute("CALL plant()").is_err());
    assert_eq!(
        writer.execute("CALL plant()")?.rows(),
        &[vec![Value::Int64(7)]]
    );
    writer.execute("ROLLBACK")?;
    assert!(writer.execute("CALL plant()").is_err());
    assert_eq!(db.node_count(), 0);
    Ok(())
}

#[test]
fn catalog_savepoint_discards_only_later_metadata() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let writer = db.session();
    writer.execute("START TRANSACTION")?;
    writer.execute("CREATE SCHEMA retained")?;
    writer.savepoint("cut")?;
    writer.execute("CREATE SCHEMA discarded")?;
    writer
        .execute("CREATE PROCEDURE transient() RETURNS (value INTEGER) AS { RETURN 1 AS value }")?;
    writer.rollback_to_savepoint("cut")?;
    assert!(writer.execute("SESSION SET SCHEMA discarded").is_err());
    assert!(writer.execute("CALL transient()").is_err());
    writer.execute("COMMIT")?;
    db.session().execute("SESSION SET SCHEMA retained")?;
    assert!(
        db.session()
            .execute("SESSION SET SCHEMA discarded")
            .is_err()
    );
    Ok(())
}

#[test]
fn concurrent_catalog_writers_do_not_overwrite_each_other() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let first = db.session();
    let second = db.session();
    first.execute("START TRANSACTION")?;
    second.execute("START TRANSACTION")?;
    first.execute("CREATE SCHEMA first")?;
    second.execute("CREATE SCHEMA second")?;
    first.execute("COMMIT")?;
    assert!(second.execute("COMMIT").is_err());
    assert!(!second.in_transaction());
    db.session().execute("SESSION SET SCHEMA first")?;
    assert!(db.session().execute("SESSION SET SCHEMA second").is_err());
    Ok(())
}

#[test]
fn catalog_cut_keeps_procedure_body_after_concurrent_replacement() -> TestResult {
    for isolation in [
        grafeo_engine::transaction::IsolationLevel::SnapshotIsolation,
        grafeo_engine::transaction::IsolationLevel::ReadCommitted,
    ] {
        let db = GrafeoDB::new_in_memory();
        let mut reader = db.session();
        let writer = db.session();
        writer.execute(
            "CREATE PROCEDURE answer() RETURNS (value INTEGER) AS { RETURN 1 AS value }",
        )?;
        reader.begin_transaction_with_isolation(isolation)?;
        writer.execute(
            "CREATE OR REPLACE PROCEDURE answer() RETURNS (value INTEGER) AS { RETURN 2 AS value }",
        )?;
        assert_eq!(
            reader.execute("CALL answer()")?.rows(),
            &[vec![Value::Int64(1)]]
        );
        reader.execute("COMMIT")?;
        assert_eq!(
            reader.execute("CALL answer()")?.rows(),
            &[vec![Value::Int64(2)]]
        );
    }
    Ok(())
}

#[test]
fn schema_types_and_index_share_one_transaction() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let writer = db.session();
    writer.execute("START TRANSACTION")?;
    writer.execute("CREATE SCHEMA typed")?;
    writer.execute("SESSION SET SCHEMA typed")?;
    writer.execute("CREATE NODE TYPE Item (value INTEGER NOT NULL)")?;
    writer.execute("CREATE GRAPH TYPE Items (NODE TYPE Item)")?;
    writer.execute("CREATE GRAPH items TYPED Items")?;
    writer.execute("SESSION SET GRAPH items")?;
    writer.execute("INSERT (:Item {value: 1})")?;
    assert!(writer.execute("INSERT (:Item {value: 'bad'})").is_err());
    writer.execute("CREATE INDEX item_value FOR (n:Item) ON (n.value)")?;
    writer.execute("COMMIT")?;
    let reader = db.session();
    reader.execute("SESSION SET SCHEMA typed")?;
    reader.execute("SESSION SET GRAPH items")?;
    assert_eq!(
        reader.execute("MATCH (n:Item) RETURN n.value")?.rows(),
        &[vec![Value::Int64(1)]]
    );
    assert_eq!(reader.execute("SHOW INDEXES")?.rows().len(), 1);
    Ok(())
}

#[test]
fn new_constraint_checks_own_prior_writes() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let writer = db.session();
    writer.execute("START TRANSACTION")?;
    writer.execute("INSERT (:Item {value: 1}), (:Item {value: 1})")?;
    assert!(
        writer
            .execute("CREATE CONSTRAINT unique_value FOR (n:Item) ON (n.value) UNIQUE")
            .is_err()
    );
    writer.execute("COMMIT")?;
    assert_eq!(db.node_count(), 2);
    assert!(db.session().execute("SHOW CONSTRAINTS")?.rows().is_empty());
    Ok(())
}

#[test]
fn drop_schema_rejects_own_uncommitted_data() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let writer = db.session();
    writer.execute("CREATE SCHEMA occupied")?;
    writer.execute("SESSION SET SCHEMA occupied")?;
    writer.execute("START TRANSACTION")?;
    writer.execute("INSERT (:Item {value: 1})")?;
    assert!(writer.execute("DROP SCHEMA occupied").is_err());
    writer.execute("COMMIT")?;
    assert_eq!(
        writer.execute("MATCH (n:Item) RETURN n.value")?.rows(),
        &[vec![Value::Int64(1)]]
    );
    Ok(())
}

#[test]
fn constraint_commit_rechecks_concurrent_data() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let ddl = db.session();
    ddl.execute("INSERT (:Item {value: 1})")?;
    ddl.execute("START TRANSACTION")?;
    ddl.execute("CREATE CONSTRAINT unique_value FOR (n:Item) ON (n.value) UNIQUE")?;
    db.session().execute("INSERT (:Item {value: 1})")?;
    assert!(ddl.execute("COMMIT").is_err());
    assert!(db.session().execute("SHOW CONSTRAINTS")?.rows().is_empty());
    assert_eq!(db.node_count(), 2);
    Ok(())
}

#[test]
fn schema_drop_commit_rechecks_concurrent_data() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let ddl = db.session();
    ddl.execute("CREATE SCHEMA occupied")?;
    let writer = db.session();
    writer.execute("SESSION SET SCHEMA occupied")?;
    ddl.execute("START TRANSACTION")?;
    ddl.execute("DROP SCHEMA occupied")?;
    writer.execute("INSERT (:Item {value: 1})")?;
    assert!(ddl.execute("COMMIT").is_err());
    assert_eq!(
        writer.execute("MATCH (n:Item) RETURN n.value")?.rows(),
        &[vec![Value::Int64(1)]]
    );
    Ok(())
}

#[test]
fn schema_drop_commit_rechecks_concurrent_empty_child() -> TestResult {
    use grafeo_common::types::GraphPath;
    let db = GrafeoDB::new_in_memory();
    let ddl = db.session();
    ddl.execute("CREATE SCHEMA occupied")?;
    ddl.execute("START TRANSACTION")?;
    ddl.execute("DROP SCHEMA occupied")?;
    let writer = db.session();
    let default = GraphPath::root().child("occupied/__default__")?;
    assert!(writer.create_graph_path(&default.child("child")?)?);
    assert!(ddl.execute("COMMIT").is_err());
    writer.use_graph_path(&default.child("child")?)?;
    assert_eq!(writer.execute("MATCH (n) RETURN n")?.rows().len(), 0);
    Ok(())
}

#[test]
fn schema_create_commit_rechecks_concurrent_namespace_claim() -> TestResult {
    use grafeo_common::types::GraphPath;
    let db = GrafeoDB::new_in_memory();
    let ddl = db.session();
    ddl.execute("START TRANSACTION")?;
    ddl.execute("CREATE SCHEMA claimed")?;
    let path = GraphPath::from_components(&["claimed/other"])?;
    assert!(db.session().create_graph_path(&path)?);
    assert!(ddl.execute("COMMIT").is_err());
    assert!(db.list_graphs().contains(&"claimed/other".into()));
    assert!(db.session().execute("SESSION SET SCHEMA claimed").is_err());
    Ok(())
}

#[test]
fn snapshot_copy_uses_its_type_cut_but_cannot_commit_after_concurrent_drop() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let setup = db.session();
    setup.execute("CREATE NODE TYPE Item (value INTEGER)")?;
    setup.execute("CREATE GRAPH TYPE Items (NODE TYPE Item)")?;
    let writer = db.session();
    writer.execute("START TRANSACTION")?;
    writer.execute("CREATE GRAPH source TYPED Items")?;
    setup.execute("DROP GRAPH TYPE Items")?;
    writer.execute("CREATE GRAPH copied AS COPY OF source")?;
    assert!(writer.execute("COMMIT").is_err());
    assert!(!writer.in_transaction());
    assert!(db.list_graphs().is_empty());
    Ok(())
}

#[test]
fn drop_index_graph_type_and_schema_observes_own_deletions() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let writer = db.session();
    writer.execute("CREATE SCHEMA removable")?;
    writer.execute("SESSION SET SCHEMA removable")?;
    writer.execute("CREATE NODE TYPE Item (value INTEGER)")?;
    writer.execute("CREATE GRAPH TYPE Items (NODE TYPE Item)")?;
    writer.execute("CREATE GRAPH data TYPED Items")?;
    writer.execute("SESSION SET GRAPH data")?;
    writer.execute("CREATE INDEX item_value FOR (n:Item) ON (n.value)")?;
    writer.execute("START TRANSACTION")?;
    writer.execute("DROP INDEX item_value")?;
    writer.execute("DROP GRAPH data")?;
    writer.execute("DROP GRAPH TYPE Items")?;
    writer.execute("DROP NODE TYPE Item")?;
    writer.execute("DROP SCHEMA removable")?;
    writer.execute("COMMIT")?;
    assert!(
        db.session()
            .execute("SESSION SET SCHEMA removable")
            .is_err()
    );
    assert!(db.list_graphs().is_empty());
    Ok(())
}
