//! Exact native graph coordinates must govern graph-type validation and lifecycle.

use super::Session;
use crate::GrafeoDB;
use crate::catalog::GraphTypeDefinition;
use grafeo_common::types::{GraphPath, Value};
use grafeo_common::utils::error::{Error, TransactionError};
use std::sync::Arc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn graph_paths() -> Result<[GraphPath; 4], Box<dyn std::error::Error>> {
    Ok([
        GraphPath::root(),
        GraphPath::from_components(&[""])?,
        GraphPath::from_components(&["a/b"])?,
        GraphPath::from_components(&["a", "b"])?,
    ])
}

fn database() -> Result<GrafeoDB, Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    db.transaction_manager
        .with_write_authority(|| -> TestResult {
            for name in ["", "a/b", "a"] {
                assert!(crate::database::testing::root_lpg_store(&db).create_graph(name)?);
            }
            assert!(
                crate::database::testing::root_lpg_store(&db)
                    .graph("a")
                    .ok_or("missing parent")?
                    .create_graph("b")?
            );
            Ok(())
        })?;
    for (name, open) in [("Open", true), ("Closed", false)] {
        db.catalog.register_graph_type(GraphTypeDefinition {
            name: name.to_string(),
            allowed_node_types: Vec::new(),
            allowed_edge_types: Vec::new(),
            open,
        })?;
    }
    for path in graph_paths()? {
        bind(&db, &path, "Open")?;
    }
    bind(&db, &GraphPath::from_components(&["a"])?, "Open")?;
    Ok(db)
}

fn bind(db: &GrafeoDB, path: &GraphPath, graph_type: &str) -> TestResult {
    db.transaction_manager
        .with_write_authority(|| db.catalog.bind_graph_type(path, graph_type.to_string()))?;
    Ok(())
}

fn assert_current_type(session: &Session, graph_type: &str, open: bool) -> TestResult {
    let result = session.execute("SHOW CURRENT GRAPH TYPE")?;
    assert_eq!(result.row_count(), 1);
    let row = result.rows().first().ok_or("missing graph type row")?;
    assert_eq!(row.get(1), Some(&Value::from(graph_type)));
    assert_eq!(row.get(2), Some(&Value::from(open)));
    Ok(())
}

#[test]
fn native_graph_type_paths_enforce_direct_and_planned_mutations() -> TestResult {
    let db = database()?;
    for path in graph_paths()? {
        let session = db.session();
        session.use_graph_path(&path)?;
        assert_current_type(&session, "Open", true)?;
        session.create_node_with_props(&["Any"], [])?;
        session.execute("INSERT (:Any)")?;
        bind(&db, &path, "Closed")?;
        assert_current_type(&session, "Closed", false)?;
        let direct = session.create_node_with_props(&["Any"], []);
        assert!(
            matches!(direct, Err(Error::Query(_))),
            "direct at {path:?}: {direct:?}"
        );
        let planned = session.execute("INSERT (:Any)");
        assert!(
            matches!(planned, Err(Error::InvalidValue(_))),
            "planned at {path:?}: {planned:?}"
        );
        assert_eq!(session.execute("MATCH (n) RETURN n")?.row_count(), 2);
    }
    Ok(())
}

#[test]
fn native_graph_type_final_validation_observes_late_binding() -> TestResult {
    let db = database()?;
    for path in graph_paths()? {
        let mut session = db.session();
        session.use_graph_path(&path)?;
        session.begin_transaction()?;
        let pending = session.create_node_with_props(&["Any"], [])?;
        session.execute("INSERT (:Any)")?;
        bind(&db, &path, "Closed")?;
        let result = session.commit();
        assert!(
            matches!(result, Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::WriteConflict(ref message)
            )) if message.contains("catalog")),
            "late binding at {path:?}: {result:?}"
        );
        assert!(!session.in_transaction());
        assert!(session.get_node(pending).is_none());
        let reader = db.session();
        reader.use_graph_path(&path)?;
        assert_current_type(&reader, "Closed", false)?;
        assert_eq!(reader.execute("MATCH (n) RETURN n")?.row_count(), 0);
    }
    Ok(())
}

#[test]
fn native_graph_type_bindings_do_not_block_in_memory_catalog_ddl() -> TestResult {
    let db = database()?;
    let session = db.session();
    let nested = GraphPath::from_components(&["a", "b"])?;
    session.use_graph_path(&nested)?;
    let before: std::collections::HashMap<_, _> =
        db.catalog.all_graph_type_bindings().into_iter().collect();
    session.execute("CREATE NODE TYPE Fresh (value STRING)")?;
    assert!(db.catalog.get_node_type("Fresh").is_some());
    session.execute("CREATE SCHEMA IF NOT EXISTS extra")?;
    let epoch = db.current_epoch();
    session.execute("CREATE SCHEMA IF NOT EXISTS extra")?;
    assert_eq!(db.current_epoch(), epoch);
    assert_eq!(session.current_graph_path(), nested);
    assert_eq!(
        db.catalog
            .all_graph_type_bindings()
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>(),
        before
    );
    assert_current_type(&session, "Open", true)?;
    Ok(())
}

#[test]
fn schema_drop_retains_binding_owned_by_literal_parent_component() -> TestResult {
    let db = database()?;
    let session = db.session();
    session.execute("CREATE SCHEMA owned")?;
    let nested = GraphPath::from_components(&["owned/__default__", "child"])?;
    bind(&db, &nested, "Open")?;
    let result = session.execute("DROP SCHEMA owned");
    assert!(
        matches!(result, Err(Error::Query(ref error)) if error.to_string().contains("not empty"))
    );
    assert!(db.catalog.schema_names().iter().any(|name| name == "owned"));
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .graph("owned/__default__")
            .is_some()
    );
    assert_eq!(
        db.catalog.get_graph_type_binding(&nested).as_deref(),
        Some("Open")
    );
    Ok(())
}

#[test]
fn schema_drop_rejects_default_graph_descendants_without_publication() -> TestResult {
    for (name, with_data) in [("literal/child", true), ("", false)] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE SCHEMA owned")?;
        let parent = crate::database::testing::root_lpg_store(&db)
            .graph("owned/__default__")
            .ok_or("missing default")?;
        assert!(
            db.transaction_manager
                .with_write_authority(|| parent.create_graph(name))?
        );
        let child = parent.graph(name).ok_or("missing child")?;
        let path = GraphPath::from_components(&["owned/__default__", name])?;
        session.use_graph_path(&path)?;
        let node = if with_data {
            Some(session.create_node_with_props(&["Kept"], [("value", Value::Int64(42))])?)
        } else {
            None
        };
        let epoch = db.current_epoch();
        let catalog = db.catalog.encode_wal_state_v1()?;
        let result = session.execute("DROP SCHEMA owned");
        assert!(
            matches!(result, Err(Error::Query(ref error)) if error.to_string().contains("not empty")),
            "DROP lost descendant {name:?}: {result:?}"
        );
        assert_eq!(db.current_epoch(), epoch);
        assert_eq!(db.catalog.encode_wal_state_v1()?, catalog);
        assert_eq!(session.current_graph_path(), path);
        assert!(Arc::ptr_eq(
            &parent,
            &crate::database::testing::root_lpg_store(&db)
                .graph("owned/__default__")
                .ok_or("lost default")?
        ));
        assert!(Arc::ptr_eq(
            &child,
            &parent.graph(name).ok_or("lost child")?
        ));
        if let Some(node) = node {
            assert!(session.get_node(node).is_some());
        }
    }
    Ok(())
}

#[test]
fn ancestor_drop_cascades_nested_bindings_only_and_rollback_restores_view() -> TestResult {
    let db = database()?;
    let parent_path = GraphPath::from_components(&["a"])?;
    let nested = GraphPath::from_components(&["a", "b"])?;
    let literal = GraphPath::from_components(&["a/b"])?;
    bind(&db, &nested, "Closed")?;
    let parent = crate::database::testing::root_lpg_store(&db)
        .graph("a")
        .ok_or("missing parent")?;
    let literal_reader = db.session();
    literal_reader.use_graph_path(&literal)?;
    let literal_node = literal_reader.create_node_with_props(&["Kept"], [])?;

    let mut dropper = db.session();
    dropper.begin_transaction()?;
    dropper.execute("DROP GRAPH a")?;
    assert_eq!(dropper.session_graph_type_binding(&parent_path), None);
    assert_eq!(dropper.session_graph_type_binding(&nested), None);
    assert_eq!(
        dropper.session_graph_type_binding(&literal).as_deref(),
        Some("Open")
    );
    assert_eq!(
        db.catalog.get_graph_type_binding(&nested).as_deref(),
        Some("Closed")
    );
    dropper.rollback()?;
    assert_eq!(
        dropper.session_graph_type_binding(&nested).as_deref(),
        Some("Closed")
    );
    assert!(Arc::ptr_eq(
        &parent,
        &crate::database::testing::root_lpg_store(&db)
            .graph("a")
            .ok_or("rollback lost parent")?
    ));
    dropper.use_graph_path(&nested)?;
    assert_current_type(&dropper, "Closed", false)?;
    dropper.use_graph_path(&GraphPath::root())?;

    dropper.begin_transaction()?;
    dropper.execute("DROP GRAPH a")?;
    dropper.commit()?;
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .graph("a")
            .is_none()
    );
    assert_eq!(db.catalog.get_graph_type_binding(&parent_path), None);
    assert_eq!(db.catalog.get_graph_type_binding(&nested), None);
    assert_eq!(
        db.catalog.get_graph_type_binding(&literal).as_deref(),
        Some("Open")
    );
    assert_eq!(
        db.catalog
            .get_graph_type_binding(&GraphPath::root())
            .as_deref(),
        Some("Open")
    );
    assert_eq!(
        db.catalog
            .get_graph_type_binding(&GraphPath::from_components(&[""])?)
            .as_deref(),
        Some("Open")
    );
    assert_current_type(&literal_reader, "Open", true)?;
    assert!(literal_reader.get_node(literal_node).is_some());
    Ok(())
}

#[test]
fn late_nested_binding_after_ancestor_drop_rejects_without_publication() -> TestResult {
    let db = database()?;
    let nested = GraphPath::from_components(&["a", "b"])?;
    let parent = crate::database::testing::root_lpg_store(&db)
        .graph("a")
        .ok_or("missing parent")?;
    let child = parent.graph("b").ok_or("missing child")?;
    assert!(db.transaction_manager.with_write_authority(|| {
        db.catalog
            .publish_graph_type_binding_if_same(&nested, Some("Open"), None)
    }));
    let mut dropper = db.session();
    dropper.begin_transaction()?;
    dropper.execute("DROP GRAPH a")?;
    bind(&db, &nested, "Closed")?;
    let result = dropper.commit();
    assert!(
        matches!(
            result,
            Err(Error::Transaction(TransactionError::WriteConflict(_)))
        ),
        "late cascade binding: {result:?}"
    );
    assert!(!dropper.in_transaction());
    let retained_parent = crate::database::testing::root_lpg_store(&db)
        .graph("a")
        .ok_or("failed commit dropped parent")?;
    assert!(Arc::ptr_eq(&parent, &retained_parent));
    assert!(Arc::ptr_eq(
        &child,
        &retained_parent
            .graph("b")
            .ok_or("failed commit dropped child")?
    ));
    assert_eq!(
        db.catalog.get_graph_type_binding(&nested).as_deref(),
        Some("Closed")
    );
    dropper.use_graph_path(&nested)?;
    assert_current_type(&dropper, "Closed", false)?;
    Ok(())
}
