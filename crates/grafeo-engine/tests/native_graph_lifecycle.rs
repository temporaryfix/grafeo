//! Public, literal GraphPath lifecycle controls without raw topology mutation.
#![cfg(feature = "lpg")]

use std::sync::Arc;

use grafeo_common::types::{GraphPath, NodeId, Value};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_engine::auth::{Grant, Identity, Role};
use grafeo_engine::{CreateIndexRequest, GrafeoDB, IndexCreateKind};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn path(parts: &[&str]) -> TestResult<GraphPath> {
    Ok(GraphPath::from_components(parts)?)
}

fn target(db: &GrafeoDB, path: &GraphPath) -> Option<Arc<LpgStore>> {
    let mut store = Arc::clone(grafeo_engine::database::testing::root_lpg_store(db));
    for component in path.components() {
        store = store.graph(component)?;
    }
    Some(store)
}

fn property_index(graph: &GraphPath, name: &str) -> CreateIndexRequest {
    CreateIndexRequest {
        graph: graph.clone(),
        name: Some(name.into()),
        label: None,
        property: "value".into(),
        kind: IndexCreateKind::Property,
    }
}

#[test]
fn literal_paths_create_parent_child_data_and_indexes_in_one_transaction() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let root = GraphPath::root();
    assert!(!db.create_graph_path(&root)?);
    assert!(db.drop_graph_path(&root).is_err());
    let missing = path(&["missing", "child"])?;
    assert!(db.create_graph_path(&missing).is_err());
    assert!(target(&db, &path(&["missing"])?).is_none());
    assert!(!db.drop_graph_path(&missing)?);
    let root_node = db
        .session()
        .create_node_with_props(&["Root"], [("value", Value::Int64(-1))])?;
    let paths = [
        path(&[""])?,
        path(&["default"])?,
        path(&["a/b"])?,
        path(&["a"])?,
        path(&["a", "b"])?,
        path(&["日本語"])?,
        path(&["日本語", "📚"])?,
    ];
    let mut owner = db.session();
    let observer = db.session();
    owner.begin_transaction()?;
    for (ordinal, graph) in paths.iter().enumerate() {
        assert!(owner.create_graph_path(graph)?);
        assert!(
            !owner.create_graph_path(graph)?,
            "duplicate creation is a no-op"
        );
        owner.use_graph_path(graph)?;
        let node = owner
            .create_node_with_props(&["Doc"], [("value", Value::Int64(i64::try_from(ordinal)?))])?;
        assert_eq!(node, NodeId::new(0));
        assert_eq!(
            owner.get_node_property(node, "value"),
            Some(Value::Int64(i64::try_from(ordinal)?))
        );
        #[cfg(feature = "gql")]
        owner.execute(&format!(
            "CREATE INDEX literal_{ordinal} FOR (n:Doc) ON (n.value)"
        ))?;
        assert!(
            target(&db, graph).is_none(),
            "staged topology is not globally visible"
        );
        assert!(observer.use_graph_path(graph).is_err());
    }
    let epoch = owner.commit()?;
    for (ordinal, graph) in paths.iter().enumerate() {
        observer.use_graph_path(graph)?;
        assert_eq!(
            observer.get_node_property(NodeId::new(0), "value"),
            Some(Value::Int64(i64::try_from(ordinal)?))
        );
        let store = target(&db, graph).ok_or("committed graph missing")?;
        assert_eq!(store.current_epoch(), epoch);
        #[cfg(feature = "gql")]
        assert!(store.has_property_index("value"));
    }
    assert_eq!(db.node_count(), 1);
    assert_eq!(
        db.get_node(root_node)
            .ok_or("root node missing")?
            .get_property("value"),
        Some(&Value::Int64(-1))
    );
    Ok(())
}

#[test]
fn cancelling_and_rolling_back_staged_subtrees_leaves_no_descendants() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let parent = path(&["parent"])?;
    let child = path(&["parent", "child"])?;
    let grandchild = path(&["parent", "child", "leaf"])?;
    let mut session = db.session();
    session.begin_transaction()?;
    for graph in [&parent, &child, &grandchild] {
        assert!(session.create_graph_path(graph)?);
    }
    session.use_graph_path(&grandchild)?;
    session.create_node_with_props(&["Cancelled"], [("value", Value::Int64(1))])?;
    #[cfg(feature = "gql")]
    session.execute("CREATE INDEX cancelled_owner FOR (n:Cancelled) ON (n.value)")?;
    assert!(session.drop_graph_path(&parent)?);
    for graph in [&parent, &child, &grandchild] {
        assert!(session.use_graph_path(graph).is_err());
    }
    assert!(!session.drop_graph_path(&parent)?);
    session.commit()?;
    assert!(target(&db, &parent).is_none());
    assert_eq!(db.node_count(), 0);

    session.begin_transaction()?;
    assert!(session.create_graph_path(&parent)?);
    assert!(session.create_graph_path(&child)?);
    session.use_graph_path(&child)?;
    session.create_node_with_props(&["RolledBack"], [("value", Value::Int64(2))])?;
    session.rollback()?;
    assert!(target(&db, &parent).is_none());
    assert_eq!(db.node_count(), 0);
    assert!(db.create_graph_path(&parent)?);
    assert!(db.create_graph_path(&child)?);
    assert!(target(&db, &grandchild).is_none());
    assert_eq!(
        target(&db, &child)
            .ok_or("fresh child missing")?
            .node_count(),
        0
    );
    #[cfg(feature = "gql")]
    assert!(db.session().execute("SHOW INDEXES")?.rows().is_empty());
    Ok(())
}

#[test]
fn savepoint_restores_staged_descendants_after_drop_and_recreate() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let parent = path(&["parent"])?;
    let child = path(&["parent", "child"])?;
    let grandchild = path(&["parent", "child", "leaf"])?;
    assert!(db.create_graph_path(&parent)?);
    let mut session = db.session();
    session.begin_transaction()?;
    for graph in [&child, &grandchild] {
        assert!(session.create_graph_path(graph)?);
        session.use_graph_path(graph)?;
        assert_eq!(
            session.create_node_with_props(&["Original"], [("value", Value::Int64(1))])?,
            NodeId::new(0)
        );
    }
    #[cfg(feature = "gql")]
    session.execute("CREATE INDEX retained_owner FOR (n:Original) ON (n.value)")?;
    session.savepoint("original_tree")?;
    assert!(session.drop_graph_path(&child)?);
    assert!(session.use_graph_path(&grandchild).is_err());
    for graph in [&child, &grandchild] {
        assert!(session.create_graph_path(graph)?);
        session.use_graph_path(graph)?;
        assert_eq!(
            session.create_node_with_props(&["Replacement"], [("value", Value::Int64(2))])?,
            NodeId::new(0)
        );
    }
    #[cfg(feature = "gql")]
    session.execute("CREATE INDEX discarded_owner FOR (n:Replacement) ON (n.value)")?;
    session.rollback_to_savepoint("original_tree")?;
    for graph in [&child, &grandchild] {
        session.use_graph_path(graph)?;
        assert_eq!(
            session.get_node_property(NodeId::new(0), "value"),
            Some(Value::Int64(1))
        );
    }
    session.commit()?;
    let old_child = target(&db, &child).ok_or("restored child missing")?;
    let old_grandchild = target(&db, &grandchild).ok_or("restored grandchild missing")?;
    #[cfg(feature = "gql")]
    assert!(old_grandchild.has_property_index("value"));
    assert_eq!(old_grandchild.node_count(), 1);

    session.begin_transaction()?;
    assert!(session.drop_graph_path(&child)?);
    assert!(session.create_graph_path(&child)?);
    session.use_graph_path(&child)?;
    session.create_node_with_props(&["Final"], [("value", Value::Int64(3))])?;
    session.commit()?;
    let replacement = target(&db, &child).ok_or("replacement child missing")?;
    assert!(!Arc::ptr_eq(&replacement, &old_child));
    assert!(target(&db, &grandchild).is_none());
    assert_eq!(replacement.node_count(), 1);
    assert!(!replacement.has_property_index("value"));
    #[cfg(feature = "gql")]
    assert!(db.session().execute("SHOW INDEXES")?.rows().is_empty());
    Ok(())
}

#[test]
fn subtree_drop_cascades_exact_index_owners_without_flat_name_collision() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let flat = path(&["a/b"])?;
    let parent = path(&["a"])?;
    let child = path(&["a", "b"])?;
    let grandchild = path(&["a", "b", "leaf"])?;
    for graph in [&flat, &parent, &child, &grandchild] {
        assert!(db.create_graph_path(graph)?);
    }
    let mut owners = Vec::new();
    for (ordinal, graph) in [&flat, &child, &grandchild].into_iter().enumerate() {
        let session = db.session();
        session.use_graph_path(graph)?;
        session
            .create_node_with_props(&["Doc"], [("value", Value::Int64(i64::try_from(ordinal)?))])?;
        owners.push(db.create_index(property_index(graph, &format!("owner_{ordinal}")))?);
    }
    assert!(db.drop_graph_path(&parent)?);
    for graph in [&parent, &child, &grandchild] {
        assert!(target(&db, graph).is_none());
    }
    assert!(!db.drop_index(owners[1])?);
    assert!(!db.drop_index(owners[2])?);
    let flat_store = target(&db, &flat).ok_or("literal slash graph was cascaded")?;
    assert_eq!(flat_store.node_count(), 1);
    assert!(flat_store.has_property_index("value"));
    assert!(db.drop_index(owners[0])?);
    assert!(!flat_store.has_property_index("value"));
    assert!(db.create_graph_path(&parent)?);
    assert!(db.create_graph_path(&child)?);
    let next = db.create_index(property_index(&child, "replacement_owner"))?;
    assert!(
        next.as_u32() > owners[2].as_u32(),
        "cascaded owners are not reused"
    );
    assert_eq!(db.node_count(), 0);
    Ok(())
}

#[test]
fn committed_parent_replacement_conflicts_with_staged_child_publication() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let parent = path(&["parent"])?;
    let child = path(&["parent", "child"])?;
    assert!(db.create_graph_path(&parent)?);
    let original = target(&db, &parent).ok_or("parent missing")?;
    let mut stale = db.session();
    stale.begin_transaction()?;
    assert!(stale.create_graph_path(&child)?);
    stale.use_graph_path(&child)?;
    stale.create_node_with_props(&["Stale"], [("value", Value::Int64(1))])?;
    #[cfg(feature = "gql")]
    stale.execute("CREATE INDEX stale_owner FOR (n:Stale) ON (n.value)")?;

    let mut replacer = db.session();
    replacer.begin_transaction()?;
    assert!(replacer.drop_graph_path(&parent)?);
    assert!(replacer.create_graph_path(&parent)?);
    replacer.commit()?;
    let replacement = target(&db, &parent).ok_or("replacement parent missing")?;
    assert!(!Arc::ptr_eq(&original, &replacement));
    assert!(
        stale.commit().is_err(),
        "a child cannot migrate to a different parent incarnation"
    );
    if stale.in_transaction() {
        stale.rollback()?;
    }
    assert!(target(&db, &child).is_none());
    assert_eq!(replacement.node_count(), 0);
    assert_eq!(db.node_count(), 0);
    #[cfg(feature = "gql")]
    assert!(db.session().execute("SHOW INDEXES")?.rows().is_empty());
    Ok(())
}

#[test]
fn lifecycle_enforces_roles_and_literal_target_grants() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let parent = path(&["a"])?;
    let flat = path(&["a/b"])?;
    let nested = path(&["a", "b"])?;
    assert!(db.create_graph_path(&parent)?);
    assert!(db.create_graph_path(&nested)?);
    let reader = db.session_with_role(Role::ReadOnly);
    assert!(reader.create_graph_path(&flat).is_err());
    assert!(reader.drop_graph_path(&nested).is_err());
    let scoped =
        db.session_with_identity(Identity::new("flat-owner", [Role::ReadWrite]).with_grants([
            Grant::new(flat.clone(), Role::ReadWrite),
            Grant::new(parent.clone(), Role::ReadOnly),
        ]));
    assert!(scoped.create_graph_path(&flat)?);
    assert!(scoped.drop_graph_path(&nested).is_err());
    assert!(scoped.create_graph_path(&path(&["a", "other"])?).is_err());
    assert!(target(&db, &nested).is_some());
    assert!(scoped.drop_graph_path(&flat)?);
    assert!(target(&db, &flat).is_none());
    assert!(target(&db, &parent).is_some());
    Ok(())
}

#[cfg(feature = "gql")]
#[test]
fn native_topology_does_not_inherit_the_current_language_schema() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA Scoped")?;
    session.execute("SESSION SET SCHEMA Scoped")?;
    let parent = path(&["a"])?;
    let nested = path(&["a", "b"])?;
    let flat = path(&["a/b"])?;
    assert!(session.create_graph_path(&parent)?);
    assert!(session.create_graph_path(&nested)?);
    assert!(session.create_graph_path(&flat)?);
    session.use_graph_path(&nested)?;
    session.create_node_with_props(&["Native"], [("value", Value::Int64(1))])?;
    assert_eq!(session.current_schema().as_deref(), Some("Scoped"));
    assert_eq!(session.current_graph_path(), nested);
    assert!(target(&db, &path(&["Scoped/a"])?).is_none());
    assert!(target(&db, &path(&["Scoped/a/b"])?).is_none());
    assert_eq!(
        target(&db, &nested)
            .ok_or("native child missing")?
            .node_count(),
        1
    );
    assert!(session.drop_graph_path(&parent)?);
    assert!(target(&db, &nested).is_none());
    assert!(target(&db, &flat).is_some());
    Ok(())
}
