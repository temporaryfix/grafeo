//! Canonical property setters must distinguish rejection from committed success.

#![cfg(feature = "lpg")]

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_engine::{Config, GrafeoDB};

fn seed(db: &GrafeoDB) -> Result<(NodeId, EdgeId)> {
    let session = db.session();
    let node = session.create_node_with_props(&["Asset"], [("status", Value::from("old"))])?;
    let edge = session.create_edge_with_props(node, node, "LINK", [("weight", Value::Int64(1))])?;
    Ok((node, edge))
}

#[cfg(all(feature = "wal", feature = "testing-crash-injection"))]
fn primary_error(mut error: &Error) -> &Error {
    while let Error::Context { source, .. } = error {
        error = source;
    }
    error
}

#[test]
fn database_setters_report_missing_entities() {
    let db = GrafeoDB::new_in_memory();
    let before = db.current_epoch();
    let node = NodeId::new(42);
    let edge = EdgeId(42);
    let node_result: Result<()> = db.set_node_property(node, "status", Value::from("ready"));
    let edge_result: Result<()> = db.set_edge_property(edge, "weight", Value::Int64(7));
    assert!(matches!(node_result, Err(Error::NodeNotFound(id)) if id == node));
    assert!(matches!(edge_result, Err(Error::EdgeNotFound(id)) if id == edge));
    assert_eq!(db.current_epoch(), before);
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

#[test]
fn session_missing_node_is_an_error_without_publication() {
    let db = GrafeoDB::new_in_memory();
    let before = db.current_epoch();
    let session = db.session();
    let missing = NodeId::new(42);
    let result = session.set_node_property(missing, "status", Value::from("ready"));
    assert!(matches!(result, Err(Error::NodeNotFound(id)) if id == missing));
    assert!(!session.in_transaction());
    assert_eq!(db.current_epoch(), before);
    assert_eq!(db.node_count(), 0);
}

#[test]
fn session_missing_edge_is_an_error_without_publication() {
    let db = GrafeoDB::new_in_memory();
    let before = db.current_epoch();
    let session = db.session();
    let missing = EdgeId(42);
    let result = session.set_edge_property(missing, "status", Value::from("ready"));
    assert!(matches!(result, Err(Error::EdgeNotFound(id)) if id == missing));
    assert!(!session.in_transaction());
    assert_eq!(db.current_epoch(), before);
    assert_eq!(db.edge_count(), 0);
}

#[test]
fn setters_see_transaction_local_entities_and_rollback() -> Result<()> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction()?;
    let node = session.create_node_with_props(&["Asset"], [])?;
    let edge = session.create_edge_with_props(node, node, "LINK", [])?;
    session.set_node_property(node, "status", Value::from("ready"))?;
    session.set_edge_property(edge, "weight", Value::Int64(7))?;
    assert_eq!(
        session
            .get_node(node)
            .and_then(|node| node.get_property("status").cloned()),
        Some(Value::from("ready")),
    );
    assert_eq!(
        session
            .get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(7)),
    );
    session.rollback()?;
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
    Ok(())
}

#[cfg(feature = "gql")]
#[test]
fn read_only_transaction_rejects_setters_and_remains_active() -> Result<()> {
    use grafeo_common::utils::error::TransactionError;

    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let node = session.create_node_with_props(&["Asset"], [])?;
    let edge = session.create_edge_with_props(node, node, "LINK", [])?;
    let before = db.current_epoch();
    session.execute("START TRANSACTION READ ONLY")?;
    assert!(matches!(
        session.set_node_property(node, "status", Value::from("ready")),
        Err(Error::Transaction(TransactionError::ReadOnly)),
    ));
    assert!(matches!(
        session.set_edge_property(edge, "weight", Value::Int64(7)),
        Err(Error::Transaction(TransactionError::ReadOnly)),
    ));
    assert!(session.in_transaction());
    assert_eq!(db.current_epoch(), before);
    assert_eq!(
        session
            .get_node(node)
            .and_then(|node| node.get_property("status").cloned()),
        None,
    );
    assert_eq!(
        session
            .get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        None,
    );
    session.rollback()?;
    Ok(())
}

#[test]
fn database_setters_commit_properties_and_reject_oversized_values() -> Result<()> {
    let db = GrafeoDB::with_config(Config::in_memory().with_max_property_size(64))?;
    let (node, edge) = seed(&db)?;
    let old = db.current_epoch();
    db.set_node_property(node, "status", Value::from("ready"))?;
    db.set_edge_property(edge, "weight", Value::Int64(7))?;
    assert_eq!(
        db.session().get_node_property(node, "status"),
        Some(Value::from("ready"))
    );
    assert_eq!(
        db.get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(7))
    );
    assert_eq!(
        db.get_node_property_at_epoch(node, "status", old),
        Some(Value::from("old"))
    );
    let before = db.current_epoch();
    let too_large = Value::from("x".repeat(128));
    assert!(
        db.set_node_property(node, "status", too_large.clone())
            .is_err()
    );
    assert!(db.set_edge_property(edge, "weight", too_large).is_err());
    assert_eq!(db.current_epoch(), before);
    assert_eq!(
        db.session().get_node_property(node, "status"),
        Some(Value::from("ready"))
    );
    assert_eq!(
        db.get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(7))
    );
    Ok(())
}

#[test]
fn deleted_targets_are_missing_in_their_own_transaction() -> Result<()> {
    let db = GrafeoDB::new_in_memory();
    let (node, edge) = seed(&db)?;
    let before = db.current_epoch();
    let mut session = db.session();
    session.begin_transaction()?;
    assert!(session.delete_edge(edge));
    assert!(session.delete_node(node));
    assert!(
        matches!(session.set_node_property(node, "status", Value::from("new")), Err(Error::NodeNotFound(id)) if id == node)
    );
    assert!(
        matches!(session.set_edge_property(edge, "weight", Value::Int64(7)), Err(Error::EdgeNotFound(id)) if id == edge)
    );
    assert!(session.in_transaction());
    session.rollback()?;
    assert_eq!(db.current_epoch(), before);
    assert_eq!(
        db.session().get_node_property(node, "status"),
        Some(Value::from("old"))
    );
    assert_eq!(
        db.get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(1))
    );
    Ok(())
}

#[test]
fn role_and_historical_rejections_do_not_publish() -> Result<()> {
    use grafeo_common::utils::error::{QueryErrorKind, TransactionError};
    use grafeo_engine::auth::Role;

    let db = GrafeoDB::new_in_memory();
    let (node, edge) = seed(&db)?;
    let before = db.current_epoch();
    let session = db.session_with_role(Role::ReadOnly);
    for result in [
        session.set_node_property(node, "status", Value::from("new")),
        session.set_edge_property(edge, "weight", Value::Int64(7)),
    ] {
        assert!(
            matches!(result, Err(Error::Query(query)) if query.kind == QueryErrorKind::Semantic && query.message.contains("permission denied"))
        );
    }
    assert!(!session.in_transaction());
    let historical = db.session();
    historical.set_viewing_epoch(before);
    for result in [
        historical.set_node_property(node, "status", Value::from("new")),
        historical.set_edge_property(edge, "weight", Value::Int64(7)),
    ] {
        assert!(
            matches!(result, Err(Error::Transaction(TransactionError::InvalidState(message))) if message.contains("historical view is read-only"))
        );
    }
    assert!(!historical.in_transaction());
    assert_eq!(db.current_epoch(), before);
    assert_eq!(
        db.session().get_node_property(node, "status"),
        Some(Value::from("old"))
    );
    assert_eq!(
        db.get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(1))
    );
    Ok(())
}

#[test]
fn closed_database_setters_preserve_the_lifecycle_error() -> Result<()> {
    use grafeo_common::utils::error::TransactionError;

    let db = GrafeoDB::new_in_memory();
    let (node, edge) = seed(&db)?;
    let session = db.session();
    db.close()?;
    for result in [
        db.set_node_property(node, "status", Value::from("new")),
        db.set_edge_property(edge, "weight", Value::Int64(7)),
        session.set_node_property(node, "status", Value::from("new")),
        session.set_edge_property(edge, "weight", Value::Int64(7)),
    ] {
        assert!(
            matches!(result, Err(Error::Transaction(TransactionError::InvalidState(message))) if message == "database is closed")
        );
    }
    Ok(())
}

#[cfg(feature = "triple-store")]
#[test]
fn rdf_database_setters_do_not_hide_model_errors() -> Result<()> {
    use grafeo_engine::config::GraphModel;

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))?;
    for result in [
        db.set_node_property(NodeId::new(42), "status", Value::from("new")),
        db.set_edge_property(EdgeId(42), "weight", Value::Int64(7)),
    ] {
        assert!(
            matches!(result, Err(Error::Internal(message)) if message.contains("RDF database"))
        );
    }
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn database_setters_update_compacted_entities_and_keep_history() -> Result<()> {
    let mut db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    let (node, edge) = seed(&db)?;
    let old = db.current_epoch();
    db.compact()?;
    db.set_node_property(node, "status", Value::from("new"))?;
    db.set_edge_property(edge, "weight", Value::Int64(7))?;
    assert_eq!(
        db.session().get_node_property(node, "status"),
        Some(Value::from("new"))
    );
    assert_eq!(
        db.get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(7))
    );
    assert_eq!(
        db.get_node_property_at_epoch(node, "status", old),
        Some(Value::from("old"))
    );
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn a_stale_snapshot_cannot_commit_a_setter_after_target_deletion() -> Result<()> {
    for compact in [false, true] {
        for set_edge in [false, true] {
            let mut db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
            let (node, edge) = seed(&db)?;
            if compact {
                db.compact()?;
            }
            let mut stale = db.session();
            stale.begin_transaction()?;
            assert!(stale.get_node(node).is_some());
            assert!(stale.get_edge(edge).is_some());
            if set_edge {
                assert!(db.delete_edge(edge));
            } else {
                assert!(db.delete_node(node));
            }
            let before = db.current_epoch();
            let staged = if set_edge {
                stale.set_edge_property(edge, "weight", Value::Int64(7))
            } else {
                stale.set_node_property(node, "status", Value::from("new"))
            };
            assert!(
                matches!(
                    staged,
                    Err(Error::Transaction(
                        grafeo_common::utils::error::TransactionError::WriteConflict(_)
                    ))
                ),
                "stale setter must reject before buffering: compact={compact}, edge={set_edge}"
            );
            assert!(stale.in_transaction());
            stale.rollback()?;
            assert_eq!(db.current_epoch(), before);
            assert!(db.get_edge(edge).is_none());
            if !set_edge {
                assert!(db.get_node(node).is_none());
            }
        }
    }
    Ok(())
}

#[test]
fn setters_cannot_modify_another_transactions_pending_creates() -> Result<()> {
    let db = GrafeoDB::new_in_memory();
    let mut owner = db.session();
    owner.begin_transaction()?;
    let node = owner.create_node_with_props(&["Asset"], [])?;
    let edge = owner.create_edge_with_props(node, node, "LINK", [])?;
    let before = db.current_epoch();
    assert!(
        matches!(db.set_node_property(node, "status", Value::from("new")), Err(Error::NodeNotFound(id)) if id == node)
    );
    assert!(
        matches!(db.set_edge_property(edge, "weight", Value::Int64(7)), Err(Error::EdgeNotFound(id)) if id == edge)
    );
    assert_eq!(db.current_epoch(), before);
    assert_eq!(owner.get_node_property(node, "status"), None);
    assert_eq!(
        owner
            .get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        None
    );
    owner.rollback()?;
    Ok(())
}

#[cfg(feature = "wal")]
#[test]
fn database_setter_results_correspond_to_wal_reopen() -> Result<()> {
    use grafeo_engine::config::StorageFormat;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("setter-wal");
    let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
    let (node, edge);
    {
        let db = GrafeoDB::with_config(config.clone())?;
        (node, edge) = seed(&db)?;
        db.set_node_property(node, "status", Value::from("new"))?;
        db.set_edge_property(edge, "weight", Value::Int64(7))?;
        assert!(matches!(
            db.set_node_property(NodeId::new(42), "status", Value::from("ghost")),
            Err(Error::NodeNotFound(_))
        ));
        assert!(matches!(
            db.set_edge_property(EdgeId(42), "weight", Value::Int64(9)),
            Err(Error::EdgeNotFound(_))
        ));
        db.close()?;
    }
    let reopened = GrafeoDB::with_config(config)?;
    assert_eq!(reopened.node_count(), 1);
    assert_eq!(reopened.edge_count(), 1);
    assert_eq!(
        reopened.session().get_node_property(node, "status"),
        Some(Value::from("new"))
    );
    assert_eq!(
        reopened
            .get_edge(edge)
            .and_then(|edge| edge.get_property("weight").cloned()),
        Some(Value::Int64(7))
    );
    reopened.close()?;
    Ok(())
}

#[cfg(all(feature = "wal", feature = "testing-crash-injection"))]
#[test]
fn missing_targets_do_not_log_mutations_and_durability_errors_are_returned() -> Result<()> {
    use grafeo_common::testing::wal_failure::{
        disable_mutation_log_failure, enable_mutation_log_failure_once,
    };
    use grafeo_common::utils::error::TransactionError;
    use grafeo_engine::config::StorageFormat;

    for set_edge in [false, true] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("setter-poison");
        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let (node, edge);
        {
            let db = GrafeoDB::with_config(config.clone())?;
            (node, edge) = seed(&db)?;
            enable_mutation_log_failure_once();
            let missing = if set_edge {
                db.set_edge_property(EdgeId(42), "weight", Value::Int64(7))
            } else {
                db.set_node_property(NodeId::new(42), "status", Value::from("new"))
            };
            let present = if set_edge {
                db.set_edge_property(edge, "weight", Value::Int64(7))
            } else {
                db.set_node_property(node, "status", Value::from("new"))
            };
            disable_mutation_log_failure();
            assert!(matches!(
                missing,
                Err(Error::NodeNotFound(_)) | Err(Error::EdgeNotFound(_))
            ));
            assert!(matches!(
                present.as_ref().map_err(primary_error),
                Err(Error::Transaction(TransactionError::DurabilityFailure(_)))
            ));
            assert!(db.is_durability_poisoned());
            assert!(matches!(
                db.set_node_property(node, "other", Value::Int64(9)),
                Err(Error::Transaction(TransactionError::DurabilityFailure(_)))
            ));
            assert!(matches!(
                db.set_edge_property(edge, "other", Value::Int64(9)),
                Err(Error::Transaction(TransactionError::DurabilityFailure(_)))
            ));
        }
        let reopened = GrafeoDB::with_config(config)?;
        assert_eq!(
            reopened.session().get_node_property(node, "status"),
            Some(Value::from("old"))
        );
        assert_eq!(
            reopened
                .get_edge(edge)
                .and_then(|edge| edge.get_property("weight").cloned()),
            Some(Value::Int64(1))
        );
        reopened.close()?;
    }
    Ok(())
}

#[cfg(feature = "cdc")]
#[test]
fn setters_emit_one_committed_update_and_no_rejected_updates() -> Result<()> {
    use grafeo_common::types::EpochId;
    use grafeo_engine::cdc::{ChangeKind, EntityId};

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc())?;
    let (node, edge) = seed(&db)?;
    db.set_node_property(node, "status", Value::from("new"))?;
    db.set_edge_property(edge, "weight", Value::Int64(7))?;
    assert!(
        db.set_node_property(NodeId::new(42), "status", Value::from("ghost"))
            .is_err()
    );
    assert!(
        db.set_edge_property(EdgeId(42), "weight", Value::Int64(9))
            .is_err()
    );
    let events = db.fixture_changes(EpochId::INITIAL..=EpochId::PENDING)?;
    let updates: Vec<_> = events
        .iter()
        .filter(|event| event.kind == ChangeKind::Update)
        .collect();
    assert_eq!(updates.len(), 2);
    assert_eq!(
        updates
            .iter()
            .filter(|event| event.entity_id == EntityId::Node(node))
            .count(),
        1
    );
    assert_eq!(
        updates
            .iter()
            .filter(|event| event.entity_id == EntityId::Edge(edge))
            .count(),
        1
    );
    assert!(updates.iter().all(|event| event.epoch != EpochId::PENDING));
    Ok(())
}

#[cfg(feature = "cdc")]
#[path = "support/cdc_pages.rs"]
mod cdc_pages;
#[cfg(feature = "cdc")]
use cdc_pages::CdcFixtureChanges;
