//! C1 durability: fail-closed abort/log, commit+epoch, no abort resurrection.
//!
//! ```text
//! cargo test -p grafeo-engine --features "triple-store,sparql,wal,grafeo-file,testing-crash-injection" \
//!   --test c1_durability -- --test-threads=1
//! ```

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection",
    feature = "lpg"
))]

use grafeo_common::testing::wal_failure::{
    disable_abort_log_failure, disable_commit_ack_failure, disable_commit_log_failure,
    enable_abort_log_failure_once, enable_commit_ack_failure_once, enable_commit_log_failure_once,
};
use grafeo_common::types::{EpochId, PropertyKey, Value};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn persistent_sync(path: &std::path::Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Rdf)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open rdf db")
}

fn count_pred(db: &GrafeoDB, pred: &str) -> usize {
    db.execute_sparql(&format!("SELECT ?o WHERE {{ ?s <{pred}> ?o }}"))
        .unwrap()
        .row_count()
}

fn persistent_lpg_sync(path: &std::path::Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Lpg)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open lpg db")
}

/// Session::commit must return the crash-stable epoch, not a later sample.
#[test]
fn commit_returns_force_synced_epoch() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("epoch.grafeo");
    let db = persistent_sync(&path);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" . }"#)
        .unwrap();
    let epoch: EpochId = session.commit().expect("commit must return EpochId");
    assert_eq!(
        epoch,
        db.rdf_store_commit_epoch(),
        "returned epoch must be the RDF commit epoch, not a later sample"
    );
    assert!(epoch.as_u64() > 0, "commit epoch must advance");
}

/// Abort WAL log failure must be Err and poison the database (all sessions).
#[test]
fn abort_wal_failure_returns_err_and_poisons() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("poison.grafeo");
    let db = persistent_sync(&path);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(r#"INSERT DATA { <http://ex.org/dead> <http://ex.org/p> "x" . }"#)
        .unwrap();
    enable_abort_log_failure_once();
    let err = session.rollback();
    disable_abort_log_failure();
    assert!(
        err.is_err(),
        "abort WAL log failure must return Err, got {err:?}"
    );
    assert!(
        db.is_durability_poisoned(),
        "abort WAL log failure must poison the database"
    );
    let mut other = db.session();
    let later = other.begin_transaction().and_then(|()| other.commit());
    assert!(
        later.is_err(),
        "poisoned database must refuse a later begin/commit, got {later:?}"
    );
}

/// Aborted SPARQL INSERT must not reappear after a later successful commit.
#[test]
fn aborted_insert_not_resurrected_by_later_commit() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("resurrect.grafeo");
    {
        let db = persistent_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/aborted> <http://ex.org/p> "dead" . }"#)
            .unwrap();
        enable_abort_log_failure_once();
        let _ = session.rollback();
        disable_abort_log_failure();

        if !db.is_durability_poisoned() {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute_sparql(r#"INSERT DATA { <http://ex.org/live> <http://ex.org/p> "ok" . }"#)
                .unwrap();
            let _ = session.commit();
        }
        let _ = db.close();
    }

    let db = persistent_sync(&path);
    assert_eq!(
        count_pred(&db, "http://ex.org/p"),
        0,
        "aborted insert must not be resurrected by a later commit"
    );
}

/// A commit append rejected before the marker is written must remain
/// unpublished.  The caller still treats the result as outcome-ambiguous and
/// poisons the handle; recovery discovers that no marker exists and aborts it.
#[test]
fn rdf_commit_append_failure_is_unpublished_and_recovery_aborts_it() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_commit_failure.grafeo");
    {
        let db = persistent_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(
                r#"INSERT DATA { <http://ex.org/uncommitted> <http://ex.org/p> "dead" . }"#,
            )
            .unwrap();

        enable_commit_log_failure_once();
        let result = session.commit();
        disable_commit_log_failure();

        assert!(result.is_err(), "failed commit append must be reported");
        assert!(db.is_durability_poisoned());
        assert!(
            db.rdf_store().is_empty(),
            "RDF pending data must not become live after commit-marker failure"
        );
    }

    let reopened = persistent_sync(&path);
    assert_eq!(count_pred(&reopened, "http://ex.org/p"), 0);
}

/// Explicit LPG transactions reserve an epoch but do not finalize or publish
/// their versions after a pre-append failure; recovery sees no commit marker.
#[test]
fn lpg_commit_append_failure_is_unpublished_and_recovery_aborts_it() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("lpg_commit_failure.grafeo");
    {
        let db = persistent_lpg_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let id = session.create_node(&["Uncommitted"]);
        assert!(id.is_valid());

        enable_commit_log_failure_once();
        let result = session.commit();
        disable_commit_log_failure();

        assert!(result.is_err(), "failed commit append must be reported");
        assert!(db.is_durability_poisoned());
        assert_eq!(db.node_count(), 0, "PENDING node must stay invisible");
    }

    let reopened = persistent_lpg_sync(&path);
    assert_eq!(reopened.node_count(), 0);
}

/// Losing the acknowledgement after the RDF commit marker was force-synced is
/// the opposite recovery outcome: runtime publication is withheld, but reopen
/// must honor the durable marker.  Writing a compensating abort would corrupt
/// this case.
#[test]
fn rdf_lost_commit_ack_is_resolved_as_committed_by_recovery() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf_lost_commit_ack.grafeo");
    {
        let db = persistent_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(
                r#"INSERT DATA { <http://ex.org/committed> <http://ex.org/p> "alive" . }"#,
            )
            .unwrap();

        enable_commit_ack_failure_once();
        let result = session.commit();
        disable_commit_ack_failure();

        let message = result.expect_err("lost acknowledgement must be reported");
        assert!(message.to_string().contains("outcome is unknown"));
        assert!(db.is_durability_poisoned());
        assert!(db.close().is_err(), "poisoned close must preserve the WAL");
    }

    let reopened = persistent_sync(&path);
    assert_eq!(count_pred(&reopened, "http://ex.org/p"), 1);
}

/// The LPG half of the same C1 ambiguity: the prepared node stays invisible in
/// the failed process, and the force-synced commit marker makes it visible only
/// after recovery establishes the outcome.
#[test]
fn lpg_lost_commit_ack_is_resolved_as_committed_by_recovery() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("lpg_lost_commit_ack.grafeo");
    {
        let db = persistent_lpg_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let id = session.create_node(&["Committed"]);
        assert!(id.is_valid());

        enable_commit_ack_failure_once();
        let result = session.commit();
        disable_commit_ack_failure();

        let message = result.expect_err("lost acknowledgement must be reported");
        assert!(message.to_string().contains("outcome is unknown"));
        assert!(db.is_durability_poisoned());
        assert_eq!(
            db.node_count(),
            0,
            "prepared node must not publish in-process"
        );
        assert!(db.close().is_err(), "poisoned close must preserve the WAL");
    }

    let reopened = persistent_lpg_sync(&path);
    assert_eq!(reopened.node_count(), 1);
}

/// GrafeoDB's one-shot convenience CRUD uses the same Session transaction
/// boundary; it cannot leak a live node when its commit record fails.
#[test]
fn one_shot_lpg_commit_marker_failure_is_atomic() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("one_shot_commit_failure.grafeo");
    {
        let db = persistent_lpg_sync(&path);
        enable_commit_log_failure_once();
        let id = db.create_node(&["Uncommitted"]);
        disable_commit_log_failure();

        assert!(
            !id.is_valid(),
            "one-shot CRUD must surface failure as INVALID"
        );
        assert!(db.is_durability_poisoned());
        assert_eq!(db.node_count(), 0);
    }

    let reopened = persistent_lpg_sync(&path);
    assert_eq!(reopened.node_count(), 0);
}

/// Bulk/import entry points also share the Session transaction boundary; they
/// must roll the whole batch back when the single commit marker fails.
#[test]
fn bulk_import_commit_marker_failure_is_atomic() {
    for graph in [None, Some("import_named")] {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join(format!(
            "bulk_commit_failure_{}.grafeo",
            graph.unwrap_or("default")
        ));
        {
            let db = persistent_lpg_sync(&path);
            if let Some(graph) = graph {
                db.create_graph(graph).unwrap();
                db.set_current_graph(Some(graph)).unwrap();
            }

            enable_commit_log_failure_once();
            let result = db.import_tsv_str("1 2\n2 3\n", "CONNECTS", true);
            disable_commit_log_failure();

            assert!(result.is_err(), "bulk import must report commit failure");
            assert!(db.is_durability_poisoned());
            if let Some(graph) = graph {
                let named = grafeo_engine::database::testing::root_lpg_store(&db)
                    .graph(graph)
                    .expect("named graph remains");
                assert_eq!(named.node_count(), 0);
                assert_eq!(named.edge_count(), 0);
                assert_eq!(
                    db.node_count(),
                    0,
                    "named import must not fall back to default"
                );
                assert_eq!(
                    db.edge_count(),
                    0,
                    "named import must not fall back to default"
                );
            } else {
                assert_eq!(db.node_count(), 0);
                assert_eq!(db.edge_count(), 0);
            }
        }

        let reopened = persistent_lpg_sync(&path);
        if let Some(graph) = graph {
            let named = grafeo_engine::database::testing::root_lpg_store(&reopened)
                .graph(graph)
                .expect("committed named graph must recover");
            assert_eq!(named.node_count(), 0);
            assert_eq!(named.edge_count(), 0);
            assert_eq!(reopened.node_count(), 0);
            assert_eq!(reopened.edge_count(), 0);
        } else {
            assert_eq!(reopened.node_count(), 0);
            assert_eq!(reopened.edge_count(), 0);
        }
    }
}

/// Both public node-batch APIs share one Session commit boundary. A failed
/// commit marker must publish no prefix in either the default or a named graph.
#[test]
fn node_batch_commit_marker_failure_is_atomic() {
    for graph in [None, Some("batch_named")] {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join(format!(
            "node_batch_commit_failure_{}.grafeo",
            graph.unwrap_or("default")
        ));
        {
            let db = persistent_lpg_sync(&path);
            if let Some(graph) = graph {
                db.create_graph(graph).unwrap();
                db.set_current_graph(Some(graph)).unwrap();
            }

            enable_commit_log_failure_once();
            let ids = if graph.is_some() {
                db.batch_create_nodes_with_props(
                    "Doc",
                    vec![std::collections::HashMap::from([(
                        PropertyKey::new("embedding"),
                        Value::Vector(vec![1.0, 0.0].into()),
                    )])],
                )
            } else {
                db.batch_create_nodes("Doc", "embedding", vec![vec![1.0, 0.0], vec![0.0, 1.0]])
            };
            disable_commit_log_failure();

            assert!(ids.is_empty(), "failed batch must return no live ids");
            assert!(db.is_durability_poisoned());
            if let Some(graph) = graph {
                assert_eq!(
                    grafeo_engine::database::testing::root_lpg_store(&db)
                        .graph(graph)
                        .expect("named graph remains")
                        .node_count(),
                    0
                );
                assert_eq!(
                    db.node_count(),
                    0,
                    "named batch must not fall back to default"
                );
            } else {
                assert_eq!(db.node_count(), 0);
            }
        }

        let reopened = persistent_lpg_sync(&path);
        if let Some(graph) = graph {
            assert_eq!(
                grafeo_engine::database::testing::root_lpg_store(&reopened)
                    .graph(graph)
                    .expect("committed named graph must recover")
                    .node_count(),
                0
            );
            assert_eq!(reopened.node_count(), 0);
        } else {
            assert_eq!(reopened.node_count(), 0);
        }
    }
}
