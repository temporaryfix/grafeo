//! Catalog/data commit markers recover together after abrupt process exit.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]

use std::path::Path;
use std::process::Command;

use grafeo_common::types::Value;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel, Session};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn persistent(path: &Path) -> Result<GrafeoDB, grafeo_common::utils::error::Error> {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
}

fn stage(session: &Session) -> TestResult {
    session.execute("START TRANSACTION")?;
    session.execute("CREATE SCHEMA records")?;
    session.execute("SESSION SET SCHEMA records")?;
    session.execute("CREATE NODE TYPE Item (value INTEGER NOT NULL)")?;
    session.execute("CREATE GRAPH TYPE Items (NODE TYPE Item)")?;
    session.execute("CREATE GRAPH data TYPED Items")?;
    session.execute("SESSION SET GRAPH data")?;
    session.execute("INSERT (:Item {value: 5})")?;
    session.execute("CREATE INDEX item_value FOR (n:Item) ON (n.value)")?;
    session.execute("CREATE PROCEDURE read_item() RETURNS (value INTEGER) AS { MATCH (n:Item) RETURN n.value AS value }")?;
    session.savepoint("retained")?;
    session.execute("CREATE SCHEMA discarded")?;
    session
        .execute("CREATE PROCEDURE discarded() RETURNS (value INTEGER) AS { RETURN 9 AS value }")?;
    session.rollback_to_savepoint("retained")?;
    Ok(())
}

#[test]
fn catalog_transaction_child() -> TestResult {
    let Ok(path) = std::env::var("GRAFEO_CATALOG_TX_CHILD_PATH") else {
        return Ok(());
    };
    let mode = std::env::var("GRAFEO_CATALOG_TX_CHILD_MODE")?;
    let db = persistent(Path::new(&path))?;
    let session = db.session();
    stage(&session)?;
    match mode.as_str() {
        "before_marker" | "after_marker" => {
            std::panic::set_hook(Box::new(|_| std::process::exit(87)));
            grafeo_common::testing::crash::enable_crash_named(if mode == "before_marker" {
                "commit:before_marker"
            } else {
                "commit:after_marker_before_publication"
            });
            session.execute("COMMIT")?;
            return Err("commit crash site was not reached".into());
        }
        "commit" => {
            session.execute("COMMIT")?;
        }
        "abort" => {
            session.execute("ROLLBACK")?;
        }
        "pending" => {}
        "lost_ack" => {
            grafeo_common::testing::wal_failure::enable_commit_ack_failure_once();
            let result = session.execute("COMMIT");
            grafeo_common::testing::wal_failure::disable_commit_ack_failure();
            assert!(result.is_err());
            assert!(db.is_durability_poisoned());
        }
        "failed_append" => {
            grafeo_common::testing::wal_failure::enable_commit_log_failure_once();
            let result = session.execute("COMMIT");
            grafeo_common::testing::wal_failure::disable_commit_log_failure();
            assert!(result.is_err());
            assert!(db.is_durability_poisoned());
        }
        _ => return Err("unknown catalog recovery test mode".into()),
    }
    // No Session/DB destructor, checkpoint, or graceful shutdown may repair the tail.
    std::process::exit(88);
}

fn recover(mode: &str, published: bool) -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("catalog.grafeo");
    let output = Command::new(std::env::current_exe()?)
        .args(["--exact", "catalog_transaction_child", "--nocapture"])
        .env("GRAFEO_CATALOG_TX_CHILD_PATH", &path)
        .env("GRAFEO_CATALOG_TX_CHILD_MODE", mode)
        .output()?;
    let expected = if mode.ends_with("marker") { 87 } else { 88 };
    assert_eq!(
        output.status.code(),
        Some(expected),
        "child {mode}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for _ in 0..2 {
        let db = persistent(&path)?;
        let session = db.session();
        assert!(session.execute("SESSION SET SCHEMA discarded").is_err());
        assert!(session.execute("CALL discarded()").is_err());
        if published {
            session.execute("SESSION SET SCHEMA records")?;
            session.execute("SESSION SET GRAPH data")?;
            assert_eq!(
                session.execute("CALL read_item()")?.rows(),
                &[vec![Value::Int64(5)]]
            );
            assert_eq!(session.execute("SHOW INDEXES")?.rows().len(), 1);
            assert!(session.execute("INSERT (:Item {value: 'wrong'})").is_err());
        } else {
            assert!(session.execute("SESSION SET SCHEMA records").is_err());
            assert!(session.execute("CALL read_item()").is_err());
            assert!(db.list_graphs().is_empty());
        }
        drop(session);
        db.close()?;
    }
    Ok(())
}

#[test]
fn acknowledged_catalog_data_and_owners_recover() -> TestResult {
    recover("commit", true)
}

#[test]
fn uncommitted_catalog_and_data_do_not_recover() -> TestResult {
    recover("pending", false)
}

#[test]
fn aborted_catalog_and_data_do_not_recover() -> TestResult {
    recover("abort", false)
}

#[test]
fn catalog_postimage_before_marker_is_not_a_commit() -> TestResult {
    recover("before_marker", false)
}

#[test]
fn durable_marker_recovers_before_in_memory_publication() -> TestResult {
    recover("after_marker", true)
}

#[test]
fn failed_commit_append_recovers_the_preimage() -> TestResult {
    recover("failed_append", false)
}

#[test]
fn lost_commit_ack_recovers_the_postimage() -> TestResult {
    recover("lost_ack", true)
}
