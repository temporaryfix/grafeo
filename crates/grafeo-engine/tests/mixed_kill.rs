//! Mixed LPG+RDF subprocess kill / failpoint recovery.
//!
//! Complements in-process `mixed_publish` failpoints with SIGKILL children
//! (no `close()` / checkpoint) and an env-armed named failpoint.
//!
//! Exclusive file lock means two processes cannot share a live DB, so the
//! concurrent-reader card stays in-process (`mixed_publish`). This file is
//! the subprocess/failpoint recovery matrix.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,triple-store,sparql,wal,grafeo-file,testing-crash-injection" \
//!   --test mixed_kill -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "sparql",
    feature = "triple-store",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::process::Child;
use std::time::{Duration, Instant};

use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn both_sync(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open Both")
}

fn sidecar_wal_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    PathBuf::from(p)
}

fn gql_count(db: &GrafeoDB) -> i64 {
    db.session()
        .execute("MATCH (n) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap()
}

fn sparql_count(db: &GrafeoDB) -> usize {
    db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        .unwrap()
        .row_count()
}

fn wait_ready(ready: &Path) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline && !ready.exists() {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        ready.exists(),
        "child did not durable-ack at {}",
        ready.display()
    );
}

fn kill_wait(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn spawn_role(test_name: &str, role: &str, path: &Path, ready: &Path) -> Child {
    let exe = std::env::current_exe().expect("current_exe for kill-9 child");
    std::process::Command::new(&exe)
        .arg(test_name)
        .arg("--exact")
        .env("MIXED_KILL_ROLE", role)
        .env("MIXED_KILL_PATH", path)
        .env("MIXED_KILL_READY", ready)
        .spawn()
        .expect("spawn child")
}

fn mixed_insert(session: &grafeo_engine::Session, subject: &str, label: &str) {
    session
        .execute(&format!("INSERT (:Person {{name: '{label}'}})"))
        .unwrap();
    session
        .execute_sparql(&format!(
            r#"INSERT DATA {{ <http://ex.org/{subject}> <http://ex.org/p> "{label}" }}"#
        ))
        .unwrap();
}

/// Two-crash mixed: uncommitted write → SIGKILL → reopen → acknowledged
/// mixed commit → SIGKILL (no close) → recover from retained WAL.
fn two_crash_child_main() {
    let path = std::env::var("MIXED_KILL_PATH").expect("MIXED_KILL_PATH");
    let ready = std::env::var("MIXED_KILL_READY").expect("MIXED_KILL_READY");
    let role = std::env::var("MIXED_KILL_ROLE").expect("MIXED_KILL_ROLE");
    let db = both_sync(Path::new(&path));
    match role.as_str() {
        "orphan" => {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            mixed_insert(&session, "orphan", "dead");
            db.wal()
                .expect("WAL attached")
                .sync()
                .expect("uncommitted mixed insert must be durable before kill");
            std::fs::write(&ready, b"ready").expect("ready file");
            std::mem::forget(session);
            std::mem::forget(db);
        }
        "live" => {
            assert_eq!(gql_count(&db), 0, "orphaned LPG must not appear");
            assert_eq!(sparql_count(&db), 0, "orphaned RDF must not appear");
            let mut session = db.session();
            session.begin_transaction().unwrap();
            mixed_insert(&session, "live", "ok");
            session.commit().unwrap();
            db.wal()
                .expect("WAL attached")
                .sync()
                .expect("acknowledged mixed commit must be durable before second kill");
            std::fs::write(&ready, b"ready").expect("ready file");
            std::mem::forget(session);
            std::mem::forget(db);
        }
        other => panic!("unknown MIXED_KILL_ROLE {other}"),
    }
    std::thread::park();
}

#[test]
fn mixed_orphaned_tx_not_authenticated_by_later_commit() {
    if std::env::var("MIXED_KILL_ROLE").is_ok() {
        two_crash_child_main();
        return;
    }

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("mixed_orphan.grafeo");
    let ready1 = dir.path().join("ready1");
    let ready2 = dir.path().join("ready2");

    let child = spawn_role(
        "mixed_orphaned_tx_not_authenticated_by_later_commit",
        "orphan",
        &path,
        &ready1,
    );
    wait_ready(&ready1);
    kill_wait(child);

    let child = spawn_role(
        "mixed_orphaned_tx_not_authenticated_by_later_commit",
        "live",
        &path,
        &ready2,
    );
    wait_ready(&ready2);
    kill_wait(child);
    assert!(
        sidecar_wal_path(&path).exists(),
        "second kill must leave the sidecar WAL (no close/checkpoint)"
    );

    let db = both_sync(&path);
    assert_eq!(gql_count(&db), 1, "only the acknowledged LPG commit");
    assert_eq!(sparql_count(&db), 1, "only the acknowledged RDF commit");
    let names = db
        .session()
        .execute("MATCH (n:Person) RETURN n.name")
        .unwrap();
    assert_eq!(
        names.rows()[0][0],
        grafeo_common::types::Value::String("ok".into())
    );
    assert_eq!(
        db.execute_sparql("SELECT ?o WHERE { <http://ex.org/orphan> ?p ?o }")
            .unwrap()
            .row_count(),
        0
    );
    assert_eq!(
        db.execute_sparql("SELECT ?o WHERE { <http://ex.org/live> ?p ?o }")
            .unwrap()
            .row_count(),
        1
    );
}

/// Child panics at `commit:after_lpg_before_rdf` via `GRAFEO_CRASH_NAMED`.
///
/// `catch_unwind` + `mem::forget` so `Drop` cannot checkpoint a torn commit
/// (LPG applied, RDF not yet). Then `abort` so the parent sees a crash exit.
#[cfg(feature = "testing-crash-injection")]
fn failpoint_child_main() {
    use std::panic::AssertUnwindSafe;

    let path = std::env::var("MIXED_KILL_PATH").expect("MIXED_KILL_PATH");
    let db = both_sync(Path::new(&path));
    let mut session = db.session();
    session.begin_transaction().unwrap();
    mixed_insert(&session, "torn", "alix");
    let crashed = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = session.commit();
    }));
    assert!(crashed.is_err(), "commit must hit after_lpg_before_rdf");
    std::mem::forget(session);
    std::mem::forget(db);
    std::process::abort();
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn subprocess_failpoint_after_lpg_before_rdf_recovers_both() {
    if std::env::var("MIXED_KILL_ROLE").as_deref() == Ok("failpoint") {
        failpoint_child_main();
        return;
    }

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("mixed_failpoint.grafeo");
    let exe = std::env::current_exe().expect("current_exe");
    let status = std::process::Command::new(&exe)
        .arg("subprocess_failpoint_after_lpg_before_rdf_recovers_both")
        .arg("--exact")
        .env("MIXED_KILL_ROLE", "failpoint")
        .env("MIXED_KILL_PATH", &path)
        .env("GRAFEO_CRASH_NAMED", "commit:after_lpg_before_rdf")
        .status()
        .expect("spawn failpoint child");
    assert!(
        !status.success(),
        "child must crash at after_lpg_before_rdf, got {status}"
    );
    assert!(
        sidecar_wal_path(&path).exists() || path.exists(),
        "failpoint child left no database files"
    );

    let db = both_sync(&path);
    assert_eq!(
        gql_count(&db),
        1,
        "LPG must recover from WAL after subprocess failpoint"
    );
    assert_eq!(
        sparql_count(&db),
        1,
        "RDF must recover with LPG after subprocess failpoint"
    );
}

/// Interrupted-checkpoint sequence in one aborting child:
/// Sync Both, commit, checkpoint, second commit, named failpoint on the
/// next checkpoint, `forget` + `abort` so Drop cannot install a torn file.
///
/// `GRAFEO_CRASH_NAMED` cannot be set at spawn: the first checkpoint uses
/// the same `write_sections:*` sites. Arm the named failpoint after gen-1
/// via thread-local `enable_crash_named`.
#[cfg(feature = "testing-crash-injection")]
fn snapshot_install_child_main() {
    use std::panic::AssertUnwindSafe;

    use grafeo_common::testing::crash::enable_crash_named;

    let path = std::env::var("MIXED_KILL_PATH").expect("MIXED_KILL_PATH");
    let point = std::env::var("MIXED_KILL_CRASH").expect("MIXED_KILL_CRASH");
    let named: &'static str = match point.as_str() {
        "write_sections:after_data" => "write_sections:after_data",
        "write_sections:after_fsync" => "write_sections:after_fsync",
        "checkpoint:after_snapshot_before_wal_retire" => {
            "checkpoint:after_snapshot_before_wal_retire"
        }
        other => panic!("unknown MIXED_KILL_CRASH {other}"),
    };
    let db = both_sync(Path::new(&path));
    db.create_node(&["First"]);
    db.wal_checkpoint().unwrap();
    db.create_node(&["Second"]);
    db.wal()
        .expect("WAL")
        .sync()
        .expect("second generation must be durable before crash");
    enable_crash_named(named);
    let crashed = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = db.wal_checkpoint();
    }));
    assert!(
        crashed.is_err(),
        "child must hit the named snapshot-install failpoint {named}"
    );
    std::mem::forget(db);
    std::process::abort();
}

#[cfg(feature = "testing-crash-injection")]
fn run_snapshot_install_kill(point: &str, test_name: &str) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("install_kill.grafeo");
    let exe = std::env::current_exe().expect("current_exe");
    let status = std::process::Command::new(&exe)
        .arg(test_name)
        .arg("--exact")
        .env("MIXED_KILL_ROLE", "snapshot-install")
        .env("MIXED_KILL_PATH", &path)
        .env("MIXED_KILL_CRASH", point)
        .status()
        .expect("spawn snapshot-install child");
    assert!(
        !status.success(),
        "child must crash at {point}, got {status}"
    );
    assert!(
        path.exists(),
        "child left no primary file after crash at {point}"
    );
    assert!(
        sidecar_wal_path(&path).exists(),
        "crash at {point} must not retire the sidecar WAL"
    );
    let db = both_sync(&path);
    assert_eq!(
        gql_count(&db),
        2,
        "last-good snapshot plus retained WAL must recover both generations after {point}"
    );
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn subprocess_snapshot_install_after_data_recovers_both() {
    if std::env::var("MIXED_KILL_ROLE").as_deref() == Ok("snapshot-install") {
        snapshot_install_child_main();
        return;
    }
    run_snapshot_install_kill(
        "write_sections:after_data",
        "subprocess_snapshot_install_after_data_recovers_both",
    );
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn subprocess_snapshot_install_after_fsync_keeps_previous() {
    if std::env::var("MIXED_KILL_ROLE").as_deref() == Ok("snapshot-install") {
        snapshot_install_child_main();
        return;
    }
    run_snapshot_install_kill(
        "write_sections:after_fsync",
        "subprocess_snapshot_install_after_fsync_keeps_previous",
    );
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn subprocess_checkpoint_after_snapshot_before_wal_retire_recovers() {
    if std::env::var("MIXED_KILL_ROLE").as_deref() == Ok("snapshot-install") {
        snapshot_install_child_main();
        return;
    }
    run_snapshot_install_kill(
        "checkpoint:after_snapshot_before_wal_retire",
        "subprocess_checkpoint_after_snapshot_before_wal_retire_recovers",
    );
}
