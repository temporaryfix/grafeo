//! Transaction ids must be unique across retained WAL history.
//!
//! Two-crash sequence (no graceful `close()` / checkpoint):
//! uncommitted write → kill → reopen in a child → acknowledged commit →
//! kill again → recover from the retained WAL. The orphaned write stays absent.
//!
//! ```text
//! cargo test -p grafeo-engine --features "triple-store,sparql,wal,grafeo-file" \
//!   --test tid_high_water -- --test-threads=1
//! ```

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::process::Child;
use std::time::{Duration, Instant};

use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn persistent_rdf_sync(path: &Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Rdf)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open rdf db")
}

fn sidecar_wal_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    PathBuf::from(p)
}

fn count_s(db: &GrafeoDB, subject: &str) -> usize {
    db.execute_sparql(&format!("SELECT ?o WHERE {{ <{subject}> ?p ?o }}"))
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

fn spawn_role(role: &str, path: &Path, ready: &Path) -> Child {
    let exe = std::env::current_exe().expect("current_exe for kill-9 child");
    std::process::Command::new(&exe)
        .arg("orphaned_tid_not_authenticated_by_later_commit")
        .arg("--exact")
        .env("TID_ORPHAN_ROLE", role)
        .env("TID_ORPHAN_PATH", path)
        .env("TID_ORPHAN_READY", ready)
        .spawn()
        .expect("spawn child")
}

/// Role `orphan`: uncommitted insert, fsync WAL, ready, park (no Drop).
/// Role `live`: reopen, commit an unrelated triple, fsync WAL, ready, park
/// (no `close()`, so the sidecar WAL is not checkpointed away).
fn child_main() {
    let path = std::env::var("TID_ORPHAN_PATH").expect("TID_ORPHAN_PATH");
    let ready = std::env::var("TID_ORPHAN_READY").expect("TID_ORPHAN_READY");
    let role = std::env::var("TID_ORPHAN_ROLE").expect("TID_ORPHAN_ROLE");
    let db = persistent_rdf_sync(Path::new(&path));
    match role.as_str() {
        "orphan" => {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute_sparql(
                    r#"INSERT DATA { <http://ex.org/orphan> <http://ex.org/p> "dead" . }"#,
                )
                .unwrap();
            db.wal()
                .expect("WAL attached")
                .sync()
                .expect("uncommitted insert must be durable before kill");
            std::fs::write(&ready, b"ready").expect("ready file");
            std::mem::forget(session);
            std::mem::forget(db);
        }
        "live" => {
            assert_eq!(
                count_s(&db, "http://ex.org/orphan"),
                0,
                "uncommitted write must not appear after first kill"
            );
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute_sparql(r#"INSERT DATA { <http://ex.org/live> <http://ex.org/p> "ok" . }"#)
                .unwrap();
            session.commit().unwrap();
            db.wal()
                .expect("WAL attached")
                .sync()
                .expect("acknowledged commit must be durable before second kill");
            std::fs::write(&ready, b"ready").expect("ready file");
            std::mem::forget(session);
            std::mem::forget(db);
        }
        other => panic!("unknown TID_ORPHAN_ROLE {other}"),
    }
    std::thread::park();
}

/// Two hard kills; recovery is from the retained WAL, not a close() checkpoint.
#[test]
fn orphaned_tid_not_authenticated_by_later_commit() {
    if std::env::var("TID_ORPHAN_ROLE").is_ok() {
        child_main();
        return;
    }

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("tid_orphan.grafeo");
    let ready1 = dir.path().join("ready1");
    let ready2 = dir.path().join("ready2");

    let child = spawn_role("orphan", &path, &ready1);
    wait_ready(&ready1);
    kill_wait(child);
    assert!(
        sidecar_wal_path(&path).exists() || path.exists(),
        "first kill left no database files"
    );

    let child = spawn_role("live", &path, &ready2);
    wait_ready(&ready2);
    kill_wait(child);
    assert!(
        sidecar_wal_path(&path).exists(),
        "second kill must leave the sidecar WAL (no close/checkpoint)"
    );

    let db = persistent_rdf_sync(&path);
    assert_eq!(
        count_s(&db, "http://ex.org/orphan"),
        0,
        "crash-orphaned write must not be authenticated by a later commit recovered from WAL"
    );
    assert_eq!(
        count_s(&db, "http://ex.org/live"),
        1,
        "acknowledged commit must survive recovery from retained WAL"
    );
}
