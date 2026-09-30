//! G1 RDF crash / failpoint tests.
//!
//! After DurabilityMode::Sync success, crash without a clean checkpoint must
//! still recover committed RDF (default and named graphs). Crash before
//! TransactionCommit must not replay a partial SPARQL UPDATE.
//!
//! ```text
//! cargo test -p grafeo-engine --features "triple-store,sparql,wal,grafeo-file,testing-crash-injection" \
//!   --test g1_rdf_crash -- --test-threads=1
//! ```

#![cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]

use std::panic::AssertUnwindSafe;
use std::path::Path;

use grafeo_common::testing::crash::with_crash_at;
use grafeo_core::graph::rdf::{Term, Triple};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn persistent_rdf_sync(path: &Path) -> GrafeoDB {
    let config = Config::persistent(path)
        .with_graph_model(GraphModel::Rdf)
        .with_wal_durability(DurabilityMode::Sync);
    GrafeoDB::with_config(config).expect("open rdf db")
}

fn sidecar_wal_path(path: &Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

fn count_pred(db: &GrafeoDB, pred: &str) -> usize {
    db.execute_sparql(&format!("SELECT ?o WHERE {{ ?s <{pred}> ?o }}"))
        .unwrap()
        .row_count()
}

fn count_graph(db: &GrafeoDB, graph: &str) -> usize {
    db.execute_sparql(&format!(
        "SELECT ?s WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
    ))
    .unwrap()
    .row_count()
}

/// After Sync INSERT DATA (default + named GRAPH), crash during close.
/// Reopen must see both graphs — WAL, not only snapshot-on-close.
#[test]
fn crash_during_close_after_sync_sparql_insert_recovers() {
    for crash_point in 1..=8 {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("rdf_crash.grafeo");
        {
            let db = persistent_rdf_sync(&path);
            db.execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/s> <http://ex.org/name> "def" .
                    GRAPH <http://ex.org/claims> {
                        <http://ex.org/s> <http://ex.org/claim> "c" .
                    }
                }"#,
            )
            .unwrap();
            let db = AssertUnwindSafe(db);
            let _ = with_crash_at(crash_point, move || {
                let _ = db.close();
            });
        }
        let wal = sidecar_wal_path(&path);
        assert!(
            path.exists() || wal.exists(),
            "crash_point={crash_point}: neither file nor WAL"
        );
        let db = persistent_rdf_sync(&path);
        assert_eq!(
            count_pred(&db, "http://ex.org/name"),
            1,
            "crash_point={crash_point}: default graph lost"
        );
        assert_eq!(
            count_graph(&db, "http://ex.org/claims"),
            1,
            "crash_point={crash_point}: named GRAPH lost"
        );
    }
}

/// Two triples in one UPDATE: crash during WAL write before commit.
/// Recovery must not show exactly one of the two (partial tx).
#[test]
fn crash_mid_wal_does_not_replay_partial_insert_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("partial.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        let db = AssertUnwindSafe(db);
        let _ = with_crash_at(1, move || {
            let _ = db.execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/p> "1" .
                    <http://ex.org/b> <http://ex.org/p> "2" .
                }"#,
            );
        });
    }
    if path.exists() || sidecar_wal_path(&path).exists() {
        let db = persistent_rdf_sync(&path);
        let n = count_pred(&db, "http://ex.org/p");
        assert!(
            n == 0 || n == 2,
            "partial RDF transaction on reopen: {n} triples"
        );
    }
}

/// batch_insert_rdf after Sync + crash on close recovers via WAL.
#[test]
fn crash_during_close_after_batch_insert_recovers() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("batch_crash.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        let t = Triple::new(
            Term::iri("http://ex.org/s"),
            Term::iri("http://ex.org/p"),
            Term::literal("v"),
        );
        db.batch_insert_rdf([t]).unwrap();
        let db = AssertUnwindSafe(db);
        let _ = with_crash_at(1, move || {
            let _ = db.close();
        });
    }
    let db = persistent_rdf_sync(&path);
    assert_eq!(count_pred(&db, "http://ex.org/p"), 1);
}

/// COPY DEFAULT TO named, then crash on close: dest graph must recover.
#[test]
fn crash_during_close_after_copy_recovers_named_graph() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("copy_crash.grafeo");
    {
        let db = persistent_rdf_sync(&path);
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/name> "n" . }"#)
            .unwrap();
        db.execute_sparql("COPY DEFAULT TO <http://ex.org/g>")
            .unwrap();
        let db = AssertUnwindSafe(db);
        let _ = with_crash_at(1, move || {
            let _ = db.close();
        });
    }
    let db = persistent_rdf_sync(&path);
    assert_eq!(count_graph(&db, "http://ex.org/g"), 1);
}

/// Real abrupt termination after Sync acknowledgement, including alias no-ops.
#[test]
fn kill9_after_sync_ack_child() {
    if std::env::var("G1_RDF_KILL9_CHILD").ok().as_deref() == Some("1") {
        let path = std::env::var("G1_RDF_KILL9_PATH").expect("path");
        let ready = std::env::var("G1_RDF_KILL9_READY").expect("ready path");
        let db = persistent_rdf_sync(Path::new(&path));
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/name> "k" . }"#)
            .unwrap();
        db.execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "hello"@EN . GRAPH <urn:g> { <urn:s> <urn:p> "hello"@EN } }"#).unwrap();
        db.execute_sparql(r#"INSERT DATA { <urn:s> <urn:p> "hello"@en . GRAPH <urn:g> { <urn:s> <urn:p> "hello"@en } }"#).unwrap();
        std::fs::write(ready, b"sync insert acknowledged").expect("publish child readiness");
        std::mem::forget(db);
        std::thread::park();
        return;
    }

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("kill9.grafeo");
    let ready = dir.path().join("kill9.ready");
    let exe = std::env::current_exe().expect("kill-9 executable");
    let mut child = std::process::Command::new(&exe)
        .arg("kill9_after_sync_ack_child")
        .arg("--exact")
        .env("G1_RDF_KILL9_CHILD", "1")
        .env("G1_RDF_KILL9_PATH", &path)
        .env("G1_RDF_KILL9_READY", &ready)
        .spawn()
        .expect("spawn kill-9 child");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while std::time::Instant::now() < deadline {
        if ready.exists() {
            break;
        }
        if child.try_wait().expect("poll kill-9 child").is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if !ready.exists() {
        let _ = child.kill();
        let status = child.wait().expect("reap unready kill-9 child");
        panic!("kill-9 child did not acknowledge its Sync insert: {status}");
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        path.exists() || sidecar_wal_path(&path).exists(),
        "acknowledged child left no durable image"
    );
    let db = persistent_rdf_sync(&path);
    assert_eq!(
        count_pred(&db, "http://ex.org/name"),
        1,
        "kill -9 after Sync ack lost the RDF insert"
    );
    let canonical = Triple::new(
        Term::iri("urn:s"),
        Term::iri("urn:p"),
        Term::lang_literal("hello", "EN"),
    );
    assert_eq!(count_pred(&db, "urn:p"), 1);
    assert_eq!(count_graph(&db, "urn:g"), 1);
    assert!(
        db.rdf_store()
            .triples()
            .iter()
            .any(|t| t.as_ref() == &canonical)
    );
    assert_eq!(
        db.rdf_store().graph("urn:g").unwrap().triples().as_slice(),
        &[std::sync::Arc::new(canonical)]
    );
}
