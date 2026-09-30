//! Mixed-model publication protocol regression tests.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,triple-store,sparql,wal,grafeo-file,testing-crash-injection" \
//!   --test mixed_publish -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "sparql",
    feature = "triple-store",
    feature = "wal",
    feature = "grafeo-file"
))]

#[cfg(feature = "testing-crash-injection")]
use grafeo_common::types::Value;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

fn sidecar_wal_dir(path: &std::path::Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

fn copy_tree(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), to).unwrap();
        }
    }
}

fn snapshot_db(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::copy(src, dst).unwrap();
    let wal = sidecar_wal_dir(src);
    if wal.exists() {
        copy_tree(&wal, &sidecar_wal_dir(dst));
    }
}

fn both_sync(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Both)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open Both")
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

/// Stored GraphModel::Both is restored by open() and rejects a pinned RDF open.
#[test]
fn graph_model_both_persists_and_rejects_mismatch() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("both_meta.grafeo");
    {
        let db = both_sync(&path);
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
            .unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.config().graph_model,
        GraphModel::Both,
        "open() must adopt stored Both"
    );
    assert_eq!(gql_count(&db), 1);
    assert_eq!(sparql_count(&db), 1);

    let err = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Rdf)
            .with_wal_durability(DurabilityMode::Sync),
    );
    assert!(
        err.is_err(),
        "pinned RDF open of a Both file must fail, got {}",
        err.err().unwrap()
    );
}

fn stage_mixed_writer(session: &grafeo_engine::Session, id: usize) {
    session
        .execute(&format!("INSERT (:Person {{id: {id}}})"))
        .unwrap();
    session
        .execute_sparql(&format!(
            r#"INSERT DATA {{ <http://ex.org/s{id}> <http://ex.org/p> "{id}" }}"#,
        ))
        .unwrap();
}

fn commit_mixed_writer_with_retry(db: &GrafeoDB, id: usize) -> grafeo_common::types::EpochId {
    let mut session = db.session();
    for _ in 0..16 {
        session.begin_transaction().unwrap();
        stage_mixed_writer(&session, id);
        match session.commit() {
            Ok(epoch) => return epoch,
            Err(error) => {
                assert_eq!(
                    error.error_code(),
                    grafeo_common::utils::error::ErrorCode::TransactionConflict,
                    "{error}"
                );
                assert!(
                    !session.in_transaction(),
                    "failed commit must abort both planes"
                );
                std::thread::yield_now();
            }
        }
    }
    panic!("eight mixed writers must finish within sixteen attempts per writer");
}

#[test]
fn mixed_conflict_discards_both_planes_and_retries_exactly_once() {
    use grafeo_common::utils::error::ErrorCode;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed-conflict.grafeo");
    let db = both_sync(&path);
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    stage_mixed_writer(&first, 0);
    stage_mixed_writer(&second, 1);
    let committed = first.commit().unwrap();
    let error = second.commit().unwrap_err();
    assert_eq!(error.error_code(), ErrorCode::TransactionConflict);
    assert!(error.error_code().is_retryable());
    assert!(!second.in_transaction());
    assert_eq!(db.current_epoch(), committed);
    assert_eq!(db.rdf_store_commit_epoch(), committed);
    assert_eq!(gql_count(&db), 1);
    assert_eq!(sparql_count(&db), 1);
    assert_eq!(
        db.execute("MATCH (n:Person {id: 1}) RETURN n")
            .unwrap()
            .row_count(),
        0
    );
    assert_eq!(
        db.execute_sparql("SELECT ?o WHERE { <http://ex.org/s1> <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        0
    );

    second.begin_transaction().unwrap();
    stage_mixed_writer(&second, 1);
    let retried = second.commit().unwrap();
    assert!(retried > committed);
    assert_eq!(gql_count(&db), 2);
    assert_eq!(sparql_count(&db), 2);
    drop(first);
    drop(second);
    db.close().unwrap();
    let reopened = both_sync(&path);
    assert_eq!(reopened.current_epoch(), retried);
    assert_eq!(reopened.rdf_store_commit_epoch(), retried);
    assert_eq!(gql_count(&reopened), 2);
    assert_eq!(sparql_count(&reopened), 2);
}

/// Concurrent mixed commits get unique epochs; a reader never sees LPG without RDF.
#[test]
fn concurrent_mixed_commits_and_readers_atomic() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let db = Arc::new(
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap(),
    );
    let n = 8usize;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut samples = 0u32;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let g = gql_count(&db);
                let s = i64::try_from(sparql_count(&db)).unwrap();
                assert!(g <= s, "LPG published without RDF: gql={g} sparql={s}");
                samples += 1;
            }
            samples
        })
    };
    let barrier = Arc::new(Barrier::new(n + 1));
    let writers: Vec<_> = (0..n)
        .map(|i| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                commit_mixed_writer_with_retry(&db, i)
            })
        })
        .collect();
    barrier.wait();
    let mut epochs: Vec<u64> = writers
        .into_iter()
        .map(|h| h.join().unwrap().as_u64())
        .collect();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = reader.join().unwrap();
    epochs.sort_unstable();
    let mut unique = epochs.clone();
    unique.dedup();
    assert_eq!(unique.len(), n, "mixed commit epochs must be unique");
    let max = *epochs.last().unwrap();
    assert_eq!(
        db.current_epoch().as_u64(),
        max,
        "LPG overlay epoch must be the max assigned epoch, not a later-applied earlier one"
    );
    assert_eq!(
        db.rdf_store_commit_epoch().as_u64(),
        max,
        "RDF store commit epoch must match the max assigned epoch"
    );
    assert_eq!(gql_count(&db), i64::try_from(n).unwrap());
    assert_eq!(sparql_count(&db), n);
}

/// DROP GRAPH existence survives close/reopen after commit.
#[test]
fn drop_graph_existence_recovers() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("drop_exist.grafeo");
    {
        let db = both_sync(&path);
        db.execute_sparql(
            r#"INSERT DATA { GRAPH <http://ex.org/g1> { <http://ex.org/s> <http://ex.org/p> "a" } }"#,
        )
        .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql("DROP GRAPH <http://ex.org/g1>")
            .unwrap();
        session.commit().unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let created = db
        .session()
        .execute_sparql("CREATE GRAPH <http://ex.org/g1>")
        .is_ok();
    assert!(created, "recovered DROP must leave the named graph absent");
}

/// DROP ALL and MOVE survive close/reopen; repeated DROP does not resurrect.
#[test]
fn drop_all_and_move_recover() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("ddl.grafeo");
    {
        let db = both_sync(&path);
        db.execute_sparql(
            r#"INSERT DATA {
                GRAPH <http://ex.org/g1> { <http://ex.org/s1> <http://ex.org/p> "a" }
                GRAPH <http://ex.org/g2> { <http://ex.org/s2> <http://ex.org/p> "b" }
            }"#,
        )
        .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql("MOVE <http://ex.org/g1> TO <http://ex.org/g3>")
            .unwrap();
        session.commit().unwrap();
        session.begin_transaction().unwrap();
        session.execute_sparql("DROP ALL").unwrap();
        session.commit().unwrap();
        session.begin_transaction().unwrap();
        session
            .execute_sparql("DROP SILENT GRAPH <http://ex.org/g3>")
            .unwrap();
        session.commit().unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(sparql_count(&db), 0, "DROP ALL must survive reopen");
    assert!(
        db.session()
            .execute_sparql("CREATE GRAPH <http://ex.org/g1>")
            .is_ok(),
        "source of MOVE must be absent after reopen"
    );
    assert!(
        db.session()
            .execute_sparql("CREATE GRAPH <http://ex.org/g3>")
            .is_ok(),
        "repeated DROP must not resurrect g3"
    );
}

/// Named-graph Session LPG writes replay into that graph, not default.
#[test]
fn named_graph_session_lpg_not_default() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_lpg.grafeo");
    {
        let db = both_sync(&path);
        let session = db.session();
        session
            .execute("CREATE GRAPH g1")
            .expect("create named LPG graph");
        session
            .use_graph_path(
                &grafeo_common::types::GraphPath::from_components(&["g1"])
                    .expect("literal graph path"),
            )
            .expect("select existing graph");
        let id = session.create_node(&["Person"]);
        assert!(id.is_valid());
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        gql_count(&db),
        0,
        "named-graph node must not appear in the default graph after reopen"
    );
    let session = db.session();
    session
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["g1"]).expect("literal graph path"),
        )
        .expect("select existing graph");
    let n = session.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 1, "named-graph node must replay into g1");
}

/// Concurrent abort must not steal another session's named-graph LPG commit.
#[test]
fn concurrent_named_graph_abort_does_not_move_survivor() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_abort.grafeo");
    {
        let db = Arc::new(both_sync(&path));
        db.session().execute("CREATE GRAPH g1").unwrap();
        db.session().execute("CREATE GRAPH g2").unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let dead = {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut session = db.session();
                session.execute("USE GRAPH g1").unwrap();
                session.begin_transaction().unwrap();
                session.create_node(&["Dead"]);
                barrier.wait();
                session.rollback().unwrap();
            })
        };
        let live = {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut session = db.session();
                session.execute("USE GRAPH g2").unwrap();
                session.begin_transaction().unwrap();
                let id = session.create_node(&["Live"]);
                barrier.wait();
                session.commit().unwrap();
                id
            })
        };
        dead.join().unwrap();
        let id = live.join().unwrap();
        assert!(id.is_valid());
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(gql_count(&db), 0, "neither named graph is the default");
    let s = db.session();
    s.execute("USE GRAPH g1").unwrap();
    let n1 = s.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    s.execute("USE GRAPH g2").unwrap();
    let n2 = s.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n1, 0, "aborted g1 node must not recover");
    assert_eq!(n2, 1, "committed g2 node must recover into g2");
}

/// DROP NAMED must not clear the default graph.
#[test]
fn drop_named_leaves_default() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_vs_all.grafeo");
    {
        let db = both_sync(&path);
        db.execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/def> <http://ex.org/p> "d" .
                GRAPH <http://ex.org/g1> { <http://ex.org/s> <http://ex.org/p> "n" }
            }"#,
        )
        .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute_sparql("DROP NAMED").unwrap();
        session.commit().unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(sparql_count(&db), 1, "DROP NAMED must keep default triples");
    assert!(
        db.session()
            .execute_sparql("CREATE GRAPH <http://ex.org/g1>")
            .is_ok(),
        "DROP NAMED must drop named-graph identity"
    );
}

/// Crash between LPG apply and RDF apply recovers both from WAL Committed.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn crash_after_lpg_before_rdf_recovers_both() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_named};

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("torn_commit.grafeo");
    let copy = dir.path().join("torn_commit_copy.grafeo");
    let db = both_sync(&path);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session
        .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
        .unwrap();
    let crashed = with_crash_named("commit:after_lpg_before_rdf", || {
        let _ = session.commit();
    });
    assert!(
        matches!(crashed, CrashResult::Crashed),
        "commit must hit after_lpg_before_rdf"
    );
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = both_sync(&copy);
    assert_eq!(gql_count(&db), 1, "LPG must recover");
    assert_eq!(sparql_count(&db), 1, "RDF must recover with LPG");
}

/// LPG mutation WAL log failure returns Err and poisons immediately.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn lpg_mutation_wal_failure_returns_err_and_poisons() {
    use grafeo_common::testing::wal_failure::{
        disable_mutation_log_failure, enable_mutation_log_failure_once,
    };

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("lpg_poison.grafeo");
    let db = both_sync(&path);
    let session = db.session();
    enable_mutation_log_failure_once();
    let err = session.execute("INSERT (:Person {name: 'dead'})");
    disable_mutation_log_failure();
    assert!(
        err.is_err(),
        "LPG WAL log failure must return Err, got {err:?}"
    );
    assert!(
        db.is_durability_poisoned(),
        "LPG WAL log failure must poison immediately"
    );
}

/// After Sync mixed commit, crash during close must recover both models.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn mixed_crash_during_close_after_sync_commit_recovers_both() {
    use std::panic::AssertUnwindSafe;

    use grafeo_common::testing::crash::with_crash_at;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("mixed_crash.grafeo");
    {
        let db = both_sync(&path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
            .unwrap();
        session.commit().unwrap();
        let db = AssertUnwindSafe(db);
        let _ = with_crash_at(1, move || {
            let _ = db.close();
        });
    }
    let db = both_sync(&path);
    assert_eq!(gql_count(&db), 1, "LPG must recover after crash-on-close");
    assert_eq!(
        sparql_count(&db),
        1,
        "RDF must recover after crash-on-close"
    );
    assert_eq!(db.node_count(), 1, "no silent IRI≡node mirror on recovery");
}

fn gql_count_session(session: &grafeo_engine::Session) -> i64 {
    session.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0]
        .as_int64()
        .unwrap()
}

fn sparql_count_session(session: &grafeo_engine::Session) -> usize {
    session
        .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        .unwrap()
        .row_count()
}

/// A mixed snapshot holds one publication instant: GQL and SPARQL counts match
/// while concurrent mixed writers run.
#[test]
fn mixed_snapshot_gql_equals_sparql_under_writers() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let db = Arc::new(
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap(),
    );
    let n = 8usize;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let session = db.session();
            let mut samples = 0u32;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let _snap = session.snapshot().expect("snapshot");
                let g = gql_count_session(&session);
                let s = i64::try_from(sparql_count_session(&session)).unwrap();
                assert_eq!(
                    g, s,
                    "mixed snapshot must see the same committed instant: gql={g} sparql={s}"
                );
                samples += 1;
            }
            samples
        })
    };
    let barrier = Arc::new(Barrier::new(n + 1));
    let writers: Vec<_> = (0..n)
        .map(|i| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                commit_mixed_writer_with_retry(&db, i)
            })
        })
        .collect();
    barrier.wait();
    for h in writers {
        h.join().unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(reader.join().unwrap() > 0);
    assert_eq!(gql_count(&db), i64::try_from(n).unwrap());
    assert_eq!(sparql_count(&db), n);
}

/// Mixed snapshot blocks publication until it is dropped.
#[test]
fn mixed_snapshot_blocks_commit_until_drop() {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    let db = Arc::new(
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap(),
    );
    let reader = db.session();
    let snap = reader.snapshot().expect("snapshot");
    let db_w = Arc::clone(&db);
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        started_tx.send(()).ok();
        let mut session = db_w.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
            .unwrap();
        session.commit().unwrap();
        done_tx.send(()).ok();
    });
    started_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("writer started");
    std::thread::sleep(Duration::from_millis(80));
    assert!(
        done_rx.try_recv().is_err(),
        "mixed commit must wait for snapshot drop"
    );
    drop(snap);
    done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("commit completes after snapshot drop");
    handle.join().unwrap();
    assert_eq!(gql_count(&db), 1);
    assert_eq!(sparql_count(&db), 1);
}

/// Mutating while a mixed snapshot is held must fail, not deadlock.
#[test]
fn mixed_snapshot_rejects_mutation_on_same_thread() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
    let session = db.session();
    let _snap = session.snapshot().expect("snapshot");
    let err = session.execute("INSERT (:Person {name: 'x'})");
    assert!(
        err.is_err(),
        "mutate under snapshot must be Err, got {err:?}"
    );
}

/// Session CRUD without begin/close still durable: auto-commit writes Committed.
#[test]
fn session_crud_auto_commit_survives_no_close() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("crud.grafeo");
    let copy = dir.path().join("crud_copy.grafeo");
    let db = both_sync(&path);
    let session = db.session();
    let a = session.create_node(&["Person"]);
    let b = session.create_node(&["Person"]);
    let e = session.create_edge(a, b, "KNOWS");
    assert!(a.is_valid() && b.is_valid() && e.is_valid());
    db.wal().expect("WAL").sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(session);
    std::mem::forget(db);
    let db = both_sync(&copy);
    assert_eq!(gql_count(&db), 2, "auto-committed nodes survive no-close");
    assert_eq!(db.edge_count(), 1, "auto-committed edge survives no-close");
}

/// `GrafeoDB` one-shot CRUD is framed with `Committed`, so kill before close recovers.
#[test]
fn db_create_node_framed_survives_no_close() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("db_crud.grafeo");
    let copy = dir.path().join("db_crud_copy.grafeo");
    let db = both_sync(&path);
    let id = db.create_node(&["Person"]);
    assert!(id.is_valid());
    db.wal().expect("WAL").sync().unwrap();
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = both_sync(&copy);
    assert_eq!(
        gql_count(&db),
        1,
        "framed DB create_node must survive no-close"
    );
}

/// Checkpoint crash after the snapshot is written must leave WAL for recovery.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn checkpoint_crash_after_snapshot_keeps_wal() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_named};

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("ckpt.grafeo");
    let copy = dir.path().join("ckpt_copy.grafeo");
    let db = both_sync(&path);
    let a = db.create_node(&["Person"]);
    let b = db.create_node(&["Person"]);
    let edge = db.create_edge(a, b, "KNOWS");
    assert!(a.is_valid() && b.is_valid() && edge.is_valid());
    db.set_node_property(a, "version", Value::Int64(1))
        .expect("set node property");
    db.set_node_property(a, "version", Value::Int64(2))
        .expect("set node property");
    let history_before = db.get_node_property_history(a, "version");
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "v" }"#)
        .unwrap();
    let crashed = with_crash_named("checkpoint:after_snapshot_before_wal_retire", || {
        let _ = db.wal_checkpoint();
    });
    assert!(
        matches!(crashed, CrashResult::Crashed),
        "checkpoint must hit after_snapshot_before_wal_retire"
    );
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    let db = both_sync(&copy);
    assert_eq!(db.node_count(), 2, "snapshot nodes must not replay twice");
    assert_eq!(db.edge_count(), 1, "snapshot edge must not replay twice");
    assert_eq!(gql_count(&db), 2);
    let adjacency = db
        .session()
        .execute("MATCH (:Person)-[e:KNOWS]->(:Person) RETURN count(e)")
        .unwrap();
    assert_eq!(adjacency.rows()[0][0], Value::Int64(1));
    assert_eq!(
        db.get_node_property_history(a, "version"),
        history_before,
        "retained pre-boundary WAL must not duplicate or replace MVCC history"
    );
    assert_eq!(
        sparql_count(&db),
        1,
        "RDF snapshot prefix must not replay twice"
    );
}

/// Crash while installing a new snapshot must keep the previous container.
/// WAL still has post-checkpoint records, so both generations recover.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn interrupted_snapshot_install_recovers_from_previous_and_wal() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_named};

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("install.grafeo");
    let copy = dir.path().join("install_copy.grafeo");
    let db = both_sync(&path);
    db.create_node(&["First"]);
    db.wal_checkpoint().unwrap();
    db.create_node(&["Second"]);
    db.wal().expect("WAL").sync().unwrap();
    let crashed = with_crash_named("write_sections:after_data", || {
        let _ = db.wal_checkpoint();
    });
    assert!(
        matches!(crashed, CrashResult::Crashed),
        "second checkpoint must crash while writing the temp container"
    );
    snapshot_db(&path, &copy);
    std::mem::forget(db);
    assert!(
        sidecar_wal_dir(&copy).exists(),
        "interrupted second checkpoint must not retire the sidecar WAL"
    );
    let db = both_sync(&copy);
    assert_eq!(
        gql_count(&db),
        2,
        "last-good snapshot plus retained WAL must recover both nodes"
    );
}

/// Default-graph LPG and named-graph LPG recover independently (no cursor).
#[test]
fn default_and_named_lpg_recover_independently() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("graph_identity.grafeo");
    {
        let db = both_sync(&path);
        let session = db.session();
        session.execute("INSERT (:Default {name: 'root'})").unwrap();
        session.execute("CREATE GRAPH g1").unwrap();
        session.execute("USE GRAPH g1").unwrap();
        session.execute("INSERT (:Named {name: 'g'})").unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(gql_count(&db), 1, "default graph node must recover");
    let s = db.session();
    s.execute("USE GRAPH g1").unwrap();
    let n = s.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 1, "named graph node must recover into g1");
}

/// `CREATE GRAPH … AS COPY OF` WAL-logs dest contents, not just an empty graph.
#[test]
fn create_graph_as_copy_of_recovers_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("copy_of.grafeo");
    {
        let db = both_sync(&path);
        let session = db.session();
        session.execute("CREATE GRAPH src").unwrap();
        session.execute("USE GRAPH src").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("CREATE GRAPH dst AS COPY OF src").unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let s = db.session();
    s.execute("USE GRAPH dst").unwrap();
    let n = s.execute("MATCH (n:Person) RETURN n.name").unwrap();
    assert_eq!(n.rows().len(), 1, "copied node must survive reopen");
    assert_eq!(
        n.rows()[0][0],
        grafeo_common::types::Value::String("Alix".into())
    );
}
