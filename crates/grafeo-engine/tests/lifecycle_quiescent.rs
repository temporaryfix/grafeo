//! Close / checkpoint / backup must not persist an uncommitted Session mutation.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,wal,grafeo-file" --test lifecycle_quiescent -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB};
use std::time::Duration;

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

fn persistent(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
        .expect("open")
}

fn secret_count(db: &GrafeoDB) -> i64 {
    db.session()
        .execute("MATCH (n:Secret) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap()
}

fn begin_uncommitted(db: &GrafeoDB) -> grafeo_engine::Session {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Secret {name: 'uncommitted'})")
        .unwrap();
    db.wal().expect("WAL").sync().unwrap();
    session
}

fn assert_uncommitted_absent(path: &std::path::Path) {
    let db = persistent(path);
    assert_eq!(
        secret_count(&db),
        0,
        "uncommitted Session mutation must not survive reopen"
    );
}

#[test]
fn close_with_open_tx_does_not_persist_uncommitted() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("close_tx.grafeo");
    let copy = dir.path().join("close_tx_copy.grafeo");
    let db = persistent(&path);
    db.session()
        .execute("INSERT (:Person {name: 'committed'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    let session = begin_uncommitted(&db);
    let err = db.close();
    assert!(
        err.is_err(),
        "close with an explicit open transaction must not snapshot, got {err:?}"
    );
    snapshot_db(&path, &copy);
    std::mem::forget(session);
    std::mem::forget(db);
    assert_uncommitted_absent(&copy);
    let db = persistent(&copy);
    let n = db
        .session()
        .execute("MATCH (n:Person) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 1, "previously committed node must still recover");
}

#[test]
fn wal_checkpoint_with_open_tx_does_not_persist_uncommitted() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("ckpt_tx.grafeo");
    let copy = dir.path().join("ckpt_tx_copy.grafeo");
    let db = persistent(&path);
    db.session()
        .execute("INSERT (:Person {name: 'committed'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    let session = begin_uncommitted(&db);
    let err = db.wal_checkpoint();
    assert!(
        err.is_err(),
        "wal_checkpoint with an explicit open transaction must not snapshot, got {err:?}"
    );
    snapshot_db(&path, &copy);
    std::mem::forget(session);
    std::mem::forget(db);
    assert_uncommitted_absent(&copy);
}

#[test]
fn backup_full_with_open_tx_does_not_persist_uncommitted() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("backup_tx.grafeo");
    let copy = dir.path().join("backup_tx_copy.grafeo");
    let backup_dir = dir.path().join("backups");
    let db = persistent(&path);
    db.session()
        .execute("INSERT (:Person {name: 'committed'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    let session = begin_uncommitted(&db);
    let err = db.backup_full(&backup_dir);
    assert!(
        err.is_err(),
        "backup_full with an explicit open transaction must not snapshot, got {err:?}"
    );
    let backup_file = backup_dir.join("backup_full_0000.grafeo");
    assert!(
        !backup_file.exists(),
        "rejected backup must not write a restorable container"
    );
    snapshot_db(&path, &copy);
    std::mem::forget(session);
    std::mem::forget(db);
    assert_uncommitted_absent(&copy);
    let db = persistent(&copy);
    let n = db
        .session()
        .execute("MATCH (n:Person) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 1, "previously committed node must still recover");
}

#[test]
fn checkpoint_on_fresh_database_does_not_create_a_phantom_transaction() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("fresh_checkpoint.grafeo");
    let db = persistent(&path);

    db.wal_checkpoint().expect("fresh checkpoint");
    db.close()
        .expect("checkpoint metadata must not pin a synthetic active transaction");
}

#[test]
fn successful_close_is_terminal_for_existing_and_new_sessions() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("terminal_close.grafeo");
    let db = persistent(&path);
    let mut existing = db.session();

    db.close().expect("close");

    assert!(
        existing.begin_transaction().is_err(),
        "a session obtained before close must not begin new work"
    );
    assert!(
        existing.execute("INSERT (:AfterClose)").is_err(),
        "an existing session must reject mutations after close"
    );
    assert!(
        db.session().execute("MATCH (n) RETURN n").is_err(),
        "a session obtained after close must be unusable"
    );
}

#[test]
fn checkpoint_recovery_starts_after_the_exact_snapshot_boundary() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("exact_checkpoint.grafeo");
    let copy = dir.path().join("exact_checkpoint_copy.grafeo");
    let db = persistent(&path);

    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    let edge = db.create_edge(a, b, "R");
    assert!(a.is_valid() && b.is_valid() && edge.is_valid());
    db.set_node_property(a, "version", Value::Int64(1))
        .expect("set node property");
    db.set_node_property(a, "version", Value::Int64(2))
        .expect("set node property");
    let history_before = db.get_node_property_history(a, "version");

    db.wal_checkpoint().expect("durable snapshot boundary");
    snapshot_db(&path, &copy);
    std::mem::forget(db); // process loss after checkpoint, before close

    let reopened = persistent(&copy);
    assert_eq!(
        reopened.node_count(),
        2,
        "snapshot nodes must not be replayed twice"
    );
    assert_eq!(
        reopened.edge_count(),
        1,
        "snapshot edge must not be replayed twice"
    );
    let paths = reopened
        .session()
        .execute("MATCH (:A)-[e:R]->(:B) RETURN count(e)")
        .unwrap();
    assert_eq!(paths.rows()[0][0], Value::Int64(1));
    assert_eq!(
        reopened.get_node_property_history(a, "version"),
        history_before,
        "checkpoint recovery must preserve, not replace, the snapshot's property history"
    );
}

#[test]
fn close_joins_periodic_checkpoint_without_publication_lock_inversion() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("checkpoint_timer_close.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_wal_durability(DurabilityMode::Sync)
            .with_checkpoint_interval(Duration::from_millis(1)),
    )
    .expect("open with periodic checkpoint");

    db.session().execute("INSERT (:Timer)").unwrap();
    // Let the background thread enter at least one checkpoint cycle before
    // close asks it to stop and joins it.
    std::thread::sleep(Duration::from_millis(150));
    db.close()
        .expect("close must not join a timer while holding its publication lock");
}

#[cfg(feature = "compact-store")]
#[test]
fn periodic_checkpoint_cannot_flatten_a_layered_database() {
    use grafeo_common::storage::SectionType;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("layered_timer.grafeo");
    let copy = dir.path().join("layered_timer_crash_copy.grafeo");
    let mut db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_wal_durability(DurabilityMode::Sync)
            .with_checkpoint_interval(Duration::from_millis(1)),
    )
    .expect("open with periodic checkpoint");

    db.session().execute("INSERT (:Cold)").unwrap();
    db.compact()
        .expect("enter layered mode and stop flat timer");
    db.wal_checkpoint()
        .expect("write a topology-aware layered checkpoint");
    db.session().execute("INSERT (:Hot)").unwrap();
    db.wal().unwrap().sync().unwrap();

    // More than two timer polling periods: an erroneously retained timer would
    // replace the section directory with its stale flat-store snapshot.
    std::thread::sleep(Duration::from_millis(250));
    let directory = db
        .file_manager()
        .unwrap()
        .read_section_directory()
        .unwrap()
        .unwrap();
    assert!(
        directory.find(SectionType::CompactStore).is_some(),
        "the layered cold base must remain in the durable generation"
    );

    snapshot_db(&path, &copy);
    std::mem::forget(db); // crash image: do not let close repair the checkpoint
    let reopened = persistent(&copy);
    let result = reopened
        .session()
        .execute("MATCH (n) RETURN count(n)")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(2),
        "cold snapshot data and hot WAL tail must both survive"
    );
}
