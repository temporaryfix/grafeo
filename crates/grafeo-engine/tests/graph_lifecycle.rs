//! Transactional LPG graph lifecycle and WAL-backed mutation chokepoint.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,wal,grafeo-file" --test graph_lifecycle -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_common::types::Value;
use grafeo_engine::{
    Config, DurabilityMode, GrafeoDB,
    transaction::{EntityId, IsolationLevel},
};

fn persistent(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
        .expect("open")
}

fn sidecar_wal_dir(path: &std::path::Path) -> std::path::PathBuf {
    let mut wal = path.as_os_str().to_owned();
    wal.push(".wal");
    std::path::PathBuf::from(wal)
}

fn copy_live_database(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::copy(src, dst).expect("copy live container");
    let src_wal = sidecar_wal_dir(src);
    if src_wal.exists() {
        let dst_wal = sidecar_wal_dir(dst);
        std::fs::create_dir_all(&dst_wal).expect("create copied WAL directory");
        for entry in std::fs::read_dir(src_wal).expect("read WAL directory") {
            let entry = entry.expect("WAL entry");
            if entry.path().is_file() {
                std::fs::copy(entry.path(), dst_wal.join(entry.file_name()))
                    .expect("copy WAL segment");
            }
        }
    }
}

#[test]
fn wal_graph_store_mut_is_none() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = persistent(&dir.path().join("raw.grafeo"));
    assert!(
        db.graph_store_mut().is_none(),
        "WAL-backed databases must not expose raw GraphStoreMut"
    );
}

#[test]
fn create_graph_rollback_removes_graph() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE GRAPH g1").unwrap();
    session.execute("USE GRAPH g1").unwrap();
    session.execute("INSERT (:Person {name: 'temp'})").unwrap();
    session.rollback().unwrap();
    let err = session.execute("USE GRAPH g1");
    assert!(
        err.is_err(),
        "rolled-back CREATE GRAPH must not leave the graph, got {err:?}"
    );
}

#[test]
fn uncommitted_create_graph_is_session_private() {
    let db = GrafeoDB::new_in_memory();
    let owner = db.session();
    let observer = db.session();

    owner.execute("START TRANSACTION").unwrap();
    owner.execute("CREATE GRAPH hidden").unwrap();
    owner.execute("USE GRAPH hidden").unwrap();
    owner.execute("INSERT (:Private {value: 1})").unwrap();

    assert!(
        !db.list_graphs().contains(&"hidden".to_string()),
        "an uncommitted graph must not appear in the shared catalog"
    );
    assert!(
        observer.execute("USE GRAPH hidden").is_err(),
        "another session must not observe an uncommitted graph"
    );

    owner.execute("COMMIT").unwrap();
    observer.execute("USE GRAPH hidden").unwrap();
    let result = observer
        .execute("MATCH (n:Private) RETURN n.value")
        .unwrap();
    assert_eq!(result.row_count(), 1);
}

#[test]
fn create_drop_rollback_leaves_no_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("START TRANSACTION").unwrap();
    session.execute("CREATE GRAPH transient").unwrap();
    session.execute("DROP GRAPH transient").unwrap();
    session.execute("ROLLBACK").unwrap();

    assert!(!db.list_graphs().contains(&"transient".to_string()));
    assert!(session.execute("USE GRAPH transient").is_err());
}

#[test]
fn concurrent_drop_conflicts_with_pinned_writer_without_default_leak() {
    let db = GrafeoDB::new_in_memory();
    db.create_graph("shared").unwrap();
    let dropper = db.session();
    let writer = db.session();

    dropper.execute("START TRANSACTION").unwrap();
    dropper.execute("DROP GRAPH shared").unwrap();

    writer.execute("START TRANSACTION").unwrap();
    writer.execute("USE GRAPH shared").unwrap();
    writer.execute("INSERT (:MustNotLeak {value: 1})").unwrap();

    dropper.execute("COMMIT").unwrap();
    assert!(
        writer.execute("COMMIT").is_err(),
        "a writer pinned to a concurrently dropped graph must conflict"
    );
    assert!(!db.list_graphs().contains(&"shared".to_string()));

    writer.execute("USE GRAPH default").unwrap();
    let count = writer
        .execute("MATCH (n:MustNotLeak) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(
        count, 0,
        "orphan cleanup must never target the default graph"
    );
}

#[test]
fn drop_graph_rollback_keeps_contents() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute("CREATE GRAPH g1").unwrap();
    session.execute("USE GRAPH g1").unwrap();
    session.execute("INSERT (:Person {name: 'keep'})").unwrap();
    session.begin_transaction().unwrap();
    session.execute("DROP GRAPH g1").unwrap();
    session.rollback().unwrap();
    session.execute("USE GRAPH g1").unwrap();
    let n = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
    assert_eq!(
        n.rows().len(),
        1,
        "DROP GRAPH rollback must restore contents"
    );
}

#[test]
fn missing_named_graph_does_not_write_default() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Root {name: 'default'})").unwrap();
    let ghost = grafeo_common::types::GraphPath::from_components(&["ghost"]).unwrap();
    session
        .use_graph_path(&ghost)
        .expect_err("missing native target is rejected");
    assert!(session.current_graph_path().components().is_empty());
    assert!(db.create_graph("ghost").unwrap());
    session.use_graph_path(&ghost).unwrap();
    assert!(db.drop_graph("ghost").expect("drop graph"));
    let missing_read = session.execute("MATCH (n:Root) RETURN count(n)").unwrap();
    assert_eq!(
        missing_read.rows()[0][0].as_int64().unwrap(),
        0,
        "a missing named-graph read must not fall back to the default graph"
    );
    let id = session.create_node(&["Leaked"]);
    assert!(
        !id.is_valid() || {
            session
                .use_graph_path(&grafeo_common::types::GraphPath::root())
                .unwrap();
            session
                .execute("MATCH (n:Leaked) RETURN count(n)")
                .unwrap()
                .rows()[0][0]
                .as_int64()
                .unwrap()
                == 0
        },
        "write against a missing named graph must not land in the default graph"
    );
    session
        .use_graph_path(&grafeo_common::types::GraphPath::root())
        .unwrap();
    let n = session
        .execute("MATCH (n:Root) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn stale_named_graph_handle_never_reads_default_after_concurrent_drop() {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("INSERT (:Root {name: 'default'})")
        .unwrap();
    db.create_graph("gone").unwrap();
    let stale = db.session();
    stale.execute("USE GRAPH gone").unwrap();

    assert!(db.drop_graph("gone").expect("drop graph"));
    let result = stale.execute("MATCH (n:Root) RETURN count(n)").unwrap();
    assert_eq!(result.rows()[0][0].as_int64().unwrap(), 0);
}

#[test]
fn db_create_graph_survives_no_close() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("g.grafeo");
    let copy = dir.path().join("g_copy.grafeo");
    let db = persistent(&path);
    db.create_graph("g1").unwrap();
    db.wal().unwrap().sync().unwrap();
    std::fs::copy(&path, &copy).unwrap();
    let wal = {
        let mut p = path.as_os_str().to_owned();
        p.push(".wal");
        std::path::PathBuf::from(p)
    };
    if wal.exists() {
        let mut dst = copy.as_os_str().to_owned();
        dst.push(".wal");
        let dst = std::path::PathBuf::from(dst);
        std::fs::create_dir_all(&dst).unwrap();
        for e in std::fs::read_dir(&wal).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), dst.join(e.file_name())).unwrap();
        }
    }
    std::mem::forget(db);
    let db = persistent(&copy);
    assert!(
        db.list_graphs()
            .iter()
            .any(|g| g == "g1" || g.ends_with("/g1")),
        "framed create_graph must survive no-close, graphs={:?}",
        db.list_graphs()
    );
}

#[test]
fn untouched_create_and_replacement_graph_epochs_match_live_and_recovered_commit_cut() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("graph_epochs.grafeo");
    let copy = dir.path().join("graph_epochs_copy.grafeo");
    let db = persistent(&path);
    let session = db.session();

    session.execute("START TRANSACTION").unwrap();
    session.execute("CREATE GRAPH epoch_target").unwrap();
    session.execute("COMMIT").unwrap();
    let create_epoch = db.current_epoch();
    let original = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("epoch_target")
        .expect("created graph must publish");
    assert_eq!(
        original.current_epoch(),
        create_epoch,
        "an untouched created graph must join its creating commit cut"
    );

    session.execute("START TRANSACTION").unwrap();
    session.execute("DROP GRAPH epoch_target").unwrap();
    session.execute("CREATE GRAPH epoch_target").unwrap();
    session.execute("COMMIT").unwrap();
    let replacement_epoch = db.current_epoch();
    let replacement = grafeo_engine::database::testing::root_lpg_store(&db)
        .graph("epoch_target")
        .expect("replacement graph must publish");
    assert!(
        !std::sync::Arc::ptr_eq(&original, &replacement),
        "DROP+CREATE must install a new graph incarnation"
    );
    assert_eq!(
        replacement.current_epoch(),
        replacement_epoch,
        "an untouched replacement graph must join its replacing commit cut"
    );

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(original);
    drop(replacement);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    let recovered_graph = grafeo_engine::database::testing::root_lpg_store(&recovered)
        .graph("epoch_target")
        .expect("replacement graph must recover");
    assert_eq!(recovered.current_epoch(), replacement_epoch);
    assert_eq!(
        recovered_graph.current_epoch(),
        replacement_epoch,
        "live and WAL-recovered graph incarnations must expose the same commit cut"
    );
}

#[test]
fn rolled_back_create_drop_is_not_recovered() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rollback.grafeo");
    {
        let db = persistent(&path);
        let session = db.session();
        session.execute("START TRANSACTION").unwrap();
        session.execute("CREATE GRAPH transient").unwrap();
        session.execute("DROP GRAPH transient").unwrap();
        session.execute("ROLLBACK").unwrap();
        db.close().unwrap();
    }

    let db = persistent(&path);
    assert!(!db.list_graphs().contains(&"transient".to_string()));
    assert!(db.session().execute("USE GRAPH transient").is_err());
}

#[test]
fn rollback_to_savepoint_restores_graph_lifecycle_live_and_after_recovery() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("savepoint_graphs.grafeo");
    let copy = dir.path().join("savepoint_graphs_copy.grafeo");
    let db = persistent(&path);
    db.create_graph("keep").unwrap();

    let session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session.execute("CREATE GRAPH transient").unwrap();
    session.execute("DROP GRAPH transient").unwrap();
    session.execute("DROP GRAPH keep").unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("COMMIT").unwrap();

    assert!(db.list_graphs().contains(&"keep".to_string()));
    assert!(!db.list_graphs().contains(&"transient".to_string()));
    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    assert!(recovered.list_graphs().contains(&"keep".to_string()));
    assert!(!recovered.list_graphs().contains(&"transient".to_string()));
}

#[test]
fn rollback_to_savepoint_restores_node_and_edge_deletes_and_can_repeat() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:N {id: 1})-[:R]->(:N {id: 2})")
        .unwrap();
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();

    session.execute("MATCH ()-[e:R]->() DELETE e").unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    let edges = session
        .execute("MATCH ()-[e:R]->() RETURN count(e)")
        .unwrap();
    assert_eq!(edges.rows()[0][0].as_int64(), Some(1));

    session
        .execute("MATCH (n:N {id: 1}) DETACH DELETE n")
        .unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("COMMIT").unwrap();

    let nodes = session.execute("MATCH (n:N) RETURN count(n)").unwrap();
    let edges = session
        .execute("MATCH ()-[e:R]->() RETURN count(e)")
        .unwrap();
    assert_eq!(nodes.rows()[0][0].as_int64(), Some(2));
    assert_eq!(edges.rows()[0][0].as_int64(), Some(1));
}

#[test]
fn rollback_to_savepoint_retains_conservative_conflict_footprint() {
    let db = GrafeoDB::new_in_memory();
    let setup = db.session();
    let target = setup
        .create_node_with_props(&["Account"], [("balance", Value::Int64(10))])
        .unwrap();

    let mut owner = db.session();
    owner
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    let owner_tid = owner.active_transaction_id().unwrap();
    owner.savepoint("stable").unwrap();
    owner
        .execute(&format!(
            "MATCH (n) WHERE id(n) = {} RETURN n.balance",
            target.as_u64()
        ))
        .unwrap();
    owner
        .execute(&format!(
            "MATCH (n) WHERE id(n) = {} SET n.balance = 20",
            target.as_u64()
        ))
        .unwrap();
    owner.rollback_to_savepoint("stable").unwrap();

    let manager = owner.transaction_manager_ref();
    assert!(
        manager
            .get_write_set(owner_tid)
            .unwrap()
            .contains(&EntityId::Node(target)),
        "savepoint rollback must conservatively retain the transaction's write footprint"
    );
    assert!(
        manager
            .read_set(owner_tid)
            .contains(&EntityId::Node(target)),
        "savepoint rollback must conservatively retain SIREAD state"
    );
    assert_eq!(
        owner
            .execute(&format!(
                "MATCH (n) WHERE id(n) = {} RETURN n.balance",
                target.as_u64()
            ))
            .unwrap()
            .rows()[0][0]
            .as_int64(),
        Some(10),
        "data state still rewinds exactly"
    );

    let mut contender = db.session();
    contender.begin_transaction().unwrap();
    let conflict = contender
        .execute(&format!(
            "MATCH (n) WHERE id(n) = {} SET n.balance = 30",
            target.as_u64()
        ))
        .unwrap_err();
    assert!(
        conflict.to_string().contains("Write-write conflict"),
        "retained footprint must keep first-writer-wins sound: {conflict}"
    );
    contender.rollback().unwrap();
    owner.rollback().unwrap();
}

#[cfg(feature = "compact-store")]
#[test]
fn rollback_to_savepoint_restores_compact_base_tombstones() {
    let mut db = GrafeoDB::new_in_memory();
    db.session()
        .execute("INSERT (:N {id: 1})-[:R]->(:N {id: 2})")
        .unwrap();
    db.compact().unwrap();

    let session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session
        .execute("MATCH (n:N {id: 1}) DETACH DELETE n")
        .unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("COMMIT").unwrap();

    let nodes = session.execute("MATCH (n:N) RETURN count(n)").unwrap();
    let edges = session
        .execute("MATCH ()-[e:R]->() RETURN count(e)")
        .unwrap();
    assert_eq!(nodes.rows()[0][0].as_int64(), Some(2));
    assert_eq!(edges.rows()[0][0].as_int64(), Some(1));
}

#[test]
fn duplicate_savepoint_name_is_rejected() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    let error = session.execute("SAVEPOINT stable").unwrap_err();
    assert!(error.to_string().contains("already exists"), "{error}");
    session.execute("ROLLBACK").unwrap();
}

#[test]
fn structural_savepoint_restore_failure_poison_blocks_commit() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Before {id: 1})").unwrap();
    let tid = session.active_transaction_id().unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session.execute("INSERT (:After {id: 2})").unwrap();

    // Deliberately violate the internal structural-prefix invariant after the
    // savepoint. This models a corrupt/misbehaving mutation layer without a
    // test-only production hook.
    let _ = grafeo_engine::database::testing::root_lpg_store(&db).take_pending_creates(tid);
    let error = session.execute("ROLLBACK TO SAVEPOINT stable").unwrap_err();
    assert!(
        error.to_string().contains("recorded durably") && error.to_string().contains("reopen"),
        "post-marker restore failure must be reported as recovery-required: {error}"
    );

    let commit_error = session.commit().unwrap_err();
    assert!(
        commit_error.to_string().contains("durability")
            || commit_error.to_string().contains("poison"),
        "a divergent live transaction must never commit: {commit_error}"
    );
    session.rollback().unwrap();
}

#[test]
fn graph_type_bindings_follow_typed_like_drop_and_savepoint_lifecycle() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Person (name STRING)")
        .unwrap();
    session
        .execute("CREATE NODE TYPE Animal (species STRING)")
        .unwrap();
    session
        .execute("CREATE GRAPH TYPE PeopleOnly (NODE TYPE Person)")
        .unwrap();

    // Full rollback must discard both the detached graph and its TYPED binding.
    session.execute("START TRANSACTION").unwrap();
    session
        .execute("CREATE GRAPH typed_temp TYPED PeopleOnly")
        .unwrap();
    session.execute("USE GRAPH typed_temp").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'forbidden'})")
            .is_err(),
        "a staged TYPED binding must constrain writes before commit"
    );
    session.execute("ROLLBACK").unwrap();
    session.execute("CREATE GRAPH typed_temp").unwrap();
    session.execute("USE GRAPH typed_temp").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'untyped'})")
            .is_ok(),
        "a rolled-back TYPED binding must not constrain a later untyped graph incarnation"
    );
    session.execute("DROP GRAPH typed_temp").unwrap();

    session
        .execute("CREATE GRAPH source TYPED PeopleOnly")
        .unwrap();
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session.execute("CREATE GRAPH liked LIKE source").unwrap();
    session.execute("USE GRAPH liked").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'forbidden'})")
            .is_err(),
        "LIKE must stage the source graph's binding"
    );
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("CREATE GRAPH liked").unwrap();
    session.execute("USE GRAPH liked").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'savepoint-restored'})")
            .is_ok(),
        "rollback to savepoint must discard the staged LIKE binding"
    );
    session.execute("COMMIT").unwrap();

    // DROP-to-savepoint restores the committed binding; committing the later
    // DROP removes it, so name reuse cannot inherit a stale catalog entry.
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT before_drop").unwrap();
    session.execute("DROP GRAPH source").unwrap();
    session
        .execute("ROLLBACK TO SAVEPOINT before_drop")
        .unwrap();
    session.execute("USE GRAPH source").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'still-forbidden'})")
            .is_err(),
        "rollback to savepoint must restore a dropped graph's committed binding"
    );
    session.execute("DROP GRAPH source").unwrap();
    session.execute("COMMIT").unwrap();

    session.execute("CREATE GRAPH source").unwrap();
    session.execute("USE GRAPH source").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'reused'})")
            .is_ok()
    );
}

#[test]
fn typed_and_like_bindings_survive_crash_recovery_with_savepoint_filtering() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("typed_create.grafeo");
    let copy = dir.path().join("typed_create_copy.grafeo");
    let db = persistent(&path);
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Person (name STRING)")
        .unwrap();
    session
        .execute("CREATE NODE TYPE Animal (species STRING)")
        .unwrap();
    session
        .execute("CREATE GRAPH TYPE PeopleOnly (NODE TYPE Person)")
        .unwrap();
    session
        .execute("CREATE GRAPH source TYPED PeopleOnly")
        .unwrap();

    // Recovery must discard the LIKE binding logged after this savepoint and
    // retain the later untyped incarnation committed under the same name.
    session.execute("START TRANSACTION").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session.execute("CREATE GRAPH liked LIKE source").unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("CREATE GRAPH liked").unwrap();
    session.execute("COMMIT").unwrap();

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    let session = recovered.session();
    session.execute("USE GRAPH source").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'forbidden'})")
            .is_err(),
        "recovery must restore the committed TYPED binding"
    );
    session.execute("USE GRAPH liked").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'untyped'})")
            .is_ok(),
        "rollback-to-savepoint WAL filtering must discard the staged LIKE binding"
    );
}

#[test]
fn dropped_graph_binding_does_not_survive_crash_recovery() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("typed_drop.grafeo");
    let copy = dir.path().join("typed_drop_copy.grafeo");

    // Put the original binding in the checkpoint, then crash after the DROP's
    // durable commit but before a newer checkpoint can replace it.
    {
        let db = persistent(&path);
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING)")
            .unwrap();
        session
            .execute("CREATE NODE TYPE Animal (species STRING)")
            .unwrap();
        session
            .execute("CREATE GRAPH TYPE PeopleOnly (NODE TYPE Person)")
            .unwrap();
        session
            .execute("CREATE GRAPH source TYPED PeopleOnly")
            .unwrap();
        drop(session);
        db.close().unwrap();
    }

    let db = persistent(&path);
    db.session().execute("DROP GRAPH source").unwrap();
    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    assert!(!recovered.list_graphs().contains(&"source".to_string()));
    let session = recovered.session();
    session.execute("CREATE GRAPH source").unwrap();
    session.execute("USE GRAPH source").unwrap();
    assert!(
        session
            .execute("INSERT (:Animal {species: 'reused'})")
            .is_ok(),
        "recovery must remove the checkpointed binding together with the committed DROP"
    );
}

#[test]
fn rolled_back_savepoint_data_and_nested_transaction_do_not_recover() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("savepoint_data.grafeo");
    let copy = dir.path().join("savepoint_data_copy.grafeo");
    let db = persistent(&path);
    let session = db.session();

    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Keep {id: 1})").unwrap();
    session.execute("SAVEPOINT stable").unwrap();
    session.execute("INSERT (:Discard {id: 2})").unwrap();
    session.execute("ROLLBACK TO SAVEPOINT stable").unwrap();
    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:NestedDiscard {id: 3})").unwrap();
    session.execute("ROLLBACK").unwrap();
    session.execute("COMMIT").unwrap();

    db.wal().unwrap().sync().unwrap();
    copy_live_database(&path, &copy);
    drop(session);
    std::mem::forget(db);

    let recovered = persistent(&copy);
    let result = recovered
        .session()
        .execute("MATCH (n) RETURN labels(n)")
        .unwrap();
    assert_eq!(result.row_count(), 1);
}
