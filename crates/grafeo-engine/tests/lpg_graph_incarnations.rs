//! Native graph lifetime acceptance through the managed persistence callers.
#![cfg(all(
    feature = "lpg",
    feature = "wal",
    feature = "grafeo-file",
    feature = "gql"
))]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::types::{GraphIncarnationId, GraphPath};
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn root(db: &GrafeoDB) -> Arc<LpgStore> {
    Arc::clone(grafeo_engine::database::testing::root_lpg_store(db))
}

fn coordinates(db: &GrafeoDB) -> TestResult<(BTreeMap<GraphPath, GraphIncarnationId>, u64)> {
    let root = root(db);
    let rows = LpgStoreSection::new(Arc::clone(&root))
        .capture_graphs()?
        .into_iter()
        .map(|(path, store)| (path, store.graph_incarnation_id()))
        .collect();
    Ok((rows, root.next_graph_incarnation_id()))
}

fn path(parts: &[&str]) -> TestResult<GraphPath> {
    Ok(GraphPath::from_components(parts)?)
}

fn populate(db: &GrafeoDB) -> TestResult {
    for parts in [&[""][..], &["a"], &["a", "b"], &["a/b"]] {
        assert!(db.create_graph_path(&path(parts)?)?);
    }
    let before = coordinates(db)?;
    assert_eq!(before.0.values().copied().collect::<BTreeSet<_>>().len(), 5);
    assert_eq!(
        before.0[&GraphPath::root()],
        GraphIncarnationId::DEFAULT_GRAPH
    );
    let retained = root(db).graph("a").ok_or("missing a")?;
    assert!(db.drop_graph_path(&path(&["a"])?)?);
    assert!(db.create_graph_path(&path(&["a"])?)?);
    assert!(db.create_graph_path(&path(&["a", "b"])?)?);
    let after = coordinates(db)?;
    for parts in [&["a"][..], &["a", "b"]] {
        assert!(after.0[&path(parts)?] > before.0[&path(parts)?]);
    }
    assert_eq!(retained.graph_incarnation_id(), before.0[&path(&["a"])?]);
    assert_eq!(after.0[&path(&["a/b"])?], before.0[&path(&["a/b"])?]);
    Ok(())
}

#[test]
fn exact_snapshot_memory_container_and_live_restore_preserve_lifetimes() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    populate(&db)?;
    let expected = coordinates(&db)?;
    let snapshot = db.export_snapshot()?;
    let imported = GrafeoDB::import_snapshot(&snapshot)?;
    assert_eq!(coordinates(&imported)?, expected);
    assert_eq!(coordinates(&db.to_memory()?)?, expected);
    let target = GrafeoDB::new_in_memory();
    target.create_graph("old")?;
    let retained = root(&target).graph("old").ok_or("missing old")?;
    target.restore_snapshot(&snapshot)?;
    assert_eq!(coordinates(&target)?, expected);
    assert!(!root(&target).install_graph_if_absent("old", retained));
    let dir = tempfile::tempdir()?;
    let saved = dir.path().join("exact.grafeo");
    db.save(&saved)?;
    for _ in 0..2 {
        let copy = GrafeoDB::open(&saved)?;
        assert_eq!(coordinates(&copy)?, expected);
        copy.close()?;
    }
    Ok(())
}

#[test]
fn rollback_and_savepoint_do_not_publish_or_reuse_reserved_lifetimes() -> TestResult {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction()?;
    session.create_graph_path(&path(&["aborted"])?)?;
    let reserved = root(&db).next_graph_incarnation_id();
    assert!(!coordinates(&db)?.0.contains_key(&path(&["aborted"])?));
    session.rollback()?;
    session.begin_transaction()?;
    session.create_graph_path(&path(&["kept"])?)?;
    session.savepoint("keep")?;
    session.create_graph_path(&path(&["discarded"])?)?;
    let floor = root(&db).next_graph_incarnation_id();
    session.rollback_to_savepoint("keep")?;
    session.commit()?;
    let (rows, next) = coordinates(&db)?;
    assert_eq!(next, floor);
    assert!(rows[&path(&["kept"])?].as_u64() >= reserved);
    assert!(!rows.contains_key(&path(&["discarded"])?));
    db.create_graph("after")?;
    assert_eq!(coordinates(&db)?.0[&path(&["after"])?].as_u64(), floor);
    Ok(())
}

fn sidecar(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(".wal");
    PathBuf::from(value)
}

fn process_image(source: &Path, destination: &Path) -> TestResult {
    std::fs::copy(source, destination)?;
    let source = sidecar(source);
    let destination = sidecar(destination);
    std::fs::create_dir(&destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        assert!(entry.file_type()?.is_file());
        std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
    }
    Ok(())
}

#[test]
fn checkpoint_and_committed_tail_keep_retired_lifetimes_after_two_reopens() -> TestResult {
    let dir = tempfile::tempdir()?;
    let live = dir.path().join("live.grafeo");
    let image = dir.path().join("image.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&live)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )?;
    populate(&db)?;
    db.wal_checkpoint()?;
    db.create_graph("tail")?;
    let retired = root(&db)
        .graph("tail")
        .ok_or("missing tail")?
        .graph_incarnation_id();
    db.drop_graph("tail")?;
    db.session().execute("CREATE SCHEMA identity_schema")?;
    let expected = coordinates(&db)?;
    // Capture Sync-acknowledged disk bytes while the live owner is still open;
    // its eventual close cannot put the tail into this copied checkpoint.
    process_image(&live, &image)?;
    db.close()?;
    for pass in 0..2 {
        let reopened = GrafeoDB::open(&image)?;
        assert_eq!(coordinates(&reopened)?, expected);
        assert!(root(&reopened).next_graph_incarnation_id() > retired.as_u64());
        if pass == 1 {
            reopened.create_graph("tail")?;
            assert_eq!(
                root(&reopened)
                    .graph("tail")
                    .ok_or("tail not recreated")?
                    .graph_incarnation_id()
                    .as_u64(),
                expected.1
            );
        }
        reopened.close()?;
    }
    Ok(())
}

#[test]
fn wal_replays_out_of_order_reservations_but_rejects_reused_or_wrong_lifetimes() -> TestResult {
    use grafeo_common::types::TransactionId;
    use grafeo_storage::wal::{LpgWal, WalRecord};
    for kind in ["out_of_order", "reuse", "wrong_drop", "below_checkpoint"] {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("lifetimes.grafeo");
        let db = GrafeoDB::with_config(
            Config::persistent(&file).with_wal_durability(DurabilityMode::Sync),
        )?;
        if kind == "below_checkpoint" {
            db.create_graph("retired")?;
            db.drop_graph("retired")?;
        }
        db.wal_checkpoint()?;
        let epoch = db.current_epoch().next();
        db.close()?;
        let transaction_id = TransactionId::new(8001);
        let a = path(&["a"])?;
        let b = path(&["b"])?;
        let create = |graph, id| WalRecord::CreateLpgGraph {
            graph,
            incarnation: GraphIncarnationId::new(id),
            transaction_id,
        };
        let drop = |id| WalRecord::DropLpgGraph {
            graph: a.clone(),
            incarnation: GraphIncarnationId::new(id),
            transaction_id,
        };
        let records = match kind {
            "out_of_order" => vec![create(a.clone(), 7), create(b.clone(), 3), drop(7)],
            "reuse" => vec![create(a.clone(), 1), drop(1), create(b.clone(), 1)],
            "wrong_drop" => vec![create(a.clone(), 1), drop(9)],
            _ => vec![create(a.clone(), 1)],
        };
        let wal = LpgWal::open(sidecar(&file))?;
        for record in records {
            wal.log(&record)?;
        }
        wal.log(&WalRecord::Committed {
            transaction_id,
            epoch,
        })?;
        wal.close()?;
        if kind == "out_of_order" {
            for _ in 0..2 {
                let db = GrafeoDB::open(&file)?;
                assert_eq!(coordinates(&db)?.0[&b].as_u64(), 3);
                assert_eq!(coordinates(&db)?.1, 8);
                assert!(!coordinates(&db)?.0.contains_key(&a));
                db.close()?;
            }
        } else {
            let before = std::fs::read(&file)?;
            let error = GrafeoDB::open(&file)
                .err()
                .ok_or("invalid lifetime was admitted")?;
            assert!(
                error.to_string().contains("invalid WAL graph incarnation"),
                "{error}"
            );
            assert_eq!(std::fs::read(&file)?, before);
        }
    }
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_and_recompact_keep_native_lifetimes_and_allocator_authority() -> TestResult {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db)?;
    let first = db.create_node(&["Before"]);
    assert!(first.is_valid());
    let expected = coordinates(&db)?;
    db.compact()?;
    assert_eq!(coordinates(&db)?, expected);
    assert!(db.create_node(&["After"]).is_valid());
    db.compact()?;
    assert_eq!(coordinates(&db)?, expected);
    db.create_graph("after_compact")?;
    assert_eq!(
        coordinates(&db)?.0[&path(&["after_compact"])?].as_u64(),
        expected.1
    );
    assert_eq!(
        coordinates(&GrafeoDB::import_snapshot(&db.export_snapshot()?)?)?,
        coordinates(&db)?
    );
    Ok(())
}
