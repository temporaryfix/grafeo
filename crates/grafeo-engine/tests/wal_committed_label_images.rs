//! Real commit writes must replay exactly the label history they published.

#![cfg(all(feature = "lpg", feature = "wal"))]

use grafeo_common::types::{EpochId, GraphPath, NodeId};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, config::StorageFormat};

#[derive(Clone, Copy, Debug)]
enum Case {
    Birth,
    ZeroWidth,
    Changed,
    NoOp,
    AbsentLabel,
    Empty,
    Deleted,
    DeletedEmpty,
    Rollback,
    Savepoint,
}

fn labels(db: &GrafeoDB, graph: Option<&str>, id: NodeId) -> Vec<(EpochId, Vec<String>)> {
    let named = graph.map(|name| {
        grafeo_engine::database::testing::root_lpg_store(db)
            .graph(name)
            .expect("named store")
    });
    let store = named
        .as_deref()
        .unwrap_or(grafeo_engine::database::testing::root_lpg_store(db));
    store
        .node_label_history(id)
        .into_iter()
        .map(|(epoch, labels)| {
            let mut labels: Vec<_> = labels.into_iter().map(|label| label.to_string()).collect();
            labels.sort_unstable();
            (epoch, labels)
        })
        .collect()
}

fn exercise(case: Case, graph: Option<&str>, compact: bool) {
    let temp = tempfile::tempdir().expect("temporary directory");
    let path = temp.path().join("labels");
    let config = Config::persistent(&path)
        .with_storage_format(StorageFormat::WalDirectory)
        .with_wal_durability(DurabilityMode::Sync);
    let mut db = GrafeoDB::with_config(config.clone()).expect("create database");
    if let Some(name) = graph {
        if name.is_empty() {
            // Empty names are legal exact storage coordinates, but the current
            // public CREATE GRAPH API rejects them. Recover that existing
            // coordinate first, then exercise real managed Session writes.
            use grafeo_common::types::TransactionId;
            use grafeo_storage::wal::{LpgWal, WalConfig, WalRecord};
            db.close().expect("close empty graph fixture");
            drop(db);
            let wal = LpgWal::with_config(path.join("wal"), WalConfig::default())
                .expect("open closed fixture WAL");
            let transaction_id = TransactionId::new(700_001);
            wal.log(&WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: GraphPath::from_components(&[name]).expect("checked literal graph path"),
                transaction_id,
            })
            .expect("record exact empty graph coordinate");
            wal.log(&WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(1),
            })
            .expect("commit empty graph fixture");
            wal.sync().expect("sync exact empty graph coordinate");
            wal.close().expect("close fixture WAL");
            drop(wal);
            db = GrafeoDB::with_config(config.clone()).expect("recover empty named graph");
            assert!(
                grafeo_engine::database::testing::root_lpg_store(&db)
                    .graph(name)
                    .is_some(),
                "fixture must recover the empty graph"
            );
        } else {
            assert!(db.create_graph(name).expect("create named graph"));
        }
    }
    let birth = matches!(case, Case::Birth | Case::ZeroWidth);
    let initial = if birth {
        None
    } else {
        let session = db.session();
        if let Some(name) = graph {
            session
                .use_graph_path(
                    &grafeo_common::types::GraphPath::from_components(&[name])
                        .expect("literal graph path"),
                )
                .expect("select existing graph");
        }
        let id = session.create_node(&["A", "Old"]);
        assert!(id.is_valid());
        Some((id, db.current_epoch()))
    };
    #[cfg(feature = "compact-store")]
    if compact {
        db.compact().expect("make baseline cold");
    }
    #[cfg(not(feature = "compact-store"))]
    assert!(!compact, "compact case requires compact-store");

    let mut session = db.session();
    if let Some(name) = graph {
        session
            .use_graph_path(
                &grafeo_common::types::GraphPath::from_components(&[name])
                    .expect("literal graph path"),
            )
            .expect("select existing graph");
    }
    session
        .begin_transaction()
        .expect("begin label transaction");
    let id = initial.map_or_else(
        || {
            session
                .create_node_with_props(
                    &["A", "Old"],
                    std::iter::empty::<(&str, grafeo_common::types::Value)>(),
                )
                .expect("create transaction birth")
        },
        |(id, _)| id,
    );
    match case {
        Case::AbsentLabel => {
            assert!(!session.remove_node_label(id, "NeverRegistered"));
        }
        Case::NoOp => {
            assert!(session.add_node_label(id, "Transient"));
            assert!(session.remove_node_label(id, "Transient"));
        }
        Case::Empty | Case::DeletedEmpty => {
            assert!(session.remove_node_label(id, "A"));
            assert!(session.remove_node_label(id, "Old"));
        }
        _ => {
            assert!(session.add_node_label(id, "B"));
            assert!(session.remove_node_label(id, "Old"));
            assert!(session.add_node_label(id, "Transient"));
            assert!(session.remove_node_label(id, "Transient"));
        }
    }
    if matches!(case, Case::Savepoint) {
        session.savepoint("keep").expect("savepoint");
        assert!(session.add_node_label(id, "Discarded"));
        assert!(session.create_node(&["DiscardedBirth"]).is_valid());
        session
            .rollback_to_savepoint("keep")
            .expect("rollback suffix");
    }
    let deleted = matches!(case, Case::ZeroWidth | Case::Deleted | Case::DeletedEmpty);
    if deleted {
        assert!(session.delete_node(id));
    }
    if matches!(case, Case::Rollback) {
        session.rollback().expect("rollback labels");
    } else {
        session.commit().expect("commit labels");
    }
    drop(session);
    let epoch = db.current_epoch();
    let mut expected = Vec::new();
    if let Some((_, created)) = initial {
        expected.push((created, vec!["A".to_owned(), "Old".to_owned()]));
    } else if matches!(case, Case::ZeroWidth) {
        expected.push((epoch, vec!["A".to_owned(), "Old".to_owned()]));
    }
    if !matches!(case, Case::NoOp | Case::AbsentLabel | Case::Rollback) {
        let final_labels = if matches!(case, Case::Empty | Case::DeletedEmpty) {
            Vec::new()
        } else {
            vec!["A".to_owned(), "B".to_owned()]
        };
        expected.push((epoch, final_labels));
    }
    assert_eq!(
        labels(&db, graph, id),
        expected,
        "live {case:?}, graph {graph:?}"
    );
    let graph_cut = graph.map_or_else(
        || db.current_epoch(),
        |name| {
            grafeo_engine::database::testing::root_lpg_store(&db)
                .graph(name)
                .expect("published graph")
                .current_epoch()
        },
    );
    db.close()
        .expect("close writer without replacing WAL history");
    drop(db);

    let reopened = GrafeoDB::with_config(config).expect("replay actual commit WAL");
    assert_eq!(
        labels(&reopened, graph, id),
        expected,
        "replay {case:?}, graph {graph:?}"
    );
    let named = graph.map(|name| {
        grafeo_engine::database::testing::root_lpg_store(&reopened)
            .graph(name)
            .expect("recovered named store")
    });
    let store = named
        .as_deref()
        .unwrap_or(grafeo_engine::database::testing::root_lpg_store(&reopened));
    assert_eq!(
        store.current_epoch(),
        graph_cut,
        "graph publication cut for {case:?}"
    );
    assert_eq!(store.get_node(id).is_none(), deleted);
    assert_eq!(store.node_count(), usize::from(!deleted));
    for label in ["Transient", "Discarded", "DiscardedBirth"] {
        assert!(store.nodes_by_label(label).is_empty(), "leaked {label}");
    }
    if graph.is_some() {
        assert_eq!(
            grafeo_engine::database::testing::root_lpg_store(&reopened).node_count(),
            0,
            "named IDs must not alias root"
        );
    }
    reopened.close().expect("close recovered database");
}

#[test]
fn real_commits_preserve_normalized_label_images() {
    for case in [
        Case::Birth,
        Case::ZeroWidth,
        Case::Changed,
        Case::NoOp,
        Case::AbsentLabel,
        Case::Empty,
        Case::Deleted,
        Case::DeletedEmpty,
        Case::Rollback,
        Case::Savepoint,
    ] {
        exercise(case, None, false);
    }
}

#[test]
fn named_and_empty_named_commits_preserve_exact_label_images() {
    for name in ["named", ""] {
        for case in [
            Case::Birth,
            Case::ZeroWidth,
            Case::Changed,
            Case::NoOp,
            Case::AbsentLabel,
            Case::DeletedEmpty,
        ] {
            exercise(case, Some(name), false);
        }
    }
}

#[cfg(feature = "compact-store")]
#[test]
fn cold_node_commits_preserve_exact_label_images() {
    for case in [
        Case::Changed,
        Case::NoOp,
        Case::Empty,
        Case::Deleted,
        Case::DeletedEmpty,
    ] {
        exercise(case, None, true);
    }
}

#[cfg(feature = "gql")]
#[test]
fn query_label_writes_replay_one_committed_image() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let config = Config::persistent(temp.path().join("query-labels"))
        .with_storage_format(StorageFormat::WalDirectory);
    let db = GrafeoDB::with_config(config.clone()).expect("open writer");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute("INSERT (:A:Old)")
        .expect("create through query");
    session
        .execute("MATCH (n:A) SET n:B")
        .expect("add through query");
    session
        .execute("MATCH (n:A) REMOVE n:Old")
        .expect("remove through query");
    session.commit().expect("commit query birth");
    drop(session);
    let id = grafeo_engine::database::testing::root_lpg_store(&db).nodes_by_label("A")[0];
    let expected = vec![(db.current_epoch(), vec!["A".to_owned(), "B".to_owned()])];
    assert_eq!(labels(&db, None, id), expected);
    db.close().expect("close writer");
    drop(db);
    let reopened = GrafeoDB::with_config(config).expect("replay query WAL");
    assert_eq!(labels(&reopened, None, id), expected);
    reopened.close().expect("close recovered database");
}

#[cfg(feature = "gql")]
#[test]
fn replacement_graph_birth_images_do_not_rewrite_dropped_incarnations() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let config = Config::persistent(temp.path().join("replacement"))
        .with_storage_format(StorageFormat::WalDirectory);
    let db = GrafeoDB::with_config(config.clone()).expect("open writer");
    assert!(db.create_graph("swap").expect("create original graph"));
    let mut session = db.session();
    session.begin_transaction().expect("begin replacement");
    session
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["swap"])
                .expect("literal graph path"),
        )
        .expect("select existing graph");
    let old_id = session.create_node(&["Dropped"]);
    assert!(old_id.is_valid());
    assert!(session.add_node_label(old_id, "Discarded"));
    session
        .execute("DROP GRAPH swap")
        .expect("drop old incarnation");
    session
        .execute("CREATE GRAPH swap")
        .expect("create new incarnation");
    session
        .use_graph_path(
            &grafeo_common::types::GraphPath::from_components(&["swap"])
                .expect("literal graph path"),
        )
        .expect("select existing graph");
    let id = session.create_node(&["New", "Temporary"]);
    assert_eq!(old_id, id, "the test must collide graph-local IDs");
    assert!(session.remove_node_label(id, "Temporary"));
    assert!(session.add_node_label(id, "Final"));
    session.commit().expect("commit replacement");
    drop(session);
    let expected = vec![(
        db.current_epoch(),
        vec!["Final".to_owned(), "New".to_owned()],
    )];
    assert_eq!(labels(&db, Some("swap"), id), expected);
    db.close().expect("close writer");
    drop(db);
    let recovered = GrafeoDB::with_config(config).expect("replay replacement");
    assert_eq!(labels(&recovered, Some("swap"), id), expected);
    assert_eq!(
        grafeo_engine::database::testing::root_lpg_store(&recovered).node_count(),
        0
    );
    recovered.close().expect("close recovered database");
}

#[cfg(all(feature = "grafeo-file", feature = "compact-store"))]
#[test]
fn post_checkpoint_images_replay_into_cold_nodes_before_deletion() {
    use std::path::{Path, PathBuf};

    fn sidecar(path: &Path) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(".wal");
        name.into()
    }

    for deleted in [false, true] {
        let temp = tempfile::tempdir().expect("temporary directory");
        let live_path = temp.path().join("live.grafeo");
        let image_path = temp.path().join("process-image.grafeo");
        let config = Config::persistent(&live_path).with_wal_durability(DurabilityMode::Sync);
        let mut db = GrafeoDB::with_config(config.clone()).expect("create container");
        let id = db.session().create_node(&["A", "Old"]);
        assert!(id.is_valid());
        let created = db.current_epoch();
        db.compact().expect("make baseline cold");
        db.close().expect("checkpoint cold baseline");
        drop(db);

        let db = GrafeoDB::with_config(config).expect("reopen cold baseline");
        let mut session = db.session();
        session.begin_transaction().expect("begin WAL tail");
        assert!(session.add_node_label(id, "B"));
        assert!(session.remove_node_label(id, "Old"));
        if deleted {
            assert!(session.delete_node(id));
        }
        session.commit().expect("commit WAL tail");
        drop(session);
        let changed = db.current_epoch();

        // Capture the quiescent, synced process image before close checkpoints
        // the tail. The destination must execute actual Layered WAL recovery.
        std::fs::copy(&live_path, &image_path).expect("copy cold checkpoint");
        let target_wal = sidecar(&image_path);
        std::fs::create_dir(&target_wal).expect("create copied WAL directory");
        for entry in std::fs::read_dir(sidecar(&live_path)).expect("read synced WAL tail") {
            let entry = entry.expect("WAL entry");
            assert!(entry.file_type().expect("WAL entry type").is_file());
            std::fs::copy(entry.path(), target_wal.join(entry.file_name())).expect("copy WAL file");
        }
        db.close().expect("close original after image capture");
        drop(db);

        let recovered = GrafeoDB::with_config(Config::persistent(&image_path))
            .expect("recover cold checkpoint and label-image WAL tail");
        assert_eq!(
            labels(&recovered, None, id),
            vec![
                (created, vec!["A".to_owned(), "Old".to_owned()]),
                (changed, vec!["A".to_owned(), "B".to_owned()]),
            ]
        );
        assert_eq!(recovered.get_node(id).is_none(), deleted);
        recovered.close().expect("close recovered image");
    }
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn label_image_failure_and_lost_commit_ack_keep_distinct_recovery_outcomes() {
    use grafeo_common::testing::wal_failure::{
        disable_commit_ack_failure, disable_mutation_log_failure, enable_commit_ack_failure_once,
        enable_mutation_log_failure_once,
    };

    for lost_commit_ack in [false, true] {
        let temp = tempfile::tempdir().expect("temporary directory");
        let config = Config::persistent(temp.path().join("failed-labels"))
            .with_storage_format(StorageFormat::WalDirectory)
            .with_wal_durability(DurabilityMode::Sync);
        let db = GrafeoDB::with_config(config.clone()).expect("open writer");
        let id = db.session().create_node(&["A", "Old"]);
        let baseline = labels(&db, None, id);
        let mut session = db.session();
        session.begin_transaction().expect("begin");
        assert!(session.add_node_label(id, "B"));
        assert!(session.remove_node_label(id, "Old"));
        if lost_commit_ack {
            enable_commit_ack_failure_once();
        } else {
            // There are no buffered label-intent frames. The next mutation
            // append must be the prepared image inside commit.
            enable_mutation_log_failure_once();
        }
        let result = session.commit();
        disable_commit_ack_failure();
        disable_mutation_log_failure();
        assert!(result.is_err(), "the requested append failure must fire");
        assert!(db.is_durability_poisoned());
        assert_eq!(
            labels(&db, None, id),
            baseline,
            "no prepared image installs after failure"
        );
        drop(session);
        drop(db);

        let recovered = GrafeoDB::with_config(config).expect("recover failed writer");
        let mut expected = baseline;
        if lost_commit_ack {
            expected.push((
                recovered.current_epoch(),
                vec!["A".to_owned(), "B".to_owned()],
            ));
        }
        assert_eq!(labels(&recovered, None, id), expected);
        recovered.close().expect("close recovered database");
    }
}
