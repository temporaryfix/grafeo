//! WAL data never substitutes for graph lifecycle authority.

#![cfg(all(feature = "lpg", feature = "wal", feature = "grafeo-file"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, GraphPath, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB};
use grafeo_storage::wal::{LpgMutationOp, LpgWal, WalRecord};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".wal");
    PathBuf::from(name)
}

fn checkpoint(path: &Path) -> TestResult<EpochId> {
    let db =
        GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))?;
    db.wal_checkpoint()?;
    let epoch = db.current_epoch();
    db.close()?;
    Ok(epoch)
}

fn append(path: &Path, records: &[WalRecord]) -> TestResult {
    let wal = LpgWal::open(sidecar(path))?;
    for record in records {
        wal.log(record)?;
    }
    wal.close()?;
    Ok(())
}

fn node(transaction_id: TransactionId, graph: GraphPath) -> WalRecord {
    WalRecord::lpg(
        transaction_id,
        graph,
        LpgMutationOp::CreateNode {
            id: NodeId::new(701),
            labels: vec!["Probe".into()],
        },
    )
}

fn rejected_unchanged(path: &Path) -> TestResult {
    let checkpoint = std::fs::read(path)?;
    let error = GrafeoDB::open(path)
        .err()
        .ok_or("missing graph was invented")?;
    assert!(
        matches!(error, Error::Storage(StorageError::InvalidWalEntry(_))),
        "{error}"
    );
    assert!(
        error.to_string().contains("absent named LPG graph"),
        "{error}"
    );
    assert_eq!(
        std::fs::read(path)?,
        checkpoint,
        "rejection rewrote checkpoint"
    );
    Ok(())
}

#[test]
fn data_without_lifecycle_cannot_create_empty_or_literal_named_graphs() -> TestResult {
    for components in [&[""][..], &["missing"], &["a/b"], &["a", "b"]] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("missing.grafeo");
        let epoch = checkpoint(&path)?.next();
        let transaction_id = TransactionId::new(8001);
        append(
            &path,
            &[
                node(transaction_id, GraphPath::from_components(components)?),
                WalRecord::Committed {
                    transaction_id,
                    epoch,
                },
            ],
        )?;
        rejected_unchanged(&path)?;
    }
    Ok(())
}

#[test]
fn data_cannot_resurrect_dropped_aborted_or_rolled_back_graph_creation() -> TestResult {
    for kind in ["drop", "abort", "savepoint"] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("retired.grafeo");
        let epoch = checkpoint(&path)?.next();
        let creation = TransactionId::new(8001);
        let data = TransactionId::new(8002);
        let graph = GraphPath::from_components(&["g"])?;
        let create = WalRecord::CreateLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(1),
            graph: graph.clone(),
            transaction_id: creation,
        };
        let mut records = match kind {
            "drop" => vec![
                create,
                WalRecord::DropLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: graph.clone(),
                    transaction_id: creation,
                },
                WalRecord::Committed {
                    transaction_id: creation,
                    epoch,
                },
            ],
            "abort" => vec![
                create,
                WalRecord::TransactionAbort {
                    transaction_id: creation,
                },
            ],
            _ => vec![
                WalRecord::TransactionSavepoint {
                    transaction_id: creation,
                    name: "before-create".into(),
                },
                create,
                WalRecord::TransactionRollbackToSavepoint {
                    transaction_id: creation,
                    name: "before-create".into(),
                },
                WalRecord::Committed {
                    transaction_id: creation,
                    epoch,
                },
            ],
        };
        records.extend([
            node(data, graph),
            WalRecord::Committed {
                transaction_id: data,
                epoch: epoch.next(),
            },
        ]);
        append(&path, &records)?;
        rejected_unchanged(&path)?;
    }
    Ok(())
}

#[test]
fn explicit_lifecycle_preserves_root_empty_literal_and_nested_graph_coordinates() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("qualified.grafeo");
    let mut epoch = checkpoint(&path)?;
    let mut records = Vec::new();
    let paths = [
        GraphPath::root(),
        GraphPath::from_components(&[""])?,
        GraphPath::from_components(&["a/b"])?,
        GraphPath::from_components(&["a"])?,
        GraphPath::from_components(&["a", "b"])?,
    ];
    for (offset, graph) in paths.iter().enumerate() {
        let transaction_id = TransactionId::new(8001 + u64::try_from(offset)?);
        epoch = epoch.next();
        if !graph.components().is_empty() {
            records.push(WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(offset as u64),
                graph: graph.clone(),
                transaction_id,
            });
        }
        records.extend([
            node(transaction_id, graph.clone()),
            WalRecord::Committed {
                transaction_id,
                epoch,
            },
        ]);
    }
    append(&path, &records)?;
    let recovered = GrafeoDB::open(&path)?;
    let id = NodeId::new(701);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&recovered)
            .get_node(id)
            .is_some()
    );
    for path in &paths {
        let mut graph =
            std::sync::Arc::clone(grafeo_engine::database::testing::root_lpg_store(&recovered));
        for component in path.components() {
            graph = graph.graph(component).ok_or("explicit graph missing")?;
        }
        assert!(graph.get_node(id).is_some());
    }
    assert_eq!(recovered.current_epoch(), epoch);
    recovered.close()?;
    Ok(())
}

#[test]
fn nested_creation_requires_its_exact_parent() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("orphan.grafeo");
    let epoch = checkpoint(&path)?.next();
    let transaction_id = TransactionId::new(8001);
    append(
        &path,
        &[
            // A slash-bearing sibling is not authority for ["a", "b"].
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: GraphPath::from_components(&["a/b"])?,
                transaction_id,
            },
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: GraphPath::from_components(&["a", "b"])?,
                transaction_id,
            },
            WalRecord::Committed {
                transaction_id,
                epoch,
            },
        ],
    )?;
    rejected_unchanged(&path)
}

#[test]
fn ancestor_drop_invalidates_cached_descendants_and_recreated_parents() -> TestResult {
    for recreate_parent in [false, true] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("descendant.grafeo");
        let epoch = checkpoint(&path)?.next();
        let parent = GraphPath::from_components(&["a"])?;
        let child = parent.child("b")?;
        let creation = TransactionId::new(8001);
        let retirement = TransactionId::new(8002);
        let mutation = TransactionId::new(8003);
        let mut records = vec![
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: parent.clone(),
                transaction_id: creation,
            },
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(2),
                graph: child.clone(),
                transaction_id: creation,
            },
            node(creation, child.clone()),
            WalRecord::Committed {
                transaction_id: creation,
                epoch,
            },
            WalRecord::DropLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: parent.clone(),
                transaction_id: retirement,
            },
        ];
        if recreate_parent {
            records.push(WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(3),
                graph: parent,
                transaction_id: retirement,
            });
        }
        records.extend([
            WalRecord::Committed {
                transaction_id: retirement,
                epoch: epoch.next(),
            },
            node(mutation, child),
            WalRecord::Committed {
                transaction_id: mutation,
                epoch: epoch.next().next(),
            },
        ]);
        append(&path, &records)?;
        rejected_unchanged(&path)?;
    }
    Ok(())
}
