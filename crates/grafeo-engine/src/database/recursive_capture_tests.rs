//! Actual save must exclude child replacement between LPG and auxiliary capture.

use super::GrafeoDB;
use crate::{Config, CreateIndexRequest, GraphModel, IndexCreateKind};
use grafeo_common::storage::SectionType;
use grafeo_common::types::{GraphPath, IndexId, NodeId, Value};
use grafeo_storage::file::GrafeoFileManager;
use std::sync::{Arc, mpsc};
use std::time::Duration;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

thread_local! {
    static AFTER_LPG: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const {
        std::cell::RefCell::new(None)
    };
}

pub(super) fn after_lpg_capture() {
    if let Some(pause) = AFTER_LPG.with(|slot| slot.borrow_mut().take()) {
        pause();
    }
}

fn seed(db: &GrafeoDB, path: &GraphPath, token: &str) -> TestResult<(NodeId, Vec<IndexId>)> {
    let session = db.session();
    session.use_graph_path(path)?;
    let node = session.create_node_with_props(
        &["Doc"],
        [
            ("body", Value::from(token)),
            ("embedding", Value::Vector(vec![1.0, 0.0].into())),
        ],
    )?;
    let mut owners = Vec::new();
    for (property, label, kind) in [
        ("body", None, IndexCreateKind::Property),
        (
            "body",
            Some("Doc"),
            IndexCreateKind::Text {
                min_token_length: None,
            },
        ),
        (
            "embedding",
            Some("Doc"),
            IndexCreateKind::Vector {
                dimensions: Some(2),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        ),
    ] {
        owners.push(db.create_index(CreateIndexRequest {
            graph: path.clone(),
            name: None,
            label: label.map(str::to_owned),
            property: property.into(),
            kind,
        })?);
    }
    Ok((node, owners))
}

fn images(path: &std::path::Path) -> TestResult<Vec<(SectionType, u8, Vec<u8>)>> {
    let file = GrafeoFileManager::open_read_only(path)?;
    let directory = file.read_section_directory()?.ok_or("missing directory")?;
    let mut result = Vec::new();
    for entry in directory.entries() {
        if matches!(
            entry.section_type,
            SectionType::LpgStore
                | SectionType::Catalog
                | SectionType::TextIndex
                | SectionType::VectorStore
        ) {
            result.push((
                entry.section_type,
                entry.version,
                file.read_section_data(entry)?,
            ));
        }
    }
    file.close()?;
    assert_eq!(result.len(), 4);
    Ok(result)
}

#[test]
fn save_holds_one_recursive_incarnation_through_auxiliary_encoding() -> TestResult {
    let directory = tempfile::tempdir()?;
    let before_path = directory.path().join("before.grafeo");
    let captured_path = directory.path().join("captured.grafeo");
    let after_path = directory.path().join("after.grafeo");
    let db = Arc::new(GrafeoDB::with_config(
        Config::in_memory().with_graph_model(GraphModel::Lpg),
    )?);
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .seal_unframed_writes(db.transaction_manager.write_authority())
    );
    assert!(db.create_graph("branch")?);
    assert!(db.create_graph("branch/deep")?);
    db.transaction_manager
        .with_write_authority(|| -> TestResult {
            assert!(
                crate::database::testing::root_lpg_store(&db)
                    .graph("branch")
                    .ok_or("branch absent")?
                    .create_graph("deep")?
            );
            Ok(())
        })?;
    let branch = GraphPath::from_components(&["branch"])?;
    let deep = GraphPath::from_components(&["branch", "deep"])?;
    let literal = GraphPath::from_components(&["branch/deep"])?;
    let (old_node, old_owners) = seed(&db, &branch, "oldneedle")?;
    let (deep_node, _) = seed(&db, &deep, "deepneedle")?;
    let (literal_node, _) = seed(&db, &literal, "literalneedle")?;
    assert_eq!(old_node, deep_node);
    assert_eq!(old_node, literal_node);
    let old_branch = crate::database::testing::root_lpg_store(&db)
        .graph("branch")
        .ok_or("branch absent")?;
    db.save(&before_path)?;
    let expected = images(&before_path)?;

    // Scoped workers cannot outlive this fixture, including failed handshakes.
    let (new_node, new_owners) = std::thread::scope(|scope| -> TestResult<_> {
        let (paused_tx, paused_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let saving_db = Arc::clone(&db);
        let saving_path = captured_path.clone();
        let saver = scope.spawn(move || -> TestResult {
            let proof_db = Arc::clone(&saving_db);
            AFTER_LPG.with(|slot| {
                *slot.borrow_mut() = Some(Box::new(move || {
                    assert!(
                        proof_db.is_open.try_read().is_none(),
                        "save released lifecycle capture"
                    );
                    assert!(
                        proof_db
                            .transaction_manager
                            .publication()
                            .try_write()
                            .is_none(),
                        "save released publication capture before auxiliary encoding"
                    );
                    paused_tx.send(()).expect("report captured LPG");
                    resume_rx
                        .recv_timeout(Duration::from_secs(5))
                        .expect("resume save");
                }));
            });
            saving_db.save(saving_path)?;
            Ok(())
        });
        paused_rx.recv_timeout(Duration::from_secs(5))?;

        let (attempt_tx, attempt_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let writing_db = Arc::clone(&db);
        let replacement_path = branch.clone();
        let writer = scope.spawn(move || -> TestResult<(NodeId, Vec<IndexId>)> {
            attempt_tx.send(())?;
            assert!(writing_db.drop_graph("branch")?);
            assert!(writing_db.create_graph("branch")?);
            let replacement = seed(&writing_db, &replacement_path, "newneedle")?;
            finished_tx.send(())?;
            Ok(replacement)
        });
        let attempted = attempt_rx.recv_timeout(Duration::from_secs(5));
        let blocked = matches!(
            finished_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        // Always release the saver before asserting, including a broken-gate RED.
        let resumed = resume_tx.send(());
        drop(resume_tx);
        let saved = saver.join();
        let written = writer.join();
        attempted?;
        resumed?;
        saved.map_err(|_| "save thread panicked")??;
        let replacement = written.map_err(|_| "replacement thread panicked")??;
        assert!(
            blocked,
            "child replacement crossed the captured LPG/auxiliary boundary"
        );
        Ok(replacement)
    })?;
    assert_eq!(
        new_node, old_node,
        "replacement deliberately reuses local node ID"
    );
    assert!(
        new_owners
            .iter()
            .all(|new| old_owners.iter().all(|old| new.0 > old.0))
    );
    let new_branch = crate::database::testing::root_lpg_store(&db)
        .graph("branch")
        .ok_or("replacement absent")?;
    assert!(!Arc::ptr_eq(&old_branch, &new_branch));
    assert!(new_branch.graph("deep").is_none());
    assert!(
        crate::database::testing::root_lpg_store(&db)
            .graph("branch/deep")
            .is_some(),
        "literal slash sibling survived"
    );

    assert_eq!(
        images(&captured_path)?,
        expected,
        "all four sections belong to the old cut"
    );
    db.save(&after_path)?;
    assert_ne!(images(&after_path)?, expected);
    for (path, token, has_deep) in [
        (&captured_path, "oldneedle", true),
        (&after_path, "newneedle", false),
    ] {
        let reopened = GrafeoDB::open(path)?;
        let graph = crate::database::testing::root_lpg_store(&reopened)
            .graph("branch")
            .ok_or("reopened branch absent")?;
        assert_eq!(graph.graph("deep").is_some(), has_deep);
        assert_eq!(
            graph.find_nodes_by_property("body", &Value::from(token)),
            vec![old_node]
        );
        let text = graph
            .get_text_index("Doc", "body")
            .ok_or("Text owner absent")?;
        assert_eq!(
            text.read().search(token, 1).first().map(|hit| hit.0),
            Some(old_node)
        );
        assert!(graph.get_vector_index("Doc", "embedding").is_some());
        assert!(
            crate::database::testing::root_lpg_store(&reopened)
                .graph("branch/deep")
                .is_some()
        );
        reopened.close()?;
    }
    db.close()?;
    Ok(())
}
