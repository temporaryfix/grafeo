use super::*;
use crate::transaction::EntityId;
use grafeo_common::types::NodeId;

#[cfg(all(feature = "wal", any(feature = "lpg", feature = "triple-store")))]
fn persistent_commit_fixture(path: &std::path::Path, model: crate::GraphModel) -> crate::GrafeoDB {
    crate::GrafeoDB::with_config(
        crate::Config::persistent(path)
            .with_storage_format(crate::config::StorageFormat::WalDirectory)
            .with_graph_model(model)
            .with_wal_durability(crate::DurabilityMode::Sync),
    )
    .expect("open WAL-directory commit fixture")
}

#[cfg(all(feature = "wal", any(feature = "lpg", feature = "triple-store")))]
fn assert_injected_pre_marker_rejection(error: Error) {
    assert!(
        !REJECT_NEXT_FINAL_BIND.with(|reject| reject.replace(false)),
        "the real Session commit must reach final binding"
    );
    assert!(
        matches!(
            error,
            Error::Transaction(TransactionError::WriteConflict(ref reason))
                if reason == "test-only transaction finalization rebind rejection"
        ),
        "expected pre-marker conflict, received {error:?}"
    );
}

#[cfg(all(feature = "lpg", feature = "wal"))]
#[test]
fn session_lpg_final_bind_rejection_rolls_back_and_survives_reopen() {
    use grafeo_common::types::Value;

    for explicit in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("database");
        let db = persistent_commit_fixture(&path, crate::GraphModel::Lpg);
        let mut session = db.session();
        let before_epoch = db.current_epoch();
        let error = if explicit {
            session.begin_transaction().unwrap();
            let rejected = session
                .create_node_with_props(&["Probe"], [("marker", Value::from("rejected"))])
                .unwrap();
            assert!(session.get_node(rejected).is_some());
            assert!(db.get_node(rejected).is_none());
            #[cfg(feature = "gql")]
            session
                .execute("CREATE INDEX idx_rejected FOR (n:Probe) ON (n.marker)")
                .unwrap();
            REJECT_NEXT_FINAL_BIND.with(|reject| reject.set(true));
            let error = session.commit().expect_err("reject explicit final binding");
            assert!(session.get_node(rejected).is_none());
            error
        } else {
            REJECT_NEXT_FINAL_BIND.with(|reject| reject.set(true));
            session
                .create_node_with_props(&["Probe"], [("marker", Value::from("rejected"))])
                .expect_err("reject direct auto-commit final binding")
        };
        assert_injected_pre_marker_rejection(error);
        assert_eq!(db.current_epoch(), before_epoch);
        assert!(!session.in_transaction());
        assert_eq!(db.node_count(), 0);
        assert!(!db.has_property_index("marker"));
        #[cfg(feature = "gql")]
        assert!(session.execute("SHOW INDEXES").unwrap().rows().is_empty());

        // Reuse the same Session: a pre-marker conflict must not poison it.
        if explicit {
            session.begin_transaction().unwrap();
        }
        let good = session
            .create_node_with_props(&["Probe"], [("marker", Value::from("good"))])
            .unwrap();
        if explicit {
            session.commit().unwrap();
        }
        assert!(!session.in_transaction());
        assert_eq!(db.node_count(), 1);
        assert_eq!(
            session.get_node_property(good, "marker"),
            Some(Value::from("good"))
        );
        let committed_epoch = db.current_epoch();
        assert!(committed_epoch.as_u64() > before_epoch.as_u64() + 1);
        drop(session);
        db.close().unwrap();
        drop(db);

        let reopened = persistent_commit_fixture(&path, crate::GraphModel::Lpg);
        assert_eq!(reopened.current_epoch(), committed_epoch);
        assert_eq!(reopened.node_count(), 1);
        assert_eq!(
            reopened.session().get_node_property(good, "marker"),
            Some(Value::from("good"))
        );
        assert!(!reopened.has_property_index("marker"));
        #[cfg(feature = "gql")]
        assert!(
            reopened
                .session()
                .execute("SHOW INDEXES")
                .unwrap()
                .rows()
                .is_empty()
        );
        reopened.close().unwrap();
    }
}

#[cfg(all(feature = "triple-store", feature = "wal"))]
#[test]
fn session_rdf_final_bind_rejection_rolls_back_and_survives_reopen() {
    use grafeo_core::graph::rdf::{Term, Triple};

    for explicit in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("database");
        let db = persistent_commit_fixture(&path, crate::GraphModel::Rdf);
        let mut session = db.session();
        let before_epoch = db.current_epoch();
        let rejected = Triple::new(
            Term::iri("urn:rejected"),
            Term::iri("urn:predicate"),
            Term::iri("urn:object"),
        );
        let good = Triple::new(
            Term::iri("urn:good"),
            Term::iri("urn:predicate"),
            Term::iri("urn:object"),
        );
        let error = if explicit {
            session.begin_transaction().unwrap();
            assert_eq!(session.insert_rdf_batch([rejected.clone()]).unwrap(), 1);
            assert!(!db.rdf_store().contains(&rejected));
            REJECT_NEXT_FINAL_BIND.with(|reject| reject.set(true));
            session
                .commit()
                .expect_err("reject explicit RDF final binding")
        } else {
            REJECT_NEXT_FINAL_BIND.with(|reject| reject.set(true));
            session
                .insert_rdf_batch([rejected.clone()])
                .expect_err("reject direct RDF auto-commit final binding")
        };
        assert_injected_pre_marker_rejection(error);
        assert_eq!(db.current_epoch(), before_epoch);
        assert!(!session.in_transaction());
        assert!(!db.rdf_store().contains(&rejected));
        if explicit {
            session.begin_transaction().unwrap();
        }
        assert_eq!(session.insert_rdf_batch([good.clone()]).unwrap(), 1);
        if explicit {
            session.commit().unwrap();
        }
        assert!(!session.in_transaction());
        assert!(db.rdf_store().contains(&good));
        assert!(!db.rdf_store().contains(&rejected));
        let committed_epoch = db.current_epoch();
        assert!(committed_epoch.as_u64() > before_epoch.as_u64() + 1);
        drop(session);
        db.close().unwrap();
        drop(db);

        let reopened = persistent_commit_fixture(&path, crate::GraphModel::Rdf);
        assert_eq!(reopened.current_epoch(), committed_epoch);
        assert!(reopened.rdf_store().contains(&good));
        assert!(!reopened.rdf_store().contains(&rejected));
        reopened.close().unwrap();
    }
}

#[test]
fn occupied_full_capacity_epoch_replacement_has_no_final_allocator_traffic() {
    let manager = TransactionManager::new();
    let mut workspace = TransactionFinalizationWorkspace::new();
    let _publication = manager.publication.write();
    let tx = manager.begin();
    let commit = manager.prepare_durable_commit(tx).unwrap();
    let old_committed = EpochId::new(7);
    let old_retired = EpochId::new(9);
    for (map, previous) in [
        (&manager.committed_epochs, old_committed),
        (&manager.retired_readers, old_retired),
    ] {
        let mut map = map.write();
        map.insert(tx, previous);
        let mut id = 100_u64;
        while map.len() < map.capacity() {
            map.insert(TransactionId::new(id), EpochId::new(0));
            id += 1;
        }
        assert_eq!(map.len(), map.capacity());
        assert_eq!(map.get(&tx), Some(&previous));
    }
    let released = manager
        .prepare_finalization(tx, commit, &mut workspace)
        .unwrap();
    for map in [&manager.committed_epochs, &manager.retired_readers] {
        let map = map.read();
        assert_eq!(
            map.len(),
            map.capacity(),
            "preparation must not hide the occupied-slot case"
        );
    }

    crate::allocation_test::start();
    let cleanup = released.rebind().map(|ready| ready.install().release());
    let counts = crate::allocation_test::stop();

    let cleanup = cleanup.expect("exact occupied entries remain qualified");
    assert_eq!(counts, crate::allocation_test::Counts::default());
    assert_writers_released(&manager);
    assert_eq!(manager.committed_epoch(tx), Some(commit));
    assert_eq!(manager.retired_readers.read().get(&tx), Some(&commit));
    cleanup.finish();
    assert_eq!(workspace.displaced_committed, Some(old_committed));
    assert_eq!(workspace.displaced_retired, Some(old_retired));
}

#[test]
fn tombstone_heavy_vacant_epoch_slots_have_no_final_allocator_traffic() {
    let manager = TransactionManager::new();
    let mut workspace = TransactionFinalizationWorkspace::new();
    let _publication = manager.publication.write();
    let tx = manager.begin();
    let commit = manager.prepare_durable_commit(tx).unwrap();
    for map in [&manager.committed_epochs, &manager.retired_readers] {
        let mut map = map.write();
        map.reserve(512);
        let mut next = 100_u64;
        while map.len() < map.capacity() {
            map.insert(TransactionId::new(next), EpochId::new(0));
            next += 1;
        }
        let full = map.len();
        for id in (100..next).step_by(2) {
            map.remove(&TransactionId::new(id));
        }
        assert!(map.len() < full);
        assert!(!map.contains_key(&tx));
    }
    let before_committed = manager.committed_epochs.read().clone();
    let before_retired = manager.retired_readers.read().clone();
    let released = manager
        .prepare_finalization(tx, commit, &mut workspace)
        .unwrap();
    for map in [&manager.committed_epochs, &manager.retired_readers] {
        let map = map.read();
        assert!(map.len() < map.capacity());
    }

    crate::allocation_test::start();
    let cleanup = released.rebind().map(|ready| ready.install().release());
    let counts = crate::allocation_test::stop();

    let cleanup = cleanup.expect("reserved vacant slots bind");
    assert_eq!(counts, crate::allocation_test::Counts::default());
    for (map, before) in [
        (&manager.committed_epochs, &before_committed),
        (&manager.retired_readers, &before_retired),
    ] {
        let map = map.read();
        assert_eq!(map.len(), before.len() + 1);
        assert_eq!(map.get(&tx), Some(&commit));
        for (id, epoch) in before {
            assert_eq!(map.get(id), Some(epoch));
        }
    }
    cleanup.finish();
    assert_eq!(workspace.displaced_committed, None);
    assert_eq!(workspace.displaced_retired, None);
}

#[test]
fn begun_record_can_publish_count_increment_while_finalization_writers_are_held() {
    let manager = Arc::new(TransactionManager::new());
    let mut workspace = TransactionFinalizationWorkspace::new();
    let _publication = manager.publication.write();
    let tx = manager.begin();
    let commit = manager.prepare_durable_commit(tx).unwrap();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    *manager.begin_count_pause.write() = Some(super::super::BeginCountPause {
        entered: Arc::clone(&entered),
        resume: Arc::clone(&resume),
    });
    let beginning = {
        let manager = Arc::clone(&manager);
        std::thread::spawn(move || manager.begin_with_isolation(IsolationLevel::Serializable))
    };
    entered.wait();
    // Exercise the real BEGIN split: both records exist, but only the older
    // transaction has completed its active-count publication.
    assert_eq!(manager.active_count(), 2);
    assert_eq!(manager.active_count.load(Ordering::Relaxed), 1);
    let ready = manager
        .prepare_finalization(tx, commit, &mut workspace)
        .unwrap()
        .rebind()
        .unwrap();
    let installed = ready.install();
    assert_eq!(manager.active_count.load(Ordering::Relaxed), 0);
    resume.wait();
    let other = beginning
        .join()
        .expect("BEGIN completes without the retained transaction writer");
    assert_eq!(manager.active_count.load(Ordering::Relaxed), 1);
    assert!(manager.transactions.try_read().is_none());
    let cleanup = installed.release();
    assert_eq!(manager.active_count(), 1);
    assert_eq!(manager.state(other), Some(TransactionState::Active));
    assert_eq!(manager.state(tx), Some(TransactionState::Committed));
    assert_eq!(manager.committed_epoch(tx), Some(commit));
    cleanup.finish();
    manager.abort(other).unwrap();
    assert_eq!(manager.active_count.load(Ordering::Relaxed), 0);
}

fn assert_unpublished(manager: &TransactionManager, tx: TransactionId, commit: EpochId) {
    let transactions = manager.transactions.read();
    let info = transactions.get(&tx).expect("retained transaction");
    assert_eq!(info.state, TransactionState::Active);
    assert_eq!(info.reserved_commit_epoch, Some(commit));
    assert_eq!(manager.active_count.load(Ordering::Relaxed), 1);
    assert!(!manager.committed_epochs.read().contains_key(&tx));
    assert!(!manager.retired_readers.read().contains_key(&tx));
}

fn assert_writers_released(manager: &TransactionManager) {
    assert!(manager.transactions.try_write().is_some());
    assert!(manager.committed_epochs.try_write().is_some());
    assert!(manager.retired_readers.try_write().is_some());
}

#[derive(Debug, PartialEq, Eq)]
struct Observation {
    epoch: EpochId,
    watermark: EpochId,
    active: usize,
    atomic_active: u64,
    committed: FxHashMap<TransactionId, EpochId>,
    retired: FxHashMap<TransactionId, EpochId>,
    states: Vec<(TransactionId, TransactionState, Option<EpochId>)>,
    readers: Vec<TransactionId>,
}

fn observe(manager: &TransactionManager, entity: EntityId) -> Observation {
    let mut states: Vec<_> = manager
        .transactions
        .read()
        .iter()
        .map(|(tx, info)| (*tx, info.state, info.reserved_commit_epoch))
        .collect();
    states.sort_unstable_by_key(|(tx, _, _)| *tx);
    let mut readers = manager.read_registry.readers_of_compatible(entity, None);
    readers.sort_unstable();
    Observation {
        epoch: manager.current_epoch(),
        watermark: manager.min_active_epoch(),
        active: manager.active_count(),
        atomic_active: manager.active_count.load(Ordering::Relaxed),
        committed: manager.committed_epochs.read().clone(),
        retired: manager.retired_readers.read().clone(),
        states,
        readers,
    }
}

fn finalize_prepared(
    manager: &TransactionManager,
    tx: TransactionId,
    commit: EpochId,
    workspace: &mut TransactionFinalizationWorkspace,
) {
    let installed = manager
        .prepare_finalization(tx, commit, workspace)
        .expect("prepare finalization")
        .rebind()
        .expect("bind finalization")
        .install();
    assert!(manager.transactions.try_read().is_none());
    assert!(manager.committed_epochs.try_read().is_none());
    assert!(manager.retired_readers.try_read().is_none());
    let cleanup = installed.release();
    assert_writers_released(manager);
    cleanup.finish();
    assert!(workspace.phase == Phase::Cleaned);
}

#[test]
fn prepared_finalization_matches_current_ssi_and_epoch_lifecycle() {
    let entity = EntityId::Node(NodeId::new(11));
    for concurrent_reader in [false, true] {
        let ordinary = TransactionManager::new();
        let prepared = TransactionManager::new();
        let mut first_workspace = TransactionFinalizationWorkspace::new();
        let mut second_workspace = TransactionFinalizationWorkspace::new();
        let _ordinary_publication = ordinary.publication.write();
        let _prepared_publication = prepared.publication.write();
        let ordinary_tx = ordinary.begin_with_isolation(IsolationLevel::Serializable);
        let prepared_tx = prepared.begin_with_isolation(IsolationLevel::Serializable);
        ordinary.record_read(ordinary_tx, entity, None).unwrap();
        prepared.record_read(prepared_tx, entity, None).unwrap();
        let others = concurrent_reader.then(|| {
            (
                ordinary.begin_with_isolation(IsolationLevel::Serializable),
                prepared.begin_with_isolation(IsolationLevel::Serializable),
            )
        });
        let ordinary_c = ordinary.prepare_durable_commit(ordinary_tx).unwrap();
        let prepared_c = prepared.prepare_durable_commit(prepared_tx).unwrap();
        assert_eq!(ordinary_c, prepared_c);
        ordinary
            .finalize_durable_commit(ordinary_tx, ordinary_c)
            .unwrap();
        finalize_prepared(&prepared, prepared_tx, prepared_c, &mut first_workspace);
        assert_eq!(observe(&ordinary, entity), observe(&prepared, entity));
        assert_eq!(
            prepared.read_registry.readers_of_compatible(entity, None),
            if concurrent_reader {
                vec![prepared_tx]
            } else {
                Vec::new()
            }
        );
        if let Some((ordinary_other, prepared_other)) = others {
            let ordinary_c = ordinary.prepare_durable_commit(ordinary_other).unwrap();
            let prepared_c = prepared.prepare_durable_commit(prepared_other).unwrap();
            ordinary
                .finalize_durable_commit(ordinary_other, ordinary_c)
                .unwrap();
            finalize_prepared(&prepared, prepared_other, prepared_c, &mut second_workspace);
            assert_eq!(observe(&ordinary, entity), observe(&prepared, entity));
            assert!(prepared.retired_readers.read().is_empty());
            assert!(
                prepared
                    .read_registry
                    .readers_of_compatible(entity, None)
                    .is_empty()
            );
        }
    }
}

#[test]
fn preparation_and_abandoned_proofs_do_not_finalize_or_prune_readers() {
    for abandon_ready in [false, true] {
        let manager = TransactionManager::new();
        let mut workspace = TransactionFinalizationWorkspace::new();
        let _publication = manager.publication.write();
        let tx = manager.begin_with_isolation(IsolationLevel::Serializable);
        let entity = EntityId::Node(NodeId::new(19));
        manager.record_read(tx, entity, Some(7)).unwrap();
        let commit = manager.prepare_durable_commit(tx).unwrap();
        let before = observe(&manager, entity);
        let released = manager
            .prepare_finalization(tx, commit, &mut workspace)
            .unwrap();
        assert_eq!(observe(&manager, entity), before);
        if abandon_ready {
            drop(released.rebind().unwrap());
        } else {
            drop(released);
        }
        assert_writers_released(&manager);
        assert_eq!(observe(&manager, entity), before);
        assert!(manager.finish_finalization(&mut workspace).is_err());
        assert_eq!(observe(&manager, entity), before);
        manager.abort(tx).unwrap();
        assert!(
            manager
                .read_registry
                .readers_of_compatible(entity, None)
                .is_empty()
        );
    }
}

#[test]
fn every_final_writer_contention_rejects_and_releases_partial_acquisition() {
    for blocked in 0..3 {
        let manager = TransactionManager::new();
        let mut workspace = TransactionFinalizationWorkspace::new();
        let mut fresh = TransactionFinalizationWorkspace::new();
        let _publication = manager.publication.write();
        let tx = manager.begin();
        let commit = manager.prepare_durable_commit(tx).unwrap();
        let released = manager
            .prepare_finalization(tx, commit, &mut workspace)
            .unwrap();
        let transactions = (blocked == 0).then(|| manager.transactions.read());
        let committed = (blocked == 1).then(|| manager.committed_epochs.read());
        let retired = (blocked == 2).then(|| manager.retired_readers.read());
        let Err(error) = released.rebind() else {
            panic!("held reader must reject final writer")
        };
        assert!(matches!(error, TransactionFinalizationError::Conflict(_)));
        if blocked > 0 {
            assert!(manager.transactions.try_write().is_some());
        }
        if blocked > 1 {
            assert!(manager.committed_epochs.try_write().is_some());
        }
        drop(retired);
        drop(committed);
        drop(transactions);
        assert_writers_released(&manager);
        assert_unpublished(&manager, tx, commit);
        drop(
            manager
                .prepare_finalization(tx, commit, &mut fresh)
                .unwrap()
                .rebind()
                .unwrap(),
        );
        assert_unpublished(&manager, tx, commit);
    }
}

#[test]
fn stale_state_or_epoch_metadata_rejects_without_partial_publication() {
    for changed_state in [false, true] {
        let manager = TransactionManager::new();
        let mut workspace = TransactionFinalizationWorkspace::new();
        let _publication = manager.publication.write();
        let tx = manager.begin();
        let commit = manager.prepare_durable_commit(tx).unwrap();
        let released = manager
            .prepare_finalization(tx, commit, &mut workspace)
            .unwrap();
        if changed_state {
            manager.abort(tx).unwrap();
        } else {
            manager
                .committed_epochs
                .write()
                .insert(tx, EpochId::new(77));
        }
        let before = observe(&manager, EntityId::Node(NodeId::new(1)));
        assert!(released.rebind().is_err());
        assert_writers_released(&manager);
        assert_eq!(observe(&manager, EntityId::Node(NodeId::new(1))), before);
    }
}

#[test]
fn consumed_map_reservation_rejects_before_any_install() {
    for committed_map in [false, true] {
        let manager = TransactionManager::new();
        let mut workspace = TransactionFinalizationWorkspace::new();
        let _publication = manager.publication.write();
        let tx = manager.begin();
        let commit = manager.prepare_durable_commit(tx).unwrap();
        let released = manager
            .prepare_finalization(tx, commit, &mut workspace)
            .unwrap();
        {
            let mut map = if committed_map {
                manager.committed_epochs.write()
            } else {
                manager.retired_readers.write()
            };
            let mut id = 100_u64;
            while map.len() < map.capacity() {
                map.insert(TransactionId::new(id), EpochId::new(0));
                id += 1;
            }
        }
        let before = observe(&manager, EntityId::Node(NodeId::new(1)));
        assert!(matches!(
            released.rebind(),
            Err(TransactionFinalizationError::Invalid(
                "prepared transaction epoch capacity was consumed"
            ))
        ));
        assert_writers_released(&manager);
        assert_eq!(observe(&manager, EntityId::Node(NodeId::new(1))), before);
    }
}

#[test]
fn cleanup_waits_for_explicit_tail_and_preserves_nonconcurrent_reader_rule() {
    let manager = TransactionManager::new();
    let mut workspace = TransactionFinalizationWorkspace::new();
    let _publication = manager.publication.write();
    let tx = manager.begin_with_isolation(IsolationLevel::Serializable);
    let entity = EntityId::Node(NodeId::new(37));
    manager.record_read(tx, entity, None).unwrap();
    let commit = manager.prepare_durable_commit(tx).unwrap();
    let cleanup = manager
        .prepare_finalization(tx, commit, &mut workspace)
        .unwrap()
        .rebind()
        .unwrap()
        .install()
        .release();
    assert_eq!(manager.retired_readers.read().get(&tx), Some(&commit));
    assert_eq!(
        manager.read_registry.readers_of_compatible(entity, None),
        vec![tx]
    );
    // Deferred removal cannot create an rw-edge for a writer starting at C.
    assert!(!manager.reader_concurrent_with(tx, Some(commit)));
    assert!(manager.reader_concurrent_with(tx, Some(EpochId::new(0))));
    drop(cleanup);
    assert!(workspace.phase == Phase::Installed);
    let foreign = TransactionManager::new();
    assert!(foreign.finish_finalization(&mut workspace).is_err());
    assert_eq!(manager.retired_readers.read().get(&tx), Some(&commit));
    manager.finish_finalization(&mut workspace).unwrap();
    assert!(manager.retired_readers.read().is_empty());
    assert!(
        manager
            .read_registry
            .readers_of_compatible(entity, None)
            .is_empty()
    );
    assert_eq!(manager.state(tx), Some(TransactionState::Committed));
    assert_eq!(manager.committed_epoch(tx), Some(commit));
}

#[test]
fn installed_fence_unwind_never_rolls_back_or_runs_gc() {
    let manager = TransactionManager::new();
    let mut workspace = TransactionFinalizationWorkspace::new();
    let _publication = manager.publication.write();
    let tx = manager.begin_with_isolation(IsolationLevel::Serializable);
    let entity = EntityId::Node(NodeId::new(43));
    manager.record_read(tx, entity, None).unwrap();
    let commit = manager.prepare_durable_commit(tx).unwrap();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _installed = manager
            .prepare_finalization(tx, commit, &mut workspace)
            .unwrap()
            .rebind()
            .unwrap()
            .install();
        panic!("companion unwound after durable installation");
    }));
    assert!(unwind.is_err());
    assert_writers_released(&manager);
    assert_eq!(manager.state(tx), Some(TransactionState::Committed));
    assert_eq!(manager.committed_epoch(tx), Some(commit));
    assert_eq!(manager.retired_readers.read().get(&tx), Some(&commit));
    manager.finish_finalization(&mut workspace).unwrap();
    assert!(
        manager
            .read_registry
            .readers_of_compatible(entity, None)
            .is_empty()
    );
}

#[test]
fn preparation_rejects_wrong_epoch_and_workspace_reuse() {
    let manager = TransactionManager::new();
    let mut wrong = TransactionFinalizationWorkspace::new();
    let mut workspace = TransactionFinalizationWorkspace::new();
    let _publication = manager.publication.write();
    let tx = manager.begin();
    let commit = manager.prepare_durable_commit(tx).unwrap();
    assert!(
        manager
            .prepare_finalization(tx, EpochId::PENDING, &mut wrong)
            .is_err()
    );
    assert_unpublished(&manager, tx, commit);
    drop(
        manager
            .prepare_finalization(tx, commit, &mut workspace)
            .unwrap(),
    );
    assert!(
        manager
            .prepare_finalization(tx, commit, &mut workspace)
            .is_err()
    );
    assert_unpublished(&manager, tx, commit);
}
