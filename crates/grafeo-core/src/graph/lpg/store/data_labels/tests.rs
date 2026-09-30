use super::*;

fn epoch(value: u64) -> EpochId {
    EpochId::new(value)
}
fn tx() -> TransactionId {
    TransactionId::new(401)
}
fn labels(values: &[u32]) -> FxHashSet<u32> {
    values.iter().copied().collect()
}

#[test]
fn prepared_labels_publish_one_final_history_and_sparse_membership() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["A", "B"]);
    let untouched = store.create_node(&["A"]);
    let a = store.get_or_create_label_id("A");
    let b = store.get_or_create_label_id("B");
    let c = store.get_or_create_label_id("C");
    let mut workspace = LabelCommitWorkspace::new(
        vec![
            (node, b, LabelOp::Remove),
            (node, c, LabelOp::Add),
            (node, c, LabelOp::Remove),
            (node, c, LabelOp::Add),
        ],
        vec![],
        vec![],
    );
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let proof = store
        .pending_creation_proof_for_test(&transition, tx(), &[])
        .unwrap();
    let ready = store
        .prepare_commit_labels(
            epoch(2),
            epoch(3),
            tx(),
            &mut workspace,
            &transition,
            &proof,
        )
        .unwrap();
    assert_eq!(ready.label_counts(), &[(node, 2)]);
    assert!(store.label_index.try_read().is_none());
    assert!(store.node_labels.try_read().is_none());
    let ready = ready.release(&transition).unwrap().rebind().unwrap();
    let installed = ready.install();
    assert!(store.label_index.try_read().is_none());
    assert!(store.node_labels.try_read().is_none());
    drop(installed);
    let histories = store.node_labels.read();
    assert_eq!(
        histories[&node].history(),
        &[
            (EpochId::INITIAL, labels(&[a, b])),
            (epoch(3), labels(&[a, c])),
        ]
    );
    assert_eq!(histories[&untouched].len(), 1);
    drop(histories);
    assert_eq!(store.nodes_by_label("A"), vec![node, untouched]);
    assert!(store.nodes_by_label("B").is_empty());
    assert_eq!(store.nodes_by_label("C"), vec![node]);
    assert!(workspace.histories.is_empty());
    assert!(workspace.histories.capacity() >= 1);
    assert!(workspace.extension.is_empty());
    assert!(!workspace.retired_histories.is_empty());
}

#[test]
fn prepared_creation_labels_require_structure_and_leave_foreign_pending_untouched() {
    let store = LpgStore::new().unwrap();
    let own = store.create_node_versioned(&["A"], epoch(2), tx());
    let foreign = store.create_node_versioned(&["Foreign"], epoch(2), TransactionId::new(402));
    let a = store.get_or_create_label_id("A");
    let b = store.get_or_create_label_id("B");
    let foreign_before = store.node_labels.read()[&foreign].history().to_vec();
    let created = [own];
    let mut workspace = LabelCommitWorkspace::new(
        vec![(own, a, LabelOp::Remove), (own, b, LabelOp::Add)],
        vec![],
        created.to_vec(),
    );
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let proof = store
        .pending_creation_proof_for_test(&transition, tx(), &created)
        .unwrap();
    let ready = store
        .prepare_commit_labels(
            epoch(2),
            epoch(3),
            tx(),
            &mut workspace,
            &transition,
            &proof,
        )
        .unwrap();
    assert_eq!(ready.label_counts(), &[(own, 1)]);
    drop(ready.install());
    assert_eq!(
        store.node_labels.read()[&own].history(),
        &[(epoch(3), labels(&[b]))]
    );
    assert_eq!(store.node_labels.read()[&foreign].history(), foreign_before);
    assert_eq!(store.nodes_by_label("Foreign"), vec![foreign]);
    assert!(store.nodes_by_label("A").is_empty());
    assert_eq!(store.nodes_by_label("B"), vec![own]);
}

#[test]
fn prepared_labels_deletion_retains_final_history_without_live_memberships() {
    let store = LpgStore::new().unwrap();
    let existing = store.create_node(&["A"]);
    let own = store.create_node_versioned(&["A"], epoch(2), tx());
    let a = store.get_or_create_label_id("A");
    let b = store.get_or_create_label_id("B");
    let created = [own];
    let mut workspace = LabelCommitWorkspace::new(
        vec![(existing, b, LabelOp::Add), (own, b, LabelOp::Add)],
        vec![existing, own],
        created.to_vec(),
    );
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let proof = store
        .pending_creation_proof_for_test(&transition, tx(), &created)
        .unwrap();
    let ready = store
        .prepare_commit_labels(
            epoch(2),
            epoch(3),
            tx(),
            &mut workspace,
            &transition,
            &proof,
        )
        .unwrap();
    assert_eq!(ready.label_counts(), &[(existing, 0), (own, 0)]);
    drop(ready.install());
    assert_eq!(
        store.node_labels.read()[&existing].history(),
        &[
            (EpochId::INITIAL, labels(&[a])),
            (epoch(3), labels(&[a, b])),
        ]
    );
    assert_eq!(
        store.node_labels.read()[&own].history(),
        &[(epoch(3), labels(&[a])), (epoch(3), labels(&[a, b]))]
    );
    assert!(store.nodes_by_label("A").is_empty());
    assert!(store.nodes_by_label("B").is_empty());
}

#[test]
fn prepared_deleted_labels_retain_empty_final_set_but_no_synthetic_noop_history() {
    for remove_initial in [false, true] {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["A"]);
        let a = store.get_or_create_label_id("A");
        let b = store.get_or_create_label_id("B");
        let mut ops = vec![(node, b, LabelOp::Add), (node, b, LabelOp::Remove)];
        if remove_initial {
            ops.push((node, a, LabelOp::Remove));
        }
        let mut workspace = LabelCommitWorkspace::new(ops, vec![node], vec![]);
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let proof = store
            .pending_creation_proof_for_test(&transition, tx(), &[])
            .unwrap();
        let ready = store
            .prepare_commit_labels(
                epoch(2),
                epoch(3),
                tx(),
                &mut workspace,
                &transition,
                &proof,
            )
            .unwrap();
        assert_eq!(ready.label_counts(), &[(node, 0)]);
        drop(ready.install());
        let mut expected = vec![(EpochId::INITIAL, labels(&[a]))];
        if remove_initial {
            expected.push((epoch(3), labels(&[])));
        }
        assert_eq!(store.node_labels.read()[&node].history(), expected);
        assert!(store.nodes_by_label("A").is_empty());
        assert!(store.nodes_by_label("B").is_empty());
    }
}

#[test]
fn prepared_deleted_labels_reject_unrepresentable_retained_final_set() {
    for created in [false, true] {
        let store = LpgStore::new().unwrap();
        let own = if created {
            store.create_node_versioned(&["Initial"], epoch(2), tx())
        } else {
            store.create_node(&["Initial"])
        };
        assert!(store.delete_node_transactional(own, epoch(2), tx()));
        let initial = store.get_or_create_label_id("Initial");
        let ops: Vec<_> = (0..u16::MAX)
            .map(|ordinal| {
                let label = store.get_or_create_label_id(&format!("Added{ordinal}"));
                (own, label, LabelOp::Add)
            })
            .collect();
        let created_ids = if created { vec![own] } else { vec![] };
        let mut workspace = LabelCommitWorkspace::new(ops, vec![own], created_ids.clone());
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let proof = store
            .pending_creation_proof_for_test(&transition, tx(), &created_ids)
            .unwrap();
        let result = store.prepare_commit_labels(
            epoch(2),
            epoch(3),
            tx(),
            &mut workspace,
            &transition,
            &proof,
        );
        assert!(matches!(
            result,
            Err(Error::Transaction(TransactionError::InvalidState(ref message)))
                if message.contains("label count")
        ));
        drop(result);
        assert_eq!(store.current_epoch(), epoch(0));
        assert_eq!(
            store.node_labels.read()[&own].history(),
            &[(
                if created {
                    EpochId::PENDING
                } else {
                    EpochId::INITIAL
                },
                labels(&[initial])
            )]
        );
        assert_eq!(store.nodes_by_label("Initial"), vec![own]);
        assert!(store.nodes_by_label("Added0").is_empty());
        assert_eq!(store.pending_node_creates(tx()), created_ids);
        assert_eq!(store.pending_node_deletes_peek(tx()), vec![own]);
    }
}

#[test]
fn prepared_labels_reject_ambiguous_pending_and_missing_creation_proof() {
    for created in [false, true] {
        let store = LpgStore::new().unwrap();
        let node = if created {
            store.create_node_versioned(&["A"], epoch(2), tx())
        } else {
            store.create_node(&["A"])
        };
        assert!(store.add_label_versioned(node, "Pending", TransactionId::new(403)));
        let a = store.get_or_create_label_id("A");
        let before = store.node_labels.read()[&node].history().to_vec();
        let ids = if created { vec![node] } else { vec![] };
        let mut workspace =
            LabelCommitWorkspace::new(vec![(node, a, LabelOp::Remove)], vec![], ids.clone());
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let proof = store
            .pending_creation_proof_for_test(&transition, tx(), &ids)
            .unwrap();
        assert!(
            store
                .prepare_commit_labels(
                    epoch(2),
                    epoch(3),
                    tx(),
                    &mut workspace,
                    &transition,
                    &proof
                )
                .is_err()
        );
        assert_eq!(store.node_labels.read()[&node].history(), before);
        assert_eq!(store.nodes_by_label("Pending"), vec![node]);
        assert_eq!(store.nodes_by_label("A"), vec![node]);
    }
    let store = LpgStore::new().unwrap();
    let own = store.create_node_versioned(&["A"], epoch(2), tx());
    let mut workspace = LabelCommitWorkspace::new(vec![], vec![], vec![own]);
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let proof = store
        .pending_creation_proof_for_test(&transition, tx(), &[])
        .unwrap();
    assert!(
        store
            .prepare_commit_labels(
                epoch(2),
                epoch(3),
                tx(),
                &mut workspace,
                &transition,
                &proof
            )
            .is_err()
    );
    assert_eq!(
        store.node_labels.read()[&own].latest_epoch(),
        Some(EpochId::PENDING)
    );
}

#[test]
fn prepared_labels_noop_and_late_failure_preserve_live_state_and_candidates() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FAIL_AFTER_HISTORIES.with(|fail| fail.set(false));
        }
    }
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["A"]);
    let a = store.get_or_create_label_id("A");
    let b = store.get_or_create_label_id("B");
    let mut noop = LabelCommitWorkspace::new(vec![(node, a, LabelOp::Add)], vec![], vec![]);
    let mut failed = LabelCommitWorkspace::new(vec![(node, b, LabelOp::Add)], vec![], vec![]);
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let proof = store
        .pending_creation_proof_for_test(&transition, tx(), &[])
        .unwrap();
    drop(
        store
            .prepare_commit_labels(epoch(2), epoch(3), tx(), &mut noop, &transition, &proof)
            .unwrap()
            .install(),
    );
    assert!(noop.histories.is_empty());
    assert!(noop.retired_histories.is_empty());
    assert_eq!(store.node_labels.read()[&node].len(), 1);
    {
        let _reset = Reset;
        FAIL_AFTER_HISTORIES.with(|fail| fail.set(true));
        assert!(
            store
                .prepare_commit_labels(epoch(2), epoch(3), tx(), &mut failed, &transition, &proof)
                .is_err()
        );
    }
    assert_eq!(
        failed.histories.len(),
        1,
        "late candidate stays anchored in outer workspace"
    );
    assert_eq!(
        failed.final_labels.get(&node),
        Some(&labels(&[a, b])),
        "failed preparation retains final-label scratch in the outer owner"
    );
    assert!(store.label_index.try_write().is_some());
    assert!(store.node_labels.try_write().is_some());
    assert_eq!(store.node_labels.read()[&node].len(), 1);
    assert!(store.nodes_by_label("B").is_empty());
    assert!(
        store
            .prepare_commit_labels(epoch(2), epoch(3), tx(), &mut failed, &transition, &proof)
            .is_err()
    );
}

#[test]
fn prepared_labels_reject_invalid_frontier_registry_id_and_wrong_rebind_proof() {
    let store = LpgStore::new().unwrap();
    let other = LpgStore::new().unwrap();
    let node = store.create_node(&["A"]);
    let a = store.get_or_create_label_id("A");
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let wrong = other.pin_exclusive_unframed_transition().unwrap();
    let proof = store
        .pending_creation_proof_for_test(&transition, tx(), &[])
        .unwrap();
    for (p, c) in [
        (epoch(3), epoch(3)),
        (EpochId::PENDING, epoch(4)),
        (epoch(2), EpochId::PENDING),
    ] {
        let mut workspace = LabelCommitWorkspace::new(vec![], vec![], vec![]);
        assert!(
            store
                .prepare_commit_labels(p, c, tx(), &mut workspace, &transition, &proof)
                .is_err()
        );
    }
    let mut unknown =
        LabelCommitWorkspace::new(vec![(node, u32::MAX, LabelOp::Add)], vec![], vec![]);
    assert!(
        store
            .prepare_commit_labels(epoch(2), epoch(3), tx(), &mut unknown, &transition, &proof)
            .is_err()
    );
    let mut workspace = LabelCommitWorkspace::new(vec![(node, a, LabelOp::Remove)], vec![], vec![]);
    let ready = store
        .prepare_commit_labels(
            epoch(2),
            epoch(3),
            tx(),
            &mut workspace,
            &transition,
            &proof,
        )
        .unwrap();
    assert!(ready.release(&wrong).is_err());
    assert_eq!(store.nodes_by_label("A"), vec![node]);
    store
        .node_labels
        .write()
        .get_mut(&node)
        .unwrap()
        .append(epoch(9), labels(&[a]));
    let mut future = LabelCommitWorkspace::new(vec![(node, a, LabelOp::Remove)], vec![], vec![]);
    assert!(
        store
            .prepare_commit_labels(epoch(2), epoch(3), tx(), &mut future, &transition, &proof)
            .is_err()
    );
    assert_eq!(
        store.node_labels.read()[&node].latest_epoch(),
        Some(epoch(9))
    );
}

#[test]
fn prepared_labels_rebind_and_install_have_zero_allocator_traffic() {
    use crate::allocation_test as allocation;

    let store = LpgStore::new().unwrap();
    let changed = store.create_node(&["Before"]);
    let deleted = store.create_node(&["Deleted"]);
    let before = store.get_or_create_label_id("Before");
    let after = store.get_or_create_label_id("After");
    let created = store.create_node_versioned(&["Created"], epoch(0), tx());
    let created_label = store.get_or_create_label_id("Created");
    let new_slot = store.get_or_create_label_id("NewSlot");
    let created_ids = [created];
    let full_capacity = {
        let histories = store.node_labels.read();
        assert_eq!(histories.len(), histories.capacity());
        histories.capacity()
    };
    let mut workspace = LabelCommitWorkspace::new(
        vec![
            (changed, before, LabelOp::Remove),
            (changed, after, LabelOp::Add),
            (deleted, after, LabelOp::Add),
            (created, new_slot, LabelOp::Add),
        ],
        vec![deleted],
        created_ids.to_vec(),
    );
    let transition = store.pin_exclusive_unframed_transition().unwrap();
    let proof = store
        .pending_creation_proof_for_test(&transition, tx(), &created_ids)
        .unwrap();
    let ready = store
        .prepare_commit_labels(
            epoch(0),
            epoch(1),
            tx(),
            &mut workspace,
            &transition,
            &proof,
        )
        .unwrap();
    assert_eq!(
        ready.guards.labels.capacity(),
        full_capacity,
        "replacement may not overreserve a workaround slot"
    );
    assert_eq!(ready.guards.labels.len(), full_capacity);
    assert_eq!(
        ready.workspace.extension.len(),
        1,
        "fixture also exercises a prebuilt new membership slot"
    );
    let released = ready.release(&transition).unwrap();
    allocation::start();
    let result = std::hint::black_box(released).rebind();
    let rebound = allocation::stop();
    let ready = result.unwrap();
    allocation::start();
    let installed = std::hint::black_box(ready).install();
    let installed_counts = allocation::stop();
    drop(installed);
    assert_eq!(rebound, allocation::Counts::default());
    assert_eq!(installed_counts, allocation::Counts::default());
    assert!(store.nodes_by_label("Before").is_empty());
    assert!(store.nodes_by_label("Deleted").is_empty());
    assert_eq!(store.nodes_by_label("After"), vec![changed]);
    assert_eq!(store.nodes_by_label("Created"), vec![created]);
    assert_eq!(store.nodes_by_label("NewSlot"), vec![created]);
    assert_eq!(
        store.node_labels.read()[&created].history(),
        &[(epoch(1), labels(&[created_label, new_slot]))]
    );
    assert_eq!(workspace.retired_histories.len(), 3);
}

#[test]
fn final_label_contention_releases_partial_guards_without_allocator_traffic() {
    for late_history_contention in [false, true] {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Before"]);
        let before_label = store.get_or_create_label_id("Before");
        let after_label = store.get_or_create_label_id("After");
        let history = store.node_labels.read()[&node].history().to_vec();
        let mut workspace = LabelCommitWorkspace::new(
            vec![
                (node, before_label, LabelOp::Remove),
                (node, after_label, LabelOp::Add),
            ],
            vec![],
            vec![],
        );
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let proof = store
            .pending_creation_proof_for_test(&transition, tx(), &[])
            .unwrap();
        let released = store
            .prepare_commit_labels(
                epoch(0),
                epoch(1),
                tx(),
                &mut workspace,
                &transition,
                &proof,
            )
            .unwrap()
            .release(&transition)
            .unwrap();
        let index_blocker = (!late_history_contention).then(|| store.label_index.read());
        let history_blocker = late_history_contention.then(|| store.node_labels.read());
        crate::allocation_test::start();
        let result = released.rebind();
        let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
        drop(result);
        let traffic = crate::allocation_test::stop();
        assert!(conflict);
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        if late_history_contention {
            // Membership was acquired before the contested history writer;
            // rejection must release it while the history reader still lives.
            assert!(store.label_index.try_write().is_some());
        }
        drop(history_blocker);
        drop(index_blocker);
        assert!(store.label_index.try_write().is_some());
        assert!(store.node_labels.try_write().is_some());
        assert_eq!(
            store.node_labels.read()[&node].history(),
            history.as_slice()
        );
        assert_eq!(store.nodes_by_label("Before"), vec![node]);
        assert!(store.nodes_by_label("After").is_empty());
        assert!(workspace.retired_histories.is_empty());
    }
}
