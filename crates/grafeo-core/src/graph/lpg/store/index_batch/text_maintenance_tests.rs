//! Connected survivor witnesses at the shared data/registry publication seam.

use super::super::data_publication::StoreDataWorkspace;
use super::*;
use crate::allocation_test::{self, Counts};
use crate::index::text::{BM25Config, Tokenizer};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

const P: EpochId = EpochId::new(5);
const C: EpochId = EpochId::new(6);
const TX: TransactionId = TransactionId::new(941);

struct CountingTokenizer(Arc<AtomicUsize>);

impl Tokenizer for CountingTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.0.fetch_add(1, Ordering::Relaxed);
        text.split_whitespace().map(str::to_owned).collect()
    }
}

struct Fixture {
    store: LpgStore,
    updated: NodeId,
    deleted: NodeId,
    untouched: NodeId,
    born: NodeId,
    caller: Arc<RwLock<InvertedIndex>>,
    target: Arc<RwLock<InvertedIndex>>,
    tokenizations: Arc<AtomicUsize>,
}

impl Fixture {
    fn new() -> Self {
        let store = LpgStore::new().unwrap();
        let updated = store.create_node(&["Doc"]);
        let deleted = store.create_node(&["Doc"]);
        let untouched = store.create_node(&["Doc"]);
        let born = store.create_node(&["Doc"]);
        store.set_node_property(updated, "body", Value::from("ancient phrase"));
        store.set_node_property_at_epoch(
            updated,
            "body",
            Value::from("old phrase"),
            EpochId::new(3),
        );
        store.set_node_property(deleted, "body", Value::from("gone record"));
        store.set_node_property(untouched, "body", Value::from("steady payload"));
        store.set_epoch(P);
        let tokenizations = Arc::new(AtomicUsize::new(0));
        let mut index = InvertedIndex::with_tokenizer(
            BM25Config { k1: 2.4, b: 0.5 },
            Box::new(CountingTokenizer(Arc::clone(&tokenizations))),
        );
        index.insert_versioned(updated, "ancient phrase", EpochId::INITIAL, None);
        index.insert_versioned(updated, "old phrase", EpochId::new(3), None);
        index.insert_versioned(deleted, "gone record", EpochId::INITIAL, None);
        index.insert_versioned(untouched, "steady payload", EpochId::INITIAL, None);
        let caller = Arc::new(RwLock::new(index));
        store.add_text_index("Doc", "body", Arc::clone(&caller));
        let target = store.text_indexes.read()[&encode_index_key("Doc", "body")]
            .target_identity()
            .upgrade()
            .unwrap();
        Self {
            store,
            updated,
            deleted,
            untouched,
            born,
            caller,
            target,
            tokenizations,
        }
    }

    fn maintenance(&self) -> IndexRegistryEdit {
        IndexRegistryEdit::Maintain {
            expected: self.store.observe_text_index("Doc", "body").unwrap(),
            changes: IndexRegistryMaintenance::Text {
                rows: vec![
                    (self.updated, Some("fresh fresh word".into())),
                    (self.deleted, None),
                    (self.born, Some("birth token".into())),
                ],
                frontier: P,
                commit_epoch: C,
                transaction_id: TX,
            },
        }
    }

    fn registry_workspace(&self) -> IndexRegistryWorkspace<'_> {
        IndexRegistryWorkspace::new(vec![StoreIndexEdits {
            store: &self.store,
            edits: vec![self.maintenance()],
        }])
    }

    fn buffer_final_rows(&self) {
        // Intermediate delta entries are deliberately overwritten. The survivor
        // receives the final rows once, not a replay of this operation sequence.
        self.store.set_node_property_buffered(
            self.updated,
            "body",
            Value::from("intermediate discarded"),
            TX,
        );
        self.store.set_node_property_buffered(
            self.updated,
            "body",
            Value::from("fresh fresh word"),
            TX,
        );
        self.store
            .remove_node_property_buffered(self.deleted, "body", TX);
        self.store
            .set_node_property_buffered(self.born, "body", Value::from("birth token"), TX);
    }

    fn assert_unpublished(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert_eq!(self.store.current_epoch(), P);
        assert_eq!(
            self.store
                .get_node_property(self.updated, &PropertyKey::new("body")),
            Some(Value::from("old phrase"))
        );
        assert_eq!(
            self.store
                .get_node_property(self.deleted, &PropertyKey::new("body")),
            Some(Value::from("gone record"))
        );
        assert_eq!(
            self.store
                .get_node_property(self.born, &PropertyKey::new("body")),
            None
        );
        let index = self.target.read();
        assert_eq!(index.doc_count_at(C, TransactionId::INVALID)?, 3);
        assert_eq!(index.total_length_at(C, TransactionId::INVALID)?, 6);
        assert!(index.search("fresh", 10).is_empty());
        assert!(index.search("birth", 10).is_empty());
        assert_eq!(index.search("old", 10)[0].0, self.updated);
        assert_eq!(index.search("gone", 10)[0].0, self.deleted);
        Ok(())
    }
}

/// Probe creation, public error materialization and destruction deliberately
/// occur outside allocator witnesses. The concrete lock is never reacquired
/// while a collective final fence is held.
fn scope_available(target: &Arc<RwLock<InvertedIndex>>) -> bool {
    let mut probe = TextCommitWorkspace::new(Vec::new(), P, C, TransactionId::new(942));
    let result = target.read().pin_commit_scope(&mut probe);
    let available = result.is_ok();
    drop(result);
    drop(probe);
    available
}

fn held_scope_count(fences: &RegistryFences<'_, '_>) -> usize {
    fences
        .stores
        .iter()
        .flat_map(|store| &store.edits)
        .filter(|edit| {
            matches!(
                edit.maintenance.as_ref(),
                Some(PrivateMaintenance::Text { scope: Some(_), .. })
            )
        })
        .count()
}

#[test]
fn shared_data_and_text_survivor_publish_once_at_c_without_allocator_traffic()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let fixture = Fixture::new();
    let key = encode_index_key("Doc", "body");
    let observation = fixture.store.observe_text_index("Doc", "body").unwrap();
    let registration = fixture.store.text_indexes.read()[&key].clone();
    let slot = fixture.store.index_slots.lock()[&key];
    fixture.buffer_final_rows();
    let mut data_workspace = StoreDataWorkspace::new(TX, P, C);
    let mut workspace = fixture.registry_workspace();
    workspace.prepare_inputs().unwrap();
    {
        let authority =
            RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
                .unwrap();
        let data = fixture
            .store
            .prepare_buffered_commit_data(
                authority.transition(&fixture.store).unwrap(),
                &mut data_workspace,
            )
            .unwrap();
        let tokenizations = fixture.tokenizations.load(Ordering::Relaxed);
        let indexes =
            prepare_under_authority(&mut workspace.registry, authority.workspace).unwrap();
        assert_eq!(
            fixture.tokenizations.load(Ordering::Relaxed),
            tokenizations + 2
        );
        assert_eq!(held_scope_count(&indexes.fences), 1);
        assert!(indexes.fences.active.is_empty());
        assert!(fixture.target.try_write().is_some());
        assert!(fixture.caller.try_write().is_some());
        assert!(!scope_available(&fixture.target));
        fixture
            .assert_unpublished()
            .expect("unpublished Text state");

        allocation_test::start();
        let data = data.rebind().unwrap();
        let mut indexes = indexes.rebind().unwrap();
        let data = data.install();
        indexes.fences.install();
        let retained_scope = held_scope_count(&indexes.fences) == 1;
        let target_excluded = fixture.target.try_read().is_none();
        let caller_excluded = fixture.caller.try_read().is_none();
        let registry_excluded = fixture.store.text_indexes.try_read().is_none();
        drop(indexes);
        drop(data);
        let traffic = allocation_test::stop();
        assert_eq!(traffic, Counts::default());
        assert!(retained_scope && target_excluded && caller_excluded && registry_excluded);
        assert!(scope_available(&fixture.target));
    }

    fixture
        .store
        .validate_index_registration(&observation)
        .unwrap();
    let current = fixture.store.text_indexes.read()[&key].clone();
    assert!(Arc::ptr_eq(
        &registration.registration,
        &current.registration
    ));
    assert!(Arc::ptr_eq(
        &fixture.target,
        &current.target_identity().upgrade().unwrap()
    ));
    assert_eq!(fixture.store.index_slots.lock()[&key], slot);
    assert_eq!(fixture.store.current_epoch(), C);
    assert_eq!(
        fixture
            .store
            .get_node_property(fixture.updated, &PropertyKey::new("body")),
        Some(Value::from("fresh fresh word"))
    );
    assert_eq!(
        fixture
            .store
            .get_node_property(fixture.deleted, &PropertyKey::new("body")),
        None
    );
    assert_eq!(
        fixture
            .store
            .get_node_property(fixture.born, &PropertyKey::new("body")),
        Some(Value::from("birth token"))
    );
    assert_eq!(
        fixture
            .store
            .node_properties
            .get_at(fixture.updated, &PropertyKey::new("body"), P),
        Some(Value::from("old phrase"))
    );
    assert!(!fixture.store.tx_property_overlay.read().contains_key(&TX));
    assert!(!fixture.store.text_index_overlay.read().contains_key(&TX));
    {
        let index = fixture.target.read();
        assert_eq!(index.config().k1, 2.4);
        assert_eq!(index.config().b, 0.5);
        assert_eq!(index.doc_count_at(P, TransactionId::INVALID)?, 3);
        assert_eq!(index.total_length_at(P, TransactionId::INVALID)?, 6);
        assert_eq!(index.doc_count_at(C, TransactionId::INVALID)?, 3);
        assert_eq!(index.total_length_at(C, TransactionId::INVALID)?, 7);
        let score = |node, query, epoch| {
            index.score_document_visible(node, query, epoch, TransactionId::INVALID, None, false)
        };
        assert!(
            score(fixture.updated, "ancient", EpochId::INITIAL)?.is_some_and(|score| score > 0.0)
        );
        assert!(score(fixture.updated, "old", P)?.is_some_and(|score| score > 0.0));
        assert!(score(fixture.updated, "fresh", C)?.is_some_and(|score| score > 0.0));
        assert!(score(fixture.deleted, "gone", P)?.is_some_and(|score| score > 0.0));
        assert_eq!(score(fixture.deleted, "gone", C)?, None);
        assert_eq!(
            score(fixture.born, "birth", P)?,
            None,
            "survivor birth is C, not INITIAL"
        );
        assert!(score(fixture.born, "birth", C)?.is_some_and(|score| score > 0.0));
        assert!(score(fixture.untouched, "steady", C)?.is_some_and(|score| score > 0.0));
        assert!(index.search("intermediate", 10).is_empty());
    }
    allocation_test::start();
    drop(workspace);
    drop(data_workspace);
    assert!(
        allocation_test::stop().dealloc > 0,
        "outer workspaces own retirement"
    );
    Ok(())
}

#[test]
fn released_and_final_text_maintenance_abandonment_clear_scopes_without_publication() {
    for final_phase in [false, true] {
        let fixture = Fixture::new();
        let mut workspace = fixture.registry_workspace();
        workspace.prepare_inputs().unwrap();
        {
            let authority = RegistryAuthority::acquire(
                &mut workspace.registry.stores,
                &mut workspace.authority,
            )
            .unwrap();
            let released =
                prepare_under_authority(&mut workspace.registry, authority.workspace).unwrap();
            assert!(released.fences.active.is_empty());
            assert_eq!(held_scope_count(&released.fences), 1);
            assert!(!scope_available(&fixture.target));
            if final_phase {
                let ready = released.rebind().unwrap();
                assert_eq!(held_scope_count(&ready.fences), 1);
                assert!(fixture.target.try_read().is_none());
                allocation_test::start();
                drop(ready);
            } else {
                allocation_test::start();
                drop(released);
            }
            assert_eq!(allocation_test::stop(), Counts::default());
            assert!(scope_available(&fixture.target));
            assert!(fixture.store.text_indexes.try_write().is_some());
            assert!(fixture.caller.try_write().is_some());
        }
        fixture
            .assert_unpublished()
            .expect("unpublished Text state");
        assert!(
            workspace
                .registry
                .stores
                .iter()
                .flat_map(|store| &store.edits)
                .all(|edit| {
                    !matches!(
                        edit.maintenance.as_ref(),
                        Some(PrivateMaintenance::Text { scope: Some(_), .. })
                    )
                })
        );
    }
}

#[test]
fn later_initial_edit_failure_clears_already_prepared_text_scope() {
    let fixture = Fixture::new();
    fixture.store.create_property_index("occupied");
    let observation = fixture.store.observe_text_index("Doc", "body").unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &fixture.store,
        edits: vec![
            fixture.maintenance(),
            IndexRegistryEdit::Create {
                key: IndexRegistryKey::Property(PropertyKey::new("occupied")),
                contents: IndexRegistryContents::Property(Vec::new()),
            },
        ],
    }]);
    workspace.prepare_inputs().unwrap();
    {
        let authority =
            RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
                .unwrap();
        let tokenizations = fixture.tokenizations.load(Ordering::Relaxed);
        let result = prepare_under_authority(&mut workspace.registry, authority.workspace);
        assert!(result.is_err());
        drop(result);
        assert_eq!(
            fixture.tokenizations.load(Ordering::Relaxed),
            tokenizations + 2,
            "the survivor completed before the later edit failed"
        );
        assert!(scope_available(&fixture.target));
        assert!(fixture.store.text_indexes.try_write().is_some());
        assert!(fixture.caller.try_write().is_some());
    }
    fixture
        .store
        .validate_index_registration(&observation)
        .unwrap();
    fixture
        .assert_unpublished()
        .expect("unpublished Text state");
    // The caller may retry with a fresh outer workspace after the failed batch.
    let mut retry = fixture.registry_workspace();
    drop(prepare_index_registry_batch(&mut retry).unwrap());
    assert!(scope_available(&fixture.target));
}

#[test]
fn final_caller_and_target_contention_release_text_scopes_under_data_writers() {
    for held_target in [false, true] {
        let fixture = Fixture::new();
        fixture.buffer_final_rows();
        let observation = fixture.store.observe_text_index("Doc", "body").unwrap();
        let mut data_workspace = StoreDataWorkspace::new(TX, P, C);
        let mut workspace = fixture.registry_workspace();
        workspace.prepare_inputs().unwrap();
        {
            let authority = RegistryAuthority::acquire(
                &mut workspace.registry.stores,
                &mut workspace.authority,
            )
            .unwrap();
            let data = fixture
                .store
                .prepare_buffered_commit_data(
                    authority.transition(&fixture.store).unwrap(),
                    &mut data_workspace,
                )
                .unwrap();
            let released =
                prepare_under_authority(&mut workspace.registry, authority.workspace).unwrap();
            assert!(!scope_available(&fixture.target));
            let held = if held_target {
                fixture.target.read()
            } else {
                fixture.caller.read()
            };
            let data = data.rebind().unwrap();
            allocation_test::start();
            let rejected = released.rebind();
            let error = match rejected {
                Err(error) => Some(error),
                Ok(ready) => {
                    drop(ready);
                    None
                }
            };
            drop(data);
            let traffic = allocation_test::stop();
            assert_eq!(traffic, Counts::default());
            assert!(matches!(error, Some(DataRebindError::Conflict(_))));
            drop(held);
            assert!(scope_available(&fixture.target));
            assert!(fixture.caller.try_write().is_some());
            assert!(fixture.target.try_write().is_some());
            assert!(fixture.store.text_indexes.try_write().is_some());
            let public_error = error.unwrap().into_error();
            assert!(matches!(
                public_error,
                Error::Transaction(TransactionError::WriteConflict(_))
            ));
        }
        fixture
            .store
            .validate_index_registration(&observation)
            .unwrap();
        fixture
            .assert_unpublished()
            .expect("unpublished Text state");
        assert!(fixture.store.tx_property_overlay.read().contains_key(&TX));
        assert!(fixture.store.text_index_overlay.read().contains_key(&TX));
        let mut retry = fixture.registry_workspace();
        drop(prepare_index_registry_batch(&mut retry).unwrap());
        assert!(scope_available(&fixture.target));
    }
}

#[test]
fn forgotten_released_and_final_borrowed_fences_are_cleaned_by_outer_workspace() {
    for final_phase in [false, true] {
        let fixture = Fixture::new();
        let mut workspace = fixture.registry_workspace();
        workspace.prepare_inputs().unwrap();
        let authority =
            RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
                .unwrap();
        let released =
            prepare_under_authority(&mut workspace.registry, authority.workspace).unwrap();
        assert_eq!(held_scope_count(&released.fences), 1);
        assert!(!scope_available(&fixture.target));
        if final_phase {
            let ready = released.rebind().unwrap();
            std::mem::forget(ready);
            assert!(fixture.target.try_read().is_none());
            assert!(fixture.caller.try_read().is_none());
            assert!(fixture.store.text_indexes.try_read().is_none());
        } else {
            std::mem::forget(released);
            assert!(
                workspace.registry.active.is_empty(),
                "released stage still needs scope cleanup"
            );
            assert!(fixture.store.text_indexes.try_write().is_some());
            assert!(!scope_available(&fixture.target));
        }
        std::mem::forget(authority);
        drop(workspace);
        assert!(scope_available(&fixture.target));
        assert!(fixture.target.try_write().is_some());
        assert!(fixture.caller.try_write().is_some());
        assert!(fixture.store.text_indexes.try_write().is_some());
        assert!(fixture.store.property_indexes.try_write().is_some());
        assert!(fixture.store.index_slots.try_lock().is_some());
        // The outer owner also recovers a forgotten authority loan; all guards
        // drain before either registry or topology payload storage is retired.
        fixture
            .assert_unpublished()
            .expect("unpublished Text state");
    }
}
