use super::*;
use crate::allocation_test::{self, Counts};
use crate::index::text::{
    BM25Config, RegisteredTextIndex, TextRegistryBatchFence, TextRegistryFenceWorkspace,
};
use grafeo_common::utils::error::{Error, TransactionError};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const P: EpochId = EpochId::new(5);
const C: EpochId = EpochId::new(6);
const TX: TransactionId = TransactionId::new(42);
const FOREIGN: TransactionId = TransactionId::new(91);

fn workspace(changes: Vec<(NodeId, Option<String>)>) -> TextCommitWorkspace {
    TextCommitWorkspace::new(changes, P, C, TX)
}

fn snapshot(index: &InvertedIndex) -> String {
    let mut postings: Vec<_> = index.postings.iter().collect();
    postings.sort_by(|left, right| left.0.cmp(right.0));
    let mut documents: Vec<_> = index.doc_lengths.iter().collect();
    documents.sort_by_key(|(id, _)| **id);
    format!("{postings:?}|{documents:?}|{:?}", index.agg_log)
}

#[test]
fn sparse_commit_rejects_frontier_below_retained_floor()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut index = InvertedIndex::new(BM25Config::default());
    index.insert_versioned(NodeId::new(1), "alpha", EpochId::new(1), None);
    index.gc(C)?;
    let before = snapshot(&index);
    let mut workspace = workspace(vec![(NodeId::new(1), Some("beta".into()))]);
    let scope = index.pin_commit_scope(&mut workspace)?;
    assert!(index.prepare_commit(&mut workspace, &scope).is_err());
    assert_eq!(snapshot(&index), before);
    assert_eq!(index.retained_from(), C);
    Ok(())
}

#[test]
fn normalized_final_rows_preserve_history_and_configuration_with_zero_install_traffic()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    // Positive controls for all four counters, in this exact unit-test binary.
    allocation_test::start();
    let mut ordinary = Vec::<u64>::with_capacity(1);
    ordinary.push(1);
    ordinary.reserve(200);
    let zeroed = vec![0u64; 200];
    std::hint::black_box(&ordinary);
    std::hint::black_box(&zeroed);
    drop(ordinary);
    drop(zeroed);
    let positive = allocation_test::stop();
    assert!(
        positive.alloc > 0 && positive.zeroed > 0 && positive.realloc > 0 && positive.dealloc > 0
    );

    let mut index = InvertedIndex::with_simple_tokenizer(BM25Config { k1: 2.3, b: 0.4 }, 1);
    let updated = NodeId::new(1);
    let deleted = NodeId::new(2);
    let unchanged = NodeId::new(3);
    let created = NodeId::new(4);
    index.insert_versioned(updated, "old word", EpochId::new(1), None);
    index.insert_versioned(updated, "current word", EpochId::new(3), None);
    index.insert_versioned(deleted, "delete word", EpochId::new(2), None);
    index.insert_versioned(unchanged, "unrelated", EpochId::new(2), None);
    let tokenizer = Arc::clone(&index.tokenizer);
    let unrelated = format!("{:?}", index.postings["unrelated"]);
    let unrelated_pointer = index.postings["unrelated"].postings.as_ptr();
    let mut workspace = workspace(vec![
        (updated, Some("intermediate transient".into())),
        (deleted, Some("must not survive".into())),
        (updated, Some("new new word".into())),
        (deleted, None),
        (created, Some("birth".into())),
    ]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    let before = snapshot(&index);
    let prepared = index.prepare_commit(&mut workspace, &scope).unwrap();
    assert_eq!(
        snapshot(&index),
        before,
        "preparation changes capacity, never logical content"
    );
    allocation_test::start();
    {
        let _installed = prepared.bind(&mut index).unwrap().install();
    }
    drop(scope);
    let observed = allocation_test::stop();
    assert_eq!(observed, Counts::default());

    assert!(Arc::ptr_eq(&tokenizer, &index.tokenizer));
    assert_eq!(index.config.k1, 2.3);
    assert_eq!(index.config.b, 0.4);
    assert_eq!(
        index.postings["unrelated"].postings.as_ptr(),
        unrelated_pointer
    );
    assert_eq!(format!("{:?}", index.postings["unrelated"]), unrelated);
    assert_eq!(index.doc_lengths[&updated].len(), 3);
    assert_eq!(index.doc_lengths[&updated][1].deleted_epoch, Some(C));
    assert_eq!(index.doc_lengths[&updated][2].created_epoch, C);
    assert_eq!(index.doc_lengths[&created][0].created_epoch, C);
    assert_eq!(index.doc_count_at(P, TransactionId::INVALID)?, 3);
    assert_eq!(index.doc_count_at(C, TransactionId::INVALID)?, 3);
    assert_eq!(index.total_length_at(P, TransactionId::INVALID)?, 5);
    assert_eq!(index.total_length_at(C, TransactionId::INVALID)?, 5);
    assert!(!index.postings.contains_key("intermediate"));
    assert!(!index.postings.contains_key("survive"));
    assert_eq!(index.search("new", 10)[0].0, updated);
    assert!(index.search("delete", 10).is_empty());
    assert_eq!(workspace.documents.len(), 3);
    allocation_test::start();
    drop(workspace);
    assert!(
        allocation_test::stop().dealloc > 0,
        "outer owner retires scratch after fences"
    );
    Ok(())
}

#[test]
fn foreign_pending_insert_suffixes_and_aggregate_entries_remain_exact()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut index = InvertedIndex::new(BM25Config::default());
    let id = NodeId::new(1);
    let foreign = NodeId::new(2);
    index.insert_versioned(id, "shared old", EpochId::new(1), None);
    index.insert_versioned(foreign, "shared foreign", EpochId::PENDING, Some(FOREIGN));
    // A concurrent creation of the touched identity is retained too. It has
    // no deletion claim over the committed document being closed at C.
    index
        .doc_lengths
        .get_mut(&id)
        .unwrap()
        .push(VersionedDocLen::new(1, EpochId::PENDING, Some(FOREIGN)));
    index
        .postings
        .get_mut("shared")
        .unwrap()
        .postings
        .push(VersionedPosting::new(
            id,
            1,
            EpochId::PENDING,
            Some(FOREIGN),
        ));
    index.agg_log.push(AggDelta {
        epoch: EpochId::PENDING,
        tx: Some(FOREIGN),
        d_total_len: 1,
        d_doc_count: 1,
    });
    let pending_postings: Vec<_> = index.postings["shared"]
        .postings
        .iter()
        .filter(|posting| posting.created_by == Some(FOREIGN))
        .map(|posting| format!("{posting:?}"))
        .collect();
    let pending_lengths = format!("{:?}", index.doc_lengths[&id][1]);
    let pending_aggregates: Vec<_> = index
        .agg_log
        .iter()
        .filter(|delta| delta.tx == Some(FOREIGN))
        .map(|delta| format!("{delta:?}"))
        .collect();
    let mut workspace = workspace(vec![(id, Some("shared shared new".into()))]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    let ready = index.prepare_commit(&mut workspace, &scope).unwrap();
    allocation_test::start();
    {
        let _consumed_proof = ready.bind(&mut index).unwrap().install();
    }
    let observed = allocation_test::stop();
    assert_eq!(observed, Counts::default());
    drop(scope);
    assert_eq!(index.doc_lengths[&id][1].created_epoch, C);
    assert_eq!(format!("{:?}", index.doc_lengths[&id][2]), pending_lengths);
    let list = &index.postings["shared"].postings;
    assert_eq!(list[1].created_epoch, C);
    assert_eq!(
        list.iter()
            .filter(|posting| posting.created_by == Some(FOREIGN))
            .map(|posting| format!("{posting:?}"))
            .collect::<Vec<_>>(),
        pending_postings
    );
    assert_eq!(
        index
            .agg_log
            .iter()
            .filter(|delta| delta.tx == Some(FOREIGN))
            .map(|delta| format!("{delta:?}"))
            .collect::<Vec<_>>(),
        pending_aggregates
    );
    let first_pending = index
        .agg_log
        .iter()
        .position(|delta| delta.tx.is_some())
        .unwrap();
    assert_eq!(index.agg_log[first_pending - 1].epoch, C);
    assert_eq!(index.total_length_at(P, TransactionId::INVALID)?, 2);
    assert_eq!(index.total_length_at(C, TransactionId::INVALID)?, 3);
    Ok(())
}

#[test]
fn sparse_scratch_does_not_clone_unrelated_posting_histories() {
    let mut index = InvertedIndex::new(BM25Config::default());
    for value in 0..256 {
        index.insert_versioned(
            NodeId::new(value),
            "common untouched",
            EpochId::new(1),
            None,
        );
    }
    let shared_history_pointer = index.postings["common"].postings.as_ptr();
    let shared_history_capacity = index.postings["common"].postings.capacity();
    let mut workspace = workspace(vec![(NodeId::new(0), None)]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    {
        let _prepared = index.prepare_commit(&mut workspace, &scope).unwrap();
    }
    assert_eq!(workspace.documents.len(), 1);
    assert_eq!(workspace.terms.len(), 2);
    assert_eq!(
        workspace
            .terms
            .iter()
            .map(|term| term.close.len())
            .sum::<usize>(),
        2
    );
    assert!(
        workspace
            .terms
            .iter()
            .all(|term| term.additions.postings.is_empty())
    );
    assert_eq!(
        index.postings["common"].postings.as_ptr(),
        shared_history_pointer
    );
    assert_eq!(
        index.postings["common"].postings.capacity(),
        shared_history_capacity
    );
}

#[test]
fn touched_pending_deletes_and_own_write_through_are_rejected_without_changes() {
    for owner in [TX, FOREIGN] {
        let mut index = InvertedIndex::new(BM25Config::default());
        let id = NodeId::new(1);
        index.insert_versioned(id, "committed", EpochId::new(1), None);
        index.remove_versioned(id, EpochId::PENDING, Some(owner));
        let before = snapshot(&index);
        let mut workspace = workspace(vec![(id, Some("replacement".into()))]);
        let scope = index.pin_commit_scope(&mut workspace).unwrap();
        let Err(error) = index.prepare_commit(&mut workspace, &scope) else {
            panic!("pending deletion was overwritten");
        };
        assert_eq!(snapshot(&index), before);
        if owner == FOREIGN {
            assert!(matches!(
                error,
                Error::Transaction(TransactionError::WriteConflict(_))
            ));
        }
        assert_eq!(workspace.documents[0].tokens, vec!["replacement"]);
    }
    let mut index = InvertedIndex::new(BM25Config::default());
    let id = NodeId::new(1);
    index.insert_versioned(id, "own pending", EpochId::PENDING, Some(TX));
    let before = snapshot(&index);
    let mut workspace = workspace(vec![(id, None)]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    assert!(index.prepare_commit(&mut workspace, &scope).is_err());
    assert_eq!(snapshot(&index), before);
}

#[test]
fn late_reservation_failures_abandonment_and_bind_rejection_have_no_logical_effect() {
    let mut saw_late = false;
    let mut reached_success = false;
    for point in 0..96 {
        let mut index = InvertedIndex::new(BM25Config::default());
        let id = NodeId::new(1);
        index.insert_versioned(id, "old terms", EpochId::new(1), None);
        let before = snapshot(&index);
        let mut workspace = workspace(vec![
            (id, Some("new terms".into())),
            (NodeId::new(2), Some("birth".into())),
        ]);
        let scope = index.pin_commit_scope(&mut workspace).unwrap();
        FAIL_RESERVATION.with(|failure| failure.set(Some(point)));
        let result = index.prepare_commit(&mut workspace, &scope);
        FAIL_RESERVATION.with(|failure| failure.set(None));
        let succeeded = result.is_ok();
        drop(result);
        assert_eq!(snapshot(&index), before);
        saw_late |= !succeeded && !workspace.terms.is_empty() && !workspace.documents.is_empty();
        if succeeded {
            reached_success = true;
            break;
        }
    }
    assert!(saw_late && reached_success);

    let mut index = InvertedIndex::new(BM25Config::default());
    let mut other = InvertedIndex::new(BM25Config::default());
    let mut workspace = workspace(vec![(NodeId::new(1), Some("new text".into()))]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    let ready = index.prepare_commit(&mut workspace, &scope).unwrap();
    allocation_test::start();
    let Err(error) = ready.bind(&mut other) else {
        panic!("foreign scope bound");
    };
    let observed = allocation_test::stop();
    assert_eq!(observed, Counts::default());
    assert!(matches!(error, DataRebindError::Invalid(_)));
    assert!(index.postings.is_empty() && other.postings.is_empty());
}

#[test]
fn scope_excludes_alias_mutation_and_collective_conflict_does_not_deadlock() {
    let gate = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
    let target = gate.write().pin_registry_target();
    let registration = RegisteredTextIndex::new(Arc::clone(&gate), Arc::clone(&target));
    let mut workspace = workspace(vec![(NodeId::new(1), Some("prepared".into()))]);
    let scope = target.read().pin_commit_scope(&mut workspace).unwrap();
    {
        let _ready = target
            .write()
            .prepare_commit(&mut workspace, &scope)
            .unwrap();
        let mut fences = TextRegistryFenceWorkspace::new();
        fences.prepare(&[registration]).unwrap();
        // Model an ordinary alias that owns caller+target before entering its
        // mutation gate. Nonblocking collective acquisition rejects it promptly.
        let caller_writer = gate.write();
        let target_writer = target.write();
        assert!(target_writer.mutation_scope_gate.try_read().is_none());
        allocation_test::start();
        let result = TextRegistryBatchFence::try_acquire(&mut fences);
        assert!(matches!(result, Err(DataRebindError::Conflict(_))));
        drop(result);
        let observed = allocation_test::stop();
        assert_eq!(observed, Counts::default());
        drop(target_writer);
        drop(caller_writer);
        allocation_test::start();
        drop(TextRegistryBatchFence::try_acquire(&mut fences).unwrap());
        let observed = allocation_test::stop();
        assert_eq!(observed, Counts::default());
    }
    drop(scope);
    assert!(target.read().mutation_scope_gate.try_read().is_some());
}

struct ProbeTokenizer {
    outer: Arc<RwLock<()>>,
    dropped: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    panic_at: Option<usize>,
}

impl Tokenizer for ProbeTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        assert_ne!(Some(call), self.panic_at, "injected tokenizer unwind");
        text.split_whitespace().map(str::to_owned).collect()
    }
}

impl Drop for ProbeTokenizer {
    fn drop(&mut self) {
        assert!(
            self.outer.try_write().is_some(),
            "tokenizer retired under outer gate"
        );
        self.dropped.store(true, Ordering::Relaxed);
    }
}

#[test]
fn partial_tokenization_and_tokenizer_anchors_survive_outer_gate_unwind() {
    let outer = Arc::new(RwLock::new(()));
    let dropped = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut index = InvertedIndex::with_tokenizer(
        BM25Config::default(),
        Box::new(ProbeTokenizer {
            outer: Arc::clone(&outer),
            dropped: Arc::clone(&dropped),
            calls: Arc::clone(&calls),
            panic_at: Some(1),
        }),
    );
    let mut workspace = workspace(vec![
        (NodeId::new(1), Some("first payload".into())),
        (NodeId::new(2), Some("second payload".into())),
    ]);
    let outer_guard = outer.write();
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = index.prepare_commit(&mut workspace, &scope);
    }));
    assert!(outcome.is_err());
    assert_eq!(workspace.documents[0].tokens, vec!["first", "payload"]);
    assert_eq!(workspace.inputs.len(), 2);
    drop(scope);
    drop(index);
    assert!(
        !dropped.load(Ordering::Relaxed),
        "workspace anchors exact tokenizer after rejection"
    );
    drop(outer_guard);
    drop(workspace);
    assert!(dropped.load(Ordering::Relaxed));
}

#[test]
fn empty_token_output_removes_existing_document_without_initial_birth()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut index = InvertedIndex::new(BM25Config::default());
    let id = NodeId::new(1);
    index.insert_versioned(id, "existing", EpochId::new(1), None);
    let mut workspace = workspace(vec![
        (id, Some("the a".into())),
        (NodeId::new(2), Some(String::new())),
    ]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    let ready = index.prepare_commit(&mut workspace, &scope).unwrap();
    allocation_test::start();
    {
        let _consumed_proof = ready.bind(&mut index).unwrap().install();
    }
    assert_eq!(allocation_test::stop(), Counts::default());
    assert_eq!(index.doc_count_at(P, TransactionId::INVALID)?, 1);
    assert_eq!(index.doc_count_at(C, TransactionId::INVALID)?, 0);
    assert_eq!(index.doc_lengths[&id].len(), 1);
    assert!(!index.doc_lengths.contains_key(&NodeId::new(2)));
    Ok(())
}

#[test]
fn installed_abandoned_and_rejected_candidates_retain_tokenizer_beyond_outer_gate() {
    for outcome in 0..3 {
        let outer = Arc::new(RwLock::new(()));
        let dropped = Arc::new(AtomicBool::new(false));
        let mut index = InvertedIndex::with_tokenizer(
            BM25Config::default(),
            Box::new(ProbeTokenizer {
                outer: Arc::clone(&outer),
                dropped: Arc::clone(&dropped),
                calls: Arc::new(AtomicUsize::new(0)),
                panic_at: None,
            }),
        );
        let mut workspace = workspace(vec![(NodeId::new(1), Some("retained tokens".into()))]);
        let outer_guard = outer.write();
        let scope = index.pin_commit_scope(&mut workspace).unwrap();
        if outcome == 2 {
            // Tokenization and completed document fragments precede aggregate
            // validation; its late failure must retain all such payloads.
            index.agg_log.push(AggDelta {
                epoch: EpochId::PENDING,
                tx: Some(TX),
                d_total_len: 1,
                d_doc_count: 1,
            });
            assert!(index.prepare_commit(&mut workspace, &scope).is_err());
        } else {
            {
                let ready = index.prepare_commit(&mut workspace, &scope).unwrap();
                allocation_test::start();
                if outcome == 0 {
                    let _installed = ready.bind(&mut index).unwrap().install();
                }
            }
            assert_eq!(allocation_test::stop(), Counts::default());
        }
        drop(scope);
        drop(index);
        assert!(!dropped.load(Ordering::Relaxed));
        assert_eq!(workspace.documents[0].tokens.len(), 2);
        drop(outer_guard);
        drop(workspace);
        assert!(dropped.load(Ordering::Relaxed));
    }
}

#[test]
fn scope_identity_survives_concrete_value_move_and_rejects_lost_capacity() {
    let mut index = InvertedIndex::new(BM25Config::default());
    let mut workspace = workspace(vec![(NodeId::new(1), Some("new text".into()))]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    let ready = index.prepare_commit(&mut workspace, &scope).unwrap();
    let mut moved = index;
    allocation_test::start();
    {
        let _consumed_proof = ready.bind(&mut moved).unwrap().install();
    }
    assert_eq!(allocation_test::stop(), Counts::default());
    drop(scope);
    assert_eq!(moved.search("text", 10)[0].0, NodeId::new(1));

    let mut workspace = TextCommitWorkspace::new(
        vec![(NodeId::new(1), Some("text changed".into()))],
        C,
        EpochId::new(7),
        TX,
    );
    let scope = moved.pin_commit_scope(&mut workspace).unwrap();
    let ready = moved.prepare_commit(&mut workspace, &scope).unwrap();
    // Private corruption of capacity is not a legal alias mutation, but must
    // still reject before marker without allocating or changing logical rows.
    moved
        .postings
        .get_mut("text")
        .unwrap()
        .postings
        .shrink_to_fit();
    let before = snapshot(&moved);
    allocation_test::start();
    let result = ready.bind(&mut moved);
    assert!(matches!(result, Err(DataRebindError::Invalid(_))));
    drop(result);
    assert_eq!(allocation_test::stop(), Counts::default());
    assert_eq!(snapshot(&moved), before);
}

#[test]
fn stale_publication_frontier_is_rejected_before_any_install() {
    let mut index = InvertedIndex::new(BM25Config::default());
    let id = NodeId::new(1);
    index.insert_versioned(id, "future row", C, None);
    let before = snapshot(&index);
    let mut workspace = workspace(vec![(id, None)]);
    let scope = index.pin_commit_scope(&mut workspace).unwrap();
    let Err(error) = index.prepare_commit(&mut workspace, &scope) else {
        panic!("stale frontier accepted");
    };
    assert!(matches!(
        error,
        Error::Transaction(TransactionError::WriteConflict(_))
    ));
    assert_eq!(snapshot(&index), before);
}
