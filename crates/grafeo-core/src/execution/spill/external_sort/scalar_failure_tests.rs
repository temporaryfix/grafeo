//! Real push-finalize scalar fallback controls; all fixture APIs predate the fix.

use crate::execution::operators::push::{SortKey, SpillableSortPushOperator};
use crate::execution::operators::{AccountedFailureClassification, OperatorError};
use crate::execution::pipeline::PushOperator;
use crate::execution::sink::CollectorSink;
use crate::execution::spill::{
    BorrowedSpillFixture, CleartextSpillRecordProvider, OpenSpillRecord, SpillFileIdentity,
    SpillFrameLimits, SpillIo, SpillIoOperation, SpillRecordMeta, SpillRecordProvider,
};
use crate::execution::{DataChunk, QueryExecutionControl, ValueVector};
use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
use grafeo_common::types::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const PAYLOAD_BYTES: usize = 8192;
const PROVIDER_BOUND: usize = PAYLOAD_BYTES + 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Success,
    Construction,
    Data,
    Finish,
    IntermediateData,
}

struct State {
    memory: Arc<BufferManager>,
    phase: Phase,
    panic: bool,
    cleanup_fault: bool,
    armed: AtomicBool,
    hits: AtomicUsize,
    drops: AtomicUsize,
    display_calls: AtomicUsize,
    payload_address: AtomicUsize,
    dropped_address: AtomicUsize,
    drop_charge: AtomicUsize,
    cleanup_hits: AtomicUsize,
    data_reads: AtomicUsize,
    intermediate_rows: AtomicUsize,
}

impl State {
    fn fail(self: &Arc<Self>, phase: Phase) -> std::io::Result<()> {
        if !self.armed.load(Ordering::Relaxed) || self.phase != phase {
            return Ok(());
        }
        if self
            .hits
            .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return Ok(());
        }
        let payload = Payload {
            bytes: Box::new([0x6d; PAYLOAD_BYTES]),
            state: Arc::clone(self),
        };
        self.payload_address
            .store(payload.bytes.as_ptr() as usize, Ordering::Relaxed);
        if self.panic {
            std::panic::panic_any(payload);
        }
        Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, payload))
    }
}

struct Payload {
    bytes: Box<[u8; PAYLOAD_BYTES]>,
    state: Arc<State>,
}

impl std::fmt::Debug for Payload {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output.write_str("ScalarProviderPayload")
    }
}

impl std::fmt::Display for Payload {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.state.display_calls.fetch_add(1, Ordering::Relaxed);
        output.write_str("original scalar provider primary")
    }
}

impl std::error::Error for Payload {}

impl Drop for Payload {
    fn drop(&mut self) {
        assert_eq!(self.bytes[0], 0x6d);
        self.state
            .drop_charge
            .store(self.state.memory.allocated(), Ordering::Relaxed);
        self.state
            .dropped_address
            .store(self.bytes.as_ptr() as usize, Ordering::Relaxed);
        self.state.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct Provider(Arc<State>);

impl SpillRecordProvider for Provider {
    fn seals(&self) -> bool {
        false
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        // One record wrapper plus its original heap-bearing error/panic and
        // erased I/O carrier. Shared fixture state was allocated before calls.
        Some(PROVIDER_BOUND)
    }

    fn begin_file(&self, identity: SpillFileIdentity) -> std::io::Result<Box<dyn OpenSpillRecord>> {
        self.0.fail(Phase::Construction)?;
        Ok(Box::new(Record {
            inner: CleartextSpillRecordProvider.begin_file(identity)?,
            state: Arc::clone(&self.0),
        }))
    }
}

struct Record {
    inner: Box<dyn OpenSpillRecord>,
    state: Arc<State>,
}

impl OpenSpillRecord for Record {
    fn stored_len(&self, length: usize) -> std::io::Result<usize> {
        self.inner.stored_len(length)
    }

    fn seal_allocation_bound(&self, length: usize) -> Option<usize> {
        self.inner.seal_allocation_bound(length)
    }

    fn open_allocation_bound(&self, length: usize) -> Option<usize> {
        self.inner
            .open_allocation_bound(length)
            .map(|bound| bound.max(PROVIDER_BOUND))
    }

    fn seal(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        plaintext: &[u8],
    ) -> std::io::Result<Vec<u8>> {
        let bytes = self.inner.seal(meta, aad, plaintext)?;
        if self.state.armed.load(Ordering::Relaxed)
            && meta.kind() == super::super::SpillRecordKind::SortRow
        {
            self.state.intermediate_rows.fetch_add(1, Ordering::Relaxed);
        }
        Ok(bytes)
    }

    fn open(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        stored: &[u8],
    ) -> std::io::Result<Vec<u8>> {
        if self.state.armed.load(Ordering::Relaxed) {
            match meta.kind() {
                super::super::SpillRecordKind::SortRow => {
                    self.state.data_reads.fetch_add(1, Ordering::Relaxed);
                    self.state.fail(Phase::Data)?;
                    if self.state.intermediate_rows.load(Ordering::Relaxed) > 0 {
                        self.state.fail(Phase::IntermediateData)?;
                    }
                }
                super::super::SpillRecordKind::FileEnd => self.state.fail(Phase::Finish)?,
                _ => {}
            }
        }
        self.inner.open(meta, aad, stored)
    }
}

impl SpillIo for State {
    fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
        Some(0)
    }

    fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        Some(0)
    }

    fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
        if self.cleanup_fault
            && self.hits.load(Ordering::Relaxed) == 1
            && operation == SpillIoOperation::Delete
            && self
                .cleanup_hits
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            return Err(std::io::ErrorKind::PermissionDenied.into());
        }
        Ok(())
    }
}

fn chunk(values: &[i64]) -> DataChunk {
    DataChunk::new(vec![ValueVector::from_values(
        &values.iter().copied().map(Value::Int64).collect::<Vec<_>>(),
    )])
}

fn run(phase: Phase, panic: bool, cleanup_fault: bool, runs: usize) {
    let mut config = BufferManagerConfig::with_budget(3 << 20);
    config.soft_limit_fraction = 1.0;
    config.evict_limit_fraction = 1.0;
    config.hard_limit_fraction = 1.0;
    let memory = BufferManager::new(config);
    let state = Arc::new(State {
        memory: Arc::clone(&memory),
        phase,
        panic,
        cleanup_fault,
        armed: AtomicBool::new(false),
        hits: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        display_calls: AtomicUsize::new(0),
        payload_address: AtomicUsize::new(0),
        dropped_address: AtomicUsize::new(0),
        drop_charge: AtomicUsize::new(0),
        cleanup_hits: AtomicUsize::new(0),
        data_reads: AtomicUsize::new(0),
        intermediate_rows: AtomicUsize::new(0),
    });
    let directory = tempfile::tempdir().unwrap();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(Provider(Arc::clone(&state))),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::clone(&state) as Arc<dyn SpillIo>)
        .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
        .unwrap();
    let mut sort = SpillableSortPushOperator::with_resource_context(
        vec![SortKey::ascending(0)],
        resources.clone(),
    )
    .unwrap();
    // CollectorSink offers no exact handoff. The resident tail independently
    // forces finalize through merge_cursor_accounted_observing as well.
    let mut sink = CollectorSink::new();
    for _ in 0..runs {
        sort.push(chunk(&[3, 1, 1]), &mut sink).unwrap();
        sort.flush_pull_batch().unwrap();
    }
    assert_eq!(manager.active_file_count(), runs);
    sort.push(chunk(&[2]), &mut sink).unwrap();
    assert_eq!(sort.pull_buffered_rows(), 1);
    state.armed.store(true, Ordering::Relaxed);
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sort.finalize(&mut sink)));
    drop(sort);
    drop(resources);
    if phase == Phase::Success {
        outcome.unwrap().unwrap();
        let actual: Vec<_> = sink
            .chunks()
            .iter()
            .flat_map(|chunk| {
                (0..chunk.len()).map(move |row| chunk.column(0).unwrap().get_value(row).unwrap())
            })
            .collect();
        let mut expected = vec![Value::Int64(1); 2 * runs];
        expected.push(Value::Int64(2));
        expected.extend(vec![Value::Int64(3); runs]);
        assert_eq!(actual, expected);
        assert_eq!(state.hits.load(Ordering::Relaxed), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(memory.allocated(), 0);
        if runs > 16 {
            assert!(state.intermediate_rows.load(Ordering::Relaxed) > 0);
        }
        eprintln!(
            "scalar finalize success: initial_runs={runs}, exact_rows={}",
            actual.len()
        );
        return;
    }
    drop(sink);
    assert_eq!(state.hits.load(Ordering::Relaxed), 1, "{phase:?}");
    assert_eq!(
        state.drops.load(Ordering::Relaxed),
        0,
        "original {phase:?} payload must outlive operator teardown"
    );
    assert!(memory.allocated() >= PROVIDER_BOUND, "{phase:?}");
    if phase == Phase::IntermediateData {
        assert!(runs > 16);
        assert!(state.intermediate_rows.load(Ordering::Relaxed) > 0);
        assert!(state.data_reads.load(Ordering::Relaxed) > 1);
    }
    assert_eq!(
        state.cleanup_hits.load(Ordering::Relaxed),
        usize::from(cleanup_fault)
    );
    manager.cleanup().unwrap();
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.spilled_bytes(), 0);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    assert!(memory.allocated() >= PROVIDER_BOUND);

    // Both forms compile against the parent. Its ordinary returned I/O error
    // destroys the payload too soon, and its escaped panic loses the grant.
    // The repaired path must use the already-existing accounted diagnostic.
    let Ok(Err(error)) = &outcome else {
        panic!("{phase:?} must escape as a retained operator diagnostic");
    };
    assert!(matches!(
        error,
        OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::Execution,
            ..
        }
    ));
    let _ = error.to_string();
    assert_eq!(state.display_calls.load(Ordering::Relaxed), 0);
    drop(outcome);
    assert_eq!(state.drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        state.dropped_address.load(Ordering::Relaxed),
        state.payload_address.load(Ordering::Relaxed),
        "the original allocation must be destroyed exactly once"
    );
    assert!(state.drop_charge.load(Ordering::Relaxed) >= PROVIDER_BOUND);
    assert_eq!(
        memory.allocated(),
        0,
        "authority releases after payload destruction"
    );
    eprintln!(
        "scalar finalize accepted: phase={phase:?}, panic={panic}, cleanup={cleanup_fault}, initial_runs={runs}, reads={}, intermediate_rows={}, payload_drop_charge={}",
        state.data_reads.load(Ordering::Relaxed),
        state.intermediate_rows.load(Ordering::Relaxed),
        state.drop_charge.load(Ordering::Relaxed),
    );
}

fn failure_cases(phase: Phase) {
    for panic in [false, true] {
        for cleanup in [false, true] {
            run(
                phase,
                panic,
                cleanup,
                if phase == Phase::IntermediateData {
                    17
                } else {
                    1
                },
            );
        }
    }
}

#[test]
fn scalar_push_finalize_retains_construction_failure_after_operator_drop() {
    failure_cases(Phase::Construction);
}

#[test]
fn scalar_push_finalize_retains_data_failure_after_operator_drop() {
    failure_cases(Phase::Data);
}

#[test]
fn scalar_push_finalize_retains_finish_failure_after_operator_drop() {
    failure_cases(Phase::Finish);
}

#[test]
fn scalar_push_finalize_retains_intermediate_failure_after_operator_drop() {
    failure_cases(Phase::IntermediateData);
}

#[test]
fn scalar_push_finalize_preserves_exact_duplicate_rows_and_intermediate_output() {
    for runs in [1, 17] {
        run(Phase::Success, false, false, runs);
    }
}

#[test]
fn scalar_push_finalize_quarantines_panicking_payload_drop_and_keeps_primary_classification() {
    const BOUND: usize = 2 * PROVIDER_BOUND;

    struct DropState {
        memory: Arc<BufferManager>,
        armed: AtomicBool,
        hits: AtomicUsize,
        primary_drops: AtomicUsize,
        secondary_drops: AtomicUsize,
        drop_charge: AtomicUsize,
        cleanup_hits: AtomicUsize,
    }

    struct SecondaryPanic {
        _bytes: Box<[u8; 1024]>,
        state: Arc<DropState>,
    }

    impl Drop for SecondaryPanic {
        fn drop(&mut self) {
            self.state.secondary_drops.fetch_add(1, Ordering::Relaxed);
            panic!("quarantined secondary must not be destructed during primary cleanup");
        }
    }

    struct Primary {
        bytes: Box<[u8; PAYLOAD_BYTES]>,
        state: Arc<DropState>,
    }

    impl std::fmt::Debug for Primary {
        fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            output.write_str("ScalarStorageFullPrimary")
        }
    }

    impl std::fmt::Display for Primary {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("opaque primary must not be formatted to publish its error");
        }
    }

    impl std::error::Error for Primary {}

    impl Drop for Primary {
        fn drop(&mut self) {
            assert_eq!(self.bytes[0], 0x4f);
            self.state.primary_drops.fetch_add(1, Ordering::Relaxed);
            self.state
                .drop_charge
                .store(self.state.memory.allocated(), Ordering::Relaxed);
            // The primary allocation unwinds normally; its new opaque panic
            // allocation must remain quarantined under the same admission.
            std::panic::panic_any(SecondaryPanic {
                _bytes: Box::new([0x7e; 1024]),
                state: Arc::clone(&self.state),
            });
        }
    }

    struct DropProvider(Arc<DropState>);

    impl SpillRecordProvider for DropProvider {
        fn seals(&self) -> bool {
            false
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            Some(BOUND)
        }

        fn begin_file(
            &self,
            identity: SpillFileIdentity,
        ) -> std::io::Result<Box<dyn OpenSpillRecord>> {
            if self.0.armed.load(Ordering::Relaxed)
                && self
                    .0
                    .hits
                    .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::StorageFull,
                    Primary {
                        bytes: Box::new([0x4f; PAYLOAD_BYTES]),
                        state: Arc::clone(&self.0),
                    },
                ));
            }
            CleartextSpillRecordProvider.begin_file(identity)
        }
    }

    impl SpillIo for DropState {
        fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
            Some(0)
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            Some(0)
        }

        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            if self.hits.load(Ordering::Relaxed) == 1
                && operation == SpillIoOperation::Delete
                && self
                    .cleanup_hits
                    .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                return Err(std::io::ErrorKind::PermissionDenied.into());
            }
            Ok(())
        }
    }

    let memory = BufferManager::with_budget(3 << 20);
    let state = Arc::new(DropState {
        memory: Arc::clone(&memory),
        armed: AtomicBool::new(false),
        hits: AtomicUsize::new(0),
        primary_drops: AtomicUsize::new(0),
        secondary_drops: AtomicUsize::new(0),
        drop_charge: AtomicUsize::new(0),
        cleanup_hits: AtomicUsize::new(0),
    });
    let directory = tempfile::tempdir().unwrap();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(DropProvider(Arc::clone(&state))),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::clone(&state) as Arc<dyn SpillIo>)
        .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
        .unwrap();
    let mut sort =
        SpillableSortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
            .unwrap();
    let mut sink = CollectorSink::new();
    sort.push(chunk(&[3, 1, 1]), &mut sink).unwrap();
    sort.flush_pull_batch().unwrap();
    assert_eq!(manager.active_file_count(), 1);
    sort.push(chunk(&[2]), &mut sink).unwrap();
    state.armed.store(true, Ordering::Relaxed);
    let error = sort.finalize(&mut sink).unwrap_err();
    assert!(matches!(
        &error,
        OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::StorageFull,
            ..
        }
    ));
    assert_eq!(state.hits.load(Ordering::Relaxed), 1);
    assert_eq!(state.cleanup_hits.load(Ordering::Relaxed), 1);
    drop(sort);
    drop(sink);
    assert_eq!(state.primary_drops.load(Ordering::Relaxed), 0);
    assert!(memory.allocated() >= BOUND);
    // Cloning only the accounted handle cannot duplicate the physical payload.
    let last = error.clone();
    drop(error);
    assert_eq!(state.primary_drops.load(Ordering::Relaxed), 0);
    let destroyed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(last)));
    assert!(destroyed.is_ok(), "opaque destructor panic is contained");
    drop(destroyed);
    assert_eq!(state.primary_drops.load(Ordering::Relaxed), 1);
    assert_eq!(state.secondary_drops.load(Ordering::Relaxed), 0);
    assert!(state.drop_charge.load(Ordering::Relaxed) >= BOUND);
    assert!(
        memory.allocated() >= BOUND,
        "failed teardown retains its authority"
    );
    manager.cleanup().unwrap();
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.spilled_bytes(), 0);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    drop(manager);
    assert_eq!(state.primary_drops.load(Ordering::Relaxed), 1);
    assert_eq!(state.secondary_drops.load(Ordering::Relaxed), 0);
    assert!(memory.allocated() >= BOUND);
}

#[test]
fn scalar_push_finalize_quarantines_successful_reader_destructor_before_releasing_frontier() {
    struct ReaderState {
        memory: Arc<BufferManager>,
        armed: AtomicBool,
        data: AtomicUsize,
        finish: AtomicUsize,
        reader_drops: AtomicUsize,
        payload_drops: AtomicUsize,
        drop_charge: AtomicUsize,
    }

    struct ReaderDropPanic {
        _bytes: Box<[u8; PAYLOAD_BYTES]>,
        state: Arc<ReaderState>,
    }

    impl Drop for ReaderDropPanic {
        fn drop(&mut self) {
            self.state.payload_drops.fetch_add(1, Ordering::Relaxed);
            panic!("reader destructor panic payload must stay quarantined");
        }
    }

    struct ReaderRecord {
        inner: Box<dyn OpenSpillRecord>,
        state: Arc<ReaderState>,
    }

    impl OpenSpillRecord for ReaderRecord {
        fn stored_len(&self, length: usize) -> std::io::Result<usize> {
            self.inner.stored_len(length)
        }

        fn seal_allocation_bound(&self, length: usize) -> Option<usize> {
            self.inner.seal_allocation_bound(length)
        }

        fn open_allocation_bound(&self, length: usize) -> Option<usize> {
            self.inner.open_allocation_bound(length)
        }

        fn seal(
            &mut self,
            meta: &SpillRecordMeta,
            aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner.seal(meta, aad, plaintext)
        }

        fn open(
            &mut self,
            meta: &SpillRecordMeta,
            aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            let bytes = self.inner.open(meta, aad, stored)?;
            match meta.kind() {
                super::super::SpillRecordKind::SortRow => {
                    self.state.data.fetch_add(1, Ordering::Relaxed);
                }
                super::super::SpillRecordKind::FileEnd => {
                    self.state.finish.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
            Ok(bytes)
        }
    }

    impl Drop for ReaderRecord {
        fn drop(&mut self) {
            self.state.reader_drops.fetch_add(1, Ordering::Relaxed);
            self.state
                .drop_charge
                .store(self.state.memory.allocated(), Ordering::Relaxed);
            std::panic::panic_any(ReaderDropPanic {
                _bytes: Box::new([0xa3; PAYLOAD_BYTES]),
                state: Arc::clone(&self.state),
            });
        }
    }

    struct ReaderProvider(Arc<ReaderState>);

    impl SpillRecordProvider for ReaderProvider {
        fn seals(&self) -> bool {
            false
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            // The record's destructor and its escaping panic allocation are
            // part of the existing file-lifetime provider contract.
            Some(PROVIDER_BOUND)
        }

        fn begin_file(
            &self,
            identity: SpillFileIdentity,
        ) -> std::io::Result<Box<dyn OpenSpillRecord>> {
            let inner = CleartextSpillRecordProvider.begin_file(identity)?;
            if self.0.armed.load(Ordering::Relaxed) {
                Ok(Box::new(ReaderRecord {
                    inner,
                    state: Arc::clone(&self.0),
                }))
            } else {
                Ok(inner)
            }
        }
    }

    let memory = BufferManager::with_budget(3 << 20);
    let state = Arc::new(ReaderState {
        memory: Arc::clone(&memory),
        armed: AtomicBool::new(false),
        data: AtomicUsize::new(0),
        finish: AtomicUsize::new(0),
        reader_drops: AtomicUsize::new(0),
        payload_drops: AtomicUsize::new(0),
        drop_charge: AtomicUsize::new(0),
    });
    let directory = tempfile::tempdir().unwrap();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(ReaderProvider(Arc::clone(&state))),
            SpillFrameLimits::format_max(),
        )
        .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
        .unwrap();
    let mut sort =
        SpillableSortPushOperator::with_resource_context(vec![SortKey::ascending(0)], resources)
            .unwrap();
    let mut sink = CollectorSink::new();
    sort.push(chunk(&[3, 1, 1]), &mut sink).unwrap();
    sort.flush_pull_batch().unwrap();
    assert_eq!(manager.active_file_count(), 1);
    sort.push(chunk(&[2]), &mut sink).unwrap();
    state.armed.store(true, Ordering::Relaxed);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sort.finalize(&mut sink)));
    let result = match result {
        Ok(result) => result,
        Err(payload) => {
            std::mem::forget(payload);
            panic!("provider destructor panic must be contained");
        }
    };
    let error = result.expect_err("successful frame reads cannot hide failed reader destruction");
    assert!(matches!(
        &error,
        OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::Execution,
            ..
        }
    ));
    assert_eq!(state.data.load(Ordering::Relaxed), 3);
    assert_eq!(state.finish.load(Ordering::Relaxed), 1);
    assert_eq!(state.reader_drops.load(Ordering::Relaxed), 1);
    assert_eq!(state.payload_drops.load(Ordering::Relaxed), 0);
    assert!(state.drop_charge.load(Ordering::Relaxed) >= PROVIDER_BOUND);
    drop(sort);
    drop(sink);
    assert!(memory.allocated() >= PROVIDER_BOUND);
    let destroyed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(error)));
    if let Err(payload) = destroyed {
        std::mem::forget(payload);
        panic!("accounted reader diagnostic destruction must contain opaque panics");
    }
    assert_eq!(state.reader_drops.load(Ordering::Relaxed), 1);
    assert_eq!(state.payload_drops.load(Ordering::Relaxed), 0);
    assert!(memory.allocated() >= PROVIDER_BOUND);
    manager.cleanup().unwrap();
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.spilled_bytes(), 0);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    assert!(memory.allocated() >= PROVIDER_BOUND);
}
