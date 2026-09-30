//! Real resource-aggregate spill/revisit controls using the existing spill fixture.
#![cfg(feature = "spill")]

use super::{AggregateExpr, GroupState, SpillableAggregatePushOperator};
use crate::execution::AccountedDataChunk;
use crate::execution::operators::{AccountedFailureClassification, OperatorError};
use crate::execution::pipeline::{
    AccountedSinkPermit, PushOperator, Sink, qualified_accounted_transport::QualifiedSink,
};
use crate::execution::sink::{CollectorSink, CountingSink};
use crate::execution::spill::{
    BorrowedSpillFixture, CleartextSpillRecordProvider, OpenSpillRecord, PartitionOperationError,
    PartitionedState, SpillFileIdentity, SpillFrameLimits, SpillIo, SpillIoOperation,
    SpillRecordKind, SpillRecordMeta, SpillRecordProvider,
};
use crate::execution::{DataChunk, QueryExecutionControl, ValueVector};
use grafeo_common::memory::buffer::{
    BufferManager, BufferManagerConfig, MemoryGrant, MemoryRegion,
};
use grafeo_common::types::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// Same predeclared budget and opaque payload envelope as scalar_failure_tests.
const BUDGET: usize = 3 << 20;
const PAYLOAD_BYTES: usize = 8192;
const PROVIDER_BOUND: usize = PAYLOAD_BYTES + 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Begin,
    ReadOpen,
    ReadPayload,
    Entry,
    Finish,
    Decode,
    Consolidate,
    ReaderDrop,
    CounterScratch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Revisit,
    Finalize,
    Consolidate,
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
    creates: AtomicUsize,
    entries: AtomicUsize,
    finishes: AtomicUsize,
    pressure: std::sync::Mutex<Option<MemoryGrant>>,
}

impl State {
    fn new(
        memory: Arc<BufferManager>,
        phase: Phase,
        panic: bool,
        cleanup_fault: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            memory,
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
            creates: AtomicUsize::new(0),
            entries: AtomicUsize::new(0),
            finishes: AtomicUsize::new(0),
            pressure: std::sync::Mutex::new(None),
        })
    }
    fn fail(self: &Arc<Self>, phase: Phase) -> std::io::Result<()> {
        if !self.armed.load(Ordering::Relaxed)
            || self.phase != phase
            || self
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
        output.write_str("AggregateProviderPayload")
    }
}

impl std::fmt::Display for Payload {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.state.display_calls.fetch_add(1, Ordering::Relaxed);
        output.write_str("original aggregate provider primary")
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
        Some(PROVIDER_BOUND)
    }

    fn begin_file(&self, identity: SpillFileIdentity) -> std::io::Result<Box<dyn OpenSpillRecord>> {
        self.0.fail(Phase::Begin)?;
        Ok(Box::new(Record {
            inner: CleartextSpillRecordProvider.begin_file(identity)?,
            state: Arc::clone(&self.0),
            read_started: false,
        }))
    }
}

struct Record {
    inner: Box<dyn OpenSpillRecord>,
    state: Arc<State>,
    read_started: bool,
}

impl Drop for Record {
    fn drop(&mut self) {
        if self.read_started && self.state.phase == Phase::ReaderDrop {
            self.state
                .drop_charge
                .store(self.state.memory.allocated(), Ordering::Relaxed);
            let _ = self.state.fail(Phase::ReaderDrop);
        }
    }
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
        if meta.kind() == SpillRecordKind::PartitionStart
            && self.state.phase == Phase::CounterScratch
            && self.state.armed.load(Ordering::Relaxed)
        {
            // Competing, independently owned pressure arrives after writer
            // setup but before the protected GroupState serializer admission.
            // This observes the real boundary without reproducing its formula.
            let mut pressure = self.state.pressure.lock().unwrap();
            assert!(pressure.is_none());
            let remaining = BUDGET - self.state.memory.allocated();
            assert!(remaining > 0);
            *pressure = Some(
                self.state
                    .memory
                    .try_allocate(remaining, MemoryRegion::ExecutionBuffers)
                    .unwrap(),
            );
        }
        self.inner.seal(meta, aad, plaintext)
    }

    fn open(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        stored: &[u8],
    ) -> std::io::Result<Vec<u8>> {
        match meta.kind() {
            SpillRecordKind::PartitionEntry => self.state.fail(Phase::Entry)?,
            SpillRecordKind::FileEnd => self.state.fail(Phase::Finish)?,
            _ => {}
        }
        let bytes = self.inner.open(meta, aad, stored)?;
        self.read_started = true;
        match meta.kind() {
            SpillRecordKind::PartitionEntry => {
                self.state.entries.fetch_add(1, Ordering::Relaxed);
            }
            SpillRecordKind::FileEnd => {
                self.state.finishes.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        Ok(bytes)
    }
}

struct Io(Arc<State>);

impl SpillIo for Io {
    fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        Some(PROVIDER_BOUND)
    }

    fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
        Some(0)
    }

    fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
        if operation == SpillIoOperation::Create {
            self.0.creates.fetch_add(1, Ordering::Relaxed);
        }
        match operation {
            SpillIoOperation::ReadOpen => self.0.fail(Phase::ReadOpen)?,
            SpillIoOperation::ReadPayload => self.0.fail(Phase::ReadPayload)?,
            _ => {}
        }
        if self.0.cleanup_fault
            && self.0.hits.load(Ordering::Relaxed) == 1
            && operation == SpillIoOperation::Delete
            && self
                .0
                .cleanup_hits
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            return Err(std::io::ErrorKind::PermissionDenied.into());
        }
        Ok(())
    }
}

fn chunk(value: i64) -> DataChunk {
    DataChunk::new(vec![
        ValueVector::from_values(&[Value::Int64(7)]),
        ValueVector::from_values(&[Value::Int64(value)]),
    ])
}

fn other_key_in_same_partition(partitions: &PartitionedState<GroupState>) -> i64 {
    let target = partitions.partition_for(&[Value::Int64(7)]);
    (0..65_536)
        .find(|&key| key != 7 && partitions.partition_for(&[Value::Int64(key)]) == target)
        .expect("bounded fixture search must find a second key in the target partition")
}

fn retained_failure(
    phase: Phase,
    panic: bool,
    cleanup_fault: bool,
    diagnostic_first: bool,
    route: Route,
) {
    let mut config = BufferManagerConfig::with_budget(BUDGET);
    config.soft_limit_fraction = 1.0;
    config.evict_limit_fraction = 1.0;
    config.hard_limit_fraction = 1.0;
    let memory = BufferManager::new(config);
    let state = State::new(Arc::clone(&memory), phase, panic, cleanup_fault);
    let directory = tempfile::tempdir().unwrap();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(Provider(Arc::clone(&state))),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(Io(Arc::clone(&state))))
        .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
        .unwrap();
    let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
        vec![0],
        vec![AggregateExpr::sum(1)],
        resources.clone(),
    )
    .unwrap();
    // Keep the production codecs; the wrapper injects an opaque failure at
    // their actual invocation boundary while using the same resource root.
    // This candidate-only seam supplements the unchanged parent RED snapshot.
    if matches!(phase, Phase::Decode | Phase::Consolidate) {
        let encode_state = Arc::clone(&state);
        let decode_state = Arc::clone(&state);
        aggregate.partitioned_groups = Some(
            PartitionedState::new_accounted_admitted_with_cancellation(
                Arc::clone(&manager),
                256,
                move |value: &GroupState, writer, limits| {
                    encode_state.fail(Phase::Consolidate)?;
                    super::serialize_group_state_bounded(value, writer, limits)
                },
                move |reader, limits| {
                    decode_state.fail(Phase::Decode)?;
                    super::deserialize_group_state_bounded(reader, limits)
                },
                GroupState::retained_heap_bytes,
                resources.try_allocate(0).unwrap(),
                resources.cancellation_token().clone(),
            )
            .unwrap(),
        );
    }
    let mut sink = CountingSink::new();
    aggregate.push(chunk(2), &mut sink).unwrap();
    if route == Route::Consolidate && matches!(phase, Phase::Decode | Phase::Consolidate) {
        // Leave one durable key without an override: a completely covered
        // base legitimately retires without another codec invocation.
        let other = other_key_in_same_partition(aggregate.partitioned_groups.as_ref().unwrap());
        aggregate
            .push(
                DataChunk::new(vec![
                    ValueVector::from_values(&[Value::Int64(other)]),
                    ValueVector::from_values(&[Value::Int64(11)]),
                ]),
                &mut sink,
            )
            .unwrap();
    }
    let partitions = aggregate.partitioned_groups.as_mut().unwrap();
    let partition = partitions.partition_for(&[Value::Int64(7)]);
    assert!(partitions.spill_partition_controlled(partition).unwrap() > 0);
    assert_eq!(manager.active_file_count(), 1);
    if route == Route::Consolidate {
        // Publish a real resident override of the durable base, then force the
        // normal finalize path to reconcile that delta before emitting rows.
        aggregate.push(chunk(3), &mut sink).unwrap();
        assert_eq!(manager.active_file_count(), 1);
    }
    state.armed.store(true, Ordering::Relaxed);

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match route {
        Route::Revisit => aggregate.push(chunk(3), &mut sink),
        Route::Finalize | Route::Consolidate => aggregate.finalize(&mut sink).map(|()| true),
    }));
    assert_eq!(state.hits.load(Ordering::Relaxed), 1, "{phase:?}");
    assert_eq!(
        state.drops.load(Ordering::Relaxed),
        0,
        "original payload: route={route:?} phase={phase:?} panic={panic} cleanup={cleanup_fault} diagnostic_first={diagnostic_first}"
    );
    let Ok(Err(error)) = &outcome else {
        panic!("{phase:?} must return a retained diagnostic instead of unwinding");
    };
    assert!(matches!(
        error,
        OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::Execution,
            ..
        }
    ));
    assert_eq!(
        PartitionedState::<GroupState>::inspect_failure(
            error,
            |primary, operator, original_panic, cleanup_error, cleanup_panic| {
                assert!(operator.is_none());
                assert!(cleanup_panic.is_none());
                if panic {
                    let original = original_panic.unwrap().downcast_ref::<Payload>().unwrap();
                    assert_eq!(
                        original.bytes.as_ptr() as usize,
                        state.payload_address.load(Ordering::Relaxed)
                    );
                } else {
                    assert!(original_panic.is_none());
                    let original = match primary.unwrap() {
                        PartitionOperationError::Io(error)
                        | PartitionOperationError::IoWithCleanup { error, .. } => error,
                        other => panic!("opaque I/O primary changed type: {other:?}"),
                    };
                    assert_eq!(original.kind(), std::io::ErrorKind::BrokenPipe);
                    let payload = original
                        .get_ref()
                        .unwrap()
                        .downcast_ref::<Payload>()
                        .unwrap();
                    assert_eq!(
                        payload.bytes.as_ptr() as usize,
                        state.payload_address.load(Ordering::Relaxed)
                    );
                }
                if let Some(cleanup) = cleanup_error {
                    assert!(cleanup_fault);
                    assert_eq!(cleanup.kind(), std::io::ErrorKind::PermissionDenied);
                }
            },
        ),
        Some(())
    );
    let _ = error.to_string();
    assert_eq!(state.display_calls.load(Ordering::Relaxed), 0);
    // The consuming cursor checks FileEnd on the call after its final entry.
    // Direct finalize may therefore have transferred this one valid owned row
    // before the footer fails. Consolidation checks the footer before output.
    // The eager engine still returns the error before publishing QueryResult.
    let prior_rows = usize::from(route == Route::Finalize && phase == Phase::Finish);
    assert_eq!(
        sink.count(),
        prior_rows,
        "prior rows: route={route:?} phase={phase:?} panic={panic} cleanup={cleanup_fault} diagnostic_first={diagnostic_first}"
    );
    assert!(memory.allocated() >= PROVIDER_BOUND);

    if diagnostic_first {
        drop(outcome);
        assert_eq!(
            state.drops.load(Ordering::Relaxed),
            1,
            "diagnostic-first: route={route:?} phase={phase:?} panic={panic} cleanup={cleanup_fault}"
        );
        drop(aggregate);
    } else {
        drop(aggregate);
        assert_eq!(
            state.drops.load(Ordering::Relaxed),
            0,
            "operator-first: route={route:?} phase={phase:?} panic={panic} cleanup={cleanup_fault}"
        );
        assert!(memory.allocated() >= PROVIDER_BOUND);
        drop(outcome);
    }
    drop(resources);
    manager.cleanup().unwrap();
    assert_eq!(state.drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        state.dropped_address.load(Ordering::Relaxed),
        state.payload_address.load(Ordering::Relaxed)
    );
    assert!(state.drop_charge.load(Ordering::Relaxed) >= PROVIDER_BOUND);
    assert_eq!(
        state.cleanup_hits.load(Ordering::Relaxed),
        usize::from(cleanup_fault)
    );
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.spilled_bytes(), 0);
    assert_eq!(memory.allocated(), 0);
}

#[test]
fn resource_revisit_retains_original_provider_failure_after_operator_drop() {
    for phase in [
        Phase::Begin,
        Phase::ReadOpen,
        Phase::ReadPayload,
        Phase::Entry,
        Phase::Finish,
    ] {
        for panic in [false, true] {
            for cleanup in [false, true] {
                retained_failure(phase, panic, cleanup, false, Route::Revisit);
            }
        }
    }
}

#[test]
fn resource_revisit_retains_original_provider_failure_with_diagnostic_dropped_first() {
    for phase in [Phase::Begin, Phase::Entry, Phase::Finish] {
        for panic in [false, true] {
            for cleanup in [false, true] {
                retained_failure(phase, panic, cleanup, true, Route::Revisit);
            }
        }
    }
}

#[test]
fn resource_finalize_retains_original_failure_in_both_drop_orders() {
    for route in [Route::Finalize, Route::Consolidate] {
        for phase in [
            Phase::Begin,
            Phase::ReadOpen,
            Phase::ReadPayload,
            Phase::Entry,
            Phase::Finish,
            Phase::Decode,
        ] {
            for panic in [false, true] {
                for cleanup in [false, true] {
                    for diagnostic_first in [false, true] {
                        retained_failure(phase, panic, cleanup, diagnostic_first, route);
                    }
                }
            }
        }
    }
}

#[test]
fn resource_consolidation_retains_original_codec_failure_in_both_drop_orders() {
    for panic in [false, true] {
        for cleanup in [false, true] {
            for diagnostic_first in [false, true] {
                retained_failure(
                    Phase::Consolidate,
                    panic,
                    cleanup,
                    diagnostic_first,
                    Route::Consolidate,
                );
            }
        }
    }
}

#[test]
fn resource_revisit_retains_original_decoder_failure_in_both_drop_orders() {
    for panic in [false, true] {
        for cleanup in [false, true] {
            for diagnostic_first in [false, true] {
                retained_failure(
                    Phase::Decode,
                    panic,
                    cleanup,
                    diagnostic_first,
                    Route::Revisit,
                );
            }
        }
    }
}

#[test]
fn resource_successful_reader_drop_panic_quarantines_authority() {
    let memory = BufferManager::with_budget(BUDGET);
    let state = State::new(Arc::clone(&memory), Phase::ReaderDrop, true, false);
    let directory = tempfile::tempdir().unwrap();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(Provider(Arc::clone(&state))),
            SpillFrameLimits::format_max(),
        )
        .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
        .unwrap();
    let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
        vec![0],
        vec![AggregateExpr::sum(1)],
        resources.clone(),
    )
    .unwrap();
    let mut sink = CountingSink::new();
    aggregate.push(chunk(2), &mut sink).unwrap();
    let partitions = aggregate.partitioned_groups.as_mut().unwrap();
    let index = partitions.partition_for(&[Value::Int64(7)]);
    partitions.spill_partition_controlled(index).unwrap();
    state.armed.store(true, Ordering::Relaxed);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        aggregate.finalize(&mut sink)
    }));
    let error = outcome
        .expect("reader destructor panic must not escape")
        .unwrap_err();
    assert!(matches!(
        &error,
        OperatorError::ClassifiedAccountedFailure {
            classification: AccountedFailureClassification::Execution,
            ..
        }
    ));
    assert!(state.entries.load(Ordering::Relaxed) > 0);
    assert_eq!(state.finishes.load(Ordering::Relaxed), 1);
    assert_eq!(state.hits.load(Ordering::Relaxed), 1);
    assert_eq!(state.drops.load(Ordering::Relaxed), 0);
    assert!(state.drop_charge.load(Ordering::Relaxed) >= PROVIDER_BOUND);
    drop(aggregate);
    drop(resources);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(error))).is_ok());
    assert_eq!(
        state.drops.load(Ordering::Relaxed),
        0,
        "opaque destructor panic is quarantined"
    );
    manager.cleanup().unwrap();
    assert_eq!(manager.active_file_count(), 0);
    assert!(
        memory.allocated() >= PROVIDER_BOUND,
        "quarantine retains authority"
    );
}

#[derive(Default)]
struct RetainedSink(Vec<AccountedDataChunk>);

impl Sink for RetainedSink {
    fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
        panic!("resource aggregate must use sealed accounted transport");
    }
    fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
        Some(AccountedSinkPermit::new(self))
    }
    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }
    fn name(&self) -> &'static str {
        "RetainedAggregateControl"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}
impl QualifiedSink for RetainedSink {
    fn consume_accounted_qualified(
        &mut self,
        chunk: AccountedDataChunk,
    ) -> Result<bool, OperatorError> {
        assert!(chunk.granted_bytes() > 0);
        self.0.push(chunk);
        Ok(true)
    }
}

#[test]
fn resource_grouped_output_retains_authority_and_refuses_plain_sink_before_consumption() {
    let memory = BufferManager::with_budget(BUDGET);
    let directory = tempfile::tempdir().unwrap();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
        .unwrap();
    let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
        vec![0],
        vec![AggregateExpr::sum(1)],
        resources.clone(),
    )
    .unwrap();
    let mut plain = CollectorSink::new();
    aggregate.push(chunk(2), &mut plain).unwrap();
    let partitions = aggregate.partitioned_groups.as_mut().unwrap();
    let index = partitions.partition_for(&[Value::Int64(7)]);
    partitions.spill_partition_controlled(index).unwrap();
    aggregate.push(chunk(3), &mut plain).unwrap();
    let before_size = aggregate.partitioned_groups.as_ref().unwrap().total_size();
    let before_files = manager.active_file_count();
    let before_grants = resources.query_stats().allocated_bytes;
    aggregate
        .finalize(&mut plain)
        .expect_err("plain output cannot detach resource authority");
    assert!(plain.chunks().is_empty());
    assert_eq!(
        aggregate.partitioned_groups.as_ref().unwrap().total_size(),
        before_size
    );
    assert_eq!(manager.active_file_count(), before_files);
    assert_eq!(resources.query_stats().allocated_bytes, before_grants);
    let mut retained = RetainedSink::default();
    aggregate.finalize(&mut retained).unwrap();
    assert_eq!(
        retained
            .0
            .iter()
            .map(|chunk| chunk.chunk().len())
            .sum::<usize>(),
        1
    );
    let output = retained.0[0].chunk();
    assert_eq!(
        output.column(0).unwrap().get_value(0),
        Some(Value::Int64(7))
    );
    assert_eq!(
        output.column(1).unwrap().get_value(0),
        Some(Value::Int64(5))
    );
    drop(aggregate);
    assert!(resources.query_stats().allocated_bytes > 0);
    assert!(memory.allocated() > 0);
    drop(retained);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
    assert_eq!(memory.allocated(), 0);
    assert_eq!(manager.active_file_count(), 0);
}

#[test]
fn resource_grouped_finalizer_panic_keeps_original_string_and_authority() {
    let memory = BufferManager::with_budget(BUDGET);
    let directory = tempfile::tempdir().unwrap();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
        .unwrap();
    let mut percentile = AggregateExpr::percentile_disc(1, 0.5);
    percentile.percentile = Some(2.0);
    let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
        vec![0],
        vec![percentile],
        resources.clone(),
    )
    .unwrap();
    let mut sink = CountingSink::new();
    aggregate.push(chunk(2), &mut sink).unwrap();
    aggregate.push(chunk(3), &mut sink).unwrap();
    let partitions = aggregate.partitioned_groups.as_mut().unwrap();
    let index = partitions.partition_for(&[Value::Int64(7)]);
    partitions.spill_partition_controlled(index).unwrap();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        aggregate.finalize(&mut sink)
    }));
    let error = outcome
        .expect("actual finalizer panic must be owned")
        .unwrap_err();
    assert_eq!(
        PartitionedState::<GroupState>::inspect_failure(&error, |_, _, panic, _, _| {
            let text = panic
                .unwrap()
                .downcast_ref::<String>()
                .expect("original indexing panic String");
            assert!(text.contains("index out of bounds"), "{text}");
        }),
        Some(())
    );
    assert_eq!(sink.count(), 0);
    drop(aggregate);
    assert!(resources.query_stats().allocated_bytes > 0);
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
    manager.cleanup().unwrap();
    assert_eq!(manager.active_file_count(), 0);
}

struct CheckingSink {
    seen: Vec<bool>,
    rows: usize,
}

impl Sink for CheckingSink {
    fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
        panic!("grouped resource output must be accounted");
    }
    fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
        Some(AccountedSinkPermit::new(self))
    }
    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }
    fn name(&self) -> &'static str {
        "CheckingAggregateControl"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

fn fixed_payload(group: usize, variant: usize) -> Value {
    let mut text = format!("group-{group:05}-kind-{variant}-");
    text.push_str(&"x".repeat(512 - text.len()));
    assert_eq!(text.len(), 512);
    Value::from(text)
}

impl QualifiedSink for CheckingSink {
    fn consume_accounted_qualified(
        &mut self,
        owner: AccountedDataChunk,
    ) -> Result<bool, OperatorError> {
        assert!(owner.granted_bytes() > 0);
        let chunk = owner.chunk();
        for row in 0..chunk.len() {
            let Value::Int64(group) = chunk.column(0).unwrap().get_value(row).unwrap() else {
                panic!("group key changed");
            };
            let group = usize::try_from(group).unwrap();
            assert!(group < self.seen.len() && !self.seen[group]);
            self.seen[group] = true;
            assert_eq!(
                chunk.column(1).unwrap().get_value(row),
                Some(Value::Int64(4))
            );
            let Value::List(collected) = chunk.column(2).unwrap().get_value(row).unwrap() else {
                panic!("COLLECT lost its list");
            };
            let Value::List(unique) = chunk.column(3).unwrap().get_value(row).unwrap() else {
                panic!("COLLECT DISTINCT lost its list");
            };
            assert_eq!(collected.len(), 4);
            assert_eq!(unique.len(), 2);
            for variant in 0..2 {
                let expected = fixed_payload(group, variant);
                assert_eq!(
                    collected.iter().filter(|value| **value == expected).count(),
                    2
                );
                assert_eq!(unique.iter().filter(|value| **value == expected).count(), 1);
            }
            self.rows += 1;
        }
        Ok(true)
    }
}

#[test]
fn resource_grouped_heap_n_2n_4n_spills_revisits_consolidates_with_fixed_budget() {
    // Matches the public fixture's balanced input exactly, but drains each
    // accounted chunk before the next. Eager output size therefore cannot turn
    // this required completing ownership witness into an all-denial gate.
    for rows in [4096, 8192, 16384] {
        let mut config = BufferManagerConfig::with_budget(BUDGET);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        let memory = BufferManager::new(config);
        let state = State::new(Arc::clone(&memory), Phase::Begin, false, false);
        let directory = tempfile::tempdir().unwrap();
        let (resources, manager) = BorrowedSpillFixture::new(directory.path())
            .io(Arc::new(Io(Arc::clone(&state))))
            .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
            .unwrap();
        let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
            vec![0],
            vec![
                AggregateExpr::count_star(),
                AggregateExpr::collect(1),
                AggregateExpr::collect(1).with_distinct(),
            ],
            resources.clone(),
        )
        .unwrap();
        let groups = rows / 4;
        let mut sink = CheckingSink {
            seen: vec![false; groups],
            rows: 0,
        };
        let mut observed_recovery = false;
        let mut forced_spills = 0;
        for pass in 0..4 {
            for group in 0..groups {
                let creates_before = state.creates.load(Ordering::Relaxed);
                aggregate
                    .push(
                        DataChunk::new(vec![
                            ValueVector::from_values(&[Value::Int64(
                                i64::try_from(group).unwrap(),
                            )]),
                            ValueVector::from_values(&[fixed_payload(group, pass % 2)]),
                        ]),
                        &mut sink,
                    )
                    .unwrap_or_else(|error| {
                        panic!("aggregate push rows={rows} pass={pass} group={group}: {error:?}");
                    });
                let creates = state.creates.load(Ordering::Relaxed) - creates_before;
                assert!(
                    creates <= 1,
                    "one row update may not perform a second recovery spill: {creates}"
                );
                observed_recovery |= creates == 1;
                assert!(resources.query_stats().allocated_bytes <= BUDGET);
            }
            if pass < 3 {
                let partitions = aggregate.partitioned_groups.as_mut().unwrap();
                // Proactive recovery may already have emptied any fixed
                // partition's resident delta. The public size includes its
                // durable base, so probe real group partitions until one
                // controlled spill actually writes a nonempty delta.
                let mut forced = None;
                for group in 0..groups {
                    let index =
                        partitions.partition_for(&[Value::Int64(i64::try_from(group).unwrap())]);
                    let creates_before = state.creates.load(Ordering::Relaxed);
                    let bytes = partitions
                        .spill_partition_controlled(index)
                        .unwrap_or_else(|error| {
                            panic!(
                                "forced spill rows={rows} pass={pass} group={group} partition={index}: {error:?}"
                            );
                        });
                    let creates = state.creates.load(Ordering::Relaxed) - creates_before;
                    assert_eq!(
                        creates,
                        usize::from(bytes > 0),
                        "forced spill rows={rows} pass={pass} group={group} partition={index} bytes={bytes}"
                    );
                    if bytes > 0 {
                        forced = Some(index);
                        break;
                    }
                }
                assert!(
                    forced.is_some(),
                    "no live delta for forced spill rows={rows} pass={pass} groups={groups}"
                );
                forced_spills += 1;
            }
        }
        assert_eq!(sink.rows, 0);
        assert_eq!(forced_spills, 3);
        assert!(
            observed_recovery,
            "size={rows} must exercise admission-triggered spill as well as forced spill"
        );
        let before_finalize = manager.profile_totals();
        aggregate.finalize(&mut sink).unwrap_or_else(|error| {
            panic!("aggregate finalize rows={rows} groups={groups}: {error:?}");
        });
        assert_eq!(sink.rows, groups);
        assert!(sink.seen.iter().all(|seen| *seen));
        let stats = resources.profile_stats();
        assert!((1..=BUDGET).contains(&stats.resident_peak_bytes));
        assert!(stats.spilled_bytes > 0 && stats.spill_partitions >= 3);
        assert!(
            manager.profile_totals().0 > before_finalize.0,
            "resident deltas must traverse consolidation before output"
        );
        drop(aggregate);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        assert_eq!(memory.allocated(), 0);
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        println!(
            "aggregate owned memory: rows={rows} budget={BUDGET} resident_peak={} spill_partitions={} one_update_max_create=1",
            stats.resident_peak_bytes, stats.spill_partitions
        );
    }
}

#[test]
fn resource_revisit_cancellation_precedes_armed_provider_failure() {
    let memory = BufferManager::with_budget(BUDGET);
    let state = State::new(Arc::clone(&memory), Phase::Begin, true, false);
    let directory = tempfile::tempdir().unwrap();
    let control = QueryExecutionControl::new();
    let (resources, manager) = BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(Provider(Arc::clone(&state))),
            SpillFrameLimits::format_max(),
        )
        .build_operator_resources(Arc::clone(&memory), control.token())
        .unwrap();
    let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
        vec![0],
        vec![AggregateExpr::sum(1)],
        resources.clone(),
    )
    .unwrap();
    let mut sink = CountingSink::new();
    aggregate.push(chunk(2), &mut sink).unwrap();
    let partitions = aggregate.partitioned_groups.as_mut().unwrap();
    let index = partitions.partition_for(&[Value::Int64(7)]);
    partitions.spill_partition_controlled(index).unwrap();
    state.armed.store(true, Ordering::Relaxed);
    control.cancellation_handle().cancel();
    let error = aggregate.push(chunk(3), &mut sink).unwrap_err();
    assert!(matches!(error, OperatorError::QueryCancelled(_)));
    assert_eq!(state.hits.load(Ordering::Relaxed), 0);
    assert_eq!(state.drops.load(Ordering::Relaxed), 0);
    assert_eq!(sink.count(), 0);
    drop(aggregate);
    drop(error);
    drop(resources);
    manager.cleanup().unwrap();
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(memory.allocated(), 0);
}

#[test]
fn resource_nested_counter_spill_revisit_and_scratch_denial_preserve_ownership() {
    use grafeo_common::types::PropertyKey;
    use std::collections::{BTreeMap, HashMap};

    // Nontrivial reverse-insertion maps force the actual deterministic counter
    // sort scratch inside both nested COLLECT and DISTINCT aggregate state.
    let replicas = |offset: u64| -> HashMap<String, u64> {
        (0..256)
            .rev()
            .map(|index| {
                (
                    format!("replica-{index:04}-{}", "x".repeat(48)),
                    offset + index,
                )
            })
            .collect()
    };
    let nested = Value::List(Arc::from(vec![
        Value::GCounter(Arc::new(replicas(1))),
        Value::Map(Arc::new(BTreeMap::from([(
            PropertyKey::new("signed"),
            Value::OnCounter {
                pos: Arc::new(replicas(2)),
                neg: Arc::new(replicas(3)),
            },
        )]))),
    ]));
    for denied in [false, true] {
        let mut config = BufferManagerConfig::with_budget(BUDGET);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        let memory = BufferManager::new(config);
        let state = State::new(Arc::clone(&memory), Phase::CounterScratch, false, false);
        let directory = tempfile::tempdir().unwrap();
        let (resources, manager) = BorrowedSpillFixture::new(directory.path())
            .provider(
                Arc::new(Provider(Arc::clone(&state))),
                SpillFrameLimits::format_max(),
            )
            .io(Arc::new(Io(Arc::clone(&state))))
            .build_operator_resources(Arc::clone(&memory), QueryExecutionControl::new().token())
            .unwrap();
        let mut aggregate = SpillableAggregatePushOperator::with_qualified_resource_context(
            vec![0],
            vec![
                AggregateExpr::count_star(),
                AggregateExpr::collect(1),
                AggregateExpr::collect(1).with_distinct(),
            ],
            resources.clone(),
        )
        .unwrap();
        let encoded = Arc::new(AtomicUsize::new(0));
        let decoded = Arc::new(AtomicUsize::new(0));
        let encode_calls = Arc::clone(&encoded);
        let decode_calls = Arc::clone(&decoded);
        aggregate.partitioned_groups = Some(
            PartitionedState::new_accounted_admitted_with_cancellation(
                Arc::clone(&manager),
                256,
                move |value: &GroupState, writer, limits| {
                    encode_calls.fetch_add(1, Ordering::Relaxed);
                    super::serialize_group_state_bounded(value, writer, limits)
                },
                move |reader, limits| {
                    decode_calls.fetch_add(1, Ordering::Relaxed);
                    super::deserialize_group_state_bounded(reader, limits)
                },
                GroupState::retained_heap_bytes,
                resources.try_allocate(0).unwrap(),
                resources.cancellation_token().clone(),
            )
            .unwrap(),
        );
        let input = || {
            DataChunk::new(vec![
                ValueVector::from_values(&[Value::Int64(7)]),
                ValueVector::from_values(std::slice::from_ref(&nested)),
            ])
        };
        let mut sink = RetainedSink::default();
        aggregate.push(input(), &mut sink).unwrap();
        let unmodified_key = if denied {
            None
        } else {
            // Force actual partial-coverage consolidation while keeping the
            // second group small and the fixed 3 MiB envelope unchanged.
            let key = other_key_in_same_partition(aggregate.partitioned_groups.as_ref().unwrap());
            aggregate
                .push(
                    DataChunk::new(vec![
                        ValueVector::from_values(&[Value::Int64(key)]),
                        ValueVector::from_values(&[Value::Int64(11)]),
                    ]),
                    &mut sink,
                )
                .unwrap();
            Some(key)
        };
        state.armed.store(denied, Ordering::Relaxed);
        let partitions = aggregate.partitioned_groups.as_mut().unwrap();
        let index = partitions.partition_for(&[Value::Int64(7)]);
        let spilled = partitions.spill_partition_controlled(index);
        if denied {
            let error = SpillableAggregatePushOperator::map_partition_error(spilled.unwrap_err());
            assert!(
                matches!(
                    &error,
                    OperatorError::ClassifiedAccountedFailure {
                        classification: AccountedFailureClassification::ResidentMemory(_),
                        ..
                    }
                ),
                "{error:?}"
            );
            assert_eq!(
                encoded.load(Ordering::Relaxed),
                0,
                "admission must precede the opaque GroupState serializer"
            );
            assert_eq!(decoded.load(Ordering::Relaxed), 0);
            assert!(
                state.pressure.lock().unwrap().is_some(),
                "denial reached the actual PartitionStart boundary"
            );
            assert_eq!(
                state.creates.load(Ordering::Relaxed),
                1,
                "one failed staging attempt"
            );
            assert!(sink.0.is_empty());
            drop(aggregate);
            drop(error);
        } else {
            assert!(spilled.unwrap() > 0);
            assert!(encoded.load(Ordering::Relaxed) > 0);
            assert_eq!(manager.active_file_count(), 1);
            aggregate.push(input(), &mut sink).unwrap();
            assert!(
                decoded.load(Ordering::Relaxed) > 0,
                "revisit must decode the durable nested counters"
            );
            let encoded_before_finalize = encoded.load(Ordering::Relaxed);
            aggregate.finalize(&mut sink).unwrap();
            assert_eq!(
                encoded.load(Ordering::Relaxed),
                encoded_before_finalize + 1,
                "finalization must serialize the one revisited delta and copy its untouched peer"
            );
            assert_eq!(sink.0.len(), 2);
            let mut seen_nested = false;
            let mut seen_unmodified = false;
            for chunk in &sink.0 {
                let output = chunk.chunk();
                assert_eq!(output.len(), 1);
                let key = output.column(0).unwrap().get_value(0).unwrap();
                let (count, collected, distinct) = if key == Value::Int64(7) {
                    assert!(!seen_nested);
                    seen_nested = true;
                    (
                        2,
                        vec![nested.clone(), nested.clone()],
                        vec![nested.clone()],
                    )
                } else {
                    assert_eq!(key, Value::Int64(unmodified_key.unwrap()));
                    assert!(!seen_unmodified);
                    seen_unmodified = true;
                    (1, vec![Value::Int64(11)], vec![Value::Int64(11)])
                };
                assert_eq!(
                    output.column(1).unwrap().get_value(0),
                    Some(Value::Int64(count))
                );
                assert_eq!(
                    output.column(2).unwrap().get_value(0),
                    Some(Value::List(Arc::from(collected)))
                );
                assert_eq!(
                    output.column(3).unwrap().get_value(0),
                    Some(Value::List(Arc::from(distinct)))
                );
            }
            assert!(seen_nested && seen_unmodified);
            drop(aggregate);
            assert!(
                resources.query_stats().allocated_bytes > 0,
                "retained nested output owns its authority"
            );
        }
        assert!(resources.profile_stats().resident_peak_bytes <= BUDGET);
        drop(sink);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
        drop(resources);
        drop(state.pressure.lock().unwrap().take());
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(memory.allocated(), 0);
    }
}
