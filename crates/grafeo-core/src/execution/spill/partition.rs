//! Hash partitioning for spillable aggregation.
//!
//! This module implements hash partitioning that allows aggregate state
//! to be partitioned and spilled to disk when memory pressure is high.
//!
//! # Design
//!
//! - Groups are assigned to partitions based on their key's hash
//! - In-memory partitions can be spilled to disk under memory pressure
//! - Cold (least recently accessed) partitions are spilled first
//! - When iterating results, spilled partitions are reloaded

use super::file::{
    MAX_FIXED_CONTROL_PAYLOAD_BYTES, SpillFile, SpillFileReader, SpillFileRole, SpillFrameLimits,
    SpillRecordBuffer, SpillWriterBuffer, qualified_partition_reader_buffer_requested_bytes,
    qualified_writer_buffer_requested_bytes,
};
use super::manager::{SpillManager, SpillQuotaExceeded};
use crate::execution::operators::{AccountedFailureClassification, OperatorError};
use crate::execution::value_codec::{
    CodecLimits, CounterSortScratch, deserialize_framed_row_exact,
    measure_serialized_row_with_limits, serialize_row_with_limits,
    serialize_row_with_prepared_scratch,
};
use crate::execution::{NativeMapAllocationError, QueryCancellationError, QueryCancellationToken};
use grafeo_common::memory::buffer::{
    AccountedError, AccountedErrorPublisher, AccountedErrorPublisherBuildError,
    AccountedErrorPublisherBuildFailure, MemoryGrant, MemoryGrantError,
};
use grafeo_common::types::Value;
#[cfg(test)]
use std::collections::HashMap;
use std::collections::TryReserveError;
use std::collections::hash_map::RandomState;
use std::io::{Read, Write};
use std::sync::Arc;
use thiserror::Error;

mod resident_cursor;

/// Default number of partitions for hash partitioning.
pub const DEFAULT_NUM_PARTITIONS: usize = 256;

/// A serialized key for use as a HashMap key.
/// We serialize Value vectors to bytes since Value doesn't implement Hash/Eq.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SerializedKey(Vec<u8>);

impl SerializedKey {
    fn from_values(values: &[Value], limits: SpillFrameLimits) -> std::io::Result<Self> {
        let maximum = usize::try_from(limits.max_plaintext_bytes()).unwrap_or(usize::MAX);
        let mut buf = SpillRecordBuffer::new(maximum);
        serialize_row_with_limits(values, &mut buf, limits.codec_limits())?;
        Ok(Self(buf.into_inner()))
    }

    fn to_values(
        &self,
        num_columns: usize,
        limits: SpillFrameLimits,
    ) -> std::io::Result<Vec<Value>> {
        deserialize_framed_row_exact(&self.0, num_columns, limits.codec_limits())
    }
}

/// A serialized key whose retained buffer capacity remains covered by its
/// own child grant until publication transfers that authority to the state.
/// Physical bytes are deliberately dropped before their accounting token.
struct AccountedSerializedKey {
    key: Option<SerializedKey>,
    grant: Option<MemoryGrant>,
}

impl AccountedSerializedKey {
    fn serialized(&self) -> &SerializedKey {
        self.key
            .as_ref()
            .expect("unpublished accounted key retains its bytes")
    }

    #[cfg(test)]
    fn granted_bytes(&self) -> usize {
        self.grant
            .as_ref()
            .expect("unpublished accounted key retains its grant")
            .size()
    }
}

impl Drop for AccountedSerializedKey {
    fn drop(&mut self) {
        // Keep this explicit: future field changes must not release authority
        // before the physical key allocation is gone.
        drop(self.key.take());
        drop(self.grant.take());
    }
}

/// Deterministic counter-ordering scratch paired with its temporary grant.
/// Field order preserves physical-before-authority destruction on every exit.
struct AccountedCounterScratch {
    scratch: CounterSortScratch,
    _grant: MemoryGrant,
}

/// Operation-scoped budget permitting at most one cold-partition reclamation
/// across every transient admission phase.
#[derive(Debug, Default)]
struct OneSpillRetryBudget {
    spent: bool,
    // Idle scheduling authority; it covers no physical allocation. Only the
    // protected aggregate caller supplies it. It is idle between rows only.
    recovery: Option<MemoryGrant>,
    recovery_required: usize,
}

/// Immutable proof carried from root-grant admission through native-map
/// allocation. The replacement ceiling is the pinned table's complete
/// allocation request (entry buckets, padding, and control bytes); the root
/// grant covers both old and replacement maps until the old allocation is
/// physically destroyed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingEntryCapacityPlan {
    observed_bytes: usize,
    pending_bytes: usize,
    old_map_bytes: usize,
    replacement_required_entries: Option<usize>,
    replacement_allocation_ceiling: Option<usize>,
    admitted_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReservedEntryCapacities {
    peak_bytes: usize,
    final_bytes: usize,
}

#[derive(Clone, Copy)]
enum MapAdmissionCapacity {
    InspectResident { pending_bytes: usize },
    AggregateRoot,
}

/// Rollback boundary for the only interval where key authority has moved into
/// the root grant but the physical key is not yet resident in the map.
struct AccountedKeyPublication<'a> {
    staged: Option<AccountedSerializedKey>,
    root_grant: &'a mut MemoryGrant,
    rollback_size: usize,
    merged: bool,
    committed: bool,
}

/// Caller-declared memory contract for one complete aggregate replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PartitionUpdateAdmission {
    /// Maximum heap capacity retained below the published replacement value.
    pub(crate) retained_upper_bound: usize,
    /// Total candidate-construction authority held before the builder runs.
    pub(crate) construction_peak: usize,
}

enum PartitionValueConstructionError<E> {
    Builder(E),
    Capacity(MemoryGrantError),
}

mod shared_immutable_partition_value {
    pub trait Sealed {}

    impl Sealed for i64 {}
    impl Sealed for Vec<u8> {}
    impl Sealed for crate::execution::operators::push::GroupState {}
}

/// Audited promise that sharing `&Self` with a replacement callback cannot
/// mutate the authoritative value or any allocation retained below it.
///
/// The replacement protocol deliberately lends the live value instead of
/// cloning or serializing it. Implementations therefore stay sealed and are
/// admitted individually only after reviewing every safe aliasing path.
pub(crate) trait SharedImmutablePartitionValue:
    shared_immutable_partition_value::Sealed
{
}

impl SharedImmutablePartitionValue for i64 {}
impl SharedImmutablePartitionValue for Vec<u8> {}
// GroupState owns its mutable vectors/maps. Value's shared descendants are
// immutable; their mutation APIs require exclusive access and detach aliases.
impl SharedImmutablePartitionValue for crate::execution::operators::push::GroupState {}

/// Shared lifetime witness for one accounted partition operation. Readers and
/// escaped diagnostics retain the same witness, including after state drop.
pub(super) struct PartitionFailureCleanup {
    retained: std::cell::RefCell<Option<MemoryGrant>>,
    secondary_error: std::cell::RefCell<Option<std::io::Error>>,
    secondary_panic: std::cell::RefCell<Option<Box<dyn std::any::Any + Send>>>,
    retaining: std::sync::atomic::AtomicBool,
    failed: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for PartitionFailureCleanup {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("PartitionFailureCleanup")
            .finish_non_exhaustive()
    }
}

impl PartitionFailureCleanup {
    fn new() -> Self {
        Self {
            retained: std::cell::RefCell::new(None),
            secondary_error: std::cell::RefCell::new(None),
            secondary_panic: std::cell::RefCell::new(None),
            retaining: std::sync::atomic::AtomicBool::new(false),
            failed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub(super) fn mark_failed(&self) {
        self.mark_operation_failed();
        self.failed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(super) fn mark_operation_failed(&self) {
        self.retaining
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn retain(&self, grant: MemoryGrant) {
        let Ok(mut retained) = self.retained.try_borrow_mut() else {
            self.mark_failed();
            std::mem::forget(grant);
            return;
        };
        if let Some(root) = retained.as_mut() {
            if let Err(grant) = root.try_merge(grant) {
                self.mark_failed();
                std::mem::forget(grant);
            }
        } else {
            *retained = Some(grant);
        }
    }

    fn retain_cleanup_error(&self, error: std::io::Error) {
        self.mark_operation_failed();
        let mut slot = self.secondary_error.borrow_mut();
        if slot.is_none() {
            *slot = Some(error);
        } else if !super::run_cleanup_backstop(|| {
            drop(error);
            Ok::<_, ()>(())
        }) {
            self.mark_failed();
        }
    }

    fn retain_cleanup_panic(&self, panic: Box<dyn std::any::Any + Send>) {
        self.mark_operation_failed();
        let mut slot = self.secondary_panic.borrow_mut();
        if slot.is_none() {
            *slot = Some(panic);
        } else if !super::run_cleanup_backstop(|| {
            drop(panic);
            Ok::<_, ()>(())
        }) {
            self.mark_failed();
        }
    }
}

impl std::fmt::Display for PartitionFailureCleanup {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str("accounted partition workspace lifetime")
    }
}
impl std::error::Error for PartitionFailureCleanup {}
impl Drop for PartitionFailureCleanup {
    fn drop(&mut self) {
        let error = self.secondary_error.get_mut().take();
        let panic = self.secondary_panic.get_mut().take();
        if !super::run_cleanup_backstop(|| {
            drop(error);
            Ok::<_, ()>(())
        }) {
            self.mark_failed();
        }
        if !super::run_cleanup_backstop(|| {
            drop(panic);
            Ok::<_, ()>(())
        }) {
            self.mark_failed();
        }
        if self.failed.load(std::sync::atomic::Ordering::Acquire)
            && let Some(grant) = self.retained.get_mut().take()
        {
            std::mem::forget(grant);
        }
    }
}

/// A child remains charged when a protected operation exits unsuccessfully.
/// Declare this before its physical allocation; release it explicitly only
/// after successful physical retirement or transfer it to the next owner.
struct PartitionWorkspace {
    grant: Option<MemoryGrant>,
    cleanup: Option<AccountedError>,
    release: bool,
}
impl PartitionWorkspace {
    fn new(grant: MemoryGrant, cleanup: Option<&AccountedError>) -> Self {
        Self {
            grant: Some(grant),
            cleanup: cleanup.cloned(),
            release: false,
        }
    }
    fn into_grant(mut self) -> Result<MemoryGrant, PartitionOperationError> {
        self.grant
            .take()
            .ok_or_else(|| native_map_invariant("partition workspace has no authority to transfer"))
    }
    fn grant_mut(&mut self) -> Result<&mut MemoryGrant, PartitionOperationError> {
        self.grant
            .as_mut()
            .ok_or_else(|| native_map_invariant("partition workspace has no live authority"))
    }
    fn release(mut self) {
        self.release = true;
    }
}
impl Drop for PartitionWorkspace {
    fn drop(&mut self) {
        let Some(grant) = self.grant.take() else {
            return;
        };
        let mut grant = Some(grant);
        if let Some(cleanup) = &self.cleanup {
            cleanup.inspect::<PartitionFailureCleanup, _>(|witness| {
                if (!self.release
                    || witness.retaining.load(std::sync::atomic::Ordering::Acquire)
                    || std::thread::panicking())
                    && let Some(grant) = grant.take()
                {
                    witness.retain(grant);
                }
            });
        }
        drop(grant);
    }
}

struct PartitionFailure {
    primary: Option<PartitionOperationError>,
    operator: Option<OperatorError>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    cleanup_error: Option<std::io::Error>,
    cleanup_panic: Option<Box<dyn std::any::Any + Send>>,
    cleanup: AccountedError,
}

enum PartitionMutableFailure {
    Operation(PartitionOperationError),
    Operator(OperatorError),
}
impl From<PartitionOperationError> for PartitionMutableFailure {
    fn from(error: PartitionOperationError) -> Self {
        Self::Operation(error)
    }
}
impl From<OperatorError> for PartitionMutableFailure {
    fn from(error: OperatorError) -> Self {
        Self::Operator(error)
    }
}
impl std::fmt::Debug for PartitionFailure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("PartitionFailure").finish_non_exhaustive()
    }
}
impl std::fmt::Display for PartitionFailure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // This core-owned diagnostic contains only three u64 fields. Inspect
        // the direct primary without formatting or traversing opaque sources;
        // secondary failures never replace the primary diagnostic.
        if let Some(
            PartitionOperationError::Io(error)
            | PartitionOperationError::IoWithCleanup { error, .. },
        ) = self.primary.as_ref()
            && let Some(quota) = error
                .get_ref()
                .and_then(|source| source.downcast_ref::<SpillQuotaExceeded>())
        {
            return std::fmt::Display::fmt(quota, out);
        }
        out.write_str("accounted partition operation failed")
    }
}
impl std::error::Error for PartitionFailure {}
impl Drop for PartitionFailure {
    fn drop(&mut self) {
        let mut complete = true;
        if let Some(error) = self.primary.take() {
            complete &= super::run_cleanup_backstop(|| {
                drop(error);
                Ok::<_, ()>(())
            });
        }
        if let Some(error) = self.operator.take() {
            complete &= super::run_cleanup_backstop(|| {
                drop(error);
                Ok::<_, ()>(())
            });
        }
        if let Some(error) = self.cleanup_error.take() {
            complete &= super::run_cleanup_backstop(|| {
                drop(error);
                Ok::<_, ()>(())
            });
        }
        for payload in [self.panic.take(), self.cleanup_panic.take()]
            .into_iter()
            .flatten()
        {
            complete &= super::run_cleanup_backstop(|| {
                drop(payload);
                Ok::<_, ()>(())
            });
        }
        if !complete {
            self.cleanup
                .inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_failed);
        }
    }
}

/// A callback failure paired with any authority that covered its construction.
///
/// Declaration failures occur before a construction grant exists. Builder
/// failures retain the complete declared construction grant until the caller
/// drops this owner. The physical error is always dropped before that grant.
#[derive(Debug)]
pub(crate) struct AccountedCallerFailure<E> {
    error: Option<E>,
    construction_grant: Option<MemoryGrant>,
}

impl<E> AccountedCallerFailure<E> {
    fn from_declaration(error: E) -> Self {
        Self {
            error: Some(error),
            construction_grant: None,
        }
    }

    fn from_builder(error: E, construction_grant: MemoryGrant) -> Self {
        Self {
            error: Some(error),
            construction_grant: Some(construction_grant),
        }
    }

    pub(crate) fn error(&self) -> &E {
        self.error
            .as_ref()
            .expect("accounted callback failure retains its error")
    }
}

impl<E> Drop for AccountedCallerFailure<E> {
    fn drop(&mut self) {
        drop(self.error.take());
        drop(self.construction_grant.take());
    }
}

/// A partition failure paired with the authority that covered its payload.
///
/// The outer [`std::io::Error`] retains its original kind, while this inner
/// owner preserves the source chain and destroys the physical error before
/// releasing its construction, decode, or reader-workspace authority.
#[derive(Debug)]
struct AccountedPartitionIoFailure {
    error: Option<std::io::Error>,
    grant_authority: Option<MemoryGrant>,
    // `std::io::Error` requires its source to be `Sync`, while an open spill
    // record is deliberately only `Send`. The mutex is an ownership boundary,
    // not shared mutable access: this private reader is never exposed or locked.
    reader_authority: Option<std::sync::Mutex<BackstoppedPartitionReader>>,
}

impl AccountedPartitionIoFailure {
    fn new(error: std::io::Error, authority: MemoryGrant) -> Self {
        Self {
            error: Some(error),
            grant_authority: Some(authority),
            reader_authority: None,
        }
    }

    fn with_reader(error: std::io::Error, reader: BackstoppedPartitionReader) -> Self {
        Self {
            error: Some(error),
            grant_authority: None,
            reader_authority: Some(std::sync::Mutex::new(reader)),
        }
    }

    fn error(&self) -> &std::io::Error {
        self.error
            .as_ref()
            .expect("accounted partition I/O failure retains its source")
    }

    fn resident_memory_error(&self) -> Option<&MemoryGrantError> {
        self.error().get_ref().and_then(|source| {
            source.downcast_ref::<MemoryGrantError>().or_else(|| {
                source
                    .downcast_ref::<Self>()
                    .and_then(Self::resident_memory_error)
            })
        })
    }
}

impl std::fmt::Display for AccountedPartitionIoFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.error(), formatter)
    }
}

impl std::error::Error for AccountedPartitionIoFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error())
    }
}

impl Drop for AccountedPartitionIoFailure {
    fn drop(&mut self) {
        drop(self.error.take());
        drop(self.reader_authority.take());
        drop(self.grant_authority.take());
    }
}

/// Escaped accounted-partition callback panic paired with the authority that covered its
/// payload's construction. The original payload remains inspectable through
/// [`Self::payload`], but cannot be detached from its authority.
pub(crate) struct AccountedPartitionPanic {
    payload: Option<Box<dyn std::any::Any + Send>>,
    construction_grant: Option<MemoryGrant>,
}

impl AccountedPartitionPanic {
    fn new(payload: Box<dyn std::any::Any + Send>, construction_grant: MemoryGrant) -> Self {
        Self {
            payload: Some(payload),
            construction_grant: Some(construction_grant),
        }
    }

    #[allow(
        dead_code,
        reason = "partition callers inspect the original payload after catching the wrapper"
    )]
    pub(crate) fn payload(&self) -> &(dyn std::any::Any + Send) {
        self.payload
            .as_deref()
            .expect("accounted partition panic retains its payload")
    }
}

impl Drop for AccountedPartitionPanic {
    fn drop(&mut self) {
        drop(self.payload.take());
        drop(self.construction_grant.take());
    }
}

/// Typed failure from a mediated accounted partition replacement.
#[derive(Debug)]
pub(crate) enum PartitionUpdateError<E> {
    /// The caller's declaration or builder returned its own typed error.
    Caller(AccountedCallerFailure<E>),
    /// The partition/resource protocol rejected the operation.
    Partition(PartitionOperationError),
    /// Construction authority cannot be smaller than retained authority.
    InvalidAdmission {
        retained_upper_bound: usize,
        construction_peak: usize,
    },
    /// The completed value retained more capacity than the caller declared.
    RetainedCapacityExceeded {
        retained_upper_bound: usize,
        observed_retained: usize,
    },
}

impl<E> From<PartitionOperationError> for PartitionUpdateError<E> {
    fn from(error: PartitionOperationError) -> Self {
        Self::Partition(error)
    }
}

impl<E: std::fmt::Display> std::fmt::Display for PartitionUpdateError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Caller(failure) => write!(
                formatter,
                "partition update callback failed: {}",
                failure.error()
            ),
            Self::Partition(error) => error.fmt(formatter),
            Self::InvalidAdmission {
                retained_upper_bound,
                construction_peak,
            } => write!(
                formatter,
                "partition update construction peak {construction_peak} is smaller than its {retained_upper_bound}-byte retained bound"
            ),
            Self::RetainedCapacityExceeded {
                retained_upper_bound,
                observed_retained,
            } => write!(
                formatter,
                "partition update retained {observed_retained} bytes, exceeding its declared {retained_upper_bound}-byte bound"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for PartitionUpdateError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Caller(failure) => Some(failure.error()),
            Self::Partition(error) => Some(error),
            Self::InvalidAdmission { .. } | Self::RetainedCapacityExceeded { .. } => None,
        }
    }
}

/// A complete candidate value paired with its construction/retained authority.
struct AccountedPartitionCandidate<V> {
    value: Option<V>,
    grant: Option<MemoryGrant>,
    retained_bytes: usize,
}

impl<V> Drop for AccountedPartitionCandidate<V> {
    fn drop(&mut self) {
        drop(self.value.take());
        drop(self.grant.take());
    }
}

/// One decoded immutable spilled-base record and its child authority.
///
/// Replacement callers inspect it immutably without first publishing it;
/// compatibility callers may explicitly transfer it into the resident delta.
struct AccountedSpilledBaseEntry<V> {
    key: Option<SerializedKey>,
    entry: Option<PartitionEntry<V>>,
    grant: Option<MemoryGrant>,
}

/// Reader plus the authority for every allocation retained below it.
///
/// Destruction is a Drop-only backstop: a hostile public `OpenSpillRecord`
/// destructor cannot replace an in-flight decoder/capacity panic, and physical
/// reader state is destroyed before either workspace grant. Construction-time
/// and active read/provider panic payloads remain an audited trust boundary;
/// this guard does not claim to qualify those arbitrary callbacks.
#[derive(Debug)]
struct BackstoppedPartitionReader {
    reader: Option<SpillFileReader>,
    frame_workspace: Option<MemoryGrant>,
    reader_workspace: Option<MemoryGrant>,
    cleanup: Option<AccountedError>,
    complete: bool,
}

enum AccountedSpilledBaseLookup<V> {
    Complete(Option<AccountedSpilledBaseEntry<V>>),
    AdmissionDenied(AccountedSpilledBaseAdmissionDenial),
}

/// Known grant callback that denied one bounded base-scan attempt.
///
/// This side channel is deliberately separate from the returned I/O error:
/// retry logic must never walk an arbitrary provider/callback source chain and
/// accidentally replay user code.
enum AccountedSpilledBaseAdmissionDenial {
    ReaderWorkspace(MemoryGrantError),
    FrameWorkspace(MemoryGrantError),
    DecodeEnvelope(MemoryGrantError),
}

impl AccountedSpilledBaseAdmissionDenial {
    fn into_error(self) -> MemoryGrantError {
        match self {
            Self::ReaderWorkspace(error)
            | Self::FrameWorkspace(error)
            | Self::DecodeEnvelope(error) => error,
        }
    }
}

impl<V> AccountedSpilledBaseEntry<V> {
    fn entry(&self) -> &PartitionEntry<V> {
        self.entry
            .as_ref()
            .expect("owned spilled-base lookup retains its decoded entry")
    }

    fn stored_value_bound_bytes(&self) -> Result<usize, MemoryGrantError> {
        let key_capacity = self
            .key
            .as_ref()
            .expect("owned spilled-base lookup retains its decoded key")
            .0
            .capacity();
        let resident_bound = self.entry().resident_bound;
        key_capacity
            .checked_add(resident_bound)
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: key_capacity,
                additional_bytes: resident_bound,
            })
    }

    fn grant_mut(&mut self) -> &mut MemoryGrant {
        self.grant
            .as_mut()
            .expect("owned spilled-base lookup retains its child grant")
    }
}

impl BackstoppedPartitionReader {
    fn new(
        reader: SpillFileReader,
        frame_workspace: MemoryGrant,
        reader_workspace: MemoryGrant,
    ) -> Self {
        Self {
            reader: Some(reader),
            frame_workspace: Some(frame_workspace),
            reader_workspace: Some(reader_workspace),
            cleanup: None,
            complete: false,
        }
    }

    fn reader_mut(&mut self) -> &mut SpillFileReader {
        self.reader
            .as_mut()
            .expect("backstopped partition reader remains live")
    }

    fn finish_cleanup(&mut self) -> bool {
        let complete = self
            .reader
            .take()
            .is_none_or(SpillFileReader::teardown_for_accounted_failure);
        if !complete && let Some(cleanup) = &self.cleanup {
            cleanup.inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_failed);
        }
        self.complete = complete;
        complete
    }

    fn read_partition_entry_recording_denial(
        &mut self,
        denial: &mut Option<MemoryGrantError>,
    ) -> std::io::Result<Vec<u8>> {
        let Self {
            reader,
            frame_workspace,
            ..
        } = self;
        reader
            .as_mut()
            .expect("backstopped partition reader remains live")
            .read_partition_entry_with_admission(|required| {
                grow_workspace_recording_denial(
                    frame_workspace
                        .as_mut()
                        .expect("backstopped reader retains its frame workspace"),
                    required,
                    denial,
                )
            })
    }
}

impl Drop for BackstoppedPartitionReader {
    fn drop(&mut self) {
        if let Some(reader) = self.reader.take()
            && !reader.teardown_for_accounted_failure()
            && let Some(cleanup) = &self.cleanup
        {
            cleanup.inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_failed);
        }
        if let Some(workspace) = self.frame_workspace.take() {
            let workspace = PartitionWorkspace::new(workspace, self.cleanup.as_ref());
            if self.complete {
                workspace.release();
            }
        }
        if let Some(workspace) = self.reader_workspace.take() {
            let workspace = PartitionWorkspace::new(workspace, self.cleanup.as_ref());
            if self.complete {
                workspace.release();
            }
        }
    }
}

impl<V> Drop for AccountedSpilledBaseEntry<V> {
    fn drop(&mut self) {
        drop(self.entry.take());
        drop(self.key.take());
        drop(self.grant.take());
    }
}

/// Rollback boundary for publishing a decoded base entry into its delta.
struct AccountedBaseDeltaPublication<'a, V> {
    staged: Option<AccountedSpilledBaseEntry<V>>,
    root_grant: &'a mut MemoryGrant,
    rollback_size: usize,
    merged: bool,
    committed: bool,
}

/// Authority transfer for replacing one already-resident entry.
struct ResidentReplacementPublication<'a, V> {
    candidate: AccountedPartitionCandidate<V>,
    old_grant: Option<MemoryGrant>,
    root_grant: &'a mut MemoryGrant,
    committed: bool,
}

/// Authority transfer for publishing one absent resident entry.
struct AbsentReplacementPublication<'a, V> {
    key: AccountedSerializedKey,
    candidate: AccountedPartitionCandidate<V>,
    root_grant: &'a mut MemoryGrant,
    rollback_size: usize,
    committed: bool,
}

/// Entry in a partition: the original key columns count and value.
struct PartitionEntry<V> {
    num_key_columns: usize,
    /// Conservative retained heap bound that remains valid for spill reload.
    resident_bound: usize,
    value: V,
}

/// Native aggregate table whose allocation request is observable and whose
/// pinned growth/allocator layout can therefore be admitted before allocation.
type PartitionMap<V> = hashbrown::HashMap<
    SerializedKey,
    PartitionEntry<V>,
    RandomState,
    allocator_api2::alloc::Global,
>;

fn new_partition_map<V>() -> PartitionMap<V> {
    PartitionMap::with_hasher(RandomState::new())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrainState {
    Idle,
    Draining,
    Poisoned,
}

/// Structured failure for resource-qualified partition operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PartitionOperationError {
    /// Constructor rejection before any provider or codec operation.
    #[error(transparent)]
    Admission(PartitionAdmissionError),
    /// Constructor admission remains primary over an explicit cleanup failure.
    #[error("{error}; {phase} also failed: {cleanup}")]
    AdmissionWithCleanup {
        /// Original typed admission failure.
        error: PartitionAdmissionError,
        /// Original cleanup failure.
        cleanup: std::io::Error,
        /// Static cleanup phase.
        phase: &'static str,
    },
    /// A pre-admitted diagnostic retains all callback and cleanup authority.
    #[error("{authority}")]
    Accounted {
        /// Allocation-free public classification of the original primary.
        classification: AccountedFailureClassification,
        /// Non-detachable ownership of the original typed primary/secondaries.
        authority: AccountedError,
    },
    /// A checked admission failed before any opaque callback.
    #[error(transparent)]
    Memory(MemoryGrantError),
    /// Checked admission remains primary over an explicit cleanup failure.
    #[error("{error}; {phase} also failed: {cleanup}")]
    MemoryWithCleanup {
        /// Original checked admission failure.
        error: MemoryGrantError,
        /// Original opaque cleanup error.
        cleanup: std::io::Error,
        /// Static cleanup phase.
        phase: &'static str,
    },
    /// Cooperative execution stopped before the next publication boundary.
    #[error(transparent)]
    Cancelled(QueryCancellationError),
    /// Cancellation remains primary and explicit cleanup also failed.
    #[error("{error}; {phase} also failed: {cleanup}")]
    CancelledWithCleanup {
        /// Typed cancellation/deadline reason.
        #[source]
        error: QueryCancellationError,
        /// Secondary retryable cleanup failure.
        cleanup: std::io::Error,
        /// Static operation phase that attempted cleanup.
        phase: &'static str,
    },
    /// The allocator or platform address space rejected a retained catalog.
    #[error("allocator refused {container} capacity: {error}")]
    Allocation {
        /// Static catalog identity; constructing this error never owns an
        /// allocation-dependent message.
        container: &'static str,
        /// Original reservation failure, retained without allocation.
        #[source]
        error: TryReserveError,
    },
    /// The pinned native partition table rejected its pre-admitted allocation.
    #[error("allocator refused native partition-map capacity: {error}")]
    NativeMapAllocation {
        /// Original allocation failure retained inline without formatting.
        #[source]
        error: NativeMapAllocationError,
    },
    /// Native-map allocation failed and releasing its provisional admission
    /// also exposed a poisoned or inconsistent memory account.
    #[error(
        "allocator refused native partition-map capacity: {error}; partition grant rollback also failed: {rollback}"
    )]
    NativeMapAllocationWithRollback {
        /// Original table allocation failure.
        #[source]
        error: NativeMapAllocationError,
        /// Secondary accounting failure retained without heap conversion.
        rollback: MemoryGrantError,
    },
    /// The pinned table violated its statically precomputed layout contract.
    #[error("native partition-map invariant failed: {message}")]
    NativeMapInvariant {
        /// Allocation-free invariant description.
        message: &'static str,
    },
    /// A table-layout violation was followed by failed grant reconciliation.
    #[error(
        "native partition-map invariant failed: {message}; partition grant rollback also failed: {rollback}"
    )]
    NativeMapInvariantWithRollback {
        /// Allocation-free invariant description.
        message: &'static str,
        /// Secondary accounting failure retained without heap conversion.
        rollback: MemoryGrantError,
    },
    /// Concrete spill, codec, or resident-grant failure.
    #[error(transparent)]
    Io(std::io::Error),
    /// Original I/O diagnostic with a fixed, independently owned secondary.
    #[error("{error}; {phase} also failed: {cleanup}")]
    IoWithCleanup {
        /// Original I/O failure.
        error: std::io::Error,
        /// Original cleanup failure.
        cleanup: std::io::Error,
        /// Static cleanup phase.
        phase: &'static str,
    },
}

// Keep invariant diagnostics on their failing branch: this enum also owns
// opaque errors, so eagerly constructing it adds drop work to successful rows.
#[cold]
fn native_map_invariant(message: &'static str) -> PartitionOperationError {
    PartitionOperationError::NativeMapInvariant { message }
}

/// Private physical-plus-authority boundary shared by partition drain owners.
///
/// `Option` ownership is load-bearing: it lets `Drop` prove that physical
/// destruction completed before releasing authority. A hostile destructor is
/// contained by the spill cleanup backstop; because its retained allocation
/// can no longer be proven absent, the corresponding child grant is then
/// retained permanently rather than advertising memory that may still live.
struct AccountedPartitionOwner<T> {
    physical: Option<T>,
    grant: Option<MemoryGrant>,
}

impl<T> AccountedPartitionOwner<T> {
    fn new(physical: T, grant: MemoryGrant) -> Self {
        Self {
            physical: Some(physical),
            grant: Some(grant),
        }
    }

    fn get(&self) -> &T {
        self.physical
            .as_ref()
            .expect("live accounted partition owner retains its physical value")
    }

    fn granted_bytes(&self) -> usize {
        self.grant
            .as_ref()
            .expect("live accounted partition owner retains its child grant")
            .size()
    }
}

impl<T> Drop for AccountedPartitionOwner<T> {
    fn drop(&mut self) {
        let Some(physical) = self.physical.take() else {
            drop(self.grant.take());
            return;
        };
        let destroyed = super::run_cleanup_backstop(|| {
            drop(physical);
            Ok::<(), std::convert::Infallible>(())
        });
        let grant = self.grant.take();
        if destroyed {
            drop(grant);
        } else if let Some(grant) = grant {
            // A caught destructor failure means physical release is no longer
            // provable. Leaking this exact child is the fail-closed accounting
            // result, not ordinary cleanup behavior.
            std::mem::forget(grant);
        }
    }
}

/// Move-only decoded partition key coupled to its exact child grant.
///
/// Only shared access is exposed: the key cannot grow, be cloned with its
/// authority, or be detached from the grant. A failing physical destructor is
/// contained and permanently retains the child charge fail-closed.
///
/// ```compile_fail
/// use grafeo_core::execution::spill::AccountedPartitionKey;
///
/// fn duplicate(key: &AccountedPartitionKey) -> AccountedPartitionKey {
///     key.clone()
/// }
/// ```
///
/// ```compile_fail
/// use grafeo_core::execution::spill::AccountedPartitionKey;
///
/// fn detach(key: AccountedPartitionKey) {
///     let AccountedPartitionKey { owner } = key;
///     let _raw_owner = owner;
/// }
/// ```
///
/// ```compile_fail
/// use grafeo_common::types::Value;
/// use grafeo_core::execution::spill::AccountedPartitionKey;
///
/// fn require_growing_access<T: std::ops::DerefMut<Target = Vec<Value>>>() {}
/// fn detach_mutably() {
///     require_growing_access::<AccountedPartitionKey>();
/// }
/// ```
#[must_use = "dropping an accounted partition key releases its exact child grant"]
pub struct AccountedPartitionKey {
    owner: AccountedPartitionOwner<Vec<Value>>,
}

impl AccountedPartitionKey {
    fn new(key: Vec<Value>, grant: MemoryGrant) -> Self {
        Self {
            owner: AccountedPartitionOwner::new(key, grant),
        }
    }

    /// Returns shared, non-growing access to the decoded key.
    #[must_use]
    pub fn key(&self) -> &[Value] {
        self.owner.get()
    }

    /// Returns the exact decoded-key bound owned by this child grant.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self.owner.granted_bytes()
    }
}

/// Move-only aggregate value coupled to its declared resident-bound grant.
///
/// The boundary exposes no `&mut V` or structural detachment, and cannot be
/// cloned with its authority. An arbitrary `V` may still contain interior
/// mutability; keeping any resulting retained growth within `resident_bound`
/// remains part of the accounted constructor's audited capacity contract. A
/// failing physical destructor is contained and permanently retains the child
/// charge fail-closed.
///
/// ```compile_fail
/// use grafeo_core::execution::spill::AccountedPartitionValue;
///
/// fn duplicate(
///     value: &AccountedPartitionValue<Vec<u8>>,
/// ) -> AccountedPartitionValue<Vec<u8>> {
///     value.clone()
/// }
/// ```
///
/// ```compile_fail
/// use grafeo_core::execution::spill::AccountedPartitionValue;
///
/// fn detach(value: AccountedPartitionValue<Vec<u8>>) {
///     let AccountedPartitionValue { owner } = value;
///     let _raw_owner = owner;
/// }
/// ```
///
/// ```compile_fail
/// use grafeo_core::execution::spill::AccountedPartitionValue;
///
/// fn require_growing_access<T: std::ops::DerefMut<Target = Vec<u8>>>() {}
/// fn detach_mutably() {
///     require_growing_access::<AccountedPartitionValue<Vec<u8>>>();
/// }
/// ```
#[must_use = "dropping an accounted partition value releases its exact child grant"]
pub struct AccountedPartitionValue<V> {
    owner: AccountedPartitionOwner<V>,
}

impl<V> AccountedPartitionValue<V> {
    fn new(value: V, grant: MemoryGrant) -> Self {
        Self {
            owner: AccountedPartitionOwner::new(value, grant),
        }
    }

    /// Returns shared, non-growing access to the aggregate value.
    #[must_use]
    pub fn value(&self) -> &V {
        self.owner.get()
    }

    /// Returns the stored resident bound owned by this child grant.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self.owner.granted_bytes()
    }
}

/// One grant-owned result from a consuming accounted partition drain.
///
/// The decoded key and aggregate value each own an exact child of the same
/// backing account and region. They move together until
/// [`Self::into_accounted_parts`] deliberately gives them independent
/// lifetimes; neither boundary exposes raw physical storage or grant authority.
///
/// ```compile_fail
/// use grafeo_common::memory::buffer::MemoryGrant;
/// use grafeo_common::types::Value;
/// use grafeo_core::execution::spill::PartitionDrainEntry;
///
/// fn detach<V>(entry: PartitionDrainEntry<V>) -> (Vec<Value>, V, MemoryGrant) {
///     entry.into_parts()
/// }
/// ```
#[must_use = "dropping a partition drain entry releases both exact child grants"]
pub struct PartitionDrainEntry<V> {
    key: AccountedPartitionKey,
    value: AccountedPartitionValue<V>,
}

impl<V> PartitionDrainEntry<V> {
    /// Returns the decoded group key.
    #[must_use]
    pub fn key(&self) -> &[Value] {
        self.key.key()
    }

    /// Returns the aggregate state.
    #[must_use]
    pub fn value(&self) -> &V {
        self.value.value()
    }

    /// Consumes the entry into independently lived, authority-coupled owners.
    pub fn into_accounted_parts(self) -> (AccountedPartitionKey, AccountedPartitionValue<V>) {
        let Self { key, value } = self;
        (key, value)
    }

    /// Returns the bytes retained by both exact child grants.
    ///
    /// # Panics
    ///
    /// Panics only if the private cursor mint violated its invariant that both
    /// children were split from one representable parent grant.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self.key
            .granted_bytes()
            .checked_add(self.value.granted_bytes())
            .expect("partition drain child grants came from one representable parent")
    }
}

/// Consuming, partition-at-a-time cursor for resource-qualified aggregation.
///
/// Construction does not cross the consuming boundary until every resident
/// delta has a durable partition representation. Once constructed, any error
/// is terminal: the cursor cleans all remaining owned state, and `Drop` is a
/// retrying backstop when the caller abandons iteration.
pub struct PartitionDrainCursor<'a, V: Clone + Send + Sync + 'static> {
    state: &'a mut PartitionedState<V>,
    next_partition: usize,
    active_partition: Option<usize>,
    reader: Option<SpillFileReader>,
    reader_grant: Option<PartitionWorkspace>,
    resident: Option<resident_cursor::ResidentPartition<V>>,
    remaining: u64,
    finished: bool,
}

impl From<std::io::Error> for PartitionOperationError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<QueryCancellationError> for PartitionOperationError {
    fn from(error: QueryCancellationError) -> Self {
        Self::Cancelled(error)
    }
}

impl PartitionOperationError {
    // Inspect only direct typed cancellation and the core-owned I/O envelope.
    // Opaque sources and matching I/O kinds do not establish cancellation.
    fn cancellation_error(&self) -> Option<&QueryCancellationError> {
        let mut error = match self {
            Self::Cancelled(reason) | Self::CancelledWithCleanup { error: reason, .. } => {
                return Some(reason);
            }
            Self::Io(error) | Self::IoWithCleanup { error, .. } => error,
            _ => return None,
        };
        loop {
            let payload = error.get_ref()?;
            if let Some(reason) = payload.downcast_ref::<QueryCancellationError>() {
                return Some(reason);
            }
            error = payload
                .downcast_ref::<AccountedPartitionIoFailure>()?
                .error
                .as_ref()?;
        }
    }

    /// Returns the underlying resident-memory grant failure, when this error
    /// represents structured capacity exhaustion.
    #[must_use]
    pub fn resident_memory_error(&self) -> Option<&MemoryGrantError> {
        match self {
            Self::Admission(PartitionAdmissionError::Memory(error))
            | Self::AdmissionWithCleanup {
                error: PartitionAdmissionError::Memory(error),
                ..
            } => Some(error),
            Self::Admission(_) | Self::AdmissionWithCleanup { .. } => None,
            Self::Memory(error) | Self::MemoryWithCleanup { error, .. } => Some(error),
            Self::Accounted {
                classification: AccountedFailureClassification::ResidentMemory(error),
                ..
            } => Some(error),
            Self::Accounted { .. } => None,
            Self::Io(error) | Self::IoWithCleanup { error, .. } => {
                let source = error.get_ref()?;
                source.downcast_ref::<MemoryGrantError>().or_else(|| {
                    source
                        .downcast_ref::<AccountedPartitionIoFailure>()
                        .and_then(AccountedPartitionIoFailure::resident_memory_error)
                })
            }
            Self::Cancelled(_)
            | Self::CancelledWithCleanup { .. }
            | Self::Allocation { .. }
            | Self::NativeMapAllocation { .. }
            | Self::NativeMapAllocationWithRollback { .. }
            | Self::NativeMapInvariant { .. }
            | Self::NativeMapInvariantWithRollback { .. } => None,
        }
    }

    fn into_io(self) -> std::io::Error {
        match self {
            Self::Admission(_) => std::io::ErrorKind::OutOfMemory.into(),
            Self::AdmissionWithCleanup { cleanup, .. } => {
                let _ = super::run_cleanup_backstop(|| {
                    drop(cleanup);
                    Ok::<_, ()>(())
                });
                std::io::ErrorKind::OutOfMemory.into()
            }
            Self::Memory(error) => grant_io_error(error),
            Self::MemoryWithCleanup {
                error,
                cleanup,
                phase,
            } => super::combine_primary_and_cleanup(grant_io_error(error), cleanup, phase),
            Self::Accounted { authority, .. } => {
                // Compatibility entry points reject this owner before work.
                // Its Drop retires payloads before authority and quarantines
                // hostile destruction if an invalid crossing still occurs.
                let _ = super::run_cleanup_backstop(|| {
                    drop(authority);
                    Ok::<_, ()>(())
                });
                std::io::ErrorKind::Unsupported.into()
            }
            Self::Cancelled(_) | Self::CancelledWithCleanup { .. } => unreachable!(
                "resource-qualified partition cancellation cannot cross the compatibility I/O API"
            ),
            // Preserve the legacy I/O surface without allocating while
            // handling the allocator's refusal.
            Self::Allocation { .. }
            | Self::NativeMapAllocation { .. }
            | Self::NativeMapAllocationWithRollback { .. } => {
                std::io::ErrorKind::OutOfMemory.into()
            }
            Self::NativeMapInvariant { .. } | Self::NativeMapInvariantWithRollback { .. } => {
                std::io::ErrorKind::InvalidData.into()
            }
            Self::Io(error) => error,
            Self::IoWithCleanup {
                error,
                cleanup,
                phase,
            } => super::combine_primary_and_cleanup(error, cleanup, phase),
        }
    }
}

fn with_cleanup(
    primary: PartitionOperationError,
    cleanup: std::io::Error,
    phase: &'static str,
) -> PartitionOperationError {
    match primary {
        PartitionOperationError::Admission(error) => {
            PartitionOperationError::AdmissionWithCleanup {
                error,
                cleanup,
                phase,
            }
        }
        PartitionOperationError::AdmissionWithCleanup {
            error,
            cleanup: first,
            phase: first_phase,
        } => {
            let _ = super::run_cleanup_backstop(|| {
                drop(cleanup);
                Ok::<_, ()>(())
            });
            PartitionOperationError::AdmissionWithCleanup {
                error,
                cleanup: first,
                phase: first_phase,
            }
        }
        PartitionOperationError::Memory(error) => PartitionOperationError::MemoryWithCleanup {
            error,
            cleanup,
            phase,
        },
        PartitionOperationError::MemoryWithCleanup {
            error,
            cleanup: existing,
            phase: existing_phase,
        } => PartitionOperationError::MemoryWithCleanup {
            error,
            cleanup: super::combine_primary_and_cleanup(existing, cleanup, phase),
            phase: existing_phase,
        },
        PartitionOperationError::Accounted { .. } => {
            // Accounted callers retain their fixed secondary in the admitted
            // transport; this compatibility helper must never unwrap it.
            let _ = super::run_cleanup_backstop(|| {
                drop(cleanup);
                Ok::<_, ()>(())
            });
            primary
        }
        PartitionOperationError::Cancelled(error) => {
            PartitionOperationError::CancelledWithCleanup {
                error,
                cleanup,
                phase,
            }
        }
        PartitionOperationError::CancelledWithCleanup {
            error,
            cleanup: existing,
            phase: existing_phase,
        } => PartitionOperationError::CancelledWithCleanup {
            error,
            cleanup: super::combine_primary_and_cleanup(existing, cleanup, phase),
            phase: existing_phase,
        },
        PartitionOperationError::Allocation { .. }
        | PartitionOperationError::NativeMapAllocation { .. }
        | PartitionOperationError::NativeMapAllocationWithRollback { .. }
        | PartitionOperationError::NativeMapInvariant { .. }
        | PartitionOperationError::NativeMapInvariantWithRollback { .. } => unreachable!(
            "container reservation fails before partition publication and cannot require cleanup"
        ),
        PartitionOperationError::Io(error) => PartitionOperationError::IoWithCleanup {
            error,
            cleanup,
            phase,
        },
        PartitionOperationError::IoWithCleanup {
            error,
            cleanup: first,
            phase: first_phase,
        } => {
            let _ = super::run_cleanup_backstop(|| {
                drop(cleanup);
                Ok::<_, ()>(())
            });
            PartitionOperationError::IoWithCleanup {
                error,
                cleanup: first,
                phase: first_phase,
            }
        }
    }
}

fn check_cancellation(
    cancellation: Option<&QueryCancellationToken>,
) -> Result<(), QueryCancellationError> {
    match cancellation {
        Some(cancellation) => cancellation.check(),
        None => Ok(()),
    }
}

fn partition_memory_error(error: MemoryGrantError) -> PartitionOperationError {
    PartitionOperationError::Io(grant_io_error(error))
}

fn accounted_partition_io_failure(
    error: std::io::Error,
    authority: MemoryGrant,
) -> PartitionOperationError {
    let kind = error.kind();
    PartitionOperationError::Io(std::io::Error::new(
        kind,
        AccountedPartitionIoFailure::new(error, authority),
    ))
}

fn accounted_partition_reader_io_failure(
    error: std::io::Error,
    reader: BackstoppedPartitionReader,
) -> PartitionOperationError {
    let kind = error.kind();
    PartitionOperationError::Io(std::io::Error::new(
        kind,
        AccountedPartitionIoFailure::with_reader(error, reader),
    ))
}

fn is_grant_denial(error: &MemoryGrantError) -> bool {
    matches!(
        error,
        MemoryGrantError::LimitExceeded { .. } | MemoryGrantError::Denied { .. }
    )
}

fn is_grant_denial_io(error: &std::io::Error) -> bool {
    error
        .get_ref()
        .and_then(|source| source.downcast_ref::<MemoryGrantError>())
        .is_some_and(is_grant_denial)
}

fn grant_io_error(error: MemoryGrantError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::OutOfMemory, error)
}

fn grow_workspace(workspace: &mut MemoryGrant, required: usize) -> std::io::Result<()> {
    if required > workspace.size() {
        workspace.try_resize(required).map_err(grant_io_error)?;
    }
    Ok(())
}

fn grow_workspace_recording_denial(
    workspace: &mut MemoryGrant,
    required: usize,
    denial: &mut Option<MemoryGrantError>,
) -> std::io::Result<()> {
    if required <= workspace.size() {
        return Ok(());
    }
    match workspace.try_resize(required) {
        Ok(()) => Ok(()),
        Err(error) => {
            if is_grant_denial(&error) {
                *denial = Some(error.clone());
            }
            Err(grant_io_error(error))
        }
    }
}

fn create_accounted_partition_file(
    manager: &SpillManager,
    root_grant: &mut MemoryGrant,
    cleanup: Option<&AccountedError>,
) -> std::io::Result<(SpillFile, MemoryGrant)> {
    let requested = qualified_writer_buffer_requested_bytes();
    let provisional = requested
        .checked_mul(2)
        .ok_or_else(|| std::io::Error::other("partition writer-buffer grant overflow"))?;
    let mut workspace = PartitionWorkspace::new(
        root_grant.split(0).ok_or(std::io::ErrorKind::InvalidData)?,
        cleanup,
    );
    grow_workspace(
        workspace
            .grant_mut()
            .map_err(PartitionOperationError::into_io)?,
        provisional,
    )?;
    let writer_buffer = SpillWriterBuffer::prepare_with_capacity(requested)?;
    let observed = writer_buffer.capacity();
    if observed
        > workspace
            .grant_mut()
            .map_err(PartitionOperationError::into_io)?
            .size()
        && let Err(error) = grow_workspace(
            workspace
                .grant_mut()
                .map_err(PartitionOperationError::into_io)?,
            observed,
        )
    {
        drop(writer_buffer);
        return Err(error);
    }
    let file = manager.create_qualified_file_with_writer_buffer(
        SpillFileRole::NativePartition,
        writer_buffer,
        |provider_required| {
            let total = observed
                .checked_add(provider_required)
                .ok_or_else(|| std::io::Error::other("partition file-workspace grant overflow"))?;
            grow_workspace(
                workspace
                    .grant_mut()
                    .map_err(PartitionOperationError::into_io)?,
                total,
            )
        },
    )?;
    Ok((
        file,
        workspace
            .into_grant()
            .map_err(PartitionOperationError::into_io)?,
    ))
}

/// Owns a partition staging file and any grant covering its physical writer
/// and provider state. Custom destruction contains hostile provider drops and
/// destroys the complete file before releasing its workspace authority.
struct PartitionStaging {
    file: Option<SpillFile>,
    workspace: Option<MemoryGrant>,
    cleanup: Option<AccountedError>,
}

impl PartitionStaging {
    fn accounted(
        file: SpillFile,
        workspace: MemoryGrant,
        cleanup: Option<&AccountedError>,
    ) -> Self {
        Self {
            file: Some(file),
            workspace: Some(workspace),
            cleanup: cleanup.cloned(),
        }
    }

    fn unaccounted(file: SpillFile) -> Self {
        Self {
            file: Some(file),
            workspace: None,
            cleanup: None,
        }
    }

    fn file_mut(&mut self) -> &mut SpillFile {
        self.file
            .as_mut()
            .expect("partition staging file remains owned until publication")
    }

    fn finish_write(&mut self) -> std::io::Result<()> {
        let cleanup = self.cleanup.clone();
        match cleanup.as_ref() {
            Some(cleanup) => self.file_mut().finish_partition_write(cleanup),
            None => self.file_mut().finish_write(),
        }
    }

    // Call only after `finish_write`: that transition has already destroyed
    // the writer and per-file OpenSpillRecord covered by this workspace.
    fn into_finished_file(mut self) -> SpillFile {
        drop(self.workspace.take());
        self.file
            .take()
            .expect("finished partition staging retains its file")
    }

    fn cleanup_preserving_primary(
        &mut self,
        primary: PartitionOperationError,
        phase: &'static str,
    ) -> PartitionOperationError {
        let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let cleanup = self.cleanup.clone();
            match cleanup.as_ref() {
                Some(cleanup) => self.file_mut().close_partition_and_delete(cleanup),
                None => self.file_mut().close_and_delete(),
            }
        }));
        match cleanup {
            Ok(Ok(())) => primary,
            Ok(Err(error)) if self.cleanup.is_some() => {
                self.cleanup.as_ref().and_then(|cleanup| {
                    cleanup.inspect::<PartitionFailureCleanup, _>(|witness| {
                        witness.retain_cleanup_error(error);
                    })
                });
                primary
            }
            Ok(Err(cleanup)) => with_cleanup(primary, cleanup, phase),
            Err(panic) => {
                if let Some(cleanup) = &self.cleanup {
                    cleanup.inspect::<PartitionFailureCleanup, _>(|witness| {
                        witness.retain_cleanup_panic(panic);
                    });
                } else {
                    super::forget_cleanup_failure(panic);
                }
                primary
            }
        }
    }
}

impl Drop for PartitionStaging {
    fn drop(&mut self) {
        if let Some(file) = self.file.take()
            && !super::run_cleanup_backstop(|| {
                if file.retire_owned_file() {
                    Ok(())
                } else {
                    Err(())
                }
            })
            && let Some(cleanup) = &self.cleanup
        {
            cleanup.inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_failed);
        }
        if let Some(workspace) = self.workspace.take() {
            drop(PartitionWorkspace::new(workspace, self.cleanup.as_ref()));
        }
    }
}

struct GrantRecordWriter {
    bytes: Vec<u8>,
    grant: MemoryGrant,
    maximum: usize,
}

impl GrantRecordWriter {
    fn new(root_grant: &mut MemoryGrant, maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            grant: root_grant
                .split(0)
                .expect("zero-byte record grant can always be split"),
            maximum,
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    fn prepare_capacity(&mut self, required: usize) -> std::io::Result<()> {
        self.prepare_capacity_inner(required, true)
    }

    fn prepare_capacity_inner(&mut self, required: usize, reconcile: bool) -> std::io::Result<()> {
        if required > self.maximum {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "partition payload length {required} exceeds maximum {}",
                    self.maximum
                ),
            ));
        }
        if required <= self.bytes.capacity() {
            return Ok(());
        }

        // `try_reserve_exact` may round its physical capacity. Admit a
        // provisional growth envelope before asking it to allocate, then
        // reconcile to the observed capacity (including any excess rounding).
        let provisional = required
            .checked_mul(2)
            .ok_or_else(|| std::io::Error::other("partition staging grant overflow"))?;
        grow_workspace(&mut self.grant, provisional)?;
        let additional = required.saturating_sub(self.bytes.len());
        if self.bytes.try_reserve_exact(additional).is_err() {
            self.discard_capacity();
            return Err(std::io::ErrorKind::OutOfMemory.into());
        }
        if self.bytes.capacity() > self.grant.size()
            && let Err(error) = grow_workspace(&mut self.grant, self.bytes.capacity())
        {
            self.discard_capacity();
            return Err(error);
        }
        if reconcile && let Err(error) = self.grant.try_resize(self.bytes.capacity()) {
            self.discard_capacity();
            return Err(grant_io_error(error));
        }
        Ok(())
    }

    fn patch_u64_le(&mut self, offset: usize, value: u64) -> std::io::Result<()> {
        let end = offset
            .checked_add(8)
            .ok_or_else(|| std::io::Error::other("partition patch offset overflow"))?;
        let destination = self.bytes.get_mut(offset..end).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "partition patch range exceeds staged record",
            )
        })?;
        destination.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn discard_capacity(&mut self) {
        self.bytes = Vec::new();
        let _ = self.grant.try_resize(0);
    }
}

impl Write for GrantRecordWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("partition payload length overflow"))?;
        if next > self.maximum {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "partition payload length {next} exceeds maximum {}",
                    self.maximum
                ),
            ));
        }
        if next > self.bytes.capacity() {
            let provisional = next
                .checked_mul(2)
                .ok_or_else(|| std::io::Error::other("partition staging grant overflow"))?;
            grow_workspace(&mut self.grant, provisional)?;
            let additional = next.saturating_sub(self.bytes.len());
            if self.bytes.try_reserve_exact(additional).is_err() {
                self.discard_capacity();
                return Err(std::io::ErrorKind::OutOfMemory.into());
            }
            if self.bytes.capacity() > self.grant.size()
                && let Err(error) = grow_workspace(&mut self.grant, self.bytes.capacity())
            {
                self.discard_capacity();
                return Err(error);
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl AccountedCounterScratch {
    fn prepare(
        root_grant: &mut MemoryGrant,
        required_entries: usize,
        required_key_bytes: usize,
    ) -> std::io::Result<Self> {
        let requested =
            CounterSortScratch::requested_capacity_bytes(required_entries, required_key_bytes)?;
        let provisional = requested
            .checked_mul(2)
            .ok_or_else(|| std::io::Error::other("counter sort scratch grant overflow"))?;
        let mut grant = root_grant
            .split(0)
            .expect("zero-byte counter scratch grant can always be split");
        grow_workspace(&mut grant, provisional)?;

        // The grant is live before either scratch Vec is allowed to grow.
        let mut scratch = CounterSortScratch::new();
        let observed = match scratch.prepare(required_entries, required_key_bytes) {
            Ok(observed) => observed,
            Err(error) => {
                scratch.discard_capacity();
                let _ = grant.try_resize(0);
                return Err(error);
            }
        };
        let observed_bytes = CounterSortScratch::requested_capacity_bytes(
            observed.entry_capacity,
            observed.key_capacity,
        )?;
        if observed_bytes > grant.size()
            && let Err(error) = grow_workspace(&mut grant, observed_bytes)
        {
            scratch.discard_capacity();
            let _ = grant.try_resize(0);
            return Err(error);
        }
        if let Err(error) = grant.try_resize(observed_bytes) {
            scratch.discard_capacity();
            let _ = grant.try_resize(0);
            return Err(grant_io_error(error));
        }
        Ok(Self {
            scratch,
            _grant: grant,
        })
    }
}

impl AccountedSerializedKey {
    fn from_values(
        values: &[Value],
        limits: SpillFrameLimits,
        root_grant: &mut MemoryGrant,
    ) -> std::io::Result<Self> {
        Self::from_values_with(
            values,
            limits,
            root_grant,
            |row, writer, codec_limits, scratch| {
                serialize_row_with_prepared_scratch(row, writer, codec_limits, scratch)
            },
        )
    }

    fn from_values_with<F>(
        values: &[Value],
        limits: SpillFrameLimits,
        root_grant: &mut MemoryGrant,
        encode: F,
    ) -> std::io::Result<Self>
    where
        F: FnOnce(
            &[Value],
            &mut GrantRecordWriter,
            CodecLimits,
            &mut CounterSortScratch,
        ) -> std::io::Result<usize>,
    {
        let measurement = measure_serialized_row_with_limits(values, limits.codec_limits())?;
        let maximum = usize::try_from(limits.max_plaintext_bytes()).unwrap_or(usize::MAX);
        let mut writer = GrantRecordWriter::new(root_grant, maximum);
        writer.prepare_capacity(measurement.encoded_bytes)?;
        let mut counter_scratch = AccountedCounterScratch::prepare(
            root_grant,
            measurement.counter_sort_entries,
            measurement.counter_sort_key_bytes,
        )?;
        let encoded = encode(
            values,
            &mut writer,
            limits.codec_limits(),
            &mut counter_scratch.scratch,
        )?;
        if encoded != measurement.encoded_bytes || writer.bytes.len() != measurement.encoded_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "serialized partition key differs from its measured length",
            ));
        }

        let GrantRecordWriter { bytes, grant, .. } = writer;
        debug_assert_eq!(grant.size(), bytes.capacity());
        Ok(Self {
            key: Some(SerializedKey(bytes)),
            grant: Some(grant),
        })
    }
}

impl<'a> AccountedKeyPublication<'a> {
    fn new(staged: AccountedSerializedKey, root_grant: &'a mut MemoryGrant) -> Self {
        let rollback_size = root_grant.size();
        Self {
            staged: Some(staged),
            root_grant,
            rollback_size,
            merged: false,
            committed: false,
        }
    }

    fn merge_key_grant(&mut self) -> std::io::Result<()> {
        let staged = self
            .staged
            .as_mut()
            .expect("uncommitted key publication retains its staging owner");
        let key_grant = staged
            .grant
            .take()
            .expect("uncommitted accounted key retains its child grant");
        match self.root_grant.try_merge(key_grant) {
            Ok(()) => {
                self.merged = true;
                Ok(())
            }
            Err(key_grant) => {
                staged.grant = Some(key_grant);
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "serialized-key grant is incompatible with its partition root",
                ))
            }
        }
    }

    fn publish_into<V, F>(
        mut self,
        partition: &mut PartitionMap<V>,
        value: PartitionEntry<V>,
        after_merge: F,
    ) -> std::io::Result<&mut PartitionEntry<V>>
    where
        F: FnOnce(),
    {
        self.merge_key_grant()?;
        // This hook is inert in production. Hostile tests use it to prove the
        // guard restores authority before preserving an arbitrary panic.
        after_merge();

        let key = self
            .staged
            .as_mut()
            .expect("uncommitted publication retains its staged key")
            .key
            .take()
            .expect("uncommitted publication retains its physical key");
        let hashbrown::hash_map::Entry::Vacant(vacant) = partition.entry(key) else {
            drop(value);
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "serialized key became resident before publication",
            ));
        };
        // Map capacity was pre-reserved, and hashing these owned bytes is
        // allocation-free. Vacant-entry installation is therefore the
        // allocation-free publication point.
        let published = vacant.insert(value);
        self.committed = true;
        drop(self.staged.take());
        Ok(published)
    }
}

impl Drop for AccountedKeyPublication<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // If publication unwinds after the merge, destroy any still-staged
        // physical bytes before removing their authority from the root.
        drop(self.staged.take());
        if self.merged {
            let _ = self.root_grant.try_resize(self.rollback_size);
        }
    }
}

impl<'a, V> AccountedBaseDeltaPublication<'a, V> {
    fn new(staged: AccountedSpilledBaseEntry<V>, root_grant: &'a mut MemoryGrant) -> Self {
        Self {
            staged: Some(staged),
            rollback_size: root_grant.size(),
            root_grant,
            merged: false,
            committed: false,
        }
    }

    fn publish_into(mut self, partition: &mut PartitionMap<V>) -> std::io::Result<()> {
        let grant = self
            .staged
            .as_mut()
            .expect("uncommitted base publication retains its decoded entry")
            .grant
            .take()
            .expect("uncommitted base publication retains its child grant");
        if let Err(grant) = self.root_grant.try_merge(grant) {
            self.staged
                .as_mut()
                .expect("failed base publication retains its decoded entry")
                .grant = Some(grant);
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "spilled-base grant is incompatible with its partition root",
            ));
        }
        self.merged = true;

        let staged = self
            .staged
            .as_mut()
            .expect("uncommitted base publication retains its decoded entry");
        let key = staged
            .key
            .take()
            .expect("uncommitted base publication retains its decoded key");
        let entry = staged
            .entry
            .take()
            .expect("uncommitted base publication retains its decoded value");
        let hashbrown::hash_map::Entry::Vacant(vacant) = partition.entry(key) else {
            drop(entry);
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "spilled-base key became resident before delta publication",
            ));
        };
        vacant.insert(entry);
        self.committed = true;
        drop(self.staged.take());
        Ok(())
    }
}

impl<V> Drop for AccountedBaseDeltaPublication<'_, V> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // Drop decoded physical state before removing any authority that was
        // already merged into the root.
        drop(self.staged.take());
        if self.merged {
            let _ = self.root_grant.try_resize(self.rollback_size);
        }
    }
}

impl<'a, V> ResidentReplacementPublication<'a, V> {
    fn begin(
        root_grant: &'a mut MemoryGrant,
        old_retained_bytes: usize,
        mut candidate: AccountedPartitionCandidate<V>,
    ) -> Result<Self, PartitionOperationError> {
        let old_grant = root_grant.split(old_retained_bytes).ok_or_else(|| {
            partition_memory_error(MemoryGrantError::ArithmeticOverflow {
                current_bytes: root_grant.size(),
                additional_bytes: old_retained_bytes,
            })
        })?;
        let candidate_grant = candidate
            .grant
            .take()
            .expect("unpublished replacement retains its candidate grant");
        if let Err(candidate_grant) = root_grant.try_merge(candidate_grant) {
            candidate.grant = Some(candidate_grant);
            root_grant
                .try_merge(old_grant)
                .expect("split old-value grant must merge back into its root");
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "replacement grant is incompatible with its partition root",
            )
            .into());
        }
        Ok(Self {
            candidate,
            old_grant: Some(old_grant),
            root_grant,
            committed: false,
        })
    }

    fn publish(mut self, slot: &mut PartitionEntry<V>, resident_bound: usize) {
        let replacement = PartitionEntry {
            num_key_columns: slot.num_key_columns,
            resident_bound,
            value: self
                .candidate
                .value
                .take()
                .expect("unpublished replacement retains its complete value"),
        };
        let old = std::mem::replace(slot, replacement);
        self.committed = true;
        drop(old);
        drop(self.old_grant.take());
    }
}

impl<V> Drop for ResidentReplacementPublication<'_, V> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // The root currently accounts for the candidate. Destroy its physical
        // value before restoring the old value's split authority.
        drop(self.candidate.value.take());
        let candidate_grant = self
            .root_grant
            .split(self.candidate.retained_bytes)
            .expect("uncommitted root retains the candidate authority");
        self.candidate.grant = Some(candidate_grant);
        self.root_grant
            .try_merge(
                self.old_grant
                    .take()
                    .expect("uncommitted replacement retains old-value authority"),
            )
            .expect("old-value child grant must merge back into its root");
    }
}

impl<'a, V> AbsentReplacementPublication<'a, V> {
    fn new(
        root_grant: &'a mut MemoryGrant,
        key: AccountedSerializedKey,
        candidate: AccountedPartitionCandidate<V>,
    ) -> Self {
        let rollback_size = root_grant.size();
        Self {
            key,
            candidate,
            root_grant,
            rollback_size,
            committed: false,
        }
    }

    fn merge_candidate_grant(&mut self) -> std::io::Result<()> {
        let grant = self
            .candidate
            .grant
            .take()
            .expect("unpublished absent candidate retains its grant");
        match self.root_grant.try_merge(grant) {
            Ok(()) => Ok(()),
            Err(grant) => {
                self.candidate.grant = Some(grant);
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "candidate grant is incompatible with its partition root",
                ))
            }
        }
    }

    fn merge_key_grant(&mut self) -> std::io::Result<()> {
        let grant = self
            .key
            .grant
            .take()
            .expect("unpublished absent key retains its grant");
        match self.root_grant.try_merge(grant) {
            Ok(()) => Ok(()),
            Err(grant) => {
                self.key.grant = Some(grant);
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "serialized-key grant is incompatible with its partition root",
                ))
            }
        }
    }

    fn publish(
        mut self,
        partition: &mut PartitionMap<V>,
        num_key_columns: usize,
        resident_bound: usize,
    ) -> std::io::Result<()> {
        self.merge_candidate_grant()?;
        self.merge_key_grant()?;
        let key = self
            .key
            .key
            .take()
            .expect("unpublished absent update retains its physical key");
        let value = self
            .candidate
            .value
            .take()
            .expect("unpublished absent update retains its complete value");
        let hashbrown::hash_map::Entry::Vacant(vacant) = partition.entry(key) else {
            drop(value);
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "absent replacement key became resident before publication",
            ));
        };
        vacant.insert(PartitionEntry {
            num_key_columns,
            resident_bound,
            value,
        });
        self.committed = true;
        Ok(())
    }
}

impl<V> Drop for AbsentReplacementPublication<'_, V> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // Destroy both physical allocations before releasing any authority
        // that may already have transferred into the root.
        drop(self.candidate.value.take());
        drop(self.key.key.take());
        drop(self.candidate.grant.take());
        drop(self.key.grant.take());
        let _ = self.root_grant.try_resize(self.rollback_size);
    }
}

fn write_accounted_partition_entry<V>(
    spill_file: &mut SpillFile,
    key: &SerializedKey,
    entry: &PartitionEntry<V>,
    serializer: &(dyn Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync),
    frame_limits: SpillFrameLimits,
    root_grant: &mut MemoryGrant,
    cleanup: Option<&AccountedError>,
) -> std::io::Result<()> {
    let maximum = usize::try_from(frame_limits.max_plaintext_bytes()).unwrap_or(usize::MAX);
    // The protected GroupState codec's only independent scratch is Counter
    // sorting. Its two backing arrays peak at twice the retained value bound;
    // this is physical authority, separate from idle recovery scheduling.
    let mut codec_workspace = if cleanup.is_some() {
        let bytes = entry.resident_bound.checked_mul(2).ok_or_else(|| {
            grant_io_error(MemoryGrantError::ArithmeticOverflow {
                current_bytes: entry.resident_bound,
                additional_bytes: entry.resident_bound,
            })
        })?;
        let grant = root_grant.split(0).ok_or(std::io::ErrorKind::InvalidData)?;
        let mut workspace = PartitionWorkspace::new(grant, cleanup);
        grow_workspace(
            workspace
                .grant_mut()
                .map_err(PartitionOperationError::into_io)?,
            bytes,
        )?;
        Some(workspace)
    } else {
        None
    };
    let mut provider_grant = root_grant
        .split(0)
        .expect("zero-byte provider grant can always be split");
    let mut record = GrantRecordWriter::new(root_grant, maximum);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        write_u64(
            &mut record,
            u64::try_from(key.0.len()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "partition key length exceeds u64",
                )
            })?,
        )?;
        record.write_all(&key.0)?;
        write_u64(
            &mut record,
            u64::try_from(entry.num_key_columns).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "partition key column count exceeds u64",
                )
            })?,
        )?;
        let custom_length_offset = record.as_slice().len();
        write_u64(&mut record, 0)?;
        let custom_start = record.as_slice().len();
        serializer(&entry.value, &mut record, frame_limits.codec_limits())?;
        // Counter scratch has physically retired before serializer success.
        // Provider sealing has its own separately admitted workspace.
        if let Some(workspace) = codec_workspace.take() {
            workspace.release();
        }
        let custom_len = record
            .as_slice()
            .len()
            .checked_sub(custom_start)
            .ok_or_else(|| std::io::Error::other("partition custom-state length underflow"))?;
        record.patch_u64_le(
            custom_length_offset,
            u64::try_from(custom_len).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "partition custom-state length exceeds u64",
                )
            })?,
        )?;
        write_u64(
            &mut record,
            u64::try_from(entry.resident_bound).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "partition resident bound exceeds u64",
                )
            })?,
        )?;
        spill_file.write_partition_entry_with_admission(record.as_slice(), |required| {
            grow_workspace(&mut provider_grant, required)
        })
    }));
    let GrantRecordWriter { bytes, grant, .. } = record;
    drop(bytes);
    let record_grant = PartitionWorkspace::new(grant, cleanup);
    let provider_grant = PartitionWorkspace::new(provider_grant, cleanup);
    match outcome {
        Ok(Ok(())) => {
            record_grant.release();
            provider_grant.release();
            Ok(())
        }
        Ok(Err(error)) => Err(error),
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn read_accounted_partition_entry(
    reader: &mut SpillFileReader,
    workspace: &mut MemoryGrant,
) -> std::io::Result<Vec<u8>> {
    reader.read_partition_entry_with_admission(|required| grow_workspace(workspace, required))
}

fn open_partition_reader(
    file: &SpillFile,
    admit: impl FnMut(usize) -> std::io::Result<()>,
    cleanup: Option<&AccountedError>,
) -> std::io::Result<SpillFileReader> {
    match cleanup {
        Some(cleanup) => file.reader_with_partition_admission_and_cleanup(admit, cleanup.clone()),
        None => file.reader_with_admission(admit),
    }
}

/// The five parallel catalogs retained by [`PartitionedState`].
///
/// Keeping construction separate makes the accounted path transactional: no
/// catalog is published until every fallible reservation has succeeded.
struct PartitionCatalogs<V> {
    partitions: Vec<Option<PartitionMap<V>>>,
    spill_files: Vec<Option<SpillFile>>,
    spill_base_sizes: Vec<usize>,
    partition_sizes: Vec<usize>,
    access_times: Vec<u64>,
}

/// Constructor failures occur before provider or codec callbacks can run.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PartitionAdmissionError {
    /// The buffer manager refused the constructor's admitted storage.
    #[error(transparent)]
    Memory(MemoryGrantError),
    /// Native allocation failed before any callback ran.
    #[error("allocator refused {container}: {error}")]
    Allocation {
        /// Catalog being allocated.
        container: &'static str,
        /// Original allocation failure.
        error: TryReserveError,
    },
    /// The fixed error publication allocation failed.
    #[error("allocator refused partition failure publication")]
    PublisherAllocation,
    /// Error publication allocation and its grant rollback both failed.
    #[error("allocator refused partition failure publication; rollback failed: {0}")]
    PublisherAllocationWithRollback(MemoryGrantError),
    /// A constructor child unexpectedly already contained authority.
    #[error("partition failure publisher received {bytes} bytes")]
    NonZeroPublisherGrant {
        /// Unexpected admitted byte count.
        bytes: usize,
    },
    /// The fixed publisher rejected an internal layout invariant.
    #[error("partition failure publisher invariant failed")]
    PublisherInvariant,
    /// A custom hook did not declare a bounded failure workspace.
    #[error("partition I/O hooks do not declare a bounded workspace")]
    UnboundedHooks,
}

impl From<PartitionAdmissionError> for PartitionOperationError {
    fn from(error: PartitionAdmissionError) -> Self {
        match error {
            PartitionAdmissionError::Allocation { container, error } => {
                Self::Allocation { container, error }
            }
            PartitionAdmissionError::Memory(error) => partition_memory_error(error),
            other => Self::Admission(other),
        }
    }
}

fn partition_publisher_admission(
    error: AccountedErrorPublisherBuildError,
) -> PartitionAdmissionError {
    let failure = match error.failure() {
        AccountedErrorPublisherBuildFailure::Admission(error) => {
            PartitionAdmissionError::Memory(error.clone())
        }
        AccountedErrorPublisherBuildFailure::Allocation => {
            PartitionAdmissionError::PublisherAllocation
        }
        AccountedErrorPublisherBuildFailure::AllocationWithRollback(error) => {
            PartitionAdmissionError::PublisherAllocationWithRollback(error.clone())
        }
        AccountedErrorPublisherBuildFailure::NonZeroGrant { bytes } => {
            PartitionAdmissionError::NonZeroPublisherGrant { bytes: *bytes }
        }
        _ => PartitionAdmissionError::PublisherInvariant,
    };
    // No callback has run and no payload occupies the unpublished block.
    drop(error);
    failure
}

impl<V> PartitionCatalogs<V> {
    fn new(num_partitions: usize) -> Self {
        let mut partitions = Vec::with_capacity(num_partitions);
        let mut spill_files = Vec::with_capacity(num_partitions);
        for _ in 0..num_partitions {
            partitions.push(Some(new_partition_map()));
            spill_files.push(None);
        }

        Self {
            partitions,
            spill_files,
            spill_base_sizes: vec![0; num_partitions],
            partition_sizes: vec![0; num_partitions],
            access_times: vec![0; num_partitions],
        }
    }

    fn try_new(num_partitions: usize) -> Result<Self, PartitionAdmissionError> {
        // Reserve the widest catalog first. Besides failing early, this makes
        // impossible address-space requests deterministic without asking the
        // allocator for a merely very large but representable allocation.
        let mut spill_files = Vec::new();
        Self::try_reserve(
            &mut spill_files,
            num_partitions,
            "partition spill-file catalog",
        )?;

        let mut partitions = Vec::new();
        Self::try_reserve(
            &mut partitions,
            num_partitions,
            "resident partition catalog",
        )?;

        let mut spill_base_sizes = Vec::new();
        Self::try_reserve(
            &mut spill_base_sizes,
            num_partitions,
            "spilled-base size catalog",
        )?;

        let mut partition_sizes = Vec::new();
        Self::try_reserve(
            &mut partition_sizes,
            num_partitions,
            "partition size catalog",
        )?;

        let mut access_times = Vec::new();
        Self::try_reserve(
            &mut access_times,
            num_partitions,
            "partition access-time catalog",
        )?;

        // Every push is allocation-free because all five exact reservations
        // completed above. `new_partition_map` itself does not allocate buckets.
        for _ in 0..num_partitions {
            spill_files.push(None);
            partitions.push(Some(new_partition_map()));
            spill_base_sizes.push(0);
            partition_sizes.push(0);
            access_times.push(0);
        }

        Ok(Self {
            partitions,
            spill_files,
            spill_base_sizes,
            partition_sizes,
            access_times,
        })
    }

    fn try_reserve<T>(
        catalog: &mut Vec<T>,
        capacity: usize,
        container: &'static str,
    ) -> Result<(), PartitionAdmissionError> {
        catalog
            .try_reserve_exact(capacity)
            .map_err(|error| PartitionAdmissionError::Allocation { container, error })
    }
}

/// Partitioned accumulator state for spillable aggregation.
///
/// Manages aggregate state across multiple partitions, with the ability
/// to spill cold partitions to disk under memory pressure.
/// `drain_all` and `iter_all` still materialize all output and reload whole
/// native partitions; Task 7 replaces these compatibility methods for RDF
/// resumable aggregation rather than claiming end-to-end boundedness here.
pub struct PartitionedState<V> {
    /// Spill manager for file creation.
    manager: Arc<SpillManager>,
    /// Number of partitions.
    num_partitions: usize,
    /// In-memory partitions (None = spilled to disk).
    partitions: Vec<Option<PartitionMap<V>>>,
    /// Spill files for spilled partitions.
    spill_files: Vec<Option<SpillFile>>,
    /// Physical entry count in each immutable spilled base.
    spill_base_sizes: Vec<usize>,
    /// Number of groups per partition (for spilled partitions too).
    partition_sizes: Vec<usize>,
    /// Access timestamps for LRU eviction.
    access_times: Vec<u64>,
    /// Global timestamp counter.
    timestamp: u64,
    /// Serializer for V values.
    value_serializer:
        Box<dyn Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync>,
    /// Deserializer for V values.
    value_deserializer: Box<dyn Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync>,
    /// Exact retained heap capacity below each aggregate value. Accounted
    /// callers supply the type-specific traversal; compatibility callers use
    /// an inert callback and make no bounded-memory claim.
    value_resident_capacity: Box<dyn Fn(&V) -> Result<usize, MemoryGrantError> + Send + Sync>,
    /// Per-entry frame limits (Task 3 supplies grant-derived limits).
    frame_limits: SpillFrameLimits,
    /// Immutable check-only capability for resource-qualified operations.
    cancellation: Option<QueryCancellationToken>,
    /// A failed/in-progress destructive drain poisons ordinary use until an
    /// explicit successful cleanup reset; a completed drain remains reusable.
    drain_state: DrainState,
    /// Real resident-memory authority for resource-qualified partition state.
    ///
    /// This is declared after every owned allocation so field drop frees the
    /// physical containers before releasing their accounting capability.
    grant: Option<MemoryGrant>,
    callback_bytes: usize,
    recovery_allowance: Option<MemoryGrant>,
    // Required codec headroom stays separate from reusable surplus; deriving
    // it from the idle grant would grow the next row's preflight recursively.
    recovery_working_bytes: usize,
    failure_publisher: Option<AccountedErrorPublisher<PartitionFailure>>,
    failure_cleanup: Option<AccountedError>,
}

impl<V: Clone + Send + Sync + 'static> PartitionedState<V> {
    /// Updates an aggregate in place after admitting its additional peak.
    /// Callback failure is terminal; no partially updated group is reusable.
    pub(crate) fn try_update_accounted<K, D, B, U>(
        &mut self,
        key_bytes: usize,
        make_key: K,
        declare: D,
        build: B,
        update: U,
    ) -> Result<(), PartitionOperationError>
    where
        V: SharedImmutablePartitionValue,
        K: FnOnce() -> Result<Vec<Value>, OperatorError>,
        D: FnOnce(Option<&V>) -> Result<(PartitionUpdateAdmission, usize), OperatorError>,
        B: FnOnce() -> Result<V, OperatorError>,
        U: FnOnce(&mut V) -> Result<(), OperatorError>,
    {
        if self.failure_publisher.is_none() {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "accounted aggregate has no live failure publisher",
            });
        }
        let recovery = self.recovery_allowance.take();
        let recovery_required = self.recovery_working_bytes;
        let mut retry = OneSpillRetryBudget {
            spent: false,
            recovery,
            recovery_required,
        };
        let cancellation = self.cancellation.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<(), PartitionMutableFailure> {
                check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;
                let index = self.try_update_accounted_inner(
                    key_bytes, make_key, declare, build, update, &mut retry,
                )?;
                self.check_physical_cleanup()?;
                // The inner scope has retired the key, reader and update workspaces.
                // Restoring future scheduling capacity may consume only this row's
                // still-unspent victim budget; it never replays the completed update.
                self.restore_recovery_allowance(Some(index), cancellation.as_ref(), &mut retry)?;
                self.check_physical_cleanup()?;
                Ok(())
            },
        ));
        self.recovery_working_bytes = retry.recovery_required;
        self.recovery_allowance = retry.recovery.take();
        match outcome {
            Ok(Ok(())) => Ok(()),
            Ok(Err(PartitionMutableFailure::Operation(error))) => {
                Err(self.publish_failure(Some(error), None, None))
            }
            Ok(Err(PartitionMutableFailure::Operator(error))) => {
                Err(self.publish_failure(None, Some(error), None))
            }
            Err(payload) => Err(self.publish_failure(None, None, Some(payload))),
        }
    }

    fn try_update_accounted_inner<K, D, B, U>(
        &mut self,
        key_bytes: usize,
        make_key: K,
        declare: D,
        build: B,
        update: U,
        retry: &mut OneSpillRetryBudget,
    ) -> Result<usize, PartitionMutableFailure>
    where
        K: FnOnce() -> Result<Vec<Value>, OperatorError>,
        D: FnOnce(Option<&V>) -> Result<(PartitionUpdateAdmission, usize), OperatorError>,
        B: FnOnce() -> Result<V, OperatorError>,
        U: FnOnce(&mut V) -> Result<(), OperatorError>,
    {
        if self.grant.is_none() {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "accounted update has no root authority",
            }
            .into());
        }
        let cancellation = self.cancellation.clone();
        self.ensure_not_draining()
            .map_err(PartitionOperationError::from)?;
        if retry.recovery.is_none() {
            retry.recovery = Some(self.split_workspace_grant(0)?);
        }
        let key_workspace =
            self.lend_row_workspace(key_bytes, None, cancellation.as_ref(), retry)?;
        // The target is intentionally unknown until the admitted key exists.
        let key = make_key()?;
        check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;
        let index = self.partition_for(&key);
        let columns = key.len();
        let mut serialized =
            self.stage_reused_aggregate_key(&key, index, cancellation.as_ref(), retry)?;
        drop(key);
        self.return_row_grant(key_workspace.into_grant()?, retry)?;
        let resident = self.partitions[index]
            .as_ref()
            .ok_or_else(|| native_map_invariant("accounted update has no resident delta"))?
            .get(serialized.serialized());
        let needs_map_entry = resident.is_none();
        let mut base = None;
        let (old_bytes, present, (admission, working_bytes)) = if let Some(entry) = resident {
            let old_bytes =
                (self.value_resident_capacity)(&entry.value).map_err(partition_memory_error)?;
            (old_bytes, true, declare(Some(&entry.value))?)
        } else {
            if self.spill_files[index].is_some() {
                Self::release_idle_row_authority(retry)?;
                base = self.lookup_spilled_base_entry_accounted(
                    index,
                    serialized.serialized(),
                    columns,
                    cancellation.as_ref(),
                    retry,
                )?;
            }
            // Inspect the staged base under its existing child grant. Publication
            // and map growth wait until the complete row's scheduling preflight.
            let old = base
                .as_ref()
                .map(|base| {
                    base.entry.as_ref().ok_or_else(|| {
                        native_map_invariant("accounted update lost its staged base value")
                    })
                })
                .transpose()?;
            let old_bytes = old
                .map_or(Ok(0), |entry| (self.value_resident_capacity)(&entry.value))
                .map_err(partition_memory_error)?;
            (
                old_bytes,
                old.is_some(),
                declare(old.map(|entry| &entry.value))?,
            )
        };
        check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;
        if old_bytes
            .checked_add(admission.construction_peak)
            .is_none_or(|peak| peak < admission.retained_upper_bound)
        {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "aggregate update peak does not cover its retained bound",
            }
            .into());
        }
        // A caller-derived scheduling allowance leaves room for the next
        // bounded codec operation. Provider workspaces still admit independently.
        let working_bytes = working_bytes
            .checked_add(
                qualified_writer_buffer_requested_bytes()
                    .checked_mul(2)
                    // Cleartext consolidation also opens a borrowed reader;
                    // its control stored-Vec envelope coexists with the writer.
                    // Other provider/hook requirements still admit independently.
                    .and_then(|bytes| {
                        MAX_FIXED_CONTROL_PAYLOAD_BYTES
                            .checked_mul(2)
                            .and_then(|control| bytes.checked_add(control))
                    })
                    .ok_or_else(|| native_map_invariant("recovery allowance overflow"))?,
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    // Missing metadata does not promise zero physical usage:
                    // the unchanged file/provider admission rejects it later.
                    self.manager
                        .qualified_sort_provider_workspace_bound()
                        .unwrap_or(0),
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    self.manager
                        .qualified_reader_hook_workspace_bound()
                        .unwrap_or(0),
                )
            })
            // One qualified partition reader can coexist with the writer and
            // row workspace; its persistent buffer admits independently.
            .and_then(|bytes| {
                bytes.checked_add(qualified_partition_reader_buffer_requested_bytes())
            })
            .ok_or_else(|| native_map_invariant("recovery allowance overflow"))?;
        retry.recovery_required = retry.recovery_required.max(working_bytes);
        let map_peak = if needs_map_entry {
            let map = self.partitions[index]
                .as_ref()
                .ok_or_else(|| native_map_invariant("accounted preflight has no resident map"))?;
            let required = map
                .len()
                .checked_add(1)
                .ok_or_else(|| native_map_invariant("accounted preflight entry count overflow"))?;
            if required > map.capacity() {
                Self::planned_partition_map_allocation_bytes(required)
                    .map_err(partition_memory_error)?
            } else {
                0
            }
        } else {
            0
        };
        let preflight_bytes = retry
            .recovery_required
            .checked_add(admission.construction_peak)
            .and_then(|bytes| bytes.checked_add(map_peak))
            .ok_or_else(|| native_map_invariant("aggregate update scheduling peak overflow"))?;
        // Keep the admitted peak: map and update children borrow from it, so
        // publication never races to reacquire authority already proved here.
        self.ensure_row_capacity(preflight_bytes, Some(index), cancellation.as_ref(), retry)?;
        if let Some(base) = base.take() {
            self.reserve_new_entry_with_one_spill_inner(
                index,
                MapAdmissionCapacity::AggregateRoot,
                cancellation.as_ref(),
                retry,
            )?;
            AccountedBaseDeltaPublication::new(
                base,
                self.grant.as_mut().ok_or_else(|| {
                    native_map_invariant("accounted update has no root authority")
                })?,
            )
            .publish_into(
                self.partitions[index].as_mut().ok_or_else(|| {
                    native_map_invariant("accounted update has no resident delta")
                })?,
            )
            .map_err(PartitionOperationError::from)?;
        }
        let cleanup = self.failure_cleanup.clone();
        let workspace = self.lend_row_workspace(
            admission.construction_peak,
            Some(index),
            cancellation.as_ref(),
            retry,
        )?;
        if !present {
            self.reserve_new_entry_with_one_spill_inner(
                index,
                MapAdmissionCapacity::AggregateRoot,
                cancellation.as_ref(),
                retry,
            )?;
        }
        check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;
        let (mut created, observed, updated_entry) = if present {
            let entry = self.partitions[index]
                .as_mut()
                .ok_or_else(|| native_map_invariant("accounted update has no resident delta"))?
                .get_mut(serialized.serialized())
                .ok_or_else(|| native_map_invariant("accounted update lost its observed entry"))?;
            update(&mut entry.value)?;
            let observed =
                (self.value_resident_capacity)(&entry.value).map_err(partition_memory_error)?;
            (None, observed, Some(entry))
        } else {
            let mut value = build()?;
            update(&mut value)?;
            let observed =
                (self.value_resident_capacity)(&value).map_err(partition_memory_error)?;
            (Some(value), observed, None)
        };
        if observed > admission.retained_upper_bound {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "aggregate update exceeded its retained bound",
            }
            .into());
        }
        let root = self
            .grant
            .as_ref()
            .ok_or_else(|| native_map_invariant("accounted update has no root authority"))?;
        let final_root = root
            .size()
            .checked_sub(old_bytes)
            .and_then(|n| n.checked_add(observed))
            .ok_or_else(|| {
                partition_memory_error(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: root.size(),
                    additional_bytes: observed,
                })
            })?;
        let update_grant = workspace.into_grant()?;
        let unchanged_retained = present && observed == old_bytes;
        if unchanged_retained {
            // The mutation retired its scratch and retained exactly the bytes
            // already covered by root. Return the intact child instead of
            // merging it into root only to split those same bytes back out.
            Self::return_row_grant_with_cleanup(update_grant, retry, cleanup.as_ref())?;
        } else {
            let root = self
                .grant
                .as_mut()
                .ok_or_else(|| native_map_invariant("accounted update has no root authority"))?;
            if let Err(grant) = root.try_merge(update_grant) {
                drop(PartitionWorkspace::new(grant, cleanup.as_ref()));
                return Err(PartitionOperationError::NativeMapInvariant {
                    message: "aggregate update grant lost root identity",
                }
                .into());
            }
        }
        if present {
            updated_entry
                .ok_or_else(|| native_map_invariant("accounted update lost its updated entry"))?
                .resident_bound = observed;
        } else {
            let next = self.partition_sizes[index]
                .checked_add(1)
                .ok_or_else(|| native_map_invariant("aggregate group count overflow"))?;
            let root = self
                .grant
                .as_mut()
                .ok_or_else(|| native_map_invariant("accounted update has no root authority"))?;
            AccountedKeyPublication::new(serialized, root)
                .publish_into(
                    self.partitions[index].as_mut().ok_or_else(|| {
                        native_map_invariant("accounted update has no resident delta")
                    })?,
                    PartitionEntry {
                        num_key_columns: columns,
                        resident_bound: observed,
                        value: created.take().ok_or_else(|| {
                            native_map_invariant("accounted update has no created value")
                        })?,
                    },
                    || {},
                )
                .map_err(PartitionOperationError::from)?;
            self.partition_sizes[index] = next;
            // Key publication moved its separately admitted bytes into root.
            let root = self
                .grant
                .as_mut()
                .ok_or_else(|| native_map_invariant("accounted update has no root authority"))?;
            let release = admission
                .construction_peak
                .checked_sub(observed)
                .ok_or_else(|| {
                    native_map_invariant(
                        "aggregate update retained more than its construction authority",
                    )
                })?;
            let surplus = root
                .split(release)
                .ok_or_else(|| native_map_invariant("aggregate update lost its retired surplus"))?;
            self.return_row_grant(surplus, retry)?;
            self.touch(index);
            return Ok(index);
        }
        if !unchanged_retained {
            let root = self
                .grant
                .as_mut()
                .ok_or_else(|| native_map_invariant("accounted update has no root authority"))?;
            let release = root.size().checked_sub(final_root).ok_or_else(|| {
                native_map_invariant("aggregate update lost its retained authority")
            })?;
            let surplus = root
                .split(release)
                .ok_or_else(|| native_map_invariant("aggregate update lost its retired surplus"))?;
            self.return_row_grant(surplus, retry)?;
        }
        drop(serialized.key.take());
        let grant = serialized
            .grant
            .take()
            .ok_or_else(|| native_map_invariant("aggregate lookup key lost its authority"))?;
        self.return_row_grant(grant, retry)?;
        self.touch(index);
        Ok(index)
    }
    pub(crate) fn fail_operator(&mut self, error: OperatorError) -> PartitionOperationError {
        self.publish_failure(None, Some(error), None)
    }

    fn check_physical_cleanup(&self) -> Result<(), PartitionOperationError> {
        if self
            .failure_cleanup
            .as_ref()
            .and_then(|cleanup| {
                cleanup.inspect::<PartitionFailureCleanup, _>(|witness| {
                    witness.failed.load(std::sync::atomic::Ordering::Acquire)
                })
            })
            .unwrap_or(false)
        {
            Err(PartitionOperationError::NativeMapInvariant {
                message: "partition physical cleanup failed",
            })
        } else {
            Ok(())
        }
    }

    fn publish_failure(
        &mut self,
        primary: Option<PartitionOperationError>,
        operator: Option<OperatorError>,
        panic: Option<Box<dyn std::any::Any + Send>>,
    ) -> PartitionOperationError {
        let consuming_drain = self.drain_state == DrainState::Draining;
        drop(self.recovery_allowance.take());
        if operator.is_none()
            && panic.is_none()
            && matches!(primary, Some(PartitionOperationError::Accounted { .. }))
        {
            return primary.unwrap_or(PartitionOperationError::NativeMapInvariant {
                message: "missing accounted partition failure",
            });
        }
        // Inspect only the closed, core-owned context chain. The complete
        // original operator tree still moves into the failure owner below.
        let mut operator_primary = operator.as_ref();
        while let Some(OperatorError::Context { source, .. }) = operator_primary {
            operator_primary = Some(source.as_ref());
        }
        let classification = if let Some(error) = primary
            .as_ref()
            .and_then(PartitionOperationError::resident_memory_error)
        {
            AccountedFailureClassification::ResidentMemory(error.clone())
        } else if let Some(reason) = primary
            .as_ref()
            .and_then(PartitionOperationError::cancellation_error)
        {
            AccountedFailureClassification::QueryCancelled(*reason)
        } else {
            match primary.as_ref() {
                Some(
                    PartitionOperationError::Io(error)
                    | PartitionOperationError::IoWithCleanup { error, .. },
                ) if matches!(
                    error.kind(),
                    std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded
                ) =>
                {
                    AccountedFailureClassification::StorageFull
                }
                Some(
                    PartitionOperationError::Admission(
                        PartitionAdmissionError::Allocation { .. }
                        | PartitionAdmissionError::PublisherAllocation
                        | PartitionAdmissionError::PublisherAllocationWithRollback(_),
                    )
                    | PartitionOperationError::AdmissionWithCleanup {
                        error:
                            PartitionAdmissionError::Allocation { .. }
                            | PartitionAdmissionError::PublisherAllocation
                            | PartitionAdmissionError::PublisherAllocationWithRollback(_),
                        ..
                    },
                ) => AccountedFailureClassification::ResidentAllocation,
                Some(
                    PartitionOperationError::Allocation { .. }
                    | PartitionOperationError::NativeMapAllocation { .. }
                    | PartitionOperationError::NativeMapAllocationWithRollback { .. },
                ) => AccountedFailureClassification::ResidentAllocation,
                _ => match operator_primary {
                    Some(OperatorError::QueryCancelled(reason)) => {
                        AccountedFailureClassification::QueryCancelled(*reason)
                    }
                    Some(OperatorError::ResidentMemory(error)) => {
                        AccountedFailureClassification::ResidentMemory(error.clone())
                    }
                    Some(OperatorError::ClassifiedAccountedFailure { classification, .. }) => {
                        classification.clone()
                    }
                    Some(OperatorError::UnsupportedAccountedTransport { .. }) => {
                        AccountedFailureClassification::UnsupportedAccountedTransport
                    }
                    Some(OperatorError::StorageFull(_)) => {
                        AccountedFailureClassification::StorageFull
                    }
                    Some(OperatorError::TypeMismatch { .. }) => {
                        AccountedFailureClassification::TypeMismatch
                    }
                    Some(OperatorError::ColumnNotFound(_)) => {
                        AccountedFailureClassification::ColumnNotFound
                    }
                    Some(OperatorError::ConstraintViolation(_)) => {
                        AccountedFailureClassification::ConstraintViolation
                    }
                    Some(OperatorError::WriteConflict(_)) => {
                        AccountedFailureClassification::WriteConflict
                    }
                    Some(
                        OperatorError::ResidentAllocation(_)
                        | OperatorError::ResidentContainerAllocation { .. }
                        | OperatorError::ResidentNativeMapAllocation { .. }
                        | OperatorError::ResidentNativeMapAllocationWithRollback { .. },
                    ) => AccountedFailureClassification::ResidentAllocation,
                    Some(OperatorError::ResidentExactVectorAllocation(error)) => {
                        AccountedFailureClassification::ResidentExactVectorAllocation(error.clone())
                    }
                    Some(
                        OperatorError::ResidentContainerInvariant { .. }
                        | OperatorError::ResidentContainerInvariantWithRollback { .. },
                    ) => AccountedFailureClassification::ResidentInvariant,
                    _ => AccountedFailureClassification::Execution,
                },
            }
        };
        let Some(cleanup) = self.failure_cleanup.clone() else {
            // A broken owner invariant cannot safely retire opaque payloads or
            // release the root that covers them. Preserve both fail-closed.
            self.drain_state = DrainState::Poisoned;
            std::mem::forget((self.grant.take(), primary, operator, panic));
            return PartitionOperationError::NativeMapInvariant {
                message: "partition failure has no cleanup owner",
            };
        };
        cleanup
            .inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_operation_failed);
        if let Some(grant) = self.grant.take() {
            let mut grant = Some(grant);
            cleanup.inspect::<PartitionFailureCleanup, _>(|witness| {
                if let Some(grant) = grant.take() {
                    witness.retain(grant);
                }
            });
            if let Some(grant) = grant {
                std::mem::forget(grant);
            }
        }
        let (mut cleanup_error, mut cleanup_panic) = cleanup
            .inspect::<PartitionFailureCleanup, _>(|witness| {
                (
                    witness.secondary_error.borrow_mut().take(),
                    witness.secondary_panic.borrow_mut().take(),
                )
            })
            .unwrap_or((None, None));
        for (index, slot) in self.spill_files.iter_mut().enumerate() {
            let Some(file) = slot.as_mut() else {
                continue;
            };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| file.close_and_delete()))
            {
                Ok(Ok(())) => {
                    *slot = None;
                    if consuming_drain {
                        // Retired files no longer own base groups. A resident
                        // delta remains physically owned and keeps its count.
                        self.spill_base_sizes[index] = 0;
                        self.partition_sizes[index] =
                            self.partitions[index].as_ref().map_or(0, PartitionMap::len);
                    }
                }
                Ok(Err(error)) if cleanup_error.is_none() => cleanup_error = Some(error),
                Err(payload) if cleanup_panic.is_none() => cleanup_panic = Some(payload),
                Ok(Err(error)) => {
                    if !super::run_cleanup_backstop(|| {
                        drop(error);
                        Ok::<_, ()>(())
                    }) {
                        cleanup.inspect::<PartitionFailureCleanup, _>(
                            PartitionFailureCleanup::mark_failed,
                        );
                    }
                }
                Err(payload) => {
                    if !super::run_cleanup_backstop(|| {
                        drop(payload);
                        Ok::<_, ()>(())
                    }) {
                        cleanup.inspect::<PartitionFailureCleanup, _>(
                            PartitionFailureCleanup::mark_failed,
                        );
                    }
                }
            }
        }
        self.drain_state = DrainState::Poisoned;
        let Some(publisher) = self.failure_publisher.take() else {
            let failure = PartitionFailure {
                primary,
                operator,
                panic,
                cleanup_error,
                cleanup_panic,
                cleanup,
            };
            drop(failure);
            return PartitionOperationError::NativeMapInvariant {
                message: "accounted partition is terminally poisoned",
            };
        };
        PartitionOperationError::Accounted {
            classification,
            authority: publisher.publish(PartitionFailure {
                primary,
                operator,
                panic,
                cleanup_error,
                cleanup_panic,
                cleanup,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn inspect_failure<R>(
        error: &OperatorError,
        inspect: impl FnOnce(
            Option<&PartitionOperationError>,
            Option<&OperatorError>,
            Option<&(dyn std::any::Any + Send)>,
            Option<&std::io::Error>,
            Option<&(dyn std::any::Any + Send)>,
        ) -> R,
    ) -> Option<R> {
        let OperatorError::ClassifiedAccountedFailure { authority, .. } = error else {
            return None;
        };
        authority.inspect::<PartitionFailure, _>(|failure| {
            inspect(
                failure.primary.as_ref(),
                failure.operator.as_ref(),
                failure.panic.as_deref(),
                failure.cleanup_error.as_ref(),
                failure.cleanup_panic.as_deref(),
            )
        })
    }
    /// Live aggregate constructor with typed, pre-callback admission failures.
    /// Callback captures must own no additional heap; the live aggregate passes
    /// function items. Codec retained/decode bounds also cover their diagnostics.
    /// This private GroupState seam admits serializer scratch at twice the
    /// stored retained bound; codec callbacks must satisfy that audited bound.
    pub(crate) fn new_accounted_admitted_with_cancellation<S, D, C>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
        value_resident_capacity: C,
        mut grant: MemoryGrant,
        cancellation: QueryCancellationToken,
    ) -> Result<Self, PartitionAdmissionError>
    where
        S: Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync + 'static,
        C: Fn(&V) -> Result<usize, MemoryGrantError> + Send + Sync + 'static,
    {
        if grant.size() != 0 {
            return Err(PartitionAdmissionError::NonZeroPublisherGrant {
                bytes: grant.size(),
            });
        }
        // Trusted diagnostic text is at most 256 bytes (static text plus up to
        // three 20-digit usize values). 1024 covers two such retained messages,
        // their io::Error custom boxes, and one formatting growth overlap.
        // Fixed MemoryGrantError boxes are smaller than that text envelope.
        // Opaque provider, codec and hook payloads have separate declared bounds.
        const INTERNAL_DIAGNOSTIC_BYTES: usize = 1024;
        let hook_bytes = manager
            .qualified_sort_hook_workspace_bound()
            .ok_or(PartitionAdmissionError::UnboundedHooks)?;
        let callback_bytes = std::mem::size_of::<S>()
            .checked_add(std::mem::size_of::<D>())
            .and_then(|n| n.checked_add(std::mem::size_of::<C>()))
            .and_then(|n| n.checked_add(INTERNAL_DIAGNOSTIC_BYTES))
            .and_then(|n| n.checked_add(hook_bytes))
            .ok_or(PartitionAdmissionError::Memory(
                MemoryGrantError::ArithmeticOverflow {
                    current_bytes: std::mem::size_of::<S>(),
                    additional_bytes: std::mem::size_of::<D>(),
                },
            ))?;
        let initial = Self::initial_resident_capacity_bytes(num_partitions)
            .map_err(PartitionAdmissionError::Memory)?;
        let admitted =
            initial
                .checked_add(callback_bytes)
                .ok_or(PartitionAdmissionError::Memory(
                    MemoryGrantError::ArithmeticOverflow {
                        current_bytes: initial,
                        additional_bytes: callback_bytes,
                    },
                ))?;
        grant
            .try_resize(admitted)
            .map_err(PartitionAdmissionError::Memory)?;
        let cleanup = AccountedErrorPublisher::try_new(
            grant
                .split(0)
                .ok_or(PartitionAdmissionError::PublisherInvariant)?,
        )
        .map_err(partition_publisher_admission)?;
        let publisher = AccountedErrorPublisher::try_new(
            grant
                .split(0)
                .ok_or(PartitionAdmissionError::PublisherInvariant)?,
        )
        .map_err(partition_publisher_admission)?;
        let catalogs = PartitionCatalogs::try_new(num_partitions)?;
        let mut state = Self::new_with_bounded_codec_inner_from_catalogs(
            manager,
            value_serializer,
            value_deserializer,
            Box::new(value_resident_capacity),
            Some(cancellation),
            Some(grant),
            catalogs,
        );
        state.callback_bytes = callback_bytes;
        state.failure_cleanup = Some(cleanup.publish(PartitionFailureCleanup::new()));
        state.failure_publisher = Some(publisher);
        // Catalog allocation may reserve more than requested. No codec or
        // resident-value callback runs for these empty catalogs.
        state
            .reconcile_grant()
            .map_err(PartitionAdmissionError::Memory)?;
        Ok(state)
    }
    /// Creates a new partitioned state with custom serialization.
    pub fn new<S, D>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
    ) -> Self
    where
        S: Fn(&V, &mut dyn Write) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read) -> std::io::Result<V> + Send + Sync + 'static,
    {
        Self::new_with_bounded_deserializer(
            manager,
            num_partitions,
            move |value, writer| value_serializer(value, writer),
            move |reader, _limits| value_deserializer(reader),
        )
    }

    /// Creates partitioned state whose custom decoder receives the exact
    /// frame-derived codec limits for the custom-state slice.
    pub fn new_with_bounded_deserializer<S, D>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
    ) -> Self
    where
        S: Fn(&V, &mut dyn Write) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync + 'static,
    {
        Self::new_with_bounded_codec(
            manager,
            num_partitions,
            move |value, writer, _limits| value_serializer(value, writer),
            value_deserializer,
        )
    }

    /// Creates partitioned state whose custom codec receives one shared
    /// decoded-resident policy for both serialization and deserialization.
    pub fn new_with_bounded_codec<S, D>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
    ) -> Self
    where
        S: Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync + 'static,
    {
        Self::new_with_bounded_codec_inner(
            manager,
            num_partitions,
            value_serializer,
            value_deserializer,
            Box::new(|_| Ok(0)),
            None,
            None,
        )
    }

    pub(crate) fn new_with_bounded_codec_and_cancellation<S, D>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
        cancellation: QueryCancellationToken,
    ) -> Self
    where
        S: Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync + 'static,
    {
        Self::new_with_bounded_codec_inner(
            manager,
            num_partitions,
            value_serializer,
            value_deserializer,
            Box::new(|_| Ok(0)),
            Some(cancellation),
            None,
        )
    }

    /// Creates resource-qualified state backed by a dedicated zero-sized
    /// resident-memory root grant.
    ///
    /// # Errors
    ///
    /// Returns structured grant exhaustion before allocating a retained
    /// partition container, or a structured allocator/address-space refusal
    /// while reserving the catalogs. Failed construction drops every staged
    /// catalog and releases the complete pre-admitted root grant.
    ///
    /// The custom decoder and resident-capacity callback are audited trust
    /// boundaries. On bounded spilled-base lookup, their successful value and
    /// any returned-error or panic-payload construction must fit the admitted
    /// decoded-key plus stored-value envelope. The protocol retains that grant
    /// with an escaping payload, but cannot measure allocations hidden inside
    /// an arbitrary callback error or panic.
    ///
    /// # Panics
    ///
    /// Panics when `grant` is not a dedicated zero-sized root grant. Accepting
    /// a shared/non-empty grant would make later split ownership ambiguous.
    pub fn new_accounted_with_cancellation<S, D, C>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
        value_resident_capacity: C,
        mut grant: MemoryGrant,
        cancellation: QueryCancellationToken,
    ) -> Result<Self, PartitionOperationError>
    where
        S: Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync + 'static,
        C: Fn(&V) -> Result<usize, MemoryGrantError> + Send + Sync + 'static,
    {
        assert_eq!(
            grant.size(),
            0,
            "accounted partition state requires a dedicated zero-sized root grant"
        );
        let initial_capacity = Self::initial_resident_capacity_bytes(num_partitions)
            .map_err(partition_memory_error)?;
        grant
            .try_resize(initial_capacity)
            .map_err(partition_memory_error)?;
        let catalogs = PartitionCatalogs::try_new(num_partitions)?;
        let mut state = Self::new_with_bounded_codec_inner_from_catalogs(
            manager,
            value_serializer,
            value_deserializer,
            Box::new(value_resident_capacity),
            Some(cancellation),
            Some(grant),
            catalogs,
        );
        state.reconcile_grant().map_err(partition_memory_error)?;
        Ok(state)
    }

    fn new_with_bounded_codec_inner<S, D>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
        value_resident_capacity: Box<dyn Fn(&V) -> Result<usize, MemoryGrantError> + Send + Sync>,
        cancellation: Option<QueryCancellationToken>,
        grant: Option<MemoryGrant>,
    ) -> Self
    where
        S: Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync + 'static,
    {
        let catalogs = PartitionCatalogs::new(num_partitions);
        Self::new_with_bounded_codec_inner_from_catalogs(
            manager,
            value_serializer,
            value_deserializer,
            value_resident_capacity,
            cancellation,
            grant,
            catalogs,
        )
    }

    fn new_with_bounded_codec_inner_from_catalogs<S, D>(
        manager: Arc<SpillManager>,
        value_serializer: S,
        value_deserializer: D,
        value_resident_capacity: Box<dyn Fn(&V) -> Result<usize, MemoryGrantError> + Send + Sync>,
        cancellation: Option<QueryCancellationToken>,
        grant: Option<MemoryGrant>,
        catalogs: PartitionCatalogs<V>,
    ) -> Self
    where
        S: Fn(&V, &mut dyn Write, CodecLimits) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read, CodecLimits) -> std::io::Result<V> + Send + Sync + 'static,
    {
        let PartitionCatalogs {
            partitions,
            spill_files,
            spill_base_sizes,
            partition_sizes,
            access_times,
        } = catalogs;
        let num_partitions = partitions.len();

        let frame_limits = manager.frame_limits();
        Self {
            manager,
            num_partitions,
            partitions,
            spill_files,
            spill_base_sizes,
            partition_sizes,
            access_times,
            timestamp: 0,
            value_serializer: Box::new(value_serializer),
            value_deserializer: Box::new(value_deserializer),
            value_resident_capacity,
            frame_limits,
            cancellation,
            drain_state: DrainState::Idle,
            grant,
            failure_publisher: None,
            failure_cleanup: None,
            callback_bytes: 0,
            recovery_allowance: None,
            recovery_working_bytes: 0,
        }
    }

    /// Returns the bytes retained by this state's real resident grant.
    #[must_use]
    pub fn granted_bytes(&self) -> usize {
        self.grant
            .as_ref()
            .map_or(0, MemoryGrant::size)
            .saturating_add(
                self.recovery_allowance
                    .as_ref()
                    .map_or(0, MemoryGrant::size),
            )
            .saturating_add(
                self.failure_publisher
                    .as_ref()
                    .map_or(0, AccountedErrorPublisher::granted_bytes),
            )
            .saturating_add(
                self.failure_cleanup
                    .as_ref()
                    .map_or(0, AccountedError::granted_bytes),
            )
    }

    fn checked_capacity_bytes<T>(total: usize, capacity: usize) -> Result<usize, MemoryGrantError> {
        let bytes = capacity.checked_mul(std::mem::size_of::<T>()).ok_or(
            MemoryGrantError::ArithmeticOverflow {
                current_bytes: total,
                additional_bytes: capacity,
            },
        )?;
        total
            .checked_add(bytes)
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: total,
                additional_bytes: bytes,
            })
    }

    /// Conservative allocation request for `hashbrown` 0.17.1's empty-table
    /// `try_reserve(required_entries)` path using `allocator-api2` 0.2.21's
    /// `Global`, which reports exactly the requested allocation length.
    ///
    /// This is deliberately version-coupled to both exact workspace pins. The
    /// allocator's exact-length receipt prevents hashbrown's optional
    /// oversized-block bucket expansion. The table allocates power-of-two
    /// entry buckets followed by one control byte per bucket and one trailing
    /// SIMD control group. Its largest selected group is 16 bytes; using 16 on
    /// scalar/NEON targets is conservative. Tests compare this pre-allocation
    /// proof with public `allocation_size()` across every growth threshold we
    /// rely on.
    fn planned_partition_map_allocation_bytes(
        required_entries: usize,
    ) -> Result<usize, MemoryGrantError> {
        if required_entries == 0 {
            return Ok(0);
        }

        const MAX_CONTROL_GROUP_WIDTH: usize = 16;
        let entry_size = std::mem::size_of::<(SerializedKey, PartitionEntry<V>)>();
        let buckets = if required_entries < 15 {
            let minimum_capacity = match entry_size {
                0..=1 => 14,
                2..=3 => 7,
                _ => 3,
            };
            let capacity = required_entries.max(minimum_capacity);
            if capacity < 4 {
                4
            } else if capacity < 8 {
                8
            } else {
                16
            }
        } else {
            required_entries
                .checked_mul(8)
                .map(|adjusted| adjusted / 7)
                .and_then(usize::checked_next_power_of_two)
                .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: required_entries,
                    additional_bytes: required_entries,
                })?
        };
        let control_alignment =
            std::mem::align_of::<(SerializedKey, PartitionEntry<V>)>().max(MAX_CONTROL_GROUP_WIDTH);
        let entry_bytes =
            entry_size
                .checked_mul(buckets)
                .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: entry_size,
                    additional_bytes: buckets,
                })?;
        let control_offset = entry_bytes
            .checked_add(control_alignment - 1)
            .map(|bytes| bytes & !(control_alignment - 1))
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: entry_bytes,
                additional_bytes: control_alignment - 1,
            })?;
        let allocation_bytes = control_offset
            .checked_add(buckets)
            .and_then(|bytes| bytes.checked_add(MAX_CONTROL_GROUP_WIDTH))
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: control_offset,
                additional_bytes: buckets.saturating_add(MAX_CONTROL_GROUP_WIDTH),
            })?;
        if allocation_bytes > isize::MAX as usize - (control_alignment - 1) {
            return Err(MemoryGrantError::ArithmeticOverflow {
                current_bytes: allocation_bytes,
                additional_bytes: control_alignment - 1,
            });
        }
        Ok(allocation_bytes)
    }

    fn partition_map_allocation_bytes(partition: &PartitionMap<V>) -> usize {
        partition.allocation_size()
    }

    fn initial_resident_capacity_bytes(num_partitions: usize) -> Result<usize, MemoryGrantError> {
        let mut total = 0;
        total = Self::checked_capacity_bytes::<Option<PartitionMap<V>>>(total, num_partitions)?;
        total = Self::checked_capacity_bytes::<Option<SpillFile>>(total, num_partitions)?;
        total = Self::checked_capacity_bytes::<usize>(total, num_partitions)?;
        total = Self::checked_capacity_bytes::<usize>(total, num_partitions)?;
        Self::checked_capacity_bytes::<u64>(total, num_partitions)
    }

    /// Recomputes the exact declared capacity of every retained resident
    /// partition container and nested value allocation.
    ///
    /// # Errors
    ///
    /// Returns structured arithmetic or caller-supplied capacity failures.
    pub fn observed_resident_capacity_bytes(&self) -> Result<usize, MemoryGrantError> {
        let mut total = self.callback_bytes;
        total = Self::checked_capacity_bytes::<Option<PartitionMap<V>>>(
            total,
            self.partitions.capacity(),
        )?;
        total =
            Self::checked_capacity_bytes::<Option<SpillFile>>(total, self.spill_files.capacity())?;
        total = Self::checked_capacity_bytes::<usize>(total, self.spill_base_sizes.capacity())?;
        total = Self::checked_capacity_bytes::<usize>(total, self.partition_sizes.capacity())?;
        total = Self::checked_capacity_bytes::<u64>(total, self.access_times.capacity())?;
        for partition in self.partitions.iter().flatten() {
            let map_bytes = Self::partition_map_allocation_bytes(partition);
            total = total
                .checked_add(map_bytes)
                .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: total,
                    additional_bytes: map_bytes,
                })?;
            for key in partition.keys() {
                total = total.checked_add(key.0.capacity()).ok_or(
                    MemoryGrantError::ArithmeticOverflow {
                        current_bytes: total,
                        additional_bytes: key.0.capacity(),
                    },
                )?;
            }
            for entry in partition.values() {
                let value_bytes = (self.value_resident_capacity)(&entry.value)?;
                total =
                    total
                        .checked_add(value_bytes)
                        .ok_or(MemoryGrantError::ArithmeticOverflow {
                            current_bytes: total,
                            additional_bytes: value_bytes,
                        })?;
            }
        }
        Ok(total)
    }

    fn reconcile_grant(&mut self) -> Result<(), MemoryGrantError> {
        let observed = self.observed_resident_capacity_bytes()?;
        if let Some(grant) = self.grant.as_mut() {
            grant.try_resize(observed)?;
        }
        Ok(())
    }

    fn pending_entry_capacity_plan(
        &self,
        partition_idx: usize,
        pending_capacity: usize,
    ) -> Result<PendingEntryCapacityPlan, MemoryGrantError> {
        let observed = self.observed_resident_capacity_bytes()?;
        self.pending_entry_capacity_plan_with_observed(partition_idx, pending_capacity, observed)
    }

    fn pending_entry_capacity_plan_with_observed(
        &self,
        partition_idx: usize,
        pending_capacity: usize,
        observed: usize,
    ) -> Result<PendingEntryCapacityPlan, MemoryGrantError> {
        let partition = self.partitions[partition_idx]
            .as_ref()
            .expect("capacity admission requires a resident partition");
        let old_map_bytes = Self::partition_map_allocation_bytes(partition);
        let required_entries =
            partition
                .len()
                .checked_add(1)
                .ok_or(MemoryGrantError::ArithmeticOverflow {
                    current_bytes: partition.len(),
                    additional_bytes: 1,
                })?;
        let replacement_required_entries = if required_entries > partition.capacity() {
            Some(required_entries)
        } else {
            None
        };
        let replacement_allocation_ceiling = replacement_required_entries
            .map(Self::planned_partition_map_allocation_bytes)
            .transpose()?;
        let replacement_bytes = replacement_allocation_ceiling.unwrap_or(0);
        let admitted_bytes = observed
            .checked_add(replacement_bytes)
            .and_then(|bytes| bytes.checked_add(pending_capacity))
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: observed,
                additional_bytes: replacement_bytes.saturating_add(pending_capacity),
            })?;
        Ok(PendingEntryCapacityPlan {
            observed_bytes: observed,
            pending_bytes: pending_capacity,
            old_map_bytes,
            replacement_required_entries,
            replacement_allocation_ceiling,
            admitted_bytes,
        })
    }

    #[cfg(test)]
    fn pending_entry_capacity_bytes(
        &self,
        partition_idx: usize,
        pending_capacity: usize,
    ) -> Result<usize, MemoryGrantError> {
        self.pending_entry_capacity_plan(partition_idx, pending_capacity)
            .map(|plan| plan.admitted_bytes)
    }

    fn validated_reserved_entry_capacities(
        plan: PendingEntryCapacityPlan,
        replacement: &PartitionMap<V>,
    ) -> Result<ReservedEntryCapacities, &'static str> {
        let required_entries = plan
            .replacement_required_entries
            .ok_or("native partition allocated an unplanned replacement map")?;
        let allocation_ceiling = plan
            .replacement_allocation_ceiling
            .ok_or("native partition replacement lacks an allocation ceiling")?;
        let actual_map_bytes = Self::partition_map_allocation_bytes(replacement);
        if actual_map_bytes > allocation_ceiling {
            return Err("native partition replacement exceeds its pre-admitted allocation ceiling");
        }
        if replacement.capacity() < required_entries {
            return Err("native partition replacement did not retain its requested entry capacity");
        }
        let peak_bytes = plan
            .observed_bytes
            .checked_add(actual_map_bytes)
            .and_then(|bytes| bytes.checked_add(plan.pending_bytes))
            .ok_or("native partition replacement peak capacity overflow")?;
        let stable_without_old = plan
            .observed_bytes
            .checked_sub(plan.old_map_bytes)
            .ok_or("native partition map capacity exceeds observed resident capacity")?;
        let final_bytes = stable_without_old
            .checked_add(actual_map_bytes)
            .and_then(|bytes| bytes.checked_add(plan.pending_bytes))
            .ok_or("native partition replacement final capacity overflow")?;
        if peak_bytes > plan.admitted_bytes || final_bytes > plan.admitted_bytes {
            return Err("native partition replacement exceeds its pre-admitted byte ceiling");
        }
        Ok(ReservedEntryCapacities {
            peak_bytes,
            final_bytes,
        })
    }

    /// Releases a provisional native-map admission to its immutable receipt.
    /// The rejected physical replacement must be destroyed before this call.
    /// In particular, rollback never re-enters the caller-supplied retained-
    /// capacity inspector and therefore cannot replace the primary failure.
    fn rollback_native_map_admission(
        &mut self,
        observed_bytes: usize,
    ) -> Result<(), MemoryGrantError> {
        self.grant
            .as_mut()
            .expect("accounted map admission retains its root grant")
            .try_resize(observed_bytes)
    }

    fn reserve_new_entry_with_one_spill(
        &mut self,
        partition_idx: usize,
        pending_capacity: usize,
        cancellation: Option<&QueryCancellationToken>,
        retry_budget: &mut OneSpillRetryBudget,
    ) -> Result<(), PartitionOperationError> {
        self.reserve_new_entry_with_one_spill_inner(
            partition_idx,
            MapAdmissionCapacity::InspectResident {
                pending_bytes: pending_capacity,
            },
            cancellation,
            retry_budget,
        )
    }

    fn reserve_new_entry_with_one_spill_inner(
        &mut self,
        partition_idx: usize,
        capacity: MapAdmissionCapacity,
        cancellation: Option<&QueryCancellationToken>,
        retry_budget: &mut OneSpillRetryBudget,
    ) -> Result<(), PartitionOperationError> {
        if self.grant.is_none() {
            self.partitions[partition_idx]
                .as_mut()
                .expect("ordinary insertion keeps its partition resident")
                .try_reserve(1)
                .map_err(|error| {
                    std::io::Error::other(format!(
                        "failed to reserve native partition entry: {error}"
                    ))
                })?;
            return Ok(());
        }

        let plan = loop {
            check_cancellation(cancellation)?;
            let provisional = match capacity {
                MapAdmissionCapacity::InspectResident { pending_bytes } => {
                    self.pending_entry_capacity_plan(partition_idx, pending_bytes)
                }
                MapAdmissionCapacity::AggregateRoot => {
                    // Only the mediated aggregate reaches this boundary: its
                    // successful updates reconcile retained bytes, while key,
                    // base, update and scheduling workspaces are separate
                    // children. The root therefore covers the complete live
                    // resident state without rescanning unrelated entries.
                    // Read it again after any spill, which changes that state.
                    let observed = self
                        .grant
                        .as_ref()
                        .ok_or_else(|| {
                            native_map_invariant("accounted map admission lost its root grant")
                        })?
                        .size();
                    self.pending_entry_capacity_plan_with_observed(partition_idx, 0, observed)
                }
            }
            .map_err(partition_memory_error)?;
            let admission = match capacity {
                MapAdmissionCapacity::InspectResident { .. } => self
                    .grant
                    .as_mut()
                    .expect("accounted path retains its root grant")
                    .try_resize(provisional.admitted_bytes),
                MapAdmissionCapacity::AggregateRoot => {
                    let additional = provisional
                        .admitted_bytes
                        .checked_sub(provisional.observed_bytes)
                        .ok_or_else(|| {
                            native_map_invariant(
                                "aggregate map admission lost its retained baseline",
                            )
                        })?;
                    let already_recovered = retry_budget.spent;
                    let workspace = self.lend_row_workspace(
                        additional,
                        Some(partition_idx),
                        cancellation,
                        retry_budget,
                    )?;
                    if retry_budget.spent != already_recovered {
                        // A victim changed the retained root. Recompute the
                        // immutable map plan before merging its borrowed peak.
                        self.return_row_grant(workspace.into_grant()?, retry_budget)?;
                        continue;
                    }
                    let grant = workspace.into_grant()?;
                    let root = self.grant.as_mut().ok_or_else(|| {
                        native_map_invariant("aggregate map admission lost its root authority")
                    })?;
                    if let Err(grant) = root.try_merge(grant) {
                        drop(PartitionWorkspace::new(
                            grant,
                            self.failure_cleanup.as_ref(),
                        ));
                        return Err(PartitionOperationError::NativeMapInvariant {
                            message: "aggregate map admission received incompatible authority",
                        });
                    }
                    Ok(())
                }
            };
            match admission {
                Ok(()) => break provisional,
                Err(error) if is_grant_denial(&error) => {
                    if !self.spill_one_cold_non_target(
                        Some(partition_idx),
                        cancellation,
                        retry_budget,
                        &error,
                    )? {
                        return Err(partition_memory_error(error));
                    }
                }
                Err(error) => return Err(partition_memory_error(error)),
            }
        };

        let final_bytes = if let Some(required_entries) = plan.replacement_required_entries {
            let mut replacement = new_partition_map();
            if let Err(error) = replacement.try_reserve(required_entries) {
                drop(replacement);
                let error = NativeMapAllocationError::from_hashbrown(error);
                let rollback = self.rollback_native_map_admission(plan.observed_bytes);
                return match rollback {
                    Ok(()) => Err(PartitionOperationError::NativeMapAllocation { error }),
                    Err(rollback) => {
                        Err(PartitionOperationError::NativeMapAllocationWithRollback {
                            error,
                            rollback,
                        })
                    }
                };
            }

            let capacities = match Self::validated_reserved_entry_capacities(plan, &replacement) {
                Ok(capacities) => capacities,
                Err(message) => {
                    drop(replacement);
                    let rollback = self.rollback_native_map_admission(plan.observed_bytes);
                    return match rollback {
                        Ok(()) => Err(PartitionOperationError::NativeMapInvariant { message }),
                        Err(rollback) => {
                            Err(PartitionOperationError::NativeMapInvariantWithRollback {
                                message,
                                rollback,
                            })
                        }
                    };
                }
            };
            debug_assert!(capacities.peak_bytes <= self.granted_bytes());

            let partition = self.partitions[partition_idx]
                .as_mut()
                .expect("capacity admission keeps target partition resident");
            std::mem::swap(partition, &mut replacement);
            // `capacity()` is the number of entries insertable without
            // reallocating. Reserving old_len + 1 therefore makes moving the
            // old entries formally non-growing and leaves one publication
            // slot. The old map remains charged in `peak_bytes` until drop.
            partition.extend(replacement.drain());
            drop(replacement);
            capacities.final_bytes
        } else {
            plan.admitted_bytes
        };
        debug_assert!(final_bytes <= plan.admitted_bytes);
        // This is only a shrink after the old map was physically destroyed.
        // If a poisoned account rejects release, the state remains
        // conservatively over-authorized and no logical entry was published.
        let root = self
            .grant
            .as_mut()
            .expect("accounted path retains its root grant");
        match capacity {
            MapAdmissionCapacity::InspectResident { .. } => {
                root.try_resize(final_bytes).map_err(partition_memory_error)
            }
            MapAdmissionCapacity::AggregateRoot => {
                let retired = root.size().checked_sub(final_bytes).ok_or_else(|| {
                    native_map_invariant("aggregate map lost its completed backing authority")
                })?;
                let surplus = root.split(retired).ok_or_else(|| {
                    native_map_invariant("aggregate map lost its retired backing surplus")
                })?;
                self.return_row_grant(surplus, retry_budget)
            }
        }
    }

    fn release_idle_row_authority(
        retry: &mut OneSpillRetryBudget,
    ) -> Result<(), PartitionOperationError> {
        if let Some(idle) = retry.recovery.as_mut() {
            idle.try_resize(0).map_err(partition_memory_error)?;
        }
        Ok(())
    }

    fn ensure_row_capacity(
        &mut self,
        required: usize,
        target: Option<usize>,
        cancellation: Option<&QueryCancellationToken>,
        retry: &mut OneSpillRetryBudget,
    ) -> Result<(), PartitionOperationError> {
        let mut idle = retry
            .recovery
            .take()
            .ok_or_else(|| native_map_invariant("aggregate row lost its idle authority"))?;
        if idle.size() >= required {
            retry.recovery = Some(idle);
            return Ok(());
        }
        // Only growth can call the global account or its eviction callbacks.
        // Pure local transfers remain within the surrounding row checkpoints.
        check_cancellation(cancellation)?;
        let growth = idle.try_resize(required);
        match growth {
            Ok(()) => {}
            Err(error) if is_grant_denial(&error) => {
                check_cancellation(cancellation)?;
                idle.try_resize(0).map_err(partition_memory_error)?;
                if !self.spill_one_cold_non_target(target, cancellation, retry, &error)? {
                    return Err(partition_memory_error(error));
                }
                check_cancellation(cancellation)?;
                idle.try_resize(required).map_err(partition_memory_error)?;
            }
            Err(error) => return Err(partition_memory_error(error)),
        }
        retry.recovery = Some(idle);
        check_cancellation(cancellation)?;
        Ok(())
    }

    fn lend_row_workspace(
        &mut self,
        bytes: usize,
        target: Option<usize>,
        cancellation: Option<&QueryCancellationToken>,
        retry: &mut OneSpillRetryBudget,
    ) -> Result<PartitionWorkspace, PartitionOperationError> {
        self.ensure_row_capacity(bytes, target, cancellation, retry)?;
        let grant = retry
            .recovery
            .as_mut()
            .and_then(|idle| idle.split(bytes))
            .ok_or_else(|| {
                native_map_invariant("aggregate row cannot lend its admitted authority")
            })?;
        Ok(PartitionWorkspace::new(
            grant,
            self.failure_cleanup.as_ref(),
        ))
    }

    fn return_row_grant(
        &self,
        grant: MemoryGrant,
        retry: &mut OneSpillRetryBudget,
    ) -> Result<(), PartitionOperationError> {
        Self::return_row_grant_with_cleanup(grant, retry, self.failure_cleanup.as_ref())
    }

    fn return_row_grant_with_cleanup(
        grant: MemoryGrant,
        retry: &mut OneSpillRetryBudget,
        cleanup: Option<&AccountedError>,
    ) -> Result<(), PartitionOperationError> {
        let Some(idle) = retry.recovery.as_mut() else {
            drop(PartitionWorkspace::new(grant, cleanup));
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "aggregate row lost its idle authority during return",
            });
        };
        if let Err(grant) = idle.try_merge(grant) {
            drop(PartitionWorkspace::new(grant, cleanup));
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "aggregate row returned incompatible authority",
            });
        }
        Ok(())
    }

    fn stage_reused_aggregate_key(
        &mut self,
        key: &[Value],
        index: usize,
        cancellation: Option<&QueryCancellationToken>,
        retry: &mut OneSpillRetryBudget,
    ) -> Result<AccountedSerializedKey, PartitionOperationError> {
        let measurement =
            measure_serialized_row_with_limits(key, self.frame_limits.codec_limits())?;
        if measurement.counter_sort_entries != 0 || measurement.counter_sort_key_bytes != 0 {
            // Counter scratch keeps its existing independently admitted codec
            // path. Only genuinely idle bytes are returned before it starts.
            Self::release_idle_row_authority(retry)?;
            return self.stage_accounted_key_with_one_spill(key, index, cancellation, retry);
        }
        let peak = measurement
            .encoded_bytes
            .checked_mul(2)
            .ok_or_else(|| native_map_invariant("aggregate key staging peak overflow"))?;
        let workspace = self.lend_row_workspace(peak, Some(index), cancellation, retry)?;
        let mut writer = GrantRecordWriter {
            bytes: Vec::new(),
            grant: workspace.into_grant()?,
            maximum: usize::try_from(self.frame_limits.max_plaintext_bytes()).unwrap_or(usize::MAX),
        };
        writer.prepare_capacity_inner(measurement.encoded_bytes, false)?;
        let mut scratch = CounterSortScratch::new();
        let encoded = serialize_row_with_prepared_scratch(
            key,
            &mut writer,
            self.frame_limits.codec_limits(),
            &mut scratch,
        )?;
        if encoded != measurement.encoded_bytes || writer.bytes.len() != measurement.encoded_bytes {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "aggregate key differs from its measured length",
            });
        }
        check_cancellation(cancellation)?;
        let unused = writer
            .grant
            .size()
            .checked_sub(writer.bytes.capacity())
            .ok_or_else(|| native_map_invariant("aggregate key exceeds its retained authority"))?;
        let surplus = writer
            .grant
            .split(unused)
            .ok_or_else(|| native_map_invariant("aggregate key lost its retired surplus"))?;
        self.return_row_grant(surplus, retry)?;
        let GrantRecordWriter { bytes, grant, .. } = writer;
        Ok(AccountedSerializedKey {
            key: Some(SerializedKey(bytes)),
            grant: Some(grant),
        })
    }

    fn stage_accounted_key_with_one_spill(
        &mut self,
        key: &[Value],
        partition_idx: usize,
        cancellation: Option<&QueryCancellationToken>,
        retry_budget: &mut OneSpillRetryBudget,
    ) -> Result<AccountedSerializedKey, PartitionOperationError> {
        check_cancellation(cancellation)?;
        self.ensure_not_draining()?;

        let first_error = match AccountedSerializedKey::from_values(
            key,
            self.frame_limits,
            self.grant
                .as_mut()
                .expect("accounted key staging retains its root grant"),
        ) {
            Ok(staged) => {
                check_cancellation(cancellation)?;
                return Ok(staged);
            }
            Err(error) if is_grant_denial_io(&error) => error,
            Err(error) => return Err(error.into()),
        };

        check_cancellation(cancellation)?;
        let Some(denial) = first_error
            .get_ref()
            .and_then(|source| source.downcast_ref::<MemoryGrantError>())
        else {
            return Err(first_error.into());
        };
        if !self.spill_one_cold_non_target(
            Some(partition_idx),
            cancellation,
            retry_budget,
            denial,
        )? {
            return Err(first_error.into());
        }
        check_cancellation(cancellation)?;

        let staged = AccountedSerializedKey::from_values(
            key,
            self.frame_limits,
            self.grant
                .as_mut()
                .expect("accounted key retry retains its root grant"),
        )?;
        check_cancellation(cancellation)?;
        Ok(staged)
    }

    fn restore_recovery_allowance(
        &mut self,
        target: Option<usize>,
        cancellation: Option<&QueryCancellationToken>,
        retry: &mut OneSpillRetryBudget,
    ) -> Result<(), PartitionOperationError> {
        let mut reserve = retry
            .recovery
            .take()
            .ok_or_else(|| native_map_invariant("aggregate recovery allowance is absent"))?;
        let required = retry.recovery_required;
        if reserve.size() < required
            && let Err(error) = reserve.try_resize(required)
        {
            if !is_grant_denial(&error) {
                return Err(partition_memory_error(error));
            }
            reserve.try_resize(0).map_err(partition_memory_error)?;
            if !self.spill_one_cold_non_target(target, cancellation, retry, &error)? {
                return Err(partition_memory_error(error));
            }
            reserve
                .try_resize(required)
                .map_err(partition_memory_error)?;
        }
        retry.recovery = Some(reserve);
        Ok(())
    }

    fn resize_transient_grant_with_one_spill(
        &mut self,
        grant: &mut MemoryGrant,
        required: usize,
        partition_idx: Option<usize>,
        cancellation: Option<&QueryCancellationToken>,
        retry_budget: &mut OneSpillRetryBudget,
    ) -> Result<(), PartitionOperationError> {
        check_cancellation(cancellation)?;
        let first_error = match grant.try_resize(required) {
            Ok(()) => {
                check_cancellation(cancellation)?;
                return Ok(());
            }
            Err(error) if is_grant_denial(&error) => error,
            Err(error) => return Err(partition_memory_error(error)),
        };

        check_cancellation(cancellation)?;
        if !self.spill_one_cold_non_target(
            partition_idx,
            cancellation,
            retry_budget,
            &first_error,
        )? {
            return Err(partition_memory_error(first_error));
        }
        check_cancellation(cancellation)?;

        grant.try_resize(required).map_err(partition_memory_error)?;
        check_cancellation(cancellation)?;
        Ok(())
    }

    fn recovery_partition(
        &self,
        target: Option<usize>,
        denial: &MemoryGrantError,
    ) -> Option<usize> {
        // Only the private aggregate path stores observed retained bytes in
        // every entry. Compatibility callers keep their original LRU policy.
        let minimum_reclaim = if self.failure_cleanup.is_some() {
            Some(match denial {
                MemoryGrantError::LimitExceeded {
                    requested_bytes,
                    limit_bytes,
                    ..
                } => requested_bytes.checked_sub(*limit_bytes)?,
                MemoryGrantError::Denied { additional_bytes } => *additional_bytes,
                _ => return None,
            })
        } else {
            None
        };
        self.partitions
            .iter()
            .enumerate()
            .filter_map(|(index, partition)| {
                let map = partition.as_ref()?;
                if Some(index) == target || map.is_empty() {
                    return None;
                }
                let reclaim = if let Some(minimum) = minimum_reclaim {
                    let reclaim = map.iter().try_fold(
                        Self::partition_map_allocation_bytes(map),
                        |bytes, (key, entry)| {
                            bytes
                                .checked_add(key.0.capacity())?
                                .checked_add(entry.resident_bound)
                        },
                    )?;
                    if reclaim < minimum {
                        return None;
                    }
                    reclaim
                } else {
                    0
                };
                Some((index, reclaim))
            })
            // Protected recovery amortizes a durable spill over the largest
            // observed reclaim. Legacy candidates all rank zero, preserving
            // their LRU ordering; equal protected reclaims also use age.
            .min_by_key(|(index, reclaim)| (std::cmp::Reverse(*reclaim), self.access_times[*index]))
            .map(|(index, _)| index)
    }

    fn spill_one_cold_non_target(
        &mut self,
        partition_idx: Option<usize>,
        cancellation: Option<&QueryCancellationToken>,
        retry_budget: &mut OneSpillRetryBudget,
        denial: &MemoryGrantError,
    ) -> Result<bool, PartitionOperationError> {
        if retry_budget.spent {
            return Ok(false);
        }
        let Some(candidate) = self.recovery_partition(partition_idx, denial) else {
            return Ok(false);
        };
        Self::release_idle_row_authority(retry_budget)?;
        // The finished file occupies an already-admitted catalog slot; its
        // writer/provider workspaces retire before resident authority releases.
        // Sufficient final reclaim is not a promise that consolidation fits:
        // reader, frame and codec peaks still admit independently and may fail.
        retry_budget.spent = true;
        let spilled = self.spill_partition_inner(candidate, cancellation)?;
        Ok(spilled != 0)
    }

    fn ensure_not_draining(&self) -> std::io::Result<()> {
        if self.drain_state != DrainState::Idle {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "partitioned state has already begun its destructive drain",
            ));
        }
        Ok(())
    }

    fn reject_legacy_accounted(&self, operation: &'static str) -> std::io::Result<()> {
        if self.grant.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("accounted partition state requires the bounded {operation} API"),
            ));
        }
        Ok(())
    }

    fn decoded_key_resident_bound(encoded_len: usize) -> Result<usize, MemoryGrantError> {
        // Every recursively decoded item must consume at least one wire byte.
        // Charge one Value slot, one possible Vec header, and allocator slack
        // per byte, plus the top-level Vec allocation. This deliberately
        // conservative envelope is retained by each yielded cursor entry.
        let per_wire_byte = std::mem::size_of::<Value>()
            .checked_add(std::mem::size_of::<Vec<Value>>())
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: std::mem::size_of::<Value>(),
                additional_bytes: std::mem::size_of::<Vec<Value>>(),
            })?;
        encoded_len
            .checked_mul(per_wire_byte)
            .ok_or(MemoryGrantError::ArithmeticOverflow {
                current_bytes: encoded_len,
                additional_bytes: per_wire_byte,
            })
    }

    fn split_workspace_grant(
        &mut self,
        bytes: usize,
    ) -> Result<MemoryGrant, PartitionOperationError> {
        let grant = self
            .grant
            .as_mut()
            .expect("resource-qualified partition operation retains its root grant");
        let expanded = grant.size().checked_add(bytes).ok_or_else(|| {
            partition_memory_error(MemoryGrantError::ArithmeticOverflow {
                current_bytes: grant.size(),
                additional_bytes: bytes,
            })
        })?;
        grant.try_resize(expanded).map_err(partition_memory_error)?;
        Ok(grant
            .split(bytes)
            .expect("just-admitted workspace bytes are available to split"))
    }

    /// Returns the partition index for a key.
    #[must_use]
    pub fn partition_for(&self, key: &[Value]) -> usize {
        let hash = hash_key(key);
        // reason: on 64-bit targets u64 == usize; on 32-bit the modulo handles overflow
        #[allow(clippy::cast_possible_truncation)]
        {
            hash as usize % self.num_partitions
        }
    }

    /// Updates access time for a partition.
    fn touch(&mut self, partition_idx: usize) {
        self.timestamp += 1;
        self.access_times[partition_idx] = self.timestamp;
    }

    /// Gets the in-memory partition, loading from disk if spilled.
    ///
    /// # Errors
    ///
    /// Returns an error if reading from disk fails.
    fn get_partition_mut(&mut self, partition_idx: usize) -> std::io::Result<&mut PartitionMap<V>> {
        self.get_partition_mut_inner(partition_idx, None)
            .map_err(PartitionOperationError::into_io)
    }

    fn get_partition_mut_controlled(
        &mut self,
        partition_idx: usize,
    ) -> Result<&mut PartitionMap<V>, PartitionOperationError> {
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        self.get_partition_mut_inner(partition_idx, cancellation.as_ref())
    }

    fn get_partition_mut_inner(
        &mut self,
        partition_idx: usize,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<&mut PartitionMap<V>, PartitionOperationError> {
        self.ensure_not_draining()?;
        self.get_partition_mut_unchecked_inner(partition_idx, cancellation)
    }

    fn get_partition_mut_unchecked_inner(
        &mut self,
        partition_idx: usize,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<&mut PartitionMap<V>, PartitionOperationError> {
        check_cancellation(cancellation)?;
        self.touch(partition_idx);

        if self.grant.is_some() && self.spill_files[partition_idx].is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "accounted spilled partition requires bounded base/delta access",
            )
            .into());
        }

        // If partition is in memory, return it
        if self.partitions[partition_idx].is_some() {
            // Invariant: just checked is_some() above
            return Ok(self.partitions[partition_idx]
                .as_mut()
                .expect("partition is Some: checked on previous line"));
        }

        // Load from disk
        if self.spill_files[partition_idx].is_some() {
            let loaded = {
                let spill_file = self.spill_files[partition_idx]
                    .as_ref()
                    .expect("spill file presence checked above");
                self.load_partition_inner(
                    spill_file,
                    self.partition_sizes[partition_idx],
                    cancellation,
                )?
            };
            // Cancellation before deletion leaves the authoritative spill file
            // installed and retryable. A successful delete is followed by an
            // atomic in-memory installation before the next poll.
            check_cancellation(cancellation)?;
            self.spill_files[partition_idx]
                .as_mut()
                .expect("spill file presence checked above")
                .close_and_delete()?;
            self.spill_files[partition_idx] = None;
            self.spill_base_sizes[partition_idx] = 0;
            self.partitions[partition_idx] = Some(loaded);
            check_cancellation(cancellation)?;
        } else {
            // Neither in memory nor on disk - create empty partition
            self.partitions[partition_idx] = Some(new_partition_map());
        }

        // Invariant: partition was either loaded from disk or created empty above
        Ok(self.partitions[partition_idx]
            .as_mut()
            .expect("partition is Some: set to Some in if/else branches above"))
    }

    /// Loads a partition from a spill file.
    #[cfg(test)]
    fn load_partition(
        &self,
        spill_file: &SpillFile,
        expected_entries: usize,
    ) -> std::io::Result<PartitionMap<V>> {
        self.load_partition_inner(spill_file, expected_entries, None)
            .map_err(PartitionOperationError::into_io)
    }

    fn load_partition_inner(
        &self,
        spill_file: &SpillFile,
        expected_entries: usize,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<PartitionMap<V>, PartitionOperationError> {
        check_cancellation(cancellation)?;
        let mut reader = spill_file.reader()?;
        let num_entries = reader.read_partition_start()?;
        check_cancellation(cancellation)?;
        let expected_entries = u64::try_from(expected_entries).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tracked partition entry count exceeds u64",
            )
        })?;
        if num_entries != expected_entries {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "framed partition declares {num_entries} entries, tracked {expected_entries}"
                ),
            )
            .into());
        }
        let mut partition = new_partition_map();

        for _ in 0..num_entries {
            check_cancellation(cancellation)?;
            let payload = reader.read_partition_entry()?;
            let (serialized_key, num_key_columns, resident_bound, value) =
                self.decode_partition_entry(&payload)?;
            partition.try_reserve(1).map_err(|error| {
                std::io::Error::other(format!("failed to reserve partition entry: {error}"))
            })?;
            if partition
                .insert(
                    serialized_key,
                    PartitionEntry {
                        num_key_columns,
                        resident_bound,
                        value,
                    },
                )
                .is_some()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "duplicate key in framed native partition",
                )
                .into());
            }
            check_cancellation(cancellation)?;
        }
        reader.finish()?;
        check_cancellation(cancellation)?;
        Ok(partition)
    }

    fn decode_partition_entry(
        &self,
        payload: &[u8],
    ) -> std::io::Result<(SerializedKey, usize, usize, V)> {
        self.decode_partition_entry_inner(payload, true)
    }

    fn decode_partition_entry_without_key_validation(
        &self,
        payload: &[u8],
    ) -> std::io::Result<(SerializedKey, usize, usize, V)> {
        self.decode_partition_entry_inner(payload, false)
    }

    fn decode_partition_entry_inner(
        &self,
        payload: &[u8],
        validate_key: bool,
    ) -> std::io::Result<(SerializedKey, usize, usize, V)> {
        let mut cursor = std::io::Cursor::new(payload);
        let key_len_u64 = read_u64(&mut cursor)
            .map_err(|error| malformed_partition_payload(error, "partition key length"))?;
        let key_start = usize::try_from(cursor.position()).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "partition cursor overflow")
        })?;
        let remaining_after_length = payload.len().checked_sub(key_start).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition cursor past payload",
            )
        })?;
        let Some(maximum_key_len) = remaining_after_length.checked_sub(16) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key length exceeds framed payload",
            ));
        };
        let maximum_encoded_length = u64::try_from(maximum_key_len).unwrap_or(u64::MAX);
        if key_len_u64 > maximum_encoded_length {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key length exceeds framed payload",
            ));
        }
        let key_len = usize::try_from(key_len_u64).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key length is not addressable",
            )
        })?;
        let mut key = Vec::new();
        key.try_reserve_exact(key_len).map_err(|error| {
            std::io::Error::other(format!("failed to reserve partition key: {error}"))
        })?;
        key.resize(key_len, 0);
        cursor.read_exact(&mut key)?;
        let num_key_columns_u64 = read_u64(&mut cursor)
            .map_err(|error| malformed_partition_payload(error, "partition key column count"))?;
        let num_key_columns = usize::try_from(num_key_columns_u64).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key column count is not addressable",
            )
        })?;
        let serialized_key = SerializedKey(key);
        // Fail malformed keys before invoking a potentially expensive custom
        // state decoder.
        if validate_key {
            serialized_key.to_values(num_key_columns, self.frame_limits)?;
        }
        let custom_len_u64 = read_u64(&mut cursor)
            .map_err(|error| malformed_partition_payload(error, "partition custom-state length"))?;
        let custom_len = usize::try_from(custom_len_u64).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition custom-state length is not addressable",
            )
        })?;
        let position = usize::try_from(cursor.position()).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "partition cursor overflow")
        })?;
        let remaining = payload.len().checked_sub(position).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition cursor past payload",
            )
        })?;
        let trailer_len = remaining.checked_sub(custom_len).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("partition custom-state length {custom_len} exceeds {remaining} bytes"),
            )
        })?;
        if trailer_len != 0 && trailer_len != 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "partition custom-state length {custom_len} leaves invalid {trailer_len}-byte metadata"
                ),
            ));
        }
        let custom_end = position.checked_add(custom_len).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition custom-state range overflow",
            )
        })?;
        let custom = &payload[position..custom_end];
        let resident_bound = if trailer_len == 0 {
            0
        } else {
            let mut trailer = std::io::Cursor::new(&payload[custom_end..]);
            usize::try_from(read_u64(&mut trailer)?).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "partition resident bound is not addressable",
                )
            })?
        };
        let mut custom_cursor = std::io::Cursor::new(custom);
        let codec_limits = self
            .frame_limits
            .codec_limits()
            .bounded_to_payload(custom.len());
        let value =
            (self.value_deserializer)(&mut custom_cursor, codec_limits).map_err(|error| {
                malformed_partition_payload(error, "partition custom-state payload")
            })?;
        let consumed = usize::try_from(custom_cursor.position()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition custom-state cursor overflow",
            )
        })?;
        if consumed != custom.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "trailing bytes in partition custom-state payload",
            ));
        }
        Ok((serialized_key, num_key_columns, resident_bound, value))
    }

    fn partition_payload_key_and_bound(payload: &[u8]) -> std::io::Result<(&[u8], usize, usize)> {
        let key_len_bytes = payload.get(..8).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated partition key length",
            )
        })?;
        let key_len = usize::try_from(u64::from_le_bytes(
            key_len_bytes
                .try_into()
                .expect("checked eight-byte partition key length"),
        ))
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key length is not addressable",
            )
        })?;
        let key_start = 8usize;
        let key_end = key_start.checked_add(key_len).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key range overflow",
            )
        })?;
        let key = payload.get(key_start..key_end).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key length exceeds framed payload",
            )
        })?;
        let key_columns_end = key_end.checked_add(8).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key-column range overflow",
            )
        })?;
        let key_columns_slice = payload.get(key_end..key_columns_end).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated partition key column count",
            )
        })?;
        let num_key_columns = usize::try_from(u64::from_le_bytes(
            key_columns_slice
                .try_into()
                .expect("checked eight-byte partition key column count"),
        ))
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition key column count is not addressable",
            )
        })?;
        let custom_len_start = key_columns_end;
        let custom_len_end = custom_len_start.checked_add(8).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition custom-length range overflow",
            )
        })?;
        let custom_len_slice = payload
            .get(custom_len_start..custom_len_end)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "truncated partition custom-state length",
                )
            })?;
        let custom_len = usize::try_from(u64::from_le_bytes(
            custom_len_slice
                .try_into()
                .expect("checked eight-byte custom-state length"),
        ))
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition custom-state length is not addressable",
            )
        })?;
        let custom_end = custom_len_end.checked_add(custom_len).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition custom-state range overflow",
            )
        })?;
        let trailer = payload.get(custom_end..).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "partition custom-state length exceeds framed payload",
            )
        })?;
        let resident_bound = match trailer {
            [] => 0,
            bytes if bytes.len() == 8 => usize::try_from(u64::from_le_bytes(
                bytes
                    .try_into()
                    .expect("checked eight-byte partition resident bound"),
            ))
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "partition resident bound is not addressable",
                )
            })?,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid partition resident-bound metadata",
                ));
            }
        };
        Ok((key, num_key_columns, resident_bound))
    }

    fn lookup_spilled_base_entry_accounted(
        &mut self,
        partition_idx: usize,
        target: &SerializedKey,
        expected_num_key_columns: usize,
        cancellation: Option<&QueryCancellationToken>,
        retry_budget: &mut OneSpillRetryBudget,
    ) -> Result<Option<AccountedSpilledBaseEntry<V>>, PartitionOperationError> {
        loop {
            let attempt = self.lookup_spilled_base_entry_accounted_once(
                partition_idx,
                target,
                expected_num_key_columns,
                cancellation,
            )?;
            match attempt {
                AccountedSpilledBaseLookup::Complete(found) => return Ok(found),
                AccountedSpilledBaseLookup::AdmissionDenied(denial) => {
                    // Reader markers precede all decoding; frame markers are
                    // emitted only while no matching record has been decoded;
                    // decode-envelope markers precede the decoder/capacity callbacks.
                    // Arbitrary I/O source chains are never classified, so a
                    // retry cannot duplicate a callback side effect.
                    let denial = denial.into_error();
                    if self.spill_one_cold_non_target(
                        Some(partition_idx),
                        cancellation,
                        retry_budget,
                        &denial,
                    )? {
                        continue;
                    }
                    return Err(partition_memory_error(denial));
                }
            }
        }
    }

    fn lookup_spilled_base_entry_accounted_once(
        &mut self,
        partition_idx: usize,
        target: &SerializedKey,
        expected_num_key_columns: usize,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<AccountedSpilledBaseLookup<V>, PartitionOperationError> {
        if self.spill_files[partition_idx].is_none() {
            return Ok(AccountedSpilledBaseLookup::Complete(None));
        }
        let (reader_workspace, frame_workspace) = {
            let grant = self
                .grant
                .as_mut()
                .expect("accounted base lookup retains its root grant");
            (
                grant
                    .split(0)
                    .expect("zero-byte reader workspace can always be split"),
                grant
                    .split(0)
                    .expect("zero-byte frame workspace can always be split"),
            )
        };
        let cleanup = self.failure_cleanup.clone();
        let mut reader_workspace = PartitionWorkspace::new(reader_workspace, cleanup.as_ref());
        let frame_workspace = PartitionWorkspace::new(frame_workspace, cleanup.as_ref());
        let file = self.spill_files[partition_idx]
            .as_ref()
            .expect("spill-file presence checked above");
        check_cancellation(cancellation)?;
        let mut reader_admission_denial = None;
        let reader = open_partition_reader(
            file,
            |required| {
                grow_workspace_recording_denial(
                    reader_workspace
                        .grant_mut()
                        .map_err(PartitionOperationError::into_io)?,
                    required,
                    &mut reader_admission_denial,
                )
            },
            cleanup.as_ref(),
        );
        let reader = match reader {
            Ok(reader) => reader,
            Err(error) => {
                if let Some(denial) = reader_admission_denial {
                    drop(error);
                    reader_workspace.release();
                    frame_workspace.release();
                    return Ok(AccountedSpilledBaseLookup::AdmissionDenied(
                        AccountedSpilledBaseAdmissionDenial::ReaderWorkspace(denial),
                    ));
                }
                return Err(error.into());
            }
        };
        let mut reader = BackstoppedPartitionReader::new(
            reader,
            frame_workspace.into_grant()?,
            reader_workspace.into_grant()?,
        );
        reader.cleanup.clone_from(&cleanup);
        let declared = match reader.reader_mut().read_partition_start() {
            Ok(declared) => declared,
            Err(error) if cleanup.is_some() => return Err(error.into()),
            Err(error) => return Err(accounted_partition_reader_io_failure(error, reader)),
        };
        let expected = u64::try_from(self.spill_base_sizes[partition_idx]).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tracked spilled-base entry count exceeds u64",
            )
        })?;
        if declared != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("framed spilled base declares {declared} entries, tracked {expected}"),
            )
            .into());
        }
        check_cancellation(cancellation)?;

        let mut found = None;
        let scan_result: Result<
            Option<AccountedSpilledBaseAdmissionDenial>,
            PartitionOperationError,
        > = (|| {
            for _ in 0..declared {
                check_cancellation(cancellation)?;
                let mut frame_admission_denial = None;
                let payload =
                    reader.read_partition_entry_recording_denial(&mut frame_admission_denial);
                let payload = match payload {
                    Ok(payload) => payload,
                    Err(error) => {
                        if let Some(denial) = frame_admission_denial {
                            drop(error);
                            if found.is_some() {
                                // A previous match already ran the custom
                                // decoder/capacity callback. Fail terminally
                                // rather than restart and replay that callback.
                                return Err(partition_memory_error(denial));
                            }
                            return Ok(Some(AccountedSpilledBaseAdmissionDenial::FrameWorkspace(
                                denial,
                            )));
                        }
                        return Err(error.into());
                    }
                };
                let (encoded_key, num_key_columns, resident_bound) =
                    Self::partition_payload_key_and_bound(&payload)?;
                if encoded_key != target.0 {
                    check_cancellation(cancellation)?;
                    continue;
                }
                if num_key_columns != expected_num_key_columns {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "spilled-base key column count disagrees with the lookup key",
                    )
                    .into());
                }
                if found.is_some() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "duplicate key in accounted spilled base",
                    )
                    .into());
                }

                // This existing decoded-key bound is deliberately broad:
                // lookup currently duplicates the serialized-key Vec rather
                // than materializing typed Values. It provides transient
                // safety authority for that duplicate and callback payload;
                // a follow-up can transfer the already-accounted staged key
                // and shrink this peak. The stored value bound is the audited
                // ceiling for the custom value and an escaping callback error
                // or panic construction.
                let decoded_key_bound = Self::decoded_key_resident_bound(encoded_key.len())
                    .map_err(partition_memory_error)?;
                let provisional_decode =
                    decoded_key_bound
                        .checked_add(resident_bound)
                        .ok_or_else(|| {
                            partition_memory_error(MemoryGrantError::ArithmeticOverflow {
                                current_bytes: decoded_key_bound,
                                additional_bytes: resident_bound,
                            })
                        })?;
                let decode_grant = self
                    .grant
                    .as_mut()
                    .ok_or(PartitionOperationError::NativeMapInvariant {
                        message: "accounted base lookup lost root authority",
                    })?
                    .split(0)
                    .ok_or(PartitionOperationError::NativeMapInvariant {
                        message: "accounted base lookup could not split decode authority",
                    })?;
                let mut decode_grant = PartitionWorkspace::new(decode_grant, cleanup.as_ref());
                if let Err(error) = decode_grant.grant_mut()?.try_resize(provisional_decode) {
                    if is_grant_denial(&error) {
                        decode_grant.release();
                        return Ok(Some(AccountedSpilledBaseAdmissionDenial::DecodeEnvelope(
                            error,
                        )));
                    }
                    return Err(partition_memory_error(error));
                }

                let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> std::io::Result<(SerializedKey, V)> {
                        let (key, decoded_num_key_columns, decoded_bound, value) =
                            self.decode_partition_entry_without_key_validation(&payload)?;
                        if decoded_num_key_columns != num_key_columns
                            || decoded_bound != resident_bound
                            || key != *target
                        {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "spilled-base metadata changed during bounded decode",
                            ));
                        }
                        let observed =
                            (self.value_resident_capacity)(&value).map_err(grant_io_error)?;
                        if observed > resident_bound {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!(
                                    "decoded partition value retained {observed} bytes, exceeding its stored {resident_bound}-byte bound"
                                ),
                            ));
                        }
                        let exact_retained =
                            key.0.capacity().checked_add(observed).ok_or_else(|| {
                                grant_io_error(MemoryGrantError::ArithmeticOverflow {
                                    current_bytes: key.0.capacity(),
                                    additional_bytes: observed,
                                })
                            })?;
                        decode_grant
                            .grant_mut()
                            .map_err(PartitionOperationError::into_io)?
                            .try_resize(exact_retained)
                            .map_err(grant_io_error)?;
                        Ok((key, value))
                    },
                ));
                let (key, value) = match decoded {
                    Ok(Ok(decoded)) => decoded,
                    Ok(Err(error)) => {
                        if cleanup.is_some() {
                            return Err(error.into());
                        }
                        return Err(accounted_partition_io_failure(
                            error,
                            decode_grant.into_grant()?,
                        ));
                    }
                    Err(panic) => {
                        if cleanup.is_some() {
                            std::panic::resume_unwind(panic);
                        }
                        let accounted_panic =
                            AccountedPartitionPanic::new(panic, decode_grant.into_grant()?);
                        std::panic::resume_unwind(Box::new(accounted_panic));
                    }
                };
                found = Some(AccountedSpilledBaseEntry {
                    key: Some(key),
                    entry: Some(PartitionEntry {
                        num_key_columns,
                        resident_bound,
                        value,
                    }),
                    grant: Some(decode_grant.into_grant()?),
                });
                check_cancellation(cancellation)?;
            }
            reader.reader_mut().finish()?;
            check_cancellation(cancellation)?;
            Ok(None)
        })();

        match scan_result {
            Ok(None) => {
                if !reader.finish_cleanup() {
                    return Err(PartitionOperationError::NativeMapInvariant {
                        message: "partition reader physical cleanup failed",
                    });
                }
                Ok(AccountedSpilledBaseLookup::Complete(found))
            }
            Ok(Some(denial)) => {
                drop(found);
                if !reader.finish_cleanup() {
                    return Err(PartitionOperationError::NativeMapInvariant {
                        message: "partition reader physical cleanup failed",
                    });
                }
                Ok(AccountedSpilledBaseLookup::AdmissionDenied(denial))
            }
            Err(primary) if cleanup.is_some() => {
                drop(found);
                Err(primary)
            }
            Err(PartitionOperationError::Io(error)) => {
                drop(found);
                Err(accounted_partition_reader_io_failure(error, reader))
            }
            Err(primary) => {
                drop(found);
                Err(primary)
            }
        }
    }

    /// Returns whether a partition is in memory.
    #[must_use]
    pub fn is_in_memory(&self, partition_idx: usize) -> bool {
        self.partitions[partition_idx].is_some()
    }

    /// Returns the number of groups in a partition.
    #[must_use]
    pub fn partition_size(&self, partition_idx: usize) -> usize {
        self.partition_sizes[partition_idx]
    }

    /// Returns the total number of groups across all partitions.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.partition_sizes.iter().sum()
    }

    /// Returns the number of in-memory partitions.
    #[must_use]
    pub fn in_memory_count(&self) -> usize {
        self.partitions.iter().filter(|p| p.is_some()).count()
    }

    /// Returns the number of spilled partitions.
    #[must_use]
    pub fn spilled_count(&self) -> usize {
        self.spill_files.iter().filter(|f| f.is_some()).count()
    }

    /// Spills a specific partition to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_partition(&mut self, partition_idx: usize) -> std::io::Result<usize> {
        if self.failure_cleanup.is_some() {
            return Err(std::io::ErrorKind::Unsupported.into());
        }
        self.spill_partition_inner(partition_idx, None)
            .map_err(PartitionOperationError::into_io)
    }

    #[cfg(test)]
    pub(crate) fn spill_partition_controlled(
        &mut self,
        partition_idx: usize,
    ) -> Result<usize, PartitionOperationError> {
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        self.protect_operation(|state| {
            state.spill_partition_inner(partition_idx, cancellation.as_ref())
        })
    }

    fn protect_operation<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T, PartitionOperationError>,
    ) -> Result<T, PartitionOperationError> {
        if self.failure_cleanup.is_none() {
            return operation(self);
        }
        if self.failure_publisher.is_none() {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "accounted partition is terminally poisoned",
            });
        }
        let mut reserve = self.recovery_allowance.take();
        let restore_bytes = reserve.as_ref().map_or(0, MemoryGrant::size);
        if let Some(grant) = reserve.as_mut()
            && let Err(error) = grant.try_resize(0)
        {
            self.recovery_allowance = reserve;
            return Err(self.publish_failure(Some(partition_memory_error(error)), None, None));
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let value = operation(self)?;
            if let Some(grant) = reserve.as_mut() {
                grant
                    .try_resize(restore_bytes)
                    .map_err(partition_memory_error)?;
            }
            Ok(value)
        }));
        self.recovery_allowance = reserve;
        match outcome {
            Ok(Ok(value)) => match self.check_physical_cleanup() {
                Ok(()) => Ok(value),
                Err(error) => Err(self.publish_failure(Some(error), None, None)),
            },
            Ok(Err(error)) => Err(self.publish_failure(Some(error), None, None)),
            Err(payload) => Err(self.publish_failure(None, None, Some(payload))),
        }
    }

    fn spill_partition_inner(
        &mut self,
        partition_idx: usize,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<usize, PartitionOperationError> {
        check_cancellation(cancellation)?;
        self.ensure_not_draining()?;
        if self.grant.is_some() && self.spill_files[partition_idx].is_some() {
            return self.consolidate_accounted_partition(partition_idx, cancellation, false);
        }
        let mut spill_workspace_root = self.grant.as_mut().map(|grant| {
            grant
                .split(0)
                .expect("zero-byte spill workspace can always be split")
        });
        let Some(partition) = self.partitions[partition_idx].as_ref() else {
            return Ok(0); // Already spilled
        };

        if partition.is_empty() {
            return Ok(0);
        }
        if self.grant.is_some() {
            for entry in partition.values() {
                let observed =
                    (self.value_resident_capacity)(&entry.value).map_err(partition_memory_error)?;
                if observed > entry.resident_bound {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "partition value retained {observed} bytes, exceeding its declared {}-byte lifetime bound",
                            entry.resident_bound
                        ),
                    )
                    .into());
                }
            }
        }
        let partition_len = partition.len();

        let entry_count = u64::try_from(partition.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "partition entry count exceeds u64",
            )
        })?;
        check_cancellation(cancellation)?;
        let mut staging = if let Some(root) = spill_workspace_root.as_mut() {
            let (file, workspace) = create_accounted_partition_file(
                &self.manager,
                root,
                self.failure_cleanup.as_ref(),
            )?;
            PartitionStaging::accounted(file, workspace, self.failure_cleanup.as_ref())
        } else {
            PartitionStaging::unaccounted(self.manager.create_file(SpillFileRole::NativePartition)?)
        };
        let write_result: Result<(), PartitionOperationError> = (|| {
            check_cancellation(cancellation)?;
            staging.file_mut().write_partition_start(entry_count)?;
            check_cancellation(cancellation)?;
            let maximum =
                usize::try_from(self.frame_limits.max_plaintext_bytes()).unwrap_or(usize::MAX);
            for (key, entry) in partition {
                check_cancellation(cancellation)?;
                if let Some(grant) = spill_workspace_root.as_mut() {
                    write_accounted_partition_entry(
                        staging.file_mut(),
                        key,
                        entry,
                        &*self.value_serializer,
                        self.frame_limits,
                        grant,
                        self.failure_cleanup.as_ref(),
                    )?;
                    check_cancellation(cancellation)?;
                    continue;
                }
                let mut payload = SpillRecordBuffer::new(maximum);
                write_u64(
                    &mut payload,
                    u64::try_from(key.0.len()).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "partition key length exceeds u64",
                        )
                    })?,
                )?;
                payload.write_all(&key.0)?;
                write_u64(
                    &mut payload,
                    u64::try_from(entry.num_key_columns).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "partition key column count exceeds u64",
                        )
                    })?,
                )?;
                let custom_length_offset = payload.len();
                write_u64(&mut payload, 0)?;
                let custom_start = payload.len();
                (self.value_serializer)(
                    &entry.value,
                    &mut payload,
                    self.frame_limits.codec_limits(),
                )?;
                let custom_len = payload.len().checked_sub(custom_start).ok_or_else(|| {
                    std::io::Error::other("partition custom-state length underflow")
                })?;
                payload.patch_u64_le(
                    custom_length_offset,
                    u64::try_from(custom_len).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "partition custom-state length exceeds u64",
                        )
                    })?,
                )?;
                staging
                    .file_mut()
                    .write_partition_entry(&payload.into_inner())?;
                check_cancellation(cancellation)?;
            }
            staging.finish_write()?;
            check_cancellation(cancellation)?;
            Ok(())
        })();

        if let Err(primary) = write_result {
            return Err(staging.cleanup_preserving_primary(primary, "partition spill cleanup"));
        }

        let bytes_written = staging.file_mut().bytes_written();
        let Ok(bytes_written_usize) = usize::try_from(bytes_written) else {
            let primary = PartitionOperationError::Io(std::io::Error::other(
                "published partition byte count exceeds addressable range",
            ));
            return Err(staging.cleanup_preserving_primary(primary, "partition byte-count cleanup"));
        };
        if let Err(error) = check_cancellation(cancellation) {
            let primary = PartitionOperationError::Cancelled(error);
            return Err(staging.cleanup_preserving_primary(primary, "partition spill cleanup"));
        }
        let spill_file = staging.into_finished_file();
        self.partitions[partition_idx] = if self.grant.is_some() {
            Some(new_partition_map())
        } else {
            None
        };
        self.spill_base_sizes[partition_idx] = partition_len;
        self.spill_files[partition_idx] = Some(spill_file);
        self.reconcile_grant().map_err(partition_memory_error)?;
        Ok(bytes_written_usize)
    }

    fn consolidate_accounted_partition(
        &mut self,
        partition_idx: usize,
        cancellation: Option<&QueryCancellationToken>,
        final_drain: bool,
    ) -> Result<usize, PartitionOperationError> {
        let (reader_workspace, read_workspace, copy_provider_workspace, entry_workspace_root) = {
            let grant = self
                .grant
                .as_mut()
                .expect("accounted consolidation retains its root grant");
            (
                grant
                    .split(0)
                    .expect("zero-byte reader workspace can always be split"),
                grant
                    .split(0)
                    .expect("zero-byte read workspace can always be split"),
                grant
                    .split(0)
                    .expect("zero-byte provider workspace can always be split"),
                grant
                    .split(0)
                    .expect("zero-byte entry workspace can always be split"),
            )
        };
        let cleanup = self.failure_cleanup.clone();
        let mut reader_workspace = PartitionWorkspace::new(reader_workspace, cleanup.as_ref());
        let mut read_workspace = PartitionWorkspace::new(read_workspace, cleanup.as_ref());
        let mut copy_provider_workspace =
            PartitionWorkspace::new(copy_provider_workspace, cleanup.as_ref());
        let mut entry_workspace_root =
            PartitionWorkspace::new(entry_workspace_root, cleanup.as_ref());
        let delta = self.partitions[partition_idx]
            .as_ref()
            .expect("accounted spilled partition retains a resident delta");
        if delta.is_empty() {
            reader_workspace.release();
            read_workspace.release();
            copy_provider_workspace.release();
            entry_workspace_root.release();
            return Ok(0);
        }
        for entry in delta.values() {
            let observed =
                (self.value_resident_capacity)(&entry.value).map_err(partition_memory_error)?;
            if observed > entry.resident_bound {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "partition value retained {observed} bytes, exceeding its declared {}-byte lifetime bound",
                        entry.resident_bound
                    ),
                )
                .into());
            }
        }

        check_cancellation(cancellation)?;
        let base_count = self.spill_base_sizes[partition_idx];
        let mut overridden = 0usize;
        {
            let file = self.spill_files[partition_idx]
                .as_ref()
                .expect("accounted consolidation requires its spilled base");
            let mut reader = open_partition_reader(
                file,
                |required| {
                    grow_workspace(
                        reader_workspace
                            .grant_mut()
                            .map_err(PartitionOperationError::into_io)?,
                        required,
                    )
                },
                cleanup.as_ref(),
            )?;
            let declared = reader.read_partition_start()?;
            if declared != u64::try_from(base_count).unwrap_or(u64::MAX) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "tracked spilled-base count disagrees with its framed declaration",
                )
                .into());
            }
            check_cancellation(cancellation)?;
            for _ in 0..declared {
                let payload =
                    read_accounted_partition_entry(&mut reader, read_workspace.grant_mut()?)?;
                let (encoded_key, _, _) = Self::partition_payload_key_and_bound(&payload)?;
                if delta
                    .keys()
                    .any(|candidate| candidate.0.as_slice() == encoded_key)
                {
                    overridden = overridden.checked_add(1).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "spilled-base override count overflow",
                        )
                    })?;
                }
                check_cancellation(cancellation)?;
            }
            reader.finish()?;
            if !reader.teardown_for_accounted_failure() {
                return Err(PartitionOperationError::NativeMapInvariant {
                    message: "partition consolidation reader cleanup failed",
                });
            }
            check_cancellation(cancellation)?;
        }

        // Every scanned payload is out of scope and reader teardown succeeded.
        // Reopen admission will cover the second scan's physical allocations.
        reader_workspace
            .grant_mut()?
            .try_resize(0)
            .map_err(partition_memory_error)?;
        read_workspace
            .grant_mut()?
            .try_resize(0)
            .map_err(partition_memory_error)?;

        let consolidated_count = base_count
            .checked_sub(overridden)
            .and_then(|count| count.checked_add(delta.len()))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "consolidated partition entry count overflow",
                )
            })?;
        if consolidated_count != self.partition_sizes[partition_idx] {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "consolidated partition contains {consolidated_count} groups, tracked {}",
                    self.partition_sizes[partition_idx]
                ),
            )
            .into());
        }

        check_cancellation(cancellation)?;
        if final_drain && overridden == base_count {
            // The complete, validated scan proved every base entry has a
            // resident override. During final drain that delta can feed the
            // resident cursor directly; an eviction still needs to write and
            // reclaim it through the ordinary consolidation below.
            let cleanup = cleanup.as_ref().ok_or_else(|| {
                native_map_invariant("covered partition retirement has no cleanup owner")
            })?;
            self.spill_files[partition_idx]
                .as_mut()
                .ok_or_else(|| native_map_invariant("covered partition retirement lost its base"))?
                .close_partition_and_delete(cleanup)?;
            // Successful deletion is the last fallible step before removing
            // the redundant base from the authoritative catalog.
            self.spill_files[partition_idx] = None;
            self.spill_base_sizes[partition_idx] = 0;
            reader_workspace.release();
            read_workspace.release();
            copy_provider_workspace.release();
            entry_workspace_root.release();
            return Ok(0);
        }
        let (file, workspace) = create_accounted_partition_file(
            &self.manager,
            entry_workspace_root.grant_mut()?,
            self.failure_cleanup.as_ref(),
        )?;
        let mut staging =
            PartitionStaging::accounted(file, workspace, self.failure_cleanup.as_ref());
        let write_result: Result<(), PartitionOperationError> = (|| {
            staging.file_mut().write_partition_start(
                u64::try_from(consolidated_count).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "consolidated partition entry count exceeds u64",
                    )
                })?,
            )?;
            check_cancellation(cancellation)?;
            {
                let file = self.spill_files[partition_idx]
                    .as_ref()
                    .expect("accounted consolidation retains its old base until publication");
                let mut reader = open_partition_reader(
                    file,
                    |required| {
                        grow_workspace(
                            reader_workspace
                                .grant_mut()
                                .map_err(PartitionOperationError::into_io)?,
                            required,
                        )
                    },
                    cleanup.as_ref(),
                )?;
                let declared = reader.read_partition_start()?;
                for _ in 0..declared {
                    let payload =
                        read_accounted_partition_entry(&mut reader, read_workspace.grant_mut()?)?;
                    if cleanup.is_some() {
                        // The successful frame read retired stored bytes and
                        // provider scratch. Only this plaintext Vec remains;
                        // persistent provider state has its reader workspace.
                        read_workspace
                            .grant_mut()?
                            .try_resize(payload.capacity())
                            .map_err(partition_memory_error)?;
                    }
                    let (encoded_key, _, _) = Self::partition_payload_key_and_bound(&payload)?;
                    if !delta
                        .keys()
                        .any(|candidate| candidate.0.as_slice() == encoded_key)
                    {
                        staging.file_mut().write_partition_entry_with_admission(
                            &payload,
                            |required| {
                                grow_workspace(
                                    copy_provider_workspace
                                        .grant_mut()
                                        .map_err(PartitionOperationError::into_io)?,
                                    required,
                                )
                            },
                        )?;
                    }
                    check_cancellation(cancellation)?;
                    if cleanup.is_some() {
                        // The copy's sealed Vec retired before write returned.
                        // Drop the plaintext before releasing its own authority.
                        drop(payload);
                        read_workspace
                            .grant_mut()?
                            .try_resize(0)
                            .map_err(partition_memory_error)?;
                        copy_provider_workspace
                            .grant_mut()?
                            .try_resize(0)
                            .map_err(partition_memory_error)?;
                    }
                }
                reader.finish()?;
                if !reader.teardown_for_accounted_failure() {
                    return Err(PartitionOperationError::NativeMapInvariant {
                        message: "partition consolidation reader cleanup failed",
                    });
                }
                check_cancellation(cancellation)?;
            }

            // Both the reader and its payloads have retired. SpillFile's
            // write_frame_after_validation also drops each local sealed Vec
            // before returning; persistent provider state remains covered by
            // the staging file workspace, not this per-record allowance.
            reader_workspace
                .grant_mut()?
                .try_resize(0)
                .map_err(partition_memory_error)?;
            read_workspace
                .grant_mut()?
                .try_resize(0)
                .map_err(partition_memory_error)?;
            copy_provider_workspace
                .grant_mut()?
                .try_resize(0)
                .map_err(partition_memory_error)?;

            for (key, entry) in delta {
                check_cancellation(cancellation)?;
                write_accounted_partition_entry(
                    staging.file_mut(),
                    key,
                    entry,
                    &*self.value_serializer,
                    self.frame_limits,
                    entry_workspace_root.grant_mut()?,
                    self.failure_cleanup.as_ref(),
                )?;
                check_cancellation(cancellation)?;
            }
            staging.finish_write()?;
            check_cancellation(cancellation)?;
            Ok(())
        })();
        if let Err(primary) = write_result {
            return Err(staging
                .cleanup_preserving_primary(primary, "partition consolidation staging cleanup"));
        }

        let bytes = usize::try_from(staging.file_mut().bytes_written()).map_err(|_| {
            std::io::Error::other("consolidated partition byte count exceeds addressable range")
        })?;
        if let Err(primary) = self.spill_files[partition_idx]
            .as_mut()
            .expect("accounted consolidation retains its old base until publication")
            .close_and_delete()
        {
            let primary = PartitionOperationError::Io(primary);
            return Err(staging
                .cleanup_preserving_primary(primary, "partition consolidation staging cleanup"));
        }

        // No fallible operation or cancellation poll may intervene between
        // deleting the old authoritative base and installing its complete
        // replacement.
        let staging = staging.into_finished_file();
        self.spill_files[partition_idx] = Some(staging);
        self.spill_base_sizes[partition_idx] = consolidated_count;
        self.partitions[partition_idx] = Some(new_partition_map());
        self.reconcile_grant().map_err(partition_memory_error)?;
        reader_workspace.release();
        read_workspace.release();
        copy_provider_workspace.release();
        entry_workspace_root.release();
        Ok(bytes)
    }

    /// Spills the largest in-memory partition.
    ///
    /// Returns the number of bytes spilled, or 0 if no partition to spill.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_largest(&mut self) -> std::io::Result<usize> {
        // Find largest in-memory partition
        let largest_idx = self
            .partitions
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| p.as_ref().map(|m| (idx, m.len())))
            .max_by_key(|(_, size)| *size)
            .map(|(idx, _)| idx);

        match largest_idx {
            Some(idx) => self.spill_partition(idx),
            None => Ok(0),
        }
    }

    pub(crate) fn spill_largest_controlled(&mut self) -> Result<usize, PartitionOperationError> {
        self.protect_operation(Self::spill_largest_controlled_inner)
    }

    fn spill_largest_controlled_inner(&mut self) -> Result<usize, PartitionOperationError> {
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        check_cancellation(cancellation.as_ref())?;
        let largest_idx = self
            .partitions
            .iter()
            .enumerate()
            .filter_map(|(idx, partition)| partition.as_ref().map(|map| (idx, map.len())))
            .max_by_key(|(_, size)| *size)
            .map(|(idx, _)| idx);
        check_cancellation(cancellation.as_ref())?;

        match largest_idx {
            Some(idx) => self.spill_partition_inner(idx, cancellation.as_ref()),
            None => Ok(0),
        }
    }

    /// Spills the least recently used in-memory partition.
    ///
    /// Returns the number of bytes spilled, or 0 if no partition to spill.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_lru(&mut self) -> std::io::Result<usize> {
        // Find LRU in-memory partition
        let lru_idx = self
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .min_by_key(|(idx, _)| self.access_times[*idx])
            .map(|(idx, _)| idx);

        match lru_idx {
            Some(idx) => self.spill_partition(idx),
            None => Ok(0),
        }
    }

    /// Inserts or updates a value for a key.
    ///
    /// # Errors
    ///
    /// Returns an error if key serialization or loading from disk fails.
    pub fn insert(&mut self, key: Vec<Value>, value: V) -> std::io::Result<Option<V>> {
        self.reject_legacy_accounted("insertion")?;
        let partition_idx = self.partition_for(&key);
        let num_key_columns = key.len();
        let serialized_key = SerializedKey::from_values(&key, self.frame_limits)?;
        let partition = self.get_partition_mut(partition_idx)?;

        let old = partition.insert(
            serialized_key,
            PartitionEntry {
                num_key_columns,
                resident_bound: 0,
                value,
            },
        );

        if old.is_none() {
            self.partition_sizes[partition_idx] += 1;
        }

        Ok(old.map(|e| e.value))
    }

    /// Gets a value for a key.
    ///
    /// # Errors
    ///
    /// Returns an error if key serialization or loading from disk fails.
    pub fn get(&mut self, key: &[Value]) -> std::io::Result<Option<&V>> {
        let partition_idx = self.partition_for(key);
        let serialized_key = SerializedKey::from_values(key, self.frame_limits)?;
        let partition = self.get_partition_mut(partition_idx)?;
        Ok(partition.get(&serialized_key).map(|e| &e.value))
    }

    /// Gets a mutable value for a key, or inserts a default.
    ///
    /// # Errors
    ///
    /// Returns an error if key serialization or loading from disk fails.
    ///
    /// # Panics
    ///
    /// Panics if the key was just inserted but cannot be found (invariant violation).
    pub fn get_or_insert_with<F>(&mut self, key: Vec<Value>, default: F) -> std::io::Result<&mut V>
    where
        F: FnOnce() -> V,
    {
        self.reject_legacy_accounted("insertion")?;
        let partition_idx = self.partition_for(&key);
        let num_key_columns = key.len();
        let serialized_key = SerializedKey::from_values(&key, self.frame_limits)?;

        let was_new;
        {
            let partition = self.get_partition_mut(partition_idx)?;
            was_new = !partition.contains_key(&serialized_key);
            if was_new {
                partition.insert(
                    serialized_key.clone(),
                    PartitionEntry {
                        num_key_columns,
                        resident_bound: 0,
                        value: default(),
                    },
                );
            }
        }
        if was_new {
            self.partition_sizes[partition_idx] += 1;
        }

        let partition = self.get_partition_mut(partition_idx)?;
        // Invariant: key was either already present or inserted in the block above
        Ok(&mut partition
            .get_mut(&serialized_key)
            .expect("key exists: just inserted or already present in partition")
            .value)
    }

    pub(crate) fn get_or_insert_with_controlled<F>(
        &mut self,
        key: Vec<Value>,
        default: F,
    ) -> Result<&mut V, PartitionOperationError>
    where
        F: FnOnce() -> V,
    {
        self.reject_legacy_accounted("insertion")?;
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        check_cancellation(cancellation.as_ref())?;
        let partition_idx = self.partition_for(&key);
        let num_key_columns = key.len();
        let serialized_key = SerializedKey::from_values(&key, self.frame_limits)?;
        check_cancellation(cancellation.as_ref())?;

        let was_new = {
            let partition = self.get_partition_mut_controlled(partition_idx)?;
            !partition.contains_key(&serialized_key)
        };
        if was_new {
            if self.grant.is_some() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "accounted partition insertion requires an explicit value-capacity bound",
                )
                .into());
            }
            // Compatibility insertion is one standalone operation, so it
            // deliberately starts with a fresh retry budget.
            let mut retry_budget = OneSpillRetryBudget::default();
            self.reserve_new_entry_with_one_spill(
                partition_idx,
                serialized_key.0.capacity(),
                cancellation.as_ref(),
                &mut retry_budget,
            )?;
            self.partitions[partition_idx]
                .as_mut()
                .expect("accounted insertion keeps target partition resident")
                .insert(
                    serialized_key.clone(),
                    PartitionEntry {
                        num_key_columns,
                        resident_bound: 0,
                        value: default(),
                    },
                );
            self.reconcile_grant().map_err(partition_memory_error)?;
        }
        if was_new {
            self.partition_sizes[partition_idx] += 1;
        }

        // Do not poll after insertion: the caller owns the rest of this
        // aggregate-row mutation and must update all accumulators before its
        // next cancellation boundary.
        Ok(&mut self.partitions[partition_idx]
            .as_mut()
            .expect("partition is in memory after controlled lookup")
            .get_mut(&serialized_key)
            .expect("key exists after controlled insertion or lookup")
            .value)
    }

    /// Builds and atomically replaces one resident accounted value without
    /// exposing mutable access to the authoritative entry.
    ///
    /// A spilled target remains an immutable on-disk base plus a resident
    /// override delta. Only the matching base record is decoded, and a complete
    /// replacement is published into the delta at the final boundary. One
    /// shared victim budget spans key staging, bounded base reader/frame/decode
    /// workspaces, candidate construction, and delta-map admission. A frame
    /// denial after a matching record has run the decoder is terminal, so
    /// retry cannot replay a custom callback.
    /// Reader destruction is backstopped during unwind, but construction/open/
    /// read provider panics and heap-bearing error payloads remain an audited
    /// provider trust contract rather than a fully mediated boundary.
    /// The declaration callback is a trust boundary: its construction peak
    /// must cover the complete builder allocation graph, including any error
    /// or panic payload that escapes. The method-specific sealed value bound
    /// separately ensures that lending the live old value cannot mutate it.
    /// Declaration itself runs before construction authority exists and must
    /// therefore perform non-allocating inspection/arithmetic; returning or
    /// panicking with a heap-bearing declaration payload violates the contract.
    /// Builder success and retained-capacity inspection share one unwind
    /// boundary: a capacity panic destroys the complete candidate while the
    /// construction grant is live, then escapes only through the grant-owning
    /// [`AccountedPartitionPanic`]. A returned capacity error likewise retains
    /// the complete construction grant until the caller drops that error.
    #[allow(
        dead_code,
        reason = "production aggregate adoption follows the resident replacement protocol"
    )]
    pub(crate) fn try_replace_accounted<E, D, B>(
        &mut self,
        key: Vec<Value>,
        declare: D,
        build: B,
    ) -> Result<(), PartitionUpdateError<E>>
    where
        V: SharedImmutablePartitionValue,
        D: FnOnce(Option<&V>) -> Result<PartitionUpdateAdmission, E>,
        B: FnOnce(Option<&V>) -> Result<V, E>,
    {
        if self.grant.is_none() {
            return Err(PartitionOperationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "accounted partition replacement requires a resident-memory grant",
            ))
            .into());
        }
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;
        self.ensure_not_draining()
            .map_err(PartitionOperationError::from)?;

        let partition_idx = self.partition_for(&key);
        // A saturated logical catalog cannot represent another addressable
        // entry and is therefore fenced as invalid before key serialization.
        // This intentionally rejects even a putative existing-key update: at
        // `usize::MAX` the catalog cannot be a valid in-process partition.
        let saturated_next_partition_size = self.partition_sizes[partition_idx]
            .checked_add(1)
            .ok_or_else(|| {
                PartitionOperationError::Io(std::io::Error::other("partition entry count overflow"))
            })?;
        let has_spilled_base = self.spill_files[partition_idx].is_some();
        let num_key_columns = key.len();
        let mut retry_budget = OneSpillRetryBudget::default();
        let serialized_key = self
            .stage_accounted_key_with_one_spill(
                &key,
                partition_idx,
                cancellation.as_ref(),
                &mut retry_budget,
            )
            .map_err(PartitionUpdateError::from)?;
        self.touch(partition_idx);

        let was_present = self.partitions[partition_idx]
            .as_ref()
            .expect("resident replacement retains its target partition")
            .contains_key(serialized_key.serialized());
        let spilled_base = if has_spilled_base && !was_present {
            self.lookup_spilled_base_entry_accounted(
                partition_idx,
                serialized_key.serialized(),
                num_key_columns,
                cancellation.as_ref(),
                &mut retry_budget,
            )
            .map_err(PartitionUpdateError::from)?
        } else {
            None
        };
        let publication_num_key_columns = if let Some(base) = spilled_base.as_ref() {
            let stored = base.entry().num_key_columns;
            if stored != num_key_columns {
                return Err(PartitionOperationError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "spilled-base key column count disagrees with the lookup key",
                ))
                .into());
            }
            stored
        } else {
            num_key_columns
        };
        let logical_present = was_present || spilled_base.is_some();
        // Validate the authoritative count before growing either the root
        // grant or the replacement candidate. A synthetic overflow therefore
        // cannot leave any resident/accounting state changed.
        let next_partition_size = if logical_present {
            None
        } else {
            Some(saturated_next_partition_size)
        };
        let old_retained_bytes = {
            let old = self.partitions[partition_idx]
                .as_ref()
                .expect("resident replacement retains its target partition")
                .get(serialized_key.serialized())
                .map(|entry| &entry.value);
            old.map_or(Ok(0), |value| (self.value_resident_capacity)(value))
                .map_err(partition_memory_error)?
        };
        let admission = {
            let old = self.partitions[partition_idx]
                .as_ref()
                .expect("resident replacement retains its target partition")
                .get(serialized_key.serialized())
                .map(|entry| &entry.value);
            let old = old.or_else(|| spilled_base.as_ref().map(|base| &base.entry().value));
            declare(old).map_err(|error| {
                PartitionUpdateError::Caller(AccountedCallerFailure::from_declaration(error))
            })?
        };
        if admission.construction_peak < admission.retained_upper_bound {
            return Err(PartitionUpdateError::InvalidAdmission {
                retained_upper_bound: admission.retained_upper_bound,
                construction_peak: admission.construction_peak,
            });
        }
        check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;

        let mut construction_grant = self
            .grant
            .as_mut()
            .expect("accounted replacement retains its root grant")
            .split(0)
            .expect("zero-byte construction grant can always be split");
        self.resize_transient_grant_with_one_spill(
            &mut construction_grant,
            admission.construction_peak,
            Some(partition_idx),
            cancellation.as_ref(),
            &mut retry_budget,
        )?;
        check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;

        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let old = self.partitions[partition_idx]
                .as_ref()
                .expect("resident replacement retains its target partition")
                .get(serialized_key.serialized())
                .map(|entry| &entry.value);
            let old = old.or_else(|| spilled_base.as_ref().map(|base| &base.entry().value));
            let value = build(old).map_err(PartitionValueConstructionError::Builder)?;
            let observed = (self.value_resident_capacity)(&value)
                .map_err(PartitionValueConstructionError::Capacity)?;
            Ok::<_, PartitionValueConstructionError<E>>((value, observed))
        }));
        let (value, observed_retained) = match built {
            Ok(Ok(built)) => built,
            Ok(Err(PartitionValueConstructionError::Capacity(error))) => {
                return Err(PartitionUpdateError::from(accounted_partition_io_failure(
                    grant_io_error(error),
                    construction_grant,
                )));
            }
            Ok(Err(PartitionValueConstructionError::Builder(error))) => {
                return Err(PartitionUpdateError::Caller(
                    AccountedCallerFailure::from_builder(error, construction_grant),
                ));
            }
            Err(panic) => {
                let accounted_panic = AccountedPartitionPanic::new(panic, construction_grant);
                drop(serialized_key);
                std::panic::resume_unwind(Box::new(accounted_panic));
            }
        };
        if observed_retained > admission.retained_upper_bound {
            return Err(PartitionUpdateError::RetainedCapacityExceeded {
                retained_upper_bound: admission.retained_upper_bound,
                observed_retained,
            });
        }
        construction_grant
            .try_resize(observed_retained)
            .map_err(partition_memory_error)?;
        let candidate = AccountedPartitionCandidate {
            value: Some(value),
            grant: Some(construction_grant),
            retained_bytes: observed_retained,
        };

        if !was_present {
            // Both the serialized key and complete candidate already retain
            // child grants. Pending zero admits only resident map growth and
            // may reclaim one cold, non-target partition on denial.
            self.reserve_new_entry_with_one_spill(
                partition_idx,
                0,
                cancellation.as_ref(),
                &mut retry_budget,
            )?;
        }
        // This is the last cooperative boundary: publication and its authority
        // transfer are one indivisible operation from the caller's perspective.
        check_cancellation(cancellation.as_ref()).map_err(PartitionOperationError::from)?;
        if was_present {
            let slot = self.partitions[partition_idx]
                .as_mut()
                .expect("resident replacement retains its target partition")
                .get_mut(serialized_key.serialized())
                .expect("immutable prepublication lookup found this resident key");
            drop(serialized_key);
            let publication = ResidentReplacementPublication::begin(
                self.grant
                    .as_mut()
                    .expect("accounted replacement retains its root grant"),
                old_retained_bytes,
                candidate,
            )?;
            publication.publish(slot, admission.retained_upper_bound);
        } else {
            let partition = self.partitions[partition_idx]
                .as_mut()
                .expect("absent publication keeps its target partition resident");
            let publication = AbsentReplacementPublication::new(
                self.grant
                    .as_mut()
                    .expect("accounted replacement retains its root grant"),
                serialized_key,
                candidate,
            );
            publication
                .publish(
                    partition,
                    publication_num_key_columns,
                    admission.retained_upper_bound,
                )
                .map_err(PartitionOperationError::from)?;
            if let Some(next_partition_size) = next_partition_size {
                self.partition_sizes[partition_idx] = next_partition_size;
            }
        }
        Ok(())
    }

    /// Gets an accounted value or inserts one after pre-admitting its nested
    /// resident-capacity bound.
    ///
    /// The bound covers heap allocations retained below `V`; the partition
    /// separately charges key bytes and a conservative raw hash-bucket bound.
    /// A newly-built value whose observed retained capacity exceeds the bound
    /// is dropped before publication and returns an error. No cancellation
    /// poll occurs after publication, preserving one aggregate row as the
    /// caller's atomic mutation unit. One shared victim budget spans key
    /// staging, base reader/frame/decode workspaces, the legacy stored-value
    /// bound, and map admission. A later frame denial is terminal once the
    /// custom decoder has run. Default construction and retained-capacity
    /// inspection use a dedicated child grant; either panic escapes only in a
    /// non-detachable [`AccountedPartitionPanic`] that keeps this authority.
    /// A returned capacity error keeps the same child until the error drops.
    /// On success that child reconciles to the observed retained bytes before
    /// publication. The unrestricted mutable reference returned by this
    /// legacy seam remains outside the lifetime-qualified guarantee; mediated
    /// replacement is the qualified mutation API.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, spill I/O, allocation, capacity-contract,
    /// or resident-grant failures without publishing a partial entry.
    #[allow(
        dead_code,
        reason = "P3b2 replaces this internal construction seam with mediated updates"
    )]
    pub(crate) fn get_or_insert_with_accounted<F>(
        &mut self,
        key: Vec<Value>,
        default_resident_bound: usize,
        default: F,
    ) -> Result<&mut V, PartitionOperationError>
    where
        F: FnOnce() -> V,
    {
        if self.grant.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "accounted partition insertion requires a resident-memory grant",
            )
            .into());
        }
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        check_cancellation(cancellation.as_ref())?;
        self.ensure_not_draining()?;
        let partition_idx = self.partition_for(&key);
        let num_key_columns = key.len();
        // Saturation fences the catalog before serialization. Even a putative
        // existing key is rejected because a `usize::MAX` logical count cannot
        // describe a valid addressable in-process partition.
        let next_partition_size = self.partition_sizes[partition_idx]
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("partition entry count overflow"))?;
        let mut retry_budget = OneSpillRetryBudget::default();
        let serialized_key = self.stage_accounted_key_with_one_spill(
            &key,
            partition_idx,
            cancellation.as_ref(),
            &mut retry_budget,
        )?;

        self.touch(partition_idx);
        if self.partitions[partition_idx]
            .as_ref()
            .expect("accounted partitions always retain a resident delta")
            .contains_key(serialized_key.serialized())
        {
            return Ok(&mut self.partitions[partition_idx]
                .as_mut()
                .expect("accounted lookup leaves the partition resident")
                .get_mut(serialized_key.serialized())
                .expect("accounted lookup found the existing key")
                .value);
        }

        // A qualified spilled partition is an immutable on-disk base plus a
        // small resident override delta. Revisit only the matching record;
        // never invoke the compatibility whole-partition reload path.
        if self.spill_files[partition_idx].is_some()
            && let Some(mut base_entry) = self.lookup_spilled_base_entry_accounted(
                partition_idx,
                serialized_key.serialized(),
                num_key_columns,
                cancellation.as_ref(),
                &mut retry_budget,
            )?
        {
            // Legacy callers receive mutable access up to the persisted bound,
            // so the lookup child must cover that whole bound before the
            // decoded entry can become resident authority.
            let retained = base_entry
                .stored_value_bound_bytes()
                .map_err(partition_memory_error)?;
            // Retry only the child-grant resize. Replaying lookup or either
            // custom capacity callback could duplicate observable caller work.
            self.resize_transient_grant_with_one_spill(
                base_entry.grant_mut(),
                retained,
                Some(partition_idx),
                cancellation.as_ref(),
                &mut retry_budget,
            )?;
            // Decoded key/value authority already belongs to the lookup child;
            // this admission covers only the resident delta map.
            self.reserve_new_entry_with_one_spill(
                partition_idx,
                0,
                cancellation.as_ref(),
                &mut retry_budget,
            )?;
            let partition = self.partitions[partition_idx]
                .as_mut()
                .expect("accounted spilled partition retains its delta");
            let publication = AccountedBaseDeltaPublication::new(
                base_entry,
                self.grant
                    .as_mut()
                    .expect("accounted base publication retains its root grant"),
            );
            publication.publish_into(partition)?;
            return Ok(&mut self.partitions[partition_idx]
                .as_mut()
                .expect("accounted spilled partition retains its delta")
                .get_mut(serialized_key.serialized())
                .expect("matching spilled-base entry was installed in the delta")
                .value);
        }

        // The staged key's child grant is already live, so this helper must
        // exclude it rather than double-charge it. Reserve the physical map
        // before invoking either custom construction callback.
        self.reserve_new_entry_with_one_spill(
            partition_idx,
            0,
            cancellation.as_ref(),
            &mut retry_budget,
        )?;

        // Keep construction authority in a dedicated child rather than the
        // resident root. A caught callback panic can then move this child into
        // the non-detachable panic owner without reconciling or releasing any
        // bytes that may cover its heap-bearing payload.
        let mut construction_grant = self
            .grant
            .as_mut()
            .expect("accounted insertion retains its root grant")
            .split(0)
            .expect("zero-byte construction grant can always be split");
        self.resize_transient_grant_with_one_spill(
            &mut construction_grant,
            default_resident_bound,
            Some(partition_idx),
            cancellation.as_ref(),
            &mut retry_budget,
        )?;

        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let value = default();
            let observed = (self.value_resident_capacity)(&value)?;
            Ok::<_, MemoryGrantError>((value, observed))
        }));
        let (value, observed_value) = match built {
            Ok(Ok(built)) => built,
            Ok(Err(primary)) => {
                return Err(accounted_partition_io_failure(
                    grant_io_error(primary),
                    construction_grant,
                ));
            }
            Err(panic) => {
                let accounted_panic = AccountedPartitionPanic::new(panic, construction_grant);
                drop(serialized_key);
                std::panic::resume_unwind(Box::new(accounted_panic));
            }
        };
        if observed_value > default_resident_bound {
            drop(value);
            drop(construction_grant);
            let primary = std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "partition value retained {observed_value} bytes, exceeding its declared {default_resident_bound}-byte bound"
                ),
            );
            return Err(primary.into());
        }
        if let Err(primary) = construction_grant.try_resize(observed_value) {
            drop(value);
            drop(construction_grant);
            return Err(partition_memory_error(primary));
        }
        if let Err(construction_grant) = self
            .grant
            .as_mut()
            .expect("accounted insertion retains its root grant")
            .try_merge(construction_grant)
        {
            drop(value);
            drop(construction_grant);
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "partition construction grant is incompatible with its resident root",
            )
            .into());
        }

        let partition = self.partitions[partition_idx]
            .as_mut()
            .expect("accounted insertion keeps its target partition resident");
        let publication = AccountedKeyPublication::new(
            serialized_key,
            self.grant
                .as_mut()
                .expect("accounted insertion retains its root grant"),
        );
        let published = publication.publish_into(
            partition,
            PartitionEntry {
                num_key_columns,
                resident_bound: default_resident_bound,
                value,
            },
            || {},
        )?;
        self.partition_sizes[partition_idx] = next_partition_size;
        Ok(&mut published.value)
    }

    /// Drains all entries from all partitions.
    ///
    /// Loads spilled partitions as needed.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from disk fails.
    pub fn drain_all(&mut self) -> std::io::Result<Vec<(Vec<Value>, V)>> {
        self.reject_legacy_accounted("partition cursor")?;
        self.drain_all_inner(None)
            .map_err(PartitionOperationError::into_io)
    }

    pub(crate) fn drain_all_controlled(
        &mut self,
    ) -> Result<Vec<(Vec<Value>, V)>, PartitionOperationError> {
        self.reject_legacy_accounted("partition cursor")?;
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        self.drain_all_inner(cancellation.as_ref())
    }

    /// Begins a consuming, resource-qualified partition drain.
    ///
    /// Before the consuming boundary, a partially overridden spill base and
    /// its resident delta become one complete immutable partition file. A fully
    /// overridden base retires after validation and leaves its delta resident.
    /// Cancellation during preparation preserves logically authoritative state.
    /// Once this returns a cursor,
    /// cancellation or any concrete error consumes the operation and triggers
    /// explicit cleanup; abandoning the cursor invokes the same cleanup as a
    /// `Drop` backstop. The cursor decodes at most one partition record at a
    /// time and never calls the compatibility `drain_all`/whole-reload path.
    ///
    /// # Errors
    ///
    /// Returns structured capacity, spill, codec, or cancellation failures.
    pub fn drain_partitioned_accounted(
        &mut self,
    ) -> Result<PartitionDrainCursor<'_, V>, PartitionOperationError> {
        if self.failure_cleanup.is_some() && self.failure_publisher.is_none() {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "failed accounted partition cannot be drained again",
            });
        }
        drop(self.recovery_allowance.take());
        if self.failure_publisher.is_some() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.prepare_accounted_drain()
            })) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(self.publish_failure(Some(error), None, None)),
                Err(payload) => return Err(self.publish_failure(None, None, Some(payload))),
            }
        } else {
            self.prepare_accounted_drain()?;
        }
        Ok(PartitionDrainCursor {
            state: self,
            next_partition: 0,
            active_partition: None,
            reader: None,
            reader_grant: None,
            resident: None,
            remaining: 0,
            finished: false,
        })
    }

    fn prepare_accounted_drain(&mut self) -> Result<(), PartitionOperationError> {
        if self.grant.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "accounted partition drain requires a resident-memory grant",
            )
            .into());
        }
        let cancellation = self.cancellation.clone();
        debug_assert!(cancellation.is_some());
        check_cancellation(cancellation.as_ref())?;
        self.ensure_not_draining()?;

        // Preparation may change physical representation but never consumes a
        // group: each published file remains owned by this state on failure.
        // Preserve the legacy cursor's spill preparation; the live aggregate
        // moves partitions without a base directly into its resident cursor.
        for partition_idx in 0..self.num_partitions {
            if (self.failure_cleanup.is_none() || self.spill_files[partition_idx].is_some())
                && self.partitions[partition_idx]
                    .as_ref()
                    .is_some_and(|partition| !partition.is_empty())
            {
                if self.failure_cleanup.is_some() {
                    check_cancellation(cancellation.as_ref())?;
                    self.consolidate_accounted_partition(
                        partition_idx,
                        cancellation.as_ref(),
                        true,
                    )?;
                } else {
                    self.spill_partition_inner(partition_idx, cancellation.as_ref())?;
                }
            }
        }
        check_cancellation(cancellation.as_ref())?;
        self.drain_state = DrainState::Draining;
        Ok(())
    }

    fn drain_all_inner(
        &mut self,
        cancellation: Option<&QueryCancellationToken>,
    ) -> Result<Vec<(Vec<Value>, V)>, PartitionOperationError> {
        // This checkpoint is before the consuming boundary. A pre-cancelled
        // aggregate therefore retains all state and can still run explicit
        // cleanup through its ordinary owner/drop path.
        check_cancellation(cancellation)?;
        self.ensure_not_draining()?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.total_size())
            .map_err(|error| {
                std::io::Error::other(format!(
                    "failed to reserve drained partition output: {error}"
                ))
            })?;
        check_cancellation(cancellation)?;
        // From this point cancellation is terminal and consuming. It drops
        // ephemeral output and explicitly cleans every owned partition.
        self.drain_state = DrainState::Draining;
        let frame_limits = self.frame_limits;
        let mut destructive_progress = false;

        let drain_result: Result<(), PartitionOperationError> = (|| {
            for partition_idx in 0..self.num_partitions {
                check_cancellation(cancellation)?;
                let partition =
                    self.get_partition_mut_unchecked_inner(partition_idx, cancellation)?;
                check_cancellation(cancellation)?;
                // Constructing (and then dropping) a non-empty `HashMap::drain`
                // removes every remaining entry, even if key decoding fails on
                // the first item. Record progress before creating that iterator.
                destructive_progress |= !partition.is_empty();
                for (serialized_key, entry) in partition.drain() {
                    let key = serialized_key.to_values(entry.num_key_columns, frame_limits)?;
                    result.push((key, entry.value));
                    check_cancellation(cancellation)?;
                }
                self.partition_sizes[partition_idx] = 0;
                check_cancellation(cancellation)?;
            }
            Ok(())
        })();
        match drain_result {
            Ok(()) => {
                self.drain_state = DrainState::Idle;
                Ok(result)
            }
            Err(error @ PartitionOperationError::Cancelled(_))
            | Err(error @ PartitionOperationError::CancelledWithCleanup { .. }) => {
                self.drain_state = DrainState::Poisoned;
                match self.cleanup() {
                    Ok(()) => Err(error),
                    Err(cleanup) => Err(with_cleanup(
                        error,
                        cleanup,
                        "cancelled partition drain cleanup",
                    )),
                }
            }
            Err(PartitionOperationError::Allocation { .. })
            | Err(PartitionOperationError::NativeMapAllocation { .. })
            | Err(PartitionOperationError::NativeMapAllocationWithRollback { .. })
            | Err(PartitionOperationError::NativeMapInvariant { .. })
            | Err(PartitionOperationError::NativeMapInvariantWithRollback { .. }) => unreachable!(
                "container reservation fails before PartitionedState publication and cannot reach a drain"
            ),
            Err(error) => {
                self.drain_state = if destructive_progress {
                    DrainState::Poisoned
                } else {
                    DrainState::Idle
                };
                Err(error)
            }
        }
    }

    /// Iterates over all entries without draining.
    ///
    /// Loads spilled partitions as needed.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from disk fails.
    pub fn iter_all(&mut self) -> std::io::Result<Vec<(Vec<Value>, V)>> {
        self.reject_legacy_accounted("partition cursor")?;
        let mut result = Vec::with_capacity(self.total_size());
        let frame_limits = self.frame_limits;

        for partition_idx in 0..self.num_partitions {
            let partition = self.get_partition_mut(partition_idx)?;
            for (serialized_key, entry) in partition.iter() {
                let key = serialized_key.to_values(entry.num_key_columns, frame_limits)?;
                result.push((key, entry.value.clone()));
            }
        }

        Ok(result)
    }

    /// Explicitly deletes all spill files and resets state.
    ///
    /// Failed handles remain installed for retry and prevent a false cleanup
    /// success result.
    ///
    /// # Errors
    ///
    /// Returns the first owned deletion error without formatting it.
    pub fn cleanup(&mut self) -> std::io::Result<()> {
        let mut first_error = None;
        let mut destructive_progress = false;
        for slot in &mut self.spill_files {
            let result = {
                let Some(file) = slot.as_mut() else {
                    continue;
                };
                file.close_and_delete()
            };
            match result {
                Ok(()) => {
                    *slot = None;
                    destructive_progress = true;
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    } else {
                        // A later opaque failure must not replace the first by
                        // formatting, dropping, or panicking during disposal.
                        let _ = super::run_cleanup_backstop(|| Err::<(), _>(error));
                    }
                }
            }
        }
        if let Some(error) = first_error {
            if destructive_progress {
                self.drain_state = DrainState::Poisoned;
            }
            return Err(error);
        }

        self.spill_base_sizes.fill(0);
        for partition in &mut self.partitions {
            *partition = Some(new_partition_map());
        }
        self.partition_sizes.fill(0);
        self.access_times.fill(0);
        self.timestamp = 0;
        self.drain_state = DrainState::Idle;
        self.reconcile_grant()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::OutOfMemory, error))?;
        Ok(())
    }
}

impl<V: Clone + Send + Sync + 'static> PartitionDrainCursor<'_, V> {
    /// Returns one while a single partition reader is active, otherwise zero.
    /// No second partition is opened until the active reader is finished and
    /// its file has been explicitly deleted.
    #[must_use]
    pub const fn active_partition_count(&self) -> usize {
        if self.active_partition.is_some() {
            1
        } else {
            0
        }
    }

    /// Advances the consuming cursor by one grant-owned group.
    ///
    /// # Errors
    ///
    /// A concrete read/decode/delete failure wins over cancellation observed
    /// during that same callback. Any terminal failure cleans remaining state;
    /// cleanup failure is retained as secondary context.
    pub fn next_entry(
        &mut self,
    ) -> Result<Option<PartitionDrainEntry<V>>, PartitionOperationError> {
        if self.finished {
            return Ok(None);
        }
        if self.state.failure_publisher.is_some() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.next_entry_inner()))
            {
                Ok(Ok(entry)) => Ok(entry),
                Ok(Err(error)) => Err(self.fail_terminal(error)),
                Err(payload) => {
                    self.retire_readers();
                    self.finished = true;
                    Err(self.state.publish_failure(None, None, Some(payload)))
                }
            }
        } else {
            match self.next_entry_inner() {
                Ok(entry) => Ok(entry),
                Err(error) => Err(self.fail_terminal(error)),
            }
        }
    }

    fn next_entry_inner(
        &mut self,
    ) -> Result<Option<PartitionDrainEntry<V>>, PartitionOperationError> {
        if self.finished {
            return Ok(None);
        }
        let cancellation = self.state.cancellation.clone();

        loop {
            if let Some(resident) = self.resident.as_mut() {
                check_cancellation(cancellation.as_ref())?;
                if let Some(entry) = resident.next(self.state)? {
                    return Ok(Some(entry));
                }
                self.resident = None;
                self.active_partition = None;
            }
            if self.reader.is_some() && self.remaining == 0 {
                let mut reader = self
                    .reader
                    .take()
                    .expect("active zero-remaining partition has its reader");
                reader.finish()?;
                if !reader.teardown_for_accounted_failure() {
                    return Err(PartitionOperationError::NativeMapInvariant {
                        message: "partition reader physical cleanup failed",
                    });
                }
                if let Some(grant) = self.reader_grant.take() {
                    grant.release();
                }
                check_cancellation(cancellation.as_ref())?;
                let partition_idx = self
                    .active_partition
                    .take()
                    .expect("active reader retains its partition index");
                self.state.spill_files[partition_idx]
                    .as_mut()
                    .expect("active reader retains its spill-file owner")
                    .close_and_delete()?;
                // File deletion is the last fallible step before the consumed
                // partition is removed from the authoritative catalog.
                self.state.spill_files[partition_idx] = None;
                self.state.spill_base_sizes[partition_idx] = 0;
                self.state.partition_sizes[partition_idx] = 0;
                check_cancellation(cancellation.as_ref())?;
            }

            if let Some(reader) = self.reader.as_mut() {
                check_cancellation(cancellation.as_ref())?;
                let mut frame_workspace = PartitionWorkspace::new(
                    self.state.split_workspace_grant(0)?,
                    self.state.failure_cleanup.as_ref(),
                );
                let payload = read_accounted_partition_entry(reader, frame_workspace.grant_mut()?)?;
                // Poll only after the concrete read succeeded, so an I/O error
                // raised by a callback that also cancels remains primary.
                check_cancellation(cancellation.as_ref())?;
                let (encoded_key, _, resident_bound) =
                    PartitionedState::<V>::partition_payload_key_and_bound(&payload)?;
                let decoded_key_bound =
                    PartitionedState::<V>::decoded_key_resident_bound(encoded_key.len())
                        .map_err(partition_memory_error)?;
                let retained_bound =
                    decoded_key_bound
                        .checked_add(resident_bound)
                        .ok_or_else(|| {
                            partition_memory_error(MemoryGrantError::ArithmeticOverflow {
                                current_bytes: decoded_key_bound,
                                additional_bytes: resident_bound,
                            })
                        })?;
                let decode_peak =
                    retained_bound
                        .checked_add(encoded_key.len())
                        .ok_or_else(|| {
                            partition_memory_error(MemoryGrantError::ArithmeticOverflow {
                                current_bytes: retained_bound,
                                additional_bytes: encoded_key.len(),
                            })
                        })?;
                let mut workspace = PartitionWorkspace::new(
                    self.state.split_workspace_grant(decode_peak)?,
                    self.state.failure_cleanup.as_ref(),
                );
                let (serialized_key, num_key_columns, decoded_bound, value) = self
                    .state
                    .decode_partition_entry_without_key_validation(&payload)?;
                // A custom decoder may cooperatively cancel and still return a
                // complete value. Observe that only after its concrete result,
                // so a decoder error remains primary and no cancelled row can
                // escape the cursor.
                check_cancellation(cancellation.as_ref())?;
                if decoded_bound != resident_bound {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "partition resident bound changed during cursor decode",
                    )
                    .into());
                }
                let observed =
                    (self.state.value_resident_capacity)(&value).map_err(partition_memory_error)?;
                if observed > resident_bound {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "decoded partition value retained {observed} bytes, exceeding its stored {resident_bound}-byte bound"
                        ),
                    )
                    .into());
                }
                let key = serialized_key.to_values(num_key_columns, self.state.frame_limits)?;
                // Key decoding can itself be a bounded unit of work. A
                // concrete key error wins through `?`; successful decoding is
                // followed by the final cancellation boundary before publish.
                check_cancellation(cancellation.as_ref())?;
                drop(serialized_key);
                drop(payload);
                frame_workspace.release();
                let key_grant = workspace
                    .grant_mut()?
                    .split(decoded_key_bound)
                    .expect("decoded key bound is part of the admitted peak");
                let value_grant = workspace
                    .grant_mut()?
                    .split(resident_bound)
                    .expect("resident value bound is part of the admitted peak");
                workspace.release();
                self.remaining -= 1;
                return Ok(Some(PartitionDrainEntry {
                    key: AccountedPartitionKey::new(key, key_grant),
                    value: AccountedPartitionValue::new(value, value_grant),
                }));
            }

            if self.next_partition == self.state.num_partitions {
                self.state.drain_state = DrainState::Idle;
                self.state
                    .reconcile_grant()
                    .map_err(partition_memory_error)?;
                self.finished = true;
                return Ok(None);
            }

            let partition_idx = self.next_partition;
            self.next_partition += 1;
            if self.state.spill_files[partition_idx].is_none() {
                if self.state.partition_sizes[partition_idx] != 0 {
                    self.resident = Some(resident_cursor::ResidentPartition::begin(
                        self.state,
                        partition_idx,
                    ));
                    self.active_partition = Some(partition_idx);
                }
                continue;
            }

            check_cancellation(cancellation.as_ref())?;
            let mut reader_grant = PartitionWorkspace::new(
                self.state.split_workspace_grant(0)?,
                self.state.failure_cleanup.as_ref(),
            );
            let mut reader = open_partition_reader(
                self.state.spill_files[partition_idx].as_ref().ok_or(
                    PartitionOperationError::NativeMapInvariant {
                        message: "accounted cursor lost its spill-file owner",
                    },
                )?,
                |required| {
                    grow_workspace(
                        reader_grant
                            .grant_mut()
                            .map_err(PartitionOperationError::into_io)?,
                        required,
                    )
                },
                self.state.failure_cleanup.as_ref(),
            )?;
            let declared = reader.read_partition_start()?;
            if declared
                != u64::try_from(self.state.spill_base_sizes[partition_idx]).unwrap_or(u64::MAX)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "accounted cursor partition count disagrees with its catalog",
                )
                .into());
            }
            check_cancellation(cancellation.as_ref())?;
            self.remaining = declared;
            self.active_partition = Some(partition_idx);
            self.reader = Some(reader);
            self.reader_grant = Some(reader_grant);
        }
    }

    fn retire_readers(&mut self) {
        let mut complete = true;
        if let Some(reader) = self.reader.take() {
            complete &= reader.teardown_for_accounted_failure();
        }
        if let Some(resident) = self.resident.take() {
            complete &= resident.destroy();
        }
        if !complete && let Some(cleanup) = &self.state.failure_cleanup {
            cleanup.inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_failed);
        }
        self.reader_grant = None;
        self.active_partition = None;
    }

    pub(crate) fn fail_operator(&mut self, error: OperatorError) -> PartitionOperationError {
        self.retire_readers();
        self.finished = true;
        self.state.fail_operator(error)
    }

    pub(crate) fn fail_panic(
        &mut self,
        payload: Box<dyn std::any::Any + Send>,
        grant: MemoryGrant,
    ) -> PartitionOperationError {
        drop(PartitionWorkspace::new(
            grant,
            self.state.failure_cleanup.as_ref(),
        ));
        self.retire_readers();
        self.finished = true;
        self.state.publish_failure(None, None, Some(payload))
    }

    fn fail_terminal(&mut self, primary: PartitionOperationError) -> PartitionOperationError {
        self.retire_readers();
        self.finished = true;
        if self.state.failure_publisher.is_some() {
            return self.state.publish_failure(Some(primary), None, None);
        }
        self.state.drain_state = DrainState::Poisoned;
        match self.state.cleanup() {
            Ok(()) => primary,
            Err(cleanup) => with_cleanup(primary, cleanup, "accounted partition drain cleanup"),
        }
    }
}

impl<V: Clone + Send + Sync + 'static> Drop for PartitionDrainCursor<'_, V> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.retire_readers();
        self.state.drain_state = DrainState::Poisoned;
        let _ = super::run_cleanup_backstop(|| self.state.cleanup());
    }
}

impl<V> Drop for PartitionedState<V> {
    fn drop(&mut self) {
        if let Some(cleanup) = &self.failure_cleanup {
            let mut complete = true;
            for partition in &mut self.partitions {
                if let Some(partition) = partition.take() {
                    let mut entries = partition.into_iter();
                    for entry in entries.by_ref() {
                        complete &= super::run_cleanup_backstop(|| {
                            drop(entry);
                            Ok::<_, ()>(())
                        });
                    }
                    drop(entries);
                }
            }
            for file in &mut self.spill_files {
                if let Some(file) = file.take() {
                    complete &= super::run_cleanup_backstop(|| {
                        if file.retire_owned_file() {
                            Ok(())
                        } else {
                            Err(())
                        }
                    });
                }
            }
            if !complete {
                cleanup.inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_failed);
                if let Some(grant) = self.grant.take() {
                    cleanup.inspect::<PartitionFailureCleanup, _>(|witness| witness.retain(grant));
                }
            }
            return;
        }
        let stranded = u64::try_from(self.spill_files.iter().flatten().count()).unwrap_or(u64::MAX);
        let succeeded = super::run_cleanup_backstop(|| {
            let mut failed = false;
            for file in self.spill_files.iter_mut().flatten() {
                if !super::run_cleanup_backstop(|| file.close_and_delete()) {
                    failed = true;
                }
            }
            if failed { Err(()) } else { Ok(()) }
        });
        if !succeeded {
            super::manager::record_orphan_cleanup_failures(stranded);
        }
    }
}

/// Hashes a key (vector of values) to a u64.
fn hash_key(key: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for value in key {
        match value {
            Value::Null => 0u8.hash(&mut hasher),
            Value::Bool(b) => {
                1u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Value::Int64(n) => {
                2u8.hash(&mut hasher);
                n.hash(&mut hasher);
            }
            Value::Float64(f) => {
                3u8.hash(&mut hasher);
                // Canonicalize -0.0 to +0.0 so the two zeros co-partition.
                grafeo_common::types::canonical_f64_bits(*f).hash(&mut hasher);
            }
            Value::String(s) => {
                4u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            Value::Bytes(b) => {
                5u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Value::Timestamp(t) => {
                6u8.hash(&mut hasher);
                t.hash(&mut hasher);
            }
            Value::Date(d) => {
                10u8.hash(&mut hasher);
                d.hash(&mut hasher);
            }
            Value::Time(t) => {
                11u8.hash(&mut hasher);
                t.hash(&mut hasher);
            }
            Value::Duration(d) => {
                12u8.hash(&mut hasher);
                d.hash(&mut hasher);
            }
            Value::ZonedDatetime(zdt) => {
                14u8.hash(&mut hasher);
                zdt.hash(&mut hasher);
            }
            Value::List(l) => {
                7u8.hash(&mut hasher);
                l.len().hash(&mut hasher);
            }
            Value::Map(m) => {
                8u8.hash(&mut hasher);
                m.len().hash(&mut hasher);
            }
            Value::Vector(v) => {
                9u8.hash(&mut hasher);
                v.len().hash(&mut hasher);
                // Hash first few elements for distribution
                for &f in v.iter().take(4) {
                    f.to_bits().hash(&mut hasher);
                }
            }
            Value::Path { nodes, edges } => {
                13u8.hash(&mut hasher);
                nodes.len().hash(&mut hasher);
                edges.len().hash(&mut hasher);
            }
            Value::GCounter(counts) => {
                15u8.hash(&mut hasher);
                counts.len().hash(&mut hasher);
            }
            Value::OnCounter { pos, neg } => {
                16u8.hash(&mut hasher);
                pos.len().hash(&mut hasher);
                neg.len().hash(&mut hasher);
            }
            _ => {
                255u8.hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

/// Helper to read u64 in little endian.
fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn malformed_partition_payload(error: std::io::Error, context: &str) -> std::io::Error {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("truncated {context}: {error}"),
        )
    } else {
        error
    }
}

/// Helper to write u64 in little endian.
fn write_u64<W: Write>(writer: &mut W, value: u64) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::spill::{CleartextSpillRecordProvider, SpillFrameLimits};
    use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig, MemoryRegion};
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tempfile::TempDir;

    struct PanicOnPartitionProviderDrop;

    impl super::super::SpillRecordProvider for PanicOnPartitionProviderDrop {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            super::super::SpillRecordProvider::begin_file(
                &super::super::CleartextSpillRecordProvider,
                identity,
            )
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            super::super::SpillRecordProvider::file_workspace_allocation_bound(
                &super::super::CleartextSpillRecordProvider,
            )
        }
    }

    impl Drop for PanicOnPartitionProviderDrop {
        fn drop(&mut self) {
            panic!("deterministic partition provider destructor panic");
        }
    }

    /// Cleartext provider whose next reported file-workspace bound can be
    /// raised deterministically. The raised bound is still conservative for
    /// the delegated provider; consuming it once lets a retry use the normal
    /// cleartext bound.
    struct OneRaisedWorkspaceBoundProvider {
        next_bound: AtomicUsize,
    }

    impl OneRaisedWorkspaceBoundProvider {
        fn new() -> Self {
            Self {
                next_bound: AtomicUsize::new(0),
            }
        }

        fn raise_next_bound_to(&self, bytes: usize) {
            self.next_bound.store(bytes, Ordering::Release);
        }
    }

    impl super::super::SpillRecordProvider for OneRaisedWorkspaceBoundProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            super::super::SpillRecordProvider::begin_file(
                &super::super::CleartextSpillRecordProvider,
                identity,
            )
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            let raised = self.next_bound.swap(0, Ordering::AcqRel);
            if raised != 0 {
                Some(raised)
            } else {
                super::super::SpillRecordProvider::file_workspace_allocation_bound(
                    &super::super::CleartextSpillRecordProvider,
                )
            }
        }
    }

    struct PanicOnOpenRecordDropProvider {
        inner: Arc<dyn super::super::SpillRecordProvider>,
    }

    impl super::super::SpillRecordProvider for PanicOnOpenRecordDropProvider {
        fn seals(&self) -> bool {
            self.inner.seals()
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            Ok(Box::new(PanicOnOpenRecordDrop {
                inner: Some(self.inner.begin_file(identity)?),
            }))
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            self.inner
                .file_workspace_allocation_bound()?
                .checked_add(std::mem::size_of::<PanicOnOpenRecordDrop>())
        }
    }

    struct PanicOnOpenRecordDrop {
        inner: Option<Box<dyn super::super::OpenSpillRecord>>,
    }

    impl PanicOnOpenRecordDrop {
        fn inner(&self) -> &dyn super::super::OpenSpillRecord {
            self.inner
                .as_deref()
                .expect("hostile record retains its delegate until destruction")
        }

        fn inner_mut(&mut self) -> &mut dyn super::super::OpenSpillRecord {
            self.inner
                .as_deref_mut()
                .expect("hostile record retains its delegate until destruction")
        }
    }

    impl super::super::OpenSpillRecord for PanicOnOpenRecordDrop {
        fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
            self.inner().stored_len(plaintext_len)
        }

        fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
            self.inner().seal_allocation_bound(plaintext_len)
        }

        fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
            self.inner().open_allocation_bound(stored_len)
        }

        fn seal(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner_mut().seal(meta, aad, plaintext)
        }

        fn open(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner_mut().open(meta, aad, stored)
        }
    }

    impl Drop for PanicOnOpenRecordDrop {
        fn drop(&mut self) {
            drop(self.inner.take());
            panic!("deterministic partition open-record destructor panic");
        }
    }

    /// Uses an ordinary writer record, then returns a hostile record for the
    /// first reader of that finished file.
    struct PanicOnReaderOpenRecordDropProvider {
        begins: AtomicUsize,
        reader_drops: Arc<AtomicUsize>,
    }

    impl PanicOnReaderOpenRecordDropProvider {
        fn new() -> Self {
            Self {
                begins: AtomicUsize::new(0),
                reader_drops: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl super::super::SpillRecordProvider for PanicOnReaderOpenRecordDropProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            let inner = super::super::SpillRecordProvider::begin_file(
                &super::super::CleartextSpillRecordProvider,
                identity,
            )?;
            if self.begins.fetch_add(1, Ordering::AcqRel) == 0 {
                Ok(inner)
            } else {
                Ok(Box::new(ObservedPanicOnOpenRecordDrop {
                    inner: Some(inner),
                    drops: Arc::clone(&self.reader_drops),
                }))
            }
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            super::super::SpillRecordProvider::file_workspace_allocation_bound(
                &super::super::CleartextSpillRecordProvider,
            )?
            .checked_add(std::mem::size_of::<ObservedPanicOnOpenRecordDrop>())
        }
    }

    struct ObservedPanicOnOpenRecordDrop {
        inner: Option<Box<dyn super::super::OpenSpillRecord>>,
        drops: Arc<AtomicUsize>,
    }

    impl super::super::OpenSpillRecord for ObservedPanicOnOpenRecordDrop {
        fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
            self.inner
                .as_deref()
                .expect("observed hostile record retains its delegate")
                .stored_len(plaintext_len)
        }

        fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
            self.inner
                .as_deref()
                .expect("observed hostile record retains its delegate")
                .seal_allocation_bound(plaintext_len)
        }

        fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
            self.inner
                .as_deref()
                .expect("observed hostile record retains its delegate")
                .open_allocation_bound(stored_len)
        }

        fn seal(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner
                .as_deref_mut()
                .expect("observed hostile record retains its delegate")
                .seal(meta, aad, plaintext)
        }

        fn open(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner
                .as_deref_mut()
                .expect("observed hostile record retains its delegate")
                .open(meta, aad, stored)
        }
    }

    impl Drop for ObservedPanicOnOpenRecordDrop {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::AcqRel);
            drop(self.inner.take());
            panic!("deterministic reader open-record destructor panic");
        }
    }

    const READER_PRIMARY_ERROR_AUTHORITY: usize = 64 * 1024;

    /// Uses an ordinary writer record, then returns a reader record whose
    /// partition-start open fails with a heap-bearing provider error.
    struct HeapErrorOnReaderPartitionStartProvider {
        begins: AtomicUsize,
        buffer_manager: Arc<BufferManager>,
        allocated_when_dropped: Arc<AtomicUsize>,
    }

    impl super::super::SpillRecordProvider for HeapErrorOnReaderPartitionStartProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            identity: super::super::SpillFileIdentity,
        ) -> std::io::Result<Box<dyn super::super::OpenSpillRecord>> {
            let inner = super::super::SpillRecordProvider::begin_file(
                &super::super::CleartextSpillRecordProvider,
                identity,
            )?;
            if self.begins.fetch_add(1, Ordering::AcqRel) == 0 {
                Ok(inner)
            } else {
                Ok(Box::new(HeapErrorOnPartitionStartOpenRecord {
                    inner,
                    buffer_manager: Arc::clone(&self.buffer_manager),
                    allocated_when_dropped: Arc::clone(&self.allocated_when_dropped),
                }))
            }
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            super::super::SpillRecordProvider::file_workspace_allocation_bound(
                &super::super::CleartextSpillRecordProvider,
            )?
            .checked_add(READER_PRIMARY_ERROR_AUTHORITY)
        }
    }

    struct HeapErrorOnPartitionStartOpenRecord {
        inner: Box<dyn super::super::OpenSpillRecord>,
        buffer_manager: Arc<BufferManager>,
        allocated_when_dropped: Arc<AtomicUsize>,
    }

    impl super::super::OpenSpillRecord for HeapErrorOnPartitionStartOpenRecord {
        fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
            self.inner.stored_len(plaintext_len)
        }

        fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
            self.inner.seal_allocation_bound(plaintext_len)
        }

        fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
            self.inner.open_allocation_bound(stored_len)
        }

        fn seal(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            plaintext: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            self.inner.seal(meta, aad, plaintext)
        }

        fn open(
            &mut self,
            meta: &super::super::SpillRecordMeta,
            aad: &[u8; 32],
            stored: &[u8],
        ) -> std::io::Result<Vec<u8>> {
            if meta.kind() == super::super::SpillRecordKind::PartitionStart {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    HeapDropProbe {
                        bytes: vec![0xe1; READER_PRIMARY_ERROR_AUTHORITY / 2],
                        buffer_manager: Arc::clone(&self.buffer_manager),
                        allocated_when_dropped: Arc::clone(&self.allocated_when_dropped),
                    },
                ));
            }
            self.inner.open(meta, aad, stored)
        }
    }

    struct CancelNthIo {
        target: super::super::SpillIoOperation,
        trigger: usize,
        armed: AtomicBool,
        matching: AtomicUsize,
        cancellation: crate::execution::QueryCancellationHandle,
        failure: Option<(std::io::ErrorKind, &'static str)>,
        fail_delete: AtomicBool,
    }

    impl CancelNthIo {
        fn new(
            target: super::super::SpillIoOperation,
            trigger: usize,
            cancellation: crate::execution::QueryCancellationHandle,
        ) -> Self {
            Self {
                target,
                trigger,
                armed: AtomicBool::new(true),
                matching: AtomicUsize::new(0),
                cancellation,
                failure: None,
                fail_delete: AtomicBool::new(false),
            }
        }

        fn with_failure(
            target: super::super::SpillIoOperation,
            trigger: usize,
            cancellation: crate::execution::QueryCancellationHandle,
            kind: std::io::ErrorKind,
            message: &'static str,
        ) -> Self {
            Self {
                failure: Some((kind, message)),
                ..Self::new(target, trigger, cancellation)
            }
        }

        fn with_delete_failure(self) -> Self {
            self.fail_delete.store(true, Ordering::Relaxed);
            self
        }

        fn disarm(&self) {
            self.armed.store(false, Ordering::Release);
        }

        fn arm(&self) {
            self.matching.store(0, Ordering::Relaxed);
            self.armed.store(true, Ordering::Release);
        }

        fn permit_delete(&self) {
            self.fail_delete.store(false, Ordering::Release);
        }
    }

    impl super::super::SpillIo for CancelNthIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if self.armed.load(Ordering::Acquire) && operation == self.target {
                let matching = self.matching.fetch_add(1, Ordering::Relaxed) + 1;
                if matching == self.trigger {
                    self.cancellation.cancel();
                    if let Some((kind, message)) = self.failure {
                        return Err(std::io::Error::new(kind, message));
                    }
                }
            }
            if operation == super::super::SpillIoOperation::Delete
                && self.fail_delete.load(Ordering::Acquire)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "deterministic cancelled partition delete failure",
                ));
            }
            Ok(())
        }
    }

    struct PanicThenAllowThenFailDeleteIo {
        deletes: std::sync::atomic::AtomicUsize,
    }

    impl PanicThenAllowThenFailDeleteIo {
        fn new() -> Self {
            Self {
                deletes: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl super::super::SpillIo for PanicThenAllowThenFailDeleteIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            if operation != super::super::SpillIoOperation::Delete {
                return Ok(());
            }
            match self
                .deletes
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            {
                0 => panic!("first partition delete hook panic"),
                1 => Ok(()),
                2 => Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "third partition delete hook failure",
                )),
                _ => Ok(()),
            }
        }
    }

    #[derive(Debug)]
    struct PrimaryDropPanic;

    struct HeapDropProbe {
        bytes: Vec<u8>,
        buffer_manager: Arc<BufferManager>,
        allocated_when_dropped: Arc<AtomicUsize>,
    }

    impl std::fmt::Debug for HeapDropProbe {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("HeapDropProbe")
                .field("bytes", &self.bytes.len())
                .finish_non_exhaustive()
        }
    }

    impl std::fmt::Display for HeapDropProbe {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "heap drop probe ({} bytes)", self.bytes.len())
        }
    }

    impl std::error::Error for HeapDropProbe {}

    impl Drop for HeapDropProbe {
        fn drop(&mut self) {
            self.allocated_when_dropped
                .store(self.buffer_manager.allocated(), Ordering::Release);
        }
    }

    #[derive(Clone)]
    struct HeapValueDropProbe {
        bytes: Vec<u8>,
        buffer_manager: Arc<BufferManager>,
        allocated_when_dropped: Option<Arc<AtomicUsize>>,
    }

    impl HeapValueDropProbe {
        fn unobserved(bytes: Vec<u8>, buffer_manager: Arc<BufferManager>) -> Self {
            Self {
                bytes,
                buffer_manager,
                allocated_when_dropped: None,
            }
        }

        fn observed(
            bytes: Vec<u8>,
            buffer_manager: Arc<BufferManager>,
            allocated_when_dropped: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                bytes,
                buffer_manager,
                allocated_when_dropped: Some(allocated_when_dropped),
            }
        }
    }

    impl Drop for HeapValueDropProbe {
        fn drop(&mut self) {
            if let Some(observation) = self.allocated_when_dropped.as_ref() {
                observation.store(self.buffer_manager.allocated(), Ordering::Release);
            }
        }
    }

    struct PanickingDrainPhysical {
        buffer_manager: Arc<BufferManager>,
        allocated_when_dropped: Arc<AtomicUsize>,
    }

    impl Drop for PanickingDrainPhysical {
        fn drop(&mut self) {
            self.allocated_when_dropped
                .store(self.buffer_manager.allocated(), Ordering::Release);
            panic!("hostile accounted drain physical Drop");
        }
    }

    #[derive(Clone)]
    struct HostileDrainValue {
        bytes: Vec<u8>,
        buffer_manager: Arc<BufferManager>,
        allocated_when_dropped: Option<Arc<AtomicUsize>>,
        panic_on_drop: bool,
    }

    impl Drop for HostileDrainValue {
        fn drop(&mut self) {
            if let Some(observation) = self.allocated_when_dropped.as_ref() {
                observation.store(self.buffer_manager.allocated(), Ordering::Release);
            }
            assert!(
                !self.panic_on_drop,
                "hostile accounted partition value Drop"
            );
        }
    }

    impl super::shared_immutable_partition_value::Sealed for HeapValueDropProbe {}
    impl SharedImmutablePartitionValue for HeapValueDropProbe {}

    /// Creates a test manager. Returns (TempDir, manager). TempDir must be kept alive.
    fn create_manager() -> (TempDir, Arc<SpillManager>) {
        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(temp_dir.path())
                .build()
                .unwrap(),
        );
        (temp_dir, manager)
    }

    /// Simple i64 serializer for tests.
    #[allow(clippy::trivially_copy_pass_by_ref)] // Required by PartitionedState::new signature
    fn serialize_i64(value: &i64, w: &mut dyn Write) -> std::io::Result<()> {
        w.write_all(&value.to_le_bytes())
    }

    /// Simple i64 deserializer for tests.
    fn deserialize_i64(r: &mut dyn Read) -> std::io::Result<i64> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn encode_partition_entry(key: &[u8], columns: u64, custom: &[u8]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(key.len() as u64).to_le_bytes());
        payload.extend_from_slice(key);
        payload.extend_from_slice(&columns.to_le_bytes());
        payload.extend_from_slice(&(custom.len() as u64).to_le_bytes());
        payload.extend_from_slice(custom);
        payload
    }

    fn install_partition_file(
        state: &mut PartitionedState<i64>,
        manager: &Arc<SpillManager>,
        payloads: &[Vec<u8>],
    ) {
        let mut file = manager.create_file(SpillFileRole::NativePartition).unwrap();
        file.write_partition_start(payloads.len() as u64).unwrap();
        for payload in payloads {
            file.write_partition_entry(payload).unwrap();
        }
        file.finish_write().unwrap();
        state.partitions[0] = None;
        state.spill_files[0] = Some(file);
        state.partition_sizes[0] = payloads.len();
    }

    fn key(values: &[i64]) -> Vec<Value> {
        values.iter().map(|&v| Value::Int64(v)).collect()
    }

    fn over_depth_key() -> Vec<Value> {
        let mut value = Value::Null;
        for _ in 0..129 {
            value = Value::List(Arc::from([value]));
        }
        vec![value]
    }

    fn cancellation_manager(directory: &TempDir, io: Arc<CancelNthIo>) -> Arc<SpillManager> {
        Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(io)
                .build()
                .unwrap(),
        )
    }

    fn qualified_i64_state(
        manager: Arc<SpillManager>,
        cancellation: crate::execution::QueryCancellationToken,
    ) -> PartitionedState<i64> {
        PartitionedState::new_with_bounded_codec_and_cancellation(
            manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            cancellation,
        )
    }

    fn key_for_partition<V: Clone + Send + Sync + 'static>(
        state: &PartitionedState<V>,
        partition: usize,
    ) -> Vec<Value> {
        (0..10_000)
            .map(|value| key(&[value]))
            .find(|candidate| state.partition_for(candidate) == partition)
            .expect("test key search must cover every small partition")
    }

    fn wide_key_for_partition<V: Clone + Send + Sync + 'static>(
        state: &PartitionedState<V>,
        partition: usize,
        string_bytes: usize,
        fill: char,
    ) -> Vec<Value> {
        (0..10_000)
            .map(|nonce| {
                let suffix = nonce.to_string();
                let body = string_bytes
                    .checked_sub(suffix.len())
                    .expect("wide test key must leave room for its nonce");
                let mut value = fill.to_string().repeat(body);
                value.push_str(&suffix);
                vec![Value::from(value)]
            })
            .find(|candidate| state.partition_for(candidate) == partition)
            .expect("wide test key search must cover every small partition")
    }

    fn accounted_i64_state(
        manager: Arc<SpillManager>,
        buffer_manager: &Arc<BufferManager>,
        partitions: usize,
    ) -> PartitionedState<i64> {
        accounted_i64_state_with_cancellation(
            manager,
            buffer_manager,
            partitions,
            crate::execution::QueryExecutionControl::new().token(),
        )
    }

    fn accounted_i64_state_with_cancellation(
        manager: Arc<SpillManager>,
        buffer_manager: &Arc<BufferManager>,
        partitions: usize,
        cancellation: crate::execution::QueryCancellationToken,
    ) -> PartitionedState<i64> {
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .expect("zero-sized root grant must be available");
        PartitionedState::new_accounted_with_cancellation(
            manager,
            partitions,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            |_value: &i64| Ok(0),
            grant,
            cancellation,
        )
        .expect("accounted partition construction must fit the test budget")
    }

    fn accounted_bytes_state(
        manager: Arc<SpillManager>,
        buffer_manager: &Arc<BufferManager>,
        cancellation: crate::execution::QueryCancellationToken,
    ) -> PartitionedState<Vec<u8>> {
        accounted_bytes_state_with_partitions(manager, buffer_manager, cancellation, 1)
    }

    fn accounted_bytes_state_with_partitions(
        manager: Arc<SpillManager>,
        buffer_manager: &Arc<BufferManager>,
        cancellation: crate::execution::QueryCancellationToken,
        partitions: usize,
    ) -> PartitionedState<Vec<u8>> {
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .expect("zero-sized root grant must be available");
        PartitionedState::new_accounted_with_cancellation(
            manager,
            partitions,
            |value: &Vec<u8>, writer: &mut dyn Write, _limits| writer.write_all(value),
            |reader: &mut dyn Read, _limits| {
                let mut value = Vec::new();
                reader.read_to_end(&mut value)?;
                Ok(value)
            },
            |value: &Vec<u8>| Ok(value.capacity()),
            grant,
            cancellation,
        )
        .expect("accounted byte-state construction must fit the test budget")
    }

    fn buffer_manager_with_exact_budget(budget: usize) -> Arc<BufferManager> {
        let mut config = BufferManagerConfig::with_budget(budget);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        BufferManager::new(config)
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn accounted_partition_constructor_returns_allocation_error_and_releases_grant() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(usize::MAX);
        let spill_file_bytes = std::mem::size_of::<Option<SpillFile>>();
        let impossible_partitions = (isize::MAX as usize / spill_file_bytes) + 1;
        let admitted =
            PartitionedState::<i64>::initial_resident_capacity_bytes(impossible_partitions)
                .expect("the complete catalog envelope fits usize before Vec rejects one catalog");
        assert!(admitted > 0);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .expect("zero-sized root grant must be available");

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PartitionedState::new_accounted_with_cancellation(
                spill_manager,
                impossible_partitions,
                |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
                |reader: &mut dyn Read, _limits| deserialize_i64(reader),
                |_value: &i64| Ok(0),
                grant,
                crate::execution::QueryExecutionControl::new().token(),
            )
        }));

        let result = outcome.expect("impossible catalog reservation must not panic");
        let Err(error) = result else {
            panic!("impossible catalog reservation unexpectedly succeeded");
        };
        assert!(matches!(
            &error,
            PartitionOperationError::Allocation {
                container: "partition spill-file catalog",
                ..
            }
        ));
        let source = std::error::Error::source(&error)
            .expect("structured allocation failure must retain its source");
        assert!(
            source.downcast_ref::<TryReserveError>().is_some(),
            "allocation failure source must remain the original TryReserveError"
        );
        assert_eq!(
            buffer_manager.allocated(),
            0,
            "failed construction must release its pre-admitted root grant"
        );
    }

    #[test]
    fn accounted_partition_charges_observed_capacity_and_releases_on_drop() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = BufferManager::with_budget(1 << 20);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        let baseline = buffer_manager.allocated();
        let lookup = key_for_partition(&state, 0);

        *state
            .get_or_insert_with_accounted(lookup, 0, || 10)
            .unwrap() += 1;

        assert!(buffer_manager.allocated() > baseline);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        drop(state);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_serialized_key_retains_its_capacity_authority_until_drop() {
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut root = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let limits = SpillFrameLimits::new(4096, 4096).unwrap();

        let key = AccountedSerializedKey::from_values(&key(&[42]), limits, &mut root).unwrap();

        assert_eq!(root.size(), 0, "construction must retain a child grant");
        assert_eq!(key.granted_bytes(), key.serialized().0.capacity());
        assert_eq!(buffer_manager.allocated(), key.granted_bytes());
        drop(key);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_serialized_key_denial_precedes_encoding_and_allocation() {
        let buffer_manager = buffer_manager_with_exact_budget(0);
        let mut root = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let limits = SpillFrameLimits::new(4096, 4096).unwrap();
        let encoded = AtomicBool::new(false);

        let Err(error) = AccountedSerializedKey::from_values_with(
            &key(&[42]),
            limits,
            &mut root,
            |_, _, _, _| {
                encoded.store(true, Ordering::Release);
                Ok(0)
            },
        ) else {
            panic!("zero-byte budget must deny key admission")
        };

        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(!encoded.load(Ordering::Acquire));
        assert_eq!(root.size(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_serialized_key_pre_admits_counter_sort_scratch() {
        let limits = SpillFrameLimits::new(4096, 4096).unwrap();
        let values = [Value::GCounter(Arc::new(HashMap::from([
            ("replica-a".to_owned(), 1),
            ("replica-b".to_owned(), 2),
        ])))];
        let measurement =
            measure_serialized_row_with_limits(&values, limits.codec_limits()).unwrap();
        let scratch = CounterSortScratch::requested_capacity_bytes(
            measurement.counter_sort_entries,
            measurement.counter_sort_key_bytes,
        )
        .unwrap()
        .checked_mul(2)
        .unwrap();

        let calibration_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut calibration_root = calibration_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let calibration =
            AccountedSerializedKey::from_values(&values, limits, &mut calibration_root).unwrap();
        let key_capacity = calibration.granted_bytes();
        drop(calibration);
        drop(calibration_root);

        let buffer_manager = buffer_manager_with_exact_budget(
            key_capacity.checked_add(scratch).unwrap().saturating_sub(1),
        );
        let mut root = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let encoded = AtomicBool::new(false);

        let Err(error) =
            AccountedSerializedKey::from_values_with(&values, limits, &mut root, |_, _, _, _| {
                encoded.store(true, Ordering::Release);
                Ok(measurement.encoded_bytes)
            })
        else {
            panic!("insufficient scratch budget must deny key construction")
        };

        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(!encoded.load(Ordering::Acquire));
        assert_eq!(root.size(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn accounted_serialized_key_preserves_serializer_error_and_panic() {
        let limits = SpillFrameLimits::new(4096, 4096).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut root = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();

        let Err(error) = AccountedSerializedKey::from_values_with(
            &key(&[42]),
            limits,
            &mut root,
            |_, writer, _, _| {
                writer.write_all(&[0xaa])?;
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "deterministic key serializer error",
                ))
            },
        ) else {
            panic!("injected serializer error must be preserved")
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "deterministic key serializer error");
        assert_eq!(buffer_manager.allocated(), root.size());

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = AccountedSerializedKey::from_values_with(
                &key(&[42]),
                limits,
                &mut root,
                |_, writer, _, _| {
                    writer.write_all(&[0xbb])?;
                    std::panic::panic_any(0x5eed_u64)
                },
            );
        }))
        .expect_err("hostile key serializer must unwind");
        assert_eq!(panic.downcast_ref::<u64>(), Some(&0x5eed_u64));
        assert_eq!(buffer_manager.allocated(), root.size());
    }

    #[test]
    fn accounted_key_publication_unwind_drops_key_before_root_rollback() {
        let limits = SpillFrameLimits::new(4096, 4096).unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut root = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let staged = AccountedSerializedKey::from_values(&key(&[42]), limits, &mut root).unwrap();
        let mut partition = new_partition_map();
        partition.try_reserve(1).unwrap();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let publication = AccountedKeyPublication::new(staged, &mut root);
            let _ = publication.publish_into(
                &mut partition,
                PartitionEntry {
                    num_key_columns: 1,
                    resident_bound: 0,
                    value: 7_i64,
                },
                || std::panic::panic_any(0xcafe_u64),
            );
        }))
        .expect_err("publication hook must unwind");

        assert_eq!(panic.downcast_ref::<u64>(), Some(&0xcafe_u64));
        assert!(partition.is_empty());
        assert_eq!(root.size(), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn existing_accounted_key_releases_staging_authority_without_replacement() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[42]);
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 7)
            .unwrap() += 1;
        let resident = buffer_manager.allocated();
        let replaced = AtomicBool::new(false);

        let value = state
            .get_or_insert_with_accounted(lookup, 0, || {
                replaced.store(true, Ordering::Release);
                99
            })
            .unwrap();

        assert_eq!(*value, 8);
        assert!(!replaced.load(Ordering::Acquire));
        assert_eq!(buffer_manager.allocated(), resident);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn resident_update_precharges_construction_peak_before_builder() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[42]);
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 7)
            .unwrap() += 1;
        let resident = buffer_manager.allocated();
        let construction_peak = 4096;

        state
            .try_replace_accounted(
                lookup.clone(),
                |old| {
                    assert_eq!(old, Some(&8));
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak,
                    })
                },
                |old| {
                    assert_eq!(old, Some(&8));
                    assert!(
                        buffer_manager.allocated() >= resident + construction_peak,
                        "the complete declared construction peak must be held before build"
                    );
                    Ok::<_, std::convert::Infallible>(11)
                },
            )
            .unwrap();

        assert_eq!(state.get(&lookup).unwrap(), Some(&11));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_update_preserves_typed_caller_error_and_old_value() {
        #[derive(Debug, PartialEq, Eq)]
        struct CallerFailure(u64);

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[7]);
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap() += 1;

        let result = state.try_replace_accounted(
            lookup.clone(),
            |old| {
                assert_eq!(old, Some(&11));
                Ok::<_, CallerFailure>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 64,
                })
            },
            |old| {
                assert_eq!(old, Some(&11));
                Err(CallerFailure(0x5eed))
            },
        );

        match result {
            Err(PartitionUpdateError::Caller(error)) => {
                assert_eq!(error.error(), &CallerFailure(0x5eed));
            }
            other => panic!("typed caller error was not preserved: {other:?}"),
        }
        assert_eq!(state.get(&lookup).unwrap(), Some(&11));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_update_heap_error_retains_construction_authority_until_drop() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[17]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap();
        let resident = state.granted_bytes();
        let construction_peak = 64 * 1024;
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let drop_observation = Arc::clone(&allocated_when_dropped);
        let failure_manager = Arc::clone(&buffer_manager);

        let result = state.try_replace_accounted(
            lookup.clone(),
            |_| {
                Ok::<_, HeapDropProbe>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak,
                })
            },
            |_| {
                Err(HeapDropProbe {
                    bytes: vec![0x5e; 32 * 1024],
                    buffer_manager: failure_manager,
                    allocated_when_dropped: drop_observation,
                })
            },
        );

        let failure = match result {
            Err(PartitionUpdateError::Caller(error)) => error,
            other => panic!("heap-bearing caller error was not preserved: {other:?}"),
        };
        assert_eq!(failure.error().bytes.len(), 32 * 1024);
        assert_eq!(
            buffer_manager.allocated(),
            resident + construction_peak,
            "the escaped error must keep its construction authority"
        );
        assert_eq!(state.get(&lookup).unwrap(), Some(&10));

        drop(failure);
        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            resident + construction_peak,
            "physical error bytes must drop before their authority"
        );
        assert_eq!(buffer_manager.allocated(), resident);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn resident_update_preserves_panic_payload_old_value_and_grant() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let panic_on_capacity = Arc::new(AtomicBool::new(false));
        let callback_guard = Arc::clone(&panic_on_capacity);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .expect("zero-sized root grant must be available");
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            move |_value: &i64| {
                assert!(
                    !callback_guard.load(Ordering::Acquire),
                    "builder-panic rollback must not re-enter the capacity callback"
                );
                Ok(0)
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .expect("accounted partition construction must fit the test budget");
        let lookup = key(&[7]);
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap() += 1;
        let resident = state.granted_bytes();
        let construction_peak = 64 * 1024;
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let drop_observation = Arc::clone(&allocated_when_dropped);
        let panic_manager = Arc::clone(&buffer_manager);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.try_replace_accounted(
                lookup.clone(),
                |_| {
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak,
                    })
                },
                |_| -> Result<i64, std::convert::Infallible> {
                    panic_on_capacity.store(true, Ordering::Release);
                    std::panic::panic_any(HeapDropProbe {
                        bytes: vec![0xfe; 32 * 1024],
                        buffer_manager: panic_manager,
                        allocated_when_dropped: drop_observation,
                    })
                },
            );
        }))
        .expect_err("replacement builder must unwind");

        assert_eq!(
            buffer_manager.allocated(),
            resident + construction_peak,
            "the escaped panic payload must keep its construction authority"
        );
        let accounted_panic = panic
            .downcast_ref::<AccountedPartitionPanic>()
            .expect("the escaped panic must retain its construction authority");
        assert_eq!(
            accounted_panic
                .payload()
                .downcast_ref::<HeapDropProbe>()
                .map(|payload| payload.bytes.len()),
            Some(32 * 1024)
        );
        panic_on_capacity.store(false, Ordering::Release);
        assert_eq!(state.get(&lookup).unwrap(), Some(&11));
        drop(panic);
        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            resident + construction_peak,
            "physical panic bytes must drop before their authority"
        );
        assert_eq!(buffer_manager.allocated(), resident);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn callback_panic_authority_resident_capacity_drops_candidate_before_escape() {
        const CONSTRUCTION_PEAK: usize = 64 * 1024;
        const CANDIDATE_BYTES: usize = 24 * 1024;
        const PANIC_BYTES: usize = 16 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let candidate_drop_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_panic_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let panic_payload_drop_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_manager = Arc::clone(&buffer_manager);
        let capacity_panic_observation = Arc::clone(&capacity_panic_allocation);
        let panic_payload_drop_observation = Arc::clone(&panic_payload_drop_allocation);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &HeapValueDropProbe, writer: &mut dyn Write, _limits| {
                writer.write_all(&value.bytes)
            },
            |_reader: &mut dyn Read, _limits| {
                Err(std::io::Error::other(
                    "heap value probe decoder is unused in this test",
                ))
            },
            move |value: &HeapValueDropProbe| {
                if value.bytes.first() == Some(&0x22) {
                    capacity_panic_observation
                        .store(capacity_manager.allocated(), Ordering::Release);
                    std::panic::panic_any(HeapDropProbe {
                        bytes: vec![0xca; PANIC_BYTES],
                        buffer_manager: Arc::clone(&capacity_manager),
                        allocated_when_dropped: Arc::clone(&panic_payload_drop_observation),
                    });
                }
                Ok(value.bytes.capacity())
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[71]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 16, || {
                HeapValueDropProbe::unobserved(vec![0x11; 16], Arc::clone(&buffer_manager))
            })
            .unwrap();
        let resident = state.granted_bytes();
        let candidate_manager = Arc::clone(&buffer_manager);
        let candidate_drop_observation = Arc::clone(&candidate_drop_allocation);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.try_replace_accounted(
                lookup.clone(),
                |_| {
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: CANDIDATE_BYTES,
                        construction_peak: CONSTRUCTION_PEAK,
                    })
                },
                |_| {
                    Ok::<_, std::convert::Infallible>(HeapValueDropProbe::observed(
                        vec![0x22; CANDIDATE_BYTES],
                        candidate_manager,
                        candidate_drop_observation,
                    ))
                },
            );
        }))
        .expect_err("replacement capacity callback must unwind");

        let allocation_at_capacity_panic = capacity_panic_allocation.load(Ordering::Acquire);
        assert_eq!(
            candidate_drop_allocation.load(Ordering::Acquire),
            allocation_at_capacity_panic,
            "the built value must be physically destroyed while its full construction authority remains live"
        );
        assert_eq!(
            buffer_manager.allocated(),
            resident + CONSTRUCTION_PEAK,
            "the escaped capacity-panic payload must retain construction authority"
        );
        let accounted_panic = panic
            .downcast_ref::<AccountedPartitionPanic>()
            .expect("capacity panic must be paired with construction authority");
        assert_eq!(
            accounted_panic
                .payload()
                .downcast_ref::<HeapDropProbe>()
                .map(|payload| payload.bytes.len()),
            Some(PANIC_BYTES)
        );
        assert_eq!(
            state
                .get(&lookup)
                .unwrap()
                .map(|value| value.bytes.as_slice()),
            Some([0x11; 16].as_slice()),
            "a capacity panic cannot publish the candidate"
        );

        drop(panic);
        assert_eq!(
            panic_payload_drop_allocation.load(Ordering::Acquire),
            resident + CONSTRUCTION_PEAK,
            "the panic payload must be destroyed before its authority"
        );
        assert_eq!(buffer_manager.allocated(), resident);
        state
            .try_replace_accounted(
                lookup.clone(),
                |_| {
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 8,
                        construction_peak: 8,
                    })
                },
                |_| {
                    Ok::<_, std::convert::Infallible>(HeapValueDropProbe::unobserved(
                        vec![0x33; 8],
                        Arc::clone(&buffer_manager),
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            state
                .get(&lookup)
                .unwrap()
                .map(|value| value.bytes.as_slice()),
            Some([0x33; 8].as_slice())
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn capacity_error_authority_mediated_value_and_error_remain_covered() {
        const CONSTRUCTION_PEAK: usize = 64 * 1024;
        const CANDIDATE_BYTES: usize = 24 * 1024;
        const DENIED_BYTES: usize = 17;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let candidate_drop_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_error_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_manager = Arc::clone(&buffer_manager);
        let capacity_error_observation = Arc::clone(&capacity_error_allocation);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &HeapValueDropProbe, writer: &mut dyn Write, _limits| {
                writer.write_all(&value.bytes)
            },
            |_reader: &mut dyn Read, _limits| {
                Err(std::io::Error::other(
                    "heap value probe decoder is unused in this test",
                ))
            },
            move |value: &HeapValueDropProbe| {
                if value.bytes.first() == Some(&0x22) {
                    capacity_error_observation
                        .store(capacity_manager.allocated(), Ordering::Release);
                    return Err(MemoryGrantError::Denied {
                        additional_bytes: DENIED_BYTES,
                    });
                }
                Ok(value.bytes.capacity())
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[72]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 16, || {
                HeapValueDropProbe::unobserved(vec![0x11; 16], Arc::clone(&buffer_manager))
            })
            .unwrap();
        let resident = state.granted_bytes();
        let candidate_manager = Arc::clone(&buffer_manager);
        let candidate_drop_observation = Arc::clone(&candidate_drop_allocation);

        let result = state.try_replace_accounted(
            lookup.clone(),
            |_| {
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: CANDIDATE_BYTES,
                    construction_peak: CONSTRUCTION_PEAK,
                })
            },
            |_| {
                Ok::<_, std::convert::Infallible>(HeapValueDropProbe::observed(
                    vec![0x22; CANDIDATE_BYTES],
                    candidate_manager,
                    candidate_drop_observation,
                ))
            },
        );

        let error = match result {
            Err(PartitionUpdateError::Partition(error)) => error,
            Err(
                PartitionUpdateError::Caller(_)
                | PartitionUpdateError::InvalidAdmission { .. }
                | PartitionUpdateError::RetainedCapacityExceeded { .. },
            ) => panic!("capacity refusal must retain partition-error classification"),
            Ok(()) => panic!("capacity refusal must fail mediated replacement"),
        };
        assert!(matches!(
            error.resident_memory_error(),
            Some(MemoryGrantError::Denied { additional_bytes })
                if *additional_bytes == DENIED_BYTES
        ));
        assert_eq!(
            candidate_drop_allocation.load(Ordering::Acquire),
            capacity_error_allocation.load(Ordering::Acquire),
            "the completed value must be destroyed while full construction authority remains live"
        );
        assert_eq!(
            buffer_manager.allocated(),
            resident + CONSTRUCTION_PEAK,
            "the returned heap-backed error must retain construction authority until Drop"
        );
        assert_eq!(
            state
                .get(&lookup)
                .unwrap()
                .map(|value| value.bytes.as_slice()),
            Some([0x11; 16].as_slice()),
            "a capacity error cannot publish the candidate"
        );

        drop(error);
        assert_eq!(buffer_manager.allocated(), resident);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        state
            .try_replace_accounted(
                lookup.clone(),
                |_| {
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 8,
                        construction_peak: 8,
                    })
                },
                |_| {
                    Ok::<_, std::convert::Infallible>(HeapValueDropProbe::unobserved(
                        vec![0x33; 8],
                        Arc::clone(&buffer_manager),
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            state
                .get(&lookup)
                .unwrap()
                .map(|value| value.bytes.as_slice()),
            Some([0x33; 8].as_slice())
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_update_rejects_invalid_peak_before_builder() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let built = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            key(&[9]),
            |_| {
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 65,
                    construction_peak: 64,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(1)
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::InvalidAdmission {
                retained_upper_bound: 65,
                construction_peak: 64
            })
        ));
        assert!(!built.load(Ordering::Acquire));
        assert_eq!(state.total_size(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_mutations_saturated_count_fences_before_key_staging_or_callbacks() {
        const COLD_KEY_BYTES: usize = 32 * 1024;
        const TARGET_KEY_BYTES: usize = 110 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        let cold = wide_key_for_partition(&state, 0, COLD_KEY_BYTES, 'c');
        let target = wide_key_for_partition(&state, 1, TARGET_KEY_BYTES, 't');
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();
        *state.get_or_insert_with_accounted(cold, 0, || 10).unwrap() += 1;
        state.partition_sizes[1] = usize::MAX;
        let resident_before = state.granted_bytes();
        let headroom = serialized_target
            .0
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_sub(1))
            .unwrap();
        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available().checked_sub(headroom).unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let allocated_before = buffer_manager.allocated();
        let declared = AtomicBool::new(false);
        let built = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            target.clone(),
            |_| {
                declared.store(true, Ordering::Release);
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 64,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(1)
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(
                ref error
            ))) if error.kind() == std::io::ErrorKind::Other
        ));
        assert!(!declared.load(Ordering::Acquire));
        assert!(!built.load(Ordering::Acquire));
        let default_called = AtomicBool::new(false);
        let legacy_error = state
            .get_or_insert_with_accounted(target, 0, || {
                default_called.store(true, Ordering::Release);
                1
            })
            .unwrap_err();
        assert!(
            matches!(legacy_error, PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::Other
                    && error.to_string().contains("partition entry count overflow"))
        );
        assert!(!default_called.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert!(
            state.partitions[0]
                .as_ref()
                .is_some_and(|partition| !partition.is_empty()),
            "the cold non-target must remain resident"
        );
        assert!(state.partitions[1].as_ref().unwrap().is_empty());
        assert_eq!(state.partition_sizes[1], usize::MAX);
        assert_eq!(buffer_manager.allocated(), allocated_before);
        assert_eq!(state.granted_bytes(), resident_before);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn resident_update_underdeclaration_does_not_publish_absent_value() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_bytes_state(
            spill_manager,
            &buffer_manager,
            crate::execution::QueryExecutionControl::new().token(),
        );

        let result = state.try_replace_accounted(
            key(&[9]),
            |old| {
                assert!(old.is_none());
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 128,
                })
            },
            |_| Ok(vec![1; 32]),
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::RetainedCapacityExceeded {
                retained_upper_bound: 0,
                observed_retained
            }) if observed_retained >= 32
        ));
        assert_eq!(state.total_size(), 0);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_update_cancellation_after_build_preserves_old_value() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let cancellation = control.cancellation_handle();
        let mut state = accounted_bytes_state(spill_manager, &buffer_manager, control.token());
        let lookup = key(&[4]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 16, || vec![1; 16])
            .unwrap();

        let result = state.try_replace_accounted(
            lookup.clone(),
            |_| {
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 32,
                    construction_peak: 128,
                })
            },
            |_| {
                cancellation.cancel();
                Ok(vec![2; 32])
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(
                PartitionOperationError::Cancelled(_)
            ))
        ));
        assert_eq!(state.get(&lookup).unwrap(), Some(&vec![1; 16]));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_update_cancellation_after_declaration_skips_builder() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let cancellation = control.cancellation_handle();
        let mut state = accounted_bytes_state(spill_manager, &buffer_manager, control.token());
        let lookup = key(&[14]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 16, || vec![1; 16])
            .unwrap();
        let resident = state.granted_bytes();
        let built = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            lookup.clone(),
            |_| {
                cancellation.cancel();
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 32,
                    construction_peak: 128,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(vec![2; 32])
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(
                PartitionOperationError::Cancelled(_)
            ))
        ));
        assert!(!built.load(Ordering::Acquire));
        assert_eq!(state.get(&lookup).unwrap(), Some(&vec![1; 16]));
        assert_eq!(buffer_manager.allocated(), resident);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn resident_update_denial_preserves_old_value_without_building() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[5]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 17)
            .unwrap();
        let filler = RefCell::new(None);
        let built = AtomicBool::new(false);
        let construction_peak = 4096;

        let result = state.try_replace_accounted(
            lookup.clone(),
            |_| {
                let leave_available = construction_peak - 1;
                let fill = buffer_manager.available() - leave_available;
                filler.replace(Some(
                    buffer_manager
                        .try_allocate(fill, MemoryRegion::ExecutionBuffers)
                        .unwrap(),
                ));
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(99)
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(
                ref error
            ))) if error.kind() == std::io::ErrorKind::OutOfMemory
        ));
        assert!(!built.load(Ordering::Acquire));
        assert_eq!(state.get(&lookup).unwrap(), Some(&17));
        drop(filler.into_inner());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_update_successfully_replaces_value_with_exact_grant() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_bytes_state(
            spill_manager,
            &buffer_manager,
            crate::execution::QueryExecutionControl::new().token(),
        );
        let lookup = key(&[3]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 64, || vec![1; 64])
            .unwrap();

        state
            .try_replace_accounted(
                lookup.clone(),
                |old| {
                    assert_eq!(old.map(Vec::len), Some(64));
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 32,
                        construction_peak: 128,
                    })
                },
                |old| {
                    assert_eq!(old.map(Vec::len), Some(64));
                    Ok(vec![2; 32])
                },
            )
            .unwrap();

        assert_eq!(state.get(&lookup).unwrap(), Some(&vec![2; 32]));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_update_successfully_inserts_absent_value_with_exact_grant() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_bytes_state(
            spill_manager,
            &buffer_manager,
            crate::execution::QueryExecutionControl::new().token(),
        );
        let lookup = key(&[6]);

        state
            .try_replace_accounted(
                lookup.clone(),
                |old| {
                    assert!(old.is_none());
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 24,
                        construction_peak: 128,
                    })
                },
                |old| {
                    assert!(old.is_none());
                    Ok(vec![3; 24])
                },
            )
            .unwrap();

        assert_eq!(state.get(&lookup).unwrap(), Some(&vec![3; 24]));
        assert_eq!(state.total_size(), 1);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_present_base_publishes_complete_delta_override() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let serialized = SerializedKey::from_values(&lookup, state.frame_limits).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let base_bytes = state.spill_files[0].as_ref().unwrap().bytes_written();

        state
            .try_replace_accounted(
                lookup,
                |old| {
                    assert_eq!(old, Some(&10));
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |old| {
                    assert_eq!(old, Some(&10));
                    Ok(16)
                },
            )
            .unwrap();

        let base = state.spill_files[0].as_ref().unwrap();
        assert_eq!(base.identity(), base_identity);
        assert_eq!(base.bytes_written(), base_bytes);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        let delta = state.partitions[0].as_ref().unwrap();
        assert_eq!(delta.len(), 1);
        assert_eq!(delta.get(&serialized).map(|entry| entry.value), Some(16));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_resident_delta_override_skips_base_scan() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::with_failure(
            super::super::SpillIoOperation::ReadOpen,
            1,
            control.cancellation_handle(),
            std::io::ErrorKind::PermissionDenied,
            "resident delta must bypass its immutable base",
        ));
        io.disarm();
        let spill_manager = cancellation_manager(&directory, Arc::clone(&io));
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            |_value: &i64| Ok(0),
            grant,
            control.token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 99)
            .unwrap() += 5;
        let serialized = SerializedKey::from_values(&lookup, state.frame_limits).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        io.arm();

        state
            .try_replace_accounted(
                lookup,
                |old| {
                    assert_eq!(old, Some(&15));
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |old| {
                    assert_eq!(old, Some(&15));
                    Ok(21)
                },
            )
            .unwrap();

        assert_eq!(io.matching.load(Ordering::Acquire), 0);
        assert!(!control.token().is_cancelled());
        assert_eq!(
            state.spill_files[0].as_ref().unwrap().identity(),
            base_identity
        );
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert_eq!(
            state.partitions[0]
                .as_ref()
                .unwrap()
                .get(&serialized)
                .map(|entry| entry.value),
            Some(21)
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_absent_base_increments_only_logical_count() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let base_key = key(&[1]);
        state
            .get_or_insert_with_accounted(base_key, 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let lookup = key(&[2]);
        let serialized = SerializedKey::from_values(&lookup, state.frame_limits).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let base_bytes = state.spill_files[0].as_ref().unwrap().bytes_written();

        state
            .try_replace_accounted(
                lookup,
                |old| {
                    assert!(old.is_none());
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |old| {
                    assert!(old.is_none());
                    Ok(20)
                },
            )
            .unwrap();

        let base = state.spill_files[0].as_ref().unwrap();
        assert_eq!(base.identity(), base_identity);
        assert_eq!(base.bytes_written(), base_bytes);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 2);
        let delta = state.partitions[0].as_ref().unwrap();
        assert_eq!(delta.len(), 1);
        assert_eq!(delta.get(&serialized).map(|entry| entry.value), Some(20));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_cancellation_after_build_preserves_base() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let cancellation = control.cancellation_handle();
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            |_value: &i64| Ok(0),
            grant,
            control.token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let base_bytes = state.spill_files[0].as_ref().unwrap().bytes_written();

        let result = state.try_replace_accounted(
            lookup,
            |old| {
                assert_eq!(old, Some(&10));
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |old| {
                assert_eq!(old, Some(&10));
                cancellation.cancel();
                Ok(16)
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(
                PartitionOperationError::Cancelled(_)
            ))
        ));
        let base = state.spill_files[0].as_ref().unwrap();
        assert_eq!(base.identity(), base_identity);
        assert_eq!(base.bytes_written(), base_bytes);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_read_error_wins_when_read_also_cancels() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::with_failure(
            super::super::SpillIoOperation::ReadPayload,
            1,
            control.cancellation_handle(),
            std::io::ErrorKind::PermissionDenied,
            "deterministic spilled update read failure",
        ));
        io.disarm();
        let spill_manager = cancellation_manager(&directory, Arc::clone(&io));
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            |_value: &i64| Ok(0),
            grant,
            control.token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let built = AtomicBool::new(false);
        io.arm();

        let result = state.try_replace_accounted(
            lookup,
            |_| {
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(16)
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(
                ref error
            ))) if error.kind() == std::io::ErrorKind::PermissionDenied
        ));
        assert!(control.token().is_cancelled());
        assert!(!built.load(Ordering::Acquire));
        assert_eq!(
            state.spill_files[0].as_ref().unwrap().identity(),
            base_identity
        );
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn spilled_update_provisional_decode_envelope_spills_once_then_retries() {
        const COLD_KEY_BYTES: usize = 64 * 1024;
        const STORED_VALUE_BOUND: usize = 512 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        // Nested wire data deliberately inflates the current broad key
        // envelope. Lookup does not materialize these typed Values; this is an
        // admission/retry witness, not evidence of exact retained key demand.
        let target = vec![Value::List(Arc::from(vec![Value::Null; 17]))];
        let target_partition = state.partition_for(&target);
        let cold_partition = 1 - target_partition;
        let cold = wide_key_for_partition(&state, cold_partition, COLD_KEY_BYTES, 'c');
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();
        let serialized_cold = SerializedKey::from_values(&cold, state.frame_limits).unwrap();
        state
            .get_or_insert_with_accounted(target.clone(), STORED_VALUE_BOUND, || 10)
            .unwrap();
        state.spill_partition_controlled(target_partition).unwrap();
        state.get_or_insert_with_accounted(cold, 0, || 20).unwrap();
        let base_identity = state.spill_files[target_partition]
            .as_ref()
            .unwrap()
            .identity();
        let base_bytes = state.spill_files[target_partition]
            .as_ref()
            .unwrap()
            .bytes_written();

        let decoded_key_bound =
            PartitionedState::<i64>::decoded_key_resident_bound(serialized_target.0.len()).unwrap();
        let decode_envelope = decoded_key_bound.checked_add(STORED_VALUE_BOUND).unwrap();
        let headroom = decode_envelope.checked_sub(1).unwrap();
        let cold_record = serialized_cold
            .0
            .len()
            .checked_add(32 + std::mem::size_of::<i64>())
            .unwrap();
        let spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(cold_record.checked_mul(4).unwrap())
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        assert!(spill_peak < headroom);
        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available().checked_sub(headroom).unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();

        state
            .try_replace_accounted(
                target,
                |old| {
                    assert_eq!(old, Some(&10));
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |old| {
                    assert_eq!(old, Some(&10));
                    Ok(16)
                },
            )
            .unwrap();

        let base = state.spill_files[target_partition].as_ref().unwrap();
        assert_eq!(base.identity(), base_identity);
        assert_eq!(base.bytes_written(), base_bytes);
        assert_eq!(state.spill_base_sizes[target_partition], 1);
        assert_eq!(state.partition_sizes[target_partition], 1);
        assert_eq!(state.spill_base_sizes[cold_partition], 1);
        assert_eq!(state.partition_sizes[cold_partition], 1);
        assert_eq!(state.spilled_count(), 2);
        assert_eq!(
            state.partitions[target_partition]
                .as_ref()
                .unwrap()
                .get(&serialized_target)
                .map(|entry| entry.value),
            Some(16)
        );
        assert!(
            state.partitions[cold_partition]
                .as_ref()
                .unwrap()
                .is_empty()
        );
        assert!(buffer_manager.allocated() <= buffer_manager.budget());
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_reader_workspace_admission_spills_once_then_restarts_scan() {
        const COLD_KEY_BYTES: usize = 128 * 1024;
        const HEADROOM: usize = 1024 * 1024;
        const RAISED_READER_BOUND: usize = 2 * 1024 * 1024;

        let directory = TempDir::new().unwrap();
        let provider = Arc::new(OneRaisedWorkspaceBoundProvider::new());
        let spill_manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(provider.clone(), SpillFrameLimits::format_max())
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(8 * 1024 * 1024);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        let target = key(&[1]);
        let target_partition = state.partition_for(&target);
        let cold_partition = 1 - target_partition;
        let cold = wide_key_for_partition(&state, cold_partition, COLD_KEY_BYTES, 'w');
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();
        state
            .get_or_insert_with_accounted(target.clone(), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(target_partition).unwrap();
        state.get_or_insert_with_accounted(cold, 0, || 20).unwrap();
        let base_identity = state.spill_files[target_partition]
            .as_ref()
            .unwrap()
            .identity();

        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available().checked_sub(HEADROOM).unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        provider.raise_next_bound_to(RAISED_READER_BOUND);

        state
            .try_replace_accounted(
                target,
                |old| {
                    assert_eq!(old, Some(&10));
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |old| {
                    assert_eq!(old, Some(&10));
                    Ok(16)
                },
            )
            .unwrap();

        assert_eq!(provider.next_bound.load(Ordering::Acquire), 0);
        assert_eq!(state.spilled_count(), 2);
        assert_eq!(state.spill_base_sizes[cold_partition], 1);
        assert!(
            state.partitions[cold_partition]
                .as_ref()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            state.spill_files[target_partition]
                .as_ref()
                .unwrap()
                .identity(),
            base_identity
        );
        assert_eq!(
            state.partitions[target_partition]
                .as_ref()
                .unwrap()
                .get(&serialized_target)
                .map(|entry| entry.value),
            Some(16)
        );
        assert!(buffer_manager.allocated() <= buffer_manager.budget());
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_post_match_frame_denial_never_replays_decoder() {
        const LATER_KEY_BYTES: usize = 512 * 1024;
        const HEADROOM: usize = 256 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(8 * 1024 * 1024);
        let decoder_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&decoder_calls);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&spill_manager),
            2,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            move |reader: &mut dyn Read, _limits| {
                calls.fetch_add(1, Ordering::AcqRel);
                deserialize_i64(reader)
            },
            |_value: &i64| Ok(0),
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let target_partition = 0;
        let cold_partition = 1;
        let target = key_for_partition(&state, target_partition);
        let later = wide_key_for_partition(&state, target_partition, LATER_KEY_BYTES, 'l');
        let cold = key_for_partition(&state, cold_partition);
        let target_key = SerializedKey::from_values(&target, state.frame_limits).unwrap();
        let later_key = SerializedKey::from_values(&later, state.frame_limits).unwrap();

        let mut target_payload = encode_partition_entry(&target_key.0, 1, &10_i64.to_le_bytes());
        target_payload.extend_from_slice(&0_u64.to_le_bytes());
        let mut later_payload = encode_partition_entry(&later_key.0, 1, &20_i64.to_le_bytes());
        later_payload.extend_from_slice(&0_u64.to_le_bytes());
        let mut file = spill_manager
            .create_file(SpillFileRole::NativePartition)
            .unwrap();
        file.write_partition_start(2).unwrap();
        file.write_partition_entry(&target_payload).unwrap();
        file.write_partition_entry(&later_payload).unwrap();
        file.finish_write().unwrap();
        state.spill_files[target_partition] = Some(file);
        state.spill_base_sizes[target_partition] = 2;
        state.partition_sizes[target_partition] = 2;
        state.get_or_insert_with_accounted(cold, 0, || 30).unwrap();

        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available().checked_sub(HEADROOM).unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let declared = AtomicBool::new(false);
        let result = state.try_replace_accounted(
            target,
            |_| {
                declared.store(true, Ordering::Release);
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |_| Ok(16),
        );

        let error = match result {
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(error)))
                if error.kind() == std::io::ErrorKind::OutOfMemory =>
            {
                error
            }
            other => panic!("post-match frame denial was not preserved: {other:?}"),
        };
        assert_eq!(decoder_calls.load(Ordering::Acquire), 1);
        assert!(!declared.load(Ordering::Acquire));
        assert_eq!(state.spill_base_sizes[target_partition], 2);
        assert_eq!(state.partition_sizes[target_partition], 2);
        assert_eq!(state.spill_base_sizes[cold_partition], 0);
        assert_eq!(state.partition_sizes[cold_partition], 1);
        assert!(
            state.partitions[target_partition]
                .as_ref()
                .unwrap()
                .is_empty()
        );
        let retained_without_reader = state.granted_bytes() + filler.size();
        assert!(
            buffer_manager.allocated() > retained_without_reader,
            "the returned frame-denial error must retain live reader workspace authority"
        );
        drop(error);
        assert_eq!(buffer_manager.allocated(), retained_without_reader);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_lookup_provider_error_retains_reader_authority_until_payload_drops() {
        let directory = TempDir::new().unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let provider = Arc::new(HeapErrorOnReaderPartitionStartProvider {
            begins: AtomicUsize::new(0),
            buffer_manager: Arc::clone(&buffer_manager),
            allocated_when_dropped: Arc::clone(&allocated_when_dropped),
        });
        let spill_manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(provider, SpillFrameLimits::format_max())
                .build()
                .unwrap(),
        );
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let declared = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            lookup,
            |_| {
                declared.store(true, Ordering::Release);
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |_| Ok(11),
        );
        let error = match result {
            Err(PartitionUpdateError::Partition(error)) => error,
            other => panic!("heap-bearing reader error was not preserved: {other:?}"),
        };

        assert!(!declared.load(Ordering::Acquire));
        assert_eq!(allocated_when_dropped.load(Ordering::Acquire), usize::MAX);
        let allocated_with_error = buffer_manager.allocated();
        assert!(
            allocated_with_error > state.granted_bytes(),
            "reader and record workspace must remain live with the provider error"
        );
        drop(error);
        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            allocated_with_error,
            "physical provider error must drop before reader workspace authority"
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_decode_error_retains_authority_until_heap_payload_drops() {
        const DECODE_AUTHORITY: usize = 64 * 1024;
        const ERROR_BYTES: usize = 32 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let decoder_manager = Arc::clone(&buffer_manager);
        let decoder_drop_observation = Arc::clone(&allocated_when_dropped);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            move |_reader: &mut dyn Read, _limits| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    HeapDropProbe {
                        bytes: vec![0xde; ERROR_BYTES],
                        buffer_manager: Arc::clone(&decoder_manager),
                        allocated_when_dropped: Arc::clone(&decoder_drop_observation),
                    },
                ))
            },
            |_value: &i64| Ok(0),
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), DECODE_AUTHORITY, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let base_bytes = state.spill_files[0].as_ref().unwrap().bytes_written();
        let declared = AtomicBool::new(false);
        let built = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            lookup,
            |_| {
                declared.store(true, Ordering::Release);
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(16)
            },
        );
        let error = match result {
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(error))) => error,
            other => panic!("heap-bearing decode error was not preserved: {other:?}"),
        };

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        let reader_failure = error
            .get_ref()
            .and_then(|source| source.downcast_ref::<AccountedPartitionIoFailure>())
            .expect("decode error must retain its reader authority");
        assert!(reader_failure.reader_authority.is_some());
        assert!(reader_failure.grant_authority.is_none());
        let decode_error = std::error::Error::source(reader_failure)
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .expect("reader-accounted failure must preserve the decode error");
        assert_eq!(decode_error.kind(), std::io::ErrorKind::PermissionDenied);
        let decode_failure = decode_error
            .get_ref()
            .and_then(|source| source.downcast_ref::<AccountedPartitionIoFailure>())
            .expect("decode error must retain its child decode authority");
        assert!(decode_failure.reader_authority.is_none());
        assert!(decode_failure.grant_authority.is_some());
        let original = std::error::Error::source(decode_failure)
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .expect("decode-accounted failure must preserve the original decoder error");
        assert_eq!(original.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            original
                .get_ref()
                .and_then(|source| source.downcast_ref::<HeapDropProbe>())
                .map(|probe| probe.bytes.len()),
            Some(ERROR_BYTES)
        );
        assert!(!declared.load(Ordering::Acquire));
        assert!(!built.load(Ordering::Acquire));
        let allocated_with_error = buffer_manager.allocated();
        assert!(allocated_with_error > state.granted_bytes());
        let base = state.spill_files[0].as_ref().unwrap();
        assert_eq!(base.identity(), base_identity);
        assert_eq!(base.bytes_written(), base_bytes);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());

        drop(error);
        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            allocated_with_error,
            "physical decoder error must drop before its decode authority"
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_capacity_panic_retains_authority_until_heap_payload_drops() {
        const DECODE_AUTHORITY: usize = 64 * 1024;
        const PANIC_BYTES: usize = 32 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let panic_enabled = Arc::new(AtomicBool::new(false));
        let capacity_panic = Arc::clone(&panic_enabled);
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let panic_manager = Arc::clone(&buffer_manager);
        let panic_drop_observation = Arc::clone(&allocated_when_dropped);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            move |_value: &i64| {
                if capacity_panic.load(Ordering::Acquire) {
                    std::panic::panic_any(HeapDropProbe {
                        bytes: vec![0xca; PANIC_BYTES],
                        buffer_manager: Arc::clone(&panic_manager),
                        allocated_when_dropped: Arc::clone(&panic_drop_observation),
                    });
                }
                Ok(0)
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), DECODE_AUTHORITY, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let base_bytes = state.spill_files[0].as_ref().unwrap().bytes_written();
        let declared = AtomicBool::new(false);
        let built = AtomicBool::new(false);
        panic_enabled.store(true, Ordering::Release);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.try_replace_accounted(
                lookup,
                |_| {
                    declared.store(true, Ordering::Release);
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |_| {
                    built.store(true, Ordering::Release);
                    Ok(16)
                },
            );
        }))
        .expect_err("capacity callback must unwind");

        let accounted_panic = panic
            .downcast_ref::<AccountedPartitionPanic>()
            .expect("capacity panic must retain its decode authority");
        assert_eq!(
            accounted_panic
                .payload()
                .downcast_ref::<HeapDropProbe>()
                .map(|payload| payload.bytes.len()),
            Some(PANIC_BYTES)
        );
        assert!(!declared.load(Ordering::Acquire));
        assert!(!built.load(Ordering::Acquire));
        let allocated_with_panic = buffer_manager.allocated();
        assert!(allocated_with_panic > state.granted_bytes());
        let base = state.spill_files[0].as_ref().unwrap();
        assert_eq!(base.identity(), base_identity);
        assert_eq!(base.bytes_written(), base_bytes);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());

        panic_enabled.store(false, Ordering::Release);
        drop(panic);
        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            allocated_with_panic,
            "physical panic payload must drop before its decode authority"
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_reader_drop_cannot_replace_accounted_decode_panic() {
        const CHILD_ENV: &str = "GRAFEO_PARTITION_READER_DROP_PANIC_CHILD";
        const HANDSHAKE: &str = "GRAFEO_PARTITION_READER_DROP_PANIC_OK";
        const PRIMARY_PAYLOAD: u64 = 0xacce_5510;

        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let provider = Arc::new(PanicOnReaderOpenRecordDropProvider::new());
            let spill_manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(provider.clone(), SpillFrameLimits::format_max())
                    .build()
                    .unwrap(),
            );
            let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
            let panic_enabled = Arc::new(AtomicBool::new(false));
            let capacity_panic = Arc::clone(&panic_enabled);
            let grant = buffer_manager
                .try_allocate(0, MemoryRegion::ExecutionBuffers)
                .unwrap();
            let mut state = PartitionedState::new_accounted_with_cancellation(
                Arc::clone(&spill_manager),
                1,
                |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
                |reader: &mut dyn Read, _limits| deserialize_i64(reader),
                move |_value: &i64| {
                    if capacity_panic.load(Ordering::Acquire) {
                        std::panic::panic_any(PRIMARY_PAYLOAD);
                    }
                    Ok(0)
                },
                grant,
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
            let lookup = key(&[1]);
            state
                .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
                .unwrap();
            state.spill_partition_controlled(0).unwrap();
            panic_enabled.store(true, Ordering::Release);

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = state.try_replace_accounted(
                    lookup,
                    |_| {
                        Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                            retained_upper_bound: 0,
                            construction_peak: 0,
                        })
                    },
                    |_| Ok(16),
                );
            }))
            .expect_err("capacity callback must unwind through the hostile reader");

            let accounted = panic
                .downcast_ref::<AccountedPartitionPanic>()
                .expect("reader cleanup must preserve the accounted primary panic");
            assert_eq!(
                accounted.payload().downcast_ref::<u64>(),
                Some(&PRIMARY_PAYLOAD)
            );
            assert_eq!(provider.reader_drops.load(Ordering::Acquire), 1);
            assert_eq!(state.spill_base_sizes[0], 1);
            assert_eq!(state.partition_sizes[0], 1);
            assert!(state.partitions[0].as_ref().unwrap().is_empty());

            panic_enabled.store(false, Ordering::Release);
            drop(panic);
            assert_eq!(buffer_manager.allocated(), state.granted_bytes());
            assert_eq!(
                state.granted_bytes(),
                state.observed_resident_capacity_bytes().unwrap()
            );
            state.cleanup().unwrap();
            drop(state);
            assert_eq!(buffer_manager.allocated(), 0);
            println!("{HANDSHAKE}");
            return;
        }

        let test_name = "execution::spill::partition::tests::hostile_reader_drop_cannot_replace_accounted_decode_panic";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(HANDSHAKE),
            "reader-drop child did not preserve the accounted decode panic\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[test]
    fn spilled_update_builder_panic_preserves_payload_without_reconcile() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let panic_on_capacity = Arc::new(AtomicBool::new(false));
        let callback_guard = Arc::clone(&panic_on_capacity);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            move |_value: &i64| {
                assert!(
                    !callback_guard.load(Ordering::Acquire),
                    "spilled builder-panic rollback must not reconcile callbacks"
                );
                Ok(0)
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.try_replace_accounted(
                lookup,
                |_| {
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |_| -> Result<i64, std::convert::Infallible> {
                    panic_on_capacity.store(true, Ordering::Release);
                    std::panic::panic_any(0x51_11ed_u64)
                },
            );
        }))
        .expect_err("spilled replacement builder must unwind");

        let accounted_panic = panic
            .downcast_ref::<AccountedPartitionPanic>()
            .expect("builder panic must retain its construction authority");
        assert_eq!(
            accounted_panic.payload().downcast_ref::<u64>(),
            Some(&0x51_11ed_u64)
        );
        panic_on_capacity.store(false, Ordering::Release);
        drop(panic);
        assert_eq!(
            state.spill_files[0].as_ref().unwrap().identity(),
            base_identity
        );
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_map_denial_preserves_base() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let filler = RefCell::new(None);
        let built = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            key(&[2]),
            |old| {
                assert!(old.is_none());
                filler.replace(Some(
                    buffer_manager
                        .try_allocate(buffer_manager.available(), MemoryRegion::ExecutionBuffers)
                        .unwrap(),
                ));
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |old| {
                assert!(old.is_none());
                built.store(true, Ordering::Release);
                Ok(20)
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(
                ref error
            ))) if error.kind() == std::io::ErrorKind::OutOfMemory
        ));
        assert!(built.load(Ordering::Acquire));
        assert_eq!(
            state.spill_files[0].as_ref().unwrap().identity(),
            base_identity
        );
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        drop(filler.into_inner());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_rejects_key_column_mismatch_before_callbacks() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let decoded = Arc::new(AtomicBool::new(false));
        let decoded_callback = Arc::clone(&decoded);
        let capacity_observed = Arc::new(AtomicBool::new(false));
        let capacity_callback = Arc::clone(&capacity_observed);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&manager),
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            move |reader: &mut dyn Read, _limits| {
                decoded_callback.store(true, Ordering::Release);
                deserialize_i64(reader)
            },
            move |_value: &i64| {
                capacity_callback.store(true, Ordering::Release);
                Ok(0)
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        let serialized = SerializedKey::from_values(&lookup, state.frame_limits).unwrap();
        let mut payload = encode_partition_entry(&serialized.0, 2, &10i64.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        install_partition_file(&mut state, &manager, &[payload]);
        state.partitions[0] = Some(new_partition_map());
        state.spill_base_sizes[0] = 1;
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let declared = AtomicBool::new(false);
        let built = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            lookup,
            |_| {
                declared.store(true, Ordering::Release);
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(20)
            },
        );

        let error = match result {
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(error)))
                if error.kind() == std::io::ErrorKind::InvalidData =>
            {
                error
            }
            other => panic!("key-column mismatch was not preserved: {other:?}"),
        };
        assert!(!decoded.load(Ordering::Acquire));
        assert!(!capacity_observed.load(Ordering::Acquire));
        assert!(!declared.load(Ordering::Acquire));
        assert!(!built.load(Ordering::Acquire));
        assert_eq!(
            state.spill_files[0].as_ref().unwrap().identity(),
            base_identity
        );
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        assert!(
            buffer_manager.allocated() > state.granted_bytes(),
            "the returned mismatch error must retain live reader workspace authority"
        );
        drop(error);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_update_underdeclaration_preserves_immutable_base() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_bytes_state(
            spill_manager,
            &buffer_manager,
            crate::execution::QueryExecutionControl::new().token(),
        );
        let lookup = key(&[1]);
        state
            .get_or_insert_with_accounted(lookup.clone(), 0, Vec::new)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        let base_identity = state.spill_files[0].as_ref().unwrap().identity();
        let base_bytes = state.spill_files[0].as_ref().unwrap().bytes_written();

        let result = state.try_replace_accounted(
            lookup,
            |old| {
                assert_eq!(old.map(Vec::len), Some(0));
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 128,
                })
            },
            |old| {
                assert_eq!(old.map(Vec::len), Some(0));
                Ok(vec![1; 32])
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::RetainedCapacityExceeded {
                retained_upper_bound: 0,
                observed_retained
            }) if observed_retained >= 32
        ));
        let base = state.spill_files[0].as_ref().unwrap();
        assert_eq!(base.identity(), base_identity);
        assert_eq!(base.bytes_written(), base_bytes);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partition_sizes[0], 1);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[derive(Default)]
    struct RecoveryIo {
        calls: AtomicUsize,
        creates: AtomicUsize,
        deletes: AtomicUsize,
        fail_delete: AtomicBool,
        cancel_on_delete: Option<crate::execution::QueryCancellationHandle>,
    }

    impl super::super::SpillIo for RecoveryIo {
        fn check(&self, operation: super::super::SpillIoOperation) -> std::io::Result<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if operation == super::super::SpillIoOperation::Create {
                self.creates.fetch_add(1, Ordering::Relaxed);
            }
            if operation == super::super::SpillIoOperation::Delete {
                self.deletes.fetch_add(1, Ordering::Relaxed);
                if let Some(cancellation) = &self.cancel_on_delete {
                    cancellation.cancel();
                }
                if self.fail_delete.load(Ordering::Acquire) {
                    return Err(std::io::ErrorKind::PermissionDenied.into());
                }
            }
            Ok(())
        }

        fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
            Some(0)
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            // Reader phases only increment the atomic call counter. Failure
            // and cancellation injection are restricted to Delete above.
            Some(0)
        }
    }

    fn protected_recovery_bytes_state(
        manager: Arc<SpillManager>,
        buffer_manager: &Arc<BufferManager>,
    ) -> PartitionedState<Vec<u8>> {
        let mut state = PartitionedState::new_accounted_admitted_with_cancellation(
            manager,
            3,
            |value: &Vec<u8>, writer: &mut dyn Write, _limits| writer.write_all(value),
            |reader: &mut dyn Read, _limits| {
                let mut value = Vec::new();
                reader.read_to_end(&mut value)?;
                Ok(value)
            },
            |value: &Vec<u8>| Ok(value.capacity()),
            buffer_manager
                .try_allocate(0, MemoryRegion::ExecutionBuffers)
                .unwrap(),
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        for (index, bytes) in [64, 4096].into_iter().enumerate() {
            let lookup = key_for_partition(&state, index);
            state
                .try_update_accounted(
                    std::mem::size_of::<Value>(),
                    || Ok(lookup),
                    |_| {
                        Ok((
                            PartitionUpdateAdmission {
                                retained_upper_bound: bytes,
                                construction_peak: bytes,
                            },
                            0,
                        ))
                    },
                    || Ok(vec![0x61; bytes]),
                    |_| Ok(()),
                )
                .unwrap();
        }
        state
    }

    #[test]
    fn protected_aggregate_map_admission_preserves_observed_coverage_without_rescanning() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 << 20);
        let inspections = Arc::new(AtomicUsize::new(0));
        let observed_inspections = Arc::clone(&inspections);
        let mut state = PartitionedState::new_accounted_admitted_with_cancellation(
            manager,
            1,
            |value: &Vec<u8>, writer: &mut dyn Write, _limits| {
                writer.write_all(&u64::try_from(value.len()).unwrap().to_le_bytes())?;
                writer.write_all(value)
            },
            |reader: &mut dyn Read, _limits| {
                let mut header = [0; 8];
                reader.read_exact(&mut header)?;
                let len = usize::try_from(u64::from_le_bytes(header)).unwrap();
                let mut value = vec![0; len];
                reader.read_exact(&mut value)?;
                Ok(value)
            },
            move |value: &Vec<u8>| {
                observed_inspections.fetch_add(1, Ordering::Relaxed);
                Ok(value.capacity())
            },
            buffer_manager
                .try_allocate(0, MemoryRegion::ExecutionBuffers)
                .unwrap(),
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let expected_live = std::cell::Cell::new(None);
        let check_live = || {
            if let Some(expected) = expected_live.get() {
                assert_eq!(buffer_manager.allocated(), expected);
            }
        };
        let update = |state: &mut PartitionedState<Vec<u8>>, group: i64, bytes: usize, fill: u8| {
            state
                .try_update_accounted(
                    std::mem::size_of::<Value>(),
                    || {
                        check_live();
                        Ok(key(&[group]))
                    },
                    |_| {
                        check_live();
                        Ok((
                            PartitionUpdateAdmission {
                                retained_upper_bound: bytes,
                                construction_peak: bytes * 2,
                            },
                            bytes * 4,
                        ))
                    },
                    || Ok(Vec::new()),
                    |value| {
                        check_live();
                        *value = vec![fill; bytes];
                        Ok(())
                    },
                )
                .unwrap();
        };
        let assert_exact_coverage = |state: &PartitionedState<Vec<u8>>| {
            assert_eq!(
                state.grant.as_ref().unwrap().size(),
                state.observed_resident_capacity_bytes().unwrap()
            );
            assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        };

        // Repeated native-map growth must inspect only the newly built value,
        // independently of how many earlier groups remain resident.
        for group in 0..32 {
            inspections.store(0, Ordering::Relaxed);
            update(&mut state, group, 64, 1);
            assert_eq!(inspections.load(Ordering::Relaxed), 1);
            assert_exact_coverage(&state);
        }
        update(&mut state, 0, 64, 1);
        let idle = state.recovery_allowance.as_ref().unwrap().size();
        let working = state.recovery_working_bytes;
        expected_live.set(Some(buffer_manager.allocated()));
        for _ in 0..128 {
            update(&mut state, 0, 64, 1);
            assert_eq!(state.recovery_allowance.as_ref().unwrap().size(), idle);
            assert_eq!(state.recovery_working_bytes, working);
            assert_exact_coverage(&state);
        }
        expected_live.set(None);
        update(&mut state, 0, 128, 2);
        assert_exact_coverage(&state);
        update(&mut state, 0, 32, 3);
        assert_exact_coverage(&state);
        state.spill_partition_controlled(0).unwrap();
        assert_exact_coverage(&state);
        let base = state.spill_files[0].as_ref().unwrap().identity();

        // Revisit stages its base under a child; its map admission must not
        // count those bytes twice before publishing the changed value.
        update(&mut state, 0, 128, 4);
        assert_exact_coverage(&state);
        update(&mut state, 32, 64, 5);
        assert_exact_coverage(&state);
        state.spill_partition_controlled(0).unwrap();
        assert_ne!(state.spill_files[0].as_ref().unwrap().identity(), base);
        assert_eq!(state.spill_base_sizes[0], 33);
        assert_exact_coverage(&state);
        update(&mut state, 0, 32, 6);
        assert_exact_coverage(&state);

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let mut counts = [0; 7];
        while let Some(entry) = cursor.next_entry().unwrap() {
            let value = entry.value();
            assert!(value.iter().all(|byte| *byte == value[0]));
            counts[usize::from(value[0])] += 1;
            assert_eq!(value.len(), if value[0] == 6 { 32 } else { 64 });
        }
        drop(cursor);
        assert_eq!(counts, [0, 31, 0, 0, 0, 1, 1]);
        assert_exact_coverage(&state);
        drop(state);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn protected_row_workspace_failure_retains_lent_authority() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        let error = state
            .try_update_accounted(
                4096,
                || Err(OperatorError::Execution("x".repeat(4096))),
                |_| unreachable!("key failure must not invoke declaration"),
                || unreachable!("key failure must not invoke construction"),
                |_| unreachable!("key failure must not invoke mutation"),
            )
            .unwrap_err();
        drop(state);
        assert!(buffer_manager.allocated() >= 4096);
        drop(error);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn protected_covered_base_retires_only_for_final_drain() {
        for (full_coverage, pressure_spill) in [(true, false), (true, true), (false, false)] {
            let directory = TempDir::new().unwrap();
            let io = Arc::new(RecoveryIo::default());
            let manager = Arc::new(
                super::super::BorrowedSpillFixture::new(directory.path())
                    .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                    .build()
                    .unwrap(),
            );
            let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
            let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
            let lookup = key_for_partition(&state, 0);
            let other = wide_key_for_partition(&state, 0, 64, 'k');
            let update = |state: &mut PartitionedState<Vec<u8>>, key: Vec<Value>, fill| {
                state
                    .try_update_accounted(
                        256,
                        || Ok(key),
                        |_| {
                            Ok((
                                PartitionUpdateAdmission {
                                    retained_upper_bound: 64,
                                    construction_peak: 128,
                                },
                                256,
                            ))
                        },
                        || Ok(Vec::new()),
                        |value| {
                            *value = vec![fill; 64];
                            Ok(())
                        },
                    )
                    .unwrap();
            };
            update(&mut state, other.clone(), 0x62);
            state.spill_partition_controlled(0).unwrap();
            let old_base = state.spill_files[0].as_ref().unwrap().identity();
            update(&mut state, lookup, 0x63);
            if full_coverage {
                update(&mut state, other, 0x64);
            }
            assert_eq!(io.creates.load(Ordering::Relaxed), 1);
            assert_eq!(state.partition_sizes[0], 2);
            assert_eq!(state.spill_base_sizes[0], 2);
            if pressure_spill {
                let resident_before = state.grant.as_ref().unwrap().size();
                assert!(state.spill_partition_controlled(0).unwrap() > 0);
                assert!(state.partitions[0].as_ref().unwrap().is_empty());
                assert!(state.grant.as_ref().unwrap().size() < resident_before);
                assert_ne!(state.spill_files[0].as_ref().unwrap().identity(), old_base);
                assert_eq!(io.creates.load(Ordering::Relaxed), 2);
            }
            let mut cursor = state.drain_partitioned_accounted().unwrap();
            let expected_creates = if full_coverage && !pressure_spill {
                1
            } else {
                2
            };
            assert_eq!(io.creates.load(Ordering::Relaxed), expected_creates);
            let mut rows = Vec::new();
            while let Some(entry) = cursor.next_entry().unwrap() {
                assert!(entry.value().iter().all(|byte| *byte == entry.value()[0]));
                rows.push((entry.value().len(), entry.value()[0]));
            }
            drop(cursor);
            rows.sort_unstable();
            let mut expected = vec![
                (64, 0x63),
                (64, if full_coverage { 0x64 } else { 0x62 }),
                (4096, 0x61),
            ];
            expected.sort_unstable();
            assert_eq!(rows, expected);
            assert_eq!(io.deletes.load(Ordering::Relaxed), expected_creates);
            assert_eq!(state.total_size(), 0);
            assert_eq!(state.spilled_count(), 0);
            assert_eq!(buffer_manager.allocated(), state.granted_bytes());
            drop(state);
            assert_eq!(buffer_manager.allocated(), 0);
        }
    }

    #[test]
    fn protected_covered_base_delete_failure_retains_error_authority() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(RecoveryIo::default());
        let manager = Arc::new(
            super::super::BorrowedSpillFixture::new(directory.path())
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        let lookup = key_for_partition(&state, 0);
        state.spill_partition_controlled(0).unwrap();
        state
            .try_update_accounted(
                std::mem::size_of::<Value>(),
                || Ok(lookup),
                |_| {
                    Ok((
                        PartitionUpdateAdmission {
                            retained_upper_bound: 64,
                            construction_peak: 0,
                        },
                        256,
                    ))
                },
                || unreachable!("the spilled group already exists"),
                |value| {
                    value.fill(0x62);
                    Ok(())
                },
            )
            .unwrap();
        io.fail_delete.store(true, Ordering::Release);
        let Err(error) = state.drain_partitioned_accounted() else {
            panic!("covered-base deletion must fail");
        };
        assert!(matches!(error, PartitionOperationError::Accounted { .. }));
        assert_eq!(io.creates.load(Ordering::Relaxed), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partitions[0].as_ref().unwrap().len(), 1);
        io.fail_delete.store(false, Ordering::Release);
        drop(state);
        assert!(buffer_manager.allocated() > 0);
        drop(error);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn protected_covered_base_delete_cancellation_precedes_next_partition_callback() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(RecoveryIo {
            cancel_on_delete: Some(control.cancellation_handle()),
            ..RecoveryIo::default()
        });
        let manager = Arc::new(
            super::super::BorrowedSpillFixture::new(directory.path())
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        state.cancellation = Some(control.token());
        for (index, bytes) in [64, 4096].into_iter().enumerate() {
            let lookup = key_for_partition(&state, index);
            state.spill_partition_controlled(index).unwrap();
            state
                .try_update_accounted(
                    std::mem::size_of::<Value>(),
                    || Ok(lookup),
                    |_| {
                        Ok((
                            PartitionUpdateAdmission {
                                retained_upper_bound: bytes,
                                construction_peak: 0,
                            },
                            bytes * 4,
                        ))
                    },
                    || unreachable!("the spilled group already exists"),
                    |_| Ok(()),
                )
                .unwrap();
        }
        let inspections = Arc::new(AtomicUsize::new(0));
        let callback_inspections = Arc::clone(&inspections);
        state.value_resident_capacity = Box::new(move |value| {
            callback_inspections.fetch_add(1, Ordering::Relaxed);
            Ok(value.capacity())
        });
        let Err(error) = state.drain_partitioned_accounted() else {
            panic!("deletion cancellation must stop drain preparation");
        };
        assert!(matches!(
            error,
            PartitionOperationError::Accounted {
                classification: AccountedFailureClassification::QueryCancelled(
                    crate::execution::QueryCancellationError::Cancelled
                ),
                ..
            }
        ));
        assert_eq!(inspections.load(Ordering::Relaxed), 1);
        assert_eq!(io.creates.load(Ordering::Relaxed), 2);
        assert!(state.spill_files[0].is_none());
        assert_eq!(state.spill_base_sizes[0], 0);
        assert_eq!(state.partitions[0].as_ref().unwrap().len(), 1);
        drop(state);
        assert!(buffer_manager.allocated() > 0);
        drop(error);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[derive(Debug)]
    struct OpaqueQuotaSource {
        formatting_or_source_calls: Arc<AtomicUsize>,
        nested: SpillQuotaExceeded,
    }

    impl std::fmt::Display for OpaqueQuotaSource {
        fn fmt(&self, _out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.formatting_or_source_calls
                .fetch_add(1, Ordering::Relaxed);
            panic!("opaque provider Display must not run");
        }
    }

    impl std::error::Error for OpaqueQuotaSource {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.formatting_or_source_calls
                .fetch_add(1, Ordering::Relaxed);
            Some(&self.nested)
        }
    }

    fn assert_protected_context_failure_preserves_owner(
        primary: OperatorError,
        inspect_classification: impl FnOnce(&AccountedFailureClassification),
        with_cleanup: bool,
        diagnostic_first: bool,
    ) {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        let calls = Arc::new(AtomicUsize::new(0));
        let operator = primary
            .with_context("inner context")
            .with_context("outer context");
        let OperatorError::Context { source: inner, .. } = &operator else {
            panic!("fixture must wrap the primary twice");
        };
        let OperatorError::Context { source: leaf, .. } = inner.as_ref() else {
            panic!("fixture must retain the inner context");
        };
        let original = std::ptr::from_ref(leaf.as_ref());
        if with_cleanup {
            state
                .failure_cleanup
                .as_ref()
                .unwrap()
                .inspect::<PartitionFailureCleanup, _>(|witness| {
                    witness.retain_cleanup_error(std::io::Error::other(OpaqueQuotaSource {
                        formatting_or_source_calls: Arc::clone(&calls),
                        nested: SpillQuotaExceeded::from_usage(0, 0, 99),
                    }));
                })
                .unwrap();
        }

        let error = state.fail_operator(operator);
        let PartitionOperationError::Accounted {
            classification,
            authority,
        } = &error
        else {
            panic!("context failure must retain its accounted owner");
        };
        inspect_classification(classification);
        authority
            .inspect::<PartitionFailure, _>(|failure| {
                assert!(failure.primary.is_none());
                assert!(failure.panic.is_none());
                assert!(failure.cleanup_panic.is_none());
                let OperatorError::Context {
                    source: inner,
                    context,
                } = failure.operator.as_ref().unwrap()
                else {
                    panic!("outer context must remain owned");
                };
                assert_eq!(context, "outer context");
                let OperatorError::Context {
                    source: leaf,
                    context,
                } = inner.as_ref()
                else {
                    panic!("inner context must remain owned");
                };
                assert_eq!(context, "inner context");
                assert_eq!(std::ptr::from_ref(leaf.as_ref()), original);
                assert_eq!(failure.cleanup_error.is_some(), with_cleanup);
                if let Some(cleanup) = &failure.cleanup_error {
                    assert!(
                        cleanup
                            .get_ref()
                            .unwrap()
                            .downcast_ref::<OpaqueQuotaSource>()
                            .is_some()
                    );
                }
            })
            .unwrap();
        assert_eq!(error.to_string(), "accounted partition operation failed");
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        if diagnostic_first {
            drop(error);
            assert!(buffer_manager.allocated() > 0);
            drop(state);
        } else {
            drop(state);
            assert!(buffer_manager.allocated() > 0);
            drop(error);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn protected_context_cancellation_preserves_classification_and_authority() {
        use crate::execution::QueryCancellationError;

        for reason in [
            QueryCancellationError::Cancelled,
            QueryCancellationError::DeadlineExceeded { timeout: None },
            QueryCancellationError::DeadlineExceeded {
                timeout: Some(std::time::Duration::from_secs(30)),
            },
        ] {
            for with_cleanup in [false, true] {
                for diagnostic_first in [false, true] {
                    assert_protected_context_failure_preserves_owner(
                        OperatorError::QueryCancelled(reason),
                        |classification| {
                            assert!(matches!(
                                classification,
                                AccountedFailureClassification::QueryCancelled(actual)
                                    if *actual == reason
                            ));
                        },
                        with_cleanup,
                        diagnostic_first,
                    );
                }
            }
        }
    }

    #[test]
    fn protected_context_memory_preserves_classification_and_authority() {
        for reason in [
            MemoryGrantError::Denied {
                additional_bytes: 17,
            },
            MemoryGrantError::LimitExceeded {
                scope: grafeo_common::memory::buffer::MemoryLimitScope::Global,
                requested_bytes: 1025,
                limit_bytes: 1024,
            },
        ] {
            for with_cleanup in [false, true] {
                for diagnostic_first in [false, true] {
                    assert_protected_context_failure_preserves_owner(
                        OperatorError::ResidentMemory(reason.clone()),
                        |classification| {
                            assert!(matches!(
                                classification,
                                AccountedFailureClassification::ResidentMemory(actual)
                                    if actual == &reason
                            ));
                        },
                        with_cleanup,
                        diagnostic_first,
                    );
                }
            }
        }
    }

    #[test]
    fn protected_io_cancellation_preserves_classification_and_authority() {
        for reason in [
            QueryCancellationError::Cancelled,
            QueryCancellationError::DeadlineExceeded { timeout: None },
            QueryCancellationError::DeadlineExceeded {
                timeout: Some(std::time::Duration::from_secs(30)),
            },
        ] {
            for wrapper_depth in 0..=2 {
                for with_cleanup in [false, true] {
                    for diagnostic_first in [false, true] {
                        let (_directory, manager) = create_manager();
                        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
                        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
                        let calls = Arc::new(AtomicUsize::new(0));
                        let mut io = std::io::Error::new(std::io::ErrorKind::Interrupted, reason);
                        let original = std::ptr::from_ref(
                            io.get_ref()
                                .unwrap()
                                .downcast_ref::<QueryCancellationError>()
                                .unwrap(),
                        );
                        for _ in 0..wrapper_depth {
                            let grant = buffer_manager
                                .try_allocate(4096, MemoryRegion::ExecutionBuffers)
                                .unwrap();
                            io = std::io::Error::new(
                                io.kind(),
                                AccountedPartitionIoFailure::new(io, grant),
                            );
                        }
                        let primary = if with_cleanup {
                            PartitionOperationError::IoWithCleanup {
                                error: io,
                                cleanup: std::io::Error::other(OpaqueQuotaSource {
                                    formatting_or_source_calls: Arc::clone(&calls),
                                    nested: SpillQuotaExceeded::from_usage(0, 0, 99),
                                }),
                                phase: "cancellation cleanup",
                            }
                        } else {
                            PartitionOperationError::Io(io)
                        };
                        let error = state.publish_failure(Some(primary), None, None);
                        let PartitionOperationError::Accounted {
                            classification,
                            authority,
                        } = &error
                        else {
                            panic!("I/O cancellation must retain its accounted owner");
                        };
                        assert!(matches!(
                            classification,
                            AccountedFailureClassification::QueryCancelled(actual)
                                if *actual == reason
                        ));
                        authority
                            .inspect::<PartitionFailure, _>(|failure| {
                                let primary = failure.primary.as_ref().unwrap();
                                let retained = primary.cancellation_error().unwrap();
                                assert_eq!(*retained, reason);
                                assert_eq!(std::ptr::from_ref(retained), original);
                                assert_eq!(
                                    matches!(
                                        primary,
                                        PartitionOperationError::IoWithCleanup { .. }
                                    ),
                                    with_cleanup
                                );
                                if let PartitionOperationError::IoWithCleanup { cleanup, .. } =
                                    primary
                                {
                                    assert!(cleanup.get_ref().unwrap().is::<OpaqueQuotaSource>());
                                }
                                assert!(failure.operator.is_none());
                                assert!(failure.panic.is_none());
                            })
                            .unwrap();
                        assert_eq!(error.to_string(), "accounted partition operation failed");
                        assert_eq!(calls.load(Ordering::Relaxed), 0);
                        if diagnostic_first {
                            drop(error);
                            assert!(buffer_manager.allocated() > 0);
                            drop(state);
                        } else {
                            drop(state);
                            assert!(buffer_manager.allocated() >= wrapper_depth * 4096);
                            assert!(buffer_manager.allocated() > 0);
                            drop(error);
                        }
                        assert_eq!(calls.load(Ordering::Relaxed), 0);
                        assert_eq!(buffer_manager.allocated(), 0);
                    }
                }
            }
        }
    }

    #[derive(Debug)]
    struct OpaqueCancellationSource {
        formatting_or_source_calls: Arc<AtomicUsize>,
        nested: QueryCancellationError,
    }

    impl std::fmt::Display for OpaqueCancellationSource {
        fn fmt(&self, _out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.formatting_or_source_calls
                .fetch_add(1, Ordering::Relaxed);
            panic!("opaque cancellation Display must not run");
        }
    }

    impl std::error::Error for OpaqueCancellationSource {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.formatting_or_source_calls
                .fetch_add(1, Ordering::Relaxed);
            Some(&self.nested)
        }
    }

    #[test]
    fn protected_io_cancellation_does_not_traverse_opaque_source() {
        for kind in [
            std::io::ErrorKind::Interrupted,
            std::io::ErrorKind::TimedOut,
        ] {
            for diagnostic_first in [false, true] {
                let (_directory, manager) = create_manager();
                let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
                let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
                let calls = Arc::new(AtomicUsize::new(0));
                let reason = QueryCancellationError::DeadlineExceeded {
                    timeout: Some(std::time::Duration::from_secs(30)),
                };
                let io = std::io::Error::new(
                    kind,
                    OpaqueCancellationSource {
                        formatting_or_source_calls: Arc::clone(&calls),
                        nested: reason,
                    },
                );
                let original = std::ptr::from_ref(
                    io.get_ref()
                        .unwrap()
                        .downcast_ref::<OpaqueCancellationSource>()
                        .unwrap(),
                );
                let error = state.publish_failure(
                    Some(PartitionOperationError::IoWithCleanup {
                        error: io,
                        cleanup: std::io::Error::new(std::io::ErrorKind::Interrupted, reason),
                        phase: "secondary cancellation",
                    }),
                    None,
                    None,
                );
                let PartitionOperationError::Accounted {
                    classification,
                    authority,
                } = &error
                else {
                    panic!("opaque I/O failure must retain its accounted owner");
                };
                assert!(matches!(
                    classification,
                    AccountedFailureClassification::Execution
                ));
                authority
                    .inspect::<PartitionFailure, _>(|failure| {
                        let primary = failure.primary.as_ref().unwrap();
                        assert!(primary.cancellation_error().is_none());
                        let PartitionOperationError::IoWithCleanup { error, cleanup, .. } = primary
                        else {
                            panic!("opaque primary and secondary must remain owned");
                        };
                        assert_eq!(error.kind(), kind);
                        let retained = error
                            .get_ref()
                            .unwrap()
                            .downcast_ref::<OpaqueCancellationSource>()
                            .unwrap();
                        assert_eq!(std::ptr::from_ref(retained), original);
                        assert_eq!(
                            cleanup
                                .get_ref()
                                .unwrap()
                                .downcast_ref::<QueryCancellationError>(),
                            Some(&reason)
                        );
                    })
                    .unwrap();
                assert_eq!(error.to_string(), "accounted partition operation failed");
                assert_eq!(calls.load(Ordering::Relaxed), 0);
                if diagnostic_first {
                    drop(error);
                    assert!(buffer_manager.allocated() > 0);
                    drop(state);
                } else {
                    drop(state);
                    assert!(buffer_manager.allocated() > 0);
                    drop(error);
                }
                assert_eq!(calls.load(Ordering::Relaxed), 0);
                assert_eq!(buffer_manager.allocated(), 0);
            }
        }
    }

    #[test]
    fn protected_quota_format_preserves_original_fields_identity_and_authority() {
        for with_cleanup in [false, true] {
            for diagnostic_first in [false, true] {
                let (_directory, manager) = create_manager();
                let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
                let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
                let calls = Arc::new(AtomicUsize::new(0));
                let quota = SpillQuotaExceeded::from_usage(42, 11, 7);
                let io = std::io::Error::new(std::io::ErrorKind::StorageFull, quota);
                let original = std::ptr::from_ref(
                    io.get_ref()
                        .unwrap()
                        .downcast_ref::<SpillQuotaExceeded>()
                        .unwrap(),
                );
                let primary = if with_cleanup {
                    PartitionOperationError::IoWithCleanup {
                        error: io,
                        cleanup: std::io::Error::other(OpaqueQuotaSource {
                            formatting_or_source_calls: Arc::clone(&calls),
                            nested: SpillQuotaExceeded::from_usage(0, 0, 99),
                        }),
                        phase: "quota cleanup",
                    }
                } else {
                    PartitionOperationError::Io(io)
                };
                let error = state.publish_failure(Some(primary), None, None);
                assert_eq!(
                    error.to_string(),
                    "spill disk quota exceeded: requested 7 bytes with 11 of 42 bytes in use"
                );
                let PartitionOperationError::Accounted {
                    classification,
                    authority,
                } = &error
                else {
                    panic!("quota failure must retain its accounted owner");
                };
                assert!(matches!(
                    classification,
                    AccountedFailureClassification::StorageFull
                ));
                authority
                    .inspect::<PartitionFailure, _>(|failure| {
                        let (PartitionOperationError::Io(io)
                        | PartitionOperationError::IoWithCleanup { error: io, .. }) =
                            failure.primary.as_ref().unwrap()
                        else {
                            panic!("original quota I/O must remain typed");
                        };
                        let retained = io
                            .get_ref()
                            .unwrap()
                            .downcast_ref::<SpillQuotaExceeded>()
                            .unwrap();
                        assert_eq!(std::ptr::from_ref(retained), original);
                        assert_eq!(retained.limit_bytes(), 42);
                        assert_eq!(retained.used_bytes(), 11);
                        assert_eq!(retained.requested_bytes(), 7);
                    })
                    .unwrap();
                assert_eq!(calls.load(Ordering::Relaxed), 0);
                if diagnostic_first {
                    drop(error);
                    assert!(buffer_manager.allocated() > 0);
                    drop(state);
                } else {
                    drop(state);
                    assert!(buffer_manager.allocated() > 0);
                    assert_eq!(error.to_string(), quota.to_string());
                    drop(error);
                }
                assert_eq!(calls.load(Ordering::Relaxed), 0);
                assert_eq!(buffer_manager.allocated(), 0);
            }
        }
    }

    #[test]
    fn protected_quota_format_does_not_traverse_or_format_opaque_primary() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        let calls = Arc::new(AtomicUsize::new(0));
        let primary = std::io::Error::new(
            std::io::ErrorKind::StorageFull,
            OpaqueQuotaSource {
                formatting_or_source_calls: Arc::clone(&calls),
                nested: SpillQuotaExceeded::from_usage(42, 11, 7),
            },
        );
        let error = state.publish_failure(
            Some(PartitionOperationError::IoWithCleanup {
                error: primary,
                cleanup: std::io::Error::new(
                    std::io::ErrorKind::StorageFull,
                    SpillQuotaExceeded::from_usage(0, 0, 99),
                ),
                phase: "unrelated quota cleanup",
            }),
            None,
            None,
        );
        assert_eq!(error.to_string(), "accounted partition operation failed");
        drop(state);
        assert!(buffer_manager.allocated() > 0);
        assert_eq!(error.to_string(), "accounted partition operation failed");
        drop(error);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn protected_recovery_prefers_largest_sufficient_reclaim_then_oldest() {
        let (_directory, manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        let denial = MemoryGrantError::Denied {
            additional_bytes: 1,
        };
        assert!(state.access_times[0] < state.access_times[1]);
        assert_eq!(state.recovery_partition(Some(2), &denial), Some(1));
        assert_eq!(state.recovery_partition(Some(1), &denial), Some(0));

        // Give the older, smaller partition the same observed backing as the
        // larger one. The other partition is now oldest among equal reclaims.
        let lookup = key_for_partition(&state, 0);
        state
            .try_update_accounted(
                std::mem::size_of::<Value>(),
                || Ok(lookup),
                |_| {
                    Ok((
                        PartitionUpdateAdmission {
                            retained_upper_bound: 4096,
                            construction_peak: 4096,
                        },
                        0,
                    ))
                },
                || Ok(Vec::new()),
                |value| {
                    *value = vec![0x62; 4096];
                    Ok(())
                },
            )
            .unwrap();
        assert!(state.access_times[1] < state.access_times[0]);
        assert_eq!(state.recovery_partition(Some(2), &denial), Some(1));
        // Touch only the other partition: with equal byte ranks, selection
        // follows the changed age rather than the partition's index.
        state.touch(1);
        assert!(state.access_times[0] < state.access_times[1]);
        assert_eq!(state.recovery_partition(Some(2), &denial), Some(0));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn protected_recovery_skips_lru_victim_that_is_55_bytes_short() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(RecoveryIo::default());
        let manager = Arc::new(
            super::super::BorrowedSpillFixture::new(directory.path())
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        let oldest = state.partitions[0].as_ref().unwrap();
        let oldest_reclaim = PartitionedState::<Vec<u8>>::partition_map_allocation_bytes(oldest)
            + oldest
                .iter()
                .map(|(key, entry)| key.0.capacity() + entry.resident_bound)
                .sum::<usize>();
        assert!(state.access_times[0] < state.access_times[1]);
        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available() - 65536,
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let mut workspace = state.split_workspace_grant(0).unwrap();
        let required = buffer_manager.available() + oldest_reclaim + 55;
        let denial = workspace.try_resize(required).unwrap_err();
        assert_eq!(state.recovery_partition(Some(2), &denial), Some(1));
        assert_eq!(state.recovery_partition(Some(1), &denial), None);
        let mut retry = OneSpillRetryBudget::default();
        let cancellation = state.cancellation.clone();

        state
            .resize_transient_grant_with_one_spill(
                &mut workspace,
                required,
                Some(2),
                cancellation.as_ref(),
                &mut retry,
            )
            .unwrap();

        assert!(retry.spent);
        assert_eq!(workspace.size(), required);
        assert_eq!(io.creates.load(Ordering::Relaxed), 1);
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes, [0, 1, 0]);
        assert_eq!(state.partitions[0].as_ref().unwrap().len(), 1);
        drop(workspace);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn protected_recovery_without_adequate_victim_preserves_denial_without_io() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(RecoveryIo::default());
        let manager = Arc::new(
            super::super::BorrowedSpillFixture::new(directory.path())
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = protected_recovery_bytes_state(manager, &buffer_manager);
        let largest = state.partitions[1].as_ref().unwrap();
        let largest_reclaim = PartitionedState::<Vec<u8>>::partition_map_allocation_bytes(largest)
            + largest
                .iter()
                .map(|(key, entry)| key.0.capacity() + entry.resident_bound)
                .sum::<usize>();
        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available() - 65536,
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let mut workspace = state.split_workspace_grant(0).unwrap();
        let required = buffer_manager.available() + largest_reclaim + 55;
        let original = workspace.try_resize(required).unwrap_err();
        let mut retry = OneSpillRetryBudget::default();
        let cancellation = state.cancellation.clone();

        let error = state
            .resize_transient_grant_with_one_spill(
                &mut workspace,
                required,
                Some(2),
                cancellation.as_ref(),
                &mut retry,
            )
            .unwrap_err();

        assert_eq!(error.resident_memory_error(), Some(&original));
        assert!(!retry.spent);
        assert_eq!(workspace.size(), 0);
        assert_eq!(io.calls.load(Ordering::Relaxed), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(state.partition_sizes, [1, 1, 0]);
        drop(workspace);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn denied_key_staging_spills_one_cold_partition_then_retries_existing_lookup() {
        const COLD_KEY_BYTES: usize = 32 * 1024;
        const TARGET_KEY_BYTES: usize = 110 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        let cold = wide_key_for_partition(&state, 0, COLD_KEY_BYTES, 'c');
        let target = wide_key_for_partition(&state, 1, TARGET_KEY_BYTES, 't');
        let serialized_cold = SerializedKey::from_values(&cold, state.frame_limits).unwrap();
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();

        *state.get_or_insert_with_accounted(cold, 0, || 10).unwrap() += 1;
        *state
            .get_or_insert_with_accounted(target.clone(), 0, || 20)
            .unwrap() += 1;

        let target_admission = serialized_target.0.len().checked_mul(2).unwrap();
        let headroom = target_admission.checked_sub(1).unwrap();
        let cold_record = serialized_cold
            .0
            .len()
            .checked_add(32 + std::mem::size_of::<i64>())
            .unwrap();
        let spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(cold_record.checked_mul(4).unwrap())
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        assert!(
            spill_peak < headroom,
            "the test must leave enough transient room to spill the cold partition"
        );
        let filler_bytes = buffer_manager.available().checked_sub(headroom).unwrap();
        let filler = buffer_manager
            .try_allocate(filler_bytes, MemoryRegion::ExecutionBuffers)
            .unwrap();
        assert_eq!(buffer_manager.available(), headroom);
        let default_called = AtomicBool::new(false);

        let existing = state
            .get_or_insert_with_accounted(target.clone(), 0, || {
                default_called.store(true, Ordering::Release);
                99
            })
            .unwrap();

        assert_eq!(*existing, 21);
        assert!(!default_called.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.spill_base_sizes[1], 0);
        assert_eq!(state.get(&target).unwrap(), Some(&21));
        assert!(buffer_manager.allocated() <= buffer_manager.budget());
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn denied_replacement_key_staging_spills_once_then_inserts_absent_value() {
        const COLD_KEY_BYTES: usize = 32 * 1024;
        const TARGET_KEY_BYTES: usize = 110 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        let cold = wide_key_for_partition(&state, 0, COLD_KEY_BYTES, 'c');
        let target = wide_key_for_partition(&state, 1, TARGET_KEY_BYTES, 't');
        let serialized_cold = SerializedKey::from_values(&cold, state.frame_limits).unwrap();
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();
        *state.get_or_insert_with_accounted(cold, 0, || 10).unwrap() += 1;

        let headroom = serialized_target
            .0
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_sub(1))
            .unwrap();
        let cold_record = serialized_cold
            .0
            .len()
            .checked_add(32 + std::mem::size_of::<i64>())
            .unwrap();
        let spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(cold_record.checked_mul(4).unwrap())
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        assert!(spill_peak < headroom);
        let filler_bytes = buffer_manager.available().checked_sub(headroom).unwrap();
        let filler = buffer_manager
            .try_allocate(filler_bytes, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let declared = AtomicBool::new(false);
        let built = AtomicBool::new(false);

        state
            .try_replace_accounted(
                target.clone(),
                |old| {
                    assert!(old.is_none());
                    declared.store(true, Ordering::Release);
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |old| {
                    assert!(old.is_none());
                    built.store(true, Ordering::Release);
                    Ok(77)
                },
            )
            .unwrap();

        assert!(declared.load(Ordering::Acquire));
        assert!(built.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.spill_base_sizes[1], 0);
        assert_eq!(state.get(&target).unwrap(), Some(&77));
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn key_and_map_admission_share_one_spill_budget() {
        const COLD_KEY_BYTES: usize = 32 * 1024;
        const TARGET_KEY_BYTES: usize = 300 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 3);
        let first_cold = wide_key_for_partition(&state, 0, COLD_KEY_BYTES, 'a');
        let second_cold = wide_key_for_partition(&state, 1, COLD_KEY_BYTES, 'b');
        let target = wide_key_for_partition(&state, 2, TARGET_KEY_BYTES, 't');
        let serialized_cold = SerializedKey::from_values(&first_cold, state.frame_limits).unwrap();
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();
        *state
            .get_or_insert_with_accounted(first_cold, 0, || 10)
            .unwrap() += 1;
        *state
            .get_or_insert_with_accounted(second_cold, 0, || 20)
            .unwrap() += 1;

        let headroom = serialized_target
            .0
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_sub(1))
            .unwrap();
        let cold_record = serialized_cold
            .0
            .len()
            .checked_add(32 + std::mem::size_of::<i64>())
            .unwrap();
        let spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(cold_record.checked_mul(4).unwrap())
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        assert!(
            spill_peak < serialized_target.0.len(),
            "the second cold spill would fit after successful key staging"
        );
        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available().checked_sub(headroom).unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let cancellation = state.cancellation.clone();
        let mut retry_budget = OneSpillRetryBudget::default();

        let staged = state
            .stage_accounted_key_with_one_spill(
                &target,
                2,
                cancellation.as_ref(),
                &mut retry_budget,
            )
            .unwrap();
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(|partition| !partition.is_empty())
        );

        let error = state
            .reserve_new_entry_with_one_spill(
                2,
                buffer_manager.budget(),
                cancellation.as_ref(),
                &mut retry_budget,
            )
            .unwrap_err();

        assert!(matches!(error, PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::OutOfMemory));
        assert_eq!(
            state.spilled_count(),
            1,
            "one logical operation may reclaim only one cold partition"
        );
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.spill_base_sizes[1], 0);
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(|partition| !partition.is_empty()),
            "map admission must not spend a second spill after key staging"
        );
        assert!(state.partitions[2].as_ref().unwrap().is_empty());

        drop(staged);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn map_growth_plan_rejects_allocation_above_preadmitted_ceiling() {
        let (_directory, spill_manager) = create_manager();
        let state: PartitionedState<i64> =
            PartitionedState::new(spill_manager, 1, serialize_i64, deserialize_i64);
        let mut plan = state.pending_entry_capacity_plan(0, 17).unwrap();
        let required_entries = plan
            .replacement_required_entries
            .expect("an empty partition requires a replacement map");
        let mut replacement = new_partition_map();
        replacement.try_reserve(required_entries).unwrap();
        plan.replacement_allocation_ceiling = Some(replacement.allocation_size() - 1);

        let error =
            PartitionedState::<i64>::validated_reserved_entry_capacities(plan, &replacement)
                .unwrap_err();

        assert_eq!(
            error,
            "native partition replacement exceeds its pre-admitted allocation ceiling"
        );
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_byte_short_map_admission_precedes_allocation_and_default_callback() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[1]);
        let baseline = state.granted_bytes();
        let planned_map_bytes = state
            .pending_entry_capacity_plan(0, 0)
            .unwrap()
            .replacement_allocation_ceiling
            .expect("an empty partition needs its first native allocation");

        // Measure the retained key authority independently. The map ceiling is
        // much larger than this key's transient 2x encoding envelope, so the
        // staged key succeeds and leaves exactly one byte too little for the
        // subsequent native-map admission.
        let staged = AccountedSerializedKey::from_values(
            &lookup,
            state.frame_limits,
            state.grant.as_mut().unwrap(),
        )
        .unwrap();
        let key_bytes = staged.granted_bytes();
        drop(staged);
        assert_eq!(buffer_manager.allocated(), baseline);
        let remaining = key_bytes
            .checked_add(planned_map_bytes)
            .and_then(|bytes| bytes.checked_sub(1))
            .unwrap();
        let filler_bytes = buffer_manager.available().checked_sub(remaining).unwrap();
        let filler = buffer_manager
            .try_allocate(filler_bytes, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let callback_ran = AtomicBool::new(false);

        let error = state
            .get_or_insert_with_accounted(lookup.clone(), 0, || {
                callback_ran.store(true, Ordering::Release);
                10
            })
            .unwrap_err();

        assert!(matches!(error, PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::OutOfMemory));
        assert!(!callback_ran.load(Ordering::Acquire));
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.partition_sizes[0], 0);
        assert_eq!(state.partitions[0].as_ref().unwrap().allocation_size(), 0);
        assert_eq!(state.granted_bytes(), baseline);
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );

        drop(filler);
        *state
            .get_or_insert_with_accounted(lookup, 0, || {
                callback_ran.store(true, Ordering::Release);
                10
            })
            .unwrap() += 1;
        assert!(callback_ran.load(Ordering::Acquire));
        assert_eq!(state.total_size(), 1);
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn native_map_rollback_uses_receipt_without_reentering_capacity_inspector() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let reject_inspection = Arc::new(AtomicBool::new(false));
        let inspection_gate = Arc::clone(&reject_inspection);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            move |_value: &i64| {
                assert!(
                    !inspection_gate.load(Ordering::Acquire),
                    "rollback re-entered the caller-supplied capacity inspector"
                );
                Ok(0)
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap();
        let receipt = state.observed_resident_capacity_bytes().unwrap();
        state
            .grant
            .as_mut()
            .unwrap()
            .try_resize(receipt + 64)
            .unwrap();
        reject_inspection.store(true, Ordering::Release);

        let rollback = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.rollback_native_map_admission(receipt)
        }));

        assert_eq!(
            rollback.expect("receipt rollback must not invoke callbacks"),
            Ok(())
        );
        assert_eq!(state.granted_bytes(), receipt);
        assert_eq!(buffer_manager.allocated(), receipt);
    }

    #[test]
    fn native_map_secondary_rollback_is_not_reported_as_primary_capacity_exhaustion() {
        let rollback = MemoryGrantError::AccountingPoisoned {
            account: "native map test",
        };
        let allocation = PartitionOperationError::NativeMapAllocationWithRollback {
            error: NativeMapAllocationError::capacity_overflow(),
            rollback: rollback.clone(),
        };
        let invariant = PartitionOperationError::NativeMapInvariantWithRollback {
            message: "synthetic layout mismatch",
            rollback,
        };

        assert!(allocation.resident_memory_error().is_none());
        assert!(invariant.resident_memory_error().is_none());
    }

    #[test]
    fn successful_map_growth_receipt_matches_exact_peak_and_final_allocation() {
        let (_directory, spill_manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(spill_manager, 1, serialize_i64, deserialize_i64);
        let partition = state.partitions[0].as_mut().unwrap();
        partition.try_reserve(3).unwrap();
        for byte in 0_u8..3 {
            partition.insert(
                SerializedKey(vec![byte]),
                PartitionEntry {
                    num_key_columns: 1,
                    resident_bound: 0,
                    value: i64::from(byte),
                },
            );
        }
        assert_eq!(partition.len(), partition.capacity());
        let pending_bytes = 37;
        let plan = state.pending_entry_capacity_plan(0, pending_bytes).unwrap();
        let required_entries = plan.replacement_required_entries.unwrap();
        let mut replacement = new_partition_map();
        replacement.try_reserve(required_entries).unwrap();
        let actual_replacement_bytes = replacement.allocation_size();

        let receipt =
            PartitionedState::<i64>::validated_reserved_entry_capacities(plan, &replacement)
                .unwrap();

        assert_eq!(
            receipt.peak_bytes,
            plan.observed_bytes + actual_replacement_bytes + pending_bytes
        );
        assert_eq!(
            receipt.final_bytes,
            plan.observed_bytes - plan.old_map_bytes + actual_replacement_bytes + pending_bytes
        );
        assert!(receipt.peak_bytes <= plan.admitted_bytes);
        assert!(receipt.final_bytes <= receipt.peak_bytes);
    }

    #[test]
    fn successful_first_map_publication_reconciles_to_observed_allocation_size() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let baseline = state.granted_bytes();

        state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap();

        let partition = state.partitions[0].as_ref().unwrap();
        let map_bytes = partition.allocation_size();
        let key_bytes = partition.keys().next().unwrap().0.capacity();
        assert_ne!(map_bytes, 0);
        assert_eq!(state.granted_bytes(), baseline + map_bytes + key_bytes);
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_map_growth_rehashes_every_entry_into_the_randomized_replacement() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let mut keys = Vec::new();
        let mut nonce = 0_i64;

        let first = key(&[nonce]);
        state
            .get_or_insert_with_accounted(first.clone(), 0, || nonce)
            .unwrap();
        keys.push((first, nonce));
        nonce += 1;
        let old_capacity = state.partitions[0].as_ref().unwrap().capacity();
        while state.partitions[0].as_ref().unwrap().len() < old_capacity {
            let lookup = key(&[nonce]);
            state
                .get_or_insert_with_accounted(lookup.clone(), 0, || nonce)
                .unwrap();
            keys.push((lookup, nonce));
            nonce += 1;
        }
        let old_allocation = state.partitions[0].as_ref().unwrap().allocation_size();

        let growth_key = key(&[nonce]);
        state
            .get_or_insert_with_accounted(growth_key.clone(), 0, || nonce)
            .unwrap();
        keys.push((growth_key, nonce));

        let partition = state.partitions[0].as_ref().unwrap();
        assert!(partition.capacity() > old_capacity);
        assert!(partition.allocation_size() > old_allocation);
        for (lookup, expected) in keys {
            let serialized = SerializedKey::from_values(&lookup, state.frame_limits).unwrap();
            assert_eq!(
                partition.get(&serialized).map(|entry| entry.value),
                Some(expected)
            );
        }
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn native_partition_map_type_locks_randomized_hasher_and_exact_allocator() {
        fn require_contract<V>(
            _map: &hashbrown::HashMap<
                SerializedKey,
                PartitionEntry<V>,
                RandomState,
                allocator_api2::alloc::Global,
            >,
        ) {
        }

        let map: PartitionMap<i64> = new_partition_map();
        require_contract(&map);
        assert_eq!(map.allocation_size(), 0);
    }

    #[test]
    fn partition_map_layout_plan_covers_pinned_hashbrown_allocation() {
        for required_entries in 1..=4096 {
            let planned =
                PartitionedState::<i64>::planned_partition_map_allocation_bytes(required_entries)
                    .unwrap();
            let mut map: PartitionMap<i64> = new_partition_map();
            map.try_reserve(required_entries).unwrap();

            assert!(map.capacity() >= required_entries);
            assert!(
                map.allocation_size() <= planned,
                "pinned hashbrown allocation {} exceeded the {planned}-byte preplan for {required_entries} entries",
                map.allocation_size()
            );
        }
    }

    #[test]
    fn partition_map_layout_plan_covers_high_alignment_entries() {
        #[repr(align(64))]
        #[derive(Clone)]
        struct HighAlignmentValue;

        for required_entries in [1, 3, 7, 15, 29, 257] {
            let planned =
                PartitionedState::<HighAlignmentValue>::planned_partition_map_allocation_bytes(
                    required_entries,
                )
                .unwrap();
            let mut map: PartitionMap<HighAlignmentValue> = new_partition_map();
            map.try_reserve(required_entries).unwrap();

            assert!(map.allocation_size() <= planned);
        }
    }

    #[test]
    fn partition_map_layout_plan_rejects_overflow_before_allocation() {
        assert!(matches!(
            PartitionedState::<i64>::planned_partition_map_allocation_bytes(usize::MAX),
            Err(MemoryGrantError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn counter_sort_scratch_spill_exhausts_later_map_admission_budget() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(16 * 1024 * 1024);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 3);
        let first_cold = key_for_partition(&state, 0);
        let second_cold = key_for_partition(&state, 1);
        for (partition_idx, cold) in [(0, first_cold), (1, second_cold)] {
            state.partitions[partition_idx]
                .as_mut()
                .unwrap()
                .try_reserve(4096)
                .unwrap();
            let serialized = SerializedKey::from_values(&cold, state.frame_limits).unwrap();
            state.partitions[partition_idx].as_mut().unwrap().insert(
                serialized,
                PartitionEntry {
                    num_key_columns: 1,
                    resident_bound: 0,
                    value: 1,
                },
            );
            state.partition_sizes[partition_idx] = 1;
        }
        state.access_times = vec![1, 2, 3];
        state.timestamp = 3;
        state.reconcile_grant().unwrap();

        let counter = Arc::new(
            (0..256)
                .map(|replica| (format!("replica-{replica:04}-{}", "x".repeat(112)), replica))
                .collect::<HashMap<_, _>>(),
        );
        let target = (0_i64..10_000)
            .map(|nonce| vec![Value::GCounter(Arc::clone(&counter)), Value::Int64(nonce)])
            .find(|candidate| state.partition_for(candidate) == 2)
            .expect("counter test key search must cover the target partition");
        let measurement =
            measure_serialized_row_with_limits(&target, state.frame_limits.codec_limits()).unwrap();
        let scratch_provisional = CounterSortScratch::requested_capacity_bytes(
            measurement.counter_sort_entries,
            measurement.counter_sort_key_bytes,
        )
        .unwrap()
        .checked_mul(2)
        .unwrap();
        let calibration_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let mut calibration_root = calibration_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let calibration =
            AccountedSerializedKey::from_values(&target, state.frame_limits, &mut calibration_root)
                .unwrap();
        let key_capacity = calibration.granted_bytes();
        drop(calibration);
        drop(calibration_root);
        let scratch_denial_headroom = key_capacity
            .checked_add(scratch_provisional)
            .and_then(|bytes| bytes.checked_sub(1))
            .unwrap();
        assert!(
            measurement.encoded_bytes.checked_mul(2).unwrap() <= scratch_denial_headroom,
            "writer preflight must fit before the counter scratch denial"
        );
        let filler = buffer_manager
            .try_allocate(
                buffer_manager
                    .available()
                    .checked_sub(scratch_denial_headroom)
                    .unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let cancellation = state.cancellation.clone();
        let mut retry_budget = OneSpillRetryBudget::default();

        let staged = state
            .stage_accounted_key_with_one_spill(
                &target,
                2,
                cancellation.as_ref(),
                &mut retry_budget,
            )
            .unwrap();

        assert!(retry_budget.spent);
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert!(
            state.partitions[0]
                .as_ref()
                .is_some_and(PartitionMap::is_empty)
        );
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(|partition| !partition.is_empty())
        );

        let zero_pending = state.pending_entry_capacity_bytes(2, 0).unwrap();
        let map_growth = zero_pending.checked_sub(state.granted_bytes()).unwrap();
        let available = buffer_manager.available();
        assert!(available > map_growth);
        let pending = available
            .checked_sub(map_growth)
            .unwrap()
            .checked_add(1)
            .unwrap();
        let error = state
            .reserve_new_entry_with_one_spill(2, pending, cancellation.as_ref(), &mut retry_budget)
            .unwrap_err();
        assert!(matches!(error, PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::OutOfMemory));
        assert_eq!(state.spilled_count(), 1);
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(|partition| !partition.is_empty())
        );

        // A fresh logical operation can spend victim two and satisfy this
        // exact same admission, proving the denial above was the shared token
        // rather than an impossible capacity request.
        let mut fresh_budget = OneSpillRetryBudget::default();
        state
            .reserve_new_entry_with_one_spill(2, pending, cancellation.as_ref(), &mut fresh_budget)
            .unwrap();
        assert!(fresh_budget.spent);
        assert_eq!(state.spilled_count(), 2);
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(PartitionMap::is_empty)
        );

        state.reconcile_grant().unwrap();
        drop(staged);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn replacement_threads_one_spill_budget_across_key_and_map_admission() {
        const COLD_KEY_BYTES: usize = 32 * 1024;
        const TARGET_KEY_BYTES: usize = 300 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let mut state = accounted_bytes_state_with_partitions(
            Arc::clone(&spill_manager),
            &buffer_manager,
            control.token(),
            3,
        );
        let first_cold = wide_key_for_partition(&state, 0, COLD_KEY_BYTES, 'a');
        state
            .get_or_insert_with_accounted(first_cold, 0, Vec::new)
            .unwrap();

        state.partitions[2]
            .as_mut()
            .unwrap()
            .try_reserve(2048)
            .unwrap();
        let target_capacity = state.partitions[2].as_ref().unwrap().capacity();
        let mut nonce = 0_i64;
        while state.partitions[2].as_ref().unwrap().len() < target_capacity {
            let candidate = key(&[nonce]);
            nonce += 1;
            if state.partition_for(&candidate) != 2 {
                continue;
            }
            let serialized = SerializedKey::from_values(&candidate, state.frame_limits).unwrap();
            state.partitions[2].as_mut().unwrap().insert(
                serialized,
                PartitionEntry {
                    num_key_columns: 1,
                    resident_bound: 0,
                    value: Vec::new(),
                },
            );
        }
        let target_count = state.partitions[2].as_ref().unwrap().len();
        state.partition_sizes[2] = target_count;

        state.partitions[1]
            .as_mut()
            .unwrap()
            .try_reserve(target_capacity.checked_mul(4).unwrap())
            .unwrap();
        let second_cold = key_for_partition(&state, 1);
        let serialized_second_cold =
            SerializedKey::from_values(&second_cold, state.frame_limits).unwrap();
        state.partitions[1].as_mut().unwrap().insert(
            serialized_second_cold.clone(),
            PartitionEntry {
                num_key_columns: 1,
                resident_bound: 0,
                value: Vec::new(),
            },
        );
        state.partition_sizes[1] = 1;
        state.access_times = vec![1, 2, 3];
        state.timestamp = 3;
        state.reconcile_grant().unwrap();

        let observed = state.observed_resident_capacity_bytes().unwrap();
        let map_growth = state
            .pending_entry_capacity_bytes(2, 0)
            .unwrap()
            .checked_sub(observed)
            .unwrap();
        let cold_record = serialized_second_cold.0.len().checked_add(32).unwrap();
        let second_spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(cold_record.checked_mul(4).unwrap())
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        let second_spill_headroom = second_spill_peak.checked_add(16 * 1024).unwrap();
        assert!(second_spill_headroom < map_growth);

        let target = wide_key_for_partition(&state, 2, TARGET_KEY_BYTES, 't');
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();
        assert!(
            !state.partitions[2]
                .as_ref()
                .unwrap()
                .contains_key(&serialized_target)
        );

        let headroom = serialized_target
            .0
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_sub(1))
            .unwrap();
        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available().checked_sub(headroom).unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let admitted_peak = AtomicUsize::new(0);
        let built = AtomicBool::new(false);

        let result = state.try_replace_accounted(
            target.clone(),
            |_| {
                assert_eq!(
                    spill_manager.active_file_count(),
                    1,
                    "key staging must spend the operation's only spill budget"
                );
                let peak = buffer_manager
                    .available()
                    .checked_sub(second_spill_headroom)
                    .expect("key staging must leave candidate and spill headroom");
                assert!(peak > 0, "the candidate must retain real capacity");
                admitted_peak.store(peak, Ordering::Release);
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: peak,
                    construction_peak: peak,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(vec![0x77; admitted_peak.load(Ordering::Acquire)])
            },
        );

        assert!(
            matches!(
                result,
                Err(PartitionUpdateError::Partition(PartitionOperationError::Io(
                    ref error
                ))) if error.kind() == std::io::ErrorKind::OutOfMemory
            ),
            "a second spill let the replacement escape its retry budget: {result:?}"
        );
        assert!(built.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.spill_base_sizes[1], 0);
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(|partition| !partition.is_empty()),
            "replacement map admission must not reclaim a second victim"
        );
        assert_eq!(state.partition_sizes[2], target_count);
        assert_eq!(state.partitions[2].as_ref().unwrap().len(), target_count);
        assert!(
            !state.partitions[2]
                .as_ref()
                .unwrap()
                .contains_key(&serialized_target),
            "failed replacement must not publish the candidate"
        );
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn non_idle_state_precedes_saturation_and_callbacks_for_both_accounted_apis() {
        const TARGET_KEY_BYTES: usize = 110 * 1024;

        for drain_state in [DrainState::Draining, DrainState::Poisoned] {
            let (_directory, spill_manager) = create_manager();
            let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
            let target = wide_key_for_partition(&state, 1, TARGET_KEY_BYTES, 't');
            let cold = key_for_partition(&state, 0);
            *state.get_or_insert_with_accounted(cold, 0, || 10).unwrap() += 1;

            let serialized_target =
                SerializedKey::from_values(&target, state.frame_limits).unwrap();
            let headroom = serialized_target
                .0
                .len()
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_sub(1))
                .unwrap();
            let filler_bytes = buffer_manager.available().checked_sub(headroom).unwrap();
            let filler = buffer_manager
                .try_allocate(filler_bytes, MemoryRegion::ExecutionBuffers)
                .unwrap();
            state.partition_sizes[1] = usize::MAX;
            state.drain_state = drain_state;
            let default_called = AtomicBool::new(false);

            let error = state
                .get_or_insert_with_accounted(target.clone(), 0, || {
                    default_called.store(true, Ordering::Release);
                    99
                })
                .unwrap_err();

            assert!(matches!(error, PartitionOperationError::Io(ref error)
                    if error.kind() == std::io::ErrorKind::InvalidInput
                        && error.to_string().contains("destructive drain")));
            assert!(!default_called.load(Ordering::Acquire));
            assert_eq!(state.spilled_count(), 0);
            assert_eq!(spill_manager.active_file_count(), 0);
            assert_eq!(state.partition_sizes[1], usize::MAX);

            let declared = AtomicBool::new(false);
            let built = AtomicBool::new(false);
            let result = state.try_replace_accounted(
                target,
                |_| {
                    declared.store(true, Ordering::Release);
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 0,
                        construction_peak: 0,
                    })
                },
                |_| {
                    built.store(true, Ordering::Release);
                    Ok(99)
                },
            );
            assert!(matches!(
                result,
                Err(PartitionUpdateError::Partition(PartitionOperationError::Io(
                    ref error
                ))) if error.kind() == std::io::ErrorKind::InvalidInput
                    && error.to_string().contains("destructive drain")
            ));
            assert!(!declared.load(Ordering::Acquire));
            assert!(!built.load(Ordering::Acquire));
            assert_eq!(state.spilled_count(), 0);
            drop(filler);
        }
    }

    #[test]
    fn cancellation_precedes_lifecycle_saturation_and_callbacks_for_accounted_mutations() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let mut state = accounted_i64_state_with_cancellation(
            Arc::clone(&spill_manager),
            &buffer_manager,
            2,
            control.token(),
        );
        let target = key_for_partition(&state, 1);
        let cold = key_for_partition(&state, 0);
        *state.get_or_insert_with_accounted(cold, 0, || 10).unwrap() += 1;
        state.partition_sizes[1] = usize::MAX;
        state.drain_state = DrainState::Poisoned;
        control.cancellation_handle().cancel();

        let default_called = AtomicBool::new(false);
        let error = state
            .get_or_insert_with_accounted(target.clone(), 0, || {
                default_called.store(true, Ordering::Release);
                99
            })
            .unwrap_err();
        assert!(matches!(error, PartitionOperationError::Cancelled(_)));
        assert!(!default_called.load(Ordering::Acquire));

        let declared = AtomicBool::new(false);
        let built = AtomicBool::new(false);
        let result = state.try_replace_accounted(
            target,
            |_| {
                declared.store(true, Ordering::Release);
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: 0,
                    construction_peak: 0,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                Ok(99)
            },
        );
        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(
                PartitionOperationError::Cancelled(_)
            ))
        ));
        assert!(!declared.load(Ordering::Acquire));
        assert!(!built.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(state.partition_sizes[1], usize::MAX);
    }

    #[test]
    fn non_grant_key_codec_error_never_spills_a_candidate_partition() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        let invalid = over_depth_key();
        let target_partition = state.partition_for(&invalid);
        let cold = key_for_partition(&state, 1 - target_partition);
        *state.get_or_insert_with_accounted(cold, 0, || 10).unwrap() += 1;
        let default_called = AtomicBool::new(false);

        let error = state
            .get_or_insert_with_accounted(invalid, 0, || {
                default_called.store(true, Ordering::Release);
                99
            })
            .unwrap_err();

        assert!(matches!(error, PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidInput));
        assert!(!default_called.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    fn denied_partition_growth_spills_once_then_retries_under_the_same_grant() {
        let (_calibration_directory, calibration_spill) = create_manager();
        let calibration_manager = BufferManager::with_budget(1 << 20);
        let mut calibration = accounted_i64_state(calibration_spill, &calibration_manager, 2);
        let cold = key_for_partition(&calibration, 0);
        *calibration
            .get_or_insert_with_accounted(cold.clone(), 0, || 10)
            .unwrap() += 1;
        let target_candidates = (0_i64..10_000)
            .map(|nonce| key(&[nonce]))
            .filter(|candidate| calibration.partition_for(candidate) == 1)
            .collect::<Vec<_>>();
        let mut target_candidates = target_candidates.into_iter();
        let first_target = target_candidates.next().unwrap();
        *calibration
            .get_or_insert_with_accounted(first_target.clone(), 0, || 20)
            .unwrap() += 1;
        let target_capacity = calibration.partitions[1].as_ref().unwrap().capacity();
        let mut target_keys = vec![first_target];
        while target_keys.len() < target_capacity {
            let candidate = target_candidates.next().unwrap();
            *calibration
                .get_or_insert_with_accounted(candidate.clone(), 0, || 20)
                .unwrap() += 1;
            target_keys.push(candidate);
        }
        let growth_key = target_candidates.next().unwrap();
        let staged_growth_key = AccountedSerializedKey::from_values(
            &growth_key,
            calibration.frame_limits,
            calibration.grant.as_mut().unwrap(),
        )
        .unwrap();
        let growth_key_bytes = staged_growth_key.granted_bytes();
        drop(staged_growth_key);
        let replacement_bytes = calibration
            .pending_entry_capacity_plan(1, 0)
            .unwrap()
            .replacement_allocation_ceiling
            .expect("a full target map must require replacement");
        let serialized_cold = SerializedKey::from_values(&cold, calibration.frame_limits).unwrap();
        let spill_envelope = serialized_cold
            .0
            .len()
            .checked_add(32 + std::mem::size_of::<i64>())
            .and_then(|record| record.checked_mul(4))
            .unwrap();
        assert!(
            replacement_bytes > spill_envelope,
            "the forced map replacement must exceed its cold-victim spill workspace"
        );
        let calibrated_resident = calibration.granted_bytes();
        let exact_budget = calibrated_resident
            .checked_add(growth_key_bytes)
            .and_then(|bytes| bytes.checked_add(replacement_bytes))
            .and_then(|bytes| bytes.checked_sub(1))
            .unwrap();
        drop(calibration);

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(exact_budget);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 2);
        *state.get_or_insert_with_accounted(cold, 0, || 10).unwrap() += 1;
        for target in target_keys {
            *state
                .get_or_insert_with_accounted(target, 0, || 20)
                .unwrap() += 1;
        }
        assert_eq!(state.granted_bytes(), calibrated_resident);
        *state
            .get_or_insert_with_accounted(growth_key.clone(), 0, || 30)
            .unwrap() += 1;

        assert_eq!(
            state.spilled_count(),
            1,
            "denial spills exactly one eligible partition"
        );
        assert_eq!(state.get(&growth_key).unwrap(), Some(&31));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert!(buffer_manager.allocated() <= exact_budget);
    }

    #[test]
    fn underestimated_value_capacity_is_dropped_before_entry_publication() {
        fn serialize_bytes(
            value: &[u8],
            writer: &mut dyn Write,
            _limits: CodecLimits,
        ) -> std::io::Result<()> {
            writer.write_all(value)
        }
        fn deserialize_bytes(
            reader: &mut dyn Read,
            _limits: CodecLimits,
        ) -> std::io::Result<Vec<u8>> {
            let mut value = Vec::new();
            reader.read_to_end(&mut value)?;
            Ok(value)
        }

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &Vec<u8>, writer, limits| serialize_bytes(value, writer, limits),
            deserialize_bytes,
            |value: &Vec<u8>| Ok(value.capacity()),
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();

        let error = state
            .get_or_insert_with_accounted(key(&[1]), 0, || vec![7; 1024])
            .unwrap_err();

        assert!(
            matches!(error, PartitionOperationError::Io(ref error) if error.kind() == std::io::ErrorKind::InvalidInput)
        );
        assert_eq!(state.total_size(), 0);
        assert!(state.partitions[0].as_ref().unwrap().is_empty());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        drop(state);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn denied_growth_with_only_empty_partitions_never_spills_or_inserts_unaccounted() {
        let (_calibration_directory, calibration_spill) = create_manager();
        let calibration_manager = buffer_manager_with_exact_budget(1 << 20);
        let calibration = accounted_i64_state(calibration_spill, &calibration_manager, 2);
        let initial_capacity = calibration.granted_bytes();
        drop(calibration);

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(initial_capacity);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 2);
        let error = state
            .get_or_insert_with_accounted(key_for_partition(&state, 0), 0, || 10)
            .unwrap_err();

        assert!(
            matches!(error, PartitionOperationError::Io(ref error) if error.kind() == std::io::ErrorKind::OutOfMemory)
        );
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), initial_capacity);
    }

    #[test]
    fn denied_writer_buffer_admission_precedes_allocation_and_file_creation() {
        let writer_provisional = qualified_writer_buffer_requested_bytes() * 2;
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::Create,
            usize::MAX,
            control.cancellation_handle(),
        ));
        let spill_manager = cancellation_manager(&directory, Arc::clone(&io));
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 1);
        *state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap() += 1;
        let resident = buffer_manager.allocated();
        let leave_available = writer_provisional.saturating_sub(1);
        let filler = buffer_manager
            .try_allocate(
                buffer_manager.available() - leave_available,
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let before_spill = buffer_manager.allocated();

        let error = state.spill_partition_controlled(0).unwrap_err();

        assert!(
            matches!(error, PartitionOperationError::Io(ref error) if error.kind() == std::io::ErrorKind::OutOfMemory)
        );
        assert_eq!(state.total_size(), 1);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(io.matching.load(Ordering::Acquire), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), before_spill);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), resident);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn denied_reader_construction_admission_cleans_consumed_cursor_state() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::ReadOpen,
            usize::MAX,
            control.cancellation_handle(),
        ));
        let spill_manager = cancellation_manager(&directory, Arc::clone(&io));
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(Arc::clone(&spill_manager), &buffer_manager, 1);
        *state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap() += 1;
        state.spill_partition_controlled(0).unwrap();
        let filler = buffer_manager
            .try_allocate(buffer_manager.available(), MemoryRegion::ExecutionBuffers)
            .unwrap();

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let Err(error) = cursor.next_entry() else {
            panic!("reader construction must fail at its admission boundary");
        };

        assert!(
            matches!(error, PartitionOperationError::Io(ref error) if error.kind() == std::io::ErrorKind::OutOfMemory)
        );
        drop(cursor);
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(io.matching.load(Ordering::Acquire), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn callback_panic_authority_legacy_default_survives_until_payload_drop() {
        const CONSTRUCTION_BOUND: usize = 64 * 1024;
        const PANIC_BYTES: usize = 32 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[1]);
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let panic_manager = Arc::clone(&buffer_manager);
        let drop_observation = Arc::clone(&allocated_when_dropped);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.get_or_insert_with_accounted(lookup.clone(), CONSTRUCTION_BOUND, || {
                std::panic::panic_any(HeapDropProbe {
                    bytes: vec![0xde; PANIC_BYTES],
                    buffer_manager: panic_manager,
                    allocated_when_dropped: drop_observation,
                })
            });
        }))
        .expect_err("accounted default must unwind");

        assert_eq!(state.total_size(), 0);
        assert_eq!(
            buffer_manager.allocated(),
            state.granted_bytes() + CONSTRUCTION_BOUND,
            "the escaped default-panic payload must retain its construction authority"
        );
        let accounted_panic = panic
            .downcast_ref::<AccountedPartitionPanic>()
            .expect("default panic must be paired with construction authority");
        assert_eq!(
            accounted_panic
                .payload()
                .downcast_ref::<HeapDropProbe>()
                .map(|payload| payload.bytes.len()),
            Some(PANIC_BYTES)
        );
        let allocated_with_panic = buffer_manager.allocated();
        drop(panic);
        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            allocated_with_panic,
            "the default panic payload must be destroyed before its authority"
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap() += 1;
        assert_eq!(state.get(&lookup).unwrap(), Some(&11));
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn callback_panic_authority_legacy_capacity_drops_value_before_escape() {
        const CONSTRUCTION_BOUND: usize = 64 * 1024;
        const VALUE_BYTES: usize = 24 * 1024;
        const PANIC_BYTES: usize = 16 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let value_drop_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_panic_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let panic_payload_drop_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_manager = Arc::clone(&buffer_manager);
        let capacity_panic_observation = Arc::clone(&capacity_panic_allocation);
        let panic_payload_drop_observation = Arc::clone(&panic_payload_drop_allocation);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &HeapValueDropProbe, writer: &mut dyn Write, _limits| {
                writer.write_all(&value.bytes)
            },
            |_reader: &mut dyn Read, _limits| {
                Err(std::io::Error::other(
                    "heap value probe decoder is unused in this test",
                ))
            },
            move |value: &HeapValueDropProbe| {
                if value.bytes.first() == Some(&0x22) {
                    capacity_panic_observation
                        .store(capacity_manager.allocated(), Ordering::Release);
                    std::panic::panic_any(HeapDropProbe {
                        bytes: vec![0xca; PANIC_BYTES],
                        buffer_manager: Arc::clone(&capacity_manager),
                        allocated_when_dropped: Arc::clone(&panic_payload_drop_observation),
                    });
                }
                Ok(value.bytes.capacity())
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[81]);
        let value_manager = Arc::clone(&buffer_manager);
        let value_drop_observation = Arc::clone(&value_drop_allocation);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.get_or_insert_with_accounted(lookup.clone(), CONSTRUCTION_BOUND, || {
                HeapValueDropProbe::observed(
                    vec![0x22; VALUE_BYTES],
                    value_manager,
                    value_drop_observation,
                )
            });
        }))
        .expect_err("legacy capacity callback must unwind");

        assert_eq!(
            value_drop_allocation.load(Ordering::Acquire),
            capacity_panic_allocation.load(Ordering::Acquire),
            "the built value must be physically destroyed while its construction authority remains live"
        );
        assert_eq!(state.total_size(), 0);
        assert_eq!(
            buffer_manager.allocated(),
            state.granted_bytes() + CONSTRUCTION_BOUND,
            "the escaped capacity-panic payload must retain its construction authority"
        );
        let accounted_panic = panic
            .downcast_ref::<AccountedPartitionPanic>()
            .expect("legacy capacity panic must be paired with construction authority");
        assert_eq!(
            accounted_panic
                .payload()
                .downcast_ref::<HeapDropProbe>()
                .map(|payload| payload.bytes.len()),
            Some(PANIC_BYTES)
        );
        let allocated_with_panic = buffer_manager.allocated();
        drop(panic);
        assert_eq!(
            panic_payload_drop_allocation.load(Ordering::Acquire),
            allocated_with_panic,
            "the capacity panic payload must be destroyed before its authority"
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );

        state
            .get_or_insert_with_accounted(lookup.clone(), 8, || {
                HeapValueDropProbe::unobserved(vec![0x33; 8], Arc::clone(&buffer_manager))
            })
            .unwrap();
        assert_eq!(
            state
                .get(&lookup)
                .unwrap()
                .map(|value| value.bytes.as_slice()),
            Some([0x33; 8].as_slice())
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn capacity_error_authority_legacy_value_and_error_remain_covered() {
        const CONSTRUCTION_BOUND: usize = 64 * 1024;
        const VALUE_BYTES: usize = 24 * 1024;
        const DENIED_BYTES: usize = 23;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let value_drop_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_error_allocation = Arc::new(AtomicUsize::new(usize::MAX));
        let capacity_manager = Arc::clone(&buffer_manager);
        let capacity_error_observation = Arc::clone(&capacity_error_allocation);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &HeapValueDropProbe, writer: &mut dyn Write, _limits| {
                writer.write_all(&value.bytes)
            },
            |_reader: &mut dyn Read, _limits| {
                Err(std::io::Error::other(
                    "heap value probe decoder is unused in this test",
                ))
            },
            move |value: &HeapValueDropProbe| {
                if value.bytes.first() == Some(&0x22) {
                    capacity_error_observation
                        .store(capacity_manager.allocated(), Ordering::Release);
                    return Err(MemoryGrantError::Denied {
                        additional_bytes: DENIED_BYTES,
                    });
                }
                Ok(value.bytes.capacity())
            },
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[82]);
        let value_manager = Arc::clone(&buffer_manager);
        let value_drop_observation = Arc::clone(&value_drop_allocation);

        let result = state.get_or_insert_with_accounted(lookup.clone(), CONSTRUCTION_BOUND, || {
            HeapValueDropProbe::observed(
                vec![0x22; VALUE_BYTES],
                value_manager,
                value_drop_observation,
            )
        });

        let Err(error) = result else {
            panic!("legacy capacity refusal must fail insertion");
        };
        assert!(matches!(
            error.resident_memory_error(),
            Some(MemoryGrantError::Denied { additional_bytes })
                if *additional_bytes == DENIED_BYTES
        ));
        assert_eq!(
            value_drop_allocation.load(Ordering::Acquire),
            capacity_error_allocation.load(Ordering::Acquire),
            "the completed value must be destroyed while full construction authority remains live"
        );
        assert_eq!(state.total_size(), 0);
        assert_eq!(
            buffer_manager.allocated(),
            state.granted_bytes() + CONSTRUCTION_BOUND,
            "the returned heap-backed error must retain construction authority until Drop"
        );

        drop(error);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        state
            .get_or_insert_with_accounted(lookup.clone(), 8, || {
                HeapValueDropProbe::unobserved(vec![0x33; 8], Arc::clone(&buffer_manager))
            })
            .unwrap();
        assert_eq!(
            state
                .get(&lookup)
                .unwrap()
                .map(|value| value.bytes.as_slice()),
            Some([0x33; 8].as_slice())
        );
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn accounted_spilled_base_revisit_hydrates_only_one_bounded_delta_entry() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[1]);
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap() += 1;
        state.spill_partition_controlled(0).unwrap();
        let after_spill = buffer_manager.allocated();

        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 99)
            .unwrap() += 5;

        assert_eq!(
            state
                .get_or_insert_with_accounted(lookup, 0, || 99)
                .map(|value| *value)
                .unwrap(),
            16
        );
        assert_eq!(state.total_size(), 1);
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.partitions[0].as_ref().unwrap().len(), 1);
        assert!(buffer_manager.allocated() > after_spill);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_spilled_base_hydration_retains_its_mutation_bound() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_bytes_state(
            spill_manager,
            &buffer_manager,
            crate::execution::QueryExecutionControl::new().token(),
        );
        let lookup = key(&[1]);
        let bound = Vec::<u8>::with_capacity(64).capacity();
        state
            .get_or_insert_with_accounted(lookup.clone(), bound, Vec::new)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();

        let value = state
            .get_or_insert_with_accounted(lookup, bound, || {
                panic!("matching spilled-base hydration must not build a default")
            })
            .unwrap();
        let grant_before_growth = buffer_manager.allocated();
        let grown = Vec::with_capacity(bound);
        assert_eq!(grown.capacity(), bound);
        *value = grown;
        assert_eq!(buffer_manager.allocated(), grant_before_growth);

        assert_eq!(state.partition_sizes[0], 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.partitions[0].as_ref().unwrap().len(), 1);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn spilled_base_decode_spill_carries_budget_through_mutation_bound_hydration() {
        const STORED_BOUND: usize = 1024 * 1024;
        const TRANSIENT_HEADROOM: usize = 512 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(8 * 1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let mut state = accounted_bytes_state_with_partitions(
            Arc::clone(&spill_manager),
            &buffer_manager,
            control.token(),
            2,
        );
        state.partitions[0]
            .as_mut()
            .unwrap()
            .try_reserve(8192)
            .unwrap();
        let cold = key_for_partition(&state, 0);
        let serialized_cold = SerializedKey::from_values(&cold, state.frame_limits).unwrap();
        state.partitions[0].as_mut().unwrap().insert(
            serialized_cold.clone(),
            PartitionEntry {
                num_key_columns: 1,
                resident_bound: 0,
                value: Vec::new(),
            },
        );
        state.partition_sizes[0] = 1;
        state.reconcile_grant().unwrap();

        let target = key_for_partition(&state, 1);
        state
            .get_or_insert_with_accounted(target.clone(), STORED_BOUND, Vec::new)
            .unwrap();
        state.spill_partition_controlled(1).unwrap();
        state.access_times = vec![1, 2];
        state.timestamp = 2;
        let cold_reclaim = PartitionedState::<Vec<u8>>::partition_map_allocation_bytes(
            state.partitions[0].as_ref().unwrap(),
        )
        .checked_add(serialized_cold.0.capacity())
        .unwrap();
        assert!(cold_reclaim > STORED_BOUND - TRANSIENT_HEADROOM);
        let cold_spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(
                serialized_cold
                    .0
                    .len()
                    .checked_add(32)
                    .unwrap()
                    .checked_mul(4)
                    .unwrap(),
            )
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        assert!(cold_spill_peak < TRANSIENT_HEADROOM);
        let filler = buffer_manager
            .try_allocate(
                buffer_manager
                    .available()
                    .checked_sub(TRANSIENT_HEADROOM)
                    .unwrap(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();

        let value = state
            .get_or_insert_with_accounted(target.clone(), STORED_BOUND, || {
                panic!("matching spilled-base hydration must not build a default")
            })
            .unwrap();

        assert!(value.is_empty());
        assert_eq!(state.spilled_count(), 2);
        assert_eq!(state.spill_base_sizes, vec![1, 1]);
        assert!(
            state.partitions[0]
                .as_ref()
                .is_some_and(PartitionMap::is_empty)
        );
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(|partition| partition.len() == 1)
        );
        assert_eq!(
            state
                .get_or_insert_with_accounted(target, STORED_BOUND, || {
                    panic!("resident hydration must not rebuild the default")
                })
                .map(|value| value.len())
                .unwrap(),
            0
        );
        drop(filler);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state
                .observed_resident_capacity_bytes()
                .unwrap()
                .checked_add(STORED_BOUND)
                .unwrap(),
            "legacy mutable access retains its stored growth authority"
        );
    }

    #[test]
    fn accounted_partition_cursor_streams_one_grant_owned_entry_at_a_time() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 4);
        for value in 0..64 {
            *state
                .get_or_insert_with_accounted(key(&[value]), 0, || value)
                .unwrap() += 1;
        }
        for partition in 0..4 {
            state.spill_partition_controlled(partition).unwrap();
        }

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let mut values = Vec::new();
        while let Some(entry) = cursor.next_entry().unwrap() {
            assert_eq!(cursor.active_partition_count(), 1);
            assert!(entry.granted_bytes() > 0);
            values.push(*entry.value());
        }
        drop(cursor);

        values.sort_unstable();
        assert_eq!(values, (1..=64).collect::<Vec<_>>());
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_partition_entry_splits_exact_independently_lived_owners() {
        const VALUE_BOUND: usize = 4096;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_bytes_state(
            spill_manager,
            &buffer_manager,
            crate::execution::QueryExecutionControl::new().token(),
        );
        state
            .get_or_insert_with_accounted(key(&[7]), VALUE_BOUND, || vec![0x5a; 64])
            .unwrap();
        state.spill_partition_controlled(0).unwrap();

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let entry = cursor.next_entry().unwrap().unwrap();
        let combined_bytes = entry.granted_bytes();
        let allocated_with_entry = buffer_manager.allocated();
        let (accounted_key, accounted_value) = entry.into_accounted_parts();

        assert_eq!(accounted_key.key(), [Value::Int64(7)]);
        assert_eq!(accounted_value.value().as_slice(), [0x5a; 64]);
        assert_eq!(accounted_value.granted_bytes(), VALUE_BOUND);
        assert_eq!(
            accounted_key
                .granted_bytes()
                .checked_add(accounted_value.granted_bytes()),
            Some(combined_bytes)
        );

        let key_bytes = accounted_key.granted_bytes();
        drop(accounted_key);
        assert_eq!(
            buffer_manager.allocated(),
            allocated_with_entry - key_bytes,
            "dropping the key owner must debit only its exact child"
        );
        drop(accounted_value);
        assert_eq!(
            buffer_manager.allocated(),
            allocated_with_entry - combined_bytes,
            "the value owner must retain an independent lifetime and debit only its child"
        );

        assert!(cursor.next_entry().unwrap().is_none());
        drop(cursor);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_partition_owners_release_exact_children_during_outer_unwind() {
        const VALUE_BOUND: usize = 4096;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_bytes_state(
            spill_manager,
            &buffer_manager,
            crate::execution::QueryExecutionControl::new().token(),
        );
        state
            .get_or_insert_with_accounted(key(&[8]), VALUE_BOUND, || vec![0x6b; 64])
            .unwrap();
        state.spill_partition_controlled(0).unwrap();

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let entry = cursor.next_entry().unwrap().unwrap();
        let owned_bytes = entry.granted_bytes();
        let allocated_with_entry = buffer_manager.allocated();
        let (accounted_key, accounted_value) = entry.into_accounted_parts();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _key_until_unwind = accounted_key;
            let _value_until_unwind = accounted_value;
            panic!("exercise independent accounted partition owner unwind");
        }));

        assert!(panic.is_err());
        assert_eq!(
            buffer_manager.allocated(),
            allocated_with_entry - owned_bytes,
            "ordinary physical owners and both exact children must unwind as one safe boundary"
        );
        assert!(cursor.next_entry().unwrap().is_none());
        drop(cursor);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn hostile_partition_value_drop_keeps_its_child_authority_fail_closed() {
        const VALUE_BOUND: usize = 4096;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let decoder_manager = Arc::clone(&buffer_manager);
        let decoder_observation = Arc::clone(&allocated_when_dropped);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            spill_manager,
            1,
            |value: &HostileDrainValue, writer: &mut dyn Write, _limits| {
                writer.write_all(&value.bytes)
            },
            move |reader: &mut dyn Read, _limits| {
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes)?;
                Ok(HostileDrainValue {
                    bytes,
                    buffer_manager: Arc::clone(&decoder_manager),
                    allocated_when_dropped: Some(Arc::clone(&decoder_observation)),
                    panic_on_drop: true,
                })
            },
            |value: &HostileDrainValue| Ok(value.bytes.capacity()),
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        state
            .get_or_insert_with_accounted(key(&[9]), VALUE_BOUND, || HostileDrainValue {
                bytes: vec![0x7c; 64],
                buffer_manager: Arc::clone(&buffer_manager),
                allocated_when_dropped: None,
                panic_on_drop: false,
            })
            .unwrap();
        state.spill_partition_controlled(0).unwrap();

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let entry = cursor.next_entry().unwrap().unwrap();
        let (accounted_key, accounted_value) = entry.into_accounted_parts();
        drop(accounted_key);
        let allocated_with_value = buffer_manager.allocated();
        assert_eq!(accounted_value.granted_bytes(), VALUE_BOUND);

        drop(accounted_value);

        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            allocated_with_value,
            "the hostile value destructor must observe its exact child authority live"
        );
        assert_eq!(
            buffer_manager.allocated(),
            allocated_with_value,
            "failed physical destruction must retain authority permanently"
        );
        assert!(cursor.next_entry().unwrap().is_none());
        drop(cursor);
        drop(state);
        assert_eq!(buffer_manager.allocated(), VALUE_BOUND);
    }

    #[test]
    fn shared_accounted_owner_backstop_is_fail_closed_for_key_physical_drop() {
        const OWNER_BYTES: usize = 1024;

        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let allocated_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let grant = buffer_manager
            .try_allocate(OWNER_BYTES, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let owner = AccountedPartitionOwner::new(
            PanickingDrainPhysical {
                buffer_manager: Arc::clone(&buffer_manager),
                allocated_when_dropped: Arc::clone(&allocated_when_dropped),
            },
            grant,
        );

        drop(owner);

        assert_eq!(
            allocated_when_dropped.load(Ordering::Acquire),
            OWNER_BYTES,
            "the shared key/value owner backstop must destroy physical state while authority is live"
        );
        assert_eq!(
            buffer_manager.allocated(),
            OWNER_BYTES,
            "a hostile physical Drop must permanently retain its authority"
        );
    }

    #[test]
    fn accounted_cursor_consolidates_spilled_base_and_revisit_delta_without_duplicates() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let first = key(&[1]);
        let second = key(&[2]);
        let third = key(&[3]);
        *state
            .get_or_insert_with_accounted(first.clone(), 0, || 10)
            .unwrap() += 1;
        *state
            .get_or_insert_with_accounted(second.clone(), 0, || 20)
            .unwrap() += 1;
        state.spill_partition_controlled(0).unwrap();

        *state
            .get_or_insert_with_accounted(first, 0, || 999)
            .unwrap() += 5;
        *state.get_or_insert_with_accounted(third, 0, || 30).unwrap() += 1;
        assert_eq!(state.total_size(), 3);
        assert_eq!(state.spill_base_sizes[0], 2);
        assert_eq!(state.partitions[0].as_ref().unwrap().len(), 2);

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let mut values = Vec::new();
        while let Some(entry) = cursor.next_entry().unwrap() {
            values.push(*entry.value());
        }
        drop(cursor);

        values.sort_unstable();
        assert_eq!(values, vec![16, 21, 31]);
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_consolidation_unwind_drops_staging_provider_before_workspace() {
        let directory = TempDir::new().unwrap();
        let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
        let resources =
            crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
        let (provider, witness) =
            super::super::framing_tests::grant_lifetime_provider(resources.clone(), 256 * 1024);
        let spill_manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(provider, SpillFrameLimits::format_max())
                .build()
                .unwrap(),
        );
        let serializer_calls = Arc::new(AtomicUsize::new(0));
        let serializer_call_count = Arc::clone(&serializer_calls);
        let grant = resources.try_allocate(0).unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&spill_manager),
            1,
            move |value: &i64, writer: &mut dyn Write, _limits| {
                assert!(
                    serializer_call_count.fetch_add(1, Ordering::AcqRel) != 1,
                    "deterministic consolidation serializer unwind"
                );
                serialize_i64(value, writer)
            },
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            |_value: &i64| Ok(0),
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        let lookup = key(&[1]);
        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap() += 1;
        state.spill_partition_controlled(0).unwrap();
        *state
            .get_or_insert_with_accounted(lookup, 0, || 99)
            .unwrap() += 5;

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.spill_partition_controlled(0);
        }))
        .unwrap_err();

        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"deterministic consolidation serializer unwind")
        );
        assert!(witness.drops() >= 3);
        assert!(
            !witness.released_before_drop(),
            "staging provider state outlived its admitted file-workspace grant"
        );
        assert_eq!(spill_manager.active_file_count(), 1);
        assert_eq!(
            resources.query_stats().allocated_bytes,
            state.granted_bytes()
        );

        state.spill_partition_controlled(0).unwrap();
        assert_eq!(spill_manager.active_file_count(), 1);
        state.cleanup().unwrap();
        drop(state);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_initial_open_record_destructor_cannot_replace_primary_or_release_grant_early() {
        const CHILD_ENV: &str = "GRAFEO_PARTITION_INITIAL_OPEN_RECORD_DROP_CHILD";
        const HANDSHAKE: &str = "GRAFEO_PARTITION_INITIAL_OPEN_RECORD_DROP_OK";
        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let buffer_manager = buffer_manager_with_exact_budget(4 * 1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let (inner, witness) =
                super::super::framing_tests::grant_lifetime_provider(resources.clone(), 256 * 1024);
            let spill_manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(PanicOnOpenRecordDropProvider { inner }),
                        SpillFrameLimits::format_max(),
                    )
                    .build()
                    .unwrap(),
            );
            let grant = resources.try_allocate(0).unwrap();
            let mut state = PartitionedState::new_accounted_with_cancellation(
                Arc::clone(&spill_manager),
                1,
                |_value: &i64, _writer: &mut dyn Write, _limits| {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "deterministic initial partition serializer failure",
                    ))
                },
                |reader: &mut dyn Read, _limits| deserialize_i64(reader),
                |_value: &i64| Ok(0),
                grant,
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
            state
                .get_or_insert_with_accounted(key(&[1]), 0, || 10)
                .unwrap();

            let error = state.spill_partition_controlled(0).unwrap_err();

            assert!(matches!(
                error,
                PartitionOperationError::Io(ref error)
                    if error.kind() == std::io::ErrorKind::PermissionDenied
                        && error.to_string()
                            == "deterministic initial partition serializer failure"
            ));
            assert_eq!(witness.drops(), 1);
            assert!(
                !witness.released_before_drop(),
                "initial spill provider state outlived its admitted file-workspace grant"
            );
            assert!(state.is_in_memory(0));
            assert_eq!(state.total_size(), 1);
            assert_eq!(state.spilled_count(), 0);
            assert_eq!(spill_manager.active_file_count(), 0);
            assert_eq!(
                resources.query_stats().allocated_bytes,
                state.granted_bytes()
            );
            state.cleanup().unwrap();
            drop(state);
            assert_eq!(resources.query_stats().allocated_bytes, 0);

            let legacy_directory = TempDir::new().unwrap();
            let legacy_manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(legacy_directory.path())
                    .provider(
                        Arc::new(PanicOnOpenRecordDropProvider {
                            inner: Arc::new(CleartextSpillRecordProvider),
                        }),
                        SpillFrameLimits::format_max(),
                    )
                    .build()
                    .unwrap(),
            );
            let mut legacy = PartitionedState::new(
                Arc::clone(&legacy_manager),
                1,
                |_value: &i64, _writer: &mut dyn Write| {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "deterministic legacy partition serializer failure",
                    ))
                },
                |reader: &mut dyn Read| deserialize_i64(reader),
            );
            legacy.insert(key(&[2]), 20).unwrap();

            let error = legacy.spill_partition(0).unwrap_err();

            assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
            assert_eq!(
                error.to_string(),
                "deterministic legacy partition serializer failure"
            );
            assert!(legacy.is_in_memory(0));
            assert_eq!(legacy.total_size(), 1);
            assert_eq!(legacy.spilled_count(), 0);
            assert_eq!(legacy_manager.active_file_count(), 0);
            legacy.cleanup().unwrap();
            println!("{HANDSHAKE}");
            return;
        }

        let test_name = "execution::spill::partition::tests::hostile_initial_open_record_destructor_cannot_replace_primary_or_release_grant_early";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(HANDSHAKE),
            "open-record-drop child did not preserve the initial spill primary and grant lifetime\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_provider_destructor_cannot_replace_partition_staging_unwind_or_strand_grant() {
        const CHILD_ENV: &str = "GRAFEO_PARTITION_STAGING_PROVIDER_DROP_CHILD";
        const HANDSHAKE: &str = "GRAFEO_PARTITION_STAGING_PROVIDER_DROP_OK";
        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let buffer_manager = buffer_manager_with_exact_budget(1024 * 1024);
            let resources =
                crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
            let manager = Arc::new(
                crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .provider(
                        Arc::new(PanicOnPartitionProviderDrop),
                        SpillFrameLimits::format_max(),
                    )
                    .build()
                    .unwrap(),
            );
            let mut root = resources.try_allocate(0).unwrap();
            let (file, workspace) =
                create_accounted_partition_file(&manager, &mut root, None).unwrap();
            assert!(workspace.size() > 0);
            let staging = PartitionStaging::accounted(file, workspace, None);
            drop(manager);

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _staging = staging;
                panic!("deterministic partition staging primary panic");
            }))
            .unwrap_err();

            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"deterministic partition staging primary panic")
            );
            assert_eq!(resources.query_stats().allocated_bytes, root.size());
            drop(root);
            assert_eq!(resources.query_stats().allocated_bytes, 0);
            println!("{HANDSHAKE}");
            return;
        }

        let test_name = "execution::spill::partition::tests::hostile_provider_destructor_cannot_replace_partition_staging_unwind_or_strand_grant";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(HANDSHAKE),
            "provider-drop child did not preserve the partition primary and release its grant\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[test]
    fn accounted_near_limit_frame_is_charged_through_write_read_and_decode() {
        let directory = TempDir::new().unwrap();
        let limits = SpillFrameLimits::new(4096, 4096).unwrap();
        let spill_manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(Arc::new(CleartextSpillRecordProvider), limits)
                .build()
                .unwrap(),
        );
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let write_observations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let read_observations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let write_buffer_manager = Arc::clone(&buffer_manager);
        let serializer_observations = Arc::clone(&write_observations);
        let serializer_calls = Arc::new(AtomicUsize::new(0));
        let serializer_call_count = Arc::clone(&serializer_calls);
        let read_buffer_manager = Arc::clone(&buffer_manager);
        let deserializer_observations = Arc::clone(&read_observations);
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&spill_manager),
            1,
            move |value: &Vec<u8>, writer: &mut dyn Write, _limits| {
                let call = serializer_call_count.fetch_add(1, Ordering::AcqRel);
                if call != 0 {
                    return Err(std::io::Error::other(
                        "qualified serializer must not be invoked twice",
                    ));
                }
                serializer_observations
                    .lock()
                    .unwrap()
                    .push(write_buffer_manager.allocated());
                writer.write_all(value)
            },
            move |reader: &mut dyn Read, _limits| {
                deserializer_observations
                    .lock()
                    .unwrap()
                    .push(read_buffer_manager.allocated());
                let mut value = Vec::new();
                reader.read_to_end(&mut value)?;
                Ok(value)
            },
            |value: &Vec<u8>| Ok(value.capacity()),
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        state
            .get_or_insert_with_accounted(key(&[1]), 3500, || vec![7; 3500])
            .unwrap();
        let resident = buffer_manager.allocated();

        state.spill_partition_controlled(0).unwrap();
        let writes = write_observations.lock().unwrap();
        assert_eq!(serializer_calls.load(Ordering::Acquire), 1);
        assert_eq!(writes.len(), 1, "qualified serializer runs exactly once");
        assert!(
            writes[0] > resident,
            "serializer runs after staging admission"
        );
        drop(writes);
        let after_spill = buffer_manager.allocated();
        assert_eq!(after_spill, state.granted_bytes());

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let entry = cursor.next_entry().unwrap().unwrap();
        assert_eq!(entry.value().len(), 3500);
        assert!(entry.granted_bytes() >= 3500);
        let reads = read_observations.lock().unwrap();
        assert_eq!(reads.len(), 1);
        assert!(
            reads[0] > after_spill,
            "frame and decode grants precede custom deserialization"
        );
        drop(reads);
        drop(entry);
        assert!(cursor.next_entry().unwrap().is_none());
        drop(cursor);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(spill_manager.active_file_count(), 0);
    }

    #[test]
    fn accounted_cursor_decode_failure_releases_frame_and_decode_workspaces() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let fail_decode = Arc::new(AtomicBool::new(false));
        let decoder_flag = Arc::clone(&fail_decode);
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&spill_manager),
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            move |reader: &mut dyn Read, _limits| {
                if decoder_flag.load(Ordering::Acquire) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "hostile aggregate decode failure",
                    ));
                }
                deserialize_i64(reader)
            },
            |_value: &i64| Ok(0),
            grant,
            crate::execution::QueryExecutionControl::new().token(),
        )
        .unwrap();
        state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();
        fail_decode.store(true, Ordering::Release);

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let error = cursor
            .next_entry()
            .err()
            .expect("hostile decoder must fail the consuming cursor");
        assert!(matches!(
            error,
            PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidData
        ));
        drop(cursor);
        assert_eq!(state.total_size(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_cursor_observes_cancellation_from_successful_decoder_before_returning_row() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let decoder_cancellation = control.cancellation_handle();
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&spill_manager),
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            move |reader: &mut dyn Read, _limits| {
                let value = deserialize_i64(reader)?;
                decoder_cancellation.cancel();
                Ok(value)
            },
            |_value: &i64| Ok(0),
            grant,
            control.token(),
        )
        .unwrap();
        state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let error = cursor
            .next_entry()
            .err()
            .expect("cancellation raised by a successful decoder must suppress its row");

        assert!(matches!(
            error,
            PartitionOperationError::Cancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        drop(cursor);
        assert_eq!(state.total_size(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_cursor_decoder_error_wins_when_decoder_also_cancels() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let control = crate::execution::QueryExecutionControl::new();
        let decoder_cancellation = control.cancellation_handle();
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&spill_manager),
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            move |reader: &mut dyn Read, _limits| {
                let _ = deserialize_i64(reader)?;
                decoder_cancellation.cancel();
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "decoder failure wins over cancellation",
                ))
            },
            |_value: &i64| Ok(0),
            grant,
            control.token(),
        )
        .unwrap();
        state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let error = cursor
            .next_entry()
            .err()
            .expect("a concrete decoder failure must remain primary");

        assert!(matches!(
            error,
            PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::InvalidData
                    && error.to_string().contains("decoder failure wins")
        ));
        drop(cursor);
        assert_eq!(state.total_size(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_cursor_cancellation_after_payload_releases_frame_workspace() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::ReadPayload,
            3,
            control.cancellation_handle(),
        ));
        let spill_manager = cancellation_manager(&directory, Arc::clone(&io));
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let grant = buffer_manager
            .try_allocate(0, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut state = PartitionedState::new_accounted_with_cancellation(
            Arc::clone(&spill_manager),
            1,
            |value: &i64, writer: &mut dyn Write, _limits| serialize_i64(value, writer),
            |reader: &mut dyn Read, _limits| deserialize_i64(reader),
            |_value: &i64| Ok(0),
            grant,
            control.token(),
        )
        .unwrap();
        state
            .get_or_insert_with_accounted(key(&[1]), 0, || 10)
            .unwrap();
        state.spill_partition_controlled(0).unwrap();

        let mut cursor = state.drain_partitioned_accounted().unwrap();
        let error = cursor
            .next_entry()
            .err()
            .expect("payload callback cancellation must terminate the cursor");
        assert!(matches!(
            error,
            PartitionOperationError::Cancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        drop(cursor);
        assert_eq!(state.total_size(), 0);
        assert_eq!(spill_manager.active_file_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn accounted_state_rejects_every_legacy_public_mutation_seam() {
        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(1 << 20);
        let mut state = accounted_i64_state(spill_manager, &buffer_manager, 1);
        let lookup = key(&[1]);

        assert_eq!(
            state.insert(lookup.clone(), 99).unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
        assert_eq!(state.total_size(), 0);

        *state
            .get_or_insert_with_accounted(lookup.clone(), 0, || 10)
            .unwrap() += 1;
        let default_called = std::cell::Cell::new(false);
        assert_eq!(
            state
                .get_or_insert_with(lookup.clone(), || {
                    default_called.set(true);
                    100
                })
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::Unsupported
        );
        assert!(!default_called.get());
        assert_eq!(state.get(&lookup).unwrap(), Some(&11));

        assert!(matches!(
            state
                .get_or_insert_with_controlled(lookup.clone(), || 200)
                .unwrap_err(),
            PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::Unsupported
        ));
        assert_eq!(state.get(&lookup).unwrap(), Some(&11));
        assert_eq!(
            state.drain_all().unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
        assert_eq!(
            state.iter_all().unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
        assert_eq!(state.total_size(), 1);
        assert!(state.spill_partition(0).unwrap() > 0);
        assert_eq!(state.spilled_count(), 1);
        state.cleanup().unwrap();
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
        assert_eq!(
            state.granted_bytes(),
            state.observed_resident_capacity_bytes().unwrap()
        );
        *state
            .get_or_insert_with_accounted(key(&[2]), 0, || 20)
            .unwrap() += 1;
    }

    #[test]
    fn qualified_spill_stops_after_a_successful_entry_and_cleans_staging() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::WritePayload,
            2,
            control.cancellation_handle(),
        ));
        let manager = cancellation_manager(&directory, Arc::clone(&io));
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();

        let error = state.spill_partition_controlled(0).unwrap_err();

        assert!(matches!(
            error,
            PartitionOperationError::Cancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(io.matching.load(Ordering::Relaxed), 2);
        assert!(state.is_in_memory(0));
        assert_eq!(state.total_size(), 1);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    }

    #[test]
    fn qualified_spill_io_error_wins_when_the_same_callback_cancels() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::with_failure(
            super::super::SpillIoOperation::WritePayload,
            2,
            control.cancellation_handle(),
            std::io::ErrorKind::PermissionDenied,
            "deterministic cancelled partition write failure",
        ));
        let manager = cancellation_manager(&directory, Arc::clone(&io));
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();

        let error = state.spill_partition_controlled(0).unwrap_err();

        assert!(matches!(
            error,
            PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
                    && error.to_string().contains("cancelled partition write failure")
        ));
        assert!(control.token().is_cancelled());
        assert!(state.is_in_memory(0));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    }

    #[test]
    fn qualified_spill_cancellation_retains_delete_failure_context() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(
            CancelNthIo::new(
                super::super::SpillIoOperation::WritePayload,
                2,
                control.cancellation_handle(),
            )
            .with_delete_failure(),
        );
        let manager = cancellation_manager(&directory, Arc::clone(&io));
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();

        let error = state.spill_partition_controlled(0).unwrap_err();

        let PartitionOperationError::CancelledWithCleanup {
            error: crate::execution::QueryCancellationError::Cancelled,
            cleanup,
            phase,
        } = error
        else {
            panic!("partition cancellation and cleanup failure were flattened")
        };
        assert_eq!(cleanup.kind(), std::io::ErrorKind::WouldBlock);
        assert!(cleanup.to_string().contains("partition delete failure"));
        assert_eq!(phase, "partition spill cleanup");
        assert!(state.is_in_memory(0));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 1);
        assert!(manager.disk_stats().reserved_live_bytes > 0);

        io.permit_delete();
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    }

    #[test]
    fn qualified_drain_cancellation_after_record_is_consuming_and_cleans_owned_state() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::new(
            super::super::SpillIoOperation::ReadPayload,
            2,
            control.cancellation_handle(),
        ));
        io.disarm();
        let manager = cancellation_manager(&directory, Arc::clone(&io));
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();
        state.insert(key(&[2]), 20).unwrap();
        state.spill_partition_controlled(0).unwrap();
        io.arm();

        let error = state.drain_all_controlled().unwrap_err();

        assert!(matches!(
            error,
            PartitionOperationError::Cancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(io.matching.load(Ordering::Relaxed), 2);
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    }

    #[test]
    fn qualified_drain_read_error_wins_when_the_same_callback_cancels() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(CancelNthIo::with_failure(
            super::super::SpillIoOperation::ReadPayload,
            2,
            control.cancellation_handle(),
            std::io::ErrorKind::PermissionDenied,
            "deterministic cancelled partition read failure",
        ));
        io.disarm();
        let manager = cancellation_manager(&directory, Arc::clone(&io));
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();
        state.spill_partition_controlled(0).unwrap();
        io.arm();

        let error = state.drain_all_controlled().unwrap_err();

        assert!(matches!(
            error,
            PartitionOperationError::Io(ref error)
                if error.kind() == std::io::ErrorKind::PermissionDenied
                    && error.to_string().contains("cancelled partition read failure")
        ));
        assert!(control.token().is_cancelled());
        assert_eq!(state.total_size(), 1);
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(manager.active_file_count(), 1);
        state.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn qualified_drain_cancellation_retains_cleanup_failure_and_retry() {
        let directory = TempDir::new().unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let io = Arc::new(
            CancelNthIo::new(
                super::super::SpillIoOperation::ReadPayload,
                2,
                control.cancellation_handle(),
            )
            .with_delete_failure(),
        );
        io.disarm();
        let manager = cancellation_manager(&directory, Arc::clone(&io));
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();
        state.spill_partition_controlled(0).unwrap();
        io.arm();

        let error = state.drain_all_controlled().unwrap_err();

        let PartitionOperationError::CancelledWithCleanup {
            error: crate::execution::QueryCancellationError::Cancelled,
            cleanup,
            phase,
        } = error
        else {
            panic!("partition drain cancellation lost cleanup context")
        };
        assert_eq!(cleanup.kind(), std::io::ErrorKind::WouldBlock);
        assert!(cleanup.to_string().contains("partition delete failure"));
        assert_eq!(phase, "cancelled partition drain cleanup");
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(manager.active_file_count(), 1);

        io.permit_delete();
        state.cleanup().unwrap();
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn pre_cancelled_qualified_drain_preserves_state_before_consuming_boundary() {
        let (_directory, manager) = create_manager();
        let control = crate::execution::QueryExecutionControl::new();
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();
        state.spill_partition_controlled(0).unwrap();
        let published = manager.spilled_bytes();
        control.cancellation_handle().cancel();

        let error = state.drain_all_controlled().unwrap_err();

        assert!(matches!(
            error,
            PartitionOperationError::Cancelled(crate::execution::QueryCancellationError::Cancelled)
        ));
        assert_eq!(state.total_size(), 1);
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), published);
    }

    #[test]
    fn qualified_state_public_compatibility_spill_and_drain_ignore_stored_cancellation() {
        let (_directory, manager) = create_manager();
        let control = crate::execution::QueryExecutionControl::new();
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();
        control.cancellation_handle().cancel();

        state.spill_partition(0).unwrap();
        let drained = state.drain_all().unwrap();

        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, key(&[1]));
        assert_eq!(drained[0].1, 10);
        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn qualified_partition_preserves_deadline_reason() {
        let (_directory, manager) = create_manager();
        let timeout = std::time::Duration::ZERO;
        let control = crate::execution::QueryExecutionControl::with_timeout(timeout).unwrap();
        let mut state = qualified_i64_state(Arc::clone(&manager), control.token());
        state.insert(key(&[1]), 10).unwrap();

        let error = state.spill_partition_controlled(0).unwrap_err();

        assert!(matches!(
            error,
            PartitionOperationError::Cancelled(
                crate::execution::QueryCancellationError::DeadlineExceeded {
                    timeout: Some(observed)
                }
            ) if observed == timeout
        ));
        assert!(state.is_in_memory(0));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn test_partition_for() {
        let (_temp_dir, manager) = create_manager();
        let state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // Same key should always go to same partition
        let k1 = key(&[1, 2, 3]);
        let p1 = state.partition_for(&k1);
        let p2 = state.partition_for(&k1);
        assert_eq!(p1, p2);

        // Partition should be in range
        assert!(p1 < 16);
    }

    #[test]
    fn test_insert_and_get() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // Insert some values
        state.insert(key(&[1]), 100).unwrap();
        state.insert(key(&[2]), 200).unwrap();
        state.insert(key(&[3]), 300).unwrap();

        assert_eq!(state.total_size(), 3);

        // Get values
        assert_eq!(state.get(&key(&[1])).unwrap(), Some(&100));
        assert_eq!(state.get(&key(&[2])).unwrap(), Some(&200));
        assert_eq!(state.get(&key(&[3])).unwrap(), Some(&300));
        assert_eq!(state.get(&key(&[4])).unwrap(), None);
    }

    #[test]
    fn test_get_or_insert_with() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // First access creates the entry
        let v1 = state.get_or_insert_with(key(&[1]), || 42).unwrap();
        assert_eq!(*v1, 42);

        // Second access returns existing value
        let v2 = state.get_or_insert_with(key(&[1]), || 100).unwrap();
        assert_eq!(*v2, 42);

        // Mutate via returned reference
        *state.get_or_insert_with(key(&[1]), || 0).unwrap() = 999;
        assert_eq!(state.get(&key(&[1])).unwrap(), Some(&999));
    }

    #[test]
    fn insert_propagates_key_serialization_refusal() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        let error = state.insert(over_depth_key(), 42).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(state.total_size(), 0);
    }

    #[test]
    fn get_propagates_key_serialization_refusal() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);
        let key = over_depth_key();

        let error = state.get(&key).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(state.total_size(), 0);
    }

    #[test]
    fn get_or_insert_with_propagates_key_serialization_refusal() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);
        let default_called = std::cell::Cell::new(false);

        let error = state
            .get_or_insert_with(over_depth_key(), || {
                default_called.set(true);
                42
            })
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!default_called.get());
        assert_eq!(state.total_size(), 0);
    }

    #[test]
    fn test_spill_and_reload() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values that go to different partitions
        for i in 0..20 {
            state.insert(key(&[i]), i * 10).unwrap();
        }

        let initial_total = state.total_size();
        assert!(initial_total > 0);

        // Spill the largest partition
        let bytes_spilled = state.spill_largest().unwrap();
        assert!(bytes_spilled > 0);
        assert!(state.spilled_count() > 0);

        // Values should still be accessible (reloads from disk)
        for i in 0..20 {
            let expected = i * 10;
            assert_eq!(state.get(&key(&[i])).unwrap(), Some(&expected));
        }
    }

    #[test]
    fn spilled_partition_starts_with_the_v1_frame_magic() {
        let (temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 1, serialize_i64, deserialize_i64);
        state.insert(key(&[1]), 10).unwrap();
        state.spill_partition(0).unwrap();

        let path = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let bytes = std::fs::read(path).unwrap();

        assert_eq!(&bytes[..4], b"GRSP");
    }

    #[test]
    fn framed_partition_key_round_trips_exact_rdf_and_nested_values() {
        let (_directory, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 1, serialize_i64, deserialize_i64);
        let mut map = BTreeMap::new();
        map.insert(
            grafeo_common::types::PropertyKey::new("nested"),
            Value::List(Arc::from([Value::Null, Value::Bool(true)])),
        );
        let mut grow = HashMap::new();
        grow.insert("replica-a".to_owned(), 1);
        let mut positive = HashMap::new();
        positive.insert("replica-a".to_owned(), 7);
        let mut negative = HashMap::new();
        negative.insert("replica-b".to_owned(), 3);
        let exact_key = vec![
            Value::RdfLiteral {
                lexical: "plain".into(),
                language: None,
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "colour".into(),
                language: Some("EN".into()),
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "18446744073709551616".into(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
            },
            Value::Time(
                grafeo_common::types::Time::from_nanos(12_345)
                    .unwrap()
                    .with_offset(-3_600),
            ),
            Value::List(Arc::from([Value::Vector(Arc::from([1.0_f32, -2.5, 3.25]))])),
            Value::Map(Arc::new(map)),
            Value::Path {
                nodes: Arc::from([Value::Int64(1), Value::Int64(2)]),
                edges: Arc::from([Value::String("edge".into())]),
            },
            Value::GCounter(Arc::new(grow)),
            Value::OnCounter {
                pos: Arc::new(positive),
                neg: Arc::new(negative),
            },
        ];
        state.insert(exact_key.clone(), 42).unwrap();
        state.spill_partition(0).unwrap();

        assert_eq!(state.get(&exact_key).unwrap(), Some(&42));
    }

    #[test]
    fn wire_record_limits_are_independent_from_the_decoded_codec_grant() {
        let compact_key = vec![Value::List(Arc::from(vec![Value::Null; 8]))];

        let roomy_directory = TempDir::new().unwrap();
        let roomy_limits = SpillFrameLimits::new(64, 64)
            .unwrap()
            .with_codec_limits(CodecLimits::new(1024, 64, 8, 16));
        let roomy_manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(roomy_directory.path())
                .provider(Arc::new(CleartextSpillRecordProvider), roomy_limits)
                .build()
                .unwrap(),
        );
        let mut roomy: PartitionedState<i64> = PartitionedState::new(
            Arc::clone(&roomy_manager),
            1,
            serialize_i64,
            deserialize_i64,
        );
        roomy.insert(compact_key.clone(), 42).unwrap();
        roomy.spill_partition(0).unwrap();
        let denied_limits = SpillFrameLimits::new(64, 64)
            .unwrap()
            .with_codec_limits(CodecLimits::new(32, 64, 8, 16));
        roomy.frame_limits = denied_limits;
        let spill_file = roomy.spill_files[0].as_ref().unwrap();
        let denied = roomy
            .load_partition(spill_file, 1)
            .err()
            .expect("low codec grant must reject the framed partition");
        assert_eq!(denied.kind(), std::io::ErrorKind::InvalidData);
        assert!(roomy.partitions[0].is_none());
        assert!(roomy.spill_files[0].is_some());
        roomy.frame_limits = roomy_limits;
        assert_eq!(roomy.get(&compact_key).unwrap(), Some(&42));

        let small_directory = TempDir::new().unwrap();
        let small_limits = denied_limits;
        let small_manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(small_directory.path())
                .provider(Arc::new(CleartextSpillRecordProvider), small_limits)
                .build()
                .unwrap(),
        );
        let mut small: PartitionedState<i64> = PartitionedState::new(
            Arc::clone(&small_manager),
            1,
            serialize_i64,
            deserialize_i64,
        );

        assert_eq!(
            small.insert(compact_key, 42).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(small.total_size(), 0);
        assert_eq!(small_manager.active_file_count(), 0);
        assert_eq!(small_manager.spilled_bytes(), 0);
    }

    #[test]
    fn duplicate_partition_key_is_rejected_instead_of_overwritten() {
        let (_directory, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        let serialized = SerializedKey::from_values(&key(&[1]), state.frame_limits).unwrap();
        let payload = encode_partition_entry(&serialized.0, 1, &10i64.to_le_bytes());
        install_partition_file(&mut state, &manager, &[payload.clone(), payload]);

        let error = state.get(&key(&[1])).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(state.spill_files[0].is_some());
        assert!(state.partitions[0].is_none());
    }

    #[test]
    fn framed_partition_count_must_match_the_tracked_partition_size() {
        let (_directory, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        let serialized = SerializedKey::from_values(&key(&[1]), state.frame_limits).unwrap();
        let payload = encode_partition_entry(&serialized.0, 1, &10i64.to_le_bytes());
        install_partition_file(&mut state, &manager, &[payload]);
        state.partition_sizes[0] = 2;

        assert_eq!(
            state.get(&key(&[1])).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert!(state.spill_files[0].is_some());
        assert!(state.partitions[0].is_none());
    }

    #[test]
    fn delete_failure_after_decode_preserves_the_spill_for_atomic_reload_retry() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Delete,
                    1,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        let lookup = key(&[1]);
        state.insert(lookup.clone(), 10).unwrap();
        state.spill_partition(0).unwrap();
        let published = manager.spilled_bytes();

        assert_eq!(
            state.get(&lookup).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert!(state.partitions[0].is_none());
        assert!(state.spill_files[0].is_some());
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), published);

        assert_eq!(state.get(&lookup).unwrap(), Some(&10));
        assert!(state.partitions[0].is_some());
        assert!(state.spill_files[0].is_none());
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
    }

    #[test]
    fn partition_key_and_custom_state_require_exact_payload_consumption() {
        let (_directory, manager) = create_manager();
        let serialized = SerializedKey::from_values(&key(&[1]), manager.frame_limits()).unwrap();

        let mut bad_key = serialized.0.clone();
        bad_key.push(0xff);
        let key_payload = encode_partition_entry(&bad_key, 1, &10i64.to_le_bytes());
        let mut key_state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        install_partition_file(&mut key_state, &manager, &[key_payload]);
        assert_eq!(
            key_state.get(&key(&[1])).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        let mut custom = 10i64.to_le_bytes().to_vec();
        custom.push(0xff);
        let custom_payload = encode_partition_entry(&serialized.0, 1, &custom);
        let mut custom_state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        install_partition_file(&mut custom_state, &manager, &[custom_payload]);
        assert_eq!(
            custom_state.get(&key(&[1])).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn partition_length_bomb_is_rejected_before_key_reserve() {
        let (_directory, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        let payload = u64::MAX.to_le_bytes().to_vec();
        install_partition_file(&mut state, &manager, &[payload]);

        let error = state.get(&key(&[1])).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "partition key length exceeds framed payload"
        );
    }

    #[test]
    fn many_entries_spill_with_per_entry_limit_not_whole_partition_staging() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::new(64, 64).unwrap(),
                )
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        for value in 0..100 {
            state.insert(key(&[value]), value).unwrap();
        }

        state.spill_partition(0).unwrap();

        assert_eq!(state.spilled_count(), 1);
        assert!(manager.spilled_bytes() > 64);
    }

    #[test]
    fn entry_over_limit_preserves_resident_partition_and_returns_error() {
        fn serialize_bytes(value: &[u8], writer: &mut dyn Write) -> std::io::Result<()> {
            writer.write_all(value)
        }
        fn deserialize_bytes(reader: &mut dyn Read) -> std::io::Result<Vec<u8>> {
            let mut value = Vec::new();
            reader.read_to_end(&mut value)?;
            Ok(value)
        }

        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::new(64, 64).unwrap(),
                )
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<Vec<u8>> = PartitionedState::new(
            Arc::clone(&manager),
            1,
            |value: &Vec<u8>, writer| serialize_bytes(value, writer),
            deserialize_bytes,
        );
        let value = vec![7u8; 80];
        state.insert(key(&[1]), value.clone()).unwrap();

        let error = state.spill_partition(0).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(state.is_in_memory(0));
        assert_eq!(state.get(&key(&[1])).unwrap(), Some(&value));
        assert_eq!(manager.spilled_bytes(), 0);
    }

    #[test]
    fn test_spill_lru() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values
        state.insert(key(&[1]), 10).unwrap();
        state.insert(key(&[2]), 20).unwrap();
        state.insert(key(&[3]), 30).unwrap();

        // Access key 3 to make it recently used
        state.get(&key(&[3])).unwrap();

        // Spill LRU - should not spill partition containing key 3
        state.spill_lru().unwrap();

        // Key 3 should still be in memory
        let partition_idx = state.partition_for(&key(&[3]));
        assert!(state.is_in_memory(partition_idx));
    }

    #[test]
    fn test_drain_all() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values
        for i in 0..10 {
            state.insert(key(&[i]), i * 10).unwrap();
        }

        // Spill some partitions
        state.spill_largest().unwrap();
        state.spill_largest().unwrap();

        // Drain all
        let entries = state.drain_all().unwrap();
        assert_eq!(entries.len(), 10);

        // Verify all entries are present
        let mut values: Vec<i64> = entries.iter().map(|(_, v)| *v).collect();
        values.sort_unstable();
        assert_eq!(values, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);

        // State should be empty
        assert_eq!(state.total_size(), 0);
        state.insert(key(&[99]), 990).unwrap();
        assert_eq!(state.get(&key(&[99])).unwrap(), Some(&990));
    }

    #[test]
    fn failed_partial_drain_is_terminal_but_cleanup_remains_retryable() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 2, serialize_i64, deserialize_i64);
        let valid = SerializedKey::from_values(&key(&[1]), state.frame_limits).unwrap();
        state.partitions[0].as_mut().unwrap().insert(
            valid,
            PartitionEntry {
                num_key_columns: 1,
                resident_bound: 0,
                value: 10,
            },
        );
        state.partition_sizes[0] = 1;
        state.partitions[1].as_mut().unwrap().insert(
            SerializedKey(vec![0xff]),
            PartitionEntry {
                num_key_columns: 1,
                resident_bound: 0,
                value: 20,
            },
        );
        state.partition_sizes[1] = 1;

        assert_eq!(
            state.drain_all().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            state.drain_all().unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(
            state.insert(key(&[2]), 30).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        state.cleanup().unwrap();
        state.insert(key(&[3]), 40).unwrap();
        assert_eq!(state.get(&key(&[3])).unwrap(), Some(&40));
    }

    #[test]
    fn drain_failure_before_destructive_progress_remains_retryable() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::ReadOpen,
                    1,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 1, serialize_i64, deserialize_i64);
        state.insert(key(&[1]), 10).unwrap();
        state.spill_partition(0).unwrap();

        assert_eq!(
            state.drain_all().unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(state.drain_state, DrainState::Idle);

        let drained = state.drain_all().unwrap();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].1, 10);
        assert_eq!(state.drain_state, DrainState::Idle);
    }

    #[test]
    fn test_iter_all() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values
        for i in 0..5 {
            state.insert(key(&[i]), i * 10).unwrap();
        }

        // Iterate without draining
        let entries = state.iter_all().unwrap();
        assert_eq!(entries.len(), 5);

        // State should still have values
        assert_eq!(state.total_size(), 5);

        // Should be able to iterate again
        let entries2 = state.iter_all().unwrap();
        assert_eq!(entries2.len(), 5);
    }

    #[test]
    fn test_many_groups() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // Insert many groups
        for i in 0..1000 {
            state.insert(key(&[i]), i).unwrap();
        }

        assert_eq!(state.total_size(), 1000);

        // Spill multiple partitions
        for _ in 0..8 {
            state.spill_largest().unwrap();
        }

        assert!(state.spilled_count() >= 8);

        // All values should still be retrievable
        for i in 0..1000 {
            assert_eq!(state.get(&key(&[i])).unwrap(), Some(&i));
        }
    }

    #[test]
    fn test_cleanup() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 4, serialize_i64, deserialize_i64);

        // Insert and spill
        for i in 0..20 {
            state.insert(key(&[i]), i).unwrap();
        }
        state.spill_largest().unwrap();
        state.spill_largest().unwrap();

        let spilled_before = manager.spilled_bytes();
        assert!(spilled_before > 0);

        // Cleanup
        state.cleanup().unwrap();

        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
    }

    #[test]
    fn drop_isolates_each_partition_cleanup_attempt_and_preserves_primary_panic() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(PanicThenAllowThenFailDeleteIo::new());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 2, serialize_i64, deserialize_i64);
        let first = (0..100)
            .map(|value| key(&[value]))
            .find(|candidate| state.partition_for(candidate) == 0)
            .unwrap();
        let second = (100..200)
            .map(|value| key(&[value]))
            .find(|candidate| state.partition_for(candidate) == 1)
            .unwrap();
        state.insert(first, 10).unwrap();
        state.insert(second, 20).unwrap();
        state.spill_partition(0).unwrap();
        state.spill_partition(1).unwrap();
        let first_path = state.spill_files[0].as_ref().unwrap().path().to_path_buf();
        let second_path = state.spill_files[1].as_ref().unwrap().path().to_path_buf();
        let before = SpillManager::orphan_cleanup_failures();

        let primary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _state = state;
            std::panic::panic_any(PrimaryDropPanic);
        }))
        .unwrap_err();

        assert!(primary.is::<PrimaryDropPanic>());
        assert_eq!(io.deletes.load(std::sync::atomic::Ordering::Acquire), 3);
        assert!(first_path.exists());
        assert!(!second_path.exists());
        assert_eq!(manager.active_file_count(), 1);
        assert!(SpillManager::orphan_cleanup_failures() > before);
        manager.cleanup().unwrap();
        assert!(!first_path.exists());
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn cleanup_failure_before_any_delete_preserves_reuse_and_retry() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Delete,
                    1,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 1, serialize_i64, deserialize_i64);
        let lookup = key(&[1]);
        state.insert(lookup.clone(), 10).unwrap();
        state.spill_partition(0).unwrap();
        let published = manager.spilled_bytes();

        assert_eq!(
            state.cleanup().unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(state.drain_state, DrainState::Idle);
        assert!(state.spill_files[0].is_some());
        assert_eq!(manager.spilled_bytes(), published);
        assert_eq!(state.get(&lookup).unwrap(), Some(&10));

        state.cleanup().unwrap();
        state.insert(key(&[2]), 20).unwrap();
        assert_eq!(state.get(&key(&[2])).unwrap(), Some(&20));
    }

    #[test]
    fn cleanup_failure_after_a_delete_poisoned_until_successful_retry() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                    super::super::SpillIoOperation::Delete,
                    2,
                    std::io::ErrorKind::PermissionDenied,
                )))
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 2, serialize_i64, deserialize_i64);
        let first = (0..100)
            .map(|value| key(&[value]))
            .find(|candidate| state.partition_for(candidate) == 0)
            .unwrap();
        let second = (100..200)
            .map(|value| key(&[value]))
            .find(|candidate| state.partition_for(candidate) == 1)
            .unwrap();
        state.insert(first, 10).unwrap();
        state.insert(second, 20).unwrap();
        state.spill_partition(0).unwrap();
        state.spill_partition(1).unwrap();

        assert_eq!(
            state.cleanup().unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(state.drain_state, DrainState::Poisoned);
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(
            state.insert(key(&[999]), 30).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );

        state.cleanup().unwrap();
        assert_eq!(state.drain_state, DrainState::Idle);
        assert_eq!(manager.active_file_count(), 0);
        state.insert(key(&[999]), 30).unwrap();
        assert_eq!(state.get(&key(&[999])).unwrap(), Some(&30));
    }

    #[test]
    fn partition_cleanup_retains_first_owned_error_without_formatting_or_dropping_later_errors() {
        let directory = TempDir::new().unwrap();
        let io = Arc::new(super::super::framing_tests::HostileCleanupIo::new());
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::clone(&io) as Arc<dyn super::super::SpillIo>)
                .build()
                .unwrap(),
        );
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 3, serialize_i64, deserialize_i64);
        for partition in 0..3 {
            state
                .insert(
                    key_for_partition(&state, partition),
                    i64::try_from(partition).expect("three-partition test index fits i64"),
                )
                .unwrap();
            state.spill_partition(partition).unwrap();
        }

        let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.cleanup()));

        assert!(
            cleanup.is_ok(),
            "explicit partition cleanup must not invoke hostile error Display or Drop"
        );
        let error = cleanup.unwrap().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            error
                .get_ref()
                .and_then(|source| {
                    source.downcast_ref::<super::super::framing_tests::HostileCleanupError>()
                })
                .map(super::super::framing_tests::HostileCleanupError::attempt),
            Some(0),
            "cleanup must return the first owned error, not a reconstruction"
        );
        assert_eq!(
            io.attempts(),
            3,
            "every partition receives one cleanup attempt"
        );
        assert_eq!(io.display_calls(), 0);
        assert_eq!(io.secondary_drops(), 0);
        assert_eq!(state.spilled_count(), 2, "only failed handles remain");
        assert_eq!(manager.active_file_count(), 2);
        assert_eq!(state.drain_state, DrainState::Poisoned);

        drop(error);
        assert_eq!(io.primary_drops(), 1);
        io.permit_delete();
        state.cleanup().unwrap();
        assert_eq!(io.attempts(), 5, "retry visits only retained failures");
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(state.drain_state, DrainState::Idle);
    }

    #[test]
    fn replacement_construction_admission_spills_one_cold_partition_then_builds() {
        const COLD_KEY_BYTES: usize = 64 * 1024;
        const CONSTRUCTION_PEAK: usize = 512 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(8 * 1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let mut state = accounted_bytes_state_with_partitions(
            Arc::clone(&spill_manager),
            &buffer_manager,
            control.token(),
            2,
        );
        let cold = wide_key_for_partition(&state, 0, COLD_KEY_BYTES, 'c');
        let target = key_for_partition(&state, 1);
        let serialized_cold = SerializedKey::from_values(&cold, state.frame_limits).unwrap();
        state
            .get_or_insert_with_accounted(cold, 0, Vec::new)
            .unwrap();
        state
            .get_or_insert_with_accounted(target.clone(), 16, || vec![0x11; 16])
            .unwrap();
        let spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(
                serialized_cold
                    .0
                    .len()
                    .checked_add(32)
                    .unwrap()
                    .checked_mul(4)
                    .unwrap(),
            )
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        assert!(spill_peak < CONSTRUCTION_PEAK - 1);
        let filler = RefCell::new(None);
        let built = AtomicBool::new(false);

        state
            .try_replace_accounted(
                target.clone(),
                |_| {
                    let leave_available = CONSTRUCTION_PEAK - 1;
                    let fill = buffer_manager
                        .available()
                        .checked_sub(leave_available)
                        .expect("the test budget must admit construction calibration");
                    filler.replace(Some(
                        buffer_manager
                            .try_allocate(fill, MemoryRegion::ExecutionBuffers)
                            .unwrap(),
                    ));
                    Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                        retained_upper_bound: 32,
                        construction_peak: CONSTRUCTION_PEAK,
                    })
                },
                |_| {
                    built.store(true, Ordering::Release);
                    assert_eq!(spill_manager.active_file_count(), 1);
                    Ok(vec![0x22; 32])
                },
            )
            .unwrap();

        assert!(built.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.spill_base_sizes[1], 0);
        assert_eq!(state.get(&target).unwrap(), Some(&vec![0x22; 32]));
        drop(filler.into_inner());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn construction_spill_exhausts_absent_full_map_replacement_retry_budget() {
        const CONSTRUCTION_PEAK: usize = 512 * 1024;

        let (_directory, spill_manager) = create_manager();
        let buffer_manager = buffer_manager_with_exact_budget(64 * 1024 * 1024);
        let control = crate::execution::QueryExecutionControl::new();
        let mut state = accounted_bytes_state_with_partitions(
            Arc::clone(&spill_manager),
            &buffer_manager,
            control.token(),
            3,
        );

        state.partitions[0]
            .as_mut()
            .unwrap()
            .try_reserve(1024)
            .unwrap();
        let first_cold = key_for_partition(&state, 0);
        let serialized_first_cold =
            SerializedKey::from_values(&first_cold, state.frame_limits).unwrap();
        state.partitions[0].as_mut().unwrap().insert(
            serialized_first_cold.clone(),
            PartitionEntry {
                num_key_columns: 1,
                resident_bound: 0,
                value: Vec::new(),
            },
        );
        state.partition_sizes[0] = 1;

        state.partitions[1]
            .as_mut()
            .unwrap()
            .try_reserve(16_384)
            .unwrap();
        let second_cold = key_for_partition(&state, 1);
        let serialized_second_cold =
            SerializedKey::from_values(&second_cold, state.frame_limits).unwrap();
        state.partitions[1].as_mut().unwrap().insert(
            serialized_second_cold.clone(),
            PartitionEntry {
                num_key_columns: 1,
                resident_bound: 0,
                value: Vec::new(),
            },
        );
        state.partition_sizes[1] = 1;

        state.partitions[2]
            .as_mut()
            .unwrap()
            .try_reserve(4096)
            .unwrap();
        let target_capacity = state.partitions[2].as_ref().unwrap().capacity();
        let mut nonce = 0_i64;
        while state.partitions[2].as_ref().unwrap().len() < target_capacity {
            let candidate = key(&[nonce]);
            nonce += 1;
            if state.partition_for(&candidate) != 2 {
                continue;
            }
            let serialized = SerializedKey::from_values(&candidate, state.frame_limits).unwrap();
            state.partitions[2].as_mut().unwrap().insert(
                serialized,
                PartitionEntry {
                    num_key_columns: 1,
                    resident_bound: 0,
                    value: Vec::new(),
                },
            );
        }
        let target_count = state.partitions[2].as_ref().unwrap().len();
        state.partition_sizes[2] = target_count;
        let search_end = nonce
            .checked_add(10_000)
            .expect("bounded absent-key search fits i64");
        let target = (nonce..search_end)
            .map(|candidate| key(&[candidate]))
            .find(|candidate| {
                state.partition_for(candidate) == 2
                    && !state.partitions[2].as_ref().unwrap().contains_key(
                        &SerializedKey::from_values(candidate, state.frame_limits).unwrap(),
                    )
            })
            .expect("the test must find an absent target key");
        let serialized_target = SerializedKey::from_values(&target, state.frame_limits).unwrap();

        state.access_times = vec![1, 2, 3];
        state.timestamp = 3;
        state.reconcile_grant().unwrap();
        let observed = state.observed_resident_capacity_bytes().unwrap();
        let map_growth = state
            .pending_entry_capacity_bytes(2, 0)
            .unwrap()
            .checked_sub(observed)
            .unwrap();
        let first_reclaim = PartitionedState::<Vec<u8>>::partition_map_allocation_bytes(
            state.partitions[0].as_ref().unwrap(),
        )
        .checked_add(serialized_first_cold.0.capacity())
        .unwrap();
        let second_reclaim = PartitionedState::<Vec<u8>>::partition_map_allocation_bytes(
            state.partitions[1].as_ref().unwrap(),
        )
        .checked_add(serialized_second_cold.0.capacity())
        .unwrap();
        let second_spill_peak = qualified_writer_buffer_requested_bytes()
            .checked_add(
                serialized_second_cold
                    .0
                    .len()
                    .checked_add(32)
                    .unwrap()
                    .checked_mul(4)
                    .unwrap(),
            )
            .unwrap()
            .max(qualified_writer_buffer_requested_bytes() * 2);
        assert!(second_spill_peak < first_reclaim - 1);
        assert!(first_reclaim < map_growth);
        assert!(
            first_reclaim.checked_add(second_reclaim).unwrap() > map_growth,
            "victim reclamation ({first_reclaim} + {second_reclaim}) must exceed map growth {map_growth}"
        );

        let filler = RefCell::new(None);
        let built = AtomicBool::new(false);
        let result = state.try_replace_accounted(
            target.clone(),
            |_| {
                let leave_available = CONSTRUCTION_PEAK - 1;
                let fill = buffer_manager
                    .available()
                    .checked_sub(leave_available)
                    .expect("the test budget must admit construction calibration");
                filler.replace(Some(
                    buffer_manager
                        .try_allocate(fill, MemoryRegion::ExecutionBuffers)
                        .unwrap(),
                ));
                Ok::<_, std::convert::Infallible>(PartitionUpdateAdmission {
                    retained_upper_bound: CONSTRUCTION_PEAK,
                    construction_peak: CONSTRUCTION_PEAK,
                })
            },
            |_| {
                built.store(true, Ordering::Release);
                assert_eq!(spill_manager.active_file_count(), 1);
                assert!(buffer_manager.available() < map_growth);
                assert!(second_spill_peak < buffer_manager.available());
                Ok(vec![0x77; CONSTRUCTION_PEAK])
            },
        );

        assert!(matches!(
            result,
            Err(PartitionUpdateError::Partition(PartitionOperationError::Io(
                ref error
            ))) if error.kind() == std::io::ErrorKind::OutOfMemory
        ));
        assert!(built.load(Ordering::Acquire));
        assert_eq!(state.spilled_count(), 1);
        assert_eq!(state.spill_base_sizes[0], 1);
        assert_eq!(state.spill_base_sizes[1], 0);
        assert!(
            state.partitions[1]
                .as_ref()
                .is_some_and(|partition| !partition.is_empty())
        );
        assert_eq!(state.partition_sizes[2], target_count);
        assert_eq!(state.partitions[2].as_ref().unwrap().len(), target_count);
        assert!(
            !state.partitions[2]
                .as_ref()
                .unwrap()
                .contains_key(&serialized_target)
        );
        drop(filler.into_inner());
        assert_eq!(buffer_manager.allocated(), state.granted_bytes());
    }

    #[test]
    fn test_multi_column_key() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 8, serialize_i64, deserialize_i64);

        // Insert with multi-column keys
        state
            .insert(vec![Value::String("a".into()), Value::Int64(1)], 100)
            .unwrap();
        state
            .insert(vec![Value::String("a".into()), Value::Int64(2)], 200)
            .unwrap();
        state
            .insert(vec![Value::String("b".into()), Value::Int64(1)], 300)
            .unwrap();

        assert_eq!(state.total_size(), 3);

        // Retrieve by multi-column key
        assert_eq!(
            state
                .get(&[Value::String("a".into()), Value::Int64(1)])
                .unwrap(),
            Some(&100)
        );
        assert_eq!(
            state
                .get(&[Value::String("a".into()), Value::Int64(2)])
                .unwrap(),
            Some(&200)
        );
        assert_eq!(
            state
                .get(&[Value::String("b".into()), Value::Int64(1)])
                .unwrap(),
            Some(&300)
        );
    }

    #[test]
    fn test_update_existing() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert
        state.insert(key(&[1]), 100).unwrap();
        assert_eq!(state.total_size(), 1);

        // Update
        let old = state.insert(key(&[1]), 200).unwrap();
        assert_eq!(old, Some(100));
        assert_eq!(state.total_size(), 1); // Size shouldn't increase

        // Verify update
        assert_eq!(state.get(&key(&[1])).unwrap(), Some(&200));
    }
}
