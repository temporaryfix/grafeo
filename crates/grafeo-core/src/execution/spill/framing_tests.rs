use super::{
    MAX_FIXED_CONTROL_PAYLOAD_BYTES, OpenSpillRecord, SPILL_RECORD_HEADER_BYTES, SpillDiskQuota,
    SpillFileIdentity, SpillFileRole, SpillFrameLimits, SpillIo, SpillIoOperation, SpillRecordKind,
    SpillRecordMeta, SpillRecordProvider,
};
use grafeo_common::memory::buffer::{
    AccountedError, AccountedErrorPublisher, BufferManager, BufferManagerConfig,
};
use parking_lot::Mutex;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tempfile::TempDir;

fn exact_buffer_manager(budget: usize) -> Arc<BufferManager> {
    let mut config = BufferManagerConfig::with_budget(budget);
    config.soft_limit_fraction = 1.0;
    config.evict_limit_fraction = 1.0;
    config.hard_limit_fraction = 1.0;
    BufferManager::new(config)
}

fn reader_operation_error_publication_bytes() -> usize {
    AccountedErrorPublisher::<super::file::ProviderAccountedReaderOperationError>::required_bytes()
}

fn qualified_copy_into(stored: &[u8], plaintext: &mut [u8]) -> Option<io::Result<()>> {
    if stored.len() != plaintext.len() {
        return Some(Err(io::Error::from(io::ErrorKind::InvalidData)));
    }
    plaintext.copy_from_slice(stored);
    Some(Ok(()))
}

struct CountingQualifiedCleartextProvider {
    begin_calls: Arc<AtomicUsize>,
}

impl SpillRecordProvider for CountingQualifiedCleartextProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        self.begin_calls.fetch_add(1, Ordering::AcqRel);
        super::CleartextSpillRecordProvider.begin_file(identity)
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        Some(0)
    }

    fn supports_qualified_exact_open(&self) -> bool {
        true
    }
}

#[derive(Clone)]
pub(super) struct GrantLifetimeWitness {
    drops: Arc<AtomicUsize>,
    released_before_drop: Arc<AtomicBool>,
}

impl GrantLifetimeWitness {
    pub(super) fn drops(&self) -> usize {
        self.drops.load(Ordering::Acquire)
    }

    pub(super) fn released_before_drop(&self) -> bool {
        self.released_before_drop.load(Ordering::Acquire)
    }
}

pub(super) fn grant_lifetime_provider(
    resources: crate::execution::QueryResourceContext,
    file_workspace_bound: usize,
) -> (Arc<dyn SpillRecordProvider>, GrantLifetimeWitness) {
    let witness = GrantLifetimeWitness {
        drops: Arc::new(AtomicUsize::new(0)),
        released_before_drop: Arc::new(AtomicBool::new(false)),
    };
    (
        Arc::new(GrantLifetimeProvider {
            resources,
            file_workspace_bound,
            witness: witness.clone(),
        }),
        witness,
    )
}

struct GrantLifetimeProvider {
    resources: crate::execution::QueryResourceContext,
    file_workspace_bound: usize,
    witness: GrantLifetimeWitness,
}

impl SpillRecordProvider for GrantLifetimeProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, _identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        Ok(Box::new(GrantLifetimeOpenRecord {
            resources: self.resources.clone(),
            admitted_bytes: self.resources.query_stats().allocated_bytes,
            witness: self.witness.clone(),
        }))
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        Some(self.file_workspace_bound)
    }

    fn supports_qualified_exact_open(&self) -> bool {
        true
    }
}

struct GrantLifetimeOpenRecord {
    resources: crate::execution::QueryResourceContext,
    admitted_bytes: usize,
    witness: GrantLifetimeWitness,
}

impl OpenSpillRecord for GrantLifetimeOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        Ok(plaintext_len)
    }

    fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
        Some(plaintext_len)
    }

    fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
        Some(stored_len)
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(plaintext.to_vec())
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(stored.to_vec())
    }

    fn open_qualified_into(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
        plaintext: &mut [u8],
    ) -> Option<io::Result<()>> {
        qualified_copy_into(stored, plaintext)
    }
}

impl Drop for GrantLifetimeOpenRecord {
    fn drop(&mut self) {
        if self.resources.query_stats().allocated_bytes < self.admitted_bytes {
            self.witness
                .released_before_drop
                .store(true, Ordering::Release);
        }
        self.witness.drops.fetch_add(1, Ordering::AcqRel);
    }
}

const READER_CONSTRUCTION_WORKSPACE: usize = 64 * 1024;
const SHARED_FAILURE_WORKSPACE: usize = 16 * 1024;

struct SharedHeapRollback {
    retained: Arc<Mutex<Vec<u8>>>,
}

impl SharedHeapRollback {
    fn retain_for_failure(
        retained: Arc<Mutex<Vec<u8>>>,
        resources: &crate::execution::QueryResourceContext,
        authority_during_growth: &AtomicUsize,
    ) -> Self {
        let guard = Self { retained };
        *guard.retained.lock() = vec![0x47; SHARED_FAILURE_WORKSPACE];
        authority_during_growth.store(resources.query_stats().allocated_bytes, Ordering::Release);
        guard
    }
}

impl Drop for SharedHeapRollback {
    fn drop(&mut self) {
        drop(std::mem::take(&mut *self.retained.lock()));
    }
}

#[derive(Clone, Copy)]
enum ReaderConstructionFailure {
    BeginError,
    BeginPanic,
    ControlError,
    ControlPanic,
    StoredLenErrorWithHostileDrop,
    StoredLenPanicWithHostileDrop,
    OpenBoundPanicWithHostileDrop,
    ControlErrorWithHostileDrop,
    ControlPanicWithHostileDrop,
    SuccessWithHostileDrop,
}

struct ReaderConstructionFailureProvider {
    resources: crate::execution::QueryResourceContext,
    begin_calls: AtomicUsize,
    failure: ReaderConstructionFailure,
    payload_drop_observation: Arc<AtomicUsize>,
    hostile_record_drops: Arc<AtomicUsize>,
    hostile_authority_when_dropped: Arc<AtomicUsize>,
    hostile_panic_payload_dropped: Arc<AtomicBool>,
    shared_failure_heap: Arc<Mutex<Vec<u8>>>,
    authority_during_shared_growth: AtomicUsize,
}

impl SpillRecordProvider for ReaderConstructionFailureProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, _identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        if self.begin_calls.fetch_add(1, Ordering::AcqRel) == 0 {
            return Ok(Box::new(PassthroughOpenRecord));
        }

        match self.failure {
            ReaderConstructionFailure::BeginError => {
                let _rollback = SharedHeapRollback::retain_for_failure(
                    Arc::clone(&self.shared_failure_heap),
                    &self.resources,
                    &self.authority_during_shared_growth,
                );
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    HeapReaderConstructionPayload::new(
                        self.resources.clone(),
                        Arc::clone(&self.payload_drop_observation),
                    ),
                ))
            }
            ReaderConstructionFailure::BeginPanic => {
                let _rollback = SharedHeapRollback::retain_for_failure(
                    Arc::clone(&self.shared_failure_heap),
                    &self.resources,
                    &self.authority_during_shared_growth,
                );
                std::panic::panic_any(HeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.payload_drop_observation),
                ));
            }
            ReaderConstructionFailure::ControlError | ReaderConstructionFailure::ControlPanic => {
                Ok(Box::new(ReaderControlFailureOpenRecord {
                    resources: self.resources.clone(),
                    failure: self.failure,
                    payload_drop_observation: Arc::clone(&self.payload_drop_observation),
                }))
            }
            ReaderConstructionFailure::StoredLenErrorWithHostileDrop
            | ReaderConstructionFailure::StoredLenPanicWithHostileDrop
            | ReaderConstructionFailure::OpenBoundPanicWithHostileDrop
            | ReaderConstructionFailure::ControlErrorWithHostileDrop
            | ReaderConstructionFailure::ControlPanicWithHostileDrop
            | ReaderConstructionFailure::SuccessWithHostileDrop => {
                Ok(Box::new(HostileReaderConstructionOpenRecord {
                    resources: self.resources.clone(),
                    failure: self.failure,
                    drops: Arc::clone(&self.hostile_record_drops),
                    authority_when_dropped: Arc::clone(&self.hostile_authority_when_dropped),
                    panic_payload_dropped: Arc::clone(&self.hostile_panic_payload_dropped),
                }))
            }
        }
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        Some(READER_CONSTRUCTION_WORKSPACE)
    }

    fn supports_qualified_exact_open(&self) -> bool {
        true
    }
}

struct ReaderControlFailureOpenRecord {
    resources: crate::execution::QueryResourceContext,
    failure: ReaderConstructionFailure,
    payload_drop_observation: Arc<AtomicUsize>,
}

impl OpenSpillRecord for ReaderControlFailureOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        Ok(plaintext_len)
    }

    fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
        Some(stored_len)
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(plaintext.to_vec())
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        _stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        let payload = HeapReaderConstructionPayload::new(
            self.resources.clone(),
            Arc::clone(&self.payload_drop_observation),
        );
        match self.failure {
            ReaderConstructionFailure::ControlError => {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, payload))
            }
            ReaderConstructionFailure::ControlPanic => std::panic::panic_any(payload),
            ReaderConstructionFailure::BeginError
            | ReaderConstructionFailure::BeginPanic
            | ReaderConstructionFailure::StoredLenErrorWithHostileDrop
            | ReaderConstructionFailure::StoredLenPanicWithHostileDrop
            | ReaderConstructionFailure::OpenBoundPanicWithHostileDrop
            | ReaderConstructionFailure::ControlErrorWithHostileDrop
            | ReaderConstructionFailure::ControlPanicWithHostileDrop
            | ReaderConstructionFailure::SuccessWithHostileDrop => {
                unreachable!("control failure record has a control failure mode")
            }
        }
    }

    fn open_qualified_into(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        _stored: &[u8],
        _plaintext: &mut [u8],
    ) -> Option<io::Result<()>> {
        let payload = HeapReaderConstructionPayload::new(
            self.resources.clone(),
            Arc::clone(&self.payload_drop_observation),
        );
        Some(match self.failure {
            ReaderConstructionFailure::ControlError => {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, payload))
            }
            ReaderConstructionFailure::ControlPanic => std::panic::panic_any(payload),
            ReaderConstructionFailure::BeginError
            | ReaderConstructionFailure::BeginPanic
            | ReaderConstructionFailure::StoredLenErrorWithHostileDrop
            | ReaderConstructionFailure::StoredLenPanicWithHostileDrop
            | ReaderConstructionFailure::OpenBoundPanicWithHostileDrop
            | ReaderConstructionFailure::ControlErrorWithHostileDrop
            | ReaderConstructionFailure::ControlPanicWithHostileDrop
            | ReaderConstructionFailure::SuccessWithHostileDrop => {
                unreachable!("control failure record has a control failure mode")
            }
        })
    }
}

struct PassthroughOpenRecord;

struct CompatibilityOnlyProvider {
    begin_calls: Arc<AtomicUsize>,
}

impl SpillRecordProvider for CompatibilityOnlyProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, _identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        self.begin_calls.fetch_add(1, Ordering::AcqRel);
        Ok(Box::new(PassthroughOpenRecord))
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        Some(64)
    }
}

impl OpenSpillRecord for PassthroughOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        Ok(plaintext_len)
    }

    fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
        Some(plaintext_len)
    }

    fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
        Some(stored_len)
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(plaintext.to_vec())
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(stored.to_vec())
    }

    fn open_qualified_into(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
        plaintext: &mut [u8],
    ) -> Option<io::Result<()>> {
        qualified_copy_into(stored, plaintext)
    }
}

struct HeapReaderConstructionPayload {
    bytes: Vec<u8>,
    resources: crate::execution::QueryResourceContext,
    allocated_when_dropped: Arc<AtomicUsize>,
}

impl HeapReaderConstructionPayload {
    fn new(
        resources: crate::execution::QueryResourceContext,
        allocated_when_dropped: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            bytes: vec![0xa5; 32 * 1024],
            resources,
            allocated_when_dropped,
        }
    }
}

impl std::fmt::Debug for HeapReaderConstructionPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HeapReaderConstructionPayload")
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for HeapReaderConstructionPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("heap-bearing reader-construction failure")
    }
}

impl std::error::Error for HeapReaderConstructionPayload {}

impl Drop for HeapReaderConstructionPayload {
    fn drop(&mut self) {
        self.allocated_when_dropped.store(
            self.resources.query_stats().allocated_bytes,
            Ordering::Release,
        );
    }
}

struct HostileReaderConstructionOpenRecord {
    resources: crate::execution::QueryResourceContext,
    failure: ReaderConstructionFailure,
    drops: Arc<AtomicUsize>,
    authority_when_dropped: Arc<AtomicUsize>,
    panic_payload_dropped: Arc<AtomicBool>,
}

impl OpenSpillRecord for HostileReaderConstructionOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        match self.failure {
            ReaderConstructionFailure::StoredLenErrorWithHostileDrop => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "hostile-record stored-length error",
            )),
            ReaderConstructionFailure::StoredLenPanicWithHostileDrop => {
                std::panic::panic_any(0x5702_ed1e_u64)
            }
            ReaderConstructionFailure::BeginError
            | ReaderConstructionFailure::BeginPanic
            | ReaderConstructionFailure::ControlError
            | ReaderConstructionFailure::ControlPanic
            | ReaderConstructionFailure::OpenBoundPanicWithHostileDrop
            | ReaderConstructionFailure::ControlErrorWithHostileDrop
            | ReaderConstructionFailure::ControlPanicWithHostileDrop
            | ReaderConstructionFailure::SuccessWithHostileDrop => Ok(plaintext_len),
        }
    }

    fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
        if matches!(
            self.failure,
            ReaderConstructionFailure::OpenBoundPanicWithHostileDrop
        ) {
            std::panic::panic_any(0x0b0d_5ca1_u64);
        }
        Some(stored_len)
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(plaintext.to_vec())
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        match self.failure {
            ReaderConstructionFailure::ControlErrorWithHostileDrop => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "hostile-record control error",
            )),
            ReaderConstructionFailure::ControlPanicWithHostileDrop => {
                std::panic::panic_any(0xc011_57a7_u64)
            }
            ReaderConstructionFailure::SuccessWithHostileDrop => Ok(stored.to_vec()),
            ReaderConstructionFailure::BeginError
            | ReaderConstructionFailure::BeginPanic
            | ReaderConstructionFailure::ControlError
            | ReaderConstructionFailure::ControlPanic
            | ReaderConstructionFailure::StoredLenErrorWithHostileDrop
            | ReaderConstructionFailure::StoredLenPanicWithHostileDrop
            | ReaderConstructionFailure::OpenBoundPanicWithHostileDrop => {
                unreachable!("hostile record has a hostile-drop mode")
            }
        }
    }

    fn open_qualified_into(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
        plaintext: &mut [u8],
    ) -> Option<io::Result<()>> {
        Some(match self.failure {
            ReaderConstructionFailure::ControlErrorWithHostileDrop => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "hostile-record control error",
            )),
            ReaderConstructionFailure::ControlPanicWithHostileDrop => {
                std::panic::panic_any(0xc011_57a7_u64)
            }
            ReaderConstructionFailure::SuccessWithHostileDrop => {
                if stored.len() != plaintext.len() {
                    Err(io::Error::from(io::ErrorKind::InvalidData))
                } else {
                    plaintext.copy_from_slice(stored);
                    Ok(())
                }
            }
            ReaderConstructionFailure::BeginError
            | ReaderConstructionFailure::BeginPanic
            | ReaderConstructionFailure::ControlError
            | ReaderConstructionFailure::ControlPanic
            | ReaderConstructionFailure::StoredLenErrorWithHostileDrop
            | ReaderConstructionFailure::StoredLenPanicWithHostileDrop
            | ReaderConstructionFailure::OpenBoundPanicWithHostileDrop => {
                unreachable!("hostile record has a hostile-drop mode")
            }
        })
    }
}

impl Drop for HostileReaderConstructionOpenRecord {
    fn drop(&mut self) {
        self.authority_when_dropped.store(
            self.resources.query_stats().allocated_bytes,
            Ordering::Release,
        );
        self.drops.fetch_add(1, Ordering::AcqRel);
        std::panic::panic_any(HeapHostileDestructorPanic {
            bytes: vec![0x5a; 32 * 1024],
            dropped: Arc::clone(&self.panic_payload_dropped),
        });
    }
}

struct HeapHostileDestructorPanic {
    bytes: Vec<u8>,
    dropped: Arc<AtomicBool>,
}

impl std::fmt::Debug for HeapHostileDestructorPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HeapHostileDestructorPanic")
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl Drop for HeapHostileDestructorPanic {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
    }
}

#[derive(Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "orthogonal hostile-provider fixture switches are clearer as named flags"
)]
struct RecordingSealedProvider {
    aad: Arc<Mutex<Vec<Vec<u8>>>>,
    identities: Arc<Mutex<Vec<SpillFileIdentity>>>,
    calls: Arc<Mutex<Vec<RecordedProviderCall>>>,
    fail_seal: bool,
    fail_open: bool,
    wrong_stored_len: bool,
    wrong_open_len: bool,
}

#[derive(Clone, Copy, Debug)]
struct RecordedProviderCall {
    opening: bool,
    meta: SpillRecordMeta,
    aad: [u8; 32],
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the open-record fixture mirrors its provider's orthogonal failure switches"
)]
struct RecordingOpenRecord {
    aad: Arc<Mutex<Vec<Vec<u8>>>>,
    calls: Arc<Mutex<Vec<RecordedProviderCall>>>,
    fail_seal: bool,
    fail_open: bool,
    wrong_stored_len: bool,
    wrong_open_len: bool,
}

struct FailSortRowProvider;

struct FailSortRowOpenRecord;

struct FailBeginFileProvider;

struct PanicOnceBeginFileProvider {
    fired: Arc<AtomicBool>,
}

pub(super) fn fail_begin_file_provider() -> Arc<dyn SpillRecordProvider> {
    Arc::new(FailBeginFileProvider)
}

pub(super) fn panic_once_begin_file_provider() -> Arc<dyn SpillRecordProvider> {
    Arc::new(PanicOnceBeginFileProvider {
        fired: Arc::new(AtomicBool::new(false)),
    })
}

struct UnsealedLengthChangingProvider {
    seal_called: Arc<AtomicBool>,
}

struct UnsealedLengthChangingOpenRecord {
    seal_called: Arc<AtomicBool>,
}

struct PanicOnceProvider {
    panic_on_open: bool,
    fired: Arc<AtomicBool>,
}

struct PanicOnceOpenRecord {
    panic_on_open: bool,
    fired: Arc<AtomicBool>,
}

pub(super) fn panic_once_seal_provider() -> Arc<dyn SpillRecordProvider> {
    Arc::new(PanicOnceProvider {
        panic_on_open: false,
        fired: Arc::new(AtomicBool::new(false)),
    })
}

impl SpillRecordProvider for PanicOnceProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, _identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        Ok(Box::new(PanicOnceOpenRecord {
            panic_on_open: self.panic_on_open,
            fired: Arc::clone(&self.fired),
        }))
    }
}

impl OpenSpillRecord for PanicOnceOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        Ok(plaintext_len)
    }

    fn seal(
        &mut self,
        meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        if !self.panic_on_open
            && meta.kind() == SpillRecordKind::SortRow
            && !self.fired.swap(true, Ordering::AcqRel)
        {
            panic!("seal callback panic")
        }
        Ok(plaintext.to_vec())
    }

    fn open(
        &mut self,
        meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        if self.panic_on_open
            && meta.kind() == SpillRecordKind::SortRow
            && !self.fired.swap(true, Ordering::AcqRel)
        {
            panic!("open callback panic")
        }
        Ok(stored.to_vec())
    }
}

impl SpillRecordProvider for FailBeginFileProvider {
    fn seals(&self) -> bool {
        true
    }

    fn begin_file(
        &self,
        _identity: super::SpillFileIdentity,
    ) -> io::Result<Box<dyn OpenSpillRecord>> {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "begin-file failpoint",
        ))
    }
}

impl SpillRecordProvider for PanicOnceBeginFileProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, _identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        assert!(
            self.fired.swap(true, Ordering::AcqRel),
            "begin-file callback panic"
        );
        Ok(Box::new(PanicOnceOpenRecord {
            panic_on_open: false,
            fired: Arc::clone(&self.fired),
        }))
    }
}

impl SpillRecordProvider for UnsealedLengthChangingProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, _identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        Ok(Box::new(UnsealedLengthChangingOpenRecord {
            seal_called: Arc::clone(&self.seal_called),
        }))
    }
}

impl OpenSpillRecord for UnsealedLengthChangingOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        plaintext_len
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "stored length overflow"))
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        self.seal_called.store(true, Ordering::Release);
        let mut stored = plaintext.to_vec();
        stored.push(0);
        Ok(stored)
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(stored.to_vec())
    }
}

pub(super) struct FailNthIo {
    operation: SpillIoOperation,
    fail_on: usize,
    observed: AtomicUsize,
    kind: io::ErrorKind,
}

/// A mixed-success delete hook for explicit cleanup aggregation tests.
///
/// The first and third delete attempts fail. Formatting either error panics;
/// dropping the later error also panics. The first error has a safe destructor
/// so a broken formatter path can be caught without aborting the test process.
pub(super) struct HostileCleanupIo {
    failing: AtomicBool,
    attempts: AtomicUsize,
    display_calls: Arc<AtomicUsize>,
    primary_drops: Arc<AtomicUsize>,
    secondary_drops: Arc<AtomicUsize>,
}

impl HostileCleanupIo {
    pub(super) fn new() -> Self {
        Self {
            failing: AtomicBool::new(true),
            attempts: AtomicUsize::new(0),
            display_calls: Arc::new(AtomicUsize::new(0)),
            primary_drops: Arc::new(AtomicUsize::new(0)),
            secondary_drops: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(super) fn permit_delete(&self) {
        self.failing.store(false, Ordering::Release);
    }

    pub(super) fn attempts(&self) -> usize {
        self.attempts.load(Ordering::Acquire)
    }

    pub(super) fn display_calls(&self) -> usize {
        self.display_calls.load(Ordering::Acquire)
    }

    pub(super) fn primary_drops(&self) -> usize {
        self.primary_drops.load(Ordering::Acquire)
    }

    pub(super) fn secondary_drops(&self) -> usize {
        self.secondary_drops.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub(super) struct HostileCleanupError {
    attempt: usize,
    display_calls: Arc<AtomicUsize>,
    primary_drops: Arc<AtomicUsize>,
    secondary_drops: Arc<AtomicUsize>,
}

impl HostileCleanupError {
    pub(super) fn attempt(&self) -> usize {
        self.attempt
    }
}

impl std::fmt::Display for HostileCleanupError {
    fn fmt(&self, _formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_calls.fetch_add(1, Ordering::AcqRel);
        panic!("explicit spill cleanup formatted a hostile owned error")
    }
}

impl std::error::Error for HostileCleanupError {}

impl Drop for HostileCleanupError {
    fn drop(&mut self) {
        if self.attempt == 0 {
            self.primary_drops.fetch_add(1, Ordering::AcqRel);
        } else {
            self.secondary_drops.fetch_add(1, Ordering::AcqRel);
            panic!("secondary explicit-cleanup error destructor ran")
        }
    }
}

impl SpillIo for HostileCleanupIo {
    fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
        if operation != SpillIoOperation::Delete {
            return Ok(());
        }
        let attempt = self.attempts.fetch_add(1, Ordering::AcqRel);
        if !self.failing.load(Ordering::Acquire) || attempt == 1 {
            return Ok(());
        }
        let kind = if attempt == 0 {
            io::ErrorKind::PermissionDenied
        } else {
            io::ErrorKind::WouldBlock
        };
        Err(io::Error::new(
            kind,
            HostileCleanupError {
                attempt,
                display_calls: Arc::clone(&self.display_calls),
                primary_drops: Arc::clone(&self.primary_drops),
                secondary_drops: Arc::clone(&self.secondary_drops),
            },
        ))
    }
}

impl FailNthIo {
    pub(super) fn new(operation: SpillIoOperation, fail_on: usize, kind: io::ErrorKind) -> Self {
        Self {
            operation,
            fail_on,
            observed: AtomicUsize::new(0),
            kind,
        }
    }
}

impl SpillIo for FailNthIo {
    fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
        if operation != self.operation {
            return Ok(());
        }
        let observed = self.observed.fetch_add(1, Ordering::Relaxed) + 1;
        if observed == self.fail_on {
            return Err(io::Error::new(self.kind, "deterministic I/O failpoint"));
        }
        Ok(())
    }
}

struct PanicNthIo {
    operation: SpillIoOperation,
    panic_on: usize,
    observed: AtomicUsize,
}

impl PanicNthIo {
    fn new(operation: SpillIoOperation, panic_on: usize) -> Self {
        Self {
            operation,
            panic_on,
            observed: AtomicUsize::new(0),
        }
    }
}

impl SpillIo for PanicNthIo {
    fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
        if operation == self.operation {
            let observed = self.observed.fetch_add(1, Ordering::Relaxed) + 1;
            assert_ne!(observed, self.panic_on, "deterministic I/O hook panic");
        }
        Ok(())
    }
}

#[derive(Default)]
struct UnqualifiedFixedControlReadIo {
    read_callbacks: AtomicUsize,
}

impl SpillIo for UnqualifiedFixedControlReadIo {
    fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
        if matches!(
            operation,
            SpillIoOperation::ReadHeader | SpillIoOperation::ReadPayload
        ) {
            self.read_callbacks.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }
}

const READER_CONTROL_HOOK_WORKSPACE: usize = 48 * 1024;

#[derive(Clone, Copy)]
enum ReaderControlHookFailure {
    Error,
    Panic,
    HostileErrorDrop,
    HostilePanicDrop,
}

struct QualifiedFailingFixedControlReadIo {
    resources: crate::execution::QueryResourceContext,
    target: SpillIoOperation,
    failure: ReaderControlHookFailure,
    armed: AtomicBool,
    fired: AtomicBool,
    payload_drop_observation: Arc<AtomicUsize>,
    hostile_authority_when_dropped: Arc<AtomicUsize>,
    hostile_panic_payload_dropped: Arc<AtomicBool>,
    shared_failure_heap: Arc<Mutex<Vec<u8>>>,
    authority_during_shared_growth: AtomicUsize,
}

impl SpillIo for QualifiedFailingFixedControlReadIo {
    fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
        if operation != self.target
            || !self.armed.load(Ordering::Acquire)
            || self.fired.swap(true, Ordering::AcqRel)
        {
            return Ok(());
        }
        let _rollback = SharedHeapRollback::retain_for_failure(
            Arc::clone(&self.shared_failure_heap),
            &self.resources,
            &self.authority_during_shared_growth,
        );
        match self.failure {
            ReaderControlHookFailure::Error => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                HeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.payload_drop_observation),
                ),
            )),
            ReaderControlHookFailure::Panic => {
                std::panic::panic_any(HeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.payload_drop_observation),
                ));
            }
            ReaderControlHookFailure::HostileErrorDrop => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                HostileHeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.hostile_authority_when_dropped),
                    Arc::clone(&self.hostile_panic_payload_dropped),
                ),
            )),
            ReaderControlHookFailure::HostilePanicDrop => {
                std::panic::panic_any(HostileHeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.hostile_authority_when_dropped),
                    Arc::clone(&self.hostile_panic_payload_dropped),
                ));
            }
        }
    }

    fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        Some(READER_CONTROL_HOOK_WORKSPACE)
    }
}

struct HostileHeapReaderConstructionPayload {
    bytes: Vec<u8>,
    resources: crate::execution::QueryResourceContext,
    authority_when_dropped: Arc<AtomicUsize>,
    secondary_panic_payload_dropped: Arc<AtomicBool>,
}

impl HostileHeapReaderConstructionPayload {
    fn new(
        resources: crate::execution::QueryResourceContext,
        authority_when_dropped: Arc<AtomicUsize>,
        secondary_panic_payload_dropped: Arc<AtomicBool>,
    ) -> Self {
        Self {
            bytes: vec![0xc3; 16 * 1024],
            resources,
            authority_when_dropped,
            secondary_panic_payload_dropped,
        }
    }
}

impl std::fmt::Debug for HostileHeapReaderConstructionPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostileHeapReaderConstructionPayload")
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for HostileHeapReaderConstructionPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("hostile heap-bearing reader-construction payload")
    }
}

impl std::error::Error for HostileHeapReaderConstructionPayload {}

impl Drop for HostileHeapReaderConstructionPayload {
    fn drop(&mut self) {
        self.authority_when_dropped.store(
            self.resources.query_stats().allocated_bytes,
            Ordering::Release,
        );
        std::panic::panic_any(HeapHostileDestructorPanic {
            bytes: vec![0x6d; 16 * 1024],
            dropped: Arc::clone(&self.secondary_panic_payload_dropped),
        });
    }
}

const QUALIFIED_OPERATION_PROVIDER_WORKSPACE: usize = 64 * 1024;

#[derive(Clone, Copy)]
enum QualifiedOperationFailure {
    DataError,
    DataPanic,
    DataHostileErrorDrop,
    FinishError,
}

struct QualifiedOperationFailureProvider {
    resources: crate::execution::QueryResourceContext,
    failure: QualifiedOperationFailure,
    payload_drop_observation: Arc<AtomicUsize>,
    hostile_authority_when_dropped: Arc<AtomicUsize>,
    hostile_panic_payload_dropped: Arc<AtomicBool>,
}

impl SpillRecordProvider for QualifiedOperationFailureProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(&self, _identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        Ok(Box::new(QualifiedOperationFailureOpenRecord {
            resources: self.resources.clone(),
            failure: self.failure,
            payload_drop_observation: Arc::clone(&self.payload_drop_observation),
            hostile_authority_when_dropped: Arc::clone(&self.hostile_authority_when_dropped),
            hostile_panic_payload_dropped: Arc::clone(&self.hostile_panic_payload_dropped),
        }))
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        Some(QUALIFIED_OPERATION_PROVIDER_WORKSPACE)
    }

    fn supports_qualified_exact_open(&self) -> bool {
        true
    }
}

struct QualifiedOperationFailureOpenRecord {
    resources: crate::execution::QueryResourceContext,
    failure: QualifiedOperationFailure,
    payload_drop_observation: Arc<AtomicUsize>,
    hostile_authority_when_dropped: Arc<AtomicUsize>,
    hostile_panic_payload_dropped: Arc<AtomicBool>,
}

impl OpenSpillRecord for QualifiedOperationFailureOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        Ok(plaintext_len)
    }

    fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
        Some(plaintext_len)
    }

    fn open_allocation_bound(&self, _stored_len: usize) -> Option<usize> {
        Some(QUALIFIED_OPERATION_PROVIDER_WORKSPACE)
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(plaintext.to_vec())
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        _stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        panic!("qualified operation owner reached compatibility OpenSpillRecord::open")
    }

    fn open_qualified_into(
        &mut self,
        meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
        plaintext: &mut [u8],
    ) -> Option<io::Result<()>> {
        let fails_data = meta.kind() == SpillRecordKind::SortRow;
        let fails_finish = meta.kind() == SpillRecordKind::FileEnd;
        Some(match self.failure {
            QualifiedOperationFailure::DataError if fails_data => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                HeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.payload_drop_observation),
                ),
            )),
            QualifiedOperationFailure::DataPanic if fails_data => {
                std::panic::panic_any(HeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.payload_drop_observation),
                ))
            }
            QualifiedOperationFailure::DataHostileErrorDrop if fails_data => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                HostileHeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.hostile_authority_when_dropped),
                    Arc::clone(&self.hostile_panic_payload_dropped),
                ),
            )),
            QualifiedOperationFailure::FinishError if fails_finish => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                HeapReaderConstructionPayload::new(
                    self.resources.clone(),
                    Arc::clone(&self.payload_drop_observation),
                ),
            )),
            _ => {
                if stored.len() != plaintext.len() {
                    Err(io::Error::from(io::ErrorKind::InvalidData))
                } else {
                    plaintext.copy_from_slice(stored);
                    Ok(())
                }
            }
        })
    }
}

#[derive(Default)]
struct DriftingStoredLengthProvider {
    files_begun: Arc<AtomicUsize>,
}

struct DriftingStoredLengthOpenRecord {
    extra: usize,
}

impl SpillRecordProvider for DriftingStoredLengthProvider {
    fn seals(&self) -> bool {
        true
    }

    fn begin_file(
        &self,
        _identity: super::SpillFileIdentity,
    ) -> io::Result<Box<dyn OpenSpillRecord>> {
        let begin = self.files_begun.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(DriftingStoredLengthOpenRecord {
            extra: if begin == 0 { 1 } else { 2 },
        }))
    }
}

impl OpenSpillRecord for DriftingStoredLengthOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        plaintext_len
            .checked_add(self.extra)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "stored length overflow"))
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        let mut stored = vec![0xa5];
        stored.extend_from_slice(plaintext);
        Ok(stored)
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(stored[1..].to_vec())
    }
}

impl SpillRecordProvider for FailSortRowProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(
        &self,
        _identity: super::SpillFileIdentity,
    ) -> io::Result<Box<dyn OpenSpillRecord>> {
        Ok(Box::new(FailSortRowOpenRecord))
    }
}

impl OpenSpillRecord for FailSortRowOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        Ok(plaintext_len)
    }

    fn seal(
        &mut self,
        meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        if meta.kind() == SpillRecordKind::SortRow {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "row failpoint"));
        }
        Ok(plaintext.to_vec())
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        Ok(stored.to_vec())
    }
}

impl SpillRecordProvider for RecordingSealedProvider {
    fn seals(&self) -> bool {
        true
    }

    fn begin_file(
        &self,
        identity: super::SpillFileIdentity,
    ) -> io::Result<Box<dyn OpenSpillRecord>> {
        self.identities.lock().push(identity);
        Ok(Box::new(RecordingOpenRecord {
            aad: Arc::clone(&self.aad),
            calls: Arc::clone(&self.calls),
            fail_seal: self.fail_seal,
            fail_open: self.fail_open,
            wrong_stored_len: self.wrong_stored_len,
            wrong_open_len: self.wrong_open_len,
        }))
    }
}

impl OpenSpillRecord for RecordingOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        plaintext_len
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "stored length overflow"))
    }

    fn seal(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        if self.fail_seal {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "seal failpoint",
            ));
        }
        self.aad.lock().push(aad.to_vec());
        self.calls.lock().push(RecordedProviderCall {
            opening: false,
            meta: *meta,
            aad: *aad,
        });
        let mut stored = Vec::with_capacity(plaintext.len() + 1);
        stored.push(0xa5);
        stored.extend(plaintext.iter().map(|byte| byte ^ 0x5a));
        if self.wrong_stored_len {
            stored.pop();
        }
        Ok(stored)
    }

    fn open(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        if self.fail_open {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "open failpoint",
            ));
        }
        self.aad.lock().push(aad.to_vec());
        self.calls.lock().push(RecordedProviderCall {
            opening: true,
            meta: *meta,
            aad: *aad,
        });
        let Some((&0xa5, body)) = stored.split_first() else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "sealed prefix"));
        };
        let mut plaintext: Vec<u8> = body.iter().map(|byte| byte ^ 0x5a).collect();
        if self.wrong_open_len {
            plaintext.pop();
        }
        Ok(plaintext)
    }
}

fn read_valid_sort(file: &super::SpillFile) -> io::Result<Vec<Vec<u8>>> {
    let mut reader = file.reader()?;
    let (columns, rows) = reader.read_sort_run_start()?;
    if columns != 1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "columns"));
    }
    let mut result = Vec::new();
    for _ in 0..rows {
        result.push(reader.read_sort_row()?);
    }
    reader.finish()?;
    Ok(result)
}

fn rewrite_record_crc(bytes: &mut [u8], record_offset: usize) {
    let stored_len = usize::try_from(u64::from_le_bytes(
        bytes[record_offset + 24..record_offset + 32]
            .try_into()
            .unwrap(),
    ))
    .expect("test record length fits usize");
    let payload_start = record_offset + 36;
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&bytes[record_offset..record_offset + 32]);
    checksum.update(&bytes[payload_start..payload_start + stored_len]);
    bytes[record_offset + 32..record_offset + 36]
        .copy_from_slice(&checksum.finalize().to_le_bytes());
}

#[test]
fn cleartext_v1_file_start_is_byte_for_byte_fixed() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let bytes = std::fs::read(file.path()).unwrap();

    assert_eq!(
        &bytes[..44],
        &[
            0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x8d, 0xf8, 0x5e, 0x1e, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]
    );
}

#[test]
fn complete_cleartext_sort_and_partition_streams_are_byte_fixed() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut sort = manager.create_file(SpillFileRole::SortRun).unwrap();
    sort.write_sort_run_start(1, 1).unwrap();
    sort.write_sort_row(b"row").unwrap();
    sort.finish_write().unwrap();
    let sort_bytes = std::fs::read(sort.path()).unwrap();
    assert_eq!(
        sort_bytes,
        [
            0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x8d, 0xf8, 0x5e, 0x1e, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x10, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x42, 0xc4, 0xc3, 0xb3, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x52, 0x53, 0x50, 0x01, 0x00,
            0x11, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0xd0,
            0x98, 0x58, 0x72, 0x6f, 0x77, 0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x7f, 0x00, 0x03,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd7, 0xd9, 0x7f, 0xef, 0x03,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
    );

    let mut partition = manager.create_file(SpillFileRole::NativePartition).unwrap();
    partition.write_partition_start(1).unwrap();
    partition.write_partition_entry(b"entry").unwrap();
    partition.finish_write().unwrap();
    let partition_bytes = std::fs::read(partition.path()).unwrap();
    assert_eq!(partition_bytes.len(), 173);
    assert_eq!(
        &partition_bytes[..44],
        &[
            0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x6e, 0xff, 0xd1, 0x90, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]
    );
    assert_eq!(
        &partition_bytes[44..88],
        &[
            0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x20, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0xe3, 0x94, 0x1d, 0x26, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]
    );
    assert_eq!(
        &partition_bytes[88..129],
        &[
            0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x21, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x8b, 0xb0, 0x13, 0x77, 0x65, 0x6e, 0x74, 0x72, 0x79,
        ]
    );
    assert_eq!(&partition_bytes[129..], &sort_bytes[131..]);
}

#[test]
fn complete_cleartext_rdf_aggregate_stream_is_byte_fixed() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut aggregate = manager
        .create_file(SpillFileRole::RdfAggregateState)
        .unwrap();
    aggregate.write_aggregate_state(b"rdf").unwrap();
    aggregate.finish_write().unwrap();

    assert_eq!(
        std::fs::read(aggregate.path()).unwrap(),
        [
            0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0xf0, 0xff, 0x7b, 0x5c, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x47, 0x52, 0x53, 0x50, 0x01, 0x00, 0x30, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x84, 0xc1, 0xaf, 0xd7, 0x72, 0x64, 0x66, 0x47,
            0x52, 0x53, 0x50, 0x01, 0x00, 0x7f, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0xe3, 0xdc, 0x0f, 0xd2, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00,
        ]
    );
}

#[test]
fn v1_file_start_has_fixed_header_bytes_and_sealed_round_trip() {
    let directory = TempDir::new().unwrap();
    let provider = Arc::new(RecordingSealedProvider::default());
    let aad = Arc::clone(&provider.aad);
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"row").unwrap();
    file.finish_write().unwrap();

    let bytes = std::fs::read(file.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GRSP");
    assert_eq!(&bytes[4..6], &[1, 0]);
    assert_eq!(bytes[6], SpillRecordKind::FileStart as u8);
    assert_eq!(bytes[7], 1);
    assert_eq!(&bytes[8..16], &0u64.to_le_bytes());
    assert_eq!(&bytes[16..24], &8u64.to_le_bytes());
    assert_eq!(&bytes[24..32], &9u64.to_le_bytes());
    assert_eq!(aad.lock()[0], bytes[0..32]);
    assert_eq!(read_valid_sort(&file).unwrap(), vec![b"row".to_vec()]);
}

#[test]
fn provider_receives_distinct_file_identities_and_exact_header_metadata() {
    let directory = TempDir::new().unwrap();
    let provider = Arc::new(RecordingSealedProvider::default());
    let identities = Arc::clone(&provider.identities);
    let calls = Arc::clone(&provider.calls);
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .build()
        .unwrap();
    let mut first = manager.create_file(SpillFileRole::SortRun).unwrap();
    first.write_sort_run_start(1, 0).unwrap();
    first.finish_write().unwrap();
    let mut second = manager.create_file(SpillFileRole::NativePartition).unwrap();
    second.write_partition_start(0).unwrap();
    second.finish_write().unwrap();
    let mut first_reader = first.reader().unwrap();
    first_reader.read_sort_run_start().unwrap();
    first_reader.finish().unwrap();
    let mut second_reader = second.reader().unwrap();
    second_reader.read_partition_start().unwrap();
    second_reader.finish().unwrap();

    let identities = identities.lock();
    assert_ne!(first.identity(), second.identity());
    assert_eq!(
        identities.as_slice(),
        [
            first.identity(),
            second.identity(),
            first.identity(),
            second.identity()
        ]
    );
    let calls = calls.lock();
    for identity in [first.identity(), second.identity()] {
        let per_file: Vec<_> = calls
            .iter()
            .filter(|call| call.meta.identity() == identity)
            .collect();
        assert_eq!(per_file.len(), 6);
        for opening in [false, true] {
            let phase: Vec<_> = per_file
                .iter()
                .filter(|call| call.opening == opening)
                .collect();
            assert_eq!(phase.len(), 3);
            for (sequence, call) in phase.into_iter().enumerate() {
                assert_eq!(call.meta.sequence(), sequence as u64);
                assert_eq!(&call.aad[..4], b"GRSP");
                assert_eq!(call.aad[6], call.meta.kind() as u8);
                assert_eq!(call.aad[7], u8::from(call.meta.is_sealed()));
                assert_eq!(&call.aad[8..16], &call.meta.sequence().to_le_bytes());
                assert_eq!(&call.aad[16..24], &call.meta.plaintext_len().to_le_bytes());
                assert_eq!(&call.aad[24..32], &call.meta.stored_len().to_le_bytes());
            }
        }
    }
}

#[test]
fn staging_file_cannot_be_read_before_publication() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();

    assert_eq!(
        file.reader().unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(manager.spilled_bytes(), 0);
}

#[test]
fn rdf_aggregate_role_round_trips_cleartext_and_sealed_records() {
    let clear_directory = TempDir::new().unwrap();
    let clear_manager = crate::execution::spill::BorrowedSpillFixture::new(clear_directory.path())
        .build()
        .unwrap();
    let mut clear = clear_manager
        .create_file(SpillFileRole::RdfAggregateState)
        .unwrap();
    clear.write_aggregate_state(b"first").unwrap();
    clear.write_aggregate_state(b"second").unwrap();
    clear.finish_write().unwrap();
    let mut clear_reader = clear.reader().unwrap();
    assert_eq!(clear_reader.read_aggregate_state().unwrap(), b"first");
    assert_eq!(clear_reader.read_aggregate_state().unwrap(), b"second");
    clear_reader.finish().unwrap();

    let sealed_directory = TempDir::new().unwrap();
    let sealed_manager =
        crate::execution::spill::BorrowedSpillFixture::new(sealed_directory.path())
            .provider(
                Arc::new(RecordingSealedProvider::default()),
                SpillFrameLimits::format_max(),
            )
            .build()
            .unwrap();
    let mut sealed = sealed_manager
        .create_file(SpillFileRole::RdfAggregateState)
        .unwrap();
    sealed.write_aggregate_state(b"sealed").unwrap();
    sealed.finish_write().unwrap();
    let mut sealed_reader = sealed.reader().unwrap();
    assert_eq!(sealed_reader.read_aggregate_state().unwrap(), b"sealed");
    sealed_reader.finish().unwrap();

    let mut wrong_role = clear_manager.create_file(SpillFileRole::SortRun).unwrap();
    assert_eq!(
        wrong_role
            .write_aggregate_state(b"wrong")
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    wrong_role.write_sort_run_start(1, 0).unwrap();
    wrong_role.finish_write().unwrap();
}

#[test]
fn every_truncated_header_or_payload_is_unexpected_eof() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"row payload").unwrap();
    file.finish_write().unwrap();
    let valid = std::fs::read(file.path()).unwrap();

    for cut in 0..valid.len() {
        std::fs::write(file.path(), &valid[..cut]).unwrap();
        let error = read_valid_sort(&file).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof, "cut={cut}");
    }
}

#[test]
fn bad_magic_version_kind_flags_sequence_crc_and_trailing_bytes_fail_closed() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"row").unwrap();
    file.finish_write().unwrap();
    let valid = std::fs::read(file.path()).unwrap();

    for offset in [0usize, 4, 6, 7, 8, 32] {
        let mut corrupt = valid.clone();
        corrupt[offset] ^= 0x80;
        std::fs::write(file.path(), corrupt).unwrap();
        assert_eq!(
            read_valid_sort(&file).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    let mut corrupt_payload = valid.clone();
    corrupt_payload[128] ^= 0x80;
    std::fs::write(file.path(), corrupt_payload).unwrap();
    assert_eq!(
        read_valid_sort(&file).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );

    let mut trailing = valid;
    trailing.push(0xff);
    std::fs::write(file.path(), trailing).unwrap();
    assert_eq!(
        read_valid_sort(&file).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn crc_failure_is_rejected_before_the_sealed_provider_open_callback() {
    let directory = TempDir::new().unwrap();
    let provider = Arc::new(RecordingSealedProvider::default());
    let calls = Arc::clone(&provider.calls);
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"row").unwrap();
    file.finish_write().unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    let row_offset = 45 + 49;
    bytes[row_offset + 37] ^= 0x80;
    std::fs::write(file.path(), bytes).unwrap();

    assert_eq!(
        read_valid_sort(&file).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    let opened: Vec<_> = calls
        .lock()
        .iter()
        .filter(|call| call.opening)
        .map(|call| call.meta.kind())
        .collect();
    assert_eq!(
        opened,
        [SpillRecordKind::FileStart, SpillRecordKind::SortRunStart]
    );
}

#[test]
fn premature_finish_is_retryable_before_any_terminal_bytes_are_consumed() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 2).unwrap();
    file.write_sort_row(b"one").unwrap();
    file.write_sort_row(b"two").unwrap();
    file.finish_write().unwrap();
    let mut reader = file.reader().unwrap();
    reader.read_sort_run_start().unwrap();
    assert_eq!(reader.read_sort_row().unwrap(), b"one");

    assert_eq!(
        reader.finish().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(reader.read_sort_row().unwrap(), b"two");
    reader.finish().unwrap();
}

#[test]
fn read_header_hook_failure_is_retryable_because_it_consumes_no_bytes() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(FailNthIo::new(
            SpillIoOperation::ReadHeader,
            3,
            io::ErrorKind::Interrupted,
        )))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let mut reader = file.reader().unwrap();
    reader.read_sort_run_start().unwrap();

    assert_eq!(
        reader.finish().unwrap_err().kind(),
        io::ErrorKind::Interrupted
    );
    reader.finish().unwrap();
}

#[test]
fn valid_crc_wrong_file_end_count_poisons_the_reader() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    let file_end_offset = 44 + 48;
    bytes[file_end_offset + 36..file_end_offset + 44].copy_from_slice(&99u64.to_le_bytes());
    rewrite_record_crc(&mut bytes, file_end_offset);
    std::fs::write(file.path(), bytes).unwrap();
    let mut reader = file.reader().unwrap();
    reader.read_sort_run_start().unwrap();

    assert_eq!(
        reader.finish().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(
        reader.finish().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn bad_role_reserved_transition_and_declared_count_fail_with_valid_crc() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"row").unwrap();
    file.finish_write().unwrap();
    let valid = std::fs::read(file.path()).unwrap();

    for payload_offset in [36usize, 39] {
        let mut corrupt = valid.clone();
        corrupt[payload_offset] ^= 0x04;
        rewrite_record_crc(&mut corrupt, 0);
        std::fs::write(file.path(), corrupt).unwrap();
        assert_eq!(
            read_valid_sort(&file).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    let mut bad_transition = valid.clone();
    bad_transition[44 + 6] = SpillRecordKind::PartitionStart as u8;
    rewrite_record_crc(&mut bad_transition, 44);
    std::fs::write(file.path(), bad_transition).unwrap();
    assert_eq!(
        read_valid_sort(&file).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );

    let mut bad_count = valid;
    bad_count[44 + 36 + 4..44 + 36 + 12].copy_from_slice(&2u64.to_le_bytes());
    rewrite_record_crc(&mut bad_count, 44);
    std::fs::write(file.path(), bad_count).unwrap();
    assert_eq!(
        read_valid_sort(&file).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn oversized_declared_lengths_are_rejected_before_payload_allocation() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    let oversized = u64::from(u32::MAX) + 1;
    bytes[16..24].copy_from_slice(&oversized.to_le_bytes());
    bytes[24..32].copy_from_slice(&oversized.to_le_bytes());
    std::fs::write(file.path(), bytes).unwrap();

    let error = file.reader().unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn platform_record_length_guard_checks_the_address_space_boundary() {
    assert!(super::file::validate_platform_record_lengths(64, 64, 64).is_ok());
    assert_eq!(
        super::file::validate_platform_record_lengths(65, 64, 64)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(
        super::file::validate_platform_record_lengths(64, 65, 64)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn format_max_declared_length_on_a_short_file_fails_before_reserve() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"x").unwrap();
    file.finish_write().unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    let declared = u64::from(u32::MAX);
    let row_offset = 44 + 48;
    bytes[row_offset + 16..row_offset + 24].copy_from_slice(&declared.to_le_bytes());
    bytes[row_offset + 24..row_offset + 32].copy_from_slice(&declared.to_le_bytes());
    std::fs::write(file.path(), bytes).unwrap();

    let error = read_valid_sort(&file).unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn configured_record_limit_rejects_before_writing_or_allocating() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::new(16, 16).unwrap(),
        )
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();

    let error = file.write_sort_row(&[0; 17]).unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn provider_failure_or_length_mismatch_never_retries_in_cleartext() {
    for provider in [
        RecordingSealedProvider {
            fail_seal: true,
            ..RecordingSealedProvider::default()
        },
        RecordingSealedProvider {
            wrong_stored_len: true,
            ..RecordingSealedProvider::default()
        },
    ] {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(Arc::new(provider), SpillFrameLimits::format_max())
            .build()
            .unwrap();

        let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

        assert!(matches!(
            error.kind(),
            io::ErrorKind::PermissionDenied | io::ErrorKind::InvalidData
        ));
        assert_eq!(manager.active_file_count(), 0);
        assert!(
            std::fs::read_dir(directory.path())
                .unwrap()
                .next()
                .is_none()
        );
    }
}

#[test]
fn unsealed_length_changing_provider_is_rejected_before_seal_or_publication() {
    let directory = TempDir::new().unwrap();
    let seal_called = Arc::new(AtomicBool::new(false));
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(UnsealedLengthChangingProvider {
                seal_called: Arc::clone(&seal_called),
            }),
            SpillFrameLimits::format_max(),
        )
        .build()
        .unwrap();

    let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(!seal_called.load(Ordering::Acquire));
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.spilled_bytes(), 0);
    assert!(
        std::fs::read_dir(directory.path())
            .unwrap()
            .next()
            .is_none()
    );
}

#[test]
fn begin_file_failure_leaves_no_registered_or_physical_artifact() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(FailBeginFileProvider),
            SpillFrameLimits::format_max(),
        )
        .build()
        .unwrap();

    let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    assert_eq!(manager.active_file_count(), 0);
    assert!(
        std::fs::read_dir(directory.path())
            .unwrap()
            .next()
            .is_none()
    );
}

#[test]
fn record_failure_poisons_the_staging_file_and_refuses_retry_or_finish() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(FailSortRowProvider),
            SpillFrameLimits::format_max(),
        )
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();

    assert_eq!(
        file.write_sort_row(b"partial").unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(
        file.write_sort_row(b"retry").unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        file.finish_write().unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(manager.spilled_bytes(), 0);
}

#[test]
fn provider_panics_poison_writer_and_reader_against_retry() {
    let writer_directory = TempDir::new().unwrap();
    let writer_manager =
        crate::execution::spill::BorrowedSpillFixture::new(writer_directory.path())
            .provider(
                Arc::new(PanicOnceProvider {
                    panic_on_open: false,
                    fired: Arc::new(AtomicBool::new(false)),
                }),
                SpillFrameLimits::format_max(),
            )
            .build()
            .unwrap();
    let mut writer = writer_manager.create_file(SpillFileRole::SortRun).unwrap();
    writer.write_sort_run_start(1, 1).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = writer.write_sort_row(b"row");
        }))
        .is_err()
    );
    assert_eq!(
        writer.write_sort_row(b"retry").unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        writer.finish_write().unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    drop(writer);
    assert_eq!(writer_manager.active_file_count(), 0);

    let reader_directory = TempDir::new().unwrap();
    let reader_manager =
        crate::execution::spill::BorrowedSpillFixture::new(reader_directory.path())
            .provider(
                Arc::new(PanicOnceProvider {
                    panic_on_open: true,
                    fired: Arc::new(AtomicBool::new(false)),
                }),
                SpillFrameLimits::format_max(),
            )
            .build()
            .unwrap();
    let mut file = reader_manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"row").unwrap();
    file.finish_write().unwrap();
    let mut reader = file.reader().unwrap();
    reader.read_sort_run_start().unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = reader.read_sort_row();
        }))
        .is_err()
    );
    assert_eq!(
        reader.read_sort_row().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn provider_open_failure_is_returned_without_plaintext_retry() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(RecordingSealedProvider {
                fail_open: true,
                ..RecordingSealedProvider::default()
            }),
            SpillFrameLimits::format_max(),
        )
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();

    let error = file.reader().unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
}

#[test]
fn provider_open_length_mismatch_fails_closed() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(RecordingSealedProvider {
                wrong_open_len: true,
                ..RecordingSealedProvider::default()
            }),
            SpillFrameLimits::format_max(),
        )
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();

    let error = file.reader().unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn reader_rejects_provider_stored_length_policy_drift_before_open() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(DriftingStoredLengthProvider::default()),
            SpillFrameLimits::format_max(),
        )
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();

    let error = file.reader().unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn deterministic_partial_record_write_poisons_file_and_blocks_publication() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(FailNthIo::new(
            SpillIoOperation::WritePayload,
            3,
            io::ErrorKind::BrokenPipe,
        )))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();

    assert_eq!(
        file.write_sort_row(b"torn").unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(
        file.write_sort_row(b"retry").unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        file.finish_write().unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(manager.spilled_bytes(), 0);
}

#[test]
fn sealed_provider_quota_charges_stored_framed_bytes_not_plaintext() {
    const BEFORE_FILE_END: u64 = 2 * SPILL_RECORD_HEADER_BYTES as u64 + (8 + 1) + (12 + 1);

    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(RecordingSealedProvider::default()),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(super::NoopSpillIo))
        .quota(SpillDiskQuota::new(BEFORE_FILE_END))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();

    assert_eq!(manager.disk_stats().reserved_live_bytes, BEFORE_FILE_END);
    let error = file.finish_write().unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::QuotaExceeded);
    assert_eq!(manager.disk_stats().reserved_live_bytes, BEFORE_FILE_END);

    file.close_and_delete().unwrap();
    assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
}

#[test]
fn failed_record_retains_full_conservative_charge_until_confirmed_delete() {
    const CHARGED_THROUGH_ROW: u64 = 3 * SPILL_RECORD_HEADER_BYTES as u64 + 8 + 12 + 4;

    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(FailNthIo::new(
            SpillIoOperation::WritePayload,
            3,
            io::ErrorKind::BrokenPipe,
        )))
        .quota(SpillDiskQuota::new(CHARGED_THROUGH_ROW))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();

    assert_eq!(
        file.write_sort_row(b"torn").unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(file.bytes_written(), CHARGED_THROUGH_ROW - 40);
    assert_eq!(
        manager.disk_stats().reserved_live_bytes,
        CHARGED_THROUGH_ROW
    );
    assert_eq!(manager.disk_stats().published_live_bytes, 0);

    file.close_and_delete().unwrap();
    assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
}

#[test]
fn panic_after_reservation_retains_full_conservative_charge_until_delete() {
    const CHARGED_THROUGH_ROW: u64 = 3 * SPILL_RECORD_HEADER_BYTES as u64 + 8 + 12 + 4;

    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(PanicNthIo::new(SpillIoOperation::WritePayload, 3)))
        .quota(SpillDiskQuota::new(CHARGED_THROUGH_ROW))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();

    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = file.write_sort_row(b"torn");
        }))
        .is_err()
    );
    assert_eq!(file.bytes_written(), CHARGED_THROUGH_ROW - 40);
    assert_eq!(
        manager.disk_stats().reserved_live_bytes,
        CHARGED_THROUGH_ROW
    );
    assert_eq!(manager.disk_stats().published_live_bytes, 0);

    file.close_and_delete().unwrap();
    assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
}

#[test]
fn publication_rejects_synced_physical_length_drift_without_releasing_charge() {
    const EMPTY_SORT_BYTES: u64 = 3 * SPILL_RECORD_HEADER_BYTES as u64 + 8 + 12 + 8;

    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .quota(SpillDiskQuota::new(EMPTY_SORT_BYTES))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    {
        use std::io::Write as _;
        let mut intruder = std::fs::OpenOptions::new()
            .append(true)
            .open(file.path())
            .unwrap();
        intruder.write_all(&[0xa5; 512]).unwrap();
        intruder.sync_all().unwrap();
    }

    let error = file.finish_write().unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(manager.spilled_bytes(), 0);
    assert_eq!(manager.disk_stats().reserved_live_bytes, EMPTY_SORT_BYTES);

    file.close_and_delete().unwrap();
    assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
}

#[test]
fn deterministic_flush_and_sync_failures_block_publication() {
    for operation in [SpillIoOperation::Flush, SpillIoOperation::Sync] {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(
                Arc::new(super::CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(Arc::new(FailNthIo::new(
                operation,
                1,
                io::ErrorKind::WriteZero,
            )))
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();

        assert_eq!(
            file.finish_write().unwrap_err().kind(),
            io::ErrorKind::WriteZero
        );
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(
            file.finish_write().unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}

#[test]
fn deterministic_read_failure_propagates_without_accepting_the_stream() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(FailNthIo::new(
            SpillIoOperation::ReadPayload,
            1,
            io::ErrorKind::TimedOut,
        )))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();

    assert_eq!(file.reader().unwrap_err().kind(), io::ErrorKind::TimedOut);
}

#[test]
fn deterministic_delete_failure_retains_registration_for_retry() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(FailNthIo::new(
            SpillIoOperation::Delete,
            1,
            io::ErrorKind::PermissionDenied,
        )))
        .quota(SpillDiskQuota::new(1024))
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let published = manager.spilled_bytes();
    let charged = manager.disk_stats().reserved_live_bytes;

    assert_eq!(
        file.close_and_delete().unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    assert_eq!(manager.active_file_count(), 1);
    assert_eq!(manager.spilled_bytes(), published);
    assert_eq!(manager.disk_stats().reserved_live_bytes, charged);

    file.close_and_delete().unwrap();
    assert_eq!(manager.active_file_count(), 0);
    assert_eq!(manager.spilled_bytes(), 0);
    assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
}

#[test]
fn deterministic_create_failure_registers_no_ghost_file() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(
            Arc::new(super::CleartextSpillRecordProvider),
            SpillFrameLimits::format_max(),
        )
        .io(Arc::new(FailNthIo::new(
            SpillIoOperation::Create,
            1,
            io::ErrorKind::PermissionDenied,
        )))
        .build()
        .unwrap();

    assert_eq!(
        manager
            .create_file(SpillFileRole::SortRun)
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    assert_eq!(manager.active_file_count(), 0);
    assert!(
        std::fs::read_dir(directory.path())
            .unwrap()
            .next()
            .is_none()
    );
}

#[test]
fn protocol_error_poisons_reader_after_consuming_a_record() {
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"row").unwrap();
    file.finish_write().unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    bytes[44 + 6] = SpillRecordKind::SortRow as u8;
    rewrite_record_crc(&mut bytes, 44);
    std::fs::write(file.path(), bytes).unwrap();
    let mut reader = file.reader().unwrap();

    assert_eq!(
        reader.read_sort_run_start().unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    let retry = reader.read_sort_run_start().unwrap_err();
    assert_eq!(retry.kind(), io::ErrorKind::InvalidData);
    assert!(retry.to_string().contains("poisoned"));
}

#[test]
fn provider_accounted_reader_keeps_provider_state_before_workspace_grant() {
    let directory = TempDir::new().unwrap();
    let buffer_manager = exact_buffer_manager(1 << 20);
    let resources =
        crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
    let (provider, witness) = grant_lifetime_provider(resources.clone(), 64);
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    assert_eq!(witness.drops(), 1, "writer provider state is already gone");

    let grant = resources.try_allocate(0).unwrap();
    let reader = file
        .reader_with_owned_provider_admission(grant)
        .expect("provider-accounted reader construction must fit the test budget");
    let receipt = reader.receipt();
    assert!(receipt.bytes() >= 64);
    assert_eq!(resources.query_stats().allocated_bytes, receipt.bytes());
    drop(reader);

    assert_eq!(witness.drops(), 2);
    assert!(
        !witness.released_before_drop(),
        "provider state must drop before its file-workspace grant"
    );
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn reader_accounted_error_construction_pre_admits_exact_terminal_failure_transport() {
    const READER_WORKSPACE: usize = 4 * MAX_FIXED_CONTROL_PAYLOAD_BYTES;

    let publication_bytes = reader_operation_error_publication_bytes();
    let total_bytes = READER_WORKSPACE
        .checked_add(publication_bytes)
        .expect("test reader authority total is representable");
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();

    let exact_resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(total_bytes)).unwrap();
    let reader = file
        .reader_with_owned_provider_admission(exact_resources.try_allocate(0).unwrap())
        .expect("exact reader plus terminal-error publication authority must be sufficient");
    assert_eq!(reader.receipt().bytes(), total_bytes);
    assert_eq!(exact_resources.query_stats().allocated_bytes, total_bytes);
    drop(reader);
    assert_eq!(exact_resources.query_stats().allocated_bytes, 0);

    let short_resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(total_bytes - 1)).unwrap();
    let error = file
        .reader_with_owned_provider_admission(short_resources.try_allocate(0).unwrap())
        .expect_err("one byte short must deny qualified reader construction");
    assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);
    drop(error);
    assert_eq!(short_resources.query_stats().allocated_bytes, 0);

    let denied_directory = TempDir::new().unwrap();
    let begin_calls = Arc::new(AtomicUsize::new(0));
    let denied_manager =
        crate::execution::spill::BorrowedSpillFixture::new(denied_directory.path())
            .provider(
                Arc::new(CountingQualifiedCleartextProvider {
                    begin_calls: Arc::clone(&begin_calls),
                }),
                SpillFrameLimits::format_max(),
            )
            .build()
            .unwrap();
    let mut denied_file = denied_manager.create_file(SpillFileRole::SortRun).unwrap();
    denied_file.write_sort_run_start(1, 0).unwrap();
    denied_file.finish_write().unwrap();
    assert_eq!(begin_calls.load(Ordering::Acquire), 1);
    let publication_short_resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(publication_bytes - 1))
            .unwrap();
    let error = denied_file
        .reader_with_owned_provider_admission(publication_short_resources.try_allocate(0).unwrap())
        .expect_err("one-byte-short publication authority must fail before provider begin");
    assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);
    assert!(error.operation_error_publication_failure().is_some());
    assert!(matches!(
        error.memory_error(),
        Some(grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded { .. })
    ));
    assert_eq!(
        begin_calls.load(Ordering::Acquire),
        1,
        "publication admission must precede the reader provider callback"
    );
    drop(error);
    assert_eq!(publication_short_resources.query_stats().allocated_bytes, 0);
}

#[test]
fn reader_accounted_error_start_mismatch_publishes_typed_owner_and_auto_quiesces() {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let reader = file
        .reader_with_owned_provider_admission(resources.try_allocate(0).unwrap())
        .expect("qualified reader construction succeeds");

    let error: AccountedError = reader
        .read_sort_run_start_owned(2, 0)
        .expect_err("column mismatch must escape through the accounted owner");
    assert!(error.is::<super::file::ProviderAccountedReaderOperationError>());
    assert_eq!(
        error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
            source.kind()
        }),
        Some(io::ErrorKind::InvalidData)
    );
    let admitted = resources.query_stats().allocated_bytes;
    file.close_and_delete()
        .expect("published declaration failure must already have quiesced its reader");
    assert_eq!(resources.query_stats().allocated_bytes, admitted);
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn provider_accounted_reader_fails_closed_before_an_unqualified_control_hook() {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let (provider, witness) = grant_lifetime_provider(resources.clone(), 64);
    let io = Arc::new(UnqualifiedFixedControlReadIo::default());
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .io(io.clone())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    assert_eq!(witness.drops(), 1, "writer provider state is already gone");

    let grant = resources.try_allocate(0).unwrap();
    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
        panic!("an undeclared qualified reader hook must fail closed");
    };

    assert!(error.to_string().contains("qualified reader hook"));
    assert_eq!(
        witness.drops(),
        1,
        "provider begin_file must not run before hook admission"
    );
    assert_eq!(io.read_callbacks.load(Ordering::Acquire), 0);
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

fn published_reader_hook_failure_file(
    failure: ReaderControlHookFailure,
) -> (
    TempDir,
    crate::execution::QueryResourceContext,
    Arc<QualifiedFailingFixedControlReadIo>,
    GrantLifetimeWitness,
    super::SpillFile,
) {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let (provider, witness) = grant_lifetime_provider(resources.clone(), 64);
    let io = Arc::new(QualifiedFailingFixedControlReadIo {
        resources: resources.clone(),
        target: SpillIoOperation::ReadPayload,
        failure,
        armed: AtomicBool::new(false),
        fired: AtomicBool::new(false),
        payload_drop_observation: Arc::new(AtomicUsize::new(usize::MAX)),
        hostile_authority_when_dropped: Arc::new(AtomicUsize::new(usize::MAX)),
        hostile_panic_payload_dropped: Arc::new(AtomicBool::new(false)),
        shared_failure_heap: Arc::new(Mutex::new(Vec::new())),
        authority_during_shared_growth: AtomicUsize::new(usize::MAX),
    });
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .io(io.clone())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    io.armed.store(true, Ordering::Release);
    (directory, resources, io, witness, file)
}

#[test]
fn provider_accounted_reader_admits_control_hook_error_payload_with_control_buffers() {
    let (_directory, resources, io, witness, file) =
        published_reader_hook_failure_file(ReaderControlHookFailure::Error);
    let grant = resources.try_allocate(0).unwrap();

    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
        panic!("fixed-control hook error must escape");
    };
    assert_eq!(error.io_error_kind(), Some(io::ErrorKind::PermissionDenied));
    assert_eq!(
        error.granted_bytes(),
        64 + READER_CONTROL_HOOK_WORKSPACE + 2 * MAX_FIXED_CONTROL_PAYLOAD_BYTES,
        "provider, hook, exact stored, and exact plaintext control buffers must be co-live"
    );
    let admitted = resources.query_stats().allocated_bytes;
    assert_eq!(io.shared_failure_heap.lock().capacity(), 0);
    assert_eq!(
        io.authority_during_shared_growth.load(Ordering::Acquire),
        admitted
            .checked_add(reader_operation_error_publication_bytes())
            .expect("test hook authority total is representable"),
        "hook growth occurs while workspace and publication authority are live; the empty publication block is reclaimed before error return"
    );
    assert_eq!(
        io.payload_drop_observation.load(Ordering::Acquire),
        usize::MAX
    );
    assert_eq!(witness.drops(), 2);
    assert!(!witness.released_before_drop());

    drop(error);

    assert_eq!(
        io.payload_drop_observation.load(Ordering::Acquire),
        admitted,
        "hook error payload must drop before its co-live workspace"
    );
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn provider_accounted_reader_captures_control_hook_panic_and_reclaims_clean_drop() {
    let (_directory, resources, io, witness, file) =
        published_reader_hook_failure_file(ReaderControlHookFailure::Panic);
    let grant = resources.try_allocate(0).unwrap();

    let accounted = file
        .reader_with_owned_provider_admission(grant)
        .expect_err("fixed-control hook panic must be captured as fatal");
    assert!(accounted.is_fatal_captured_panic());
    assert!(accounted.panic_payload_is::<HeapReaderConstructionPayload>());
    assert_eq!(
        accounted
            .panic_payload_for_test()
            .expect("captured hook panic retains its original payload")
            .downcast_ref::<HeapReaderConstructionPayload>()
            .map(|payload| payload.bytes.len()),
        Some(32 * 1024)
    );
    assert_eq!(
        accounted.granted_bytes(),
        64 + READER_CONTROL_HOOK_WORKSPACE + 2 * MAX_FIXED_CONTROL_PAYLOAD_BYTES,
        "provider, hook, exact stored, and exact plaintext control buffers must be co-live"
    );
    let admitted = resources.query_stats().allocated_bytes;
    assert_eq!(io.shared_failure_heap.lock().capacity(), 0);
    assert_eq!(
        io.authority_during_shared_growth.load(Ordering::Acquire),
        admitted
            .checked_add(reader_operation_error_publication_bytes())
            .expect("test hook authority total is representable"),
        "hook unwind occurs while workspace and publication authority are live; the empty publication block is reclaimed before error return"
    );
    assert_eq!(witness.drops(), 2);
    assert!(!witness.released_before_drop());

    drop(accounted);

    assert_eq!(
        io.payload_drop_observation.load(Ordering::Acquire),
        admitted,
        "hook panic payload must drop before its co-live workspace"
    );
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn hostile_hook_primary_drop_retains_its_forgotten_payload_authority() {
    const CHILD_ENV: &str = "GRAFEO_READER_HOOK_HOSTILE_PRIMARY_DROP_CHILD";
    const HANDSHAKE: &str = "GRAFEO_READER_HOOK_HOSTILE_PRIMARY_DROP_OK";

    if std::env::var_os(CHILD_ENV).is_some() {
        for failure in [
            ReaderControlHookFailure::HostileErrorDrop,
            ReaderControlHookFailure::HostilePanicDrop,
        ] {
            let (_directory, resources, io, _witness, file) =
                published_reader_hook_failure_file(failure);
            let grant = resources.try_allocate(0).unwrap();
            match failure {
                ReaderControlHookFailure::HostileErrorDrop => {
                    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
                        panic!("hostile hook error must escape");
                    };
                    assert_eq!(error.io_error_kind(), Some(io::ErrorKind::PermissionDenied));
                    drop(error);
                }
                ReaderControlHookFailure::HostilePanicDrop => {
                    let error = file
                        .reader_with_owned_provider_admission(grant)
                        .expect_err("hostile hook panic must be captured");
                    assert!(error.is_fatal_captured_panic());
                    drop(error);
                }
                ReaderControlHookFailure::Error | ReaderControlHookFailure::Panic => {
                    unreachable!("hostile child iterates only hostile-drop modes")
                }
            }

            let retained = resources.query_stats().allocated_bytes;
            assert!(retained >= READER_CONTROL_HOOK_WORKSPACE);
            assert_eq!(
                io.hostile_authority_when_dropped.load(Ordering::Acquire),
                retained,
                "hostile primary must drop while its authority is live"
            );
            assert!(
                !io.hostile_panic_payload_dropped.load(Ordering::Acquire),
                "the cleanup backstop intentionally forgets the secondary heap panic payload"
            );
        }
        println!("{HANDSHAKE}");
        return;
    }

    let test_name = "execution::spill::framing_tests::hostile_hook_primary_drop_retains_its_forgotten_payload_authority";
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
        "hostile hook primary drop released forgotten-payload authority\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

fn published_reader_construction_failure_file(
    failure: ReaderConstructionFailure,
) -> (
    TempDir,
    crate::execution::QueryResourceContext,
    Arc<ReaderConstructionFailureProvider>,
    super::SpillFile,
) {
    let directory = TempDir::new().unwrap();
    let buffer_manager = exact_buffer_manager(1 << 20);
    let resources =
        crate::execution::QueryResourceContext::new(Arc::clone(&buffer_manager)).unwrap();
    let provider = Arc::new(ReaderConstructionFailureProvider {
        resources: resources.clone(),
        begin_calls: AtomicUsize::new(0),
        failure,
        payload_drop_observation: Arc::new(AtomicUsize::new(usize::MAX)),
        hostile_record_drops: Arc::new(AtomicUsize::new(0)),
        hostile_authority_when_dropped: Arc::new(AtomicUsize::new(usize::MAX)),
        hostile_panic_payload_dropped: Arc::new(AtomicBool::new(false)),
        shared_failure_heap: Arc::new(Mutex::new(Vec::new())),
        authority_during_shared_growth: AtomicUsize::new(usize::MAX),
    });
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider.clone(), SpillFrameLimits::format_max())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    (directory, resources, provider, file)
}

fn published_qualified_operation_failure_file(
    failure: QualifiedOperationFailure,
) -> (
    TempDir,
    crate::execution::QueryResourceContext,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<AtomicBool>,
    super::SpillFile,
) {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let payload_drop_observation = Arc::new(AtomicUsize::new(usize::MAX));
    let hostile_authority_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
    let hostile_panic_payload_dropped = Arc::new(AtomicBool::new(false));
    let provider = Arc::new(QualifiedOperationFailureProvider {
        resources: resources.clone(),
        failure,
        payload_drop_observation: Arc::clone(&payload_drop_observation),
        hostile_authority_when_dropped: Arc::clone(&hostile_authority_when_dropped),
        hostile_panic_payload_dropped: Arc::clone(&hostile_panic_payload_dropped),
    });
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    let rows = u64::from(!matches!(failure, QualifiedOperationFailure::FinishError));
    file.write_sort_run_start(1, rows).unwrap();
    if rows == 1 {
        file.write_sort_row(b"qualified-operation-row").unwrap();
    }
    file.finish_write().unwrap();
    (
        directory,
        resources,
        payload_drop_observation,
        hostile_authority_when_dropped,
        hostile_panic_payload_dropped,
        file,
    )
}

#[test]
fn provider_accounted_reader_error_drops_heap_payload_before_workspace() {
    for failure in [
        ReaderConstructionFailure::BeginError,
        ReaderConstructionFailure::ControlError,
    ] {
        let (_directory, resources, provider, file) =
            published_reader_construction_failure_file(failure);
        let grant = resources.try_allocate(0).unwrap();

        let Err(error) = file.reader_with_owned_provider_admission(grant) else {
            panic!("reader provider error must escape");
        };
        assert_eq!(error.io_error_kind(), Some(io::ErrorKind::PermissionDenied));
        assert!(error.granted_bytes() >= READER_CONSTRUCTION_WORKSPACE);
        let admitted = resources.query_stats().allocated_bytes;
        if matches!(failure, ReaderConstructionFailure::BeginError) {
            assert_eq!(provider.shared_failure_heap.lock().capacity(), 0);
            assert_eq!(
                provider
                    .authority_during_shared_growth
                    .load(Ordering::Acquire),
                admitted
                    .checked_add(reader_operation_error_publication_bytes())
                    .expect("test provider authority total is representable"),
                "provider Err occurs while workspace and publication authority are live; the empty publication block is reclaimed before error return"
            );
        }
        assert_eq!(
            provider.payload_drop_observation.load(Ordering::Acquire),
            usize::MAX
        );

        drop(error);

        assert_eq!(
            provider.payload_drop_observation.load(Ordering::Acquire),
            admitted,
            "physical error payload must drop before its admitted workspace"
        );
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
}

#[test]
fn provider_accounted_reader_panic_drops_heap_payload_before_workspace() {
    for failure in [
        ReaderConstructionFailure::BeginPanic,
        ReaderConstructionFailure::ControlPanic,
    ] {
        let (_directory, resources, provider, file) =
            published_reader_construction_failure_file(failure);
        let grant = resources.try_allocate(0).unwrap();

        let accounted = file
            .reader_with_owned_provider_admission(grant)
            .expect_err("reader provider panic must be captured as fatal");
        assert!(accounted.is_fatal_captured_panic());
        assert!(accounted.panic_payload_is::<HeapReaderConstructionPayload>());
        assert_eq!(
            accounted
                .panic_payload_for_test()
                .expect("captured provider panic retains its original payload")
                .downcast_ref::<HeapReaderConstructionPayload>()
                .map(|payload| payload.bytes.len()),
            Some(32 * 1024)
        );
        assert!(accounted.granted_bytes() >= READER_CONSTRUCTION_WORKSPACE);
        let admitted = resources.query_stats().allocated_bytes;
        if matches!(failure, ReaderConstructionFailure::BeginPanic) {
            assert_eq!(provider.shared_failure_heap.lock().capacity(), 0);
            assert_eq!(
                provider
                    .authority_during_shared_growth
                    .load(Ordering::Acquire),
                admitted
                    .checked_add(reader_operation_error_publication_bytes())
                    .expect("test provider authority total is representable"),
                "provider unwind occurs while workspace and publication authority are live; the empty publication block is reclaimed before error return"
            );
        }
        assert_eq!(
            provider.payload_drop_observation.load(Ordering::Acquire),
            usize::MAX
        );

        drop(accounted);

        assert_eq!(
            provider.payload_drop_observation.load(Ordering::Acquire),
            admitted,
            "physical panic payload must drop before its admitted workspace"
        );
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
}

#[test]
fn reader_accounted_error_row_failure_publishes_cloneable_typed_owner_and_auto_quiesces() {
    for failure in [
        QualifiedOperationFailure::DataError,
        QualifiedOperationFailure::DataPanic,
    ] {
        let (
            _directory,
            resources,
            payload_drop_observation,
            _hostile_authority,
            _secondary_panic,
            mut file,
        ) = published_qualified_operation_failure_file(failure);
        let reader_grant = resources.try_allocate(0).unwrap();
        let reader = file
            .reader_with_owned_provider_admission(reader_grant)
            .expect("qualified reader construction succeeds");
        let reader = reader
            .read_sort_run_start_owned(1, 1)
            .expect("qualified declaration succeeds");
        let payload_grant = resources.try_allocate(0).unwrap();
        let Err(error): Result<_, AccountedError> = reader.read_sort_row_owned(payload_grant)
        else {
            panic!("qualified provider row failure must escape");
        };
        assert!(std::error::Error::source(&error).is_none());
        assert!(error.is::<super::file::ProviderAccountedReaderOperationError>());
        match failure {
            QualifiedOperationFailure::DataError => {
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.kind()
                    ),
                    Some(io::ErrorKind::PermissionDenied)
                );
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.is_fatal_captured_panic()
                    ),
                    Some(false)
                );
            }
            QualifiedOperationFailure::DataPanic => {
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.kind()
                    ),
                    Some(io::ErrorKind::Other)
                );
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.is_fatal_captured_panic()
                    ),
                    Some(true)
                );
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.panic_payload_is::<HeapReaderConstructionPayload>()
                    ),
                    Some(true)
                );
            }
            QualifiedOperationFailure::DataHostileErrorDrop
            | QualifiedOperationFailure::FinishError => unreachable!(),
        }
        assert_eq!(payload_drop_observation.load(Ordering::Acquire), usize::MAX);
        let admitted = resources.query_stats().allocated_bytes;
        let retained_payload_authority = error
            .inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
                source.retained_authority_bytes()
            })
            .expect("typed reader-operation owner remains inspectable");
        assert_eq!(
            retained_payload_authority
                .checked_add(error.granted_bytes())
                .expect("test publication authority total is representable"),
            admitted
        );

        let clone = error.clone();
        assert!(error.ptr_eq(&clone));
        assert_eq!(resources.query_stats().allocated_bytes, admitted);
        file.close_and_delete()
            .expect("published operation failure must already have quiesced its reader");
        assert_eq!(
            resources.query_stats().allocated_bytes,
            admitted,
            "file deletion cannot release published diagnostic authority"
        );
        drop(error);
        assert_eq!(resources.query_stats().allocated_bytes, admitted);
        assert_eq!(payload_drop_observation.load(Ordering::Acquire), usize::MAX);
        drop(clone);
        assert_eq!(payload_drop_observation.load(Ordering::Acquire), admitted);
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn reader_accounted_error_row_hostile_drop_retains_every_colive_authority() {
    const CHILD_ENV: &str = "GRAFEO_QUALIFIED_ROW_HOSTILE_ERROR_DROP_CHILD";
    const HANDSHAKE: &str = "GRAFEO_QUALIFIED_ROW_HOSTILE_ERROR_DROP_OK";

    if std::env::var_os(CHILD_ENV).is_some() {
        let (
            _directory,
            resources,
            _payload_drop,
            hostile_authority,
            secondary_panic_dropped,
            file,
        ) = published_qualified_operation_failure_file(
            QualifiedOperationFailure::DataHostileErrorDrop,
        );
        let reader = file
            .reader_with_owned_provider_admission(resources.try_allocate(0).unwrap())
            .unwrap()
            .read_sort_run_start_owned(1, 1)
            .unwrap();
        let Err(error) = reader.read_sort_row_owned(resources.try_allocate(0).unwrap()) else {
            panic!("hostile provider error must escape");
        };
        assert_eq!(
            error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
                source.kind()
            }),
            Some(io::ErrorKind::PermissionDenied)
        );
        let publication_bytes = error.granted_bytes();
        drop(error);

        let retained = resources.query_stats().allocated_bytes;
        assert!(retained >= QUALIFIED_OPERATION_PROVIDER_WORKSPACE);
        assert_eq!(
            hostile_authority.load(Ordering::Acquire),
            retained
                .checked_add(publication_bytes)
                .expect("test hostile-drop authority total is representable"),
            "the inner hostile payload drops while publication authority is still live"
        );
        assert!(
            !secondary_panic_dropped.load(Ordering::Acquire),
            "the hostile secondary panic payload is intentionally forgotten"
        );
        println!("{HANDSHAKE}");
        return;
    }

    let test_name = "execution::spill::framing_tests::reader_accounted_error_row_hostile_drop_retains_every_colive_authority";
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
        "hostile qualified row error released forgotten-payload authority\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

#[test]
fn reader_accounted_error_row_hook_failure_keeps_workspace_owned() {
    for (target, failure) in [
        (
            SpillIoOperation::ReadHeader,
            ReaderControlHookFailure::Error,
        ),
        (
            SpillIoOperation::ReadPayload,
            ReaderControlHookFailure::Panic,
        ),
    ] {
        let directory = TempDir::new().unwrap();
        let resources =
            crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
        let (provider, witness) = grant_lifetime_provider(resources.clone(), 64);
        let io = Arc::new(QualifiedFailingFixedControlReadIo {
            resources: resources.clone(),
            target,
            failure,
            armed: AtomicBool::new(false),
            fired: AtomicBool::new(false),
            payload_drop_observation: Arc::new(AtomicUsize::new(usize::MAX)),
            hostile_authority_when_dropped: Arc::new(AtomicUsize::new(usize::MAX)),
            hostile_panic_payload_dropped: Arc::new(AtomicBool::new(false)),
            shared_failure_heap: Arc::new(Mutex::new(Vec::new())),
            authority_during_shared_growth: AtomicUsize::new(usize::MAX),
        });
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(provider, SpillFrameLimits::format_max())
            .io(io.clone())
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        file.write_sort_row(b"hook-failure-row").unwrap();
        file.finish_write().unwrap();
        let reader = file
            .reader_with_owned_provider_admission(resources.try_allocate(0).unwrap())
            .unwrap()
            .read_sort_run_start_owned(1, 1)
            .unwrap();
        io.armed.store(true, Ordering::Release);
        let Err(error) = reader.read_sort_row_owned(resources.try_allocate(0).unwrap()) else {
            panic!("qualified row hook failure must escape");
        };
        match failure {
            ReaderControlHookFailure::Error => {
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.kind()
                    ),
                    Some(io::ErrorKind::PermissionDenied)
                );
            }
            ReaderControlHookFailure::Panic => {
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.is_fatal_captured_panic()
                    ),
                    Some(true)
                );
                assert_eq!(
                    error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(
                        |source| source.panic_payload_is::<HeapReaderConstructionPayload>()
                    ),
                    Some(true)
                );
            }
            ReaderControlHookFailure::HostileErrorDrop
            | ReaderControlHookFailure::HostilePanicDrop => unreachable!(),
        }
        let admitted = resources.query_stats().allocated_bytes;
        assert!(admitted >= READER_CONTROL_HOOK_WORKSPACE);
        file.close_and_delete()
            .expect("published hook failure must already have quiesced its reader");
        drop(error);
        assert_eq!(
            io.payload_drop_observation.load(Ordering::Acquire),
            admitted
        );
        assert!(!witness.released_before_drop());
        assert_eq!(resources.query_stats().allocated_bytes, 0);
    }
}

#[test]
fn reader_accounted_error_row_payload_denial_is_preallocation_and_reclaims_on_drop() {
    const PAYLOAD_LEN: usize = 4_097;
    const READER_WORKSPACE: usize = 4 * MAX_FIXED_CONTROL_PAYLOAD_BYTES;
    let row_peak = 4 * PAYLOAD_LEN;
    let publication_bytes = reader_operation_error_publication_bytes();
    let reader_authority = READER_WORKSPACE
        .checked_add(publication_bytes)
        .expect("test reader authority total is representable");
    let directory = TempDir::new().unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(&vec![0x7e; PAYLOAD_LEN]).unwrap();
    file.finish_write().unwrap();
    let resources = crate::execution::QueryResourceContext::new(exact_buffer_manager(
        reader_authority + row_peak - 1,
    ))
    .unwrap();
    let reader = file
        .reader_with_owned_provider_admission(resources.try_allocate(0).unwrap())
        .unwrap()
        .read_sort_run_start_owned(1, 1)
        .unwrap();
    assert_eq!(reader.receipt().bytes(), reader_authority);
    let Err(error) = reader.read_sort_row_owned(resources.try_allocate(0).unwrap()) else {
        panic!("one-byte-short row peak must be denied before exact allocation");
    };
    assert_eq!(
        error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
            source.kind()
        }),
        Some(io::ErrorKind::OutOfMemory)
    );
    assert!(matches!(
        error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
            source.operator_classification()
        }),
        Some(
            super::file::ProviderAccountedReaderFailureClassification::Memory(
                grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded { .. }
            )
        )
    ));
    assert_eq!(
        error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
            source.retained_authority_bytes()
        }),
        Some(READER_WORKSPACE)
    );
    file.close_and_delete()
        .expect("published admission failure must already have quiesced its reader");
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn reader_accounted_error_finish_quiesces_file_but_retains_diagnostic_authority() {
    let (
        _directory,
        resources,
        payload_drop_observation,
        _hostile_authority,
        _secondary_panic,
        mut file,
    ) = published_qualified_operation_failure_file(QualifiedOperationFailure::FinishError);
    let reader = file
        .reader_with_owned_provider_admission(resources.try_allocate(0).unwrap())
        .unwrap()
        .read_sort_run_start_owned(1, 0)
        .unwrap();
    let resolution = reader
        .finish_sort_run_owned()
        .expect_err("qualified FileEnd provider error must escape");
    let super::file::ProviderAccountedReaderResolutionError::Accounted(error) = resolution else {
        panic!("provider finish failure lost its pre-admitted diagnostic owner")
    };
    assert_eq!(
        error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
            source.kind()
        }),
        Some(io::ErrorKind::PermissionDenied)
    );
    let admitted = resources.query_stats().allocated_bytes;
    file.close_and_delete()
        .expect("published finish error must already have quiesced its reader");
    assert_eq!(resources.query_stats().allocated_bytes, admitted);
    drop(error);
    assert_eq!(payload_drop_observation.load(Ordering::Acquire), admitted);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn reader_accounted_error_construction_hostile_open_record_drop_retains_authority() {
    const CHILD_ENV: &str = "GRAFEO_READER_CONSTRUCTION_HOSTILE_DROP_CHILD";
    const HANDSHAKE: &str = "GRAFEO_READER_CONSTRUCTION_HOSTILE_DROP_OK";
    const PRIMARY_PAYLOAD: u64 = 0xc011_57a7;

    if std::env::var_os(CHILD_ENV).is_some() {
        for failure in [
            ReaderConstructionFailure::StoredLenErrorWithHostileDrop,
            ReaderConstructionFailure::StoredLenPanicWithHostileDrop,
            ReaderConstructionFailure::OpenBoundPanicWithHostileDrop,
            ReaderConstructionFailure::ControlErrorWithHostileDrop,
            ReaderConstructionFailure::ControlPanicWithHostileDrop,
            ReaderConstructionFailure::SuccessWithHostileDrop,
        ] {
            let (_directory, resources, provider, file) =
                published_reader_construction_failure_file(failure);
            let grant = resources.try_allocate(0).unwrap();
            match failure {
                ReaderConstructionFailure::StoredLenErrorWithHostileDrop
                | ReaderConstructionFailure::ControlErrorWithHostileDrop => {
                    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
                        panic!("provider callback must return its primary error");
                    };
                    assert_eq!(error.io_error_kind(), Some(io::ErrorKind::PermissionDenied));
                    drop(error);
                }
                ReaderConstructionFailure::StoredLenPanicWithHostileDrop
                | ReaderConstructionFailure::OpenBoundPanicWithHostileDrop
                | ReaderConstructionFailure::ControlPanicWithHostileDrop => {
                    let accounted = file
                        .reader_with_owned_provider_admission(grant)
                        .expect_err("provider callback panic must be captured");
                    assert!(accounted.is_fatal_captured_panic());
                    assert!(accounted.panic_payload_is::<u64>());
                    let expected = match failure {
                        ReaderConstructionFailure::StoredLenPanicWithHostileDrop => 0x5702_ed1e,
                        ReaderConstructionFailure::OpenBoundPanicWithHostileDrop => 0x0b0d_5ca1,
                        ReaderConstructionFailure::ControlPanicWithHostileDrop => PRIMARY_PAYLOAD,
                        _ => unreachable!("matched only provider panic modes"),
                    };
                    assert_eq!(
                        accounted
                            .panic_payload_for_test()
                            .expect("captured panic retains original identity")
                            .downcast_ref::<u64>(),
                        Some(&expected)
                    );
                    drop(accounted);
                }
                ReaderConstructionFailure::SuccessWithHostileDrop => {
                    let reader = file
                        .reader_with_owned_provider_admission(grant)
                        .expect("reader construction itself succeeds");
                    drop(reader);
                }
                ReaderConstructionFailure::BeginError
                | ReaderConstructionFailure::BeginPanic
                | ReaderConstructionFailure::ControlError
                | ReaderConstructionFailure::ControlPanic => {
                    unreachable!("hostile child iterates only hostile-drop modes")
                }
            }

            assert_eq!(provider.hostile_record_drops.load(Ordering::Acquire), 1);
            let retained = resources.query_stats().allocated_bytes;
            assert!(retained >= READER_CONSTRUCTION_WORKSPACE);
            assert_eq!(
                provider
                    .hostile_authority_when_dropped
                    .load(Ordering::Acquire),
                retained
                    .checked_add(reader_operation_error_publication_bytes())
                    .expect("test hostile construction authority total is representable"),
                "provider state drops while its workspace and the independent publication block are both live"
            );
            assert!(
                !provider
                    .hostile_panic_payload_dropped
                    .load(Ordering::Acquire),
                "the cleanup backstop intentionally forgets the hostile heap panic payload"
            );
        }
        println!("{HANDSHAKE}");
        return;
    }

    let test_name = "execution::spill::framing_tests::reader_accounted_error_construction_hostile_open_record_drop_retains_authority";
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
        "hostile OpenSpillRecord drop replaced a primary or released leaked-payload authority\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

#[test]
fn provider_accounted_reader_rejects_a_nonempty_workspace_without_opening() {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let grant = resources.try_allocate(1).unwrap();

    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
        panic!("nonempty workspace must be rejected");
    };
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(error.granted_bytes(), 1);
    assert!(error.to_string().contains("dedicated zero-sized grant"));
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn provider_accounted_reader_rejects_unpublished_file_with_inline_primary() {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let file = manager.create_file(SpillFileRole::SortRun).unwrap();
    let grant = resources.try_allocate(0).unwrap();

    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
        panic!("unpublished spill file must be rejected");
    };

    assert_eq!(
        error.unpublished_static_message(),
        Some("spill file is not finished/published"),
        "unpublished preflight must return the inline typed primary"
    );
    assert_eq!(error.granted_bytes(), 0);
    assert_eq!(error.to_string(), "spill file is not finished/published");
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn provider_accounted_reader_keeps_malformed_file_start_as_typed_core_failure() {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    bytes[SPILL_RECORD_HEADER_BYTES] = 0xff;
    rewrite_record_crc(&mut bytes, 0);
    std::fs::write(file.path(), bytes).unwrap();

    let Err(error) = file.reader_with_owned_provider_admission(resources.try_allocate(0).unwrap())
    else {
        panic!("unknown FileStart role must fail construction");
    };
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(
        error.core_error(),
        Some(super::file::QualifiedFrameCoreError::UnknownRole(0xff))
    );
    assert!(std::error::Error::source(&error).is_none());
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn reader_accounted_error_construction_preserves_typed_admission_denial() {
    let directory = TempDir::new().unwrap();
    let resources = crate::execution::QueryResourceContext::new(exact_buffer_manager(1)).unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    let grant = resources.try_allocate(0).unwrap();

    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
        panic!("reader workspace must exceed the one-byte budget");
    };
    assert!(matches!(
        error.memory_error(),
        Some(grafeo_common::memory::buffer::MemoryGrantError::LimitExceeded { .. })
    ));
    assert_eq!(error.granted_bytes(), 0);
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn compatibility_only_provider_is_rejected_preconsume_without_losing_compatibility_reader() {
    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let begin_calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(CompatibilityOnlyProvider {
        begin_calls: Arc::clone(&begin_calls),
    });
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .provider(provider, SpillFrameLimits::format_max())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 0).unwrap();
    file.finish_write().unwrap();
    assert_eq!(begin_calls.load(Ordering::Acquire), 1);

    let grant = resources.try_allocate(0).unwrap();
    let Err(error) = file.reader_with_owned_provider_admission(grant) else {
        panic!("provider without exact output must fail before reader consumption");
    };
    assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    assert!(error.is_missing_exact_provider_output());
    assert_eq!(begin_calls.load(Ordering::Acquire), 1);
    drop(error);

    let mut compatibility = file
        .reader()
        .expect("the existing compatibility reader remains supported");
    assert_eq!(compatibility.read_sort_run_start().unwrap(), (1, 0));
    compatibility.finish().unwrap();
    drop(compatibility);
    assert_eq!(begin_calls.load(Ordering::Acquire), 2);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn provider_accounted_sort_row_uses_exact_plaintext_storage_and_exact_final_authority() {
    const PAYLOAD_LEN: usize = 4_097;

    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    let payload = vec![0x5a; PAYLOAD_LEN];
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(&payload).unwrap();
    file.finish_write().unwrap();

    let reader_grant = resources.try_allocate(0).unwrap();
    let reader = file
        .reader_with_owned_provider_admission(reader_grant)
        .expect("qualified reader construction succeeds");
    let reader = reader
        .read_sort_run_start_owned(1, 1)
        .expect("qualified sort declaration succeeds");
    let reader_bytes = reader.receipt().bytes();
    let row_grant = resources.try_allocate(0).unwrap();
    let (reader, row) = reader
        .read_sort_row_owned(row_grant)
        .expect("qualified sort row succeeds");

    assert_eq!(row.payload_for_test(), payload.as_slice());
    assert_eq!(row.payload_capacity_for_test(), PAYLOAD_LEN);
    assert_eq!(
        row.granted_bytes_for_test(),
        std::alloc::Layout::array::<u8>(PAYLOAD_LEN).unwrap().size(),
        "transient stored/provider workspace must be released before row handoff"
    );
    assert_eq!(
        resources.query_stats().allocated_bytes,
        reader_bytes + row.granted_bytes_for_test()
    );

    reader.finish_sort_run_owned().unwrap();
    assert_eq!(
        resources.query_stats().allocated_bytes,
        row.granted_bytes_for_test(),
        "row and its child authority outlive the reader"
    );
    drop(row);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}

#[test]
fn reader_accounted_error_preserves_typed_core_failure_without_compatibility_conversion() {
    const FILE_START_FRAME_BYTES: usize = SPILL_RECORD_HEADER_BYTES + 8;
    const SORT_START_FRAME_BYTES: usize = SPILL_RECORD_HEADER_BYTES + 12;
    const ROW_OFFSET: usize = FILE_START_FRAME_BYTES + SORT_START_FRAME_BYTES;

    let directory = TempDir::new().unwrap();
    let resources =
        crate::execution::QueryResourceContext::new(exact_buffer_manager(1 << 20)).unwrap();
    let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
        .build()
        .unwrap();
    let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
    file.write_sort_run_start(1, 1).unwrap();
    file.write_sort_row(b"typed-core-row").unwrap();
    file.finish_write().unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    bytes[ROW_OFFSET + 6] = SpillRecordKind::PartitionEntry as u8;
    rewrite_record_crc(&mut bytes, ROW_OFFSET);
    std::fs::write(file.path(), bytes).unwrap();

    let reader = file
        .reader_with_owned_provider_admission(resources.try_allocate(0).unwrap())
        .unwrap()
        .read_sort_run_start_owned(1, 1)
        .unwrap();
    let Err(error) = reader.read_sort_row_owned(resources.try_allocate(0).unwrap()) else {
        panic!("mutated row kind must fail in the qualified parser");
    };
    assert_eq!(
        error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
            source.kind()
        }),
        Some(io::ErrorKind::InvalidData)
    );
    assert_eq!(
        error.inspect::<super::file::ProviderAccountedReaderOperationError, _>(|source| {
            source.core_error()
        }),
        Some(Some(
            super::file::QualifiedFrameCoreError::UnexpectedRecordKind {
                expected: SpillRecordKind::SortRow,
                actual: SpillRecordKind::PartitionEntry,
            }
        ))
    );
    file.close_and_delete()
        .expect("published core failure must already have quiesced its reader");
    drop(error);
    assert_eq!(resources.query_stats().allocated_bytes, 0);
}
