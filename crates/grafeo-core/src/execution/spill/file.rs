//! Versioned, framed spill-file storage.
//!
//! Spill files are query-temporary and have no legacy raw-stream decoder. Every
//! record carries the v1 `GRSP` header, a checksum, and a closed record kind.

use super::manager::SpillManagerState;
use crate::execution::value_codec::CodecLimits;
use allocator_api2::alloc::Global;
use allocator_api2::vec::Vec as ExactVec;
#[cfg(not(target_arch = "wasm32"))]
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt as _, OpenOptionsSyncExt as _};
use grafeo_common::memory::buffer::{
    AccountedError, AccountedErrorPublisher, AccountedErrorPublisherBuildError,
    AccountedErrorPublisherBuildFailure, MemoryGrant, MemoryGrantError,
};
use std::any::Any;
use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
std::thread_local! {
    static TEST_UNUSED_PUBLISHER_RELEASE_FAILURE:
        std::cell::RefCell<Option<MemoryGrantError>> = const { std::cell::RefCell::new(None) };
}

/// Scoped test-only refusal for the next unused publication-grant release.
///
/// This lives in `grafeo-core`, where spill tests are compiled with `cfg(test)`;
/// it does not expose a production allocator or accounting trapdoor.
#[cfg(test)]
#[must_use = "the guard must stay live until the injected release attempt"]
pub(crate) struct TestUnusedPublisherReleaseFailureGuard;

#[cfg(test)]
impl Drop for TestUnusedPublisherReleaseFailureGuard {
    fn drop(&mut self) {
        let _ =
            TEST_UNUSED_PUBLISHER_RELEASE_FAILURE.try_with(|failure| failure.borrow_mut().take());
    }
}

#[cfg(test)]
/// Arms one scoped failure without changing any production-visible API.
pub(crate) fn fail_next_unused_publisher_release_for_test(
    error: MemoryGrantError,
) -> TestUnusedPublisherReleaseFailureGuard {
    TEST_UNUSED_PUBLISHER_RELEASE_FAILURE.with(|failure| {
        assert!(
            failure.borrow_mut().replace(error).is_none(),
            "unused-publisher release-failure hook is already armed"
        );
    });
    TestUnusedPublisherReleaseFailureGuard
}

/// The v1 spill-file magic.
pub const SPILL_FILE_MAGIC: [u8; 4] = *b"GRSP";
/// The current spill-file format version.
pub const SPILL_FORMAT_VERSION: u16 = 1;
/// The value-codec version embedded in v1 `FileStart` records.
pub const SPILL_VALUE_CODEC_VERSION: u16 = 1;
/// Number of bytes in every v1 record header.
pub const SPILL_RECORD_HEADER_BYTES: usize = 36;
/// Maximum plaintext or stored payload length representable by spill format v1.
pub const MAX_SPILL_RECORD_BYTES: u64 = u32::MAX as u64;

/// Whether this target has the no-heap native capability opener required by
/// the hard-qualified owned-reader seam. Compatibility readers remain
/// available on every target supported by the spill feature.
pub(crate) const fn hard_qualified_reader_platform_supported() -> bool {
    cfg!(all(
        unix,
        not(target_os = "wasi"),
        not(target_arch = "wasm32")
    ))
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn hard_qualified_file_len(file: &File) -> Result<u64, QualifiedReaderFailure> {
    let stat = rustix::fs::fstat(file).map_err(|error| QualifiedReaderFailure::Io(error.into()))?;
    u64::try_from(stat.st_size).map_err(|_| {
        QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
            "spill file length is negative or unrepresentable",
        ))
    })
}

#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn hard_qualified_file_len(_file: &File) -> Result<u64, QualifiedReaderFailure> {
    Err(QualifiedReaderFailure::core(
        QualifiedFrameCoreError::Unsupported(
            "hard-qualified spill file length is not implemented on this platform",
        ),
    ))
}

const HEADER_AAD_BYTES: usize = 32;
const BUFFER_SIZE: usize = 64 * 1024;

pub(super) const fn qualified_partition_reader_buffer_requested_bytes() -> usize {
    512
}
const FLAG_SEALED: u8 = 0x01;
/// Largest plaintext payload emitted by any fixed spill-format control record.
pub const MAX_FIXED_CONTROL_PAYLOAD_BYTES: usize = 12;

pub(super) const fn qualified_writer_buffer_requested_bytes() -> usize {
    SPILL_RECORD_HEADER_BYTES
}

fn close_file(file: File) {
    #[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
    drop(file);
    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    let _ = file;
}

/// Closed observable I/O stages used for deterministic failure injection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpillIoOperation {
    /// Exclusive physical file creation.
    Create,
    /// Record header/checksum write.
    WriteHeader,
    /// Stored record payload write.
    WritePayload,
    /// Buffered-writer flush.
    Flush,
    /// File synchronization boundary.
    Sync,
    /// Opening a published file for reading.
    ReadOpen,
    /// Reading a record header.
    ReadHeader,
    /// Reading a stored record payload.
    ReadPayload,
    /// Explicit file deletion.
    Delete,
    /// After marker removal, before exact owned-query directory removal.
    RemoveQueryDirectory,
    /// Before restoring a marker after query-directory removal failure.
    RestoreOwnerMarker,
    /// After root enumeration, before opening an abandoned-leaf candidate.
    ScavengeOpen,
    /// After acquiring dead-leaf leases, before validating retained authority.
    ScavengeValidate,
    /// Before deleting one retained abandoned-leaf artifact.
    ScavengeDelete,
    /// Before admitting additional durable root quota.
    QuotaReserve,
    /// Before writing replacement quota metadata.
    QuotaWrite,
    /// Before synchronizing quota metadata or its parent directory.
    QuotaSync,
    /// Before publishing replacement quota metadata.
    QuotaPublish,
    /// Before releasing quota after confirmed deletion.
    QuotaRelease,
    /// After owned data unlink, before making deletion durable.
    QuotaDeleteSync,
}

/// Injectable record-I/O boundary.
///
/// Production uses [`NoopSpillIo`]. Tests and embedders can return a specific
/// I/O error at a closed stage without relying on permissions or full disks.
/// Qualified reader and sort hooks declare their respective workspace bounds:
///
/// ```
/// use grafeo_core::execution::spill::{NoopSpillIo, SpillIo};
/// let reader_bound: fn(&NoopSpillIo) -> Option<usize> =
///     <NoopSpillIo as SpillIo>::qualified_reader_hook_workspace_bound;
/// let sort_bound: fn(&NoopSpillIo) -> Option<usize> =
///     <NoopSpillIo as SpillIo>::qualified_sort_hook_workspace_bound;
/// assert_eq!(reader_bound(&NoopSpillIo), Some(0));
/// assert_eq!(sort_bound(&NoopSpillIo), Some(0));
/// ```
///
/// The former fixed-control hook is unavailable:
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::spill::{NoopSpillIo, SpillIo};
/// let _ = NoopSpillIo.fixed_control_read_workspace_bound();
/// ```
pub trait SpillIo: Send + Sync + 'static {
    /// Checks one imminent physical I/O operation.
    ///
    /// # Errors
    ///
    /// Returns the injected or embedding-specific I/O failure for this stage.
    /// Qualified reader calls are governed by
    /// [`Self::qualified_reader_hook_workspace_bound`], and qualified non-reader
    /// sort calls by [`Self::qualified_sort_hook_workspace_bound`], including
    /// their every-outcome shared-state rollback contracts.
    fn check(&self, operation: SpillIoOperation) -> std::io::Result<()>;

    /// Bounds all simultaneously live allocations from non-reader hooks during
    /// one qualified sort, including retained errors, cleanup with earlier
    /// errors still live, and their complete destruction peaks. This is a
    /// whole-sort allowance, not a per-callback allowance. Shared-state growth
    /// must roll back on every outcome. Reporting is allocation-free/no-unwind.
    /// Returning `None` rejects typed pull-sort spill before destructive work.
    fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
        None
    }

    /// Returns the worst allocation peak of any qualified reader hook.
    ///
    /// The owned qualified reader queries this method exactly once, before
    /// acquiring a reader lease or invoking [`SpillIoOperation::ReadOpen`].
    /// The returned peak covers `ReadOpen` and every `ReadHeader`/
    /// `ReadPayload` callback for both fixed control and data records. It must
    /// include transient allocations, the complete escaping returned-error or
    /// panic payload, and that payload's teardown peak. Newly retained heap in
    /// shared hook state must be rolled back on success, error, and unwind.
    /// The immutable scalar remains admitted for the complete reader lifetime
    /// and with every captured failure.
    ///
    /// Reporting the bound is itself a pre-admission, nonallocating/no-unwind
    /// trust seam. Returning `None` makes the owned qualified reader fail
    /// closed without consuming file or provider state.
    fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        None
    }
}

/// Production I/O hook that permits every operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopSpillIo;

impl SpillIo for NoopSpillIo {
    fn check(&self, _operation: SpillIoOperation) -> std::io::Result<()> {
        Ok(())
    }

    fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
        Some(0)
    }

    fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        Some(0)
    }
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}

fn allocation_error(description: &str, error: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(format!("failed to reserve {description}: {error}"))
}

/// Fixed-capacity buffered writer whose heap reservation is fallible and
/// observable before it is installed in a spill handle.
///
/// `std::io::BufWriter::with_capacity` allocates infallibly and does not accept
/// caller-prepared backing storage. Spill construction needs both properties
/// so query-memory admission can precede the physical allocation.
/// Deliberately, this type does not flush from `Drop`: failed or unwinding
/// spill construction must discard poisoned staging rather than perform
/// unobservable best-effort writes.
struct FallibleBufWriter<W: Write> {
    inner: Option<W>,
    buffer: Vec<u8>,
}

/// Fallibly prepared writable-file backing that can be admitted before any
/// spill identity or filesystem side effect.
pub(super) struct SpillWriterBuffer {
    bytes: Vec<u8>,
}

impl SpillWriterBuffer {
    pub(super) fn prepare() -> std::io::Result<Self> {
        Self::with_capacity(BUFFER_SIZE)
    }

    fn with_capacity(capacity: usize) -> std::io::Result<Self> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_error| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
        Ok(Self { bytes })
    }

    pub(super) fn prepare_with_capacity(capacity: usize) -> std::io::Result<Self> {
        Self::with_capacity(capacity)
    }

    pub(super) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    #[cfg(test)]
    pub(super) fn pointer(&self) -> *const u8 {
        self.bytes.as_ptr()
    }
}

/// Failure from the private finish boundary that runs after the physical
/// writer is gone but before the staged file becomes reader-visible.
pub(super) enum SpillFinishError<E> {
    Io(std::io::Error),
    BeforePublish(E),
}

/// Removes every prefix byte confirmed by the inner writer on all exits,
/// including an unwind from a hostile `Write` implementation.
struct WrittenPrefixGuard<'a> {
    buffer: &'a mut Vec<u8>,
    written: usize,
}

impl Drop for WrittenPrefixGuard<'_> {
    fn drop(&mut self) {
        if self.written == 0 {
            return;
        }
        let remaining = self.buffer.len() - self.written;
        self.buffer.copy_within(self.written.., 0);
        self.buffer.truncate(remaining);
    }
}

impl<W: Write> FallibleBufWriter<W> {
    #[cfg(test)]
    fn try_with_capacity(inner: W, capacity: usize) -> std::io::Result<Self> {
        let buffer = SpillWriterBuffer::with_capacity(capacity)?;
        Ok(Self::with_prepared_buffer(inner, buffer))
    }

    fn with_prepared_buffer(inner: W, buffer: SpillWriterBuffer) -> Self {
        Self {
            inner: Some(inner),
            buffer: buffer.bytes,
        }
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    #[cfg(test)]
    fn buffer_pointer(&self) -> *const u8 {
        self.buffer.as_ptr()
    }

    #[cfg(test)]
    fn get_ref(&self) -> &W {
        self.inner
            .as_ref()
            .expect("fallible writer inner value is present")
    }

    #[cfg(test)]
    fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    fn flush_buffer(&mut self) -> std::io::Result<()> {
        let Some(inner) = self.inner.as_mut() else {
            return Err(invalid_input("spill writer inner value was already taken"));
        };
        let mut guard = WrittenPrefixGuard {
            buffer: &mut self.buffer,
            written: 0,
        };
        while guard.written < guard.buffer.len() {
            match inner.write(&guard.buffer[guard.written..]) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "failed to flush the complete spill writer buffer",
                    ));
                }
                Ok(bytes) => guard.written += bytes,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        guard.buffer.clear();
        guard.written = 0;
        Ok(())
    }

    fn into_inner(mut self) -> std::io::Result<W> {
        self.flush_buffer()?;
        self.inner
            .take()
            .ok_or_else(|| invalid_input("spill writer inner value was already taken"))
    }
}

impl<W: Write> Write for FallibleBufWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let capacity = self.buffer.capacity();
        if self.buffer.is_empty() && bytes.len() >= capacity {
            return self
                .inner
                .as_mut()
                .ok_or_else(|| invalid_input("spill writer inner value was already taken"))?
                .write(bytes);
        }
        if bytes.len() > capacity.saturating_sub(self.buffer.len()) {
            self.flush_buffer()?;
        }
        if bytes.len() >= capacity {
            return self
                .inner
                .as_mut()
                .ok_or_else(|| invalid_input("spill writer inner value was already taken"))?
                .write(bytes);
        }
        self.buffer.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_buffer()?;
        self.inner
            .as_mut()
            .ok_or_else(|| invalid_input("spill writer inner value was already taken"))?
            .flush()
    }
}

/// Explicit limits for one framed spill record.
///
/// Compatibility constructors use [`format_max`](Self::format_max). Qualified
/// callers combine explicit frame limits with query grants; callers may select
/// a smaller bound, and payloads are never truncated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillFrameLimits {
    max_plaintext_bytes: u64,
    max_stored_bytes: u64,
    codec_limits: CodecLimits,
}

impl SpillFrameLimits {
    /// Creates an explicit bounded-record policy.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` when either limit exceeds the v1 format maximum.
    pub fn new(max_plaintext_bytes: usize, max_stored_bytes: usize) -> std::io::Result<Self> {
        let plaintext = u64::try_from(max_plaintext_bytes)
            .map_err(|_| invalid_input("plaintext record limit exceeds u64"))?;
        let stored = u64::try_from(max_stored_bytes)
            .map_err(|_| invalid_input("stored record limit exceeds u64"))?;
        if plaintext > MAX_SPILL_RECORD_BYTES || stored > MAX_SPILL_RECORD_BYTES {
            return Err(invalid_input(format!(
                "spill record limit exceeds v1 maximum {MAX_SPILL_RECORD_BYTES}"
            )));
        }
        Ok(Self {
            max_plaintext_bytes: plaintext,
            max_stored_bytes: stored,
            codec_limits: CodecLimits::format_max(),
        })
    }

    /// Returns the checked v1 format maximum without an implicit small ceiling.
    #[must_use]
    pub const fn format_max() -> Self {
        Self {
            max_plaintext_bytes: MAX_SPILL_RECORD_BYTES,
            max_stored_bytes: MAX_SPILL_RECORD_BYTES,
            codec_limits: CodecLimits::format_max(),
        }
    }

    /// Returns this frame policy with an explicit decoded-resident codec grant.
    #[must_use]
    pub const fn with_codec_limits(mut self, codec_limits: CodecLimits) -> Self {
        self.codec_limits = codec_limits;
        self
    }

    /// Maximum plaintext bytes in one record.
    #[must_use]
    pub const fn max_plaintext_bytes(self) -> u64 {
        self.max_plaintext_bytes
    }

    /// Maximum stored bytes in one record.
    #[must_use]
    pub const fn max_stored_bytes(self) -> u64 {
        self.max_stored_bytes
    }

    /// Returns the independent decoded-resident codec policy.
    #[must_use]
    pub const fn codec_limits(self) -> CodecLimits {
        self.codec_limits
    }
}

impl Default for SpillFrameLimits {
    fn default() -> Self {
        Self::format_max()
    }
}

/// Closed role of a spill file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SpillFileRole {
    /// A native external-sort run.
    SortRun = 1,
    /// A native aggregate partition.
    NativePartition = 2,
    /// An RDF aggregate-state file reserved for Task 7.
    RdfAggregateState = 3,
}

impl TryFrom<u8> for SpillFileRole {
    type Error = std::io::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::SortRun),
            2 => Ok(Self::NativePartition),
            3 => Ok(Self::RdfAggregateState),
            _ => Err(invalid_data(format!(
                "unknown spill file role {value:#04x}"
            ))),
        }
    }
}

impl SpillFileRole {
    pub(crate) const fn file_prefix(self) -> &'static str {
        match self {
            Self::SortRun => "sort-run",
            Self::NativePartition => "native-partition",
            Self::RdfAggregateState => "rdf-aggregate",
        }
    }
}

/// Closed v1 record kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SpillRecordKind {
    /// File role and payload-codec declaration.
    FileStart = 0x01,
    /// Sort column and row counts.
    SortRunStart = 0x10,
    /// One Task-1 row-codec payload.
    SortRow = 0x11,
    /// Native partition entry count.
    PartitionStart = 0x20,
    /// One native partition entry.
    PartitionEntry = 0x21,
    /// Reserved RDF resumable aggregate state.
    AggregateState = 0x30,
    /// Terminal record and preceding-record count.
    FileEnd = 0x7f,
}

impl TryFrom<u8> for SpillRecordKind {
    type Error = std::io::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(Self::FileStart),
            0x10 => Ok(Self::SortRunStart),
            0x11 => Ok(Self::SortRow),
            0x20 => Ok(Self::PartitionStart),
            0x21 => Ok(Self::PartitionEntry),
            0x30 => Ok(Self::AggregateState),
            0x7f => Ok(Self::FileEnd),
            _ => Err(invalid_data(format!(
                "unknown spill record kind {value:#04x}"
            ))),
        }
    }
}

/// Opaque, immutable identity of one physical spill file.
///
/// This value is not key material. Authenticated engine providers use it as
/// public domain-separation input while database keys remain engine-owned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpillFileIdentity([u8; 16]);

impl SpillFileIdentity {
    /// Returns the public identity bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    #[cfg(test)]
    pub(crate) const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub(crate) fn random() -> std::io::Result<Self> {
        let mut bytes = [0u8; 16];
        getrandom::fill(&mut bytes)
            .map_err(|error| std::io::Error::other(format!("spill identity entropy: {error}")))?;
        Ok(Self(bytes))
    }

    pub(crate) fn hex(self) -> String {
        let mut result = String::with_capacity(32);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(result, "{byte:02x}");
        }
        result
    }
}

/// Opaque, immutable identity of one exclusive spill query leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpillQueryIdentity([u8; 16]);

impl SpillQueryIdentity {
    /// Returns the public query identity bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    #[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
    pub(crate) fn random() -> std::io::Result<Self> {
        let mut bytes = [0u8; 16];
        getrandom::fill(&mut bytes)
            .map_err(|error| std::io::Error::other(format!("spill query entropy: {error}")))?;
        Ok(Self(bytes))
    }

    pub(crate) const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    #[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
    pub(crate) fn hex(self) -> String {
        let mut result = String::with_capacity(32);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(result, "{byte:02x}");
        }
        result
    }
}

/// Canonical metadata supplied to a spill-record provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillRecordMeta {
    identity: SpillFileIdentity,
    role: SpillFileRole,
    kind: SpillRecordKind,
    sequence: u64,
    sealed: bool,
    plaintext_len: u64,
    stored_len: u64,
}

impl SpillRecordMeta {
    /// Physical file identity.
    #[must_use]
    pub const fn identity(self) -> SpillFileIdentity {
        self.identity
    }

    /// Declared file role.
    #[must_use]
    pub const fn role(self) -> SpillFileRole {
        self.role
    }

    /// Record kind.
    #[must_use]
    pub const fn kind(self) -> SpillRecordKind {
        self.kind
    }

    /// Contiguous record sequence.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    /// Whether the stored payload is sealed.
    #[must_use]
    pub const fn is_sealed(self) -> bool {
        self.sealed
    }

    /// Declared plaintext length.
    #[must_use]
    pub const fn plaintext_len(self) -> u64 {
        self.plaintext_len
    }

    /// Declared stored length.
    #[must_use]
    pub const fn stored_len(self) -> u64 {
        self.stored_len
    }
}

/// Engine-implementable factory for one file's record sealer/opener.
///
/// Core never receives database key bytes. Provider failure is terminal for
/// the attempted spill operation; there is no cleartext or resident retry.
pub trait SpillRecordProvider: Send + Sync + 'static {
    /// Reports whether this provider emits sealed records.
    fn seals(&self) -> bool;

    /// Starts provider state for one immutable physical file identity.
    ///
    /// # Errors
    ///
    /// Returns an error when per-file provider state cannot be initialized.
    /// Qualified construction permits retained heap only through the returned
    /// `OpenSpillRecord` or an escaping failure payload; all other new shared
    /// provider heap must obey the every-outcome rollback contract documented
    /// by [`Self::file_workspace_allocation_bound`].
    fn begin_file(&self, identity: SpillFileIdentity) -> std::io::Result<Box<dyn OpenSpillRecord>>;

    /// Returns a trusted file-lifetime workspace bound for qualified I/O.
    ///
    /// The bound includes the complete transient initialization peak of
    /// [`Self::begin_file`], all heap exclusively owned by the returned
    /// [`OpenSpillRecord`], that record's complete destruction peak, and the
    /// maximum additional provider allocation peak, including allocator
    /// capacity and every transient, while sealing or opening any fixed core
    /// control record (`FileStart`, role start, or `FileEnd`). It also includes
    /// construction and complete destruction of every original returned-error
    /// or panic payload permitted by those callbacks, plus any secondary
    /// teardown/error/panic payload allocation peak produced during provider,
    /// record, or original-failure destruction.
    ///
    /// Heap may remain after a callback only when exclusively owned by the
    /// returned `OpenSpillRecord` or by an escaping error/panic payload. On
    /// **every** success, error, and unwind outcome, newly retained heap in
    /// shared provider state outside those owned carriers must be rolled back
    /// before return or before the unwind leaves the callback. Core cannot
    /// inspect arbitrary safe extension state (for example an `Arc<Mutex<_>>`
    /// retained by an implementation), so this is a trusted extension
    /// contract, not mechanical containment of a malicious provider.
    ///
    /// Qualified callers retain the workspace for the entire live writer or
    /// reader and fail closed when the provider does not declare it. Overflow
    /// or an observed returned capacity above a declared bound is a
    /// qualified-path error. Reporting the bound itself, including unwinding
    /// and destroying any panic payload, must not allocate or retain shared heap
    /// because it necessarily precedes admission.
    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        None
    }

    /// Whether this provider supports the core-owned exact plaintext target
    /// required by the hard-qualified reader.
    ///
    /// Reporting is an allocation-free/no-unwind preflight contract. `false`
    /// affects only the additive hard-qualified seam; compatibility readers
    /// continue to call [`OpenSpillRecord::open`]. Providers opt in only after
    /// supplying an exact caller-owned output adapter.
    #[doc(hidden)]
    fn supports_qualified_exact_open(&self) -> bool {
        false
    }
}

/// Per-file spill-record seal/open state.
///
/// On qualified paths, heap retained directly by this exclusively owned value
/// and its method failure payloads is covered by
/// [`SpillRecordProvider::file_workspace_allocation_bound`] together with any
/// per-record bound. Heap stashed in separately shared extension state has no
/// ownership edge to those grants and must be rolled back before every method
/// returns or unwinds. The file-workspace bound also covers this value's full
/// destructor peak, including secondary teardown/panic payload construction.
pub trait OpenSpillRecord: Send {
    /// Returns the exact stored length for a plaintext length.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider cannot represent the record length.
    /// A successful query must be allocation-free. An original returned-error
    /// or panic payload and its complete destruction, including any secondary
    /// teardown or panic payload peak, may allocate only within the already-live
    /// file workspace declared by
    /// [`SpillRecordProvider::file_workspace_allocation_bound`]. Newly retained
    /// shared heap must be rolled back before every return or unwind.
    fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize>;

    /// Returns a trusted additional allocation-peak bound for [`Self::seal`].
    ///
    /// The bound includes transient allocations and returned vector capacity;
    /// persistent per-file state is covered by
    /// [`SpillRecordProvider::file_workspace_allocation_bound`]. Returning
    /// `None`, overflowing composition, or returning a vector whose observed
    /// capacity exceeds the bound fails the qualified operation. Successful
    /// reporting (including `None`) must be allocation-free. A panic payload
    /// and its complete destruction may allocate only within the already-live
    /// file workspace, and newly retained shared heap must be rolled back before
    /// every return or unwind.
    fn seal_allocation_bound(&self, _plaintext_len: usize) -> Option<usize> {
        None
    }

    /// Returns a trusted additional allocation-peak bound for [`Self::open`].
    ///
    /// The bound includes transient allocations and returned vector capacity;
    /// persistent per-file state is covered by
    /// [`SpillRecordProvider::file_workspace_allocation_bound`]. Returning
    /// `None`, overflowing composition, or returning a vector whose observed
    /// capacity exceeds the bound fails the qualified operation. Successful
    /// reporting (including `None`) must be allocation-free. A panic payload
    /// and its complete destruction may allocate only within the already-live
    /// file workspace, and newly retained shared heap must be rolled back before
    /// every return or unwind.
    fn open_allocation_bound(&self, _stored_len: usize) -> Option<usize> {
        None
    }

    /// Seals one record using the exact canonical header bytes `0..32` as AAD.
    ///
    /// # Errors
    ///
    /// Returns an error when sealing or provider allocation fails.
    fn seal(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; HEADER_AAD_BYTES],
        plaintext: &[u8],
    ) -> std::io::Result<Vec<u8>>;

    /// Opens one record after core has verified its checksum.
    ///
    /// # Errors
    ///
    /// Returns an error when authentication, opening, or allocation fails.
    fn open(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; HEADER_AAD_BYTES],
        stored: &[u8],
    ) -> std::io::Result<Vec<u8>>;

    /// Opens one record into core-owned, exactly sized plaintext storage.
    ///
    /// The hard-qualified reader allocates and admits the destination before
    /// invoking this callback, so an implementation cannot substitute a
    /// `Vec` with allocator-dependent spare capacity. `Some(Ok(()))` promises
    /// that every output byte has been written and authenticated. `None`
    /// declares that this provider supports only the compatibility reader and
    /// makes the hard-qualified path fail closed.
    ///
    /// Any implementation-owned transient, returned error, or panic payload
    /// remains governed by [`Self::open_allocation_bound`] and the containing
    /// provider's file-workspace contract. Shared-state growth must be rolled
    /// back on success, error, and unwind before the callback returns.
    #[doc(hidden)]
    fn open_qualified_into(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; HEADER_AAD_BYTES],
        _stored: &[u8],
        _plaintext: &mut [u8],
    ) -> Option<std::io::Result<()>> {
        None
    }
}

/// Cleartext identity provider used when database spill encryption is disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct CleartextSpillRecordProvider;

struct CleartextOpenRecord;

fn copy_record_bytes(bytes: &[u8], description: &str) -> std::io::Result<Vec<u8>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(bytes.len())
        .map_err(|error| allocation_error(description, error))?;
    copy.extend_from_slice(bytes);
    Ok(copy)
}

impl SpillRecordProvider for CleartextSpillRecordProvider {
    fn seals(&self) -> bool {
        false
    }

    fn begin_file(
        &self,
        _identity: SpillFileIdentity,
    ) -> std::io::Result<Box<dyn OpenSpillRecord>> {
        Ok(Box::new(CleartextOpenRecord))
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        // Cleartext has no retained state. The historical borrowed reader and
        // writer retain their conservative two-copy control envelope. The
        // hard-qualified reader never relies on a std `Vec` capacity ratio:
        // `open_qualified_into` writes into core's separately admitted exact
        // plaintext target.
        Some(MAX_FIXED_CONTROL_PAYLOAD_BYTES * 2)
    }

    fn supports_qualified_exact_open(&self) -> bool {
        true
    }
}

impl OpenSpillRecord for CleartextOpenRecord {
    fn stored_len(&self, plaintext_len: usize) -> std::io::Result<usize> {
        Ok(plaintext_len)
    }

    fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
        plaintext_len.checked_mul(2)
    }

    fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
        stored_len.checked_mul(2)
    }

    fn seal(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; HEADER_AAD_BYTES],
        plaintext: &[u8],
    ) -> std::io::Result<Vec<u8>> {
        copy_record_bytes(plaintext, "cleartext sealed record")
    }

    fn open(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; HEADER_AAD_BYTES],
        stored: &[u8],
    ) -> std::io::Result<Vec<u8>> {
        copy_record_bytes(stored, "cleartext opened record")
    }

    fn open_qualified_into(
        &mut self,
        _meta: &SpillRecordMeta,
        _aad: &[u8; HEADER_AAD_BYTES],
        stored: &[u8],
        plaintext: &mut [u8],
    ) -> Option<std::io::Result<()>> {
        if stored.len() != plaintext.len() {
            return Some(Err(std::io::Error::from(std::io::ErrorKind::InvalidData)));
        }
        plaintext.copy_from_slice(stored);
        Some(Ok(()))
    }
}

fn qualified_control_seal_bound(open_record: &dyn OpenSpillRecord) -> std::io::Result<usize> {
    [8usize, MAX_FIXED_CONTROL_PAYLOAD_BYTES]
        .into_iter()
        .try_fold(0usize, |maximum, plaintext_len| {
            let bound = open_record
                .seal_allocation_bound(plaintext_len)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "spill provider does not declare a qualified control-frame seal bound",
                    )
                })?;
            Ok(maximum.max(bound))
        })
}

fn qualified_control_open_bounds(
    open_record: &dyn OpenSpillRecord,
) -> std::io::Result<(usize, usize)> {
    // Compatibility-only borrowed admission retains its historical
    // conservative std-Vec envelope. The hard-owned path below instead uses
    // checked exact stored + plaintext `Layout`s and never calls this helper.
    [8usize, MAX_FIXED_CONTROL_PAYLOAD_BYTES]
        .into_iter()
        .try_fold(
            (0usize, 0usize),
            |(provider_max, core_max), plaintext_len| {
                let stored_len = open_record.stored_len(plaintext_len)?;
                let provider_bound = open_record.open_allocation_bound(stored_len).ok_or_else(
                    || {
                        std::io::Error::new(
                            std::io::ErrorKind::Unsupported,
                            "spill provider does not declare a qualified control-frame open bound",
                        )
                    },
                )?;
                let core_bound = stored_len.checked_mul(2).ok_or_else(|| {
                    invalid_input("qualified control-frame buffer bound overflow")
                })?;
                Ok((provider_max.max(provider_bound), core_max.max(core_bound)))
            },
        )
}

/// Allocation-free core diagnostic for the owned qualified reader path.
///
/// Every variant is inline: formatting is deferred until a caller explicitly
/// asks for text. Original filesystem, provider, and hook errors instead use
/// [`QualifiedReaderFailure::Io`] so their original identity and destruction
/// semantics are preserved opaquely. Hard owners intentionally expose no raw
/// `Error::source` chain because a provider error may contain cloneable heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QualifiedFrameCoreError {
    InvalidInput(&'static str),
    InvalidData(&'static str),
    Unsupported(&'static str),
    UnknownRole(u8),
    UnknownRecordKind(u8),
    UnsupportedCodecVersion(u16),
    UnsupportedFormatVersion(u16),
    UnexpectedRecordKind {
        expected: SpillRecordKind,
        actual: SpillRecordKind,
    },
    InvalidFlags(u8),
    SequenceMismatch {
        actual: u64,
        expected: u64,
    },
    FixedPlaintextLength {
        kind: SpillRecordKind,
        actual: u64,
        expected: u64,
    },
    ProviderStoredLength {
        provider: usize,
        declared: usize,
    },
    StoredBytesUnavailable {
        declared: u64,
        available: u64,
    },
    DeclaredRecordCount {
        declared: u64,
        observed: u64,
    },
    FileEndCount {
        declared: u64,
        preceding: u64,
    },
    IllegalTransition(SpillRecordKind),
}

/// Exact-layout plaintext produced only by the hard-qualified frame decoder.
///
/// The pinned `allocator-api2` `Global` vector is admitted from its exact
/// `Layout` before allocation. It never crosses a public compatibility API.
type QualifiedFramePayload = ExactVec<u8, Global>;

impl QualifiedFrameCoreError {
    #[cfg(test)]
    const fn kind(self) -> std::io::ErrorKind {
        match self {
            Self::InvalidInput(_) => std::io::ErrorKind::InvalidInput,
            Self::Unsupported(_) => std::io::ErrorKind::Unsupported,
            Self::StoredBytesUnavailable { .. } => std::io::ErrorKind::UnexpectedEof,
            Self::InvalidData(_)
            | Self::UnknownRole(_)
            | Self::UnknownRecordKind(_)
            | Self::UnsupportedCodecVersion(_)
            | Self::UnsupportedFormatVersion(_)
            | Self::UnexpectedRecordKind { .. }
            | Self::InvalidFlags(_)
            | Self::SequenceMismatch { .. }
            | Self::FixedPlaintextLength { .. }
            | Self::ProviderStoredLength { .. }
            | Self::DeclaredRecordCount { .. }
            | Self::FileEndCount { .. }
            | Self::IllegalTransition(_) => std::io::ErrorKind::InvalidData,
        }
    }
}

impl std::fmt::Display for QualifiedFrameCoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::InvalidData(message)
            | Self::Unsupported(message) => formatter.write_str(message),
            Self::UnknownRole(value) => write!(formatter, "unknown spill file role {value:#04x}"),
            Self::UnknownRecordKind(value) => {
                write!(formatter, "unknown spill record kind {value:#04x}")
            }
            Self::UnsupportedCodecVersion(version) => {
                write!(formatter, "unsupported spill value-codec version {version}")
            }
            Self::UnsupportedFormatVersion(version) => {
                write!(formatter, "unsupported spill format version {version}")
            }
            Self::UnexpectedRecordKind { expected, actual } => {
                write!(formatter, "expected {expected:?} record, found {actual:?}")
            }
            Self::InvalidFlags(flags) => {
                write!(formatter, "invalid spill record flags {flags:#04x}")
            }
            Self::SequenceMismatch { actual, expected } => write!(
                formatter,
                "spill sequence {actual} does not match expected {expected}"
            ),
            Self::FixedPlaintextLength {
                kind,
                actual,
                expected,
            } => write!(
                formatter,
                "{kind:?} plaintext length {actual} does not match {expected}"
            ),
            Self::ProviderStoredLength { provider, declared } => write!(
                formatter,
                "provider expects {provider} stored bytes, header declares {declared}"
            ),
            Self::StoredBytesUnavailable {
                declared,
                available,
            } => write!(
                formatter,
                "spill record declares {declared} stored bytes but only {available} remain"
            ),
            Self::DeclaredRecordCount { declared, observed } => write!(
                formatter,
                "declared {declared} records but observed {observed}"
            ),
            Self::FileEndCount {
                declared,
                preceding,
            } => write!(
                formatter,
                "FileEnd count {declared} does not match {preceding} preceding records"
            ),
            Self::IllegalTransition(kind) => {
                write!(formatter, "illegal {kind:?} transition for spill reader")
            }
        }
    }
}

impl std::error::Error for QualifiedFrameCoreError {}

enum QualifiedReaderFailure {
    Core(QualifiedFrameCoreError),
    Io(std::io::Error),
    ExactAllocation {
        context: &'static str,
        error: allocator_api2::collections::TryReserveError,
    },
}

impl std::fmt::Debug for QualifiedReaderFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(error) => formatter.debug_tuple("Core").field(error).finish(),
            Self::Io(error) => formatter
                .debug_struct("Io")
                .field("kind", &error.kind())
                .finish_non_exhaustive(),
            Self::ExactAllocation { context, error } => formatter
                .debug_struct("ExactAllocation")
                .field("context", context)
                .field("kind", &error.kind())
                .finish_non_exhaustive(),
        }
    }
}

impl QualifiedReaderFailure {
    const fn core(error: QualifiedFrameCoreError) -> Self {
        Self::Core(error)
    }

    #[cfg(test)]
    fn kind(&self) -> std::io::ErrorKind {
        match self {
            Self::Core(error) => error.kind(),
            Self::Io(error) => error.kind(),
            Self::ExactAllocation { .. } => std::io::ErrorKind::OutOfMemory,
        }
    }
}

impl From<std::io::Error> for QualifiedReaderFailure {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl std::fmt::Display for QualifiedReaderFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(error) => std::fmt::Display::fmt(error, formatter),
            Self::Io(error) => write!(
                formatter,
                "qualified reader I/O failure ({:?})",
                error.kind()
            ),
            Self::ExactAllocation { context, .. } => {
                write!(formatter, "failed to reserve {context}")
            }
        }
    }
}

// Deliberately no `source`: an original I/O/provider error may itself contain
// cloneable shared ownership. Exposing it through the safe `Error` chain would
// let a caller detach heap from the grant retained by the outer hard-qualified
// owner. Display and copy-only classification preserve diagnostics without
// weakening the ownership boundary.
impl std::error::Error for QualifiedReaderFailure {}

/// Copy/clone-only public-boundary classification for an owned reader
/// failure. The original provider/I/O payload never leaves its accounted
/// owner; only stable, non-detaching metadata crosses into operator errors.
#[derive(Clone, Debug)]
pub(crate) enum ProviderAccountedReaderFailureClassification {
    Memory(MemoryGrantError),
    Allocation,
    ExactAllocation(crate::execution::ResidentCapacityError),
    StorageFull,
    Execution,
}

fn classify_qualified_reader_failure(
    error: &QualifiedReaderFailure,
) -> ProviderAccountedReaderFailureClassification {
    match error {
        QualifiedReaderFailure::Core(_) => ProviderAccountedReaderFailureClassification::Execution,
        QualifiedReaderFailure::Io(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::QuotaExceeded | std::io::ErrorKind::StorageFull
            ) =>
        {
            ProviderAccountedReaderFailureClassification::StorageFull
        }
        QualifiedReaderFailure::Io(_) => ProviderAccountedReaderFailureClassification::Execution,
        QualifiedReaderFailure::ExactAllocation { context, error } => {
            ProviderAccountedReaderFailureClassification::ExactAllocation(
                crate::execution::ResidentCapacityError::exact_allocation(context, error.clone()),
            )
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RecordPhase {
    NeedFileStart,
    NeedSortRunStart,
    NeedPartitionStart,
    Sort { expected: u64, observed: u64 },
    Partition { expected: u64, observed: u64 },
    Aggregate { observed: u64 },
    Ended,
}

/// Physical identity retained from an exclusively opened spill file.
///
/// This is a substitution guard, not a capability-relative filesystem API.
#[derive(Clone, Debug)]
struct PhysicalFileIdentity {
    #[cfg(any(unix, windows, target_os = "wasi"))]
    device: u64,
    #[cfg(any(unix, windows, target_os = "wasi"))]
    inode: u64,
}

impl PhysicalFileIdentity {
    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    fn capture(file: &File, path: &Path) -> std::io::Result<Self> {
        reject_non_regular_or_symlink(path)?;
        let identity = Self::capture_handle(file)?;
        identity.validate_path(path)?;
        Ok(identity)
    }

    fn capture_handle(file: &File) -> std::io::Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(invalid_input("created spill handle is not a regular file"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(Self {
                device: metadata.dev(),
                inode: metadata.ino(),
            })
        }
        #[cfg(target_os = "wasi")]
        {
            let stat = rustix::fs::fstat(file)?;
            Ok(Self {
                device: stat.st_dev,
                inode: stat.st_ino,
            })
        }
        #[cfg(windows)]
        {
            use cap_fs_ext::MetadataExt as _;
            let metadata = cap_std::fs::Metadata::from_file(file)?;
            Ok(Self {
                device: metadata.dev(),
                inode: metadata.ino(),
            })
        }
        #[cfg(not(any(unix, windows, target_os = "wasi")))]
        {
            let _ = file;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "spill file identity is unsupported on this platform",
            ))
        }
    }

    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    fn validate_path(&self, path: &Path) -> std::io::Result<()> {
        reject_non_regular_or_symlink(path)?;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "spill file identity is unsupported on this wasm platform",
        ))
    }

    fn validate_file(&self, file: &File) -> std::io::Result<()> {
        #[cfg(target_os = "wasi")]
        {
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                return Err(invalid_data("spill handle is not a regular file"));
            }
            let stat = rustix::fs::fstat(file)?;
            if u64::try_from(stat.st_dev).ok() != Some(self.device)
                || u64::try_from(stat.st_ino).ok() != Some(self.inode)
            {
                return Err(invalid_data("spill file path was replaced"));
            }
            Ok(())
        }
        #[cfg(windows)]
        {
            use cap_fs_ext::MetadataExt as _;
            let metadata = cap_std::fs::Metadata::from_file(file)?;
            if !metadata.is_file() {
                return Err(invalid_data("spill handle is not a regular file"));
            }
            if metadata.dev() != self.device || metadata.ino() != self.inode {
                return Err(invalid_data("spill file path was replaced"));
            }
            Ok(())
        }
        #[cfg(unix)]
        {
            self.validate_metadata(&file.metadata()?)
        }
        #[cfg(all(target_arch = "wasm32", not(any(unix, windows, target_os = "wasi"))))]
        {
            let _ = file;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "spill file identity is unsupported on this wasm platform",
            ))
        }
    }

    fn validate_file_qualified(&self, file: &File) -> Result<(), QualifiedReaderFailure> {
        #[cfg(all(unix, not(target_arch = "wasm32")))]
        {
            let stat = rustix::fs::fstat(file)
                .map_err(|error| QualifiedReaderFailure::Io(error.into()))?;
            if rustix::fs::FileType::from_raw_mode(stat.st_mode)
                != rustix::fs::FileType::RegularFile
            {
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::InvalidData("spill handle is not a regular file"),
                ));
            }
            // Compare losslessly across Unix dev_t widths/signs; negative
            // device values still fail identity validation.
            if i128::from(stat.st_dev) != i128::from(self.device) || stat.st_ino != self.inode {
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::InvalidData("spill file path was replaced"),
                ));
            }
            Ok(())
        }
        #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
        {
            let _ = file;
            Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::Unsupported(
                    "hard-qualified spill identity validation is not implemented on this platform",
                ),
            ))
        }
    }

    #[cfg(unix)]
    fn validate_metadata(&self, metadata: &std::fs::Metadata) -> std::io::Result<()> {
        if !metadata.is_file() {
            return Err(invalid_data("spill handle is not a regular file"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.dev() != self.device || metadata.ino() != self.inode {
                return Err(invalid_data("spill file path was replaced"));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct SpillFileLifecycle {
    physical_identity: PhysicalFileIdentity,
    state: parking_lot::Mutex<SpillLifecycleState>,
    #[cfg(not(target_arch = "wasm32"))]
    directory: parking_lot::Mutex<Option<Arc<cap_std::fs::Dir>>>,
    #[cfg(not(target_arch = "wasm32"))]
    file_name: PathBuf,
    #[cfg(target_os = "wasi")]
    wasi_directory: parking_lot::Mutex<Option<Arc<File>>>,
    #[cfg(target_os = "wasi")]
    wasi_file_name: PathBuf,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    emscripten_directory: parking_lot::Mutex<Option<Arc<File>>>,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    emscripten_file_name: PathBuf,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    root_reservation: Option<Arc<super::quota::RootReservation>>,
    // Last owner to drop: handles and readers retain this lifecycle after the
    // manager is gone, so their physical I/O must keep the leaf lease alive.
    _query_lease: Option<Arc<super::root::QueryLeaseAuthority>>,
}

#[derive(Debug, Default)]
struct SpillLifecycleState {
    active_handles: usize,
    active_readers: usize,
    deleting: bool,
    deleted: bool,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    unlinked: bool,
}

#[derive(Debug)]
pub(crate) struct SpillHandleLease {
    lifecycle: Arc<SpillFileLifecycle>,
}

impl SpillHandleLease {
    pub(crate) fn acquire(lifecycle: Arc<SpillFileLifecycle>) -> std::io::Result<Self> {
        lifecycle.begin_handle()?;
        Ok(Self { lifecycle })
    }
}

#[derive(Debug)]
struct SpillReaderLease {
    lifecycle: Arc<SpillFileLifecycle>,
}

impl SpillReaderLease {
    fn acquire(lifecycle: Arc<SpillFileLifecycle>) -> std::io::Result<Self> {
        lifecycle.begin_reader()?;
        Ok(Self { lifecycle })
    }

    fn acquire_qualified(
        lifecycle: Arc<SpillFileLifecycle>,
    ) -> Result<Self, QualifiedReaderFailure> {
        lifecycle.begin_reader_qualified()?;
        Ok(Self { lifecycle })
    }
}

impl Drop for SpillReaderLease {
    fn drop(&mut self) {
        self.lifecycle.end_reader();
    }
}

impl Drop for SpillHandleLease {
    fn drop(&mut self) {
        self.lifecycle.end_handle();
    }
}

impl SpillFileLifecycle {
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn with_root_reservation(
        mut self,
        reservation: Option<Arc<super::quota::RootReservation>>,
    ) -> Self {
        self.root_reservation = reservation;
        self
    }

    pub(crate) fn reserve_physical(
        &self,
        logical: u64,
        allocated: u64,
        publication: bool,
    ) -> std::io::Result<()> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        if let Some(reservation) = &self.root_reservation {
            return reservation.ensure_capacity(logical, allocated, publication);
        }
        let _ = (logical, allocated, publication);
        Ok(())
    }
    pub(crate) fn with_query_lease(
        mut self,
        lease: Option<Arc<super::root::QueryLeaseAuthority>>,
    ) -> Self {
        self._query_lease = lease;
        self
    }

    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    pub(crate) fn capture(file: &File, path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            physical_identity: PhysicalFileIdentity::capture(file, path)?,
            state: parking_lot::Mutex::new(SpillLifecycleState::default()),
            _query_lease: None,
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            root_reservation: None,
            #[cfg(not(target_arch = "wasm32"))]
            directory: parking_lot::Mutex::new(Some(Arc::new(cap_std::fs::Dir::open_ambient_dir(
                path.parent()
                    .ok_or_else(|| invalid_input("spill path has no parent"))?,
                cap_std::ambient_authority(),
            )?))),
            #[cfg(not(target_arch = "wasm32"))]
            file_name: PathBuf::from(
                path.file_name()
                    .ok_or_else(|| invalid_input("spill path has no file name"))?,
            ),
            #[cfg(target_os = "wasi")]
            wasi_directory: parking_lot::Mutex::new(None),
            #[cfg(target_os = "wasi")]
            wasi_file_name: PathBuf::new(),
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_directory: parking_lot::Mutex::new(None),
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_file_name: PathBuf::new(),
        })
    }

    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    pub(crate) fn capture_with_emscripten_directory(
        file: &File,
        directory: Arc<File>,
        file_name: PathBuf,
    ) -> std::io::Result<Self> {
        Ok(Self {
            physical_identity: PhysicalFileIdentity::capture_handle(file)?,
            state: parking_lot::Mutex::new(SpillLifecycleState::default()),
            _query_lease: None,
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            root_reservation: None,
            emscripten_directory: parking_lot::Mutex::new(Some(directory)),
            emscripten_file_name: file_name,
        })
    }

    #[cfg(target_os = "wasi")]
    pub(crate) fn capture_with_wasi_directory(
        file: &File,
        directory: Arc<File>,
        file_name: PathBuf,
    ) -> std::io::Result<Self> {
        Ok(Self {
            physical_identity: PhysicalFileIdentity::capture_handle(file)?,
            state: parking_lot::Mutex::new(SpillLifecycleState::default()),
            _query_lease: None,
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            root_reservation: None,
            wasi_directory: parking_lot::Mutex::new(Some(directory)),
            wasi_file_name: file_name,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn capture_with_directory(
        file: &File,
        directory: Arc<cap_std::fs::Dir>,
        file_name: PathBuf,
    ) -> std::io::Result<Self> {
        let physical_identity = PhysicalFileIdentity::capture_handle(file)?;
        Ok(Self {
            physical_identity,
            state: parking_lot::Mutex::new(SpillLifecycleState::default()),
            _query_lease: None,
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            root_reservation: None,
            directory: parking_lot::Mutex::new(Some(directory)),
            file_name,
        })
    }

    fn open_reader(&self, _path: &Path) -> std::io::Result<File> {
        #[cfg(not(target_arch = "wasm32"))]
        let file = {
            let directory = self.directory.lock();
            let directory = directory
                .as_ref()
                .ok_or_else(|| invalid_input("spill file capability was released"))?;
            let mut options = cap_std::fs::OpenOptions::new();
            options.read(true).follow(FollowSymlinks::No).nonblock(true);
            directory.open_with(&self.file_name, &options)?.into_std()
        };
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let file = {
            let directory = self.emscripten_directory.lock();
            let directory = directory
                .as_ref()
                .ok_or_else(|| invalid_input("spill file capability was released"))?;
            let fd = rustix::fs::openat(
                directory,
                &self.emscripten_file_name,
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::NONBLOCK,
                rustix::fs::Mode::empty(),
            )?;
            File::from(fd)
        };
        #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
        let file = { open_ambient_spill_file_nonblocking(_path)? };
        #[cfg(target_os = "wasi")]
        let file = {
            let directory = self.wasi_directory.lock();
            let directory = directory
                .as_ref()
                .ok_or_else(|| invalid_input("spill file capability was released"))?;
            let fd = rustix::fs::openat(
                directory,
                &self.wasi_file_name,
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::NONBLOCK,
                rustix::fs::Mode::empty(),
            )?;
            File::from(fd)
        };
        self.physical_identity.validate_file(&file)?;
        Ok(file)
    }

    /// Opens the already-qualified native capability without path-conversion
    /// heap. Non-Unix adapters must provide an equivalent proof before this
    /// new hard-qualified seam is enabled there.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    fn open_reader_qualified(&self) -> Result<File, QualifiedReaderFailure> {
        use std::os::unix::ffi::OsStrExt as _;

        const MAX_QUALIFIED_FILE_NAME_BYTES: usize = 255;
        let directory = self.directory.lock();
        let directory = directory.as_ref().ok_or_else(|| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidInput(
                "spill file capability was released",
            ))
        })?;
        let name = self.file_name.as_os_str().as_bytes();
        if name.is_empty()
            || name.len() > MAX_QUALIFIED_FILE_NAME_BYTES
            || name.contains(&0)
            || name.contains(&b'/')
            || name == b"."
            || name == b".."
        {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::Unsupported(
                    "spill file name is outside the native qualified-open envelope",
                ),
            ));
        }
        let mut nul_terminated = [0_u8; MAX_QUALIFIED_FILE_NAME_BYTES + 1];
        nul_terminated[..name.len()].copy_from_slice(name);
        let name =
            std::ffi::CStr::from_bytes_with_nul(&nul_terminated[..=name.len()]).map_err(|_| {
                QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidInput(
                    "spill file name is not a valid native path component",
                ))
            })?;
        let descriptor = rustix::fs::openat(
            directory,
            name,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK,
            rustix::fs::Mode::empty(),
        )
        .map_err(|error| QualifiedReaderFailure::Io(error.into()))?;
        let file = File::from(descriptor);
        self.physical_identity.validate_file_qualified(&file)?;
        Ok(file)
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    fn open_reader_qualified(&self) -> Result<File, QualifiedReaderFailure> {
        Err(QualifiedReaderFailure::core(
            QualifiedFrameCoreError::Unsupported(
                "hard-qualified spill reader open is not implemented on this platform",
            ),
        ))
    }

    pub(crate) fn validate_entry(&self, path: &Path) -> std::io::Result<()> {
        close_file(self.open_reader(path)?);
        Ok(())
    }

    pub(crate) fn delete_path(&self, _path: &Path, caller_has_handle: bool) -> std::io::Result<()> {
        let mut state = self
            .state
            .try_lock_for(std::time::Duration::from_millis(512))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "spill file cleanup is already active",
                )
            })?;
        let expected_handles = usize::from(caller_has_handle);
        if state.deleting || state.active_readers != 0 || state.active_handles != expected_handles {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "spill file has active handle or reader leases",
            ));
        }
        if state.deleted {
            return Ok(());
        }
        state.deleting = true;
        #[cfg(not(target_arch = "wasm32"))]
        let mut directory = self.directory.lock();
        #[cfg(not(target_arch = "wasm32"))]
        let remove_result: std::io::Result<()> = (|| {
            let directory = directory
                .as_ref()
                .ok_or_else(|| invalid_input("spill file capability was released"))?;
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            if state.unlinked
                && let Some(reservation) = &self.root_reservation
            {
                return reservation.release_deleted();
            }
            let mut options = cap_std::fs::OpenOptions::new();
            options.read(true).follow(FollowSymlinks::No).nonblock(true);
            let named_file = match directory.open_with(&self.file_name, &options) {
                Ok(file) => file.into_std(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    #[cfg(all(
                        any(target_os = "linux", target_os = "macos"),
                        not(target_arch = "wasm32")
                    ))]
                    if self.root_reservation.is_some() {
                        return Err(invalid_data(
                            "quota-owned spill file disappeared without deletion proof",
                        ));
                    }
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
            self.physical_identity.validate_file(&named_file)?;
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            if self.root_reservation.is_some() {
                use std::os::unix::fs::MetadataExt as _;
                let metadata = named_file.metadata()?;
                if metadata.nlink() != 1
                    || metadata.uid() != rustix::process::geteuid().as_raw()
                    || metadata.mode() & 0o077 != 0
                {
                    return Err(invalid_data(
                        "quota-owned spill file has aliases or unsafe permissions",
                    ));
                }
            }
            directory.remove_file(&self.file_name)?;
            close_file(named_file);
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            if let Some(reservation) = &self.root_reservation {
                state.unlinked = true;
                reservation.release_deleted()?;
            }
            Ok(())
        })();
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let mut emscripten_directory = self.emscripten_directory.lock();
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let remove_result: std::io::Result<()> = (|| {
            let directory = emscripten_directory
                .as_ref()
                .ok_or_else(|| invalid_input("spill file capability was released"))?;
            let fd = match rustix::fs::openat(
                directory,
                &self.emscripten_file_name,
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::NONBLOCK,
                rustix::fs::Mode::empty(),
            ) {
                Ok(fd) => fd,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(error.into()),
            };
            let named_file = File::from(fd);
            self.physical_identity.validate_file(&named_file)?;
            close_file(named_file);
            rustix::fs::unlinkat(
                directory,
                &self.emscripten_file_name,
                rustix::fs::AtFlags::empty(),
            )?;
            Ok(())
        })();
        #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
        let remove_result: std::io::Result<()> = (|| {
            let named_file = match open_ambient_spill_file_nonblocking(_path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(error),
            };
            self.physical_identity.validate_file(&named_file)?;
            close_file(named_file);
            std::fs::remove_file(_path)
        })();
        #[cfg(target_os = "wasi")]
        let mut wasi_directory = self.wasi_directory.lock();
        #[cfg(target_os = "wasi")]
        let remove_result: std::io::Result<()> = (|| {
            let directory = wasi_directory
                .as_ref()
                .ok_or_else(|| invalid_input("spill file capability was released"))?;
            let fd = match rustix::fs::openat(
                directory,
                &self.wasi_file_name,
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::NONBLOCK,
                rustix::fs::Mode::empty(),
            ) {
                Ok(fd) => fd,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(error.into()),
            };
            let named_file = File::from(fd);
            self.physical_identity.validate_file(&named_file)?;
            close_file(named_file);
            rustix::fs::unlinkat(
                directory,
                &self.wasi_file_name,
                rustix::fs::AtFlags::empty(),
            )?;
            Ok(())
        })();
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        let remove_result = remove_result.map_err(|error| {
            if self.root_reservation.is_some() && error.kind() == std::io::ErrorKind::NotFound {
                std::io::Error::new(std::io::ErrorKind::InvalidData, error)
            } else {
                error
            }
        });
        match remove_result {
            Ok(()) => {
                state.deleted = true;
                state.deleting = false;
                #[cfg(not(target_arch = "wasm32"))]
                directory.take();
                #[cfg(target_os = "wasi")]
                wasi_directory.take();
                #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
                emscripten_directory.take();
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                state.deleted = true;
                state.deleting = false;
                #[cfg(not(target_arch = "wasm32"))]
                directory.take();
                #[cfg(target_os = "wasi")]
                wasi_directory.take();
                #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
                emscripten_directory.take();
                Ok(())
            }
            Err(error) => {
                state.deleting = false;
                Err(error)
            }
        }
    }

    fn begin_handle(&self) -> std::io::Result<()> {
        let mut state = self.state.lock();
        if state.deleting || state.deleted {
            return Err(invalid_input("spill file lifecycle is closing"));
        }
        state.active_handles = state
            .active_handles
            .checked_add(1)
            .ok_or_else(|| invalid_input("spill handle lease count overflow"))?;
        Ok(())
    }

    fn end_handle(&self) {
        let mut state = self.state.lock();
        state.active_handles = state.active_handles.saturating_sub(1);
    }

    fn begin_reader(&self) -> std::io::Result<()> {
        let mut state = self.state.lock();
        if state.deleting || state.deleted {
            return Err(invalid_input("spill file lifecycle is closing"));
        }
        state.active_readers = state
            .active_readers
            .checked_add(1)
            .ok_or_else(|| invalid_input("spill reader lease count overflow"))?;
        Ok(())
    }

    fn begin_reader_qualified(&self) -> Result<(), QualifiedReaderFailure> {
        let mut state = self.state.lock();
        if state.deleting || state.deleted {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidInput("spill file lifecycle is closing"),
            ));
        }
        state.active_readers = state.active_readers.checked_add(1).ok_or_else(|| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidInput(
                "spill reader lease count overflow",
            ))
        })?;
        Ok(())
    }

    fn end_reader(&self) {
        let mut state = self.state.lock();
        state.active_readers = state.active_readers.saturating_sub(1);
    }
}

#[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
fn open_ambient_spill_file_nonblocking(_path: &Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "spill files are unsupported on this wasm platform",
    ))
}

#[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
fn reject_non_regular_or_symlink(path: &Path) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid_data("spill path is not the original regular file"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct ExactOwnedReaderQualificationBounds {
    provider_workspace_bytes: usize,
    hook_workspace_bytes: usize,
    initial_workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
enum ExactOwnedReaderQualificationFailure {
    UnsupportedPlatform,
    MissingExactProviderOutput,
    MissingProviderBound,
    MissingHookBound,
    BoundOverflow {
        provider_bytes: usize,
        hook_bytes: usize,
    },
}

#[derive(Clone, Copy, Debug)]
enum ExactOwnedReaderQualificationCache {
    Eligible(ExactOwnedReaderQualificationBounds),
    Ineligible(ExactOwnedReaderQualificationFailure),
}

/// Unforgeable, identity-bound proof that the provider and I/O hook opted into
/// the hard-qualified reader with representable immutable bounds. A fresh
/// move-only receipt can be minted from the cached private bounds, but the
/// trusted callbacks that established them execute exactly once per file.
#[derive(Debug)]
#[must_use = "reader qualification must be consumed by its exact spill file"]
pub(crate) struct ExactOwnedReaderQualification {
    identity: SpillFileIdentity,
    bounds: ExactOwnedReaderQualificationBounds,
}

/// Sealed-in-crate publication surface used only while an exact reader owns a
/// child grant whose size can change inside the framing layer.
pub(crate) trait ProviderGrantTransitionObserver {
    fn replace_reader(&self, previous: usize, current: usize) -> Result<(), MemoryGrantError>;
    fn replace_row(&self, previous: usize, current: usize) -> Result<(), MemoryGrantError>;
    fn publish_unrepresentable(&self);
}

/// Handle for one staged or published framed spill file.
///
/// Cleanup uses [`Self::close_and_delete`]; the consuming method is unavailable:
///
/// ```compile_fail,E0599
/// use grafeo_core::execution::spill::SpillFile;
/// let _ = SpillFile::delete;
/// ```
#[expect(
    clippy::struct_excessive_bools,
    reason = "publication, cleanup, poison, and provider-sealing are independent lifecycle facts"
)]
pub struct SpillFile {
    path: PathBuf,
    identity: SpillFileIdentity,
    role: SpillFileRole,
    limits: SpillFrameLimits,
    provider: Arc<dyn SpillRecordProvider>,
    qualified_control_provider_bound: Option<usize>,
    open_record: Option<parking_lot::Mutex<Box<dyn OpenSpillRecord>>>,
    sealed: bool,
    writer: Option<FallibleBufWriter<File>>,
    manager: Arc<SpillManagerState>,
    io: Arc<dyn SpillIo>,
    lifecycle: Arc<SpillFileLifecycle>,
    _handle_lease: SpillHandleLease,
    next_sequence: u64,
    phase: RecordPhase,
    bytes_written: u64,
    finished: bool,
    deleted: bool,
    poisoned: bool,
    exact_owned_reader_qualification: OnceLock<ExactOwnedReaderQualificationCache>,
}

impl SpillFile {
    #[expect(
        clippy::too_many_arguments,
        reason = "construction transfers one open file plus provider, lifecycle, and I/O capabilities"
    )]
    pub(super) fn from_created_file(
        path: PathBuf,
        file: File,
        identity: SpillFileIdentity,
        role: SpillFileRole,
        limits: SpillFrameLimits,
        provider: Arc<dyn SpillRecordProvider>,
        manager: Arc<SpillManagerState>,
        io: Arc<dyn SpillIo>,
        lifecycle: Arc<SpillFileLifecycle>,
        handle_lease: SpillHandleLease,
        writer_buffer: SpillWriterBuffer,
        qualified_file_workspace_bound: Option<usize>,
        owned_cleanup_complete: Option<&mut bool>,
    ) -> std::io::Result<Self> {
        let mut result = Self {
            path,
            identity,
            role,
            limits,
            provider,
            qualified_control_provider_bound: None,
            open_record: None,
            sealed: false,
            writer: None,
            manager,
            io,
            lifecycle,
            _handle_lease: handle_lease,
            next_sequence: 0,
            phase: RecordPhase::NeedFileStart,
            bytes_written: 0,
            finished: false,
            deleted: false,
            poisoned: false,
            exact_owned_reader_qualification: OnceLock::new(),
        };
        // Establish cleanup ownership before invoking provider code. The
        // backing was prepared before identity/filesystem creation and moves
        // here without another allocation. A returned provider error or unwind
        // therefore runs the SpillFile Drop backstop instead of stranding
        // registered staging.
        result.writer = Some(FallibleBufWriter::with_prepared_buffer(file, writer_buffer));
        let initialization = (|| {
            result.sealed = result.provider.seals();
            let open_record = result.provider.begin_file(identity)?;
            if let Some(file_workspace_bound) = qualified_file_workspace_bound {
                let control_bound = qualified_control_seal_bound(&*open_record)?;
                if control_bound > file_workspace_bound {
                    return Err(invalid_data(format!(
                        "spill provider control-frame seal bound {control_bound} exceeds its declared {file_workspace_bound}-byte file workspace"
                    )));
                }
                result.qualified_control_provider_bound = Some(control_bound);
            }
            result.open_record = Some(parking_lot::Mutex::new(open_record));
            let mut start = [0u8; 8];
            start[0] = role as u8;
            start[1..3].copy_from_slice(&SPILL_VALUE_CODEC_VERSION.to_le_bytes());
            result.write_frame(SpillRecordKind::FileStart, &start)?;
            result.phase = match role {
                SpillFileRole::SortRun => RecordPhase::NeedSortRunStart,
                SpillFileRole::NativePartition => RecordPhase::NeedPartitionStart,
                SpillFileRole::RdfAggregateState => RecordPhase::Aggregate { observed: 0 },
            };
            Ok::<_, std::io::Error>(())
        })();
        match initialization {
            Ok(()) => Ok(result),
            Err(primary) => {
                if let Some(complete) = owned_cleanup_complete {
                    *complete = result.retire_owned_file();
                    return Err(primary);
                }
                let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    result.close_and_delete()
                }));
                match cleanup {
                    Ok(Ok(())) => Err(primary),
                    Ok(Err(cleanup)) => Err(super::combine_primary_and_cleanup(
                        primary,
                        cleanup,
                        "spill construction cleanup",
                    )),
                    Err(panic) => {
                        // Preserve the returned initialization primary. Its
                        // opaque payload may have a hostile destructor, so a
                        // panicking cleanup hook cannot remain live or unwind
                        // across it. `result` Drop retries behind the ordinary
                        // cleanup backstop after the primary moves to the
                        // return value.
                        super::forget_cleanup_failure(panic);
                        super::manager::record_orphan_cleanup_failures(1);
                        Err(primary)
                    }
                }
            }
        }
    }

    /// Returns the physical path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns this file's public identity.
    #[must_use]
    pub const fn identity(&self) -> SpillFileIdentity {
        self.identity
    }

    /// Returns the closed file role.
    #[must_use]
    pub const fn role(&self) -> SpillFileRole {
        self.role
    }

    /// Returns the explicit record limits.
    #[must_use]
    pub const fn limits(&self) -> SpillFrameLimits {
        self.limits
    }

    /// Returns bytes written after successful complete-record writes.
    #[must_use]
    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Returns whether the file is still staging writes.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        self.writer.is_some()
    }

    #[cfg(test)]
    pub(super) fn writer_buffer_identity(&self) -> Option<(*const u8, usize)> {
        self.writer
            .as_ref()
            .map(|writer| (writer.buffer_pointer(), writer.capacity()))
    }

    /// Writes the required sort-run declaration.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong role/transition or failed framed write.
    pub fn write_sort_run_start(&mut self, columns: u32, rows: u64) -> std::io::Result<()> {
        if self.role != SpillFileRole::SortRun
            || !matches!(self.phase, RecordPhase::NeedSortRunStart)
        {
            return Err(invalid_input("illegal SortRunStart transition"));
        }
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&columns.to_le_bytes());
        payload[4..].copy_from_slice(&rows.to_le_bytes());
        self.write_frame(SpillRecordKind::SortRunStart, &payload)?;
        self.phase = RecordPhase::Sort {
            expected: rows,
            observed: 0,
        };
        Ok(())
    }

    /// Writes one complete Task-1 row-codec payload.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong transition/count or failed framed write.
    pub fn write_sort_row(&mut self, payload: &[u8]) -> std::io::Result<()> {
        self.write_sort_row_inner(payload, None)
    }

    /// Writes one sort row after a caller-owned grant admits every provider
    /// allocation peak. The callback receives an absolute workspace size and
    /// must keep that authority live until this call returns.
    pub(crate) fn write_sort_row_with_admission(
        &mut self,
        payload: &[u8],
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        self.write_sort_row_inner(payload, Some(&mut admit))
    }

    fn write_sort_row_inner(
        &mut self,
        payload: &[u8],
        admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
    ) -> std::io::Result<()> {
        let RecordPhase::Sort { expected, observed } = self.phase else {
            return Err(invalid_input("illegal SortRow transition"));
        };
        if observed >= expected {
            return Err(invalid_input("sort row count exceeds declaration"));
        }
        self.write_frame_inner(SpillRecordKind::SortRow, payload, admission, None)?;
        self.phase = RecordPhase::Sort {
            expected,
            observed: observed + 1,
        };
        Ok(())
    }

    /// Writes the required native-partition declaration.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong role/transition or failed framed write.
    pub fn write_partition_start(&mut self, entries: u64) -> std::io::Result<()> {
        if self.role != SpillFileRole::NativePartition
            || !matches!(self.phase, RecordPhase::NeedPartitionStart)
        {
            return Err(invalid_input("illegal PartitionStart transition"));
        }
        self.write_frame(SpillRecordKind::PartitionStart, &entries.to_le_bytes())?;
        self.phase = RecordPhase::Partition {
            expected: entries,
            observed: 0,
        };
        Ok(())
    }

    /// Writes one bounded native partition entry.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong transition/count or failed framed write.
    pub fn write_partition_entry(&mut self, payload: &[u8]) -> std::io::Result<()> {
        self.write_partition_entry_inner(payload, None)
    }

    pub(crate) fn write_partition_entry_with_admission(
        &mut self,
        payload: &[u8],
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        self.write_partition_entry_inner(payload, Some(&mut admit))
    }

    fn write_partition_entry_inner(
        &mut self,
        payload: &[u8],
        admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
    ) -> std::io::Result<()> {
        let RecordPhase::Partition { expected, observed } = self.phase else {
            return Err(invalid_input("illegal PartitionEntry transition"));
        };
        if observed >= expected {
            return Err(invalid_input("partition entry count exceeds declaration"));
        }
        self.write_frame_inner(SpillRecordKind::PartitionEntry, payload, admission, None)?;
        self.phase = RecordPhase::Partition {
            expected,
            observed: observed + 1,
        };
        Ok(())
    }

    /// Writes one reserved RDF aggregate-state record.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong role/transition or failed framed write.
    pub fn write_aggregate_state(&mut self, payload: &[u8]) -> std::io::Result<()> {
        self.write_aggregate_state_inner(payload, None)
    }

    pub(super) fn write_aggregate_state_with_admission(
        &mut self,
        payload: &[u8],
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        self.write_aggregate_state_inner(payload, Some(&mut admit))
    }

    fn write_aggregate_state_inner(
        &mut self,
        payload: &[u8],
        admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
    ) -> std::io::Result<()> {
        if self.role != SpillFileRole::RdfAggregateState {
            return Err(invalid_input("AggregateState in non-RDF spill file"));
        }
        let RecordPhase::Aggregate { observed } = self.phase else {
            return Err(invalid_input("illegal AggregateState transition"));
        };
        self.write_frame_inner(SpillRecordKind::AggregateState, payload, admission, None)?;
        self.phase = RecordPhase::Aggregate {
            observed: observed + 1,
        };
        Ok(())
    }

    fn write_frame(&mut self, kind: SpillRecordKind, plaintext: &[u8]) -> std::io::Result<()> {
        let pre_admitted_provider_bound =
            fixed_plaintext_len(kind).and(self.qualified_control_provider_bound);
        self.write_frame_inner(kind, plaintext, None, pre_admitted_provider_bound)
    }

    fn write_frame_inner(
        &mut self,
        kind: SpillRecordKind,
        plaintext: &[u8],
        admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
        pre_admitted_provider_bound: Option<usize>,
    ) -> std::io::Result<()> {
        if self.poisoned {
            return Err(invalid_input(
                "spill file is poisoned by a prior record failure",
            ));
        }
        let plaintext_len = u64::try_from(plaintext.len())
            .map_err(|_| invalid_input("plaintext record length exceeds u64"))?;
        if plaintext_len > MAX_SPILL_RECORD_BYTES || plaintext_len > self.limits.max_plaintext_bytes
        {
            return Err(invalid_input(format!(
                "plaintext record length {plaintext_len} exceeds maximum {}",
                self.limits.max_plaintext_bytes.min(MAX_SPILL_RECORD_BYTES)
            )));
        }

        self.poisoned = true;
        let result = self.write_frame_after_validation(
            kind,
            plaintext,
            plaintext_len,
            admission,
            pre_admitted_provider_bound,
        );
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn write_frame_after_validation(
        &mut self,
        kind: SpillRecordKind,
        plaintext: &[u8],
        plaintext_len: u64,
        mut admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
        pre_admitted_provider_bound: Option<usize>,
    ) -> std::io::Result<()> {
        let open_record = self
            .open_record
            .as_ref()
            .ok_or_else(|| invalid_input("spill record provider is closed"))?;
        let mut open_record = open_record.lock();
        let reported_stored = open_record.stored_len(plaintext.len())?;
        let stored_len = u64::try_from(reported_stored)
            .map_err(|_| invalid_input("provider stored length exceeds u64"))?;
        if !self.sealed && stored_len != plaintext_len {
            return Err(invalid_data(
                "unsealed spill provider changed the plaintext length",
            ));
        }
        if stored_len > MAX_SPILL_RECORD_BYTES || stored_len > self.limits.max_stored_bytes {
            return Err(invalid_input(format!(
                "stored record length {stored_len} exceeds maximum {}",
                self.limits.max_stored_bytes.min(MAX_SPILL_RECORD_BYTES)
            )));
        }
        if stored_len > isize::MAX as u64 {
            return Err(invalid_input(
                "provider stored length exceeds the platform allocation maximum",
            ));
        }
        let provider_bound = if let Some(admit) = admission.as_mut() {
            let bound = open_record
                .seal_allocation_bound(plaintext.len())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "spill provider does not declare a qualified seal-allocation bound",
                    )
                })?;
            admit(bound)?;
            Some(bound)
        } else {
            pre_admitted_provider_bound
        };

        let meta = SpillRecordMeta {
            identity: self.identity,
            role: self.role,
            kind,
            sequence: self.next_sequence,
            sealed: self.sealed,
            plaintext_len,
            stored_len,
        };
        let aad = encode_aad(meta);
        let stored = open_record.seal(&meta, &aad, plaintext)?;
        if let Some(bound) = provider_bound
            && stored.capacity() > bound
        {
            return Err(invalid_data(format!(
                "spill provider returned capacity {}, exceeding its declared {bound}-byte seal bound",
                stored.capacity()
            )));
        }
        if stored.len() != reported_stored {
            return Err(invalid_data(format!(
                "spill provider returned {} stored bytes, expected {reported_stored}",
                stored.len()
            )));
        }

        let mut checksum = crc32fast::Hasher::new();
        checksum.update(&aad);
        checksum.update(&stored);
        let checksum = checksum.finalize();
        let record_bytes = (SPILL_RECORD_HEADER_BYTES as u64)
            .checked_add(stored_len)
            .ok_or_else(|| invalid_input("spill record byte count overflow"))?;
        // This is the final, atomic quota boundary. From this point onward a
        // header or payload write can be partial, so the conservative charge
        // remains attached to the tracked file until confirmed deletion.
        self.manager.reserve_bytes(&self.path, record_bytes)?;
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| invalid_input("spill write phase ended"))?;
        self.io.check(SpillIoOperation::WriteHeader)?;
        writer.write_all(&aad)?;
        writer.write_all(&checksum.to_le_bytes())?;
        self.io.check(SpillIoOperation::WritePayload)?;
        writer.write_all(&stored)?;

        self.bytes_written = self
            .bytes_written
            .checked_add(record_bytes)
            .ok_or_else(|| invalid_input("spill file byte count overflow"))?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid_input("spill record sequence exhausted"))?;
        Ok(())
    }

    /// Writes `FileEnd`, flushes, synchronizes, closes the writer, and publishes
    /// the file to readers and byte accounting.
    ///
    /// # Errors
    ///
    /// Returns an error for incomplete/poisoned state, failed I/O, replacement,
    /// or publication-accounting failure.
    pub fn finish_write(&mut self) -> std::io::Result<()> {
        if self.finished {
            return Ok(());
        }
        match self.finish_write_before_publish(|| Ok::<_, std::convert::Infallible>(())) {
            Ok(()) => Ok(()),
            Err(SpillFinishError::Io(error)) => Err(error),
            Err(SpillFinishError::BeforePublish(never)) => match never {},
        }
    }

    /// Keeps a protected partition's exact authority when provider destruction
    /// unwinds after the provider was removed from the file.
    pub(super) fn finish_partition_write(
        &mut self,
        cleanup: &AccountedError,
    ) -> std::io::Result<()> {
        match self.finish_write_before_publish_inner(
            || Ok::<_, std::convert::Infallible>(()),
            Some(cleanup),
        ) {
            Ok(()) => Ok(()),
            Err(SpillFinishError::Io(error)) => Err(error),
            Err(SpillFinishError::BeforePublish(never)) => match never {},
        }
    }

    /// Finishes a staged file while running one typed operation after the
    /// physical writer backing and provider write state are gone, but before
    /// lifecycle validation and publication.
    pub(super) fn finish_write_before_publish<E>(
        &mut self,
        before_publish: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), SpillFinishError<E>> {
        self.finish_write_before_publish_inner(before_publish, None)
    }

    fn finish_write_before_publish_inner<E>(
        &mut self,
        before_publish: impl FnOnce() -> Result<(), E>,
        partition_cleanup: Option<&AccountedError>,
    ) -> Result<(), SpillFinishError<E>> {
        if self.finished {
            return Err(SpillFinishError::Io(invalid_input(
                "spill file is already published",
            )));
        }
        if self.poisoned {
            return Err(SpillFinishError::Io(invalid_input(
                "cannot finish a poisoned spill file",
            )));
        }
        match self.phase {
            RecordPhase::Sort { expected, observed }
            | RecordPhase::Partition { expected, observed }
                if expected != observed =>
            {
                return Err(SpillFinishError::Io(invalid_input(format!(
                    "declared {expected} records but wrote {observed}",
                ))));
            }
            RecordPhase::NeedFileStart
            | RecordPhase::NeedSortRunStart
            | RecordPhase::NeedPartitionStart => {
                return Err(SpillFinishError::Io(invalid_input(
                    "spill file has no role start record",
                )));
            }
            RecordPhase::Ended => {
                return Err(SpillFinishError::Io(invalid_input(
                    "spill file already ended",
                )));
            }
            _ => {}
        }

        let preceding_count = self.next_sequence;
        self.write_frame(SpillRecordKind::FileEnd, &preceding_count.to_le_bytes())
            .map_err(SpillFinishError::Io)?;
        self.phase = RecordPhase::Ended;

        let mut writer = self
            .writer
            .take()
            .ok_or_else(|| SpillFinishError::Io(invalid_input("spill write phase ended")))?;
        if let Err(error) = self.io.check(SpillIoOperation::Flush) {
            self.poisoned = true;
            return Err(SpillFinishError::Io(error));
        }
        if let Err(error) = writer.flush() {
            self.poisoned = true;
            return Err(SpillFinishError::Io(error));
        }
        let file = match writer.into_inner() {
            Ok(file) => file,
            Err(error) => {
                self.poisoned = true;
                return Err(SpillFinishError::Io(error));
            }
        };
        if let Err(error) = self.io.check(SpillIoOperation::Sync) {
            self.poisoned = true;
            return Err(SpillFinishError::Io(error));
        }
        if let Err(error) = file.sync_all() {
            self.poisoned = true;
            return Err(SpillFinishError::Io(error));
        }
        let physical_bytes = match file.metadata() {
            Ok(metadata) => metadata.len(),
            Err(error) => {
                self.poisoned = true;
                return Err(SpillFinishError::Io(error));
            }
        };
        if physical_bytes != self.bytes_written {
            self.poisoned = true;
            return Err(SpillFinishError::Io(invalid_data(format!(
                "synced spill file length {physical_bytes} does not match reserved writer bytes {}",
                self.bytes_written
            ))));
        }
        close_file(file);
        self.retire_partition_record(partition_cleanup);

        if let Err(error) = before_publish() {
            self.poisoned = true;
            return Err(SpillFinishError::BeforePublish(error));
        }

        // Provider destructors and the caller's publication hook have now
        // completed. Measure through the retained creation receipt, then make
        // the final named-identity check after all quota authority callbacks.
        if let Err(error) = self.lifecycle.reserve_physical(self.bytes_written, 0, true) {
            self.poisoned = true;
            return Err(SpillFinishError::Io(error));
        }
        if let Err(error) = self.lifecycle.validate_entry(&self.path) {
            self.poisoned = true;
            return Err(SpillFinishError::Io(error));
        }
        if let Err(error) = self.manager.publish(&self.path, self.bytes_written) {
            self.poisoned = true;
            return Err(SpillFinishError::Io(error));
        }
        self.finished = true;
        Ok(())
    }

    /// Opens a validated framed reader after successful publication.
    ///
    /// # Errors
    ///
    /// Returns an error before publication, while leased/deleting, on physical
    /// replacement, provider initialization, or initial-frame validation.
    pub fn reader(&self) -> std::io::Result<SpillFileReader> {
        if !self.finished {
            return Err(invalid_input("spill file is not finished/published"));
        }
        let reader_lease = SpillReaderLease::acquire(Arc::clone(&self.lifecycle))?;
        self.io.check(SpillIoOperation::ReadOpen)?;
        let file = self.lifecycle.open_reader(&self.path)?;
        let open_record = self.provider.begin_file(self.identity)?;
        SpillFileReader::new(
            BufReader::with_capacity(BUFFER_SIZE, file),
            self.identity,
            self.role,
            self.limits,
            self.sealed,
            BackstoppedOpenRecord::new(open_record),
            Arc::clone(&self.io),
            Arc::clone(&self.lifecycle),
            reader_lease,
            None,
            None,
            None,
        )
    }

    /// Mints an identity-bound receipt from the cached hard-reader contract.
    ///
    /// Support, provider-bound, and hook-bound callbacks execute only in the
    /// `OnceLock` initializer. Unsupported, missing, or overflowing contracts
    /// cache `None` and remain compatibility-reader shapes forever.
    pub(crate) fn exact_owned_reader_qualification(&self) -> Option<ExactOwnedReaderQualification> {
        // Publication is transient lifecycle state, not an immutable provider
        // or hook capability. A staged probe must not poison the one-shot
        // bounds cache for the same file after `finish_write` publishes it.
        if !self.finished {
            return None;
        }
        let ExactOwnedReaderQualificationCache::Eligible(bounds) =
            *self.exact_owned_reader_qualification.get_or_init(|| {
                if !hard_qualified_reader_platform_supported() {
                    return ExactOwnedReaderQualificationCache::Ineligible(
                        ExactOwnedReaderQualificationFailure::UnsupportedPlatform,
                    );
                }
                if !self.provider.supports_qualified_exact_open() {
                    return ExactOwnedReaderQualificationCache::Ineligible(
                        ExactOwnedReaderQualificationFailure::MissingExactProviderOutput,
                    );
                }
                let Some(provider_workspace_bytes) =
                    self.provider.file_workspace_allocation_bound()
                else {
                    return ExactOwnedReaderQualificationCache::Ineligible(
                        ExactOwnedReaderQualificationFailure::MissingProviderBound,
                    );
                };
                let Some(hook_workspace_bytes) = self.io.qualified_reader_hook_workspace_bound()
                else {
                    return ExactOwnedReaderQualificationCache::Ineligible(
                        ExactOwnedReaderQualificationFailure::MissingHookBound,
                    );
                };
                let Some(initial_workspace_bytes) =
                    provider_workspace_bytes.checked_add(hook_workspace_bytes)
                else {
                    return ExactOwnedReaderQualificationCache::Ineligible(
                        ExactOwnedReaderQualificationFailure::BoundOverflow {
                            provider_bytes: provider_workspace_bytes,
                            hook_bytes: hook_workspace_bytes,
                        },
                    );
                };
                ExactOwnedReaderQualificationCache::Eligible(ExactOwnedReaderQualificationBounds {
                    provider_workspace_bytes,
                    hook_workspace_bytes,
                    initial_workspace_bytes,
                })
            })
        else {
            return None;
        };
        Some(ExactOwnedReaderQualification {
            identity: self.identity,
            bounds,
        })
    }

    fn exact_owned_reader_qualification_failure(&self) -> ProviderAccountedReaderPrimary {
        let cache = *self
            .exact_owned_reader_qualification
            .get()
            .expect("qualification failure is requested only after cached preflight");
        let ExactOwnedReaderQualificationCache::Ineligible(failure) = cache else {
            unreachable!("eligible qualification has a receipt")
        };
        match failure {
            ExactOwnedReaderQualificationFailure::UnsupportedPlatform => {
                ProviderAccountedReaderPrimary::Read(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::Unsupported(
                        "hard-qualified spill reader open is not implemented on this platform",
                    ),
                ))
            }
            ExactOwnedReaderQualificationFailure::MissingExactProviderOutput => {
                ProviderAccountedReaderPrimary::MissingExactProviderOutput
            }
            ExactOwnedReaderQualificationFailure::MissingProviderBound => {
                ProviderAccountedReaderPrimary::MissingProviderBound
            }
            ExactOwnedReaderQualificationFailure::MissingHookBound => {
                ProviderAccountedReaderPrimary::MissingQualifiedIoHookBound
            }
            ExactOwnedReaderQualificationFailure::BoundOverflow {
                provider_bytes,
                hook_bytes,
            } => ProviderAccountedReaderPrimary::InitialWorkspaceBoundOverflow {
                provider_bytes,
                hook_bytes,
            },
        }
    }

    /// Opens a framed reader without the compatibility 64 KiB buffer after a
    /// caller-owned grant admits the provider and core control-frame
    /// workspace. The caller must retain that grant until the reader is
    /// dropped. Providers without a trusted bound fail closed before open.
    ///
    /// This borrowed compatibility seam does not retain admission authority
    /// with an escaping provider error or panic. Resource-qualified adapters
    /// must use [`Self::reader_with_owned_provider_admission`] instead. The provider
    /// bound callback, all I/O hooks (including fixed-control reads), lease,
    /// and filesystem-open seams remain outside this borrowed admission and
    /// subject to their documented nonallocating/audited contracts.
    pub(crate) fn reader_with_admission(
        &self,
        admit: impl FnMut(usize) -> std::io::Result<()>,
    ) -> std::io::Result<SpillFileReader> {
        self.reader_with_admission_inner(admit, None, false)
    }

    /// Opens the scalar compatibility reader with a pre-admitted shared
    /// cleanup witness retained by its caller through physical destruction.
    pub(super) fn reader_with_admission_and_cleanup(
        &self,
        admit: impl FnMut(usize) -> std::io::Result<()>,
        cleanup: AccountedError,
    ) -> std::io::Result<SpillFileReader> {
        self.reader_with_admission_inner(admit, Some(cleanup), false)
    }

    /// Partition adapter retains the established Vec provider and platform
    /// opener while charging every reader-hook outcome before opening.
    pub(super) fn reader_with_partition_admission_and_cleanup(
        &self,
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
        cleanup: AccountedError,
    ) -> std::io::Result<SpillFileReader> {
        let hook = self
            .io
            .qualified_reader_hook_workspace_bound()
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::Unsupported))?;
        let buffered = self.role == SpillFileRole::NativePartition;
        let buffer_bytes = if buffered {
            qualified_partition_reader_buffer_requested_bytes()
        } else {
            0
        };
        self.reader_with_admission_inner(
            |required| {
                let required = required
                    .checked_add(hook)
                    .and_then(|bytes| bytes.checked_add(buffer_bytes))
                    .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
                admit(required)
            },
            Some(cleanup),
            buffered,
        )
    }

    fn reader_with_admission_inner(
        &self,
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
        cleanup: Option<AccountedError>,
        buffered_partition: bool,
    ) -> std::io::Result<SpillFileReader> {
        let _operation = ScalarReaderOperationGuard::new(cleanup.as_ref());
        if !self.finished {
            return Err(invalid_input("spill file is not finished/published"));
        }
        let provider_bound = self
            .provider
            .file_workspace_allocation_bound()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "spill provider does not declare a qualified file-workspace bound",
                )
            })?;
        // Provider initialization itself can allocate or run a KDF, so its
        // complete file-workspace authority must exist before begin_file.
        admit(provider_bound)?;

        let reader_lease = SpillReaderLease::acquire(Arc::clone(&self.lifecycle))?;
        self.io.check(SpillIoOperation::ReadOpen)?;
        let file = self.lifecycle.open_reader(&self.path)?;
        let mut open_record = BackstoppedOpenRecord {
            inner: None,
            scalar_cleanup: cleanup,
        };
        open_record.inner = Some(self.provider.begin_file(self.identity)?);
        let (control_provider_bound, control_core_bound) =
            qualified_control_open_bounds(&*open_record)?;
        if control_provider_bound > provider_bound {
            return Err(invalid_data(format!(
                "spill provider control-frame open bound {control_provider_bound} exceeds its declared {provider_bound}-byte file workspace"
            )));
        }
        let full_workspace_bound = provider_bound
            .checked_add(control_core_bound)
            .ok_or_else(|| invalid_input("qualified spill reader workspace bound overflow"))?;
        admit(full_workspace_bound)?;
        // The private partition adapter includes this exact fixed backing in
        // every workspace admission; scalar and ordinary borrowed readers keep
        // their existing zero-capacity buffer and allocation protocol.
        let partition_read_ahead = if buffered_partition {
            Some(PartitionReadAhead::try_new()?)
        } else {
            None
        };
        SpillFileReader::new(
            BufReader::with_capacity(0, file),
            self.identity,
            self.role,
            self.limits,
            self.sealed,
            open_record,
            Arc::clone(&self.io),
            Arc::clone(&self.lifecycle),
            reader_lease,
            Some(control_provider_bound),
            Some(control_core_bound),
            partition_read_ahead,
        )
    }

    /// Opens a provider-accounted reader while owning its dedicated workspace.
    ///
    /// The input must be a dedicated zero-sized child grant. Successful
    /// construction returns a non-detachable reader/grant owner. Provider
    /// errors and panics after `begin_file` starts remain paired with the
    /// workspace that covers their payload construction and destruction.
    /// Provider `begin_file`, control-bound callbacks, and control-record open
    /// implementations must keep all success, returned-error, and panic
    /// payloads within the declared file workspace. One immutable general
    /// reader-hook peak covers `ReadOpen` plus control/data `ReadHeader` and
    /// `ReadPayload` callbacks and is admitted before any of them. A hostile
    /// provider-record destructor is backstopped and cannot replace an
    /// in-flight primary; its authority is permanently retained if the
    /// physical destructor failure itself must be forgotten.
    /// One exact cloneable error-publication block is separately admitted
    /// before the reader lease or any provider/I/O operation covered by the
    /// declared bounds. It remains with the live reader and is consumed only
    /// by a terminal owned-operation failure; construction failure retains its
    /// typed build error and child grant without flattening either into the
    /// provider workspace.
    ///
    /// Provider and hook bound-reporting callbacks remain explicit
    /// pre-admission nonallocating/no-unwind trust seams. Native Unix opens use
    /// a fixed-stack `openat` capability path; unsupported platforms fail
    /// closed before consumption and retain the compatibility reader API.
    /// Providers must also opt into core-owned exact plaintext output. The
    /// built-in cleartext and authenticated engine providers both do so;
    /// third-party providers retain compatibility-reader capability when they
    /// do not implement this additive hard-qualified seam.
    #[allow(
        clippy::result_large_err,
        dead_code,
        reason = "boxing would allocate outside the admitted transport and lose by-value recovery authority; the generic owned reader lifetime primitive also awaits partition migration"
    )]
    pub(crate) fn reader_with_owned_provider_admission(
        &self,
        workspace: MemoryGrant,
    ) -> Result<ProviderAccountedSpillFileReader, ProviderAccountedReaderError> {
        // Preserve the established validation precedence and keep an invalid
        // caller-owned grant entirely outside the optional qualification
        // seam. In particular, a non-empty grant must not initialize the
        // one-shot provider/hook capability cache.
        if workspace.size() != 0 {
            return Err(ProviderAccountedReaderError::new(
                ProviderAccountedReaderPrimary::NonEmptyWorkspace {
                    actual_bytes: workspace.size(),
                },
                workspace,
            ));
        }
        if !self.finished {
            return Err(ProviderAccountedReaderError::new(
                ProviderAccountedReaderPrimary::Unpublished,
                workspace,
            ));
        }
        let Some(qualification) = self.exact_owned_reader_qualification() else {
            return Err(ProviderAccountedReaderError::new(
                self.exact_owned_reader_qualification_failure(),
                workspace,
            ));
        };
        self.reader_with_owned_provider_qualification_inner(qualification, workspace, None)
    }

    /// Exact-sort adapter: consumes the cached identity-bound qualification
    /// and publishes every construction failure into a pre-admitted cloneable
    /// owner without formatting or source erasure. On success, the unused
    /// construction slot is physically deallocated and its live grant moves
    /// into the reader receipt for explicit terminal resolution.
    pub(crate) fn reader_with_exact_owned_provider_admission(
        &self,
        qualification: ExactOwnedReaderQualification,
        workspace: MemoryGrant,
        failure_publisher: AccountedErrorPublisher<ProviderAccountedReaderError>,
        observer: &dyn ProviderGrantTransitionObserver,
    ) -> Result<ProviderAccountedSpillFileReader, AccountedError> {
        match self.reader_with_owned_provider_qualification_inner(
            qualification,
            workspace,
            Some(observer),
        ) {
            Ok(mut reader) => {
                // The construction-failure slot was never published. Destroy
                // its empty block now, but keep the exact grant non-detachably
                // with the reader until explicit terminal resolution can
                // observe any release failure.
                reader.install_construction_publication_grant(
                    failure_publisher.into_unpublished_grant(),
                );
                Ok(reader)
            }
            Err(error) => Err(failure_publisher.publish(error)),
        }
    }

    #[allow(
        clippy::result_large_err,
        reason = "boxing would allocate outside the admitted transport and lose the construction failure's by-value recovery authority"
    )]
    fn reader_with_owned_provider_qualification_inner(
        &self,
        qualification: ExactOwnedReaderQualification,
        mut workspace: MemoryGrant,
        observer: Option<&dyn ProviderGrantTransitionObserver>,
    ) -> Result<ProviderAccountedSpillFileReader, ProviderAccountedReaderError> {
        if workspace.size() != 0 {
            return Err(ProviderAccountedReaderError::new(
                ProviderAccountedReaderPrimary::NonEmptyWorkspace {
                    actual_bytes: workspace.size(),
                },
                workspace,
            ));
        }
        if !self.finished {
            return Err(ProviderAccountedReaderError::new(
                ProviderAccountedReaderPrimary::Unpublished,
                workspace,
            ));
        }
        if qualification.identity != self.identity {
            return Err(ProviderAccountedReaderError::new(
                ProviderAccountedReaderPrimary::InvalidQualificationIdentity,
                workspace,
            ));
        }
        let provider_bound = qualification.bounds.provider_workspace_bytes;
        let hook_bound = qualification.bounds.hook_workspace_bytes;
        let initial_workspace_bound = qualification.bounds.initial_workspace_bytes;
        let previous_workspace = workspace.size();
        let resize = workspace.try_resize(initial_workspace_bound);
        let observation =
            observer.map(|observer| observer.replace_reader(previous_workspace, workspace.size()));
        if let Err(error) = resize {
            return Err(ProviderAccountedReaderError::new(
                ProviderAccountedReaderPrimary::Memory(error),
                workspace,
            ));
        }
        if let Some(Err(error)) = observation {
            return Err(ProviderAccountedReaderError::new(
                ProviderAccountedReaderPrimary::Memory(error),
                workspace,
            ));
        }

        // The reader can suffer exactly one terminal consuming-operation
        // failure. The nonallocating bound-reporting trust seams have already
        // run; reserve the cloneable publication block before acquiring a lease
        // or invoking an admitted provider/I/O operation, so every such failure
        // can move its original typed owner out without a fallible allocation.
        let publication_grant = workspace
            .split(0)
            .expect("a validated zero-byte workspace can always split a zero child");
        let operation_error_publisher =
            match AccountedErrorPublisher::try_new(publication_grant) {
                Ok(publisher) => {
                    let publication_bytes = publisher.granted_bytes();
                    let total = workspace.size().checked_add(publication_bytes);
                    let Some(total) = total else {
                        if let Some(observer) = observer {
                            observer.publish_unrepresentable();
                        }
                        return Err(
                        ProviderAccountedReaderError::after_cleanup_with_unused_operation_publisher(
                        ProviderAccountedReaderPrimary::Memory(
                            MemoryGrantError::ArithmeticOverflow {
                                current_bytes: workspace.size(),
                                additional_bytes: publication_bytes,
                            },
                        ),
                        workspace,
                        true,
                        publisher,
                    ));
                    };
                    if let Some(observer) = observer
                        && let Err(error) = observer.replace_reader(workspace.size(), total)
                    {
                        return Err(
                        ProviderAccountedReaderError::after_cleanup_with_unused_operation_publisher(
                        ProviderAccountedReaderPrimary::Memory(error),
                        workspace,
                        true,
                        publisher,
                    ));
                    }
                    publisher
                }
                Err(error) => {
                    let current = workspace.size().checked_add(error.grant().size());
                    match (observer, current) {
                        (Some(observer), Some(current)) => {
                            let _ = observer.replace_reader(workspace.size(), current);
                        }
                        (Some(observer), None) => observer.publish_unrepresentable(),
                        (None, _) => {}
                    }
                    return Err(ProviderAccountedReaderError::publication(error, workspace));
                }
            };

        let mut staged_open_record = None;
        let construction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let reader_lease = SpillReaderLease::acquire_qualified(Arc::clone(&self.lifecycle))
                .map_err(ProviderAccountedReaderPrimary::Read)?;
            self.io.check(SpillIoOperation::ReadOpen).map_err(|error| {
                ProviderAccountedReaderPrimary::Read(QualifiedReaderFailure::Io(error))
            })?;
            let file = self
                .lifecycle
                .open_reader_qualified()
                .map_err(ProviderAccountedReaderPrimary::Read)?;
            staged_open_record = Some(BackstoppedOpenRecord::new(
                self.provider.begin_file(self.identity).map_err(|error| {
                    ProviderAccountedReaderPrimary::Read(QualifiedReaderFailure::Io(error))
                })?,
            ));
            let open_record = staged_open_record
                .as_ref()
                .expect("provider record was installed before bound validation");
            let (control_provider_bound, control_core_bound) =
                [8usize, MAX_FIXED_CONTROL_PAYLOAD_BYTES]
                    .into_iter()
                    .try_fold(
                        (0usize, 0usize),
                        |(provider_max, core_max), plaintext_bytes| {
                            let stored_bytes =
                                open_record.stored_len(plaintext_bytes).map_err(|error| {
                                    ProviderAccountedReaderPrimary::Read(
                                        QualifiedReaderFailure::Io(error),
                                    )
                                })?;
                            let control_provider_bound = open_record
                                .open_allocation_bound(stored_bytes)
                                .ok_or(ProviderAccountedReaderPrimary::MissingControlOpenBound {
                                    plaintext_bytes,
                                })?;
                            let stored_layout = std::alloc::Layout::array::<u8>(stored_bytes)
                                .map_err(|_| {
                                    ProviderAccountedReaderPrimary::ControlBufferBoundOverflow {
                                        stored_bytes,
                                    }
                                })?;
                            let plaintext_layout = std::alloc::Layout::array::<u8>(plaintext_bytes)
                                .map_err(|_| {
                                    ProviderAccountedReaderPrimary::ControlBufferBoundOverflow {
                                        stored_bytes,
                                    }
                                })?;
                            let control_core_bound = stored_layout
                                .size()
                                .checked_add(plaintext_layout.size())
                                .ok_or(
                                    ProviderAccountedReaderPrimary::ControlBufferBoundOverflow {
                                        stored_bytes,
                                    },
                                )?;
                            Ok::<_, ProviderAccountedReaderPrimary>((
                                provider_max.max(control_provider_bound),
                                core_max.max(control_core_bound),
                            ))
                        },
                    )?;
            if control_provider_bound > provider_bound {
                return Err(
                    ProviderAccountedReaderPrimary::ControlProviderBoundExceeded {
                        required_bytes: control_provider_bound,
                        declared_bytes: provider_bound,
                    },
                );
            }
            let full_workspace_bound = initial_workspace_bound
                .checked_add(control_core_bound)
                .ok_or(ProviderAccountedReaderPrimary::WorkspaceBoundOverflow {
                    provider_bytes: provider_bound,
                    hook_bytes: hook_bound,
                    core_bytes: control_core_bound,
                })?;
            let previous_total = workspace
                .size()
                .checked_add(operation_error_publisher.granted_bytes())
                .ok_or(ProviderAccountedReaderPrimary::Memory(
                    MemoryGrantError::ArithmeticOverflow {
                        current_bytes: workspace.size(),
                        additional_bytes: operation_error_publisher.granted_bytes(),
                    },
                ))?;
            let resize = workspace.try_resize(full_workspace_bound);
            let current_total = workspace
                .size()
                .checked_add(operation_error_publisher.granted_bytes())
                .ok_or(ProviderAccountedReaderPrimary::Memory(
                    MemoryGrantError::ArithmeticOverflow {
                        current_bytes: workspace.size(),
                        additional_bytes: operation_error_publisher.granted_bytes(),
                    },
                ))?;
            if let Some(observer) = observer {
                observer
                    .replace_reader(previous_total, current_total)
                    .map_err(ProviderAccountedReaderPrimary::Memory)?;
            }
            resize.map_err(ProviderAccountedReaderPrimary::Memory)?;

            // Zero-capacity buffering avoids a hidden 64 KiB allocation. The
            // control-frame core, provider, and hook peaks coexist within the
            // admitted workspace above.
            let open_record = staged_open_record
                .take()
                .expect("validated provider record remains staged");
            Ok::<_, ProviderAccountedReaderPrimary>(SpillFileReader::new_provider_accounted(
                BufReader::with_capacity(0, file),
                self.identity,
                self.role,
                self.limits,
                self.sealed,
                open_record,
                Arc::clone(&self.io),
                Arc::clone(&self.lifecycle),
                reader_lease,
                control_provider_bound,
                control_core_bound,
            ))
        }));

        match construction {
            Ok(Ok(ProviderAccountedReaderConstruction::Reader(reader))) => Ok(
                ProviderAccountedSpillFileReader::new(reader, workspace, operation_error_publisher),
            ),
            Ok(Ok(ProviderAccountedReaderConstruction::Error {
                primary,
                cleanup_complete,
            })) => Err(
                ProviderAccountedReaderError::after_cleanup_with_unused_operation_publisher(
                    ProviderAccountedReaderPrimary::Read(primary),
                    workspace,
                    cleanup_complete,
                    operation_error_publisher,
                ),
            ),
            Ok(Ok(ProviderAccountedReaderConstruction::Panic {
                payload,
                cleanup_complete,
            })) => Err(
                ProviderAccountedReaderError::after_cleanup_with_unused_operation_publisher(
                    ProviderAccountedReaderPrimary::CapturedPanic(payload),
                    workspace,
                    cleanup_complete,
                    operation_error_publisher,
                ),
            ),
            Ok(Err(primary)) => {
                let cleanup_complete = staged_open_record
                    .as_mut()
                    .is_none_or(BackstoppedOpenRecord::take_and_drop_inner);
                Err(
                    ProviderAccountedReaderError::after_cleanup_with_unused_operation_publisher(
                        primary,
                        workspace,
                        cleanup_complete,
                        operation_error_publisher,
                    ),
                )
            }
            Err(payload) => {
                let cleanup_complete = staged_open_record
                    .as_mut()
                    .is_none_or(BackstoppedOpenRecord::take_and_drop_inner);
                Err(
                    ProviderAccountedReaderError::after_cleanup_with_unused_operation_publisher(
                        ProviderAccountedReaderPrimary::CapturedPanic(payload),
                        workspace,
                        cleanup_complete,
                        operation_error_publisher,
                    ),
                )
            }
        }
    }

    /// Closes local handles and explicitly deletes this file.
    ///
    /// A failure retains manager registration and byte accounting for retry.
    ///
    /// # Errors
    ///
    /// Returns an error for a live lease, hook/filesystem failure, replacement,
    /// or accounting failure.
    pub fn close_and_delete(&mut self) -> std::io::Result<()> {
        self.discard_writer_backing();
        self.open_record = None;
        if self.deleted {
            return Ok(());
        }
        self.io.check(SpillIoOperation::Delete)?;
        self.lifecycle.delete_path(&self.path, true)?;
        self.manager.unregister(&self.path)?;
        self.deleted = true;
        Ok(())
    }

    pub(super) fn close_partition_and_delete(
        &mut self,
        cleanup: &AccountedError,
    ) -> std::io::Result<()> {
        self.discard_writer_backing();
        self.retire_partition_record(Some(cleanup));
        self.close_and_delete()
    }

    fn retire_partition_record(&mut self, cleanup: Option<&AccountedError>) {
        let record = self.open_record.take();
        if let Some(cleanup) = cleanup {
            if let Err(payload) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(record)))
            {
                cleanup.inspect::<super::partition::PartitionFailureCleanup, _>(
                    super::partition::PartitionFailureCleanup::mark_failed,
                );
                std::panic::resume_unwind(payload);
            }
        } else {
            drop(record);
        }
    }

    /// Retires physical backing and attempts deletion once. Failed deletion
    /// leaves the manager's tracked receipt intact; it does not itself imply
    /// that a provider allocation was leaked. Only an undestroyable diagnostic
    /// or provider state requires permanent workspace retention.
    pub(super) fn retire_owned_file(mut self) -> bool {
        let mut complete = self.retire_owned_writer();
        let cleanup =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.close_and_delete()));
        let destruction = match cleanup {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || drop(error),
            ))),
            Err(payload) => Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || drop(payload),
            ))),
        };
        if let Some(destruction) = destruction {
            super::manager::record_orphan_cleanup_failures(1);
            if let Err(payload) = destruction {
                super::forget_cleanup_failure(payload);
                complete = false;
            }
        }
        // This consumed owner already attempted cleanup. Suppress its legacy
        // Drop retry, which could forget a new opaque error after reporting
        // successful payload retirement. The ledger still owns failed deletes.
        self.deleted = true;
        complete
    }

    pub(super) fn retire_owned_writer(&mut self) -> bool {
        self.discard_writer_backing();
        self.open_record.take().is_none_or(|record| {
            super::run_cleanup_backstop(|| {
                drop(record);
                Ok::<(), std::convert::Infallible>(())
            })
        })
    }

    /// Drops only the fixed writer allocation and its file handle. Lifecycle
    /// cleanup remains owned by this `SpillFile` and is still retryable.
    pub(super) fn discard_writer_backing(&mut self) {
        self.writer = None;
    }
}

impl std::fmt::Debug for SpillFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpillFile")
            .field("path", &self.path)
            .field("identity", &self.identity)
            .field("role", &self.role)
            .field("bytes_written", &self.bytes_written)
            .field("finished", &self.finished)
            .field("deleted", &self.deleted)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        if self.deleted {
            return;
        }
        if !super::run_cleanup_backstop(|| self.close_and_delete()) {
            super::manager::record_orphan_cleanup_failures(1);
        }
    }
}

/// Shared scalar-reader cleanup witness stored in an admitted publication.
#[derive(Debug)]
pub(super) struct ScalarReaderCleanup {
    workspaces: std::cell::RefCell<[Option<MemoryGrant>; 3]>,
    failed: AtomicBool,
    operation_panicked: AtomicBool,
}

impl ScalarReaderCleanup {
    pub(super) fn new() -> Self {
        Self {
            workspaces: std::cell::RefCell::new([None, None, None]),
            failed: AtomicBool::new(false),
            operation_panicked: AtomicBool::new(false),
        }
    }

    pub(super) fn retain_workspaces(&self, workspaces: [Option<MemoryGrant>; 3]) {
        if let Ok(mut retained) = self.workspaces.try_borrow_mut()
            && retained.iter().all(Option::is_none)
        {
            *retained = workspaces;
            return;
        }
        self.mark_failed();
        for grant in workspaces.into_iter().flatten() {
            std::mem::forget(grant);
        }
    }

    pub(super) fn granted_bytes(&self) -> usize {
        self.workspaces.try_borrow().map_or(usize::MAX, |grants| {
            grants
                .iter()
                .flatten()
                .fold(0usize, |bytes, grant| bytes.saturating_add(grant.size()))
        })
    }

    pub(super) fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub(super) fn operation_panicked(&self) -> bool {
        self.operation_panicked.load(Ordering::Acquire)
    }

    pub(super) fn mark_failed(&self) {
        self.failed.store(true, Ordering::Release);
    }
}

impl Drop for ScalarReaderCleanup {
    fn drop(&mut self) {
        if self.failed() {
            for grant in self
                .workspaces
                .get_mut()
                .iter_mut()
                .filter_map(Option::take)
            {
                std::mem::forget(grant);
            }
        }
    }
}

impl std::fmt::Display for ScalarReaderCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("scalar spill reader cleanup failed")
    }
}

impl std::error::Error for ScalarReaderCleanup {}

/// Marks unwinding through reader operations for diagnostic classification.
struct ScalarReaderOperationGuard(Option<AccountedError>);

impl ScalarReaderOperationGuard {
    fn new(cleanup: Option<&AccountedError>) -> Self {
        Self(cleanup.cloned())
    }
}

impl Drop for ScalarReaderOperationGuard {
    fn drop(&mut self) {
        if let Some(cleanup) = self.0.as_ref()
            && std::thread::panicking()
        {
            cleanup.inspect::<ScalarReaderCleanup, _>(|state| {
                state.operation_panicked.store(true, Ordering::Release);
            });
            cleanup.inspect::<super::partition::PartitionFailureCleanup, _>(
                super::partition::PartitionFailureCleanup::mark_operation_failed,
            );
        }
    }
}

/// Drop backstop for provider-owned per-file state.
///
/// A provider is an extension boundary, so its destructor can panic while a
/// reader is already unwinding through another provider callback. The inner
/// record is never exposed or detached and is destroyed behind the spill
/// cleanup backstop.
struct BackstoppedOpenRecord {
    inner: Option<Box<dyn OpenSpillRecord>>,
    scalar_cleanup: Option<AccountedError>,
}

impl BackstoppedOpenRecord {
    fn new(inner: Box<dyn OpenSpillRecord>) -> Self {
        Self {
            inner: Some(inner),
            scalar_cleanup: None,
        }
    }

    fn take_inner(&mut self) -> Option<Box<dyn OpenSpillRecord>> {
        self.inner.take()
    }

    /// Destroys the provider record behind the hostile-cleanup boundary and
    /// reports whether no physical error or panic payload had to be forgotten.
    fn take_and_drop_inner(&mut self) -> bool {
        let Some(inner) = self.take_inner() else {
            return true;
        };
        let cleanup_complete = super::run_cleanup_backstop(|| {
            drop(inner);
            Ok::<(), std::convert::Infallible>(())
        });
        if !cleanup_complete && let Some(cleanup) = self.scalar_cleanup.as_ref() {
            cleanup.inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::mark_failed);
            cleanup.inspect::<super::partition::PartitionFailureCleanup, _>(
                super::partition::PartitionFailureCleanup::mark_failed,
            );
        }
        cleanup_complete
    }
}

impl Deref for BackstoppedOpenRecord {
    type Target = dyn OpenSpillRecord;

    fn deref(&self) -> &Self::Target {
        self.inner
            .as_deref()
            .expect("live spill reader retains its provider record")
    }
}

impl DerefMut for BackstoppedOpenRecord {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
            .as_deref_mut()
            .expect("live spill reader retains its provider record")
    }
}

impl Drop for BackstoppedOpenRecord {
    fn drop(&mut self) {
        // Compatibility readers do not own resource authority. Accounted
        // owners call `take_and_drop_inner` explicitly so this result can
        // govern whether their authority may be released.
        let _ = self.take_and_drop_inner();
    }
}

const UNPUBLISHED_READER_MESSAGE: &str = "spill file is not finished/published";

/// Failed explicit release of an already-deallocated publication slot.
///
/// The original shared block no longer exists, but this move-only value keeps
/// the exact accounting capability alive so its caller can retry or merge it
/// into a longer-lived frontier without minting scalar authority.
#[derive(Debug)]
#[must_use = "the retained grant is the caller's release or transfer authority"]
pub(crate) struct ProviderAccountedReaderUnusedPublicationError {
    error: Option<MemoryGrantError>,
    grant: Option<MemoryGrant>,
}

impl ProviderAccountedReaderUnusedPublicationError {
    fn new(error: MemoryGrantError, grant: MemoryGrant) -> Self {
        Self {
            error: Some(error),
            grant: Some(grant),
        }
    }

    /// Returns the structured release failure without detaching its grant.
    pub(crate) fn error(&self) -> &MemoryGrantError {
        self.error
            .as_ref()
            .expect("live unused-publication failure retains its diagnostic")
    }

    /// Returns the exact still-accounted bytes owned by the retry grant.
    #[cfg(test)]
    pub(crate) fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }

    #[must_use = "both the diagnostic and retry authority must be preserved"]
    pub(crate) fn into_parts(mut self) -> (MemoryGrantError, MemoryGrant) {
        (
            self.error
                .take()
                .expect("unused-publication failure transfers its diagnostic exactly once"),
            self.grant
                .take()
                .expect("unused-publication failure transfers its grant exactly once"),
        )
    }
}

impl std::fmt::Display for ProviderAccountedReaderUnusedPublicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unused accounted-error publication grant release failed: {}",
            self.error()
        )
    }
}

impl std::error::Error for ProviderAccountedReaderUnusedPublicationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error())
    }
}

impl Drop for ProviderAccountedReaderUnusedPublicationError {
    fn drop(&mut self) {
        release_unused_publication_grant_backstop(self.grant.take());
    }
}

fn release_unused_error_publisher<T>(
    publisher: AccountedErrorPublisher<T>,
) -> Result<(), ProviderAccountedReaderUnusedPublicationError>
where
    T: std::error::Error + Send + 'static,
{
    let mut grant = publisher.into_unpublished_grant();

    #[cfg(test)]
    if let Some(error) =
        TEST_UNUSED_PUBLISHER_RELEASE_FAILURE.with(|failure| failure.borrow_mut().take())
    {
        return Err(ProviderAccountedReaderUnusedPublicationError::new(
            error, grant,
        ));
    }

    match grant.try_resize(0) {
        Ok(()) => Ok(()),
        Err(error) => Err(ProviderAccountedReaderUnusedPublicationError::new(
            error, grant,
        )),
    }
}

/// Explicit terminal resolution of a provider-accounted reader.
#[derive(Debug)]
#[must_use = "reader terminal failures retain diagnostic and accounting ownership"]
pub(crate) enum ProviderAccountedReaderResolutionError {
    /// A pre-admitted reader-operation diagnostic owns the original failure.
    Accounted(AccountedError),
    /// The unused operation slot was deallocated, but its grant release failed.
    UnusedPublication(ProviderAccountedReaderUnusedPublicationError),
}

impl std::fmt::Display for ProviderAccountedReaderResolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Accounted(error) => std::fmt::Display::fmt(error, formatter),
            Self::UnusedPublication(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for ProviderAccountedReaderResolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Accounted(error) => Some(error),
            Self::UnusedPublication(error) => Some(error),
        }
    }
}

impl From<AccountedError> for ProviderAccountedReaderResolutionError {
    fn from(error: AccountedError) -> Self {
        Self::Accounted(error)
    }
}

enum ProviderAccountedReaderPrimary {
    Read(QualifiedReaderFailure),
    CapturedPanic(Box<dyn Any + Send>),
    Memory(MemoryGrantError),
    Unpublished,
    InvalidQualificationIdentity,
    NonEmptyWorkspace {
        actual_bytes: usize,
    },
    MissingProviderBound,
    MissingExactProviderOutput,
    MissingQualifiedIoHookBound,
    InitialWorkspaceBoundOverflow {
        provider_bytes: usize,
        hook_bytes: usize,
    },
    MissingControlOpenBound {
        plaintext_bytes: usize,
    },
    ControlBufferBoundOverflow {
        stored_bytes: usize,
    },
    ControlProviderBoundExceeded {
        required_bytes: usize,
        declared_bytes: usize,
    },
    WorkspaceBoundOverflow {
        provider_bytes: usize,
        hook_bytes: usize,
        core_bytes: usize,
    },
}

impl std::fmt::Debug for ProviderAccountedReaderPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(error) => formatter.debug_tuple("Read").field(error).finish(),
            Self::CapturedPanic(payload) => formatter
                .debug_struct("CapturedPanic")
                .field("payload_type_id", &payload.as_ref().type_id())
                .finish(),
            Self::Memory(error) => formatter.debug_tuple("Memory").field(error).finish(),
            Self::Unpublished => formatter.write_str("Unpublished"),
            Self::InvalidQualificationIdentity => {
                formatter.write_str("InvalidQualificationIdentity")
            }
            Self::NonEmptyWorkspace { actual_bytes } => formatter
                .debug_struct("NonEmptyWorkspace")
                .field("actual_bytes", actual_bytes)
                .finish(),
            Self::MissingProviderBound => formatter.write_str("MissingProviderBound"),
            Self::MissingExactProviderOutput => formatter.write_str("MissingExactProviderOutput"),
            Self::MissingQualifiedIoHookBound => formatter.write_str("MissingQualifiedIoHookBound"),
            Self::InitialWorkspaceBoundOverflow {
                provider_bytes,
                hook_bytes,
            } => formatter
                .debug_struct("InitialWorkspaceBoundOverflow")
                .field("provider_bytes", provider_bytes)
                .field("hook_bytes", hook_bytes)
                .finish(),
            Self::MissingControlOpenBound { plaintext_bytes } => formatter
                .debug_struct("MissingControlOpenBound")
                .field("plaintext_bytes", plaintext_bytes)
                .finish(),
            Self::ControlBufferBoundOverflow { stored_bytes } => formatter
                .debug_struct("ControlBufferBoundOverflow")
                .field("stored_bytes", stored_bytes)
                .finish(),
            Self::ControlProviderBoundExceeded {
                required_bytes,
                declared_bytes,
            } => formatter
                .debug_struct("ControlProviderBoundExceeded")
                .field("required_bytes", required_bytes)
                .field("declared_bytes", declared_bytes)
                .finish(),
            Self::WorkspaceBoundOverflow {
                provider_bytes,
                hook_bytes,
                core_bytes,
            } => formatter
                .debug_struct("WorkspaceBoundOverflow")
                .field("provider_bytes", provider_bytes)
                .field("hook_bytes", hook_bytes)
                .field("core_bytes", core_bytes)
                .finish(),
        }
    }
}

impl ProviderAccountedReaderPrimary {
    #[cfg(test)]
    fn kind(&self) -> std::io::ErrorKind {
        match self {
            Self::Read(error) => error.kind(),
            Self::CapturedPanic(_) => std::io::ErrorKind::Other,
            Self::Memory(_) => std::io::ErrorKind::OutOfMemory,
            Self::Unpublished
            | Self::InvalidQualificationIdentity
            | Self::NonEmptyWorkspace { .. } => std::io::ErrorKind::InvalidInput,
            Self::MissingProviderBound
            | Self::MissingExactProviderOutput
            | Self::MissingQualifiedIoHookBound
            | Self::MissingControlOpenBound { .. } => std::io::ErrorKind::Unsupported,
            Self::InitialWorkspaceBoundOverflow { .. }
            | Self::ControlBufferBoundOverflow { .. }
            | Self::WorkspaceBoundOverflow { .. } => std::io::ErrorKind::InvalidInput,
            Self::ControlProviderBoundExceeded { .. } => std::io::ErrorKind::InvalidData,
        }
    }
}

impl std::fmt::Display for ProviderAccountedReaderPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(error) => std::fmt::Display::fmt(error, formatter),
            Self::CapturedPanic(_) => {
                formatter.write_str("captured fatal panic while constructing qualified reader")
            }
            Self::Memory(error) => std::fmt::Display::fmt(error, formatter),
            Self::Unpublished => formatter.write_str(UNPUBLISHED_READER_MESSAGE),
            Self::InvalidQualificationIdentity => formatter
                .write_str("exact owned reader qualification belongs to another spill file"),
            Self::NonEmptyWorkspace { actual_bytes } => write!(
                formatter,
                "qualified spill reader requires a dedicated zero-sized grant, got {actual_bytes} bytes"
            ),
            Self::MissingProviderBound => formatter
                .write_str("spill provider does not declare a qualified file-workspace bound"),
            Self::MissingExactProviderOutput => formatter
                .write_str("spill provider does not support exact qualified plaintext output"),
            Self::MissingQualifiedIoHookBound => formatter.write_str(
                "spill I/O boundary does not declare a qualified reader hook workspace bound",
            ),
            Self::InitialWorkspaceBoundOverflow {
                provider_bytes,
                hook_bytes,
            } => write!(
                formatter,
                "qualified spill reader initial workspace bound overflow: {provider_bytes} provider bytes plus {hook_bytes} reader-hook bytes"
            ),
            Self::MissingControlOpenBound { plaintext_bytes } => write!(
                formatter,
                "spill provider does not declare a qualified open bound for a {plaintext_bytes}-byte control frame"
            ),
            Self::ControlBufferBoundOverflow { stored_bytes } => write!(
                formatter,
                "qualified control-frame buffer bound overflow for {stored_bytes} stored bytes"
            ),
            Self::ControlProviderBoundExceeded {
                required_bytes,
                declared_bytes,
            } => write!(
                formatter,
                "spill provider control-frame open bound {required_bytes} exceeds its declared {declared_bytes}-byte file workspace"
            ),
            Self::WorkspaceBoundOverflow {
                provider_bytes,
                hook_bytes,
                core_bytes,
            } => write!(
                formatter,
                "qualified spill reader workspace bound overflow: {provider_bytes} provider bytes plus {hook_bytes} reader-hook bytes plus {core_bytes} core bytes"
            ),
        }
    }
}

impl std::error::Error for ProviderAccountedReaderPrimary {}

#[derive(Debug)]
enum ProviderAccountedReaderErrorState {
    Ordinary {
        primary: ProviderAccountedReaderPrimary,
        cleanup_complete: bool,
    },
    OperationErrorPublication(AccountedErrorPublisherBuildError),
}

/// Owned provider-accounted reader-construction failure.
///
/// The primary is deliberately not detachable: a provider may return a
/// heap-bearing error whose construction and destruction are covered by the
/// retained workspace. Field extraction is prevented and `Drop` destroys the
/// physical payload behind a hostile-destructor backstop before releasing its
/// authority. A publication-slot build failure likewise remains paired with
/// its dedicated child grant. If physical cleanup itself must be forgotten,
/// its authority is permanently retained instead.
#[derive(Debug)]
#[must_use = "dropping the failure resolves every retained construction authority"]
pub(crate) struct ProviderAccountedReaderError {
    state: Option<ProviderAccountedReaderErrorState>,
    workspace: Option<MemoryGrant>,
    unused_publication_release: Option<ProviderAccountedReaderUnusedPublicationError>,
}

impl ProviderAccountedReaderError {
    fn new(primary: ProviderAccountedReaderPrimary, workspace: MemoryGrant) -> Self {
        Self::after_cleanup(primary, workspace, true)
    }

    fn after_cleanup(
        primary: ProviderAccountedReaderPrimary,
        workspace: MemoryGrant,
        cleanup_complete: bool,
    ) -> Self {
        Self {
            state: Some(ProviderAccountedReaderErrorState::Ordinary {
                primary,
                cleanup_complete,
            }),
            workspace: Some(workspace),
            unused_publication_release: None,
        }
    }

    fn after_cleanup_with_unused_operation_publisher(
        primary: ProviderAccountedReaderPrimary,
        workspace: MemoryGrant,
        cleanup_complete: bool,
        publisher: AccountedErrorPublisher<ProviderAccountedReaderOperationError>,
    ) -> Self {
        let unused_publication_release = release_unused_error_publisher(publisher).err();
        Self {
            state: Some(ProviderAccountedReaderErrorState::Ordinary {
                primary,
                cleanup_complete,
            }),
            workspace: Some(workspace),
            unused_publication_release,
        }
    }

    fn publication(failure: AccountedErrorPublisherBuildError, workspace: MemoryGrant) -> Self {
        Self {
            state: Some(ProviderAccountedReaderErrorState::OperationErrorPublication(failure)),
            workspace: Some(workspace),
            unused_publication_release: None,
        }
    }

    fn state(&self) -> &ProviderAccountedReaderErrorState {
        self.state
            .as_ref()
            .expect("provider-accounted reader failure retains its state")
    }

    fn primary(&self) -> Option<&ProviderAccountedReaderPrimary> {
        match self.state() {
            ProviderAccountedReaderErrorState::Ordinary { primary, .. } => Some(primary),
            ProviderAccountedReaderErrorState::OperationErrorPublication(_) => None,
        }
    }

    pub(crate) fn operator_classification(&self) -> ProviderAccountedReaderFailureClassification {
        match self.state() {
            ProviderAccountedReaderErrorState::Ordinary {
                primary: ProviderAccountedReaderPrimary::Read(error),
                ..
            } => classify_qualified_reader_failure(error),
            ProviderAccountedReaderErrorState::Ordinary {
                primary: ProviderAccountedReaderPrimary::Memory(error),
                ..
            } => ProviderAccountedReaderFailureClassification::Memory(error.clone()),
            ProviderAccountedReaderErrorState::OperationErrorPublication(error) => {
                match error.failure() {
                    AccountedErrorPublisherBuildFailure::Admission(error) => {
                        ProviderAccountedReaderFailureClassification::Memory(error.clone())
                    }
                    AccountedErrorPublisherBuildFailure::Allocation
                    | AccountedErrorPublisherBuildFailure::AllocationWithRollback(_) => {
                        ProviderAccountedReaderFailureClassification::Allocation
                    }
                    AccountedErrorPublisherBuildFailure::NonZeroGrant { .. } => {
                        ProviderAccountedReaderFailureClassification::Execution
                    }
                    _ => ProviderAccountedReaderFailureClassification::Execution,
                }
            }
            ProviderAccountedReaderErrorState::Ordinary { .. } => {
                ProviderAccountedReaderFailureClassification::Execution
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn kind(&self) -> std::io::ErrorKind {
        match self.state() {
            ProviderAccountedReaderErrorState::Ordinary { primary, .. } => primary.kind(),
            ProviderAccountedReaderErrorState::OperationErrorPublication(error) => {
                match error.failure() {
                    AccountedErrorPublisherBuildFailure::NonZeroGrant { .. } => {
                        std::io::ErrorKind::InvalidInput
                    }
                    AccountedErrorPublisherBuildFailure::Admission(_)
                    | AccountedErrorPublisherBuildFailure::Allocation
                    | AccountedErrorPublisherBuildFailure::AllocationWithRollback(_) => {
                        std::io::ErrorKind::OutOfMemory
                    }
                    _ => std::io::ErrorKind::Other,
                }
            }
        }
    }

    #[allow(
        dead_code,
        reason = "reader adapters borrow the exact publication-build failure during migration"
    )]
    pub(crate) fn operation_error_publication_failure(
        &self,
    ) -> Option<&AccountedErrorPublisherBuildError> {
        match self.state() {
            ProviderAccountedReaderErrorState::OperationErrorPublication(error) => Some(error),
            ProviderAccountedReaderErrorState::Ordinary { .. } => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn unused_publication_release_error(
        &self,
    ) -> Option<&ProviderAccountedReaderUnusedPublicationError> {
        self.unused_publication_release.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn is_missing_exact_provider_output(&self) -> bool {
        matches!(
            self.primary(),
            Some(ProviderAccountedReaderPrimary::MissingExactProviderOutput)
        )
    }

    #[allow(
        dead_code,
        reason = "reader adapters classify only the typed memory seam by borrowing"
    )]
    pub(crate) fn memory_error(&self) -> Option<&MemoryGrantError> {
        let primary = match self.state() {
            ProviderAccountedReaderErrorState::OperationErrorPublication(error) => {
                match error.failure() {
                    AccountedErrorPublisherBuildFailure::Admission(error)
                    | AccountedErrorPublisherBuildFailure::AllocationWithRollback(error) => {
                        Some(error)
                    }
                    _ => None,
                }
            }
            ProviderAccountedReaderErrorState::Ordinary {
                primary: ProviderAccountedReaderPrimary::Memory(error),
                ..
            } => Some(error),
            ProviderAccountedReaderErrorState::Ordinary { .. } => None,
        };
        primary.or_else(|| {
            self.unused_publication_release
                .as_ref()
                .map(ProviderAccountedReaderUnusedPublicationError::error)
        })
    }

    #[allow(
        dead_code,
        reason = "reader adapters inspect I/O failures without detaching their authority"
    )]
    pub(crate) fn io_error_kind(&self) -> Option<std::io::ErrorKind> {
        match self.primary() {
            Some(ProviderAccountedReaderPrimary::Read(QualifiedReaderFailure::Io(error))) => {
                Some(error.kind())
            }
            _ => None,
        }
    }

    /// Classifies the original fatal payload without exposing a cloneable
    /// reference that could detach heap from this owner's authority.
    #[cfg(test)]
    pub(crate) fn panic_payload_is<T: Any>(&self) -> bool {
        matches!(
            self.primary(),
            Some(ProviderAccountedReaderPrimary::CapturedPanic(payload)) if payload.is::<T>()
        )
    }

    #[cfg(test)]
    pub(crate) fn panic_payload_for_test(&self) -> Option<&(dyn Any + Send)> {
        match self.primary() {
            Some(ProviderAccountedReaderPrimary::CapturedPanic(payload)) => Some(payload.as_ref()),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn is_fatal_captured_panic(&self) -> bool {
        matches!(
            self.primary(),
            Some(ProviderAccountedReaderPrimary::CapturedPanic(_))
        )
    }

    #[cfg(test)]
    pub(crate) fn core_error(&self) -> Option<QualifiedFrameCoreError> {
        match self.primary() {
            Some(ProviderAccountedReaderPrimary::Read(QualifiedReaderFailure::Core(error))) => {
                Some(*error)
            }
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn unpublished_static_message(&self) -> Option<&'static str> {
        matches!(
            self.primary(),
            Some(ProviderAccountedReaderPrimary::Unpublished)
        )
        .then_some(UNPUBLISHED_READER_MESSAGE)
    }

    #[cfg(test)]
    pub(crate) fn granted_bytes(&self) -> usize {
        self.workspace
            .as_ref()
            .map_or(0, MemoryGrant::size)
            .checked_add(
                self.operation_error_publication_failure()
                    .map_or(0, |error| error.grant().size()),
            )
            .and_then(|bytes| {
                bytes.checked_add(self.unused_publication_release.as_ref().map_or(
                    0,
                    ProviderAccountedReaderUnusedPublicationError::granted_bytes,
                ))
            })
            .expect("test reader-construction authority total is representable")
    }
}

impl std::fmt::Display for ProviderAccountedReaderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.state() {
            ProviderAccountedReaderErrorState::Ordinary { primary, .. } => {
                std::fmt::Display::fmt(primary, formatter)
            }
            ProviderAccountedReaderErrorState::OperationErrorPublication(error) => {
                std::fmt::Display::fmt(error, formatter)
            }
        }?;
        if let Some(release) = &self.unused_publication_release {
            write!(formatter, "; {release}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ProviderAccountedReaderError {}

impl Drop for ProviderAccountedReaderError {
    fn drop(&mut self) {
        let cleanup_complete = self.state.take().is_none_or(|state| match state {
            ProviderAccountedReaderErrorState::Ordinary {
                primary,
                cleanup_complete,
            } => {
                let primary_cleanup_complete = super::run_cleanup_backstop(|| {
                    drop(primary);
                    Ok::<(), std::convert::Infallible>(())
                });
                cleanup_complete && primary_cleanup_complete
            }
            ProviderAccountedReaderErrorState::OperationErrorPublication(failure) => {
                // This typed failure owns only its still-attached child grant;
                // no reader callback can have run before this path. Resolve
                // it explicitly so a refused release is retained fail-closed
                // instead of disappearing through `MemoryGrant::drop`.
                release_unused_publication_grant_backstop(Some(failure.into_grant()));
                true
            }
        });
        // The publication block was already physically deallocated before
        // this failure became externally visible. Destroy its release
        // diagnostic only after the original provider primary, then let the
        // retained grant make one final best-effort release attempt.
        drop(self.unused_publication_release.take());
        release_or_retain_accounted_workspace(self.workspace.take(), cleanup_complete);
    }
}

/// Observer-only receipt for all authority owned by one provider-accounted reader.
///
/// This scalar is minted from the live owner's private grant and carries no
/// allocation capability. It may be copied for checked observer arithmetic,
/// but can never release, resize, or replace the underlying authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProviderAccountedReaderReceipt {
    bytes: usize,
}

impl ProviderAccountedReaderReceipt {
    pub(crate) const fn bytes(self) -> usize {
        self.bytes
    }
}

/// Reader workspace retained after its physical file/provider state is shut.
///
/// This sealed intermediate lets an error or panic outlive the sorter without
/// keeping a reader lease that would block sorter-owned run deletion. Its
/// authority remains non-detachable and is released only after the original
/// diagnostic has been destroyed and physical quiescence succeeded.
#[derive(Debug)]
struct QuiescedProviderAccountedReader {
    workspace: Option<MemoryGrant>,
    construction_publication_grant: Option<MemoryGrant>,
    physical_cleanup_complete: bool,
}

impl QuiescedProviderAccountedReader {
    fn resolve_after_prior_cleanup(mut self, prior_cleanup_complete: bool) -> bool {
        let cleanup_complete = prior_cleanup_complete && self.physical_cleanup_complete;
        release_or_retain_accounted_workspace(self.workspace.take(), cleanup_complete);
        release_unused_publication_grant_backstop(self.construction_publication_grant.take());
        cleanup_complete
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        self.workspace
            .as_ref()
            .map_or(0, MemoryGrant::size)
            .checked_add(
                self.construction_publication_grant
                    .as_ref()
                    .map_or(0, MemoryGrant::size),
            )
            .expect("qualified reader diagnostic authority is representable")
    }
}

impl Drop for QuiescedProviderAccountedReader {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            // Missing the explicit diagnostic-first resolution path is an
            // internal protocol violation. Retaining authority is safer than
            // advertising capacity while an opaque payload may remain live.
            std::mem::forget(workspace);
        }
        // This grant covers an already-deallocated empty publication block,
        // not the opaque reader payload. It is independently safe to resolve;
        // a refused release is retained fail-closed by the backstop.
        release_unused_publication_grant_backstop(self.construction_publication_grant.take());
    }
}

/// Original failure from an operation on an owned qualified reader.
enum ProviderAccountedReaderOperationPrimary {
    Read(QualifiedReaderFailure),
    CapturedPanic {
        payload: Box<dyn Any + Send>,
        physical_cleanup_complete: bool,
    },
    Memory(MemoryGrantError),
    MemoryWithRead {
        error: MemoryGrantError,
        cleanup: QualifiedReaderFailure,
        phase: &'static str,
    },
    InvalidInput(&'static str),
    InvalidData(&'static str),
    SortColumnMismatch {
        declared: u32,
        expected: u32,
    },
    SortRowCountMismatch {
        declared: u64,
        expected: u64,
    },
}

impl std::fmt::Debug for ProviderAccountedReaderOperationPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(error) => formatter.debug_tuple("Read").field(error).finish(),
            Self::CapturedPanic {
                payload,
                physical_cleanup_complete,
            } => formatter
                .debug_struct("CapturedPanic")
                .field("payload_type_id", &payload.as_ref().type_id())
                .field("physical_cleanup_complete", physical_cleanup_complete)
                .finish_non_exhaustive(),
            Self::Memory(error) => formatter.debug_tuple("Memory").field(error).finish(),
            Self::MemoryWithRead {
                error,
                cleanup,
                phase,
            } => formatter
                .debug_struct("MemoryWithRead")
                .field("error", error)
                .field("cleanup", cleanup)
                .field("phase", phase)
                .finish(),
            Self::InvalidInput(message) => formatter
                .debug_tuple("InvalidInput")
                .field(message)
                .finish(),
            Self::InvalidData(message) => {
                formatter.debug_tuple("InvalidData").field(message).finish()
            }
            Self::SortColumnMismatch { declared, expected } => formatter
                .debug_struct("SortColumnMismatch")
                .field("declared", declared)
                .field("expected", expected)
                .finish(),
            Self::SortRowCountMismatch { declared, expected } => formatter
                .debug_struct("SortRowCountMismatch")
                .field("declared", declared)
                .field("expected", expected)
                .finish(),
        }
    }
}

impl ProviderAccountedReaderOperationPrimary {
    #[cfg(test)]
    fn kind(&self) -> std::io::ErrorKind {
        match self {
            Self::Read(error) => error.kind(),
            Self::CapturedPanic { .. } => std::io::ErrorKind::Other,
            Self::Memory(_) | Self::MemoryWithRead { .. } => std::io::ErrorKind::OutOfMemory,
            Self::InvalidInput(_) => std::io::ErrorKind::InvalidInput,
            Self::InvalidData(_)
            | Self::SortColumnMismatch { .. }
            | Self::SortRowCountMismatch { .. } => std::io::ErrorKind::InvalidData,
        }
    }
}

impl std::fmt::Display for ProviderAccountedReaderOperationPrimary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(error) => std::fmt::Display::fmt(error, formatter),
            Self::CapturedPanic { .. } => {
                formatter.write_str("captured fatal panic in qualified reader operation")
            }
            Self::Memory(error) => std::fmt::Display::fmt(error, formatter),
            Self::MemoryWithRead {
                error,
                cleanup,
                phase,
            } => write!(formatter, "{error}; {phase} also failed: {cleanup}"),
            Self::InvalidInput(message) | Self::InvalidData(message) => {
                formatter.write_str(message)
            }
            Self::SortColumnMismatch { declared, expected } => write!(
                formatter,
                "sort run declares {declared} columns, expected {expected}"
            ),
            Self::SortRowCountMismatch { declared, expected } => write!(
                formatter,
                "sort run declares {declared} rows, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for ProviderAccountedReaderOperationPrimary {}

/// Non-detachable reader-operation failure.
///
/// The original diagnostic, complete reader owner, and optional row-payload
/// child remain together. Destruction attempts every physical cleanup before
/// releasing either authority; any hostile destructor panic permanently
/// retains the authorities that may still cover stranded state.
#[derive(Debug)]
#[must_use = "dropping the reader failure resolves every retained authority"]
pub(crate) struct ProviderAccountedReaderOperationError {
    primary: Option<ProviderAccountedReaderOperationPrimary>,
    reader: Option<ProviderAccountedSpillFileReader>,
    quiesced_reader: Option<QuiescedProviderAccountedReader>,
    payload_grant: Option<MemoryGrant>,
}

impl ProviderAccountedReaderOperationError {
    fn new(
        primary: ProviderAccountedReaderOperationPrimary,
        reader: ProviderAccountedSpillFileReader,
        payload_grant: Option<MemoryGrant>,
    ) -> Self {
        Self {
            primary: Some(primary),
            reader: Some(reader),
            quiesced_reader: None,
            payload_grant,
        }
    }

    fn primary(&self) -> &ProviderAccountedReaderOperationPrimary {
        self.primary
            .as_ref()
            .expect("live reader-operation failure retains its primary")
    }

    pub(crate) fn operator_classification(&self) -> ProviderAccountedReaderFailureClassification {
        match self.primary() {
            ProviderAccountedReaderOperationPrimary::Read(error) => {
                classify_qualified_reader_failure(error)
            }
            ProviderAccountedReaderOperationPrimary::Memory(error)
            | ProviderAccountedReaderOperationPrimary::MemoryWithRead { error, .. } => {
                ProviderAccountedReaderFailureClassification::Memory(error.clone())
            }
            ProviderAccountedReaderOperationPrimary::CapturedPanic { .. }
            | ProviderAccountedReaderOperationPrimary::InvalidInput(_)
            | ProviderAccountedReaderOperationPrimary::InvalidData(_)
            | ProviderAccountedReaderOperationPrimary::SortColumnMismatch { .. }
            | ProviderAccountedReaderOperationPrimary::SortRowCountMismatch { .. } => {
                ProviderAccountedReaderFailureClassification::Execution
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn kind(&self) -> std::io::ErrorKind {
        self.primary().kind()
    }

    /// Whether this failure is a fatal captured unwind rather than an ordinary
    /// recoverable reader error.
    #[cfg(test)]
    pub(crate) fn is_fatal_captured_panic(&self) -> bool {
        matches!(
            self.primary(),
            ProviderAccountedReaderOperationPrimary::CapturedPanic { .. }
        )
    }

    /// Classifies the original fatal payload without exposing a cloneable
    /// reference that could detach heap from reader or row authority.
    #[cfg(test)]
    pub(crate) fn panic_payload_is<T: Any>(&self) -> bool {
        matches!(
            self.primary(),
            ProviderAccountedReaderOperationPrimary::CapturedPanic { payload, .. }
                if payload.is::<T>()
        )
    }

    #[cfg(test)]
    pub(crate) fn core_error(&self) -> Option<QualifiedFrameCoreError> {
        match self.primary() {
            ProviderAccountedReaderOperationPrimary::Read(QualifiedReaderFailure::Core(error)) => {
                Some(*error)
            }
            _ => None,
        }
    }

    /// Closes the physical reader while keeping its workspace sealed beside
    /// the original diagnostic. This lets the sorter delete its durable run
    /// after the error moves out without releasing authority that still
    /// covers the error payload.
    pub(super) fn quiesce_reader(&mut self) {
        let Some(reader) = self.reader.take() else {
            return;
        };
        assert!(
            self.quiesced_reader.is_none(),
            "reader-operation failure quiesces its reader exactly once"
        );
        self.quiesced_reader = Some(reader.into_quiesced_error_owner());
    }

    #[cfg(test)]
    pub(crate) fn retained_authority_bytes(&self) -> usize {
        let live_reader = self
            .reader
            .as_ref()
            .map_or(0, |reader| reader.receipt().bytes());
        let quiesced_reader = self
            .quiesced_reader
            .as_ref()
            .map_or(0, QuiescedProviderAccountedReader::retained_bytes);
        live_reader
            .checked_add(quiesced_reader)
            .and_then(|bytes| {
                bytes.checked_add(self.payload_grant.as_ref().map_or(0, MemoryGrant::size))
            })
            .expect("test reader authority total is representable")
    }
}

impl std::fmt::Display for ProviderAccountedReaderOperationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.primary(), formatter)
    }
}

impl std::error::Error for ProviderAccountedReaderOperationError {}

impl Drop for ProviderAccountedReaderOperationError {
    fn drop(&mut self) {
        self.quiesce_reader();
        let primary_cleanup_complete = self.primary.take().is_none_or(|primary| {
            let declared_cleanup_complete = match &primary {
                ProviderAccountedReaderOperationPrimary::CapturedPanic {
                    physical_cleanup_complete,
                    ..
                } => *physical_cleanup_complete,
                _ => true,
            };
            let primary_payload_cleanup_complete = super::run_cleanup_backstop(|| {
                drop(primary);
                Ok::<(), std::convert::Infallible>(())
            });
            declared_cleanup_complete && primary_payload_cleanup_complete
        });
        let reader_cleanup_complete = self
            .quiesced_reader
            .take()
            .is_none_or(|reader| reader.resolve_after_prior_cleanup(primary_cleanup_complete));
        release_or_retain_accounted_workspace(
            self.payload_grant.take(),
            primary_cleanup_complete && reader_cleanup_complete,
        );
    }
}

/// Encoded sort-row payload coupled to its dedicated child grant.
#[derive(Debug)]
#[must_use = "encoded row bytes must remain coupled to their payload authority"]
pub(crate) struct ProviderAccountedSortRow {
    payload: Option<QualifiedFramePayload>,
    grant: Option<MemoryGrant>,
}

impl ProviderAccountedSortRow {
    fn new_validated(payload: QualifiedFramePayload, grant: MemoryGrant) -> Self {
        debug_assert_eq!(grant.size(), payload.capacity());
        Self {
            payload: Some(payload),
            grant: Some(grant),
        }
    }

    /// Returns the exact provider-owned plaintext without exposing growing
    /// access or separating it from its child grant.
    pub(crate) fn payload(&self) -> &[u8] {
        self.payload
            .as_deref()
            .expect("accounted row retains its exact plaintext")
    }

    /// Returns the exact physical payload capacity admitted by this owner.
    pub(crate) fn payload_capacity(&self) -> usize {
        self.payload
            .as_ref()
            .map_or(0, allocator_api2::vec::Vec::capacity)
    }

    /// Returns the live child-grant size without exposing its authority.
    pub(crate) fn granted_bytes(&self) -> usize {
        self.grant.as_ref().map_or(0, MemoryGrant::size)
    }

    /// Expands the same child while the exact payload remains physically
    /// coupled to it. Shrinking is deliberately unavailable until consuming
    /// payload release transfers the grant to the decoded-row constructor.
    pub(crate) fn grow_grant_to(&mut self, bytes: usize) -> Result<(), MemoryGrantError> {
        let bytes = bytes.max(self.granted_bytes());
        self.grant
            .as_mut()
            .expect("accounted row retains its payload grant")
            .try_resize(bytes)
    }

    /// Consumes the owner, destroys the exact encoded allocation, and only
    /// then transfers its still-live grant to the decoded-row constructor.
    pub(crate) fn into_released_grant(mut self) -> MemoryGrant {
        drop(self.payload.take());
        self.grant
            .take()
            .expect("accounted row transfers its payload grant exactly once")
    }

    #[cfg(test)]
    pub(crate) fn payload_for_test(&self) -> &[u8] {
        self.payload()
    }

    #[cfg(test)]
    pub(crate) fn payload_capacity_for_test(&self) -> usize {
        self.payload_capacity()
    }

    #[cfg(test)]
    pub(crate) fn granted_bytes_for_test(&self) -> usize {
        self.granted_bytes()
    }
}

impl Drop for ProviderAccountedSortRow {
    fn drop(&mut self) {
        drop(self.payload.take());
        drop(self.grant.take());
    }
}

/// Reader and its non-detachable file/control workspace authority.
///
/// There is deliberately no raw mutable-reader accessor: it would permit
/// `mem::replace` to detach the physical reader from its grant. Adapters add
/// only narrowly forwarded framed-read operations as they migrate.
#[must_use = "dropping the owner closes the reader and resolves its workspace"]
pub(crate) struct ProviderAccountedSpillFileReader {
    reader: Option<SpillFileReader>,
    workspace: Option<MemoryGrant>,
    construction_publication_grant: Option<MemoryGrant>,
    operation_error_publisher:
        Option<AccountedErrorPublisher<ProviderAccountedReaderOperationError>>,
}

#[allow(
    clippy::result_large_err,
    reason = "boxing would allocate outside the authority retained by this non-detachable error owner"
)]
impl ProviderAccountedSpillFileReader {
    fn new(
        reader: SpillFileReader,
        workspace: MemoryGrant,
        operation_error_publisher: AccountedErrorPublisher<ProviderAccountedReaderOperationError>,
    ) -> Self {
        Self {
            reader: Some(reader),
            workspace: Some(workspace),
            construction_publication_grant: None,
            operation_error_publisher: Some(operation_error_publisher),
        }
    }

    fn install_construction_publication_grant(&mut self, grant: MemoryGrant) {
        assert!(
            self.construction_publication_grant.is_none(),
            "qualified reader installs its construction publication grant exactly once"
        );
        self.construction_publication_grant = Some(grant);
    }

    /// Mints an observer-only receipt from every live private reader grant.
    pub(crate) fn receipt(&self) -> ProviderAccountedReaderReceipt {
        let workspace_bytes = self.workspace.as_ref().map_or(0, MemoryGrant::size);
        let publication_bytes = self
            .operation_error_publisher
            .as_ref()
            .map_or(0, AccountedErrorPublisher::granted_bytes);
        let construction_publication_bytes = self
            .construction_publication_grant
            .as_ref()
            .map_or(0, MemoryGrant::size);
        ProviderAccountedReaderReceipt {
            bytes: workspace_bytes
                .checked_add(publication_bytes)
                .and_then(|bytes| bytes.checked_add(construction_publication_bytes))
                .expect("qualified reader authority was representable at construction"),
        }
    }

    fn publish_operation_error(
        mut self,
        primary: ProviderAccountedReaderOperationPrimary,
        payload_grant: Option<MemoryGrant>,
    ) -> AccountedError {
        let publisher = self
            .operation_error_publisher
            .take()
            .expect("live qualified reader retains its one-shot error publisher");
        let mut error = ProviderAccountedReaderOperationError::new(primary, self, payload_grant);
        // Immutable shared publication cannot expose a mutable cleanup seam.
        // Close provider/file state first while retaining its workspace beside
        // the original typed diagnostic, then publish without allocating.
        error.quiesce_reader();
        publisher.publish(error)
    }

    fn reader_mut(&mut self) -> &mut SpillFileReader {
        self.reader
            .as_mut()
            .expect("live provider-accounted owner retains its reader")
    }

    pub(super) fn read_declaration_owned(mut self) -> Result<(Self, (u32, u64)), AccountedError> {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let reader = self.reader_mut();
            match reader.role {
                SpillFileRole::SortRun => reader.read_sort_run_start_qualified(),
                SpillFileRole::NativePartition => {
                    reader.read_partition_start_qualified().map(|n| (0, n))
                }
                SpillFileRole::RdfAggregateState => Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::InvalidInput(
                        "aggregate files have no declaration record",
                    ),
                )),
            }
        }));
        match outcome {
            Ok(Ok(declaration)) => Ok((self, declaration)),
            Ok(Err(error)) => Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::Read(error),
                None,
            )),
            Err(payload) => Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::CapturedPanic {
                    payload,
                    physical_cleanup_complete: true,
                },
                None,
            )),
        }
    }

    /// Reads a sort-run declaration while moving the complete reader owner
    /// into every failure or panic path.
    pub(crate) fn read_sort_run_start_owned(
        mut self,
        expected_columns: u32,
        expected_rows: u64,
    ) -> Result<Self, AccountedError> {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.reader_mut().read_sort_run_start_qualified()
        }));
        match outcome {
            Ok(Ok((columns, _))) if columns != expected_columns => Err(self
                .publish_operation_error(
                    ProviderAccountedReaderOperationPrimary::SortColumnMismatch {
                        declared: columns,
                        expected: expected_columns,
                    },
                    None,
                )),
            Ok(Ok((_, rows))) if rows != expected_rows => Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::SortRowCountMismatch {
                    declared: rows,
                    expected: expected_rows,
                },
                None,
            )),
            Ok(Ok(_)) => Ok(self),
            Ok(Err(error)) => Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::Read(error),
                None,
            )),
            Err(payload) => Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::CapturedPanic {
                    payload,
                    physical_cleanup_complete: true,
                },
                None,
            )),
        }
    }

    /// Reads one sort row under an owned, dedicated payload child.
    ///
    /// On success, the encoded allocation and child move together into a
    /// sealed row owner. On failure or panic, the original diagnostics,
    /// reader workspace, and payload child remain non-detachable.
    #[cfg(test)]
    pub(crate) fn read_sort_row_owned(
        self,
        payload_grant: MemoryGrant,
    ) -> Result<(Self, ProviderAccountedSortRow), AccountedError> {
        self.read_record_owned_inner(SpillRecordKind::SortRow, payload_grant, None)
    }

    pub(crate) fn read_sort_row_owned_observing(
        self,
        payload_grant: MemoryGrant,
        observer: &dyn ProviderGrantTransitionObserver,
    ) -> Result<(Self, ProviderAccountedSortRow), AccountedError> {
        self.read_record_owned_inner(SpillRecordKind::SortRow, payload_grant, Some(observer))
    }

    pub(super) fn read_record_owned(
        self,
        kind: SpillRecordKind,
        payload_grant: MemoryGrant,
    ) -> Result<(Self, ProviderAccountedSortRow), AccountedError> {
        self.read_record_owned_inner(kind, payload_grant, None)
    }

    fn read_record_owned_inner(
        mut self,
        kind: SpillRecordKind,
        mut payload_grant: MemoryGrant,
        observer: Option<&dyn ProviderGrantTransitionObserver>,
    ) -> Result<(Self, ProviderAccountedSortRow), AccountedError> {
        if payload_grant.size() != 0 {
            return Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::InvalidInput(
                    "qualified sort-row read requires a dedicated zero-sized payload grant",
                ),
                Some(payload_grant),
            ));
        }

        let mut admission_failure = None;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.reader_mut()
                .read_record_qualified(kind, &mut |required| {
                    // Exact stored layout plus the provider peak is admitted once
                    // before the pinned exact allocation. There is no post-hoc
                    // capacity correction or transient top-up.
                    let previous = payload_grant.size();
                    let resize = payload_grant.try_resize(required);
                    let observation = observer
                        .map(|observer| observer.replace_row(previous, payload_grant.size()));
                    match (resize, observation) {
                        (Ok(()), None | Some(Ok(()))) => Ok(()),
                        (Err(error), _) | (Ok(()), Some(Err(error))) => {
                            admission_failure = Some(error);
                            Err(QualifiedReaderFailure::core(
                                QualifiedFrameCoreError::InvalidInput(
                                    "qualified sort-row payload admission was denied",
                                ),
                            ))
                        }
                    }
                })
        }));
        let payload = match outcome {
            Ok(result) => match (result, admission_failure) {
                (Ok(payload), None) => payload,
                (Ok(payload), Some(error)) => {
                    drop(payload);
                    return Err(self.publish_operation_error(
                        ProviderAccountedReaderOperationPrimary::Memory(error),
                        Some(payload_grant),
                    ));
                }
                (Err(reported), Some(error)) => {
                    return Err(self.publish_operation_error(
                        ProviderAccountedReaderOperationPrimary::MemoryWithRead {
                            error,
                            cleanup: reported,
                            phase: "sort-row admission",
                        },
                        Some(payload_grant),
                    ));
                }
                (Err(error), None) => {
                    return Err(self.publish_operation_error(
                        ProviderAccountedReaderOperationPrimary::Read(error),
                        Some(payload_grant),
                    ));
                }
            },
            Err(payload) => {
                return Err(self.publish_operation_error(
                    ProviderAccountedReaderOperationPrimary::CapturedPanic {
                        payload,
                        physical_cleanup_complete: true,
                    },
                    Some(payload_grant),
                ));
            }
        };

        let retained_payload_bytes = match std::alloc::Layout::array::<u8>(payload.capacity()) {
            Ok(layout) => layout.size(),
            Err(_) => {
                drop(payload);
                return Err(self.publish_operation_error(
                    ProviderAccountedReaderOperationPrimary::InvalidData(
                        "qualified sort-row retained layout is unrepresentable",
                    ),
                    Some(payload_grant),
                ));
            }
        };
        if payload_grant.size() < retained_payload_bytes {
            drop(payload);
            return Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::InvalidData(
                    "qualified sort-row read returned capacity beyond its admitted payload grant",
                ),
                Some(payload_grant),
            ));
        }
        let previous = payload_grant.size();
        let resize = payload_grant.try_resize(retained_payload_bytes);
        let observation =
            observer.map(|observer| observer.replace_row(previous, payload_grant.size()));
        if let Err(error) = resize {
            drop(payload);
            return Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::Memory(error),
                Some(payload_grant),
            ));
        }
        if let Some(Err(error)) = observation {
            drop(payload);
            return Err(self.publish_operation_error(
                ProviderAccountedReaderOperationPrimary::Memory(error),
                Some(payload_grant),
            ));
        }
        Ok((
            self,
            ProviderAccountedSortRow::new_validated(payload, payload_grant),
        ))
    }

    /// Validates the terminal frame and then resolves the complete reader.
    ///
    /// # Errors
    ///
    /// Returns a pre-admitted operation error, or the live unused-publication
    /// grant when its final explicit release is refused.
    pub(crate) fn finish_sort_run_owned(
        mut self,
    ) -> Result<(), ProviderAccountedReaderResolutionError> {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.reader_mut().finish_qualified()
        }));
        match outcome {
            Ok(Ok(())) => self.resolve_reader_owned(),
            Ok(Err(error)) => Err(self
                .publish_operation_error(ProviderAccountedReaderOperationPrimary::Read(error), None)
                .into()),
            Err(payload) => Err(self
                .publish_operation_error(
                    ProviderAccountedReaderOperationPrimary::CapturedPanic {
                        payload,
                        physical_cleanup_complete: true,
                    },
                    None,
                )
                .into()),
        }
    }

    /// Closes a partially consumed reader without validating its terminal
    /// frame, then explicitly resolves every unused publication grant.
    ///
    /// # Errors
    ///
    /// Returns a pre-admitted cleanup error, or the live unused-publication
    /// grant when its final explicit release is refused.
    pub(crate) fn abort_owned(self) -> Result<(), ProviderAccountedReaderResolutionError> {
        self.resolve_reader_owned()
    }

    fn resolve_reader_owned(mut self) -> Result<(), ProviderAccountedReaderResolutionError> {
        let mut reader = self
            .reader
            .take()
            .expect("live provider-accounted owner retains its reader");
        if let Some(record) = reader.open_record.take_inner() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(record))) {
                Ok(()) => {}
                Err(payload) => {
                    self.reader = Some(reader);
                    return Err(self
                        .publish_operation_error(
                            ProviderAccountedReaderOperationPrimary::CapturedPanic {
                                payload,
                                physical_cleanup_complete: false,
                            },
                            None,
                        )
                        .into());
                }
            }
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(reader))) {
            Ok(()) => {}
            Err(payload) => {
                return Err(self
                    .publish_operation_error(
                        ProviderAccountedReaderOperationPrimary::CapturedPanic {
                            payload,
                            physical_cleanup_complete: false,
                        },
                        None,
                    )
                    .into());
            }
        }
        if let Some(workspace) = self.workspace.as_mut()
            && let Err(error) = workspace.try_resize(0)
        {
            return Err(self
                .publish_operation_error(
                    ProviderAccountedReaderOperationPrimary::Memory(error),
                    None,
                )
                .into());
        }
        if let Some(grant) = self.construction_publication_grant.as_mut()
            && let Err(error) = grant.try_resize(0)
        {
            return Err(self
                .publish_operation_error(
                    ProviderAccountedReaderOperationPrimary::Memory(error),
                    None,
                )
                .into());
        }
        let operation_error_publisher = self
            .operation_error_publisher
            .take()
            .expect("live qualified reader retains its one-shot error publisher");
        let publication_release = release_unused_error_publisher(operation_error_publisher);
        drop(self);
        publication_release.map_err(ProviderAccountedReaderResolutionError::UnusedPublication)
    }

    /// Shuts the physical reader while transferring its workspace only into a
    /// sealed diagnostic owner. No raw reader/grant pair is exposed.
    fn into_quiesced_error_owner(mut self) -> QuiescedProviderAccountedReader {
        let physical_cleanup_complete = self.destroy_physical_reader();
        QuiescedProviderAccountedReader {
            workspace: self.workspace.take(),
            construction_publication_grant: self.construction_publication_grant.take(),
            physical_cleanup_complete,
        }
    }

    fn resolve_fields_after_prior_cleanup(&mut self, prior_cleanup_complete: bool) -> bool {
        let reader_cleanup_complete = self.destroy_physical_reader();
        let cleanup_complete = prior_cleanup_complete && reader_cleanup_complete;
        release_or_retain_accounted_workspace(self.workspace.take(), cleanup_complete);
        release_unused_publication_grant_backstop(self.construction_publication_grant.take());
        let operation_publication_grant = self
            .operation_error_publisher
            .take()
            .map(AccountedErrorPublisher::into_unpublished_grant);
        release_unused_publication_grant_backstop(operation_publication_grant);
        cleanup_complete
    }

    fn destroy_physical_reader(&mut self) -> bool {
        self.reader.take().is_none_or(|mut reader| {
            let provider_cleanup_complete = reader.teardown_provider_record();
            let remaining_reader_cleanup_complete = super::run_cleanup_backstop(|| {
                drop(reader);
                Ok::<(), std::convert::Infallible>(())
            });
            provider_cleanup_complete && remaining_reader_cleanup_complete
        })
    }
}

impl std::fmt::Debug for ProviderAccountedSpillFileReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderAccountedSpillFileReader")
            .field("workspace_bytes", &self.receipt().bytes())
            .field("has_reader", &self.reader.is_some())
            .field(
                "has_operation_error_publisher",
                &self.operation_error_publisher.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl Drop for ProviderAccountedSpillFileReader {
    fn drop(&mut self) {
        let _ = self.resolve_fields_after_prior_cleanup(true);
    }
}

fn release_or_retain_accounted_workspace(workspace: Option<MemoryGrant>, cleanup_complete: bool) {
    let Some(workspace) = workspace else {
        return;
    };
    if cleanup_complete {
        drop(workspace);
    } else {
        // `run_cleanup_backstop` intentionally forgets hostile physical
        // failures. Their matching authority must be forgotten as well rather
        // than advertising bytes as available while leaked payloads remain.
        std::mem::forget(workspace);
    }
}

fn release_unused_publication_grant_backstop(grant: Option<MemoryGrant>) {
    let Some(mut grant) = grant else {
        return;
    };
    if grant.try_resize(0).is_err() {
        // Drop cannot return a retry token. Retaining both the account charge
        // and the grant fail-closed is safer than allowing `MemoryGrant::drop`
        // to swallow a second release failure after the publication
        // allocation has already disappeared. This is intentionally permanent
        // retention: explicit result paths never rely on this backstop.
        std::mem::forget(grant);
    }
}

/// Fixed exact backing admitted with an owned native-partition reader.
struct PartitionReadAhead {
    bytes: QualifiedFramePayload,
    position: usize,
    filled: usize,
}

impl PartitionReadAhead {
    fn try_new() -> std::io::Result<Self> {
        let requested = qualified_partition_reader_buffer_requested_bytes();
        let mut bytes = ExactVec::new_in(Global);
        bytes
            .try_reserve_exact(requested)
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
        if bytes.capacity() != requested {
            return Err(invalid_data(
                "pinned partition read-ahead allocator returned an unexpected capacity",
            ));
        }
        bytes.resize(requested, 0);
        Ok(Self {
            bytes,
            position: 0,
            filled: 0,
        })
    }

    fn unread(&self) -> usize {
        self.filled - self.position
    }

    fn read(&mut self, reader: &mut BufReader<File>, output: &mut [u8]) -> std::io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.position == self.filled {
            // Large admitted payloads need no extra copy through the small buffer.
            if output.len() >= self.bytes.len() {
                return reader.read(output);
            }
            self.position = 0;
            self.filled = 0;
            self.filled = reader.read(self.bytes.as_mut_slice())?;
        }
        let count = output.len().min(self.unread());
        output[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}

/// Validating, transition-aware framed spill reader.
pub struct SpillFileReader {
    reader: BufReader<File>,
    partition_read_ahead: Option<PartitionReadAhead>,
    identity: SpillFileIdentity,
    role: SpillFileRole,
    limits: SpillFrameLimits,
    sealed: bool,
    open_record: BackstoppedOpenRecord,
    qualified_control_provider_bound: Option<usize>,
    qualified_control_core_bound: Option<usize>,
    next_sequence: u64,
    phase: RecordPhase,
    finished: bool,
    poisoned: bool,
    io: Arc<dyn SpillIo>,
    _reader_lease: SpillReaderLease,
}

enum ProviderAccountedReaderConstruction {
    Reader(SpillFileReader),
    Error {
        primary: QualifiedReaderFailure,
        cleanup_complete: bool,
    },
    Panic {
        payload: Box<dyn Any + Send>,
        cleanup_complete: bool,
    },
}

impl SpillFileReader {
    #[expect(
        clippy::too_many_arguments,
        reason = "reader construction transfers framed role, provider, limits, hook, and physical lease together"
    )]
    fn new(
        reader: BufReader<File>,
        identity: SpillFileIdentity,
        role: SpillFileRole,
        limits: SpillFrameLimits,
        sealed: bool,
        open_record: BackstoppedOpenRecord,
        io: Arc<dyn SpillIo>,
        lifecycle: Arc<SpillFileLifecycle>,
        reader_lease: SpillReaderLease,
        qualified_control_provider_bound: Option<usize>,
        qualified_control_core_bound: Option<usize>,
        partition_read_ahead: Option<PartitionReadAhead>,
    ) -> std::io::Result<Self> {
        let mut result = Self::new_unvalidated(
            reader,
            identity,
            role,
            limits,
            sealed,
            open_record,
            io,
            lifecycle,
            reader_lease,
            qualified_control_provider_bound,
            qualified_control_core_bound,
        );
        result.partition_read_ahead = partition_read_ahead;
        if result.open_record.scalar_cleanup.is_none() {
            result.validate_file_start()?;
            return Ok(result);
        }
        let validation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            result.validate_file_start()
        }));
        match validation {
            Ok(Ok(())) => Ok(result),
            Ok(Err(error)) => {
                result.teardown_for_accounted_failure();
                Err(error)
            }
            Err(payload) => {
                result.teardown_for_accounted_failure();
                std::panic::resume_unwind(payload)
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "accounted construction transfers the same framed capabilities plus cleanup reporting"
    )]
    fn new_provider_accounted(
        reader: BufReader<File>,
        identity: SpillFileIdentity,
        role: SpillFileRole,
        limits: SpillFrameLimits,
        sealed: bool,
        open_record: BackstoppedOpenRecord,
        io: Arc<dyn SpillIo>,
        lifecycle: Arc<SpillFileLifecycle>,
        reader_lease: SpillReaderLease,
        qualified_control_provider_bound: usize,
        qualified_control_core_bound: usize,
    ) -> ProviderAccountedReaderConstruction {
        let mut result = Self::new_unvalidated(
            reader,
            identity,
            role,
            limits,
            sealed,
            open_record,
            io,
            lifecycle,
            reader_lease,
            Some(qualified_control_provider_bound),
            Some(qualified_control_core_bound),
        );
        let validation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            result.validate_file_start_qualified()
        }));
        match validation {
            Ok(Ok(())) => ProviderAccountedReaderConstruction::Reader(result),
            Ok(Err(primary)) => {
                let cleanup_complete = result.teardown_for_accounted_failure();
                ProviderAccountedReaderConstruction::Error {
                    primary,
                    cleanup_complete,
                }
            }
            Err(payload) => {
                let cleanup_complete = result.teardown_for_accounted_failure();
                ProviderAccountedReaderConstruction::Panic {
                    payload,
                    cleanup_complete,
                }
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "reader construction transfers framed role, provider, limits, hook, and physical lease together"
    )]
    fn new_unvalidated(
        reader: BufReader<File>,
        identity: SpillFileIdentity,
        role: SpillFileRole,
        limits: SpillFrameLimits,
        sealed: bool,
        open_record: BackstoppedOpenRecord,
        io: Arc<dyn SpillIo>,
        lifecycle: Arc<SpillFileLifecycle>,
        reader_lease: SpillReaderLease,
        qualified_control_provider_bound: Option<usize>,
        qualified_control_core_bound: Option<usize>,
    ) -> Self {
        let result = Self {
            reader,
            partition_read_ahead: None,
            identity,
            role,
            limits,
            sealed,
            open_record,
            qualified_control_provider_bound,
            qualified_control_core_bound,
            next_sequence: 0,
            phase: RecordPhase::NeedFileStart,
            finished: false,
            poisoned: false,
            io,
            _reader_lease: reader_lease,
        };
        let _ = lifecycle;
        result
    }

    fn validate_file_start(&mut self) -> std::io::Result<()> {
        let role = self.role;
        let (_kind, payload) = self.read_frame(SpillRecordKind::FileStart)?;
        let payload_role = SpillFileRole::try_from(payload[0])?;
        if payload_role != role {
            return Err(invalid_data("spill FileStart role does not match handle"));
        }
        let codec_version = u16::from_le_bytes([payload[1], payload[2]]);
        if codec_version != SPILL_VALUE_CODEC_VERSION {
            return Err(invalid_data(format!(
                "unsupported spill value-codec version {codec_version}"
            )));
        }
        if payload[3..].iter().any(|byte| *byte != 0) {
            return Err(invalid_data("non-zero reserved FileStart bytes"));
        }
        self.phase = match role {
            SpillFileRole::SortRun => RecordPhase::NeedSortRunStart,
            SpillFileRole::NativePartition => RecordPhase::NeedPartitionStart,
            SpillFileRole::RdfAggregateState => RecordPhase::Aggregate { observed: 0 },
        };
        Ok(())
    }

    fn validate_file_start_qualified(&mut self) -> Result<(), QualifiedReaderFailure> {
        let role = self.role;
        let (_kind, payload) = self.read_frame_qualified_control(SpillRecordKind::FileStart)?;
        let payload_role = match payload[0] {
            1 => SpillFileRole::SortRun,
            2 => SpillFileRole::NativePartition,
            3 => SpillFileRole::RdfAggregateState,
            value => {
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::UnknownRole(value),
                ));
            }
        };
        if payload_role != role {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData("spill FileStart role does not match handle"),
            ));
        }
        let codec_version = u16::from_le_bytes([payload[1], payload[2]]);
        if codec_version != SPILL_VALUE_CODEC_VERSION {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::UnsupportedCodecVersion(codec_version),
            ));
        }
        if payload[3..].iter().any(|byte| *byte != 0) {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData("non-zero reserved FileStart bytes"),
            ));
        }
        self.phase = match role {
            SpillFileRole::SortRun => RecordPhase::NeedSortRunStart,
            SpillFileRole::NativePartition => RecordPhase::NeedPartitionStart,
            SpillFileRole::RdfAggregateState => RecordPhase::Aggregate { observed: 0 },
        };
        Ok(())
    }

    fn teardown_provider_record(&mut self) -> bool {
        self.open_record.take_and_drop_inner()
    }

    pub(super) fn teardown_for_accounted_failure(mut self) -> bool {
        let cleanup = self.open_record.scalar_cleanup.clone();
        let provider_cleanup_complete = self.teardown_provider_record();
        let remaining_reader_cleanup_complete = super::run_cleanup_backstop(|| {
            drop(self);
            Ok::<(), std::convert::Infallible>(())
        });
        let cleanup_complete = provider_cleanup_complete && remaining_reader_cleanup_complete;
        if !cleanup_complete && let Some(cleanup) = cleanup.as_ref() {
            cleanup.inspect::<ScalarReaderCleanup, _>(ScalarReaderCleanup::mark_failed);
            cleanup.inspect::<super::partition::PartitionFailureCleanup, _>(
                super::partition::PartitionFailureCleanup::mark_failed,
            );
        }
        cleanup_complete
    }

    /// Reads and validates the sort-run declaration.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong role/transition or malformed/truncated frame.
    pub fn read_sort_run_start(&mut self) -> std::io::Result<(u32, u64)> {
        if self.role != SpillFileRole::SortRun
            || !matches!(self.phase, RecordPhase::NeedSortRunStart)
        {
            return Err(invalid_data("illegal SortRunStart transition"));
        }
        let (_kind, payload) = self.read_frame(SpillRecordKind::SortRunStart)?;
        let columns = u32::from_le_bytes(
            payload
                .get(..4)
                .ok_or_else(|| invalid_data("short sort column payload"))?
                .try_into()
                .map_err(|_| invalid_data("invalid sort column payload"))?,
        );
        let rows = u64::from_le_bytes(
            payload
                .get(4..)
                .ok_or_else(|| invalid_data("short sort row-count payload"))?
                .try_into()
                .map_err(|_| invalid_data("invalid sort row-count payload"))?,
        );
        self.phase = RecordPhase::Sort {
            expected: rows,
            observed: 0,
        };
        Ok((columns, rows))
    }

    fn read_sort_run_start_qualified(&mut self) -> Result<(u32, u64), QualifiedReaderFailure> {
        if self.role != SpillFileRole::SortRun
            || !matches!(self.phase, RecordPhase::NeedSortRunStart)
        {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData("illegal SortRunStart transition"),
            ));
        }
        let (_kind, payload) = self.read_frame_qualified_control(SpillRecordKind::SortRunStart)?;
        let columns = payload.get(..4).ok_or_else(|| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "short sort column payload",
            ))
        })?;
        let columns: [u8; 4] = columns.try_into().map_err(|_| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "invalid sort column payload",
            ))
        })?;
        let rows = payload.get(4..).ok_or_else(|| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "short sort row-count payload",
            ))
        })?;
        let rows: [u8; 8] = rows.try_into().map_err(|_| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "invalid sort row-count payload",
            ))
        })?;
        let columns = u32::from_le_bytes(columns);
        let rows = u64::from_le_bytes(rows);
        self.phase = RecordPhase::Sort {
            expected: rows,
            observed: 0,
        };
        Ok((columns, rows))
    }

    /// Reads one complete row payload.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong transition/count or invalid framed record.
    pub fn read_sort_row(&mut self) -> std::io::Result<Vec<u8>> {
        self.read_sort_row_inner(None)
    }

    pub(crate) fn read_sort_row_with_admission(
        &mut self,
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
    ) -> std::io::Result<Vec<u8>> {
        self.read_sort_row_inner(Some(&mut admit))
    }

    fn read_sort_row_inner(
        &mut self,
        admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
    ) -> std::io::Result<Vec<u8>> {
        let RecordPhase::Sort { expected, observed } = self.phase else {
            return Err(invalid_data("illegal SortRow transition"));
        };
        if observed >= expected {
            return Err(invalid_data("more sort rows than declared"));
        }
        let (_kind, payload) =
            self.read_frame_inner(SpillRecordKind::SortRow, admission, None, None)?;
        self.phase = RecordPhase::Sort {
            expected,
            observed: observed + 1,
        };
        Ok(payload)
    }

    fn read_record_qualified(
        &mut self,
        kind: SpillRecordKind,
        admission: &mut dyn FnMut(usize) -> Result<(), QualifiedReaderFailure>,
    ) -> Result<QualifiedFramePayload, QualifiedReaderFailure> {
        let invalid = || {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "illegal qualified data-record transition",
            ))
        };
        let (expected, observed) = match (kind, self.phase) {
            (SpillRecordKind::SortRow, RecordPhase::Sort { expected, observed })
            | (SpillRecordKind::PartitionEntry, RecordPhase::Partition { expected, observed })
                if observed < expected =>
            {
                (Some(expected), observed)
            }
            (SpillRecordKind::AggregateState, RecordPhase::Aggregate { observed }) => {
                (None, observed)
            }
            _ => return Err(invalid()),
        };
        let next_observed = observed.checked_add(1).ok_or_else(invalid)?;
        let (_, payload) = self.read_frame_qualified_data(kind, admission)?;
        self.phase = match kind {
            SpillRecordKind::SortRow => RecordPhase::Sort {
                expected: expected.expect("sort declaration"),
                observed: next_observed,
            },
            SpillRecordKind::PartitionEntry => RecordPhase::Partition {
                expected: expected.expect("partition declaration"),
                observed: next_observed,
            },
            SpillRecordKind::AggregateState => RecordPhase::Aggregate {
                observed: next_observed,
            },
            _ => unreachable!("validated data kind"),
        };
        Ok(payload)
    }

    fn read_partition_start_qualified(&mut self) -> Result<u64, QualifiedReaderFailure> {
        let invalid = || {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "illegal qualified partition declaration",
            ))
        };
        if self.role != SpillFileRole::NativePartition
            || !matches!(self.phase, RecordPhase::NeedPartitionStart)
        {
            return Err(invalid());
        }
        let (_, payload) = self.read_frame_qualified_control(SpillRecordKind::PartitionStart)?;
        let entries = u64::from_le_bytes(payload.as_slice().try_into().map_err(|_| invalid())?);
        self.phase = RecordPhase::Partition {
            expected: entries,
            observed: 0,
        };
        Ok(entries)
    }

    /// Reads and validates the native-partition declaration.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong role/transition or malformed/truncated frame.
    pub fn read_partition_start(&mut self) -> std::io::Result<u64> {
        if self.role != SpillFileRole::NativePartition
            || !matches!(self.phase, RecordPhase::NeedPartitionStart)
        {
            return Err(invalid_data("illegal PartitionStart transition"));
        }
        let (_kind, payload) = self.read_frame(SpillRecordKind::PartitionStart)?;
        let entries = u64::from_le_bytes(
            payload
                .as_slice()
                .try_into()
                .map_err(|_| invalid_data("invalid partition-count payload"))?,
        );
        self.phase = RecordPhase::Partition {
            expected: entries,
            observed: 0,
        };
        Ok(entries)
    }

    /// Reads one bounded native partition-entry payload.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong transition/count or invalid framed record.
    pub fn read_partition_entry(&mut self) -> std::io::Result<Vec<u8>> {
        self.read_partition_entry_inner(None)
    }

    /// Reads one partition entry while handing every plaintext/stored
    /// allocation peak to a caller-owned accounting protocol before the bytes
    /// can escape this method.
    pub(crate) fn read_partition_entry_with_admission(
        &mut self,
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
    ) -> std::io::Result<Vec<u8>> {
        self.read_partition_entry_inner(Some(&mut admit))
    }

    fn read_partition_entry_inner(
        &mut self,
        admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
    ) -> std::io::Result<Vec<u8>> {
        let RecordPhase::Partition { expected, observed } = self.phase else {
            return Err(invalid_data("illegal PartitionEntry transition"));
        };
        if observed >= expected {
            return Err(invalid_data("more partition entries than declared"));
        }
        let (_kind, payload) =
            self.read_frame_inner(SpillRecordKind::PartitionEntry, admission, None, None)?;
        self.phase = RecordPhase::Partition {
            expected,
            observed: observed + 1,
        };
        Ok(payload)
    }

    /// Reads one reserved RDF aggregate-state payload.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong role/transition or invalid framed record.
    pub fn read_aggregate_state(&mut self) -> std::io::Result<Vec<u8>> {
        if self.role != SpillFileRole::RdfAggregateState {
            return Err(invalid_data("AggregateState in non-RDF spill file"));
        }
        let RecordPhase::Aggregate { observed } = self.phase else {
            return Err(invalid_data("illegal AggregateState transition"));
        };
        let (_kind, payload) = self.read_frame(SpillRecordKind::AggregateState)?;
        self.phase = RecordPhase::Aggregate {
            observed: observed + 1,
        };
        Ok(payload)
    }

    /// Validates declared counts, terminal record, and absence of trailing data.
    ///
    /// # Errors
    ///
    /// Returns an error for incomplete counts, an invalid `FileEnd`, trailing
    /// bytes, or prior reader poisoning.
    pub fn finish(&mut self) -> std::io::Result<()> {
        let _operation = ScalarReaderOperationGuard::new(self.open_record.scalar_cleanup.as_ref());
        if self.finished {
            return Ok(());
        }
        match self.phase {
            RecordPhase::Sort { expected, observed }
            | RecordPhase::Partition { expected, observed }
                if expected != observed =>
            {
                return Err(invalid_data(format!(
                    "declared {expected} records but observed {observed}"
                )));
            }
            RecordPhase::NeedFileStart
            | RecordPhase::NeedSortRunStart
            | RecordPhase::NeedPartitionStart => {
                return Err(invalid_data("missing role start record"));
            }
            RecordPhase::Ended => return Err(invalid_data("duplicate FileEnd")),
            _ => {}
        }
        let preceding_count = self.next_sequence;
        let (_kind, payload) = self.read_frame(SpillRecordKind::FileEnd)?;
        let declared = u64::from_le_bytes(
            payload
                .as_slice()
                .try_into()
                .map_err(|_| invalid_data("invalid FileEnd payload"))?,
        );
        if declared != preceding_count {
            self.poisoned = true;
            return Err(invalid_data(format!(
                "FileEnd count {declared} does not match {preceding_count} preceding records"
            )));
        }
        let mut trailing = [0u8; 1];
        match self.read_partition_buffered(&mut trailing) {
            Ok(0) => {}
            Ok(_) => {
                self.poisoned = true;
                return Err(invalid_data("trailing bytes after FileEnd"));
            }
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        }
        self.phase = RecordPhase::Ended;
        self.finished = true;
        Ok(())
    }

    fn finish_qualified(&mut self) -> Result<(), QualifiedReaderFailure> {
        if self.finished {
            return Ok(());
        }
        match self.phase {
            RecordPhase::Sort { expected, observed }
            | RecordPhase::Partition { expected, observed }
                if expected != observed =>
            {
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::DeclaredRecordCount {
                        declared: expected,
                        observed,
                    },
                ));
            }
            RecordPhase::NeedFileStart
            | RecordPhase::NeedSortRunStart
            | RecordPhase::NeedPartitionStart => {
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::InvalidData("missing role start record"),
                ));
            }
            RecordPhase::Ended => {
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::InvalidData("duplicate FileEnd"),
                ));
            }
            _ => {}
        }
        let preceding_count = self.next_sequence;
        let (_kind, payload) = self.read_frame_qualified_control(SpillRecordKind::FileEnd)?;
        let declared: [u8; 8] = payload.as_slice().try_into().map_err(|_| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "invalid FileEnd payload",
            ))
        })?;
        let declared = u64::from_le_bytes(declared);
        if declared != preceding_count {
            self.poisoned = true;
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::FileEndCount {
                    declared,
                    preceding: preceding_count,
                },
            ));
        }
        let mut trailing = [0u8; 1];
        match self.reader.read(&mut trailing) {
            Ok(0) => {}
            Ok(_) => {
                self.poisoned = true;
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::InvalidData("trailing bytes after FileEnd"),
                ));
            }
            Err(error) => {
                self.poisoned = true;
                return Err(QualifiedReaderFailure::Io(error));
            }
        }
        self.phase = RecordPhase::Ended;
        self.finished = true;
        Ok(())
    }

    fn read_partition_buffered(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        match self.partition_read_ahead.as_mut() {
            Some(buffer) => buffer.read(&mut self.reader, output),
            None => self.reader.read(output),
        }
    }

    fn read_partition_buffered_exact(&mut self, mut output: &mut [u8]) -> std::io::Result<()> {
        if self.partition_read_ahead.is_none() {
            return self.reader.read_exact(output);
        }
        while !output.is_empty() {
            match self.read_partition_buffered(output) {
                Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                Ok(count) => output = &mut output[count..],
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn partition_buffered_position(&mut self) -> std::io::Result<u64> {
        let physical = self.reader.stream_position()?;
        let unread = self
            .partition_read_ahead
            .as_ref()
            .map_or(0, PartitionReadAhead::unread);
        physical
            .checked_sub(unread as u64)
            .ok_or_else(|| std::io::ErrorKind::InvalidData.into())
    }

    fn read_frame(
        &mut self,
        expected_kind: SpillRecordKind,
    ) -> std::io::Result<(SpillRecordKind, Vec<u8>)> {
        let pre_admitted_provider_bound =
            fixed_plaintext_len(expected_kind).and(self.qualified_control_provider_bound);
        let pre_admitted_core_bound =
            fixed_plaintext_len(expected_kind).and(self.qualified_control_core_bound);
        self.read_frame_inner(
            expected_kind,
            None,
            pre_admitted_provider_bound,
            pre_admitted_core_bound,
        )
    }

    fn read_frame_qualified_control(
        &mut self,
        expected_kind: SpillRecordKind,
    ) -> Result<(SpillRecordKind, QualifiedFramePayload), QualifiedReaderFailure> {
        let provider_bound = fixed_plaintext_len(expected_kind)
            .and(self.qualified_control_provider_bound)
            .ok_or_else(|| {
                QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                    "qualified control-frame provider receipt is missing",
                ))
            })?;
        let core_bound = fixed_plaintext_len(expected_kind)
            .and(self.qualified_control_core_bound)
            .ok_or_else(|| {
                QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                    "qualified control-frame core receipt is missing",
                ))
            })?;
        self.read_frame_qualified(expected_kind, None, Some(provider_bound), Some(core_bound))
    }

    fn read_frame_qualified_data(
        &mut self,
        expected_kind: SpillRecordKind,
        admission: &mut dyn FnMut(usize) -> Result<(), QualifiedReaderFailure>,
    ) -> Result<(SpillRecordKind, QualifiedFramePayload), QualifiedReaderFailure> {
        self.read_frame_qualified(expected_kind, Some(admission), None, None)
    }

    fn read_frame_qualified(
        &mut self,
        expected_kind: SpillRecordKind,
        admission: Option<&mut dyn FnMut(usize) -> Result<(), QualifiedReaderFailure>>,
        pre_admitted_provider_bound: Option<usize>,
        pre_admitted_core_bound: Option<usize>,
    ) -> Result<(SpillRecordKind, QualifiedFramePayload), QualifiedReaderFailure> {
        if self.poisoned {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData(
                    "spill reader is poisoned by a prior consumed error",
                ),
            ));
        }
        self.validate_expected_transition_qualified(expected_kind)?;
        self.io
            .check(SpillIoOperation::ReadHeader)
            .map_err(QualifiedReaderFailure::Io)?;
        self.poisoned = true;
        let result = self.read_frame_after_header_consumption_qualified(
            expected_kind,
            admission,
            pre_admitted_provider_bound,
            pre_admitted_core_bound,
        );
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the qualified parser keeps every validation and allocation transition in one auditable sequence"
    )]
    fn read_frame_after_header_consumption_qualified(
        &mut self,
        expected_kind: SpillRecordKind,
        mut admission: Option<&mut dyn FnMut(usize) -> Result<(), QualifiedReaderFailure>>,
        pre_admitted_provider_bound: Option<usize>,
        pre_admitted_core_bound: Option<usize>,
    ) -> Result<(SpillRecordKind, QualifiedFramePayload), QualifiedReaderFailure> {
        let mut header = [0u8; SPILL_RECORD_HEADER_BYTES];
        self.reader
            .read_exact(&mut header)
            .map_err(QualifiedReaderFailure::Io)?;
        if header[..4] != SPILL_FILE_MAGIC {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData("missing GRSP spill magic"),
            ));
        }
        let version = u16::from_le_bytes([header[4], header[5]]);
        if version != SPILL_FORMAT_VERSION {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::UnsupportedFormatVersion(version),
            ));
        }
        let kind = match header[6] {
            0x01 => SpillRecordKind::FileStart,
            0x10 => SpillRecordKind::SortRunStart,
            0x11 => SpillRecordKind::SortRow,
            0x20 => SpillRecordKind::PartitionStart,
            0x21 => SpillRecordKind::PartitionEntry,
            0x30 => SpillRecordKind::AggregateState,
            0x7f => SpillRecordKind::FileEnd,
            value => {
                return Err(QualifiedReaderFailure::core(
                    QualifiedFrameCoreError::UnknownRecordKind(value),
                ));
            }
        };
        if kind != expected_kind {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::UnexpectedRecordKind {
                    expected: expected_kind,
                    actual: kind,
                },
            ));
        }
        let flags = header[7];
        if flags & !FLAG_SEALED != 0 {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidFlags(flags),
            ));
        }
        let record_sealed = flags & FLAG_SEALED != 0;
        if record_sealed != self.sealed {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData("spill provider/record sealed flag mismatch"),
            ));
        }
        let sequence = u64::from_le_bytes(
            header[8..16]
                .try_into()
                .expect("fixed eight-byte sequence field"),
        );
        if sequence != self.next_sequence {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::SequenceMismatch {
                    actual: sequence,
                    expected: self.next_sequence,
                },
            ));
        }
        let plaintext_len = u64::from_le_bytes(
            header[16..24]
                .try_into()
                .expect("fixed eight-byte plaintext field"),
        );
        let stored_len = u64::from_le_bytes(
            header[24..32]
                .try_into()
                .expect("fixed eight-byte stored field"),
        );
        if let Some(fixed_len) = fixed_plaintext_len(expected_kind)
            && plaintext_len != fixed_len
        {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::FixedPlaintextLength {
                    kind: expected_kind,
                    actual: plaintext_len,
                    expected: fixed_len,
                },
            ));
        }
        if plaintext_len > MAX_SPILL_RECORD_BYTES
            || stored_len > MAX_SPILL_RECORD_BYTES
            || plaintext_len > self.limits.max_plaintext_bytes
            || stored_len > self.limits.max_stored_bytes
        {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData(
                    "declared spill record length exceeds configured limit",
                ),
            ));
        }
        if plaintext_len > isize::MAX as u64 || stored_len > isize::MAX as u64 {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData(
                    "declared spill record length exceeds the platform allocation maximum",
                ),
            ));
        }
        if !record_sealed && plaintext_len != stored_len {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData("cleartext record lengths differ"),
            ));
        }
        let stored_len_usize = usize::try_from(stored_len).map_err(|_| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "stored spill length is not addressable",
            ))
        })?;
        let plaintext_len_usize = usize::try_from(plaintext_len).map_err(|_| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "plaintext spill length is not addressable",
            ))
        })?;
        let provider_stored_len = self
            .open_record
            .stored_len(plaintext_len_usize)
            .map_err(QualifiedReaderFailure::Io)?;
        if provider_stored_len != stored_len_usize {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::ProviderStoredLength {
                    provider: provider_stored_len,
                    declared: stored_len_usize,
                },
            ));
        }
        let payload_position = self
            .reader
            .stream_position()
            .map_err(QualifiedReaderFailure::Io)?;
        let file_len = hard_qualified_file_len(self.reader.get_ref())?;
        let available = file_len.checked_sub(payload_position).ok_or_else(|| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "spill reader position exceeds physical file length",
            ))
        })?;
        if stored_len > available {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::StoredBytesUnavailable {
                    declared: stored_len,
                    available,
                },
            ));
        }
        let provider_bound = if admission.is_some() {
            Some(
                self.open_record
                    .open_allocation_bound(stored_len_usize)
                    .ok_or_else(|| {
                        QualifiedReaderFailure::core(QualifiedFrameCoreError::Unsupported(
                            "spill provider does not declare a qualified open-allocation bound",
                        ))
                    })?,
            )
        } else {
            pre_admitted_provider_bound
        };
        let stored_layout = std::alloc::Layout::array::<u8>(stored_len_usize).map_err(|_| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "stored spill allocation layout overflow",
            ))
        })?;
        let plaintext_layout =
            std::alloc::Layout::array::<u8>(plaintext_len_usize).map_err(|_| {
                QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                    "plaintext spill allocation layout overflow",
                ))
            })?;
        let core_frame_bound = stored_layout
            .size()
            .checked_add(plaintext_layout.size())
            .ok_or_else(|| {
                QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                    "spill core frame-allocation peak overflow",
                ))
            })?;
        if let (Some(admit), Some(provider_bound)) = (admission.as_mut(), provider_bound) {
            let provisional_peak =
                core_frame_bound
                    .checked_add(provider_bound)
                    .ok_or_else(|| {
                        QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                            "spill record allocation peak overflow",
                        ))
                    })?;
            admit(provisional_peak)?;
        } else if let Some(admitted_core_bound) = pre_admitted_core_bound
            && core_frame_bound > admitted_core_bound
        {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData(
                    "fixed spill control frame exceeds its admitted core workspace",
                ),
            ));
        }
        let mut stored = ExactVec::new_in(Global);
        stored
            .try_reserve_exact(stored_len_usize)
            .map_err(|error| QualifiedReaderFailure::ExactAllocation {
                context: "stored spill record",
                error,
            })?;
        if stored.capacity() != stored_len_usize {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData(
                    "pinned exact stored-record allocator returned an unexpected capacity",
                ),
            ));
        }
        stored.resize(stored_len_usize, 0);
        self.io
            .check(SpillIoOperation::ReadPayload)
            .map_err(QualifiedReaderFailure::Io)?;
        self.reader
            .read_exact(stored.as_mut_slice())
            .map_err(QualifiedReaderFailure::Io)?;

        let expected_checksum = u32::from_le_bytes(
            header[32..36]
                .try_into()
                .expect("fixed four-byte checksum field"),
        );
        let mut checksum = crc32fast::Hasher::new();
        checksum.update(&header[..HEADER_AAD_BYTES]);
        checksum.update(stored.as_slice());
        if checksum.finalize() != expected_checksum {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData("spill record checksum mismatch"),
            ));
        }
        let meta = SpillRecordMeta {
            identity: self.identity,
            role: self.role,
            kind,
            sequence,
            sealed: record_sealed,
            plaintext_len,
            stored_len,
        };
        let aad: &[u8; HEADER_AAD_BYTES] = header[..HEADER_AAD_BYTES]
            .try_into()
            .expect("fixed 32-byte spill AAD");
        let mut plaintext = ExactVec::new_in(Global);
        plaintext
            .try_reserve_exact(plaintext_len_usize)
            .map_err(|error| QualifiedReaderFailure::ExactAllocation {
                context: "plaintext spill record",
                error,
            })?;
        if plaintext.capacity() != plaintext_len_usize {
            return Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::InvalidData(
                    "pinned exact plaintext allocator returned an unexpected capacity",
                ),
            ));
        }
        plaintext.resize(plaintext_len_usize, 0);
        let qualified_open = self
            .open_record
            .open_qualified_into(&meta, aad, stored.as_slice(), plaintext.as_mut_slice())
            .ok_or_else(|| {
                QualifiedReaderFailure::core(QualifiedFrameCoreError::Unsupported(
                    "spill provider does not implement exact qualified plaintext output",
                ))
            })?;
        qualified_open.map_err(QualifiedReaderFailure::Io)?;
        drop(stored);
        self.next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
            QualifiedReaderFailure::core(QualifiedFrameCoreError::InvalidData(
                "spill record sequence overflow",
            ))
        })?;
        Ok((kind, plaintext))
    }

    fn read_frame_inner(
        &mut self,
        expected_kind: SpillRecordKind,
        admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
        pre_admitted_provider_bound: Option<usize>,
        pre_admitted_core_bound: Option<usize>,
    ) -> std::io::Result<(SpillRecordKind, Vec<u8>)> {
        let _operation = ScalarReaderOperationGuard::new(self.open_record.scalar_cleanup.as_ref());
        if self.poisoned {
            return Err(invalid_data(
                "spill reader is poisoned by a prior consumed error",
            ));
        }
        self.validate_expected_transition(expected_kind)?;
        self.io.check(SpillIoOperation::ReadHeader)?;
        self.poisoned = true;
        let result = self.read_frame_after_header_consumption(
            expected_kind,
            admission,
            pre_admitted_provider_bound,
            pre_admitted_core_bound,
        );
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn read_frame_after_header_consumption(
        &mut self,
        expected_kind: SpillRecordKind,
        mut admission: Option<&mut dyn FnMut(usize) -> std::io::Result<()>>,
        pre_admitted_provider_bound: Option<usize>,
        pre_admitted_core_bound: Option<usize>,
    ) -> std::io::Result<(SpillRecordKind, Vec<u8>)> {
        let mut header = [0u8; SPILL_RECORD_HEADER_BYTES];
        self.read_partition_buffered_exact(&mut header)?;
        if header[..4] != SPILL_FILE_MAGIC {
            return Err(invalid_data("missing GRSP spill magic"));
        }
        let version = u16::from_le_bytes([header[4], header[5]]);
        if version != SPILL_FORMAT_VERSION {
            return Err(invalid_data(format!(
                "unsupported spill format version {version}"
            )));
        }
        let kind = SpillRecordKind::try_from(header[6])?;
        if kind != expected_kind {
            return Err(invalid_data(format!(
                "expected {expected_kind:?} record, found {kind:?}"
            )));
        }
        let flags = header[7];
        if flags & !FLAG_SEALED != 0 {
            return Err(invalid_data(format!(
                "invalid spill record flags {flags:#04x}"
            )));
        }
        let record_sealed = flags & FLAG_SEALED != 0;
        if record_sealed != self.sealed {
            return Err(invalid_data("spill provider/record sealed flag mismatch"));
        }
        let sequence = u64::from_le_bytes(
            header[8..16]
                .try_into()
                .expect("fixed eight-byte sequence field"),
        );
        if sequence != self.next_sequence {
            return Err(invalid_data(format!(
                "spill sequence {sequence} does not match expected {}",
                self.next_sequence
            )));
        }
        let plaintext_len = u64::from_le_bytes(
            header[16..24]
                .try_into()
                .expect("fixed eight-byte plaintext field"),
        );
        let stored_len = u64::from_le_bytes(
            header[24..32]
                .try_into()
                .expect("fixed eight-byte stored field"),
        );
        if let Some(fixed_len) = fixed_plaintext_len(expected_kind)
            && plaintext_len != fixed_len
        {
            return Err(invalid_data(format!(
                "{expected_kind:?} plaintext length {plaintext_len} does not match {fixed_len}"
            )));
        }
        if plaintext_len > MAX_SPILL_RECORD_BYTES
            || stored_len > MAX_SPILL_RECORD_BYTES
            || plaintext_len > self.limits.max_plaintext_bytes
            || stored_len > self.limits.max_stored_bytes
        {
            return Err(invalid_data(
                "declared spill record length exceeds configured limit",
            ));
        }
        validate_platform_record_lengths(plaintext_len, stored_len, isize::MAX as u64)?;
        if !record_sealed && plaintext_len != stored_len {
            return Err(invalid_data("cleartext record lengths differ"));
        }
        let stored_len_usize = usize::try_from(stored_len)
            .map_err(|_| invalid_data("stored spill length is not addressable"))?;
        let plaintext_len_usize = usize::try_from(plaintext_len)
            .map_err(|_| invalid_data("plaintext spill length is not addressable"))?;
        let provider_stored_len = self.open_record.stored_len(plaintext_len_usize)?;
        if provider_stored_len != stored_len_usize {
            return Err(invalid_data(format!(
                "provider expects {provider_stored_len} stored bytes, header declares {stored_len_usize}"
            )));
        }
        let payload_position = self.partition_buffered_position()?;
        let file_len = self.reader.get_ref().metadata()?.len();
        let available = file_len
            .checked_sub(payload_position)
            .ok_or_else(|| invalid_data("spill reader position exceeds physical file length"))?;
        if stored_len > available {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "spill record declares {stored_len} stored bytes but only {available} remain"
                ),
            ));
        }
        let provider_bound = if admission.is_some() {
            Some(
                self.open_record
                    .open_allocation_bound(stored_len_usize)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::Unsupported,
                            "spill provider does not declare a qualified open-allocation bound",
                        )
                    })?,
            )
        } else {
            pre_admitted_provider_bound
        };
        let core_stored_bound = stored_len_usize
            .checked_mul(2)
            .ok_or_else(|| invalid_data("stored spill allocation bound overflow"))?;
        if let (Some(admit), Some(provider_bound)) = (admission.as_mut(), provider_bound) {
            let provisional_peak = core_stored_bound
                .checked_add(provider_bound)
                .ok_or_else(|| invalid_data("spill record allocation peak overflow"))?;
            admit(provisional_peak)?;
        } else if let Some(admitted_core_bound) = pre_admitted_core_bound
            && core_stored_bound > admitted_core_bound
        {
            return Err(invalid_data(
                "fixed spill control frame exceeds its admitted core workspace",
            ));
        }
        let mut stored = Vec::new();
        stored
            .try_reserve_exact(stored_len_usize)
            .map_err(|error| allocation_error("stored spill record", error))?;
        stored.resize(stored_len_usize, 0);
        if let (Some(admit), Some(provider_bound)) = (admission.as_mut(), provider_bound)
            && stored.capacity() > core_stored_bound
        {
            let observed_peak = stored
                .capacity()
                .checked_add(provider_bound)
                .ok_or_else(|| invalid_data("spill record allocation peak overflow"))?;
            if let Err(error) = admit(observed_peak) {
                drop(stored);
                return Err(error);
            }
        } else if let Some(admitted_core_bound) = pre_admitted_core_bound
            && stored.capacity() > admitted_core_bound
        {
            drop(stored);
            return Err(invalid_data(
                "fixed spill control buffer exceeded its admitted capacity",
            ));
        }
        self.io.check(SpillIoOperation::ReadPayload)?;
        self.read_partition_buffered_exact(&mut stored)?;

        let expected_checksum = u32::from_le_bytes(
            header[32..36]
                .try_into()
                .expect("fixed four-byte checksum field"),
        );
        let mut checksum = crc32fast::Hasher::new();
        checksum.update(&header[..HEADER_AAD_BYTES]);
        checksum.update(&stored);
        if checksum.finalize() != expected_checksum {
            return Err(invalid_data("spill record checksum mismatch"));
        }
        let meta = SpillRecordMeta {
            identity: self.identity,
            role: self.role,
            kind,
            sequence,
            sealed: record_sealed,
            plaintext_len,
            stored_len,
        };
        let aad: &[u8; HEADER_AAD_BYTES] = header[..HEADER_AAD_BYTES]
            .try_into()
            .expect("fixed 32-byte spill AAD");
        let plaintext = self.open_record.open(&meta, aad, &stored)?;
        if let Some(provider_bound) = provider_bound
            && plaintext.capacity() > provider_bound
        {
            return Err(invalid_data(format!(
                "spill provider returned capacity {}, exceeding its declared {provider_bound}-byte open bound",
                plaintext.capacity()
            )));
        }
        if plaintext.len() != plaintext_len_usize {
            return Err(invalid_data(format!(
                "spill provider returned {} plaintext bytes, expected {plaintext_len_usize}",
                plaintext.len()
            )));
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid_data("spill record sequence overflow"))?;
        Ok((kind, plaintext))
    }

    fn validate_expected_transition(&self, expected_kind: SpillRecordKind) -> std::io::Result<()> {
        let legal = match expected_kind {
            SpillRecordKind::FileStart => {
                self.next_sequence == 0 && matches!(self.phase, RecordPhase::NeedFileStart)
            }
            SpillRecordKind::SortRunStart => {
                self.role == SpillFileRole::SortRun
                    && self.next_sequence == 1
                    && matches!(self.phase, RecordPhase::NeedSortRunStart)
            }
            SpillRecordKind::SortRow => matches!(
                self.phase,
                RecordPhase::Sort { expected, observed } if observed < expected
            ),
            SpillRecordKind::PartitionStart => {
                self.role == SpillFileRole::NativePartition
                    && self.next_sequence == 1
                    && matches!(self.phase, RecordPhase::NeedPartitionStart)
            }
            SpillRecordKind::PartitionEntry => matches!(
                self.phase,
                RecordPhase::Partition { expected, observed } if observed < expected
            ),
            SpillRecordKind::AggregateState => {
                self.role == SpillFileRole::RdfAggregateState
                    && matches!(self.phase, RecordPhase::Aggregate { .. })
            }
            SpillRecordKind::FileEnd => {
                matches!(
                    self.phase,
                    RecordPhase::Sort { expected, observed }
                        | RecordPhase::Partition { expected, observed }
                        if expected == observed
                ) || matches!(self.phase, RecordPhase::Aggregate { .. })
            }
        };
        if legal {
            Ok(())
        } else {
            Err(invalid_data(format!(
                "illegal {expected_kind:?} transition for spill reader"
            )))
        }
    }

    fn validate_expected_transition_qualified(
        &self,
        expected_kind: SpillRecordKind,
    ) -> Result<(), QualifiedReaderFailure> {
        let legal = match expected_kind {
            SpillRecordKind::FileStart => {
                self.next_sequence == 0 && matches!(self.phase, RecordPhase::NeedFileStart)
            }
            SpillRecordKind::SortRunStart => {
                self.role == SpillFileRole::SortRun
                    && self.next_sequence == 1
                    && matches!(self.phase, RecordPhase::NeedSortRunStart)
            }
            SpillRecordKind::SortRow => matches!(
                self.phase,
                RecordPhase::Sort { expected, observed } if observed < expected
            ),
            SpillRecordKind::PartitionStart => {
                self.role == SpillFileRole::NativePartition
                    && self.next_sequence == 1
                    && matches!(self.phase, RecordPhase::NeedPartitionStart)
            }
            SpillRecordKind::PartitionEntry => matches!(
                self.phase,
                RecordPhase::Partition { expected, observed } if observed < expected
            ),
            SpillRecordKind::AggregateState => {
                self.role == SpillFileRole::RdfAggregateState
                    && matches!(self.phase, RecordPhase::Aggregate { .. })
            }
            SpillRecordKind::FileEnd => {
                matches!(
                    self.phase,
                    RecordPhase::Sort { expected, observed }
                        | RecordPhase::Partition { expected, observed }
                        if expected == observed
                ) || matches!(self.phase, RecordPhase::Aggregate { .. })
            }
        };
        if legal {
            Ok(())
        } else {
            Err(QualifiedReaderFailure::core(
                QualifiedFrameCoreError::IllegalTransition(expected_kind),
            ))
        }
    }
}

pub(super) fn validate_platform_record_lengths(
    plaintext_len: u64,
    stored_len: u64,
    platform_allocation_max: u64,
) -> std::io::Result<()> {
    if plaintext_len > platform_allocation_max || stored_len > platform_allocation_max {
        return Err(invalid_data(
            "declared spill record length exceeds the platform allocation maximum",
        ));
    }
    Ok(())
}

const fn fixed_plaintext_len(kind: SpillRecordKind) -> Option<u64> {
    match kind {
        SpillRecordKind::FileStart => Some(8),
        SpillRecordKind::SortRunStart => Some(12),
        SpillRecordKind::PartitionStart | SpillRecordKind::FileEnd => Some(8),
        SpillRecordKind::SortRow
        | SpillRecordKind::PartitionEntry
        | SpillRecordKind::AggregateState => None,
    }
}

impl std::fmt::Debug for SpillFileReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpillFileReader")
            .field("identity", &self.identity)
            .field("role", &self.role)
            .field("next_sequence", &self.next_sequence)
            .field("finished", &self.finished)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

fn encode_aad(meta: SpillRecordMeta) -> [u8; HEADER_AAD_BYTES] {
    let mut header = [0u8; HEADER_AAD_BYTES];
    header[..4].copy_from_slice(&SPILL_FILE_MAGIC);
    header[4..6].copy_from_slice(&SPILL_FORMAT_VERSION.to_le_bytes());
    header[6] = meta.kind as u8;
    header[7] = if meta.sealed { FLAG_SEALED } else { 0 };
    header[8..16].copy_from_slice(&meta.sequence.to_le_bytes());
    header[16..24].copy_from_slice(&meta.plaintext_len.to_le_bytes());
    header[24..32].copy_from_slice(&meta.stored_len.to_le_bytes());
    header
}

/// Fallible, explicitly bounded staging storage for one plaintext record.
pub(crate) struct SpillRecordBuffer {
    bytes: Vec<u8>,
    maximum: usize,
    #[cfg(test)]
    write_growths: usize,
}

impl SpillRecordBuffer {
    pub(crate) fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
            #[cfg(test)]
            write_growths: 0,
        }
    }

    pub(crate) fn into_inner(self) -> Vec<u8> {
        self.bytes
    }

    fn clear(&mut self) {
        self.bytes.clear();
    }

    pub(crate) fn prepare_record(&mut self, required_len: usize) -> std::io::Result<usize> {
        self.clear();
        if required_len > self.maximum {
            return Err(invalid_input(format!(
                "spill record staging length {required_len} exceeds maximum {}",
                self.maximum
            )));
        }
        if required_len > self.bytes.capacity() {
            self.bytes
                .try_reserve_exact(required_len)
                .map_err(|error| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        format!("failed to reserve plaintext spill record: {error}"),
                    )
                })?;
        }
        Ok(self.bytes.capacity())
    }

    pub(crate) fn record_scope(
        &mut self,
        prepared_len: usize,
    ) -> std::io::Result<SpillRecordScope<'_>> {
        self.clear();
        if prepared_len > self.maximum {
            return Err(invalid_input(format!(
                "prepared spill record length {prepared_len} exceeds maximum {}",
                self.maximum
            )));
        }
        if prepared_len > self.bytes.capacity() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!(
                    "prepared spill record length {prepared_len} exceeds available capacity {}",
                    self.bytes.capacity()
                ),
            ));
        }
        Ok(SpillRecordScope {
            buffer: self,
            prepared_len,
        })
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn discard_capacity(&mut self) {
        self.bytes = Vec::new();
    }

    pub(crate) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub(crate) const fn maximum(&self) -> usize {
        self.maximum
    }

    #[cfg(test)]
    pub(crate) fn write_growths(&self) -> usize {
        self.write_growths
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn patch_u64_le(&mut self, offset: usize, value: u64) -> std::io::Result<()> {
        let end = offset
            .checked_add(8)
            .ok_or_else(|| invalid_input("spill record patch offset overflow"))?;
        let destination = self
            .bytes
            .get_mut(offset..end)
            .ok_or_else(|| invalid_input("spill record patch is out of bounds"))?;
        destination.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}

/// One unwind-safe use of reusable plaintext record staging.
///
/// Dropping the scope restores a logically empty buffer while preserving its
/// allocation for the next record.
pub(crate) struct SpillRecordScope<'a> {
    buffer: &'a mut SpillRecordBuffer,
    prepared_len: usize,
}

impl SpillRecordScope<'_> {
    pub(crate) fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

impl Write for SpillRecordScope<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let new_len = self
            .buffer
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| invalid_input("prepared spill record length overflow"))?;
        if new_len > self.prepared_len {
            return Err(invalid_input(format!(
                "spill record length {new_len} exceeds prepared length {}",
                self.prepared_len
            )));
        }
        if new_len > self.buffer.bytes.capacity() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "prepared spill record capacity invariant violated",
            ));
        }
        self.buffer.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for SpillRecordScope<'_> {
    fn drop(&mut self) {
        self.buffer.clear();
    }
}

impl Write for SpillRecordBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let new_len = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| invalid_input("spill record staging length overflow"))?;
        if new_len > self.maximum {
            return Err(invalid_input(format!(
                "spill record staging length {new_len} exceeds maximum {}",
                self.maximum
            )));
        }
        if new_len > self.bytes.capacity() {
            #[cfg(test)]
            {
                self.write_growths = self.write_growths.saturating_add(1);
            }
            let doubled = self
                .bytes
                .capacity()
                .checked_mul(2)
                .unwrap_or(self.maximum)
                .max(8)
                .min(self.maximum);
            let target_capacity = new_len.max(doubled).min(self.maximum);
            self.bytes
                .try_reserve_exact(target_capacity.saturating_sub(self.bytes.len()))
                .map_err(|error| allocation_error("plaintext spill record", error))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(all(test, unix, not(target_os = "wasi"), not(target_arch = "wasm32")))]
mod exact_reader_qualification_tests {
    use super::super::manager::SpillManager;
    use super::{
        CleartextSpillRecordProvider, NoopSpillIo, OpenSpillRecord, ProviderAccountedReaderError,
        ProviderAccountedReaderPrimary, ProviderAccountedReaderResolutionError,
        ProviderAccountedReaderUnusedPublicationError, ProviderGrantTransitionObserver,
        SpillFileIdentity, SpillFileRole, SpillFrameLimits, SpillIo, SpillIoOperation,
        SpillRecordProvider, fail_next_unused_publisher_release_for_test,
    };
    use grafeo_common::memory::buffer::{
        AccountedErrorPublisher, BufferManager, BufferManagerConfig, MemoryGrantError,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingProvider {
        support_queries: AtomicUsize,
        bound_queries: AtomicUsize,
        workspace_bound: Option<usize>,
    }

    impl CountingProvider {
        fn qualified() -> Self {
            Self {
                support_queries: AtomicUsize::new(0),
                bound_queries: AtomicUsize::new(0),
                workspace_bound: CleartextSpillRecordProvider.file_workspace_allocation_bound(),
            }
        }

        fn missing_bound() -> Self {
            Self {
                workspace_bound: None,
                ..Self::qualified()
            }
        }
    }

    impl SpillRecordProvider for CountingProvider {
        fn seals(&self) -> bool {
            false
        }

        fn begin_file(
            &self,
            identity: SpillFileIdentity,
        ) -> std::io::Result<Box<dyn OpenSpillRecord>> {
            CleartextSpillRecordProvider.begin_file(identity)
        }

        fn file_workspace_allocation_bound(&self) -> Option<usize> {
            self.bound_queries.fetch_add(1, Ordering::Relaxed);
            self.workspace_bound
        }

        fn supports_qualified_exact_open(&self) -> bool {
            self.support_queries.fetch_add(1, Ordering::Relaxed);
            true
        }
    }

    struct CountingIo {
        bound_queries: AtomicUsize,
        workspace_bound: Option<usize>,
        fail_read_open: bool,
    }

    impl CountingIo {
        fn qualified() -> Self {
            Self {
                bound_queries: AtomicUsize::new(0),
                workspace_bound: Some(0),
                fail_read_open: false,
            }
        }

        fn missing_bound() -> Self {
            Self {
                bound_queries: AtomicUsize::new(0),
                workspace_bound: None,
                fail_read_open: false,
            }
        }

        fn failing_read_open() -> Self {
            Self {
                fail_read_open: true,
                ..Self::qualified()
            }
        }
    }

    impl SpillIo for CountingIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            if self.fail_read_open && operation == SpillIoOperation::ReadOpen {
                return Err(std::io::ErrorKind::PermissionDenied.into());
            }
            NoopSpillIo.check(operation)
        }

        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            self.bound_queries.fetch_add(1, Ordering::Relaxed);
            self.workspace_bound
        }
    }

    fn publish_empty_sort_run(manager: &SpillManager) -> super::SpillFile {
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        file
    }

    fn grant(bytes: usize) -> (Arc<BufferManager>, super::MemoryGrant) {
        let mut config = BufferManagerConfig::with_budget(1024 * 1024);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        let manager = BufferManager::new(config);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let grant = resources.try_allocate(bytes).unwrap();
        (manager, grant)
    }

    fn publish_native_partition(manager: &SpillManager, payloads: &[Vec<u8>]) -> super::SpillFile {
        let mut file = manager.create_file(SpillFileRole::NativePartition).unwrap();
        file.write_partition_start(u64::try_from(payloads.len()).unwrap())
            .unwrap();
        for payload in payloads {
            file.write_partition_entry(payload).unwrap();
        }
        file.finish_write().unwrap();
        file
    }

    fn partition_cleanup(
        workspace: &mut super::MemoryGrant,
    ) -> grafeo_common::memory::buffer::AccountedError {
        AccountedErrorPublisher::try_new(workspace.split(0).unwrap())
            .unwrap()
            .publish(super::ScalarReaderCleanup::new())
    }

    #[test]
    fn partition_adapter_read_ahead_crosses_boundaries_and_checks_buffered_trailing_bytes() {
        use std::io::Write as _;
        for trailing in [false, true] {
            let directory = tempfile::TempDir::new().unwrap();
            let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .build()
                .unwrap();
            let payloads = [
                vec![1; 100],
                vec![2; 100],
                vec![3; 100],
                vec![4; 100],
                Vec::new(),
                vec![5; 600],
                vec![6; 100],
            ];
            let mut file = publish_native_partition(&manager, &payloads);
            if trailing {
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(&file.path)
                    .unwrap()
                    .write_all(&[0xff])
                    .unwrap();
            }
            let (memory, mut workspace) = grant(0);
            let mut frame = workspace.split(0).unwrap();
            let cleanup = partition_cleanup(&mut workspace);
            let mut reader = file
                .reader_with_partition_admission_and_cleanup(
                    |bytes| workspace.try_resize(bytes).map_err(std::io::Error::other),
                    cleanup.clone(),
                )
                .unwrap();
            let buffer = reader.partition_read_ahead.as_ref().unwrap();
            assert_eq!(
                buffer.bytes.capacity(),
                super::qualified_partition_reader_buffer_requested_bytes()
            );
            assert_eq!(buffer.filled, 512);
            assert!(buffer.unread() > 0);
            let reader_bytes = workspace.size();
            assert!(reader_bytes >= 512);
            assert_eq!(
                reader.read_partition_start().unwrap(),
                u64::try_from(payloads.len()).unwrap()
            );
            for expected in &payloads {
                let row = reader
                    .read_partition_entry_with_admission(|bytes| {
                        frame.try_resize(bytes).map_err(std::io::Error::other)
                    })
                    .unwrap();
                assert_eq!(row, *expected);
                drop(row);
                frame.try_resize(0).unwrap();
                assert_eq!(
                    workspace.size(),
                    reader_bytes,
                    "frame retirement cannot release read-ahead authority"
                );
            }
            let result = reader.finish();
            if trailing {
                assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidData);
            } else {
                result.unwrap();
            }
            assert!(reader.teardown_for_accounted_failure());
            assert_eq!(memory.allocated(), reader_bytes + cleanup.granted_bytes());
            drop(workspace);
            drop(frame);
            drop(cleanup);
            assert_eq!(memory.allocated(), 0);
            // Ordinary/borrowed and hard-owned partition entry points stay unchanged.
            assert!(file.reader().unwrap().partition_read_ahead.is_none());
            let (_, unbuffered_workspace) = grant(0);
            let unbuffered = file
                .reader_with_owned_provider_admission(unbuffered_workspace)
                .unwrap();
            assert!(
                unbuffered
                    .reader
                    .as_ref()
                    .unwrap()
                    .partition_read_ahead
                    .is_none()
            );
            drop(unbuffered);
            file.close_and_delete().unwrap();
        }
    }

    #[test]
    fn partition_adapter_read_ahead_rejects_prefetched_truncation_and_retains_workspace() {
        for (second_len, cut, finish, kind) in [
            (100, 259, false, std::io::ErrorKind::InvalidData),
            (0, 259, false, std::io::ErrorKind::InvalidData),
            (100, 359, false, std::io::ErrorKind::UnexpectedEof),
            (100, 403, true, std::io::ErrorKind::UnexpectedEof),
        ] {
            for cleanup_first in [false, true] {
                let directory = tempfile::TempDir::new().unwrap();
                let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                    .build()
                    .unwrap();
                let mut file =
                    publish_native_partition(&manager, &[vec![1; 100], vec![2; second_len]]);
                let (memory, mut workspace) = grant(0);
                let mut frame = workspace.split(0).unwrap();
                let cleanup = partition_cleanup(&mut workspace);
                let mut reader = file
                    .reader_with_partition_admission_and_cleanup(
                        |bytes| workspace.try_resize(bytes).map_err(std::io::Error::other),
                        cleanup.clone(),
                    )
                    .unwrap();
                assert_eq!(reader.read_partition_start().unwrap(), 2);
                let row = reader
                    .read_partition_entry_with_admission(|bytes| {
                        frame.try_resize(bytes).map_err(std::io::Error::other)
                    })
                    .unwrap();
                assert_eq!(row, vec![1; 100]);
                drop(row);
                frame.try_resize(0).unwrap();
                assert!(reader.partition_read_ahead.as_ref().unwrap().unread() > 0);
                if finish {
                    let row = reader
                        .read_partition_entry_with_admission(|bytes| {
                            frame.try_resize(bytes).map_err(std::io::Error::other)
                        })
                        .unwrap();
                    drop(row);
                    frame.try_resize(0).unwrap();
                }
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&file.path)
                    .unwrap()
                    .set_len(cut)
                    .unwrap();
                let mut admission_calls = 0;
                let error = if finish {
                    reader.finish().unwrap_err()
                } else {
                    reader
                        .read_partition_entry_with_admission(|bytes| {
                            admission_calls += 1;
                            frame.try_resize(bytes).map_err(std::io::Error::other)
                        })
                        .unwrap_err()
                };
                assert_eq!(error.kind(), kind);
                assert_eq!(
                    admission_calls, 0,
                    "fresh EOF check precedes frame allocation"
                );
                assert!(reader.poisoned);
                drop(error);
                let reader_bytes = workspace.size();
                assert!(reader_bytes >= 512);
                cleanup
                    .inspect::<super::ScalarReaderCleanup, _>(|owner| {
                        owner.retain_workspaces([Some(workspace), Some(frame), None]);
                    })
                    .unwrap();
                if cleanup_first {
                    drop(cleanup);
                    assert!(memory.allocated() >= reader_bytes);
                    assert!(reader.teardown_for_accounted_failure());
                } else {
                    assert!(reader.teardown_for_accounted_failure());
                    assert!(memory.allocated() >= reader_bytes);
                    drop(cleanup);
                }
                assert_eq!(memory.allocated(), 0);
                file.close_and_delete().unwrap();
            }
        }
    }

    #[test]
    fn partition_adapter_read_ahead_denial_precedes_allocation_and_header_read() {
        struct HeaderCounter(AtomicUsize);
        impl SpillIo for HeaderCounter {
            fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
                if operation == SpillIoOperation::ReadHeader {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            }
            fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
                Some(0)
            }
        }
        let directory = tempfile::TempDir::new().unwrap();
        let io = Arc::new(HeaderCounter(AtomicUsize::new(0)));
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .io(io.clone())
            .build()
            .unwrap();
        let mut file = publish_native_partition(&manager, &[]);
        let (memory, mut workspace) = grant(0);
        let cleanup = partition_cleanup(&mut workspace);
        let reader = file
            .reader_with_partition_admission_and_cleanup(
                |bytes| workspace.try_resize(bytes).map_err(std::io::Error::other),
                cleanup.clone(),
            )
            .unwrap();
        let required = workspace.size();
        assert!(required >= 512);
        assert!(reader.teardown_for_accounted_failure());
        workspace.try_resize(0).unwrap();
        let held = memory
            .try_allocate(
                (1 << 20) - cleanup.granted_bytes() - required + 1,
                grafeo_common::memory::buffer::MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        io.0.store(0, Ordering::Relaxed);
        let mut requested = Vec::new();
        let result = file.reader_with_partition_admission_and_cleanup(
            |bytes| {
                requested.push(bytes);
                workspace
                    .try_resize(bytes)
                    .map_err(|_| std::io::Error::from(std::io::ErrorKind::OutOfMemory))
            },
            cleanup.clone(),
        );
        let Err(error) = result else {
            panic!("buffer and control admission must fail before allocation");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(requested[0] >= 512);
        assert_eq!(requested.last(), Some(&required));
        assert_eq!(io.0.load(Ordering::Relaxed), 0);
        drop(error);
        drop(workspace);
        drop(cleanup);
        assert_eq!(memory.allocated(), held.size());
        drop(held);
        assert_eq!(memory.allocated(), 0);
        file.close_and_delete().unwrap();
    }

    struct NoopGrantObserver;

    impl ProviderGrantTransitionObserver for NoopGrantObserver {
        fn replace_reader(
            &self,
            _previous: usize,
            _current: usize,
        ) -> Result<(), MemoryGrantError> {
            Ok(())
        }

        fn replace_row(&self, _previous: usize, _current: usize) -> Result<(), MemoryGrantError> {
            Ok(())
        }

        fn publish_unrepresentable(&self) {}
    }

    #[test]
    fn staged_probe_does_not_poison_published_qualification_and_callbacks_run_once() {
        let directory = tempfile::TempDir::new().unwrap();
        let provider = Arc::new(CountingProvider::qualified());
        let io = Arc::new(CountingIo::qualified());
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(provider.clone(), SpillFrameLimits::format_max())
            .io(io.clone())
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();

        assert!(file.exact_owned_reader_qualification().is_none());
        assert_eq!(provider.support_queries.load(Ordering::Relaxed), 0);
        assert_eq!(provider.bound_queries.load(Ordering::Relaxed), 0);
        assert_eq!(io.bound_queries.load(Ordering::Relaxed), 0);

        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        for _ in 0..3 {
            drop(file.exact_owned_reader_qualification().unwrap());
        }

        assert_eq!(provider.support_queries.load(Ordering::Relaxed), 1);
        assert_eq!(provider.bound_queries.load(Ordering::Relaxed), 1);
        assert_eq!(io.bound_queries.load(Ordering::Relaxed), 1);
        file.close_and_delete().unwrap();
    }

    #[test]
    fn missing_provider_or_hook_bound_is_cached_as_compatibility_only() {
        let provider_directory = tempfile::TempDir::new().unwrap();
        let missing_provider = Arc::new(CountingProvider::missing_bound());
        let unused_io = Arc::new(CountingIo::qualified());
        let provider_manager =
            crate::execution::spill::BorrowedSpillFixture::new(provider_directory.path())
                .provider(missing_provider.clone(), SpillFrameLimits::format_max())
                .io(unused_io.clone())
                .build()
                .unwrap();
        let mut provider_file = publish_empty_sort_run(&provider_manager);
        for _ in 0..3 {
            assert!(provider_file.exact_owned_reader_qualification().is_none());
        }
        assert_eq!(missing_provider.support_queries.load(Ordering::Relaxed), 1);
        assert_eq!(missing_provider.bound_queries.load(Ordering::Relaxed), 1);
        assert_eq!(unused_io.bound_queries.load(Ordering::Relaxed), 0);
        provider_file.close_and_delete().unwrap();

        let hook_directory = tempfile::TempDir::new().unwrap();
        let qualified_provider = Arc::new(CountingProvider::qualified());
        let missing_io = Arc::new(CountingIo::missing_bound());
        let hook_manager =
            crate::execution::spill::BorrowedSpillFixture::new(hook_directory.path())
                .provider(qualified_provider.clone(), SpillFrameLimits::format_max())
                .io(missing_io.clone())
                .build()
                .unwrap();
        let mut hook_file = publish_empty_sort_run(&hook_manager);
        for _ in 0..3 {
            assert!(hook_file.exact_owned_reader_qualification().is_none());
        }
        assert_eq!(
            qualified_provider.support_queries.load(Ordering::Relaxed),
            1
        );
        assert_eq!(qualified_provider.bound_queries.load(Ordering::Relaxed), 1);
        assert_eq!(missing_io.bound_queries.load(Ordering::Relaxed), 1);
        hook_file.close_and_delete().unwrap();
    }

    #[test]
    fn nonempty_workspace_precedes_qualification_without_invoking_callbacks() {
        let directory = tempfile::TempDir::new().unwrap();
        let provider = Arc::new(CountingProvider::qualified());
        let io = Arc::new(CountingIo::qualified());
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(provider.clone(), SpillFrameLimits::format_max())
            .io(io.clone())
            .build()
            .unwrap();
        let mut file = publish_empty_sort_run(&manager);
        let (memory, workspace) = grant(1);

        let error = file
            .reader_with_owned_provider_admission(workspace)
            .unwrap_err();

        assert!(matches!(
            error.primary(),
            Some(ProviderAccountedReaderPrimary::NonEmptyWorkspace { actual_bytes: 1 })
        ));
        assert_eq!(provider.support_queries.load(Ordering::Relaxed), 0);
        assert_eq!(provider.bound_queries.load(Ordering::Relaxed), 0);
        assert_eq!(io.bound_queries.load(Ordering::Relaxed), 0);
        assert_eq!(memory.allocated(), 1);
        drop(error);
        assert_eq!(memory.allocated(), 0);
        file.close_and_delete().unwrap();
    }

    #[test]
    fn exact_qualification_is_identity_bound_without_requerying_foreign_file() {
        let directory = tempfile::TempDir::new().unwrap();
        let provider = Arc::new(CountingProvider::qualified());
        let io = Arc::new(CountingIo::qualified());
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(provider.clone(), SpillFrameLimits::format_max())
            .io(io.clone())
            .build()
            .unwrap();
        let mut first = publish_empty_sort_run(&manager);
        let mut second = publish_empty_sort_run(&manager);
        let qualification = first.exact_owned_reader_qualification().unwrap();
        let (memory, workspace) = grant(0);

        let error = second
            .reader_with_owned_provider_qualification_inner(qualification, workspace, None)
            .unwrap_err();

        assert!(matches!(
            error.primary(),
            Some(ProviderAccountedReaderPrimary::InvalidQualificationIdentity)
        ));
        assert_eq!(provider.support_queries.load(Ordering::Relaxed), 1);
        assert_eq!(provider.bound_queries.load(Ordering::Relaxed), 1);
        assert_eq!(io.bound_queries.load(Ordering::Relaxed), 1);
        assert_eq!(memory.allocated(), 0);
        drop(error);
        first.close_and_delete().unwrap();
        second.close_and_delete().unwrap();
    }

    #[test]
    fn exact_reader_success_retains_then_explicitly_resolves_unused_publishers() {
        let directory = tempfile::TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();
        let mut file = publish_empty_sort_run(&manager);
        let qualification = file.exact_owned_reader_qualification().unwrap();
        let mut config = BufferManagerConfig::with_budget(1024 * 1024);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        let memory = BufferManager::new(config);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&memory)).unwrap();
        let workspace = resources.try_allocate(0).unwrap();
        let failure_publisher = AccountedErrorPublisher::<ProviderAccountedReaderError>::try_new(
            resources.try_allocate(0).unwrap(),
        )
        .unwrap();
        let construction_publication_bytes = failure_publisher.granted_bytes();

        let reader = file
            .reader_with_exact_owned_provider_admission(
                qualification,
                workspace,
                failure_publisher,
                &NoopGrantObserver,
            )
            .unwrap()
            .read_sort_run_start_owned(1, 0)
            .unwrap();
        assert!(reader.receipt().bytes() > construction_publication_bytes);
        assert_eq!(
            memory.allocated(),
            reader.receipt().bytes(),
            "reader receipt includes both live publisher grants and its workspace"
        );

        let refusal =
            fail_next_unused_publisher_release_for_test(MemoryGrantError::AccountingPoisoned {
                account: "unused-reader-publication-test",
            });
        let error = reader
            .finish_sort_run_owned()
            .expect_err("injected unused operation-publisher release must escape");
        drop(refusal);
        let ProviderAccountedReaderResolutionError::UnusedPublication(error) = error else {
            panic!("unused operation publisher must retain direct retry authority");
        };
        assert!(matches!(
            error.error(),
            MemoryGrantError::AccountingPoisoned {
                account: "unused-reader-publication-test"
            }
        ));
        assert_eq!(memory.allocated(), error.granted_bytes());
        let (_primary, mut grant) = error.into_parts();
        grant
            .try_resize(0)
            .expect("caller can retry the retained publication grant release");
        assert_eq!(memory.allocated(), 0);
        file.close_and_delete().unwrap();
    }

    #[test]
    fn construction_primary_survives_unused_operation_publisher_release_failure() {
        let directory = tempfile::TempDir::new().unwrap();
        let provider = Arc::new(CountingProvider::qualified());
        let io = Arc::new(CountingIo::failing_read_open());
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(provider, SpillFrameLimits::format_max())
            .io(io)
            .build()
            .unwrap();
        let mut file = publish_empty_sort_run(&manager);
        let qualification = file.exact_owned_reader_qualification().unwrap();
        let mut config = BufferManagerConfig::with_budget(1024 * 1024);
        config.soft_limit_fraction = 1.0;
        config.evict_limit_fraction = 1.0;
        config.hard_limit_fraction = 1.0;
        let memory = BufferManager::new(config);
        let resources = crate::execution::QueryResourceContext::new(Arc::clone(&memory)).unwrap();
        let failure_publisher = AccountedErrorPublisher::<ProviderAccountedReaderError>::try_new(
            resources.try_allocate(0).unwrap(),
        )
        .unwrap();
        let refusal =
            fail_next_unused_publisher_release_for_test(MemoryGrantError::AccountingPoisoned {
                account: "unused-reader-publication-test",
            });

        let error = file
            .reader_with_exact_owned_provider_admission(
                qualification,
                resources.try_allocate(0).unwrap(),
                failure_publisher,
                &NoopGrantObserver,
            )
            .expect_err("injected ReadOpen failure must be published");
        drop(refusal);
        assert!(error.is::<ProviderAccountedReaderError>());
        assert_eq!(
            error.inspect::<ProviderAccountedReaderError, _>(|source| {
                assert_eq!(
                    source.io_error_kind(),
                    Some(std::io::ErrorKind::PermissionDenied),
                    "unused-publication release failure must not replace the provider primary"
                );
                assert!(matches!(
                    source
                        .unused_publication_release_error()
                        .map(ProviderAccountedReaderUnusedPublicationError::error),
                    Some(MemoryGrantError::AccountingPoisoned {
                        account: "unused-reader-publication-test"
                    })
                ));
            }),
            Some(())
        );
        let retained = memory.allocated();
        assert!(retained > error.granted_bytes());
        drop(error);
        assert_eq!(
            memory.allocated(),
            0,
            "error Drop retries the deallocated operation-slot grant after destroying the primary"
        );
        file.close_and_delete().unwrap();
    }
}

#[cfg(test)]
mod fallible_writer_tests {
    use super::{FallibleBufWriter, SpillWriterBuffer};
    use std::io::{self, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct InterruptedPartialWriter {
        bytes: Vec<u8>,
        interrupted: bool,
    }

    impl Write for InterruptedPartialWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(io::ErrorKind::Interrupted.into());
            }
            let written = bytes.len().min(2);
            self.bytes.extend_from_slice(&bytes[..written]);
            Ok(written)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct PartialThenErrorWriter {
        bytes: Vec<u8>,
        calls: usize,
        flushes: usize,
    }

    impl Write for PartialThenErrorWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.calls == 2 {
                return Err(io::ErrorKind::PermissionDenied.into());
            }
            let written = if self.calls == 1 {
                bytes.len().min(2)
            } else {
                bytes.len()
            };
            self.bytes.extend_from_slice(&bytes[..written]);
            Ok(written)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[derive(Default)]
    struct PartialThenPanicWriter {
        bytes: Vec<u8>,
        calls: usize,
    }

    impl Write for PartialThenPanicWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            assert_ne!(
                self.calls, 2,
                "writer panic after a confirmed partial write"
            );
            let written = if self.calls == 1 {
                bytes.len().min(2)
            } else {
                bytes.len()
            };
            self.bytes.extend_from_slice(&bytes[..written]);
            Ok(written)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct WriteZeroOnceWriter {
        bytes: Vec<u8>,
        calls: usize,
        partial_first: bool,
    }

    impl WriteZeroOnceWriter {
        fn new(partial_first: bool) -> Self {
            Self {
                bytes: Vec::new(),
                calls: 0,
                partial_first,
            }
        }
    }

    impl Write for WriteZeroOnceWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.partial_first && self.calls == 1 {
                let written = bytes.len().min(2);
                self.bytes.extend_from_slice(&bytes[..written]);
                return Ok(written);
            }
            if self.calls == usize::from(self.partial_first) + 1 {
                return Ok(0);
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct DropProbeWriter {
        drops: Arc<AtomicUsize>,
        fail_writes: bool,
    }

    impl Write for DropProbeWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_writes {
                return Err(io::ErrorKind::PermissionDenied.into());
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Drop for DropProbeWriter {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::AcqRel);
        }
    }

    struct SharedWriteProbe {
        writes: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        bytes: Arc<std::sync::Mutex<Vec<u8>>>,
    }

    impl Write for SharedWriteProbe {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes.fetch_add(1, Ordering::AcqRel);
            self.bytes.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Drop for SharedWriteProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[derive(Debug)]
    struct PrimaryDropPanic;

    #[test]
    fn prepared_writer_installation_reuses_exact_backing() {
        let prepared = SpillWriterBuffer::prepare().unwrap();
        let pointer = prepared.pointer();
        let capacity = prepared.capacity();

        let mut writer = FallibleBufWriter::with_prepared_buffer(Vec::new(), prepared);

        assert_eq!(writer.buffer_pointer(), pointer);
        assert_eq!(writer.capacity(), capacity);
        writer.write_all(b"prepared backing").unwrap();
        assert_eq!(writer.buffer_pointer(), pointer);
        assert_eq!(writer.capacity(), capacity);
    }

    #[test]
    fn fallible_writer_reserves_exact_backing_and_flushes_without_growth() {
        let mut writer = FallibleBufWriter::try_with_capacity(Vec::new(), 8).unwrap();
        let capacity = writer.capacity();
        assert!(capacity >= 8);

        writer.write_all(b"abc").unwrap();
        assert!(writer.get_ref().is_empty());
        assert_eq!(writer.capacity(), capacity);

        writer.flush().unwrap();
        assert_eq!(writer.get_ref(), b"abc");
        assert_eq!(writer.capacity(), capacity);
        assert_eq!(writer.into_inner().unwrap(), b"abc");
    }

    #[test]
    fn fallible_writer_flushes_pending_bytes_before_large_direct_write() {
        let mut writer = FallibleBufWriter::try_with_capacity(Vec::new(), 4).unwrap();

        writer.write_all(b"ab").unwrap();
        writer.write_all(b"cdefgh").unwrap();

        assert_eq!(writer.get_ref(), b"abcdefgh");
        assert_eq!(writer.into_inner().unwrap(), b"abcdefgh");
    }

    #[test]
    fn fallible_writer_retries_interrupted_and_partial_flushes() {
        let mut writer =
            FallibleBufWriter::try_with_capacity(InterruptedPartialWriter::default(), 8).unwrap();

        writer.write_all(b"abcdef").unwrap();
        writer.flush().unwrap();

        assert_eq!(writer.get_ref().bytes, b"abcdef");
        assert_eq!(writer.buffered_len(), 0);
    }

    #[test]
    fn fallible_writer_retains_only_unwritten_bytes_after_partial_error() {
        let mut writer =
            FallibleBufWriter::try_with_capacity(PartialThenErrorWriter::default(), 8).unwrap();
        writer.write_all(b"abcdef").unwrap();

        let error = writer.flush().unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(writer.get_ref().bytes, b"ab");
        assert_eq!(writer.get_ref().flushes, 0);
        assert_eq!(writer.buffered_len(), 4);
        writer.flush().unwrap();
        assert_eq!(writer.get_ref().bytes, b"abcdef");
        assert_eq!(writer.get_ref().flushes, 1);
        assert_eq!(writer.buffered_len(), 0);
    }

    #[test]
    fn fallible_writer_removes_confirmed_prefix_when_inner_panics() {
        let mut writer =
            FallibleBufWriter::try_with_capacity(PartialThenPanicWriter::default(), 8).unwrap();
        writer.write_all(b"abcdef").unwrap();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| writer.flush()));

        assert!(panic.is_err());
        assert_eq!(writer.get_ref().bytes, b"ab");
        assert_eq!(writer.buffered_len(), 4);
        writer.flush().unwrap();
        assert_eq!(writer.get_ref().bytes, b"abcdef");
        assert_eq!(writer.buffered_len(), 0);
    }

    #[test]
    fn fallible_writer_write_zero_retains_only_the_unwritten_suffix() {
        for partial_first in [false, true] {
            let mut writer =
                FallibleBufWriter::try_with_capacity(WriteZeroOnceWriter::new(partial_first), 8)
                    .unwrap();
            writer.write_all(b"abcdef").unwrap();

            let error = writer.flush().unwrap_err();

            assert_eq!(error.kind(), io::ErrorKind::WriteZero);
            let confirmed = if partial_first { 2 } else { 0 };
            assert_eq!(writer.get_ref().bytes, &b"abcdef"[..confirmed]);
            assert_eq!(writer.buffered_len(), 6 - confirmed);
            writer.flush().unwrap();
            assert_eq!(writer.get_ref().bytes, b"abcdef");
            assert_eq!(writer.buffered_len(), 0);
        }
    }

    #[test]
    fn fallible_writer_into_inner_error_is_terminal_and_drops_inner_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut writer = FallibleBufWriter::try_with_capacity(
            DropProbeWriter {
                drops: Arc::clone(&drops),
                fail_writes: true,
            },
            8,
        )
        .unwrap();
        writer.write_all(b"abc").unwrap();

        let Err(error) = writer.into_inner() else {
            panic!("failing inner writer unexpectedly completed")
        };

        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }

    #[test]
    fn fallible_writer_drop_during_unwind_discards_buffer_without_writing() {
        let writes = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let bytes = Arc::new(std::sync::Mutex::new(Vec::new()));

        let primary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut writer = FallibleBufWriter::try_with_capacity(
                SharedWriteProbe {
                    writes: Arc::clone(&writes),
                    drops: Arc::clone(&drops),
                    bytes: Arc::clone(&bytes),
                },
                8,
            )
            .unwrap();
            writer.write_all(b"abc").unwrap();
            std::panic::panic_any(PrimaryDropPanic);
        }))
        .unwrap_err();

        assert!(primary.is::<PrimaryDropPanic>());
        assert_eq!(writes.load(Ordering::Acquire), 0);
        assert!(bytes.lock().unwrap().is_empty());
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }

    #[test]
    fn fallible_writer_reports_impossible_capacity_without_panicking() {
        let drops = Arc::new(AtomicUsize::new(0));
        let result = FallibleBufWriter::try_with_capacity(
            DropProbeWriter {
                drops: Arc::clone(&drops),
                fail_writes: false,
            },
            usize::MAX,
        );

        assert!(matches!(
            result,
            Err(ref error) if error.kind() == std::io::ErrorKind::OutOfMemory
        ));
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }
}
