//! Move-only framed operations for schedulers. No executor or wire format lives here.

use super::file::{
    ProviderAccountedReaderError, ProviderAccountedReaderResolutionError, ProviderAccountedSortRow,
    ProviderAccountedSpillFileReader, SpillWriterBuffer, qualified_writer_buffer_requested_bytes,
};
use super::{SpillFile, SpillFileIdentity, SpillFileRole, SpillRecordKind};
use crate::execution::{QueryExecutionId, QueryResourceContext, QueryResourceContextError};
use allocator_api2::{alloc::Global, vec::Vec as ExactVec};
use grafeo_common::memory::buffer::{AccountedError, MemoryGrant, MemoryGrantError};
use std::any::Any;
use std::fmt;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Debug)]
enum Failure {
    Io(io::Error),
    Resource(QueryResourceContextError),
    Memory(MemoryGrantError),
    Reader(ProviderAccountedReaderError),
    Resolution(ProviderAccountedReaderResolutionError),
    Accounted(AccountedError),
    Panic { _payload: Box<dyn Any + Send> },
}

/// A failure together with the physical state and memory authority it still owns.
/// The original provider error cannot be detached from its workspace.
#[must_use]
pub struct OwnedSpillError {
    failure: Option<Failure>,
    writer: Option<OwnedSpillFile>,
}

impl OwnedSpillError {
    fn new(failure: Failure) -> Self {
        Self {
            failure: Some(failure),
            writer: None,
        }
    }

    fn io(error: io::Error) -> Self {
        Self::new(Failure::Io(error))
    }
}

impl fmt::Debug for OwnedSpillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedSpillError")
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for OwnedSpillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.failure.as_ref().expect("live diagnostic") {
            Failure::Io(e) => fmt::Display::fmt(e, f),
            Failure::Resource(e) => fmt::Display::fmt(e, f),
            Failure::Memory(e) => fmt::Display::fmt(e, f),
            Failure::Reader(e) => fmt::Display::fmt(e, f),
            Failure::Resolution(e) => fmt::Display::fmt(e, f),
            Failure::Accounted(e) => fmt::Display::fmt(e, f),
            Failure::Panic { .. } => f.write_str("spill provider or I/O callback panicked"),
        }
    }
}

impl std::error::Error for OwnedSpillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.failure.as_ref()? {
            Failure::Io(e) => Some(e),
            Failure::Resource(e) => Some(e),
            Failure::Memory(e) => Some(e),
            Failure::Reader(e) => Some(e),
            Failure::Resolution(e) => Some(e),
            Failure::Accounted(e) => Some(e),
            Failure::Panic { .. } => None,
        }
    }
}

impl Drop for OwnedSpillError {
    fn drop(&mut self) {
        // Destroy opaque diagnostics before their accounting. A hostile
        // destructor may leave heap behind, so retain the corresponding grant.
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(self.failure.take()))) {
            std::mem::forget(payload);
            if let Some(writer) = &mut self.writer {
                writer.retain_workspace = true;
            }
        }
    }
}

impl From<QueryResourceContextError> for OwnedSpillError {
    fn from(error: QueryResourceContextError) -> Self {
        Self::new(Failure::Resource(error))
    }
}
impl From<MemoryGrantError> for OwnedSpillError {
    fn from(error: MemoryGrantError) -> Self {
        Self::new(Failure::Memory(error))
    }
}

/// Immutable input bytes whose exact allocation remains query-accounted.
#[derive(Debug)]
pub struct OwnedSpillBytes {
    bytes: ExactVec<u8, Global>,
    query_id: QueryExecutionId,
    _grant: MemoryGrant,
}

#[allow(
    clippy::result_large_err,
    reason = "failures retain their move-only accounting authority"
)]
impl OwnedSpillBytes {
    /// Copies one bounded input after admitting its exact allocation.
    ///
    /// # Errors
    /// Returns resource exhaustion or allocation failure before enqueue.
    pub fn copy_from(
        resources: &QueryResourceContext,
        bytes: &[u8],
    ) -> Result<Self, OwnedSpillError> {
        let grant = resources.try_allocate(bytes.len())?;
        let mut copy = ExactVec::new_in(Global);
        copy.try_reserve_exact(bytes.len())
            .map_err(|_| OwnedSpillError::io(io::ErrorKind::OutOfMemory.into()))?;
        copy.extend_from_slice(bytes);
        Ok(Self {
            bytes: copy,
            query_id: resources.query_id(),
            _grant: grant,
        })
    }

    /// Borrows the admitted payload without detaching its grant.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

/// One operation in the shared framed writer state machine.
pub enum OwnedSpillWrite {
    /// Declares the sort columns and row count.
    SortStart {
        /// Number of encoded columns.
        columns: u32,
        /// Exact number of records that must precede publication.
        rows: u64,
    },
    /// Writes one sort row.
    SortRow(OwnedSpillBytes),
    /// Declares the partition entry count.
    PartitionStart(u64),
    /// Writes one partition entry.
    PartitionEntry(OwnedSpillBytes),
    /// Writes one RDF aggregate record.
    AggregateState(OwnedSpillBytes),
    /// Flushes, syncs, validates and publishes through the shared file owner.
    Finish,
}

/// A file and its writer workspace, transferred as a unit into physical jobs.
#[must_use]
pub struct OwnedSpillFile {
    file: Option<SpillFile>,
    workspace: Option<MemoryGrant>,
    resources: QueryResourceContext,
    base_bytes: usize,
    retain_workspace: bool,
}

#[allow(
    clippy::result_large_err,
    reason = "failures retain their move-only accounting authority"
)]
#[allow(
    clippy::missing_panics_doc,
    reason = "private move-only state guarantees live file and grant; provider panics are caught"
)]
impl OwnedSpillFile {
    /// Creates a framed file from the query's configured root and quota owner.
    ///
    /// # Errors
    /// Returns admission, provider, framing or filesystem errors with ownership.
    pub fn create(
        resources: &QueryResourceContext,
        role: SpillFileRole,
    ) -> Result<Self, OwnedSpillError> {
        let manager = resources
            .ensure_spill_manager()?
            .ok_or(QueryResourceContextError::SpillManagerUnavailable)?;
        let mut owner = Self {
            file: None,
            workspace: Some(resources.try_allocate(0)?),
            resources: resources.clone(),
            base_bytes: 0,
            retain_workspace: false,
        };
        let mut cleanup_complete = true;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let provider = manager.qualified_file_workspace_bound()?;
            let requested = qualified_writer_buffer_requested_bytes()
                .checked_add(provider)
                .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
            owner
                .workspace
                .as_mut()
                .expect("writer grant")
                .try_resize(requested)
                .map_err(io::Error::other)?;
            let buffer = SpillWriterBuffer::prepare()?;
            let buffer_bytes = buffer.capacity();
            owner.base_bytes = buffer_bytes
                .checked_add(provider)
                .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
            owner
                .workspace
                .as_mut()
                .expect("writer grant")
                .try_resize(owner.base_bytes)
                .map_err(io::Error::other)?;
            owner.file = Some(manager.create_owned_file_with_writer_buffer(
                role,
                buffer,
                |required| {
                    let bytes = buffer_bytes
                        .checked_add(required)
                        .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
                    owner.base_bytes = bytes;
                    owner
                        .workspace
                        .as_mut()
                        .expect("writer grant")
                        .try_resize(bytes)
                        .map_err(io::Error::other)
                },
                &mut cleanup_complete,
            )?);
            Ok(())
        }));
        owner.retain_workspace |= !cleanup_complete;
        owner.resolve(outcome)
    }

    fn resolve(
        mut self,
        outcome: std::thread::Result<io::Result<()>>,
    ) -> Result<Self, OwnedSpillError> {
        let failure = match outcome {
            Ok(Ok(())) => return Ok(self),
            Ok(Err(error)) => Failure::Io(error),
            Err(payload) => {
                // An unwind may have crossed a provider destructor. Its
                // failure payload can outlive physical state we cannot inspect.
                self.retain_workspace = true;
                Failure::Panic { _payload: payload }
            }
        };
        Err(OwnedSpillError {
            failure: Some(failure),
            writer: Some(self),
        })
    }

    /// Applies one bounded framed operation, consuming the owner on failure.
    ///
    /// # Errors
    /// Returns the primary failure with file/workspace ownership still attached.
    pub fn write(mut self, operation: OwnedSpillWrite) -> Result<Self, OwnedSpillError> {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            self.resources.check_cancelled().map_err(io::Error::other)?;
            if let OwnedSpillWrite::SortRow(bytes)
            | OwnedSpillWrite::PartitionEntry(bytes)
            | OwnedSpillWrite::AggregateState(bytes) = &operation
                && bytes.query_id != self.resources.query_id()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "spill payload is charged to another query",
                ));
            }
            let file = self.file.as_mut().expect("live owned file");
            let grant = self.workspace.as_mut().expect("writer grant");
            let base = self.base_bytes;
            let admit = |required: usize| {
                let bytes = base
                    .checked_add(required)
                    .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
                grant
                    .try_resize(bytes.max(grant.size()))
                    .map_err(io::Error::other)
            };
            match &operation {
                OwnedSpillWrite::SortStart { columns, rows } => {
                    file.write_sort_run_start(*columns, *rows)
                }
                OwnedSpillWrite::SortRow(bytes) => {
                    file.write_sort_row_with_admission(bytes.as_slice(), admit)
                }
                OwnedSpillWrite::PartitionStart(entries) => file.write_partition_start(*entries),
                OwnedSpillWrite::PartitionEntry(bytes) => {
                    file.write_partition_entry_with_admission(bytes.as_slice(), admit)
                }
                OwnedSpillWrite::AggregateState(bytes) => {
                    file.write_aggregate_state_with_admission(bytes.as_slice(), admit)
                }
                OwnedSpillWrite::Finish => file.finish_write(),
            }
        }));
        self.resolve(outcome)
    }

    /// Opens the same exact, grant-owning framed reader used by qualified sort.
    ///
    /// # Errors
    /// Returns unpublished, unsupported, admission or provider errors.
    pub fn reader(&self) -> Result<OwnedSpillReader, OwnedSpillError> {
        let file = self.file.as_ref().expect("live owned file");
        let inner = file
            .reader_with_owned_provider_admission(self.resources.try_allocate(0)?)
            .map_err(|e| OwnedSpillError::new(Failure::Reader(e)))?;
        Ok(OwnedSpillReader {
            inner,
            role: file.role(),
            resources: self.resources.clone(),
        })
    }

    /// Physical file identity used by the framed codec.
    #[must_use]
    pub fn identity(&self) -> SpillFileIdentity {
        self.file.as_ref().expect("live owned file").identity()
    }

    /// Closes and deletes through the retained capability and quota receipt.
    ///
    /// # Errors
    /// Returns the original cleanup failure with the owner available to Drop.
    pub fn close_and_delete(mut self) -> Result<(), OwnedSpillError> {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            self.file
                .as_mut()
                .expect("live owned file")
                .close_and_delete()
        }));
        self.resolve(outcome).map(drop)
    }
}

impl Drop for OwnedSpillFile {
    fn drop(&mut self) {
        if let Some(file) = self.file.take()
            && !file.retire_owned_file()
        {
            self.retain_workspace = true;
        }
        if self.retain_workspace
            && let Some(grant) = self.workspace.take()
        {
            std::mem::forget(grant);
        }
    }
}

/// An exact decoded record coupled to its output allocation grant.
#[derive(Debug)]
pub struct OwnedSpillRecord(ProviderAccountedSortRow);
impl OwnedSpillRecord {
    /// Borrows bytes while preserving their accounting owner.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.0.payload()
    }
}

/// A framed reader whose provider state and grants cannot escape separately.
#[must_use]
pub struct OwnedSpillReader {
    inner: ProviderAccountedSpillFileReader,
    role: SpillFileRole,
    resources: QueryResourceContext,
}

#[allow(
    clippy::result_large_err,
    reason = "failures retain their move-only accounting authority"
)]
impl OwnedSpillReader {
    /// Reads a sort or partition declaration (partition columns are zero).
    ///
    /// # Errors
    /// Returns a terminal owned error for invalid transitions or failed reads.
    pub fn read_declaration(self) -> Result<(Self, (u32, u64)), OwnedSpillError> {
        let (inner, declaration) = self
            .inner
            .read_declaration_owned()
            .map_err(|e| OwnedSpillError::new(Failure::Accounted(e)))?;
        Ok((
            Self {
                inner,
                role: self.role,
                resources: self.resources,
            },
            declaration,
        ))
    }

    /// Reads one bounded data record for this file's declared role.
    ///
    /// # Errors
    /// Returns resource, framing, provider or filesystem errors with grants.
    pub fn read_record(self) -> Result<(Self, OwnedSpillRecord), OwnedSpillError> {
        let kind = match self.role {
            SpillFileRole::SortRun => SpillRecordKind::SortRow,
            SpillFileRole::NativePartition => SpillRecordKind::PartitionEntry,
            SpillFileRole::RdfAggregateState => SpillRecordKind::AggregateState,
        };
        let (inner, row) = self
            .inner
            .read_record_owned(kind, self.resources.try_allocate(0)?)
            .map_err(|e| OwnedSpillError::new(Failure::Accounted(e)))?;
        Ok((
            Self {
                inner,
                role: self.role,
                resources: self.resources,
            },
            OwnedSpillRecord(row),
        ))
    }

    /// Validates the terminal record and closes the physical reader.
    ///
    /// # Errors
    /// Returns framing, I/O or explicit accounting-release failures.
    pub fn finish(self) -> Result<(), OwnedSpillError> {
        self.inner
            .finish_sort_run_owned()
            .map_err(|e| OwnedSpillError::new(Failure::Resolution(e)))
    }

    /// Closes a partially consumed reader without validating a terminal record.
    ///
    /// # Errors
    /// Returns physical or accounting-release failures.
    pub fn close(self) -> Result<(), OwnedSpillError> {
        self.inner
            .abort_owned()
            .map_err(|e| OwnedSpillError::new(Failure::Resolution(e)))
    }
}
