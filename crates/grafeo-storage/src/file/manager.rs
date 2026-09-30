//! High-level manager for `.grafeo` database files.
//!
//! [`GrafeoFileManager`] owns the file handle and provides create, open,
//! snapshot write/read, and sidecar WAL lifecycle management.

use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use grafeo_common::utils::error::{Error, Result};
use parking_lot::Mutex;

use super::format::{DATA_OFFSET, DbHeader, FileHeader};
use super::header;
use super::ownership::{
    ContainerDestination, ContainerLease, LockedFile, check_single_link, sync_parent,
};

// Every I/O path holds this cut before header/slot locks. In each owner state
// the primary descriptor is declared before the lease and retires first.
enum FileState {
    Open {
        file: LockedFile,
        lease: ContainerLease,
    },
    Failed {
        file: Option<LockedFile>,
        lease: ContainerLease,
    },
    Closed,
}

impl FileState {
    fn file_mut(&mut self) -> Result<&mut File> {
        match self {
            Self::Open { file, .. } => Ok(file),
            Self::Failed { .. } => Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "database file manager failed; close or drop it before reopening",
            )
            .into()),
            Self::Closed => Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "database file manager is closed",
            )
            .into()),
        }
    }

    fn enter_failed(&mut self) {
        let previous = std::mem::replace(self, Self::Closed);
        *self = match previous {
            Self::Open { file, lease } => Self::Failed {
                file: Some(file),
                lease,
            },
            other => other,
        };
    }
}

/// One encoded container section with its serializer's exact wire version.
#[derive(Debug, Clone, Copy)]
pub struct SectionWrite<'a> {
    section_type: grafeo_common::storage::SectionType,
    version: u8,
    data: &'a [u8],
}

impl<'a> SectionWrite<'a> {
    /// Creates a versioned section payload.
    #[must_use]
    pub const fn new(
        section_type: grafeo_common::storage::SectionType,
        version: u8,
        data: &'a [u8],
    ) -> Self {
        Self {
            section_type,
            version,
            data,
        }
    }

    /// Section type written to the directory.
    #[must_use]
    pub const fn section_type(self) -> grafeo_common::storage::SectionType {
        self.section_type
    }

    /// Exact serializer wire version.
    #[must_use]
    pub const fn version(self) -> u8 {
        self.version
    }

    /// Opaque encoded payload.
    #[must_use]
    pub const fn data(self) -> &'a [u8] {
        self.data
    }
}

/// Manages a single `.grafeo` database file.
///
/// # Lifecycle
///
/// 1. [`create`](Self::create) or [`open`](Self::open)
/// 2. Mutations flow through a sidecar WAL (managed externally by the engine)
/// 3. [`write_snapshot`](Self::write_snapshot) checkpoints memory to the file
/// 4. After a successful checkpoint, call [`remove_sidecar_wal`](Self::remove_sidecar_wal)
/// 5. [`close`](Self::close) (or drop) releases the file handle
pub struct GrafeoFileManager {
    /// Path to the `.grafeo` file.
    path: PathBuf,
    /// Whole-operation admission; acquired before active header and slot locks.
    state: Mutex<FileState>,
    /// File header (read once on open, immutable afterwards).
    file_header: FileHeader,
    /// Currently active database header.
    active_header: Mutex<DbHeader>,
    /// Slot index (0 or 1) of the active header.
    active_slot: Mutex<u8>,
    /// Whether this manager was opened in read-only mode.
    read_only: bool,
    /// Encryptor for section data (None = unencrypted).
    #[cfg(feature = "encryption")]
    section_encryptor: Option<grafeo_common::encryption::PageEncryptor>,
}

/// A source view retaining the container's whole-operation admission.
pub struct ContainerCapture<'a> {
    manager: &'a GrafeoFileManager,
    state: parking_lot::MutexGuard<'a, FileState>,
}

impl ContainerCapture<'_> {
    /// Streams the admitted container bytes without reopening its pathname.
    ///
    /// # Errors
    /// Returns source seek/read or destination write errors.
    pub fn write_image_to(&mut self, output: &mut impl std::io::Write) -> Result<u64> {
        let file = self.state.file_mut()?;
        file.seek(SeekFrom::Start(0))?;
        Ok(std::io::copy(file, output)?)
    }

    /// Copies from the admitted source into an already-bound destination.
    ///
    /// # Errors
    /// Returns checked copy errors or source/destination alias rejection.
    pub fn copy_to_destination(&mut self, destination: &mut ContainerDestination) -> Result<u64> {
        destination.overwrite_from(self.state.file_mut()?, &self.manager.path)
    }
}

/// Writable container admission held through sidecar retirement and close.
pub struct ContainerRetirement<'a> {
    manager: &'a GrafeoFileManager,
    state: parking_lot::MutexGuard<'a, FileState>,
}

impl ContainerRetirement<'_> {
    /// Removes exactly this container's sidecar while borrowing its actual seal.
    ///
    /// # Errors
    /// Rejects foreign seals and returns filesystem errors.
    pub fn retire_sidecar(&mut self, wal: &mut crate::wal::SealedWal) -> Result<()> {
        let sidecar = crate::ownership::resolve_components(&self.manager.sidecar_wal_path())?;
        if wal.path() != sidecar {
            return Err(Error::InvalidValue(
                "WAL seal does not belong to this container".into(),
            ));
        }
        #[cfg(feature = "testing-crash-injection")]
        ownership_test_point("sidecar-retire")?;
        wal.retire_directory()
    }

    /// Acquires unopened sidecar authority before inspecting or removing it.
    ///
    /// # Errors
    /// Returns contention, namespace or checked retirement errors.
    pub fn remove_unopened_sidecar(&mut self) -> Result<()> {
        let mut wal = crate::wal::WalDestination::acquire(self.manager.sidecar_wal_path())?;
        wal.retire_directory()
    }

    /// Closes C through this admitted view without reacquiring its mutex.
    ///
    /// # Errors
    /// Returns the original close error, retaining failed C ownership.
    pub fn close(mut self) -> Result<()> {
        self.manager.close_admitted(&mut self.state)
    }
}

impl GrafeoFileManager {
    /// Creates a new `.grafeo` file at `path`.
    ///
    /// Writes the file header and two empty database headers. The file must
    /// not already exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file already exists or cannot be created.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Self::create_with_graph_model(path, 0)
    }

    /// Creates a new `.grafeo` file tagged with a graph model.
    ///
    /// # Errors
    ///
    /// Returns an error if the file already exists or cannot be created.
    pub fn create_with_graph_model(path: impl AsRef<Path>, graph_model: u8) -> Result<Self> {
        let lease = ContainerLease::acquire(path.as_ref(), false, true)?;
        Self::create_with_lease(lease, graph_model)
    }

    pub(super) fn create_with_lease(lease: ContainerLease, graph_model: u8) -> Result<Self> {
        let path = lease.path().to_path_buf();

        if path.exists() {
            return Err(Error::Internal(format!(
                "file already exists: {}",
                path.display()
            )));
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists || e.raw_os_error() == Some(183) {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        format!(
                            "database file already exists (may be open by another process): {}",
                            path.display()
                        ),
                    ))
                } else {
                    Error::Io(e)
                }
            })?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        let mut file = LockedFile::acquire(file, false)?;
        check_single_link(&file)?;

        let mut file_header = FileHeader::new();
        file_header.graph_model = graph_model;
        header::write_file_header(&mut file, &file_header)?;
        header::write_db_header(&mut file, 0, &DbHeader::EMPTY)?;
        header::write_db_header(&mut file, 1, &DbHeader::EMPTY)?;
        file.sync_all()?;

        Ok(Self {
            path,
            state: Mutex::new(FileState::Open { file, lease }),
            file_header,
            active_header: Mutex::new(DbHeader::EMPTY),
            active_slot: Mutex::new(0),
            read_only: false,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Opens an existing `.grafeo` file.
    ///
    /// Validates the magic bytes and format version, then selects the
    /// active database header.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, cannot be exclusively
    /// locked, has invalid headers, or its stale staging file cannot be removed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let lease = ContainerLease::acquire(path.as_ref(), false, false)?;
        Self::open_with_lease(lease)
    }

    pub(super) fn open_with_lease(lease: ContainerLease) -> Result<Self> {
        let path = lease.path().to_path_buf();

        let file = crate::ownership::checked_file(&path, true, false)?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        let mut file = LockedFile::acquire(file, false)?;
        check_single_link(&file)?;

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        // Discard leftover staging only after locking and checking the primary
        // headers. Rejected opens must preserve recovery evidence.
        match fs::remove_file(Self::installing_path(&path)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        Ok(Self {
            path,
            state: Mutex::new(FileState::Open { file, lease }),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: false,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Opens an existing `.grafeo` file in read-only mode.
    ///
    /// Uses a **shared** file lock (`try_lock_shared`), allowing multiple
    /// readers to open the same file concurrently. Readers and an exclusive
    /// writer exclude each other, including throughout checkpoint replacement.
    ///
    /// The returned manager only supports [`read_snapshot`](Self::read_snapshot)
    /// and other read-only operations. Calling [`write_snapshot`](Self::write_snapshot)
    /// will return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, has invalid magic, or
    /// an unsupported format version.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let lease = ContainerLease::acquire(path.as_ref(), true, false)?;
        Self::open_read_only_with_lease(lease)
    }

    /// Opens one checked backup image under canonical container ownership.
    /// The same read-only descriptor supplies header decoding and later reads.
    ///
    /// # Errors
    /// Rejects source links, reserved paths, contention, invalid headers and I/O errors.
    pub fn open_backup_image(path: impl AsRef<Path>) -> Result<Self> {
        let lease = ContainerLease::acquire(path.as_ref(), true, false)?;
        let file = super::open_backup_source(path.as_ref())?;
        Self::open_read_only_file(lease, file)
    }

    pub(super) fn open_read_only_with_lease(lease: ContainerLease) -> Result<Self> {
        let file = crate::ownership::checked_file(lease.path(), false, false)?;
        Self::open_read_only_file(lease, file)
    }

    fn open_read_only_file(lease: ContainerLease, file: File) -> Result<Self> {
        let path = lease.path().to_path_buf();

        // Acquire a shared lock: coexists with other shared locks but
        // blocks if an exclusive lock cannot be shared (platform-dependent).
        let mut file = LockedFile::acquire(file, true)?;
        check_single_link(&file)?;

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        Ok(Self {
            path,
            state: Mutex::new(FileState::Open { file, lease }),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: true,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Sets the encryptor for section-level encryption.
    ///
    /// When set, all section data is encrypted on write and decrypted on read.
    /// The GCM authentication tag provides integrity verification, replacing
    /// the CRC-32 checksum for encrypted sections.
    #[cfg(feature = "encryption")]
    pub fn set_section_encryptor(&mut self, encryptor: grafeo_common::encryption::PageEncryptor) {
        self.section_encryptor = Some(encryptor);
    }

    /// Returns `true` if this manager was opened in read-only mode.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// On-disk graph model tag (0 LPG, 1 RDF, 2 Both).
    #[must_use]
    pub fn graph_model_tag(&self) -> u8 {
        self.file_header.graph_model
    }

    /// Writes snapshot data into the file and updates the inactive DB header.
    ///
    /// Steps:
    /// 1. Write `data` at [`DATA_OFFSET`]
    /// 2. Compute CRC-32 checksum
    /// 3. Build a new [`DbHeader`] and write it to the inactive slot
    /// 4. `fsync` the file
    /// 5. Update internal active header/slot state
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O operation fails.
    pub fn write_snapshot(
        &self,
        data: &[u8],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        let mut state = self.state.lock();
        let file = state.file_mut()?;
        if self.read_only {
            return Err(Error::Internal(
                "cannot write snapshot: database is open in read-only mode".to_string(),
            ));
        }
        check_single_link(file)?;

        use grafeo_common::testing::crash::maybe_crash;

        let checksum = crc32fast::hash(data);
        // reason: millis since UNIX epoch fits in u64 for ~585 million years
        #[allow(clippy::cast_possible_truncation)]
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut active_header = self.active_header.lock();
        let mut active_slot = self.active_slot.lock();

        let new_iteration = active_header.iteration + 1;
        let target_slot = u8::from(*active_slot == 0);

        maybe_crash("write_snapshot:before_data_write");

        // Write snapshot data
        file.seek(SeekFrom::Start(DATA_OFFSET))?;
        file.write_all(data)?;

        maybe_crash("write_snapshot:after_data_write");

        // Truncate file to exact size (remove stale trailing data)
        let file_end = DATA_OFFSET + data.len() as u64;
        file.set_len(file_end)?;

        maybe_crash("write_snapshot:after_truncate");

        // Build and write new header to inactive slot
        let new_header = DbHeader {
            iteration: new_iteration,
            checksum,
            snapshot_length: data.len() as u64,
            epoch,
            transaction_id,
            node_count,
            edge_count,
            timestamp_ms,
        };
        header::write_db_header(file, target_slot, &new_header)?;

        maybe_crash("write_snapshot:after_header_write");

        // Ensure everything is on disk before we consider this committed
        file.sync_all()?;

        maybe_crash("write_snapshot:after_fsync");

        *active_header = new_header;
        *active_slot = target_slot;

        Ok(())
    }

    /// Reads snapshot data from the file using the active database header.
    ///
    /// Returns an empty `Vec` if the database has never been checkpointed
    /// (both headers are empty).
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails or the CRC checksum does not match.
    pub fn read_snapshot(&self) -> Result<Vec<u8>> {
        let mut state = self.state.lock();
        let file = state.file_mut()?;
        let active_header = self.active_header.lock();

        if active_header.is_empty() {
            return Ok(Vec::new());
        }

        // v2 files store sections rather than a v1 snapshot blob. They set
        // snapshot_length == 0 and put the directory CRC in the checksum field.
        // Reading 0 bytes here would CRC to 0 and mismatch the directory CRC.
        if active_header.snapshot_length == 0 {
            return Ok(Vec::new());
        }

        // reason: snapshot_length is the size of serialized in-memory data, fits in usize on 64-bit targets;
        // on 32-bit targets the database would OOM long before reaching 4 GiB
        // reason: value bounded by collection size, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let length = active_header.snapshot_length as usize;
        let expected_checksum = active_header.checksum;
        drop(active_header);

        file.seek(SeekFrom::Start(DATA_OFFSET))?;

        let mut data = vec![0u8; length];
        std::io::Read::read_exact(file, &mut data)?;

        // Verify CRC
        let actual_checksum = crc32fast::hash(&data);
        if actual_checksum != expected_checksum {
            return Err(Error::Internal(format!(
                "snapshot checksum mismatch: expected {expected_checksum:#010X}, got {actual_checksum:#010X}"
            )));
        }

        Ok(data)
    }

    /// Returns the path for the sidecar WAL directory.
    ///
    /// For a database at `mydb.grafeo`, the sidecar is `mydb.grafeo.wal/`.
    #[must_use]
    pub fn sidecar_wal_path(&self) -> PathBuf {
        let mut wal_path = self.path.as_os_str().to_owned();
        wal_path.push(".wal");
        PathBuf::from(wal_path)
    }

    /// Best-effort observation of sidecar existence, including after close.
    /// This does not admit or authorize any filesystem mutation.
    #[must_use]
    pub fn has_sidecar_wal(&self) -> bool {
        self.sidecar_wal_path().exists()
    }

    /// Removes the sidecar WAL directory after a successful checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory exists but cannot be removed.
    pub fn remove_sidecar_wal(&self) -> Result<()> {
        self.sidecar_retirement()?.remove_unopened_sidecar()
    }

    /// Admits a source view, including a read-only source.
    ///
    /// # Errors
    /// Rejects a closed or failed container.
    pub fn capture(&self) -> Result<ContainerCapture<'_>> {
        let mut state = self.state.lock();
        state.file_mut()?;
        Ok(ContainerCapture {
            manager: self,
            state,
        })
    }

    /// Admits writable retirement before acquiring any WAL operation.
    ///
    /// # Errors
    /// Rejects closed, failed or read-only containers.
    pub fn sidecar_retirement(&self) -> Result<ContainerRetirement<'_>> {
        let mut state = self.state.lock();
        state.file_mut()?;
        if self.read_only {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "cannot remove sidecar from a read-only manager",
            )
            .into());
        }
        #[cfg(feature = "testing-crash-injection")]
        ownership_test_point("retirement-admission")?;
        Ok(ContainerRetirement {
            manager: self,
            state,
        })
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns a clone of the currently active database header.
    #[must_use]
    pub fn active_header(&self) -> DbHeader {
        self.active_header.lock().clone()
    }

    /// Returns the file header (written at creation, immutable).
    #[must_use]
    pub fn file_header(&self) -> &FileHeader {
        &self.file_header
    }

    /// Returns the total file size on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file metadata cannot be read.
    pub fn file_size(&self) -> Result<u64> {
        let mut state = self.state.lock();
        let file = state.file_mut()?;
        let metadata = file.metadata()?;
        Ok(metadata.len())
    }

    /// Flushes and syncs the file.
    ///
    /// # Errors
    ///
    /// Returns an error if sync fails.
    pub fn sync(&self) -> Result<()> {
        let mut state = self.state.lock();
        let file = state.file_mut()?;
        if !self.read_only {
            file.sync_all()?;
        }
        Ok(())
    }

    // ── Section-based I/O (v2 container format) ─────────────────────

    fn installing_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".installing");
        PathBuf::from(p)
    }

    fn reopen_primary(&self) -> Result<LockedFile> {
        let new_file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        let new_file = LockedFile::acquire(new_file, false)?;
        check_single_link(&new_file)?;
        Ok(new_file)
    }

    /// Atomically replace the primary file with a fully-written temp container.
    ///
    /// The operation guard and permanent path lease survive closing the old
    /// descriptor. Any error or unwind from that point retains Failed ownership.
    fn install_tmp_container(&self, state: &mut FileState, tmp: &Path) -> Result<()> {
        check_single_link(state.file_mut()?)?;
        state.enter_failed();
        if let FileState::Failed { file, .. } = state {
            drop(file.take());
        }
        #[cfg(feature = "testing-crash-injection")]
        ownership_test_point("after-close")?;
        #[cfg(feature = "testing-crash-injection")]
        ownership_test_point("before-rename")?;
        fs::rename(tmp, &self.path)?;
        #[cfg(feature = "testing-crash-injection")]
        ownership_test_point("after-rename")?;
        #[cfg(feature = "testing-crash-injection")]
        ownership_test_point("before-parent-sync")?;
        sync_parent(&self.path)?;
        #[cfg(feature = "testing-crash-injection")]
        ownership_test_point("before-reopen")?;
        let file = self.reopen_primary()?;
        match std::mem::replace(state, FileState::Closed) {
            FileState::Failed {
                file: previous,
                lease,
            } => {
                drop(previous);
                *state = FileState::Open { file, lease };
                Ok(())
            }
            other => {
                *state = other;
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "replacement lost its admitted failed state",
                )
                .into())
            }
        }
    }

    /// Writes multiple sections using a temp file + atomic rename.
    ///
    /// The live `.grafeo` file is never overwritten in place. A crash during
    /// installation leaves the previous container intact; the sidecar WAL can
    /// replay records since that snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if write or sync fails.
    pub fn write_sections(
        &self,
        sections: &[(grafeo_common::storage::SectionType, &[u8])],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        let versioned: Vec<_> = sections
            .iter()
            .map(|(section_type, data)| SectionWrite::new(*section_type, 1, data))
            .collect();
        self.write_versioned_sections(&versioned, epoch, transaction_id, node_count, edge_count)
    }

    /// Writes multiple sections with their exact independent wire versions.
    ///
    /// Version zero and duplicate section types are rejected before the temp
    /// container is created. The live file is installed atomically only after
    /// all payloads, the versioned directory, and both headers are durable.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid section metadata or any write/sync/install
    /// failure.
    pub fn write_versioned_sections(
        &self,
        sections: &[SectionWrite<'_>],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        use crate::container::SectionDirectory;
        use crate::container::directory::{DIRECTORY_OFFSET, SECTION_DATA_OFFSET};
        use grafeo_common::storage::SectionDirectoryEntry;
        use grafeo_common::testing::crash::maybe_crash;

        let mut state = self.state.lock();
        state.file_mut()?;

        if self.read_only {
            return Err(Error::Internal(
                "cannot write sections: database is open in read-only mode".to_string(),
            ));
        }

        let mut section_types = std::collections::HashSet::with_capacity(sections.len());
        for section in sections {
            if section.version == 0 {
                return Err(Error::Serialization(format!(
                    "section {:?} uses reserved wire version zero",
                    section.section_type
                )));
            }
            if !section_types.insert(section.section_type) {
                return Err(Error::Serialization(format!(
                    "duplicate {:?} section in one container image",
                    section.section_type
                )));
            }
        }

        let new_iteration = self.active_header.lock().iteration + 1;
        #[cfg(feature = "encryption")]
        #[allow(clippy::cast_possible_truncation)]
        let nonce_iteration = new_iteration as u32;

        #[allow(clippy::cast_possible_truncation)]
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Directory checksum is computed while writing the temp file.
        let tmp = Self::installing_path(&self.path);
        match fs::remove_file(&tmp) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let mut tmp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;

        header::write_file_header(&mut tmp_file, &self.file_header)?;
        header::write_db_header(&mut tmp_file, 0, &DbHeader::EMPTY)?;
        header::write_db_header(&mut tmp_file, 1, &DbHeader::EMPTY)?;

        maybe_crash("write_sections:before_data");

        let mut dir = SectionDirectory::new();
        let page_size = 4096u64;
        let mut current_offset = SECTION_DATA_OFFSET;

        for section in sections {
            let section_type = section.section_type;
            let data = section.data;
            #[cfg(feature = "encryption")]
            let encrypted_buf: Option<Vec<u8>> = if let Some(ref enc) = self.section_encryptor {
                let nonce_high = (nonce_iteration << 8) | (section_type as u32 & 0xFF);
                let nonce = grafeo_common::encryption::build_nonce(nonce_high, current_offset);
                let aad = format!("grafeo-section:{}", section_type as u32);
                Some(
                    enc.encrypt(data, &nonce, aad.as_bytes())
                        .map_err(|e| Error::Internal(format!("section encryption failed: {e}")))?,
                )
            } else {
                None
            };
            #[cfg(feature = "encryption")]
            let write_data: &[u8] = encrypted_buf.as_deref().unwrap_or(data);
            #[cfg(not(feature = "encryption"))]
            let write_data: &[u8] = data;

            let checksum = crc32fast::hash(write_data);
            let length = write_data.len() as u64;

            tmp_file.seek(SeekFrom::Start(current_offset))?;
            tmp_file.write_all(write_data)?;

            dir.upsert(SectionDirectoryEntry {
                section_type,
                version: section.version,
                flags: section_type.default_flags(),
                offset: current_offset,
                length,
                checksum,
            })?;

            let section_end = current_offset + length;
            current_offset = (section_end + page_size - 1) / page_size * page_size;
        }

        maybe_crash("write_sections:after_data");

        tmp_file.set_len(current_offset)?;
        let dir_bytes = dir.to_bytes();
        tmp_file.seek(SeekFrom::Start(DIRECTORY_OFFSET))?;
        tmp_file.write_all(&dir_bytes)?;

        maybe_crash("write_sections:after_directory");

        let new_header = DbHeader {
            iteration: new_iteration,
            checksum: dir.checksum(),
            snapshot_length: 0,
            epoch,
            transaction_id,
            node_count,
            edge_count,
            timestamp_ms,
        };
        header::write_db_header(&mut tmp_file, 0, &new_header)?;
        header::write_db_header(&mut tmp_file, 1, &DbHeader::EMPTY)?;
        tmp_file.sync_all()?;
        drop(tmp_file);

        maybe_crash("write_sections:after_fsync");

        self.install_tmp_container(&mut state, &tmp)?;
        *self.active_header.lock() = new_header;
        *self.active_slot.lock() = 0;
        Ok(())
    }

    /// Reads the section directory from the file.
    ///
    /// Detects v2 format by checking the `snapshot_length` field in the active
    /// DbHeader: v2 writes set `snapshot_length = 0`, while v1 always has a
    /// non-zero snapshot length when data exists.
    ///
    /// Returns `None` only when the file is unambiguously v1
    /// (`snapshot_length` non-zero) or uninitialized (header iteration is 0).
    /// Once the header asserts v2, any failure to locate or parse the
    /// directory is surfaced as an error: misreporting v2 corruption as a v1
    /// file would cause callers to fall back to v1 read paths and mask the
    /// underlying problem.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - I/O fails
    /// - The header asserts v2 but the file is too short to hold a directory page
    /// - The directory page bytes fail to parse as a `SectionDirectory`
    /// - The directory page CRC does not match the value recorded in the active header
    pub fn read_section_directory(&self) -> Result<Option<crate::container::SectionDirectory>> {
        use crate::container::SectionDirectory;
        use crate::container::directory::DIRECTORY_OFFSET;

        let mut state = self.state.lock();
        let file = state.file_mut()?;
        let active_header = self.active_header.lock();

        // v1 files have snapshot_length > 0; v2 files set it to 0 and put the
        // directory CRC in the checksum field. An uninitialized header (iteration
        // == 0) means no data has been written yet.
        if active_header.is_empty() || active_header.snapshot_length > 0 {
            return Ok(None);
        }
        let expected_checksum = active_header.checksum;
        drop(active_header);

        // Past this point the header asserts v2: any failure to read or parse
        // the directory is real corruption, not a v1/v2 misdetection. Surface
        // it instead of silently falling through to read_snapshot, where v1 CRC
        // logic would mask the underlying cause.
        let file_size = file.metadata()?.len();
        if file_size < DIRECTORY_OFFSET + 4096 {
            return Err(Error::Internal(format!(
                "v2 header indicates section directory at offset {DIRECTORY_OFFSET:#X}, \
                 but file is only {file_size} bytes",
            )));
        }

        file.seek(SeekFrom::Start(DIRECTORY_OFFSET))?;

        let mut buf = vec![0u8; 4096];
        std::io::Read::read_exact(file, &mut buf)?;

        let dir = SectionDirectory::from_bytes(&buf).map_err(|e| {
            Error::Internal(format!(
                "v2 section directory at offset {DIRECTORY_OFFSET:#X} failed to parse: {e}",
            ))
        })?;

        // Cross-check the directory bytes against the CRC the writer recorded
        // in the active header. A mismatch means the directory page is torn or
        // corrupted (e.g. a partial write from a crashed checkpoint), not a
        // format ambiguity.
        let actual_checksum = crc32fast::hash(&buf);
        if actual_checksum != expected_checksum {
            return Err(Error::Internal(format!(
                "v2 section directory checksum mismatch: \
                 header recorded {expected_checksum:#010X}, computed {actual_checksum:#010X}",
            )));
        }

        if dir.is_empty() {
            return Ok(None);
        }
        Ok(Some(dir))
    }

    /// Reads a single section's data from the file.
    ///
    /// Uses the section directory entry to locate and verify the data.
    ///
    /// # Errors
    ///
    /// Returns an error if read fails or CRC checksum doesn't match.
    pub fn read_section_data(
        &self,
        entry: &grafeo_common::storage::SectionDirectoryEntry,
    ) -> Result<Vec<u8>> {
        let mut state = self.state.lock();
        let file = state.file_mut()?;
        file.seek(SeekFrom::Start(entry.offset))?;

        // reason: section length is bounded by file size, which fits in usize on 64-bit targets;
        // on 32-bit targets sections would OOM long before reaching 4 GiB
        // reason: value bounded by collection size, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let mut data = vec![0u8; entry.length as usize];
        std::io::Read::read_exact(file, &mut data)?;

        // Verify CRC on the raw bytes (encrypted or plaintext)
        let actual_crc = crc32fast::hash(&data);
        if actual_crc != entry.checksum {
            return Err(Error::Internal(format!(
                "section {:?} CRC mismatch: expected {:#010X}, got {actual_crc:#010X}",
                entry.section_type, entry.checksum
            )));
        }

        // Decrypt if encryption is enabled
        #[cfg(feature = "encryption")]
        if let Some(ref enc) = self.section_encryptor {
            let aad = format!("grafeo-section:{}", entry.section_type as u32);
            return enc.decrypt(&data, aad.as_bytes()).map_err(|_| {
                Error::Internal(format!(
                    "section {:?} decryption failed: wrong key or corrupted data",
                    entry.section_type
                ))
            });
        }

        Ok(data)
    }

    /// Memory-maps a single section for zero-copy read access.
    ///
    /// The section's CRC-32 is verified against the mmap'd bytes before
    /// returning, which also warms the OS page cache. Only sections with
    /// `flags.mmap_able = true` can be mapped (index sections).
    ///
    /// The returned [`MmapSection`](crate::container::MmapSection) is
    /// independent of the file mutex: multiple mmaps can coexist. However,
    /// all `MmapSection` handles **must be dropped before writing** (via
    /// `write_sections()` or `write_snapshot()`). On Windows the OS rejects
    /// writes to a file with active mappings; on Linux/macOS stale mappings
    /// would read outdated data. See [`MmapSection`](crate::container::MmapSection)
    /// for the full lifecycle.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The section is not mmap-able (data section)
    /// - The mmap system call fails
    /// - The CRC-32 checksum does not match (corrupt data)
    #[allow(unsafe_code)]
    pub fn mmap_section(
        &self,
        entry: &grafeo_common::storage::SectionDirectoryEntry,
    ) -> Result<crate::container::MmapSection> {
        let mut state = self.state.lock();
        let file = state.file_mut()?;
        if !entry.flags.mmap_able {
            return Err(Error::Internal(format!(
                "section {:?} is not mmap-able (data sections must be deserialized)",
                entry.section_type
            )));
        }

        if entry.length == 0 {
            return Err(Error::Internal(format!(
                "section {:?} has zero length, cannot mmap",
                entry.section_type
            )));
        }

        // SAFETY: We hold an exclusive lock on the `.grafeo` file, preventing
        // concurrent modification by other processes. The mapping is read-only.
        // The section region [offset .. offset+length] was written by
        // write_sections() and its CRC is verified below before the mmap
        // is exposed to callers.
        // reason: section length is bounded by file size, fits in usize on 64-bit targets
        #[allow(clippy::cast_possible_truncation)]
        let section_len = entry.length as usize;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(entry.offset)
                .len(section_len)
                .map(&*file)
        }
        .map_err(Error::Io)?;

        // Verify CRC on the mmap'd bytes. This reads through the mapping,
        // which triggers page faults and warms the OS page cache: a free
        // prefetch disguised as an integrity check.
        let actual_crc = crc32fast::hash(&mmap);
        if actual_crc != entry.checksum {
            return Err(Error::Internal(format!(
                "section {:?} CRC mismatch: expected {:#010X}, got {actual_crc:#010X}",
                entry.section_type, entry.checksum
            )));
        }

        Ok(crate::container::MmapSection::new(
            mmap,
            entry.section_type,
            entry.checksum,
        ))
    }

    /// Copies the database file to `dest` using the already-locked file handle.
    ///
    /// `std::fs::copy()` opens the source with a new handle, which fails on
    /// Windows when an exclusive lock is held. This method reads through the
    /// existing handle, avoiding lock conflicts.
    ///
    /// # Errors
    ///
    /// Returns an error if the read or write fails.
    pub fn copy_to(&self, dest: &Path) -> Result<u64> {
        let mut state = self.state.lock();
        let file = state.file_mut()?;
        let mut destination = ContainerDestination::acquire(dest)?;
        destination.overwrite_from(file, &self.path)
    }

    /// Copies through checked handles while the caller retains destination
    /// authority for subsequent checksum, manifest, and cursor publication.
    ///
    /// # Errors
    /// Returns closed/source I/O errors or destination lock/link/copy failures.
    pub fn copy_to_destination(&self, destination: &mut ContainerDestination) -> Result<u64> {
        let mut state = self.state.lock();
        destination.overwrite_from(state.file_mut()?, &self.path)
    }

    /// Terminal, idempotent close. A sync failure retains non-writable ownership;
    /// another close retries the sync, or Drop releases it without a sync claim.
    ///
    /// # Errors
    ///
    /// Returns the actual sync error without releasing the path lease.
    pub fn close(&self) -> Result<()> {
        let mut state = self.state.lock();
        self.close_admitted(&mut state)
    }

    fn close_admitted(&self, state: &mut FileState) -> Result<()> {
        state.enter_failed();
        if let FileState::Failed {
            file: Some(file), ..
        } = &*state
            && !self.read_only
        {
            #[cfg(feature = "testing-crash-injection")]
            ownership_test_point("close-sync")?;
            file.sync_all()?;
        }
        // Enum fields retire in declaration order: primary before path lease.
        *state = FileState::Closed;
        Ok(())
    }
}

impl Drop for GrafeoFileManager {
    fn drop(&mut self) {
        *self.state.get_mut() = FileState::Closed;
    }
}

// Process-local opt-in rendezvous for owned child tests. The parent bounds the
// handshake and kills/reaps its child on timeout or assertion failure.
#[cfg(feature = "testing-crash-injection")]
pub(super) fn ownership_test_point(point: &str) -> Result<()> {
    #[cfg(test)]
    OPERATION_TEST_HOOK.with(|hook| {
        if let Some(hook) = hook.borrow().as_ref() {
            hook(point)?;
        }
        Ok::<(), Error>(())
    })?;
    if std::env::var_os("GRAFEO_OWNERSHIP_FAIL").as_deref() == Some(std::ffi::OsStr::new(point)) {
        static FAILED_ONCE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !FAILED_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("injected container ownership error at {point}"),
            )
            .into());
        }
    }
    if std::env::var("GRAFEO_OWNERSHIP_RENDEZVOUS")
        .is_ok_and(|points| points.split(',').any(|value| value == point))
    {
        let mut output = std::io::stdout().lock();
        writeln!(output, "READY")?;
        output.flush()?;
        drop(output);
        let mut response = String::new();
        std::io::stdin().read_line(&mut response)?;
        if response.trim() != "RELEASE" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ownership rendezvous requires RELEASE",
            )
            .into());
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "testing-crash-injection"))]
type OwnershipTestHook = Box<dyn Fn(&str) -> Result<()>>;
#[cfg(all(test, feature = "testing-crash-injection"))]
thread_local! {
    static OPERATION_TEST_HOOK: std::cell::RefCell<Option<OwnershipTestHook>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs2::FileExt;
    use tempfile::TempDir;

    #[test]
    fn ownership_capture_holds_admission_until_copy_finishes_before_close() {
        use std::sync::{Arc, mpsc};
        let dir = test_dir();
        let manager =
            Arc::new(GrafeoFileManager::create(dir.path().join("source.grafeo")).unwrap());
        let mut capture = manager.capture().unwrap();
        assert!(manager.state.try_lock().is_none());
        let other = Arc::clone(&manager);
        let (attempt, attempted) = mpsc::channel();
        let (done, completed) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            attempt.send(()).unwrap();
            other.close().unwrap();
            done.send(()).unwrap();
        });
        attempted.recv().unwrap();
        let mut destination =
            ContainerDestination::acquire(dir.path().join("copy.grafeo")).unwrap();
        assert!(capture.copy_to_destination(&mut destination).unwrap() > 0);
        assert!(completed.try_recv().is_err());
        assert!(manager.state.try_lock().is_none());
        drop(capture);
        completed.recv().unwrap();
        worker.join().unwrap();
        assert!(manager.capture().is_err());
    }

    fn test_dir() -> TempDir {
        TempDir::new().expect("create temp dir")
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn ownership_same_manager_serializes_complete_checkpoints() {
        use grafeo_common::storage::SectionType;
        use std::sync::{Arc, mpsc};
        use std::time::Duration;
        let dir = test_dir();
        let path = dir.path().join("serial.grafeo");
        let manager = Arc::new(GrafeoFileManager::create(&path).unwrap());
        manager.write_snapshot(b"initial", 1, 1, 0, 0).unwrap();
        let (ready_send, ready_receive) = mpsc::channel();
        let (release_send, release_receive) = mpsc::channel();
        let (first_done_send, first_done_receive) = mpsc::channel();
        let first_manager = Arc::clone(&manager);
        let first = std::thread::spawn(move || {
            OPERATION_TEST_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move |point| {
                    if point == "after-close" {
                        ready_send.send(()).unwrap();
                        release_receive
                            .recv_timeout(Duration::from_secs(10))
                            .unwrap();
                    }
                    Ok(())
                }));
            });
            let result = first_manager.write_sections(
                &[(SectionType::LpgStore, b"first complete cut")],
                2,
                2,
                0,
                0,
            );
            OPERATION_TEST_HOOK.with(|hook| *hook.borrow_mut() = None);
            first_done_send.send(result).unwrap();
        });
        ready_receive.recv_timeout(Duration::from_secs(10)).unwrap();
        let (attempt_send, attempt_receive) = mpsc::channel();
        let (second_done_send, second_done_receive) = mpsc::channel();
        let second_manager = Arc::clone(&manager);
        let second = std::thread::spawn(move || {
            attempt_send.send(()).unwrap();
            let result = second_manager.write_sections(
                &[(SectionType::LpgStore, b"second complete cut")],
                3,
                3,
                0,
                0,
            );
            second_done_send.send(result).unwrap();
        });
        attempt_receive
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        assert!(matches!(
            second_done_receive.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release_send.send(()).unwrap();
        first_done_receive
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        second_done_receive
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(manager.active_header().iteration, 3);
        let directory = manager.read_section_directory().unwrap().unwrap();
        assert_eq!(
            manager
                .read_section_data(directory.find(SectionType::LpgStore).unwrap())
                .unwrap(),
            b"second complete cut"
        );
        manager.close().unwrap();
        assert!(manager.write_snapshot(b"closed", 4, 4, 0, 0).is_err());
        let reopened = GrafeoFileManager::open(path).unwrap();
        assert_eq!(reopened.active_header().epoch, 3);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn ownership_close_waits_for_complete_checkpoint_before_releasing_lease() {
        use grafeo_common::storage::SectionType;
        use std::sync::{Arc, mpsc};
        use std::time::Duration;
        let dir = test_dir();
        let path = dir.path().join("closing.grafeo");
        let manager = Arc::new(GrafeoFileManager::create(&path).unwrap());
        let (ready_send, ready_receive) = mpsc::channel();
        let (release_send, release_receive) = mpsc::channel();
        let (write_done_send, write_done_receive) = mpsc::channel();
        let writer_manager = Arc::clone(&manager);
        let writer = std::thread::spawn(move || {
            OPERATION_TEST_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move |point| {
                    if point == "after-close" {
                        ready_send.send(()).unwrap();
                        release_receive
                            .recv_timeout(Duration::from_secs(10))
                            .unwrap();
                    }
                    Ok(())
                }));
            });
            let result = writer_manager.write_sections(
                &[(SectionType::LpgStore, b"completed before terminal close")],
                1,
                1,
                0,
                0,
            );
            OPERATION_TEST_HOOK.with(|hook| *hook.borrow_mut() = None);
            write_done_send.send(result).unwrap();
        });
        ready_receive.recv_timeout(Duration::from_secs(10)).unwrap();
        let (attempt_send, attempt_receive) = mpsc::channel();
        let (close_done_send, close_done_receive) = mpsc::channel();
        let close_manager = Arc::clone(&manager);
        let closer = std::thread::spawn(move || {
            attempt_send.send(()).unwrap();
            close_done_send.send(close_manager.close()).unwrap();
        });
        attempt_receive
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        assert!(matches!(
            close_done_receive.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(GrafeoFileManager::open(&path).is_err());
        release_send.send(()).unwrap();
        write_done_receive
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        close_done_receive
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        writer.join().unwrap();
        closer.join().unwrap();
        assert!(manager.file_size().is_err());
        let reopened = GrafeoFileManager::open(path).unwrap();
        let directory = reopened.read_section_directory().unwrap().unwrap();
        assert_eq!(
            reopened
                .read_section_data(directory.find(SectionType::LpgStore).unwrap())
                .unwrap(),
            b"completed before terminal close"
        );
    }

    #[test]
    fn create_and_open() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        // Create
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(path.exists());
        assert!(manager.active_header().is_empty());
        drop(manager);

        // Open
        let manager = GrafeoFileManager::open(&path).unwrap();
        assert!(manager.active_header().is_empty());
    }

    #[test]
    fn create_fails_if_exists() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        GrafeoFileManager::create(&path).unwrap();
        let result = GrafeoFileManager::create(&path);
        assert!(result.is_err());
    }

    #[test]
    fn open_fails_if_not_exists() {
        let dir = test_dir();
        let path = dir.path().join("nonexistent.grafeo");

        let result = GrafeoFileManager::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn open_staging_cleanup_preserves_active_owners_files() {
        let dir = test_dir();
        let path = dir.path().join("locked.grafeo");
        let staging = dir.path().join("locked.grafeo.installing");
        let before = dir.path().join("primary-before");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.copy_to(&before).unwrap();
        fs::write(&staging, b"active owner's unfinished checkpoint").unwrap();

        let error = GrafeoFileManager::open(&path)
            .err()
            .expect("competing opener must fail");
        assert!(error.to_string().contains("locked"));
        drop(manager);
        assert_eq!(fs::read(&path).unwrap(), fs::read(&before).unwrap());
        assert!(staging.exists(), "rejected opener deleted active staging");
        assert_eq!(
            fs::read(&staging).unwrap(),
            b"active owner's unfinished checkpoint"
        );
    }

    #[test]
    fn open_staging_cleanup_preserves_evidence_without_primary() {
        let dir = test_dir();
        let path = dir.path().join("missing.grafeo");
        let staging = dir.path().join("missing.grafeo.installing");
        fs::write(&staging, b"uninstalled evidence").unwrap();

        let error = GrafeoFileManager::open(&path)
            .err()
            .expect("missing primary must fail");
        assert!(matches!(error, Error::Io(error) if error.kind() == std::io::ErrorKind::NotFound));
        assert!(!path.exists());
        assert!(staging.exists(), "missing primary must retain staging");
        assert_eq!(fs::read(&staging).unwrap(), b"uninstalled evidence");
    }

    #[test]
    fn open_staging_cleanup_preserves_evidence_with_invalid_file_header() {
        let dir = test_dir();
        let path = dir.path().join("invalid.grafeo");
        let staging = dir.path().join("invalid.grafeo.installing");
        let invalid_header = [0u8; 4096];
        fs::write(&path, invalid_header).unwrap();
        fs::write(&staging, b"recoverable staging evidence").unwrap();

        let error = GrafeoFileManager::open(&path)
            .err()
            .expect("invalid primary must fail");
        assert!(error.to_string().contains("invalid magic"));
        assert_eq!(fs::read(&path).unwrap(), invalid_header);
        assert!(staging.exists(), "invalid primary must retain staging");
        assert_eq!(fs::read(&staging).unwrap(), b"recoverable staging evidence");
    }

    #[test]
    fn open_staging_cleanup_preserves_evidence_with_truncated_db_headers() {
        let dir = test_dir();
        let path = dir.path().join("truncated.grafeo");
        let staging = dir.path().join("truncated.grafeo.installing");
        drop(GrafeoFileManager::create(&path).unwrap());
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(super::super::format::FILE_HEADER_SIZE)
            .unwrap();
        let before = fs::read(&path).unwrap();
        fs::write(&staging, b"staging after incomplete primary headers").unwrap();

        let error = GrafeoFileManager::open(&path)
            .err()
            .expect("incomplete database headers must fail");
        assert!(
            matches!(error, Error::Io(error) if error.kind() == std::io::ErrorKind::UnexpectedEof)
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        assert!(
            staging.exists(),
            "unreadable database headers must retain staging"
        );
        assert_eq!(
            fs::read(&staging).unwrap(),
            b"staging after incomplete primary headers"
        );
    }

    #[test]
    fn open_staging_cleanup_removes_stale_file_and_preserves_committed_payload() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("valid.grafeo");
        let staging = dir.path().join("valid.grafeo.installing");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_versioned_sections(
                &[SectionWrite::new(
                    SectionType::Catalog,
                    1,
                    b"committed payload",
                )],
                1,
                1,
                0,
                0,
            )
            .unwrap();
        drop(manager);
        fs::write(&staging, b"stale uninstalled payload").unwrap();

        let manager = GrafeoFileManager::open(&path).unwrap();
        assert!(!staging.exists());
        let directory = manager.read_section_directory().unwrap().unwrap();
        let entry = directory.find(SectionType::Catalog).unwrap();
        assert_eq!(
            manager.read_section_data(entry).unwrap(),
            b"committed payload"
        );
    }

    #[test]
    fn open_staging_cleanup_accepts_absent_staging_file() {
        let dir = test_dir();
        let path = dir.path().join("no-staging.grafeo");
        let staging = dir.path().join("no-staging.grafeo.installing");
        drop(GrafeoFileManager::create(&path).unwrap());

        let manager = GrafeoFileManager::open(&path).unwrap();
        assert!(manager.active_header().is_empty());
        assert!(!staging.exists());
    }

    #[test]
    fn open_staging_cleanup_reports_directory_error_and_releases_primary_lock() {
        let dir = test_dir();
        let path = dir.path().join("directory.grafeo");
        let staging = dir.path().join("directory.grafeo.installing");
        drop(GrafeoFileManager::create(&path).unwrap());
        let before = fs::read(&path).unwrap();
        fs::create_dir(&staging).unwrap();
        let marker = staging.join("marker");
        fs::write(&marker, b"not an ordinary staging file").unwrap();

        let error = GrafeoFileManager::open(&path)
            .err()
            .expect("staging directory removal error must be returned");
        assert!(matches!(error, Error::Io(error) if error.kind() != std::io::ErrorKind::NotFound));
        assert!(staging.is_dir());
        assert_eq!(fs::read(&marker).unwrap(), b"not an ordinary staging file");
        assert_eq!(fs::read(&path).unwrap(), before);
        let raw = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        raw.try_lock_exclusive()
            .expect("failed open must release the primary lock");
        raw.unlock().unwrap();
    }

    #[test]
    fn write_and_read_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        let snapshot_data = b"hello grafeo snapshot data";
        manager.write_snapshot(snapshot_data, 1, 1, 10, 20).unwrap();

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, snapshot_data);

        // Verify header was updated
        let header = manager.active_header();
        assert_eq!(header.iteration, 1);
        assert_eq!(header.snapshot_length, snapshot_data.len() as u64);
        assert_eq!(header.epoch, 1);
        assert_eq!(header.node_count, 10);
        assert_eq!(header.edge_count, 20);
    }

    #[test]
    fn snapshot_persists_across_reopen() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let snapshot_data = b"persistent data across reopen";

        // Write
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_snapshot(snapshot_data, 5, 3, 100, 200)
                .unwrap();
        }

        // Reopen and read
        {
            let manager = GrafeoFileManager::open(&path).unwrap();
            let loaded = manager.read_snapshot().unwrap();
            assert_eq!(loaded, snapshot_data);

            let header = manager.active_header();
            assert_eq!(header.iteration, 1);
            assert_eq!(header.epoch, 5);
            assert_eq!(header.node_count, 100);
        }
    }

    #[test]
    fn alternating_snapshots() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // First checkpoint
        let data1 = b"snapshot version 1";
        manager.write_snapshot(data1, 1, 1, 10, 5).unwrap();
        assert_eq!(manager.active_header().iteration, 1);

        // Second checkpoint (alternates to other slot)
        let data2 = b"snapshot version 2 with more data";
        manager.write_snapshot(data2, 2, 2, 20, 10).unwrap();
        assert_eq!(manager.active_header().iteration, 2);

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, data2);
    }

    #[test]
    fn read_empty_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let data = manager.read_snapshot().unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn read_snapshot_returns_empty_on_v2_header() {
        // After write_sections, snapshot_length == 0 in the active header and the
        // checksum field holds the section-directory CRC. The pre-fix v1 reader
        // would read 0 bytes, CRC empty data to 0, and mismatch the directory CRC.
        // The fix early-returns Ok(Vec::new()) when snapshot_length == 0.
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("v2.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
            .unwrap();

        // Pre-fix: this returned Err("snapshot checksum mismatch").
        // Post-fix: returns Ok(Vec::new()), letting engine fall through to v2 dispatch.
        let data = manager.read_snapshot().unwrap();
        assert!(
            data.is_empty(),
            "v2 file should produce empty snapshot vec, not an error"
        );

        // Sanity: header confirms this is a v2 file (snapshot_length == 0 with non-zero checksum).
        let header = manager.active_header();
        assert_eq!(header.snapshot_length, 0);
        assert!(!header.is_empty());
    }

    #[test]
    fn section_writer_preserves_each_encoder_version() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("section-versions.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_versioned_sections(
                &[
                    SectionWrite::new(SectionType::WorldMetadata, 1, b"world"),
                    SectionWrite::new(SectionType::LpgStore, 7, b"lpg"),
                ],
                1,
                1,
                0,
                0,
            )
            .unwrap();

        let directory = manager.read_section_directory().unwrap().unwrap();
        assert_eq!(
            directory.find(SectionType::WorldMetadata).unwrap().version,
            1
        );
        assert_eq!(directory.find(SectionType::LpgStore).unwrap().version, 7);
    }

    #[test]
    fn section_writer_rejects_zero_versions_and_duplicates_before_install() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("invalid-section-versions.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();

        assert!(
            manager
                .write_versioned_sections(
                    &[SectionWrite::new(SectionType::WorldMetadata, 0, b"world")],
                    1,
                    1,
                    0,
                    0,
                )
                .is_err()
        );
        assert!(manager.active_header().is_empty());

        assert!(
            manager
                .write_versioned_sections(
                    &[
                        SectionWrite::new(SectionType::WorldMetadata, 1, b"first"),
                        SectionWrite::new(SectionType::WorldMetadata, 1, b"second"),
                    ],
                    1,
                    1,
                    0,
                    0,
                )
                .is_err()
        );
        assert!(manager.active_header().is_empty());
    }

    #[test]
    fn read_section_directory_surfaces_parse_error_on_v2_header() {
        // A v2 header with a corrupted directory page must not silently
        // degrade to "this is a v1 file" — that masking is what made the
        // GRAFEO-X001 in #323 surface as a misleading snapshot CRC error
        // instead of pointing at the real directory corruption.
        use crate::container::directory::DIRECTORY_OFFSET;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("corrupt_dir.grafeo");

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
                .unwrap();
        }

        // Overwrite the directory page count field with a value above MAX_SECTIONS
        // so SectionDirectory::from_bytes rejects it as malformed.
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DIRECTORY_OFFSET)).unwrap();
            file.write_all(&u32::MAX.to_le_bytes()).unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let err = manager
            .read_section_directory()
            .expect_err("corrupt v2 directory must surface as Err, not Ok(None)");
        let msg = err.to_string();
        assert!(
            msg.contains("v2 section directory") && msg.contains("failed to parse"),
            "error should name the v2 directory and the parse failure, got: {msg}"
        );
    }

    #[test]
    fn read_section_directory_surfaces_checksum_mismatch_on_v2_header() {
        // A torn write (e.g. a crashed checkpoint) can leave the directory page
        // bytes inconsistent with the CRC the writer recorded in the active
        // header. The pre-fix wildcard match swallowed this, falling through to
        // v1 read logic that reported a misleading snapshot checksum mismatch.
        use crate::container::directory::DIRECTORY_OFFSET;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("torn_dir.grafeo");

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
                .unwrap();
        }

        // Flip a byte in the reserved area of the directory page (bytes 4-7).
        // The page still parses (count is intact, no entries change) but the
        // CRC over the page no longer matches the value in the active header.
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DIRECTORY_OFFSET + 4)).unwrap();
            file.write_all(&[0xAA]).unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let err = manager
            .read_section_directory()
            .expect_err("torn v2 directory must surface as Err, not Ok(None)");
        let msg = err.to_string();
        assert!(
            msg.contains("v2 section directory checksum mismatch"),
            "error should identify the directory CRC mismatch, got: {msg}"
        );
    }

    #[test]
    fn sidecar_wal_path_computation() {
        let dir = test_dir();
        let path = dir.path().join("mydb.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let wal_path = manager.sidecar_wal_path();

        assert_eq!(
            wal_path.file_name().unwrap().to_str().unwrap(),
            "mydb.grafeo.wal"
        );
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn sidecar_wal_detect_and_remove() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(!manager.has_sidecar_wal());

        // Create sidecar directory manually (simulating engine behavior)
        fs::create_dir_all(manager.sidecar_wal_path()).unwrap();
        assert!(manager.has_sidecar_wal());

        // Remove it
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn file_size_grows_with_data() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let empty_size = manager.file_size().unwrap();

        // Empty file should be at least 12 KiB (3 headers)
        assert!(empty_size >= DATA_OFFSET, "empty size: {empty_size}");

        let big_data = vec![0xAB; 100_000];
        manager.write_snapshot(&big_data, 1, 1, 0, 0).unwrap();

        let full_size = manager.file_size().unwrap();
        assert!(full_size > empty_size);
        assert_eq!(full_size, DATA_OFFSET + big_data.len() as u64);
    }

    #[test]
    fn exclusive_lock_prevents_second_open() {
        let dir = test_dir();
        let path = dir.path().join("locked.grafeo");

        let _manager1 = GrafeoFileManager::create(&path).unwrap();

        // Second open should fail
        let result = GrafeoFileManager::open(&path);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("locked"));
    }

    #[test]
    fn lock_released_after_close() {
        let dir = test_dir();
        let path = dir.path().join("lockclose.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"data", 1, 1, 0, 0).unwrap();
        manager.close().unwrap();

        // Should succeed after close
        let manager2 = GrafeoFileManager::open(&path).unwrap();
        let data = manager2.read_snapshot().unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn lock_released_on_drop() {
        let dir = test_dir();
        let path = dir.path().join("lockdrop.grafeo");

        {
            let _manager = GrafeoFileManager::create(&path).unwrap();
            // Drop without explicit close
        }

        // Should succeed after drop
        let _manager2 = GrafeoFileManager::open(&path).unwrap();
    }

    #[test]
    fn checksum_mismatch_detected() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"valid data", 1, 1, 0, 0).unwrap();
        drop(manager);

        // Corrupt the snapshot data in the file
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
            file.write_all(b"CORRUPT!!!").unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let result = manager.read_snapshot();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn open_read_only_reads_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro.grafeo");

        // Create and write snapshot, then close
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_snapshot(b"read-only test data", 3, 2, 5, 10)
                .unwrap();
            manager.close().unwrap();
        }

        // Open read-only
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        assert!(ro.is_read_only());
        let data = ro.read_snapshot().unwrap();
        assert_eq!(data, b"read-only test data");

        let header = ro.active_header();
        assert_eq!(header.epoch, 3);
        assert_eq!(header.node_count, 5);
        assert_eq!(header.edge_count, 10);
    }

    #[test]
    fn read_only_rejects_write_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro_write.grafeo");

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.close().unwrap();
        }

        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        let result = ro.write_snapshot(b"nope", 1, 1, 0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("read-only"));
    }

    #[test]
    fn read_only_coexists_with_exclusive_after_close() {
        let dir = test_dir();
        let path = dir.path().join("coexist.grafeo");

        // Create, write, close
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.write_snapshot(b"coexist data", 1, 1, 1, 1).unwrap();
            manager.close().unwrap();
        }

        // Two read-only opens should coexist
        let ro1 = GrafeoFileManager::open_read_only(&path).unwrap();
        let ro2 = GrafeoFileManager::open_read_only(&path).unwrap();

        assert_eq!(ro1.read_snapshot().unwrap(), b"coexist data");
        assert_eq!(ro2.read_snapshot().unwrap(), b"coexist data");
    }

    // ── Mmap section tests ─────────────────────────────────────────

    #[test]
    fn mmap_section_roundtrip() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // Write two sections: one data (LPG), one index (VectorStore)
        let lpg_data = b"lpg node data here";
        let vector_data = vec![0x42u8; 8192]; // 8 KiB of vector embeddings

        manager
            .write_sections(
                &[
                    (SectionType::LpgStore, lpg_data.as_slice()),
                    (SectionType::VectorStore, &vector_data),
                ],
                1,
                1,
                10,
                5,
            )
            .unwrap();

        // Read the directory to get entries
        let section_dir = manager.read_section_directory().unwrap().unwrap();

        // Mmap the VectorStore section (mmap-able)
        let vector_entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(vector_entry).unwrap();

        assert_eq!(mmap.section_type(), SectionType::VectorStore);
        assert_eq!(mmap.len(), vector_data.len());
        assert_eq!(mmap.as_bytes(), &vector_data);
        assert!(!mmap.is_empty());
    }

    #[test]
    fn mmap_rejects_data_sections() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_reject.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"data")], 1, 1, 1, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let lpg_entry = section_dir.find(SectionType::LpgStore).unwrap();

        let result = manager.mmap_section(lpg_entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not mmap-able"));
    }

    #[test]
    fn mmap_detects_corruption() {
        use grafeo_common::storage::SectionType;
        use std::io::Write as IoWrite;

        let dir = test_dir();
        let path = dir.path().join("mmap_corrupt.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let vector_data = vec![0xAB; 4096];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_data)], 1, 1, 0, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap().clone();

        // Corrupt the section data by writing directly to the file
        drop(manager);
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(entry.offset)).unwrap();
            file.write_all(b"CORRUPTED!").unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let result = manager.mmap_section(&entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRC mismatch"));
    }

    #[test]
    fn mmap_multiple_sections_coexist() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_multi.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        let vector_data = vec![0x11; 4096];
        let text_data = vec![0x22; 2048];

        manager
            .write_sections(
                &[
                    (SectionType::VectorStore, &vector_data),
                    (SectionType::TextIndex, &text_data),
                ],
                1,
                1,
                0,
                0,
            )
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();

        // Mmap both index sections simultaneously
        let vec_entry = section_dir.find(SectionType::VectorStore).unwrap();
        let text_entry = section_dir.find(SectionType::TextIndex).unwrap();

        let vec_mmap = manager.mmap_section(vec_entry).unwrap();
        let text_mmap = manager.mmap_section(text_entry).unwrap();

        // Both are valid and independent
        assert_eq!(vec_mmap.as_bytes(), &vector_data);
        assert_eq!(text_mmap.as_bytes(), &text_data);
        assert_eq!(vec_mmap.section_type(), SectionType::VectorStore);
        assert_eq!(text_mmap.section_type(), SectionType::TextIndex);
    }

    #[test]
    fn mmap_drop_then_checkpoint_lifecycle() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_lifecycle.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // Checkpoint 1: write vector section
        let vector_v1 = vec![0x11; 4096];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_v1)], 1, 1, 0, 0)
            .unwrap();

        // Mmap the section, read it
        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();
        assert_eq!(mmap.as_bytes(), &vector_v1);

        // Drop the mmap before next checkpoint.
        // On Windows, writes fail if mmaps are still active (error 1224).
        // On all platforms, the intended lifecycle is: drop mmaps, checkpoint,
        // re-mmap. This keeps the flow simple and cross-platform.
        drop(mmap);

        // Checkpoint 2: write updated vector section
        let vector_v2 = vec![0x22; 8192];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_v2)], 2, 2, 0, 0)
            .unwrap();

        // Re-mmap the new section
        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();
        assert_eq!(mmap.as_bytes(), &vector_v2);
        assert_eq!(mmap.len(), 8192);
    }

    #[test]
    fn mmap_section_debug_format() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_debug.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::VectorStore, &[1, 2, 3, 4])], 1, 1, 0, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();

        let debug = format!("{mmap:?}");
        assert!(debug.contains("MmapSection"));
        assert!(debug.contains("VectorStore"));
    }

    #[test]
    fn path_returns_resolved_database_file_path() {
        let dir = test_dir();
        let path = dir.path().join("alix.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert_eq!(manager.path(), fs::canonicalize(path).unwrap());
    }

    #[test]
    fn file_header_returns_valid_header() {
        use crate::file::format;
        let dir = test_dir();
        let path = dir.path().join("gus.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        let header = manager.file_header();
        assert_eq!(header.magic, format::MAGIC);
        assert_eq!(header.format_version, format::FORMAT_VERSION);
    }

    #[test]
    fn sync_succeeds_for_writable_manager() {
        let dir = test_dir();
        let path = dir.path().join("vincent.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"sync test", 1, 1, 5, 3).unwrap();
        manager.sync().unwrap();
    }

    #[test]
    fn sync_skips_for_read_only_manager() {
        let dir = test_dir();
        let path = dir.path().join("jules.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.write_snapshot(b"ro sync", 1, 1, 0, 0).unwrap();
            manager.close().unwrap();
        }
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        ro.sync().unwrap();
    }

    #[test]
    fn close_succeeds_for_read_only_manager() {
        let dir = test_dir();
        let path = dir.path().join("mia.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.close().unwrap();
        }
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        ro.close().unwrap();
    }

    #[test]
    fn remove_sidecar_wal_no_op_when_absent() {
        let dir = test_dir();
        let path = dir.path().join("django.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(!manager.has_sidecar_wal());
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn multiple_snapshots_alternate_slots() {
        let dir = test_dir();
        let path = dir.path().join("shosanna.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();

        manager.write_snapshot(b"epoch one", 1, 1, 1, 0).unwrap();
        assert_eq!(manager.active_header().iteration, 1);

        manager.write_snapshot(b"epoch two", 2, 2, 2, 1).unwrap();
        assert_eq!(manager.active_header().iteration, 2);

        manager
            .write_snapshot(b"epoch three, longer data", 3, 3, 3, 2)
            .unwrap();
        assert_eq!(manager.active_header().iteration, 3);

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, b"epoch three, longer data");

        let header = manager.active_header();
        assert_eq!(header.epoch, 3);
        assert_eq!(header.node_count, 3);
        assert!(header.timestamp_ms > 0);
    }

    #[test]
    fn snapshot_truncates_stale_trailing_data() {
        let dir = test_dir();
        let path = dir.path().join("hans.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();

        let large_data = vec![0xAA; 50_000];
        manager.write_snapshot(&large_data, 1, 1, 0, 0).unwrap();
        let size_after_large = manager.file_size().unwrap();

        let small_data = b"tiny";
        manager.write_snapshot(small_data, 2, 2, 0, 0).unwrap();
        let size_after_small = manager.file_size().unwrap();

        assert!(
            size_after_small < size_after_large,
            "file should shrink: {size_after_small} >= {size_after_large}"
        );
        assert_eq!(manager.read_snapshot().unwrap(), small_data);
    }

    #[test]
    fn open_read_only_fails_for_nonexistent_file() {
        let dir = test_dir();
        let path = dir.path().join("beatrix_missing.grafeo");
        assert!(GrafeoFileManager::open_read_only(&path).is_err());
    }

    #[test]
    fn copy_to_produces_identical_file() {
        let dir = test_dir();
        let src = dir.path().join("copy_src.grafeo");
        let dest = dir.path().join("copy_dest.grafeo");

        let manager = GrafeoFileManager::create(&src).unwrap();
        manager
            .write_snapshot(b"copy test payload", 5, 3, 10, 20)
            .unwrap();

        // copy_to reads through the locked handle (no new open)
        let bytes = manager.copy_to(&dest).unwrap();
        assert!(bytes > 0);

        // The original is still usable
        let snap = manager.read_snapshot().unwrap();
        assert_eq!(snap, b"copy test payload");
        manager.close().unwrap();

        // The copy is a valid .grafeo file
        let copy = GrafeoFileManager::open(&dest).unwrap();
        let snap = copy.read_snapshot().unwrap();
        assert_eq!(snap, b"copy test payload");

        let header = copy.active_header();
        assert_eq!(header.epoch, 5);
        assert_eq!(header.node_count, 10);
        assert_eq!(header.edge_count, 20);
        copy.close().unwrap();
    }

    #[test]
    fn copy_to_from_read_only_manager() {
        let dir = test_dir();
        let src = dir.path().join("ro_copy_src.grafeo");
        let dest = dir.path().join("ro_copy_dest.grafeo");

        {
            let manager = GrafeoFileManager::create(&src).unwrap();
            manager
                .write_snapshot(b"read-only copy data", 7, 4, 3, 1)
                .unwrap();
            manager.close().unwrap();
        }

        let ro = GrafeoFileManager::open_read_only(&src).unwrap();
        let bytes = ro.copy_to(&dest).unwrap();
        assert!(bytes > 0);

        let copy = GrafeoFileManager::open(&dest).unwrap();
        assert_eq!(copy.read_snapshot().unwrap(), b"read-only copy data");
        copy.close().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn admitted_capture_streams_and_copies_original_after_path_replacement() {
        let dir = test_dir();
        let src = dir.path().join("admitted_src.grafeo");
        let moved = dir.path().join("admitted_src.moved.grafeo");
        let dest = dir.path().join("admitted_dest.grafeo");
        let manager = GrafeoFileManager::create(&src).unwrap();
        manager
            .write_snapshot(b"descriptor-bound image", 5, 3, 10, 20)
            .unwrap();

        let mut capture = manager.capture().unwrap();
        let mut expected = Vec::new();
        capture.write_image_to(&mut expected).unwrap();

        std::fs::rename(&src, &moved).unwrap();
        std::fs::write(&src, b"replacement at the old pathname").unwrap();

        let mut streamed = Vec::new();
        assert_eq!(
            capture.write_image_to(&mut streamed).unwrap(),
            expected.len() as u64
        );
        assert_eq!(streamed, expected);

        let mut destination = ContainerDestination::acquire(&dest).unwrap();
        assert_eq!(
            capture.copy_to_destination(&mut destination).unwrap(),
            expected.len() as u64
        );
        drop(destination);
        assert_eq!(std::fs::read(&dest).unwrap(), expected);
        drop(capture);
        manager.close().unwrap();
    }

    #[test]
    #[cfg(all(feature = "encryption", not(miri)))]
    fn encrypted_section_roundtrip() {
        use grafeo_common::encryption::KeyChain;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("encrypted.grafeo");

        let kc = KeyChain::new([0xAB; 32]);

        let section_data = b"sensitive graph data that must be encrypted";

        // Write with encryption
        {
            let mut manager = GrafeoFileManager::create(&path).unwrap();
            manager.set_section_encryptor(kc.encryptor_for("section", b"test"));
            manager
                .write_sections(&[(SectionType::LpgStore, &section_data[..])], 1, 0, 0, 0)
                .unwrap();
            manager.close().unwrap();
        }

        // Read back with same key
        {
            let mut manager = GrafeoFileManager::open(&path).unwrap();
            manager.set_section_encryptor(kc.encryptor_for("section", b"test"));
            let dir_opt = manager.read_section_directory().unwrap();
            let section_dir = dir_opt.expect("directory should exist");
            let entry = section_dir
                .entries()
                .iter()
                .find(|e| e.section_type == SectionType::LpgStore)
                .expect("LpgStore section should exist");
            let decrypted = manager.read_section_data(entry).unwrap();
            assert_eq!(decrypted, section_data);
        }
    }

    #[test]
    #[cfg(all(feature = "encryption", not(miri)))]
    fn encrypted_section_wrong_key_fails() {
        use grafeo_common::encryption::KeyChain;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("wrong_key.grafeo");

        let kc_a = KeyChain::new([0xAA; 32]);
        let kc_b = KeyChain::new([0xBB; 32]);

        // Write with key A
        {
            let mut manager = GrafeoFileManager::create(&path).unwrap();
            manager.set_section_encryptor(kc_a.encryptor_for("section", b"test"));
            manager
                .write_sections(&[(SectionType::LpgStore, b"secret data")], 1, 0, 0, 0)
                .unwrap();
            manager.close().unwrap();
        }

        // Read with key B: CRC passes (computed on encrypted bytes), but decryption fails
        {
            let mut manager = GrafeoFileManager::open(&path).unwrap();
            manager.set_section_encryptor(kc_b.encryptor_for("section", b"test"));
            let dir_opt = manager.read_section_directory().unwrap();
            let section_dir = dir_opt.expect("directory should exist");
            let entry = section_dir
                .entries()
                .iter()
                .find(|e| e.section_type == SectionType::LpgStore)
                .expect("section should exist");
            let result = manager.read_section_data(entry);
            assert!(result.is_err(), "decryption with wrong key should fail");
        }
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn write_sections_crash_after_data_keeps_previous_container() {
        use grafeo_common::storage::SectionType;
        use grafeo_common::testing::crash::{CrashResult, with_crash_named};

        let dir = test_dir();
        let path = dir.path().join("atomic.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"first-payload")], 1, 1, 1, 0)
            .unwrap();

        let crashed = with_crash_named("write_sections:after_data", || {
            let _ = manager.write_sections(
                &[(SectionType::LpgStore, b"second-payload-should-not-commit")],
                2,
                2,
                2,
                0,
            );
        });
        assert!(
            matches!(crashed, CrashResult::Crashed),
            "must crash after writing temp section data"
        );
        assert!(
            GrafeoFileManager::installing_path(&path).exists(),
            "interrupted install must leave a leftover .installing, not a torn primary"
        );
        drop(manager);

        let manager = GrafeoFileManager::open(&path).unwrap();
        assert!(
            !GrafeoFileManager::installing_path(&path).exists(),
            "open must discard leftover .installing"
        );
        let section_dir = manager
            .read_section_directory()
            .unwrap()
            .expect("previous v2 directory must survive");
        let entry = section_dir
            .find(SectionType::LpgStore)
            .expect("previous LPG section");
        let bytes = manager.read_section_data(entry).unwrap();
        assert_eq!(
            bytes.as_slice(),
            b"first-payload",
            "in-place overwrite must not destroy the last committed container"
        );
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn write_sections_crash_after_fsync_does_not_install_tmp() {
        use grafeo_common::storage::SectionType;
        use grafeo_common::testing::crash::{CrashResult, with_crash_named};

        let dir = test_dir();
        let path = dir.path().join("atomic_fsync.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"first-payload")], 1, 1, 1, 0)
            .unwrap();

        let crashed = with_crash_named("write_sections:after_fsync", || {
            let _ = manager.write_sections(
                &[(SectionType::LpgStore, b"second-payload-should-not-commit")],
                2,
                2,
                2,
                0,
            );
        });
        assert!(
            matches!(crashed, CrashResult::Crashed),
            "must crash after fsync of the temp container"
        );
        drop(manager);

        let manager = GrafeoFileManager::open(&path).unwrap();
        let section_dir = manager
            .read_section_directory()
            .unwrap()
            .expect("previous v2 directory must survive");
        let entry = section_dir
            .find(SectionType::LpgStore)
            .expect("previous LPG section");
        let bytes = manager.read_section_data(entry).unwrap();
        assert_eq!(
            bytes.as_slice(),
            b"first-payload",
            "uninstalled temp container must not replace the last-good primary"
        );
    }
}
