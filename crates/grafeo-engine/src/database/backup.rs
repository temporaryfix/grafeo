//! Incremental backup and point-in-time recovery.
//!
//! Provides `backup_full()`, `backup_incremental()`, and `restore_to_epoch()`
//! APIs on [`GrafeoDB`](super::GrafeoDB). Full backups capture the entire
//! database state; incremental backups export only the WAL records since the
//! last backup. Current v2 metadata binds full images to store identity,
//! content digests and WorldCut. Recovery replays eligible incremental
//! segments toward a requested epoch. Retained WAL intervals and immutable
//! matched publication preserve committed generations across interruption.
//! Restore validates and materializes a detached image before atomic installation.
//!
//! # Backup chain model
//!
//! ```text
//! [Full Snapshot] -> [Incr 1] -> [Incr 2] -> ... -> [Incr N]
//!   epoch 0-100      101-200     201-300              901-1000
//! ```
//!
//! To restore to epoch 750: load full snapshot (epoch 100), replay
//! incrementals 1-7, stop at epoch 750.

use std::path::Path;

use grafeo_common::types::EpochId;
#[cfg(any(test, all(feature = "wal", feature = "grafeo-file")))]
use grafeo_common::utils::error::StorageError;
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Serialize};

mod chain;
mod publication;

// ── Backup types ───────────────────────────────────────────────────

/// The type of a backup segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BackupKind {
    /// A full snapshot of the entire database.
    Full,
    /// WAL records since the last backup checkpoint.
    Incremental,
}

/// Metadata for a single backup segment (full or incremental).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSegment {
    /// Backup lineage containing this segment.
    pub chain_id: [u8; 32],
    /// Native store identity shared by every segment.
    pub store_id: [u8; 32],
    /// Native graph model (`0` LPG, `1` RDF, `2` Both).
    pub model: u8,
    /// Zero-based segment position in the chain.
    pub sequence: u64,
    /// Inclusive first source WAL file sequence (full images use their high-water mark).
    pub wal_start_sequence: u64,
    /// Inclusive ending source WAL file sequence.
    pub wal_end_sequence: u64,
    /// Canonical descriptor digest of the preceding segment; zero for the first.
    pub predecessor_digest: [u8; 32],
    /// Exact number of physical WAL frames carried by this segment; zero for full images.
    pub record_count: u64,
    /// Segment type.
    pub kind: BackupKind,
    /// File name (relative to backup directory).
    pub filename: String,
    /// Start epoch (inclusive).
    pub start_epoch: EpochId,
    /// End epoch (inclusive).
    pub end_epoch: EpochId,
    /// CRC-32 checksum of the segment file.
    pub checksum: u32,
    /// Domain-separated BLAKE3 digest of the exact segment bytes.
    pub content_digest: [u8; 32],
    /// Ending durable WorldCut encoded with this segment.
    pub world_cut: Option<Vec<u8>>,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Timestamp when this backup was created (ms since UNIX epoch).
    pub created_at_ms: u64,
}

/// Tracks the full backup chain for a database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    /// Manifest format version.
    pub version: u32,
    /// Random identity of this backup lineage.
    pub chain_id: [u8; 32],
    /// Identity of the source store captured by this chain.
    pub store_id: [u8; 32],
    /// Native model captured by this chain (`0` LPG, `1` RDF, `2` Both).
    pub model: u8,
    /// Monotonically increasing metadata generation.
    pub generation: u64,
    /// Encoded WorldCut for the most recent segment, when available.
    pub world_cut: Option<Vec<u8>>,
    /// Ordered list of backup segments (full first, then incrementals).
    pub segments: Vec<BackupSegment>,
}

impl BackupManifest {
    /// Creates a new empty manifest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: chain::VERSION,
            chain_id: [0; 32],
            store_id: [0; 32],
            model: 0,
            generation: 0,
            world_cut: None,
            segments: Vec::new(),
        }
    }

    /// Returns the most recent full backup segment, if any.
    #[must_use]
    pub fn latest_full(&self) -> Option<&BackupSegment> {
        self.segments
            .iter()
            .rev()
            .find(|s| s.kind == BackupKind::Full)
    }

    /// Returns incremental segments after the given epoch, in order.
    pub fn incrementals_after(&self, epoch: EpochId) -> Vec<&BackupSegment> {
        self.segments
            .iter()
            .filter(|s| s.kind == BackupKind::Incremental && s.start_epoch > epoch)
            .collect()
    }

    /// Returns the epoch range covered by this manifest.
    #[must_use]
    pub fn epoch_range(&self) -> Option<(EpochId, EpochId)> {
        let first = self.segments.first()?;
        let last = self.segments.last()?;
        Some((first.start_epoch, last.end_epoch))
    }
}

impl Default for BackupManifest {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks the WAL position of the last completed backup.
///
/// Persisted as `backup_cursor.meta` in the WAL directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackupCursor {
    /// Backup lineage whose manifest this cursor advances.
    pub chain_id: [u8; 32],
    /// Published manifest generation.
    pub generation: u64,
    /// Digest of the canonical manifest envelope for this generation.
    pub manifest_digest: [u8; 32],
    /// The epoch up to which WAL records have been backed up.
    pub backed_up_epoch: EpochId,
    /// The WAL log sequence number at the time of the last backup.
    pub log_sequence: u64,
    /// Timestamp of the last backup.
    pub timestamp_ms: u64,
}

fn validate_manifest(manifest: &BackupManifest) -> Result<()> {
    let invalid = || Error::Serialization("inconsistent backup segment metadata".into());
    if manifest.version != chain::VERSION || manifest.model > 2 || manifest.segments.is_empty() {
        return Err(invalid());
    }
    chain::validate_digest(&manifest.chain_id)?;
    chain::validate_digest(&manifest.store_id)?;
    chain::validate_manifest_segments(&manifest.segments)?;
    let mut predecessor = [0; 32];
    let mut previous: Option<&BackupSegment> = None;
    for (index, segment) in manifest.segments.iter().enumerate() {
        if segment.chain_id != manifest.chain_id
            || segment.store_id != manifest.store_id
            || segment.model != manifest.model
            || segment.sequence != u64::try_from(index).map_err(|_| invalid())?
            || segment.predecessor_digest != predecessor
            || segment.wal_start_sequence > segment.wal_end_sequence
        {
            return Err(invalid());
        }
        chain::validate_digest(&segment.content_digest)?;
        let cut = grafeo_common::types::WorldCut::decode(
            segment.world_cut.as_deref().ok_or_else(invalid)?,
        )
        .map_err(|error| Error::Serialization(format!("invalid backup WorldCut: {error}")))?;
        if cut.store_id().into_bytes() != manifest.store_id
            || cut.descriptor().graph_model().as_u8() != manifest.model
            || cut.epoch() != segment.end_epoch
        {
            return Err(invalid());
        }
        match segment.kind {
            BackupKind::Full => {
                if segment.record_count != 0
                    || segment.wal_start_sequence != segment.wal_end_sequence
                {
                    return Err(invalid());
                }
                if let Some(prior) = previous
                    && segment.wal_end_sequence < prior.wal_end_sequence
                {
                    return Err(invalid());
                }
            }
            BackupKind::Incremental => {
                let prior = previous.ok_or_else(invalid)?;
                if segment.record_count == 0
                    || segment.size_bytes <= BACKUP_HEADER_SIZE as u64
                    || segment.start_epoch <= prior.end_epoch
                    || prior.wal_end_sequence.checked_add(1) != Some(segment.wal_start_sequence)
                {
                    return Err(invalid());
                }
            }
        }
        predecessor = chain::segment_digest(segment)?;
        previous = Some(segment);
    }
    if manifest.world_cut.as_deref() != previous.and_then(|segment| segment.world_cut.as_deref()) {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(feature = "lpg")]
fn predecessor_digest(manifest: &BackupManifest) -> Result<[u8; 32]> {
    manifest
        .segments
        .last()
        .map_or(Ok([0; 32]), chain::segment_digest)
}

// ── Manifest I/O ───────────────────────────────────────────────────

const MANIFEST_FILENAME: &str = "backup_manifest.json";
// Cursor wire data stays in the engine; fixed artifact names belong to storage.

/// Reads the backup manifest from a backup directory.
///
/// Returns `None` if no manifest exists.
///
/// # Errors
///
/// Returns an error if the manifest file exists but cannot be read or parsed.
pub fn read_manifest(backup_dir: &Path) -> Result<Option<BackupManifest>> {
    publication::recover_manifest(backup_dir)
}

/// Writes the advisory manifest locator to a backup directory.
///
/// This does not publish an immutable committed generation by itself.
///
/// Uses write-to-temp-then-rename for atomicity.
///
/// # Errors
///
/// Returns an error if the manifest cannot be written.
pub fn write_manifest(backup_dir: &Path, manifest: &BackupManifest) -> Result<()> {
    let path = backup_dir.join(MANIFEST_FILENAME);
    let temp_path = backup_dir.join(format!("{MANIFEST_FILENAME}.tmp"));

    validate_manifest(manifest)?;
    let data = chain::encode_manifest(manifest)?;

    std::fs::create_dir_all(backup_dir)
        .map_err(|e| Error::Internal(format!("failed to create backup directory: {e}")))?;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temp_path, &path)
        .map_err(|e| Error::Internal(format!("failed to finalize backup manifest: {e}")))?;

    #[cfg(unix)]
    std::fs::File::open(backup_dir)?.sync_all()?;
    Ok(())
}

// ── Backup cursor I/O ──────────────────────────────────────────────

/// Reads the backup cursor from a WAL directory.
///
/// Returns `None` if no cursor exists (no backup has been taken).
///
/// # Errors
///
/// Returns an error if the cursor file exists but cannot be read.
pub fn read_backup_cursor(
    capture: &mut grafeo_storage::wal::WalCapture<'_>,
) -> Result<Option<BackupCursor>> {
    let Some(data) = capture.read_backup_cursor_bytes()? else {
        return Ok(None);
    };
    let cursor = chain::decode_cursor(&data)?;
    chain::validate_digest(&cursor.chain_id)?;
    chain::validate_digest(&cursor.manifest_digest)?;
    if cursor.backed_up_epoch == EpochId::PENDING {
        return Err(Error::Serialization("invalid current backup cursor".into()));
    }
    Ok(Some(cursor))
}

#[cfg(all(test, feature = "lpg"))]
fn manifest_digest(manifest: &BackupManifest) -> Result<[u8; 32]> {
    Ok(chain::digest(
        b"grafeo/backup-v2/manifest",
        &chain::encode_manifest(manifest)?,
    ))
}

/// Writes the backup cursor to a WAL directory.
///
/// Uses write-to-temp-then-rename for atomicity.
///
/// # Errors
///
/// Returns an error if the cursor cannot be written.
pub fn write_backup_cursor(
    capture: &mut grafeo_storage::wal::WalCapture<'_>,
    cursor: &BackupCursor,
) -> Result<()> {
    let data = chain::encode_cursor(cursor)?;
    capture.write_backup_cursor_bytes(&data)
}

// ── Incremental backup file format ─────────────────────────────────

/// Magic bytes for incremental backup files.
pub const BACKUP_MAGIC: [u8; 4] = *b"GBAK";
/// Current backup file version.
pub const BACKUP_VERSION: u32 = 2;

/// Header for an incremental backup file.
///
/// ```text
/// [magic: 4 bytes "GBAK"]
/// [version: u32 LE]
/// [start_epoch: u64 LE]
/// [end_epoch: u64 LE]
/// [record_count: u64 LE]
/// ... WAL frames ...
/// ```
pub const BACKUP_HEADER_SIZE: usize = 32;

/// Writes the incremental backup file header.
pub fn write_backup_header(
    buf: &mut Vec<u8>,
    start_epoch: EpochId,
    end_epoch: EpochId,
    record_count: u64,
) {
    buf.extend_from_slice(&BACKUP_MAGIC);
    buf.extend_from_slice(&BACKUP_VERSION.to_le_bytes());
    buf.extend_from_slice(&start_epoch.as_u64().to_le_bytes());
    buf.extend_from_slice(&end_epoch.as_u64().to_le_bytes());
    buf.extend_from_slice(&record_count.to_le_bytes());
}

/// Reads and validates the incremental backup file header.
///
/// Returns `(start_epoch, end_epoch, record_count)` on success.
///
/// # Errors
///
/// Returns an error if the header is invalid.
///
/// # Panics
///
/// Cannot panic: all slice indexing is bounds-checked by the length guard.
pub fn read_backup_header(data: &[u8]) -> Result<(EpochId, EpochId, u64)> {
    if data.len() < BACKUP_HEADER_SIZE {
        return Err(Error::Internal(
            "incremental backup file too short".to_string(),
        ));
    }
    if data[0..4] != BACKUP_MAGIC {
        return Err(Error::Internal(
            "invalid backup file magic bytes".to_string(),
        ));
    }
    let version = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| Error::Serialization("backup header version is truncated".into()))?,
    );
    if version != BACKUP_VERSION {
        return Err(Error::Internal(format!(
            "unsupported backup version {version}, current version is {BACKUP_VERSION}"
        )));
    }
    let start_epoch =
        EpochId::new(u64::from_le_bytes(data[8..16].try_into().map_err(
            |_| Error::Serialization("backup header start is truncated".into()),
        )?));
    let end_epoch =
        EpochId::new(u64::from_le_bytes(data[16..24].try_into().map_err(
            |_| Error::Serialization("backup header end is truncated".into()),
        )?));
    let record_count = u64::from_le_bytes(
        data[24..32]
            .try_into()
            .map_err(|_| Error::Serialization("backup header count is truncated".into()))?,
    );
    Ok((start_epoch, end_epoch, record_count))
}

/// Returns the timestamp in milliseconds since UNIX epoch.
#[cfg(any(test, feature = "lpg"))]
pub(super) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}

// ── Backup operations (called from GrafeoDB) ───────────────────────

#[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
use grafeo_storage::file::GrafeoFileManager;
#[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
use grafeo_storage::wal::LpgWal;

/// Digest the same admitted descriptor that will supply the image copy.
fn image_digest(
    source: &mut grafeo_storage::file::ContainerCapture<'_>,
    size: u64,
) -> Result<(u32, [u8; 32])> {
    struct Sink {
        crc: crc32fast::Hasher,
        digest: blake3::Hasher,
    }
    impl std::io::Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.crc.update(bytes);
            self.digest.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let domain = b"grafeo/backup-v2/segment";
    let mut digest = blake3::Hasher::new();
    digest.update(&(domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(&size.to_le_bytes());
    let mut sink = Sink {
        crc: crc32fast::Hasher::new(),
        digest,
    };
    if source.write_image_to(&mut sink)? != size {
        return Err(Error::Storage(StorageError::Corruption(
            "full backup image size mismatch".into(),
        )));
    }
    Ok((sink.crc.finalize(), *sink.digest.finalize().as_bytes()))
}

/// Creates a full backup by copying the .grafeo container file.
///
/// 1. Copies the container file to the backup directory via the locked handle.
/// 2. Updates the manifest and backup cursor.
///
/// Uses [`GrafeoFileManager::copy_to`] instead of `std::fs::copy()` so the
/// copy reads through the already-locked file handle. `std::fs::copy()` opens
/// a new handle, which fails on Windows when an exclusive lock is held.
///
/// # Errors
///
/// Returns an error if the database has no file manager, or if I/O fails.
#[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
pub(super) fn do_backup_full(
    backup_dir: &Path,
    fm: &GrafeoFileManager,
    wal: Option<&LpgWal>,
    current_epoch: EpochId,
    store_id: [u8; 32],
    model: u8,
    world_cut: Option<Vec<u8>>,
) -> Result<BackupSegment> {
    validate_committed_backup_epoch(current_epoch, "full backup")?;

    let _chain_owner =
        grafeo_storage::file::ContainerDestination::acquire(backup_dir.join(MANIFEST_FILENAME))?;

    // Determine backup filename
    let mut manifest = match read_manifest(backup_dir)? {
        Some(manifest) => manifest,
        None => BackupManifest {
            version: chain::VERSION,
            chain_id: chain::chain_id()?,
            store_id,
            model,
            generation: 0,
            world_cut: None,
            segments: Vec::new(),
        },
    };
    if manifest.store_id != [0; 32] && manifest.store_id != store_id {
        return Err(Error::InvalidValue(
            "backup source store identity differs from the existing chain".into(),
        ));
    }
    if manifest.model != model && !manifest.segments.is_empty() {
        return Err(Error::InvalidValue(
            "backup graph model differs from the existing chain".into(),
        ));
    }
    manifest.store_id = store_id;
    manifest.model = model;
    let next_generation = manifest
        .generation
        .checked_add(1)
        .ok_or(Error::Storage(StorageError::Full))?;
    let segment_idx = manifest.segments.len();
    let sequence = u64::try_from(segment_idx).map_err(|_| Error::Storage(StorageError::Full))?;
    let predecessor_digest = predecessor_digest(&manifest)?;
    let filename = format!("backup_full_{segment_idx:04}.grafeo");
    if backup_dir.join(&filename).try_exists()? {
        return Err(Error::Serialization(
            "incomplete backup generation: full segment already exists".into(),
        ));
    }
    let mut destination =
        grafeo_storage::file::ContainerDestination::acquire(backup_dir.join(&filename))?;

    // Copy the .grafeo file to the backup directory through the locked handle
    let mut source = fm.capture()?;
    let mut wal = wal.map(LpgWal::capture).transpose()?;
    if !manifest.segments.is_empty()
        && let Some(capture) = wal.as_mut()
    {
        publication::recover_cursor(capture, &manifest)?;
    }
    let file_size = source.copy_to_destination(&mut destination)?;

    let (checksum, content_digest) = image_digest(&mut source, file_size)?;
    drop(destination);
    let copied = GrafeoFileManager::open_read_only(backup_dir.join(&filename))?;
    let observed = image_digest(&mut copied.capture()?, file_size)?;
    if observed != (checksum, content_digest) {
        return Err(Error::Serialization(
            "full backup reread differs from source".into(),
        ));
    }
    drop(copied);
    grafeo_common::testing::crash::maybe_crash("backup:segment_validated");

    let wal_sequence = wal
        .as_ref()
        .map_or(0, grafeo_storage::wal::WalCapture::current_log_sequence);
    let segment = BackupSegment {
        chain_id: manifest.chain_id,
        store_id: manifest.store_id,
        model: manifest.model,
        sequence,
        wal_start_sequence: wal_sequence,
        wal_end_sequence: wal_sequence,
        predecessor_digest,
        record_count: 0,
        kind: BackupKind::Full,
        filename,
        start_epoch: EpochId::new(0),
        end_epoch: current_epoch,
        checksum,
        content_digest,
        world_cut: world_cut.clone(),
        size_bytes: file_size,
        created_at_ms: now_ms(),
    };

    manifest.segments.push(segment.clone());
    manifest.generation = next_generation;
    manifest.world_cut = world_cut;
    // Seal the inclusive source cut before acknowledging either immutable side.
    if let Some(capture) = wal.as_mut() {
        capture.rotate()?;
    }
    publication::publish(backup_dir, &manifest, wal.as_mut())?;

    Ok(segment)
}

/// Creates an incremental backup containing WAL records since the last backup.
///
/// Reads WAL log files from the backup cursor's position through one frozen
/// segment boundary and copies the raw frames into a backup segment file. The
/// caller retains GrafeoDB's quiescent capture guard for this complete call.
///
/// # Errors
///
/// Returns an error if no full backup or cursor exists, the admitted epoch
/// range is impossible, or the selected WAL window has missing files or frames.
/// The retained interval stays pinned through both durable generation barriers.
#[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
pub(super) fn do_backup_incremental(
    backup_dir: &Path,
    wal: &LpgWal,
    current_epoch: EpochId,
    world_cut: Vec<u8>,
) -> Result<BackupSegment> {
    validate_committed_backup_epoch(current_epoch, "incremental backup")?;

    let _chain_owner =
        grafeo_storage::file::ContainerDestination::acquire(backup_dir.join(MANIFEST_FILENAME))?;
    let manifest = read_manifest(backup_dir)?.ok_or_else(|| {
        Error::Internal("no backup manifest found; run a full backup first".to_string())
    })?;
    if manifest.version != chain::VERSION
        || manifest.chain_id == [0; 32]
        || manifest.store_id == [0; 32]
    {
        return Err(Error::Serialization(
            "incremental backup requires an authenticated v2 full chain".into(),
        ));
    }

    if manifest.latest_full().is_none() {
        return Err(Error::Internal(
            "no full backup in manifest; run a full backup first".to_string(),
        ));
    }
    let next_generation = manifest
        .generation
        .checked_add(1)
        .ok_or(Error::Storage(StorageError::Full))?;

    let mut capture = wal.capture()?;
    #[cfg(feature = "encryption")]
    if capture.is_encrypted() {
        return Err(Error::InvalidValue(
            "encrypted incremental backup requires a keyed restore contract".into(),
        ));
    }
    let cursor = publication::recover_cursor(&mut capture, &manifest)?;
    let start_epoch = incremental_start_epoch(cursor.backed_up_epoch)?;
    let wal_start_sequence = cursor
        .log_sequence
        .checked_add(1)
        .ok_or(Error::Storage(StorageError::Full))?;
    let cut = grafeo_common::types::WorldCut::decode(&world_cut)
        .map_err(|error| Error::Serialization(format!("invalid incremental WorldCut: {error}")))?;
    if cut.store_id().into_bytes() != manifest.store_id
        || cut.descriptor().graph_model().as_u8() != manifest.model
        || cut.epoch() != current_epoch
    {
        return Err(Error::Serialization(
            "incremental source identity differs from chain".into(),
        ));
    }
    if current_epoch < start_epoch {
        return Err(Error::InvalidValue(format!(
            "incremental backup epoch {} precedes the first unbacked committed epoch {}",
            current_epoch.as_u64(),
            start_epoch.as_u64()
        )));
    }

    // Register retention before selecting the physical interval. The cursor
    // read uses the existing owner; reacquisition verifies that publication
    // did not change it while retirement and lease registration serialized.
    drop(capture);
    let retention = wal.retain_from(wal_start_sequence)?;
    let mut capture = wal.capture_with_lease(&retention)?;
    if publication::recover_cursor(&mut capture, &manifest)? != cursor {
        return Err(Error::Serialization(
            "backup cursor changed before retained capture".into(),
        ));
    }

    // Freeze an exact inclusive segment boundary before copying. The caller's
    // quiescent capture guard excludes commits and concurrent backups; rotation
    // fsyncs the old active file and directs every later append to a strictly
    // greater sequence. Cursor publication can therefore never skip a frame
    // appended while an earlier segment was being copied.
    let end_sequence = capture.current_log_sequence();
    let candidate_files = capture
        .segments()?
        .into_iter()
        .filter(|segment| {
            segment.sequence() > cursor.log_sequence && segment.sequence() <= end_sequence
        })
        .collect::<Vec<_>>();
    if !candidate_files
        .iter()
        .any(|segment| segment.size_bytes() > 0)
    {
        return Err(Error::Internal(
            "no new WAL records since last backup".to_string(),
        ));
    }
    let expected_count = end_sequence
        .checked_sub(wal_start_sequence)
        .and_then(|distance| distance.checked_add(1))
        .ok_or_else(|| Error::Serialization("invalid incremental WAL interval".into()))?;
    if u64::try_from(candidate_files.len()).ok() != Some(expected_count)
        || candidate_files.iter().enumerate().any(|(index, segment)| {
            u64::try_from(index)
                .ok()
                .and_then(|offset| wal_start_sequence.checked_add(offset))
                != Some(segment.sequence())
        })
    {
        return Err(Error::Serialization(
            "incremental WAL interval is not continuous".into(),
        ));
    }

    // Read WAL files from cursor position onward
    let mut wal_data = Vec::new();
    let mut record_count = 0u64;

    for segment in &candidate_files {
        let file_bytes = capture.read_segment(segment)?;

        if !file_bytes.is_empty() {
            wal_data.extend_from_slice(&file_bytes);
            record_count = record_count
                .checked_add(capture.count_frames(&file_bytes)?)
                .ok_or(Error::Storage(StorageError::Full))?;
        }
    }

    if wal_data.is_empty() {
        return Err(Error::Storage(StorageError::Corruption(
            "the sealed incremental-backup WAL cut became empty while it was copied".to_string(),
        )));
    }

    let end_epoch = current_epoch;

    // Write incremental backup file
    let segment_idx = manifest.segments.len();
    let sequence = u64::try_from(segment_idx).map_err(|_| Error::Storage(StorageError::Full))?;
    let predecessor_digest = predecessor_digest(&manifest)?;
    capture
        .rotate()
        .map_err(|error| error.with_context("failed to seal the WAL cut for incremental backup"))?;
    let filename = format!("backup_incr_{segment_idx:04}.wal");
    let dest_path = backup_dir.join(&filename);

    let mut output = Vec::new();
    write_backup_header(&mut output, start_epoch, end_epoch, record_count);
    output.extend_from_slice(&wal_data);

    publication::write_segment(&dest_path, &output)?;

    let checksum = crc32fast::hash(&output);
    let content_digest = chain::digest(b"grafeo/backup-v2/segment", &output);
    let segment = BackupSegment {
        chain_id: manifest.chain_id,
        store_id: manifest.store_id,
        model: manifest.model,
        sequence,
        wal_start_sequence,
        wal_end_sequence: end_sequence,
        predecessor_digest,
        record_count,
        kind: BackupKind::Incremental,
        filename,
        start_epoch,
        end_epoch,
        checksum,
        content_digest,
        world_cut: Some(world_cut.clone()),
        size_bytes: output.len() as u64,
        created_at_ms: now_ms(),
    };

    // Update manifest
    let mut manifest = manifest;
    manifest.world_cut = Some(world_cut);
    manifest.segments.push(segment.clone());
    manifest.generation = next_generation;
    publication::publish(backup_dir, &manifest, Some(&mut capture))?;

    Ok(segment)
}

#[cfg(all(
    feature = "lpg",
    any(test, all(feature = "wal", feature = "grafeo-file"))
))]
/// Rejects an impossible transaction-manager publication coordinate before a
/// backup can mutate its destination, manifest, cursor, or WAL boundary.
///
/// # Errors
///
/// Returns storage corruption when `epoch` is the reserved uncommitted
/// sentinel rather than a committed publication coordinate.
pub(super) fn validate_committed_backup_epoch(epoch: EpochId, operation: &str) -> Result<()> {
    if epoch == EpochId::PENDING {
        return Err(Error::Storage(StorageError::Corruption(format!(
            "the transaction manager exposes the reserved PENDING epoch for {operation}"
        ))));
    }
    Ok(())
}

#[cfg(any(test, all(feature = "wal", feature = "grafeo-file", feature = "lpg")))]
fn incremental_start_epoch(backed_up_epoch: EpochId) -> Result<EpochId> {
    if backed_up_epoch == EpochId::PENDING {
        return Err(Error::Storage(StorageError::Corruption(
            "backup cursor contains the reserved PENDING epoch".to_string(),
        )));
    }
    backed_up_epoch
        .as_u64()
        .checked_add(1)
        .filter(|successor| *successor < EpochId::PENDING.as_u64())
        .map(EpochId::new)
        .ok_or_else(|| {
            Error::Storage(StorageError::Full)
                .with_context("committed epoch identity space is exhausted for incremental backup")
        })
}

// ── Restore ────────────────────────────────────────────────────────

/// Validates and replays a complete selected chain in owned detached storage.
/// The destination changes only after the materialized target cut reopens and
/// verifies, with all staging handles closed before the atomic replacement.
///
/// # Errors
/// Returns an error for unsupported targets, corrupt or discontinuous sources,
/// destination contention/existing WAL, replay/verification or installation I/O.
pub(super) fn do_restore_to_epoch(
    backup_dir: &Path,
    target_epoch: EpochId,
    output_path: &Path,
) -> Result<()> {
    if target_epoch == EpochId::PENDING {
        return Err(Error::InvalidValue(
            "PENDING is not a committed restore target".into(),
        ));
    }
    let manifest = read_manifest(backup_dir)?
        .ok_or_else(|| Error::InvalidValue("no backup manifest found".into()))?;
    let last = manifest
        .segments
        .last()
        .ok_or_else(|| Error::Serialization("empty backup chain".into()))?;
    if target_epoch > last.end_epoch {
        return Err(Error::InvalidValue(
            "restore target exceeds retained backup coverage".into(),
        ));
    }
    let full_index = manifest
        .segments
        .iter()
        .rposition(|segment| segment.kind == BackupKind::Full && segment.end_epoch <= target_epoch)
        .ok_or_else(|| Error::InvalidValue("no full backup covers the restore target".into()))?;
    let full = &manifest.segments[full_index];
    // A restore must never replace any object in its own source chain. This
    // also catches an existing output symlink resolving into the backup tree.
    let source_directory = std::fs::canonicalize(backup_dir)?;
    if output_path.exists() && std::fs::canonicalize(output_path)?.starts_with(&source_directory) {
        return Err(Error::InvalidValue(
            "restore output aliases its backup source".into(),
        ));
    }
    let full_path = backup_dir.join(&full.filename);
    let source_db = open_restore_image(
        grafeo_storage::file::GrafeoFileManager::open_backup_image(&full_path)?,
        None,
    )?;
    verify_segment_cut(&source_db, full)?;
    let source_manager = source_db
        .file_manager
        .as_ref()
        .ok_or_else(|| Error::Internal("verified backup source has no file manager".into()))?;
    let mut source = source_manager.capture()?;
    let (checksum, digest) = image_digest(&mut source, full.size_bytes)?;
    if checksum != full.checksum || digest != full.content_digest {
        return Err(Error::Storage(StorageError::Corruption(
            "full backup image checksum or content digest mismatch".into(),
        )));
    }
    let destination = grafeo_storage::file::ContainerDestination::acquire(output_path)?;
    if destination.path().starts_with(&source_directory) {
        return Err(Error::InvalidValue(
            "restore output is inside its backup source".into(),
        ));
    }
    let mut stage = destination.into_restore_stage()?;
    let mut import = stage.import()?;
    let mut selected_end = full;
    // Include the segment containing the target (including legal commit gaps).
    // Every selected file is read exactly once and the verified bytes go
    // directly to owned import; no pathname is reopened after validation.
    for segment in &manifest.segments[full_index + 1..] {
        if selected_end.end_epoch >= target_epoch {
            break;
        }
        if segment.kind != BackupKind::Incremental {
            return Err(Error::Serialization(
                "selected restore interval crosses another full image".into(),
            ));
        }
        let bytes = read_increment(backup_dir, segment)
            .map_err(|error| error.with_context("read backup increment"))?;
        import
            .write_segment(segment.sequence, &bytes[BACKUP_HEADER_SIZE..])
            .map_err(|error| error.with_context("import backup increment"))?;
        selected_end = segment;
    }
    let mut imported = import.into_recovery()?;
    // Recovery validates complete frames and committed groups across all
    // selected segments, including the suffix beyond an inside-segment target.
    let report = imported
        .recover_report()
        .map_err(|error| error.with_context("validate imported backup WAL"))?;
    let replay_allocation_floor = report
        .max_transaction_id
        .map(|id| super::validated_transaction_allocation_floor(id.as_u64(), "backup import"))
        .transpose()?
        .flatten();
    let all_records = report.committed;
    let target_records = imported
        .recover_until_epoch(target_epoch)
        .map_err(|error| error.with_context("select target backup records"))?;
    drop(imported.seal()?);

    if target_epoch < selected_end.end_epoch {
        // Prove the selected segment's ending cut too: an early target must
        // not hide a foreign or incomplete committed suffix in that segment.
        stage.copy_validation_from(&mut source)?;
        write_replay(stage.validation_replay()?, &all_records)?;
        let validation =
            open_restore_image(stage.open_validation()?, Some(stage.validation_replay()?))?;
        verify_segment_cut(&validation, selected_end)?;
        validation
            .close()
            .map_err(|error| error.with_context("materialize validation restore image"))?;
        drop(validation);
    }
    stage
        .copy_from(&mut source)
        .map_err(|error| error.with_context("copy target restore image"))?;
    write_replay(stage.replay()?, &target_records)
        .map_err(|error| error.with_context("write target restore WAL"))?;
    let restored = open_restore_image(
        stage
            .open()
            .map_err(|error| error.with_context("admit target restore image"))?,
        Some(
            stage
                .replay()
                .map_err(|error| error.with_context("admit target replay"))?,
        ),
    )?;
    if target_epoch == selected_end.end_epoch {
        verify_segment_cut(&restored, selected_end)?;
    }
    let target_cut = restored
        .world_cut()?
        .encode()
        .map_err(|error| Error::Serialization(format!("restored WorldCut: {error}")))?;
    // Even aborted or target-excluded transactions consumed identities in the
    // retained source interval; do not reuse them in the restored database.
    if let Some(floor) = replay_allocation_floor {
        restored
            .transaction_manager
            .advance_next_transaction_id(floor);
    }
    let allocation_floor = restored.transaction_manager.last_assigned_transaction_id();
    restored
        .close()
        .map_err(|error| error.with_context("materialize target restore image"))?;
    drop(restored);
    let reopened = open_restore_image(
        stage
            .open_read_only()
            .map_err(|error| error.with_context("admit materialized target image"))?,
        None,
    )?;
    let reopened_cut = reopened
        .world_cut()?
        .encode()
        .map_err(|error| Error::Serialization(format!("reopened WorldCut: {error}")))?;
    if target_cut != reopened_cut
        || reopened.transaction_manager.last_assigned_transaction_id() != allocation_floor
    {
        return Err(Error::Storage(StorageError::Corruption(
            "materialized restore changed its WorldCut or transaction allocation floor".into(),
        )));
    }
    reopened.close()?;
    drop(reopened);
    stage.install()
}

fn verify_segment_cut(db: &super::GrafeoDB, segment: &BackupSegment) -> Result<()> {
    let actual = db
        .world_cut()?
        .encode()
        .map_err(|error| Error::Serialization(format!("restore WorldCut: {error}")))?;
    if Some(actual.as_slice()) != segment.world_cut.as_deref()
        || db.graph_model().as_u8() != segment.model
    {
        return Err(Error::Storage(StorageError::Corruption(
            "restored segment WorldCut differs from its manifest".into(),
        )));
    }
    Ok(())
}

fn open_restore_image(
    file: grafeo_storage::file::GrafeoFileManager,
    recovery: Option<grafeo_storage::wal::WalRecovery>,
) -> Result<super::GrafeoDB> {
    let config = if file.is_read_only() {
        crate::Config::read_only(file.path())
    } else {
        crate::Config::persistent(file.path())
    };
    super::GrafeoDB::with_owned_config(config, Some(file), recovery)
        .map_err(|error| error.with_context("open owned restore image"))
}

fn write_replay(
    mut recovery: grafeo_storage::wal::WalRecovery,
    records: &[grafeo_storage::wal::WalRecord],
) -> Result<()> {
    recovery.recover()?;
    let wal = grafeo_storage::wal::LpgWal::from_manager(
        recovery.into_wal(grafeo_storage::wal::WalConfig::default())?,
    );
    for record in records {
        wal.log(record)?;
    }
    drop(wal.seal()?);
    Ok(())
}

fn read_increment(backup_dir: &Path, segment: &BackupSegment) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = grafeo_storage::file::open_backup_source(&backup_dir.join(&segment.filename))?;
    let length = file.metadata()?.len();
    if length != segment.size_bytes {
        return Err(Error::Storage(StorageError::Corruption(
            "incremental backup size mismatch".into(),
        )));
    }
    let capacity = usize::try_from(length)
        .map_err(|_| Error::Serialization("incremental backup size overflow".into()))?;
    let bound = length
        .checked_add(1)
        .ok_or_else(|| Error::Serialization("incremental backup read bound overflow".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Storage(StorageError::Full))?;
    file.take(bound).read_to_end(&mut bytes)?;
    if bytes.len() != capacity
        || crc32fast::hash(&bytes) != segment.checksum
        || chain::digest(b"grafeo/backup-v2/segment", &bytes) != segment.content_digest
    {
        return Err(Error::Storage(StorageError::Corruption(
            "incremental backup integrity mismatch".into(),
        )));
    }
    let (start, end, count) = read_backup_header(&bytes)?;
    if start != segment.start_epoch
        || end != segment.end_epoch
        || count != segment.record_count
        || grafeo_storage::wal::count_wal_frames(&bytes[BACKUP_HEADER_SIZE..])? != count
    {
        return Err(Error::Storage(StorageError::Corruption(
            "incremental backup header/frame count mismatch".into(),
        )));
    }
    Ok(bytes)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_cut(epoch: EpochId) -> Vec<u8> {
        use grafeo_common::types::{
            AuthoritativeFormat, Digest256, GraphModelTag, HistoryCompleteness, ModelFormatVersion,
            SchemaCut, StateDigest, StoreId, WorldCut, WorldCutDescriptor,
        };
        let descriptor = WorldCutDescriptor::new(
            StoreId::from_bytes([2; 32]).unwrap(),
            epoch,
            GraphModelTag::Lpg,
            vec![
                ModelFormatVersion::new(AuthoritativeFormat::Catalog, 1).unwrap(),
                ModelFormatVersion::new(AuthoritativeFormat::Lpg, 4).unwrap(),
            ],
            SchemaCut::new(1, Digest256::from_bytes([3; 32])).unwrap(),
            vec![],
            HistoryCompleteness::Complete,
        )
        .unwrap();
        WorldCut::seal(descriptor, StateDigest::snapshot_bytes(b"test image"))
            .unwrap()
            .encode()
            .unwrap()
    }

    fn bind_test_manifest(manifest: &mut BackupManifest) {
        let mut predecessor = [0; 32];
        for (index, segment) in manifest.segments.iter_mut().enumerate() {
            segment.sequence = index as u64;
            segment.predecessor_digest = predecessor;
            segment.content_digest = [3; 32];
            segment.world_cut = Some(test_cut(segment.end_epoch));
            predecessor = chain::segment_digest(segment).unwrap();
        }
        manifest.world_cut = manifest.segments.last().unwrap().world_cut.clone();
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn hostile_segment_metadata_rejects_before_destination_mutation() {
        let dir = TempDir::new().unwrap();
        let db = super::super::GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
        db.execute("INSERT (:N {v: 1})").unwrap();
        let backup = dir.path().join("backup");
        db.backup_full(&backup).unwrap();
        db.execute("INSERT (:N {v: 2})").unwrap();
        db.execute("INSERT (:N {v: 3})").unwrap();
        let increment = db.backup_incremental(&backup).unwrap();
        let valid = read_manifest(&backup).unwrap().unwrap();
        let mut epoch_gap = valid.clone();
        epoch_gap.segments[1].start_epoch = epoch_gap.segments[1].end_epoch;
        validate_manifest(&epoch_gap).unwrap();
        let destination = dir.path().join("destination.grafeo");
        std::fs::write(&destination, b"preserved destination").unwrap();
        type Mutation = fn(&mut BackupManifest);
        let mutations: [Mutation; 9] = [
            |m| m.segments[1].chain_id = [8; 32],
            |m| m.segments[1].store_id = [8; 32],
            |m| m.segments[1].model = 2,
            |m| m.segments[1].sequence += 1,
            |m| m.segments[1].predecessor_digest = [8; 32],
            |m| m.segments[1].wal_start_sequence += 1,
            |m| m.segments[1].record_count += 1,
            |m| m.segments[1].world_cut = None,
            |m| m.segments[0].record_count = 1,
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut hostile = valid.clone();
            mutate(&mut hostile);
            // Reseal the canonical envelope: this exercises semantic validation,
            // rather than merely detecting a damaged outer digest.
            let bytes = chain::encode_manifest(&hostile).unwrap();
            std::fs::write(backup.join(MANIFEST_FILENAME), bytes).unwrap();
            assert!(
                do_restore_to_epoch(&backup, increment.end_epoch, &destination).is_err(),
                "case {index}"
            );
            assert_eq!(
                std::fs::read(&destination).unwrap(),
                b"preserved destination",
                "case {index}"
            );
        }
        db.close().unwrap();
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn cursor_wal_coordinate_must_match_manifest_before_capture() {
        let dir = TempDir::new().unwrap();
        let db = super::super::GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
        db.execute("INSERT (:N {v: 1})").unwrap();
        let backup = dir.path().join("backup");
        db.backup_full(&backup).unwrap();
        db.execute("INSERT (:N {v: 2})").unwrap();
        let mut cursor = db.backup_cursor().unwrap().unwrap();
        cursor.log_sequence += 1;
        let wal = db.wal.as_ref().unwrap();
        write_backup_cursor(&mut wal.capture().unwrap(), &cursor).unwrap();
        let sequence = wal.current_sequence();
        let before = std::fs::read(backup.join(MANIFEST_FILENAME)).unwrap();
        let error = db.backup_incremental(&backup).unwrap_err();
        assert!(
            error.to_string().contains("cursor does not match"),
            "{error}"
        );
        assert_eq!(wal.current_sequence(), sequence);
        assert_eq!(
            std::fs::read(backup.join(MANIFEST_FILENAME)).unwrap(),
            before
        );
        db.close().unwrap();
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn foreign_chain_advisory_cursor_must_match_its_immutable_generation() {
        let dir = TempDir::new().unwrap();
        let db = super::super::GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
        db.execute("INSERT (:N {v: 1})").unwrap();
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        db.backup_full(&first).unwrap();
        db.backup_full(&second).unwrap();
        db.execute("INSERT (:N {v: 2})").unwrap();
        let mut cursor = db.backup_cursor().unwrap().unwrap();
        cursor.log_sequence += 1;
        let wal = db.wal.as_ref().unwrap();
        write_backup_cursor(&mut wal.capture().unwrap(), &cursor).unwrap();
        let sequence = wal.current_sequence();
        assert!(
            db.backup_incremental(&first)
                .unwrap_err()
                .to_string()
                .contains("cursor does not match")
        );
        assert_eq!(wal.current_sequence(), sequence);
        db.close().unwrap();
    }

    #[test]
    fn test_manifest_new() {
        let manifest = BackupManifest::new();
        assert_eq!(manifest.version, chain::VERSION);
        assert!(manifest.segments.is_empty());
        assert!(manifest.latest_full().is_none());
        assert!(manifest.epoch_range().is_none());
    }

    #[test]
    fn test_manifest_with_segments() {
        let mut manifest = BackupManifest::new();
        manifest.segments.push(BackupSegment {
            chain_id: [1; 32],
            store_id: [2; 32],
            model: 0,
            sequence: 0,
            wal_start_sequence: 0,
            wal_end_sequence: 0,
            predecessor_digest: [0; 32],
            record_count: 0,
            kind: BackupKind::Full,
            filename: "backup_full_0000.grafeo".to_string(),
            start_epoch: EpochId::new(0),
            end_epoch: EpochId::new(100),
            checksum: 12345,
            content_digest: [0; 32],
            world_cut: None,
            size_bytes: 1024,
            created_at_ms: 1000,
        });
        manifest.segments.push(BackupSegment {
            chain_id: [1; 32],
            store_id: [2; 32],
            model: 0,
            sequence: 0,
            wal_start_sequence: 0,
            wal_end_sequence: 0,
            predecessor_digest: [0; 32],
            record_count: 0,
            kind: BackupKind::Incremental,
            filename: "backup_incr_0001.wal".to_string(),
            start_epoch: EpochId::new(101),
            end_epoch: EpochId::new(200),
            checksum: 67890,
            content_digest: [0; 32],
            world_cut: None,
            size_bytes: 256,
            created_at_ms: 2000,
        });

        let full = manifest.latest_full().unwrap();
        assert_eq!(full.end_epoch, EpochId::new(100));

        let incrs = manifest.incrementals_after(EpochId::new(100));
        assert_eq!(incrs.len(), 1);
        assert_eq!(incrs[0].start_epoch, EpochId::new(101));

        let (start, end) = manifest.epoch_range().unwrap();
        assert_eq!(start, EpochId::new(0));
        assert_eq!(end, EpochId::new(200));
    }

    #[test]
    fn test_manifest_round_trip() {
        let dir = TempDir::new().unwrap();
        let mut manifest = BackupManifest::new();
        manifest.segments.push(BackupSegment {
            chain_id: [1; 32],
            store_id: [2; 32],
            model: 0,
            sequence: 0,
            wal_start_sequence: 0,
            wal_end_sequence: 0,
            predecessor_digest: [0; 32],
            record_count: 0,
            kind: BackupKind::Full,
            filename: "test.grafeo".to_string(),
            start_epoch: EpochId::new(0),
            end_epoch: EpochId::new(50),
            checksum: 0,
            content_digest: [0; 32],
            world_cut: None,
            size_bytes: 512,
            created_at_ms: 0,
        });
        manifest.chain_id = [1; 32];
        manifest.store_id = [2; 32];

        bind_test_manifest(&mut manifest);
        publication::publish(dir.path(), &manifest, None).unwrap();
        let loaded = read_manifest(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.segments.len(), 1);
        assert_eq!(loaded.segments[0].filename, "test.grafeo");
    }

    #[test]
    fn test_manifest_not_found() {
        let dir = TempDir::new().unwrap();
        assert!(read_manifest(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_backup_cursor_round_trip() {
        let dir = TempDir::new().unwrap();
        let cursor = BackupCursor {
            chain_id: [1; 32],
            generation: 1,
            manifest_digest: [2; 32],
            backed_up_epoch: EpochId::new(42),
            log_sequence: 7,
            timestamp_ms: 12345,
        };

        let wal = grafeo_storage::wal::WalManager::open(dir.path().join("wal")).unwrap();
        let mut capture = wal.capture().unwrap();
        write_backup_cursor(&mut capture, &cursor).unwrap();
        let loaded = read_backup_cursor(&mut capture).unwrap().unwrap();
        assert_eq!(loaded.backed_up_epoch, EpochId::new(42));
        assert_eq!(loaded.log_sequence, 7);
        assert_eq!(loaded.timestamp_ms, 12345);
    }

    #[test]
    fn test_backup_cursor_not_found() {
        let dir = TempDir::new().unwrap();
        let wal = grafeo_storage::wal::WalManager::open(dir.path().join("wal")).unwrap();
        assert!(
            read_backup_cursor(&mut wal.capture().unwrap())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn initial_backup_cursor_successor_is_epoch_one() {
        assert_eq!(
            incremental_start_epoch(EpochId::INITIAL).unwrap(),
            EpochId::new(1)
        );
    }

    #[test]
    fn committed_epoch_domain_has_no_successor_at_its_upper_boundary() {
        let error = incremental_start_epoch(EpochId::new(u64::MAX - 1)).unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    fn directory_image(path: &Path) -> Vec<(String, Vec<u8>)> {
        let mut image = std::fs::read_dir(path)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    std::fs::read(entry.path()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        image.sort_by(|left, right| left.0.cmp(&right.0));
        image
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn pending_transaction_manager_epoch_is_corruption_before_backup_mutation() {
        let root = TempDir::new().unwrap();
        let backup_dir = root.path().join("backup");
        let wal_dir = root.path().join("wal");
        let wal = LpgWal::open(&wal_dir).unwrap();
        let wal_before = directory_image(&wal_dir);

        let error = do_backup_incremental(&backup_dir, &wal, EpochId::PENDING, vec![]).unwrap_err();

        assert!(matches!(error, Error::Storage(StorageError::Corruption(_))));
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageCorrupted
        );
        assert!(!backup_dir.exists());
        assert_eq!(directory_image(&wal_dir), wal_before);
        assert_eq!(wal.current_sequence(), 0);
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn exhausted_cursor_rejection_preserves_manifest_cursor_and_wal() {
        let root = TempDir::new().unwrap();
        let backup_dir = root.path().join("backup");
        let wal_dir = root.path().join("wal");
        let wal = LpgWal::open(&wal_dir).unwrap();

        let mut manifest = BackupManifest::new();
        manifest.segments.push(BackupSegment {
            chain_id: [1; 32],
            store_id: [2; 32],
            model: 0,
            sequence: 0,
            wal_start_sequence: 0,
            wal_end_sequence: 0,
            predecessor_digest: [0; 32],
            record_count: 0,
            kind: BackupKind::Full,
            filename: "seed.grafeo".to_string(),
            start_epoch: EpochId::INITIAL,
            end_epoch: EpochId::new(u64::MAX - 1),
            checksum: 0,
            content_digest: [0; 32],
            world_cut: None,
            size_bytes: 0,
            created_at_ms: 0,
        });
        manifest.chain_id = [1; 32];
        manifest.store_id = [2; 32];
        bind_test_manifest(&mut manifest);
        let owner =
            grafeo_storage::file::ContainerDestination::acquire(backup_dir.join(MANIFEST_FILENAME))
                .unwrap();
        publication::publish(&backup_dir, &manifest, Some(&mut wal.capture().unwrap())).unwrap();
        drop(owner);
        write_backup_cursor(
            &mut wal.capture().unwrap(),
            &BackupCursor {
                chain_id: manifest.chain_id,
                generation: manifest.generation,
                manifest_digest: manifest_digest(&manifest).unwrap(),
                backed_up_epoch: EpochId::new(u64::MAX - 1),
                log_sequence: 0,
                timestamp_ms: 0,
            },
        )
        .unwrap();
        let backup_before = directory_image(&backup_dir);
        let wal_before = directory_image(&wal_dir);

        let error = do_backup_incremental(&backup_dir, &wal, EpochId::new(u64::MAX - 1), vec![])
            .unwrap_err();

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(
            error
                .to_string()
                .contains("committed epoch identity space is exhausted")
        );
        assert_eq!(directory_image(&backup_dir), backup_before);
        assert_eq!(directory_image(&wal_dir), wal_before);
        assert_eq!(wal.current_sequence(), 0);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn replace_committed_test_manifest(dir: &Path, manifest: &BackupManifest) {
        let last = manifest.segments.last().unwrap();
        use std::fmt::Write as _;
        let mut chain = String::new();
        for byte in manifest.chain_id {
            write!(chain, "{byte:02x}").unwrap();
        }
        let bytes = chain::encode_unchecked_test_manifest(manifest).unwrap();
        let manifest_digest = chain::digest(b"grafeo/backup-v2/manifest", &bytes);
        std::fs::write(dir.join(MANIFEST_FILENAME), &bytes).unwrap();
        std::fs::write(
            dir.join(format!(
                "backup_manifest_{chain}_{:020}.meta",
                manifest.generation
            )),
            bytes,
        )
        .unwrap();
        let cursor = BackupCursor {
            chain_id: manifest.chain_id,
            generation: manifest.generation,
            manifest_digest,
            backed_up_epoch: last.end_epoch,
            log_sequence: last.wal_end_sequence,
            timestamp_ms: 0,
        };
        std::fs::write(
            dir.join(format!(
                "backup_commit_{chain}_{:020}.meta",
                manifest.generation
            )),
            chain::encode_cursor(&cursor).unwrap(),
        )
        .unwrap();
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn checksummed_hostile_chain_topology_preserves_populated_destination() {
        let root = TempDir::new().unwrap();
        let db = super::super::GrafeoDB::open(root.path().join("source.grafeo")).unwrap();
        db.execute("INSERT (:N {n: 1})").unwrap();
        let backup = root.path().join("backup");
        db.backup_full(&backup).unwrap();
        for n in 2..=3 {
            db.execute(&format!("INSERT (:N {{n: {n}}})")).unwrap();
            db.backup_incremental(&backup).unwrap();
        }
        let original = read_manifest(&backup).unwrap().unwrap();
        let target = original.segments.last().unwrap().end_epoch;
        let output = root.path().join("output.grafeo");
        let sentinel = super::super::GrafeoDB::open(&output).unwrap();
        sentinel.execute("INSERT (:Sentinel {n: 99})").unwrap();
        let before_cut = sentinel.world_cut().unwrap();
        sentinel.close().unwrap();
        drop(sentinel);
        let before = std::fs::read(&output).unwrap();
        for case in 0..5 {
            let mut hostile = original.clone();
            match case {
                0 => {
                    hostile.segments[1].chain_id = [91; 32];
                }
                1 => {
                    hostile.segments.remove(1);
                }
                2 => hostile.segments.swap(1, 2),
                3 => {
                    hostile.segments[2] = hostile.segments[1].clone();
                }
                _ => {
                    hostile.segments[1].wal_start_sequence += 1;
                }
            }
            replace_committed_test_manifest(&backup, &hostile);
            assert!(
                do_restore_to_epoch(&backup, target, &output).is_err(),
                "case {case}"
            );
            assert_eq!(std::fs::read(&output).unwrap(), before, "case {case}");
            let unchanged = super::super::GrafeoDB::open_read_only(&output).unwrap();
            assert_eq!(unchanged.world_cut().unwrap(), before_cut, "case {case}");
            unchanged.close().unwrap();
        }
        db.close().unwrap();
    }

    #[cfg(feature = "lpg")]
    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn checksummed_foreign_increment_rejects_before_destination_replacement() {
        let root = TempDir::new().unwrap();
        let mut backups = Vec::new();
        for name in ["a", "b"] {
            let db =
                super::super::GrafeoDB::open(root.path().join(format!("{name}.grafeo"))).unwrap();
            db.execute("INSERT (:N {n: 1})").unwrap();
            let backup = root.path().join(name);
            db.backup_full(&backup).unwrap();
            db.execute(&format!("INSERT (:N {{source: '{name}'}})"))
                .unwrap();
            db.backup_incremental(&backup).unwrap();
            backups.push(backup);
            db.close().unwrap();
        }
        let mut manifest = read_manifest(&backups[0]).unwrap().unwrap();
        let other = read_manifest(&backups[1]).unwrap().unwrap();
        let foreign = &other.segments[1];
        let bytes = std::fs::read(backups[1].join(&foreign.filename)).unwrap();
        let segment = &mut manifest.segments[1];
        assert_eq!(segment.end_epoch, foreign.end_epoch);
        segment.size_bytes = bytes.len() as u64;
        segment.checksum = crc32fast::hash(&bytes);
        segment.content_digest = chain::digest(b"grafeo/backup-v2/segment", &bytes);
        segment.record_count = foreign.record_count;
        std::fs::write(backups[0].join(&segment.filename), &bytes).unwrap();
        let target = segment.end_epoch;
        replace_committed_test_manifest(&backups[0], &manifest);
        let output = root.path().join("output.grafeo");
        let sentinel = super::super::GrafeoDB::open(&output).unwrap();
        sentinel.execute("INSERT (:Sentinel)").unwrap();
        sentinel.close().unwrap();
        drop(sentinel);
        let before = std::fs::read(&output).unwrap();
        assert!(do_restore_to_epoch(&backups[0], target, &output).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), before);
        let unchanged = super::super::GrafeoDB::open_read_only(&output).unwrap();
        assert_eq!(unchanged.node_count(), 1);
        unchanged.close().unwrap();
    }

    #[test]
    fn test_backup_header_round_trip() {
        let mut buf = Vec::new();
        write_backup_header(&mut buf, EpochId::new(101), EpochId::new(200), 500);
        assert_eq!(buf.len(), BACKUP_HEADER_SIZE);

        let (start, end, count) = read_backup_header(&buf).unwrap();
        assert_eq!(start, EpochId::new(101));
        assert_eq!(end, EpochId::new(200));
        assert_eq!(count, 500);
    }

    #[test]
    fn test_backup_header_invalid_magic() {
        let data = vec![
            0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        assert!(read_backup_header(&data).is_err());
    }

    #[test]
    fn test_backup_header_too_short() {
        let data = vec![0, 0, 0, 0];
        assert!(read_backup_header(&data).is_err());
    }

    #[test]
    fn test_backup_kind_serialization() {
        let config = bincode::config::standard();
        let encoded = bincode::serde::encode_to_vec(BackupKind::Full, config).unwrap();
        let (parsed, _): (BackupKind, _) =
            bincode::serde::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(parsed, BackupKind::Full);
    }
}
