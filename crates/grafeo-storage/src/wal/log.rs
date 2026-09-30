//! WAL log file management.

use super::frame_buffer::encode_frame;
use super::ownership::{SealedWal, WalLease};
use super::{WalEntry, WalRecord, validate_wal_frame_payload_len};
use crate::ownership::{checked_file, validate_file};
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result, StorageError};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Checkpoint metadata stored in a separate file.
///
/// This file is written atomically (via rename) during checkpoint and read
/// during recovery to determine which WAL files can be skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Current checkpoint schema, coupled to the current WAL group envelope.
    pub format_version: u8,
    /// The epoch at which the checkpoint was taken.
    pub epoch: EpochId,
    /// The log sequence number at the time of checkpoint.
    pub log_sequence: u64,
    /// Timestamp of the checkpoint (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,
    /// Transaction ID at checkpoint.
    pub transaction_id: TransactionId,
    /// First segment whose complete group history must still be available.
    /// Earlier files may remain after interrupted retirement or conservative
    /// cleanup, but their groups are no longer replay or append authority.
    pub retired_before: u64,
}

/// Maximum accepted size of the fixed checkpoint metadata envelope.
///
/// The current version byte and five varints occupy at most 46 bytes. The additional
/// headroom permits compatible representation changes without allowing a
/// hostile metadata file to drive an unbounded allocation during open.
pub(crate) const MAX_CHECKPOINT_METADATA_BYTES: usize = 256;

/// Read one byte beyond the admitted envelope so an oversized file is
/// distinguishable from an exact-boundary file without reading it in full.
pub(crate) const CHECKPOINT_METADATA_READ_LIMIT_BYTES: u64 =
    MAX_CHECKPOINT_METADATA_BYTES as u64 + 1;

impl CheckpointMetadata {
    /// Sole supported checkpoint schema.
    pub const FORMAT_VERSION: u8 = 5;

    fn corruption(path: &Path, reason: impl std::fmt::Display) -> Error {
        Error::Storage(StorageError::Corruption(format!(
            "invalid checkpoint metadata {}: {reason}",
            path.display()
        )))
    }

    /// Decodes one complete checkpoint metadata envelope.
    pub(crate) fn decode_exact(data: &[u8], path: &Path) -> Result<Self> {
        if data.len() > MAX_CHECKPOINT_METADATA_BYTES {
            return Err(Self::corruption(
                path,
                format_args!(
                    "{} bytes exceeds the {MAX_CHECKPOINT_METADATA_BYTES}-byte limit",
                    data.len()
                ),
            ));
        }
        if data.first() != Some(&Self::FORMAT_VERSION) {
            return Err(Self::corruption(path, "unsupported checkpoint generation"));
        }

        let (metadata, consumed): (Self, usize) = bincode::serde::decode_from_slice(
            data,
            bincode::config::standard().with_limit::<MAX_CHECKPOINT_METADATA_BYTES>(),
        )
        .map_err(|error| Self::corruption(path, format_args!("decode failed: {error}")))?;

        if consumed != data.len() {
            return Err(Self::corruption(
                path,
                format_args!(
                    "decoded {consumed} of {} bytes; trailing bytes are forbidden",
                    data.len()
                ),
            ));
        }
        if metadata.epoch == EpochId::PENDING {
            return Err(Self::corruption(
                path,
                "epoch is the reserved PENDING sentinel",
            ));
        }
        if !metadata.transaction_id.is_valid() {
            return Err(Self::corruption(
                path,
                "transaction ID is the reserved INVALID sentinel",
            ));
        }
        if metadata.format_version != Self::FORMAT_VERSION
            || metadata.retired_before > metadata.log_sequence
        {
            return Err(Self::corruption(
                path,
                "invalid checkpoint generation or group retirement boundary",
            ));
        }

        Ok(metadata)
    }

    /// Reads and decodes a checkpoint metadata file with a fixed allocation
    /// ceiling. A concurrently absent file is the same as no checkpoint.
    pub(crate) fn read_from_path(path: &Path) -> Result<Option<Self>> {
        let file = match checked_file(path, false, false) {
            Ok(file) => file,
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let mut reader = file.take(CHECKPOINT_METADATA_READ_LIMIT_BYTES);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        Self::decode_exact(&data, path).map(Some)
    }
}

/// Name of the checkpoint metadata file.
const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

/// Durability mode for the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Sync (fsync) after every commit for maximum durability.
    /// Slowest but safest.
    Sync,
    /// Batch sync - fsync after at most N records or N milliseconds from the
    /// first dirty frame, including when the writer becomes idle.
    Batch {
        /// Maximum time between syncs in milliseconds.
        max_delay_ms: u64,
        /// Maximum records between syncs.
        max_records: u64,
    },
    /// Adaptive sync - background thread adjusts timing based on flush duration.
    ///
    /// Unlike `Batch` which checks thresholds inline, `Adaptive` spawns a
    /// dedicated flusher thread that maintains consistent flush cadence
    /// regardless of disk speed. Use [`AdaptiveFlusher`](super::AdaptiveFlusher)
    /// to manage the background thread.
    ///
    /// The WAL itself only buffers writes; the flusher thread handles syncing.
    Adaptive {
        /// Target interval between flushes in milliseconds.
        /// The flusher adjusts wait times to maintain this cadence.
        target_interval_ms: u64,
    },
    /// No sync - rely on OS buffer flushing.
    /// Fastest but may lose recent data on crash.
    NoSync,
}

impl Default for DurabilityMode {
    fn default() -> Self {
        Self::Sync
    }
}

/// Configuration for the WAL manager.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Durability mode.
    pub durability: DurabilityMode,
    /// Maximum log file size before rotation (in bytes).
    pub max_log_size: u64,
    /// Whether to enable compression.
    pub compression: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            durability: DurabilityMode::default(),
            max_log_size: 64 * 1024 * 1024, // 64 MB
            compression: false,
        }
    }
}

/// State for a single log file.
struct LogFile {
    /// Sequence encoded in this file's name.
    ///
    /// This is authoritative for both recovery ordering and encryption
    /// nonces; it must not be inferred from a separately changing atomic.
    sequence: u64,
    /// File handle.
    writer: BufWriter<File>,
    /// Current size in bytes.
    size: u64,
}

/// Manages the Write-Ahead Log with rotation, checkpointing, and durability modes.
struct WalInner {
    operation: Mutex<OperationState>,
    /// Directory for WAL files.
    dir: PathBuf,
    /// Configuration.
    config: WalConfig,
    /// Active log file.
    active_log: Arc<Mutex<Option<LogFile>>>,
    /// Total number of records written across all log files.
    total_record_count: AtomicU64,
    /// Records since last sync (for batch mode).
    records_since_sync: Arc<AtomicU64>,
    /// Time of last sync (for batch mode).
    last_sync: Arc<Mutex<Instant>>,
    /// Current log sequence number.
    current_sequence: AtomicU64,
    /// Latest checkpoint epoch.
    checkpoint_epoch: Mutex<Option<EpochId>>,
    /// Segment floor from the same successfully published checkpoint metadata.
    checkpoint_sequence: Mutex<Option<u64>>,
    checkpoint_retired_before: Mutex<u64>,
    retention: Arc<Mutex<RetentionRegistry>>,
    /// Encryptor for WAL records (None = unencrypted).
    #[cfg(feature = "encryption")]
    encryptor: Option<grafeo_common::encryption::PageEncryptor>,
}

impl WalInner {
    /// Returns whether encryption is active.
    #[cfg(feature = "encryption")]
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.encryptor.is_some()
    }

    /// Writes a pre-serialized frame to the active WAL log.
    ///
    /// Frame format: `[length: u32 LE][data: bytes][crc32: u32 LE]`.
    /// Handles durability mode (sync/batch/adaptive/nosync) and log rotation.
    ///
    /// `force_sync` controls whether an fsync is performed in Sync durability
    /// mode. Callers typically set this to `true` for commit markers. This
    /// low-level path enforces the recoverable frame-size envelope; its opaque
    /// bytes have already been semantically validated by typed callers.
    pub(crate) fn write_frame(&self, data: &[u8], force_sync: bool) -> Result<()> {
        use grafeo_common::testing::crash::maybe_crash;

        self.validate_frame_payload(data)?;
        self.ensure_active_log()?;

        // A frame write, its durability decision, and any resulting rotation
        // form one serialized state transition. Two writers must never reserve
        // replacement sequences and install their files in reverse order:
        // recovery orders physical WAL history by that sequence.
        {
            let mut guard = self.active_log.lock();
            let log_file = guard
                .as_mut()
                .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;
            let projected_size = self.projected_frame_end(log_file, data)?;
            let needs_rotation = projected_size >= self.config.max_log_size;
            if needs_rotation {
                // Rotation is part of this write's successful state transition.
                // Prove its identity is representable before appending bytes or
                // changing counters; otherwise the failed call would have an
                // outcome-ambiguous durable prefix.
                log_file.sequence.checked_add(1).ok_or_else(|| {
                    Error::Storage(StorageError::Full).with_context(
                        "WAL frame reaches the rotation threshold, but log sequence identity space is exhausted",
                    )
                })?;
            }
            self.append_frame_locked(log_file, data)?;
            debug_assert_eq!(log_file.size, projected_size);

            // Decide whether we need to fsync based on durability mode.
            // Always flush the BufWriter so data reaches the OS page cache.
            let needs_sync = match &self.config.durability {
                DurabilityMode::Sync => {
                    if force_sync {
                        maybe_crash("wal_before_flush");
                    }
                    force_sync
                }
                DurabilityMode::Batch {
                    max_delay_ms,
                    max_records,
                } => {
                    let records = self.records_since_sync.load(Ordering::Relaxed);
                    let elapsed = self.last_sync.lock().elapsed();
                    records >= *max_records || elapsed >= Duration::from_millis(*max_delay_ms)
                }
                DurabilityMode::Adaptive { .. } | DurabilityMode::NoSync => false,
            };

            // Flush the BufWriter while holding the lock (pushes data to OS).
            validate_file(log_file.writer.get_ref())?;
            log_file.writer.flush()?;

            if needs_rotation {
                // Rotation syncs the old file before exposing the new one, so
                // it is also the durability barrier for this frame.
                let _ = self.rotate_locked(&mut guard)?;
                self.records_since_sync.store(0, Ordering::Relaxed);
                *self.last_sync.lock() = Instant::now();
            } else if needs_sync {
                log_file.writer.get_ref().sync_all()?;
                self.records_since_sync.store(0, Ordering::Relaxed);
                *self.last_sync.lock() = Instant::now();
            }
        }

        Ok(())
    }

    /// Computes the exact active-file size after encoding `data`, without
    /// allocating ciphertext or mutating the writer.
    fn projected_frame_end(&self, log_file: &LogFile, data: &[u8]) -> Result<u64> {
        let payload_len = self.encoded_payload_len(data)?;
        #[cfg(feature = "encryption")]
        let encrypted = self.encryptor.is_some();
        #[cfg(not(feature = "encryption"))]
        let encrypted = false;

        let payload_len = u32::try_from(payload_len).map_err(|_| {
            Error::Serialization("WAL frame exceeds the 32-bit wire-format limit".to_string())
        })?;
        let framing_len = if encrypted { 4 } else { 8 };
        log_file
            .size
            .checked_add(framing_len + u64::from(payload_len))
            .ok_or_else(|| {
                Error::Storage(StorageError::Full)
                    .with_context("projected WAL file size overflows u64")
            })
    }

    fn encoded_payload_len(&self, data: &[u8]) -> Result<usize> {
        self.encoded_payload_len_from_plaintext(data.len())
    }

    fn encoded_payload_len_from_plaintext(&self, plaintext_len: usize) -> Result<usize> {
        #[cfg(feature = "encryption")]
        if self.encryptor.is_some() {
            return plaintext_len
                .checked_add(grafeo_common::encryption::ENCRYPTION_OVERHEAD)
                .ok_or_else(|| {
                    Error::Storage(StorageError::Full)
                        .with_context("encrypted WAL frame length overflows usize")
                });
        }
        Ok(plaintext_len)
    }

    fn validate_frame_payload(&self, data: &[u8]) -> Result<()> {
        validate_wal_frame_payload_len(self.encoded_payload_len(data)?)?;
        super::frame_buffer::record_bytes(data).map_err(Error::InvalidValue)?;
        Ok(())
    }

    /// Appends one encoded frame while the caller holds `active_log`.
    ///
    /// Keeping framing and encryption here lets normal writes and checkpoint
    /// publication share the exact same on-disk format without recursively
    /// acquiring the active-file mutex.
    fn append_frame_locked(&self, log_file: &mut LogFile, data: &[u8]) -> Result<()> {
        use grafeo_common::testing::crash::maybe_crash;

        // If the complete frame fits strictly inside the remaining buffer, no
        // write_all below can reach the file. The flush/rotation owner validates
        // immediately before that I/O; failed ownership discards the buffer via
        // into_parts. Larger frames can flush while appending and need this
        // additional check before any of their bytes are buffered or written.
        #[cfg(feature = "encryption")]
        let framing_len = if self.is_encrypted() { 4 } else { 8 };
        #[cfg(not(feature = "encryption"))]
        let framing_len = 8;
        let frame_len = self
            .encoded_payload_len(data)?
            .checked_add(framing_len)
            .ok_or(Error::Storage(StorageError::Full))?;
        if frame_len
            >= log_file
                .writer
                .capacity()
                .saturating_sub(log_file.writer.buffer().len())
        {
            validate_file(log_file.writer.get_ref())?;
        }
        maybe_crash("wal_before_write");

        // Encrypt or write plaintext depending on encryption configuration.
        // Encrypted frame: [len:4][nonce(12) || ciphertext || tag(16)]
        // Plaintext frame:  [len:4][data][crc32:4]
        #[cfg(feature = "encryption")]
        if let Some(ref enc) = self.encryptor {
            let file_seq = log_file.sequence;
            // Use the file byte offset as the nonce counter, not the ephemeral
            // record count. The byte offset survives restarts (file is append-only)
            // and is unique per record within a file. Combined with the file sequence,
            // this guarantees nonce uniqueness even after crash + restart.
            //
            // The nonce high word is 4 bytes, so the file sequence must fit in u32.
            // With one rotation per ~64 MB of WAL, this allows ~256 exabytes of
            // total WAL writes before exhaustion, which is effectively unlimited.
            let seq_u32 = u32::try_from(file_seq).map_err(|_| {
                Error::Internal(
                    "WAL file sequence exceeds u32::MAX: encryption nonce space exhausted"
                        .to_string(),
                )
            })?;
            let byte_offset = log_file.size;
            let nonce = grafeo_common::encryption::build_nonce(seq_u32, byte_offset);
            let aad = b"grafeo-wal";
            let encrypted = enc
                .encrypt(data, &nonce, aad)
                .map_err(|e| Error::Internal(format!("WAL encryption failed: {e}")))?;
            let len = u32::try_from(encrypted.len()).map_err(|_| {
                Error::Serialization("WAL frame exceeds the 32-bit wire-format limit".to_string())
            })?;
            let record_size = 4u64 + u64::from(len);
            let new_size = log_file
                .size
                .checked_add(record_size)
                .ok_or_else(|| Error::Internal("WAL file size overflow".to_string()))?;
            log_file.writer.write_all(&len.to_le_bytes())?;
            log_file.writer.write_all(&encrypted)?;
            maybe_crash("wal_after_write");
            log_file.size = new_size;
            self.total_record_count.fetch_add(1, Ordering::Relaxed);
            self.records_since_sync.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        let len = u32::try_from(data.len()).map_err(|_| {
            Error::Serialization("WAL frame exceeds the 32-bit wire-format limit".to_string())
        })?;
        let record_size = 8u64 + u64::from(len);
        let new_size = log_file
            .size
            .checked_add(record_size)
            .ok_or_else(|| Error::Internal("WAL file size overflow".to_string()))?;
        log_file.writer.write_all(&len.to_le_bytes())?;
        log_file.writer.write_all(data)?;
        let checksum = crc32fast::hash(data);
        log_file.writer.write_all(&checksum.to_le_bytes())?;

        maybe_crash("wal_after_write");
        log_file.size = new_size;
        self.total_record_count.fetch_add(1, Ordering::Relaxed);
        self.records_since_sync.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Syncs the active log and returns the exact sequence that was synced.
    ///
    /// Observing the sequence under `active_log` prevents a concurrent
    /// rotation from making checkpoint metadata skip the file containing the
    /// checkpoint boundary.
    fn sync_active_log(&self) -> Result<u64> {
        let mut guard = self.active_log.lock();
        let sequence = if let Some(log_file) = guard.as_mut() {
            validate_file(log_file.writer.get_ref())?;
            log_file.writer.flush()?;
            log_file.writer.get_ref().sync_all()?;
            log_file.sequence
        } else {
            self.current_sequence.load(Ordering::Acquire)
        };
        self.records_since_sync.store(0, Ordering::Relaxed);
        *self.last_sync.lock() = Instant::now();
        Ok(sequence)
    }

    /// Publishes a serialized checkpoint as one ordered critical section.
    ///
    /// Marker append, old-segment sync, fresh-segment publication, metadata
    /// rename, in-memory epoch update, and truncation all occur while holding
    /// `active_log`. This prevents normal writers from crossing the boundary
    /// and concurrent checkpoints from installing metadata in reverse order.
    pub(crate) fn write_checkpoint_frame(
        &self,
        data: &[u8],
        transaction_id: TransactionId,
        epoch: EpochId,
        retired_before: u64,
    ) -> Result<()> {
        use grafeo_common::testing::crash::maybe_crash;

        Self::validate_checkpoint_epoch(epoch)?;
        Self::validate_checkpoint_transaction(transaction_id)?;
        self.validate_frame_payload(data)?;
        self.ensure_active_log()?;
        let mut guard = self.active_log.lock();

        // A newer durable snapshot subsumes an older concurrent retry. Treat
        // the stale request as an idempotent success instead of regressing the
        // epoch/sequence pair (or poisoning a typed WAL for a benign race).
        if self
            .checkpoint_epoch
            .lock()
            .is_some_and(|current| epoch < current)
        {
            return Ok(());
        }

        let log_file = guard
            .as_mut()
            .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;
        // A checkpoint necessarily rotates. Prove that its fresh-segment
        // sequence exists before appending the marker, otherwise exhaustion
        // would leave a marker without a publishable recovery boundary.
        log_file.sequence.checked_add(1).ok_or_else(|| {
            Error::Storage(StorageError::Full)
                .with_context("WAL log sequence identity space is exhausted")
        })?;
        self.append_frame_locked(log_file, data)?;

        // Checkpoint metadata stores a segment sequence rather than a byte
        // offset, so the marker remains in the old segment and metadata names
        // the exact fresh segment installed by this durability barrier.
        let log_sequence = self.rotate_locked(&mut guard)?;
        self.records_since_sync.store(0, Ordering::Relaxed);
        *self.last_sync.lock() = Instant::now();
        maybe_crash("wal_checkpoint_before_metadata");

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            // reason: millis since UNIX epoch fits in u64 for ~585 million years
            .map_or(0, |d| {
                // reason: value is bounded by format constraints
                #[allow(clippy::cast_possible_truncation)]
                let ms = d.as_millis() as u64;
                ms
            });

        let metadata = CheckpointMetadata {
            format_version: CheckpointMetadata::FORMAT_VERSION,
            epoch,
            log_sequence,
            timestamp_ms,
            transaction_id,
            retired_before,
        };

        self.write_checkpoint_metadata(&metadata)?;
        maybe_crash("wal_checkpoint_after_metadata");

        *self.checkpoint_epoch.lock() = Some(epoch);
        *self.checkpoint_sequence.lock() = Some(log_sequence);
        *self.checkpoint_retired_before.lock() = retired_before;
        self.truncate_old_logs()?;
        maybe_crash("wal_checkpoint_after_truncate");

        Ok(())
    }

    /// Writes checkpoint metadata to disk atomically.
    ///
    /// Uses a write-to-temp-then-rename pattern for atomicity.
    fn write_checkpoint_metadata(&self, metadata: &CheckpointMetadata) -> Result<()> {
        Self::validate_checkpoint_epoch(metadata.epoch)?;
        Self::validate_checkpoint_transaction(metadata.transaction_id)?;
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);
        let temp_path = self.dir.join(format!("{}.tmp", CHECKPOINT_METADATA_FILE));

        // Serialize metadata
        let data = bincode::serde::encode_to_vec(metadata, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        // Write to temp file
        match fs::symlink_metadata(&metadata_path) {
            Ok(_) => {
                let _ = checked_file(&metadata_path, true, false)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let mut file = checked_file(&temp_path, true, true)?;
        validate_file(&file)?;
        file.set_len(0)?;
        file.write_all(&data)?;
        file.sync_all()?;
        drop(file);

        // Atomic rename
        fs::rename(&temp_path, &metadata_path)?;
        // Persist the directory entry before any older WAL segment may be
        // reclaimed. Syncing only the metadata file does not make the rename
        // power-loss durable on filesystems with a separate directory journal.
        self.sync_directory()?;

        Ok(())
    }

    /// Reads checkpoint metadata from disk.
    ///
    /// Returns `None` if no checkpoint metadata exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata file cannot be read or is oversized,
    /// malformed, has trailing bytes, or contains a reserved identity sentinel.
    pub fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);
        CheckpointMetadata::read_from_path(&metadata_path)
    }

    fn validate_checkpoint_epoch(epoch: EpochId) -> Result<()> {
        if epoch == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "checkpoint epoch is the reserved PENDING sentinel".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_checkpoint_transaction(transaction_id: TransactionId) -> Result<()> {
        if !transaction_id.is_valid() {
            return Err(Error::InvalidValue(
                "checkpoint transaction ID is the reserved INVALID sentinel".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_entry_for_write<R: WalEntry>(record: &R) -> Result<()> {
        record
            .validate_recovery()
            .map_err(|reason| Error::InvalidValue(format!("invalid WAL record: {reason}")))
    }

    /// Rotates to a new log file.
    ///
    /// # Errors
    ///
    /// Returns an error if rotation fails.
    pub fn rotate(&self) -> Result<()> {
        self.rotate_to_new_segment().map(|_| ())
    }

    /// Rotates and returns the exact fresh segment sequence.
    fn rotate_to_new_segment(&self) -> Result<u64> {
        self.ensure_active_log()?;
        let mut guard = self.active_log.lock();
        let sequence = self.rotate_locked(&mut guard)?;
        self.records_since_sync.store(0, Ordering::Relaxed);
        *self.last_sync.lock() = Instant::now();
        Ok(sequence)
    }

    /// Replaces the active file while `active_log` is held exclusively.
    fn rotate_locked(&self, guard: &mut Option<LogFile>) -> Result<u64> {
        let old_log = guard
            .as_mut()
            .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;
        validate_file(old_log.writer.get_ref())?;
        old_log.writer.flush()?;
        old_log.writer.get_ref().sync_all()?;

        let new_sequence = old_log.sequence.checked_add(1).ok_or_else(|| {
            Error::Storage(StorageError::Full)
                .with_context("WAL log sequence identity space is exhausted")
        })?;
        let new_path = self.log_path(new_sequence);

        let mut file = checked_file(&new_path, true, true)?;
        file.seek(SeekFrom::End(0))?;
        // Make the new directory entry durable before it can become the
        // active destination of a force-synced commit.
        file.sync_all()?;
        self.sync_directory()?;

        let new_log = LogFile {
            sequence: new_sequence,
            writer: BufWriter::new(file),
            size: 0,
        };
        *guard = Some(new_log);
        self.current_sequence.store(new_sequence, Ordering::Release);
        Ok(new_sequence)
    }

    /// Flushes the WAL buffer to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<()> {
        let mut guard = self.active_log.lock();
        if let Some(log_file) = guard.as_mut() {
            validate_file(log_file.writer.get_ref())?;
            log_file.writer.flush()?;
        }
        Ok(())
    }

    /// Syncs the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub fn sync(&self) -> Result<()> {
        self.sync_active_log()?;
        Ok(())
    }

    /// Returns the total number of records written.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.total_record_count.load(Ordering::Relaxed)
    }

    /// Returns the WAL directory path.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns the current WAL log sequence number.
    ///
    /// Each log file has a sequence number embedded in its name
    /// (`wal_XXXXXXXX.log`). This returns the sequence of the active log file.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.current_sequence.load(Ordering::Relaxed)
    }

    /// Returns the current durability mode.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.config.durability
    }

    /// Returns all WAL log file paths in sequence order.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL directory cannot be read.
    pub fn log_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "log") {
                files.push(path);
            }
        }

        // Sort by sequence number
        files.sort_by(|a, b| {
            let seq_a = Self::sequence_from_path(a).unwrap_or(0);
            let seq_b = Self::sequence_from_path(b).unwrap_or(0);
            seq_a.cmp(&seq_b)
        });

        Ok(files)
    }

    /// Returns the latest checkpoint epoch, if any.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> Option<EpochId> {
        *self.checkpoint_epoch.lock()
    }

    /// Returns the total size of all WAL files in bytes.
    pub fn size_bytes(&self) -> Result<usize> {
        let mut total = 0usize;
        for path in self
            .log_files()?
            .into_iter()
            .chain(std::iter::once(self.dir.join(CHECKPOINT_METADATA_FILE)))
        {
            let length = match checked_file(&path, false, false) {
                Ok(file) => file.metadata()?.len(),
                Err(Error::Io(error))
                    if error.kind() == std::io::ErrorKind::NotFound
                        && path
                            .file_name()
                            .is_some_and(|name| name == CHECKPOINT_METADATA_FILE) =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            };
            let length = usize::try_from(length).map_err(|_| Error::Storage(StorageError::Full))?;
            total = total
                .checked_add(length)
                .ok_or(Error::Storage(StorageError::Full))?;
        }
        Ok(total)
    }

    /// Returns the timestamp of the last checkpoint (Unix epoch seconds), if any.
    pub fn last_checkpoint_timestamp(&self) -> Result<Option<u64>> {
        Ok(self
            .read_checkpoint_metadata()?
            .map(|metadata| metadata.timestamp_ms / 1000))
    }

    // === Private methods ===

    /// Persists WAL directory-entry changes on platforms that support
    /// syncing directory handles.
    fn sync_directory(&self) -> Result<()> {
        #[cfg(unix)]
        {
            File::open(&self.dir)?.sync_all()?;
        }
        Ok(())
    }

    fn ensure_active_log(&self) -> Result<()> {
        let mut guard = self.active_log.lock();
        if guard.is_none() {
            let sequence = self.current_sequence.load(Ordering::Relaxed);
            let path = self.log_path(sequence);

            let mut file = checked_file(&path, true, true)?;
            file.seek(SeekFrom::End(0))?;

            let size = file.metadata()?.len();
            file.sync_all()?;
            self.sync_directory()?;

            *guard = Some(LogFile {
                sequence,
                writer: BufWriter::new(file),
                size,
            });
        }
        Ok(())
    }

    fn log_path(&self, sequence: u64) -> PathBuf {
        self.dir.join(format!("wal_{:08}.log", sequence))
    }

    fn sequence_from_path(path: &Path) -> Option<u64> {
        if path.extension().is_none_or(|extension| extension != "log") {
            return None;
        }
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("wal_"))
            .and_then(|s| s.parse().ok())
    }

    fn truncate_old_logs(&self) -> Result<()> {
        let Some(checkpoint_sequence) = *self.checkpoint_sequence.lock() else {
            return Ok(());
        };
        // Registration shares operation admission with retirement. Hold the
        // registry while selecting/deleting so a concurrent drop only delays
        // reclamation, and never permits retirement past a live floor.
        let retention = self.retention.lock();
        let retire_before = retention
            .starts
            .first_key_value()
            .map_or(checkpoint_sequence, |(&first, _)| {
                checkpoint_sequence.min(first)
            })
            .min(*self.checkpoint_retired_before.lock());

        // Keep logs that might still be needed
        // For now, keep the two most recent logs after checkpoint
        let files = self.log_files()?;
        let current_seq = self.current_sequence.load(Ordering::Relaxed);

        let mut removed_any = false;
        for file in files {
            if let Some(seq) = Self::sequence_from_path(&file) {
                // Keep the last 2 log files before current, and only delete
                // files covered by the durable checkpoint.
                if seq.checked_add(2).is_some_and(|keep| keep < current_seq) && seq < retire_before
                {
                    let handle = checked_file(&file, true, false)?;
                    validate_file(&handle)?;
                    fs::remove_file(&file)?;
                    removed_any = true;
                }
            }
        }

        if removed_any {
            self.sync_directory()?;
        }

        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Open,
    Closing,
    Failed,
    Closed,
}

struct OperationState {
    phase: Phase,
    pending_cause: Option<Error>,
    lease: Option<WalLease>,
    groups: super::group::Groups,
}

struct Operation<'a> {
    state: parking_lot::MutexGuard<'a, OperationState>,
    physical: bool,
}

fn terminal_error() -> Error {
    std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        "WAL is terminal; drop its owner and recover before further I/O",
    )
    .into()
}

impl Operation<'_> {
    fn rotate(&mut self, inner: &WalInner) -> Result<()> {
        inner.current_sequence().checked_add(1).ok_or_else(|| {
            Error::Storage(StorageError::Full)
                .with_context("WAL log sequence identity space is exhausted")
        })?;
        self.state.groups.reserve_segment()?;
        self.physical = true;
        let result = inner.rotate();
        if result.is_ok() {
            self.state
                .groups
                .record_segment_reserved(inner.current_sequence());
        }
        self.complete(result)
    }

    fn complete<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() && self.physical {
            self.state.phase = Phase::Failed;
        }
        self.physical = false;
        result
    }
}

impl Drop for Operation<'_> {
    fn drop(&mut self) {
        if self.physical {
            self.state.phase = Phase::Failed;
        }
    }
}

#[derive(Default)]
struct Supervisor {
    stop: Option<std::sync::mpsc::Sender<()>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

/// A synchronous WAL with one terminal physical-operation authority.
pub struct WalManager {
    inner: Arc<WalInner>,
    supervisor: Mutex<Supervisor>,
}

#[derive(Default)]
struct RetentionRegistry {
    starts: BTreeMap<u64, u64>,
    count: u64,
}

/// A move-only claim retaining a WAL generation's segments from one sequence.
///
/// Dropping the claim makes its interval eligible for the next checkpoint's
/// retirement. It does not acquire a pathname owner or keep a closed WAL open.
pub struct WalRetentionLease {
    registry: Arc<Mutex<RetentionRegistry>>,
    first_sequence: u64,
}

impl Drop for WalRetentionLease {
    fn drop(&mut self) {
        // Never acquire operation admission: the caller may drop this lease
        // while holding a capture of the same WAL.
        let mut registry = self.registry.lock();
        if let Some(count) = registry.starts.get_mut(&self.first_sequence) {
            *count -= 1;
            if *count == 0 {
                registry.starts.remove(&self.first_sequence);
            }
            registry.count -= 1;
        }
    }
}

/// Opaque identity of a checked segment opened by one WAL capture.
pub struct WalSegmentDescriptor {
    capture: Arc<()>,
    index: usize,
    sequence: u64,
    size: u64,
}

impl WalSegmentDescriptor {
    /// Physical sequence copied when the segment was opened.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    /// Size copied from the checked opened handle.
    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.size
    }
}

/// One admitted capture spanning checked reads, rotation and cursor publication.
pub struct WalCapture<'a> {
    // Opened data handles retire before operation admission.
    files: Vec<File>,
    identity: Arc<()>,
    inner: &'a WalInner,
    operation: Operation<'a>,
}

impl WalCapture<'_> {
    fn ensure_open(&self) -> Result<()> {
        if self.operation.state.phase != Phase::Open {
            return Err(terminal_error());
        }
        Ok(())
    }
    fn backup_generation_name(chain_id: &[u8; 32], generation: u64) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut chain = String::with_capacity(64);
        for byte in chain_id {
            chain.push(char::from(HEX[usize::from(byte >> 4)]));
            chain.push(char::from(HEX[usize::from(byte & 15)]));
        }
        format!("backup_cursor_{chain}_{generation:020}.meta")
    }

    /// Reports whether this owned WAL contains backup cursor state to preserve.
    ///
    /// Includes immutable generations, interrupted installing artifacts and the
    /// advisory cursor. Scans names without reading or allocating cursor payloads.
    ///
    /// # Errors
    /// Rejects malformed reserved generation names, unsafe files, oversized
    /// cursor artifacts and terminal captures.
    pub fn has_backup_generations(&self) -> Result<bool> {
        self.ensure_open()?;
        let mut present = false;
        for entry in fs::read_dir(&self.inner.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let fixed = matches!(name, "backup_cursor.meta" | "backup_cursor.meta.tmp");
            if !fixed && !name.starts_with("backup_cursor_") {
                continue;
            }
            if !fixed {
                let base = name.strip_suffix(".installing").unwrap_or(name);
                let fields = base
                    .strip_prefix("backup_cursor_")
                    .and_then(|tail| tail.strip_suffix(".meta"))
                    .and_then(|tail| tail.split_once('_'));
                let valid = fields.is_some_and(|(chain, generation)| {
                    chain.len() == 64
                        && chain
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                        && generation.len() == 20
                        && generation.bytes().all(|byte| byte.is_ascii_digit())
                        && generation.parse::<u64>().is_ok()
                });
                if !valid {
                    return Err(Error::Storage(StorageError::Corruption(
                        "malformed backup cursor generation artifact name".into(),
                    )));
                }
            }
            let file = checked_file(&entry.path(), false, false)?;
            if file.metadata()?.len() > 64 * 1024 * 1024 + 17 {
                return Err(Error::Storage(StorageError::Corruption(
                    "backup cursor exceeds its encoded size limit".into(),
                )));
            }
            present = true;
        }
        Ok(present)
    }

    fn read_cursor_artifact(&self, path: &Path) -> Result<Option<Vec<u8>>> {
        const MAX_CURSOR_BYTES: u64 = 64 * 1024 * 1024 + 17;
        self.ensure_open()?;
        let file = match checked_file(path, false, false) {
            Ok(file) => file,
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if file.metadata()?.len() > MAX_CURSOR_BYTES {
            return Err(Error::Storage(StorageError::Corruption(
                "backup cursor exceeds its encoded size limit".into(),
            )));
        }
        let mut bytes = Vec::new();
        file.take(MAX_CURSOR_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_CURSOR_BYTES {
            return Err(Error::Storage(StorageError::Corruption(
                "backup cursor exceeds its encoded size limit".into(),
            )));
        }
        Ok(Some(bytes))
    }

    /// Reads the fixed advisory backup cursor; absence alone yields None.
    ///
    /// # Errors
    /// Returns checked-open, size-limit, and read errors.
    pub fn read_backup_cursor_bytes(&mut self) -> Result<Option<Vec<u8>>> {
        self.read_cursor_artifact(&self.inner.dir.join("backup_cursor.meta"))
    }

    /// Reads one immutable chain-qualified backup generation cursor.
    ///
    /// # Errors
    /// Rejects unsafe artifacts and oversized encodings; absence alone yields None.
    pub fn read_backup_generation_bytes(
        &mut self,
        chain_id: &[u8; 32],
        generation: u64,
    ) -> Result<Option<Vec<u8>>> {
        let name = Self::backup_generation_name(chain_id, generation);
        self.read_cursor_artifact(&self.inner.dir.join(name))
    }

    /// Durably publishes a new immutable chain-qualified backup cursor.
    ///
    /// Existing generations and interrupted installing artifacts are never
    /// overwritten. The existing WAL authority excludes cooperating publishers.
    ///
    /// # Errors
    /// Rejects oversized data and incomplete/conflicting generations. Physical
    /// publication failures poison this WAL owner before admission is released.
    pub fn write_backup_generation_bytes(
        &mut self,
        chain_id: &[u8; 32],
        generation: u64,
        bytes: &[u8],
    ) -> Result<()> {
        use grafeo_common::testing::crash::maybe_crash;
        use grafeo_common::testing::wal_failure::check_backup_publication_failure;
        self.ensure_open()?;
        if bytes.len() > 64 * 1024 * 1024 + 17 {
            return Err(Error::InvalidValue(
                "backup cursor exceeds its encoded size limit".into(),
            ));
        }
        let name = Self::backup_generation_name(chain_id, generation);
        let path = self.inner.dir.join(&name);
        let temporary = self.inner.dir.join(format!("{name}.installing"));
        for artifact in [&path, &temporary] {
            match fs::symlink_metadata(artifact) {
                Ok(_) => {
                    return Err(Error::Storage(StorageError::Corruption(format!(
                        "incomplete or conflicting backup generation: {} already exists",
                        artifact.display(),
                    ))));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        self.operation.physical = true;
        let result =
            (|| {
                let mut file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                validate_file(&file)?;
                check_backup_publication_failure("backup:cursor_write")?;
                file.write_all(bytes)?;
                file.sync_all()?;
                drop(file);
                maybe_crash("backup:cursor_installing_sync");
                // Namespace replacement by non-cooperating actors is outside the
                // owner contract; still refuse a final artifact that appeared.
                match fs::symlink_metadata(&path) {
                    Ok(_) => return Err(Error::Storage(StorageError::Corruption(
                        "incomplete or conflicting backup generation appeared before publication"
                            .into(),
                    ))),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
                check_backup_publication_failure("backup:cursor_rename_failure")?;
                fs::rename(&temporary, &path)?;
                maybe_crash("backup:cursor_rename");
                check_backup_publication_failure("backup:cursor_parent_sync")?;
                crate::ownership::sync_parent(&path)?;
                maybe_crash("backup:cursor_directory_sync");
                Ok(())
            })();
        self.operation.complete(result)
    }

    /// Returns whether the captured WAL uses encrypted physical frames.
    #[cfg(feature = "encryption")]
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.inner.encryptor.is_some()
    }

    /// Validates and counts physical frames using this WAL's encryption mode.
    ///
    /// # Errors
    /// Rejects closed captures and malformed, incomplete or corrupt frames.
    pub fn count_frames(&self, bytes: &[u8]) -> Result<u64> {
        self.ensure_open()?;
        super::recovery::count_frames(
            bytes,
            #[cfg(feature = "encryption")]
            self.inner.encryptor.as_ref(),
        )
    }

    /// Atomically publishes the fixed cursor through checked artifact handles.
    ///
    /// # Errors
    /// A physical publication failure poisons W before admission is released.
    pub fn write_backup_cursor_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.ensure_open()?;
        let path = self.inner.dir.join("backup_cursor.meta");
        let temporary = self.inner.dir.join("backup_cursor.meta.tmp");
        match checked_file(&path, false, false) {
            Ok(file) => drop(file),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        self.operation.physical = true;
        let result = (|| {
            let mut file = checked_file(&temporary, true, true)?;
            file.set_len(0)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &path)?;
            crate::ownership::sync_parent(&path)
        })();
        self.operation.complete(result)
    }

    /// Returns cached sequence while capture excludes every physical operation.
    #[must_use]
    pub fn current_log_sequence(&self) -> u64 {
        self.inner.current_sequence()
    }

    /// Rotates without reacquiring the raw operation mutex.
    ///
    /// # Errors
    /// Returns deterministic capacity or poisoning physical errors.
    pub fn rotate(&mut self) -> Result<()> {
        self.ensure_open()?;
        self.operation.rotate(self.inner)
    }

    /// Flushes buffered bytes and retains checked segment handles for this capture.
    ///
    /// # Errors
    /// Returns flush, enumeration, name or checked-open errors.
    pub fn segments(&mut self) -> Result<Vec<WalSegmentDescriptor>> {
        self.ensure_open()?;
        self.operation.physical = true;
        self.operation.complete(self.inner.flush())?;
        let mut descriptors = Vec::new();
        for path in self.inner.log_files()? {
            let sequence = WalInner::sequence_from_path(&path).ok_or_else(|| {
                Error::InvalidValue("WAL segment has an invalid sequence name".into())
            })?;
            let file = checked_file(&path, false, false)?;
            let size = file.metadata()?.len();
            descriptors.push(WalSegmentDescriptor {
                capture: Arc::clone(&self.identity),
                index: self.files.len(),
                sequence,
                size,
            });
            self.files.push(file);
        }
        Ok(descriptors)
    }

    /// Reads only a descriptor minted by this exact capture, using its open FD.
    ///
    /// # Errors
    /// Rejects a foreign descriptor and returns actual read errors.
    pub fn read_segment(&mut self, segment: &WalSegmentDescriptor) -> Result<Vec<u8>> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.identity, &segment.capture) {
            return Err(Error::InvalidValue(
                "segment belongs to another WAL capture".into(),
            ));
        }
        let file = self
            .files
            .get_mut(segment.index)
            .ok_or_else(|| Error::InvalidValue("unknown captured WAL segment".into()))?;
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }
}

pub(crate) struct WalFrameIntent {
    force_sync: bool,
    #[cfg(feature = "testing-crash-injection")]
    before_append: Option<Error>,
    #[cfg(feature = "testing-crash-injection")]
    after_append: Option<Error>,
}

impl WalFrameIntent {
    pub(crate) fn raw(force_sync: bool) -> Self {
        Self {
            force_sync,
            #[cfg(feature = "testing-crash-injection")]
            before_append: None,
            #[cfg(feature = "testing-crash-injection")]
            after_append: None,
        }
    }

    pub(crate) fn capture<R: WalEntry>(record: &R) -> Self {
        #[cfg(feature = "testing-crash-injection")]
        let (before_append, after_append) = {
            use grafeo_common::testing::wal_failure as hooks;
            let before = (|| {
                if record.is_commit() {
                    hooks::maybe_fail_commit_log()?;
                } else if record.is_abort() {
                    hooks::maybe_fail_abort_log()?;
                }
                if record.is_data_mutation() {
                    hooks::maybe_fail_mutation_log()?;
                }
                if record.is_catalog_batch() {
                    hooks::maybe_fail_catalog_batch_log()?;
                }
                Ok::<(), hooks::InjectedWalLogFailure>(())
            })()
            .err()
            .map(|error| Error::Internal(error.to_string()));
            let commit = if record.is_commit() {
                hooks::take_commit_ack_failure()
            } else {
                None
            };
            let catalog = if record.is_catalog_batch() {
                hooks::take_catalog_batch_ack_failure()
            } else {
                None
            };
            (
                before,
                commit
                    .or(catalog)
                    .map(|error| Error::Internal(error.to_string())),
            )
        };
        Self {
            force_sync: record.requires_sync(),
            #[cfg(feature = "testing-crash-injection")]
            before_append,
            #[cfg(feature = "testing-crash-injection")]
            after_append,
        }
    }
}

impl WalManager {
    /// Opens a WAL with default configuration.
    ///
    /// # Errors
    /// Returns admission or physical initialization errors.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(dir, WalConfig::default())
    }

    /// Opens a WAL with the supplied configuration.
    ///
    /// # Errors
    /// Returns admission or physical initialization errors.
    pub fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        let lease = WalLease::acquire(dir.as_ref())?;
        Self::from_lease(
            lease,
            config,
            None,
            #[cfg(feature = "encryption")]
            None,
        )
    }

    /// Opens a WAL with its encryptor fixed before physical initialization.
    ///
    /// # Errors
    /// Returns admission or physical initialization errors.
    #[cfg(feature = "encryption")]
    pub fn with_config_and_encryptor(
        dir: impl AsRef<Path>,
        config: WalConfig,
        encryptor: grafeo_common::encryption::PageEncryptor,
    ) -> Result<Self> {
        let lease = WalLease::acquire(dir.as_ref())?;
        Self::from_lease(lease, config, None, Some(encryptor))
    }

    pub(super) fn from_lease(
        lease: WalLease,
        config: WalConfig,
        observed: Option<(u64, Option<CheckpointMetadata>, super::group::Groups)>,
        #[cfg(feature = "encryption")] encryptor: Option<grafeo_common::encryption::PageEncryptor>,
    ) -> Result<Self> {
        let dir = lease.path().to_path_buf();
        fs::create_dir_all(&dir)?;
        let Some((max_sequence, checkpoint, mut groups)) = observed else {
            return super::WalRecovery::validate_append_lease(
                lease,
                #[cfg(feature = "encryption")]
                encryptor,
            )?
            .into_wal(config);
        };
        groups.initialize_empty_directory(max_sequence)?;
        let manager = Self {
            inner: Arc::new(WalInner {
                operation: Mutex::new(OperationState {
                    phase: Phase::Open,
                    pending_cause: None,
                    lease: Some(lease),
                    groups: groups.resume_writer(),
                }),
                dir,
                config,
                active_log: Arc::new(Mutex::new(None)),
                total_record_count: AtomicU64::new(0),
                records_since_sync: Arc::new(AtomicU64::new(0)),
                last_sync: Arc::new(Mutex::new(Instant::now())),
                current_sequence: AtomicU64::new(max_sequence),
                checkpoint_epoch: Mutex::new(checkpoint.as_ref().map(|metadata| metadata.epoch)),
                checkpoint_retired_before: Mutex::new(
                    checkpoint
                        .as_ref()
                        .map_or(0, |metadata| metadata.retired_before),
                ),
                checkpoint_sequence: Mutex::new(checkpoint.map(|metadata| metadata.log_sequence)),
                retention: Arc::new(Mutex::new(RetentionRegistry::default())),
                #[cfg(feature = "encryption")]
                encryptor,
            }),
            supervisor: Mutex::new(Supervisor::default()),
        };
        manager.inner.ensure_active_log()?;
        manager.start_worker()?;
        Ok(manager)
    }

    fn admit(&self) -> Result<Operation<'_>> {
        #[cfg(test)]
        tests::admission_test_point();
        let mut state = self.inner.operation.lock();
        if state.phase != Phase::Open {
            return Err(state.pending_cause.take().unwrap_or_else(terminal_error));
        }
        Ok(Operation {
            state,
            physical: false,
        })
    }

    /// Captures one raw operation authority through backup reads and publication.
    ///
    /// # Errors
    /// Rejects closed, closing or failed WAL owners.
    pub fn capture(&self) -> Result<WalCapture<'_>> {
        Ok(WalCapture {
            files: Vec::new(),
            identity: Arc::new(()),
            inner: &self.inner,
            operation: self.admit()?,
        })
    }

    /// Retains segments from `sequence` until the returned lease is dropped.
    ///
    /// Registration is serialized with checkpoint retirement. Already retired
    /// segments are not recreated; callers must validate their captured interval.
    ///
    /// # Errors
    /// Rejects terminal owners or more than 65,536 simultaneous live leases.
    pub fn retain_from(&self, sequence: u64) -> Result<WalRetentionLease> {
        const MAX_RETENTION_LEASES: u64 = 65_536;
        let _operation = self.admit()?;
        let mut registry = self.inner.retention.lock();
        let next = registry
            .count
            .checked_add(1)
            .filter(|count| *count <= MAX_RETENTION_LEASES)
            .ok_or_else(|| Error::InvalidValue("too many live WAL retention leases".into()))?;
        let count = registry.starts.entry(sequence).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| Error::InvalidValue("WAL retention sequence count overflow".into()))?;
        registry.count = next;
        Ok(WalRetentionLease {
            registry: Arc::clone(&self.inner.retention),
            first_sequence: sequence,
        })
    }

    /// Captures the already owned WAL generation protected by `lease`.
    ///
    /// # Errors
    /// Rejects foreign or reopened generations and terminal WAL owners.
    pub fn capture_with_lease(&self, lease: &WalRetentionLease) -> Result<WalCapture<'_>> {
        let capture = self.capture()?;
        if !Arc::ptr_eq(&self.inner.retention, &lease.registry) {
            return Err(Error::InvalidValue(
                "retention lease belongs to another WAL generation".into(),
            ));
        }
        Ok(capture)
    }

    fn start_worker(&self) -> Result<()> {
        let DurabilityMode::Batch { max_delay_ms, .. } = self.inner.config.durability else {
            return Ok(());
        };
        let (stop, receiver) = std::sync::mpsc::channel();
        let inner = Arc::downgrade(&self.inner);
        let mut supervisor = self.supervisor.lock();
        let worker = std::thread::Builder::new()
            .name("grafeo-wal-batch-sync".into())
            .spawn(move || {
                loop {
                    match receiver.recv_timeout(Duration::from_millis(max_delay_ms.max(1))) {
                        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    let Some(inner) = inner.upgrade() else {
                        break;
                    };
                    let state = inner.operation.lock();
                    if state.phase != Phase::Open {
                        break;
                    }
                    let mut operation = Operation {
                        state,
                        physical: false,
                    };
                    if inner.records_since_sync.load(Ordering::Acquire) > 0 {
                        operation.physical = true;
                        let result = inner.sync_active_log();
                        if let Err(error) = operation.complete(result) {
                            operation.state.pending_cause = Some(error);
                            break;
                        }
                    }
                }
            })?;
        supervisor.stop = Some(stop);
        supervisor.worker = Some(worker);
        Ok(())
    }

    /// Logs one validated record.
    ///
    /// # Errors
    /// Returns validation, terminal-state or physical errors.
    pub fn log(&self, record: &WalRecord) -> Result<()> {
        WalInner::validate_entry_for_write(record)?;
        let intent = WalFrameIntent::capture(record);
        let data = encode_frame(record)?;
        self.write_frame(&data, intent)
    }

    pub(crate) fn write_frame(&self, data: &[u8], intent: WalFrameIntent) -> Result<()> {
        let mut operation = self.admit()?;
        let previous_sequence = self.inner.current_sequence();
        let prepared = operation.state.groups.prepare(data, true)?;
        let data = if let Some(seal) = prepared.seal {
            let mut sealed = super::frame_buffer::copy_frame(data)?;
            let seal_at = sealed.len() - super::group::SEAL_LEN;
            sealed[seal_at..].copy_from_slice(&seal);
            std::borrow::Cow::Owned(sealed)
        } else {
            std::borrow::Cow::Borrowed(data)
        };
        let data = data.as_ref();
        self.inner.validate_frame_payload(data)?;
        {
            let active = self.inner.active_log.lock();
            let file = active.as_ref().ok_or_else(terminal_error)?;
            let end = self.inner.projected_frame_end(file, data)?;
            if end >= self.inner.config.max_log_size {
                operation.state.groups.reserve_segment()?;
                file.sequence.checked_add(1).ok_or_else(|| Error::Storage(StorageError::Full).with_context("WAL frame reaches the rotation threshold, but log sequence identity space is exhausted"))?;
            }
            #[cfg(feature = "encryption")]
            if self.inner.encryptor.is_some() {
                u32::try_from(file.sequence).map_err(|_| Error::Storage(StorageError::Full))?;
            }
        }
        operation.physical = true;
        let result = (|| {
            #[cfg(feature = "testing-crash-injection")]
            if let Some(error) = intent.before_append {
                return Err(error);
            }
            self.inner.write_frame(data, intent.force_sync)?;
            #[cfg(feature = "testing-crash-injection")]
            if let Some(error) = intent.after_append {
                return Err(error);
            }
            Ok(())
        })();
        if result.is_ok() {
            operation.state.groups.apply(prepared, data);
            let sequence = self.inner.current_sequence();
            if sequence != previous_sequence {
                operation.state.groups.record_segment_reserved(sequence);
            }
        }
        operation.complete(result)
    }

    /// Publishes a checkpoint in the owned WAL.
    ///
    /// # Errors
    /// Returns validation, terminal-state or physical errors.
    pub fn checkpoint(&self, transaction: TransactionId, epoch: EpochId) -> Result<()> {
        let record = WalRecord::Checkpoint {
            transaction_id: transaction,
        };
        let data = encode_frame(&record)?;
        self.write_checkpoint_frame(&data, transaction, epoch)
    }

    pub(crate) fn write_checkpoint_frame(
        &self,
        data: &[u8],
        transaction: TransactionId,
        epoch: EpochId,
    ) -> Result<()> {
        let mut operation = self.admit()?;
        WalInner::validate_checkpoint_epoch(epoch)?;
        WalInner::validate_checkpoint_transaction(transaction)?;
        self.inner.validate_frame_payload(data)?;
        if self
            .inner
            .checkpoint_epoch()
            .is_some_and(|current| epoch < current)
        {
            return Ok(());
        }
        if !super::group::envelope(data)
            .map_err(Error::InvalidValue)?
            .coordinate
            .is_checkpoint()
        {
            return Err(Error::InvalidValue(
                "WAL checkpoint payload is not a checkpoint".into(),
            ));
        }
        let prepared = operation.state.groups.prepare(data, true)?;
        let next_sequence = self
            .inner
            .current_sequence()
            .checked_add(1)
            .ok_or_else(|| {
                Error::Storage(StorageError::Full)
                    .with_context("WAL log sequence identity space is exhausted")
            })?;
        operation.state.groups.reserve_segment()?;
        let requested_floor = self
            .inner
            .retention
            .lock()
            .starts
            .first_key_value()
            .map_or(next_sequence, |(&first, _)| next_sequence.min(first));
        let retired_before = if requested_floor == next_sequence {
            next_sequence
        } else {
            operation.state.groups.retirement_floor(requested_floor)
        };
        #[cfg(feature = "encryption")]
        if self.inner.encryptor.is_some() {
            u32::try_from(self.inner.current_sequence())
                .map_err(|_| Error::Storage(StorageError::Full))?;
        }
        operation.physical = true;
        let result = self
            .inner
            .write_checkpoint_frame(data, transaction, epoch, retired_before);
        if result.is_ok() {
            operation.state.groups.apply(prepared, data);
            operation
                .state
                .groups
                .record_segment_reserved(next_sequence);
            operation.state.groups.retire_boundaries(retired_before);
        }
        operation.complete(result)
    }

    /// Flushes buffered bytes.
    ///
    /// # Errors
    /// Returns terminal-state or physical errors.
    pub fn flush(&self) -> Result<()> {
        let mut operation = self.admit()?;
        operation.physical = true;
        operation.complete(self.inner.flush())
    }

    /// Synchronizes accepted bytes.
    ///
    /// # Errors
    /// Returns terminal-state or physical errors.
    pub fn sync(&self) -> Result<()> {
        let mut operation = self.admit()?;
        operation.physical = true;
        operation.complete(self.inner.sync())
    }

    /// Rotates the active segment.
    ///
    /// # Errors
    /// Returns capacity, terminal-state or physical errors.
    pub fn rotate(&self) -> Result<()> {
        self.admit()?.rotate(&self.inner)
    }

    /// Lists physical segments while holding operation admission.
    ///
    /// # Errors
    /// Returns terminal-state or filesystem errors.
    pub fn log_files(&self) -> Result<Vec<PathBuf>> {
        let _operation = self.admit()?;
        self.inner.log_files()
    }

    /// Reads checkpoint metadata under operation admission.
    ///
    /// # Errors
    /// Returns terminal-state, filesystem or corruption errors.
    pub fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        let _operation = self.admit()?;
        self.inner.read_checkpoint_metadata()
    }

    /// Computes physical WAL bytes with checked arithmetic.
    ///
    /// # Errors
    /// Returns terminal-state, filesystem or capacity errors.
    pub fn size_bytes(&self) -> Result<usize> {
        let _operation = self.admit()?;
        self.inner.size_bytes()
    }

    /// Reads the durable checkpoint timestamp.
    ///
    /// # Errors
    /// Returns terminal-state, filesystem or corruption errors.
    pub fn last_checkpoint_timestamp(&self) -> Result<Option<u64>> {
        let _operation = self.admit()?;
        self.inner.last_checkpoint_timestamp()
    }

    /// Returns the cached written record count.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.inner.record_count()
    }
    /// Returns the canonical WAL directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        self.inner.dir()
    }
    /// Returns the cached physical sequence.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.inner.current_sequence()
    }
    /// Returns the configured durability mode.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.inner.durability_mode()
    }
    /// Returns the cached checkpoint epoch.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> Option<EpochId> {
        self.inner.checkpoint_epoch()
    }
    /// Returns the cached physical segment locator.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.inner.log_path(self.current_sequence())
    }
    /// Returns whether this owner has failed.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.inner.operation.lock().phase == Phase::Failed
    }
    /// Returns whether encryption was configured.
    #[cfg(feature = "encryption")]
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.inner.is_encrypted()
    }

    fn drain(&self, seal: bool, fail: bool) -> Result<Option<SealedWal>> {
        let mut supervisor = self.supervisor.lock();
        {
            let mut state = self.inner.operation.lock();
            if state.phase == Phase::Closed {
                return if seal || fail {
                    Err(terminal_error())
                } else {
                    Ok(None)
                };
            }
            if fail {
                state.phase = Phase::Failed;
            } else if state.phase == Phase::Open {
                state.phase = Phase::Closing;
            }
        }
        if let Some(stop) = supervisor.stop.take() {
            let _ = stop.send(());
        }
        let joined = match supervisor.worker.take() {
            Some(worker) => worker
                .join()
                .map_err(|_| Error::Internal("WAL Batch worker unwound".into())),
            None => Ok(()),
        };
        let state = self.inner.operation.lock();
        let mut operation = Operation {
            state,
            physical: false,
        };
        // Failed authority is non-writable, including cleanup. A failed join
        // also forbids final synchronization. Only healthy Closing may flush.
        let drain_result = joined.and_then(|()| {
            if operation.state.phase == Phase::Closing {
                operation.physical = true;
                self.inner.sync_active_log().map(|_| ())
            } else {
                Ok(())
            }
        });
        // BufWriter can perform I/O on Drop: consume it explicitly before W.
        if let Some(file) = self.inner.active_log.lock().take() {
            let (file, _buffer) = file.writer.into_parts();
            drop(file);
        }
        if let Err(error) = drain_result {
            operation.state.phase = Phase::Failed;
            operation.physical = false;
            return Err(operation.state.pending_cause.take().unwrap_or(error));
        }
        operation.physical = false;
        if operation.state.phase == Phase::Failed {
            if let Some(error) = operation.state.pending_cause.take() {
                return Err(error);
            }
            return if fail {
                Ok(None)
            } else {
                Err(terminal_error())
            };
        }
        operation.state.phase = Phase::Closed;
        let lease = operation.state.lease.take();
        match (seal, lease) {
            (true, Some(lease)) => Ok(Some(SealedWal { lease })),
            (true, None) => Err(terminal_error()),
            (false, lease) => {
                drop(lease);
                Ok(None)
            }
        }
    }

    /// Terminal, idempotent successful close; retires physical files before W.
    ///
    /// # Errors
    /// Failed close retains non-writable ownership until Drop.
    pub fn close(&self) -> Result<()> {
        self.drain(false, false).map(|_| ())
    }

    /// Drains the writer and transfers its authority exactly once.
    ///
    /// # Errors
    /// Returns physical errors or terminal-state errors.
    pub fn seal(&self) -> Result<SealedWal> {
        self.drain(true, false)?.ok_or_else(terminal_error)
    }

    /// Marks failed, joins workers and retires files while retaining W.
    ///
    /// # Errors
    /// Reports drain errors; success does not mean successful close.
    pub fn fail_and_drain(&self) -> Result<()> {
        self.drain(false, true).map(|_| ())
    }
}

impl Drop for WalManager {
    fn drop(&mut self) {
        let _ = self.drain(false, false);
        // Failed ownership is retained through every possible buffered I/O.
        if let Some(file) = self.inner.active_log.lock().take() {
            let (file, _buffer) = file.writer.into_parts();
            drop(file);
        }
        self.inner.operation.lock().lease.take();
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_wal_dir as tempdir;
    use super::*;
    use grafeo_common::types::NodeId;
    use std::sync::{Arc, Barrier};

    #[cfg(unix)]
    #[test]
    fn invalid_file_refusal_discards_buffered_and_large_frames_on_close_and_drop() {
        for size in [1, 16_384] {
            let dir = tempdir().unwrap();
            let wal = WalManager::open(dir.path()).unwrap();
            let source = wal.log_files().unwrap().pop().unwrap();
            let alias = dir.path().join("unexpected-link");
            fs::hard_link(&source, &alias).unwrap();
            let record = WalRecord::lpg(
                TransactionId::new(2),
                grafeo_common::types::GraphPath::root(),
                super::super::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["x".repeat(size)],
                },
            );
            assert!(wal.log(&record).is_err());
            assert!(fs::read(&source).unwrap().is_empty());
            assert!(wal.close().is_err());
            drop(wal);
            assert!(fs::read(&source).unwrap().is_empty());
            assert!(fs::read(&alias).unwrap().is_empty());
        }
    }

    thread_local! {
        static ADMISSION_RENDEZVOUS: std::cell::RefCell<Option<std::sync::mpsc::Sender<()>>> = const { std::cell::RefCell::new(None) };
    }

    pub(super) fn admission_test_point() {
        ADMISSION_RENDEZVOUS.with(|point| {
            if let Some(arrived) = point.borrow_mut().take() {
                arrived.send(()).unwrap();
            }
        });
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn backup_generation_detection_preserves_durable_and_installing_state() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let mut capture = wal.capture().unwrap();
        assert!(!capture.has_backup_generations().unwrap());
        let name = WalCapture::backup_generation_name(&[3; 32], 1);
        let installing = dir.path().join(format!("{name}.installing"));
        fs::write(&installing, b"interrupted").unwrap();
        assert!(capture.has_backup_generations().unwrap());
        fs::remove_file(installing).unwrap();
        capture
            .write_backup_generation_bytes(&[3; 32], 1, b"durable")
            .unwrap();
        assert!(capture.has_backup_generations().unwrap());
        fs::remove_file(dir.path().join(name)).unwrap();
        capture.write_backup_cursor_bytes(b"advisory").unwrap();
        assert!(capture.has_backup_generations().unwrap());
    }

    #[test]
    fn backup_generation_detection_rejects_malformed_reserved_names() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        fs::write(dir.path().join("backup_cursor_unknown.meta"), b"foreign").unwrap();
        assert!(wal.capture().unwrap().has_backup_generations().is_err());
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn backup_generation_io_failures_poison_and_preserve_prior_generation() {
        use grafeo_common::testing::wal_failure::with_backup_publication_failure;
        for point in [
            "backup:cursor_write",
            "backup:cursor_rename_failure",
            "backup:cursor_parent_sync",
        ] {
            let dir = tempdir().unwrap();
            let wal = WalManager::open(dir.path()).unwrap();
            wal.capture()
                .unwrap()
                .write_backup_generation_bytes(&[8; 32], 1, b"old")
                .unwrap();
            let result = with_backup_publication_failure(point, || {
                wal.capture()
                    .unwrap()
                    .write_backup_generation_bytes(&[8; 32], 2, b"new")
            });
            if point == "backup:cursor_write" {
                assert!(matches!(result, Err(Error::Storage(StorageError::Full))));
            } else {
                assert!(matches!(result, Err(Error::Io(_))));
            }
            assert!(wal.is_poisoned());
            assert!(wal.capture().is_err());
            assert_eq!(
                fs::read(
                    dir.path()
                        .join(WalCapture::backup_generation_name(&[8; 32], 1))
                )
                .unwrap(),
                b"old"
            );
            drop(wal);
            let reopened = WalManager::open(dir.path()).unwrap();
            let mut capture = reopened.capture().unwrap();
            assert_eq!(
                capture.read_backup_generation_bytes(&[8; 32], 1).unwrap(),
                Some(b"old".to_vec())
            );
            assert!(
                capture
                    .write_backup_generation_bytes(&[8; 32], 2, b"retry")
                    .is_err()
            );
        }
    }

    #[test]
    fn backup_generation_is_immutable_and_chain_qualified() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let mut capture = wal.capture().unwrap();
        assert!(
            capture
                .read_backup_generation_bytes(&[0xab; 32], 1)
                .unwrap()
                .is_none()
        );
        capture
            .write_backup_generation_bytes(&[0xab; 32], 1, b"first")
            .unwrap();
        capture
            .write_backup_generation_bytes(&[0xcd; 32], 1, b"other chain")
            .unwrap();
        assert_eq!(
            capture
                .read_backup_generation_bytes(&[0xab; 32], 1)
                .unwrap(),
            Some(b"first".to_vec())
        );
        assert!(
            capture
                .write_backup_generation_bytes(&[0xab; 32], 1, b"replacement")
                .is_err()
        );
        assert_eq!(
            fs::read(dir.path().join(format!(
                "backup_cursor_{}_00000000000000000001.meta",
                "ab".repeat(32)
            )))
            .unwrap(),
            b"first"
        );
        let name = WalCapture::backup_generation_name(&[0xab; 32], 2);
        fs::write(dir.path().join(format!("{name}.installing")), b"debris").unwrap();
        assert!(
            capture
                .write_backup_generation_bytes(&[0xab; 32], 2, b"retry")
                .is_err()
        );
        assert!(!dir.path().join(name).exists());
        assert!(WalManager::open(dir.path()).is_err());
    }

    #[test]
    fn backup_generation_read_is_bounded_and_exactly_named() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let name = WalCapture::backup_generation_name(&[0xab; 32], u64::MAX);
        assert_eq!(
            name,
            format!(
                "backup_cursor_{}_18446744073709551615.meta",
                "ab".repeat(32)
            )
        );
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.path().join(&name))
            .unwrap();
        file.set_len(64 * 1024 * 1024 + 18).unwrap();
        let mut capture = wal.capture().unwrap();
        assert!(
            capture
                .read_backup_generation_bytes(&[0xab; 32], u64::MAX)
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn backup_generation_rejects_symlink_artifacts() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let target = dir.path().join("foreign");
        fs::write(&target, b"foreign bytes").unwrap();
        let name = WalCapture::backup_generation_name(&[1; 32], 1);
        std::os::unix::fs::symlink(&target, dir.path().join(name)).unwrap();
        let mut capture = wal.capture().unwrap();
        assert!(capture.read_backup_generation_bytes(&[1; 32], 1).is_err());
        assert!(
            capture
                .write_backup_generation_bytes(&[1; 32], 1, b"replacement")
                .is_err()
        );
        assert_eq!(fs::read(target).unwrap(), b"foreign bytes");
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn backup_generation_publication_crashes_preserve_prior_bytes() {
        use grafeo_common::testing::crash::{CrashResult, with_crash_named};
        for point in [
            "backup:cursor_installing_sync",
            "backup:cursor_rename",
            "backup:cursor_directory_sync",
        ] {
            let dir = tempdir().unwrap();
            let wal = WalManager::open(dir.path()).unwrap();
            wal.capture()
                .unwrap()
                .write_backup_generation_bytes(&[7; 32], 1, b"old")
                .unwrap();
            let result = with_crash_named(point, || {
                wal.capture()
                    .unwrap()
                    .write_backup_generation_bytes(&[7; 32], 2, b"new")
                    .unwrap();
            });
            assert!(matches!(result, CrashResult::Crashed));
            assert!(wal.is_poisoned());
            assert_eq!(
                fs::read(
                    dir.path()
                        .join(WalCapture::backup_generation_name(&[7; 32], 1))
                )
                .unwrap(),
                b"old"
            );
            drop(wal);
            let reopened = WalManager::open(dir.path()).unwrap();
            let mut capture = reopened.capture().unwrap();
            assert_eq!(
                capture.read_backup_generation_bytes(&[7; 32], 1).unwrap(),
                Some(b"old".to_vec())
            );
            assert!(
                capture
                    .write_backup_generation_bytes(&[7; 32], 2, b"retry")
                    .is_err()
            );
        }
    }

    #[test]
    fn retention_lease_survives_rotations_and_concurrent_checkpoint() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());
        let lease = wal.retain_from(0).unwrap();
        let registered = Arc::new(Barrier::new(2));
        let worker_wal = Arc::clone(&wal);
        let worker_barrier = Arc::clone(&registered);
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            for id in 1..=6 {
                worker_wal
                    .log(&WalRecord::lpg(
                        TransactionId::new(id),
                        grafeo_common::types::GraphPath::root(),
                        super::super::LpgMutationOp::CreateNode {
                            id: NodeId::new(id),
                            labels: vec!["retained".into()],
                        },
                    ))
                    .unwrap();
                worker_wal
                    .log(&WalRecord::TransactionCommit {
                        transaction_id: TransactionId::new(id),
                    })
                    .unwrap();
                worker_wal.rotate().unwrap();
            }
            worker_wal
                .checkpoint(TransactionId::new(7), EpochId::new(1))
                .unwrap();
        });
        registered.wait();
        worker.join().unwrap();
        let mut capture = wal.capture_with_lease(&lease).unwrap();
        let segments = capture.segments().unwrap();
        assert_eq!(
            segments
                .iter()
                .map(WalSegmentDescriptor::sequence)
                .collect::<Vec<_>>(),
            (0..=7).collect::<Vec<_>>()
        );
        let captured_dir = tempdir().unwrap();
        let mut count = 0;
        for segment in &segments {
            let bytes = capture.read_segment(segment).unwrap();
            count += capture.count_frames(&bytes).unwrap();
            fs::write(
                captured_dir
                    .path()
                    .join(format!("wal_{:08}.log", segment.sequence())),
                bytes,
            )
            .unwrap();
        }
        assert_eq!(count, 13);
        // Replay the exact captured bytes through the existing recovery engine;
        // no source checkpoint metadata skips the retained comparison prefix.
        let mut recovery = super::super::WalRecovery::new(captured_dir.path()).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 13);
        let nodes: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::LpgMutation {
                    transaction_id,
                    op: super::super::LpgMutationOp::CreateNode { id, .. },
                    ..
                } => Some((transaction_id.as_u64(), id.as_u64())),
                _ => None,
            })
            .collect();
        let commits: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::TransactionCommit { transaction_id } => Some(transaction_id.as_u64()),
                _ => None,
            })
            .collect();
        assert_eq!(nodes, (1..=6).map(|id| (id, id)).collect::<Vec<_>>());
        assert_eq!(commits, (1..=6).collect::<Vec<_>>());
        assert_eq!(
            records
                .iter()
                .filter(|record| matches!(record, WalRecord::Checkpoint { .. }))
                .count(),
            1
        );
        // Dropping a lease during capture must not reacquire operation admission.
        drop(lease);
        drop(capture);
        wal.checkpoint(TransactionId::new(8), EpochId::new(2))
            .unwrap();
        assert!(!dir.path().join("wal_00000000.log").exists());
    }

    #[test]
    fn retention_lease_minimum_and_duplicate_starts_bound_retirement() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let oldest = wal.retain_from(1).unwrap();
        let duplicate = wal.retain_from(1).unwrap();
        let newer = wal.retain_from(3).unwrap();
        for _ in 0..6 {
            wal.rotate().unwrap();
        }
        wal.checkpoint(TransactionId::new(1), EpochId::new(1))
            .unwrap();
        assert!(!dir.path().join("wal_00000000.log").exists());
        assert!(dir.path().join("wal_00000001.log").exists());
        drop(oldest);
        wal.checkpoint(TransactionId::new(2), EpochId::new(2))
            .unwrap();
        assert!(dir.path().join("wal_00000001.log").exists());
        drop(duplicate);
        wal.checkpoint(TransactionId::new(3), EpochId::new(3))
            .unwrap();
        assert!(!dir.path().join("wal_00000002.log").exists());
        assert!(dir.path().join("wal_00000003.log").exists());
        drop(newer);
        wal.checkpoint(TransactionId::new(4), EpochId::new(4))
            .unwrap();
        // Epoch 4 is deliberately unrelated to segment 10: retirement uses
        // durable checkpoint sequence while preserving the recent-file margin.
        assert!(!dir.path().join("wal_00000007.log").exists());
        assert!(dir.path().join("wal_00000008.log").exists());
    }

    #[test]
    fn retention_lease_rejects_foreign_reopened_and_terminal_owners() {
        let dir = tempdir().unwrap();
        let other_dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let other = WalManager::open(other_dir.path()).unwrap();
        let lease = wal.retain_from(0).unwrap();
        assert!(other.capture_with_lease(&lease).is_err());
        wal.close().unwrap();
        assert!(wal.retain_from(0).is_err());
        assert!(wal.capture_with_lease(&lease).is_err());
        drop(wal);
        let reopened = WalManager::open(dir.path()).unwrap();
        assert!(reopened.capture_with_lease(&lease).is_err());
        let current = reopened.retain_from(0).unwrap();
        reopened.fail_and_drain().unwrap();
        assert!(reopened.retain_from(0).is_err());
        assert!(reopened.capture_with_lease(&current).is_err());
    }

    #[test]
    fn retention_lease_capacity_refusal_does_not_poison_owner() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        wal.inner.retention.lock().count = 65_536;
        assert!(wal.retain_from(0).is_err());
        assert!(wal.inner.retention.lock().starts.is_empty());
        wal.inner.retention.lock().count = 0;
        let lease = wal.retain_from(0).unwrap();
        assert!(wal.capture_with_lease(&lease).is_ok());
    }

    #[test]
    fn backup_cursor_read_rejects_oversized_file() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let file = File::create(dir.path().join("backup_cursor.meta")).unwrap();
        file.set_len(64 * 1024 * 1024 + 18).unwrap();
        let error = wal
            .capture()
            .unwrap()
            .read_backup_cursor_bytes()
            .unwrap_err();
        assert!(error.to_string().contains("encoded size limit"));
    }

    #[test]
    fn capture_excludes_checkpoint_retirement() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());
        wal.checkpoint(TransactionId::new(1), EpochId::new(1))
            .unwrap();
        wal.rotate().unwrap();
        wal.rotate().unwrap();
        assert_eq!(wal.current_sequence(), 3);
        let mut capture = wal.capture().unwrap();
        let segments = capture.segments().unwrap();
        let old = segments
            .iter()
            .find(|segment| segment.sequence() == 0)
            .unwrap();
        let before = capture.read_segment(old).unwrap();
        assert!(!before.is_empty());
        capture.write_backup_cursor_bytes(b"before-trim").unwrap();
        let (arrived, arrival) = std::sync::mpsc::channel();
        let contender = Arc::clone(&wal);
        let worker = std::thread::spawn(move || {
            ADMISSION_RENDEZVOUS.with(|point| *point.borrow_mut() = Some(arrived));
            contender.checkpoint(TransactionId::new(2), EpochId::new(10))
        });
        arrival.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(wal.inner.operation.try_lock().is_none());
        assert_eq!(capture.read_segment(old).unwrap(), before);
        assert_eq!(
            fs::read(dir.path().join("wal_00000000.log")).unwrap(),
            before
        );
        assert_eq!(
            capture.read_backup_cursor_bytes().unwrap(),
            Some(b"before-trim".to_vec())
        );
        drop(capture);
        worker.join().unwrap().unwrap();
        assert_eq!(wal.current_sequence(), 4);
        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(10)));
        assert!(!dir.path().join("wal_00000000.log").exists());
        assert!(!dir.path().join("wal_00000001.log").exists());
        assert_eq!(
            wal.capture().unwrap().read_backup_cursor_bytes().unwrap(),
            Some(b"before-trim".to_vec())
        );
        wal.close().unwrap();
    }

    #[test]
    fn capture_serializes_competing_cursor_publication() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());
        wal.checkpoint(TransactionId::new(1), EpochId::new(1))
            .unwrap();
        let mut capture = wal.capture().unwrap();
        let segments = capture.segments().unwrap();
        let before = capture.read_segment(&segments[0]).unwrap();
        capture.write_backup_cursor_bytes(b"held-cursor").unwrap();
        let (arrived, arrival) = std::sync::mpsc::channel();
        let contender = Arc::clone(&wal);
        let worker = std::thread::spawn(move || {
            ADMISSION_RENDEZVOUS.with(|point| *point.borrow_mut() = Some(arrived));
            let mut next = contender.capture().unwrap();
            assert_eq!(
                next.read_backup_cursor_bytes().unwrap(),
                Some(b"held-cursor".to_vec())
            );
            next.write_backup_cursor_bytes(b"contender-cursor").unwrap();
        });
        arrival.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(wal.inner.operation.try_lock().is_none());
        assert_eq!(capture.read_segment(&segments[0]).unwrap(), before);
        assert_eq!(
            capture.read_backup_cursor_bytes().unwrap(),
            Some(b"held-cursor".to_vec())
        );
        assert_eq!(
            fs::read(dir.path().join("backup_cursor.meta")).unwrap(),
            b"held-cursor"
        );
        drop(capture);
        worker.join().unwrap();
        assert_eq!(
            wal.capture().unwrap().read_backup_cursor_bytes().unwrap(),
            Some(b"contender-cursor".to_vec())
        );
        wal.close().unwrap();
    }

    #[test]
    fn capture_owns_operation_across_rotation_and_cursor_publication() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());
        let mut capture = wal.capture().unwrap();
        assert!(wal.inner.operation.try_lock().is_none());
        let (attempt, attempted) = std::sync::mpsc::channel();
        let (done, completed) = std::sync::mpsc::channel();
        let contender = Arc::clone(&wal);
        let worker = std::thread::spawn(move || {
            attempt.send(()).unwrap();
            contender.rotate().unwrap();
            done.send(()).unwrap();
        });
        attempted.recv().unwrap();
        capture.rotate().unwrap();
        capture.write_backup_cursor_bytes(b"held").unwrap();
        assert_eq!(capture.current_log_sequence(), 1);
        assert!(completed.try_recv().is_err());
        assert!(wal.inner.operation.try_lock().is_none());
        drop(capture);
        completed.recv().unwrap();
        worker.join().unwrap();
        assert_eq!(wal.current_sequence(), 2);
        assert_eq!(
            wal.capture().unwrap().read_backup_cursor_bytes().unwrap(),
            Some(b"held".to_vec())
        );
        wal.close().unwrap();
    }

    #[test]
    fn failed_drain_discards_pending_bytes() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let path = wal.path();
        let before = fs::read(&path).unwrap();
        {
            // Successful public writes flush; seed the real private buffer to
            // isolate externally forced failure with bytes still pending.
            let mut active = wal.inner.active_log.lock();
            let file = active.as_mut().unwrap();
            file.writer.write_all(b"pending external failure").unwrap();
            assert!(!file.writer.buffer().is_empty());
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        wal.fail_and_drain().unwrap();
        assert!(wal.is_poisoned());
        assert_eq!(
            fs::read(&path).unwrap(),
            before,
            "Failed drain wrote buffered bytes"
        );
        assert!(wal.close().is_err());
        assert!(wal.seal().is_err());
        assert!(WalManager::open(dir.path()).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(wal);
        assert_eq!(fs::read(&path).unwrap(), before);
        WalManager::open(dir.path()).unwrap().close().unwrap();
    }

    #[test]
    fn healthy_close_and_seal_flush_pending_bytes() {
        for seal in [false, true] {
            let dir = tempdir().unwrap();
            let wal = WalManager::open(dir.path()).unwrap();
            let path = wal.path();
            wal.inner
                .active_log
                .lock()
                .as_mut()
                .unwrap()
                .writer
                .write_all(b"healthy")
                .unwrap();
            assert!(fs::read(&path).unwrap().is_empty());
            if seal {
                let sealed = wal.seal().unwrap();
                assert_eq!(fs::read(&path).unwrap(), b"healthy");
                assert!(WalManager::open(dir.path()).is_err());
                drop(sealed);
            } else {
                wal.close().unwrap();
                assert_eq!(fs::read(&path).unwrap(), b"healthy");
            }
        }
    }

    #[test]
    fn failed_final_sync_is_not_retried_after_obstacle_removed() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let path = wal.path();
        wal.inner
            .active_log
            .lock()
            .as_mut()
            .unwrap()
            .writer
            .write_all(b"pending")
            .unwrap();
        let link = dir.path().join("obstacle");
        fs::hard_link(&path, &link).unwrap();
        assert!(wal.close().is_err());
        assert!(wal.is_poisoned());
        fs::remove_file(&link).unwrap();
        assert!(wal.close().is_err());
        assert!(wal.seal().is_err());
        assert!(WalManager::open(dir.path()).is_err());
        assert!(fs::read(&path).unwrap().is_empty());
        drop(wal);
        assert!(fs::read(&path).unwrap().is_empty());
    }

    #[test]
    #[cfg(feature = "testing-crash-injection")]
    fn failed_drain_after_physical_unwind_discards_pending_bytes() {
        use grafeo_common::testing::crash::{CrashResult, with_crash_named};
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let path = wal.path();
        let before = fs::read(&path).unwrap();
        let record = WalRecord::Checkpoint {
            transaction_id: TransactionId::new(1),
        };
        let outcome = with_crash_named("wal_after_write", || wal.log(&record));
        assert!(matches!(outcome, CrashResult::Crashed));
        assert!(wal.is_poisoned());
        assert!(
            !wal.inner
                .active_log
                .lock()
                .as_ref()
                .unwrap()
                .writer
                .buffer()
                .is_empty()
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        assert!(wal.close().is_err());
        assert_eq!(
            fs::read(&path).unwrap(),
            before,
            "Failed close retried the unwound frame"
        );
        assert!(wal.log(&record).is_err());
        assert!(wal.close().is_err());
        assert!(WalManager::open(dir.path()).is_err());
        drop(wal);
        assert_eq!(fs::read(&path).unwrap(), before);
        WalManager::open(dir.path()).unwrap().close().unwrap();
    }

    fn wal_file_image(wal: &WalManager) -> Vec<(PathBuf, Vec<u8>)> {
        wal.log_files()
            .unwrap()
            .into_iter()
            .map(|path| {
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect()
    }

    #[test]
    fn test_wal_write() {
        let dir = tempdir().unwrap();

        let wal = WalManager::open(dir.path()).unwrap();

        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            },
        );

        wal.log(&record).unwrap();
        wal.flush().unwrap();

        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn invalid_record_and_checkpoint_identity_are_rejected_before_mutation() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let before = wal_file_image(&wal);

        let record_error = wal
            .log(&WalRecord::Committed {
                transaction_id: TransactionId::INVALID,
                epoch: EpochId::new(1),
            })
            .unwrap_err();
        assert!(matches!(record_error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal), before);
        assert_eq!(wal.record_count(), 0);

        let checkpoint_error = wal
            .checkpoint(TransactionId::INVALID, EpochId::new(1))
            .unwrap_err();
        assert!(matches!(checkpoint_error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal), before);
        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.current_sequence(), 0);

        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .expect("deterministic validation must leave the WAL usable");
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn oversized_raw_frame_is_rejected_before_writer_state_changes() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let before = wal_file_image(&wal);
        let count_before = wal.record_count();
        let dirty_before = wal.inner.records_since_sync.load(Ordering::Acquire);
        let sync_before = *wal.inner.last_sync.lock();
        let payload = vec![0u8; super::super::MAX_WAL_FRAME_BYTES + 1];

        let error = wal
            .write_frame(
                &payload,
                WalFrameIntent::capture(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(1),
                }),
            )
            .unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal), before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(
            wal.inner.records_since_sync.load(Ordering::Acquire),
            dirty_before
        );
        assert_eq!(*wal.inner.last_sync.lock(), sync_before);
        assert_eq!(wal.current_sequence(), 0);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn catalog_failpoints_bracket_direct_wal_manager_append() {
        let dir = tempdir().unwrap();
        let wal = WalManager::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                ..WalConfig::default()
            },
        )
        .unwrap();
        let batch = |epoch| WalRecord::CatalogBatchV3 {
            created_graph_incarnations: vec![],
            dropped_graph_incarnations: vec![],
            version: 2,
            epoch: EpochId::new(epoch),
            catalog_state: Vec::new(),
            created_graphs: Vec::new(),
            dropped_graphs: Vec::new(),
        };

        grafeo_common::testing::wal_failure::enable_catalog_batch_log_failure_once();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .expect("the catalog hook must ignore non-catalog records");
        assert!(wal.log(&batch(2)).is_err());
        assert!(
            grafeo_common::testing::wal_failure::maybe_fail_catalog_batch_log().is_ok(),
            "the pre-append hook is one-shot"
        );
        assert_eq!(wal.record_count(), 1);

        assert!(wal.is_poisoned());
        drop(wal);
        let wal = WalManager::open(dir.path()).unwrap();
        grafeo_common::testing::wal_failure::enable_catalog_batch_ack_failure_once();
        assert!(wal.log(&batch(3)).is_err());
        assert!(
            grafeo_common::testing::wal_failure::maybe_fail_catalog_batch_ack().is_ok(),
            "the acknowledgement hook is one-shot"
        );
        assert_eq!(wal.record_count(), 1);

        drop(wal);
        let recovered = super::super::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(matches!(
            recovered.as_slice(),
            [WalRecord::EpochAdvance { epoch: first }, WalRecord::CatalogBatchV3 { epoch: second, .. }]
                if *first == EpochId::new(1) && *second == EpochId::new(3)
        ));
    }

    #[test]
    fn test_wal_rotation() {
        let dir = tempdir().unwrap();

        // Small max size to force rotation
        let config = WalConfig {
            max_log_size: 100,
            ..Default::default()
        };

        let wal = WalManager::with_config(dir.path(), config).unwrap();

        // Write enough records to trigger rotation
        for i in 0..10 {
            let record = WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Person".to_string()],
                },
            );
            wal.log(&record).unwrap();
        }

        wal.flush().unwrap();

        // Should have multiple log files
        let files = wal.log_files().unwrap();
        assert!(
            files.len() > 1,
            "Expected multiple log files after rotation"
        );
    }

    #[test]
    fn concurrent_rotation_preserves_file_order_and_every_frame() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(
            WalManager::with_config(
                dir.path(),
                WalConfig {
                    durability: DurabilityMode::NoSync,
                    // Every frame rotates, maximizing contention on the
                    // active-file transition.
                    max_log_size: 1,
                    compression: false,
                },
            )
            .unwrap(),
        );
        let threads = 8usize;
        let records_per_thread = 12usize;
        let start = Arc::new(Barrier::new(threads));
        let mut workers = Vec::new();
        for thread in 0..threads {
            let wal = Arc::clone(&wal);
            let start = Arc::clone(&start);
            workers.push(std::thread::spawn(move || {
                start.wait();
                for offset in 0..records_per_thread {
                    let epoch = thread * records_per_thread + offset + 1;
                    wal.log(&WalRecord::EpochAdvance {
                        epoch: EpochId::new(epoch as u64),
                    })
                    .unwrap();
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        wal.sync().unwrap();

        let record_count = (threads * records_per_thread) as u64;
        assert_eq!(wal.record_count(), record_count);
        assert_eq!(wal.current_sequence(), record_count);
        assert_eq!(
            WalInner::sequence_from_path(&wal.path()),
            Some(wal.current_sequence()),
            "the published sequence must name the actual active file"
        );
        let files = wal.log_files().unwrap();
        assert_eq!(files.len() as u64, record_count + 1);
        for (expected, path) in files.iter().enumerate() {
            assert_eq!(
                WalInner::sequence_from_path(path),
                Some(expected as u64),
                "rotation must leave one contiguous physical history"
            );
        }

        wal.close().unwrap();
        let recovered = crate::wal::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        let epochs: std::collections::HashSet<_> = recovered
            .into_iter()
            .filter_map(|record| match record {
                WalRecord::EpochAdvance { epoch } => Some(epoch.as_u64()),
                _ => None,
            })
            .collect();
        assert_eq!(epochs.len() as u64, record_count);
        assert!((1..=record_count).all(|epoch| epochs.contains(&epoch)));
    }

    #[test]
    fn concurrent_checkpoints_publish_monotonic_metadata_and_recover_post_boundary() {
        use super::super::{LpgMutationOp, WalRecovery};
        use std::sync::atomic::AtomicBool;

        const THREADS: usize = 12;
        const CHECKPOINTS_PER_THREAD: usize = 8;
        const POST_BOUNDARY_TRANSACTIONS: u64 = 24;

        let dir = tempdir().unwrap();
        let wal = Arc::new(
            WalManager::with_config(
                dir.path(),
                WalConfig {
                    durability: DurabilityMode::Sync,
                    max_log_size: u64::MAX,
                    compression: false,
                },
            )
            .unwrap(),
        );

        // The engine persists its store snapshot before invoking checkpoint.
        // Seed one such boundary so the monitor always has metadata to read.
        wal.checkpoint(TransactionId::new(1), EpochId::new(1))
            .unwrap();

        let stop_monitor = Arc::new(AtomicBool::new(false));
        let observations = Arc::new(std::sync::Mutex::new(Vec::<(u64, u64)>::new()));
        let monitor = {
            let wal = Arc::clone(&wal);
            let stop_monitor = Arc::clone(&stop_monitor);
            let observations = Arc::clone(&observations);
            std::thread::spawn(move || {
                while !stop_monitor.load(Ordering::Acquire) {
                    if let Some(metadata) = wal.read_checkpoint_metadata().unwrap() {
                        observations
                            .lock()
                            .unwrap()
                            .push((metadata.log_sequence, metadata.epoch.as_u64()));
                    }
                    std::thread::yield_now();
                }
            })
        };

        let start = Arc::new(Barrier::new(THREADS));
        let next_epoch = Arc::new(AtomicU64::new(2));
        let mut workers = Vec::new();
        for _ in 0..THREADS {
            let wal = Arc::clone(&wal);
            let start = Arc::clone(&start);
            let next_epoch = Arc::clone(&next_epoch);
            workers.push(std::thread::spawn(move || {
                start.wait();
                for _ in 0..CHECKPOINTS_PER_THREAD {
                    let epoch = next_epoch.fetch_add(1, Ordering::SeqCst);
                    wal.checkpoint(TransactionId::new(epoch), EpochId::new(epoch))
                        .unwrap();
                    std::thread::yield_now();
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        stop_monitor.store(true, Ordering::Release);
        monitor.join().unwrap();

        let max_epoch = 1 + u64::try_from(THREADS * CHECKPOINTS_PER_THREAD).unwrap();
        let metadata = wal
            .read_checkpoint_metadata()
            .unwrap()
            .expect("checkpoint metadata");
        assert_eq!(metadata.epoch, EpochId::new(max_epoch));
        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(max_epoch)));
        assert_eq!(
            metadata.log_sequence,
            wal.current_sequence(),
            "metadata must name the exact currently installed fresh segment"
        );
        assert_eq!(
            fs::metadata(wal.path()).unwrap().len(),
            0,
            "the published checkpoint boundary must initially be fresh"
        );

        let observations = observations.lock().unwrap();
        assert!(!observations.is_empty());
        for pair in observations.windows(2) {
            assert!(
                pair[0].0 <= pair[1].0 && pair[0].1 <= pair[1].1,
                "checkpoint metadata regressed from {:?} to {:?}",
                pair[0],
                pair[1]
            );
        }
        drop(observations);

        for offset in 0..POST_BOUNDARY_TRANSACTIONS {
            let value = 10_000 + offset;
            let transaction_id = TransactionId::new(value);
            wal.log(&WalRecord::lpg(
                transaction_id,
                grafeo_common::types::GraphPath::root(),
                LpgMutationOp::CreateNode {
                    id: NodeId::new(value),
                    labels: vec!["PostBoundary".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(max_epoch + offset + 1),
            })
            .unwrap();
        }
        wal.sync().unwrap();
        drop(wal);

        let recovered = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert_eq!(
            recovered.len(),
            usize::try_from(POST_BOUNDARY_TRANSACTIONS * 2).unwrap()
        );
        assert!(recovered.iter().all(|record| match record {
            WalRecord::LpgMutation { transaction_id, .. }
            | WalRecord::Committed { transaction_id, .. } => {
                transaction_id.as_u64() >= 10_000
            }
            _ => false,
        }));
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn concurrent_encrypted_rotation_uses_the_active_file_sequence_for_nonces() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [73u8; KEY_SIZE];
        let chain = KeyChain::new(key);
        let manager = WalManager::with_config_and_encryptor(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::NoSync,
                max_log_size: 1,
                compression: false,
            },
            chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
        )
        .unwrap();

        let wal = Arc::new(manager);
        let threads = 8usize;
        let records_per_thread = 8usize;
        let start = Arc::new(Barrier::new(threads));
        let mut workers = Vec::new();
        for thread in 0..threads {
            let wal = Arc::clone(&wal);
            let start = Arc::clone(&start);
            workers.push(std::thread::spawn(move || {
                start.wait();
                for offset in 0..records_per_thread {
                    wal.log(&WalRecord::EpochAdvance {
                        epoch: EpochId::new((thread * records_per_thread + offset + 1) as u64),
                    })
                    .unwrap();
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        wal.sync().unwrap();

        wal.close().unwrap();
        let mut recovery = crate::wal::WalRecovery::with_encryptor(
            dir.path(),
            chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
        )
        .unwrap();

        let records = recovery.recover().unwrap();
        let recovered = records
            .iter()
            .filter(|record| matches!(record, WalRecord::EpochAdvance { .. }))
            .count();
        assert_eq!(recovered, threads * records_per_thread);
        assert_eq!(wal.current_sequence(), recovered as u64);
        assert_eq!(
            WalInner::sequence_from_path(&wal.path()),
            Some(wal.current_sequence())
        );
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn encrypted_checkpoint_keeps_the_fresh_boundary_recovery_compatible() {
        use super::super::LpgMutationOp;
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [91u8; KEY_SIZE];
        let chain = KeyChain::new(key);
        let wal = WalManager::with_config_and_encryptor(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                max_log_size: u64::MAX,
                compression: false,
            },
            chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
        )
        .unwrap();

        let before = TransactionId::new(1);
        wal.log(&WalRecord::lpg(
            before,
            grafeo_common::types::GraphPath::root(),
            LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Before".to_string()],
            },
        ))
        .unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: before,
            epoch: EpochId::new(1),
        })
        .unwrap();
        wal.checkpoint(before, EpochId::new(1)).unwrap();

        let after = TransactionId::new(2);
        wal.log(&WalRecord::lpg(
            after,
            grafeo_common::types::GraphPath::root(),
            LpgMutationOp::CreateNode {
                id: NodeId::new(2),
                labels: vec!["After".to_string()],
            },
        ))
        .unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: after,
            epoch: EpochId::new(2),
        })
        .unwrap();
        wal.sync().unwrap();
        drop(wal);

        let mut recovery = crate::wal::WalRecovery::with_encryptor(
            dir.path(),
            chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
        )
        .unwrap();

        let recovered = recovery.recover().unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(recovered.iter().all(|record| match record {
            WalRecord::LpgMutation { transaction_id, .. }
            | WalRecord::Committed { transaction_id, .. } => *transaction_id == after,
            _ => false,
        }));
    }

    #[test]
    fn test_durability_modes() {
        let dir = tempdir().unwrap();

        assert_eq!(
            DurabilityMode::default(),
            DurabilityMode::Sync,
            "the WAL must default to strict commit-ack durability"
        );
        assert_eq!(WalConfig::default().durability, DurabilityMode::Sync);

        // Test Sync mode
        let config = WalConfig {
            durability: DurabilityMode::Sync,
            ..Default::default()
        };
        let wal =
            WalManager::with_config(dir.path().parent().unwrap().join("sync.wal"), config).unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .unwrap();

        // Test NoSync mode
        let config = WalConfig {
            durability: DurabilityMode::NoSync,
            ..Default::default()
        };
        let wal = WalManager::with_config(dir.path().parent().unwrap().join("nosync.wal"), config)
            .unwrap();
        wal.log(&WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec![],
            },
        ))
        .unwrap();

        // Test Batch mode
        let config = WalConfig {
            durability: DurabilityMode::Batch {
                max_delay_ms: 10,
                max_records: 5,
            },
            ..Default::default()
        };
        let wal = WalManager::with_config(dir.path().parent().unwrap().join("batch.wal"), config)
            .unwrap();
        for i in 0..10 {
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(i),
                    labels: vec![],
                },
            ))
            .unwrap();
        }

        // Test Adaptive mode (just buffer flush, no inline sync)
        let config = WalConfig {
            durability: DurabilityMode::Adaptive {
                target_interval_ms: 100,
            },
            ..Default::default()
        };
        let wal =
            WalManager::with_config(dir.path().parent().unwrap().join("adaptive.wal"), config)
                .unwrap();
        for i in 0..10 {
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(i),
                    labels: vec![],
                },
            ))
            .unwrap();
        }
        // Manually sync since no flusher thread in this test
        wal.sync().unwrap();
    }

    #[test]
    fn test_checkpoint() {
        let dir = tempdir().unwrap();

        let wal = WalManager::open(dir.path()).unwrap();

        // Write some records
        wal.log(&WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            },
        ))
        .unwrap();

        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .unwrap();

        // Create checkpoint
        wal.checkpoint(TransactionId::new(1), EpochId::new(10))
            .unwrap();

        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(10)));
    }

    #[test]
    fn pending_checkpoint_is_rejected_without_mutation_and_initial_round_trips() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let files_before = wal_file_image(&wal);
        let count_before = wal.record_count();
        let sequence_before = wal.current_sequence();

        let error = wal
            .checkpoint(TransactionId::new(1), EpochId::PENDING)
            .unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert_eq!(wal_file_image(&wal), files_before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(wal.current_sequence(), sequence_before);
        assert_eq!(wal.checkpoint_epoch(), None);
        assert!(!dir.path().join(CHECKPOINT_METADATA_FILE).exists());

        wal.checkpoint(TransactionId::new(1), EpochId::INITIAL)
            .unwrap();
        let metadata = wal.read_checkpoint_metadata().unwrap().unwrap();
        assert_eq!(metadata.epoch, EpochId::INITIAL);
        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::INITIAL));
    }

    #[test]
    fn hostile_pending_checkpoint_metadata_is_corruption() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let hostile = CheckpointMetadata {
            format_version: 5,
            retired_before: 0,
            epoch: EpochId::PENDING,
            log_sequence: 0,
            timestamp_ms: 0,
            transaction_id: TransactionId::new(1),
        };
        let data = bincode::serde::encode_to_vec(hostile, bincode::config::standard()).unwrap();
        fs::write(dir.path().join(CHECKPOINT_METADATA_FILE), data).unwrap();

        let error = wal.read_checkpoint_metadata().unwrap_err();
        assert!(matches!(error, Error::Storage(StorageError::Corruption(_))));
    }

    #[test]
    fn checkpoint_metadata_reader_is_exact_bounded_and_validates_identity() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let path = dir.path().join(CHECKPOINT_METADATA_FILE);

        assert!(wal.read_checkpoint_metadata().unwrap().is_none());

        let valid = CheckpointMetadata {
            format_version: 5,
            retired_before: 0,
            epoch: EpochId::new(7),
            log_sequence: 3,
            timestamp_ms: 42,
            transaction_id: TransactionId::new(11),
        };
        let valid_bytes =
            bincode::serde::encode_to_vec(&valid, bincode::config::standard()).unwrap();
        fs::write(&path, &valid_bytes).unwrap();
        let decoded = wal.read_checkpoint_metadata().unwrap().unwrap();
        assert_eq!(decoded.epoch, valid.epoch);
        assert_eq!(decoded.log_sequence, valid.log_sequence);
        assert_eq!(decoded.timestamp_ms, valid.timestamp_ms);
        assert_eq!(decoded.transaction_id, valid.transaction_id);

        let assert_corruption_without_mutation = |bytes: &[u8]| {
            fs::write(&path, bytes).unwrap();
            let original = fs::read(&path).unwrap();
            let error = wal.read_checkpoint_metadata().unwrap_err();
            assert!(
                matches!(error, Error::Storage(StorageError::Corruption(_))),
                "{error:?}"
            );
            assert_eq!(fs::read(&path).unwrap(), original);
        };

        let mut trailing = valid_bytes;
        trailing.push(0xa5);
        assert_corruption_without_mutation(&trailing);

        let oversized = vec![0u8; MAX_CHECKPOINT_METADATA_BYTES + 1];
        assert_corruption_without_mutation(&oversized);

        let invalid_transaction = CheckpointMetadata {
            transaction_id: TransactionId::INVALID,
            ..valid
        };
        let invalid_transaction_bytes =
            bincode::serde::encode_to_vec(invalid_transaction, bincode::config::standard())
                .unwrap();
        assert_corruption_without_mutation(&invalid_transaction_bytes);
    }

    #[test]
    fn exhausted_sequence_cannot_partially_append_a_checkpoint() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        {
            let mut guard = wal.inner.active_log.lock();
            guard.as_mut().unwrap().sequence = u64::MAX;
        }
        wal.inner
            .current_sequence
            .store(u64::MAX, Ordering::Release);
        let files_before = wal_file_image(&wal);
        let count_before = wal.record_count();

        let error = wal
            .checkpoint(TransactionId::new(1), EpochId::INITIAL)
            .unwrap_err();

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(
            error
                .to_string()
                .contains("WAL log sequence identity space is exhausted")
        );
        assert_eq!(wal_file_image(&wal), files_before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(wal.checkpoint_epoch(), None);
        assert!(!dir.path().join(CHECKPOINT_METADATA_FILE).exists());
    }

    #[test]
    fn threshold_crossing_at_exhausted_sequence_is_byte_and_count_identical() {
        let dir = tempdir().unwrap();
        let wal = WalManager::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::NoSync,
                // A complete current metadata frame reaches this threshold.
                max_log_size: 9,
                compression: false,
            },
        )
        .unwrap();
        {
            let mut guard = wal.inner.active_log.lock();
            guard.as_mut().unwrap().sequence = u64::MAX;
        }
        wal.inner
            .current_sequence
            .store(u64::MAX, Ordering::Release);
        let files_before = wal_file_image(&wal);
        let count_before = wal.record_count();
        let dirty_before = wal.inner.records_since_sync.load(Ordering::Acquire);
        let sync_before = *wal.inner.last_sync.lock();

        let error = wal
            .write_frame(
                &encode_frame(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(1),
                })
                .unwrap(),
                WalFrameIntent::capture(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(1),
                }),
            )
            .unwrap_err();

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert!(error.to_string().contains("rotation threshold"));
        assert_eq!(wal_file_image(&wal), files_before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(
            wal.inner.records_since_sync.load(Ordering::Acquire),
            dirty_before
        );
        assert_eq!(*wal.inner.last_sync.lock(), sync_before);
        assert_eq!(wal.inner.active_log.lock().as_ref().unwrap().size, 0);
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn encrypted_threshold_projection_accounts_for_nonce_and_tag_before_append() {
        use grafeo_common::encryption::{ENCRYPTION_OVERHEAD, KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let encoded = encode_frame(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .unwrap();
        let data = encoded.as_slice();
        let encoded_size = 4 + data.len() + ENCRYPTION_OVERHEAD;
        let chain = KeyChain::new([0x5Au8; KEY_SIZE]);
        let wal = WalManager::with_config_and_encryptor(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::NoSync,
                max_log_size: u64::try_from(encoded_size).unwrap(),
                compression: false,
            },
            chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
        )
        .unwrap();

        assert_eq!(
            wal.inner
                .encoded_payload_len_from_plaintext(
                    super::super::MAX_WAL_FRAME_BYTES - ENCRYPTION_OVERHEAD,
                )
                .unwrap(),
            super::super::MAX_WAL_FRAME_BYTES
        );
        assert!(
            validate_wal_frame_payload_len(
                wal.inner
                    .encoded_payload_len_from_plaintext(
                        super::super::MAX_WAL_FRAME_BYTES - ENCRYPTION_OVERHEAD + 1,
                    )
                    .unwrap(),
            )
            .is_err(),
            "the recovery cap includes nonce and authentication-tag overhead"
        );
        {
            let mut guard = wal.inner.active_log.lock();
            guard.as_mut().unwrap().sequence = u64::MAX;
        }
        wal.inner
            .current_sequence
            .store(u64::MAX, Ordering::Release);
        let files_before = wal_file_image(&wal);
        let count_before = wal.record_count();

        let error = wal
            .write_frame(
                data,
                WalFrameIntent::capture(&WalRecord::EpochAdvance {
                    epoch: EpochId::new(1),
                }),
            )
            .unwrap_err();

        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::StorageFull
        );
        assert_eq!(wal_file_image(&wal), files_before);
        assert_eq!(wal.record_count(), count_before);
        assert_eq!(wal.inner.active_log.lock().as_ref().unwrap().size, 0);
    }

    #[test]
    fn batch_mode_syncs_an_idle_dirty_wal_at_its_time_deadline() {
        let dir = tempdir().unwrap();
        let wal = WalManager::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Batch {
                    max_delay_ms: 25,
                    max_records: u64::MAX,
                },
                max_log_size: u64::MAX,
                compression: false,
            },
        )
        .unwrap();
        let data = encode_frame(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .unwrap();
        let before = {
            // Seed the real framed writer while the idle worker is excluded.
            // Public log may legitimately sync inline or lose the observation
            // race to the worker; neither isolates the idle-worker contract.
            let mut operation = wal.admit().unwrap();
            operation.physical = true;
            let mut active = wal.inner.active_log.lock();
            let log = active.as_mut().unwrap();
            wal.inner.append_frame_locked(log, &data).unwrap();
            log.writer.flush().unwrap();
            assert_eq!(wal.inner.records_since_sync.load(Ordering::Acquire), 1);
            let before = *wal.inner.last_sync.lock();
            operation.complete(Ok(())).unwrap();
            before
        };

        let deadline = Instant::now() + Duration::from_secs(2);
        while wal.inner.records_since_sync.load(Ordering::Acquire) != 0 && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(2));
        }
        {
            // Observe one completed worker transition: its dirty counter is
            // cleared just before the timestamp update under this admission.
            let _operation = wal.admit().unwrap();
            assert_eq!(
                wal.inner.records_since_sync.load(Ordering::Acquire),
                0,
                "Batch must fsync at max_delay even when no later append arrives"
            );
            assert!(*wal.inner.last_sync.lock() > before);
        }

        wal.close().unwrap();
        let recovered = crate::wal::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(matches!(
            recovered.as_slice(),
            [WalRecord::EpochAdvance { epoch }] if *epoch == EpochId::new(1)
        ));
    }
}
