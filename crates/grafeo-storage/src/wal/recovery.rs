//! WAL recovery.

use super::ownership::WalLease;
use super::record::WalEntry;
use super::{CheckpointMetadata, MAX_WAL_FRAME_BYTES, WalConfig, WalManager, WalRecord};
use crate::ownership::{checked_file, validate_file};
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_common::{grafeo_debug, grafeo_info};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The recovery owner reads sequentially; count consumed bytes rather than
/// issuing a file-position syscall at every frame boundary and length check.
struct WalReader {
    reader: BufReader<File>,
    position: u64,
}

impl WalReader {
    fn new(mut file: File) -> std::io::Result<Self> {
        let position = file.stream_position()?;
        Ok(Self {
            reader: BufReader::new(file),
            position,
        })
    }

    fn get_ref(&self) -> &File {
        self.reader.get_ref()
    }

    fn into_inner(self) -> File {
        self.reader.into_inner()
    }
}

impl Read for WalReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.reader.read(buffer)?;
        self.position = self
            .position
            .checked_add(count as u64)
            .ok_or_else(|| std::io::Error::other("WAL reader position overflow"))?;
        Ok(count)
    }
}

impl Seek for WalReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.position = self.reader.seek(position)?;
        Ok(self.position)
    }

    fn stream_position(&mut self) -> std::io::Result<u64> {
        Ok(self.position)
    }
}

/// One WAL frame: a complete record, clean EOF, a torn tail, or a corrupt frame.
enum FrameRead<R> {
    Record(R),
    Eof,
    /// Length/payload/checksum cut short. Truncate to the start of this frame.
    Incomplete,
    /// Fully-sized frame failed its checksum or authentication boundary.
    Corrupt(Error),
    /// Checksum/authentication-valid payload is not one exact WAL entry.
    InvalidEntry(String),
}

/// Name of the checkpoint metadata file.
const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

/// Committed WAL records plus the replay-range/checkpoint transaction high-water.
///
/// `max_transaction_id` includes aborted and crash-orphaned records so a
/// later process never reuses an id still present in retained WAL history.
#[derive(Debug, Clone)]
pub struct RecoveryReport<R> {
    /// Records that were part of a committed transaction (plus checkpoints).
    pub committed: Vec<R>,
    /// Greatest valid transaction id in the replay range or checkpoint,
    /// including uncommitted tails. Comparison-only prefix records do not
    /// contribute. `None` if neither source contained a transaction id.
    pub max_transaction_id: Option<TransactionId>,
}

struct RecoveryScan<R> {
    has_prior_state: bool,
    last_commit_epoch: Option<EpochId>,
    finished_transactions: std::collections::HashSet<TransactionId>,
    torn_tail: Option<(File, u64, u64)>,
    prefix_authority: Vec<R>,
    groups: super::group::Groups,
}

impl<R> Default for RecoveryScan<R> {
    fn default() -> Self {
        Self {
            has_prior_state: false,
            last_commit_epoch: None,
            finished_transactions: std::collections::HashSet::new(),
            torn_tail: None,
            prefix_authority: Vec::new(),
            groups: super::group::Groups::default(),
        }
    }
}

/// Handles WAL recovery after a crash.
pub struct WalRecovery {
    lease: WalLease,
    ready: Option<(u64, Option<CheckpointMetadata>, super::group::Groups)>,
    /// Directory containing WAL files.
    dir: std::path::PathBuf,
    /// Encryptor for decrypting WAL records (None = unencrypted).
    #[cfg(feature = "encryption")]
    encryptor: Option<grafeo_common::encryption::PageEncryptor>,
}

impl WalRecovery {
    /// Acquires recovery ownership before observing any WAL content.
    ///
    /// # Errors
    /// Returns namespace, filesystem, or nonblocking contention errors.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let lease = WalLease::acquire(dir.as_ref())?;
        Ok(Self::from_lease(lease))
    }

    pub(super) fn from_lease(lease: WalLease) -> Self {
        Self {
            dir: lease.path().to_path_buf(),
            lease,
            ready: None,
            #[cfg(feature = "encryption")]
            encryptor: None,
        }
    }

    #[cfg(feature = "grafeo-file")]
    pub(crate) fn for_container_restore(lease: WalLease) -> Self {
        Self::from_lease(lease)
    }

    /// Direct append admission validates the shared physical/current-generation
    /// boundary without assuming the caller's custom record type is WalRecord.
    pub(super) fn validate_append_lease(
        lease: WalLease,
        #[cfg(feature = "encryption")] encryptor: Option<grafeo_common::encryption::PageEncryptor>,
    ) -> Result<Self> {
        let mut recovery = Self::from_lease(lease);
        #[cfg(feature = "encryption")]
        {
            recovery.encryptor = encryptor;
        }
        recovery.refuse_if_quarantined()?;
        let checkpoint = recovery.read_checkpoint_metadata()?;
        let files = recovery.get_log_files()?;
        Self::validate_checkpoint_boundary(&files, checkpoint.as_ref())?;
        let mut groups = super::group::Groups::default();
        let group_floor = checkpoint
            .as_ref()
            .map_or(0, |metadata| metadata.retired_before);
        for (sequence, path) in &files {
            if *sequence >= group_floor {
                groups.observe_segment(*sequence)?;
            }
            recovery.validate_segment_for_append(path, &mut groups, *sequence >= group_floor)?;
        }
        recovery.ready = Some((
            files.last().map_or(0, |(sequence, _)| *sequence),
            checkpoint,
            groups,
        ));
        Ok(recovery)
    }

    fn validate_segment_for_append(
        &self,
        path: &Path,
        groups: &mut super::group::Groups,
        verify_group: bool,
    ) -> Result<()> {
        let mut reader = WalReader::new(checked_file(path, false, false)?)?;
        loop {
            let offset = reader.stream_position()?;
            match self.read_frame_payload(&mut reader)? {
                FrameRead::Record(data) => {
                    if verify_group {
                        let prepared = groups.prepare(&data, false)?;
                        groups.apply(prepared, &data);
                    }
                }
                FrameRead::Eof => return Ok(()),
                FrameRead::Incomplete => {
                    return Err(Error::Storage(StorageError::Corruption(format!(
                        "incomplete WAL frame in {} at byte offset {offset}; recover before append",
                        path.display()
                    ))));
                }
                FrameRead::Corrupt(error) => {
                    return Err(error.with_context(format!(
                        "WAL frame in {} at byte offset {offset}",
                        path.display()
                    )));
                }
                FrameRead::InvalidEntry(reason) => {
                    return Err(Self::invalid_wal_entry(path, offset, &reason));
                }
            }
        }
    }

    /// Canonical path bound to this owned recovery request.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// Retires a successfully scanned recovery without creating a segment.
    ///
    /// # Errors
    /// Rejects a recovery that has not completed successfully.
    pub fn seal(self) -> Result<super::SealedWal> {
        if self.ready.is_none() {
            return Err(Error::InvalidValue(
                "WAL recovery must finish successfully before sealing".into(),
            ));
        }
        Ok(super::SealedWal { lease: self.lease })
    }

    /// Acquires recovery with an encryptor fixed before scanning.
    ///
    /// # Errors
    /// Returns admission errors.
    #[cfg(feature = "encryption")]
    pub fn with_encryptor(
        dir: impl AsRef<Path>,
        encryptor: grafeo_common::encryption::PageEncryptor,
    ) -> Result<Self> {
        let mut recovery = Self::new(dir)?;
        recovery.encryptor = Some(encryptor);
        Ok(recovery)
    }

    /// Transfers a successfully scanned owner into its writer.
    ///
    /// # Errors
    /// Rejects never-run or failed recovery and returns initialization errors.
    pub fn into_wal(self, config: WalConfig) -> Result<WalManager> {
        let observed = self.ready.ok_or_else(|| {
            Error::InvalidValue(
                "WAL recovery must finish successfully before writer handoff".into(),
            )
        })?;
        WalManager::from_lease(
            self.lease,
            config,
            Some(observed),
            #[cfg(feature = "encryption")]
            self.encryptor,
        )
    }

    fn complete_scan(
        &mut self,
        checkpoint: Option<CheckpointMetadata>,
        groups: super::group::Groups,
    ) -> Result<()> {
        let high = self
            .get_log_files()?
            .last()
            .map_or(0, |(sequence, _)| *sequence);
        self.ready = Some((high, checkpoint, groups));
        Ok(())
    }

    /// Reads and structurally validates checkpoint metadata if it exists.
    ///
    /// Returns `None` if no checkpoint metadata is found.
    /// Physical WAL-boundary validation is performed atomically with stream
    /// enumeration by recovery; this accessor alone does not prove that the
    /// named segment exists or that its replay suffix is contiguous.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata file cannot be read or is oversized,
    /// malformed, has trailing bytes, or contains a reserved identity sentinel.
    pub fn read_checkpoint_metadata(&mut self) -> Result<Option<CheckpointMetadata>> {
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);
        CheckpointMetadata::read_from_path(&metadata_path)
    }

    /// Recovers committed records from all WAL files.
    ///
    /// Returns only records that were part of committed transactions.
    /// If checkpoint metadata exists, only replays files from the
    /// checkpoint sequence onwards.
    ///
    /// A prior full-frame quarantine (`*.log.corrupt` or `WAL_CORRUPT`) is a
    /// durable fail-closed condition: retrying recover/open returns corruption
    /// instead of skipping the quarantined segment.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails, a checkpoint names an absent or
    /// ambiguous WAL segment, its replay range contains a gap, or a
    /// quarantined WAL file is present.
    pub fn recover(&mut self) -> Result<Vec<WalRecord>> {
        Ok(self.recover_report()?.committed)
    }

    /// Recovers committed records and the transaction-id high-water mark.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_report(&mut self) -> Result<RecoveryReport<WalRecord>> {
        self.recover_report_as()
    }

    /// Scans without modifying the WAL, validates the recovered evidence,
    /// then repairs an incomplete final frame only after validation succeeds.
    ///
    /// The callback receives committed records and whether checkpoint metadata
    /// or any physical segment bytes exist. An empty committed list alone is
    /// not proof of fresh startup: aborted, orphaned and torn frames count as
    /// prior state. A newly allocated zero-byte segment alone does not.
    /// Identity/model declarations in retained pre-checkpoint segments are
    /// passed separately as comparison-only evidence. They must not supply
    /// missing suffix authority, enter replay, or advance its high-water mark.
    ///
    /// The caller must exclude concurrent writers and the callback must not
    /// modify the source WAL. Replay segments are opened writable so any deferred
    /// repair uses the exact scanned file handle, never a re-resolved pathname.
    /// Retained prefix segments are read-only and never repaired. Corrupt frames and rejected
    /// validation leave it in place, without truncation or quarantine. Normal
    /// [`Self::recover_report`] retains its existing repair/quarantine policy.
    ///
    /// # Errors
    ///
    /// Returns scan, validation or repair errors. No repair is attempted if
    /// scanning or validation fails.
    pub fn recover_validated<T>(
        &mut self,
        validate: impl FnOnce(RecoveryReport<WalRecord>, bool, &[WalRecord]) -> Result<T>,
    ) -> Result<T> {
        self.ready = None;
        self.refuse_if_quarantined()?;
        let checkpoint = self.read_checkpoint_metadata()?;
        let mut max_transaction_id = checkpoint
            .as_ref()
            .and_then(|cp| cp.transaction_id.is_valid().then_some(cp.transaction_id));
        let mut scan = RecoveryScan {
            has_prior_state: checkpoint.is_some(),
            last_commit_epoch: None,
            finished_transactions: std::collections::HashSet::new(),
            torn_tail: None,
            prefix_authority: Vec::new(),
            groups: super::group::Groups::default(),
        };
        let committed = self.recover_internal_as(
            checkpoint.clone(),
            &mut max_transaction_id,
            true,
            &mut scan,
            |record| {
                matches!(
                    record,
                    WalRecord::StoreIdentityMeta { .. } | WalRecord::GraphModelMeta { .. }
                )
            },
        )?;
        let validated = validate(
            RecoveryReport {
                committed,
                max_transaction_id,
            },
            scan.has_prior_state,
            &scan.prefix_authority,
        )?;
        if let Some((file, valid_len, observed_len)) = scan.torn_tail {
            if file.metadata()?.len() != observed_len {
                return Err(Error::Storage(StorageError::Corruption(
                    "WAL tail changed between validation and repair".to_string(),
                )));
            }
            validate_file(&file)?;
            file.set_len(valid_len)?;
            file.sync_all()?;
        }
        self.complete_scan(checkpoint, scan.groups)?;
        Ok(validated)
    }

    /// Recovers committed records of a specific type from all WAL files.
    ///
    /// This is the generic version of [`recover`](Self::recover). Use it
    /// when recovering a WAL that stores a custom record type.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_as<R: WalEntry>(&mut self) -> Result<Vec<R>> {
        Ok(self.recover_report_as()?.committed)
    }

    /// Generic [`recover_report`](Self::recover_report).
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_report_as<R: WalEntry>(&mut self) -> Result<RecoveryReport<R>> {
        self.ready = None;
        self.refuse_if_quarantined()?;
        let checkpoint = self.read_checkpoint_metadata()?;
        let mut max_transaction_id = checkpoint
            .as_ref()
            .and_then(|cp| cp.transaction_id.is_valid().then_some(cp.transaction_id));
        let mut scan = RecoveryScan::default();
        let committed = self.recover_internal_as::<R>(
            checkpoint.clone(),
            &mut max_transaction_id,
            false,
            &mut scan,
            |_| false,
        )?;
        self.complete_scan(checkpoint, scan.groups)?;
        Ok(RecoveryReport {
            committed,
            max_transaction_id,
        })
    }

    /// Recovers committed records up to and including the given epoch.
    ///
    /// Returns only records belonging to transactions or epoch-framed metadata
    /// publications committed at or before `max_epoch`. Records from the first
    /// transaction or framed publication after `max_epoch` are excluded.
    ///
    /// The WAL commit sequence is: `[data records] [TxCommit] [EpochAdvance]`.
    /// When an `EpochAdvance { epoch }` where `epoch > max_epoch` is seen, we
    /// discard the preceding transaction's records (everything since the last
    /// `EpochAdvance` that was within range). `CatalogBatchV3` and V3 RDF→LPG
    /// declarations carry their own publication epoch and are therefore
    /// included exactly at that boundary. Standalone metadata without an epoch
    /// is retained at the preceding accepted cut.
    ///
    /// Used by point-in-time recovery to restore a database to a specific epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_until_epoch(
        &mut self,
        max_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<WalRecord>> {
        let all_records = self.recover()?;
        let mut committed = Vec::new();
        let mut pending = Vec::new();

        for record in all_records {
            let standalone_epoch = match &record {
                WalRecord::CatalogBatchV3 { epoch, .. }
                | WalRecord::CdcRetention { epoch, .. }
                | WalRecord::RdfLpgProjectionDeclaredV3 { epoch, .. } => Some(*epoch),
                _ => None,
            };
            if let Some(epoch) = standalone_epoch {
                if epoch > max_epoch {
                    break;
                }
                committed.push(record);
                continue;
            }

            let epoch = match &record {
                WalRecord::EpochAdvance { epoch } => Some(*epoch),
                WalRecord::Committed { epoch, .. } | WalRecord::CommittedWithCdc { epoch, .. } => {
                    Some(*epoch)
                }
                _ => None,
            };
            if let Some(epoch) = epoch {
                if epoch > max_epoch {
                    break;
                }
                committed.append(&mut pending);
                committed.push(record);
            } else if record.is_metadata() {
                // Standalone metadata is its own durable publication and does
                // not wait for a later transaction epoch. Legacy records such
                // as CatalogBatchV2 have no epoch of their own, so anchor them
                // to the most recently accepted cut by retaining them
                // immediately; otherwise metadata between an in-range
                // transaction and the first out-of-range transaction (or at
                // the WAL tail) disappears from point-in-time recovery.
                committed.push(record);
            } else {
                pending.push(record);
            }
        }

        // Any remaining non-metadata records lack a confirming EpochAdvance
        // within range, so they belong to a transaction beyond max_epoch (or
        // an incomplete transaction) and are intentionally dropped.

        Ok(committed)
    }

    fn recover_internal_as<R: WalEntry>(
        &self,
        checkpoint: Option<CheckpointMetadata>,
        max_transaction_id: &mut Option<TransactionId>,
        defer_repair: bool,
        scan: &mut RecoveryScan<R>,
        retain_prefix_authority: impl Fn(&R) -> bool,
    ) -> Result<Vec<R>> {
        let mut current_tx_records: Vec<R> = Vec::new();
        let mut committed_records: Vec<R> = Vec::new();

        // Resolve the physical stream before trusting checkpoint metadata to
        // hide any older history. A checkpoint sequence is a durable filename
        // identity, not merely a numeric lower bound: if that exact segment is
        // absent (or ambiguous), replay must fail closed.
        let log_files = self.get_log_files()?;
        Self::validate_checkpoint_boundary(&log_files, checkpoint.as_ref())?;

        // Determine the minimum sequence number to process
        let min_sequence = checkpoint.as_ref().map_or(0, |cp| cp.log_sequence);
        let group_floor = checkpoint.as_ref().map_or(0, |cp| cp.retired_before);

        if checkpoint.is_some() {
            grafeo_info!(
                "Recovering from checkpoint at epoch {:?}, starting from log sequence {}",
                checkpoint.as_ref().map(|c| c.epoch),
                min_sequence
            );
        }

        // Read log files in sequence, skipping those before checkpoint
        let n_files = log_files.len();
        for (idx, (sequence, log_file)) in log_files.into_iter().enumerate() {
            let is_last_file = idx + 1 == n_files;
            if sequence >= group_floor {
                scan.groups.observe_segment(sequence)?;
            }
            if defer_repair {
                scan.has_prior_state |= std::fs::metadata(&log_file)?.len() != 0;
            }

            // Skip files that are completely before the checkpoint
            // We include the checkpoint sequence file because it may contain
            // records after the checkpoint record itself
            if sequence < min_sequence && !defer_repair {
                self.validate_segment_for_append(
                    &log_file,
                    &mut scan.groups,
                    sequence >= group_floor,
                )?;
                grafeo_debug!(
                    "Skipping log file {:?} (sequence {} < checkpoint {})",
                    log_file,
                    sequence,
                    min_sequence
                );
                continue;
            }

            let opened = checked_file(&log_file, sequence >= min_sequence, false);
            let file = match opened {
                Ok(f) => f,
                Err(Error::Io(e))
                    if e.kind() == std::io::ErrorKind::NotFound && checkpoint.is_some() =>
                {
                    return Err(Self::invalid_checkpoint_boundary(
                        min_sequence,
                        format_args!(
                            "replay segment {sequence} disappeared before it could be opened"
                        ),
                    ));
                }
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            let mut reader = WalReader::new(file)?;

            // Read all records from this file
            let mut last_good_offset = 0u64;
            loop {
                let frame_start = reader.stream_position()?;
                match self.read_frame_as::<R>(
                    &mut reader,
                    &mut scan.groups,
                    sequence >= group_floor,
                ) {
                    Ok(FrameRead::Record(record)) => {
                        scan.has_prior_state = true;
                        last_good_offset = reader.stream_position()?;
                        if sequence < min_sequence {
                            if retain_prefix_authority(&record) {
                                scan.prefix_authority.push(record);
                            }
                            continue;
                        }
                        if let Some(tid) = record.transaction_id()
                            && tid.is_valid()
                        {
                            *max_transaction_id = Some(match *max_transaction_id {
                                Some(m) if m >= tid => m,
                                _ => tid,
                            });
                        }
                        if record.feed_epoch().is_some_and(|epoch| {
                            scan.last_commit_epoch.is_some_and(|last| epoch <= last)
                                || record
                                    .transaction_id()
                                    .is_some_and(|id| scan.finished_transactions.contains(&id))
                        }) {
                            return Err(Self::invalid_wal_entry(
                                &log_file,
                                frame_start,
                                "CDC batch or manifest follows its transaction terminal marker or publication epoch",
                            ));
                        }
                        // IDs may commit out of allocation order. An epoch
                        // alone cannot detect a reused ID claiming a later
                        // epoch after Commit/Abort. SYSTEM remains reusable
                        // for non-feed synthetic publications.
                        if (record.is_commit() || record.is_abort())
                            && let Some(id) = record.transaction_id()
                            && id != TransactionId::SYSTEM
                        {
                            scan.finished_transactions
                                .try_reserve(1)
                                .map_err(|_| Error::Storage(StorageError::Full))?;
                            scan.finished_transactions.insert(id);
                        }
                        if record.is_commit() {
                            record
                                .validate_commit_group(current_tx_records.iter().filter(
                                    |pending| {
                                        record.transaction_id().is_none()
                                            || pending.transaction_id().is_none()
                                            || pending.transaction_id() == record.transaction_id()
                                    },
                                ))
                                .map_err(|reason| {
                                    Self::invalid_wal_entry(&log_file, frame_start, &reason)
                                })?;
                            if let Some(epoch) = record.commit_epoch() {
                                scan.last_commit_epoch = Some(
                                    scan.last_commit_epoch.map_or(epoch, |last| last.max(epoch)),
                                );
                            }
                            if let Some(tid) = record.transaction_id() {
                                let mut kept = Vec::new();
                                for pending in current_tx_records.drain(..) {
                                    match pending.transaction_id() {
                                        Some(id) if id != tid => kept.push(pending),
                                        // Savepoint protocol records exist only
                                        // to shape the pending transaction. They
                                        // must never reach store replay.
                                        _ if pending.savepoint_name().is_some()
                                            || pending.rollback_to_savepoint_name().is_some() => {}
                                        _ => committed_records.push(pending),
                                    }
                                }
                                current_tx_records = kept;
                            } else {
                                for pending in current_tx_records.drain(..) {
                                    if pending.savepoint_name().is_none()
                                        && pending.rollback_to_savepoint_name().is_none()
                                    {
                                        committed_records.push(pending);
                                    }
                                }
                            }
                            committed_records.push(record);
                        } else if record.is_abort() {
                            if let Some(tid) = record.transaction_id() {
                                // Tagged records of this tx, and untagged sequential
                                // records in the same window, belong to the aborted
                                // transaction. Other tagged tids stay pending.
                                current_tx_records.retain(|pending| {
                                    matches!(pending.transaction_id(), Some(id) if id != tid)
                                });
                            } else {
                                current_tx_records.clear();
                            }
                        } else if let Some(savepoint_name) = record.rollback_to_savepoint_name() {
                            let transaction_id = record.transaction_id().ok_or_else(|| {
                                Error::Storage(StorageError::InvalidWalEntry(format!(
                                    "rollback-to-savepoint marker {savepoint_name:?} has no transaction id"
                                )))
                            })?;
                            let savepoint_position = current_tx_records
                                .iter()
                                .rposition(|pending| {
                                    pending.transaction_id() == Some(transaction_id)
                                        && pending.savepoint_name() == Some(savepoint_name)
                                })
                                .ok_or_else(|| {
                                    Error::Storage(StorageError::InvalidWalEntry(format!(
                                        "transaction {} rolls back to unknown savepoint {savepoint_name:?}",
                                        transaction_id.as_u64()
                                    )))
                                })?;

                            // The WAL is shared, so records from other
                            // transactions may sit physically between this
                            // transaction's savepoint and rollback marker.
                            // Remove only the matching transaction's logical
                            // tail. Retain the target marker: SQL savepoints
                            // remain active after ROLLBACK TO and may be reused.
                            let mut position = 0usize;
                            current_tx_records.retain(|pending| {
                                let keep = position <= savepoint_position
                                    || pending.transaction_id() != Some(transaction_id);
                                position += 1;
                                keep
                            });
                        } else if record.is_checkpoint() {
                            // A current checkpoint abandons recovered orphaned
                            // groups; live writers refuse checkpoint retirement
                            // until their pending groups finish.
                            current_tx_records.clear();
                            committed_records.push(record);
                        } else if record.is_metadata() {
                            // Checkpoint and metadata records (e.g. EpochAdvance)
                            // are not part of any transaction: always include them.
                            committed_records.push(record);
                        } else {
                            current_tx_records.push(record);
                        }
                    }
                    Ok(FrameRead::Eof) => break,
                    Ok(FrameRead::Incomplete) => {
                        if defer_repair && sequence < min_sequence {
                            return Err(Self::handle_corrupt_frame(
                                &log_file,
                                frame_start,
                                "incomplete retained WAL prefix",
                                true,
                            ));
                        }
                        let observed_len = reader.get_ref().metadata()?.len();
                        scan.has_prior_state = true;
                        if is_last_file && defer_repair {
                            scan.torn_tail =
                                Some((reader.into_inner(), last_good_offset, observed_len));
                            break;
                        }
                        if is_last_file {
                            // Crash tore the last frame. Truncate it and keep the prefix.
                            let file = reader.into_inner();
                            validate_file(&file)?;
                            if file.metadata()?.len() != observed_len {
                                return Err(Error::Storage(StorageError::Corruption(
                                    "WAL tail changed before repair".into(),
                                )));
                            }
                            file.set_len(last_good_offset)?;
                            file.sync_all()?;
                            grafeo_info!(
                                "truncated incomplete WAL tail in {:?} to offset {}",
                                log_file,
                                last_good_offset
                            );
                            break;
                        }
                        return Err(Self::handle_corrupt_frame(
                            &log_file,
                            frame_start,
                            "incomplete frame in a non-last WAL file",
                            defer_repair,
                        ));
                    }
                    Ok(FrameRead::Corrupt(e)) => {
                        drop(reader);
                        return Err(Self::handle_corrupt_frame(
                            &log_file,
                            frame_start,
                            &format!("{e}"),
                            defer_repair,
                        ));
                    }
                    Ok(FrameRead::InvalidEntry(reason)) => {
                        return Err(Self::invalid_wal_entry(&log_file, frame_start, &reason));
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }

        // Uncommitted records in current_tx_records are discarded

        Ok(committed_records)
    }

    /// Extracts the sequence number from a WAL log file path.
    fn sequence_from_path(path: &Path) -> Option<u64> {
        let digits = path
            .file_name()
            .and_then(|name| name.to_str())?
            .strip_prefix("wal_")?
            .strip_suffix(".log")?;
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        digits.parse().ok()
    }

    fn invalid_checkpoint_boundary(sequence: u64, reason: impl std::fmt::Display) -> Error {
        Error::Storage(StorageError::Corruption(format!(
            "invalid checkpoint metadata {}: log sequence {sequence} cannot select a replay boundary: {reason}",
            CHECKPOINT_METADATA_FILE
        )))
    }

    fn validate_checkpoint_boundary(
        log_files: &[(u64, std::path::PathBuf)],
        checkpoint: Option<&CheckpointMetadata>,
    ) -> Result<()> {
        if let Some(checkpoint) = checkpoint {
            log_files
                .binary_search_by_key(&checkpoint.log_sequence, |(sequence, _)| *sequence)
                .map_err(|_| {
                    Self::invalid_checkpoint_boundary(
                        checkpoint.log_sequence,
                        "the claimed WAL segment is absent",
                    )
                })?;
            let group_index = log_files
                .binary_search_by_key(&checkpoint.retired_before, |(sequence, _)| *sequence)
                .map_err(|_| {
                    Self::invalid_checkpoint_boundary(
                        checkpoint.retired_before,
                        "the retained group boundary is absent",
                    )
                })?;
            let suffix = log_files.get(group_index..).ok_or_else(|| {
                Self::invalid_checkpoint_boundary(
                    checkpoint.log_sequence,
                    "invalid segment boundary",
                )
            })?;
            for pair in suffix.windows(2) {
                if let [(current, _), (next, _)] = pair {
                    let expected = current.checked_add(1).ok_or_else(|| {
                        Self::invalid_checkpoint_boundary(
                            checkpoint.log_sequence,
                            "a segment follows the maximum WAL sequence",
                        )
                    })?;
                    if *next != expected {
                        return Err(Self::invalid_checkpoint_boundary(
                            checkpoint.log_sequence,
                            format_args!(
                                "post-boundary WAL sequence {expected} is absent between {current} and {next}"
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn invalid_wal_entry(path: &Path, offset: u64, reason: &str) -> Error {
        Error::Storage(StorageError::InvalidWalEntry(format!(
            "invalid WAL record in {} at byte offset {offset}: {reason}",
            path.display()
        )))
    }

    fn get_log_files(&self) -> Result<Vec<(u64, std::path::PathBuf)>> {
        let mut files = Vec::new();

        if !self.dir.try_exists()? {
            return Ok(files);
        }

        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "log") {
                continue;
            }
            if let Some(sequence) = Self::sequence_from_path(&path) {
                if !entry.file_type()?.is_file() {
                    return Err(Error::Storage(StorageError::Corruption(format!(
                        "WAL segment identity {} does not name a regular file",
                        path.display()
                    ))));
                }
                files.push((sequence, path));
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("wal_"))
            {
                return Err(Error::Storage(StorageError::Corruption(format!(
                    "malformed WAL segment identity {}",
                    path.display()
                ))));
            }
        }

        files.sort_unstable_by(|(left_sequence, left_path), (right_sequence, right_path)| {
            left_sequence
                .cmp(right_sequence)
                .then_with(|| left_path.cmp(right_path))
        });
        for duplicate in files.windows(2) {
            if let [(left_sequence, left_path), (right_sequence, right_path)] = duplicate
                && left_sequence == right_sequence
            {
                return Err(Error::Storage(StorageError::Corruption(format!(
                    "ambiguous WAL sequence {left_sequence}: both {} and {} claim the same numeric identity",
                    left_path.display(),
                    right_path.display()
                ))));
            }
        }

        Ok(files)
    }

    fn refuse_if_quarantined(&self) -> Result<()> {
        if !self.dir.try_exists()? {
            return Ok(());
        }
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "WAL_CORRUPT" || name.ends_with(".corrupt") {
                return Err(Error::Storage(StorageError::Corruption(format!(
                    "WAL remains fail-closed after quarantine ({}); explicit repair required",
                    entry.path().display()
                ))));
            }
        }
        Ok(())
    }

    fn quarantine_wal_file(path: &Path, offset: u64, reason: &str) -> Error {
        let quarantined = {
            let mut p = path.to_path_buf();
            let name = p.file_name().map_or_else(
                || "wal.corrupt".into(),
                |n| format!("{}.corrupt", n.to_string_lossy()),
            );
            p.set_file_name(name);
            p
        };
        let quarantine = (|| -> Result<()> {
            let source = checked_file(path, true, false)?;
            validate_file(&source)?;
            match std::fs::symlink_metadata(&quarantined) {
                Ok(_) => {
                    let _ = checked_file(&quarantined, true, false)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            std::fs::rename(path, &quarantined)?;
            crate::ownership::sync_parent(&quarantined)?;
            Ok(())
        })();
        match quarantine {
            Ok(()) => Error::Storage(StorageError::Corruption(format!(
                "WAL corruption in {} at offset {offset}; quarantined to {} ({reason})",
                path.display(),
                quarantined.display()
            ))),
            Err(error) => {
                let marker_result = (|| -> Result<()> {
                    use std::io::Write;
                    let parent = path.parent().ok_or_else(|| {
                        std::io::Error::other("WAL corruption path has no parent")
                    })?;
                    let marker = parent.join("WAL_CORRUPT");
                    let mut file = checked_file(&marker, true, true)?;
                    validate_file(&file)?;
                    file.set_len(0)?;
                    file.write_all(reason.as_bytes())?;
                    file.sync_all()?;
                    crate::ownership::sync_parent(&marker)
                })();
                match marker_result {
                    Ok(()) => error.with_context(format!(
                        "WAL corruption in {} at offset {offset} ({reason}); quarantine failed, fail-closed marker recorded",
                        path.display()
                    )),
                    Err(marker_error) => marker_error.with_context(format!(
                        "WAL corruption in {} at offset {offset} ({reason}); quarantine failed ({error}) and marker could not be recorded",
                        path.display()
                    )),
                }
            }
        }
    }

    fn handle_corrupt_frame(path: &Path, offset: u64, reason: &str, defer_repair: bool) -> Error {
        if defer_repair {
            Error::Storage(StorageError::Corruption(format!(
                "WAL corruption in {} at offset {offset}: {reason}",
                path.display(),
            )))
        } else {
            Self::quarantine_wal_file(path, offset, reason)
        }
    }

    fn read_frame_payload(&self, reader: &mut WalReader) -> Result<FrameRead<Vec<u8>>> {
        let file_len = reader.get_ref().metadata()?.len();
        read_frame_payload(
            reader,
            file_len,
            #[cfg(feature = "encryption")]
            self.encryptor.as_ref(),
        )
    }

    fn read_frame_as<R: WalEntry>(
        &self,
        reader: &mut WalReader,
        groups: &mut super::group::Groups,
        verify_group: bool,
    ) -> Result<FrameRead<R>> {
        let data = match self.read_frame_payload(reader)? {
            FrameRead::Record(data) => data,
            FrameRead::Eof => return Ok(FrameRead::Eof),
            FrameRead::Incomplete => return Ok(FrameRead::Incomplete),
            FrameRead::Corrupt(error) => return Ok(FrameRead::Corrupt(error)),
            FrameRead::InvalidEntry(reason) => return Ok(FrameRead::InvalidEntry(reason)),
        };
        let frame = match super::group::envelope(&data) {
            Ok(frame) => frame,
            Err(reason) => return Ok(FrameRead::InvalidEntry(reason)),
        };
        match bincode::serde::decode_from_slice::<R, _>(
            frame.record,
            bincode::config::standard().with_limit::<MAX_WAL_FRAME_BYTES>(),
        ) {
            Ok((record, consumed)) if consumed == frame.record.len() => {
                if let Err(reason) = record.validate_recovery() {
                    return Ok(FrameRead::InvalidEntry(reason));
                }
                if super::group::Coordinate::of(&record)? != frame.coordinate {
                    return Ok(FrameRead::InvalidEntry(
                        "WAL group coordinate differs from typed record".into(),
                    ));
                }
                if verify_group {
                    let prepared = groups.prepare_envelope(&frame, false)?;
                    groups.apply(prepared, &data);
                }
                Ok(FrameRead::Record(record))
            }
            Ok((_, consumed)) => Ok(FrameRead::InvalidEntry(format!(
                "decoded {consumed} of {} payload bytes",
                frame.record.len()
            ))),
            Err(error) => Ok(FrameRead::InvalidEntry(format!(
                "WAL record decode failed: {error}"
            ))),
        }
    }
}

fn read_exact_or_incomplete(reader: &mut impl Read, buf: &mut [u8]) -> Result<bool> {
    match reader.read_exact(buf) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e.into()),
    }
}

fn read_frame_payload(
    reader: &mut (impl Read + Seek),
    file_len: u64,
    #[cfg(feature = "encryption")] encryptor: Option<&grafeo_common::encryption::PageEncryptor>,
) -> Result<FrameRead<Vec<u8>>> {
    let start = reader.stream_position()?;
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            let now = reader.stream_position()?;
            return Ok(if now == start {
                FrameRead::Eof
            } else {
                FrameRead::Incomplete
            });
        }
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_WAL_FRAME_BYTES {
        return Ok(FrameRead::Corrupt(Error::Storage(
            StorageError::Corruption(format!(
                "WAL frame length {len} exceeds {MAX_WAL_FRAME_BYTES}"
            )),
        )));
    }
    let pos = reader.stream_position()?;
    let remaining = file_len.saturating_sub(pos);
    // payload + crc32; encrypted frames have no trailing crc
    #[cfg(not(feature = "encryption"))]
    let need = len as u64 + 4;
    #[cfg(feature = "encryption")]
    let need = if encryptor.is_some() {
        len as u64
    } else {
        len as u64 + 4
    };
    if remaining < need {
        return Ok(FrameRead::Incomplete);
    }

    #[cfg(feature = "encryption")]
    let data = if let Some(enc) = encryptor {
        let mut encrypted = vec![0u8; len];
        if !read_exact_or_incomplete(reader, &mut encrypted)? {
            return Ok(FrameRead::Incomplete);
        }
        let aad = b"grafeo-wal";
        match enc.decrypt(&encrypted, aad) {
            Ok(plain) => plain,
            Err(_) => {
                return Ok(FrameRead::Corrupt(Error::Storage(
                    StorageError::Corruption(
                        "WAL decryption failed: wrong key or corrupted record".to_string(),
                    ),
                )));
            }
        }
    } else {
        let mut data = vec![0u8; len];
        if !read_exact_or_incomplete(reader, &mut data)? {
            return Ok(FrameRead::Incomplete);
        }
        let mut checksum_buf = [0u8; 4];
        if !read_exact_or_incomplete(reader, &mut checksum_buf)? {
            return Ok(FrameRead::Incomplete);
        }
        let stored_checksum = u32::from_le_bytes(checksum_buf);
        let computed_checksum = crc32fast::hash(&data);
        if stored_checksum != computed_checksum {
            return Ok(FrameRead::Corrupt(Error::Storage(
                StorageError::Corruption("WAL checksum mismatch".to_string()),
            )));
        }
        data
    };

    #[cfg(not(feature = "encryption"))]
    let data = {
        let mut data = vec![0u8; len];
        if !read_exact_or_incomplete(reader, &mut data)? {
            return Ok(FrameRead::Incomplete);
        }
        let mut checksum_buf = [0u8; 4];
        if !read_exact_or_incomplete(reader, &mut checksum_buf)? {
            return Ok(FrameRead::Incomplete);
        }
        let stored_checksum = u32::from_le_bytes(checksum_buf);
        let computed_checksum = crc32fast::hash(&data);
        if stored_checksum != computed_checksum {
            return Ok(FrameRead::Corrupt(Error::Storage(
                StorageError::Corruption("WAL checksum mismatch".to_string()),
            )));
        }
        data
    };

    match super::frame_buffer::record_bytes(&data) {
        Ok(_) => Ok(FrameRead::Record(data)),
        Err(reason) => Ok(FrameRead::InvalidEntry(reason)),
    }
}

/// Counts complete physical frames in plaintext WAL bytes.
///
/// Includes transaction markers and uncommitted records; does not replay or
/// decode typed entries. Validates the shared checksum and record envelope.
///
/// # Errors
/// Rejects incomplete, oversized, corrupt, or unsupported-generation frames.
pub fn count_wal_frames(bytes: &[u8]) -> Result<u64> {
    count_frames(
        bytes,
        #[cfg(feature = "encryption")]
        None,
    )
}

pub(super) fn count_frames(
    bytes: &[u8],
    #[cfg(feature = "encryption")] encryptor: Option<&grafeo_common::encryption::PageEncryptor>,
) -> Result<u64> {
    let mut reader = std::io::Cursor::new(bytes);
    let mut count = 0_u64;
    loop {
        let offset = reader.position();
        match read_frame_payload(
            &mut reader,
            bytes.len() as u64,
            #[cfg(feature = "encryption")]
            encryptor,
        )? {
            FrameRead::Record(_) => {
                count = count.checked_add(1).ok_or_else(|| {
                    Error::Storage(StorageError::Corruption("WAL frame count overflow".into()))
                })?;
            }
            FrameRead::Eof => return Ok(count),
            FrameRead::Incomplete => {
                return Err(Error::Storage(StorageError::Corruption(format!(
                    "incomplete WAL frame at byte offset {offset}"
                ))));
            }
            FrameRead::Corrupt(error) => {
                return Err(error.with_context(format!("WAL frame at byte offset {offset}")));
            }
            FrameRead::InvalidEntry(reason) => {
                return Err(Error::Storage(StorageError::Corruption(format!(
                    "invalid WAL envelope at byte offset {offset}: {reason}"
                ))));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_wal_dir as tempdir;
    use super::*;
    use grafeo_common::types::{EpochId, NodeId, TransactionId};
    use grafeo_common::utils::error::ErrorCode;

    #[test]
    fn committed_group_rejects_checksum_valid_missing_duplicate_reordered_or_altered_frames() {
        for mutation in [
            "missing",
            "duplicate",
            "reordered",
            "altered",
            "foreign",
            "epoch",
        ] {
            let dir = tempdir().unwrap();
            let wal = WalManager::open(dir.path()).unwrap();
            for id in 1..=3 {
                wal.log(&WalRecord::lpg(
                    TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    super::super::LpgMutationOp::CreateNode {
                        id: NodeId::new(id),
                        labels: vec![format!("Node{id}")],
                    },
                ))
                .unwrap();
            }
            wal.log(&WalRecord::Committed {
                transaction_id: TransactionId::new(1),
                epoch: EpochId::new(1),
            })
            .unwrap();
            let paths = wal.log_files().unwrap();
            assert_eq!(paths.len(), 1);
            wal.close().unwrap();
            drop(wal);
            let original = std::fs::read(&paths[0]).unwrap();
            let mut frames = Vec::new();
            let mut offset = 0;
            while offset < original.len() {
                let len =
                    u32::from_le_bytes(original[offset..offset + 4].try_into().unwrap()) as usize;
                let end = offset + 4 + len + 4;
                frames.push(original[offset..end].to_vec());
                offset = end;
            }
            assert_eq!(frames.len(), 4);
            match mutation {
                "missing" => {
                    frames.remove(1);
                }
                "duplicate" => frames.insert(1, frames[0].clone()),
                "reordered" => frames.swap(0, 1),
                "foreign" => frames[0] = encoded_frame(&group_node(42, 1)),
                "epoch" => {
                    let index = frames.len() - 1;
                    let original = &frames[index];
                    let mut data =
                        super::super::frame_buffer::encode_frame(&WalRecord::Committed {
                            transaction_id: TransactionId::new(1),
                            epoch: EpochId::new(2),
                        })
                        .unwrap();
                    let at = data.len() - 40;
                    data[at..].copy_from_slice(&original[original.len() - 44..original.len() - 4]);
                    let mut frame = u32::try_from(data.len()).unwrap().to_le_bytes().to_vec();
                    frame.extend_from_slice(&data);
                    frame.extend_from_slice(&crc32fast::hash(&data).to_le_bytes());
                    frames[index] = frame;
                }
                "altered" => {
                    let frame = &mut frames[1];
                    let at = frame
                        .windows(5)
                        .position(|bytes| bytes == b"Node2")
                        .unwrap();
                    frame[at + 4] = b'9';
                    let end = frame.len() - 4;
                    let crc = crc32fast::hash(&frame[4..end]);
                    frame[end..].copy_from_slice(&crc.to_le_bytes());
                }
                _ => unreachable!(),
            }
            let damaged = frames.concat();
            std::fs::write(&paths[0], &damaged).unwrap();
            let mut applied = false;
            let result = WalRecovery::new(dir.path())
                .unwrap()
                .recover_validated(|_, _, _| {
                    applied = true;
                    Ok(())
                });
            assert!(result.is_err(), "{mutation} committed frame was admitted");
            assert!(!applied, "{mutation} reached the application callback");
            assert_eq!(std::fs::read(&paths[0]).unwrap(), damaged);
        }
    }

    fn group_node(transaction: u64, id: u64) -> WalRecord {
        WalRecord::lpg(
            TransactionId::new(transaction),
            grafeo_common::types::GraphPath::root(),
            super::super::LpgMutationOp::CreateNode {
                id: NodeId::new(id),
                labels: vec![format!("Node{id}")],
            },
        )
    }

    #[test]
    fn committed_group_binds_order_across_custom_transaction_coordinates() {
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        struct Entry {
            transaction: Option<TransactionId>,
            kind: u8,
            value: u64,
        }
        impl WalEntry for Entry {
            fn requires_sync(&self) -> bool {
                self.is_commit()
            }
            fn is_commit(&self) -> bool {
                self.kind == 1
            }
            fn is_abort(&self) -> bool {
                false
            }
            fn is_checkpoint(&self) -> bool {
                self.kind == 2
            }
            fn transaction_id(&self) -> Option<TransactionId> {
                self.transaction
            }
            fn make_checkpoint(transaction_id: TransactionId) -> Self {
                Self {
                    transaction: Some(transaction_id),
                    kind: 2,
                    value: 0,
                }
            }
        }
        for (second, marker) in [(None, Some(7)), (None, None), (Some(8), None)] {
            let dir = tempdir().unwrap();
            let wal = super::super::TypedWal::<Entry>::open(dir.path()).unwrap();
            for (transaction, kind, value) in [(Some(7), 0, 1), (second, 0, 2), (marker, 1, 0)] {
                wal.log(&Entry {
                    transaction: transaction.map(TransactionId::new),
                    kind,
                    value,
                })
                .unwrap();
            }
            let path = wal.log_files().unwrap().pop().unwrap();
            wal.close().unwrap();
            drop(wal);
            let records = WalRecovery::new(dir.path())
                .unwrap()
                .recover_as::<Entry>()
                .unwrap();
            assert_eq!(
                records.iter().map(|entry| entry.value).collect::<Vec<_>>(),
                [1, 2, 0]
            );
            let original = std::fs::read(&path).unwrap();
            let mut frames = Vec::new();
            let mut offset = 0;
            while offset < original.len() {
                let length =
                    u32::from_le_bytes(original[offset..offset + 4].try_into().unwrap()) as usize;
                let end = offset + 4 + length + 4;
                frames.push(original[offset..end].to_vec());
                offset = end;
            }
            assert_eq!(frames.len(), 3);
            frames.swap(0, 1);
            let swapped = frames.concat();
            std::fs::write(&path, &swapped).unwrap();
            let mut recovery = WalRecovery::new(dir.path()).unwrap();
            let error = recovery
                .recover_as::<Entry>()
                .expect_err("cross-coordinate record reordering was admitted");
            assert!(matches!(
                error,
                Error::Storage(StorageError::InvalidWalEntry(_))
            ));
            drop(recovery);
            assert!(matches!(
                WalManager::open(dir.path()),
                Err(Error::Storage(StorageError::InvalidWalEntry(_)))
            ));
            assert_eq!(std::fs::read(&path).unwrap(), swapped);
        }
    }

    #[test]
    fn committed_group_capacity_refusal_preserves_writer_and_recovery() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        for transaction in 2..65_538 {
            wal.log(&group_node(transaction, transaction)).unwrap();
        }
        let rejected = group_node(65_538, 65_538);
        let count = wal.record_count();
        let path = wal.log_files().unwrap().pop().unwrap();
        let before = std::fs::read(&path).unwrap();
        assert!(wal.log(&rejected).is_err());
        assert_eq!(wal.record_count(), count);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        wal.log(&WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(2),
        })
        .unwrap();
        wal.log(&rejected).unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: TransactionId::new(65_538),
            epoch: EpochId::new(1),
        })
        .unwrap();
        wal.close().unwrap();
        drop(wal);
        let recovered = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(matches!(
            recovered.last(),
            Some(WalRecord::Committed { transaction_id, .. })
                if *transaction_id == TransactionId::new(65_538)
        ));
    }

    #[test]
    fn committed_group_checkpoint_refuses_live_groups_and_recovers_retired_partial_prefix() {
        let dir = tempdir().unwrap();
        let wal = WalManager::with_config(
            dir.path(),
            WalConfig {
                max_log_size: 1,
                ..WalConfig::default()
            },
        )
        .unwrap();
        for id in 1..=3 {
            wal.log(&group_node(41, id)).unwrap();
        }
        let before: Vec<_> = wal
            .log_files()
            .unwrap()
            .into_iter()
            .map(|path| {
                let bytes = std::fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        let count = wal.record_count();
        assert!(
            wal.checkpoint(TransactionId::new(41), EpochId::new(1))
                .is_err()
        );
        assert_eq!(wal.record_count(), count);
        for (path, bytes) in &before {
            assert_eq!(std::fs::read(path).unwrap(), *bytes);
        }
        wal.log(&WalRecord::Committed {
            transaction_id: TransactionId::new(41),
            epoch: EpochId::new(1),
        })
        .unwrap();
        wal.checkpoint(TransactionId::new(41), EpochId::new(1))
            .unwrap();
        let metadata = wal.read_checkpoint_metadata().unwrap().unwrap();
        assert_eq!(metadata.retired_before, metadata.log_sequence);
        assert!(!before[0].0.exists());
        assert!(wal.log_files().unwrap().len() >= 2);
        wal.close().unwrap();
        drop(wal);
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        assert!(recovery.recover().unwrap().is_empty());
        let writer = recovery.into_wal(WalConfig::default()).unwrap();
        writer.log(&group_node(42, 4)).unwrap();
        writer
            .log(&WalRecord::Committed {
                transaction_id: TransactionId::new(42),
                epoch: EpochId::new(2),
            })
            .unwrap();
        writer.close().unwrap();
        drop(writer);
        assert_eq!(
            WalRecovery::new(dir.path())
                .unwrap()
                .recover()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn committed_group_append_handoff_preserves_or_checkpoints_orphaned_groups() {
        for finish in [true, false] {
            let dir = tempdir().unwrap();
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&group_node(41, 1)).unwrap();
            wal.close().unwrap();
            drop(wal);
            let wal = WalManager::open(dir.path()).unwrap();
            let transaction = if finish {
                41
            } else {
                wal.checkpoint(TransactionId::new(41), EpochId::INITIAL)
                    .unwrap();
                42
            };
            wal.log(&group_node(transaction, 2)).unwrap();
            wal.log(&WalRecord::Committed {
                transaction_id: TransactionId::new(transaction),
                epoch: EpochId::new(1),
            })
            .unwrap();
            wal.close().unwrap();
            drop(wal);
            let recovered = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
            assert_eq!(recovered.len(), if finish { 3 } else { 2 });
            assert!(recovered.iter().any(|record| matches!(record, WalRecord::Committed { transaction_id, .. } if *transaction_id == TransactionId::new(transaction))));
        }
    }

    #[test]
    fn committed_group_lease_preserves_precheckpoint_verification_before_apply_and_append() {
        let dir = tempdir().unwrap();
        let wal = WalManager::open(dir.path()).unwrap();
        let lease = wal.retain_from(0).unwrap();
        wal.log(&group_node(41, 1)).unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: TransactionId::new(41),
            epoch: EpochId::new(1),
        })
        .unwrap();
        wal.checkpoint(TransactionId::new(41), EpochId::new(1))
            .unwrap();
        assert_eq!(
            wal.read_checkpoint_metadata()
                .unwrap()
                .unwrap()
                .retired_before,
            0
        );
        wal.close().unwrap();
        drop(wal);
        drop(lease);
        assert!(
            WalRecovery::new(dir.path())
                .unwrap()
                .recover()
                .unwrap()
                .is_empty()
        );
        let path = dir.path().join("wal_00000000.log");
        let mut damaged = std::fs::read(&path).unwrap();
        let length = u32::from_le_bytes(damaged[..4].try_into().unwrap()) as usize;
        let at = damaged[..length + 4]
            .windows(5)
            .position(|bytes| bytes == b"Node1")
            .unwrap();
        damaged[at + 4] = b'9';
        let crc = crc32fast::hash(&damaged[4..length + 4]);
        damaged[length + 4..length + 8].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &damaged).unwrap();
        let mut applied = false;
        assert!(
            WalRecovery::new(dir.path())
                .unwrap()
                .recover_validated(|_, _, _| {
                    applied = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!applied);
        assert!(WalManager::open(dir.path()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), damaged);
    }

    #[test]
    fn committed_group_lease_inside_transaction_rounds_back_until_a_newer_safe_lease() {
        let dir = tempdir().unwrap();
        let wal = WalManager::with_config(
            dir.path(),
            WalConfig {
                max_log_size: 1,
                ..WalConfig::default()
            },
        )
        .unwrap();
        wal.log(&group_node(41, 1)).unwrap();
        let inside = wal.retain_from(wal.current_sequence()).unwrap();
        wal.log(&group_node(41, 2)).unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: TransactionId::new(41),
            epoch: EpochId::new(1),
        })
        .unwrap();
        let safe = wal.current_sequence();
        let newer = wal.retain_from(safe).unwrap();
        wal.checkpoint(TransactionId::new(41), EpochId::new(1))
            .unwrap();
        assert_eq!(
            wal.read_checkpoint_metadata()
                .unwrap()
                .unwrap()
                .retired_before,
            0
        );
        assert!(dir.path().join("wal_00000000.log").exists());
        drop(inside);
        wal.checkpoint(TransactionId::new(41), EpochId::new(1))
            .unwrap();
        assert_eq!(
            wal.read_checkpoint_metadata()
                .unwrap()
                .unwrap()
                .retired_before,
            safe
        );
        assert!(!dir.path().join("wal_00000000.log").exists());
        wal.close().unwrap();
        drop(wal);
        drop(newer);
        assert!(
            WalRecovery::new(dir.path())
                .unwrap()
                .recover()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn physical_frame_count_includes_markers_and_batch_once() {
        let frames = [
            encoded_frame(&WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(1),
            }),
            encoded_frame(&WalRecord::CatalogBatchV3 {
                created_graph_incarnations: vec![],
                dropped_graph_incarnations: vec![],
                version: 2,
                epoch: EpochId::new(1),
                catalog_state: vec![1, 2],
                created_graphs: vec![],
                dropped_graphs: vec![],
            }),
            encoded_frame(&WalRecord::Committed {
                transaction_id: TransactionId::new(2),
                epoch: EpochId::new(2),
            }),
        ];
        assert_eq!(count_wal_frames(&[]).unwrap(), 0);
        assert_eq!(count_wal_frames(&frames.concat()).unwrap(), 3);
        for frame in &frames {
            assert_eq!(count_wal_frames(frame).unwrap(), 1);
        }
    }

    #[test]
    fn physical_frame_count_rejects_every_partial_tail_and_corruption() {
        let frame = encoded_frame(&WalRecord::GraphModelMeta { model: 1 });
        for cut in 1..frame.len() {
            let mut bytes = frame.clone();
            bytes.extend_from_slice(&frame[..cut]);
            assert!(count_wal_frames(&bytes).is_err(), "partial tail {cut}");
        }
        let mut corrupt = frame;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(count_wal_frames(&corrupt).is_err());
        let payload = b"unsupported envelope";
        let unsupported = [
            u32::try_from(payload.len())
                .unwrap()
                .to_le_bytes()
                .as_slice(),
            payload.as_slice(),
            crc32fast::hash(payload).to_le_bytes().as_slice(),
        ]
        .concat();
        assert!(count_wal_frames(&unsupported).is_err());
        assert!(
            count_wal_frames(
                &u32::try_from(MAX_WAL_FRAME_BYTES + 1)
                    .unwrap()
                    .to_le_bytes()
            )
            .is_err()
        );
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn physical_frame_count_authenticates_encrypted_capture() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};
        let dir = tempdir().unwrap();
        let chain = KeyChain::new([42; KEY_SIZE]);
        let wal = WalManager::with_config_and_encryptor(
            dir.path(),
            WalConfig::default(),
            chain.encryptor_for("grafeo-wal", &0_u64.to_be_bytes()),
        )
        .unwrap();
        wal.log(&WalRecord::GraphModelMeta { model: 1 }).unwrap();
        wal.log(&WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(1),
        })
        .unwrap();
        let mut capture = wal.capture().unwrap();
        let segments = capture.segments().unwrap();
        let bytes = capture.read_segment(&segments[0]).unwrap();
        assert_eq!(capture.count_frames(&bytes).unwrap(), 2);
        assert!(count_wal_frames(&bytes).is_err());
        assert!(capture.count_frames(&bytes[..bytes.len() - 1]).is_err());
        let wrong = KeyChain::new([43; KEY_SIZE]);
        let wrong_enc = wrong.encryptor_for("grafeo-wal", &0_u64.to_be_bytes());
        assert!(count_frames(&bytes, Some(&wrong_enc)).is_err());
        let mut corrupt = bytes;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(capture.count_frames(&corrupt).is_err());
    }

    #[test]
    fn validated_recovery_defers_tail_repair_until_acceptance() {
        let dir = tempdir().unwrap();
        let payload = bincode::serde::encode_to_vec(
            WalRecord::GraphModelMeta { model: 0 },
            bincode::config::standard(),
        )
        .unwrap();
        let mut bytes = checksum_valid_frame(&payload);
        let valid_len = bytes.len();
        bytes.extend_from_slice(&[1, 2]);
        let path = dir.path().join("wal_00000000.log");
        std::fs::write(&path, &bytes).unwrap();
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let rejected: Result<()> =
            recovery.recover_validated(|report, has_prior_state, _prefix| {
                assert!(has_prior_state);
                assert_eq!(report.committed.len(), 1);
                assert_eq!(std::fs::read(&path).unwrap(), bytes);
                Err(Error::Serialization("authority rejected".into()))
            });
        assert!(rejected.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let records = recovery
            .recover_validated(|report, has_prior_state, _prefix| {
                assert!(has_prior_state);
                assert_eq!(std::fs::read(&path).unwrap(), bytes);
                Ok(report.committed)
            })
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(std::fs::read(&path).unwrap(), bytes[..valid_len]);
    }

    #[test]
    fn validated_recovery_corruption_is_read_only_and_never_calls_validator() {
        let dir = tempdir().unwrap();
        let payload = bincode::serde::encode_to_vec(
            WalRecord::GraphModelMeta { model: 0 },
            bincode::config::standard(),
        )
        .unwrap();
        let mut bytes = checksum_valid_frame(&payload);
        *bytes.last_mut().unwrap() ^= 1;
        let path = dir.path().join("wal_00000000.log");
        std::fs::write(&path, &bytes).unwrap();
        let mut called = false;
        let result = WalRecovery::new(dir.path())
            .unwrap()
            .recover_validated(|_, _, _prefix| {
                called = true;
                Ok(())
            });
        assert!(result.is_err());
        assert!(!called);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn validated_recovery_empty_allocated_segment_is_fresh() {
        let dir = tempdir().unwrap();
        drop(WalManager::open(dir.path()).unwrap());
        WalRecovery::new(dir.path())
            .unwrap()
            .recover_validated(|report, has_prior_state, _prefix| {
                assert!(!has_prior_state);
                assert!(report.committed.is_empty());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn validated_recovery_prefix_is_comparison_only_not_replay_or_high_water() {
        use grafeo_common::types::{
            GraphIncarnationId, HistoryCompleteness, StoreId, WorldIdentityMetadataV1,
        };
        let dir = tempdir().unwrap();
        let identity = WorldIdentityMetadataV1::new(
            StoreId::generate().unwrap(),
            HistoryCompleteness::Complete,
        )
        .unwrap();
        let records = [
            WalRecord::StoreIdentityMeta { metadata: identity },
            WalRecord::GraphModelMeta { model: 1 },
            WalRecord::InsertRdfQuadV3 {
                subject: "<http://example/prefix>".into(),
                predicate: "<http://example/p>".into(),
                object: "\"old\"".into(),
                graph: None,
                graph_incarnation: GraphIncarnationId::DEFAULT_GRAPH,
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id: TransactionId::new(99),
            },
            WalRecord::Committed {
                transaction_id: TransactionId::new(99),
                epoch: EpochId::new(6),
            },
            WalRecord::EpochAdvance {
                epoch: EpochId::new(6),
            },
        ];
        let prefix: Vec<u8> = records.iter().flat_map(encoded_frame).collect();
        std::fs::write(dir.path().join("wal_00000000.log"), prefix).unwrap();
        let suffix: Vec<u8> = [
            WalRecord::GraphModelMeta { model: 1 },
            WalRecord::Committed {
                transaction_id: TransactionId::new(2),
                epoch: EpochId::new(8),
            },
        ]
        .iter()
        .flat_map(encoded_frame)
        .collect();
        std::fs::write(dir.path().join("wal_00000001.log"), suffix).unwrap();
        write_checkpoint_metadata(dir.path(), 1);
        WalRecovery::new(dir.path()).unwrap().recover_validated(|report, has_prior_state, prefix| {
            assert!(has_prior_state);
            assert_eq!(prefix.len(), 2);
            assert!(matches!(prefix[0], WalRecord::StoreIdentityMeta { .. }));
            assert!(matches!(prefix[1], WalRecord::GraphModelMeta { model: 1 }));
            assert_eq!(report.committed.len(), 2);
            assert!(matches!(report.committed[0], WalRecord::GraphModelMeta { model: 1 }));
            assert!(matches!(report.committed[1], WalRecord::Committed { epoch, .. } if epoch == EpochId::new(8)));
            assert_eq!(report.max_transaction_id, Some(TransactionId::new(2)));
            Ok(())
        }).unwrap();
    }

    #[test]
    fn validated_recovery_rejects_bad_prefix_without_repair_or_quarantine() {
        for incomplete in [false, true] {
            let dir = tempdir().unwrap();
            let mut prefix = encoded_frame(&WalRecord::GraphModelMeta { model: 1 });
            if incomplete {
                prefix = vec![1, 2];
            } else {
                *prefix.last_mut().unwrap() ^= 1;
            }
            let path = dir.path().join("wal_00000000.log");
            std::fs::write(&path, &prefix).unwrap();
            std::fs::write(
                dir.path().join("wal_00000001.log"),
                encoded_frame(&WalRecord::GraphModelMeta { model: 1 }),
            )
            .unwrap();
            write_checkpoint_metadata(dir.path(), 1);
            let mut recovery = WalRecovery::new(dir.path()).unwrap();
            let mut called = false;
            let result = recovery.recover_validated(|_, _, _| {
                called = true;
                Ok(())
            });
            assert!(result.is_err());
            assert!(!called);
            assert_eq!(std::fs::read(&path).unwrap(), prefix);
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 3);
            // Retired record semantics are skipped, never their physical boundary.
            assert!(recovery.recover().is_err());
            assert_eq!(std::fs::read(&path).unwrap(), prefix);
        }
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct ClaimedStringEntry(String);

    impl WalEntry for ClaimedStringEntry {
        fn requires_sync(&self) -> bool {
            false
        }

        fn is_commit(&self) -> bool {
            false
        }

        fn is_abort(&self) -> bool {
            false
        }

        fn is_checkpoint(&self) -> bool {
            false
        }

        fn make_checkpoint(_transaction_id: TransactionId) -> Self {
            Self(String::new())
        }
    }

    fn write_checksum_valid_frame(dir: &Path, payload: &[u8]) -> std::path::PathBuf {
        let path = dir.join("wal_00000000.log");
        std::fs::write(&path, checksum_valid_frame(payload)).unwrap();
        path
    }

    fn checksum_valid_frame(payload: &[u8]) -> Vec<u8> {
        // Preserve hostile body bytes. A bounded best-effort decode supplies
        // the native coordinate where possible; malformed bodies still reach
        // the real bounded decoder unchanged.
        let coordinate = bincode::serde::decode_from_slice::<WalRecord, _>(
            payload,
            bincode::config::standard().with_limit::<MAX_WAL_FRAME_BYTES>(),
        )
        .ok()
        .filter(|(_, consumed)| *consumed == payload.len())
        .and_then(|(record, _)| super::super::group::Coordinate::of(&record).ok())
        .unwrap_or_else(super::super::group::Coordinate::untagged_data);
        let mut data = super::super::group::HEADER.to_vec();
        data.extend_from_slice(&coordinate.bytes());
        data.extend_from_slice(payload);
        if coordinate.has_seal() {
            data.extend_from_slice(&[0; 40]);
        }
        framed_group_payload(data, &mut super::super::group::Groups::default())
    }

    fn framed_group_payload(
        mut data: Vec<u8>,
        groups: &mut super::super::group::Groups,
    ) -> Vec<u8> {
        let prepared = groups.prepare(&data, true).unwrap();
        if let Some(seal) = prepared.seal {
            let at = data.len() - 40;
            data[at..].copy_from_slice(&seal);
        }
        groups.apply(prepared, &data);
        let mut frame = u32::try_from(data.len()).unwrap().to_le_bytes().to_vec();
        frame.extend_from_slice(&data);
        frame.extend_from_slice(&crc32fast::hash(&data).to_le_bytes());
        frame
    }

    fn encoded_frames(records: &[WalRecord]) -> Vec<Vec<u8>> {
        let mut groups = super::super::group::Groups::default();
        records
            .iter()
            .map(|record| {
                let data = super::super::frame_buffer::encode_frame(record).unwrap();
                framed_group_payload(data, &mut groups)
            })
            .collect()
    }

    fn encoded_frame(record: &WalRecord) -> Vec<u8> {
        encoded_frames(std::slice::from_ref(record)).remove(0)
    }

    fn write_checkpoint_metadata(dir: &Path, log_sequence: u64) -> std::path::PathBuf {
        let metadata = CheckpointMetadata {
            format_version: 5,
            retired_before: log_sequence,
            epoch: EpochId::new(7),
            log_sequence,
            timestamp_ms: 42,
            transaction_id: TransactionId::new(1),
        };
        let bytes = bincode::serde::encode_to_vec(metadata, bincode::config::standard()).unwrap();
        let path = dir.join(CHECKPOINT_METADATA_FILE);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn nested_catalog_batch_wire(depth: usize) -> Vec<u8> {
        assert!(depth > 0);
        let empty = bincode::serde::encode_to_vec(
            WalRecord::CatalogBatchV2 {
                version: 1,
                records: Vec::new(),
            },
            bincode::config::standard(),
        )
        .unwrap();
        if depth == 1 {
            return empty;
        }
        let wrapped = bincode::serde::encode_to_vec(
            WalRecord::CatalogBatchV2 {
                version: 1,
                records: vec![WalRecord::CatalogBatchV2 {
                    version: 1,
                    records: Vec::new(),
                }],
            },
            bincode::config::standard(),
        )
        .unwrap();
        assert!(wrapped.ends_with(&empty));
        let prefix = &wrapped[..wrapped.len() - empty.len()];
        let mut wire = Vec::with_capacity(prefix.len() * (depth - 1) + empty.len());
        for _ in 1..depth {
            wire.extend_from_slice(prefix);
        }
        wire.extend_from_slice(&empty);
        wire
    }

    fn assert_persistent_invalid_entry(dir: &Path, path: &Path, original: &[u8]) {
        assert_persistent_invalid_entry_as::<WalRecord>(dir, path, original);
    }

    fn assert_persistent_invalid_entry_as<R: WalEntry>(dir: &Path, path: &Path, original: &[u8]) {
        let mut recovery = WalRecovery::new(dir).unwrap();
        for attempt in 1..=2 {
            let error = recovery
                .recover_as::<R>()
                .expect_err("checksum-valid structural invalidity must fail closed");
            assert!(
                matches!(error, Error::Storage(StorageError::InvalidWalEntry(_))),
                "attempt {attempt} returned {error:?}"
            );
            assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
            assert_eq!(std::fs::read(path).unwrap(), original);
            assert!(
                std::fs::read_dir(dir).unwrap().all(|entry| {
                    !entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".corrupt")
                }),
                "checksum-valid invalid input must remain in place for repair"
            );
            assert!(
                !dir.join("WAL_CORRUPT").exists(),
                "checksum-valid invalid input must not create a quarantine marker"
            );
        }
    }

    fn assert_persistent_checkpoint_corruption(
        dir: &Path,
        preserved: &[(std::path::PathBuf, Vec<u8>)],
    ) {
        let mut recovery = WalRecovery::new(dir).unwrap();
        for attempt in 1..=2 {
            let error = recovery
                .recover()
                .expect_err("invalid checkpoint boundary must fail closed");
            assert!(
                matches!(error, Error::Storage(StorageError::Corruption(_))),
                "attempt {attempt} returned {error:?}"
            );
            assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
            for (path, original) in preserved {
                let actual = std::fs::read(path).unwrap();
                assert_eq!(
                    actual.as_slice(),
                    original.as_slice(),
                    "semantic checkpoint corruption must preserve {}",
                    path.display()
                );
            }
            assert!(
                std::fs::read_dir(dir).unwrap().all(|entry| {
                    !entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".corrupt")
                }),
                "semantic checkpoint corruption must remain in place for repair"
            );
            assert!(
                !dir.join("WAL_CORRUPT").exists(),
                "semantic checkpoint corruption must not create a quarantine marker"
            );
        }
    }

    #[test]
    fn existing_non_directory_recovery_path_fails_closed() {
        let dir = tempdir().unwrap();
        let not_a_directory = dir.path().join("wal-path-is-a-file");
        std::fs::write(&not_a_directory, b"not a WAL directory").unwrap();

        let error = WalRecovery::new(&not_a_directory)
            .err()
            .expect("an unreadable/invalid WAL directory must never look like an empty WAL");

        assert_eq!(error.error_code(), ErrorCode::IoError, "got {error}");
    }

    #[test]
    fn test_recovery_committed() {
        let dir = tempdir().unwrap();

        // Write some records
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ))
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();

        assert_eq!(records.len(), 2);
    }

    #[test]
    fn semantic_epoch_corruption_fails_persistently_without_quarantining_valid_frames() {
        let dir = tempdir().unwrap();
        let mutation = bincode::serde::encode_to_vec(
            WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["pending".into()],
                },
            ),
            bincode::config::standard(),
        )
        .unwrap();
        let invalid_commit = bincode::serde::encode_to_vec(
            WalRecord::Committed {
                transaction_id: TransactionId::new(1),
                epoch: EpochId::PENDING,
            },
            bincode::config::standard(),
        )
        .unwrap();
        let log_path = dir.path().join("wal_00000000.log");
        let mut bytes = checksum_valid_frame(&mutation);
        bytes.extend_from_slice(&checksum_valid_frame(&invalid_commit));
        std::fs::write(&log_path, bytes).unwrap();
        let original = std::fs::read(&log_path).unwrap();

        for attempt in 0..2 {
            let error = WalRecovery::new(dir.path())
                .unwrap()
                .recover()
                .expect_err("PENDING commit epoch must fail before group release");
            assert!(
                matches!(error, Error::Storage(StorageError::InvalidWalEntry(_))),
                "attempt {attempt} returned {error:?}"
            );
            assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
            assert_eq!(std::fs::read(&log_path).unwrap(), original);
            assert!(
                std::fs::read_dir(dir.path()).unwrap().all(|entry| {
                    let name = entry.unwrap().file_name();
                    !name.to_string_lossy().ends_with(".corrupt")
                }),
                "semantic validation must leave the checksum-valid segment repairable"
            );
        }
    }

    #[test]
    fn checksum_valid_trailing_payload_is_persistent_invalid_entry() {
        let dir = tempdir().unwrap();
        let mut payload = bincode::serde::encode_to_vec(
            WalRecord::EpochAdvance {
                epoch: EpochId::new(1),
            },
            bincode::config::standard(),
        )
        .unwrap();
        payload.push(0xA5);
        let path = write_checksum_valid_frame(dir.path(), &payload);
        let original = std::fs::read(&path).unwrap();

        assert_persistent_invalid_entry(dir.path(), &path, &original);
    }

    #[test]
    fn checksum_valid_unknown_variant_is_persistent_invalid_entry() {
        let dir = tempdir().unwrap();
        let payload = bincode::serde::encode_to_vec(u32::MAX, bincode::config::standard()).unwrap();
        let path = write_checksum_valid_frame(dir.path(), &payload);
        let original = std::fs::read(&path).unwrap();

        assert_persistent_invalid_entry(dir.path(), &path, &original);
    }

    #[test]
    fn deeply_nested_catalog_batch_is_rejected_without_stack_growth() {
        const CHILD_ENV: &str = "GRAFEO_WAL_DEEP_CATALOG_BATCH_CHILD";
        const TEST_NAME: &str =
            "wal::recovery::tests::deeply_nested_catalog_batch_is_rejected_without_stack_growth";

        if std::env::var_os(CHILD_ENV).is_some() {
            let dir = tempdir().unwrap();
            let payload = nested_catalog_batch_wire(100_000);
            let path = write_checksum_valid_frame(dir.path(), &payload);
            let original = std::fs::read(&path).unwrap();
            assert_persistent_invalid_entry(dir.path(), &path, &original);

            let valid = bincode::serde::encode_to_vec(
                WalRecord::CatalogBatchV2 {
                    version: 1,
                    records: vec![WalRecord::CreateSchema { name: "s".into() }],
                },
                bincode::config::standard(),
            )
            .unwrap();
            std::fs::write(&path, checksum_valid_frame(&valid)).unwrap();
            let recovered = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
            assert!(matches!(
                recovered.as_slice(),
                [WalRecord::CatalogBatchV2 { version: 1, records }]
                    if matches!(records.as_slice(), [WalRecord::CreateSchema { name }] if name == "s")
            ));
            return;
        }

        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(TEST_NAME)
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "deep nested-batch child failed: status={}\nstdout={}\nstderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn oversized_string_claim_is_rejected_by_bounded_decode() {
        let dir = tempdir().unwrap();
        let claimed_len = u64::MAX;
        // A one-field tuple struct containing String starts with the same
        // standard-bincode length integer. No string bytes follow: a bounded
        // decoder must reject the claim without attempting that allocation.
        let payload =
            bincode::serde::encode_to_vec(claimed_len, bincode::config::standard()).unwrap();
        let path = write_checksum_valid_frame(dir.path(), &payload);
        let original = std::fs::read(&path).unwrap();

        let error = WalRecovery::new(dir.path())
            .unwrap()
            .recover_as::<ClaimedStringEntry>()
            .unwrap_err();
        assert!(
            error
                .to_string()
                .to_ascii_lowercase()
                .contains("limitexceeded"),
            "bounded decode must reject the claimed allocation by limit: {error}"
        );
        assert_persistent_invalid_entry_as::<ClaimedStringEntry>(dir.path(), &path, &original);
    }

    #[test]
    fn invalid_transaction_id_cannot_authenticate_lpg_mutation() {
        let dir = tempdir().unwrap();
        let mutation = bincode::serde::encode_to_vec(
            WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["orphan".to_string()],
                },
            ),
            bincode::config::standard(),
        )
        .unwrap();
        let invalid_commit = bincode::serde::encode_to_vec(
            WalRecord::Committed {
                transaction_id: TransactionId::INVALID,
                epoch: EpochId::new(1),
            },
            bincode::config::standard(),
        )
        .unwrap();
        let path = dir.path().join("wal_00000000.log");
        let mut bytes = checksum_valid_frame(&mutation);
        bytes.extend_from_slice(&checksum_valid_frame(&invalid_commit));
        std::fs::write(&path, bytes).unwrap();
        let original = std::fs::read(&path).unwrap();

        assert_persistent_invalid_entry(dir.path(), &path, &original);
    }

    #[test]
    fn synthetic_epoch_zero_commit_remains_available_to_point_in_time_recovery() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["historical-zero".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::Committed {
                transaction_id: TransactionId::new(1),
                epoch: EpochId::INITIAL,
            })
            .unwrap();
            wal.sync().unwrap();
        }

        let records = WalRecovery::new(dir.path())
            .unwrap()
            .recover_until_epoch(EpochId::INITIAL)
            .expect("epoch-zero exact-history group remains compatible");
        assert_eq!(records.len(), 2);
        assert!(matches!(
            records[0],
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::CreateNode { .. },
                ..
            }
        ));
        assert!(matches!(
            records[1],
            WalRecord::Committed { epoch, .. } if epoch == EpochId::INITIAL
        ));
    }

    #[test]
    fn pending_checkpoint_metadata_is_rejected_before_wal_selection() {
        let dir = tempdir().unwrap();
        let metadata = CheckpointMetadata {
            format_version: 5,
            retired_before: 0,
            epoch: EpochId::PENDING,
            log_sequence: 9_999,
            timestamp_ms: 0,
            transaction_id: TransactionId::new(1),
        };
        let bytes = bincode::serde::encode_to_vec(&metadata, bincode::config::standard()).unwrap();
        std::fs::write(dir.path().join(CHECKPOINT_METADATA_FILE), bytes).unwrap();

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let error = recovery
            .read_checkpoint_metadata()
            .expect_err("PENDING checkpoint cannot choose a replay sequence");
        assert!(
            matches!(error, Error::Storage(StorageError::Corruption(_))),
            "{error:?}"
        );
        assert!(
            recovery.recover().is_err(),
            "generic recovery must propagate the same checkpoint corruption"
        );
    }

    #[test]
    fn checkpoint_generation_and_retirement_floor_are_validated_before_selection() {
        for (format_version, retired_before) in [(2, 0), (3, 0), (4, 0), (6, 0), (5, 10)] {
            let dir = tempdir().unwrap();
            let path = dir.path().join(CHECKPOINT_METADATA_FILE);
            let metadata = CheckpointMetadata {
                format_version,
                retired_before,
                epoch: EpochId::new(3),
                log_sequence: 9,
                timestamp_ms: 42,
                transaction_id: TransactionId::new(1),
            };
            let bytes =
                bincode::serde::encode_to_vec(metadata, bincode::config::standard()).unwrap();
            std::fs::write(&path, &bytes).unwrap();
            let mut recovery = WalRecovery::new(dir.path()).unwrap();
            assert!(recovery.read_checkpoint_metadata().is_err());
            assert!(recovery.recover().is_err());
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    fn malformed_checkpoint_remains_fallible_through_all_readers() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(CHECKPOINT_METADATA_FILE);
        let metadata = CheckpointMetadata {
            format_version: 5,
            retired_before: 0,
            epoch: EpochId::new(3),
            log_sequence: 0,
            timestamp_ms: 42,
            transaction_id: TransactionId::new(1),
        };
        let mut bytes =
            bincode::serde::encode_to_vec(metadata, bincode::config::standard()).unwrap();
        bytes.push(0xa5);
        std::fs::write(&path, &bytes).unwrap();

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let error = recovery.read_checkpoint_metadata().unwrap_err();
        assert!(
            matches!(error, Error::Storage(StorageError::Corruption(_))),
            "{error:?}"
        );
        let recovery_error = recovery.recover().unwrap_err();
        assert!(
            matches!(recovery_error, Error::Storage(StorageError::Corruption(_))),
            "{recovery_error:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(!dir.path().join("WAL_CORRUPT").exists());
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".corrupt")
        }));
    }

    #[test]
    fn nonexistent_checkpoint_boundaries_fail_before_older_history_is_skipped() {
        for claimed_sequence in [1, 9] {
            let dir = tempdir().unwrap();
            let old_path = dir.path().join("wal_00000000.log");
            let later_path = dir.path().join("wal_00000002.log");
            let old_bytes = encoded_frames(&[
                WalRecord::lpg(
                    TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(1),
                        labels: vec!["must-not-be-silently-skipped".into()],
                    },
                ),
                WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                },
            ])
            .concat();
            let later_bytes = encoded_frame(&WalRecord::EpochAdvance {
                epoch: EpochId::new(8),
            });
            std::fs::write(&old_path, &old_bytes).unwrap();
            std::fs::write(&later_path, &later_bytes).unwrap();
            let metadata_path = write_checkpoint_metadata(dir.path(), claimed_sequence);
            let metadata_bytes = std::fs::read(&metadata_path).unwrap();

            assert_persistent_checkpoint_corruption(
                dir.path(),
                &[
                    (old_path, old_bytes),
                    (later_path, later_bytes),
                    (metadata_path, metadata_bytes),
                ],
            );
        }
    }

    #[test]
    fn wal_segments_are_replayed_in_numeric_sequence_order_without_metadata() {
        let dir = tempdir().unwrap();
        let mutation = WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["numeric-order".into()],
            },
        );
        let commit = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        };

        let frames = encoded_frames(&[mutation, commit]);
        // Lexicographic order is the reverse of numeric order at the first
        // sequence wider than the writer's minimum eight digits.
        std::fs::write(dir.path().join("wal_99999999.log"), &frames[0]).unwrap();
        std::fs::write(dir.path().join("wal_100000000.log"), &frames[1]).unwrap();

        let records = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert!(matches!(
            records.as_slice(),
            [WalRecord::LpgMutation { op: crate::wal::LpgMutationOp::CreateNode { id, .. }, .. }, WalRecord::TransactionCommit { transaction_id }]
                if *id == NodeId::new(1) && *transaction_id == TransactionId::new(1)
        ));
    }

    #[test]
    fn duplicate_numeric_wal_sequence_aliases_are_ambiguous() {
        let dir = tempdir().unwrap();
        let canonical_path = dir.path().join("wal_00000001.log");
        let alias_path = dir.path().join("wal_1.log");
        let canonical_bytes = encoded_frame(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        });
        let alias_bytes = encoded_frame(&WalRecord::EpochAdvance {
            epoch: EpochId::new(2),
        });
        std::fs::write(&canonical_path, &canonical_bytes).unwrap();
        std::fs::write(&alias_path, &alias_bytes).unwrap();
        let metadata_path = write_checkpoint_metadata(dir.path(), 1);
        let metadata_bytes = std::fs::read(&metadata_path).unwrap();

        assert_persistent_checkpoint_corruption(
            dir.path(),
            &[
                (canonical_path, canonical_bytes),
                (alias_path, alias_bytes),
                (metadata_path, metadata_bytes),
            ],
        );
    }

    #[test]
    fn checkpoint_replay_range_must_be_contiguous_from_its_boundary() {
        let dir = tempdir().unwrap();
        let boundary_path = dir.path().join("wal_00000001.log");
        let later_path = dir.path().join("wal_00000003.log");
        let boundary_bytes = encoded_frame(&WalRecord::EpochAdvance {
            epoch: EpochId::new(7),
        });
        let later_bytes = encoded_frame(&WalRecord::EpochAdvance {
            epoch: EpochId::new(9),
        });
        std::fs::write(&boundary_path, &boundary_bytes).unwrap();
        std::fs::write(&later_path, &later_bytes).unwrap();
        let metadata_path = write_checkpoint_metadata(dir.path(), 1);
        let metadata_bytes = std::fs::read(&metadata_path).unwrap();

        assert_persistent_checkpoint_corruption(
            dir.path(),
            &[
                (boundary_path, boundary_bytes),
                (later_path, later_bytes),
                (metadata_path, metadata_bytes),
            ],
        );
    }

    #[test]
    fn normal_checkpoint_boundary_survives_manager_reopen() {
        let dir = tempdir().unwrap();
        let boundary_sequence;
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["snapshot-covered".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.checkpoint(TransactionId::new(1), EpochId::new(7))
                .unwrap();
            boundary_sequence = wal
                .read_checkpoint_metadata()
                .unwrap()
                .unwrap()
                .log_sequence;
            wal.log(&WalRecord::lpg(
                TransactionId::new(2),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(2),
                    labels: vec!["after-checkpoint".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        assert!(
            dir.path()
                .join(format!("wal_{boundary_sequence:08}.log"))
                .is_file()
        );
        {
            let reopened = WalManager::open(dir.path()).unwrap();
            let metadata = reopened.read_checkpoint_metadata().unwrap().unwrap();
            assert_eq!(metadata.log_sequence, boundary_sequence);
        }

        let records = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert!(matches!(
            records.as_slice(),
            [WalRecord::LpgMutation { op: crate::wal::LpgMutationOp::CreateNode { id, .. }, .. }, WalRecord::TransactionCommit { transaction_id }]
                if *id == NodeId::new(2) && *transaction_id == TransactionId::new(2)
        ));
    }

    #[test]
    fn test_recovery_uncommitted() {
        let dir = tempdir().unwrap();

        // Write some records without commit
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ))
            .unwrap();

            // No commit!
            wal.sync().unwrap();
        }

        // Recover
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();

        // Uncommitted records should be discarded
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn store_identity_metadata_survives_recovery_without_a_transaction() {
        use grafeo_common::types::{HistoryCompleteness, StoreId, WorldIdentityMetadataV1};

        let dir = tempdir().unwrap();
        let store_id = StoreId::from_bytes([0x6b; StoreId::LEN]).unwrap();
        let metadata = WorldIdentityMetadataV1::new(
            store_id,
            HistoryCompleteness::LegacyCurrentState {
                observed_at: EpochId::new(4),
                source_version: 8,
            },
        )
        .unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::StoreIdentityMeta {
                metadata: metadata.clone(),
            })
            .unwrap();
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(9),
                    labels: vec!["uncommitted".into()],
                },
            ))
            .unwrap();
            wal.sync().unwrap();
        }

        let records = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert_eq!(records.len(), 1);
        assert!(matches!(
            &records[0],
            WalRecord::StoreIdentityMeta { metadata: actual } if actual == &metadata
        ));
    }

    #[test]
    fn uncommitted_tid_is_high_water_so_it_cannot_be_reused() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::InsertRdfQuadV3 {
                subject: "<http://ex.org/orphan>".into(),
                predicate: "<http://ex.org/p>".into(),
                object: "\"dead\"".into(),
                graph: None,
                graph_incarnation: grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
            wal.sync().unwrap();
        }
        let report = WalRecovery::new(dir.path())
            .unwrap()
            .recover_report()
            .unwrap();
        assert!(
            report.committed.is_empty(),
            "orphaned write must not be recovered as committed"
        );
        assert_eq!(
            report.max_transaction_id,
            Some(TransactionId::new(2)),
            "crash-orphaned tid must remain in the high-water mark"
        );
    }

    #[test]
    fn test_recovery_multiple_files() {
        let dir = tempdir().unwrap();

        // Write records across multiple files
        {
            let config = super::super::WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First transaction
            for i in 0..5 {
                wal.log(&WalRecord::lpg(
                    TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(i),
                        labels: vec!["Test".to_string()],
                    },
                ))
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Second transaction
            for i in 5..10 {
                wal.log(&WalRecord::lpg(
                    TransactionId::new(2),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(i),
                        labels: vec!["Test".to_string()],
                    },
                ))
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();

        // Should have 10 CreateNode + 2 TransactionCommit
        assert_eq!(records.len(), 12);
    }

    #[test]
    fn test_checkpoint_metadata() {
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records and create a checkpoint
        {
            let wal = WalManager::open(dir.path()).unwrap();

            // First transaction
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

            // Second transaction after checkpoint
            wal.log(&WalRecord::lpg(
                TransactionId::new(2),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(2),
                    labels: vec!["Test".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Verify checkpoint metadata was written
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let checkpoint = recovery.read_checkpoint_metadata().unwrap();
        assert!(checkpoint.is_some(), "Checkpoint metadata should exist");

        let cp = checkpoint.unwrap();
        assert_eq!(cp.epoch.as_u64(), 10);
        assert_eq!(cp.transaction_id.as_u64(), 1);
    }

    #[test]
    fn test_recovery_from_checkpoint() {
        use super::super::WalConfig;
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records across multiple log files with checkpoint
        {
            let config = WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First batch of records (should end up in early log files)
            for i in 0..5 {
                wal.log(&WalRecord::lpg(
                    TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(i),
                        labels: vec!["Before".to_string()],
                    },
                ))
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TransactionId::new(1), EpochId::new(100))
                .unwrap();

            // Second batch after checkpoint
            for i in 100..103 {
                wal.log(&WalRecord::lpg(
                    TransactionId::new(2),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(i),
                        labels: vec!["After".to_string()],
                    },
                ))
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recovery should use checkpoint metadata to skip old files
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();

        // We should get all committed records (checkpoint metadata is used for optimization)
        // The number depends on how many log files were skipped
        assert!(!records.is_empty(), "Should recover some records");
    }

    #[test]
    fn test_recover_as_generic() {
        let dir = tempdir().unwrap();

        // Write records using WalManager
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ))
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover using the generic method
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records: Vec<WalRecord> = recovery.recover_as().unwrap();

        assert_eq!(records.len(), 2);

        // Verify the records are correct via WalEntry trait methods
        assert!(!records[0].is_commit());
        assert!(records[1].is_commit());
    }

    #[test]
    fn test_recovery_truncated_wal_mid_record() {
        let dir = tempdir().unwrap();

        // Write valid records first
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Find the WAL file and append a truncated record (length prefix only, no data)
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty());

        // Append a partial record: just a length prefix, then truncate
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_files[0])
            .unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap(); // length=100 but no data follows

        // Recovery should still return the committed records (best-effort)
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(
            records.len(),
            2,
            "committed records before truncation should be recovered"
        );
    }

    #[test]
    fn test_recovery_corrupted_checksum() {
        let dir = tempdir().unwrap();

        // Write valid records
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["First".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Find the WAL file and corrupt a byte in the data section
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty());

        let mut data = std::fs::read(&wal_files[0]).unwrap();
        // Flip a byte in the middle of the data (after the 4-byte length prefix)
        if data.len() > 8 {
            data[6] ^= 0xFF;
        }
        std::fs::write(&wal_files[0], &data).unwrap();

        // Mid-file checksum failure is quarantined, not silently skipped.
        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let result = recovery.recover();
        let err = result.expect_err("checksum corruption with a later frame must quarantine");
        assert_eq!(
            err.error_code(),
            ErrorCode::StorageCorrupted,
            "first recover must be GRAFEO-S002, got {err}"
        );
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("corrupt"))
            .collect();
        assert!(
            !leftovers.is_empty(),
            "corrupt WAL file must be quarantined beside the original"
        );

        let retry = recovery.recover();
        let retry_err = retry.expect_err("retry must stay fail-closed after quarantine");
        assert_eq!(
            retry_err.error_code(),
            ErrorCode::StorageCorrupted,
            "retry must be GRAFEO-S002, got {retry_err}"
        );
    }

    #[test]
    fn test_recovery_incomplete_tail_truncated() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        let len_before = std::fs::metadata(&wal_files[0]).unwrap().len();
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_files[0])
            .unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap();
        drop(f);

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2);
        let len_after = std::fs::metadata(&wal_files[0]).unwrap().len();
        assert_eq!(
            len_after, len_before,
            "incomplete tail must be truncated back to the last complete frame"
        );
    }

    #[test]
    fn test_recovery_last_frame_checksum_quarantines() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["First".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        let mut data = std::fs::read(&wal_files[0]).unwrap();
        let last = data.len() - 5;
        data[last] ^= 0xFF;
        std::fs::write(&wal_files[0], &data).unwrap();

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let result = recovery.recover();
        let err = result.expect_err("full-frame checksum failure is bitrot, not a torn tail");
        assert_eq!(
            err.error_code(),
            ErrorCode::StorageCorrupted,
            "last-frame checksum must be GRAFEO-S002, got {err}"
        );
        let retry = recovery.recover();
        let retry_err =
            retry.expect_err("retry after last-frame checksum quarantine must stay fail-closed");
        assert_eq!(
            retry_err.error_code(),
            ErrorCode::StorageCorrupted,
            "retry must be GRAFEO-S002, got {retry_err}"
        );
    }

    #[test]
    fn test_recovery_truncated_tail_then_new_commit() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["First".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_files[0])
            .unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap();
        drop(f);

        WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(2),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(2),
                    labels: vec!["Second".to_string()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
            wal.sync().unwrap();
        }
        let records = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert_eq!(
            records.len(),
            4,
            "commit after truncated tail must recover both transactions"
        );
    }

    #[test]
    fn test_recovery_empty_wal_file() {
        let dir = tempdir().unwrap();

        // Create an empty WAL file
        std::fs::write(dir.path().join("wal_00000000.log"), []).unwrap();

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "empty WAL should produce no records");
    }

    // ── recover_until_epoch tests ─────────────────────────────────────

    /// Helper: writes a committed transaction with an EpochAdvance marker,
    /// matching the pattern used by the engine session commit path.
    fn write_tx_with_epoch(wal: &WalManager, node_id: u64, tx_id: u64, epoch: u64) {
        wal.log(&WalRecord::lpg(
            TransactionId::new(tx_id),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(node_id),
                labels: vec![format!("N{node_id}")],
            },
        ))
        .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(tx_id),
        })
        .unwrap();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(epoch),
        })
        .unwrap();
    }

    #[test]
    fn test_recover_until_epoch_includes_target() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 10);
            write_tx_with_epoch(&wal, 3, 3, 15);
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover_until_epoch(EpochId::new(10)).unwrap();

        // Should include tx1 (epoch 5) and tx2 (epoch 10), each producing
        // CreateNode + TxCommit + EpochAdvance = 3 records per tx.
        assert_eq!(records.len(), 6, "should include epochs 5 and 10");

        // Verify no records from epoch 15
        assert!(
            !records.iter().any(
                |r| matches!(r, WalRecord::EpochAdvance { epoch } if *epoch > EpochId::new(10))
            ),
            "should not include EpochAdvance beyond target"
        );
    }

    #[test]
    fn test_recover_until_epoch_excludes_next_epoch() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 6);
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover_until_epoch(EpochId::new(5)).unwrap();

        // Only tx1 (epoch 5): CreateNode + TxCommit + EpochAdvance = 3
        assert_eq!(records.len(), 3, "should exclude epoch 6 transaction");

        // Verify the CreateNode is for node 1 only
        let nodes: Vec<_> = records
            .iter()
            .filter_map(|r| match r {
                WalRecord::LpgMutation {
                    op: crate::wal::LpgMutationOp::CreateNode { id, .. },
                    ..
                } => Some(id.as_u64()),
                _ => None,
            })
            .collect();
        assert_eq!(nodes, vec![1], "only node from epoch 5 should be present");
    }

    #[test]
    fn test_recover_until_epoch_zero_returns_empty() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover_until_epoch(EpochId::new(0)).unwrap();

        // Epoch 5 > 0, so all records should be excluded
        assert!(
            records.is_empty(),
            "no records should be at or below epoch 0"
        );
    }

    #[test]
    fn test_recover_until_epoch_beyond_max_returns_all() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 10);
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover_until_epoch(EpochId::new(999)).unwrap();

        // Both transactions are within range: 6 records total
        assert_eq!(records.len(), 6, "all records should be included");
    }

    #[test]
    fn recover_until_epoch_retains_standalone_metadata_at_the_selected_cut() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            wal.log(&WalRecord::CatalogBatchV2 {
                version: 1,
                records: vec![WalRecord::CreateSchema {
                    name: "knowledge".into(),
                }],
            })
            .unwrap();
            write_tx_with_epoch(&wal, 2, 2, 6);
            wal.sync().unwrap();
        }

        let records = WalRecovery::new(dir.path())
            .unwrap()
            .recover_until_epoch(EpochId::new(5))
            .unwrap();

        assert!(records.iter().any(|record| matches!(
            record,
            WalRecord::CatalogBatchV2 { records, .. }
                if matches!(records.as_slice(), [WalRecord::CreateSchema { name }] if name == "knowledge")
        )));
        assert!(
            !records.iter().any(
                |record| matches!(record, WalRecord::LpgMutation { op: crate::wal::LpgMutationOp::CreateNode { id, .. }, .. } if id.as_u64() == 2)
            )
        );
    }

    #[test]
    fn recover_until_epoch_retains_trailing_standalone_metadata() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            wal.log(&WalRecord::CatalogBatchV2 {
                version: 1,
                records: vec![WalRecord::CreateSchema {
                    name: "tail".into(),
                }],
            })
            .unwrap();
            wal.sync().unwrap();
        }

        let records = WalRecovery::new(dir.path())
            .unwrap()
            .recover_until_epoch(EpochId::new(999))
            .unwrap();
        assert!(records.iter().any(|record| matches!(
            record,
            WalRecord::CatalogBatchV2 { records, .. }
                if matches!(records.as_slice(), [WalRecord::CreateSchema { name }] if name == "tail")
        )));
    }

    #[test]
    fn recover_until_epoch_honors_catalog_batch_v3_tail_epoch() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            wal.log(&WalRecord::CatalogBatchV3 {
                created_graph_incarnations: vec![grafeo_common::types::GraphIncarnationId::new(1)],
                dropped_graph_incarnations: vec![],
                version: 2,
                epoch: EpochId::new(6),
                catalog_state: vec![6],
                created_graphs: vec![
                    grafeo_common::types::GraphPath::from_components(&["knowledge"]).unwrap(),
                ],
                dropped_graphs: Vec::new(),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let before = recovery.recover_until_epoch(EpochId::new(5)).unwrap();
        assert_eq!(before.len(), 3, "epoch 5 should contain only transaction 1");
        assert!(
            !before
                .iter()
                .any(|record| matches!(record, WalRecord::CatalogBatchV3 { .. }))
        );

        for target in [6, 999] {
            let records = recovery.recover_until_epoch(EpochId::new(target)).unwrap();
            assert_eq!(records.len(), 4, "target epoch {target}");
            assert!(records.iter().any(|record| matches!(
                record,
                WalRecord::CatalogBatchV3 {
                    version: 2,
                    epoch,
                    catalog_state,
                    created_graphs,
                    dropped_graphs,
                    created_graph_incarnations,
                    dropped_graph_incarnations,
                } if *epoch == EpochId::new(6)
                    && created_graph_incarnations.as_slice() == [grafeo_common::types::GraphIncarnationId::new(1)]
                    && dropped_graph_incarnations.is_empty()
                    && catalog_state == &[6]
                    && matches!(created_graphs.as_slice(), [graph] if graph.components() == ["knowledge"])
                    && dropped_graphs.is_empty()
            )));
        }
    }

    #[test]
    fn recover_until_epoch_stops_after_catalog_batch_v3_before_next_transaction() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            wal.log(&WalRecord::CatalogBatchV3 {
                created_graph_incarnations: vec![],
                dropped_graph_incarnations: vec![grafeo_common::types::GraphIncarnationId::new(1)],
                version: 2,
                epoch: EpochId::new(6),
                catalog_state: vec![6],
                created_graphs: Vec::new(),
                dropped_graphs: vec![
                    grafeo_common::types::GraphPath::from_components(&["stale"]).unwrap(),
                ],
            })
            .unwrap();
            write_tx_with_epoch(&wal, 7, 7, 7);
            wal.sync().unwrap();
        }

        let records = WalRecovery::new(dir.path())
            .unwrap()
            .recover_until_epoch(EpochId::new(6))
            .unwrap();
        assert_eq!(records.len(), 4, "transaction 7 must remain beyond the cut");
        assert!(records.iter().any(|record| matches!(
            record,
            WalRecord::CatalogBatchV3 { epoch, .. } if *epoch == EpochId::new(6)
        )));
        assert!(
            !records.iter().any(
                |record| matches!(record, WalRecord::LpgMutation { op: crate::wal::LpgMutationOp::CreateNode { id, .. }, .. } if id.as_u64() == 7)
            )
        );
        assert!(!records.iter().any(
            |record| matches!(record, WalRecord::EpochAdvance { epoch } if *epoch == EpochId::new(7))
        ));
    }

    #[test]
    fn projection_v3_receipt_is_visible_only_with_its_owning_commit() {
        let dir = tempdir().unwrap();
        let transaction_id = TransactionId::new(71);
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::RdfLpgProjectionDeclaredV3 {
                projection_id: 17,
                mapping_digest: grafeo_common::types::Digest256::from_bytes([0x44; 32]),
                mapping_format_version: 2,
                source_graph: None,
                type_iri: "http://example.org/Person".into(),
                node_label: "Person".into(),
                epoch: EpochId::new(6),
            })
            .unwrap();
            wal.log(&WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id,
                receipt: b"uncommitted-receipt".to_vec(),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let before_commit = recovery.recover().unwrap();
        assert!(before_commit.iter().any(|record| matches!(
            record,
            WalRecord::RdfLpgProjectionDeclaredV3 { epoch, .. }
                if *epoch == EpochId::new(6)
        )));
        assert!(
            !before_commit
                .iter()
                .any(|record| matches!(record, WalRecord::RdfLpgProjectionPublishedV3 { .. }))
        );

        {
            let wal = recovery.into_wal(WalConfig::default()).unwrap();
            wal.log(&WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(7),
            })
            .unwrap();
        }

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let after_commit = recovery.recover().unwrap();
        assert!(after_commit.iter().any(|record| matches!(
            record,
            WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id: tid,
                receipt,
            } if *tid == transaction_id && receipt == b"uncommitted-receipt"
        )));
        let at_six = recovery.recover_until_epoch(EpochId::new(6)).unwrap();
        assert!(
            !at_six
                .iter()
                .any(|record| matches!(record, WalRecord::RdfLpgProjectionPublishedV3 { .. }))
        );
        let at_seven = recovery.recover_until_epoch(EpochId::new(7)).unwrap();
        assert!(
            at_seven
                .iter()
                .any(|record| matches!(record, WalRecord::RdfLpgProjectionPublishedV3 { .. }))
        );
    }

    #[test]
    fn test_recover_until_epoch_empty_wal() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path()).unwrap();

        let mut recovery = WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover_until_epoch(EpochId::new(100)).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn rollback_to_savepoint_filters_only_own_interleaved_tail() {
        use crate::wal::LpgMutationOp;

        let dir = tempdir().unwrap();
        let tx1 = TransactionId::new(11);
        let tx2 = TransactionId::new(12);
        let wal = WalManager::open(dir.path()).unwrap();

        let create = |transaction_id, id, label: &str| {
            WalRecord::lpg(
                transaction_id,
                grafeo_common::types::GraphPath::root(),
                LpgMutationOp::CreateNode {
                    id: NodeId::new(id),
                    labels: vec![label.into()],
                },
            )
        };
        wal.log(&create(tx1, 1, "Before")).unwrap();
        wal.log(&WalRecord::TransactionSavepoint {
            transaction_id: tx1,
            name: "keep".into(),
        })
        .unwrap();
        wal.log(&create(tx1, 2, "Discard")).unwrap();
        // tx2 is physically interleaved inside tx1's savepoint window.
        wal.log(&create(tx2, 20, "Other")).unwrap();
        wal.log(&WalRecord::TransactionRollbackToSavepoint {
            transaction_id: tx1,
            name: "keep".into(),
        })
        .unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: tx2,
            epoch: EpochId::new(1),
        })
        .unwrap();
        // The target marker remains reusable after the first rollback.
        wal.log(&create(tx1, 3, "DiscardAgain")).unwrap();
        wal.log(&WalRecord::TransactionRollbackToSavepoint {
            transaction_id: tx1,
            name: "keep".into(),
        })
        .unwrap();
        wal.log(&create(tx1, 4, "After")).unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: tx1,
            epoch: EpochId::new(2),
        })
        .unwrap();
        drop(wal);

        let records = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        let recovered_ids: Vec<NodeId> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::LpgMutation {
                    op: LpgMutationOp::CreateNode { id, .. },
                    ..
                } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(
            recovered_ids,
            vec![NodeId::new(20), NodeId::new(1), NodeId::new(4)]
        );
        assert!(records.iter().all(|record| !matches!(
            record,
            WalRecord::TransactionSavepoint { .. }
                | WalRecord::TransactionRollbackToSavepoint { .. }
        )));
    }

    #[test]
    fn rollback_to_unknown_savepoint_fails_recovery_closed() {
        let dir = tempdir().unwrap();
        let tx = TransactionId::new(21);
        let wal = WalManager::open(dir.path()).unwrap();
        wal.log(&WalRecord::TransactionRollbackToSavepoint {
            transaction_id: tx,
            name: "missing".into(),
        })
        .unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: tx,
            epoch: EpochId::new(1),
        })
        .unwrap();
        drop(wal);

        let error = WalRecovery::new(dir.path()).unwrap().recover().unwrap_err();
        assert!(error.to_string().contains("unknown savepoint"), "{error}");
    }
}

/// Crash injection tests for WAL recovery.
///
/// These tests verify that WAL recovery produces a consistent state after
/// simulated crashes at every crash point in the write path. The three crash
/// points are:
/// - `wal_before_write`: before writing length prefix + data + checksum
/// - `wal_after_write`: after writing data but before durability handling
/// - `wal_before_flush`: before fsync on TransactionCommit in Sync mode
///
/// Run with:
/// ```bash
/// cargo test -p grafeo-adapters --features "wal,testing-crash-injection" -- crash
/// ```
#[cfg(all(test, feature = "testing-crash-injection"))]
mod crash_tests {
    use super::super::test_wal_dir as tempdir;
    use super::*;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};

    /// Helper: Sync durability config so all three crash points are reachable.
    fn sync_config() -> super::super::WalConfig {
        super::super::WalConfig {
            durability: super::super::DurabilityMode::Sync,
            ..Default::default()
        }
    }

    /// Crash at `wal_before_write`: no record bytes reach disk.
    /// Recovery should only return previously committed data.
    #[test]
    fn test_crash_before_write_discards_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Seed one committed transaction
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Committed".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash at the first crash point (wal_before_write)
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(2),
                    labels: vec!["Lost".into()],
                },
            ))
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Only the first committed tx should survive
        let mut recovery = WalRecovery::new(&path).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2, "CreateNode(1) + TransactionCommit(1)");
    }

    /// Crash at `wal_after_write`: data may be in BufWriter but no commit
    /// marker. Recovery should discard the uncommitted record.
    #[test]
    fn test_crash_after_write_uncommitted_discarded() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // For a non-commit record the crash points are:
        //   1 = wal_before_write, 2 = wal_after_write
        let p = path.clone();
        let result = with_crash_at(2, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Partial".into()],
                },
            ))
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // No committed tx ⇒ recovery returns nothing
        let mut recovery = WalRecovery::new(&path).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "Uncommitted records must be discarded");
    }

    /// Two committed transactions, then crash during the third.
    /// Recovery should preserve exactly the first two.
    #[test]
    fn test_crash_preserves_prior_committed_transactions() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Commit two transactions
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["T1".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(2),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(2),
                    labels: vec!["T2".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Third transaction crashes immediately
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(3),
                    labels: vec!["T3".into()],
                },
            ))
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Both committed txs intact, third discarded
        let mut recovery = WalRecovery::new(&path).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 4, "2 CreateNode + 2 TransactionCommit");
    }

    /// A durable checkpoint boundary is safe at every publication crash point.
    ///
    /// Before `checkpoint.meta` is durable, recovery retains the pre-snapshot
    /// WAL. After it is durable, recovery may skip that WAL because the engine
    /// has already made the matching store snapshot durable.
    #[test]
    fn test_crash_during_checkpoint_preserves_snapshot_or_wal_source_of_truth() {
        for crash_at in 1..=6 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed committed data
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::lpg(
                    TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(1),
                        labels: vec!["A".into()],
                    },
                ))
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Model the engine contract: the store snapshot is synced before
            // WAL checkpoint metadata is allowed to point beyond old segments.
            let snapshot_path = path.join("snapshot.complete");
            let mut snapshot = std::fs::File::create(&snapshot_path).unwrap();
            std::io::Write::write_all(&mut snapshot, b"epoch=10").unwrap();
            snapshot.sync_all().unwrap();
            #[cfg(unix)]
            std::fs::File::open(&path).unwrap().sync_all().unwrap();

            // Crash during checkpoint
            let p = path.clone();
            let _result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.checkpoint(TransactionId::new(1), EpochId::new(10))
                    .unwrap();
            });

            let mut recovery = WalRecovery::new(&path).unwrap();
            let checkpoint = recovery.read_checkpoint_metadata().unwrap();
            let records = recovery.recover().unwrap();
            if let Some(checkpoint) = checkpoint {
                assert_eq!(checkpoint.epoch, EpochId::new(10));
                assert!(snapshot_path.exists());
                assert!(
                    path.join(format!("wal_{:08}.log", checkpoint.log_sequence))
                        .exists()
                );
                assert!(records.iter().all(|record| !matches!(
                    record,
                    WalRecord::LpgMutation { op: crate::wal::LpgMutationOp::CreateNode { id, .. }, .. } if *id == NodeId::new(1)
                )));
            } else {
                assert!(records.iter().any(|record| matches!(
                    record,
                    WalRecord::LpgMutation { op: crate::wal::LpgMutationOp::CreateNode { id, .. }, .. } if *id == NodeId::new(1)
                )));
                assert!(records.iter().any(|record| matches!(
                    record,
                    WalRecord::TransactionCommit { transaction_id }
                        if *transaction_id == TransactionId::new(1)
                )));
            }
        }
    }

    /// Crash with rotated log files: recovery should span all files.
    #[test]
    fn test_crash_with_log_rotation() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write enough to trigger rotation
        {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100, // force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(&path, config).unwrap();
            for i in 0..5 {
                wal.log(&WalRecord::lpg(
                    TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(i),
                        labels: vec!["Rotated".into()],
                    },
                ))
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash during additional write
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100,
                ..Default::default()
            };
            let wal = WalManager::with_config(&p, config).unwrap();
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(99),
                    labels: vec!["Crash".into()],
                },
            ))
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // All committed data across rotated files should survive
        let mut recovery = WalRecovery::new(&path).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 6, "5 CreateNode + 1 TransactionCommit");
    }

    /// Exhaustive sweep: crash at every possible point during a multi-record
    /// transaction and verify recovery invariants.
    ///
    /// Invariants checked:
    /// 1. Previously committed transactions always survive
    /// 2. Recovery output never contains partial (uncommitted) transactions
    #[test]
    fn test_crash_sweep_all_points() {
        for crash_at in 1..20 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed one committed transaction
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::lpg(
                    TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(1),
                        labels: vec!["Base".into()],
                    },
                ))
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Attempt a second transaction with crash injection
            let p = path.clone();
            let result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.log(&WalRecord::lpg(
                    TransactionId::new(2),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(100),
                        labels: vec!["New".into()],
                    },
                ))
                .unwrap();
                wal.log(&WalRecord::lpg(
                    TransactionId::new(2),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::SetNodeProperty {
                        id: NodeId::new(100),
                        key: "name".into(),
                        value: Value::String("test".into()),
                    },
                ))
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(2),
                })
                .unwrap();
            });

            // Verify recovery invariants
            let mut recovery = WalRecovery::new(&path).unwrap();
            let records = recovery.recover().unwrap();

            // Invariant 1: base committed tx always survives
            assert!(
                records.len() >= 2,
                "crash_at={crash_at}: base tx must survive, got {} records",
                records.len()
            );

            // Invariant 2: no partial transactions in output
            let mut pending = 0usize;
            for record in &records {
                match record {
                    WalRecord::TransactionCommit { .. }
                    | WalRecord::TransactionAbort { .. }
                    | WalRecord::Checkpoint { .. } => pending = 0,
                    _ => pending += 1,
                }
            }
            assert_eq!(
                pending, 0,
                "crash_at={crash_at}: recovery must not output partial transactions"
            );

            // If the operation completed, the second tx should also be present
            if matches!(result, CrashResult::Completed(())) {
                assert!(
                    records.len() >= 5,
                    "crash_at={crash_at}: completed run should include second tx"
                );
            }
        }
    }

    /// Aborted transactions are not recovered even without a crash.
    /// Verifies that TransactionAbort correctly discards pending records.
    #[test]
    fn test_abort_then_crash_discards_aborted_tx() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            // Committed tx
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Keep".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            // Aborted tx
            wal.log(&WalRecord::lpg(
                TransactionId::new(2),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(2),
                    labels: vec!["Discard".into()],
                },
            ))
            .unwrap();
            wal.log(&WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Crash during a third transaction
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::lpg(
                grafeo_common::types::TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(3),
                    labels: vec!["Also lost".into()],
                },
            ))
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        let mut recovery = WalRecovery::new(&path).unwrap();
        let records = recovery.recover().unwrap();
        // Only the committed tx (2 records)
        assert_eq!(
            records.len(),
            2,
            "Aborted + crashed records should both be discarded"
        );
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn test_encrypted_wal_roundtrip() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [42u8; KEY_SIZE];
        let chain = KeyChain::new(key);

        // Write encrypted records
        {
            let wal = WalManager::with_config_and_encryptor(
                dir.path(),
                WalConfig::default(),
                chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
            )
            .unwrap();

            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ))
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover with same key
        let mut recovery = WalRecovery::with_encryptor(
            dir.path(),
            chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
        )
        .unwrap();

        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2, "should recover both encrypted records");
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn test_encrypted_wal_wrong_key_fails() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [42u8; KEY_SIZE];
        let chain = KeyChain::new(key);

        // Write with key A
        {
            let wal = WalManager::with_config_and_encryptor(
                dir.path(),
                WalConfig::default(),
                chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
            )
            .unwrap();

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

            wal.sync().unwrap();
        }

        // Try recovery with wrong key B: authentication failure is corruption.
        // Recovery must fail closed and quarantine the unreadable segment.
        let wrong_key = [99u8; KEY_SIZE];
        let wrong_chain = KeyChain::new(wrong_key);
        let mut recovery = WalRecovery::with_encryptor(
            dir.path(),
            wrong_chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()),
        )
        .unwrap();

        let error = recovery.recover().unwrap_err();
        assert!(error.to_string().contains("decryption failed"), "{error}");
        assert!(
            std::fs::read_dir(dir.path()).unwrap().any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".corrupt")),
            "wrong-key recovery must leave a durable fail-closed quarantine"
        );
    }
}

#[cfg(test)]
mod cdc_group_tests {
    use super::*;
    use grafeo_common::types::EpochId;

    fn batch(tid: u64, epoch: u64, model: u8) -> WalRecord {
        WalRecord::CdcBatch {
            transaction_id: TransactionId::new(tid + 10),
            epoch: EpochId::new(epoch),
            model,
            payload: vec![1],
        }
    }
    fn commit(tid: u64, epoch: u64, models: u8) -> WalRecord {
        WalRecord::CommittedWithCdc {
            transaction_id: TransactionId::new(tid + 10),
            epoch: EpochId::new(epoch),
            models,
        }
    }
    fn recover(records: &[WalRecord]) -> Result<Vec<WalRecord>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("wal");
        let wal = WalManager::open(&path)?;
        for record in records {
            wal.log(record)?;
        }
        wal.close()?;
        drop(wal);
        WalRecovery::new(path)?.recover()
    }

    #[test]
    fn cdc_manifest_recovers_exact_mixed_and_empty_groups() {
        let records = [
            batch(1, 2, 1),
            batch(1, 2, 2),
            commit(1, 2, 3),
            commit(2, 4, 0),
        ];
        let recovered = recover(&records).unwrap();
        assert_eq!(recovered.len(), records.len());
        assert!(matches!(
            recovered.last(),
            Some(WalRecord::CommittedWithCdc { models: 0, .. })
        ));
    }

    #[test]
    fn cdc_manifest_rejects_missing_duplicate_foreign_epoch_order_and_late_batches() {
        for records in [
            vec![commit(1, 2, 1)],
            vec![batch(1, 2, 1), batch(1, 2, 1), commit(1, 2, 1)],
            vec![batch(2, 2, 1), commit(1, 2, 1)],
            vec![batch(1, 3, 1), commit(1, 2, 1)],
            vec![batch(1, 2, 2), batch(1, 2, 1), commit(1, 2, 3)],
            vec![commit(1, 2, 0), batch(1, 2, 1)],
            vec![commit(1, 2, 0), commit(1, 2, 0)],
            vec![
                batch(1, 2, 1),
                WalRecord::Committed {
                    transaction_id: TransactionId::new(11),
                    epoch: EpochId::new(2),
                },
            ],
        ] {
            let error = recover(&records)
                .expect_err("invalid semantic group must fail before native replay");
            assert!(
                matches!(error, Error::Storage(StorageError::InvalidWalEntry(_))),
                "{error:?}"
            );
        }
    }

    #[test]
    fn cdc_cannot_reopen_a_finished_transaction_with_a_new_epoch() {
        for finished in [
            commit(1, 2, 0),
            WalRecord::Committed {
                transaction_id: TransactionId::new(11),
                epoch: EpochId::new(2),
            },
            WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(11),
            },
        ] {
            assert!(
                recover(&[finished.clone(), batch(1, 3, 1)]).is_err(),
                "post-terminal batch was admitted"
            );
            assert!(
                recover(&[finished, commit(1, 3, 0)]).is_err(),
                "post-terminal manifest was admitted"
            );
        }
    }

    #[test]
    fn cdc_allows_distinct_transactions_to_commit_out_of_allocation_order() {
        assert_eq!(
            recover(&[
                batch(2, 2, 1),
                commit(2, 2, 1),
                batch(1, 3, 1),
                commit(1, 3, 1)
            ])
            .unwrap()
            .len(),
            4
        );
    }

    #[test]
    fn orphaned_and_aborted_cdc_batches_never_recover() {
        assert!(recover(&[batch(1, 2, 1)]).unwrap().is_empty());
        assert!(
            recover(&[
                batch(1, 2, 1),
                WalRecord::TransactionAbort {
                    transaction_id: TransactionId::new(11)
                }
            ])
            .unwrap()
            .is_empty()
        );
    }
}
