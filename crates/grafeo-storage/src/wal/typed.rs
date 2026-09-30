//! Type-safe WAL wrapper.
//!
//! [`TypedWal`] wraps a [`WalManager`] and ensures that only records of type `R`
//! can be written. This prevents accidentally mixing record types (e.g., LPG
//! and RDF) in the same WAL instance.
//!
//! Use [`LpgWal`] for the standard labeled property graph WAL.

use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use grafeo_common::types::EpochId;
use grafeo_common::types::TransactionId;
use grafeo_common::utils::error::{Error, Result};

use super::frame_buffer::encode_frame;
use super::log::{CheckpointMetadata, DurabilityMode, WalConfig, WalManager};
use super::record::WalEntry;
use super::{WalRecord, validate_wal_frame_payload_len};

/// A type-safe wrapper around [`WalManager`] that constrains record types
/// at compile time.
///
/// `TypedWal<R>` ensures that only records implementing [`WalEntry`] with
/// the specific type `R` can be logged. This prevents accidentally writing
/// the wrong record type to a WAL instance.
///
/// # Example
///
/// ```no_run
/// use grafeo_storage::wal::{LpgWal, WalRecord};
/// use grafeo_common::types::NodeId;
///
/// # fn main() -> grafeo_common::utils::error::Result<()> {
/// let wal = LpgWal::open("wal")?;
/// wal.log(&WalRecord::lpg(grafeo_common::types::TransactionId::new(1), grafeo_common::types::GraphPath::root(), grafeo_storage::wal::LpgMutationOp::CreateNode {
///     id: NodeId::new(1),
///     labels: vec!["Person".to_string()],
/// }))?;
/// # Ok(())
/// # }
/// ```
pub struct TypedWal<R: WalEntry> {
    manager: WalManager,
    _record: PhantomData<R>,
}

impl<R: WalEntry> TypedWal<R> {
    /// Borrows the raw WAL's existing operation authority for a complete capture.
    ///
    /// # Errors
    /// Rejects terminal WAL owners.
    pub fn capture(&self) -> Result<super::WalCapture<'_>> {
        self.manager.capture()
    }

    /// Retains this owned WAL generation from the requested segment sequence.
    ///
    /// # Errors
    /// Rejects terminal owners or exhausted retention lease capacity.
    pub fn retain_from(&self, sequence: u64) -> Result<super::WalRetentionLease> {
        self.manager.retain_from(sequence)
    }

    /// Captures the generation protected by an existing retention lease.
    ///
    /// # Errors
    /// Rejects terminal owners and leases from foreign or reopened generations.
    pub fn capture_with_lease(
        &self,
        lease: &super::WalRetentionLease,
    ) -> Result<super::WalCapture<'_>> {
        self.manager.capture_with_lease(lease)
    }

    /// Wraps the consuming recovery handoff without reopening its path.
    pub fn from_manager(manager: WalManager) -> Self {
        Self {
            manager,
            _record: PhantomData,
        }
    }

    /// Opens or creates a typed WAL in the given directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            manager: WalManager::open(dir)?,
            _record: PhantomData,
        })
    }

    /// Opens or creates a typed WAL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        Ok(Self {
            manager: WalManager::with_config(dir, config)?,
            _record: PhantomData,
        })
    }

    /// Logs a typed record to the WAL.
    ///
    /// The record is serialized via bincode and written with a length prefix
    /// and CRC32 checksum. Durability handling (fsync) is determined by the
    /// record's [`WalEntry::requires_sync`] method.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub fn log(&self, record: &R) -> Result<()> {
        record
            .validate_recovery()
            .map_err(|reason| Error::InvalidValue(format!("invalid WAL record: {reason}")))?;
        let data = encode_frame(record)?;
        validate_wal_frame_payload_len(data.len())?;
        self.manager
            .write_frame(&data, super::log::WalFrameIntent::capture(record))
    }

    /// Writes a checkpoint marker and persists checkpoint metadata.
    ///
    /// The caller must first make the matching store snapshot durable. The
    /// checkpoint then atomically moves recovery to an exact fresh WAL segment;
    /// pre-checkpoint segments are not replayed over that snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be written.
    pub fn checkpoint(&self, current_transaction: TransactionId, epoch: EpochId) -> Result<()> {
        // Deterministic caller validation is not an outcome-ambiguous WAL
        // failure and must not poison an otherwise healthy typed handle.
        if epoch == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "checkpoint epoch is the reserved PENDING sentinel".to_string(),
            ));
        }
        if !current_transaction.is_valid() {
            return Err(Error::InvalidValue(
                "checkpoint transaction ID is the reserved INVALID sentinel".to_string(),
            ));
        }
        let checkpoint_record = R::make_checkpoint(current_transaction);
        checkpoint_record.validate_recovery().map_err(|reason| {
            Error::InvalidValue(format!("invalid checkpoint WAL record: {reason}"))
        })?;
        let data = encode_frame(&checkpoint_record)?;
        validate_wal_frame_payload_len(data.len())?;
        self.manager
            .write_checkpoint_frame(&data, current_transaction, epoch)
    }

    /// Syncs the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub fn sync(&self) -> Result<()> {
        self.manager.sync()
    }

    /// Flushes the WAL buffer to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<()> {
        self.manager.flush()
    }

    /// Rotates to a new log file.
    ///
    /// # Errors
    ///
    /// Returns an error if rotation fails.
    pub fn rotate(&self) -> Result<()> {
        self.manager.rotate()
    }

    /// Returns whether an earlier WAL operation failed.
    ///
    /// The state is sticky for the lifetime of this handle. Reopening the WAL
    /// performs recovery and creates a fresh handle, which is the only safe way
    /// to determine whether an outcome-ambiguous append reached stable storage.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.manager.is_poisoned()
    }

    /// Terminal close of the underlying owner.
    ///
    /// # Errors
    /// Returns the raw close error.
    pub fn close(&self) -> Result<()> {
        self.manager.close()
    }

    /// Drains and consumes the underlying ownership capability.
    ///
    /// # Errors
    /// Returns the raw seal error.
    pub fn seal(&self) -> Result<super::SealedWal> {
        self.manager.seal()
    }

    /// Drains physical resources while retaining failed ownership.
    ///
    /// # Errors
    /// Returns the raw drain error.
    pub fn fail_and_drain(&self) -> Result<()> {
        self.manager.fail_and_drain()
    }

    /// Returns the total number of records written.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.manager.record_count()
    }

    /// Returns the WAL directory path.
    #[must_use]
    pub fn dir(&self) -> &Path {
        self.manager.dir()
    }

    /// Returns the current durability mode.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.manager.durability_mode()
    }

    /// Returns the total size of all WAL files in bytes.
    ///
    /// # Errors
    /// Returns terminal-state, filesystem or capacity errors.
    pub fn size_bytes(&self) -> Result<usize> {
        self.manager.size_bytes()
    }

    /// Returns the timestamp of the last checkpoint (Unix epoch seconds), if any.
    ///
    /// # Errors
    /// Returns terminal-state, filesystem or checkpoint metadata corruption errors.
    pub fn last_checkpoint_timestamp(&self) -> Result<Option<u64>> {
        self.manager.last_checkpoint_timestamp()
    }

    /// Returns the latest checkpoint epoch, if any.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> Option<EpochId> {
        self.manager.checkpoint_epoch()
    }

    /// Returns all WAL log file paths in sequence order.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL directory cannot be read.
    pub fn log_files(&self) -> Result<Vec<PathBuf>> {
        self.manager.log_files()
    }

    /// Reads checkpoint metadata from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata file cannot be read or deserialized.
    pub fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        self.manager.read_checkpoint_metadata()
    }

    /// Returns the path to the active WAL file.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.manager.path()
    }

    /// Returns the current WAL log sequence number.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.manager.current_sequence()
    }
}

/// Type alias for the LPG (labeled property graph) WAL.
pub type LpgWal = TypedWal<WalRecord>;

#[cfg(test)]
mod tests {
    use super::super::test_wal_dir as tempdir;
    use super::*;
    use grafeo_common::types::NodeId;

    #[cfg(feature = "testing-crash-injection")]
    fn catalog_batch(epoch: u64) -> WalRecord {
        WalRecord::CatalogBatchV3 {
            created_graph_incarnations: vec![],
            dropped_graph_incarnations: vec![],
            version: 2,
            epoch: EpochId::new(epoch),
            catalog_state: Vec::new(),
            created_graphs: Vec::new(),
            dropped_graphs: Vec::new(),
        }
    }

    #[test]
    fn test_typed_wal_write() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

        let record = WalRecord::lpg(
            TransactionId::new(2),
            grafeo_common::types::GraphPath::root(),
            super::super::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            },
        );

        wal.log(&record).unwrap();
        wal.flush().unwrap();
        assert_eq!(wal.record_count(), 1);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn append_failure_poison_is_sticky() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();
        let record = WalRecord::lpg(
            TransactionId::new(2),
            grafeo_common::types::GraphPath::root(),
            super::super::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            },
        );

        assert!(super::super::record::WalEntry::is_lpg_mutation(&record));
        grafeo_common::testing::wal_failure::enable_mutation_log_failure_once();
        assert!(wal.log(&record).is_err());
        assert!(wal.is_poisoned());
        assert!(
            wal.log(&WalRecord::Committed {
                transaction_id: TransactionId::new(2),
                epoch: EpochId::new(1),
            })
            .is_err(),
            "a later commit must not bridge over a missing WAL prefix"
        );
        assert_eq!(wal.record_count(), 0);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn lost_commit_ack_is_poisoned_but_the_durable_marker_remains_recoverable() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                ..WalConfig::default()
            },
        )
        .unwrap();
        let tid = TransactionId::new(2);
        let record = WalRecord::Committed {
            transaction_id: tid,
            epoch: EpochId::new(7),
        };

        grafeo_common::testing::wal_failure::enable_commit_ack_failure_once();
        assert!(wal.log(&record).is_err());
        assert!(
            grafeo_common::testing::wal_failure::maybe_fail_commit_ack().is_ok(),
            "the acknowledgement hook is one-shot"
        );
        grafeo_common::testing::wal_failure::disable_commit_ack_failure();
        assert!(wal.is_poisoned());
        assert_eq!(wal.record_count(), 1, "the commit frame reached the WAL");

        drop(wal);
        let recovered = super::super::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(recovered.iter().any(|entry| {
            matches!(
                entry,
                WalRecord::Committed {
                    transaction_id,
                    epoch
                } if *transaction_id == tid && *epoch == EpochId::new(7)
            )
        }));
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn catalog_batch_append_failure_is_targeted_one_shot_and_sticky() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();
        grafeo_common::testing::wal_failure::enable_catalog_batch_log_failure_once();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .expect("the catalog hook must ignore non-catalog records");

        let record = catalog_batch(2);
        assert!(WalEntry::is_catalog_batch(&record));
        assert!(wal.log(&record).is_err());
        assert!(
            grafeo_common::testing::wal_failure::maybe_fail_catalog_batch_log().is_ok(),
            "the pre-append hook is one-shot"
        );
        grafeo_common::testing::wal_failure::disable_catalog_batch_log_failure();
        assert!(wal.is_poisoned());
        assert_eq!(wal.record_count(), 1, "catalog frame was not appended");
        assert!(
            wal.log(&WalRecord::EpochAdvance {
                epoch: EpochId::new(3),
            })
            .is_err(),
            "the typed WAL poison must remain sticky"
        );

        drop(wal);
        let reopened: LpgWal = TypedWal::open(dir.path()).unwrap();
        reopened
            .log(&catalog_batch(3))
            .expect("the one-shot failure is consumed and reopen clears poison");
        assert_eq!(
            reopened.record_count(),
            1,
            "a reopened writer counts only frames appended by that handle"
        );
        drop(reopened);
        let recovered = super::super::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(matches!(
            recovered.as_slice(),
            [WalRecord::EpochAdvance { epoch: first }, WalRecord::CatalogBatchV3 { epoch: second, .. }]
                if *first == EpochId::new(1) && *second == EpochId::new(3)
        ));
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn lost_catalog_batch_ack_keeps_the_durable_frame_recoverable() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                ..WalConfig::default()
            },
        )
        .unwrap();
        let record = catalog_batch(9);

        grafeo_common::testing::wal_failure::enable_catalog_batch_ack_failure_once();
        assert!(wal.log(&record).is_err());
        assert!(
            grafeo_common::testing::wal_failure::maybe_fail_catalog_batch_ack().is_ok(),
            "the acknowledgement hook is one-shot"
        );
        grafeo_common::testing::wal_failure::disable_catalog_batch_ack_failure();
        assert!(wal.is_poisoned());
        assert_eq!(wal.record_count(), 1, "catalog frame reached the WAL");

        drop(wal);
        let recovered = super::super::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(recovered.iter().any(|entry| {
            matches!(entry, WalRecord::CatalogBatchV3 { epoch, .. } if *epoch == EpochId::new(9))
        }));
    }

    #[test]
    fn test_typed_wal_checkpoint() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

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

        wal.checkpoint(TransactionId::new(1), EpochId::new(10))
            .unwrap();
        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(10)));
    }

    #[test]
    fn invalid_checkpoint_epoch_does_not_poison_typed_wal() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

        let error = wal
            .checkpoint(TransactionId::new(1), EpochId::PENDING)
            .unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .expect("deterministic checkpoint validation must leave the writer usable");
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn invalid_record_and_checkpoint_identity_do_not_poison_typed_wal() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

        let record_error = wal
            .log(&WalRecord::Committed {
                transaction_id: TransactionId::INVALID,
                epoch: EpochId::new(1),
            })
            .unwrap_err();
        assert!(matches!(record_error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);

        let checkpoint_error = wal
            .checkpoint(TransactionId::INVALID, EpochId::new(1))
            .unwrap_err();
        assert!(matches!(checkpoint_error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);

        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .expect("deterministic identity validation must leave the typed WAL usable");
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn oversized_record_does_not_poison_typed_wal() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();
        let oversized = WalRecord::CreateSchema {
            name: "x".repeat(super::super::MAX_WAL_FRAME_BYTES),
        };

        let error = wal.log(&oversized).unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .expect("oversized-record preflight must leave the typed WAL usable");
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn test_typed_wal_recovery_compatible() {
        // Verify TypedWal writes are recoverable by existing WalRecovery
        let dir = tempdir().unwrap();

        {
            let wal: LpgWal = TypedWal::open(dir.path()).unwrap();
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

        let mut recovery = super::super::WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_typed_wal_delegates_admin_methods() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

        // Verify delegation works
        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.dir(), std::fs::canonicalize(dir.path()).unwrap());
        assert!(wal.size_bytes().unwrap() > 0 || wal.size_bytes().unwrap() == 0);
        assert!(wal.checkpoint_epoch().is_none());
        assert!(wal.last_checkpoint_timestamp().unwrap().is_none());

        let files = wal.log_files().unwrap();
        assert!(!files.is_empty());

        let _path = wal.path();
        let _mode = wal.durability_mode();
    }
}
