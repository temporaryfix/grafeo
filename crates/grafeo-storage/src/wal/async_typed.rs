//! Type-safe borrowed-record facade over the bounded async WAL dispatcher.

use super::{
    AsyncWalManager, DurabilityMode, WalCapture, WalCloseError, WalConfig, WalEntry, WalRecord,
    WalRetentionLease,
};
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::Result;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

/// An async WAL accepting only records of one model, without another I/O state.
///
/// ```no_run
/// # async fn example() -> grafeo_common::utils::error::Result<()> {
/// use grafeo_storage::wal::{AsyncLpgWal, WalRecord};
/// let wal = AsyncLpgWal::open("wal").await?;
/// wal.log(&WalRecord::EpochAdvance {
///     epoch: grafeo_common::types::EpochId::new(1),
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub struct AsyncTypedWal<R: WalEntry> {
    manager: AsyncWalManager,
    _record: PhantomData<R>,
}

impl<R: WalEntry> AsyncTypedWal<R> {
    /// Opens the typed WAL with default configuration.
    /// # Errors
    /// Returns runtime, ownership or initialization errors.
    pub async fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            manager: AsyncWalManager::open(dir).await?,
            _record: PhantomData,
        })
    }

    /// Opens the typed WAL with the exact supplied configuration.
    /// # Errors
    /// Returns runtime, ownership or initialization errors.
    pub async fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        Ok(Self {
            manager: AsyncWalManager::with_config(dir, config).await?,
            _record: PhantomData,
        })
    }

    /// Validates and serializes a borrowed record under bounded preparation.
    /// # Errors
    /// Returns capacity, validation, admission or physical errors.
    pub async fn log(&self, record: &R) -> Result<()> {
        self.manager.log_entry(record).await
    }

    /// Writes the model's checkpoint under the same bounded admission.
    /// # Errors
    /// Returns capacity, validation, admission or physical errors.
    pub async fn checkpoint(&self, transaction: TransactionId, epoch: EpochId) -> Result<()> {
        self.manager.checkpoint_entry::<R>(transaction, epoch).await
    }

    /// Synchronizes accepted bytes.
    /// # Errors
    /// Returns admission or physical errors.
    pub async fn sync(&self) -> Result<()> {
        self.manager.sync().await
    }

    /// Flushes accepted bytes.
    /// # Errors
    /// Returns admission or physical errors.
    pub async fn flush(&self) -> Result<()> {
        self.manager.flush().await
    }

    /// Rotates the owned active segment.
    /// # Errors
    /// Returns admission, capacity or physical errors.
    pub async fn rotate(&self) -> Result<()> {
        self.manager.rotate().await
    }

    /// Retains the WAL generation containing `sequence` for a later capture.
    ///
    /// # Errors
    /// Returns admission, ownership or retention errors.
    pub async fn retain_from(&self, sequence: u64) -> Result<WalRetentionLease> {
        self.manager.retain_from(sequence).await
    }

    /// Runs an operation against a lease-bound capture on the owned worker.
    ///
    /// # Errors
    /// Returns admission, retention, capture or action errors.
    pub async fn capture_with_lease<T, F>(&self, lease: WalRetentionLease, action: F) -> Result<T>
    where
        T: Send + 'static,
        F: for<'a> FnOnce(&mut WalCapture<'a>) -> Result<T> + Send + 'static,
    {
        self.manager.capture_with_lease(lease, action).await
    }

    /// Irreversibly seals admission and observes the shared physical close.
    /// # Errors
    /// Returns observer capacity refusal or the original shared close failure.
    pub async fn close(&self) -> std::result::Result<(), WalCloseError> {
        self.manager.close().await
    }

    pub(crate) async fn write_serialized_frame(&self, data: &[u8], force_sync: bool) -> Result<()> {
        self.manager.write_frame(data, force_sync).await
    }

    /// Returns the sole raw core's sticky physical failure state.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.manager.is_poisoned()
    }
    /// Returns the cached count.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.manager.record_count()
    }
    /// Returns the canonical directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        self.manager.dir()
    }
    /// Returns the configured durability mode.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.manager.durability_mode()
    }

    /// Enumerates physical segments under owned admission.
    /// # Errors
    /// Returns admission or filesystem errors.
    pub async fn log_files(&self) -> Result<Vec<PathBuf>> {
        self.manager.log_files().await
    }

    /// Returns the cached checkpoint epoch without filesystem work.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> Option<EpochId> {
        self.manager.checkpoint_epoch()
    }
}

/// Async WAL for the native LPG record model.
pub type AsyncLpgWal = AsyncTypedWal<WalRecord>;

#[cfg(test)]
mod tests {
    use super::super::test_wal_dir as tempdir;
    use super::*;
    use grafeo_common::types::NodeId;
    use grafeo_common::utils::error::Error;

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

    #[tokio::test]
    async fn test_async_typed_wal_write() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            },
        );

        wal.log(&record).await.unwrap();
        wal.flush().await.unwrap();
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn lease_capture_forwarders_use_the_owned_async_path() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .unwrap();
        let lease = wal.retain_from(0).await.unwrap();
        let sequence = wal
            .capture_with_lease(lease, |capture| Ok(capture.current_log_sequence()))
            .await
            .unwrap();
        assert_eq!(sequence, 0);
    }

    #[tokio::test]
    async fn foreign_async_lease_is_rejected_without_running_action() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let first_dir = tempdir().unwrap();
        let second_dir = tempdir().unwrap();
        let first: AsyncLpgWal = AsyncTypedWal::open(first_dir.path()).await.unwrap();
        let second: AsyncLpgWal = AsyncTypedWal::open(second_dir.path()).await.unwrap();
        let lease = first.retain_from(0).await.unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let action_ran = Arc::clone(&ran);
        let result = second
            .capture_with_lease(lease, move |_capture| {
                action_ran.store(true, Ordering::Release);
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert!(!ran.load(Ordering::Acquire));
        second.close().await.unwrap();
        assert!(second.retain_from(0).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_capture_waiter_keeps_lease_until_owned_worker_cleanup() {
        use std::sync::Arc;
        use std::time::Duration;

        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .unwrap();
        for _ in 0..4 {
            wal.rotate().await.unwrap();
        }
        let first_segment = dir.path().join("wal_00000000.log");
        assert!(first_segment.exists());

        let wal = Arc::new(wal);
        let lease = wal.retain_from(0).await.unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_wal = Arc::clone(&wal);
        let waiter = tokio::spawn(async move {
            worker_wal
                .capture_with_lease(lease, move |capture| {
                    started_tx.send(()).unwrap();
                    release_rx
                        .recv_timeout(Duration::from_secs(5))
                        .map_err(|error| {
                            grafeo_common::utils::error::Error::Internal(error.to_string())
                        })?;
                    Ok(capture.segments()?.len())
                })
                .await
        });
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("capture worker must start before waiter cancellation");
        waiter.abort();
        assert!(
            waiter.await.is_err(),
            "capture waiter must observe cancellation"
        );
        assert!(
            first_segment.exists(),
            "live worker lease must retain the prefix"
        );
        release_tx.send(()).unwrap();

        // A later owned operation waits for the worker, proving lease cleanup
        // has completed before retirement is attempted.
        wal.checkpoint(TransactionId::new(1), EpochId::new(2))
            .await
            .unwrap();
        assert!(
            !first_segment.exists(),
            "released lease must permit retirement"
        );
    }

    #[cfg(feature = "testing-crash-injection")]
    #[tokio::test]
    async fn failed_async_append_poison_rejects_a_later_commit() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
        let transaction_id = TransactionId::new(7);
        let mutation = WalRecord::lpg(
            transaction_id,
            grafeo_common::types::GraphPath::root(),
            super::super::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["MissingPrefix".to_string()],
            },
        );

        grafeo_common::testing::wal_failure::enable_mutation_log_failure_once();
        assert!(wal.log(&mutation).await.is_err());
        assert!(wal.is_poisoned());
        assert!(
            wal.log(&WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(1),
            })
            .await
            .is_err(),
            "an async commit must not bridge over a failed mutation append"
        );
        assert_eq!(wal.record_count(), 0);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn catalog_batch_failpoints_preserve_async_append_boundaries() {
        let before_dir = tempdir().unwrap();
        let before: AsyncLpgWal = AsyncTypedWal::open(before_dir.path()).await.unwrap();
        grafeo_common::testing::wal_failure::enable_catalog_batch_log_failure_once();
        before
            .log(&WalRecord::EpochAdvance {
                epoch: EpochId::new(1),
            })
            .await
            .expect("the catalog hook must ignore non-catalog records");
        before.sync().await.unwrap();
        assert!(before.log(&catalog_batch(2)).await.is_err());
        assert!(
            grafeo_common::testing::wal_failure::maybe_fail_catalog_batch_log().is_ok(),
            "the pre-append hook is one-shot"
        );
        grafeo_common::testing::wal_failure::disable_catalog_batch_log_failure();
        assert!(before.is_poisoned());
        assert_eq!(before.record_count(), 1);
        drop(before);
        let recovered_before = super::super::WalRecovery::new(before_dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(
            recovered_before
                .iter()
                .all(|record| !record.is_catalog_batch())
        );

        let ack_dir = tempdir().unwrap();
        let ack: AsyncLpgWal = AsyncTypedWal::with_config(
            ack_dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                ..WalConfig::default()
            },
        )
        .await
        .unwrap();
        grafeo_common::testing::wal_failure::enable_catalog_batch_ack_failure_once();
        assert!(ack.log(&catalog_batch(9)).await.is_err());
        grafeo_common::testing::wal_failure::disable_catalog_batch_ack_failure();
        assert!(ack.is_poisoned());
        assert_eq!(ack.record_count(), 1);
        drop(ack);
        let recovered_ack = super::super::WalRecovery::new(ack_dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(recovered_ack.iter().any(|record| {
            matches!(record, WalRecord::CatalogBatchV3 { epoch, .. } if *epoch == EpochId::new(9))
        }));
    }

    #[cfg(feature = "testing-crash-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lost_commit_ack_is_durable_in_a_multi_thread_runtime() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                ..WalConfig::default()
            },
        )
        .await
        .unwrap();
        let transaction_id = TransactionId::new(27);

        grafeo_common::testing::wal_failure::enable_commit_ack_failure_once();
        assert!(
            wal.log(&WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(11),
            })
            .await
            .is_err()
        );
        assert!(wal.is_poisoned());
        assert_eq!(wal.record_count(), 1);

        drop(wal);
        let recovered = super::super::WalRecovery::new(dir.path())
            .unwrap()
            .recover()
            .unwrap();
        assert!(recovered.iter().any(|record| {
            matches!(record, WalRecord::Committed { transaction_id: recovered_id, epoch }
                if *recovered_id == transaction_id && *epoch == EpochId::new(11))
        }));
    }

    #[tokio::test]
    async fn successful_async_close_is_terminal_and_idempotent() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .unwrap();

        wal.close().await.unwrap();
        wal.close().await.unwrap();
        assert!(
            wal.log(&WalRecord::EpochAdvance {
                epoch: EpochId::new(2),
            })
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn test_async_typed_wal_checkpoint() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        wal.log(&WalRecord::lpg(
            TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            },
        ))
        .await
        .unwrap();

        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .await
        .unwrap();

        wal.checkpoint(TransactionId::new(1), EpochId::new(10))
            .await
            .unwrap();

        // Checkpoint record + the two records above = 3 records total
        assert_eq!(wal.record_count(), 3);
    }

    #[tokio::test]
    async fn invalid_checkpoint_epoch_does_not_poison_async_typed_wal() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        let error = wal
            .checkpoint(TransactionId::new(1), EpochId::PENDING)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .expect("deterministic checkpoint validation must leave the writer usable");
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn invalid_record_and_checkpoint_identity_do_not_poison_async_typed_wal() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        let record_error = wal
            .log(&WalRecord::Committed {
                transaction_id: TransactionId::INVALID,
                epoch: EpochId::new(1),
            })
            .await
            .unwrap_err();
        assert!(matches!(record_error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);

        let checkpoint_error = wal
            .checkpoint(TransactionId::INVALID, EpochId::new(1))
            .await
            .unwrap_err();
        assert!(matches!(checkpoint_error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);

        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .expect("deterministic identity validation must leave the async typed WAL usable");
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn oversized_record_does_not_poison_async_typed_wal() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
        let oversized = WalRecord::CreateSchema {
            name: "x".repeat(super::super::MAX_WAL_FRAME_BYTES),
        };

        let error = wal.log(&oversized).await.unwrap_err();

        assert!(matches!(error, Error::InvalidValue(_)));
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 0);
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        })
        .await
        .expect("oversized-record preflight must leave the async typed WAL usable");
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn checkpoint_publishes_an_exact_fresh_recovery_segment() {
        use super::super::{LpgMutationOp, WalRecovery};

        let dir = tempdir().unwrap();
        let before_transaction = TransactionId::new(1);
        let after_transaction = TransactionId::new(2);
        let checkpoint_epoch = EpochId::new(10);

        let wal: AsyncLpgWal = AsyncTypedWal::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::NoSync,
                max_log_size: u64::MAX,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        wal.log(&WalRecord::lpg(
            before_transaction,
            grafeo_common::types::GraphPath::root(),
            LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Before".to_string()],
            },
        ))
        .await
        .unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: before_transaction,
            epoch: EpochId::new(1),
        })
        .await
        .unwrap();

        wal.checkpoint(before_transaction, checkpoint_epoch)
            .await
            .unwrap();

        let metadata = wal
            .manager
            .read_checkpoint_metadata()
            .await
            .unwrap()
            .expect("durable checkpoint metadata");
        assert_eq!(metadata.log_sequence, 1);
        let boundary = dir.path().join("wal_00000001.log");
        assert_eq!(
            std::fs::metadata(&boundary).unwrap().len(),
            0,
            "checkpoint metadata must name an initially fresh segment"
        );

        wal.log(&WalRecord::lpg(
            after_transaction,
            grafeo_common::types::GraphPath::root(),
            LpgMutationOp::CreateNode {
                id: NodeId::new(2),
                labels: vec!["After".to_string()],
            },
        ))
        .await
        .unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: after_transaction,
            epoch: EpochId::new(11),
        })
        .await
        .unwrap();
        wal.sync().await.unwrap();
        drop(wal);

        let recovered = WalRecovery::new(dir.path()).unwrap().recover().unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(
            recovered
                .iter()
                .all(|record| { record.transaction_id() == Some(after_transaction) })
        );
        assert!(recovered.iter().any(|record| matches!(
            record,
            WalRecord::LpgMutation {
                op: LpgMutationOp::CreateNode { id, .. },
                ..
            } if *id == NodeId::new(2)
        )));

        let reopened: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
        assert_eq!(reopened.checkpoint_epoch(), Some(checkpoint_epoch));
    }

    #[tokio::test]
    async fn test_async_typed_wal_recovery_compatible() {
        // Verify AsyncTypedWal writes are recoverable by existing sync WalRecovery.
        // The on-disk format (length-prefix + bincode + CRC32) is identical.
        let dir = tempdir().unwrap();

        {
            let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
            wal.log(&WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ))
            .await
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .await
            .unwrap();
            wal.sync().await.unwrap();
        }

        let mut recovery = super::super::WalRecovery::new(dir.path()).unwrap();
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn test_async_sync_byte_equivalence() {
        // Same mutation sequence through sync TypedWal and async AsyncTypedWal
        // should produce identical WAL frames (same serialization + CRC).
        use super::super::TypedWal;

        let sync_dir = tempdir().unwrap();
        let async_dir = tempdir().unwrap();

        let mut records = vec![
            WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
            ),
            WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::SetNodeProperty {
                    id: NodeId::new(1),
                    key: "name".to_string(),
                    value: grafeo_common::types::Value::String("Alix".into()),
                },
            ),
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            },
        ];

        for graph in [
            grafeo_common::types::GraphPath::from_components(&[""]).unwrap(),
            grafeo_common::types::GraphPath::from_components(&["a/b"]).unwrap(),
            grafeo_common::types::GraphPath::from_components(&["a", "", "b"]).unwrap(),
        ] {
            records.extend([
                WalRecord::CreateLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph: graph.clone(),
                    transaction_id: TransactionId::new(2),
                },
                WalRecord::lpg(
                    TransactionId::new(2),
                    graph.clone(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(1),
                        labels: vec![],
                    },
                ),
                WalRecord::SetGraphTypeBinding {
                    transaction_id: TransactionId::new(2),
                    graph: graph.clone(),
                    graph_type: Some("T".into()),
                },
                WalRecord::DropLpgGraph {
                    incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                    graph,
                    transaction_id: TransactionId::new(2),
                },
            ]);
        }
        records.extend([
            WalRecord::IndexOwnerBatch {
                transaction_id: TransactionId::new(2),
                payload: vec![1],
            },
            WalRecord::Committed {
                transaction_id: TransactionId::new(2),
                epoch: EpochId::new(2),
            },
            WalRecord::CatalogBatchV3 {
                created_graph_incarnations: vec![grafeo_common::types::GraphIncarnationId::new(1)],
                dropped_graph_incarnations: vec![grafeo_common::types::GraphIncarnationId::new(2)],
                version: 2,
                epoch: EpochId::new(3),
                catalog_state: vec![1],
                created_graphs: vec![
                    grafeo_common::types::GraphPath::from_components(&["catalog", "child"])
                        .unwrap(),
                ],
                dropped_graphs: vec![
                    grafeo_common::types::GraphPath::from_components(&["catalog/child"]).unwrap(),
                ],
            },
        ]);

        // Write via sync path
        {
            let wal: super::super::LpgWal = TypedWal::open(sync_dir.path()).unwrap();
            for record in &records {
                wal.log(record).unwrap();
            }
            wal.sync().unwrap();
        }

        // Write via async path
        {
            let wal: AsyncLpgWal = AsyncTypedWal::open(async_dir.path()).await.unwrap();
            for record in &records {
                wal.log(record).await.unwrap();
            }
            wal.sync().await.unwrap();
        }

        assert_eq!(
            std::fs::read(sync_dir.path().join("wal_00000000.log")).unwrap(),
            std::fs::read(async_dir.path().join("wal_00000000.log")).unwrap(),
            "sync and async must emit identical canonical framed bytes",
        );

        // Both should be recoverable with the same records
        let mut sync_recovery = super::super::WalRecovery::new(sync_dir.path()).unwrap();
        let mut async_recovery = super::super::WalRecovery::new(async_dir.path()).unwrap();

        let sync_records = sync_recovery.recover().unwrap();
        let async_records = async_recovery.recover().unwrap();

        assert_eq!(sync_records.len(), async_records.len());
        assert_eq!(sync_records.len(), records.len());

        // Compare record-by-record via Debug representation
        for (sync_rec, async_rec) in sync_records.iter().zip(async_records.iter()) {
            assert_eq!(format!("{sync_rec:?}"), format!("{async_rec:?}"));
        }
    }

    #[tokio::test]
    async fn test_async_typed_wal_delegates_admin_methods() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.dir(), std::fs::canonicalize(dir.path()).unwrap());
        assert!(wal.checkpoint_epoch().is_none());

        let files = wal.log_files().await.unwrap();
        assert!(!files.is_empty());

        let _mode = wal.durability_mode();
    }
}
