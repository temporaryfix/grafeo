//! Built-in local filesystem async storage backend.
//!
//! Wraps [`AsyncLpgWal`] to provide a local implementation of
//! [`AsyncStorageBackend`]. This is the default backend used when the
//! `async-storage` feature is enabled.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use grafeo_common::utils::error::{Error, Result};

use super::async_backend::{AsyncStorageBackend, SnapshotMetadata};
use super::wal::{AsyncLpgWal, WalCloseError};

/// Local filesystem async storage backend.
///
/// Delegates WAL operations to [`AsyncLpgWal`] and provides snapshot
/// read/write via the filesystem. This is the built-in backend for
/// `grafeo-server` and other tokio-based applications.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> grafeo_common::utils::error::Result<()> {
/// use std::sync::Arc;
/// use grafeo_storage::async_local::AsyncLocalBackend;
/// use grafeo_storage::async_backend::AsyncStorageBackend;
/// use grafeo_storage::wal::AsyncLpgWal;
///
/// let wal = Arc::new(AsyncLpgWal::open("wal").await?);
/// let backend = AsyncLocalBackend::new(wal);
/// assert_eq!(backend.name(), "local-async");
/// # Ok(())
/// # }
/// ```
pub struct AsyncLocalBackend {
    wal: Arc<AsyncLpgWal>,
}

impl AsyncLocalBackend {
    /// Creates a new local backend wrapping the given async WAL.
    #[must_use]
    pub fn new(wal: Arc<AsyncLpgWal>) -> Self {
        Self { wal }
    }

    /// Returns a reference to the underlying async WAL.
    #[must_use]
    pub fn wal(&self) -> &Arc<AsyncLpgWal> {
        &self.wal
    }
}

impl AsyncStorageBackend for AsyncLocalBackend {
    fn name(&self) -> &str {
        "local-async"
    }

    fn write_wal_batch<'a>(
        &'a self,
        records: &'a [Vec<u8>],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            for record_data in records {
                // Each payload includes the current envelope from encode_record;
                // admission checks it before adding physical WAL framing.
                // force_sync is false: the caller should call sync() after a batch.
                self.wal.write_serialized_frame(record_data, false).await?;
            }
            Ok(())
        })
    }

    fn sync(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async { self.wal.sync().await })
    }

    fn write_snapshot<'a>(
        &'a self,
        _data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        // Snapshot writes for local storage are handled by the GrafeoFileManager
        // in the engine layer, not by the WAL backend. This method is a no-op
        // for local storage but is meaningful for remote backends (S3, Postgres).
        Box::pin(async {
            Err(Error::Internal(
                "local backend: use GrafeoDB::async_write_snapshot() instead".to_string(),
            ))
        })
    }

    fn read_snapshot(&self) -> Pin<Box<dyn Future<Output = Result<Option<Vec<u8>>>> + Send + '_>> {
        // Same as write_snapshot: local snapshots are managed by GrafeoFileManager.
        Box::pin(async {
            Err(Error::Internal(
                "local backend: use GrafeoDB::open() for snapshot recovery".to_string(),
            ))
        })
    }

    fn list_snapshots(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<SnapshotMetadata>>> + Send + '_>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn close(
        &self,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), WalCloseError>> + Send + '_>> {
        Box::pin(async { self.wal.close().await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::{AsyncTypedWal, WalRecord, encode_record};
    use grafeo_common::types::NodeId;

    #[tokio::test]
    async fn local_backend_name() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(AsyncTypedWal::open(dir.path().join("wal")).await.unwrap());
        let backend = AsyncLocalBackend::new(wal);
        assert_eq!(backend.name(), "local-async");
    }

    #[tokio::test]
    async fn local_backend_write_wal_batch() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(AsyncTypedWal::open(dir.path().join("wal")).await.unwrap());
        let backend = AsyncLocalBackend::new(Arc::clone(&wal));

        // Prepare complete current-generation payloads.
        let records: Vec<Vec<u8>> = (0..3)
            .map(|i| {
                let record = WalRecord::lpg(
                    grafeo_common::types::TransactionId::new(1),
                    grafeo_common::types::GraphPath::root(),
                    crate::wal::LpgMutationOp::CreateNode {
                        id: NodeId::new(i),
                        labels: vec!["Test".to_string()],
                    },
                );
                encode_record(&record).unwrap()
            })
            .collect();

        backend.write_wal_batch(&records).await.unwrap();
        backend.sync().await.unwrap();

        assert_eq!(wal.record_count(), 3);
    }

    #[tokio::test]
    async fn local_backend_close_flushes() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(AsyncTypedWal::open(dir.path().join("wal")).await.unwrap());
        let backend = AsyncLocalBackend::new(Arc::clone(&wal));

        // Write a record via the WAL directly
        wal.log(&WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            },
        ))
        .await
        .unwrap();

        // Close should flush + sync without error
        backend.close().await.unwrap();
    }

    #[tokio::test]
    async fn local_backend_as_trait_object() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(AsyncTypedWal::open(dir.path().join("wal")).await.unwrap());
        let backend: Arc<dyn AsyncStorageBackend> = Arc::new(AsyncLocalBackend::new(wal));

        assert_eq!(backend.name(), "local-async");
        backend.write_wal_batch(&[]).await.unwrap();
        backend.sync().await.unwrap();
        backend.close().await.unwrap();
    }

    #[tokio::test]
    async fn bare_bincode_batch_refusal_preserves_prefix_and_backend_health()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::EpochId;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("wal");
        let wal = Arc::new(AsyncTypedWal::open(&path).await?);
        let backend = AsyncLocalBackend::new(Arc::clone(&wal));
        let first = WalRecord::EpochAdvance {
            epoch: EpochId::new(19),
        };
        let rejected = WalRecord::EpochAdvance {
            epoch: EpochId::new(20),
        };
        let records = [
            encode_record(&first)?,
            bincode::serde::encode_to_vec(&rejected, bincode::config::standard())?,
        ];
        assert!(backend.write_wal_batch(&records).await.is_err());
        assert_eq!(wal.record_count(), 1);
        assert!(!wal.is_poisoned());
        let following = WalRecord::EpochAdvance {
            epoch: EpochId::new(21),
        };
        backend
            .write_wal_batch(&[encode_record(&following)?])
            .await?;
        assert_eq!(wal.record_count(), 2);
        backend.sync().await?;
        backend.close().await?;
        let recovered = crate::wal::WalRecovery::new(&path)?.recover()?;
        assert!(matches!(
            recovered.as_slice(),
            [WalRecord::EpochAdvance { epoch: first }, WalRecord::EpochAdvance { epoch: following }]
                if *first == EpochId::new(19) && *following == EpochId::new(21)
        ));
        Ok(())
    }

    #[tokio::test]
    async fn invalid_later_batch_frame_preserves_the_valid_prefix_and_close_is_terminal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = Arc::new(AsyncTypedWal::open(&path).await.unwrap());
        let backend = AsyncLocalBackend::new(Arc::clone(&wal));
        let record = WalRecord::EpochAdvance {
            epoch: grafeo_common::types::EpochId::new(19),
        };
        let valid = encode_record(&record).unwrap();
        let records = [valid, vec![0; 64 * 1024 * 1024 + 1]];
        assert!(backend.write_wal_batch(&records).await.is_err());
        assert_eq!(wal.record_count(), 1);
        assert!(!wal.is_poisoned());
        backend.sync().await.unwrap();
        backend.close().await.unwrap();
        assert!(backend.write_wal_batch(&records[..1]).await.is_err());
        assert!(wal.log(&record).await.is_err());
        let recovered = crate::wal::WalRecovery::new(&path)
            .unwrap()
            .recover()
            .unwrap();
        assert!(
            matches!(recovered.as_slice(), [WalRecord::EpochAdvance { epoch }] if *epoch == grafeo_common::types::EpochId::new(19))
        );
    }
}
