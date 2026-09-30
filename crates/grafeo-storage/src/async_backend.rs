//! Async storage backend trait for pluggable persistence.
//!
//! [`AsyncStorageBackend`] covers the I/O layer (WAL writes, snapshots, sync)
//! while the query engine uses the sync `GraphStore` trait (in `grafeo-core`)
//! for in-memory access. This separation keeps CPU-bound query execution
//! free of async overhead while allowing I/O-bound persistence to use tokio.
//!
//! # Built-in Implementations
//!
//! - [`AsyncLocalBackend`](super::async_local::AsyncLocalBackend): wraps
//!   [`AsyncLpgWal`](super::wal::AsyncLpgWal) for local filesystem persistence.
//!
//! # Community Implementations
//!
//! Implement this trait to add remote persistence (Postgres, S3, etc.):
//!
//! ```no_run
//! use std::future::Future;
//! use std::pin::Pin;
//! use grafeo_storage::async_backend::{AsyncStorageBackend, SnapshotMetadata};
//! use grafeo_common::utils::error::Result;
//! use grafeo_storage::wal::WalCloseError;
//!
//! struct MyRemoteBackend { /* ... */ }
//!
//! impl AsyncStorageBackend for MyRemoteBackend {
//!     fn name(&self) -> &str { "my-remote" }
//!
//!     fn write_wal_batch<'a>(
//!         &'a self, records: &'a [Vec<u8>],
//!     ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
//!         Box::pin(async move { todo!() })
//!     }
//!
//!     fn sync(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
//!         Box::pin(async move { todo!() })
//!     }
//!
//!     fn write_snapshot<'a>(
//!         &'a self, data: &'a [u8],
//!     ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
//!         Box::pin(async move { todo!() })
//!     }
//!
//!     fn read_snapshot(
//!         &self,
//!     ) -> Pin<Box<dyn Future<Output = Result<Option<Vec<u8>>>> + Send + '_>> {
//!         Box::pin(async move { todo!() })
//!     }
//!
//!     fn list_snapshots(
//!         &self,
//!     ) -> Pin<Box<dyn Future<Output = Result<Vec<SnapshotMetadata>>> + Send + '_>> {
//!         Box::pin(async move { todo!() })
//!     }
//!
//!     fn close(&self) -> Pin<Box<dyn Future<Output = std::result::Result<(), WalCloseError>> + Send + '_>> {
//!         Box::pin(async move { Ok(()) })
//!     }
//! }
//! ```

use std::future::Future;
use std::pin::Pin;

use crate::wal::WalCloseError;
use grafeo_common::utils::error::Result;

/// Metadata for a stored database snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    /// The epoch at which the snapshot was taken.
    pub epoch: u64,
    /// Unix timestamp (seconds) when the snapshot was created.
    pub timestamp: u64,
    /// Size of the snapshot in bytes.
    pub size_bytes: u64,
}

/// Async storage backend for I/O-bound persistence operations.
///
/// This trait covers the persistence layer (WAL batches, snapshots, sync),
/// **not** the query execution layer. Query operators use the sync
/// `GraphStore` trait (in `grafeo-core`) for in-memory access.
///
/// All methods return `Pin<Box<dyn Future<...>>>` for object safety, enabling
/// `Arc<dyn AsyncStorageBackend>` usage without the `async_trait` crate.
///
/// # Implementors
///
/// - [`AsyncLocalBackend`](super::async_local::AsyncLocalBackend): built-in
///   local filesystem backend wrapping `AsyncLpgWal`.
/// - External: implement this trait for Postgres, S3, or any async-capable storage.
pub trait AsyncStorageBackend: Send + Sync {
    /// Backend name for diagnostics and logging.
    fn name(&self) -> &str;

    /// Write a batch of current-generation WAL record payloads.
    ///
    /// Each element is the complete output of [`crate::wal::encode_record`]:
    /// `GRAFOWAL`, a little-endian `u16` generation, the group coordinate, and
    /// the serialized record. Commit/abort payloads include a zeroed seal
    /// reservation that the serialized WAL owner fills from accepted records.
    /// It excludes the physical length/checksum or authenticated-encryption
    /// framing, which the backend supplies. Bare bincode records and unsupported
    /// generations must be rejected before that element is appended.
    /// A batch is not atomic: an error can leave earlier valid elements appended.
    fn write_wal_batch<'a>(
        &'a self,
        records: &'a [Vec<u8>],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Flush and fsync pending writes to durable storage.
    fn sync(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    /// Write a full database snapshot.
    ///
    /// The data is the output of `GrafeoDB::export_snapshot()`, a self-contained
    /// binary blob that can be imported with `GrafeoDB::import_snapshot()`.
    fn write_snapshot<'a>(
        &'a self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Read the latest snapshot, if one exists.
    fn read_snapshot(&self) -> Pin<Box<dyn Future<Output = Result<Option<Vec<u8>>>> + Send + '_>>;

    /// List available snapshots for point-in-time recovery.
    fn list_snapshots(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<SnapshotMetadata>>> + Send + '_>>;

    /// Close the backend, flushing any pending writes.
    fn close(
        &self,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), WalCloseError>> + Send + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Verify that `AsyncStorageBackend` is object-safe: we can create
    /// `Arc<dyn AsyncStorageBackend>`.
    struct MockBackend;

    impl AsyncStorageBackend for MockBackend {
        fn name(&self) -> &str {
            "mock"
        }

        fn write_wal_batch<'a>(
            &'a self,
            _records: &'a [Vec<u8>],
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }

        fn sync(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
            Box::pin(async { Ok(()) })
        }

        fn write_snapshot<'a>(
            &'a self,
            _data: &'a [u8],
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }

        fn read_snapshot(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<Option<Vec<u8>>>> + Send + '_>> {
            Box::pin(async { Ok(None) })
        }

        fn list_snapshots(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<SnapshotMetadata>>> + Send + '_>> {
            Box::pin(async { Ok(vec![]) })
        }

        fn close(
            &self,
        ) -> Pin<Box<dyn Future<Output = std::result::Result<(), WalCloseError>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn trait_is_object_safe() {
        let backend: Arc<dyn AsyncStorageBackend> = Arc::new(MockBackend);
        assert_eq!(backend.name(), "mock");
        backend.write_wal_batch(&[]).await.unwrap();
        backend.sync().await.unwrap();
        backend.write_snapshot(&[]).await.unwrap();
        assert!(backend.read_snapshot().await.unwrap().is_none());
        assert!(backend.list_snapshots().await.unwrap().is_empty());
        backend.close().await.unwrap();
    }

    #[tokio::test]
    async fn mock_backend_roundtrip() {
        use grafeo_common::utils::error::Error;
        use std::sync::Mutex;

        /// A mock that stores snapshots in memory.
        struct MemoryBackend {
            snapshot: Mutex<Option<Vec<u8>>>,
        }

        impl AsyncStorageBackend for MemoryBackend {
            fn name(&self) -> &str {
                "memory-mock"
            }

            fn write_wal_batch<'a>(
                &'a self,
                _records: &'a [Vec<u8>],
            ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
                Box::pin(async { Ok(()) })
            }

            fn sync(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
                Box::pin(async { Ok(()) })
            }

            fn write_snapshot<'a>(
                &'a self,
                data: &'a [u8],
            ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
                Box::pin(async move {
                    *self
                        .snapshot
                        .lock()
                        .map_err(|e| Error::Internal(format!("lock poisoned: {e}")))? =
                        Some(data.to_vec());
                    Ok(())
                })
            }

            fn read_snapshot(
                &self,
            ) -> Pin<Box<dyn Future<Output = Result<Option<Vec<u8>>>> + Send + '_>> {
                Box::pin(async {
                    Ok(self
                        .snapshot
                        .lock()
                        .map_err(|e| Error::Internal(format!("lock poisoned: {e}")))?
                        .clone())
                })
            }

            fn list_snapshots(
                &self,
            ) -> Pin<Box<dyn Future<Output = Result<Vec<SnapshotMetadata>>> + Send + '_>>
            {
                Box::pin(async {
                    let guard = self
                        .snapshot
                        .lock()
                        .map_err(|e| Error::Internal(format!("lock poisoned: {e}")))?;
                    if let Some(ref data) = *guard {
                        Ok(vec![SnapshotMetadata {
                            epoch: 1,
                            timestamp: 0,
                            size_bytes: data.len() as u64,
                        }])
                    } else {
                        Ok(vec![])
                    }
                })
            }

            fn close(
                &self,
            ) -> Pin<Box<dyn Future<Output = std::result::Result<(), WalCloseError>> + Send + '_>>
            {
                Box::pin(async { Ok(()) })
            }
        }

        let backend: Arc<dyn AsyncStorageBackend> = Arc::new(MemoryBackend {
            snapshot: Mutex::new(None),
        });

        // No snapshot initially
        assert!(backend.read_snapshot().await.unwrap().is_none());
        assert!(backend.list_snapshots().await.unwrap().is_empty());

        // Write a snapshot
        let data = b"test snapshot data";
        backend.write_snapshot(data).await.unwrap();

        // Read it back
        let read = backend.read_snapshot().await.unwrap().unwrap();
        assert_eq!(read, data);

        // List shows one snapshot
        let list = backend.list_snapshots().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].size_bytes, data.len() as u64);
    }
}
