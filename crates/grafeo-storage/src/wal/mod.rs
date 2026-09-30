//! Write-Ahead Log - your safety net for crashes.
//!
//! Every mutation goes to the WAL before being published to the main store.
//! If a process crashes mid-transaction, [`WalRecovery`] replays only complete,
//! committed frames and restores a consistent state. A successful commit is
//! guaranteed to survive storage/power failure only in [`Sync`](DurabilityMode::Sync)
//! mode; the other modes deliberately trade an explicitly bounded or OS-managed
//! loss window for throughput. Grafeo's persistent default remains `Sync`.
//!
//! | Durability mode | What it does | When to use |
//! | --------------- | ------------ | ----------- |
//! | [`Sync`](DurabilityMode::Sync) | fsync after every commit | Can't lose any data |
//! | [`Batch`](DurabilityMode::Batch) | Periodic fsync | Balance of safety and speed |
//! | [`Adaptive`](DurabilityMode::Adaptive) | Self-tuning background sync | Variable disk latency |
//! | [`NoSync`](DurabilityMode::NoSync) | Let OS decide | Testing, when speed matters most |
//!
//! ## Adaptive Mode
//!
//! For workloads with variable disk latency, use [`Adaptive`](DurabilityMode::Adaptive)
//! mode with an [`AdaptiveFlusher`]:
//!
//! ```no_run
//! use grafeo_storage::wal::{WalManager, WalConfig, DurabilityMode, AdaptiveFlusher};
//! use std::sync::Arc;
//!
//! # fn main() -> grafeo_common::utils::error::Result<()> {
//! let config = WalConfig {
//!     durability: DurabilityMode::Adaptive { target_interval_ms: 100 },
//!     ..Default::default()
//! };
//! let wal = Arc::new(WalManager::with_config("wal", config)?);
//! let flusher = AdaptiveFlusher::new(Arc::clone(&wal), 100);
//!
//! // Use wal normally - flusher handles background syncing
//! // Drop flusher for graceful shutdown with final flush
//! # Ok(())
//! # }
//! ```
//!
//! Choose [`WalManager`] for sync code, [`AsyncWalManager`] for async.

use grafeo_common::utils::error::{Error, Result};

/// Largest complete encoded payload recovery will admit for one WAL frame.
const MAX_WAL_FRAME_BYTES: usize = 64 * 1024 * 1024;

fn validate_wal_frame_payload_len(payload_len: usize) -> Result<()> {
    if payload_len > MAX_WAL_FRAME_BYTES {
        return Err(Error::InvalidValue(format!(
            "WAL frame payload is {payload_len} bytes; maximum is {MAX_WAL_FRAME_BYTES}"
        )));
    }
    Ok(())
}

mod async_log;
#[cfg(feature = "async-storage")]
mod async_typed;
#[cfg(test)]
mod encoding_tests;
mod flusher;
mod frame_buffer;
#[cfg(test)]
mod generation_tests;
mod group;
mod log;
mod ownership;
mod record;
mod recovery;
mod staging;
mod typed;

pub use async_log::{AsyncWalManager, WalCloseError};
#[cfg(feature = "async-storage")]
pub use async_typed::{AsyncLpgWal, AsyncTypedWal};
pub use flusher::{AdaptiveFlusher, FlusherStats};
pub use frame_buffer::encode_record;
pub use log::{
    CheckpointMetadata, DurabilityMode, WalCapture, WalConfig, WalManager, WalRetentionLease,
    WalSegmentDescriptor,
};
pub use ownership::{SealedWal, WalDestination};
pub use record::{LpgMutationOp, WalEntry, WalRecord};
pub use recovery::{RecoveryReport, WalRecovery, count_wal_frames};
pub use staging::{WalImport, WalRestoreStage};
pub use typed::{LpgWal, TypedWal};

#[cfg(test)]
struct TestWalDirectory {
    _temporary: tempfile::TempDir,
    path: std::path::PathBuf,
}

#[cfg(test)]
impl TestWalDirectory {
    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

#[cfg(test)]
fn test_wal_dir() -> std::io::Result<TestWalDirectory> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("wal");
    std::fs::create_dir(&path)?;
    Ok(TestWalDirectory {
        _temporary: temporary,
        path,
    })
}

#[cfg(test)]
mod frame_limit_tests {
    use super::*;

    #[test]
    fn frame_payload_limit_is_inclusive_and_fail_closed() {
        assert!(validate_wal_frame_payload_len(MAX_WAL_FRAME_BYTES).is_ok());
        let error = validate_wal_frame_payload_len(MAX_WAL_FRAME_BYTES + 1).unwrap_err();
        assert!(matches!(error, Error::InvalidValue(_)));
    }
}
