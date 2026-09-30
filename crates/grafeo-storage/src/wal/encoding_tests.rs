//! Writer-level proof that frame preparation stops before exceeding its budget.

use super::{MAX_WAL_FRAME_BYTES, TypedWal, WalEntry, test_wal_dir};
use grafeo_common::types::{EpochId, TransactionId};
use serde::ser::SerializeTupleVariant;
use serde::{Deserialize, Serialize, Serializer};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;

type TestResult = Result<(), Box<dyn std::error::Error>>;
const CHUNK_BYTES: usize = 64 * 1024;
const CHUNKS: usize = 2 * MAX_WAL_FRAME_BYTES / CHUNK_BYTES;
static CHUNK: [u8; CHUNK_BYTES] = [7; CHUNK_BYTES];
std::thread_local! {
    static SERIALIZED_CHUNKS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Debug, Deserialize)]
enum ProbeRecord {
    Oversized,
    Small,
}

struct Chunk;

impl Serialize for Chunk {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SERIALIZED_CHUNKS.with(|count| count.set(count.get() + 1));
        serializer.serialize_bytes(&CHUNK)
    }
}

impl Serialize for ProbeRecord {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            // Adversarial streaming serializer: no oversized input allocation.
            // This variant must never reach the WAL or its decoder.
            Self::Oversized => {
                let mut tuple =
                    serializer.serialize_tuple_variant("ProbeRecord", 0, "Oversized", CHUNKS)?;
                for _ in 0..CHUNKS {
                    tuple.serialize_field(&Chunk)?;
                }
                tuple.end()
            }
            Self::Small => serializer.serialize_unit_variant("ProbeRecord", 1, "Small"),
        }
    }
}

impl WalEntry for ProbeRecord {
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
        matches!(self, Self::Oversized)
    }
    fn make_checkpoint(_: TransactionId) -> Self {
        Self::Oversized
    }
}

fn directory_image(path: &Path) -> std::io::Result<BTreeMap<OsString, Vec<u8>>> {
    std::fs::read_dir(path)?
        .map(|entry| {
            let entry = entry?;
            Ok((entry.file_name(), std::fs::read(entry.path())?))
        })
        .collect()
}

fn assert_bounded_preparation() {
    let count = SERIALIZED_CHUNKS.with(Cell::get);
    assert!(count > 0);
    assert!(
        count <= MAX_WAL_FRAME_BYTES / CHUNK_BYTES,
        "serialization must stop at the frame budget, not after preparing {count} chunks"
    );
}

#[test]
fn sync_preparation_refusal_is_bounded_and_pre_admission() -> TestResult {
    for checkpoint in [false, true] {
        let dir = test_wal_dir()?;
        let wal = TypedWal::<ProbeRecord>::open(dir.path())?;
        wal.log(&ProbeRecord::Small)?;
        wal.flush()?;
        let before = directory_image(dir.path())?;
        let size = wal.size_bytes()?;
        let sequence = wal.current_sequence();
        SERIALIZED_CHUNKS.with(|count| count.set(0));
        let result = if checkpoint {
            wal.checkpoint(TransactionId::new(1), EpochId::new(1))
        } else {
            wal.log(&ProbeRecord::Oversized)
        };
        assert!(result.is_err());
        assert_bounded_preparation();
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 1);
        assert_eq!(wal.size_bytes()?, size);
        assert_eq!(wal.current_sequence(), sequence);
        assert_eq!(wal.checkpoint_epoch(), None);
        assert!(wal.read_checkpoint_metadata()?.is_none());
        assert_eq!(directory_image(dir.path())?, before);
        wal.log(&ProbeRecord::Small)?;
        assert_eq!(wal.record_count(), 2);
        wal.close()?;
    }
    Ok(())
}

#[cfg(feature = "async-storage")]
#[tokio::test]
async fn async_preparation_has_the_same_bound_and_refusal_semantics() -> TestResult {
    for checkpoint in [false, true] {
        let dir = test_wal_dir()?;
        let wal = super::AsyncTypedWal::<ProbeRecord>::open(dir.path()).await?;
        wal.log(&ProbeRecord::Small).await?;
        wal.flush().await?;
        let before = directory_image(dir.path())?;
        SERIALIZED_CHUNKS.with(|count| count.set(0));
        let result = if checkpoint {
            wal.checkpoint(TransactionId::new(1), EpochId::new(1)).await
        } else {
            wal.log(&ProbeRecord::Oversized).await
        };
        assert!(result.is_err());
        assert_bounded_preparation();
        assert!(!wal.is_poisoned());
        assert_eq!(wal.record_count(), 1);
        assert_eq!(wal.checkpoint_epoch(), None);
        assert_eq!(directory_image(dir.path())?, before);
        wal.log(&ProbeRecord::Small).await?;
        assert_eq!(wal.record_count(), 2);
        wal.close().await?;
    }
    Ok(())
}
