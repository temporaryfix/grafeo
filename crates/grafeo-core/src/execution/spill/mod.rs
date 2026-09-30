//! Framed synchronous spill primitives and native blocking-operator adapters.
//!
//! Individual staged records and external-sort merge cursors are bounded and
//! carry explicit spill-file lifecycle safety. [`ExternalSort::merge_all`]
//! remains a collecting compatibility API, and [`PartitionedState::drain_all`]
//! / `iter_all` still reload/materialize compatibility state. This module
//! therefore does not claim end-to-end bounded query execution yet.
//!
//! | Component | Purpose |
//! | --------- | ------- |
//! | [`SpillManager`] | Manages spill file lifecycle with automatic cleanup |
//! | [`SpillFile`] | Read/write individual spill files |
//! | [`ExternalSort`] | External merge sort for big ORDER BY |
//! | [`PartitionedState`] | Hash partitioning for spillable GROUP BY |

//! Orphan recovery is admitted through [`SpillRoot::scavenge`]. The former
//! path-only inspection and independently supplied verifiers are absent:
//!
//! ```compile_fail,E0432
//! use grafeo_core::execution::spill::SpillOwnerVerifier;
//! ```
//! ```compile_fail,E0432
//! use grafeo_core::execution::spill::SpillOwnerMarker;
//! ```
//! ```compile_fail,E0432
//! use grafeo_core::execution::spill::StructuralSpillOwnerVerifier;
//! ```
//! ```compile_fail,E0432
//! use grafeo_core::execution::spill::SpillOrphanCandidate;
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::SpillManager;
//! let _ = SpillManager::inspect_orphan_query_directories::<&std::path::Path>;
//! ```

//! Borrowed and implicit temporary directories cannot grant query ownership:
//!
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::new(std::path::PathBuf::new());
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::new_with_disk_quota(std::path::PathBuf::new(), SpillDiskQuota::new(0));
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::new_with_provider(std::path::PathBuf::new(), Arc::new(CleartextSpillRecordProvider), SpillFrameLimits::format_max());
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::new_with_provider_and_io(std::path::PathBuf::new(), Arc::new(CleartextSpillRecordProvider), SpillFrameLimits::format_max(), Arc::new(NoopSpillIo));
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::new_with_provider_io_and_disk_quota(std::path::PathBuf::new(), Arc::new(CleartextSpillRecordProvider), SpillFrameLimits::format_max(), Arc::new(NoopSpillIo), SpillDiskQuota::new(0));
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::with_temp_dir();
//! ```

//! Owned query storage requires authenticated root admission:
//!
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::create_query(std::path::Path::new("."));
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::create_query_with_provider(std::path::Path::new("."), Arc::new(CleartextSpillRecordProvider), SpillFrameLimits::format_max());
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::create_query_with_provider_and_io(std::path::Path::new("."), Arc::new(CleartextSpillRecordProvider), SpillFrameLimits::format_max(), Arc::new(NoopSpillIo));
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::create_query_with_provider_io_and_disk_quota(std::path::Path::new("."), Arc::new(CleartextSpillRecordProvider), SpillFrameLimits::format_max(), Arc::new(NoopSpillIo), SpillDiskQuota::new(0));
//! ```
//! ```compile_fail,E0599
//! use grafeo_core::execution::spill::{SpillManager, SpillDiskQuota, CleartextSpillRecordProvider, SpillFrameLimits, NoopSpillIo};
//! use std::sync::Arc;
//! let _ = SpillManager::create_query_with_disk_quota(std::path::Path::new("."), SpillDiskQuota::new(0));
//! ```

mod external_sort;
mod file;
#[cfg(target_os = "macos")]
mod macos_acl;
mod manager;
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub use macos_acl::validate_no_acl_grants;
mod owned;
mod partition;
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
mod quota;
mod root;

pub use root::{SpillQueryLease, SpillRoot, SpillRootAuthority, SpillScavengeReport};

pub(crate) use external_sort::OwnedExactSortCursor;
pub(crate) use external_sort::recover_exact_final_operator_primary;

#[derive(Debug)]
struct SpillCleanupContext {
    primary: std::io::Error,
    cleanup: std::io::Error,
    cleanup_context: &'static str,
}

impl std::fmt::Display for SpillCleanupContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}; {} also failed: {}",
            self.primary, self.cleanup_context, self.cleanup
        )
    }
}

impl std::error::Error for SpillCleanupContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.primary)
    }
}

fn combine_primary_and_cleanup(
    primary: std::io::Error,
    cleanup: std::io::Error,
    cleanup_context: &'static str,
) -> std::io::Error {
    let kind = primary.kind();
    std::io::Error::new(
        kind,
        SpillCleanupContext {
            primary,
            cleanup,
            cleanup_context,
        },
    )
}

/// Forgets an untrusted cleanup failure without invoking its destructor from a
/// best-effort rollback or `Drop` backstop.
///
/// Cleanup hooks may return errors or panic with arbitrary user-owned values.
/// Their destructors are equally untrusted and can double-panic through their
/// own fields before another unwind boundary can catch them. Leaking each
/// opaque failure on an already-failed Drop path is the only fail-safe choice.
/// Explicit cleanup APIs still return values normally. Construction rollback
/// uses this only for a caught panic that would otherwise replace its primary.
fn forget_cleanup_failure<T>(failure: T) {
    std::mem::forget(failure);
}

/// Resolves a fallible cleanup attempted behind an unwind boundary.
///
/// This is only for best-effort `Drop` paths. Explicit cleanup APIs continue
/// to return their structured errors to the caller.
fn cleanup_backstop_succeeded<E>(outcome: std::thread::Result<Result<(), E>>) -> bool {
    match outcome {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            forget_cleanup_failure(error);
            false
        }
        Err(panic) => {
            forget_cleanup_failure(panic);
            false
        }
    }
}

/// Runs one direct cleanup attempt behind the Drop-only failure policy.
fn run_cleanup_backstop<E>(cleanup: impl FnOnce() -> Result<(), E>) -> bool {
    cleanup_backstop_succeeded(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        cleanup,
    )))
}

pub(crate) use external_sort::{
    ExternalSortGrantObserver, ExternalSortOperationError, ExternalSortPrimary, PullSortFailure,
    PullSortHookAuthority, PullSortHookGuard, classify_external_sort_error,
    classify_operator_error,
};

#[cfg(test)]
mod framing_tests;

pub use crate::execution::value_codec::{
    CodecLimits, SpillCodecDecodeBudget, SpillCodecEncodeBudget, deserialize_row,
    deserialize_row_with_limits, deserialize_value, deserialize_value_with_limits, serialize_row,
    serialize_row_with_limits, serialize_value, serialize_value_with_limits,
};
pub use external_sort::{
    ExternalSort, ExternalSortChunk, ExternalSortCursor, NullOrder, SemanticRowComparator,
    SortDirection, SortKey,
};
pub use file::{
    CleartextSpillRecordProvider, MAX_FIXED_CONTROL_PAYLOAD_BYTES, MAX_SPILL_RECORD_BYTES,
    NoopSpillIo, OpenSpillRecord, SPILL_FILE_MAGIC, SPILL_FORMAT_VERSION,
    SPILL_RECORD_HEADER_BYTES, SpillFile, SpillFileIdentity, SpillFileReader, SpillFileRole,
    SpillFrameLimits, SpillIo, SpillIoOperation, SpillQueryIdentity, SpillRecordKind,
    SpillRecordMeta, SpillRecordProvider,
};
#[cfg(test)]
pub(crate) use manager::BorrowedSpillFixture;
pub use manager::{
    SpillDiskQuota, SpillDiskStats, SpillManager, SpillPhysicalStats, SpillQuotaExceeded,
};
pub use owned::{
    OwnedSpillBytes, OwnedSpillError, OwnedSpillFile, OwnedSpillReader, OwnedSpillRecord,
    OwnedSpillWrite,
};
pub(crate) use partition::PartitionUpdateAdmission;
pub use partition::{
    AccountedPartitionKey, AccountedPartitionValue, DEFAULT_NUM_PARTITIONS,
    PartitionAdmissionError, PartitionDrainCursor, PartitionDrainEntry, PartitionOperationError,
    PartitionedState,
};
#[cfg(test)]
pub(crate) use root::RootedSpillFixture;

#[cfg(test)]
mod tests {
    use super::{
        CodecLimits, deserialize_row_with_limits, deserialize_value_with_limits,
        serialize_row_with_limits, serialize_value_with_limits,
    };
    use grafeo_common::types::Value;
    use std::io::Cursor;

    #[test]
    fn limit_aware_codec_api_is_reexported_from_spill() {
        let limits = CodecLimits::new(1024, 16, 4, 8);
        let value = Value::Int64(42);
        let row = [value.clone()];

        let mut value_bytes = Vec::new();
        serialize_value_with_limits(&value, &mut value_bytes, limits).unwrap();
        let mut direct_value_bytes = Vec::new();
        crate::execution::value_codec::serialize_value_with_limits(
            &value,
            &mut direct_value_bytes,
            limits,
        )
        .unwrap();
        assert_eq!(value_bytes, direct_value_bytes);
        assert_eq!(
            deserialize_value_with_limits(&mut Cursor::new(value_bytes), limits).unwrap(),
            value
        );

        let mut row_bytes = Vec::new();
        serialize_row_with_limits(&row, &mut row_bytes, limits).unwrap();
        assert_eq!(
            deserialize_row_with_limits(&mut Cursor::new(row_bytes), 1, limits).unwrap(),
            row
        );
    }
}
