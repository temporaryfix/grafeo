//! Single-file database format (`.grafeo`).
//!
//! This module implements a portable, crash-safe, single-file storage format.
//! At rest, the `.grafeo` file has permanent `<path>.grafeo-owner` coordination
//! metadata. Its OS lock covers primary replacement; the entry is never stale
//! cleaned. First read-only admission may provision this metadata and fails if
//! that is impossible. Existing readable metadata needs no writable parent.
//! This is cooperating-process exclusion, not protection against external
//! namespace replacement or arbitrary retained memory mappings.
//! During operation a sidecar
//! WAL directory (`<path>.wal/`) captures in-flight mutations and is
//! removed after each checkpoint.
//!
//! ## File layout
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0 | 4 KiB | [`FileHeader`]: magic `GRAF`, version, page size |
//! | 4 KiB | 4 KiB | [`DbHeader`] slot 0 (H1) |
//! | 8 KiB | 4 KiB | [`DbHeader`] slot 1 (H2) |
//! | 12 KiB+ | variable | Snapshot data payload (bincode-encoded) |
//!
//! ## Crash safety
//!
//! Two database headers alternate writes. On checkpoint, the inactive slot
//! is overwritten with metadata pointing to the freshly written snapshot.
//! If the process crashes mid-write, the other header is still valid.

pub mod format;
pub mod header;
pub mod manager;
mod ownership;

pub use format::{DbHeader, FileHeader, MAGIC};
pub use manager::{ContainerCapture, ContainerRetirement, GrafeoFileManager, SectionWrite};
pub(crate) use ownership::ContainerRestoreContext;
pub use ownership::{ContainerDestination, ContainerRestoreStage, OwnedContainerStage};

/// Opens a public backup artifact read-only, rejecting links and special files.
///
/// # Errors
/// Rejects reserved private paths, symbolic/hard links, non-regular files and
/// filesystem errors using the same checks as storage-owned artifacts.
pub fn open_backup_source(
    path: &std::path::Path,
) -> grafeo_common::utils::error::Result<std::fs::File> {
    // Raw backup segments legitimately end in .wal; container-leaf admission
    // would reject them. Still exclude private staging ancestors, and retain
    // regular-file/single-link checks on the exact read-only descriptor.
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    crate::ownership::validate_public_path(parent)?;
    crate::ownership::validate_public_path(&std::fs::canonicalize(parent)?)?;
    crate::ownership::checked_file(path, false, false)
}
