//! Move-only authority over one canonical WAL directory.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use grafeo_common::utils::error::Result;

use crate::ownership::{
    LockedFile, ascii_contains, ascii_ends_with, checked_file, resolve_components, suffixed,
    validate_public_path,
};

pub(crate) struct WalLease {
    path: PathBuf,
    _coordination: LockedFile,
    // Retires after the coordination FD. The last child owns private cleanup.
    pub(super) context: Option<std::sync::Arc<super::staging::StageContext>>,
    #[cfg(feature = "grafeo-file")]
    pub(crate) restore: Option<std::sync::Arc<crate::file::ContainerRestoreContext>>,
}

fn validate(path: &Path) -> Result<()> {
    let leaf = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "WAL path requires a wal or .wal leaf",
        )
    })?;
    let bytes = leaf.as_encoded_bytes();
    if !bytes.eq_ignore_ascii_case(b"wal") && !ascii_ends_with(bytes, b".wal") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WAL path requires a wal or .wal leaf",
        )
        .into());
    }
    if bytes.starts_with(b".")
        && (ascii_contains(bytes, b".grafeo-file-save-") || ascii_contains(bytes, b".grafeo-save-"))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WAL leaf uses a reserved private-stage name",
        )
        .into());
    }
    #[cfg(windows)]
    if bytes.contains(&b':') || bytes.ends_with(b".") || bytes.ends_with(b" ") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WAL leaf uses a Windows normalization or alternate-stream alias",
        )
        .into());
    }
    if let Some(parent) = path.parent() {
        validate_public_path(parent)?;
    }
    Ok(())
}

impl WalLease {
    pub(crate) fn acquire(path: &Path) -> Result<Self> {
        validate(path)?;
        let path = resolve_components(path)?;
        validate(&path)?;
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "WAL path requires a parent")
        })?;
        fs::create_dir_all(parent)?;
        let parent = fs::canonicalize(parent)?;
        validate_public_path(&parent)?;
        let leaf = path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "WAL path requires a leaf")
        })?;
        let path = parent.join(leaf);
        Self::at_resolved_path(path, None)
    }

    pub(super) fn at_resolved_path(
        path: PathBuf,
        context: Option<std::sync::Arc<super::staging::StageContext>>,
    ) -> Result<Self> {
        let coordination = checked_file(&suffixed(&path, ".grafeo-owner"), false, true)?;
        let coordination = LockedFile::acquire(coordination, false)?;
        Ok(Self {
            path,
            _coordination: coordination,
            context,
            #[cfg(feature = "grafeo-file")]
            restore: None,
        })
    }

    #[cfg(feature = "grafeo-file")]
    pub(crate) fn for_container_restore(
        path: PathBuf,
        context: std::sync::Arc<crate::file::ContainerRestoreContext>,
    ) -> Result<Self> {
        let mut lease = Self::at_resolved_path(path, None)?;
        lease.restore = Some(context);
        Ok(lease)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// A drained WAL whose permanent coordination authority is still held.
pub struct SealedWal {
    pub(crate) lease: WalLease,
}

impl SealedWal {
    /// Returns the canonical directory retained by this capability.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.lease.path()
    }

    #[cfg(feature = "grafeo-file")]
    pub(crate) fn retire_directory(&mut self) -> Result<()> {
        if let Some(context) = &self.lease.restore {
            return context.retire_wal(self.path());
        }
        remove_owned_directory(self.path())
    }
}

/// Exclusive destination authority without opening or recovering WAL content.
pub struct WalDestination {
    pub(super) lease: WalLease,
}

impl WalDestination {
    #[cfg(feature = "grafeo-file")]
    pub(crate) fn restore_child(
        path: PathBuf,
        context: std::sync::Arc<crate::file::ContainerRestoreContext>,
    ) -> Result<Self> {
        Ok(Self {
            lease: WalLease::for_container_restore(path, context)?,
        })
    }

    #[cfg(feature = "grafeo-file")]
    pub(crate) fn into_container_import(self) -> super::WalImport {
        super::WalImport::for_container_restore(self.lease)
    }

    #[cfg(feature = "grafeo-file")]
    pub(crate) fn into_container_recovery(self) -> super::WalRecovery {
        super::WalRecovery::for_container_restore(self.lease)
    }

    /// Acquires the permanent canonical sibling without creating WAL content.
    ///
    /// # Errors
    /// Returns namespace, filesystem, or nonblocking contention errors.
    pub fn acquire(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            lease: WalLease::acquire(path.as_ref())?,
        })
    }

    /// Returns the canonical directory retained by this capability.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.lease.path()
    }

    #[cfg(feature = "grafeo-file")]
    pub(crate) fn retire_directory(&mut self) -> Result<()> {
        if let Some(context) = &self.lease.restore {
            return context.retire_wal(self.path());
        }
        remove_owned_directory(self.path())
    }
}

pub(super) fn remove_owned_directory(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
        Ok(metadata) if !metadata.file_type().is_dir() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "owned WAL must be a non-symlink directory",
            )
            .into());
        }
        Ok(_) => {}
    }
    // Check every artifact before any destructive reuse; never traverse an
    // unexpected nested tree or a symlink/hard-linked artifact.
    for entry in fs::read_dir(path)? {
        let file = checked_file(&entry?.path(), false, false)?;
        drop(file);
    }
    fs::remove_dir_all(path)?;
    Ok(())
}
