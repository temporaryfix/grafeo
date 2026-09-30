//! Stable path ownership for cooperating container users.
//!
//! The permanent sibling inode, not its contents or the replaceable primary,
//! carries the OS lock. External namespace replacement remains out of scope.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::GrafeoFileManager;
pub(super) use crate::ownership::{LockedFile, check_single_link, suffixed, sync_parent};
use crate::ownership::{require_regular, resolve, validate_public_path};
use grafeo_common::utils::error::{Error, Result};

pub(super) struct ContainerLease {
    path: PathBuf,
    // Retiring this handle releases the lease; never unlink its public entry.
    _coordination: LockedFile,
    // Drops after the child's coordination FD; last child performs private cleanup.
    restore: Option<Arc<ContainerRestoreContext>>,
}

impl ContainerLease {
    pub(super) fn acquire(path: &Path, shared: bool, provision_parent: bool) -> Result<Self> {
        validate_public_path(path)?;
        let resolved = resolve(path, provision_parent)?;
        validate_public_path(&resolved)?;
        Self::at_resolved_path(resolved, shared)
    }

    fn at_resolved_path(path: PathBuf, shared: bool) -> Result<Self> {
        let coordinate = suffixed(&path, ".grafeo-owner");
        let coordination = match fs::symlink_metadata(&coordinate) {
            Ok(metadata) => {
                require_regular(&metadata)?;
                OpenOptions::new().read(true).open(&coordinate)?
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                match OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&coordinate)
                {
                    Ok(file) => file,
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        require_regular(&fs::symlink_metadata(&coordinate)?)?;
                        OpenOptions::new().read(true).open(&coordinate)?
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            Err(error) => return Err(error.into()),
        };
        require_regular(&coordination.metadata()?)?;
        require_regular(&fs::symlink_metadata(&coordinate)?)?;
        check_single_link(&coordination)?;
        let coordination = LockedFile::acquire(coordination, shared)?;
        Ok(Self {
            path,
            _coordination: coordination,
            restore: None,
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

/// Exclusive destination authority retained across copy and follow-on work.
/// No primary bytes are changed by acquisition. The resolved path is the only
/// destination accepted by the checked copying and staging operations.
pub struct ContainerDestination {
    lease: ContainerLease,
}

impl ContainerDestination {
    /// Binds a read-only source and writable output in canonical order.
    ///
    /// # Errors
    /// Rejects aliases before provisioning or locking and returns admission errors.
    pub fn acquire_copy_pair(source: &Path, output: &Path) -> Result<(GrafeoFileManager, Self)> {
        validate_public_path(source)?;
        validate_public_path(output)?;
        let source = resolve(source, false)?;
        let output = resolve(output, false)?;
        if source == output {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot copy a container onto itself",
            )
            .into());
        }
        if source < output {
            let source = GrafeoFileManager::open_read_only(source)?;
            Ok((source, Self::acquire(output)?))
        } else {
            let output = Self::acquire(output)?;
            Ok((GrafeoFileManager::open_read_only(source)?, output))
        }
    }

    /// Acquires a nonblocking exclusive lease, provisioning the parent if needed.
    ///
    /// # Errors
    /// Returns the actual admission error, including contention and reserved paths.
    pub fn acquire(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            lease: ContainerLease::acquire(path.as_ref(), false, true)?,
        })
    }

    /// The canonical resource bound to this authority.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.lease.path()
    }

    /// Acquires the exact sidecar WAL destination while C is already owned.
    ///
    /// # Errors
    /// Returns WAL namespace, filesystem or nonblocking contention errors.
    pub fn acquire_sidecar_wal(&self) -> Result<crate::wal::WalDestination> {
        crate::wal::WalDestination::acquire(suffixed(self.path(), ".wal"))
    }

    /// Creates the only admitted container inside a fresh, private staging tree.
    /// The returned stage owns the destination and sidecar WAL leases and only
    /// lends its manager. Existing sidecar contents are never adopted or erased.
    ///
    /// # Errors
    /// Rejects existing destinations or sidecars and returns actual admission errors.
    pub fn into_stage(self, graph_model: u8) -> Result<OwnedContainerStage> {
        match fs::symlink_metadata(self.path()) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "refusing to save over existing destination",
                )
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        // Acquire C -> W before admitting an absent sidecar. Keep W until the
        // final installation, parent sync and private cleanup have completed,
        // so an independent WAL owner cannot create a replay tail for this copy.
        let sidecar = self.acquire_sidecar_wal()?;
        match fs::symlink_metadata(suffixed(self.path(), ".wal")) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "refusing to save beside an existing destination WAL",
                )
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let parent = self.path().parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "bound destination has no parent",
            )
        })?;
        let leaf = self.path().file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "bound destination has no file name",
            )
        })?;
        static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let id = NEXT_STAGE
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| io::Error::other("private container staging identity exhausted"))?;
            let mut name = std::ffi::OsString::from(".");
            name.push(leaf);
            name.push(format!(".grafeo-file-save-{}-{id}.tmp", std::process::id()));
            let path = parent.join(name);
            match fs::create_dir(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
            let directory = OwnedStageDirectory {
                path,
                present: true,
            };
            // This is the sole internal admission exception: a fixed leaf in
            // the fresh directory just minted beneath our bound destination.
            let lease =
                ContainerLease::at_resolved_path(directory.path.join("container.grafeo"), false)?;
            let manager = GrafeoFileManager::create_with_lease(lease, graph_model)?;
            return Ok(OwnedContainerStage {
                manager,
                directory,
                _sidecar: sidecar,
                destination: self,
            });
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "private container staging names are exhausted",
        )
        .into())
    }

    pub(super) fn overwrite_from(&mut self, source: &mut File, source_path: &Path) -> Result<u64> {
        use std::io::{Seek, SeekFrom};
        if self.path() == source_path {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot copy a container onto itself",
            )
            .into());
        }
        let destination = crate::ownership::checked_file(self.path(), true, true)?;
        let mut destination = LockedFile::acquire(destination, false)?;
        check_single_link(&destination)?;
        source.seek(SeekFrom::Start(0))?;
        destination.set_len(0)?;
        let bytes = io::copy(source, &mut *destination)?;
        grafeo_common::testing::crash::maybe_crash("backup:segment_write");
        destination.sync_all()?;
        drop(destination);
        sync_parent(self.path())?;
        grafeo_common::testing::crash::maybe_crash("backup:segment_sync");
        Ok(bytes)
    }
}

struct OwnedStageDirectory {
    path: PathBuf,
    present: bool,
}

impl OwnedStageDirectory {
    fn remove(&mut self) -> Result<()> {
        fs::remove_dir_all(&self.path)?;
        self.present = false;
        Ok(())
    }
}

impl Drop for OwnedStageDirectory {
    fn drop(&mut self) {
        if self.present {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

/// A narrowly owned single-file stage. Field order retires its manager before
/// private-tree cleanup and retains final W/C ownership through both, releasing
/// the sidecar WAL lease before its enclosing container lease.
pub struct OwnedContainerStage {
    manager: GrafeoFileManager,
    directory: OwnedStageDirectory,
    _sidecar: crate::wal::WalDestination,
    destination: ContainerDestination,
}

impl OwnedContainerStage {
    /// Borrows the stage manager; no independent manager can outlive this owner.
    #[must_use]
    pub fn manager(&self) -> &GrafeoFileManager {
        &self.manager
    }

    /// Publishes with the engine's existing native no-replace rename primitive.
    /// Both paths come from this capability; no caller destination is accepted.
    ///
    /// # Errors
    /// Returns close, directory-sync, install or private-cleanup errors. A failure
    /// after rename may leave the completed destination published.
    pub fn install(mut self, install: fn(&Path, &Path) -> Result<()>) -> Result<()> {
        self.manager.close()?;
        #[cfg(feature = "testing-crash-injection")]
        super::manager::ownership_test_point("stage-before-install")?;
        sync_parent(self.manager.path())?;
        install(self.manager.path(), self.destination.path())?;
        let sync_installed_parent = || {
            // The final file has moved out of the private directory. Retain
            // destination ownership across both observation and parent sync;
            // any failure may retire only the remaining private stage.
            #[cfg(feature = "testing-crash-injection")]
            super::manager::ownership_test_point("stage-after-install")?;
            #[cfg(feature = "testing-crash-injection")]
            super::manager::ownership_test_point("stage-parent-sync")?;
            sync_parent(self.destination.path())
        };
        sync_installed_parent().map_err(|error| {
            Error::Transaction(grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                ".grafeo save was published at {} but syncing its parent directory failed: {error}", self.destination.path().display()
            )))
        })?;
        self.directory.remove().map_err(|error| error.with_context(format!(
            ".grafeo save was published and its parent sync completed at {}; private stage cleanup failed at {}",
            self.destination.path().display(), self.directory.path.display()
        )))?;
        Ok(())
    }
}

// Final W precedes C in drop order; both remain owned through private cleanup.
pub(crate) struct ContainerRestoreContext {
    path: PathBuf,
    _sidecar: crate::wal::WalDestination,
    destination: ContainerDestination,
}

fn remove_known_file(path: &Path) {
    if let Ok(file) = crate::ownership::checked_file(path, false, false) {
        drop(file);
        let _ = fs::remove_file(path);
    }
}

fn clean_restore_wal(path: &Path) {
    if !fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_dir()) {
        return;
    }
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let segment = name
                .strip_prefix("wal_")
                .and_then(|name| name.strip_suffix(".log"))
                .is_some_and(|sequence| {
                    sequence.len() >= 8
                        && sequence.bytes().all(|byte| byte.is_ascii_digit())
                        && sequence
                            .parse::<u64>()
                            .is_ok_and(|value| format!("{value:08}") == sequence)
                });
            if segment || matches!(name, "checkpoint.meta" | "checkpoint.meta.tmp") {
                remove_known_file(&entry.path());
            }
        }
    }
    // Unknown files, unsafe links and subtrees prevent removal and are preserved.
    let _ = fs::remove_dir(path);
}

impl ContainerRestoreContext {
    pub(crate) fn retire_wal(&self, path: &Path) -> Result<()> {
        if ![
            "import.wal",
            "container.grafeo.wal",
            "validation.grafeo.wal",
        ]
        .iter()
        .any(|name| path == self.path.join(name))
        {
            return Err(Error::InvalidValue("foreign restore WAL retirement".into()));
        }
        clean_restore_wal(path);
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
            Ok(_) => Err(io::Error::other(
                "private restore WAL contains unknown or unsafe artifacts; preserved",
            )
            .into()),
        }
    }
}

impl Drop for ContainerRestoreContext {
    fn drop(&mut self) {
        for name in [
            "import.wal",
            "container.grafeo.wal",
            "validation.grafeo.wal",
        ] {
            clean_restore_wal(&self.path.join(name));
            remove_known_file(&self.path.join(format!("{name}.grafeo-owner")));
        }
        for name in ["container.grafeo", "validation.grafeo"] {
            for suffix in ["", ".installing", ".grafeo-owner"] {
                remove_known_file(&self.path.join(format!("{name}{suffix}")));
            }
        }
        let _ = fs::remove_dir(&self.path);
    }
}

/// Detached restore images whose children retain final container and WAL ownership.
/// Public pathname opens cannot enter its reserved private directory.
pub struct ContainerRestoreStage {
    context: Arc<ContainerRestoreContext>,
    imported: bool,
    output_copied: bool,
}

impl ContainerDestination {
    /// Creates private restore images while retaining final C and W authority.
    /// Existing regular single-link primary bytes are preserved until install.
    ///
    /// # Errors
    /// Rejects unsafe primary files, any existing sidecar, or stage creation errors.
    pub fn into_restore_stage(self) -> Result<ContainerRestoreStage> {
        match crate::ownership::checked_file(self.path(), false, false) {
            Ok(file) => drop(file),
            Err(Error::Io(error)) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let sidecar = self.acquire_sidecar_wal()?;
        match fs::symlink_metadata(sidecar.path()) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "refusing to restore beside an existing destination WAL",
                )
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let parent = self
            .path()
            .parent()
            .ok_or_else(|| io::Error::other("restore destination has no parent"))?;
        let leaf = self
            .path()
            .file_name()
            .ok_or_else(|| io::Error::other("restore destination has no leaf"))?;
        static NEXT_RESTORE: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let id = NEXT_RESTORE
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| io::Error::other("private restore identity exhausted"))?;
            let mut name = std::ffi::OsString::from(".");
            name.push(leaf);
            name.push(format!(
                ".grafeo-file-save-restore-{}-{id}.tmp",
                std::process::id()
            ));
            let path = parent.join(name);
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(ContainerRestoreStage {
                        context: Arc::new(ContainerRestoreContext {
                            path,
                            _sidecar: sidecar,
                            destination: self,
                        }),
                        imported: false,
                        output_copied: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "private restore names exhausted",
        )
        .into())
    }
}

impl ContainerRestoreStage {
    fn container_lease(&self, name: &str, shared: bool) -> Result<ContainerLease> {
        let mut lease = ContainerLease::at_resolved_path(self.context.path.join(name), shared)?;
        lease.restore = Some(Arc::clone(&self.context));
        Ok(lease)
    }

    fn copy_image(&self, name: &str, source: &mut super::ContainerCapture<'_>) -> Result<u64> {
        let lease = self.container_lease(name, false)?;
        match fs::symlink_metadata(lease.path()) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "restore image already exists",
                )
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        source.copy_to_destination(&mut ContainerDestination { lease })
    }

    fn wal_child(&self, name: &str) -> Result<crate::wal::WalDestination> {
        let path = self.context.path.join(name);
        let child =
            crate::wal::WalDestination::restore_child(path.clone(), Arc::clone(&self.context))?;
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "private restore WAL is not a directory",
                )
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => fs::create_dir(&path)?,
            Err(error) => return Err(error.into()),
        }
        Ok(child)
    }

    /// Copies the selected full image from an already admitted source.
    ///
    /// # Errors
    /// Rejects repeat copies and returns checked copy or sync errors.
    pub fn copy_from(&mut self, source: &mut super::ContainerCapture<'_>) -> Result<u64> {
        let copied = self.copy_image("container.grafeo", source)?;
        self.output_copied = true;
        Ok(copied)
    }

    /// Copies a second image for validating a later complete backup cut.
    ///
    /// # Errors
    /// Rejects repeat copies and returns checked copy or sync errors.
    pub fn copy_validation_from(
        &mut self,
        source: &mut super::ContainerCapture<'_>,
    ) -> Result<u64> {
        self.copy_image("validation.grafeo", source)
    }

    /// Opens the output image, retaining final authority until this child closes.
    ///
    /// # Errors
    /// Returns private-image contention, format or filesystem errors.
    pub fn open(&self) -> Result<GrafeoFileManager> {
        GrafeoFileManager::open_with_lease(self.container_lease("container.grafeo", false)?)
    }

    /// Opens the output image read-only through its private capability.
    ///
    /// # Errors
    /// Returns private-image contention, format or filesystem errors.
    pub fn open_read_only(&self) -> Result<GrafeoFileManager> {
        GrafeoFileManager::open_read_only_with_lease(
            self.container_lease("container.grafeo", true)?,
        )
    }

    /// Opens the validation image through its private capability.
    ///
    /// # Errors
    /// Returns private-image contention, format or filesystem errors.
    pub fn open_validation(&self) -> Result<GrafeoFileManager> {
        GrafeoFileManager::open_with_lease(self.container_lease("validation.grafeo", false)?)
    }

    /// Mints the one fixed import WAL for checked raw segment admission.
    ///
    /// # Errors
    /// Rejects repeated minting, contention and filesystem errors.
    pub fn import(&mut self) -> Result<crate::wal::WalImport> {
        if self.imported {
            return Err(Error::InvalidValue("restore import already minted".into()));
        }
        let child = self.wal_child("import.wal")?;
        #[cfg(feature = "testing-crash-injection")]
        super::manager::ownership_test_point("restore-import")?;
        self.imported = true;
        Ok(child.into_container_import())
    }

    /// Opens or reopens the output image's fixed replay WAL.
    ///
    /// # Errors
    /// Rejects live replay owners and unsafe filesystem artifacts.
    pub fn replay(&self) -> Result<crate::wal::WalRecovery> {
        Ok(self
            .wal_child("container.grafeo.wal")?
            .into_container_recovery())
    }

    /// Opens or reopens the validation image's fixed replay WAL.
    ///
    /// # Errors
    /// Rejects live replay owners and unsafe filesystem artifacts.
    pub fn validation_replay(&self) -> Result<crate::wal::WalRecovery> {
        Ok(self
            .wal_child("validation.grafeo.wal")?
            .into_container_recovery())
    }

    /// Atomically replaces the final primary after all children have closed.
    /// The output replay WAL must have been materialized and retired first.
    ///
    /// # Errors
    /// Rejects live descendants or an unretired replay WAL. Returns checked
    /// file/sync/rename errors; failures after rename may leave the complete new
    /// image installed, while preserving final ownership through cleanup.
    pub fn install(self) -> Result<()> {
        use grafeo_common::testing::{
            crash::maybe_crash, wal_failure::check_backup_publication_failure,
        };
        if !self.output_copied {
            return Err(Error::InvalidValue(
                "restore output image was not completely copied".into(),
            ));
        }
        if Arc::strong_count(&self.context) != 1 {
            return Err(Error::InvalidValue(
                "restore descendants must close before installation".into(),
            ));
        }
        let replay = self.context.path.join("container.grafeo.wal");
        match fs::symlink_metadata(&replay) {
            Ok(_) => {
                return Err(Error::InvalidValue(
                    "restore replay WAL must be materialized and retired before installation"
                        .into(),
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let staged = self.context.path.join("container.grafeo");
        let file = crate::ownership::checked_file(&staged, true, false)?;
        file.sync_all()?;
        drop(file);
        maybe_crash("backup:restore_before_replace");
        check_backup_publication_failure("backup:restore_before_replace")?;
        #[cfg(feature = "testing-crash-injection")]
        super::manager::ownership_test_point("restore-before-install")?;
        fs::rename(&staged, self.context.destination.path())?;
        let synced = (|| -> Result<()> {
            #[cfg(feature = "testing-crash-injection")]
            super::manager::ownership_test_point("restore-after-install")?;
            maybe_crash("backup:restore_after_replace");
            check_backup_publication_failure("backup:restore_after_replace")?;
            check_backup_publication_failure("backup:restore_parent_sync")?;
            #[cfg(feature = "testing-crash-injection")]
            super::manager::ownership_test_point("restore-parent-sync")?;
            sync_parent(self.context.destination.path())?;
            maybe_crash("backup:restore_parent_sync");
            Ok(())
        })();
        synced.map_err(|error| Error::Transaction(
            grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                ".grafeo restore was published at {} but its durability acknowledgement failed: {error}",
                self.context.destination.path().display(),
            )),
        ))
    }
}

#[cfg(test)]
mod restore_tests {
    use super::*;

    fn image(path: &Path, bytes: &[u8]) -> GrafeoFileManager {
        let manager = GrafeoFileManager::create(path).unwrap();
        manager.write_snapshot(bytes, 1, 1, 0, 0).unwrap();
        manager
    }

    #[test]
    fn restore_stage_children_retain_final_authority_and_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let source = image(&dir.path().join("source.grafeo"), b"new");
        let output = dir.path().join("output.grafeo");
        let mut stage = ContainerDestination::acquire(&output)
            .unwrap()
            .into_restore_stage()
            .unwrap();
        stage.copy_from(&mut source.capture().unwrap()).unwrap();
        let child = stage.open().unwrap();
        let private = child.path().parent().unwrap().to_path_buf();
        assert!(GrafeoFileManager::open(child.path()).is_err());
        assert!(super::super::open_backup_source(child.path()).is_err());
        drop(stage);
        assert!(private.exists());
        assert!(ContainerDestination::acquire(&output).is_err());
        assert!(crate::wal::WalDestination::acquire(suffixed(&output, ".wal")).is_err());
        assert_eq!(child.read_snapshot().unwrap(), b"new");
        drop(child);
        assert!(!private.exists());
        assert!(ContainerDestination::acquire(&output).is_ok());
    }

    #[test]
    fn restore_stage_install_replaces_only_after_descendants_close() {
        let dir = tempfile::tempdir().unwrap();
        let source = image(&dir.path().join("source.grafeo"), b"new");
        let output = dir.path().join("output.grafeo");
        drop(image(&output, b"old"));
        let before = fs::read(&output).unwrap();
        let mut stage = ContainerDestination::acquire(&output)
            .unwrap()
            .into_restore_stage()
            .unwrap();
        stage.copy_from(&mut source.capture().unwrap()).unwrap();
        assert_eq!(fs::read(&output).unwrap(), before);
        let child = stage.open().unwrap();
        assert!(stage.install().is_err());
        assert!(ContainerDestination::acquire(&output).is_err());
        drop(child);
        let mut stage = ContainerDestination::acquire(&output)
            .unwrap()
            .into_restore_stage()
            .unwrap();
        stage.copy_from(&mut source.capture().unwrap()).unwrap();
        stage.install().unwrap();
        assert_eq!(
            GrafeoFileManager::open_read_only(&output)
                .unwrap()
                .read_snapshot()
                .unwrap(),
            b"new"
        );
    }

    #[test]
    fn restore_stage_replay_reopens_and_validation_has_distinct_ownership() {
        let dir = tempfile::tempdir().unwrap();
        let source = image(&dir.path().join("source.grafeo"), b"new");
        let output = dir.path().join("output.grafeo");
        let mut stage = ContainerDestination::acquire(&output)
            .unwrap()
            .into_restore_stage()
            .unwrap();
        stage.copy_from(&mut source.capture().unwrap()).unwrap();
        stage
            .copy_validation_from(&mut source.capture().unwrap())
            .unwrap();
        let child = stage.open().unwrap();
        let validation = stage.open_validation().unwrap();
        let mut replay = stage.replay().unwrap();
        assert!(stage.replay().is_err());
        replay.recover().unwrap();
        drop(replay.seal().unwrap());
        let mut replay = stage.replay().unwrap();
        replay.recover().unwrap();
        let mut seal = replay.seal().unwrap();
        child
            .sidecar_retirement()
            .unwrap()
            .retire_sidecar(&mut seal)
            .unwrap();
        drop(seal);
        let mut validation_replay = stage.validation_replay().unwrap();
        validation_replay.recover().unwrap();
        drop(validation_replay.seal().unwrap());
        let imported = stage.import().unwrap();
        assert!(stage.import().is_err());
        drop(imported);
        child.close().unwrap();
        validation.close().unwrap();
        stage.install().unwrap();
    }

    #[test]
    fn restore_stage_rejects_sidecars_without_mutating_primary() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("output.grafeo");
        drop(image(&output, b"old"));
        let before = fs::read(&output).unwrap();
        let sidecar = suffixed(&output, ".wal");
        fs::create_dir(&sidecar).unwrap();
        fs::write(sidecar.join("unowned"), b"preserve").unwrap();
        assert!(
            ContainerDestination::acquire(&output)
                .unwrap()
                .into_restore_stage()
                .is_err()
        );
        assert_eq!(fs::read(output).unwrap(), before);
        assert_eq!(fs::read(sidecar.join("unowned")).unwrap(), b"preserve");
    }

    #[test]
    fn restore_stage_cleanup_preserves_unknown_files_and_subtrees() {
        let dir = tempfile::tempdir().unwrap();
        let source = image(&dir.path().join("source.grafeo"), b"new");
        let output = dir.path().join("output.grafeo");
        let mut stage = ContainerDestination::acquire(output)
            .unwrap()
            .into_restore_stage()
            .unwrap();
        stage.copy_from(&mut source.capture().unwrap()).unwrap();
        let private = stage.context.path.clone();
        let replay = stage.replay().unwrap();
        let unknown_wal = replay.path().join("unowned");
        fs::create_dir(&unknown_wal).unwrap();
        fs::write(unknown_wal.join("keep"), b"keep").unwrap();
        fs::write(private.join("unknown"), b"preserve").unwrap();
        drop(stage);
        assert!(private.join("container.grafeo").exists());
        drop(replay);
        assert!(!private.join("container.grafeo").exists());
        assert_eq!(fs::read(private.join("unknown")).unwrap(), b"preserve");
        assert_eq!(fs::read(unknown_wal.join("keep")).unwrap(), b"keep");
    }

    #[cfg(unix)]
    #[test]
    fn backup_source_rejects_links_and_read_only_manager_rejects_hardlinks() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.grafeo");
        drop(image(&source, b"source"));
        let link = dir.path().join("link.grafeo");
        std::os::unix::fs::symlink(&source, &link).unwrap();
        assert!(GrafeoFileManager::open_read_only(&link).is_ok());
        assert!(super::super::open_backup_source(&link).is_err());
        fs::remove_file(&link).unwrap();
        fs::hard_link(&source, &link).unwrap();
        assert!(GrafeoFileManager::open_read_only(&source).is_err());
        assert!(super::super::open_backup_source(&source).is_err());
    }
}
