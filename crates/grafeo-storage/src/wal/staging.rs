//! Fixed-purpose publication capabilities. No public private-path admission.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use super::WalRecovery;
use super::ownership::{SealedWal, WalDestination, WalLease, remove_owned_directory};
use crate::ownership::{checked_file, sync_parent};
use grafeo_common::utils::error::{Error, Result};

// The outer lease lives through last-reference private cleanup. Child W FDs
// precede their Arc in WalLease, so cleanup cannot remove a live child's files.
pub(super) struct StageContext {
    path: PathBuf,
    published: AtomicBool,
    outer: WalLease,
}

impl Drop for StageContext {
    fn drop(&mut self) {
        if !self.published.load(Ordering::Acquire) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn fresh_stage(outer: WalLease) -> Result<Arc<StageContext>> {
    let parent = outer
        .path()
        .parent()
        .ok_or_else(|| io::Error::other("destination has no parent"))?;
    let leaf = outer
        .path()
        .file_name()
        .ok_or_else(|| io::Error::other("destination has no leaf"))?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    for _ in 0..128 {
        let id = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| io::Error::other("private WAL staging identity exhausted"))?;
        let mut name = std::ffi::OsString::from(".");
        name.push(leaf);
        name.push(format!(".grafeo-save-{}-{id}", std::process::id()));
        let path = parent.join(name);
        match fs::create_dir(&path) {
            Ok(()) => {
                return Ok(Arc::new(StageContext {
                    path,
                    outer,
                    published: AtomicBool::new(false),
                }));
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "private WAL stage names exhausted",
    )
    .into())
}

fn child(context: &Arc<StageContext>, name: &str) -> Result<WalLease> {
    let path = context.path.join(name);
    fs::create_dir(&path)?;
    WalLease::at_resolved_path(path, Some(Arc::clone(context)))
}

fn require_child(context: &Arc<StageContext>, wal: &SealedWal, name: &str) -> Result<()> {
    if wal.path() != context.path.join(name)
        || !wal
            .lease
            .context
            .as_ref()
            .is_some_and(|c| Arc::ptr_eq(c, context))
    {
        return Err(Error::InvalidValue(
            "publication requires this exact private WAL child seal".into(),
        ));
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Private import and trim staging retaining the permanent final W coordinate.
pub struct WalRestoreStage {
    context: Arc<StageContext>,
    imported: bool,
    trimmed: bool,
}

impl WalDestination {
    /// Mints a fresh restore context without reopening the final WAL.
    ///
    /// # Errors
    /// Returns fresh-stage construction errors.
    pub fn into_restore_stage(self) -> Result<WalRestoreStage> {
        Ok(WalRestoreStage {
            context: fresh_stage(self.lease)?,
            imported: false,
            trimmed: false,
        })
    }
}

impl WalRestoreStage {
    /// Mints the single fixed import child, at most once.
    ///
    /// # Errors
    /// Rejects repeated minting and construction errors.
    pub fn import(&mut self) -> Result<WalImport> {
        if self.imported {
            return Err(Error::InvalidValue("restore import already minted".into()));
        }
        self.imported = true;
        let lease = child(&self.context, "import.wal")?;
        test_point("restore-import")?;
        Ok(WalImport {
            lease,
            failed: false,
        })
    }

    /// Mints the single empty trim recovery, at most once.
    ///
    /// # Errors
    /// Rejects repeated minting and construction errors.
    pub fn trim_recovery(&mut self) -> Result<WalRecovery> {
        if self.trimmed {
            return Err(Error::InvalidValue("restore trim already minted".into()));
        }
        self.trimmed = true;
        let lease = child(&self.context, "trim.wal")?;
        test_point("restore-trim")?;
        Ok(WalRecovery::from_lease(lease))
    }

    /// Installs only these exact import/trim seals under the final W lease.
    ///
    /// # Errors
    /// Returns seal validation, removal, rename, sync or cleanup errors.
    pub fn install(
        self,
        imported: SealedWal,
        trimmed: SealedWal,
        rename: fn(&Path, &Path) -> Result<()>,
    ) -> Result<()> {
        require_child(&self.context, &imported, "import.wal")?;
        require_child(&self.context, &trimmed, "trim.wal")?;
        sync_directory(trimmed.path())?;
        test_point("restore-before-remove")?;
        remove_owned_directory(self.context.outer.path())?;
        test_point("restore-before-rename")?;
        rename(trimmed.path(), self.context.outer.path())?;
        test_point("restore-after-rename")?;
        sync_parent(self.context.outer.path())?;
        // Both private coordination entries stay in the private tree. The
        // permanent final coordinate never moves or unlinks.
        drop(imported);
        drop(trimmed);
        fs::remove_dir_all(&self.context.path)?;
        self.context.published.store(true, Ordering::Release);
        Ok(())
    }
}

/// One checked fixed-sequence importer, consumed directly into recovery.
pub struct WalImport {
    lease: WalLease,
    failed: bool,
}

impl WalImport {
    #[cfg(feature = "grafeo-file")]
    pub(crate) fn for_container_restore(lease: WalLease) -> Self {
        Self {
            lease,
            failed: false,
        }
    }

    /// Writes a fresh fixed-sequence segment through its checked owned handle.
    ///
    /// # Errors
    /// Rejects duplicate segments and retains failed import authority on I/O error.
    pub fn write_segment(&mut self, sequence: u64, bytes: &[u8]) -> Result<()> {
        if self.failed {
            return Err(Error::InvalidValue("restore import failed".into()));
        }
        let path = self.lease.path().join(format!("wal_{sequence:08}.log"));
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "restore segment already exists",
                )
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        self.failed = true;
        let mut file = checked_file(&path, true, true)?;
        file.write_all(bytes)?;
        test_point("restore-import-after-write")?;
        file.sync_all()?;
        self.failed = false;
        Ok(())
    }

    /// Transfers this child's W directly into recovery without a public reopen.
    ///
    /// # Errors
    /// Rejects an importer after a physical failure.
    pub fn into_recovery(self) -> Result<WalRecovery> {
        if self.failed {
            return Err(Error::InvalidValue("restore import failed".into()));
        }
        Ok(WalRecovery::from_lease(self.lease))
    }
}

fn test_point(point: &str) -> Result<()> {
    #[cfg(all(test, feature = "testing-crash-injection"))]
    assert!(
        std::env::var_os("GRAFEO_WAL_STAGE_UNWIND").as_deref() != Some(std::ffi::OsStr::new(point)),
        "injected WAL stage unwind at {point}"
    );
    #[cfg(feature = "testing-crash-injection")]
    {
        if std::env::var_os("GRAFEO_WAL_STAGE_FAIL").as_deref() == Some(std::ffi::OsStr::new(point))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("injected WAL stage failure at {point}"),
            )
            .into());
        }
        if std::env::var("GRAFEO_WAL_STAGE_RENDEZVOUS")
            .is_ok_and(|points| points.split(',').any(|value| value == point))
        {
            println!("READY");
            io::stdout().flush()?;
            let mut response = String::new();
            io::stdin().read_line(&mut response)?;
            if response.trim() != "RELEASE" {
                return Err(io::Error::other("WAL stage rendezvous requires RELEASE").into());
            }
        }
    }
    #[cfg(not(feature = "testing-crash-injection"))]
    let _ = point;
    Ok(())
}

#[cfg(all(test, feature = "testing-crash-injection"))]
mod tests {
    use super::*;

    fn assert_owned(path: &Path) {
        assert!(matches!(
            WalDestination::acquire(path),
            Err(Error::Io(error)) if error.kind() == io::ErrorKind::WouldBlock
        ));
    }

    fn run_child(mode: &str, failure: &str, unwind: &str) {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "wal::staging::tests::private_stage_failure_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("GRAFEO_PRIVATE_STAGE_CHILD", mode)
            .env("GRAFEO_WAL_STAGE_FAIL", failure)
            .env("GRAFEO_WAL_STAGE_UNWIND", unwind)
            .output()
            .unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        assert!(
            output.status.success(),
            "private stage child: {}",
            output.status
        );
    }

    #[test]
    fn private_stage_collision_and_child_failure_preserve_unowned_tree() {
        run_child("collision", "restore-import", "");
    }

    #[test]
    fn import_physical_error_latches_and_retains_final_w() {
        run_child("error", "restore-import-after-write", "");
    }

    #[test]
    fn import_physical_unwind_latches_and_retains_final_w() {
        run_child("unwind", "", "restore-import-after-write");
    }

    #[test]
    fn private_stage_failure_child() {
        let Ok(mode) = std::env::var("GRAFEO_PRIVATE_STAGE_CHILD") else {
            return;
        };
        let root = tempfile::tempdir().unwrap();
        if mode == "collision" {
            let final_path = root.path().join("output.wal");
            // Fresh child process: NEXT's first candidate is exactly zero.
            let stale = root
                .path()
                .join(format!(".output.wal.grafeo-save-{}-0", std::process::id()));
            fs::create_dir(&stale).unwrap();
            fs::write(stale.join("unowned-sentinel"), b"do not adopt or erase").unwrap();
            let mut stage = WalDestination::acquire(&final_path)
                .unwrap()
                .into_restore_stage()
                .unwrap();
            let minted = stage.context.path.clone();
            assert_eq!(
                minted,
                fs::canonicalize(root.path())
                    .unwrap()
                    .join(format!(".output.wal.grafeo-save-{}-1", std::process::id()))
            );
            let Err(error) = stage.import() else {
                panic!("import construction must fail after lease acquisition");
            };
            assert!(
                matches!(error, Error::Io(ref error) if error.kind() == io::ErrorKind::PermissionDenied)
            );
            assert!(
                error
                    .to_string()
                    .contains("injected WAL stage failure at restore-import")
            );
            // The child's actual coordinate was minted before the failure.
            assert!(minted.join("import.wal.grafeo-owner").is_file());
            assert!(minted.join("import.wal").is_dir());
            assert_owned(&final_path);
            assert!(stage.import().is_err());
            let trim = stage.trim_recovery().unwrap();
            drop(stage);
            assert!(minted.exists());
            assert_owned(&final_path);
            assert_eq!(
                fs::read(stale.join("unowned-sentinel")).unwrap(),
                b"do not adopt or erase"
            );
            drop(trim);
            assert!(!minted.exists());
            assert_eq!(
                fs::read(stale.join("unowned-sentinel")).unwrap(),
                b"do not adopt or erase"
            );
            let _released = WalDestination::acquire(&final_path).unwrap();
            return;
        }

        assert!(mode == "error" || mode == "unwind");
        for stage_first in [false, true] {
            let final_path = root.path().join(format!("output-{stage_first}.wal"));
            let mut stage = Some(
                WalDestination::acquire(&final_path)
                    .unwrap()
                    .into_restore_stage()
                    .unwrap(),
            );
            let minted = stage.as_ref().unwrap().context.path.clone();
            let mut import = stage.as_mut().unwrap().import().unwrap();
            let trim = stage.as_mut().unwrap().trim_recovery().unwrap();
            let imported_file = import.lease.path().join("wal_00000000.log");
            if mode == "error" {
                let error = import
                    .write_segment(0, b"accepted once before sync")
                    .unwrap_err();
                assert!(
                    matches!(error, Error::Io(ref error) if error.kind() == io::ErrorKind::PermissionDenied)
                );
                assert!(
                    error
                        .to_string()
                        .contains("injected WAL stage failure at restore-import-after-write")
                );
            } else {
                let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    import.write_segment(0, b"accepted once before sync")
                }))
                .expect_err("physical import seam must unwind");
                let message = unwind
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| unwind.downcast_ref::<&str>().copied());
                assert_eq!(
                    message,
                    Some("injected WAL stage unwind at restore-import-after-write")
                );
            }
            assert!(import.failed);
            assert_eq!(
                fs::read(&imported_file).unwrap(),
                b"accepted once before sync"
            );
            assert!(matches!(
                import.write_segment(1, b"must not retry"),
                Err(Error::InvalidValue(_))
            ));
            assert!(!import.lease.path().join("wal_00000001.log").exists());
            if stage_first {
                drop(stage.take());
            }
            assert_owned(&final_path);
            assert!(matches!(
                import.into_recovery(),
                Err(Error::InvalidValue(_))
            ));
            assert_eq!(
                fs::read(&imported_file).unwrap(),
                b"accepted once before sync"
            );
            assert_owned(&final_path);
            assert!(minted.exists());
            drop(trim);
            if !stage_first {
                assert!(minted.exists());
                assert_eq!(
                    fs::read(&imported_file).unwrap(),
                    b"accepted once before sync"
                );
                assert_owned(&final_path);
                drop(stage.take());
            }
            assert!(!minted.exists());
            let _released = WalDestination::acquire(&final_path).unwrap();
        }
    }
}
