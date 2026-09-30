//! Atomic, no-clobber publication of a completed filesystem object.
//!
//! A preflight existence check is never sufficient: another process can create
//! the destination between that check and a replacing rename. Supported Unix
//! kernels expose an atomic rename-with-no-replace operation, which is the only
//! primitive used here. Other targets fail closed until an equivalent safe
//! platform implementation is available.

use std::path::Path;

use grafeo_common::utils::error::{Error, Result};

fn absolute_pair(
    source: &Path,
    destination: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    if source.is_absolute() && destination.is_absolute() {
        return Ok((source.to_owned(), destination.to_owned()));
    }
    let working_directory = std::env::current_dir()?;
    let absolute = |path: &Path| {
        if path.is_absolute() {
            path.to_owned()
        } else {
            working_directory.join(path)
        }
    };
    Ok((absolute(source), absolute(destination)))
}

/// Atomically renames `source` to `destination` without replacing any existing
/// filesystem object at `destination`.
///
/// On failure the source directory remains at its original path. Apple and
/// Linux kernels provide the required operation directly. Unsupported targets
/// return an error rather than degrading to a check-then-rename race.
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos",
    target_os = "watchos",
))]
pub(super) fn rename_directory_noreplace(source: &Path, destination: &Path) -> Result<()> {
    use rustix::fs::{CWD, RenameFlags, renameat_with};
    use rustix::io::Errno;

    let (source, destination) = absolute_pair(source, destination)?;
    match renameat_with(CWD, &source, CWD, &destination, RenameFlags::NOREPLACE) {
        Ok(()) => Ok(()),
        Err(error) if error == Errno::EXIST || error == Errno::NOTEMPTY => {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "refusing to replace existing destination {}",
                    destination.display()
                ),
            )))
        }
        Err(error)
            if error == Errno::NOSYS
                || error == Errno::INVAL
                || error == Errno::NOTSUP
                || error == Errno::OPNOTSUPP =>
        {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "the filesystem does not support atomic no-clobber directory installation for {}",
                    destination.display()
                ),
            )))
        }
        Err(error) => Err(std::io::Error::from_raw_os_error(error.raw_os_error()).into()),
    }
}

/// Windows' standard-library rename requests replacement. Call the underlying
/// `MoveFileExW` operation with no flags instead: flags=0 is an atomic move that
/// refuses an existing destination and requires the same volume (the staging
/// directory is deliberately a sibling, so that condition always holds).
#[cfg(windows)]
#[allow(unsafe_code)]
pub(super) fn rename_directory_noreplace(source: &Path, destination: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{
        ERROR_ACCESS_DENIED, ERROR_ALREADY_EXISTS, ERROR_FILE_EXISTS,
    };
    use windows_sys::Win32::Storage::FileSystem::MoveFileExW;

    fn nul_terminated(path: &Path) -> Result<Vec<u16>> {
        let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        if wide.contains(&0) {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "filesystem path contains an interior NUL",
            )));
        }
        wide.push(0);
        Ok(wide)
    }

    let (source, destination) = absolute_pair(source, destination)?;
    let source = nul_terminated(&source)?;
    let destination_wide = nul_terminated(&destination)?;
    // SAFETY: both pointers reference live, NUL-terminated UTF-16 buffers for
    // the duration of the call. Flags=0 requests neither replacement nor a
    // delayed reboot move. MoveFileExW does not retain either pointer.
    if unsafe { MoveFileExW(source.as_ptr(), destination_wide.as_ptr(), 0) } != 0 {
        return Ok(());
    }

    let error = std::io::Error::last_os_error();
    let raw = error.raw_os_error().map(|code| code as u32);
    let destination_exists = std::fs::symlink_metadata(&destination).is_ok();
    if matches!(raw, Some(ERROR_FILE_EXISTS | ERROR_ALREADY_EXISTS))
        || (raw == Some(ERROR_ACCESS_DENIED) && destination_exists)
    {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "refusing to replace existing destination {}",
                destination.display()
            ),
        )));
    }
    Err(error.into())
}

#[cfg(not(any(
    target_os = "android",
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos",
    target_os = "watchos",
    windows,
)))]
pub(super) fn rename_directory_noreplace(_source: &Path, destination: &Path) -> Result<()> {
    Err(Error::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!(
            "atomic no-clobber directory installation is not supported on this platform for {}",
            destination.display()
        ),
    )))
}

/// Atomically installs a completed file without replacing any existing
/// filesystem object at `destination`.
///
/// The underlying platform primitives are path-generic (`renameat2` /
/// `renameatx_np` / `MoveFileExW`), so the same implementation safely publishes
/// both staged directory trees and staged single-file containers.
#[cfg(feature = "wal")]
pub(super) fn rename_file_noreplace(source: &Path, destination: &Path) -> Result<()> {
    rename_directory_noreplace(source, destination)
}

#[cfg(all(
    test,
    any(
        target_os = "android",
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos",
        target_os = "watchos",
        windows,
    )
))]
mod tests {
    use super::rename_directory_noreplace;
    use grafeo_common::utils::error::Error;
    use std::fs;

    #[cfg(feature = "wal")]
    #[test]
    fn file_wrapper_installs_once_without_replacing_destination() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        let destination = root.path().join("destination");
        fs::write(&source, b"first").unwrap();
        super::rename_file_noreplace(&source, &destination).unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"first");
        fs::write(&source, b"second").unwrap();
        assert!(matches!(
            super::rename_file_noreplace(&source, &destination),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(fs::read(&source).unwrap(), b"second");
        assert_eq!(fs::read(&destination).unwrap(), b"first");
    }

    fn source_tree(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let source = root.join(name);
        fs::create_dir(&source).unwrap();
        fs::write(source.join("sentinel"), name.as_bytes()).unwrap();
        source
    }

    #[test]
    fn installs_complete_tree_when_destination_is_absent() {
        let root = tempfile::tempdir().unwrap();
        let source = source_tree(root.path(), "staged");
        let destination = root.path().join("published");

        rename_directory_noreplace(&source, &destination).unwrap();

        assert!(!source.exists());
        assert_eq!(fs::read(destination.join("sentinel")).unwrap(), b"staged");
    }

    #[test]
    fn refuses_every_existing_destination_kind_and_retains_source() {
        for kind in ["file", "empty-dir", "non-empty-dir", "symlink"] {
            let root = tempfile::tempdir().unwrap();
            let source = source_tree(root.path(), "staged");
            let destination = root.path().join("published");

            match kind {
                "file" => fs::write(&destination, b"owner").unwrap(),
                "empty-dir" => fs::create_dir(&destination).unwrap(),
                "non-empty-dir" => {
                    fs::create_dir(&destination).unwrap();
                    fs::write(destination.join("owner"), b"owner").unwrap();
                }
                "symlink" => {
                    #[cfg(unix)]
                    std::os::unix::fs::symlink(root.path().join("missing"), &destination).unwrap();
                    #[cfg(windows)]
                    if let Err(error) = std::os::windows::fs::symlink_file(
                        root.path().join("missing"),
                        &destination,
                    ) {
                        if error.kind() == std::io::ErrorKind::PermissionDenied {
                            continue;
                        }
                        panic!("create destination symlink: {error}");
                    }
                }
                _ => unreachable!(),
            }

            let error = rename_directory_noreplace(&source, &destination).unwrap_err();
            let Error::Io(error) = error else {
                panic!("expected I/O error for {kind}");
            };
            assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
            assert!(source.join("sentinel").is_file(), "source lost for {kind}");
            match kind {
                "file" => assert_eq!(fs::read(&destination).unwrap(), b"owner"),
                "empty-dir" => assert!(destination.is_dir()),
                "non-empty-dir" => {
                    assert_eq!(fs::read(destination.join("owner")).unwrap(), b"owner");
                }
                "symlink" => assert_eq!(
                    fs::read_link(&destination).unwrap(),
                    root.path().join("missing")
                ),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn missing_source_does_not_create_destination() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("missing");
        let destination = root.path().join("published");

        let Error::Io(error) = rename_directory_noreplace(&source, &destination).unwrap_err()
        else {
            panic!("expected I/O error");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        assert!(!destination.exists());
    }

    #[test]
    fn concurrent_publishers_have_exactly_one_winner() {
        use std::sync::{Arc, Barrier};

        let root = tempfile::tempdir().unwrap();
        let first = source_tree(root.path(), "first");
        let second = source_tree(root.path(), "second");
        let destination = root.path().join("published");
        let barrier = Arc::new(Barrier::new(3));

        let spawn = |source: std::path::PathBuf| {
            let destination = destination.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                (
                    source.clone(),
                    rename_directory_noreplace(&source, &destination),
                )
            })
        };
        let first_thread = spawn(first);
        let second_thread = spawn(second);
        barrier.wait();

        let outcomes = [first_thread.join().unwrap(), second_thread.join().unwrap()];
        assert_eq!(
            outcomes
                .iter()
                .filter(|(_, outcome)| outcome.is_ok())
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|(_, outcome)| outcome.is_err())
                .count(),
            1
        );
        for (source, outcome) in outcomes {
            assert_eq!(source.exists(), outcome.is_err());
        }
        let published = fs::read(destination.join("sentinel")).unwrap();
        assert!(published == b"first" || published == b"second");
    }
}
