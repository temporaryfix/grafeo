//! Shared canonical namespace and checked artifact primitives.

use fs2::FileExt;
use grafeo_common::utils::error::{Error, Result};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Component, Path, PathBuf};

pub(crate) fn require_regular(metadata: &fs::Metadata) -> Result<()> {
    if !metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "container coordination must be a regular non-symlink file",
        )
        .into());
    }
    Ok(())
}

/// Owns one lock acquisition and its descriptor; never clone the descriptor.
///
/// Closing alone is insufficient on Unix: an unrelated fork can retain the
/// open-file description until exec, even with CLOEXEC. Retire the lock at the
/// owner's actual lifetime boundary, after callers finish all physical I/O.
pub(crate) struct LockedFile {
    file: File,
    process_id: u32,
}

impl LockedFile {
    pub(crate) fn acquire(file: File, shared: bool) -> Result<Self> {
        lock_file(&file, shared)?;
        Ok(Self {
            file,
            process_id: std::process::id(),
        })
    }
}

impl std::ops::Deref for LockedFile {
    type Target = File;

    fn deref(&self) -> &File {
        &self.file
    }
}

impl std::ops::DerefMut for LockedFile {
    fn deref_mut(&mut self) -> &mut File {
        &mut self.file
    }
}

impl Drop for LockedFile {
    fn drop(&mut self) {
        // A forked copy must not unlock its parent's still-live acquisition.
        // An unlock error remains fail-closed; closing the descriptor still
        // releases it when no inherited descriptor remains. Never panic in Drop.
        if self.process_id == std::process::id() {
            let _ = FileExt::unlock(&self.file);
        }
    }
}

fn lock_file(file: &File, shared: bool) -> Result<()> {
    let result = if shared {
        FileExt::try_lock_shared(file)
    } else {
        file.try_lock_exclusive()
    };
    result.map_err(|error| {
        if error.kind() == io::ErrorKind::WouldBlock
            || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
        {
            Error::Io(io::Error::new(
                io::ErrorKind::WouldBlock,
                "database file is locked by another owner",
            ))
        } else {
            Error::Io(error)
        }
    })
}

pub(crate) fn check_single_link(file: &File) -> Result<()> {
    #[cfg(unix)]
    let count = {
        use std::os::unix::fs::MetadataExt;
        file.metadata()?.nlink()
    };
    #[cfg(windows)]
    let count = u64::from(windows_link_count(file)?);
    #[cfg(not(any(unix, windows)))]
    return Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "container handle link-count checks are unavailable on this platform",
    )
    .into());
    #[cfg(any(unix, windows))]
    if count != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "writable containers and coordination entries require exactly one hard link",
        )
        .into());
    }
    #[cfg(any(unix, windows))]
    Ok(())
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn windows_link_count(file: &File) -> io::Result<u32> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: file retains its live handle through this synchronous call.
    // information is initialized, aligned repr(C) output with an exclusive
    // borrow. The API retains neither pointer nor handle ownership.
    let result = unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(information.nNumberOfLinks)
}

#[cfg(feature = "grafeo-file")]
pub(crate) fn resolve(path: &Path, provision_parent: bool) -> Result<PathBuf> {
    if path.file_name().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "container path requires a file name",
        )
        .into());
    }
    let resolved = resolve_components(path)?;
    // No parent provisioning precedes resolved-namespace validation.
    validate_public_path(&resolved)?;
    let parent = resolved.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "container path requires a parent",
        )
    })?;
    if provision_parent {
        fs::create_dir_all(parent)?;
    }
    Ok(resolved)
}

pub(crate) fn resolve_components(path: &Path) -> Result<PathBuf> {
    // Do not call absolute/full-path normalization on the complete request:
    // Windows expands `junction/..` lexically before following the junction.
    let mut resolved = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir()?
    };
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        match component {
            Component::Prefix(prefix) => {
                if !path.has_root() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "drive-relative container paths have ambiguous per-drive authority",
                    )
                    .into());
                }
                resolved = PathBuf::from(prefix.as_os_str());
            }
            Component::RootDir => resolved.push(std::path::MAIN_SEPARATOR_STR),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                let candidate = resolved.join(name);
                match fs::symlink_metadata(&candidate) {
                    Ok(_) => {
                        resolved = fs::canonicalize(candidate)?;
                        if components.peek().is_some() && !fs::metadata(&resolved)?.is_dir() {
                            return Err(io::Error::new(
                                io::ErrorKind::NotADirectory,
                                "container path crosses a non-directory component",
                            )
                            .into());
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => resolved = candidate,
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
    Ok(resolved)
}

pub(crate) fn ascii_ends_with(value: &[u8], suffix: &[u8]) -> bool {
    value.len() >= suffix.len() && value[value.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
}

pub(crate) fn ascii_contains(value: &[u8], marker: &[u8]) -> bool {
    value
        .windows(marker.len())
        .any(|part| part.eq_ignore_ascii_case(marker))
}

pub(crate) fn reserved_component(component: &OsStr) -> bool {
    let bytes = component.as_encoded_bytes();
    bytes.eq_ignore_ascii_case(b"wal")
        || ascii_ends_with(bytes, b".wal")
        || ascii_ends_with(bytes, b".restore_wal")
        || ascii_ends_with(bytes, b".trimmed_wal")
        || (bytes.starts_with(b".")
            && (ascii_contains(bytes, b".grafeo-file-save-")
                || ascii_contains(bytes, b".grafeo-save-")))
}

pub(crate) fn validate_public_path(path: &Path) -> Result<()> {
    for component in path.components() {
        #[cfg(windows)]
        if let Component::Prefix(prefix) = component {
            if matches!(
                prefix.kind(),
                std::path::Prefix::DeviceNS(_) | std::path::Prefix::Verbatim(_)
            ) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "container paths cannot use a device namespace",
                )
                .into());
            }
        }
        if let Component::Normal(name) = component {
            let bytes = name.as_encoded_bytes();
            #[cfg(windows)]
            if bytes.contains(&b':') || bytes.ends_with(b".") || bytes.ends_with(b" ") {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "container path uses a Windows normalization or alternate-stream alias",
                )
                .into());
            }
            if reserved_component(name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "container path enters a reserved disposable namespace",
                )
                .into());
            }
            // Reserve these at any component, also preventing a data directory
            // from occupying another container's coordination/staging name.
            if ascii_ends_with(bytes, b".grafeo-owner") || ascii_ends_with(bytes, b".installing") {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "container path uses a reserved coordination or checkpoint name",
                )
                .into());
            }
        }
    }
    Ok(())
}

pub(crate) fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

pub(crate) fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "bound container has no parent")
        })?;
        File::open(parent)?.sync_all()?;
    }
    // Portable Rust does not provide a directory sync on every non-Unix OS.
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

pub(crate) fn checked_file(path: &Path, writable: bool, create: bool) -> Result<File> {
    let open_existing = || -> Result<File> {
        require_regular(&fs::symlink_metadata(path)?)?;
        let file = OpenOptions::new().read(true).write(writable).open(path)?;
        validate_file(&file)?;
        require_regular(&fs::symlink_metadata(path)?)?;
        Ok(file)
    };
    match fs::symlink_metadata(path) {
        Ok(_) => open_existing(),
        Err(error) if error.kind() == io::ErrorKind::NotFound && create => {
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(path)
            {
                Ok(file) => {
                    validate_file(&file)?;
                    Ok(file)
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => open_existing(),
                Err(error) => Err(error.into()),
            }
        }
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn validate_file(file: &File) -> Result<()> {
    require_regular(&file.metadata()?)?;
    check_single_link(file)
}
