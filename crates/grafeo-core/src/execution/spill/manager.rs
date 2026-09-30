//! Exclusive spill-query and spill-file lifecycle management.

#[cfg(test)]
use super::file::{CleartextSpillRecordProvider, NoopSpillIo};
use super::file::{
    SpillFile, SpillFileIdentity, SpillFileLifecycle, SpillFileRole, SpillFrameLimits,
    SpillHandleLease, SpillIo, SpillIoOperation, SpillQueryIdentity, SpillRecordProvider,
    SpillWriterBuffer,
};
#[cfg(all(test, unix, not(target_arch = "wasm32")))]
use cap_fs_ext::OpenOptionsSyncExt as _;
#[cfg(not(target_arch = "wasm32"))]
use cap_fs_ext::{DirExt as _, FollowSymlinks, OpenOptionsFollowExt as _};
#[cfg(all(not(target_arch = "wasm32"), unix))]
use cap_std::fs::DirBuilderExt as _;
#[cfg(not(target_arch = "wasm32"))]
use cap_std::fs::{Dir as CapabilityDir, DirBuilder as CapabilityDirBuilder};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
#[cfg(any(
    target_os = "wasi",
    all(target_arch = "wasm32", unix, not(target_os = "wasi"))
))]
use std::fs::File;
#[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
use std::fs::OpenOptions;
#[cfg(all(test, unix, not(target_arch = "wasm32")))]
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const QUERY_PREFIX: &str = "grafeo-query-";
pub(super) const OWNER_MARKER: &str = ".grafeo-spill-owner";
const OWNER_MAGIC: [u8; 4] = *b"GRAQ";
const OWNER_MARKER_PREFIX_BYTES: usize = 21;
const OWNER_MARKER_AUTH_BYTES: usize = 32;
const OWNER_MARKER_AUTH_END: usize = OWNER_MARKER_PREFIX_BYTES + OWNER_MARKER_AUTH_BYTES;
pub(super) const OWNER_MARKER_BYTES: usize = OWNER_MARKER_AUTH_END + 112;
const FILE_CREATE_ATTEMPTS: usize = 64;
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
const QUERY_CREATE_ATTEMPTS: usize = 64;

static ORPHAN_CLEANUP_FAILURES: AtomicU64 = AtomicU64::new(0);

// A thread-local seam places substitution precisely between restoration and rebind.
#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
thread_local! {
    pub(super) static AFTER_OWNER_MARKER_RESTORE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

fn close_file(file: std::fs::File) {
    #[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
    drop(file);
    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    let _ = file;
}

#[cfg(target_os = "wasi")]
fn wasi_open_directory_at(directory: &File, name: &Path) -> std::io::Result<File> {
    let fd = rustix::fs::openat(
        directory,
        name,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::empty(),
    )?;
    Ok(File::from(fd))
}

#[cfg(target_os = "wasi")]
fn wasi_open_ambient_directory(path: &Path) -> std::io::Result<File> {
    let fd = rustix::fs::openat(
        rustix::fs::CWD,
        path,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::empty(),
    )?;
    Ok(File::from(fd))
}

#[cfg(target_os = "wasi")]
fn wasi_visit_directory(
    directory: &File,
    mut visitor: impl FnMut(&[u8]) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut reader = rustix::fs::Dir::read_from(directory)?;
    while let Some(entry) = reader.read() {
        let entry = entry?;
        let name = entry.file_name().to_bytes();
        if name != b"." && name != b".." {
            visitor(name)?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
fn emscripten_open_directory_at(directory: &File, name: &Path) -> std::io::Result<File> {
    let fd = rustix::fs::openat(
        directory,
        name,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::empty(),
    )?;
    Ok(File::from(fd))
}

#[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
fn emscripten_open_ambient_directory(path: &Path) -> std::io::Result<File> {
    let fd = rustix::fs::openat(
        rustix::fs::CWD,
        path,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::empty(),
    )?;
    Ok(File::from(fd))
}

#[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
fn emscripten_visit_directory(
    directory: &File,
    mut visitor: impl FnMut(&[u8]) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut reader = rustix::fs::Dir::read_from(directory)?;
    while let Some(entry) = reader.read() {
        let entry = entry?;
        let name = entry.file_name().to_bytes();
        if name != b"." && name != b".." {
            visitor(name)?;
        }
    }
    Ok(())
}

pub(crate) fn record_orphan_cleanup_failures(count: u64) {
    ORPHAN_CLEANUP_FAILURES.fetch_add(count.max(1), Ordering::Relaxed);
}

// Only exclusive marker creation mints this cleanup receipt. The live file
// handle prevents inode reuse while the retained lifecycle validates deletion.
struct MarkerCreationReceipt {
    _file: Arc<std::fs::File>,
    lifecycle: SpillFileLifecycle,
}

// Quota release syncs this retained namespace. A foreign hook must not move
// the leaf to another parent whose deletion would require a different sync.
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
fn validate_query_parent(
    directory: &CapabilityDir,
    expected: &CapabilityDir,
) -> std::io::Result<()> {
    let parent = rustix::fs::openat(
        directory,
        "..",
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )?;
    let actual = rustix::fs::fstat(&parent)?;
    let expected = rustix::fs::fstat(expected)?;
    if actual.st_dev != expected.st_dev || actual.st_ino != expected.st_ino {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "spill query moved outside its retained accounting namespace",
        ));
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
struct QueryLeafConstructionGuard {
    armed: bool,
    marker: Option<[u8; OWNER_MARKER_BYTES]>,
    marker_receipt: Option<MarkerCreationReceipt>,
    #[cfg(not(target_arch = "wasm32"))]
    root: Arc<CapabilityDir>,
    #[cfg(not(target_arch = "wasm32"))]
    leaf: std::ffi::OsString,
    #[cfg(target_os = "wasi")]
    wasi_root: Arc<File>,
    #[cfg(target_os = "wasi")]
    wasi_leaf: PathBuf,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    emscripten_root: Arc<File>,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    emscripten_leaf: PathBuf,
    identity: Option<PhysicalDirectoryIdentity>,
    // Keep the inode lease through this guard's explicit and unwind cleanup.
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    _root_leaf_lock: Option<Arc<std::fs::File>>,
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    root_reservation: Option<Arc<super::quota::RootReservation>>,
}

#[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
impl QueryLeafConstructionGuard {
    fn new(
        _path: PathBuf,
        #[cfg(not(target_arch = "wasm32"))] root: Arc<CapabilityDir>,
        #[cfg(not(target_arch = "wasm32"))] leaf: std::ffi::OsString,
        #[cfg(target_os = "wasi")] wasi_root: Arc<File>,
        #[cfg(target_os = "wasi")] wasi_leaf: PathBuf,
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))] emscripten_root: Arc<
            File,
        >,
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))] emscripten_leaf: PathBuf,
    ) -> Self {
        Self {
            armed: true,
            marker: None,
            marker_receipt: None,
            #[cfg(not(target_arch = "wasm32"))]
            root,
            #[cfg(not(target_arch = "wasm32"))]
            leaf,
            #[cfg(target_os = "wasi")]
            wasi_root,
            #[cfg(target_os = "wasi")]
            wasi_leaf,
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_root,
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_leaf,
            identity: None,
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            _root_leaf_lock: None,
            #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
            root_reservation: None,
        }
    }

    fn set_identity(&mut self, identity: PhysicalDirectoryIdentity) {
        self.identity = Some(identity);
    }

    fn set_marker(&mut self, marker: [u8; OWNER_MARKER_BYTES]) {
        self.marker = Some(marker);
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    fn remove_created_marker(&mut self, marker_present: bool) -> std::io::Result<bool> {
        let Some(receipt) = &self.marker_receipt else {
            return if marker_present {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "query construction does not own the named marker",
                ))
            } else {
                Ok(false)
            };
        };
        match receipt.lifecycle.validate_entry(Path::new(OWNER_MARKER)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        }
        receipt
            .lifecycle
            .delete_path(Path::new(OWNER_MARKER), false)?;
        self.marker_receipt = None;
        Ok(true)
    }

    fn cleanup_now(&mut self) -> std::io::Result<()> {
        if !self.armed {
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let directory = self.root.open_dir_nofollow(&self.leaf)?;
            self.identity
                .as_ref()
                .ok_or_else(|| std::io::Error::other("query construction identity unavailable"))?
                .validate_capability(&directory)?;
            let mut marker_present = false;
            for entry in directory.entries()? {
                let entry = entry?;
                if entry.file_name() != std::ffi::OsStr::new(OWNER_MARKER) {
                    return Err(std::io::Error::other(format!(
                        "unknown content prevents spill query construction cleanup: {}",
                        entry.file_name().to_string_lossy()
                    )));
                }
                marker_present = true;
            }
            let marker_removed = self.remove_created_marker(marker_present)?;
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            validate_query_parent(&directory, &self.root)?;
            if let Err(remove_error) = directory.remove_open_dir() {
                let Some(marker) = self.marker.as_ref().filter(|_| marker_removed) else {
                    return Err(remove_error);
                };
                let restore_result =
                    self.root
                        .open_dir_nofollow(&self.leaf)
                        .and_then(|directory| {
                            self.identity
                            .as_ref()
                            .ok_or_else(|| {
                                std::io::Error::other(
                                    "query construction identity unavailable for marker restore",
                                )
                            })?
                            .validate_capability(&directory)?;
                            restore_owner_marker(
                                Path::new(OWNER_MARKER),
                                marker,
                                &Arc::new(directory),
                                Path::new(OWNER_MARKER),
                                Some(&mut self.marker_receipt),
                            )
                        });
                return match restore_result {
                    Ok(()) => Err(remove_error),
                    Err(restore_error) => Err(combine_io_errors(
                        remove_error,
                        restore_error,
                        "query-construction owner marker restore",
                    )),
                };
            }
        }
        #[cfg(target_os = "wasi")]
        {
            let directory = wasi_open_directory_at(&self.wasi_root, &self.wasi_leaf)?;
            self.identity
                .as_ref()
                .ok_or_else(|| std::io::Error::other("query construction identity unavailable"))?
                .validate_wasi_handle(&directory)?;
            let mut marker_present = false;
            wasi_visit_directory(&directory, |entry| {
                if entry != OWNER_MARKER.as_bytes() {
                    return Err(std::io::Error::other(format!(
                        "unknown content prevents spill query construction cleanup: {}",
                        String::from_utf8_lossy(entry)
                    )));
                }
                marker_present = true;
                Ok(())
            })?;
            let marker_removed = self.remove_created_marker(marker_present)?;
            close_file(directory);
            let remove_result = rustix::fs::unlinkat(
                &self.wasi_root,
                &self.wasi_leaf,
                rustix::fs::AtFlags::REMOVEDIR,
            )
            .map_err(std::io::Error::from);
            if let Err(remove_error) = remove_result {
                let Some(marker) = self.marker.as_ref().filter(|_| marker_removed) else {
                    return Err(remove_error);
                };
                let restore_result = wasi_open_directory_at(&self.wasi_root, &self.wasi_leaf)
                    .and_then(|directory| {
                        self.identity
                            .as_ref()
                            .ok_or_else(|| {
                                std::io::Error::other(
                                    "query construction identity unavailable for marker restore",
                                )
                            })?
                            .validate_wasi_handle(&directory)?;
                        restore_owner_marker(
                            Path::new(OWNER_MARKER),
                            marker,
                            &directory,
                            Some(&mut self.marker_receipt),
                        )
                    });
                return match restore_result {
                    Ok(()) => Err(remove_error),
                    Err(restore_error) => Err(combine_io_errors(
                        remove_error,
                        restore_error,
                        "WASI query-construction owner marker restore",
                    )),
                };
            }
        }
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        {
            let directory =
                emscripten_open_directory_at(&self.emscripten_root, &self.emscripten_leaf)?;
            self.identity
                .as_ref()
                .ok_or_else(|| std::io::Error::other("query construction identity unavailable"))?
                .validate_directory_handle(&directory)?;
            let mut marker_present = false;
            emscripten_visit_directory(&directory, |entry| {
                if entry != OWNER_MARKER.as_bytes() {
                    return Err(std::io::Error::other(format!(
                        "unknown content prevents spill query construction cleanup: {}",
                        String::from_utf8_lossy(entry)
                    )));
                }
                marker_present = true;
                Ok(())
            })?;
            let marker_removed = self.remove_created_marker(marker_present)?;
            close_file(directory);
            let remove_result = rustix::fs::unlinkat(
                &self.emscripten_root,
                &self.emscripten_leaf,
                rustix::fs::AtFlags::REMOVEDIR,
            )
            .map_err(std::io::Error::from);
            if let Err(remove_error) = remove_result {
                let Some(marker) = self.marker.as_ref().filter(|_| marker_removed) else {
                    return Err(remove_error);
                };
                let restore_result =
                    emscripten_open_directory_at(&self.emscripten_root, &self.emscripten_leaf)
                        .and_then(|directory| {
                            self.identity
                        .as_ref()
                        .ok_or_else(|| {
                            std::io::Error::other(
                                "query construction identity unavailable for marker restore",
                            )
                        })?
                        .validate_directory_handle(&directory)?;
                            restore_owner_marker(
                                Path::new(OWNER_MARKER),
                                marker,
                                &directory,
                                Some(&mut self.marker_receipt),
                            )
                        });
                return match restore_result {
                    Ok(()) => Err(remove_error),
                    Err(restore_error) => Err(combine_io_errors(
                        remove_error,
                        restore_error,
                        "Emscripten query-construction owner marker restore",
                    )),
                };
            }
        }
        #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
        if let Some(reservation) = &self.root_reservation {
            reservation.confirm_directory_unlink()?;
        }
        self.armed = false;
        Ok(())
    }
}

#[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
impl Drop for QueryLeafConstructionGuard {
    fn drop(&mut self) {
        let failed =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.cleanup_now())) {
                Ok(result) => result.is_err(),
                Err(panic) => {
                    super::forget_cleanup_failure(panic);
                    true
                }
            };
        if failed {
            ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Hard limit for logical v1 framed bytes retained by one spill manager.
///
/// The limit covers staging, poisoned, and published files from this manager.
/// It deliberately measures exact framed file length rather than filesystem
/// allocation-unit, directory-entry, or owner-marker overhead.
///
/// The manager's documented stable-namespace precondition applies: this
/// logical ledger does not police out-of-band rename, hard-link, unlink, copy,
/// or retained-descriptor activity by callers with direct filesystem authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillDiskQuota {
    limit_bytes: u64,
}

impl SpillDiskQuota {
    /// Creates a hard logical-byte quota. Zero is a valid deny-all policy.
    #[must_use]
    pub const fn new(limit_bytes: u64) -> Self {
        Self { limit_bytes }
    }

    /// Returns the maximum live logical spill bytes.
    #[must_use]
    pub const fn limit_bytes(self) -> u64 {
        self.limit_bytes
    }
}

/// Coherent point-in-time disk accounting for one spill manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillDiskStats {
    /// Configured hard logical-byte limit.
    pub limit_bytes: u64,
    /// Bytes retained by all live staging, poisoned, and published files.
    pub reserved_live_bytes: u64,
    /// Subset of live bytes that completed sync and publication.
    pub published_live_bytes: u64,
    /// High-water mark of live reserved bytes.
    pub peak_reserved_bytes: u64,
}

/// Query-local physical accounting sampled from the existing reservation owner.
/// These observations never authorize admission or quota credit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillPhysicalStats {
    /// Known retained physical reservation, including query metadata.
    /// Consult `reservation_uncertain` before treating this as an exact amount.
    pub reserved_bytes: u64,
    /// High-water reservation including observed debt above the policy limit.
    pub peak_reserved_bytes: u64,
    /// File allocation sampled at publication, retained until retirement completes.
    /// Excludes directory/control metadata and unsampled staging growth.
    pub observed_file_bytes: u64,
    /// High-water sum of sampled file allocation.
    pub peak_observed_file_bytes: u64,
    /// Retained reservation after a failed explicit query cleanup attempt.
    pub cleanup_debt_bytes: u64,
    /// Whether the latest explicit query cleanup attempt failed.
    pub cleanup_failed: bool,
    /// A failed quota transaction makes the durable amount ambiguous. The
    /// reported reservation/debt then gives known retained bytes, not a total.
    pub reservation_uncertain: bool,
}

/// Structured failure from an atomic spill-byte quota admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillQuotaExceeded {
    limit_bytes: u64,
    used_bytes: u64,
    requested_bytes: u64,
}

impl SpillQuotaExceeded {
    pub(super) const fn from_usage(
        limit_bytes: u64,
        used_bytes: u64,
        requested_bytes: u64,
    ) -> Self {
        Self {
            limit_bytes,
            used_bytes,
            requested_bytes,
        }
    }

    /// Configured hard limit at the failed admission boundary.
    #[must_use]
    pub const fn limit_bytes(self) -> u64 {
        self.limit_bytes
    }

    /// Live bytes already charged when admission failed.
    #[must_use]
    pub const fn used_bytes(self) -> u64 {
        self.used_bytes
    }

    /// Additional exact framed bytes that were denied.
    #[must_use]
    pub const fn requested_bytes(self) -> u64 {
        self.requested_bytes
    }
}

impl std::fmt::Display for SpillQuotaExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "spill disk quota exceeded: requested {} bytes with {} of {} bytes in use",
            self.requested_bytes, self.used_bytes, self.limit_bytes
        )
    }
}

impl std::error::Error for SpillQuotaExceeded {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackedFilePhase {
    Staging,
    Published { bytes: u64 },
}

#[derive(Clone, Debug)]
struct TrackedFile {
    role: SpillFileRole,
    reserved_bytes: u64,
    phase: TrackedFilePhase,
    lifecycle: Arc<SpillFileLifecycle>,
}

impl TrackedFile {
    const fn published_bytes(&self) -> u64 {
        match self.phase {
            TrackedFilePhase::Staging => 0,
            TrackedFilePhase::Published { bytes } => bytes,
        }
    }
}

#[derive(Debug)]
struct SpillLedger {
    files: HashMap<PathBuf, TrackedFile>,
    reserved_live_bytes: u64,
    published_live_bytes: u64,
    peak_reserved_bytes: u64,
    total_published_bytes: u64,
    total_runs: u64,
    total_partitions: u64,
    merge_time_ns: u64,
}

#[derive(Clone, Debug)]
pub(super) struct PhysicalDirectoryIdentity {
    #[cfg(any(unix, windows, target_os = "wasi"))]
    device: u64,
    #[cfg(any(unix, windows, target_os = "wasi"))]
    inode: u64,
}

impl PhysicalDirectoryIdentity {
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    fn capture_directory_handle(file: &File) -> std::io::Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill directory handle is not a directory",
            ));
        }
        Self::capture_metadata(&metadata)
    }

    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    fn validate_directory_handle(&self, file: &File) -> std::io::Result<()> {
        self.validate_metadata(&file.metadata()?)
    }

    #[cfg(target_os = "wasi")]
    fn capture_wasi_handle(file: &std::fs::File) -> std::io::Result<Self> {
        if !file.metadata()?.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill directory handle is not a directory",
            ));
        }
        let stat = rustix::fs::fstat(file)?;
        Ok(Self {
            device: stat.st_dev,
            inode: stat.st_ino,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn capture_capability(
        directory: &CapabilityDir,
        path: &Path,
    ) -> std::io::Result<Self> {
        let identity = Self::capture_capability_handle(directory)?;
        identity.validate_path(path)?;
        Ok(identity)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn capture_capability_handle(directory: &CapabilityDir) -> std::io::Result<Self> {
        use cap_fs_ext::MetadataExt as _;
        #[cfg(target_os = "macos")]
        super::validate_no_acl_grants(directory)?;
        let metadata = directory.dir_metadata()?;
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }

    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    fn capture_metadata(metadata: &std::fs::Metadata) -> std::io::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }

    #[cfg(not(target_os = "wasi"))]
    pub(super) fn validate_path(&self, path: &Path) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            // A single no-follow snapshot proves both the entry kind and its
            // identity; a second following stat could observe a substituted link.
            let metadata = std::fs::symlink_metadata(path)?;
            if metadata.file_type().is_symlink() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("spill directory {} is a symlink", path.display()),
                ));
            }
            if !metadata.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("spill path {} is not a directory", path.display()),
                ));
            }
            self.validate_metadata(&metadata)
        }
        #[cfg(windows)]
        {
            validate_existing_directory(path)?;
            let directory = CapabilityDir::open_ambient_dir(path, cap_std::ambient_authority())?;
            self.validate_capability(&directory)
        }
        #[cfg(not(any(unix, windows, target_os = "wasi")))]
        {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "spill directory identity is unsupported on this wasm platform",
            ))
        }
    }

    #[cfg(target_os = "wasi")]
    fn validate_wasi_handle(&self, file: &std::fs::File) -> std::io::Result<()> {
        if !file.metadata()?.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill directory handle is not a directory",
            ));
        }
        let stat = rustix::fs::fstat(file)?;
        if stat.st_dev != self.device || stat.st_ino != self.inode {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill query directory was replaced",
            ));
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn validate_capability(&self, directory: &CapabilityDir) -> std::io::Result<()> {
        use cap_fs_ext::MetadataExt as _;
        #[cfg(target_os = "macos")]
        super::validate_no_acl_grants(directory)?;
        let metadata = directory.dir_metadata()?;
        if !metadata.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill query handle is not a directory",
            ));
        }
        if metadata.dev() != self.device || metadata.ino() != self.inode {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill query directory was replaced",
            ));
        }
        Ok(())
    }

    #[cfg(unix)]
    fn validate_metadata(&self, metadata: &std::fs::Metadata) -> std::io::Result<()> {
        if !metadata.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill directory handle is not a directory",
            ));
        }
        use std::os::unix::fs::MetadataExt;
        if metadata.dev() != self.device || metadata.ino() != self.inode {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill query directory was replaced",
            ));
        }
        Ok(())
    }
}

pub(crate) struct SpillManagerState {
    profile_merge: AtomicBool,
    ledger: Mutex<SpillLedger>,
    disk_quota: SpillDiskQuota,
    io: Arc<dyn SpillIo>,
}

trait SpillIdentitySource: Send + Sync {
    fn next_file_identity(&self) -> std::io::Result<SpillFileIdentity>;
}

#[cfg(any(
    test,
    all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    )
))]
struct RandomSpillIdentitySource;

#[cfg(any(
    test,
    all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    )
))]
impl SpillIdentitySource for RandomSpillIdentitySource {
    fn next_file_identity(&self) -> std::io::Result<SpillFileIdentity> {
        SpillFileIdentity::random()
    }
}

impl SpillManagerState {
    fn register_staging(
        &self,
        path: PathBuf,
        lifecycle: Arc<SpillFileLifecycle>,
        role: SpillFileRole,
    ) -> std::io::Result<()> {
        match self.ledger.lock().files.entry(path) {
            Entry::Vacant(entry) => {
                entry.insert(TrackedFile {
                    role,
                    reserved_bytes: 0,
                    phase: TrackedFilePhase::Staging,
                    lifecycle,
                });
                Ok(())
            }
            Entry::Occupied(_) => Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "spill path is already tracked by this manager",
            )),
        }
    }

    pub(crate) fn reserve_bytes(&self, path: &Path, bytes: u64) -> std::io::Result<()> {
        let mut ledger = self.ledger.lock();
        let tracked = ledger
            .files
            .get(path)
            .ok_or_else(|| std::io::Error::other("spill file is not manager-owned"))?;
        if tracked.phase != TrackedFilePhase::Staging {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "published spill files cannot reserve additional bytes",
            ));
        }
        let file_reserved = tracked.reserved_bytes;
        let lifecycle = Arc::clone(&tracked.lifecycle);
        if ledger.reserved_live_bytes > self.disk_quota.limit_bytes {
            return Err(std::io::Error::other(
                "live spill byte accounting exceeds its configured quota",
            ));
        }
        let remaining = self.disk_quota.limit_bytes - ledger.reserved_live_bytes;
        if bytes > remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::QuotaExceeded,
                SpillQuotaExceeded {
                    limit_bytes: self.disk_quota.limit_bytes,
                    used_bytes: ledger.reserved_live_bytes,
                    requested_bytes: bytes,
                },
            ));
        }
        let next_file = file_reserved
            .checked_add(bytes)
            .ok_or_else(|| std::io::Error::other("per-file spill byte accounting overflow"))?;
        let next_live = ledger
            .reserved_live_bytes
            .checked_add(bytes)
            .ok_or_else(|| std::io::Error::other("live spill byte accounting overflow"))?;
        ledger
            .files
            .get_mut(path)
            .expect("tracked spill file remains present while ledger is locked")
            .reserved_bytes = next_file;
        ledger.reserved_live_bytes = next_live;
        ledger.peak_reserved_bytes = ledger.peak_reserved_bytes.max(next_live);
        drop(ledger);
        lifecycle.reserve_physical(next_file, 0, false)
    }

    pub(crate) fn publish(&self, path: &Path, bytes: u64) -> std::io::Result<()> {
        let mut ledger = self.ledger.lock();
        let tracked = ledger
            .files
            .get(path)
            .ok_or_else(|| std::io::Error::other("spill file is not manager-owned"))?;
        if tracked.phase != TrackedFilePhase::Staging {
            return Err(std::io::Error::other("spill file was already published"));
        }
        if tracked.reserved_bytes != bytes {
            return Err(std::io::Error::other(format!(
                "spill publication byte mismatch: file reserved {}, writer reported {bytes}",
                tracked.reserved_bytes
            )));
        }
        let role = tracked.role;
        let next_published = ledger
            .published_live_bytes
            .checked_add(bytes)
            .ok_or_else(|| std::io::Error::other("published spill byte accounting overflow"))?;
        if next_published > ledger.reserved_live_bytes {
            return Err(std::io::Error::other(
                "published spill bytes exceed live reserved bytes",
            ));
        }
        ledger
            .files
            .get_mut(path)
            .expect("tracked spill file remains present while ledger is locked")
            .phase = TrackedFilePhase::Published { bytes };
        ledger.published_live_bytes = next_published;
        ledger.total_published_bytes = ledger.total_published_bytes.saturating_add(bytes);
        match role {
            SpillFileRole::SortRun => ledger.total_runs = ledger.total_runs.saturating_add(1),
            SpillFileRole::NativePartition => {
                ledger.total_partitions = ledger.total_partitions.saturating_add(1);
            }
            SpillFileRole::RdfAggregateState => {}
        }
        Ok(())
    }

    pub(crate) fn unregister(&self, path: &Path) -> std::io::Result<()> {
        let mut ledger = self.ledger.lock();
        if let Some(tracked) = ledger.files.get(path) {
            let next_reserved = ledger
                .reserved_live_bytes
                .checked_sub(tracked.reserved_bytes)
                .ok_or_else(|| std::io::Error::other("live spill byte accounting underflow"))?;
            let next_published = ledger
                .published_live_bytes
                .checked_sub(tracked.published_bytes())
                .ok_or_else(|| {
                    std::io::Error::other("published spill byte accounting underflow")
                })?;
            ledger.files.remove(path);
            ledger.reserved_live_bytes = next_reserved;
            ledger.published_live_bytes = next_published;
        }
        Ok(())
    }

    fn disk_stats(&self) -> SpillDiskStats {
        let ledger = self.ledger.lock();
        SpillDiskStats {
            limit_bytes: self.disk_quota.limit_bytes,
            reserved_live_bytes: ledger.reserved_live_bytes,
            published_live_bytes: ledger.published_live_bytes,
            peak_reserved_bytes: ledger.peak_reserved_bytes,
        }
    }
}

/// Manages one spill directory and its exclusively created files.
///
/// Construction consumes an authenticated [`super::SpillQueryLease`] admitted
/// by [`super::SpillRoot`]. The retained lease owns query cleanup authority.
pub struct SpillManager {
    spill_dir: PathBuf,
    provider: Arc<dyn SpillRecordProvider>,
    io: Arc<dyn SpillIo>,
    limits: SpillFrameLimits,
    state: Arc<SpillManagerState>,
    identity_source: Arc<dyn SpillIdentitySource>,
    owns_dir: bool,
    query_state: Mutex<QueryLifecycle>,
    directory_identity: PhysicalDirectoryIdentity,
    #[cfg(not(target_arch = "wasm32"))]
    directory: Arc<CapabilityDir>,
    #[cfg(not(target_arch = "wasm32"))]
    query_leaf: Option<std::ffi::OsString>,
    #[cfg(target_os = "wasi")]
    wasi_directory: Arc<File>,
    #[cfg(target_os = "wasi")]
    wasi_query_leaf: Option<PathBuf>,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    emscripten_directory: Arc<File>,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    emscripten_query_leaf: Option<PathBuf>,
    // Drop after directory handles and file ledger owners.
    root_lease: Option<Arc<super::root::QueryLeaseAuthority>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueryLifecycleState {
    Open,
    Finishing,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    Releasing,
    Poisoned,
    Closed,
}

#[derive(Debug)]
struct QueryLifecycle {
    phase: QueryLifecycleState,
    active_creates: usize,
    cleanup_failed: bool,
}

struct QueryCreateLease<'a> {
    state: &'a Mutex<QueryLifecycle>,
}

struct QueryFinishLease<'a> {
    state: &'a Mutex<QueryLifecycle>,
    committed: bool,
}

impl QueryFinishLease<'_> {
    fn commit_closed(&mut self) {
        self.state.lock().phase = QueryLifecycleState::Closed;
        self.committed = true;
    }
}

impl Drop for QueryFinishLease<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let mut state = self.state.lock();
            state.cleanup_failed = true;
            if state.phase == QueryLifecycleState::Finishing {
                state.phase = QueryLifecycleState::Open;
            }
        }
    }
}

impl Drop for QueryCreateLease<'_> {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.active_creates = state.active_creates.saturating_sub(1);
    }
}

// Private codec/operator fixture: it owns only framed files in a test directory.
// It does not model root admission, query authority, or authenticated reclamation.
#[cfg(test)]
pub(crate) struct BorrowedSpillFixture {
    spill_dir: PathBuf,
    provider: Arc<dyn SpillRecordProvider>,
    limits: SpillFrameLimits,
    io: Arc<dyn SpillIo>,
    disk_quota: SpillDiskQuota,
    identity_source: Arc<dyn SpillIdentitySource>,
}

#[cfg(test)]
impl BorrowedSpillFixture {
    pub(crate) fn new(spill_dir: impl Into<PathBuf>) -> Self {
        Self {
            spill_dir: spill_dir.into(),
            provider: Arc::new(CleartextSpillRecordProvider),
            limits: SpillFrameLimits::format_max(),
            io: Arc::new(NoopSpillIo),
            disk_quota: SpillDiskQuota::new(u64::MAX),
            identity_source: Arc::new(RandomSpillIdentitySource),
        }
    }

    pub(crate) fn provider(
        mut self,
        provider: Arc<dyn SpillRecordProvider>,
        limits: SpillFrameLimits,
    ) -> Self {
        self.provider = provider;
        self.limits = limits;
        self
    }

    pub(crate) fn io(mut self, io: Arc<dyn SpillIo>) -> Self {
        self.io = io;
        self
    }

    pub(crate) fn quota(mut self, quota: SpillDiskQuota) -> Self {
        self.disk_quota = quota;
        self
    }

    fn identities(mut self, source: Arc<dyn SpillIdentitySource>) -> Self {
        self.identity_source = source;
        self
    }

    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    pub(crate) fn build(self) -> std::io::Result<SpillManager> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "codec directory fixture requires directory capabilities",
        ))
    }

    #[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
    pub(crate) fn build(self) -> std::io::Result<SpillManager> {
        let Self {
            spill_dir,
            provider,
            limits,
            io,
            disk_quota,
            identity_source,
        } = self;
        ensure_directory(&spill_dir)?;
        #[cfg(not(target_arch = "wasm32"))]
        let directory = Arc::new(CapabilityDir::open_ambient_dir(
            &spill_dir,
            cap_std::ambient_authority(),
        )?);
        #[cfg(target_os = "wasi")]
        let wasi_directory = Arc::new(wasi_open_ambient_directory(&spill_dir)?);
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let emscripten_directory = Arc::new(emscripten_open_ambient_directory(&spill_dir)?);
        SpillManager::build(
            spill_dir,
            provider,
            limits,
            io,
            disk_quota,
            identity_source,
            false,
            #[cfg(not(target_arch = "wasm32"))]
            directory,
            #[cfg(not(target_arch = "wasm32"))]
            None,
            #[cfg(target_os = "wasi")]
            wasi_directory,
            #[cfg(target_os = "wasi")]
            None,
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_directory,
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            None,
            None,
        )
    }
}

impl SpillManager {
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn create_from_root(
        root: &Arc<super::root::SpillRoot>,
        identity: SpillQueryIdentity,
        provider: Arc<dyn SpillRecordProvider>,
        query_id: crate::execution::QueryExecutionId,
        cancellation: crate::execution::QueryCancellationToken,
    ) -> std::io::Result<Self> {
        let retained_root =
            super::root::RootQueryConstruction::new(root, identity, query_id, cancellation)?;
        let spill_root = root.namespace_path();
        let limits = root.frame_limits();
        let io = root.io();
        let disk_quota = root.disk_quota();
        validate_production_spill_root(spill_root)?;
        let root_directory = Arc::clone(&retained_root.directory);
        let _root_identity =
            PhysicalDirectoryIdentity::capture_capability(&root_directory, spill_root)?;
        validate_unix_spill_root_metadata(
            &root_directory.try_clone()?.into_std_file().metadata()?,
            spill_root,
        )?;
        for _ in 0..QUERY_CREATE_ATTEMPTS {
            let query_leaf = format!("{QUERY_PREFIX}{}", identity.hex());
            let query_dir = spill_root.join(&query_leaf);
            retained_root.attempted();
            let create_result = create_owner_only_directory_at(&root_directory, &query_leaf);
            match create_result {
                Ok(()) => {
                    let mut construction_guard = QueryLeafConstructionGuard::new(
                        query_dir.clone(),
                        Arc::clone(&root_directory),
                        std::ffi::OsString::from(&query_leaf),
                    );
                    let directory = match root_directory.open_dir_nofollow(&query_leaf) {
                        Ok(directory) => directory,
                        Err(error) => {
                            match root_directory.remove_dir(&query_leaf) {
                                Ok(()) => construction_guard.disarm(),
                                Err(cleanup_error) => {
                                    ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
                                    return Err(combine_io_errors(
                                        error,
                                        cleanup_error,
                                        "query-directory open cleanup",
                                    ));
                                }
                            }
                            return Err(error);
                        }
                    };
                    {
                        retained_root.bind(&directory)?;
                        #[cfg(target_os = "macos")]
                        {
                            construction_guard.root_reservation = Some(retained_root.reservation());
                        }
                    }
                    let created_directory_identity =
                        match PhysicalDirectoryIdentity::capture_capability_handle(&directory) {
                            Ok(identity) => identity,
                            Err(error) => {
                                let cleanup = (|| {
                                    if directory.entries()?.next().is_some() {
                                        return Err(std::io::Error::other(
                                            "unknown content prevents spill query identity-failure cleanup",
                                        ));
                                    }
                                    directory.remove_open_dir()
                                })();
                                construction_guard.disarm();
                                return match cleanup {
                                    Ok(()) => Err(error),
                                    Err(cleanup_error) => {
                                        ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
                                        Err(combine_io_errors(
                                            error,
                                            cleanup_error,
                                            "query-directory identity-failure cleanup",
                                        ))
                                    }
                                };
                            }
                        };
                    let directory = Arc::new(directory);
                    construction_guard.set_identity(created_directory_identity.clone());
                    if let Err(error) = created_directory_identity.validate_path(&query_dir) {
                        drop(directory);
                        return match construction_guard.cleanup_now() {
                            Ok(()) => Err(error),
                            Err(cleanup_error) => {
                                ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
                                Err(combine_io_errors(
                                    error,
                                    cleanup_error,
                                    "query-directory identity cleanup",
                                ))
                            }
                        };
                    }
                    let prepared = match retained_root.prepare(identity, &directory) {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            return match construction_guard.cleanup_now() {
                                Ok(()) => Err(error),
                                Err(cleanup_error) => {
                                    ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
                                    Err(combine_io_errors(
                                        error,
                                        cleanup_error,
                                        "owner-marker preparation cleanup",
                                    ))
                                }
                            };
                        }
                    };
                    construction_guard._root_leaf_lock = Some(prepared.directory_lock());
                    let marker = *prepared.marker();
                    construction_guard.set_marker(marker);
                    let marker_result = write_marker_bytes(
                        &query_dir.join(OWNER_MARKER),
                        &marker,
                        Some(io.as_ref()),
                        &directory,
                        Path::new(OWNER_MARKER),
                        Some(&mut construction_guard.marker_receipt),
                    );
                    let root_lease = match marker_result.and_then(|()| {
                        let receipt =
                            construction_guard.marker_receipt.as_ref().ok_or_else(|| {
                                std::io::Error::other(
                                    "published owner marker has no creation receipt",
                                )
                            })?;
                        prepared.finish(&receipt._file)
                    }) {
                        Ok(lease) => lease,
                        Err(error) => {
                            drop(directory);
                            return match construction_guard.cleanup_now() {
                                Ok(()) => Err(error),
                                Err(cleanup_error) => {
                                    ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
                                    Err(combine_io_errors(
                                        error,
                                        cleanup_error,
                                        "owner-marker creation cleanup",
                                    ))
                                }
                            };
                        }
                    };
                    drop(directory);
                    let result = (|| {
                        root_lease.validate(true)?;
                        Self::build(
                            query_dir.clone(),
                            provider,
                            limits,
                            io,
                            disk_quota,
                            Arc::new(RandomSpillIdentitySource),
                            true,
                            Arc::clone(&root_directory),
                            Some(std::ffi::OsString::from(&query_leaf)),
                            Some(created_directory_identity),
                        )
                    })();
                    match result {
                        Ok(mut manager) => {
                            manager.root_lease = Some(Arc::new(root_lease));
                            construction_guard.disarm();
                            return Ok(manager);
                        }
                        Err(error) => {
                            return match construction_guard.cleanup_now() {
                                Ok(()) => Err(error),
                                Err(cleanup_error) => {
                                    ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
                                    Err(combine_io_errors(
                                        error,
                                        cleanup_error,
                                        "query-manager construction cleanup",
                                    ))
                                }
                            };
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create a unique spill query directory",
        ))
    }

    /// Consumes the authenticated query lease without creating another manager.
    #[must_use]
    pub fn from_query_lease(lease: super::root::SpillQueryLease) -> Self {
        lease.into_manager()
    }

    #[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
    #[allow(
        clippy::too_many_arguments,
        reason = "platform-gated root capabilities and a carried directory identity must be assembled atomically"
    )]
    fn build(
        spill_dir: PathBuf,
        provider: Arc<dyn SpillRecordProvider>,
        limits: SpillFrameLimits,
        io: Arc<dyn SpillIo>,
        disk_quota: SpillDiskQuota,
        identity_source: Arc<dyn SpillIdentitySource>,
        owns_dir: bool,
        #[cfg(not(target_arch = "wasm32"))] directory: Arc<CapabilityDir>,
        #[cfg(not(target_arch = "wasm32"))] query_leaf: Option<std::ffi::OsString>,
        #[cfg(target_os = "wasi")] wasi_directory: Arc<File>,
        #[cfg(target_os = "wasi")] wasi_query_leaf: Option<PathBuf>,
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        emscripten_directory: Arc<File>,
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        emscripten_query_leaf: Option<PathBuf>,
        expected_directory_identity: Option<PhysicalDirectoryIdentity>,
    ) -> std::io::Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let directory_identity = {
            let opened_directory = if let Some(leaf) = query_leaf.as_ref() {
                directory.open_dir_nofollow(leaf)?
            } else {
                directory.try_clone()?
            };
            if let Some(identity) = expected_directory_identity {
                identity.validate_capability(&opened_directory)?;
                identity.validate_path(&spill_dir)?;
                identity
            } else {
                PhysicalDirectoryIdentity::capture_capability(&opened_directory, &spill_dir)?
            }
        };
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let directory_identity = {
            let opened = if let Some(leaf) = emscripten_query_leaf.as_ref() {
                emscripten_open_directory_at(&emscripten_directory, leaf)?
            } else {
                emscripten_directory.try_clone()?
            };
            if let Some(identity) = expected_directory_identity {
                identity.validate_directory_handle(&opened)?;
                identity.validate_path(&spill_dir)?;
                identity
            } else {
                let identity = PhysicalDirectoryIdentity::capture_directory_handle(&opened)?;
                identity.validate_path(&spill_dir)?;
                identity
            }
        };
        #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
        let directory_identity = if let Some(identity) = expected_directory_identity {
            identity.validate_path(&spill_dir)?;
            identity
        } else {
            PhysicalDirectoryIdentity::capture(&spill_dir)?
        };
        #[cfg(target_os = "wasi")]
        let directory_identity = {
            let opened = if let Some(leaf) = wasi_query_leaf.as_ref() {
                wasi_open_directory_at(&wasi_directory, leaf)?
            } else {
                wasi_directory.try_clone()?
            };
            if let Some(identity) = expected_directory_identity {
                identity.validate_wasi_handle(&opened)?;
                identity
            } else {
                PhysicalDirectoryIdentity::capture_wasi_handle(&opened)?
            }
        };
        Ok(Self {
            spill_dir,
            provider,
            io: Arc::clone(&io),
            limits,
            state: Arc::new(SpillManagerState {
                profile_merge: AtomicBool::new(false),
                ledger: Mutex::new(SpillLedger {
                    files: HashMap::new(),
                    reserved_live_bytes: 0,
                    published_live_bytes: 0,
                    peak_reserved_bytes: 0,
                    total_published_bytes: 0,
                    total_runs: 0,
                    total_partitions: 0,
                    merge_time_ns: 0,
                }),
                disk_quota,
                io,
            }),
            identity_source,
            owns_dir,
            root_lease: None,
            query_state: Mutex::new(QueryLifecycle {
                phase: QueryLifecycleState::Open,
                active_creates: 0,
                cleanup_failed: false,
            }),
            directory_identity,
            #[cfg(not(target_arch = "wasm32"))]
            directory,
            #[cfg(not(target_arch = "wasm32"))]
            query_leaf,
            #[cfg(target_os = "wasi")]
            wasi_directory,
            #[cfg(target_os = "wasi")]
            wasi_query_leaf,
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_directory,
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_query_leaf,
        })
    }

    /// Returns the spill directory path.
    #[must_use]
    pub fn spill_dir(&self) -> &Path {
        &self.spill_dir
    }

    /// Returns the explicit record limits used for newly created files.
    #[must_use]
    pub const fn frame_limits(&self) -> SpillFrameLimits {
        self.limits
    }

    /// Returns the provider-owned file-lifetime workspace. Callers separately
    /// compose writer backing or the reader's core control-frame buffer.
    pub(crate) fn qualified_file_workspace_bound(&self) -> std::io::Result<usize> {
        self.provider
            .file_workspace_allocation_bound()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "spill provider does not declare a qualified file-workspace bound",
                )
            })
    }

    pub(crate) fn qualified_sort_provider_workspace_bound(&self) -> Option<usize> {
        self.provider.file_workspace_allocation_bound()
    }

    pub(crate) fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
        self.io.qualified_sort_hook_workspace_bound()
    }

    pub(crate) fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
        self.io.qualified_reader_hook_workspace_bound()
    }

    /// Creates one exclusively opened file for a closed role.
    ///
    /// The entropy-derived name contains no caller-controlled path component.
    /// Registration happens only after `create_new(true)` succeeds. Provider or
    /// initial-frame failure attempts exact explicit cleanup. The Drop backstop
    /// retries a failure; any artifact that still remains stays manager-tracked
    /// and never retries in cleartext.
    ///
    /// # Errors
    ///
    /// Returns an error for a closed query, identity exhaustion, capability or
    /// provider failure, initial framing failure, or failed cleanup.
    pub fn create_file(&self, role: SpillFileRole) -> std::io::Result<SpillFile> {
        let _create_lease = self.begin_create()?;
        let writer_buffer = SpillWriterBuffer::prepare()?;
        self.create_file_after_begin(role, writer_buffer, None, None)
    }

    /// Creates a file using caller-prepared writer backing. The backing must
    /// already have passed any caller-specific admission protocol.
    #[cfg(test)]
    pub(super) fn create_file_with_writer_buffer(
        &self,
        role: SpillFileRole,
        writer_buffer: SpillWriterBuffer,
    ) -> std::io::Result<SpillFile> {
        let _create_lease = self.begin_create()?;
        self.create_file_after_begin(role, writer_buffer, None, None)
    }

    /// Creates a file through a caller-owned admission protocol. The caller
    /// must compose the callback amount with the prepared writer allocation
    /// and retain the resulting authority until the writer is closed.
    pub(super) fn create_qualified_file_with_writer_buffer(
        &self,
        role: SpillFileRole,
        writer_buffer: SpillWriterBuffer,
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
    ) -> std::io::Result<SpillFile> {
        let provider_bound = self.qualified_file_workspace_bound()?;
        // Admission precedes the create lease, filesystem identity, and
        // provider initialization. The caller composes this provider amount
        // with any simultaneously live writer-buffer charge.
        admit(provider_bound)?;
        let _create_lease = self.begin_create()?;
        self.create_file_after_begin(role, writer_buffer, Some(provider_bound), None)
    }

    /// Owned scheduling seam: report whether failed construction retired every
    /// opaque cleanup payload before the caller releases its writer workspace.
    pub(super) fn create_owned_file_with_writer_buffer(
        &self,
        role: SpillFileRole,
        writer_buffer: SpillWriterBuffer,
        mut admit: impl FnMut(usize) -> std::io::Result<()>,
        cleanup_complete: &mut bool,
    ) -> std::io::Result<SpillFile> {
        let provider_bound = self.qualified_file_workspace_bound()?;
        admit(provider_bound)?;
        let _create_lease = self.begin_create()?;
        self.create_file_after_begin(
            role,
            writer_buffer,
            Some(provider_bound),
            Some(cleanup_complete),
        )
    }

    fn create_file_after_begin(
        &self,
        role: SpillFileRole,
        writer_buffer: SpillWriterBuffer,
        qualified_file_workspace_bound: Option<usize>,
        owned_cleanup_complete: Option<&mut bool>,
    ) -> std::io::Result<SpillFile> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(leaf) = self.query_leaf.as_ref() {
                let query_directory = self.directory.open_dir_nofollow(leaf)?;
                self.directory_identity
                    .validate_capability(&query_directory)?;
            } else {
                self.directory_identity
                    .validate_capability(&self.directory)?;
            }
        }
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        {
            let directory = if let Some(leaf) = self.emscripten_query_leaf.as_ref() {
                emscripten_open_directory_at(&self.emscripten_directory, leaf)?
            } else {
                self.emscripten_directory.try_clone()?
            };
            self.directory_identity
                .validate_directory_handle(&directory)?;
        }
        #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
        self.directory_identity.validate_path(&self.spill_dir)?;
        #[cfg(target_os = "wasi")]
        {
            let directory = if let Some(leaf) = self.wasi_query_leaf.as_ref() {
                wasi_open_directory_at(&self.wasi_directory, leaf)?
            } else {
                self.wasi_directory.try_clone()?
            };
            self.directory_identity.validate_wasi_handle(&directory)?;
        }
        for _ in 0..FILE_CREATE_ATTEMPTS {
            let identity = self.identity_source.next_file_identity()?;
            let file_name = format!("{}-{}.grsp", role.file_prefix(), identity.hex());
            let path = self.spill_dir.join(&file_name);
            #[cfg(not(target_arch = "wasm32"))]
            let file_directory = if let Some(leaf) = self.query_leaf.as_ref() {
                let directory = Arc::new(self.directory.open_dir_nofollow(leaf)?);
                self.directory_identity.validate_capability(&directory)?;
                directory
            } else {
                Arc::clone(&self.directory)
            };
            #[cfg(not(target_arch = "wasm32"))]
            let relative_name = PathBuf::from(&file_name);
            #[cfg(target_os = "wasi")]
            let file_directory = Arc::new(if let Some(leaf) = self.wasi_query_leaf.as_ref() {
                let directory = wasi_open_directory_at(&self.wasi_directory, leaf)?;
                self.directory_identity.validate_wasi_handle(&directory)?;
                directory
            } else {
                self.wasi_directory.try_clone()?
            });
            #[cfg(target_os = "wasi")]
            let relative_name = PathBuf::from(&file_name);
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            let file_directory =
                Arc::new(if let Some(leaf) = self.emscripten_query_leaf.as_ref() {
                    let directory = emscripten_open_directory_at(&self.emscripten_directory, leaf)?;
                    self.directory_identity
                        .validate_directory_handle(&directory)?;
                    directory
                } else {
                    self.emscripten_directory.try_clone()?
                });
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            let relative_name = PathBuf::from(&file_name);
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            let reservation = self
                .root_lease
                .as_ref()
                .map(|lease| lease.reserve_file(identity))
                .transpose()?;
            self.io.check(SpillIoOperation::Create)?;
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            if let Some(reservation) = &reservation {
                reservation.attempted();
            }
            #[cfg(not(target_arch = "wasm32"))]
            let opened = {
                let mut options = cap_std::fs::OpenOptions::new();
                options.read(true).write(true).create_new(true);
                options.follow(FollowSymlinks::No);
                #[cfg(unix)]
                {
                    use cap_std::fs::OpenOptionsExt as _;
                    options.mode(0o600);
                }
                file_directory
                    .open_with(&relative_name, &options)
                    .map(cap_std::fs::File::into_std)
            };
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            let opened = rustix::fs::openat(
                &file_directory,
                &relative_name,
                rustix::fs::OFlags::RDWR
                    | rustix::fs::OFlags::CREATE
                    | rustix::fs::OFlags::EXCL
                    | rustix::fs::OFlags::NOFOLLOW,
                rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            )
            .map(File::from)
            .map_err(std::io::Error::from);
            #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
            let opened = OpenOptions::new()
                .write(true)
                .read(true)
                .create_new(true)
                .open(&path);
            #[cfg(target_os = "wasi")]
            let opened = rustix::fs::openat(
                &file_directory,
                &relative_name,
                rustix::fs::OFlags::RDWR
                    | rustix::fs::OFlags::CREATE
                    | rustix::fs::OFlags::EXCL
                    | rustix::fs::OFlags::NOFOLLOW,
                rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            )
            .map(File::from)
            .map_err(std::io::Error::from);
            let file = match opened {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            };
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            if let Some(reservation) = &reservation {
                reservation.bind(file.try_clone()?, Arc::clone(&file_directory))?;
            }
            #[cfg(not(target_arch = "wasm32"))]
            let captured = SpillFileLifecycle::capture_with_directory(
                &file,
                Arc::clone(&file_directory),
                relative_name.clone(),
            );
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            let captured = SpillFileLifecycle::capture_with_emscripten_directory(
                &file,
                Arc::clone(&file_directory),
                relative_name.clone(),
            );
            #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
            let captured = SpillFileLifecycle::capture(&file, &path);
            #[cfg(target_os = "wasi")]
            let captured = SpillFileLifecycle::capture_with_wasi_directory(
                &file,
                Arc::clone(&file_directory),
                relative_name.clone(),
            );
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            let captured = captured.map(|lifecycle| lifecycle.with_root_reservation(reservation));
            let lifecycle = match captured {
                Ok(lifecycle) => Arc::new(lifecycle.with_query_lease(self.root_lease.clone())),
                Err(error) => {
                    close_file(file);
                    #[cfg(not(target_arch = "wasm32"))]
                    let cleanup = file_directory.remove_file(&relative_name);
                    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
                    let cleanup = rustix::fs::unlinkat(
                        &file_directory,
                        &relative_name,
                        rustix::fs::AtFlags::empty(),
                    )
                    .map_err(std::io::Error::from);
                    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
                    let cleanup = std::fs::remove_file(&path);
                    #[cfg(target_os = "wasi")]
                    let cleanup = rustix::fs::unlinkat(
                        &file_directory,
                        &relative_name,
                        rustix::fs::AtFlags::empty(),
                    )
                    .map_err(std::io::Error::from);
                    return match cleanup {
                        Ok(()) => Err(error),
                        Err(cleanup_error)
                            if cleanup_error.kind() == std::io::ErrorKind::NotFound =>
                        {
                            Err(error)
                        }
                        Err(cleanup_error) => {
                            ORPHAN_CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
                            Err(combine_io_errors(
                                error,
                                cleanup_error,
                                "created spill identity-capture cleanup",
                            ))
                        }
                    };
                }
            };
            let handle_lease = SpillHandleLease::acquire(Arc::clone(&lifecycle))?;
            if let Err(registration_error) =
                self.state
                    .register_staging(path.clone(), Arc::clone(&lifecycle), role)
            {
                close_file(file);
                return match lifecycle.delete_path(&path, true) {
                    Ok(()) => Err(registration_error),
                    Err(cleanup_error) => Err(combine_io_errors(
                        registration_error,
                        cleanup_error,
                        "spill registration cleanup",
                    )),
                };
            }
            let validation = lifecycle.validate_entry(&path);
            #[cfg(target_os = "macos")]
            let validation = validation.and_then(|()| super::validate_no_acl_grants(&file));
            if let Err(validation_error) = validation {
                close_file(file);
                return match lifecycle.delete_path(&path, true) {
                    Ok(()) => {
                        self.state.unregister(&path)?;
                        Err(validation_error)
                    }
                    Err(cleanup_error) => Err(combine_io_errors(
                        validation_error,
                        cleanup_error,
                        "spill construction cleanup",
                    )),
                };
            }
            return SpillFile::from_created_file(
                path,
                file,
                identity,
                role,
                self.limits,
                Arc::clone(&self.provider),
                Arc::clone(&self.state),
                Arc::clone(&self.io),
                lifecycle,
                handle_lease,
                writer_buffer,
                qualified_file_workspace_bound,
                owned_cleanup_complete,
            );
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create a unique spill file",
        ))
    }

    fn begin_create(&self) -> std::io::Result<QueryCreateLease<'_>> {
        let mut query_state = self.query_state.lock();
        if query_state.phase != QueryLifecycleState::Open {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "spill query is already closed",
            ));
        }
        query_state.active_creates = query_state
            .active_creates
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("active spill create count overflow"))?;
        drop(query_state);
        // Fence finish before authority validation, without holding the query
        // mutex across filesystem/provider work. Errors and unwinds release
        // the active create through the same guard as successful construction.
        let create_lease = QueryCreateLease {
            state: &self.query_state,
        };
        if let Some(lease) = &self.root_lease {
            lease.validate(true)?;
        }
        Ok(create_lease)
    }

    pub(crate) fn enable_profile_merge(&self) {
        self.state.profile_merge.store(true, Ordering::Relaxed);
    }

    pub(crate) fn profile_merge_enabled(&self) -> bool {
        self.state.profile_merge.load(Ordering::Relaxed)
    }

    /// Cumulative publication and active merge counters, preserved after deletion.
    #[must_use]
    pub fn profile_totals(&self) -> (u64, u64, u64, Option<u64>) {
        let ledger = self.state.ledger.lock();
        (
            ledger.total_published_bytes,
            ledger.total_runs,
            ledger.total_partitions,
            if cfg!(target_arch = "wasm32") || !self.profile_merge_enabled() {
                None
            } else {
                Some(ledger.merge_time_ns)
            },
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn record_merge_time(&self, elapsed: u64) {
        let mut ledger = self.state.ledger.lock();
        ledger.merge_time_ns = ledger.merge_time_ns.saturating_add(elapsed);
    }

    /// Returns total bytes in successfully published files.
    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.state.disk_stats().published_live_bytes
    }

    /// Returns a coherent snapshot of this manager's hard quota and logical
    /// spill-byte ledger.
    #[must_use]
    pub fn disk_stats(&self) -> SpillDiskStats {
        self.state.disk_stats()
    }

    /// Samples physical reservation history without filesystem reads or waits.
    /// Returns `None` while an accounting owner is busy.
    /// Compatibility managers have no authenticated physical reservation owner.
    #[must_use]
    pub fn physical_stats(&self) -> Option<SpillPhysicalStats> {
        let cleanup_failed = self.query_state.try_lock()?.cleanup_failed;
        let mut stats = self.root_lease.as_ref()?.physical_stats()?;
        stats.cleanup_failed = cleanup_failed;
        if cleanup_failed {
            stats.cleanup_debt_bytes = stats.reserved_bytes;
        }
        Some(stats)
    }

    /// Returns the number of tracked staging and published files.
    #[must_use]
    pub fn active_file_count(&self) -> usize {
        self.state.ledger.lock().files.len()
    }

    /// Returns monotonic process-local cleanup-failure telemetry.
    ///
    /// The counter covers affected artifacts or query leaves from explicit
    /// construction cleanup and best-effort Drop. Retries may increment it
    /// more than once, so it is not a count of unique stranded files.
    #[must_use]
    pub fn orphan_cleanup_failures() -> u64 {
        ORPHAN_CLEANUP_FAILURES.load(Ordering::Relaxed)
    }

    /// Explicitly deletes every tracked file.
    ///
    /// Failed paths and their full staging/published byte charge remain
    /// registered for a later retry. A missing canonical path is treated as
    /// deletion success under the manager's documented stable-namespace
    /// precondition.
    ///
    /// # Errors
    ///
    /// Returns the first cleanup error kind with all failed-file contexts.
    pub fn cleanup(&self) -> std::io::Result<()> {
        let Some(_operation_lease) = self.begin_finish()? else {
            return Ok(());
        };
        self.cleanup_unlocked()
    }

    fn cleanup_unlocked(&self) -> std::io::Result<()> {
        let tracked: Vec<(PathBuf, Arc<SpillFileLifecycle>)> = self
            .state
            .ledger
            .lock()
            .files
            .iter()
            .map(|(path, tracked)| (path.clone(), Arc::clone(&tracked.lifecycle)))
            .collect();
        let mut failures = Vec::new();
        let mut first_kind = None;
        for (path, lifecycle) in tracked {
            match self.state.io.check(SpillIoOperation::Delete) {
                Err(error) => {
                    first_kind.get_or_insert(error.kind());
                    failures.push(format!("{}: {error}", path.display()));
                }
                Ok(()) => match lifecycle.delete_path(&path, false) {
                    Ok(()) => {
                        self.state.unregister(&path)?;
                    }
                    Err(error) => {
                        first_kind.get_or_insert(error.kind());
                        failures.push(format!("{}: {error}", path.display()));
                    }
                },
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(std::io::Error::new(
                first_kind.unwrap_or(std::io::ErrorKind::Other),
                format!("spill cleanup failed: {}", failures.join("; ")),
            ))
        }
    }

    /// Best-effort tracked-file cleanup used only from `Drop`.
    ///
    /// Explicit cleanup keeps its aggregated diagnostics. This path instead
    /// isolates every direct operation and forgets opaque failures so hostile
    /// destructors cannot run while another panic is already unwinding.
    fn cleanup_for_drop(&self) -> Result<(), ()> {
        let _operation_lease = match self.begin_finish() {
            Ok(Some(lease)) => lease,
            Ok(None) => return Ok(()),
            Err(_error) => return Err(()),
        };
        if self.cleanup_unlocked_for_drop() {
            Ok(())
        } else {
            Err(())
        }
    }

    fn cleanup_unlocked_for_drop(&self) -> bool {
        let tracked: Vec<(PathBuf, Arc<SpillFileLifecycle>)> = self
            .state
            .ledger
            .lock()
            .files
            .iter()
            .map(|(path, tracked)| (path.clone(), Arc::clone(&tracked.lifecycle)))
            .collect();
        let mut succeeded = true;
        for (path, lifecycle) in tracked {
            if !super::run_cleanup_backstop(|| self.state.io.check(SpillIoOperation::Delete)) {
                succeeded = false;
                continue;
            }
            if !super::run_cleanup_backstop(|| lifecycle.delete_path(&path, false)) {
                succeeded = false;
                continue;
            }
            if !super::run_cleanup_backstop(|| self.state.unregister(&path)) {
                succeeded = false;
            }
        }
        succeeded
    }

    /// Explicitly completes an owned query after deleting all tracked files.
    ///
    /// Unknown content blocks the final non-recursive directory removal.
    ///
    /// # Errors
    ///
    /// Returns an error while files/creates are active, when ownership/content
    /// validation fails, or when marker/directory cleanup fails.
    pub fn finish_query(&self) -> std::io::Result<()> {
        let result = (|| {
            let Some(mut finish_lease) = self.begin_finish()? else {
                return Ok(());
            };
            let result = self.finish_query_after_transition();
            if result.is_ok() {
                finish_lease.commit_closed();
            }
            result
        })();
        self.query_state.lock().cleanup_failed = result.is_err();
        result
    }

    fn begin_finish(&self) -> std::io::Result<Option<QueryFinishLease<'_>>> {
        let mut state = self.query_state.lock();
        match state.phase {
            QueryLifecycleState::Closed => Ok(None),
            QueryLifecycleState::Finishing => Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "spill query is already finishing",
            )),
            QueryLifecycleState::Poisoned => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "spill query cleanup is poisoned and requires external recovery",
            )),
            QueryLifecycleState::Open if state.active_creates != 0 => Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "spill query has active file creation",
            )),
            #[cfg(all(
                any(target_os = "linux", target_os = "macos"),
                not(target_arch = "wasm32")
            ))]
            QueryLifecycleState::Releasing => {
                state.phase = QueryLifecycleState::Finishing;
                Ok(Some(QueryFinishLease {
                    state: &self.query_state,
                    committed: false,
                }))
            }
            QueryLifecycleState::Open => {
                state.phase = QueryLifecycleState::Finishing;
                Ok(Some(QueryFinishLease {
                    state: &self.query_state,
                    committed: false,
                }))
            }
        }
    }

    fn finish_query_after_transition(&self) -> std::io::Result<()> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        if let Some(lease) = &self.root_lease
            && lease.deletion_pending()
        {
            self.query_state.lock().phase = QueryLifecycleState::Releasing;
            return lease.release_deleted_leaf();
        }
        // begin_finish fences new file creation. An empty ledger cannot gain
        // files, so only a nonempty ledger needs the pre-deletion authority check.
        if self.active_file_count() != 0
            && let Some(lease) = &self.root_lease
        {
            lease.validate(false)?;
        }
        self.cleanup_unlocked()?;
        if !self.owns_dir {
            return Ok(());
        }
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        self.directory_identity.validate_path(&self.spill_dir)?;
        let marker_path = self.spill_dir.join(OWNER_MARKER);
        #[cfg(not(target_arch = "wasm32"))]
        let query_directory = self.open_owned_query_directory()?;
        #[cfg(target_os = "wasi")]
        let query_directory = self.open_owned_query_directory()?;
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let query_directory = self.open_owned_query_directory()?;
        // Data-deletion hooks may mutate the seal. Authenticate again after
        // cleanup, then use the exact authenticated bytes for restoration.
        let marker = self
            .root_lease
            .as_ref()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "owned spill query has no authenticated lease",
                )
            })?
            .validated_marker()?;
        let leaf_name = self
            .spill_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| std::io::Error::other("owned spill query leaf has no UTF-8 name"))?;
        let parsed_marker = parse_owner_marker(&marker)?;
        if !marker_matches_leaf(leaf_name, parsed_marker) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "owned spill marker does not match query leaf",
            ));
        }
        {
            #[cfg(not(target_arch = "wasm32"))]
            let entries = query_directory.entries()?;
            #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
            let entries = std::fs::read_dir(&self.spill_dir)?;
            #[cfg(any(
                not(target_arch = "wasm32"),
                all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))
            ))]
            for entry in entries {
                let entry = entry?;
                let is_marker = entry.file_name() == std::ffi::OsStr::new(OWNER_MARKER);
                if !is_marker {
                    let display = entry.file_name().to_string_lossy().into_owned();
                    return Err(std::io::Error::other(format!(
                        "unknown content prevents spill query completion: {}",
                        display
                    )));
                }
            }
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            emscripten_visit_directory(&query_directory, |entry| {
                if entry != OWNER_MARKER.as_bytes() {
                    return Err(std::io::Error::other(format!(
                        "unknown content prevents spill query completion: {}",
                        String::from_utf8_lossy(entry)
                    )));
                }
                Ok(())
            })?;
            #[cfg(target_os = "wasi")]
            wasi_visit_directory(&query_directory, |entry| {
                if entry != OWNER_MARKER.as_bytes() {
                    return Err(std::io::Error::other(format!(
                        "unknown content prevents spill query completion: {}",
                        String::from_utf8_lossy(entry)
                    )));
                }
                Ok(())
            })?;
        }
        remove_owner_marker(
            &self.spill_dir,
            #[cfg(not(target_arch = "wasm32"))]
            &query_directory,
            #[cfg(not(target_arch = "wasm32"))]
            Path::new(OWNER_MARKER),
            #[cfg(target_os = "wasi")]
            &query_directory,
            #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
            &query_directory,
        )?;
        // The owned leaf is markerless from this point. Fail closed before
        // invoking another public hook or filesystem operation so an unwind
        // cannot make the query writable again.
        self.query_state.lock().phase = QueryLifecycleState::Poisoned;
        #[cfg(not(target_arch = "wasm32"))]
        let remove_result = self
            .io
            .check(SpillIoOperation::RemoveQueryDirectory)
            .and_then(|()| {
                use cap_fs_ext::MetadataExt as _;
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                if self.root_lease.is_some() {
                    validate_query_parent(&query_directory, &self.directory)?;
                }
                if let Some(leaf) = self.query_leaf.as_ref() {
                    match self.directory.symlink_metadata(leaf) {
                        Ok(metadata)
                            if metadata.is_dir()
                                && metadata.dev() == self.directory_identity.device
                                && metadata.ino() == self.directory_identity.inode =>
                        {
                            // Recheck the known name after the foreign remove hook.
                            // The same stable-namespace check/unlink contract as
                            // remove_open_dir avoids scanning all sibling queries.
                            return self.directory.remove_dir(leaf);
                        }
                        Ok(_) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => return Err(error),
                    }
                }
                // A renamed original must still be located through its retained
                // handle; never remove a foreign replacement at the old name.
                query_directory.remove_open_dir()
            });
        #[cfg(target_os = "wasi")]
        let remove_result = self
            .io
            .check(SpillIoOperation::RemoveQueryDirectory)
            .and_then(|()| {
                drop(query_directory);
                let leaf = self
                    .wasi_query_leaf
                    .as_ref()
                    .ok_or_else(|| std::io::Error::other("owned spill query has no leaf"))?;
                rustix::fs::unlinkat(&self.wasi_directory, leaf, rustix::fs::AtFlags::REMOVEDIR)
                    .map_err(std::io::Error::from)
            });
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let remove_result = self
            .io
            .check(SpillIoOperation::RemoveQueryDirectory)
            .and_then(|()| {
                drop(query_directory);
                let leaf = self.emscripten_query_leaf.as_ref().ok_or_else(|| {
                    std::io::Error::other("owned Emscripten spill query has no leaf")
                })?;
                rustix::fs::unlinkat(
                    &self.emscripten_directory,
                    leaf,
                    rustix::fs::AtFlags::REMOVEDIR,
                )
                .map_err(std::io::Error::from)
            });
        #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
        let remove_result = self
            .io
            .check(SpillIoOperation::RemoveQueryDirectory)
            .and_then(|()| std::fs::remove_dir(&self.spill_dir));
        match remove_result {
            Ok(()) => {
                #[cfg(all(
                    any(target_os = "linux", target_os = "macos"),
                    not(target_arch = "wasm32")
                ))]
                if let Some(lease) = &self.root_lease {
                    self.query_state.lock().phase = QueryLifecycleState::Releasing;
                    lease.release_deleted_leaf()?;
                }
                Ok(())
            }
            Err(remove_error) => {
                // A public remove hook can return an error with an arbitrary
                // destructor, then the restore hook can panic. Keep the first
                // error out of unwind drop glue until the second operation has
                // settled, otherwise two hostile destructors can abort before
                // the caller's unwind boundary sees either failure.
                let remove_error = std::mem::ManuallyDrop::new(remove_error);
                let restore_outcome =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut restored_marker_receipt = None;
                        #[cfg(not(target_arch = "wasm32"))]
                        let restore_result = self
                            .io
                            .check(SpillIoOperation::RestoreOwnerMarker)
                            .and_then(|()| self.open_owned_query_directory())
                            .and_then(|directory| {
                                restore_owner_marker(
                                    &marker_path,
                                    &marker,
                                    &Arc::new(directory),
                                    Path::new(OWNER_MARKER),
                                    Some(&mut restored_marker_receipt),
                                )
                            });
                        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
                        let restore_result = self
                            .io
                            .check(SpillIoOperation::RestoreOwnerMarker)
                            .and_then(|()| self.open_owned_query_directory())
                            .and_then(|directory| {
                                restore_owner_marker(
                                    &marker_path,
                                    &marker,
                                    &directory,
                                    Some(&mut restored_marker_receipt),
                                )
                            });
                        #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
                        let restore_result = self
                            .io
                            .check(SpillIoOperation::RestoreOwnerMarker)
                            .and_then(|()| self.directory_identity.validate_path(&self.spill_dir))
                            .and_then(|()| {
                                restore_owner_marker(
                                    &marker_path,
                                    &marker,
                                    Some(&mut restored_marker_receipt),
                                )
                            });
                        #[cfg(target_os = "wasi")]
                        let restore_result = self
                            .io
                            .check(SpillIoOperation::RestoreOwnerMarker)
                            .and_then(|()| self.open_owned_query_directory())
                            .and_then(|directory| {
                                restore_owner_marker(
                                    &marker_path,
                                    &marker,
                                    &directory,
                                    Some(&mut restored_marker_receipt),
                                )
                            });
                        restore_result.and_then(|()| {
                            #[cfg(all(
                                test,
                                any(target_os = "linux", target_os = "macos"),
                                not(target_arch = "wasm32")
                            ))]
                            if let Some(hook) =
                                AFTER_OWNER_MARKER_RESTORE.with(|slot| slot.borrow_mut().take())
                            {
                                hook();
                            }
                            self.root_lease.as_ref().map_or(Ok(()), |lease| {
                                let receipt =
                                    restored_marker_receipt.as_ref().ok_or_else(|| {
                                        std::io::Error::other(
                                            "restored spill seal lacks creation receipt",
                                        )
                                    })?;
                                lease.rebind_restored_marker(&receipt._file)
                            })
                        })
                    }));
                match restore_outcome {
                    Ok(Ok(())) => {
                        self.query_state.lock().phase = QueryLifecycleState::Finishing;
                        Err(std::mem::ManuallyDrop::into_inner(remove_error))
                    }
                    Ok(Err(restore_error)) => {
                        let restore_error = std::mem::ManuallyDrop::new(restore_error);
                        self.query_state.lock().phase = QueryLifecycleState::Poisoned;
                        Err(combine_io_errors(
                            std::mem::ManuallyDrop::into_inner(remove_error),
                            std::mem::ManuallyDrop::into_inner(restore_error),
                            "owner marker restore",
                        ))
                    }
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn open_owned_query_directory(&self) -> std::io::Result<CapabilityDir> {
        let leaf = self
            .query_leaf
            .as_ref()
            .ok_or_else(|| std::io::Error::other("owned spill query has no leaf"))?;
        let directory = self.directory.open_dir_nofollow(leaf)?;
        self.directory_identity.validate_capability(&directory)?;
        Ok(directory)
    }

    #[cfg(target_os = "wasi")]
    pub(super) fn open_owned_query_directory(&self) -> std::io::Result<File> {
        let leaf = self
            .wasi_query_leaf
            .as_ref()
            .ok_or_else(|| std::io::Error::other("owned spill query has no leaf"))?;
        let directory = wasi_open_directory_at(&self.wasi_directory, leaf)?;
        self.directory_identity.validate_wasi_handle(&directory)?;
        Ok(directory)
    }

    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    pub(super) fn open_owned_query_directory(&self) -> std::io::Result<File> {
        let leaf = self
            .emscripten_query_leaf
            .as_ref()
            .ok_or_else(|| std::io::Error::other("owned spill query has no leaf"))?;
        let directory = emscripten_open_directory_at(&self.emscripten_directory, leaf)?;
        self.directory_identity
            .validate_directory_handle(&directory)?;
        Ok(directory)
    }
}

impl Drop for SpillManager {
    fn drop(&mut self) {
        let stranded_before = u64::try_from(self.active_file_count()).unwrap_or(u64::MAX);
        if !super::run_cleanup_backstop(|| self.cleanup_for_drop()) {
            record_orphan_cleanup_failures(stranded_before);
            return;
        }
        if self.owns_dir && !super::run_cleanup_backstop(|| self.finish_query()) {
            record_orphan_cleanup_failures(1);
        }
    }
}

impl std::fmt::Debug for SpillManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpillManager")
            .field("spill_dir", &self.spill_dir)
            .field("active_files", &self.active_file_count())
            .field("spilled_bytes", &self.spilled_bytes())
            .field("owns_dir", &self.owns_dir)
            .finish()
    }
}

#[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
#[cfg(test)]
fn ensure_directory(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => validate_existing_directory(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(path)?;
            validate_existing_directory(path)
        }
        Err(error) => Err(error),
    }
}

#[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
pub(super) fn ensure_production_directory(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => validate_existing_directory(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt as _;
                builder.mode(0o700);
            }
            builder.create(path)?;
            validate_existing_directory(path)
        }
        Err(error) => Err(error),
    }
}

fn validate_existing_directory(path: &Path) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("spill directory {} is a symlink", path.display()),
        ));
    }
    if !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("spill path {} is not a directory", path.display()),
        ));
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
pub(super) fn validate_production_spill_root(path: &Path) -> std::io::Result<()> {
    validate_existing_directory(path)?;
    #[cfg(unix)]
    {
        let metadata = std::fs::metadata(path)?;
        validate_unix_spill_root_metadata(&metadata, path)?;
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn validate_unix_spill_root_metadata(
    metadata: &std::fs::Metadata,
    path: &Path,
) -> std::io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let effective_uid = rustix::process::geteuid().as_raw();
    if unix_spill_root_policy(metadata.uid(), effective_uid, metadata.mode()) {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "spill root {} lacks trusted ownership or sticky shared-directory semantics",
            path.display()
        ),
    ))
}

#[cfg(unix)]
const fn unix_spill_root_policy(owner_uid: u32, effective_uid: u32, mode: u32) -> bool {
    let trusted_owner = owner_uid == 0 || owner_uid == effective_uid;
    let shared_writable = mode & 0o022 != 0;
    trusted_owner && (!shared_writable || mode & 0o1000 != 0)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn create_owner_only_directory_at(
    root: &CapabilityDir,
    leaf: &str,
) -> std::io::Result<()> {
    #[cfg(not(unix))]
    let builder = CapabilityDirBuilder::new();
    #[cfg(unix)]
    let mut builder = CapabilityDirBuilder::new();
    #[cfg(unix)]
    builder.mode(0o700);
    root.create_dir_with(leaf, &builder)
}

fn write_marker_bytes(
    _path: &Path,
    marker: &[u8],
    io: Option<&dyn SpillIo>,
    #[cfg(not(target_arch = "wasm32"))] directory: &Arc<CapabilityDir>,
    #[cfg(not(target_arch = "wasm32"))] marker_relative: &Path,
    #[cfg(target_os = "wasi")] wasi_directory: &File,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))] emscripten_directory: &File,
    mut marker_receipt: Option<&mut Option<MarkerCreationReceipt>>,
) -> std::io::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let file = {
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        directory.open_with(marker_relative, &options)?.into_std()
    };
    #[cfg(target_os = "wasi")]
    let file = File::from(rustix::fs::openat(
        wasi_directory,
        OWNER_MARKER,
        rustix::fs::OFlags::WRONLY
            | rustix::fs::OFlags::CREATE
            | rustix::fs::OFlags::EXCL
            | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )?);
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    let file = File::from(rustix::fs::openat(
        emscripten_directory,
        OWNER_MARKER,
        rustix::fs::OFlags::WRONLY
            | rustix::fs::OFlags::CREATE
            | rustix::fs::OFlags::EXCL
            | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )?);
    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(_path)?;
    let file = Arc::new(file);
    if let Some(receipt) = marker_receipt.as_deref_mut() {
        #[cfg(not(target_arch = "wasm32"))]
        let lifecycle = SpillFileLifecycle::capture_with_directory(
            &file,
            Arc::clone(directory),
            marker_relative.to_owned(),
        )?;
        #[cfg(target_os = "wasi")]
        let lifecycle = SpillFileLifecycle::capture_with_wasi_directory(
            &file,
            Arc::new(wasi_directory.try_clone()?),
            PathBuf::from(OWNER_MARKER),
        )?;
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        let lifecycle = SpillFileLifecycle::capture_with_emscripten_directory(
            &file,
            Arc::new(emscripten_directory.try_clone()?),
            PathBuf::from(OWNER_MARKER),
        )?;
        #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
        let lifecycle = SpillFileLifecycle::capture(&file, _path)?;
        *receipt = Some(MarkerCreationReceipt {
            _file: Arc::clone(&file),
            lifecycle,
        });
    }
    #[cfg(target_os = "macos")]
    super::validate_no_acl_grants(file.as_ref())?;
    if let Some(io) = io {
        io.check(SpillIoOperation::WritePayload)?;
    }
    file.as_ref().write_all(marker)?;
    if let Some(io) = io {
        io.check(SpillIoOperation::Flush)?;
    }
    file.as_ref().flush()?;
    if let Some(io) = io {
        io.check(SpillIoOperation::Sync)?;
    }
    file.sync_all()?;
    if let Some(receipt) = marker_receipt.as_ref().and_then(|slot| slot.as_ref()) {
        receipt.lifecycle.validate_entry(_path)?;
    }
    Ok(())
}

fn restore_owner_marker(
    path: &Path,
    marker: &[u8],
    #[cfg(not(target_arch = "wasm32"))] directory: &Arc<CapabilityDir>,
    #[cfg(not(target_arch = "wasm32"))] marker_relative: &Path,
    #[cfg(target_os = "wasi")] wasi_directory: &File,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))] emscripten_directory: &File,
    marker_receipt: Option<&mut Option<MarkerCreationReceipt>>,
) -> std::io::Result<()> {
    write_marker_bytes(
        path,
        marker,
        None,
        #[cfg(not(target_arch = "wasm32"))]
        directory,
        #[cfg(not(target_arch = "wasm32"))]
        marker_relative,
        #[cfg(target_os = "wasi")]
        wasi_directory,
        #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
        emscripten_directory,
        marker_receipt,
    )
}

fn combine_io_errors(
    primary: std::io::Error,
    cleanup: std::io::Error,
    cleanup_context: &'static str,
) -> std::io::Error {
    super::combine_primary_and_cleanup(primary, cleanup, cleanup_context)
}

#[cfg(all(test, unix, not(target_arch = "wasm32")))]
fn read_owner_marker(
    _path: &Path,
    #[cfg(not(target_arch = "wasm32"))] directory: &CapabilityDir,
    #[cfg(not(target_arch = "wasm32"))] marker_relative: &Path,
    #[cfg(target_os = "wasi")] wasi_directory: &File,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))] emscripten_directory: &File,
) -> std::io::Result<[u8; OWNER_MARKER_BYTES]> {
    #[cfg(not(target_arch = "wasm32"))]
    let mut file = {
        let mut options = cap_std::fs::OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No).nonblock(true);
        directory.open_with(marker_relative, &options)?.into_std()
    };
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    let mut file = File::from(rustix::fs::openat(
        emscripten_directory,
        OWNER_MARKER,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK,
        rustix::fs::Mode::empty(),
    )?);
    #[cfg(target_os = "wasi")]
    let mut file = File::from(rustix::fs::openat(
        wasi_directory,
        OWNER_MARKER,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK,
        rustix::fs::Mode::empty(),
    )?);
    let metadata = file.metadata()?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != OWNER_MARKER_BYTES as u64
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid spill owner marker shape",
        ));
    }
    let mut marker = [0u8; OWNER_MARKER_BYTES];
    file.read_exact(&mut marker)?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "trailing spill owner marker bytes",
        ));
    }
    Ok(marker)
}

fn remove_owner_marker(
    _path: &Path,
    #[cfg(not(target_arch = "wasm32"))] directory: &CapabilityDir,
    #[cfg(not(target_arch = "wasm32"))] marker_relative: &Path,
    #[cfg(target_os = "wasi")] wasi_directory: &File,
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))] emscripten_directory: &File,
) -> std::io::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let remove_result = directory.remove_file(marker_relative);
    #[cfg(all(target_arch = "wasm32", not(any(unix, target_os = "wasi"))))]
    let remove_result = std::fs::remove_file(_path.join(OWNER_MARKER));
    #[cfg(all(target_arch = "wasm32", unix, not(target_os = "wasi")))]
    let remove_result = rustix::fs::unlinkat(
        emscripten_directory,
        OWNER_MARKER,
        rustix::fs::AtFlags::empty(),
    )
    .map_err(std::io::Error::from);
    #[cfg(target_os = "wasi")]
    let remove_result =
        rustix::fs::unlinkat(wasi_directory, OWNER_MARKER, rustix::fs::AtFlags::empty())
            .map_err(std::io::Error::from);
    match remove_result {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub(super) fn is_query_leaf_name(name: &str) -> bool {
    name.len() == QUERY_PREFIX.len() + 32
        && name.starts_with(QUERY_PREFIX)
        && name[QUERY_PREFIX.len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub(super) fn marker_matches_leaf(name: &str, identity: SpillQueryIdentity) -> bool {
    if !is_query_leaf_name(name) {
        return false;
    }
    let suffix = &name[QUERY_PREFIX.len()..];
    suffix
        .as_bytes()
        .chunks_exact(2)
        .zip(identity.as_bytes())
        .all(|(hex, expected)| decode_hex_byte(hex) == Some(*expected))
}

fn parse_owner_marker(marker: &[u8; OWNER_MARKER_BYTES]) -> std::io::Result<SpillQueryIdentity> {
    if marker[..4] != OWNER_MAGIC || marker[4] != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid spill owner marker magic/version",
        ));
    }
    let identity = SpillQueryIdentity::from_bytes(
        marker[5..OWNER_MARKER_PREFIX_BYTES]
            .try_into()
            .expect("fixed query identity bytes"),
    );
    Ok(identity)
}

fn decode_hex_byte(hex: &[u8]) -> Option<u8> {
    if hex.len() != 2 {
        return None;
    }
    Some(hex_value(hex[0])? << 4 | hex_value(hex[1])?)
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(not(all(target_arch = "wasm32", not(any(unix, target_os = "wasi")))))]
pub(super) fn is_spill_artifact_name(name: &str) -> bool {
    let Some((stem, extension)) = name.rsplit_once('.') else {
        return false;
    };
    if extension != "grsp" {
        return false;
    }
    [
        SpillFileRole::SortRun.file_prefix(),
        SpillFileRole::NativePartition.file_prefix(),
        SpillFileRole::RdfAggregateState.file_prefix(),
    ]
    .iter()
    .any(|prefix| {
        let expected = prefix.len() + 1 + 32;
        stem.len() == expected
            && stem.starts_with(prefix)
            && stem.as_bytes()[prefix.len()] == b'-'
            && stem[prefix.len() + 1..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

#[cfg(test)]
mod tests {
    use super::super::{RootedSpillFixture, SPILL_RECORD_HEADER_BYTES};
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Barrier;
    use std::sync::mpsc;
    use tempfile::TempDir;

    #[cfg(unix)]
    fn create_fifo(path: &Path) {
        let status = std::process::Command::new("mkfifo")
            .arg(path)
            .status()
            .expect("mkfifo must be available for Unix special-file containment tests");
        assert!(status.success(), "mkfifo failed for {}", path.display());
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn assert_drop_child_survives(test_name: &str, child_env: &str, handshake: &str) {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .env(child_env, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(handshake),
            "best-effort Drop child did not complete its exact scenario\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    struct BlockingIo {
        operation: SpillIoOperation,
        entered: mpsc::SyncSender<()>,
        release: Mutex<mpsc::Receiver<()>>,
        blocked: std::sync::atomic::AtomicBool,
    }

    struct DeterministicIdentitySource {
        files: Mutex<VecDeque<SpillFileIdentity>>,
    }

    impl DeterministicIdentitySource {
        fn new(files: impl IntoIterator<Item = [u8; 16]>) -> Self {
            Self {
                files: Mutex::new(
                    files
                        .into_iter()
                        .map(SpillFileIdentity::from_bytes)
                        .collect(),
                ),
            }
        }
    }

    impl SpillIdentitySource for DeterministicIdentitySource {
        fn next_file_identity(&self) -> std::io::Result<SpillFileIdentity> {
            self.files
                .lock()
                .pop_front()
                .ok_or_else(|| std::io::Error::other("file identity test source exhausted"))
        }
    }

    impl SpillIo for BlockingIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            if operation == self.operation
                && !self.blocked.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                self.entered.send(()).unwrap();
                self.release.lock().recv().unwrap();
            }
            Ok(())
        }
    }

    struct PanicDeleteIo;

    #[derive(Debug)]
    struct PanicOnDrop(&'static str);

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("{}", self.0);
        }
    }

    struct HostilePanicDeleteIo {
        fired: std::sync::atomic::AtomicBool,
    }

    impl HostilePanicDeleteIo {
        fn new() -> Self {
            Self {
                fired: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl SpillIo for HostilePanicDeleteIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            if operation == SpillIoOperation::Delete
                && !self.fired.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                std::panic::panic_any(PanicOnDrop("delete panic payload dropped"));
            }
            Ok(())
        }
    }

    struct DoublePanicOnDrop {
        _nested: PanicOnDrop,
    }

    impl Drop for DoublePanicOnDrop {
        fn drop(&mut self) {
            panic!("outer delete panic payload dropped");
        }
    }

    struct HostileDoublePanicDeleteIo {
        fired: std::sync::atomic::AtomicBool,
    }

    impl HostileDoublePanicDeleteIo {
        fn new() -> Self {
            Self {
                fired: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl SpillIo for HostileDoublePanicDeleteIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            if operation == SpillIoOperation::Delete
                && !self.fired.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                std::panic::panic_any(DoublePanicOnDrop {
                    _nested: PanicOnDrop("nested delete panic payload dropped"),
                });
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct HostileError;

    impl std::fmt::Display for HostileError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("hostile cleanup error")
        }
    }

    impl std::error::Error for HostileError {}

    impl Drop for HostileError {
        fn drop(&mut self) {
            panic!("delete error payload dropped");
        }
    }

    #[derive(Debug)]
    struct DoublePanicError {
        _nested: PanicOnDrop,
    }

    impl std::fmt::Display for DoublePanicError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("hostile nested cleanup error")
        }
    }

    impl std::error::Error for DoublePanicError {}

    impl Drop for DoublePanicError {
        fn drop(&mut self) {
            panic!("outer delete error payload dropped");
        }
    }

    struct HostileErrorDeleteIo {
        fired: std::sync::atomic::AtomicBool,
    }

    impl HostileErrorDeleteIo {
        fn new() -> Self {
            Self {
                fired: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl SpillIo for HostileErrorDeleteIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            if operation == SpillIoOperation::Delete
                && !self.fired.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    HostileError,
                ));
            }
            Ok(())
        }
    }

    struct HostileDoubleErrorDeleteIo {
        fired: std::sync::atomic::AtomicBool,
    }

    impl HostileDoubleErrorDeleteIo {
        fn new() -> Self {
            Self {
                fired: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl SpillIo for HostileDoubleErrorDeleteIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            if operation == SpillIoOperation::Delete
                && !self.fired.swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    DoublePanicError {
                        _nested: PanicOnDrop("nested delete error payload dropped"),
                    },
                ));
            }
            Ok(())
        }
    }

    struct HostileFinishErrorThenPanicIo;

    impl SpillIo for HostileFinishErrorThenPanicIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            match operation {
                SpillIoOperation::RemoveQueryDirectory => Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    DoublePanicError {
                        _nested: PanicOnDrop("nested query-remove error payload dropped"),
                    },
                )),
                SpillIoOperation::RestoreOwnerMarker => {
                    panic!("owner-marker restore hook panic")
                }
                _ => Ok(()),
            }
        }
    }

    #[derive(Debug)]
    struct PrimaryDropPanic;

    impl SpillIo for PanicDeleteIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            assert_ne!(operation, SpillIoOperation::Delete, "delete-hook panic");
            Ok(())
        }
    }

    struct FailQueryFinishIo;

    impl SpillIo for FailQueryFinishIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            match operation {
                SpillIoOperation::RemoveQueryDirectory => Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "query remove failpoint",
                )),
                SpillIoOperation::RestoreOwnerMarker => Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "marker restore failpoint",
                )),
                _ => Ok(()),
            }
        }
    }

    struct PanicQueryFinishIo {
        operation: SpillIoOperation,
    }

    impl SpillIo for PanicQueryFinishIo {
        fn check(&self, operation: SpillIoOperation) -> std::io::Result<()> {
            assert_ne!(operation, self.operation, "query-finish hook panic");
            if self.operation == SpillIoOperation::RestoreOwnerMarker
                && operation == SpillIoOperation::RemoveQueryDirectory
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "force marker restoration",
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn prepared_writer_backing_moves_through_manager_without_reallocation() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();
        let prepared = SpillWriterBuffer::prepare().unwrap();
        let identity = (prepared.pointer(), prepared.capacity());

        let mut file = manager
            .create_file_with_writer_buffer(SpillFileRole::SortRun, prepared)
            .unwrap();

        assert_eq!(file.writer_buffer_identity(), Some(identity));
        file.close_and_delete().unwrap();
    }

    #[test]
    fn exact_disk_quota_includes_file_end_and_publication_is_charge_neutral() {
        const EMPTY_SORT_BYTES: u64 = 3 * SPILL_RECORD_HEADER_BYTES as u64 + 8 + 12 + 8;

        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .quota(SpillDiskQuota::new(EMPTY_SORT_BYTES))
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();

        assert_eq!(
            manager.disk_stats(),
            SpillDiskStats {
                limit_bytes: EMPTY_SORT_BYTES,
                reserved_live_bytes: SPILL_RECORD_HEADER_BYTES as u64 + 8,
                published_live_bytes: 0,
                peak_reserved_bytes: SPILL_RECORD_HEADER_BYTES as u64 + 8,
            }
        );
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();

        assert_eq!(file.bytes_written(), EMPTY_SORT_BYTES);
        assert_eq!(manager.spilled_bytes(), EMPTY_SORT_BYTES);
        assert_eq!(
            manager.disk_stats(),
            SpillDiskStats {
                limit_bytes: EMPTY_SORT_BYTES,
                reserved_live_bytes: EMPTY_SORT_BYTES,
                published_live_bytes: EMPTY_SORT_BYTES,
                peak_reserved_bytes: EMPTY_SORT_BYTES,
            }
        );
        let before_invalid_reservation = manager.disk_stats();
        let error = manager.state.reserve_bytes(file.path(), 1).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(manager.disk_stats(), before_invalid_reservation);

        file.close_and_delete().unwrap();
        assert_eq!(
            manager.disk_stats(),
            SpillDiskStats {
                limit_bytes: EMPTY_SORT_BYTES,
                reserved_live_bytes: 0,
                published_live_bytes: 0,
                peak_reserved_bytes: EMPTY_SORT_BYTES,
            }
        );
    }

    #[test]
    fn quota_denial_mutates_no_ledger_bytes_and_staging_charge_lives_until_delete() {
        const BEFORE_FILE_END: u64 = 2 * SPILL_RECORD_HEADER_BYTES as u64 + 8 + 12;

        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .quota(SpillDiskQuota::new(BEFORE_FILE_END))
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();

        let before = manager.disk_stats();
        let error = file.finish_write().unwrap_err();
        let quota = error
            .get_ref()
            .and_then(|source| source.downcast_ref::<SpillQuotaExceeded>())
            .expect("quota errors remain typed through io::Error");

        assert_eq!(error.kind(), std::io::ErrorKind::QuotaExceeded);
        assert_eq!(quota.limit_bytes(), BEFORE_FILE_END);
        assert_eq!(quota.used_bytes(), BEFORE_FILE_END);
        assert_eq!(
            quota.requested_bytes(),
            SPILL_RECORD_HEADER_BYTES as u64 + 8
        );
        assert_eq!(manager.disk_stats(), before);
        assert_eq!(manager.spilled_bytes(), 0);

        file.close_and_delete().unwrap();
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(manager.disk_stats().peak_reserved_bytes, BEFORE_FILE_END);
    }

    #[test]
    fn concurrent_staging_reservations_cannot_overcommit_one_frame_quota() {
        const FILE_START_BYTES: u64 = SPILL_RECORD_HEADER_BYTES as u64 + 8;

        let directory = TempDir::new().unwrap();
        let manager = Arc::new(
            crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .quota(SpillDiskQuota::new(FILE_START_BYTES))
                .build()
                .unwrap(),
        );
        let start = Arc::new(Barrier::new(3));
        let completed = Arc::new(Barrier::new(3));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let manager = Arc::clone(&manager);
                let start = Arc::clone(&start);
                let completed = Arc::clone(&completed);
                std::thread::spawn(move || {
                    start.wait();
                    let result = manager.create_file(SpillFileRole::SortRun);
                    completed.wait();
                    result
                })
            })
            .collect();

        start.wait();
        completed.wait();
        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        let successes = results.iter().filter(|result| result.is_ok()).count();
        let quota_errors = results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .filter(|error| error.kind() == std::io::ErrorKind::QuotaExceeded)
            .count();

        assert_eq!(successes, 1);
        assert_eq!(quota_errors, 1);
        assert_eq!(manager.disk_stats().reserved_live_bytes, FILE_START_BYTES);
        drop(results);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
    }

    #[test]
    fn zero_quota_denies_initial_frame_without_a_ghost_or_ledger_mutation() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .quota(SpillDiskQuota::new(0))
            .build()
            .unwrap();

        let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::QuotaExceeded);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(
            manager.disk_stats(),
            SpillDiskStats {
                limit_bytes: 0,
                reserved_live_bytes: 0,
                published_live_bytes: 0,
                peak_reserved_bytes: 0,
            }
        );
        assert!(
            std::fs::read_dir(directory.path())
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[test]
    fn two_managers_in_one_directory_never_share_or_truncate_a_file() {
        let directory = TempDir::new().unwrap();
        let first_manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();
        let second_manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();

        let mut first = first_manager.create_file(SpillFileRole::SortRun).unwrap();
        first.write_sort_run_start(1, 0).unwrap();
        first.finish_write().unwrap();
        let first_bytes = std::fs::read(first.path()).unwrap();

        let second = second_manager.create_file(SpillFileRole::SortRun).unwrap();

        assert_ne!(first.path(), second.path());
        assert_eq!(std::fs::read(first.path()).unwrap(), first_bytes);
    }

    #[test]
    fn deterministic_file_collision_retries_without_truncation_or_ghosts() {
        let directory = TempDir::new().unwrap();
        let colliding = [0x11; 16];
        let fresh = [0x22; 16];
        let colliding_name = format!(
            "sort-run-{}.grsp",
            SpillFileIdentity::from_bytes(colliding).hex()
        );
        let colliding_path = directory.path().join(colliding_name);
        std::fs::write(&colliding_path, b"preserve").unwrap();
        let manager =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path().to_path_buf())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(NoopSpillIo))
                .quota(SpillDiskQuota::new(u64::MAX))
                .identities(Arc::new(DeterministicIdentitySource::new([
                    colliding, fresh,
                ])))
                .build()
                .unwrap();

        let file = manager.create_file(SpillFileRole::SortRun).unwrap();

        assert_eq!(file.identity(), SpillFileIdentity::from_bytes(fresh));
        assert_eq!(std::fs::read(&colliding_path).unwrap(), b"preserve");
        assert_eq!(manager.active_file_count(), 1);
        drop(file);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn deterministic_file_collision_exhaustion_is_clean_and_non_destructive() {
        let directory = TempDir::new().unwrap();
        let colliding = [0x33; 16];
        let colliding_path = directory.path().join(format!(
            "sort-run-{}.grsp",
            SpillFileIdentity::from_bytes(colliding).hex()
        ));
        std::fs::write(&colliding_path, b"preserve").unwrap();
        let manager =
            crate::execution::spill::BorrowedSpillFixture::new(directory.path().to_path_buf())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(NoopSpillIo))
                .quota(SpillDiskQuota::new(u64::MAX))
                .identities(Arc::new(DeterministicIdentitySource::new(
                    std::iter::repeat_n(colliding, FILE_CREATE_ATTEMPTS),
                )))
                .build()
                .unwrap();

        let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&colliding_path).unwrap(), b"preserve");
        assert_eq!(manager.active_file_count(), 0);
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn deterministic_query_collision_exhausts_without_touching_owner_then_fresh_identity_succeeds()
    {
        let parent = TempDir::new().unwrap();
        let root = RootedSpillFixture::new(parent.path()).root().unwrap();
        let colliding = SpillQueryIdentity::from_bytes([0x44; 16]);
        let fresh = SpillQueryIdentity::from_bytes([0x55; 16]);
        let colliding_path = root
            .namespace_path()
            .join(format!("{QUERY_PREFIX}{}", colliding.hex()));
        std::fs::create_dir(&colliding_path).unwrap();
        std::fs::write(colliding_path.join("sentinel"), b"preserve").unwrap();
        let control = crate::execution::QueryExecutionControl::new();
        let context = crate::execution::QueryResourceContext::new_with_cancellation(
            grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20),
            control.token(),
        )
        .unwrap();
        let error = SpillManager::create_from_root(
            &root,
            colliding,
            Arc::new(CleartextSpillRecordProvider),
            context.query_id(),
            control.token(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read(colliding_path.join("sentinel")).unwrap(),
            b"preserve"
        );
        let manager = SpillManager::create_from_root(
            &root,
            fresh,
            Arc::new(CleartextSpillRecordProvider),
            context.query_id(),
            control.token(),
        )
        .unwrap();
        assert_eq!(
            manager.spill_dir().file_name().unwrap(),
            format!("{QUERY_PREFIX}{}", fresh.hex()).as_str()
        );
        manager.finish_query().unwrap();
        assert_eq!(
            std::fs::read(colliding_path.join("sentinel")).unwrap(),
            b"preserve"
        );
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn exclusive_query_directories_are_distinct_and_owned() {
        let root = TempDir::new().unwrap();
        let first = RootedSpillFixture::new(root.path()).build().unwrap();
        let second = RootedSpillFixture::new(root.path()).build().unwrap();

        assert_ne!(first.spill_dir(), second.spill_dir());
        assert_eq!(first.spill_dir().parent(), second.spill_dir().parent());
        assert_eq!(
            first.spill_dir().parent().unwrap().parent(),
            Some(root.path())
        );
        assert!(first.spill_dir().join(OWNER_MARKER).is_file());
    }

    #[test]
    fn cleanup_failure_retains_path_and_accounting_for_retry() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        let path = file.path().to_path_buf();
        let published = manager.spilled_bytes();
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();

        assert!(manager.cleanup().is_err());
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(manager.spilled_bytes(), published);

        std::fs::remove_dir(&path).unwrap();
        drop(file);
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
    }

    #[test]
    fn missing_path_releases_charge_under_the_stable_namespace_contract() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .quota(SpillDiskQuota::new(1024))
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        assert!(manager.disk_stats().reserved_live_bytes > 0);
        std::fs::remove_file(file.path()).unwrap();

        file.close_and_delete().unwrap();

        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.disk_stats().reserved_live_bytes, 0);
        assert_eq!(manager.spilled_bytes(), 0);
    }

    #[test]
    fn drop_cleanup_failure_records_orphan_telemetry() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("blocks-remove-file"), b"orphan").unwrap();
        let before = SpillManager::orphan_cleanup_failures();

        drop(file);
        drop(manager);

        assert!(SpillManager::orphan_cleanup_failures() > before);
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn finish_query_preserves_marker_when_unknown_content_blocks_removal() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let marker = manager.spill_dir().join(OWNER_MARKER);
        std::fs::write(manager.spill_dir().join("unknown"), b"keep").unwrap();

        assert!(manager.finish_query().is_err());
        assert!(marker.exists());
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn finish_query_is_idempotent_after_owned_directory_removal() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let path = manager.spill_dir().to_path_buf();

        manager.finish_query().unwrap();
        manager.finish_query().unwrap();

        assert!(!path.exists());
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn finish_query_rejects_a_live_marker_with_invalid_structure() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let marker_path = manager.spill_dir().join(OWNER_MARKER);
        let mut marker = std::fs::read(&marker_path).unwrap();
        marker[0] ^= 0xff;
        std::fs::write(&marker_path, marker).unwrap();

        let error = manager.finish_query().unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(marker_path.exists());
        assert!(manager.spill_dir().exists());
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn closed_owned_query_refuses_new_files() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        manager.finish_query().unwrap();

        let error = manager.create_file(SpillFileRole::SortRun).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn partial_owner_marker_failure_removes_marker_and_query_leaf() {
        let root = TempDir::new().unwrap();
        let error = RootedSpillFixture::new(root.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(Arc::new(super::super::framing_tests::FailNthIo::new(
                SpillIoOperation::Sync,
                1,
                std::io::ErrorKind::BrokenPipe,
            )))
            .build()
            .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
        let namespace = RootedSpillFixture::new(root.path()).root().unwrap();
        assert_eq!(
            std::fs::read_dir(namespace.namespace_path())
                .unwrap()
                .count(),
            3
        );
        assert!(
            std::fs::read_dir(namespace.namespace_path())
                .unwrap()
                .all(|entry| {
                    !entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with(QUERY_PREFIX)
                })
        );
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn active_reader_blocks_delete_then_retry_and_live_file_allows_query_finish() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        let reader = file.reader().unwrap();

        let error = file.close_and_delete().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
        drop(reader);

        file.close_and_delete().unwrap();
        manager.finish_query().unwrap();
        assert!(!manager.spill_dir().exists());
        drop(file);
    }

    #[test]
    fn file_can_delete_after_manager_value_is_dropped() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let path = file.path().to_path_buf();

        drop(manager);
        file.close_and_delete().unwrap();

        assert!(!path.exists());
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn create_and_finish_are_serialized_without_holding_query_mutex_during_io() {
        let root = TempDir::new().unwrap();
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let io = Arc::new(BlockingIo {
            operation: SpillIoOperation::Create,
            entered: entered_tx,
            release: Mutex::new(release_rx),
            blocked: std::sync::atomic::AtomicBool::new(false),
        });
        let manager = Arc::new(
            RootedSpillFixture::new(root.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(io)
                .build()
                .unwrap(),
        );
        let creator_manager = Arc::clone(&manager);
        let creator =
            std::thread::spawn(move || creator_manager.create_file(SpillFileRole::SortRun));
        entered_rx.recv().unwrap();

        let error = manager.finish_query().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
        release_tx.send(()).unwrap();
        let file = creator.join().unwrap().unwrap();
        drop(file);
        manager.finish_query().unwrap();
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn concurrent_finish_is_excluded_and_first_failure_is_retryable() {
        let root = TempDir::new().unwrap();
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let io = Arc::new(BlockingIo {
            operation: SpillIoOperation::Delete,
            entered: entered_tx,
            release: Mutex::new(release_rx),
            blocked: std::sync::atomic::AtomicBool::new(false),
        });
        let manager = Arc::new(
            RootedSpillFixture::new(root.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(io)
                .build()
                .unwrap(),
        );
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        let finishing_manager = Arc::clone(&manager);
        let finisher = std::thread::spawn(move || finishing_manager.finish_query());
        entered_rx.recv().unwrap();

        let error = manager.finish_query().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
        release_tx.send(()).unwrap();
        assert_eq!(
            finisher.join().unwrap().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );

        drop(file);
        manager.finish_query().unwrap();
    }

    #[test]
    fn panicking_delete_hook_never_escapes_file_or_manager_drop() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(Arc::new(PanicDeleteIo))
            .build()
            .unwrap();
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let before = SpillManager::orphan_cleanup_failures();

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(file))).is_ok());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(manager))).is_ok());
        assert!(SpillManager::orphan_cleanup_failures() > before);
    }

    #[test]
    fn hostile_panic_payload_destructor_never_escapes_file_drop() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(Arc::new(HostilePanicDeleteIo::new()))
            .build()
            .unwrap();
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let path = file.path().to_path_buf();
        let before = SpillManager::orphan_cleanup_failures();

        let escaped = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(file))) {
            Ok(()) => false,
            Err(payload) => {
                // The RED implementation returns the hostile payload here;
                // never run its deliberately panicking destructor in the test.
                std::mem::forget(payload);
                true
            }
        };

        assert!(
            !escaped,
            "best-effort file Drop must contain hostile panics"
        );
        assert!(path.exists());
        assert_eq!(manager.active_file_count(), 1);
        assert!(SpillManager::orphan_cleanup_failures() > before);
        manager.cleanup().unwrap();
        assert!(!path.exists());
        assert_eq!(manager.active_file_count(), 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_nested_panic_destructors_cannot_abort_file_drop() {
        const CHILD_ENV: &str = "GRAFEO_SPILL_HOSTILE_DROP_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(HostileDoublePanicDeleteIo::new()))
                .build()
                .unwrap();
            let file = manager.create_file(SpillFileRole::SortRun).unwrap();
            let path = file.path().to_path_buf();
            let before = SpillManager::orphan_cleanup_failures();
            let primary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _file = file;
                std::panic::panic_any(PrimaryDropPanic);
            }))
            .unwrap_err();
            assert!(primary.is::<PrimaryDropPanic>());
            assert!(path.exists());
            assert_eq!(manager.active_file_count(), 1);
            assert!(SpillManager::orphan_cleanup_failures() > before);
            manager.cleanup().unwrap();
            assert!(!path.exists());
            println!("GRAFEO_HOSTILE_FILE_PANIC_DROP_OK");
            return;
        }

        let test_name = "execution::spill::manager::tests::hostile_nested_panic_destructors_cannot_abort_file_drop";
        assert_drop_child_survives(test_name, CHILD_ENV, "GRAFEO_HOSTILE_FILE_PANIC_DROP_OK");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_nested_error_destructors_cannot_abort_manager_drop() {
        const CHILD_ENV: &str = "GRAFEO_SPILL_HOSTILE_MANAGER_ERROR_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            let directory = TempDir::new().unwrap();
            let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(HostileDoubleErrorDeleteIo::new()))
                .build()
                .unwrap();
            let file = manager.create_file(SpillFileRole::SortRun).unwrap();
            let path = file.path().to_path_buf();
            std::mem::forget(file);
            let before = SpillManager::orphan_cleanup_failures();
            let primary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _manager = manager;
                std::panic::panic_any(PrimaryDropPanic);
            }))
            .unwrap_err();
            assert!(primary.is::<PrimaryDropPanic>());
            assert!(path.exists());
            assert!(SpillManager::orphan_cleanup_failures() > before);
            println!("GRAFEO_HOSTILE_MANAGER_ERROR_DROP_OK");
            return;
        }

        let test_name = "execution::spill::manager::tests::hostile_nested_error_destructors_cannot_abort_manager_drop";
        assert_drop_child_survives(test_name, CHILD_ENV, "GRAFEO_HOSTILE_MANAGER_ERROR_DROP_OK");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn hostile_finish_error_then_panic_cannot_abort_manager_drop() {
        const CHILD_ENV: &str = "GRAFEO_SPILL_HOSTILE_MANAGER_FINISH_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            let root = TempDir::new().unwrap();
            let manager = RootedSpillFixture::new(root.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(HostileFinishErrorThenPanicIo))
                .build()
                .unwrap();
            let query_path = manager.spill_dir().to_path_buf();
            let before = SpillManager::orphan_cleanup_failures();
            let primary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _manager = manager;
                std::panic::panic_any(PrimaryDropPanic);
            }))
            .unwrap_err();
            assert!(primary.is::<PrimaryDropPanic>());
            assert!(query_path.exists());
            assert!(!query_path.join(OWNER_MARKER).exists());
            assert!(SpillManager::orphan_cleanup_failures() > before);
            println!("GRAFEO_HOSTILE_MANAGER_FINISH_DROP_OK");
            return;
        }

        let test_name = "execution::spill::manager::tests::hostile_finish_error_then_panic_cannot_abort_manager_drop";
        assert_drop_child_survives(
            test_name,
            CHILD_ENV,
            "GRAFEO_HOSTILE_MANAGER_FINISH_DROP_OK",
        );
    }

    #[test]
    fn hostile_error_destructor_never_escapes_file_drop() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(Arc::new(HostileErrorDeleteIo::new()))
            .build()
            .unwrap();
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let path = file.path().to_path_buf();
        let before = SpillManager::orphan_cleanup_failures();

        let escaped = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(file))) {
            Ok(()) => false,
            Err(payload) => {
                std::mem::forget(payload);
                true
            }
        };

        assert!(
            !escaped,
            "best-effort file Drop must contain hostile errors"
        );
        assert!(path.exists());
        assert_eq!(manager.active_file_count(), 1);
        assert!(SpillManager::orphan_cleanup_failures() > before);
        manager.cleanup().unwrap();
        assert!(!path.exists());
        assert_eq!(manager.active_file_count(), 0);
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn failed_marker_restore_poisoned_query_never_reopens_or_adopts_leaf() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path())
            .provider(
                Arc::new(CleartextSpillRecordProvider),
                SpillFrameLimits::format_max(),
            )
            .io(Arc::new(FailQueryFinishIo))
            .build()
            .unwrap();
        let query_path = manager.spill_dir().to_path_buf();

        let error = manager.finish_query().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(query_path.exists());
        assert!(!query_path.join(OWNER_MARKER).exists());
        assert_eq!(
            manager
                .create_file(SpillFileRole::SortRun)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(
            manager.cleanup().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            manager.finish_query().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        let before = SpillManager::orphan_cleanup_failures();

        drop(manager);

        assert!(query_path.exists());
        assert!(SpillManager::orphan_cleanup_failures() > before);
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn panic_after_marker_removal_leaves_query_poisoned() {
        for operation in [
            SpillIoOperation::RemoveQueryDirectory,
            SpillIoOperation::RestoreOwnerMarker,
        ] {
            let root = TempDir::new().unwrap();
            let manager = RootedSpillFixture::new(root.path())
                .provider(
                    Arc::new(CleartextSpillRecordProvider),
                    SpillFrameLimits::format_max(),
                )
                .io(Arc::new(PanicQueryFinishIo { operation }))
                .build()
                .unwrap();
            let query_path = manager.spill_dir().to_path_buf();

            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = manager.finish_query();
                }))
                .is_err()
            );
            assert!(query_path.exists());
            assert!(!query_path.join(OWNER_MARKER).exists());
            assert!(manager.create_file(SpillFileRole::SortRun).is_err());
            assert_eq!(
                manager.cleanup().unwrap_err().kind(),
                std::io::ErrorKind::InvalidData
            );
            assert_eq!(
                manager.finish_query().unwrap_err().kind(),
                std::io::ErrorKind::InvalidData
            );
        }
    }

    #[cfg(unix)]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn writable_nonsticky_root_is_rejected_but_trusted_sticky_root_is_accepted() {
        use std::os::unix::fs::PermissionsExt as _;
        let directory = TempDir::new().unwrap();
        let shared = directory.path().join("shared");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();

        let error = RootedSpillFixture::new(&shared).build().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(std::fs::read_dir(&shared).unwrap().next().is_none());

        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o1777)).unwrap();
        let manager = RootedSpillFixture::new(&shared).build().unwrap();
        manager.finish_query().unwrap();
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn construction_cleanup_restores_marker_when_leaf_removal_fails() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let query_path = manager.spill_dir().to_path_buf();
        let namespace = query_path.parent().unwrap().to_path_buf();
        let leaf = query_path.file_name().unwrap().to_os_string();
        let marker: [u8; OWNER_MARKER_BYTES] = std::fs::read(query_path.join(OWNER_MARKER))
            .unwrap()
            .try_into()
            .unwrap();
        let root_directory = Arc::new(
            CapabilityDir::open_ambient_dir(&namespace, cap_std::ambient_authority()).unwrap(),
        );
        let query_directory = Arc::new(root_directory.open_dir_nofollow(&leaf).unwrap());
        let identity =
            PhysicalDirectoryIdentity::capture_capability(&query_directory, &query_path).unwrap();
        std::mem::forget(manager);

        let mut guard =
            QueryLeafConstructionGuard::new(query_path.clone(), Arc::clone(&root_directory), leaf);
        guard.set_identity(identity);
        guard.set_marker(marker);
        // Replace the setup manager's marker through the real exclusive-create
        // path so this construction guard owns its exact cleanup receipt.
        query_directory.remove_file(OWNER_MARKER).unwrap();
        write_marker_bytes(
            &query_path.join(OWNER_MARKER),
            &marker,
            None,
            &query_directory,
            Path::new(OWNER_MARKER),
            Some(&mut guard.marker_receipt),
        )
        .unwrap();
        drop(query_directory);
        let original_mode = std::fs::metadata(&namespace).unwrap().permissions().mode();
        std::fs::set_permissions(&namespace, std::fs::Permissions::from_mode(0o500)).unwrap();

        let remove_error = guard.cleanup_now().unwrap_err();
        let restored = std::fs::read(query_path.join(OWNER_MARKER)).unwrap();

        std::fs::set_permissions(&namespace, std::fs::Permissions::from_mode(original_mode))
            .unwrap();
        assert_eq!(remove_error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(restored, marker);
        guard.cleanup_now().unwrap();
        assert!(!query_path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn unix_root_policy_requires_trusted_owner_and_sticky_shared_writes() {
        let current = 1000;
        assert!(unix_spill_root_policy(current, current, 0o700));
        assert!(unix_spill_root_policy(0, current, 0o700));
        assert!(!unix_spill_root_policy(2000, current, 0o700));
        assert!(unix_spill_root_policy(current, current, 0o1777));
        assert!(unix_spill_root_policy(0, current, 0o1777));
        assert!(!unix_spill_root_policy(2000, current, 0o1777));
        assert!(!unix_spill_root_policy(current, current, 0o0777));
    }

    #[cfg(unix)]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn symlink_replacements_are_rejected_without_touching_external_targets() {
        use std::os::unix::fs::symlink;
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        let original = file.path().to_path_buf();
        let saved = root.path().join("saved-spill");
        let sentinel = root.path().join("sentinel");
        std::fs::write(&sentinel, b"preserve").unwrap();
        std::fs::rename(&original, &saved).unwrap();
        symlink(&sentinel, &original).unwrap();

        assert!(file.reader().is_err());
        assert!(file.close_and_delete().is_err());
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"preserve");

        std::fs::remove_file(&original).unwrap();
        std::fs::rename(&saved, &original).unwrap();
        file.close_and_delete().unwrap();
        manager.finish_query().unwrap();
    }

    #[cfg(unix)]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn query_leaf_symlink_replacement_is_rejected_and_target_is_preserved() {
        use std::os::unix::fs::symlink;
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let query_path = manager.spill_dir().to_path_buf();
        let saved = root.path().join("saved-query");
        let target = root.path().join("foreign-query");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("sentinel"), b"preserve").unwrap();
        std::fs::rename(&query_path, &saved).unwrap();
        symlink(&target, &query_path).unwrap();

        assert!(manager.create_file(SpillFileRole::SortRun).is_err());
        assert!(manager.finish_query().is_err());
        assert_eq!(std::fs::read(target.join("sentinel")).unwrap(), b"preserve");

        std::fs::remove_file(&query_path).unwrap();
        std::fs::rename(&saved, &query_path).unwrap();
        manager.finish_query().unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn regular_file_replacement_is_rejected_preserved_and_recoverable() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        let original = file.path().to_path_buf();
        let saved = root.path().join("saved-regular-spill");
        std::fs::rename(&original, &saved).unwrap();
        std::fs::write(&original, b"foreign-sentinel").unwrap();

        assert_eq!(
            file.reader().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            file.close_and_delete().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(std::fs::read(&original).unwrap(), b"foreign-sentinel");

        std::fs::remove_file(&original).unwrap();
        std::fs::rename(&saved, &original).unwrap();
        file.close_and_delete().unwrap();
        manager.finish_query().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn fifo_file_replacement_cannot_block_or_enter_reader_or_delete_paths() {
        let directory = TempDir::new().unwrap();
        let manager = crate::execution::spill::BorrowedSpillFixture::new(directory.path())
            .build()
            .unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        let original = file.path().to_path_buf();
        let saved = directory.path().join("saved-fifo-replacement");
        std::fs::rename(&original, &saved).unwrap();
        create_fifo(&original);

        assert_eq!(
            file.reader().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            file.close_and_delete().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );

        std::fs::remove_file(&original).unwrap();
        std::fs::rename(&saved, &original).unwrap();
        file.close_and_delete().unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn regular_query_directory_replacement_is_rejected_preserved_and_recoverable() {
        let root = TempDir::new().unwrap();
        let manager = RootedSpillFixture::new(root.path()).build().unwrap();
        let query_path = manager.spill_dir().to_path_buf();
        let saved = root.path().join("saved-regular-query");
        std::fs::rename(&query_path, &saved).unwrap();
        std::fs::create_dir(&query_path).unwrap();
        std::fs::write(query_path.join("sentinel"), b"foreign-sentinel").unwrap();

        assert_eq!(
            manager
                .create_file(SpillFileRole::SortRun)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            manager.finish_query().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            std::fs::read(query_path.join("sentinel")).unwrap(),
            b"foreign-sentinel"
        );

        std::fs::remove_dir_all(&query_path).unwrap();
        std::fs::rename(&saved, &query_path).unwrap();
        manager.finish_query().unwrap();
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn fifo_owner_marker_is_opened_nonblocking_and_rejected_by_handle_shape() {
        let directory = TempDir::new().unwrap();
        let marker_path = directory.path().join(OWNER_MARKER);
        create_fifo(&marker_path);
        let capability =
            CapabilityDir::open_ambient_dir(directory.path(), cap_std::ambient_authority())
                .unwrap();

        let error = read_owner_marker(&marker_path, &capability, Path::new(OWNER_MARKER))
            .expect_err("a FIFO can never be an owner marker");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[cfg(unix)]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn symlink_spill_root_is_rejected() {
        use std::os::unix::fs::symlink;
        let directory = TempDir::new().unwrap();
        let target = directory.path().join("target");
        let link = directory.path().join("link");
        std::fs::create_dir(&target).unwrap();
        symlink(&target, &link).unwrap();

        let error = RootedSpillFixture::new(&link).build().unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(std::fs::read_dir(target).unwrap().next().is_none());
    }
}
