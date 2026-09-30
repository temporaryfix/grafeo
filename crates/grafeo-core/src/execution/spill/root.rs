//! Authenticated retained root authority for the existing synchronous manager.

use super::{
    SpillDiskQuota, SpillFrameLimits, SpillIo, SpillManager, SpillQueryIdentity,
    SpillRecordProvider,
};
use crate::execution::{QueryCancellationToken, QueryExecutionId};
use grafeo_common::types::StoreId;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Engine-held identity and authentication authority. Secret keys never enter core.
pub trait SpillRootAuthority: Send + Sync {
    /// Store namespace authenticated by this authority.
    fn store_id(&self) -> StoreId;
    /// Stable, non-secret identity of the authentication key/provider.
    fn key_id(&self) -> [u8; 32];
    /// Authenticates a bounded, domain-separated canonical root or leaf marker.
    ///
    /// # Errors
    /// Returns provider authentication failure.
    fn authenticate_marker(&self, message: &[u8]) -> io::Result<[u8; 32]>;
    /// Verifies a canonical marker using the engine's authentication primitive.
    ///
    /// # Errors
    /// Returns provider authentication failure.
    fn verify_marker(&self, message: &[u8], authenticator: &[u8; 32]) -> io::Result<bool>;
    /// Creates a fresh framed-record provider for one random query-leaf identity.
    ///
    /// # Errors
    /// Returns provider/key admission failure.
    fn record_provider(
        &self,
        identity: SpillQueryIdentity,
    ) -> io::Result<Arc<dyn SpillRecordProvider>>;
}

// Private test authority: deterministic authentication is only a fixture. All
// ownership, quota and lease admission still go through the production root.
#[cfg(test)]
pub(crate) struct RootedSpillFixture {
    parent: PathBuf,
    provider: Arc<dyn SpillRecordProvider>,
    limits: SpillFrameLimits,
    io: Arc<dyn SpillIo>,
    quota: SpillDiskQuota,
}

#[cfg(test)]
impl RootedSpillFixture {
    pub(crate) fn new(parent: impl Into<PathBuf>) -> Self {
        Self {
            parent: parent.into(),
            provider: Arc::new(super::CleartextSpillRecordProvider),
            limits: SpillFrameLimits::format_max(),
            io: Arc::new(super::NoopSpillIo),
            quota: SpillDiskQuota::new(u64::MAX),
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

    pub(crate) fn root(self) -> io::Result<Arc<SpillRoot>> {
        struct FixtureAuthority(Arc<dyn SpillRecordProvider>);
        impl SpillRootAuthority for FixtureAuthority {
            fn store_id(&self) -> StoreId {
                StoreId::from_bytes([0x71; 32]).expect("nonzero fixture store")
            }
            fn key_id(&self) -> [u8; 32] {
                [0x39; 32]
            }
            fn authenticate_marker(&self, message: &[u8]) -> io::Result<[u8; 32]> {
                let mut result = self.key_id();
                for (index, byte) in message.iter().enumerate() {
                    result[index % 32] = result[index % 32].wrapping_mul(31).wrapping_add(*byte);
                }
                Ok(result)
            }
            fn verify_marker(&self, message: &[u8], auth: &[u8; 32]) -> io::Result<bool> {
                Ok(self.authenticate_marker(message)? == *auth)
            }
            fn record_provider(
                &self,
                _: SpillQueryIdentity,
            ) -> io::Result<Arc<dyn SpillRecordProvider>> {
                Ok(Arc::clone(&self.0))
            }
        }
        SpillRoot::open(
            self.parent,
            Arc::new(FixtureAuthority(self.provider)),
            self.limits,
            self.io,
            self.quota,
            None,
        )
    }

    pub(crate) fn build(self) -> io::Result<SpillManager> {
        let root = self.root()?;
        let control = crate::execution::QueryExecutionControl::new();
        let resources = crate::execution::QueryResourceContext::new_with_cancellation(
            grafeo_common::memory::buffer::BufferManager::with_budget(1 << 20),
            control.token(),
        )
        .map_err(io::Error::other)?;
        root.begin_query(resources.query_id(), control.token())
            .map(SpillManager::from_query_lease)
    }
}

/// Bounded outcomes from one authenticated root cleanup pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpillScavengeReport {
    /// Non-control root entries inspected (at most 4096).
    pub inspected: u64,
    /// Dead leaves deleted with durable quota credit.
    pub removed: u64,
    /// Live, contended or oversized leaves preserved.
    pub deferred: u64,
    /// Unrecognized root entries preserved.
    pub foreign: u64,
    /// Recognized names with invalid authority or contents, preserved.
    pub invalid: u64,
    /// Durable root charge at the end of this pass, including live queries and debt.
    pub reserved_bytes: u64,
    /// Enumeration reached its fixed work limit; another pass may be needed.
    pub truncated: bool,
}

/// Validated per-store namespace retained beneath the configured spill parent.
///
/// Physical reservations are durable and shared across processes; each query
/// also retains its logical byte limit. Unknown crash debt stays charged.
pub struct SpillRoot {
    path: PathBuf,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    authority: Arc<dyn SpillRootAuthority>,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    limits: SpillFrameLimits,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    io: Arc<dyn SpillIo>,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    quota: SpillDiskQuota,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    native: native::RootCapability,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    ledger: Arc<super::quota::RootLedger>,
    last_scavenge: parking_lot::Mutex<Option<SpillScavengeReport>>,
}

impl std::fmt::Debug for SpillRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillRoot")
            .field("namespace", &self.path)
            .finish_non_exhaustive()
    }
}

impl SpillRoot {
    /// Opens or exclusively initializes the authenticated store namespace.
    /// Existing namespaces without a valid root seal are never adopted.
    ///
    /// # Errors
    /// Rejects unknown/foreign markers, substituted identities, unsafe ownership,
    /// concurrent initialization and platforms without qualified trust/locking.
    pub fn open(
        parent: impl AsRef<Path>,
        authority: Arc<dyn SpillRootAuthority>,
        limits: SpillFrameLimits,
        io: Arc<dyn SpillIo>,
        quota: SpillDiskQuota,
        root_limit: Option<u64>,
    ) -> io::Result<Arc<Self>> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            let root_limit = root_limit.unwrap_or(u64::MAX);
            super::quota::RootLedger::validate_limit(root_limit)?;
            let (path, native) = native::RootCapability::open(parent.as_ref(), authority.as_ref())?;
            let ledger = super::quota::RootLedger::open(
                Arc::clone(&native.directory),
                &path,
                native.binding(),
                Arc::clone(&authority),
                Arc::clone(&io),
                root_limit,
            )?;
            Ok(Arc::new(Self {
                path,
                authority,
                limits,
                io,
                quota,
                native,
                ledger,
                last_scavenge: parking_lot::Mutex::new(None),
            }))
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            let _ = (parent, authority, limits, io, quota, root_limit);
            Err(unsupported())
        }
    }

    /// Returns the last successful bounded cleanup pass, not current root usage.
    /// The report is root-wide and may include other queries or unresolved debt.
    #[must_use]
    pub fn last_scavenge_report(&self) -> Option<SpillScavengeReport> {
        *self.last_scavenge.lock()
    }

    /// Reclaims only authenticated abandoned leaves under exclusive inode leases.
    /// Scans at most 4096 root entries and admits at most 256 files per leaf.
    /// Unknown content, live leases and ambiguous cleanup retain their quota debt.
    ///
    /// # Errors
    /// Returns root/ledger validation, enumeration or cleanup failure; unsupported
    /// platforms fail structurally. Invalid and live leaves are counted and preserved.
    pub fn scavenge(&self) -> io::Result<SpillScavengeReport> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            let report = self.native.scavenge(self)?;
            *self.last_scavenge.lock() = Some(report);
            Ok(report)
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            Err(unsupported())
        }
    }

    /// Revalidates configured root authority without creating a query leaf.
    pub(crate) fn validate_for_query(
        &self,
        cancellation: &QueryCancellationToken,
    ) -> io::Result<()> {
        check_cancelled(cancellation)?;
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            self.native.validate(&self.path)
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            Err(unsupported())
        }
    }

    /// Creates one sealed, exclusively locked leaf using this retained root.
    ///
    /// # Errors
    /// Returns cancellation, identity/authentication, lock, creation or cleanup failure.
    pub fn begin_query(
        self: &Arc<Self>,
        query_id: QueryExecutionId,
        cancellation: QueryCancellationToken,
    ) -> io::Result<SpillQueryLease> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            check_cancelled(&cancellation)?;
            self.native.validate(&self.path)?;
            let identity = SpillQueryIdentity::random()?;
            let provider = self.authority.record_provider(identity)?;
            check_cancelled(&cancellation)?;
            let manager =
                SpillManager::create_from_root(self, identity, provider, query_id, cancellation)?;
            Ok(SpillQueryLease { manager })
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            let _ = (query_id, cancellation);
            Err(unsupported())
        }
    }

    /// Path for diagnostics; filesystem authority remains in the retained handle.
    #[must_use]
    pub fn namespace_path(&self) -> &Path {
        &self.path
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn directory(&self) -> Arc<cap_std::fs::Dir> {
        Arc::clone(&self.native.directory)
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn frame_limits(&self) -> SpillFrameLimits {
        self.limits
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn io(&self) -> Arc<dyn SpillIo> {
        Arc::clone(&self.io)
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn disk_quota(&self) -> SpillDiskQuota {
        self.quota
    }
}

/// Move-only construction authority consumed by [`SpillManager::from_query_lease`].
#[must_use = "the query lease must remain owned until cleanup"]
#[derive(Debug)]
pub struct SpillQueryLease {
    manager: SpillManager,
}
impl SpillQueryLease {
    pub(super) fn into_manager(self) -> SpillManager {
        self.manager
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct RootQueryConstruction {
    pub(super) directory: Arc<cap_std::fs::Dir>,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    root: Arc<SpillRoot>,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    query_id: QueryExecutionId,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    cancellation: QueryCancellationToken,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    reservation: Arc<super::quota::RootReservation>,
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
impl RootQueryConstruction {
    pub(super) fn new(
        root: &Arc<SpillRoot>,
        identity: SpillQueryIdentity,
        query_id: QueryExecutionId,
        cancellation: QueryCancellationToken,
    ) -> io::Result<Self> {
        let reservation = super::quota::RootReservation::new(
            Arc::clone(&root.ledger),
            super::quota::ReservationKey::leaf(*identity.as_bytes()),
            cancellation.clone(),
        )?;
        Ok(Self {
            directory: root.directory(),
            root: Arc::clone(root),
            query_id,
            cancellation,
            reservation,
        })
    }
    #[cfg(target_os = "macos")]
    pub(super) fn reservation(&self) -> Arc<super::quota::RootReservation> {
        Arc::clone(&self.reservation)
    }
    pub(super) fn attempted(&self) {
        self.reservation.attempted();
    }
    pub(super) fn bind(&self, directory: &cap_std::fs::Dir) -> io::Result<()> {
        self.reservation.bind(
            directory.try_clone()?.into_std_file(),
            self.directory.clone(),
        )
    }
    pub(super) fn prepare(
        &self,
        identity: SpillQueryIdentity,
        directory: &Arc<cap_std::fs::Dir>,
    ) -> io::Result<PreparedQueryLease> {
        check_cancelled(&self.cancellation)?;
        Ok(PreparedQueryLease {
            root: Arc::clone(&self.root),
            cancellation: self.cancellation.clone(),
            identity,
            reservation: Arc::clone(&self.reservation),
            native: native::PreparedLeafCapability::prepare(
                &self.root,
                Arc::clone(directory),
                identity,
                self.query_id,
            )?,
        })
    }
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
pub(super) struct PreparedQueryLease {
    root: Arc<SpillRoot>,
    cancellation: QueryCancellationToken,
    native: native::PreparedLeafCapability,
    identity: SpillQueryIdentity,
    reservation: Arc<super::quota::RootReservation>,
}
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
impl PreparedQueryLease {
    pub(super) fn directory_lock(&self) -> Arc<std::fs::File> {
        self.native.directory_lock()
    }
    pub(super) fn marker(&self) -> &[u8; super::manager::OWNER_MARKER_BYTES] {
        &self.native.bytes
    }
    pub(super) fn finish(&self, marker: &Arc<std::fs::File>) -> io::Result<QueryLeaseAuthority> {
        // The manager performs the final root/leaf validation with its
        // construction receipt still armed, after all publication hooks.
        Ok(QueryLeaseAuthority {
            root: Arc::clone(&self.root),
            cancellation: self.cancellation.clone(),
            native: parking_lot::Mutex::new(Some(self.native.finish(marker)?)),
            identity: self.identity,
            reservation: Arc::clone(&self.reservation),
        })
    }
}

pub(super) struct QueryLeaseAuthority {
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    root: Arc<SpillRoot>,
    cancellation: QueryCancellationToken,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    native: parking_lot::Mutex<Option<native::LeafCapability>>,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    identity: SpillQueryIdentity,
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    reservation: Arc<super::quota::RootReservation>,
}
impl std::fmt::Debug for QueryLeaseAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = formatter.debug_struct("QueryLeaseAuthority");
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        output.field("namespace", &self.root.path);
        output.finish_non_exhaustive()
    }
}

impl QueryLeaseAuthority {
    pub(super) fn physical_stats(&self) -> Option<super::SpillPhysicalStats> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            self.reservation.physical_stats()
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            None
        }
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn deletion_pending(&self) -> bool {
        self.native.lock().is_none()
    }

    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn release_deleted_leaf(&self) -> io::Result<()> {
        // Only query completion calls this, after proving all files closed and
        // removing the exact owned leaf. Close marker/directory handles before
        // the reservation's last allocation handle and durable credit.
        drop(self.native.lock().take());
        #[cfg(target_os = "macos")]
        self.reservation.confirm_directory_unlink()?;
        self.reservation.release_deleted()
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos"),
        not(target_arch = "wasm32")
    ))]
    pub(super) fn reserve_file(
        &self,
        identity: super::SpillFileIdentity,
    ) -> io::Result<Arc<super::quota::RootReservation>> {
        self.validate(true)?;
        super::quota::RootReservation::new_file(
            &self.reservation,
            super::quota::ReservationKey::file(*self.identity.as_bytes(), *identity.as_bytes()),
        )
    }
    pub(super) fn rebind_restored_marker(
        &self,
        created_marker: &Arc<std::fs::File>,
    ) -> io::Result<()> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            self.native
                .lock()
                .as_ref()
                .ok_or_else(|| io::Error::other("spill leaf is closed"))?
                .rebind_restored_marker(created_marker)
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            let _ = created_marker;
            Err(unsupported())
        }
    }
    pub(super) fn validated_marker(&self) -> io::Result<[u8; super::manager::OWNER_MARKER_BYTES]> {
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            self.root.native.validate(&self.root.path)?;
            self.native
                .lock()
                .as_ref()
                .ok_or_else(|| io::Error::other("spill leaf is closed"))?
                .validated_marker()
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            Err(unsupported())
        }
    }

    pub(super) fn validate(&self, cancellation: bool) -> io::Result<()> {
        if cancellation {
            check_cancelled(&self.cancellation)?;
        }
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        ))]
        {
            self.root.native.validate(&self.root.path)?;
            self.native
                .lock()
                .as_ref()
                .ok_or_else(|| io::Error::other("spill leaf is closed"))?
                .validate()
        }
        #[cfg(not(all(
            any(target_os = "linux", target_os = "macos"),
            not(target_arch = "wasm32")
        )))]
        {
            Err(unsupported())
        }
    }
}

fn check_cancelled(token: &QueryCancellationToken) -> io::Result<()> {
    token
        .check()
        .map_err(|error| io::Error::new(io::ErrorKind::Interrupted, error))
}
#[cfg(not(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
)))]
fn unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "authenticated spill roots require qualified directory ownership and exclusive marker locks on this platform",
    )
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
mod native {
    mod scavenge;
    use super::{QueryExecutionId, SpillQueryIdentity, SpillRoot, SpillRootAuthority, StoreId};
    use crate::execution::spill::manager::{
        OWNER_MARKER, OWNER_MARKER_BYTES, PhysicalDirectoryIdentity,
        create_owner_only_directory_at, ensure_production_directory,
        validate_production_spill_root,
    };
    use cap_fs_ext::{
        DirExt as _, FollowSymlinks, OpenOptionsFollowExt as _, OpenOptionsSyncExt as _,
    };
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt as _};
    use std::fs::File;
    use std::io;
    use std::io::Write;
    use std::os::unix::fs::FileExt as _;
    use std::os::unix::fs::MetadataExt as _;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const ROOT_MARKER: &str = ".grafeo-spill-root";
    const ROOT_PREFIX: usize = 109;
    const ROOT_BYTES: usize = ROOT_PREFIX + 32;

    pub(super) struct RootCapability {
        pub(super) directory: Arc<Dir>,
        identity: PhysicalDirectoryIdentity,
        marker: RetainedMarker,
        bytes: [u8; ROOT_BYTES],
    }
    pub(super) struct LeafCapability {
        directory: Arc<Dir>,
        marker: parking_lot::Mutex<RetainedMarker>,
        // Stable inode lease survives owner-marker unlink/restoration on cleanup failure.
        _directory_lock: Arc<File>,
        bytes: [u8; OWNER_MARKER_BYTES],
    }

    fn invalid(message: &'static str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, message)
    }
    fn namespace(store: StoreId) -> String {
        let mut name = String::from("grafeo-store-");
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in store.as_bytes() {
            name.push(char::from(HEX[usize::from(byte >> 4)]));
            name.push(char::from(HEX[usize::from(byte & 15)]));
        }
        name
    }
    // cap_std may retain an O_PATH descriptor. Reopen the same directory inode
    // relative to that capability for operations requiring a readable descriptor.
    fn directory_io_handle(directory: &Dir) -> io::Result<File> {
        rustix::fs::openat(
            directory,
            ".",
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .map(File::from)
        .map_err(io::Error::from)
    }

    fn open_marker(dir: &Dir, name: &str, create: bool) -> io::Result<File> {
        open_marker_with_metadata(dir, name, create).map(|(file, _)| file)
    }
    fn open_marker_with_metadata(
        dir: &Dir,
        name: &str,
        create: bool,
    ) -> io::Result<(File, std::fs::Metadata)> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create_new(create)
            .follow(FollowSymlinks::No)
            .nonblock(true)
            .mode(0o600);
        let file = dir.open_with(name, &options)?.into_std();
        let metadata = file.metadata()?;
        if !metadata.is_file()
            || metadata.nlink() != 1
            || metadata.uid() != rustix::process::geteuid().as_raw()
            || metadata.mode() & 0o077 != 0
        {
            return Err(invalid(
                "spill marker is not an exclusively owned regular file",
            ));
        }
        #[cfg(target_os = "macos")]
        super::super::validate_no_acl_grants(&file)?;
        Ok((file, metadata))
    }
    fn read_exact_marker<const N: usize>(file: &File) -> io::Result<[u8; N]> {
        if file.metadata()?.len() != N as u64 {
            return Err(invalid("invalid spill marker length"));
        }
        let mut bytes = [0; N];
        file.read_exact_at(&mut bytes, 0)?;
        Ok(bytes)
    }

    // The held descriptor pins this inode. Only its identity is immutable:
    // permissions, link count, size and contents are checked on every use.
    struct RetainedMarker {
        file: Arc<File>,
        device: u64,
        inode: u64,
    }
    impl RetainedMarker {
        fn new(file: Arc<File>) -> io::Result<Self> {
            let metadata = file.metadata()?;
            Ok(Self {
                file,
                device: metadata.dev(),
                inode: metadata.ino(),
            })
        }

        fn validate<const N: usize>(
            &self,
            directory: &Dir,
            name: &str,
            expected: &[u8; N],
        ) -> io::Result<()> {
            use cap_std::fs::MetadataExt as _;
            let metadata = directory.symlink_metadata(name)?;
            if metadata.dev() != self.device || metadata.ino() != self.inode {
                return Err(invalid("spill marker identity was replaced"));
            }
            if !metadata.is_file()
                || metadata.nlink() != 1
                || metadata.uid() != rustix::process::geteuid().as_raw()
                || metadata.mode() & 0o077 != 0
                || metadata.len() != N as u64
            {
                return Err(invalid(
                    "spill marker identity, ownership or length changed",
                ));
            }
            #[cfg(target_os = "macos")]
            {
                super::super::validate_no_acl_grants(directory)?;
                super::super::validate_no_acl_grants(self.file.as_ref())?;
            }
            let mut bytes = [0; N];
            // Positional reads keep shared root validation independent of any
            // other query and of the writer's retained descriptor cursor.
            self.file.read_exact_at(&mut bytes, 0)?;
            if &bytes != expected {
                return Err(invalid("authenticated spill seal changed"));
            }
            Ok(())
        }
    }
    impl RootCapability {
        pub(super) fn binding(&self) -> [u8; 32] {
            *blake3::hash(&self.bytes).as_bytes()
        }

        pub(super) fn open(
            parent: &Path,
            authority: &dyn SpillRootAuthority,
        ) -> io::Result<(PathBuf, Self)> {
            ensure_production_directory(parent)?;
            validate_production_spill_root(parent)?;
            let parent_dir = Dir::open_ambient_dir(parent, cap_std::ambient_authority())?;
            PhysicalDirectoryIdentity::capture_capability(&parent_dir, parent)?;
            let name = namespace(authority.store_id());
            let created = match create_owner_only_directory_at(&parent_dir, &name) {
                Ok(()) => true,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => false,
                Err(error) => return Err(error),
            };
            let path = parent.join(&name);
            let directory = Arc::new(parent_dir.open_dir_nofollow(&name)?);
            let metadata = directory.try_clone()?.into_std_file().metadata()?;
            if metadata.uid() != rustix::process::geteuid().as_raw() || metadata.mode() & 0o077 != 0
            {
                return Err(invalid("spill namespace requires private caller ownership"));
            }
            let identity = PhysicalDirectoryIdentity::capture_capability(&directory, &path)?;
            if created && directory.entries()?.next().is_some() {
                return Err(invalid("new spill namespace contains unknown content"));
            }
            let mut marker = open_marker(&directory, ROOT_MARKER, created)?;
            // Other processes may be authenticating this same root. Root
            // opening has no query token yet, so use a finite admission wait.
            let mut acquired = false;
            for _ in 0..512 {
                match marker.try_lock().map_err(io::Error::from) {
                    Ok(()) => {
                        acquired = true;
                        break;
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(error) => return Err(error),
                }
            }
            if !acquired {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "spill root initialization is busy",
                ));
            }
            let bytes = if created {
                let mut bytes = [0; ROOT_BYTES];
                bytes[..4].copy_from_slice(b"GRAR");
                bytes[4] = 1;
                bytes[5..37].copy_from_slice(authority.store_id().as_bytes());
                bytes[37..69].copy_from_slice(&authority.key_id());
                bytes[69..85].copy_from_slice(SpillQueryIdentity::random()?.as_bytes());
                bytes[85..93].copy_from_slice(&metadata.dev().to_le_bytes());
                bytes[93..101].copy_from_slice(&metadata.ino().to_le_bytes());
                bytes[101..109].copy_from_slice(&1u64.to_le_bytes());
                let auth = authority.authenticate_marker(&bytes[..ROOT_PREFIX])?;
                bytes[ROOT_PREFIX..].copy_from_slice(&auth);
                marker.write_all(&bytes)?;
                marker.sync_all()?;
                directory_io_handle(&directory)?.sync_all()?;
                directory_io_handle(&parent_dir)?.sync_all()?;
                bytes
            } else {
                let bytes = read_exact_marker::<ROOT_BYTES>(&marker)?;
                let authenticator = bytes[ROOT_PREFIX..]
                    .try_into()
                    .map_err(|_| invalid("root authenticator shape"))?;
                if bytes[..4] != *b"GRAR"
                    || bytes[4] != 1
                    || bytes[5..37] != *authority.store_id().as_bytes()
                    || bytes[37..69] != authority.key_id()
                    || bytes[85..93] != metadata.dev().to_le_bytes()
                    || bytes[93..101] != metadata.ino().to_le_bytes()
                    || bytes[101..109] != 1u64.to_le_bytes()
                    || !authority.verify_marker(&bytes[..ROOT_PREFIX], &authenticator)?
                {
                    return Err(invalid(
                        "foreign, unsupported, or substituted spill root seal",
                    ));
                }
                bytes
            };
            marker.unlock()?;
            let result = Self {
                directory,
                identity,
                marker: RetainedMarker::new(Arc::new(marker))?,
                bytes,
            };
            result.validate(&path)?;
            Ok((path, result))
        }
        pub(super) fn validate(&self, path: &Path) -> io::Result<()> {
            self.identity.validate_path(path)?;
            // The private retained directory descriptor pins the identity
            // captured at open; only the ambient entry can be substituted.
            self.marker
                .validate(&self.directory, ROOT_MARKER, &self.bytes)
        }
    }
    // v3 seals commit to the retained directory inode as well as the root and
    // logical query. Copying a valid seal into a replacement leaf cannot inherit
    // its live/crashed predecessor's durable reservation.
    fn leaf_auth_message(
        bytes: &[u8; OWNER_MARKER_BYTES],
        directory: &File,
    ) -> io::Result<[u8; 149]> {
        let metadata = directory.metadata()?;
        let mut message = [0; 149];
        message[..21].copy_from_slice(&bytes[..21]);
        message[21..133].copy_from_slice(&bytes[53..]);
        message[133..141].copy_from_slice(&metadata.dev().to_le_bytes());
        message[141..].copy_from_slice(&metadata.ino().to_le_bytes());
        Ok(message)
    }

    pub(super) struct PreparedLeafCapability {
        directory: Arc<Dir>,
        directory_lock: Arc<File>,
        pub(super) bytes: [u8; OWNER_MARKER_BYTES],
    }
    impl PreparedLeafCapability {
        pub(super) fn directory_lock(&self) -> Arc<File> {
            Arc::clone(&self.directory_lock)
        }
        pub(super) fn prepare(
            root: &Arc<SpillRoot>,
            directory: Arc<Dir>,
            identity: SpillQueryIdentity,
            query_id: QueryExecutionId,
        ) -> io::Result<Self> {
            #[cfg(target_os = "macos")]
            super::super::validate_no_acl_grants(directory.as_ref())?;
            let directory_lock = Arc::new(directory_io_handle(&directory)?);
            directory_lock.try_lock().map_err(io::Error::from)?;
            let mut bytes = [0; OWNER_MARKER_BYTES];
            bytes[..4].copy_from_slice(b"GRAQ");
            bytes[4] = 3;
            bytes[5..21].copy_from_slice(identity.as_bytes());
            bytes[53..157].copy_from_slice(&root.native.bytes[5..109]);
            bytes[157..165].copy_from_slice(&query_id.get().to_le_bytes());
            let message = leaf_auth_message(&bytes, &directory_lock)?;
            let auth = root.authority.authenticate_marker(&message)?;
            bytes[21..53].copy_from_slice(&auth);
            Ok(Self {
                directory,
                directory_lock,
                bytes,
            })
        }
        pub(super) fn finish(&self, created_marker: &Arc<File>) -> io::Result<LeafCapability> {
            // Carry the exact-created receipt into the lease, never reopen a
            // pathname and accidentally adopt a same-byte replacement inode.
            let marker = Arc::clone(created_marker);
            marker.try_lock().map_err(io::Error::from)?;
            self.directory_lock.sync_all()?;
            Ok(LeafCapability {
                directory: Arc::clone(&self.directory),
                marker: parking_lot::Mutex::new(RetainedMarker::new(marker)?),
                _directory_lock: Arc::clone(&self.directory_lock),
                bytes: self.bytes,
            })
        }
    }
    impl LeafCapability {
        pub(super) fn rebind_restored_marker(&self, created_marker: &Arc<File>) -> io::Result<()> {
            let restored = RetainedMarker::new(Arc::clone(created_marker))?;
            restored.validate(&self.directory, OWNER_MARKER, &self.bytes)?;
            restored.file.try_lock().map_err(io::Error::from)?;
            *self.marker.lock() = restored;
            self.validate()
        }

        pub(super) fn validated_marker(&self) -> io::Result<[u8; OWNER_MARKER_BYTES]> {
            self.validate()?;
            Ok(self.bytes)
        }

        pub(super) fn validate(&self) -> io::Result<()> {
            self.marker
                .lock()
                .validate(&self.directory, OWNER_MARKER, &self.bytes)
        }
    }
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos"),
    not(target_arch = "wasm32")
))]
mod tests {
    use super::*;
    use crate::execution::spill::{CleartextSpillRecordProvider, NoopSpillIo, SpillFileRole};
    use crate::execution::{QueryExecutionControl, QueryResourceContext};
    use grafeo_common::memory::buffer::BufferManager;
    use std::fs;
    use tempfile::TempDir;

    struct TestAuthority {
        store: StoreId,
        key: u8,
    }
    impl SpillRootAuthority for TestAuthority {
        fn store_id(&self) -> StoreId {
            self.store
        }
        fn key_id(&self) -> [u8; 32] {
            [self.key; 32]
        }
        // Deterministic test-only authenticator, never a production provider.
        fn authenticate_marker(&self, message: &[u8]) -> io::Result<[u8; 32]> {
            let mut result = [self.key; 32];
            for (index, byte) in message.iter().enumerate() {
                result[index % 32] = result[index % 32].wrapping_mul(31).wrapping_add(*byte);
            }
            Ok(result)
        }
        fn verify_marker(&self, message: &[u8], auth: &[u8; 32]) -> io::Result<bool> {
            Ok(self.authenticate_marker(message)? == *auth)
        }
        fn record_provider(
            &self,
            _: SpillQueryIdentity,
        ) -> io::Result<Arc<dyn SpillRecordProvider>> {
            Ok(Arc::new(CleartextSpillRecordProvider))
        }
    }
    fn authority(store: u8, key: u8) -> Arc<dyn SpillRootAuthority> {
        Arc::new(TestAuthority {
            store: StoreId::from_bytes([store; 32]).unwrap(),
            key,
        })
    }
    fn open(parent: &Path, store: u8, key: u8) -> io::Result<Arc<SpillRoot>> {
        SpillRoot::open(
            parent,
            authority(store, key),
            SpillFrameLimits::format_max(),
            Arc::new(NoopSpillIo),
            SpillDiskQuota::new(1 << 20),
            None,
        )
    }
    fn query() -> (QueryResourceContext, QueryExecutionControl) {
        let control = QueryExecutionControl::new();
        let resources = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1 << 20),
            control.token(),
        )
        .unwrap();
        (resources, control)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn acl_mutation_rejects_root_and_leaf_before_new_spill_writes() {
        use std::process::Command;
        for target in ["root", "root_marker", "leaf", "leaf_marker"] {
            let parent = TempDir::new().unwrap();
            let root = open(parent.path(), 1, 7).unwrap();
            let (context, control) = query();
            let manager = SpillManager::from_query_lease(
                root.begin_query(context.query_id(), control.token())
                    .unwrap(),
            );
            let path = match target {
                "root" => root.namespace_path().to_owned(),
                "root_marker" => root.namespace_path().join(".grafeo-spill-root"),
                "leaf" => manager.spill_dir().to_owned(),
                _ => manager.spill_dir().join(".grafeo-spill-owner"),
            };
            assert!(
                Command::new("/bin/chmod")
                    .args(["+a", "everyone allow read"])
                    .arg(&path)
                    .status()
                    .unwrap()
                    .success()
            );
            assert!(
                manager.create_file(SpillFileRole::SortRun).is_err(),
                "{target}"
            );
            assert_eq!(fs::read_dir(manager.spill_dir()).unwrap().count(), 1);
            assert!(
                Command::new("/bin/chmod")
                    .arg("-N")
                    .arg(&path)
                    .status()
                    .unwrap()
                    .success()
            );
            let file = manager.create_file(SpillFileRole::SortRun).unwrap();
            drop(file);
        }
    }

    #[test]
    fn data_cleanup_hook_cannot_replace_authenticated_leaf_seal() {
        use super::super::{SpillIo, SpillIoOperation};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct TamperOnCleanup {
            deletes: AtomicUsize,
            marker: parking_lot::Mutex<Option<PathBuf>>,
        }
        impl SpillIo for TamperOnCleanup {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::Delete {
                    if self.deletes.fetch_add(1, Ordering::Relaxed) == 0 {
                        return Err(io::Error::other("retain closed file for manager cleanup"));
                    }
                    if let Some(marker) = self.marker.lock().take() {
                        let mut bytes = fs::read(&marker)?;
                        bytes[40] ^= 1; // Preserve framing/identity, corrupt authentication.
                        fs::write(marker, bytes)?;
                    }
                }
                Ok(())
            }
        }
        let parent = TempDir::new().unwrap();
        let io = Arc::new(TamperOnCleanup {
            deletes: AtomicUsize::new(0),
            marker: parking_lot::Mutex::new(None),
        });
        let root = SpillRoot::open(
            parent.path(),
            authority(1, 7),
            SpillFrameLimits::format_max(),
            io.clone(),
            SpillDiskQuota::new(1 << 20),
            None,
        )
        .unwrap();
        let (resources, control) = query();
        let manager = SpillManager::from_query_lease(
            root.begin_query(resources.query_id(), control.token())
                .unwrap(),
        );
        let marker = manager.spill_dir().join(".grafeo-spill-owner");
        let original = fs::read(&marker).unwrap();
        *io.marker.lock() = Some(marker.clone());
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let file_path = file.path().to_owned();
        drop(file); // The first delete hook leaves a closed, tracked file.
        assert_eq!(manager.active_file_count(), 1);
        let error = manager.finish_query().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("authenticated spill seal changed")
        );
        assert!(
            !file_path.exists(),
            "the data-cleanup hook must have executed"
        );
        assert_ne!(fs::read(&marker).unwrap(), original);
        assert!(
            manager.spill_dir().is_dir(),
            "tampered leaf must remain unadopted"
        );
        fs::write(&marker, original).unwrap();
        manager.finish_query().unwrap();
    }

    #[test]
    fn query_cleanup_after_rename_preserves_empty_foreign_named_leaf() {
        use super::super::{SpillIo, SpillIoOperation};
        use std::os::unix::fs::MetadataExt as _;
        use std::sync::atomic::{AtomicU64, Ordering};

        struct RenameOnRemove {
            paths: parking_lot::Mutex<Option<(PathBuf, PathBuf)>>,
            foreign_inode: AtomicU64,
        }
        impl SpillIo for RenameOnRemove {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::RemoveQueryDirectory
                    && let Some((original, moved)) = self.paths.lock().take()
                {
                    fs::rename(&original, moved)?;
                    fs::create_dir(&original)?;
                    self.foreign_inode
                        .store(fs::metadata(original)?.ino(), Ordering::Relaxed);
                }
                Ok(())
            }
        }
        for outside in [false, true] {
            let parent = TempDir::new().unwrap();
            let io = Arc::new(RenameOnRemove {
                paths: parking_lot::Mutex::new(None),
                foreign_inode: AtomicU64::new(0),
            });
            let root = SpillRoot::open(
                parent.path(),
                authority(1, 7),
                SpillFrameLimits::format_max(),
                io.clone(),
                SpillDiskQuota::new(1 << 20),
                None,
            )
            .unwrap();
            let (resources, control) = query();
            let manager = SpillManager::from_query_lease(
                root.begin_query(resources.query_id(), control.token())
                    .unwrap(),
            );
            let original = manager.spill_dir().to_owned();
            let moved = if outside {
                parent.path().join("moved-outside-namespace")
            } else {
                root.namespace_path().join("renamed-owned-query")
            };
            *io.paths.lock() = Some((original.clone(), moved.clone()));
            let debt = quota_used(&root);
            let result = manager.finish_query();
            if outside {
                assert!(
                    result.is_err(),
                    "a different parent was not admitted for durable deletion"
                );
                assert!(
                    moved.is_dir(),
                    "moved allocation must remain preserved and charged"
                );
                assert_eq!(quota_used(&root), debt);
            } else {
                result.unwrap();
                assert!(
                    !moved.exists(),
                    "cleanup must find the original retained inode"
                );
            }
            assert_eq!(
                fs::metadata(&original).unwrap().ino(),
                io.foreign_inode.load(Ordering::Relaxed)
            );
            assert!(fs::read_dir(original).unwrap().next().is_none());
        }
    }

    #[test]
    fn concurrent_retained_root_validation_does_not_share_read_offsets() {
        let parent = TempDir::new().unwrap();
        let root = open(parent.path(), 1, 7).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let root = Arc::clone(&root);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..64 {
                        root.native.validate(root.namespace_path()).unwrap();
                    }
                });
            }
        });
    }

    #[test]
    fn root_rejects_foreign_key_unknown_version_and_unknown_namespace_unchanged() {
        let parent = TempDir::new().unwrap();
        let root = open(parent.path(), 1, 7).unwrap();
        let marker = root.namespace_path().join(".grafeo-spill-root");
        let original = fs::read(&marker).unwrap();
        assert!(open(parent.path(), 1, 8).is_err());
        assert_eq!(fs::read(&marker).unwrap(), original);
        let mut changed = original.clone();
        changed[4] = 255;
        fs::write(&marker, &changed).unwrap();
        assert!(open(parent.path(), 1, 7).is_err());
        assert_eq!(fs::read(&marker).unwrap(), changed);
        fs::write(&marker, &original).unwrap();
        let foreign = open(parent.path(), 2, 7).unwrap();
        fs::write(
            foreign.namespace_path().join(".grafeo-spill-root"),
            &original,
        )
        .unwrap();
        assert!(open(parent.path(), 2, 7).is_err());
        assert_eq!(fs::read_dir(root.namespace_path()).unwrap().count(), 3);
        fs::remove_file(&marker).unwrap();
        fs::write(root.namespace_path().join("sentinel"), b"preserve").unwrap();
        assert!(open(parent.path(), 1, 7).is_err());
        assert_eq!(
            fs::read(root.namespace_path().join("sentinel")).unwrap(),
            b"preserve"
        );
    }

    #[test]
    fn leases_have_unique_sealed_leaves_locks_and_independent_cancellation() {
        let parent = TempDir::new().unwrap();
        let root = open(parent.path(), 1, 7).unwrap();
        let (first_context, first_control) = query();
        let (second_context, second_control) = query();
        let first = SpillManager::from_query_lease(
            root.begin_query(first_context.query_id(), first_control.token())
                .unwrap(),
        );
        let second = SpillManager::from_query_lease(
            root.begin_query(second_context.query_id(), second_control.token())
                .unwrap(),
        );
        assert_ne!(first.spill_dir(), second.spill_dir());
        let marker = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(first.spill_dir().join(".grafeo-spill-owner"))
            .unwrap();
        assert!(matches!(
            marker.try_lock(),
            Err(fs::TryLockError::WouldBlock)
        ));
        let seal = fs::read(first.spill_dir().join(".grafeo-spill-owner")).unwrap();
        assert_eq!(seal[4], 3);
        assert_eq!(&seal[53..85], &[1; 32]);
        assert_eq!(
            &seal[157..165],
            &first_context.query_id().get().to_le_bytes()
        );
        first_control.cancellation_handle().cancel();
        assert!(first.create_file(SpillFileRole::SortRun).is_err());
        let file = second.create_file(SpillFileRole::SortRun).unwrap();
        drop(file);
        first.finish_query().unwrap();
        assert!(second.spill_dir().exists());
        second.finish_query().unwrap();
        assert!(!first.spill_dir().exists());
        assert!(!second.spill_dir().exists());
    }

    #[test]
    fn retained_root_rejects_path_substitution_before_creating_query_leaf() {
        let parent = TempDir::new().unwrap();
        let root = open(parent.path(), 1, 7).unwrap();
        let path = root.namespace_path().to_owned();
        fs::rename(&path, parent.path().join("retained-root")).unwrap();
        fs::create_dir(&path).unwrap();
        fs::write(path.join("sentinel"), b"foreign").unwrap();
        let (context, control) = query();
        assert!(
            root.begin_query(context.query_id(), control.token())
                .is_err()
        );
        assert_eq!(fs::read_dir(&path).unwrap().count(), 1);
        assert_eq!(fs::read(path.join("sentinel")).unwrap(), b"foreign");
    }
    #[test]
    fn failed_leaf_removal_restores_seal_under_continuous_directory_lease() {
        use crate::execution::spill::SpillIoOperation;
        use std::sync::atomic::{AtomicBool, Ordering};
        struct FailRemoveOnce(AtomicBool);
        impl SpillIo for FailRemoveOnce {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::RemoveQueryDirectory
                    && self.0.swap(false, Ordering::SeqCst)
                {
                    return Err(io::Error::other("injected directory removal failure"));
                }
                Ok(())
            }
        }
        let parent = TempDir::new().unwrap();
        let root = SpillRoot::open(
            parent.path(),
            authority(1, 7),
            SpillFrameLimits::format_max(),
            Arc::new(FailRemoveOnce(AtomicBool::new(true))),
            SpillDiskQuota::new(1 << 20),
            None,
        )
        .unwrap();
        let (context, control) = query();
        let manager = SpillManager::from_query_lease(
            root.begin_query(context.query_id(), control.token())
                .unwrap(),
        );
        let path = manager.spill_dir().to_owned();
        assert!(manager.finish_query().is_err());
        let marker = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.join(".grafeo-spill-owner"))
            .unwrap();
        assert!(matches!(
            marker.try_lock(),
            Err(fs::TryLockError::WouldBlock)
        ));
        let directory = fs::File::open(&path).unwrap();
        assert!(matches!(
            directory.try_lock(),
            Err(fs::TryLockError::WouldBlock)
        ));
        manager.finish_query().unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn restored_marker_rebind_rejects_identical_bytes_on_a_foreign_inode() {
        use crate::execution::spill::SpillIoOperation;
        use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
        struct FailRemove;
        impl SpillIo for FailRemove {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::RemoveQueryDirectory {
                    return Err(io::Error::other("injected directory removal failure"));
                }
                Ok(())
            }
        }
        let parent = TempDir::new().unwrap();
        let root = SpillRoot::open(
            parent.path(),
            authority(1, 7),
            SpillFrameLimits::format_max(),
            Arc::new(FailRemove),
            SpillDiskQuota::new(1 << 20),
            None,
        )
        .unwrap();
        let (context, control) = query();
        let manager = SpillManager::from_query_lease(
            root.begin_query(context.query_id(), control.token())
                .unwrap(),
        );
        let marker = manager.spill_dir().join(".grafeo-spill-owner");
        let original = fs::read(&marker).unwrap();
        let saved = parent.path().join("exact-restored-marker");
        let hook_marker = marker.clone();
        let hook_saved = saved.clone();
        super::super::manager::AFTER_OWNER_MARKER_RESTORE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                let bytes = fs::read(&hook_marker).unwrap();
                fs::rename(&hook_marker, &hook_saved).unwrap();
                let mut foreign = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&hook_marker)
                    .unwrap();
                std::io::Write::write_all(&mut foreign, &bytes).unwrap();
            }));
        });
        let error = manager.finish_query().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("spill marker identity was replaced")
        );
        let foreign = fs::metadata(&marker).unwrap();
        assert_ne!(foreign.ino(), fs::metadata(&saved).unwrap().ino());
        assert_eq!(fs::read(&marker).unwrap(), original);
        assert_eq!(fs::read(&saved).unwrap(), original);
        let directory = fs::File::open(manager.spill_dir()).unwrap();
        assert!(matches!(
            directory.try_lock(),
            Err(fs::TryLockError::WouldBlock)
        ));
        assert!(manager.finish_query().is_err());
        drop(manager);
        let retained = fs::metadata(&marker).unwrap();
        assert_eq!(
            (retained.dev(), retained.ino()),
            (foreign.dev(), foreign.ino())
        );
        assert_eq!(fs::read(&marker).unwrap(), original);
    }

    #[test]
    fn root_bound_leaf_rejects_truncated_or_unknown_canonical_marker_without_cleanup() {
        let parent = TempDir::new().unwrap();
        let root = open(parent.path(), 1, 7).unwrap();
        let (context, control) = query();
        let manager = SpillManager::from_query_lease(
            root.begin_query(context.query_id(), control.token())
                .unwrap(),
        );
        let marker = manager.spill_dir().join(".grafeo-spill-owner");
        let original = fs::read(&marker).unwrap();
        fs::write(&marker, &original[..53]).unwrap();
        assert!(manager.finish_query().is_err());
        assert_eq!(fs::read(&marker).unwrap(), original[..53]);
        let mut unknown = original.clone();
        unknown[4] = 255;
        fs::write(&marker, &unknown).unwrap();
        assert!(manager.create_file(SpillFileRole::SortRun).is_err());
        assert_eq!(fs::read(&marker).unwrap(), unknown);
        fs::write(&marker, original).unwrap();
        manager.finish_query().unwrap();
    }
    #[test]
    fn canonical_leaf_is_written_once_with_directory_lease_already_held() {
        use crate::execution::spill::SpillIoOperation;
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct ObservePublication {
            namespace: parking_lot::Mutex<Option<PathBuf>>,
            writes: AtomicUsize,
            syncs: AtomicUsize,
        }
        impl SpillIo for ObservePublication {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::WritePayload {
                    self.writes.fetch_add(1, Ordering::SeqCst);
                    let namespace = self.namespace.lock();
                    let path = namespace.as_ref().unwrap();
                    let leaf = fs::read_dir(path)?
                        .map(|entry| entry.unwrap().path())
                        .find(|path| path.is_dir())
                        .unwrap();
                    let directory = fs::File::open(leaf)?;
                    assert!(matches!(
                        directory.try_lock(),
                        Err(fs::TryLockError::WouldBlock)
                    ));
                }
                if operation == SpillIoOperation::Sync {
                    self.syncs.fetch_add(1, Ordering::SeqCst);
                }
                Ok(())
            }
        }
        let parent = TempDir::new().unwrap();
        let observer = Arc::new(ObservePublication {
            namespace: parking_lot::Mutex::new(None),
            writes: AtomicUsize::new(0),
            syncs: AtomicUsize::new(0),
        });
        let root = SpillRoot::open(
            parent.path(),
            authority(1, 7),
            SpillFrameLimits::format_max(),
            observer.clone(),
            SpillDiskQuota::new(1 << 20),
            None,
        )
        .unwrap();
        *observer.namespace.lock() = Some(root.namespace_path().to_owned());
        let (context, control) = query();
        let manager = SpillManager::from_query_lease(
            root.begin_query(context.query_id(), control.token())
                .unwrap(),
        );
        assert_eq!(observer.writes.load(Ordering::SeqCst), 1);
        assert_eq!(observer.syncs.load(Ordering::SeqCst), 1);
        let marker = fs::read(manager.spill_dir().join(".grafeo-spill-owner")).unwrap();
        assert_eq!(marker[4], 3);
        manager.finish_query().unwrap();
    }
    #[test]
    fn namespace_rename_during_provider_admission_never_recreates_ambient_path() {
        struct RenameProvider {
            authority: TestAuthority,
            rename: parking_lot::Mutex<Option<PathBuf>>,
        }
        impl SpillRootAuthority for RenameProvider {
            fn store_id(&self) -> StoreId {
                self.authority.store_id()
            }
            fn key_id(&self) -> [u8; 32] {
                self.authority.key_id()
            }
            fn authenticate_marker(&self, message: &[u8]) -> io::Result<[u8; 32]> {
                self.authority.authenticate_marker(message)
            }
            fn verify_marker(&self, message: &[u8], auth: &[u8; 32]) -> io::Result<bool> {
                self.authority.verify_marker(message, auth)
            }
            fn record_provider(
                &self,
                identity: SpillQueryIdentity,
            ) -> io::Result<Arc<dyn SpillRecordProvider>> {
                if let Some(path) = self.rename.lock().take() {
                    fs::rename(&path, path.with_file_name("renamed-authorized-root"))?;
                }
                self.authority.record_provider(identity)
            }
        }
        let parent = TempDir::new().unwrap();
        let authority = Arc::new(RenameProvider {
            authority: TestAuthority {
                store: StoreId::from_bytes([1; 32]).unwrap(),
                key: 7,
            },
            rename: parking_lot::Mutex::new(None),
        });
        let root = SpillRoot::open(
            parent.path(),
            authority.clone(),
            SpillFrameLimits::format_max(),
            Arc::new(NoopSpillIo),
            SpillDiskQuota::new(1 << 20),
            None,
        )
        .unwrap();
        let path = root.namespace_path().to_owned();
        *authority.rename.lock() = Some(path.clone());
        let (context, control) = query();
        assert!(
            root.begin_query(context.query_id(), control.token())
                .is_err()
        );
        assert!(!path.exists());
        assert_eq!(
            fs::read_dir(parent.path().join("renamed-authorized-root"))
                .unwrap()
                .count(),
            3
        );
    }
    #[test]
    fn failed_leaf_authentication_preserves_foreign_owner_marker() {
        struct InjectForeignMarker {
            authority: TestAuthority,
            fail_auth: bool,
            namespace: parking_lot::Mutex<Option<PathBuf>>,
            marker: parking_lot::Mutex<Option<PathBuf>>,
        }
        impl SpillRootAuthority for InjectForeignMarker {
            fn store_id(&self) -> StoreId {
                self.authority.store_id()
            }
            fn key_id(&self) -> [u8; 32] {
                self.authority.key_id()
            }
            fn authenticate_marker(&self, message: &[u8]) -> io::Result<[u8; 32]> {
                if message.len() == 149 && message.starts_with(b"GRAQ") {
                    let namespace = self.namespace.lock();
                    let leaf = fs::read_dir(namespace.as_ref().unwrap())?
                        .map(|entry| entry.unwrap().path())
                        .find(|path| path.is_dir())
                        .unwrap();
                    let marker = leaf.join(".grafeo-spill-owner");
                    fs::write(&marker, b"foreign owner marker must survive")?;
                    *self.marker.lock() = Some(marker);
                    if self.fail_auth {
                        return Err(io::Error::other("injected leaf authentication failure"));
                    }
                }
                self.authority.authenticate_marker(message)
            }
            fn verify_marker(&self, message: &[u8], auth: &[u8; 32]) -> io::Result<bool> {
                self.authority.verify_marker(message, auth)
            }
            fn record_provider(
                &self,
                identity: SpillQueryIdentity,
            ) -> io::Result<Arc<dyn SpillRecordProvider>> {
                self.authority.record_provider(identity)
            }
        }
        for fail_auth in [true, false] {
            let parent = TempDir::new().unwrap();
            let authority = Arc::new(InjectForeignMarker {
                fail_auth,
                authority: TestAuthority {
                    store: StoreId::from_bytes([1; 32]).unwrap(),
                    key: 7,
                },
                namespace: parking_lot::Mutex::new(None),
                marker: parking_lot::Mutex::new(None),
            });
            let root = SpillRoot::open(
                parent.path(),
                authority.clone(),
                SpillFrameLimits::format_max(),
                Arc::new(NoopSpillIo),
                SpillDiskQuota::new(1 << 20),
                None,
            )
            .unwrap();
            *authority.namespace.lock() = Some(root.namespace_path().to_owned());
            let (context, control) = query();
            assert!(
                root.begin_query(context.query_id(), control.token())
                    .is_err()
            );
            let marker = authority.marker.lock();
            assert_eq!(
                fs::read(marker.as_ref().unwrap()).unwrap(),
                b"foreign owner marker must survive"
            );
        }
    }
    #[test]
    fn replaced_marker_is_preserved_even_with_matching_bytes_and_successful_write_hook() {
        use crate::execution::spill::SpillIoOperation;
        use std::os::unix::fs::PermissionsExt as _;
        struct ReplaceMarker {
            namespace: parking_lot::Mutex<Option<PathBuf>>,
            replaced: parking_lot::Mutex<Option<(PathBuf, Vec<u8>)>>,
            fail_after_replace: bool,
        }
        impl SpillIo for ReplaceMarker {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::Sync {
                    let namespace = self.namespace.lock();
                    let leaf = fs::read_dir(namespace.as_ref().unwrap())?
                        .map(|entry| entry.unwrap().path())
                        .find(|path| path.is_dir())
                        .unwrap();
                    let marker = leaf.join(".grafeo-spill-owner");
                    let bytes = fs::read(&marker)?;
                    fs::remove_file(&marker)?;
                    fs::write(&marker, &bytes)?;
                    fs::set_permissions(&marker, fs::Permissions::from_mode(0o600))?;
                    *self.replaced.lock() = Some((marker, bytes));
                    if self.fail_after_replace {
                        return Err(io::Error::other("injected replacement failure"));
                    }
                }
                Ok(())
            }
        }
        for fail_after_replace in [true, false] {
            let parent = TempDir::new().unwrap();
            let io = Arc::new(ReplaceMarker {
                namespace: parking_lot::Mutex::new(None),
                replaced: parking_lot::Mutex::new(None),
                fail_after_replace,
            });
            let root = SpillRoot::open(
                parent.path(),
                authority(1, 7),
                SpillFrameLimits::format_max(),
                io.clone(),
                SpillDiskQuota::new(1 << 20),
                None,
            )
            .unwrap();
            *io.namespace.lock() = Some(root.namespace_path().to_owned());
            let (context, control) = query();
            assert!(
                root.begin_query(context.query_id(), control.token())
                    .is_err()
            );
            let replaced = io.replaced.lock();
            let (path, expected) = replaced.as_ref().unwrap();
            assert_eq!(&fs::read(path).unwrap(), expected);
        }
    }

    #[test]
    fn owned_partial_marker_is_removed_after_write_failure_or_unwind() {
        use crate::execution::spill::SpillIoOperation;
        struct PartialWrite {
            namespace: parking_lot::Mutex<Option<PathBuf>>,
            panic_after_write: bool,
        }
        impl SpillIo for PartialWrite {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::WritePayload {
                    let namespace = self.namespace.lock();
                    let leaf = fs::read_dir(namespace.as_ref().unwrap())?
                        .map(|entry| entry.unwrap().path())
                        .find(|path| path.is_dir())
                        .unwrap();
                    fs::write(leaf.join(".grafeo-spill-owner"), b"partial owned write")?;
                    assert!(!self.panic_after_write, "injected marker writer unwind");
                    return Err(io::Error::other("injected partial write failure"));
                }
                Ok(())
            }
        }
        for panic_after_write in [false, true] {
            let parent = TempDir::new().unwrap();
            let io = Arc::new(PartialWrite {
                namespace: parking_lot::Mutex::new(None),
                panic_after_write,
            });
            let root = SpillRoot::open(
                parent.path(),
                authority(1, 7),
                SpillFrameLimits::format_max(),
                io.clone(),
                SpillDiskQuota::new(1 << 20),
                None,
            )
            .unwrap();
            *io.namespace.lock() = Some(root.namespace_path().to_owned());
            let (context, control) = query();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                root.begin_query(context.query_id(), control.token())
            }));
            if panic_after_write {
                assert!(outcome.is_err());
            } else {
                assert!(outcome.unwrap().is_err());
            }
            assert_eq!(fs::read_dir(root.namespace_path()).unwrap().count(), 3);
        }
    }
    #[test]
    fn leaf_lease_survives_manager_and_file_until_the_last_reader_drops() {
        for keep_reader in [false, true] {
            let parent = TempDir::new().unwrap();
            let root = open(parent.path(), 1, 7).unwrap();
            let (context, control) = query();
            let manager = SpillManager::from_query_lease(
                root.begin_query(context.query_id(), control.token())
                    .unwrap(),
            );
            let marker = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(manager.spill_dir().join(".grafeo-spill-owner"))
                .unwrap();
            let directory = fs::File::open(manager.spill_dir()).unwrap();
            let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
            drop(manager);
            assert!(matches!(
                marker.try_lock(),
                Err(fs::TryLockError::WouldBlock)
            ));
            assert!(matches!(
                directory.try_lock(),
                Err(fs::TryLockError::WouldBlock)
            ));
            file.write_sort_run_start(1, 0).unwrap();
            file.finish_write().unwrap();
            let reader = keep_reader.then(|| file.reader().unwrap());
            drop(file);
            if keep_reader {
                assert!(matches!(
                    marker.try_lock(),
                    Err(fs::TryLockError::WouldBlock)
                ));
                assert!(matches!(
                    directory.try_lock(),
                    Err(fs::TryLockError::WouldBlock)
                ));
            }
            drop(reader);
            marker.try_lock().unwrap();
            marker.unlock().unwrap();
            directory.try_lock().unwrap();
            directory.unlock().unwrap();
        }
    }

    fn quota_root(parent: &Path, limit: u64, io: Arc<dyn SpillIo>) -> Arc<SpillRoot> {
        SpillRoot::open(
            parent,
            authority(1, 7),
            SpillFrameLimits::format_max(),
            io,
            SpillDiskQuota::new(64 << 20),
            Some(limit),
        )
        .unwrap()
    }

    fn quota_used(root: &SpillRoot) -> u64 {
        let bytes = fs::read(root.namespace_path().join(".grafeo-spill-quota")).unwrap();
        u64::from_le_bytes(bytes[64..72].try_into().unwrap())
    }

    fn quota_manager(root: &Arc<SpillRoot>) -> SpillManager {
        let (context, control) = query();
        SpillManager::from_query_lease(
            root.begin_query(context.query_id(), control.token())
                .unwrap(),
        )
    }

    #[test]
    fn root_quota_real_writer_denies_before_growth_and_releases_after_cleanup() {
        use super::super::quota::{ALLOCATION_UNIT, CONTROL_RESERVE, FILE_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let limit = CONTROL_RESERVE + LEAF_RESERVE + FILE_RESERVE + ALLOCATION_UNIT;
        let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        let before = fs::metadata(file.path()).unwrap().len();
        let error = file
            .write_sort_row(&vec![0; usize::try_from(ALLOCATION_UNIT).unwrap()])
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::QuotaExceeded);
        assert_eq!(fs::metadata(file.path()).unwrap().len(), before);
        assert_eq!(quota_used(&root), limit);
        drop(file);
        assert_eq!(
            quota_used(&root),
            limit,
            "query retains durable reusable capacity"
        );
        manager.finish_query().unwrap();
        drop(manager);
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        file.write_sort_row(b"small record").unwrap();
        file.finish_write().unwrap();
        file.close_and_delete().unwrap();
        manager.finish_query().unwrap();
        drop(manager);
        assert_eq!(
            quota_used(&root),
            CONTROL_RESERVE,
            "explicit completion releases the closed leaf even while the deleted file object survives"
        );
        drop(file);
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn root_quota_reuses_deleted_capacity_without_durable_transactions() {
        use super::super::quota::{ALLOCATION_UNIT, CONTROL_RESERVE, FILE_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let limit = CONTROL_RESERVE + LEAF_RESERVE + FILE_RESERVE + ALLOCATION_UNIT;
        let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
        let manager = quota_manager(&root);
        let ledger = root.namespace_path().join(".grafeo-spill-quota");
        let mut previous = None;
        for _ in 0..12 {
            let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
            file.write_sort_run_start(1, 1).unwrap();
            file.write_sort_row(b"capacity reused only after proved deletion")
                .unwrap();
            file.finish_write().unwrap();
            assert_eq!(quota_used(&root), limit);
            let current = fs::read(&ledger).unwrap();
            if let Some(previous) = &previous {
                assert_eq!(
                    &current, previous,
                    "no new durable generation for reused capacity"
                );
            }
            // A live receipt cannot lend its allocation to a second file.
            assert_eq!(
                manager
                    .create_file(SpillFileRole::SortRun)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::QuotaExceeded
            );
            file.close_and_delete().unwrap();
            assert_eq!(
                fs::read(&ledger).unwrap(),
                current,
                "local credit leaves root debt intact"
            );
            previous = Some(current);
        }
        manager.finish_query().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn root_quota_concurrent_file_receipts_cannot_overcommit_query_capacity() {
        use super::super::quota::{ALLOCATION_UNIT, CONTROL_RESERVE, FILE_RESERVE, LEAF_RESERVE};
        use std::sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        };
        let parent = TempDir::new().unwrap();
        let limit = CONTROL_RESERVE + LEAF_RESERVE + 4 * (FILE_RESERVE + ALLOCATION_UNIT);
        let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
        let manager = Arc::new(quota_manager(&root));
        let barrier = Arc::new(Barrier::new(9));
        let admitted = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let manager = Arc::clone(&manager);
            let barrier = Arc::clone(&barrier);
            let admitted = Arc::clone(&admitted);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                let file = manager.create_file(SpillFileRole::SortRun);
                if file.is_ok() {
                    admitted.fetch_add(1, Ordering::SeqCst);
                }
                barrier.wait();
                barrier.wait();
                match file {
                    Ok(mut file) => file.close_and_delete().unwrap(),
                    Err(error) => assert_eq!(error.kind(), io::ErrorKind::QuotaExceeded),
                }
            }));
        }
        barrier.wait();
        barrier.wait();
        let count = admitted.load(Ordering::SeqCst);
        let used = quota_used(&root);
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(count, 4);
        assert_eq!(used, limit);
        assert_eq!(
            quota_used(&root),
            limit,
            "unused pool remains charged until leaf cleanup"
        );
        manager.finish_query().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn root_quota_renamed_or_aliased_files_keep_restart_debt() {
        use super::super::quota::{ALLOCATION_UNIT, CONTROL_RESERVE, FILE_RESERVE, LEAF_RESERVE};
        for alias in [false, true] {
            let parent = TempDir::new().unwrap();
            let limit = CONTROL_RESERVE + LEAF_RESERVE + FILE_RESERVE + ALLOCATION_UNIT;
            let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
            let manager = quota_manager(&root);
            let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
            let foreign = manager.spill_dir().join("retained-data");
            if alias {
                fs::hard_link(file.path(), &foreign).unwrap();
            } else {
                fs::rename(file.path(), &foreign).unwrap();
            }
            assert!(file.close_and_delete().is_err());
            drop(file);
            assert!(manager.finish_query().is_err());
            drop(manager);
            assert!(foreign.exists());
            assert_eq!(quota_used(&root), limit);
            drop(root);
            let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
            let (context, control) = query();
            assert_eq!(
                root.begin_query(context.query_id(), control.token())
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::QuotaExceeded
            );
            assert_eq!(quota_used(&root), limit);
            assert!(foreign.exists());
        }
    }

    #[test]
    fn root_quota_missing_corrupt_and_conflicting_policy_never_reset_debt() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        for corrupt in [false, true] {
            let parent = TempDir::new().unwrap();
            let limit = CONTROL_RESERVE + 4 * LEAF_RESERVE;
            let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
            let manager = quota_manager(&root);
            let ledger = root.namespace_path().join(".grafeo-spill-quota");
            let original = fs::read(&ledger).unwrap();
            assert!(
                SpillRoot::open(
                    parent.path(),
                    authority(1, 7),
                    SpillFrameLimits::format_max(),
                    Arc::new(NoopSpillIo),
                    SpillDiskQuota::new(64 << 20),
                    Some(limit + 1)
                )
                .is_err()
            );
            assert_eq!(fs::read(&ledger).unwrap(), original);
            if corrupt {
                let mut changed = original.clone();
                changed[64] ^= 1;
                fs::write(&ledger, changed).unwrap();
            } else {
                fs::remove_file(&ledger).unwrap();
            }
            assert!(
                SpillRoot::open(
                    parent.path(),
                    authority(1, 7),
                    SpillFrameLimits::format_max(),
                    Arc::new(NoopSpillIo),
                    SpillDiskQuota::new(64 << 20),
                    Some(limit)
                )
                .is_err()
            );
            assert!(manager.spill_dir().exists());
            // Restore only our fixture for ordinary owned cleanup.
            fs::write(&ledger, original).unwrap();
            use std::os::unix::fs::PermissionsExt as _;
            fs::set_permissions(&ledger, fs::Permissions::from_mode(0o600)).unwrap();
            manager.finish_query().unwrap();
        }
    }

    #[test]
    fn root_quota_stale_foreign_and_overflow_receipts_fail_closed() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE, ReservationKey};
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + 4 * LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let key = ReservationKey::leaf([3; 16]);
        let (birth, _) = root.ledger.reserve(key, None).unwrap();
        let used = quota_used(&root);
        assert!(
            root.ledger
                .ensure_capacity(key, birth, u64::MAX, false, None)
                .is_err()
        );
        assert!(
            root.ledger
                .ensure_capacity(key, birth + 1, LEAF_RESERVE, false, None)
                .is_err()
        );
        assert!(
            root.ledger
                .release_after_delete(ReservationKey::leaf([9; 16]), birth, &mut None)
                .is_err()
        );
        assert_eq!(quota_used(&root), used);
        root.ledger
            .release_after_delete(key, birth, &mut None)
            .unwrap();
        let (new_birth, _) = root.ledger.reserve(key, None).unwrap();
        assert!(new_birth > birth);
        assert!(
            root.ledger
                .release_after_delete(key, birth, &mut None)
                .is_err()
        );
        assert_eq!(quota_used(&root), used);
        root.ledger
            .release_after_delete(key, new_birth, &mut None)
            .unwrap();
    }

    #[test]
    fn root_quota_release_sync_failure_retries_without_double_credit() {
        use super::super::{
            SpillIoOperation,
            quota::{CONTROL_RESERVE, LEAF_RESERVE},
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct FailNthSync(AtomicUsize);
        impl SpillIo for FailNthSync {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::QuotaSync {
                    let old = self.0.load(Ordering::SeqCst);
                    if old > 0 && self.0.fetch_sub(1, Ordering::SeqCst) == 1 {
                        return Err(io::Error::other("injected post-rename sync failure"));
                    }
                }
                Ok(())
            }
        }
        let parent = TempDir::new().unwrap();
        let faults = Arc::new(FailNthSync(AtomicUsize::new(0)));
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + 8 * LEAF_RESERVE,
            faults.clone(),
        );
        let manager = quota_manager(&root);
        let before_growth = quota_used(&root);
        faults.0.store(2, Ordering::SeqCst);
        assert!(manager.create_file(SpillFileRole::SortRun).is_err());
        let retained = quota_used(&root);
        assert!(
            retained > before_growth,
            "published capacity survives a failed parent sync"
        );
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        assert_eq!(
            quota_used(&root),
            retained,
            "retry consumes the durable capacity exactly once"
        );
        faults.0.store(2, Ordering::SeqCst);
        file.close_and_delete().unwrap();
        assert!(!file.path().exists());
        file.close_and_delete().unwrap();
        assert_eq!(quota_used(&root), retained);
        assert_eq!(
            faults.0.load(Ordering::SeqCst),
            2,
            "file credit has no durable ledger update"
        );
        drop(file);
        assert!(manager.finish_query().is_err());
        assert!(!manager.spill_dir().exists());
        manager.finish_query().unwrap();
        manager.finish_query().unwrap();
        drop(manager);
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    fn wait_quota_fixture(path: &Path) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !path.exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "quota child barrier timeout: {}",
                path.display()
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn root_quota_process_child() {
        use super::super::quota::{ALLOCATION_UNIT, CONTROL_RESERVE, FILE_RESERVE, LEAF_RESERVE};
        let Ok(parent) = std::env::var("GRAFEO_QUOTA_TEST_PARENT") else {
            return;
        };
        let id = std::env::var("GRAFEO_QUOTA_TEST_ID").unwrap();
        let parent = Path::new(&parent);
        let limit = CONTROL_RESERVE + 2 * LEAF_RESERVE + 2 * FILE_RESERVE + 100 * ALLOCATION_UNIT;
        let root = quota_root(parent, limit, Arc::new(NoopSpillIo));
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        fs::write(parent.join(format!("ready-{id}")), []).unwrap();
        wait_quota_fixture(&parent.join("go"));
        // Headers and terminator fit within the last of 60 allocation units.
        let result = file
            .write_sort_row(&vec![
                0x5a;
                usize::try_from(60 * ALLOCATION_UNIT - 4096).unwrap()
            ])
            .and_then(|()| file.finish_write());
        let outcome = match result {
            Ok(()) => "admitted",
            Err(error) => {
                assert_eq!(error.kind(), io::ErrorKind::QuotaExceeded, "{error}");
                "denied"
            }
        };
        // The parent uses existence as its readiness signal. Publish complete
        // bytes atomically so it cannot observe create-before-write emptiness.
        let installing = parent.join(format!("result-{id}.installing"));
        fs::write(&installing, outcome).unwrap();
        fs::rename(installing, parent.join(format!("result-{id}"))).unwrap();
        wait_quota_fixture(&parent.join("exit"));
        if std::env::var_os("GRAFEO_QUOTA_TEST_CRASH").is_some() {
            file.close_and_delete().unwrap();
            // A process can die with reusable capacity and no live files. The
            // query high-water debt must still survive the missing leaf cleanup.
            std::process::exit(0);
        }
        drop(file);
        manager.finish_query().unwrap();
    }

    #[test]
    fn root_quota_two_process_sixty_plus_sixty_over_hundred_and_restart_debt() {
        use super::super::quota::{ALLOCATION_UNIT, CONTROL_RESERVE, FILE_RESERVE, LEAF_RESERVE};
        for crash in [false, true] {
            let parent = TempDir::new().unwrap();
            let limit =
                CONTROL_RESERVE + 2 * LEAF_RESERVE + 2 * FILE_RESERVE + 100 * ALLOCATION_UNIT;
            let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
            struct Children(Vec<std::process::Child>);
            impl Drop for Children {
                fn drop(&mut self) {
                    for child in &mut self.0 {
                        if child.try_wait().is_ok_and(|status| status.is_none()) {
                            let _ = child.kill();
                        }
                        let _ = child.wait();
                    }
                }
            }
            let mut children = Children(Vec::new());
            for id in ["a", "b"] {
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                command
                    .args([
                        "--exact",
                        "execution::spill::root::tests::root_quota_process_child",
                        "--nocapture",
                    ])
                    .env("GRAFEO_QUOTA_TEST_PARENT", parent.path())
                    .env("GRAFEO_QUOTA_TEST_ID", id);
                if crash {
                    command.env("GRAFEO_QUOTA_TEST_CRASH", "1");
                }
                children.0.push(command.spawn().unwrap());
            }
            for id in ["a", "b"] {
                wait_quota_fixture(&parent.path().join(format!("ready-{id}")));
            }
            fs::write(parent.path().join("go"), []).unwrap();
            let mut outcomes = Vec::new();
            for id in ["a", "b"] {
                let path = parent.path().join(format!("result-{id}"));
                wait_quota_fixture(&path);
                outcomes.push(fs::read_to_string(path).unwrap());
            }
            outcomes.sort();
            assert_eq!(outcomes, ["admitted", "denied"]);
            let debt = quota_used(&root);
            assert!(debt <= limit);
            assert!(debt >= CONTROL_RESERVE + 60 * ALLOCATION_UNIT);
            fs::write(parent.path().join("exit"), []).unwrap();
            for child in &mut children.0 {
                assert!(child.wait().unwrap().success());
            }
            drop(root);
            let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
            assert_eq!(
                quota_used(&root),
                if crash { debt } else { CONTROL_RESERVE }
            );
            if crash {
                let manager = quota_manager(&root);
                let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
                file.write_sort_run_start(1, 1).unwrap();
                assert_eq!(
                    file.write_sort_row(&vec![
                        0;
                        usize::try_from(60 * ALLOCATION_UNIT - 4096).unwrap()
                    ])
                    .unwrap_err()
                    .kind(),
                    io::ErrorKind::QuotaExceeded
                );
            }
        }
    }

    #[test]
    fn root_quota_reserve_and_release_faults_keep_accounting_retryable() {
        use super::super::{
            SpillIoOperation as Op,
            quota::{CONTROL_RESERVE, LEAF_RESERVE},
        };
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Fault {
            operation: Op,
            armed: AtomicBool,
        }
        impl SpillIo for Fault {
            fn check(&self, operation: Op) -> io::Result<()> {
                if operation == self.operation && self.armed.swap(false, Ordering::SeqCst) {
                    Err(io::Error::other("injected quota boundary failure"))
                } else {
                    Ok(())
                }
            }
        }
        for operation in [
            Op::QuotaReserve,
            Op::QuotaWrite,
            Op::QuotaSync,
            Op::QuotaPublish,
        ] {
            let parent = TempDir::new().unwrap();
            let faults = Arc::new(Fault {
                operation,
                armed: AtomicBool::new(false),
            });
            let root = quota_root(
                parent.path(),
                CONTROL_RESERVE + 8 * LEAF_RESERVE,
                faults.clone(),
            );
            faults.armed.store(true, Ordering::SeqCst);
            let (context, control) = query();
            assert!(
                root.begin_query(context.query_id(), control.token())
                    .is_err(),
                "{operation:?}"
            );
            assert_eq!(quota_used(&root), CONTROL_RESERVE, "{operation:?}");
            assert!(
                !root
                    .namespace_path()
                    .join(".grafeo-spill-quota.installing")
                    .exists()
            );
            let manager = quota_manager(&root);
            let before_growth = quota_used(&root);
            faults.armed.store(true, Ordering::SeqCst);
            assert!(
                manager.create_file(SpillFileRole::SortRun).is_err(),
                "pool growth {operation:?}"
            );
            assert!(manager.physical_stats().unwrap().reservation_uncertain);
            assert_eq!(
                quota_used(&root),
                before_growth,
                "failed unpublished growth {operation:?}"
            );
            let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
            assert!(!manager.physical_stats().unwrap().reservation_uncertain);
            file.close_and_delete().unwrap();
            manager.finish_query().unwrap();
            assert!(!manager.physical_stats().unwrap().reservation_uncertain);
            assert_eq!(quota_used(&root), CONTROL_RESERVE);
        }
        for operation in [Op::QuotaDeleteSync, Op::QuotaRelease] {
            let parent = TempDir::new().unwrap();
            let faults = Arc::new(Fault {
                operation,
                armed: AtomicBool::new(false),
            });
            let root = quota_root(
                parent.path(),
                CONTROL_RESERVE + 8 * LEAF_RESERVE,
                faults.clone(),
            );
            let manager = quota_manager(&root);
            let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
            let before = quota_used(&root);
            faults.armed.store(true, Ordering::SeqCst);
            assert!(file.close_and_delete().is_err(), "{operation:?}");
            assert_eq!(quota_used(&root), before, "{operation:?}");
            assert!(!file.path().exists());
            file.close_and_delete().unwrap();
            assert_eq!(
                quota_used(&root),
                before,
                "reusable capacity stays durably charged"
            );
            drop(file);
            manager.finish_query().unwrap();
            drop(manager);
            assert_eq!(quota_used(&root), CONTROL_RESERVE);
        }
    }

    #[test]
    fn root_quota_missing_ledger_during_delete_is_an_error_and_remains_retryable() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        use std::os::unix::fs::PermissionsExt as _;
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + 8 * LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let ledger = root.namespace_path().join(".grafeo-spill-quota");
        let original = fs::read(&ledger).unwrap();
        fs::remove_file(&ledger).unwrap();
        assert!(file.close_and_delete().is_err());
        assert!(!file.path().exists());
        assert_eq!(
            manager.active_file_count(),
            1,
            "failed cleanup must remain tracked"
        );
        fs::write(&ledger, original).unwrap();
        fs::set_permissions(&ledger, fs::Permissions::from_mode(0o600)).unwrap();
        file.close_and_delete().unwrap();
        assert_eq!(manager.active_file_count(), 0);
        drop(file);
        manager.finish_query().unwrap();
        drop(manager);
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn root_quota_publication_rejects_replaced_lock_and_private_mode_changes() {
        use super::super::{
            SpillIoOperation,
            quota::{CONTROL_RESERVE, LEAF_RESERVE},
        };
        use std::os::unix::fs::PermissionsExt as _;
        struct Mutate {
            path: parking_lot::Mutex<Option<PathBuf>>,
            lock: bool,
        }
        impl SpillIo for Mutate {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::QuotaPublish
                    && let Some(path) = self.path.lock().take()
                {
                    if self.lock {
                        fs::rename(
                            path.join(".grafeo-spill-quota.lock"),
                            path.join("owned-lock"),
                        )?;
                        fs::write(path.join(".grafeo-spill-quota.lock"), [])?;
                        fs::set_permissions(
                            path.join(".grafeo-spill-quota.lock"),
                            fs::Permissions::from_mode(0o600),
                        )?;
                    } else {
                        fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
                    }
                }
                Ok(())
            }
        }
        for lock in [false, true] {
            let parent = TempDir::new().unwrap();
            let mutation = Arc::new(Mutate {
                path: parking_lot::Mutex::new(None),
                lock,
            });
            let root = quota_root(
                parent.path(),
                CONTROL_RESERVE + 8 * LEAF_RESERVE,
                mutation.clone(),
            );
            let before = fs::read(root.namespace_path().join(".grafeo-spill-quota")).unwrap();
            *mutation.path.lock() = Some(root.namespace_path().to_owned());
            let (context, control) = query();
            assert!(
                root.begin_query(context.query_id(), control.token())
                    .is_err()
            );
            assert_eq!(
                fs::read(root.namespace_path().join(".grafeo-spill-quota")).unwrap(),
                before
            );
            if lock {
                assert!(root.namespace_path().join("owned-lock").exists());
                fs::remove_file(root.namespace_path().join(".grafeo-spill-quota.lock")).unwrap();
                fs::rename(
                    root.namespace_path().join("owned-lock"),
                    root.namespace_path().join(".grafeo-spill-quota.lock"),
                )
                .unwrap();
            } else {
                fs::set_permissions(root.namespace_path(), fs::Permissions::from_mode(0o700))
                    .unwrap();
            }
            let manager = quota_manager(&root);
            manager.finish_query().unwrap();
        }
    }

    #[test]
    fn root_quota_explicit_leaf_release_reports_failures_and_retries_without_reopening() {
        use super::super::{
            SpillIoOperation as Op,
            quota::{CONTROL_RESERVE, LEAF_RESERVE},
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Fault {
            operation: Op,
            countdown: AtomicUsize,
        }
        impl SpillIo for Fault {
            fn check(&self, operation: Op) -> io::Result<()> {
                if operation == self.operation
                    && self.countdown.load(Ordering::SeqCst) > 0
                    && self.countdown.fetch_sub(1, Ordering::SeqCst) == 1
                {
                    return Err(io::Error::other("injected leaf quota release failure"));
                }
                Ok(())
            }
        }
        for (operation, nth) in [
            (Op::QuotaDeleteSync, 1),
            (Op::QuotaRelease, 1),
            (Op::QuotaWrite, 1),
            (Op::QuotaSync, 1),
            (Op::QuotaPublish, 1),
            (Op::QuotaSync, 2),
        ] {
            let parent = TempDir::new().unwrap();
            let faults = Arc::new(Fault {
                operation,
                countdown: AtomicUsize::new(0),
            });
            let root = quota_root(
                parent.path(),
                CONTROL_RESERVE + LEAF_RESERVE,
                faults.clone(),
            );
            let manager = quota_manager(&root);
            faults.countdown.store(nth, Ordering::SeqCst);
            assert!(manager.finish_query().is_err(), "{operation:?}/{nth}");
            assert!(!manager.spill_dir().exists());
            assert!(manager.create_file(SpillFileRole::SortRun).is_err());
            if nth == 1 {
                assert_eq!(quota_used(&root), CONTROL_RESERVE + LEAF_RESERVE);
            }
            manager.finish_query().unwrap();
            manager.finish_query().unwrap();
            assert_eq!(quota_used(&root), CONTROL_RESERVE);
            let second = quota_manager(&root); // Credit usable before the first owner drops.
            second.finish_query().unwrap();
        }
    }

    #[test]
    fn root_quota_rejects_authenticated_generation_regression() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + 8 * LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let ledger = root.namespace_path().join(".grafeo-spill-quota");
        let old = fs::read(&ledger).unwrap();
        let manager = quota_manager(&root);
        let current = fs::read(&ledger).unwrap();
        fs::write(&ledger, old).unwrap();
        assert!(manager.create_file(SpillFileRole::SortRun).is_err());
        fs::write(&ledger, current).unwrap();
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        drop(file);
        manager.finish_query().unwrap();
    }

    #[test]
    fn root_quota_lock_contention_observes_cancellation_without_admission() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + 8 * LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(root.namespace_path().join(".grafeo-spill-quota.lock"))
            .unwrap();
        lock.try_lock().unwrap();
        let (context, control) = query();
        let token = control.token();
        let cancel = control.cancellation_handle();
        let started = std::time::Instant::now();
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| root.begin_query(context.query_id(), token));
            std::thread::sleep(std::time::Duration::from_millis(20));
            cancel.cancel();
            assert_eq!(
                worker.join().unwrap().unwrap_err().kind(),
                io::ErrorKind::Interrupted
            );
        });
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
        lock.unlock().unwrap();
        let manager = quota_manager(&root);
        manager.finish_query().unwrap();
    }

    #[test]
    fn root_quota_prepaid_writes_stay_bounded_and_publication_revalidates_authority() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + 8 * LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        let ledger = root.namespace_path().join(".grafeo-spill-quota");
        let original = fs::read(&ledger).unwrap();
        let mut corrupt = original.clone();
        corrupt[64] ^= 1;
        fs::write(&ledger, corrupt).unwrap();
        // The receipt still owns its prepaid capacity, but corruption prevents
        // both further admission and publication of the paid-for data.
        file.write_sort_row(b"within the durable prepaid capacity")
            .unwrap();
        assert!(file.finish_write().is_err());
        assert!(file.reader().is_err());
        fs::write(&ledger, original).unwrap();
        file.close_and_delete().unwrap();
        manager.finish_query().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn root_quota_concurrent_root_authentication_waits_for_the_retained_marker() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let limit = CONTROL_RESERVE + 8 * LEAF_RESERVE;
        let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
        let marker = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(root.namespace_path().join(".grafeo-spill-root"))
            .unwrap();
        marker.try_lock().unwrap();
        std::thread::scope(|scope| {
            let opened = scope.spawn(|| quota_root(parent.path(), limit, Arc::new(NoopSpillIo)));
            std::thread::sleep(std::time::Duration::from_millis(25));
            marker.unlock().unwrap();
            let second = opened.join().unwrap();
            assert_eq!(second.namespace_path(), root.namespace_path());
            let manager = quota_manager(&second);
            manager.finish_query().unwrap();
        });
    }

    #[test]
    fn root_quota_publication_hook_cannot_grow_the_file_after_reconciliation() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + 8 * LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        let path = file.path().to_owned();
        let result = file.finish_write_before_publish(|| {
            let handle = fs::OpenOptions::new().write(true).open(&path).unwrap();
            handle.set_len(256 << 10).unwrap();
            Ok::<_, std::convert::Infallible>(())
        });
        assert!(result.is_err());
        assert!(file.reader().is_err());
        file.close_and_delete().unwrap();
        manager.finish_query().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn root_quota_revalidates_authority_after_locked_authentication() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        use std::os::unix::fs::PermissionsExt as _;
        struct MutatingAuthority {
            base: TestAuthority,
            action: usize,
            armed: parking_lot::Mutex<Option<PathBuf>>,
        }
        impl SpillRootAuthority for MutatingAuthority {
            fn store_id(&self) -> StoreId {
                self.base.store_id()
            }
            fn key_id(&self) -> [u8; 32] {
                self.base.key_id()
            }
            fn authenticate_marker(&self, message: &[u8]) -> io::Result<[u8; 32]> {
                self.base.authenticate_marker(message)
            }
            fn verify_marker(&self, message: &[u8], auth: &[u8; 32]) -> io::Result<bool> {
                if message.starts_with(b"GRAQLED1")
                    && let Some(path) = self.armed.lock().take()
                {
                    match self.action {
                        0 => {
                            fs::rename(
                                path.join(".grafeo-spill-quota.lock"),
                                path.join("owned-lock"),
                            )?;
                            fs::write(path.join(".grafeo-spill-quota.lock"), [])?;
                            fs::set_permissions(
                                path.join(".grafeo-spill-quota.lock"),
                                fs::Permissions::from_mode(0o600),
                            )?;
                        }
                        1 => {
                            let marker = path.join(".grafeo-spill-root");
                            let mut bytes = fs::read(&marker)?;
                            bytes[90] ^= 1;
                            fs::write(marker, bytes)?;
                        }
                        _ => fs::set_permissions(path, fs::Permissions::from_mode(0o755))?,
                    }
                }
                self.base.verify_marker(message, auth)
            }
            fn record_provider(
                &self,
                identity: SpillQueryIdentity,
            ) -> io::Result<Arc<dyn SpillRecordProvider>> {
                self.base.record_provider(identity)
            }
        }
        for action in 0..3 {
            let parent = TempDir::new().unwrap();
            let authority = Arc::new(MutatingAuthority {
                base: TestAuthority {
                    store: StoreId::from_bytes([1; 32]).unwrap(),
                    key: 7,
                },
                action,
                armed: parking_lot::Mutex::new(None),
            });
            let root = SpillRoot::open(
                parent.path(),
                authority.clone(),
                SpillFrameLimits::format_max(),
                Arc::new(NoopSpillIo),
                SpillDiskQuota::new(64 << 20),
                Some(CONTROL_RESERVE + 8 * LEAF_RESERVE),
            )
            .unwrap();
            let manager = quota_manager(&root);
            let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
            file.write_sort_run_start(1, 0).unwrap();
            let namespace = root.namespace_path();
            let marker = fs::read(namespace.join(".grafeo-spill-root")).unwrap();
            let ledger = fs::read(namespace.join(".grafeo-spill-quota")).unwrap();
            *authority.armed.lock() = Some(namespace.to_owned());
            assert!(
                file.finish_write().is_err(),
                "authority changed after acquiring lock: {action}"
            );
            assert!(file.reader().is_err());
            assert_eq!(
                fs::read(namespace.join(".grafeo-spill-quota")).unwrap(),
                ledger
            );
            match action {
                0 => {
                    fs::remove_file(namespace.join(".grafeo-spill-quota.lock")).unwrap();
                    fs::rename(
                        namespace.join("owned-lock"),
                        namespace.join(".grafeo-spill-quota.lock"),
                    )
                    .unwrap();
                }
                1 => fs::write(namespace.join(".grafeo-spill-root"), marker).unwrap(),
                _ => fs::set_permissions(namespace, fs::Permissions::from_mode(0o700)).unwrap(),
            }
            file.close_and_delete().unwrap();
            manager.finish_query().unwrap();
            assert_eq!(quota_used(&root), CONTROL_RESERVE);
        }
    }

    #[test]
    fn root_quota_authority_callback_cannot_change_the_published_allocation() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        struct MutatingAuthority {
            base: TestAuthority,
            mutate: parking_lot::Mutex<Option<PathBuf>>,
        }
        impl SpillRootAuthority for MutatingAuthority {
            fn store_id(&self) -> StoreId {
                self.base.store_id()
            }
            fn key_id(&self) -> [u8; 32] {
                self.base.key_id()
            }
            fn authenticate_marker(&self, message: &[u8]) -> io::Result<[u8; 32]> {
                self.base.authenticate_marker(message)
            }
            fn verify_marker(&self, message: &[u8], auth: &[u8; 32]) -> io::Result<bool> {
                if message.starts_with(b"GRAQLED1")
                    && let Some(path) = self.mutate.lock().take()
                {
                    fs::OpenOptions::new()
                        .write(true)
                        .open(path)?
                        .set_len(256 << 10)?;
                }
                self.base.verify_marker(message, auth)
            }
            fn record_provider(
                &self,
                identity: SpillQueryIdentity,
            ) -> io::Result<Arc<dyn SpillRecordProvider>> {
                self.base.record_provider(identity)
            }
        }
        let parent = TempDir::new().unwrap();
        let authority = Arc::new(MutatingAuthority {
            base: TestAuthority {
                store: StoreId::from_bytes([1; 32]).unwrap(),
                key: 7,
            },
            mutate: parking_lot::Mutex::new(None),
        });
        let root = SpillRoot::open(
            parent.path(),
            authority.clone(),
            SpillFrameLimits::format_max(),
            Arc::new(NoopSpillIo),
            SpillDiskQuota::new(64 << 20),
            Some(CONTROL_RESERVE + 8 * LEAF_RESERVE),
        )
        .unwrap();
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        *authority.mutate.lock() = Some(file.path().to_owned());
        assert!(file.finish_write().is_err());
        assert!(file.reader().is_err());
        file.close_and_delete().unwrap();
        manager.finish_query().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn root_quota_reconciles_allocated_blocks_beyond_logical_eof() {
        use super::super::quota::{ALLOCATION_UNIT, CONTROL_RESERVE, FILE_RESERVE, LEAF_RESERVE};
        use std::os::unix::fs::MetadataExt as _;
        for tight in [false, true] {
            let parent = TempDir::new().unwrap();
            let limit = if tight {
                CONTROL_RESERVE + 2 * LEAF_RESERVE + FILE_RESERVE + ALLOCATION_UNIT
            } else {
                CONTROL_RESERVE + 8 * LEAF_RESERVE
            };
            let root = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
            let manager = quota_manager(&root);
            let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
            file.write_sort_run_start(1, 0).unwrap();
            let before = quota_used(&root);
            {
                let allocation = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(file.path())
                    .unwrap();
                // KEEP_SIZE exercises physical capacity independent of EOF and
                // record framing. Close this fixture handle before cleanup.
                rustix::fs::fallocate(
                    &allocation,
                    rustix::fs::FallocateFlags::KEEP_SIZE,
                    0,
                    4 * ALLOCATION_UNIT,
                )
                .unwrap();
                assert!(allocation.metadata().unwrap().blocks() * 512 >= 4 * ALLOCATION_UNIT);
            }
            let result = file.finish_write();
            let allocated = fs::metadata(file.path()).unwrap().blocks() * 512;
            assert!(allocated >= 4 * ALLOCATION_UNIT);
            if tight {
                assert_eq!(result.unwrap_err().kind(), io::ErrorKind::QuotaExceeded);
                assert!(file.reader().is_err());
                assert!(
                    quota_used(&root) > limit,
                    "rejected publication persists the actual physical debt"
                );
                assert!(
                    quota_used(&root) >= CONTROL_RESERVE + LEAF_RESERVE + FILE_RESERVE + allocated
                );
                let reopened = quota_root(parent.path(), limit, Arc::new(NoopSpillIo));
                let (context, control) = query();
                assert_eq!(
                    reopened
                        .begin_query(context.query_id(), control.token())
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::QuotaExceeded
                );
            } else {
                result.unwrap();
                assert!(quota_used(&root) > before);
                assert!(
                    quota_used(&root) >= CONTROL_RESERVE + LEAF_RESERVE + FILE_RESERVE + allocated
                );
            }
            file.close_and_delete().unwrap();
            manager.finish_query().unwrap();
            assert_eq!(quota_used(&root), CONTROL_RESERVE);
        }
    }
    #[test]
    fn deferred_query_leaf_is_shared_across_concurrent_clones_and_profiled() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE};
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let control = QueryExecutionControl::new();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(1 << 20),
            &root,
            control.token(),
        )
        .unwrap();
        assert!(resources.has_spill_manager());
        assert!(resources.spill_manager().is_none());
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
        resources.enable_spill_profile_merge();
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let managers = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let resources = resources.clone();
                    let barrier = barrier.clone();
                    scope.spawn(move || {
                        barrier.wait();
                        resources.ensure_spill_manager().unwrap().unwrap().clone()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        for manager in &managers {
            assert!(Arc::ptr_eq(manager, &managers[0]));
            assert!(manager.profile_merge_enabled());
        }
        assert_eq!(quota_used(&root), CONTROL_RESERVE + LEAF_RESERVE);
        managers[0].finish_query().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn deferred_query_leaf_preserves_cancellation_and_quota_errors() {
        use super::super::quota::CONTROL_RESERVE;
        for cancelled in [false, true] {
            let parent = TempDir::new().unwrap();
            let root = quota_root(parent.path(), CONTROL_RESERVE, Arc::new(NoopSpillIo));
            let control = QueryExecutionControl::new();
            let resources = QueryResourceContext::with_spill_root(
                BufferManager::with_budget(1 << 20),
                &root,
                control.token(),
            )
            .unwrap();
            if cancelled {
                control.cancellation_handle().cancel();
            }
            let error = resources.ensure_spill_manager().unwrap_err();
            assert!(
                matches!(error, crate::execution::QueryResourceContextError::SpillAdmission { kind, .. }
                if kind == if cancelled { io::ErrorKind::Interrupted } else { io::ErrorKind::QuotaExceeded })
            );
            assert!(resources.spill_manager().is_none());
            assert_eq!(quota_used(&root), CONTROL_RESERVE);
        }
    }
    #[test]
    fn quota_reused_authentication_rejects_changed_headers_and_entry_bodies() {
        use super::super::{
            SpillIoOperation,
            quota::{CONTROL_RESERVE, LEAF_RESERVE},
        };
        use std::os::unix::fs::FileExt as _;
        struct Mutate(parking_lot::Mutex<Option<(PathBuf, u64)>>);
        impl SpillIo for Mutate {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::QuotaPublish
                    && let Some((path, offset)) = self.0.lock().take()
                {
                    let file = fs::OpenOptions::new().read(true).write(true).open(path)?;
                    let mut byte = [0];
                    file.read_exact_at(&mut byte, offset)?;
                    byte[0] ^= 1;
                    file.write_all_at(&byte, offset)?;
                }
                Ok(())
            }
        }
        for name in [".grafeo-spill-quota", ".grafeo-spill-quota.installing"] {
            for offset in [48, 144] {
                let parent = TempDir::new().unwrap();
                let mutation = Arc::new(Mutate(parking_lot::Mutex::new(None)));
                let root = quota_root(
                    parent.path(),
                    CONTROL_RESERVE + 4 * LEAF_RESERVE,
                    mutation.clone(),
                );
                let manager = quota_manager(&root);
                let ledger = root.namespace_path().join(".grafeo-spill-quota");
                let before = fs::read(&ledger).unwrap();
                *mutation.0.lock() = Some((root.namespace_path().join(name), offset));
                let (context, control) = query();
                assert_eq!(
                    root.begin_query(context.query_id(), control.token())
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidData
                );
                if name.ends_with(".installing") {
                    assert_eq!(fs::read(&ledger).unwrap(), before);
                } else {
                    assert_ne!(fs::read(&ledger).unwrap(), before);
                    fs::write(&ledger, &before).unwrap();
                }
                assert!(
                    !root
                        .namespace_path()
                        .join(".grafeo-spill-quota.installing")
                        .exists()
                );
                manager.finish_query().unwrap();
                assert_eq!(quota_used(&root), CONTROL_RESERVE);
            }
        }
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_quota_directory_credit_requires_owned_unlink_and_catalog_absence() {
        use super::super::quota::{CONTROL_RESERVE, LEAF_RESERVE, ReservationKey, RootReservation};
        let parent = TempDir::new().unwrap();
        let root = quota_root(
            parent.path(),
            CONTROL_RESERVE + LEAF_RESERVE,
            Arc::new(NoopSpillIo),
        );
        let control = QueryExecutionControl::new();
        let reservation = RootReservation::new(
            root.ledger.clone(),
            ReservationKey::leaf([7; 16]),
            control.token(),
        )
        .unwrap();
        let original = root.namespace_path().join("owned-probe");
        let moved = root.namespace_path().join("moved-probe");
        reservation.attempted();
        fs::create_dir(&original).unwrap();
        reservation
            .bind(fs::File::open(&original).unwrap(), root.directory())
            .unwrap();
        assert!(reservation.release_deleted().is_err());
        fs::rename(&original, &moved).unwrap();
        assert!(
            reservation.release_deleted().is_err(),
            "missing original path cannot release a moved directory"
        );
        reservation.confirm_directory_unlink().unwrap();
        assert!(
            reservation.release_deleted().is_err(),
            "an asserted unlink cannot release a live catalog object"
        );
        assert_eq!(quota_used(&root), CONTROL_RESERVE + LEAF_RESERVE);
        fs::remove_dir(&moved).unwrap();
        reservation.release_deleted().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn scavenge_crash_child() {
        let Some(parent) = std::env::var_os("GRAFEO_SCAVENGE_CHILD_PARENT") else {
            return;
        };
        let parent = Path::new(&parent);
        let root = quota_root(parent, 64 << 20, Arc::new(NoopSpillIo));
        let manager = quota_manager(&root);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        file.write_sort_row(b"crash-owned record").unwrap();
        file.finish_write().unwrap();
        fs::write(
            parent.join("scavenge-leaf"),
            manager.spill_dir().as_os_str().as_encoded_bytes(),
        )
        .unwrap();
        // No Rust destructors: the parent must acquire real released OS leases.
        std::process::exit(0);
    }

    fn abandoned_leaf(parent: &Path) -> PathBuf {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "execution::spill::root::tests::scavenge_crash_child",
                "--nocapture",
            ])
            .env("GRAFEO_SCAVENGE_CHILD_PARENT", parent)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        PathBuf::from(fs::read_to_string(parent.join("scavenge-leaf")).unwrap())
    }

    type ScavengeAction = Box<dyn FnOnce() -> io::Result<()> + Send>;

    struct ScavengeHook {
        stage: super::super::SpillIoOperation,
        action: parking_lot::Mutex<Option<ScavengeAction>>,
    }
    impl SpillIo for ScavengeHook {
        fn check(&self, stage: super::super::SpillIoOperation) -> io::Result<()> {
            if stage == self.stage
                && let Some(action) = self.action.lock().take()
            {
                action()?;
            }
            Ok(())
        }
    }
    fn scavenge_hook(
        stage: super::super::SpillIoOperation,
        action: impl FnOnce() -> io::Result<()> + Send + 'static,
    ) -> Arc<dyn SpillIo> {
        Arc::new(ScavengeHook {
            stage,
            action: parking_lot::Mutex::new(Some(Box::new(action))),
        })
    }

    #[test]
    fn scavenge_deletes_only_dead_authenticated_leaf_and_credits_its_debt() {
        use super::super::quota::CONTROL_RESERVE;
        let parent = TempDir::new().unwrap();
        let dead = abandoned_leaf(parent.path());
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let dead_charge = quota_used(&root) - CONTROL_RESERVE;
        let live = quota_manager(&root);
        let foreign = root.namespace_path().join("operator-notes");
        fs::write(&foreign, b"preserve me").unwrap();
        let before = quota_used(&root);
        let report = root.scavenge().unwrap();
        assert_eq!(
            (
                report.removed,
                report.deferred,
                report.foreign,
                report.invalid
            ),
            (1, 1, 1, 0)
        );
        assert!(!dead.exists());
        assert!(live.spill_dir().exists());
        assert_eq!(fs::read(foreign).unwrap(), b"preserve me");
        assert_eq!(report.reserved_bytes, before - dead_charge);
        live.finish_query().unwrap();
        assert_eq!(quota_used(&root), CONTROL_RESERVE);
    }

    #[test]
    fn scavenge_preserves_unknown_content_links_fifo_and_invalid_seals() {
        use std::os::unix::fs::symlink;
        for kind in [
            "unknown",
            "symlink",
            "hardlink",
            "fifo",
            "seal",
            "store",
            "short-marker",
            "long-marker",
            "leaf-identity",
        ] {
            let parent = TempDir::new().unwrap();
            let mut leaf = abandoned_leaf(parent.path());
            let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
            let mut file = fs::read_dir(&leaf)
                .unwrap()
                .map(|e| e.unwrap().path())
                .find(|p| p.extension().is_some_and(|e| e == "grsp"))
                .unwrap();
            match kind {
                "unknown" => fs::write(leaf.join("foreign"), b"unknown").unwrap(),
                "symlink" => {
                    fs::remove_file(&file).unwrap();
                    symlink(parent.path().join("scavenge-leaf"), &file).unwrap();
                }
                "hardlink" => fs::hard_link(&file, parent.path().join("alias")).unwrap(),
                "fifo" => {
                    fs::remove_file(&file).unwrap();
                    assert!(
                        std::process::Command::new("mkfifo")
                            .arg(&file)
                            .status()
                            .unwrap()
                            .success()
                    );
                }
                "short-marker" | "long-marker" => {
                    let path = leaf.join(".grafeo-spill-owner");
                    let mut marker = fs::read(&path).unwrap();
                    if kind == "short-marker" {
                        marker.pop();
                    } else {
                        marker.push(0);
                    }
                    fs::write(path, marker).unwrap();
                }
                "leaf-identity" => {
                    let mut name = leaf.file_name().unwrap().to_str().unwrap().to_owned();
                    let last = name.pop().unwrap();
                    name.push(if last == '0' { '1' } else { '0' });
                    let renamed = leaf.with_file_name(name);
                    fs::rename(&leaf, &renamed).unwrap();
                    file = renamed.join(file.file_name().unwrap());
                    leaf = renamed;
                }
                "seal" | "store" => {
                    let p = leaf.join(".grafeo-spill-owner");
                    let mut b = fs::read(&p).unwrap();
                    b[if kind == "seal" { 21 } else { 53 }] ^= 1;
                    fs::write(p, b).unwrap();
                }
                _ => unreachable!(),
            }
            let before = quota_used(&root);
            let report = root.scavenge().unwrap();
            assert_eq!(report.invalid, 1, "{kind}");
            assert_eq!(report.removed, 0, "{kind}");
            assert!(leaf.exists(), "{kind}");
            assert!(file.symlink_metadata().is_ok(), "{kind}");
            assert_eq!(report.reserved_bytes, before, "{kind}");
        }
    }

    #[test]
    fn scavenge_fifo_marker_preserves_debt_without_blocking_healthy_leaf() {
        use std::os::unix::fs::FileTypeExt as _;
        let parent = TempDir::new().unwrap();
        let bad = abandoned_leaf(parent.path());
        let good = abandoned_leaf(parent.path());
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let marker = bad.join(".grafeo-spill-owner");
        fs::remove_file(&marker).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&marker)
                .status()
                .unwrap()
                .success()
        );
        let before = quota_used(&root);
        let report = root.scavenge().unwrap();
        assert_eq!((report.invalid, report.removed), (1, 1));
        assert!(!good.exists());
        assert!(bad.exists());
        assert!(marker.symlink_metadata().unwrap().file_type().is_fifo());
        assert!(report.reserved_bytes < before);
        assert!(report.reserved_bytes > super::super::quota::CONTROL_RESERVE);
        assert_eq!(report.reserved_bytes, quota_used(&root));
    }

    #[test]
    fn scavenge_substitution_barriers_preserve_replacement_inodes() {
        use super::super::SpillIoOperation;
        for stage in [
            SpillIoOperation::ScavengeOpen,
            SpillIoOperation::ScavengeValidate,
            SpillIoOperation::ScavengeDelete,
        ] {
            let parent = TempDir::new().unwrap();
            let leaf = abandoned_leaf(parent.path());
            let target = match stage {
                SpillIoOperation::ScavengeOpen => leaf.clone(),
                SpillIoOperation::ScavengeValidate => leaf.join(".grafeo-spill-owner"),
                _ => fs::read_dir(&leaf)
                    .unwrap()
                    .map(|e| e.unwrap().path())
                    .find(|p| p.extension().is_some_and(|e| e == "grsp"))
                    .unwrap(),
            };
            let saved = parent.path().join("retained-original");
            let a = target.clone();
            let b = saved.clone();
            let hook = scavenge_hook(stage, move || {
                let is_dir = a.is_dir();
                let bytes = if is_dir { Vec::new() } else { fs::read(&a)? };
                fs::rename(&a, &b)?;
                if is_dir {
                    fs::create_dir(&a)?;
                } else {
                    fs::write(&a, bytes)?;
                }
                Ok(())
            });
            let root = quota_root(parent.path(), 64 << 20, hook);
            let before = quota_used(&root);
            let result = root.scavenge();
            if stage == SpillIoOperation::ScavengeOpen {
                assert_eq!(result.unwrap().invalid, 1);
            } else {
                assert!(result.is_err());
            }
            assert!(target.exists());
            assert!(saved.exists());
            assert_eq!(quota_used(&root), before);
        }
    }

    #[test]
    fn scavenge_failed_directory_delete_restores_seal_and_retries_after_restart() {
        use super::super::{SpillIoOperation, quota::CONTROL_RESERVE};
        let parent = TempDir::new().unwrap();
        let leaf = abandoned_leaf(parent.path());
        let seal = fs::read(leaf.join(".grafeo-spill-owner")).unwrap();
        let root = quota_root(
            parent.path(),
            64 << 20,
            scavenge_hook(SpillIoOperation::RemoveQueryDirectory, || {
                Err(io::Error::other("directory-delete fixture"))
            }),
        );
        let before = quota_used(&root);
        let error = root.scavenge().unwrap_err();
        assert!(error.to_string().contains("directory-delete fixture"));
        assert_eq!(fs::read(leaf.join(".grafeo-spill-owner")).unwrap(), seal);
        assert_eq!(quota_used(&root), before);
        drop(root);
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let report = root.scavenge().unwrap();
        assert_eq!(report.removed, 1);
        assert_eq!(report.reserved_bytes, CONTROL_RESERVE);
        assert!(!leaf.exists());
    }

    #[test]
    fn scavenge_failed_credit_retains_debt_after_restart() {
        use super::super::SpillIoOperation;
        let parent = TempDir::new().unwrap();
        let leaf = abandoned_leaf(parent.path());
        let root = quota_root(
            parent.path(),
            64 << 20,
            scavenge_hook(SpillIoOperation::QuotaRelease, || {
                Err(io::Error::other("credit fixture"))
            }),
        );
        let before = quota_used(&root);
        assert!(root.scavenge().is_err());
        assert!(!leaf.exists());
        drop(root);
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let report = root.scavenge().unwrap();
        assert_eq!(report.removed, 0);
        assert_eq!(report.reserved_bytes, before);
    }
    #[test]
    fn scavenge_root_replacement_preserves_both_namespaces() {
        use super::super::SpillIoOperation;
        let parent = TempDir::new().unwrap();
        let leaf = abandoned_leaf(parent.path());
        let namespace = leaf.parent().unwrap().to_path_buf();
        let saved = parent.path().join("original-root");
        let from = namespace.clone();
        let to = saved.clone();
        let root = quota_root(
            parent.path(),
            64 << 20,
            scavenge_hook(SpillIoOperation::ScavengeOpen, move || {
                fs::rename(&from, &to)?;
                fs::create_dir(&from)?;
                fs::write(from.join("foreign"), b"preserve")
            }),
        );
        assert!(root.scavenge().is_err());
        assert_eq!(fs::read(namespace.join("foreign")).unwrap(), b"preserve");
        assert!(
            saved
                .join(leaf.file_name().unwrap())
                .join(".grafeo-spill-owner")
                .exists()
        );
    }

    #[test]
    fn scavenge_marker_lease_alone_defers_dead_leaf() {
        let parent = TempDir::new().unwrap();
        let leaf = abandoned_leaf(parent.path());
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let marker = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(leaf.join(".grafeo-spill-owner"))
            .unwrap();
        marker.try_lock().unwrap();
        let before = quota_used(&root);
        assert_eq!(root.scavenge().unwrap().deferred, 1);
        assert_eq!(quota_used(&root), before);
        drop(marker);
        assert_eq!(root.scavenge().unwrap().removed, 1);
    }

    #[test]
    fn scavenge_oversized_leaf_is_preserved_before_any_deletion() {
        use std::os::unix::fs::PermissionsExt;
        let parent = TempDir::new().unwrap();
        let leaf = abandoned_leaf(parent.path());
        // All names are recognized; exceeding the fixed descriptor envelope
        // must preserve even the original valid artifact and its charge.
        for index in 0..257 {
            let path = leaf.join(format!("sort-run-{index:032x}.grsp"));
            fs::write(&path, b"fixture").unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
        }
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let before = quota_used(&root);
        let report = root.scavenge().unwrap();
        assert_eq!(report.deferred, 1);
        assert_eq!(report.reserved_bytes, before);
        assert_eq!(fs::read_dir(leaf).unwrap().count(), 259);
    }
    #[test]
    fn scavenge_replayed_live_marker_cannot_credit_original_leaf() {
        use std::os::unix::fs::PermissionsExt;
        let parent = TempDir::new().unwrap();
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let live = quota_manager(&root);
        let leaf = live.spill_dir().to_path_buf();
        let before = quota_used(&root);
        let marker = fs::read(leaf.join(".grafeo-spill-owner")).unwrap();
        let moved = parent.path().join("live-original");
        fs::rename(&leaf, &moved).unwrap();
        fs::create_dir(&leaf).unwrap();
        fs::set_permissions(&leaf, fs::Permissions::from_mode(0o700)).unwrap();
        let seal = leaf.join(".grafeo-spill-owner");
        fs::write(&seal, &marker).unwrap();
        fs::set_permissions(&seal, fs::Permissions::from_mode(0o600)).unwrap();
        let report = root.scavenge().unwrap();
        assert_eq!(
            report.invalid, 1,
            "a signed marker alone is not ownership of this inode"
        );
        assert_eq!(quota_used(&root), before);
        assert_eq!(fs::read(seal).unwrap(), marker);
        assert!(moved.exists());
        // Put the real live capability back so normal ownership cleanup remains valid.
        fs::remove_file(leaf.join(".grafeo-spill-owner")).unwrap();
        fs::remove_dir(&leaf).unwrap();
        fs::rename(moved, &leaf).unwrap();
        live.finish_query().unwrap();
    }

    #[test]
    fn scavenge_permission_mutation_rejects_before_artifact_delete() {
        use super::super::SpillIoOperation;
        use std::os::unix::fs::PermissionsExt;
        let parent = TempDir::new().unwrap();
        let leaf = abandoned_leaf(parent.path());
        let namespace = leaf.parent().unwrap().to_path_buf();
        let changed = namespace.clone();
        let root = quota_root(
            parent.path(),
            64 << 20,
            scavenge_hook(SpillIoOperation::ScavengeDelete, move || {
                fs::set_permissions(&changed, fs::Permissions::from_mode(0o777))
            }),
        );
        let before = quota_used(&root);
        assert!(root.scavenge().is_err());
        fs::set_permissions(namespace, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(
            leaf.is_dir(),
            "unsafe root must reject before deleting its leaf"
        );
        assert_eq!(
            fs::read_dir(leaf).unwrap().count(),
            2,
            "retain marker and artifact"
        );
        assert_eq!(quota_used(&root), before);
    }

    #[test]
    fn scavenge_unbound_version_two_marker_is_preserved_as_invalid() {
        let parent = TempDir::new().unwrap();
        let leaf = abandoned_leaf(parent.path());
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        let path = leaf.join(".grafeo-spill-owner");
        let mut bytes = fs::read(&path).unwrap();
        bytes[4] = 2;
        let mut message = [0; 133];
        message[..21].copy_from_slice(&bytes[..21]);
        message[21..].copy_from_slice(&bytes[53..]);
        bytes[21..53].copy_from_slice(&root.authority.authenticate_marker(&message).unwrap());
        fs::write(&path, &bytes).unwrap();
        let before = quota_used(&root);
        assert_eq!(root.scavenge().unwrap().invalid, 1);
        assert_eq!(quota_used(&root), before);
        assert_eq!(fs::read(path).unwrap(), bytes);
    }

    #[test]
    fn scavenge_root_enumeration_stops_at_fixed_work_bound() {
        let parent = TempDir::new().unwrap();
        let root = quota_root(parent.path(), 64 << 20, Arc::new(NoopSpillIo));
        for index in 0..4100 {
            fs::write(root.namespace_path().join(format!("foreign-{index}")), []).unwrap();
        }
        let report = root.scavenge().unwrap();
        assert!(report.truncated);
        assert!(report.inspected <= 4096);
        assert_eq!(report.removed, 0);
        assert_eq!(fs::read_dir(root.namespace_path()).unwrap().count(), 4103);
    }

    #[test]
    fn physical_profile_retains_peaks_and_failed_cleanup_debt() {
        use crate::execution::QueryResourceContext;
        use std::os::unix::fs::MetadataExt as _;
        let parent = TempDir::new().unwrap();
        let root = open(parent.path(), 1, 7).unwrap();
        let report = root.scavenge().unwrap();
        let (memory, control) = query();
        let resources = QueryResourceContext::with_spill_root(
            Arc::clone(memory.buffer_manager()),
            &root,
            control.token(),
        )
        .unwrap();
        assert_eq!(resources.profile_stats().spill_recovery, Some(report));
        assert!(resources.profile_stats().spill_physical.is_none());
        let manager = resources.ensure_spill_manager().unwrap().unwrap();
        let initial = manager.physical_stats().unwrap();
        assert!(initial.reserved_bytes > 0);
        assert_eq!(initial.reserved_bytes, initial.peak_reserved_bytes);
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        file.write_sort_row(b"measured published allocation")
            .unwrap();
        file.finish_write().unwrap();
        let first_allocation = fs::metadata(file.path()).unwrap().blocks() * 512;
        assert_eq!(
            manager.physical_stats().unwrap().observed_file_bytes,
            first_allocation
        );
        let mut second = manager.create_file(SpillFileRole::SortRun).unwrap();
        second.write_sort_run_start(1, 1).unwrap();
        second
            .write_sort_row(b"independently measured second file")
            .unwrap();
        second.finish_write().unwrap();
        let second_allocation = fs::metadata(second.path()).unwrap().blocks() * 512;
        let live = manager.physical_stats().unwrap();
        assert_eq!(
            live.observed_file_bytes,
            first_allocation + second_allocation
        );
        assert!(live.reserved_bytes >= live.observed_file_bytes);
        assert!(live.observed_file_bytes > 0);
        assert_eq!(live.observed_file_bytes, live.peak_observed_file_bytes);
        assert!(!live.cleanup_failed);
        assert!(!live.reservation_uncertain);
        let foreign = manager.spill_dir().join("preserve-me");
        fs::write(&foreign, b"foreign content").unwrap();
        drop(file);
        assert_eq!(
            manager.physical_stats().unwrap().observed_file_bytes,
            second_allocation
        );
        drop(second);
        assert!(manager.finish_query().is_err());
        let failed = resources.profile_stats().spill_physical.unwrap();
        assert!(failed.cleanup_failed);
        assert_eq!(failed.cleanup_debt_bytes, failed.reserved_bytes);
        assert!(failed.cleanup_debt_bytes > 0);
        assert_eq!(failed.observed_file_bytes, 0);
        assert_eq!(
            failed.peak_observed_file_bytes,
            live.peak_observed_file_bytes
        );
        assert_eq!(fs::read(&foreign).unwrap(), b"foreign content");
        fs::remove_file(foreign).unwrap();
        manager.finish_query().unwrap();
        let closed = resources.profile_stats().spill_physical.unwrap();
        assert_eq!(closed.reserved_bytes, 0);
        assert_eq!(closed.cleanup_debt_bytes, 0);
        assert!(!closed.cleanup_failed);
        assert_eq!(closed.peak_reserved_bytes, live.peak_reserved_bytes);
        assert_eq!(
            closed.peak_observed_file_bytes,
            live.peak_observed_file_bytes
        );
        assert_eq!(resources.profile_stats().spill_recovery, Some(report));
    }

    #[test]
    fn physical_profile_does_not_wait_for_inflight_quota_io() {
        use super::super::SpillIoOperation;
        use std::sync::{
            Barrier,
            atomic::{AtomicBool, Ordering},
            mpsc,
        };
        struct HoldQuota {
            armed: AtomicBool,
            entered: Barrier,
            release: Barrier,
        }
        impl SpillIo for HoldQuota {
            fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
                if operation == SpillIoOperation::QuotaWrite
                    && self.armed.swap(false, Ordering::SeqCst)
                {
                    self.entered.wait();
                    self.release.wait();
                }
                Ok(())
            }
        }
        let parent = TempDir::new().unwrap();
        let hook = Arc::new(HoldQuota {
            armed: AtomicBool::new(false),
            entered: Barrier::new(2),
            release: Barrier::new(2),
        });
        let root = quota_root(parent.path(), 16 << 20, hook.clone());
        let (memory, control) = query();
        let resources = crate::execution::QueryResourceContext::with_spill_root(
            Arc::clone(memory.buffer_manager()),
            &root,
            control.token(),
        )
        .unwrap();
        let manager = Arc::clone(resources.ensure_spill_manager().unwrap().unwrap());
        hook.armed.store(true, Ordering::SeqCst);
        let owner = Arc::clone(&manager);
        let writer = std::thread::spawn(move || owner.create_file(SpillFileRole::SortRun));
        hook.entered.wait();
        let sampler = resources.clone();
        let (sent, received) = mpsc::channel();
        let sample =
            std::thread::spawn(move || sent.send(sampler.profile_stats().spill_physical).unwrap());
        let result = received.recv_timeout(std::time::Duration::from_secs(2));
        // Always unblock the fixture before asserting, including the RED case.
        hook.release.wait();
        drop(writer.join().unwrap().unwrap());
        sample.join().unwrap();
        assert_eq!(
            result.unwrap(),
            None,
            "busy ownership must be reported unavailable"
        );
        assert!(manager.physical_stats().is_some());
        manager.finish_query().unwrap();
    }
}
