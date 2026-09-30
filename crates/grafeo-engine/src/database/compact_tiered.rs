//! Disk-backed tier for the compact columnar base.
//!
//! Wraps a [`CompactStore`] in a two-state machine:
//!
//! - `InMemory`: the store lives entirely on the heap (default after
//!   [`compact()`](super::GrafeoDB::compact)).
//! - `OnDisk`: the store has been serialized to a file, mmapped, and
//!   re-deserialized. The mmap keeps the page cache populated so OS-level
//!   paging can reclaim cold pages under memory pressure without a hard
//!   error on reads.
//!
//! The wrapper is additive: the inner [`CompactStore`] is always a valid
//! `Arc<CompactStore>`, so [`LayeredStore`](grafeo_core::graph::compact::layered::LayeredStore)
//! keeps serving reads transparently across tier transitions.
//!
//! # Lifecycle
//!
//! ```text
//! new_in_memory(store)  -> InMemory(Arc<CompactStore>)
//!        |
//!        | persist_to_mmap(exact caller path)
//!        v
//!     OnDisk(PersistentExact, path, Mmap, Arc<CompactStore>)
//!        ^
//!        |
//!        | engine spill(generation root)
//!        |
//!     OnDisk(EngineEphemeral, unique path, Mmap, Arc<CompactStore>)
//!        |
//!        | reload_to_ram()
//!        v
//!     InMemory(Arc<CompactStore>)
//! ```
//!
//! # Memory accounting
//!
//! The bytes freed by a tier transition depend on whether the caller drops
//! the old in-memory store. [`persist_to_mmap`](CompactStoreTiered::persist_to_mmap)
//! consumes the old `Arc<CompactStore>` and replaces it with a fresh one
//! deserialized from mmap bytes. If the caller kept another `Arc` around
//! (e.g. in a [`LayeredStore`](grafeo_core::graph::compact::layered::LayeredStore)),
//! that clone still keeps the old allocation live; callers under memory
//! pressure should route reads through
//! [`store()`](CompactStoreTiered::store) and hold the tiered wrapper, not
//! raw CompactStore clones.
//!
//! # Feature flags
//!
//! Compiled only when both `compact-store` and `mmap` are enabled.

#![cfg(all(feature = "compact-store", feature = "mmap"))]

use std::ffi::OsString;
use std::fs::{File, TryLockError};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use grafeo_common::grafeo_warn;
use grafeo_common::storage::section::Section;
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::compact::CompactStore;
use grafeo_core::graph::compact::section::CompactStoreSection;
use memmap2::Mmap;
use parking_lot::{Mutex, RwLock};

/// Two-state disk-backed wrapper around a [`CompactStore`].
pub struct CompactStoreTiered {
    state: RwLock<PublishedTierState>,
    /// The live process lease used for engine-owned immutable generations.
    ///
    /// A wrapper normally uses one spill root. Retaining the owner here avoids
    /// allocating a lease for each generation; mappings also retain it, so a
    /// changed root cannot release the old namespace while readers still exist.
    generation_owner: Mutex<Option<Arc<GenerationOwner>>>,
    #[cfg(test)]
    fail_before_generation_publish: std::sync::atomic::AtomicBool,
}

/// One coherent observation of tier metadata and the exact store generation
/// that metadata describes.
pub(super) struct TierSnapshot {
    pub(super) is_on_disk: bool,
    pub(super) path: Option<PathBuf>,
}

/// A fully serialized and decoded compact generation that has not yet been
/// made visible through its owning [`LayeredStore`].
///
/// The backing file, when present, has an immutable generation-unique name and
/// is already mmap/decode-ready. Dropping an unpublished plan releases its
/// mapping and retires that engine-owned file. Publication itself is therefore
/// only an infallible in-memory state move performed under the Layered
/// generation cut.
pub(super) struct PreparedTierGeneration {
    state: Option<TierState>,
}

/// Owns the exact tier state displaced by a reversible metadata publication.
///
/// Publication has already moved the live metadata. After the outer Layered
/// cut drains, [`Self::commit`](PublishedTierGeneration::commit) explicitly
/// retires the displaced state; rollback instead passes the token to
/// [`CompactStoreTiered::restore_published_reversibly`]. The token retains
/// every prior mmap slice and backing lease, so neither path reopens mutable
/// filesystem state.
#[must_use = "explicitly commit after outer guards drain, or restore this exact publication"]
pub(super) struct PublishedTierGeneration {
    previous: Option<TierState>,
    published_store: Arc<CompactStore>,
    published_revision: u64,
}

/// Owns tier state made unreachable by an attempted rollback.
///
/// Layered transport publication carries this value out of every generation,
/// LPG, and routing guard before it is dropped, so unmapping and ephemeral-file
/// cleanup never run on an infallible publication cut.
pub(super) struct TierRollbackRetirement {
    _state: TierRollbackState,
}

enum TierRollbackState {
    Restored { _retired: TierState },
    Rejected { _stale: PublishedTierGeneration },
}

struct PublishedTierState {
    revision: u64,
    tier: TierState,
}

impl PreparedTierGeneration {
    #[must_use]
    pub(super) fn store(&self) -> Arc<CompactStore> {
        match self.state.as_ref().expect("prepared tier state consumed") {
            TierState::InMemory(store) => Arc::clone(store),
            TierState::OnDisk { store, .. } => Arc::clone(store),
        }
    }
}

impl PublishedTierGeneration {
    /// Finalizes an already published generation and retires displaced state.
    ///
    /// Call this only after the owning Layered generation transition has
    /// released its publication and mutation guards. Any mmap file remains
    /// leased by readers that captured the displaced store.
    pub(super) fn commit(mut self) {
        // The Layered caller reaches commit only after its reader-visible base
        // swap has completed and the previous base guard has drained.
        compact_crash_test_pause("base-publication");
        drop(self.previous.take());
    }
}

impl TierRollbackRetirement {
    #[cfg(test)]
    fn was_restored(&self) -> bool {
        matches!(&self._state, TierRollbackState::Restored { .. })
    }
}

enum TierState {
    /// Store lives entirely on the heap.
    InMemory(Arc<CompactStore>),
    /// Store is backed by a mmap'd file. Phase 3c: the entire mmap is
    /// wrapped as a refcounted [`Bytes`] via [`Bytes::from_owner`], and
    /// every column codec inside `store` holds a `Bytes::slice(range)`
    /// view into it — so column data is read directly from mmap'd
    /// memory with zero copies. The `Bytes` here keeps the Mmap alive
    /// for the lifetime of the tier state; dropping it after `store`
    /// drops releases the mapping.
    OnDisk {
        backing: DiskBacking,
        _mmap_bytes: Bytes,
        store: Arc<CompactStore>,
    },
}

/// Files supplied by callers and cache generations owned by the engine have
/// deliberately different lifetime rules. Keeping that distinction in the
/// type prevents a future tier transition from deleting a caller's durable
/// file or treating it as a cache-generation naming root.
enum DiskBacking {
    /// The caller chose this exact durable path. Grafeo never removes it.
    PersistentExact {
        caller_path: PathBuf,
        io_path: PathBuf,
        identity: FileIdentity,
    },
    /// The engine allocated a unique immutable cache generation beside `root`.
    /// Its file retires after the final mmap-backed reader drains.
    EngineEphemeral {
        path: PathBuf,
        identity: FileIdentity,
        /// Keeps the namespace lease live for as long as tier metadata names
        /// this generation, independently of mmap-derived readers.
        _owner: Arc<GenerationOwner>,
    },
}

impl DiskBacking {
    fn path(&self) -> &Path {
        match self {
            Self::PersistentExact { caller_path, .. } => caller_path,
            Self::EngineEphemeral { path, .. } => path,
        }
    }

    fn io_path(&self) -> &Path {
        match self {
            Self::PersistentExact { io_path, .. } => io_path,
            Self::EngineEphemeral { path, .. } => path,
        }
    }

    fn identity(&self) -> FileIdentity {
        match self {
            Self::PersistentExact { identity, .. } | Self::EngineEphemeral { identity, .. } => {
                *identity
            }
        }
    }

    #[cfg(test)]
    fn generation_root(&self) -> Option<&Path> {
        match self {
            Self::PersistentExact { .. } => None,
            Self::EngineEphemeral { _owner, .. } => Some(&_owner.generation_root),
        }
    }
}

impl TierState {
    fn store(&self) -> &Arc<CompactStore> {
        match self {
            Self::InMemory(store) | Self::OnDisk { store, .. } => store,
        }
    }

    fn snapshot(&self) -> TierSnapshot {
        match self {
            Self::InMemory(_) => TierSnapshot {
                is_on_disk: false,
                path: None,
            },
            Self::OnDisk { backing, .. } => TierSnapshot {
                is_on_disk: true,
                path: Some(backing.path().to_path_buf()),
            },
        }
    }
}

impl PublishedTierState {
    fn next_revision(&self) -> u64 {
        self.revision
            .checked_add(1)
            .expect("compact tier publication revision exhausted")
    }
}

/// One path captured at an operation boundary.
///
/// `io_path` uses the canonical physical parent plus the original final
/// component, freezing current-directory and ancestor-symlink interpretation.
/// It does not claim capability-style protection from a hostile process that
/// concurrently renames or replaces directory entries after this resolution.
struct ResolvedPath {
    caller_path: PathBuf,
    io_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FileIdentity {
    #[cfg(unix)]
    Unix { device: u64, inode: u64 },
    #[cfg(windows)]
    Windows {
        volume_serial: Option<u32>,
        file_index: Option<u64>,
    },
    #[cfg(not(any(unix, windows)))]
    Unsupported,
}

/// Owns a read-only mapping and, for engine-created cache generations, attempts
/// identity-checked removal only after the final mmap-backed `Bytes` slice
/// drains. This is portable to Windows, where a mapped file cannot be renamed
/// or unlinked while any view remains live.
struct MappedFileOwner {
    mmap: Option<Mmap>,
    cleanup: MappedFileCleanup,
}

enum MappedFileCleanup {
    /// Caller-owned persistent files always outlive a mapping.
    Retain,
    /// Engine cache files are removed only after the mapping and every derived
    /// `Bytes` slice have drained.
    RemoveWhenUnmapped {
        path: PathBuf,
        expected_identity: Option<FileIdentity>,
        /// Prevents a stale-process scavenger from considering the namespace
        /// dead while any mmap slice from it remains live.
        owner: Arc<GenerationOwner>,
    },
}

impl AsRef<[u8]> for MappedFileOwner {
    fn as_ref(&self) -> &[u8] {
        self.mmap
            .as_ref()
            .expect("mapped file owner already dropped")
    }
}

impl Drop for MappedFileOwner {
    fn drop(&mut self) {
        drop(self.mmap.take());
        let cleanup = std::mem::replace(&mut self.cleanup, MappedFileCleanup::Retain);
        match cleanup {
            MappedFileCleanup::Retain => {}
            MappedFileCleanup::RemoveWhenUnmapped {
                path,
                expected_identity: Some(expected_identity),
                owner,
            } => {
                let _ = remove_owned_file_if_unchanged(
                    &path,
                    expected_identity,
                    "retired mmap generation",
                );
                drop(owner);
            }
            MappedFileCleanup::RemoveWhenUnmapped {
                path,
                expected_identity: None,
                owner,
            } => {
                grafeo_warn!(
                    "refusing to remove compact retired mmap generation {} without a bound file identity",
                    path.display()
                );
                drop(owner);
            }
        }
    }
}

/// Removes a unique unpublished candidate on every early-return path.
struct UnpublishedFile {
    path: Option<PathBuf>,
    expected_identity: FileIdentity,
    /// Engine-generation candidates keep their namespace lease alive while
    /// fallible write/map/decode work is in progress. Persistent staging files
    /// do not belong to an owner namespace.
    _owner: Option<Arc<GenerationOwner>>,
}

impl UnpublishedFile {
    fn path(&self) -> &Path {
        self.path
            .as_deref()
            .expect("unpublished file ownership already transferred")
    }

    fn disarm(&mut self) {
        self.path = None;
    }

    fn expected_identity(&self) -> FileIdentity {
        self.expected_identity
    }
}

impl Drop for UnpublishedFile {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = remove_owned_file_if_unchanged(
                &path,
                self.expected_identity,
                "unpublished candidate",
            );
        }
    }
}

static NEXT_UNIQUE_FILE: AtomicU64 = AtomicU64::new(1);

const OWNER_LEASE_MAGIC: &str = "grafeo-compact-owner-v1";
const OWNER_ROLE: &str = "grafeo-owner";
const GENERATION_ROLE: &str = "grafeo-gen";

/// Debug-build process-crash failpoint used by the integration matrix. The
/// child publishes a durable ready marker and parks until its parent sends a
/// real SIGKILL. Release builds and ordinary debug runs pay only an environment
/// lookup, and no failpoint is reachable unless both variables are present.
fn compact_crash_test_pause(point: &str) {
    #[cfg(debug_assertions)]
    {
        const POINT_ENV: &str = "GRAFEO_COMPACT_CRASH_POINT";
        const READY_ENV: &str = "GRAFEO_COMPACT_CRASH_READY";
        if std::env::var(POINT_ENV).as_deref() != Ok(point) {
            return;
        }
        let Some(ready_path) = std::env::var_os(READY_ENV).map(PathBuf::from) else {
            return;
        };
        if let Some(parent) = ready_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut ready) = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&ready_path)
        {
            let _ = ready.write_all(point.as_bytes());
            let _ = ready.sync_all();
            let _ = sync_parent_directory(&ready_path);
        }
        loop {
            std::thread::park();
        }
    }
    #[cfg(not(debug_assertions))]
    let _ = point;
}

/// One process-scoped namespace for immutable spill generations rooted at one
/// resolved path. The open lease file remains exclusively locked until every
/// wrapper/mapping/candidate retaining this owner has drained. Kernel lock
/// release on process death is the sole liveness oracle used by scavenging.
struct GenerationOwner {
    generation_root: PathBuf,
    token: String,
    lease_path: PathBuf,
    lease_identity: FileIdentity,
    lease_file: Option<File>,
    next_generation: AtomicU64,
}

impl Drop for GenerationOwner {
    fn drop(&mut self) {
        // Keep the lock held while checking that no generation remains. If
        // cleanup refused a hostile replacement, the lease deliberately stays
        // behind so a future allocator never loses the ownership boundary.
        let namespace_empty = namespace_entries(&self.generation_root, &self.token)
            .is_ok_and(|entries| entries.is_empty());
        if let Some(file) = self.lease_file.take() {
            if let Err(error) = file.unlock() {
                grafeo_warn!(
                    "failed to unlock compact owner lease {}: {error}",
                    self.lease_path.display()
                );
            }
            drop(file);
        }
        if namespace_empty {
            let _ = remove_owned_file_if_unchanged(
                &self.lease_path,
                self.lease_identity,
                "owner lease",
            );
        }
    }
}

impl CompactStoreTiered {
    /// Creates a tiered wrapper starting in the in-memory state.
    #[must_use]
    pub fn new_in_memory(store: Arc<CompactStore>) -> Self {
        Self {
            state: RwLock::new(PublishedTierState {
                revision: 0,
                tier: TierState::InMemory(store),
            }),
            generation_owner: Mutex::new(None),
            #[cfg(test)]
            fail_before_generation_publish: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Returns the current store, whether in-memory or mmap-backed.
    ///
    /// Cheap `Arc::clone`, safe to call on the query hot path.
    #[must_use]
    pub fn store(&self) -> Arc<CompactStore> {
        Arc::clone(self.state.read().tier.store())
    }

    /// Returns `true` when the backing file is mmap'd.
    #[must_use]
    pub fn is_on_disk(&self) -> bool {
        matches!(&self.state.read().tier, TierState::OnDisk { .. })
    }

    /// Returns the exact immutable backing path, if mmapped.
    ///
    /// Public exact-path persistence returns the caller's path. Engine-owned
    /// cache spill returns its unique immutable generation rather than the
    /// generation root.
    #[must_use]
    pub fn path(&self) -> Option<PathBuf> {
        match &self.state.read().tier {
            TierState::OnDisk { backing, .. } => Some(backing.path().to_path_buf()),
            TierState::InMemory(_) => None,
        }
    }

    /// Returns the engine-owned filename root used to allocate immutable
    /// successor generations.
    ///
    /// Caller-owned exact mappings deliberately return `None`: a persistent
    /// file is neither an engine cache root nor eligible for automatic
    /// rotation/deletion.
    #[cfg(test)]
    #[must_use]
    pub(super) fn generation_root(&self) -> Option<PathBuf> {
        match &self.state.read().tier {
            TierState::OnDisk { backing, .. } => backing.generation_root().map(Path::to_path_buf),
            TierState::InMemory(_) => None,
        }
    }

    /// Returns tier metadata only when it describes `expected` exactly.
    ///
    /// Database views call this while holding the Layered publication read cut,
    /// so the matching base cannot advance between the two observations.
    #[must_use]
    pub(super) fn snapshot_for_store(&self, expected: &Arc<CompactStore>) -> Option<TierSnapshot> {
        let state = self.state.read();
        Arc::ptr_eq(state.tier.store(), expected).then(|| state.tier.snapshot())
    }

    /// Serializes the current store to `path` without switching tier state.
    ///
    /// Returns the number of bytes written. Useful for checkpoint flows
    /// that want a snapshot on disk without giving up the RAM copy.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization or the file write fails, or
    /// if `path` names (directly, through a symlink, or through a hard link) a
    /// file backing the live tier. Replacing a mapped backing would make the
    /// path and the bytes serving reads describe different generations.
    pub fn persist(&self, path: &Path) -> Result<usize> {
        let target = resolve_output_path(path)?;
        // Keep the read guard through installation so no tier transition can
        // make `target` live after the alias check and before the rename.
        let state = self.state.read();
        reject_active_target(&state.tier, &target, false)?;
        let store = Arc::clone(state.tier.store());
        let section = CompactStoreSection::new(store);
        let bytes = section.serialize()?;
        write_atomically(&target.io_path, &bytes)?;
        Ok(bytes.len())
    }

    /// Serializes the store to exactly `path`, mmaps that caller-owned file, and
    /// swaps into the `OnDisk` state, returning the number of bytes written.
    /// [`Self::path`] returns exactly the path supplied here. The file remains
    /// in place after reload or drop; only explicit caller action removes it.
    ///
    /// Repeating this call for the exact persistent path already mapped by this
    /// immutable wrapper is a no-op after verifying that the path still names
    /// the mapped file. A missing, replaced, symlink-aliased, or hard-linked
    /// active target is rejected. Replacing an unrelated existing target is
    /// attempted with an atomic rename; platforms that cannot perform that
    /// replacement return an error without changing the installed tier state.
    ///
    /// The candidate is fully decoded before rename. If opening or mapping the
    /// installed file then fails for resource reasons, this method returns an
    /// error and retains the prior tier state, while `path` still contains a
    /// complete validated compact image. Likewise, a parent-directory sync
    /// failure is reported after the complete image has been installed.
    ///
    /// After this call, the wrapper holds a fresh `Arc<CompactStore>`
    /// deserialized from the mmap. The caller's previous `Arc<CompactStore>`
    /// (obtained via an earlier `store()` call) is still valid but stale
    /// relative to future reads routed through this wrapper.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization, the file write, or the
    /// subsequent mmap + deserialize cycle fails.
    pub fn persist_to_mmap(&self, path: &Path) -> Result<usize> {
        let target = resolve_output_path(path)?;
        // Retain the state write lock across capture, serialization, and
        // install. Otherwise a concurrent generation install can land between
        // `store()` and the final assignment and be overwritten by stale mmap
        // bytes.
        let mut state = self.state.write();
        let active_target = reject_active_target(&state.tier, &target, true)?;
        let store = Arc::clone(state.tier.store());
        let section = CompactStoreSection::new(store);
        let bytes = section.serialize()?;
        let written = bytes.len();

        if active_target == ActiveTarget::VerifiedPersistentExact {
            return Ok(written);
        }

        // Validate independently of the destination before its atomic path
        // installation. A post-rename mmap resource failure can still leave a
        // complete validated file at `path`, but never a corrupt partial image.
        deserialize_store(
            Bytes::from(bytes.clone()),
            "serialized exact-path candidate",
        )?;
        write_atomically(&target.io_path, &bytes)?;
        let (mmap_bytes, store, identity) =
            open_and_deserialize(&target.io_path, MappedFileCleanup::Retain)?;
        let next_revision = state.next_revision();
        let previous = std::mem::replace(
            &mut state.tier,
            TierState::OnDisk {
                backing: DiskBacking::PersistentExact {
                    caller_path: target.caller_path,
                    io_path: target.io_path,
                    identity,
                },
                _mmap_bytes: mmap_bytes,
                store,
            },
        );
        state.revision = next_revision;
        drop(state);
        drop(previous);
        Ok(written)
    }

    /// Serializes the wrapper's current store into an engine-owned immutable
    /// mmap generation rooted at `generation_root` and publishes it directly.
    ///
    /// This is only for maintenance fallback paths that no longer have an
    /// owning `LayeredStore`. Normal database spill uses the prepared
    /// generation API under the Layered generation cut.
    #[allow(
        dead_code,
        reason = "retained for maintenance fallbacks without an owning LayeredStore"
    )]
    pub(super) fn persist_to_ephemeral_mmap(&self, generation_root: &Path) -> Result<usize> {
        let generation_root = resolve_output_path(generation_root)?;
        let owner = self.generation_owner(&generation_root.io_path)?;
        let mut state = self.state.write();
        let section = CompactStoreSection::new(Arc::clone(state.tier.store()));
        let bytes = section.serialize()?;
        let (generation_path, identity, mmap_bytes, store) =
            create_mapped_generation(&owner, &bytes)?;
        let written = bytes.len();
        let next_revision = state.next_revision();
        let previous = std::mem::replace(
            &mut state.tier,
            TierState::OnDisk {
                backing: DiskBacking::EngineEphemeral {
                    path: generation_path,
                    identity,
                    _owner: Arc::clone(&owner),
                },
                _mmap_bytes: mmap_bytes,
                store,
            },
        );
        state.revision = next_revision;
        drop(state);
        drop(previous);
        Ok(written)
    }

    /// Serializes and validates one caller-pinned compact generation as an
    /// engine-owned ephemeral mmap without publishing it.
    ///
    /// `LayeredStore` callers invoke this inside their generation write cut.
    /// The immutable generation file is created with `create_new`, closed,
    /// reopened read-only, mmap'd, and fully decoded before this returns. The
    /// returned plan can then be installed infallibly alongside the sole base
    /// `ArcSwap`; dropping it instead leaves the current tier untouched.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization, writing, mmap, or decode
    /// fails. The prior tier state remains installed on failure and no shared
    /// final path is ever renamed while mapped.
    pub(super) fn prepare_generation_to_mmap(
        &self,
        generation: Arc<CompactStore>,
        generation_root: &Path,
    ) -> Result<PreparedTierGeneration> {
        let generation_root = resolve_output_path(generation_root)?;
        let owner = self.generation_owner(&generation_root.io_path)?;
        let section = CompactStoreSection::new(generation);
        let bytes = section.serialize()?;
        let (generation_path, identity, mmap_bytes, store) =
            create_mapped_generation(&owner, &bytes)?;
        #[cfg(test)]
        if self
            .fail_before_generation_publish
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            drop(store);
            drop(mmap_bytes);
            return Err(Error::Internal(
                "injected compact generation failure before in-memory publication".to_string(),
            ));
        }
        Ok(PreparedTierGeneration {
            state: Some(TierState::OnDisk {
                backing: DiskBacking::EngineEphemeral {
                    path: generation_path,
                    identity,
                    _owner: Arc::clone(&owner),
                },
                _mmap_bytes: mmap_bytes,
                store,
            }),
        })
    }

    #[cfg(test)]
    pub(super) fn fail_next_generation_publish_for_test(&self) {
        self.fail_before_generation_publish
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Prepares an exact compact generation in RAM for publication by the
    /// owning `LayeredStore`.
    #[must_use]
    pub(super) fn prepare_generation_in_memory(
        &self,
        generation: Arc<CompactStore>,
    ) -> PreparedTierGeneration {
        PreparedTierGeneration {
            state: Some(TierState::InMemory(generation)),
        }
    }

    /// Installs already validated tier metadata under the same Layered
    /// generation cut that publishes `prepared.store()`.
    ///
    /// This operation performs no I/O, allocation, serialization, or decode;
    /// taking an uncontended internal lock and moving an owned enum are
    /// infallible. Public graph views source data from the Layered base rather
    /// than this metadata, so there is only one graph-generation authority.
    #[cfg(test)]
    pub(super) fn publish_prepared(&self, prepared: PreparedTierGeneration) {
        self.publish_prepared_reversibly(prepared).commit();
    }

    /// Installs prepared metadata while retaining the exact displaced state in
    /// an owned rollback token.
    ///
    /// The Layered caller invokes this while its generation publication cut is
    /// held, then carries the returned token out of that cut. Only after the
    /// outer guards drain may it call [`PublishedTierGeneration::commit`]. A
    /// rollback performed on the cut uses [`Self::restore_published_reversibly`]
    /// and likewise carries that method's retirement token out before drop. No
    /// I/O, serialization, decode, or path lookup occurs here.
    pub(super) fn publish_prepared_reversibly(
        &self,
        mut prepared: PreparedTierGeneration,
    ) -> PublishedTierGeneration {
        let next = prepared
            .state
            .take()
            .expect("prepared tier generation already published");
        let published_store = Arc::clone(next.store());
        let mut state = self.state.write();
        let published_revision = state.next_revision();
        let previous = std::mem::replace(&mut state.tier, next);
        state.revision = published_revision;
        drop(state);
        // At this milestone tier metadata names the immutable candidate while
        // the outer Layered generation cut still owns reader publication.
        compact_crash_test_pause("tier-publication");
        PublishedTierGeneration {
            previous: Some(previous),
            published_store,
            published_revision,
        }
    }

    /// Restores a reversibly displaced tier state without panicking.
    ///
    /// Returns `false` only if the live tier has advanced past the token's
    /// published generation, which indicates a violated caller contract. The
    /// normal transport rollback path holds the Layered generation write cut,
    /// so the comparison is invariant-backed and restoration returns `true`.
    /// Candidate mmap storage is retired only after any reader that captured it
    /// during publication releases its derived `Bytes` slices.
    #[cfg(test)]
    pub(super) fn restore_published(&self, published: PublishedTierGeneration) -> bool {
        self.restore_published_reversibly(published).was_restored()
    }

    /// Attempts rollback while returning every displaced state for retirement
    /// by the caller after its outer publication guards have drained.
    pub(super) fn restore_published_reversibly(
        &self,
        mut published: PublishedTierGeneration,
    ) -> TierRollbackRetirement {
        let mut state = self.state.write();
        if state.revision != published.published_revision
            || !Arc::ptr_eq(state.tier.store(), &published.published_store)
        {
            grafeo_warn!(
                "refusing stale compact tier rollback because the live generation advanced"
            );
            return TierRollbackRetirement {
                _state: TierRollbackState::Rejected { _stale: published },
            };
        }
        let Some(previous) = published.previous.take() else {
            grafeo_warn!("refusing already-consumed compact tier rollback token");
            return TierRollbackRetirement {
                _state: TierRollbackState::Rejected { _stale: published },
            };
        };
        let next_revision = state.next_revision();
        let retired_candidate = std::mem::replace(&mut state.tier, previous);
        state.revision = next_revision;
        drop(state);
        TierRollbackRetirement {
            _state: TierRollbackState::Restored {
                _retired: retired_candidate,
            },
        }
    }

    /// Opens an existing on-disk store via mmap, without writing.
    ///
    /// Returns a tiered wrapper starting in the `OnDisk` state.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if the file cannot be opened, mmapped,
    /// or deserialized into a valid `CompactStore`.
    pub fn open_mmap(path: &Path) -> Result<Self> {
        let target = resolve_existing_path(path)?;
        let (mmap_bytes, store, identity) =
            open_and_deserialize(&target.io_path, MappedFileCleanup::Retain)?;
        Ok(Self {
            state: RwLock::new(PublishedTierState {
                revision: 0,
                tier: TierState::OnDisk {
                    backing: DiskBacking::PersistentExact {
                        caller_path: target.caller_path,
                        io_path: target.io_path,
                        identity,
                    },
                    _mmap_bytes: mmap_bytes,
                    store,
                },
            }),
            generation_owner: Mutex::new(None),
            #[cfg(test)]
            fail_before_generation_publish: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Reloads the store into a heap-owning `InMemory` tier and drops the mmap.
    /// Caller-owned exact files remain in place; an engine-owned ephemeral
    /// generation retires after its final mmap-backed reader drains.
    ///
    /// Naively re-tagging the existing `Arc<CompactStore>` as `InMemory`
    /// would still leave column codec storage referencing the mmap-backed
    /// `Bytes` produced by the original open path: the data would continue
    /// to be served from the OS page cache and the `Mmap` would stay alive
    /// through the codec slices. To make the tier label truthful we
    /// re-serialize the live store and deserialize from a heap-backed
    /// `Bytes`, so the new codec storage no longer references the mapping.
    ///
    /// No-op when already `InMemory`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization or deserialization
    /// fails.
    pub fn reload_to_ram(&self) -> Result<()> {
        let mut guard = self.state.write();
        let TierState::OnDisk { store, .. } = &guard.tier else {
            return Ok(());
        };
        let section = CompactStoreSection::new(Arc::clone(store));
        let bytes = section.serialize()?;
        let mut reloaded = CompactStoreSection::empty();
        reloaded.deserialize_from_bytes(Bytes::from(bytes))?;
        let new_store = reloaded.store().ok_or_else(|| {
            Error::Internal("empty CompactStoreSection after reload_to_ram".to_string())
        })?;
        let next_revision = guard.next_revision();
        let previous = std::mem::replace(&mut guard.tier, TierState::InMemory(new_store));
        guard.revision = next_revision;
        drop(guard);
        drop(previous);
        Ok(())
    }

    /// Re-materializes one caller-pinned compact generation into heap-backed
    /// bytes and returns the representation to publish.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization or deserialization fails.
    pub(super) fn prepare_generation_to_ram(
        &self,
        generation: Arc<CompactStore>,
    ) -> Result<PreparedTierGeneration> {
        let section = CompactStoreSection::new(generation);
        let bytes = section.serialize()?;
        let mut reloaded = CompactStoreSection::empty();
        reloaded.deserialize_from_bytes(Bytes::from(bytes))?;
        let store = reloaded.store().ok_or_else(|| {
            Error::Internal("empty CompactStoreSection after generation reload_to_ram".to_string())
        })?;
        Ok(PreparedTierGeneration {
            state: Some(TierState::InMemory(store)),
        })
    }

    /// Estimated heap memory footprint of the wrapped store, in bytes.
    ///
    /// When the state is `OnDisk`, this counts the heap copy alone: the
    /// mmap bytes live outside the heap and are managed by the OS page
    /// cache.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.store().memory_bytes()
    }

    fn generation_owner(&self, generation_root: &Path) -> Result<Arc<GenerationOwner>> {
        // Allocation is also the startup/retry reclamation boundary. A live
        // namespace is skipped solely because its kernel lease cannot be
        // acquired; all malformed or identity-ambiguous state is retained.
        scavenge_stale_generation_namespaces(generation_root)?;

        let mut cached = self.generation_owner.lock();
        if let Some(owner) = cached.as_ref()
            && owner.generation_root == generation_root
        {
            return Ok(Arc::clone(owner));
        }

        let owner = Arc::new(GenerationOwner::create(generation_root)?);
        *cached = Some(Arc::clone(&owner));
        Ok(owner)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActiveTarget {
    Inactive,
    VerifiedPersistentExact,
}

fn resolve_output_path(path: &Path) -> Result<ResolvedPath> {
    let io_path = absolute_path_at_entry(path)?;
    let parent = io_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|error| {
        Error::Internal(format!(
            "create compact output directory {}: {error}",
            parent.display()
        ))
    })?;
    resolved_path(path, io_path)
}

fn resolve_existing_path(path: &Path) -> Result<ResolvedPath> {
    let io_path = absolute_path_at_entry(path)?;
    resolved_path(path, io_path)
}

fn absolute_path_at_entry(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|current| current.join(path))
        .map_err(|error| Error::Internal(format!("resolve current directory: {error}")))
}

fn resolved_path(caller_path: &Path, io_path: PathBuf) -> Result<ResolvedPath> {
    let parent = io_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let resolved_parent = std::fs::canonicalize(parent).map_err(|error| {
        Error::Internal(format!(
            "resolve compact path parent {}: {error}",
            parent.display()
        ))
    })?;
    let file_name = io_path.file_name().ok_or_else(|| {
        Error::Internal(format!(
            "compact path must name a file: {}",
            caller_path.display()
        ))
    })?;
    let io_path = resolved_parent.join(file_name);
    Ok(ResolvedPath {
        caller_path: caller_path.to_path_buf(),
        io_path,
    })
}

fn reject_active_target(
    tier: &TierState,
    target: &ResolvedPath,
    allow_verified_persistent_exact: bool,
) -> Result<ActiveTarget> {
    let TierState::OnDisk { backing, .. } = tier else {
        return Ok(ActiveTarget::Inactive);
    };

    let target_identity = match std::fs::File::open(&target.io_path) {
        Ok(file) => Some(file_identity(&file).map_err(|error| {
            Error::Internal(format!(
                "identify compact target {}: {error}",
                target.caller_path.display()
            ))
        })?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(Error::Internal(format!(
                "inspect compact target {}: {error}",
                target.caller_path.display()
            )));
        }
    };
    let same_path = backing.io_path() == target.io_path;
    let same_file =
        target_identity.is_some_and(|identity| file_identities_match(identity, backing.identity()));

    if allow_verified_persistent_exact
        && same_path
        && same_file
        && matches!(backing, DiskBacking::PersistentExact { .. })
    {
        return Ok(ActiveTarget::VerifiedPersistentExact);
    }

    if same_path || same_file {
        let reason = if same_path && target_identity.is_none() {
            "the active backing path no longer exists"
        } else if same_path && !same_file {
            "the active backing path now names a different file"
        } else {
            "the target aliases the active backing file"
        };
        return Err(Error::Internal(format!(
            "refusing to replace compact target {} because {reason}",
            target.caller_path.display()
        )));
    }

    Ok(ActiveTarget::Inactive)
}

fn file_identities_match(left: FileIdentity, right: FileIdentity) -> bool {
    #[cfg(unix)]
    {
        left == right
    }
    #[cfg(windows)]
    {
        match (left, right) {
            (
                FileIdentity::Windows {
                    volume_serial: Some(left_volume),
                    file_index: Some(left_index),
                },
                FileIdentity::Windows {
                    volume_serial: Some(right_volume),
                    file_index: Some(right_index),
                },
            ) => left_volume == right_volume && left_index == right_index,
            _ => false,
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (left, right);
        false
    }
}

/// Removes an engine-owned path only when it is still a non-symlink regular
/// file with one link and the identity captured at creation/open time.
///
/// This protects against replacements completed before cleanup begins. The
/// portable reopen/compare/remove sequence is not an atomic filesystem
/// capability: a hostile external writer with access to Grafeo's private spill
/// directory could still race the final-component check and unlink. Engine
/// generation names are create-new and process-unique, so this is lifecycle
/// hardening rather than a security boundary against concurrent directory
/// mutation.
fn remove_owned_file_if_unchanged(path: &Path, expected: FileIdentity, role: &str) -> bool {
    let entry_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return true,
        Err(error) => {
            grafeo_warn!(
                "refusing to remove compact {role} {} because its directory entry cannot be inspected: {error}",
                path.display()
            );
            return false;
        }
    };
    if entry_metadata.file_type().is_symlink() || !entry_metadata.is_file() {
        grafeo_warn!(
            "refusing to remove compact {role} {} because it is not a regular non-symlink file",
            path.display()
        );
        return false;
    }

    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return true,
        Err(error) => {
            grafeo_warn!(
                "refusing to remove compact {role} {} because it cannot be reopened: {error}",
                path.display()
            );
            return false;
        }
    };
    let live_identity = match file_identity(&file) {
        Ok(identity) => identity,
        Err(error) => {
            grafeo_warn!(
                "refusing to remove compact {role} {} because its identity cannot be verified: {error}",
                path.display()
            );
            return false;
        }
    };
    let open_metadata = match file.metadata() {
        Ok(metadata) => metadata,
        Err(error) => {
            grafeo_warn!(
                "refusing to remove compact {role} {} because its open metadata cannot be read: {error}",
                path.display()
            );
            return false;
        }
    };

    if !file_identities_match(live_identity, expected)
        || !file_identities_match(metadata_identity(&entry_metadata), expected)
    {
        grafeo_warn!(
            "refusing to remove compact {role} {} because the path now names a different file",
            path.display()
        );
        return false;
    }
    if !metadata_has_one_link(&open_metadata) || !metadata_has_one_link(&entry_metadata) {
        grafeo_warn!(
            "refusing to remove compact {role} {} because it has a foreign hard link or its link count is unavailable",
            path.display()
        );
        return false;
    }

    // Re-read the final component after opening. This cannot turn pathname
    // cleanup into an openat-style capability on every supported platform, but
    // it closes the ordinary symlink/replacement window and fails closed.
    let final_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return true,
        Err(error) => {
            grafeo_warn!(
                "refusing to remove compact {role} {} because its final identity cannot be rechecked: {error}",
                path.display()
            );
            return false;
        }
    };
    if final_metadata.file_type().is_symlink()
        || !final_metadata.is_file()
        || !metadata_has_one_link(&final_metadata)
        || !file_identities_match(metadata_identity(&final_metadata), expected)
    {
        grafeo_warn!(
            "refusing to remove compact {role} {} because its final component changed during verification",
            path.display()
        );
        return false;
    }

    match std::fs::remove_file(path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => {
            grafeo_warn!(
                "failed to remove compact {role} {} after identity verification: {error}",
                path.display()
            );
            false
        }
    }
}

#[cfg(unix)]
fn metadata_identity(metadata: &std::fs::Metadata) -> FileIdentity {
    use std::os::unix::fs::MetadataExt;

    FileIdentity::Unix {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

#[cfg(windows)]
fn metadata_identity(metadata: &std::fs::Metadata) -> FileIdentity {
    use std::os::windows::fs::MetadataExt;

    FileIdentity::Windows {
        volume_serial: metadata.volume_serial_number(),
        file_index: metadata.file_index(),
    }
}

#[cfg(not(any(unix, windows)))]
fn metadata_identity(_metadata: &std::fs::Metadata) -> FileIdentity {
    FileIdentity::Unsupported
}

#[cfg(unix)]
fn metadata_has_one_link(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    metadata.nlink() == 1
}

#[cfg(windows)]
fn metadata_has_one_link(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;

    metadata.number_of_links() == Some(1)
}

#[cfg(not(any(unix, windows)))]
fn metadata_has_one_link(_metadata: &std::fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn file_identity(file: &std::fs::File) -> std::io::Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file.metadata()?;
    Ok(FileIdentity::Unix {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(windows)]
fn file_identity(file: &std::fs::File) -> std::io::Result<FileIdentity> {
    use std::os::windows::fs::MetadataExt;

    let metadata = file.metadata()?;
    Ok(FileIdentity::Windows {
        volume_serial: metadata.volume_serial_number(),
        file_index: metadata.file_index(),
    })
}

#[cfg(not(any(unix, windows)))]
fn file_identity(file: &std::fs::File) -> std::io::Result<FileIdentity> {
    let _ = file.metadata()?;
    Ok(FileIdentity::Unsupported)
}

impl GenerationOwner {
    fn create(generation_root: &Path) -> Result<Self> {
        let root_key = generation_root_key(generation_root);
        let parent = generation_root
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::Internal(format!(
                "create compact generation owner directory {}: {error}",
                parent.display()
            ))
        })?;

        for _ in 0..128 {
            let token = new_owner_token()?;
            let pending_path = owner_pending_path(generation_root, &root_key, &token);
            let mut file = match std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&pending_path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(Error::Internal(format!(
                        "create compact owner lease {}: {error}",
                        pending_path.display()
                    )));
                }
            };
            let identity = file_identity(&file).map_err(|error| {
                Error::Internal(format!(
                    "identify compact owner lease {}: {error}",
                    pending_path.display()
                ))
            })?;
            let mut pending = UnpublishedFile {
                path: Some(pending_path.clone()),
                expected_identity: identity,
                _owner: None,
            };
            let identity_tag = file_identity_tag(identity)?;
            let final_path = owner_lease_path(generation_root, &root_key, &token, &identity_tag);

            if let Err(error) = file.try_lock() {
                drop(file);
                drop(pending);
                return Err(Error::Internal(format!(
                    "lock new compact owner lease {}: {error}",
                    pending_path.display()
                )));
            }
            // The exact root-hash + 128-bit-token pending namespace is reserved
            // for this cache protocol. A scavenger may now safely distinguish
            // it from malformed names, and the held lock protects live setup.
            compact_crash_test_pause("owner-create");

            let lease_bytes = owner_lease_bytes(&root_key, &token, &identity_tag);
            if let Err(error) = file.write_all(lease_bytes.as_bytes()) {
                let _ = file.unlock();
                drop(file);
                drop(pending);
                return Err(Error::Internal(format!(
                    "write compact owner lease {}: {error}",
                    pending_path.display()
                )));
            }
            if let Err(error) = file.sync_all() {
                let _ = file.unlock();
                drop(file);
                drop(pending);
                return Err(Error::Internal(format!(
                    "sync compact owner lease {}: {error}",
                    pending_path.display()
                )));
            }
            compact_crash_test_pause("owner-write");

            if let Err(error) = std::fs::hard_link(&pending_path, &final_path) {
                let _ = file.unlock();
                drop(file);
                drop(pending);
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    continue;
                }
                return Err(Error::Internal(format!(
                    "install identity-bound compact owner lease {}: {error}",
                    final_path.display()
                )));
            }
            compact_crash_test_pause("owner-link");

            pending.disarm();
            if !identity_link_pair_is_exact(&pending_path, &final_path, identity) {
                let _ = file.unlock();
                drop(file);
                let _ =
                    remove_identity_link_pair(&pending_path, &final_path, identity, "owner lease");
                return Err(Error::Internal(format!(
                    "compact owner lease link identity changed while installing {}",
                    final_path.display()
                )));
            }
            if let Err(error) = std::fs::remove_file(&pending_path) {
                let _ = file.unlock();
                drop(file);
                let _ =
                    remove_identity_link_pair(&pending_path, &final_path, identity, "owner lease");
                return Err(Error::Internal(format!(
                    "retire compact owner lease installation link {}: {error}",
                    pending_path.display()
                )));
            }
            let mut final_lease = UnpublishedFile {
                path: Some(final_path.clone()),
                expected_identity: identity,
                _owner: None,
            };
            if !path_is_exact_regular_file(&final_path, identity, 1) {
                let _ = file.unlock();
                drop(file);
                drop(final_lease);
                return Err(Error::Internal(format!(
                    "compact owner lease identity changed after installation {}",
                    final_path.display()
                )));
            }
            if let Err(error) = sync_parent_directory(&final_path) {
                let _ = file.unlock();
                drop(file);
                drop(final_lease);
                return Err(Error::Internal(format!(
                    "sync compact owner lease directory for {}: {error}",
                    final_path.display()
                )));
            }
            final_lease.disarm();

            return Ok(Self {
                generation_root: generation_root.to_path_buf(),
                token,
                lease_path: final_path,
                lease_identity: identity,
                lease_file: Some(file),
                next_generation: AtomicU64::new(1),
            });
        }

        Err(Error::Internal(format!(
            "could not reserve a compact generation owner beside {}",
            generation_root.display()
        )))
    }
}

fn generation_root_key(generation_root: &Path) -> String {
    let digest = blake3::hash(generation_root.as_os_str().as_encoded_bytes());
    encode_hex(digest.as_bytes())
}

fn new_owner_token() -> Result<String> {
    let mut token = [0_u8; 16];
    getrandom::fill(&mut token).map_err(|error| {
        Error::Internal(format!(
            "generate compact owner namespace token from operating-system entropy: {error}"
        ))
    })?;
    Ok(encode_hex(&token))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn file_identity_tag(identity: FileIdentity) -> Result<String> {
    #[cfg(unix)]
    {
        let FileIdentity::Unix { device, inode } = identity;
        Ok(format!("u{device:016x}.{inode:016x}"))
    }
    #[cfg(windows)]
    {
        let FileIdentity::Windows {
            volume_serial: Some(volume_serial),
            file_index: Some(file_index),
        } = identity
        else {
            return Err(Error::Internal(
                "engine-owned compact generations require a stable Windows file identity"
                    .to_string(),
            ));
        };
        Ok(format!("w{volume_serial:08x}.{file_index:016x}"))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = identity;
        Err(Error::Internal(
            "engine-owned compact generations require stable filesystem identities".to_string(),
        ))
    }
}

fn parse_file_identity_tag(tag: &str) -> Option<FileIdentity> {
    #[cfg(unix)]
    {
        let (device, inode) = tag.strip_prefix('u')?.split_once('.')?;
        if !is_lower_hex(device, 16) || !is_lower_hex(inode, 16) {
            return None;
        }
        Some(FileIdentity::Unix {
            device: u64::from_str_radix(device, 16).ok()?,
            inode: u64::from_str_radix(inode, 16).ok()?,
        })
    }
    #[cfg(windows)]
    {
        let (volume, index) = tag.strip_prefix('w')?.split_once('.')?;
        if !is_lower_hex(volume, 8) || !is_lower_hex(index, 16) {
            return None;
        }
        Some(FileIdentity::Windows {
            volume_serial: Some(u32::from_str_radix(volume, 16).ok()?),
            file_index: Some(u64::from_str_radix(index, 16).ok()?),
        })
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = tag;
        None
    }
}

fn owner_pending_path(generation_root: &Path, root_key: &str, token: &str) -> PathBuf {
    generation_root.with_file_name(format!(".{OWNER_ROLE}-{root_key}-{token}.pending"))
}

fn owner_lease_path(
    generation_root: &Path,
    root_key: &str,
    token: &str,
    identity_tag: &str,
) -> PathBuf {
    generation_root.with_file_name(format!(
        ".{OWNER_ROLE}-{root_key}-{token}-{identity_tag}.lease"
    ))
}

fn owner_lease_bytes(root_key: &str, token: &str, identity_tag: &str) -> String {
    format!("{OWNER_LEASE_MAGIC}\nroot={root_key}\ntoken={token}\nidentity={identity_tag}\n")
}

fn generation_prefix(root_key: &str, token: &str) -> String {
    format!(".{GENERATION_ROLE}-{root_key}-{token}-")
}

fn generation_pending_path(
    generation_root: &Path,
    root_key: &str,
    token: &str,
    sequence: u64,
) -> PathBuf {
    generation_root.with_file_name(format!(
        "{}{sequence:016x}.pending",
        generation_prefix(root_key, token)
    ))
}

fn generation_final_path(
    generation_root: &Path,
    root_key: &str,
    token: &str,
    sequence: u64,
    identity_tag: &str,
) -> PathBuf {
    generation_root.with_file_name(format!(
        "{}{sequence:016x}-{identity_tag}",
        generation_prefix(root_key, token)
    ))
}

fn metadata_link_count(metadata: &std::fs::Metadata) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(metadata.nlink())
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.number_of_links().map(u64::from)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = metadata;
        None
    }
}

fn path_is_exact_regular_file(path: &Path, expected: FileIdentity, links: u64) -> bool {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    !metadata.file_type().is_symlink()
        && metadata.is_file()
        && metadata_link_count(&metadata) == Some(links)
        && file_identities_match(metadata_identity(&metadata), expected)
}

fn identity_link_pair_is_exact(left: &Path, right: &Path, expected: FileIdentity) -> bool {
    path_is_exact_regular_file(left, expected, 2) && path_is_exact_regular_file(right, expected, 2)
}

/// Removes the protocol's temporary and final names only when they are exactly
/// the two links to the identity encoded in the final name. A third/foreign
/// hard link or either substituted entry makes the operation fail closed.
fn remove_identity_link_pair(
    pending: &Path,
    final_path: &Path,
    expected: FileIdentity,
    role: &str,
) -> bool {
    if !identity_link_pair_is_exact(pending, final_path, expected) {
        grafeo_warn!(
            "refusing to retire compact {role} link pair {} and {} because its identity or link count changed",
            pending.display(),
            final_path.display()
        );
        return false;
    }
    if let Err(error) = std::fs::remove_file(pending) {
        grafeo_warn!(
            "failed to retire compact {role} installation link {}: {error}",
            pending.display()
        );
        return false;
    }
    remove_owned_file_if_unchanged(final_path, expected, role)
}

fn parse_owner_lease_name<'a>(name: &'a str, root_key: &str) -> Option<(&'a str, FileIdentity)> {
    let remainder = name
        .strip_prefix(&format!(".{OWNER_ROLE}-{root_key}-"))?
        .strip_suffix(".lease")?;
    let (token, identity_tag) = remainder.split_once('-')?;
    if !is_lower_hex(token, 32) {
        return None;
    }
    Some((token, parse_file_identity_tag(identity_tag)?))
}

fn parse_owner_pending_name<'a>(name: &'a str, root_key: &str) -> Option<&'a str> {
    let token = name
        .strip_prefix(&format!(".{OWNER_ROLE}-{root_key}-"))?
        .strip_suffix(".pending")?;
    is_lower_hex(token, 32).then_some(token)
}

enum ParsedGenerationEntry {
    Pending {
        sequence: String,
    },
    Final {
        sequence: String,
        identity: FileIdentity,
    },
}

fn parse_generation_entry(name: &str, prefix: &str) -> Option<ParsedGenerationEntry> {
    let remainder = name.strip_prefix(prefix)?;
    if let Some(sequence) = remainder.strip_suffix(".pending") {
        if !is_lower_hex(sequence, 16) {
            return None;
        }
        return Some(ParsedGenerationEntry::Pending {
            sequence: sequence.to_string(),
        });
    }
    let (sequence, identity_tag) = remainder.split_once('-')?;
    if !is_lower_hex(sequence, 16) {
        return None;
    }
    Some(ParsedGenerationEntry::Final {
        sequence: sequence.to_string(),
        identity: parse_file_identity_tag(identity_tag)?,
    })
}

fn namespace_entries(generation_root: &Path, token: &str) -> std::io::Result<Vec<PathBuf>> {
    let root_key = generation_root_key(generation_root);
    let prefix = generation_prefix(&root_key, token);
    let parent = generation_root
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        if entry
            .file_name()
            .as_encoded_bytes()
            .starts_with(prefix.as_bytes())
        {
            paths.push(entry.path());
        }
    }
    Ok(paths)
}

/// Reclaims only namespaces whose exact identity-bound lease can be locked.
/// An acquired lock proves that no live process retains that namespace: file
/// locks are released by the kernel after normal exit and process death alike.
fn scavenge_stale_generation_namespaces(generation_root: &Path) -> Result<()> {
    let root_key = generation_root_key(generation_root);
    let owner_prefix = format!(".{OWNER_ROLE}-{root_key}-");
    let parent = generation_root
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let entries = std::fs::read_dir(parent).map_err(|error| {
        Error::Internal(format!(
            "scan compact generation owners beside {}: {error}",
            generation_root.display()
        ))
    })?;
    let mut owners: std::collections::BTreeMap<
        String,
        (Vec<PathBuf>, Vec<(PathBuf, FileIdentity)>),
    > = std::collections::BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            Error::Internal(format!(
                "read compact generation owner entry beside {}: {error}",
                generation_root.display()
            ))
        })?;
        let name = entry.file_name();
        if !name.as_encoded_bytes().starts_with(owner_prefix.as_bytes()) {
            continue;
        }
        let Some(name) = name.to_str() else {
            // It shares our ASCII prefix but is not one of our codec names.
            // Retain it; ownership cannot be proven losslessly.
            continue;
        };
        if let Some(token) = parse_owner_pending_name(name, &root_key) {
            owners
                .entry(token.to_string())
                .or_default()
                .0
                .push(entry.path());
        } else if let Some((token, expected_identity)) = parse_owner_lease_name(name, &root_key) {
            owners
                .entry(token.to_string())
                .or_default()
                .1
                .push((entry.path(), expected_identity));
        }
        // Malformed, foreign, and future-version owner entries are retained:
        // they never enter a reclaimable ownership group.
    }

    for (token, (pending, leases)) in owners {
        match (pending.as_slice(), leases.as_slice()) {
            ([pending_path], []) => {
                scavenge_stale_pending_owner(generation_root, &token, pending_path);
            }
            ([pending_path], [(lease_path, expected_identity)]) => {
                scavenge_stale_owner_link_pair(
                    generation_root,
                    &root_key,
                    &token,
                    pending_path,
                    lease_path,
                    *expected_identity,
                );
            }
            ([], [(lease_path, expected_identity)]) => scavenge_one_stale_namespace(
                generation_root,
                &root_key,
                &token,
                lease_path,
                *expected_identity,
            ),
            _ => {
                // Duplicate/ambiguous names are not an ownership proof.
            }
        }
    }
    Ok(())
}

fn open_and_lock_owner_entry(
    path: &Path,
    expected_identity: FileIdentity,
    links: u64,
) -> Option<File> {
    if !path_is_exact_regular_file(path, expected_identity, links) {
        grafeo_warn!(
            "refusing compact owner reclamation because {} is a symlink, has an unexpected link count, or has a different identity",
            path.display()
        );
        return None;
    }
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
    {
        Ok(file) => file,
        Err(error) => {
            grafeo_warn!(
                "refusing compact owner reclamation because {} cannot be opened: {error}",
                path.display()
            );
            return None;
        }
    };
    if !matches!(
        file_identity(&file),
        Ok(identity) if file_identities_match(identity, expected_identity)
    ) {
        grafeo_warn!(
            "refusing compact owner reclamation because {} changed during open",
            path.display()
        );
        return None;
    }
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return None,
        Err(TryLockError::Error(error)) => {
            grafeo_warn!(
                "refusing compact owner reclamation because {} cannot be locked: {error}",
                path.display()
            );
            return None;
        }
    }
    if !path_is_exact_regular_file(path, expected_identity, links) {
        let _ = file.unlock();
        grafeo_warn!(
            "refusing compact owner reclamation because {} changed after lock",
            path.display()
        );
        return None;
    }
    Some(file)
}

fn read_owner_lease(file: &mut File) -> Option<String> {
    let mut lease_text = String::new();
    Read::by_ref(file)
        .take(4097)
        .read_to_string(&mut lease_text)
        .ok()?;
    (lease_text.len() <= 4096).then_some(lease_text)
}

fn parse_exact_owner_lease_identity(
    lease_text: &str,
    root_key: &str,
    token: &str,
) -> Option<FileIdentity> {
    let identity_tag = lease_text
        .strip_prefix(&format!(
            "{OWNER_LEASE_MAGIC}\nroot={root_key}\ntoken={token}\nidentity="
        ))?
        .strip_suffix('\n')?;
    let identity = parse_file_identity_tag(identity_tag)?;
    (lease_text == owner_lease_bytes(root_key, token, identity_tag)).then_some(identity)
}

/// Reclaims an exact crash-left pending name from the cache protocol's reserved
/// namespace, including an empty or partial file left between `create_new` and
/// record sync. The 128-bit token comes from the OS CSPRNG. The configured spill
/// root therefore reserves exact-shaped pending names; callers must not place
/// their own files there. Names outside the exact codec, directories, symlinks,
/// extra-linked entries, live locks, and entries that change identity remain
/// untouched.
fn scavenge_stale_pending_owner(generation_root: &Path, token: &str, pending_path: &Path) {
    let metadata = match std::fs::symlink_metadata(pending_path) {
        Ok(metadata)
            if !metadata.file_type().is_symlink()
                && metadata.is_file()
                && metadata_link_count(&metadata) == Some(1) =>
        {
            metadata
        }
        _ => return,
    };
    let observed_identity = metadata_identity(&metadata);
    let Some(file) = open_and_lock_owner_entry(pending_path, observed_identity, 1) else {
        return;
    };
    if !path_is_exact_regular_file(pending_path, observed_identity, 1)
        || !matches!(
            namespace_entries(generation_root, token),
            Ok(entries) if entries.is_empty()
        )
    {
        let _ = file.unlock();
        return;
    }
    if file.unlock().is_err() {
        return;
    }
    drop(file);
    let _ = remove_owned_file_if_unchanged(
        pending_path,
        observed_identity,
        "stale pending owner lease",
    );
}

/// Reclaims the narrow hard-link installation state emitted by owner creation.
/// Both names must be the only two links to the identity encoded in the final
/// lease name, and the file record must agree exactly. Any extra link or entry
/// substitution remains untouched.
fn scavenge_stale_owner_link_pair(
    generation_root: &Path,
    root_key: &str,
    token: &str,
    pending_path: &Path,
    lease_path: &Path,
    expected_identity: FileIdentity,
) {
    if !identity_link_pair_is_exact(pending_path, lease_path, expected_identity)
        || !matches!(
            namespace_entries(generation_root, token),
            Ok(entries) if entries.is_empty()
        )
    {
        return;
    }
    let Some(mut file) = open_and_lock_owner_entry(pending_path, expected_identity, 2) else {
        return;
    };
    let contents_are_exact = read_owner_lease(&mut file).is_some_and(|text| {
        parse_exact_owner_lease_identity(&text, root_key, token)
            .is_some_and(|identity| file_identities_match(identity, expected_identity))
    });
    if !contents_are_exact
        || !identity_link_pair_is_exact(pending_path, lease_path, expected_identity)
    {
        let _ = file.unlock();
        return;
    }
    if file.unlock().is_err() {
        return;
    }
    drop(file);
    let _ = remove_identity_link_pair(
        pending_path,
        lease_path,
        expected_identity,
        "stale owner lease",
    );
}

fn scavenge_one_stale_namespace(
    generation_root: &Path,
    root_key: &str,
    token: &str,
    lease_path: &Path,
    expected_identity: FileIdentity,
) {
    if !path_is_exact_regular_file(lease_path, expected_identity, 1) {
        grafeo_warn!(
            "refusing compact namespace reclamation because owner lease {} is a symlink, hard link, or identity mismatch",
            lease_path.display()
        );
        return;
    }

    let mut lease_file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(lease_path)
    {
        Ok(file) => file,
        Err(error) => {
            grafeo_warn!(
                "refusing compact namespace reclamation because owner lease {} cannot be opened: {error}",
                lease_path.display()
            );
            return;
        }
    };
    if !matches!(
        file_identity(&lease_file),
        Ok(identity) if file_identities_match(identity, expected_identity)
    ) {
        grafeo_warn!(
            "refusing compact namespace reclamation because owner lease {} changed during open",
            lease_path.display()
        );
        return;
    }

    match lease_file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return,
        Err(TryLockError::Error(error)) => {
            grafeo_warn!(
                "refusing compact namespace reclamation because owner lease {} cannot be locked: {error}",
                lease_path.display()
            );
            return;
        }
    }
    if !path_is_exact_regular_file(lease_path, expected_identity, 1) {
        let _ = lease_file.unlock();
        grafeo_warn!(
            "refusing compact namespace reclamation because owner lease {} changed after lock",
            lease_path.display()
        );
        return;
    }

    let mut lease_text = String::new();
    if Read::by_ref(&mut lease_file)
        .take(4097)
        .read_to_string(&mut lease_text)
        .is_err()
    {
        let _ = lease_file.unlock();
        grafeo_warn!(
            "refusing compact namespace reclamation because owner lease {} is unreadable",
            lease_path.display()
        );
        return;
    }
    let Ok(expected_tag) = file_identity_tag(expected_identity) else {
        let _ = lease_file.unlock();
        return;
    };
    if lease_text.len() > 4096 || lease_text != owner_lease_bytes(root_key, token, &expected_tag) {
        let _ = lease_file.unlock();
        grafeo_warn!(
            "refusing compact namespace reclamation because owner lease {} has invalid contents",
            lease_path.display()
        );
        return;
    }

    reclaim_stale_generation_entries(generation_root, root_key, token);
    let namespace_empty =
        namespace_entries(generation_root, token).is_ok_and(|entries| entries.is_empty());
    if let Err(error) = lease_file.unlock() {
        grafeo_warn!(
            "failed to unlock stale compact owner lease {}: {error}",
            lease_path.display()
        );
        return;
    }
    drop(lease_file);
    if namespace_empty {
        let _ = remove_owned_file_if_unchanged(lease_path, expected_identity, "stale owner lease");
    }
}

fn reclaim_stale_generation_entries(generation_root: &Path, root_key: &str, token: &str) {
    let prefix = generation_prefix(root_key, token);
    let entries = match namespace_entries(generation_root, token) {
        Ok(entries) => entries,
        Err(error) => {
            grafeo_warn!(
                "refusing compact namespace reclamation beside {} because entries cannot be listed: {error}",
                generation_root.display()
            );
            return;
        }
    };
    let mut grouped: std::collections::BTreeMap<
        String,
        (Vec<PathBuf>, Vec<(PathBuf, FileIdentity)>),
    > = std::collections::BTreeMap::new();
    for path in entries {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        match parse_generation_entry(name, &prefix) {
            Some(ParsedGenerationEntry::Pending { sequence }) => {
                grouped.entry(sequence).or_default().0.push(path);
            }
            Some(ParsedGenerationEntry::Final { sequence, identity }) => {
                grouped
                    .entry(sequence)
                    .or_default()
                    .1
                    .push((path, identity));
            }
            None => {}
        }
    }

    for (_sequence, (pending, final_paths)) in grouped {
        match (pending.as_slice(), final_paths.as_slice()) {
            ([pending_path], []) => {
                // The already validated and exclusively locked owner lease
                // reserves this exact root/token/sequence pending namespace.
                // Empty/partial crash state is reclaimable, but a symlink or
                // extra hard link is not.
                let metadata = match std::fs::symlink_metadata(pending_path) {
                    Ok(metadata)
                        if !metadata.file_type().is_symlink()
                            && metadata.is_file()
                            && metadata_link_count(&metadata) == Some(1) =>
                    {
                        metadata
                    }
                    _ => continue,
                };
                let expected_identity = metadata_identity(&metadata);
                let _ = remove_owned_file_if_unchanged(
                    pending_path,
                    expected_identity,
                    "stale pending generation",
                );
            }
            ([pending_path], [(final_path, expected_identity)]) => {
                let _ = remove_identity_link_pair(
                    pending_path,
                    final_path,
                    *expected_identity,
                    "stale generation",
                );
            }
            ([], [(path, expected_identity)]) => {
                let _ =
                    remove_owned_file_if_unchanged(path, *expected_identity, "stale generation");
            }
            _ => {
                // Duplicate or otherwise ambiguous names remain untouched.
            }
        }
    }
}

/// Writes, closes, reopens, mmaps, and fully decodes an immutable generation.
///
/// Each candidate uses a process- and generation-unique sibling name opened
/// with `create_new`. Distinct databases sharing one spill directory cannot
/// overwrite each other, and no mapped source or destination is ever renamed.
fn create_mapped_generation(
    owner: &Arc<GenerationOwner>,
    bytes: &[u8],
) -> Result<(PathBuf, FileIdentity, Bytes, Arc<CompactStore>)> {
    create_mapped_generation_after_close_with_owner(owner, bytes, |_| {})
}

#[cfg(test)]
fn create_mapped_generation_after_close(
    generation_root: &Path,
    bytes: &[u8],
    after_close: impl FnOnce(&Path),
) -> Result<(PathBuf, FileIdentity, Bytes, Arc<CompactStore>)> {
    scavenge_stale_generation_namespaces(generation_root)?;
    let owner = Arc::new(GenerationOwner::create(generation_root)?);
    create_mapped_generation_after_close_with_owner(&owner, bytes, after_close)
}

fn create_mapped_generation_after_close_with_owner(
    owner: &Arc<GenerationOwner>,
    bytes: &[u8],
    after_close: impl FnOnce(&Path),
) -> Result<(PathBuf, FileIdentity, Bytes, Arc<CompactStore>)> {
    let (mut candidate, mut file) = create_generation_candidate(owner)?;
    let generation = candidate.path().to_path_buf();
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        return Err(Error::Internal(format!(
            "write compact generation {}: {error}",
            generation.display()
        )));
    }
    compact_crash_test_pause("generation-write");
    drop(file);
    after_close(&generation);
    match open_and_deserialize(
        &generation,
        MappedFileCleanup::RemoveWhenUnmapped {
            path: generation.clone(),
            expected_identity: Some(candidate.expected_identity()),
            owner: Arc::clone(owner),
        },
    ) {
        Ok((mmap_bytes, store, identity)) => {
            // The mmap-backed owner now has sole cleanup responsibility.
            candidate.disarm();
            compact_crash_test_pause("generation-mmap");
            Ok((generation, identity, mmap_bytes, store))
        }
        Err(error) => Err(error),
    }
}

fn create_generation_candidate(owner: &Arc<GenerationOwner>) -> Result<(UnpublishedFile, File)> {
    let root_key = generation_root_key(&owner.generation_root);
    for _ in 0..128 {
        let sequence = owner.next_generation.fetch_add(1, Ordering::Relaxed);
        let pending_path =
            generation_pending_path(&owner.generation_root, &root_key, &owner.token, sequence);
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&pending_path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(Error::Internal(format!(
                    "create compact generation candidate {}: {error}",
                    pending_path.display()
                )));
            }
        };
        compact_crash_test_pause("generation-pending");
        let identity = match file_identity(&file) {
            Ok(identity) => identity,
            Err(error) => {
                drop(file);
                let _ = std::fs::remove_file(&pending_path);
                return Err(Error::Internal(format!(
                    "identify new compact generation candidate {}: {error}",
                    pending_path.display()
                )));
            }
        };
        let mut pending = UnpublishedFile {
            path: Some(pending_path.clone()),
            expected_identity: identity,
            _owner: Some(Arc::clone(owner)),
        };
        let identity_tag = file_identity_tag(identity)?;
        let final_path = generation_final_path(
            &owner.generation_root,
            &root_key,
            &owner.token,
            sequence,
            &identity_tag,
        );

        match std::fs::hard_link(&pending_path, &final_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                drop(file);
                drop(pending);
                continue;
            }
            Err(error) => {
                drop(file);
                drop(pending);
                return Err(Error::Internal(format!(
                    "install identity-bound compact generation {}: {error}",
                    final_path.display()
                )));
            }
        }
        compact_crash_test_pause("generation-link");
        pending.disarm();
        if !identity_link_pair_is_exact(&pending_path, &final_path, identity) {
            drop(file);
            let _ = remove_identity_link_pair(
                &pending_path,
                &final_path,
                identity,
                "generation candidate",
            );
            return Err(Error::Internal(format!(
                "compact generation identity changed while installing {}",
                final_path.display()
            )));
        }
        if let Err(error) = std::fs::remove_file(&pending_path) {
            drop(file);
            let _ = remove_identity_link_pair(
                &pending_path,
                &final_path,
                identity,
                "generation candidate",
            );
            return Err(Error::Internal(format!(
                "retire compact generation installation link {}: {error}",
                pending_path.display()
            )));
        }
        let final_candidate = UnpublishedFile {
            path: Some(final_path.clone()),
            expected_identity: identity,
            _owner: Some(Arc::clone(owner)),
        };
        if !path_is_exact_regular_file(&final_path, identity, 1) {
            drop(file);
            drop(final_candidate);
            return Err(Error::Internal(format!(
                "compact generation identity changed after installation {}",
                final_path.display()
            )));
        }
        sync_parent_directory(&final_path).map_err(|error| {
            Error::Internal(format!(
                "sync compact generation directory for {}: {error}",
                final_path.display()
            ))
        })?;
        compact_crash_test_pause("generation-create");
        return Ok((final_candidate, file));
    }

    Err(Error::Internal(format!(
        "could not reserve a unique compact generation in owner namespace {}",
        owner.token
    )))
}

fn create_unique_sibling(path: &Path, role: &str) -> Result<(UnpublishedFile, std::fs::File)> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::Internal(format!("create dir for {}: {error}", parent.display()))
        })?;
    }

    for _ in 0..128 {
        let candidate = unique_sibling_path(path, role);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                let expected_identity = match file_identity(&file) {
                    Ok(identity) => identity,
                    Err(error) => {
                        drop(file);
                        let _ = std::fs::remove_file(&candidate);
                        return Err(Error::Internal(format!(
                            "identify new compact {role} candidate {}: {error}",
                            candidate.display()
                        )));
                    }
                };
                return Ok((
                    UnpublishedFile {
                        path: Some(candidate),
                        expected_identity,
                        _owner: None,
                    },
                    file,
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(Error::Internal(format!(
                    "create compact {role} candidate {}: {error}",
                    candidate.display()
                )));
            }
        }
    }

    Err(Error::Internal(format!(
        "could not reserve a unique compact {role} candidate beside {}",
        path.display()
    )))
}

fn unique_sibling_path(path: &Path, role: &str) -> PathBuf {
    let sequence = NEXT_UNIQUE_FILE.fetch_add(1, Ordering::Relaxed);
    let mut name = path
        .file_name()
        .map_or_else(|| OsString::from("compact"), OsString::from);
    name.push(format!(".{role}-{}-{sequence}", std::process::id()));
    path.with_file_name(name)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let (mut candidate, mut file) = create_unique_sibling(path, "grafeo-stage")?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        return Err(Error::Internal(format!(
            "write compact staging file {}: {error}",
            candidate.path().display()
        )));
    }
    drop(file);
    std::fs::rename(candidate.path(), path).map_err(|error| {
        Error::Internal(format!(
            "atomically install compact staging file {} at {}: {error}; \
             the destination may already exist or be mapped on this platform",
            candidate.path().display(),
            path.display()
        ))
    })?;
    candidate.disarm();
    sync_parent_directory(path).map_err(|error| {
        Error::Internal(format!(
            "compact image was installed completely at {}, but syncing its parent directory failed: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn sync_parent_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn open_and_deserialize(
    path: &Path,
    cleanup: MappedFileCleanup,
) -> Result<(Bytes, Arc<CompactStore>, FileIdentity)> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Internal(format!("open {}: {e}", path.display())))?;
    let identity = file_identity(&file).map_err(|error| {
        Error::Internal(format!(
            "identify compact backing file {}: {error}",
            path.display()
        ))
    })?;
    let cleanup = match cleanup {
        MappedFileCleanup::Retain => MappedFileCleanup::Retain,
        MappedFileCleanup::RemoveWhenUnmapped {
            path,
            expected_identity,
            owner,
        } => {
            if expected_identity.is_none_or(|expected| {
                !file_identities_match(identity, expected)
                    || !path_is_exact_regular_file(&path, expected, 1)
            }) {
                return Err(Error::Internal(format!(
                    "refusing to mmap engine-owned compact generation {} because its file identity changed after creation or its type/link count is unsafe",
                    path.display()
                )));
            }
            MappedFileCleanup::RemoveWhenUnmapped {
                path,
                expected_identity: Some(identity),
                owner,
            }
        }
    };
    // SAFETY: we mmap a file that's owned by this process for the duration
    // of the `Mmap` lifetime. The file is read-only from Grafeo's side
    // (we never write through the mmap); external truncation or modification
    // while an `Mmap` is held is undefined per memmap2 docs, same caveat as
    // every other mmap call site in the project.
    #[allow(unsafe_code)]
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Internal(format!("mmap {}: {e}", path.display())))?;

    // Phase 3c: wrap the Mmap as a refcounted `Bytes` so column codec
    // storage can be `data.slice(range)` against this view — zero-copy.
    // Every codec's `Bytes` here shares the refcount that keeps the
    // Mmap alive. When the last `Bytes` referring to this region drops,
    // the OS unmaps.
    let mmap_bytes = Bytes::from_owner(MappedFileOwner {
        mmap: Some(mmap),
        cleanup,
    });

    let store = deserialize_store(mmap_bytes.clone(), &path.display().to_string())?;

    Ok((mmap_bytes, store, identity))
}

fn deserialize_store(data: Bytes, source: &str) -> Result<Arc<CompactStore>> {
    let mut section = CompactStoreSection::empty();
    section.deserialize_from_bytes(data)?;
    section.store().ok_or_else(|| {
        Error::Internal(format!(
            "empty CompactStoreSection after deserialize of {source}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_core::graph::compact::builder::from_graph_store;
    use grafeo_core::graph::lpg::LpgStore;
    use grafeo_core::graph::traits::GraphStore;

    const REJECTED_GCST_FIXTURES: [&[u8]; 8] = [
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v1.bin"),
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v2.bin"),
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v3.bin"),
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v4.bin"),
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v5.bin"),
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v6.bin"),
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v7.bin"),
        include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v8.bin"),
    ];

    fn assert_mmap_gcst_rejected(bytes: &[u8], expected: &str) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("rejected.compact");
        std::fs::write(&path, bytes).unwrap();
        let error = CompactStoreTiered::open_mmap(&path)
            .err()
            .expect("unsupported compact wire must not produce a mapped tier");
        assert!(
            matches!(&error, Error::Internal(message) if message.contains(expected)),
            "{error}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
    }

    #[test]
    fn open_mmap_rejects_authentic_gcst_predecessors_without_changing_files() {
        for (index, bytes) in REJECTED_GCST_FIXTURES.iter().enumerate() {
            let version = index + 1;
            assert_eq!(usize::from(bytes[4]), version);
            assert_mmap_gcst_rejected(
                bytes,
                &format!("unsupported CompactStore section version {version}"),
            );
        }
    }

    #[test]
    fn open_mmap_rejects_unknown_gcst_flags_without_changing_files() {
        let current = include_bytes!("../../tests/fixtures/gcst/current_gcst_v9.bin");
        assert_eq!(current[4], 9);
        for flag in [0x04, 0x80] {
            let mut bytes = current.to_vec();
            bytes[5] |= flag;
            let crc_offset = bytes.len() - 4;
            let crc = crc32fast::hash(&bytes[..crc_offset]);
            bytes[crc_offset..].copy_from_slice(&crc.to_le_bytes());
            assert_mmap_gcst_rejected(&bytes, "unsupported CompactStore flags");
        }
    }

    fn build_store_with_nodes(count: u64) -> Arc<CompactStore> {
        let lpg = LpgStore::new().expect("lpg store");
        for i in 0..count {
            let id = lpg.create_node(&["Person"]);
            lpg.set_node_property(id, "age", Value::Int64(i64::try_from(i).unwrap()));
            lpg.set_node_property(id, "name", Value::String(arcstr::format!("person-{i}")));
        }
        let compact = from_graph_store(&lpg).expect("compact");
        Arc::new(compact)
    }

    fn build_sample_store() -> Arc<CompactStore> {
        build_store_with_nodes(16)
    }

    fn prepared_path(prepared: &PreparedTierGeneration) -> PathBuf {
        match prepared.state.as_ref().expect("prepared state") {
            TierState::OnDisk { backing, .. } => backing.path().to_path_buf(),
            TierState::InMemory(_) => panic!("mmap preparation returned memory state"),
        }
    }

    fn generated_files(directory: &Path) -> Vec<PathBuf> {
        let mut files: Vec<_> = std::fs::read_dir(directory)
            .expect("read generation directory")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.contains(".grafeo-gen-") || name.contains(".grafeo-stage-")
                    })
                    .then_some(path)
            })
            .collect();
        files.sort();
        files
    }

    #[test]
    fn in_memory_roundtrip() {
        let store = build_sample_store();
        let expected = store.memory_bytes();
        let tiered = CompactStoreTiered::new_in_memory(store);
        assert!(!tiered.is_on_disk());
        assert_eq!(tiered.memory_bytes(), expected);
    }

    #[test]
    fn persist_and_mmap_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let store = build_sample_store();
        let expected_nodes = store.node_count();

        let tiered = CompactStoreTiered::new_in_memory(store);
        let written = tiered.persist_to_mmap(&path).expect("persist_to_mmap");
        assert!(written > 0);
        assert!(tiered.is_on_disk());
        assert_eq!(tiered.path().as_deref(), Some(path.as_path()));
        assert!(path.exists());
        assert!(tiered.generation_root().is_none());

        let store_after = tiered.store();
        assert_eq!(store_after.node_count(), expected_nodes);
    }

    #[test]
    fn shared_spill_root_allocates_distinct_immutable_database_generations() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let shared_root = tmp.path().join("compact_base.grafeo");
        let left = CompactStoreTiered::new_in_memory(build_store_with_nodes(3));
        let right = CompactStoreTiered::new_in_memory(build_store_with_nodes(7));

        left.persist_to_ephemeral_mmap(&shared_root)
            .expect("left spill");
        right
            .persist_to_ephemeral_mmap(&shared_root)
            .expect("right spill");
        let left_path = left.path().expect("left generation");
        let right_path = right.path().expect("right generation");
        let resolved_root = std::fs::canonicalize(tmp.path())
            .expect("canonical tempdir")
            .join("compact_base.grafeo");

        assert_ne!(left_path, right_path);
        assert!(left_path.exists());
        assert!(right_path.exists());
        assert_eq!(
            left.generation_root().as_deref(),
            Some(resolved_root.as_path())
        );
        assert_eq!(
            right.generation_root().as_deref(),
            Some(resolved_root.as_path())
        );
        assert_eq!(left.store().node_count(), 3);
        assert_eq!(right.store().node_count(), 7);
    }

    #[test]
    fn repeated_mmap_generation_never_replaces_a_live_mapped_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("base.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_store_with_nodes(2));
        tiered
            .persist_to_ephemeral_mmap(&root)
            .expect("first generation");
        let first_path = tiered.path().expect("first immutable generation");
        let first_store = tiered.store();

        let prepared = tiered
            .prepare_generation_to_mmap(build_store_with_nodes(5), &root)
            .expect("prepare second generation beside live mmap");
        let second_path = prepared_path(&prepared);
        assert_ne!(first_path, second_path);
        assert!(first_path.exists());
        assert!(second_path.exists());
        tiered.publish_prepared(prepared);

        assert_eq!(tiered.path().as_deref(), Some(second_path.as_path()));
        assert_eq!(tiered.store().node_count(), 5);
        assert_eq!(first_store.node_count(), 2);
        assert!(
            first_path.exists(),
            "retired generation remains while an old reader owns mmap slices"
        );
        drop(first_store);
        assert!(!first_path.exists());
    }

    #[test]
    fn reversible_publish_restores_exact_prior_pointer_backing_and_path() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exact = tmp.path().join("prior.compact");
        let root = tmp.path().join("candidate.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_store_with_nodes(2));
        tiered.persist_to_mmap(&exact).expect("persistent prior");
        let prior_store = tiered.store();
        let prior_bytes = std::fs::read(&exact).expect("prior bytes");

        let prepared = tiered
            .prepare_generation_to_mmap(build_store_with_nodes(5), &root)
            .expect("prepare candidate");
        let candidate_path = prepared_path(&prepared);
        let candidate_reader = prepared.store();
        let published = tiered.publish_prepared_reversibly(prepared);
        assert_eq!(tiered.path().as_deref(), Some(candidate_path.as_path()));
        assert_eq!(tiered.store().node_count(), 5);

        assert!(tiered.restore_published(published));
        assert!(Arc::ptr_eq(&tiered.store(), &prior_store));
        assert_eq!(tiered.path().as_deref(), Some(exact.as_path()));
        assert!(tiered.generation_root().is_none());
        assert_eq!(
            std::fs::read(&exact).expect("restored prior bytes"),
            prior_bytes
        );
        assert!(
            candidate_path.exists(),
            "captured candidate reader must retain its mmap and file after rollback"
        );

        drop(candidate_reader);
        assert!(
            !candidate_path.exists(),
            "candidate retires only after its last reader drains"
        );
    }

    #[test]
    fn failed_generation_publish_keeps_file_and_tier_on_old_generation() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered
            .persist_to_ephemeral_mmap(&path)
            .expect("install original mmap");
        let old_path = tiered.path().expect("original generation path");
        let old_bytes = std::fs::read(&old_path).expect("read original bytes");
        let old_store = tiered.store();

        let before_candidates = generated_files(tmp.path());
        tiered.fail_next_generation_publish_for_test();
        let error = match tiered.prepare_generation_to_mmap(build_store_with_nodes(3), &path) {
            Ok(_) => panic!("expected injected failure before in-memory publication"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("injected compact generation"), "{error}");
        assert_eq!(
            std::fs::read(&old_path).expect("read retained bytes"),
            old_bytes
        );
        assert_eq!(tiered.path().as_deref(), Some(old_path.as_path()));
        assert!(Arc::ptr_eq(&tiered.store(), &old_store));
        assert_eq!(tiered.store().node_count(), 16);
        assert_eq!(
            generated_files(tmp.path()),
            before_candidates,
            "failed preparation must retire its unpublished generation"
        );
    }

    #[test]
    fn open_mmap_reads_existing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let store = build_sample_store();
        let expected_nodes = store.node_count();
        let tiered = CompactStoreTiered::new_in_memory(store);
        tiered.persist_to_mmap(&path).expect("persist_to_mmap");

        let reopened = CompactStoreTiered::open_mmap(&path).expect("open_mmap");
        assert!(reopened.is_on_disk());
        assert_eq!(reopened.path().as_deref(), Some(path.as_path()));
        assert!(reopened.generation_root().is_none());
        assert_eq!(reopened.store().node_count(), expected_nodes);
    }

    /// `reload_to_ram` must produce a store whose column codec storage
    /// is heap-backed, not a mmap slice — otherwise the tier label is a
    /// lie. We prove the disconnect by deleting the backing file after
    /// reload and confirming reads still succeed (mmap-backed reads
    /// would be unspecified after unlink on Windows and could fault).
    #[test]
    fn reload_to_ram_drops_mmap_backing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered.persist_to_mmap(&path).expect("persist_to_mmap");
        assert!(tiered.is_on_disk());
        assert_eq!(tiered.path().as_deref(), Some(path.as_path()));

        tiered.reload_to_ram().expect("reload_to_ram");
        assert!(!tiered.is_on_disk());

        assert!(
            path.exists(),
            "reload must retain the caller-owned exact-path file"
        );

        // The wrapper released its own mapping before returning. On Windows
        // this removal would fail if any mapped view remained live.
        std::fs::remove_file(&path).expect("persistent file is caller-removable after reload");

        // Reads still work after the caller removes the file — the data lives on
        // the heap now.
        let store = tiered.store();
        let person_ids = store.nodes_by_label("Person");
        assert!(!person_ids.is_empty());
        for id in person_ids.iter().take(4) {
            assert!(
                store
                    .get_node_property(*id, &PropertyKey::new("name"))
                    .is_some(),
                "name property still readable after backing file removal"
            );
        }
    }

    #[test]
    fn reload_to_ram_transitions_state() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered
            .persist_to_ephemeral_mmap(&path)
            .expect("ephemeral spill");
        assert!(tiered.is_on_disk());

        tiered.reload_to_ram().expect("reload_to_ram");
        assert!(!tiered.is_on_disk());
        assert!(tiered.path().is_none());

        // Reads still work after reload.
        let store = tiered.store();
        assert!(store.node_count() > 0);
    }

    #[test]
    fn persist_without_mmap_keeps_memory_state() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("snapshot.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        let written = tiered.persist(&path).expect("persist");
        assert!(written > 0);
        assert!(
            !tiered.is_on_disk(),
            "persist() alone must not change tier state"
        );
        assert!(path.exists());
    }

    #[test]
    fn spill_drops_original_arc() {
        // Proxy for "spill frees memory": the in-memory Arc the wrapper held
        // before `persist_to_mmap` must be dropped during the transition, so
        // the only live reference left to that specific allocation is
        // whatever the caller chose to hold. Verified by comparing strong
        // counts before and after.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        // One Arc in the wrapper, none held externally.
        assert_eq!(Arc::strong_count(&tiered.store()), 2);
        // The line above borrowed a clone then dropped it; wrapper now holds 1.

        tiered.persist_to_mmap(&path).expect("persist_to_mmap");

        // After spill: the wrapper holds a fresh Arc pointing at a
        // newly-deserialized CompactStore. The original allocation is gone.
        let after = tiered.store();
        // wrapper holds 1 + our binding holds 1 = 2
        assert_eq!(Arc::strong_count(&after), 2);
    }

    #[test]
    fn store_values_survive_mmap_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        let before = tiered.store();
        let first_id = before.nodes_by_label("Person").first().copied();
        let first_name =
            first_id.and_then(|id| before.get_node_property(id, &PropertyKey::new("name")));

        tiered.persist_to_mmap(&path).expect("persist_to_mmap");

        let after = tiered.store();
        let after_name =
            first_id.and_then(|id| after.get_node_property(id, &PropertyKey::new("name")));
        assert_eq!(first_name, after_name);
    }

    #[test]
    fn repeated_persist_to_same_exact_path_is_windows_safe_noop() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("exact.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());

        let first_size = tiered.persist_to_mmap(&path).expect("first persist");
        let first_store = tiered.store();
        let first_bytes = std::fs::read(&path).expect("first exact bytes");
        let second_size = tiered.persist_to_mmap(&path).expect("same-path repeat");

        assert_eq!(second_size, first_size);
        assert_eq!(
            std::fs::read(&path).expect("repeated exact bytes"),
            first_bytes
        );
        assert!(Arc::ptr_eq(&tiered.store(), &first_store));
        assert_eq!(tiered.path().as_deref(), Some(path.as_path()));
        assert!(generated_files(tmp.path()).is_empty());
    }

    #[test]
    fn active_persistent_and_ephemeral_targets_cannot_be_replaced() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exact = tmp.path().join("exact.compact");
        let root = tmp.path().join("spill.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());

        tiered.persist_to_mmap(&exact).expect("persistent mapping");
        let error = tiered
            .persist(&exact)
            .expect_err("snapshot write must not replace its active mapping")
            .to_string();
        assert!(error.contains("active backing"), "{error}");
        tiered
            .persist_to_mmap(&exact)
            .expect("verified exact persistent repeat is a no-op");

        tiered
            .persist_to_ephemeral_mmap(&root)
            .expect("ephemeral mapping");
        let active = tiered.path().expect("active ephemeral path");
        let error = tiered
            .persist(&active)
            .expect_err("snapshot write must not replace ephemeral mapping")
            .to_string();
        assert!(error.contains("active backing"), "{error}");
        let error = tiered
            .persist_to_mmap(&active)
            .expect_err("persistent adoption must not replace ephemeral mapping")
            .to_string();
        assert!(error.contains("active backing"), "{error}");
    }

    #[test]
    fn hard_link_alias_of_active_backing_is_rejected() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exact = tmp.path().join("exact.compact");
        let alias = tmp.path().join("hard-link.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered.persist_to_mmap(&exact).expect("persistent mapping");
        std::fs::hard_link(&exact, &alias).expect("hard-link active mapping");

        for error in [
            tiered
                .persist(&alias)
                .expect_err("persist hard-link alias")
                .to_string(),
            tiered
                .persist_to_mmap(&alias)
                .expect_err("mmap hard-link alias")
                .to_string(),
        ] {
            assert!(error.contains("aliases the active backing"), "{error}");
        }
        assert_eq!(tiered.path().as_deref(), Some(exact.as_path()));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_alias_of_active_backing_is_rejected() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().expect("tempdir");
        let exact = tmp.path().join("exact.compact");
        let alias = tmp.path().join("symlink.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered.persist_to_mmap(&exact).expect("persistent mapping");
        symlink(&exact, &alias).expect("symlink active mapping");

        let error = tiered
            .persist_to_mmap(&alias)
            .expect_err("mmap symlink alias")
            .to_string();
        assert!(error.contains("aliases the active backing"), "{error}");
        assert_eq!(tiered.path().as_deref(), Some(exact.as_path()));
    }

    #[cfg(unix)]
    #[test]
    fn same_exact_path_noop_rejects_replaced_file_identity() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exact = tmp.path().join("exact.compact");
        let displaced = tmp.path().join("displaced.compact");
        let replacement = tmp.path().join("replacement.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_store_with_nodes(2));
        tiered.persist_to_mmap(&exact).expect("persistent mapping");

        std::fs::rename(&exact, &displaced).expect("move mapped inode aside");
        CompactStoreTiered::new_in_memory(build_store_with_nodes(7))
            .persist(&replacement)
            .expect("serialize replacement");
        std::fs::rename(&replacement, &exact).expect("install replacement inode");

        let error = tiered
            .persist_to_mmap(&exact)
            .expect_err("same pathname with a different inode is not a no-op")
            .to_string();
        assert!(error.contains("different file"), "{error}");
        assert_eq!(tiered.store().node_count(), 2);
    }

    #[test]
    fn stale_rollback_token_cannot_overwrite_same_arc_at_newer_revision() {
        let shared = build_sample_store();
        let tiered = CompactStoreTiered::new_in_memory(Arc::clone(&shared));

        let stale = tiered
            .publish_prepared_reversibly(tiered.prepare_generation_in_memory(Arc::clone(&shared)));
        tiered
            .publish_prepared_reversibly(tiered.prepare_generation_in_memory(shared))
            .commit();

        assert!(
            !tiered.restore_published(stale),
            "pointer equality cannot authorize rollback across a later publication"
        );
    }

    #[test]
    fn relative_ephemeral_cleanup_is_anchored_before_cwd_changes() {
        const CHILD_ENV: &str = "GRAFEO_TIER_CWD_CLEANUP_CHILD";
        if let Some(root) = std::env::var_os(CHILD_ENV) {
            let root = PathBuf::from(root);
            let origin = root.join("origin");
            let elsewhere = root.join("elsewhere");
            std::fs::create_dir_all(&origin).expect("origin");
            std::fs::create_dir_all(&elsewhere).expect("elsewhere");
            std::env::set_current_dir(&origin).expect("enter origin");

            let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
            tiered
                .persist_to_ephemeral_mmap(Path::new("spill/base.compact"))
                .expect("relative spill");
            let generation = tiered.path().expect("absolute generation path");
            let resolved_origin = std::fs::canonicalize(&origin).expect("canonical origin");
            assert!(generation.is_absolute());
            assert!(generation.starts_with(&resolved_origin));
            assert!(generation.exists());

            std::env::set_current_dir(&elsewhere).expect("change cwd before drop");
            drop(tiered);
            assert!(
                !generation.exists(),
                "cleanup must remove the original absolute generation"
            );
            assert!(
                !elsewhere.join("spill").exists(),
                "cleanup must not resolve its path against the later cwd"
            );
            return;
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .arg("--exact")
            .arg("database::compact_tiered::tests::relative_ephemeral_cleanup_is_anchored_before_cwd_changes")
            .arg("--nocapture")
            .env(CHILD_ENV, tmp.path())
            .output()
            .expect("run cwd cleanup subprocess");
        assert!(
            output.status.success(),
            "cwd cleanup subprocess failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "cwd cleanup subprocess filter did not execute exactly one test:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    #[cfg(unix)]
    #[test]
    fn mapped_cleanup_skips_a_replacement_present_at_identity_check() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("spill.compact");
        let displaced = tmp.path().join("displaced-generation.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered
            .persist_to_ephemeral_mmap(&root)
            .expect("ephemeral mapping");
        let generation = tiered.path().expect("generation path");

        std::fs::rename(&generation, &displaced).expect("move mapped generation aside");
        std::fs::write(&generation, b"foreign replacement").expect("install replacement");
        drop(tiered);

        assert_eq!(
            std::fs::read(&generation).expect("replacement retained"),
            b"foreign replacement"
        );
        assert!(displaced.exists(), "moved mapped inode is not path-owned");
    }

    #[cfg(unix)]
    #[test]
    fn unpublished_guard_skips_a_replacement_present_at_identity_check() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("candidate.compact");
        let displaced = tmp.path().join("displaced-candidate.compact");
        let (candidate, file) =
            create_unique_sibling(&root, "identity-test").expect("reserve candidate");
        let path = candidate.path().to_path_buf();

        drop(file);
        std::fs::rename(&path, &displaced).expect("move candidate aside");
        std::fs::write(&path, b"foreign replacement").expect("install replacement");
        drop(candidate);

        assert_eq!(
            std::fs::read(&path).expect("replacement retained"),
            b"foreign replacement"
        );
        assert!(displaced.exists(), "moved candidate is not path-owned");
    }

    #[cfg(unix)]
    #[test]
    fn mapped_generation_rejects_valid_foreign_inode_substituted_after_create() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("candidate.compact");
        let displaced = tmp.path().join("displaced-created-inode.compact");
        let bytes = CompactStoreSection::new(build_sample_store())
            .serialize()
            .expect("serialize valid compact image");
        let mut replacement_path = None;

        let result = create_mapped_generation_after_close(&root, &bytes, |path| {
            replacement_path = Some(path.to_path_buf());
            std::fs::rename(path, &displaced).expect("move created inode aside");
            std::fs::write(path, &bytes).expect("install valid foreign compact image");
        });
        let error = match result {
            Ok(_) => panic!("identity substitution must fail before mmap publication"),
            Err(error) => error.to_string(),
        };

        let replacement = replacement_path.expect("replacement path captured");
        assert!(error.contains("file identity changed after creation"));
        assert_eq!(
            std::fs::read(&replacement).expect("foreign replacement retained"),
            bytes
        );
        assert!(
            displaced.exists(),
            "the displaced engine inode is no longer path-owned"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolved_io_path_is_immune_to_ancestor_symlink_retargeting() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().expect("tempdir");
        let original = tmp.path().join("original");
        let replacement = tmp.path().join("replacement");
        let link = tmp.path().join("current");
        std::fs::create_dir(&original).expect("original directory");
        std::fs::create_dir(&replacement).expect("replacement directory");
        symlink(&original, &link).expect("initial directory symlink");

        let caller = link.join("base.compact");
        let resolved = resolve_output_path(&caller).expect("resolve physical parent");
        std::fs::remove_file(&link).expect("remove initial symlink");
        symlink(&replacement, &link).expect("retarget directory symlink");
        write_atomically(&resolved.io_path, b"anchored").expect("write anchored path");

        assert_eq!(
            std::fs::read(original.join("base.compact")).expect("original target"),
            b"anchored"
        );
        assert!(
            !replacement.join("base.compact").exists(),
            "retargeted ancestor must not redirect resolved I/O"
        );
    }

    #[test]
    fn dropped_unpublished_ephemeral_generation_removes_candidate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("base.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_store_with_nodes(2));

        let prepared = tiered
            .prepare_generation_to_mmap(build_store_with_nodes(5), &root)
            .expect("prepare generation");
        let candidate = prepared_path(&prepared);
        assert!(candidate.exists());
        assert!(tiered.path().is_none());

        drop(prepared);
        assert!(
            !candidate.exists(),
            "unpublished mmap candidate must retire after its final slice drains"
        );
        assert!(tiered.path().is_none());
    }

    #[test]
    fn exact_install_failure_keeps_old_tier_and_cleans_unique_stage() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let destination_directory = tmp.path().join("not-a-file");
        std::fs::create_dir(&destination_directory).expect("destination directory");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        let old_store = tiered.store();

        let error = tiered
            .persist_to_mmap(&destination_directory)
            .expect_err("a file cannot atomically replace a directory")
            .to_string();

        assert!(
            error.contains("atomically install compact staging file"),
            "{error}"
        );
        assert!(Arc::ptr_eq(&tiered.store(), &old_store));
        assert!(!tiered.is_on_disk());
        assert!(tiered.path().is_none());
        assert!(generated_files(tmp.path()).is_empty());
        assert!(destination_directory.is_dir());
    }

    #[test]
    fn preexisting_exact_target_is_atomically_replaced_or_fails_closed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("existing.compact");
        let original = CompactStoreTiered::new_in_memory(build_store_with_nodes(2));
        original.persist(&path).expect("seed exact target");
        let original_bytes = std::fs::read(&path).expect("original bytes");

        let replacement = CompactStoreTiered::new_in_memory(build_store_with_nodes(5));
        let old_store = replacement.store();
        match replacement.persist_to_mmap(&path) {
            Ok(written) => {
                assert!(written > 0);
                assert!(replacement.is_on_disk());
                assert_eq!(replacement.path().as_deref(), Some(path.as_path()));
                assert_ne!(
                    std::fs::read(&path).expect("replacement bytes"),
                    original_bytes,
                    "successful installation must replace the complete image"
                );
                assert_eq!(replacement.store().node_count(), 5);
            }
            Err(error) => {
                // Windows and any filesystem that cannot atomically replace an
                // existing destination must fail without delete-then-rename.
                let message = error.to_string();
                assert!(
                    message.contains("atomically install compact staging file"),
                    "{message}"
                );
                assert_eq!(
                    std::fs::read(&path).expect("unchanged original bytes"),
                    original_bytes
                );
                assert!(!replacement.is_on_disk());
                assert!(replacement.path().is_none());
                assert!(Arc::ptr_eq(&replacement.store(), &old_store));
            }
        }
        assert!(
            generated_files(tmp.path()).is_empty(),
            "success transfers the stage and failure removes it"
        );
    }

    #[test]
    fn persistent_and_ephemeral_generation_roots_are_never_confused() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exact = tmp.path().join("persistent.compact");
        let root = tmp.path().join("cache.compact");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());

        tiered.persist_to_mmap(&exact).expect("persistent exact");
        assert_eq!(tiered.path().as_deref(), Some(exact.as_path()));
        assert!(tiered.generation_root().is_none());

        tiered
            .persist_to_ephemeral_mmap(&root)
            .expect("engine ephemeral");
        let ephemeral = tiered.path().expect("ephemeral path");
        let resolved_root = std::fs::canonicalize(tmp.path())
            .expect("canonical tempdir")
            .join("cache.compact");
        assert_ne!(ephemeral, root);
        assert_ne!(ephemeral, exact);
        assert_eq!(
            tiered.generation_root().as_deref(),
            Some(resolved_root.as_path())
        );
        assert!(exact.exists(), "engine rotation must retain caller file");

        tiered.reload_to_ram().expect("reload ephemeral generation");
        assert!(!ephemeral.exists());
        assert!(exact.exists());
    }

    /// Phase 3c: column data on the disk tier should be served from
    /// the mmap-backed `Bytes` rather than from a heap copy. We can't
    /// directly assert "no allocation happened" in a portable way, but
    /// we can prove the column codec storage shares the mmap refcount:
    /// if we drop the tiered wrapper, the underlying `Mmap` should
    /// still be live as long as we hold an `Arc<CompactStore>` whose
    /// codec storage references it.
    ///
    /// This test exercises the full open-mmap path and reads several
    /// values. Combined with the column codec's `from_bytes_storage`
    /// constructors using `data.slice(range)`, it confirms the
    /// zero-copy contract end-to-end.
    #[test]
    fn mmap_backed_store_serves_reads_from_mapped_bytes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("zerocopy.compact");

        // Build and persist, then retain only the published store Arc. Its
        // codec slices keep the mapping alive after the wrapper drops, while
        // the caller-owned exact file itself remains persistent.
        let store = {
            let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
            tiered.persist_to_mmap(&path).expect("persist_to_mmap");
            tiered.store()
        };
        assert!(path.exists());

        let person_ids = store.nodes_by_label("Person");
        assert!(!person_ids.is_empty());

        // Reads work; values come from the mmap-backed Bytes via
        // `data.slice(range)` constructors in `read_from_v3`.
        for &id in person_ids.iter().take(8) {
            let name = store
                .get_node_property(id, &PropertyKey::new("name"))
                .expect("name property exists");
            assert!(matches!(name, Value::String(_)));
            let age = store
                .get_node_property(id, &PropertyKey::new("age"))
                .expect("age property exists");
            assert!(matches!(age, Value::Int64(_)));
        }
        drop(store);
        assert!(
            path.exists(),
            "caller-owned exact file must outlive readers"
        );
        std::fs::remove_file(path).expect("caller removes persistent exact file");
    }
}
