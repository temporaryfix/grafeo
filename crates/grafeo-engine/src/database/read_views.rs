//! Read-only views over database-owned storage machinery.
//!
//! A persistent [`GrafeoDB`](super::GrafeoDB) owns several concrete handles
//! whose maintenance APIs can replace stores, publish tombstones, or write
//! container bytes. Returning those handles directly would let downstream
//! code bypass the Session/WAL publication protocol. These views retain an
//! `Arc` so they remain useful across engine topology changes, while exposing
//! only observation and immutable graph data.

#[cfg(any(
    feature = "grafeo-file",
    all(feature = "compact-store", feature = "lpg")
))]
use std::sync::Arc;

#[cfg(all(feature = "compact-store", feature = "lpg"))]
use grafeo_common::types::{EdgeId, EpochId, NodeId};
#[cfg(feature = "grafeo-file")]
use grafeo_common::{storage::SectionDirectoryEntry, utils::error::Result};
#[cfg(all(feature = "compact-store", feature = "lpg"))]
use grafeo_core::graph::GraphStoreSearch;
#[cfg(all(feature = "compact-store", feature = "lpg"))]
use grafeo_core::graph::compact::{CompactStore, layered::LayeredStore};
#[cfg(feature = "grafeo-file")]
use grafeo_storage::{
    container::SectionDirectory,
    file::{DbHeader, FileHeader, GrafeoFileManager},
};

#[cfg(all(feature = "compact-store", feature = "lpg"))]
/// An immutable handle to a database's compact base and layered read path.
///
/// The concrete `LayeredStore` also implements `GraphStoreMut` and exposes
/// maintenance operations such as base replacement, overlay reset, tombstone
/// seeding, and merge. This view intentionally exposes none of them.
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// let mut db = GrafeoDB::new_in_memory();
/// db.compact().unwrap();
/// db.layered_store().unwrap().reset_overlay();
/// ```
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// # let mut db = GrafeoDB::new_in_memory();
/// # db.compact().unwrap();
/// let view = db.layered_store().unwrap();
/// view.seed_deleted_from_base([], []);
/// ```
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// # let mut db = GrafeoDB::new_in_memory();
/// # db.compact().unwrap();
/// let base = db.layered_store().unwrap().base_store();
/// db.layered_store().unwrap().swap_base(base);
/// ```
#[derive(Clone)]
pub struct LayeredStoreView {
    inner: Arc<LayeredStore>,
}

#[cfg(all(feature = "compact-store", feature = "lpg"))]
impl LayeredStoreView {
    pub(crate) fn new(inner: Arc<LayeredStore>) -> Self {
        Self { inner }
    }

    /// Returns the immutable compact base currently serving cold reads.
    #[must_use]
    pub fn base_store(&self) -> Arc<CompactStore> {
        self.inner.base_store_arc()
    }

    /// Returns the complete read-only graph interface for the layered store.
    #[must_use]
    pub fn graph_store(&self) -> &dyn GraphStoreSearch {
        self.inner.as_ref()
    }

    /// Number of modified, created, or deleted entities in the overlay.
    #[must_use]
    pub fn overlay_mutation_count(&self) -> usize {
        self.inner.overlay_mutation_count()
    }

    /// Number of live nodes in the mutable overlay.
    #[must_use]
    pub fn overlay_node_count(&self) -> usize {
        self.inner.overlay_store().node_count()
    }

    /// Number of live edges in the mutable overlay.
    #[must_use]
    pub fn overlay_edge_count(&self) -> usize {
        self.inner.overlay_store().edge_count()
    }

    /// Current overlay epoch.
    #[must_use]
    pub fn overlay_epoch(&self) -> EpochId {
        self.inner.overlay_store().current_epoch()
    }

    /// Approximate heap bytes used by the overlay.
    #[must_use]
    pub fn overlay_memory_bytes(&self) -> usize {
        self.inner.overlay_memory_bytes()
    }

    /// Approximate heap bytes used by both layers.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Snapshot of committed base-node tombstones.
    #[must_use]
    pub fn deleted_nodes(&self) -> Vec<(NodeId, EpochId)> {
        self.inner.snapshot_deleted_nodes()
    }

    /// Snapshot of committed base-edge tombstones.
    #[must_use]
    pub fn deleted_edges(&self) -> Vec<(EdgeId, EpochId)> {
        self.inner.snapshot_deleted_edges()
    }

    /// Whether tombstone state differs from the last persisted cut.
    #[must_use]
    pub fn deletions_dirty(&self) -> bool {
        self.inner.deletions_dirty()
    }

    /// Returns whether two views refer to the same installed layered store.
    #[must_use]
    pub fn same_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[cfg(all(feature = "compact-store", feature = "lpg"))]
impl std::fmt::Debug for LayeredStoreView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredStoreView")
            .field("overlay_mutation_count", &self.overlay_mutation_count())
            .field("memory_bytes", &self.memory_bytes())
            .finish_non_exhaustive()
    }
}

#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
/// An immutable handle to a database's compact-store tier state.
///
/// Tier publication (`persist`, `persist_to_mmap`, and `reload_to_ram`) stays
/// private to the engine's maintenance path.
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// # let mut db = GrafeoDB::new_in_memory();
/// # db.compact().unwrap();
/// db.compact_tiered()
///     .unwrap()
///     .persist_to_mmap(std::path::Path::new("outside.grafeo"))
///     .unwrap();
/// ```
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// # let mut db = GrafeoDB::new_in_memory();
/// # db.compact().unwrap();
/// db.compact_tiered().unwrap().reload_to_ram().unwrap();
/// ```
#[derive(Clone)]
pub struct CompactStoreTieredView {
    inner: Arc<super::compact_tiered::CompactStoreTiered>,
    layered: Arc<LayeredStore>,
}

#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
impl CompactStoreTieredView {
    pub(crate) fn new(
        inner: Arc<super::compact_tiered::CompactStoreTiered>,
        layered: Arc<LayeredStore>,
    ) -> Self {
        Self { inner, layered }
    }

    /// Returns the immutable compact store currently serving reads.
    #[must_use]
    pub fn store(&self) -> Arc<CompactStore> {
        self.layered.base_store_arc()
    }

    /// Returns `true` when the compact base is mmap-backed.
    #[must_use]
    pub fn is_on_disk(&self) -> bool {
        self.layered.with_base_generation(|base, _| {
            self.inner
                .snapshot_for_store(&base)
                .is_some_and(|tier| tier.is_on_disk)
        })
    }

    /// Returns the mmap backing path, when present.
    #[must_use]
    pub fn path(&self) -> Option<std::path::PathBuf> {
        self.layered.with_base_generation(|base, _| {
            self.inner
                .snapshot_for_store(&base)
                .and_then(|tier| tier.path)
        })
    }

    /// Approximate heap bytes used by the compact base.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.store().memory_bytes()
    }

    /// Returns whether two views refer to the same installed tier wrapper.
    #[must_use]
    pub fn same_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
impl std::fmt::Debug for CompactStoreTieredView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactStoreTieredView")
            .field("is_on_disk", &self.is_on_disk())
            .field("path", &self.path())
            .field("memory_bytes", &self.memory_bytes())
            .finish()
    }
}

#[cfg(feature = "grafeo-file")]
/// Read-only access to a database's single-file container.
///
/// Checkpoint publication, WAL removal, file replacement, syncing, copying,
/// and close remain engine operations. This view supports only observation and
/// verified reads from the current installed container.
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// # let db = GrafeoDB::new_in_memory();
/// db.file_manager().unwrap().write_snapshot(&[], 0, 0, 0, 0).unwrap();
/// ```
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// # let db = GrafeoDB::new_in_memory();
/// db.file_manager().unwrap().remove_sidecar_wal().unwrap();
/// ```
///
/// ```compile_fail
/// # use grafeo_engine::GrafeoDB;
/// # let db = GrafeoDB::new_in_memory();
/// db.file_manager().unwrap().close().unwrap();
/// ```
#[derive(Clone)]
pub struct DatabaseFileView {
    inner: Arc<GrafeoFileManager>,
}

#[cfg(feature = "grafeo-file")]
impl DatabaseFileView {
    pub(crate) fn new(inner: Arc<GrafeoFileManager>) -> Self {
        Self { inner }
    }

    /// Returns whether the underlying database was opened read-only.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
    }

    /// On-disk graph-model tag (`0` LPG, `1` RDF, `2` both).
    #[must_use]
    pub fn graph_model_tag(&self) -> u8 {
        self.inner.graph_model_tag()
    }

    /// Reads and verifies a legacy snapshot payload.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking to or reading the active snapshot fails, or
    /// if the payload's CRC-32 does not match the active database header.
    pub fn read_snapshot(&self) -> Result<Vec<u8>> {
        self.inner.read_snapshot()
    }

    /// Returns the sidecar WAL path without modifying it.
    #[must_use]
    pub fn sidecar_wal_path(&self) -> std::path::PathBuf {
        self.inner.sidecar_wal_path()
    }

    /// Returns whether the sidecar WAL currently exists.
    #[must_use]
    pub fn has_sidecar_wal(&self) -> bool {
        self.inner.has_sidecar_wal()
    }

    /// Returns the database file path.
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        self.inner.path()
    }

    /// Returns the currently active database header.
    #[must_use]
    pub fn active_header(&self) -> DbHeader {
        self.inner.active_header()
    }

    /// Returns the immutable file header.
    #[must_use]
    pub fn file_header(&self) -> &FileHeader {
        self.inner.file_header()
    }

    /// Returns the current file size.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system cannot read metadata for the
    /// open database file.
    pub fn file_size(&self) -> Result<u64> {
        self.inner.file_size()
    }

    /// Reads and verifies the installed section directory.
    ///
    /// # Errors
    ///
    /// Returns an error if file metadata, seeking, or reading fails; if a v2
    /// header is installed but the file is too short for its fixed directory
    /// page; if the directory cannot be parsed; or if its CRC-32 does not match
    /// the active database header.
    pub fn read_section_directory(&self) -> Result<Option<SectionDirectory>> {
        self.inner.read_section_directory()
    }

    /// Reads and verifies one section payload.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking to or reading the directory entry's byte
    /// range fails, if the payload CRC-32 does not match the entry, or, when
    /// encryption is enabled, if the payload cannot be decrypted with the
    /// installed section key.
    pub fn read_section_data(&self, entry: &SectionDirectoryEntry) -> Result<Vec<u8>> {
        self.inner.read_section_data(entry)
    }
}

#[cfg(feature = "grafeo-file")]
impl std::fmt::Debug for DatabaseFileView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatabaseFileView")
            .field("path", &self.path())
            .field("read_only", &self.is_read_only())
            .field("graph_model_tag", &self.graph_model_tag())
            .finish_non_exhaustive()
    }
}
