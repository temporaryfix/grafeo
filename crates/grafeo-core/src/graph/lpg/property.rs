//! Columnar property storage for nodes and edges.
//!
//! Properties are stored column-wise (all "name" values together, all "age"
//! values together) rather than row-wise. This makes filtering fast - to find
//! all nodes where age > 30, we only scan the age column.
//!
//! Each column also maintains a zone map (min/max/null_count) enabling the
//! query optimizer to skip columns entirely when a predicate can't match.
//!
//! ## Compression
//!
//! Columns can be compressed to save memory. When compression is enabled,
//! the column automatically selects the best codec based on the data type:
//!
//! | Data type | Codec | Typical savings |
//! |-----------|-------|-----------------|
//! | Int64 (sorted) | DeltaBitPacked | 5-20x |
//! | Int64 (small) | BitPacked | 2-16x |
//! | Int64 (repeated) | RunLength | 2-100x |
//! | String (low cardinality) | Dictionary | 2-50x |
//! | Bool | BitVector | 8x |

use crate::codec::CompressionCodec;
use crate::index::zone_map::ZoneMapEntry;
use grafeo_common::temporal::VersionLog;
use grafeo_common::types::EpochId;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
#[cfg(feature = "lpg")]
use parking_lot::RwLockWriteGuard;
use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "lpg")]
pub(crate) mod commit;

/// Compression mode for property columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CompressionMode {
    /// Never compress - always use sparse HashMap (default).
    #[default]
    None,
    /// Automatically compress when beneficial (after threshold).
    Auto,
    /// Eagerly compress on every flush.
    Eager,
}

/// Comparison operators used for zone map predicate checks.
///
/// These map directly to GQL comparison operators like `=`, `<`, `>=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompareOp {
    /// Equal to value.
    Eq,
    /// Not equal to value.
    Ne,
    /// Less than value.
    Lt,
    /// Less than or equal to value.
    Le,
    /// Greater than value.
    Gt,
    /// Greater than or equal to value.
    Ge,
}

/// Trait for IDs that can key into property storage.
///
/// Implemented for [`NodeId`] and [`EdgeId`] - you can store properties on both.
/// Provides safe conversions to/from `u64` for compression, replacing unsafe transmute.
pub trait EntityId: Copy + Eq + Hash + 'static {
    /// Returns the raw `u64` value.
    fn as_u64(self) -> u64;
    /// Creates an ID from a raw `u64` value.
    fn from_u64(v: u64) -> Self;
}

impl EntityId for NodeId {
    #[inline]
    fn as_u64(self) -> u64 {
        self.0
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        Self(v)
    }
}

impl EntityId for EdgeId {
    #[inline]
    fn as_u64(self) -> u64 {
        self.0
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        Self(v)
    }
}

/// Thread-safe columnar property storage.
///
/// Each property key ("name", "age", etc.) gets its own column. This layout
/// is great for analytical queries that filter on specific properties -
/// you only touch the columns you need.
///
/// Generic over `Id` so the same storage works for nodes and edges.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::PropertyStorage;
/// use grafeo_common::types::{EpochId, NodeId, PropertyKey};
///
/// let storage = PropertyStorage::new();
/// let alix = NodeId::new(1);
///
/// storage.set(alix, PropertyKey::new("name"), "Alix".into(), EpochId::INITIAL);
/// storage.set(alix, PropertyKey::new("age"), 30i64.into(), EpochId::INITIAL);
///
/// // Fetch all properties at once
/// let props = storage.get_all(alix);
/// assert_eq!(props.len(), 2);
/// ```
pub struct PropertyStorage<Id: EntityId = NodeId> {
    /// Map from property key to column.
    /// Lock order: 9 (nested, acquired via LpgStore::node_properties/edge_properties)
    columns: RwLock<FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    /// Default compression mode for new columns.
    default_compression: CompressionMode,
    _marker: PhantomData<Id>,
}

/// Intrinsically paired column postimage; the source retains displaced buffers.
#[cfg(feature = "lpg")]
pub(crate) struct PreparedPropertyRestore<'target, 'source, Id: EntityId> {
    columns: RwLockWriteGuard<'target, FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    source: &'source mut PropertyStorage<Id>,
}

#[cfg(feature = "lpg")]
impl<Id: EntityId> PreparedPropertyRestore<'_, '_, Id> {
    pub(crate) fn install(&mut self) {
        std::mem::swap(&mut *self.columns, self.source.columns.get_mut());
    }
}

/// Write-locked, allocation-complete physical history removal.
/// Dropping aborts unchanged; commit only removes already-selected keys.
#[cfg(feature = "lpg")]
pub(crate) struct PreparedPropertyHistoryPurge<'a, Id: EntityId> {
    columns: RwLockWriteGuard<'a, FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    ids: Vec<Id>,
}

#[cfg(feature = "lpg")]
impl<Id: EntityId> PreparedPropertyHistoryPurge<'_, Id> {
    pub(crate) fn commit(mut self) {
        for column in self.columns.values_mut() {
            let mut changed = false;
            for id in &self.ids {
                changed |= column.values.remove(id).is_some();
            }
            if changed {
                column.zone_map_dirty = true;
                column.block_zone_maps.clear();
            }
        }
    }
}

impl<Id: EntityId> PropertyStorage<Id> {
    #[cfg(feature = "lpg")]
    pub(crate) fn is_committed_restore_image(&self, frontier: EpochId) -> bool {
        self.columns.read().values().all(|column| {
            column.values.values().all(|history| {
                history
                    .iter()
                    .all(|(epoch, _)| *epoch != EpochId::PENDING && *epoch <= frontier)
            })
        })
    }

    /// Reserves the actual empty column directory, without copying its image.
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_pristine_restore<'target, 'source>(
        &'target self,
        source: &'source mut Self,
    ) -> Option<PreparedPropertyRestore<'target, 'source, Id>> {
        let prepared = self.prepare_replacement(source)?;
        if !prepared.columns.is_empty() {
            return None;
        }
        Some(prepared)
    }

    /// Retains the actual populated column writer through a complete image swap.
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_replacement<'target, 'source>(
        &'target self,
        source: &'source mut Self,
    ) -> Option<PreparedPropertyRestore<'target, 'source, Id>> {
        let columns = self.columns.try_write()?;
        if self.default_compression != source.default_compression {
            return None;
        }
        Some(PreparedPropertyRestore { columns, source })
    }

    /// Creates a new property storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: RwLock::new(FxHashMap::default()),
            default_compression: CompressionMode::None,
            _marker: PhantomData,
        }
    }

    /// Creates a new property storage with compression enabled.
    #[must_use]
    pub fn with_compression(mode: CompressionMode) -> Self {
        Self {
            columns: RwLock::new(FxHashMap::default()),
            default_compression: mode,
            _marker: PhantomData,
        }
    }

    /// Sets the default compression mode for new columns.
    pub fn set_default_compression(&mut self, mode: CompressionMode) {
        self.default_compression = mode;
    }

    /// Sets a property value for an entity at a specific epoch.
    ///
    /// For non-transactional writes, pass the current epoch.
    /// For transactional writes, pass `EpochId::PENDING`.
    pub fn set(&self, id: Id, key: PropertyKey, value: Value, epoch: EpochId) {
        let mut columns = self.columns.write();
        let mode = self.default_compression;
        columns
            .entry(key)
            .or_insert_with(|| PropertyColumn::with_compression(mode))
            .set(id, value, epoch);
    }

    /// Applies a batch of property ops under a SINGLE write lock, so a
    /// concurrent reader (`get_all`/`get_all_at`) observes them atomically —
    /// all-or-nothing — rather than a half-applied state.
    ///
    /// `Some(value)` sets, `None` removes (tombstone at `epoch`). Used by commit
    /// (`apply_tx_overlay`) to make a transaction's property changes appear
    /// atomically, which closes the commit torn-read window: a reader can no
    /// longer slip in between two of a commit's per-property writes.
    pub fn apply_ops<I>(&self, ops: I, epoch: EpochId)
    where
        I: IntoIterator<Item = (Id, PropertyKey, Option<Value>)>,
    {
        let mut columns = self.columns.write();
        let mode = self.default_compression;
        for (id, key, value) in ops {
            match value {
                Some(v) => columns
                    .entry(key)
                    .or_insert_with(|| PropertyColumn::with_compression(mode))
                    .set(id, v, epoch),
                None => {
                    if let Some(col) = columns.get_mut(&key) {
                        col.remove(id, epoch);
                    }
                }
            }
        }
    }

    /// Enables compression for a specific column.
    pub fn enable_compression(&self, key: &PropertyKey, mode: CompressionMode) {
        let mut columns = self.columns.write();
        if let Some(col) = columns.get_mut(key) {
            col.set_compression_mode(mode);
        }
    }

    /// Compresses all columns that have compression enabled.
    pub fn compress_all(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            if col.compression_mode() != CompressionMode::None {
                col.compress();
            }
        }
    }

    /// Forces compression on all columns regardless of mode.
    pub fn force_compress_all(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.force_compress();
        }
    }

    /// Returns compression statistics for all columns.
    #[must_use]
    pub fn compression_stats(&self) -> FxHashMap<PropertyKey, CompressionStats> {
        let columns = self.columns.read();
        columns
            .iter()
            .map(|(key, col)| (key.clone(), col.compression_stats()))
            .collect()
    }

    /// Returns the total memory usage of all columns (compressed size estimate).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let columns = self.columns.read();
        columns
            .values()
            .map(|col| col.compression_stats().compressed_size)
            .sum()
    }

    /// Returns estimated heap memory for all columns including hash map overhead.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        let columns = self.columns.read();
        // Outer hash map capacity
        let map_overhead = columns.capacity()
            * (std::mem::size_of::<PropertyKey>() + std::mem::size_of::<PropertyColumn<Id>>() + 1);
        // Sum of all column heap memory
        let column_bytes: usize = columns.values().map(|col| col.heap_memory_bytes()).sum();
        map_overhead + column_bytes
    }

    /// Gets a property value for an entity.
    #[must_use]
    pub fn get(&self, id: Id, key: &PropertyKey) -> Option<Value> {
        let columns = self.columns.read();
        columns.get(key).and_then(|col| col.get(id))
    }

    /// Removes a property value for an entity (temporal: appends tombstone at epoch).
    pub fn remove(&self, id: Id, key: &PropertyKey, epoch: EpochId) -> Option<Value> {
        let mut columns = self.columns.write();
        columns.get_mut(key).and_then(|col| col.remove(id, epoch))
    }

    /// Removes all properties for an entity (temporal: tombstones at current epoch).
    pub fn remove_all(&self, id: Id, epoch: EpochId) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.remove(id, epoch);
        }
    }

    /// Validates that every selected history is committed at or before
    /// `boundary`, then prepares an allocation-free physical purge and retains
    /// the property write guard across representation publication.
    #[cfg(feature = "lpg")]
    pub(crate) fn can_purge_all_history(&self, ids: &[Id], boundary: EpochId) -> bool {
        let columns = self.columns.read();
        !columns.values().any(|column| {
            ids.iter().any(|id| {
                column.values.get(id).is_some_and(|log| {
                    log.history()
                        .iter()
                        .any(|(epoch, _)| *epoch == EpochId::PENDING || *epoch > boundary)
                })
            })
        })
    }

    /// Revalidates [`Self::can_purge_all_history`] under the retained write
    /// guard and prepares an allocation-free physical purge.
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_purge_all_history(
        &self,
        ids: &[Id],
        boundary: EpochId,
    ) -> Option<PreparedPropertyHistoryPurge<'_, Id>> {
        let columns = self.columns.write();
        if columns.values().any(|column| {
            ids.iter().any(|id| {
                column.values.get(id).is_some_and(|log| {
                    log.history()
                        .iter()
                        .any(|(epoch, _)| *epoch == EpochId::PENDING || *epoch > boundary)
                })
            })
        }) {
            return None;
        }
        Some(PreparedPropertyHistoryPurge {
            columns,
            ids: ids.to_vec(),
        })
    }

    /// Gets all properties for an entity.
    #[must_use]
    pub fn get_all(&self, id: Id) -> FxHashMap<PropertyKey, Value> {
        let columns = self.columns.read();
        let mut result = FxHashMap::default();
        for (key, col) in columns.iter() {
            if let Some(value) = col.get(id) {
                result.insert(key.clone(), value);
            }
        }
        result
    }

    /// Gets property values for multiple entities in a single lock acquisition.
    ///
    /// More efficient than calling [`Self::get`] in a loop because it acquires
    /// the read lock only once.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::PropertyStorage;
    /// use grafeo_common::types::{PropertyKey, Value};
    /// use grafeo_common::NodeId;
    ///
    /// let storage: PropertyStorage<NodeId> = PropertyStorage::new();
    /// let key = PropertyKey::new("age");
    /// let ids = vec![NodeId(1), NodeId(2), NodeId(3)];
    /// let values = storage.get_batch(&ids, &key);
    /// // values[i] is the property value for ids[i], or None if not set
    /// ```
    #[must_use]
    pub fn get_batch(&self, ids: &[Id], key: &PropertyKey) -> Vec<Option<Value>> {
        let columns = self.columns.read();
        match columns.get(key) {
            Some(col) => ids.iter().map(|&id| col.get(id)).collect(),
            None => vec![None; ids.len()],
        }
    }

    /// Gets all properties for multiple entities efficiently.
    ///
    /// More efficient than calling [`Self::get_all`] in a loop because it
    /// acquires the read lock only once.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::PropertyStorage;
    /// use grafeo_common::types::{PropertyKey, Value};
    /// use grafeo_common::NodeId;
    ///
    /// let storage: PropertyStorage<NodeId> = PropertyStorage::new();
    /// let ids = vec![NodeId(1), NodeId(2)];
    /// let all_props = storage.get_all_batch(&ids);
    /// // all_props[i] is a HashMap of all properties for ids[i]
    /// ```
    #[must_use]
    pub fn get_all_batch(&self, ids: &[Id]) -> Vec<FxHashMap<PropertyKey, Value>> {
        let columns = self.columns.read();
        let column_count = columns.len();

        // Pre-allocate result vector with exact capacity (NebulaGraph pattern)
        let mut results = Vec::with_capacity(ids.len());

        for &id in ids {
            // Pre-allocate HashMap with expected column count
            let mut result = FxHashMap::with_capacity_and_hasher(column_count, Default::default());
            for (key, col) in columns.iter() {
                if let Some(value) = col.get(id) {
                    result.insert(key.clone(), value);
                }
            }
            results.push(result);
        }

        results
    }

    /// Gets selected properties for multiple entities efficiently (projection pushdown).
    ///
    /// This is more efficient than [`Self::get_all_batch`] when you only need a subset
    /// of properties - it only iterates the requested columns instead of all columns.
    ///
    /// **Performance**: O(N × K) where N = ids.len() and K = keys.len(),
    /// compared to O(N × C) for `get_all_batch` where C = total column count.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::PropertyStorage;
    /// use grafeo_common::types::{PropertyKey, Value};
    /// use grafeo_common::NodeId;
    ///
    /// let storage: PropertyStorage<NodeId> = PropertyStorage::new();
    /// let ids = vec![NodeId::new(1), NodeId::new(2)];
    /// let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    ///
    /// // Only fetches "name" and "age" columns, ignoring other properties
    /// let props = storage.get_selective_batch(&ids, &keys);
    /// ```
    #[must_use]
    pub fn get_selective_batch(
        &self,
        ids: &[Id],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        if keys.is_empty() {
            // No properties requested - return empty maps
            return vec![FxHashMap::default(); ids.len()];
        }

        let columns = self.columns.read();

        // Pre-collect only the columns we need (avoids re-lookup per id)
        let requested_columns: Vec<_> = keys
            .iter()
            .filter_map(|key| columns.get(key).map(|col| (key, col)))
            .collect();

        // Pre-allocate result with exact capacity
        let mut results = Vec::with_capacity(ids.len());

        for &id in ids {
            let mut result =
                FxHashMap::with_capacity_and_hasher(requested_columns.len(), Default::default());
            // Only iterate requested columns, not all columns
            for (key, col) in &requested_columns {
                if let Some(value) = col.get(id) {
                    result.insert((*key).clone(), value);
                }
            }
            results.push(result);
        }

        results
    }

    /// Returns the number of property columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.read().len()
    }

    /// Returns the keys of all columns.
    #[must_use]
    pub fn keys(&self) -> Vec<PropertyKey> {
        self.columns.read().keys().cloned().collect()
    }

    /// Removes all property data.
    pub fn clear(&self) {
        self.columns.write().clear();
    }

    /// Gets a column by key for bulk access.
    #[must_use]
    pub fn column(&self, key: &PropertyKey) -> Option<PropertyColumnRef<'_, Id>> {
        let columns = self.columns.read();
        if columns.contains_key(key) {
            Some(PropertyColumnRef {
                _guard: columns,
                _key: key.clone(),
                _marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// Checks if a predicate might match any values (using zone maps).
    ///
    /// Returns `false` only when we're *certain* no values match - for example,
    /// if you're looking for age > 100 but the max age is 80. Returns `true`
    /// if the property doesn't exist (conservative - might match).
    #[must_use]
    pub fn might_match(&self, key: &PropertyKey, op: CompareOp, value: &Value) -> bool {
        let columns = self.columns.read();
        columns
            .get(key)
            .map_or(true, |col| col.might_match(op, value)) // No column = assume might match (conservative)
    }

    /// Gets the zone map for a property column.
    #[must_use]
    pub fn zone_map(&self, key: &PropertyKey) -> Option<ZoneMapEntry> {
        let columns = self.columns.read();
        columns.get(key).map(|col| col.zone_map().clone())
    }

    /// Returns the per-block zone maps for a property column, if any.
    ///
    /// Returns `None` when the column doesn't exist; returns `Some(empty)`
    /// when the column exists but is uncompressed (the hot buffer is
    /// unordered, so per-block pruning is meaningless there). Phase 4 will
    /// treat "no per-block stats" as "fall back to the column-level zone
    /// map".
    ///
    /// **Temporal mode:** always returns `Some(empty)` for any existing
    /// column. Compression is disabled for `VersionLog`-backed columns,
    /// so there is no sorted compressed array to chunk into blocks. Use
    /// the column-level [`zone_map`](Self::zone_map) instead.
    #[must_use]
    pub fn block_zone_maps_for(&self, key: &PropertyKey) -> Option<Vec<ZoneMapEntry>> {
        let columns = self.columns.read();
        columns.get(key).map(|col| col.block_zone_maps().to_vec())
    }

    /// Checks if a range predicate might match any values (using zone maps).
    ///
    /// Returns `false` only when we're *certain* no values match the range.
    /// Returns `true` if the property doesn't exist (conservative - might match).
    #[must_use]
    pub fn might_match_range(
        &self,
        key: &PropertyKey,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> bool {
        let columns = self.columns.read();
        columns.get(key).map_or(true, |col| {
            // Prepared publication can move values outside the old bounds or
            // add a column whose zone map is still empty. Dirty metadata is
            // not evidence that a range has no matches.
            col.zone_map_dirty
                || col
                    .zone_map()
                    .might_contain_range(min, max, min_inclusive, max_inclusive)
        }) // No column = assume might match (conservative)
    }

    /// Rebuilds zone maps for all columns (call after bulk removes).
    pub fn rebuild_zone_maps(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.rebuild_zone_map();
        }
    }
}

impl<Id: EntityId> Default for PropertyStorage<Id> {
    fn default() -> Self {
        Self::new()
    }
}

// === Temporal-only methods for PropertyStorage ===
impl<Id: EntityId> PropertyStorage<Id> {
    /// Returns a write guard to the columns map for targeted rollback.
    #[cfg(feature = "lpg")]
    pub(crate) fn columns_write(
        &self,
    ) -> parking_lot::RwLockWriteGuard<'_, FxHashMap<PropertyKey, PropertyColumn<Id>>> {
        self.columns.write()
    }

    /// Physically removes one unpublished identity from every property column.
    ///
    /// This rollback-only seam is deliberately single-identity and in-place:
    /// it performs no receipt allocation and cannot fail, so an unwind guard
    /// may safely call it while restoring a failed representation promotion.
    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    pub(crate) fn purge_identity_in_place(&self, id: Id) -> bool {
        let mut removed = false;
        let mut columns = self.columns.write();
        for column in columns.values_mut() {
            if column.values.remove(&id).is_some() {
                removed = true;
                column.zone_map_dirty = true;
                column.block_zone_maps.clear();
            }
        }
        removed
    }

    /// Gets a property value at a specific epoch.
    #[must_use]
    pub fn get_at(&self, id: Id, key: &PropertyKey, epoch: EpochId) -> Option<Value> {
        let columns = self.columns.read();
        columns.get(key).and_then(|col| col.get_at(id, epoch))
    }

    /// Gets all properties for an entity at a specific epoch.
    #[must_use]
    pub fn get_all_at(&self, id: Id, epoch: EpochId) -> FxHashMap<PropertyKey, Value> {
        let columns = self.columns.read();
        let mut result = FxHashMap::default();
        for (key, col) in columns.iter() {
            if let Some(value) = col.get_at(id, epoch) {
                result.insert(key.clone(), value);
            }
        }
        result
    }

    /// Replaces PENDING epochs with the real commit epoch in all columns.
    pub fn finalize_pending(&self, real_epoch: EpochId) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.finalize_pending(real_epoch);
        }
    }

    /// Removes all PENDING entries from all columns (transaction rollback).
    pub fn remove_pending(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.remove_pending();
        }
    }

    /// Garbage-collects old versions from all columns.
    pub fn gc(&self, min_epoch: EpochId) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.gc(min_epoch);
        }
    }

    /// Drops histories whose committed tombstone is visible at the GC floor.
    ///
    /// The caller must have no underlying property layer: an absent history
    /// must not expose an older base value that this tombstone was masking.
    #[cfg(feature = "lpg")]
    pub(crate) fn gc_expired_tombstones(&self, min_epoch: EpochId) {
        if min_epoch == EpochId::PENDING {
            return;
        }
        let mut columns = self.columns.write();
        for column in columns.values_mut() {
            let before = column.values.len();
            column.values.retain(|_, log| {
                !matches!(log.latest_entry(), Some((epoch, Value::Null)) if *epoch <= min_epoch)
            });
            if column.values.len() != before {
                column.zone_map_dirty = true;
                column.block_zone_maps.clear();
            }
        }
    }

    /// Returns the full version history for all properties of an entity.
    ///
    /// Each entry is `(key, Vec<(epoch, value)>)`. Useful for snapshot
    /// export that preserves temporal history.
    #[must_use]
    pub fn get_all_history(&self, id: Id) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let columns = self.columns.read();
        let mut result = Vec::new();
        for (key, col) in columns.iter() {
            if let Some(log) = col.values.get(&id) {
                let entries: Vec<(EpochId, Value)> = log
                    .history()
                    .iter()
                    .map(|(epoch, value)| (*epoch, value.clone()))
                    .collect();
                if !entries.is_empty() {
                    result.push((key.clone(), entries));
                }
            }
        }
        result
    }

    /// Returns the version history for a single property of an entity.
    ///
    /// More efficient than `get_all_history` when only one property is needed.
    #[must_use]
    pub fn get_history(&self, id: Id, key: &PropertyKey) -> Vec<(EpochId, Value)> {
        let columns = self.columns.read();
        columns
            .get(key)
            .and_then(|col| col.values.get(&id))
            .map(|log| log.history().iter().map(|(e, v)| (*e, v.clone())).collect())
            .unwrap_or_default()
    }

    /// Returns the complete version history for every identity in one column.
    ///
    /// This is used by detached index rebuilds to enumerate identities whose
    /// current value is a tombstone or whose node has since been deleted.  It
    /// intentionally exposes no write guard and performs one column read.
    #[cfg(feature = "lpg")]
    pub(crate) fn column_history(&self, key: &PropertyKey) -> Vec<(Id, Vec<(EpochId, Value)>)> {
        let columns = self.columns.read();
        columns
            .get(key)
            .map(|column| {
                column
                    .values
                    .iter()
                    .map(|(id, log)| {
                        (
                            *id,
                            log.history()
                                .iter()
                                .map(|(epoch, value)| (*epoch, value.clone()))
                                .collect(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Statistics about column compression.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Size of uncompressed data in bytes.
    pub uncompressed_size: usize,
    /// Size of compressed data in bytes.
    pub compressed_size: usize,
    /// Number of values in the column.
    pub value_count: usize,
    /// Codec used for compression.
    pub codec: Option<CompressionCodec>,
}

impl CompressionStats {
    /// Returns the compression ratio (uncompressed / compressed).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            return 1.0;
        }
        self.uncompressed_size as f64 / self.compressed_size as f64
    }
}

/// A single property column (e.g., all "age" values).
///
/// Maintains min/max/null_count for fast predicate evaluation. When you
/// filter on `age > 50`, we first check if any age could possibly match
/// before scanning the actual values.
///
/// Columns support optional compression for large datasets. When compression
/// is enabled, the column automatically selects the best codec based on the
/// data type and characteristics.
pub struct PropertyColumn<Id: EntityId = NodeId> {
    /// Versioned storage: entity ID -> append-only version log.
    /// Each value is tagged with the epoch it was written in.
    values: FxHashMap<Id, VersionLog<Value>>,
    /// Zone map tracking min/max/null_count for predicate pushdown.
    zone_map: ZoneMapEntry,
    /// Whether zone map needs rebuild (after removes).
    zone_map_dirty: bool,
    /// Compression mode for this column.
    compression_mode: CompressionMode,
    /// Per-block zone maps populated when the column is compressed.
    ///
    /// Each entry covers a contiguous slice of `DEFAULT_BLOCK_ROWS` rows of
    /// the sorted compressed array. Empty when the column is uncompressed
    /// (the hot buffer is a `HashMap` with no row order, so per-block
    /// pruning would be meaningless). Phase 4 consumes these for lazy
    /// `range_iter`-style scans.
    block_zone_maps: Vec<ZoneMapEntry>,
}

// === Temporal implementation: VersionLog-backed property column ===
//
// **Zone map limitation**: zone maps track min/max across the *latest* values
// only (see `rebuild_zone_map`). For temporal queries at old epochs, the zone
// map may produce false negatives: it could reject a column based on current
// min/max even though historical values would match. This is a known
// trade-off: temporal queries are conservative but never return wrong results
// (the `zone_map_dirty` fallback returns `true` = "might match").
//
// **Compression**: disabled in temporal mode because the underlying codecs
// (DeltaBitPacked, Dictionary, BitVector) operate on flat `FxHashMap<Id, Value>`
// arrays, not `FxHashMap<Id, VersionLog<Value>>`. Per-epoch compression is a
// potential future optimization.
impl<Id: EntityId> PropertyColumn<Id> {
    /// Creates a new empty column.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: FxHashMap::default(),
            zone_map: ZoneMapEntry::new(),
            zone_map_dirty: false,
            compression_mode: CompressionMode::None,
            block_zone_maps: Vec::new(),
        }
    }

    /// Creates a new column with the specified compression mode.
    #[must_use]
    pub fn with_compression(mode: CompressionMode) -> Self {
        Self {
            values: FxHashMap::default(),
            zone_map: ZoneMapEntry::new(),
            zone_map_dirty: false,
            compression_mode: mode,
            block_zone_maps: Vec::new(),
        }
    }

    /// Sets the compression mode for this column.
    pub fn set_compression_mode(&mut self, mode: CompressionMode) {
        self.compression_mode = mode;
    }

    /// Returns the compression mode for this column.
    #[must_use]
    pub fn compression_mode(&self) -> CompressionMode {
        self.compression_mode
    }

    /// Sets a value for an entity, appending to its version log.
    ///
    /// For non-transactional writes, pass the current epoch.
    /// For transactional writes, pass `EpochId::PENDING`.
    pub fn set(&mut self, id: Id, value: Value, epoch: EpochId) {
        self.update_zone_map_on_insert(&value);
        self.values.entry(id).or_default().append(epoch, value);
    }

    /// Updates zone map when inserting a value.
    fn update_zone_map_on_insert(&mut self, value: &Value) {
        self.zone_map.row_count += 1;

        if matches!(value, Value::Null) {
            self.zone_map.null_count += 1;
            return;
        }

        match &self.zone_map.min {
            None => self.zone_map.min = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Less) {
                    self.zone_map.min = Some(value.clone());
                }
            }
        }

        match &self.zone_map.max {
            None => self.zone_map.max = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Greater) {
                    self.zone_map.max = Some(value.clone());
                }
            }
        }
    }

    /// Gets the latest value for an entity, filtering out tombstones (Null).
    #[must_use]
    pub fn get(&self, id: Id) -> Option<Value> {
        self.values
            .get(&id)
            .and_then(|log| log.latest())
            .filter(|v| !v.is_null())
            .cloned()
    }

    /// Removes a value by appending a tombstone (Null) at the given epoch.
    pub fn remove(&mut self, id: Id, epoch: EpochId) -> Option<Value> {
        let previous = self.get(id);
        if previous.is_some() {
            self.values
                .entry(id)
                .or_default()
                .append(epoch, Value::Null);
            self.zone_map_dirty = true;
        }
        previous
    }

    /// Returns the number of live (non-tombstoned) values in this column.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values
            .values()
            .filter(|log| log.latest().is_some_and(|v| !v.is_null()))
            .count()
    }

    /// Returns true if this column is empty.
    #[cfg(test)]
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns compression statistics for this column.
    ///
    /// In temporal mode, compression is not used. Reports live value count only.
    #[must_use]
    pub fn compression_stats(&self) -> CompressionStats {
        let live_count = self.len();
        let hot_size = live_count * std::mem::size_of::<Value>();

        CompressionStats {
            uncompressed_size: hot_size,
            compressed_size: hot_size,
            value_count: live_count,
            codec: None,
        }
    }

    /// Returns estimated heap memory for this column.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        self.values.capacity()
            * (std::mem::size_of::<Id>() + std::mem::size_of::<VersionLog<Value>>() + 1)
    }

    /// Compression is not supported in temporal mode (no-op).
    pub fn compress(&mut self) {}

    /// Forces compression (no-op in temporal mode).
    pub fn force_compress(&mut self) {}

    /// Returns the zone map for this column.
    #[must_use]
    pub fn zone_map(&self) -> &ZoneMapEntry {
        &self.zone_map
    }

    /// Returns the per-block zone maps for this column.
    ///
    /// Always empty in temporal mode: compression is disabled for
    /// `VersionLog`-backed columns (see module-level note), so there is
    /// no sorted compressed array to chunk into blocks.
    #[must_use]
    pub fn block_zone_maps(&self) -> &[ZoneMapEntry] {
        &self.block_zone_maps
    }

    /// Uses zone map to check if any values could satisfy the predicate.
    #[must_use]
    pub fn might_match(&self, op: CompareOp, value: &Value) -> bool {
        if self.zone_map_dirty {
            return true;
        }

        match op {
            CompareOp::Eq => self.zone_map.might_contain_equal(value),
            CompareOp::Ne => match (&self.zone_map.min, &self.zone_map.max) {
                (Some(min), Some(max)) => {
                    !(compare_values(min, value) == Some(Ordering::Equal)
                        && compare_values(max, value) == Some(Ordering::Equal))
                }
                _ => true,
            },
            CompareOp::Lt => self.zone_map.might_contain_less_than(value, false),
            CompareOp::Le => self.zone_map.might_contain_less_than(value, true),
            CompareOp::Gt => self.zone_map.might_contain_greater_than(value, false),
            CompareOp::Ge => self.zone_map.might_contain_greater_than(value, true),
        }
    }

    /// Rebuilds zone map from current (latest) values.
    pub fn rebuild_zone_map(&mut self) {
        let mut zone_map = ZoneMapEntry::new();

        for log in self.values.values() {
            if let Some(value) = log.latest() {
                zone_map.row_count += 1;

                if matches!(value, Value::Null) {
                    zone_map.null_count += 1;
                    continue;
                }

                match &zone_map.min {
                    None => zone_map.min = Some(value.clone()),
                    Some(current) => {
                        if compare_values(value, current) == Some(Ordering::Less) {
                            zone_map.min = Some(value.clone());
                        }
                    }
                }

                match &zone_map.max {
                    None => zone_map.max = Some(value.clone()),
                    Some(current) => {
                        if compare_values(value, current) == Some(Ordering::Greater) {
                            zone_map.max = Some(value.clone());
                        }
                    }
                }
            }
        }

        self.zone_map = zone_map;
        self.zone_map_dirty = false;
    }

    // === Temporal-only methods ===

    /// Gets the value at a specific epoch via binary search, filtering tombstones.
    #[must_use]
    pub fn get_at(&self, id: Id, epoch: EpochId) -> Option<Value> {
        self.values
            .get(&id)
            .and_then(|log| log.at(epoch))
            .filter(|v| !v.is_null())
            .cloned()
    }

    /// Replaces PENDING epochs with the real commit epoch in all version logs.
    pub fn finalize_pending(&mut self, real_epoch: EpochId) {
        for log in self.values.values_mut() {
            log.finalize_pending(real_epoch);
        }
    }

    /// Removes all PENDING entries from all version logs (transaction rollback).
    pub fn remove_pending(&mut self) {
        for log in self.values.values_mut() {
            log.remove_pending();
        }
        self.values.retain(|_, log| !log.is_empty());
    }

    /// Garbage-collects old versions from all version logs.
    pub fn gc(&mut self, min_epoch: EpochId) {
        for log in self.values.values_mut() {
            log.gc(min_epoch);
        }
        self.values.retain(|_, log| !log.is_empty());
    }

    /// Removes PENDING entries for a specific entity (targeted rollback).
    #[cfg(feature = "lpg")]
    pub fn remove_pending_for(&mut self, id: Id) {
        if let Some(log) = self.values.get_mut(&id) {
            log.remove_pending();
            if log.is_empty() {
                self.values.remove(&id);
            }
        }
    }

    /// Removes up to `n` PENDING entries for a specific entity.
    ///
    /// Used by savepoint rollback to pop only the entries added after the
    /// savepoint, leaving earlier PENDING entries intact.
    #[cfg(feature = "lpg")]
    pub fn pop_n_pending_for(&mut self, id: Id, n: usize) {
        if let Some(log) = self.values.get_mut(&id) {
            log.pop_n_pending(n);
            if log.is_empty() {
                self.values.remove(&id);
            }
        }
    }
}

/// Compares two values for ordering.
fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

impl<Id: EntityId> Default for PropertyColumn<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// A borrowed reference to a property column for bulk reads.
///
/// Holds the read lock so the column can't change while you're iterating.
pub struct PropertyColumnRef<'a, Id: EntityId = NodeId> {
    _guard: parking_lot::RwLockReadGuard<'a, FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    _key: PropertyKey,
    _marker: PhantomData<Id>,
}

#[cfg(all(test, feature = "lpg"))]
mod gc_tests {
    use super::*;

    #[test]
    fn expired_tombstones_respect_floor_pending_and_live_values() {
        let storage = PropertyStorage::<NodeId>::new();
        let key = PropertyKey::new("value");
        for id in 0..4 {
            storage.set(
                NodeId::new(id),
                key.clone(),
                Value::Int64(7),
                EpochId::new(1),
            );
        }
        storage.remove(NodeId::new(0), &key, EpochId::new(5));
        storage.remove(NodeId::new(1), &key, EpochId::new(6));
        storage.remove(NodeId::new(2), &key, EpochId::PENDING);

        storage.gc_expired_tombstones(EpochId::PENDING);
        storage.gc_expired_tombstones(EpochId::new(4));
        for id in 0..4 {
            assert_eq!(
                storage.get_at(NodeId::new(id), &key, EpochId::new(4)),
                Some(Value::Int64(7))
            );
        }
        storage.gc_expired_tombstones(EpochId::new(5));
        assert!(storage.get_history(NodeId::new(0), &key).is_empty());
        assert_eq!(
            storage.get_at(NodeId::new(1), &key, EpochId::new(5)),
            Some(Value::Int64(7))
        );
        assert_eq!(
            storage.get_at(NodeId::new(2), &key, EpochId::new(5)),
            Some(Value::Int64(7))
        );
        assert_eq!(storage.get(NodeId::new(3), &key), Some(Value::Int64(7)));
        storage.gc_expired_tombstones(EpochId::new(6));
        assert!(storage.get_history(NodeId::new(1), &key).is_empty());
        assert_eq!(storage.get_history(NodeId::new(2), &key).len(), 2);
        storage.set(
            NodeId::new(0),
            key.clone(),
            Value::Int64(9),
            EpochId::new(7),
        );
        assert_eq!(storage.get(NodeId::new(0), &key), Some(Value::Int64(9)));
    }
}

#[cfg(all(test, feature = "lpg", feature = "compact-store"))]
mod rollback_tests {
    use super::*;

    #[test]
    fn in_place_identity_purge_removes_all_history_and_preserves_other_rows() {
        let storage = PropertyStorage::<NodeId>::new();
        let removed = NodeId::new(41);
        let retained = NodeId::new(42);
        let key = PropertyKey::new("proof");
        storage.set(removed, key.clone(), Value::from("old"), EpochId::new(1));
        storage.set(removed, key.clone(), Value::from("new"), EpochId::new(2));
        storage.set(
            retained,
            key.clone(),
            Value::from("retained"),
            EpochId::new(1),
        );

        assert!(storage.purge_identity_in_place(removed));
        assert!(storage.get_history(removed, &key).is_empty());
        assert_eq!(storage.get(retained, &key), Some(Value::from("retained")));
        assert!(!storage.purge_identity_in_place(removed));
    }
}
