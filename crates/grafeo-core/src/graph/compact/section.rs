//! [`Section`](grafeo_common::storage::section::Section) implementation for [`CompactStore`].
//!
//! Serializes/deserializes a CompactStore to/from the `.grafeo` container
//! format with versioned headers and CRC32 integrity.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{ContentId, EdgeId, EpochId, EpochInterval, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

use super::CompactStore;
use super::column::ColumnCodec;
use super::compaction::{FoldedEdgeRow, FoldedNodeRow, RawTemporalColumn};
use super::content_dedup::BlockPool;
use super::csr::{CsrAdjacency, PackedOpenAdjacency};
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::temporal_column::TemporalColumn;
use super::zone_map::ZoneMap;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

/// Magic bytes identifying a CompactStore section.
const MAGIC: [u8; 4] = *b"GCST";

/// v9 appends a retained-history coverage floor after the temporal sidecars.
/// PENDING encodes unknown coverage.
const FORMAT_VERSION: u8 = 9;

/// v4 node-table layout marker: an all-open identity base built from current
/// values — no validity or row-range bytes follow; reconstruct as all-open.
const LAYOUT_ALL_OPEN: u8 = 0;

/// v4 node-table layout marker: an explicit temporal base — per-node row ranges
/// and per-column validity intervals + block content-hash follow.
const LAYOUT_EXPLICIT: u8 = 1;

/// Wraps a [`CompactStore`] as a container [`Section`].
pub struct CompactStoreSection {
    store: RwLock<Option<Arc<CompactStore>>>,
    dirty: AtomicBool,
}

impl CompactStoreSection {
    /// Creates a new section wrapping an existing store.
    #[must_use]
    pub fn new(store: Arc<CompactStore>) -> Self {
        Self {
            store: RwLock::new(Some(store)),
            dirty: AtomicBool::new(false),
        }
    }

    /// Creates an empty section (for deserialization).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            store: RwLock::new(None),
            dirty: AtomicBool::new(false),
        }
    }

    /// Marks this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Returns a reference to the inner store, if any.
    #[must_use]
    pub fn store(&self) -> Option<Arc<CompactStore>> {
        self.store.read().clone()
    }

    /// Deserializes from a refcounted [`Bytes`] buffer (Phase 3c).
    ///
    /// This is the zero-copy entry point: when `data` wraps a mmap
    /// region (via [`bytes::Bytes::from_owner`]), column codec storage
    /// is constructed via `data.slice(range)` rather than copying. The
    /// trait [`Section::deserialize`] entry point still works on
    /// `&[u8]` and incurs one heap copy (a single `Bytes::copy_from_slice`
    /// at the boundary).
    ///
    /// # Errors
    ///
    /// Same error semantics as [`Section::deserialize`].
    pub fn deserialize_from_bytes(
        &mut self,
        data: bytes::Bytes,
    ) -> grafeo_common::utils::error::Result<()> {
        let store = deserialize_compact_store(&data, None).map_err(|e| {
            grafeo_common::utils::error::Error::Internal(format!(
                "CompactStore deserialization failed: {e}"
            ))
        })?;
        *self.store.write() = Some(Arc::new(store));
        Ok(())
    }

    /// Serializes the store **content-addressed**: each node value block is
    /// interned into `pool` (deduplicated by content id across base generations,
    /// so unchanged columns are stored once) and only its 32-byte content id is
    /// written inline. Reload with [`deserialize_content_addressed`], passing the
    /// same pool. The non-content-addressed serialization is byte-identical to
    /// before — content-addressing is a strictly additive header flag.
    ///
    /// # Errors
    ///
    /// Returns an error if there is no store to serialize.
    pub fn serialize_content_addressed(
        &self,
        pool: &mut BlockPool,
    ) -> grafeo_common::utils::error::Result<Vec<u8>> {
        self.serialize_with_pool(Some(pool))
    }

    /// Serializes the current format with optional content-addressed blocks.
    fn serialize_with_pool(
        &self,
        mut block_pool: Option<&mut BlockPool>,
    ) -> grafeo_common::utils::error::Result<Vec<u8>> {
        let guard = self.store.read();
        let store = guard.as_ref().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal("no CompactStore to serialize".into())
        })?;

        let mut buf = Vec::with_capacity(store.memory_bytes());

        // Header. `flags` bit 0 = preserves-ids; bit 1 = content-addressed (node
        // value blocks live in an external pool, only their content id is inline).
        let content_addressed = block_pool.is_some();
        buf.extend_from_slice(&MAGIC);
        buf.push(FORMAT_VERSION);
        let flags: u8 = u8::from(store.preserves_ids()) | (u8::from(content_addressed) << 1);
        buf.push(flags);

        // Node tables.
        write_len(&mut buf, store.node_tables_by_id.len());
        for nt in &store.node_tables_by_id {
            write_str(&mut buf, nt.label());
            write_len(&mut buf, nt.len());
            let columns = nt.temporal_columns();
            let zone_maps = nt.zone_maps();
            write_len(&mut buf, columns.len());
            // Key-sorted columns preserve byte identity across round trips.
            let mut ordered: Vec<(&PropertyKey, &TemporalColumn)> = columns.iter().collect();
            ordered.sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
            let mut codec_ranges: Vec<(usize, usize)> = Vec::with_capacity(ordered.len());
            for (key, tcol) in &ordered {
                write_str(&mut buf, key.as_str());
                // Zone map for this column.
                if let Some(zm) = zone_maps.get(*key) {
                    buf.push(1);
                    write_zone_map(&mut buf, zm);
                } else {
                    buf.push(0);
                }
                // The value layer persists the current-value projection via
                // the v3 codec body, followed by the temporal addendum.
                // In content-addressed mode the value block is interned into the
                // pool and only its 32-byte content id is written inline.
                let codec_start = buf.len();
                let hint = nt.block_zone_maps().get(*key).map(Vec::as_slice);
                if let Some(pool) = block_pool.as_deref_mut() {
                    let mut block = Vec::new();
                    write_codec(tcol.values(), &mut block, hint);
                    let cid = pool.intern(block);
                    buf.extend_from_slice(cid.as_bytes());
                } else {
                    write_codec(tcol.values(), &mut buf, hint);
                }
                codec_ranges.push((codec_start, buf.len()));
            }
            // v4 temporal addendum: per-node row ranges + per-column validity +
            // block content-hash (compact `all-open` marker for current-value bases).
            // In content-addressed mode the per-column hash is omitted — the inline
            // content id already identifies the block.
            write_temporal_addendum(&mut buf, nt, &ordered, &codec_ranges, content_addressed);
        }

        // Relationship tables.
        write_len(&mut buf, store.rel_tables_by_id.len());
        for rt in &store.rel_tables_by_id {
            write_str(&mut buf, rt.edge_type().as_str());
            write_u16(&mut buf, rt.src_table_id());
            write_u16(&mut buf, rt.dst_table_id());
            let derived_fwd;
            let fwd = if rt.fwd().num_edges() == 0 {
                if let Some(packed) = rt.packed_fwd() {
                    derived_fwd = packed.derive_current_csr();
                    &derived_fwd
                } else {
                    rt.fwd()
                }
            } else {
                rt.fwd()
            };
            fwd.write_to(&mut buf);
            if let Some(packed_bwd) = rt.packed_bwd() {
                if rt.bwd().is_some_and(|b| b.num_edges() == 0) {
                    buf.push(1);
                    // Derived current bwd has no edge_data (fwd positions).
                    // Topology lives in packed_bwd; write an empty CSR so
                    // RelTable::new accepts the load, then restore fills packed.
                    CsrAdjacency::empty(packed_bwd.num_nodes()).write_to(&mut buf);
                } else if let Some(bwd) = rt.bwd() {
                    buf.push(1);
                    bwd.write_to(&mut buf);
                } else {
                    buf.push(0);
                }
            } else if let Some(bwd) = rt.bwd() {
                buf.push(1);
                bwd.write_to(&mut buf);
            } else {
                buf.push(0);
            }
            let properties = rt.properties();
            // Sort property keys so re-serialization is byte-identical.
            let mut ordered_props: Vec<(&PropertyKey, &TemporalColumn)> =
                properties.iter().collect();
            ordered_props.sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
            write_len(&mut buf, ordered_props.len());
            for (key, tcol) in &ordered_props {
                write_str(&mut buf, key.as_str());
                // RelTable property columns stay the current-value projection
                // (all-open). Closed-interval property history lives on the
                // store-level closed-edge sidecar, persisted in the v5 addendum.
                debug_assert!(
                    tcol.is_all_open(),
                    "RelTable persist writes only TemporalColumn::values; \
                     closed property history is on CompactStore.closed_edges"
                );
                write_codec(tcol.values(), &mut buf, None);
            }
            write_rel_temporal_addendum(&mut buf, rt);
        }
        write_closed_edges(&mut buf, store).map_err(|error| {
            grafeo_common::utils::error::Error::Internal(format!(
                "CompactStore closed-edge serialization failed: {error}"
            ))
        })?;
        write_temporal_nodes(&mut buf, store).map_err(|error| {
            grafeo_common::utils::error::Error::Internal(format!(
                "CompactStore temporal-node serialization failed: {error}"
            ))
        })?;
        write_u64(
            &mut buf,
            store
                .property_history_floor()
                .unwrap_or(EpochId::PENDING)
                .as_u64(),
        );
        // Continue building buf in `serialize()` epilogue.
        Ok(self.append_id_maps_and_crc(buf, store))
    }

    /// Appends ID maps (if applicable) and trailing CRC to the buffer.
    fn append_id_maps_and_crc(&self, mut buf: Vec<u8>, store: &CompactStore) -> Vec<u8> {
        // ID maps. Entries are written in ascending id order so the output is
        // deterministic (the hash-map iteration order is not stable across a
        // deserialize→re-serialize cycle) — the byte-identity half of the
        // determinism contract (invariant #3). Order is irrelevant on read.
        if store.preserves_ids() {
            if let Some(ref node_map) = store.node_id_map {
                write_len(&mut buf, node_map.len());
                let mut entries: Vec<(&NodeId, &(u16, u64))> = node_map.iter().collect();
                entries.sort_unstable_by_key(|(nid, _)| nid.as_u64());
                for (nid, &(tid, off)) in entries {
                    write_u64(&mut buf, nid.as_u64());
                    write_u16(&mut buf, tid);
                    write_u64(&mut buf, off);
                }
            }
            if let Some(ref edge_map) = store.edge_id_map {
                write_len(&mut buf, edge_map.len());
                let mut entries: Vec<(&EdgeId, &(u16, u64))> = edge_map.iter().collect();
                entries.sort_unstable_by_key(|(eid, _)| eid.as_u64());
                for (eid, &(rtid, pos)) in entries {
                    write_u64(&mut buf, eid.as_u64());
                    write_u16(&mut buf, rtid);
                    write_u64(&mut buf, pos);
                }
            }
        }

        // CRC32 at end.
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }
}

/// Writes the current column codec body with per-block zone maps.
///
/// When `block_stats_hint` is `None` or has a mismatched length,
/// [`ColumnCodec::write_to_v3`] computes the stats
/// from the column itself.
fn write_codec(codec: &ColumnCodec, buf: &mut Vec<u8>, block_stats_hint: Option<&[ZoneMap]>) {
    codec.write_to_v3(buf, block_stats_hint);
}

/// Writes the v4 temporal addendum for a node table.
///
/// An all-open base (built from current values) writes only the one-byte
/// [`LAYOUT_ALL_OPEN`] marker — it reconstructs as all-open on read, keeping
/// the common case at v3 size. An explicit (post-compaction) base writes
/// [`LAYOUT_EXPLICIT`], then the per-node `(row_start, row_count)` ranges, then
/// per column (in the same key-sorted order) the validity intervals and a
/// [`physical_block_hash`](super::content_hash::physical_block_hash) over that
/// column's serialized value block — the content-id the cross-time dedup
/// follow-on keys on. History columns are never rewritten.
fn write_temporal_addendum(
    buf: &mut Vec<u8>,
    nt: &NodeTable,
    ordered: &[(&PropertyKey, &TemporalColumn)],
    codec_ranges: &[(usize, usize)],
    content_addressed: bool,
) {
    if nt.is_all_open() {
        buf.push(LAYOUT_ALL_OPEN);
        return;
    }
    buf.push(LAYOUT_EXPLICIT);
    // Per column (key-sorted), in order: this column's per-node (row_start,
    // row_count) ranges (`node_count` entries — the table `len`, already
    // persisted), then its validity intervals, then a value-block content-hash.
    let node_count = nt.len();
    let column_ranges = nt.explicit_column_ranges();
    for (idx, (key, tcol)) in ordered.iter().enumerate() {
        let ranges = column_ranges.and_then(|m| m.get(*key)).map(Vec::as_slice);
        for node in 0..node_count {
            let (start, count) = match ranges {
                Some(r) => r.get(node).copied().unwrap_or((0, 0)),
                None if column_ranges.is_none() => (u32::try_from(node).unwrap_or(u32::MAX), 1),
                None => (0, 0),
            };
            write_len(buf, start as usize);
            write_len(buf, count as usize);
        }
        let validity = tcol.validity();
        write_len(buf, validity.len());
        for iv in validity {
            write_u64(buf, iv.from().as_u64());
            write_u64(buf, iv.to().as_u64());
        }
        if !content_addressed {
            let (start, end) = codec_ranges[idx];
            let cid = super::content_hash::physical_block_hash(&buf[start..end]);
            buf.extend_from_slice(cid.as_bytes());
        }
    }
}

/// v5 rel addendum: packed fat adjacency (open prefix + closed tails).
///
/// All-open / never-merged tables write only [`LAYOUT_ALL_OPEN`]. Explicit
/// tables write packed fwd (required) and packed bwd (optional). Open-row
/// structural validity is reconstructed from the packed open prefix.
fn write_rel_temporal_addendum(buf: &mut Vec<u8>, rt: &RelTable) {
    let assembled;
    let packed_fwd = match rt.packed_for_persist() {
        Some(p) => {
            assembled = p;
            &assembled
        }
        None => {
            buf.push(LAYOUT_ALL_OPEN);
            return;
        }
    };
    buf.push(LAYOUT_EXPLICIT);
    packed_fwd.write_to(buf);
    if let Some(packed_bwd) = rt.packed_bwd() {
        buf.push(1);
        packed_bwd.write_to(buf);
    } else {
        buf.push(0);
    }
}

fn read_rel_temporal_addendum(
    data: &[u8],
    pos: &mut usize,
    table: &mut RelTable,
) -> Result<(), String> {
    let layout = *data
        .get(*pos)
        .ok_or("truncated rel temporal layout marker")?;
    *pos += 1;
    match layout {
        LAYOUT_ALL_OPEN => Ok(()),
        LAYOUT_EXPLICIT => {
            let packed_fwd = PackedOpenAdjacency::read_from(data, pos)
                .map_err(|e| format!("packed fwd: {e}"))?;
            let has_bwd = *data.get(*pos).ok_or("truncated packed bwd flag")?;
            *pos += 1;
            let packed_bwd = if has_bwd == 1 {
                Some(
                    PackedOpenAdjacency::read_from(data, pos)
                        .map_err(|e| format!("packed bwd: {e}"))?,
                )
            } else {
                None
            };
            let validity = packed_fwd.open_validity();
            table.restore_temporal_addendum(packed_fwd, packed_bwd, validity);
            Ok(())
        }
        other => Err(format!("unknown rel temporal layout marker {other}")),
    }
}

/// Writes exact fallback property runs in deterministic key order. Each value
/// is length-delimited using `Value`'s exhaustive bincode representation; the
/// surrounding compact-section version and CRC provide framing and integrity.
fn write_raw_temporal_properties(
    buf: &mut Vec<u8>,
    properties: &FxHashMap<PropertyKey, RawTemporalColumn>,
) -> Result<(), String> {
    let mut properties: Vec<_> = properties.iter().collect();
    properties.sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    write_len(buf, properties.len());
    for (key, column) in properties {
        if column.is_empty() {
            return Err(format!(
                "raw temporal property {} has no value rows",
                key.as_str()
            ));
        }
        write_str(buf, key.as_str());
        write_len(buf, column.len());
        let mut previous_epoch = None;
        for (epoch, value) in column.history() {
            if *epoch == EpochId::PENDING {
                return Err(format!(
                    "raw temporal property {} contains a pending epoch",
                    key.as_str(),
                ));
            }
            if previous_epoch.is_some_and(|prior| prior > *epoch) {
                return Err(format!(
                    "raw temporal property {} has unordered epochs",
                    key.as_str()
                ));
            }
            previous_epoch = Some(*epoch);
            write_u64(buf, epoch.as_u64());
            let encoded = value
                .serialize()
                .map_err(|error| format!("raw property {}: {error}", key.as_str()))?;
            write_len(buf, encoded.len());
            buf.extend_from_slice(&encoded);
        }
    }
    Ok(())
}

fn read_raw_temporal_properties(
    data: &[u8],
    pos: &mut usize,
) -> Result<FxHashMap<PropertyKey, RawTemporalColumn>, String> {
    let count = read_u32(data, pos)? as usize;
    let mut properties = FxHashMap::with_capacity_and_hasher(count, Default::default());
    for _ in 0..count {
        let key = PropertyKey::new(read_string(data, pos)?);
        let row_count = read_u32(data, pos)? as usize;
        if row_count == 0 {
            return Err(format!(
                "raw temporal property {} has no value rows",
                key.as_str()
            ));
        }
        let mut history = Vec::with_capacity(row_count.min(data.len().saturating_sub(*pos) / 13));
        let mut previous_epoch = None;
        for _ in 0..row_count {
            let epoch = EpochId::new(read_u64(data, pos)?);
            if epoch == EpochId::PENDING {
                return Err(format!(
                    "raw temporal property {} contains a pending epoch",
                    key.as_str(),
                ));
            }
            if previous_epoch.is_some_and(|prior| prior > epoch) {
                return Err(format!(
                    "raw temporal property {} has unordered epochs",
                    key.as_str()
                ));
            }
            previous_epoch = Some(epoch);

            let encoded_len = read_u32(data, pos)? as usize;
            let end = pos
                .checked_add(encoded_len)
                .ok_or_else(|| format!("raw property {} length overflow", key.as_str()))?;
            let encoded = data
                .get(*pos..end)
                .ok_or_else(|| format!("truncated raw temporal property {} value", key.as_str()))?;
            let (value, consumed) =
                bincode::serde::decode_from_slice::<Value, _>(encoded, bincode::config::standard())
                    .map_err(|error| format!("raw property {}: {error}", key.as_str()))?;
            if consumed != encoded.len() {
                return Err(format!(
                    "raw temporal property {} value has trailing bytes",
                    key.as_str()
                ));
            }
            *pos = end;
            history.push((epoch, value));
        }
        if properties
            .insert(key.clone(), RawTemporalColumn::new(history))
            .is_some()
        {
            return Err(format!("duplicate raw temporal property {}", key.as_str()));
        }
    }
    Ok(properties)
}

/// Persists sidecar closed rows (orphans + property-bearing packed-placed).
/// Structure-only packed-placed lives are reconstructed from the rel addendum.
fn write_closed_edges(buf: &mut Vec<u8>, store: &CompactStore) -> Result<(), String> {
    let mut rows: Vec<(&EdgeId, &FoldedEdgeRow)> = store
        .closed_edges()
        .iter()
        .flat_map(|(id, rows)| rows.iter().map(move |row| (id, row)))
        .collect();
    rows.sort_unstable_by_key(|(id, row)| (id.as_u64(), row.validity.from().as_u64()));
    write_len(buf, rows.len());
    for (id, row) in rows {
        write_u64(buf, id.as_u64());
        write_u64(buf, row.src.as_u64());
        write_u64(buf, row.dst.as_u64());
        write_str(buf, row.edge_type.as_str());
        write_u64(buf, row.validity.from().as_u64());
        write_u64(buf, row.validity.to().as_u64());
        let mut props: Vec<(&PropertyKey, &TemporalColumn)> = row.properties.iter().collect();
        props.sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        write_len(buf, props.len());
        for (key, tcol) in props {
            write_str(buf, key.as_str());
            write_codec(tcol.values(), buf, None);
            let validity = tcol.validity();
            write_len(buf, validity.len());
            for iv in validity {
                write_u64(buf, iv.from().as_u64());
                write_u64(buf, iv.to().as_u64());
            }
        }
        write_raw_temporal_properties(buf, &row.raw_properties)?;
    }
    Ok(())
}

fn read_closed_edges(
    data_bytes: &Bytes,
    data: &[u8],
    pos: &mut usize,
) -> Result<FxHashMap<EdgeId, Vec<FoldedEdgeRow>>, String> {
    let n = read_u32(data, pos)? as usize;
    let mut closed: FxHashMap<EdgeId, Vec<FoldedEdgeRow>> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    for _ in 0..n {
        let id = EdgeId::new(read_u64(data, pos)?);
        let src = NodeId::new(read_u64(data, pos)?);
        let dst = NodeId::new(read_u64(data, pos)?);
        let edge_type = arcstr::ArcStr::from(read_string(data, pos)?.as_str());
        let from = EpochId::new(read_u64(data, pos)?);
        let to = EpochId::new(read_u64(data, pos)?);
        let validity = EpochInterval::closed(from, to);
        let nprops = read_u32(data, pos)? as usize;
        let mut properties = FxHashMap::with_capacity_and_hasher(nprops, Default::default());
        for _ in 0..nprops {
            let key = PropertyKey::new(&read_string(data, pos)?);
            let (codec, _) =
                read_codec(data_bytes, pos).map_err(|e| format!("closed-edge codec: {e}"))?;
            let niv = read_u32(data, pos)? as usize;
            let mut ivs = Vec::with_capacity(niv);
            for _ in 0..niv {
                let iv_from = EpochId::new(read_u64(data, pos)?);
                let iv_to = EpochId::new(read_u64(data, pos)?);
                ivs.push(EpochInterval::closed(iv_from, iv_to));
            }
            properties.insert(key, TemporalColumn::new(codec, ivs));
        }
        let raw_properties = read_raw_temporal_properties(data, pos)?;
        if raw_properties
            .keys()
            .any(|key| properties.contains_key(key))
        {
            return Err(format!(
                "closed edge {} duplicates a property in encoded and raw columns",
                id.as_u64()
            ));
        }
        closed.entry(id).or_default().push(FoldedEdgeRow {
            id,
            src,
            dst,
            edge_type,
            validity,
            properties,
            raw_properties,
        });
    }
    Ok(closed)
}

/// Persists structural and property history for temporally compacted nodes.
/// The outer map is id-sorted and every row is validity-sorted for deterministic
/// section bytes.
fn write_temporal_nodes(buf: &mut Vec<u8>, store: &CompactStore) -> Result<(), String> {
    let mut entities: Vec<(&NodeId, &Vec<FoldedNodeRow>)> = store.temporal_nodes().iter().collect();
    entities.sort_unstable_by_key(|(id, _)| id.as_u64());
    write_len(buf, entities.len());
    for (id, rows) in entities {
        write_u64(buf, id.as_u64());
        let mut rows: Vec<&FoldedNodeRow> = rows.iter().collect();
        rows.sort_unstable_by_key(|row| row.validity.from());
        write_len(buf, rows.len());
        for row in rows {
            let mut labels = row.labels.clone();
            labels.sort_unstable();
            labels.dedup();
            write_len(buf, labels.len());
            for label in labels {
                write_str(buf, label.as_str());
            }
            write_len(buf, row.label_versions.len());
            for (epoch, version_labels) in &row.label_versions {
                write_u64(buf, epoch.as_u64());
                let mut version_labels = version_labels.clone();
                version_labels.sort_unstable();
                version_labels.dedup();
                write_len(buf, version_labels.len());
                for label in version_labels {
                    write_str(buf, label.as_str());
                }
            }
            write_u64(buf, row.validity.from().as_u64());
            write_u64(buf, row.validity.to().as_u64());
            let mut props: Vec<(&PropertyKey, &TemporalColumn)> = row.properties.iter().collect();
            props.sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
            write_len(buf, props.len());
            for (key, column) in props {
                write_str(buf, key.as_str());
                write_codec(column.values(), buf, None);
                let validity = column.validity();
                write_len(buf, validity.len());
                for iv in validity {
                    write_u64(buf, iv.from().as_u64());
                    write_u64(buf, iv.to().as_u64());
                }
            }
            write_raw_temporal_properties(buf, &row.raw_properties)?;
        }
    }
    Ok(())
}

fn read_temporal_nodes(
    data_bytes: &Bytes,
    data: &[u8],
    pos: &mut usize,
) -> Result<FxHashMap<NodeId, Vec<FoldedNodeRow>>, String> {
    let nentities = read_u32(data, pos)? as usize;
    let mut nodes = FxHashMap::with_capacity_and_hasher(nentities, Default::default());
    for _ in 0..nentities {
        let id = NodeId::new(read_u64(data, pos)?);
        let nrows = read_u32(data, pos)? as usize;
        let mut rows = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            let nlabels = read_u32(data, pos)? as usize;
            let mut labels = Vec::with_capacity(nlabels);
            for _ in 0..nlabels {
                labels.push(arcstr::ArcStr::from(read_string(data, pos)?.as_str()));
            }
            let nversions = read_u32(data, pos)? as usize;
            let mut label_versions = Vec::with_capacity(nversions);
            for _ in 0..nversions {
                let epoch = EpochId::new(read_u64(data, pos)?);
                let nversion_labels = read_u32(data, pos)? as usize;
                let mut version_labels = Vec::with_capacity(nversion_labels);
                for _ in 0..nversion_labels {
                    version_labels.push(arcstr::ArcStr::from(read_string(data, pos)?.as_str()));
                }
                label_versions.push((epoch, version_labels));
            }
            let from = EpochId::new(read_u64(data, pos)?);
            let to = EpochId::new(read_u64(data, pos)?);
            let validity = EpochInterval::closed(from, to);
            if label_versions.is_empty() {
                label_versions.push((from, labels.clone()));
            }
            let nprops = read_u32(data, pos)? as usize;
            let mut properties = FxHashMap::with_capacity_and_hasher(nprops, Default::default());
            for _ in 0..nprops {
                let key = PropertyKey::new(&read_string(data, pos)?);
                let (codec, _) =
                    read_codec(data_bytes, pos).map_err(|e| format!("temporal-node codec: {e}"))?;
                let nvalidity = read_u32(data, pos)? as usize;
                let mut intervals = Vec::with_capacity(nvalidity);
                for _ in 0..nvalidity {
                    let iv_from = EpochId::new(read_u64(data, pos)?);
                    let iv_to = EpochId::new(read_u64(data, pos)?);
                    intervals.push(EpochInterval::closed(iv_from, iv_to));
                }
                if codec.len() != intervals.len() {
                    return Err(format!(
                        "temporal-node property {} has {} values but {} intervals",
                        key.as_str(),
                        codec.len(),
                        intervals.len(),
                    ));
                }
                properties.insert(key, TemporalColumn::new(codec, intervals));
            }
            let raw_properties = read_raw_temporal_properties(data, pos)?;
            if raw_properties
                .keys()
                .any(|key| properties.contains_key(key))
            {
                return Err(format!(
                    "temporal node {} duplicates a property in encoded and raw columns",
                    id.as_u64()
                ));
            }
            rows.push(FoldedNodeRow {
                id,
                labels,
                label_versions,
                validity,
                properties,
                raw_properties,
            });
        }
        if nodes.insert(id, rows).is_some() {
            return Err(format!("duplicate temporal-node id {}", id.as_u64()));
        }
    }
    Ok(nodes)
}

impl Section for CompactStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::CompactStore
    }

    fn version(&self) -> u8 {
        FORMAT_VERSION
    }

    fn serialize(&self) -> grafeo_common::utils::error::Result<Vec<u8>> {
        self.serialize_with_pool(None)
    }

    fn deserialize(&mut self, data: &[u8]) -> grafeo_common::utils::error::Result<()> {
        // Heap-copy entry point (Section trait). Phase 3c adds
        // [`deserialize_from_bytes`](Self::deserialize_from_bytes) which
        // skips the copy on the mmap path.
        let owned = bytes::Bytes::copy_from_slice(data);
        self.deserialize_from_bytes(owned)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.read().as_ref().map_or(0, |s| s.memory_bytes())
    }
}

// ── Deserialization ────────────────────────────────────────────────

/// Reads the current column codec body and its per-block zone maps.
fn read_codec(data: &Bytes, pos: &mut usize) -> Result<(ColumnCodec, Vec<ZoneMap>), String> {
    ColumnCodec::read_from_v3(data, pos).map_err(|e| e.to_string())
}

/// Reads the v4 temporal addendum following a node table's value columns and
/// builds the [`NodeTable`].
///
/// [`LAYOUT_ALL_OPEN`] reconstructs an all-open base (the value codecs wrapped
/// open at load). [`LAYOUT_EXPLICIT`] reads the per-node `(row_start, row_count)`
/// ranges, then per column (in stream/key-sorted order) the validity intervals;
/// the per-column block content-hash is read and skipped — it backs the deferred
/// cross-time dedup and is recomputable, and the section CRC already guards
/// integrity.
#[allow(clippy::too_many_arguments)]
fn read_temporal_node_table(
    data: &[u8],
    pos: &mut usize,
    schema: TableSchema,
    ordered_cols: Vec<(PropertyKey, ColumnCodec)>,
    zone_maps: FxHashMap<PropertyKey, ZoneMap>,
    block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>>,
    len: usize,
    content_addressed: bool,
) -> Result<NodeTable, String> {
    let layout = *data.get(*pos).ok_or("truncated temporal layout marker")?;
    *pos += 1;
    match layout {
        LAYOUT_ALL_OPEN => {
            let columns: FxHashMap<PropertyKey, ColumnCodec> = ordered_cols.into_iter().collect();
            Ok(NodeTable::from_columns_with_block_stats(
                schema,
                columns,
                zone_maps,
                block_zone_maps,
                len,
            ))
        }
        LAYOUT_EXPLICIT => {
            let mut columns: FxHashMap<PropertyKey, TemporalColumn> = FxHashMap::default();
            let mut column_ranges: FxHashMap<PropertyKey, Vec<(u32, u32)>> = FxHashMap::default();
            for (key, codec) in ordered_cols {
                // This column's per-node (row_start, row_count) ranges (`len` of them).
                let mut ranges = Vec::with_capacity(len);
                for _ in 0..len {
                    let start = read_u32(data, pos)?;
                    let count = read_u32(data, pos)?;
                    ranges.push((start, count));
                }
                // Validity intervals.
                let num_iv = read_u32(data, pos)? as usize;
                let mut validity = Vec::with_capacity(num_iv);
                for _ in 0..num_iv {
                    let from = read_u64(data, pos)?;
                    let to = read_u64(data, pos)?;
                    validity.push(EpochInterval::closed(EpochId::new(from), EpochId::new(to)));
                }
                // Block content-hash (32 bytes), inline mode only. In
                // content-addressed mode the id is carried inline with the
                // column body instead, so no trailing hash is present here.
                if !content_addressed {
                    if *pos + 32 > data.len() {
                        return Err("truncated block content-hash".into());
                    }
                    *pos += 32;
                }
                column_ranges.insert(key.clone(), ranges);
                columns.insert(key, TemporalColumn::new(codec, validity));
            }
            Ok(NodeTable::from_temporal_columns(
                schema,
                columns,
                zone_maps,
                block_zone_maps,
                Some(column_ranges),
                len,
            ))
        }
        other => Err(format!("unknown temporal layout marker {other}")),
    }
}

/// Deserializes a content-addressed CompactStore (written by
/// [`CompactStoreSection::serialize_content_addressed`]): node value blocks are
/// fetched from `pool` by the content ids embedded in `data`.
///
/// # Errors
///
/// Returns an error if `data` is malformed or a referenced block is absent from
/// the pool.
pub fn deserialize_content_addressed(
    data: &Bytes,
    pool: &BlockPool,
) -> Result<CompactStore, String> {
    deserialize_compact_store(data, Some(pool))
}

fn deserialize_compact_store(
    data_bytes: &bytes::Bytes,
    block_pool: Option<&BlockPool>,
) -> Result<CompactStore, String> {
    let data: &[u8] = data_bytes.as_ref();
    if data.len() < 10 {
        return Err("data too short for CompactStore section".into());
    }

    // Verify CRC32.
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(format!(
            "CRC32 mismatch: stored {stored_crc:#010X}, computed {computed_crc:#010X}"
        ));
    }

    let mut pos = 0;

    // Header.
    if data[pos..pos + 4] != MAGIC {
        return Err("bad magic".into());
    }
    pos += 4;
    let version = data[pos];
    pos += 1;
    if version != FORMAT_VERSION {
        return Err(format!(
            "unsupported CompactStore section version {version} (supported: {FORMAT_VERSION})"
        ));
    }
    let flags = data[pos];
    pos += 1;
    if flags & !0x03 != 0 {
        return Err(format!("unsupported CompactStore flags {flags:#04x}"));
    }
    let preserves_ids = flags & 0x01 != 0;
    let content_addressed = flags & 0x02 != 0;

    // Node tables.
    let num_node_tables = read_u32(data, &mut pos)? as usize;
    let mut node_tables = Vec::with_capacity(num_node_tables);
    let mut label_to_table_id: FxHashMap<arcstr::ArcStr, u16> = FxHashMap::default();
    let mut table_id_to_label: Vec<arcstr::ArcStr> = Vec::with_capacity(num_node_tables);

    for table_idx in 0..num_node_tables {
        let table_id = u16::try_from(table_idx).unwrap_or(0);
        let label = read_string(data, &mut pos)?;
        let label = arcstr::ArcStr::from(label.as_str());
        let row_count = read_u32(data, &mut pos)? as usize;
        let num_cols = read_u32(data, &mut pos)? as usize;

        // Columns are kept in stream order so the v4 temporal addendum's
        // per-column validity can be paired with each column by position.
        let mut ordered_cols: Vec<(PropertyKey, ColumnCodec)> = Vec::with_capacity(num_cols);
        let mut zone_maps: FxHashMap<PropertyKey, ZoneMap> = FxHashMap::default();
        let mut block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>> = FxHashMap::default();
        let mut col_defs = Vec::with_capacity(num_cols);

        for _ in 0..num_cols {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);

            let has_zm = *data.get(pos).ok_or("truncated zone map flag")?;
            pos += 1;
            if has_zm == 1 {
                let zm = read_zone_map(data, &mut pos)?;
                zone_maps.insert(key.clone(), zm);
            }

            // In content-addressed mode the inline body is a 32-byte content id;
            // the value block itself comes from the pool. Otherwise it is inline.
            let (codec, stats) = if content_addressed {
                if pos + 32 > data.len() {
                    return Err("truncated content id".into());
                }
                let cid = ContentId::from_bytes(
                    data[pos..pos + 32]
                        .try_into()
                        .expect("32-byte content id slice"),
                );
                pos += 32;
                let pool = block_pool.ok_or("content-addressed store needs a block pool")?;
                let block = pool
                    .get(cid)
                    .ok_or("content-addressed block missing from pool")?;
                let block_bytes = Bytes::copy_from_slice(block);
                let mut bpos = 0;
                read_codec(&block_bytes, &mut bpos).map_err(|e| format!("codec: {e}"))?
            } else {
                read_codec(data_bytes, &mut pos).map_err(|e| format!("codec: {e}"))?
            };
            block_zone_maps.insert(key.clone(), stats);
            let col_type = infer_column_type_from_codec(&codec);
            col_defs.push(ColumnDef::new(&key_str, col_type));
            ordered_cols.push((key, codec));
        }

        let schema = TableSchema::new(label.as_str(), table_id, col_defs);
        let table = read_temporal_node_table(
            data,
            &mut pos,
            schema,
            ordered_cols,
            zone_maps,
            block_zone_maps,
            row_count,
            content_addressed,
        )?;
        node_tables.push(table);
        label_to_table_id.insert(label.clone(), table_id);
        table_id_to_label.push(label);
    }

    // Relationship tables.
    let num_rel_tables = read_u32(data, &mut pos)? as usize;
    let mut rel_tables = Vec::with_capacity(num_rel_tables);
    let mut edge_type_to_rel_id: FxHashMap<arcstr::ArcStr, Vec<u16>> = FxHashMap::default();
    let mut rel_table_id_to_type: Vec<arcstr::ArcStr> = Vec::with_capacity(num_rel_tables);

    for rel_idx in 0..num_rel_tables {
        let rel_table_id = u16::try_from(rel_idx).unwrap_or(0);
        let edge_type = read_string(data, &mut pos)?;
        let edge_type = arcstr::ArcStr::from(edge_type.as_str());
        let src_tid = read_u16(data, &mut pos)?;
        let dst_tid = read_u16(data, &mut pos)?;

        let fwd = CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("fwd CSR: {e}"))?;

        let has_bwd = *data.get(pos).ok_or("truncated bwd flag")?;
        pos += 1;
        let bwd = if has_bwd == 1 {
            Some(CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("bwd CSR: {e}"))?)
        } else {
            None
        };

        let num_props = read_u32(data, &mut pos)? as usize;
        let mut properties: FxHashMap<PropertyKey, TemporalColumn> = FxHashMap::default();
        let mut prop_defs = Vec::with_capacity(num_props);
        for _ in 0..num_props {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);
            let (codec, _block_stats) =
                read_codec(data_bytes, &mut pos).map_err(|e| format!("edge codec: {e}"))?;
            let col_type = infer_column_type_from_codec(&codec);
            prop_defs.push(ColumnDef::new(&key_str, col_type));
            // Validity is not on disk; rebuild as all-open.
            properties.insert(key, TemporalColumn::all_open(codec));
        }

        let src_label = table_id_to_label
            .get(src_tid as usize)
            .cloned()
            .unwrap_or_default();
        let dst_label = table_id_to_label
            .get(dst_tid as usize)
            .cloned()
            .unwrap_or_default();

        let schema = EdgeSchema::new(
            edge_type.as_str(),
            rel_table_id,
            src_label.as_str(),
            dst_label.as_str(),
            prop_defs,
        );

        let mut table = RelTable::new(schema, fwd, bwd, properties, src_tid, dst_tid);
        read_rel_temporal_addendum(data, &mut pos, &mut table)?;
        edge_type_to_rel_id
            .entry(edge_type.clone())
            .or_default()
            .push(rel_table_id);
        rel_table_id_to_type.push(edge_type);
        rel_tables.push(table);
    }

    // Compute statistics.
    let mut stats = Statistics::new();
    let mut total_nodes = 0u64;
    let mut total_edges = 0u64;
    for (idx, nt) in node_tables.iter().enumerate() {
        let c = nt.len() as u64;
        total_nodes += c;
        stats.update_label(table_id_to_label[idx].as_str(), LabelStatistics::new(c));
    }
    let mut edge_counts: FxHashMap<&str, u64> = FxHashMap::default();
    for (idx, rt) in rel_tables.iter().enumerate() {
        let c = rt.num_edges() as u64;
        total_edges += c;
        *edge_counts
            .entry(rel_table_id_to_type[idx].as_str())
            .or_default() += c;
    }
    for (et, count) in edge_counts {
        stats.update_edge_type(et, EdgeTypeStatistics::new(count, 0.0, 0.0));
    }
    stats.total_nodes = total_nodes;
    stats.total_edges = total_edges;

    let mut store = CompactStore::new(
        node_tables,
        label_to_table_id,
        rel_tables,
        edge_type_to_rel_id,
        table_id_to_label,
        rel_table_id_to_type,
        stats,
    );

    let closed = read_closed_edges(data_bytes, data, &mut pos)?;
    store.set_closed_edges(closed);
    let nodes = read_temporal_nodes(data_bytes, data, &mut pos)?;
    store.set_temporal_nodes(nodes);
    let floor = EpochId::new(read_u64(data, &mut pos)?);
    store.property_history_floor = (floor != EpochId::PENDING).then_some(floor);

    // ID maps.
    if preserves_ids {
        let node_map_len = read_u32(data, &mut pos)? as usize;
        let mut node_id_map = FxHashMap::with_capacity_and_hasher(node_map_len, Default::default());
        let num_tables = store.node_tables_by_id.len();
        let mut node_offset_to_id: Vec<Vec<NodeId>> = vec![Vec::new(); num_tables];
        for _ in 0..node_map_len {
            let nid = NodeId::new(read_u64(data, &mut pos)?);
            let tid = read_u16(data, &mut pos)?;
            let off = read_u64(data, &mut pos)?;
            node_id_map.insert(nid, (tid, off));
            let off_idx = usize::try_from(off).unwrap_or(usize::MAX);
            if let Some(rev) = node_offset_to_id.get_mut(tid as usize) {
                while rev.len() <= off_idx {
                    rev.push(NodeId::INVALID);
                }
                rev[off_idx] = nid;
            }
        }

        let edge_map_len = read_u32(data, &mut pos)? as usize;
        let mut edge_id_map = FxHashMap::with_capacity_and_hasher(edge_map_len, Default::default());
        let num_rel = store.rel_tables_by_id.len();
        let mut edge_offset_to_id: Vec<Vec<EdgeId>> = vec![Vec::new(); num_rel];
        for _ in 0..edge_map_len {
            let eid = EdgeId::new(read_u64(data, &mut pos)?);
            let rtid = read_u16(data, &mut pos)?;
            let csr_pos = read_u64(data, &mut pos)?;
            edge_id_map.insert(eid, (rtid, csr_pos));
            let pos_idx = usize::try_from(csr_pos).unwrap_or(usize::MAX);
            if let Some(rev) = edge_offset_to_id.get_mut(rtid as usize) {
                while rev.len() <= pos_idx {
                    rev.push(EdgeId::INVALID);
                }
                rev[pos_idx] = eid;
            }
        }

        store.set_id_maps(
            node_id_map,
            edge_id_map,
            node_offset_to_id,
            edge_offset_to_id,
        );
    }

    Ok(store)
}

// ── Write helpers ──────────────────────────────────────────────────

fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_len(buf: &mut Vec<u8>, v: usize) {
    let n = u32::try_from(v).expect("length exceeds u32::MAX in compact section");
    buf.extend_from_slice(&n.to_le_bytes());
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let slen = u16::try_from(bytes.len()).expect("string exceeds u16::MAX in compact section");
    write_u16(buf, slen);
    buf.extend_from_slice(bytes);
}

fn write_zone_map(buf: &mut Vec<u8>, zm: &ZoneMap) {
    write_len(buf, zm.null_count);
    write_len(buf, zm.row_count);
    // Encode min/max as (tag, value) pairs.
    write_optional_value(buf, &zm.min);
    write_optional_value(buf, &zm.max);
}

fn write_optional_value(buf: &mut Vec<u8>, v: &Option<grafeo_common::types::Value>) {
    match v {
        None => buf.push(0),
        Some(grafeo_common::types::Value::Int64(n)) => {
            buf.push(1);
            // Store as raw i64 bytes to avoid sign-loss lint.
            buf.extend_from_slice(&n.to_le_bytes());
        }
        Some(grafeo_common::types::Value::Bool(b)) => {
            buf.push(2);
            buf.push(u8::from(*b));
        }
        Some(grafeo_common::types::Value::String(s)) => {
            buf.push(3);
            write_str(buf, s.as_str());
        }
        Some(_) => {
            // Unsupported type for zone map: write as absent.
            buf.push(0);
        }
    }
}

// ── Read helpers ───────────────────────────────────────────────────

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() {
        return Err("truncated u16".into());
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let slen = read_u16(data, pos)? as usize;
    if *pos + slen > data.len() {
        return Err("truncated string".into());
    }
    let s =
        std::str::from_utf8(&data[*pos..*pos + slen]).map_err(|_| "invalid UTF-8".to_string())?;
    *pos += slen;
    Ok(s.to_string())
}

fn read_zone_map(data: &[u8], pos: &mut usize) -> Result<ZoneMap, String> {
    let null_count = read_u32(data, pos)? as usize;
    let row_count = read_u32(data, pos)? as usize;
    let min = read_optional_value(data, pos)?;
    let max = read_optional_value(data, pos)?;
    Ok(ZoneMap {
        min,
        max,
        null_count,
        row_count,
    })
}

fn read_optional_value(
    data: &[u8],
    pos: &mut usize,
) -> Result<Option<grafeo_common::types::Value>, String> {
    let tag = *data.get(*pos).ok_or("truncated value tag")?;
    *pos += 1;
    match tag {
        0 => Ok(None),
        1 => {
            // Read raw i64 bytes (written via i64::to_le_bytes).
            if *pos + 8 > data.len() {
                return Err("truncated i64 value".into());
            }
            let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Some(grafeo_common::types::Value::Int64(v)))
        }
        2 => {
            let b = *data.get(*pos).ok_or("truncated bool")?;
            *pos += 1;
            Ok(Some(grafeo_common::types::Value::Bool(b != 0)))
        }
        3 => {
            let s = read_string(data, pos)?;
            Ok(Some(grafeo_common::types::Value::String(
                arcstr::ArcStr::from(s.as_str()),
            )))
        }
        _ => Err(format!("unknown value tag {tag}")),
    }
}

fn infer_column_type_from_codec(codec: &ColumnCodec) -> ColumnType {
    match codec {
        ColumnCodec::BitPacked(bp) => ColumnType::UInt {
            bits: bp.bits_per_value(),
        },
        ColumnCodec::Dict(_) => ColumnType::DictString,
        ColumnCodec::Bitmap(_) => ColumnType::Bool,
        ColumnCodec::Int8Vector { dimensions, .. } => ColumnType::Int8Vector {
            dimensions: *dimensions,
        },
        ColumnCodec::Float64(_) => ColumnType::Float64,
        ColumnCodec::Float32Vector { dimensions, .. } => ColumnType::Float32Vector {
            dimensions: *dimensions,
        },
        ColumnCodec::RawI64(_) => ColumnType::Int64,
        ColumnCodec::Fsst(_) => ColumnType::FsstString,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "lpg")]
    use crate::graph::compact::from_graph_store_preserving_ids;
    #[cfg(feature = "lpg")]
    use crate::graph::lpg::LpgStore;
    use crate::graph::traits::GraphStore;
    use grafeo_common::types::Value;

    #[cfg(feature = "lpg")]
    fn captured_graph_section(preserves_ids: bool) -> CompactStoreSection {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Item"]);
        store.set_node_property(node, "v", Value::Int64(7));
        let compact = if preserves_ids {
            from_graph_store_preserving_ids(&store).unwrap()
        } else {
            super::super::builder::from_graph_store(&store).unwrap()
        }
        .with_property_history_floor(Some(EpochId::new(17)));
        CompactStoreSection::new(Arc::new(compact))
    }

    #[cfg(feature = "lpg")]
    fn altered_header(bytes: &[u8], version: u8, flags: u8) -> Vec<u8> {
        let mut altered = bytes.to_vec();
        altered[4] = version;
        altered[5] = flags;
        let crc_offset = altered.len() - 4;
        let crc = crc32fast::hash(&altered[..crc_offset]);
        altered[crc_offset..].copy_from_slice(&crc.to_le_bytes());
        altered
    }

    #[cfg(feature = "lpg")]
    fn assert_rejected_at_all_entry_points(bytes: &[u8], expected: &str) {
        let mut target = captured_graph_section(true);
        let original = target.store().unwrap();
        let original_bytes = target.serialize().unwrap();
        let mut pool = BlockPool::new();
        target.serialize_content_addressed(&mut pool).unwrap();
        let original_pool = pool.to_bytes();
        for dirty in [false, true] {
            target.dirty.store(dirty, Ordering::Release);
            for from_bytes in [false, true] {
                let error = if from_bytes {
                    target.deserialize_from_bytes(Bytes::copy_from_slice(bytes))
                } else {
                    target.deserialize(bytes)
                }
                .expect_err("unsupported header must reject before replacing the store");
                match error {
                    grafeo_common::utils::error::Error::Internal(message) => assert_eq!(
                        message,
                        format!("CompactStore deserialization failed: {expected}")
                    ),
                    other => panic!("unexpected section error: {other}"),
                }
                assert!(Arc::ptr_eq(&target.store().unwrap(), &original));
                assert_eq!(target.serialize().unwrap(), original_bytes);
                assert_eq!(target.is_dirty(), dirty);
            }
            let Err(error) = deserialize_content_addressed(&Bytes::copy_from_slice(bytes), &pool)
            else {
                panic!("unsupported content-addressed header was accepted");
            };
            assert_eq!(error, expected);
            assert_eq!(pool.to_bytes(), original_pool);
            assert!(Arc::ptr_eq(&target.store().unwrap(), &original));
            assert_eq!(target.serialize().unwrap(), original_bytes);
            assert_eq!(target.is_dirty(), dirty);
        }
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn current_gcst_v9_matches_captured_bytes() {
        let expected = include_bytes!("fixtures/current_gcst_v9.bin");
        let section = captured_graph_section(true);
        assert_eq!(section.serialize().unwrap().as_slice(), expected.as_slice());
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(expected).unwrap();
        assert_eq!(
            restored.serialize().unwrap().as_slice(),
            expected.as_slice()
        );
        assert_eq!(
            restored.store().unwrap().property_history_floor(),
            Some(EpochId::new(17))
        );
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn predecessor_gcst_versions_reject_at_all_entry_points() {
        let predecessors: [&[u8]; 8] = [
            include_bytes!("fixtures/rejected_gcst_v1.bin"),
            include_bytes!("fixtures/rejected_gcst_v2.bin"),
            include_bytes!("fixtures/rejected_gcst_v3.bin"),
            include_bytes!("fixtures/rejected_gcst_v4.bin"),
            include_bytes!("fixtures/rejected_gcst_v5.bin"),
            include_bytes!("fixtures/rejected_gcst_v6.bin"),
            include_bytes!("fixtures/rejected_gcst_v7.bin"),
            include_bytes!("fixtures/rejected_gcst_v8.bin"),
        ];
        let section = captured_graph_section(true);
        let inline = section.serialize().unwrap();
        let mut pool = BlockPool::new();
        let content_addressed = section.serialize_content_addressed(&mut pool).unwrap();
        for (version, authentic) in (1u8..=8).zip(predecessors) {
            assert_eq!(authentic[4], version);
            let expected =
                format!("unsupported CompactStore section version {version} (supported: 9)");
            assert_rejected_at_all_entry_points(authentic, &expected);
            for current in [&inline, &content_addressed] {
                let altered = altered_header(current, version, current[5]);
                assert_rejected_at_all_entry_points(&altered, &expected);
            }
        }
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn unsupported_gcst_flags_reject_at_all_entry_points() {
        let current = captured_graph_section(true).serialize().unwrap();
        for unknown in [0x04, 0x08, 0x10, 0x20, 0x40, 0x80] {
            for supported in 0..=3 {
                let flags = unknown | supported;
                let altered = altered_header(&current, FORMAT_VERSION, flags);
                assert_rejected_at_all_entry_points(
                    &altered,
                    &format!("unsupported CompactStore flags {flags:#04x}"),
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn current_gcst_flags_roundtrip() {
        for preserves_ids in [false, true] {
            let section = captured_graph_section(preserves_ids);
            let inline = section.serialize().unwrap();
            assert_eq!(inline[5], u8::from(preserves_ids));
            let mut restored = CompactStoreSection::empty();
            restored.deserialize(&inline).unwrap();
            assert_eq!(restored.serialize().unwrap(), inline);
            let mut pool = BlockPool::new();
            let addressed = section.serialize_content_addressed(&mut pool).unwrap();
            assert_eq!(addressed[5], u8::from(preserves_ids) | 0x02);
            let restored = deserialize_content_addressed(&Bytes::from(addressed.clone()), &pool)
                .expect("current content-addressed flags must remain supported");
            let mut restored_pool = BlockPool::new();
            let restored_bytes = CompactStoreSection::new(Arc::new(restored))
                .serialize_content_addressed(&mut restored_pool)
                .unwrap();
            assert_eq!(restored_bytes, addressed);
            assert_eq!(restored_pool.to_bytes(), pool.to_bytes());
        }
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_round_trip_empty() {
        let store = LpgStore::new().unwrap();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));

        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();

        let restored = section2.store().unwrap();
        assert_eq!(restored.node_count(), 0);
        assert_eq!(restored.edge_count(), 0);
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_round_trip_nodes_and_edges() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        store.create_edge(alix, amsterdam, "LIVES_IN");
        store.create_edge(gus, amsterdam, "LIVES_IN");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        assert!(compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(restored.preserves_ids());
        assert_eq!(restored.node_count(), 3);
        assert_eq!(restored.edge_count(), 2);

        // Verify original IDs survive.
        let alix_node = restored.get_node(alix).expect("Alix by original ID");
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(arcstr::ArcStr::from("Alix")))
        );
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("age")),
            Some(&Value::Int64(30))
        );

        // Verify edge traversal.
        let neighbors = restored.neighbors(alix, crate::graph::Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], amsterdam);
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_content_addressed_round_trip_and_dedup() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));
        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));
        store.create_edge(alix, amsterdam, "LIVES_IN");
        store.create_edge(gus, amsterdam, "LIVES_IN");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));

        // Content-addressed serialize: node value blocks go to the pool, only
        // their content ids are inline.
        let mut pool = BlockPool::new();
        let bytes = section.serialize_content_addressed(&mut pool).unwrap();
        assert!(pool.block_count() > 0);

        // Reconstruct the whole store (nodes, edges, ids) from (bytes + pool).
        let restored = deserialize_content_addressed(&Bytes::from(bytes), &pool).unwrap();
        assert!(restored.preserves_ids());
        assert_eq!(restored.node_count(), 3);
        assert_eq!(restored.edge_count(), 2);
        let alix_node = restored.get_node(alix).expect("Alix by original id");
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(arcstr::ArcStr::from("Alix")))
        );
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("age")),
            Some(&Value::Int64(30))
        );
        let neighbors = restored.neighbors(alix, crate::graph::Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], amsterdam);

        // Cross-time dedup: re-serializing an identical store adds no new blocks.
        let blocks = pool.block_count();
        let section2 =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&store).unwrap()));
        section2.serialize_content_addressed(&mut pool).unwrap();
        assert_eq!(pool.block_count(), blocks);
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_content_addressed_persists_through_serialized_pool() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));
        store.create_node(&["City"]);

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let mut pool = BlockPool::new();
        let bytes = section.serialize_content_addressed(&mut pool).unwrap();

        // Persist BOTH the section bytes and the pool, then reload from scratch:
        // the whole store comes back from (bytes + reloaded pool).
        let pool_blob = pool.to_bytes();
        let reloaded_pool = BlockPool::from_bytes(&pool_blob).unwrap();
        let restored = deserialize_content_addressed(&Bytes::from(bytes), &reloaded_pool).unwrap();

        assert_eq!(restored.node_count(), 2);
        let alix_node = restored.get_node(alix).expect("Alix by id");
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("age")),
            Some(&Value::Int64(30))
        );
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_content_addressed_explicit_temporal_round_trip() {
        // An EXPLICIT temporal layout (per-node ranges + validity), unlike the
        // all-open case the other CA test covers. In content-addressed mode the
        // value block is replaced inline by a 32-byte ContentId AND the addendum's
        // per-column hash is skipped — this verifies the reader stays byte-aligned.
        let store = LpgStore::new().unwrap();
        let n = store.create_node(&["Item"]);
        store.set_node_property(n, "score", Value::Int64(0));
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let key = PropertyKey::new("score");
        let temporal = compact.upgrade_nodes_temporal(|id| {
            if id == n {
                vec![(
                    key.clone(),
                    vec![
                        (EpochId::new(10), Value::Int64(100)),
                        (EpochId::new(20), Value::Int64(200)),
                    ],
                )]
            } else {
                Vec::new()
            }
        });
        assert!(
            !temporal.node_tables_by_id[0].is_all_open(),
            "expected an explicit temporal layout"
        );

        let section = CompactStoreSection::new(Arc::new(temporal));
        let mut pool = BlockPool::new();
        let bytes = section.serialize_content_addressed(&mut pool).unwrap();
        let restored = deserialize_content_addressed(&Bytes::from(bytes.clone()), &pool).unwrap();

        // As-of reads round-trip through the content-addressed explicit path.
        assert_eq!(
            restored.get_node_property_at_epoch(n, &key, EpochId::new(5)),
            None
        );
        assert_eq!(
            restored.get_node_property_at_epoch(n, &key, EpochId::new(15)),
            Some(Value::Int64(100))
        );
        assert_eq!(
            restored.get_node_property_at_epoch(n, &key, EpochId::new(25)),
            Some(Value::Int64(200))
        );

        // Determinism (invariant #3): re-serializing the restored store is byte-identical.
        let mut pool2 = BlockPool::new();
        let bytes2 = CompactStoreSection::new(Arc::new(restored))
            .serialize_content_addressed(&mut pool2)
            .unwrap();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_round_trip_without_id_preservation() {
        use crate::graph::compact::from_graph_store;

        let lpg = LpgStore::new().unwrap();
        let a = lpg.create_node(&["Node"]);
        lpg.set_node_property(a, "val", Value::Int64(42));
        let b = lpg.create_node(&["Node"]);
        lpg.set_node_property(b, "val", Value::Int64(99));
        lpg.create_edge(a, b, "LINK");

        let compact = from_graph_store(&lpg).unwrap();
        assert!(!compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(!restored.preserves_ids());
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_crc_integrity() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Test"]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();

        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte in the middle.
        if bytes.len() > 10 {
            bytes[10] ^= 0xFF;
        }

        let mut section2 = CompactStoreSection::empty();
        assert!(section2.deserialize(&bytes).is_err());
    }

    #[test]
    fn test_section_type_and_version() {
        let section = CompactStoreSection::empty();
        assert_eq!(section.section_type(), SectionType::CompactStore);
        assert_eq!(section.version(), FORMAT_VERSION);
        assert!(!section.is_dirty());
        assert_eq!(section.memory_usage(), 0);
    }

    #[test]
    fn test_dirty_tracking() {
        let section = CompactStoreSection::empty();
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    // ── Phase 2c: per-block zone maps ────────────────────────────────

    /// The builder must populate per-block zone maps for every column,
    /// one ZoneMap per block. `1024` rows per block (DEFAULT_BLOCK_ROWS).
    #[test]
    #[cfg(feature = "lpg")]
    fn alix_builder_populates_per_block_zone_maps() {
        let store = LpgStore::new().unwrap();
        // 3000 nodes → 3 blocks (1024 + 1024 + 952).
        for i in 0i64..3000 {
            let n = store.create_node(&["Person"]);
            store.set_node_property(n, "age", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let table = &compact.node_tables_by_id[0];
        let block_zms = table
            .block_zone_maps_for(&PropertyKey::new("age"))
            .expect("per-block stats present");
        assert_eq!(block_zms.len(), 3, "3000 rows should produce 3 blocks");
        assert_eq!(block_zms[0].row_count, 1024);
        assert_eq!(block_zms[1].row_count, 1024);
        assert_eq!(block_zms[2].row_count, 952);
        assert_eq!(block_zms[0].min, Some(Value::Int64(0)));
        assert_eq!(block_zms[0].max, Some(Value::Int64(1023)));
        assert_eq!(block_zms[1].min, Some(Value::Int64(1024)));
        assert_eq!(block_zms[1].max, Some(Value::Int64(2047)));
        assert_eq!(block_zms[2].min, Some(Value::Int64(2048)));
        assert_eq!(block_zms[2].max, Some(Value::Int64(2999)));
    }

    /// v3 round-trip preserves per-block zone maps verbatim.
    #[test]
    #[cfg(feature = "lpg")]
    fn gus_v3_round_trip_preserves_block_zone_maps() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..2500 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let original = &compact.node_tables_by_id[0];
        let original_zms = original
            .block_zone_maps_for(&PropertyKey::new("score"))
            .expect("original block stats")
            .to_vec();

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();
        let restored_table = &restored.node_tables_by_id[0];
        let restored_zms = restored_table
            .block_zone_maps_for(&PropertyKey::new("score"))
            .expect("restored block stats");

        assert_eq!(restored_zms.len(), original_zms.len());
        for (i, (orig, rest)) in original_zms.iter().zip(restored_zms.iter()).enumerate() {
            assert_eq!(orig.row_count, rest.row_count, "row_count mismatch at {i}");
            assert_eq!(
                orig.null_count, rest.null_count,
                "null_count mismatch at {i}"
            );
            assert_eq!(orig.min, rest.min, "min mismatch at {i}");
            assert_eq!(orig.max, rest.max, "max mismatch at {i}");
        }
    }

    /// String columns also get per-block min/max.
    #[test]
    #[cfg(feature = "lpg")]
    fn mia_block_zone_maps_for_string_column() {
        let store = LpgStore::new().unwrap();
        // Use enough nodes to force >= 2 blocks.
        for i in 0u32..1100 {
            let n = store.create_node(&["Tag"]);
            store.set_node_property(n, "name", Value::from(format!("tag_{i:04}")));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let table = &compact.node_tables_by_id[0];
        let block_zms = table
            .block_zone_maps_for(&PropertyKey::new("name"))
            .expect("string column block stats");
        assert_eq!(block_zms.len(), 2);
        assert_eq!(
            block_zms[0].min,
            Some(Value::String(arcstr::ArcStr::from("tag_0000")))
        );
        assert_eq!(
            block_zms[0].max,
            Some(Value::String(arcstr::ArcStr::from("tag_1023")))
        );
        assert_eq!(
            block_zms[1].min,
            Some(Value::String(arcstr::ArcStr::from("tag_1024")))
        );
        assert_eq!(
            block_zms[1].max,
            Some(Value::String(arcstr::ArcStr::from("tag_1099")))
        );
    }

    /// Phase 2b: an unsupported version byte must produce a clean error,
    /// not panic or silently misread the section.
    #[test]
    #[cfg(feature = "lpg")]
    fn rita_unknown_version_returns_clear_error() {
        let store = LpgStore::new().unwrap();
        let _ = store.create_node(&["Item"]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section.serialize().unwrap();
        // Strip CRC, set a version beyond the current writer, recompute CRC.
        let crc_pos = bytes.len() - 4;
        bytes[4] = FORMAT_VERSION + 1;
        let crc = crc32fast::hash(&bytes[..crc_pos]);
        bytes[crc_pos..].copy_from_slice(&crc.to_le_bytes());

        let mut section2 = CompactStoreSection::empty();
        let err = section2
            .deserialize(&bytes)
            .expect_err("expected version error");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported CompactStore section version"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_round_trip_bool_column() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "active", Value::Bool(true));
        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "active", Value::Bool(false));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert_eq!(
            restored.get_node_property(a, &PropertyKey::new("active")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            restored.get_node_property(b, &PropertyKey::new("active")),
            Some(Value::Bool(false))
        );
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn test_round_trip_edge_properties() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "weight", Value::Int64(5));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        // Find the edge via traversal.
        let edges = restored.edges_from(a, crate::graph::Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let edge = restored.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("weight")),
            Some(&Value::Int64(5))
        );
    }

    // ── Additive temporal serialization ─────────────────────────────

    #[test]
    #[cfg(feature = "lpg")]
    fn cold_tier_v9_is_default_version() {
        let store = LpgStore::new().unwrap();
        let n = store.create_node(&["Item"]);
        store.set_node_property(n, "v", Value::Int64(7));
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();
        assert_eq!(bytes[4], 9, "default serialization must be format v9");
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn v9_property_coverage_roundtrips_known_and_unknown_floors() {
        for floor in [Some(EpochId::INITIAL), Some(EpochId::new(17)), None] {
            let store = LpgStore::new().unwrap();
            let node = store.create_node(&["Item"]);
            store.set_node_property(node, "v", Value::Int64(7));
            let compact = from_graph_store_preserving_ids(&store)
                .unwrap()
                .with_property_history_floor(floor);
            let section = CompactStoreSection::new(Arc::new(compact));
            let bytes = section.serialize().unwrap();
            let mut restored = CompactStoreSection::empty();
            restored.deserialize(&bytes).unwrap();
            assert_eq!(restored.store().unwrap().property_history_floor(), floor);
            assert_eq!(restored.serialize().unwrap(), bytes);
        }
    }

    #[test]
    fn current_section_preserves_plural_closed_edge_ids() {
        let mut compact = super::super::builder::CompactStoreBuilder::new()
            .build()
            .expect("empty compact store");
        let id = EdgeId::new(9);
        let row = |from, to| FoldedEdgeRow {
            id,
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: arcstr::ArcStr::from("REPEATS"),
            validity: EpochInterval::closed(EpochId::new(from), EpochId::new(to)),
            properties: FxHashMap::default(),
            raw_properties: FxHashMap::default(),
        };
        let mut closed = FxHashMap::default();
        closed.insert(id, vec![row(1, 2), row(3, 4)]);
        compact.set_closed_edges(closed);
        let section = CompactStoreSection::new(Arc::new(compact));

        let bytes = section.serialize().expect("serialize plural closed edges");
        let mut restored = CompactStoreSection::empty();
        restored
            .deserialize(&bytes)
            .expect("restore plural closed edges");
        let base = restored.store().expect("restored compact store");
        let rows = &base.closed_edges[&id];
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].validity,
            EpochInterval::closed(EpochId::new(1), EpochId::new(2))
        );
        assert_eq!(
            rows[1].validity,
            EpochInterval::closed(EpochId::new(3), EpochId::new(4))
        );
        assert_eq!(restored.serialize().unwrap(), bytes);
    }

    #[test]
    fn current_section_preserves_exact_raw_temporal_properties() {
        let mut compact = super::super::builder::CompactStoreBuilder::new()
            .build()
            .unwrap();
        let id = EdgeId::new(9);
        let property = PropertyKey::new("mixed");
        let history = vec![
            (EpochId::new(1), Value::Int64(7)),
            (EpochId::new(2), Value::String("seven".into())),
        ];
        let row = FoldedEdgeRow {
            id,
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: "R".into(),
            validity: EpochInterval::closed(EpochId::new(1), EpochId::new(3)),
            properties: FxHashMap::default(),
            raw_properties: FxHashMap::from_iter([(
                property.clone(),
                super::super::compaction::RawTemporalColumn::new(history.clone()),
            )]),
        };
        compact.set_closed_edges(FxHashMap::from_iter([(id, vec![row])]));
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&bytes).unwrap();
        let base = restored.store().unwrap();
        assert_eq!(base.property_history_floor(), Some(EpochId::INITIAL));
        assert_eq!(
            base.closed_edges[&id][0].raw_properties[&property].runs_as_history(),
            history
        );
    }

    /// Invariant #3 (determinism): serialize -> deserialize -> serialize is
    /// byte-identical for an all-open base.
    ///
    /// Scoped to deterministic numeric codecs. `Dict` (string) columns have a
    /// pre-existing non-canonical serialization order that changes across a
    /// deserialize→re-serialize round-trip — a codec-layer concern orthogonal
    /// to the temporal format and out of scope for this cluster. Two columns
    /// also exercise the v4 key-sorted (deterministic) column order.
    #[test]
    #[cfg(feature = "lpg")]
    fn cold_tier_v4_all_open_round_trip_byte_identical() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..50 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
            store.set_node_property(n, "rank", Value::Int64(100 - i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes1 = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes1).unwrap();
        let bytes2 = section2.serialize().unwrap();
        assert_eq!(
            bytes1, bytes2,
            "v4 all-open round-trip must be byte-identical"
        );
    }

    /// The explicit temporal v4 path round-trips a multi-version node: as-of
    /// reads are preserved and re-serialization is byte-identical.
    #[test]
    fn cold_tier_v4_explicit_temporal_round_trips_as_of() {
        use crate::graph::compact::column::ColumnCodec;
        use crate::graph::compact::id::encode_node_id;
        use crate::graph::compact::node_table::NodeTable;
        use crate::graph::compact::schema::{ColumnDef, ColumnType, TableSchema};
        use crate::graph::compact::temporal_column::TemporalColumn;
        use grafeo_common::types::{EpochId, EpochInterval};
        use grafeo_common::utils::hash::FxHashMap;

        // One Person node with a 3-version history on "score" (rows [0,3)).
        let mut cols: FxHashMap<PropertyKey, TemporalColumn> = FxHashMap::default();
        cols.insert(
            PropertyKey::new("score"),
            TemporalColumn::new(
                ColumnCodec::raw_i64(vec![100, 200, 300]),
                vec![
                    EpochInterval::closed(EpochId::new(10), EpochId::new(20)),
                    EpochInterval::closed(EpochId::new(20), EpochId::new(30)),
                    EpochInterval::open(EpochId::new(30)),
                ],
            ),
        );
        let schema = TableSchema::new(
            "Person",
            0,
            vec![ColumnDef::new("score", ColumnType::Int64)],
        );
        // Per-column ranges: node 0 occupies rows [0,3) in the "score" column.
        let mut column_ranges = FxHashMap::default();
        column_ranges.insert(PropertyKey::new("score"), vec![(0u32, 3u32)]);
        let nt = NodeTable::from_temporal_columns(
            schema,
            cols,
            FxHashMap::default(),
            FxHashMap::default(),
            Some(column_ranges),
            1,
        );
        let mut label_to_table_id = FxHashMap::default();
        label_to_table_id.insert(arcstr::ArcStr::from("Person"), 0u16);
        let store = CompactStore::new(
            vec![nt],
            label_to_table_id,
            vec![],
            FxHashMap::default(),
            vec![arcstr::ArcStr::from("Person")],
            vec![],
            Statistics::new(),
        );

        let section = CompactStoreSection::new(Arc::new(store));
        let bytes = section.serialize().unwrap();
        assert_eq!(bytes[4], FORMAT_VERSION);

        let mut s2 = CompactStoreSection::empty();
        s2.deserialize(&bytes).unwrap();
        let restored = s2.store().unwrap();

        let id = encode_node_id(0, 0);
        let score = PropertyKey::new("score");
        assert_eq!(
            restored.get_node_property_at_epoch(id, &score, EpochId::new(15)),
            Some(Value::Int64(100))
        );
        assert_eq!(
            restored.get_node_property_at_epoch(id, &score, EpochId::new(25)),
            Some(Value::Int64(200))
        );
        assert_eq!(
            restored.get_node_property_at_epoch(id, &score, EpochId::new(35)),
            Some(Value::Int64(300))
        );
        // current read = the node's open row
        assert_eq!(
            restored.get_node_property(id, &score),
            Some(Value::Int64(300))
        );

        // Byte-identical re-serialization (determinism through the explicit path).
        let bytes_restored = s2.serialize().unwrap();
        assert_eq!(
            bytes, bytes_restored,
            "explicit temporal round-trip must be byte-identical"
        );
    }

    /// Closed + open edge lives must survive serialize → deserialize. Without a
    /// v5 rel addendum this fails: persist writes the current CSR only.
    #[test]
    #[cfg(feature = "lpg")]
    fn cold_tier_asof_edges_survive_serialize() {
        use crate::graph::Direction;
        use crate::graph::compact::layered::LayeredStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(base, 1000, 1000).unwrap();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        let ab =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(ab, "w", Value::Int64(1), EpochId::new(10));
        overlay.set_edge_property_at_epoch(ab, "w", Value::Int64(2), EpochId::new(20));
        let _ac =
            overlay.create_edge_versioned(a, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(40));
        assert!(overlay.delete_edge_at_epoch(ab, EpochId::new(40)));
        overlay.set_epoch(EpochId::new(40));
        layered.merge_overlay_temporal().unwrap();

        let live = layered.base_store_arc();
        let before_asof = live.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        let before_now = live.neighbors_at_epoch(a, Direction::Outgoing, EpochId::PENDING);
        assert!(
            before_asof.contains(&b),
            "pre-persist as-of must see deleted AB"
        );
        assert!(before_asof.contains(&c));
        assert!(!before_now.contains(&b));
        assert!(before_now.contains(&c));
        assert_eq!(
            live.get_edge_property_at_epoch(ab, &PropertyKey::new("w"), EpochId::new(15)),
            Some(Value::Int64(1))
        );
        assert_eq!(
            live.get_edge_property_at_epoch(ab, &PropertyKey::new("w"), EpochId::new(25)),
            Some(Value::Int64(2))
        );

        let section = CompactStoreSection::new(live);
        let bytes = section.serialize().unwrap();
        assert_eq!(bytes[4], 9, "default serialization must be format v9");

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        let after_asof = restored.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        let after_now = restored.neighbors_at_epoch(a, Direction::Outgoing, EpochId::PENDING);
        assert_eq!(
            after_asof, before_asof,
            "as-of neighbors must survive persist"
        );
        assert_eq!(
            after_now, before_now,
            "current neighbors must survive persist"
        );
        assert_eq!(
            restored.get_edge_property_at_epoch(ab, &PropertyKey::new("w"), EpochId::new(15)),
            Some(Value::Int64(1))
        );
        assert_eq!(
            restored.get_edge_property_at_epoch(ab, &PropertyKey::new("w"), EpochId::new(25)),
            Some(Value::Int64(2))
        );
        assert!(restored.get_edge_at_epoch(ab, EpochId::new(15)).is_some());
        assert!(restored.get_edge_at_epoch(ab, EpochId::new(40)).is_none());
        assert!(restored.get_edge(ab).is_none());

        let bytes2 = section2.serialize().unwrap();
        assert_eq!(
            bytes, bytes2,
            "v5 temporal edge round-trip must be byte-identical"
        );
    }

    /// Structure-only packed-placed closes persist via the rel addendum, not
    /// a duplicated sidecar row. Property-bearing closes still use the sidecar.
    #[test]
    #[cfg(feature = "lpg")]
    fn structure_only_packed_closed_survives_without_sidecar_row() {
        use crate::graph::Direction;
        use crate::graph::compact::layered::LayeredStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(base, 1000, 1000).unwrap();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        let ab =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        let _ac =
            overlay.create_edge_versioned(a, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(40));
        assert!(overlay.delete_edge_at_epoch(ab, EpochId::new(40)));
        overlay.set_epoch(EpochId::new(40));
        layered.merge_overlay_temporal().unwrap();

        let live = layered.base_store_arc();
        assert_eq!(live.closed_sidecar_len(), 0);
        assert!(live.get_edge_at_epoch(ab, EpochId::new(15)).is_some());

        let section = CompactStoreSection::new(live);
        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();
        assert_eq!(restored.closed_sidecar_len(), 0);
        assert!(restored.get_edge_at_epoch(ab, EpochId::new(15)).is_some());
        assert!(restored.get_edge_at_epoch(ab, EpochId::new(40)).is_none());
        assert!(restored.get_edge(ab).is_none());
        let after = restored.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        assert!(after.contains(&b) && after.contains(&c));
        let bytes2 = section2.serialize().unwrap();
        assert_eq!(bytes, bytes2);
    }
}
