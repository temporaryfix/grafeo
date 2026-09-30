//! Per-label columnar node storage.
//!
//! Each `NodeTable` stores all nodes of a single label as typed columns.
//! Nodes are addressed by row offset; the `NodeId` encodes (table_id, offset).

use grafeo_common::types::{EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::column::ColumnCodec;
use super::id::encode_node_id;
use super::schema::TableSchema;
use super::temporal_column::TemporalColumn;
use super::zone_map::ZoneMap;

/// Per-label columnar storage for nodes.
///
/// All nodes sharing a label are stored in a single `NodeTable` with one
/// [`ColumnCodec`] per property. Row offsets are combined with the table ID
/// via [`encode_node_id`] to produce globally unique [`NodeId`] values.
#[derive(Debug)]
pub struct NodeTable {
    /// Schema describing the label, table ID, and column definitions.
    schema: TableSchema,
    /// Per-property temporal columns. A freshly-built base wraps each current
    /// value column as all-open intervals (see [`TemporalColumn::all_open`]),
    /// so current reads route through each node's open row; temporal
    /// compaction (SP2) replaces a node's single open row with its version run.
    columns: FxHashMap<PropertyKey, TemporalColumn>,
    /// Per-column min/max statistics for predicate pushdown.
    zone_maps: FxHashMap<PropertyKey, ZoneMap>,
    /// Per-block min/max statistics, one entry per logical block in each
    /// column. Populated by the builder and by v3 deserialization; left
    /// empty after v1/v2 compat reads (Phase 4 will fall back to whole
    /// column scans in that case).
    block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>>,
    /// Per-column, per-node physical row ranges. `None` is the all-open identity
    /// layout — in every column node `i` occupies exactly physical row `i`,
    /// count 1 — used by every base built from current values. Temporal
    /// compaction (SP2) replaces this with explicit per-column `(row_start,
    /// row_count)` ranges: each property folds independently, so a node spans a
    /// different number of version rows in each column (a node lacking a property
    /// has count 0 there — no placeholder rows). `column_ranges[key][node]` gives
    /// that node's run in `key`'s column.
    column_ranges: Option<FxHashMap<PropertyKey, Vec<(u32, u32)>>>,
    /// Number of nodes in the table (the logical row count; with the all-open
    /// layout this equals each column's physical row count).
    len: usize,
}

impl NodeTable {
    /// Creates an empty table with the given schema.
    #[must_use]
    pub fn new(schema: TableSchema) -> Self {
        Self {
            schema,
            columns: FxHashMap::default(),
            zone_maps: FxHashMap::default(),
            block_zone_maps: FxHashMap::default(),
            column_ranges: None,
            len: 0,
        }
    }

    /// This table's schema (label, table id, column definitions).
    #[must_use]
    pub fn schema(&self) -> &TableSchema {
        &self.schema
    }

    /// Creates a table from pre-built columns and zone maps with no
    /// per-block stats. Used by legacy v1/v2 deserialization paths.
    #[must_use]
    pub fn from_columns(
        schema: TableSchema,
        columns: FxHashMap<PropertyKey, ColumnCodec>,
        zone_maps: FxHashMap<PropertyKey, ZoneMap>,
        len: usize,
    ) -> Self {
        Self::from_columns_with_block_stats(schema, columns, zone_maps, FxHashMap::default(), len)
    }

    /// Creates a table from pre-built value columns, zone maps, and per-block
    /// zone maps. Used by the builder and by v1–v3 deserialization.
    ///
    /// Each value [`ColumnCodec`] is wrapped as an all-open [`TemporalColumn`]
    /// (every row valid `[INITIAL, PENDING)`) and the table adopts the all-open
    /// identity row layout — so current-read semantics are byte-for-byte
    /// preserved while the storage becomes temporal.
    #[must_use]
    pub fn from_columns_with_block_stats(
        schema: TableSchema,
        columns: FxHashMap<PropertyKey, ColumnCodec>,
        zone_maps: FxHashMap<PropertyKey, ZoneMap>,
        block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>>,
        len: usize,
    ) -> Self {
        let columns = columns
            .into_iter()
            .map(|(key, codec)| (key, TemporalColumn::all_open(codec)))
            .collect();
        Self {
            schema,
            columns,
            zone_maps,
            block_zone_maps,
            column_ranges: None,
            len,
        }
    }

    /// Creates a table from pre-built temporal columns and explicit per-column
    /// per-node row ranges. Used by SP2 compaction and by v4 deserialization to
    /// install a base whose nodes span multiple version rows. `column_ranges` of
    /// `None` selects the all-open identity layout; when `Some`, every column key
    /// must have a `Vec` of `len` `(row_start, row_count)` entries (count 0 for a
    /// node that lacks the property).
    #[must_use]
    pub fn from_temporal_columns(
        schema: TableSchema,
        columns: FxHashMap<PropertyKey, TemporalColumn>,
        zone_maps: FxHashMap<PropertyKey, ZoneMap>,
        block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>>,
        column_ranges: Option<FxHashMap<PropertyKey, Vec<(u32, u32)>>>,
        len: usize,
    ) -> Self {
        Self {
            schema,
            columns,
            zone_maps,
            block_zone_maps,
            column_ranges: collapse_identity_column_ranges(column_ranges, len),
            len,
        }
    }

    /// Whether this table is the all-open identity layout — built from current
    /// values with no history (identity row ranges + every column all-open).
    /// Such a table serializes via the compact v4 all-open path.
    #[must_use]
    pub fn is_all_open(&self) -> bool {
        self.column_ranges.is_none() && self.columns.values().all(TemporalColumn::is_all_open)
    }

    /// The explicit per-column per-node `(row_start, row_count)` ranges, or
    /// `None` for the all-open identity layout (node `i` -> `(i, 1)` in every
    /// column). For serialization.
    #[must_use]
    pub fn explicit_column_ranges(&self) -> Option<&FxHashMap<PropertyKey, Vec<(u32, u32)>>> {
        self.column_ranges.as_ref()
    }

    /// Returns the number of nodes in this table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the table contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the table ID encoded into every [`NodeId`] from this table.
    #[must_use]
    pub fn table_id(&self) -> u16 {
        self.schema.table_id
    }

    /// Returns the label shared by all nodes in this table.
    #[must_use]
    pub fn label(&self) -> &str {
        self.schema.label.as_str()
    }

    /// Generates a [`NodeId`] for every row in this table.
    ///
    /// The IDs are returned in row order (offset 0, 1, 2, ...).
    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let table_id = self.schema.table_id;
        (0..self.len)
            .map(|offset| encode_node_id(table_id, offset as u64))
            .collect()
    }

    /// Returns the physical row range `(row_start, row_count)` the node at
    /// logical `offset` occupies in the column `key`, or `None` if the node is
    /// out of range (or has no rows in that column). The all-open base uses the
    /// identity layout: node `i` -> `(i, 1)` in every column.
    #[must_use]
    pub fn column_node_range(&self, key: &PropertyKey, offset: usize) -> Option<(usize, usize)> {
        match &self.column_ranges {
            None => (offset < self.len).then_some((offset, 1)),
            Some(map) => map
                .get(key)
                .and_then(|ranges| ranges.get(offset))
                .map(|&(start, count)| (start as usize, count as usize)),
        }
    }

    /// Returns the node's current property value (the value of its open
    /// interval) at the given logical row offset.
    ///
    /// Returns `None` if the column does not exist, the offset is out of bounds,
    /// or the node has been fully retracted (no open row).
    #[must_use]
    pub fn get_property(&self, offset: usize, key: &PropertyKey) -> Option<Value> {
        let (start, count) = self.column_node_range(key, offset)?;
        self.columns.get(key)?.current_value(start, count)
    }

    /// Returns all current properties for the node at the given logical offset.
    ///
    /// Out-of-bounds offsets produce an empty map.
    #[must_use]
    pub fn get_all_properties(&self, offset: usize) -> FxHashMap<PropertyKey, Value> {
        let mut props = FxHashMap::default();
        for (key, col) in &self.columns {
            if let Some((start, count)) = self.column_node_range(key, offset)
                && let Some(value) = col.current_value(start, count)
            {
                props.insert(key.clone(), value);
            }
        }
        props
    }

    /// Returns the node's property value valid at `epoch` (as-of read).
    ///
    /// Routes through the node's per-column physical row range and that column's
    /// per-row validity, block-pruned by the epoch zone-map. For the all-open
    /// base this equals the current value at every real epoch.
    #[must_use]
    pub fn get_property_at_epoch(
        &self,
        offset: usize,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        let (start, count) = self.column_node_range(key, offset)?;
        self.columns
            .get(key)?
            .value_in_range_as_of(start, count, epoch)
    }

    /// Returns all of the node's property values valid at `epoch` (as-of read).
    ///
    /// Out-of-bounds offsets produce an empty map. Properties whose version
    /// history has a gap at `epoch` are omitted (a removed property reads as
    /// absent), mirroring [`get_all_properties`](Self::get_all_properties).
    #[must_use]
    pub fn get_all_properties_at_epoch(
        &self,
        offset: usize,
        epoch: EpochId,
    ) -> FxHashMap<PropertyKey, Value> {
        let mut props = FxHashMap::default();
        for (key, col) in &self.columns {
            if let Some((start, count)) = self.column_node_range(key, offset)
                && let Some(value) = col.value_in_range_as_of(start, count, epoch)
            {
                props.insert(key.clone(), value);
            }
        }
        props
    }

    /// Returns the raw `u64` stored at the given row offset for a bit-packed column.
    ///
    /// This is primarily useful for foreign-key columns where the raw encoded ID
    /// is needed rather than the `Value::Int64` conversion. Returns `None` for
    /// non-[`BitPacked`](ColumnCodec::BitPacked) columns or out-of-bounds offsets.
    #[must_use]
    pub fn get_raw_u64(&self, offset: usize, key: &PropertyKey) -> Option<u64> {
        let (start, count) = self.column_node_range(key, offset)?;
        self.columns.get(key)?.current_raw_u64(start, count)
    }

    /// Reconstructs the `(epoch, value)` history of node `offset` in column
    /// `key` from the column's runs (see [`TemporalColumn::runs_as_history`]).
    /// Empty if the node or column is absent. Used by the SP2 merge to pull a
    /// base node's history back out.
    #[must_use]
    pub fn column_node_history(&self, offset: usize, key: &PropertyKey) -> Vec<(EpochId, Value)> {
        let Some((start, count)) = self.column_node_range(key, offset) else {
            return Vec::new();
        };
        self.columns
            .get(key)
            .map_or_else(Vec::new, |col| col.runs_as_history(start, count))
    }

    /// Rebuilds this all-open table into a temporal one: each numeric column
    /// folds its property's per-node history (`node_histories`, offset-ordered)
    /// into a temporal column via [`fold_property_across_nodes`]; other-typed
    /// columns keep their current-value all-open form (identity ranges). The
    /// caller preserves edges and id maps. The SP2 merge's per-table installer.
    ///
    /// [`fold_property_across_nodes`]: super::compaction::fold_property_across_nodes
    #[must_use]
    pub fn upgraded_temporal(
        self,
        node_histories: &[Vec<(PropertyKey, Vec<(EpochId, Value)>)>],
    ) -> Self {
        let node_count = self.len;
        let mut new_columns: FxHashMap<PropertyKey, TemporalColumn> = FxHashMap::default();
        let mut column_ranges: FxHashMap<PropertyKey, Vec<(u32, u32)>> = FxHashMap::default();
        for (key, tcol) in self.columns {
            if codec_is_temporal_foldable(tcol.values()) {
                let per_node: Vec<Vec<(EpochId, Value)>> = node_histories
                    .iter()
                    .map(|h| {
                        h.iter()
                            .find(|(k, _)| *k == key)
                            .map_or_else(Vec::new, |(_, v)| v.clone())
                    })
                    .collect();
                if let Some((col, ranges)) =
                    super::compaction::fold_property_across_nodes(&per_node)
                {
                    new_columns.insert(key.clone(), col);
                    column_ranges.insert(key, ranges);
                    continue;
                }
            }
            // Keep the all-open column with identity ranges.
            // reason: node counts are bounded by u32::MAX (section format).
            #[allow(clippy::cast_possible_truncation)]
            let identity: Vec<(u32, u32)> = (0..node_count as u32).map(|i| (i, 1)).collect();
            new_columns.insert(key.clone(), tcol);
            column_ranges.insert(key, identity);
        }
        Self::from_temporal_columns(
            self.schema,
            new_columns,
            self.zone_maps,
            self.block_zone_maps,
            Some(column_ranges),
            node_count,
        )
    }

    /// Returns the zone map for a column, if one exists.
    #[must_use]
    pub fn zone_map(&self, key: &PropertyKey) -> Option<&ZoneMap> {
        self.zone_maps.get(key)
    }

    /// Returns the value codec (current-value projection) for a property, if it
    /// exists. Backs the current-read scan path: in the all-open base, physical
    /// rows are 1:1 with nodes, so scan offsets over this codec are node offsets.
    #[must_use]
    pub fn column(&self, key: &PropertyKey) -> Option<&ColumnCodec> {
        self.columns.get(key).map(TemporalColumn::values)
    }

    /// Maps sorted codec matches back to logical nodes, retaining only each
    /// node's current row. Physical temporal offsets are never entity IDs.
    pub(crate) fn current_matching_offsets(
        &self,
        key: &PropertyKey,
        physical: Vec<usize>,
    ) -> Vec<usize> {
        if physical.is_empty() {
            return physical;
        }
        let Some(column) = self.columns.get(key) else {
            return Vec::new();
        };
        let Some(ranges) = &self.column_ranges else {
            // Identity layout says one row per node, not that the row is live.
            if column.is_all_open() {
                return physical;
            }
            return physical
                .into_iter()
                .filter(|&offset| {
                    offset < self.len && column.open_row_in(offset, 1) == Some(offset)
                })
                .collect();
        };
        let Some(ranges) = ranges.get(key) else {
            return Vec::new();
        };
        ranges
            .iter()
            .take(self.len)
            .enumerate()
            .filter_map(|(offset, &(start, count))| {
                let current = column.open_row_in(start as usize, count as usize)?;
                physical.binary_search(&current).is_ok().then_some(offset)
            })
            .collect()
    }

    /// Returns all property keys present in this table.
    #[must_use]
    pub fn property_keys(&self) -> Vec<PropertyKey> {
        self.columns.keys().cloned().collect()
    }

    /// Returns all temporal columns (for serialization).
    #[must_use]
    pub fn temporal_columns(&self) -> &FxHashMap<PropertyKey, TemporalColumn> {
        &self.columns
    }

    /// Returns all zone maps (for serialization).
    #[must_use]
    pub fn zone_maps(&self) -> &FxHashMap<PropertyKey, ZoneMap> {
        &self.zone_maps
    }

    /// Returns the per-block zone maps for a column, if any have been
    /// computed. Returns `None` for columns built from a v1 or v2 stream
    /// (which carry no per-block stats); the planner should fall back
    /// to whole-column zone-map pruning in that case.
    #[must_use]
    pub fn block_zone_maps_for(&self, key: &PropertyKey) -> Option<&[ZoneMap]> {
        self.block_zone_maps.get(key).map(Vec::as_slice)
    }

    /// Returns all per-block zone maps (for serialization).
    #[must_use]
    pub fn block_zone_maps(&self) -> &FxHashMap<PropertyKey, Vec<ZoneMap>> {
        &self.block_zone_maps
    }

    /// Returns an estimate of heap memory used by all columns (values + per-row
    /// validity) plus the explicit row-range index, in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let column_bytes: usize = self.columns.values().map(TemporalColumn::heap_bytes).sum();
        let range_bytes = self.column_ranges.as_ref().map_or(0, |map| {
            map.values()
                .map(|r| r.len() * std::mem::size_of::<(u32, u32)>())
                .sum::<usize>()
        });
        column_bytes + range_bytes
    }
}

/// Drops a range map that is the identity layout in every column (`node i` →
/// `(i, 1)`). Readers already treat `None` as that layout; keeping the vecs
/// would charge 8 B/node/column after a merge that never forked history.
fn collapse_identity_column_ranges(
    column_ranges: Option<FxHashMap<PropertyKey, Vec<(u32, u32)>>>,
    len: usize,
) -> Option<FxHashMap<PropertyKey, Vec<(u32, u32)>>> {
    let map = column_ranges?;
    if map.values().all(|ranges| ranges_are_identity(ranges, len)) {
        None
    } else {
        Some(map)
    }
}

fn ranges_are_identity(ranges: &[(u32, u32)], len: usize) -> bool {
    if ranges.len() != len {
        return false;
    }
    ranges
        .iter()
        .enumerate()
        .all(|(i, &(start, count))| u32::try_from(i).is_ok_and(|i| start == i && count == 1))
}

/// Whether a value codec's type can be folded into a temporal column by the cold
/// codec: ints (`RawI64`/`BitPacked`), floats, strings (`Dict`), bools (`Bitmap`),
/// and float vectors (`Float32Vector`). Columns of other types (e.g. int8 vectors,
/// which decode to `Value::List` rather than `Value::Vector`) keep their all-open
/// current-value form during a temporal merge.
fn codec_is_temporal_foldable(codec: &ColumnCodec) -> bool {
    matches!(
        codec,
        ColumnCodec::RawI64(_)
            | ColumnCodec::BitPacked(_)
            | ColumnCodec::Float64(_)
            | ColumnCodec::Dict(_)
            | ColumnCodec::Bitmap(_)
            | ColumnCodec::Float32Vector { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::BitPackedInts;
    use crate::graph::compact::id::decode_node_id;
    use crate::graph::compact::schema::{ColumnDef, ColumnType};

    /// Helper: build a `NodeTable` with 5 rows and two bit-packed columns
    /// ("rating" at 4 bits, "count" at 32 bits).
    fn sample_table() -> NodeTable {
        let schema = TableSchema::new(
            "Movie",
            3,
            vec![
                ColumnDef::new("rating", ColumnType::UInt { bits: 4 }),
                ColumnDef::new("count", ColumnType::UInt { bits: 32 }),
            ],
        );

        let ratings = vec![1u64, 5, 10, 15, 3];
        let counts = vec![100u64, 200, 300, 400, 500];

        let mut columns = FxHashMap::default();
        columns.insert(
            PropertyKey::new("rating"),
            ColumnCodec::BitPacked(BitPackedInts::pack(&ratings)),
        );
        columns.insert(
            PropertyKey::new("count"),
            ColumnCodec::BitPacked(BitPackedInts::pack(&counts)),
        );

        let mut zone_maps = FxHashMap::default();
        zone_maps.insert(
            PropertyKey::new("rating"),
            ZoneMap {
                min: Some(Value::Int64(1)),
                max: Some(Value::Int64(15)),
                null_count: 0,
                row_count: 5,
            },
        );
        zone_maps.insert(
            PropertyKey::new("count"),
            ZoneMap {
                min: Some(Value::Int64(100)),
                max: Some(Value::Int64(500)),
                null_count: 0,
                row_count: 5,
            },
        );

        NodeTable::from_columns(schema, columns, zone_maps, 5)
    }

    #[test]
    fn test_len_and_label() {
        let table = sample_table();
        assert_eq!(table.len(), 5);
        assert!(!table.is_empty());
        assert_eq!(table.table_id(), 3);
        assert_eq!(table.label(), "Movie");
    }

    #[test]
    fn test_empty_table() {
        let schema = TableSchema::new("Empty", 0, vec![]);
        let table = NodeTable::new(schema);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
        assert!(table.node_ids().is_empty());
    }

    #[test]
    fn test_node_ids() {
        let table = sample_table();
        let ids = table.node_ids();
        assert_eq!(ids.len(), 5);

        for (i, id) in ids.iter().enumerate() {
            let (tid, offset) = decode_node_id(*id);
            assert_eq!(tid, 3);
            assert_eq!(offset, i as u64);
        }
    }

    #[test]
    fn test_get_property() {
        let table = sample_table();

        // First row
        assert_eq!(
            table.get_property(0, &PropertyKey::new("rating")),
            Some(Value::Int64(1))
        );
        assert_eq!(
            table.get_property(0, &PropertyKey::new("count")),
            Some(Value::Int64(100))
        );

        // Last row
        assert_eq!(
            table.get_property(4, &PropertyKey::new("rating")),
            Some(Value::Int64(3))
        );
        assert_eq!(
            table.get_property(4, &PropertyKey::new("count")),
            Some(Value::Int64(500))
        );
    }

    #[test]
    fn test_get_all_properties() {
        let table = sample_table();
        let props = table.get_all_properties(2);

        assert_eq!(props.len(), 2);
        assert_eq!(props[&PropertyKey::new("rating")], Value::Int64(10));
        assert_eq!(props[&PropertyKey::new("count")], Value::Int64(300));
    }

    #[test]
    fn test_get_raw_u64() {
        let table = sample_table();

        // Raw u64 is useful for FK lookups: verify it returns the original packed value.
        assert_eq!(table.get_raw_u64(0, &PropertyKey::new("count")), Some(100));
        assert_eq!(table.get_raw_u64(3, &PropertyKey::new("count")), Some(400));
        assert_eq!(table.get_raw_u64(4, &PropertyKey::new("rating")), Some(3));
    }

    #[test]
    fn test_out_of_bounds_returns_none() {
        let table = sample_table();

        // Offset beyond table length.
        assert_eq!(table.get_property(5, &PropertyKey::new("rating")), None);
        assert_eq!(table.get_property(999, &PropertyKey::new("count")), None);
        assert_eq!(table.get_raw_u64(5, &PropertyKey::new("count")), None);

        // Non-existent property key.
        assert_eq!(table.get_property(0, &PropertyKey::new("missing")), None);
        assert_eq!(table.get_raw_u64(0, &PropertyKey::new("missing")), None);

        // get_all_properties on out-of-bounds offset returns empty map.
        let props = table.get_all_properties(100);
        assert!(props.is_empty());
    }

    #[test]
    fn test_zone_map_lookup() {
        let table = sample_table();

        let zm = table.zone_map(&PropertyKey::new("rating")).unwrap();
        assert_eq!(zm.min, Some(Value::Int64(1)));
        assert_eq!(zm.max, Some(Value::Int64(15)));
        assert_eq!(zm.row_count, 5);

        assert!(table.zone_map(&PropertyKey::new("missing")).is_none());
    }

    #[test]
    fn test_column_lookup() {
        let table = sample_table();

        assert!(table.column(&PropertyKey::new("rating")).is_some());
        assert!(table.column(&PropertyKey::new("count")).is_some());
        assert!(table.column(&PropertyKey::new("missing")).is_none());
    }

    #[test]
    fn test_property_keys() {
        let table = sample_table();
        let mut keys = table.property_keys();
        // Sort for deterministic assertion (hash map iteration order is unspecified).
        keys.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].as_ref(), "count");
        assert_eq!(keys[1].as_ref(), "rating");
    }

    #[test]
    fn test_all_open_table_identity_row_ranges() {
        // A table built from current values uses the all-open identity layout:
        // node i -> (row_start = i, row_count = 1); current reads are unchanged.
        let table = sample_table(); // 5 rows
        let rating = PropertyKey::new("rating");
        assert_eq!(table.column_node_range(&rating, 0), Some((0, 1)));
        assert_eq!(table.column_node_range(&rating, 4), Some((4, 1)));
        assert_eq!(table.column_node_range(&rating, 5), None);
        // current reads route through the node's open row.
        assert_eq!(
            table.get_property(0, &PropertyKey::new("rating")),
            Some(Value::Int64(1))
        );
        assert_eq!(
            table.get_property(4, &PropertyKey::new("count")),
            Some(Value::Int64(500))
        );
    }

    #[test]
    fn test_from_temporal_columns_explicit_ranges() {
        use grafeo_common::types::EpochInterval;
        // One column, two nodes: node 0 has 2 versions (rows 0,1), node 1 has 1 (row 2).
        let mut cols = FxHashMap::default();
        cols.insert(
            PropertyKey::new("score"),
            TemporalColumn::new(
                ColumnCodec::raw_i64(vec![100, 200, 300]),
                vec![
                    EpochInterval::closed(EpochId::new(10), EpochId::new(20)),
                    EpochInterval::open(EpochId::new(20)),
                    EpochInterval::open(EpochId::new(5)),
                ],
            ),
        );
        let schema = TableSchema::new("T", 0, vec![ColumnDef::new("score", ColumnType::Int64)]);
        let score = PropertyKey::new("score");
        let mut column_ranges = FxHashMap::default();
        column_ranges.insert(score.clone(), vec![(0u32, 2u32), (2u32, 1u32)]);
        let table = NodeTable::from_temporal_columns(
            schema,
            cols,
            FxHashMap::default(),
            FxHashMap::default(),
            Some(column_ranges),
            2,
        );
        assert!(!table.is_all_open());
        assert_eq!(table.column_node_range(&score, 0), Some((0, 2)));
        assert_eq!(table.column_node_range(&score, 1), Some((2, 1)));
        assert!(table.explicit_column_ranges().is_some());
        // as-of reads honor the explicit ranges
        assert_eq!(
            table.get_property_at_epoch(0, &score, EpochId::new(15)),
            Some(Value::Int64(100))
        );
        assert_eq!(
            table.get_property_at_epoch(0, &score, EpochId::new(25)),
            Some(Value::Int64(200))
        );
        assert_eq!(
            table.get_property_at_epoch(1, &score, EpochId::new(7)),
            Some(Value::Int64(300))
        );
        // current read = each node's open row
        assert_eq!(table.get_property(0, &score), Some(Value::Int64(200)));
        assert_eq!(table.get_property(1, &score), Some(Value::Int64(300)));
    }

    #[test]
    fn test_identity_column_ranges_collapse() {
        use grafeo_common::types::EpochInterval;
        let mut cols = FxHashMap::default();
        cols.insert(
            PropertyKey::new("score"),
            TemporalColumn::new(
                ColumnCodec::raw_i64(vec![1, 2]),
                vec![EpochInterval::open(EpochId::new(10)); 2],
            ),
        );
        let schema = TableSchema::new("T", 0, vec![ColumnDef::new("score", ColumnType::Int64)]);
        let mut column_ranges = FxHashMap::default();
        column_ranges.insert(PropertyKey::new("score"), vec![(0u32, 1u32), (1u32, 1u32)]);
        let table = NodeTable::from_temporal_columns(
            schema,
            cols,
            FxHashMap::default(),
            FxHashMap::default(),
            Some(column_ranges),
            2,
        );
        assert!(table.explicit_column_ranges().is_none());
        assert!(!table.is_all_open());
        assert_eq!(
            table.column_node_range(&PropertyKey::new("score"), 1),
            Some((1, 1))
        );
    }

    #[test]
    fn test_all_open_table_is_all_open() {
        assert!(sample_table().is_all_open());
    }

    #[test]
    fn test_get_property_at_epoch_all_open_equals_current() {
        // Invariant (SP1-5 slice 2): for an all-open base, the as-of read at
        // every real epoch equals the current read.
        let table = sample_table();
        for ep in [EpochId::INITIAL, EpochId::new(1), EpochId::new(999_999)] {
            assert_eq!(
                table.get_property_at_epoch(0, &PropertyKey::new("rating"), ep),
                table.get_property(0, &PropertyKey::new("rating"))
            );
            assert_eq!(
                table.get_property_at_epoch(2, &PropertyKey::new("count"), ep),
                Some(Value::Int64(300))
            );
            // all properties at epoch == all current properties
            assert_eq!(
                table.get_all_properties_at_epoch(3, ep),
                table.get_all_properties(3)
            );
        }
    }
}
