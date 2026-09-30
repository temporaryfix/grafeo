//! CompactStore: a read-only columnar store for memory-constrained environments.
//!
//! Implements [`GraphStore`](crate::graph::traits::GraphStore) using per-label
//! columnar tables and double-indexed CSR adjacency. Designed for static
//! snapshot data in WASM, edge workers, and embedded devices.
//! Fully behind `#[cfg(feature = "compact-store")]`.

/// Builder API for constructing a [`CompactStore`] from raw data.
pub mod builder;
/// Columnar codecs for node and edge properties.
pub mod column;
/// Compaction: fold VersionLog history into cold validity-interval rows.
pub mod compaction;
/// Content-addressed block manifest for cross-time structural sharing (dedup).
pub mod content_dedup;
/// Physical content hash (BLAKE3 Merkle leaf) for cold-base blocks.
pub mod content_hash;
/// Compressed Sparse Row (CSR) adjacency representation.
pub mod csr;
/// Container section serialization for the layered overlay deletion log.
#[cfg(feature = "lpg")]
pub mod deletions_section;
mod graph_store_impl;
/// Node/edge ID encoding and decoding helpers.
pub mod id;
/// Two-layer store: columnar base + mutable LPG overlay.
#[cfg(feature = "lpg")]
pub mod layered;
/// Per-label node tables with columnar property storage.
pub mod node_table;
/// Per-type relationship tables backed by forward/backward CSR.
pub mod rel_table;
/// Schema definitions for node tables and edge schemas.
pub mod schema;
/// Container section serialization for CompactStore.
pub mod section;
#[cfg(all(test, feature = "lpg"))]
mod slim_edge_identity_tests;
/// Claim / statement table (seven-column opaque claim layout, opaque hashes).
#[cfg(feature = "statement-table")]
pub mod statement_table;
/// Temporal columnar block: values + per-row epoch validity + epoch zone-map.
pub mod temporal_column;
#[cfg(test)]
mod tests;
/// Zone maps for skip-pruning predicate evaluation.
pub mod zone_map;

pub use builder::{CompactStoreBuilder, from_graph_store, from_graph_store_preserving_ids};

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, EpochInterval, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use self::node_table::NodeTable;
use self::rel_table::RelTable;
use crate::graph::Direction;
use crate::statistics::Statistics;

/// One node table's whole-state values at a single epoch — the columnar result
/// of [`CompactStore::scrub_at_epoch`].
///
/// Node order is offset order (`0, 1, …`): `node_ids[i]` is node `i`'s original
/// id, and `columns[key][i]` is that node's value for `key` at the epoch (`None`
/// where the property is absent then). No per-node `Node`/map is materialized —
/// this is the allocation-light scrub path for 60fps time-travel.
#[derive(Debug, Clone)]
pub struct NodeTableScrub {
    /// The table's label.
    pub label: ArcStr,
    /// Original node ids in node (offset) order.
    pub node_ids: Vec<NodeId>,
    /// Per property, the value at the epoch for each node, in node order.
    pub columns: FxHashMap<PropertyKey, Vec<Option<Value>>>,
}

/// One relationship type's whole-state values at a single epoch — the edge
/// half of [`GraphScrub`].
///
/// Row `i` is `edge_ids[i]` from `src_ids[i]` to `dst_ids[i]`;
/// `columns[key][i]` is that edge's value for `key` at the epoch (`None` when
/// the property is absent then).
#[derive(Debug, Clone)]
pub struct RelTableScrub {
    /// Relationship type (e.g. `"KNOWS"`).
    pub edge_type: ArcStr,
    /// Edge ids in frame order.
    pub edge_ids: Vec<EdgeId>,
    /// Source node ids, aligned to [`Self::edge_ids`].
    pub src_ids: Vec<NodeId>,
    /// Destination node ids, aligned to [`Self::edge_ids`].
    pub dst_ids: Vec<NodeId>,
    /// Per property, the value at the epoch for each edge, in frame order.
    pub columns: FxHashMap<PropertyKey, Vec<Option<Value>>>,
}

/// Whole-state as-of scrub: columnar node frames plus edge frames.
///
/// [`CompactStore::scrub_at_epoch`] remains the node-only 60fps path;
/// [`GraphScrub`] is the whole-graph snapshot exposed to callers.
#[derive(Debug, Clone, Default)]
pub struct GraphScrub {
    /// Per-label node tables.
    pub nodes: Vec<NodeTableScrub>,
    /// Per-type relationship tables.
    pub edges: Vec<RelTableScrub>,
}

impl RelTableScrub {
    fn with_keys(edge_type: ArcStr, keys: &[PropertyKey], cap: usize) -> Self {
        Self {
            edge_type,
            edge_ids: Vec::with_capacity(cap),
            src_ids: Vec::with_capacity(cap),
            dst_ids: Vec::with_capacity(cap),
            columns: keys
                .iter()
                .map(|k| (k.clone(), Vec::with_capacity(cap)))
                .collect(),
        }
    }

    fn push_row(
        &mut self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        value: impl Fn(&PropertyKey) -> Option<Value>,
    ) {
        self.edge_ids.push(id);
        self.src_ids.push(src);
        self.dst_ids.push(dst);
        let keys: Vec<PropertyKey> = self.columns.keys().cloned().collect();
        for key in keys {
            let v = value(&key);
            if let Some(col) = self.columns.get_mut(&key) {
                col.push(v);
            }
        }
    }

    #[cfg(feature = "lpg")]
    fn push_edge(&mut self, edge: &crate::graph::lpg::Edge) {
        let row = self.edge_ids.len();
        self.edge_ids.push(edge.id);
        self.src_ids.push(edge.src);
        self.dst_ids.push(edge.dst);
        for (key, values) in &mut self.columns {
            values.push(edge.properties.get(key).cloned());
        }
        for (key, value) in edge.properties.iter() {
            if !self.columns.contains_key(key) {
                let mut column = vec![None; row];
                column.push(Some(value.clone()));
                self.columns.insert(key.clone(), column);
            }
        }
    }
}

/// Groups materialized as-of edges into per-type [`RelTableScrub`] frames.
#[cfg(feature = "lpg")]
pub(crate) fn rel_frames_from_edges(
    edges: impl IntoIterator<Item = crate::graph::lpg::Edge>,
) -> Vec<RelTableScrub> {
    let mut frames: Vec<RelTableScrub> = Vec::new();
    let mut index: FxHashMap<ArcStr, usize> = FxHashMap::default();
    for edge in edges {
        let slot = if let Some(&i) = index.get(&edge.edge_type) {
            i
        } else {
            let i = frames.len();
            index.insert(edge.edge_type.clone(), i);
            frames.push(RelTableScrub {
                edge_type: edge.edge_type.clone(),
                edge_ids: Vec::new(),
                src_ids: Vec::new(),
                dst_ids: Vec::new(),
                columns: FxHashMap::default(),
            });
            i
        };
        frames[slot].push_edge(&edge);
    }
    frames
}

/// Groups materialized nodes into per-label columnar scrub frames.
fn node_frames_from_nodes(
    nodes: impl IntoIterator<Item = crate::graph::lpg::Node>,
) -> Vec<NodeTableScrub> {
    let mut frames: Vec<NodeTableScrub> = Vec::new();
    let mut by_label: FxHashMap<ArcStr, usize> = FxHashMap::default();
    for node in nodes {
        for label in &node.labels {
            let slot = if let Some(&slot) = by_label.get(label) {
                slot
            } else {
                let slot = frames.len();
                by_label.insert(label.clone(), slot);
                frames.push(NodeTableScrub {
                    label: label.clone(),
                    node_ids: Vec::new(),
                    columns: FxHashMap::default(),
                });
                slot
            };
            let frame = &mut frames[slot];
            let row = frame.node_ids.len();
            frame.node_ids.push(node.id);
            for (key, column) in &mut frame.columns {
                column.push(node.properties.get(key).cloned());
            }
            for (key, value) in &node.properties {
                if !frame.columns.contains_key(key) {
                    let mut column = vec![None; row];
                    column.push(Some(value.clone()));
                    frame.columns.insert(key.clone(), column);
                }
            }
        }
    }
    frames
}

/// A read-only columnar graph store.
///
/// Node data is stored in per-label [`NodeTable`]s and edge data in per-type
/// [`RelTable`]s. The store is immutable after construction: use
/// [`CompactStoreBuilder`] to populate it from raw data.
pub struct CompactStore {
    /// Proven lower coverage boundary. None marks legacy serialized input that
    /// did not record whether earlier versions had already been collected.
    property_history_floor: Option<EpochId>,
    /// Node tables indexed by table_id for O(1) lookup from NodeId.
    node_tables_by_id: Vec<NodeTable>,
    /// table_id lookup from label string (for nodes_by_label).
    label_to_table_id: FxHashMap<ArcStr, u16>,
    /// Relationship tables indexed by rel_table_id for O(1) lookup from EdgeId.
    rel_tables_by_id: Vec<RelTable>,
    /// rel_table_id lookup from edge type string (one edge type may span
    /// multiple src/dst label combinations, so the value is a Vec).
    edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
    /// Lookup: table ID -> label.
    table_id_to_label: Vec<ArcStr>,
    /// Lookup: rel table ID -> edge type.
    rel_table_id_to_type: Vec<ArcStr>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the source.
    src_rel_table_ids: Vec<Vec<u16>>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the destination.
    dst_rel_table_ids: Vec<Vec<u16>>,
    /// Cached statistics.
    statistics: Arc<Statistics>,

    // ── ID-preserving maps (for layered store integration) ──────────
    /// Maps original `NodeId` to (table_id, row_offset). Present when the
    /// store was built with [`from_graph_store_preserving_ids`].
    node_id_map: Option<FxHashMap<NodeId, (u16, u64)>>,
    /// Maps original `EdgeId` to (rel_table_id, csr_position).
    edge_id_map: Option<FxHashMap<EdgeId, (u16, u64)>>,
    /// Reverse: table_id index -> vec of original `NodeId` per row offset.
    node_offset_to_id: Option<Vec<Vec<NodeId>>>,
    /// Reverse: rel_table_id index -> vec of original `EdgeId` per CSR position.
    edge_offset_to_id: Option<Vec<Vec<EdgeId>>>,
    /// Structural and property history for every temporally compacted node,
    /// including nodes absent from the current node tables because they were
    /// deleted. Persisted in the v6 temporal-node sidecar.
    temporal_nodes: FxHashMap<NodeId, Vec<compaction::FoldedNodeRow>>,
    /// Derived current label membership for temporal nodes. This keeps the
    /// compound physical table key used by the column layout from leaking as a
    /// logical user label. Rebuilt from `temporal_nodes`; never serialized.
    temporal_label_index: FxHashMap<ArcStr, Vec<NodeId>>,
    /// Deleted-but-retained edge lifetimes from a temporal merge.
    /// Persisted in the v5 closed-edge addendum.
    ///
    /// Property-bearing rows and packed orphans live here. Structure-only
    /// packed-placed closed lives are dropped after merge and reconstructed
    /// from the packed-closed index (RelTable properties are the
    /// current-CSR projection, not fat history).
    closed_edges: FxHashMap<EdgeId, Vec<compaction::FoldedEdgeRow>>,
    /// `src → closed edge ids` (as-of hops must not scan the whole sidecar).
    closed_out: FxHashMap<NodeId, Vec<EdgeId>>,
    /// `dst → closed edge ids`.
    closed_in: FxHashMap<NodeId, Vec<EdgeId>>,
}

impl std::fmt::Debug for CompactStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactStore")
            .field("node_tables_by_id", &self.node_tables_by_id)
            .field("rel_tables_by_id", &self.rel_tables_by_id)
            .field("table_id_to_label", &self.table_id_to_label)
            .field("rel_table_id_to_type", &self.rel_table_id_to_type)
            .finish_non_exhaustive()
    }
}

fn rel_matches_types(rt: &RelTable, types: &[String]) -> bool {
    types.is_empty()
        || types
            .iter()
            .any(|t| rt.edge_type().eq_ignore_ascii_case(t.as_str()))
}

/// Appends one folded structural row's property log without confusing an
/// authentic repeated event with the copy produced by inclusive boundary
/// clipping.
///
/// Adjacent lifetimes may meet at one epoch (`previous.to == current.from`).
/// When the current row carries property events at that boundary, those events
/// are the authoritative single copy of the source log; every entry at the same
/// epoch reconstructed from the preceding row came from inclusively clipping
/// that source log into both rows. Remove that preceding boundary suffix, then
/// retain the current row verbatim. This is deliberately based on structural
/// row origin, never on value equality, so ordered
/// same-epoch/same-value writes inside a row survive exactly.
fn append_folded_row_history(
    history: &mut Vec<(EpochId, Value)>,
    row_history: Vec<(EpochId, Value)>,
    previous_to: Option<EpochId>,
    current_from: EpochId,
) {
    if previous_to == Some(current_from)
        && row_history
            .first()
            .is_some_and(|(epoch, _)| *epoch == current_from)
    {
        while history
            .last()
            .is_some_and(|(epoch, _)| *epoch == current_from)
        {
            history.pop();
        }
    }
    history.extend(row_history);
}

impl CompactStore {
    /// Earliest proven complete property-history view; legacy sections return None.
    #[must_use]
    pub fn property_history_floor(&self) -> Option<EpochId> {
        self.property_history_floor
    }

    /// Stamps coverage after a detached current-only or full-history build.
    pub(crate) fn with_property_history_floor(mut self, floor: Option<EpochId>) -> Self {
        self.property_history_floor = floor.filter(|floor| *floor != EpochId::PENDING);
        self
    }

    /// Creates a new `CompactStore` from pre-built components.
    ///
    /// Prefer using [`CompactStoreBuilder`] which validates schemas and
    /// computes statistics automatically. This constructor is `pub(crate)`
    /// because it assumes all invariants are already satisfied.
    #[must_use]
    pub(crate) fn new(
        node_tables_by_id: Vec<NodeTable>,
        label_to_table_id: FxHashMap<ArcStr, u16>,
        rel_tables_by_id: Vec<RelTable>,
        edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
        table_id_to_label: Vec<ArcStr>,
        rel_table_id_to_type: Vec<ArcStr>,
        statistics: Statistics,
    ) -> Self {
        // Pre-compute src/dst rel_table_id mappings per node table_id.
        let node_table_count = node_tables_by_id.len();
        let mut src_rel_table_ids = vec![Vec::new(); node_table_count];
        let mut dst_rel_table_ids = vec![Vec::new(); node_table_count];

        debug_assert!(
            rel_tables_by_id.len() <= usize::from(id::MAX_TABLE_ID) + 1,
            "rel table count {} exceeds 15-bit limit; caller must validate",
            rel_tables_by_id.len()
        );
        for (rel_idx, rt) in rel_tables_by_id.iter().enumerate() {
            // Caller (CompactStoreBuilder::build) validates table count fits u16.
            let rel_id = u16::try_from(rel_idx).expect("caller validated table count");

            let src_tid = rt.src_table_id() as usize;
            let dst_tid = rt.dst_table_id() as usize;
            if src_tid < node_table_count {
                src_rel_table_ids[src_tid].push(rel_id);
            }
            if dst_tid < node_table_count {
                dst_rel_table_ids[dst_tid].push(rel_id);
            }
        }

        Self {
            property_history_floor: Some(EpochId::INITIAL),
            node_tables_by_id,
            label_to_table_id,
            rel_tables_by_id,
            edge_type_to_rel_id,
            table_id_to_label,
            rel_table_id_to_type,
            src_rel_table_ids,
            dst_rel_table_ids,
            statistics: Arc::new(statistics),
            node_id_map: None,
            edge_id_map: None,
            node_offset_to_id: None,
            edge_offset_to_id: None,
            temporal_nodes: FxHashMap::default(),
            temporal_label_index: FxHashMap::default(),
            closed_edges: FxHashMap::default(),
            closed_out: FxHashMap::default(),
            closed_in: FxHashMap::default(),
        }
    }

    /// Resolves a table_id to its [`NodeTable`].
    #[inline]
    fn resolve_node_table(&self, table_id: u16) -> Option<&NodeTable> {
        self.node_tables_by_id.get(table_id as usize)
    }

    /// Resolves a rel_table_id to its [`RelTable`].
    #[inline]
    fn resolve_rel_table(&self, rel_table_id: u16) -> Option<&RelTable> {
        self.rel_tables_by_id.get(rel_table_id as usize)
    }

    /// Returns a reference to the node table for the given label, if any.
    #[must_use]
    pub fn node_table(&self, label: &str) -> Option<&NodeTable> {
        let &tid = self.label_to_table_id.get(label)?;
        self.node_tables_by_id.get(tid as usize)
    }

    /// Returns a reference to the first relationship table for the given edge type.
    ///
    /// When an edge type spans multiple label pairs, use [`Self::rel_tables_for_type`]
    /// to get all matching tables.
    #[must_use]
    pub fn rel_table(&self, edge_type: &str) -> Option<&RelTable> {
        let rids = self.edge_type_to_rel_id.get(edge_type)?;
        let &rid = rids.first()?;
        self.rel_tables_by_id.get(rid as usize)
    }

    /// Returns all relationship tables for the given edge type.
    #[must_use]
    pub fn rel_tables_for_type(&self, edge_type: &str) -> Vec<&RelTable> {
        self.edge_type_to_rel_id
            .get(edge_type)
            .map(|rids| {
                rids.iter()
                    .filter_map(|&rid| self.rel_tables_by_id.get(rid as usize))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the label for a given table ID, if valid.
    #[must_use]
    pub fn label_for_table_id(&self, table_id: u16) -> Option<&ArcStr> {
        self.table_id_to_label.get(table_id as usize)
    }

    /// Returns the edge type for a given rel table ID, if valid.
    #[must_use]
    pub fn edge_type_for_rel_table_id(&self, rel_table_id: u16) -> Option<&ArcStr> {
        self.rel_table_id_to_type.get(rel_table_id as usize)
    }

    /// Collects edges from snapshot RelTables for a given node in a direction.
    ///
    /// When ID-preserving, compact-encoded node/edge ids are translated back
    /// to the originals. Packed current 1-hop already stores original
    /// [`EdgeId`]s — those must not go through [`Self::to_original_edge_id`].
    fn collect_edges(
        &self,
        node_table_id: u16,
        node_offset: u32,
        direction: Direction,
    ) -> Vec<(NodeId, EdgeId)> {
        let tid = node_table_id as usize;
        let mut results = Vec::new();

        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(rel_ids) = self.src_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                self.extend_current_rel_edges(&mut results, rt, rt.edges_from_source(node_offset));
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if let Some(edges) = rt.edges_to_target(node_offset) {
                    self.extend_current_rel_edges(&mut results, rt, edges);
                }
            }
        }

        results
    }

    /// Dest-only current 1-hop: CSR targets, no [`EdgeId`] lookup.
    fn collect_neighbors(
        &self,
        node_table_id: u16,
        node_offset: u32,
        direction: Direction,
    ) -> Vec<NodeId> {
        let tid = node_table_id as usize;
        let mut results = Vec::new();
        let preserve = self.preserves_ids();

        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(rel_ids) = self.src_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                for &dst in rt.current_targets(node_offset) {
                    let nid = id::encode_node_id(rt.dst_table_id(), u64::from(dst));
                    results.push(if preserve {
                        self.to_original_node_id(nid)
                    } else {
                        nid
                    });
                }
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if let Some(bwd) = rt.bwd() {
                    for &src in bwd.neighbors(node_offset) {
                        let nid = id::encode_node_id(rt.src_table_id(), u64::from(src));
                        results.push(if preserve {
                            self.to_original_node_id(nid)
                        } else {
                            nid
                        });
                    }
                } else if let Some(edges) = rt.edges_to_target(node_offset) {
                    for (src, _) in edges {
                        results.push(if preserve {
                            self.to_original_node_id(src)
                        } else {
                            src
                        });
                    }
                }
            }
        }

        results
    }

    fn extend_current_rel_edges(
        &self,
        out: &mut Vec<(NodeId, EdgeId)>,
        rt: &RelTable,
        edges: impl IntoIterator<Item = (NodeId, EdgeId)>,
    ) {
        let packed_ids = rt.current_uses_original_ids();
        let preserve = self.preserves_ids();
        for (target, eid) in edges {
            let target = if preserve {
                self.to_original_node_id(target)
            } else {
                target
            };
            let eid = if preserve && !packed_ids {
                self.to_original_edge_id(eid)
            } else {
                eid
            };
            out.push((target, eid));
        }
    }

    /// Returns a rough estimate of heap memory used by the snapshot data
    /// (node columns + CSR structures + edge property columns), in bytes.
    ///
    /// Does not include `FxHashMap` overhead or schema metadata. For precise
    /// measurement, use a heap profiler.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let node_bytes: usize = self
            .node_tables_by_id
            .iter()
            .map(|nt| nt.memory_bytes())
            .sum();
        let rel_bytes: usize = self
            .rel_tables_by_id
            .iter()
            .map(|rt| rt.memory_bytes())
            .sum();
        let id_map_bytes = self.id_map_memory_bytes();
        let closed_bytes: usize = self
            .closed_edges
            .values()
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        row.properties
                            .values()
                            .map(temporal_column::TemporalColumn::heap_bytes)
                            .sum::<usize>()
                            + row
                                .raw_properties
                                .values()
                                .map(compaction::RawTemporalColumn::heap_bytes)
                                .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum();
        let temporal_node_bytes: usize = self
            .temporal_nodes
            .values()
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        row.properties
                            .values()
                            .map(temporal_column::TemporalColumn::heap_bytes)
                            .sum::<usize>()
                            + row
                                .raw_properties
                                .values()
                                .map(compaction::RawTemporalColumn::heap_bytes)
                                .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum();
        node_bytes + rel_bytes + id_map_bytes + closed_bytes + temporal_node_bytes
    }

    // ── ID-preserving accessors ────────────────────────────────────

    /// Returns `true` if original IDs are preserved (built via
    /// [`from_graph_store_preserving_ids`]).
    #[must_use]
    pub fn preserves_ids(&self) -> bool {
        self.node_id_map.is_some()
    }

    /// Attaches ID maps to an already-built `CompactStore`.
    pub(crate) fn closed_edges(&self) -> &FxHashMap<EdgeId, Vec<compaction::FoldedEdgeRow>> {
        &self.closed_edges
    }

    pub(crate) fn temporal_nodes(&self) -> &FxHashMap<NodeId, Vec<compaction::FoldedNodeRow>> {
        &self.temporal_nodes
    }

    pub(crate) fn set_temporal_nodes(
        &mut self,
        nodes: FxHashMap<NodeId, Vec<compaction::FoldedNodeRow>>,
    ) {
        self.temporal_nodes = nodes;
        self.reindex_temporal_labels();
    }

    fn reindex_temporal_labels(&mut self) {
        let mut index: FxHashMap<ArcStr, Vec<NodeId>> = FxHashMap::default();
        for (&id, rows) in &self.temporal_nodes {
            let Some(row) = rows.iter().find(|row| row.validity.is_open()) else {
                continue;
            };
            let labels = row
                .label_versions
                .last()
                .map_or(row.labels.as_slice(), |(_, labels)| labels.as_slice());
            for label in labels {
                index.entry(label.clone()).or_default().push(id);
            }
        }
        for ids in index.values_mut() {
            ids.sort_unstable();
            ids.dedup();
        }
        self.temporal_label_index = index;
    }

    pub(crate) fn set_closed_edges(
        &mut self,
        closed: FxHashMap<EdgeId, Vec<compaction::FoldedEdgeRow>>,
    ) {
        self.closed_edges = closed;
        self.reindex_closed_edges();
    }

    /// Drops structure-only packed-placed closed rows from the sidecar.
    ///
    /// RelTable property columns are the current-CSR projection, so rows
    /// with property history must stay in `closed_edges`. Topology for the
    /// dropped rows is recovered from packed fat `edge_ids`.
    #[cfg(any(test, feature = "lpg"))]
    fn prune_packed_structure_only_closed(&mut self) {
        let mut retained = FxHashMap::default();
        for (id, mut rows) in std::mem::take(&mut self.closed_edges) {
            rows.retain(|row| {
                !row.properties.is_empty()
                    || !row.raw_properties.is_empty()
                    || !self.closed_row_in_packed(row)
            });
            if !rows.is_empty() {
                retained.insert(id, rows);
            }
        }
        self.closed_edges = retained;
        self.reindex_closed_edges();
    }

    fn packed_closed_get(&self, id: EdgeId) -> Option<(u16, u32)> {
        if self.closed_edges.contains_key(&id) {
            return None;
        }
        for (i, rt) in self.rel_tables_by_id.iter().enumerate() {
            if let Some(pos) = rt.packed_closed_pos(id) {
                return u16::try_from(i).ok().map(|rel| (rel, pos));
            }
        }
        None
    }

    fn reindex_closed_edges(&mut self) {
        self.closed_out.clear();
        self.closed_in.clear();
        for (id, rows) in &self.closed_edges {
            let Some(row) = rows.iter().find(|row| !self.closed_row_in_packed(row)) else {
                continue;
            };
            self.closed_out.entry(row.src).or_default().push(*id);
            self.closed_in.entry(row.dst).or_default().push(*id);
        }
    }

    /// True when this closed row already lives in a packed RelTable fat run.
    fn closed_row_in_packed(&self, row: &compaction::FoldedEdgeRow) -> bool {
        match self.place_closed_row(row) {
            Some((i, _, _)) => self
                .rel_tables_by_id
                .get(i)
                .is_some_and(|rt| rt.packed_fwd().is_some()),
            None => false,
        }
    }

    /// Hop-index size (orphans only). Packed-placed closed rows are excluded.
    #[cfg(all(test, feature = "lpg"))]
    pub(crate) fn closed_hop_index_len(&self) -> usize {
        self.closed_out.values().map(Vec::len).sum::<usize>()
            + self.closed_in.values().map(Vec::len).sum::<usize>()
    }

    pub(crate) fn set_id_maps(
        &mut self,
        node_id_map: FxHashMap<NodeId, (u16, u64)>,
        edge_id_map: FxHashMap<EdgeId, (u16, u64)>,
        node_offset_to_id: Vec<Vec<NodeId>>,
        edge_offset_to_id: Vec<Vec<EdgeId>>,
    ) {
        self.node_id_map = Some(node_id_map);
        self.edge_id_map = Some(edge_id_map);
        self.node_offset_to_id = Some(node_offset_to_id);
        self.edge_offset_to_id = Some(edge_offset_to_id);
        self.remap_edge_ids_to_packed();
    }

    /// After dropping the derived current CSR, `edge_id_map` positions must
    /// be packed fat-run indexes so [`Self::get_edge`] / as-of `row_contains`
    /// hit the open prefix (and closed tails) instead of an empty `fwd`.
    fn remap_edge_ids_to_packed(&mut self) {
        let remaps: Vec<(u16, Vec<(EdgeId, u64)>)> = self
            .rel_tables_by_id
            .iter()
            .enumerate()
            .filter_map(|(i, rt)| {
                if !rt.current_from_packed() {
                    return None;
                }
                let packed = rt.packed_fwd()?;
                let rel_id = u16::try_from(i).ok()?;
                let entries: Vec<(EdgeId, u64)> = packed
                    .edge_ids()
                    .iter()
                    .enumerate()
                    .filter(|(pos, _)| packed.interval(*pos).is_some_and(|iv| iv.is_open()))
                    .filter_map(|(pos, &id)| Some((id, u64::try_from(pos).ok()?)))
                    .collect();
                Some((rel_id, entries))
            })
            .collect();
        self.drop_duplicate_open_id_rev();
        if remaps.is_empty() {
            return;
        }
        if let Some(map) = self.edge_id_map.as_mut() {
            for (rel_id, entries) in &remaps {
                map.retain(|_, (rid, _)| rid != rel_id);
                for &(id, pos) in entries {
                    map.insert(id, (*rel_id, pos));
                }
            }
        }
        if let Some(rev) = self.edge_offset_to_id.as_mut() {
            for (i, rt) in self.rel_tables_by_id.iter().enumerate() {
                if !rt.current_from_packed() {
                    continue;
                }
                let Some(packed) = rt.packed_fwd() else {
                    continue;
                };
                let Some(slot) = rev.get_mut(i) else {
                    continue;
                };
                slot.clear();
                slot.resize(packed.num_versions(), EdgeId::INVALID);
                for (pos, &id) in packed.edge_ids().iter().enumerate() {
                    slot[pos] = id;
                }
            }
        }
        self.drop_duplicate_open_id_rev();
    }

    /// Slimmed tables already store original ids on `open_edge_ids`.
    fn drop_duplicate_open_id_rev(&mut self) {
        let Some(rev) = self.edge_offset_to_id.as_mut() else {
            return;
        };
        for (i, rt) in self.rel_tables_by_id.iter().enumerate() {
            if rt.open_edge_ids_empty() {
                continue;
            }
            if let Some(slot) = rev.get_mut(i) {
                slot.clear();
                slot.shrink_to_fit();
            }
        }
    }

    /// Resolves an input `NodeId` to (table_id, offset).
    ///
    /// When ID-preserving, looks up the original ID in the map.
    /// Otherwise, decodes the compact-encoded bits.
    #[inline]
    pub(crate) fn resolve_node(&self, id: NodeId) -> Option<(u16, u64)> {
        if let Some(ref map) = self.node_id_map {
            map.get(&id).copied()
        } else {
            Some(id::decode_node_id(id))
        }
    }

    /// Resolves an input `EdgeId` to (rel_table_id, csr_position).
    #[inline]
    pub(crate) fn resolve_edge(&self, id: EdgeId) -> Option<(u16, u64)> {
        if let Some(ref map) = self.edge_id_map {
            map.get(&id).copied()
        } else {
            Some(id::decode_edge_id(id))
        }
    }

    /// Translates a compact-encoded `NodeId` (from internal CSR/table lookups)
    /// back to the original preserved ID. No-op when not ID-preserving.
    #[inline]
    pub(crate) fn to_original_node_id(&self, compact_id: NodeId) -> NodeId {
        if let Some(ref offsets) = self.node_offset_to_id {
            let (table_id, offset) = id::decode_node_id(compact_id);
            offsets
                .get(table_id as usize)
                .and_then(|v| v.get(usize::try_from(offset).ok()?))
                .copied()
                .unwrap_or(compact_id)
        } else {
            compact_id
        }
    }

    /// Translates a compact-encoded `EdgeId` back to the original preserved ID.
    #[inline]
    pub(crate) fn to_original_edge_id(&self, compact_id: EdgeId) -> EdgeId {
        if let Some(ref offsets) = self.edge_offset_to_id {
            let (rel_table_id, csr_pos) = id::decode_edge_id(compact_id);
            offsets
                .get(rel_table_id as usize)
                .and_then(|v| v.get(usize::try_from(csr_pos).ok()?))
                .copied()
                .unwrap_or(compact_id)
        } else {
            compact_id
        }
    }

    /// Approximate heap cost of the ID maps.
    fn id_map_memory_bytes(&self) -> usize {
        // ~24 bytes per entry (key + value) in FxHashMap, plus Vec overhead.
        let node_map = self.node_id_map.as_ref().map_or(0, |m| m.len() * 24);
        let edge_map = self.edge_id_map.as_ref().map_or(0, |m| m.len() * 24);
        let node_rev = self
            .node_offset_to_id
            .as_ref()
            .map_or(0, |v| v.iter().map(|inner| inner.len() * 8).sum());
        let edge_rev = self
            .edge_offset_to_id
            .as_ref()
            .map_or(0, |v| v.iter().map(|inner| inner.len() * 8).sum());
        node_map + edge_map + node_rev + edge_rev
    }

    /// Reconstructs a base node's full per-property history from its temporal
    /// columns (an all-open column yields a single `[(INITIAL, value)]` version;
    /// removal gaps become `Null` tombstones). Used by the SP2 merge to fold a
    /// base-resident node's history into the new temporal base.
    #[must_use]
    pub fn node_property_history(&self, id: NodeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        if let Some(history) = self.temporal_node_history(id) {
            return history.properties.into_iter().collect();
        }
        let Some((table_id, offset)) = self.resolve_node(id) else {
            return Vec::new();
        };
        let Some(nt) = self.resolve_node_table(table_id) else {
            return Vec::new();
        };
        let Ok(row) = usize::try_from(offset) else {
            return Vec::new();
        };
        let mut result = Vec::new();
        for key in nt.property_keys() {
            let history = nt.column_node_history(row, &key);
            if !history.is_empty() {
                result.push((key, history));
            }
        }
        result
    }

    /// Reconstructs a node's structural and property history from the v6
    /// temporal-node sidecar. Returns `None` for legacy/all-open bases.
    pub(crate) fn temporal_node_history(&self, id: NodeId) -> Option<compaction::NodeFullHistory> {
        let rows = self.temporal_nodes.get(&id)?;
        let mut history = compaction::NodeFullHistory::default();
        let mut previous_to = None;
        for row in rows {
            history.labels.clone_from(&row.labels);
            history.label_versions.extend(row.label_versions.clone());
            history.lifetimes.push(compaction::EdgeLifetime::new(
                row.validity.from(),
                (!row.validity.is_open()).then_some(row.validity.to()),
            ));
            for (key, column) in &row.properties {
                let property_history = history.properties.entry(key.clone()).or_default();
                append_folded_row_history(
                    property_history,
                    column.runs_as_history(0, column.len()),
                    previous_to,
                    row.validity.from(),
                );
            }
            for (key, column) in &row.raw_properties {
                let property_history = history.properties.entry(key.clone()).or_default();
                append_folded_row_history(
                    property_history,
                    column.runs_as_history(),
                    previous_to,
                    row.validity.from(),
                );
            }
            previous_to = (!row.validity.is_open()).then(|| row.validity.to());
        }
        history.label_versions.sort_by_key(|(epoch, _)| *epoch);
        if let Some((_, labels)) = history.label_versions.last() {
            history.labels.clone_from(labels);
        }
        Some(history)
    }

    /// Every node id retained by the temporal sidecar (open or closed).
    #[must_use]
    pub fn temporal_node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.temporal_nodes.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Installs folded structural/property node history for current and closed
    /// node ids after rebuilding the current node tables.
    ///
    /// Rejects ambiguous lifetime boundaries that leave a row without an
    /// explicit label image: the current section decoder would otherwise
    /// synthesize an additional image when that row is reopened.
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn install_temporal_nodes(
        mut self,
        mut history_for: impl FnMut(NodeId) -> compaction::NodeFullHistory,
        ids: impl IntoIterator<Item = NodeId>,
    ) -> Result<Self, String> {
        let mut temporal = FxHashMap::default();
        for id in ids {
            let rows = compaction::fold_node_rows(id, &history_for(id));
            if let Some(row) = rows.iter().find(|row| row.label_versions.is_empty()) {
                return Err(format!(
                    "node {id} has an ambiguous label-image boundary at {:?}: folded lifetime has no complete image",
                    row.validity.from()
                ));
            }
            if !rows.is_empty() {
                temporal.insert(id, rows);
            }
        }
        self.set_temporal_nodes(temporal);
        Ok(self)
    }

    /// Temporal row containing `epoch`, treating `PENDING` as the current open
    /// lifetime.
    pub(crate) fn temporal_node_row_at(
        &self,
        id: NodeId,
        epoch: EpochId,
    ) -> Option<&compaction::FoldedNodeRow> {
        self.temporal_nodes.get(&id)?.iter().find(|row| {
            if epoch == EpochId::PENDING {
                row.validity.is_open()
            } else {
                row.validity.contains(epoch)
            }
        })
    }

    /// Materializes one temporal-node row at `epoch`.
    pub(crate) fn node_from_temporal(
        &self,
        row: &compaction::FoldedNodeRow,
        epoch: EpochId,
    ) -> crate::graph::lpg::Node {
        let mut node = crate::graph::lpg::Node::new(row.id);
        let labels = row
            .label_versions
            .iter()
            .rev()
            .find(|(from, _)| *from <= epoch)
            .map_or(row.labels.as_slice(), |(_, labels)| labels.as_slice());
        for label in labels {
            node.add_label(label.clone());
        }
        for (key, column) in &row.properties {
            if let Some(value) = column.value_in_range_as_of(0, column.len(), epoch) {
                node.set_property(key.clone(), value);
            }
        }
        for (key, column) in &row.raw_properties {
            if let Some(value) = column.value_as_of(epoch) {
                node.set_property(key.clone(), value);
            }
        }
        node
    }

    /// Rebuilds every node table's numeric columns into temporal columns folded
    /// from each node's full history (`history_for`, keyed by the node's
    /// ORIGINAL id), keeping other columns all-open and preserving edges, id
    /// maps, and node offsets. The SP2 temporal-merge installer: build an
    /// all-open base for structure, then upgrade its node columns.
    #[must_use]
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn upgrade_nodes_temporal(
        mut self,
        history_for: impl Fn(NodeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)>,
    ) -> Self {
        let tables = std::mem::take(&mut self.node_tables_by_id);
        let mut upgraded = Vec::with_capacity(tables.len());
        for (idx, table) in tables.into_iter().enumerate() {
            let table_id = u16::try_from(idx).unwrap_or(0);
            let node_count = table.len();
            let node_histories: Vec<Vec<(PropertyKey, Vec<(EpochId, Value)>)>> = (0..node_count)
                .map(|off| {
                    let compact_id = id::encode_node_id(table_id, off as u64);
                    history_for(self.to_original_node_id(compact_id))
                })
                .collect();
            upgraded.push(table.upgraded_temporal(&node_histories));
        }
        self.node_tables_by_id = upgraded;
        self
    }

    /// Stamps live CSR rows with folded structural validity, packs Option A
    /// adjacency (open prefix + closed tails), and rebuilds `RelTable.fwd`
    /// from the open prefix (derived current CSR).
    ///
    /// Property columns stay all-open (current-value projection). Closed-
    /// interval property history lives on [`compaction::FoldedEdgeRow`] for
    /// deleted lifetimes. v5 persist writes packed adjacency + the closed
    /// sidecar so as-of survives save/load.
    ///
    /// `history_for` is keyed by the original `EdgeId`. `extra_ids` covers
    /// overlay-deleted and previously-closed edges that `from_graph_store`
    /// dropped from the current snapshot.
    #[must_use]
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn upgrade_rels_temporal(
        mut self,
        mut history_for: impl FnMut(EdgeId) -> compaction::EdgeFullHistory,
        extra_ids: impl IntoIterator<Item = EdgeId>,
    ) -> Self {
        use csr::{PackedOpenAdjacency, TemporalEdgeRow};

        let mut seen: FxHashSet<EdgeId> = FxHashSet::default();
        let n_rels = self.rel_tables_by_id.len();
        let mut per_rel: Vec<Vec<TemporalEdgeRow>> = vec![Vec::new(); n_rels];
        let mut closed: FxHashMap<EdgeId, Vec<compaction::FoldedEdgeRow>> = FxHashMap::default();

        for (rel_idx, rt) in self.rel_tables_by_id.iter().enumerate() {
            if rt.current_from_packed()
                && let Some(packed) = rt.packed_fwd()
            {
                per_rel[rel_idx].reserve(packed.num_open());
                for src in 0..packed.num_nodes() {
                    let src_u = u32::try_from(src).unwrap_or(u32::MAX);
                    for (dst, id) in packed.current_edges(src_u) {
                        seen.insert(id);
                        let hist = history_for(id);
                        let rows = compaction::fold_edge_row(id, &hist);
                        if rows.is_empty() {
                            per_rel[rel_idx].push(TemporalEdgeRow {
                                src: src_u,
                                dst,
                                validity: EpochInterval::open(EpochId::INITIAL),
                                edge_id: id,
                            });
                        }
                        for row in rows {
                            per_rel[rel_idx].push(TemporalEdgeRow {
                                src: src_u,
                                dst,
                                validity: row.validity,
                                edge_id: id,
                            });
                            if !row.properties.is_empty() || !row.raw_properties.is_empty() {
                                closed.entry(id).or_default().push(row);
                            }
                        }
                    }
                }
                continue;
            }
            let n = rt.num_edges();
            per_rel[rel_idx].reserve(n);
            for pos in 0..n {
                let id = self.original_edge_id(rel_idx, pos);
                seen.insert(id);
                let hist = history_for(id);
                let Some((src, dst)) = rt.fwd_src_dst(pos) else {
                    continue;
                };
                let rows = compaction::fold_edge_row(id, &hist);
                if rows.is_empty() {
                    per_rel[rel_idx].push(TemporalEdgeRow {
                        src,
                        dst,
                        validity: EpochInterval::open(EpochId::INITIAL),
                        edge_id: id,
                    });
                }
                for row in rows {
                    per_rel[rel_idx].push(TemporalEdgeRow {
                        src,
                        dst,
                        validity: row.validity,
                        edge_id: id,
                    });
                    if !row.properties.is_empty() || !row.raw_properties.is_empty() {
                        closed.entry(id).or_default().push(row);
                    }
                }
            }
        }

        for id in extra_ids {
            if !seen.insert(id) {
                continue;
            }
            let hist = history_for(id);
            for row in compaction::fold_edge_row(id, &hist) {
                let placement = self.place_closed_row(&row);
                if let Some((rel_idx, src, dst)) = placement {
                    per_rel[rel_idx].push(TemporalEdgeRow {
                        src,
                        dst,
                        validity: row.validity,
                        edge_id: id,
                    });
                }
                if placement.is_none()
                    || !row.properties.is_empty()
                    || !row.raw_properties.is_empty()
                {
                    closed.entry(id).or_default().push(row);
                }
            }
        }

        for (rel_idx, rt) in self.rel_tables_by_id.iter_mut().enumerate() {
            let rows = &per_rel[rel_idx];
            if rows.is_empty() {
                continue;
            }
            let num_src = rt
                .packed_fwd()
                .map_or_else(|| rt.fwd().num_nodes(), PackedOpenAdjacency::num_nodes)
                .max(rt.fwd().num_nodes())
                .max(rows.iter().map(|r| r.src as usize + 1).max().unwrap_or(0));
            let (packed_fwd, derived) = compaction::pack_and_derive_current(num_src, rows);
            let packed_bwd = if rt.properties().is_empty() {
                None
            } else {
                let bwd_rows: Vec<TemporalEdgeRow> = rows
                    .iter()
                    .map(|r| TemporalEdgeRow {
                        src: r.dst,
                        dst: r.src,
                        validity: r.validity,
                        edge_id: r.edge_id,
                    })
                    .collect();
                let max_dst = bwd_rows
                    .iter()
                    .map(|r| r.src as usize + 1)
                    .max()
                    .unwrap_or(0);
                let num_dst = rt.bwd().map_or(max_dst, |b| b.num_nodes().max(max_dst));
                Some(PackedOpenAdjacency::from_rows(num_dst, &bwd_rows))
            };
            rt.install_packed_open(packed_fwd, packed_bwd, derived);
        }

        self.closed_edges = closed;
        self.prune_packed_structure_only_closed();
        self.remap_edge_ids_to_packed();
        self
    }

    /// Maps a closed folded row onto a RelTable source/dest offset pair.
    fn place_closed_row(&self, row: &compaction::FoldedEdgeRow) -> Option<(usize, u32, u32)> {
        let (src_tid, src_off) = self.resolve_node(row.src)?;
        let (dst_tid, dst_off) = self.resolve_node(row.dst)?;
        let src_off = u32::try_from(src_off).ok()?;
        let dst_off = u32::try_from(dst_off).ok()?;
        self.rel_tables_by_id
            .iter()
            .enumerate()
            .find_map(|(i, rt)| {
                (rt.src_table_id() == src_tid
                    && rt.dst_table_id() == dst_tid
                    && rt.edge_type() == &row.edge_type)
                    .then_some((i, src_off, dst_off))
            })
    }

    /// Original `EdgeId` at a current-CSR coordinate, or the compact encoding.
    /// Slim tables retain original IDs on the relation after dropping the
    /// duplicate reverse map. Packed fat-run callers read packed IDs directly.
    fn original_edge_id(&self, rel_idx: usize, pos: usize) -> EdgeId {
        if let Some(id) = self
            .rel_tables_by_id
            .get(rel_idx)
            .and_then(|table| table.open_edge_id_at(pos))
        {
            return id;
        }
        self.edge_offset_to_id
            .as_ref()
            .and_then(|tables| tables.get(rel_idx))
            .and_then(|ids| ids.get(pos))
            .copied()
            .unwrap_or_else(|| id::encode_edge_id(u16::try_from(rel_idx).unwrap_or(0), pos as u64))
    }

    /// Neighbors visible at `epoch`.
    ///
    /// `PENDING` is the derived current CSR (open prefix). Other epochs use
    /// packed validity (or the all-open identity). Only **orphan** closed lives
    /// (not already in packed RelTables) are appended from the sidecar.
    #[must_use]
    pub fn neighbors_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
    ) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.fill_neighbors_at_epoch(node, direction, epoch, &mut out);
        out
    }

    /// Fills `out` with neighbors visible at `epoch` (clears `out` first).
    ///
    /// Whole-graph readers should call this instead of [`Self::neighbors_at_epoch`]
    /// when expanding many sources (reuse `out`).
    pub fn fill_neighbors_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<NodeId>,
    ) {
        self.fill_neighbors_at_epoch_of_types(node, direction, epoch, &[], out);
    }

    /// Dest-only as-of fill restricted to `types` (empty = every RelTable).
    ///
    /// Temporal workload hops (`OBSERVED_BY` / `CORRELATED_WITH` / `DERIVED_FROM`)
    /// share a node table; untyped fill would mix them.
    pub fn fill_neighbors_at_epoch_of_types(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        types: &[String],
        out: &mut Vec<NodeId>,
    ) {
        out.clear();
        if epoch == EpochId::PENDING && types.is_empty() {
            out.extend(self.neighbors_current(node, direction));
            return;
        }
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            out.extend(self.closed_neighbors_of_types_at(node, direction, epoch, types));
            return;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return;
        };
        let tid = node_table_id as usize;
        let mut scratch = Vec::new();

        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(rel_ids) = self.src_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if !rel_matches_types(rt, types) {
                    continue;
                }
                scratch.clear();
                rt.extend_neighbors_at_epoch(offset, epoch, &mut scratch);
                for dst_off in &scratch {
                    let compact = id::encode_node_id(rt.dst_table_id(), u64::from(*dst_off));
                    out.push(self.to_original_node_id(compact));
                }
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if !rel_matches_types(rt, types) {
                    continue;
                }
                scratch.clear();
                if rt.extend_incoming_at_epoch(offset, epoch, &mut scratch) {
                    for src_off in &scratch {
                        let compact = id::encode_node_id(rt.src_table_id(), u64::from(*src_off));
                        out.push(self.to_original_node_id(compact));
                    }
                }
            }
        }

        for nid in self.closed_neighbors_of_types_at(node, direction, epoch, types) {
            out.push(nid);
        }
        out.sort_unstable();
        out.dedup();
    }

    /// `(neighbor, edge_id)` pairs visible at `epoch`.
    ///
    /// Same packed / closed-sidecar walk as [`Self::neighbors_at_epoch`], but
    /// yields edge ids so a layered reader can apply
    /// Appends `(neighbor, edge_id)` pairs visible at `epoch` (does not clear).
    pub fn extend_edges_from_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<(NodeId, EdgeId)>,
    ) {
        if epoch == EpochId::PENDING {
            out.extend(self.edges_from_current(node, direction));
            return;
        }
        let start = out.len();
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            out.extend(self.closed_edges_at(node, direction, epoch));
            let mut tail = out.split_off(start);
            tail.sort_unstable_by_key(|&(_, eid)| eid);
            tail.dedup_by_key(|&mut (_, eid)| eid);
            out.append(&mut tail);
            return;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return;
        };
        let tid = node_table_id as usize;

        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(rel_ids) = self.src_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                let packed = rt.packed_fwd().is_some();
                let start = out.len();
                rt.extend_edges_from_at_epoch(offset, epoch, out);
                for pair in &mut out[start..] {
                    pair.0 = self.to_original_node_id(pair.0);
                    if !packed {
                        pair.1 = self.to_original_edge_id(pair.1);
                    }
                }
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                let packed = rt.packed_fwd().is_some() || rt.packed_bwd().is_some();
                if let Some(edges) = rt.incoming_edges_at_epoch(offset, epoch) {
                    for (source, eid) in edges {
                        let source = self.to_original_node_id(source);
                        let eid = if packed {
                            eid
                        } else {
                            self.to_original_edge_id(eid)
                        };
                        out.push((source, eid));
                    }
                }
            }
        }

        out.extend(self.closed_edges_at(node, direction, epoch));
        let mut tail = out.split_off(start);
        tail.sort_unstable_by_key(|&(_, eid)| eid);
        tail.dedup_by_key(|&mut (_, eid)| eid);
        out.append(&mut tail);
    }

    /// Fills `out` with `(neighbor, edge_id)` pairs visible at `epoch`.
    ///
    /// Clears `out` first. Same packed / orphan-sidecar walk as
    /// [`Self::fill_neighbors_at_epoch`].
    pub fn fill_edges_from_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<(NodeId, EdgeId)>,
    ) {
        out.clear();
        self.extend_edges_from_at_epoch(node, direction, epoch, out);
    }

    /// `(neighbor, edge_id)` pairs visible at `epoch`.
    ///
    /// Same packed / closed-sidecar walk as [`Self::neighbors_at_epoch`], but
    /// yields edge ids so a layered reader can apply
    /// `get_edge_at_epoch` / overlay tombstone visibility.
    #[must_use]
    pub fn edges_from_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
    ) -> Vec<(NodeId, EdgeId)> {
        let mut out = Vec::new();
        self.fill_edges_from_at_epoch(node, direction, epoch, &mut out);
        out
    }

    fn edges_from_current(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return Vec::new();
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return Vec::new();
        };
        self.collect_edges(node_table_id, offset, direction)
    }

    fn neighbors_current(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return Vec::new();
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return Vec::new();
        };
        self.collect_edges(node_table_id, offset, direction)
            .into_iter()
            .map(|(target, _)| target)
            .collect()
    }

    fn closed_neighbors_of_types_at(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        types: &[String],
    ) -> Vec<NodeId> {
        self.closed_edges_at(node, direction, epoch)
            .into_iter()
            .filter(|(_, eid)| {
                types.is_empty()
                    || self.retained_edge_row_at(*eid, epoch).is_some_and(|row| {
                        types
                            .iter()
                            .any(|t| row.edge_type.eq_ignore_ascii_case(t.as_str()))
                    })
            })
            .map(|(nid, _)| nid)
            .collect()
    }

    fn closed_edges_at(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
    ) -> Vec<(NodeId, EdgeId)> {
        if epoch == EpochId::PENDING {
            return Vec::new();
        }
        let mut out = Vec::new();
        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(ids) = self.closed_out.get(&node)
        {
            for id in ids {
                if let Some(row) = self.retained_edge_row_at(*id, epoch) {
                    out.push((row.dst, *id));
                }
            }
        }
        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(ids) = self.closed_in.get(&node)
        {
            for id in ids {
                if let Some(row) = self.retained_edge_row_at(*id, epoch) {
                    out.push((row.src, *id));
                }
            }
        }
        out
    }

    /// Every cold edge whose interval contains `epoch`, plus current edges
    /// when `epoch == PENDING`.
    #[must_use]
    pub fn edges_at_epoch(&self, epoch: EpochId) -> Vec<crate::graph::lpg::Edge> {
        let mut out = Vec::new();
        let mut seen: FxHashSet<EdgeId> = FxHashSet::default();
        for (rel_idx, rt) in self.rel_tables_by_id.iter().enumerate() {
            if let Some(packed) = rt.packed_fwd() {
                if epoch == EpochId::PENDING {
                    if rt.open_edge_ids_empty() {
                        for id in packed.open_edge_ids() {
                            if seen.insert(id)
                                && let Some(edge) = self.get_edge_at_epoch_inner(id, epoch)
                            {
                                out.push(edge);
                            }
                        }
                    } else {
                        for id in rt.open_edge_ids() {
                            if seen.insert(id)
                                && let Some(edge) = self.get_edge_at_epoch_inner(id, epoch)
                            {
                                out.push(edge);
                            }
                        }
                    }
                } else {
                    if !rt.open_edge_ids_empty() {
                        for (pos, id) in rt.open_edge_ids().into_iter().enumerate() {
                            if !rt.row_contains(pos, epoch) {
                                continue;
                            }
                            if seen.insert(id)
                                && let Some(edge) = self.get_edge_at_epoch_inner(id, epoch)
                            {
                                out.push(edge);
                            }
                        }
                    }
                    for pos in 0..packed.num_versions() {
                        let Some(iv) = packed.interval(pos) else {
                            continue;
                        };
                        if !iv.contains(epoch) {
                            continue;
                        }
                        let id = packed
                            .edge_ids()
                            .get(pos)
                            .copied()
                            .unwrap_or_else(|| self.original_edge_id(rel_idx, pos));
                        if seen.insert(id)
                            && let Some(edge) = self.get_edge_at_epoch_inner(id, epoch)
                        {
                            out.push(edge);
                        }
                    }
                }
            } else {
                for pos in 0..rt.num_edges() {
                    if !rt.row_contains(pos, epoch) {
                        continue;
                    }
                    let id = self.original_edge_id(rel_idx, pos);
                    if seen.insert(id)
                        && let Some(edge) = self.get_edge_at_epoch_inner(id, epoch)
                    {
                        out.push(edge);
                    }
                }
            }
        }
        for (id, rows) in &self.closed_edges {
            if seen.contains(id) {
                continue;
            }
            if let Some(row) = rows
                .iter()
                .find(|row| epoch != EpochId::PENDING && row.validity.contains(epoch))
            {
                out.push(self.edge_from_closed(row, epoch));
            }
        }
        out
    }

    fn get_edge_at_epoch_inner(
        &self,
        id: EdgeId,
        epoch: EpochId,
    ) -> Option<crate::graph::lpg::Edge> {
        // Local as-of so `edges_at_epoch` does not depend on the GraphStore impl.
        if epoch == EpochId::PENDING {
            return self.edge_current(id);
        }
        if let Some(row) = self.retained_edge_row_at(id, epoch) {
            return Some(self.edge_from_closed(row, epoch));
        }
        if let Some(edge) = self.edge_from_packed_closed(id, epoch) {
            return Some(edge);
        }
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = self.rel_edge_pos(rt, id, csr_position)?;
        let pos_us = pos as usize;
        if !rt.row_contains(pos_us, epoch) {
            return None;
        }
        let src_compact = rt.source_node_id(pos)?;
        let dst_compact = rt.dest_node_id(pos)?;
        let src = self.to_original_node_id(src_compact);
        let dst = self.to_original_node_id(dst_compact);
        let mut edge = crate::graph::lpg::Edge::new(id, src, dst, rt.edge_type().clone());
        for key in rt.property_keys() {
            if let Some(value) = rt.get_property_at_epoch(pos_us, &key, epoch) {
                edge.set_property(key, value);
            }
        }
        Some(edge)
    }

    fn edge_current(&self, id: EdgeId) -> Option<crate::graph::lpg::Edge> {
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = self.rel_edge_pos(rt, id, csr_position)?;
        let src_compact = rt.source_node_id(pos)?;
        let dst_compact = rt.dest_node_id(pos)?;
        let src = self.to_original_node_id(src_compact);
        let dst = self.to_original_node_id(dst_compact);
        let mut edge = crate::graph::lpg::Edge::new(id, src, dst, rt.edge_type().clone());
        for (k, v) in rt.get_all_edge_properties(pos as usize) {
            edge.set_property(k, v);
        }
        Some(edge)
    }

    fn rel_edge_pos(&self, rt: &RelTable, id: EdgeId, hint: u64) -> Option<u32> {
        if rt.current_from_packed() {
            u32::try_from(rt.fat_pos_for_edge(id, hint)?).ok()
        } else {
            u32::try_from(hint).ok()
        }
    }

    /// Live CSR + retained closed rows as `(id, interval)` pairs.
    #[must_use]
    pub fn structural_edge_rows(&self) -> Vec<(EdgeId, EpochInterval)> {
        let mut rows = Vec::new();
        for (rel_idx, rt) in self.rel_tables_by_id.iter().enumerate() {
            if rt.current_from_packed()
                && let Some(packed) = rt.packed_fwd()
            {
                for src in 0..packed.num_nodes() {
                    let start = packed.offsets()[src] as usize;
                    let end = packed.open_ends()[src] as usize;
                    for pos in start..end {
                        rows.push((
                            packed.edge_ids()[pos],
                            packed
                                .interval(pos)
                                .unwrap_or_else(|| EpochInterval::open(EpochId::INITIAL)),
                        ));
                    }
                }
                continue;
            }
            for pos in 0..rt.num_edges() {
                rows.push((self.original_edge_id(rel_idx, pos), rt.interval_at(pos)));
            }
        }
        // Current CSR contains the packed open prefix (or fresh all-open
        // rows); closed_id_ord contains only closed intervals. Without a
        // sidecar these representations are disjoint, so no count maps are
        // needed even when closed zero-width lifetimes repeat.
        if self.closed_edges.is_empty() {
            for rt in &self.rel_tables_by_id {
                let Some(packed) = rt.packed_fwd() else {
                    continue;
                };
                for pos in rt.closed_id_ord() {
                    if let Some(interval) = packed.interval(pos as usize) {
                        rows.push((packed.edge_ids()[pos as usize], interval));
                    }
                }
            }
            rows.sort_unstable_by_key(|(id, iv)| {
                (id.as_u64(), iv.from().as_u64(), iv.to().as_u64())
            });
            return rows;
        }

        // Each representation can retain the same structural lifetime. Merge
        // occurrence counts across current CSR, sidecar, and packed closed rows;
        // repeated zero-width lives within one representation remain distinct.
        let mut represented = FxHashMap::<(EdgeId, EpochId, EpochId), usize>::default();
        for (id, interval) in &rows {
            *represented
                .entry((*id, interval.from(), interval.to()))
                .or_default() += 1;
        }
        let mut occurrences = FxHashMap::<(EdgeId, EpochId, EpochId), usize>::default();
        for (id, retained) in &self.closed_edges {
            for row in retained {
                let key = (*id, row.validity.from(), row.validity.to());
                let seen = occurrences.entry(key).or_default();
                *seen += 1;
                let retained_count = represented.entry(key).or_default();
                if *seen > *retained_count {
                    rows.push((*id, row.validity));
                    *retained_count = *seen;
                }
            }
        }
        occurrences.clear();
        for rt in &self.rel_tables_by_id {
            let Some(packed) = rt.packed_fwd() else {
                continue;
            };
            for pos in rt.closed_id_ord() {
                let id = packed.edge_ids()[pos as usize];
                if let Some(iv) = packed.interval(pos as usize) {
                    let key = (id, iv.from(), iv.to());
                    let seen = occurrences.entry(key).or_default();
                    *seen += 1;
                    let retained_count = represented.entry(key).or_default();
                    if *seen > *retained_count {
                        rows.push((id, iv));
                        *retained_count = *seen;
                    }
                }
            }
        }
        rows.sort_unstable_by_key(|(id, iv)| (id.as_u64(), iv.from().as_u64(), iv.to().as_u64()));
        rows
    }

    /// Structural interval of `id` (live CSR, sidecar, or packed-placed closed).
    #[must_use]
    pub fn edge_validity(&self, id: EdgeId) -> Option<EpochInterval> {
        if let Some(rows) = self.closed_edges.get(&id)
            && let Some(row) = rows
                .iter()
                .find(|row| row.validity.is_open())
                .or_else(|| rows.last())
        {
            return Some(row.validity);
        }
        if let Some(iv) = self.packed_closed_validity(id) {
            return Some(iv);
        }
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = if rt.current_from_packed() {
            rt.fat_pos_for_edge(id, csr_position)?
        } else {
            usize::try_from(csr_position).ok()?
        };
        if rt.current_from_packed() {
            return Some(rt.interval_at(pos));
        }
        (pos < rt.num_edges()).then(|| rt.interval_at(pos))
    }

    /// Ids of retained closed lives (sidecar + structure-only packed-placed).
    #[must_use]
    pub fn closed_edge_ids(&self) -> Vec<EdgeId> {
        let mut ids: Vec<EdgeId> = self.closed_edges.keys().copied().collect();
        for rt in &self.rel_tables_by_id {
            let Some(packed) = rt.packed_fwd() else {
                continue;
            };
            for pos in rt.closed_id_ord() {
                let id = packed.edge_ids()[pos as usize];
                if !self.closed_edges.contains_key(&id) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Number of structurally OPEN identities retained outside the derived
    /// current CSR.
    ///
    /// A portable transport shard may intentionally carry an edge whose
    /// destination node is absent. The temporal sidecar must retain that exact
    /// identity even though the local CSR cannot encode its endpoint row. Such
    /// edges are logically live and must participate in layered counts without
    /// being double-counted when a normal current-CSR row also owns history.
    #[cfg(feature = "lpg")]
    pub(crate) fn retained_open_edge_count(&self) -> usize {
        self.closed_edges
            .iter()
            .filter(|(id, rows)| {
                self.resolve_edge(**id).is_none() && rows.iter().any(|row| row.validity.is_open())
            })
            .count()
    }

    /// Whether `id` contributes one logically live identity to the compact
    /// generation, including manifest-qualified OPEN sidecar rows.
    #[cfg(feature = "lpg")]
    pub(crate) fn has_logically_open_edge(&self, id: EdgeId) -> bool {
        self.retained_edge_row_at(id, EpochId::PENDING).is_some() || self.edge_current(id).is_some()
    }

    /// Sidecar-only count (property history / orphans). Packed-placed
    /// structure-only closed lives are not stored here.
    #[cfg(all(test, feature = "lpg"))]
    pub(crate) fn closed_sidecar_len(&self) -> usize {
        self.closed_edges.values().map(Vec::len).sum()
    }

    /// Original ids of live CSR edges.
    #[must_use]
    pub fn live_original_edge_ids(&self) -> Vec<EdgeId> {
        let mut ids = Vec::new();
        for (rel_idx, rt) in self.rel_tables_by_id.iter().enumerate() {
            if rt.current_from_packed()
                && let Some(packed) = rt.packed_fwd()
            {
                ids.extend(packed.open_edge_ids());
                continue;
            }
            if !rt.open_edge_ids_empty() {
                ids.extend(rt.open_edge_ids());
                continue;
            }
            if let Some(rev) = self
                .edge_offset_to_id
                .as_ref()
                .and_then(|tables| tables.get(rel_idx))
            {
                ids.extend(rev.iter().copied().filter(|id| *id != EdgeId::INVALID));
            } else {
                for pos in 0..rt.num_edges() {
                    ids.push(self.original_edge_id(rel_idx, pos));
                }
            }
        }
        ids
    }

    pub(crate) fn packed_closed_edge_type(&self, id: EdgeId) -> Option<arcstr::ArcStr> {
        let (rel_idx, _) = self.packed_closed_get(id)?;
        self.rel_tables_by_id
            .get(rel_idx as usize)
            .map(|rt| rt.edge_type().clone())
    }

    pub(crate) fn packed_closed_validity(&self, id: EdgeId) -> Option<EpochInterval> {
        let (rel_idx, pos) = self.packed_closed_get(id)?;
        self.rel_tables_by_id
            .get(rel_idx as usize)
            .and_then(|rt| rt.packed_fwd())
            .and_then(|p| p.interval(pos as usize))
    }

    /// Structural identity for retained history, independent of visibility.
    /// A closed [C,C) lifetime has no epoch at which an as-of read can recover
    /// its endpoints/type. This must not be used to admit a live graph read.
    #[cfg(feature = "lpg")]
    fn retained_edge_identity(&self, id: EdgeId) -> Option<crate::graph::lpg::Edge> {
        if let Some(row) = self.closed_edge_row(id) {
            return Some(crate::graph::lpg::Edge::new(
                id,
                row.src,
                row.dst,
                row.edge_type.clone(),
            ));
        }
        let (rel_idx, pos) = self.packed_closed_get(id)?;
        let table = self.rel_tables_by_id.get(usize::from(rel_idx))?;
        let packed = table.packed_fwd()?;
        let pos = usize::try_from(pos).ok()?;
        let src = self.to_original_node_id(id::encode_node_id(
            table.src_table_id(),
            u64::from(packed.src_of(pos)?),
        ));
        let dst = self.to_original_node_id(id::encode_node_id(
            table.dst_table_id(),
            u64::from(*packed.targets().get(pos)?),
        ));
        Some(crate::graph::lpg::Edge::new(
            id,
            src,
            dst,
            table.edge_type().clone(),
        ))
    }

    /// Topology-only as-of reconstruct for a structure-only packed-placed close.
    pub(crate) fn edge_from_packed_closed(
        &self,
        id: EdgeId,
        epoch: EpochId,
    ) -> Option<crate::graph::lpg::Edge> {
        if epoch == EpochId::PENDING {
            return None;
        }

        // One EdgeId may have several closed structural lives. A singular
        // binary-search result is therefore insufficient, and an unrelated
        // property-bearing sidecar life must not mask a structure-only packed
        // life. Walk the small per-table closed-id index and select by epoch.
        for rt in &self.rel_tables_by_id {
            let Some(packed) = rt.packed_fwd() else {
                continue;
            };
            for pos in rt.closed_id_ord() {
                let fat_pos = pos as usize;
                if packed.edge_ids().get(fat_pos) != Some(&id) {
                    continue;
                }
                let Some(iv) = packed.interval(fat_pos) else {
                    continue;
                };
                if !iv.contains(epoch) {
                    continue;
                }
                let src_off = packed.src_of(fat_pos)?;
                let dst_off = *packed.targets().get(fat_pos)?;
                let src = self
                    .to_original_node_id(id::encode_node_id(rt.src_table_id(), u64::from(src_off)));
                let dst = self
                    .to_original_node_id(id::encode_node_id(rt.dst_table_id(), u64::from(dst_off)));
                return Some(crate::graph::lpg::Edge::new(
                    id,
                    src,
                    dst,
                    rt.edge_type().clone(),
                ));
            }
        }
        None
    }

    /// Reconstructs a closed row as an `Edge` with properties valid at `epoch`.
    fn edge_from_closed(
        &self,
        row: &compaction::FoldedEdgeRow,
        epoch: EpochId,
    ) -> crate::graph::lpg::Edge {
        let mut edge =
            crate::graph::lpg::Edge::new(row.id, row.src, row.dst, row.edge_type.clone());
        for (key, col) in &row.properties {
            if let Some(value) = col.value_in_range_as_of(0, col.len(), epoch) {
                edge.set_property(key.clone(), value);
            }
        }
        for (key, col) in &row.raw_properties {
            if let Some(value) = col.value_as_of(epoch) {
                edge.set_property(key.clone(), value);
            }
        }
        edge
    }

    /// As-of read of a single edge property at `epoch` from the cold base.
    #[must_use]
    pub fn get_edge_property_at_epoch(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        if let Some(row) = self.retained_edge_row_at(id, epoch) {
            return row
                .properties
                .get(key)
                .and_then(|col| col.value_in_range_as_of(0, col.len(), epoch))
                .or_else(|| {
                    row.raw_properties
                        .get(key)
                        .and_then(|col| col.value_as_of(epoch))
                });
        }
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = if rt.current_from_packed() {
            rt.fat_pos_for_edge(id, csr_position)?
        } else {
            usize::try_from(csr_position).ok()?
        };
        rt.get_property_at_epoch(pos, key, epoch)
    }

    /// Per-property history reconstructed from a closed folded row, or empty.
    #[cfg(any(test, feature = "lpg"))]
    fn closed_edge_property_history(
        &self,
        id: EdgeId,
    ) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let Some(rows) = self.closed_edges.get(&id) else {
            return Vec::new();
        };
        let mut histories: FxHashMap<PropertyKey, Vec<(EpochId, Value)>> = FxHashMap::default();
        let mut previous_to = None;
        for row in rows {
            for (key, column) in &row.properties {
                append_folded_row_history(
                    histories.entry(key.clone()).or_default(),
                    column.runs_as_history(0, column.len()),
                    previous_to,
                    row.validity.from(),
                );
            }
            for (key, column) in &row.raw_properties {
                append_folded_row_history(
                    histories.entry(key.clone()).or_default(),
                    column.runs_as_history(),
                    previous_to,
                    row.validity.from(),
                );
            }
            previous_to = (!row.validity.is_open()).then(|| row.validity.to());
        }
        histories.into_iter().collect()
    }

    fn retained_edge_row_at(
        &self,
        id: EdgeId,
        epoch: EpochId,
    ) -> Option<&compaction::FoldedEdgeRow> {
        self.closed_edges.get(&id)?.iter().find(|row| {
            if epoch == EpochId::PENDING {
                row.validity.is_open()
            } else {
                row.validity.contains(epoch)
            }
        })
    }

    /// Current retained row, or the latest closed row for legacy callers.
    pub(crate) fn closed_edge_row(&self, id: EdgeId) -> Option<&compaction::FoldedEdgeRow> {
        let rows = self.closed_edges.get(&id)?;
        rows.iter()
            .find(|row| row.validity.is_open())
            .or_else(|| rows.last())
    }

    /// Whole-state as-of scrub over the temporal cold base, **columnar**: for
    /// each node table, the value at `epoch` for every node and property,
    /// without materializing a `Node` per node. Each property's values are
    /// zone-pruned validity lookups over its temporal column, collected in node
    /// (offset) order. The allocation-light scrub path for 60fps time-travel —
    /// see [`NodeTableScrub`]. (`get_node_at_epoch`/`nodes_at_epoch` remain the
    /// `Node`-materializing convenience APIs.)
    #[must_use]
    pub fn scrub_at_epoch(&self, epoch: EpochId) -> Vec<NodeTableScrub> {
        if epoch != EpochId::PENDING && !self.temporal_nodes.is_empty() {
            let nodes = self.temporal_node_ids().into_iter().filter_map(|id| {
                let row = self.temporal_node_row_at(id, epoch)?;
                Some(self.node_from_temporal(row, epoch))
            });
            return node_frames_from_nodes(nodes);
        }
        self.node_tables_by_id
            .iter()
            .map(|nt| {
                let n = nt.len();
                let table_id = nt.table_id();
                let node_ids: Vec<NodeId> = (0..n)
                    .map(|off| self.to_original_node_id(id::encode_node_id(table_id, off as u64)))
                    .collect();
                let mut columns: FxHashMap<PropertyKey, Vec<Option<Value>>> = FxHashMap::default();
                for key in nt.property_keys() {
                    let values: Vec<Option<Value>> = (0..n)
                        .map(|off| nt.get_property_at_epoch(off, &key, epoch))
                        .collect();
                    columns.insert(key, values);
                }
                NodeTableScrub {
                    label: ArcStr::from(nt.label()),
                    node_ids,
                    columns,
                }
            })
            .collect()
    }

    /// Per-type edge frames at `epoch` (see [`RelTableScrub`]).
    ///
    /// Walks current CSR (and packed closed tails) into columnar frames.
    /// Does not materialize [`crate::graph::lpg::Edge`] (HashMap properties).
    /// `PENDING` is the derived current CSR. Closed lives appear only at
    /// epochs inside their interval.
    #[must_use]
    pub fn edge_scrub_at_epoch(&self, epoch: EpochId) -> Vec<RelTableScrub> {
        let mut frames = Vec::with_capacity(self.rel_tables_by_id.len());
        let mut seen: FxHashSet<EdgeId> = FxHashSet::default();

        for (rel_idx, rt) in self.rel_tables_by_id.iter().enumerate() {
            let keys = rt.property_keys();
            let packed_n = rt
                .packed_fwd()
                .map_or(0, csr::PackedOpenAdjacency::num_versions);
            let cap = rt.fwd().num_edges() + packed_n;
            let mut frame = RelTableScrub::with_keys(rt.edge_type().clone(), &keys, cap);

            for src in 0..rt.fwd().num_nodes() {
                let src_u = u32::try_from(src).unwrap_or(u32::MAX);
                let dests = rt.fwd().neighbors(src_u);
                let start = rt.fwd().offset_of(src_u) as usize;
                for (i, &dst) in dests.iter().enumerate() {
                    let pos = start + i;
                    if epoch != EpochId::PENDING && !rt.row_contains(pos, epoch) {
                        continue;
                    }
                    let id = rt
                        .open_edge_id_at(pos)
                        .unwrap_or_else(|| self.original_edge_id(rel_idx, pos));
                    if !seen.insert(id) {
                        continue;
                    }
                    let src_id =
                        self.to_original_node_id(id::encode_node_id(rt.src_table_id(), src as u64));
                    let dst_id = self
                        .to_original_node_id(id::encode_node_id(rt.dst_table_id(), u64::from(dst)));
                    frame.push_row(id, src_id, dst_id, |key| {
                        rt.get_property_at_epoch(pos, key, epoch)
                    });
                }
            }

            if epoch != EpochId::PENDING
                && let Some(packed) = rt.packed_fwd()
            {
                for pos in 0..packed.num_versions() {
                    let Some(iv) = packed.interval(pos) else {
                        continue;
                    };
                    if !iv.contains(epoch) {
                        continue;
                    }
                    let id = packed
                        .edge_ids()
                        .get(pos)
                        .copied()
                        .unwrap_or_else(|| self.original_edge_id(rel_idx, pos));
                    if !seen.insert(id) {
                        continue;
                    }
                    let Some(src) = packed.src_of(pos) else {
                        continue;
                    };
                    let dst = packed.targets()[pos];
                    let src_id = self
                        .to_original_node_id(id::encode_node_id(rt.src_table_id(), u64::from(src)));
                    let dst_id = self
                        .to_original_node_id(id::encode_node_id(rt.dst_table_id(), u64::from(dst)));
                    let retained = self.retained_edge_row_at(id, epoch);
                    if let Some(row) = retained {
                        for key in row.properties.keys().chain(row.raw_properties.keys()) {
                            frame
                                .columns
                                .entry(key.clone())
                                .or_insert_with(|| vec![None; frame.edge_ids.len()]);
                        }
                    }
                    frame.push_row(id, src_id, dst_id, |key| {
                        retained.and_then(|row| {
                            row.properties
                                .get(key)
                                .and_then(|col| col.value_as_of(0, epoch))
                                .or_else(|| {
                                    row.raw_properties
                                        .get(key)
                                        .and_then(|col| col.value_as_of(epoch))
                                })
                        })
                    });
                }
            }

            frames.push(frame);
        }

        for (id, rows) in &self.closed_edges {
            if !seen.insert(*id) {
                continue;
            }
            let Some(row) = rows
                .iter()
                .find(|row| epoch != EpochId::PENDING && row.validity.contains(epoch))
            else {
                continue;
            };
            let slot = frames.iter().position(|f| f.edge_type == row.edge_type);
            let slot = if let Some(i) = slot {
                i
            } else {
                let keys: Vec<PropertyKey> = row
                    .properties
                    .keys()
                    .chain(row.raw_properties.keys())
                    .cloned()
                    .collect();
                frames.push(RelTableScrub::with_keys(row.edge_type.clone(), &keys, 1));
                frames.len() - 1
            };
            let frame = &mut frames[slot];
            for key in row.properties.keys().chain(row.raw_properties.keys()) {
                frame
                    .columns
                    .entry(key.clone())
                    .or_insert_with(|| vec![None; frame.edge_ids.len()]);
            }
            frame.push_row(*id, row.src, row.dst, |key| {
                row.properties
                    .get(key)
                    .and_then(|col| col.value_as_of(0, epoch))
                    .or_else(|| {
                        row.raw_properties
                            .get(key)
                            .and_then(|col| col.value_as_of(epoch))
                    })
            });
        }

        frames
    }

    /// Visits every source that has neighbors at `epoch` (no per-id HashMap).
    ///
    /// Whole-graph expansion: walk table offsets, reuse `buf`.
    pub fn visit_neighbors_at_epoch(
        &self,
        direction: Direction,
        epoch: EpochId,
        mut visit: impl FnMut(NodeId, &[NodeId]),
    ) {
        let mut buf = Vec::new();
        for (tid_usz, nt) in self.node_tables_by_id.iter().enumerate() {
            let tid = u16::try_from(tid_usz).unwrap_or(u16::MAX);
            for off in 0..nt.len() {
                let off_u = u32::try_from(off).unwrap_or(u32::MAX);
                let nid = if self.preserves_ids() {
                    self.to_original_node_id(id::encode_node_id(tid, off as u64))
                } else {
                    id::encode_node_id(tid, off as u64)
                };
                if epoch == EpochId::PENDING {
                    buf.clear();
                    buf.extend(self.collect_neighbors(tid, off_u, direction));
                } else {
                    self.fill_neighbors_at_epoch(nid, direction, epoch, &mut buf);
                }
                if !buf.is_empty() {
                    visit(nid, &buf);
                }
            }
        }
    }

    /// Node + edge as-of scrub of the cold base.
    #[must_use]
    pub fn graph_scrub_at_epoch(&self, epoch: EpochId) -> GraphScrub {
        GraphScrub {
            nodes: self.scrub_at_epoch(epoch),
            edges: self.edge_scrub_at_epoch(epoch),
        }
    }
}

#[cfg(all(test, feature = "lpg"))]
mod structural_occurrence_tests {
    use super::*;

    #[test]
    fn structural_rows_merge_current_sidecar_and_packed_occurrences() {
        let store = crate::graph::lpg::LpgStore::new().unwrap();
        let src = store.create_node(&["Source"]);
        let dst = store.create_node(&["Destination"]);
        let id = store.create_edge(src, dst, "R");
        let epoch = EpochId::new(1);
        let zero = compaction::EdgeLifetime::new(epoch, Some(epoch));
        let history = compaction::EdgeFullHistory {
            src,
            dst,
            edge_type: "R".into(),
            lifetimes: vec![zero, zero, compaction::EdgeLifetime::new(epoch, None)],
            properties: FxHashMap::default(),
        };
        let mut compact = from_graph_store_preserving_ids(&store)
            .unwrap()
            .upgrade_rels_temporal(|_| history.clone(), [id]);
        let expected = vec![
            (id, EpochInterval::closed(epoch, epoch)),
            (id, EpochInterval::closed(epoch, epoch)),
            (id, EpochInterval::open(epoch)),
        ];
        assert_eq!(compact.structural_edge_rows(), expected);
        // Recovery may retain the same open CSR and closed packed lives in
        // the property sidecar. These are representation copies, not new lives.
        compact.closed_edges.insert(
            id,
            expected
                .iter()
                .map(|(_, validity)| compaction::FoldedEdgeRow {
                    id,
                    src,
                    dst,
                    edge_type: "R".into(),
                    validity: *validity,
                    properties: FxHashMap::default(),
                    raw_properties: FxHashMap::default(),
                })
                .collect(),
        );
        assert_eq!(compact.structural_edge_rows(), expected);
        compact.closed_edges.get_mut(&id).unwrap().remove(0);
        assert_eq!(
            compact.structural_edge_rows(),
            expected,
            "packed multiplicity exceeds the sidecar's partial overlap"
        );
    }
}
