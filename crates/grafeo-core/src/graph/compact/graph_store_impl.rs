//! [`GraphStore`](crate::graph::GraphStore) trait implementation for [`CompactStore`].
//!
//! All read operations (point lookups, traversal, scans, property access,
//! filtered search, statistics, and visibility checks) are implemented here.
//! The store is read-only: all data comes from immutable columnar tables.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::CompactStore;
use super::id::encode_node_id;
use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreSearch};
use crate::statistics::Statistics;

impl CompactStore {
    /// As-of read of a single node property at `epoch` from the temporal cold
    /// base (SP1-5 slice 2). For the all-open base this equals
    /// [`get_node_property`](GraphStore::get_node_property) at every real epoch.
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        if let Some(row) = self.temporal_node_row_at(id, epoch) {
            return row
                .properties
                .get(key)
                .and_then(|column| column.value_in_range_as_of(0, column.len(), epoch))
                .or_else(|| {
                    row.raw_properties
                        .get(key)
                        .and_then(|column| column.value_as_of(epoch))
                });
        }
        // A v6 sidecar entry with no row at this epoch is authoritative absence;
        // do not fall through to the current projection and resurrect the node.
        if self.temporal_nodes().contains_key(&id) {
            return None;
        }
        let (table_id, offset) = self.resolve_node(id)?;
        let nt = self.resolve_node_table(table_id)?;
        let row = usize::try_from(offset).ok()?;
        nt.get_property_at_epoch(row, key, epoch)
    }
}

impl GraphStore for CompactStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        if self.temporal_nodes().contains_key(&id) {
            return self
                .temporal_node_row_at(id, EpochId::PENDING)
                .map(|row| self.node_from_temporal(row, EpochId::PENDING));
        }
        let (table_id, offset) = self.resolve_node(id)?;
        let nt = self.resolve_node_table(table_id)?;
        let row = usize::try_from(offset).ok()?;
        if row >= nt.len() {
            return None;
        }

        let mut node = Node::new(id);
        node.add_label(nt.label());
        let props = nt.get_all_properties(row);
        for (k, v) in props {
            node.set_property(k, v);
        }
        Some(node)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if let Some(row) = self.retained_edge_row_at(id, EpochId::PENDING) {
            return Some(self.edge_from_closed(row, EpochId::PENDING));
        }
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = self.rel_edge_pos(rt, id, csr_position)?;

        let src_compact = rt.source_node_id(pos)?;
        let dst_compact = rt.dest_node_id(pos)?;
        let src = self.to_original_node_id(src_compact);
        let dst = self.to_original_node_id(dst_compact);
        let edge_type = rt.edge_type().clone();

        let mut edge = Edge::new(id, src, dst, edge_type);
        let props = rt.get_all_edge_properties(pos as usize);
        for (k, v) in props {
            edge.set_property(k, v);
        }
        Some(edge)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Node> {
        self.get_node_at_epoch(id, epoch)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.get_edge_at_epoch(id, epoch)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        // `PENDING` (u64::MAX) is the open-interval "latest/current" sentinel —
        // no half-open `[from, PENDING)` validity contains it — so an as-of read
        // at PENDING means the current view.
        if epoch == EpochId::PENDING {
            return self.get_node(id);
        }
        if let Some(row) = self.temporal_node_row_at(id, epoch) {
            return Some(self.node_from_temporal(row, epoch));
        }
        if self.temporal_nodes().contains_key(&id) {
            return None;
        }
        let (table_id, offset) = self.resolve_node(id)?;
        let nt = self.resolve_node_table(table_id)?;
        let row = usize::try_from(offset).ok()?;
        if row >= nt.len() {
            return None;
        }
        let props = nt.get_all_properties_at_epoch(row, epoch);
        // A node that holds properties now but none at `epoch` did not yet exist
        // (or was fully removed) at `epoch`, so it is absent from an as-of scrub.
        // For an all-open base, as-of props == current props, so this never fires
        // — every node exists for all of time. A genuinely property-less node
        // (label only) is still returned.
        if props.is_empty() && !nt.get_all_properties(row).is_empty() {
            return None;
        }
        let mut node = Node::new(id);
        node.add_label(nt.label());
        for (k, v) in props {
            node.set_property(k, v);
        }
        Some(node)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        if epoch == EpochId::PENDING {
            return self.get_edge(id);
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
        let edge_type = rt.edge_type().clone();

        let mut edge = Edge::new(id, src, dst, edge_type);
        for key in rt.property_keys() {
            if let Some(value) = rt.get_property_at_epoch(pos_us, &key, epoch) {
                edge.set_property(key, value);
            }
        }
        Some(edge)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if self.temporal_nodes().contains_key(&id) {
            return self.get_node_property_at_epoch(id, key, EpochId::PENDING);
        }
        let (table_id, offset) = self.resolve_node(id)?;
        let nt = self.resolve_node_table(table_id)?;
        let row = usize::try_from(offset).ok()?;
        nt.get_property(row, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if self.retained_edge_row_at(id, EpochId::PENDING).is_some() {
            return self.get_edge_property_at_epoch(id, key, EpochId::PENDING);
        }
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let row = if rt.current_from_packed() {
            rt.fat_pos_for_edge(id, csr_position)?
        } else {
            usize::try_from(csr_position).ok()?
        };
        rt.get_edge_property(row, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|id| self.get_node_property(*id, key))
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                self.get_node(*id)
                    .map(|n| {
                        n.properties
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_node_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_edge_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
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

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return Vec::new();
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return Vec::new();
        };
        self.collect_edges(node_table_id, offset, direction)
    }

    fn edges_from_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
    ) -> Vec<(NodeId, EdgeId)> {
        CompactStore::edges_from_at_epoch(self, node, direction, epoch)
    }

    fn fill_edges_from(&self, node: NodeId, direction: Direction, out: &mut Vec<(NodeId, EdgeId)>) {
        out.extend(self.edges_from(node, direction));
    }

    fn fill_neighbors(&self, node: NodeId, direction: Direction, out: &mut Vec<NodeId>) {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return;
        };
        out.extend(self.collect_neighbors(node_table_id, offset, direction));
    }

    fn snapshot_neighbors(&self, direction: Direction) -> Vec<(NodeId, Vec<NodeId>)> {
        self.snapshot_csr_neighbors(direction)
    }

    fn try_count_directed_triangles(
        &self,
        starts: &[NodeId],
        dest_label: Option<&str>,
    ) -> Option<u64> {
        self.count_csr_triangles(Some(starts), dest_label)
    }

    fn try_count_all_directed_triangles(&self, dest_label: Option<&str>) -> Option<u64> {
        self.count_csr_triangles(None, dest_label)
    }

    fn fill_neighbors_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<NodeId>,
    ) {
        CompactStore::fill_neighbors_at_epoch(self, node, direction, epoch, out);
    }

    fn fill_neighbors_of_types_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        types: &[String],
        out: &mut Vec<NodeId>,
    ) {
        CompactStore::fill_neighbors_at_epoch_of_types(self, node, direction, epoch, types, out);
    }

    fn fill_edges_from_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<(NodeId, EdgeId)>,
    ) {
        CompactStore::extend_edges_from_at_epoch(self, node, direction, epoch, out);
    }

    fn has_property_index(&self, property: &str) -> bool {
        let key = PropertyKey::new(property);
        self.node_tables_by_id
            .iter()
            .any(|nt| nt.column(&key).is_some())
    }

    fn all_edges_have_types(&self, types: &[String]) -> bool {
        if types.is_empty() {
            return true;
        }
        self.rel_tables_by_id.iter().all(|rt| {
            types
                .iter()
                .any(|t| rt.edge_type().eq_ignore_ascii_case(t.as_str()))
        })
    }

    fn count_edges_from(&self, node: NodeId, direction: Direction, types: &[String]) -> usize {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return 0;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return 0;
        };
        let tid = node_table_id as usize;
        let type_ok = |rt: &crate::graph::compact::rel_table::RelTable| {
            types.is_empty()
                || types
                    .iter()
                    .any(|t| rt.edge_type().eq_ignore_ascii_case(t.as_str()))
        };
        let mut n = 0;
        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(rel_ids) = self.src_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if type_ok(rt) {
                    n += rt.out_degree(offset);
                }
            }
        }
        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if type_ok(rt)
                    && let Some(d) = rt.in_degree(offset)
                {
                    n += d;
                }
            }
        }
        n
    }

    fn out_degree(&self, node: NodeId) -> usize {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return 0;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return 0;
        };
        let mut degree = 0;
        if let Some(rel_ids) = self.src_rel_table_ids.get(node_table_id as usize) {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                degree += rt.out_degree(offset);
            }
        }
        degree
    }

    fn in_degree(&self, node: NodeId) -> usize {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return 0;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return 0;
        };
        let mut degree = 0;
        if let Some(rel_ids) = self.dst_rel_table_ids.get(node_table_id as usize) {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if let Some(d) = rt.in_degree(offset) {
                    degree += d;
                }
            }
        }
        degree
    }

    fn has_backward_adjacency(&self) -> bool {
        self.rel_tables_by_id.iter().any(|rt| rt.has_backward())
    }

    fn node_ids(&self) -> Vec<NodeId> {
        if let Some(ref map) = self.node_id_map {
            let mut ids: Vec<NodeId> = map.keys().copied().collect();
            ids.sort_unstable();
            ids
        } else {
            let mut ids = Vec::new();
            for nt in &self.node_tables_by_id {
                ids.extend(nt.node_ids());
            }
            ids.sort_unstable();
            ids
        }
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let mut ids = self
            .temporal_label_index
            .get(label)
            .cloned()
            .unwrap_or_default();
        if let Some(&table_id) = self.label_to_table_id.get(label) {
            for compact_id in self.node_tables_by_id[table_id as usize].node_ids() {
                let id = if self.preserves_ids() {
                    self.to_original_node_id(compact_id)
                } else {
                    compact_id
                };
                if !self.temporal_nodes().contains_key(&id) {
                    ids.push(id);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Membership without walking a table's ids: a temporal node answers from the
    /// temporal label index, and a base node carries exactly its table's label —
    /// the same two sources `nodes_by_label` unions.
    fn node_has_label(&self, id: NodeId, label: &str) -> bool {
        if self.temporal_nodes().contains_key(&id) {
            return self
                .temporal_label_index
                .get(label)
                .is_some_and(|ids| ids.contains(&id));
        }
        let Some(&table_id) = self.label_to_table_id.get(label) else {
            return false;
        };
        self.resolve_node(id)
            .is_some_and(|(resolved, _)| resolved == table_id)
    }

    fn node_has_label_visible(
        &self,
        id: NodeId,
        label: &str,
        _transaction_id: Option<TransactionId>,
    ) -> bool {
        self.node_has_label(id, label)
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        self.nodes_by_label(label).len()
    }

    fn node_count(&self) -> usize {
        self.node_tables_by_id.iter().map(|nt| nt.len()).sum()
    }

    fn edge_count(&self) -> usize {
        self.rel_tables_by_id.iter().map(|rt| rt.num_edges()).sum()
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        if let Some(row) = self.closed_edge_row(id) {
            return Some(row.edge_type.clone());
        }
        if let Some(ty) = self.packed_closed_edge_type(id) {
            return Some(ty);
        }
        let (rel_table_id, _) = self.resolve_edge(id)?;
        self.rel_table_id_to_type
            .get(rel_table_id as usize)
            .cloned()
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let mut results = Vec::new();
        for nt in &self.node_tables_by_id {
            if let Some(zm) = nt.zone_map(&key)
                && !zm.might_match(CompareOp::Eq, value)
            {
                continue;
            }
            if let Some(col) = nt.column(&key) {
                let table_id = nt.table_id();
                for offset in nt.current_matching_offsets(&key, col.find_eq(value)) {
                    let compact_id = encode_node_id(table_id, offset as u64);
                    results.push(self.to_original_node_id(compact_id));
                }
            }
        }
        results
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }

        let (first_prop, first_val) = &conditions[0];
        let candidates = self.find_nodes_by_property(first_prop, first_val);

        if conditions.len() == 1 {
            return candidates;
        }

        candidates
            .into_iter()
            .filter(|nid| {
                for (prop, val) in &conditions[1..] {
                    let key = PropertyKey::new(*prop);
                    match self.get_node_property(*nid, &key) {
                        Some(ref v) if v == val => {}
                        _ => return false,
                    }
                }
                true
            })
            .collect()
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let mut results = Vec::new();

        for nt in &self.node_tables_by_id {
            if let Some(zm) = nt.zone_map(&key) {
                if let Some(min_val) = min {
                    let op = if min_inclusive {
                        CompareOp::Ge
                    } else {
                        CompareOp::Gt
                    };
                    if !zm.might_match(op, min_val) {
                        continue;
                    }
                }
                if let Some(max_val) = max {
                    let op = if max_inclusive {
                        CompareOp::Le
                    } else {
                        CompareOp::Lt
                    };
                    if !zm.might_match(op, max_val) {
                        continue;
                    }
                }
            }
            if let Some(col) = nt.column(&key) {
                let table_id = nt.table_id();
                for offset in nt.current_matching_offsets(
                    &key,
                    col.find_in_range(min, max, min_inclusive, max_inclusive),
                ) {
                    let compact_id = encode_node_id(table_id, offset as u64);
                    results.push(self.to_original_node_id(compact_id));
                }
            }
        }

        results
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let mut might_match = false;
        for nt in &self.node_tables_by_id {
            match nt.zone_map(property) {
                Some(zm) => {
                    if zm.might_match(op, value) {
                        return true;
                    }
                }
                None => {
                    // No stats for this property in this table: conservatively assume match
                    might_match = true;
                }
            }
        }
        might_match
    }

    fn edge_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        // Conservative: no zone maps on edge properties
        true
    }

    fn statistics(&self) -> Arc<Statistics> {
        Arc::clone(&self.statistics)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.label_to_table_id
            .get(label)
            .and_then(|&tid| self.node_tables_by_id.get(tid as usize))
            .map_or(0.0, |nt| nt.len() as f64)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        let Some(rids) = self.edge_type_to_rel_id.get(edge_type) else {
            return 0.0;
        };
        let mut total_edges: usize = 0;
        let mut seen_tables = FxHashSet::default();
        for &rid in rids {
            let Some(rt) = self.rel_tables_by_id.get(rid as usize) else {
                continue;
            };
            total_edges += rt.num_edges();
            let table_id = if outgoing {
                rt.src_table_id()
            } else {
                rt.dst_table_id()
            };
            seen_tables.insert(table_id);
        }
        let total_nodes: usize = seen_tables
            .iter()
            .map(|&tid| self.resolve_node_table(tid).map_or(1, |nt| nt.len().max(1)))
            .sum();
        if total_nodes == 0 {
            return 0.0;
        }
        total_edges as f64 / total_nodes as f64
    }

    fn current_epoch(&self) -> EpochId {
        EpochId(1)
    }

    fn all_labels(&self) -> Vec<String> {
        let mut labels: FxHashSet<String> = self
            .temporal_label_index
            .keys()
            .map(ToString::to_string)
            .collect();
        for table in &self.node_tables_by_id {
            let has_non_temporal_node = table.node_ids().into_iter().any(|compact_id| {
                let id = if self.preserves_ids() {
                    self.to_original_node_id(compact_id)
                } else {
                    compact_id
                };
                !self.temporal_nodes().contains_key(&id)
            });
            if has_non_temporal_node {
                labels.insert(table.label().to_string());
            }
        }
        let mut labels: Vec<_> = labels.into_iter().collect();
        labels.sort();
        labels
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.edge_type_to_rel_id
            .keys()
            .map(|s| s.to_string())
            .collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        let mut keys = FxHashSet::<String>::default();

        for nt in &self.node_tables_by_id {
            for pk in nt.property_keys() {
                keys.insert(pk.as_str().to_string());
            }
        }

        for rt in &self.rel_tables_by_id {
            for pk in rt.property_keys() {
                keys.insert(pk.as_str().to_string());
            }
        }

        keys.into_iter().collect()
    }

    fn get_node_history(&self, _id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        Vec::new()
    }

    fn get_edge_history(&self, _id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        Vec::new()
    }
}

impl CompactStore {
    fn node_id_at(&self, table_id: u16, offset: u32) -> NodeId {
        let compact = encode_node_id(table_id, u64::from(offset));
        if self.preserves_ids() {
            self.to_original_node_id(compact)
        } else {
            compact
        }
    }

    /// Dest lists from current CSR (table-local offsets mapped to NodeIds).
    pub(crate) fn snapshot_csr_neighbors(
        &self,
        direction: Direction,
    ) -> Vec<(NodeId, Vec<NodeId>)> {
        let mut out = Vec::new();
        for (tid_usz, nt) in self.node_tables_by_id.iter().enumerate() {
            let tid = u16::try_from(tid_usz).unwrap_or(u16::MAX);
            let n = nt.len();
            for off in 0..n {
                let off_u = u32::try_from(off).unwrap_or(u32::MAX);
                let dests = self.collect_neighbors(tid, off_u, direction);
                if dests.is_empty() {
                    continue;
                }
                out.push((self.node_id_at(tid, off_u), dests));
            }
        }
        out
    }

    /// `|out(b) ∩ in(a)|` on dest-sorted `fwd`/`bwd` CSR slices.
    ///
    /// Requires one self-loop rel table (src label = dst label) with a
    /// derived current `fwd` and `bwd`. Mid-hop labels only when every
    /// node already has that label.
    pub(crate) fn count_csr_triangles(
        &self,
        starts: Option<&[NodeId]>,
        dest_label: Option<&str>,
    ) -> Option<u64> {
        if dest_label.is_some_and(|label| self.nodes_by_label_count(label) != self.node_count()) {
            return None;
        }
        if self.rel_tables_by_id.len() != 1 {
            return None;
        }
        let rt = self.rel_tables_by_id.first()?;
        if rt.src_table_id() != rt.dst_table_id() {
            return None;
        }
        if rt.fwd().num_edges() == 0 {
            return None;
        }
        let fwd = rt.fwd();
        let bwd = rt.bwd()?;
        let table = rt.src_table_id();
        let mut n = 0u64;
        let count_offset = |off: u32| {
            let in_a = bwd.neighbors(off);
            if in_a.is_empty() {
                return 0u64;
            }
            let mut local = 0u64;
            for &b in fwd.neighbors(off) {
                local += intersect_sorted_u32(fwd.neighbors(b), in_a) as u64;
            }
            local
        };
        match starts {
            None => {
                let limit = fwd.num_nodes().min(bwd.num_nodes());
                for off in 0..limit {
                    n += count_offset(u32::try_from(off).unwrap_or(u32::MAX));
                }
            }
            Some(starts) => {
                for &start in starts {
                    let Some((tid, off64)) = self.resolve_node(start) else {
                        continue;
                    };
                    if tid != table {
                        continue;
                    }
                    let Ok(off) = u32::try_from(off64) else {
                        continue;
                    };
                    n += count_offset(off);
                }
            }
        }
        Some(n)
    }
}

fn intersect_sorted_u32(left: &[u32], right: &[u32]) -> usize {
    let mut count = 0usize;
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Equal => {
                let value = left[left_index];
                let left_start = left_index;
                let right_start = right_index;
                while left_index < left.len() && left[left_index] == value {
                    left_index += 1;
                }
                while right_index < right.len() && right[right_index] == value {
                    right_index += 1;
                }
                count += (left_index - left_start) * (right_index - right_start);
            }
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
        }
    }
    count
}

impl GraphStoreSearch for CompactStore {
    fn find_nodes_in_range_iter<'a>(
        &'a self,
        property: &'a str,
        min: Option<&'a Value>,
        max: Option<&'a Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Box<dyn Iterator<Item = NodeId> + 'a> {
        let key = PropertyKey::new(property);

        let per_table = self.node_tables_by_id.iter().filter_map(move |nt| {
            // Whole-table skip via per-label zone map (existing behavior).
            if let Some(zm) = nt.zone_map(&key) {
                if let Some(min_val) = min {
                    let op = if min_inclusive {
                        CompareOp::Ge
                    } else {
                        CompareOp::Gt
                    };
                    if !zm.might_match(op, min_val) {
                        return None;
                    }
                }
                if let Some(max_val) = max {
                    let op = if max_inclusive {
                        CompareOp::Le
                    } else {
                        CompareOp::Lt
                    };
                    if !zm.might_match(op, max_val) {
                        return None;
                    }
                }
            }

            let col = nt.column(&key)?;
            let block_zones = nt.block_zone_maps_for(&key);
            let table_id = nt.table_id();
            let store = self;
            // reason: usize → u64 fits on every supported target (row count
            // bounded by u32::MAX per the section format).
            #[allow(clippy::cast_possible_truncation)]
            let iter = col
                .range_iter(block_zones, min, max, min_inclusive, max_inclusive)
                .map(move |offset| {
                    let compact_id = encode_node_id(table_id, offset as u64);
                    store.to_original_node_id(compact_id)
                });
            Some(iter)
        });

        Box::new(per_table.flatten())
    }
}

#[cfg(test)]
mod as_of_tests {
    use super::*;
    use crate::graph::compact::builder::CompactStoreBuilder;

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_native_current_excludes_closed_temporal_edge() {
        use crate::execution::operators::count_directed_triangles;
        use crate::graph::compact::layered::LayeredStore;
        use crate::graph::lpg::LpgStore;

        for with_properties in [false, true] {
            let source = Arc::new(LpgStore::new().unwrap());
            source.sync_epoch(EpochId::new(1));
            let a = source.create_node(&["V"]);
            let b = source.create_node(&["V"]);
            let c = source.create_node(&["V"]);
            let ab = source.create_edge(a, b, "R");
            let bc = source.create_edge(b, c, "R");
            let ca = source.create_edge(c, a, "R");
            if with_properties {
                for edge in [ab, bc, ca] {
                    source.set_edge_property(edge, "weight", Value::Int64(7));
                }
            }
            source.sync_epoch(EpochId::new(2));
            assert!(source.delete_edge(ca));
            let base = LayeredStore::from_native_temporal(source)
                .unwrap()
                .base_store_arc();
            // Exercise the actual native API: a fallback could hide a CSR bug.
            assert_eq!(base.try_count_all_directed_triangles(None), Some(0));
            assert_eq!(base.try_count_directed_triangles(&[a, b, c], None), Some(0));
            assert_eq!(
                count_directed_triangles(
                    base.as_ref(),
                    &[a, b, c],
                    &["R".into()],
                    None,
                    Some(EpochId::new(1)),
                    None,
                    false
                ),
                3
            );
            assert_eq!(
                count_directed_triangles(
                    base.as_ref(),
                    &[a, b, c],
                    &["R".into()],
                    None,
                    Some(EpochId::new(2)),
                    None,
                    false
                ),
                0
            );
            if with_properties {
                assert_eq!(
                    base.get_edge_at_epoch(ca, EpochId::new(1))
                        .unwrap()
                        .properties
                        .get(&PropertyKey::new("weight")),
                    Some(&Value::Int64(7))
                );
            }
        }
    }

    /// Invariant (SP1-5 slice 2): a base built from current values is all-open,
    /// so the as-of read at every real epoch equals the current read — both for
    /// a single property and for the whole node.
    #[test]
    fn all_open_store_as_of_equals_current() {
        let store = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("age", &[25, 30, 35], 6)
                    .column_dict("name", &["Alix", "Gus", "Vincent"])
            })
            .build()
            .unwrap();

        let age = PropertyKey::new("age");
        let name = PropertyKey::new("name");
        for id in store.nodes_by_label("Person") {
            let cur_age = store.get_node_property(id, &age);
            let cur_name = store.get_node_property(id, &name);
            for ep in [EpochId::INITIAL, EpochId::new(7), EpochId::new(10_000)] {
                // whole-node as-of read is present and carries current properties
                let node = store
                    .get_node_at_epoch(id, ep)
                    .expect("node visible at epoch");
                assert_eq!(node.properties.get(&age).cloned(), cur_age);
                assert_eq!(node.properties.get(&name).cloned(), cur_name);
                // single-property as-of accessor matches current
                assert_eq!(
                    store.get_node_property_at_epoch(id, &age, ep),
                    cur_age,
                    "property as-of must equal current for all-open base"
                );
            }
        }
    }

    /// SP2 slice 4: upgrading an all-open base's numeric column to temporal —
    /// folding a node's full history — preserves as-of (invariant #1) while the
    /// current read still returns the open value.
    #[test]
    fn upgrade_nodes_temporal_folds_numeric_history() {
        let base = CompactStoreBuilder::new()
            .node_table("Item", |t| t.column_bitpacked("score", &[300], 16))
            .build()
            .unwrap();
        let nid = encode_node_id(0, 0);
        let key = PropertyKey::new("score");
        let temporal = base.upgrade_nodes_temporal(|id| {
            if id == nid {
                vec![(
                    key.clone(),
                    vec![
                        (EpochId::new(10), Value::Int64(100)),
                        (EpochId::new(20), Value::Int64(200)),
                        (EpochId::new(30), Value::Int64(300)),
                    ],
                )]
            } else {
                Vec::new()
            }
        });
        assert_eq!(
            temporal.get_node_property_at_epoch(nid, &key, EpochId::new(15)),
            Some(Value::Int64(100))
        );
        assert_eq!(
            temporal.get_node_property_at_epoch(nid, &key, EpochId::new(25)),
            Some(Value::Int64(200))
        );
        assert_eq!(
            temporal.get_node_property_at_epoch(nid, &key, EpochId::new(35)),
            Some(Value::Int64(300))
        );
        // Current read is the node's open (latest) value.
        assert_eq!(
            temporal.get_node_property(nid, &key),
            Some(Value::Int64(300))
        );
    }

    /// The columnar `scrub_at_epoch` returns, per node and property, exactly the
    /// pointwise `get_node_property_at_epoch` value — without materializing a
    /// `Node` per node.
    #[test]
    fn scrub_at_epoch_columns_match_pointwise_as_of() {
        let base = CompactStoreBuilder::new()
            .node_table("Item", |t| t.column_bitpacked("score", &[0, 0], 16))
            .build()
            .unwrap();
        let key = PropertyKey::new("score");
        let n0 = encode_node_id(0, 0);
        let n1 = encode_node_id(0, 1);
        let temporal = base.upgrade_nodes_temporal(|id| {
            if id == n0 {
                vec![(
                    key.clone(),
                    vec![
                        (EpochId::new(10), Value::Int64(100)),
                        (EpochId::new(20), Value::Int64(200)),
                    ],
                )]
            } else if id == n1 {
                vec![(key.clone(), vec![(EpochId::new(5), Value::Int64(999))])]
            } else {
                Vec::new()
            }
        });

        let epoch = EpochId::new(15);
        let scrub = temporal.scrub_at_epoch(epoch);
        assert_eq!(scrub.len(), 1);
        let frame = &scrub[0];
        assert_eq!(frame.label.as_str(), "Item");
        assert_eq!(frame.node_ids.len(), 2);
        let col = &frame.columns[&key];
        for (i, id) in frame.node_ids.iter().enumerate() {
            assert_eq!(
                col[i],
                temporal.get_node_property_at_epoch(*id, &key, epoch)
            );
        }
        assert_eq!(col[0], Some(Value::Int64(100))); // n0 as-of 15 -> [10,20) version
        assert_eq!(col[1], Some(Value::Int64(999))); // n1 as-of 15 -> [5,PENDING) version
    }

    /// String property history folds into a temporal (Dict) column: invariant #1
    /// holds for strings — an old epoch reads the historical string, not current.
    #[test]
    fn upgrade_nodes_temporal_folds_string_history() {
        let base = CompactStoreBuilder::new()
            .node_table("Item", |t| t.column_dict("name", &["current"]))
            .build()
            .unwrap();
        let nid = encode_node_id(0, 0);
        let key = PropertyKey::new("name");
        let temporal = base.upgrade_nodes_temporal(|id| {
            if id == nid {
                vec![(
                    key.clone(),
                    vec![
                        (EpochId::new(10), Value::from("alpha")),
                        (EpochId::new(20), Value::from("beta")),
                    ],
                )]
            } else {
                Vec::new()
            }
        });
        assert_eq!(
            temporal.get_node_property_at_epoch(nid, &key, EpochId::new(15)),
            Some(Value::from("alpha"))
        );
        assert_eq!(
            temporal.get_node_property_at_epoch(nid, &key, EpochId::new(25)),
            Some(Value::from("beta"))
        );
        assert_eq!(
            temporal.get_node_property(nid, &key),
            Some(Value::from("beta"))
        );
    }

    /// Vector property history folds into a temporal (Float32Vector) column:
    /// invariant #1 holds for vectors — an old epoch reads the historical
    /// embedding, current reads the latest.
    #[test]
    fn upgrade_nodes_temporal_folds_vector_history() {
        use crate::graph::compact::column::ColumnCodec;

        let vec_value = |c: &[f32]| Value::Vector(Arc::from(c));
        let base = CompactStoreBuilder::new()
            .node_table("Item", |t| {
                t.column(
                    "embedding",
                    ColumnCodec::float32_vector(vec![0.0, 0.0, 0.0], 3),
                )
            })
            .build()
            .unwrap();
        let nid = encode_node_id(0, 0);
        let key = PropertyKey::new("embedding");
        let v_old = [0.1f32, 0.2, 0.3];
        let v_new = [0.4f32, 0.5, 0.6];
        let temporal = base.upgrade_nodes_temporal(|id| {
            if id == nid {
                vec![(
                    key.clone(),
                    vec![
                        (EpochId::new(10), vec_value(&v_old)),
                        (EpochId::new(20), vec_value(&v_new)),
                    ],
                )]
            } else {
                Vec::new()
            }
        });
        assert_eq!(
            temporal.get_node_property_at_epoch(nid, &key, EpochId::new(15)),
            Some(vec_value(&v_old))
        );
        assert_eq!(
            temporal.get_node_property_at_epoch(nid, &key, EpochId::new(25)),
            Some(vec_value(&v_new))
        );
        assert_eq!(
            temporal.get_node_property(nid, &key),
            Some(vec_value(&v_new))
        );
    }
}

#[cfg(test)]
mod triangle_multiplicity_tests {
    use super::*;
    use crate::graph::compact::builder::CompactStoreBuilder;

    #[test]
    fn intersect_sorted_u32_counts_run_product() {
        assert_eq!(intersect_sorted_u32(&[1, 1, 1], &[1, 1]), 6);
        assert_eq!(intersect_sorted_u32(&[1, 1], &[1, 1, 1]), 6);
    }

    #[test]
    fn csr_triangle_count_preserves_parallel_edge_walks() {
        let store = CompactStoreBuilder::new()
            .node_table("Node", |t| t.column_bitpacked("id", &[0, 1, 2], 2))
            .rel_table("R", "Node", "Node", |r| {
                r.edges([
                    (0, 1),
                    (0, 1),
                    (1, 2),
                    (1, 2),
                    (1, 2),
                    (2, 0),
                    (2, 0),
                    (2, 0),
                    (2, 0),
                ])
                .backward(true)
            })
            .build()
            .unwrap();
        let a = encode_node_id(0, 0);
        assert_eq!(store.count_csr_triangles(None, None), Some(72));
        assert_eq!(store.count_csr_triangles(Some(&[a]), None), Some(24));
    }

    #[test]
    fn csr_triangle_count_preserves_parallel_self_loop_walks() {
        let store = CompactStoreBuilder::new()
            .node_table("Node", |t| t.column_bitpacked("id", &[0], 1))
            .rel_table("R", "Node", "Node", |r| {
                r.edges([(0, 0), (0, 0), (0, 0)]).backward(true)
            })
            .build()
            .unwrap();
        let a = encode_node_id(0, 0);
        assert_eq!(store.count_csr_triangles(None, None), Some(27));
        assert_eq!(store.count_csr_triangles(Some(&[a]), None), Some(27));
    }
}
