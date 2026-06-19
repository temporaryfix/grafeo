//! `GraphStore` and `GraphStoreMut` trait implementations for `LpgStore`.
//!
//! Every method here is pure delegation to the existing `LpgStore` method.
//! The only adapters are `neighbors()` and `edges_from()`, which collect
//! the `impl Iterator` return into `Vec` for trait object safety.

use super::LpgStore;
use crate::execution::operators::{SharedReadTracker, SharedWriteTracker};
use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreMut, GraphStoreSearch};
#[cfg(feature = "vector-index")]
use crate::index::vector::{
    DistanceMetric, PropertyVectorAccessor, brute_force_knn, compute_distance,
};
use crate::statistics::Statistics;
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

impl GraphStore for LpgStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        LpgStore::get_node(self, id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        LpgStore::get_edge(self, id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        LpgStore::get_node_versioned(self, id, epoch, transaction_id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        LpgStore::get_edge_versioned(self, id, epoch, transaction_id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        LpgStore::get_node_at_epoch(self, id, epoch)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        LpgStore::get_edge_at_epoch(self, id, epoch)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        LpgStore::get_node_property(self, id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        LpgStore::get_edge_property(self, id, key)
    }

    fn pending_node_creates(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        LpgStore::pending_node_creates(self, transaction_id)
    }

    fn pending_edge_creates(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        LpgStore::pending_edge_creates(self, transaction_id)
    }

    fn register_read_tracker(&self, tx: TransactionId, tracker: SharedReadTracker) {
        LpgStore::register_read_tracker(self, tx, tracker);
    }

    fn unregister_read_tracker(&self, tx: TransactionId) {
        LpgStore::unregister_read_tracker(self, tx);
    }

    fn register_write_tracker(&self, tx: TransactionId, tracker: SharedWriteTracker) {
        LpgStore::register_write_tracker(self, tx, tracker);
    }

    fn unregister_write_tracker(&self, tx: TransactionId) {
        LpgStore::unregister_write_tracker(self, tx);
    }

    fn pending_node_deletes_peek(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        LpgStore::pending_node_deletes_peek(self, transaction_id)
    }

    fn pending_edge_deletes_peek(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        LpgStore::pending_edge_deletes_peek(self, transaction_id)
    }

    fn overlay_touched_entities(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<NodeId>, Vec<EdgeId>) {
        LpgStore::overlay_touched_entities(self, transaction_id)
    }

    fn overlay_touched_properties(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<(NodeId, Option<String>)>, Vec<(EdgeId, Option<String>)>) {
        LpgStore::overlay_touched_properties(self, transaction_id)
    }

    fn read_node_property_visible(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        LpgStore::read_node_property_visible(self, id, key, epoch, transaction_id)
    }

    fn read_edge_property_visible(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        LpgStore::read_edge_property_visible(self, id, key, epoch, transaction_id)
    }

    fn read_node_properties_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashMap<PropertyKey, Value> {
        LpgStore::read_node_properties_visible(self, id, epoch, transaction_id)
    }

    fn read_edge_properties_visible(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashMap<PropertyKey, Value> {
        LpgStore::read_edge_properties_visible(self, id, epoch, transaction_id)
    }

    fn read_node_labels_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashSet<ArcStr> {
        LpgStore::read_node_labels_visible(self, id, epoch, transaction_id)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        LpgStore::get_node_property_batch(self, ids, key)
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        LpgStore::get_nodes_properties_batch(self, ids)
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        LpgStore::get_nodes_properties_selective_batch(self, ids, keys)
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        LpgStore::get_edges_properties_selective_batch(self, ids, keys)
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        LpgStore::neighbors(self, node, direction).collect()
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        LpgStore::edges_from(self, node, direction).collect()
    }

    fn edges_from_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<(NodeId, EdgeId)> {
        LpgStore::edges_from_versioned(self, node, direction, epoch, transaction_id)
    }

    fn neighbors_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        LpgStore::neighbors_versioned(self, node, direction, epoch, transaction_id)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        LpgStore::out_degree(self, node)
    }

    fn in_degree(&self, node: NodeId) -> usize {
        LpgStore::in_degree(self, node)
    }

    fn has_backward_adjacency(&self) -> bool {
        LpgStore::has_backward_adjacency(self)
    }

    fn node_ids(&self) -> Vec<NodeId> {
        LpgStore::node_ids(self)
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        LpgStore::all_node_ids(self)
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        LpgStore::nodes_by_label(self, label)
    }

    fn nodes_by_label_visible(
        &self,
        label: &str,
        transaction_id: Option<TransactionId>,
    ) -> Vec<NodeId> {
        LpgStore::nodes_by_label_visible(self, label, transaction_id)
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        LpgStore::nodes_by_label_count(self, label)
    }

    fn node_count(&self) -> usize {
        LpgStore::node_count(self)
    }

    fn edge_count(&self) -> usize {
        LpgStore::edge_count(self)
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        LpgStore::edge_type(self, id)
    }

    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        LpgStore::edge_type_versioned(self, id, epoch, transaction_id)
    }

    fn has_property_index(&self, property: &str) -> bool {
        LpgStore::has_property_index(self, property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        LpgStore::find_nodes_by_property(self, property, value)
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        LpgStore::find_nodes_by_properties(self, conditions)
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        LpgStore::find_nodes_in_range(self, property, min, max, min_inclusive, max_inclusive)
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        LpgStore::node_property_might_match(self, property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        LpgStore::edge_property_might_match(self, property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        LpgStore::statistics(self)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        LpgStore::estimate_label_cardinality(self, label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        LpgStore::estimate_avg_degree(self, edge_type, outgoing)
    }

    fn current_epoch(&self) -> EpochId {
        LpgStore::current_epoch(self)
    }

    fn all_labels(&self) -> Vec<String> {
        LpgStore::all_labels(self)
    }

    fn all_edge_types(&self) -> Vec<String> {
        LpgStore::all_edge_types(self)
    }

    fn all_property_keys(&self) -> Vec<String> {
        LpgStore::all_property_keys(self)
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        LpgStore::is_node_visible_at_epoch(self, id, epoch)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::is_node_visible_versioned(self, id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        LpgStore::is_edge_visible_at_epoch(self, id, epoch)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::is_edge_visible_versioned(self, id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        LpgStore::filter_visible_node_ids(self, ids, epoch)
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        LpgStore::filter_visible_node_ids_versioned(self, ids, epoch, transaction_id)
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        LpgStore::get_node_history(self, id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        LpgStore::get_edge_history(self, id)
    }
}

impl GraphStoreSearch for LpgStore {
    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.get_text_index(label, property).is_some()
    }

    /// Returns the label part of every `"label:property"` index key whose
    /// property component equals `property`.
    ///
    /// The key format is `"label:property"`.  We split on the **last** `':'`
    /// so that label names containing a colon (unusual but legal) are handled
    /// correctly.
    #[cfg(feature = "text-index")]
    fn text_index_labels_for_property(&self, property: &str) -> Vec<String> {
        // Match by suffix `":{property}"` rather than splitting the key on a
        // single ':'.  The key format `"{label}:{property}"` is ambiguous under
        // a *single* split when EITHER the label or the property contains a ':'
        // (`split_once` mis-handles colon-properties; `rsplit_once` mis-handles
        // colon-labels).  Stripping the exact `":{property}"` suffix recovers the
        // label unambiguously for the precise property string queried, so a node
        // whose `(label, property)` index write is recorded as
        // `format!("{label}:{property}")` is always enumerated here.
        let suffix = format!(":{property}");
        self.text_indexes
            .read()
            .keys()
            .filter_map(|key| key.strip_suffix(&suffix).map(str::to_string))
            .collect()
    }

    #[cfg(feature = "text-index")]
    fn score_text(&self, node_id: NodeId, label: &str, property: &str, query: &str) -> Option<f64> {
        let index = self.get_text_index(label, property)?;
        let guard = index.read();
        let score = guard.score_document(node_id, query);
        Some(score)
    }

    /// Snapshot-aware per-row BM25 score — records the index read for SSI.
    ///
    /// Builds the `"label:property"` index key and delegates to
    /// [`LpgStore::score_text_visible_impl`], which records the index read in
    /// the SSI read-set (anti-phantom) and then scores the node using postings
    /// visible at `(epoch, tx)`.
    #[cfg(feature = "text-index")]
    fn score_text_visible(
        &self,
        node_id: grafeo_common::types::NodeId,
        label: &str,
        property: &str,
        query: &str,
        epoch: grafeo_common::types::EpochId,
        tx: grafeo_common::types::TransactionId,
    ) -> Option<f64> {
        let index_key = format!("{label}:{property}");
        self.score_text_visible_impl(&index_key, node_id, query, epoch, tx)
    }

    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        if let Some(index) = self.get_text_index(label, property) {
            index.read().search(query, k)
        } else {
            Vec::new()
        }
    }

    /// Snapshot-aware top-`k` BM25 search — records the index read for SSI.
    ///
    /// Builds the `"label:property"` index key and delegates to
    /// [`LpgStore::search_text_visible`], which merges committed postings with
    /// the per-transaction write delta and records the index read in the SSI
    /// read-set so that a concurrent indexed SET forms an rw-antidependency.
    #[cfg(feature = "text-index")]
    fn text_search_visible(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
        epoch: grafeo_common::types::EpochId,
        tx: grafeo_common::types::TransactionId,
    ) -> Vec<(grafeo_common::types::NodeId, f64)> {
        let index_key = format!("{label}:{property}");
        self.search_text_visible(&index_key, query, k, epoch, tx)
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
    ) -> Vec<(NodeId, f64)> {
        if let Some(index) = self.get_text_index(label, property) {
            index.read().search_with_threshold(query, threshold)
        } else {
            Vec::new()
        }
    }

    /// Snapshot-aware threshold BM25 search — records the index read for SSI.
    ///
    /// Builds the `"label:property"` index key and delegates to
    /// [`LpgStore::search_text_with_threshold_visible`], which merges committed
    /// postings with the per-transaction write delta and records the index read
    /// in the SSI read-set.
    #[cfg(feature = "text-index")]
    fn text_search_with_threshold_visible(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
        epoch: grafeo_common::types::EpochId,
        tx: grafeo_common::types::TransactionId,
    ) -> Vec<(grafeo_common::types::NodeId, f64)> {
        let index_key = format!("{label}:{property}");
        self.search_text_with_threshold_visible(&index_key, query, threshold, epoch, tx)
    }

    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.get_vector_index(label, property).is_some()
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_metric(&self, label: &str, property: &str) -> Option<DistanceMetric> {
        self.get_vector_index(label, property)
            .map(|idx| idx.config().metric)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        // HNSW path: matching index + matching metric.
        if let Some(label_name) = label
            && let Some(index) = self.get_vector_index(label_name, property)
            && index.config().metric == metric
        {
            let store_ref: &dyn GraphStore = self;
            let accessor = PropertyVectorAccessor::new(store_ref, property);
            return index
                .search_with_ef(query, k, 64, &accessor)
                .into_iter()
                .map(|(id, d)| (id, f64::from(d)))
                .collect();
        }

        // Brute-force fallback: scan nodes, compute distance, take top-k.
        // Keep `Arc<[f32]>` from `Value::Vector` instead of copying each vector
        // into an owned `Vec<f32>`: `brute_force_knn` only needs `&[f32]` and
        // the store already owns the embedding data behind an Arc.
        let node_ids = match label {
            Some(l) => <Self as GraphStore>::nodes_by_label(self, l),
            None => <Self as GraphStore>::node_ids(self),
        };
        let property_key = PropertyKey::new(property);
        let vectors: Vec<(NodeId, Arc<[f32]>)> = node_ids
            .into_iter()
            .filter_map(|id| {
                <Self as GraphStore>::get_node_property(self, id, &property_key).and_then(|v| {
                    if let Value::Vector(arc) = v {
                        Some((id, arc))
                    } else {
                        None
                    }
                })
            })
            .collect();
        let iter = vectors.iter().map(|(id, arc)| (*id, arc.as_ref()));
        brute_force_knn(iter, query, k, metric)
            .into_iter()
            .map(|(id, d)| (id, f64::from(d)))
            .collect()
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        // Threshold mode always scans: HNSW has no threshold API. Iterate all
        // candidates, compute exact distance, keep those under the threshold,
        // then sort nearest-first.
        let node_ids = match label {
            Some(l) => <Self as GraphStore>::nodes_by_label(self, l),
            None => <Self as GraphStore>::node_ids(self),
        };
        let property_key = PropertyKey::new(property);
        let mut results: Vec<(NodeId, f64)> = node_ids
            .into_iter()
            .filter_map(|id| {
                <Self as GraphStore>::get_node_property(self, id, &property_key).and_then(|v| {
                    if let Value::Vector(vec) = v {
                        let d = f64::from(compute_distance(query, &vec, metric));
                        (d <= threshold).then_some((id, d))
                    } else {
                        None
                    }
                })
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Snapshot-aware top-`k` vector search — delegates to
    /// [`LpgStore::search_vector_visible`], which uses snapshot visibility,
    /// as-of-`epoch` scoring via `SnapshotVectorAccessor`, and a brute-force
    /// tx-overlay merge for read-your-writes completeness.
    #[cfg(feature = "vector-index")]
    fn vector_search_visible(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        epoch: grafeo_common::types::EpochId,
        tx: grafeo_common::types::TransactionId,
    ) -> Vec<(grafeo_common::types::NodeId, f64)> {
        let index_key = format!("{label}:{property}");
        self.search_vector_visible(&index_key, query, k, epoch, tx)
            .into_iter()
            .map(|(id, d)| (id, f64::from(d)))
            .collect()
    }
}

impl GraphStoreMut for LpgStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        LpgStore::create_node(self, labels)
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        LpgStore::create_node_versioned(self, labels, epoch, transaction_id)
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        LpgStore::create_edge(self, src, dst, edge_type)
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        LpgStore::create_edge_versioned(self, src, dst, edge_type, epoch, transaction_id)
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        LpgStore::batch_create_edges(self, edges)
    }

    fn delete_node(&self, id: NodeId) -> bool {
        LpgStore::delete_node(self, id)
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if transaction_id == TransactionId::SYSTEM {
            LpgStore::delete_node_at_epoch(self, id, epoch)
        } else {
            LpgStore::delete_node_transactional(self, id, epoch, transaction_id)
        }
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        LpgStore::delete_node_edges(self, node_id);
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        LpgStore::delete_edge(self, id)
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if transaction_id == TransactionId::SYSTEM {
            LpgStore::delete_edge_at_epoch(self, id, epoch)
        } else {
            LpgStore::delete_edge_transactional(self, id, epoch, transaction_id)
        }
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        LpgStore::set_node_property(self, id, key, value);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        LpgStore::set_edge_property(self, id, key, value);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        LpgStore::set_node_property_versioned(self, id, key, value, transaction_id);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        LpgStore::set_edge_property_versioned(self, id, key, value, transaction_id);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        LpgStore::remove_node_property(self, id, key)
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        LpgStore::remove_edge_property(self, id, key)
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        LpgStore::remove_node_property_versioned(self, id, key, transaction_id)
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        LpgStore::remove_edge_property_versioned(self, id, key, transaction_id)
    }

    fn set_node_property_buffered(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        LpgStore::set_node_property_buffered(self, id, key, value, transaction_id);
    }

    fn remove_node_property_buffered(&self, id: NodeId, key: &str, transaction_id: TransactionId) {
        LpgStore::remove_node_property_buffered(self, id, key, transaction_id);
    }

    fn set_edge_property_buffered(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        LpgStore::set_edge_property_buffered(self, id, key, value, transaction_id);
    }

    fn remove_edge_property_buffered(&self, id: EdgeId, key: &str, transaction_id: TransactionId) {
        LpgStore::remove_edge_property_buffered(self, id, key, transaction_id);
    }

    fn apply_tx_overlay(&self, transaction_id: TransactionId) {
        LpgStore::apply_tx_overlay(self, transaction_id);
    }

    fn drop_tx_overlay(&self, transaction_id: TransactionId) {
        LpgStore::drop_tx_overlay(self, transaction_id);
    }

    fn tx_overlay_snapshot(&self, transaction_id: TransactionId) -> crate::graph::lpg::TxDelta {
        LpgStore::tx_overlay_snapshot(self, transaction_id)
    }

    fn tx_overlay_restore(
        &self,
        transaction_id: TransactionId,
        snapshot: crate::graph::lpg::TxDelta,
    ) {
        LpgStore::tx_overlay_restore(self, transaction_id, snapshot);
    }

    fn finalize_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        node_ids: &[NodeId],
    ) {
        LpgStore::finalize_deletes_by_id(self, transaction_id, commit_epoch, node_ids);
    }

    fn take_pending_deletes(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        LpgStore::take_pending_deletes(self, transaction_id)
    }

    fn finalize_edge_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        edges: &[(NodeId, EdgeId, NodeId)],
    ) {
        LpgStore::finalize_edge_deletes_by_id(self, transaction_id, commit_epoch, edges);
    }

    fn take_pending_edge_deletes(
        &self,
        transaction_id: TransactionId,
    ) -> Vec<(NodeId, EdgeId, NodeId)> {
        LpgStore::take_pending_edge_deletes(self, transaction_id)
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        LpgStore::add_label(self, node_id, label)
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        LpgStore::remove_label(self, node_id, label)
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::add_label_versioned(self, node_id, label, transaction_id)
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        LpgStore::remove_label_versioned(self, node_id, label, transaction_id)
    }

    fn add_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        LpgStore::add_label_buffered(self, node_id, label, transaction_id);
    }

    fn remove_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        LpgStore::remove_label_buffered(self, node_id, label, transaction_id);
    }

    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: &[(PropertyKey, Value)],
    ) -> NodeId {
        // Delegate to LpgStore's optimized version that sets props under a single lock.
        LpgStore::create_node_with_props(
            self,
            labels,
            properties.iter().map(|(k, v)| (k.clone(), v.clone())),
        )
    }

    fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(PropertyKey, Value)],
    ) -> EdgeId {
        LpgStore::create_edge_with_props(
            self,
            src,
            dst,
            edge_type,
            properties.iter().map(|(k, v)| (k.clone(), v.clone())),
        )
    }
}
