//! WAL-aware graph store wrapper.
//!
//! Wraps an [`LpgStore`] and logs every mutation to the WAL so that
//! query-engine mutations (INSERT, DELETE, SET via GQL/Cypher/etc.)
//! survive a close/reopen cycle.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, EpochId, GraphPath, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::execution::operators::{SharedReadTracker, SharedWriteTracker};
#[cfg(test)]
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::lpg::{CompareOp, Edge, Node};
use grafeo_core::graph::{
    Direction, GraphStore, GraphStoreMut, GraphStoreSearch, PropertyIndexRequest,
    TxStructuralSnapshot,
};
use grafeo_core::statistics::Statistics;
use grafeo_storage::wal::{LpgMutationOp, LpgWal, WalRecord};

use arcstr::ArcStr;

/// A [`GraphStoreMut`] decorator that delegates every call to an inner store
/// and additionally logs mutation operations to the WAL.
///
/// Read-only methods are forwarded without any WAL interaction.
///
/// Every mutation carries the exact root-relative GraphPath; no replay cursor.
pub(crate) struct WalGraphStore {
    inner: Arc<dyn GraphStoreMut>,
    wal: Arc<LpgWal>,
    graph: GraphPath,
    /// Shared durability poison flag (None in unit tests).
    poison: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl WalGraphStore {
    /// Wraps one exact graph incarnation and its canonical durable coordinate.
    pub fn new(inner: Arc<dyn GraphStoreMut>, wal: Arc<LpgWal>, graph: GraphPath) -> Self {
        Self {
            inner,
            wal,
            graph,
            poison: None,
        }
    }

    /// Attach the database durability poison flag.
    pub fn with_poison(mut self, poison: Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.poison = Some(poison);
        self
    }

    fn poison(&self) {
        if let Some(p) = &self.poison {
            p.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn log_lpg(&self, transaction_id: TransactionId, op: LpgMutationOp) {
        if let Err(_e) = self
            .wal
            .log(&WalRecord::lpg(transaction_id, self.graph.clone(), op))
        {
            self.poison();
        }
    }
}

// ---------------------------------------------------------------------------
// GraphStore (read-only): pure delegation
// ---------------------------------------------------------------------------

impl GraphStore for WalGraphStore {
    fn lpg_commit_target(
        &self,
    ) -> grafeo_common::utils::error::Result<grafeo_core::graph::traits::LpgCommitTarget<'_>> {
        self.inner.lpg_commit_target()
    }

    fn get_node(&self, id: NodeId) -> Option<Node> {
        self.inner.get_node(id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.inner.get_edge(id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        self.inner.get_node_versioned(id, epoch, transaction_id)
    }

    fn prepare_index_node_rows(
        &self,
        publication_epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> grafeo_common::utils::error::Result<Vec<Node>> {
        self.inner
            .prepare_index_node_rows(publication_epoch, transaction_id)
    }

    fn prepare_index_node_rows_by_id(
        &self,
        publication_epoch: EpochId,
        transaction_id: Option<TransactionId>,
        ids: &[NodeId],
    ) -> grafeo_common::utils::error::Result<Vec<Node>> {
        self.inner
            .prepare_index_node_rows_by_id(publication_epoch, transaction_id, ids)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.inner.get_edge_versioned(id, epoch, transaction_id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        self.inner.get_node_at_epoch(id, epoch)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        self.inner.get_edge_at_epoch(id, epoch)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_edge_property(id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        self.inner.get_node_property_batch(ids, key)
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_nodes_properties_batch(ids)
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_nodes_properties_selective_batch(ids, keys)
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_edges_properties_selective_batch(ids, keys)
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        GraphStore::neighbors(self.inner.as_ref(), node, direction)
    }

    fn fill_neighbors(&self, node: NodeId, direction: Direction, out: &mut Vec<NodeId>) {
        self.inner.fill_neighbors(node, direction, out);
    }

    fn snapshot_neighbors(&self, direction: Direction) -> Vec<(NodeId, Vec<NodeId>)> {
        self.inner.snapshot_neighbors(direction)
    }

    fn try_count_directed_triangles(
        &self,
        starts: &[NodeId],
        dest_label: Option<&str>,
    ) -> Option<u64> {
        self.inner.try_count_directed_triangles(starts, dest_label)
    }

    fn try_count_all_directed_triangles(&self, dest_label: Option<&str>) -> Option<u64> {
        self.inner.try_count_all_directed_triangles(dest_label)
    }

    fn fill_neighbors_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<NodeId>,
    ) {
        self.inner
            .fill_neighbors_at_epoch(node, direction, epoch, out);
    }

    fn fill_neighbors_of_types_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        types: &[String],
        out: &mut Vec<NodeId>,
    ) {
        self.inner
            .fill_neighbors_of_types_at_epoch(node, direction, epoch, types, out);
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        GraphStore::edges_from(self.inner.as_ref(), node, direction)
    }

    fn edges_from_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<(NodeId, EdgeId)> {
        self.inner
            .edges_from_versioned(node, direction, epoch, transaction_id)
    }

    fn neighbors_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        self.inner
            .neighbors_versioned(node, direction, epoch, transaction_id)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.inner.out_degree(node)
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.inner.in_degree(node)
    }

    fn has_backward_adjacency(&self) -> bool {
        self.inner.has_backward_adjacency()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.node_ids()
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        self.inner.all_node_ids()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        self.inner.nodes_by_label(label)
    }

    fn nodes_with_buffered_property(
        &self,
        transaction_id: TransactionId,
        key: &PropertyKey,
    ) -> Option<Vec<NodeId>> {
        self.inner.nodes_with_buffered_property(transaction_id, key)
    }

    fn node_has_label(&self, id: NodeId, label: &str) -> bool {
        self.inner.node_has_label(id, label)
    }

    fn node_has_label_visible(
        &self,
        id: NodeId,
        label: &str,
        transaction_id: Option<TransactionId>,
    ) -> bool {
        self.inner.node_has_label_visible(id, label, transaction_id)
    }

    fn node_has_label_at_epoch(
        &self,
        id: NodeId,
        label: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .node_has_label_at_epoch(id, label, epoch, transaction_id)
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        self.inner.nodes_by_label_count(label)
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        self.inner.edge_type(id)
    }

    fn has_property_index(&self, property: &str) -> bool {
        self.inner.has_property_index(property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        self.inner.find_nodes_by_property(property, value)
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        self.inner.find_nodes_by_properties(conditions)
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        self.inner
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.edge_property_might_match(property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        self.inner.statistics()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.inner.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.inner.estimate_avg_degree(edge_type, outgoing)
    }

    fn current_epoch(&self) -> EpochId {
        self.inner.current_epoch()
    }

    fn all_labels(&self) -> Vec<String> {
        self.inner.all_labels()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.inner.all_edge_types()
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.inner.all_property_keys()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.inner.is_node_visible_at_epoch(id, epoch)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .is_node_visible_versioned(id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        self.inner.is_edge_visible_at_epoch(id, epoch)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .is_edge_visible_versioned(id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        self.inner.filter_visible_node_ids(ids, epoch)
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        self.inner
            .filter_visible_node_ids_versioned(ids, epoch, transaction_id)
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        self.inner.get_node_history(id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        self.inner.get_edge_history(id)
    }

    // --- Task 6: snapshot-aware read delegation (unified-MVCC) ---
    //
    // Reads have no WAL log side effects, so we can safely delegate to the
    // inner LpgStore's snapshot-aware accessors. The per-transaction property
    // delta lives in the inner LpgStore.

    fn pending_node_creates(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.inner.pending_node_creates(transaction_id)
    }

    fn pending_edge_creates(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        self.inner.pending_edge_creates(transaction_id)
    }

    fn register_read_tracker(&self, tx: TransactionId, tracker: SharedReadTracker) {
        self.inner.register_read_tracker(tx, tracker);
    }

    fn unregister_read_tracker(&self, tx: TransactionId) {
        self.inner.unregister_read_tracker(tx);
    }

    fn record_label_predicate_read(&self, tx: TransactionId, label: &str) {
        self.inner.record_label_predicate_read(tx, label);
    }

    fn record_rel_type_predicate_read(&self, tx: TransactionId, rel_type: &str) {
        self.inner.record_rel_type_predicate_read(tx, rel_type);
    }

    fn record_lpg_dataset_read(&self, tx: TransactionId) {
        self.inner.record_lpg_dataset_read(tx);
    }

    fn register_write_tracker(&self, tx: TransactionId, tracker: SharedWriteTracker) {
        self.inner.register_write_tracker(tx, tracker);
    }

    fn unregister_write_tracker(&self, tx: TransactionId) {
        self.inner.unregister_write_tracker(tx);
    }

    fn pending_node_deletes_peek(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.inner.pending_node_deletes_peek(transaction_id)
    }

    fn pending_edge_deletes_peek(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        self.inner.pending_edge_deletes_peek(transaction_id)
    }

    fn overlay_touched_entities(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<NodeId>, Vec<EdgeId>) {
        self.inner.overlay_touched_entities(transaction_id)
    }

    fn overlay_touched_properties(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<(NodeId, Option<String>)>, Vec<(EdgeId, Option<String>)>) {
        self.inner.overlay_touched_properties(transaction_id)
    }

    fn read_node_property_visible(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        self.inner
            .read_node_property_visible(id, key, epoch, transaction_id)
    }

    fn read_edge_property_visible(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        self.inner
            .read_edge_property_visible(id, key, epoch, transaction_id)
    }

    fn read_node_properties_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> grafeo_common::utils::hash::FxHashMap<PropertyKey, Value> {
        self.inner
            .read_node_properties_visible(id, epoch, transaction_id)
    }

    fn read_edge_properties_visible(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> grafeo_common::utils::hash::FxHashMap<PropertyKey, Value> {
        self.inner
            .read_edge_properties_visible(id, epoch, transaction_id)
    }

    // --- Task 5 (label reads, unified-MVCC) ---
    //
    // Label reads have no WAL log side effects, so delegate to the inner
    // LpgStore's snapshot-aware label accessors. The per-transaction label
    // delta lives in the inner LpgStore.
    //
    // Buffered label methods delegate to the same overlay delta below.
    // Prepared commit logs complete images, not intermediate label intents.

    fn read_node_labels_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> grafeo_common::utils::hash::FxHashSet<arcstr::ArcStr> {
        self.inner
            .read_node_labels_visible(id, epoch, transaction_id)
    }

    fn nodes_by_label_visible(
        &self,
        label: &str,
        transaction_id: Option<TransactionId>,
    ) -> Vec<NodeId> {
        self.inner.nodes_by_label_visible(label, transaction_id)
    }
}

// Pure delegation: the WAL wrapper logs mutations but owns no index state,
// so every text/vector lookup has to fall through to the underlying
// `LpgStore`. A stub impl silently turns into "no index exists" at every
// call site (has_text_index → false, text_search → [], etc.), which
// regressed hybrid queries on persistent DBs until it was caught by the
// `_persistent` spec variants — see issue #308.
impl GraphStoreSearch for WalGraphStore {
    fn lookup_nodes_indexed(
        &self,
        request: PropertyIndexRequest<'_>,
    ) -> grafeo_common::utils::error::Result<Option<Vec<NodeId>>> {
        self.inner.lookup_nodes_indexed(request)
    }

    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.inner.has_text_index(label, property)
    }

    #[cfg(feature = "text-index")]
    fn text_index_labels_for_property(&self, property: &str) -> Vec<String> {
        self.inner.text_index_labels_for_property(property)
    }

    #[cfg(feature = "text-index")]
    fn score_text(&self, node_id: NodeId, label: &str, property: &str, query: &str) -> Option<f64> {
        self.inner.score_text(node_id, label, property, query)
    }

    #[cfg(feature = "text-index")]
    fn score_text_visible(
        &self,
        node_id: NodeId,
        label: &str,
        property: &str,
        query: &str,
        epoch: EpochId,
        tx: TransactionId,
    ) -> grafeo_common::utils::error::Result<Option<f64>> {
        self.inner
            .score_text_visible(node_id, label, property, query, epoch, tx)
    }

    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        self.inner.text_search(label, property, query, k)
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
    ) -> Vec<(NodeId, f64)> {
        self.inner
            .text_search_with_threshold(label, property, query, threshold)
    }

    #[cfg(feature = "text-index")]
    fn text_search_visible(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
        epoch: EpochId,
        tx: TransactionId,
    ) -> grafeo_common::utils::error::Result<Vec<(NodeId, f64)>> {
        self.inner
            .text_search_visible(label, property, query, k, epoch, tx)
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold_visible(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
        epoch: EpochId,
        tx: TransactionId,
    ) -> grafeo_common::utils::error::Result<Vec<(NodeId, f64)>> {
        self.inner
            .text_search_with_threshold_visible(label, property, query, threshold, epoch, tx)
    }

    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.inner.has_vector_index(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_metric(
        &self,
        label: &str,
        property: &str,
    ) -> Option<grafeo_core::index::vector::DistanceMetric> {
        self.inner.vector_index_metric(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: grafeo_core::index::vector::DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        self.inner.vector_search(label, property, query, k, metric)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: grafeo_core::index::vector::DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        self.inner
            .vector_search_with_threshold(label, property, query, threshold, metric)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_visible(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        epoch: grafeo_common::types::EpochId,
        tx: grafeo_common::types::TransactionId,
    ) -> Vec<(NodeId, f64)> {
        self.inner
            .vector_search_visible(label, property, query, k, epoch, tx)
    }
}

// ---------------------------------------------------------------------------
// GraphStoreMut: delegate + WAL log
// ---------------------------------------------------------------------------

impl GraphStoreMut for WalGraphStore {
    fn lpg_commit_store(self: Arc<Self>) -> Option<Arc<grafeo_core::graph::lpg::LpgStore>> {
        Arc::clone(&self.inner).lpg_commit_store()
    }

    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.inner.create_node(labels);
        self.log_lpg(
            TransactionId::SYSTEM,
            LpgMutationOp::CreateNode {
                id,
                labels: labels.iter().map(|s| (*s).to_string()).collect(),
            },
        );
        id
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self
            .inner
            .create_node_versioned(labels, epoch, transaction_id);
        self.log_lpg(
            transaction_id,
            LpgMutationOp::CreateNode {
                id,
                labels: labels.iter().map(|s| (*s).to_string()).collect(),
            },
        );
        id
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let id = self.inner.create_edge(src, dst, edge_type);
        if !id.is_valid() {
            return id;
        }
        self.log_lpg(
            TransactionId::SYSTEM,
            LpgMutationOp::CreateEdge {
                id,
                src,
                dst,
                edge_type: edge_type.to_string(),
            },
        );
        id
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let id = self
            .inner
            .create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
        if !id.is_valid() {
            return id;
        }
        self.log_lpg(
            transaction_id,
            LpgMutationOp::CreateEdge {
                id,
                src,
                dst,
                edge_type: edge_type.to_string(),
            },
        );
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let ids = self.inner.batch_create_edges(edges);
        for (id, (src, dst, edge_type)) in ids.iter().zip(edges) {
            self.log_lpg(
                TransactionId::SYSTEM,
                LpgMutationOp::CreateEdge {
                    id: *id,
                    src: *src,
                    dst: *dst,
                    edge_type: (*edge_type).to_string(),
                },
            );
        }
        ids
    }

    fn delete_node(&self, id: NodeId) -> bool {
        let deleted = self.inner.delete_node(id);
        if deleted {
            self.log_lpg(TransactionId::SYSTEM, LpgMutationOp::DeleteNode { id });
        }
        deleted
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let deleted = self.inner.delete_node_versioned(id, epoch, transaction_id);
        if deleted {
            self.log_lpg(transaction_id, LpgMutationOp::DeleteNode { id });
        }
        deleted
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Collect edge IDs before deletion so we can log them
        let outgoing: Vec<EdgeId> = self
            .inner
            .edges_from(node_id, Direction::Outgoing)
            .into_iter()
            .map(|(_, eid)| eid)
            .collect();
        let incoming: Vec<EdgeId> = self
            .inner
            .edges_from(node_id, Direction::Incoming)
            .into_iter()
            .map(|(_, eid)| eid)
            .collect();

        self.inner.delete_node_edges(node_id);

        for id in outgoing.into_iter().chain(incoming) {
            self.log_lpg(TransactionId::SYSTEM, LpgMutationOp::DeleteEdge { id });
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        let deleted = self.inner.delete_edge(id);
        if deleted {
            self.log_lpg(TransactionId::SYSTEM, LpgMutationOp::DeleteEdge { id });
        }
        deleted
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let deleted = self.inner.delete_edge_versioned(id, epoch, transaction_id);
        if deleted {
            self.log_lpg(transaction_id, LpgMutationOp::DeleteEdge { id });
        }
        deleted
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        // Store first, WAL second: consistent lock ordering with create/delete
        // methods to prevent ABBA deadlock between store locks and WAL locks.
        self.inner.set_node_property(id, key, value.clone());
        self.log_lpg(
            TransactionId::SYSTEM,
            LpgMutationOp::SetNodeProperty {
                id,
                key: key.to_string(),
                value,
            },
        );
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.inner.set_edge_property(id, key, value.clone());
        self.log_lpg(
            TransactionId::SYSTEM,
            LpgMutationOp::SetEdgeProperty {
                id,
                key: key.to_string(),
                value,
            },
        );
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_node_property(id, key);
        if removed.is_some() {
            self.log_lpg(
                TransactionId::SYSTEM,
                LpgMutationOp::RemoveNodeProperty {
                    id,
                    key: key.to_string(),
                },
            );
        }
        removed
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_edge_property(id, key);
        if removed.is_some() {
            self.log_lpg(
                TransactionId::SYSTEM,
                LpgMutationOp::RemoveEdgeProperty {
                    id,
                    key: key.to_string(),
                },
            );
        }
        removed
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let added = self.inner.add_label(node_id, label);
        if added {
            self.log_lpg(
                TransactionId::SYSTEM,
                LpgMutationOp::AddNodeLabel {
                    id: node_id,
                    label: label.to_string(),
                },
            );
        }
        added
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let removed = self.inner.remove_label(node_id, label);
        if removed {
            self.log_lpg(
                TransactionId::SYSTEM,
                LpgMutationOp::RemoveNodeLabel {
                    id: node_id,
                    label: label.to_string(),
                },
            );
        }
        removed
    }

    // --- Transactional buffered writes (unified-MVCC) ---
    //
    // Override `*_buffered` to delegate to the inner `LpgStore`'s buffered path,
    // which records the mutation in the transaction's overlay delta instead of
    // writing through to the committed column. This is what gives WAL-wrapped
    // persistent stores the same uncommitted-write isolation as a bare
    // `LpgStore`: other sessions no longer observe a transaction's uncommitted
    // Cypher writes (see `tests/wal_mvcc_isolation.rs`).
    //
    // WAL logging is intentionally unchanged from the non-buffered overrides
    // above: each mutation is logged immediately (store-first, WAL-second for
    // lock ordering). Durability is governed by the WAL's positional
    // transaction framing — the session emits `TransactionCommit` /
    // `TransactionAbort` markers around the record stream, and recovery
    // (`WalRecovery::recover`) flushes a pending transaction's records to the
    // committed set only on commit, discarding them on abort or an incomplete
    // tail. So only the in-memory routing changes (write-through -> overlay);
    // the WAL byte stream and recovery semantics are unchanged.

    fn set_node_property_buffered(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.inner
            .set_node_property_buffered(id, key, value.clone(), transaction_id);
        self.log_lpg(
            transaction_id,
            LpgMutationOp::SetNodeProperty {
                id,
                key: key.to_string(),
                value,
            },
        );
    }

    fn remove_node_property_buffered(&self, id: NodeId, key: &str, transaction_id: TransactionId) {
        self.inner
            .remove_node_property_buffered(id, key, transaction_id);
        self.log_lpg(
            transaction_id,
            LpgMutationOp::RemoveNodeProperty {
                id,
                key: key.to_string(),
            },
        );
    }

    fn set_edge_property_buffered(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.inner
            .set_edge_property_buffered(id, key, value.clone(), transaction_id);
        self.log_lpg(
            transaction_id,
            LpgMutationOp::SetEdgeProperty {
                id,
                key: key.to_string(),
                value,
            },
        );
    }

    fn remove_edge_property_buffered(&self, id: EdgeId, key: &str, transaction_id: TransactionId) {
        self.inner
            .remove_edge_property_buffered(id, key, transaction_id);
        self.log_lpg(
            transaction_id,
            LpgMutationOp::RemoveEdgeProperty {
                id,
                key: key.to_string(),
            },
        );
    }

    fn add_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        self.inner
            .add_label_buffered(node_id, label, transaction_id);
        // Prepared commit logs the exact final images. Logging each buffered
        // intent would invent intermediate committed label-history entries.
    }

    fn remove_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        self.inner
            .remove_label_buffered(node_id, label, transaction_id);
    }

    // Overlay lifecycle (apply/drop) is delegated to the inner `LpgStore`: the
    // property mutations were logged at buffer time; exact label images are
    // logged by prepared commit. These have no WAL side effects and must reach
    // the inner store so the delta is committed or discarded.

    fn apply_tx_overlay(&self, transaction_id: TransactionId) {
        self.inner.apply_tx_overlay(transaction_id);
    }

    fn drop_tx_overlay(&self, transaction_id: TransactionId) {
        self.inner.drop_tx_overlay(transaction_id);
    }

    fn finalize_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        node_ids: &[NodeId],
    ) {
        self.inner
            .finalize_deletes_by_id(transaction_id, commit_epoch, node_ids);
    }

    fn take_pending_deletes(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.inner.take_pending_deletes(transaction_id)
    }

    fn finalize_edge_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        edges: &[(NodeId, EdgeId, NodeId)],
    ) {
        self.inner
            .finalize_edge_deletes_by_id(transaction_id, commit_epoch, edges);
    }

    fn take_pending_edge_deletes(
        &self,
        transaction_id: TransactionId,
    ) -> Vec<(NodeId, EdgeId, NodeId)> {
        self.inner.take_pending_edge_deletes(transaction_id)
    }

    #[cfg(feature = "lpg")]
    fn tx_overlay_snapshot(
        &self,
        transaction_id: TransactionId,
    ) -> grafeo_core::graph::lpg::TxDelta {
        self.inner.tx_overlay_snapshot(transaction_id)
    }

    #[cfg(feature = "lpg")]
    fn tx_overlay_restore(
        &self,
        transaction_id: TransactionId,
        snapshot: grafeo_core::graph::lpg::TxDelta,
    ) {
        self.inner.tx_overlay_restore(transaction_id, snapshot);
    }

    fn tx_structural_snapshot(&self, transaction_id: TransactionId) -> TxStructuralSnapshot {
        self.inner.tx_structural_snapshot(transaction_id)
    }

    fn tx_structural_restore(
        &self,
        transaction_id: TransactionId,
        snapshot: TxStructuralSnapshot,
    ) -> std::result::Result<(), String> {
        self.inner.tx_structural_restore(transaction_id, snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_storage::wal::{TypedWal, WalRecovery};

    #[test]
    fn index_preparation_forwards_epoch_and_transaction_without_wal_writes()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(LpgStore::new()?);
        let id = store.create_node(&["Doc"]);
        store.set_node_property(id, "value", Value::Int64(1));
        let frontier = EpochId::new(5);
        store.sync_epoch(frontier);
        store.set_node_property(id, "value", Value::Int64(2));
        let own = TransactionId::new(11);
        let foreign = TransactionId::new(12);
        store.set_node_property_buffered(id, "value", Value::Int64(3), own);
        store.set_node_property_buffered(id, "value", Value::Int64(4), foreign);
        let wal = Arc::new(TypedWal::open(dir.path().join("wal"))?);
        let wrapper =
            WalGraphStore::new(store, Arc::clone(&wal), GraphPath::from_components(&[""])?);
        for (epoch, transaction, expected) in [
            (EpochId::INITIAL, None, 1),
            (frontier, None, 2),
            (frontier, Some(own), 3),
        ] {
            let rows = wrapper.prepare_index_node_rows(epoch, transaction)?;
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].id, id);
            assert_eq!(rows[0].get_property("value"), Some(&Value::Int64(expected)));
        }
        assert!(matches!(
            wrapper.prepare_index_node_rows(EpochId::PENDING, None),
            Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(_)
            ))
        ));
        assert_eq!(wal.record_count(), 0);
        assert!(!wal.is_poisoned());
        Ok(())
    }

    #[test]
    fn compact_edge_creation_rejection_emits_no_wal_operation() {
        for versioned in [false, true] {
            let (_dir, writer, wal) = setup();
            let src = writer.create_node(&["Existing"]);
            let before = wal.record_count();
            assert_eq!(before, 1, "the accepted node must reach a healthy WAL");
            assert!(!wal.is_poisoned());
            let edge = if versioned {
                writer.create_edge_versioned(
                    src,
                    NodeId::new(999),
                    "MISSING",
                    EpochId::INITIAL,
                    TransactionId::SYSTEM,
                )
            } else {
                writer.create_edge(src, NodeId::new(999), "MISSING")
            };
            assert!(
                !edge.is_valid(),
                "real inner store rejects missing endpoint"
            );
            assert_eq!(writer.edge_count(), 0);
            assert_eq!(
                wal.record_count(),
                before,
                "rejected ID must never enter WAL"
            );
            assert!(!wal.is_poisoned());
        }
    }

    fn setup() -> (tempfile::TempDir, WalGraphStore, Arc<LpgWal>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal = Arc::new(TypedWal::open(dir.path().join("wal")).unwrap());
        let wal_ref = Arc::clone(&wal);
        (
            dir,
            WalGraphStore::new(store, wal, GraphPath::root()),
            wal_ref,
        )
    }

    #[test]
    fn create_node_delegates_and_logs() {
        let (_dir, ws, wal) = setup();
        let id = ws.create_node(&["Person", "Employee"]);

        assert!(ws.get_node(id).is_some());
        assert_eq!(ws.node_count(), 1);
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn create_edge_delegates_and_logs() {
        let (_dir, ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "KNOWS");

        assert!(ws.get_edge(eid).is_some());
        assert_eq!(ws.edge_count(), 1);
        // 2 CreateNode + 1 CreateEdge
        assert_eq!(wal.record_count(), 3);
    }

    #[test]
    fn set_property_delegates_and_logs() {
        let (_dir, ws, wal) = setup();
        let nid = ws.create_node(&["Person"]);
        ws.set_node_property(nid, "name", Value::String("Alix".into()));

        assert_eq!(
            ws.get_node_property(nid, &PropertyKey::from("name")),
            Some(Value::String("Alix".into()))
        );
        // CreateNode + SetNodeProperty
        assert_eq!(wal.record_count(), 2);

        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "LINK");
        ws.set_edge_property(eid, "weight", Value::Int64(42));

        assert_eq!(
            ws.get_edge_property(eid, &PropertyKey::from("weight")),
            Some(Value::Int64(42))
        );
        // +2 CreateNode + 1 CreateEdge + 1 SetEdgeProperty = 6 total
        assert_eq!(wal.record_count(), 6);
    }

    #[test]
    fn delete_node_only_logs_on_success() {
        let (_dir, ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 1);

        // Delete nonexistent: no new record
        assert!(!ws.delete_node(NodeId::new(999)));
        assert_eq!(wal.record_count(), 1);

        // Delete real node: logs
        assert!(ws.delete_node(id));
        assert_eq!(wal.record_count(), 2);
        assert!(ws.get_node(id).is_none());
    }

    #[test]
    fn delete_edge_only_logs_on_success() {
        let (_dir, ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "LINK");
        assert_eq!(wal.record_count(), 3);

        // Delete nonexistent: no new record
        assert!(!ws.delete_edge(EdgeId::new(999)));
        assert_eq!(wal.record_count(), 3);

        // Delete real edge: logs
        assert!(ws.delete_edge(eid));
        assert_eq!(wal.record_count(), 4);
        assert!(ws.get_edge(eid).is_none());
    }

    #[test]
    fn remove_property_only_logs_on_success() {
        let (_dir, ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        ws.set_node_property(id, "age", Value::Int64(30));
        assert_eq!(wal.record_count(), 2);

        // Remove nonexistent: no log
        assert!(ws.remove_node_property(id, "missing").is_none());
        assert_eq!(wal.record_count(), 2);

        // Remove real property: logs
        assert_eq!(ws.remove_node_property(id, "age"), Some(Value::Int64(30)));
        assert_eq!(wal.record_count(), 3);

        // Edge property variant
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge(a, b, "X");
        ws.set_edge_property(eid, "w", Value::Int64(1));
        let before = wal.record_count();

        assert!(ws.remove_edge_property(eid, "missing").is_none());
        assert_eq!(wal.record_count(), before);

        assert_eq!(ws.remove_edge_property(eid, "w"), Some(Value::Int64(1)));
        assert_eq!(wal.record_count(), before + 1);
    }

    #[test]
    fn add_remove_label_conditional_logging() {
        let (_dir, ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 1);

        // Add duplicate label: no log
        assert!(!ws.add_label(id, "Person"));
        assert_eq!(wal.record_count(), 1);

        // Add new label: logs
        assert!(ws.add_label(id, "Employee"));
        assert_eq!(wal.record_count(), 2);

        // Remove label: logs
        assert!(ws.remove_label(id, "Employee"));
        assert_eq!(wal.record_count(), 3);

        // Remove absent label: no log
        assert!(!ws.remove_label(id, "Employee"));
        assert_eq!(wal.record_count(), 3);
    }

    #[test]
    fn batch_create_edges_logs_each() {
        let (_dir, ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let c = ws.create_node(&["Node"]);
        assert_eq!(wal.record_count(), 3);

        let eids = ws.batch_create_edges(&[(a, b, "X"), (b, c, "Y")]);
        assert_eq!(eids.len(), 2);
        assert_eq!(ws.edge_count(), 2);
        // One WAL record per edge
        assert_eq!(wal.record_count(), 5);
    }

    #[test]
    fn delete_node_edges_logs_each_edge() {
        let (_dir, ws, wal) = setup();
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let c = ws.create_node(&["Node"]);
        ws.create_edge(a, b, "X");
        ws.create_edge(c, a, "Y");
        assert_eq!(wal.record_count(), 5);

        ws.delete_node_edges(a);
        // 2 DeleteEdge records (one outgoing, one incoming)
        assert_eq!(wal.record_count(), 7);
        assert_eq!(ws.edge_count(), 0);
    }

    fn setup_named_graph(name: &str) -> (tempfile::TempDir, WalGraphStore, Arc<LpgWal>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let wal = Arc::new(TypedWal::open(dir.path().join("wal")).unwrap());
        let wal_ref = Arc::clone(&wal);
        (
            dir,
            WalGraphStore::new(store, wal, GraphPath::from_components(&[name]).unwrap()),
            wal_ref,
        )
    }

    #[test]
    fn named_graph_emits_graph_tagged_mutation() {
        let (_dir, ws, wal) = setup_named_graph("social");
        let _id = ws.create_node(&["Person"]);

        assert_eq!(wal.record_count(), 1, "graph identity is on the mutation");
    }

    #[test]
    fn named_graph_each_mutation_is_self_describing() {
        let (_dir, ws, wal) = setup_named_graph("social");
        ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 1);

        ws.create_node(&["Person"]);
        assert_eq!(wal.record_count(), 2);
    }

    #[test]
    fn empty_named_graph_writer_preserves_empty_component() {
        let (_dir, ws, wal) = setup_named_graph("");
        ws.create_node(&["EmptyNamed"]);
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::SYSTEM,
        })
        .unwrap();
        wal.sync().unwrap();

        wal.close().unwrap();
        let records = WalRecovery::new(wal.dir()).unwrap().recover().unwrap();
        assert!(records.iter().any(|record| matches!(
            record,
            WalRecord::LpgMutation {
                graph,
                op: LpgMutationOp::CreateNode { labels, .. },
                ..
            } if graph.components() == [""] && labels == &["EmptyNamed"]
        )));
        assert!(!records.iter().any(|record| matches!(
            record,
            WalRecord::LpgMutation { graph, .. } if graph.components().is_empty()
        )));
    }

    #[test]
    fn default_graph_writer_preserves_root_coordinate() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, ws, wal) = setup();
        let src = ws.create_node(&["Default"]);
        let dst = ws.create_node(&["Destination"]);
        ws.set_node_property(src, "name", Value::String("Alix".into()));
        let edge = ws.create_edge(src, dst, "KNOWS");
        ws.set_edge_property(edge, "since", Value::Int64(2020));
        assert!(src.is_valid() && dst.is_valid() && edge.is_valid());
        assert_eq!(wal.record_count(), 5);
        assert!(!wal.is_poisoned());
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::SYSTEM,
        })?;
        wal.sync()?;

        wal.close()?;
        let records = WalRecovery::new(wal.dir())?.recover()?;
        assert_eq!(records.len(), 6, "five exact mutations and their commit");
        assert!(matches!(
            records.last(),
            Some(WalRecord::TransactionCommit { transaction_id })
                if *transaction_id == TransactionId::SYSTEM
        ));
        let operations: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::LpgMutation {
                    transaction_id,
                    graph,
                    op,
                } => {
                    assert_eq!(*transaction_id, TransactionId::SYSTEM);
                    assert_eq!(graph, &GraphPath::root());
                    Some(op)
                }
                _ => None,
            })
            .collect();
        assert!(matches!(
            operations.as_slice(),
            [
                LpgMutationOp::CreateNode { id: src_id, labels: src_labels },
                LpgMutationOp::CreateNode { id: dst_id, labels: dst_labels },
                LpgMutationOp::SetNodeProperty { id: property_node, key: node_key, value: node_value },
                LpgMutationOp::CreateEdge { id: edge_id, src: edge_src, dst: edge_dst, edge_type },
                LpgMutationOp::SetEdgeProperty { id: property_edge, key: edge_key, value: edge_value },
            ] if *src_id == src && src_labels == &["Default"]
                && *dst_id == dst && dst_labels == &["Destination"]
                && *property_node == src && node_key == "name"
                && *node_value == Value::String("Alix".into())
                && *edge_id == edge && *edge_src == src && *edge_dst == dst
                && edge_type == "KNOWS" && *property_edge == edge
                && edge_key == "since" && *edge_value == Value::Int64(2020)
        ));
        Ok(())
    }

    #[test]
    fn create_node_versioned_delegates_and_logs() {
        let (_dir, ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let id = ws.create_node_versioned(&["Person"], epoch, tx);

        assert!(id.is_valid());
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn create_edge_versioned_delegates_and_logs() {
        let (_dir, ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge_versioned(a, b, "KNOWS", epoch, tx);

        assert!(eid.is_valid());
        // 2 CreateNode + 1 CreateEdge
        assert_eq!(wal.record_count(), 3);
    }

    #[test]
    fn delete_node_versioned_only_logs_on_success() {
        let (_dir, ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let id = ws.create_node_versioned(&["Person"], epoch, tx);
        assert_eq!(wal.record_count(), 1);

        // Delete nonexistent: no log
        assert!(!ws.delete_node_versioned(NodeId::new(999), epoch, tx));
        assert_eq!(wal.record_count(), 1);

        // Delete real node: logs
        assert!(ws.delete_node_versioned(id, epoch, tx));
        assert_eq!(wal.record_count(), 2);
    }

    #[test]
    fn delete_edge_versioned_only_logs_on_success() {
        let (_dir, ws, wal) = setup();
        let epoch = ws.current_epoch();
        let tx = TransactionId::new(1);
        let a = ws.create_node(&["Node"]);
        let b = ws.create_node(&["Node"]);
        let eid = ws.create_edge_versioned(a, b, "LINK", epoch, tx);
        assert_eq!(wal.record_count(), 3);

        // Delete nonexistent: no log
        assert!(!ws.delete_edge_versioned(EdgeId::new(999), epoch, tx));
        assert_eq!(wal.record_count(), 3);

        // Delete real edge: logs
        assert!(ws.delete_edge_versioned(eid, epoch, tx));
        assert_eq!(wal.record_count(), 4);
    }

    #[test]
    fn create_node_with_props_via_trait_default() {
        use grafeo_core::graph::GraphStoreMut;

        let (_dir, ws, wal) = setup();
        let store: &dyn GraphStoreMut = &ws;
        let id = store.create_node_with_props(
            &["Person"],
            &[
                (PropertyKey::from("name"), Value::String("Alix".into())),
                (PropertyKey::from("age"), Value::Int64(30)),
            ],
        );

        assert!(ws.get_node(id).is_some());
        // 1 CreateNode + 2 SetNodeProperty
        assert_eq!(wal.record_count(), 3);

        assert_eq!(
            ws.get_node_property(id, &PropertyKey::from("name")),
            Some(Value::String("Alix".into()))
        );
        assert_eq!(
            ws.get_node_property(id, &PropertyKey::from("age")),
            Some(Value::Int64(30))
        );
    }

    #[test]
    fn create_edge_with_props_via_trait_default() {
        use grafeo_core::graph::GraphStoreMut;

        let (_dir, ws, wal) = setup();
        let store: &dyn GraphStoreMut = &ws;
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let eid = store.create_edge_with_props(
            a,
            b,
            "KNOWS",
            &[(PropertyKey::from("since"), Value::Int64(2020))],
        );

        assert!(ws.get_edge(eid).is_some());
        // 2 CreateNode + 1 CreateEdge + 1 SetEdgeProperty
        assert_eq!(wal.record_count(), 4);

        assert_eq!(
            ws.get_edge_property(eid, &PropertyKey::from("since")),
            Some(Value::Int64(2020))
        );
    }

    #[test]
    fn read_operations_do_not_log() {
        let (_dir, ws, wal) = setup();
        let id = ws.create_node(&["Person"]);
        ws.set_node_property(id, "name", Value::String("Alix".into()));
        assert_eq!(wal.record_count(), 2);

        // Exercise read-only methods
        let _ = ws.get_node(id);
        let _ = ws.node_count();
        let _ = ws.node_ids();
        let _ = ws.nodes_by_label("Person");
        let _ = ws.get_node_property(id, &PropertyKey::from("name"));
        let _ = ws.neighbors(id, Direction::Outgoing);
        let _ = ws.edge_count();
        let _ = ws.out_degree(id);
        let _ = ws.in_degree(id);
        let _ = ws.has_backward_adjacency();
        let _ = ws.statistics();

        // No additional records
        assert_eq!(wal.record_count(), 2);
    }
}
