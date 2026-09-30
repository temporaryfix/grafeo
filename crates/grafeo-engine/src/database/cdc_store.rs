//! CDC-aware graph store wrapper.
//!
//! Wraps a [`GraphStoreMut`] and buffers CDC events for every mutation.
//! Events are held in the session's transaction-owned change accumulator,
//! which publishes to [`CdcLog`] on commit or discards on rollback.
//!
//! This mirrors the [`WalGraphStore`](super::wal_store::WalGraphStore)
//! decorator pattern but targets the CDC audit trail instead of WAL
//! durability.

use std::collections::HashMap;
use std::sync::Arc;

use crate::cdc::{CdcLog, ChangeEvent, ChangeKind, EntityId, TransactionChangeAccumulator};
use arcstr::ArcStr;
use grafeo_common::types::{
    EdgeId, EpochId, GraphPath, HlcTimestamp, NodeId, PropertyKey, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::execution::operators::{SharedReadTracker, SharedWriteTracker};
use grafeo_core::graph::lpg::{CompareOp, Edge, LpgStore, Node};
use grafeo_core::graph::{
    Direction, GraphStore, GraphStoreMut, GraphStoreSearch, PropertyIndexRequest,
    TxStructuralSnapshot,
};
use grafeo_core::statistics::Statistics;

/// A [`GraphStoreMut`] decorator that buffers CDC events for every mutation.
///
/// Read-only methods are forwarded to the inner store without CDC interaction.
///
/// Versioned (transactional) mutations stage events into the shared
/// accumulator. The owning session publishes it to `CdcLog` on commit or
/// clears it on rollback.
///
/// Non-versioned internal mutations record directly to `CdcLog`
/// since they have no transaction context and are immediately visible. Public
/// database CRUD is routed through Session framing by this tranche.
pub(crate) struct CdcGraphStore {
    inner: Arc<dyn GraphStoreMut>,
    /// Transaction-owned event staging boundary shared across graph wrappers.
    /// It also seals the one clock/publication target used by direct writes.
    pending_events: Arc<TransactionChangeAccumulator>,
    /// Exact store incarnation that accepts this wrapper's LPG mutations.
    graph_incarnation: Arc<LpgStore>,
    /// Exact component-qualified LPG coordinate; an empty path is root.
    graph: GraphPath,
}

impl CdcGraphStore {
    /// Creates a new CDC-aware store with a fresh event buffer.
    pub fn new(
        inner: Arc<dyn GraphStoreMut>,
        cdc_log: Arc<CdcLog>,
        graph_incarnation: Arc<LpgStore>,
        graph: GraphPath,
    ) -> Self {
        let pending_events = Arc::new(TransactionChangeAccumulator::new(&cdc_log));
        Self {
            inner,
            pending_events,
            graph_incarnation,
            graph,
        }
    }

    /// Wraps a store sharing an existing event buffer.
    ///
    /// Used for named graphs so all mutations in a transaction (across
    /// default and named graphs) stage into the same accumulator for atomic
    /// publication/discard.
    pub fn wrap(
        inner: Arc<dyn GraphStoreMut>,
        pending_events: Arc<TransactionChangeAccumulator>,
        graph_incarnation: Arc<LpgStore>,
        graph: GraphPath,
    ) -> Self {
        Self {
            inner,
            pending_events,
            graph_incarnation,
            graph,
        }
    }

    /// Returns the transaction-owned change accumulator.
    pub fn pending_events(&self) -> Arc<TransactionChangeAccumulator> {
        Arc::clone(&self.pending_events)
    }

    /// Constructs one edge through this wrapper's admitted inner writer, then
    /// stages its complete Create only after the checked construction succeeds.
    /// Ordinary SET operations still use the independent Update methods below.
    pub(crate) fn create_edge_compound(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(&str, Value)],
        construct: impl FnOnce(&dyn GraphStoreMut) -> grafeo_common::utils::error::Result<EdgeId>,
    ) -> grafeo_common::utils::error::Result<EdgeId> {
        let id = construct(self.inner.as_ref())?;
        if !id.is_valid() {
            return Err(grafeo_common::utils::error::Error::Internal(
                "compound edge construction returned an invalid identity".to_string(),
            ));
        }
        let mut event = make_event(EntityId::Edge(id), ChangeKind::Create, EpochId::PENDING);
        event.src_id = Some(src.as_u64());
        event.dst_id = Some(dst.as_u64());
        event.edge_type = Some(edge_type.to_string());
        if !properties.is_empty() {
            event.after = Some(
                properties
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), value.clone()))
                    .collect(),
            );
        }
        self.buffer_event(event);
        Ok(id)
    }

    /// Buffers a CDC event for later flush on commit.
    ///
    /// The epoch is always set to `PENDING`: the real commit epoch is assigned
    /// when the session flushes the buffer in `commit_inner()`. This ensures
    /// each transaction's events get the unique epoch from `fetch_add(1, SeqCst)`.
    fn buffer_event(&self, event: ChangeEvent) {
        self.pending_events
            .stage_lpg(event, &self.graph, Arc::clone(&self.graph_incarnation));
    }

    /// Records a CDC event directly for non-versioned internal mutations.
    fn record_directly(&self, event: ChangeEvent) {
        self.pending_events.record_direct(event, &self.graph);
    }

    /// Collects all properties of a node as a `HashMap` for before/after snapshots.
    fn collect_node_properties(&self, id: NodeId) -> Option<HashMap<String, Value>> {
        let node = self.inner.get_node(id)?;
        let map: HashMap<String, Value> = node
            .properties
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.clone()))
            .collect();
        if map.is_empty() { None } else { Some(map) }
    }

    /// Collects all properties of an edge as a `HashMap` for before/after snapshots.
    fn collect_edge_properties(&self, id: EdgeId) -> Option<HashMap<String, Value>> {
        let edge = self.inner.get_edge(id)?;
        let map: HashMap<String, Value> = edge
            .properties
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.clone()))
            .collect();
        if map.is_empty() { None } else { Some(map) }
    }

    /// Collects labels for a node.
    fn collect_node_labels(&self, id: NodeId) -> Option<Vec<String>> {
        let node = self.inner.get_node(id)?;
        Some(node.labels.iter().map(|l| l.to_string()).collect())
    }

    /// Collects properties from the writing transaction's visible view.
    ///
    /// Transactional CDC must not use the committed convenience getters: a
    /// second mutation in one transaction observes the first buffered write.
    fn collect_node_properties_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<HashMap<String, Value>> {
        let properties = self
            .inner
            .read_node_properties_visible(id, epoch, Some(transaction_id))
            .into_iter()
            .map(|(key, value)| (key.as_str().to_string(), value))
            .collect::<HashMap<_, _>>();
        if properties.is_empty() {
            None
        } else {
            Some(properties)
        }
    }

    /// Edge counterpart to [`Self::collect_node_properties_visible`].
    fn collect_edge_properties_visible(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<HashMap<String, Value>> {
        let properties = self
            .inner
            .read_edge_properties_visible(id, epoch, Some(transaction_id))
            .into_iter()
            .map(|(key, value)| (key.as_str().to_string(), value))
            .collect::<HashMap<_, _>>();
        if properties.is_empty() {
            None
        } else {
            Some(properties)
        }
    }

    /// Collects labels from the writing transaction's visible view.
    fn collect_node_labels_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<String> {
        self.inner
            .read_node_labels_visible(id, epoch, Some(transaction_id))
            .into_iter()
            .map(|label| label.to_string())
            .collect()
    }
}

fn make_event(entity_id: EntityId, kind: ChangeKind, epoch: EpochId) -> ChangeEvent {
    ChangeEvent {
        graph_incarnation: None,
        entity_id,
        kind,
        epoch,
        // The shared accumulator mints the one final timestamp while it fixes
        // transactional staging order. Direct compatibility recording stamps
        // this sentinel immediately before publication.
        timestamp: HlcTimestamp::zero(),
        before: None,
        after: None,
        labels: None,
        edge_type: None,
        src_id: None,
        dst_id: None,
        triple_subject: None,
        triple_predicate: None,
        triple_object: None,
        lpg_graph: Some(GraphPath::root()),
        triple_graph: None,
    }
}

// ---------------------------------------------------------------------------
// GraphStore (read-only): pure delegation
// ---------------------------------------------------------------------------

impl GraphStore for CdcGraphStore {
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
        self.inner.neighbors(node, direction)
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
        self.inner.edges_from(node, direction)
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
    // Reads have no CDC event or log side effects, so we can safely delegate
    // to the inner store's snapshot-aware accessors. The per-transaction
    // property delta lives in the inner LpgStore (or another LpgStore beneath
    // the CDC wrapper), so these delegates route through the real delta.

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
    ) -> FxHashMap<PropertyKey, Value> {
        self.inner
            .read_node_properties_visible(id, epoch, transaction_id)
    }

    fn read_edge_properties_visible(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashMap<PropertyKey, Value> {
        self.inner
            .read_edge_properties_visible(id, epoch, transaction_id)
    }

    // --- Task 5 (label reads, unified-MVCC) ---
    //
    // Label reads have no CDC event or log side effects, so delegate to the
    // inner store's snapshot-aware label accessors. The per-transaction label
    // delta lives in the inner LpgStore (or another LpgStore beneath the CDC
    // wrapper), so these delegates route through the real delta.
    //
    // `add_label_buffered` / `remove_label_buffered` ARE overridden (in the
    // transactional buffered-write block below) to delegate to the inner store's
    // overlay and buffer the CDC event — same pattern as the property
    // `*_buffered` overrides — giving CDC-wrapped stores transactional label
    // isolation while still recording the change at commit.

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

// Pure delegation: CDC wraps the store to buffer mutation events but owns no
// index state, so every text/vector lookup has to fall through to the
// underlying `GraphStoreMut` (which is a `GraphStoreSearch` by bound). A stub
// impl silently turns into "no index exists" at every call site — the same
// regression that hit the WAL wrapper in issue #308.
impl GraphStoreSearch for CdcGraphStore {
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
// GraphStoreMut: delegate + CDC buffer/record
// ---------------------------------------------------------------------------

impl GraphStoreMut for CdcGraphStore {
    fn lpg_commit_store(self: Arc<Self>) -> Option<Arc<LpgStore>> {
        Arc::clone(&self.inner).lpg_commit_store()
    }

    // --- Node creation ---

    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.inner.create_node(labels);
        let epoch = self.inner.current_epoch();
        let mut event = make_event(EntityId::Node(id), ChangeKind::Create, epoch);
        event.labels = Some(labels.iter().map(|s| (*s).to_string()).collect());
        self.record_directly(event);
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
        // Use PENDING epoch: the real commit epoch is assigned during flush.
        let mut event = make_event(EntityId::Node(id), ChangeKind::Create, EpochId::PENDING);
        event.labels = Some(labels.iter().map(|s| (*s).to_string()).collect());
        self.buffer_event(event);
        id
    }

    // --- Edge creation ---

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let id = self.inner.create_edge(src, dst, edge_type);
        if !id.is_valid() {
            return id;
        }
        let epoch = self.inner.current_epoch();
        let mut event = make_event(EntityId::Edge(id), ChangeKind::Create, epoch);
        event.edge_type = Some(edge_type.to_string());
        event.src_id = Some(src.as_u64());
        event.dst_id = Some(dst.as_u64());
        self.record_directly(event);
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
        let mut event = make_event(EntityId::Edge(id), ChangeKind::Create, epoch);
        event.edge_type = Some(edge_type.to_string());
        event.src_id = Some(src.as_u64());
        event.dst_id = Some(dst.as_u64());
        self.buffer_event(event);
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let ids = self.inner.batch_create_edges(edges);
        let epoch = self.inner.current_epoch();
        for (id, (src, dst, edge_type)) in ids.iter().zip(edges) {
            let mut event = make_event(EntityId::Edge(*id), ChangeKind::Create, epoch);
            event.edge_type = Some((*edge_type).to_string());
            event.src_id = Some(src.as_u64());
            event.dst_id = Some(dst.as_u64());
            self.record_directly(event);
        }
        ids
    }

    // --- Deletion ---

    fn delete_node(&self, id: NodeId) -> bool {
        let before_props = self.collect_node_properties(id);
        let deleted = self.inner.delete_node(id);
        if deleted {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(EntityId::Node(id), ChangeKind::Delete, epoch);
            event.before = before_props;
            self.record_directly(event);
        }
        deleted
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let before_props = self.collect_node_properties_visible(id, epoch, transaction_id);
        let labels = self.collect_node_labels_visible(id, epoch, transaction_id);
        let deleted = self.inner.delete_node_versioned(id, epoch, transaction_id);
        if deleted {
            let mut event = make_event(EntityId::Node(id), ChangeKind::Delete, epoch);
            event.before = before_props;
            event.labels = (!labels.is_empty()).then_some(labels);
            self.buffer_event(event);
        }
        deleted
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Collect edge info before deletion
        let outgoing: Vec<(NodeId, EdgeId)> = self.inner.edges_from(node_id, Direction::Outgoing);
        let incoming: Vec<(NodeId, EdgeId)> = self.inner.edges_from(node_id, Direction::Incoming);

        let edge_infos: Vec<(EdgeId, Option<HashMap<String, Value>>)> = outgoing
            .iter()
            .chain(incoming.iter())
            .map(|(_, eid)| (*eid, self.collect_edge_properties(*eid)))
            .collect();

        self.inner.delete_node_edges(node_id);

        let epoch = self.inner.current_epoch();
        for (eid, props) in edge_infos {
            let mut event = make_event(EntityId::Edge(eid), ChangeKind::Delete, epoch);
            event.before = props;
            self.record_directly(event);
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        let before_props = self.collect_edge_properties(id);
        let deleted = self.inner.delete_edge(id);
        if deleted {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(EntityId::Edge(id), ChangeKind::Delete, epoch);
            event.before = before_props;
            self.record_directly(event);
        }
        deleted
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let before_props = self.collect_edge_properties_visible(id, epoch, transaction_id);
        let deleted = self.inner.delete_edge_versioned(id, epoch, transaction_id);
        if deleted {
            let mut event = make_event(EntityId::Edge(id), ChangeKind::Delete, epoch);
            event.before = before_props;
            self.buffer_event(event);
        }
        deleted
    }

    // --- Property mutation ---

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let old_value = self.inner.get_node_property(id, &PropertyKey::new(key));
        self.inner.set_node_property(id, key, value.clone());
        let epoch = self.inner.current_epoch();
        let mut event = make_event(EntityId::Node(id), ChangeKind::Update, epoch);
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.record_directly(event);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        let old_value = self.inner.get_edge_property(id, &PropertyKey::new(key));
        self.inner.set_edge_property(id, key, value.clone());
        let epoch = self.inner.current_epoch();
        let mut event = make_event(EntityId::Edge(id), ChangeKind::Update, epoch);
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.record_directly(event);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_node_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        self.inner
            .set_node_property_versioned(id, key, value.clone(), transaction_id);
        let mut event = make_event(EntityId::Node(id), ChangeKind::Update, epoch);
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.buffer_event(event);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_edge_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        self.inner
            .set_edge_property_versioned(id, key, value.clone(), transaction_id);
        let mut event = make_event(EntityId::Edge(id), ChangeKind::Update, epoch);
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.buffer_event(event);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_node_property(id, key);
        if let Some(ref old_val) = removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(EntityId::Node(id), ChangeKind::Update, epoch);
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val.clone());
            event.before = Some(before);
            self.record_directly(event);
        }
        removed
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_edge_property(id, key);
        if let Some(ref old_val) = removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(EntityId::Edge(id), ChangeKind::Update, epoch);
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val.clone());
            event.before = Some(before);
            self.record_directly(event);
        }
        removed
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_node_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        let removed = self
            .inner
            .remove_node_property_versioned(id, key, transaction_id);
        if let Some(removed_value) = removed.as_ref() {
            let mut event = make_event(EntityId::Node(id), ChangeKind::Update, epoch);
            let mut before = HashMap::new();
            before.insert(
                key.to_string(),
                old_value.unwrap_or_else(|| removed_value.clone()),
            );
            event.before = Some(before);
            self.buffer_event(event);
        }
        removed
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_edge_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        let removed = self
            .inner
            .remove_edge_property_versioned(id, key, transaction_id);
        if let Some(removed_value) = removed.as_ref() {
            let mut event = make_event(EntityId::Edge(id), ChangeKind::Update, epoch);
            let mut before = HashMap::new();
            before.insert(
                key.to_string(),
                old_value.unwrap_or_else(|| removed_value.clone()),
            );
            event.before = Some(before);
            self.buffer_event(event);
        }
        removed
    }

    // --- Label mutation ---

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let added = self.inner.add_label(node_id, label);
        if added {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(EntityId::Node(node_id), ChangeKind::Update, epoch);
            event.labels = self.collect_node_labels(node_id);
            self.record_directly(event);
        }
        added
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let old_labels = self.collect_node_labels(node_id);
        let removed = self.inner.remove_label(node_id, label);
        if removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(EntityId::Node(node_id), ChangeKind::Update, epoch);
            event.labels = old_labels;
            self.record_directly(event);
        }
        removed
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let epoch = self.inner.current_epoch();
        let added = self
            .inner
            .add_label_versioned(node_id, label, transaction_id);
        if added {
            let mut event = make_event(EntityId::Node(node_id), ChangeKind::Update, epoch);
            event.labels = Some(self.collect_node_labels_visible(node_id, epoch, transaction_id));
            self.buffer_event(event);
        }
        added
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let epoch = self.inner.current_epoch();
        let old_labels = self.collect_node_labels_visible(node_id, epoch, transaction_id);
        let removed = self
            .inner
            .remove_label_versioned(node_id, label, transaction_id);
        if removed {
            let mut event = make_event(EntityId::Node(node_id), ChangeKind::Update, epoch);
            event.labels = Some(old_labels);
            self.buffer_event(event);
        }
        removed
    }

    // --- Transactional buffered writes (unified-MVCC) ---
    //
    // Override `*_buffered` to delegate to the inner store's buffered path
    // (overlay delta) instead of the trait-default write-through. This gives
    // CDC-wrapped persistent stores the same uncommitted-write isolation as a
    // bare `LpgStore`: other sessions no longer observe a transaction's
    // uncommitted Cypher writes (see `tests/cdc_mvcc_isolation.rs`).
    //
    // The CDC `ChangeEvent` is still produced exactly as the `*_versioned`
    // overrides do and `buffer_event`-ed into the per-tx pending buffer, so
    // events are NOT lost: the session flushes them to the committed `CdcLog`
    // at commit (`session::commit_inner`) and drains them on rollback. Emission
    // stays at session-commit, NOT the wrapper's `apply_tx_overlay` — commit
    // promotes the overlay via `resolve_store()` -> the raw `LpgStore`, so the
    // wrapper's `apply_tx_overlay` never fires.

    fn set_node_property_buffered(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_node_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        self.inner
            .set_node_property_buffered(id, key, value.clone(), transaction_id);
        let mut event = make_event(EntityId::Node(id), ChangeKind::Update, epoch);
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.buffer_event(event);
    }

    fn remove_node_property_buffered(&self, id: NodeId, key: &str, transaction_id: TransactionId) {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_node_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        self.inner
            .remove_node_property_buffered(id, key, transaction_id);
        if let Some(old_val) = old_value {
            let mut event = make_event(EntityId::Node(id), ChangeKind::Update, epoch);
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val);
            event.before = Some(before);
            self.buffer_event(event);
        }
    }

    fn set_edge_property_buffered(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_edge_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        self.inner
            .set_edge_property_buffered(id, key, value.clone(), transaction_id);
        let mut event = make_event(EntityId::Edge(id), ChangeKind::Update, epoch);
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.buffer_event(event);
    }

    fn remove_edge_property_buffered(&self, id: EdgeId, key: &str, transaction_id: TransactionId) {
        let epoch = self.inner.current_epoch();
        let old_value = self.inner.read_edge_property_visible(
            id,
            &PropertyKey::new(key),
            epoch,
            Some(transaction_id),
        );
        self.inner
            .remove_edge_property_buffered(id, key, transaction_id);
        if let Some(old_val) = old_value {
            let mut event = make_event(EntityId::Edge(id), ChangeKind::Update, epoch);
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val);
            event.before = Some(before);
            self.buffer_event(event);
        }
    }

    fn add_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        let epoch = self.inner.current_epoch();
        let labels_before = self.collect_node_labels_visible(node_id, epoch, transaction_id);
        let already_present = labels_before.iter().any(|existing| existing == label);
        self.inner
            .add_label_buffered(node_id, label, transaction_id);
        if !already_present {
            let mut event = make_event(EntityId::Node(node_id), ChangeKind::Update, epoch);
            event.labels = Some(self.collect_node_labels_visible(node_id, epoch, transaction_id));
            self.buffer_event(event);
        }
    }

    fn remove_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        let epoch = self.inner.current_epoch();
        let old_labels = self.collect_node_labels_visible(node_id, epoch, transaction_id);
        let had = old_labels.iter().any(|existing| existing == label);
        self.inner
            .remove_label_buffered(node_id, label, transaction_id);
        if had {
            let mut event = make_event(EntityId::Node(node_id), ChangeKind::Update, epoch);
            event.labels = Some(old_labels);
            self.buffer_event(event);
        }
    }

    // Overlay lifecycle (apply/drop) is delegated to the inner store; the CDC
    // events were buffered at write time above, so these have no CDC side
    // effects and must reach the inner store so the delta is committed/dropped.

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
    use grafeo_core::graph::lpg::LpgStore;

    #[test]
    fn wrappers_and_survival_filter_keep_exact_component_paths()
    -> Result<(), Box<dyn std::error::Error>> {
        let log = Arc::new(CdcLog::new());
        let pending = Arc::new(TransactionChangeAccumulator::new(&log));
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["default"])?,
            GraphPath::from_components(&["a/b"])?,
            GraphPath::from_components(&["a", "b"])?,
        ];
        let mut stores = Vec::new();
        for path in &paths {
            let store = Arc::new(LpgStore::new()?);
            let wrapper = CdcGraphStore::wrap(
                Arc::clone(&store) as Arc<dyn GraphStoreMut>,
                Arc::clone(&pending),
                Arc::clone(&store),
                path.clone(),
            );
            let direct = wrapper.create_node(&["Direct"]);
            let direct_events = log.history_in_graph(EntityId::Node(direct), path);
            assert_eq!(direct_events.len(), 1);
            assert_eq!(direct_events[0].graph_path(), Some(path));
            assert!(direct_events[0].triple_graph.is_none());
            let staged =
                wrapper.create_node_versioned(&["Staged"], EpochId::INITIAL, TransactionId::new(9));
            stores.push((path.clone(), store, staged));
        }
        let retired = GraphPath::from_components(&["a/b"])?;
        pending
            .prepare_committed_lpg(EpochId::new(12), |path, incarnation| {
                path != &retired
                    && stores.iter().any(|(expected, store, _)| {
                        expected == path && Arc::ptr_eq(store, incarnation)
                    })
            })
            .unwrap()
            .prepare_publication()
            .unwrap()
            .publish();
        for (path, _, node) in stores {
            let events = log.history_in_graph(EntityId::Node(node), &path);
            assert_eq!(events.len(), usize::from(path != retired));
            assert!(events.iter().all(|event| event.epoch == EpochId::new(12)));
        }
        assert!(pending.lock().is_empty());
        Ok(())
    }

    #[test]
    fn index_preparation_forwards_epoch_and_transaction_without_cdc_events()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(LpgStore::new()?);
        let id = store.create_node(&["Doc"]);
        store.set_node_property(id, "value", Value::Int64(1));
        let frontier = EpochId::new(5);
        store.sync_epoch(frontier);
        store.set_node_property(id, "value", Value::Int64(2));
        let own = TransactionId::new(11);
        store.set_node_property_buffered(id, "value", Value::Int64(3), own);
        store.set_node_property_buffered(id, "value", Value::Int64(4), TransactionId::new(12));
        let log = Arc::new(CdcLog::new());
        let pending = Arc::new(TransactionChangeAccumulator::new(&log));
        let wrapper = CdcGraphStore::wrap(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Arc::clone(&pending),
            store,
            GraphPath::from_components(&[""])?,
        );
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
            wrapper.prepare_index_node_rows(frontier, Some(TransactionId::SYSTEM)),
            Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(_)
            ))
        ));
        assert!(pending.lock().is_empty());
        assert!(
            log.history_in_graph(EntityId::Node(id), &GraphPath::from_components(&[""])?)
                .is_empty()
        );
        Ok(())
    }

    #[cfg(all(feature = "wal", feature = "compact-store"))]
    #[test]
    fn index_preparation_crosses_cdc_and_wal_without_losing_cold_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        use grafeo_core::graph::compact::builder::from_graph_store_preserving_ids;
        use grafeo_core::graph::compact::layered::LayeredStore;
        use grafeo_storage::wal::TypedWal;

        let source = LpgStore::new()?;
        let cold = source.create_node(&["Doc"]);
        source.set_node_property(cold, "value", Value::Int64(1));
        let base = from_graph_store_preserving_ids(&source)?;
        let layered = Arc::new(LayeredStore::new(base, cold.as_u64(), 0)?);
        let overlay = layered.overlay_store();
        let own = TransactionId::new(21);
        let pending_node = overlay.create_node_versioned(&["Doc"], EpochId::INITIAL, own);
        overlay.set_node_property_buffered(pending_node, "value", Value::Int64(2), own);
        let dir = tempfile::tempdir()?;
        let wal = Arc::new(TypedWal::open(dir.path().join("wal"))?);
        let wal_store = Arc::new(super::super::wal_store::WalGraphStore::new(
            layered,
            Arc::clone(&wal),
            GraphPath::root(),
        ));
        let log = Arc::new(CdcLog::new());
        let wrapper = CdcGraphStore::new(wal_store, Arc::clone(&log), overlay, GraphPath::root());
        let committed = wrapper.prepare_index_node_rows(EpochId::INITIAL, None)?;
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].id, cold);
        assert_eq!(committed[0].get_property("value"), Some(&Value::Int64(1)));
        let final_rows = wrapper.prepare_index_node_rows(EpochId::INITIAL, Some(own))?;
        assert_eq!(final_rows.len(), 2);
        assert_eq!(final_rows[0].id, cold);
        assert_eq!(final_rows[1].id, pending_node);
        assert_eq!(final_rows[1].get_property("value"), Some(&Value::Int64(2)));
        assert_eq!(wal.record_count(), 0);
        assert!(wrapper.pending_events().lock().is_empty());
        assert!(log.history(EntityId::Node(cold)).is_empty());
        assert!(log.history(EntityId::Node(pending_node)).is_empty());
        Ok(())
    }

    #[test]
    fn compact_edge_creation_rejection_emits_no_cdc_operation() {
        for versioned in [false, true] {
            let (writer, log) = setup();
            let src = writer.create_node(&["Existing"]);
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
            assert!(writer.pending_events().lock().is_empty());
            assert!(
                log.history(EntityId::Edge(edge)).is_empty(),
                "rejected ID must never enter CDC"
            );
        }
    }

    /// Creates a `CdcGraphStore` wrapping a fresh `LpgStore`.
    fn setup() -> (CdcGraphStore, Arc<CdcLog>) {
        let store = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let cdc = CdcGraphStore::new(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Arc::clone(&log),
            Arc::clone(&store),
            GraphPath::root(),
        );
        (cdc, log)
    }

    // ---------------------------------------------------------------
    // Constructor and accessors
    // ---------------------------------------------------------------

    #[test]
    fn new_creates_empty_pending_buffer() {
        let (cdc, _log) = setup();
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn wrap_shares_event_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let pending = Arc::new(TransactionChangeAccumulator::new(&log));
        let cdc = CdcGraphStore::wrap(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Arc::clone(&pending),
            Arc::clone(&store),
            GraphPath::from_components(&["analytics"])?,
        );
        // Mutation through cdc should write to the shared buffer
        let id = cdc.create_node(&["Person"]);
        // create_node records directly, not into the buffer
        assert!(pending.lock().is_empty());
        // But the log should have the event
        assert_eq!(log.history(EntityId::Node(id)).len(), 1);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Read-only delegation (spot checks)
    // ---------------------------------------------------------------

    #[test]
    fn get_node_delegates_to_inner() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let node = cdc.get_node(id);
        assert!(node.is_some());
        assert!(node.unwrap().labels.iter().any(|l| l.as_str() == "Person"));
    }

    #[test]
    fn get_edge_delegates_to_inner() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        let eid = cdc.create_edge(a, b, "KNOWS");
        assert!(cdc.get_edge(eid).is_some());
    }

    #[test]
    fn node_count_and_edge_count_delegate() {
        let (cdc, _log) = setup();
        assert_eq!(cdc.node_count(), 0);
        assert_eq!(cdc.edge_count(), 0);
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        cdc.create_edge(a, b, "E");
        assert_eq!(cdc.node_count(), 2);
        assert_eq!(cdc.edge_count(), 1);
    }

    #[test]
    fn node_ids_and_all_node_ids_delegate() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["X"]);
        assert!(cdc.node_ids().contains(&a));
        assert!(cdc.all_node_ids().contains(&a));
    }

    #[test]
    fn nodes_by_label_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["City"]);
        assert!(cdc.nodes_by_label("City").contains(&a));
        assert!(cdc.nodes_by_label("Unknown").is_empty());
    }

    #[test]
    fn edge_type_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "LIKES");
        assert_eq!(&*cdc.edge_type(e).unwrap(), "LIKES");
    }

    #[test]
    fn neighbors_and_edges_from_delegate() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        cdc.create_edge(a, b, "E");
        assert!(cdc.neighbors(a, Direction::Outgoing).contains(&b));
        assert!(!cdc.edges_from(a, Direction::Outgoing).is_empty());
    }

    #[test]
    fn degree_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        cdc.create_edge(a, b, "E");
        assert_eq!(cdc.out_degree(a), 1);
        assert_eq!(cdc.in_degree(b), 1);
    }

    #[test]
    fn property_access_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        cdc.set_node_property(a, "name", Value::from("Alix"));
        assert_eq!(
            cdc.get_node_property(a, &PropertyKey::new("name")),
            Some(Value::from("Alix"))
        );
    }

    #[test]
    fn all_labels_and_edge_types_delegate() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["Person"]);
        let b = cdc.create_node(&["City"]);
        cdc.create_edge(a, b, "LIVES_IN");
        let labels = cdc.all_labels();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"City".to_string()));
        let types = cdc.all_edge_types();
        assert!(types.contains(&"LIVES_IN".to_string()));
    }

    #[test]
    fn all_property_keys_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        cdc.set_node_property(a, "colour", Value::from("orange"));
        let keys = cdc.all_property_keys();
        assert!(keys.contains(&"colour".to_string()));
    }

    #[test]
    fn statistics_delegates() {
        let (cdc, _log) = setup();
        let _stats = cdc.statistics();
    }

    #[test]
    fn current_epoch_delegates() {
        let (cdc, _log) = setup();
        let _epoch = cdc.current_epoch();
    }

    #[test]
    fn has_backward_adjacency_delegates() {
        let (cdc, _log) = setup();
        let _ = cdc.has_backward_adjacency();
    }

    #[test]
    fn has_property_index_delegates() {
        let (cdc, _log) = setup();
        assert!(!cdc.has_property_index("nonexistent"));
    }

    #[test]
    fn find_nodes_by_property_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        cdc.set_node_property(a, "x", Value::Int64(42));
        // find_nodes_by_property may or may not use indexes, just verify no panic
        let _found = cdc.find_nodes_by_property("x", &Value::Int64(42));
    }

    #[test]
    fn find_nodes_by_properties_delegates() {
        let (cdc, _log) = setup();
        let _found = cdc.find_nodes_by_properties(&[("x", Value::Int64(1))]);
    }

    #[test]
    fn find_nodes_in_range_delegates() {
        let (cdc, _log) = setup();
        let _found = cdc.find_nodes_in_range(
            "x",
            Some(&Value::Int64(0)),
            Some(&Value::Int64(100)),
            true,
            true,
        );
    }

    #[test]
    fn estimate_label_cardinality_delegates() {
        let (cdc, _log) = setup();
        let _est = cdc.estimate_label_cardinality("Person");
    }

    #[test]
    fn estimate_avg_degree_delegates() {
        let (cdc, _log) = setup();
        let _est = cdc.estimate_avg_degree("KNOWS", true);
    }

    #[test]
    fn property_might_match_delegates() {
        let (cdc, _log) = setup();
        let pk = PropertyKey::new("x");
        let _ = cdc.node_property_might_match(&pk, CompareOp::Eq, &Value::Int64(1));
        let _ = cdc.edge_property_might_match(&pk, CompareOp::Eq, &Value::Int64(1));
    }

    #[test]
    fn visibility_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "E");
        let epoch = cdc.current_epoch();
        let _ = cdc.is_node_visible_at_epoch(a, epoch);
        let _ = cdc.is_edge_visible_at_epoch(e, epoch);
        let _ = cdc.filter_visible_node_ids(&[a], epoch);
    }

    #[test]
    fn history_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "E");
        let _ = cdc.get_node_history(a);
        let _ = cdc.get_edge_history(e);
    }

    #[test]
    fn batch_property_access_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        let b = cdc.create_node(&["N"]);
        cdc.set_node_property(a, "x", Value::Int64(1));
        cdc.set_node_property(b, "x", Value::Int64(2));
        let pk = PropertyKey::new("x");
        let batch = cdc.get_node_property_batch(&[a, b], &pk);
        assert_eq!(batch.len(), 2);
        let props = cdc.get_nodes_properties_batch(&[a, b]);
        assert_eq!(props.len(), 2);
        let selective =
            cdc.get_nodes_properties_selective_batch(&[a, b], std::slice::from_ref(&pk));
        assert_eq!(selective.len(), 2);

        let ea = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(ea, "w", Value::Int64(10));
        let edge_sel = cdc.get_edges_properties_selective_batch(&[ea], &[PropertyKey::new("w")]);
        assert_eq!(edge_sel.len(), 1);
    }

    // ---------------------------------------------------------------
    // Direct mutations (non-versioned): record to CdcLog immediately
    // ---------------------------------------------------------------

    #[test]
    fn create_node_records_directly() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person", "Employee"]);
        let events = log.history(EntityId::Node(id));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, ChangeKind::Create);
        assert_eq!(events[0].labels.as_ref().unwrap(), &["Person", "Employee"]);
        // pending buffer should be empty (direct recording)
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn create_edge_records_directly() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        let eid = cdc.create_edge(a, b, "KNOWS");
        let events = log.history(EntityId::Edge(eid));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, ChangeKind::Create);
        assert_eq!(events[0].edge_type.as_deref(), Some("KNOWS"));
        assert_eq!(events[0].src_id, Some(a.as_u64()));
        assert_eq!(events[0].dst_id, Some(b.as_u64()));
    }

    #[test]
    fn batch_create_edges_records_directly() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let c = cdc.create_node(&[]);
        let ids = cdc.batch_create_edges(&[(a, b, "X"), (b, c, "Y")]);
        assert_eq!(ids.len(), 2);
        for id in &ids {
            let events = log.history(EntityId::Edge(*id));
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].kind, ChangeKind::Create);
        }
    }

    #[test]
    fn delete_node_records_directly_with_before_props() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["P"]);
        cdc.set_node_property(id, "name", Value::from("Alix"));
        let deleted = cdc.delete_node(id);
        assert!(deleted);
        let events = log.history(EntityId::Node(id));
        // create + update(set) + delete
        let del_event = events
            .iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        let before = del_event.before.as_ref().unwrap();
        assert_eq!(before.get("name"), Some(&Value::from("Alix")));
    }

    #[test]
    fn delete_node_no_event_when_not_found() {
        let (cdc, log) = setup();
        let fake_id = NodeId::new(999);
        let deleted = cdc.delete_node(fake_id);
        assert!(!deleted);
        assert!(log.history(EntityId::Node(fake_id)).is_empty());
    }

    #[test]
    fn delete_edge_records_directly_with_before_props() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "weight", Value::Float64(1.5));
        let deleted = cdc.delete_edge(eid);
        assert!(deleted);
        let del_event = log
            .history(EntityId::Edge(eid))
            .into_iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        let before = del_event.before.as_ref().unwrap();
        assert_eq!(before.get("weight"), Some(&Value::Float64(1.5)));
    }

    #[test]
    fn delete_edge_no_event_when_not_found() {
        let (cdc, log) = setup();
        let fake = EdgeId::new(999);
        assert!(!cdc.delete_edge(fake));
        assert!(log.history(EntityId::Edge(fake)).is_empty());
    }

    #[test]
    fn delete_node_edges_records_each_edge() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let c = cdc.create_node(&[]);
        let e1 = cdc.create_edge(a, b, "X");
        let e2 = cdc.create_edge(c, a, "Y");
        cdc.set_edge_property(e1, "p", Value::Int64(1));

        cdc.delete_node_edges(a);

        // Both edges should have Delete events
        let e1_del = log
            .history(EntityId::Edge(e1))
            .into_iter()
            .any(|e| e.kind == ChangeKind::Delete);
        let e2_del = log
            .history(EntityId::Edge(e2))
            .into_iter()
            .any(|e| e.kind == ChangeKind::Delete);
        assert!(e1_del, "Outgoing edge should have Delete event");
        assert!(e2_del, "Incoming edge should have Delete event");
    }

    #[test]
    fn set_node_property_records_old_and_new() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["N"]);
        cdc.set_node_property(id, "city", Value::from("Amsterdam"));
        cdc.set_node_property(id, "city", Value::from("Berlin"));

        let events = log.history(EntityId::Node(id));
        let updates: Vec<_> = events
            .iter()
            .filter(|e| e.kind == ChangeKind::Update)
            .collect();
        assert_eq!(updates.len(), 2);
        // First update: no before (new property), after = Amsterdam
        assert!(updates[0].before.is_none());
        assert_eq!(
            updates[0].after.as_ref().unwrap().get("city"),
            Some(&Value::from("Amsterdam"))
        );
        // Second update: before = Amsterdam, after = Berlin
        assert_eq!(
            updates[1].before.as_ref().unwrap().get("city"),
            Some(&Value::from("Amsterdam"))
        );
        assert_eq!(
            updates[1].after.as_ref().unwrap().get("city"),
            Some(&Value::from("Berlin"))
        );
    }

    #[test]
    fn set_edge_property_records_old_and_new() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Int64(1));
        cdc.set_edge_property(eid, "w", Value::Int64(2));

        let events = log.history(EntityId::Edge(eid));
        let updates: Vec<_> = events
            .iter()
            .filter(|e| e.kind == ChangeKind::Update)
            .collect();
        assert_eq!(updates.len(), 2);
        assert!(updates[0].before.is_none());
        assert_eq!(
            updates[1].before.as_ref().unwrap().get("w"),
            Some(&Value::Int64(1))
        );
        assert_eq!(
            updates[1].after.as_ref().unwrap().get("w"),
            Some(&Value::Int64(2))
        );
    }

    #[test]
    fn remove_node_property_records_before() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&[]);
        cdc.set_node_property(id, "x", Value::Int64(42));
        let removed = cdc.remove_node_property(id, "x");
        assert_eq!(removed, Some(Value::Int64(42)));

        let events = log.history(EntityId::Node(id));
        let last = events.last().unwrap();
        assert_eq!(last.kind, ChangeKind::Update);
        assert_eq!(
            last.before.as_ref().unwrap().get("x"),
            Some(&Value::Int64(42))
        );
        assert!(last.after.is_none());
    }

    #[test]
    fn remove_node_property_no_event_when_missing() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&[]);
        let removed = cdc.remove_node_property(id, "nope");
        assert!(removed.is_none());
        // Only the Create event, no Update
        let events = log.history(EntityId::Node(id));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn remove_edge_property_records_before() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Float64(19.88));
        let removed = cdc.remove_edge_property(eid, "w");
        assert_eq!(removed, Some(Value::Float64(19.88)));

        let events = log.history(EntityId::Edge(eid));
        let last = events.last().unwrap();
        assert_eq!(last.kind, ChangeKind::Update);
        assert_eq!(
            last.before.as_ref().unwrap().get("w"),
            Some(&Value::Float64(19.88))
        );
    }

    #[test]
    fn remove_edge_property_no_event_when_missing() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        let removed = cdc.remove_edge_property(eid, "nope");
        assert!(removed.is_none());
        // Only Create event
        let events = log.history(EntityId::Edge(eid));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn add_label_records_update() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person"]);
        let added = cdc.add_label(id, "Employee");
        assert!(added);

        let events = log.history(EntityId::Node(id));
        let update = events
            .iter()
            .find(|e| e.kind == ChangeKind::Update)
            .unwrap();
        let labels = update.labels.as_ref().unwrap();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"Employee".to_string()));
    }

    #[test]
    fn add_label_no_event_when_already_present() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person"]);
        let added = cdc.add_label(id, "Person");
        assert!(!added);
        // Only the Create event
        assert_eq!(log.history(EntityId::Node(id)).len(), 1);
    }

    #[test]
    fn remove_label_records_old_labels() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person", "Employee"]);
        let removed = cdc.remove_label(id, "Employee");
        assert!(removed);

        let events = log.history(EntityId::Node(id));
        let update = events
            .iter()
            .find(|e| e.kind == ChangeKind::Update)
            .unwrap();
        // labels field captures the labels BEFORE removal
        let labels = update.labels.as_ref().unwrap();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"Employee".to_string()));
    }

    #[test]
    fn remove_label_no_event_when_missing() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person"]);
        let removed = cdc.remove_label(id, "Nonexistent");
        assert!(!removed);
        assert_eq!(log.history(EntityId::Node(id)).len(), 1);
    }

    // ---------------------------------------------------------------
    // Versioned mutations: buffer events for transactional flush
    // ---------------------------------------------------------------

    #[test]
    fn create_node_versioned_buffers_event() {
        let (cdc, log) = setup();
        let epoch = EpochId(1);
        let tx = TransactionId::new(1);
        let id = cdc.create_node_versioned(&["Person"], epoch, tx);

        // Event goes to buffer, not the log
        assert!(log.history(EntityId::Node(id)).is_empty());
        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].kind, ChangeKind::Create);
        assert_eq!(pending[0].epoch, EpochId::PENDING);
        assert_eq!(pending[0].labels.as_ref().unwrap(), &["Person"]);
    }

    #[test]
    fn create_edge_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        let epoch = EpochId(1);
        let tx = TransactionId::new(1);
        let eid = cdc.create_edge_versioned(a, b, "KNOWS", epoch, tx);

        let pending = cdc.pending_events().lock().clone();
        let edge_events: Vec<_> = pending
            .iter()
            .filter(|e| e.entity_id == EntityId::Edge(eid))
            .collect();
        assert_eq!(edge_events.len(), 1);
        assert_eq!(edge_events[0].kind, ChangeKind::Create);
        assert_eq!(edge_events[0].edge_type.as_deref(), Some("KNOWS"));
        assert_eq!(edge_events[0].src_id, Some(a.as_u64()));
        assert_eq!(edge_events[0].dst_id, Some(b.as_u64()));
    }

    #[test]
    fn delete_node_versioned_buffers_event_with_snapshot() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        cdc.set_node_property(id, "name", Value::from("Alix"));

        let epoch = EpochId(2);
        let tx = TransactionId::new(1);
        let deleted = cdc.delete_node_versioned(id, epoch, tx);
        assert!(deleted);

        let pending = cdc.pending_events().lock().clone();
        let del_event = pending
            .iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        assert_eq!(del_event.epoch, EpochId::PENDING);
        let before = del_event.before.as_ref().unwrap();
        assert_eq!(before.get("name"), Some(&Value::from("Alix")));
        // labels captured
        let labels = del_event.labels.as_ref().unwrap();
        assert!(labels.contains(&"Person".to_string()));
    }

    #[test]
    fn delete_node_versioned_no_buffer_when_not_found() {
        let (cdc, _log) = setup();
        let tx = TransactionId::new(1);
        let deleted = cdc.delete_node_versioned(NodeId::new(999), EpochId(1), tx);
        assert!(!deleted);
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn delete_edge_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Int64(5));

        let tx = TransactionId::new(1);
        let deleted = cdc.delete_edge_versioned(eid, EpochId(2), tx);
        assert!(deleted);

        let pending = cdc.pending_events().lock().clone();
        let del = pending
            .iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        assert_eq!(
            del.before.as_ref().unwrap().get("w"),
            Some(&Value::Int64(5))
        );
    }

    #[test]
    fn delete_edge_versioned_no_buffer_when_not_found() {
        let (cdc, _log) = setup();
        let tx = TransactionId::new(1);
        assert!(!cdc.delete_edge_versioned(EdgeId::new(999), EpochId(1), tx));
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn set_node_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["N"]);
        cdc.set_node_property(id, "x", Value::Int64(1));

        let tx = TransactionId::new(1);
        cdc.set_node_property_versioned(id, "x", Value::Int64(2), tx);

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        let event = &pending[0];
        assert_eq!(event.kind, ChangeKind::Update);
        assert_eq!(
            event.before.as_ref().unwrap().get("x"),
            Some(&Value::Int64(1))
        );
        assert_eq!(
            event.after.as_ref().unwrap().get("x"),
            Some(&Value::Int64(2))
        );
    }

    #[test]
    fn set_edge_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Float64(1.0));

        let tx = TransactionId::new(1);
        cdc.set_edge_property_versioned(eid, "w", Value::Float64(2.0), tx);

        let pending = cdc.pending_events().lock().clone();
        let edge_events: Vec<_> = pending
            .iter()
            .filter(|e| e.entity_id == EntityId::Edge(eid))
            .collect();
        assert_eq!(edge_events.len(), 1);
        assert_eq!(
            edge_events[0].before.as_ref().unwrap().get("w"),
            Some(&Value::Float64(1.0))
        );
        assert_eq!(
            edge_events[0].after.as_ref().unwrap().get("w"),
            Some(&Value::Float64(2.0))
        );
    }

    #[test]
    fn remove_node_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&[]);
        cdc.set_node_property(id, "x", Value::Int64(42));

        let tx = TransactionId::new(1);
        let removed = cdc.remove_node_property_versioned(id, "x", tx);
        assert_eq!(removed, Some(Value::Int64(42)));

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].kind, ChangeKind::Update);
        assert_eq!(
            pending[0].before.as_ref().unwrap().get("x"),
            Some(&Value::Int64(42))
        );
        assert!(pending[0].after.is_none());
    }

    #[test]
    fn remove_node_property_versioned_no_event_when_missing() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&[]);
        let tx = TransactionId::new(1);
        let removed = cdc.remove_node_property_versioned(id, "nope", tx);
        assert!(removed.is_none());
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn remove_edge_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Int64(7));

        let tx = TransactionId::new(1);
        let removed = cdc.remove_edge_property_versioned(eid, "w", tx);
        assert_eq!(removed, Some(Value::Int64(7)));

        let pending = cdc.pending_events().lock().clone();
        let edge_events: Vec<_> = pending
            .iter()
            .filter(|e| e.entity_id == EntityId::Edge(eid))
            .collect();
        assert_eq!(edge_events.len(), 1);
        assert_eq!(
            edge_events[0].before.as_ref().unwrap().get("w"),
            Some(&Value::Int64(7))
        );
    }

    #[test]
    fn remove_edge_property_versioned_no_event_when_missing() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        let tx = TransactionId::new(1);
        let removed = cdc.remove_edge_property_versioned(eid, "nope", tx);
        assert!(removed.is_none());
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn add_label_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let tx = TransactionId::new(1);
        let added = cdc.add_label_versioned(id, "Employee", tx);
        assert!(added);

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].kind, ChangeKind::Update);
        let labels = pending[0].labels.as_ref().unwrap();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"Employee".to_string()));
    }

    #[test]
    fn add_label_versioned_no_buffer_when_duplicate() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let tx = TransactionId::new(1);
        let added = cdc.add_label_versioned(id, "Person", tx);
        assert!(!added);
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn remove_label_versioned_buffers_event_with_old_labels() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person", "Employee"]);
        let tx = TransactionId::new(1);
        let removed = cdc.remove_label_versioned(id, "Employee", tx);
        assert!(removed);

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        let labels = pending[0].labels.as_ref().unwrap();
        // Captures labels BEFORE removal
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"Employee".to_string()));
    }

    #[test]
    fn remove_label_versioned_no_buffer_when_missing() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let tx = TransactionId::new(1);
        let removed = cdc.remove_label_versioned(id, "Nonexistent", tx);
        assert!(!removed);
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn buffered_node_property_chain_snapshots_the_transaction_visible_predecessor() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Chain"]);
        let tx = TransactionId::new(41);

        cdc.set_node_property_buffered(id, "x", Value::Int64(1), tx);
        cdc.set_node_property_buffered(id, "x", Value::Int64(2), tx);
        cdc.remove_node_property_buffered(id, "x", tx);

        let pending = cdc.pending_events().lock();
        assert_eq!(pending.len(), 3);
        assert!(pending[0].before.is_none());
        assert_eq!(
            pending[1]
                .before
                .as_ref()
                .and_then(|before| before.get("x")),
            Some(&Value::Int64(1))
        );
        assert_eq!(
            pending[2]
                .before
                .as_ref()
                .and_then(|before| before.get("x")),
            Some(&Value::Int64(2)),
            "SET new; REMOVE new must retain the removal event"
        );
        assert!(pending[2].after.is_none());
    }

    #[test]
    fn buffered_edge_property_chain_snapshots_the_transaction_visible_predecessor() {
        let (cdc, _log) = setup();
        let source = cdc.create_node(&[]);
        let target = cdc.create_node(&[]);
        let edge = cdc.create_edge(source, target, "CHAIN");
        let tx = TransactionId::new(42);

        cdc.set_edge_property_buffered(edge, "x", Value::Int64(1), tx);
        cdc.set_edge_property_buffered(edge, "x", Value::Int64(2), tx);
        cdc.remove_edge_property_buffered(edge, "x", tx);

        let pending = cdc.pending_events().lock();
        let edge_events: Vec<_> = pending
            .iter()
            .filter(|event| event.entity_id == EntityId::Edge(edge))
            .collect();
        assert_eq!(edge_events.len(), 3);
        assert!(edge_events[0].before.is_none());
        assert_eq!(
            edge_events[1]
                .before
                .as_ref()
                .and_then(|before| before.get("x")),
            Some(&Value::Int64(1))
        );
        assert_eq!(
            edge_events[2]
                .before
                .as_ref()
                .and_then(|before| before.get("x")),
            Some(&Value::Int64(2))
        );
    }

    #[test]
    fn buffered_label_chain_uses_the_transaction_visible_label_set() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Base"]);
        let tx = TransactionId::new(43);

        cdc.add_label_buffered(id, "First", tx);
        cdc.add_label_buffered(id, "Second", tx);
        cdc.add_label_buffered(id, "First", tx);
        cdc.remove_label_buffered(id, "First", tx);

        let pending = cdc.pending_events().lock();
        assert_eq!(pending.len(), 3, "duplicate buffered ADD emits no event");
        let first = pending[0].labels.as_ref().expect("first post-image");
        assert!(first.iter().any(|label| label == "Base"));
        assert!(first.iter().any(|label| label == "First"));
        let second = pending[1].labels.as_ref().expect("second post-image");
        assert!(second.iter().any(|label| label == "First"));
        assert!(second.iter().any(|label| label == "Second"));
        let before_remove = pending[2].labels.as_ref().expect("remove pre-image");
        assert!(before_remove.iter().any(|label| label == "First"));
        assert!(before_remove.iter().any(|label| label == "Second"));
    }

    #[test]
    fn versioned_node_delete_snapshots_buffered_properties_and_labels() {
        let (cdc, _log) = setup();
        let tx = TransactionId::new(44);
        let epoch = cdc.current_epoch();
        let id = cdc.create_node_versioned(&["Base"], epoch, tx);
        cdc.set_node_property_buffered(id, "pending", Value::from("visible"), tx);
        cdc.add_label_buffered(id, "PendingLabel", tx);

        assert!(cdc.delete_node_versioned(id, epoch, tx));

        let pending = cdc.pending_events().lock();
        assert_eq!(
            pending
                .iter()
                .map(|event| event.kind.clone())
                .collect::<Vec<_>>(),
            vec![
                ChangeKind::Create,
                ChangeKind::Update,
                ChangeKind::Update,
                ChangeKind::Delete
            ]
        );
        let deleted = pending
            .iter()
            .find(|event| event.entity_id == EntityId::Node(id) && event.kind == ChangeKind::Delete)
            .expect("versioned node delete event");
        assert_eq!(
            deleted
                .before
                .as_ref()
                .and_then(|before| before.get("pending")),
            Some(&Value::from("visible"))
        );
        assert!(
            deleted
                .labels
                .as_ref()
                .is_some_and(|labels| labels.iter().any(|label| label == "PendingLabel"))
        );
    }

    #[test]
    fn versioned_edge_delete_snapshots_buffered_properties() {
        let (cdc, _log) = setup();
        let source = cdc.create_node(&[]);
        let target = cdc.create_node(&[]);
        let tx = TransactionId::new(45);
        let epoch = cdc.current_epoch();
        let edge = cdc.create_edge_versioned(source, target, "DELETE_CHAIN", epoch, tx);
        cdc.set_edge_property_buffered(edge, "pending", Value::from("visible"), tx);

        assert!(cdc.delete_edge_versioned(edge, epoch, tx));

        let pending = cdc.pending_events().lock();
        let edge_kinds: Vec<_> = pending
            .iter()
            .filter(|event| event.entity_id == EntityId::Edge(edge))
            .map(|event| event.kind.clone())
            .collect();
        assert_eq!(
            edge_kinds,
            vec![ChangeKind::Create, ChangeKind::Update, ChangeKind::Delete]
        );
        let deleted = pending
            .iter()
            .find(|event| {
                event.entity_id == EntityId::Edge(edge) && event.kind == ChangeKind::Delete
            })
            .expect("versioned edge delete event");
        assert_eq!(
            deleted
                .before
                .as_ref()
                .and_then(|before| before.get("pending")),
            Some(&Value::from("visible"))
        );
    }

    // ---------------------------------------------------------------
    // Helper methods
    // ---------------------------------------------------------------

    #[test]
    fn collect_node_properties_returns_none_for_empty() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["N"]);
        assert!(cdc.collect_node_properties(id).is_none());
    }

    #[test]
    fn collect_node_properties_returns_map() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["N"]);
        cdc.set_node_property(id, "a", Value::Int64(1));
        cdc.set_node_property(id, "b", Value::from("hello"));
        let map = cdc.collect_node_properties(id).unwrap();
        assert_eq!(map.get("a"), Some(&Value::Int64(1)));
        assert_eq!(map.get("b"), Some(&Value::from("hello")));
    }

    #[test]
    fn collect_node_properties_returns_none_for_nonexistent() {
        let (cdc, _log) = setup();
        assert!(cdc.collect_node_properties(NodeId::new(999)).is_none());
    }

    #[test]
    fn collect_edge_properties_returns_none_for_empty() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        assert!(cdc.collect_edge_properties(eid).is_none());
    }

    #[test]
    fn collect_edge_properties_returns_map() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Float64(2.5));
        let map = cdc.collect_edge_properties(eid).unwrap();
        assert_eq!(map.get("w"), Some(&Value::Float64(2.5)));
    }

    #[test]
    fn collect_node_labels_returns_labels() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person", "Employee"]);
        let labels = cdc.collect_node_labels(id).unwrap();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"Employee".to_string()));
    }

    #[test]
    fn collect_node_labels_returns_none_for_nonexistent() {
        let (cdc, _log) = setup();
        assert!(cdc.collect_node_labels(NodeId::new(999)).is_none());
    }

    #[test]
    fn make_event_creates_minimal_event() {
        let event = make_event(
            EntityId::Node(NodeId::new(1)),
            ChangeKind::Create,
            EpochId(5),
        );
        assert_eq!(event.entity_id, EntityId::Node(NodeId::new(1)));
        assert_eq!(event.kind, ChangeKind::Create);
        assert_eq!(event.epoch, EpochId(5));
        assert_eq!(event.timestamp, HlcTimestamp::zero());
        assert!(event.before.is_none());
        assert!(event.after.is_none());
        assert!(event.labels.is_none());
        assert!(event.edge_type.is_none());
        assert!(event.src_id.is_none());
        assert!(event.dst_id.is_none());
    }

    // ---------------------------------------------------------------
    // Versioned read delegation (spot checks)
    // ---------------------------------------------------------------

    #[test]
    fn versioned_read_delegates() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["N"]);
        let epoch = cdc.current_epoch();
        let tx = TransactionId::new(0);
        // These should delegate without panic
        let _ = cdc.get_node_versioned(id, epoch, tx);
        let _ = cdc.get_node_at_epoch(id, epoch);
        let _ = cdc.is_node_visible_versioned(id, epoch, tx);
        let _ = cdc.filter_visible_node_ids_versioned(&[id], epoch, tx);
    }

    #[test]
    fn versioned_edge_read_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "E");
        let epoch = cdc.current_epoch();
        let tx = TransactionId::new(0);
        let _ = cdc.get_edge_versioned(e, epoch, tx);
        let _ = cdc.get_edge_at_epoch(e, epoch);
        let _ = cdc.is_edge_visible_versioned(e, epoch, tx);
    }

    #[test]
    fn versioned_traversal_preserves_pending_delete_isolation() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let edge = cdc.create_edge(a, b, "E");
        let epoch = cdc.current_epoch();
        let writer = TransactionId::new(41);
        let reader = TransactionId::new(42);
        assert!(cdc.delete_edge_versioned(edge, epoch, writer));
        assert!(
            cdc.edges_from_versioned(a, Direction::Outgoing, epoch, writer)
                .is_empty()
        );
        assert!(
            cdc.neighbors_versioned(a, Direction::Outgoing, epoch, writer)
                .is_empty()
        );
        assert_eq!(
            cdc.edges_from_versioned(a, Direction::Outgoing, epoch, reader),
            vec![(b, edge)]
        );
        assert_eq!(
            cdc.neighbors_versioned(a, Direction::Outgoing, epoch, reader),
            vec![b]
        );
    }

    // ---------------------------------------------------------------
    // Edge-direction-specific cases for delete_node_edges
    //
    // The loop walks `outgoing.chain(incoming)`. An isolated node or
    // a node with edges only in one direction must still deliver exactly
    // one Delete per edge, and zero Deletes when there are none.
    // ---------------------------------------------------------------

    #[test]
    fn delete_node_edges_on_isolated_node_emits_no_events() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Solo"]);
        let create_count = log.history(EntityId::Node(id)).len();

        cdc.delete_node_edges(id);

        // No edges existed, so no Delete events anywhere in the log
        // should be tagged for a nonexistent edge id, and the node's
        // own history is unchanged.
        assert_eq!(
            log.history(EntityId::Node(id)).len(),
            create_count,
            "delete_node_edges on isolated node must not touch the node's history"
        );
    }

    #[test]
    fn delete_node_edges_with_only_outgoing_edges() {
        let (cdc, log) = setup();
        let src = cdc.create_node(&[]);
        let dst_a = cdc.create_node(&[]);
        let dst_b = cdc.create_node(&[]);
        let e1 = cdc.create_edge(src, dst_a, "OUT1");
        let e2 = cdc.create_edge(src, dst_b, "OUT2");

        cdc.delete_node_edges(src);

        let e1_deletes: Vec<_> = log
            .history(EntityId::Edge(e1))
            .into_iter()
            .filter(|ev| ev.kind == ChangeKind::Delete)
            .collect();
        let e2_deletes: Vec<_> = log
            .history(EntityId::Edge(e2))
            .into_iter()
            .filter(|ev| ev.kind == ChangeKind::Delete)
            .collect();
        assert_eq!(e1_deletes.len(), 1);
        assert_eq!(e2_deletes.len(), 1);
    }

    #[test]
    fn delete_node_edges_with_only_incoming_edges() -> Result<(), Box<dyn std::error::Error>> {
        let (cdc, log) = setup();
        let target = cdc.create_node(&[]);
        let src_a = cdc.create_node(&[]);
        let src_b = cdc.create_node(&[]);
        let e1 = cdc.create_edge(src_a, target, "IN1");
        let e2 = cdc.create_edge(src_b, target, "IN2");

        cdc.delete_node_edges(target);

        for edge_id in [e1, e2] {
            let deletes: Vec<_> = log
                .history(EntityId::Edge(edge_id))
                .into_iter()
                .filter(|ev| ev.kind == ChangeKind::Delete)
                .collect();
            assert_eq!(
                deletes.len(),
                1,
                "incoming-only edge {edge_id:?} should have exactly one Delete event"
            );
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // Batch and labels edge cases
    // ---------------------------------------------------------------

    #[test]
    fn batch_create_edges_empty_slice_records_nothing() {
        let (cdc, _log) = setup();
        let ids = cdc.batch_create_edges(&[]);
        assert!(ids.is_empty());
        // No transactional buffering either (batch uses direct recording).
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn create_node_with_empty_labels_still_populates_labels_field() {
        // Documents a deliberate choice: the labels field on a Create
        // event is Some(empty vec), not None, even with no labels. This
        // matters for downstream consumers that distinguish
        // "missing field" vs "empty list of labels."
        let (cdc, log) = setup();
        let id = cdc.create_node(&[]);
        let events = log.history(EntityId::Node(id));
        assert_eq!(events.len(), 1);
        let labels = events[0]
            .labels
            .as_ref()
            .expect("labels must be Some even with empty input");
        assert!(labels.is_empty());
    }

    // ---------------------------------------------------------------
    // Shared-buffer semantics via wrap()
    //
    // Two CdcGraphStores wrapping the same transaction accumulator
    // (e.g., default graph + named graph) must buffer versioned events
    // into the same Vec so the session can flush them atomically.
    // ---------------------------------------------------------------

    #[test]
    fn wrap_routes_versioned_events_to_shared_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let inner = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let shared = Arc::new(TransactionChangeAccumulator::new(&log));

        let cdc = CdcGraphStore::wrap(
            Arc::clone(&inner) as Arc<dyn GraphStoreMut>,
            shared,
            Arc::clone(&inner),
            GraphPath::from_components(&["analytics"])?,
        );

        let id = cdc.create_node_versioned(&["P"], EpochId(1), TransactionId::new(1));
        let pending = cdc.pending_events();
        let buf = pending.lock();
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].entity_id, EntityId::Node(id));
        assert_eq!(buf[0].kind, ChangeKind::Create);
        assert_eq!(
            buf[0].graph_path(),
            Some(&GraphPath::from_components(&["analytics"])?)
        );
        Ok(())
    }

    #[test]
    fn two_wrapped_stores_share_buffer_for_versioned_mutations()
    -> Result<(), Box<dyn std::error::Error>> {
        let inner_a = Arc::new(LpgStore::new().unwrap());
        let inner_b = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let shared = Arc::new(TransactionChangeAccumulator::new(&log));

        let cdc_a = CdcGraphStore::wrap(
            Arc::clone(&inner_a) as Arc<dyn GraphStoreMut>,
            Arc::clone(&shared),
            Arc::clone(&inner_a),
            GraphPath::from_components(&["alpha"])?,
        );
        let cdc_b = CdcGraphStore::wrap(
            Arc::clone(&inner_b) as Arc<dyn GraphStoreMut>,
            Arc::clone(&shared),
            Arc::clone(&inner_b),
            GraphPath::from_components(&["beta"])?,
        );

        let tx = TransactionId::new(42);
        let _ = cdc_a.create_node_versioned(&["A"], EpochId(1), tx);
        let _ = cdc_b.create_node_versioned(&["B"], EpochId(1), tx);

        // Both events land in the single shared buffer, in order.
        let buf = shared.lock();
        assert_eq!(buf.len(), 2);
        let labels_a = buf[0].labels.as_ref().unwrap();
        let labels_b = buf[1].labels.as_ref().unwrap();
        assert_eq!(labels_a, &["A"]);
        assert_eq!(labels_b, &["B"]);
        // Both use PENDING since the commit epoch hasn't been assigned.
        assert_eq!(buf[0].epoch, EpochId::PENDING);
        assert_eq!(buf[1].epoch, EpochId::PENDING);
        assert_eq!(
            buf[0].graph_path(),
            Some(&GraphPath::from_components(&["alpha"])?)
        );
        assert_eq!(
            buf[1].graph_path(),
            Some(&GraphPath::from_components(&["beta"])?)
        );
        Ok(())
    }
}
