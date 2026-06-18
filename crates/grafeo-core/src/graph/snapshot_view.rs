//! Snapshot-recording `GraphStore` wrapper for graph algorithms.
//!
//! [`SnapshotView`] wraps any `&dyn GraphStoreSearch` and presents the
//! 6 read methods that graph algorithms exercise through snapshot-visible,
//! SSI-read-recording counterparts.  Every other `GraphStore` /
//! `GraphStoreSearch` method is forwarded unchanged to the inner store.
//!
//! # Why this exists
//!
//! Graph-algorithm CALL procedures (e.g. shortest-path, centrality) accept
//! `&dyn GraphStore` and are read-only.  Without a wrapper they call the
//! *current-epoch* variants (`node_ids`, `edges_from`, `get_node`,
//! `get_edge`, `find_nodes_by_property`) which are neither snapshot-pinned
//! to the transaction's epoch nor recorded into the SSI read-set.
//!
//! `SnapshotView` corrects both problems **without** changing any algorithm
//! code: construct a view at the transaction's `(epoch, tx)` and pass `&view`
//! where the algorithm expects `&dyn GraphStore`.
//!
//! # Recording contract
//!
//! `filter_visible_node_ids_versioned` (used for `node_ids`) and
//! `get_node_versioned` / `get_edge_versioned` / `edges_from_versioned` all
//! record into the SSI read-set when a `SharedReadTracker` is registered for
//! `tx` on the inner store.  For `find_nodes_by_property` there is no single
//! versioned counterpart, so we synthesize the filter:
//! `inner.find_nodes_by_property(p, v)` followed by
//! `inner.is_node_visible_versioned(id, epoch, tx)` for each candidate — the
//! `is_node_visible_versioned` call goes through `get_node_versioned`, which
//! records the visible ones.  Snapshot-invisible candidates are dropped
//! *and* never recorded, matching the contract of every other method.
//!
//! # Non-Serializable transactions
//!
//! The read-tracker is only registered for Serializable transactions.  For
//! SI / Read-Committed the tracker is absent, so every `record_*` call is a
//! no-op.  Wrapping is therefore harmless under any isolation level.
//!
//! # Completeness invariant (maintainers)
//!
//! SSI soundness depends on *every* read an algorithm makes routing through a
//! recording override here — true today because algorithms touch only the 6
//! methods above.  **A new algorithm that calls another `GraphStore` read
//! (e.g. `get_node_property`, `nodes_by_label`, `get_nodes_properties_batch`)
//! MUST get a recording override** — a delegated read silently bypasses the SSI
//! read-set + snapshot pin (unrecorded read = missed conflict = unsound). Re-grep
//! `grafeo-adapters/src/plugins/algorithms` for `store.` when adding one.

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use std::sync::Arc;

use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreSearch};
use crate::statistics::Statistics;

/// A snapshot-pinned, SSI-read-recording view of a graph store.
///
/// Forwards all read calls to `inner`, but the 6 methods that graph
/// algorithms exercise are routed through their versioned/recording
/// counterparts so that:
///
/// * results are filtered to nodes/edges visible at `epoch`, and
/// * each visible entity is recorded into the SSI read-set for `tx`.
///
/// All other `GraphStore` / `GraphStoreSearch` methods are delegated
/// unchanged.
pub struct SnapshotView<'a> {
    inner: &'a dyn GraphStoreSearch,
    epoch: EpochId,
    tx: TransactionId,
}

impl<'a> SnapshotView<'a> {
    /// Creates a new snapshot view pinned to `(epoch, tx)`.
    pub fn new(inner: &'a dyn GraphStoreSearch, epoch: EpochId, tx: TransactionId) -> Self {
        Self { inner, epoch, tx }
    }
}

// ─── GraphStore implementation ───────────────────────────────────────────────

impl GraphStore for SnapshotView<'_> {
    // ── Overridden: the 6 algorithm read methods ─────────────────────────────

    /// Returns the node if visible at `(epoch, tx)`, recording it into the
    /// SSI read-set when visible.
    fn get_node(&self, id: NodeId) -> Option<Node> {
        self.inner.get_node_versioned(id, self.epoch, self.tx)
    }

    /// Returns the edge if visible at `(epoch, tx)`, recording it into the
    /// SSI read-set when visible.
    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.inner.get_edge_versioned(id, self.epoch, self.tx)
    }

    /// Returns snapshot-visible `(target, edge_id)` pairs, recording each
    /// visible edge into the SSI read-set.
    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        self.inner
            .edges_from_versioned(node, direction, self.epoch, self.tx)
    }

    /// Returns snapshot-visible neighbor node IDs.
    ///
    /// Delegates to `edges_from` (which is already overridden to use the
    /// versioned path), so the read-recording contract is preserved.
    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        self.inner
            .neighbors_versioned(node, direction, self.epoch, self.tx)
    }

    /// Returns all node IDs visible at `(epoch, tx)`, recording each
    /// visible node into the SSI read-set.
    fn node_ids(&self) -> Vec<NodeId> {
        let all = self.inner.all_node_ids();
        self.inner
            .filter_visible_node_ids_versioned(&all, self.epoch, self.tx)
    }

    /// Returns nodes matching `(property, value)` that are also visible at
    /// `(epoch, tx)`.
    ///
    /// There is no direct `find_nodes_by_property_versioned` counterpart, so
    /// we synthesize: call `inner.find_nodes_by_property` to get candidates,
    /// then filter through `inner.is_node_visible_versioned` (which calls
    /// `get_node_versioned` internally and records visible nodes).
    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        self.inner
            .find_nodes_by_property(property, value)
            .into_iter()
            .filter(|&id| {
                self.inner
                    .is_node_visible_versioned(id, self.epoch, self.tx)
            })
            .collect()
    }

    // ── Delegated: versioned pass-throughs (already snapshot-aware) ──────────

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        self.inner.get_node_versioned(id, epoch, transaction_id)
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

    // ── Delegated: property access ────────────────────────────────────────────

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_edge_property(id, key)
    }

    fn pending_node_creates(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.inner.pending_node_creates(transaction_id)
    }

    fn pending_edge_creates(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        self.inner.pending_edge_creates(transaction_id)
    }

    fn register_read_tracker(
        &self,
        tx: TransactionId,
        tracker: crate::execution::operators::SharedReadTracker,
    ) {
        self.inner.register_read_tracker(tx, tracker);
    }

    fn unregister_read_tracker(&self, tx: TransactionId) {
        self.inner.unregister_read_tracker(tx);
    }

    fn register_write_tracker(
        &self,
        tx: TransactionId,
        tracker: crate::execution::operators::SharedWriteTracker,
    ) {
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

    fn read_node_labels_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> grafeo_common::utils::hash::FxHashSet<ArcStr> {
        self.inner
            .read_node_labels_visible(id, epoch, transaction_id)
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

    // ── Delegated: traversal metadata ────────────────────────────────────────

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

    // ── Delegated: scans ─────────────────────────────────────────────────────

    fn all_node_ids(&self) -> Vec<NodeId> {
        self.inner.all_node_ids()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        self.inner.nodes_by_label(label)
    }

    fn nodes_by_label_visible(
        &self,
        label: &str,
        transaction_id: Option<TransactionId>,
    ) -> Vec<NodeId> {
        self.inner.nodes_by_label_visible(label, transaction_id)
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

    // ── Delegated: entity metadata ────────────────────────────────────────────

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        self.inner.edge_type(id)
    }

    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        self.inner.edge_type_versioned(id, epoch, transaction_id)
    }

    // ── Delegated: index introspection ───────────────────────────────────────

    fn has_property_index(&self, property: &str) -> bool {
        self.inner.has_property_index(property)
    }

    // ── Delegated: additional filtered search ────────────────────────────────

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

    // ── Delegated: zone maps ──────────────────────────────────────────────────

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

    // ── Delegated: statistics ─────────────────────────────────────────────────

    fn statistics(&self) -> Arc<Statistics> {
        self.inner.statistics()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.inner.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.inner.estimate_avg_degree(edge_type, outgoing)
    }

    // ── Delegated: epoch ──────────────────────────────────────────────────────

    fn current_epoch(&self) -> EpochId {
        self.inner.current_epoch()
    }

    // ── Delegated: schema introspection ──────────────────────────────────────

    fn all_labels(&self) -> Vec<String> {
        self.inner.all_labels()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.inner.all_edge_types()
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.inner.all_property_keys()
    }

    // ── Delegated: visibility checks ─────────────────────────────────────────

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

    // ── Delegated: history ────────────────────────────────────────────────────

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        self.inner.get_node_history(id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        self.inner.get_edge_history(id)
    }
}

// ─── GraphStoreSearch implementation ─────────────────────────────────────────
//
// All `GraphStoreSearch` methods have default implementations that call
// `GraphStore` methods (which we already override above) or are purely
// optional search features that delegate to the inner store.

impl GraphStoreSearch for SnapshotView<'_> {
    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.inner.has_text_index(label, property)
    }

    #[cfg(feature = "text-index")]
    fn score_text(&self, node_id: NodeId, label: &str, property: &str, query: &str) -> Option<f64> {
        self.inner.score_text(node_id, label, property, query)
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

    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.inner.has_vector_index(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_metric(
        &self,
        label: &str,
        property: &str,
    ) -> Option<crate::index::vector::DistanceMetric> {
        self.inner.vector_index_metric(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: crate::index::vector::DistanceMetric,
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
        metric: crate::index::vector::DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        self.inner
            .vector_search_with_threshold(label, property, query, threshold, metric)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "lpg")]
mod tests {
    use super::*;
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use crate::graph::lpg::LpgStore;
    use grafeo_common::types::EdgeId;
    use parking_lot::Mutex;
    use std::collections::HashSet;

    // ── Spy tracker ──────────────────────────────────────────────────────────

    struct SpyTracker {
        nodes: Mutex<HashSet<NodeId>>,
        edges: Mutex<HashSet<EdgeId>>,
    }

    impl SpyTracker {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                nodes: Mutex::new(HashSet::new()),
                edges: Mutex::new(HashSet::new()),
            })
        }
        fn recorded_nodes(&self) -> HashSet<NodeId> {
            self.nodes.lock().clone()
        }
        fn recorded_edges(&self) -> HashSet<EdgeId> {
            self.edges.lock().clone()
        }
    }

    impl ReadTracker for SpyTracker {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.nodes.lock().insert(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().insert(id);
        }
    }

    // ── Graph fixture ────────────────────────────────────────────────────────
    //
    // Two epochs:
    //   epoch E1: node A, node B, edge A→B ("LINK")  — committed
    //   epoch E2 (after snapshot): node C              — committed AFTER our snapshot
    //
    // Snapshot epoch = E1.  Under the SnapshotView C must be invisible.

    struct Fixture {
        store: Arc<LpgStore>,
        spy: Arc<SpyTracker>,
        tx: TransactionId,
        snapshot_epoch: EpochId,
        /// node created before snapshot
        node_a: NodeId,
        node_b: NodeId,
        edge_ab: EdgeId,
        /// node created AFTER snapshot — must be invisible
        node_c: NodeId,
    }

    fn build_fixture() -> Fixture {
        let store = Arc::new(LpgStore::new().unwrap());

        // E1: create A, B and edge A→B, commit them.
        let e1 = store.new_epoch();
        let node_a = store.create_node_versioned(&["Person"], e1, TransactionId::SYSTEM);
        let node_b = store.create_node_versioned(&["Person"], e1, TransactionId::SYSTEM);
        store.finalize_entities_by_id(TransactionId::SYSTEM, e1, &[node_a, node_b], &[]);

        let e1b = store.new_epoch();
        let edge_ab =
            store.create_edge_versioned(node_a, node_b, "LINK", e1b, TransactionId::SYSTEM);
        let snapshot_epoch = store.new_epoch();
        store.finalize_entities_by_id(TransactionId::SYSTEM, snapshot_epoch, &[], &[edge_ab]);

        // Set a property on A for find_nodes_by_property tests.
        store.set_node_property(node_a, "name", Value::from("alix"));
        store.set_node_property(node_b, "name", Value::from("gus"));

        // E_late: create C *after* the snapshot, commit it.
        let e_late = store.new_epoch();
        let node_c = store.create_node_versioned(&["Person"], e_late, TransactionId::SYSTEM);
        let e_commit = store.new_epoch();
        store.finalize_entities_by_id(TransactionId::SYSTEM, e_commit, &[node_c], &[]);

        // Register a spy read-tracker for `tx`.
        let tx = TransactionId::new(42);
        let spy = SpyTracker::new();
        store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);

        Fixture {
            store,
            spy,
            tx,
            snapshot_epoch,
            node_a,
            node_b,
            edge_ab,
            node_c,
        }
    }

    // ── node_ids: snapshot visibility + recording ─────────────────────────────

    #[test]
    fn node_ids_returns_only_snapshot_visible_nodes() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let ids: HashSet<NodeId> = view.node_ids().into_iter().collect();

        assert!(ids.contains(&f.node_a), "node_a must be visible");
        assert!(ids.contains(&f.node_b), "node_b must be visible");
        assert!(
            !ids.contains(&f.node_c),
            "node_c (post-snapshot) must NOT be visible"
        );
    }

    #[test]
    fn node_ids_records_visible_nodes_into_read_set() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let _ = view.node_ids();

        let recorded = f.spy.recorded_nodes();
        assert!(
            recorded.contains(&f.node_a),
            "node_a must appear in SSI read-set after node_ids()"
        );
        assert!(
            recorded.contains(&f.node_b),
            "node_b must appear in SSI read-set after node_ids()"
        );
        assert!(
            !recorded.contains(&f.node_c),
            "node_c (post-snapshot) must NOT appear in SSI read-set"
        );
    }

    // ── edges_from: snapshot visibility + recording ───────────────────────────

    #[test]
    fn edges_from_returns_only_snapshot_visible_edges() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let edges: HashSet<EdgeId> = view
            .edges_from(f.node_a, Direction::Outgoing)
            .into_iter()
            .map(|(_, eid)| eid)
            .collect();

        assert!(edges.contains(&f.edge_ab), "edge_ab must be visible");
    }

    #[test]
    fn edges_from_matches_inner_versioned_directly() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let via_view = view.edges_from(f.node_a, Direction::Outgoing);
        let via_inner =
            f.store
                .edges_from_versioned(f.node_a, Direction::Outgoing, f.snapshot_epoch, f.tx);

        // Same set of (target, edge) pairs (order may differ).
        let set_view: HashSet<_> = via_view.into_iter().collect();
        let set_inner: HashSet<_> = via_inner.into_iter().collect();
        assert_eq!(set_view, set_inner);
    }

    #[test]
    fn edges_from_records_visible_edges_into_read_set() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let _ = view.edges_from(f.node_a, Direction::Outgoing);

        assert!(
            f.spy.recorded_edges().contains(&f.edge_ab),
            "edge_ab must appear in SSI read-set after edges_from()"
        );
    }

    // ── get_node: snapshot visibility + recording ─────────────────────────────

    #[test]
    fn get_node_returns_visible_node() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        assert!(
            view.get_node(f.node_a).is_some(),
            "node_a is visible at snapshot epoch"
        );
    }

    #[test]
    fn get_node_excludes_post_snapshot_node() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        assert!(
            view.get_node(f.node_c).is_none(),
            "node_c was created after snapshot; must be invisible"
        );
    }

    #[test]
    fn get_node_records_visible_node_into_read_set() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let _ = view.get_node(f.node_a);

        assert!(
            f.spy.recorded_nodes().contains(&f.node_a),
            "node_a must appear in SSI read-set after get_node()"
        );
    }

    #[test]
    fn get_node_does_not_record_invisible_node() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let _ = view.get_node(f.node_c);

        assert!(
            !f.spy.recorded_nodes().contains(&f.node_c),
            "invisible node must NOT be recorded"
        );
    }

    // ── get_edge: snapshot visibility + recording ─────────────────────────────

    #[test]
    fn get_edge_returns_visible_edge() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        assert!(
            view.get_edge(f.edge_ab).is_some(),
            "edge_ab is visible at snapshot epoch"
        );
    }

    #[test]
    fn get_edge_records_visible_edge_into_read_set() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let _ = view.get_edge(f.edge_ab);

        assert!(
            f.spy.recorded_edges().contains(&f.edge_ab),
            "edge_ab must appear in SSI read-set after get_edge()"
        );
    }

    // ── find_nodes_by_property: snapshot visibility + recording ───────────────

    #[test]
    fn find_nodes_by_property_returns_only_snapshot_visible() {
        let f = build_fixture();
        // Give node_c a "name" property too, but it is post-snapshot and must be excluded.
        f.store
            .set_node_property(f.node_c, "name", Value::from("charlie"));

        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        // "alix" belongs to node_a which IS visible.
        let results: HashSet<NodeId> = view
            .find_nodes_by_property("name", &Value::from("alix"))
            .into_iter()
            .collect();
        assert!(
            results.contains(&f.node_a),
            "node_a must be found (visible, property matches)"
        );
        assert!(
            !results.contains(&f.node_c),
            "node_c (post-snapshot) must not be returned"
        );
    }

    #[test]
    fn find_nodes_by_property_records_visible_matches() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);

        let _ = view.find_nodes_by_property("name", &Value::from("alix"));

        assert!(
            f.spy.recorded_nodes().contains(&f.node_a),
            "node_a must appear in SSI read-set after find_nodes_by_property()"
        );
    }

    // ── Delegation: non-overridden methods reach inner store ──────────────────

    #[test]
    fn delegation_current_epoch_reaches_inner() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);
        assert_eq!(view.current_epoch(), f.store.current_epoch());
    }

    #[test]
    fn delegation_node_count_reaches_inner() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);
        // node_count is delegated (not snapshot-filtered); returns the inner store value.
        assert_eq!(view.node_count(), f.store.node_count());
    }

    #[test]
    fn delegation_statistics_reaches_inner() {
        let f = build_fixture();
        let view = SnapshotView::new(f.store.as_ref(), f.snapshot_epoch, f.tx);
        // Should not panic; just confirm delegation compiles and runs.
        let _stats = view.statistics();
    }
}
