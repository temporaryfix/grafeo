//! Index management methods for [`LpgStore`].

use super::LpgStore;
use dashmap::DashMap;
#[cfg(feature = "text-index")]
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::types::{HashableValue, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashSet;
#[cfg(feature = "text-index")]
use parking_lot::RwLock;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::sync::Arc;

#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;

impl LpgStore {
    /// Creates an index on a node property for O(1) lookups by value.
    ///
    /// After creating an index, calls to [`Self::find_nodes_by_property`] will be
    /// O(1) instead of O(n) for this property. The index is automatically
    /// maintained when properties are set or removed.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    ///
    /// // Create nodes with an 'id' property
    /// let alix = store.create_node(&["Person"]);
    /// store.set_node_property(alix, "id", Value::from("alice_123"));
    ///
    /// // Create an index on the 'id' property
    /// store.create_property_index("id");
    ///
    /// // Now lookups by 'id' are O(1)
    /// let found = store.find_nodes_by_property("id", &Value::from("alice_123"));
    /// assert!(found.contains(&alix));
    /// ```
    pub fn create_property_index(&self, property: &str) {
        let key = PropertyKey::new(property);

        let mut indexes = self.property_indexes.write();
        if indexes.contains_key(&key) {
            return; // Already indexed
        }

        // Create the index and populate it with existing data
        let index: DashMap<HashableValue, FxHashSet<NodeId>> = DashMap::new();

        // Scan all nodes to build the index
        for node_id in self.node_ids() {
            if let Some(value) = self.node_properties.get(node_id, &key) {
                let hv = HashableValue::new(value);
                index.entry(hv).or_default().insert(node_id);
            }
        }

        indexes.insert(key, index);
    }

    /// Drops an index on a node property.
    ///
    /// Returns `true` if the index existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        let key = PropertyKey::new(property);
        self.property_indexes.write().remove(&key).is_some()
    }

    /// Returns `true` if the property has an index.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        let key = PropertyKey::new(property);
        self.property_indexes.read().contains_key(&key)
    }

    /// Returns the names of all indexed properties.
    #[must_use]
    pub fn property_index_keys(&self) -> Vec<String> {
        self.property_indexes
            .read()
            .keys()
            .map(|k| k.to_string())
            .collect()
    }

    /// Updates property indexes when a property is set.
    pub(super) fn update_property_index_on_set(
        &self,
        node_id: NodeId,
        key: &PropertyKey,
        new_value: &Value,
    ) {
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            // Get old value to remove from index
            if let Some(old_value) = self.node_properties.get(node_id, key) {
                let old_hv = HashableValue::new(old_value);
                if let Some(mut nodes) = index.get_mut(&old_hv) {
                    nodes.remove(&node_id);
                    if nodes.is_empty() {
                        drop(nodes);
                        index.remove(&old_hv);
                    }
                }
            }

            // Add new value to index
            let new_hv = HashableValue::new(new_value.clone());
            index
                .entry(new_hv)
                .or_insert_with(FxHashSet::default)
                .insert(node_id);
        }
    }

    /// Updates property indexes when a property is removed.
    pub(super) fn update_property_index_on_remove(&self, node_id: NodeId, key: &PropertyKey) {
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            // Get old value to remove from index
            if let Some(old_value) = self.node_properties.get(node_id, key) {
                let old_hv = HashableValue::new(old_value);
                if let Some(mut nodes) = index.get_mut(&old_hv) {
                    nodes.remove(&node_id);
                    if nodes.is_empty() {
                        drop(nodes);
                        index.remove(&old_hv);
                    }
                }
            }
        }
    }

    /// Stores a vector index for a label+property pair.
    #[cfg(feature = "vector-index")]
    pub fn add_vector_index(&self, label: &str, property: &str, index: Arc<VectorIndexKind>) {
        let key = format!("{label}:{property}");
        self.vector_indexes.write().insert(key, index);
    }

    /// Retrieves the vector index for a label+property pair.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn get_vector_index(&self, label: &str, property: &str) -> Option<Arc<VectorIndexKind>> {
        let key = format!("{label}:{property}");
        self.vector_indexes.read().get(&key).cloned()
    }

    /// Removes a vector index for a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "vector-index")]
    pub fn remove_vector_index(&self, label: &str, property: &str) -> bool {
        let key = format!("{label}:{property}");
        self.vector_indexes.write().remove(&key).is_some()
    }

    /// Returns all vector index entries as `(key, index)` pairs.
    ///
    /// Keys are in `"label:property"` format.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn vector_index_entries(&self) -> Vec<(String, Arc<VectorIndexKind>)> {
        self.vector_indexes
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Looks up a vector index by its `"label:property"` key.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn get_vector_index_by_key(&self, key: &str) -> Option<Arc<VectorIndexKind>> {
        self.vector_indexes.read().get(key).cloned()
    }

    /// Stores a text index for a label+property pair.
    #[cfg(feature = "text-index")]
    pub fn add_text_index(
        &self,
        label: &str,
        property: &str,
        index: Arc<RwLock<crate::index::text::InvertedIndex>>,
    ) {
        let key = format!("{label}:{property}");
        self.text_indexes.write().insert(key, index);
    }

    /// Retrieves the text index for a label+property pair.
    #[cfg(feature = "text-index")]
    #[must_use]
    pub fn get_text_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<Arc<RwLock<crate::index::text::InvertedIndex>>> {
        let key = format!("{label}:{property}");
        self.text_indexes.read().get(&key).cloned()
    }

    /// Removes a text index for a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "text-index")]
    pub fn remove_text_index(&self, label: &str, property: &str) -> bool {
        let key = format!("{label}:{property}");
        self.text_indexes.write().remove(&key).is_some()
    }

    /// Returns all text index entries as `(key, index)` pairs.
    ///
    /// The key format is `"label:property"`.
    #[cfg(feature = "text-index")]
    pub fn text_index_entries(
        &self,
    ) -> Vec<(String, Arc<RwLock<crate::index::text::InvertedIndex>>)> {
        self.text_indexes
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Updates text indexes when a node property is set.
    ///
    /// If the node has a label with a text index on this property key,
    /// the index is updated with the new value (if it's a string).
    #[cfg(feature = "text-index")]
    pub(super) fn update_text_index_on_set(&self, id: NodeId, key: &str, value: &Value) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let label_set = node_labels.get(&id);
        #[cfg(feature = "temporal")]
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            // Single-source invariant (TI5): stamp the index posting at the SAME
            // epoch the property version is finalized to. On the commit path this
            // runs inside `apply_tx_overlay` -> `set_node_property`, where
            // `current_epoch()` is already the commit epoch `C` (the engine calls
            // `finalize_entities_by_id` -> `sync_epoch(C)` BEFORE `apply_tx_overlay`),
            // so the new posting's `created_epoch == C` exactly matches the
            // property version's visibility boundary. On the auto-commit /
            // non-transactional path `current_epoch()` is the live committed epoch,
            // identical to the epoch stamped on the property column write in the
            // same `set_node_property` call.
            let epoch = self.current_epoch();
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let index_key = format!("{label_name}:{key}");
                    if let Some(index) = text_indexes.get(&index_key) {
                        let mut idx = index.write();
                        // Soft-delete the old posting at `epoch`, then insert the
                        // new one at `epoch` if it's a string. A snapshot `< epoch`
                        // sees the old text; `>= epoch` the new text.
                        idx.remove_versioned(id, epoch, None);
                        if let Value::String(text) = value {
                            idx.insert_versioned(id, text, epoch, None);
                        }
                    }
                }
            }
        }
    }

    /// Updates text indexes when a node property is removed.
    #[cfg(feature = "text-index")]
    pub(super) fn update_text_index_on_remove(&self, id: NodeId, key: &str) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let label_set = node_labels.get(&id);
        #[cfg(feature = "temporal")]
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            // Single-source invariant (TI5): stamp the deletion at the property's
            // commit epoch `C` (see `update_text_index_on_set`). The posting's
            // `deleted_epoch == C` so a snapshot `< C` still sees the old text and
            // `>= C` sees the removal — matching the property version boundary.
            let epoch = self.current_epoch();
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let index_key = format!("{label_name}:{key}");
                    if let Some(index) = text_indexes.get(&index_key) {
                        index.write().remove_versioned(id, epoch, None);
                    }
                }
            }
        }
    }

    /// Removes a node from all text indexes.
    #[cfg(feature = "text-index")]
    pub(super) fn remove_from_all_text_indexes(&self, id: NodeId) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        // Single-source: stamp the removal at the current (commit) epoch so a
        // deleted node's text postings are `deleted_epoch = C` — visible to a
        // snapshot before C, hidden at/after C — matching the node's property
        // version chain. (Legacy `remove` stamped epoch 0 = hidden from every
        // snapshot, which lied to pre-delete readers.)
        let epoch = self.current_epoch();
        for (_, index) in text_indexes.iter() {
            index.write().remove_versioned(id, epoch, None);
        }
    }

    // === Snapshot threshold search (TI-threshold-visible) ===

    /// Searches a text index at `(epoch, tx)` returning every document whose
    /// BM25 score meets or exceeds `threshold`, and records the index read for
    /// SSI conflict detection.
    ///
    /// Mirrors [`search_text_visible`] exactly, but uses a threshold cutoff
    /// instead of a top-k limit.  The `index_key` format is `"label:property"`.
    #[cfg(feature = "text-index")]
    #[must_use]
    pub fn search_text_with_threshold_visible(
        &self,
        index_key: &str,
        query: &str,
        threshold: f64,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Vec<(NodeId, f64)> {
        // Record the index read for anti-phantom SSI — must happen even when
        // no postings match so that a zero-result threshold scan still closes
        // the rw-antidependency cycle if a concurrent tx inserts a matching doc.
        self.record_read_index(tx, index_key);

        // Build delta_docs and delta_removed from the overlay (identical to
        // search_text_visible).
        let (delta_docs, delta_removed): (Vec<(NodeId, String)>, FxHashSet<NodeId>) = {
            let overlay = self.text_index_overlay.read();
            match overlay.get(&tx) {
                None => (Vec::new(), FxHashSet::default()),
                Some(delta) => {
                    let mut docs = Vec::new();
                    let mut removed = FxHashSet::default();
                    for (node_id, opt_text) in delta.changes_for(index_key) {
                        match opt_text {
                            Some(text) => docs.push((node_id, text.clone())),
                            None => {
                                removed.insert(node_id);
                            }
                        }
                    }
                    (docs, removed)
                }
            }
        };

        // Look up the committed index.
        let committed_idx = {
            let text_indexes = self.text_indexes.read();
            text_indexes.get(index_key).cloned()
        };

        match committed_idx {
            Some(idx_arc) => {
                let idx = idx_arc.read();
                idx.search_with_threshold_visible(
                    query,
                    threshold,
                    epoch,
                    tx,
                    &delta_docs,
                    &delta_removed,
                )
            }
            None => {
                // No committed index — search only the delta docs.
                if delta_docs.is_empty() {
                    return Vec::new();
                }
                let mut transient = crate::index::text::InvertedIndex::new(
                    crate::index::text::BM25Config::default(),
                );
                for (node_id, text) in &delta_docs {
                    transient.insert(*node_id, text);
                }
                transient.search_with_threshold(query, threshold)
            }
        }
    }

    // === Snapshot per-row score (per-row filter path, anti-phantom SSI) ===

    /// Scores a single node against a text query at `(epoch, tx)`, recording
    /// the index read for SSI conflict detection before computing the score.
    ///
    /// This is the snapshot-aware counterpart of
    /// [`GraphStoreSearch::score_text`]: it records
    /// `record_read_index(tx, index_key)` **first** (so even a non-matching row
    /// closes the rw-antidependency cycle against a concurrent phantom insert),
    /// then scores the node using postings visible at `(epoch, tx)`.
    ///
    /// The `index_key` format is `"label:property"`.
    #[cfg(feature = "text-index")]
    #[must_use]
    pub fn score_text_visible_impl(
        &self,
        index_key: &str,
        node_id: NodeId,
        query: &str,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Option<f64> {
        // Record the index read for anti-phantom SSI — must happen even when the
        // node ultimately scores 0.0 so that a Serializable scan that returns 0
        // results still closes the rw-antidependency cycle if a concurrent tx
        // inserts a matching document.
        self.record_read_index(tx, index_key);

        // Look up the per-tx delta entry for this specific node.
        // We use `get(index_key, node_id)` to avoid iterating all changes.
        let (delta_doc_opt, delta_removed): (
            Option<(u32, std::collections::HashMap<String, u32>)>,
            bool,
        ) = {
            let overlay = self.text_index_overlay.read();
            match overlay.get(&tx).and_then(|d| d.get(index_key, node_id)) {
                None => (None, false),
                Some(None) => (None, true), // tombstone
                Some(Some(text)) => {
                    let tokenizer = crate::index::text::SimpleTokenizer::new();
                    use crate::index::text::Tokenizer as _;
                    let tokens = tokenizer.tokenize(text);
                    #[allow(clippy::cast_possible_truncation)]
                    let doc_len = tokens.len() as u32;
                    let mut freq_map = std::collections::HashMap::new();
                    for t in tokens {
                        *freq_map.entry(t).or_insert(0u32) += 1;
                    }
                    (Some((doc_len, freq_map)), false)
                }
            }
        };

        // Look up the committed index.
        let committed_idx = {
            let text_indexes = self.text_indexes.read();
            text_indexes.get(index_key).cloned()
        };

        match committed_idx {
            Some(idx_arc) => {
                let idx = idx_arc.read();
                let delta_ref = delta_doc_opt.as_ref().map(|(dl, fm)| (*dl, fm));
                idx.score_document_visible(node_id, query, epoch, tx, delta_ref, delta_removed)
            }
            None => {
                // No committed index — score only if the node is a delta insert.
                if delta_removed {
                    return None;
                }
                let (_, freq_map) = delta_doc_opt?;
                // Build a transient single-document index for scoring.
                // Re-fetch the raw text from the overlay so we can use `insert`.
                let raw_text = {
                    let overlay = self.text_index_overlay.read();
                    overlay
                        .get(&tx)
                        .and_then(|d| d.get(index_key, node_id))
                        .and_then(|opt| opt.as_deref().map(str::to_owned))
                };
                // Suppress unused variable warning; the raw text is only needed if
                // we can still retrieve it (race-free since we hold no lock here, but
                // the overlay is append-only within a transaction so it will be present).
                let _ = freq_map;
                let text = raw_text?;
                let mut transient = crate::index::text::InvertedIndex::new(
                    crate::index::text::BM25Config::default(),
                );
                transient.insert(node_id, &text);
                let score = transient.score_document(node_id, query);
                if score > 0.0 { Some(score) } else { None }
            }
        }
    }

    // === Text-index GC ===

    /// Garbage collects versioned postings and aggregate-log entries in all
    /// text indexes that are no longer visible to any snapshot at or above
    /// `horizon`.
    ///
    /// Mirrors the per-field version-chain GC in `gc_versions`: the caller
    /// (the db-level `gc()`) provides the `min_active_epoch` so the text
    /// indexes compact in lock-step with the rest of the MVCC store.
    #[cfg(feature = "text-index")]
    pub fn gc_text_indexes(&self, horizon: EpochId) {
        let indexes = self.text_indexes.read();
        for idx in indexes.values() {
            idx.write().gc(horizon);
        }
    }

    // === Snapshot text search (TI4) ===

    /// Searches a text index at `(epoch, tx)`, merging committed postings with
    /// the transaction's buffered delta (read-your-writes).
    ///
    /// - Committed postings visible at `(epoch, tx)` that the tx has not
    ///   tombstoned are included.
    /// - Delta inserts (the tx's buffered `Some(text)` entries) appear with
    ///   their new text, replacing any committed posting for the same node.
    /// - Delta tombstones (`None` entries) suppress the committed posting.
    ///
    /// If no committed index exists for `index_key`, only delta inserts are
    /// searched (the delta is a self-contained mini corpus).
    ///
    /// The `index_key` format is `"label:property"`.
    #[cfg(feature = "text-index")]
    #[must_use]
    pub fn search_text_visible(
        &self,
        index_key: &str,
        query: &str,
        k: usize,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Vec<(NodeId, f64)> {
        // Coarse predicate-read recording for anti-phantom SSI (Task 7).
        // A Serializable text search reads every document matching `query` in
        // this index; a concurrent indexed SET is a phantom. Record the whole
        // index as read so the existing rw-detection can form the edge. This is
        // a no-op for SI/ReadCommitted (no read tracker registered for `tx`).
        self.record_read_index(tx, index_key);

        // Build delta_docs and delta_removed from the overlay for this tx.
        let (delta_docs, delta_removed): (Vec<(NodeId, String)>, FxHashSet<NodeId>) = {
            let overlay = self.text_index_overlay.read();
            match overlay.get(&tx) {
                None => (Vec::new(), FxHashSet::default()),
                Some(delta) => {
                    let mut docs = Vec::new();
                    let mut removed = FxHashSet::default();
                    for (node_id, opt_text) in delta.changes_for(index_key) {
                        match opt_text {
                            Some(text) => docs.push((node_id, text.clone())),
                            None => {
                                removed.insert(node_id);
                            }
                        }
                    }
                    (docs, removed)
                }
            }
        };

        // Look up the committed index.
        let committed_idx = {
            let text_indexes = self.text_indexes.read();
            text_indexes.get(index_key).cloned()
        };

        match committed_idx {
            Some(idx_arc) => {
                let idx = idx_arc.read();
                idx.search_visible(query, k, epoch, tx, &delta_docs, &delta_removed)
            }
            None => {
                // No committed index — search only the delta docs.
                if delta_docs.is_empty() {
                    return Vec::new();
                }
                // Build a transient index from the delta and search it.
                let mut transient = crate::index::text::InvertedIndex::new(
                    crate::index::text::BM25Config::default(),
                );
                for (node_id, text) in &delta_docs {
                    transient.insert(*node_id, text);
                }
                transient.search(query, k)
            }
        }
    }

    // === Transactional text-index buffering ===
    //
    // These two helpers are called from `set_node_property_buffered` /
    // `remove_node_property_buffered` (property_ops.rs) when the property is
    // covered by a text index.  They buffer the change into `text_index_overlay`
    // WITHOUT touching the committed `InvertedIndex`.

    /// Buffers a text-index set for a transactional node property write.
    ///
    /// For every label of `id` that has a text index on `key`, records
    /// `Some(text)` into `text_index_overlay[transaction_id]`.  Only called
    /// when the value is a `Value::String`; non-string values record a removal
    /// (the committed index path treats non-string as "remove").
    #[cfg(feature = "text-index")]
    pub(super) fn buffer_text_index_set(
        &self,
        id: NodeId,
        key: &str,
        value: &Value,
        transaction_id: TransactionId,
    ) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let label_set = node_labels.get(&id);
        #[cfg(feature = "temporal")]
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let index_key = format!("{label_name}:{key}");
                    if text_indexes.contains_key(&index_key) {
                        // Coarse index-write recording for anti-phantom SSI (Task 7).
                        // A transactional SET on an indexed property writes to this
                        // index; a concurrent Serializable text search is a phantom.
                        // Record the index write so the rw-detection can form the edge.
                        // No-op for SI/ReadCommitted (no write tracker registered).
                        self.record_write_index(transaction_id, &index_key);

                        let mut overlay = self.text_index_overlay.write();
                        let delta = overlay.entry(transaction_id).or_default();
                        match value {
                            Value::String(text) => {
                                delta.buffer_set(&index_key, id, text.to_string());
                            }
                            _ => {
                                // Non-string value: the index treats this as a removal
                                // (mirrors `update_text_index_on_set` which calls
                                // `idx.remove(id)` when the value is not a string).
                                delta.buffer_remove(&index_key, id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Buffers a text-index removal for a transactional node property removal.
    ///
    /// For every label of `id` that has a text index on `key`, records
    /// `None` (tombstone) into `text_index_overlay[transaction_id]`.
    #[cfg(feature = "text-index")]
    pub(super) fn buffer_text_index_remove(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let label_set = node_labels.get(&id);
        #[cfg(feature = "temporal")]
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let index_key = format!("{label_name}:{key}");
                    if text_indexes.contains_key(&index_key) {
                        // Coarse index-write recording for anti-phantom SSI (Task 7).
                        self.record_write_index(transaction_id, &index_key);

                        self.text_index_overlay
                            .write()
                            .entry(transaction_id)
                            .or_default()
                            .buffer_remove(&index_key, id);
                    }
                }
            }
        }
    }
}
