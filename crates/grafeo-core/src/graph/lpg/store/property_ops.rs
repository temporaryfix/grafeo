//! Property operations for the LPG store.

use super::LpgStore;
use super::PropertyUndoEntry;
use grafeo_common::types::EpochId;
use grafeo_common::types::{
    EdgeId, HashableValue, LabelId, NodeId, PropertyKey, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashMap;
use std::sync::atomic::Ordering;

#[cfg(test)]
std::thread_local! {
    // Count entries actually examined by whole-entity overlay materialization.
    static OVERLAY_PROPERTY_VISITS: std::cell::Cell<[usize; 2]> = const {
        std::cell::Cell::new([0, 0])
    };
}

#[cfg(test)]
fn record_overlay_property_visit(edge: bool) {
    OVERLAY_PROPERTY_VISITS.with(|visits| {
        let mut count = visits.get();
        count[usize::from(edge)] += 1;
        visits.set(count);
    });
}

#[cfg(test)]
pub(super) fn take_overlay_property_visits() -> [usize; 2] {
    OVERLAY_PROPERTY_VISITS.with(|visits| visits.replace([0, 0]))
}

impl LpgStore {
    /// Publishes the label-independent property-index phantom guard used by
    /// legacy versioned writers. Buffered engine writes use the same hook from
    /// their mutation operator; SYSTEM/recovery writes never enter SSI.
    fn record_property_index_write(&self, transaction_id: TransactionId, key: &str) {
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        if let Some(tracker) = self.write_trackers.read().get(&transaction_id).cloned() {
            tracker.record_property_index_write(transaction_id, key);
        }
    }

    /// Removes one value from the compatibility posting map without recording
    /// a history event. Rollback only removes PENDING property versions; its
    /// retained history must remain untouched.
    fn remove_property_index_current(
        &self,
        node_id: NodeId,
        key: &PropertyKey,
        value: Option<&Value>,
    ) {
        let Some(value) = value.filter(|value| !value.is_null()) else {
            return;
        };
        let indexes = self.property_indexes.read();
        let Some(index) = indexes.get(key) else {
            return;
        };
        let hash = HashableValue::new(value.clone());
        if let Some(mut nodes) = index.get_mut(&hash) {
            nodes.remove(&node_id);
            if nodes.is_empty() {
                drop(nodes);
                index.remove(&hash);
            }
        }
    }

    /// Adds one value to the compatibility posting map without recording a
    /// history event. This is the inverse of rollback removal above.
    fn add_property_index_current(
        &self,
        node_id: NodeId,
        key: &PropertyKey,
        value: Option<&Value>,
    ) {
        let Some(value) = value.filter(|value| !value.is_null()) else {
            return;
        };
        let indexes = self.property_indexes.read();
        let Some(index) = indexes.get(key) else {
            return;
        };
        index
            .entry(HashableValue::new(value.clone()))
            .or_default()
            .insert(node_id);
    }

    /// Sets a property on a node.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let prop_key: PropertyKey = key.into();

        // Update property index before setting the property (needs to read old value)
        self.update_property_index_on_set(id, &prop_key, &value);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_set(id, key, &value);

        self.node_properties
            .set(id, prop_key, value, self.current_epoch());

        #[cfg(feature = "vector-index")]
        self.refresh_vector_indexes_for_property(id, key);

        // Update props_count in record
        let count = u16::try_from(self.node_properties.get_all(id).len()).unwrap_or(u16::MAX);
        if let Some(chain) = self.nodes.write().get_mut(&id)
            && let Some(record) = chain.latest_mut()
        {
            record.props_count = count;
        }
    }

    /// Sets a property on a node.
    /// (Tiered storage version: properties stored separately, record is immutable)
    #[cfg(feature = "tiered-storage")]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let prop_key: PropertyKey = key.into();

        // Update property index before setting the property (needs to read old value)
        self.update_property_index_on_set(id, &prop_key, &value);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_set(id, key, &value);

        self.node_properties
            .set(id, prop_key, value, self.current_epoch());

        #[cfg(feature = "vector-index")]
        self.refresh_vector_indexes_for_property(id, key);
    }

    /// Sets a property on an edge.
    pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.edge_properties
            .set(id, key.into(), value, self.current_epoch());
    }

    /// Sets a node property at a specific epoch (for snapshot/WAL recovery).
    ///
    /// Maintains property membership at the supplied committed epoch. Text
    /// indexes retain their separate replay protocol.
    pub fn set_node_property_at_epoch(&self, id: NodeId, key: &str, value: Value, epoch: EpochId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let property = PropertyKey::new(key);
        self.update_property_index_on_set_at_epoch(id, &property, &value, epoch);
        self.node_properties.set(id, property, value, epoch);
    }

    /// Hydrates one authoritative property version into an overlay. The
    /// complete logical index is built from the source image by the caller;
    /// replaying this row must therefore not append history or touch local
    /// compatibility postings.
    #[cfg(feature = "compact-store")]
    pub(crate) fn hydrate_node_property_at_epoch(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        epoch: EpochId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.node_properties
            .set(id, PropertyKey::new(key), value, epoch);
    }

    /// Sets an edge property at a specific epoch (for snapshot/WAL recovery).
    pub fn set_edge_property_at_epoch(&self, id: EdgeId, key: &str, value: Value, epoch: EpochId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.edge_properties.set(id, key.into(), value, epoch);
    }

    /// Reconciles overlay-local metadata after same-incarnation hydration.
    ///
    /// [`Self::set_node_property_at_epoch`] intentionally updates only the
    /// authoritative version log. Compact-tier promotion replays the complete
    /// log through that API, then calls this once to populate overlay-local
    /// equality membership and refresh the mutable record's cached property
    /// count. Text and Vector indexes already span the complete logical graph:
    /// hydration must not retokenize, reinsert, or otherwise change their exact
    /// retained histories. This method never writes the property log.
    #[cfg(feature = "compact-store")]
    pub(crate) fn reconcile_node_overlay_metadata_after_hydration(
        &self,
        _id: NodeId,
        _keys: &[PropertyKey],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };

        #[cfg(not(feature = "tiered-storage"))]
        {
            let count = u16::try_from(self.node_properties.get_all(_id).len()).unwrap_or(u16::MAX);
            if let Some(chain) = self.nodes.write().get_mut(&_id)
                && let Some(record) = chain.latest_mut()
            {
                record.props_count = count;
            }
        }
    }

    /// Returns the denormalized property count carried by the latest node
    /// record. Test-only because callers must use the property store as the
    /// source of truth; this exposes the cached field solely so compact
    /// promotion tests can prove their derived metadata was reconciled.
    #[cfg(all(test, feature = "compact-store", not(feature = "tiered-storage")))]
    pub(crate) fn node_record_props_count_for_test(&self, id: NodeId) -> Option<u16> {
        self.nodes
            .read()
            .get(&id)
            .and_then(|chain| chain.latest())
            .map(|record| record.props_count)
    }

    /// Returns the full version history for all properties of a node.
    ///
    /// Each entry is `(key, Vec<(epoch, value)>)`. Used for temporal
    /// snapshot export.
    #[must_use]
    pub fn node_property_history(&self, id: NodeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        self.node_properties.get_all_history(id)
    }

    /// Returns a property value at a specific epoch.
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        self.node_properties.get_at(id, key, epoch)
    }

    /// Returns the version history for a single property of a node.
    #[must_use]
    pub fn node_property_history_for_key(&self, id: NodeId, key: &str) -> Vec<(EpochId, Value)> {
        self.node_properties.get_history(id, &PropertyKey::new(key))
    }

    /// Returns the full version history for all properties of an edge.
    #[must_use]
    pub fn edge_property_history(&self, id: EdgeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        self.edge_properties.get_all_history(id)
    }

    /// Removes a property from a node.
    ///
    /// Returns the previous value if it existed, or None if the property didn't exist.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let _mutation = self.pin_mutation()?;
        let prop_key: PropertyKey = key.into();

        // Update property index before removing (needs to read old value)
        self.update_property_index_on_remove(id, &prop_key);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_remove(id, key);

        let result = self
            .node_properties
            .remove(id, &prop_key, self.current_epoch());

        #[cfg(feature = "vector-index")]
        self.refresh_vector_indexes_for_property(id, key);

        // Update props_count in record
        let count = u16::try_from(self.node_properties.get_all(id).len()).unwrap_or(u16::MAX);
        if let Some(chain) = self.nodes.write().get_mut(&id)
            && let Some(record) = chain.latest_mut()
        {
            record.props_count = count;
        }

        result
    }

    /// Removes a property from a node.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let _mutation = self.pin_mutation()?;
        let prop_key: PropertyKey = key.into();

        // Update property index before removing (needs to read old value)
        self.update_property_index_on_remove(id, &prop_key);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_remove(id, key);

        let result = self
            .node_properties
            .remove(id, &prop_key, self.current_epoch());

        #[cfg(feature = "vector-index")]
        self.refresh_vector_indexes_for_property(id, key);

        result
    }

    /// Removes a property from an edge.
    ///
    /// Returns the previous value if it existed, or None if the property didn't exist.
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        let _mutation = self.pin_mutation()?;
        self.edge_properties
            .remove(id, &key.into(), self.current_epoch())
    }

    /// Gets a single property from a node without loading all properties.
    ///
    /// This is O(1) vs O(properties) for `get_node().get_property()`.
    /// Use this for filter predicates where you only need one property value.
    ///
    /// # Example
    ///
    /// ```
    /// # use grafeo_core::graph::lpg::LpgStore;
    /// # use grafeo_common::types::{PropertyKey, Value};
    /// let store = LpgStore::new().expect("arena allocation");
    /// let node_id = store.create_node(&["Person"]);
    /// store.set_node_property(node_id, "age", Value::from(30i64));
    ///
    /// // Fast: Direct single-property lookup
    /// let age = store.get_node_property(node_id, &PropertyKey::new("age"));
    ///
    /// // Slow: Loads all properties, then extracts one
    /// let age = store.get_node(node_id).and_then(|n| n.get_property("age").cloned());
    /// ```
    #[must_use]
    pub fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.node_properties.get(id, key)
    }

    /// Gets a single property from an edge without loading all properties.
    ///
    /// This is O(1) vs O(properties) for `get_edge().get_property()`.
    #[must_use]
    pub fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.edge_properties.get(id, key)
    }

    // === Batch Property Operations ===

    /// Gets a property for multiple nodes in a single batch operation.
    ///
    /// More efficient than calling [`Self::get_node_property`] in a loop because it
    /// reduces lock overhead and enables better cache utilization.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::{NodeId, PropertyKey, Value};
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// let n1 = store.create_node(&["Person"]);
    /// let n2 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "age", Value::from(25i64));
    /// store.set_node_property(n2, "age", Value::from(30i64));
    ///
    /// let ages = store.get_node_property_batch(&[n1, n2], &PropertyKey::new("age"));
    /// assert_eq!(ages, vec![Some(Value::from(25i64)), Some(Value::from(30i64))]);
    /// ```
    #[must_use]
    pub fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        self.node_properties.get_batch(ids, key)
    }

    /// Gets all properties for multiple nodes in a single batch operation.
    ///
    /// Returns a vector of property maps, one per node ID (empty map if no properties).
    /// More efficient than calling [`Self::get_node`] in a loop.
    #[must_use]
    pub fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.node_properties.get_all_batch(ids)
    }

    /// Gets selected properties for multiple nodes (projection pushdown).
    ///
    /// This is more efficient than [`Self::get_nodes_properties_batch`] when you only
    /// need a subset of properties. It only iterates the requested columns instead of
    /// all columns.
    ///
    /// **Use this for**: Queries with explicit projections like `RETURN n.name, n.age`
    /// instead of `RETURN n` (which requires all properties).
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::{PropertyKey, Value};
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// let n1 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "name", Value::from("Alix"));
    /// store.set_node_property(n1, "age", Value::from(30i64));
    /// store.set_node_property(n1, "email", Value::from("alix@example.com"));
    ///
    /// // Only fetch name and age (faster than get_nodes_properties_batch)
    /// let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    /// let props = store.get_nodes_properties_selective_batch(&[n1], &keys);
    ///
    /// assert_eq!(props[0].len(), 2); // Only name and age, not email
    /// ```
    #[must_use]
    pub fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.node_properties.get_selective_batch(ids, keys)
    }

    /// Gets selected properties for multiple edges (projection pushdown).
    ///
    /// Edge-property version of [`Self::get_nodes_properties_selective_batch`].
    #[must_use]
    pub fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.edge_properties.get_selective_batch(ids, keys)
    }

    // === Versioned Property Operations (with undo log) ===

    /// Sets a node property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let prop_key: PropertyKey = key.into();

        // Capture the current value before overwriting
        let old_value = self.node_properties.get(id, &prop_key);

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::NodeProperty {
                node_id: id,
                key: prop_key,
                old_value,
            });

        // Use PENDING epoch directly (finalized on commit)
        let prop_key2: PropertyKey = key.into();
        self.update_property_index_on_set_at_epoch(id, &prop_key2, &value, EpochId::PENDING);
        if transaction_id != TransactionId::SYSTEM {
            self.record_property_index_write(transaction_id, key);
        }
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        self.record_index_writes_for_node_property(id, key, transaction_id);
        #[cfg(feature = "text-index")]
        self.update_text_index_on_set(id, key, &value);
        self.node_properties
            .set(id, prop_key2, value, grafeo_common::types::EpochId::PENDING);
    }

    /// Sets an edge property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let prop_key: PropertyKey = key.into();

        // Capture the current value before overwriting
        let old_value = self.edge_properties.get(id, &prop_key);

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::EdgeProperty {
                edge_id: id,
                key: prop_key,
                old_value,
            });

        // Use PENDING epoch directly (finalized on commit)
        self.edge_properties.set(
            id,
            key.into(),
            value,
            grafeo_common::types::EpochId::PENDING,
        );
    }

    /// Removes a node property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let _mutation = self.pin_mutation()?;
        let prop_key: PropertyKey = key.into();

        // Capture the current value before removing
        let old_value = self.node_properties.get(id, &prop_key);

        // Only record if the property actually exists
        if old_value.is_some() {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::NodeProperty {
                    node_id: id,
                    key: prop_key.clone(),
                    old_value: old_value.clone(),
                });
        }

        // Keep the removal in the transaction's PENDING epoch. Calling the
        // unversioned operation here would close the committed value and
        // make the change visible to every reader before commit.
        self.update_property_index_on_remove_at_epoch(id, &prop_key, EpochId::PENDING);
        if transaction_id != TransactionId::SYSTEM {
            self.record_property_index_write(transaction_id, key);
        }
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        self.record_index_writes_for_node_property(id, key, transaction_id);
        #[cfg(feature = "text-index")]
        self.update_text_index_on_remove(id, key);
        let result = self.node_properties.remove(id, &prop_key, EpochId::PENDING);

        #[cfg(feature = "vector-index")]
        self.refresh_vector_indexes_for_property(id, key);

        #[cfg(not(feature = "tiered-storage"))]
        {
            let count = u16::try_from(self.node_properties.get_all(id).len()).unwrap_or(u16::MAX);
            if let Some(chain) = self.nodes.write().get_mut(&id)
                && let Some(record) = chain.latest_mut()
            {
                record.props_count = count;
            }
        }

        result
    }

    /// Removes an edge property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let _mutation = self.pin_mutation()?;
        let prop_key: PropertyKey = key.into();

        // Capture the current value before removing
        let old_value = self.edge_properties.get(id, &prop_key);

        // Only record if the property actually exists
        if old_value.is_some() {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::EdgeProperty {
                    edge_id: id,
                    key: prop_key,
                    old_value: old_value.clone(),
                });
        }

        // Delegate to the normal (unversioned) remove
        self.remove_edge_property(id, key)
    }

    /// Rolls back property/label changes by removing PENDING entries from
    /// version logs, and replays entity deletions from the undo log.
    ///
    /// With temporal properties, there is no need to replay old property
    /// values: `remove_pending()` pops the uncommitted PENDING entries
    /// from the back of each VersionLog, restoring the previous state.
    pub fn rollback_transaction_properties(&self, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let entries = self.property_undo_log.write().remove(&transaction_id);
        if let Some(entries) = entries {
            // Collect which node/edge properties and labels were touched
            let mut node_props: grafeo_common::utils::hash::FxHashSet<(NodeId, PropertyKey)> =
                grafeo_common::utils::hash::FxHashSet::default();
            let mut edge_props: grafeo_common::utils::hash::FxHashSet<(EdgeId, PropertyKey)> =
                grafeo_common::utils::hash::FxHashSet::default();
            let mut label_nodes: grafeo_common::utils::hash::FxHashSet<NodeId> =
                grafeo_common::utils::hash::FxHashSet::default();

            // Remove the transaction-visible posting for each touched value
            // before popping PENDING versions. This deliberately bypasses
            // the normal index helpers, which also append retained history.
            for entry in &entries {
                if let PropertyUndoEntry::NodeProperty { node_id, key, .. } = entry {
                    let value = self.node_properties.get(*node_id, key);
                    self.remove_property_index_current(*node_id, key, value.as_ref());
                }
            }

            // First pass: collect touched entries and handle entity deletions
            for entry in entries.into_iter().rev() {
                match entry {
                    PropertyUndoEntry::NodeProperty { node_id, key, .. } => {
                        node_props.insert((node_id, key));
                    }
                    PropertyUndoEntry::EdgeProperty { edge_id, key, .. } => {
                        edge_props.insert((edge_id, key));
                    }
                    PropertyUndoEntry::LabelAdded { node_id, .. }
                    | PropertyUndoEntry::LabelRemoved { node_id, .. } => {
                        label_nodes.insert(node_id);
                    }
                    PropertyUndoEntry::NodeDeleted {
                        node_id,
                        labels,
                        properties,
                    } => {
                        self.restore_deleted_node(node_id, transaction_id, &labels, properties);
                    }
                    PropertyUndoEntry::EdgeDeleted {
                        edge_id,
                        src,
                        dst,
                        edge_type,
                        properties,
                    } => {
                        self.restore_deleted_edge(
                            edge_id,
                            src,
                            dst,
                            transaction_id,
                            &edge_type,
                            properties,
                        );
                    }
                }
            }

            // Remove PENDING entries from affected property version logs
            if !node_props.is_empty() {
                let mut columns = self.node_properties.columns_write();
                for (node_id, key) in &node_props {
                    if let Some(col) = columns.get_mut(key) {
                        col.remove_pending_for(*node_id);
                    }
                }
            }

            // Re-add the restored current value, if any. Historical postings
            // were never changed by either rollback operation.
            for (node_id, key) in &node_props {
                let value = self.node_properties.get(*node_id, key);
                self.add_property_index_current(*node_id, key, value.as_ref());
            }

            if !edge_props.is_empty() {
                let mut columns = self.edge_properties.columns_write();
                for (edge_id, key) in &edge_props {
                    if let Some(col) = columns.get_mut(key) {
                        col.remove_pending_for(*edge_id);
                    }
                }
            }

            // Remove PENDING entries from affected label version logs and
            // reconcile label_index to match the restored state.
            if !label_nodes.is_empty() {
                let mut labels = self.node_labels.write();
                let mut index = self.label_index.write();

                for node_id in &label_nodes {
                    // Get the label set BEFORE removing PENDING (the transactional state)
                    let tx_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    // Remove PENDING entries to restore pre-transaction state
                    if let Some(log) = labels.get_mut(node_id) {
                        log.remove_pending();
                    }

                    // Get the restored (pre-transaction) label set
                    let restored_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    // Reconcile label_index: remove labels that were added by the
                    // transaction, re-add labels that were removed by the transaction.
                    for label_id in &tx_labels {
                        if !restored_labels.contains(label_id) && (*label_id as usize) < index.len()
                        {
                            index[*label_id as usize].remove(node_id);
                        }
                    }
                    for label_id in &restored_labels {
                        if !tx_labels.contains(label_id) && (*label_id as usize) < index.len() {
                            index[*label_id as usize].insert(*node_id, ());
                        }
                    }
                }
            }
        }
    }

    /// Discards the undo log entries for a committed transaction.
    ///
    /// Called during commit: properties are already written, so just
    /// clean up the log.
    pub fn commit_transaction_properties(&self, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.property_undo_log.write().remove(&transaction_id);
    }

    /// Returns the current number of undo log entries for a transaction.
    ///
    /// Used by savepoints to record the position so that partial rollback
    /// can replay only entries added after the savepoint.
    #[must_use]
    pub fn property_undo_log_position(&self, transaction_id: TransactionId) -> usize {
        self.property_undo_log
            .read()
            .get(&transaction_id)
            .map_or(0, Vec::len)
    }

    /// Rolls back property mutations recorded after position `since` in the undo log.
    ///
    /// Temporal version: instead of replaying old values (which would create
    /// new VersionLog entries), this pops the PENDING entries that were appended
    /// after the savepoint. Entity deletions are still restored via the normal
    /// `restore_deleted_node`/`restore_deleted_edge` helpers.
    pub fn rollback_transaction_properties_to(&self, transaction_id: TransactionId, since: usize) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let mut log = self.property_undo_log.write();
        if let Some(entries) = log.get_mut(&transaction_id)
            && since < entries.len()
        {
            let to_undo: Vec<PropertyUndoEntry> = entries.drain(since..).collect();
            drop(log);

            // Count how many PENDING entries to pop per (entity, key) and per label node.
            let mut node_prop_counts: grafeo_common::utils::hash::FxHashMap<
                (NodeId, PropertyKey),
                usize,
            > = grafeo_common::utils::hash::FxHashMap::default();
            let mut edge_prop_counts: grafeo_common::utils::hash::FxHashMap<
                (EdgeId, PropertyKey),
                usize,
            > = grafeo_common::utils::hash::FxHashMap::default();
            let mut label_counts: grafeo_common::utils::hash::FxHashMap<NodeId, usize> =
                grafeo_common::utils::hash::FxHashMap::default();

            for entry in &to_undo {
                match entry {
                    PropertyUndoEntry::NodeProperty { node_id, key, .. } => {
                        *node_prop_counts.entry((*node_id, key.clone())).or_default() += 1;
                    }
                    PropertyUndoEntry::EdgeProperty { edge_id, key, .. } => {
                        *edge_prop_counts.entry((*edge_id, key.clone())).or_default() += 1;
                    }
                    PropertyUndoEntry::LabelAdded { node_id, .. }
                    | PropertyUndoEntry::LabelRemoved { node_id, .. } => {
                        *label_counts.entry(*node_id).or_default() += 1;
                    }
                    PropertyUndoEntry::NodeDeleted { .. }
                    | PropertyUndoEntry::EdgeDeleted { .. } => {}
                }
            }

            // Savepoint rollback has the same current-map requirement as a
            // full rollback, while retaining PENDING entries created before
            // the savepoint. Purge only the touched key/value pairs.
            for (node_id, key) in node_prop_counts.keys() {
                let value = self.node_properties.get(*node_id, key);
                self.remove_property_index_current(*node_id, key, value.as_ref());
            }

            for entry in to_undo.into_iter().rev() {
                match entry {
                    PropertyUndoEntry::NodeProperty { .. }
                    | PropertyUndoEntry::EdgeProperty { .. }
                    | PropertyUndoEntry::LabelAdded { .. }
                    | PropertyUndoEntry::LabelRemoved { .. } => {}
                    PropertyUndoEntry::NodeDeleted {
                        node_id,
                        labels,
                        properties,
                    } => {
                        self.restore_deleted_node(node_id, transaction_id, &labels, properties);
                    }
                    PropertyUndoEntry::EdgeDeleted {
                        edge_id,
                        src,
                        dst,
                        edge_type,
                        properties,
                    } => {
                        self.restore_deleted_edge(
                            edge_id,
                            src,
                            dst,
                            transaction_id,
                            &edge_type,
                            properties,
                        );
                    }
                }
            }

            // Pop PENDING entries from node property version logs
            if !node_prop_counts.is_empty() {
                let mut columns = self.node_properties.columns_write();
                for ((node_id, key), count) in &node_prop_counts {
                    if let Some(col) = columns.get_mut(key) {
                        col.pop_n_pending_for(*node_id, *count);
                    }
                }
            }

            for (node_id, key) in node_prop_counts.keys() {
                let value = self.node_properties.get(*node_id, key);
                self.add_property_index_current(*node_id, key, value.as_ref());
            }

            // Pop PENDING entries from edge property version logs
            if !edge_prop_counts.is_empty() {
                let mut columns = self.edge_properties.columns_write();
                for ((edge_id, key), count) in &edge_prop_counts {
                    if let Some(col) = columns.get_mut(key) {
                        col.pop_n_pending_for(*edge_id, *count);
                    }
                }
            }

            // Pop PENDING entries from label version logs and reconcile label_index
            if !label_counts.is_empty() {
                let mut labels = self.node_labels.write();
                let mut index = self.label_index.write();

                for (node_id, count) in &label_counts {
                    let tx_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    if let Some(version_log) = labels.get_mut(node_id) {
                        version_log.pop_n_pending(*count);
                    }

                    let restored_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    // Reconcile label_index
                    for label_id in &tx_labels {
                        if !restored_labels.contains(label_id) && (*label_id as usize) < index.len()
                        {
                            index[*label_id as usize].remove(node_id);
                        }
                    }
                    for label_id in &restored_labels {
                        if !tx_labels.contains(label_id) && (*label_id as usize) < index.len() {
                            index[*label_id as usize].insert(*node_id, ());
                        }
                    }
                }
            }
        }
    }

    // === Deletion Restoration Helpers ===

    /// Restores a node that was deleted within a transaction.
    ///
    /// Called during rollback to undo a transactional node deletion.
    fn restore_deleted_node(
        &self,
        node_id: NodeId,
        transaction_id: TransactionId,
        labels: &[String],
        properties: Vec<(PropertyKey, Value)>,
    ) {
        // Unmark deletion on version chain
        #[cfg(not(feature = "tiered-storage"))]
        {
            let mut nodes = self.nodes.write();
            if let Some(chain) = nodes.get_mut(&node_id) {
                chain.unmark_deleted_by(transaction_id);
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            let mut versions = self.node_versions.write();
            if let Some(index) = versions.get_mut(&node_id) {
                index.unmark_deleted_by(transaction_id);
            }
        }

        // Restore label index entries
        for label in labels {
            self.add_label(node_id, label);
        }

        // Restore properties
        for (key, value) in properties {
            self.set_node_property(node_id, key.as_str(), value);
        }

        self.live_node_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Restores an edge that was deleted within a transaction.
    ///
    /// Called during rollback to undo a transactional edge deletion.
    fn restore_deleted_edge(
        &self,
        edge_id: EdgeId,
        src: NodeId,
        dst: NodeId,
        transaction_id: TransactionId,
        edge_type: &str,
        properties: Vec<(PropertyKey, Value)>,
    ) {
        // Unmark deletion on version chain
        #[cfg(not(feature = "tiered-storage"))]
        {
            let mut edges = self.edges.write();
            if let Some(chain) = edges.get_mut(&edge_id) {
                chain.unmark_deleted_by(transaction_id);
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            let mut versions = self.edge_versions.write();
            if let Some(index) = versions.get_mut(&edge_id) {
                index.unmark_deleted_by(transaction_id);
            }
        }

        // Restore adjacency (unmark soft-delete)
        self.forward_adj.unmark_deleted(src, edge_id);
        if let Some(ref backward) = self.backward_adj {
            backward.unmark_deleted(dst, edge_id);
        }

        // Restore properties
        for (key, value) in properties {
            self.set_edge_property(edge_id, key.as_str(), value);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);

        // Restore edge type count
        let type_id = {
            let type_map = self.edge_type_to_id.read();
            type_map.get(edge_type).copied()
        };
        if let Some(type_id) = type_id {
            self.increment_edge_type_count(type_id);
        }
    }

    // === Unified-MVCC first increment: per-transaction property delta ===
    //
    // Uncommitted property writes are buffered into a transaction's delta instead
    // of write-through to the committed column, and merged over it by the
    // snapshot-aware read accessor. This is the "hot tier / one read accessor"
    // foundation of the unified MVCC design. These methods are additive: the
    // existing write-through/undo-log path is unchanged until callers are routed
    // through the accessor in a later increment.

    /// Buffers an uncommitted node property write into the transaction's delta.
    ///
    /// If the property is covered by a text index for any of the node's labels,
    /// the change is also buffered into `text_index_overlay` so the committed
    /// [`InvertedIndex`] is NOT mutated on this path.
    #[doc(hidden)]
    pub fn set_node_property_buffered(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        // Property predicates survive index DROP/rebuild, so record every
        // tracked property write, independent of the current registration.
        let tracker = self.write_trackers.read().get(&transaction_id).cloned();
        if let Some(tracker) = tracker {
            tracker.record_property_index_write(transaction_id, key);
        }

        // Buffer the text-index change before moving `value`.
        #[cfg(feature = "text-index")]
        self.buffer_text_index_set(id, key, &value, transaction_id);

        // Coarse index-write recording for anti-phantom SSI — vector index path
        // (Task 5). For every label of `id` that has a vector index on `key`,
        // record the index write so the rw-detection can form the edge. No-op
        // for SI/ReadCommitted (no write tracker registered).
        #[cfg(feature = "vector-index")]
        self.buffer_vector_index_write_record(id, key, transaction_id);

        self.tx_property_overlay
            .write()
            .entry(transaction_id)
            .or_default()
            .node_props
            .insert((id, PropertyKey::new(key)), super::PropOp::Set(value));

        // Coarse SSI phantom-write fan-out: a property SET is a write to `id`, and
        // an escalated structural `Label(L)` reader records each scanned node as the
        // wildcard `(Node(n), None)` (compatible with any tag, including this
        // property write). Escalation drops the fine `Node(n)` SIREAD entry, so the
        // only way that reader is still caught is the coarse `Label(L)` write.
        //
        // We fan out ONLY the coarse Label(L) key (not the fine Node write): the
        // fine `Node(id)` write is already recorded by the SET operator under its
        // property tag (or completed at commit time), so re-recording a None-tagged
        // fine Node write here would be a wildcard that defeats Property
        // granularity. Uses the non-recording committed label set (NOT
        // read_node_labels_visible, which would pollute the writer's read-set).
        // Passes `key` so the tracker can carry Some(prop_tag(key)) on the coarse
        // write — the Part-G disjoint-property knob. No-op for SI/RC and SYSTEM.
        let label_ids = self.committed_node_label_ids(id);
        self.record_coarse_node_labels_only(transaction_id, &label_ids, key);
    }

    /// Buffers an uncommitted node property removal (tombstone) into the delta.
    ///
    /// If the property is covered by a text index for any of the node's labels,
    /// a removal tombstone is buffered into `text_index_overlay` (committed index
    /// is NOT touched).
    #[doc(hidden)]
    pub fn remove_node_property_buffered(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        // Property predicates survive index DROP/rebuild, so record every
        // tracked property write, independent of the current registration.
        let tracker = self.write_trackers.read().get(&transaction_id).cloned();
        if let Some(tracker) = tracker {
            tracker.record_property_index_write(transaction_id, key);
        }

        // Buffer the text-index removal.
        #[cfg(feature = "text-index")]
        self.buffer_text_index_remove(id, key, transaction_id);

        // Coarse index-write recording for anti-phantom SSI — vector index path
        // (Task 5). Mirrors the SET path above.
        #[cfg(feature = "vector-index")]
        self.buffer_vector_index_write_record(id, key, transaction_id);

        self.tx_property_overlay
            .write()
            .entry(transaction_id)
            .or_default()
            .node_props
            .insert((id, PropertyKey::new(key)), super::PropOp::Remove);

        // Coarse SSI phantom-write fan-out (see `set_node_property_buffered`): a
        // property REMOVE is also a write to `id` and must fan out ONLY the coarse
        // `Label(L)` (fine Node write already recorded under its property tag) so an
        // escalated structural reader (whose fine `Node(n)` entry was dropped) is
        // caught. Passes `key` for the Part-G disjoint-property knob.
        let label_ids = self.committed_node_label_ids(id);
        self.record_coarse_node_labels_only(transaction_id, &label_ids, key);
    }

    /// Buffers an uncommitted edge property write into the transaction's delta.
    #[doc(hidden)]
    pub fn set_edge_property_buffered(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.tx_property_overlay
            .write()
            .entry(transaction_id)
            .or_default()
            .edge_props
            .insert((id, PropertyKey::new(key)), super::PropOp::Set(value));

        // Coarse SSI phantom-write fan-out (edge mirror of the node path): a
        // property SET on an edge is a write to `id`, and an escalated
        // `RelType(T)` reader records each scanned edge as the wildcard
        // `(Edge(e), None)`. Escalation drops the fine `Edge(e)` SIREAD entry, so
        // the coarse `RelType(T)` write is the only way that reader is caught. Fan
        // out ONLY the coarse RelType(T) key (fine Edge write already recorded
        // under its property tag). `committed_edge_type_id` reads the edge record
        // with NO read recording. Passes `key` for the Part-G disjoint-property knob.
        if let Some(rel_type) = self.committed_edge_type_id(id) {
            self.record_coarse_edge_type_only(transaction_id, rel_type, key);
        }
    }

    /// Buffers an uncommitted edge property removal (tombstone) into the delta.
    #[doc(hidden)]
    pub fn remove_edge_property_buffered(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.tx_property_overlay
            .write()
            .entry(transaction_id)
            .or_default()
            .edge_props
            .insert((id, PropertyKey::new(key)), super::PropOp::Remove);

        // Coarse SSI phantom-write fan-out (see `set_edge_property_buffered`): a
        // property REMOVE on an edge is also a write to `id` and must fan out ONLY
        // the coarse `RelType(T)` (fine Edge write already recorded under its
        // property tag) so an escalated `RelType(T)` reader is caught.
        // Passes `key` for the Part-G disjoint-property knob.
        if let Some(rel_type) = self.committed_edge_type_id(id) {
            self.record_coarse_edge_type_only(transaction_id, rel_type, key);
        }
    }

    /// Snapshot-consistent node property read (the unified-MVCC read accessor).
    ///
    /// For the writing transaction the delta wins (read-your-writes): a buffered
    /// `Set` returns the value, a buffered `Remove` returns `None`. For everyone
    /// else (`transaction_id == None` — another session or auto-commit) it reads
    /// the committed column exactly as before, so uncommitted writes are never
    /// visible (no dirty reads).
    #[doc(hidden)]
    #[must_use]
    pub fn read_node_property_visible(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        let _read = self.pin_read();
        if let Some(tx) = transaction_id {
            // Record the property-level read with escalation: the engine bridge
            // routes through the node's committed labels so fine Node reads can
            // escalate to a coarse (Label(L), Some(prop_tag)) key when the tx has
            // already scanned that label. Falls back to a fine read when no labels
            // intersect with the transaction's scanned predicates.
            self.record_read_node_property_escalating(tx, id, key.as_str());
            let overlay = self.tx_property_overlay.read();
            if let Some(delta) = overlay.get(&tx)
                && let Some(op) = delta.node_props.get(&(id, key.clone()))
            {
                return match op {
                    super::PropOp::Set(v) => Some(v.clone()),
                    super::PropOp::Remove => None,
                };
            }
        }
        self.node_properties.get_at(id, key, epoch)
    }

    /// Nodes for which `transaction_id` has buffered an uncommitted write to
    /// `key`.
    ///
    /// The property index and the committed column hold committed values only, so
    /// a writer's own uncommitted `SET` can both create a match the index cannot
    /// know about and destroy one the index still reports. An index-backed lookup
    /// serving a writing transaction must union its index hits with this set and
    /// then confirm every candidate through
    /// [`read_node_property_visible`](Self::read_node_property_visible).
    ///
    /// Buffered `Remove`s are included: a removal is exactly the case where the
    /// index still reports a node that no longer matches.
    ///
    /// Sized by the transaction's own writes, never by the database, so this is
    /// not counted as scan work.
    #[doc(hidden)]
    #[must_use]
    pub fn nodes_with_buffered_property(
        &self,
        transaction_id: TransactionId,
        key: &PropertyKey,
    ) -> Vec<NodeId> {
        let _read = self.pin_read();
        let overlay = self.tx_property_overlay.read();
        let Some(delta) = overlay.get(&transaction_id) else {
            return Vec::new();
        };
        let mut ids: Vec<NodeId> = delta
            .node_props
            .keys()
            .filter(|(_, written)| written == key)
            .map(|(id, _)| *id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Snapshot-consistent edge property read. See
    /// [`read_node_property_visible`](Self::read_node_property_visible).
    #[doc(hidden)]
    #[must_use]
    pub fn read_edge_property_visible(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        let _read = self.pin_read();
        if let Some(tx) = transaction_id {
            // Record the property-level read with escalation: symmetric with
            // record_read_node_property_escalating — routes through the edge's
            // committed rel-type so fine Edge reads escalate to
            // (RelType(T), Some(prop_tag)) when the tx has scanned that type.
            self.record_read_edge_property_escalating(tx, id, key.as_str());
            let overlay = self.tx_property_overlay.read();
            if let Some(delta) = overlay.get(&tx)
                && let Some(op) = delta.edge_props.get(&(id, key.clone()))
            {
                return match op {
                    super::PropOp::Set(v) => Some(v.clone()),
                    super::PropOp::Remove => None,
                };
            }
        }
        self.edge_properties.get_at(id, key, epoch)
    }

    /// Overlays transaction `tid`'s buffered (uncommitted) property and label
    /// deltas onto an already-materialized `node`, giving a direct point read
    /// read-your-writes semantics.
    ///
    /// This is the whole-`Node` analogue of
    /// [`read_node_property_visible`](Self::read_node_property_visible): query
    /// execution reads properties one at a time through that accessor, but the
    /// direct lookup APIs (`get_node`, …) materialize a full `Node` from the
    /// committed columns, so they must apply the writing transaction's pending
    /// delta here to observe their own uncommitted writes. No-op for a
    /// transaction with no buffered ops (e.g. `SYSTEM`/auto-commit).
    #[doc(hidden)]
    pub fn apply_node_tx_delta(&self, node: &mut crate::graph::lpg::Node, tid: TransactionId) {
        let overlay = self.tx_property_overlay.read();
        let Some(delta) = overlay.get(&tid) else {
            return;
        };
        for ((_, key), op) in delta.node_props.for_entity(node.id) {
            #[cfg(test)]
            record_overlay_property_visit(false);
            match op {
                super::PropOp::Set(v) => {
                    node.properties.insert(key.clone(), v.clone());
                }
                super::PropOp::Remove => {
                    node.properties.remove(key);
                }
            }
        }
        if delta.node_labels.keys().any(|(nid, _)| *nid == node.id) {
            let registry = self.label_registry.read();
            for ((nid, label_id), op) in &delta.node_labels {
                if *nid != node.id {
                    continue;
                }
                let Some(name) = registry.get_name(*label_id) else {
                    continue;
                };
                match op {
                    super::LabelOp::Add => {
                        if !node.labels.iter().any(|l| l == name) {
                            node.labels.push(name.clone());
                        }
                    }
                    super::LabelOp::Remove => {
                        node.labels.retain(|l| l != name);
                    }
                }
            }
        }
    }

    /// Edge analogue of [`apply_node_tx_delta`](Self::apply_node_tx_delta):
    /// overlays `tid`'s buffered edge-property delta onto `edge`.
    #[doc(hidden)]
    pub fn apply_edge_tx_delta(&self, edge: &mut crate::graph::lpg::Edge, tid: TransactionId) {
        let overlay = self.tx_property_overlay.read();
        let Some(delta) = overlay.get(&tid) else {
            return;
        };
        for ((_, key), op) in delta.edge_props.for_entity(edge.id) {
            #[cfg(test)]
            record_overlay_property_visit(true);
            match op {
                super::PropOp::Set(v) => {
                    edge.properties.insert(key.clone(), v.clone());
                }
                super::PropOp::Remove => {
                    edge.properties.remove(key);
                }
            }
        }
    }

    /// Buffers an uncommitted label add into the transaction's delta.
    ///
    /// Uses the same `get_or_create_label_id` path as `add_label` so the
    /// buffered `u32` id agrees with `label_index` / `node_labels`.
    ///
    /// Also records a coarse `Label(L)` phantom write so a concurrent
    /// escalated `Label(L)` reader forms an rw-antidependency with this
    /// `SET n:L` operation.
    #[doc(hidden)]
    pub fn add_label_buffered(&self, id: NodeId, label: &str, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let label_id = self.get_or_create_label_id(label);
        self.tx_property_overlay
            .write()
            .entry(transaction_id)
            .or_default()
            .node_labels
            .insert((id, label_id), super::LabelOp::Add);
        // Phantom coarse write: adding label L to a node changes the :L set.
        self.record_coarse_node_write(transaction_id, id, &[LabelId::from(label_id)]);
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        self.record_index_writes_for_label(label, transaction_id);
    }

    /// Buffers an uncommitted label remove into the transaction's delta.
    ///
    /// If the label name is not yet in the registry (has never been used) the
    /// remove is a no-op: there is nothing to remove.
    ///
    /// Also records a coarse `Label(L)` phantom write so a concurrent escalated
    /// `Label(L)` reader forms an rw-antidependency with this `REMOVE n:L`
    /// operation: removing label L from a node changes the `:L` set, and an
    /// escalated reader (which dropped its fine `Node(n)` read) only catches the
    /// conflict via the coarse key. Mirrors the set-addition path
    /// [`add_label_buffered`](Self::add_label_buffered).
    #[doc(hidden)]
    pub fn remove_label_buffered(&self, id: NodeId, label: &str, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if let Some(label_id) = self.label_registry.read().get_id(label) {
            self.tx_property_overlay
                .write()
                .entry(transaction_id)
                .or_default()
                .node_labels
                .insert((id, label_id), super::LabelOp::Remove);
            // Phantom coarse write: removing label L from a node changes the :L set.
            self.record_coarse_node_write(transaction_id, id, &[LabelId::from(label_id)]);
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            self.record_index_writes_for_label(label, transaction_id);
        }
    }

    /// Applies a transaction's buffered property delta to the committed column
    /// (commit), then drops the delta.
    #[doc(hidden)]
    pub fn apply_tx_overlay(&self, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let delta = self.tx_property_overlay.write().remove(&transaction_id);
        if let Some(delta) = delta {
            let epoch = self.current_epoch();

            // Node properties: do per-op index maintenance first (it reads the
            // OLD value, before any change), then apply ALL value changes under a
            // SINGLE lock (`apply_ops`) so a concurrent reader observes the whole
            // commit atomically — never a half-applied state (the torn-read fix).
            let mut node_value_ops: Vec<(NodeId, PropertyKey, Option<Value>)> =
                Vec::with_capacity(delta.node_props.len());
            for ((id, key), op) in delta.node_props {
                match op {
                    super::PropOp::Set(v) => {
                        self.update_property_index_on_set(id, &key, &v);
                        #[cfg(feature = "text-index")]
                        self.update_text_index_on_set(id, key.as_str(), &v);
                        node_value_ops.push((id, key, Some(v)));
                    }
                    super::PropOp::Remove => {
                        self.update_property_index_on_remove(id, &key);
                        #[cfg(feature = "text-index")]
                        self.update_text_index_on_remove(id, key.as_str());
                        node_value_ops.push((id, key, None));
                    }
                }
            }
            // props_count is carried on the (mutable) node record only in the
            // non-tiered store; capture the affected nodes before `apply_ops`
            // consumes the op list, then refresh after applying.
            #[cfg(not(feature = "tiered-storage"))]
            let affected_nodes: grafeo_common::utils::hash::FxHashSet<NodeId> =
                node_value_ops.iter().map(|(id, _, _)| *id).collect();
            #[cfg(feature = "vector-index")]
            let affected_vector_properties: Vec<(NodeId, PropertyKey)> = node_value_ops
                .iter()
                .map(|(id, key, _)| (*id, key.clone()))
                .collect();
            self.node_properties.apply_ops(node_value_ops, epoch);
            #[cfg(feature = "vector-index")]
            for (id, key) in affected_vector_properties {
                self.refresh_vector_indexes_for_property(id, key.as_str());
            }
            #[cfg(not(feature = "tiered-storage"))]
            for id in affected_nodes {
                let count =
                    u16::try_from(self.node_properties.get_all(id).len()).unwrap_or(u16::MAX);
                if let Some(chain) = self.nodes.write().get_mut(&id)
                    && let Some(record) = chain.latest_mut()
                {
                    record.props_count = count;
                }
            }

            // Edge properties: apply VALUES atomically too (edges carry no
            // property index / props_count maintenance).
            let edge_value_ops: Vec<(EdgeId, PropertyKey, Option<Value>)> = delta
                .edge_props
                .into_iter()
                .map(|((id, key), op)| match op {
                    super::PropOp::Set(v) => (id, key, Some(v)),
                    super::PropOp::Remove => (id, key, None),
                })
                .collect();
            self.edge_properties.apply_ops(edge_value_ops, epoch);
            // Promote buffered label ops through the normal committed paths so
            // that both `node_labels` and `label_index` update consistently.
            for ((id, label_id), op) in delta.node_labels {
                let label_name = self.label_registry.read().get_name(label_id).cloned();
                if let Some(name) = label_name {
                    match op {
                        super::LabelOp::Add => {
                            self.add_label(id, name.as_str());
                        }
                        super::LabelOp::Remove => {
                            self.remove_label(id, name.as_str());
                        }
                    }
                }
            }
        }
        // TI5 (single-source): the committed `InvertedIndex` is promoted by the
        // property writes above — `set_node_property` / `remove_node_property`
        // each drive `update_text_index_on_set` / `_on_remove`, which stamp the
        // index posting at `current_epoch()`. By this point the engine has
        // already advanced the store epoch to the commit epoch `C` (via
        // `finalize_entities_by_id` -> `sync_epoch(C)`, which runs BEFORE this
        // function), so the index posting's epoch == the property version's
        // commit epoch `C`. The per-tx `text_index_overlay` (used only for the
        // tx's own `search_text_visible`) is therefore redundant at commit and is
        // simply dropped — re-promoting it would double-apply the update.
        #[cfg(feature = "text-index")]
        {
            self.text_index_overlay.write().remove(&transaction_id);
        }
    }

    /// Drops a transaction's buffered property delta (rollback) without applying.
    #[doc(hidden)]
    pub fn drop_tx_overlay(&self, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.tx_property_overlay.write().remove(&transaction_id);
        // Drop the text-index delta too (rollback — nothing to promote).
        #[cfg(feature = "text-index")]
        {
            self.text_index_overlay.write().remove(&transaction_id);
        }
    }

    /// Clones a transaction's buffered property delta for savepoint capture.
    ///
    /// Returns a clone of the current `TxDelta` for `transaction_id`, or an
    /// empty default delta if no overlay exists for this transaction yet.
    #[doc(hidden)]
    pub fn tx_overlay_snapshot(&self, transaction_id: TransactionId) -> super::TxDelta {
        self.tx_property_overlay
            .read()
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Restores a transaction's buffered property delta from a savepoint snapshot.
    ///
    /// Replaces the current overlay entry for `transaction_id` with `snapshot`.
    /// If the snapshot is empty (default), removes the entry entirely so that
    /// a subsequent `apply_tx_overlay` finds nothing buffered.
    #[doc(hidden)]
    pub fn tx_overlay_restore(&self, transaction_id: TransactionId, snapshot: super::TxDelta) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let mut overlay = self.tx_property_overlay.write();
        if snapshot.node_props.is_empty()
            && snapshot.edge_props.is_empty()
            && snapshot.node_labels.is_empty()
        {
            overlay.remove(&transaction_id);
        } else {
            overlay.insert(transaction_id, snapshot);
        }
    }

    /// Snapshot-consistent whole-node property map (whole-entity MVCC accessor).
    ///
    /// Returns all committed properties for `id`, then overlays the writing
    /// transaction's buffered delta for this node (`PropOp::Set` → insert,
    /// `PropOp::Remove` → remove).  When `transaction_id` is `None` (another
    /// session or auto-commit) the delta is never consulted, so the returned map
    /// is byte-for-byte what `node_properties.get_all(id)` returns today.
    ///
    /// This is the whole-entity form of [`read_node_property_visible`](Self::read_node_property_visible)
    /// used by `RETURN n` / `NodeResolve` materialization so that a writing
    /// transaction sees its own buffered writes in the materialized map.
    #[doc(hidden)]
    #[must_use]
    pub fn read_node_properties_visible(
        &self,
        id: NodeId,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashMap<PropertyKey, Value> {
        self.read_node_properties_visible_inner(id, epoch, transaction_id, true)
    }

    /// Materializes node properties for commit-time post-image validation
    /// without adding the validator's internal read to the transaction SSI
    /// read set. Buffered writes remain visible to the validator.
    #[doc(hidden)]
    pub fn read_node_properties_visible_for_validation(
        &self,
        id: NodeId,
        epoch: grafeo_common::types::EpochId,
        transaction_id: TransactionId,
    ) -> FxHashMap<PropertyKey, Value> {
        self.read_node_properties_visible_inner(id, epoch, Some(transaction_id), false)
    }

    fn read_node_properties_visible_inner(
        &self,
        id: NodeId,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<TransactionId>,
        track_read: bool,
    ) -> FxHashMap<PropertyKey, Value> {
        // Start from the committed whole-property map.
        let mut props = self.node_properties.get_all_at(id, epoch);

        // Overlay the writing transaction's buffered delta for this node.
        if let Some(tx) = transaction_id {
            // Record the node read through the escalation-aware materialization
            // path (symmetric with `read_edge_properties_visible`): a whole-node
            // `RETURN n` after a label scan escalated must short-circuit on the
            // coarse `Label(L)` key instead of re-adding a fine `(Node, _)` entry.
            if track_read {
                self.record_read_node_materialized(tx, id);
            }
            let overlay = self.tx_property_overlay.read();
            if let Some(delta) = overlay.get(&tx) {
                for ((_, key), op) in delta.node_props.for_entity(id) {
                    #[cfg(test)]
                    record_overlay_property_visit(false);
                    match op {
                        super::PropOp::Set(v) => {
                            props.insert(key.clone(), v.clone());
                        }
                        super::PropOp::Remove => {
                            props.remove(key);
                        }
                    }
                }
            }
        }
        props
    }

    /// Snapshot-consistent whole-edge property map (whole-entity MVCC accessor).
    ///
    /// Edge twin of [`read_node_properties_visible`](Self::read_node_properties_visible).
    #[doc(hidden)]
    #[must_use]
    pub fn read_edge_properties_visible(
        &self,
        id: EdgeId,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashMap<PropertyKey, Value> {
        self.read_edge_properties_visible_inner(id, epoch, transaction_id, true)
    }

    /// Materializes edge properties for commit-time post-image validation
    /// without adding the validator's internal read to the transaction SSI
    /// read set. Buffered writes remain visible to the validator.
    #[doc(hidden)]
    pub fn read_edge_properties_visible_for_validation(
        &self,
        id: EdgeId,
        epoch: grafeo_common::types::EpochId,
        transaction_id: TransactionId,
    ) -> FxHashMap<PropertyKey, Value> {
        self.read_edge_properties_visible_inner(id, epoch, Some(transaction_id), false)
    }

    fn read_edge_properties_visible_inner(
        &self,
        id: EdgeId,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<TransactionId>,
        track_read: bool,
    ) -> FxHashMap<PropertyKey, Value> {
        // Start from the committed whole-property map.
        let mut props = self.edge_properties.get_all_at(id, epoch);

        // Overlay the writing transaction's buffered delta for this edge.
        if let Some(tx) = transaction_id {
            // Record the edge read via the intrinsic-type chokepoint (Task 1)
            // so an already-escalated RelType short-circuits re-adding a fine entry.
            if track_read {
                if let Some(rel_type) = self.committed_edge_type_id(id) {
                    self.record_read_edge_in_rel_type(tx, id, rel_type);
                } else {
                    self.record_read_edge(tx, id); // typeless fallback (shouldn't happen)
                }
            }
            let overlay = self.tx_property_overlay.read();
            if let Some(delta) = overlay.get(&tx) {
                for ((_, key), op) in delta.edge_props.for_entity(id) {
                    #[cfg(test)]
                    record_overlay_property_visit(true);
                    match op {
                        super::PropOp::Set(v) => {
                            props.insert(key.clone(), v.clone());
                        }
                        super::PropOp::Remove => {
                            props.remove(key);
                        }
                    }
                }
            }
        }
        props
    }
}

#[cfg(test)]
mod pending_property_index_tests {
    use super::LpgStore;
    use crate::execution::operators::{OperatorError, SharedWriteTracker, WriteTracker};
    use crate::graph::{PropertyIndexPredicate, PropertyIndexRequest};
    use grafeo_common::types::{EpochId, TransactionId, Value};
    use std::sync::{Arc, Mutex};

    fn indexed_eq(
        store: &LpgStore,
        property: &str,
        value: &Value,
        epoch: EpochId,
    ) -> Vec<grafeo_common::types::NodeId> {
        store
            .lookup_nodes_indexed(PropertyIndexRequest {
                property,
                predicate: PropertyIndexPredicate::Equal(value),
                epoch,
                transaction_id: None,
            })
            .expect("registered equality index")
            .expect("property index should be available")
    }

    #[test]
    fn pending_set_full_rollback_restores_current_and_retained_postings() {
        let store = LpgStore::new().expect("store");
        let node = store.create_node(&[]);
        let old = Value::from("old");
        let new = Value::from("new");
        store.set_node_property(node, "state", old.clone());
        store.create_property_index("state");
        let epoch = store.current_epoch();
        let history = store.node_property_history_for_key(node, "state");

        let tx = TransactionId::new(41);
        store.set_node_property_versioned(node, "state", new.clone(), tx);
        assert_eq!(store.find_nodes_by_property("state", &new), vec![node]);
        store.rollback_transaction_properties(tx);

        assert_eq!(store.find_nodes_by_property("state", &old), vec![node]);
        assert!(store.find_nodes_by_property("state", &new).is_empty());
        assert_eq!(indexed_eq(&store, "state", &old, epoch), vec![node]);
        assert!(indexed_eq(&store, "state", &new, epoch).is_empty());
        assert_eq!(store.node_property_history_for_key(node, "state"), history);
    }

    #[test]
    fn pending_remove_full_rollback_restores_current_and_retained_postings() {
        let store = LpgStore::new().expect("store");
        let node = store.create_node(&[]);
        let value = Value::from("kept");
        store.set_node_property(node, "state", value.clone());
        store.create_property_index("state");
        let epoch = store.current_epoch();
        let history = store.node_property_history_for_key(node, "state");

        let tx = TransactionId::new(42);
        assert_eq!(
            store.remove_node_property_versioned(node, "state", tx),
            Some(value.clone())
        );
        assert!(store.find_nodes_by_property("state", &value).is_empty());
        store.rollback_transaction_properties(tx);

        assert_eq!(store.find_nodes_by_property("state", &value), vec![node]);
        assert_eq!(indexed_eq(&store, "state", &value, epoch), vec![node]);
        assert_eq!(store.node_property_history_for_key(node, "state"), history);
    }

    #[test]
    fn pending_remove_savepoint_rollback_restores_prior_pending_value() {
        let store = LpgStore::new().expect("store");
        let node = store.create_node(&[]);
        let old = Value::from("old");
        let pending = Value::from("pending");
        store.set_node_property(node, "state", old);
        store.create_property_index("state");

        let tx = TransactionId::new(43);
        store.set_node_property_versioned(node, "state", pending.clone(), tx);
        let savepoint = store.property_undo_log_position(tx);
        assert_eq!(
            store.remove_node_property_versioned(node, "state", tx),
            Some(pending.clone())
        );
        assert!(store.find_nodes_by_property("state", &pending).is_empty());
        store.rollback_transaction_properties_to(tx, savepoint);

        assert_eq!(store.find_nodes_by_property("state", &pending), vec![node]);
        store.rollback_transaction_properties(tx);
        assert_eq!(
            store.find_nodes_by_property("state", &Value::from("old")),
            vec![node]
        );
    }

    #[test]
    fn finalized_pending_set_preserves_historical_equal_and_in_queries() {
        let store = LpgStore::new().expect("store");
        let node = store.create_node(&[]);
        let old = Value::from("old");
        let new = Value::from("new");
        store.set_node_property(node, "state", old.clone());
        store.create_property_index("state");
        let before = store.current_epoch();
        let tx = TransactionId::new(44);
        store.set_node_property_versioned(node, "state", new.clone(), tx);
        let committed = EpochId::new(before.as_u64().saturating_add(1));

        store.finalize_property_index_history(tx, committed);
        store.node_properties.finalize_pending(committed);
        store.commit_transaction_properties(tx);
        store.sync_epoch(committed);

        assert_eq!(indexed_eq(&store, "state", &old, before), vec![node]);
        assert!(indexed_eq(&store, "state", &old, committed).is_empty());
        assert_eq!(indexed_eq(&store, "state", &new, committed), vec![node]);
        let values = [old.clone(), new.clone()];
        assert_eq!(
            store
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "state",
                    predicate: PropertyIndexPredicate::In(&values),
                    epoch: before,
                    transaction_id: None,
                })
                .expect("registered IN index")
                .expect("property index should be available"),
            vec![node]
        );
        assert_eq!(
            store
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "state",
                    predicate: PropertyIndexPredicate::In(&values),
                    epoch: committed,
                    transaction_id: None,
                })
                .expect("registered IN index")
                .expect("property index should be available"),
            vec![node]
        );
    }

    struct PropertyWriteSpy(Mutex<Vec<String>>);

    impl WriteTracker for PropertyWriteSpy {
        fn record_node_write(
            &self,
            _transaction_id: TransactionId,
            _node_id: grafeo_common::types::NodeId,
        ) -> Result<(), OperatorError> {
            Ok(())
        }

        fn record_edge_write(
            &self,
            _transaction_id: TransactionId,
            _edge_id: grafeo_common::types::EdgeId,
        ) -> Result<(), OperatorError> {
            Ok(())
        }

        fn record_property_index_write(&self, transaction_id: TransactionId, property: &str) {
            self.0
                .lock()
                .expect("spy lock")
                .push(format!("{transaction_id:?}:{property}"));
        }
    }

    #[test]
    fn versioned_property_writers_publish_property_guard_except_system() {
        let store = LpgStore::new().expect("store");
        let node = store.create_node(&[]);
        store.set_node_property(node, "state", Value::from("old"));
        store.create_property_index("state");

        let tx = TransactionId::new(4_501);
        let spy = Arc::new(PropertyWriteSpy(Mutex::new(Vec::new())));
        store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);
        store.set_node_property_versioned(node, "state", Value::from("new"), tx);
        store.remove_node_property_versioned(node, "state", tx);
        assert_eq!(spy.0.lock().expect("spy lock").len(), 2);
        store.unregister_write_tracker(tx);

        let system_spy = Arc::new(PropertyWriteSpy(Mutex::new(Vec::new())));
        store.register_write_tracker(
            TransactionId::SYSTEM,
            Arc::clone(&system_spy) as SharedWriteTracker,
        );
        store.set_node_property_versioned(
            node,
            "state",
            Value::from("recovery"),
            TransactionId::SYSTEM,
        );
        assert!(system_spy.0.lock().expect("spy lock").is_empty());
        store.unregister_write_tracker(TransactionId::SYSTEM);
    }
}
