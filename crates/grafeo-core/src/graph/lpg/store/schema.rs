//! Schema, label, edge-type, and property-key methods for [`LpgStore`].

use super::{LpgStore, PropertyUndoEntry};
use arcstr::ArcStr;
#[cfg(feature = "temporal")]
use grafeo_common::types::EpochId;
use grafeo_common::types::{NodeId, TransactionId};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

impl LpgStore {
    /// Adds a label to a node.
    ///
    /// Returns true if the label was added, false if the node doesn't exist
    /// or already has the label.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let nodes = self.nodes.read();
        if let Some(chain) = nodes.get(&node_id) {
            if chain.visible_at(epoch).map_or(true, |r| r.is_deleted()) {
                return false;
            }
        } else {
            return false;
        }
        drop(nodes);

        // Get or create label ID
        let label_id = self.get_or_create_label_id(label);

        // Add to node_labels map
        let mut node_labels = self.node_labels.write();

        #[cfg(not(feature = "temporal"))]
        {
            let label_set = node_labels.entry(node_id).or_default();
            if label_set.contains(&label_id) {
                return false;
            }
            label_set.insert(label_id);
        }

        #[cfg(feature = "temporal")]
        {
            let current = node_labels
                .get(&node_id)
                .and_then(|log| log.latest())
                .cloned()
                .unwrap_or_default();
            if current.contains(&label_id) {
                return false;
            }
            let mut new_set = current;
            new_set.insert(label_id);
            node_labels
                .entry(node_id)
                .or_default()
                .append(self.current_epoch(), new_set);
        }

        drop(node_labels);

        // Add to label_index
        let mut index = self.label_index.write();
        if (label_id as usize) >= index.len() {
            index.resize(label_id as usize + 1, FxHashMap::default());
        }
        index[label_id as usize].insert(node_id, ());

        // Update label count in node record
        #[cfg(not(feature = "temporal"))]
        if let Some(chain) = self.nodes.write().get_mut(&node_id)
            && let Some(record) = chain.latest_mut()
        {
            let count = self.node_labels.read().get(&node_id).map_or(0, |s| s.len());
            record.set_label_count(u16::try_from(count).unwrap_or(u16::MAX));
        }

        true
    }

    /// Adds a label to a node.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let versions = self.node_versions.read();
        if let Some(index) = versions.get(&node_id) {
            if let Some(vref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&vref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
        drop(versions);

        // Get or create label ID
        let label_id = self.get_or_create_label_id(label);

        // Add to node_labels map
        let mut node_labels = self.node_labels.write();

        #[cfg(not(feature = "temporal"))]
        {
            let label_set = node_labels.entry(node_id).or_default();
            if label_set.contains(&label_id) {
                return false;
            }
            label_set.insert(label_id);
        }

        #[cfg(feature = "temporal")]
        {
            let current = node_labels
                .get(&node_id)
                .and_then(|log| log.latest())
                .cloned()
                .unwrap_or_default();
            if current.contains(&label_id) {
                return false;
            }
            let mut new_set = current;
            new_set.insert(label_id);
            node_labels
                .entry(node_id)
                .or_default()
                .append(self.current_epoch(), new_set);
        }

        drop(node_labels);

        // Add to label_index
        let mut index = self.label_index.write();
        if (label_id as usize) >= index.len() {
            index.resize(label_id as usize + 1, FxHashMap::default());
        }
        index[label_id as usize].insert(node_id, ());

        true
    }

    /// Removes a label from a node.
    ///
    /// Returns true if the label was removed, false if the node doesn't exist
    /// or doesn't have the label.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let nodes = self.nodes.read();
        if let Some(chain) = nodes.get(&node_id) {
            if chain.visible_at(epoch).map_or(true, |r| r.is_deleted()) {
                return false;
            }
        } else {
            return false;
        }
        drop(nodes);

        // Get label ID
        let label_id = {
            let reg = self.label_registry.read();
            match reg.get_id(label) {
                Some(id) => id,
                None => return false, // Label doesn't exist
            }
        };

        // Remove from node_labels map
        let mut node_labels = self.node_labels.write();

        #[cfg(not(feature = "temporal"))]
        {
            if let Some(label_set) = node_labels.get_mut(&node_id) {
                if !label_set.remove(&label_id) {
                    return false;
                }
            } else {
                return false;
            }
        }

        #[cfg(feature = "temporal")]
        {
            let current = node_labels
                .get(&node_id)
                .and_then(|log| log.latest())
                .cloned()
                .unwrap_or_default();
            if !current.contains(&label_id) {
                return false;
            }
            let mut new_set = current;
            new_set.remove(&label_id);
            node_labels
                .entry(node_id)
                .or_default()
                .append(self.current_epoch(), new_set);
        }

        drop(node_labels);

        // Remove from label_index
        let mut index = self.label_index.write();
        if (label_id as usize) < index.len() {
            index[label_id as usize].remove(&node_id);
        }

        // Update label count in node record
        #[cfg(not(feature = "temporal"))]
        if let Some(chain) = self.nodes.write().get_mut(&node_id)
            && let Some(record) = chain.latest_mut()
        {
            let count = self.node_labels.read().get(&node_id).map_or(0, |s| s.len());
            record.set_label_count(u16::try_from(count).unwrap_or(u16::MAX));
        }

        true
    }

    /// Removes a label from a node.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let versions = self.node_versions.read();
        if let Some(index) = versions.get(&node_id) {
            if let Some(vref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&vref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
        drop(versions);

        // Get label ID
        let label_id = {
            let reg = self.label_registry.read();
            match reg.get_id(label) {
                Some(id) => id,
                None => return false,
            }
        };

        // Remove from node_labels map
        let mut node_labels = self.node_labels.write();

        #[cfg(not(feature = "temporal"))]
        {
            if let Some(label_set) = node_labels.get_mut(&node_id) {
                if !label_set.remove(&label_id) {
                    return false;
                }
            } else {
                return false;
            }
        }

        #[cfg(feature = "temporal")]
        {
            let current = node_labels
                .get(&node_id)
                .and_then(|log| log.latest())
                .cloned()
                .unwrap_or_default();
            if !current.contains(&label_id) {
                return false;
            }
            let mut new_set = current;
            new_set.remove(&label_id);
            node_labels
                .entry(node_id)
                .or_default()
                .append(self.current_epoch(), new_set);
        }

        drop(node_labels);

        // Remove from label_index
        let mut index = self.label_index.write();
        if (label_id as usize) < index.len() {
            index[label_id as usize].remove(&node_id);
        }

        true
    }

    /// Returns all nodes with a specific label.
    ///
    /// Uses the label index for O(1) lookup per label. Returns a snapshot -
    /// concurrent modifications won't affect the returned vector. Results are
    /// sorted by NodeId for deterministic iteration order.
    pub fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let reg = self.label_registry.read();
        if let Some(label_id) = reg.get_id(label) {
            let index = self.label_index.read();
            if let Some(set) = index.get(label_id as usize) {
                let mut ids: Vec<NodeId> = set.keys().copied().collect();
                ids.sort_unstable();
                return ids;
            }
        }
        Vec::new()
    }

    /// Returns committed nodes with `label` merged with this transaction's
    /// buffered label delta — the writer-bypass for `MATCH (:Label)` scans.
    ///
    /// When `transaction_id` is `Some`, the result set starts from the
    /// committed `label_index` (the same source as `nodes_by_label`), then:
    /// - adds nodes for which this tx buffered a `LabelOp::Add` for this label;
    /// - removes nodes for which this tx buffered a `LabelOp::Remove`.
    ///
    /// When `transaction_id` is `None`, this is equivalent to `nodes_by_label`.
    ///
    /// **MVCC contract:** uncommitted labels are never inserted into
    /// `label_index` — the merge is a read-time view only.
    #[doc(hidden)]
    pub fn nodes_by_label_visible(
        &self,
        label: &str,
        transaction_id: Option<TransactionId>,
    ) -> Vec<NodeId> {
        // Committed base from label_index.
        let reg = self.label_registry.read();
        let Some(label_id) = reg.get_id(label) else {
            // Label not in registry at all — but the tx might have buffered it
            // (label id created by add_label_buffered via get_or_create_label_id).
            // Re-check after releasing the registry read lock.
            drop(reg);
            // Attempt to merge delta-only (committed set empty).
            if let Some(tx) = transaction_id {
                let overlay = self.tx_property_overlay.read();
                if let Some(delta) = overlay.get(&tx) {
                    // Find if any delta entry uses this label name.
                    // We must look up the id from the registry (now re-locked).
                    let reg2 = self.label_registry.read();
                    if let Some(lid) = reg2.get_id(label) {
                        drop(reg2);
                        let mut ids: Vec<NodeId> = delta
                            .node_labels
                            .iter()
                            .filter_map(|((nid, l), op)| {
                                if *l == lid {
                                    if matches!(op, super::LabelOp::Add) {
                                        Some(*nid)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        ids.sort_unstable();
                        for &id in &ids {
                            self.record_read_node(tx, id);
                        }
                        return ids;
                    }
                }
            }
            return Vec::new();
        };
        drop(reg);

        let index = self.label_index.read();
        let committed: std::collections::HashSet<NodeId> = index
            .get(label_id as usize)
            .map(|set| set.keys().copied().collect())
            .unwrap_or_default();
        drop(index);

        // Short-circuit for non-writing sessions.
        let Some(tx) = transaction_id else {
            let mut ids: Vec<NodeId> = committed.into_iter().collect();
            ids.sort_unstable();
            return ids;
        };

        // Merge delta.
        let mut visible: std::collections::HashSet<NodeId> = committed;
        {
            let overlay = self.tx_property_overlay.read();
            if let Some(delta) = overlay.get(&tx) {
                for ((nid, lid), op) in &delta.node_labels {
                    if *lid == label_id {
                        match op {
                            super::LabelOp::Add => {
                                visible.insert(*nid);
                            }
                            super::LabelOp::Remove => {
                                visible.remove(nid);
                            }
                        }
                    }
                }
            }
        }

        let mut ids: Vec<NodeId> = visible.into_iter().collect();
        ids.sort_unstable();
        // Record each returned node for the Serializable read-set.
        for &id in &ids {
            self.record_read_node(tx, id);
        }
        ids
    }

    /// Returns the number of nodes with a specific label without allocating
    /// the full ID list. O(1) via the label index.
    #[must_use]
    pub fn nodes_by_label_count(&self, label: &str) -> usize {
        let reg = self.label_registry.read();
        let Some(label_id) = reg.get_id(label) else {
            return 0;
        };
        self.label_index
            .read()
            .get(label_id as usize)
            .map_or(0, |set| set.len())
    }

    /// Returns the number of distinct labels in the store.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.label_registry.read().len()
    }

    /// Returns the number of distinct property keys in the store.
    ///
    /// This counts unique property keys across both nodes and edges.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        let node_keys = self.node_properties.column_count();
        let edge_keys = self.edge_properties.column_count();
        // Note: This may count some keys twice if the same key is used
        // for both nodes and edges. A more precise count would require
        // tracking unique keys across both storages.
        node_keys + edge_keys
    }

    /// Returns the number of distinct edge types in the store.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.id_to_edge_type.read().len()
    }

    /// Returns all label names in the database.
    pub fn all_labels(&self) -> Vec<String> {
        self.label_registry
            .read()
            .names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Returns all edge type names in the database.
    pub fn all_edge_types(&self) -> Vec<String> {
        self.id_to_edge_type
            .read()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Returns all property keys used in the database.
    pub fn all_property_keys(&self) -> Vec<String> {
        let mut keys = std::collections::HashSet::new();
        for key in self.node_properties.keys() {
            keys.insert(key.to_string());
        }
        for key in self.edge_properties.keys() {
            keys.insert(key.to_string());
        }
        keys.into_iter().collect()
    }

    /// Returns the next node ID that will be allocated.
    #[must_use]
    pub fn peek_next_node_id(&self) -> u64 {
        self.next_node_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the next edge ID that will be allocated.
    #[must_use]
    pub fn peek_next_edge_id(&self) -> u64 {
        self.next_edge_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Adds a label to a node within a transaction, recording the change
    /// in the undo log so it can be reversed on rollback.
    #[cfg(not(feature = "temporal"))]
    pub fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let added = self.add_label(node_id, label);
        if added {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::LabelAdded {
                    node_id,
                    label: label.to_string(),
                });
        }
        added
    }

    /// Adds a label to a node within a transaction (temporal version).
    ///
    /// Uses `EpochId::PENDING` for the version log entry, finalized on commit.
    #[cfg(feature = "temporal")]
    pub fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let label_id = self.get_or_create_label_id(label);

        let mut node_labels = self.node_labels.write();
        let current = node_labels
            .get(&node_id)
            .and_then(|log| log.latest())
            .cloned()
            .unwrap_or_default();
        if current.contains(&label_id) {
            return false;
        }
        let mut new_set = current;
        new_set.insert(label_id);
        node_labels
            .entry(node_id)
            .or_default()
            .append(EpochId::PENDING, new_set);
        drop(node_labels);

        // Update label_index
        let mut index = self.label_index.write();
        if (label_id as usize) >= index.len() {
            index.resize(label_id as usize + 1, FxHashMap::default());
        }
        index[label_id as usize].insert(node_id, ());

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::LabelAdded {
                node_id,
                label: label.to_string(),
            });

        true
    }

    /// Removes a label from a node within a transaction, recording the change
    /// in the undo log so it can be restored on rollback.
    #[cfg(not(feature = "temporal"))]
    pub fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let removed = self.remove_label(node_id, label);
        if removed {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::LabelRemoved {
                    node_id,
                    label: label.to_string(),
                });
        }
        removed
    }

    /// Removes a label from a node within a transaction (temporal version).
    #[cfg(feature = "temporal")]
    pub fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let label_id = {
            let reg = self.label_registry.read();
            match reg.get_id(label) {
                Some(id) => id,
                None => return false,
            }
        };

        let mut node_labels = self.node_labels.write();
        let current = node_labels
            .get(&node_id)
            .and_then(|log| log.latest())
            .cloned()
            .unwrap_or_default();
        if !current.contains(&label_id) {
            return false;
        }
        let mut new_set = current;
        new_set.remove(&label_id);
        node_labels
            .entry(node_id)
            .or_default()
            .append(EpochId::PENDING, new_set);
        drop(node_labels);

        // Update label_index
        let mut index = self.label_index.write();
        if (label_id as usize) < index.len() {
            index[label_id as usize].remove(&node_id);
        }

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::LabelRemoved {
                node_id,
                label: label.to_string(),
            });

        true
    }

    /// Looks up the numeric label id for a given name.
    ///
    /// Returns `None` if the label has never been interned.  Use this to
    /// check membership without creating a new registry entry.
    #[doc(hidden)]
    #[must_use]
    pub fn label_id(&self, name: &str) -> Option<u32> {
        self.label_registry.read().get_id(name)
    }

    /// Snapshot-consistent node label set: committed labels merged with the
    /// writing transaction's buffered label ops.
    ///
    /// With `transaction_id = Some(tx)` the writer's buffered `Add` ops are
    /// inserted and buffered `Remove` ops are deleted before returning.  With
    /// `transaction_id = None` (another session or auto-commit) only the
    /// committed set is returned.
    ///
    /// Returns label **names** (`ArcStr`) rather than numeric IDs so that the
    /// `GraphStore` trait default (which delegates to `get_node`) and this
    /// override agree on the return type and callers never receive an opaque ID.
    #[doc(hidden)]
    #[must_use]
    pub fn read_node_labels_visible(
        &self,
        id: NodeId,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashSet<ArcStr> {
        // --- Build the final id-set (committed base + buffered delta) ---

        // Committed base.
        #[cfg(not(feature = "temporal"))]
        let mut label_ids: FxHashSet<u32> = self
            .node_labels
            .read()
            .get(&id)
            .cloned()
            .unwrap_or_default();
        //
        // Writer-vs-other distinction (mirrors the property accessor):
        // a node inline-created within this transaction (`MERGE (:Item)` /
        // `CREATE (:Item)`) registers its labels directly into `node_labels`
        // at `EpochId::PENDING` rather than into the buffered overlay below.
        // The writing transaction must see that PENDING tail entry (so a later
        // UNWIND row's MERGE can dedupe against it), while other readers must
        // not. So for the writer (`Some(tx)`) read the latest entry — which is
        // the PENDING tail when uncommitted, exactly as `build_node` does —
        // and for others (`None`) read `at(epoch)`, which skips the PENDING
        // tail and yields only committed state. Without this, `at(real_epoch)`
        // dropped the writer's own inline-create label (empty set), breaking
        // MERGE-in-UNWIND dedup.
        #[cfg(feature = "temporal")]
        let mut label_ids: FxHashSet<u32> = {
            let node_labels = self.node_labels.read();
            let log = node_labels.get(&id);
            if transaction_id.is_some() {
                log.and_then(|log| log.latest().cloned())
                    .unwrap_or_default()
            } else {
                log.and_then(|log| log.at(epoch).cloned())
                    .unwrap_or_default()
            }
        };
        // Suppress the unused-variable warning for `epoch` under non-temporal.
        #[cfg(not(feature = "temporal"))]
        let _ = epoch;

        // Merge the writing transaction's buffered label delta.
        if let Some(tx) = transaction_id {
            // Record the node read: accessing a node's labels IS the read.
            self.record_read_node(tx, id);
            let overlay = self.tx_property_overlay.read();
            if let Some(delta) = overlay.get(&tx) {
                for ((nid, label_id), op) in &delta.node_labels {
                    if *nid == id {
                        match op {
                            super::LabelOp::Add => {
                                label_ids.insert(*label_id);
                            }
                            super::LabelOp::Remove => {
                                label_ids.remove(label_id);
                            }
                        }
                    }
                }
            }
        }

        // --- Convert numeric IDs → names via the registry ---
        let reg = self.label_registry.read();
        label_ids
            .iter()
            .filter_map(|&lid| reg.get_name(lid).cloned())
            .collect()
    }
}
