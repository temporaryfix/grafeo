//! Node and edge CRUD operations for GrafeoDB.

use crate::config::GraphModel;
use grafeo_common::types::NodeId;
#[cfg(feature = "compact-store")]
use grafeo_core::graph::traits::GraphStore;

impl super::GrafeoDB {
    /// LPG mutations are closed on an RDF-labelled database and after WAL poison.
    pub(super) fn lpg_mutation_closed(&self) -> bool {
        self.config.graph_model == GraphModel::Rdf || self.is_durability_poisoned()
    }

    // === Node Operations ===

    /// Creates a node with the given labels and returns its ID.
    ///
    /// Labels categorize nodes - think of them like tags. A node can have
    /// multiple labels (e.g., `["Person", "Employee"]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"]);
    /// let company = db.create_node(&["Company", "Startup"]);
    /// ```
    pub fn create_node(&self, labels: &[&str]) -> grafeo_common::types::NodeId {
        if self.lpg_mutation_closed() {
            return NodeId::INVALID;
        }
        self.session().create_node(labels)
    }

    /// Creates a new node with labels and properties.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<
            Item = (
                impl Into<grafeo_common::types::PropertyKey>,
                impl Into<grafeo_common::types::Value>,
            ),
        >,
    ) -> grafeo_common::types::NodeId {
        if self.lpg_mutation_closed() {
            return NodeId::INVALID;
        }
        // Collect properties first so we can log them to WAL
        let props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = properties
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let Ok(id) = self.session().create_node_with_props(
            labels,
            props
                .iter()
                .map(|(key, value)| (key.as_str(), value.clone())),
        ) else {
            return NodeId::INVALID;
        };

        id
    }

    /// Gets a node by ID.
    #[must_use]
    pub fn get_node(
        &self,
        id: grafeo_common::types::NodeId,
    ) -> Option<grafeo_core::graph::lpg::Node> {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view().get_node(id)
    }

    /// Gets a node as it existed at a specific epoch.
    ///
    /// Uses pure epoch-based visibility (not transaction-aware), so the node
    /// is visible if and only if `created_epoch <= epoch` and it was not
    /// deleted at or before `epoch`.
    ///
    /// After [`compact()`](Self::compact) this
    /// routes through the `LayeredStore`, which combines the temporal cold
    /// base's as-of view with the overlay — so history folded into the cold base
    /// is visible. Without a layered store, it reads the built-in `LpgStore`.
    #[must_use]
    pub fn get_node_at_epoch(
        &self,
        id: grafeo_common::types::NodeId,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_core::graph::lpg::Node> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.get_node_at_epoch(id, epoch);
        }
        self.lpg_store().get_node_at_epoch(id, epoch)
    }

    /// Gets an edge as it existed at a specific epoch.
    ///
    /// Uses pure epoch-based visibility (not transaction-aware). Routes through
    /// the `LayeredStore` (cold base + overlay) when present, like
    /// [`get_node_at_epoch`](Self::get_node_at_epoch).
    #[must_use]
    pub fn get_edge_at_epoch(
        &self,
        id: grafeo_common::types::EdgeId,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_core::graph::lpg::Edge> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.get_edge_at_epoch(id, epoch);
        }
        self.lpg_store().get_edge_at_epoch(id, epoch)
    }

    /// Neighbors of `node` visible at `epoch`.
    ///
    /// `epoch == `[`EpochId::PENDING`](grafeo_common::types::EpochId::PENDING)
    /// is the current 1-hop: the derived open CSR plus overlay. Any other
    /// epoch filters structural validity. A closed (deleted) edge is absent
    /// at and after its delete epoch.
    ///
    /// After [`compact()`](Self::compact) this is the `LayeredStore` as-of
    /// path (packed base ∪ overlay). Before compact it uses the overlay
    /// version log (`neighbors_versioned`).
    #[must_use]
    pub fn neighbors_at_epoch(
        &self,
        node: grafeo_common::types::NodeId,
        direction: grafeo_core::graph::Direction,
        epoch: grafeo_common::types::EpochId,
    ) -> Vec<grafeo_common::types::NodeId> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.neighbors_at_epoch(node, direction, epoch);
        }
        self.lpg_store().neighbors_versioned(
            node,
            direction,
            epoch,
            grafeo_common::types::TransactionId::INVALID,
        )
    }

    /// Fills `out` with neighbors visible at `epoch` (clears `out` first).
    ///
    /// Prefer this over [`Self::neighbors_at_epoch`] when expanding many sources.
    pub fn fill_neighbors_at_epoch(
        &self,
        node: grafeo_common::types::NodeId,
        direction: grafeo_core::graph::Direction,
        epoch: grafeo_common::types::EpochId,
        out: &mut Vec<grafeo_common::types::NodeId>,
    ) {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            layered.fill_neighbors_at_epoch(node, direction, epoch, out);
            return;
        }
        out.clear();
        out.extend(self.lpg_store().neighbors_versioned(
            node,
            direction,
            epoch,
            grafeo_common::types::TransactionId::INVALID,
        ));
    }

    /// Dest-only as-of fill restricted to `types` (empty = every edge type).
    pub fn fill_neighbors_of_types_at_epoch(
        &self,
        node: grafeo_common::types::NodeId,
        direction: grafeo_core::graph::Direction,
        epoch: grafeo_common::types::EpochId,
        types: &[String],
        out: &mut Vec<grafeo_common::types::NodeId>,
    ) {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            layered.fill_neighbors_of_types_at_epoch(node, direction, epoch, types, out);
            return;
        }
        self.lpg_store()
            .fill_neighbors_of_types_at_epoch(node, direction, epoch, types, out);
    }

    /// Every edge visible at `epoch`.
    ///
    /// `PENDING` is the current snapshot (open CSR ∪ overlay). Closed edges
    /// are absent at and after their delete epoch. After compact this
    /// composes the temporal cold base with the overlay; before compact it
    /// walks the overlay version log.
    #[must_use]
    pub fn edges_at_epoch(
        &self,
        epoch: grafeo_common::types::EpochId,
    ) -> Vec<grafeo_core::graph::lpg::Edge> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.edges_at_epoch(epoch);
        }
        let store = self.live_overlay();
        let mut ids = store.all_known_edge_ids();
        ids.sort_unstable();
        ids.into_iter()
            .filter_map(|id| store.get_edge_at_epoch(id, epoch))
            .collect()
    }

    /// Every node visible at `epoch` (materialized `Node`s).
    ///
    /// After compact this is [`grafeo_core::graph::compact::layered::LayeredStore::nodes_at_epoch`]; otherwise it
    /// evaluates [`get_node_at_epoch`](Self::get_node_at_epoch) for every
    /// retained identity, including nodes deleted after the requested cut.
    #[must_use]
    pub fn nodes_at_epoch(
        &self,
        epoch: grafeo_common::types::EpochId,
    ) -> Vec<grafeo_core::graph::lpg::Node> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.nodes_at_epoch(epoch);
        }
        let store = self.lpg_store();
        store
            .all_node_ids()
            .into_iter()
            .filter_map(|id| store.get_node_at_epoch(id, epoch))
            .collect()
    }

    /// Whole-state as-of "scrub": node frames plus edge frames at `epoch`.
    ///
    /// Node frames stay columnar (per label: ids + per-property columns
    /// aligned by offset) — the allocation-light 60fps path. Edge frames
    /// are per relationship type (`edge_ids` / `src_ids` / `dst_ids` +
    /// property columns).
    ///
    /// `PENDING` is the current snapshot. Requires a compacted
    /// (`LayeredStore`) database; returns an empty [`grafeo_core::graph::compact::GraphScrub`] when
    /// nothing has been compacted yet.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    #[must_use]
    pub fn scrub_at_epoch(
        &self,
        epoch: grafeo_common::types::EpochId,
    ) -> grafeo_core::graph::compact::GraphScrub {
        let _publication = self.transaction_manager.publication().read();
        self.layered_store
            .as_ref()
            .map(|layered| layered.graph_scrub_at_epoch(epoch))
            .unwrap_or_default()
    }

    /// Returns all structural lifetimes of a node, newest first.
    ///
    /// Each materialized node contains the labels and properties visible at its
    /// creation epoch. History remains available after temporal compaction.
    #[must_use]
    pub fn get_node_history(
        &self,
        id: grafeo_common::types::NodeId,
    ) -> Vec<(
        grafeo_common::types::EpochId,
        Option<grafeo_common::types::EpochId>,
        grafeo_core::graph::lpg::Node,
    )> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.complete_node_history(id);
        }
        self.live_overlay().get_node_history(id)
    }

    /// Returns all structural lifetimes of an edge, newest first.
    ///
    /// Each materialized edge contains the properties visible at its creation
    /// epoch. History remains available after temporal compaction.
    #[must_use]
    pub fn get_edge_history(
        &self,
        id: grafeo_common::types::EdgeId,
    ) -> Vec<(
        grafeo_common::types::EpochId,
        Option<grafeo_common::types::EpochId>,
        grafeo_core::graph::lpg::Edge,
    )> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.complete_edge_history(id);
        }
        self.live_overlay().get_edge_history(id)
    }

    /// Returns a property value as it existed at a specific epoch.
    ///
    /// Uses the internal `VersionLog` to do a point-in-time read. Returns
    /// `None` if the property didn't exist or was deleted at that epoch.
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: grafeo_common::types::NodeId,
        key: &str,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_common::types::Value> {
        let _publication = self.transaction_manager.publication().read();
        let prop_key = grafeo_common::types::PropertyKey::new(key);
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.get_node_property_at_epoch(id, &prop_key, epoch);
        }
        self.live_overlay()
            .get_node_property_at_epoch(id, &prop_key, epoch)
    }

    /// Returns the full version timeline for a single property of a node.
    ///
    /// Each entry is `(epoch, value)` in ascending epoch order. Tombstones
    /// (deletions) appear as `Value::Null`.
    #[must_use]
    pub fn get_node_property_history(
        &self,
        id: grafeo_common::types::NodeId,
        key: &str,
    ) -> Vec<(grafeo_common::types::EpochId, grafeo_common::types::Value)> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.complete_node_property_history_for_key(id, key);
        }
        self.live_overlay().node_property_history_for_key(id, key)
    }

    /// Returns the full version history for ALL properties of a node.
    ///
    /// Each entry is `(property_key, Vec<(epoch, value)>)`.
    #[must_use]
    pub fn get_all_node_property_history(
        &self,
        id: grafeo_common::types::NodeId,
    ) -> Vec<(
        grafeo_common::types::PropertyKey,
        Vec<(grafeo_common::types::EpochId, grafeo_common::types::Value)>,
    )> {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if let Some(ref layered) = self.layered_store {
            return layered.complete_node_property_history(id);
        }
        let mut history = self.live_overlay().node_property_history(id);
        history.sort_unstable_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
        history
    }

    /// Atomically deletes a node and all of its incident edges.
    ///
    /// Incoming, outgoing, and self-loop edges share the node's transaction and
    /// WAL outcome. On failure this returns `false`; no proper prefix of the
    /// detach can be committed or recovered.
    pub fn delete_node(&self, id: grafeo_common::types::NodeId) -> bool {
        if self.lpg_mutation_closed() {
            return false;
        }
        // The session's CdcGraphStore captures the pre-image and emits the
        // node plus incident-edge events at the transaction's commit epoch.
        // Recording again here would duplicate the node Delete event.
        self.session().delete_node(id)
    }

    /// Sets a property on a node.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    ///
    /// # Errors
    ///
    /// Returns an error if the node is not visible, the value or post-image
    /// violates a configured limit or schema, or the write cannot be committed.
    pub fn set_node_property(
        &self,
        id: grafeo_common::types::NodeId,
        key: &str,
        value: grafeo_common::types::Value,
    ) -> grafeo_common::utils::error::Result<()> {
        // CdcGraphStore owns the single transactional before/after event.
        self.session().set_node_property(id, key, value)
    }

    /// Adds a label to an existing node.
    ///
    /// Returns `true` if the label was added, `false` if the node doesn't exist
    /// or already has the label.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"]);
    ///
    /// // Promote Alix to Employee
    /// let added = db.add_node_label(alix, "Employee");
    /// assert!(added);
    /// ```
    pub fn add_node_label(&self, id: grafeo_common::types::NodeId, label: &str) -> bool {
        if self.lpg_mutation_closed() {
            return false;
        }
        self.session().add_node_label(id, label)
    }

    /// Removes a label from a node.
    ///
    /// Returns `true` if the label was removed, `false` if the node doesn't exist
    /// or doesn't have the label.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person", "Employee"]);
    ///
    /// // Remove Employee status
    /// let removed = db.remove_node_label(alix, "Employee");
    /// assert!(removed);
    /// ```
    pub fn remove_node_label(&self, id: grafeo_common::types::NodeId, label: &str) -> bool {
        if self.lpg_mutation_closed() {
            return false;
        }
        self.session().remove_node_label(id, label)
    }

    /// Gets all labels for a node.
    ///
    /// Returns `None` if the node doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person", "Employee"]);
    ///
    /// let labels = db.get_node_labels(alix).unwrap();
    /// assert!(labels.contains(&"Person".to_string()));
    /// assert!(labels.contains(&"Employee".to_string()));
    /// ```
    #[must_use]
    pub fn get_node_labels(&self, id: grafeo_common::types::NodeId) -> Option<Vec<String>> {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view()
            .get_node(id)
            .map(|node| node.labels.iter().map(|s| s.to_string()).collect())
    }

    // === Edge Operations ===

    /// Creates an edge (relationship) between two nodes.
    ///
    /// Edges connect nodes and have a type that describes the relationship.
    /// They're directed - the order of `src` and `dst` matters.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alix = db.create_node(&["Person"]);
    /// let gus = db.create_node(&["Person"]);
    ///
    /// // Alix knows Gus (directed: Alix -> Gus)
    /// let edge = db.create_edge(alix, gus, "KNOWS");
    /// ```
    pub fn create_edge(
        &self,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
    ) -> grafeo_common::types::EdgeId {
        if self.lpg_mutation_closed() {
            return grafeo_common::types::EdgeId::INVALID;
        }
        self.session().create_edge(src, dst, edge_type)
    }

    /// Creates a new edge with properties.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn create_edge_with_props(
        &self,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
        properties: impl IntoIterator<
            Item = (
                impl Into<grafeo_common::types::PropertyKey>,
                impl Into<grafeo_common::types::Value>,
            ),
        >,
    ) -> grafeo_common::types::EdgeId {
        if self.lpg_mutation_closed() {
            return grafeo_common::types::EdgeId::INVALID;
        }
        // Collect properties first so we can log them to WAL
        let props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = properties
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let Ok(id) = self.session().create_edge_with_props(
            src,
            dst,
            edge_type,
            props
                .iter()
                .map(|(key, value)| (key.as_str(), value.clone())),
        ) else {
            return grafeo_common::types::EdgeId::INVALID;
        };

        id
    }

    /// Gets an edge by ID.
    #[must_use]
    pub fn get_edge(
        &self,
        id: grafeo_common::types::EdgeId,
    ) -> Option<grafeo_core::graph::lpg::Edge> {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view().get_edge(id)
    }

    /// Deletes an edge.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn delete_edge(&self, id: grafeo_common::types::EdgeId) -> bool {
        if self.lpg_mutation_closed() {
            return false;
        }
        // CdcGraphStore owns the single transactional pre-image event.
        self.session().delete_edge(id)
    }

    /// Sets a property on an edge.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge is not visible, the value exceeds the
    /// configured size limit, or the write cannot be committed.
    pub fn set_edge_property(
        &self,
        id: grafeo_common::types::EdgeId,
        key: &str,
        value: grafeo_common::types::Value,
    ) -> grafeo_common::utils::error::Result<()> {
        // CdcGraphStore owns the single transactional before/after event.
        self.session().set_edge_property(id, key, value)
    }

    /// Removes a property from a node.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    pub fn remove_node_property(&self, id: grafeo_common::types::NodeId, key: &str) -> bool {
        if self.lpg_mutation_closed() {
            return false;
        }
        self.session().remove_node_property(id, key)
    }

    /// Removes a property from an edge.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    pub fn remove_edge_property(&self, id: grafeo_common::types::EdgeId, key: &str) -> bool {
        if self.lpg_mutation_closed() {
            return false;
        }
        self.session().remove_edge_property(id, key)
    }

    /// Creates multiple nodes in bulk, each with a single vector property.
    ///
    /// Much faster than individual `create_node_with_props` calls because it
    /// acquires internal locks once and loops in Rust rather than crossing
    /// the FFI boundary per vector.
    ///
    /// The batch is atomic: either every node is committed, or none are.
    ///
    /// # Arguments
    ///
    /// * `label` - Label applied to all created nodes
    /// * `property` - Property name for the vector data
    /// * `vectors` - Vector data for each node
    ///
    /// # Returns
    ///
    /// Vector of created `NodeId`s in the same order as the input vectors.
    pub fn batch_create_nodes(
        &self,
        label: &str,
        property: &str,
        vectors: Vec<Vec<f32>>,
    ) -> Vec<grafeo_common::types::NodeId> {
        use grafeo_common::types::Value;

        if self.lpg_mutation_closed() || vectors.is_empty() {
            return Vec::new();
        }

        let labels: &[&str] = &[label];
        let mut session = self.session();
        if session.begin_transaction().is_err() {
            return Vec::new();
        }

        let mut ids = Vec::with_capacity(vectors.len());
        for vector in vectors {
            let value = Value::Vector(vector.into());
            match session.create_node_with_props(labels, std::iter::once((property, value))) {
                Ok(id) => ids.push(id),
                Err(_) => {
                    let _ = session.rollback();
                    return Vec::new();
                }
            }
        }
        let Ok(_) = session.commit() else {
            return Vec::new();
        };

        ids
    }

    /// Batch-creates nodes with full property maps.
    ///
    /// Each entry in `properties_list` is a complete property map for one node.
    /// Vector values (`Value::Vector`) are automatically inserted into matching
    /// vector indexes. Text values are automatically inserted into matching text
    /// indexes.
    ///
    /// The batch is atomic: either every node is committed, or none are.
    ///
    /// # Arguments
    ///
    /// * `label` - Label for all created nodes.
    /// * `properties_list` - One property map per node to create.
    ///
    /// # Returns
    ///
    /// Vector of created `NodeId`s in the same order as the input.
    pub fn batch_create_nodes_with_props(
        &self,
        label: &str,
        properties_list: Vec<
            std::collections::HashMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            >,
        >,
    ) -> Vec<grafeo_common::types::NodeId> {
        if self.lpg_mutation_closed() || properties_list.is_empty() {
            return Vec::new();
        }

        let labels: &[&str] = &[label];
        let mut session = self.session();
        if session.begin_transaction().is_err() {
            return Vec::new();
        }

        let mut ids = Vec::with_capacity(properties_list.len());
        for properties in &properties_list {
            let result = session.create_node_with_props(
                labels,
                properties
                    .iter()
                    .map(|(key, value)| (key.as_str(), value.clone())),
            );
            match result {
                Ok(id) => ids.push(id),
                Err(_) => {
                    let _ = session.rollback();
                    return Vec::new();
                }
            }
        }
        let Ok(_) = session.commit() else {
            return Vec::new();
        };

        ids
    }
}
