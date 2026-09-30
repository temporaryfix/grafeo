//! Index management methods for [`LpgStore`].

use super::index_registration::RegisteredIndex;
#[cfg(feature = "text-index")]
use super::index_registration::RegistrationIncarnation;
use super::property_index::PropertyIndexRows;
use super::{LpgStore, PinnedMutation};
#[cfg(feature = "compact-store")]
use super::{PinnedLpgTransition, PinnedNamedGraphTopology};
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use crate::graph::lpg::{canonicalize_index_key, decode_index_key, encode_index_key};
use grafeo_common::types::EpochId;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::types::TransactionId;
use grafeo_common::types::{HashableValue, NodeId, PropertyKey, Value};
#[cfg(feature = "compact-store")]
use grafeo_common::utils::hash::FxHashMap;
use grafeo_common::utils::hash::FxHashSet;
#[cfg(feature = "text-index")]
use parking_lot::RwLock;
use std::sync::Arc;

#[cfg(feature = "vector-index")]
use super::vector_accessor::SnapshotVectorAccessor;
#[cfg(all(feature = "text-index", feature = "compact-store"))]
use crate::index::text::TextIndexReadGuard;
#[cfg(feature = "text-index")]
use crate::index::text::{RegisteredTextIndex, TextIndexView};
#[cfg(all(feature = "vector-index", feature = "compact-store"))]
use crate::index::vector::VectorIndexReadFence;
#[cfg(feature = "vector-index")]
use crate::index::vector::{
    VectorAccessor, VectorIndexKind, VectorIndexView, compute_distance, value_to_vector,
};

/// Stable logical key used by transaction overlays and SSI trackers.
///
/// Derived-index registries and permanent alias bindings use the collision-free
/// encoded key. Transactional consumers retain their established
/// `"label:property"` namespace so routing and conflict receipts remain
/// compatible across the registry-key migration.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn logical_index_key(label: &str, property: &str) -> String {
    let mut key = String::with_capacity(label.len() + 1 + property.len());
    key.push_str(label);
    key.push(':');
    key.push_str(property);
    key
}

/// Borrowed tier-merged read context for a snapshot-visible vector search.
///
/// HNSW deliberately stores topology rather than duplicating vectors. An LPG
/// store normally supplies its own property storage as the accessor; a
/// [`LayeredStore`](crate::graph::compact::LayeredStore) instead supplies a
/// generation-pinned view spanning its compact base and mutable overlay. The
/// context is borrowed for one search, so an index can never retain the store
/// and form an ownership cycle.
#[cfg(feature = "vector-index")]
pub(crate) struct VisibleVectorReadContext<'a> {
    accessor: &'a dyn VectorAccessor,
    all_node_ids: &'a [NodeId],
    is_visible: &'a dyn Fn(NodeId) -> bool,
    has_label: &'a dyn Fn(NodeId, &str) -> bool,
}

#[cfg(feature = "vector-index")]
impl<'a> VisibleVectorReadContext<'a> {
    pub(crate) fn new(
        accessor: &'a dyn VectorAccessor,
        all_node_ids: &'a [NodeId],
        is_visible: &'a dyn Fn(NodeId) -> bool,
        has_label: &'a dyn Fn(NodeId, &str) -> bool,
    ) -> Self {
        Self {
            accessor,
            all_node_ids,
            is_visible,
            has_label,
        }
    }
}

/// Allocation-complete transfer of one same-incarnation representation
/// registry into an unpublished successor.
///
/// The source remains untouched during preparation so readers can continue on
/// the old Layered generation. Preparation builds exact frozen vector/text
/// forks for that source and a frozen named-graph topology. Publication later
/// moves the original runtime objects and named-graph registry to the logical
/// successor, installs the prepared forks on the retired source, and does no
/// allocation or validation under the Layered generation write cut. Thus a
/// caller-owned index handle follows the logical successor, while a fresh read
/// through an already-owned `overlay_store()` snapshot cannot observe later
/// successor identities.
#[cfg(feature = "compact-store")]
pub(crate) struct PreparedRepresentationTransfer {
    source: Arc<LpgStore>,
    target: Arc<LpgStore>,
    #[cfg(feature = "vector-index")]
    frozen_vector_indexes: FxHashMap<String, RegisteredIndex<Arc<VectorIndexKind>>>,
    #[cfg(feature = "vector-index")]
    vector_fences: Vec<VectorIndexReadFence>,
    #[cfg(feature = "text-index")]
    frozen_text_indexes: FxHashMap<String, RegisteredIndex<RegisteredTextIndex>>,
    #[cfg(feature = "text-index")]
    text_fences: Vec<TextIndexReadGuard>,
    frozen_named_graphs: FxHashMap<String, Arc<LpgStore>>,
}

/// Reversible proof that the exact runtime registries are published on a
/// same-incarnation successor and the source is a coherent read-only snapshot.
#[cfg(feature = "compact-store")]
pub(crate) struct PublishedRepresentationTransfer {
    source: Arc<LpgStore>,
    target: Arc<LpgStore>,
    #[cfg(feature = "vector-index")]
    _vector_fences: Vec<VectorIndexReadFence>,
    #[cfg(feature = "text-index")]
    _text_fences: Vec<TextIndexReadGuard>,
}

#[cfg(feature = "compact-store")]
impl PinnedLpgTransition<'_> {
    /// Qualifies an active native representation under its exact transition.
    /// This binding is representation ownership even without vector indexes.
    pub(crate) fn require_native_compact_source(&self) -> Result<(), &'static str> {
        if self.store.compact_base.get().is_some() {
            return Err("LPG representation already belongs to a compact generation");
        }
        Ok(())
    }

    /// Claims a fully prepared compact predecessor and raises both allocator
    /// floors without recursively acquiring the shared mutation gate.
    pub(crate) fn bind_compact_base(
        &self,
        base: Arc<crate::graph::compact::CompactStore>,
        node_floor: u64,
        edge_floor: u64,
    ) -> Result<(), &'static str> {
        self.store.install_compact_base(base)?;
        self.store
            .next_node_id
            .fetch_max(node_floor, std::sync::atomic::Ordering::AcqRel);
        self.store
            .next_edge_id
            .fetch_max(edge_floor, std::sync::atomic::Ordering::AcqRel);
        Ok(())
    }

    /// Prepares the runtime-registry post-images for an empty successor of the
    /// exact store pinned by this transition and named topology proof.
    ///
    /// Equality/property indexes deliberately retain only their definitions on
    /// the successor: the compact base supplies cold membership and a
    /// retained-hot replay populates the successor overlay before publication.
    /// Vector and text indexes cover the complete tier-merged graph. Exact
    /// frozen forks preserve all private runtime state for retained source
    /// snapshots; their original objects remain on the source until coherent
    /// publication moves them to the successor. Named child stores remain the
    /// same logical stores, but the source receives its own frozen registry map
    /// so later successor topology changes cannot leak backward.
    pub(crate) fn prepare_same_incarnation_representation_transfer(
        &self,
        _named_topology: &PinnedNamedGraphTopology,
        source: Arc<LpgStore>,
        target: Arc<LpgStore>,
    ) -> Result<PreparedRepresentationTransfer, String> {
        if !std::ptr::eq(self.store, source.as_ref()) {
            return Err("representation transfer source is not the pinned LPG incarnation".into());
        }
        if !source.representation_is_active() || !target.representation_is_active() {
            return Err("representation transfer requires two active LPG representations".into());
        }

        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        {
            if source.index_owner_id != target.index_owner_id
                || *source.index_slots.lock() != *target.index_slots.lock()
            {
                return Err(
                    "representation successor does not preserve the source owner and slot registry"
                        .into(),
                );
            }
        }

        let source_properties = source.property_indexes.read();
        let mut successor_properties = FxHashMap::default();
        successor_properties
            .try_reserve(source_properties.len())
            .map_err(|_| "allocating successor property-index registry failed".to_owned())?;
        for (key, index) in source_properties.iter() {
            successor_properties.insert(
                key.clone(),
                index.with_same_registration(Arc::new(
                    PropertyIndexRows::from_image(
                        source
                            .property_index_image(key.as_str())
                            .map_err(|error| error.to_string())?,
                    )
                    .map_err(|error| error.to_string())?,
                )),
            );
        }
        drop(source_properties);
        let mut target_properties = target.property_indexes.write();
        if !target_properties.is_empty() {
            return Err("derived-index successor property registry is not pristine".into());
        }
        *target_properties = successor_properties;
        drop(target_properties);

        #[cfg(feature = "vector-index")]
        let (vector_indexes, vector_fences) = {
            if !target.vector_indexes.read().is_empty() {
                return Err("representation successor vector registry is not pristine".into());
            }
            let source_indexes = source.vector_indexes.read();
            let mut prepared = FxHashMap::default();
            prepared
                .try_reserve(source_indexes.len())
                .map_err(|_| "allocating frozen vector-index registry failed".to_owned())?;
            let mut fences = Vec::new();
            fences
                .try_reserve(source_indexes.len())
                .map_err(|_| "allocating vector-index transfer fences failed".to_owned())?;
            for (key, index) in source_indexes.iter() {
                let (frozen, fence) =
                    index
                        .fork_exact_read_snapshot_with_fence()
                        .map_err(|error| {
                            format!(
                                "freezing vector index {key:?} for representation transfer: {error}"
                            )
                        })?;
                prepared.insert(key.clone(), index.with_same_registration(frozen));
                fences.push(fence);
            }
            (prepared, fences)
        };

        #[cfg(feature = "text-index")]
        let (text_indexes, text_fences) = {
            if !target.text_indexes.read().is_empty() {
                return Err("representation successor text registry is not pristine".into());
            }
            let source_indexes = source.text_indexes.read();
            let mut prepared = FxHashMap::default();
            prepared
                .try_reserve(source_indexes.len())
                .map_err(|_| "allocating frozen text-index registry failed".to_owned())?;
            let mut fences = Vec::new();
            fences
                .try_reserve(source_indexes.len())
                .map_err(|_| "allocating text-index transfer fences failed".to_owned())?;
            for (key, index) in source_indexes.iter() {
                let (frozen, fence) = index.exact_runtime_fork_with_fence();
                prepared.insert(key.clone(), index.with_same_registration(frozen));
                fences.push(fence);
            }
            (prepared, fences)
        };

        if !target.named_graphs.read().is_empty() {
            return Err("representation successor named-graph registry is not pristine".into());
        }
        let frozen_named_graphs = {
            let source_graphs = source.named_graphs.read();
            let mut prepared = FxHashMap::default();
            prepared
                .try_reserve(source_graphs.len())
                .map_err(|_| "allocating frozen named-graph registry failed".to_owned())?;
            prepared.extend(
                source_graphs
                    .iter()
                    .map(|(name, graph)| (name.clone(), Arc::clone(graph))),
            );
            prepared
        };

        Ok(PreparedRepresentationTransfer {
            source,
            target,
            #[cfg(feature = "vector-index")]
            frozen_vector_indexes: vector_indexes,
            #[cfg(feature = "vector-index")]
            vector_fences,
            #[cfg(feature = "text-index")]
            frozen_text_indexes: text_indexes,
            #[cfg(feature = "text-index")]
            text_fences,
            frozen_named_graphs,
        })
    }
}

#[cfg(feature = "compact-store")]
impl PreparedRepresentationTransfer {
    /// Revalidates that retained-hot replay did not populate a registry whose
    /// publication is owned by this token. This is fallible only before any
    /// representation pointer or external metadata can move.
    pub(crate) fn validate_unpublished_target(&self) -> Result<(), String> {
        if !self.source.representation_is_active() || !self.target.representation_is_active() {
            return Err(
                "representation transfer target changed activity before publication".into(),
            );
        }
        #[cfg(feature = "vector-index")]
        if !self.target.vector_indexes.read().is_empty() {
            return Err("representation successor vector registry changed during replay".into());
        }
        #[cfg(feature = "text-index")]
        if !self.target.text_indexes.read().is_empty() {
            return Err("representation successor text registry changed during replay".into());
        }
        if !self.target.named_graphs.read().is_empty() {
            return Err(
                "representation successor named-graph registry changed during replay".into(),
            );
        }
        Ok(())
    }

    /// Publishes exact runtime registries without allocation and retires the
    /// old representation from mutation. The caller holds the named topology,
    /// source LPG transition, and Layered generation write cuts, so no store
    /// mutation can observe the handoff.
    pub(crate) fn publish(self) -> PublishedRepresentationTransfer {
        let Self {
            source,
            target,
            #[cfg(feature = "vector-index")]
            frozen_vector_indexes,
            #[cfg(feature = "vector-index")]
            vector_fences,
            #[cfg(feature = "text-index")]
            frozen_text_indexes,
            #[cfg(feature = "text-index")]
            text_fences,
            frozen_named_graphs,
        } = self;
        #[cfg(feature = "vector-index")]
        {
            let mut source_indexes = source.vector_indexes.write();
            let mut target_indexes = target.vector_indexes.write();
            debug_assert!(target_indexes.is_empty());
            *target_indexes = std::mem::replace(&mut *source_indexes, frozen_vector_indexes);
        }
        #[cfg(feature = "text-index")]
        {
            let mut source_indexes = source.text_indexes.write();
            let mut target_indexes = target.text_indexes.write();
            debug_assert!(target_indexes.is_empty());
            *target_indexes = std::mem::replace(&mut *source_indexes, frozen_text_indexes);
        }
        {
            let mut source_graphs = source.named_graphs.write();
            let mut target_graphs = target.named_graphs.write();
            debug_assert!(target_graphs.is_empty());
            *target_graphs = std::mem::replace(&mut *source_graphs, frozen_named_graphs);
        }
        source.retire_representation();
        PublishedRepresentationTransfer {
            source,
            target,
            #[cfg(feature = "vector-index")]
            _vector_fences: vector_fences,
            #[cfg(feature = "text-index")]
            _text_fences: text_fences,
        }
    }
}

#[cfg(feature = "compact-store")]
impl PublishedRepresentationTransfer {
    /// Reverses a transient publication under the still-exclusive generation,
    /// LPG-transition, and named-topology cuts.
    pub(crate) fn rollback(self) {
        // Keep direct caller mutation fenced until every original runtime
        // object is back on the source and that representation is active.
        #[cfg(feature = "vector-index")]
        std::mem::swap(
            &mut *self.source.vector_indexes.write(),
            &mut *self.target.vector_indexes.write(),
        );
        #[cfg(feature = "text-index")]
        std::mem::swap(
            &mut *self.source.text_indexes.write(),
            &mut *self.target.text_indexes.write(),
        );
        std::mem::swap(
            &mut *self.source.named_graphs.write(),
            &mut *self.target.named_graphs.write(),
        );
        self.target.retire_representation();
        self.source.reactivate_representation();
    }

    /// Commits mutation ownership to the successor. The source keeps read-only
    /// exact index forks and frozen named topology, and is retired with the
    /// displaced Layered generation.
    pub(crate) fn commit(self) {
        // Dropping the fences here releases direct caller writers only after
        // Layered plus external publication crossed the hostile rollback
        // boundary and the coherent generation read cut reopened.
        drop(self);
    }
}

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
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let key = PropertyKey::new(property);

        let mut indexes = self.property_indexes.write();
        if indexes.contains_key(&key) {
            return; // Already indexed
        }

        let Ok(image) = self.property_index_image(property) else {
            return;
        };
        let Ok(rows) = PropertyIndexRows::from_image(image) else {
            return;
        };
        let index = RegisteredIndex::new(Arc::new(rows));

        indexes.insert(key, index);
    }

    /// Drops an index on a node property.
    ///
    /// Returns `true` if the index existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let key = PropertyKey::new(property);
        let mut indexes = self.property_indexes.write();
        indexes.remove(&key).is_some()
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
        self.update_property_index_on_set_at_epoch(node_id, key, new_value, self.current_epoch());
    }

    pub(super) fn update_property_index_on_set_at_epoch(
        &self,
        node_id: NodeId,
        key: &PropertyKey,
        new_value: &Value,
        epoch: EpochId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            let before = self.node_properties.get(node_id, key);
            index
                .history
                .write()
                .record_direct(node_id, before.as_ref(), Some(new_value), epoch);
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
        self.update_property_index_on_remove_at_epoch(node_id, key, self.current_epoch());
    }

    pub(super) fn update_property_index_on_remove_at_epoch(
        &self,
        node_id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            let before = self.node_properties.get(node_id, key);
            index
                .history
                .write()
                .record_direct(node_id, before.as_ref(), None, epoch);
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

    /// Removes a node from every derived index before its labels and properties
    /// are discarded.
    ///
    /// Node deletion is owned by the store rather than an engine wrapper, so
    /// direct, GQL, and explicit-transaction deletes all converge here.
    pub(super) fn remove_node_from_derived_indexes(&self, id: NodeId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let indexed_properties: Vec<PropertyKey> =
            self.property_indexes.read().keys().cloned().collect();
        for property in indexed_properties {
            self.update_property_index_on_remove(id, &property);
        }

        #[cfg(feature = "vector-index")]
        if !self.recorded_vector_recovery_active() {
            let indexes: Vec<Arc<VectorIndexKind>> = self
                .vector_indexes
                .read()
                .values()
                .map(|index| Arc::clone(&index.payload))
                .collect();
            for index in indexes {
                index.remove(id);
            }
        }

        #[cfg(feature = "text-index")]
        self.remove_from_all_text_indexes(id);
    }

    /// Records every registered text/vector index whose membership can change
    /// when a node property is written. This is deliberately separate from
    /// index maintenance: legacy versioned writers update the committed
    /// representation directly, but must still publish the SSI phantom guard.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    pub(super) fn record_index_writes_for_node_property(
        &self,
        id: NodeId,
        property: &str,
        transaction_id: TransactionId,
    ) {
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        let labels = self.committed_node_label_ids(id);
        let registry = self.label_registry.read();
        let mut keys = Vec::new();
        for label_id in labels {
            let Some(label) = registry.get_name(label_id.into()) else {
                continue;
            };
            let registry_key = encode_index_key(label, property);
            #[cfg(feature = "text-index")]
            if self.text_indexes.read().contains_key(&registry_key) {
                keys.push(logical_index_key(label, property));
            }
            #[cfg(feature = "vector-index")]
            if self.vector_indexes.read().contains_key(&registry_key) {
                keys.push(logical_index_key(label, property));
            }
        }
        drop(registry);
        for key in keys {
            self.record_write_index(transaction_id, &key);
        }
    }

    /// Records all text/vector indexes bound to `label`. Label membership
    /// changes can alter every property posting on that label, including a
    /// same-transaction label-plus-property SET sequence.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    pub(super) fn record_index_writes_for_label(&self, label: &str, transaction_id: TransactionId) {
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        let mut keys = Vec::new();
        #[cfg(feature = "text-index")]
        for key in self.text_indexes.read().keys() {
            if let Some((indexed_label, property)) = decode_index_key(key)
                && indexed_label == label
            {
                keys.push(logical_index_key(indexed_label, property));
            }
        }
        #[cfg(feature = "vector-index")]
        for key in self.vector_indexes.read().keys() {
            if let Some((indexed_label, property)) = decode_index_key(key)
                && indexed_label == label
            {
                keys.push(logical_index_key(indexed_label, property));
            }
        }
        for key in keys {
            self.record_write_index(transaction_id, &key);
        }
    }

    /// Records every derived index on each committed label carried by a node
    /// being deleted. The call belongs to the transactional delete chokepoint;
    /// commit-time physical removal may run after the Serializable tracker is
    /// unregistered.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    pub(super) fn record_index_writes_for_node_delete(
        &self,
        id: NodeId,
        transaction_id: TransactionId,
    ) {
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        let labels = self.committed_node_label_ids(id);
        let registry = self.label_registry.read();
        let names: Vec<_> = labels
            .into_iter()
            .filter_map(|label_id| registry.get_name(label_id.into()).cloned())
            .collect();
        drop(registry);
        for label in names {
            self.record_index_writes_for_label(&label, transaction_id);
        }
    }

    /// Same-incarnation hydration changes no logical property membership.
    /// All registered property payloads now cover the complete graph, so its
    /// rollback must preserve their current and historical cold memberships.
    #[cfg(feature = "compact-store")]
    pub(super) fn purge_node_from_local_property_indexes_inner(
        &self,
        _id: NodeId,
        _mutation: &PinnedMutation<'_>,
    ) {
    }

    /// Fresh-identity rollback removes every property membership belonging only
    /// to the aborted identity, including history, before that ID can be reused.
    pub(super) fn purge_rolled_back_node_from_derived_indexes_inner(
        &self,
        id: NodeId,
        _mutation: &PinnedMutation<'_>,
    ) {
        {
            let indexes = self.property_indexes.read();
            for index in indexes.values() {
                index.retain(|_, nodes| {
                    nodes.remove(&id);
                    !nodes.is_empty()
                });
                index.history.write().purge_identity(id);
            }
        }

        #[cfg(feature = "text-index")]
        if !self.recorded_text_recovery_active() {
            let indexes = self.text_indexes.read();
            for index in indexes.values() {
                index.write().purge_identity(id);
            }
        }
    }

    /// Stores a vector index for a label+property pair.
    #[cfg(feature = "vector-index")]
    pub fn add_vector_index(&self, label: &str, property: &str, index: Arc<VectorIndexKind>) {
        let transition = VectorIndexKind::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let key = encode_index_key(label, property);
        let Ok(slot) = self.prepare_index_slot(key.clone()) else {
            return;
        };
        let slot_id = slot.id();
        let scope = self
            .mutation_scope
            .load(std::sync::atomic::Ordering::Acquire);
        if !index.binding_is_compatible(self.index_owner_id, slot_id, &transition)
            || if scope == 0 {
                !index.scope_is_unsealed(&transition)
            } else {
                !index.scope_is_compatible(scope, &transition)
            }
        {
            return;
        }
        self.vector_indexes.write().reserve(1);
        let index = RegisteredIndex::new(index);
        if !index.bind_under_transition(self.index_owner_id, slot_id, &transition)
            || (scope != 0 && !index.seal_with_scope_under_transition(scope, &transition))
        {
            return;
        }
        slot.commit();
        self.vector_indexes.write().insert(key, index);
    }

    /// Binds the compact predecessor of this unpublished LPG successor.
    ///
    /// Each physical LPG representation is bound at most once. The base owns
    /// no reference back to the overlay, so the retained Arc keeps old readers
    /// valid without a Layered/index ownership cycle.
    #[cfg(feature = "compact-store")]
    pub(crate) fn install_compact_base(
        &self,
        base: Arc<crate::graph::compact::CompactStore>,
    ) -> Result<(), &'static str> {
        let floor = base
            .property_history_floor()
            .unwrap_or(self.current_epoch());
        #[cfg(not(feature = "vector-index"))]
        let base = Arc::downgrade(&base);
        self.compact_base
            .set(base)
            .map_err(|_| "compact base is already bound for this LPG representation")?;
        self.advance_retained_history_floor(floor);
        Ok(())
    }

    /// Resolves the committed-latest vector backing one physical index.
    ///
    /// A local structural identity is authoritative even when its property is
    /// absent or the node is deleted; falling through in that case would
    /// resurrect a compact value shadowed by the hot representation. Only an
    /// identity absent from this LPG representation may consult its immutable
    /// compact predecessor.
    #[cfg(feature = "vector-index")]
    fn vector_for_physical_index(&self, id: NodeId, property: &PropertyKey) -> Option<Arc<[f32]>> {
        if self.contains_node_identity(id) {
            return self
                .get_node_property(id, property)
                .as_ref()
                .and_then(value_to_vector);
        }

        #[cfg(feature = "compact-store")]
        if let Some(base) = self.compact_base.get() {
            return crate::graph::traits::GraphStore::get_node_property(
                base.as_ref(),
                id,
                property,
            )
            .as_ref()
            .and_then(value_to_vector);
        }

        None
    }

    /// Retrieves a read-only vector index view for a label+property pair.
    ///
    /// The view deliberately has no topology mutation methods. Index contents
    /// are maintained by the owning store as graph transactions publish.
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use grafeo_common::types::NodeId;
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    ///
    /// let store = LpgStore::new().unwrap();
    /// store.add_vector_index(
    ///     "Doc",
    ///     "embedding",
    ///     Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
    ///         2,
    ///         DistanceMetric::Cosine,
    ///     )))),
    /// );
    /// let view = store.get_vector_index("Doc", "embedding").unwrap();
    /// view.remove(NodeId::new(1));
    /// ```
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn get_vector_index(&self, label: &str, property: &str) -> Option<VectorIndexView> {
        let key = encode_index_key(label, property);
        self.vector_indexes
            .read()
            .get(&key)
            .map(|index| VectorIndexView::new(Arc::clone(&index.payload)))
    }

    /// Removes a vector index for a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "vector-index")]
    pub fn remove_vector_index(&self, label: &str, property: &str) -> bool {
        let _transition = VectorIndexKind::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let key = encode_index_key(label, property);
        let mut indexes = self.vector_indexes.write();
        indexes.remove(&key).is_some()
    }

    /// Returns all vector index entries as `(key, index)` pairs.
    ///
    /// Keys use [`encode_index_key`].
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn vector_index_entries(&self) -> Vec<(String, VectorIndexView)> {
        self.vector_indexes
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), VectorIndexView::new(Arc::clone(v))))
            .collect()
    }

    /// Looks up a vector index by its encoded key or a legacy-simple
    /// `"label:property"` key.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn get_vector_index_by_key(&self, key: &str) -> Option<VectorIndexView> {
        let key = canonicalize_index_key(key)?;
        self.vector_indexes
            .read()
            .get(&key)
            .map(|index| VectorIndexView::new(Arc::clone(&index.payload)))
    }

    /// Reconciles every vector index on `property` with the node's committed
    /// labels and committed latest value.
    ///
    /// This is the single maintenance path used after both direct writes and
    /// transaction-overlay publication. Removing first makes vector-to-vector
    /// replacement correct for indexes that retain their own vector payload,
    /// while the following insert rebuilds HNSW membership against the newly
    /// committed property value.
    #[cfg(feature = "vector-index")]
    pub(super) fn refresh_vector_indexes_for_property(&self, id: NodeId, property: &str) {
        if self.recorded_vector_recovery_active() {
            return;
        }
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let entries: Vec<(String, Arc<VectorIndexKind>)> = self
            .vector_indexes
            .read()
            .iter()
            .filter_map(|(index_key, index)| {
                let (label, indexed_property) = decode_index_key(index_key)?;
                (indexed_property == property).then(|| (label.to_string(), Arc::clone(index)))
            })
            .collect();
        if entries.is_empty() {
            return;
        }

        let node = self.get_node(id);
        let vector = node
            .as_ref()
            .and_then(|node| node.properties.get(&PropertyKey::new(property)))
            .and_then(value_to_vector);
        let property_key = PropertyKey::new(property);
        let accessor = |candidate| self.vector_for_physical_index(candidate, &property_key);

        for (label, index) in entries {
            index.remove(id);
            if node.as_ref().is_some_and(|node| node.has_label(&label))
                && let Some(vector) = vector.as_deref()
            {
                index.insert(id, vector, &accessor);
            }
        }
    }

    /// Reconciles every vector index for `label` after committed label
    /// membership changes.
    #[cfg(feature = "vector-index")]
    pub(super) fn refresh_vector_indexes_for_label(&self, id: NodeId, label: &str) {
        if self.recorded_vector_recovery_active() {
            return;
        }
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let entries: Vec<(String, Arc<VectorIndexKind>)> = self
            .vector_indexes
            .read()
            .iter()
            .filter_map(|(index_key, index)| {
                let (indexed_label, property) = decode_index_key(index_key)?;
                (indexed_label == label).then(|| (property.to_string(), Arc::clone(index)))
            })
            .collect();
        if entries.is_empty() {
            return;
        }

        let node = self.get_node(id);
        let has_label = node.as_ref().is_some_and(|node| node.has_label(label));
        for (property, index) in entries {
            index.remove(id);
            if !has_label {
                continue;
            }
            let vector = node
                .as_ref()
                .and_then(|node| node.properties.get(&PropertyKey::new(&property)))
                .and_then(value_to_vector);
            if let Some(vector) = vector.as_deref() {
                let property_key = PropertyKey::new(&property);
                let accessor = |candidate| self.vector_for_physical_index(candidate, &property_key);
                index.insert(id, vector, &accessor);
            }
        }
    }

    // === Snapshot vector search (VI4) ===

    /// Searches a vector index at `(epoch, tx)`, returning the `k` nearest
    /// **visible** nodes scored **as-of-E**, with read-your-writes support for
    /// the writing transaction's uncommitted vector `SET`s.
    ///
    /// ## Algorithm
    ///
    /// 1. Delegate to `VectorIndexKind::search_visible` with a snapshot
    ///    visibility predicate and a `SnapshotVectorAccessor` that reads each
    ///    node's vector as-of `epoch` (or from the tx's uncommitted delta for
    ///    read-your-writes).
    /// 2. Brute-force merge the tx's uncommitted property delta for the target
    ///    `property`, so nodes whose vector was `SET` in this tx but are not yet
    ///    in the committed HNSW graph also appear in the result (read-your-writes
    ///    completeness).  A same-tx node that is already in the HNSW result has
    ///    its entry replaced so the score reflects the uncommitted value.
    /// 3. Re-sort and truncate to `k`.
    ///
    /// If no committed index exists for `index_key` the method falls back to
    /// brute-force over the tx overlay only.
    ///
    /// `index_key` may be encoded or use the legacy-simple `"label:property"`
    /// representation.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn search_vector_visible(
        &self,
        index_key: &str,
        query: &[f32],
        k: usize,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Vec<(NodeId, f32)> {
        let Some(canonical_key) = canonicalize_index_key(index_key) else {
            return Vec::new();
        };
        let Some((_, property)) = decode_index_key(&canonical_key) else {
            return Vec::new();
        };
        let property = PropertyKey::new(property);
        let accessor = SnapshotVectorAccessor {
            store: self,
            property,
            epoch,
            tx: Some(tx),
        };
        let all_node_ids = self.all_node_ids();
        let is_visible = |id| self.is_node_visible_versioned(id, epoch, tx);
        let has_label = |id, label: &str| {
            self.read_node_labels_visible(id, epoch, Some(tx))
                .iter()
                .any(|candidate| candidate.as_str() == label)
        };
        let context =
            VisibleVectorReadContext::new(&accessor, &all_node_ids, &is_visible, &has_label);
        self.search_vector_visible_with_context(&canonical_key, query, k, tx, &context)
    }

    /// Tier-merged form of [`Self::search_vector_visible`].
    ///
    /// The registry and transactional write delta remain owned by `self`, but
    /// vector values, visibility, labels, and fallback candidates are supplied
    /// by the caller's pinned logical generation. This is required after LPG
    /// compaction because ordinary HNSW retains exact topology while its vector
    /// properties move into the compact tier.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub(crate) fn search_vector_visible_with_context(
        &self,
        index_key: &str,
        query: &[f32],
        k: usize,
        tx: TransactionId,
        context: &VisibleVectorReadContext<'_>,
    ) -> Vec<(NodeId, f32)> {
        let Some(index_key) = canonicalize_index_key(index_key) else {
            return Vec::new();
        };

        let Some((label, property_str)) = decode_index_key(&index_key) else {
            return Vec::new();
        };
        let logical_key = logical_index_key(label, property_str);

        // Coarse predicate-read recording for anti-phantom SSI (Task 5).
        // A Serializable vector search reads every node matching the query in
        // this index; a concurrent indexed SET is a phantom. Record the whole
        // index as read so the existing rw-detection can form the edge. This is
        // a no-op for SI/ReadCommitted (no read tracker registered for `tx`).
        // Must fire BEFORE any early-return so a zero-result search still records.
        self.record_read_index(tx, &logical_key);
        let property_key = PropertyKey::new(property_str);

        // Empty label = "any label" (a label-less vector scan). The `index_key`
        // recorded above (":property") matches no write key on its own, so record
        // a read for EVERY vector index on this property — that matches the
        // per-(label,property) granularity the write side records, so a concurrent
        // indexed SET on any label forms the anti-phantom rw-edge.
        if label.is_empty() {
            let keys: Vec<String> = self
                .vector_indexes
                .read()
                .keys()
                .filter_map(|key| {
                    let (indexed_label, indexed_property) = decode_index_key(key)?;
                    (indexed_property == property_str)
                        .then(|| logical_index_key(indexed_label, indexed_property))
                })
                .collect();
            for key in &keys {
                self.record_read_index(tx, key);
            }
        }

        // Look up the committed index.
        let committed_idx = self.get_vector_index_by_key(&index_key);

        // Determine the distance metric from the committed index (fallback: Cosine).
        let metric = committed_idx
            .as_ref()
            .map_or(crate::index::vector::DistanceMetric::Cosine, |idx| {
                idx.config().metric
            });

        // Step 1: search the committed HNSW or brute-force all committed nodes.
        let mut results: Vec<(NodeId, f32)> = match &committed_idx {
            Some(idx) => {
                let ef = idx.config().ef.max(k.saturating_mul(4));
                let indexed_is_visible = |id| {
                    (context.is_visible)(id) && (label.is_empty() || (context.has_label)(id, label))
                };
                idx.search_visible(query, k, ef, &indexed_is_visible, context.accessor)
            }
            None => {
                // No HNSW — brute-force scan all nodes visible at (epoch, tx)
                // that have the target property and label. Label version logs
                // are retained after deletion, so the same predicate is valid
                // for current and historical snapshots. An empty label remains
                // the explicit "any label" form.
                let mut bf: Vec<(NodeId, f32)> = context
                    .all_node_ids
                    .iter()
                    .copied()
                    .filter(|&id| (context.is_visible)(id))
                    .filter(|&id| label.is_empty() || (context.has_label)(id, label))
                    .filter_map(|id| {
                        // Get the snapshot-consistent vector via the accessor.
                        // Returns None if the node doesn't have the property at (epoch, tx).
                        let vec = context.accessor.get_vector(id)?;
                        let dist = compute_distance(query, &vec, metric);
                        Some((id, dist))
                    })
                    .collect();
                bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                bf.truncate(k);
                bf
            }
        };

        // Step 2: brute-force merge the tx's uncommitted property delta.
        //
        // Walk every (node, property) entry in the tx overlay.  For entries whose
        // property matches our target and whose node carries `label` (visible to
        // the tx), compute the distance and merge into `results`.  A committed
        // entry for the same node_id is replaced (the uncommitted value wins).
        let overlay_entries: Vec<(NodeId, Value)> = {
            let overlay = self.tx_property_overlay.read();
            match overlay.get(&tx) {
                None => Vec::new(),
                Some(delta) => delta
                    .node_props
                    .iter()
                    .filter_map(|((node_id, prop_key), op)| {
                        if prop_key != &property_key {
                            return None;
                        }
                        match op {
                            super::PropOp::Set(v) => Some((*node_id, v.clone())),
                            super::PropOp::Remove => None,
                        }
                    })
                    .collect(),
            }
        };

        for (node_id, value) in overlay_entries {
            // Only include nodes that are visible at (epoch, tx) — this
            // filters tx-deleted nodes via is_node_visible_versioned.
            if !(context.is_visible)(node_id) {
                continue;
            }
            // Only include nodes that carry the target label (tx-visible label
            // check). Empty label = any label (label-less scan): skip the filter.
            if !label.is_empty() && !(context.has_label)(node_id, label) {
                continue;
            }
            let Some(vector) = value_to_vector(&value) else {
                continue;
            };
            let dist = compute_distance(query, &vector, metric);
            // Replace any committed hit for the same node (uncommitted value wins).
            if let Some(existing) = results.iter_mut().find(|(id, _)| *id == node_id) {
                existing.1 = dist;
            } else {
                results.push((node_id, dist));
            }
        }

        // Step 3: re-sort and take k.
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Stores a text index for a label+property pair.
    ///
    /// Registration pins the concrete index behind a registry-owned outer lock.
    /// The supplied handle continues to forward normal reads and authorized
    /// writes, but replacing the value inside its retained outer lock cannot
    /// replace the store's authority-bound index.
    #[cfg(feature = "text-index")]
    pub fn add_text_index(
        &self,
        label: &str,
        property: &str,
        index: Arc<RwLock<crate::index::text::InvertedIndex>>,
    ) {
        let transition = crate::index::text::InvertedIndex::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let key = encode_index_key(label, property);
        let Ok(slot) = self.prepare_index_slot(key.clone()) else {
            return;
        };
        let slot_id = slot.id();
        let scope = self
            .mutation_scope
            .load(std::sync::atomic::Ordering::Acquire);
        // Reserve before binding or moving caller state so registration has no
        // registry allocation left after its authority transition.
        self.text_indexes.write().reserve(1);

        // Retain one exclusive caller-handle guard across compatibility,
        // pinning, binding, and sealing. This is the first half of the fixed
        // gate→target order used by every registered index operation.
        let mut index_guard = index.write();
        if !index_guard.binding_is_compatible(self.index_owner_id, slot_id, &transition)
            || if scope == 0 {
                !index_guard.scope_is_unsealed(&transition)
            } else {
                !index_guard.scope_is_compatible(scope, &transition)
            }
        {
            return;
        }
        let registration = Arc::new(RegistrationIncarnation);
        let pinned = index_guard.pin_registry_target();

        let target_guard = pinned.read();
        if !target_guard.bind_under_transition(self.index_owner_id, slot_id, &transition)
            || (scope != 0 && !target_guard.seal_with_scope_under_transition(scope, &transition))
        {
            return;
        }
        drop(target_guard);

        let registered = RegisteredIndex {
            registration,
            payload: RegisteredTextIndex::new(Arc::clone(&index), pinned),
        };
        drop(index_guard);
        slot.commit();
        self.text_indexes.write().insert(key, registered);
    }

    /// Retrieves a read-only text index view for a label+property pair.
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use parking_lot::RwLock;
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_core::index::text::{BM25Config, InvertedIndex};
    ///
    /// let store = LpgStore::new().unwrap();
    /// store.add_text_index(
    ///     "Doc",
    ///     "body",
    ///     Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
    /// );
    /// let view = store.get_text_index("Doc", "body").unwrap();
    /// view.write().insert(grafeo_common::types::NodeId::new(1), "forged");
    /// ```
    #[cfg(feature = "text-index")]
    #[must_use]
    pub fn get_text_index(&self, label: &str, property: &str) -> Option<TextIndexView> {
        let key = encode_index_key(label, property);
        self.text_indexes
            .read()
            .get(&key)
            .map(|index| TextIndexView::from_registered(&index.payload))
    }

    /// Removes a text index for a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "text-index")]
    pub fn remove_text_index(&self, label: &str, property: &str) -> bool {
        let _transition = crate::index::text::InvertedIndex::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let key = encode_index_key(label, property);
        let mut indexes = self.text_indexes.write();
        let Some(index) = indexes.get(&key).cloned() else {
            return false;
        };
        // Registry-map → caller gate → concrete target. Retain the composite
        // guard until the entry is no longer discoverable so a caller compound
        // cannot straddle the DDL removal boundary.
        let _index = index.write();
        indexes.remove(&key).is_some()
    }

    /// Returns all text index entries as `(key, index)` pairs.
    ///
    /// Keys use [`encode_index_key`].
    #[cfg(feature = "text-index")]
    pub fn text_index_entries(&self) -> Vec<(String, TextIndexView)> {
        self.text_indexes
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), TextIndexView::from_registered(v)))
            .collect()
    }

    /// Reconciles every text index for `label` after committed label
    /// membership changes.
    #[cfg(feature = "text-index")]
    pub(super) fn refresh_text_indexes_for_label(&self, id: NodeId, label: &str) {
        if self.recorded_text_recovery_active() {
            return;
        }
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let entries: Vec<(String, RegisteredTextIndex)> = self
            .text_indexes
            .read()
            .iter()
            .filter_map(|(index_key, index)| {
                let (indexed_label, property) = decode_index_key(index_key)?;
                (indexed_label == label).then(|| (property.to_string(), index.payload.clone()))
            })
            .collect();
        if entries.is_empty() {
            return;
        }

        let node = self.get_node(id);
        let has_label = node.as_ref().is_some_and(|node| node.has_label(label));
        let epoch = self.current_epoch();
        for (property, index) in entries {
            let mut index = index.write();
            index.remove_versioned(id, epoch, None);
            if has_label
                && let Some(Value::String(text)) = node
                    .as_ref()
                    .and_then(|node| node.properties.get(&PropertyKey::new(&property)))
            {
                index.insert_versioned(id, text, epoch, None);
            }
        }
    }

    /// Updates text indexes when a node property is set.
    ///
    /// If the node has a label with a text index on this property key,
    /// the index is updated with the new value (if it's a string).
    #[cfg(feature = "text-index")]
    pub(super) fn update_text_index_on_set(&self, id: NodeId, key: &str, value: &Value) {
        if self.recorded_text_recovery_active() {
            return;
        }
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
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
                    let index_key = encode_index_key(label_name, key);
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
        if self.recorded_text_recovery_active() {
            return;
        }
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            // Single-source invariant (TI5): stamp the deletion at the property's
            // commit epoch `C` (see `update_text_index_on_set`). The posting's
            // `deleted_epoch == C` so a snapshot `< C` still sees the old text and
            // `>= C` sees the removal — matching the property version boundary.
            let epoch = self.current_epoch();
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let index_key = encode_index_key(label_name, key);
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
        if self.recorded_text_recovery_active() {
            return;
        }
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
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
    /// Mirrors [`Self::search_text_visible`] exactly, but uses a threshold cutoff
    /// instead of a top-k limit.  The `index_key` format is `"label:property"`.
    ///
    /// # Errors
    /// Rejects epochs below the Text index's retained history floor.
    #[cfg(feature = "text-index")]
    pub fn search_text_with_threshold_visible(
        &self,
        index_key: &str,
        query: &str,
        threshold: f64,
        epoch: EpochId,
        tx: TransactionId,
    ) -> grafeo_common::utils::error::Result<Vec<(NodeId, f64)>> {
        let Some(index_key) = canonicalize_index_key(index_key) else {
            return Ok(Vec::new());
        };
        let Some((label, property)) = decode_index_key(&index_key) else {
            return Ok(Vec::new());
        };
        let logical_key = logical_index_key(label, property);

        // Record the index read for anti-phantom SSI — must happen even when
        // no postings match so that a zero-result threshold scan still closes
        // the rw-antidependency cycle if a concurrent tx inserts a matching doc.
        self.record_read_index(tx, &logical_key);

        // Build delta_docs and delta_removed from the overlay (identical to
        // search_text_visible).
        let (delta_docs, delta_removed): (Vec<(NodeId, String)>, FxHashSet<NodeId>) = {
            let overlay = self.text_index_overlay.read();
            match overlay.get(&tx) {
                None => (Vec::new(), FxHashSet::default()),
                Some(delta) => {
                    let mut docs = Vec::new();
                    let mut removed = FxHashSet::default();
                    for (node_id, opt_text) in delta.changes_for(&index_key) {
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
            text_indexes.get(&index_key).cloned()
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
                    return Ok(Vec::new());
                }
                let mut transient = crate::index::text::InvertedIndex::new(
                    crate::index::text::BM25Config::default(),
                );
                for (node_id, text) in &delta_docs {
                    transient.insert(*node_id, text);
                }
                Ok(transient.search_with_threshold(query, threshold))
            }
        }
    }

    // === Snapshot per-row score (per-row filter path, anti-phantom SSI) ===

    /// Scores a single node against a text query at `(epoch, tx)`, recording
    /// the index read for SSI conflict detection before computing the score.
    ///
    /// This is the snapshot-aware counterpart of
    /// [`crate::graph::traits::GraphStoreSearch::score_text`]: it records
    /// `record_read_index(tx, index_key)` **first** (so even a non-matching row
    /// closes the rw-antidependency cycle against a concurrent phantom insert),
    /// then scores the node using postings visible at `(epoch, tx)`.
    ///
    /// The `index_key` format is `"label:property"`.
    ///
    /// # Errors
    /// Rejects epochs below the Text index's retained history floor.
    #[cfg(feature = "text-index")]
    pub fn score_text_visible_impl(
        &self,
        index_key: &str,
        node_id: NodeId,
        query: &str,
        epoch: EpochId,
        tx: TransactionId,
    ) -> grafeo_common::utils::error::Result<Option<f64>> {
        let Some(index_key) = canonicalize_index_key(index_key) else {
            return Ok(None);
        };
        let Some((label, property)) = decode_index_key(&index_key) else {
            return Ok(None);
        };
        let logical_key = logical_index_key(label, property);

        // Record the index read for anti-phantom SSI — must happen even when the
        // node ultimately scores 0.0 so that a Serializable scan that returns 0
        // results still closes the rw-antidependency cycle if a concurrent tx
        // inserts a matching document.
        self.record_read_index(tx, &logical_key);

        // Look up the per-tx delta entry for this specific node.
        // We use `get(index_key, node_id)` to avoid iterating all changes.
        let (delta_doc_opt, delta_removed): (
            Option<(u32, std::collections::HashMap<String, u32>)>,
            bool,
        ) = {
            let overlay = self.text_index_overlay.read();
            match overlay.get(&tx).and_then(|d| d.get(&index_key, node_id)) {
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
            text_indexes.get(&index_key).cloned()
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
                    return Ok(None);
                }
                let Some((_, freq_map)) = delta_doc_opt else {
                    return Ok(None);
                };
                // Build a transient single-document index for scoring.
                // Re-fetch the raw text from the overlay so we can use `insert`.
                let raw_text = {
                    let overlay = self.text_index_overlay.read();
                    overlay
                        .get(&tx)
                        .and_then(|d| d.get(&index_key, node_id))
                        .and_then(|opt| opt.as_deref().map(str::to_owned))
                };
                // Suppress unused variable warning; the raw text is only needed if
                // we can still retrieve it (race-free since we hold no lock here, but
                // the overlay is append-only within a transaction so it will be present).
                let _ = freq_map;
                let Some(text) = raw_text else {
                    return Ok(None);
                };
                let mut transient = crate::index::text::InvertedIndex::new(
                    crate::index::text::BM25Config::default(),
                );
                transient.insert(node_id, &text);
                let score = transient.score_document(node_id, query);
                Ok(if score > 0.0 { Some(score) } else { None })
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
    ///
    /// # Errors
    /// Rejects PENDING horizons, denied mutation authority or invalid Text aggregates.
    #[cfg(feature = "text-index")]
    pub fn gc_text_indexes(&self, horizon: EpochId) -> grafeo_common::utils::error::Result<()> {
        if horizon == EpochId::PENDING {
            return Err(grafeo_common::utils::error::Error::InvalidValue(
                "Text GC horizon cannot be PENDING".into(),
            ));
        }
        let Some(_mutation) = self.pin_mutation() else {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        };
        let indexes = self.text_indexes.read();
        for idx in indexes.values() {
            idx.write().gc(horizon)?;
        }
        Ok(())
    }

    // === Vector-index GC ===

    /// Returns `true` if `id` was deleted at a committed epoch that is at or
    /// below `horizon`, meaning no active snapshot can see it as alive.
    ///
    /// Every retained incarnation must have a committed deletion at or below
    /// the horizon. An earlier deletion cannot retire a live, pending or
    /// younger successor sharing the same graph-local ID.
    ///
    /// Used by [`gc_vector_indexes`](Self::gc_vector_indexes) to decide
    /// whether a soft-deleted HNSW node can be permanently dropped.
    #[cfg(feature = "vector-index")]
    pub(crate) fn node_deleted_at_or_below(&self, id: NodeId, horizon: EpochId) -> bool {
        #[cfg(not(feature = "tiered-storage"))]
        {
            let nodes = self.nodes.read();
            if let Some(chain) = nodes.get(&id) {
                return Self::vector_gc_lifetimes_deleted(
                    chain
                        .history()
                        .map(|(info, _)| (info.created_epoch, info.deleted_epoch)),
                    horizon,
                );
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            let versions = self.node_versions.read();
            if let Some(index) = versions.get(&id) {
                return Self::vector_gc_lifetimes_deleted(
                    index
                        .version_history()
                        .into_iter()
                        .map(|(created, deleted, _)| (created, deleted)),
                    horizon,
                );
            }
        }
        #[cfg(feature = "compact-store")]
        if let Some(history) = self
            .compact_base
            .get()
            .and_then(|base| base.temporal_node_history(id))
        {
            return Self::vector_gc_lifetimes_deleted(
                history
                    .lifetimes
                    .iter()
                    .map(|life| (life.created, life.deleted)),
                horizon,
            );
        }
        false
    }

    #[cfg(feature = "vector-index")]
    pub(crate) fn vector_gc_lifetimes_deleted(
        lifetimes: impl IntoIterator<Item = (EpochId, Option<EpochId>)>,
        horizon: EpochId,
    ) -> bool {
        let mut found = false;
        for (created, deleted) in lifetimes {
            if created == EpochId::PENDING
                || !matches!(deleted, Some(epoch) if epoch != EpochId::PENDING && epoch <= horizon)
            {
                return false;
            }
            found = true;
        }
        found
    }

    #[cfg(feature = "vector-index")]
    fn node_gc_incarnation(&self, id: NodeId) -> Option<(EpochId, Option<EpochId>)> {
        let incarnation;
        #[cfg(not(feature = "tiered-storage"))]
        {
            let nodes = self.nodes.read();
            incarnation = nodes.get(&id).map(|chain| {
                chain
                    .history()
                    .filter(|(info, _)| info.created_epoch != EpochId::PENDING)
                    .max_by_key(|(info, _)| info.created_epoch)
                    .map(|(info, _)| (info.created_epoch, info.deleted_epoch))
            });
        }
        #[cfg(feature = "tiered-storage")]
        {
            let versions = self.node_versions.read();
            incarnation = versions.get(&id).map(|index| {
                index
                    .version_history()
                    .into_iter()
                    .filter(|(created, _, _)| *created != EpochId::PENDING)
                    .max_by_key(|(created, _, _)| *created)
                    .map(|(created, deleted, _)| (created, deleted))
            });
        }
        if let Some(incarnation) = incarnation {
            return incarnation;
        }
        #[cfg(feature = "compact-store")]
        if let Some(base) = self.compact_base.get() {
            return if let Some(history) = base.temporal_node_history(id) {
                let lifetime = history
                    .lifetimes
                    .into_iter()
                    .filter(|lifetime| lifetime.created != EpochId::PENDING)
                    .max_by_key(|lifetime| lifetime.created)?;
                Some((lifetime.created, lifetime.deleted))
            } else {
                base.resolve_node(id)?;
                Some((EpochId::INITIAL, None))
            };
        }
        None
    }

    #[cfg(feature = "vector-index")]
    fn vector_gc_property_history(
        &self,
        id: NodeId,
        property: &PropertyKey,
    ) -> Vec<(EpochId, Value)> {
        if self.contains_node_identity(id) {
            return self.node_property_history_for_key(id, property.as_str());
        }
        #[cfg(feature = "compact-store")]
        if let Some(base) = self.compact_base.get()
            && let Some((_, history)) = base
                .node_property_history(id)
                .into_iter()
                .find(|(key, _)| key == property)
        {
            return history;
        }
        Vec::new()
    }

    /// GC keeps routing vertices for retained tombstones. Their latest property
    /// is often Null, so use the latest committed vector in the newest committed
    /// incarnation, never one from an older incarnation or a shadowed base.
    #[cfg(feature = "vector-index")]
    fn vector_for_gc(&self, id: NodeId, property: &PropertyKey) -> Option<Arc<[f32]>> {
        let (created, deleted) = self.node_gc_incarnation(id)?;
        Self::vector_gc_backing_from_history(
            created,
            deleted,
            &self.vector_gc_property_history(id, property),
        )
    }

    #[cfg(feature = "vector-index")]
    pub(crate) fn vector_gc_backing_from_history(
        created: EpochId,
        deleted: Option<EpochId>,
        history: &[(EpochId, Value)],
    ) -> Option<Arc<[f32]>> {
        history
            .iter()
            .rev()
            .filter(|(epoch, _)| {
                *epoch != EpochId::PENDING
                    && *epoch >= created
                    && deleted.is_none_or(|end| *epoch <= end)
            })
            .find_map(|(_, value)| value_to_vector(value))
    }

    #[cfg(feature = "vector-index")]
    pub(crate) fn vector_gc_property_retired(
        created: EpochId,
        horizon: EpochId,
        properties: &[(EpochId, Value)],
    ) -> bool {
        properties
            .iter()
            .rposition(|(epoch, _)| *epoch <= horizon)
            .is_some_and(|baseline| {
                properties[baseline].0 >= created
                    && properties[baseline..].iter().all(|(epoch, value)| {
                        *epoch != EpochId::PENDING && value_to_vector(value).is_none()
                    })
            })
    }

    #[cfg(feature = "vector-index")]
    pub(crate) fn vector_gc_label_retired(
        created: EpochId,
        horizon: EpochId,
        label: &str,
        labels: &[(EpochId, Vec<arcstr::ArcStr>)],
    ) -> bool {
        labels
            .iter()
            .rposition(|(epoch, _)| *epoch <= horizon)
            .is_some_and(|baseline| {
                labels[baseline].0 >= created
                    && labels[baseline..].iter().all(|(epoch, labels)| {
                        *epoch != EpochId::PENDING
                            && !labels.iter().any(|candidate| candidate.as_str() == label)
                    })
            })
    }

    /// A soft-deleted membership can also retire while its node remains live.
    /// Require a removal witness in this incarnation and no later re-entry;
    /// missing history alone is not permission to discard a physical vertex.
    #[cfg(feature = "vector-index")]
    fn vector_membership_retired(
        &self,
        id: NodeId,
        label: &str,
        property: &PropertyKey,
        horizon: EpochId,
    ) -> bool {
        let Some((created, _)) = self.node_gc_incarnation(id) else {
            return false;
        };
        let properties = self.vector_gc_property_history(id, property);
        if Self::vector_gc_property_retired(created, horizon, &properties) {
            return true;
        }
        let labels = if self.contains_node_identity(id) {
            self.node_label_history(id)
        } else {
            #[cfg(feature = "compact-store")]
            {
                self.compact_base
                    .get()
                    .and_then(|base| base.temporal_node_history(id))
                    .map_or_else(Vec::new, |history| history.label_versions)
            }
            #[cfg(not(feature = "compact-store"))]
            {
                Vec::new()
            }
        };
        Self::vector_gc_label_retired(created, horizon, label, &labels)
    }

    /// Garbage collects soft-deleted nodes in all vector indexes that are no
    /// longer visible to any active snapshot at or above `horizon`.
    ///
    /// For each vector index, rebuilds the HNSW topology retaining only nodes
    /// whose retained incarnations are not all deleted at or below `horizon`.
    /// Retained deleted nodes keep their historical vector backing for routing.
    ///
    /// Mirrors `gc_text_indexes`: the caller (the db-level `gc()`) provides
    /// `min_active_epoch` so the vector indexes compact in lock-step.
    ///
    /// # Errors
    /// Rejects PENDING horizons, denied authority, malformed registry keys or
    /// missing/invalid retained vectors. Collection is atomic per index, not
    /// across independently maintained indexes.
    #[cfg(feature = "vector-index")]
    pub fn gc_vector_indexes(&self, horizon: EpochId) -> grafeo_common::utils::error::Result<()> {
        self.gc_vector_indexes_with_context(
            horizon,
            |id, label, property, physically_live| {
                !self.node_deleted_at_or_below(id, horizon)
                    && (physically_live
                        || !self.vector_membership_retired(id, label, property, horizon))
            },
            |id, property| self.vector_for_gc(id, property),
        )
    }

    /// Shared registry/authority driver. The logical generation supplies only
    /// candidate retention and historical backing, never an alternate index.
    #[cfg(feature = "vector-index")]
    pub(crate) fn gc_vector_indexes_with_context(
        &self,
        horizon: EpochId,
        retain: impl Fn(NodeId, &str, &PropertyKey, bool) -> bool,
        vector: impl Fn(NodeId, &PropertyKey) -> Option<Arc<[f32]>> + Sync,
    ) -> grafeo_common::utils::error::Result<()> {
        use grafeo_common::utils::error::{Error, TransactionError};
        if horizon == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "Vector GC horizon cannot be PENDING".into(),
            ));
        }
        let Some(_mutation) = self.pin_mutation() else {
            return Err(Error::Transaction(TransactionError::ReadOnly));
        };
        // Snapshot the index map (cheap Arc clones); avoids holding the
        // write lock during the (potentially expensive) rebuild.
        let mut indexes: Vec<(String, Arc<VectorIndexKind>)> = {
            let guard = self.vector_indexes.read();
            guard
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect()
        };

        indexes.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        let indexes = indexes
            .into_iter()
            .map(|(key, index)| {
                let (label, property) = decode_index_key(&key).ok_or_else(|| {
                    Error::InvalidValue("Vector GC encountered an invalid registry key".into())
                })?;
                Ok((label.to_owned(), PropertyKey::new(property), index))
            })
            .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
        for (label, property_key, index) in indexes {
            let accessor = |candidate| vector(candidate, &property_key);
            let is_live = |id: NodeId| retain(id, &label, &property_key, index.contains(id));
            index
                .gc(&is_live, &accessor)
                .map_err(|message| Error::InvalidValue(format!("Vector GC failed: {message}")))?;
        }
        Ok(())
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
    ///
    /// # Errors
    /// Rejects epochs below the Text index's retained history floor.
    #[cfg(feature = "text-index")]
    pub fn search_text_visible(
        &self,
        index_key: &str,
        query: &str,
        k: usize,
        epoch: EpochId,
        tx: TransactionId,
    ) -> grafeo_common::utils::error::Result<Vec<(NodeId, f64)>> {
        let Some(index_key) = canonicalize_index_key(index_key) else {
            return Ok(Vec::new());
        };
        let Some((label, property)) = decode_index_key(&index_key) else {
            return Ok(Vec::new());
        };
        let logical_key = logical_index_key(label, property);

        // Coarse predicate-read recording for anti-phantom SSI (Task 7).
        // A Serializable text search reads every document matching `query` in
        // this index; a concurrent indexed SET is a phantom. Record the whole
        // index as read so the existing rw-detection can form the edge. This is
        // a no-op for SI/ReadCommitted (no read tracker registered for `tx`).
        self.record_read_index(tx, &logical_key);

        // Build delta_docs and delta_removed from the overlay for this tx.
        let (delta_docs, delta_removed): (Vec<(NodeId, String)>, FxHashSet<NodeId>) = {
            let overlay = self.text_index_overlay.read();
            match overlay.get(&tx) {
                None => (Vec::new(), FxHashSet::default()),
                Some(delta) => {
                    let mut docs = Vec::new();
                    let mut removed = FxHashSet::default();
                    for (node_id, opt_text) in delta.changes_for(&index_key) {
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
            text_indexes.get(&index_key).cloned()
        };

        match committed_idx {
            Some(idx_arc) => {
                let idx = idx_arc.read();
                idx.search_visible(query, k, epoch, tx, &delta_docs, &delta_removed)
            }
            None => {
                // No committed index — search only the delta docs.
                if delta_docs.is_empty() {
                    return Ok(Vec::new());
                }
                // Build a transient index from the delta and search it.
                let mut transient = crate::index::text::InvertedIndex::new(
                    crate::index::text::BM25Config::default(),
                );
                for (node_id, text) in &delta_docs {
                    transient.insert(*node_id, text);
                }
                Ok(transient.search(query, k))
            }
        }
    }

    // === Transactional text-index buffering ===
    //
    // These two helpers are called from `set_node_property_buffered` /
    // `remove_node_property_buffered` (property_ops.rs) when the property is
    // covered by a text index.  They buffer the change into `text_index_overlay`
    // WITHOUT touching the committed `InvertedIndex`.

    /// Records a vector-index write for anti-phantom SSI (Task 5).
    ///
    /// For every label of `id` that has a vector index on `key`, calls
    /// `record_write_index` so the SSI rw-detection can form the anti-phantom
    /// edge. Does NOT buffer anything into an overlay — the value is already
    /// captured by `tx_property_overlay` via `set_node_property_buffered` /
    /// `remove_node_property_buffered`.
    ///
    /// No-op for SI/ReadCommitted (no write tracker registered).
    #[cfg(feature = "vector-index")]
    pub(super) fn buffer_vector_index_write_record(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let vector_indexes = self.vector_indexes.read();
        if vector_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let registry_key = encode_index_key(label_name, key);
                    if vector_indexes.contains_key(&registry_key) {
                        let logical_key = logical_index_key(label_name, key);
                        self.record_write_index(transaction_id, &logical_key);
                    }
                }
            }
        }
    }

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
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let registry_key = encode_index_key(label_name, key);
                    if text_indexes.contains_key(&registry_key) {
                        let logical_key = logical_index_key(label_name, key);
                        // Coarse index-write recording for anti-phantom SSI (Task 7).
                        // A transactional SET on an indexed property writes to this
                        // index; a concurrent Serializable text search is a phantom.
                        // Record the index write so the rw-detection can form the edge.
                        // No-op for SI/ReadCommitted (no write tracker registered).
                        self.record_write_index(transaction_id, &logical_key);

                        let mut overlay = self.text_index_overlay.write();
                        let delta = overlay.entry(transaction_id).or_default();
                        match value {
                            Value::String(text) => {
                                delta.buffer_set(&registry_key, id, text.to_string());
                            }
                            _ => {
                                // Non-string value: the index treats this as a removal
                                // (mirrors `update_text_index_on_set` which calls
                                // `idx.remove(id)` when the value is not a string).
                                delta.buffer_remove(&registry_key, id);
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
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let registry_key = encode_index_key(label_name, key);
                    if text_indexes.contains_key(&registry_key) {
                        let logical_key = logical_index_key(label_name, key);
                        // Coarse index-write recording for anti-phantom SSI (Task 7).
                        self.record_write_index(transaction_id, &logical_key);

                        self.text_index_overlay
                            .write()
                            .entry(transaction_id)
                            .or_default()
                            .buffer_remove(&registry_key, id);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod property_identity_rollback_tests {
    use super::*;
    use crate::graph::PropertyIndexPredicate;
    use grafeo_common::types::TransactionId;

    #[test]
    fn discarded_fresh_identity_cannot_leave_property_history_for_reuse() {
        let store = LpgStore::new().unwrap();
        store.set_epoch(EpochId::new(1));
        let stable = store.create_node(&["Stable"]);
        store.set_node_property(stable, "state", Value::Int64(10));
        store.create_property_index("state");
        let property = PropertyKey::new("state");
        let registration = store.property_indexes.read()[&property].clone();
        let tx = TransactionId::new(77);
        let aborted = store.create_node_versioned(&["Fresh"], EpochId::PENDING, tx);
        // Legacy direct property APIs can populate a freshly allocated identity
        // before its structural transaction is finalized. Discard must remove it.
        store.set_node_property(aborted, "state", Value::Int64(20));
        store.discard_uncommitted_versions(tx);
        assert!(store.get_node(aborted).is_none());
        assert!(
            store
                .find_nodes_by_property("state", &Value::Int64(20))
                .is_empty()
        );
        assert_eq!(
            store.find_nodes_by_property("state", &Value::Int64(10)),
            vec![stable]
        );
        {
            let indexes = store.property_indexes.read();
            assert!(Arc::ptr_eq(
                &registration.registration,
                &indexes[&property].registration
            ));
            assert!(Arc::ptr_eq(
                &registration.payload,
                &indexes[&property].payload
            ));
            let history = indexes[&property].history.read();
            assert!(
                !history
                    .rows
                    .contains_key(&HashableValue::new(Value::Int64(20)))
            );
            assert_eq!(
                history
                    .candidates(
                        PropertyIndexPredicate::Equal(&Value::Int64(10)),
                        EpochId::new(1)
                    )
                    .unwrap()
                    .0,
                vec![stable]
            );
        }
        store.set_epoch(EpochId::new(2));
        store.create_node_with_id(aborted, &["Reused"]).unwrap();
        store.set_node_property(aborted, "state", Value::Int64(30));
        let history = registration.history.read();
        assert!(
            history
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(20)),
                    EpochId::new(2)
                )
                .unwrap()
                .0
                .is_empty()
        );
        assert_eq!(
            history
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(30)),
                    EpochId::new(2)
                )
                .unwrap()
                .0,
            vec![aborted]
        );
    }
}

#[cfg(all(test, feature = "text-index"))]
mod recorded_text_recovery_tests {
    use super::*;
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::text::{BM25Config, InvertedIndex, TextIndexSection};
    use arcstr::ArcStr;
    use grafeo_common::storage::Section;
    use grafeo_common::types::GraphPath;
    use grafeo_common::utils::error::Result;
    use std::sync::Arc;

    fn text_section(store: &LpgStore) -> TextIndexSection {
        TextIndexSection::from_views(
            store
                .text_index_entries()
                .into_iter()
                .map(|(key, view)| {
                    let (label, property) = decode_index_key(&key).expect("fixture key");
                    (
                        crate::graph::lpg::PhysicalIndexKey::text(
                            GraphPath::root(),
                            label,
                            property,
                        ),
                        view,
                    )
                })
                .collect(),
        )
    }

    fn fixture() -> (LpgStore, NodeId) {
        let store = LpgStore::new().unwrap();
        store.sync_epoch(EpochId::new(1));
        store.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );
        let node = store.create_node_with_props(&["Doc"], [("body", Value::from("originaltoken"))]);
        store.create_property_index("body");
        (store, node)
    }

    #[test]
    fn recorded_text_recovery_is_exact_store_scoped_and_preserves_property_maintenance()
    -> Result<()> {
        let (store, node) = fixture();
        #[cfg(feature = "vector-index")]
        let vector = {
            use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex};
            let vector = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Euclidean,
            ))));
            store.add_vector_index("Doc", "embedding", Arc::clone(&vector));
            store.set_node_property(node, "embedding", Value::Vector(Arc::from([1.0_f32, 0.0])));
            assert!(vector.contains(node));
            vector
        };
        let deleted =
            store.create_node_with_props(&["Doc"], [("body", Value::from("deletedtoken"))]);
        let (other, other_node) = fixture();
        let before = text_section(&store).serialize()?;
        let other_before = text_section(&other).serialize()?;
        store.sync_epoch(EpochId::new(2));
        store.with_recorded_index_recovery(true, false, || {
            store.set_node_property(node, "body", Value::from("intermediate"));
            assert!(store.remove_node_property(node, "body").is_some());
            store.set_node_property(node, "body", Value::from("finaltoken"));
            assert!(store.remove_label(node, "Doc"));
            #[cfg(feature = "vector-index")]
            assert!(
                !vector.contains(node),
                "Vector maintenance is not suppressed"
            );
            assert!(store.add_label(node, "Doc"));
            #[cfg(feature = "vector-index")]
            assert!(vector.contains(node));
            store.replay_node_labels_at_epoch(node, EpochId::new(2), &[])?;
            #[cfg(feature = "vector-index")]
            assert!(!vector.contains(node));
            store.replay_node_labels_at_epoch(node, EpochId::new(2), &[ArcStr::from("Doc")])?;
            #[cfg(feature = "vector-index")]
            assert!(vector.contains(node));
            assert!(store.delete_node(deleted));
            let mutation = store.pin_mutation().expect("retained exact startup proof");
            store.purge_rolled_back_node_from_derived_indexes_inner(deleted, &mutation);
            other.set_node_property(other_node, "body", Value::from("otherchanged"));
            Ok(())
        })?;
        assert_eq!(text_section(&store).serialize()?, before);
        assert_ne!(text_section(&other).serialize()?, other_before);
        assert_eq!(
            store.find_nodes_by_property("body", &Value::from("finaltoken")),
            vec![node]
        );
        assert!(
            store
                .find_nodes_by_property("body", &Value::from("originaltoken"))
                .is_empty()
        );
        assert!(
            store
                .find_nodes_by_property("body", &Value::from("deletedtoken"))
                .is_empty()
        );
        store.set_node_property(node, "body", Value::from("normalmutation"));
        assert_ne!(text_section(&store).serialize()?, before);
        Ok(())
    }

    #[test]
    fn recorded_text_recovery_unwind_error_and_sealed_admission_do_not_leak_suppression()
    -> Result<()> {
        let (store, node) = fixture();
        let before = text_section(&store).serialize()?;
        store.sync_epoch(EpochId::new(2));
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.with_recorded_index_recovery(true, false, || -> Result<()> {
                store.set_node_property(node, "body", Value::from("unpublishedtext"));
                panic!("recovery operation unwinds");
            })
        }));
        assert!(unwind.is_err());
        assert_eq!(text_section(&store).serialize()?, before);
        assert!(
            store
                .with_recorded_index_recovery(true, false, || -> Result<()> {
                    Err(grafeo_common::Error::InvalidValue("fixture failure".into()))
                })
                .is_err()
        );
        store.set_node_property(node, "body", Value::from("normalafterunwind"));
        assert_ne!(text_section(&store).serialize()?, before);
        let owner = WriteAuthority::new();
        assert!(store.seal_unframed_writes(&owner));
        let called = std::cell::Cell::new(false);
        let attempt = || {
            store.with_recorded_index_recovery(true, false, || {
                called.set(true);
                Ok(())
            })
        };
        assert!(attempt().is_err());
        assert!(with_authority(&owner, attempt).is_err());
        assert!(
            !called.get(),
            "even held write authority must not admit sealed startup replay"
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "vector-index"))]
mod recorded_vector_recovery_tests {
    use super::*;
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorStoreSection};
    use grafeo_common::storage::Section;
    use grafeo_common::types::GraphPath;

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn vector(value: f32) -> Value {
        Value::Vector(Arc::from([value, 0.0]))
    }

    fn fixture() -> TestResult<(LpgStore, NodeId, NodeId)> {
        let store = LpgStore::new()?;
        store.sync_epoch(EpochId::new(1));
        store.add_vector_index(
            "Doc",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
                HnswConfig::new(2, DistanceMetric::Euclidean),
                19,
            ))),
        );
        #[cfg(feature = "text-index")]
        store.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(crate::index::text::InvertedIndex::new(
                Default::default(),
            ))),
        );
        let node = store.create_node_with_props(
            &["Doc"],
            [
                ("embedding", vector(1.0)),
                ("body", Value::from("original")),
                ("code", Value::Int64(1)),
            ],
        );
        let deleted = store.create_node_with_props(&["Doc"], [("embedding", vector(2.0))]);
        store.create_property_index("code");
        Ok((store, node, deleted))
    }

    fn image(store: &LpgStore) -> TestResult<Vec<u8>> {
        let view = store
            .get_vector_index("Doc", "embedding")
            .ok_or("missing Vector")?;
        Ok(VectorStoreSection::from_views(vec![(
            crate::graph::lpg::PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
            view,
        )])
        .serialize()?)
    }

    #[test]
    fn recorded_vector_recovery_preserves_images_not_property_or_label_mutations() -> TestResult {
        for record_text in [false, true]
            .into_iter()
            .filter(|enabled| !*enabled || cfg!(feature = "text-index"))
        {
            let (store, node, deleted) = fixture()?;
            let (other, other_node, _) = fixture()?;
            let before = image(&store)?;
            let other_before = image(&other)?;
            #[cfg(feature = "text-index")]
            let text_before = store
                .get_text_index("Doc", "body")
                .ok_or("missing Text")?
                .read()
                .encode_wal_birth()?;
            store.sync_epoch(EpochId::new(2));
            store.with_recorded_index_recovery(record_text, true, || {
                store.set_node_property(node, "embedding", vector(3.0));
                assert!(store.remove_node_property(node, "embedding").is_some());
                store.set_node_property(node, "embedding", vector(4.0));
                store.set_node_property(node, "body", Value::from("changed"));
                store.set_node_property(node, "code", Value::Int64(2));
                assert!(store.remove_label(node, "Doc"));
                assert!(store.add_label(node, "Doc"));
                store.replay_node_labels_at_epoch(node, EpochId::new(2), &[])?;
                store.replay_node_labels_at_epoch(
                    node,
                    EpochId::new(2),
                    &[arcstr::ArcStr::from("Doc")],
                )?;
                assert!(store.delete_node(deleted));
                let created = store.create_node_with_props(&["Doc"], [("embedding", vector(5.0))]);
                assert!(created.is_valid());
                assert!(
                    store
                        .with_recorded_index_recovery(false, true, || Ok(()))
                        .is_err()
                );
                other.set_node_property(other_node, "embedding", vector(6.0));
                Ok(())
            })?;
            assert_eq!(image(&store)?, before);
            assert_ne!(image(&other)?, other_before);
            assert_eq!(
                store.find_nodes_by_property("code", &Value::Int64(2)),
                vec![node]
            );
            assert!(
                store
                    .find_nodes_by_property("code", &Value::Int64(1))
                    .is_empty()
            );
            assert!(store.get_node(deleted).is_none());
            assert!(
                store
                    .get_node(node)
                    .is_some_and(|node| node.has_label("Doc"))
            );
            let history = store.node_property_history_for_key(node, "embedding");
            assert!(history.contains(&(EpochId::new(1), vector(1.0))));
            assert!(history.contains(&(EpochId::new(2), vector(4.0))));
            #[cfg(feature = "text-index")]
            {
                let text_after = store
                    .get_text_index("Doc", "body")
                    .ok_or("missing Text")?
                    .read()
                    .encode_wal_birth()?;
                assert_eq!(text_before == text_after, record_text);
            }
            store.set_node_property(node, "embedding", vector(7.0));
            assert_ne!(
                image(&store)?,
                before,
                "suppression must end with its scope"
            );
        }
        Ok(())
    }

    #[test]
    fn recorded_vector_recovery_unwind_error_and_authority_admission() -> TestResult {
        let (store, node, _) = fixture()?;
        let before = image(&store)?;
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.with_recorded_index_recovery(
                false,
                true,
                || -> grafeo_common::utils::error::Result<()> {
                    store.set_node_property(node, "embedding", vector(3.0));
                    std::panic::resume_unwind(Box::new("injected Vector recovery unwind"));
                },
            )
        }));
        assert!(unwind.is_err());
        assert_eq!(image(&store)?, before);
        assert!(
            store
                .with_recorded_index_recovery(
                    false,
                    true,
                    || -> grafeo_common::utils::error::Result<()> {
                        Err(grafeo_common::Error::InvalidValue(
                            "injected recovery error".into(),
                        ))
                    }
                )
                .is_err()
        );
        store.set_node_property(node, "embedding", vector(4.0));
        assert_ne!(image(&store)?, before);
        let owner = WriteAuthority::new();
        assert!(store.seal_unframed_writes(&owner));
        let called = std::cell::Cell::new(false);
        let attempt = || {
            store.with_recorded_index_recovery(false, true, || {
                called.set(true);
                Ok(())
            })
        };
        assert!(attempt().is_err());
        assert!(with_authority(&owner, attempt).is_err());
        assert!(!called.get());
        Ok(())
    }

    #[cfg(not(feature = "text-index"))]
    #[test]
    fn recorded_recovery_rejects_unavailable_text_family() -> TestResult {
        let (store, _, _) = fixture()?;
        let called = std::cell::Cell::new(false);
        assert!(
            store
                .with_recorded_index_recovery(true, true, || {
                    called.set(true);
                    Ok(())
                })
                .is_err()
        );
        assert!(!called.get());
        Ok(())
    }
}

#[cfg(test)]
mod mutation_barrier_tests {
    use super::*;
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    fn wait_for_pinned_mutation(store: &LpgStore) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !store.mutation_scope_gate.is_locked() {
            assert!(
                Instant::now() < deadline,
                "mutation did not acquire the shared transition barrier"
            );
            thread::yield_now();
        }
        assert!(
            !store.mutation_scope_gate.is_locked_exclusive(),
            "ordinary mutation unexpectedly acquired the exclusive barrier"
        );
    }

    #[test]
    fn property_index_creation_linearizes_before_concurrent_seal() {
        let store = Arc::new(LpgStore::new().unwrap());
        let authority = Arc::new(WriteAuthority::new());
        let property_indexes = store.property_indexes.write();

        let creator_store = Arc::clone(&store);
        let creator = thread::spawn(move || creator_store.create_property_index("serial"));
        wait_for_pinned_mutation(&store);

        let sealer_store = Arc::clone(&store);
        let sealer_authority = Arc::clone(&authority);
        let (sealed_tx, sealed_rx) = mpsc::channel();
        let sealer = thread::spawn(move || {
            sealed_tx
                .send(sealer_store.seal_unframed_writes(&sealer_authority))
                .unwrap();
        });

        assert!(
            sealed_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "seal overtook an already-pinned property-index mutation"
        );
        drop(property_indexes);
        creator.join().unwrap();
        assert!(sealed_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        sealer.join().unwrap();

        assert!(store.has_property_index("serial"));
        assert!(
            !store.drop_property_index("serial"),
            "sealed index DDL must fail without the exact authority"
        );
        with_authority(&authority, || {
            assert!(store.drop_property_index("serial"));
        });
    }

    #[test]
    fn clear_cannot_overtake_a_pinned_property_mutation() {
        let store = Arc::new(LpgStore::new().unwrap());
        let authority = Arc::new(WriteAuthority::new());
        let node = store.create_node(&["Item"]);
        store.create_property_index("serial");
        assert!(store.seal_unframed_writes(&authority));

        let property_indexes = store.property_indexes.write();
        let writer_store = Arc::clone(&store);
        let writer_authority = Arc::clone(&authority);
        let writer = thread::spawn(move || {
            with_authority(&writer_authority, || {
                writer_store.set_node_property(node, "serial", Value::Int64(7));
            });
        });
        wait_for_pinned_mutation(&store);

        let clearer_store = Arc::clone(&store);
        let clearer_authority = Arc::clone(&authority);
        let (started_tx, started_rx) = mpsc::channel();
        let clearer = thread::spawn(move || {
            started_tx.send(()).unwrap();
            with_authority(&clearer_authority, || clearer_store.clear());
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();

        // A clear that does not take the opposing exclusive barrier can erase
        // authoritative entity storage before blocking on the index lock held
        // here. The pinned mutation must keep the whole pre-clear generation
        // intact until it completes.
        let observation_deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < observation_deadline {
            assert!(
                store.get_node(node).is_some(),
                "clear partially erased a generation beneath a pinned mutation"
            );
            thread::yield_now();
        }

        drop(property_indexes);
        writer.join().unwrap();
        clearer.join().unwrap();

        assert_eq!(store.node_count(), 0);
        assert!(!store.has_property_index("serial"));
        assert_eq!(
            store.get_node_property(node, &PropertyKey::new("serial")),
            None
        );
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn retained_text_outer_replacement_cannot_replace_registry_authority() {
        use crate::index::text::{BM25Config, InvertedIndex};

        let store = LpgStore::new().unwrap();
        let retained = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        store.add_text_index("Doc", "body", Arc::clone(&retained));

        let node = store.create_node(&["Doc"]);
        store.set_node_property(node, "body", Value::from("baseline retained target"));
        let registry_view = store
            .get_text_index("Doc", "body")
            .expect("registered text index");
        assert_eq!(registry_view.read().search("baseline", 10)[0].0, node);
        assert_eq!(retained.read().search("baseline", 10)[0].0, node);

        let authority = WriteAuthority::new();
        assert!(store.seal_unframed_writes(&authority));

        // The retained handle remains a fully functional handle to the pinned
        // target, including exact authority enforcement.
        let authorized_alias_node = NodeId::new(9_001);
        with_authority(&authority, || {
            retained
                .write()
                .insert(authorized_alias_node, "authorized retained write");
        });
        assert_eq!(
            registry_view.read().search("authorized", 10)[0].0,
            authorized_alias_node
        );
        retained
            .write()
            .insert(NodeId::new(9_002), "unframed retained write");
        assert!(registry_view.read().search("unframed", 10).is_empty());

        // A hostile caller can replace only its own outer-lock value. The new
        // standalone object is mutable, proving the attack really regained a
        // zero-scope object, but the registry still owns the original sealed
        // target and its query image is unchanged.
        *retained.write() = InvertedIndex::new(BM25Config::default());
        let forged = NodeId::new(9_003);
        retained.write().insert(forged, "forged replacement object");
        assert_eq!(retained.read().search("forged", 10)[0].0, forged);
        assert!(registry_view.read().search("forged", 10).is_empty());
        assert_eq!(
            registry_view.read().search("authorized", 10)[0].0,
            authorized_alias_node
        );

        // Replacement also cannot turn an unframed store mutation into an
        // authorized one, while the exact authority continues to work.
        store.set_node_property(node, "body", Value::from("raw bypass attempt"));
        assert!(registry_view.read().search("bypass", 10).is_empty());
        assert_eq!(registry_view.read().search("baseline", 10)[0].0, node);

        with_authority(&authority, || {
            store.set_node_property(node, "body", Value::from("authorized store update"));
        });
        assert_eq!(registry_view.read().search("update", 10)[0].0, node);
        assert!(registry_view.read().search("baseline", 10).is_empty());
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn retained_text_write_guard_linearizes_compound_with_recursive_seal() {
        use crate::index::text::{BM25Config, InvertedIndex};

        let store = Arc::new(LpgStore::new().unwrap());
        let retained = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        store.add_text_index("Doc", "body", Arc::clone(&retained));
        let document = NodeId::new(9_101);
        retained.write().insert(document, "before compound");

        let writer_index = Arc::clone(&retained);
        let (removed_tx, removed_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let writer = thread::spawn(move || {
            // One caller-owned outer guard is the transaction boundary for this
            // compound. Store/view operations must retain that same gate even
            // though each forwarded mutation takes its own target guard.
            let mut index = writer_index.write();
            assert!(index.remove(document));
            removed_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            index.insert(document, "after compound");
        });
        removed_rx.recv_timeout(Duration::from_secs(5)).unwrap();

        let authority = Arc::new(WriteAuthority::new());
        let sealer_store = Arc::clone(&store);
        let sealer_authority = Arc::clone(&authority);
        let (sealer_started_tx, sealer_started_rx) = mpsc::channel();
        let (sealed_tx, sealed_rx) = mpsc::channel();
        let sealer = thread::spawn(move || {
            sealer_started_tx.send(()).unwrap();
            sealed_tx
                .send(sealer_store.seal_unframed_writes(&sealer_authority))
                .unwrap();
        });
        sealer_started_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();

        assert!(
            sealed_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "recursive seal observed the remove→insert compound's paused midpoint"
        );
        resume_tx.send(()).unwrap();
        writer.join().unwrap();
        assert!(sealed_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        sealer.join().unwrap();

        let view = store
            .get_text_index("Doc", "body")
            .expect("registered text index");
        assert!(view.read().search("before", 10).is_empty());
        assert_eq!(view.read().search("after", 10)[0].0, document);

        retained
            .write()
            .insert(NodeId::new(9_102), "unframed after seal");
        assert!(view.read().search("unframed", 10).is_empty());
    }
}

#[cfg(all(test, feature = "text-index"))]
mod transactional_text_index_guard_tests {
    use super::*;
    use crate::execution::operators::{OperatorError, SharedWriteTracker, WriteTracker};
    use crate::index::text::{BM25Config, InvertedIndex};
    use parking_lot::RwLock;
    use std::sync::{Arc, Mutex};

    struct IndexWriteSpy(Mutex<Vec<(TransactionId, String)>>);

    impl WriteTracker for IndexWriteSpy {
        fn record_node_write(
            &self,
            _transaction_id: TransactionId,
            _node_id: NodeId,
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

        fn record_index_write(&self, transaction_id: TransactionId, key: &str) {
            self.0
                .lock()
                .expect("spy lock")
                .push((transaction_id, key.to_owned()));
        }
    }

    #[test]
    fn versioned_label_and_delete_mutations_publish_text_guards() {
        let store = LpgStore::new().expect("store");
        store.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );
        let node = store.create_node(&["Doc"]);
        let label_target = store.create_node(&[]);
        let tx = TransactionId::new(7_001);
        let spy = Arc::new(IndexWriteSpy(Mutex::new(Vec::new())));
        store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);

        store.set_node_property_versioned(node, "body", Value::from("one"), tx);
        store.remove_node_property_versioned(node, "body", tx);
        store.add_label_buffered(label_target, "Doc", tx);
        store.remove_label_buffered(label_target, "Doc", tx);
        assert!(store.delete_node_transactional(node, store.current_epoch(), tx));

        let calls = spy.0.lock().expect("spy lock").clone();
        assert_eq!(
            calls.len(),
            5,
            "every versioned text membership mutation must publish one guard: {calls:?}"
        );
        assert!(calls.iter().all(|(_, key)| key == "Doc:body"));
        store.unregister_write_tracker(tx);
    }

    #[test]
    fn system_versioned_text_mutation_does_not_publish_ssi_guard() {
        let store = LpgStore::new().expect("store");
        store.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );
        let node = store.create_node(&["Doc"]);
        let spy = Arc::new(IndexWriteSpy(Mutex::new(Vec::new())));
        store.register_write_tracker(
            TransactionId::SYSTEM,
            Arc::clone(&spy) as SharedWriteTracker,
        );

        store.set_node_property_versioned(
            node,
            "body",
            Value::from("recovery"),
            TransactionId::SYSTEM,
        );

        assert!(spy.0.lock().expect("spy lock").is_empty());
        store.unregister_write_tracker(TransactionId::SYSTEM);
    }
}

#[cfg(all(test, feature = "vector-index"))]
mod transactional_vector_index_guard_tests {
    use super::*;
    use crate::execution::operators::{OperatorError, SharedWriteTracker, WriteTracker};
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
    use std::sync::{Arc, Mutex};

    struct IndexWriteSpy(Mutex<Vec<String>>);

    impl WriteTracker for IndexWriteSpy {
        fn record_node_write(
            &self,
            _transaction_id: TransactionId,
            _node_id: NodeId,
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

        fn record_index_write(&self, _transaction_id: TransactionId, key: &str) {
            self.0.lock().expect("spy lock").push(key.to_owned());
        }
    }

    #[test]
    fn versioned_vector_set_and_delete_publish_index_guards() {
        let store = LpgStore::new().expect("store");
        let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            3,
            DistanceMetric::Cosine,
        ))));
        store.add_vector_index("Doc", "embedding", index);
        let node = store.create_node(&["Doc"]);
        let tx = TransactionId::new(7_101);
        let spy = Arc::new(IndexWriteSpy(Mutex::new(Vec::new())));
        store.register_write_tracker(tx, Arc::clone(&spy) as SharedWriteTracker);

        store.set_node_property_versioned(
            node,
            "embedding",
            Value::Vector(vec![1.0_f32, 0.0, 0.0].into()),
            tx,
        );
        assert!(store.delete_node_transactional(node, store.current_epoch(), tx));

        assert_eq!(
            spy.0.lock().expect("spy lock").as_slice(),
            ["Doc:embedding", "Doc:embedding"]
        );
        store.unregister_write_tracker(tx);
    }
}
