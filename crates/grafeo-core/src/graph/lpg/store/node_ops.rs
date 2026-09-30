use super::{LpgStore, NodeIdentityReservation};
use crate::graph::lpg::{Node, NodeRecord};
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, LabelId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::sync::atomic::Ordering;

#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{HotVersionRef, VersionIndex, VersionRef};

#[derive(Default)]
struct IndexNodeDelta {
    properties: Vec<(PropertyKey, super::PropOp)>,
    labels: Vec<(u32, super::LabelOp)>,
}

#[cfg(all(test, feature = "tiered-storage", not(target_arch = "wasm32")))]
mod tiered_lock_order_tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn allocation_releases_registry_before_publication(
        node_publication: bool,
        create: impl FnOnce(&LpgStore, NodeId, NodeId) + Send,
    ) {
        let store = LpgStore::new().unwrap();
        let src = store.create_node(&[]);
        let dst = store.create_node(&[]);
        let epoch = store.current_epoch();
        let used = store.arena_allocator.arena(epoch).unwrap().total_used();
        let progressed = std::thread::scope(|scope| {
            // Block publication, without blocking allocation or identity checks.
            let nodes = node_publication.then(|| store.node_versions.read());
            let edges = (!node_publication).then(|| store.edge_versions.read());
            let creator_store = &store;
            let creator = scope.spawn(move || create(creator_store, src, dst));
            let deadline = Instant::now() + Duration::from_secs(5);
            let allocated = loop {
                if store.arena_allocator.arena(epoch).unwrap().total_used() > used {
                    break true;
                }
                if Instant::now() >= deadline {
                    break false;
                }
                std::thread::yield_now();
            };
            let (send, receive) = mpsc::channel();
            let allocator_store = &store;
            let allocator = scope.spawn(move || {
                let result = allocator_store
                    .arena_allocator
                    .ensure_epoch(EpochId::new(100));
                send.send(result.is_ok()).unwrap();
            });
            // A queued registry writer must finish while publication is blocked.
            // Release both guards and join before asserting, even on the old bug.
            let progressed = receive.recv_timeout(Duration::from_secs(2)) == Ok(true);
            drop(edges);
            drop(nodes);
            creator.join().unwrap();
            allocator.join().unwrap();
            allocated && progressed
        });
        assert!(
            progressed,
            "allocation retained the arena registry during publication"
        );
    }

    #[test]
    fn node_creation_releases_arena_before_publication() {
        allocation_releases_registry_before_publication(true, |store, _, _| {
            assert!(store.create_node(&["Created"]).is_valid());
        });
    }

    #[test]
    fn node_recovery_releases_arena_before_publication() {
        allocation_releases_registry_before_publication(true, |store, _, _| {
            store
                .create_node_with_id(NodeId::new(50), &["Recovered"])
                .unwrap();
        });
    }

    #[test]
    fn edge_creation_releases_arena_before_publication() {
        allocation_releases_registry_before_publication(false, |store, src, dst| {
            assert!(store.create_edge(src, dst, "R").is_valid());
        });
    }

    #[test]
    fn edge_recovery_releases_arena_before_publication() {
        allocation_releases_registry_before_publication(false, |store, src, dst| {
            store
                .create_edge_with_id(EdgeId::new(50), src, dst, "R")
                .unwrap();
        });
    }

    #[test]
    fn edge_transport_releases_arena_before_publication() {
        allocation_releases_registry_before_publication(false, |store, src, dst| {
            assert!(
                store
                    .create_transport_edge_with_id(EdgeId::new(50), src, dst, "R")
                    .unwrap()
                    .is_some()
            );
        });
    }

    #[test]
    fn edge_batch_releases_arena_before_publication() {
        allocation_releases_registry_before_publication(false, |store, src, dst| {
            assert_eq!(
                store
                    .batch_create_edges(&[(src, dst, "R"), (src, dst, "R")])
                    .len(),
                2
            );
        });
    }
    #[test]
    fn arena_denial_preserves_graph_publication_and_identity_metadata() {
        use grafeo_common::memory::arena::{AllocError, ArenaAllocator};

        let mut store = LpgStore::new().unwrap();
        // Install the small allocator before publishing any hot offsets.
        store.arena_allocator = std::sync::Arc::new(ArenaAllocator::with_chunk_size(512).unwrap());
        let src = store.create_node_with_props(&["Existing"], [("kept", 7i64)]);
        let dst = store.create_node(&["Existing"]);
        let edge = store.create_edge(src, dst, "EXISTING");
        assert!(src.is_valid() && dst.is_valid() && edge.is_valid());
        let epoch = store.current_epoch();
        {
            let arena = store.arena_allocator.arena(epoch).unwrap();
            let remaining = arena.total_allocated() - arena.total_used();
            assert!(remaining > 0);
            arena.alloc(remaining, 1).unwrap();
            assert_eq!(arena.stats().chunk_count, 1);
            assert_eq!(arena.total_used(), 512);
        }
        let node_high_water = store.next_node_id();
        let edge_high_water = store.next_edge_id();
        let labels = store.all_labels();
        let edge_types = store.id_to_edge_type.read().clone();
        let type_counts = store.edge_type_live_counts.read().clone();

        // The generated-ID contract consumes its reservation number on denial,
        // but must publish no identity, label, property, or graph record.
        assert_eq!(
            store.create_node_with_props(&["Denied"], [("unpublished", 99i64)]),
            NodeId::INVALID
        );
        assert_eq!(store.next_node_id(), node_high_water + 1);
        let rejected_id = NodeId::new(node_high_water);
        assert!(!store.contains_node_identity(rejected_id));
        assert_eq!(
            store.get_node_property(rejected_id, &PropertyKey::new("unpublished")),
            None
        );
        // Repeating the same explicit identities proves failed reservations
        // were released. These Result paths preserve their high-water marks.
        for _ in 0..2 {
            assert!(matches!(
                store.create_node_with_id(NodeId::new(50), &["DeniedRecovery"]),
                Err(AllocError::InsufficientSpace)
            ));
            assert!(matches!(
                store.create_edge_with_id(EdgeId::new(50), src, dst, "DENIED"),
                Err(grafeo_common::Error::Storage(
                    grafeo_common::utils::error::StorageError::Full
                ))
            ));
        }
        assert_eq!(store.next_node_id(), node_high_water + 1);
        assert_eq!(store.next_edge_id(), edge_high_water);
        assert_eq!(store.current_epoch(), epoch);
        assert_eq!(store.node_count(), 2);
        assert_eq!(store.edge_count(), 1);
        assert_eq!(store.node_versions.read().len(), 2);
        assert_eq!(store.edge_versions.read().len(), 1);
        assert!(store.contains_node_identity(src) && store.contains_node_identity(dst));
        assert!(!store.contains_node_identity(NodeId::new(50)));
        assert!(store.get_edge(EdgeId::new(50)).is_none());
        assert!(
            store
                .node_identity_reservations
                .identities
                .lock()
                .is_empty()
        );
        assert!(
            store
                .edge_identity_reservations
                .identities
                .lock()
                .is_empty()
        );
        assert_eq!(store.all_labels(), labels);
        assert_eq!(*store.id_to_edge_type.read(), edge_types);
        assert_eq!(*store.edge_type_live_counts.read(), type_counts);
        assert_eq!(
            store.get_node_property(src, &PropertyKey::new("kept")),
            Some(Value::Int64(7))
        );
        assert_eq!(
            store
                .edges_from(src, crate::graph::Direction::Outgoing)
                .collect::<Vec<_>>(),
            vec![(dst, edge)]
        );
        let arena = store.arena_allocator.arena(epoch).unwrap();
        assert_eq!(arena.stats().chunk_count, 1);
        assert_eq!(arena.total_allocated(), 512);
        assert_eq!(arena.total_used(), 512);
    }
}

impl LpgStore {
    /// Creates a new node with the given labels.
    ///
    /// Uses the system transaction for non-transactional operations.
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        self.create_node_versioned(labels, self.current_epoch(), TransactionId::SYSTEM)
    }

    /// Registers labels for a node: builds the label ID set, updates the
    /// label index (single lock acquisition), and stores the node-to-labels
    /// mapping.
    pub(super) fn register_node_labels(&self, id: NodeId, labels: &[&str], epoch: EpochId) {
        use grafeo_common::temporal::VersionLog;

        let mut node_label_set = FxHashSet::default();
        let mut label_ids = Vec::with_capacity(labels.len());
        for label in labels {
            let label_id = self.get_or_create_label_id(label);
            node_label_set.insert(label_id);
            label_ids.push(label_id);
        }

        // Update label index with a single lock acquisition
        let mut index = self.label_index.write();
        for label_id in label_ids {
            if index.len() <= label_id as usize {
                index.resize_with(label_id as usize + 1, FxHashMap::default);
            }
            index[label_id as usize].insert(id, ());
        }
        drop(index);

        self.node_labels
            .write()
            .insert(id, VersionLog::with_value(epoch, node_label_set));
    }

    /// Builds a `Node` populated with labels and properties for the given ID.
    ///
    /// Returns the current (latest) state of the node.
    fn build_node(&self, id: NodeId) -> Node {
        self.work_counters.record_node_materialization();
        let mut node = Node::new(id);

        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();

        if let Some(log) = node_labels.get(&id)
            && let Some(label_ids) = log.latest()
        {
            for &label_id in label_ids {
                if let Some(label) = registry.get_name(label_id) {
                    node.labels.push(label.clone());
                }
            }
        }

        node.properties = self.node_properties.get_all(id).into_iter().collect();
        node
    }

    /// Builds a `Node` with labels and properties as they were at a specific epoch.
    ///
    /// This is the critical method that makes `get_node_at_epoch()` return
    /// correct historical property values instead of current ones.
    fn build_node_at(&self, id: NodeId, epoch: EpochId) -> Node {
        self.work_counters.record_node_materialization();
        let mut node = Node::new(id);

        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        if let Some(log) = node_labels.get(&id)
            && let Some(label_ids) = log.at(epoch)
        {
            for &label_id in label_ids {
                if let Some(label) = registry.get_name(label_id) {
                    node.labels.push(label.clone());
                }
            }
        }

        node.properties = self
            .node_properties
            .get_all_at(id, epoch)
            .into_iter()
            .collect();
        node
    }

    /// Creates a new node with the given labels within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let Some(_mutation) = self.pin_mutation() else {
            return NodeId::INVALID;
        };
        let Some(identity_reservation) = NodeIdentityReservation::generated(self) else {
            return NodeId::INVALID;
        };
        let id = identity_reservation.id();

        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };

        // Resolve label IDs before registering (so we can fan out coarse writes).
        let label_ids: Vec<LabelId> = labels
            .iter()
            .map(|l| LabelId::from(self.get_or_create_label_id(l)))
            .collect();

        self.register_node_labels(id, labels, version_epoch);

        let chain = VersionChain::with_initial(record, version_epoch, transaction_id);
        self.nodes.write().insert(id, chain);
        self.record_pending_node(transaction_id, id);

        // Phantom coarse write: record Label(L) for each new node label so a
        // concurrent escalated Label(L) reader forms an rw-antidependency.
        self.record_coarse_node_write(transaction_id, id, &label_ids);

        self.live_node_count.fetch_add(1, Ordering::Relaxed);
        identity_reservation.commit();
        id
    }

    /// Creates a new node with the given labels within a transaction context.
    /// (Tiered storage version: stores data in arena, metadata in VersionIndex)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let Some(_mutation) = self.pin_mutation() else {
            return NodeId::INVALID;
        };
        let Some(identity_reservation) = NodeIdentityReservation::generated(self) else {
            return NodeId::INVALID;
        };
        let id = identity_reservation.id();

        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };

        // Allocate record in arena and get offset (create epoch if needed)
        let Ok(arena) = self.arena_allocator.arena_or_create(epoch) else {
            return NodeId::INVALID;
        };
        let Ok((offset, _stored)) = arena.alloc_value_with_offset(record) else {
            return NodeId::INVALID;
        };
        // The mutation pin keeps the arena alive through publication. Release
        // its registry guard before taking version locks: readers acquire them
        // in the opposite order, and a queued arena writer blocks new readers.
        drop(arena);

        // Resolve/register labels only after fallible arena work succeeds.
        let label_ids: Vec<LabelId> = labels
            .iter()
            .map(|l| LabelId::from(self.get_or_create_label_id(l)))
            .collect();
        self.register_node_labels(id, labels, version_epoch);

        // Create HotVersionRef pointing to arena data
        let hot_ref = HotVersionRef::new(version_epoch, epoch, offset, transaction_id);

        // Create or update version index
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            index.add_hot(hot_ref);
        } else {
            versions.insert(id, VersionIndex::with_initial(hot_ref));
        }
        drop(versions);
        self.record_pending_node(transaction_id, id);

        // Phantom coarse write: record Label(L) for each new node label.
        self.record_coarse_node_write(transaction_id, id, &label_ids);

        self.live_node_count.fetch_add(1, Ordering::Relaxed);
        identity_reservation.commit();
        id
    }

    /// Creates a new node with labels and properties.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> NodeId {
        self.create_node_with_props_versioned(
            labels,
            properties,
            self.current_epoch(),
            TransactionId::SYSTEM,
        )
    }

    /// Creates a new node with labels and properties within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let Some(_mutation) = self.pin_mutation() else {
            return NodeId::INVALID;
        };
        let id = self.create_node_versioned(labels, epoch, transaction_id);
        if !id.is_valid() {
            return NodeId::INVALID;
        }
        let properties: Vec<(PropertyKey, Value)> = properties
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();

        for (key, value) in &properties {
            // A compound PENDING create must not publish any derived state.
            if transaction_id == TransactionId::SYSTEM {
                self.update_property_index_on_set(id, key, value);
            }
            self.node_properties
                .set(id, key.clone(), value.clone(), epoch);
        }
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        if transaction_id == TransactionId::SYSTEM {
            for (key, value) in &properties {
                let _ = value;
                #[cfg(feature = "vector-index")]
                self.refresh_vector_indexes_for_property(id, key.as_str());
                #[cfg(feature = "text-index")]
                self.update_text_index_on_set(id, key.as_str(), value);
            }
        }

        // Update props_count in record
        let count = u16::try_from(self.node_properties.get_all(id).len()).unwrap_or(u16::MAX);
        if let Some(chain) = self.nodes.write().get_mut(&id)
            && let Some(record) = chain.latest_mut()
        {
            record.props_count = count;
        }

        id
    }

    /// Creates a new node with labels and properties within a transaction context.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let Some(_mutation) = self.pin_mutation() else {
            return NodeId::INVALID;
        };
        let id = self.create_node_versioned(labels, epoch, transaction_id);
        if !id.is_valid() {
            return NodeId::INVALID;
        }
        let properties: Vec<(PropertyKey, Value)> = properties
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();

        for (key, value) in &properties {
            if transaction_id == TransactionId::SYSTEM {
                self.update_property_index_on_set(id, key, value);
            }
            self.node_properties
                .set(id, key.clone(), value.clone(), epoch);
        }
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        if transaction_id == TransactionId::SYSTEM {
            for (key, value) in &properties {
                let _ = value;
                #[cfg(feature = "vector-index")]
                self.refresh_vector_indexes_for_property(id, key.as_str());
                #[cfg(feature = "text-index")]
                self.update_text_index_on_set(id, key.as_str(), value);
            }
        }

        // Note: props_count in record is not updated for tiered storage.
        // The record is immutable once allocated in the arena.

        id
    }

    /// Gets a node by ID (latest visible version).
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let _read = self.pin_read();
        self.get_node_at_epoch(id, self.current_epoch())
    }

    /// Gets a node by ID at a specific epoch.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let _read = self.pin_read();
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = chain.visible_at(epoch)?;
        if record.is_deleted() {
            return None;
        }
        drop(nodes);

        // Fast path: current-epoch reads use latest() instead of at(epoch).
        // Safe because non-transactional reads have no PENDING entries.
        if epoch >= self.current_epoch() {
            Some(self.build_node(id))
        } else {
            Some(self.build_node_at(id, epoch))
        }
    }

    /// Gets a node by ID at a specific epoch.
    /// (Tiered storage version: reads from arena via VersionIndex)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let _read = self.pin_read();
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_at(epoch)?;
        let record = self.read_node_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);

        if epoch >= self.current_epoch() {
            Some(self.build_node(id))
        } else {
            Some(self.build_node_at(id, epoch))
        }
    }

    /// Confirms structural visibility without recording a query read.
    #[cfg(not(feature = "tiered-storage"))]
    fn node_read_visibility(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<()> {
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = match transaction_id {
            Some(tx) => chain.visible_to(epoch, tx),
            None => chain.visible_at(epoch),
        }?;
        if record.is_deleted() {
            return None;
        }
        Some(())
    }

    /// Tiered structural visibility uses the same non-recording boundary.
    #[cfg(feature = "tiered-storage")]
    fn node_read_visibility(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<()> {
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = match transaction_id {
            Some(tx) => index.visible_to(epoch, tx),
            None => index.visible_at(epoch),
        }?;
        let record = self.read_node_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        Some(())
    }

    /// Gets a node visible to a specific transaction.
    #[must_use]
    #[doc(hidden)]
    pub fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        let _read = self.pin_read();
        self.node_read_visibility(id, epoch, Some(transaction_id))?;
        // Keep query materialization on the tracking accessors. Maintenance
        // preparation below deliberately does not record these query reads.
        let mut node = self.build_node_at(id, epoch);
        let mut labels: Vec<_> = self
            .read_node_labels_visible(id, epoch, Some(transaction_id))
            .into_iter()
            .collect();
        labels.sort_unstable();
        node.labels = labels.into_iter().collect();
        node.properties = self
            .read_node_properties_visible(id, epoch, Some(transaction_id))
            .into_iter()
            .collect();
        self.record_read_node_materialized(transaction_id, id);
        Some(node)
    }

    /// Captures this transaction's final delta once, grouped by node, so a
    /// full index build does not rescan all writes for every candidate row.
    pub(crate) fn index_node_rows(
        &self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
        ids: impl IntoIterator<Item = NodeId>,
    ) -> Vec<Node> {
        let mut changes: FxHashMap<NodeId, IndexNodeDelta> = FxHashMap::default();
        if let Some(tx) = transaction_id {
            let overlay = self.tx_property_overlay.read();
            if let Some(delta) = overlay.get(&tx) {
                for ((id, key), operation) in &delta.node_props {
                    changes
                        .entry(*id)
                        .or_default()
                        .properties
                        .push((key.clone(), operation.clone()));
                }
                for ((id, label_id), operation) in &delta.node_labels {
                    changes
                        .entry(*id)
                        .or_default()
                        .labels
                        .push((*label_id, *operation));
                }
            }
        }
        ids.into_iter()
            .filter_map(|id| self.index_node_row(id, epoch, transaction_id, changes.remove(&id)))
            .collect()
    }

    /// Materializes one final preparation row without invoking read trackers.
    /// Structural visibility is shared with the public versioned reader.
    fn index_node_row(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
        delta: Option<IndexNodeDelta>,
    ) -> Option<Node> {
        self.node_read_visibility(id, epoch, transaction_id)?;
        // Inline-created labels have a PENDING tail; only their creating
        // transaction may use it instead of the committed label version.
        let own_create = transaction_id.is_some_and(|tx| {
            #[cfg(not(feature = "tiered-storage"))]
            {
                self.nodes
                    .read()
                    .get(&id)
                    .is_some_and(|chain| chain.modified_by(tx))
            }
            #[cfg(feature = "tiered-storage")]
            {
                self.node_versions
                    .read()
                    .get(&id)
                    .is_some_and(|index| index.modified_by(tx))
            }
        });
        let mut properties = self.node_properties.get_all_at(id, epoch);
        let mut label_ids = {
            let node_labels = self.node_labels.read();
            let log = node_labels.get(&id);
            if own_create {
                log.and_then(|log| log.latest().cloned())
                    .unwrap_or_default()
            } else {
                log.and_then(|log| log.at(epoch).cloned())
                    .unwrap_or_default()
            }
        };
        if let Some(delta) = delta {
            for (key, operation) in delta.properties {
                match operation {
                    // Committed property columns encode NULL as absence. The
                    // preparation image must match that final row, not expose
                    // the overlay's tombstone as an indexable property value.
                    super::PropOp::Remove | super::PropOp::Set(Value::Null) => {
                        properties.remove(&key);
                    }
                    super::PropOp::Set(value) => {
                        properties.insert(key, value);
                    }
                }
            }
            for (label_id, operation) in delta.labels {
                match operation {
                    super::LabelOp::Add => {
                        label_ids.insert(label_id);
                    }
                    super::LabelOp::Remove => {
                        label_ids.remove(&label_id);
                    }
                }
            }
        }
        let registry = self.label_registry.read();
        let mut labels: Vec<_> = label_ids
            .into_iter()
            .filter_map(|label_id| registry.get_name(label_id).cloned())
            .collect();
        labels.sort_unstable();
        let mut node = Node::new(id);
        node.labels = labels.into_iter().collect();
        node.properties = properties.into_iter().collect();
        Some(node)
    }

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Node)`.
    /// Without `temporal`: labels and properties reflect the current state.
    /// With `temporal`: each version has correct historical properties and labels.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let _read = self.pin_read();
        let nodes = self.nodes.read();
        let Some(chain) = nodes.get(&id) else {
            return Vec::new();
        };

        chain
            .history()
            .map(|(info, _record)| {
                let node = self.build_node_at(id, info.created_epoch);
                (info.created_epoch, info.deleted_epoch, node)
            })
            .collect()
    }

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let _read = self.pin_read();
        let versions = self.node_versions.read();
        let Some(index) = versions.get(&id) else {
            return Vec::new();
        };

        index
            .version_history()
            .into_iter()
            .map(|(created, deleted, _vref)| {
                let node = self.build_node_at(id, created);
                (created, deleted, node)
            })
            .collect()
    }

    /// Reads a NodeRecord from arena (hot) or epoch store (cold) using a VersionRef.
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub(super) fn read_node_record(&self, version_ref: &VersionRef) -> Option<NodeRecord> {
        match version_ref {
            VersionRef::Hot(hot_ref) => {
                let arena = self
                    .arena_allocator
                    .arena(hot_ref.arena_epoch)
                    .expect("arena epoch must exist for hot version ref");
                // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                Some(*record)
            }
            VersionRef::Cold(cold_ref) => {
                // Read from compressed epoch store
                self.epoch_store
                    .get_node(cold_ref.epoch, cold_ref.block_offset, cold_ref.length)
            }
            _ => None,
        }
    }

    /// Returns the committed label IDs currently registered for a node.
    ///
    /// Reads the `node_labels` map directly (no read recording, no registry
    /// name resolution). Used by the transactional delete path to fan out a
    /// coarse `Label(L)` phantom write for every label the node leaves — the
    /// committed set is still present at the delete site because transactional
    /// deletes defer current `label_index` removal to
    /// [`finalize_deletes_by_id`](Self::finalize_deletes_by_id).
    ///
    /// Any label the *deleting* transaction itself added/removed in-flight has
    /// already recorded its own coarse write (via `add_label_buffered` /
    /// `remove_label_buffered`), so the committed base set is sufficient here.
    pub(crate) fn committed_node_label_ids(&self, id: NodeId) -> Vec<LabelId> {
        let node_labels = self.node_labels.read();
        node_labels
            .get(&id)
            .and_then(|log| log.latest())
            .map(|set| set.iter().copied().map(LabelId::from).collect())
            .unwrap_or_default()
    }

    /// Returns the node's complete label-set history in ascending epoch order.
    ///
    /// Each version contains the full set visible from that epoch, sorted by
    /// label name for deterministic persistence. Same-epoch entries retain
    /// their original order; the last entry at an epoch is authoritative.
    /// Closed/deleted nodes remain available because deletion removes only
    /// current label-index membership, not this temporal log.
    #[must_use]
    pub fn node_label_history(&self, id: NodeId) -> Vec<(EpochId, Vec<ArcStr>)> {
        let _read = self.pin_read();
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        let Some(log) = node_labels.get(&id) else {
            return Vec::new();
        };
        log.history()
            .iter()
            .map(|(epoch, label_ids)| {
                let mut labels: Vec<ArcStr> = label_ids
                    .iter()
                    .filter_map(|&label_id| registry.get_name(label_id).cloned())
                    .collect();
                labels.sort_unstable();
                labels.dedup();
                (*epoch, labels)
            })
            .collect()
    }

    /// Removes a deleted node from the current label index without discarding
    /// its temporal label log.
    ///
    /// Structural visibility is governed by the node version chain. Retaining
    /// `node_labels[id]` therefore cannot resurrect the node in current reads,
    /// while it is required to reconstruct the labels carried by a closed
    /// lifetime. Acquire `label_index` before `node_labels` to follow the store
    /// lock order documented on [`LpgStore`].
    fn remove_node_from_current_label_index(&self, id: NodeId) {
        let mut index = self.label_index.write();
        let node_labels = self.node_labels.read();
        let Some(label_ids) = node_labels.get(&id).and_then(|log| log.latest()) else {
            return;
        };
        for &label_id in label_ids {
            if let Some(set) = index.get_mut(label_id as usize) {
                set.remove(&id);
            }
        }
    }

    /// Deletes a node and all its edges (using latest epoch).
    pub fn delete_node(&self, id: NodeId) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        self.delete_node_at_epoch(id, self.current_epoch())
    }

    /// Deletes a node at a specific epoch.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut nodes = self.nodes.write();
        if let Some(chain) = nodes.get_mut(&id) {
            // Check if visible at this epoch (not already deleted)
            if let Some(record) = chain.visible_at(epoch) {
                if record.is_deleted() {
                    return false;
                }
            } else {
                // Not visible at this epoch (already deleted or doesn't exist)
                return false;
            }

            // Mark the version chain as deleted at this epoch
            chain.mark_deleted(epoch, TransactionId::SYSTEM);
            drop(nodes);

            // Derived state must be removed while properties still exist. This
            // central path covers raw-store and Session/GQL deletion alike.
            self.remove_node_from_derived_indexes(id);

            // Remove only current index membership. The VersionLog remains so
            // closed lifetimes retain their historical labels.
            self.remove_node_from_current_label_index(id);

            // Remove properties
            self.node_properties.remove_all(id, epoch);

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Deletes a node at a specific epoch.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Check if visible at this epoch
            if let Some(version_ref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&version_ref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }

            // Mark as deleted in version index
            index.mark_deleted(epoch, TransactionId::SYSTEM);
            drop(versions);

            // Derived state must be removed while properties still exist.
            self.remove_node_from_derived_indexes(id);

            // Remove only current index membership. The VersionLog remains so
            // closed lifetimes retain their historical labels.
            self.remove_node_from_current_label_index(id);

            // Remove properties
            self.node_properties.remove_all(id, epoch);

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Deletes a node within a transaction using PENDING-epoch isolation.
    ///
    /// Unlike `delete_node_at_epoch`, this method:
    /// 1. Marks the version with `deleted_epoch = EpochId::PENDING` so the deleting
    ///    transaction sees the node as gone (read-your-writes), while all other
    ///    sessions still see it (PENDING > any real epoch in `is_visible_at`).
    /// 2. Does NOT eagerly strip label-index or properties — deferred to commit
    ///    via `finalize_deletes_by_id`.
    /// 3. Records the node ID in `pending_tx_deletes` so commit can finalize and
    ///    rollback can clear the deferred set (calling `unmark_deleted_by` via
    ///    `rollback_pending_deletes`).
    ///
    /// NOTE: `DETACH DELETE` edge adjacency tombstones still use `TransactionId::SYSTEM`
    /// with eager marking (see `delete_node_edges`). Deferring adjacency isolation for
    /// nodes-with-edges is tracked separately.
    /// TODO(unified-mvcc): defer adjacency tombstones for transactional DETACH.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_node_transactional(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut nodes = self.nodes.write();
        if let Some(chain) = nodes.get_mut(&id) {
            if let Some(record) = chain.visible_to(epoch, transaction_id) {
                if record.is_deleted() {
                    return false;
                }
            } else {
                return false;
            }

            // Capture the node's committed labels BEFORE tombstoning, for the
            // coarse phantom write below (the node leaves every label set).
            let label_ids = self.committed_node_label_ids(id);

            // Stamp PENDING so the deleter sees it gone, others still see it.
            chain.mark_deleted(EpochId::PENDING, transaction_id);
            drop(nodes);

            // Record for deferred finalize/rollback — label-index/property removal
            // is deferred to `finalize_deletes_by_id` at commit time.
            self.pending_tx_deletes
                .write()
                .entry(transaction_id)
                .or_default()
                .push(id);

            // Phantom coarse write: deleting the node removes it from every :L
            // set, so an escalated Label(L) reader must form an rw-antidependency.
            self.record_coarse_node_write(transaction_id, id, &label_ids);
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            self.record_index_writes_for_node_delete(id, transaction_id);

            true
        } else {
            false
        }
    }

    /// Deletes a node within a transaction using PENDING-epoch isolation.
    /// (Tiered storage version)
    ///
    /// Stamps `deleted_epoch = EpochId::PENDING` so other sessions still see
    /// the node while the delete is uncommitted. Label-index and property removal
    /// are deferred to `finalize_deletes_by_id` at commit time.
    /// TODO(unified-mvcc): defer adjacency tombstones for transactional DETACH.
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_node_transactional(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            if let Some(version_ref) = index.visible_to(epoch, transaction_id) {
                if let Some(record) = self.read_node_record(&version_ref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }

            // Capture the node's committed labels BEFORE tombstoning, for the
            // coarse phantom write below (the node leaves every label set).
            let label_ids = self.committed_node_label_ids(id);

            // Stamp PENDING so the deleter sees it gone, others still see it.
            index.mark_deleted(EpochId::PENDING, transaction_id);
            drop(versions);

            // Record for deferred finalize/rollback.
            self.pending_tx_deletes
                .write()
                .entry(transaction_id)
                .or_default()
                .push(id);

            // Phantom coarse write: deleting the node removes it from every :L
            // set, so an escalated Label(L) reader must form an rw-antidependency.
            self.record_coarse_node_write(transaction_id, id, &label_ids);
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            self.record_index_writes_for_node_delete(id, transaction_id);

            true
        } else {
            false
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    ///
    /// Call this before `delete_node()` if you want to remove a node that
    /// has edges. Grafeo doesn't auto-delete edges, you have to be explicit.
    ///
    /// Uses a `HashSet` to dedup self-loops (where src == dst) and holds
    /// the edges write lock for the entire batch to prevent concurrent
    /// readers from observing a partially detached node.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let epoch = self.current_epoch();

        // Collect all edge IDs into a set to dedup self-loops
        let mut edge_ids: FxHashSet<EdgeId> = FxHashSet::default();

        for (_, edge_id) in self.forward_adj.edges_from(node_id) {
            edge_ids.insert(edge_id);
        }

        if let Some(ref backward) = self.backward_adj {
            for (_, edge_id) in backward.edges_from(node_id) {
                edge_ids.insert(edge_id);
            }
        } else {
            // No backward adjacency: scan all edges for incoming
            let edges = self.edges.read();
            for (id, chain) in edges.iter() {
                if let Some(r) = chain.visible_at(epoch)
                    && !r.is_deleted()
                    && r.dst == node_id
                {
                    edge_ids.insert(*id);
                }
            }
        }

        if edge_ids.is_empty() {
            return;
        }

        // Hold the write lock for the entire batch: mark all edges as deleted
        // atomically so concurrent readers never see a partially detached node.
        let deleted: Vec<(EdgeId, NodeId, NodeId, u32)>;
        {
            let mut edges = self.edges.write();
            deleted = edge_ids
                .iter()
                .filter_map(|&edge_id| {
                    let chain = edges.get_mut(&edge_id)?;
                    let record = chain.visible_at(epoch)?;
                    if record.is_deleted() {
                        return None;
                    }
                    let src = record.src;
                    let dst = record.dst;
                    let type_id = record.type_id;
                    chain.mark_deleted(epoch, TransactionId::SYSTEM);
                    Some((edge_id, src, dst, type_id))
                })
                .collect();
        }

        // Batch-update adjacency under a single write lock so concurrent
        // readers of neighbors()/edges_from() never see a partially detached node.
        let fwd_batch: Vec<(NodeId, EdgeId)> =
            deleted.iter().map(|&(eid, src, _, _)| (src, eid)).collect();
        self.forward_adj.batch_mark_deleted(&fwd_batch);

        if let Some(ref backward) = self.backward_adj {
            let bwd_batch: Vec<(NodeId, EdgeId)> =
                deleted.iter().map(|&(eid, _, dst, _)| (dst, eid)).collect();
            backward.batch_mark_deleted(&bwd_batch);
        }

        // Properties and counters (no atomicity requirement with adjacency)
        for &(edge_id, _, _, type_id) in &deleted {
            self.edge_properties.remove_all(edge_id, epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    /// (Tiered storage version)
    ///
    /// Uses a `HashSet` to dedup self-loops (where src == dst) and holds
    /// the edge_versions write lock for the entire batch to prevent
    /// concurrent readers from observing a partially detached node.
    #[cfg(feature = "tiered-storage")]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let epoch = self.current_epoch();

        // Collect all edge IDs into a set to dedup self-loops
        let mut edge_ids: FxHashSet<EdgeId> = FxHashSet::default();

        for (_, edge_id) in self.forward_adj.edges_from(node_id) {
            edge_ids.insert(edge_id);
        }

        if let Some(ref backward) = self.backward_adj {
            for (_, edge_id) in backward.edges_from(node_id) {
                edge_ids.insert(edge_id);
            }
        } else {
            // No backward adjacency: scan all edges for incoming
            let versions = self.edge_versions.read();
            for (id, index) in versions.iter() {
                if let Some(vref) = index.visible_at(epoch)
                    && let Some(r) = self.read_edge_record(&vref)
                    && !r.is_deleted()
                    && r.dst == node_id
                {
                    edge_ids.insert(*id);
                }
            }
        }

        if edge_ids.is_empty() {
            return;
        }

        // Hold the write lock for the entire batch: mark all edges as deleted
        // atomically so concurrent readers never see a partially detached node.
        let deleted: Vec<(EdgeId, NodeId, NodeId, u32)>;
        {
            let mut versions = self.edge_versions.write();
            deleted = edge_ids
                .iter()
                .filter_map(|&edge_id| {
                    let index = versions.get_mut(&edge_id)?;
                    let vref = index.visible_at(epoch)?;
                    let record = self.read_edge_record(&vref)?;
                    if record.is_deleted() {
                        return None;
                    }
                    let src = record.src;
                    let dst = record.dst;
                    let type_id = record.type_id;
                    index.mark_deleted(epoch, TransactionId::SYSTEM);
                    Some((edge_id, src, dst, type_id))
                })
                .collect();
        }

        // Batch-update adjacency under a single write lock so concurrent
        // readers of neighbors()/edges_from() never see a partially detached node.
        let fwd_batch: Vec<(NodeId, EdgeId)> =
            deleted.iter().map(|&(eid, src, _, _)| (src, eid)).collect();
        self.forward_adj.batch_mark_deleted(&fwd_batch);

        if let Some(ref backward) = self.backward_adj {
            let bwd_batch: Vec<(NodeId, EdgeId)> =
                deleted.iter().map(|&(eid, _, dst, _)| (dst, eid)).collect();
            backward.batch_mark_deleted(&bwd_batch);
        }

        // Properties and counters (no atomicity requirement with adjacency)
        for &(edge_id, _, _, type_id) in &deleted {
            self.edge_properties.remove_all(edge_id, epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);
        }
    }

    // --- Visibility checks (no label/property loading) ---

    /// Checks if a node is visible at the given epoch.
    ///
    /// Only checks the version chain, skips label and property loading.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.work_counters.record_node_visibility_check();
        let nodes = self.nodes.read();
        nodes
            .get(&id)
            .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
    }

    /// Checks if a node is visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.work_counters.record_node_visibility_check();
        let versions = self.node_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_at(epoch).is_some_and(|vref| {
                self.read_node_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }

    /// Checks if a node is visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let visible = self.nodes.read().get(&id).is_some_and(|chain| {
            chain
                .visible_to(epoch, transaction_id)
                .is_some_and(|r| !r.is_deleted())
        });
        if visible {
            self.record_read_node(transaction_id, id);
        }
        visible
    }

    /// Checks if a node is visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let visible = self.node_versions.read().get(&id).is_some_and(|index| {
            index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                self.read_node_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        });
        if visible {
            self.record_read_node(transaction_id, id);
        }
        visible
    }

    /// Filters node IDs to only those visible at the given epoch.
    ///
    /// Holds a single lock for the entire batch instead of per-node locking.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        ids.iter()
            .copied()
            .filter(|id| {
                nodes
                    .get(id)
                    .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
            })
            .collect()
    }

    /// Filters node IDs to only those visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let versions = self.node_versions.read();
        ids.iter()
            .copied()
            .filter(|id| {
                versions.get(id).is_some_and(|index| {
                    index.visible_at(epoch).is_some_and(|vref| {
                        self.read_node_record(&vref)
                            .is_some_and(|r| !r.is_deleted())
                    })
                })
            })
            .collect()
    }

    /// Filters node IDs to only those visible to a specific transaction.
    ///
    /// Holds a single lock for the entire batch instead of per-node locking.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        let visible: Vec<NodeId> = ids
            .iter()
            .copied()
            .filter(|id| {
                nodes.get(id).is_some_and(|chain| {
                    chain
                        .visible_to(epoch, transaction_id)
                        .is_some_and(|r| !r.is_deleted())
                })
            })
            .collect();
        drop(nodes);
        for &id in &visible {
            self.record_read_node(transaction_id, id);
        }
        visible
    }

    /// Filters node IDs to only those visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let versions = self.node_versions.read();
        let visible: Vec<NodeId> = ids
            .iter()
            .copied()
            .filter(|id| {
                versions.get(id).is_some_and(|index| {
                    index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                        self.read_node_record(&vref)
                            .is_some_and(|r| !r.is_deleted())
                    })
                })
            })
            .collect();
        drop(versions);
        for &id in &visible {
            self.record_read_node(transaction_id, id);
        }
        visible
    }

    /// Label-scan variant of [`filter_visible_node_ids_versioned`]: MVCC-filters
    /// `ids` and records each visible node under the `label_id` predicate bucket
    /// (GE3 escalation path). Reads are tagged with `label_id` so the manager's
    /// `record_read_in_label` machinery can promote fine `Node` entries to the
    /// coarse `EntityId::Label(L)` key once the escalation threshold is exceeded.
    ///
    /// This is the canonical recording chokepoint for `MATCH (n:L)` scans. All
    /// other node read chokepoints use [`record_read_node`](Self::record_read_node)
    /// (fine, no predicate) — only label-scan-driven reads go through here.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn filter_visible_node_ids_in_label_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
        label_id: LabelId,
    ) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        let visible: Vec<NodeId> = ids
            .iter()
            .copied()
            .filter(|id| {
                nodes.get(id).is_some_and(|chain| {
                    chain
                        .visible_to(epoch, transaction_id)
                        .is_some_and(|r| !r.is_deleted())
                })
            })
            .collect();
        drop(nodes);
        for &id in &visible {
            self.record_read_node_in_label(transaction_id, id, label_id);
        }
        visible
    }

    /// Label-scan variant of [`filter_visible_node_ids_versioned`]: tiered-storage
    /// edition. Symmetric with the non-tiered variant above.
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn filter_visible_node_ids_in_label_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
        label_id: LabelId,
    ) -> Vec<NodeId> {
        let versions = self.node_versions.read();
        let visible: Vec<NodeId> = ids
            .iter()
            .copied()
            .filter(|id| {
                versions.get(id).is_some_and(|index| {
                    index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                        self.read_node_record(&vref)
                            .is_some_and(|r| !r.is_deleted())
                    })
                })
            })
            .collect();
        drop(versions);
        for &id in &visible {
            self.record_read_node_in_label(transaction_id, id, label_id);
        }
        visible
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn node_count(&self) -> usize {
        let _read = self.pin_read();
        let epoch = self.current_epoch();
        self.nodes
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn node_count(&self) -> usize {
        let _read = self.pin_read();
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_node_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count()
    }

    /// Returns all node IDs in the store.
    ///
    /// This returns a snapshot of current node IDs. The returned vector
    /// excludes deleted nodes. Results are sorted by NodeId for deterministic
    /// iteration order.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let _read = self.pin_read();
        let epoch = self.current_epoch();
        let mut ids: Vec<NodeId> = self
            .nodes
            .read()
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Returns all node IDs in the store.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let _read = self.pin_read();
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        let mut ids: Vec<NodeId> = versions
            .iter()
            .filter_map(|(id, index)| {
                index.visible_at(epoch).and_then(|vref| {
                    self.read_node_record(&vref)
                        .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
                })
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Returns all node IDs including uncommitted/PENDING versions.
    ///
    /// Unlike `node_ids()` which pre-filters by current epoch, this returns
    /// every node that has a version chain entry. Used by scan operators that
    /// perform their own MVCC visibility filtering with transaction context.
    #[must_use]
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        let ids = self.all_node_ids_uncounted();
        self.work_counters.record_full_scan(ids.len());
        ids
    }

    /// `all_node_ids` without the work accounting.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn all_node_ids_uncounted(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.nodes.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns all node IDs including uncommitted/PENDING versions.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn all_node_ids_uncounted(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.node_versions.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns whether any structural lifetime exists for `id`, including a
    /// PENDING or committed-deleted lifetime that is not currently visible.
    ///
    /// Transport mutation admission uses this to distinguish a destination
    /// omitted by subgraph extraction from an endpoint concurrently created or
    /// deleted by another transaction. It is intentionally stronger than a
    /// current-value lookup.
    #[doc(hidden)]
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn contains_node_identity(&self, id: NodeId) -> bool {
        self.nodes.read().contains_key(&id)
    }

    /// Tiered-storage counterpart of [`Self::contains_node_identity`].
    #[doc(hidden)]
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn contains_node_identity(&self, id: NodeId) -> bool {
        self.node_versions.read().contains_key(&id)
    }

    /// Finalizes PENDING deletes for a committed transaction: stamps each
    /// deleted version's `deleted_epoch` PENDING→`commit_epoch` and applies the
    /// deferred label-index/property removal now that the delete is committed.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn finalize_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        node_ids: &[NodeId],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if node_ids.is_empty() {
            return;
        }
        {
            let mut nodes = self.nodes.write();
            for &id in node_ids {
                if let Some(chain) = nodes.get_mut(&id) {
                    chain.finalize_deleted_epochs(transaction_id, commit_epoch);
                }
            }
        }
        for &id in node_ids {
            self.remove_node_from_derived_indexes(id);
        }
        // Apply current label-index removal now that the delete is committed,
        // retaining each node's VersionLog for historical reconstruction.
        for &id in node_ids {
            self.remove_node_from_current_label_index(id);
        }

        // Remove properties now that the delete is committed.
        for &id in node_ids {
            self.node_properties.remove_all(id, commit_epoch);

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Finalizes PENDING deletes for a committed transaction.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn finalize_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        node_ids: &[NodeId],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if node_ids.is_empty() {
            return;
        }
        {
            let mut versions = self.node_versions.write();
            for &id in node_ids {
                if let Some(index) = versions.get_mut(&id) {
                    index.finalize_deleted_epochs(transaction_id, commit_epoch);
                }
            }
        }
        for &id in node_ids {
            self.remove_node_from_derived_indexes(id);
        }
        // Apply current label-index removal now that the delete is committed,
        // retaining each node's VersionLog for historical reconstruction.
        for &id in node_ids {
            self.remove_node_from_current_label_index(id);
        }

        // Remove properties now that the delete is committed.
        for &id in node_ids {
            self.node_properties.remove_all(id, commit_epoch);

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Rolls back PENDING deletes for a transaction: clears the PENDING
    /// `deleted_epoch` on each node's version chain so the node is visible again.
    /// Labels and properties were never removed (deferred path), so no restoration
    /// is needed — just unmark the version chain.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn rollback_pending_deletes(&self, transaction_id: TransactionId, node_ids: &[NodeId]) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if node_ids.is_empty() {
            return;
        }
        let mut nodes = self.nodes.write();
        for &id in node_ids {
            if let Some(chain) = nodes.get_mut(&id) {
                chain.unmark_deleted_by(transaction_id);
            }
        }
    }

    /// Rolls back PENDING deletes for a transaction.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn rollback_pending_deletes(&self, transaction_id: TransactionId, node_ids: &[NodeId]) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if node_ids.is_empty() {
            return;
        }
        let mut versions = self.node_versions.write();
        for &id in node_ids {
            if let Some(index) = versions.get_mut(&id) {
                index.unmark_deleted_by(transaction_id);
            }
        }
    }

    /// Takes (removes and returns) the pending delete list for a transaction.
    #[doc(hidden)]
    pub fn take_pending_deletes(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        let Some(_mutation) = self.pin_mutation() else {
            return Vec::new();
        };
        self.pending_tx_deletes
            .write()
            .remove(&transaction_id)
            .unwrap_or_default()
    }
}
