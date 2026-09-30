use super::{
    EdgeIdentityReservation, LpgStore, TransportEdgeMutationGrant, TransportEdgeReceipt,
    TransportEdgeState,
};
use crate::graph::lpg::{Edge, EdgeRecord};
use arcstr::ArcStr;
use grafeo_common::types::{
    EdgeId, EdgeTypeId, EpochId, NodeId, PropertyKey, TransactionId, Value,
};
use std::sync::atomic::Ordering;

#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{HotVersionRef, VersionIndex, VersionRef};

/// Owns every value returned by preparation until the caller's broader cut
/// can drain. A committed-then-rolled-back publication is `Rejected`, never
/// an invented pre-publication value.
pub(crate) enum PreparedPurgeOutcome<P, R> {
    Rejected,
    Abandoned(P),
    Published(R),
    UnwoundBeforePublication {
        prepared: P,
        payload: Box<dyn std::any::Any + Send>,
    },
}

impl LpgStore {
    /// Builds an `Edge` from a record, resolving the type name and loading properties.
    fn build_edge(&self, id: EdgeId, record: &EdgeRecord) -> Option<Edge> {
        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };
        let mut edge = Edge::new(id, record.src, record.dst, edge_type);
        edge.properties = self.edge_properties.get_all(id).into_iter().collect();
        Some(edge)
    }

    /// Builds an `Edge` with properties as they were at a specific epoch.
    fn build_edge_at(&self, id: EdgeId, record: &EdgeRecord, epoch: EpochId) -> Option<Edge> {
        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };
        let mut edge = Edge::new(id, record.src, record.dst, edge_type);
        edge.properties = self
            .edge_properties
            .get_all_at(id, epoch)
            .into_iter()
            .collect();
        Some(edge)
    }

    /// Creates a new edge.
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.create_edge_versioned(
            src,
            dst,
            edge_type,
            self.current_epoch(),
            TransactionId::SYSTEM,
        )
    }

    /// Creates a new edge within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let Some(_mutation) = self.pin_mutation() else {
            return EdgeId::INVALID;
        };
        let authority = self.transport_extract_authority.read();
        let Some(identity_reservation) = EdgeIdentityReservation::generated(self, 1) else {
            return EdgeId::INVALID;
        };
        let id = identity_reservation.id();
        {
            let nodes = self.nodes.read();
            if !nodes.contains_key(&src) || !nodes.contains_key(&dst) {
                return EdgeId::INVALID;
            }
        }
        #[cfg(test)]
        self.pause_edge_publication_for_test();
        let Ok(prepared_type) = self.prepare_edge_type(edge_type) else {
            return EdgeId::INVALID;
        };
        let type_id = prepared_type.id();
        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };
        let chain = VersionChain::with_initial(record, version_epoch, transaction_id);
        let nodes = self.nodes.read();
        if !nodes.contains_key(&src) || !nodes.contains_key(&dst) {
            return EdgeId::INVALID;
        }
        let mut edges = self.edges.write();
        if edges.try_reserve(1).is_err() {
            return EdgeId::INVALID;
        }
        let committed_type_id = prepared_type.commit(self);
        debug_assert_eq!(committed_type_id, type_id);
        // Ordinary publication is the last-line stale-receipt fence. Revocation
        // is deferred until every fallible preparation and final endpoint
        // recheck has succeeded.
        authority.revoke_id(id);
        debug_assert!(!edges.contains_key(&id), "reserved edge identity is vacant");
        edges.insert(id, chain);
        drop(edges);
        self.record_pending_edge(transaction_id, id);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);

        // Phantom coarse write: record RelType(T) so a concurrent escalated
        // RelType(T) reader forms an rw-antidependency with this CREATE edge.
        self.record_coarse_edge_write(transaction_id, id, EdgeTypeId::from(type_id));

        drop(nodes);
        identity_reservation.commit();
        id
    }

    /// Creates a new edge within a transaction context.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let Some(_mutation) = self.pin_mutation() else {
            return EdgeId::INVALID;
        };
        let authority = self.transport_extract_authority.read();
        let Some(identity_reservation) = EdgeIdentityReservation::generated(self, 1) else {
            return EdgeId::INVALID;
        };
        let id = identity_reservation.id();
        {
            let nodes = self.node_versions.read();
            if !nodes.contains_key(&src) || !nodes.contains_key(&dst) {
                return EdgeId::INVALID;
            }
        }
        #[cfg(test)]
        self.pause_edge_publication_for_test();
        let Ok(prepared_type) = self.prepare_edge_type(edge_type) else {
            return EdgeId::INVALID;
        };
        let type_id = prepared_type.id();

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let Ok(arena) = self.arena_allocator.arena_or_create(epoch) else {
            return EdgeId::INVALID;
        };
        let Ok((offset, _stored)) = arena.alloc_value_with_offset(record) else {
            return EdgeId::INVALID;
        };
        // The mutation pin keeps the arena alive through publication. Release
        // its registry guard before taking version locks: readers acquire them
        // in the opposite order, and a queued arena writer blocks new readers.
        drop(arena);

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };

        // Create HotVersionRef pointing to arena data
        let hot_ref = HotVersionRef::new(version_epoch, epoch, offset, transaction_id);

        let nodes = self.node_versions.read();
        if !nodes.contains_key(&src) || !nodes.contains_key(&dst) {
            return EdgeId::INVALID;
        }
        let mut versions = self.edge_versions.write();
        if versions.try_reserve(1).is_err() {
            return EdgeId::INVALID;
        }
        let committed_type_id = prepared_type.commit(self);
        debug_assert_eq!(committed_type_id, type_id);
        // Publish the fully allocated version into the out-of-map reservation.
        authority.revoke_id(id);
        debug_assert!(
            !versions.contains_key(&id),
            "reserved edge identity is vacant"
        );
        versions.insert(id, VersionIndex::with_initial(hot_ref));
        drop(versions);
        self.record_pending_edge(transaction_id, id);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);

        // Phantom coarse write: record RelType(T) for phantom detection.
        self.record_coarse_edge_write(transaction_id, id, EdgeTypeId::from(type_id));

        drop(nodes);
        identity_reservation.commit();
        id
    }

    /// Creates a new edge with properties.
    pub fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> EdgeId {
        let Some(_mutation) = self.pin_mutation() else {
            return EdgeId::INVALID;
        };
        let id = self.create_edge(src, dst, edge_type);
        if !id.is_valid() {
            return EdgeId::INVALID;
        }

        for (key, value) in properties {
            self.edge_properties
                .set(id, key.into(), value.into(), self.current_epoch());
        }

        id
    }

    /// Gets an edge by ID (latest visible version).
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let _read = self.pin_read();
        self.get_edge_at_epoch(id, self.current_epoch())
    }

    /// Gets an edge by ID at a specific epoch.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let _read = self.pin_read();
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_at(epoch)?;
        if record.is_deleted() {
            return None;
        }
        let record = *record;
        drop(edges);

        if epoch >= self.current_epoch() {
            self.build_edge(id, &record)
        } else {
            self.build_edge_at(id, &record, epoch)
        }
    }

    /// Gets an edge by ID at a specific epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let _read = self.pin_read();
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_at(epoch)?;
        let record = self.read_edge_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);

        if epoch >= self.current_epoch() {
            self.build_edge(id, &record)
        } else {
            self.build_edge_at(id, &record, epoch)
        }
    }

    /// Gets an edge visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        let _read = self.pin_read();
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_to(epoch, transaction_id)?;
        if record.is_deleted() {
            return None;
        }
        let record = *record;
        drop(edges);
        // Materialize values AS OF `epoch` (snapshot isolation), not current.
        let mut edge = self.build_edge_at(id, &record, epoch)?;
        edge.properties = self
            .read_edge_properties_visible(id, epoch, Some(transaction_id))
            .into_iter()
            .collect();
        self.record_read_edge_in_rel_type(transaction_id, id, EdgeTypeId::from(record.type_id));
        Some(edge)
    }

    /// Gets an edge visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        let _read = self.pin_read();
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_to(epoch, transaction_id)?;
        let record = self.read_edge_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);
        // Materialize values AS OF `epoch` (snapshot isolation), not current.
        let mut edge = self.build_edge_at(id, &record, epoch)?;
        edge.properties = self
            .read_edge_properties_visible(id, epoch, Some(transaction_id))
            .into_iter()
            .collect();
        self.record_read_edge_in_rel_type(transaction_id, id, EdgeTypeId::from(record.type_id));
        Some(edge)
    }

    /// Reads an EdgeRecord from arena using a VersionRef.
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub(super) fn read_edge_record(&self, version_ref: &VersionRef) -> Option<EdgeRecord> {
        match version_ref {
            VersionRef::Hot(hot_ref) => {
                let arena = self
                    .arena_allocator
                    .arena(hot_ref.arena_epoch)
                    .expect("arena epoch must exist for hot version ref");
                // SAFETY: The offset was returned by alloc_value_with_offset for an EdgeRecord
                let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                Some(*record)
            }
            VersionRef::Cold(cold_ref) => {
                // Read from compressed epoch store
                self.epoch_store
                    .get_edge(cold_ref.epoch, cold_ref.block_offset, cold_ref.length)
            }
            _ => None,
        }
    }

    /// Every edge id that still has a version chain, including deleted ones.
    ///
    /// Used by temporal merge to retain committed-deleted edges as closed rows.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn all_known_edge_ids(&self) -> Vec<EdgeId> {
        self.edges.read().keys().copied().collect()
    }

    /// Every edge id that still has a version index, including deleted ones.
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn all_known_edge_ids(&self) -> Vec<EdgeId> {
        self.edge_versions.read().keys().copied().collect()
    }

    /// Returns all versions of an edge with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Edge)`.
    /// Without `temporal`: properties reflect the current state.
    /// With `temporal`: each version has correct historical properties.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let _read = self.pin_read();
        let edges = self.edges.read();
        let Some(chain) = edges.get(&id) else {
            return Vec::new();
        };

        let id_to_type = self.id_to_edge_type.read();

        chain
            .history()
            .filter_map(|(info, record)| {
                let edge_type = id_to_type.get(record.type_id as usize)?.clone();
                let mut edge = Edge::new(id, record.src, record.dst, edge_type);
                edge.properties = self
                    .edge_properties
                    .get_all_at(id, info.created_epoch)
                    .into_iter()
                    .collect();
                Some((info.created_epoch, info.deleted_epoch, edge))
            })
            .collect()
    }

    /// Returns all versions of an edge with their creation/deletion epochs, newest first.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let _read = self.pin_read();
        let versions = self.edge_versions.read();
        let Some(index) = versions.get(&id) else {
            return Vec::new();
        };

        let id_to_type = self.id_to_edge_type.read();
        let properties: grafeo_common::types::PropertyMap =
            self.edge_properties.get_all(id).into_iter().collect();

        index
            .version_history()
            .into_iter()
            .filter_map(|(created, deleted, vref)| {
                let record = self.read_edge_record(&vref)?;
                let edge_type = id_to_type.get(record.type_id as usize)?.clone();
                let mut edge = Edge::new(id, record.src, record.dst, edge_type);
                edge.properties.clone_from(&properties);
                Some((created, deleted, edge))
            })
            .collect()
    }

    /// Returns whether an open edge is covered by this exact move-only
    /// subgraph-transport receipt.
    ///
    /// Recovery and ordinary graph mutations never install this receipt. The
    /// engine uses this read-only preflight before it frames the logical delete
    /// through a Session transaction.
    #[doc(hidden)]
    #[must_use]
    pub fn is_transport_extract_edge(&self, receipt: &TransportEdgeReceipt) -> bool {
        self.is_transport_extract_edge_unchanged_since(receipt, self.current_epoch())
    }

    /// Authenticates receipt provenance against this overlay's current store
    /// incarnation without requiring a resident structural edge.
    ///
    /// Compact/base-only classification uses this narrow predicate instead of
    /// reading the private authority or nonce map. The recursive shared
    /// transition pin makes the result one clear/replacement cut and is safe
    /// when called from an already-pinned Layered operation.
    #[cfg(feature = "compact-store")]
    pub(crate) fn transport_receipts_belong_to_current_incarnation(
        &self,
        receipts: &[&TransportEdgeReceipt],
    ) -> bool {
        let _transition = self.mutation_scope_gate.read_recursive();
        let authority = self.transport_extract_authority.read();
        super::TransportExtractAuthority::contains_all(&authority, receipts.iter().copied())
    }

    /// Single-receipt form of
    /// [`Self::transport_receipts_belong_to_current_incarnation`].
    #[cfg(feature = "compact-store")]
    pub(crate) fn transport_receipt_belongs_to_current_incarnation(
        &self,
        receipt: &TransportEdgeReceipt,
    ) -> bool {
        self.transport_receipts_belong_to_current_incarnation(&[receipt])
    }

    /// Authenticates a transaction-local transport mutation grant against this
    /// overlay incarnation even when its carried edge resides only in the
    /// compact base. Structural and endpoint admission remain separate commit
    /// checks; this predicate exposes no nonce or authority handle.
    #[cfg(feature = "compact-store")]
    pub(crate) fn transport_grant_belongs_to_current_incarnation(
        &self,
        grant: &TransportEdgeMutationGrant,
    ) -> bool {
        let _transition = self.mutation_scope_gate.read_recursive();
        let authority = self.transport_extract_authority.read();
        grant.belongs_to(&authority)
    }

    /// Arms the crate-private hostile boundary used by compact rollback tests.
    #[cfg(test)]
    pub(crate) fn panic_after_transport_publication_once_for_test(&self) {
        self.transport_post_publish_panic
            .store(true, Ordering::SeqCst);
    }

    #[cfg(all(test, feature = "compact-store"))]
    pub(crate) fn generation_purge_late_once_for_test(&self, unwind: bool) {
        self.generation_purge_late_action
            .store(if unwind { 2 } else { 1 }, Ordering::SeqCst);
    }

    #[cfg(all(test, feature = "compact-store"))]
    pub(crate) fn generation_purge_late_hits_for_test(&self) -> usize {
        self.generation_purge_late_hits.load(Ordering::SeqCst)
    }

    /// Nonblocking observations only: Drop probes never wait or assert while
    /// an unrelated unwind is already in flight.
    #[cfg(all(test, feature = "compact-store"))]
    pub(crate) fn generation_guards_drained_for_test(&self) -> bool {
        let mutation = self.mutation_scope_gate.try_write().is_some();
        let authority = self.transport_extract_authority.try_write().is_some();
        let reservation = self
            .edge_identity_reservations
            .identities
            .try_lock()
            .is_some();
        #[cfg(not(feature = "tiered-storage"))]
        let entities = self.edges.try_write().is_some();
        #[cfg(feature = "tiered-storage")]
        let entities = self.edge_versions.try_write().is_some();
        let creates = self.pending_tx_creates.try_write().is_some();
        let deletes = self.pending_tx_edge_deletes.try_write().is_some();
        mutation && authority && reservation && entities && creates && deletes
    }

    /// Returns whether this receipt still covers its original history with an open final
    /// structural identity.
    ///
    /// The engine uses this at a quiescent cut to rebase transport metadata
    /// across unrelated commits. Edge properties may change: the receipt proves
    /// the edge was born as transport state, and orphan cleanup owns its whole
    /// payload. Delete/recreate, store replacement, or a foreign receipt fails.
    #[doc(hidden)]
    #[must_use]
    pub fn is_transport_extract_edge_unchanged_since(
        &self,
        receipt: &TransportEdgeReceipt,
        since: EpochId,
    ) -> bool {
        let authority = self.transport_extract_authority.read();
        let reservations = self.edge_identity_reservations.identities.lock();
        if reservations.contains(&receipt.id) || !receipt.belongs_to(&authority) {
            return false;
        }
        let expected_type_id = self.pinned_transport_edge_type_id(receipt.edge_type());
        let Some(expected_type_id) = expected_type_id else {
            return false;
        };
        self.transport_extract_identity_state(
            &receipt.lifetimes,
            receipt.id,
            receipt.src,
            receipt.dst,
            expected_type_id,
            since,
        ) == TransportEdgeState::Open
    }

    /// Mints a transaction-local, property-mutation-only grant for one exact
    /// open transport identity.
    ///
    /// The caller must hold the supplied database's write authority. A foreign
    /// authority cannot issue a grant for a sealed store, and an unsealed store
    /// still requires the supplied scope to be actively held. The move-only
    /// receipt remains the sole source of transport provenance.
    #[doc(hidden)]
    #[must_use]
    pub fn grant_transport_edge_mutation(
        &self,
        receipt: &TransportEdgeReceipt,
        write_authority: &crate::graph::write_permit::WriteAuthority,
    ) -> Option<TransportEdgeMutationGrant> {
        if !self.accepts_held_write_authority(write_authority)
            || !self.is_transport_extract_edge(receipt)
        {
            return None;
        }
        Some(TransportEdgeMutationGrant::from_receipt(receipt))
    }

    /// Revalidates a staged property-mutation grant at the publication cut for
    /// its one allowed missing endpoint: the transport destination omitted by
    /// extraction and never structurally present in this store.
    ///
    /// A deleted or PENDING destination still has structural identity and is
    /// rejected, as is a missing source. The incarnation authority and edge-ID
    /// registry remain pinned across nonce, endpoint, and row validation, so a
    /// physical remove/recreate cannot splice two lifetimes into one decision.
    #[doc(hidden)]
    #[must_use]
    pub fn accepts_transport_edge_missing_destination(
        &self,
        grant: &TransportEdgeMutationGrant,
        missing_endpoint: NodeId,
        write_authority: &crate::graph::write_permit::WriteAuthority,
    ) -> bool {
        if missing_endpoint != grant.dst || !self.accepts_held_write_authority(write_authority) {
            return false;
        }
        let authority = self.transport_extract_authority.read();
        let reservations = self.edge_identity_reservations.identities.lock();
        if reservations.contains(&grant.id) || !grant.belongs_to(&authority) {
            return false;
        }
        let expected_type_id = self.pinned_transport_edge_type_id(grant.edge_type.as_str());
        let Some(expected_type_id) = expected_type_id else {
            return false;
        };
        #[cfg(not(feature = "tiered-storage"))]
        let nodes = self.nodes.read();
        #[cfg(feature = "tiered-storage")]
        let nodes = self.node_versions.read();
        if nodes.contains_key(&missing_endpoint) || !nodes.contains_key(&grant.src) {
            return false;
        }
        self.transport_extract_identity_state(
            &grant.lifetimes,
            grant.id,
            grant.src,
            grant.dst,
            expected_type_id,
            self.current_epoch(),
        ) == TransportEdgeState::Open
    }

    fn transport_extract_identity_state(
        &self,
        lifetimes: &[(EpochId, Option<EpochId>)],
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        expected_type_id: u32,
        since: EpochId,
    ) -> TransportEdgeState {
        #[cfg(not(feature = "tiered-storage"))]
        {
            let edges = self.edges.read();
            let Some(chain) = edges.get(&id) else {
                return TransportEdgeState::Invalid;
            };
            super::transport_lifetime_state(
                lifetimes,
                chain.history().map(|(info, record)| {
                    if info.created_by == TransactionId::SYSTEM
                        && record.src == src
                        && record.dst == dst
                        && record.type_id == expected_type_id
                    {
                        (info.created_epoch, info.deleted_epoch)
                    } else {
                        (EpochId::PENDING, None)
                    }
                }),
                since,
                since,
            )
        }
        #[cfg(feature = "tiered-storage")]
        {
            let edges = self.edge_versions.read();
            let Some(index) = edges.get(&id) else {
                return TransportEdgeState::Invalid;
            };
            super::transport_lifetime_state(
                lifetimes,
                index
                    .version_history()
                    .iter()
                    .map(|(created, deleted, version)| {
                        if version.created_by() == TransactionId::SYSTEM
                            && self.read_edge_record(version).is_some_and(|record| {
                                record.src == src
                                    && record.dst == dst
                                    && record.type_id == expected_type_id
                            })
                        {
                            (*created, *deleted)
                        } else {
                            (EpochId::PENDING, None)
                        }
                    }),
                since,
                since,
            )
        }
    }

    /// Resolves a receipt-carried type name without carrying a catalog guard
    /// into entity locks. Callers hold the incarnation authority read side:
    /// type IDs are append-only until `clear`, and `clear` needs its write side,
    /// so the returned numeric identity stays stable for the caller's cut.
    fn pinned_transport_edge_type_id(&self, edge_type: &str) -> Option<u32> {
        self.edge_type_to_id.read().get(edge_type).copied()
    }

    /// Returns whether this receipt covers its original history with a committed closed
    /// final transport lifetime unchanged since `since`.
    ///
    /// This is the retry counterpart of
    /// [`Self::is_transport_extract_edge_unchanged_since`]. It lets the engine
    /// carry an explicit physical-purge debt across unrelated commits without
    /// ever treating an ordinary changed/deleted identity as purgeable.
    #[doc(hidden)]
    #[must_use]
    pub fn is_transport_extract_closed_edge_unchanged_since(
        &self,
        receipt: &TransportEdgeReceipt,
        since: EpochId,
    ) -> bool {
        let authority = self.transport_extract_authority.read();
        let reservations = self.edge_identity_reservations.identities.lock();
        if reservations.contains(&receipt.id)
            || !receipt.belongs_to(&authority)
            || !self
                .forward_adj
                .is_exact_deleted_edge(receipt.src, receipt.id)
            || self
                .backward_adj
                .as_ref()
                .is_some_and(|backward| !backward.is_exact_deleted_edge(receipt.dst, receipt.id))
        {
            return false;
        }
        let expected_type_id = self.pinned_transport_edge_type_id(receipt.edge_type());
        let Some(expected_type_id) = expected_type_id else {
            return false;
        };
        self.transport_extract_identity_state(
            &receipt.lifetimes,
            receipt.id,
            receipt.src,
            receipt.dst,
            expected_type_id,
            since,
        ) == TransportEdgeState::Closed
    }

    /// Classifies a receipt batch under one authority/history/adjacency cut.
    ///
    /// Structural maps are looked up once per receipt and every affected
    /// adjacency list is scanned once for all closed receipts, avoiding the
    /// quadratic high-degree-star behavior of repeated point qualification.
    #[doc(hidden)]
    #[must_use]
    pub fn classify_transport_extract_edges(
        &self,
        receipts: &[&TransportEdgeReceipt],
        open_since: EpochId,
        closed_since: EpochId,
    ) -> Vec<TransportEdgeState> {
        let authority = self.transport_extract_authority.read();
        let reservations = self.edge_identity_reservations.identities.lock();
        let mut states = vec![TransportEdgeState::Invalid; receipts.len()];
        let mut counts: grafeo_common::utils::hash::FxHashMap<EdgeId, usize> =
            grafeo_common::utils::hash::FxHashMap::default();
        for receipt in receipts {
            *counts.entry(receipt.id).or_default() += 1;
        }
        // The authority guard pins this append-only type-ID incarnation. The
        // catalog guard ends with this block, before structural/adjacency locks.
        let expected_type_ids: Vec<Option<u32>> = {
            let edge_types = self.edge_type_to_id.read();
            receipts
                .iter()
                .map(|receipt| edge_types.get(receipt.edge_type()).copied())
                .collect()
        };

        #[cfg(not(feature = "tiered-storage"))]
        {
            let edges = self.edges.read();
            for (position, receipt) in receipts.iter().enumerate() {
                if counts.get(&receipt.id) != Some(&1)
                    || reservations.contains(&receipt.id)
                    || !receipt.belongs_to(&authority)
                {
                    continue;
                }
                let Some(chain) = edges.get(&receipt.id) else {
                    continue;
                };
                states[position] = receipt.history_state(
                    chain.history().map(|(info, record)| {
                        if info.created_by == TransactionId::SYSTEM
                            && record.src == receipt.src
                            && record.dst == receipt.dst
                            && expected_type_ids[position] == Some(record.type_id)
                        {
                            (info.created_epoch, info.deleted_epoch)
                        } else {
                            (EpochId::PENDING, None)
                        }
                    }),
                    open_since,
                    closed_since,
                );
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            let edges = self.edge_versions.read();
            for (position, receipt) in receipts.iter().enumerate() {
                if counts.get(&receipt.id) != Some(&1)
                    || reservations.contains(&receipt.id)
                    || !receipt.belongs_to(&authority)
                {
                    continue;
                }
                let Some(index) = edges.get(&receipt.id) else {
                    continue;
                };
                states[position] = receipt.history_state(
                    index
                        .version_history()
                        .iter()
                        .map(|(created, deleted, version)| {
                            if version.created_by() == TransactionId::SYSTEM
                                && self.read_edge_record(version).is_some_and(|record| {
                                    record.src == receipt.src
                                        && record.dst == receipt.dst
                                        && expected_type_ids[position] == Some(record.type_id)
                                })
                            {
                                (*created, *deleted)
                            } else {
                                (EpochId::PENDING, None)
                            }
                        }),
                    open_since,
                    closed_since,
                );
            }
        }
        let forward_candidates: Vec<_> = receipts
            .iter()
            .zip(&states)
            .filter_map(|(receipt, state)| {
                (*state == TransportEdgeState::Closed).then_some((receipt.src, receipt.id))
            })
            .collect();
        let exact_forward = self.forward_adj.exact_deleted_edges(&forward_candidates);
        let exact_backward = self.backward_adj.as_ref().map(|backward| {
            let candidates: Vec<_> = receipts
                .iter()
                .zip(&states)
                .filter_map(|(receipt, state)| {
                    (*state == TransportEdgeState::Closed).then_some((receipt.dst, receipt.id))
                })
                .collect();
            backward.exact_deleted_edges(&candidates)
        });
        for (receipt, state) in receipts.iter().zip(&mut states) {
            if *state == TransportEdgeState::Closed
                && (!exact_forward.contains(&receipt.id)
                    || exact_backward
                        .as_ref()
                        .is_some_and(|backward| !backward.contains(&receipt.id)))
            {
                *state = TransportEdgeState::Invalid;
            }
        }
        states
    }

    /// Physically discards closed resident edge histories from an ephemeral
    /// transport extract, as authorized by store-local move-only provenance.
    ///
    /// This convenience route has no compact/base-only identities: every
    /// receipt must name one exact resident closed SYSTEM lifetime. Receipt
    /// provenance may have been minted by transport creation or exact transport
    /// recovery. Allocator high-water marks remain monotonic after removal.
    #[doc(hidden)]
    pub fn purge_transport_extract_edges(&self, receipts: &[&TransportEdgeReceipt]) -> bool {
        match self.purge_transport_extract_edges_after_prepare(receipts, || {
            Ok::<(), std::convert::Infallible>(())
        }) {
            Ok(purged) => purged,
            Err(never) => match never {},
        }
    }

    /// Performs the read-only structural half of resident purge qualification.
    /// No guard escapes this call, so an external representation preparer can
    /// safely read the overlay after it returns.
    #[cfg(not(feature = "tiered-storage"))]
    fn resident_transport_histories_match(
        &self,
        identities: &[(EdgeId, NodeId, NodeId, u32, &TransportEdgeReceipt)],
        boundary: EpochId,
    ) -> bool {
        let edges = self.edges.read();
        identities
            .iter()
            .all(|&(id, src, dst, expected_type_id, receipt)| {
                let Some(chain) = edges.get(&id) else {
                    return false;
                };
                receipt.history_state(
                    chain.history().map(|(info, record)| {
                        if info.created_by == TransactionId::SYSTEM
                            && record.src == src
                            && record.dst == dst
                            && record.type_id == expected_type_id
                        {
                            (info.created_epoch, info.deleted_epoch)
                        } else {
                            (EpochId::PENDING, None)
                        }
                    }),
                    boundary,
                    boundary,
                ) == TransportEdgeState::Closed
            })
    }

    /// Performs the read-only structural half of resident purge qualification.
    /// No guard escapes this call, so an external representation preparer can
    /// safely read the overlay after it returns.
    #[cfg(feature = "tiered-storage")]
    fn resident_transport_histories_match(
        &self,
        identities: &[(EdgeId, NodeId, NodeId, u32, &TransportEdgeReceipt)],
        boundary: EpochId,
    ) -> bool {
        let edges = self.edge_versions.read();
        identities
            .iter()
            .all(|&(id, src, dst, expected_type_id, receipt)| {
                let Some(index) = edges.get(&id) else {
                    return false;
                };
                receipt.history_state(
                    index
                        .version_history()
                        .iter()
                        .map(|(created, deleted, version)| {
                            if version.created_by() == TransactionId::SYSTEM
                                && self.read_edge_record(version).is_some_and(|record| {
                                    record.src == src
                                        && record.dst == dst
                                        && record.type_id == expected_type_id
                                })
                            {
                                (*created, *deleted)
                            } else {
                                (EpochId::PENDING, None)
                            }
                        }),
                    boundary,
                    boundary,
                ) == TransportEdgeState::Closed
            })
    }

    /// Completes all resident qualification before invoking external compact
    /// preparation. The exclusive transition and identity reservation held by
    /// the caller keep this result stable against authoritative writers, while
    /// phase three still revalidates beneath the commit guards.
    fn resident_transport_purge_preflight(
        &self,
        resident_ids: &[EdgeId],
        identities: &[(EdgeId, NodeId, NodeId, u32, &TransportEdgeReceipt)],
        forward_edges: &[(NodeId, EdgeId)],
        backward_edges: &[(NodeId, EdgeId)],
        boundary: EpochId,
    ) -> bool {
        if !self.resident_transport_histories_match(identities, boundary)
            || !self
                .edge_properties
                .can_purge_all_history(resident_ids, boundary)
        {
            return false;
        }
        let exact_forward = self.forward_adj.exact_deleted_edges(forward_edges);
        if identities
            .iter()
            .any(|(id, _, _, _, _)| !exact_forward.contains(id))
        {
            return false;
        }
        self.backward_adj.as_ref().is_none_or(|backward| {
            let exact_backward = backward.exact_deleted_edges(backward_edges);
            identities
                .iter()
                .all(|(id, _, _, _, _)| exact_backward.contains(id))
        })
    }

    /// Resident-only convenience wrapper around the two-batch purge seam.
    pub(crate) fn purge_transport_extract_edges_after_prepare<E>(
        &self,
        resident_receipts: &[&TransportEdgeReceipt],
        prepare: impl FnOnce() -> Result<(), E>,
    ) -> Result<bool, E> {
        self.purge_transport_extract_edges_after_prepare_and_publish(
            resident_receipts,
            prepare,
            || {},
        )
    }

    /// Resident-only publication wrapper. A publisher that can partially
    /// publish before unwinding must catch and roll itself back.
    pub(crate) fn purge_transport_extract_edges_after_prepare_and_publish<E>(
        &self,
        resident_receipts: &[&TransportEdgeReceipt],
        prepare: impl FnOnce() -> Result<(), E>,
        publish: impl FnOnce(),
    ) -> Result<bool, E> {
        self.purge_transport_extract_edges_after_prepare_and_publish_with_rollback(
            resident_receipts,
            &[],
            |_| prepare(),
            |()| publish(),
            |()| {},
        )
        .map(|outcome| match outcome {
            PreparedPurgeOutcome::Rejected | PreparedPurgeOutcome::Abandoned(()) => false,
            PreparedPurgeOutcome::Published(()) => true,
            PreparedPurgeOutcome::UnwoundBeforePublication {
                prepared: (),
                payload,
            } => std::panic::resume_unwind(payload),
        })
    }

    /// Publishes an explicitly receipt-free empty generation under one exclusive
    /// LPG transition cut.
    ///
    /// This is deliberately distinct from transport purge: an empty receipt
    /// slice cannot mean either "no authority work" or "base-only authority
    /// work". On success the opaque publication token is returned only after
    /// the LPG transition guard has been released, so its destructor may safely
    /// retire an old generation or re-enter unrelated code. A hostile
    /// post-publication failpoint passes the token to the allocation-free
    /// rollback while the exact LPG incarnation remains pinned, then releases
    /// the guard before resuming the unwind. Rollback must not re-enter this LPG
    /// store.
    #[cfg(test)]
    pub(crate) fn publish_empty_generation_after_prepare_and_publish_with_rollback<E, R>(
        &self,
        prepare: impl FnOnce(&super::PinnedLpgTransition<'_>) -> Result<(), E>,
        publish: impl FnOnce() -> R,
        rollback: impl FnOnce(R),
    ) -> Result<Option<R>, E> {
        let Some(transition) = self.pin_exclusive_unframed_transition() else {
            return Ok(None);
        };
        #[cfg(test)]
        if let Some(barrier) = self.transport_purge_barrier.read().clone() {
            barrier.wait();
            barrier.wait();
        }
        prepare(&transition)?;
        let mut rollback_token = Some(publish());
        let boundary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(test)]
            if self
                .transport_post_publish_panic
                .swap(false, Ordering::SeqCst)
            {
                panic!("hostile post-publication LPG generation failpoint");
            }
        }));
        if let Err(payload) = boundary {
            rollback(rollback_token.take().expect("publisher returned a token"));
            drop(transition);
            std::panic::resume_unwind(payload);
        }
        drop(transition);
        Ok(rollback_token)
    }

    /// Atomically consumes resident and compact/base-only transport authority.
    ///
    /// `resident_receipts` must each name an exact resident, committed-closed
    /// SYSTEM lifetime with exact tombstoned adjacency. Their structural,
    /// adjacency, and property histories are physically removed.
    /// `authority_only_receipts` must each name a structurally absent overlay
    /// identity whose live nonce is carried solely for the compact base. The two
    /// batches must be disjoint and duplicate-free. Both batches are validated
    /// against one pinned store incarnation, then revalidated and revoked in one
    /// nonce-map write cut after external publication.
    ///
    /// `Rejected` is a qualification/authority miss (or completed rollback).
    /// `Abandoned` and `UnwoundBeforePublication` retain the prepared value;
    /// `Published` retains the committed token. Every outcome is returned only
    /// after all LPG guards and identity reservations have been released.
    /// A preparation error is returned before publication. The supplied hostile
    /// boundary rollback consumes its token after LPG guards are released.
    /// `publish` runs beneath allocation-complete resident entity/adjacency/
    /// property guards and must only swap the caller's already-prepared
    /// representation; it must not re-enter this LPG store. All overlay reads
    /// and other fallible work belong in `prepare`.
    pub(crate) fn purge_transport_extract_edges_after_prepare_and_publish_with_rollback<E, P, R>(
        &self,
        resident_receipts: &[&TransportEdgeReceipt],
        authority_only_receipts: &[&TransportEdgeReceipt],
        prepare: impl FnOnce(&super::PinnedLpgTransition<'_>) -> Result<P, E>,
        publish: impl FnOnce(P) -> R,
        rollback: impl FnOnce(R),
    ) -> Result<PreparedPurgeOutcome<P, R>, E> {
        if resident_receipts.is_empty() && authority_only_receipts.is_empty() {
            return Ok(PreparedPurgeOutcome::Rejected);
        }
        let Some(transition) = self.pin_exclusive_unframed_transition() else {
            return Ok(PreparedPurgeOutcome::Rejected);
        };
        let authority = transition.transport_authority();
        let resident_ids: Vec<EdgeId> =
            resident_receipts.iter().map(|receipt| receipt.id).collect();
        let authority_only_ids: Vec<EdgeId> = authority_only_receipts
            .iter()
            .map(|receipt| receipt.id)
            .collect();
        let Some(identity_reservation) =
            EdgeIdentityReservation::matching_occupancy(self, &resident_ids, &authority_only_ids)
        else {
            return Ok(PreparedPurgeOutcome::Rejected);
        };
        if !super::TransportExtractAuthority::contains_all(
            authority,
            resident_receipts
                .iter()
                .copied()
                .chain(authority_only_receipts.iter().copied()),
        ) {
            return Ok(PreparedPurgeOutcome::Rejected);
        }

        let mut all_ids = grafeo_common::utils::hash::FxHashSet::default();
        all_ids.reserve(resident_ids.len() + authority_only_ids.len());
        all_ids.extend(resident_ids.iter().chain(&authority_only_ids).copied());

        // Never consume authority while transaction-local state still refers to
        // either a resident or a compact/base-only identity.
        let pending_tx_creates = self.pending_tx_creates.read();
        let pending_tx_edge_deletes = self.pending_tx_edge_deletes.read();
        let tx_property_overlay = self.tx_property_overlay.read();
        let property_undo_log = self.property_undo_log.read();
        if pending_tx_creates
            .values()
            .any(|(_, edges)| edges.iter().any(|id| all_ids.contains(id)))
            || pending_tx_edge_deletes
                .values()
                .any(|edges| edges.iter().any(|(_, id, _)| all_ids.contains(id)))
            || tx_property_overlay
                .values()
                .any(|delta| delta.edge_props.keys().any(|(id, _)| all_ids.contains(id)))
            || property_undo_log.values().any(|entries| {
                entries.iter().any(|entry| match entry {
                    super::PropertyUndoEntry::EdgeProperty { edge_id, .. }
                    | super::PropertyUndoEntry::EdgeDeleted { edge_id, .. } => {
                        all_ids.contains(edge_id)
                    }
                    _ => false,
                })
            })
        {
            return Ok(PreparedPurgeOutcome::Rejected);
        }
        drop((
            pending_tx_creates,
            pending_tx_edge_deletes,
            tx_property_overlay,
            property_undo_log,
        ));

        // Authority-only batches still transport the required prepared value
        // directly; rollback sees its required result only after LPG drains.
        if resident_receipts.is_empty() {
            #[cfg(test)]
            if let Some(barrier) = self.transport_purge_barrier.read().clone() {
                barrier.wait();
                barrier.wait();
            }
            let prepared = prepare(&transition)?;
            let published = publish(prepared);
            let committed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                #[cfg(test)]
                if self
                    .transport_post_publish_panic
                    .swap(false, Ordering::SeqCst)
                {
                    panic!("hostile post-publication transport purge failpoint");
                }
                super::TransportExtractAuthority::revoke_all_if_present(
                    authority,
                    authority_only_receipts.iter().copied(),
                )
            }));
            return match committed {
                Ok(true) => {
                    identity_reservation.commit();
                    drop(transition);
                    Ok(PreparedPurgeOutcome::Published(published))
                }
                Ok(false) => {
                    drop(identity_reservation);
                    drop(transition);
                    rollback(published);
                    Ok(PreparedPurgeOutcome::Rejected)
                }
                Err(payload) => {
                    drop(identity_reservation);
                    drop(transition);
                    rollback(published);
                    std::panic::resume_unwind(payload)
                }
            };
        }

        // Phase one fully qualifies resident structure and derived history
        // without retaining any inner guard. Invalid input must not invoke an
        // external compact preparer or allocate a candidate generation.
        let expected_type_ids: Option<Vec<u32>> = {
            let edge_types = self.edge_type_to_id.read();
            resident_receipts
                .iter()
                .map(|receipt| edge_types.get(receipt.edge_type()).copied())
                .collect()
        };
        let Some(expected_type_ids) = expected_type_ids else {
            return Ok(PreparedPurgeOutcome::Rejected);
        };
        let identities: Vec<(EdgeId, NodeId, NodeId, u32, &TransportEdgeReceipt)> =
            resident_receipts
                .iter()
                .zip(expected_type_ids)
                .map(|(receipt, type_id)| (receipt.id, receipt.src, receipt.dst, type_id, *receipt))
                .collect();
        let boundary = self.current_epoch();
        let forward_edges: Vec<_> = identities
            .iter()
            .map(|(id, src, _, _, _)| (*src, *id))
            .collect();
        let backward_edges: Vec<_> = identities
            .iter()
            .map(|(id, _, dst, _, _)| (*dst, *id))
            .collect();
        if !self.resident_transport_purge_preflight(
            &resident_ids,
            &identities,
            &forward_edges,
            &backward_edges,
            boundary,
        ) {
            return Ok(PreparedPurgeOutcome::Rejected);
        }

        // Keep P outside the borrow-only late qualification catch. No
        // rejected/unwinding resident guard may destroy caller-owned metadata.
        let prepared = prepare(&transition)?;
        let qualified = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let exact_forward = self.forward_adj.exact_deleted_edges(&forward_edges);
            let exact_backward = self
                .backward_adj
                .as_ref()
                .map(|backward| backward.exact_deleted_edges(&backward_edges));
            if identities.iter().any(|(id, _, _, _, _)| {
                !exact_forward.contains(id)
                    || exact_backward
                        .as_ref()
                        .is_some_and(|backward| !backward.contains(id))
            }) {
                return None;
            }

            #[cfg(not(feature = "tiered-storage"))]
            let edges = {
                let edges = self.edges.write();
                for &(id, src, dst, expected_type_id, receipt) in &identities {
                    let chain = edges.get(&id)?;
                    if receipt.history_state(
                        chain.history().map(|(info, record)| {
                            if info.created_by == TransactionId::SYSTEM
                                && record.src == src
                                && record.dst == dst
                                && record.type_id == expected_type_id
                            {
                                (info.created_epoch, info.deleted_epoch)
                            } else {
                                (EpochId::PENDING, None)
                            }
                        }),
                        boundary,
                        boundary,
                    ) != TransportEdgeState::Closed
                    {
                        return None;
                    }
                }

                edges
            };

            #[cfg(feature = "tiered-storage")]
            let edges = {
                let edges = self.edge_versions.write();
                for &(id, src, dst, expected_type_id, receipt) in &identities {
                    let index = edges.get(&id)?;
                    if receipt.history_state(
                        index
                            .version_history()
                            .iter()
                            .map(|(created, deleted, version)| {
                                if version.created_by() == TransactionId::SYSTEM
                                    && self.read_edge_record(version).is_some_and(|record| {
                                        record.src == src
                                            && record.dst == dst
                                            && record.type_id == expected_type_id
                                    })
                                {
                                    (*created, *deleted)
                                } else {
                                    (EpochId::PENDING, None)
                                }
                            }),
                        boundary,
                        boundary,
                    ) != TransportEdgeState::Closed
                    {
                        return None;
                    }
                }

                edges
            };

            #[cfg(test)]
            if let Some(barrier) = self.transport_purge_barrier.read().clone() {
                barrier.wait();
                barrier.wait();
            }
            let property_purge = self
                .edge_properties
                .prepare_purge_all_history(&resident_ids, boundary)?;
            let forward_purge = self.forward_adj.prepare_purge_edges(&forward_edges)?;
            let backward_purge = if let Some(backward) = &self.backward_adj {
                let prepared = backward.prepare_purge_edges(&backward_edges)?;
                Some(prepared)
            } else {
                None
            };

            #[cfg(test)]
            match self.generation_purge_late_action.swap(0, Ordering::SeqCst) {
                1 => {
                    self.generation_purge_late_hits
                        .fetch_add(1, Ordering::SeqCst);
                    return None;
                }
                2 => {
                    self.generation_purge_late_hits
                        .fetch_add(1, Ordering::SeqCst);
                    panic!("generation late resident qualification unwind");
                }
                _ => {}
            }
            Some((edges, property_purge, forward_purge, backward_purge))
        }));
        let (mut edges, property_purge, forward_purge, backward_purge) = match qualified {
            Ok(Some(guards)) => guards,
            Ok(None) => {
                drop(identity_reservation);
                drop(transition);
                return Ok(PreparedPurgeOutcome::Abandoned(prepared));
            }
            Err(payload) => {
                drop(identity_reservation);
                drop(transition);
                return Ok(PreparedPurgeOutcome::UnwoundBeforePublication { prepared, payload });
            }
        };
        let mut property_purge = Some(property_purge);
        let mut forward_purge = Some(forward_purge);
        let mut backward_purge = backward_purge;
        let published = publish(prepared);
        let committed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(test)]
            if self
                .transport_post_publish_panic
                .swap(false, Ordering::SeqCst)
            {
                panic!("hostile post-publication transport purge failpoint");
            }
            if !super::TransportExtractAuthority::revoke_all_if_present(
                authority,
                resident_receipts
                    .iter()
                    .copied()
                    .chain(authority_only_receipts.iter().copied()),
            ) {
                return false;
            }
            for &(id, _, _, _, _) in &identities {
                drop(edges.remove(&id));
            }
            forward_purge
                .take()
                .expect("resident forward purge is prepared")
                .commit();
            if let Some(purge) = backward_purge.take() {
                purge.commit();
            }
            property_purge
                .take()
                .expect("resident property purge is prepared")
                .commit();
            true
        }));
        drop(backward_purge);
        drop(forward_purge);
        drop(property_purge);
        drop(edges);

        match committed {
            Ok(true) => {
                identity_reservation.commit();
                drop(transition);
                Ok(PreparedPurgeOutcome::Published(published))
            }
            Ok(false) => {
                drop(identity_reservation);
                drop(transition);
                rollback(published);
                Ok(PreparedPurgeOutcome::Rejected)
            }
            Err(payload) => {
                drop(identity_reservation);
                drop(transition);
                rollback(published);
                std::panic::resume_unwind(payload)
            }
        }
    }
    /// Deletes an edge (using latest epoch).
    pub fn delete_edge(&self, id: EdgeId) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        self.delete_edge_at_epoch(id, self.current_epoch())
    }

    /// Deletes an edge at a specific epoch.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut edges = self.edges.write();
        if let Some(chain) = edges.get_mut(&id) {
            // Get the visible record to check if deleted and get src/dst/type_id
            let (src, dst, type_id) = {
                match chain.visible_at(epoch) {
                    Some(record) => {
                        if record.is_deleted() {
                            return false;
                        }
                        (record.src, record.dst, record.type_id)
                    }
                    None => return false, // Not visible at this epoch (already deleted)
                }
            };

            // Mark the version chain as deleted
            chain.mark_deleted(epoch, TransactionId::SYSTEM);

            drop(edges); // Release lock

            // Mark as deleted in adjacency (soft delete)
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            self.edge_properties.remove_all(id, epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);

            true
        } else {
            false
        }
    }

    /// Deletes an edge at a specific epoch.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut versions = self.edge_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Get the visible record to check if deleted and get src/dst/type_id
            let (src, dst, type_id) = {
                match index.visible_at(epoch) {
                    Some(version_ref) => {
                        if let Some(record) = self.read_edge_record(&version_ref) {
                            if record.is_deleted() {
                                return false;
                            }
                            (record.src, record.dst, record.type_id)
                        } else {
                            return false;
                        }
                    }
                    None => return false,
                }
            };

            // Mark as deleted in version index
            index.mark_deleted(epoch, TransactionId::SYSTEM);

            drop(versions); // Release lock

            // Mark as deleted in adjacency (soft delete)
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            self.edge_properties.remove_all(id, epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);

            true
        } else {
            false
        }
    }

    /// Deletes an edge within a transaction using PENDING-epoch isolation.
    ///
    /// Mirrors `delete_node_transactional`: this method stamps the edge's version
    /// `deleted_epoch = EpochId::PENDING` so the deleting transaction sees the edge
    /// as gone (read-your-writes via `visible_to`, where `deleted_by == tx`), while
    /// every other session still sees it (PENDING > any real epoch in `is_visible_at`).
    ///
    /// It defers THREE things to commit (`finalize_edge_deletes_by_id`):
    ///  1. the adjacency tombstone (`forward_adj`/`backward_adj.mark_deleted`) — the
    ///     candidate index is non-MVCC, and the expand operators post-filter every
    ///     candidate through `is_edge_visible_versioned`, so it must stay populated
    ///     until commit (other sessions still traverse it);
    ///  2. edge-property removal — another session reading the edge's properties while
    ///     the delete is uncommitted must still see them; on rollback the writer's edge
    ///     keeps them (the writer never sees its own deleted edge — the chain hides it);
    ///  3. the live-edge / edge-type count decrements — an uncommitted or rolled-back
    ///     delete must not under-count.
    ///
    /// Because nothing but the version chain is touched, rollback is just
    /// `unmark_deleted_by(tx)` per edge (see `rollback_pending_edge_deletes`) — no
    /// heavy `PropertyUndoEntry::EdgeDeleted` undo entry is needed.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_edge_transactional(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut edges = self.edges.write();
        if let Some(chain) = edges.get_mut(&id) {
            // Tx-aware re-delete guard: `visible_to` hides an edge from the tx
            // that deleted it, so a PENDING delete by THIS tx returns `None`
            // here and the call is an idempotent no-op (mirrors the eager path,
            // which returned false on re-delete). A plain `visible_at(epoch)`
            // check would still see the record (PENDING `deleted_epoch` =
            // u64::MAX > epoch) and push a DUPLICATE pending entry, making
            // `finalize_edge_deletes_by_id` double-decrement the counts.
            let (src, dst, type_id) = match chain.visible_to(epoch, transaction_id) {
                Some(record) => (record.src, record.dst, record.type_id),
                None => return false,
            };

            // Stamp PENDING so the deleter sees it gone, others still see it.
            chain.mark_deleted(EpochId::PENDING, transaction_id);
            drop(edges);

            // Record for deferred finalize/rollback — adjacency tombstone, property
            // removal, and count decrements are deferred to `finalize_edge_deletes_by_id`.
            self.pending_tx_edge_deletes
                .write()
                .entry(transaction_id)
                .or_default()
                .push((src, id, dst));

            // Phantom coarse write: deleting the edge removes it from the :T set,
            // so an escalated RelType(T) reader must form an rw-antidependency.
            self.record_coarse_edge_write(transaction_id, id, EdgeTypeId::from(type_id));

            true
        } else {
            false
        }
    }

    /// Deletes an edge within a transaction using PENDING-epoch isolation.
    /// (Tiered storage version)
    ///
    /// Stamps `deleted_epoch = EpochId::PENDING` so other sessions still see the
    /// edge while the delete is uncommitted. The adjacency tombstone, edge-property
    /// removal, and live/edge-type count decrements are deferred to
    /// `finalize_edge_deletes_by_id` at commit time; rollback is just
    /// `unmark_deleted_by(tx)` (see `rollback_pending_edge_deletes`).
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_edge_transactional(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut versions = self.edge_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Tx-aware re-delete guard: `visible_to` hides an edge from the tx
            // that deleted it, so a PENDING delete by THIS tx returns `None`
            // here and the call is an idempotent no-op (mirrors the eager path,
            // which returned false on re-delete). A plain `visible_at(epoch)`
            // check would still see the record (PENDING `deleted_epoch` =
            // u64::MAX > epoch) and push a DUPLICATE pending entry, making
            // `finalize_edge_deletes_by_id` double-decrement the counts.
            let (src, dst, type_id) = match index.visible_to(epoch, transaction_id) {
                Some(version_ref) => match self.read_edge_record(&version_ref) {
                    Some(record) => (record.src, record.dst, record.type_id),
                    None => return false,
                },
                None => return false,
            };

            // Stamp PENDING so the deleter sees it gone, others still see it.
            index.mark_deleted(EpochId::PENDING, transaction_id);
            drop(versions);

            // Record for deferred finalize/rollback — adjacency tombstone, property
            // removal, and count decrements are deferred to `finalize_edge_deletes_by_id`.
            self.pending_tx_edge_deletes
                .write()
                .entry(transaction_id)
                .or_default()
                .push((src, id, dst));

            // Phantom coarse write: deleting the edge removes it from the :T set,
            // so an escalated RelType(T) reader must form an rw-antidependency.
            self.record_coarse_edge_write(transaction_id, id, EdgeTypeId::from(type_id));

            true
        } else {
            false
        }
    }

    /// Finalizes PENDING edge deletes for a committed transaction: stamps each
    /// deleted version's `deleted_epoch` PENDING→`commit_epoch` and applies the
    /// deferred adjacency tombstone, edge-property removal, and live/edge-type
    /// count decrements now that the delete is committed.
    ///
    /// Each tuple is `(src, edge, dst)`. The edge type for the count decrement is
    /// re-resolved from the chain (the eager path only decremented the counter, it
    /// never removed the type mapping), so the head version's `type_id` is still
    /// available even though the edge is now logically deleted.
    ///
    /// Wired into commit via the `GraphStoreMut::finalize_edge_deletes_by_id`
    /// trait override (`graph_store_impl.rs`), mirroring node's
    /// `finalize_deletes_by_id`.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn finalize_edge_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        edges: &[(NodeId, EdgeId, NodeId)],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if edges.is_empty() {
            return;
        }

        // Stamp PENDING→commit_epoch and capture each edge's type for the count
        // decrement, all under a single edges write lock.
        let mut type_ids: Vec<Option<u32>> = Vec::with_capacity(edges.len());
        {
            let mut edge_map = self.edges.write();
            for &(_, id, _) in edges {
                if let Some(chain) = edge_map.get_mut(&id) {
                    chain.finalize_deleted_epochs(transaction_id, commit_epoch);
                    type_ids.push(chain.latest().map(|r| r.type_id));
                } else {
                    type_ids.push(None);
                }
            }
        }

        // Apply the deferred adjacency tombstone now that the delete is committed.
        for &(src, id, dst) in edges {
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }
        }

        // Remove edge properties and decrement counts now that the delete is committed.
        for (&(_, id, _), type_id) in edges.iter().zip(type_ids) {
            self.edge_properties.remove_all(id, commit_epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            if let Some(type_id) = type_id {
                self.decrement_edge_type_count(type_id);
            }
        }
    }

    /// Finalizes PENDING edge deletes for a committed transaction.
    /// (Tiered storage version)
    // See the non-tiered variant: wired into commit via the
    // `GraphStoreMut::finalize_edge_deletes_by_id` trait override.
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn finalize_edge_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        edges: &[(NodeId, EdgeId, NodeId)],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if edges.is_empty() {
            return;
        }

        // Stamp PENDING→commit_epoch and capture each edge's type for the count
        // decrement, all under a single versions write lock.
        let mut type_ids: Vec<Option<u32>> = Vec::with_capacity(edges.len());
        {
            let mut versions = self.edge_versions.write();
            for &(_, id, _) in edges {
                if let Some(index) = versions.get_mut(&id) {
                    index.finalize_deleted_epochs(transaction_id, commit_epoch);
                    let type_id = index
                        .latest()
                        .and_then(|vref| self.read_edge_record(&vref))
                        .map(|r| r.type_id);
                    type_ids.push(type_id);
                } else {
                    type_ids.push(None);
                }
            }
        }

        // Apply the deferred adjacency tombstone now that the delete is committed.
        for &(src, id, dst) in edges {
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }
        }

        // Remove edge properties and decrement counts now that the delete is committed.
        for (&(_, id, _), type_id) in edges.iter().zip(type_ids) {
            self.edge_properties.remove_all(id, commit_epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            if let Some(type_id) = type_id {
                self.decrement_edge_type_count(type_id);
            }
        }
    }

    /// Takes (removes and returns) the pending edge-delete list for a transaction.
    #[doc(hidden)]
    pub fn take_pending_edge_deletes(
        &self,
        transaction_id: TransactionId,
    ) -> Vec<(NodeId, EdgeId, NodeId)> {
        let Some(_mutation) = self.pin_mutation() else {
            return Vec::new();
        };
        self.pending_tx_edge_deletes
            .write()
            .remove(&transaction_id)
            .unwrap_or_default()
    }

    /// Rolls back PENDING edge deletes for a transaction: clears the PENDING
    /// `deleted_epoch` on each edge's version chain so the edge is visible again.
    /// Adjacency, properties, and counts were never touched (deferred path), so no
    /// restoration is needed — just unmark the version chain.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn rollback_pending_edge_deletes(
        &self,
        transaction_id: TransactionId,
        edges: &[(NodeId, EdgeId, NodeId)],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if edges.is_empty() {
            return;
        }
        let mut edge_map = self.edges.write();
        for &(_, id, _) in edges {
            if let Some(chain) = edge_map.get_mut(&id) {
                chain.unmark_deleted_by(transaction_id);
            }
        }
    }

    /// Rolls back PENDING edge deletes for a transaction.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn rollback_pending_edge_deletes(
        &self,
        transaction_id: TransactionId,
        edges: &[(NodeId, EdgeId, NodeId)],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if edges.is_empty() {
            return;
        }
        let mut versions = self.edge_versions.write();
        for &(_, id, _) in edges {
            if let Some(index) = versions.get_mut(&id) {
                index.unmark_deleted_by(transaction_id);
            }
        }
    }

    /// Returns the number of edges (non-deleted at current epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_count(&self) -> usize {
        let _read = self.pin_read();
        let epoch = self.current_epoch();
        self.edges
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns the number of edges (non-deleted at current epoch).
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_count(&self) -> usize {
        let _read = self.pin_read();
        let epoch = self.current_epoch();
        let versions = self.edge_versions.read();
        versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_edge_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count()
    }

    /// Creates multiple edges in batch, significantly faster than calling
    /// `create_edge()` in a loop.
    ///
    /// Each tuple is `(src, dst, edge_type)`. Returns the assigned `EdgeId`s
    /// in the same order. Acquires the adjacency write lock once for all
    /// edges, rather than once per edge.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let Some(_mutation) = self.pin_mutation() else {
            return Vec::new();
        };
        if edges.is_empty() {
            return Vec::new();
        }
        {
            let nodes = self.nodes.read();
            if edges
                .iter()
                .any(|(src, dst, _)| !nodes.contains_key(src) || !nodes.contains_key(dst))
            {
                return Vec::new();
            }
        }

        let epoch = self.current_epoch();
        let authority = self.transport_extract_authority.read();
        let Some(identity_reservation) = EdgeIdentityReservation::generated(self, edges.len())
        else {
            return Vec::new();
        };
        let ids = identity_reservation.ids().to_vec();
        #[cfg(test)]
        self.pause_edge_publication_for_test();
        let type_names: Vec<&str> = edges.iter().map(|(_, _, edge_type)| *edge_type).collect();
        let Ok(prepared_types) = self.prepare_edge_types(&type_names) else {
            return Vec::new();
        };
        let type_ids: Vec<u32> = type_names
            .iter()
            .map(|edge_type| prepared_types.id(edge_type))
            .collect();
        let mut forward_batch = Vec::with_capacity(edges.len());
        let mut backward_batch = Vec::with_capacity(edges.len());
        let mut type_increments: grafeo_common::utils::hash::FxHashMap<u32, i64> =
            grafeo_common::utils::hash::FxHashMap::default();
        let records: Vec<_> = edges
            .iter()
            .zip(&ids)
            .zip(&type_ids)
            .map(|((&(src, dst, _), &id), &type_id)| {
                (
                    id,
                    VersionChain::with_initial(
                        EdgeRecord::new(id, src, dst, type_id, epoch),
                        epoch,
                        TransactionId::SYSTEM,
                    ),
                )
            })
            .collect();

        let nodes = self.nodes.read();
        if edges
            .iter()
            .any(|(src, dst, _)| !nodes.contains_key(src) || !nodes.contains_key(dst))
        {
            return Vec::new();
        }
        // Publish only fully built chains under the structural write lock.
        let mut edge_map = self.edges.write();
        if edge_map.try_reserve(records.len()).is_err() {
            return Vec::new();
        }
        prepared_types.commit(self);
        for id in &ids {
            authority.revoke_id(*id);
        }
        for (id, chain) in records {
            debug_assert!(
                !edge_map.contains_key(&id),
                "reserved edge identity is vacant"
            );
            edge_map.insert(id, chain);
        }
        drop(edge_map);
        for ((&(src, dst, _), &id), &type_id) in edges.iter().zip(&ids).zip(&type_ids) {
            forward_batch.push((src, dst, id));
            if self.backward_adj.is_some() {
                backward_batch.push((dst, src, id));
            }
            *type_increments.entry(type_id).or_default() += 1;
        }

        // Batch adjacency updates (single lock per direction)
        self.forward_adj.batch_add_edges(&forward_batch);
        if let Some(ref backward) = self.backward_adj {
            backward.batch_add_edges(&backward_batch);
        }

        // Update live counters
        // reason: edge batch size fits i64 for practical sizes
        #[allow(clippy::cast_possible_wrap)]
        let edge_count_i64 = edges.len() as i64;
        self.live_edge_count
            .fetch_add(edge_count_i64, Ordering::Relaxed);
        {
            let mut counts = self.edge_type_live_counts.write();
            for (type_id, increment) in type_increments {
                let idx = type_id as usize;
                if counts.len() <= idx {
                    counts.resize(idx + 1, 0);
                }
                counts[idx] += increment;
            }
        }

        drop(nodes);
        identity_reservation.commit();
        ids
    }

    /// Creates multiple edges in batch, significantly faster than calling
    /// `create_edge()` in a loop.
    /// (Tiered storage version)
    ///
    /// An arena allocation failure fails closed and returns an empty batch.
    #[cfg(feature = "tiered-storage")]
    pub fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let Some(_mutation) = self.pin_mutation() else {
            return Vec::new();
        };
        if edges.is_empty() {
            return Vec::new();
        }
        {
            let nodes = self.node_versions.read();
            if edges
                .iter()
                .any(|(src, dst, _)| !nodes.contains_key(src) || !nodes.contains_key(dst))
            {
                return Vec::new();
            }
        }

        let epoch = self.current_epoch();
        let authority = self.transport_extract_authority.read();
        let Some(identity_reservation) = EdgeIdentityReservation::generated(self, edges.len())
        else {
            return Vec::new();
        };
        let ids = identity_reservation.ids().to_vec();
        #[cfg(test)]
        self.pause_edge_publication_for_test();
        let type_names: Vec<&str> = edges.iter().map(|(_, _, edge_type)| *edge_type).collect();
        let Ok(prepared_types) = self.prepare_edge_types(&type_names) else {
            return Vec::new();
        };
        let type_ids: Vec<u32> = type_names
            .iter()
            .map(|edge_type| prepared_types.id(edge_type))
            .collect();
        let Ok(arena) = self.arena_allocator.arena_or_create(epoch) else {
            return Vec::new();
        };

        let mut forward_batch = Vec::with_capacity(edges.len());
        let mut backward_batch = Vec::with_capacity(edges.len());
        let mut type_increments: grafeo_common::utils::hash::FxHashMap<u32, i64> =
            grafeo_common::utils::hash::FxHashMap::default();
        let mut prepared = Vec::with_capacity(edges.len());
        for ((&(src, dst, _), &id), &type_id) in edges.iter().zip(&ids).zip(&type_ids) {
            let record = EdgeRecord::new(id, src, dst, type_id, epoch);
            let Ok((offset, _stored)) = arena.alloc_value_with_offset(record) else {
                return Vec::new();
            };
            prepared.push((
                id,
                VersionIndex::with_initial(HotVersionRef::new(
                    epoch,
                    epoch,
                    offset,
                    TransactionId::SYSTEM,
                )),
            ));
            forward_batch.push((src, dst, id));
            if self.backward_adj.is_some() {
                backward_batch.push((dst, src, id));
            }
            *type_increments.entry(type_id).or_default() += 1;
        }
        // The mutation pin keeps the arena alive through publication. Release
        // its registry guard before taking version locks: readers acquire them
        // in the opposite order, and a queued arena writer blocks new readers.
        drop(arena);

        let nodes = self.node_versions.read();
        if edges
            .iter()
            .any(|(src, dst, _)| !nodes.contains_key(src) || !nodes.contains_key(dst))
        {
            return Vec::new();
        }
        // Publish only fully allocated indexes under the structural lock.
        let mut versions = self.edge_versions.write();
        if versions.try_reserve(prepared.len()).is_err() {
            return Vec::new();
        }
        prepared_types.commit(self);
        for id in &ids {
            authority.revoke_id(*id);
        }
        for (id, index) in prepared {
            debug_assert!(
                !versions.contains_key(&id),
                "reserved edge identity is vacant"
            );
            versions.insert(id, index);
        }
        drop(versions);

        // Batch adjacency updates (single lock per direction)
        self.forward_adj.batch_add_edges(&forward_batch);
        if let Some(ref backward) = self.backward_adj {
            backward.batch_add_edges(&backward_batch);
        }

        // Update live counters
        // reason: edge batch size fits i64 for practical sizes
        #[allow(clippy::cast_possible_wrap)]
        let edge_count_i64 = edges.len() as i64;
        self.live_edge_count
            .fetch_add(edge_count_i64, Ordering::Relaxed);
        {
            let mut counts = self.edge_type_live_counts.write();
            for (type_id, increment) in type_increments {
                let idx = type_id as usize;
                if counts.len() <= idx {
                    counts.resize(idx + 1, 0);
                }
                counts[idx] += increment;
            }
        }

        drop(nodes);
        identity_reservation.commit();
        ids
    }

    /// Returns the committed numeric edge-type id for an edge, with NO read
    /// recording.
    ///
    /// Reads the edge version chain directly (mirrors
    /// [`committed_node_label_ids`](Self::committed_node_label_ids) on the node
    /// side): no SSI read-set pollution and no type-name resolution. Used by the
    /// transactional property-write paths to fan out the coarse `RelType(T)`
    /// phantom write so an escalated `RelType(T)` reader forms an
    /// rw-antidependency with a `SET e.p` / `REMOVE e.p`. Must NOT use
    /// `edge_type_versioned` here — that records the writer's own read.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn committed_edge_type_id(&self, id: EdgeId) -> Option<EdgeTypeId> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_at(self.current_epoch())?;
        Some(EdgeTypeId::from(record.type_id))
    }

    /// Returns the committed numeric edge-type id for an edge, with NO read
    /// recording. (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn committed_edge_type_id(&self, id: EdgeId) -> Option<EdgeTypeId> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let vref = index.visible_at(self.current_epoch())?;
        let record = self.read_edge_record(&vref)?;
        Some(EdgeTypeId::from(record.type_id))
    }

    /// Gets the type of an edge by ID.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let _read = self.pin_read();
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let epoch = self.current_epoch();
        let record = chain.visible_at(epoch)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Gets the type of an edge by ID.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let _read = self.pin_read();
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let epoch = self.current_epoch();
        let vref = index.visible_at(epoch)?;
        let record = self.read_edge_record(&vref)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Gets the type of an edge visible to a specific transaction.
    ///
    /// Used by operators that need edge type info for PENDING (uncommitted) edges.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        let _read = self.pin_read();
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_to(epoch, transaction_id)?;
        // Store-level read recording: record under the intrinsic RelType so that
        // Expand's per-candidate type-filter call participates in coarse escalation
        // (MATCH ()-[:T]->() avoids O(N) fine Edge entries).
        self.record_read_edge_in_rel_type(transaction_id, id, EdgeTypeId::from(record.type_id));
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Gets the type of an edge visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        let _read = self.pin_read();
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let vref = index.visible_to(epoch, transaction_id)?;
        // Resolve the record before recording so we know the intrinsic RelType.
        let record = self.read_edge_record(&vref)?;
        // Store-level read recording: record under the intrinsic RelType so that
        // Expand's per-candidate type-filter call participates in coarse escalation
        // (MATCH ()-[:T]->() avoids O(N) fine Edge entries).
        self.record_read_edge_in_rel_type(transaction_id, id, EdgeTypeId::from(record.type_id));
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    // --- Visibility checks (no type resolution or property loading) ---

    /// Checks if an edge is visible at the given epoch.
    ///
    /// Only checks the version chain, skips type resolution and property loading.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let edges = self.edges.read();
        edges
            .get(&id)
            .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
    }

    /// Checks if an edge is visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let versions = self.edge_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_at(epoch).is_some_and(|vref| {
                self.read_edge_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }

    /// Checks if an edge is visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let visible_type = self.edges.read().get(&id).and_then(|chain| {
            chain
                .visible_to(epoch, transaction_id)
                .filter(|r| !r.is_deleted())
                .map(|r| r.type_id)
        });
        if let Some(type_id) = visible_type {
            self.record_read_edge_in_rel_type(transaction_id, id, EdgeTypeId::from(type_id));
            true
        } else {
            false
        }
    }

    /// Checks if an edge is visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let vref = self
            .edge_versions
            .read()
            .get(&id)
            .and_then(|index| index.visible_to(epoch, transaction_id));
        let visible_type = vref
            .and_then(|vref| self.read_edge_record(&vref))
            .filter(|r| !r.is_deleted())
            .map(|r| r.type_id);
        if let Some(type_id) = visible_type {
            self.record_read_edge_in_rel_type(transaction_id, id, EdgeTypeId::from(type_id));
            true
        } else {
            false
        }
    }
}
