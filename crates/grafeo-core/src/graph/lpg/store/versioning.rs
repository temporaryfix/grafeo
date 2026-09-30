use super::{
    EdgeIdentityReservation, LpgStore, NodeIdentityReservation, PinnedMutation,
    TransportEdgeReceipt, TransportNonceReservation,
};
#[cfg(feature = "compact-store")]
use super::{PreparedNodeLabels, PreparedPromotionEdgeType};
use crate::graph::lpg::{EdgeRecord, NodeRecord};
#[cfg(feature = "compact-store")]
use crate::index::adjacency::AdjacencyUnpublishedMutation;
use arcstr::ArcStr;
use grafeo_common::memory::AllocError;
use grafeo_common::temporal::VersionLog;
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[cfg(test)]
mod fresh_edge_tests;
#[cfg(test)]
mod label_gc_tests;

fn fresh_edge_successor(id: EdgeId, src: NodeId, dst: NodeId) -> Result<u64, Error> {
    if !id.is_valid() || !src.is_valid() || !dst.is_valid() {
        return Err(Error::InvalidValue(
            "fresh edge insertion requires valid edge and endpoint IDs".into(),
        ));
    }
    id.as_u64()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidValue("fresh edge identity has no allocator successor".into()))
}

#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{ColdVersionRef, HotVersionRef, VersionIndex};

#[derive(Debug, Clone, Copy)]
struct RemovedEdgeIdentity {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    type_id: u32,
}

struct EdgeHistoryRestoreRequest<'data, 'store> {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: &'data str,
    lifetimes: &'data [(EpochId, Option<EpochId>)],
    allow_missing_destination: bool,
    identity_reservation: Option<EdgeIdentityReservation<'store>>,
    preserve_transport_nonce: bool,
}

/// Unforgeable, unwind-safe ownership of an unpublished compact-base node
/// promotion. Layered publication commits the token only after property replay
/// and dirty routing are both visible; dropping it restores exact vacancy.
#[must_use = "a same-incarnation promotion must be committed after routing publication"]
#[cfg(feature = "compact-store")]
pub(crate) struct SameIncarnationNodePromotion<'store> {
    store: &'store LpgStore,
    id: NodeId,
    structural_published: bool,
    live_counted: bool,
    identity_reservation: Option<NodeIdentityReservation<'store>>,
    prepared_labels: Option<PreparedNodeLabels<'store>>,
    mutation: Option<PinnedMutation<'store>>,
}

#[cfg(feature = "compact-store")]
impl SameIncarnationNodePromotion<'_> {
    pub(crate) fn commit(mut self) {
        self.store
            .next_node_id
            .fetch_max(self.id.as_u64().saturating_add(1), Ordering::AcqRel);
        if let Some(reservation) = self.identity_reservation.take() {
            reservation.commit();
        }
        if let Some(labels) = self.prepared_labels.take() {
            labels.commit();
        }
        self.mutation.take();
    }
}

#[cfg(feature = "compact-store")]
impl Drop for SameIncarnationNodePromotion<'_> {
    fn drop(&mut self) {
        if let Some(mutation) = self.mutation.as_ref() {
            self.store.rollback_same_incarnation_node_promotion(
                self.id,
                self.structural_published,
                self.live_counted,
                mutation,
            );
        }
    }
}

/// Unforgeable, unwind-safe ownership of an unpublished compact-base edge
/// promotion. Rollback deliberately preserves transport nonce state because
/// representation-only hydration never grants or revokes transport authority.
#[must_use = "a same-incarnation promotion must be committed after routing publication"]
#[cfg(feature = "compact-store")]
pub(crate) struct SameIncarnationEdgePromotion<'store> {
    store: &'store LpgStore,
    removed: RemovedEdgeIdentity,
    structural_published: bool,
    live_counted: bool,
    type_counted: bool,
    identity_reservation: Option<EdgeIdentityReservation<'store>>,
    prepared_type: Option<PreparedPromotionEdgeType<'store>>,
    forward_adjacency: Option<AdjacencyUnpublishedMutation<'store>>,
    backward_adjacency: Option<AdjacencyUnpublishedMutation<'store>>,
    mutation: Option<PinnedMutation<'store>>,
}

#[cfg(feature = "compact-store")]
impl SameIncarnationEdgePromotion<'_> {
    pub(crate) fn commit(mut self) {
        if let Some(adjacency) = self.forward_adjacency.take() {
            adjacency.commit();
        }
        if let Some(adjacency) = self.backward_adjacency.take() {
            adjacency.commit();
        }
        self.store
            .next_edge_id
            .fetch_max(self.removed.id.as_u64().saturating_add(1), Ordering::AcqRel);
        if let Some(reservation) = self.identity_reservation.take() {
            reservation.commit();
        }
        if let Some(edge_type) = self.prepared_type.take() {
            edge_type.commit();
        }
        self.mutation.take();
    }
}

#[cfg(feature = "compact-store")]
impl Drop for SameIncarnationEdgePromotion<'_> {
    fn drop(&mut self) {
        // Adjacency was acquired beneath the structural map. Release and
        // rollback those receipts first: another creator may already hold
        // `edges` while waiting for adjacency, so taking `edges` while these
        // guards remain held would invert the global entity→derived order.
        drop(self.backward_adjacency.take());
        drop(self.forward_adjacency.take());
        if let Some(mutation) = self.mutation.as_ref() {
            self.store.rollback_same_incarnation_edge_promotion(
                self.removed,
                self.structural_published,
                self.live_counted,
                self.type_counted,
                mutation,
            );
        }
    }
}

/// Validates committed half-open structural lifetimes in oldest-first order.
fn validate_restore_lifetimes(
    entity: &str,
    lifetimes: &[(EpochId, Option<EpochId>)],
) -> Result<EpochId, String> {
    let Some(&(first_created, _)) = lifetimes.first() else {
        return Err(format!(
            "{entity} history must contain at least one lifetime"
        ));
    };
    if first_created == EpochId::PENDING {
        return Err(format!("{entity} history cannot contain PENDING epochs"));
    }

    let mut max_epoch = first_created;
    for (index, &(created, deleted)) in lifetimes.iter().enumerate() {
        if created == EpochId::PENDING || deleted == Some(EpochId::PENDING) {
            return Err(format!(
                "{entity} lifetime {index} contains a PENDING epoch"
            ));
        }
        if let Some(deleted) = deleted {
            if deleted < created {
                return Err(format!(
                    "{entity} lifetime {index} must satisfy created <= deleted"
                ));
            }
            max_epoch = max_epoch.max(deleted);
        }
        max_epoch = max_epoch.max(created);

        if let Some(&(prior_created, prior_deleted)) =
            index.checked_sub(1).and_then(|prior| lifetimes.get(prior))
        {
            let Some(prior_deleted) = prior_deleted else {
                return Err(format!(
                    "{entity} lifetime {index} follows open lifetime {prior_created:?}"
                ));
            };
            if created < prior_deleted {
                return Err(format!(
                    "{entity} lifetime {index} overlaps its predecessor"
                ));
            }
        }
    }
    Ok(max_epoch)
}

/// Validates epoch-ordered complete label-set versions against node lives.
fn validate_restore_label_versions(
    lifetimes: &[(EpochId, Option<EpochId>)],
    label_versions: &[(EpochId, Vec<ArcStr>)],
) -> Result<EpochId, String> {
    let Some(&(first_epoch, _)) = label_versions.first() else {
        return Err("node history must contain a complete label set at creation".into());
    };
    if first_epoch != lifetimes[0].0 {
        return Err("node label history must begin at the first create epoch".into());
    }

    let mut prior_epoch = None;
    let mut max_epoch = first_epoch;
    for (index, (epoch, labels)) in label_versions.iter().enumerate() {
        if *epoch == EpochId::PENDING {
            return Err(format!("node label version {index} has a PENDING epoch"));
        }
        if prior_epoch.is_some_and(|prior| *epoch < prior) {
            return Err("node label versions must be ordered by ascending epoch".into());
        }
        prior_epoch = Some(*epoch);
        max_epoch = max_epoch.max(*epoch);

        if labels.len() > usize::from(u16::MAX) {
            return Err(format!(
                "node label version {index} exceeds u16::MAX labels"
            ));
        }

        let mut unique = FxHashSet::default();
        for label in labels {
            if !unique.insert(label.as_str()) {
                return Err(format!(
                    "node label version {index} contains duplicate label {label}"
                ));
            }
        }

        let inside_or_delete_boundary = lifetimes.iter().any(|(created, deleted)| {
            *created <= *epoch && deleted.is_none_or(|deleted| *epoch <= deleted)
        });
        if !inside_or_delete_boundary {
            return Err(format!(
                "node label version {index} is outside every structural lifetime and delete boundary"
            ));
        }
    }

    for (index, (created, _)) in lifetimes.iter().enumerate() {
        if !label_versions.iter().any(|(epoch, _)| epoch == created) {
            return Err(format!(
                "node lifetime {index} has no complete label set at its create epoch"
            ));
        }
    }
    Ok(max_epoch)
}

fn canonical_label_witness(mut labels: Vec<ArcStr>) -> Arc<[ArcStr]> {
    labels.sort_unstable();
    labels.dedup();
    labels.into()
}

impl LpgStore {
    /// Physically removes every non-structural trace of nodes whose only
    /// lifetime was rolled back. The caller retains the node-map write guard,
    /// preventing same-ID recovery from publishing until cleanup is complete.
    fn purge_removed_node_identities(
        &self,
        ids: &[NodeId],
        live_count: usize,
        mutation: &PinnedMutation<'_>,
    ) {
        if ids.is_empty() {
            return;
        }

        for &id in ids {
            self.purge_rolled_back_node_from_derived_indexes_inner(id, mutation);
        }
        {
            let mut index = self.label_index.write();
            let mut labels = self.node_labels.write();
            for id in ids {
                if let Some(label_ids) = labels.get(id).and_then(VersionLog::latest) {
                    for &label_id in label_ids {
                        if let Some(members) = index.get_mut(label_id as usize) {
                            members.remove(id);
                        }
                    }
                }
                labels.remove(id);
            }
        }
        {
            let mut columns = self.node_properties.columns_write();
            for column in columns.values_mut() {
                for id in ids {
                    column.remove_pending_for(*id);
                }
            }
        }
        self.node_properties
            .prepare_purge_all_history(ids, self.current_epoch())
            .expect("an absent node identity cannot retain future committed properties")
            .commit();
        self.live_node_count.fetch_sub(
            i64::try_from(live_count).unwrap_or(i64::MAX),
            Ordering::Relaxed,
        );
    }

    /// Physically removes adjacency, properties, counters, and stale transport
    /// provenance for edge identities whose only lifetime was rolled back.
    /// The caller retains an out-of-map identity reservation through this cut.
    fn purge_removed_edge_identities(
        &self,
        removed: &[RemovedEdgeIdentity],
        authority: &Arc<super::TransportExtractAuthority>,
        live_count: usize,
        revoke_transport_nonce: bool,
    ) {
        if removed.is_empty() {
            return;
        }

        let forward: Vec<_> = removed.iter().map(|edge| (edge.src, edge.id)).collect();
        let backward: Vec<_> = removed.iter().map(|edge| (edge.dst, edge.id)).collect();
        for &(src, id) in &forward {
            if !self.forward_adj.is_exact_deleted_edge(src, id) {
                self.forward_adj.mark_deleted(src, id);
            }
        }
        if let Some(backward_index) = &self.backward_adj {
            for &(dst, id) in &backward {
                if !backward_index.is_exact_deleted_edge(dst, id) {
                    backward_index.mark_deleted(dst, id);
                }
            }
        }

        let ids: Vec<_> = removed.iter().map(|edge| edge.id).collect();
        {
            let mut columns = self.edge_properties.columns_write();
            for column in columns.values_mut() {
                for id in &ids {
                    column.remove_pending_for(*id);
                }
            }
        }
        let property_purge = self
            .edge_properties
            .prepare_purge_all_history(&ids, self.current_epoch())
            .expect("an absent edge identity cannot retain future committed properties");
        let forward_purge = self
            .forward_adj
            .prepare_purge_edges(&forward)
            .expect("a published edge identity has exact forward adjacency");
        let backward_purge = self.backward_adj.as_ref().map(|index| {
            index
                .prepare_purge_edges(&backward)
                .expect("a published edge identity has exact backward adjacency")
        });

        forward_purge.commit();
        if let Some(purge) = backward_purge {
            purge.commit();
        }
        property_purge.commit();
        self.live_edge_count.fetch_sub(
            i64::try_from(live_count).unwrap_or(i64::MAX),
            Ordering::Relaxed,
        );
        for edge in removed {
            self.decrement_edge_type_count(edge.type_id);
            if revoke_transport_nonce {
                authority.revoke_id(edge.id);
            }
        }
    }

    #[cfg(feature = "compact-store")]
    fn rollback_same_incarnation_node_promotion(
        &self,
        id: NodeId,
        structural_published: bool,
        live_counted: bool,
        _mutation: &PinnedMutation<'_>,
    ) {
        #[cfg(not(feature = "tiered-storage"))]
        if structural_published {
            self.nodes.write().remove(&id);
        }
        #[cfg(feature = "tiered-storage")]
        if structural_published {
            self.node_versions.write().remove(&id);
        }
        self.purge_node_from_local_property_indexes_inner(id, _mutation);
        {
            let mut index = self.label_index.write();
            let mut labels = self.node_labels.write();
            if let Some(label_ids) = labels.get(&id).and_then(VersionLog::latest) {
                for &label_id in label_ids {
                    if let Some(members) = index.get_mut(label_id as usize) {
                        members.remove(&id);
                    }
                }
            }
            labels.remove(&id);
        }
        self.node_properties.purge_identity_in_place(id);
        if live_counted {
            self.live_node_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    #[cfg(feature = "compact-store")]
    fn rollback_same_incarnation_edge_promotion(
        &self,
        removed: RemovedEdgeIdentity,
        structural_published: bool,
        live_counted: bool,
        type_counted: bool,
        _mutation: &PinnedMutation<'_>,
    ) {
        #[cfg(not(feature = "tiered-storage"))]
        if structural_published {
            self.edges.write().remove(&removed.id);
        }
        #[cfg(feature = "tiered-storage")]
        if structural_published {
            self.edge_versions.write().remove(&removed.id);
        }
        self.edge_properties.purge_identity_in_place(removed.id);
        if live_counted {
            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
        }
        if type_counted {
            self.decrement_edge_type_count(removed.type_id);
        }
    }

    #[cfg(not(feature = "tiered-storage"))]
    fn discard_edge_versions_by(&self, transaction_id: TransactionId, selected: Option<&[EdgeId]>) {
        let authority = Arc::clone(&self.transport_extract_authority.read());
        let mut cleanup_reservation = EdgeIdentityReservation {
            registry: &self.edge_identity_reservations,
            ids: Vec::new(),
            armed: true,
        };
        let mut reservations = self.edge_identity_reservations.identities.lock();
        let mut edges = self.edges.write();
        let candidates: Vec<EdgeId> =
            selected.map_or_else(|| edges.keys().copied().collect(), |ids| ids.to_vec());
        let mut removed = Vec::with_capacity(candidates.len());
        for id in candidates {
            if reservations.contains(&id) {
                continue;
            }
            let Some(chain) = edges.get_mut(&id) else {
                continue;
            };
            let removed_record = chain
                .history()
                .find_map(|(info, record)| (info.created_by == transaction_id).then_some(*record));
            chain.remove_versions_by(transaction_id);
            if chain.is_empty() {
                let record = removed_record
                    .expect("an emptied edge chain contained a version owned by the rollback");
                edges.remove(&id);
                reservations.insert(id);
                cleanup_reservation.ids.push(id);
                removed.push(RemovedEdgeIdentity {
                    id,
                    src: record.src,
                    dst: record.dst,
                    type_id: record.type_id,
                });
            }
        }
        drop(edges);
        drop(reservations);
        self.purge_removed_edge_identities(&removed, &authority, removed.len(), true);
        cleanup_reservation.commit();
    }

    #[cfg(feature = "tiered-storage")]
    fn discard_edge_versions_by(&self, transaction_id: TransactionId, selected: Option<&[EdgeId]>) {
        let authority = Arc::clone(&self.transport_extract_authority.read());
        let mut cleanup_reservation = EdgeIdentityReservation {
            registry: &self.edge_identity_reservations,
            ids: Vec::new(),
            armed: true,
        };
        let mut reservations = self.edge_identity_reservations.identities.lock();
        let mut edges = self.edge_versions.write();
        let candidates: Vec<EdgeId> =
            selected.map_or_else(|| edges.keys().copied().collect(), |ids| ids.to_vec());
        let mut removed = Vec::with_capacity(candidates.len());
        for id in candidates {
            if reservations.contains(&id) {
                continue;
            }
            let Some(index) = edges.get_mut(&id) else {
                continue;
            };
            let removed_record = index.version_history().iter().find_map(|(_, _, version)| {
                (version.created_by() == transaction_id)
                    .then(|| self.read_edge_record(version))
                    .flatten()
            });
            index.remove_versions_by(transaction_id);
            if index.is_empty() {
                let record = removed_record
                    .expect("an emptied edge index referenced the rollback-owned record");
                edges.remove(&id);
                reservations.insert(id);
                cleanup_reservation.ids.push(id);
                removed.push(RemovedEdgeIdentity {
                    id,
                    src: record.src,
                    dst: record.dst,
                    type_id: record.type_id,
                });
            }
        }
        drop(edges);
        drop(reservations);
        self.purge_removed_edge_identities(&removed, &authority, removed.len(), true);
        cleanup_reservation.commit();
    }

    #[cfg(not(feature = "tiered-storage"))]
    fn discard_node_versions_by(
        &self,
        transaction_id: TransactionId,
        selected: Option<&[NodeId]>,
        mutation: &PinnedMutation<'_>,
    ) {
        let candidates: Vec<NodeId> = selected.map_or_else(
            || self.nodes.read().keys().copied().collect(),
            <[NodeId]>::to_vec,
        );
        for id in candidates {
            let Some(identity_reservation) = NodeIdentityReservation::resident(self, id) else {
                continue;
            };
            let removed = {
                let mut nodes = self.nodes.write();
                let Some(chain) = nodes.get_mut(&id) else {
                    continue;
                };
                chain.remove_versions_by(transaction_id);
                if chain.is_empty() {
                    nodes.remove(&id);
                    true
                } else {
                    false
                }
            };
            if removed {
                self.purge_removed_node_identities(&[id], 1, mutation);
                identity_reservation.commit();
            }
        }
    }

    #[cfg(feature = "tiered-storage")]
    fn discard_node_versions_by(
        &self,
        transaction_id: TransactionId,
        selected: Option<&[NodeId]>,
        mutation: &PinnedMutation<'_>,
    ) {
        let candidates: Vec<NodeId> = selected.map_or_else(
            || self.node_versions.read().keys().copied().collect(),
            <[NodeId]>::to_vec,
        );
        for id in candidates {
            let Some(identity_reservation) = NodeIdentityReservation::resident(self, id) else {
                continue;
            };
            let removed = {
                let mut nodes = self.node_versions.write();
                let Some(index) = nodes.get_mut(&id) else {
                    continue;
                };
                index.remove_versions_by(transaction_id);
                if index.is_empty() {
                    nodes.remove(&id);
                    true
                } else {
                    false
                }
            };
            if removed {
                self.purge_removed_node_identities(&[id], 1, mutation);
                identity_reservation.commit();
            }
        }
    }

    /// Discards all uncommitted versions created by a transaction.
    ///
    /// This is called during transaction rollback to clean up uncommitted changes.
    /// The method removes version chain entries created by the specified transaction,
    /// and replays the property undo log to restore property values.
    #[doc(hidden)]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn discard_uncommitted_versions(&self, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        self.discard_edge_versions_by(transaction_id, None);
        self.discard_node_versions_by(transaction_id, None, &_mutation);

        // Replay property undo log to restore pre-transaction property values
        self.rollback_transaction_properties(transaction_id);

        // Counters may be out of sync after rollback: force full recompute
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards uncommitted versions for specific entities created by a transaction.
    ///
    /// Used for savepoint rollback: only reverts the entities written after
    /// the savepoint, keeping earlier writes intact.
    #[doc(hidden)]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn discard_entities_by_id(
        &self,
        transaction_id: TransactionId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        self.discard_edge_versions_by(transaction_id, Some(edge_ids));
        self.discard_node_versions_by(transaction_id, Some(node_ids), &_mutation);

        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards all uncommitted versions created by a transaction.
    /// (Tiered storage version)
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    pub fn discard_uncommitted_versions(&self, transaction_id: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        self.discard_edge_versions_by(transaction_id, None);
        self.discard_node_versions_by(transaction_id, None, &_mutation);

        // Replay property undo log to restore pre-transaction property values
        self.rollback_transaction_properties(transaction_id);

        // Counters may be out of sync after rollback: force full recompute
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards uncommitted versions for specific entities (tiered storage version).
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    pub fn discard_entities_by_id(
        &self,
        transaction_id: TransactionId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if transaction_id == TransactionId::SYSTEM {
            return;
        }
        self.discard_edge_versions_by(transaction_id, Some(edge_ids));
        self.discard_node_versions_by(transaction_id, Some(node_ids), &_mutation);

        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Finalizes PENDING epochs for all versions created by a transaction.
    ///
    /// Called at commit time: updates `created_epoch` from `EpochId::PENDING`
    /// to the real `commit_epoch`, making the versions visible to other sessions.
    /// Also advances the store's epoch so non-transactional reads can see the
    /// newly committed versions.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn finalize_version_epochs(&self, transaction_id: TransactionId, commit_epoch: EpochId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.finalize_epochs(transaction_id, commit_epoch);
            }
        }
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.finalize_epochs(transaction_id, commit_epoch);
            }
        }

        // Finalize PENDING epochs in property and label version logs
        {
            self.finalize_property_index_history(transaction_id, commit_epoch);
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Finalizes PENDING epochs for the named entities only (write-set-scoped).
    ///
    /// Commit-time analogue of [`discard_entities_by_id`](Self::discard_entities_by_id):
    /// finalizes only the version chains of `node_ids`/`edge_ids` instead of
    /// scanning every chain, turning O(all entities) commit into O(entities
    /// written). The temporal property/label `finalize_pending` stays bulk
    /// (per-entity property finalize is part of the Wave 2b storage restructure).
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn finalize_entities_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if !node_ids.is_empty() {
            let mut nodes = self.nodes.write();
            for &nid in node_ids {
                if let Some(chain) = nodes.get_mut(&nid) {
                    chain.finalize_epochs(transaction_id, commit_epoch);
                }
            }
        }
        if !edge_ids.is_empty() {
            let mut edges = self.edges.write();
            for &eid in edge_ids {
                if let Some(chain) = edges.get_mut(&eid) {
                    chain.finalize_epochs(transaction_id, commit_epoch);
                }
            }
        }

        // Finalize PENDING epochs in property and label version logs (bulk).
        {
            self.finalize_property_index_history(transaction_id, commit_epoch);
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Finalizes PENDING epochs for all versions created by a transaction.
    /// (Tiered storage version, also syncs the store epoch.)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn finalize_version_epochs(&self, transaction_id: TransactionId, commit_epoch: EpochId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.finalize_epochs(transaction_id, commit_epoch);
            }
        }
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.finalize_epochs(transaction_id, commit_epoch);
            }
        }

        // Finalize PENDING epochs in property and label version logs
        {
            self.finalize_property_index_history(transaction_id, commit_epoch);
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Finalizes PENDING epochs for the named entities only (write-set-scoped).
    /// (Tiered storage version.)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn finalize_entities_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if !node_ids.is_empty() {
            let mut versions = self.node_versions.write();
            for &nid in node_ids {
                if let Some(index) = versions.get_mut(&nid) {
                    index.finalize_epochs(transaction_id, commit_epoch);
                }
            }
        }
        if !edge_ids.is_empty() {
            let mut versions = self.edge_versions.write();
            for &eid in edge_ids {
                if let Some(index) = versions.get_mut(&eid) {
                    index.finalize_epochs(transaction_id, commit_epoch);
                }
            }
        }

        {
            self.finalize_property_index_history(transaction_id, commit_epoch);
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Garbage collects old versions that are no longer visible to any transaction.
    ///
    /// Versions older than `min_epoch` are pruned from version chains, keeping
    /// at most one old version per entity as a baseline. Empty chains are removed.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn gc_versions(&self, min_epoch: EpochId) {
        if min_epoch == EpochId::PENDING {
            return;
        }
        // Exporters must not observe a partially pruned history with its old
        // coverage floor. Use the existing exclusive representation barrier.
        let Some(_mutation) = self.pin_exclusive_authorized_scope() else {
            return;
        };
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.gc(min_epoch);
            }
            nodes.retain(|_, chain| !chain.is_empty());
        }
        {
            let authority = self.transport_extract_authority.read();
            let reservations = self.edge_identity_reservations.identities.lock();
            let mut edges = self.edges.write();
            let mut removed = Vec::with_capacity(edges.len());
            for (id, chain) in edges.iter_mut() {
                if reservations.contains(id) {
                    continue;
                }
                chain.gc(min_epoch);
                if chain.is_empty() {
                    removed.push(*id);
                }
            }
            for id in removed {
                edges.remove(&id);
                authority.revoke_id(id);
            }
        }

        // GC old property and label versions
        {
            self.node_properties.gc(min_epoch);
            self.edge_properties.gc(min_epoch);
            // Structural metadata precedes label logs in the lock order. Each
            // surviving lifetime must retain its exact creation-time labels,
            // even when ordinary label GC would keep only a newer baseline.
            let nodes = self.nodes.read();
            let mut labels = self.node_labels.write();
            for (id, log) in labels.iter_mut() {
                if let Some(chain) = nodes.get(id) {
                    log.gc_preserving(min_epoch, |epoch| {
                        chain.history().any(|(info, _)| info.created_epoch == epoch)
                    });
                } else {
                    log.gc(min_epoch);
                }
            }
            labels.retain(|_, log| !log.is_empty());
        }
        self.finish_history_gc(min_epoch);
    }

    /// Garbage collects old versions (tiered storage variant).
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn gc_versions(&self, min_epoch: EpochId) {
        if min_epoch == EpochId::PENDING {
            return;
        }
        let Some(_mutation) = self.pin_exclusive_authorized_scope() else {
            return;
        };
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.gc(min_epoch);
            }
            versions.retain(|_, index| !index.is_empty());
        }
        {
            let authority = self.transport_extract_authority.read();
            let reservations = self.edge_identity_reservations.identities.lock();
            let mut versions = self.edge_versions.write();
            let mut removed = Vec::with_capacity(versions.len());
            for (id, index) in versions.iter_mut() {
                if reservations.contains(id) {
                    continue;
                }
                index.gc(min_epoch);
                if index.is_empty() {
                    removed.push(*id);
                }
            }
            for id in removed {
                versions.remove(&id);
                authority.revoke_id(id);
            }
        }

        // GC old property and label versions
        {
            self.node_properties.gc(min_epoch);
            self.edge_properties.gc(min_epoch);
            // Read structural metadata, never reconstructed nodes: building a
            // node would read this same label log while its write lock is held.
            let versions = self.node_versions.read();
            let mut labels = self.node_labels.write();
            for (id, log) in labels.iter_mut() {
                if let Some(index) = versions.get(id) {
                    log.gc_preserving(min_epoch, |epoch| index.has_creation_epoch(epoch));
                } else {
                    log.gc(min_epoch);
                }
            }
            labels.retain(|_, log| !log.is_empty());
        }
        self.finish_history_gc(min_epoch);
    }

    fn finish_history_gc(&self, min_epoch: EpochId) {
        self.advance_retained_history_floor(min_epoch);
        // A terminal property tombstone is obsolete once every reader sees
        // its removal. Compact overlays still need it to mask their base.
        #[cfg(feature = "compact-store")]
        let has_base = self.compact_base.get().is_some();
        #[cfg(not(feature = "compact-store"))]
        let has_base = false;
        // Vector GC follows history GC and requires a removal witness before
        // retiring a soft-deleted membership. Missing history is not proof.
        #[cfg(feature = "vector-index")]
        let has_vector_indexes = !self.vector_indexes.read().is_empty();
        #[cfg(not(feature = "vector-index"))]
        let has_vector_indexes = false;
        if !has_base && !has_vector_indexes {
            self.node_properties.gc_expired_tombstones(min_epoch);
        }
        if !has_base {
            self.edge_properties.gc_expired_tombstones(min_epoch);
        }
        let indexes = self.property_indexes.read();
        for index in indexes.values() {
            index.history.write().gc(min_epoch);
        }
    }

    /// Freezes an epoch from hot (arena) storage to cold (compressed) storage.
    ///
    /// This is called by the transaction manager when an epoch becomes eligible
    /// for freezing (no active transactions can see it). The freeze process:
    ///
    /// 1. Collects all hot version refs for the epoch
    /// 2. Reads the corresponding records from arena
    /// 3. Compresses them into a `CompressedEpochBlock`
    /// 4. Updates `VersionIndex` entries to point to cold storage
    /// 5. The arena can be deallocated after all epochs in it are frozen
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch to freeze
    ///
    /// # Returns
    ///
    /// The number of records frozen (nodes + edges).
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub fn freeze_epoch(&self, epoch: EpochId) -> usize {
        let Some(_mutation) = self.pin_mutation() else {
            return 0;
        };
        // Collect node records to freeze
        let mut node_records: Vec<(u64, NodeRecord)> = Vec::new();
        let mut node_hot_refs: Vec<(NodeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.node_versions.read();
            for (node_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist for hot version ref");
                    // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                    let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    node_records.push((node_id.as_u64(), *record));
                    node_hot_refs.push((*node_id, *hot_ref));
                }
            }
        }

        // Collect edge records to freeze
        let mut edge_records: Vec<(u64, EdgeRecord)> = Vec::new();
        let mut edge_hot_refs: Vec<(EdgeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.edge_versions.read();
            for (edge_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist for hot version ref");
                    // SAFETY: The offset was returned by alloc_value_with_offset for an EdgeRecord
                    let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    edge_records.push((edge_id.as_u64(), *record));
                    edge_hot_refs.push((*edge_id, *hot_ref));
                }
            }
        }

        let total_frozen = node_records.len() + edge_records.len();

        if total_frozen == 0 {
            return 0;
        }

        // Freeze to compressed storage
        let (node_entries, edge_entries) =
            self.epoch_store
                .freeze_epoch(epoch, node_records, edge_records);

        // Build lookup maps for index entries
        let node_entry_map: FxHashMap<u64, _> = node_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();
        let edge_entry_map: FxHashMap<u64, _> = edge_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();

        // Update version indexes to use cold refs
        {
            let mut versions = self.node_versions.write();
            for (node_id, hot_ref) in &node_hot_refs {
                if let Some(index) = versions.get_mut(node_id)
                    && let Some(&(offset, length)) = node_entry_map.get(&node_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                        deleted_by: hot_ref.deleted_by,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        {
            let mut versions = self.edge_versions.write();
            for (edge_id, hot_ref) in &edge_hot_refs {
                if let Some(index) = versions.get_mut(edge_id)
                    && let Some(&(offset, length)) = edge_entry_map.get(&edge_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                        deleted_by: hot_ref.deleted_by,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        total_frozen
    }

    // === Recovery Support ===

    /// Restores one node identity with exact committed structural and label history.
    ///
    /// `lifetimes` must be non-empty, oldest first, non-overlapping half-open
    /// `(created, deleted)` intervals with no `PENDING` epochs. Every lifetime
    /// must have a complete label-set entry at its create epoch;
    /// `label_versions` must be epoch-ascending and each entry must fall inside
    /// a structural lifetime or exactly on its delete boundary. A boundary
    /// entry is retained but structurally invisible. If one lifetime ends at
    /// the exact epoch at which the next begins, entries at that epoch retain
    /// their input order: any closed-life boundary states come first and the
    /// last entry is the required complete state for the new lifetime. Thus a
    /// boundary state cannot seed the later lifetime. Every later lifetime
    /// must likewise provide a complete set at its own create epoch.
    ///
    /// The identity must not already exist. Validation completes before any
    /// graph state is installed, so invalid input cannot partially replace an
    /// entity. This is a recovery-only seam: call it on an unpublished or
    /// otherwise quiescent store, then replay property logs with
    /// [`set_node_property_at_epoch`](Self::set_node_property_at_epoch).
    ///
    /// # Errors
    ///
    /// Returns an error for invalid/overlapping history, an existing or invalid
    /// ID, disabled write authority, or tiered-arena allocation failure.
    #[doc(hidden)]
    pub fn restore_node_history_exact(
        &self,
        id: NodeId,
        lifetimes: &[(EpochId, Option<EpochId>)],
        label_versions: &[(EpochId, Vec<ArcStr>)],
    ) -> Result<(), String> {
        let Some(_mutation) = self.pin_mutation() else {
            return Err("exact node-history restore requires write authority".into());
        };
        if !id.is_valid() {
            return Err("exact node-history restore requires a valid NodeId".into());
        }
        let structural_max = validate_restore_lifetimes("node", lifetimes)?;
        let label_max = validate_restore_label_versions(lifetimes, label_versions)?;
        let identity_reservation = NodeIdentityReservation::vacant(self, id)
            .ok_or_else(|| format!("node {id} already exists or is being published"))?;

        let label_count_at = |created: EpochId| {
            label_versions
                .iter()
                .rev()
                .find(|(epoch, _)| *epoch <= created)
                .map_or(0, |(_, labels)| labels.len())
        };

        #[cfg(not(feature = "tiered-storage"))]
        let restored_versions = {
            let mut chain = VersionChain::new();
            for &(created, deleted) in lifetimes {
                let mut record = NodeRecord::new(id, created);
                record.set_label_count(
                    u16::try_from(label_count_at(created))
                        .map_err(|_| "node label count exceeds u16::MAX")?,
                );
                chain.add_version(record, created, TransactionId::SYSTEM);
                if let Some(deleted) = deleted {
                    let marked = chain.mark_deleted(deleted, TransactionId::SYSTEM);
                    debug_assert!(marked, "just-restored node lifetime must be open");
                }
            }
            chain
        };

        #[cfg(feature = "tiered-storage")]
        let restored_versions = {
            let mut index = VersionIndex::new();
            for &(created, deleted) in lifetimes {
                let mut record = NodeRecord::new(id, created);
                record.set_label_count(
                    u16::try_from(label_count_at(created))
                        .map_err(|_| "node label count exceeds u16::MAX")?,
                );
                let arena = self
                    .arena_allocator
                    .arena_or_create(created)
                    .map_err(|error| format!("restore node {id} arena: {error}"))?;
                let (offset, _) = arena
                    .alloc_value_with_offset(record)
                    .map_err(|error| format!("restore node {id} record: {error}"))?;
                index.add_hot(HotVersionRef::new(
                    created,
                    created,
                    offset,
                    TransactionId::SYSTEM,
                ));
                if let Some(deleted) = deleted {
                    let marked = index.mark_deleted(deleted, TransactionId::SYSTEM);
                    debug_assert!(marked, "just-restored node lifetime must be open");
                }
            }
            index
        };

        // Resolve all label names only after the complete input has validated
        // and tiered allocations have succeeded.
        let mut restored_labels = VersionLog::new();
        for (epoch, labels) in label_versions {
            let ids: FxHashSet<u32> = labels
                .iter()
                .map(|label| self.get_or_create_label_id(label.as_str()))
                .collect();
            restored_labels.append(*epoch, ids);
        }
        let current_label_ids = restored_labels.latest().cloned().unwrap_or_default();
        let is_open = lifetimes
            .last()
            .is_some_and(|(_, deleted)| deleted.is_none());

        // Install the entity, temporal labels, and current label-index
        // projection under the documented entity -> indexes lock order.
        #[cfg(not(feature = "tiered-storage"))]
        {
            let mut nodes = self.nodes.write();
            if nodes.contains_key(&id) {
                return Err(format!("node {id} was concurrently restored"));
            }
            let mut label_index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            nodes.insert(id, restored_versions);
            node_labels.insert(id, restored_labels);
            if is_open {
                for label_id in current_label_ids {
                    if label_index.len() <= label_id as usize {
                        label_index.resize_with(label_id as usize + 1, FxHashMap::default);
                    }
                    label_index[label_id as usize].insert(id, ());
                }
                self.live_node_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            let mut nodes = self.node_versions.write();
            if nodes.contains_key(&id) {
                return Err(format!("node {id} was concurrently restored"));
            }
            let mut label_index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            nodes.insert(id, restored_versions);
            node_labels.insert(id, restored_labels);
            if is_open {
                for label_id in current_label_ids {
                    if label_index.len() <= label_id as usize {
                        label_index.resize_with(label_id as usize + 1, FxHashMap::default);
                    }
                    label_index[label_id as usize].insert(id, ());
                }
                self.live_node_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.next_node_id
            .fetch_max(id.as_u64().saturating_add(1), Ordering::SeqCst);
        self.sync_epoch(structural_max.max(label_max));
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        identity_reservation.commit();
        Ok(())
    }

    /// Restores one immutable-endpoint edge identity with exact structural history.
    ///
    /// `lifetimes` follows the same committed, oldest-first, non-overlapping
    /// contract as [`restore_node_history_exact`](Self::restore_node_history_exact).
    /// Both endpoint identities must already be restored (they may themselves
    /// be currently deleted). The edge identity must not exist. Current
    /// adjacency and live counters are derived from whether the final lifetime
    /// is open; closed identities remain in the soft-deleted adjacency so
    /// versioned/as-of traversal can still recover them.
    ///
    /// Call this on an unpublished/quiescent recovery target, then replay edge
    /// property logs with [`set_edge_property_at_epoch`](Self::set_edge_property_at_epoch).
    ///
    /// # Errors
    ///
    /// Returns an error for invalid/overlapping history, missing endpoints, an
    /// existing or invalid ID, disabled write authority, or tiered allocation
    /// failure. Invalid input is rejected before graph state is installed.
    #[doc(hidden)]
    pub fn restore_edge_history_exact(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        lifetimes: &[(EpochId, Option<EpochId>)],
    ) -> Result<(), String> {
        let Some(mutation) = self.pin_mutation() else {
            return Err("exact edge-history restore requires write authority".into());
        };
        let authority = Arc::clone(&self.transport_extract_authority.read());
        let identity_reservation = self.restore_edge_history_exact_inner(
            EdgeHistoryRestoreRequest {
                id,
                src,
                dst,
                edge_type,
                lifetimes,
                allow_missing_destination: false,
                identity_reservation: None,
                preserve_transport_nonce: false,
            },
            &mutation,
            &authority,
        )?;
        identity_reservation.commit();
        Ok(())
    }

    /// Restores one closed ordinary cross-shard history whose source is
    /// present but whose destination may be supplied only by a sibling shard.
    ///
    /// This narrow detached-import seam grants no transport receipt, mutation
    /// exception, or physical-purge authority. It accepts exactly one committed
    /// closed lifetime; open edges must use the receipt-bearing transport path,
    /// while ordinary exact restore continues to require both endpoints.
    /// Portable v10 validation is responsible for binding the omitted endpoint
    /// declaration before the engine calls this on an unpublished target.
    ///
    /// # Errors
    ///
    /// Returns an error unless the sole lifetime is committed closed, the
    /// source identity exists, IDs are valid/vacant, write authority is active,
    /// and structural installation succeeds.
    #[doc(hidden)]
    pub fn restore_closed_cross_shard_edge_history_exact(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        lifetimes: &[(EpochId, Option<EpochId>)],
    ) -> Result<(), String> {
        let [(created, Some(deleted))] = lifetimes else {
            return Err(
                "closed cross-shard restore requires exactly one closed structural lifetime".into(),
            );
        };
        if *created == EpochId::PENDING || *deleted == EpochId::PENDING || deleted <= created {
            return Err("closed cross-shard restore requires a committed positive lifetime".into());
        }
        self.with_exclusive_bulk_restore(|transition| {
            let mutation = self
                .pin_mutation()
                .expect("bulk-restore context carries the exact mutation proof");
            let identity_reservation = self.restore_edge_history_exact_inner(
                EdgeHistoryRestoreRequest {
                    id,
                    src,
                    dst,
                    edge_type,
                    lifetimes,
                    allow_missing_destination: true,
                    identity_reservation: None,
                    preserve_transport_nonce: false,
                },
                &mutation,
                transition.transport_authority(),
            )?;
            identity_reservation.commit();
            Ok(())
        })
        .ok_or_else(|| "closed cross-shard restore requires write authority".to_string())?
    }

    /// Restores one exact, vacant edge identity and returns move-only transport
    /// provenance bound to that newly installed history.
    ///
    /// This is the portable-import counterpart of
    /// [`Self::create_transport_edge_with_id`]. It shares the ordinary exact
    /// restore's final vacant-ID insertion, so a receipt is never issued for a
    /// pre-existing or concurrently restored identity. The receipt binds every
    /// original structural lifetime. Its source must already exist; its destination
    /// may be absent only because this transport-only path deliberately
    /// reconstructs an isolated shard for later framed cleanup or union.
    ///
    /// # Errors
    ///
    /// Returns the same validation/allocation errors as
    /// [`Self::restore_edge_history_exact`].
    #[doc(hidden)]
    pub fn restore_transport_edge_history_exact(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        lifetimes: &[(EpochId, Option<EpochId>)],
    ) -> Result<TransportEdgeReceipt, String> {
        self.restore_transport_edge_history_exact_with_destination_labels(
            id,
            src,
            dst,
            edge_type,
            lifetimes,
            &[],
        )
    }

    /// Restores one exact transport identity with its destination type witness.
    ///
    /// The witness is a complete, canonical label set captured from the
    /// destination at the source extract cut. It conveys no structural or
    /// mutation authority by itself; only the returned store-incarnation
    /// receipt can later attest it at commit.
    #[doc(hidden)]
    pub fn restore_transport_edge_history_exact_with_destination_labels(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        lifetimes: &[(EpochId, Option<EpochId>)],
        destination_labels: &[ArcStr],
    ) -> Result<TransportEdgeReceipt, String> {
        // Build every receipt-owned allocation before reserving provenance or
        // publishing graph structure. Once installation succeeds, returning
        // the receipt is allocation-free and cannot strand a live nonce.
        let receipt_edge_type: ArcStr = edge_type.into();
        let receipt_destination_labels = canonical_label_witness(destination_labels.to_vec());
        let receipt_lifetimes: Arc<[(EpochId, Option<EpochId>)]> = lifetimes.into();
        self.with_exclusive_bulk_restore(|transition| {
            let mutation = self
                .pin_mutation()
                .expect("bulk-restore context carries the exact mutation proof");
            let authority = Arc::clone(transition.transport_authority());
            let identity_reservation = EdgeIdentityReservation::vacant(self, id)
                .ok_or_else(|| format!("edge {id} already exists or is being published"))?;
            let nonce_reservation = TransportNonceReservation::new(&authority, id)
                .ok_or_else(|| format!("transport edge {id} already has live provenance"))?;
            self.transport_edges_may_be_unresolved
                .store(true, Ordering::Release);
            let identity_reservation = self.restore_edge_history_exact_inner(
                EdgeHistoryRestoreRequest {
                    id,
                    src,
                    dst,
                    edge_type,
                    lifetimes,
                    allow_missing_destination: true,
                    identity_reservation: Some(identity_reservation),
                    preserve_transport_nonce: true,
                },
                &mutation,
                &authority,
            )?;
            #[cfg(test)]
            if let Some(barrier) = self.transport_create_barrier.read().clone() {
                barrier.wait();
                barrier.wait();
            }
            let receipt = TransportEdgeReceipt {
                authority,
                id,
                src,
                dst,
                edge_type: receipt_edge_type,
                destination_labels: receipt_destination_labels,
                lifetimes: receipt_lifetimes,
                nonce: nonce_reservation.nonce,
            };
            identity_reservation.commit();
            let nonce = nonce_reservation.commit();
            debug_assert_eq!(receipt.nonce, nonce);
            Ok(receipt)
        })
        .ok_or_else(|| "transport edge restore requires write authority".to_string())?
    }

    fn restore_edge_history_exact_inner<'store>(
        &'store self,
        request: EdgeHistoryRestoreRequest<'_, 'store>,
        _mutation: &PinnedMutation<'_>,
        authority: &Arc<super::TransportExtractAuthority>,
    ) -> Result<EdgeIdentityReservation<'store>, String> {
        let EdgeHistoryRestoreRequest {
            id,
            src,
            dst,
            edge_type,
            lifetimes,
            allow_missing_destination,
            identity_reservation,
            preserve_transport_nonce,
        } = request;
        if !id.is_valid() || !src.is_valid() || !dst.is_valid() {
            return Err("exact edge-history restore requires valid IDs".into());
        }
        let structural_max = validate_restore_lifetimes("edge", lifetimes)?;

        let identity_reservation = match identity_reservation {
            Some(reservation) => reservation,
            None => EdgeIdentityReservation::vacant(self, id)
                .ok_or_else(|| format!("edge {id} already exists or is being published"))?,
        };
        // Preflight endpoints without retaining an entity guard across type
        // preparation or tiered arena allocation. They are reacquired and
        // revalidated for the final publication cut below.
        #[cfg(not(feature = "tiered-storage"))]
        let endpoints_present = {
            let endpoints = self.nodes.read();
            endpoints.contains_key(&src)
                && (allow_missing_destination || endpoints.contains_key(&dst))
        };
        #[cfg(feature = "tiered-storage")]
        let endpoints_present = {
            let endpoints = self.node_versions.read();
            endpoints.contains_key(&src)
                && (allow_missing_destination || endpoints.contains_key(&dst))
        };
        if !endpoints_present {
            return Err(format!("edge {id} endpoints must be restored first"));
        }
        #[cfg(test)]
        self.pause_edge_publication_for_test();

        let prepared_type = self
            .prepare_edge_type(edge_type)
            .map_err(|error| format!("prepare edge type {edge_type}: {error}"))?;
        let type_id = prepared_type.id();
        #[cfg(not(feature = "tiered-storage"))]
        let restored_versions = {
            let mut chain = VersionChain::new();
            for &(created, deleted) in lifetimes {
                chain.add_version(
                    EdgeRecord::new(id, src, dst, type_id, created),
                    created,
                    TransactionId::SYSTEM,
                );
                if let Some(deleted) = deleted {
                    let marked = chain.mark_deleted(deleted, TransactionId::SYSTEM);
                    debug_assert!(marked, "just-restored edge lifetime must be open");
                }
            }
            chain
        };
        #[cfg(feature = "tiered-storage")]
        let restored_versions = {
            let mut index = VersionIndex::new();
            for &(created, deleted) in lifetimes {
                let arena = self
                    .arena_allocator
                    .arena_or_create(created)
                    .map_err(|error| format!("restore edge {id} arena: {error}"))?;
                let (offset, _) = arena
                    .alloc_value_with_offset(EdgeRecord::new(id, src, dst, type_id, created))
                    .map_err(|error| format!("restore edge {id} record: {error}"))?;
                index.add_hot(HotVersionRef::new(
                    created,
                    created,
                    offset,
                    TransactionId::SYSTEM,
                ));
                if let Some(deleted) = deleted {
                    let marked = index.mark_deleted(deleted, TransactionId::SYSTEM);
                    debug_assert!(marked, "just-restored edge lifetime must be open");
                }
            }
            index
        };

        #[cfg(not(feature = "tiered-storage"))]
        let endpoint_identities = self.nodes.read();
        #[cfg(feature = "tiered-storage")]
        let endpoint_identities = self.node_versions.read();
        if !endpoint_identities.contains_key(&src)
            || (!allow_missing_destination && !endpoint_identities.contains_key(&dst))
        {
            return Err(format!("edge {id} endpoints changed during exact restore"));
        }
        #[cfg(not(feature = "tiered-storage"))]
        let mut edge_identities = self.edges.write();
        #[cfg(feature = "tiered-storage")]
        let mut edge_identities = self.edge_versions.write();
        edge_identities
            .try_reserve(1)
            .map_err(|_| format!("restore edge {id} structural map capacity"))?;

        // Everything that can return an allocation error is complete. Commit
        // the prepared catalog identity before revoking stale provenance, so
        // a failed preparation can never invalidate a still-live receipt.
        let committed_type_id = prepared_type.commit(self);
        debug_assert_eq!(committed_type_id, type_id);
        if !preserve_transport_nonce {
            authority.revoke_id(id);
        }
        if allow_missing_destination {
            self.next_node_id.fetch_max(
                src.as_u64().max(dst.as_u64()).saturating_add(1),
                Ordering::SeqCst,
            );
        }

        let is_open = lifetimes
            .last()
            .is_some_and(|(_, deleted)| deleted.is_none());
        #[cfg(not(feature = "tiered-storage"))]
        {
            debug_assert!(
                !edge_identities.contains_key(&id),
                "reserved edge identity is vacant"
            );
            edge_identities.insert(id, restored_versions);
            self.forward_adj.add_edge(src, dst, id);
            if let Some(backward) = &self.backward_adj {
                backward.add_edge(dst, src, id);
            }
            if is_open {
                self.live_edge_count.fetch_add(1, Ordering::Relaxed);
                self.increment_edge_type_count(type_id);
            } else {
                self.forward_adj.mark_deleted(src, id);
                if let Some(backward) = &self.backward_adj {
                    backward.mark_deleted(dst, id);
                }
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            debug_assert!(
                !edge_identities.contains_key(&id),
                "reserved edge identity is vacant"
            );
            edge_identities.insert(id, restored_versions);
            self.forward_adj.add_edge(src, dst, id);
            if let Some(backward) = &self.backward_adj {
                backward.add_edge(dst, src, id);
            }
            if is_open {
                self.live_edge_count.fetch_add(1, Ordering::Relaxed);
                self.increment_edge_type_count(type_id);
            } else {
                self.forward_adj.mark_deleted(src, id);
                if let Some(backward) = &self.backward_adj {
                    backward.mark_deleted(dst, id);
                }
            }
        }

        self.sync_epoch(structural_max);
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        self.next_edge_id
            .fetch_max(id.as_u64().saturating_add(1), Ordering::SeqCst);
        drop(edge_identities);
        drop(endpoint_identities);
        Ok(identity_reservation)
    }

    /// Creates a node with a specific ID during recovery.
    ///
    /// This is used for WAL recovery to restore nodes with their original IDs.
    /// The caller must ensure IDs don't conflict with existing nodes.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot allocate space
    /// (only possible with the `tiered-storage` feature).
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) -> Result<(), AllocError> {
        if !id.is_valid() {
            return Ok(());
        }
        let Some(_mutation) = self.pin_mutation() else {
            return Ok(());
        };
        let Some(identity_reservation) = NodeIdentityReservation::vacant(self, id) else {
            return Ok(());
        };
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        self.register_node_labels(id, labels, epoch);

        // Create version chain with initial version (using SYSTEM tx for recovery)
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        let prior = self.nodes.write().insert(id, chain);
        debug_assert!(prior.is_none(), "reserved node identity is vacant");
        self.live_node_count.fetch_add(1, Ordering::Relaxed);

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        identity_reservation.commit();
        Ok(())
    }

    /// Publishes one exact compact-base node identity into this same-
    /// incarnation overlay.
    ///
    /// The crate-private Layered caller supplies the current compact row while
    /// retaining its generation cut. This seam rejects occupied identities,
    /// preserves the committed creation origin, and never routes a denied or
    /// colliding hydration into later property replay.
    #[cfg(all(not(feature = "tiered-storage"), feature = "compact-store"))]
    pub(crate) fn promote_same_incarnation_node_with_id(
        &self,
        id: NodeId,
        lifetimes: &[(EpochId, Option<EpochId>)],
        label_versions: &[(EpochId, Vec<ArcStr>)],
    ) -> Result<Option<SameIncarnationNodePromotion<'_>>, AllocError> {
        if !id.is_valid() {
            return Ok(None);
        }
        let Ok(structural_max) = validate_restore_lifetimes("node", lifetimes) else {
            return Ok(None);
        };
        let Ok(label_max) = validate_restore_label_versions(lifetimes, label_versions) else {
            return Ok(None);
        };
        if structural_max.max(label_max) > self.current_epoch() {
            return Ok(None);
        }
        let Some(mutation) = self.pin_mutation() else {
            return Ok(None);
        };
        let Some(identity_reservation) = NodeIdentityReservation::vacant(self, id) else {
            return Ok(None);
        };
        let prepared_labels = self.prepare_node_labels(label_versions)?;
        let label_count_at = |created: EpochId| {
            label_versions
                .iter()
                .rev()
                .find(|(epoch, _)| *epoch <= created)
                .map_or(0, |(_, labels)| labels.len())
        };
        let mut restored_versions = VersionChain::new();
        for &(created, deleted) in lifetimes {
            let mut record = NodeRecord::new(id, created);
            record.set_label_count(u16::try_from(label_count_at(created)).unwrap_or(u16::MAX));
            restored_versions.add_version(record, created, TransactionId::SYSTEM);
            if let Some(deleted) = deleted {
                let marked = restored_versions.mark_deleted(deleted, TransactionId::SYSTEM);
                debug_assert!(marked, "just-promoted node lifetime is open");
            }
        }
        let mut restored_labels = VersionLog::new();
        for (epoch, labels) in label_versions {
            let mut label_ids = FxHashSet::default();
            for label in labels {
                let Some(label_id) = prepared_labels.id(label.as_str()) else {
                    return Ok(None);
                };
                label_ids.insert(label_id);
            }
            restored_labels.append(*epoch, label_ids);
        }
        let current_label_ids = restored_labels.latest().cloned().unwrap_or_default();
        let is_open = lifetimes
            .last()
            .is_some_and(|(_, deleted)| deleted.is_none());
        let mut promotion = SameIncarnationNodePromotion {
            store: self,
            id,
            structural_published: false,
            live_counted: false,
            identity_reservation: Some(identity_reservation),
            prepared_labels: Some(prepared_labels),
            mutation: Some(mutation),
        };
        let mut nodes = self.nodes.write();
        let mut label_index = self.label_index.write();
        let mut node_labels = self.node_labels.write();
        nodes
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        node_labels
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        if is_open {
            for &label_id in &current_label_ids {
                let Some(members) = label_index.get_mut(label_id as usize) else {
                    return Ok(None);
                };
                members
                    .try_reserve(1)
                    .map_err(|_| AllocError::InsufficientSpace)?;
            }
        }
        let prior = nodes.insert(id, restored_versions);
        debug_assert!(prior.is_none(), "reserved node identity is vacant");
        promotion.structural_published = true;
        let prior = node_labels.insert(id, restored_labels);
        debug_assert!(prior.is_none(), "reserved node labels are vacant");
        if is_open {
            for label_id in current_label_ids {
                label_index[label_id as usize].insert(id, ());
            }
            self.live_node_count.fetch_add(1, Ordering::Relaxed);
            promotion.live_counted = true;
        }
        drop(node_labels);
        drop(label_index);
        drop(nodes);
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        Ok(Some(promotion))
    }

    /// Creates a node with a specific ID during recovery.
    /// (Tiered storage version)
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot create an epoch
    /// or allocate space for the node record.
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) -> Result<(), AllocError> {
        if !id.is_valid() {
            return Ok(());
        }
        let Some(_mutation) = self.pin_mutation() else {
            return Ok(());
        };
        let Some(identity_reservation) = NodeIdentityReservation::vacant(self, id) else {
            return Ok(());
        };
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch)?;
        let (offset, _stored) = arena.alloc_value_with_offset(record)?;
        // The mutation pin keeps the arena alive through publication. Release
        // its registry guard before taking version locks: readers acquire them
        // in the opposite order, and a queued arena writer blocks new readers.
        drop(arena);

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
        self.register_node_labels(id, labels, epoch);
        let prior = self
            .node_versions
            .write()
            .insert(id, VersionIndex::with_initial(hot_ref));
        debug_assert!(prior.is_none(), "reserved node identity is vacant");
        self.live_node_count.fetch_add(1, Ordering::Relaxed);

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        identity_reservation.commit();
        Ok(())
    }

    /// Tiered-storage form of
    /// [`Self::promote_same_incarnation_node_with_id`].
    #[cfg(all(feature = "tiered-storage", feature = "compact-store"))]
    pub(crate) fn promote_same_incarnation_node_with_id(
        &self,
        id: NodeId,
        lifetimes: &[(EpochId, Option<EpochId>)],
        label_versions: &[(EpochId, Vec<ArcStr>)],
    ) -> Result<Option<SameIncarnationNodePromotion<'_>>, AllocError> {
        if !id.is_valid() {
            return Ok(None);
        }
        let Ok(structural_max) = validate_restore_lifetimes("node", lifetimes) else {
            return Ok(None);
        };
        let Ok(label_max) = validate_restore_label_versions(lifetimes, label_versions) else {
            return Ok(None);
        };
        if structural_max.max(label_max) > self.current_epoch() {
            return Ok(None);
        }
        let Some(mutation) = self.pin_mutation() else {
            return Ok(None);
        };
        let Some(identity_reservation) = NodeIdentityReservation::vacant(self, id) else {
            return Ok(None);
        };
        let prepared_labels = self.prepare_node_labels(label_versions)?;
        let label_count_at = |created: EpochId| {
            label_versions
                .iter()
                .rev()
                .find(|(epoch, _)| *epoch <= created)
                .map_or(0, |(_, labels)| labels.len())
        };
        let mut restored_versions = VersionIndex::new();
        for &(created, deleted) in lifetimes {
            let mut record = NodeRecord::new(id, created);
            record.set_label_count(u16::try_from(label_count_at(created)).unwrap_or(u16::MAX));
            let arena = self.arena_allocator.arena_or_create(created)?;
            let (offset, _stored) = arena.alloc_value_with_offset(record)?;
            restored_versions.add_hot(HotVersionRef::new(
                created,
                created,
                offset,
                TransactionId::SYSTEM,
            ));
            if let Some(deleted) = deleted {
                let marked = restored_versions.mark_deleted(deleted, TransactionId::SYSTEM);
                debug_assert!(marked, "just-promoted node lifetime is open");
            }
        }
        let mut restored_labels = VersionLog::new();
        for (epoch, labels) in label_versions {
            let mut label_ids = FxHashSet::default();
            for label in labels {
                let Some(label_id) = prepared_labels.id(label.as_str()) else {
                    return Ok(None);
                };
                label_ids.insert(label_id);
            }
            restored_labels.append(*epoch, label_ids);
        }
        let current_label_ids = restored_labels.latest().cloned().unwrap_or_default();
        let is_open = lifetimes
            .last()
            .is_some_and(|(_, deleted)| deleted.is_none());
        let mut promotion = SameIncarnationNodePromotion {
            store: self,
            id,
            structural_published: false,
            live_counted: false,
            identity_reservation: Some(identity_reservation),
            prepared_labels: Some(prepared_labels),
            mutation: Some(mutation),
        };
        let mut nodes = self.node_versions.write();
        let mut label_index = self.label_index.write();
        let mut node_labels = self.node_labels.write();
        nodes
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        node_labels
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        if is_open {
            for &label_id in &current_label_ids {
                let Some(members) = label_index.get_mut(label_id as usize) else {
                    return Ok(None);
                };
                members
                    .try_reserve(1)
                    .map_err(|_| AllocError::InsufficientSpace)?;
            }
        }
        let prior = nodes.insert(id, restored_versions);
        debug_assert!(prior.is_none(), "reserved node identity is vacant");
        promotion.structural_published = true;
        let prior = node_labels.insert(id, restored_labels);
        debug_assert!(prior.is_none(), "reserved node labels are vacant");
        if is_open {
            for label_id in current_label_ids {
                label_index[label_id as usize].insert(id, ());
            }
            self.live_node_count.fetch_add(1, Ordering::Relaxed);
            promotion.live_counted = true;
        }
        drop(node_labels);
        drop(label_index);
        drop(nodes);
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        Ok(Some(promotion))
    }

    /// Inserts a fresh edge with a specific ID during recovery.
    ///
    /// The ID must be vacant, and both latest endpoint lifetimes must be live
    /// and committed. Replay adapters, not this insertion primitive, own exact
    /// duplicate handling. Vacant IDs below the allocator high-water are valid.
    ///
    /// # Errors
    ///
    /// Returns a structured error for invalid or occupied IDs, denied write
    /// authority, absent or non-committed-live endpoints, or allocation failure.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), Error> {
        let successor = fresh_edge_successor(id, src, dst)?;
        let _mutation = self.pin_mutation().ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "fresh edge insertion requires the current store write authority".into(),
            ))
        })?;
        let authority = self.transport_extract_authority.read();
        let identity_reservation = EdgeIdentityReservation::vacant(self, id).ok_or_else(|| {
            Error::InvalidValue(format!("fresh edge identity {id} is occupied or reserved"))
        })?;
        let epoch = self.current_epoch();
        if epoch == EpochId::PENDING {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "fresh edge insertion requires a committed store epoch".into(),
            )));
        }
        let qualify_endpoints = |nodes: &FxHashMap<NodeId, VersionChain<NodeRecord>>| {
            for endpoint in [src, dst] {
                let (info, record) = nodes
                    .get(&endpoint)
                    .and_then(|chain| chain.history().next())
                    .ok_or(Error::NodeNotFound(endpoint))?;
                if info.created_epoch == EpochId::PENDING
                    || info.deleted_epoch == Some(EpochId::PENDING)
                {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        format!("fresh edge endpoint {endpoint} has an in-flight lifetime"),
                    )));
                }
                if info.created_epoch > epoch || info.deleted_epoch.is_some() || record.is_deleted()
                {
                    return Err(Error::NodeNotFound(endpoint));
                }
            }
            Ok(())
        };
        {
            let nodes = self.nodes.read();
            qualify_endpoints(&nodes)?;
        }
        #[cfg(test)]
        self.pause_edge_publication_for_test();
        let prepared_type = self.prepare_edge_type(edge_type)?;
        let type_id = prepared_type.id();
        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        let nodes = self.nodes.read();
        // Retain both endpoint lifetimes through structural and adjacency
        // publication; deletion cannot pass this final read-side cut.
        qualify_endpoints(&nodes)?;
        let mut edges = self.edges.write();
        edges
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let committed_type_id = prepared_type.commit(self);
        debug_assert_eq!(committed_type_id, type_id);
        authority.revoke_id(id);
        debug_assert!(!edges.contains_key(&id), "reserved edge identity is vacant");
        edges.insert(id, chain);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);
        drop(edges);
        drop(nodes);

        // Update next_edge_id if necessary
        self.next_edge_id.fetch_max(successor, Ordering::SeqCst);
        identity_reservation.commit();
        Ok(())
    }

    /// Publishes an exact compact-base identity into this same-incarnation
    /// overlay without revoking any already-live transport nonce.
    ///
    /// This crate-private seam is representation-only. It never reserves or
    /// inserts transport authority, so promoting an ordinary edge cannot gain
    /// purge capability. The trusted `LayeredStore` caller must hold its
    /// generation cut and must have resolved `id`, endpoints, and type from the
    /// current compact generation. A fresh overlay incarnation has a distinct
    /// authority map and therefore cannot resurrect an old receipt.
    #[cfg(all(not(feature = "tiered-storage"), feature = "compact-store"))]
    pub(crate) fn promote_same_incarnation_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        lifetimes: &[(EpochId, Option<EpochId>)],
    ) -> Result<Option<SameIncarnationEdgePromotion<'_>>, AllocError> {
        if !id.is_valid()
            || !src.is_valid()
            || !dst.is_valid()
            || validate_restore_lifetimes("promoted edge", lifetimes)
                .map_or(true, |frontier| frontier > self.current_epoch())
            || lifetimes
                .last()
                .is_none_or(|(_, deleted)| deleted.is_some())
        {
            return Ok(None);
        }
        let Some(mutation) = self.pin_mutation() else {
            return Ok(None);
        };
        let _authority_incarnation = self.transport_extract_authority.read();
        let Some(identity_reservation) = EdgeIdentityReservation::vacant(self, id) else {
            return Ok(None);
        };
        if !self.nodes.read().contains_key(&src) || !self.nodes.read().contains_key(&dst) {
            return Ok(None);
        }
        let prepared_type = self.prepare_edge_type_for_promotion(edge_type)?;
        let type_id = prepared_type.id();
        let mut chain = VersionChain::new();
        for &(created, deleted) in lifetimes {
            chain.add_version(
                EdgeRecord::new(id, src, dst, type_id, created),
                created,
                TransactionId::SYSTEM,
            );
            if let Some(deleted) = deleted {
                chain.mark_deleted(deleted, TransactionId::SYSTEM);
            }
        }

        // Pin both endpoint identities across structural and adjacency
        // publication after all fallible preparation is complete.
        let nodes = self.nodes.read();
        if !nodes.contains_key(&src) || !nodes.contains_key(&dst) {
            return Ok(None);
        }
        let mut edges = self.edges.write();
        edges
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let Some(forward_adjacency) = self.forward_adj.begin_unpublished_edge(src, id)? else {
            return Ok(None);
        };
        let backward_adjacency = if let Some(backward) = &self.backward_adj {
            let Some(publication) = backward.begin_unpublished_edge(dst, id)? else {
                return Ok(None);
            };
            Some(publication)
        } else {
            None
        };
        let mut promotion = SameIncarnationEdgePromotion {
            store: self,
            removed: RemovedEdgeIdentity {
                id,
                src,
                dst,
                type_id,
            },
            structural_published: false,
            live_counted: false,
            type_counted: false,
            identity_reservation: Some(identity_reservation),
            prepared_type: Some(prepared_type),
            forward_adjacency: Some(forward_adjacency),
            backward_adjacency,
            mutation: Some(mutation),
        };
        debug_assert!(!edges.contains_key(&id), "reserved edge identity is vacant");
        edges.insert(id, chain);
        promotion.structural_published = true;
        if let Some(forward) = promotion.forward_adjacency.as_mut() {
            forward.publish(dst);
        }
        if let Some(backward) = promotion.backward_adjacency.as_mut() {
            backward.publish(src);
        }
        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        promotion.live_counted = true;
        self.increment_edge_type_count(type_id);
        promotion.type_counted = true;
        drop(edges);
        drop(nodes);
        Ok(Some(promotion))
    }

    /// Tiered-storage form of the fresh recovery insertion described by
    /// [`Self::create_edge_with_id`]. Duplicate replay is owned by its caller.
    ///
    /// # Errors
    ///
    /// Returns a structured error for invalid or occupied IDs, denied write
    /// authority, absent or non-committed-live endpoints, or allocation failure.
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), Error> {
        let successor = fresh_edge_successor(id, src, dst)?;
        let _mutation = self.pin_mutation().ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "fresh edge insertion requires the current store write authority".into(),
            ))
        })?;
        let authority = self.transport_extract_authority.read();
        let identity_reservation = EdgeIdentityReservation::vacant(self, id).ok_or_else(|| {
            Error::InvalidValue(format!("fresh edge identity {id} is occupied or reserved"))
        })?;
        let epoch = self.current_epoch();
        if epoch == EpochId::PENDING {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "fresh edge insertion requires a committed store epoch".into(),
            )));
        }
        let qualify_endpoints = |nodes: &FxHashMap<NodeId, VersionIndex>| {
            for endpoint in [src, dst] {
                let version = nodes
                    .get(&endpoint)
                    .and_then(VersionIndex::latest)
                    .ok_or(Error::NodeNotFound(endpoint))?;
                if version.epoch() == EpochId::PENDING
                    || version.deleted_epoch() == Some(EpochId::PENDING)
                {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        format!("fresh edge endpoint {endpoint} has an in-flight lifetime"),
                    )));
                }
                if version.epoch() > epoch || version.deleted_epoch().is_some() {
                    return Err(Error::NodeNotFound(endpoint));
                }
            }
            Ok(())
        };
        {
            let nodes = self.node_versions.read();
            qualify_endpoints(&nodes)?;
        }
        #[cfg(test)]
        self.pause_edge_publication_for_test();
        let prepared_type = self.prepare_edge_type(edge_type)?;
        let type_id = prepared_type.id();

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch)?;
        let (offset, _stored) = arena.alloc_value_with_offset(record)?;
        // The mutation pin keeps the arena alive through publication. Release
        // its registry guard before taking version locks: readers acquire them
        // in the opposite order, and a queued arena writer blocks new readers.
        drop(arena);

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
        let nodes = self.node_versions.read();
        // Latest VersionRef metadata qualifies cold and hot lifetimes without
        // decoding records or taking arena guards under this structural cut.
        qualify_endpoints(&nodes)?;
        let mut versions = self.edge_versions.write();
        versions
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let committed_type_id = prepared_type.commit(self);
        debug_assert_eq!(committed_type_id, type_id);
        authority.revoke_id(id);
        debug_assert!(
            !versions.contains_key(&id),
            "reserved edge identity is vacant"
        );
        versions.insert(id, VersionIndex::with_initial(hot_ref));

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);
        drop(versions);
        drop(nodes);

        // Update next_edge_id if necessary
        self.next_edge_id.fetch_max(successor, Ordering::SeqCst);
        identity_reservation.commit();
        Ok(())
    }

    /// Tiered-storage form of
    /// [`Self::promote_same_incarnation_edge_with_id`].
    #[cfg(all(feature = "tiered-storage", feature = "compact-store"))]
    pub(crate) fn promote_same_incarnation_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        lifetimes: &[(EpochId, Option<EpochId>)],
    ) -> Result<Option<SameIncarnationEdgePromotion<'_>>, AllocError> {
        if !id.is_valid()
            || !src.is_valid()
            || !dst.is_valid()
            || validate_restore_lifetimes("promoted edge", lifetimes)
                .map_or(true, |frontier| frontier > self.current_epoch())
            || lifetimes
                .last()
                .is_none_or(|(_, deleted)| deleted.is_some())
        {
            return Ok(None);
        }
        let Some(mutation) = self.pin_mutation() else {
            return Ok(None);
        };
        let _authority_incarnation = self.transport_extract_authority.read();
        let Some(identity_reservation) = EdgeIdentityReservation::vacant(self, id) else {
            return Ok(None);
        };
        {
            let nodes = self.node_versions.read();
            if !nodes.contains_key(&src) || !nodes.contains_key(&dst) {
                return Ok(None);
            }
        }

        // No node-version guard is retained across fallible arena work.
        let prepared_type = self.prepare_edge_type_for_promotion(edge_type)?;
        let type_id = prepared_type.id();
        let mut index = VersionIndex::new();
        for &(created, deleted) in lifetimes {
            let record = EdgeRecord::new(id, src, dst, type_id, created);
            let arena = self.arena_allocator.arena_or_create(created)?;
            let (offset, _stored) = arena.alloc_value_with_offset(record)?;
            index.add_hot(HotVersionRef::new(
                created,
                created,
                offset,
                TransactionId::SYSTEM,
            ));
            if let Some(deleted) = deleted {
                index.mark_deleted(deleted, TransactionId::SYSTEM);
            }
        }

        // Revalidate and pin both endpoint identities through the exact
        // structural and adjacency publication cut.
        let nodes = self.node_versions.read();
        if !nodes.contains_key(&src) || !nodes.contains_key(&dst) {
            return Ok(None);
        }
        let mut versions = self.edge_versions.write();
        versions
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let Some(forward_adjacency) = self.forward_adj.begin_unpublished_edge(src, id)? else {
            return Ok(None);
        };
        let backward_adjacency = if let Some(backward) = &self.backward_adj {
            let Some(publication) = backward.begin_unpublished_edge(dst, id)? else {
                return Ok(None);
            };
            Some(publication)
        } else {
            None
        };
        let mut promotion = SameIncarnationEdgePromotion {
            store: self,
            removed: RemovedEdgeIdentity {
                id,
                src,
                dst,
                type_id,
            },
            structural_published: false,
            live_counted: false,
            type_counted: false,
            identity_reservation: Some(identity_reservation),
            prepared_type: Some(prepared_type),
            forward_adjacency: Some(forward_adjacency),
            backward_adjacency,
            mutation: Some(mutation),
        };
        debug_assert!(
            !versions.contains_key(&id),
            "reserved edge identity is vacant"
        );
        versions.insert(id, index);
        promotion.structural_published = true;
        if let Some(forward) = promotion.forward_adjacency.as_mut() {
            forward.publish(dst);
        }
        if let Some(backward) = promotion.backward_adjacency.as_mut() {
            backward.publish(src);
        }
        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        promotion.live_counted = true;
        self.increment_edge_type_count(type_id);
        promotion.type_counted = true;
        drop(versions);
        drop(nodes);
        Ok(Some(promotion))
    }

    /// Atomically creates one edge owned by an ephemeral subgraph transport.
    ///
    /// Success returns a move-only receipt bound to this store and the exact
    /// new identity. If `id` was already present, no graph structure is changed
    /// and `Ok(None)` is returned. Thus this API can never confer transport
    /// provenance on an ordinary, recovered, or concurrently-created edge.
    /// Dropping the receipt permanently gives up physical-purge authority; the
    /// edge remains an ordinary temporal identity.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if tiered storage cannot allocate the edge.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_transport_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<Option<TransportEdgeReceipt>, AllocError> {
        if !id.is_valid() || !src.is_valid() || !dst.is_valid() {
            return Ok(None);
        }
        let Some(_mutation) = self.pin_mutation() else {
            return Ok(None);
        };

        let receipt_edge_type: ArcStr = edge_type.into();
        let receipt_destination_labels = canonical_label_witness(
            self.get_node(dst)
                .map(|node| node.labels.into_iter().collect())
                .unwrap_or_default(),
        );
        // Hold the incarnation read side through identity reservation,
        // structural publication, and receipt minting. Full replacement takes
        // the write side and therefore cannot interleave.
        let authority = self.transport_extract_authority.read();
        let epoch = self.current_epoch();
        let receipt_lifetimes: Arc<[(EpochId, Option<EpochId>)]> = [(epoch, None)].into();
        let Some(identity_reservation) = EdgeIdentityReservation::vacant(self, id) else {
            return Ok(None);
        };
        {
            let source_identities = self.nodes.read();
            if !source_identities.contains_key(&src) {
                return Ok(None);
            }
        }
        #[cfg(test)]
        self.pause_edge_publication_for_test();
        let prepared_type = self.prepare_edge_type(edge_type)?;
        let type_id = prepared_type.id();
        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        let source_identities = self.nodes.read();
        if !source_identities.contains_key(&src) {
            return Ok(None);
        }
        let Some(nonce_reservation) = TransportNonceReservation::new(&authority, id) else {
            return Ok(None);
        };
        // Publish the slow-path discriminator before the adjacency row can
        // become visible. It remains sticky for this store incarnation so
        // receipt revocation cannot race query endpoint filtering.
        self.transport_edges_may_be_unresolved
            .store(true, Ordering::Release);
        let mut edges = self.edges.write();
        edges
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let committed_type_id = prepared_type.commit(self);
        debug_assert_eq!(committed_type_id, type_id);
        self.next_node_id.fetch_max(
            src.as_u64().max(dst.as_u64()).saturating_add(1),
            Ordering::SeqCst,
        );

        // The out-of-map reservation spans the complete structural, adjacency,
        // and counter publication; validators reject reserved identities.
        debug_assert!(!edges.contains_key(&id), "reserved edge identity is vacant");
        edges.insert(id, chain);
        self.forward_adj.add_edge(src, dst, id);
        if let Some(backward) = &self.backward_adj {
            backward.add_edge(dst, src, id);
        }
        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);
        let id_value = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                (id_value >= current).then_some(id_value + 1)
            });
        drop(edges);

        #[cfg(test)]
        if let Some(barrier) = self.transport_create_barrier.read().clone() {
            barrier.wait();
            barrier.wait();
        }

        let receipt = TransportEdgeReceipt {
            authority: Arc::clone(&authority),
            id,
            src,
            dst,
            edge_type: receipt_edge_type,
            destination_labels: receipt_destination_labels,
            lifetimes: receipt_lifetimes,
            nonce: nonce_reservation.nonce,
        };
        identity_reservation.commit();
        let nonce = nonce_reservation.commit();
        debug_assert_eq!(receipt.nonce, nonce);
        drop(source_identities);
        Ok(Some(receipt))
    }

    /// Tiered-storage form of
    /// [`create_transport_edge_with_id`](Self::create_transport_edge_with_id).
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_transport_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<Option<TransportEdgeReceipt>, AllocError> {
        if !id.is_valid() || !src.is_valid() || !dst.is_valid() {
            return Ok(None);
        }
        let Some(_mutation) = self.pin_mutation() else {
            return Ok(None);
        };

        let receipt_edge_type: ArcStr = edge_type.into();
        let receipt_destination_labels = canonical_label_witness(
            self.get_node(dst)
                .map(|node| node.labels.into_iter().collect())
                .unwrap_or_default(),
        );
        // Hold the incarnation read side through identity reservation,
        // structural publication, and receipt minting. Full replacement takes
        // the write side and therefore cannot interleave.
        let authority = self.transport_extract_authority.read();

        // Reserve outside the version map so readers and GC never observe an
        // empty index. Every exact/generated creator shares this registry.
        let Some(identity_reservation) = EdgeIdentityReservation::vacant(self, id) else {
            return Ok(None);
        };
        {
            let source_identities = self.node_versions.read();
            if !source_identities.contains_key(&src) {
                return Ok(None);
            }
        }
        #[cfg(test)]
        self.pause_edge_publication_for_test();

        let epoch = self.current_epoch();
        let receipt_lifetimes: Arc<[(EpochId, Option<EpochId>)]> = [(epoch, None)].into();
        let prepared_type = self.prepare_edge_type(edge_type)?;
        let type_id = prepared_type.id();
        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        let arena = self.arena_allocator.arena_or_create(epoch)?;
        let (offset, _stored) = arena.alloc_value_with_offset(record)?;
        // The mutation pin keeps the arena alive through publication. Release
        // its registry guard before taking version locks: readers acquire them
        // in the opposite order, and a queued arena writer blocks new readers.
        drop(arena);
        let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
        let source_identities = self.node_versions.read();
        if !source_identities.contains_key(&src) {
            return Ok(None);
        }
        let Some(nonce_reservation) = TransportNonceReservation::new(&authority, id) else {
            return Ok(None);
        };
        self.transport_edges_may_be_unresolved
            .store(true, Ordering::Release);
        let mut versions = self.edge_versions.write();
        versions
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let committed_type_id = prepared_type.commit(self);
        debug_assert_eq!(committed_type_id, type_id);
        self.next_node_id.fetch_max(
            src.as_u64().max(dst.as_u64()).saturating_add(1),
            Ordering::SeqCst,
        );

        debug_assert!(
            !versions.contains_key(&id),
            "reserved edge identity is vacant"
        );
        versions.insert(id, VersionIndex::with_initial(hot_ref));
        self.forward_adj.add_edge(src, dst, id);
        if let Some(backward) = &self.backward_adj {
            backward.add_edge(dst, src, id);
        }
        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);
        drop(versions);
        let id_value = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                (id_value >= current).then_some(id_value + 1)
            });
        #[cfg(test)]
        if let Some(barrier) = self.transport_create_barrier.read().clone() {
            barrier.wait();
            barrier.wait();
        }

        let receipt = TransportEdgeReceipt {
            authority: Arc::clone(&authority),
            id,
            src,
            dst,
            edge_type: receipt_edge_type,
            destination_labels: receipt_destination_labels,
            lifetimes: receipt_lifetimes,
            nonce: nonce_reservation.nonce,
        };
        identity_reservation.commit();
        let nonce = nonce_reservation.commit();
        debug_assert_eq!(receipt.nonce, nonce);
        drop(source_identities);
        Ok(Some(receipt))
    }

    /// Sets the current epoch during recovery.
    #[doc(hidden)]
    pub fn set_epoch(&self, epoch: EpochId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if epoch == EpochId::PENDING {
            return;
        }
        self.current_epoch.store(epoch.as_u64(), Ordering::SeqCst);
    }
}
