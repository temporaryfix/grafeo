//! Allocation-complete publication into an externally retained pristine store.

use super::{
    ExclusiveBulkRestoreContext, LpgStore, NAMED_GRAPH_TOPOLOGY_GATE, PinnedLpgTransition,
};
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, GraphPath, MAX_GRAPH_PATH_COMPONENTS};
use grafeo_common::utils::error::{Error, Result, StorageError};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// An owned, fully qualified live data image and its exact publication owners.
///
/// The root owner is the existing target Arc. Detached child owners inherit its
/// seal before exposure, and the private candidate retains displaced data after the
/// single publication attempt. Keep this workspace outside every final guard.
pub struct LpgReplacementWorkspace {
    target: Arc<LpgStore>,
    candidate: LpgStore,
    graphs: Vec<(GraphPath, Arc<LpgStore>)>,
    topology: Vec<Vec<(String, Arc<LpgStore>)>>,
    scope: u64,
    attempted: bool,
    validated: bool,
    image: Vec<u8>,
    rechecked_image: Option<Vec<u8>>,
}

impl LpgReplacementWorkspace {
    /// Qualifies an exclusively owned committed image for this exact target.
    ///
    /// # Errors
    /// Rejects shared candidates, incompatible backing and
    /// candidate authority, indexes, pending state or invalid graph paths.
    pub fn new(target: Arc<LpgStore>, mut candidate: LpgStore) -> Result<Self> {
        qualify_candidate(&mut candidate, 0).map_err(replacement_error)?;
        let scope = {
            let _scope = target.mutation_scope_gate.read_recursive();
            #[cfg(feature = "compact-store")]
            if !target.representation_is_active() || target.compact_base.get().is_some() {
                return Err(replacement_error("target has compact or retired backing"));
            }
            let scope = target.mutation_scope.load(Ordering::Acquire);
            if target.backward_adj.is_some() != candidate.backward_adj.is_some() {
                return Err(replacement_error(
                    "backward adjacency configuration differs",
                ));
            }
            scope
        };
        qualify_child_scopes(&mut candidate, scope).map_err(replacement_error)?;
        candidate.mutation_scope.store(scope, Ordering::Release);
        let mut graphs = Vec::new();
        graphs
            .try_reserve(1)
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        graphs.push((GraphPath::root(), Arc::clone(&target)));
        collect_replacement_graphs(&candidate, &GraphPath::root(), &mut graphs)?;
        graphs.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        let mut topology = Vec::new();
        topology
            .try_reserve(graphs.len())
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        for (path, anchor) in &graphs {
            let source = if path.components().is_empty() {
                &candidate
            } else {
                anchor
            };
            topology.push(
                source
                    .named_graphs
                    .read()
                    .iter()
                    .map(|(name, child)| (name.clone(), Arc::clone(child)))
                    .collect(),
            );
        }
        let image =
            crate::graph::lpg::LpgStoreSection::serialize_replacement_image(&candidate, &graphs)?;
        Ok(Self {
            target,
            candidate,
            graphs,
            topology,
            scope,
            attempted: false,
            validated: false,
            image,
            rechecked_image: None,
        })
    }

    /// Exact retained live root. Clone before preparing aggregate authority.
    #[must_use]
    pub fn target(&self) -> &Arc<LpgStore> {
        &self.target
    }

    /// Canonical root-inclusive destination owners for detached index staging.
    #[must_use]
    pub fn graphs(&self) -> &[(GraphPath, Arc<LpgStore>)] {
        &self.graphs
    }

    /// Detached incoming root for index staging before the publication attempt.
    #[must_use]
    pub fn candidate(&self) -> &LpgStore {
        &self.candidate
    }

    pub(crate) fn validate_under_transitions(
        &mut self,
        transitions: &[PinnedLpgTransition<'_>],
    ) -> std::result::Result<(), super::DataRebindError> {
        let reject = super::DataRebindError::new;
        if self.validated
            || self.attempted
            || self.target.mutation_scope.load(Ordering::Acquire) != self.scope
        {
            return Err(reject("replacement workspace was used or changed scope"));
        }
        for ((path, anchor), expected_children) in self.graphs.iter().zip(&self.topology) {
            if !transitions
                .iter()
                .any(|transition| transition.pins_store(anchor))
            {
                return Err(reject(
                    "replacement graph lacks its exact retained transition",
                ));
            }
            let source = if path.components().is_empty() {
                &self.candidate
            } else {
                anchor
            };
            qualify_shared_candidate(source, self.scope)?;
            let children = source.named_graphs.read();
            for (name, child) in expected_children {
                if !children
                    .get(name)
                    .is_some_and(|current| Arc::ptr_eq(current, child))
                {
                    return Err(reject("replacement child topology identity changed"));
                }
            }
            if children.len() != expected_children.len() {
                return Err(reject("replacement child topology changed"));
            }
        }
        self.rechecked_image = Some(
            crate::graph::lpg::LpgStoreSection::serialize_replacement_image(
                &self.candidate,
                &self.graphs,
            )
            .map_err(|_| reject("replacement data image is no longer valid"))?,
        );
        if self.rechecked_image.as_deref() != Some(self.image.as_slice()) {
            return Err(reject("replacement data changed after staging"));
        }
        self.validated = true;
        Ok(())
    }

    pub(crate) fn with_prepared_replacement<E, T, R>(
        &mut self,
        transition: &PinnedLpgTransition<'_>,
        prepare_companions: impl FnOnce() -> std::result::Result<T, E>,
        publish_companions: impl FnOnce(T) -> R,
    ) -> std::result::Result<R, E>
    where
        E: From<super::DataRebindError>,
    {
        if !self.validated
            || self.attempted
            || self.target.mutation_scope.load(Ordering::Acquire) != self.scope
        {
            return Err(super::DataRebindError::new(
                "replacement workspace was used or changed scope",
            )
            .into());
        }
        self.attempted = true;
        self.target.with_image_under_transition(
            transition,
            &mut self.candidate,
            false,
            prepare_companions,
            publish_companions,
        )
    }
}

fn replacement_error(reason: &'static str) -> Error {
    Error::Storage(StorageError::InvalidWalEntry(format!(
        "cannot prepare LPG replacement: {reason}"
    )))
}

fn qualify_shared_candidate(
    store: &LpgStore,
    scope: u64,
) -> std::result::Result<(), super::DataRebindError> {
    let reject = super::DataRebindError::new;
    if store.mutation_scope.load(Ordering::Acquire) != scope
        || !store.property_indexes.read().is_empty()
        || !store.property_undo_log.read().is_empty()
        || !store.pending_tx_creates.read().is_empty()
        || !store.tx_property_overlay.read().is_empty()
        || !store.pending_tx_deletes.read().is_empty()
        || !store.pending_tx_edge_deletes.read().is_empty()
        || !store.read_trackers.read().is_empty()
        || !store.write_trackers.read().is_empty()
        || !store
            .node_identity_reservations
            .identities
            .lock()
            .is_empty()
        || !store
            .edge_identity_reservations
            .identities
            .lock()
            .is_empty()
        || store
            .transport_edges_may_be_unresolved
            .load(Ordering::Acquire)
    {
        return Err(reject(
            "candidate acquired authority, indexes, reservations or transaction state",
        ));
    }
    let frontier = EpochId::new(store.current_epoch.load(Ordering::Acquire));
    if frontier == EpochId::PENDING
        || !store.node_properties.is_committed_restore_image(frontier)
        || !store.edge_properties.is_committed_restore_image(frontier)
        || store.node_labels.read().values().any(|history| {
            history
                .iter()
                .any(|(epoch, _)| *epoch == EpochId::PENDING || *epoch > frontier)
        })
    {
        return Err(reject("candidate contains uncommitted history"));
    }
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    if !store.index_slots.lock().is_empty() {
        return Err(reject("candidate acquired index slots"));
    }
    #[cfg(feature = "vector-index")]
    if !store.vector_indexes.read().is_empty() {
        return Err(reject("candidate acquired vector indexes"));
    }
    #[cfg(feature = "text-index")]
    if !store.text_indexes.read().is_empty() || !store.text_index_overlay.read().is_empty() {
        return Err(reject("candidate acquired text state"));
    }
    #[cfg(feature = "compact-store")]
    if !store.representation_is_active() || store.compact_base.get().is_some() {
        return Err(reject("candidate has compact or retired backing"));
    }
    #[cfg(feature = "tiered-storage")]
    if Arc::strong_count(&store.arena_allocator) != 1
        || Arc::strong_count(&store.epoch_store) != 1
        || store.epoch_store.try_empty_restore_guard().is_none()
        || store
            .node_versions
            .read()
            .values()
            .any(|index| index.cold_count() != 0)
        || store
            .edge_versions
            .read()
            .values()
            .any(|index| index.cold_count() != 0)
    {
        return Err(reject(
            "candidate arena is shared or contains cold versions",
        ));
    }
    if !store
        .transport_extract_authority
        .read_recursive()
        .edge_nonces
        .read()
        .is_empty()
    {
        return Err(reject("candidate contains transport receipts"));
    }
    Ok(())
}

fn collect_replacement_graphs(
    source: &LpgStore,
    parent: &GraphPath,
    graphs: &mut Vec<(GraphPath, Arc<LpgStore>)>,
) -> Result<()> {
    for (name, child) in source.named_graphs.read().iter() {
        let path = parent
            .child(name)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        graphs
            .try_reserve(1)
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
        graphs.push((path.clone(), Arc::clone(child)));
        collect_replacement_graphs(child, &path, graphs)?;
    }
    Ok(())
}

impl LpgStore {
    /// Builds a detached live-replacement image with this target's configuration.
    ///
    /// # Errors
    /// Returns allocation or checked physical-identity exhaustion.
    pub fn new_live_replacement_candidate(&self) -> Result<Self> {
        self.new_restore_candidate().map_err(Error::from)
    }

    /// Constructs a detached exact-restore target with this representation's
    /// actual configuration, including its backward-adjacency choice.
    ///
    /// # Errors
    /// Returns allocator or checked binding-identity exhaustion.
    pub(crate) fn new_restore_candidate(&self) -> std::result::Result<Self, AllocError> {
        Self::with_config(self.representation_config.clone())
    }

    /// Installs a fully built, committed, hot-only image into this pristine
    /// physical store without replacing its externally retained `Arc` identity.
    ///
    /// The section decoder must validate all structural history before calling
    /// this installer. The candidate and every child must be detached and unsealed, without
    /// indexes, transaction state, transport authority or compact backing.
    /// Child aliases are rejected before any target mutation. The caller must
    /// have this target's exact write authority, if sealed. Do not call beneath
    /// another store transition: this method owns its entire admission cut.
    ///
    /// All target guards and qualification precede publication; afterwards
    /// only preallocated image swaps and scalar stores occur. The candidate
    /// retires displaced buffers after every target guard has drained.
    ///
    /// # Errors
    /// Returns a structured invalid-image error on contention, denied authority,
    /// a non-pristine destination or an ineligible candidate.
    #[doc(hidden)]
    pub fn install_pristine_image(&self, mut candidate: LpgStore) -> Result<()> {
        let result = qualify_candidate(&mut candidate, 0)
            .and_then(|()| self.install_pristine_image_inner(&mut candidate));
        #[cfg(test)]
        tests::finish_probe();
        result.map_err(|reason| {
            Error::Storage(StorageError::InvalidWalEntry(format!(
                "cannot install pristine LPG image: {reason}"
            )))
        })
    }

    fn install_pristine_image_inner(
        &self,
        candidate: &mut LpgStore,
    ) -> std::result::Result<(), &'static str> {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return Err("cannot restore during recursive read capture");
        }
        // Unrelated stores share this gate. Wait for their topology work;
        // contention there does not invalidate this target's prepared image.
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        let scope_guard = self
            .mutation_scope_gate
            .try_write()
            .ok_or("store transition is busy")?;
        #[cfg(feature = "compact-store")]
        if !self.representation_is_active() || self.compact_base.get().is_some() {
            return Err("destination has compact or retired backing");
        }
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0
            && !std::num::NonZeroU64::new(scope).is_some_and(crate::graph::write_permit::is_held)
        {
            return Err("destination lacks mutation authority");
        }
        let transport = self
            .transport_extract_authority
            .try_read()
            .ok_or("transport authority is busy")?;
        let transition = PinnedLpgTransition {
            store: self,
            _scope_transition: super::PinnedExclusiveScope {
                _reads: super::ExclusiveReadScope::enter(self),
                _guard: scope_guard,
            },
            transport_authority: transport,
        };
        // These children cannot escape: the earlier complete walk required
        // unique Arc ownership throughout the candidate tree.
        qualify_child_scopes(candidate, scope)?;
        self.with_image_under_transition(
            &transition,
            candidate,
            true,
            || Ok::<(), super::DataRebindError>(()),
            |()| (),
        )
        .map_err(|error| match error {
            super::DataRebindError::Invalid(reason) | super::DataRebindError::Conflict(reason) => {
                reason
            }
            super::DataRebindError::Allocation(_) => "restore preparation allocation failed",
        })
    }

    fn with_image_under_transition<E, T, R>(
        &self,
        transition: &PinnedLpgTransition<'_>,
        candidate: &mut LpgStore,
        pristine: bool,
        prepare_companions: impl FnOnce() -> std::result::Result<T, E>,
        publish_companions: impl FnOnce(T) -> R,
    ) -> std::result::Result<R, E>
    where
        E: From<super::DataRebindError>,
    {
        let reject = |reason| E::from(super::DataRebindError::new(reason));
        if !transition.pins_store(self) {
            return Err(reject("replacement transition belongs to another store"));
        }
        #[cfg(feature = "compact-store")]
        if !self.representation_is_active() || self.compact_base.get().is_some() {
            return Err(reject("destination has compact or retired backing"));
        }
        if self.backward_adj.is_some() != candidate.backward_adj.is_some() {
            return Err(reject("backward adjacency configuration differs"));
        }
        let node_ids = self
            .node_identity_reservations
            .identities
            .try_lock()
            .ok_or_else(|| reject("node reservation is busy"))?;
        let edge_ids = self
            .edge_identity_reservations
            .identities
            .try_lock()
            .ok_or_else(|| reject("edge reservation is busy"))?;
        if !node_ids.is_empty() || !edge_ids.is_empty() {
            return Err(reject("destination has identity reservations"));
        }
        #[cfg(not(feature = "tiered-storage"))]
        let mut nodes = self
            .nodes
            .try_write()
            .ok_or_else(|| reject("node storage is busy"))?;
        #[cfg(not(feature = "tiered-storage"))]
        let mut edges = self
            .edges
            .try_write()
            .ok_or_else(|| reject("edge storage is busy"))?;
        #[cfg(feature = "tiered-storage")]
        let mut nodes = self
            .node_versions
            .try_write()
            .ok_or_else(|| reject("node storage is busy"))?;
        #[cfg(feature = "tiered-storage")]
        let mut edges = self
            .edge_versions
            .try_write()
            .ok_or_else(|| reject("edge storage is busy"))?;
        #[cfg(feature = "tiered-storage")]
        let mut arenas = {
            let source = Arc::get_mut(&mut candidate.arena_allocator)
                .ok_or_else(|| reject("candidate arena is shared"))?;
            if pristine {
                self.arena_allocator.prepare_pristine_restore(source)
            } else {
                self.arena_allocator.prepare_replacement(source)
            }
        }
        .ok_or_else(|| reject("destination arena is busy, used or incompatible"))?;
        #[cfg(feature = "tiered-storage")]
        let _empty_cold = if pristine {
            Some(
                self.epoch_store
                    .try_empty_restore_guard()
                    .ok_or_else(|| reject("destination cold storage is busy or populated"))?,
            )
        } else {
            None
        };
        #[cfg(feature = "tiered-storage")]
        let mut cold = if pristine {
            None
        } else {
            Some(
                self.epoch_store
                    .prepare_replacement(
                        Arc::get_mut(&mut candidate.epoch_store)
                            .ok_or_else(|| reject("candidate cold store is shared"))?,
                    )
                    .ok_or_else(|| reject("destination cold storage is busy"))?,
            )
        };
        let mut labels = self
            .label_registry
            .try_write()
            .ok_or_else(|| reject("label registry is busy"))?;
        let mut types = self
            .edge_type_to_id
            .try_write()
            .ok_or_else(|| reject("type registry is busy"))?;
        let mut type_names = self
            .id_to_edge_type
            .try_write()
            .ok_or_else(|| reject("type names are busy"))?;
        let mut label_index = self
            .label_index
            .try_write()
            .ok_or_else(|| reject("label index is busy"))?;
        let mut label_history = self
            .node_labels
            .try_write()
            .ok_or_else(|| reject("label histories are busy"))?;
        let properties = if pristine {
            Some(
                self.property_indexes
                    .try_write()
                    .ok_or_else(|| reject("property registry is busy"))?,
            )
        } else {
            None
        };
        #[cfg(feature = "vector-index")]
        let vectors = if pristine {
            Some(
                self.vector_indexes
                    .try_write()
                    .ok_or_else(|| reject("vector registry is busy"))?,
            )
        } else {
            None
        };
        #[cfg(feature = "text-index")]
        let texts = if pristine {
            Some(
                self.text_indexes
                    .try_write()
                    .ok_or_else(|| reject("text registry is busy"))?,
            )
        } else {
            None
        };
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        let slots = if pristine {
            Some(
                self.index_slots
                    .try_lock()
                    .ok_or_else(|| reject("index slot registry is busy"))?,
            )
        } else {
            None
        };
        let mut type_counts = self
            .edge_type_live_counts
            .try_write()
            .ok_or_else(|| reject("type counts are busy"))?;
        let mut statistics = self
            .statistics
            .try_write()
            .ok_or_else(|| reject("statistics are busy"))?;
        let mut node_properties = if pristine {
            self.node_properties
                .prepare_pristine_restore(&mut candidate.node_properties)
        } else {
            self.node_properties
                .prepare_replacement(&mut candidate.node_properties)
        }
        .ok_or_else(|| reject("node properties are busy, populated or incompatible"))?;
        let mut edge_properties = if pristine {
            self.edge_properties
                .prepare_pristine_restore(&mut candidate.edge_properties)
        } else {
            self.edge_properties
                .prepare_replacement(&mut candidate.edge_properties)
        }
        .ok_or_else(|| reject("edge properties are busy, populated or incompatible"))?;
        let mut forward = if pristine {
            self.forward_adj
                .prepare_pristine_restore(&mut candidate.forward_adj)
        } else {
            self.forward_adj
                .prepare_replacement(&mut candidate.forward_adj)
        }
        .ok_or_else(|| reject("forward adjacency is busy, populated or incompatible"))?;
        let mut backward = match (&self.backward_adj, &mut candidate.backward_adj) {
            (Some(target), Some(source)) => Some(
                if pristine {
                    target.prepare_pristine_restore(source)
                } else {
                    target.prepare_replacement(source)
                }
                .ok_or_else(|| reject("backward adjacency is busy, populated or incompatible"))?,
            ),
            (None, None) => None,
            _ => return Err(reject("backward adjacency configuration differs")),
        };
        let mut graphs = self
            .named_graphs
            .try_write()
            .ok_or_else(|| reject("child topology is busy"))?;
        let undo = self
            .property_undo_log
            .try_write()
            .ok_or_else(|| reject("property undo is busy"))?;
        let creates = self
            .pending_tx_creates
            .try_write()
            .ok_or_else(|| reject("pending creates are busy"))?;
        let overlay = self
            .tx_property_overlay
            .try_write()
            .ok_or_else(|| reject("property overlays are busy"))?;
        #[cfg(feature = "text-index")]
        let text_overlay = self
            .text_index_overlay
            .try_write()
            .ok_or_else(|| reject("text overlays are busy"))?;
        let deletes = self
            .pending_tx_deletes
            .try_write()
            .ok_or_else(|| reject("pending deletes are busy"))?;
        let edge_deletes = self
            .pending_tx_edge_deletes
            .try_write()
            .ok_or_else(|| reject("pending edge deletes are busy"))?;
        let readers = self
            .read_trackers
            .try_write()
            .ok_or_else(|| reject("read trackers are busy"))?;
        let writers = self
            .write_trackers
            .try_write()
            .ok_or_else(|| reject("write trackers are busy"))?;
        let receipts = transition
            .transport_authority
            .edge_nonces
            .try_read()
            .ok_or_else(|| reject("transport receipts are busy"))?;
        if pristine
            && (!nodes.is_empty()
                || !edges.is_empty()
                || !labels.name_to_id.is_empty()
                || !labels.id_to_name.is_empty()
                || !types.is_empty()
                || !type_names.is_empty()
                || !label_index.is_empty()
                || !label_history.is_empty()
                || properties
                    .as_ref()
                    .is_some_and(|entries| !entries.is_empty())
                || !type_counts.is_empty()
                || !graphs.is_empty()
                || self.next_node_id.load(Ordering::Acquire) != 0
                || self.next_edge_id.load(Ordering::Acquire) != 0
                || self.current_epoch.load(Ordering::Acquire) != 0
                || self.live_node_count.load(Ordering::Acquire) != 0
                || self.live_edge_count.load(Ordering::Acquire) != 0)
        {
            return Err(reject("destination is not pristine"));
        }
        if !undo.is_empty()
            || !creates.is_empty()
            || !overlay.is_empty()
            || !deletes.is_empty()
            || !edge_deletes.is_empty()
            || !readers.is_empty()
            || !writers.is_empty()
            || !receipts.is_empty()
            || self
                .transport_edges_may_be_unresolved
                .load(Ordering::Acquire)
        {
            return Err(reject("destination has transaction or transport state"));
        }
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        if slots.as_ref().is_some_and(|entries| !entries.is_empty()) {
            return Err(reject("destination has retained index slots"));
        }
        #[cfg(feature = "vector-index")]
        if vectors.as_ref().is_some_and(|entries| !entries.is_empty()) {
            return Err(reject("destination has vector indexes"));
        }
        #[cfg(feature = "text-index")]
        if texts.as_ref().is_some_and(|entries| !entries.is_empty()) || !text_overlay.is_empty() {
            return Err(reject("destination has text state"));
        }
        let mut graph_identity = self
            .graph_identity
            .try_write()
            .ok_or_else(|| reject("graph identity is busy"))?;
        let companions = prepare_companions()?;
        #[cfg(test)]
        tests::before_install().map_err(&reject)?;

        // No fallible operation, allocation, payload destruction or per-entity
        // replay below this line. Every actual backing writer remains held.
        #[cfg(feature = "tiered-storage")]
        arenas.install();
        #[cfg(feature = "tiered-storage")]
        if let Some(cold) = &mut cold {
            cold.install();
        }
        #[cfg(not(feature = "tiered-storage"))]
        {
            std::mem::swap(&mut *nodes, candidate.nodes.get_mut());
            std::mem::swap(&mut *edges, candidate.edges.get_mut());
        }
        #[cfg(feature = "tiered-storage")]
        {
            std::mem::swap(&mut *nodes, candidate.node_versions.get_mut());
            std::mem::swap(&mut *edges, candidate.edge_versions.get_mut());
        }
        std::mem::swap(&mut *labels, candidate.label_registry.get_mut());
        std::mem::swap(&mut *types, candidate.edge_type_to_id.get_mut());
        std::mem::swap(&mut *type_names, candidate.id_to_edge_type.get_mut());
        std::mem::swap(&mut *label_index, candidate.label_index.get_mut());
        std::mem::swap(&mut *label_history, candidate.node_labels.get_mut());
        std::mem::swap(&mut *type_counts, candidate.edge_type_live_counts.get_mut());
        std::mem::swap(&mut *statistics, candidate.statistics.get_mut());
        node_properties.install();
        edge_properties.install();
        forward.install();
        if let Some(backward) = &mut backward {
            backward.install();
        }
        std::mem::swap(&mut *graphs, candidate.named_graphs.get_mut());
        std::mem::swap(&mut *graph_identity, candidate.graph_identity.get_mut());
        self.next_node_id.store(
            candidate.next_node_id.load(Ordering::Relaxed),
            Ordering::Release,
        );
        self.next_edge_id.store(
            candidate.next_edge_id.load(Ordering::Relaxed),
            Ordering::Release,
        );
        self.live_node_count.store(
            candidate.live_node_count.load(Ordering::Relaxed),
            Ordering::Release,
        );
        self.live_edge_count.store(
            candidate.live_edge_count.load(Ordering::Relaxed),
            Ordering::Release,
        );
        self.needs_stats_recompute.store(
            candidate.needs_stats_recompute.load(Ordering::Relaxed),
            Ordering::Release,
        );
        self.current_epoch.store(
            candidate.current_epoch.load(Ordering::Relaxed),
            Ordering::Release,
        );
        self.retained_history_floor.store(
            candidate.retained_history_floor.load(Ordering::Relaxed),
            Ordering::Release,
        );
        Ok(publish_companions(companions))
    }
}

fn qualify_candidate(store: &mut LpgStore, depth: usize) -> std::result::Result<(), &'static str> {
    if depth > MAX_GRAPH_PATH_COMPONENTS {
        return Err("candidate graph depth exceeds the path bound");
    }
    if store.mutation_scope.load(Ordering::Acquire) != 0
        || !store.property_indexes.get_mut().is_empty()
        || !store.property_undo_log.get_mut().is_empty()
        || !store.pending_tx_creates.get_mut().is_empty()
        || !store.tx_property_overlay.get_mut().is_empty()
        || !store.pending_tx_deletes.get_mut().is_empty()
        || !store.pending_tx_edge_deletes.get_mut().is_empty()
        || !store.read_trackers.get_mut().is_empty()
        || !store.write_trackers.get_mut().is_empty()
        || !store
            .node_identity_reservations
            .identities
            .get_mut()
            .is_empty()
        || !store
            .edge_identity_reservations
            .identities
            .get_mut()
            .is_empty()
        || store
            .transport_edges_may_be_unresolved
            .load(Ordering::Acquire)
    {
        return Err("candidate carries authority, indexes, reservations or transaction state");
    }
    let frontier = EpochId::new(store.current_epoch.load(Ordering::Acquire));
    if frontier == EpochId::PENDING
        || !store.node_properties.is_committed_restore_image(frontier)
        || !store.edge_properties.is_committed_restore_image(frontier)
        || store.node_labels.get_mut().values().any(|history| {
            history
                .iter()
                .any(|(epoch, _)| *epoch == EpochId::PENDING || *epoch > frontier)
        })
    {
        return Err("candidate contains uncommitted history");
    }
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    if !store.index_slots.get_mut().is_empty() {
        return Err("candidate has retained index slots");
    }
    #[cfg(feature = "vector-index")]
    if !store.vector_indexes.get_mut().is_empty() {
        return Err("candidate has vector indexes");
    }
    #[cfg(feature = "text-index")]
    if !store.text_indexes.get_mut().is_empty() || !store.text_index_overlay.get_mut().is_empty() {
        return Err("candidate has text state");
    }
    #[cfg(feature = "compact-store")]
    if !store.representation_is_active() || store.compact_base.get().is_some() {
        return Err("candidate has compact or retired backing");
    }
    #[cfg(feature = "tiered-storage")]
    {
        if Arc::get_mut(&mut store.arena_allocator).is_none()
            || !Arc::get_mut(&mut store.epoch_store)
                .ok_or("candidate cold backing is shared")?
                .is_empty_restore_image()
            || store
                .node_versions
                .get_mut()
                .values()
                .any(|index| index.cold_count() != 0)
            || store
                .edge_versions
                .get_mut()
                .values()
                .any(|index| index.cold_count() != 0)
        {
            return Err("candidate arena is shared or contains cold versions");
        }
    }
    let authority = Arc::get_mut(store.transport_extract_authority.get_mut())
        .ok_or("candidate transport authority is shared")?;
    if !authority.edge_nonces.get_mut().is_empty() {
        return Err("candidate contains transport receipts");
    }
    for child in store.named_graphs.get_mut().values_mut() {
        qualify_candidate(
            Arc::get_mut(child).ok_or("candidate child is aliased")?,
            depth + 1,
        )?;
    }
    Ok(())
}

fn qualify_child_scopes(store: &mut LpgStore, scope: u64) -> std::result::Result<(), &'static str> {
    for child in store.named_graphs.get_mut().values_mut() {
        let child = Arc::get_mut(child).ok_or("candidate child ownership changed")?;
        qualify_child_scopes(child, scope)?;
        child.mutation_scope.store(scope, Ordering::Release);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
