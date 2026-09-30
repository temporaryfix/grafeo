//! Scoped physical/data publication consumed by the engine's commit path.
//!
//! This owns no second transaction authority. The caller supplies its exact
//! stores, write authority, publication frontier and normalized final rows.

#[cfg(any(feature = "compact-store", feature = "vector-index"))]
use super::reserve_vec;
use super::{
    BorrowedPreparedRegistry, DataRebindError, IndexRegistryEdit, IndexRegistryWorkspace, LpgStore,
    ReleasedRegistryBatch, StoreIndexEdits, conflict, prepare_under_authority,
};
#[cfg(feature = "vector-index")]
use super::{IndexRegistrationObservation, ObservedIndex};
#[cfg(feature = "compact-store")]
use super::{LocalKey, PrivateMaintenance};
use crate::graph::lpg::store::PreparedNodeLabelImages;
use crate::graph::lpg::store::data_publication::StoreDataWorkspace;
use crate::graph::lpg::store::data_publication::slots::{
    InstalledStoreDataSlots, PreparedStoreDataSlots, ReleasedStoreDataSlots, StoreDataSlot,
    StoreDataSlots, prepare_store_data_slots,
};
use crate::graph::traits::{GraphStoreMut, LpgCommitRepresentation};
use crate::graph::write_permit::WriteAuthority;
use grafeo_common::types::{EpochId, GraphPath, TransactionId};
use grafeo_common::utils::error::Result;
#[cfg(feature = "vector-index")]
use grafeo_common::{
    memory::AllocError,
    types::{NodeId, PropertyKey},
    utils::hash::{FxHashMap, FxHashSet},
};
use std::sync::Arc;

#[cfg(feature = "compact-store")]
use crate::graph::compact::layered::commit::slots::{
    InstalledLayeredCommitSlots, LayeredCommitReaderFence, LayeredCommitSlot, LayeredCommitSlots,
    PreparedLayeredCommitSlots, prepare_layered_commit_slots,
};
#[cfg(feature = "compact-store")]
use crate::graph::compact::layered::{
    LayeredStore,
    commit::{LayeredCommitPin, LayeredCommitWorkspace},
};
#[cfg(feature = "vector-index")]
use crate::index::vector::{
    VectorIndexView,
    maintenance::{
        InstalledVectorMaintenanceSlots, PreparedVectorMaintenanceSlots, VectorMaintenancePin,
        VectorMaintenanceReaderFence, VectorMaintenanceSlot, VectorMaintenanceSlots,
        prepare_vector_maintenance_slots,
    },
    value_to_vector,
};

/// One independently anchored physical target and the mutation representation
/// that accepted its transaction writes. A read-only discovery token alone
/// cannot construct this input or authorize publication.
pub struct StoreCommitInput<'target> {
    /// Exact native store or Layered overlay, independently retained by engine.
    pub store: &'target LpgStore,
    /// Exact mutation representation, including a Layered cold-delete owner.
    pub source: &'target dyn GraphStoreMut,
    /// Independently retained topology root and literal path to this target.
    /// Engine inputs supply this witness; detached core callers may omit it.
    /// A staged graph uses its retained staged root and the relative suffix.
    pub graph: Option<(&'target Arc<LpgStore>, &'target GraphPath)>,
    /// Whether this surviving target publishes transaction-owned buffered data.
    pub publish_data: bool,
    /// Normalized final registry edits; survivor Property/Text use Maintain.
    pub edits: Vec<IndexRegistryEdit>,
}

/// Surviving Vector state. The view and observation are independent outer
/// anchors; neither grants mutation authority without the exact owning store.
#[cfg(feature = "vector-index")]
pub struct VectorCommitInput<'target> {
    /// Exact registry owner.
    pub store: &'target LpgStore,
    /// Retained registered target, not a replacement Vector index.
    pub view: &'target VectorIndexView,
    /// Registration identity captured with this view.
    pub expected: &'target IndexRegistrationObservation,
    /// Exact property of the observed Vector registration.
    pub property: PropertyKey,
    /// One before-to-final transition per affected node.
    pub changes: VectorCommitChanges,
    /// Sparse touched routing vectors: final values override committed old
    /// vectors, and deleted routing nodes retain their old vector. Untouched
    /// routing rows use the captured source's non-recording committed read.
    pub routing: FxHashMap<NodeId, Arc<[f32]>>,
}

/// Either final-row maintenance or exact recorded changes, never both.
#[cfg(feature = "vector-index")]
pub enum VectorCommitChanges {
    /// Live final-row values; ordinary preparation performs graph maintenance.
    Rows(Vec<(NodeId, Option<Arc<[f32]>>)>),
    /// Opaque exact postimages, qualified during the same scoped preparation.
    Recorded(Vec<u8>),
}

struct Source<'target> {
    store: &'target LpgStore,
    source: &'target dyn GraphStoreMut,
    graph: Option<(&'target Arc<LpgStore>, &'target GraphPath)>,
    #[cfg(feature = "compact-store")]
    publish_data: bool,
}

#[cfg(feature = "vector-index")]
struct VectorSource<'target> {
    store: &'target LpgStore,
    view: &'target VectorIndexView,
    expected: &'target IndexRegistrationObservation,
    property: PropertyKey,
    source: Option<&'target dyn GraphStoreMut>,
    routing: FxHashMap<NodeId, Arc<[f32]>>,
    slot: Option<u64>,
}

/// All candidates, raw guard slots and allocated pin/topology buffers live
/// outside the engine's enclosing publication locks. Initialize this owner
/// before entering those locks, and retain it until after they have drained.
/// Its constructor owns inputs first; validation happens only in the driver.
pub struct LpgCommitWorkspace<'target> {
    registry: IndexRegistryWorkspace<'target>,
    data: StoreDataSlots<'target>,
    sources: Vec<Source<'target>>,
    transaction: TransactionId,
    publication: EpochId,
    commit: EpochId,
    attempted: bool,
    #[cfg(feature = "vector-index")]
    vectors: Vec<VectorSource<'target>>,
    #[cfg(feature = "vector-index")]
    vector_slots: VectorMaintenanceSlots<'target>,
    #[cfg(feature = "vector-index")]
    vector_pins: Vec<VectorMaintenancePin<'target>>,
    #[cfg(feature = "vector-index")]
    vector_identities: FxHashSet<usize>,
    #[cfg(feature = "compact-store")]
    layered_targets: Vec<(&'target LayeredStore, &'target LpgStore)>,
    #[cfg(feature = "compact-store")]
    layered_slots: LayeredCommitSlots<'target>,
    #[cfg(feature = "compact-store")]
    layered_pins: Vec<LayeredCommitPin<'target>>,
    #[cfg(feature = "compact-store")]
    layered_observation_pins: Vec<LayeredCommitPin<'target>>,
}

impl<'target> LpgCommitWorkspace<'target> {
    /// Captures independently anchored inputs without fallible validation.
    #[must_use]
    pub fn new(
        stores: Vec<StoreCommitInput<'target>>,
        #[cfg(feature = "vector-index")] vectors: Vec<VectorCommitInput<'target>>,
        transaction: TransactionId,
        publication: EpochId,
        commit: EpochId,
    ) -> Self {
        let mut sources = Vec::with_capacity(stores.len());
        let mut data = Vec::with_capacity(stores.len());
        let mut edits = Vec::with_capacity(stores.len());
        for input in stores {
            sources.push(Source {
                store: input.store,
                source: input.source,
                graph: input.graph,
                #[cfg(feature = "compact-store")]
                publish_data: input.publish_data,
            });
            if input.publish_data {
                data.push(StoreDataSlot::new(
                    input.store,
                    StoreDataWorkspace::new(transaction, publication, commit),
                ));
            }
            edits.push(StoreIndexEdits {
                store: input.store,
                edits: input.edits,
            });
        }
        #[cfg(feature = "vector-index")]
        let (vectors, vector_slots) = {
            let mut captured = Vec::with_capacity(vectors.len());
            let mut slots = Vec::with_capacity(vectors.len());
            for input in vectors {
                let target = input.view.commit_target();
                slots.push(match input.changes {
                    VectorCommitChanges::Rows(rows) => VectorMaintenanceSlot::new(target, rows),
                    VectorCommitChanges::Recorded(payload) => {
                        VectorMaintenanceSlot::from_recorded(target, payload)
                    }
                });
                captured.push(VectorSource {
                    store: input.store,
                    view: input.view,
                    expected: input.expected,
                    property: input.property,
                    source: None,
                    routing: input.routing,
                    slot: None,
                });
            }
            (captured, VectorMaintenanceSlots::new(slots))
        };
        Self {
            registry: IndexRegistryWorkspace::new(edits),
            data: StoreDataSlots::new(data),
            sources,
            transaction,
            publication,
            commit,
            attempted: false,
            #[cfg(feature = "vector-index")]
            vectors,
            #[cfg(feature = "vector-index")]
            vector_slots,
            #[cfg(feature = "vector-index")]
            vector_pins: Vec::new(),
            #[cfg(feature = "vector-index")]
            vector_identities: FxHashSet::default(),
            #[cfg(feature = "compact-store")]
            layered_targets: Vec::new(),
            #[cfg(feature = "compact-store")]
            layered_slots: LayeredCommitSlots::new(Vec::new()),
            #[cfg(feature = "compact-store")]
            layered_pins: Vec::new(),
            #[cfg(feature = "compact-store")]
            layered_observation_pins: Vec::new(),
        }
    }

    /// Requests bounded exact Text WAL postimages before any preparation.
    /// Memory-only commits leave this disabled and allocate no record buffers.
    ///
    /// # Errors
    /// Rejects a workspace that has already attempted preparation.
    #[cfg(feature = "text-index")]
    pub fn capture_text_postimages(&mut self) -> Result<()> {
        if self.attempted {
            return Err(super::conflict("Text capture must precede preparation"));
        }
        for store in &mut self.registry.registry.stores {
            for edit in &mut store.edits {
                edit.capture_text_wal = true;
            }
        }
        Ok(())
    }

    /// Requests exact Vector births and sparse postimages before preparation.
    ///
    /// # Errors
    /// Rejects an already attempted workspace.
    #[cfg(feature = "vector-index")]
    pub fn capture_vector_postimages(&mut self) -> Result<()> {
        if self.attempted {
            return Err(conflict("Vector capture must precede preparation"));
        }
        self.vector_slots.capture_wal();
        for store in &mut self.registry.registry.stores {
            for edit in &mut store.edits {
                edit.capture_vector_wal = true;
            }
        }
        Ok(())
    }

    /// Requests exact committed label images for a durable writer.
    ///
    /// In-memory commits do not materialize label names unless requested.
    ///
    /// # Errors
    /// Rejects a request after commit preparation has begun.
    pub fn capture_label_images(&mut self) -> Result<()> {
        if self.attempted {
            return Err(conflict(
                "label image capture requires an unprepared commit workspace",
            ));
        }
        self.data.capture_label_images();
        Ok(())
    }

    fn release_guards(&mut self) {
        // Cross-family cleanup must precede field destruction, even if the
        // callback deliberately forgot its ready/installed borrowed proof.
        self.registry.registry.release_guards();
        self.data.release_guards();
        #[cfg(feature = "vector-index")]
        self.vector_slots.release_guards();
        #[cfg(feature = "compact-store")]
        self.layered_slots.release_guards();
        #[cfg(feature = "vector-index")]
        while let Some(pin) = self.vector_pins.pop() {
            drop(pin);
        }
        self.registry.authority.release_guards();
        #[cfg(feature = "compact-store")]
        while let Some(pin) = self.layered_pins.pop() {
            drop(pin);
        }
        #[cfg(feature = "compact-store")]
        while let Some(pin) = self.layered_observation_pins.pop() {
            drop(pin);
        }
    }

    fn acquire(&mut self, authority: &WriteAuthority) -> Result<()> {
        if self.attempted {
            return Err(conflict("LPG commit workspace is one-shot"));
        }
        self.attempted = true;
        if self.transaction == TransactionId::INVALID
            || self.transaction == TransactionId::SYSTEM
            || self.publication >= self.commit
            || self.commit == EpochId::PENDING
        {
            return Err(conflict("invalid LPG commit frontier"));
        }
        #[cfg(feature = "compact-store")]
        {
            reserve_vec(&mut self.layered_targets, self.sources.len())?;
            reserve_vec(&mut self.layered_pins, self.sources.len())?;
            reserve_vec(&mut self.layered_observation_pins, self.sources.len())?;
        }
        for source in &self.sources {
            if !source.store.accepts_held_write_authority(authority) {
                return Err(conflict(
                    "LPG commit requires the exact held write authority",
                ));
            }
            match source.source.lpg_commit_target()?.representation {
                LpgCommitRepresentation::Native(store) => {
                    if !std::ptr::eq(store, source.store) {
                        return Err(conflict("native commit source differs from captured store"));
                    }
                }
                #[cfg(feature = "compact-store")]
                LpgCommitRepresentation::Layered(layered) => {
                    // Registry-only targets still need the merge pin against
                    // overlay replacement, but have no cold transaction work.
                    let pin = layered
                        .pin_commit(source.store)
                        .map_err(DataRebindError::into_error)?;
                    if source.publish_data {
                        self.layered_pins.push(pin);
                        self.layered_targets.push((layered, source.store));
                    } else {
                        self.layered_observation_pins.push(pin);
                    }
                }
            }
        }
        #[cfg(feature = "compact-store")]
        {
            let slots = self
                .layered_targets
                .iter()
                .map(|(layered, _)| {
                    LayeredCommitSlot::new(
                        layered,
                        LayeredCommitWorkspace::new(
                            self.transaction,
                            self.publication,
                            self.commit,
                        ),
                    )
                })
                .collect();
            self.layered_slots = LayeredCommitSlots::new(slots);
        }
        self.registry.prepare_inputs()?;
        self.registry
            .authority
            .acquire_pending(&mut self.registry.registry.stores)?;
        // Check again beneath continuous transitions, closing seal/retirement
        // changes between the initial authority check and admission.
        for source in &self.sources {
            if !source.store.accepts_held_write_authority(authority) {
                return Err(conflict("LPG commit authority changed during admission"));
            }
            if let Some((root, path)) = source.graph {
                self.registry
                    .authority
                    .bindings()?
                    .validate_graph(root, path, source.store)?;
            }
        }
        #[cfg(feature = "compact-store")]
        self.normalize_layered_property_before()?;
        #[cfg(feature = "vector-index")]
        {
            reserve_vec(&mut self.vector_pins, self.vectors.len())?;
            self.vector_identities
                .try_reserve(self.vectors.len())
                .map_err(|_| AllocError::OutOfMemory)?;
            for vector in &mut self.vectors {
                let transition = self
                    .registry
                    .authority
                    .transition(vector.store)
                    .map_err(DataRebindError::into_error)?;
                vector.store.validate_registration_store(
                    vector.expected,
                    &transition.transport_authority,
                )?;
                let ObservedIndex::Vector(key, _) = vector.expected.target() else {
                    return Err(conflict(
                        "Vector maintenance observation has another family",
                    ));
                };
                if crate::graph::lpg::decode_index_key(key).map(|(_, property)| property)
                    != Some(vector.property.as_str())
                {
                    return Err(conflict(
                        "Vector routing property differs from registration",
                    ));
                }
                vector.source = Some(
                    self.sources
                        .iter()
                        .find(|source| std::ptr::eq(source.store, vector.store))
                        .ok_or_else(|| conflict("Vector routing source is absent"))?
                        .source,
                );
                let slots = vector.store.index_slots.lock();
                let indexes = vector.store.vector_indexes.read();
                let current = indexes
                    .get(key)
                    .ok_or_else(|| conflict("Vector maintenance registration is absent"))?;
                if !vector.expected.matches_vector(current)
                    || !std::ptr::eq(current.payload.as_ref(), vector.view.commit_target())
                {
                    return Err(conflict(
                        "Vector maintenance target differs from captured registration",
                    ));
                }
                vector.slot = Some(
                    *slots
                        .get(key)
                        .ok_or_else(|| conflict("Vector maintenance binding slot is absent"))?,
                );
                if !self
                    .vector_identities
                    .insert(std::ptr::from_ref(vector.view.commit_target()).addr())
                {
                    return Err(conflict("Vector maintenance repeats a concrete target"));
                }
            }
            for vector in &self.vectors {
                self.vector_pins
                    .push(vector.view.commit_target().pin_maintenance()?);
            }
        }
        Ok(())
    }

    #[cfg(feature = "compact-store")]
    fn normalize_layered_property_before(&mut self) -> Result<()> {
        for pending in &mut self.registry.registry.stores {
            let Some(pin) = self
                .layered_pins
                .iter()
                .chain(&self.layered_observation_pins)
                .find(|pin| pin.pins_overlay(pending.store))
            else {
                continue;
            };
            let transition = self
                .registry
                .authority
                .transition(pending.store)
                .map_err(DataRebindError::into_error)?;
            // This preparation-only registry reader may nest a node-identity
            // reader: the already-held exclusive store transition excludes
            // every structural mutator/maintenance writer, including the
            // otherwise opposing structural-writer -> registry-writer path.
            // Never perform this normalization beneath final data writers.
            let indexes = pending.store.property_indexes.read();
            for edit in &mut pending.edits {
                if let (LocalKey::Property(key), Some(PrivateMaintenance::Property(workspace))) =
                    (&edit.key, &mut edit.maintenance)
                {
                    let current = indexes
                        .get(key)
                        .ok_or_else(|| conflict("Property maintenance registration is absent"))?;
                    workspace.normalize_unhydrated_cold_before(
                        &current.payload,
                        pin,
                        transition,
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl Drop for LpgCommitWorkspace<'_> {
    fn drop(&mut self) {
        self.release_guards();
    }
}

/// Scoped cleanup is installed BEFORE acquisition and survives the callback.
/// All heap owners stay outside; this destructor releases guards only.
struct CommitScope<'target, 'workspace>(&'workspace mut LpgCommitWorkspace<'target>);

impl Drop for CommitScope<'_, '_> {
    fn drop(&mut self) {
        self.0.release_guards();
    }
}

/// Completes preparation and reader drainage, then lends a premarker batch.
/// The callback may prepare logical companions, try-bind, append its durable
/// marker and install. It must not invoke ordinary graph/index mutators or
/// readers that reacquire this authority. Graph lifecycle runs after return.
///
/// # Errors
/// Rejects invalid frontiers/authority, stale targets, failed preparation and
/// reader admission. Callback errors preserve their existing engine meaning.
pub fn with_prepared_lpg_commit<'target, R>(
    workspace: &mut LpgCommitWorkspace<'target>,
    authority: &WriteAuthority,
    callback: impl for<'loan> FnOnce(ReleasedLpgCommit<'target, 'loan>) -> Result<R>,
) -> Result<R> {
    let scope = CommitScope(workspace);
    scope.0.acquire(authority)?;
    let data = prepare_store_data_slots(
        &mut scope.0.data,
        scope.0.registry.authority.retained_transitions(),
    )?;
    let registry =
        prepare_under_authority(&mut scope.0.registry.registry, &scope.0.registry.authority)?;
    #[cfg(feature = "compact-store")]
    let layered = prepare_layered_commit_slots(
        &mut scope.0.layered_slots,
        &scope.0.layered_pins,
        scope.0.registry.authority.retained_transitions(),
    )
    .map_err(DataRebindError::into_error)?;
    #[cfg(feature = "vector-index")]
    let vectors = prepare_vector_maintenance_slots(
        &mut scope.0.vector_slots,
        &scope.0.vector_pins,
        &|ordinal, id| {
            scope.0.vectors.get(ordinal).and_then(|v| {
                v.routing.get(&id).cloned().or_else(|| {
                    v.source?
                        .read_node_property_visible(id, &v.property, scope.0.publication, None)
                        .as_ref()
                        .and_then(value_to_vector)
                        // A soft-deleted routing hop can retain topology after
                        // its now-unindexed property changes shape. Missing
                        // routing payload is allowed; explicit upsert rows
                        // still undergo strict dimension validation.
                        .filter(|vector| vector.len() == v.view.config().dimensions)
                })
            })
        },
    )?
    .exclude_readers()
    .map_err(DataRebindError::into_error)?;
    #[cfg(feature = "compact-store")]
    let layered = layered
        .exclude_readers()
        .map_err(DataRebindError::into_error)?;
    callback(ReleasedLpgCommit {
        registry,
        data,
        #[cfg(feature = "vector-index")]
        vectors,
        #[cfg(feature = "vector-index")]
        vector_sources: &scope.0.vectors,
        #[cfg(feature = "compact-store")]
        layered,
    })
}

/// All private candidates are complete; final map writers are not retained.
#[must_use]
pub struct ReleasedLpgCommit<'target, 'loan> {
    registry: ReleasedRegistryBatch<'target, 'loan, 'loan>,
    data: ReleasedStoreDataSlots<'target, 'loan, 'loan>,
    #[cfg(feature = "vector-index")]
    vectors: VectorMaintenanceReaderFence<'target, 'loan, 'loan>,
    #[cfg(feature = "vector-index")]
    vector_sources: &'loan [VectorSource<'target>],
    #[cfg(feature = "compact-store")]
    layered: LayeredCommitReaderFence<'target, 'loan, 'loan>,
}

/// Every affected store/index kind has its final writer before any install.
#[must_use]
pub struct PreparedLpgCommit<'target, 'loan> {
    registry: BorrowedPreparedRegistry<'target, 'loan, 'loan>,
    data: PreparedStoreDataSlots<'target, 'loan, 'loan>,
    #[cfg(feature = "vector-index")]
    vectors: PreparedVectorMaintenanceSlots<'target, 'loan, 'loan>,
    #[cfg(feature = "compact-store")]
    layered: PreparedLayeredCommitSlots<'target, 'loan, 'loan>,
}

/// Retains every final writer through companion catalog/TM installation.
#[must_use]
pub struct InstalledLpgCommit<'target, 'loan> {
    _registry: BorrowedPreparedRegistry<'target, 'loan, 'loan>,
    _data: InstalledStoreDataSlots<'target, 'loan, 'loan>,
    #[cfg(feature = "vector-index")]
    _vectors: InstalledVectorMaintenanceSlots<'target, 'loan, 'loan>,
    #[cfg(feature = "compact-store")]
    _layered: InstalledLayeredCommitSlots<'target, 'loan, 'loan>,
}

impl<'target, 'loan> ReleasedLpgCommit<'target, 'loan> {
    /// Exact label images grouped by their qualified publication store.
    ///
    /// Reads only the retained workspace, without reacquiring any store reader.
    /// Empty unless image capture was requested before preparation.
    pub fn label_images(
        &self,
    ) -> impl Iterator<Item = (&'target LpgStore, &[PreparedNodeLabelImages])> {
        self.data.label_images()
    }

    /// Borrow exact bytes retained in the outer workspace. No index, store or
    /// catalog lock is reacquired after initial preparation. The boolean is
    /// true for a complete private birth and false for a sparse survivor edit.
    #[cfg(feature = "text-index")]
    pub fn text_postimages(&self) -> impl Iterator<Item = (&'target LpgStore, &str, bool, &[u8])> {
        self.registry.fences.stores.iter().flat_map(|store| {
            store.edits.iter().filter_map(move |edit| {
                let super::LocalKey::Text(key) = &edit.key else {
                    return None;
                };
                edit.text_wal
                    .as_ref()
                    .map(|(birth, bytes)| (store.store, key.as_str(), *birth, bytes.as_slice()))
            })
        })
    }

    /// Borrows captured complete replacements and sparse surviving postimages.
    #[cfg(feature = "vector-index")]
    pub fn vector_postimages(
        &self,
    ) -> impl Iterator<Item = (&'target LpgStore, &str, bool, &[u8])> {
        let complete = self.registry.fences.stores.iter().flat_map(|store| {
            store.edits.iter().filter_map(move |edit| {
                let super::LocalKey::Vector(key) = &edit.key else {
                    return None;
                };
                edit.vector_wal
                    .as_ref()
                    .map(|bytes| (store.store, key.as_str(), true, bytes.as_slice()))
            })
        });
        let sparse = self
            .vectors
            .wal_postimages()
            .filter_map(|(ordinal, bytes)| {
                let source = self.vector_sources.get(ordinal)?;
                let ObservedIndex::Vector(key, _) = source.expected.target() else {
                    return None;
                };
                Some((source.store, key.as_str(), false, bytes))
            });
        complete.chain(sparse)
    }

    /// Try-only final binding. The caller must release logical/TM companions
    /// before materializing the static error or invoking ordinary rollback.
    ///
    /// # Errors
    /// Returns a conflict for contention or changed prepared state.
    pub fn rebind(self) -> std::result::Result<PreparedLpgCommit<'target, 'loan>, DataRebindError> {
        let data = self.data.rebind()?;
        #[cfg(feature = "vector-index")]
        let vectors = self.vectors.rebind()?;
        let registry = self.registry.rebind()?;
        #[cfg(feature = "compact-store")]
        let layered = self.layered.rebind()?;
        #[cfg(feature = "vector-index")]
        for source in self.vector_sources {
            let ObservedIndex::Vector(key, _) = source.expected.target() else {
                return Err(DataRebindError::new(
                    "Vector maintenance observation family changed",
                ));
            };
            let publication = registry
                .fences
                .active
                .iter()
                .find(|p| std::ptr::eq(p.store, source.store))
                .ok_or(DataRebindError::new(
                    "Vector registry final fence is absent",
                ))?;
            let current = publication
                .guards
                .vector
                .get(key)
                .ok_or(DataRebindError::Conflict(
                    "Vector registration disappeared at final bind",
                ))?;
            let successor =
                publication
                    .postimages
                    .vector
                    .get(key)
                    .ok_or(DataRebindError::Conflict(
                        "Vector survivor is dropped by this registry batch",
                    ))?;
            if !source.expected.matches_vector(current)
                || !source.expected.matches_vector(successor)
                || !std::ptr::eq(current.payload.as_ref(), source.view.commit_target())
                || publication.guards.slots.get(key).copied() != source.slot
                || publication.postimages.slots.get(key).copied() != source.slot
            {
                return Err(DataRebindError::Conflict(
                    "Vector registration/payload/slot changed at final bind",
                ));
            }
        }
        Ok(PreparedLpgCommit {
            registry,
            data,
            #[cfg(feature = "vector-index")]
            vectors,
            #[cfg(feature = "compact-store")]
            layered,
        })
    }
}

impl<'target, 'loan> PreparedLpgCommit<'target, 'loan> {
    /// Infallible installation into the already-qualified exact target slots.
    pub fn install(mut self) -> InstalledLpgCommit<'target, 'loan> {
        let data = self.data.install();
        #[cfg(feature = "vector-index")]
        let vectors = self.vectors.install();
        self.registry.fences.install();
        #[cfg(feature = "compact-store")]
        let layered = self.layered.install();
        InstalledLpgCommit {
            _registry: self.registry,
            _data: data,
            #[cfg(feature = "vector-index")]
            _vectors: vectors,
            #[cfg(feature = "compact-store")]
            _layered: layered,
        }
    }
}

#[cfg(test)]
mod graph_tests;
#[cfg(test)]
mod tests;
