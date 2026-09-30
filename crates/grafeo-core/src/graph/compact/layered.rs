//! Two-layer graph store: read-only columnar base + mutable LPG overlay.
//!
//! `LayeredStore` coordinates reads between a [`CompactStore`](crate::graph::compact::CompactStore) (cold, columnar)
//! and an [`LpgStore`](crate::graph::lpg::LpgStore) (hot, HashMap-based). All writes go to the overlay.
//! Reads check the overlay first and fall through to the compact base for
//! unmodified entities.
//!
//! Requires both `compact-store` and `lpg` features.

use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};

use arc_swap::ArcSwap;
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, StorageError, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::{Mutex, RwLock};

use super::CompactStore;
use super::{GraphScrub, NodeTableScrub, RelTableScrub, rel_frames_from_edges};
use crate::execution::operators::{ExpressionPredicate, SharedReadTracker, SharedWriteTracker};
use crate::graph::{Direction, PropertyIndexRequest};

/// Epoch+transaction stamp for a base-edge tombstone.
///
/// Mirrors the overlay version chain's delete model
/// ([`VersionInfo`](grafeo_common::mvcc::VersionInfo)) so base-edge deletes are
/// snapshot-isolated: a versioned reader applies the same
/// `deleted_epoch <= viewing_epoch` boundary the overlay uses, instead of the
/// old epoch-blind "any tombstone hides" rule.
///
/// * `epoch == EpochId::PENDING` while the deleting transaction is uncommitted;
///   finalized to the real commit epoch on commit.
/// * `deleter == Some(tx)` for a transactional delete (read-your-writes: hidden
///   to `tx` even while PENDING); `None` for a SYSTEM/auto-commit delete or a
///   persisted-seeded (prior-session committed) delete.
#[derive(Clone, Copy)]
struct BaseEdgeDelete {
    /// Commit epoch of the delete, or [`EpochId::PENDING`] while uncommitted.
    epoch: EpochId,
    /// Transaction that requested the delete, if it was transactional.
    deleter: Option<TransactionId>,
}

/// Epoch+transaction stamp for a base-node tombstone.
///
/// The node-side mirror of [`BaseEdgeDelete`]: gives each base-node delete the
/// same `(epoch, deleter)` stamp so versioned node readers apply the overlay's
/// `deleted_epoch <= viewing_epoch` boundary instead of the old epoch-blind
/// "any tombstone hides" rule. Same semantics for the two fields as
/// [`BaseEdgeDelete`].
#[derive(Clone, Copy)]
struct BaseNodeDelete {
    /// Commit epoch of the delete, or [`EpochId::PENDING`] while uncommitted.
    epoch: EpochId,
    /// Transaction that requested the delete, if it was transactional.
    deleter: Option<TransactionId>,
}
#[cfg(feature = "vector-index")]
use crate::graph::lpg::VisibleVectorReadContext;
use crate::graph::lpg::{
    CompareOp, Edge, LpgStore, Node, PinnedLpgTransition, PinnedNamedGraphTopology,
    PreparedRepresentationTransfer, PublishedRepresentationTransfer, TransportEdgeMutationGrant,
    TransportEdgeReceipt, TransportEdgeState,
};
use crate::graph::traits::{GraphStore, GraphStoreMut, GraphStoreSearch, TxStructuralSnapshot};
#[cfg(feature = "vector-index")]
use crate::index::vector::{
    DistanceMetric, PropertyVectorAccessor, compute_distance, value_to_vector,
};
use crate::statistics::Statistics;

pub(crate) mod commit;

#[cfg(test)]
mod construction_tests;
#[cfg(test)]
mod recovery_labels_tests;

const PROMOTION_LOCK_SHARDS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerationAccess {
    Read,
    Mutation,
    Publication,
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    RecordedIndexRecovery,
}

thread_local! {
    /// Re-entrant per-thread ownership of Layered publication barriers.
    /// Nested graph reads must not reacquire a fair `RwLock` after a publisher
    /// queues, while same-store generation-write re-entry must fail before it
    /// can wait on itself.
    static ACTIVE_GENERATION_SCOPES:
        std::cell::RefCell<Vec<(*const LayeredStore, GenerationAccess)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    /// Exact overlays retained by generation readers on this thread. Raw LPG
    /// reads may reuse the publication barrier, but mutations and maintenance
    /// must still acquire their ordinary overlay authority/barriers.
    static ACTIVE_GENERATION_READ_OVERLAYS: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct GenerationReadScope<'a> {
    store: *const LayeredStore,
    _overlay: Arc<LpgStore>,
    _guard: Option<parking_lot::RwLockReadGuard<'a, ()>>,
}

/// Whether a retained Layered generation already protects this exact overlay
/// read. This is only read coherence, never mutation or maintenance authority.
pub(crate) fn has_retained_overlay_read(store: &LpgStore) -> bool {
    let identity = std::ptr::from_ref(store).addr();
    ACTIVE_GENERATION_READ_OVERLAYS.with_borrow(|active| active.contains(&identity))
}

/// Exact reader loans for startup replay. This is not generation-write
/// authority: publication still requires the exclusive Mutation variant.
#[cfg(any(feature = "text-index", feature = "vector-index"))]
struct RecordedIndexRecoveryScope<'a> {
    previous_depth: usize,
    _publication: parking_lot::RwLockReadGuard<'a, ()>,
    _generation: parking_lot::RwLockReadGuard<'a, ()>,
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
impl<'a> RecordedIndexRecoveryScope<'a> {
    fn enter(store: &'a LayeredStore) -> Result<Self, Error> {
        let pointer = std::ptr::from_ref(store);
        if ACTIVE_GENERATION_SCOPES
            .with_borrow(|active| active.iter().any(|(known, _)| *known == pointer))
        {
            return Err(Error::InvalidValue(
                "recorded index recovery cannot reenter a Layered generation scope".into(),
            ));
        }
        let generation = store.merge_guard.read();
        let publication = store.publication_guard.read();
        let previous_depth = ACTIVE_GENERATION_SCOPES.with_borrow_mut(|active| {
            let depth = active.len();
            active.push((pointer, GenerationAccess::RecordedIndexRecovery));
            depth
        });
        Ok(Self {
            previous_depth,
            _publication: publication,
            _generation: generation,
        })
    }
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
impl Drop for RecordedIndexRecoveryScope<'_> {
    fn drop(&mut self) {
        ACTIVE_GENERATION_SCOPES.with_borrow_mut(|active| active.truncate(self.previous_depth));
    }
}

impl<'a> GenerationReadScope<'a> {
    fn enter(store: &'a LayeredStore) -> Self {
        let store_ptr = std::ptr::from_ref(store);
        let nested = ACTIVE_GENERATION_SCOPES.with(|active| {
            active
                .borrow()
                .iter()
                .any(|(active_store, _)| *active_store == store_ptr)
        });
        let guard = (!nested).then(|| store.publication_guard.read());
        let overlay = store.overlay.load_full();
        ACTIVE_GENERATION_READ_OVERLAYS.with_borrow_mut(|active| {
            active.push(Arc::as_ptr(&overlay).addr());
        });
        ACTIVE_GENERATION_SCOPES.with(|active| {
            active
                .borrow_mut()
                .push((store_ptr, GenerationAccess::Read));
        });
        Self {
            store: store_ptr,
            _overlay: overlay,
            _guard: guard,
        }
    }
}

impl Drop for GenerationReadScope<'_> {
    fn drop(&mut self) {
        ACTIVE_GENERATION_READ_OVERLAYS.with_borrow_mut(|active| {
            active.pop();
        });
        ACTIVE_GENERATION_SCOPES.with(|active| {
            let removed = active.borrow_mut().pop();
            debug_assert_eq!(removed, Some((self.store, GenerationAccess::Read)));
        });
    }
}

struct GenerationWriteScope<'a> {
    store: *const LayeredStore,
    _guard: parking_lot::RwLockWriteGuard<'a, ()>,
}

impl<'a> GenerationWriteScope<'a> {
    fn enter(store: &'a LayeredStore) -> Self {
        let store_ptr = std::ptr::from_ref(store);
        ACTIVE_GENERATION_SCOPES.with(|active| {
            let active = active.borrow();
            assert!(
                active.iter().any(|(active_store, access)| {
                    *active_store == store_ptr && *access == GenerationAccess::Mutation
                }),
                "compact generation publication requires mutation exclusion"
            );
            assert!(
                !active.iter().any(|(active_store, access)| {
                    *active_store == store_ptr && *access != GenerationAccess::Mutation
                }),
                "compact generation publication re-entered a same-store read or publication"
            );
        });
        let guard = store.publication_guard.write();
        ACTIVE_GENERATION_SCOPES.with(|active| {
            active
                .borrow_mut()
                .push((store_ptr, GenerationAccess::Publication));
        });
        Self {
            store: store_ptr,
            _guard: guard,
        }
    }
}

impl Drop for GenerationWriteScope<'_> {
    fn drop(&mut self) {
        ACTIVE_GENERATION_SCOPES.with(|active| {
            let removed = active.borrow_mut().pop();
            debug_assert_eq!(removed, Some((self.store, GenerationAccess::Publication)));
        });
    }
}

struct MutationWriteScope<'a> {
    store: &'a LayeredStore,
    _guard: parking_lot::RwLockWriteGuard<'a, ()>,
}

impl<'a> MutationWriteScope<'a> {
    fn enter(store: &'a LayeredStore) -> Self {
        let store_ptr = std::ptr::from_ref(store);
        ACTIVE_GENERATION_SCOPES.with(|active| {
            assert!(
                !active
                    .borrow()
                    .iter()
                    .any(|(active_store, _)| *active_store == store_ptr),
                "compact generation callback re-entered the same LayeredStore"
            );
        });
        let guard = store.merge_guard.write();
        ACTIVE_GENERATION_SCOPES.with(|active| {
            active
                .borrow_mut()
                .push((store_ptr, GenerationAccess::Mutation));
        });
        Self {
            store,
            _guard: guard,
        }
    }

    fn checked_revision(&mut self) -> Result<CheckedLayerRevision<'_, 'a>, String> {
        let previous = self.store.generation_revision.load(Ordering::Acquire);
        let next = previous
            .checked_add(1)
            .ok_or_else(|| "LayeredStore generation revision exhausted".to_owned())?;
        Ok(CheckedLayerRevision {
            mutation: self,
            previous,
            next,
        })
    }
}

impl Drop for MutationWriteScope<'_> {
    fn drop(&mut self) {
        ACTIVE_GENERATION_SCOPES.with(|active| {
            let removed = active.borrow_mut().pop();
            debug_assert_eq!(
                removed,
                Some((std::ptr::from_ref(self.store), GenerationAccess::Mutation))
            );
        });
    }
}

/// Pins the mutable generation for a history/classification traversal.
///
/// A generation-preparation callback already owns the exclusive mutation side,
/// so same-thread read-only traversal is safe and must not recursively acquire
/// the non-reentrant lock. Conversely, acquiring mutation-read after entering a
/// publication callback would invert the global mutation-then-publication order;
/// reject that misuse promptly instead of allowing a queued writer to deadlock
/// both sides.
struct MutationReadScope<'a> {
    _guard: Option<parking_lot::RwLockReadGuard<'a, ()>>,
}

impl<'a> MutationReadScope<'a> {
    fn enter(store: &'a LayeredStore) -> Self {
        let store_ptr = std::ptr::from_ref(store);
        let owns_recovery = store.recorded_index_recovery_active();
        let (owns_mutation, publication_only) = ACTIVE_GENERATION_SCOPES.with(|active| {
            let active = active.borrow();
            let owns_mutation = owns_recovery
                || active.iter().any(|(active_store, access)| {
                    *active_store == store_ptr && *access == GenerationAccess::Mutation
                });
            let publication_only = !owns_mutation
                && active
                    .iter()
                    .any(|(active_store, _)| *active_store == store_ptr);
            (owns_mutation, publication_only)
        });
        assert!(
            !publication_only,
            "mutation-pinned Layered read re-entered a publication callback"
        );
        Self {
            _guard: (!owns_mutation).then(|| store.merge_guard.read()),
        }
    }
}

/// A complete off-side Layered generation ready for allocation-free install.
///
/// Every field participating in read routing travels with the base and overlay
/// pointers. Publication merely swaps owned values while the short generation
/// barrier is exclusive; construction, replay, and allocation happen before
/// that barrier is acquired.
struct PreparedLayerGeneration {
    base: Arc<CompactStore>,
    overlay: Arc<LpgStore>,
    dirty_node_ids: FxHashSet<NodeId>,
    dirty_edge_ids: FxHashSet<EdgeId>,
    deleted_from_base_nodes: FxHashMap<NodeId, BaseNodeDelete>,
    pending_base_node_deletes: FxHashMap<TransactionId, Vec<NodeId>>,
    deleted_from_base_edges: FxHashMap<EdgeId, BaseEdgeDelete>,
    pending_base_edge_deletes: FxHashMap<TransactionId, Vec<EdgeId>>,
    deletions_dirty: bool,
    /// Exact runtime-registry handoff prepared under the named topology and
    /// source LPG transition cuts. Property-index definitions are already
    /// installed on the unpublished successor by the same preparation.
    representation_transfer: Option<PreparedRepresentationTransfer>,
}

/// Keeps admission failures structured for native conversion without changing
/// the existing temporal merge methods' string error contract.
enum TemporalMergeFailure {
    Admission(&'static str),
    Preparation(String),
}

impl TemporalMergeFailure {
    fn into_error(self) -> Error {
        match self {
            Self::Admission(reason) => TransactionError::InvalidState(reason.to_owned()).into(),
            Self::Preparation(reason) => Error::Internal(reason),
        }
    }

    fn into_reason(self) -> String {
        match self {
            Self::Admission(reason) => reason.to_owned(),
            Self::Preparation(reason) => reason,
        }
    }
}

impl PreparedLayerGeneration {
    fn empty(base: Arc<CompactStore>, overlay: Arc<LpgStore>) -> Self {
        Self {
            base,
            overlay,
            dirty_node_ids: FxHashSet::default(),
            dirty_edge_ids: FxHashSet::default(),
            deleted_from_base_nodes: FxHashMap::default(),
            pending_base_node_deletes: FxHashMap::default(),
            deleted_from_base_edges: FxHashMap::default(),
            pending_base_edge_deletes: FxHashMap::default(),
            deletions_dirty: false,
            representation_transfer: None,
        }
    }
}

/// Keeps the publication barrier closed until a cooperating subsystem either
/// commits or restores the exact prior Layered generation.
///
/// The guard is declared first deliberately: on commit, it is released before
/// retired maps and stores are dropped, keeping reclamation off the read cut.
struct PublishedLayerGeneration<'cut, 'store> {
    _publication: GenerationWriteScope<'store>,
    previous: PreparedLayerGeneration,
    revision: CheckedLayerRevision<'cut, 'store>,
    representation_transfer: Option<PublishedRepresentationTransfer>,
}

/// Joint Layered/external publication retained by an LPG transition seam until
/// its prebuilt commit has crossed the hostile unwind boundary. Success
/// explicitly extracts the external retirement token and releases the Layered
/// publication cut; rollback explicitly restores both generations. Dropping
/// this wrapper is never used as a commit protocol.
struct ReversibleLayerPublication<'cut, 'store, R> {
    external: R,
    layered: PublishedLayerGeneration<'cut, 'store>,
}

/// A one-shot loan of the actual mutation cut, not independent authority.
struct CheckedLayerRevision<'cut, 'store> {
    mutation: &'cut mut MutationWriteScope<'store>,
    previous: u64,
    next: u64,
}

struct ReadyLayerPublication<'cut, 'store, P> {
    revision: CheckedLayerRevision<'cut, 'store>,
    image: PreparedLayerGeneration,
    external: P,
}

/// Proof-free retirement: transfer fences drain before image/external payloads.
struct UnpublishedLayerRetirement<P> {
    _transfer: Option<PreparedRepresentationTransfer>,
    _image: PreparedLayerGeneration,
    _external: P,
}

impl<P> UnpublishedLayerRetirement<P> {
    fn new(mut image: PreparedLayerGeneration, external: P) -> Self {
        Self {
            _transfer: image.representation_transfer.take(),
            _image: image,
            _external: external,
        }
    }
}

impl<P> ReadyLayerPublication<'_, '_, P> {
    fn into_retirement(self) -> UnpublishedLayerRetirement<P> {
        let Self {
            revision: _,
            image,
            external,
        } = self;
        UnpublishedLayerRetirement::new(image, external)
    }
}

/// Exact payload ownership persists even if another admitted writer clears a
/// registry after the structural cuts drain. Views grant no mutation authority.
struct GenerationRetirementAnchors {
    #[cfg(feature = "text-index")]
    _text: Vec<(String, crate::index::text::TextIndexView)>,
    #[cfg(feature = "vector-index")]
    _vector: Vec<(String, crate::index::vector::VectorIndexView)>,
    _source: Arc<LpgStore>,
    _successor: Arc<LpgStore>,
}

impl GenerationRetirementAnchors {
    fn capture(source: &Arc<LpgStore>, successor: &Arc<LpgStore>) -> Self {
        Self {
            #[cfg(feature = "text-index")]
            _text: source.text_index_entries(),
            #[cfg(feature = "vector-index")]
            _vector: source.vector_index_entries(),
            _source: Arc::clone(source),
            _successor: Arc::clone(successor),
        }
    }
}

/// Declared before the outer MutationWriteScope. Each optional sink has one
/// mutually exclusive producer per attempt; no payload is recovered from it
/// on a successful handoff. Field order releases ALL transfer fences before
/// external state and retains exact source/target anchors until last.
struct GenerationRetirements<P, Q> {
    transfer: std::cell::RefCell<Option<PreparedRepresentationTransfer>>,
    unpublished: std::cell::RefCell<Option<UnpublishedLayerRetirement<P>>>,
    partial: std::cell::RefCell<Option<(Arc<CompactStore>, P)>>,
    rollback_layer: std::cell::RefCell<Option<PreparedLayerGeneration>>,
    rollback_external: std::cell::RefCell<Option<Q>>,
    anchors: std::cell::RefCell<Option<GenerationRetirementAnchors>>,
}

impl<P, Q> GenerationRetirements<P, Q> {
    fn new() -> Self {
        Self {
            transfer: std::cell::RefCell::new(None),
            unpublished: std::cell::RefCell::new(None),
            partial: std::cell::RefCell::new(None),
            rollback_layer: std::cell::RefCell::new(None),
            rollback_external: std::cell::RefCell::new(None),
            anchors: std::cell::RefCell::new(None),
        }
    }
}

/// Already-held structural cuts and rollback sinks for one empty-overlay
/// publication. `representation_topology = None` deliberately means a fresh
/// reset: the discarded overlay's derived registries stay with that retained
/// snapshot instead of being attached to a successor that has none of its
/// rows. Merge/recompact pass the exact named-topology proof and transfer the
/// registries with their newly compacted rows.
struct PinnedEmptySuccessorPublication<'a, 'store, 'layered, P, Q> {
    current_overlay: &'a Arc<LpgStore>,
    representation_topology: Option<&'a PinnedNamedGraphTopology>,
    transition: &'a PinnedLpgTransition<'store>,
    mutation: &'a mut MutationWriteScope<'layered>,
    retirements: &'a GenerationRetirements<P, Q>,
}

#[cfg(test)]
type PromotionPublicationHook = dyn Fn() + Send + Sync;

fn promotion_lock_shard(id: u64) -> usize {
    let shard_count =
        u64::try_from(PROMOTION_LOCK_SHARDS).expect("promotion lock shard count always fits u64");
    usize::try_from(id % shard_count).expect("promotion lock shard always fits usize")
}

/// A two-layer graph store with a columnar base and mutable overlay.
///
/// The compact base serves cold reads (immutable, columnar). The LPG overlay
/// captures all mutations. Reads check the overlay first: if an entity is in
/// `dirty_node_ids` or `dirty_edge_ids`, the overlay is authoritative. If an
/// entity is in `deleted_from_base_nodes` or `deleted_from_base_edges`, it has
/// been deleted and returns `None`. Otherwise, the base is queried.
pub struct LayeredStore {
    /// Read-only columnar base (cold data).
    ///
    /// Held via [`ArcSwap`] so an already-pinned reader can retain the exact
    /// `Arc<CompactStore>` while the engine replaces its generation. Layered
    /// reads first take the shared publication barrier because base, overlay,
    /// and routing maps form one coherent logical generation; the ArcSwap then
    /// makes acquisition and retirement of the pinned base cheap.
    base: ArcSwap<CompactStore>,
    /// Mutable overlay for new and modified data.
    ///
    /// Held via [`ArcSwap`] (Phase 5c) so readers that have pinned a coherent
    /// Layered generation can retain its overlay while a successor is retired.
    /// Whole-generation replacement publishes this pointer together with the
    /// base and routing maps behind [`Self::publication_guard`].
    overlay: ArcSwap<LpgStore>,
    /// Node IDs modified or created in the overlay.
    dirty_node_ids: RwLock<FxHashSet<NodeId>>,
    /// Edge IDs modified or created in the overlay.
    dirty_edge_ids: RwLock<FxHashSet<EdgeId>>,
    /// Striped single-flight ownership for cold-node promotion. Two writers for
    /// one node share a shard and recheck `dirty_node_ids` under this lock, so
    /// exactly one hydrates and publishes the overlay row. Unrelated shards can
    /// promote concurrently, and read paths never acquire these locks.
    node_promotion_locks: [Mutex<()>; PROMOTION_LOCK_SHARDS],
    /// Edge-side promotion ownership. Kept separate from node stripes because
    /// edge hydration promotes its endpoints while holding the edge stripe;
    /// the one-way edge-to-node lock order cannot self-deadlock.
    edge_promotion_locks: [Mutex<()>; PROMOTION_LOCK_SHARDS],
    /// Deterministic race seam used only by same-node promotion tests. Every
    /// contender waits after the optimistic dirty check and before ownership,
    /// proving the inner recheck rather than relying on scheduler timing.
    #[cfg(test)]
    node_promotion_barrier: RwLock<Option<Arc<std::sync::Barrier>>>,
    /// Edge-side counterpart of `node_promotion_barrier`.
    #[cfg(test)]
    edge_promotion_barrier: RwLock<Option<Arc<std::sync::Barrier>>>,
    /// Test-only freeze point after a node is fully hydrated but immediately
    /// before `dirty_node_ids` publishes it to LayeredStore readers.
    #[cfg(test)]
    node_publication_hook: RwLock<Option<Arc<PromotionPublicationHook>>>,
    /// Edge-side counterpart of `node_publication_hook`.
    #[cfg(test)]
    edge_publication_hook: RwLock<Option<Arc<PromotionPublicationHook>>>,
    /// Test-only per-event seam for observing an incomplete node-history replay.
    #[cfg(test)]
    node_replay_event_hook: RwLock<Option<Arc<PromotionPublicationHook>>>,
    /// Edge-side counterpart of `node_replay_event_hook`.
    #[cfg(test)]
    edge_replay_event_hook: RwLock<Option<Arc<PromotionPublicationHook>>>,
    /// Test-only freeze point after structural-restore authority validation.
    #[cfg(test)]
    structural_restore_hook: RwLock<Option<Arc<PromotionPublicationHook>>>,
    /// Test-only observation point after transport tombstones are unlocked and
    /// before the corresponding dirty-edge IDs are locked for removal.
    #[cfg(test)]
    transport_cleanup_between_lock_scopes_hook: RwLock<Option<Arc<PromotionPublicationHook>>>,
    /// Deterministic test-only pause after base/overlay pointer installation
    /// and before routing-map installation. Readers must remain behind the
    /// publication barrier for both waits.
    #[cfg(test)]
    generation_publication_barrier: RwLock<Option<Arc<std::sync::Barrier>>>,
    /// Deterministic test-only pause after named-topology and source-LPG
    /// exclusion are both acquired, before temporal snapshot traversal.
    #[cfg(test)]
    temporal_snapshot_barrier: RwLock<Option<Arc<std::sync::Barrier>>>,
    /// After a real retained-hot node has been reseeded: 1 rejects, 2 unwinds.
    #[cfg(test)]
    generation_reseed_action: std::sync::atomic::AtomicU8,
    #[cfg(test)]
    generation_reseed_hits: AtomicUsize,
    /// Hostile final validation after the complete owned pair exists.
    #[cfg(test)]
    generation_image_validation_panic: AtomicBool,
    /// Number of full compact structural-inventory traversals, used to prove
    /// bulk transport paths do not rescan the cold tier per edge.
    #[cfg(test)]
    base_edge_inventory_scans: AtomicUsize,
    /// Base node IDs that have been deleted, each stamped with the epoch and
    /// transaction of the delete so versioned readers stay snapshot-isolated.
    /// Node-side mirror of [`Self::deleted_from_base_edges`] — see that field
    /// for the `PENDING`/`deleter` semantics.
    deleted_from_base_nodes: RwLock<FxHashMap<NodeId, BaseNodeDelete>>,
    /// Per-transaction list of base node ids this transaction has PENDING-
    /// tombstoned, so commit can finalize their epochs and rollback can remove
    /// them. Node-side mirror of [`Self::pending_base_edge_deletes`].
    pending_base_node_deletes: RwLock<FxHashMap<TransactionId, Vec<NodeId>>>,
    /// Base edge IDs that have been deleted, each stamped with the epoch and
    /// transaction of the delete so versioned readers stay snapshot-isolated.
    ///
    /// A `PENDING` entry with `deleter == Some(tx)` is an uncommitted
    /// transactional delete: hidden to `tx` (read-your-writes) but still visible
    /// to every other snapshot until commit finalizes its epoch. Mirrors the
    /// overlay's `pending_tx_edge_deletes` + version-chain model.
    deleted_from_base_edges: RwLock<FxHashMap<EdgeId, BaseEdgeDelete>>,
    /// Per-transaction list of base edge ids this transaction has PENDING-
    /// tombstoned, so commit can finalize their epochs and rollback can remove
    /// them. A base-only edge delete does not reach the overlay's
    /// `pending_tx_edge_deletes` (the overlay has no such edge), so the
    /// LayeredStore must track its own base tombstones to drive finalize/rollback.
    pending_base_edge_deletes: RwLock<FxHashMap<TransactionId, Vec<EdgeId>>>,
    /// Tracks whether `deleted_from_base_*` has changed since the last
    /// flush. The `OverlayDeletionsSection` checks this on each
    /// `is_dirty()` call so periodic checkpoints skip the write when no
    /// new deletions have accumulated.
    deletions_dirty: AtomicBool,
    /// Mutation/rebuild exclusion guard (Phase 5d).
    ///
    /// Mutations acquire `read()` for the duration of a single
    /// operation; merge preparation acquires `write()` to stop writers while
    /// it builds a complete successor generation off-side. Layered reads do
    /// not take this lock and continue against the currently published state.
    /// Prevents the race where concurrent writes land on an overlay
    /// that's about to be cleared, losing those writes.
    merge_guard: RwLock<()>,
    /// Short coherent-publication barrier.
    ///
    /// Every generation-sensitive Layered read takes the shared side. The
    /// exclusive side covers only allocation-free base/overlay/routing-map
    /// publication; rebuild, replay, serialization, and mmap preparation must
    /// complete before it is acquired. Re-entrant reads use the thread-local
    /// scope above so a queued publisher cannot deadlock nested graph calls.
    publication_guard: RwLock<()>,
    /// Monotonic identity of the published base/overlay/routing generation.
    /// Representation preparation outside the publication barrier revalidates
    /// this token before its final infallible install.
    generation_revision: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for LayeredStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _generation = GenerationReadScope::enter(self);
        f.debug_struct("LayeredStore")
            .field("base_node_count", &self.base.load().node_count())
            .field("overlay_node_count", &self.overlay.load().node_count())
            .field("dirty_nodes", &self.dirty_node_ids.read().len())
            .field(
                "deleted_base_nodes",
                &self.deleted_from_base_nodes.read().len(),
            )
            .finish_non_exhaustive()
    }
}

/// Filters a node's per-property history to the committed frontier — versions at
/// or before `boundary` — dropping uncommitted (PENDING, epoch > boundary) ones
/// so they are never folded into the cold base. Empty-after-filter properties
/// are dropped.
fn committed_history(
    history: Vec<(PropertyKey, Vec<(EpochId, Value)>)>,
    boundary: EpochId,
) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
    history
        .into_iter()
        .filter_map(|(key, versions)| {
            let kept: Vec<(EpochId, Value)> = versions
                .into_iter()
                .filter(|(e, _)| *e <= boundary)
                .collect();
            (!kept.is_empty()).then_some((key, kept))
        })
        .collect()
}

/// Merges a colder property log with an authoritative overlay log without
/// collapsing ordered same-epoch mutations inside either source. When both
/// layers carry entries at the same epoch, the overlay owns that epoch (the
/// usual promotion/recovery overlap), so only the colder entries for those
/// epochs are removed before a stable epoch sort.
fn merge_property_versions(base: &mut Vec<(EpochId, Value)>, overlay: Vec<(EpochId, Value)>) {
    let overlay_epochs: FxHashSet<EpochId> = overlay.iter().map(|(epoch, _)| *epoch).collect();
    base.retain(|(epoch, _)| !overlay_epochs.contains(epoch));
    base.extend(overlay);
    base.sort_by_key(|(epoch, _)| *epoch);
}

/// Last authoritative value at `epoch`. A stable history may contain several
/// writes at one epoch; walking backwards preserves the final write.
fn history_value_at(versions: &[(EpochId, Value)], epoch: EpochId) -> Option<&Value> {
    versions
        .iter()
        .rev()
        .find(|(version_epoch, _)| *version_epoch <= epoch)
        .map(|(_, value)| value)
}

/// Compares logical label sets rather than their insertion order. Duplicate
/// labels are not a distinct LPG state, so a repeated recovery record carrying
/// them remains the same structural create.
fn recovery_labels_match(actual: &[ArcStr], expected: &[&str]) -> bool {
    let mut actual: Vec<&str> = actual.iter().map(ArcStr::as_str).collect();
    actual.sort_unstable();
    actual.dedup();

    let mut expected = expected.to_vec();
    expected.sort_unstable();
    expected.dedup();
    actual == expected
}

/// Appends an overlay node (created since the last merge, so absent from the
/// cold base) to its label's columnar scrub frame, keeping every column the
/// same length: existing columns get this node's value (or `None`), and a
/// property new to the frame gets a fresh column back-filled with `None`.
fn append_scrub_node(
    frames: &mut Vec<NodeTableScrub>,
    label: &str,
    nid: NodeId,
    props: &FxHashMap<PropertyKey, Value>,
) {
    let frame = if let Some(idx) = frames.iter().position(|f| f.label.as_str() == label) {
        &mut frames[idx]
    } else {
        frames.push(NodeTableScrub {
            label: ArcStr::from(label),
            node_ids: Vec::new(),
            columns: FxHashMap::default(),
        });
        frames.last_mut().expect("just pushed a frame")
    };
    let row = frame.node_ids.len();
    frame.node_ids.push(nid);
    for (key, values) in &mut frame.columns {
        values.push(props.get(key).cloned());
    }
    for (key, value) in props {
        if !frame.columns.contains_key(key) {
            let mut column = vec![None; row];
            column.push(Some(value.clone()));
            frame.columns.insert(key.clone(), column);
        }
    }
}

/// Removes every occurrence of a node while keeping each property column
/// aligned with the frame's `node_ids` vector.
fn remove_scrub_node(frames: &mut [NodeTableScrub], nid: NodeId) {
    for frame in frames {
        while let Some(row) = frame.node_ids.iter().position(|id| *id == nid) {
            frame.node_ids.remove(row);
            for column in frame.columns.values_mut() {
                if row < column.len() {
                    column.remove(row);
                }
            }
        }
    }
}

impl LayeredStore {
    /// Runs one composite Layered mutation against a single authorized overlay
    /// incarnation.
    ///
    /// `merge_guard` prevents a Layered generation replacement while the LPG
    /// proof pins the opposing store-transition side and validates the exact
    /// sealed write scope once for the whole operation. Nested LPG mutators use
    /// recursive shared pins. No Layered routing or tombstone state may be
    /// published outside this closure.
    fn with_pinned_overlay_mutation<R>(
        &self,
        rejected: impl FnOnce() -> R,
        mutate: impl FnOnce(&Arc<LpgStore>) -> R,
    ) -> R {
        let _generation = (!self.recorded_index_recovery_active()).then(|| self.merge_guard.read());
        let overlay = self.overlay.load_full();
        let Some(_mutation) = overlay.pin_mutation() else {
            return rejected();
        };
        mutate(&overlay)
    }

    fn recorded_index_recovery_active(&self) -> bool {
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        {
            let pointer = std::ptr::from_ref(self);
            ACTIVE_GENERATION_SCOPES.with_borrow(|active| {
                active.iter().any(|(known, access)| {
                    *known == pointer && *access == GenerationAccess::RecordedIndexRecovery
                })
            })
        }
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        {
            false
        }
    }

    /// Replays data while preserving separately recorded index postimages.
    ///
    /// Pins this generation before admitting the exact unsealed overlay's
    /// exclusive startup scope. The caller must install the selected families'
    /// recorded changes before exposing the store. Property maintenance and
    /// authoritative histories are unaffected; no generation or registry is
    /// replaced by this wrapper.
    ///
    /// # Errors
    /// Rejects unavailable requested families, nested generation scopes and
    /// sealed or retired overlay targets; propagates the operation's error.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    #[doc(hidden)]
    pub fn with_recorded_index_recovery<T>(
        &self,
        record_text: bool,
        record_vector: bool,
        operation: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        let _generation = RecordedIndexRecoveryScope::enter(self)?;
        let overlay = self.overlay.load_full();
        overlay.with_recorded_index_recovery(record_text, record_vector, operation)
    }

    /// Runs one base-generation publication while the exact current overlay's
    /// mutation scope is pinned and authorized.
    ///
    /// Base replacement changes the authoritative Layered graph even though it
    /// does not call an LPG mutator. Taking only `merge_guard` would therefore
    /// let a downstream caller bypass a sealed overlay's write authority. The
    /// lock order is the same as reset/merge: first exclude Layered mutations
    /// and generation replacement, then pin the current LPG incarnation. The
    /// private LPG proof is retained across preparation, publication, callback
    /// unwind, and guard release; it cannot be forged by downstream code.
    fn with_pinned_base_transition<R>(&self, transition: impl FnOnce() -> R) -> R {
        let mutations = MutationWriteScope::enter(self);
        let overlay = self.overlay.load_full();
        let mutation = overlay.pin_mutation().unwrap_or_else(|| {
            panic!("compact base generation transition requires the current overlay write scope")
        });
        let result = transition();
        drop(mutation);
        drop(overlay);
        drop(mutations);
        result
    }

    /// Returns the first node and edge ids that are strictly above every
    /// identity retained by the cold base.
    ///
    /// Closed temporal edges are not part of the current CSR, but their ids
    /// remain addressable by as-of reads and therefore must still reserve
    /// allocator space. `live_original_edge_ids` and `closed_edge_ids`
    /// together cover both cold edge populations.
    fn cold_allocator_floors(base: &CompactStore) -> (u64, u64) {
        let next_node_id = base
            .node_ids()
            .into_iter()
            .chain(base.temporal_node_ids())
            .filter(NodeId::is_valid)
            .map(|id| id.as_u64().saturating_add(1))
            .max()
            .unwrap_or(0);
        let next_edge_id = base
            .live_original_edge_ids()
            .into_iter()
            .chain(base.closed_edge_ids())
            .filter(EdgeId::is_valid)
            .map(|id| id.as_u64().saturating_add(1))
            .max()
            .unwrap_or(0);
        (next_node_id, next_edge_id)
    }

    /// Raises an adopted overlay's allocator counters to the cold-base floors
    /// without discarding a larger high-water mark restored with the overlay.
    fn raise_overlay_allocator_floors(base: &CompactStore, overlay: &LpgStore) {
        let (cold_node_floor, cold_edge_floor) = Self::cold_allocator_floors(base);
        overlay.set_next_node_id(overlay.next_node_id().max(cold_node_floor));
        overlay.set_next_edge_id(overlay.next_edge_id().max(cold_edge_floor));
    }

    /// Creates a layered store from a compact base.
    ///
    /// The `max_node_id` and `max_edge_id` values seed the overlay's ID
    /// allocator so new entities never collide with base IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the overlay cannot be created or either supplied
    /// maximum ID cannot be advanced without wrapping. A last valid ID may
    /// leave its allocator exhausted; it must never restart at zero.
    pub fn new(base: CompactStore, max_node_id: u64, max_edge_id: u64) -> Result<Self, Error> {
        let node_floor = max_node_id.checked_add(1).ok_or_else(|| {
            Error::InvalidValue("compact maximum node ID cannot be advanced".to_owned())
        })?;
        let edge_floor = max_edge_id.checked_add(1).ok_or_else(|| {
            Error::InvalidValue("compact maximum edge ID cannot be advanced".to_owned())
        })?;
        let overlay = Arc::new(LpgStore::new()?);
        overlay.set_next_node_id(node_floor);
        overlay.set_next_edge_id(edge_floor);
        Self::with_overlay(Arc::new(base), overlay)
    }

    /// Adopts an existing `Arc<LpgStore>` as the overlay rather than allocating
    /// a fresh one. Used by the open path when reloading a compacted
    /// database whose overlay data was just deserialized into the engine's
    /// `LpgStore`.
    ///
    /// Scans the overlay against the base to reconstruct
    /// `dirty_node_ids` / `dirty_edge_ids`: any id that exists in both
    /// layers is a base modification whose overlay copy must take
    /// precedence in `get_node` / `get_edge`. Without this reseed,
    /// `get_node` would route through the dirty-check fast path,
    /// observe an empty set, and return the stale base version of any
    /// modified base node.
    ///
    /// **Note on deletions:** `deleted_from_base_nodes` /
    /// `deleted_from_base_edges` are NOT reconstructable from the
    /// in-memory state alone — a base node that was deleted simply has
    /// no overlay entry, so the scan can't tell it apart from a base
    /// node that was never touched. The deletion log is persisted in
    /// the [`OverlayDeletions`](grafeo_common::storage::section::SectionType::OverlayDeletions)
    /// section; callers should follow `with_overlay` with a call to
    /// [`seed_deleted_from_base`](Self::seed_deleted_from_base) carrying
    /// the snapshot read from that section, when one is present in the
    /// container directory.
    ///
    /// The overlay's persisted allocator state is preserved when it is already
    /// higher than the cold-base high-water marks. Otherwise it is raised above
    /// every cold node and every live or closed temporal edge identity. This is
    /// required when a compact database reopens with a genuinely empty overlay,
    /// whose serialized counters otherwise restart at zero.
    ///
    /// The adopted LPG representation is bound to this exact compact
    /// generation before allocator state is changed. Physical HNSW
    /// indexes store topology rather than duplicate vectors; that construction
    /// invariant lets a restored index resolve cold vectors while a later hot
    /// mutation reconnects its topology.
    ///
    /// # Errors
    ///
    /// Returns an error when `overlay` is retired, its write authority is not
    /// held, or it already belongs to another compact generation. Reusing one
    /// physical LPG representation across two layered
    /// stores would give index maintenance an ambiguous cold property tier and
    /// is therefore rejected before either store can be published.
    ///
    /// ```compile_fail
    /// # use std::sync::Arc;
    /// # use grafeo_core::graph::compact::CompactStoreBuilder;
    /// # use grafeo_core::graph::compact::layered::LayeredStore;
    /// # use grafeo_core::graph::lpg::LpgStore;
    /// let base = Arc::new(CompactStoreBuilder::new().build().unwrap());
    /// let overlay = Arc::new(LpgStore::new().unwrap());
    /// LayeredStore::try_with_overlay(base, overlay);
    /// ```
    pub fn with_overlay(base: Arc<CompactStore>, overlay: Arc<LpgStore>) -> Result<Self, Error> {
        // The owner and its cold anchor outlive the borrowed transition on
        // rejection. Only a complete candidate may claim the one-shot slot.
        let mut candidate = Self::from_parts(Arc::clone(&base), Arc::clone(&overlay));
        let (node_floor, edge_floor) = Self::cold_allocator_floors(&base);
        let Some(transition) = overlay.pin_exclusive_unframed_transition() else {
            return Err(TransactionError::InvalidState(
                "compact overlay adoption requires an active representation and its write authority"
                    .to_owned(),
            )
            .into());
        };
        if let Err(reason) = transition.require_native_compact_source() {
            drop(transition);
            return Err(TransactionError::InvalidState(reason.to_owned()).into());
        }
        let mut dirty_nodes: FxHashSet<NodeId> = FxHashSet::default();
        for nid in overlay.all_node_ids() {
            if Self::base_has_node_identity(&base, nid) {
                dirty_nodes.insert(nid);
            }
        }
        let mut dirty_edges: FxHashSet<EdgeId> = FxHashSet::default();
        for edge in overlay.all_edges() {
            if Self::base_has_edge_identity(&base, edge.id) {
                dirty_edges.insert(edge.id);
            }
        }

        candidate.dirty_node_ids = RwLock::new(dirty_nodes);
        candidate.dirty_edge_ids = RwLock::new(dirty_edges);
        let binding = transition.bind_compact_base(Arc::clone(&base), node_floor, edge_floor);
        drop(transition);
        binding.map_err(|reason| TransactionError::InvalidState(reason.to_owned()))?;
        Ok(candidate)
    }

    /// Converts one active native representation into a temporal layered store.
    ///
    /// The source remains native and mutable if preparation fails. An empty
    /// base is used only inside this construction; no unbound Layered store is
    /// returned. The existing generation transfer binds and publishes a fresh
    /// successor, preserving the source's exact history and runtime indexes.
    /// The caller must establish a quiescent committed frontier: pending
    /// transactions are not transferred by this maintenance operation.
    ///
    /// # Errors
    ///
    /// Rejects a retired or already adopted source, missing write authority,
    /// or a failure to prepare the temporal representation.
    #[doc(hidden)]
    pub fn from_native_temporal(overlay: Arc<LpgStore>) -> Result<Self, Error> {
        Self::from_native_temporal_with_prepare(overlay, Ok)
    }

    fn from_native_temporal_with_prepare(
        overlay: Arc<LpgStore>,
        prepare: impl FnOnce(Arc<CompactStore>) -> Result<Arc<CompactStore>, String>,
    ) -> Result<Self, Error> {
        let empty = super::CompactStoreBuilder::new()
            .build()
            .map_err(|error| Error::Internal(error.to_string()))?;
        let staging = Self::from_parts(Arc::new(empty), overlay);
        staging
            .merge_overlay_temporal_retaining_with_publication(
                None,
                true,
                |candidate| prepare(candidate).map(|base| (base, ())),
                |()| (),
                |()| (),
            )
            .map_err(TemporalMergeFailure::into_error)?;
        Ok(staging)
    }

    fn from_parts(base: Arc<CompactStore>, overlay: Arc<LpgStore>) -> Self {
        Self {
            base: ArcSwap::new(base),
            overlay: ArcSwap::new(overlay),
            dirty_node_ids: RwLock::new(FxHashSet::default()),
            dirty_edge_ids: RwLock::new(FxHashSet::default()),
            node_promotion_locks: std::array::from_fn(|_| Mutex::new(())),
            edge_promotion_locks: std::array::from_fn(|_| Mutex::new(())),
            #[cfg(test)]
            node_promotion_barrier: RwLock::new(None),
            #[cfg(test)]
            edge_promotion_barrier: RwLock::new(None),
            #[cfg(test)]
            node_publication_hook: RwLock::new(None),
            #[cfg(test)]
            edge_publication_hook: RwLock::new(None),
            #[cfg(test)]
            node_replay_event_hook: RwLock::new(None),
            #[cfg(test)]
            edge_replay_event_hook: RwLock::new(None),
            #[cfg(test)]
            structural_restore_hook: RwLock::new(None),
            #[cfg(test)]
            transport_cleanup_between_lock_scopes_hook: RwLock::new(None),
            #[cfg(test)]
            generation_publication_barrier: RwLock::new(None),
            #[cfg(test)]
            temporal_snapshot_barrier: RwLock::new(None),
            #[cfg(test)]
            generation_reseed_action: std::sync::atomic::AtomicU8::new(0),
            #[cfg(test)]
            generation_reseed_hits: AtomicUsize::new(0),
            #[cfg(test)]
            generation_image_validation_panic: AtomicBool::new(false),
            #[cfg(test)]
            base_edge_inventory_scans: AtomicUsize::new(0),
            deleted_from_base_nodes: RwLock::new(FxHashMap::default()),
            pending_base_node_deletes: RwLock::new(FxHashMap::default()),
            deleted_from_base_edges: RwLock::new(FxHashMap::default()),
            pending_base_edge_deletes: RwLock::new(FxHashMap::default()),
            deletions_dirty: AtomicBool::new(false),
            merge_guard: RwLock::new(()),
            publication_guard: RwLock::new(()),
            generation_revision: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Captures every transaction-local structural queue against one pinned
    /// overlay generation. The caller holds `merge_guard` for the complete
    /// capture/validation/restore operation so the overlay and the layered
    /// base-tombstone maps cannot be reset between those phases.
    fn tx_structural_snapshot_for_overlay(
        &self,
        transaction_id: TransactionId,
        overlay: &LpgStore,
    ) -> TxStructuralSnapshot {
        let mut snapshot = overlay.tx_structural_snapshot(transaction_id);
        snapshot.base_node_deletes = self
            .pending_base_node_deletes
            .read()
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default();
        snapshot.base_edge_deletes = self
            .pending_base_edge_deletes
            .read()
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default();
        snapshot
    }

    /// Returns a shared reference to the compact base store.
    #[must_use]
    pub fn base_store_arc(&self) -> Arc<CompactStore> {
        let _generation = GenerationReadScope::enter(self);
        self.base.load_full()
    }

    /// Runs a short observation while the exact base pointer and Layered
    /// generation revision are pinned together.
    ///
    /// Tier metadata readers use this to compare their backing state with the
    /// graph-authoritative base without a generation swap between the two
    /// observations. The callback must remain read-only and short, and should
    /// inspect only the supplied base/revision plus cooperating external
    /// metadata. Re-entering Layered operations that acquire the mutation cut
    /// is rejected rather than inverting the mutation-then-publication lock
    /// order and self-deadlocking.
    #[doc(hidden)]
    pub fn with_base_generation<R>(&self, inspect: impl FnOnce(Arc<CompactStore>, u64) -> R) -> R {
        let _generation = GenerationReadScope::enter(self);
        let revision = self.generation_revision.load(Ordering::Acquire);
        inspect(self.base.load_full(), revision)
    }

    /// Installs a fully prepared generation with no allocation or fallible
    /// work inside the coherent read cut.
    ///
    /// The caller must hold [`MutationWriteScope`]. The returned token keeps
    /// readers behind the cut so a cooperating LPG/tier publisher can either
    /// finish its own infallible install or restore this exact prior state.
    fn publish_prepared_layer_generation<'cut, 'store, P>(
        ready: ReadyLayerPublication<'cut, 'store, P>,
    ) -> (PublishedLayerGeneration<'cut, 'store>, P) {
        let ReadyLayerPublication {
            revision,
            image: prepared,
            external,
        } = ready;
        let owner = revision.mutation.store;
        let publication = GenerationWriteScope::enter(owner);
        let PreparedLayerGeneration {
            base,
            overlay,
            dirty_node_ids,
            dirty_edge_ids,
            deleted_from_base_nodes,
            pending_base_node_deletes,
            deleted_from_base_edges,
            pending_base_edge_deletes,
            deletions_dirty,
            representation_transfer,
        } = prepared;

        // Move the exact runtime registries only after every Layered reader of
        // the old generation has drained. The named topology and source LPG
        // transition remain exclusive across this handoff and the following
        // pointer swaps.
        let published_representation =
            representation_transfer.map(PreparedRepresentationTransfer::publish);

        let previous_base = owner.base.swap(base);
        let previous_overlay = owner.overlay.swap(overlay);
        #[cfg(test)]
        if let Some(barrier) = owner.generation_publication_barrier.read().clone() {
            barrier.wait();
            barrier.wait();
        }
        let previous = PreparedLayerGeneration {
            base: previous_base,
            overlay: previous_overlay,
            dirty_node_ids: std::mem::replace(&mut owner.dirty_node_ids.write(), dirty_node_ids),
            dirty_edge_ids: std::mem::replace(&mut owner.dirty_edge_ids.write(), dirty_edge_ids),
            deleted_from_base_nodes: std::mem::replace(
                &mut owner.deleted_from_base_nodes.write(),
                deleted_from_base_nodes,
            ),
            pending_base_node_deletes: std::mem::replace(
                &mut owner.pending_base_node_deletes.write(),
                pending_base_node_deletes,
            ),
            deleted_from_base_edges: std::mem::replace(
                &mut owner.deleted_from_base_edges.write(),
                deleted_from_base_edges,
            ),
            pending_base_edge_deletes: std::mem::replace(
                &mut owner.pending_base_edge_deletes.write(),
                pending_base_edge_deletes,
            ),
            deletions_dirty: owner
                .deletions_dirty
                .swap(deletions_dirty, Ordering::AcqRel),
            representation_transfer: None,
        };
        owner
            .generation_revision
            .store(revision.next, Ordering::Release);
        (
            PublishedLayerGeneration {
                _publication: publication,
                previous,
                revision,
                representation_transfer: published_representation,
            },
            external,
        )
    }

    /// Commits a reversible publication by releasing its read cut and returns
    /// the displaced generation for retirement by the caller after every
    /// broader LPG/mutation guard has drained.
    fn commit_published_layer_generation(
        published: PublishedLayerGeneration<'_, '_>,
    ) -> PreparedLayerGeneration {
        let PublishedLayerGeneration {
            _publication,
            previous,
            revision: _revision,
            representation_transfer,
        } = published;
        drop(_publication);
        if let Some(representation_transfer) = representation_transfer {
            representation_transfer.commit();
        }
        previous
    }

    /// Restores a generation while its original publication barrier remains
    /// exclusive. No Layered reader can retain the transient candidate. The
    /// displaced failed candidate is returned for retirement after the LPG
    /// rollback stack has released its structural guards.
    fn restore_published_layer_generation(
        published: PublishedLayerGeneration<'_, '_>,
    ) -> PreparedLayerGeneration {
        let PublishedLayerGeneration {
            _publication,
            previous,
            revision,
            representation_transfer,
        } = published;
        let owner = revision.mutation.store;
        let PreparedLayerGeneration {
            base,
            overlay,
            dirty_node_ids,
            dirty_edge_ids,
            deleted_from_base_nodes,
            pending_base_node_deletes,
            deleted_from_base_edges,
            pending_base_edge_deletes,
            deletions_dirty,
            representation_transfer: previous_representation_transfer,
        } = previous;
        debug_assert!(previous_representation_transfer.is_none());

        if let Some(representation_transfer) = representation_transfer {
            representation_transfer.rollback();
        }

        let displaced = PreparedLayerGeneration {
            base: owner.base.swap(base),
            overlay: owner.overlay.swap(overlay),
            dirty_node_ids: std::mem::replace(&mut owner.dirty_node_ids.write(), dirty_node_ids),
            dirty_edge_ids: std::mem::replace(&mut owner.dirty_edge_ids.write(), dirty_edge_ids),
            deleted_from_base_nodes: std::mem::replace(
                &mut owner.deleted_from_base_nodes.write(),
                deleted_from_base_nodes,
            ),
            pending_base_node_deletes: std::mem::replace(
                &mut owner.pending_base_node_deletes.write(),
                pending_base_node_deletes,
            ),
            deleted_from_base_edges: std::mem::replace(
                &mut owner.deleted_from_base_edges.write(),
                deleted_from_base_edges,
            ),
            pending_base_edge_deletes: std::mem::replace(
                &mut owner.pending_base_edge_deletes.write(),
                pending_base_edge_deletes,
            ),
            deletions_dirty: owner
                .deletions_dirty
                .swap(deletions_dirty, Ordering::AcqRel),
            representation_transfer: None,
        };
        owner
            .generation_revision
            .store(revision.previous, Ordering::Release);
        drop(_publication);
        displaced
    }

    /// Atomically replaces the compact base store.
    ///
    /// The engine calls this after spilling the base to a mmap'd file: the
    /// old in-memory `Arc<CompactStore>` drops (freeing heap memory once the
    /// last external reference is released), and subsequent reads go through
    /// the new mmap-backed `CompactStore`. Overlay state (dirty sets,
    /// deleted sets, the overlay `LpgStore`) is untouched.
    ///
    /// Returns the previous base `Arc`, so callers can inspect refcounts or
    /// keep it alive while in-flight readers drain.
    ///
    /// # Panics
    ///
    /// Panics if the current overlay is sealed and the caller does not hold
    /// that exact store's [`crate::graph::write_permit::WriteAuthority`], or if
    /// the monotonic generation revision is exhausted. Authority is checked
    /// before the base or revision can change.
    pub fn swap_base(&self, new_base: Arc<CompactStore>) -> Arc<CompactStore> {
        self.with_pinned_base_transition(move || {
            let next_revision = self
                .generation_revision
                .load(Ordering::Acquire)
                .checked_add(1)
                .expect("LayeredStore generation revision exhausted");
            let _publication = GenerationWriteScope::enter(self);
            let previous = self.base.swap(new_base);
            self.generation_revision
                .store(next_revision, Ordering::Release);
            previous
        })
    }

    /// Transitions one exact compact-base generation under the same barrier
    /// used by merges and physical transport cleanup.
    ///
    /// `prepare` receives the currently published base and returns both the
    /// representation to publish and an opaque, fully prepared metadata plan.
    /// `prepare` runs under mutation/rebuild exclusion but outside the short
    /// read-publication barrier. Disk-tier callers perform all serialization,
    /// unique-file creation, mmap, and decode there. The barrier then closes;
    /// `finalize` performs only its prevalidated, infallible metadata pointer
    /// move before the matching base and revision are installed. This fixes the
    /// global lock order at Layered-publication then tier-metadata and prevents
    /// both metadata-new/base-old observations and reader/writer inversion.
    ///
    /// If `prepare` fails, neither the base nor caller metadata changes.
    /// If either callback unwinds, the generation barrier is released. An
    /// unwind from `finalize` propagates with the prior base still installed.
    /// Same-store reads are allowed from `prepare`. Same-thread nested reads in
    /// `finalize` are reentrant, but it must not spawn and join any same-store
    /// read or mutation because other threads correctly wait behind the closed
    /// publication cut. Re-entering a generation write API from either callback
    /// panics immediately rather than waiting on the non-reentrant mutation
    /// barrier.
    ///
    /// This is an engine integration seam rather than a general transaction
    /// over arbitrary callback side effects. `finalize` must publish its opaque
    /// metadata with one non-panicking assignment after all fallible work has
    /// completed. It must not publish a second graph authority; Layered remains
    /// the sole reader-visible generation source.
    ///
    /// # Errors
    ///
    /// Returns the callback's error without changing the base.
    ///
    /// # Panics
    ///
    /// Panics before either callback runs if the current overlay is sealed and
    /// the caller does not hold that exact store's
    /// [`crate::graph::write_permit::WriteAuthority`]. Also panics if the
    /// monotonic generation revision is exhausted or either callback unwinds.
    #[doc(hidden)]
    pub fn transition_base_generation<E, P>(
        &self,
        prepare: impl FnOnce(Arc<CompactStore>) -> Result<(Arc<CompactStore>, P), E>,
        finalize: impl FnOnce(P),
    ) -> Result<Arc<CompactStore>, E> {
        self.transition_base_generation_with_retirement(prepare, |prepared| {
            finalize(prepared);
        })
        .map(|(previous, ())| previous)
    }

    /// Reversible-token form of [`Self::transition_base_generation`].
    ///
    /// `publish` returns ownership of the displaced external metadata. The
    /// token is carried out of both Layered guards rather than dropped on the
    /// publication cut, allowing mmap/file/log retirement only after every
    /// coherent reader has drained.
    ///
    /// # Panics
    ///
    /// Panics before either callback runs if the current overlay is sealed and
    /// the caller does not hold that exact store's
    /// [`crate::graph::write_permit::WriteAuthority`]. Also panics if the
    /// monotonic generation revision is exhausted or either callback unwinds.
    #[doc(hidden)]
    pub fn transition_base_generation_with_retirement<E, P, R>(
        &self,
        prepare: impl FnOnce(Arc<CompactStore>) -> Result<(Arc<CompactStore>, P), E>,
        publish: impl FnOnce(P) -> R,
    ) -> Result<(Arc<CompactStore>, R), E> {
        self.with_pinned_base_transition(|| {
            let (next, prepared) = prepare(self.base.load_full())?;
            let previous_revision = self.generation_revision.load(Ordering::Acquire);
            let next_revision = previous_revision
                .checked_add(1)
                .expect("LayeredStore generation revision exhausted");
            let publication = GenerationWriteScope::enter(self);
            let retirement = publish(prepared);
            let previous = self.base.swap(next);
            self.generation_revision
                .store(next_revision, Ordering::Release);
            drop(publication);
            Ok((previous, retirement))
        })
    }

    /// Returns the current overlay LPG store as an owned `Arc`.
    ///
    /// Phase 5c: the overlay is now wrapped in an `ArcSwap` so it can be
    /// atomically replaced after a merge. Callers receive a snapshot `Arc`
    /// that remains fully readable, including through its physical index
    /// registries, even if the overlay is later swapped. A same-incarnation
    /// compact/recompact handoff retires the displaced representation from
    /// mutation because its logical index handles move to the successor. A
    /// destructive [`Self::reset_overlay`] instead leaves the detached snapshot
    /// independently mutable and gives the live Layered store a genuinely fresh
    /// representation with none of the discarded rows or registries.
    #[must_use]
    pub fn overlay_store(&self) -> Arc<LpgStore> {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load_full()
    }

    /// Number of dirty (modified/created) entities in the overlay.
    #[must_use]
    pub fn overlay_mutation_count(&self) -> usize {
        let _generation = GenerationReadScope::enter(self);
        self.dirty_node_ids.read().len()
            + self.dirty_edge_ids.read().len()
            + self.deleted_from_base_nodes.read().len()
            + self.deleted_from_base_edges.read().len()
    }

    /// Overlay has no live data or base tombstones: current 1-hop is the cold CSR.
    fn overlay_is_cold(&self) -> bool {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        let overlay = self.overlay.load();
        // Own the Layered routing cut before consulting LPG. A promotion may
        // make its private physical row visible while this call waits on the
        // rollback gate, but the pre-publication dirty snapshot continues to
        // route that identity to the base. Never retain the guards themselves
        // across LPG reads: the promoter needs their write side to publish.
        let dirty_nodes = self.dirty_node_ids.read().clone();
        let dirty_edges = self.dirty_edge_ids.read().clone();
        let has_node_deletions = !self.deleted_from_base_nodes.read().is_empty();
        let has_edge_deletions = !self.deleted_from_base_edges.read().is_empty();
        let overlay_nodes = overlay.node_ids();
        let overlay_edges: Vec<EdgeId> = overlay.all_edges().map(|edge| edge.id).collect();
        let has_published_node = overlay_nodes
            .into_iter()
            .any(|id| Self::overlay_node_is_published(&base, &dirty_nodes, id));
        let has_published_edge = overlay_edges
            .into_iter()
            .any(|id| Self::overlay_edge_is_published(&base, &dirty_edges, id));
        !has_published_node && !has_published_edge && !has_node_deletions && !has_edge_deletions
    }

    /// Approximate heap bytes of the overlay only (excluding base).
    ///
    /// Used by `OverlayConsumer` (Phase 5c) to drive merge-on-pressure
    /// without conflating overlay growth with base size.
    #[must_use]
    pub fn overlay_memory_bytes(&self) -> usize {
        let _generation = GenerationReadScope::enter(self);
        let (store_mem, index_mem, mvcc_mem, pool_mem) = self.overlay.load().memory_breakdown();
        store_mem.total_bytes + index_mem.total_bytes + mvcc_mem.total_bytes + pool_mem.total_bytes
    }

    /// Approximate heap memory of both layers.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let _generation = GenerationReadScope::enter(self);
        self.base.load().memory_bytes() + self.overlay_memory_bytes()
    }

    /// Creates and publishes a same-incarnation empty overlay together with a
    /// complete routing generation.
    ///
    /// LPG retains its exact mutation-scope proof through preparation,
    /// Layered publication, hostile-boundary rollback, and completion. The
    /// caller holds [`MutationWriteScope`], so Layered mutations cannot overtake
    /// the sampled epoch or allocator high-water marks.
    fn install_with_empty_successor(
        &self,
        current_overlay: &Arc<LpgStore>,
        mutation: &mut MutationWriteScope<'_>,
        retirements: &GenerationRetirements<(), ()>,
        prepare: impl FnOnce(
            &PinnedLpgTransition<'_>,
            Arc<LpgStore>,
        ) -> Result<PreparedLayerGeneration, String>,
    ) -> Result<PreparedLayerGeneration, String> {
        // Reset is intentionally a fresh empty representation, not an exact
        // registry handoff. Its old overlay rows are discarded, so moving
        // their vector/text/property/named-graph registries to the row-empty
        // successor would create stale identities. The retained old Arc keeps
        // those registries and remains an independent, mutable snapshot.
        let _named_topology = current_overlay.pin_named_graph_topology();
        let Some(transition) = current_overlay.pin_exclusive_unframed_transition() else {
            return Err(
                "current overlay write scope is not held; refusing generation replacement"
                    .to_owned(),
            );
        };
        let (retired, ()) = self.install_with_pinned_empty_successor_and_publication(
            PinnedEmptySuccessorPublication {
                current_overlay,
                representation_topology: None,
                transition: &transition,
                mutation,
                retirements,
            },
            |scope, successor| prepare(scope, successor).map(|generation| (generation, ())),
            |()| (),
            |()| (),
        )?;
        Ok(retired)
    }

    /// Already-pinned form used when the temporal graph snapshot itself must
    /// be covered by the same named-topology and source LPG transition as its
    /// exact runtime-registry publication.
    fn install_with_pinned_empty_successor_and_publication<P, R, Q>(
        &self,
        context: PinnedEmptySuccessorPublication<'_, '_, '_, P, Q>,
        prepare: impl FnOnce(
            &PinnedLpgTransition<'_>,
            Arc<LpgStore>,
        ) -> Result<(PreparedLayerGeneration, P), String>,
        publish: impl FnOnce(P) -> R,
        rollback: impl FnOnce(R) -> Q,
    ) -> Result<(PreparedLayerGeneration, R), String> {
        let PinnedEmptySuccessorPublication {
            current_overlay,
            representation_topology,
            transition,
            mutation,
            retirements,
        } = context;
        let publication = transition
            .publish_empty_generation_after_prepare_and_publish_with_rollback(
                |scope| {
                    let revision = mutation.checked_revision()?;
                    let successor = Arc::new(
                        if representation_topology.is_some() {
                            scope.prepare_same_incarnation_empty_successor()
                        } else {
                            scope.prepare_reset_empty_successor()
                        }
                        .map_err(|error| error.to_string())?,
                    );
                    // Merge/recompact (`Some`) prepare empty equality indexes,
                    // frozen source views, and all registry capacity before
                    // retained-hot replay; original vector/text objects and
                    // named topology stay on the source until the coherent
                    // cut. Reset (`None`) deliberately leaves every discarded
                    // row-bearing registry on its detached source snapshot.
                    *retirements.anchors.borrow_mut() = Some(GenerationRetirementAnchors::capture(
                        current_overlay,
                        &successor,
                    ));
                    let representation_transfer = representation_topology
                        .map(|named_topology| {
                            scope.prepare_same_incarnation_representation_transfer(
                                named_topology,
                                Arc::clone(current_overlay),
                                Arc::clone(&successor),
                            )
                        })
                        .transpose()?;
                    // The completed transfer owns index fences independently of
                    // the external preparer. Never let Err/unwind retire it
                    // beneath this LPG/Layered cut.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        prepare(scope, Arc::clone(&successor))
                    }));
                    let (mut generation, external) = match result {
                        Ok(Ok(pair)) => pair,
                        Ok(Err(error)) => {
                            *retirements.transfer.borrow_mut() = representation_transfer;
                            return Err(error);
                        }
                        Err(payload) => {
                            *retirements.transfer.borrow_mut() = representation_transfer;
                            std::panic::resume_unwind(payload);
                        }
                    };
                    let validation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        #[cfg(test)]
                        if self
                            .generation_image_validation_panic
                            .swap(false, Ordering::SeqCst)
                        {
                            panic!("generation final image validation unwind");
                        }
                        if !Arc::ptr_eq(&generation.overlay, &successor) {
                            return Err(
                                "empty-successor preparation replaced the pinned LPG successor"
                                    .to_owned(),
                            );
                        }
                        if generation.representation_transfer.is_some() {
                            return Err(
                                "empty-successor image already owns a representation transfer"
                                    .to_owned(),
                            );
                        }
                        if let Some(transfer) = &representation_transfer {
                            transfer.validate_unpublished_target()?;
                        }
                        Ok(())
                    }));
                    match validation {
                        Ok(Ok(())) => {
                            generation.representation_transfer = representation_transfer;
                            Ok(ReadyLayerPublication {
                                revision,
                                image: generation,
                                external,
                            })
                        }
                        Ok(Err(error)) => {
                            *retirements.transfer.borrow_mut() = representation_transfer;
                            *retirements.unpublished.borrow_mut() =
                                Some(UnpublishedLayerRetirement::new(generation, external));
                            Err(error)
                        }
                        Err(payload) => {
                            *retirements.transfer.borrow_mut() = representation_transfer;
                            *retirements.unpublished.borrow_mut() =
                                Some(UnpublishedLayerRetirement::new(generation, external));
                            std::panic::resume_unwind(payload);
                        }
                    }
                },
                |ready| {
                    let (layered, external) = Self::publish_prepared_layer_generation(ready);
                    let external =
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            publish(external)
                        })) {
                            Ok(external) => external,
                            Err(payload) => {
                                *retirements.rollback_layer.borrow_mut() =
                                    Some(Self::restore_published_layer_generation(layered));
                                std::panic::resume_unwind(payload);
                            }
                        };
                    ReversibleLayerPublication { external, layered }
                },
                |publication| {
                    let ReversibleLayerPublication { external, layered } = publication;
                    let external_rollback =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            rollback(external)
                        }));
                    *retirements.rollback_layer.borrow_mut() =
                        Some(Self::restore_published_layer_generation(layered));
                    match external_rollback {
                        Ok(retirement) => {
                            *retirements.rollback_external.borrow_mut() = Some(retirement);
                        }
                        Err(payload) => std::panic::resume_unwind(payload),
                    }
                },
            )?;
        let ReversibleLayerPublication { external, layered } = publication;
        let retired = Self::commit_published_layer_generation(layered);
        Ok((retired, external))
    }

    /// Replaces the overlay with a fresh empty `LpgStore` and clears
    /// dirty/deleted bookkeeping (Phase 5c).
    ///
    /// Atomic: in-flight readers holding an `Arc<LpgStore>` snapshot
    /// continue against the old overlay; subsequent reads pick up the
    /// fresh empty one.
    ///
    /// The new overlay's id allocators are seeded from the *current*
    /// base so freshly created nodes/edges don't collide with base ids.
    ///
    /// # Panics
    ///
    /// Panics if the system allocator cannot build the successor, or if the
    /// current overlay is sealed and the caller does not hold that store's
    /// exact [`crate::graph::write_permit::WriteAuthority`]. The latter is a
    /// deliberate fail-closed boundary: replacing the whole mutable
    /// generation is itself a write and cannot bypass the store seal.
    pub fn reset_overlay(&self) {
        let retirements = GenerationRetirements::new();
        let mut mutations = MutationWriteScope::enter(self);
        let retired = self
            .reset_overlay_under_mutation_scope(&mut mutations, &retirements)
            .expect("resetting the Layered overlay requires its exact write scope");
        drop(mutations);
        drop(retired);
    }

    /// Replaces the overlay while the caller holds `merge_guard` exclusively.
    /// Merge paths already own that barrier and cannot call the public wrapper
    /// without recursively acquiring the non-reentrant lock.
    fn reset_overlay_under_mutation_scope(
        &self,
        mutation: &mut MutationWriteScope<'_>,
        retirements: &GenerationRetirements<(), ()>,
    ) -> Result<PreparedLayerGeneration, String> {
        let base = self.base.load_full();
        let current_overlay = self.overlay.load_full();
        self.install_with_empty_successor(
            &current_overlay,
            mutation,
            retirements,
            move |_scope, successor| {
                successor
                    .install_compact_base(Arc::clone(&base))
                    .map_err(str::to_owned)?;
                Self::raise_overlay_allocator_floors(&base, &successor);
                Ok(PreparedLayerGeneration::empty(base, successor))
            },
        )
    }

    /// Replays one complete label image onto an existing live node.
    ///
    /// The recovery schedule must place images before the node's deletion.
    /// Cold hydration retains its complete history before the exact append;
    /// this does not reopen closed cold identities. Call only on an unpublished
    /// or otherwise quiescent recovery store with its exact write authority.
    ///
    /// # Errors
    ///
    /// Rejects invalid images/frontiers, missing or tombstoned nodes, denied
    /// authority, failed hydration, and native exact-image qualification errors.
    #[doc(hidden)]
    pub fn replay_node_labels_at_epoch(
        &self,
        id: NodeId,
        epoch: EpochId,
        labels: &[ArcStr],
    ) -> Result<(), grafeo_common::Error> {
        LpgStore::validate_node_label_replay_image(id, epoch, labels)?;
        let _generation = (!self.recorded_index_recovery_active()).then(|| self.merge_guard.read());
        let overlay = self.overlay.load_full();
        {
            let _mutation = overlay.pin_mutation().ok_or_else(|| {
                grafeo_common::Error::Storage(StorageError::Corruption(
                    "compact label-image replay requires current overlay write authority"
                        .to_owned(),
                ))
            })?;
            if epoch != overlay.current_epoch() {
                return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                    "compact label-image epoch differs from recovery frontier".to_owned(),
                )));
            }
            if self.deleted_from_base_nodes.read().contains_key(&id)
                || self
                    .pending_base_node_deletes
                    .read()
                    .values()
                    .any(|nodes| nodes.contains(&id))
                || self.get_node(id).is_none()
            {
                return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                    "compact label-image replay requires a live, non-tombstoned node".to_owned(),
                )));
            }
            if !self.try_ensure_in_overlay(id)? {
                return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                    "compact label-image replay could not hydrate its node".to_owned(),
                )));
            }
        }
        // The shared LPG pin must drain before native exclusive admission.
        // Retaining only the merge pin anchors the same overlay throughout.
        overlay.replay_node_labels_at_epoch(id, epoch, labels)
    }

    /// Replays one exact node create into the current compact overlay.
    ///
    /// This is a recovery-only seam. It accepts an already-present identity
    /// only when the original structural create carries the same logical label
    /// set, making repeated replay idempotent without treating a later label or
    /// property mutation as a different create. Closed/deleted identities and
    /// occupied IDs with a different create image are corruption, not reusable
    /// allocator slots.
    ///
    /// The current overlay's sealed write scope is checked before any history,
    /// allocator, or routing state changes. A fresh row becomes authoritative
    /// in `dirty_node_ids` only after exact LPG insertion has succeeded and its
    /// full postcondition has been verified.
    ///
    /// # Errors
    ///
    /// Returns a structured storage-corruption error for invalid, conflicting,
    /// or closed identities, missing write authority, or a failed structural
    /// postcondition. Allocation failures are preserved from the LPG store.
    #[doc(hidden)]
    pub fn recover_create_node_with_id(
        &self,
        id: NodeId,
        labels: &[&str],
    ) -> Result<(), grafeo_common::Error> {
        self.with_pinned_overlay_mutation(
            || {
                Err(grafeo_common::Error::Storage(StorageError::Corruption(
                    "compact WAL node replay requires the current overlay write authority"
                        .to_owned(),
                )))
            },
            |overlay| {
                if !id.is_valid() {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        "compact WAL node replay carries an invalid node identity".to_owned(),
                    )));
                }
                if self.deleted_from_base_nodes.read().contains_key(&id) {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL node replay attempts to reuse tombstoned node {id}"
                        ),
                    )));
                }

                let existing = self.node_structural_history(id);
                if !existing.is_empty() {
                    if self.get_node(id).is_none() {
                        return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                            format!(
                                "compact WAL node replay attempts to reuse closed or deleted node {id}"
                            ),
                        )));
                    }
                    let creation_labels = existing
                        .label_versions
                        .first()
                        .map_or(existing.labels.as_slice(), |(_, labels)| labels.as_slice());
                    if recovery_labels_match(creation_labels, labels) {
                        return Ok(());
                    }
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL node replay conflicts with the structural identity of node {id}"
                        ),
                    )));
                }

                overlay.create_node_with_id(id, labels)?;
                let Some(created) = overlay.get_node(id) else {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL node replay did not publish requested node {id}"
                        ),
                    )));
                };
                if !recovery_labels_match(&created.labels, labels) {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL node replay published a non-identical node at {id}"
                        ),
                    )));
                }
                self.dirty_node_ids.write().insert(id);
                Ok(())
            },
        )
    }

    /// Replays one exact edge create into the current compact overlay.
    ///
    /// Both endpoints must already be logically live in the layered graph. A
    /// base-only endpoint is hydrated into the overlay before insertion; a
    /// tombstoned or missing endpoint is rejected before either endpoint is
    /// promoted. Replaying an existing edge is idempotent only for the exact
    /// `(source, destination, type)` structural identity, while any closed or
    /// conflicting identity fails closed.
    ///
    /// The current overlay's sealed write scope is checked before mutation.
    /// Dirty-edge routing is published only after the exact LPG row exists and
    /// its postcondition has been verified; the LPG exact-ID allocator update
    /// keeps subsequent generated identities above the recovered ID.
    ///
    /// # Errors
    ///
    /// Returns a structured storage-corruption error for invalid, conflicting,
    /// or closed identities, missing/deleted endpoints, missing write
    /// authority, or a failed structural postcondition. Allocation failures are
    /// preserved from the LPG store.
    #[doc(hidden)]
    pub fn recover_create_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), grafeo_common::Error> {
        self.with_pinned_overlay_mutation(
            || {
                Err(grafeo_common::Error::Storage(StorageError::Corruption(
                    "compact WAL edge replay requires the current overlay write authority"
                        .to_owned(),
                )))
            },
            |overlay| {
                if !id.is_valid() || !src.is_valid() || !dst.is_valid() {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        "compact WAL edge replay carries an invalid structural identity"
                            .to_owned(),
                    )));
                }
                if self.deleted_from_base_edges.read().contains_key(&id) {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL edge replay attempts to reuse tombstoned edge {id}"
                        ),
                    )));
                }

                let existing = self.edge_full_history(id);
                if !existing.is_empty() {
                    if self.get_edge(id).is_none() {
                        return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                            format!(
                                "compact WAL edge replay attempts to reuse closed or deleted edge {id}"
                            ),
                        )));
                    }
                    if existing.src != src
                        || existing.dst != dst
                        || existing.edge_type.as_str() != edge_type
                    {
                        return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                            format!(
                                "compact WAL edge replay conflicts with the structural identity of edge {id}"
                            ),
                        )));
                    }
                    if self.get_node(src).is_none() || self.get_node(dst).is_none() {
                        return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                            format!(
                                "compact WAL edge replay found live edge {id} with a missing or deleted endpoint"
                            ),
                        )));
                    }
                    return Ok(());
                }

                // Validate both logical endpoints before promotion. In
                // particular, a current compact row hidden by a Layered
                // tombstone is not a live endpoint merely because the cold
                // representation can still decode it.
                if self.get_node(src).is_none() {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL edge replay references missing or deleted source node {src}"
                        ),
                    )));
                }
                if self.get_node(dst).is_none() {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL edge replay references missing or deleted destination node {dst}"
                        ),
                    )));
                }
                if !self.try_ensure_in_overlay(src)? || !self.try_ensure_in_overlay(dst)? {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL edge replay could not hydrate endpoints for edge {id}"
                        ),
                    )));
                }

                overlay.create_edge_with_id(id, src, dst, edge_type)?;
                let Some(created) = overlay.get_edge(id) else {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL edge replay did not publish requested edge {id}"
                        ),
                    )));
                };
                if created.src != src
                    || created.dst != dst
                    || created.edge_type.as_str() != edge_type
                {
                    return Err(grafeo_common::Error::Storage(StorageError::Corruption(
                        format!(
                            "compact WAL edge replay published a non-identical edge at {id}"
                        ),
                    )));
                }
                self.dirty_edge_ids.write().insert(id);
                Ok(())
            },
        )
    }

    /// Returns a snapshot of the base node ids the overlay has marked as
    /// deleted but not yet merged. Used by the persistence layer to write
    /// the [`OverlayDeletions`](grafeo_common::storage::section::SectionType::OverlayDeletions)
    /// section so the deletions survive close/reopen cycles.
    #[must_use]
    pub fn snapshot_deleted_node_ids(&self) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        // Keys only: same keys-only on-disk format and re-seed-as-committed
        // semantics as `snapshot_deleted_edge_ids` (the (epoch, deleter) stamp
        // is in-memory MVCC bookkeeping, not persisted).
        self.deleted_from_base_nodes
            .read()
            .keys()
            .copied()
            .collect()
    }

    /// Snapshot of committed base-node tombstones with their exact delete
    /// epochs. `PENDING` entries belong to in-flight transactions and are not
    /// durable state, so they are deliberately excluded.
    #[must_use]
    pub fn snapshot_deleted_nodes(&self) -> Vec<(NodeId, EpochId)> {
        let _generation = GenerationReadScope::enter(self);
        self.deleted_from_base_nodes
            .read()
            .iter()
            .filter_map(|(&id, delete)| {
                (delete.epoch != EpochId::PENDING).then_some((id, delete.epoch))
            })
            .collect()
    }

    /// Snapshot of base edge ids deleted-but-not-merged. See
    /// [`Self::snapshot_deleted_node_ids`].
    #[must_use]
    pub fn snapshot_deleted_edge_ids(&self) -> Vec<EdgeId> {
        let _generation = GenerationReadScope::enter(self);
        // Keys only: the on-disk `OverlayDeletions` format is unchanged (just
        // edge ids). The per-tombstone (epoch, deleter) stamp is in-memory MVCC
        // bookkeeping and is not persisted — a reopened delete is re-seeded as
        // a committed delete (epoch 0, deleter None) by `seed_deleted_from_base`.
        self.deleted_from_base_edges
            .read()
            .keys()
            .copied()
            .collect()
    }

    /// Snapshot of committed base-edge tombstones with their exact delete
    /// epochs. In-flight `PENDING` tombstones are never checkpointed.
    #[must_use]
    pub fn snapshot_deleted_edges(&self) -> Vec<(EdgeId, EpochId)> {
        let _generation = GenerationReadScope::enter(self);
        self.deleted_from_base_edges
            .read()
            .iter()
            .filter_map(|(&id, delete)| {
                (delete.epoch != EpochId::PENDING).then_some((id, delete.epoch))
            })
            .collect()
    }

    /// Seeds the deleted-from-base sets from a previously-persisted
    /// snapshot (typically the `OverlayDeletions` section). The current
    /// sets are replaced atomically; any in-memory deletions accumulated
    /// before the seed are dropped (callers should only seed during
    /// open, before any new mutations are accepted).
    ///
    /// Clears the deletions-dirty flag so the next checkpoint does not
    /// re-write the section just because the seed populated it.
    ///
    /// # Panics
    ///
    /// Panics if the current overlay is sealed and the caller does not hold
    /// that store's exact write authority.
    pub fn seed_deleted_from_base(
        &self,
        nodes: impl IntoIterator<Item = NodeId>,
        edges: impl IntoIterator<Item = EdgeId>,
    ) {
        self.seed_deleted_from_base_at_epochs(
            nodes.into_iter().map(|id| (id, EpochId::INITIAL)),
            edges.into_iter().map(|id| (id, EpochId::INITIAL)),
        );
    }

    /// Seeds persisted base tombstones with their original commit epochs.
    ///
    /// The exact epoch is required for historical reads: an entity deleted at
    /// `D` remains visible before `D` and is absent at and after `D`. The legacy
    /// id-only loader delegates here with `INITIAL`, preserving v1 decoding.
    ///
    /// # Panics
    ///
    /// Panics if the current overlay is sealed and the caller does not hold
    /// that store's exact write authority. Recovery normally invokes this on an
    /// unpublished, unsealed store.
    pub fn seed_deleted_from_base_at_epochs(
        &self,
        nodes: impl IntoIterator<Item = (NodeId, EpochId)>,
        edges: impl IntoIterator<Item = (EdgeId, EpochId)>,
    ) {
        self.with_pinned_overlay_mutation(
            || panic!("seeding Layered tombstones requires the current overlay write scope"),
            |_| {
                let mut node_set = self.deleted_from_base_nodes.write();
                node_set.clear();
                node_set.extend(nodes.into_iter().map(|(id, epoch)| {
                    (
                        id,
                        BaseNodeDelete {
                            epoch,
                            deleter: None,
                        },
                    )
                }));
                let mut edge_set = self.deleted_from_base_edges.write();
                edge_set.clear();
                edge_set.extend(edges.into_iter().map(|(id, epoch)| {
                    (
                        id,
                        BaseEdgeDelete {
                            epoch,
                            deleter: None,
                        },
                    )
                }));
                drop(node_set);
                drop(edge_set);
                // A reseed replaces any in-flight transactional base
                // tombstones; their pending bookkeeping is now stale (open
                // happens before new mutations).
                self.pending_base_node_deletes.write().clear();
                self.pending_base_edge_deletes.write().clear();
                self.deletions_dirty.store(false, Ordering::Release);
            },
        );
    }

    /// Whether the deletion log has changed since the last
    /// [`mark_deletions_clean`](Self::mark_deletions_clean) call. Used by
    /// the `OverlayDeletionsSection` to decide whether a periodic
    /// checkpoint should re-emit the section.
    #[must_use]
    pub fn deletions_dirty(&self) -> bool {
        self.deletions_dirty.load(Ordering::Acquire)
    }

    /// Marks the deletion log as clean. Called by the flush path after a
    /// successful write of the `OverlayDeletions` section.
    pub fn mark_deletions_clean(&self) {
        self.deletions_dirty.store(false, Ordering::Release);
    }

    /// Merges the overlay into a fresh `CompactStore`, swaps it in as
    /// the base, and clears the overlay (Phase 5c).
    ///
    /// After this call: all previously-visible data is in the base; the
    /// overlay is empty. Used by `OverlayConsumer` to release overlay
    /// memory under pressure.
    ///
    /// # Errors
    ///
    /// Returns an error if rebuilding/preparing the successor fails or the
    /// caller does not hold the current overlay's sealed write authority.
    pub fn merge_overlay_in_place(&self) -> Result<(), String> {
        if self.overlay.load().may_have_unresolved_transport_edges() {
            return self.merge_overlay_temporal();
        }
        // Stop writers while the successor is built. Readers continue on the
        // old generation during that build and pause only for the short,
        // allocation-free coherent publication cut.
        let retirements = GenerationRetirements::new();
        let mut mutations = MutationWriteScope::enter(self);
        let current_overlay = self.overlay.load_full();
        let named_topology: PinnedNamedGraphTopology = current_overlay.pin_named_graph_topology();
        let Some(transition) = current_overlay.pin_exclusive_unframed_transition() else {
            return Err(
                "current overlay write scope is not held; refusing generation replacement"
                    .to_owned(),
            );
        };

        // A receipt may have been created after the optimistic check. The
        // exclusive source transition now excludes that publication race.
        // Release all source gates before entering the existing temporal path;
        // the sticky discriminator is transferred to successor incarnations.
        if current_overlay.may_have_unresolved_transport_edges() {
            drop(transition);
            drop(named_topology);
            drop(mutations);
            return self.merge_overlay_temporal();
        }

        let (retired, ()) = self.install_with_pinned_empty_successor_and_publication(
            PinnedEmptySuccessorPublication {
                current_overlay: &current_overlay,
                representation_topology: Some(&named_topology),
                transition: &transition,
                mutation: &mut mutations,
                retirements: &retirements,
            },
            move |_scope, successor| {
                // Exact vector/text mutation fences are already held by the
                // helper before the first graph row is sampled, so the compact
                // graph and its transferred physical indexes share one cut.
                let next_base = Arc::new(
                    super::from_graph_store_preserving_ids(self)
                        .map_err(|error| error.to_string())?,
                );
                successor
                    .install_compact_base(Arc::clone(&next_base))
                    .map_err(str::to_owned)?;
                Self::raise_overlay_allocator_floors(&next_base, &successor);
                Ok((PreparedLayerGeneration::empty(next_base, successor), ()))
            },
            |()| (),
            |()| (),
        )?;
        drop(transition);
        drop(named_topology);
        drop(mutations);
        drop(retired);
        Ok(())
    }

    /// Collects retired vector membership using this logical generation's
    /// complete lifetimes, including cold-base deletion tombstones.
    ///
    /// The overlay retains sole ownership of the physical registry and its
    /// mutation admission. Only per-candidate histories cross this boundary;
    /// no alternate index or graph-wide history snapshot is constructed.
    ///
    /// # Errors
    /// Rejects PENDING horizons, denied authority and invalid retained backing.
    /// Collection is atomic per physical index, not across the whole database.
    #[cfg(feature = "vector-index")]
    pub fn gc_vector_indexes(&self, horizon: EpochId) -> Result<(), Error> {
        if horizon == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "Vector GC horizon cannot be PENDING".into(),
            ));
        }
        self.with_pinned_overlay_mutation(
            || Err(Error::Transaction(TransactionError::ReadOnly)),
            |overlay| {
                overlay.gc_vector_indexes_with_context(
                    horizon,
                    |id, label, property, physically_live| {
                        let history = self.node_structural_history(id);
                        if LpgStore::vector_gc_lifetimes_deleted(
                            history
                                .lifetimes
                                .iter()
                                .map(|life| (life.created, life.deleted)),
                            horizon,
                        ) {
                            return false;
                        }
                        if physically_live {
                            return true;
                        }
                        let Some(lifetime) = history
                            .lifetimes
                            .iter()
                            .filter(|life| life.created != EpochId::PENDING)
                            .max_by_key(|life| life.created)
                        else {
                            return true;
                        };
                        let property_retired =
                            history.properties.get(property).is_some_and(|versions| {
                                LpgStore::vector_gc_property_retired(
                                    lifetime.created,
                                    horizon,
                                    versions,
                                )
                            });
                        !property_retired
                            && !LpgStore::vector_gc_label_retired(
                                lifetime.created,
                                horizon,
                                label,
                                &history.label_versions,
                            )
                    },
                    |id, property| {
                        let history = self.node_structural_history(id);
                        let lifetime = history
                            .lifetimes
                            .iter()
                            .filter(|life| life.created != EpochId::PENDING)
                            .max_by_key(|life| life.created)?;
                        LpgStore::vector_gc_backing_from_history(
                            lifetime.created,
                            lifetime.deleted,
                            history.properties.get(property)?,
                        )
                    },
                )
            },
        )
    }

    /// Property-only view used by the existing node-table temporal upgrader.
    fn node_property_full_history(&self, id: NodeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        self.node_structural_history(id)
            .properties
            .into_iter()
            .collect()
    }

    /// Complete node history across cold base and overlay, including structural
    /// create/delete epochs. A promoted base node has a restamped overlay create;
    /// `merge_edge_lifetimes` joins that continuation back to the original cold
    /// lifetime, while property histories are merged epoch-wise.
    fn node_structural_history(&self, id: NodeId) -> super::compaction::NodeFullHistory {
        use super::compaction::{EdgeLifetime, NodeFullHistory};

        let dirty = self.dirty_node_ids.read();
        let base = self.base.load();
        let overlay_is_published = Self::overlay_node_is_published(&base, &dirty, id);
        let mut base_history = base.temporal_node_history(id).unwrap_or_else(|| {
            let Some(node) = base.get_node(id) else {
                return NodeFullHistory::default();
            };
            let mut labels: Vec<ArcStr> = node.labels.iter().cloned().collect();
            labels.sort_unstable();
            labels.dedup();
            NodeFullHistory {
                label_versions: vec![(EpochId::INITIAL, labels.clone())],
                labels,
                lifetimes: vec![EdgeLifetime::new(EpochId::INITIAL, None)],
                properties: base.node_property_history(id).into_iter().collect(),
            }
        });
        if let Some(tomb) = self.deleted_from_base_nodes.read().get(&id)
            && tomb.epoch != EpochId::PENDING
            && let Some(open) = base_history
                .lifetimes
                .iter_mut()
                .rev()
                .find(|life| life.deleted.is_none())
        {
            open.deleted = Some(tomb.epoch);
        }
        drop(base);

        if !overlay_is_published {
            return base_history;
        }
        drop(dirty);

        let overlay = self.overlay.load();
        let mut versions = overlay.get_node_history(id);
        if versions.is_empty() {
            return base_history;
        }
        let overlay_label_versions = overlay.node_label_history(id);
        versions.reverse();
        let overlay_lifetimes: Vec<EdgeLifetime> = versions
            .iter()
            .map(|(created, deleted, _)| EdgeLifetime::new(*created, *deleted))
            .collect();
        let mut properties = base_history.properties;
        for (key, overlay_versions) in overlay.node_property_history(id) {
            let merged = properties.entry(key).or_default();
            merge_property_versions(merged, overlay_versions);
        }
        let mut label_versions = base_history.label_versions;
        if label_versions.is_empty() && !base_history.labels.is_empty() {
            let epoch = base_history
                .lifetimes
                .first()
                .map_or(EpochId::INITIAL, |life| life.created);
            label_versions.push((epoch, base_history.labels.clone()));
        }
        // Promotion hydrates the exact cold label log before appending hot
        // mutations. As with property history, the overlay is authoritative
        // for every epoch it carries: retaining the cold copy as well would
        // duplicate same-epoch user events on each promote/recompact cycle.
        let overlay_label_epochs: FxHashSet<EpochId> = overlay_label_versions
            .iter()
            .map(|(epoch, _)| *epoch)
            .collect();
        label_versions.retain(|(epoch, _)| !overlay_label_epochs.contains(epoch));
        label_versions.extend(overlay_label_versions);
        // Stable epoch sort retains every explicit image, including equal
        // same-epoch entries. Overlapping base epochs were removed above.
        label_versions.sort_by_key(|(epoch, _)| *epoch);
        let labels = label_versions
            .last()
            .map(|(_, labels)| labels.clone())
            .or_else(|| {
                versions
                    .last()
                    .map(|(_, _, node)| node.labels.iter().cloned().collect())
            })
            .unwrap_or(base_history.labels);
        NodeFullHistory {
            labels,
            label_versions,
            lifetimes: super::compaction::merge_edge_lifetimes(
                &base_history.lifetimes,
                &overlay_lifetimes,
            ),
            properties,
        }
    }

    /// Overlay chains + current and temporal-base node identities + base
    /// tombstones. Closed nodes are absent from current scans but must remain in
    /// the compaction input so their history survives re-compaction.
    fn known_node_ids(&self) -> Vec<NodeId> {
        let mut ids = FxHashSet::default();
        ids.extend(self.overlay.load().all_node_ids());
        ids.extend(self.deleted_from_base_nodes.read().keys().copied());
        let base = self.base.load();
        ids.extend(base.node_ids());
        ids.extend(base.temporal_node_ids());
        let mut ids: Vec<NodeId> = ids.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns complete histories for every known node in stable ID order.
    ///
    /// This includes current overlay/base nodes and closed or deleted identities
    /// retained only by temporal sidecars or base tombstones. Each returned
    /// [`NodeFullHistory`](super::compaction::NodeFullHistory) has structural
    /// lifetimes ordered oldest first, complete label-set versions ordered by
    /// ascending epoch, and full per-property logs ordered by ascending epoch,
    /// including `Null` tombstones.
    ///
    /// The merge guard pins one base/overlay generation for the traversal.
    /// Callers that require a transaction-publication-consistent graph-wide cut
    /// must additionally hold the database's publication barrier.
    #[must_use]
    pub fn complete_node_histories(&self) -> Vec<(NodeId, super::compaction::NodeFullHistory)> {
        let _mutations = MutationReadScope::enter(self);
        self.known_node_ids()
            .into_iter()
            .map(|id| (id, self.node_structural_history(id)))
            .collect()
    }

    /// Captures complete retained histories only for the supplied node IDs.
    ///
    /// Uses the same generation guard and exact history composition as
    /// [`Self::complete_node_histories`], without copying unselected payloads.
    /// The caller validates the selected identities and holds the database
    /// publication barrier when a transaction-consistent cut is required.
    #[doc(hidden)]
    #[must_use]
    pub fn selected_node_histories(
        &self,
        ids: &[NodeId],
    ) -> Vec<(NodeId, super::compaction::NodeFullHistory)> {
        let _mutations = MutationReadScope::enter(self);
        ids.iter()
            .map(|&id| (id, self.node_structural_history(id)))
            .collect()
    }

    /// Returns every structural lifetime for one node, newest first.
    ///
    /// Each materialized [`Node`] contains the complete labels and properties
    /// visible at that lifetime's creation epoch. Unlike the overlay-only
    /// history accessor, this composes cold temporal rows, in-flight base
    /// tombstones, and overlay history, so compaction does not erase the public
    /// audit trail.
    #[must_use]
    pub fn complete_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let _mutations = MutationReadScope::enter(self);
        let history = self.node_structural_history(id);

        history
            .lifetimes
            .iter()
            .rev()
            .map(|lifetime| {
                let labels = history
                    .label_versions
                    .iter()
                    .rev()
                    .find(|(epoch, _)| *epoch <= lifetime.created)
                    .map_or(history.labels.as_slice(), |(_, labels)| labels.as_slice());
                let mut node = Node::with_labels(id, labels.iter().cloned());

                let mut properties: Vec<_> = history.properties.iter().collect();
                properties
                    .sort_unstable_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
                for (key, versions) in properties {
                    if let Some(value) = history_value_at(versions, lifetime.created)
                        && !value.is_null()
                    {
                        node.set_property(key.clone(), value.clone());
                    }
                }

                (lifetime.created, lifetime.deleted, node)
            })
            .collect()
    }

    /// Returns the complete epoch-ascending version log for one node property.
    /// `Value::Null` entries are retained as explicit property tombstones.
    #[must_use]
    pub fn complete_node_property_history_for_key(
        &self,
        id: NodeId,
        key: &str,
    ) -> Vec<(EpochId, Value)> {
        let _mutations = MutationReadScope::enter(self);
        self.node_structural_history(id)
            .properties
            .remove(&PropertyKey::new(key))
            .unwrap_or_default()
    }

    /// Returns complete epoch-ascending logs for every property of one node.
    /// Keys are returned in lexical order and `Value::Null` tombstones remain
    /// present.
    #[must_use]
    pub fn complete_node_property_history(
        &self,
        id: NodeId,
    ) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let _mutations = MutationReadScope::enter(self);
        let mut properties: Vec<_> = self
            .node_structural_history(id)
            .properties
            .into_iter()
            .collect();
        properties.sort_unstable_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
        properties
    }

    /// Returns complete epoch-ascending logs for every property of one edge.
    ///
    /// This composes cold temporal sidecars with the mutable overlay, including
    /// explicit `Value::Null` tombstones. Keys are returned in lexical order.
    /// The merge guard pins one compact/overlay generation for the traversal.
    #[must_use]
    pub fn complete_edge_property_history(
        &self,
        id: EdgeId,
    ) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let _mutations = MutationReadScope::enter(self);
        let mut properties: Vec<_> = self.edge_full_history(id).properties.into_iter().collect();
        properties.sort_unstable_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
        properties
    }

    /// Cold-base structural lives for `id` (live CSR or retained closed row),
    /// with in-flight base tombstones applied. Empty when the base has no row.
    fn base_edge_lifetimes(&self, id: EdgeId) -> Vec<super::compaction::EdgeLifetime> {
        use super::compaction::EdgeLifetime;

        let base = self.base.load();
        let mut lifetimes: Vec<EdgeLifetime> = base
            .structural_edge_rows()
            .into_iter()
            .filter(|(edge_id, _)| *edge_id == id)
            .map(|(_, iv)| EdgeLifetime::new(iv.from(), (!iv.is_open()).then(|| iv.to())))
            .collect();
        if let Some(tomb) = self.deleted_from_base_edges.read().get(&id)
            && tomb.epoch != EpochId::PENDING
            && let Some(open) = lifetimes
                .iter_mut()
                .rev()
                .find(|lifetime| lifetime.deleted.is_none())
        {
            open.deleted = Some(tomb.epoch);
        }
        lifetimes.sort_by_key(|lifetime| {
            (
                lifetime.created,
                lifetime.deleted.unwrap_or(EpochId::PENDING),
            )
        });
        lifetimes
    }

    /// Indexes the compact generation's structural rows in one traversal.
    /// `targets` limits retained map entries while still keeping scan cost
    /// linear in the immutable inventory.
    fn base_edge_lifetime_inventory(
        &self,
        targets: Option<&FxHashSet<EdgeId>>,
    ) -> FxHashMap<EdgeId, Vec<super::compaction::EdgeLifetime>> {
        use super::compaction::EdgeLifetime;

        #[cfg(test)]
        self.base_edge_inventory_scans
            .fetch_add(1, Ordering::Relaxed);

        let mut inventory: FxHashMap<EdgeId, Vec<EdgeLifetime>> = FxHashMap::default();
        for (id, interval) in self.base.load().structural_edge_rows() {
            if targets.is_none_or(|targets| targets.contains(&id)) {
                inventory.entry(id).or_default().push(EdgeLifetime::new(
                    interval.from(),
                    (!interval.is_open()).then(|| interval.to()),
                ));
            }
        }
        for (id, tombstone) in self.deleted_from_base_edges.read().iter() {
            if tombstone.epoch == EpochId::PENDING
                || targets.is_some_and(|targets| !targets.contains(id))
            {
                continue;
            }
            if let Some(open) = inventory.get_mut(id).and_then(|lifetimes| {
                lifetimes
                    .iter_mut()
                    .rev()
                    .find(|lifetime| lifetime.deleted.is_none())
            }) {
                open.deleted = Some(tombstone.epoch);
            }
        }
        for lifetimes in inventory.values_mut() {
            lifetimes.sort_by_key(|lifetime| {
                (
                    lifetime.created,
                    lifetime.deleted.unwrap_or(EpochId::PENDING),
                )
            });
        }
        inventory
    }

    /// Overlay create/delete + property log for `id`, merged with the cold base
    /// when both exist. Overlay-only promote restamps create at `current_epoch()`;
    /// if the base already has validity, the earlier `from` (and any pre-promote
    /// closed lives) is kept. Base-only when the overlay has no chain.
    fn edge_full_history(&self, id: EdgeId) -> super::compaction::EdgeFullHistory {
        let base_lifetimes = self.base_edge_lifetimes(id);
        self.edge_full_history_with_base_lifetimes(id, base_lifetimes)
    }

    /// Merges one edge using caller-preindexed cold structural lifetimes.
    /// Bulk transport qualification and rebuilds use this to avoid rescanning
    /// the entire compact structural inventory once per receipt/edge.
    fn edge_full_history_with_base_lifetimes(
        &self,
        id: EdgeId,
        base_lifetimes: Vec<super::compaction::EdgeLifetime>,
    ) -> super::compaction::EdgeFullHistory {
        use super::compaction::{EdgeFullHistory, EdgeLifetime};

        let dirty = self.dirty_edge_ids.read();
        let base = self.base.load();
        let overlay_is_published = Self::overlay_edge_is_published(&base, &dirty, id);
        let mut properties: FxHashMap<PropertyKey, Vec<(EpochId, Value)>> =
            base.closed_edge_property_history(id).into_iter().collect();
        let base_edge = base_lifetimes
            .iter()
            .find_map(|lifetime| base.get_edge_at_epoch(id, lifetime.created))
            .or_else(|| base.get_edge(id))
            // Zero-width lifetimes are never visible, but their immutable
            // structural identity still belongs in complete audit history.
            .or_else(|| base.retained_edge_identity(id));

        // Legacy/all-open compact stores have no temporal sidecar. Seed their
        // current values at the structural creation epoch so the first temporal
        // merge does not discard them.
        if properties.is_empty()
            && let Some(edge) = &base_edge
        {
            let created = base_lifetimes
                .first()
                .map_or(EpochId::INITIAL, |lifetime| lifetime.created);
            properties.extend(
                edge.properties
                    .iter()
                    .map(|(key, value)| (key.clone(), vec![(created, value.clone())])),
            );
        }
        drop(base);

        if !overlay_is_published {
            return base_edge.map_or_else(EdgeFullHistory::default, |edge| EdgeFullHistory {
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type.clone(),
                lifetimes: base_lifetimes,
                properties,
            });
        }
        drop(dirty);

        let overlay = self.overlay.load();
        let mut overlay_hist = overlay.get_edge_history(id);
        if !overlay_hist.is_empty() {
            overlay_hist.reverse(); // oldest first
            let first = &overlay_hist[0].2;
            for (key, overlay_versions) in overlay.edge_property_history(id) {
                let merged = properties.entry(key).or_default();
                merge_property_versions(merged, overlay_versions);
            }
            let overlay_lifetimes: Vec<EdgeLifetime> = overlay_hist
                .iter()
                .map(|(created, deleted, _)| EdgeLifetime::new(*created, *deleted))
                .collect();
            return EdgeFullHistory {
                src: first.src,
                dst: first.dst,
                edge_type: first.edge_type.clone(),
                lifetimes: super::compaction::merge_edge_lifetimes(
                    &base_lifetimes,
                    &overlay_lifetimes,
                ),
                properties,
            };
        }

        if let Some(edge) = base_edge {
            return EdgeFullHistory {
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type.clone(),
                lifetimes: base_lifetimes,
                properties,
            };
        }

        EdgeFullHistory::default()
    }

    /// Overlay chains + base live/closed ids + in-flight base tombstones.
    fn known_edge_ids(&self) -> Vec<EdgeId> {
        let mut ids = FxHashSet::default();
        ids.extend(self.overlay.load().all_known_edge_ids());
        ids.extend(self.deleted_from_base_edges.read().keys().copied());
        let base = self.base.load();
        ids.extend(base.live_original_edge_ids());
        ids.extend(base.closed_edge_ids());
        let mut ids: Vec<EdgeId> = ids.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns complete histories for every known edge in stable ID order.
    ///
    /// This includes current overlay/base edges and closed or deleted identities
    /// retained only by temporal sidecars or base tombstones. Each returned
    /// [`EdgeFullHistory`](super::compaction::EdgeFullHistory) has structural
    /// lifetimes ordered oldest first and full per-property logs ordered by
    /// ascending epoch, including `Null` tombstones.
    ///
    /// The merge guard pins one base/overlay generation for the traversal.
    /// Callers that require a transaction-publication-consistent graph-wide cut
    /// must additionally hold the database's publication barrier.
    #[must_use]
    pub fn complete_edge_histories(&self) -> Vec<(EdgeId, super::compaction::EdgeFullHistory)> {
        let _mutations = MutationReadScope::enter(self);
        let ids = self.known_edge_ids();
        let mut base_lifetimes = self.base_edge_lifetime_inventory(None);
        ids.into_iter()
            .map(|id| {
                let history = self.edge_full_history_with_base_lifetimes(
                    id,
                    base_lifetimes.remove(&id).unwrap_or_default(),
                );
                (id, history)
            })
            .collect()
    }

    /// Captures complete retained histories only for the supplied edge IDs.
    ///
    /// Scans cold structural rows once for the selected set, then composes
    /// each selected payload under the same generation guard as the complete
    /// capture. It does not perform a whole-base scan for each selected edge.
    /// The caller validates IDs and retains the database publication barrier.
    #[doc(hidden)]
    #[must_use]
    pub fn selected_edge_histories(
        &self,
        ids: &[EdgeId],
    ) -> Vec<(EdgeId, super::compaction::EdgeFullHistory)> {
        let _mutations = MutationReadScope::enter(self);
        let targets = ids.iter().copied().collect();
        let base_lifetimes = self.base_edge_lifetime_inventory(Some(&targets));
        ids.iter()
            .map(|&id| {
                let history = self.edge_full_history_with_base_lifetimes(
                    id,
                    base_lifetimes.get(&id).cloned().unwrap_or_default(),
                );
                (id, history)
            })
            .collect()
    }

    /// Returns every structural lifetime for one edge, newest first.
    ///
    /// Endpoints and relationship type are immutable identity metadata. Each
    /// materialized [`Edge`] contains the properties visible at that lifetime's
    /// creation epoch, composed across the cold base and live overlay.
    #[must_use]
    pub fn complete_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let _mutations = MutationReadScope::enter(self);
        let history = self.edge_full_history(id);

        history
            .lifetimes
            .iter()
            .rev()
            .map(|lifetime| {
                let mut edge = Edge::new(id, history.src, history.dst, history.edge_type.clone());
                let mut properties: Vec<_> = history.properties.iter().collect();
                properties
                    .sort_unstable_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
                for (key, versions) in properties {
                    if let Some(value) = history_value_at(versions, lifetime.created)
                        && !value.is_null()
                    {
                        edge.set_property(key.clone(), value.clone());
                    }
                }
                (lifetime.created, lifetime.deleted, edge)
            })
            .collect()
    }

    /// Returns whether this layered generation currently exposes the exact
    /// open edge covered by `receipt`.
    ///
    /// Authority is always read from the pinned current overlay. Same-store
    /// representation changes transfer that exact authority to their prepared
    /// successor; a logical overlay incarnation change rotates it.
    #[doc(hidden)]
    #[must_use]
    pub fn is_transport_extract_edge(&self, receipt: &TransportEdgeReceipt) -> bool {
        let _generation = GenerationReadScope::enter(self);
        self.is_transport_extract_edge_unchanged_since(receipt, self.overlay.load().current_epoch())
    }

    /// Returns whether this receipt still covers its original history with an open final
    /// structural identity. Property changes do not alter transport ownership.
    #[doc(hidden)]
    #[must_use]
    pub fn is_transport_extract_edge_unchanged_since(
        &self,
        receipt: &TransportEdgeReceipt,
        since: EpochId,
    ) -> bool {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        if !overlay.transport_receipt_belongs_to_current_incarnation(receipt) {
            return false;
        }
        let history = self.edge_full_history(receipt.edge_id());
        history.src == receipt.source()
            && history.dst == receipt.destination()
            && history.edge_type.as_str() == receipt.edge_type()
            && receipt.history_state(
                history
                    .lifetimes
                    .iter()
                    .rev()
                    .map(|life| (life.created, life.deleted)),
                since,
                since,
            ) == TransportEdgeState::Open
    }

    /// Mints a property-mutation-only grant for one exact open carried edge in
    /// this pinned layered generation.
    #[doc(hidden)]
    #[must_use]
    pub fn grant_transport_edge_mutation(
        &self,
        receipt: &TransportEdgeReceipt,
        write_authority: &crate::graph::write_permit::WriteAuthority,
    ) -> Option<TransportEdgeMutationGrant> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        if !overlay.accepts_held_write_authority(write_authority)
            || !self.is_transport_extract_edge(receipt)
        {
            return None;
        }
        Some(TransportEdgeMutationGrant::from_receipt(receipt))
    }

    /// Revalidates a staged carried-edge property grant without consulting the
    /// engine's transport-state lock. Only an entirely absent destination is
    /// excused; a deleted/PENDING identity or a missing source still fails the
    /// ordinary endpoint publication rule.
    #[doc(hidden)]
    #[must_use]
    pub fn accepts_transport_edge_missing_destination(
        &self,
        grant: &TransportEdgeMutationGrant,
        missing_endpoint: NodeId,
        write_authority: &crate::graph::write_permit::WriteAuthority,
    ) -> bool {
        let _mutations = MutationReadScope::enter(self);
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        let source_overlay_not_open = overlay
            .get_node_history(grant.source())
            .first()
            .is_some_and(|(created, deleted, _)| *created == EpochId::PENDING || deleted.is_some());
        let source_deleted_from_base = self
            .deleted_from_base_nodes
            .read()
            .contains_key(&grant.source());
        if missing_endpoint != grant.destination()
            || !overlay.accepts_held_write_authority(write_authority)
            || overlay.contains_node_identity(missing_endpoint)
            || Self::base_has_node_identity(&self.base.load(), missing_endpoint)
            // A PENDING source delete remains visible to unrelated ordinary
            // reads by MVCC design, but it is not a stable endpoint for a
            // transport mutation grant. Require the newest structural life to
            // be both committed and open, in addition to the layered current
            // view being present.
            || source_overlay_not_open
            || source_deleted_from_base
            || self.get_node(grant.source()).is_none()
            || !overlay.transport_grant_belongs_to_current_incarnation(grant)
        {
            return false;
        }
        let history = self.edge_full_history(grant.edge_id());
        history.src == grant.source()
            && history.dst == grant.destination()
            && history.edge_type.as_str() == grant.edge_type()
            && grant.history_is_open(
                history
                    .lifetimes
                    .iter()
                    .rev()
                    .map(|life| (life.created, life.deleted)),
                overlay.current_epoch(),
            )
    }

    /// Returns whether this receipt covers its original history with a committed closed
    /// final transport lifetime unchanged since `since`.
    #[doc(hidden)]
    #[must_use]
    pub fn is_transport_extract_closed_edge_unchanged_since(
        &self,
        receipt: &TransportEdgeReceipt,
        since: EpochId,
    ) -> bool {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        if !overlay.transport_receipt_belongs_to_current_incarnation(receipt) {
            return false;
        }
        let history = self.edge_full_history(receipt.edge_id());
        history.src == receipt.source()
            && history.dst == receipt.destination()
            && history.edge_type.as_str() == receipt.edge_type()
            && receipt.history_state(
                history
                    .lifetimes
                    .iter()
                    .rev()
                    .map(|life| (life.created, life.deleted)),
                since,
                since,
            ) == TransportEdgeState::Closed
    }

    /// Classifies a receipt batch from one pinned layered generation using one
    /// compact structural inventory scan.
    #[doc(hidden)]
    #[must_use]
    pub fn classify_transport_extract_edges(
        &self,
        receipts: &[&TransportEdgeReceipt],
        open_since: EpochId,
        closed_since: EpochId,
    ) -> Vec<TransportEdgeState> {
        let _mutations = MutationReadScope::enter(self);
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        let mut states = vec![TransportEdgeState::Invalid; receipts.len()];
        let mut counts: FxHashMap<EdgeId, usize> = FxHashMap::default();
        let targets: FxHashSet<EdgeId> = receipts
            .iter()
            .map(|receipt| {
                *counts.entry(receipt.edge_id()).or_default() += 1;
                receipt.edge_id()
            })
            .collect();
        let mut inventory = self.base_edge_lifetime_inventory(Some(&targets));
        for (position, receipt) in receipts.iter().enumerate() {
            if counts.get(&receipt.edge_id()) != Some(&1)
                || !overlay.transport_receipt_belongs_to_current_incarnation(receipt)
            {
                continue;
            }
            let history = self.edge_full_history_with_base_lifetimes(
                receipt.edge_id(),
                inventory.remove(&receipt.edge_id()).unwrap_or_default(),
            );
            if history.src != receipt.source()
                || history.dst != receipt.destination()
                || history.edge_type.as_str() != receipt.edge_type()
            {
                continue;
            }
            states[position] = receipt.history_state(
                history
                    .lifetimes
                    .iter()
                    .rev()
                    .map(|life| (life.created, life.deleted)),
                open_since,
                closed_since,
            );
        }
        states
    }

    /// Removes the final layered bookkeeping for a successfully filtered
    /// transport batch while the caller holds the generation write barrier.
    ///
    /// The guards deliberately occupy disjoint scopes. Read-only observers do
    /// not take `merge_guard` and may read `dirty_edge_ids` before
    /// `deleted_from_base_edges`; retaining the tombstone write guard while
    /// waiting for the dirty-edge write guard would therefore form a lock-order
    /// cycle with such an observer.
    #[cfg(test)]
    fn clear_purged_transport_bookkeeping(&self, targets: &FxHashSet<EdgeId>) {
        {
            let mut tombstones = self.deleted_from_base_edges.write();
            for id in targets {
                tombstones.remove(id);
            }
        }

        #[cfg(test)]
        let between_lock_scopes_hook = self
            .transport_cleanup_between_lock_scopes_hook
            .read()
            .clone();
        #[cfg(test)]
        if let Some(hook) = between_lock_scopes_hook {
            hook();
        }

        {
            let mut dirty = self.dirty_edge_ids.write();
            for id in targets {
                dirty.remove(id);
            }
        }
        self.deletions_dirty.store(true, Ordering::Release);
    }

    /// Builds and atomically publishes a compact generation with only the
    /// receipt-qualified, already-closed transport histories removed.
    ///
    /// Every non-target node, edge lifetime, label, and property history is
    /// rebuilt through the same temporal compaction pipeline used by
    /// [`Self::merge_overlay_temporal`]. The mutable overlay and its allocator
    /// high-water counters remain installed. The caller-supplied `prepare`
    /// callback may perform fallible serialization/mmap validation but must not
    /// publish. It runs only after complete base and overlay qualification and
    /// must operate on the supplied candidate rather than call back into this
    /// Layered/LPG store: the source overlay's exclusive transition remains
    /// pinned so raw mutation cannot overtake the capture.
    /// `publish` then performs one infallible external metadata move and returns
    /// its exact rollback token. The Layered generation barrier remains closed
    /// until LPG has either completed its prebuilt physical purge or invoked
    /// `rollback` after a hostile unwind. Thus no reader can retain a failed
    /// base candidate, partial routing state, or not-yet-purged overlay.
    ///
    /// Qualification failure returns `Ok(false)` without changing any state.
    /// A preparation/build failure returns `Err` with the old base and all
    /// retryable logical tombstones still installed.
    ///
    /// `publish` and `rollback` are representation-pointer moves, not general
    /// callbacks. They must not call or spawn-and-join a same-store read or
    /// mutation while the Layered publication cut is closed. A publisher that
    /// can unwind must do so before moving external state; arbitrary callback
    /// side effects cannot be synthesized or reversed by this generic seam.
    ///
    /// # Errors
    ///
    /// Returns an error if temporal rebuilding or representation preparation
    /// fails.
    #[doc(hidden)]
    pub fn purge_transport_extract_edges<P, R, Q>(
        &self,
        receipts: &[&TransportEdgeReceipt],
        write_authority: &crate::graph::write_permit::WriteAuthority,
        prepare: impl FnOnce(Arc<CompactStore>) -> Result<(Arc<CompactStore>, P), String>,
        publish: impl FnOnce(P) -> R,
        rollback: impl FnOnce(R) -> Q,
    ) -> Result<bool, String> {
        enum PreparationFailure {
            Qualification,
            Build(String),
        }

        // Declared before the mutation scope so hostile-unwind retirement runs
        // only after that broader generation exclusion has drained.
        use crate::graph::lpg::PreparedPurgeOutcome;
        let retirements = GenerationRetirements::new();
        let mut mutations = MutationWriteScope::enter(self);
        let overlay = self.overlay.load_full();
        if !overlay.accepts_held_write_authority(write_authority) {
            return Ok(false);
        }
        if receipts.is_empty() {
            return Ok(true);
        }
        let targets: FxHashSet<EdgeId> = receipts.iter().map(|receipt| receipt.edge_id()).collect();
        if targets.len() != receipts.len() || targets.contains(&EdgeId::INVALID) {
            return Ok(false);
        }

        // Occupancy is sampled only to split the batch. The LPG seam revalidates
        // both exact sets under its exclusive transition guard, so a raw overlay
        // mutation between this snapshot and admission fails closed.
        let overlay_ids: FxHashSet<EdgeId> = overlay.all_known_edge_ids().into_iter().collect();
        let resident_receipts: Vec<_> = receipts
            .iter()
            .copied()
            .filter(|receipt| overlay_ids.contains(&receipt.edge_id()))
            .collect();
        let authority_only_receipts: Vec<_> = receipts
            .iter()
            .copied()
            .filter(|receipt| !overlay_ids.contains(&receipt.edge_id()))
            .collect();

        let overlay_for_generation = Arc::clone(&overlay);
        let publication = overlay
            .purge_transport_extract_edges_after_prepare_and_publish_with_rollback(
                &resident_receipts,
                &authority_only_receipts,
                |_scope| {
                    // The engine admits this at a lifecycle/publication cut.
                    // Defend the core boundary too: target histories and every
                    // Layered routing object are recaptured only after LPG has
                    // excluded raw-store mutation and revalidated occupancy.
                    if self
                        .pending_base_edge_deletes
                        .read()
                        .values()
                        .any(|ids| ids.iter().any(|id| targets.contains(id)))
                    {
                        return Err(PreparationFailure::Qualification);
                    }
                    let frontier = overlay.current_epoch();
                    if targets.iter().any(|id| {
                        self.deleted_from_base_edges
                            .read()
                            .get(id)
                            .is_some_and(|delete| {
                                delete.epoch == EpochId::PENDING || delete.epoch > frontier
                            })
                    }) {
                        return Err(PreparationFailure::Qualification);
                    }

                    let all_edge_ids = self.known_edge_ids();
                    let mut base_inventory = self.base_edge_lifetime_inventory(None);
                    let edge_histories: FxHashMap<EdgeId, super::compaction::EdgeFullHistory> =
                        all_edge_ids
                            .iter()
                            .copied()
                            .map(|id| {
                                let history = self.edge_full_history_with_base_lifetimes(
                                    id,
                                    base_inventory.remove(&id).unwrap_or_default(),
                                );
                                (id, history)
                            })
                            .collect();
                    for receipt in receipts {
                        let Some(history) = edge_histories.get(&receipt.edge_id()) else {
                            return Err(PreparationFailure::Qualification);
                        };
                        if history.src != receipt.source()
                            || history.dst != receipt.destination()
                            || history.edge_type.as_str() != receipt.edge_type()
                            || receipt.history_state(
                                history
                                    .lifetimes
                                    .iter()
                                    .rev()
                                    .map(|life| (life.created, life.deleted)),
                                frontier,
                                frontier,
                            ) != TransportEdgeState::Closed
                            || history
                                .properties
                                .values()
                                .flatten()
                                .any(|(epoch, _)| *epoch == EpochId::PENDING || *epoch > frontier)
                        {
                            return Err(PreparationFailure::Qualification);
                        }
                    }

                    let revision = mutations
                        .checked_revision()
                        .map_err(PreparationFailure::Build)?;
                    let node_ids = self.known_node_ids();
                    let retained_edge_ids: Vec<EdgeId> = all_edge_ids
                        .into_iter()
                        .filter(|id| !targets.contains(id))
                        .collect();
                    let all_open = super::from_graph_store_preserving_ids(self)
                        .map_err(|error| PreparationFailure::Build(error.to_string()))?;
                    let filtered = all_open
                        .upgrade_nodes_temporal(|nid| {
                            committed_history(self.node_property_full_history(nid), frontier)
                        })
                        .install_temporal_nodes(
                            |nid| {
                                super::compaction::committed_node_history(
                                    self.node_structural_history(nid),
                                    frontier,
                                )
                            },
                            node_ids,
                        )
                        .map_err(PreparationFailure::Build)?
                        .upgrade_rels_temporal(
                            |eid| {
                                edge_histories.get(&eid).cloned().map_or_else(
                                    super::compaction::EdgeFullHistory::default,
                                    |history| {
                                        super::compaction::committed_edge_history(history, frontier)
                                    },
                                )
                            },
                            retained_edge_ids,
                        )
                        .with_property_history_floor(Some(
                            overlay.retained_history_floor().max(
                                self.base
                                    .load()
                                    .property_history_floor()
                                    .unwrap_or(frontier),
                            ),
                        ));

                    let mut next_dirty_nodes = self.dirty_node_ids.read().clone();
                    let mut next_dirty_edges = self.dirty_edge_ids.read().clone();
                    let mut next_deleted_nodes = self.deleted_from_base_nodes.read().clone();
                    let mut next_pending_node_deletes =
                        self.pending_base_node_deletes.read().clone();
                    let mut next_deleted_edges = self.deleted_from_base_edges.read().clone();
                    let mut next_pending_edge_deletes =
                        self.pending_base_edge_deletes.read().clone();
                    for target in &targets {
                        next_dirty_edges.remove(target);
                        next_deleted_edges.remove(target);
                    }
                    next_pending_node_deletes.retain(|_, ids| !ids.is_empty());
                    next_pending_edge_deletes.retain(|_, ids| !ids.is_empty());

                    let (prepared_base, metadata) =
                        prepare(Arc::new(filtered)).map_err(PreparationFailure::Build)?;
                    let image = PreparedLayerGeneration {
                        base: prepared_base,
                        overlay: Arc::clone(&overlay_for_generation),
                        dirty_node_ids: std::mem::take(&mut next_dirty_nodes),
                        dirty_edge_ids: std::mem::take(&mut next_dirty_edges),
                        deleted_from_base_nodes: std::mem::take(&mut next_deleted_nodes),
                        pending_base_node_deletes: std::mem::take(&mut next_pending_node_deletes),
                        deleted_from_base_edges: std::mem::take(&mut next_deleted_edges),
                        pending_base_edge_deletes: std::mem::take(&mut next_pending_edge_deletes),
                        deletions_dirty: true,
                        representation_transfer: None,
                    };
                    Ok::<_, PreparationFailure>(ReadyLayerPublication {
                        revision,
                        image,
                        external: metadata,
                    })
                },
                |ready| {
                    let (layered, metadata) = Self::publish_prepared_layer_generation(ready);
                    // Keep the global lock order at Layered publication then
                    // external/tier metadata. The callback is contractually an
                    // infallible owned-state move; still restore Layered before
                    // propagating a violating unwind so no candidate escapes.
                    let external =
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            publish(metadata)
                        })) {
                            Ok(external) => external,
                            Err(payload) => {
                                *retirements.rollback_layer.borrow_mut() =
                                    Some(Self::restore_published_layer_generation(layered));
                                std::panic::resume_unwind(payload);
                            }
                        };
                    ReversibleLayerPublication { external, layered }
                },
                |publication| {
                    let ReversibleLayerPublication { external, layered } = publication;
                    let external_rollback =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            rollback(external)
                        }));
                    *retirements.rollback_layer.borrow_mut() =
                        Some(Self::restore_published_layer_generation(layered));
                    match external_rollback {
                        Ok(retirement) => {
                            *retirements.rollback_external.borrow_mut() = Some(retirement);
                        }
                        Err(payload) => std::panic::resume_unwind(payload),
                    }
                },
            );
        let publication = match publication {
            Ok(publication) => publication,
            Err(PreparationFailure::Qualification) => return Ok(false),
            Err(PreparationFailure::Build(error)) => return Err(error),
        };
        let ReversibleLayerPublication { external, layered } = match publication {
            PreparedPurgeOutcome::Rejected => return Ok(false),
            PreparedPurgeOutcome::Abandoned(ready) => {
                *retirements.unpublished.borrow_mut() = Some(ready.into_retirement());
                return Ok(false);
            }
            PreparedPurgeOutcome::UnwoundBeforePublication { prepared, payload } => {
                *retirements.unpublished.borrow_mut() = Some(prepared.into_retirement());
                std::panic::resume_unwind(payload);
            }
            PreparedPurgeOutcome::Published(publication) => publication,
        };
        let retired_layer = Self::commit_published_layer_generation(layered);
        drop(mutations);
        drop(external);
        drop(retired_layer);
        Ok(true)
    }

    /// Merges the overlay into a fresh **temporal** `CompactStore` — folding each
    /// node's full property history into validity-interval columns, packing
    /// Option A edge adjacency (open prefix + closed tails), and rebuilding the
    /// derived current CSR from that prefix — swaps it in as the base, and
    /// clears the overlay (SP2 / Task 4).
    ///
    /// Like [`merge_overlay_in_place`](Self::merge_overlay_in_place), but the new
    /// base preserves version history (as-of reads), not just current values:
    /// numeric columns fold their full history; other-typed columns keep their
    /// current value. Concurrency is identical: writers are excluded for the
    /// rebuild, existing readers continue on the old generation, and new reads
    /// pause only during the short coherent base/overlay/routing publication.
    ///
    /// # Errors
    ///
    /// Returns an error if rebuilding/preparing the successor fails or the
    /// caller does not hold the current overlay's sealed write authority.
    pub fn merge_overlay_temporal(&self) -> Result<(), String> {
        self.merge_overlay_temporal_with_publication(
            |candidate| Ok((candidate, ())),
            |()| (),
            |()| (),
        )
    }

    /// Temporal merge with one reversible external representation publication.
    ///
    /// `prepare` receives the complete unpublished temporal base. It may do
    /// fallible serialization, mmap, or validation and returns both the exact
    /// base representation to install and an opaque allocation-complete plan.
    /// `publish` runs only after the Layered generation and exact derived
    /// indexes are installed behind the still-exclusive generation cut. It
    /// must be an infallible owned-state move; if it can unwind, it must do so
    /// before changing external state. `rollback` reverses a returned external
    /// token and runs before Layered restoration on hostile post-publication
    /// unwind.
    ///
    /// On success the returned external retirement token is handed to the
    /// caller only after the LPG transition and Layered mutation/publication
    /// guards have drained. The displaced Layered generation is likewise
    /// retired only after those guards drain. Preparation error changes
    /// neither Layered nor external state.
    ///
    /// The exact source LPG transition and named-topology gate cover temporal
    /// traversal, external preparation, exact index freezing, publication, and
    /// rollback as one source cut. Callbacks must therefore operate only on the
    /// supplied candidate and their owned external plan: none may re-enter
    /// source LPG mutation or named-graph DDL. Publication callbacks must not
    /// call or spawn-and-join reads or mutations on this Layered store while
    /// the generation cut is closed.
    ///
    /// # Errors
    ///
    /// Returns a core or preparation error without publication, or reports a
    /// missing exact write authority before a successor can become visible.
    #[doc(hidden)]
    pub fn merge_overlay_temporal_with_publication<P, R, Q>(
        &self,
        prepare: impl FnOnce(Arc<CompactStore>) -> Result<(Arc<CompactStore>, P), String>,
        publish: impl FnOnce(P) -> R,
        rollback: impl FnOnce(R) -> Q,
    ) -> Result<R, String> {
        // Compact the whole committed frontier (retain nothing committed hot).
        self.merge_overlay_temporal_retaining_with_publication(
            None, false, prepare, publish, rollback,
        )
        .map_err(TemporalMergeFailure::into_reason)
    }

    /// Incremental temporal merge (SP2 slice 5): compacts committed history at or
    /// before `retain_after` into the cold base, while nodes with committed
    /// history *after* `retain_after` stay **whole** and hot in the overlay — so
    /// the recent scrub range isn't re-compacted on every merge.
    ///
    /// The split is per node, not per property: a node is served entirely from
    /// the base (cold) or entirely from the overlay (recently touched), so the
    /// existing dirty read-routing serves both without a combined base/overlay
    /// read — the hot-path read semantics are unchanged. Only COMMITTED history
    /// is folded or re-seeded; uncommitted (PENDING) versions are never baked
    /// into cold nor re-seeded (which would disconnect them from their
    /// transaction), matching the quiescent merge contract.
    ///
    /// `merge_overlay_temporal` is the `retain_after = latest-committed` case
    /// (nothing committed is hot, so the overlay is reset).
    ///
    /// # Errors
    ///
    /// Returns an error if rebuilding/re-seeding the successor fails or the
    /// caller does not hold the current overlay's sealed write authority.
    pub fn merge_overlay_temporal_retaining(&self, retain_after: EpochId) -> Result<(), String> {
        self.merge_overlay_temporal_retaining_with_publication(
            Some(retain_after),
            false,
            |candidate| Ok((candidate, ())),
            |()| (),
            |()| (),
        )
        .map_err(TemporalMergeFailure::into_reason)
    }

    fn merge_overlay_temporal_retaining_with_publication<P, R, Q>(
        &self,
        retain_after: Option<EpochId>,
        require_native_source: bool,
        prepare: impl FnOnce(Arc<CompactStore>) -> Result<(Arc<CompactStore>, P), String>,
        publish: impl FnOnce(P) -> R,
        rollback: impl FnOnce(R) -> Q,
    ) -> Result<R, TemporalMergeFailure> {
        // Stop writers while the complete successor generation is built.
        // Rollback retirements are declared before the mutation guard so an
        // unwind always drains the guard before either destructor can run.
        let retirements = GenerationRetirements::new();
        let mut mutations = MutationWriteScope::enter(self);
        let current_overlay = self.overlay.load_full();
        // Preserve the process-wide lock order: Layered mutation exclusion,
        // named topology, then the source LPG transition. This exact source
        // proof remains held from the first graph read through temporal/index
        // preparation, publication, hostile rollback, and commit.
        let named_topology: PinnedNamedGraphTopology = current_overlay.pin_named_graph_topology();
        let Some(transition) = current_overlay.pin_exclusive_unframed_transition() else {
            return Err(TemporalMergeFailure::Admission(
                "current overlay write scope is not held; refusing temporal generation replacement",
            ));
        };
        if require_native_source {
            transition
                .require_native_compact_source()
                .map_err(TemporalMergeFailure::Admission)?;
        }
        let retirements_ref = &retirements;
        let snapshot_overlay = Arc::clone(&current_overlay);
        let (retired, external_retirement) = self
            .install_with_pinned_empty_successor_and_publication(
                PinnedEmptySuccessorPublication {
                    current_overlay: &current_overlay,
                    representation_topology: Some(&named_topology),
                    transition: &transition,
                    mutation: &mut mutations,
                    retirements: &retirements,
                },
                move |_scope, successor| {
                    // The helper has already frozen every direct vector/text
                    // mutation surface. Together with the source LPG and named
                    // topology cuts acquired above, this makes the following
                    // temporal graph traversal and exact physical-index fork
                    // one coherent source generation.
                    #[cfg(test)]
                    if let Some(barrier) = self.temporal_snapshot_barrier.read().clone() {
                        barrier.wait();
                        barrier.wait();
                    }
                    let frontier = snapshot_overlay.current_epoch();
                    let retain_after = retain_after.unwrap_or(frontier);

                    // Capture the nodes to keep hot: those with a COMMITTED
                    // version after `retain_after`. Their whole committed
                    // history is re-seeded after the swap; their older history
                    // is also folded into the base (redundant but shadowed by
                    // the overlay copy).
                    let mut hot: Vec<(
                        NodeId,
                        Vec<String>,
                        Vec<(PropertyKey, Vec<(EpochId, Value)>)>,
                    )> = Vec::new();
                    for nid in snapshot_overlay.all_node_ids() {
                        let history =
                            committed_history(self.node_property_full_history(nid), frontier);
                        let recent = history.iter().any(|(_, versions)| {
                            versions.iter().any(|(epoch, _)| *epoch > retain_after)
                        });
                        if recent {
                            let labels = snapshot_overlay
                                .get_node(nid)
                                .map(|n| n.labels.iter().map(ToString::to_string).collect())
                                .unwrap_or_default();
                            hot.push((nid, labels, history));
                        }
                    }

                    // Compact committed history <= retain_after into the cold
                    // base while the same graph/index cut remains pinned.
                    let node_ids = self.known_node_ids();
                    let extra_edge_ids = self.known_edge_ids();
                    let all_open = super::from_graph_store_preserving_ids(self)
                        .map_err(|error| error.to_string())?;
                    let temporal = all_open
                        .upgrade_nodes_temporal(|nid| {
                            committed_history(self.node_property_full_history(nid), retain_after)
                        })
                        // Structural history is compacted through the committed
                        // frontier, including recently deleted nodes that have
                        // no current-table row. Retained hot nodes still shadow
                        // their redundant sidecar history.
                        .install_temporal_nodes(
                            |nid| {
                                super::compaction::committed_node_history(
                                    self.node_structural_history(nid),
                                    frontier,
                                )
                            },
                            node_ids,
                        )?
                        .upgrade_rels_temporal(
                            |eid| {
                                super::compaction::committed_edge_history(
                                    self.edge_full_history(eid),
                                    retain_after,
                                )
                            },
                            extra_edge_ids,
                        )
                        .with_property_history_floor(Some(
                            snapshot_overlay.retained_history_floor().max(
                                self.base
                                    .load()
                                    .property_history_floor()
                                    .unwrap_or(frontier),
                            ),
                        ));
                    let (next_base, external) = prepare(Arc::new(temporal))?;

                    // Keep the actual external pair outside this borrow-only
                    // catch. A late reseed/binding failure must retire it only
                    // after the source cut and completed transfer fences drain.
                    let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        // Re-seed the retained hot nodes (whole, committed) off-side before
                        // publication. Temporarily replay structural creates from INITIAL,
                        // then restore the exact source frontier before the successor can
                        // become visible. Every fallible allocation happens here.
                        successor.set_epoch(EpochId::INITIAL);
                        let mut dirty_nodes = FxHashSet::default();
                        for (nid, labels, history) in hot {
                            let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                            successor
                                .create_node_with_id(nid, &refs)
                                .map_err(|e| format!("re-seed node {nid:?}: {e}"))?;
                            #[cfg(test)]
                            match self.generation_reseed_action.swap(0, Ordering::SeqCst) {
                                1 => {
                                    self.generation_reseed_hits.fetch_add(1, Ordering::SeqCst);
                                    return Err(
                                        "generation retained-hot reseed rejected".to_owned()
                                    );
                                }
                                2 => {
                                    self.generation_reseed_hits.fetch_add(1, Ordering::SeqCst);
                                    panic!("generation retained-hot reseed unwind");
                                }
                                _ => {}
                            }
                            for (key, versions) in history {
                                for (epoch, value) in versions {
                                    successor.hydrate_node_property_at_epoch(
                                        nid,
                                        key.as_str(),
                                        value,
                                        epoch,
                                    );
                                }
                            }
                            if next_base.get_node(nid).is_some() {
                                dirty_nodes.insert(nid);
                            }
                        }
                        successor.set_epoch(frontier);
                        successor
                            .install_compact_base(Arc::clone(&next_base))
                            .map_err(str::to_owned)?;
                        Self::raise_overlay_allocator_floors(&next_base, &successor);
                        let mut prepared = PreparedLayerGeneration::empty(
                            Arc::clone(&next_base),
                            Arc::clone(&successor),
                        );
                        prepared.dirty_node_ids = dirty_nodes;
                        Ok::<_, String>(prepared)
                    }));
                    match prepared {
                        Ok(Ok(image)) => Ok((image, external)),
                        Ok(Err(error)) => {
                            *retirements_ref.partial.borrow_mut() = Some((next_base, external));
                            Err(error)
                        }
                        Err(payload) => {
                            *retirements_ref.partial.borrow_mut() = Some((next_base, external));
                            std::panic::resume_unwind(payload)
                        }
                    }
                },
                publish,
                rollback,
            )
            .map_err(TemporalMergeFailure::Preparation)?;
        drop(transition);
        drop(named_topology);
        drop(mutations);
        drop(retired);
        Ok(external_retirement)
    }

    /// Whole-state as-of scrub (SP3): every node as it was at `epoch`, projected
    /// to the property values valid then. Routes through the layered read path —
    /// the temporal cold base serves history older than the overlay's retained
    /// range, the overlay serves the rest — and excludes nodes that did not yet
    /// exist at `epoch`. The headline cold-tier scrub read.
    #[must_use]
    pub fn nodes_at_epoch(&self, epoch: EpochId) -> Vec<Node> {
        let _generation = GenerationReadScope::enter(self);
        let mut ids = FxHashSet::default();
        ids.extend(self.node_ids());
        ids.extend(self.base.load().temporal_node_ids());
        ids.extend(self.overlay.load().all_node_ids());
        let mut ids: Vec<NodeId> = ids.into_iter().collect();
        ids.sort_unstable();
        ids.into_iter()
            .filter_map(|id| self.get_node_at_epoch(id, epoch))
            .collect()
    }

    /// Single-property as-of. Dirty overlay versions win when they cover
    /// `epoch`; otherwise the temporal cold base (pre-promote history).
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_node_deleted_from_base_at(id, epoch, TransactionId::INVALID) {
            return None;
        }
        let dirty = self.is_node_dirty(id);
        let base = self.base.load();
        if dirty {
            let overlay = self.overlay.load();
            if let Some(value) = overlay.get_node_property_at_epoch(id, key, epoch) {
                return Some(value);
            }
            if Self::overlay_property_covers_epoch(&overlay, id, key, epoch) {
                return None;
            }
            return base.get_node_property_at_epoch(id, key, epoch);
        }
        if Self::base_has_node_identity(&base, id) {
            return base.get_node_property_at_epoch(id, key, epoch);
        }
        self.overlay
            .load()
            .get_node_property_at_epoch(id, key, epoch)
    }

    /// Neighbors visible at `epoch`.
    ///
    /// `PENDING` is current 1-hop: derived current CSR + overlay (`neighbors`).
    /// Other epochs: cold as-of ∪ overlay as-of, minus tombstones with
    /// `deleted_epoch <= viewing_epoch` (same visibility as
    /// [`Self::get_edge_at_epoch`] / `is_edge_deleted_from_base_at`).
    #[must_use]
    pub fn neighbors_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
    ) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let mut out = Vec::new();
        self.fill_neighbors_at_epoch(node, direction, epoch, &mut out);
        out
    }

    /// Fills `out` with neighbors visible at `epoch` (clears `out` first).
    pub fn fill_neighbors_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<NodeId>,
    ) {
        let _generation = GenerationReadScope::enter(self);
        self.fill_neighbors_of_types_at_epoch(node, direction, epoch, &[], out);
    }

    /// Dest-only as-of fill restricted to `types` (empty = every RelTable).
    pub fn fill_neighbors_of_types_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        types: &[String],
        out: &mut Vec<NodeId>,
    ) {
        let _generation = GenerationReadScope::enter(self);
        out.clear();
        if self.overlay_is_cold() {
            self.base
                .load()
                .fill_neighbors_at_epoch_of_types(node, direction, epoch, types, out);
            return;
        }
        if epoch == EpochId::PENDING && types.is_empty() {
            out.extend(self.neighbors(node, direction));
            return;
        }
        let tx = TransactionId::INVALID;
        let mut seen: FxHashSet<EdgeId> = FxHashSet::default();
        let type_ok = |edge_type: &arcstr::ArcStr| {
            types.is_empty()
                || types
                    .iter()
                    .any(|ty| edge_type.eq_ignore_ascii_case(ty.as_str()))
        };

        for (target, eid) in self.base.load().edges_from_at_epoch(node, direction, epoch) {
            if seen.contains(&eid) {
                continue;
            }
            if self.is_node_deleted_from_base_at(target, epoch, tx) {
                continue;
            }
            if let Some(e) = self.get_edge_at_epoch(eid, epoch)
                && type_ok(&e.edge_type)
                && seen.insert(eid)
            {
                out.push(target);
            }
        }

        let overlay = self.overlay.load();
        for (target, eid) in overlay.edges_from_versioned(node, direction, epoch, tx) {
            if seen.contains(&eid) {
                continue;
            }
            if self.is_node_deleted_from_base_at(target, epoch, tx) {
                continue;
            }
            if let Some(e) = self.get_edge_at_epoch(eid, epoch)
                && type_ok(&e.edge_type)
                && seen.insert(eid)
            {
                out.push(target);
            }
        }

        out.sort_unstable();
        out.dedup();
    }

    /// Every edge whose interval contains `epoch`, plus overlay edges
    /// visible then.
    ///
    /// `PENDING` is the current snapshot (derived open CSR ∪ overlay).
    /// A closed (deleted) edge is absent at and after its delete epoch.
    #[must_use]
    pub fn edges_at_epoch(&self, epoch: EpochId) -> Vec<Edge> {
        let _generation = GenerationReadScope::enter(self);
        if epoch == EpochId::PENDING {
            return self.current_edges();
        }

        let mut seen: FxHashSet<EdgeId> = FxHashSet::default();
        let mut out = Vec::new();

        for edge in self.base.load().edges_at_epoch(epoch) {
            if let Some(live) = self.get_edge_at_epoch(edge.id, epoch)
                && seen.insert(live.id)
            {
                out.push(live);
            }
        }

        let overlay = self.overlay.load();
        let mut nodes = overlay.all_node_ids();
        nodes.extend(self.base.load().node_ids());
        for nid in nodes {
            for (_, eid) in overlay.edges_from_versioned(
                nid,
                Direction::Outgoing,
                epoch,
                TransactionId::INVALID,
            ) {
                if seen.contains(&eid) {
                    continue;
                }
                if let Some(live) = self.get_edge_at_epoch(eid, epoch)
                    && seen.insert(live.id)
                {
                    out.push(live);
                }
            }
        }

        let dirty_edge_ids: Vec<EdgeId> = self.dirty_edge_ids.read().iter().copied().collect();
        for eid in dirty_edge_ids {
            if seen.contains(&eid) {
                continue;
            }
            if let Some(live) = self.get_edge_at_epoch(eid, epoch)
                && seen.insert(live.id)
            {
                out.push(live);
            }
        }
        let deleted_edge_ids: Vec<EdgeId> = self
            .deleted_from_base_edges
            .read()
            .keys()
            .copied()
            .collect();
        for eid in deleted_edge_ids {
            if seen.contains(&eid) {
                continue;
            }
            if let Some(live) = self.get_edge_at_epoch(eid, epoch)
                && seen.insert(live.id)
            {
                out.push(live);
            }
        }
        out
    }

    fn current_edges(&self) -> Vec<Edge> {
        let _generation = GenerationReadScope::enter(self);
        let mut seen: FxHashSet<EdgeId> = FxHashSet::default();
        let mut out = Vec::new();
        for nid in self.node_ids() {
            for (_, eid) in self.edges_from(nid, Direction::Outgoing) {
                if seen.insert(eid)
                    && let Some(edge) = self.get_edge(eid)
                {
                    out.push(edge);
                }
            }
        }
        out
    }

    /// Per-type edge frames at `epoch` (base ∪ overlay).
    #[must_use]
    pub fn edge_scrub_at_epoch(&self, epoch: EpochId) -> Vec<RelTableScrub> {
        let _generation = GenerationReadScope::enter(self);
        if self.overlay_is_cold() {
            return self.base.load().edge_scrub_at_epoch(epoch);
        }
        rel_frames_from_edges(self.edges_at_epoch(epoch))
    }

    /// Node + edge as-of scrub (columnar node frames + per-type edge frames).
    #[must_use]
    pub fn graph_scrub_at_epoch(&self, epoch: EpochId) -> GraphScrub {
        let _generation = GenerationReadScope::enter(self);
        GraphScrub {
            nodes: self.scrub_at_epoch(epoch),
            edges: self.edge_scrub_at_epoch(epoch),
        }
    }

    /// Whole-state columnar as-of scrub combining the cold base with the hot
    /// overlay — the complete [`scrub_at_epoch`](CompactStore::scrub_at_epoch)
    /// for a layered (incrementally-compacted) store.
    ///
    /// The base supplies the columnar bulk; then base nodes deleted by `epoch`
    /// are blanked, overlay-resident nodes (recently retained or modified)
    /// override their base columns with the overlay's as-of state, and overlay
    /// nodes created since the last merge are appended to their label's frame.
    /// Allocation stays bounded by the base scrub plus the (small) overlay.
    #[must_use]
    pub fn scrub_at_epoch(&self, epoch: EpochId) -> Vec<NodeTableScrub> {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        let mut frames = base.scrub_at_epoch(epoch);
        let overlay = self.overlay.load();

        // Presence lives in `node_ids`, so remove committed deletions rather
        // than merely blanking their properties (which resurrects label-only
        // nodes to scrub consumers).
        for (nid, del) in self.deleted_from_base_nodes.read().iter() {
            if del.epoch.as_u64() <= epoch.as_u64() {
                remove_scrub_node(&mut frames, *nid);
            }
        }

        // Replace overlay-resident identities by id. A temporal scrub frame may
        // omit closed rows (and may contain multiple labels), so physical base
        // table offsets are not valid frame indexes.
        let overlay_ids = overlay.all_node_ids();
        let dirty = self.dirty_node_ids.read();
        let published_overlay_ids: Vec<NodeId> = overlay_ids
            .into_iter()
            .filter(|id| Self::overlay_node_is_published(&base, &dirty, *id))
            .collect();
        drop(dirty);
        for nid in published_overlay_ids {
            remove_scrub_node(&mut frames, nid);
            if let Some(node) = self.get_node_at_epoch(nid, epoch) {
                let props: FxHashMap<PropertyKey, Value> = node
                    .properties
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                for label in &node.labels {
                    append_scrub_node(&mut frames, label.as_str(), nid, &props);
                }
            }
        }
        frames
    }

    /// Checks whether a node ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_node_dirty(&self, id: NodeId) -> bool {
        self.dirty_node_ids.read().contains(&id)
    }

    /// Checks whether a node was deleted from the base (latest view).
    ///
    /// Epoch-blind on purpose (mirror of [`Self::is_edge_deleted_from_base`]):
    /// the non-versioned accessors want the newest truth, so ANY tombstone —
    /// even an uncommitted one — hides the base node, preserving the audit-fixed
    /// base-tier-resurrection behavior. Versioned accessors use
    /// [`Self::is_node_deleted_from_base_at`] instead.
    #[inline]
    fn is_node_deleted_from_base(&self, id: NodeId) -> bool {
        self.deleted_from_base_nodes.read().contains_key(&id)
    }

    /// Snapshot-aware variant: whether a base node is hidden from a reader at
    /// `(epoch, tx)`. Node-side mirror of [`Self::is_edge_deleted_from_base_at`]
    /// — identical visibility boundary
    /// ([`VersionInfo::is_visible_at`](grafeo_common::mvcc::VersionInfo)):
    ///
    /// * not deleted → visible (`false`);
    /// * deleted by `tx` itself (even PENDING) → hidden (read-your-writes);
    /// * another tx's still-PENDING delete → visible (no dirty read);
    /// * committed delete → hidden iff the snapshot is at/after the delete's
    ///   commit epoch (`deleted_epoch <= viewing_epoch`).
    #[inline]
    fn is_node_deleted_from_base_at(&self, id: NodeId, epoch: EpochId, tx: TransactionId) -> bool {
        match self.deleted_from_base_nodes.read().get(&id) {
            None => false,
            Some(d) if d.deleter == Some(tx) => true,
            Some(d) if d.epoch == EpochId::PENDING => false,
            Some(d) => d.epoch.as_u64() <= epoch.as_u64(),
        }
    }

    /// Checks whether an edge ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_edge_dirty(&self, id: EdgeId) -> bool {
        self.dirty_edge_ids.read().contains(&id)
    }

    /// Overlay promote rematerializes at `current_epoch()`, so an as-of miss
    /// whose oldest overlay create is after `epoch` is "created later", not a
    /// delete. The pre-promote life still lives on the cold base.
    fn overlay_edge_created_after(overlay: &LpgStore, id: EdgeId, epoch: EpochId) -> bool {
        overlay
            .get_edge_history(id)
            .last()
            .is_some_and(|(created, _, _)| *created > epoch)
    }

    /// Promote restamps overlay create at `current_epoch()`; a miss whose
    /// oldest create is after `epoch` is not a delete — fall back to the
    /// base interval so as-of before promote still hits.
    fn overlay_node_created_after(overlay: &LpgStore, id: NodeId, epoch: EpochId) -> bool {
        overlay
            .get_node_history(id)
            .last()
            .is_some_and(|(created, _, _)| *created > epoch)
    }

    /// Overlay property versions start at promote. A miss with no overlay
    /// version at/before `epoch` is not a tombstone — fall back to the base.
    fn overlay_property_covers_epoch(
        overlay: &LpgStore,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> bool {
        overlay
            .node_property_history_for_key(id, key.as_str())
            .iter()
            .any(|(created, _)| *created <= epoch)
    }

    /// Checks whether an edge was deleted from the base (latest view).
    ///
    /// Epoch-blind on purpose: the non-versioned accessors want the newest
    /// truth, so ANY tombstone — even an uncommitted one — hides the base edge.
    /// This preserves the audit-fixed base-tier-resurrection behavior. Versioned
    /// accessors use [`Self::is_edge_deleted_from_base_at`] instead.
    #[inline]
    fn is_edge_deleted_from_base(&self, id: EdgeId) -> bool {
        self.deleted_from_base_edges.read().contains_key(&id)
    }

    /// Snapshot-aware variant: whether a base edge is hidden from a reader at
    /// `(epoch, tx)`, applying the same delete-visibility boundary the overlay
    /// version chain uses ([`VersionInfo::is_visible_at`](grafeo_common::mvcc::VersionInfo)).
    ///
    /// * not deleted → visible (`false`);
    /// * deleted by `tx` itself (even PENDING) → hidden (read-your-writes);
    /// * another tx's still-PENDING delete → visible (no dirty read);
    /// * committed delete → hidden iff the snapshot is at/after the delete's
    ///   commit epoch (`deleted_epoch <= viewing_epoch`).
    #[inline]
    fn is_edge_deleted_from_base_at(&self, id: EdgeId, epoch: EpochId, tx: TransactionId) -> bool {
        match self.deleted_from_base_edges.read().get(&id) {
            None => false,
            Some(d) if d.deleter == Some(tx) => true,
            Some(d) if d.epoch == EpochId::PENDING => false,
            Some(d) => d.epoch.as_u64() <= epoch.as_u64(),
        }
    }
}

impl LayeredStore {
    /// Shared full/sparse maintenance materializer. Caller retains one
    /// generation admission; routing guards drain before row/property reads.
    fn index_rows_for_ids(
        &self,
        publication_epoch: EpochId,
        transaction_id: Option<TransactionId>,
        ids: impl IntoIterator<Item = NodeId>,
    ) -> Vec<Node> {
        let base = self.base.load();
        let overlay = self.overlay.load();
        let mut routed = Vec::new();
        {
            let dirty = self.dirty_node_ids.read();
            let deleted = self.deleted_from_base_nodes.read();
            for id in ids {
                if deleted.get(&id).is_some_and(|deletion| {
                    transaction_id.is_some_and(|tx| deletion.deleter == Some(tx))
                        || (deletion.epoch != EpochId::PENDING
                            && deletion.epoch <= publication_epoch)
                }) {
                    continue;
                }
                routed.push((
                    id,
                    dirty.contains(&id) || !Self::base_has_node_identity(&base, id),
                ));
            }
        }
        let mut rows = Vec::new();
        let mut overlay_ids = Vec::new();
        for (id, hot) in routed {
            if hot {
                overlay_ids.push(id);
            } else if let Some(node) = base.get_node_at_epoch(id, publication_epoch) {
                rows.push(node);
            }
        }
        rows.extend(overlay.index_node_rows(publication_epoch, transaction_id, overlay_ids));
        rows.sort_unstable_by_key(|node| node.id);
        rows
    }
}

// ── GraphStore implementation ──────────────────────────────────────

impl GraphStore for LayeredStore {
    fn lpg_commit_target(
        &self,
    ) -> grafeo_common::utils::error::Result<crate::graph::traits::LpgCommitTarget<'_>> {
        Ok(crate::graph::traits::LpgCommitTarget::layered(self))
    }

    fn prepare_index_node_rows(
        &self,
        publication_epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> grafeo_common::utils::error::Result<Vec<Node>> {
        crate::graph::traits::validate_index_node_preparation(publication_epoch, transaction_id)?;
        let _generation = GenerationReadScope::enter(self);
        Ok(self.index_rows_for_ids(publication_epoch, transaction_id, self.known_node_ids()))
    }

    fn prepare_index_node_rows_by_id(
        &self,
        publication_epoch: EpochId,
        transaction_id: Option<TransactionId>,
        ids: &[NodeId],
    ) -> grafeo_common::utils::error::Result<Vec<Node>> {
        crate::graph::traits::validate_index_node_preparation(publication_epoch, transaction_id)?;
        let _generation = GenerationReadScope::enter(self);
        Ok(self.index_rows_for_ids(publication_epoch, transaction_id, ids.iter().copied()))
    }

    fn get_node(&self, id: NodeId) -> Option<Node> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        let dirty = self.is_node_dirty(id);
        let base = self.base.load();
        if dirty {
            return self.overlay.load().get_node(id);
        }
        if Self::base_has_node_identity(&base, id) {
            return base.get_node(id);
        }
        self.overlay.load().get_node(id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        let dirty = self.is_edge_dirty(id);
        let base = self.base.load();
        if dirty {
            return self.overlay.load().get_edge(id);
        }
        // Edges created after `compact()` live only in the overlay; fall
        // through when the base doesn't recognise the id.
        if Self::base_has_edge_identity(&base, id) {
            return base.get_edge(id);
        }
        self.overlay.load().get_edge(id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_node_deleted_from_base_at(id, epoch, transaction_id) {
            return None;
        }
        let dirty = self.is_node_dirty(id);
        let base = self.base.load();
        if dirty {
            return self
                .overlay
                .load()
                .get_node_versioned(id, epoch, transaction_id);
        }
        // `dirty_node_ids` only tracks overlay modifications of *base* nodes.
        // Overlay-only nodes (post-`compact()` writes) fall through to here;
        // the base doesn't know them, so defer to the overlay's versioned
        // fetch. CompactStore itself has no MVCC versions, so `get_node`
        // is the right base call.
        if Self::base_has_node_identity(&base, id) {
            let node = base.get_node_at_epoch(id, epoch);
            // Base-resident read: record it so Serializable read-sets include
            // base nodes (the overlay's own accessor already records overlay
            // reads; no double-record risk here since the base path is exclusive).
            if node.is_some() {
                self.overlay.load().record_read_node(transaction_id, id);
            }
            return node;
        }
        self.overlay
            .load()
            .get_node_versioned(id, epoch, transaction_id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_edge_deleted_from_base_at(id, epoch, transaction_id) {
            return None;
        }
        let dirty = self.is_edge_dirty(id);
        let base = self.base.load();
        if dirty {
            let overlay = self.overlay.load();
            if let Some(edge) = overlay.get_edge_versioned(id, epoch, transaction_id) {
                return Some(edge);
            }
            // Edge promotion moves the current open lifetime. Older
            // reincarnations remain authoritative in the compact sidecar.
            if Self::overlay_edge_created_after(&overlay, id, epoch) {
                let edge = base.get_edge_at_epoch(id, epoch);
                if edge.is_some() {
                    overlay.record_read_edge(transaction_id, id);
                }
                return edge;
            }
            return None;
        }
        if Self::base_has_edge_identity(&base, id) {
            let edge = base.get_edge_at_epoch(id, epoch);
            // Base-resident read: record for Serializable isolation.
            if edge.is_some() {
                self.overlay.load().record_read_edge(transaction_id, id);
            }
            return edge;
        }
        self.overlay
            .load()
            .get_edge_versioned(id, epoch, transaction_id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let _generation = GenerationReadScope::enter(self);
        // Epoch-only view: no transaction context, so pass INVALID (see
        // `get_edge_at_epoch`) — the read-your-writes branch is inert and an
        // uncommitted (PENDING) base delete stays visible.
        if self.is_node_deleted_from_base_at(id, epoch, TransactionId::INVALID) {
            return None;
        }
        let dirty = self.is_node_dirty(id);
        let base = self.base.load();
        if dirty {
            let overlay = self.overlay.load();
            if let Some(node) = overlay.get_node_at_epoch(id, epoch) {
                return Some(node);
            }
            if Self::overlay_node_created_after(&overlay, id, epoch) {
                return base.get_node_at_epoch(id, epoch);
            }
            return None;
        }
        // Route base-resident reads through the temporal base's as-of accessor
        // (SP3): on a compacted base this returns the node's state at `epoch`,
        // not its current state. For an all-open base it equals `get_node`.
        if Self::base_has_node_identity(&base, id) {
            return base.get_node_at_epoch(id, epoch);
        }
        self.overlay.load().get_node_at_epoch(id, epoch)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let _generation = GenerationReadScope::enter(self);
        // Epoch-only view: no transaction context, so pass INVALID — it can
        // never be a real deleter, so the read-your-writes branch is inert and
        // an uncommitted (PENDING) base delete stays visible (mirrors the
        // overlay's `is_edge_visible_at_epoch`, which only hides committed
        // deletes at/before `epoch`).
        if self.is_edge_deleted_from_base_at(id, epoch, TransactionId::INVALID) {
            return None;
        }
        let dirty = self.is_edge_dirty(id);
        let base = self.base.load();
        if dirty {
            let overlay = self.overlay.load();
            if let Some(edge) = overlay.get_edge_at_epoch(id, epoch) {
                return Some(edge);
            }
            // Promote restamps overlay create at current_epoch(); a miss whose
            // oldest create is after `epoch` is not a delete — fall back to the
            // base interval so as-of before promote (the retained-history window) still hits.
            if Self::overlay_edge_created_after(&overlay, id, epoch) {
                return base.get_edge_at_epoch(id, epoch);
            }
            return None;
        }
        if Self::base_has_edge_identity(&base, id) {
            return base.get_edge_at_epoch(id, epoch);
        }
        self.overlay.load().get_edge_at_epoch(id, epoch)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        let dirty = self.is_node_dirty(id);
        let base = self.base.load();
        if dirty {
            return self.overlay.load().get_node_property(id, key);
        }
        if Self::base_has_node_identity(&base, id) {
            return base.get_node_property(id, key);
        }
        self.overlay.load().get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        let dirty = self.is_edge_dirty(id);
        let base = self.base.load();
        if dirty {
            return self.overlay.load().get_edge_property(id, key);
        }
        if Self::base_has_edge_identity(&base, id) {
            return base.get_edge_property(id, key);
        }
        self.overlay.load().get_edge_property(id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        let _generation = GenerationReadScope::enter(self);
        ids.iter()
            .map(|id| self.get_node_property(*id, key))
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        let _generation = GenerationReadScope::enter(self);
        ids.iter()
            .map(|id| {
                self.get_node(*id)
                    .map(|n| {
                        n.properties
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let _generation = GenerationReadScope::enter(self);
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_node_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let _generation = GenerationReadScope::enter(self);
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_edge_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn fill_neighbors(&self, node: NodeId, direction: Direction, out: &mut Vec<NodeId>) {
        let _generation = GenerationReadScope::enter(self);
        if self.overlay_is_cold() {
            self.base.load().fill_neighbors(node, direction, out);
            return;
        }
        out.extend(self.neighbors(node, direction));
    }

    fn snapshot_neighbors(&self, direction: Direction) -> Vec<(NodeId, Vec<NodeId>)> {
        let _generation = GenerationReadScope::enter(self);
        if self.overlay_is_cold() {
            return self.base.load().snapshot_neighbors(direction);
        }
        let ids = self.node_ids();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let mut dests = Vec::new();
            self.fill_neighbors(id, direction, &mut dests);
            if !dests.is_empty() {
                out.push((id, dests));
            }
        }
        out
    }

    fn try_count_directed_triangles(
        &self,
        starts: &[NodeId],
        dest_label: Option<&str>,
    ) -> Option<u64> {
        let _generation = GenerationReadScope::enter(self);
        if !self.overlay_is_cold() {
            return None;
        }
        self.base
            .load()
            .try_count_directed_triangles(starts, dest_label)
    }

    fn try_count_all_directed_triangles(&self, dest_label: Option<&str>) -> Option<u64> {
        let _generation = GenerationReadScope::enter(self);
        if !self.overlay_is_cold() {
            return None;
        }
        self.base
            .load()
            .try_count_all_directed_triangles(dest_label)
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        // Current 1-hop: derived current CSR on the base (Task 4) plus overlay,
        // via edges_from. Never filter(is_open) on a fat historical run.
        // Single source of truth for tier-merged adjacency: derive neighbors
        // from edges_from so the node AND edge tombstone filters (and overlay
        // promotion) are applied in exactly one place. A hand-rolled merge
        // here previously filtered deleted_from_base_nodes but never
        // deleted_from_base_edges, so a target reachable only via a deleted
        // base edge was still reported.
        let mut targets: Vec<NodeId> = self
            .edges_from(node, direction)
            .into_iter()
            .map(|(target, _eid)| target)
            .collect();
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let _generation = GenerationReadScope::enter(self);
        let base_edges = self.base.load().edges_from(node, direction);
        let overlay_edges: Vec<(NodeId, EdgeId)> =
            self.overlay.load().edges_from(node, direction).collect();
        let deleted_nodes = self.deleted_from_base_nodes.read();
        let deleted_edges = self.deleted_from_base_edges.read();

        let mut results = Vec::new();

        // Base edges (minus deleted). After a temporal merge the base
        // `RelTable.fwd` is the derived current CSR (open prefix only).
        // The base layer must be consulted even when the source node is
        // dirty: `ensure_in_overlay` copies labels and properties into the
        // overlay but leaves adjacency in the base, so a property write —
        // or merely being the endpoint of a freshly created overlay edge —
        // must not erase the node's pre-existing snapshot edges. Promoted
        // edges (those in both tiers because their properties were
        // modified) live at the same `EdgeId` in base and overlay and are
        // folded together by the dedup-by-eid pass below.
        if !deleted_nodes.contains_key(&node) {
            for (target, eid) in base_edges {
                if !deleted_nodes.contains_key(&target) && !deleted_edges.contains_key(&eid) {
                    results.push((target, eid));
                }
            }
        }

        // Overlay edges — always consulted. The overlay stores edges
        // keyed by src/dst even when the endpoint is a base node (e.g. a
        // post-`compact()` edge from a base node to an overlay node), so
        // we can't gate this on whether the overlay has the node itself.
        // `LpgStore::edges_from` returns empty for ids with no outgoing
        // edges, so the unconditional call is cheap when there's nothing
        // to report.
        for (target, eid) in overlay_edges {
            if !deleted_nodes.contains_key(&target) && !deleted_edges.contains_key(&eid) {
                results.push((target, eid));
            }
        }

        // Deduplicate in case a promoted edge appears in both layers.
        results.sort_unstable_by_key(|&(_, eid)| eid);
        results.dedup_by_key(|&mut (_, eid)| eid);

        results
    }

    fn edges_from_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
    ) -> Vec<(NodeId, EdgeId)> {
        let _generation = GenerationReadScope::enter(self);
        let mut out = Vec::new();
        self.fill_edges_from_at_epoch(node, direction, epoch, &mut out);
        out
    }

    fn fill_edges_from(&self, node: NodeId, direction: Direction, out: &mut Vec<(NodeId, EdgeId)>) {
        let _generation = GenerationReadScope::enter(self);
        out.extend(self.edges_from(node, direction));
    }

    fn fill_neighbors_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<NodeId>,
    ) {
        let _generation = GenerationReadScope::enter(self);
        LayeredStore::fill_neighbors_at_epoch(self, node, direction, epoch, out);
    }

    fn fill_neighbors_of_types_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        types: &[String],
        out: &mut Vec<NodeId>,
    ) {
        let _generation = GenerationReadScope::enter(self);
        LayeredStore::fill_neighbors_of_types_at_epoch(self, node, direction, epoch, types, out);
    }

    fn fill_edges_from_at_epoch(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        out: &mut Vec<(NodeId, EdgeId)>,
    ) {
        let _generation = GenerationReadScope::enter(self);
        if epoch == EpochId::PENDING {
            out.extend(self.edges_from(node, direction));
            return;
        }
        let tx = TransactionId::INVALID;
        let mut seen: FxHashSet<EdgeId> = FxHashSet::default();
        let mut results = Vec::new();

        self.base
            .load()
            .extend_edges_from_at_epoch(node, direction, epoch, &mut results);
        results.retain(|(target, eid)| {
            if seen.contains(eid) {
                return false;
            }
            if self.is_node_deleted_from_base_at(*target, epoch, tx) {
                return false;
            }
            self.get_edge_at_epoch(*eid, epoch).is_some() && seen.insert(*eid)
        });

        let overlay = self.overlay.load();
        for (target, eid) in overlay.edges_from_versioned(node, direction, epoch, tx) {
            if seen.contains(&eid) {
                continue;
            }
            if self.is_node_deleted_from_base_at(target, epoch, tx) {
                continue;
            }
            if self.get_edge_at_epoch(eid, epoch).is_some() && seen.insert(eid) {
                results.push((target, eid));
            }
        }

        results.sort_unstable_by_key(|&(_, eid)| eid);
        results.dedup_by_key(|&mut (_, eid)| eid);
        out.extend(results);
    }

    fn edges_from_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<(NodeId, EdgeId)> {
        let _generation = GenerationReadScope::enter(self);
        let mut results = Vec::new();
        let base = self.base.load();
        let overlay = self.overlay.load();
        let base_edges = base.edges_from_at_epoch(node, direction, epoch);
        let overlay_edges = overlay.edges_from_including_deleted(node, direction);
        let physical_overlay_ids: FxHashSet<EdgeId> =
            overlay_edges.iter().map(|(_, id)| *id).collect();
        // Never retain the routing guard across LPG history/visibility reads:
        // promotion holds the corresponding rollback gates until it can take
        // `dirty_edge_ids.write()` for publication. Restrict the owned routing
        // cut to this adjacency, avoiding a whole-overlay clone on the historical-read hot
        // path. A promotion published after the physical snapshot is routed to
        // the still-authoritative base for this call.
        let dirty: FxHashSet<EdgeId> = {
            let routing = self.dirty_edge_ids.read();
            physical_overlay_ids
                .iter()
                .copied()
                .filter(|id| routing.contains(id))
                .collect()
        };

        // Base edges: iterate without filtering by `deleted_from_base_edges` so
        // that an edge deleted (from the base) after this snapshot's start is
        // still examined by `is_edge_visible_versioned`.  The visibility check
        // consults the overlay's version chain (which includes delete tombstones)
        // and also records the read for SSI. Node tombstones are equally
        // snapshot-aware: an old reader (or publication-time detach validator
        // using INVALID to inspect committed adjacency) must not lose the
        // pre-delete base row merely because a newer/PENDING tombstone exists.
        if !self.is_node_deleted_from_base_at(node, epoch, transaction_id) {
            for (target, eid) in base_edges {
                // A published promotion owns its current lifetime. An older
                // reincarnation predating that promoted creation remains in
                // the compact sidecar and must still feed an old snapshot.
                if dirty.contains(&eid) && !Self::overlay_edge_created_after(&overlay, eid, epoch) {
                    continue;
                }
                if !self.is_node_deleted_from_base_at(target, epoch, transaction_id)
                    && !self.is_edge_deleted_from_base_at(eid, epoch, transaction_id)
                    && base.is_edge_visible_versioned(eid, epoch, transaction_id)
                {
                    overlay.record_read_edge(transaction_id, eid);
                    results.push((target, eid));
                }
            }
        }

        // Filter raw adjacency before the visibility call: visibility records
        // an SSI read, so result-level deduplication would be too late for an
        // unpublished base-overlap row.
        for (target, eid) in overlay_edges {
            if Self::overlay_edge_is_published(&base, &dirty, eid)
                && !self.is_node_deleted_from_base_at(target, epoch, transaction_id)
                && overlay.is_edge_visible_versioned(eid, epoch, transaction_id)
            {
                results.push((target, eid));
            }
        }

        // Deduplicate promoted edges that appear in both tiers.
        results.sort_unstable_by_key(|&(_, eid)| eid);
        results.dedup_by_key(|&mut (_, eid)| eid);

        results
    }

    fn neighbors_versioned(
        &self,
        node: NodeId,
        direction: Direction,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let mut targets: Vec<NodeId> = self
            .edges_from_versioned(node, direction, epoch, transaction_id)
            .into_iter()
            .map(|(target, _)| target)
            .collect();
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    fn all_edges_have_types(&self, types: &[String]) -> bool {
        let _generation = GenerationReadScope::enter(self);
        self.base.load().all_edges_have_types(types)
            && self.overlay.load().all_edges_have_types(types)
    }

    fn count_edges_from(&self, node: NodeId, direction: Direction, types: &[String]) -> usize {
        let _generation = GenerationReadScope::enter(self);
        let no_tombstones = self.deleted_from_base_nodes.read().is_empty()
            && self.deleted_from_base_edges.read().is_empty();
        if no_tombstones {
            let base = self.base.load();
            let overlay = self.overlay.load();
            if base.edge_count() == 0 {
                return overlay.count_edges_from(node, direction, types);
            }
            if overlay.edge_count() == 0 {
                return base.count_edges_from(node, direction, types);
            }
        }
        self.edges_from(node, direction)
            .into_iter()
            .filter(|(_, eid)| {
                types.is_empty()
                    || self.edge_type(*eid).is_some_and(|actual| {
                        types
                            .iter()
                            .any(|t| actual.as_str().eq_ignore_ascii_case(t.as_str()))
                    })
            })
            .count()
    }

    fn out_degree(&self, node: NodeId) -> usize {
        let _generation = GenerationReadScope::enter(self);
        self.edges_from(node, Direction::Outgoing).len()
    }

    fn in_degree(&self, node: NodeId) -> usize {
        let _generation = GenerationReadScope::enter(self);
        self.edges_from(node, Direction::Incoming).len()
    }

    fn has_backward_adjacency(&self) -> bool {
        let _generation = GenerationReadScope::enter(self);
        self.base.load().has_backward_adjacency() || self.overlay.load().has_backward_adjacency()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let overlay_ids = self.overlay.load().node_ids();
        let deleted = self.deleted_from_base_nodes.read();

        let mut ids: Vec<NodeId> = self
            .base
            .load()
            .node_ids()
            .into_iter()
            .filter(|id| !deleted.contains_key(id))
            .collect();
        ids.extend(overlay_ids);
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        // Versioned readers need candidates from every retained lifetime, not
        // merely the current post-image. In particular, a transaction whose
        // snapshot predates a base-tier delete must still be able to ask
        // `get_node_versioned` about that tombstoned identity. The final
        // visibility decision remains in the versioned accessor.
        self.known_node_ids()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let deleted = self.deleted_from_base_nodes.read().clone();
        let dirty = self.dirty_node_ids.read().clone();
        let overlay_ids = self.overlay.load().nodes_by_label(label);

        let mut ids: Vec<NodeId> = self
            .base
            .load()
            .nodes_by_label(label)
            .into_iter()
            .filter(|id| !deleted.contains_key(id) && !dirty.contains(id))
            .collect();
        ids.extend(
            overlay_ids
                .into_iter()
                .filter(|id| !deleted.contains_key(id)),
        );
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Membership in the same union `nodes_by_label` builds: deleted nodes carry
    /// no label, the overlay answers for anything it holds, and the base answers
    /// only for ids the overlay has not taken over.
    fn node_has_label(&self, id: NodeId, label: &str) -> bool {
        let _generation = GenerationReadScope::enter(self);
        if self.deleted_from_base_nodes.read().contains_key(&id) {
            return false;
        }
        if self.overlay.load().node_has_label(id, label) {
            return true;
        }
        !self.dirty_node_ids.read().contains(&id) && self.base.load().node_has_label(id, label)
    }

    fn node_has_label_visible(
        &self,
        id: NodeId,
        label: &str,
        transaction_id: Option<TransactionId>,
    ) -> bool {
        let _generation = GenerationReadScope::enter(self);
        if self.deleted_from_base_nodes.read().contains_key(&id) {
            return false;
        }
        if self.dirty_node_ids.read().contains(&id) {
            return self
                .overlay
                .load()
                .node_has_label_visible(id, label, transaction_id);
        }
        self.base
            .load()
            .node_has_label_visible(id, label, transaction_id)
    }

    fn node_has_label_at_epoch(
        &self,
        id: NodeId,
        label: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let generation = GenerationReadScope::enter(self);
        if self.is_node_deleted_from_base_at(id, epoch, transaction_id) {
            return false;
        }
        let overlay = &generation._overlay;
        if self.is_node_dirty(id) {
            return overlay.node_has_label_at_epoch(id, label, epoch, transaction_id);
        }
        let base = self.base.load();
        if Self::base_has_node_identity(&base, id) {
            return base.node_has_label_at_epoch(id, label, epoch, transaction_id);
        }
        overlay.node_has_label_at_epoch(id, label, epoch, transaction_id)
    }

    fn node_count(&self) -> usize {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        let base_count = base.node_count();
        let overlay_ids = self.overlay.load().node_ids();
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();
        let overlay_count = overlay_ids
            .iter()
            .filter(|id| {
                !deleted.contains_key(*id) && Self::overlay_node_is_published(&base, &dirty, **id)
            })
            .count();
        // Dirty nodes that came from the base are counted once in the overlay,
        // so subtract them from the base total to avoid double counting — EXCEPT
        // a promoted node that has since been deleted: it is already accounted
        // for by `deleted`, so subtracting it as `promoted` too would
        // double-subtract it (it is no longer in the overlay either).
        let promoted = overlay_ids
            .iter()
            .filter(|id| {
                dirty.contains(*id) && base.get_node(**id).is_some() && !deleted.contains_key(*id)
            })
            .count();
        // Tombstone logs can legitimately outlive one physical cold
        // generation (or contain a historical/non-current identity). Only a
        // row present in this pinned base contributes to `base_count`, so only
        // such a row may be subtracted from it.
        let deleted_from_this_base = deleted
            .keys()
            .filter(|id| base.get_node(**id).is_some())
            .count();
        base_count
            .checked_sub(deleted_from_this_base + promoted)
            .expect("Layered node routing cannot subtract more current base rows than exist")
            + overlay_count
    }

    fn edge_count(&self) -> usize {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        let base_count = base.edge_count() + base.retained_open_edge_count();
        let overlay_edges: Vec<EdgeId> = self
            .overlay
            .load()
            .all_edges()
            .map(|edge| edge.id)
            .collect();
        let deleted = self.deleted_from_base_edges.read();
        let dirty = self.dirty_edge_ids.read();
        let overlay_count = overlay_edges
            .iter()
            .filter(|id| {
                !deleted.contains_key(*id) && Self::overlay_edge_is_published(&base, &dirty, **id)
            })
            .count();
        let promoted = overlay_edges
            .iter()
            .filter(|id| {
                dirty.contains(*id)
                    && base.has_logically_open_edge(**id)
                    && !deleted.contains_key(*id)
            })
            .count();
        let deleted_from_this_base = deleted
            .keys()
            .filter(|id| base.has_logically_open_edge(**id))
            .count();
        base_count
            .checked_sub(deleted_from_this_base + promoted)
            .expect("Layered edge routing cannot subtract more current base rows than exist")
            + overlay_count
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let _generation = GenerationReadScope::enter(self);
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        let dirty = self.is_edge_dirty(id);
        let base = self.base.load();
        if dirty {
            return self.overlay.load().edge_type(id);
        }
        if Self::base_has_edge_identity(&base, id) {
            return base.edge_type(id);
        }
        self.overlay.load().edge_type(id)
    }

    fn has_property_index(&self, property: &str) -> bool {
        let generation = GenerationReadScope::enter(self);
        // Only a registered logical index may opt query execution into the
        // historical indexed contract. Cold columns remain available through
        // the ordinary scan fallback until their registration is transferred
        // to the pinned overlay representation.
        generation._overlay.has_property_index(property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        let overlay = self.overlay.load();
        let deleted = self.deleted_from_base_nodes.read().clone();
        let dirty = self.dirty_node_ids.read().clone();
        let overlay_results = overlay.find_nodes_by_property(property, value);

        let mut results: Vec<NodeId> = base
            .find_nodes_by_property(property, value)
            .into_iter()
            .filter(|id| !deleted.contains_key(id) && !dirty.contains(id))
            .collect();

        results.extend(overlay_results.into_iter().filter(|id| {
            !deleted.contains_key(id) && Self::overlay_node_is_published(&base, &dirty, *id)
        }));
        results.sort_unstable();
        results.dedup();
        results
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        if conditions.is_empty() {
            return self.node_ids();
        }
        let base = self.base.load();
        let overlay = self.overlay.load();
        let deleted = self.deleted_from_base_nodes.read().clone();
        let dirty = self.dirty_node_ids.read().clone();
        let overlay_results = overlay.find_nodes_by_properties(conditions);

        let mut results: Vec<NodeId> = base
            .find_nodes_by_properties(conditions)
            .into_iter()
            .filter(|id| !deleted.contains_key(id) && !dirty.contains(id))
            .collect();

        results.extend(overlay_results.into_iter().filter(|id| {
            !deleted.contains_key(id) && Self::overlay_node_is_published(&base, &dirty, *id)
        }));
        results.sort_unstable();
        results.dedup();
        results
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        let overlay = self.overlay.load();
        let deleted = self.deleted_from_base_nodes.read().clone();
        let dirty = self.dirty_node_ids.read().clone();
        let overlay_results =
            overlay.find_nodes_in_range(property, min, max, min_inclusive, max_inclusive);

        let mut results: Vec<NodeId> = base
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
            .into_iter()
            .filter(|id| !deleted.contains_key(id) && !dirty.contains(id))
            .collect();

        results.extend(overlay_results.into_iter().filter(|id| {
            !deleted.contains_key(id) && Self::overlay_node_is_published(&base, &dirty, *id)
        }));
        results.sort_unstable();
        results.dedup();
        results
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        if base.node_property_might_match(property, op, value) {
            return true;
        }
        let overlay = self.overlay.load();
        let overlay_values: Vec<_> = overlay
            .node_ids()
            .into_iter()
            .map(|id| (id, overlay.get_node_property(id, property)))
            .collect();
        let dirty = self.dirty_node_ids.read();
        let deleted = self.deleted_from_base_nodes.read();
        overlay_values.into_iter().any(|(id, actual)| {
            !deleted.contains_key(&id)
                && Self::overlay_node_is_published(&base, &dirty, id)
                && actual.is_some_and(|actual| Self::value_matches_compare(&actual, op, value))
        })
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        if base.edge_property_might_match(property, op, value) {
            return true;
        }
        let overlay = self.overlay.load();
        let overlay_values: Vec<_> = overlay
            .all_edges()
            .map(|edge| (edge.id, overlay.get_edge_property(edge.id, property)))
            .collect();
        let dirty = self.dirty_edge_ids.read();
        let deleted = self.deleted_from_base_edges.read();
        overlay_values.into_iter().any(|(id, actual)| {
            !deleted.contains_key(&id)
                && Self::overlay_edge_is_published(&base, &dirty, id)
                && actual.is_some_and(|actual| Self::value_matches_compare(&actual, op, value))
        })
    }

    fn statistics(&self) -> Arc<Statistics> {
        let _generation = GenerationReadScope::enter(self);
        // Combine base + overlay statistics. Snapshot the overlay once
        // so the labels we enumerate and the per-label counts we read
        // observe the same `LpgStore` revision — otherwise a concurrent
        // `merge_overlay_in_place` (which swaps the overlay) could let
        // us see a label and then read its count from the post-swap
        // empty overlay.
        let base_stats = self.base.load().statistics();

        let mut combined = (*base_stats).clone();
        combined.total_nodes = self.node_count() as u64;
        combined.total_edges = self.edge_count() as u64;

        // Recompute logical label membership rather than adding raw physical
        // tier counts: promoted rows exist in both tiers, and an unpublished
        // hydration row must contribute to neither overlay authority nor a
        // duplicate count.
        for label in self.all_labels() {
            let count = self.nodes_by_label(&label).len() as u64;
            combined.update_label(&label, crate::statistics::LabelStatistics::new(count));
        }

        Arc::new(combined)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        let _generation = GenerationReadScope::enter(self);
        self.nodes_by_label(label).len() as f64
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        let _generation = GenerationReadScope::enter(self);
        let edges: Vec<Edge> = self
            .current_edges()
            .into_iter()
            .filter(|edge| edge.edge_type.as_str() == edge_type)
            .collect();
        if edges.is_empty() {
            return 0.0;
        }

        // Match the compact store's schema-level denominator: every node in a
        // label/table that participates on the selected side of this relation
        // type, not merely the endpoints that happen to have an edge.
        let mut endpoint_labels = FxHashSet::default();
        let mut endpoint_ids = FxHashSet::default();
        for edge in &edges {
            let endpoint = if outgoing { edge.src } else { edge.dst };
            endpoint_ids.insert(endpoint);
            if let Some(node) = self.get_node(endpoint) {
                endpoint_labels.extend(node.labels.iter().map(|label| label.to_string()));
            }
        }
        for label in endpoint_labels {
            endpoint_ids.extend(self.nodes_by_label(&label));
        }
        edges.len() as f64 / endpoint_ids.len().max(1) as f64
    }

    fn current_epoch(&self) -> EpochId {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().current_epoch()
    }

    fn all_labels(&self) -> Vec<String> {
        let _generation = GenerationReadScope::enter(self);
        let mut labels: FxHashSet<String> = self.base.load().all_labels().into_iter().collect();
        labels.extend(self.overlay.load().all_labels());
        labels.into_iter().collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        let _generation = GenerationReadScope::enter(self);
        let mut types: FxHashSet<String> = self.base.load().all_edge_types().into_iter().collect();
        types.extend(self.overlay.load().all_edge_types());
        types.into_iter().collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        let _generation = GenerationReadScope::enter(self);
        let base = self.base.load();
        let mut keys: FxHashSet<String> = base.all_property_keys().into_iter().collect();
        let overlay = self.overlay.load();
        let node_keys: Vec<_> = overlay
            .all_node_ids()
            .into_iter()
            .map(|id| {
                let keys = overlay
                    .node_property_history(id)
                    .into_iter()
                    .map(|(key, _)| key.as_str().to_owned())
                    .collect::<Vec<_>>();
                (id, keys)
            })
            .collect();
        let edge_keys: Vec<_> = overlay
            .all_known_edge_ids()
            .into_iter()
            .map(|id| {
                let keys = overlay
                    .edge_property_history(id)
                    .into_iter()
                    .map(|(key, _)| key.as_str().to_owned())
                    .collect::<Vec<_>>();
                (id, keys)
            })
            .collect();
        let dirty_nodes = self.dirty_node_ids.read();
        for (id, node_keys) in node_keys {
            if Self::overlay_node_is_published(&base, &dirty_nodes, id) {
                keys.extend(node_keys);
            }
        }
        drop(dirty_nodes);
        let dirty_edges = self.dirty_edge_ids.read();
        for (id, edge_keys) in edge_keys {
            if Self::overlay_edge_is_published(&base, &dirty_edges, id) {
                keys.extend(edge_keys);
            }
        }
        keys.into_iter().collect()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let _generation = GenerationReadScope::enter(self);
        // Epoch-only view: INVALID tx (see `get_node_at_epoch`).
        if self.is_node_deleted_from_base_at(id, epoch, TransactionId::INVALID) {
            return false;
        }
        if self.is_node_dirty(id) {
            let overlay = self.overlay.load();
            if overlay.is_node_visible_at_epoch(id, epoch) {
                return true;
            }
            if Self::overlay_node_created_after(&overlay, id, epoch) {
                return self.base.load().get_node_at_epoch(id, epoch).is_some();
            }
            return false;
        }
        // `dirty_node_ids` only tracks overlay *modifications of base nodes*
        // — overlay-only nodes (e.g. post-`compact()` writes) fall through
        // here and must be dispatched to the overlay's MVCC check. The base
        // doesn't know the id, so it would otherwise report them invisible.
        //
        // Snapshot the base once: a concurrent `swap_base` between the
        // presence check and the visibility call would otherwise dispatch
        // through a different `CompactStore` than the one we tested.
        let base = self.base.load();
        if Self::base_has_node_identity(&base, id) {
            base.is_node_visible_at_epoch(id, epoch)
        } else {
            self.overlay.load().is_node_visible_at_epoch(id, epoch)
        }
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let _generation = GenerationReadScope::enter(self);
        if self.is_node_deleted_from_base_at(id, epoch, transaction_id) {
            return false;
        }
        if self.is_node_dirty(id) {
            return self
                .overlay
                .load()
                .is_node_visible_versioned(id, epoch, transaction_id);
        }
        let base = self.base.load();
        if Self::base_has_node_identity(&base, id) {
            let visible = base.is_node_visible_versioned(id, epoch, transaction_id);
            if visible {
                // Base-resident visibility confirmed: record for Serializable read-set.
                self.overlay.load().record_read_node(transaction_id, id);
            }
            return visible;
        }
        self.overlay
            .load()
            .is_node_visible_versioned(id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let _generation = GenerationReadScope::enter(self);
        // Epoch-only view: INVALID tx (see `get_edge_at_epoch`).
        if self.is_edge_deleted_from_base_at(id, epoch, TransactionId::INVALID) {
            return false;
        }
        if self.is_edge_dirty(id) {
            let overlay = self.overlay.load();
            if overlay.is_edge_visible_at_epoch(id, epoch) {
                return true;
            }
            if Self::overlay_edge_created_after(&overlay, id, epoch) {
                return self.base.load().get_edge_at_epoch(id, epoch).is_some();
            }
            return false;
        }
        let base = self.base.load();
        if Self::base_has_edge_identity(&base, id) {
            base.is_edge_visible_at_epoch(id, epoch)
        } else {
            self.overlay.load().is_edge_visible_at_epoch(id, epoch)
        }
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let _generation = GenerationReadScope::enter(self);
        if self.is_edge_deleted_from_base_at(id, epoch, transaction_id) {
            return false;
        }
        if self.is_edge_dirty(id) {
            let overlay = self.overlay.load();
            if overlay.is_edge_visible_versioned(id, epoch, transaction_id) {
                return true;
            }
            if Self::overlay_edge_created_after(&overlay, id, epoch) {
                let visible = self.base.load().get_edge_at_epoch(id, epoch).is_some();
                if visible {
                    overlay.record_read_edge(transaction_id, id);
                }
                return visible;
            }
            return false;
        }
        let base = self.base.load();
        if Self::base_has_edge_identity(&base, id) {
            let visible = base.is_edge_visible_versioned(id, epoch, transaction_id);
            if visible {
                // Base-resident visibility confirmed: record for Serializable read-set.
                self.overlay.load().record_read_edge(transaction_id, id);
            }
            return visible;
        }
        self.overlay
            .load()
            .is_edge_visible_versioned(id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_at_epoch(*id, epoch))
            .collect()
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_versioned(*id, epoch, transaction_id))
            .collect()
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        // `complete_node_history` pins the whole mutable generation. Taking
        // the narrower publication cut first would invert the global
        // mutation-then-publication lock order.
        self.complete_node_history(id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        // `complete_edge_history` pins the whole mutable generation. Taking
        // the narrower publication cut first would invert the global
        // mutation-then-publication lock order.
        self.complete_edge_history(id)
    }

    // --- Task 6: snapshot-aware read delegation (unified-MVCC) ---
    //
    // The per-transaction property delta lives in the overlay LpgStore. For
    // nodes/edges that are dirty (promoted into the overlay), we delegate
    // entirely to the overlay's snapshot-aware accessor. For base-only
    // entities, the overlay has no entry, so we fall back to the base's
    // committed properties (no delta possible for base-only entities — any
    // transactional write will have promoted the entity to the overlay first
    // via ensure_in_overlay / ensure_edge_in_overlay before calling
    // *_buffered). No event/log side effects exist here.

    fn pending_node_creates(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().pending_node_creates(transaction_id)
    }

    fn pending_edge_creates(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().pending_edge_creates(transaction_id)
    }

    fn register_read_tracker(&self, tx: TransactionId, tracker: SharedReadTracker) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().register_read_tracker(tx, tracker);
    }

    fn unregister_read_tracker(&self, tx: TransactionId) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().unregister_read_tracker(tx);
    }

    fn record_label_predicate_read(&self, tx: TransactionId, label: &str) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().record_label_predicate_read(tx, label);
    }

    fn record_rel_type_predicate_read(&self, tx: TransactionId, rel_type: &str) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay
            .load()
            .record_rel_type_predicate_read(tx, rel_type);
    }

    fn record_lpg_dataset_read(&self, tx: TransactionId) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().record_lpg_dataset_read(tx);
    }

    fn register_write_tracker(&self, tx: TransactionId, tracker: SharedWriteTracker) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().register_write_tracker(tx, tracker);
    }

    fn unregister_write_tracker(&self, tx: TransactionId) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().unregister_write_tracker(tx);
    }

    fn pending_node_deletes_peek(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let mut pending = self
            .overlay
            .load()
            .pending_node_deletes_peek(transaction_id);
        if let Some(base_pending) = self.pending_base_node_deletes.read().get(&transaction_id) {
            for &id in base_pending {
                if !pending.contains(&id) {
                    pending.push(id);
                }
            }
        }
        pending
    }

    fn pending_edge_deletes_peek(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        let _generation = GenerationReadScope::enter(self);
        let mut pending = self
            .overlay
            .load()
            .pending_edge_deletes_peek(transaction_id);
        if let Some(base_pending) = self.pending_base_edge_deletes.read().get(&transaction_id) {
            for &id in base_pending {
                if !pending.contains(&id) {
                    pending.push(id);
                }
            }
        }
        pending
    }

    fn overlay_touched_entities(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<NodeId>, Vec<EdgeId>) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().overlay_touched_entities(transaction_id)
    }

    fn overlay_touched_properties(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<(NodeId, Option<String>)>, Vec<(EdgeId, Option<String>)>) {
        let _generation = GenerationReadScope::enter(self);
        self.overlay
            .load()
            .overlay_touched_properties(transaction_id)
    }

    fn read_node_property_visible(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        let _generation = GenerationReadScope::enter(self);
        // Snapshot-aware gate (mirrors `read_node_properties_visible` and the
        // edge accessors): an old snapshot that can still see a base node
        // deleted-after-its-start must read its property too. Epoch-only callers
        // (`None`) pass INVALID so the read-your-writes branch is inert and a
        // PENDING delete stays visible.
        if self.is_node_deleted_from_base_at(
            id,
            epoch,
            transaction_id.unwrap_or(TransactionId::INVALID),
        ) {
            return None;
        }
        let dirty = self.is_node_dirty(id);
        let base = self.base.load();
        if dirty {
            // Node is in the overlay; the delta (if any) is there too.
            return self
                .overlay
                .load()
                .read_node_property_visible(id, key, epoch, transaction_id);
        }
        // Base-only node: the overlay has no entry and no delta. Fall through
        // to the base's committed value (same as get_node_property for base).
        let overlay = self.overlay.load();
        let result = if Self::base_has_node_identity(&base, id) {
            base.get_node_property_at_epoch(id, key, epoch)
        } else {
            overlay.get_node_property(id, key)
        };
        if result.is_some()
            && let Some(tx) = transaction_id
        {
            // Record base-resident property read into the Serializable read-set.
            overlay.record_read_node_property(tx, id, key.as_str());
        }
        result
    }

    fn read_edge_property_visible(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Option<Value> {
        let _generation = GenerationReadScope::enter(self);
        // Snapshot-aware gate (mirrors `read_edge_properties_visible`): an old
        // snapshot that can still traverse a base edge deleted-after-its-start
        // must read its property too. Epoch-only callers (`None`) pass INVALID so
        // the read-your-writes branch is inert and a PENDING delete stays visible.
        if self.is_edge_deleted_from_base_at(
            id,
            epoch,
            transaction_id.unwrap_or(TransactionId::INVALID),
        ) {
            return None;
        }
        let dirty = self.is_edge_dirty(id);
        let base = self.base.load();
        if dirty {
            return self
                .overlay
                .load()
                .read_edge_property_visible(id, key, epoch, transaction_id);
        }
        let overlay = self.overlay.load();
        let result = if Self::base_has_edge_identity(&base, id) {
            base.get_edge_property_at_epoch(id, key, epoch)
        } else {
            overlay.get_edge_property(id, key)
        };
        if result.is_some()
            && let Some(tx) = transaction_id
        {
            // Record base-resident property read into the Serializable read-set.
            overlay.record_read_edge_property(tx, id, key.as_str());
        }
        result
    }

    fn read_node_properties_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashMap<PropertyKey, Value> {
        let _generation = GenerationReadScope::enter(self);
        let tx = transaction_id.unwrap_or(TransactionId::INVALID);
        if self.is_node_deleted_from_base_at(id, epoch, tx) {
            return FxHashMap::default();
        }
        if self.is_node_dirty(id) {
            return self
                .overlay
                .load()
                .read_node_properties_visible(id, epoch, transaction_id);
        }
        // Base-only: return the committed property map. Use the versioned fetch
        // (not the epoch-blind `get_node`) so the property read agrees with the
        // snapshot-aware gate above — otherwise an old snapshot that may still
        // see a base node deleted-after-its-start would get an empty map.
        let result: FxHashMap<PropertyKey, Value> = self
            .get_node_versioned(id, epoch, tx)
            .map(|n| {
                n.properties
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default();
        if !result.is_empty()
            && let Some(tx) = transaction_id
        {
            // Record base-resident read into the Serializable read-set.
            self.overlay.load().record_read_node(tx, id);
        }
        result
    }

    fn read_edge_properties_visible(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashMap<PropertyKey, Value> {
        let _generation = GenerationReadScope::enter(self);
        let tx = transaction_id.unwrap_or(TransactionId::INVALID);
        if self.is_edge_deleted_from_base_at(id, epoch, tx) {
            return FxHashMap::default();
        }
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .load()
                .read_edge_properties_visible(id, epoch, transaction_id);
        }
        // Base-only: return the committed property map. Use the versioned fetch
        // (not the epoch-blind `get_edge`) so the property read agrees with the
        // snapshot-aware gate above — otherwise an old snapshot that may still
        // see a base edge deleted-after-its-start would get an empty map.
        let result = self
            .get_edge_versioned(id, epoch, tx)
            .map(|e| {
                e.properties
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<FxHashMap<_, _>>()
            })
            .unwrap_or_default();
        if !result.is_empty()
            && let Some(tx) = transaction_id
        {
            // Record base-resident read into the Serializable read-set.
            self.overlay.load().record_read_edge(tx, id);
        }
        result
    }

    // --- Task 5 (label delegation, unified-MVCC) ---
    //
    // The per-transaction label delta lives in the overlay LpgStore. For nodes
    // that are dirty (promoted into the overlay), we delegate entirely to the
    // overlay's snapshot-aware accessor. For base-only nodes, the overlay has
    // no label entry and no delta — any transactional label write will have
    // promoted the node into the overlay via `ensure_in_overlay` before calling
    // `*_label_buffered`, so base-only nodes can fall through to the committed
    // label set from the base. No event/log side effects exist here.

    fn read_node_labels_visible(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> FxHashSet<arcstr::ArcStr> {
        let _generation = GenerationReadScope::enter(self);
        let tx = transaction_id.unwrap_or(TransactionId::INVALID);
        if self.is_node_deleted_from_base_at(id, epoch, tx) {
            return FxHashSet::default();
        }
        if self.is_node_dirty(id) {
            // Node is in the overlay; the label delta (if any) is there too.
            return self
                .overlay
                .load()
                .read_node_labels_visible(id, epoch, transaction_id);
        }
        // Base-only node: the overlay has no entry and no label delta. Return
        // the committed label set. Use the versioned fetch (not the epoch-blind
        // `get_node`) so the labels agree with the snapshot-aware gate above —
        // otherwise an old snapshot that may still see a base node
        // deleted-after-its-start would get an empty label set.
        let result: FxHashSet<arcstr::ArcStr> = self
            .get_node_versioned(id, epoch, tx)
            .map(|n| n.labels.iter().cloned().collect())
            .unwrap_or_default();
        if !result.is_empty()
            && let Some(tx) = transaction_id
        {
            // Record base-resident read into the Serializable read-set.
            self.overlay.load().record_read_node(tx, id);
        }
        result
    }

    fn nodes_by_label_visible(
        &self,
        label: &str,
        transaction_id: Option<TransactionId>,
    ) -> Vec<NodeId> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        let deleted = self.deleted_from_base_nodes.read().clone();
        let dirty = self.dirty_node_ids.read().clone();
        let overlay_ids = overlay.nodes_by_label_visible(label, transaction_id);

        // Base nodes (committed, non-dirty, non-deleted).
        let base_ids: Vec<NodeId> = self
            .base
            .load()
            .nodes_by_label(label)
            .into_iter()
            .filter(|id| !deleted.contains_key(id) && !dirty.contains(id))
            .collect();
        let overlay_ids: Vec<NodeId> = overlay_ids
            .into_iter()
            .filter(|id| !deleted.contains_key(id))
            .collect();
        drop(dirty);
        drop(deleted);
        // Record each base-resident node read for Serializable isolation.
        if let Some(tx) = transaction_id {
            for &id in &base_ids {
                overlay.record_read_node(tx, id);
            }
        }
        let mut ids = base_ids;

        // Overlay nodes — includes dirty promoted base nodes and new overlay
        // nodes; apply the tx label delta for the writing transaction.
        ids.extend(overlay_ids);
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

impl GraphStoreSearch for LayeredStore {
    fn lookup_nodes_indexed(
        &self,
        request: PropertyIndexRequest<'_>,
    ) -> grafeo_common::utils::error::Result<Option<Vec<NodeId>>> {
        let generation = GenerationReadScope::enter(self);
        let overlay = &generation._overlay;
        let Some(mut candidates) = overlay.lookup_nodes_indexed_candidates(request)? else {
            return Ok(None);
        };

        // The transferred LPG index includes the cold predecessor, but its
        // visibility view does not know Layered's base tombstone routing. Keep
        // the logical generation authoritative and recheck the property after
        // that routing decision.
        let transaction_id = request.transaction_id;
        let visibility_transaction = transaction_id.unwrap_or(TransactionId::INVALID);
        let property = PropertyKey::new(request.property);
        candidates.retain(|id| {
            if self.is_node_deleted_from_base_at(*id, request.epoch, visibility_transaction) {
                return false;
            }
            let visible = if self.is_node_dirty(*id) {
                transaction_id.map_or_else(
                    || overlay.get_node_at_epoch(*id, request.epoch).is_some(),
                    |tx| overlay.get_node_versioned(*id, request.epoch, tx).is_some(),
                )
            } else {
                let base = self.base.load();
                if Self::base_has_node_identity(&base, *id) {
                    base.get_node_at_epoch(*id, request.epoch).is_some()
                } else {
                    transaction_id.map_or_else(
                        || overlay.get_node_at_epoch(*id, request.epoch).is_some(),
                        |tx| overlay.get_node_versioned(*id, request.epoch, tx).is_some(),
                    )
                }
            };
            visible
                && self
                    .read_node_property_visible(*id, &property, request.epoch, transaction_id)
                    .is_some_and(|value| {
                        ExpressionPredicate::matches_property_index_predicate(
                            &value,
                            request.predicate,
                        )
                    })
        });
        candidates.sort_unstable();
        candidates.dedup();
        Ok(Some(candidates))
    }

    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().has_text_index(label, property)
    }

    #[cfg(feature = "text-index")]
    fn text_index_labels_for_property(&self, property: &str) -> Vec<String> {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().text_index_labels_for_property(property)
    }

    #[cfg(feature = "text-index")]
    fn score_text(&self, node_id: NodeId, label: &str, property: &str, query: &str) -> Option<f64> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        self.get_node(node_id)?;
        overlay.score_text(node_id, label, property, query)
    }

    #[cfg(feature = "text-index")]
    fn score_text_visible(
        &self,
        node_id: NodeId,
        label: &str,
        property: &str,
        query: &str,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Result<Option<f64>, Error> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        // Qualify retained history and record the index read even when row
        // visibility ultimately removes this candidate.
        let score = overlay.score_text_visible(node_id, label, property, query, epoch, tx)?;
        if self.get_node_versioned(node_id, epoch, tx).is_none() {
            return Ok(None);
        }
        Ok(score)
    }

    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        // Same-incarnation compaction moves a complete physical index to the
        // empty successor, so valid hits may now be cold-base identities. The
        // logical Layered view—not overlay row residency—is the visibility
        // authority. Over-fetch base tombstones so filtering cannot hide an
        // otherwise valid top-k result.
        let deleted = self.deleted_from_base_nodes.read().clone();
        let excluded = deleted.len();
        let mut results = overlay.text_search(label, property, query, k.saturating_add(excluded));
        results.retain(|(id, _)| !deleted.contains_key(id) && self.get_node(*id).is_some());
        results.truncate(k);
        results
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
    ) -> Vec<(NodeId, f64)> {
        let _generation = GenerationReadScope::enter(self);
        let deleted = self.deleted_from_base_nodes.read().clone();
        let overlay = self.overlay.load();
        let mut results = overlay.text_search_with_threshold(label, property, query, threshold);
        results.retain(|(id, _)| !deleted.contains_key(id) && self.get_node(*id).is_some());
        results
    }

    #[cfg(feature = "text-index")]
    fn text_search_visible(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Result<Vec<(NodeId, f64)>, Error> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        // Every cold tombstone is a possible filtered candidate, including
        // identities never hydrated into the overlay. Counting all is a safe
        // overfetch bound even when this snapshot predates some deletions.
        let excluded = self.deleted_from_base_nodes.read().len();
        let mut results = overlay.text_search_visible(
            label,
            property,
            query,
            k.saturating_add(excluded),
            epoch,
            tx,
        )?;
        results.retain(|(id, _)| self.get_node_versioned(*id, epoch, tx).is_some());
        results.truncate(k);
        Ok(results)
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold_visible(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
        epoch: EpochId,
        tx: TransactionId,
    ) -> Result<Vec<(NodeId, f64)>, Error> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        let mut results = overlay
            .text_search_with_threshold_visible(label, property, query, threshold, epoch, tx)?;
        results.retain(|(id, _)| self.get_node_versioned(*id, epoch, tx).is_some());
        Ok(results)
    }

    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().has_vector_index(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_metric(&self, label: &str, property: &str) -> Option<DistanceMetric> {
        let _generation = GenerationReadScope::enter(self);
        self.overlay.load().vector_index_metric(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        let deleted = self.deleted_from_base_nodes.read().clone();
        let excluded = deleted.len();
        let mut results = if let Some(label_name) = label
            && let Some(index) = overlay.get_vector_index(label_name, property)
            && index.config().metric == metric
        {
            // Ordinary HNSW stores exact topology but no duplicate vectors.
            // After compaction its candidates span both tiers, so score them
            // through this pinned Layered generation rather than the empty
            // successor overlay alone.
            let accessor = PropertyVectorAccessor::new(self, property);
            index
                .search(query, k.saturating_add(excluded), &accessor)
                .into_iter()
                .map(|(id, distance)| (id, f64::from(distance)))
                .collect()
        } else {
            // No matching physical index: preserve the exact brute-force
            // fallback across the complete logical Layered store.
            let node_ids = match label {
                Some(label) => self.nodes_by_label(label),
                None => self.node_ids(),
            };
            let property_key = PropertyKey::new(property);
            let mut candidates: Vec<(NodeId, f64)> = node_ids
                .into_iter()
                .filter_map(|id| {
                    self.get_node_property(id, &property_key)
                        .as_ref()
                        .and_then(value_to_vector)
                        .map(|vector| (id, f64::from(compute_distance(query, &vector, metric))))
                })
                .collect();
            candidates.sort_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k.saturating_add(excluded));
            candidates
        };
        results.retain(|(id, _)| !deleted.contains_key(id) && self.get_node(*id).is_some());
        results.truncate(k);
        results
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        let _generation = GenerationReadScope::enter(self);
        let deleted = self.deleted_from_base_nodes.read().clone();
        // HNSW has no threshold primitive. Scan the complete logical store,
        // not merely the now-empty successor overlay, and score exact values.
        let node_ids = match label {
            Some(label) => self.nodes_by_label(label),
            None => self.node_ids(),
        };
        let property_key = PropertyKey::new(property);
        let mut results: Vec<(NodeId, f64)> = node_ids
            .into_iter()
            .filter_map(|id| {
                self.get_node_property(id, &property_key)
                    .as_ref()
                    .and_then(value_to_vector)
                    .and_then(|vector| {
                        let distance = f64::from(compute_distance(query, &vector, metric));
                        (distance <= threshold).then_some((id, distance))
                    })
            })
            .collect();
        results.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.retain(|(id, _)| !deleted.contains_key(id) && self.get_node(*id).is_some());
        results
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_visible(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        epoch: grafeo_common::types::EpochId,
        tx: grafeo_common::types::TransactionId,
    ) -> Vec<(NodeId, f64)> {
        let _generation = GenerationReadScope::enter(self);
        let overlay = self.overlay.load();
        let all_node_ids = self.all_node_ids();
        let property_key = PropertyKey::new(property);
        let accessor = |id| {
            self.read_node_property_visible(id, &property_key, epoch, Some(tx))
                .as_ref()
                .and_then(value_to_vector)
        };
        let is_visible = |id| self.is_node_visible_versioned(id, epoch, tx);
        let has_label = |id, expected: &str| {
            self.read_node_labels_visible(id, epoch, Some(tx))
                .iter()
                .any(|candidate| candidate.as_str() == expected)
        };
        let context =
            VisibleVectorReadContext::new(&accessor, &all_node_ids, &is_visible, &has_label);
        let index_key = crate::graph::lpg::encode_index_key(label, property);
        let mut results = overlay
            .search_vector_visible_with_context(&index_key, query, k, tx, &context)
            .into_iter()
            .map(|(id, distance)| (id, f64::from(distance)))
            .collect::<Vec<_>>();
        results.retain(|(id, _)| self.get_node_versioned(*id, epoch, tx).is_some());
        results.truncate(k);
        results
    }
}

// ── GraphStoreMut implementation ───────────────────────────────────

impl GraphStoreMut for LayeredStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        self.with_pinned_overlay_mutation(
            || NodeId::INVALID,
            |overlay| {
                let id = overlay.create_node(labels);
                if id.is_valid() {
                    self.dirty_node_ids.write().insert(id);
                }
                id
            },
        )
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        self.with_pinned_overlay_mutation(
            || NodeId::INVALID,
            |overlay| {
                let id = overlay.create_node_versioned(labels, epoch, transaction_id);
                if id.is_valid() {
                    self.dirty_node_ids.write().insert(id);
                }
                id
            },
        )
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.with_pinned_overlay_mutation(
            || EdgeId::INVALID,
            |overlay| {
                // Promote base-only endpoints into the overlay.
                if !self.ensure_in_overlay(src) || !self.ensure_in_overlay(dst) {
                    return EdgeId::INVALID;
                }
                let id = overlay.create_edge(src, dst, edge_type);
                if id.is_valid() {
                    self.dirty_edge_ids.write().insert(id);
                }
                id
            },
        )
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        self.with_pinned_overlay_mutation(
            || EdgeId::INVALID,
            |overlay| {
                if !self.ensure_in_overlay_for_transaction(src, transaction_id)
                    || !self.ensure_in_overlay_for_transaction(dst, transaction_id)
                {
                    return EdgeId::INVALID;
                }
                let id = overlay.create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
                if id.is_valid() {
                    self.dirty_edge_ids.write().insert(id);
                }
                id
            },
        )
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        self.with_pinned_overlay_mutation(Vec::new, |overlay| {
            for &(src, dst, _) in edges {
                if !self.ensure_in_overlay(src) || !self.ensure_in_overlay(dst) {
                    return Vec::new();
                }
            }
            let ids = overlay.batch_create_edges(edges);
            let mut dirty = self.dirty_edge_ids.write();
            dirty.extend(ids.iter().copied().filter(EdgeId::is_valid));
            ids
        })
    }

    fn delete_node(&self, id: NodeId) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| {
                // Delete the overlay copy if present, and independently
                // tombstone the base copy if present. A promoted node lives in
                // both tiers (its base adjacency stays in the base), so both
                // must happen; a fresh overlay-only node has no base copy.
                let now = overlay.current_epoch();
                let is_base_node = self.base.load().get_node(id).is_some();
                let base_tombstoned = is_base_node && {
                    let mut deleted = self.deleted_from_base_nodes.write();
                    if deleted.contains_key(&id) {
                        // A committed tombstone is immutable history; a
                        // PENDING tombstone belongs to its original deleter.
                        // A repeat/competitor must never restamp either one.
                        false
                    } else {
                        deleted.insert(
                            id,
                            BaseNodeDelete {
                                epoch: now,
                                deleter: None,
                            },
                        );
                        true
                    }
                };
                // An existing base tombstone owns this stable identity. Do not
                // let a repeat/competitor reach a promoted overlay copy and
                // restamp its structural history under a different deleter.
                if is_base_node && !base_tombstoned {
                    return false;
                }
                if base_tombstoned {
                    // Publish checkpoint debt with the tombstone itself. If a
                    // later overlay-side operation unwinds, persistence must
                    // still observe the already-published base deletion.
                    self.deletions_dirty.store(true, Ordering::Release);
                }
                let overlay_removed = overlay.delete_node(id);
                overlay_removed || base_tombstoned
            },
        )
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| {
                let base_node = self.base.load().get_node(id);
                // Transactional base tombstone: stamp PENDING + deleter so the
                // deleter sees it gone immediately while other snapshots do
                // not until commit.
                let base_tombstoned = base_node.is_some() && {
                    // Global order is pending-map -> tombstone-map (the same
                    // order finalize/rollback use). Holding both makes first-
                    // writer ownership and its rollback/finalize queue one
                    // atomic bookkeeping publication.
                    let mut pending = self.pending_base_node_deletes.write();
                    let mut deleted = self.deleted_from_base_nodes.write();
                    if deleted.contains_key(&id) {
                        false
                    } else {
                        deleted.insert(
                            id,
                            BaseNodeDelete {
                                epoch: EpochId::PENDING,
                                deleter: Some(transaction_id),
                            },
                        );
                        pending.entry(transaction_id).or_default().push(id);
                        true
                    }
                };
                if base_node.is_some() && !base_tombstoned {
                    return false;
                }
                if base_tombstoned {
                    self.deletions_dirty.store(true, Ordering::Release);
                }
                let overlay_removed = overlay.delete_node_versioned(id, epoch, transaction_id);
                if base_tombstoned && !overlay_removed {
                    let labels: Vec<&str> = base_node
                        .as_ref()
                        .expect("base tombstone requires a base node")
                        .labels
                        .iter()
                        .map(ArcStr::as_str)
                        .collect();
                    overlay.record_coarse_node_predicates_by_name(transaction_id, &labels);
                }
                overlay_removed || base_tombstoned
            },
        )
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                let now = overlay.current_epoch();
                let base = self.base.load();
                let mut candidates: FxHashSet<EdgeId> = base
                    .edges_from(node_id, Direction::Both)
                    .into_iter()
                    .map(|(_, edge)| edge)
                    .collect();
                candidates.extend(
                    overlay
                        .edges_from(node_id, Direction::Both)
                        .map(|(_, edge)| edge),
                );
                let mut candidates: Vec<EdgeId> = candidates.into_iter().collect();
                candidates.sort_unstable();
                for edge in candidates {
                    let is_base_edge = base.get_edge(edge).is_some();
                    if is_base_edge {
                        let inserted = {
                            let mut tombstones = self.deleted_from_base_edges.write();
                            if tombstones.contains_key(&edge) {
                                false
                            } else {
                                tombstones.insert(
                                    edge,
                                    BaseEdgeDelete {
                                        epoch: now,
                                        deleter: None,
                                    },
                                );
                                true
                            }
                        };
                        if !inserted {
                            // Preserve the first delete's committed/PENDING
                            // owner in both layers.
                            continue;
                        }
                        self.deletions_dirty.store(true, Ordering::Release);
                    }
                    overlay.delete_edge(edge);
                }
            },
        );
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| {
                let now = overlay.current_epoch();
                let base_edge = self.base.load().get_edge(id);
                let base_tombstoned = base_edge.is_some() && {
                    let mut deleted = self.deleted_from_base_edges.write();
                    if deleted.contains_key(&id) {
                        false
                    } else {
                        deleted.insert(
                            id,
                            BaseEdgeDelete {
                                epoch: now,
                                deleter: None,
                            },
                        );
                        true
                    }
                };
                if base_edge.is_some() && !base_tombstoned {
                    return false;
                }
                if base_tombstoned {
                    self.deletions_dirty.store(true, Ordering::Release);
                }
                let overlay_removed = overlay.delete_edge(id);
                overlay_removed || base_tombstoned
            },
        )
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| {
                let base_edge = self.base.load().get_edge(id);
                let base_tombstoned = base_edge.is_some() && {
                    let mut pending = self.pending_base_edge_deletes.write();
                    let mut deleted = self.deleted_from_base_edges.write();
                    if deleted.contains_key(&id) {
                        false
                    } else {
                        deleted.insert(
                            id,
                            BaseEdgeDelete {
                                epoch: EpochId::PENDING,
                                deleter: Some(transaction_id),
                            },
                        );
                        pending.entry(transaction_id).or_default().push(id);
                        true
                    }
                };
                if base_edge.is_some() && !base_tombstoned {
                    return false;
                }
                if base_tombstoned {
                    self.deletions_dirty.store(true, Ordering::Release);
                }
                let overlay_removed = overlay.delete_edge_versioned(id, epoch, transaction_id);
                if base_tombstoned && !overlay_removed {
                    let rel_type = &base_edge
                        .as_ref()
                        .expect("base tombstone requires a base edge")
                        .edge_type;
                    overlay.record_coarse_edge_predicates_by_name(transaction_id, rel_type);
                }
                overlay_removed || base_tombstoned
            },
        )
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_in_overlay(id) {
                    overlay.set_node_property(id, key, value);
                }
            },
        );
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_in_overlay_for_transaction(id, transaction_id) {
                    overlay.set_node_property_versioned(id, key, value, transaction_id);
                }
            },
        );
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_edge_in_overlay(id) {
                    overlay.set_edge_property(id, key, value);
                }
            },
        );
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_edge_in_overlay(id) {
                    overlay.set_edge_property_versioned(id, key, value, transaction_id);
                }
            },
        );
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.with_pinned_overlay_mutation(
            || None,
            |overlay| {
                self.ensure_in_overlay(id)
                    .then(|| overlay.remove_node_property(id, key))
                    .flatten()
            },
        )
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.with_pinned_overlay_mutation(
            || None,
            |overlay| {
                self.ensure_in_overlay_for_transaction(id, transaction_id)
                    .then(|| overlay.remove_node_property_versioned(id, key, transaction_id))
                    .flatten()
            },
        )
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        self.with_pinned_overlay_mutation(
            || None,
            |overlay| {
                self.ensure_edge_in_overlay(id)
                    .then(|| overlay.remove_edge_property(id, key))
                    .flatten()
            },
        )
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.with_pinned_overlay_mutation(
            || None,
            |overlay| {
                self.ensure_edge_in_overlay(id)
                    .then(|| overlay.remove_edge_property_versioned(id, key, transaction_id))
                    .flatten()
            },
        )
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| self.ensure_in_overlay(node_id) && overlay.add_label(node_id, label),
        )
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| {
                self.ensure_in_overlay_for_transaction(node_id, transaction_id)
                    && overlay.add_label_versioned(node_id, label, transaction_id)
            },
        )
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| self.ensure_in_overlay(node_id) && overlay.remove_label(node_id, label),
        )
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        self.with_pinned_overlay_mutation(
            || false,
            |overlay| {
                self.ensure_in_overlay_for_transaction(node_id, transaction_id)
                    && overlay.remove_label_versioned(node_id, label, transaction_id)
            },
        )
    }

    // --- Task 5 label buffered writers (unified-MVCC) ---
    //
    // Mirror the property `*_buffered` overrides: `ensure_in_overlay` first so
    // the per-transaction label delta (held by the overlay LpgStore) can
    // associate the op with the promoted node, then delegate to the overlay's
    // buffered method. This guarantees read-your-writes on the overlay while
    // keeping the committed `label_index` clean (no dirty labels visible to
    // other sessions).

    fn add_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_in_overlay_for_transaction(node_id, transaction_id) {
                    overlay.add_label_buffered(node_id, label, transaction_id);
                }
            },
        );
    }

    fn remove_label_buffered(&self, node_id: NodeId, label: &str, transaction_id: TransactionId) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_in_overlay_for_transaction(node_id, transaction_id) {
                    overlay.remove_label_buffered(node_id, label, transaction_id);
                }
            },
        );
    }

    // --- Task 6: full delta delegation (unified-MVCC) ---
    //
    // The LayeredStore has no event/log side effects, so ALL new MVCC methods
    // are delegated directly to the overlay LpgStore. The overlay is the sole
    // holder of the per-transaction property delta; commit/rollback in the
    // session already operates on the overlay's write-set, so delegation is
    // end-to-end correct.

    fn set_node_property_buffered(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_in_overlay_for_transaction(id, transaction_id) {
                    overlay.set_node_property_buffered(id, key, value, transaction_id);
                }
            },
        );
    }

    fn remove_node_property_buffered(&self, id: NodeId, key: &str, transaction_id: TransactionId) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_in_overlay_for_transaction(id, transaction_id) {
                    overlay.remove_node_property_buffered(id, key, transaction_id);
                }
            },
        );
    }

    fn set_edge_property_buffered(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_edge_in_overlay_for_transaction(id, transaction_id) {
                    overlay.set_edge_property_buffered(id, key, value, transaction_id);
                }
            },
        );
    }

    fn remove_edge_property_buffered(&self, id: EdgeId, key: &str, transaction_id: TransactionId) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| {
                if self.ensure_edge_in_overlay_for_transaction(id, transaction_id) {
                    overlay.remove_edge_property_buffered(id, key, transaction_id);
                }
            },
        );
    }

    fn apply_tx_overlay(&self, transaction_id: TransactionId) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| overlay.apply_tx_overlay(transaction_id),
        );
    }

    fn drop_tx_overlay(&self, transaction_id: TransactionId) {
        let _guard = self.merge_guard.read();
        let overlay = self.overlay.load_full();
        let Some(_mutation) = overlay.pin_mutation() else {
            return;
        };
        overlay.drop_tx_overlay(transaction_id);
        // Rollback path: undo this tx's uncommitted base tombstones (nodes AND
        // edges). `drop_tx_overlay` is only invoked on the abort/conflict paths
        // (commit uses `apply_tx_overlay` + finalize), so removing the PENDING
        // base tombstones here restores the entities for everyone.
        let pending_nodes = self
            .pending_base_node_deletes
            .write()
            .remove(&transaction_id);
        if let Some(ids) = pending_nodes {
            let mut tombstones = self.deleted_from_base_nodes.write();
            for id in ids {
                // Only remove if it is still this tx's PENDING tombstone — a
                // competing or repeated delete is rejected rather than
                // overwriting the stamp, so ownership cannot migrate.
                if let Some(d) = tombstones.get(&id)
                    && d.deleter == Some(transaction_id)
                    && d.epoch == EpochId::PENDING
                {
                    tombstones.remove(&id);
                }
            }
            drop(tombstones);
            self.deletions_dirty.store(true, Ordering::Release);
        }
        let pending = self
            .pending_base_edge_deletes
            .write()
            .remove(&transaction_id);
        if let Some(ids) = pending {
            let mut tombstones = self.deleted_from_base_edges.write();
            for id in ids {
                // Only remove if it is still this tx's PENDING tombstone — a
                // competing or repeated delete is rejected rather than
                // overwriting the stamp, so ownership cannot migrate.
                if let Some(d) = tombstones.get(&id)
                    && d.deleter == Some(transaction_id)
                    && d.epoch == EpochId::PENDING
                {
                    tombstones.remove(&id);
                }
            }
            drop(tombstones);
            self.deletions_dirty.store(true, Ordering::Release);
        }
    }

    fn finalize_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        node_ids: &[NodeId],
    ) {
        let _guard = self.merge_guard.read();
        let overlay = self.overlay.load_full();
        let Some(_mutation) = overlay.pin_mutation() else {
            return;
        };
        overlay.finalize_deletes_by_id(transaction_id, commit_epoch, node_ids);
        // Commit path: stamp this tx's PENDING base-node tombstones with the
        // real commit epoch. Driven from the LayeredStore's own per-tx list, NOT
        // the `node_ids` slice: a base-only node delete never reaches the
        // overlay's pending node-delete list, so it would be absent from
        // `node_ids` (which comes from `take_pending_deletes`). Mirror of
        // `finalize_edge_deletes_by_id`.
        let pending = self
            .pending_base_node_deletes
            .write()
            .remove(&transaction_id);
        if let Some(ids) = pending {
            let mut tombstones = self.deleted_from_base_nodes.write();
            for id in ids {
                if let Some(d) = tombstones.get_mut(&id)
                    && d.deleter == Some(transaction_id)
                    && d.epoch == EpochId::PENDING
                {
                    d.epoch = commit_epoch;
                }
            }
            drop(tombstones);
            self.deletions_dirty.store(true, Ordering::Release);
        }
    }

    fn take_pending_deletes(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.with_pinned_overlay_mutation(Vec::new, |overlay| {
            overlay.take_pending_deletes(transaction_id)
        })
    }

    fn finalize_edge_deletes_by_id(
        &self,
        transaction_id: TransactionId,
        commit_epoch: EpochId,
        edges: &[(NodeId, EdgeId, NodeId)],
    ) {
        let _guard = self.merge_guard.read();
        let overlay = self.overlay.load_full();
        let Some(_mutation) = overlay.pin_mutation() else {
            return;
        };
        overlay.finalize_edge_deletes_by_id(transaction_id, commit_epoch, edges);
        // Commit path: stamp this tx's PENDING base-edge tombstones with the
        // real commit epoch. Driven from the LayeredStore's own per-tx list,
        // NOT the `edges` slice: a base-only edge delete never reaches the
        // overlay's `pending_tx_edge_deletes`, so it would be absent from
        // `edges` (which comes from `take_pending_edge_deletes`).
        let pending = self
            .pending_base_edge_deletes
            .write()
            .remove(&transaction_id);
        if let Some(ids) = pending {
            let mut tombstones = self.deleted_from_base_edges.write();
            for id in ids {
                if let Some(d) = tombstones.get_mut(&id)
                    && d.deleter == Some(transaction_id)
                    && d.epoch == EpochId::PENDING
                {
                    d.epoch = commit_epoch;
                }
            }
            drop(tombstones);
            self.deletions_dirty.store(true, Ordering::Release);
        }
    }

    fn take_pending_edge_deletes(
        &self,
        transaction_id: TransactionId,
    ) -> Vec<(NodeId, EdgeId, NodeId)> {
        self.with_pinned_overlay_mutation(Vec::new, |overlay| {
            overlay.take_pending_edge_deletes(transaction_id)
        })
    }

    fn tx_overlay_snapshot(&self, transaction_id: TransactionId) -> crate::graph::lpg::TxDelta {
        let _guard = self.merge_guard.read();
        self.overlay.load().tx_overlay_snapshot(transaction_id)
    }

    fn tx_overlay_restore(
        &self,
        transaction_id: TransactionId,
        snapshot: crate::graph::lpg::TxDelta,
    ) {
        self.with_pinned_overlay_mutation(
            || {},
            |overlay| overlay.tx_overlay_restore(transaction_id, snapshot),
        );
    }

    fn tx_structural_snapshot(&self, transaction_id: TransactionId) -> TxStructuralSnapshot {
        let _guard = self.merge_guard.read();
        let overlay = self.overlay.load_full();
        self.tx_structural_snapshot_for_overlay(transaction_id, &overlay)
    }

    fn tx_structural_restore(
        &self,
        transaction_id: TransactionId,
        snapshot: TxStructuralSnapshot,
    ) -> std::result::Result<(), String> {
        let _guard = self.merge_guard.read();
        let overlay = self.overlay.load_full();
        let Some(_mutation) = overlay.pin_mutation() else {
            return Err(format!(
                "transaction {} layered structural restore requires this store's write authority",
                transaction_id.as_u64()
            ));
        };
        #[cfg(test)]
        if let Some(hook) = self.structural_restore_hook.read().clone() {
            hook();
        }
        let current = self.tx_structural_snapshot_for_overlay(transaction_id, &overlay);
        if !current.node_creates.starts_with(&snapshot.node_creates)
            || !current.edge_creates.starts_with(&snapshot.edge_creates)
            || !current.node_deletes.starts_with(&snapshot.node_deletes)
            || !current.edge_deletes.starts_with(&snapshot.edge_deletes)
            || !current
                .base_node_deletes
                .starts_with(&snapshot.base_node_deletes)
            || !current
                .base_edge_deletes
                .starts_with(&snapshot.base_edge_deletes)
        {
            return Err(format!(
                "transaction {} layered structural queues no longer extend the savepoint snapshot",
                transaction_id.as_u64()
            ));
        }

        let rolled_back_base_nodes =
            current.base_node_deletes[snapshot.base_node_deletes.len()..].to_vec();
        let rolled_back_base_edges =
            current.base_edge_deletes[snapshot.base_edge_deletes.len()..].to_vec();
        let overlay_snapshot = TxStructuralSnapshot {
            node_creates: snapshot.node_creates,
            edge_creates: snapshot.edge_creates,
            node_deletes: snapshot.node_deletes,
            edge_deletes: snapshot.edge_deletes,
            ..TxStructuralSnapshot::default()
        };
        overlay.tx_structural_restore(transaction_id, overlay_snapshot)?;

        {
            let mut pending = self.pending_base_node_deletes.write();
            if snapshot.base_node_deletes.is_empty() {
                pending.remove(&transaction_id);
            } else {
                pending.insert(transaction_id, snapshot.base_node_deletes);
            }
        }
        {
            let mut pending = self.pending_base_edge_deletes.write();
            if snapshot.base_edge_deletes.is_empty() {
                pending.remove(&transaction_id);
            } else {
                pending.insert(transaction_id, snapshot.base_edge_deletes);
            }
        }
        {
            let mut tombstones = self.deleted_from_base_nodes.write();
            for id in rolled_back_base_nodes {
                if tombstones.get(&id).is_some_and(|delete| {
                    delete.deleter == Some(transaction_id) && delete.epoch == EpochId::PENDING
                }) {
                    tombstones.remove(&id);
                }
            }
        }
        {
            let mut tombstones = self.deleted_from_base_edges.write();
            for id in rolled_back_base_edges {
                if tombstones.get(&id).is_some_and(|delete| {
                    delete.deleter == Some(transaction_id) && delete.epoch == EpochId::PENDING
                }) {
                    tombstones.remove(&id);
                }
            }
        }
        if !current.base_node_deletes.is_empty() || !current.base_edge_deletes.is_empty() {
            self.deletions_dirty.store(true, Ordering::Release);
        }
        Ok(())
    }
}

// ── Private helpers ────────────────────────────────────────────────

impl LayeredStore {
    fn value_matches_compare(actual: &Value, op: CompareOp, expected: &Value) -> bool {
        let Some(ordering) = super::zone_map::compare_values(actual, expected) else {
            // Incomparable/custom values cannot be safely pruned.
            return true;
        };
        match op {
            CompareOp::Eq => ordering.is_eq(),
            CompareOp::Ne => !ordering.is_eq(),
            CompareOp::Lt => ordering.is_lt(),
            CompareOp::Le => !ordering.is_gt(),
            CompareOp::Gt => ordering.is_gt(),
            CompareOp::Ge => !ordering.is_lt(),
        }
    }

    /// Whether the cold generation owns this node identity, including a node
    /// retained only by the temporal sidecar after its current lifetime closed.
    /// A current-value lookup alone is insufficient: returning `None` for a
    /// closed row is authoritative and must not fall through to an overlapping
    /// physical overlay row.
    #[inline]
    fn base_has_node_identity(base: &CompactStore, id: NodeId) -> bool {
        base.get_node(id).is_some() || base.temporal_nodes().contains_key(&id)
    }

    /// Edge-side identity ownership, covering live CSR rows and every retained
    /// closed lifetime.
    #[inline]
    fn base_has_edge_identity(base: &CompactStore, id: EdgeId) -> bool {
        base.edge_validity(id).is_some()
    }

    /// Whether an overlay node is part of the logical layered view under this
    /// stable base/dirty snapshot. A row that overlaps the cold base remains
    /// private hydration state until the dirty marker publishes it.
    #[inline]
    fn overlay_node_is_published(
        base: &CompactStore,
        dirty: &FxHashSet<NodeId>,
        id: NodeId,
    ) -> bool {
        !Self::base_has_node_identity(base, id) || dirty.contains(&id)
    }

    /// Edge-side counterpart of [`Self::overlay_node_is_published`].
    #[inline]
    fn overlay_edge_is_published(
        base: &CompactStore,
        dirty: &FxHashSet<EdgeId>,
        id: EdgeId,
    ) -> bool {
        !Self::base_has_edge_identity(base, id) || dirty.contains(&id)
    }

    /// Ensures a node exists in the overlay. If the node is base-only,
    /// copies its complete structural, label, and property history into the
    /// overlay and marks it dirty.
    ///
    /// Uses the crate-private same-incarnation promotion seam, which requires a
    /// vacant exact ID and preserves every cold structural lifetime and label
    /// version. A failed or unauthorized structural publication never reaches
    /// the dirty routing set.
    fn ensure_in_overlay(&self, id: NodeId) -> bool {
        self.try_ensure_in_overlay(id)
            .expect("overlay node allocation failed during promotion")
    }

    /// Fallible form of [`Self::ensure_in_overlay`] used by recovery, where an
    /// allocator failure must remain a structured open error rather than cross
    /// the database boundary as a panic.
    fn try_ensure_in_overlay(&self, id: NodeId) -> Result<bool, grafeo_common::memory::AllocError> {
        if self.is_node_dirty(id) {
            return Ok(self.overlay.load().get_node(id).is_some());
        }
        #[cfg(test)]
        if let Some(barrier) = self.node_promotion_barrier.read().clone() {
            barrier.wait();
        }
        let _promotion = self.node_promotion_locks[promotion_lock_shard(id.as_u64())].lock();
        if self.is_node_dirty(id) {
            return Ok(self.overlay.load().get_node(id).is_some());
        }
        let base = self.base.load();
        let Some(base_node) = base.get_node(id) else {
            // Overlay-only identities need no promotion. This also rejects a
            // missing or already-deleted identity rather than publishing a
            // dirty routing marker for it.
            return Ok(self.overlay.load().get_node(id).is_some());
        };
        let (lifetimes, label_versions, property_history) =
            if let Some(history) = base.temporal_node_history(id) {
                (
                    history
                        .lifetimes
                        .iter()
                        .map(|lifetime| (lifetime.created, lifetime.deleted))
                        .collect::<Vec<_>>(),
                    history.label_versions,
                    history.properties.into_iter().collect::<Vec<_>>(),
                )
            } else {
                // A legacy/all-open base genuinely carries no temporal
                // sidecar. INITIAL + its canonical current label image is the
                // complete history representable by that format; temporal
                // metadata, when present, is never collapsed through here.
                let mut labels: Vec<ArcStr> = base_node.labels.iter().cloned().collect();
                labels.sort_unstable();
                labels.dedup();
                (
                    vec![(EpochId::INITIAL, None)],
                    vec![(EpochId::INITIAL, labels)],
                    base.node_property_history(id),
                )
            };
        drop(base);

        // Insert at the exact base id through the representation-only seam.
        let overlay = self.overlay.load();
        let Some(promotion) =
            overlay.promote_same_incarnation_node_with_id(id, &lifetimes, &label_versions)?
        else {
            return Ok(false);
        };

        // Replay every authoritative event at its original epoch. Current-value
        // copying would invent a promotion-epoch version for every untouched
        // property and erase tombstones/same-epoch ordered writes.
        let mut replayed_keys = Vec::with_capacity(property_history.len());
        for (key, versions) in property_history {
            replayed_keys.push(key.clone());
            for (epoch, value) in versions {
                overlay.hydrate_node_property_at_epoch(id, key.as_str(), value, epoch);
                #[cfg(test)]
                if let Some(hook) = self.node_replay_event_hook.read().clone() {
                    hook();
                }
            }
        }
        overlay.reconcile_node_overlay_metadata_after_hydration(id, &replayed_keys);

        #[cfg(test)]
        if let Some(hook) = self.node_publication_hook.read().clone() {
            hook();
        }
        self.dirty_node_ids.write().insert(id);
        // Disarm physical rollback only after routing publishes the complete
        // hydration. Any unwind during replay or dirty-set insertion removes
        // the private structural row and overlay-local metadata. Whole-graph
        // Property/Text/Vector indexes are never mutated by representation-only hydration.
        promotion.commit();
        Ok(true)
    }

    /// Transaction-aware admission for a node mutation.
    ///
    /// Called under the existing merge-read and exact-overlay mutation pin.
    /// Overlay-only identities include direct Session creates without a dirty
    /// marker. A base-owned physical overlap, however, is private hydration
    /// until promotion publishes it; it must retain the complete promotion path.
    fn ensure_in_overlay_for_transaction(&self, id: NodeId, tx: TransactionId) -> bool {
        let overlay = self.overlay.load();
        let epoch = overlay.current_epoch();
        if self.is_node_deleted_from_base_at(id, epoch, tx) {
            return false;
        }
        if self.is_node_dirty(id) {
            return overlay.is_node_visible_versioned(id, epoch, tx);
        }
        let base_visible = {
            let base = self.base.load();
            if !Self::base_has_node_identity(&base, id) {
                return overlay.is_node_visible_versioned(id, epoch, tx);
            }
            base.is_node_visible_versioned(id, epoch, tx)
        };
        // Release base/routing/tombstone guards before acquiring promotion stripes.
        base_visible
            && self.ensure_in_overlay(id)
            && !self.is_node_deleted_from_base_at(id, epoch, tx)
            && overlay.is_node_visible_versioned(id, epoch, tx)
    }

    /// Edge counterpart for buffered mutation, under the same existing pin.
    fn ensure_edge_in_overlay_for_transaction(&self, id: EdgeId, tx: TransactionId) -> bool {
        let overlay = self.overlay.load();
        let epoch = overlay.current_epoch();
        if self.is_edge_deleted_from_base_at(id, epoch, tx) {
            return false;
        }
        if self.is_edge_dirty(id) {
            return overlay.is_edge_visible_versioned(id, epoch, tx);
        }
        let base_visible = {
            let base = self.base.load();
            if !Self::base_has_edge_identity(&base, id) {
                return overlay.is_edge_visible_versioned(id, epoch, tx);
            }
            base.is_edge_visible_versioned(id, epoch, tx)
        };
        base_visible
            && self.ensure_edge_in_overlay(id)
            && !self.is_edge_deleted_from_base_at(id, epoch, tx)
            && overlay.is_edge_visible_versioned(id, epoch, tx)
    }

    /// Ensures an edge exists in the overlay.
    ///
    /// Uses the crate-private same-incarnation promotion seam, which preserves
    /// both structural time and any already-live transport nonce without
    /// minting new authority. A failed or unauthorized publication never
    /// reaches the dirty routing set.
    fn ensure_edge_in_overlay(&self, id: EdgeId) -> bool {
        if self.is_edge_dirty(id) {
            return self.overlay.load().get_edge(id).is_some();
        }
        #[cfg(test)]
        if let Some(barrier) = self.edge_promotion_barrier.read().clone() {
            barrier.wait();
        }
        let _promotion = self.edge_promotion_locks[promotion_lock_shard(id.as_u64())].lock();
        if self.is_edge_dirty(id) {
            return self.overlay.load().get_edge(id).is_some();
        }
        let Some(base_edge) = self.base.load().get_edge(id) else {
            return self.overlay.load().get_edge(id).is_some();
        };
        let history = self.edge_full_history(id);
        let lifetimes: Vec<_> = history
            .lifetimes
            .iter()
            .map(|life| (life.created, life.deleted))
            .collect();
        let property_history = history.properties;

        // Ensure endpoints are in the overlay first.
        if !self.ensure_in_overlay(base_edge.src) || !self.ensure_in_overlay(base_edge.dst) {
            return false;
        }

        // Insert at the exact base id through the representation-only seam.
        let overlay = self.overlay.load();
        let promotion = overlay
            .promote_same_incarnation_edge_with_id(
                id,
                base_edge.src,
                base_edge.dst,
                base_edge.edge_type.as_str(),
                &lifetimes,
            )
            .expect("overlay edge allocation failed during promotion")
            .expect("base edge promotion lost its vacant identity or exact overlay write scope");

        // Edge properties have no derived indexes/count metadata, so exact
        // at-epoch replay is the complete hydration step.
        for (key, versions) in property_history {
            for (epoch, value) in versions {
                overlay.set_edge_property_at_epoch(id, key.as_str(), value, epoch);
                #[cfg(test)]
                if let Some(hook) = self.edge_replay_event_hook.read().clone() {
                    hook();
                }
            }
        }

        #[cfg(test)]
        if let Some(hook) = self.edge_publication_hook.read().clone() {
            hook();
        }
        self.dirty_edge_ids.write().insert(id);
        promotion.commit();
        true
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::{OperatorError, WriteTracker};
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::compact::section::CompactStoreSection;
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use grafeo_common::storage::section::Section;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::Duration;

    include!("generation_totality_tests.rs");

    fn purge_transport_for_test<P, R>(
        layered: &LayeredStore,
        receipts: &[&TransportEdgeReceipt],
        prepare: impl FnOnce(Arc<CompactStore>) -> Result<(Arc<CompactStore>, P), String>,
        publish: impl FnOnce(P) -> R,
        rollback: impl FnOnce(R),
    ) -> Result<bool, String> {
        let authority = WriteAuthority::new();
        with_authority(&authority, || {
            layered.purge_transport_extract_edges(receipts, &authority, prepare, publish, rollback)
        })
    }

    fn receipt_belongs_to_current_overlay(
        layered: &LayeredStore,
        receipt: &TransportEdgeReceipt,
    ) -> bool {
        layered
            .overlay_store()
            .transport_receipt_belongs_to_current_incarnation(receipt)
    }

    fn compact_transport_source(source: Arc<LpgStore>) -> LayeredStore {
        let empty = Arc::new(
            from_graph_store_preserving_ids(&LpgStore::new().expect("empty transport base source"))
                .expect("empty transport base"),
        );
        let layered = LayeredStore::with_overlay(empty, source).expect("adopt transport source");
        layered
            .merge_overlay_in_place()
            .expect("compact transport source into a same-incarnation generation");
        layered
    }

    #[test]
    fn transport_multiple_lifetimes_survive_compaction_delete_and_purge() {
        for lives in [
            vec![
                (EpochId::new(1), Some(EpochId::new(2))),
                (EpochId::new(3), None),
            ],
            vec![
                (EpochId::new(1), Some(EpochId::new(1))),
                (EpochId::new(1), None),
            ],
            vec![
                (EpochId::new(1), Some(EpochId::new(1))),
                (EpochId::new(1), Some(EpochId::new(1))),
                (EpochId::new(1), None),
            ],
        ] {
            for promote in [false, true] {
                let source = Arc::new(LpgStore::new().unwrap());
                let src = source.create_node(&["Source"]);
                let dst = source.create_node(&["Destination"]);
                let edge = EdgeId::new(601);
                let receipt = source
                    .restore_transport_edge_history_exact(edge, src, dst, "CARRIED", &lives)
                    .unwrap();
                source.sync_epoch(EpochId::new(3));
                let layered = compact_transport_source(source);
                assert_eq!(layered.complete_edge_history(edge).len(), lives.len());
                let rows: Vec<_> = layered
                    .base
                    .load()
                    .structural_edge_rows()
                    .into_iter()
                    .filter(|(id, _)| *id == edge)
                    .map(|(_, iv)| (iv.from(), (!iv.is_open()).then(|| iv.to())))
                    .collect();
                assert_eq!(rows, lives, "exact compact structural-row multiplicity");
                assert!(layered.is_transport_extract_edge(&receipt));
                if promote {
                    layered.set_edge_property(edge, "weight", Value::from(7i64));
                    assert_eq!(
                        layered.overlay_store().get_edge_history(edge).len(),
                        lives.len()
                    );
                    assert!(layered.is_transport_extract_edge(&receipt));
                }
                layered.overlay_store().sync_epoch(EpochId::new(4));
                assert!(layered.delete_edge(edge));
                assert_eq!(
                    layered.classify_transport_extract_edges(
                        &[&receipt],
                        EpochId::new(3),
                        EpochId::new(4)
                    ),
                    [TransportEdgeState::Closed]
                );
                assert!(
                    purge_transport_for_test(
                        &layered,
                        &[&receipt],
                        |base| Ok((base, ())),
                        |()| (),
                        |()| ()
                    )
                    .unwrap()
                );
                assert!(layered.complete_edge_history(edge).is_empty());
                assert!(!layered.is_transport_extract_edge(&receipt));
            }
        }
    }

    fn closed_base_transport_fixture(edge: EdgeId) -> (LayeredStore, TransportEdgeReceipt) {
        let source = Arc::new(LpgStore::new().expect("base-only transport source"));
        let src = source.create_node(&["Source"]);
        let dst = source.create_node(&["Destination"]);
        source.sync_epoch(EpochId::new(1));
        let receipt = source
            .create_transport_edge_with_id(edge, src, dst, "CARRIED")
            .expect("transport allocation")
            .expect("fresh transport identity");
        let layered = compact_transport_source(source);
        layered.overlay_store().sync_epoch(EpochId::new(2));
        assert!(layered.delete_edge(edge));
        assert!(
            !layered.overlay_store().all_known_edge_ids().contains(&edge),
            "fixture receipt must be authority-only in the overlay"
        );
        (layered, receipt)
    }

    fn missing_destination_grant_fixture() -> (
        LayeredStore,
        NodeId,
        NodeId,
        TransportEdgeMutationGrant,
        WriteAuthority,
    ) {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let source = overlay.create_node(&["Source"]);
        let missing_destination = NodeId::new(8_900_001);
        let receipt = overlay
            .create_transport_edge_with_id(
                EdgeId::new(8_900_002),
                source,
                missing_destination,
                "CARRIED",
            )
            .expect("transport allocation")
            .expect("present source authorizes transport identity");
        let owner = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));
        let grant = with_authority(&owner, || {
            layered
                .grant_transport_edge_mutation(&receipt, &owner)
                .expect("exact held owner mints layered transport grant")
        });
        (layered, source, missing_destination, grant, owner)
    }

    #[test]
    fn missing_destination_grant_rejects_committed_source_tombstone() {
        let (layered, source, missing_destination, grant, owner) =
            missing_destination_grant_fixture();
        assert!(with_authority(&owner, || layered.delete_node(source)));
        assert_eq!(
            layered.complete_edge_history(grant.edge_id()).len(),
            1,
            "the edge remains structurally eligible so rejection is due to source visibility"
        );
        assert!(!with_authority(&owner, || {
            layered.accepts_transport_edge_missing_destination(&grant, missing_destination, &owner)
        }));
    }

    #[test]
    fn missing_destination_grant_rejects_pending_source_tombstone() {
        let (layered, source, missing_destination, grant, owner) =
            missing_destination_grant_fixture();
        let deleting_transaction = TransactionId::new(8_900_003);
        assert!(with_authority(&owner, || layered.delete_node_versioned(
            source,
            layered.overlay_store().current_epoch(),
            deleting_transaction,
        )));
        assert!(
            layered.get_node(source).is_some(),
            "an unrelated latest reader still sees a transactionally deleted source"
        );
        assert_eq!(
            layered.complete_edge_history(grant.edge_id()).len(),
            1,
            "the carried edge remains open while its source delete is pending"
        );
        assert!(!with_authority(&owner, || {
            layered.accepts_transport_edge_missing_destination(&grant, missing_destination, &owner)
        }));
    }

    #[test]
    fn layered_transport_receipt_is_revoked_before_exact_same_shape_recreation() {
        let source = Arc::new(LpgStore::new().expect("source store"));
        let src = source.create_node(&["Source"]);
        let dst = source.create_node(&["Destination"]);
        source.sync_epoch(EpochId::new(1));
        let edge = EdgeId::new(600);
        let receipt = source
            .create_transport_edge_with_id(edge, src, dst, "CARRIED")
            .expect("transport allocation")
            .expect("fresh transport identity");

        let layered = compact_transport_source(source);
        layered.overlay.load().sync_epoch(EpochId::new(2));
        assert!(layered.delete_edge(edge));
        assert!(receipt_belongs_to_current_overlay(&layered, &receipt));
        assert_eq!(
            layered
                .classify_transport_extract_edges(&[&receipt], EpochId::new(1), EpochId::new(2),),
            [TransportEdgeState::Closed],
            "closed compact identity must retain its receipt provenance"
        );
        assert!(
            purge_transport_for_test(
                &layered,
                &[&receipt],
                |generation| Ok((generation, ())),
                |()| {},
                |()| {},
            )
            .expect("purge compact transport generation")
        );

        layered.ensure_in_overlay(src);
        layered.ensure_in_overlay(dst);
        let overlay = layered.overlay.load();
        overlay
            .restore_edge_history_exact(edge, src, dst, "CARRIED", &[(EpochId::new(3), None)])
            .expect("hostile same-shape recreation in overlay");
        overlay.set_edge_property_at_epoch(edge, "proof", Value::from("ordinary"), EpochId::new(3));
        overlay.sync_epoch(EpochId::new(4));
        drop(overlay);
        layered.dirty_edge_ids.write().insert(edge);
        assert!(layered.delete_edge(edge));

        assert!(!layered.is_transport_extract_edge(&receipt));
        assert!(
            !purge_transport_for_test(
                &layered,
                &[&receipt],
                |generation| Ok((generation, ())),
                |()| {},
                |()| {},
            )
            .expect("stale receipt refusal")
        );
        assert_eq!(layered.complete_edge_history(edge).len(), 1);
    }

    #[test]
    fn layered_transport_authority_follows_overlay_clear_incarnation_rotation() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let source = NodeId::new(8_910_001);
        let destination = NodeId::new(8_910_002);
        let edge = EdgeId::new(8_910_003);
        overlay
            .create_node_with_id(source, &["Source"])
            .expect("source allocation");
        overlay
            .create_node_with_id(destination, &["Destination"])
            .expect("destination allocation");
        let stale = overlay
            .create_transport_edge_with_id(edge, source, destination, "CARRIED")
            .expect("transport allocation")
            .expect("fresh transport identity");
        assert!(layered.is_transport_extract_edge(&stale));

        overlay.clear();
        overlay
            .create_node_with_id(source, &["Source"])
            .expect("same-shape source recreation");
        overlay
            .create_node_with_id(destination, &["Destination"])
            .expect("same-shape destination recreation");
        let current = overlay
            .create_transport_edge_with_id(edge, source, destination, "CARRIED")
            .expect("same-shape transport allocation")
            .expect("rotated incarnation accepts a new receipt");

        assert!(layered.is_transport_extract_edge(&current));
        assert!(
            !layered.is_transport_extract_edge(&stale),
            "Layered provenance must follow the current overlay authority, not a cached predecessor"
        );
    }

    #[test]
    fn base_only_transport_purge_requires_the_exact_held_overlay_write_scope() {
        let (layered, receipt) = closed_base_transport_fixture(EdgeId::new(8_920_001));
        let overlay = layered.overlay_store();
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));

        let prepare_calls = AtomicUsize::new(0);
        assert!(
            !layered
                .purge_transport_extract_edges(
                    &[&receipt],
                    &owner,
                    |generation| {
                        prepare_calls.fetch_add(1, Ordering::SeqCst);
                        Ok((generation, ()))
                    },
                    |()| (),
                    |()| (),
                )
                .expect("missing scope is a qualification miss")
        );
        assert!(
            !with_authority(&foreign, || layered.purge_transport_extract_edges(
                &[&receipt],
                &foreign,
                |generation| {
                    prepare_calls.fetch_add(1, Ordering::SeqCst);
                    Ok((generation, ()))
                },
                |()| (),
                |()| (),
            ))
            .expect("foreign scope is a qualification miss")
        );
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
        assert!(receipt_belongs_to_current_overlay(&layered, &receipt));
        assert_eq!(layered.complete_edge_history(receipt.edge_id()).len(), 1);

        assert!(
            with_authority(&owner, || layered.purge_transport_extract_edges(
                &[&receipt],
                &owner,
                |generation| Ok((generation, ())),
                |()| (),
                |()| (),
            ))
            .expect("exact owner purges base-only transport authority")
        );
        assert!(!receipt_belongs_to_current_overlay(&layered, &receipt));
        assert!(layered.complete_edge_history(receipt.edge_id()).is_empty());
    }

    #[test]
    fn layered_transport_overlay_rejection_happens_before_generation_prepare() {
        let source = Arc::new(LpgStore::new().expect("source store"));
        let src = source.create_node(&["Source"]);
        let dst = source.create_node(&["Destination"]);
        source.sync_epoch(EpochId::new(1));
        let edge = EdgeId::new(601);
        let receipt = source
            .create_transport_edge_with_id(edge, src, dst, "CARRIED")
            .expect("transport allocation")
            .expect("fresh transport identity");

        let layered = compact_transport_source(source);
        layered.overlay.load().sync_epoch(EpochId::new(2));
        layered.set_edge_property(edge, "promoted", Value::from("committed"));
        assert!(layered.delete_edge(edge));

        // A still-pending overlay entry makes physical removal ineligible even
        // though the committed structural view is closed. The inner exact
        // qualifier must reject it before any disk/base preparation runs.
        layered.overlay.load().set_edge_property_buffered(
            edge,
            "pending",
            Value::from("uncommitted"),
            TransactionId::new(602),
        );
        let old_base = layered.base_store_arc();
        let prepare_called = AtomicBool::new(false);
        assert!(
            !purge_transport_for_test(
                &layered,
                &[&receipt],
                |generation| {
                    prepare_called.store(true, Ordering::SeqCst);
                    Ok((generation, ()))
                },
                |()| {},
                |()| {},
            )
            .expect("qualification miss is not a preparation error")
        );
        assert!(!prepare_called.load(Ordering::SeqCst));
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &old_base));
        assert!(receipt_belongs_to_current_overlay(&layered, &receipt));
    }

    #[test]
    fn layered_transport_finalize_panic_publishes_nothing_and_retry_succeeds() {
        let source = Arc::new(LpgStore::new().expect("source store"));
        let src = source.create_node(&["Source"]);
        let dst = source.create_node(&["Destination"]);
        source.sync_epoch(EpochId::new(1));
        let edge = EdgeId::new(603);
        let receipt = source
            .create_transport_edge_with_id(edge, src, dst, "CARRIED")
            .expect("transport allocation")
            .expect("fresh transport identity");

        let layered = compact_transport_source(source);
        layered.overlay.load().sync_epoch(EpochId::new(2));
        layered.set_edge_property(edge, "promoted", Value::from("committed"));
        assert!(layered.delete_edge(edge));
        assert!(
            receipt_belongs_to_current_overlay(&layered, &receipt),
            "same-incarnation base promotion must preserve transport provenance"
        );
        assert!(
            layered
                .overlay_store()
                .is_transport_extract_closed_edge_unchanged_since(
                    &receipt,
                    layered.overlay_store().current_epoch(),
                ),
            "resident promoted history must remain physically purge-qualified"
        );
        assert_eq!(
            layered
                .classify_transport_extract_edges(&[&receipt], EpochId::new(1), EpochId::new(2),),
            [TransportEdgeState::Closed],
            "combined compact/overlay history must remain logically purge-qualified"
        );

        let original_base = layered.base_store_arc();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<bool, String> = purge_transport_for_test(
                &layered,
                &[&receipt],
                |generation| Ok((generation, ())),
                |()| panic!("hostile transport metadata finalizer"),
                |()| {},
            );
        }));
        assert!(panic.is_err());
        assert!(
            Arc::ptr_eq(&layered.base_store_arc(), &original_base),
            "finalizer unwind must leave the exact prior compact generation installed"
        );
        assert!(receipt_belongs_to_current_overlay(&layered, &receipt));
        assert_eq!(
            layered
                .classify_transport_extract_edges(&[&receipt], EpochId::new(1), EpochId::new(2),),
            [TransportEdgeState::Closed],
            "caught unwind must retain receipt, promoted overlay history, and tombstone"
        );

        assert!(
            purge_transport_for_test(
                &layered,
                &[&receipt],
                |generation| Ok((generation, ())),
                |()| {},
                |()| {},
            )
            .expect("retry exact transport cleanup")
        );
        assert!(!receipt_belongs_to_current_overlay(&layered, &receipt));
        assert!(layered.complete_edge_history(edge).is_empty());
    }

    #[test]
    fn mixed_transport_post_publication_unwind_restores_and_retires_after_guards() {
        struct GuardDrainProbe {
            layered: Arc<LayeredStore>,
            observed_drained: Arc<AtomicBool>,
        }

        impl Drop for GuardDrainProbe {
            fn drop(&mut self) {
                let publication_drained = self.layered.publication_guard.try_write().is_some();
                let mutation_drained = self.layered.merge_guard.try_write().is_some();
                self.observed_drained
                    .store(publication_drained && mutation_drained, Ordering::SeqCst);
            }
        }

        let source = Arc::new(LpgStore::new().expect("mixed transport source"));
        let src = source.create_node(&["Source"]);
        let dst = source.create_node(&["Destination"]);
        source.sync_epoch(EpochId::new(1));
        let resident = source
            .create_transport_edge_with_id(EdgeId::new(8_930_001), src, dst, "RESIDENT")
            .expect("resident allocation")
            .expect("resident receipt");
        let base_only = source
            .create_transport_edge_with_id(EdgeId::new(8_930_002), src, dst, "BASE_ONLY")
            .expect("base-only allocation")
            .expect("base-only receipt");
        let layered = Arc::new(compact_transport_source(source));
        layered.overlay_store().sync_epoch(EpochId::new(2));
        layered.set_edge_property(resident.edge_id(), "promoted", Value::Bool(true));
        assert!(layered.delete_edge(resident.edge_id()));
        assert!(layered.delete_edge(base_only.edge_id()));
        assert!(
            layered
                .overlay_store()
                .all_known_edge_ids()
                .contains(&resident.edge_id())
        );
        assert!(
            !layered
                .overlay_store()
                .all_known_edge_ids()
                .contains(&base_only.edge_id())
        );

        let original_base = layered.base_store_arc();
        let original_overlay = layered.overlay_store();
        let owner = WriteAuthority::new();
        assert!(original_overlay.seal_unframed_writes(&owner));
        original_overlay.panic_after_transport_publication_once_for_test();
        let rollback_called = Arc::new(AtomicBool::new(false));
        let retirement_drained = Arc::new(AtomicBool::new(false));
        let rollback_witness = Arc::clone(&rollback_called);
        let probe_layered = Arc::clone(&layered);
        let probe_drained = Arc::clone(&retirement_drained);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<bool, String> = with_authority(&owner, || {
                layered.purge_transport_extract_edges(
                    &[&resident, &base_only],
                    &owner,
                    |generation| Ok((generation, ())),
                    |()| (),
                    move |()| {
                        rollback_witness.store(true, Ordering::SeqCst);
                        GuardDrainProbe {
                            layered: probe_layered,
                            observed_drained: probe_drained,
                        }
                    },
                )
            });
        }));
        assert!(panic.is_err());
        assert!(rollback_called.load(Ordering::SeqCst));
        assert!(
            retirement_drained.load(Ordering::SeqCst),
            "rollback retirement must run only after Layered publication and mutation guards drain"
        );
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original_base));
        assert!(Arc::ptr_eq(&layered.overlay_store(), &original_overlay));
        assert!(receipt_belongs_to_current_overlay(&layered, &resident));
        assert!(receipt_belongs_to_current_overlay(&layered, &base_only));
        assert_eq!(
            layered.classify_transport_extract_edges(
                &[&resident, &base_only],
                EpochId::new(1),
                EpochId::new(2),
            ),
            [TransportEdgeState::Closed, TransportEdgeState::Closed]
        );

        assert!(
            with_authority(&owner, || layered.purge_transport_extract_edges(
                &[&resident, &base_only],
                &owner,
                |generation| Ok((generation, ())),
                |()| (),
                |()| (),
            ))
            .expect("mixed transport purge retry")
        );
        assert!(!receipt_belongs_to_current_overlay(&layered, &resident));
        assert!(!receipt_belongs_to_current_overlay(&layered, &base_only));
        assert!(layered.complete_edge_history(resident.edge_id()).is_empty());
        assert!(
            layered
                .complete_edge_history(base_only.edge_id())
                .is_empty()
        );
    }

    #[test]
    fn transport_bookkeeping_never_waits_for_dirty_edges_while_holding_tombstones() {
        let layered = Arc::new(build_test_layered());
        let edge = EdgeId::new(604);
        layered.deleted_from_base_edges.write().insert(
            edge,
            BaseEdgeDelete {
                epoch: EpochId::new(1),
                deleter: None,
            },
        );
        layered.dirty_edge_ids.write().insert(edge);

        let dirty_reader_ready = Arc::new(std::sync::Barrier::new(2));
        let between_lock_scopes = Arc::new(std::sync::Barrier::new(2));
        let tombstone_probe_complete = Arc::new(std::sync::Barrier::new(2));
        let tombstones_were_unlocked = Arc::new(AtomicBool::new(false));

        let observer = {
            let layered = Arc::clone(&layered);
            let dirty_reader_ready = Arc::clone(&dirty_reader_ready);
            let between_lock_scopes = Arc::clone(&between_lock_scopes);
            let tombstone_probe_complete = Arc::clone(&tombstone_probe_complete);
            let tombstones_were_unlocked = Arc::clone(&tombstones_were_unlocked);
            std::thread::spawn(move || {
                let dirty = layered.dirty_edge_ids.read();
                dirty_reader_ready.wait();
                between_lock_scopes.wait();
                tombstones_were_unlocked.store(
                    layered.deleted_from_base_edges.try_read().is_some(),
                    Ordering::SeqCst,
                );
                tombstone_probe_complete.wait();
                drop(dirty);
            })
        };

        let hook_between_lock_scopes = Arc::clone(&between_lock_scopes);
        let hook_probe_complete = Arc::clone(&tombstone_probe_complete);
        *layered.transport_cleanup_between_lock_scopes_hook.write() = Some(Arc::new(move || {
            hook_between_lock_scopes.wait();
            hook_probe_complete.wait();
        }));

        dirty_reader_ready.wait();
        let targets = FxHashSet::from_iter([edge]);
        let _generation = layered.merge_guard.write();
        layered.clear_purged_transport_bookkeeping(&targets);
        observer.join().expect("bookkeeping observer");

        assert!(
            tombstones_were_unlocked.load(Ordering::SeqCst),
            "transport cleanup must release tombstones before waiting on a dirty-edge writer"
        );
        assert!(!layered.deleted_from_base_edges.read().contains_key(&edge));
        assert!(!layered.dirty_edge_ids.read().contains(&edge));
    }

    #[test]
    fn layered_transport_high_degree_batch_scans_compact_inventory_once_per_phase() {
        const EDGE_COUNT: usize = 1_024;

        let source = Arc::new(LpgStore::new().expect("source store"));
        let src = source.create_node(&["Source"]);
        let dst = source.create_node(&["Destination"]);
        source.sync_epoch(EpochId::new(1));
        let mut receipts = Vec::with_capacity(EDGE_COUNT);
        for offset in 0..EDGE_COUNT {
            let edge = EdgeId::new(20_000 + u64::try_from(offset).unwrap());
            receipts.push(
                source
                    .create_transport_edge_with_id(edge, src, dst, "CARRIED")
                    .expect("transport allocation")
                    .expect("fresh transport identity"),
            );
        }

        let layered = compact_transport_source(source);
        layered.overlay.load().sync_epoch(EpochId::new(2));
        for receipt in &receipts {
            assert!(layered.delete_edge(receipt.edge_id()));
        }

        let mut hostile_refs: Vec<_> = receipts.iter().collect();
        hostile_refs.push(&receipts[0]);
        layered
            .base_edge_inventory_scans
            .store(0, Ordering::Relaxed);
        let states = layered.classify_transport_extract_edges(
            &hostile_refs,
            EpochId::new(1),
            EpochId::new(2),
        );
        assert_eq!(states[0], TransportEdgeState::Invalid);
        assert_eq!(states[EDGE_COUNT], TransportEdgeState::Invalid);
        assert!(
            states[1..EDGE_COUNT]
                .iter()
                .all(|state| *state == TransportEdgeState::Closed)
        );
        assert_eq!(
            layered.base_edge_inventory_scans.load(Ordering::Relaxed),
            1,
            "classification must inventory the compact base once, not once per receipt"
        );

        let refs: Vec<_> = receipts.iter().collect();
        layered
            .base_edge_inventory_scans
            .store(0, Ordering::Relaxed);
        assert!(
            purge_transport_for_test(
                &layered,
                &refs,
                |generation| Ok((generation, ())),
                |()| {},
                |()| {},
            )
            .expect("purge compact carried star")
        );
        assert_eq!(
            layered.base_edge_inventory_scans.load(Ordering::Relaxed),
            1,
            "physical generation rebuild must reuse one compact lifetime inventory"
        );
        assert_eq!(layered.edge_count(), 0);
    }

    struct PromotionPause {
        reached: mpsc::Receiver<()>,
        release: Option<mpsc::Sender<()>>,
    }

    impl PromotionPause {
        fn wait_until_reached(&self) {
            self.reached
                .recv_timeout(Duration::from_secs(10))
                .expect("promotion hook must be reached within ten seconds");
        }

        fn release(&mut self) {
            if let Some(release) = self.release.take() {
                release
                    .send(())
                    .expect("promotion thread must still be waiting for release");
            }
        }
    }

    impl Drop for PromotionPause {
        fn drop(&mut self) {
            if let Some(release) = self.release.take() {
                let _ = release.send(());
            }
        }
    }

    fn promotion_pause() -> (Arc<PromotionPublicationHook>, PromotionPause) {
        let (reached_tx, reached_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let hook: Arc<PromotionPublicationHook> = Arc::new(move || {
            reached_tx
                .send(())
                .expect("publication test must still be waiting for the hook");
            release_rx
                .lock()
                .recv_timeout(Duration::from_secs(30))
                .expect("publication test must release the promotion thread");
        });
        (
            hook,
            PromotionPause {
                reached: reached_rx,
                release: Some(release_tx),
            },
        )
    }

    #[derive(Default)]
    struct PredicateWriteSpy {
        full_node_writes: AtomicUsize,
        full_edge_writes: AtomicUsize,
        dataset_writes: AtomicUsize,
        label_name_writes: Mutex<Vec<String>>,
        rel_type_name_writes: Mutex<Vec<String>>,
    }

    impl WriteTracker for PredicateWriteSpy {
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
            _edge_id: EdgeId,
        ) -> Result<(), OperatorError> {
            Ok(())
        }

        fn record_node_write_with_labels(
            &self,
            _transaction_id: TransactionId,
            _node_id: NodeId,
            _labels: &[grafeo_common::types::LabelId],
        ) -> Result<(), OperatorError> {
            self.full_node_writes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn record_edge_write_with_type(
            &self,
            _transaction_id: TransactionId,
            _edge_id: EdgeId,
            _rel_type: grafeo_common::types::EdgeTypeId,
        ) -> Result<(), OperatorError> {
            self.full_edge_writes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn record_lpg_dataset_write(
            &self,
            _transaction_id: TransactionId,
        ) -> Result<(), OperatorError> {
            self.dataset_writes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn record_label_name_predicate_write(
            &self,
            _transaction_id: TransactionId,
            label: &str,
        ) -> Result<(), OperatorError> {
            self.label_name_writes.lock().push(label.to_owned());
            Ok(())
        }

        fn record_rel_type_name_predicate_write(
            &self,
            _transaction_id: TransactionId,
            rel_type: &str,
        ) -> Result<(), OperatorError> {
            self.rel_type_name_writes.lock().push(rel_type.to_owned());
            Ok(())
        }
    }

    fn build_test_layered() -> LayeredStore {
        let store = LpgStore::new().unwrap();

        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        let e1 = store.create_edge(alix, amsterdam, "LIVES_IN");
        store.set_edge_property(e1, "since", Value::Int64(2020));

        let e2 = store.create_edge(gus, amsterdam, "LIVES_IN");
        store.set_edge_property(e2, "since", Value::Int64(2022));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let max_nid = store
            .node_ids()
            .into_iter()
            .map(|id| id.as_u64())
            .max()
            .unwrap_or(0);
        let max_eid = 10u64; // edges start at 0 in LpgStore
        LayeredStore::new(compact, max_nid, max_eid).unwrap()
    }

    #[test]
    fn compact_edge_pending_owner_set_remove_preserves_visibility_and_authority() {
        let layered = build_test_layered();
        let overlay = layered.overlay_store();
        let authority = WriteAuthority::new();
        let denied = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&authority));
        let owner = TransactionId::new(42);
        let foreign = TransactionId::new(43);
        let epoch = overlay.current_epoch();
        let edge = with_authority(&authority, || {
            layered.create_edge_versioned(NodeId::new(0), NodeId::new(2), "PENDING", epoch, owner)
        });
        assert!(edge.is_valid());
        assert!(layered.is_edge_dirty(edge));
        assert!(layered.get_edge_versioned(edge, epoch, owner).is_some());
        assert!(layered.get_edge_versioned(edge, epoch, foreign).is_none());
        assert!(layered.get_edge(edge).is_none());
        let key = PropertyKey::new("pending_value");
        with_authority(&denied, || {
            layered.set_edge_property_buffered(edge, key.as_str(), Value::Int64(99), owner);
        });
        assert_eq!(
            overlay.read_edge_property_visible(edge, &key, epoch, Some(owner)),
            None
        );
        with_authority(&authority, || {
            layered.set_edge_property_buffered(edge, key.as_str(), Value::Int64(7), owner);
            assert_eq!(
                layered.read_edge_property_visible(edge, &key, epoch, Some(owner)),
                Some(Value::Int64(7))
            );
            layered.set_edge_property_buffered(edge, key.as_str(), Value::Int64(99), foreign);
            layered.remove_edge_property_buffered(edge, key.as_str(), foreign);
            assert_eq!(
                layered.read_edge_property_visible(edge, &key, epoch, Some(owner)),
                Some(Value::Int64(7))
            );
            assert_eq!(
                layered.read_edge_property_visible(edge, &key, epoch, Some(foreign)),
                None
            );
            assert_eq!(
                layered.read_edge_property_visible(edge, &key, epoch, None),
                None
            );
            layered.remove_edge_property_buffered(edge, key.as_str(), owner);
            assert_eq!(
                layered.read_edge_property_visible(edge, &key, epoch, Some(owner)),
                None
            );
        });
    }

    #[test]
    fn compact_edge_pending_raw_overlay_node_admits_only_its_owner() {
        let layered = build_test_layered();
        let source = layered.create_node(&["HotSource"]);
        let overlay = layered.overlay_store();
        let epoch = overlay.current_epoch();
        let owner = TransactionId::new(42);
        let foreign = TransactionId::new(43);
        let pending = overlay.create_node_versioned(&["RawPending"], epoch, owner);
        assert!(pending.is_valid());
        assert!(!layered.is_node_dirty(pending));
        assert!(!LayeredStore::base_has_node_identity(
            &layered.base.load(),
            pending
        ));
        assert!(layered.get_node_versioned(pending, epoch, owner).is_some());
        assert!(
            layered
                .get_node_versioned(pending, epoch, foreign)
                .is_none()
        );
        assert!(
            !layered
                .create_edge_versioned(source, pending, "FOREIGN", epoch, foreign)
                .is_valid()
        );
        let edge = layered.create_edge_versioned(source, pending, "OWN", epoch, owner);
        assert!(
            edge.is_valid(),
            "unmarked overlay-only owner endpoint must be admitted"
        );
        assert!(
            !layered.is_node_dirty(pending),
            "admission must not fabricate a promotion marker"
        );
        let closed = overlay.create_node(&["Closed"]);
        assert!(overlay.delete_node(closed));
        assert!(
            !layered
                .create_edge_versioned(source, closed, "CLOSED", epoch, owner)
                .is_valid()
        );
    }

    #[test]
    fn compact_edge_pending_node_tombstone_rejects_before_promotion() {
        let layered = build_test_layered();
        let node = NodeId::new(0);
        let tx = TransactionId::new(42);
        let epoch = layered.current_epoch();
        assert!(layered.delete_node_versioned(node, epoch, tx));
        layered.set_node_property_buffered(node, "after_delete", Value::Bool(true), tx);
        assert!(!layered.is_node_dirty(node));
        assert!(
            layered
                .overlay_store()
                .overlay_touched_entities(tx)
                .0
                .is_empty()
        );
        assert!(!layered.is_node_visible_versioned(node, epoch, tx));
        assert!(layered.is_node_visible_versioned(node, epoch, TransactionId::new(43)));
        let foreign = TransactionId::new(43);
        layered.set_node_property_buffered(node, "foreign", Value::Int64(7), foreign);
        assert_eq!(
            layered.read_node_property_visible(
                node,
                &PropertyKey::new("foreign"),
                epoch,
                Some(foreign)
            ),
            Some(Value::Int64(7))
        );
        assert_eq!(
            layered
                .deleted_from_base_nodes
                .read()
                .get(&node)
                .map(|mark| (mark.epoch, mark.deleter)),
            Some((EpochId::PENDING, Some(tx)))
        );
    }

    #[test]
    fn compact_edge_pending_edge_tombstone_rejects_before_promotion() {
        let layered = build_test_layered();
        let edge = EdgeId::new(0);
        let tx = TransactionId::new(42);
        let epoch = layered.current_epoch();
        assert!(layered.delete_edge_versioned(edge, epoch, tx));
        layered.set_edge_property_buffered(edge, "after_delete", Value::Bool(true), tx);
        assert!(!layered.is_edge_dirty(edge));
        assert!(
            layered
                .overlay_store()
                .overlay_touched_entities(tx)
                .1
                .is_empty()
        );
        assert!(!layered.is_edge_visible_versioned(edge, epoch, tx));
        assert!(layered.is_edge_visible_versioned(edge, epoch, TransactionId::new(43)));
        let foreign = TransactionId::new(43);
        layered.set_edge_property_buffered(edge, "foreign", Value::Int64(7), foreign);
        assert_eq!(
            layered.read_edge_property_visible(
                edge,
                &PropertyKey::new("foreign"),
                epoch,
                Some(foreign)
            ),
            Some(Value::Int64(7))
        );
        assert_eq!(
            layered
                .deleted_from_base_edges
                .read()
                .get(&edge)
                .map(|mark| (mark.epoch, mark.deleter)),
            Some((EpochId::PENDING, Some(tx)))
        );
    }

    #[test]
    fn compact_edge_pending_closed_base_identity_owns_overlapping_physical_rows() {
        let layered = empty_layered();
        let source = layered.overlay_store();
        source.set_epoch(EpochId::new(1));
        let a = source.create_node(&["Closed"]);
        let b = source.create_node(&["Other"]);
        let edge = source.create_edge(a, b, "CLOSED");
        source.set_epoch(EpochId::new(2));
        assert!(source.delete_edge(edge));
        assert!(source.delete_node(a));
        layered
            .merge_overlay_temporal()
            .expect("retain closed cold identities");
        assert!(LayeredStore::base_has_node_identity(
            &layered.base.load(),
            a
        ));
        assert!(LayeredStore::base_has_edge_identity(
            &layered.base.load(),
            edge
        ));
        let overlay = layered.overlay_store();
        // A physical overlap is not logical authority. Check that these real
        // recovery primitives actually inserted rows, rather than accepting a no-op.
        overlay
            .create_node_with_id(a, &["PrivateOverlap"])
            .expect("physical node");
        overlay
            .create_node_with_id(b, &["PrivateOverlap"])
            .expect("physical endpoint");
        overlay
            .create_edge_with_id(edge, a, b, "PRIVATE")
            .expect("physical edge");
        assert!(overlay.get_node(a).is_some());
        assert!(overlay.get_edge(edge).is_some());
        assert!(!layered.is_node_dirty(a));
        assert!(!layered.is_edge_dirty(edge));
        let tx = TransactionId::new(42);
        layered.set_node_property_buffered(a, "forbidden", Value::Bool(true), tx);
        layered.set_edge_property_buffered(edge, "forbidden", Value::Bool(true), tx);
        assert!(overlay.overlay_touched_entities(tx).0.is_empty());
        assert!(overlay.overlay_touched_entities(tx).1.is_empty());
        assert!(layered.get_node(a).is_none());
        assert!(layered.get_edge(edge).is_none());
    }

    fn assert_compact_pending_waits_for_private_promotion(edge: bool) {
        let layered = Arc::new(build_test_layered());
        let (hook, mut pause) = promotion_pause();
        if edge {
            *layered.edge_publication_hook.write() = Some(hook);
        } else {
            *layered.node_publication_hook.write() = Some(hook);
        }
        let promoter = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                if edge {
                    layered.set_edge_property(EdgeId::new(0), "trigger", Value::Bool(true));
                } else {
                    layered.set_node_property(NodeId::new(0), "trigger", Value::Bool(true));
                }
            })
        };
        pause.wait_until_reached();
        let overlay = layered.overlay_store();
        if edge {
            assert!(overlay.get_edge(EdgeId::new(0)).is_some());
            assert!(!layered.is_edge_dirty(EdgeId::new(0)));
        } else {
            assert!(overlay.get_node(NodeId::new(0)).is_some());
            assert!(!layered.is_node_dirty(NodeId::new(0)));
        }
        let tx = TransactionId::new(42);
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let pending_writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                started_tx.send(()).expect("started");
                if edge {
                    layered.set_edge_property_buffered(
                        EdgeId::new(0),
                        "pending",
                        Value::Int64(7),
                        tx,
                    );
                } else {
                    layered.set_node_property_buffered(
                        NodeId::new(0),
                        "pending",
                        Value::Int64(7),
                        tx,
                    );
                }
                done_tx.send(()).expect("completed");
            })
        };
        started_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("writer starts");
        let completed_early = done_rx.recv_timeout(Duration::from_millis(100)).is_ok();
        let touched_during_hydration = overlay.overlay_touched_entities(tx);
        pause.release();
        promoter.join().expect("promoter");
        pending_writer.join().expect("pending writer");
        assert!(
            !completed_early,
            "physical row must not bypass unpublished promotion"
        );
        assert!(touched_during_hydration.0.is_empty() && touched_during_hydration.1.is_empty());
        let key = PropertyKey::new("pending");
        let value = if edge {
            layered.read_edge_property_visible(
                EdgeId::new(0),
                &key,
                layered.current_epoch(),
                Some(tx),
            )
        } else {
            layered.read_node_property_visible(
                NodeId::new(0),
                &key,
                layered.current_epoch(),
                Some(tx),
            )
        };
        assert_eq!(value, Some(Value::Int64(7)));
    }

    #[test]
    fn compact_edge_pending_node_mutation_waits_for_private_hydration() {
        assert_compact_pending_waits_for_private_promotion(false);
    }

    #[test]
    fn compact_edge_pending_edge_mutation_waits_for_private_hydration() {
        assert_compact_pending_waits_for_private_promotion(true);
    }

    #[test]
    fn sealed_layered_composite_mutators_fail_closed_without_routing_side_effects() {
        let layered = build_test_layered();
        let source = layered.nodes_by_label("Person")[0];
        let (destination, edge) = layered
            .edges_from(source, Direction::Outgoing)
            .into_iter()
            .next()
            .expect("fixture person has an outgoing base edge");
        let overlay = layered.overlay_store();
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));

        let before_node_count = layered.node_count();
        let before_edge_count = layered.edge_count();
        let before_next_node = overlay.next_node_id();
        let before_next_edge = overlay.next_edge_id();
        let before_deletions = (
            layered.snapshot_deleted_node_ids(),
            layered.snapshot_deleted_edge_ids(),
        );
        let blocked_transaction = TransactionId::new(43);

        with_authority(&foreign, || {
            assert_eq!(layered.create_node(&["Blocked"]), NodeId::INVALID);
            assert_eq!(
                layered.create_node_versioned(
                    &["Blocked"],
                    EpochId::new(3),
                    TransactionId::new(41),
                ),
                NodeId::INVALID
            );
            assert_eq!(
                layered.create_edge(source, destination, "BLOCKED"),
                EdgeId::INVALID
            );
            assert_eq!(
                layered.create_edge_versioned(
                    source,
                    destination,
                    "BLOCKED",
                    EpochId::new(3),
                    TransactionId::new(42),
                ),
                EdgeId::INVALID
            );
            assert!(
                layered
                    .batch_create_edges(&[(source, destination, "BLOCKED")])
                    .is_empty()
            );
            layered.set_node_property(source, "blocked", Value::Bool(true));
            layered.set_edge_property(edge, "blocked", Value::Bool(true));
            layered.set_node_property_versioned(
                source,
                "blocked_versioned",
                Value::Bool(true),
                blocked_transaction,
            );
            layered.set_edge_property_buffered(
                edge,
                "blocked_buffered",
                Value::Bool(true),
                blocked_transaction,
            );
            layered.delete_node_edges(source);
            assert!(!layered.delete_edge(edge));
            assert!(!layered.delete_node(source));
            assert!(!layered.delete_edge_versioned(edge, EpochId::new(3), blocked_transaction,));
            assert!(!layered.delete_node_versioned(source, EpochId::new(3), blocked_transaction,));
        });

        assert_eq!(layered.node_count(), before_node_count);
        assert_eq!(layered.edge_count(), before_edge_count);
        assert_eq!(overlay.next_node_id(), before_next_node);
        assert_eq!(overlay.next_edge_id(), before_next_edge);
        assert_eq!(layered.overlay_mutation_count(), 0);
        assert!(!layered.dirty_node_ids.read().contains(&NodeId::INVALID));
        assert!(!layered.dirty_edge_ids.read().contains(&EdgeId::INVALID));
        assert_eq!(
            (
                layered.snapshot_deleted_node_ids(),
                layered.snapshot_deleted_edge_ids(),
            ),
            before_deletions
        );
        assert_eq!(
            layered.get_node_property(source, &PropertyKey::new("blocked")),
            None
        );
        assert_eq!(
            layered.get_edge_property(edge, &PropertyKey::new("blocked")),
            None
        );
        assert!(layered.get_node(source).is_some());
        assert!(layered.get_edge(edge).is_some());
        let blocked_structure = layered.tx_structural_snapshot(blocked_transaction);
        assert!(blocked_structure.node_creates.is_empty());
        assert!(blocked_structure.edge_creates.is_empty());
        assert!(blocked_structure.node_deletes.is_empty());
        assert!(blocked_structure.edge_deletes.is_empty());
        assert!(blocked_structure.base_node_deletes.is_empty());
        assert!(blocked_structure.base_edge_deletes.is_empty());
        assert_eq!(
            layered.read_node_property_visible(
                source,
                &PropertyKey::new("blocked_versioned"),
                overlay.current_epoch(),
                Some(blocked_transaction),
            ),
            None
        );
        assert_eq!(
            layered.read_edge_property_visible(
                edge,
                &PropertyKey::new("blocked_buffered"),
                overlay.current_epoch(),
                Some(blocked_transaction),
            ),
            None
        );

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || panic!("hostile owner callback"));
        }));
        assert!(unwind.is_err());
        assert_eq!(layered.create_node(&["StillBlocked"]), NodeId::INVALID);
        assert!(!layered.delete_edge(edge));
        layered.set_node_property(source, "still_blocked", Value::Bool(true));
        assert_eq!(
            layered.get_node_property(source, &PropertyKey::new("still_blocked")),
            None
        );
        assert_eq!(layered.overlay_mutation_count(), 0);

        with_authority(&owner, || {
            layered.set_node_property(source, "allowed", Value::Bool(true));
            layered.set_edge_property(edge, "allowed", Value::Bool(true));
            let created = layered.create_node(&["Allowed"]);
            assert!(created.is_valid());
            let edges = layered.batch_create_edges(&[(source, destination, "ALLOWED")]);
            assert_eq!(edges.len(), 1);
            assert!(edges[0].is_valid());
        });
        assert_eq!(
            layered.get_node_property(source, &PropertyKey::new("allowed")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            layered.get_edge_property(edge, &PropertyKey::new("allowed")),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn compact_exact_recovery_is_authorized_idempotent_and_conflict_strict() {
        let layered = build_test_layered();
        let overlay = layered.overlay_store();
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));

        let source = layered.nodes_by_label("Person")[0];
        let destination = layered.nodes_by_label("City")[0];
        let missing = NodeId::new(7_700_000);
        let exact_node = NodeId::new(7_700_001);
        let missing_edge = EdgeId::new(7_700_002);
        let exact_edge = EdgeId::new(7_700_003);
        let before_next_node = overlay.next_node_id();
        let before_next_edge = overlay.next_edge_id();

        let denied = layered
            .recover_create_node_with_id(exact_node, &["Recovered"])
            .expect_err("raw replay must not cross the overlay seal");
        assert!(matches!(
            denied,
            grafeo_common::Error::Storage(StorageError::Corruption(_))
        ));
        let denied = with_authority(&foreign, || {
            layered.recover_create_node_with_id(exact_node, &["Recovered"])
        })
        .expect_err("foreign authority must not cross the overlay seal");
        assert!(matches!(
            denied,
            grafeo_common::Error::Storage(StorageError::Corruption(_))
        ));
        assert_eq!(overlay.next_node_id(), before_next_node);
        assert_eq!(overlay.next_edge_id(), before_next_edge);
        assert!(!layered.dirty_node_ids.read().contains(&exact_node));

        with_authority(&owner, || {
            let missing_error = layered
                .recover_create_edge_with_id(missing_edge, source, missing, "MISSING")
                .expect_err("missing endpoint must fail before base hydration");
            assert!(matches!(
                missing_error,
                grafeo_common::Error::Storage(StorageError::Corruption(_))
            ));
            assert!(
                !layered.dirty_node_ids.read().contains(&source),
                "endpoint validation must precede representation promotion"
            );
            assert!(!layered.dirty_edge_ids.read().contains(&missing_edge));

            layered
                .recover_create_node_with_id(exact_node, &["Recovered", "Recovered"])
                .expect("exact node replay");
            layered
                .recover_create_node_with_id(exact_node, &["Recovered"])
                .expect("same logical label set is idempotent");
            layered
                .recover_create_node_with_id(destination, &["City"])
                .expect("matching compact-base identity is idempotent");
            assert!(
                layered
                    .recover_create_node_with_id(exact_node, &["Conflicting"])
                    .is_err(),
                "an occupied node identity with different creation labels must fail closed"
            );

            layered
                .recover_create_edge_with_id(exact_edge, source, destination, "RECOVERED_EDGE")
                .expect("exact base-to-base edge replay");
            layered
                .recover_create_edge_with_id(exact_edge, source, destination, "RECOVERED_EDGE")
                .expect("exact edge replay is idempotent");
            assert!(
                layered
                    .recover_create_edge_with_id(exact_edge, source, destination, "CONFLICT",)
                    .is_err(),
                "an occupied edge identity with a different type must fail closed"
            );

            let generated_node = layered.create_node(&["Generated"]);
            let generated_edge = layered.create_edge(source, destination, "GENERATED_EDGE");
            assert!(generated_node.as_u64() > exact_node.as_u64());
            assert!(generated_edge.as_u64() > exact_edge.as_u64());

            assert!(layered.delete_edge(exact_edge));
            assert!(
                layered
                    .recover_create_edge_with_id(exact_edge, source, destination, "RECOVERED_EDGE",)
                    .is_err(),
                "a deleted stable edge identity must never be reused"
            );
            assert!(layered.delete_node(exact_node));
            assert!(
                layered
                    .recover_create_node_with_id(exact_node, &["Recovered"])
                    .is_err(),
                "a deleted stable node identity must never be reused"
            );
        });
    }

    #[test]
    fn compact_exact_node_recovery_rejects_seeded_absent_tombstone_without_side_effects() {
        let layered = build_test_layered();
        let tombstoned = NodeId::new(7_710_001);
        layered.seed_deleted_from_base([tombstoned], std::iter::empty::<EdgeId>());
        let overlay = layered.overlay_store();
        let owner = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));
        let before = (
            overlay.node_count(),
            overlay.next_node_id(),
            overlay.all_labels(),
        );

        let error = with_authority(&owner, || {
            layered.recover_create_node_with_id(tombstoned, &["MustNotExist"])
        })
        .expect_err("an absent identity remains reserved by its persisted tombstone");
        assert!(matches!(
            error,
            grafeo_common::Error::Storage(StorageError::Corruption(_))
        ));
        assert_eq!(
            (
                overlay.node_count(),
                overlay.next_node_id(),
                overlay.all_labels(),
            ),
            before,
            "rejected replay must not touch raw LPG rows, allocators, or label metadata"
        );
        assert!(overlay.get_node(tombstoned).is_none());
        assert!(!layered.dirty_node_ids.read().contains(&tombstoned));
        assert!(
            layered.snapshot_deleted_node_ids().contains(&tombstoned),
            "the rejecting tombstone must remain installed"
        );
    }

    #[test]
    fn compact_exact_edge_recovery_rejects_seeded_absent_tombstone_without_side_effects() {
        let layered = build_test_layered();
        let source = layered.nodes_by_label("Person")[0];
        let destination = layered.nodes_by_label("City")[0];
        let tombstoned = EdgeId::new(7_720_001);
        layered.seed_deleted_from_base(std::iter::empty::<NodeId>(), [tombstoned]);
        let overlay = layered.overlay_store();
        let owner = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));
        let before = (
            overlay.node_count(),
            overlay.edge_count(),
            overlay.next_edge_id(),
            overlay.all_edge_types(),
        );

        let error = with_authority(&owner, || {
            layered.recover_create_edge_with_id(tombstoned, source, destination, "MUST_NOT_EXIST")
        })
        .expect_err("an absent edge identity remains reserved by its persisted tombstone");
        assert!(matches!(
            error,
            grafeo_common::Error::Storage(StorageError::Corruption(_))
        ));
        assert_eq!(
            (
                overlay.node_count(),
                overlay.edge_count(),
                overlay.next_edge_id(),
                overlay.all_edge_types(),
            ),
            before,
            "rejected replay must not touch raw LPG rows, allocators, or edge-type metadata"
        );
        assert!(overlay.get_edge(tombstoned).is_none());
        assert!(!layered.dirty_edge_ids.read().contains(&tombstoned));
        assert!(!layered.dirty_node_ids.read().contains(&source));
        assert!(!layered.dirty_node_ids.read().contains(&destination));
        assert!(
            layered.snapshot_deleted_edge_ids().contains(&tombstoned),
            "the rejecting tombstone must remain installed"
        );
    }

    #[test]
    fn test_read_through_base() {
        let layered = build_test_layered();
        assert_eq!(layered.node_count(), 3);
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);
    }

    #[test]
    fn sealed_layered_structural_restore_requires_owning_authority() {
        use crate::graph::write_permit::{WriteAuthority, with_authority};

        let layered = build_test_layered();
        let owner = WriteAuthority::new();
        assert!(layered.overlay_store().seal_unframed_writes(&owner));
        let transaction_id = TransactionId::new(701);
        let before = layered.tx_structural_snapshot(transaction_id);
        let created = with_authority(&owner, || {
            layered.create_node_versioned(&["Pending"], EpochId::INITIAL, transaction_id)
        });
        assert!(created.is_valid());
        assert_eq!(
            layered.tx_structural_snapshot(transaction_id).node_creates,
            vec![created]
        );

        let foreign = WriteAuthority::new();
        let error = with_authority(&foreign, || {
            layered
                .tx_structural_restore(transaction_id, before.clone())
                .unwrap_err()
        });
        assert!(error.contains("write authority"), "{error}");
        assert_eq!(
            layered.tx_structural_snapshot(transaction_id).node_creates,
            vec![created],
            "foreign restore must leave the pending structure untouched"
        );

        with_authority(&owner, || {
            layered
                .tx_structural_restore(transaction_id, before)
                .expect("owner authority restores the savepoint");
        });
        assert!(
            layered
                .tx_structural_snapshot(transaction_id)
                .node_creates
                .is_empty()
        );
    }

    #[test]
    fn structural_restore_excludes_concurrent_overlay_generation_reset() {
        use crate::graph::write_permit::{WriteAuthority, with_authority};

        let layered = Arc::new(build_test_layered());
        let owner = Arc::new(WriteAuthority::new());
        assert!(layered.overlay_store().seal_unframed_writes(&owner));
        let transaction_id = TransactionId::new(702);
        let before = layered.tx_structural_snapshot(transaction_id);
        with_authority(&owner, || {
            let created =
                layered.create_node_versioned(&["Pending"], EpochId::INITIAL, transaction_id);
            assert!(created.is_valid());
        });

        let (hook, mut pause) = promotion_pause();
        *layered.structural_restore_hook.write() = Some(hook);
        let restoring = Arc::clone(&layered);
        let restore_owner = Arc::clone(&owner);
        let restore = std::thread::spawn(move || {
            with_authority(&restore_owner, || {
                restoring.tx_structural_restore(transaction_id, before)
            })
        });
        pause.wait_until_reached();

        let resetting = Arc::clone(&layered);
        let reset_owner = Arc::clone(&owner);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let reset = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            with_authority(&reset_owner, || resetting.reset_overlay());
            finished_tx.send(()).unwrap();
        });
        started_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("reset contender starts");
        let completed_while_restore_was_frozen =
            finished_rx.recv_timeout(Duration::from_millis(250)).ok();

        pause.release();
        restore
            .join()
            .expect("restore thread does not panic")
            .expect("owner restore succeeds");
        reset.join().expect("reset thread does not panic");
        assert!(
            completed_while_restore_was_frozen.is_none(),
            "overlay generation swap must block for the entire structural restore"
        );
    }

    /// SP2 slice 4: the temporal merge rebuilds a temporal base from a mix of a
    /// base-only node, a promoted+modified base node, and a fresh overlay node,
    /// preserving current reads, node/edge counts, and original ids.
    #[test]
    fn merge_overlay_temporal_preserves_reads_and_structure() {
        let layered = build_test_layered(); // base: Alix(30), Gus(25), Amsterdam + 2 edges
        let name = PropertyKey::new("name");
        let age = PropertyKey::new("age");
        let persons = layered.nodes_by_label("Person");
        let alix = *persons
            .iter()
            .find(|&&p| layered.get_node_property(p, &name) == Some(Value::from("Alix")))
            .unwrap();
        let gus = *persons
            .iter()
            .find(|&&p| layered.get_node_property(p, &name) == Some(Value::from("Gus")))
            .unwrap();

        // Promote+modify a base node, and create a fresh overlay node.
        layered.set_node_property(alix, "age", Value::Int64(31));
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        layered.set_node_property(vincent, "age", Value::Int64(40));

        layered.merge_overlay_temporal().unwrap();

        // Current reads preserved across the temporal merge.
        assert_eq!(
            layered.get_node_property(alix, &age),
            Some(Value::Int64(31))
        ); // promoted
        assert_eq!(layered.get_node_property(gus, &age), Some(Value::Int64(25))); // base-only
        assert_eq!(
            layered.get_node_property(vincent, &name),
            Some(Value::from("Vincent"))
        ); // new
        assert_eq!(
            layered.get_node_property(vincent, &age),
            Some(Value::Int64(40))
        );
        // Structure preserved.
        assert_eq!(layered.node_count(), 4);
        assert_eq!(layered.edge_count(), 2);
        // The base is now temporal and the overlay was reset: the base alone
        // serves all four nodes.
        assert_eq!(layered.base_store_arc().node_count(), 4);
    }

    #[test]
    fn temporal_merge_external_prepare_error_publishes_nothing() {
        let layered = empty_layered();
        let original_base = layered.base_store_arc();
        let original_overlay = layered.overlay_store();
        let node = original_overlay.create_node(&["Document"]);
        original_overlay.set_node_property(node, "title", Value::from("candidate"));
        let publish_called = AtomicBool::new(false);
        let rollback_called = AtomicBool::new(false);

        let error = layered
            .merge_overlay_temporal_with_publication(
                |_| Err::<(Arc<CompactStore>, ()), _>("injected external prepare error".to_owned()),
                |()| publish_called.store(true, Ordering::SeqCst),
                |()| rollback_called.store(true, Ordering::SeqCst),
            )
            .expect_err("pre-publication failure must be retryable");

        assert_eq!(error, "injected external prepare error");
        assert!(!publish_called.load(Ordering::SeqCst));
        assert!(!rollback_called.load(Ordering::SeqCst));
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original_base));
        assert!(Arc::ptr_eq(&layered.overlay_store(), &original_overlay));
        original_overlay.set_node_property(node, "retryable", Value::Bool(true));
        assert_eq!(
            original_overlay.get_node_property(node, &PropertyKey::new("retryable")),
            Some(Value::Bool(true)),
            "a preparation error must not retire the source representation"
        );
    }

    fn assert_temporal_snapshot_excludes_retained_overlay_writer(sealed: bool) {
        let layered = Arc::new(empty_layered());
        let original_overlay = layered.overlay_store();
        let seed = original_overlay.create_node(&["Seed"]);
        assert!(seed.is_valid());
        let authority = Arc::new(WriteAuthority::new());
        if sealed {
            assert!(original_overlay.seal_unframed_writes(&authority));
        }

        let barrier = Arc::new(Barrier::new(2));
        *layered.temporal_snapshot_barrier.write() = Some(Arc::clone(&barrier));
        let merge_layered = Arc::clone(&layered);
        let merge_authority = Arc::clone(&authority);
        let merger = std::thread::spawn(move || {
            if sealed {
                with_authority(&merge_authority, || merge_layered.merge_overlay_temporal())
            } else {
                merge_layered.merge_overlay_temporal()
            }
        });

        // The merger now owns named-topology, source-LPG, and direct-index
        // exclusion but has not traversed a single source identity.
        barrier.wait();
        let (started_tx, started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let writer_overlay = Arc::clone(&original_overlay);
        let writer_authority = Arc::clone(&authority);
        let writer = std::thread::spawn(move || {
            started_tx.send(()).expect("announce retained writer");
            let id = if sealed {
                with_authority(&writer_authority, || writer_overlay.create_node(&["Racer"]))
            } else {
                writer_overlay.create_node(&["Racer"])
            };
            result_tx.send(id).expect("report retained writer result");
        });
        started_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("retained writer started");
        assert!(
            result_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "a retained overlay writer must wait behind the source snapshot transition"
        );

        barrier.wait();
        merger
            .join()
            .expect("temporal merger thread")
            .expect("temporal merge");
        let raced = result_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("retained writer released after publication");
        writer.join().expect("retained writer thread");
        *layered.temporal_snapshot_barrier.write() = None;

        assert_eq!(
            raced,
            NodeId::INVALID,
            "the displaced representation must retire before the waiting writer resumes"
        );
        assert_eq!(original_overlay.node_count(), 1);
        assert_eq!(layered.node_count(), 1);
        assert!(layered.get_node(seed).is_some());

        let successor = layered.overlay_store();
        let after = if sealed {
            with_authority(&authority, || successor.create_node(&["After"]))
        } else {
            successor.create_node(&["After"])
        };
        assert!(after.is_valid(), "the exact successor remains writable");
    }

    #[test]
    fn temporal_merge_snapshot_excludes_unsealed_retained_overlay_writer() {
        assert_temporal_snapshot_excludes_retained_overlay_writer(false);
    }

    #[test]
    fn temporal_merge_snapshot_excludes_authorized_retained_overlay_writer() {
        assert_temporal_snapshot_excludes_retained_overlay_writer(true);
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn temporal_merge_fences_direct_text_index_writer_before_graph_snapshot() {
        use crate::index::text::{BM25Config, InvertedIndex};

        let layered = Arc::new(empty_layered());
        let original_overlay = layered.overlay_store();
        let caller = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        original_overlay.add_text_index("Document", "body", Arc::clone(&caller));
        let seed = original_overlay.create_node(&["Document"]);
        original_overlay.set_node_property(seed, "body", Value::from("coherent source"));

        let barrier = Arc::new(Barrier::new(2));
        *layered.temporal_snapshot_barrier.write() = Some(Arc::clone(&barrier));
        let merger = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || layered.merge_overlay_temporal())
        };

        // This barrier is reached only after the exact text fork has retained
        // the caller gate. The graph traversal itself has not started.
        barrier.wait();
        let forged = NodeId::new(9_991_001);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            started_tx.send(()).expect("announce direct index writer");
            caller.write().insert(forged, "forged successor posting");
            finished_tx.send(()).expect("report direct index writer");
        });
        started_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("direct index writer started");
        assert!(
            finished_rx
                .recv_timeout(Duration::from_millis(100))
                .is_err(),
            "a retained direct index writer must wait before the graph snapshot begins"
        );

        barrier.wait();
        merger
            .join()
            .expect("temporal merger thread")
            .expect("temporal merge");
        finished_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("direct index writer released after publication");
        writer.join().expect("direct index writer thread");
        *layered.temporal_snapshot_barrier.write() = None;

        assert!(
            original_overlay
                .get_text_index("Document", "body")
                .expect("retired overlay keeps its frozen text index")
                .read()
                .search("forged", 10)
                .is_empty(),
            "the retired physical snapshot must end at the fenced graph cut"
        );
        assert_eq!(
            layered
                .overlay_store()
                .get_text_index("Document", "body")
                .expect("successor owns the logical text index")
                .read()
                .search("forged", 10)
                .first()
                .map(|(id, _)| *id),
            Some(forged),
            "the waiting writer linearizes against the successor after publication"
        );
        assert!(
            layered
                .text_search("Document", "body", "forged", 10)
                .is_empty(),
            "logical Layered visibility still rejects a raw posting with no graph identity"
        );
        assert!(layered.get_node(seed).is_some());
    }

    #[test]
    fn temporal_merge_external_publish_panic_restores_layered_before_unwind() {
        let layered = empty_layered();
        let original_base = layered.base_store_arc();
        let original_overlay = layered.overlay_store();
        let node = original_overlay.create_node(&["Document"]);
        original_overlay.set_node_property(node, "title", Value::from("stable"));
        let rollback_called = AtomicBool::new(false);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<bool, String> = layered.merge_overlay_temporal_with_publication(
                |candidate| Ok((candidate, ())),
                |()| -> bool { panic!("injected external publication panic") },
                |_| rollback_called.store(true, Ordering::SeqCst),
            );
        }));

        assert!(panic.is_err());
        assert!(
            !rollback_called.load(Ordering::SeqCst),
            "a publisher that unwinds before returning a token has no external state to restore"
        );
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original_base));
        assert!(Arc::ptr_eq(&layered.overlay_store(), &original_overlay));
        assert_eq!(
            original_overlay.get_node_property(node, &PropertyKey::new("title")),
            Some(Value::from("stable"))
        );
        original_overlay.set_node_property(node, "after_panic", Value::Bool(true));
        assert_eq!(
            original_overlay.get_node_property(node, &PropertyKey::new("after_panic")),
            Some(Value::Bool(true)),
            "publication unwind must reactivate the exact source"
        );
        layered
            .merge_overlay_temporal()
            .expect("the exact generation remains retryable after caught unwind");
    }

    #[test]
    fn temporal_merge_hostile_post_publish_unwind_rolls_back_after_guards_drain() {
        struct GuardDrainProbe {
            layered: Arc<LayeredStore>,
            observed_drained: Arc<AtomicBool>,
        }

        impl Drop for GuardDrainProbe {
            fn drop(&mut self) {
                let publication_drained = self.layered.publication_guard.try_write().is_some();
                let mutation_drained = self.layered.merge_guard.try_write().is_some();
                self.observed_drained
                    .store(publication_drained && mutation_drained, Ordering::SeqCst);
            }
        }

        let layered = Arc::new(empty_layered());
        let original_base = layered.base_store_arc();
        let original_overlay = layered.overlay_store();
        original_overlay.create_property_index("kind");
        let node = original_overlay.create_node(&["Document"]);
        original_overlay.set_node_property(node, "kind", Value::from("exact"));
        #[cfg(feature = "text-index")]
        let text_caller = {
            use crate::index::text::{BM25Config, InvertedIndex, SimpleTokenizer};

            let index = Arc::new(RwLock::new(InvertedIndex::with_tokenizer(
                BM25Config::default(),
                Box::new(SimpleTokenizer::with_min_length(7)),
            )));
            original_overlay.add_text_index("Document", "body", Arc::clone(&index));
            original_overlay.set_node_property(node, "body", Value::from("rollback temporalmodel"));
            index
        };
        #[cfg(feature = "vector-index")]
        let vector_caller = {
            use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind};

            let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
                HnswConfig::new(2, DistanceMetric::Euclidean),
                17,
            )));
            original_overlay.add_vector_index("Document", "embedding", Arc::clone(&index));
            original_overlay.set_node_property(
                node,
                "embedding",
                Value::Vector(vec![0.25_f32, 0.75].into()),
            );
            index
        };
        let named = original_overlay
            .graph_or_create("urn:grafeo:default")
            .expect("named default graph");
        let named_node = named.create_node(&["Named"]);
        let nested = named
            .graph_or_create("urn:grafeo:nested")
            .expect("nested named graph");
        let nested_node = nested.create_node(&["Nested"]);
        original_overlay.panic_after_transport_publication_once_for_test();
        let external_published = Arc::new(AtomicBool::new(false));
        let rollback_called = Arc::new(AtomicBool::new(false));
        let retirement_drained = Arc::new(AtomicBool::new(false));
        let publish_state = Arc::clone(&external_published);
        let rollback_state = Arc::clone(&external_published);
        let rollback_witness = Arc::clone(&rollback_called);
        let probe_layered = Arc::clone(&layered);
        let probe_drained = Arc::clone(&retirement_drained);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<bool, String> = layered.merge_overlay_temporal_with_publication(
                |candidate| Ok((candidate, ())),
                move |()| publish_state.swap(true, Ordering::SeqCst),
                move |previous| {
                    rollback_witness.store(true, Ordering::SeqCst);
                    rollback_state.store(previous, Ordering::SeqCst);
                    GuardDrainProbe {
                        layered: probe_layered,
                        observed_drained: probe_drained,
                    }
                },
            );
        }));

        assert!(panic.is_err());
        assert!(rollback_called.load(Ordering::SeqCst));
        assert!(!external_published.load(Ordering::SeqCst));
        assert!(
            retirement_drained.load(Ordering::SeqCst),
            "external rollback retirement must drop only after both Layered guards drain"
        );
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original_base));
        assert!(Arc::ptr_eq(&layered.overlay_store(), &original_overlay));
        assert_eq!(
            original_overlay.find_nodes_by_property("kind", &Value::from("exact")),
            vec![node],
            "rollback must restore the source property-index post-image exactly"
        );
        assert!(Arc::ptr_eq(
            &original_overlay
                .graph("urn:grafeo:default")
                .expect("default graph restored"),
            &named
        ));
        assert!(Arc::ptr_eq(
            &named
                .graph("urn:grafeo:nested")
                .expect("nested graph restored"),
            &nested
        ));
        assert!(named.get_node(named_node).is_some());
        assert!(nested.get_node(nested_node).is_some());
        #[cfg(feature = "text-index")]
        assert_eq!(
            text_caller
                .read()
                .search("rollback", 10)
                .first()
                .map(|(id, _)| *id),
            Some(node),
            "rollback restores the original custom-tokenizer object"
        );
        #[cfg(feature = "vector-index")]
        assert!(
            vector_caller.contains(node),
            "rollback restores the original vector object"
        );
        original_overlay.set_node_property(node, "after_rollback", Value::Bool(true));
        assert_eq!(
            original_overlay.get_node_property(node, &PropertyKey::new("after_rollback")),
            Some(Value::Bool(true)),
            "rollback must reactivate source mutation"
        );
        original_overlay
            .graph_or_create("urn:grafeo:after-rollback")
            .expect("rollback releases the named topology cut");
        #[cfg(feature = "text-index")]
        {
            original_overlay.set_node_property(node, "body", Value::from("restored persistence"));
            assert_eq!(
                text_caller
                    .read()
                    .search("persistence", 10)
                    .first()
                    .map(|(id, _)| *id),
                Some(node)
            );
        }
        #[cfg(feature = "vector-index")]
        {
            let later = original_overlay.create_node(&["Document"]);
            original_overlay.set_node_property(
                later,
                "embedding",
                Value::Vector(vec![0.5_f32, 0.5].into()),
            );
            assert!(vector_caller.contains(later));
        }
    }

    #[test]
    fn temporal_merge_moves_named_and_nested_graph_topology_atomically() {
        let layered = empty_layered();
        let original_overlay = layered.overlay_store();
        let named = original_overlay
            .graph_or_create("urn:grafeo:default")
            .expect("named default graph");
        let named_node = named.create_node(&["Named"]);
        named.set_node_property(named_node, "kind", Value::from("default"));
        let nested = named
            .graph_or_create("urn:grafeo:nested")
            .expect("nested graph");
        let nested_node = nested.create_node(&["Nested"]);
        nested.set_node_property(nested_node, "kind", Value::from("nested"));

        layered.merge_overlay_temporal().expect("temporal merge");
        let successor = layered.overlay_store();
        let successor_named = successor
            .graph("urn:grafeo:default")
            .expect("successor owns named default graph");
        let successor_nested = successor_named
            .graph("urn:grafeo:nested")
            .expect("successor preserves nested graph");
        assert!(Arc::ptr_eq(&successor_named, &named));
        assert!(Arc::ptr_eq(&successor_nested, &nested));
        assert_eq!(
            successor_named.get_node_property(named_node, &PropertyKey::new("kind")),
            Some(Value::from("default"))
        );
        assert_eq!(
            successor_nested.get_node_property(nested_node, &PropertyKey::new("kind")),
            Some(Value::from("nested"))
        );

        assert!(Arc::ptr_eq(
            &original_overlay
                .graph("urn:grafeo:default")
                .expect("retired root keeps its frozen registry"),
            &named
        ));
        successor
            .graph_or_create("urn:grafeo:later")
            .expect("successor accepts later graph DDL");
        assert!(original_overlay.graph("urn:grafeo:later").is_none());
        assert!(
            !original_overlay.drop_graph("urn:grafeo:default"),
            "retired default representation cannot change its frozen topology"
        );
        assert!(successor.graph("urn:grafeo:default").is_some());
    }

    #[test]
    fn temporal_merge_requires_owner_and_retires_old_overlay_to_read_only_snapshot() {
        let layered = empty_layered();
        let original_overlay = layered.overlay_store();
        original_overlay.create_property_index("kind");
        let node = original_overlay.create_node(&["Document"]);
        original_overlay.set_node_property(node, "kind", Value::from("authoritative"));
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(original_overlay.seal_unframed_writes(&owner));

        assert!(layered.merge_overlay_temporal().is_err());
        assert!(with_authority(&foreign, || layered.merge_overlay_temporal()).is_err());
        assert!(Arc::ptr_eq(&layered.overlay_store(), &original_overlay));
        assert_eq!(
            original_overlay.find_nodes_by_property("kind", &Value::from("authoritative")),
            vec![node]
        );

        with_authority(&owner, || layered.merge_overlay_temporal())
            .expect("the exact owner publishes the replacement generation");
        let successor = layered.overlay_store();
        assert!(!Arc::ptr_eq(&successor, &original_overlay));
        assert_eq!(
            original_overlay.find_nodes_by_property("kind", &Value::from("authoritative")),
            vec![node],
            "the retained overlay snapshot keeps its complete property index"
        );
        assert_eq!(
            layered.find_nodes_by_property("kind", &Value::from("authoritative")),
            vec![node],
            "the successor routes cold equality membership without loss"
        );
        assert!(successor.has_property_index("kind"));
        assert_eq!(
            successor.find_nodes_by_property("kind", &Value::from("authoritative")),
            vec![node],
            "the successor's complete property index retains cold membership"
        );

        with_authority(&owner, || {
            original_overlay.set_node_property(node, "kind", Value::from("forged"));
        });
        assert_eq!(
            original_overlay.get_node_property(node, &PropertyKey::new("kind")),
            Some(Value::from("authoritative")),
            "even the former owner cannot mutate a retired physical representation"
        );
        assert!(
            layered
                .find_nodes_by_property("kind", &Value::from("forged"))
                .is_empty(),
            "a retained old overlay cannot corrupt successor index membership"
        );
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn temporal_merge_preserves_custom_tokenizer_and_text_mvcc_on_all_retained_views()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::index::text::{BM25Config, InvertedIndex, SimpleTokenizer};

        let layered = empty_layered();
        let original_overlay = layered.overlay_store();
        let caller = Arc::new(RwLock::new(InvertedIndex::with_tokenizer(
            BM25Config::default(),
            Box::new(SimpleTokenizer::with_min_length(7)),
        )));
        original_overlay.add_text_index("Document", "body", Arc::clone(&caller));
        let retained_view = original_overlay
            .get_text_index("Document", "body")
            .expect("registered custom text index");
        original_overlay.set_epoch(EpochId::new(5));
        let node = original_overlay.create_node(&["Document"]);
        original_overlay.set_node_property(node, "body", Value::from("ancient temporalmodel"));
        original_overlay.set_epoch(EpochId::new(9));
        original_overlay.set_node_property(node, "body", Value::from("future persistence"));

        assert_eq!(
            layered
                .text_search_visible(
                    "Document",
                    "body",
                    "ancient",
                    10,
                    EpochId::new(5),
                    TransactionId::INVALID,
                )?
                .first()
                .map(|(id, _)| *id),
            Some(node)
        );
        assert!(retained_view.read().search("tiny", 10).is_empty());
        assert_eq!(
            retained_view
                .read()
                .search("persistence", 10)
                .first()
                .map(|(id, _)| *id),
            Some(node)
        );

        layered.merge_overlay_temporal().expect("temporal merge");
        let successor = layered.overlay_store();
        let successor_view = successor
            .get_text_index("Document", "body")
            .expect("successor keeps the text registry");
        let old_overlay_view = original_overlay
            .get_text_index("Document", "body")
            .expect("retired overlay keeps a readable text registry");
        for view in [&retained_view, &old_overlay_view, &successor_view] {
            assert!(view.read().search("tiny", 10).is_empty());
            assert_eq!(
                view.read()
                    .search("persistence", 10)
                    .first()
                    .map(|(id, _)| *id),
                Some(node)
            );
        }
        assert_eq!(
            layered
                .text_search_visible(
                    "Document",
                    "body",
                    "ancient",
                    10,
                    EpochId::new(5),
                    TransactionId::INVALID,
                )?
                .first()
                .map(|(id, _)| *id),
            Some(node),
            "versioned postings must survive without re-tokenization"
        );
        assert_eq!(
            layered
                .text_search_visible(
                    "Document",
                    "body",
                    "persistence",
                    10,
                    EpochId::new(9),
                    TransactionId::INVALID,
                )?
                .first()
                .map(|(id, _)| *id),
            Some(node)
        );

        original_overlay.set_node_property(node, "body", Value::from("forged replacement"));
        assert!(retained_view.read().search("replacement", 10).is_empty());
        let later = layered.create_node(&["Document"]);
        layered.set_node_property(later, "body", Value::from("ambitious temporalmodel"));
        assert_eq!(
            caller
                .read()
                .search("ambitious", 10)
                .first()
                .map(|(id, _)| *id),
            Some(later),
            "the retained caller gate remains attached to the logical successor index"
        );
        assert_eq!(
            retained_view
                .read()
                .search("ambitious", 10)
                .first()
                .map(|(id, _)| *id),
            Some(later),
            "a pre-merge logical index handle follows the successor"
        );
        assert!(
            old_overlay_view.read().search("ambitious", 10).is_empty(),
            "a retained overlay snapshot cannot gain a successor-only identity"
        );
        assert!(
            original_overlay
                .get_text_index("Document", "body")
                .expect("retired overlay keeps its frozen registry")
                .read()
                .search("ambitious", 10)
                .is_empty()
        );
        assert!(original_overlay.get_node(later).is_none());

        let foreign_store = LpgStore::new().expect("foreign store");
        foreign_store.add_text_index("Document", "body", Arc::clone(&caller));
        assert!(
            foreign_store.get_text_index("Document", "body").is_none(),
            "handoff must preserve the logical index's exact store/slot owner binding"
        );
        Ok(())
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn restored_overlay_binds_exact_cold_vector_generation_before_mutation() {
        use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind, VectorStoreSection};

        let source = Arc::new(LpgStore::new().expect("cold vector source"));
        let config = HnswConfig::new(2, DistanceMetric::Euclidean)
            .with_m(8)
            .with_ef_construction(64)
            .with_ef(64);
        source.add_vector_index(
            "Document",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
                config.clone(),
                0x0000_c01d_ba5e,
            ))),
        );
        let mut cold = Vec::new();
        for offset in 0..96_u64 {
            let node = source.create_node(&["Document"]);
            source.set_node_property(
                node,
                "embedding",
                Value::Vector(vec![offset as f32, 0.0].into()),
            );
            cold.push(node);
        }

        let image = VectorStoreSection::from_views(
            source
                .vector_index_entries()
                .into_iter()
                .map(|(_, view)| {
                    (
                        crate::graph::lpg::PhysicalIndexKey::vector(
                            grafeo_common::types::GraphPath::root(),
                            "Document",
                            "embedding",
                        ),
                        view,
                    )
                })
                .collect(),
        )
        .serialize()
        .expect("serialize exact vector topology");
        let base = Arc::new(
            from_graph_store_preserving_ids(source.as_ref()).expect("build compact vector base"),
        );
        let overlay = Arc::new(LpgStore::new().expect("private recovery overlay"));
        overlay.add_vector_index(
            "Document",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
                config,
                0x0000_c01d_ba5e,
            ))),
        );
        let mut recovery = VectorStoreSection::for_unpublished_recovery_views(
            overlay
                .vector_index_entries()
                .into_iter()
                .map(|(_, view)| {
                    (
                        crate::graph::lpg::PhysicalIndexKey::vector(
                            grafeo_common::types::GraphPath::root(),
                            "Document",
                            "embedding",
                        ),
                        view,
                    )
                })
                .collect(),
        );
        recovery
            .deserialize(&image)
            .expect("restore exact topology into private overlay");

        let entry_point = overlay
            .get_vector_index("Document", "embedding")
            .expect("restored vector index")
            .snapshot_topology()
            .0;
        let moved = cold
            .iter()
            .rev()
            .copied()
            .find(|id| Some(*id) != entry_point)
            .expect("fixture has a non-entry-point node");
        let layered = LayeredStore::with_overlay(Arc::clone(&base), Arc::clone(&overlay))
            .expect("bind restored overlay to exact compact generation");
        assert!(
            LayeredStore::with_overlay(base, Arc::clone(&overlay)).is_err(),
            "one LPG representation cannot acquire a second compact vector tier"
        );

        layered.set_node_property(
            moved,
            "embedding",
            Value::Vector(vec![-1_000.0_f32, -1_000.0].into()),
        );
        let (_, _, topology) = overlay
            .get_vector_index("Document", "embedding")
            .expect("updated restored vector index")
            .snapshot_topology();
        let moved_layers = topology
            .iter()
            .find_map(|(id, layers)| (*id == moved).then_some(layers))
            .expect("updated node remains in topology");
        assert!(
            moved_layers
                .first()
                .is_some_and(|neighbors| neighbors.iter().any(|id| *id != moved)),
            "topology maintenance must resolve real neighbours from the compact vector tier"
        );

        assert_eq!(
            layered
                .vector_search(
                    Some("Document"),
                    "embedding",
                    &[-1_000.0, -1_000.0],
                    1,
                    DistanceMetric::Euclidean,
                )
                .first()
                .map(|(id, _)| *id),
            Some(moved),
            "the re-embedded cold node must be reachable at its new nearest-neighbour position"
        );
        let mut hits: Vec<_> = layered
            .vector_search(
                Some("Document"),
                "embedding",
                &[-1_000.0, -1_000.0],
                cold.len(),
                DistanceMetric::Euclidean,
            )
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        hits.sort_unstable();
        cold.sort_unstable();
        assert_eq!(
            hits, cold,
            "one cold-node update cannot disconnect the restored physical topology"
        );
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn temporal_merge_preserves_quantized_kind_configuration_and_exact_topology() {
        use crate::index::vector::{
            HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
        };

        let layered = empty_layered();
        let original_overlay = layered.overlay_store();
        let config = HnswConfig::new(3, DistanceMetric::Euclidean)
            .with_m(7)
            .with_m_max(19)
            .with_ef_construction(41)
            .with_ef(23)
            .with_alpha(1.3);
        let caller = Arc::new(VectorIndexKind::Quantized(
            QuantizedHnswIndex::with_seed(config.clone(), QuantizationType::Scalar, 0x5eed)
                .without_rescore()
                .with_rescore_factor(7)
                .with_training_threshold(10),
        ));
        original_overlay.add_vector_index("Document", "embedding", Arc::clone(&caller));
        let retained_view = original_overlay
            .get_vector_index("Document", "embedding")
            .expect("registered quantized vector index");
        let mut first = NodeId::INVALID;
        for offset in 0..12_u64 {
            let node = original_overlay.create_node(&["Document"]);
            if !first.is_valid() {
                first = node;
            }
            original_overlay.set_node_property(
                node,
                "embedding",
                Value::Vector(vec![offset as f32, (offset % 3) as f32, 1.0 - offset as f32].into()),
            );
        }
        let quantized = caller.as_quantized().expect("quantized kind");
        let topology = retained_view.snapshot_topology();
        let fingerprint = quantized.state_fingerprint();
        assert_eq!(retained_view.len(), 12);
        assert!(!quantized.rescoring_enabled());
        assert_eq!(quantized.rescore_factor(), 7);
        assert_eq!(quantized.training_threshold(), 10);

        layered.merge_overlay_temporal().expect("temporal merge");
        let successor_view = layered
            .overlay_store()
            .get_vector_index("Document", "embedding")
            .expect("successor keeps quantized vector index");
        let old_overlay_view = original_overlay
            .get_vector_index("Document", "embedding")
            .expect("retired overlay keeps a readable vector registry");
        for view in [&retained_view, &old_overlay_view, &successor_view] {
            let actual = view.config();
            assert_eq!(actual.dimensions, config.dimensions);
            assert_eq!(actual.metric, config.metric);
            assert_eq!(actual.m, config.m);
            assert_eq!(actual.m_max, config.m_max);
            assert_eq!(actual.ef_construction, config.ef_construction);
            assert_eq!(actual.ef, config.ef);
            assert_eq!(actual.ml.to_bits(), config.ml.to_bits());
            assert_eq!(actual.alpha.to_bits(), config.alpha.to_bits());
            assert_eq!(actual.max_elements, config.max_elements);
            assert_eq!(view.quantization_type(), Some(QuantizationType::Scalar));
            assert_eq!(view.snapshot_topology(), topology);
            assert_eq!(view.len(), 12);
        }
        assert_eq!(
            layered
                .vector_search(
                    Some("Document"),
                    "embedding",
                    &[0.0, 0.0, 1.0],
                    1,
                    DistanceMetric::Euclidean,
                )
                .first()
                .map(|(id, _)| *id),
            Some(first),
            "the transferred physical index must serve its cold-base identities through Layered"
        );
        assert_eq!(quantized.state_fingerprint(), fingerprint);
        assert!(!quantized.rescoring_enabled());
        assert_eq!(quantized.rescore_factor(), 7);
        assert_eq!(quantized.training_threshold(), 10);

        original_overlay.set_node_property(
            first,
            "embedding",
            Value::Vector(vec![999.0_f32, 999.0, 999.0].into()),
        );
        assert_eq!(
            quantized.state_fingerprint(),
            fingerprint,
            "retired-overlay mutation cannot change the shared exact index"
        );
        let later = layered.create_node(&["Document"]);
        layered.set_node_property(
            later,
            "embedding",
            Value::Vector(vec![0.25_f32, 0.5, 0.75].into()),
        );
        assert!(
            quantized.contains(later),
            "the retained exact index handle follows successor-owned mutation"
        );
        assert!(retained_view.contains(later));
        assert!(successor_view.contains(later));
        assert!(
            !old_overlay_view.contains(later),
            "a retained overlay vector view cannot gain a successor-only identity"
        );
        assert!(
            !original_overlay
                .get_vector_index("Document", "embedding")
                .expect("retired overlay keeps its frozen vector registry")
                .contains(later)
        );
        assert_eq!(old_overlay_view.snapshot_topology(), topology);
        assert!(original_overlay.get_node(later).is_none());

        let foreign_store = LpgStore::new().expect("foreign store");
        foreign_store.add_vector_index("Document", "embedding", Arc::clone(&caller));
        assert!(
            foreign_store
                .get_vector_index("Document", "embedding")
                .is_none(),
            "handoff must preserve the logical vector index's exact owner/slot binding"
        );
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn temporal_merge_preserves_standard_hnsw_across_cold_hot_and_visible_reads() {
        use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind};

        let layered = empty_layered();
        let original_overlay = layered.overlay_store();
        let config = HnswConfig::new(2, DistanceMetric::Euclidean)
            .with_m(6)
            .with_ef_construction(32)
            .with_ef(24);
        let caller = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            config,
            0x0051_a71c,
        )));
        original_overlay.add_vector_index("Document", "embedding", Arc::clone(&caller));
        let retained_view = original_overlay
            .get_vector_index("Document", "embedding")
            .expect("registered standard HNSW index");

        original_overlay.set_epoch(EpochId::new(5));
        let mut cold = Vec::new();
        for offset in 0..8_u64 {
            let node = original_overlay.create_node(&["Document"]);
            original_overlay.set_node_property(
                node,
                "embedding",
                Value::Vector(vec![(offset * 10) as f32, 0.0].into()),
            );
            cold.push(node);
        }
        let exact_topology = retained_view.snapshot_topology();

        layered.merge_overlay_temporal().expect("temporal merge");
        let successor_view = layered
            .overlay_store()
            .get_vector_index("Document", "embedding")
            .expect("successor keeps standard HNSW");
        let old_overlay_view = original_overlay
            .get_vector_index("Document", "embedding")
            .expect("retired overlay keeps a frozen standard HNSW view");
        assert_eq!(successor_view.snapshot_topology(), exact_topology);
        assert_eq!(retained_view.snapshot_topology(), exact_topology);
        assert_eq!(old_overlay_view.snapshot_topology(), exact_topology);

        assert_eq!(
            layered
                .vector_search(
                    Some("Document"),
                    "embedding",
                    &[0.0, 0.0],
                    1,
                    DistanceMetric::Euclidean,
                )
                .first()
                .map(|(id, _)| *id),
            Some(cold[0]),
            "topology-only HNSW must resolve vectors from the cold CompactStore"
        );
        assert_eq!(
            layered
                .vector_search(
                    Some("Document"),
                    "embedding",
                    &[0.0, 0.0],
                    1,
                    DistanceMetric::Manhattan,
                )
                .first()
                .map(|(id, _)| *id),
            Some(cold[0]),
            "a metric mismatch must retain the full tier-merged brute-force fallback"
        );
        let threshold = layered.vector_search_with_threshold(
            Some("Document"),
            "embedding",
            &[0.0, 0.0],
            0.5,
            DistanceMetric::Euclidean,
        );
        assert_eq!(
            threshold.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![cold[0]]
        );
        assert_eq!(
            layered
                .vector_search_visible(
                    "Document",
                    "embedding",
                    &[0.0, 0.0],
                    1,
                    EpochId::new(5),
                    TransactionId::INVALID,
                )
                .first()
                .map(|(id, _)| *id),
            Some(cold[0]),
            "snapshot-visible HNSW must score cold properties through Layered"
        );

        let old_accessor = PropertyVectorAccessor::new(original_overlay.as_ref(), "embedding");
        assert_eq!(
            retained_view
                .search(&[0.0, 0.0], 1, &old_accessor)
                .first()
                .map(|(id, _)| *id),
            Some(cold[0]),
            "the retained topology view remains usable with its retained snapshot accessor"
        );

        let hot = layered.create_node(&["Document"]);
        layered.set_node_property(
            hot,
            "embedding",
            Value::Vector(vec![100.0_f32, 100.0].into()),
        );
        assert_eq!(
            layered
                .vector_search(
                    Some("Document"),
                    "embedding",
                    &[100.0, 100.0],
                    1,
                    DistanceMetric::Euclidean,
                )
                .first()
                .map(|(id, _)| *id),
            Some(hot),
            "successor insertion must resolve cold neighbors while extending exact HNSW topology"
        );
        assert!(caller.contains(hot));
        assert!(retained_view.contains(hot));
        assert!(successor_view.contains(hot));
        assert!(!old_overlay_view.contains(hot));
        assert_eq!(old_overlay_view.snapshot_topology(), exact_topology);

        let tx = TransactionId::new(0x51a7);
        let pending = layered.create_node_versioned(&["Document"], EpochId::new(5), tx);
        layered.set_node_property_buffered(
            pending,
            "embedding",
            Value::Vector(vec![1000.0_f32, 1000.0].into()),
            tx,
        );
        let writer_results = layered.vector_search_visible(
            "Document",
            "embedding",
            &[1000.0, 1000.0],
            1,
            EpochId::new(5),
            tx,
        );
        assert_eq!(writer_results.first().map(|(id, _)| *id), Some(pending));
        assert!(
            layered
                .vector_search(
                    Some("Document"),
                    "embedding",
                    &[1000.0, 1000.0],
                    16,
                    DistanceMetric::Euclidean,
                )
                .iter()
                .all(|(id, _)| *id != pending),
            "committed search must exclude a transaction-private vector"
        );

        assert!(layered.delete_node(cold[0]));
        assert!(
            layered
                .vector_search(
                    Some("Document"),
                    "embedding",
                    &[0.0, 0.0],
                    16,
                    DistanceMetric::Euclidean,
                )
                .iter()
                .all(|(id, _)| *id != cold[0]),
            "a cold identity tombstoned in the hot tier must not leak from transferred topology"
        );
        assert!(
            layered
                .vector_search_with_threshold(
                    Some("Document"),
                    "embedding",
                    &[0.0, 0.0],
                    0.5,
                    DistanceMetric::Euclidean,
                )
                .iter()
                .all(|(id, _)| *id != cold[0])
        );
    }

    #[test]
    fn deleted_node_and_edge_history_survives_recompact_and_section_roundtrip() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let gone = overlay.create_node(&["Person"]);
        let kept = overlay.create_node(&["Person"]);
        overlay.set_node_property_at_epoch(gone, "name", Value::from("Gone"), EpochId::new(10));
        let edge = overlay.create_edge_versioned(
            gone,
            kept,
            "KNOWS",
            EpochId::new(10),
            TransactionId::SYSTEM,
        );
        overlay.set_edge_property_at_epoch(edge, "weight", Value::Int64(7), EpochId::new(10));
        layered.merge_overlay_temporal().unwrap();

        // Delete cold-base entities in a later generation. Before compaction,
        // the in-memory tombstone must already honor the exact half-open bound.
        layered.overlay_store().set_epoch(EpochId::new(20));
        assert!(layered.delete_edge(edge));
        assert!(layered.delete_node(gone));
        assert!(layered.get_node_at_epoch(gone, EpochId::new(19)).is_some());
        assert!(layered.get_node_at_epoch(gone, EpochId::new(20)).is_none());
        assert!(layered.get_edge_at_epoch(edge, EpochId::new(19)).is_some());
        assert!(layered.get_edge_at_epoch(edge, EpochId::new(20)).is_none());

        let assert_boundaries = |store: &dyn GraphStore| {
            for epoch in [9, 20, 21] {
                assert!(
                    store.get_node_at_epoch(gone, EpochId::new(epoch)).is_none(),
                    "node must be absent at epoch {epoch}"
                );
                assert!(
                    store.get_edge_at_epoch(edge, EpochId::new(epoch)).is_none(),
                    "edge must be absent at epoch {epoch}"
                );
            }
            assert!(store.get_node_at_epoch(gone, EpochId::new(10)).is_some());
            assert!(store.get_node_at_epoch(gone, EpochId::new(19)).is_some());
            assert!(store.get_edge_at_epoch(edge, EpochId::new(10)).is_some());
            assert!(store.get_edge_at_epoch(edge, EpochId::new(19)).is_some());
            assert!(store.get_node(gone).is_none());
            assert!(store.get_edge(edge).is_none());
        };

        layered.merge_overlay_temporal().unwrap();
        assert_boundaries(&layered);
        assert!(
            layered
                .nodes_at_epoch(EpochId::new(19))
                .iter()
                .any(|n| n.id == gone)
        );
        assert!(
            !layered
                .nodes_at_epoch(EpochId::new(20))
                .iter()
                .any(|n| n.id == gone)
        );

        // An empty-overlay recompact must carry the closed row forward.
        layered.merge_overlay_temporal().unwrap();
        assert_boundaries(&layered);

        // Current compact persistence must carry it across reopen as well.
        let section = CompactStoreSection::new(layered.base_store_arc());
        let bytes = section.serialize().unwrap();
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&bytes).unwrap();
        let restored = restored.store().expect("restored compact store");
        assert_boundaries(restored.as_ref());
    }

    #[test]
    fn complete_histories_preserve_labels_structure_and_properties_across_compaction() {
        use crate::graph::compact::compaction::EdgeLifetime;

        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let label_added = EpochId::new(12);
        let label_removed = EpochId::new(14);
        let property_changed = EpochId::new(16);
        let deleted = EpochId::new(20);

        overlay.set_epoch(created);
        let gone = overlay.create_node(&["Person"]);
        let kept = overlay.create_node(&["Anchor"]);
        overlay.set_node_property_at_epoch(gone, "name", Value::from("Gone"), created);
        let edge =
            overlay.create_edge_versioned(gone, kept, "KNOWS", created, TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(edge, "weight", Value::Int64(7), created);

        overlay.set_epoch(label_added);
        assert!(overlay.add_label(gone, "Researcher"));
        overlay.set_epoch(label_removed);
        assert!(overlay.remove_label(gone, "Person"));
        overlay.set_node_property_at_epoch(
            gone,
            "name",
            Value::from("Historian"),
            property_changed,
        );
        overlay.set_edge_property_at_epoch(edge, "weight", Value::Int64(9), property_changed);
        overlay.set_epoch(deleted);
        assert!(overlay.delete_edge_at_epoch(edge, deleted));
        assert!(overlay.delete_node_at_epoch(gone, deleted));

        let assert_complete = |store: &LayeredStore| {
            let nodes = store.complete_node_histories();
            assert!(nodes.windows(2).all(|pair| pair[0].0 < pair[1].0));
            let node = &nodes
                .iter()
                .find(|(id, _)| *id == gone)
                .expect("deleted node history")
                .1;
            assert_eq!(
                node.lifetimes,
                vec![EdgeLifetime::new(created, Some(deleted))]
            );
            assert_eq!(
                node.label_versions
                    .iter()
                    .map(|(epoch, labels)| {
                        (
                            *epoch,
                            labels.iter().map(ToString::to_string).collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
                vec![
                    (created, vec![String::from("Person")]),
                    (
                        label_added,
                        vec![String::from("Person"), String::from("Researcher")],
                    ),
                    (label_removed, vec![String::from("Researcher")]),
                ]
            );
            assert_eq!(
                node.properties.get(&PropertyKey::new("name")),
                Some(&vec![
                    (created, Value::from("Gone")),
                    (property_changed, Value::from("Historian")),
                    (deleted, Value::Null),
                ])
            );

            let edges = store.complete_edge_histories();
            assert!(edges.windows(2).all(|pair| pair[0].0 < pair[1].0));
            let edge_history = &edges
                .iter()
                .find(|(id, _)| *id == edge)
                .expect("deleted edge history")
                .1;
            assert_eq!(edge_history.src, gone);
            assert_eq!(edge_history.dst, kept);
            assert_eq!(edge_history.edge_type.as_str(), "KNOWS");
            assert_eq!(
                edge_history.lifetimes,
                vec![EdgeLifetime::new(created, Some(deleted))]
            );
            assert_eq!(
                edge_history.properties.get(&PropertyKey::new("weight")),
                Some(&vec![
                    (created, Value::Int64(7)),
                    (property_changed, Value::Int64(9)),
                    (deleted, Value::Null),
                ])
            );

            let at_create = store
                .get_node_at_epoch(gone, created)
                .expect("node at create");
            assert!(at_create.has_label("Person"));
            assert!(!at_create.has_label("Researcher"));
            let after_add = store
                .get_node_at_epoch(gone, label_added)
                .expect("node after label add");
            assert!(after_add.has_label("Person"));
            assert!(after_add.has_label("Researcher"));
            let after_remove = store
                .get_node_at_epoch(gone, label_removed)
                .expect("node after label remove");
            assert!(!after_remove.has_label("Person"));
            assert!(after_remove.has_label("Researcher"));
            assert!(store.get_node_at_epoch(gone, deleted).is_none());
        };

        assert_complete(&layered);
        layered.merge_overlay_temporal().unwrap();
        assert_complete(&layered);
        layered.merge_overlay_temporal().unwrap();
        assert_complete(&layered);

        let section = CompactStoreSection::new(layered.base_store_arc());
        let bytes = section.serialize().unwrap();
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&bytes).unwrap();
        let restored = restored.store().expect("restored compact store");
        let restored = LayeredStore::with_overlay(
            restored,
            Arc::new(LpgStore::new().expect("restored overlay")),
        )
        .expect("adopt restored overlay");
        assert_complete(&restored);
    }

    #[test]
    fn heterogeneous_node_and_edge_histories_survive_recompact_and_section_roundtrip() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let changed = EpochId::new(20);

        overlay.set_epoch(created);
        let subject = overlay.create_node(&["Subject"]);
        let peer = overlay.create_node(&["Peer"]);
        overlay.set_node_property_at_epoch(subject, "mixed", Value::Int64(7), created);
        overlay.set_node_property_at_epoch(
            subject,
            "embedding",
            Value::Vector(vec![1.0, 2.0].into()),
            created,
        );
        let edge =
            overlay.create_edge_versioned(subject, peer, "CHANGES", created, TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(edge, "mixed", Value::Int64(9), created);
        overlay.set_edge_property_at_epoch(
            edge,
            "embedding",
            Value::Vector(vec![3.0, 4.0].into()),
            created,
        );

        overlay.set_epoch(changed);
        overlay.set_node_property_at_epoch(subject, "mixed", Value::from("seven"), changed);
        overlay.set_node_property_at_epoch(
            subject,
            "embedding",
            Value::Vector(vec![5.0, 6.0, 7.0].into()),
            changed,
        );
        overlay.set_edge_property_at_epoch(edge, "mixed", Value::from("nine"), changed);
        overlay.set_edge_property_at_epoch(
            edge,
            "embedding",
            Value::Vector(vec![8.0, 9.0, 10.0].into()),
            changed,
        );

        let assert_history = |store: &dyn GraphStore, stage: &str| {
            let at_create = store
                .get_node_at_epoch(subject, created)
                .expect("node at its creation epoch");
            assert_eq!(
                at_create.get_property("mixed"),
                Some(&Value::Int64(7)),
                "node mixed history at {stage}"
            );
            assert_eq!(
                at_create.get_property("embedding"),
                Some(&Value::Vector(vec![1.0, 2.0].into()))
            );
            let at_change = store
                .get_node_at_epoch(subject, changed)
                .expect("node at its update epoch");
            assert_eq!(at_change.get_property("mixed"), Some(&Value::from("seven")));
            assert_eq!(
                at_change.get_property("embedding"),
                Some(&Value::Vector(vec![5.0, 6.0, 7.0].into()))
            );

            let edge_at_create = store
                .get_edge_at_epoch(edge, created)
                .expect("edge at its creation epoch");
            assert_eq!(edge_at_create.get_property("mixed"), Some(&Value::Int64(9)));
            assert_eq!(
                edge_at_create.get_property("embedding"),
                Some(&Value::Vector(vec![3.0, 4.0].into()))
            );
            let edge_at_change = store
                .get_edge_at_epoch(edge, changed)
                .expect("edge at its update epoch");
            assert_eq!(
                edge_at_change.get_property("mixed"),
                Some(&Value::from("nine"))
            );
            assert_eq!(
                edge_at_change.get_property("embedding"),
                Some(&Value::Vector(vec![8.0, 9.0, 10.0].into()))
            );
        };

        layered.merge_overlay_temporal().expect("first compaction");
        assert_eq!(
            layered
                .complete_node_histories()
                .into_iter()
                .find(|(id, _)| *id == subject)
                .expect("compacted subject history")
                .1
                .properties
                .get(&PropertyKey::new("mixed")),
            Some(&vec![
                (created, Value::Int64(7)),
                (changed, Value::from("seven")),
            ])
        );
        let base = layered.base_store_arc();
        let row = base
            .temporal_node_row_at(subject, created)
            .expect("temporal subject row");
        assert_eq!(
            row.raw_properties[&PropertyKey::new("mixed")].value_as_of(created),
            Some(Value::Int64(7))
        );
        assert_eq!(
            base.get_node_at_epoch(subject, created)
                .and_then(|node| node.get_property("mixed").cloned()),
            Some(Value::Int64(7))
        );
        assert_history(&layered, "first compact");
        layered
            .merge_overlay_temporal()
            .expect("empty-overlay recompact");
        assert_history(&layered, "empty-overlay recompact");

        let section = CompactStoreSection::new(layered.base_store_arc());
        let bytes = section.serialize().expect("serialize compact v9 section");
        assert_eq!(bytes[4], 9, "retained coverage requires compact section v9");

        let mut restored = CompactStoreSection::empty();
        restored
            .deserialize(&bytes)
            .expect("deserialize compact v9 section");
        let restored = LayeredStore::with_overlay(
            restored.store().expect("restored compact base"),
            Arc::new(LpgStore::new().expect("restored overlay")),
        )
        .expect("adopt restored overlay");
        assert_history(&restored, "section roundtrip");
        restored.overlay_store().set_epoch(changed);
        restored
            .merge_overlay_temporal()
            .expect("recompact deserialized raw histories");
        assert_history(&restored, "section roundtrip recompact");
    }

    #[test]
    fn ordered_property_mutations_at_structural_delete_survive_compaction() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let deleted = EpochId::new(20);
        let key = PropertyKey::new("boundary");

        overlay.set_epoch(created);
        let subject = overlay.create_node(&["Subject"]);
        let peer = overlay.create_node(&["Peer"]);
        overlay.set_node_property_at_epoch(subject, key.as_str(), Value::Int64(1), created);
        let edge =
            overlay.create_edge_versioned(subject, peer, "CLOSES", created, TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(edge, key.as_str(), Value::Int64(3), created);

        // Two ordered property operations and the structural delete share one
        // commit epoch. They have no additional visible interval, but they are
        // part of the durable history and must not be normalized to one tombstone.
        overlay.set_epoch(deleted);
        overlay.set_node_property_at_epoch(subject, key.as_str(), Value::Int64(2), deleted);
        overlay.set_node_property_at_epoch(subject, key.as_str(), Value::Null, deleted);
        overlay.set_edge_property_at_epoch(edge, key.as_str(), Value::Int64(4), deleted);
        overlay.set_edge_property_at_epoch(edge, key.as_str(), Value::Null, deleted);
        assert!(overlay.delete_edge_at_epoch(edge, deleted));
        assert!(overlay.delete_node_at_epoch(subject, deleted));

        let assert_exact = |store: &LayeredStore, stage: &str| {
            assert_eq!(
                store
                    .complete_node_histories()
                    .into_iter()
                    .find(|(id, _)| *id == subject)
                    .expect("subject history")
                    .1
                    .properties
                    .get(&key),
                Some(&vec![
                    (created, Value::Int64(1)),
                    (deleted, Value::Int64(2)),
                    (deleted, Value::Null),
                ])
            );
            assert_eq!(
                store
                    .complete_edge_histories()
                    .into_iter()
                    .find(|(id, _)| *id == edge)
                    .expect("edge history")
                    .1
                    .properties
                    .get(&key),
                Some(&vec![
                    (created, Value::Int64(3)),
                    (deleted, Value::Int64(4)),
                    (deleted, Value::Null),
                ])
            );
            assert_eq!(
                store.get_node_property_at_epoch(subject, &key, EpochId::new(19)),
                Some(Value::Int64(1)),
                "node property state before delete at {stage}"
            );
            assert!(store.get_node_at_epoch(subject, deleted).is_none());
            assert!(store.get_edge_at_epoch(edge, deleted).is_none());
        };

        layered
            .merge_overlay_temporal()
            .expect("compact boundary log");
        assert_exact(&layered, "first compact");
        layered
            .merge_overlay_temporal()
            .expect("recompact boundary log");
        assert_exact(&layered, "empty-overlay recompact");

        let section = CompactStoreSection::new(layered.base_store_arc());
        let bytes = section.serialize().expect("serialize boundary log");
        let mut restored = CompactStoreSection::empty();
        restored
            .deserialize(&bytes)
            .expect("deserialize boundary log");
        let restored = LayeredStore::with_overlay(
            restored.store().expect("restored compact base"),
            Arc::new(LpgStore::new().expect("restored overlay")),
        )
        .expect("adopt restored overlay");
        assert_exact(&restored, "section roundtrip");
    }

    /// SP3 slice 6: after compacting a multi-version history into the temporal
    /// base, the layered store's as-of reads and whole-state scrub return the
    /// graph's state at each epoch — not just the current state.
    #[test]
    fn scrub_at_epoch_reads_compacted_history() {
        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(base, 1000, 1000).unwrap();

        // Build a 3-version history directly in the overlay at explicit epochs,
        // then set the overlay's clock so the current value is the latest.
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let nid = overlay.create_node(&["Item"]);
        overlay.set_node_property_at_epoch(nid, "score", Value::Int64(100), EpochId::new(10));
        overlay.set_node_property_at_epoch(nid, "score", Value::Int64(200), EpochId::new(20));
        overlay.set_node_property_at_epoch(nid, "score", Value::Int64(300), EpochId::new(30));
        overlay.set_epoch(EpochId::new(30));

        layered.merge_overlay_temporal().unwrap();

        let key = PropertyKey::new("score");
        // As-of reads route through the temporal base.
        let at = |e: u64| {
            layered
                .get_node_at_epoch(nid, EpochId::new(e))
                .and_then(|n| n.properties.get(&key).cloned())
        };
        assert_eq!(at(15), Some(Value::Int64(100)));
        assert_eq!(at(25), Some(Value::Int64(200)));
        assert_eq!(at(35), Some(Value::Int64(300)));
        // Current read is still the latest.
        assert_eq!(
            layered.get_node_property(nid, &key),
            Some(Value::Int64(300))
        );

        // Whole-state scrub at epoch 25.
        let scrub = layered.nodes_at_epoch(EpochId::new(25));
        assert_eq!(scrub.len(), 1);
        assert_eq!(scrub[0].properties.get(&key), Some(&Value::Int64(200)));
        // Before the node existed, the scrub is empty.
        assert!(layered.nodes_at_epoch(EpochId::new(5)).is_empty());
    }

    /// SP2 slice 5: the temporal merge compacts only the committed frontier —
    /// an uncommitted (PENDING) version is never folded into the cold base.
    #[test]
    fn merge_overlay_temporal_excludes_uncommitted_from_base() {
        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(base, 1000, 1000).unwrap();
        let overlay = layered.overlay_store();
        let nid = overlay.create_node(&["Item"]);
        overlay.set_node_property_at_epoch(nid, "score", Value::Int64(100), EpochId::new(10));
        overlay.set_node_property_at_epoch(nid, "score", Value::Int64(200), EpochId::new(20));
        // Uncommitted write — must NOT reach the cold base.
        overlay.set_node_property_at_epoch(nid, "score", Value::Int64(999), EpochId::PENDING);
        overlay.set_epoch(EpochId::new(20)); // committed frontier = 20

        layered.merge_overlay_temporal().unwrap();

        let base = layered.base_store_arc();
        let key = PropertyKey::new("score");
        assert_eq!(
            base.get_node_property_at_epoch(nid, &key, EpochId::new(15)),
            Some(Value::Int64(100))
        );
        assert_eq!(
            base.get_node_property_at_epoch(nid, &key, EpochId::new(25)),
            Some(Value::Int64(200))
        );
        assert_eq!(base.get_node_property(nid, &key), Some(Value::Int64(200)));
        // 999 (uncommitted) appears nowhere in the cold base's history.
        assert!(
            base.node_property_history(nid)
                .iter()
                .all(|(_, h)| h.iter().all(|(_, v)| *v != Value::Int64(999)))
        );
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn recorded_text_recovery_pins_cold_overlay_and_releases_scopes_after_unwind()
    -> Result<(), Error> {
        use crate::index::text::{BM25Config, InvertedIndex, TextIndexSection};
        use grafeo_common::storage::Section;
        use grafeo_common::types::GraphPath;
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.sync_epoch(EpochId::new(1));
        overlay.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );
        let node =
            overlay.create_node_with_props(&["Doc"], [("body", Value::from("originaltoken"))]);
        overlay.create_property_index("body");
        drop(overlay);
        layered
            .merge_overlay_temporal()
            .map_err(Error::InvalidValue)?;
        let overlay = layered.overlay_store();
        overlay.sync_epoch(EpochId::new(2));
        let section = TextIndexSection::from_views(vec![(
            crate::graph::lpg::PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            overlay
                .get_text_index("Doc", "body")
                .expect("transferred Text target"),
        )]);
        let before = section.serialize()?;
        layered.with_recorded_index_recovery(true, false, || {
            layered.set_node_property(node, "body", Value::from("replayedtoken"));
            layered.replay_node_labels_at_epoch(node, EpochId::new(2), &[])?;
            layered.replay_node_labels_at_epoch(node, EpochId::new(2), &[ArcStr::from("Doc")])?;
            Ok(())
        })?;
        assert_eq!(section.serialize()?, before);
        assert_eq!(
            layered.find_nodes_by_property("body", &Value::from("replayedtoken")),
            vec![node]
        );
        assert!(
            layered
                .find_nodes_by_property("body", &Value::from("originaltoken"))
                .is_empty()
        );
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            layered.with_recorded_index_recovery(true, false, || -> Result<(), Error> {
                layered.set_node_property(node, "body", Value::from("unwinding"));
                panic!("Layered recovery unwinds");
            })
        }));
        assert!(unwind.is_err());
        assert_eq!(section.serialize()?, before);
        layered.set_node_property(node, "body", Value::from("ordinarytoken"));
        assert_ne!(section.serialize()?, before);
        let owner = crate::graph::write_permit::WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));
        let called = std::cell::Cell::new(false);
        assert!(
            crate::graph::write_permit::with_authority(&owner, || {
                layered.with_recorded_index_recovery(true, false, || {
                    called.set(true);
                    Ok(())
                })
            })
            .is_err()
        );
        assert!(!called.get());
        Ok(())
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn recorded_vector_recovery_keeps_cold_backing_and_layered_state()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::PhysicalIndexKey;
        use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind, VectorStoreSection};
        use grafeo_common::storage::Section;
        use grafeo_common::types::GraphPath;

        let vector = |value| Value::Vector(Arc::from([value, 0.0_f32]));
        let layered = empty_layered();
        let source = layered.overlay_store();
        source.sync_epoch(EpochId::new(1));
        source.add_vector_index(
            "Doc",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
                HnswConfig::new(2, DistanceMetric::Euclidean),
                29,
            ))),
        );
        #[cfg(feature = "text-index")]
        source.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(crate::index::text::InvertedIndex::new(
                Default::default(),
            ))),
        );
        let node = source.create_node_with_props(
            &["Doc"],
            [
                ("embedding", vector(1.0)),
                ("body", Value::from("original")),
                ("code", Value::Int64(1)),
            ],
        );
        let deleted = source.create_node_with_props(&["Doc"], [("embedding", vector(2.0))]);
        let cold = source.create_node_with_props(&["Doc"], [("embedding", vector(3.0))]);
        source.create_property_index("code");
        drop(source);
        layered
            .merge_overlay_temporal()
            .map_err(Error::InvalidValue)?;
        let overlay = layered.overlay_store();
        overlay.sync_epoch(EpochId::new(2));
        let base = layered.base_store_arc();
        let section = VectorStoreSection::from_views(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
            overlay
                .get_vector_index("Doc", "embedding")
                .ok_or("missing Vector")?,
        )]);
        let before = section.serialize()?;
        #[cfg(feature = "text-index")]
        let text_before = overlay
            .get_text_index("Doc", "body")
            .ok_or("missing Text")?
            .read()
            .encode_wal_birth()?;
        layered.with_recorded_index_recovery(cfg!(feature = "text-index"), true, || {
            layered.set_node_property(node, "embedding", vector(4.0));
            assert!(layered.remove_node_property(node, "embedding").is_some());
            layered.set_node_property(node, "code", Value::Int64(2));
            layered.set_node_property(node, "body", Value::from("replayed"));
            layered.replay_node_labels_at_epoch(node, EpochId::new(2), &[])?;
            layered.replay_node_labels_at_epoch(node, EpochId::new(2), &[ArcStr::from("Doc")])?;
            assert!(layered.delete_node(deleted));
            assert!(
                layered
                    .with_recorded_index_recovery(false, true, || Ok(()))
                    .is_err()
            );
            Ok(())
        })?;
        assert_eq!(section.serialize()?, before);
        #[cfg(feature = "text-index")]
        assert_eq!(
            overlay
                .get_text_index("Doc", "body")
                .ok_or("missing Text")?
                .read()
                .encode_wal_birth()?,
            text_before
        );
        assert!(Arc::ptr_eq(&base, &layered.base_store_arc()));
        assert!(Arc::ptr_eq(&overlay, &layered.overlay_store()));
        assert!(layered.is_node_dirty(node));
        assert!(layered.get_node(deleted).is_none());
        assert_eq!(
            layered.find_nodes_by_property("code", &Value::Int64(2)),
            vec![node]
        );
        assert!(
            layered
                .find_nodes_by_property("code", &Value::Int64(1))
                .is_empty()
        );
        let property = PropertyKey::new("embedding");
        assert!(
            layered.get_node_property(node, &property).is_none(),
            "hot removal must shadow cold backing"
        );
        assert_eq!(
            layered.get_node_property(cold, &property),
            Some(vector(3.0))
        );
        assert!(
            !overlay.contains_node_identity(cold),
            "cold backing need not hydrate"
        );
        assert!(
            layered
                .complete_node_property_history_for_key(node, "embedding")
                .contains(&(EpochId::new(1), vector(1.0)))
        );
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            layered.with_recorded_index_recovery(false, true, || -> Result<(), Error> {
                layered.set_node_property(node, "embedding", vector(5.0));
                std::panic::resume_unwind(Box::new("injected Layered Vector recovery unwind"));
            })
        }));
        assert!(unwind.is_err());
        assert_eq!(section.serialize()?, before);
        layered.set_node_property(node, "embedding", vector(6.0));
        assert_ne!(section.serialize()?, before);
        let owner = crate::graph::write_permit::WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));
        let called = std::cell::Cell::new(false);
        assert!(
            crate::graph::write_permit::with_authority(&owner, || {
                layered.with_recorded_index_recovery(false, true, || {
                    called.set(true);
                    Ok(())
                })
            })
            .is_err()
        );
        assert!(!called.get());
        Ok(())
    }

    fn empty_layered() -> LayeredStore {
        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        LayeredStore::new(base, 1000, 1000).unwrap()
    }

    #[test]
    fn compact_recompact_preserves_same_epoch_equal_node_property_events() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let epoch = EpochId::new(10);
        overlay.set_epoch(epoch);
        let node = overlay.create_node(&["Entity"]);
        overlay.set_node_property_at_epoch(node, "status", Value::from("same"), epoch);
        overlay.set_node_property_at_epoch(node, "status", Value::from("same"), epoch);
        let expected = vec![(epoch, Value::from("same")), (epoch, Value::from("same"))];
        assert_eq!(
            overlay.node_property_history_for_key(node, "status"),
            expected
        );

        layered
            .merge_overlay_temporal()
            .expect("compact duplicate node events");
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "status"),
            expected,
            "first compact must retain both authentic events"
        );
        layered
            .merge_overlay_temporal()
            .expect("recompact duplicate node events");
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "status"),
            expected,
            "recompact must neither collapse nor multiply authentic events"
        );
    }

    #[test]
    fn compact_recompact_preserves_same_epoch_equal_edge_property_events() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let epoch = EpochId::new(10);
        overlay.set_epoch(epoch);
        let src = overlay.create_node(&["Source"]);
        let dst = overlay.create_node(&["Target"]);
        let edge = overlay.create_edge(src, dst, "LINKS");
        overlay.set_edge_property_at_epoch(edge, "status", Value::from("same"), epoch);
        overlay.set_edge_property_at_epoch(edge, "status", Value::from("same"), epoch);
        let expected = vec![(epoch, Value::from("same")), (epoch, Value::from("same"))];
        assert_eq!(
            overlay
                .edge_property_history(edge)
                .into_iter()
                .find(|(key, _)| key.as_str() == "status")
                .map(|(_, versions)| versions),
            Some(expected.clone())
        );

        let assert_exact = |store: &LayeredStore, stage: &str| {
            assert_eq!(
                store
                    .edge_full_history(edge)
                    .properties
                    .get(&PropertyKey::new("status")),
                Some(&expected),
                "{stage} must retain both authentic events exactly once"
            );
        };
        layered
            .merge_overlay_temporal()
            .expect("compact duplicate edge events");
        assert_exact(&layered, "first compact");
        layered
            .merge_overlay_temporal()
            .expect("recompact duplicate edge events");
        assert_exact(&layered, "recompact");
    }

    #[test]
    fn compact_reconstruction_does_not_copy_contiguous_boundary_events() {
        use arcstr::ArcStr;

        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let boundary = EpochId::new(20);
        let subject = NodeId::new(40);
        let anchor = NodeId::new(41);
        let edge = EdgeId::new(50);

        overlay
            .restore_node_history_exact(
                subject,
                &[(created, Some(boundary)), (boundary, None)],
                &[
                    (created, vec![ArcStr::from("Entity")]),
                    (boundary, vec![ArcStr::from("Entity")]),
                ],
            )
            .expect("restore contiguous node lives");
        overlay
            .restore_node_history_exact(
                anchor,
                &[(created, None)],
                &[(created, vec![ArcStr::from("Anchor")])],
            )
            .expect("restore live endpoint");
        overlay
            .restore_edge_history_exact(
                edge,
                subject,
                anchor,
                "LINKS",
                &[(created, Some(boundary)), (boundary, None)],
            )
            .expect("restore contiguous edge lives");

        overlay.set_node_property_at_epoch(subject, "state", Value::from("first"), created);
        overlay.set_edge_property_at_epoch(edge, "state", Value::from("first"), created);
        overlay.set_node_property_at_epoch(subject, "state", Value::from("second"), boundary);
        overlay.set_node_property_at_epoch(subject, "state", Value::from("second"), boundary);
        overlay.set_edge_property_at_epoch(edge, "state", Value::from("second"), boundary);
        overlay.set_edge_property_at_epoch(edge, "state", Value::from("second"), boundary);
        overlay.set_epoch(boundary);

        let expected = vec![
            (created, Value::from("first")),
            (boundary, Value::from("second")),
            (boundary, Value::from("second")),
        ];
        let assert_exact = |store: &LayeredStore, stage: &str| {
            assert_eq!(
                store.complete_node_property_history_for_key(subject, "state"),
                expected,
                "{stage} must not copy node boundary events across both rows"
            );
            assert_eq!(
                store
                    .edge_full_history(edge)
                    .properties
                    .get(&PropertyKey::new("state")),
                Some(&expected),
                "{stage} must not copy edge boundary events across both rows"
            );
            assert_eq!(
                store.get_node_property_at_epoch(subject, &PropertyKey::new("state"), boundary,),
                Some(Value::from("second"))
            );
            assert_eq!(
                store
                    .get_edge_at_epoch(edge, boundary)
                    .and_then(|edge| edge.properties.get(&PropertyKey::new("state")).cloned()),
                Some(Value::from("second"))
            );
        };

        layered
            .merge_overlay_temporal()
            .expect("compact contiguous boundary events");
        assert_exact(&layered, "first compact");
        layered
            .merge_overlay_temporal()
            .expect("recompact contiguous boundary events");
        assert_exact(&layered, "recompact");
    }

    #[test]
    fn compact_closed_lifetime_does_not_manufacture_property_tombstones() {
        use arcstr::ArcStr;

        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let deleted = EpochId::new(20);
        let subject = NodeId::new(60);
        let anchor = NodeId::new(61);
        let edge = EdgeId::new(70);

        overlay
            .restore_node_history_exact(
                subject,
                &[(created, Some(deleted))],
                &[(created, vec![ArcStr::from("Ephemeral")])],
            )
            .expect("restore one closed node life");
        overlay
            .restore_node_history_exact(
                anchor,
                &[(created, None)],
                &[(created, vec![ArcStr::from("Anchor")])],
            )
            .expect("restore live endpoint");
        overlay
            .restore_edge_history_exact(edge, subject, anchor, "LINKS", &[(created, Some(deleted))])
            .expect("restore one closed edge life");
        overlay.set_node_property_at_epoch(subject, "state", Value::from("only-set"), created);
        overlay.set_edge_property_at_epoch(edge, "state", Value::from("only-set"), created);
        overlay.set_epoch(deleted);

        let expected = vec![(created, Value::from("only-set"))];
        let assert_exact = |store: &LayeredStore, stage: &str| {
            assert_eq!(
                store.complete_node_property_history_for_key(subject, "state"),
                expected,
                "{stage} must not turn structural closure into a property mutation"
            );
            assert_eq!(
                store
                    .edge_full_history(edge)
                    .properties
                    .get(&PropertyKey::new("state")),
                Some(&expected),
                "{stage} must not turn structural edge closure into a property mutation"
            );
            assert_eq!(
                store.get_node_property_at_epoch(
                    subject,
                    &PropertyKey::new("state"),
                    EpochId::new(15),
                ),
                Some(Value::from("only-set"))
            );
            assert!(store.get_node_at_epoch(subject, deleted).is_none());
            assert!(store.get_edge_at_epoch(edge, deleted).is_none());
        };

        layered
            .merge_overlay_temporal()
            .expect("compact closed lifetime without tombstone");
        assert_exact(&layered, "first compact");
        layered
            .merge_overlay_temporal()
            .expect("recompact closed lifetime without tombstone");
        assert_exact(&layered, "recompact");
    }

    #[cfg(all(feature = "text-index", feature = "vector-index"))]
    fn assert_hydration_preserves_exact_indexes(
        unwind: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::PhysicalIndexKey;
        use crate::index::text::{BM25Config, InvertedIndex, TextIndexSection};
        use crate::index::vector::{
            HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
            VectorStoreSection,
        };
        use grafeo_common::storage::section::Section;
        use grafeo_common::types::GraphPath;

        for quantization in [
            None,
            Some(QuantizationType::Scalar),
            Some(QuantizationType::Binary),
        ] {
            let layered = empty_layered();
            let source = layered.overlay_store();
            source.add_text_index(
                "Doc",
                "body",
                Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
            );
            let config = HnswConfig::new(2, DistanceMetric::Euclidean);
            let vector = match quantization {
                None => VectorIndexKind::Hnsw(HnswIndex::with_seed(config, 17)),
                Some(kind) => {
                    VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(config, kind, 17))
                }
            };
            source.add_vector_index("Doc", "embedding", Arc::new(vector));
            source.set_epoch(EpochId::new(10));
            let node = source.create_node(&["Doc"]);
            source.set_node_property(node, "body", Value::from("original retained text"));
            source.set_node_property(node, "embedding", Value::Vector(vec![1.0, 0.0].into()));
            source.set_epoch(EpochId::new(15));
            source.set_node_property(node, "body", Value::from("current document"));
            layered.merge_overlay_temporal()?;
            let overlay = layered.overlay_store();
            overlay.set_epoch(EpochId::new(20));
            let text = TextIndexSection::from_views(
                overlay
                    .text_index_entries()
                    .into_iter()
                    .map(|(_, view)| {
                        (
                            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                            view,
                        )
                    })
                    .collect(),
            );
            let vector = VectorStoreSection::from_views(
                overlay
                    .vector_index_entries()
                    .into_iter()
                    .map(|(_, view)| {
                        (
                            PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
                            view,
                        )
                    })
                    .collect(),
            );
            let text_before = text.serialize()?;
            let vector_before = vector.serialize()?;
            let retained_before = layered.text_search_visible(
                "Doc",
                "body",
                "original",
                10,
                EpochId::new(10),
                TransactionId::INVALID,
            )?;
            assert_eq!(retained_before.first().map(|(id, _)| *id), Some(node));
            if unwind {
                *layered.node_replay_event_hook.write() =
                    Some(Arc::new(|| panic!("injected hydration unwind")));
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || layered.set_node_property(node, "unrelated", Value::Bool(true))
                    ))
                    .is_err()
                );
                *layered.node_replay_event_hook.write() = None;
                assert!(!layered.is_node_dirty(node));
                assert!(overlay.get_node(node).is_none());
                assert_eq!(
                    vector.serialize()?,
                    vector_before,
                    "unwind changed Vector state: {quantization:?}"
                );
                assert_eq!(
                    text.serialize()?,
                    text_before,
                    "unwind erased retained Text state: {quantization:?}"
                );
            }
            layered.set_node_property(node, "unrelated", Value::Bool(true));
            assert!(layered.is_node_dirty(node));
            assert_eq!(
                vector.serialize()?,
                vector_before,
                "hydration changed Vector state: {quantization:?}"
            );
            assert_eq!(
                text.serialize()?,
                text_before,
                "hydration manufactured a Text event: {quantization:?}"
            );
            assert_eq!(
                layered.text_search_visible(
                    "Doc",
                    "body",
                    "original",
                    10,
                    EpochId::new(10),
                    TransactionId::INVALID
                )?,
                retained_before
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "text-index", feature = "vector-index"))]
    #[test]
    fn hydration_preserves_exact_indexes_on_success() -> Result<(), Box<dyn std::error::Error>> {
        assert_hydration_preserves_exact_indexes(false)
    }

    #[cfg(all(feature = "text-index", feature = "vector-index"))]
    #[test]
    fn hydration_preserves_exact_indexes_on_unwind_and_retry()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_hydration_preserves_exact_indexes(true)
    }

    #[test]
    fn base_node_promotion_replays_exact_history_and_reconciles_current_state() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let relabeled = EpochId::new(12);
        let changed = EpochId::new(15);
        let promoted = EpochId::new(20);

        overlay.set_epoch(created);
        let node = overlay.create_node(&["Doc"]);
        overlay.set_epoch(relabeled);
        assert!(overlay.add_label(node, "Draft"));
        overlay.set_epoch(changed);
        // Preserve the order of distinct same-epoch label mutations as well as
        // their final as-of image across cold promotion and re-compaction.
        assert!(overlay.add_label(node, "Published"));
        assert!(overlay.remove_label(node, "Doc"));
        // Same-epoch, same-value writes are distinct user events and must not be
        // collapsed while a cold row is promoted back into the hot overlay.
        overlay.set_node_property_at_epoch(node, "repeated", Value::from("same"), created);
        overlay.set_node_property_at_epoch(node, "repeated", Value::from("same"), created);
        overlay.set_node_property_at_epoch(node, "removed", Value::Int64(7), created);
        overlay.set_node_property_at_epoch(node, "removed", Value::Null, changed);
        overlay.set_node_property_at_epoch(node, "body", Value::from("old text"), created);
        overlay.set_node_property_at_epoch(node, "body", Value::from("current text"), changed);
        overlay.set_node_property_at_epoch(
            node,
            "embedding",
            Value::Vector(vec![9.0_f32, 9.0].into()),
            created,
        );
        overlay.set_node_property_at_epoch(
            node,
            "embedding",
            Value::Vector(vec![0.0_f32, 0.0].into()),
            changed,
        );
        overlay.set_node_property_at_epoch(node, "trigger", Value::Int64(1), created);
        overlay.set_epoch(changed);
        // Register complete whole-graph indexes before compaction. Promotion
        // is a physical copy, not a deferred index-build API.
        #[cfg(feature = "text-index")]
        let text_index = {
            use crate::index::text::{BM25Config, InvertedIndex};
            let mut index = InvertedIndex::new(BM25Config::default());
            index.insert_versioned(node, "current text", changed, None);
            let index = Arc::new(RwLock::new(index));
            overlay.add_text_index("Published", "body", Arc::clone(&index));
            index
        };
        #[cfg(feature = "vector-index")]
        let vector_index = {
            use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind};
            let index = HnswIndex::new(HnswConfig::new(2, DistanceMetric::Euclidean));
            index.insert(node, &[0.0, 0.0], &|_| None);
            let index = Arc::new(VectorIndexKind::Hnsw(index));
            overlay.add_vector_index("Published", "embedding", Arc::clone(&index));
            index
        };
        layered
            .merge_overlay_temporal()
            .expect("compact exact node property history");

        let expected_repeated = vec![
            (created, Value::from("same")),
            (created, Value::from("same")),
        ];
        let expected_removed = vec![(created, Value::Int64(7)), (changed, Value::Null)];
        let expected_body = vec![
            (created, Value::from("old text")),
            (changed, Value::from("current text")),
        ];
        let expected_embedding = vec![
            (created, Value::Vector(vec![9.0_f32, 9.0].into())),
            (changed, Value::Vector(vec![0.0_f32, 0.0].into())),
        ];
        let expected_labels = vec![
            (created, vec![ArcStr::from("Doc")]),
            (relabeled, vec![ArcStr::from("Doc"), ArcStr::from("Draft")]),
            (
                changed,
                vec![
                    ArcStr::from("Doc"),
                    ArcStr::from("Draft"),
                    ArcStr::from("Published"),
                ],
            ),
            (
                changed,
                vec![ArcStr::from("Draft"), ArcStr::from("Published")],
            ),
        ];
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "repeated"),
            expected_repeated
        );
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "removed"),
            expected_removed
        );
        assert_eq!(
            layered.node_structural_history(node).label_versions,
            expected_labels
        );

        let hot = layered.overlay_store();
        hot.set_epoch(promoted);
        hot.create_property_index("repeated");

        // This is the only real write at the promotion epoch. Every other
        // property version must be the exact cold-base history.
        layered.set_node_property(node, "trigger", Value::Int64(2));
        let promoted_structure = layered.complete_node_history(node);
        assert_eq!(promoted_structure.len(), 1);
        assert_eq!(promoted_structure[0].0, created);
        assert_eq!(promoted_structure[0].1, None);

        let hot = layered.overlay_store();
        assert_eq!(
            hot.node_label_history(node),
            expected_labels,
            "promotion must hydrate every cold label version without a restamp"
        );
        assert_eq!(
            hot.node_property_history_for_key(node, "repeated"),
            expected_repeated
        );
        assert_eq!(
            hot.node_property_history_for_key(node, "removed"),
            expected_removed
        );
        assert_eq!(
            hot.node_property_history_for_key(node, "body"),
            expected_body
        );
        assert_eq!(
            hot.node_property_history_for_key(node, "embedding"),
            expected_embedding
        );
        let expected_trigger = vec![(created, Value::Int64(1)), (promoted, Value::Int64(2))];
        assert_eq!(
            hot.node_property_history_for_key(node, "trigger"),
            expected_trigger,
            "promotion must not manufacture a trigger=1 version at the current epoch"
        );
        let at_create = layered
            .get_node_at_epoch(node, created)
            .expect("promoted node at creation");
        assert!(at_create.has_label("Doc"));
        assert!(!at_create.has_label("Draft"));
        let before_same_epoch_changes = layered
            .get_node_at_epoch(node, relabeled)
            .expect("promoted node after first label change");
        assert!(before_same_epoch_changes.has_label("Doc"));
        assert!(before_same_epoch_changes.has_label("Draft"));
        assert!(!before_same_epoch_changes.has_label("Published"));
        let after_same_epoch_changes = layered
            .get_node_at_epoch(node, changed)
            .expect("promoted node after ordered same-epoch label changes");
        assert!(!after_same_epoch_changes.has_label("Doc"));
        assert!(after_same_epoch_changes.has_label("Draft"));
        assert!(after_same_epoch_changes.has_label("Published"));

        assert_eq!(
            layered.find_nodes_by_property("repeated", &Value::from("same")),
            vec![node],
            "exact at-epoch replay must republish the current equality-index entry"
        );
        assert!(
            layered
                .find_nodes_by_property("removed", &Value::Int64(7))
                .is_empty(),
            "a replayed Null tombstone must remain absent from current lookup"
        );
        #[cfg(not(feature = "tiered-storage"))]
        assert_eq!(
            hot.node_record_props_count_for_test(node),
            Some(4),
            "props_count must count current non-Null values after exact replay"
        );
        #[cfg(feature = "text-index")]
        assert_eq!(
            text_index
                .read()
                .search("current", 10)
                .first()
                .map(|(id, _)| *id),
            Some(node),
            "hydration must preserve the transferred text posting"
        );
        #[cfg(feature = "vector-index")]
        {
            assert!(
                vector_index.contains(node),
                "hydration must preserve transferred vector membership"
            );
            let hits = layered.vector_search(
                Some("Published"),
                "embedding",
                &[0.0, 0.0],
                10,
                DistanceMetric::Euclidean,
            );
            assert_eq!(hits.first().map(|(id, _)| *id), Some(node));
        }

        layered
            .merge_overlay_temporal()
            .expect("recompact promoted exact node history");
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "repeated"),
            expected_repeated
        );
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "removed"),
            expected_removed
        );
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "body"),
            expected_body
        );
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "embedding"),
            expected_embedding
        );
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "trigger"),
            expected_trigger
        );
        assert_eq!(
            layered.node_structural_history(node).label_versions,
            expected_labels,
            "re-compaction must not duplicate exact hydrated label versions"
        );
    }

    #[test]
    fn base_edge_promotion_replays_exact_history_without_current_restamps() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let changed = EpochId::new(15);
        let promoted = EpochId::new(20);

        overlay.set_epoch(created);
        let src = overlay.create_node(&["Source"]);
        let dst = overlay.create_node(&["Target"]);
        let edge = overlay.create_edge(src, dst, "LINKS");
        overlay.set_edge_property_at_epoch(edge, "repeated", Value::from("same"), created);
        overlay.set_edge_property_at_epoch(edge, "repeated", Value::from("same"), created);
        overlay.set_edge_property_at_epoch(edge, "removed", Value::Int64(7), created);
        overlay.set_edge_property_at_epoch(edge, "removed", Value::Null, changed);
        overlay.set_edge_property_at_epoch(edge, "mixed", Value::Int64(9), created);
        overlay.set_edge_property_at_epoch(edge, "mixed", Value::from("nine"), changed);
        overlay.set_edge_property_at_epoch(edge, "trigger", Value::Int64(1), created);
        overlay.set_epoch(changed);
        layered
            .merge_overlay_temporal()
            .expect("compact exact edge property history");

        let expected_repeated = vec![
            (created, Value::from("same")),
            (created, Value::from("same")),
        ];
        let expected_removed = vec![(created, Value::Int64(7)), (changed, Value::Null)];
        let expected_mixed = vec![(created, Value::Int64(9)), (changed, Value::from("nine"))];
        let cold = layered.edge_full_history(edge);
        assert_eq!(
            cold.properties.get(&PropertyKey::new("repeated")),
            Some(&expected_repeated)
        );
        assert_eq!(
            cold.properties.get(&PropertyKey::new("removed")),
            Some(&expected_removed)
        );

        layered.overlay_store().set_epoch(promoted);
        layered.set_edge_property(edge, "trigger", Value::Int64(2));
        let promoted_structure = layered.complete_edge_history(edge);
        assert_eq!(promoted_structure.len(), 1);
        assert_eq!(promoted_structure[0].0, created);
        assert_eq!(promoted_structure[0].1, None);

        let hot = layered.overlay_store();
        let history = |key: &str| {
            hot.edge_property_history(edge)
                .into_iter()
                .find(|(candidate, _)| candidate.as_str() == key)
                .map(|(_, versions)| versions)
                .unwrap_or_default()
        };
        assert_eq!(history("repeated"), expected_repeated);
        assert_eq!(history("removed"), expected_removed);
        assert_eq!(history("mixed"), expected_mixed);
        let expected_trigger = vec![(created, Value::Int64(1)), (promoted, Value::Int64(2))];
        assert_eq!(
            history("trigger"),
            expected_trigger,
            "promotion must not manufacture a trigger=1 edge version at the current epoch"
        );

        layered
            .merge_overlay_temporal()
            .expect("recompact promoted exact edge history");
        let compacted = layered.edge_full_history(edge);
        assert_eq!(
            compacted.properties.get(&PropertyKey::new("repeated")),
            Some(&expected_repeated)
        );
        assert_eq!(
            compacted.properties.get(&PropertyKey::new("removed")),
            Some(&expected_removed)
        );
        assert_eq!(
            compacted.properties.get(&PropertyKey::new("mixed")),
            Some(&expected_mixed)
        );
        assert_eq!(
            compacted.properties.get(&PropertyKey::new("trigger")),
            Some(&expected_trigger)
        );
    }

    /// Track 2: no-property edges serve current 1-hop from the tight derived
    /// CSR, while packed adjacency retains only closed temporal tails. The
    /// duplicate validity/full-open-prefix representation is dropped. Must beat
    /// the prior 2.94× Option A tax vs an all-open CompactStore of the same
    /// topology.
    #[test]
    fn temporal_density_no_edge_props_beats_prior_tax() {
        const N: usize = 4_000;
        const DEG: usize = 2;
        let empty = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(empty, (N as u64) + 16, (N * DEG * 3) as u64 + 16).unwrap();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let mut ids = Vec::with_capacity(N);
        for i in 0..N {
            let id = overlay.create_node(&["Entity"]);
            overlay.set_node_property_at_epoch(
                id,
                "score",
                Value::Int64(i64::try_from(i).expect("density fixture index fits i64")),
                EpochId::new(10),
            );
            ids.push(id);
        }
        let mut replace = Vec::new();
        for (i, &src) in ids.iter().enumerate() {
            for k in 0..DEG {
                let dst = ids[(i + k + 1) % N];
                let eid = overlay.create_edge_versioned(
                    src,
                    dst,
                    "PROV",
                    EpochId::new(10),
                    TransactionId::SYSTEM,
                );
                if k % 2 == 0 {
                    replace.push((src, ids[(i + k + 2) % N], eid));
                }
            }
        }
        overlay.set_epoch(EpochId::new(20));
        for (src, new_dst, eid) in &replace {
            assert!(overlay.delete_edge(*eid));
            overlay.create_edge_versioned(
                *src,
                *new_dst,
                "PROV",
                EpochId::new(20),
                TransactionId::SYSTEM,
            );
        }
        overlay.set_epoch(EpochId::new(20));
        layered.merge_overlay_temporal().unwrap();

        assert_eq!(
            layered.out_degree(ids[0]),
            DEG,
            "current 1-hop must keep both open edges after packed prefix cut"
        );
        assert!(
            !layered.neighbors(ids[0], Direction::Outgoing).is_empty(),
            "current MATCH-style neighbors must not be empty after compact"
        );
        let base = layered.base_store_arc();
        let rt = base.rel_table("PROV").expect("PROV table");
        assert!(!rt.current_from_packed());
        assert!(rt.fwd().num_edges() > 0, "tight current CSR stays in RAM");
        assert_eq!(
            rt.packed_fwd().map(|p| p.num_open()),
            Some(0),
            "packed open prefix is stripped after slim"
        );
        assert!(
            rt.packed_bwd().is_none(),
            "fat backward twin is not kept when there are no edge properties"
        );

        let temporal_b = base.memory_bytes();
        let packed = rt.packed_fwd().expect("packed");
        let (open_from_b, open_ids_b, closed_ord_b) = rt.temporal_sidecar_memory_bytes();
        let open_ids = rt.open_edge_ids();
        let open_id_min = open_ids.iter().map(|id| id.as_u64()).min().unwrap_or(0);
        let open_id_max = open_ids.iter().map(|id| id.as_u64()).max().unwrap_or(0);
        eprintln!(
            "bytes total={temporal_b} rel={} packed_fwd={} packed_bwd={} empty_fwd={} empty_bwd={} nodes={} open_from={} open_ids={} closed_ord={} open_id_range={}..={}",
            rt.memory_bytes(),
            packed.memory_bytes(),
            rt.packed_bwd()
                .map_or(0, super::super::csr::PackedOpenAdjacency::memory_bytes),
            rt.fwd().memory_bytes(),
            rt.bwd().map_or(0, |b| b.memory_bytes()),
            base.node_table("Entity").map_or(0, |nt| nt.memory_bytes()),
            open_from_b,
            open_ids_b,
            closed_ord_b,
            open_id_min,
            open_id_max,
        );
        let open = {
            let s = LpgStore::new().unwrap();
            let mut oids = Vec::with_capacity(N);
            for i in 0..N {
                let id = s.create_node(&["Entity"]);
                s.set_node_property(
                    id,
                    "score",
                    Value::Int64(i64::try_from(i).expect("density fixture index fits i64")),
                );
                oids.push(id);
            }
            for (i, &src) in oids.iter().enumerate() {
                for k in 0..DEG {
                    s.create_edge(src, oids[(i + k + 1) % N], "PROV");
                }
            }
            from_graph_store_preserving_ids(&s).unwrap()
        };
        let open_b = open.memory_bytes();
        let open_rel = open
            .rel_table("PROV")
            .expect("open PROV table")
            .memory_bytes();
        let open_nodes = open
            .node_table("Entity")
            .expect("open Entity table")
            .memory_bytes();
        let open_other = open_b.saturating_sub(open_rel + open_nodes);
        eprintln!(
            "open bytes total={open_b} rel={open_rel} nodes={open_nodes} id_maps_and_sidecars={open_other}"
        );
        let ratio = temporal_b as f64 / open_b as f64;
        eprintln!(
            "temporal density {temporal_b} / {open_b} B = {ratio:.2}× (same-schema all-open; gate ≤1.25×, prior 2.94×)"
        );
        assert!(
            ratio <= 1.25,
            "Track 2 density gate is ≤1.25× all-open, got {ratio:.2}×"
        );
    }

    /// Delete-then-recreate same endpoints: two disjoint intervals, gap is empty.
    #[test]
    fn merge_overlay_temporal_delete_then_recreate_no_resurrection() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let first =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(first, "w", Value::Int64(1), EpochId::new(10));
        overlay.set_epoch(EpochId::new(20));
        assert!(overlay.delete_edge_at_epoch(first, EpochId::new(20)));
        overlay.set_epoch(EpochId::new(30));
        let second =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(30), TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(second, "w", Value::Int64(2), EpochId::new(30));
        overlay.set_epoch(EpochId::new(30));
        assert_ne!(first, second);

        layered.merge_overlay_temporal().unwrap();

        assert!(layered.get_edge(first).is_none());
        assert!(layered.get_edge(second).is_some());
        assert_eq!(
            layered.neighbors(a, Direction::Outgoing),
            vec![b],
            "current neighbors must show only the recreated edge"
        );

        assert!(layered.get_edge_at_epoch(first, EpochId::new(15)).is_some());
        assert!(
            layered
                .get_edge_at_epoch(second, EpochId::new(15))
                .is_none()
        );
        assert!(
            layered.get_edge_at_epoch(first, EpochId::new(25)).is_none(),
            "deleted first life must not appear after D"
        );
        assert!(
            layered
                .get_edge_at_epoch(second, EpochId::new(25))
                .is_none(),
            "recreated edge must not resurrect in the delete-recreate gap"
        );
        assert!(layered.get_edge_at_epoch(first, EpochId::new(35)).is_none());
        let live = layered
            .get_edge_at_epoch(second, EpochId::new(35))
            .expect("second life visible after recreate");
        assert_eq!(
            live.properties.get(&PropertyKey::new("w")),
            Some(&Value::Int64(2))
        );

        let rows = layered.base_store_arc().structural_edge_rows();
        assert_eq!(rows.len(), 2, "both lifetimes retained as disjoint rows");
        let first_iv = layered
            .base_store_arc()
            .edge_validity(first)
            .expect("first life retained");
        let second_iv = layered
            .base_store_arc()
            .edge_validity(second)
            .expect("second life retained");
        assert_eq!(
            first_iv,
            grafeo_common::types::EpochInterval::closed(EpochId::new(10), EpochId::new(20),)
        );
        assert!(second_iv.is_open());
        assert_eq!(second_iv.from(), EpochId::new(30));
        assert!(!first_iv.contains(EpochId::new(25)));
        assert!(!second_iv.contains(EpochId::new(25)));

        // Task 4: current CSR / packed prefix is open-only; closed life is as-of only.
        let base = layered.base_store_arc();
        let rt = base.rel_table("KNOWS").expect("KNOWS table after merge");
        assert_eq!(
            rt.num_edges(),
            1,
            "derived current CSR must contain only the open recreate"
        );
        let packed = rt
            .packed_fwd()
            .expect("Option A packed adjacency after merge");
        assert_eq!(packed.num_versions(), 2);
        assert_eq!(packed.num_open(), 1);
        assert_eq!(
            layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::PENDING),
            vec![b]
        );
        assert_eq!(
            layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15)),
            vec![b],
            "as-of during the first life"
        );
        assert!(
            layered
                .neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(25))
                .is_empty(),
            "gap between delete and recreate must not resurrect"
        );
        assert_eq!(
            layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(35)),
            vec![b]
        );
    }

    #[test]
    fn merge_current_csr_excludes_deleted_edges_packed_prefix() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        let live =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        let dead =
            overlay.create_edge_versioned(a, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(20));
        assert!(overlay.delete_edge_at_epoch(dead, EpochId::new(20)));
        overlay.set_epoch(EpochId::new(20));
        layered.merge_overlay_temporal().unwrap();

        assert_eq!(layered.neighbors(a, Direction::Outgoing), vec![b]);
        assert!(layered.get_edge(dead).is_none());
        assert!(layered.get_edge(live).is_some());

        let base = layered.base_store_arc();
        let rt = base.rel_table("KNOWS").expect("KNOWS");
        assert_eq!(
            rt.num_edges(),
            1,
            "deleted edge must not sit in current CSR"
        );
        let packed = rt.packed_fwd().expect("packed after merge");
        assert_eq!(packed.num_open(), 0);
        assert_eq!(packed.num_versions(), 1);
        assert_eq!(rt.fwd().num_edges(), 1);
        assert!(
            layered.get_edge_at_epoch(dead, EpochId::new(15)).is_some(),
            "deleted edge remains addressable for as-of"
        );
        let mut asof = layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        asof.sort_unstable();
        let mut expect = vec![b, c];
        expect.sort_unstable();
        assert_eq!(asof, expect);
        assert_eq!(
            layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(25)),
            vec![b]
        );

        let mut current_ids: Vec<EdgeId> = layered
            .edges_at_epoch(EpochId::PENDING)
            .into_iter()
            .map(|e| e.id)
            .collect();
        current_ids.sort_unstable();
        assert_eq!(current_ids, vec![live], "PENDING edges == current CSR");
        let mut asof_ids: Vec<EdgeId> = layered
            .edges_at_epoch(EpochId::new(15))
            .into_iter()
            .map(|e| e.id)
            .collect();
        asof_ids.sort_unstable();
        let mut expect_ids = vec![live, dead];
        expect_ids.sort_unstable();
        assert_eq!(asof_ids, expect_ids, "as-of before delete includes closed");
        assert_eq!(
            layered
                .edges_at_epoch(EpochId::new(25))
                .into_iter()
                .map(|e| e.id)
                .collect::<Vec<_>>(),
            vec![live],
            "closed edge absent after delete epoch"
        );
    }

    /// Compact, then overlay-delete without rematching: packed CSR stays open,
    /// so as-of must apply the hot edge tombstone (same visibility as
    /// [`LayeredStore::get_edge_at_epoch`]).
    #[test]
    fn neighbors_at_epoch_overlay_delete_without_remerge() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        let _live =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        let dead =
            overlay.create_edge_versioned(a, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(10));
        layered.merge_overlay_temporal().unwrap();

        layered.overlay_store().set_epoch(EpochId::new(20));
        assert!(layered.delete_edge(dead));

        let pending = layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::PENDING);
        assert_eq!(pending, vec![b]);
        assert_eq!(
            layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(20)),
            pending,
            "e_del must match PENDING (deleted dest gone) without rematch"
        );

        let mut at15 = layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        at15.sort_unstable();
        let mut expect = vec![b, c];
        expect.sort_unstable();
        assert_eq!(at15, expect, "as-of before delete still includes dest");
        assert!(layered.get_edge_at_epoch(dead, EpochId::new(15)).is_some());
        assert!(layered.get_edge_at_epoch(dead, EpochId::new(20)).is_none());
    }

    /// A neighbor node deleted at 30 must still appear when viewing 15.
    #[test]
    fn neighbors_at_epoch_node_tombstone_is_epoch_aware() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.create_edge_versioned(a, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(10));
        layered.merge_overlay_temporal().unwrap();

        layered.overlay_store().set_epoch(EpochId::new(30));
        assert!(layered.delete_node(c));

        let mut at15 = layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        at15.sort_unstable();
        let mut expect = vec![b, c];
        expect.sort_unstable();
        assert_eq!(
            at15, expect,
            "node tombstone at 30 must not hide dest at as-of 15"
        );
        assert_eq!(
            layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(30)),
            vec![b],
            "dest deleted at 30 is absent at/after that epoch"
        );
    }

    #[test]
    fn merge_overlay_temporal_source_to_entity_does_not_oob() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let mut ents = Vec::new();
        for _ in 0..8 {
            ents.push(overlay.create_node(&["Entity"]));
        }
        let s0 = overlay.create_node(&["Source"]);
        let s1 = overlay.create_node(&["Source"]);
        for (i, &ent) in ents.iter().enumerate() {
            overlay.create_edge(if i % 2 == 0 { s0 } else { s1 }, ent, "OBSERVED_BY");
            if i + 1 < ents.len() {
                overlay.create_edge(ent, ents[i + 1], "DERIVED_FROM");
            }
        }
        overlay.set_epoch(EpochId::new(20));
        layered
            .merge_overlay_temporal()
            .expect("Source→Entity compact must size the backward CSR to dests");
        let mut incoming = Vec::new();
        layered.fill_neighbors_of_types_at_epoch(
            ents[7],
            Direction::Incoming,
            EpochId::new(15),
            &[String::from("OBSERVED_BY")],
            &mut incoming,
        );
        assert_eq!(incoming, vec![s1]);
        let edges = layered.edges_at_epoch(EpochId::new(15));
        let scrub = layered.edge_scrub_at_epoch(EpochId::new(15));
        let mut from_edges: Vec<_> = edges
            .iter()
            .map(|e| (e.id, e.src, e.dst, e.edge_type.clone()))
            .collect();
        let mut from_scrub = Vec::new();
        for frame in &scrub {
            for i in 0..frame.edge_ids.len() {
                from_scrub.push((
                    frame.edge_ids[i],
                    frame.src_ids[i],
                    frame.dst_ids[i],
                    frame.edge_type.clone(),
                ));
            }
        }
        from_edges.sort();
        from_scrub.sort();
        assert_eq!(
            from_edges, from_scrub,
            "columnar edge scrub must match edges_at_epoch"
        );
    }

    #[test]
    fn compact_base_column_is_property_seek_after_merge() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        overlay.set_node_property(a, "name", Value::from("Alix"));
        overlay.set_epoch(EpochId::new(10));
        layered.merge_overlay_temporal().unwrap();
        assert!(
            !layered.has_property_index("name"),
            "a cold column without a registered logical index must not opt into indexed execution"
        );
        assert_eq!(
            layered.find_nodes_by_property("name", &Value::from("Alix")),
            vec![a]
        );
    }

    #[test]
    fn layered_registered_property_index_delegates_historical_predicates() {
        use crate::graph::{GraphStoreSearch, PropertyIndexPredicate, PropertyIndexRequest};

        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let epoch = EpochId::new(10);
        overlay.set_epoch(epoch);
        let first = overlay.create_node(&["Entity"]);
        overlay.set_node_property(first, "score", Value::Int64(10));
        let second = overlay.create_node(&["Entity"]);
        overlay.set_node_property(second, "score", Value::Int64(20));
        overlay.create_property_index("score");
        layered
            .merge_overlay_temporal()
            .expect("transfer registered property index");

        assert_eq!(
            layered
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "score",
                    predicate: PropertyIndexPredicate::Equal(&Value::Int64(10)),
                    epoch,
                    transaction_id: None,
                })
                .unwrap(),
            Some(vec![first])
        );
        let values = [Value::Int64(10), Value::Int64(20)];
        assert_eq!(
            layered
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "score",
                    predicate: PropertyIndexPredicate::In(&values),
                    epoch,
                    transaction_id: None,
                })
                .unwrap(),
            Some(vec![first, second])
        );
        assert_eq!(
            layered
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "score",
                    predicate: PropertyIndexPredicate::Range {
                        min: Some(&Value::Int64(10)),
                        max: Some(&Value::Int64(20)),
                        min_inclusive: true,
                        max_inclusive: false,
                    },
                    epoch,
                    transaction_id: None,
                })
                .unwrap(),
            Some(vec![first])
        );
    }

    #[test]
    fn merge_overlay_temporal_create_only_edge_is_open() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let eid =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(eid, "w", Value::Int64(1), EpochId::new(10));
        overlay.set_epoch(EpochId::new(10));
        layered.merge_overlay_temporal().unwrap();

        let iv = layered
            .base_store_arc()
            .edge_validity(eid)
            .expect("create-only row");
        assert!(iv.is_open());
        assert_eq!(iv.from(), EpochId::new(10));
        assert!(layered.get_edge_at_epoch(eid, EpochId::new(15)).is_some());
        assert!(layered.get_edge_at_epoch(eid, EpochId::new(5)).is_none());
        assert!(layered.get_edge(eid).is_some());
    }

    #[test]
    fn merge_overlay_temporal_create_delete_retains_closed_row() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let eid =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(40));
        assert!(overlay.delete_edge_at_epoch(eid, EpochId::new(40)));
        overlay.set_epoch(EpochId::new(40));
        layered.merge_overlay_temporal().unwrap();

        assert!(layered.get_edge(eid).is_none());
        assert_eq!(layered.neighbors(a, Direction::Outgoing).len(), 0);
        assert!(layered.get_edge_at_epoch(eid, EpochId::new(15)).is_some());
        assert!(layered.get_edge_at_epoch(eid, EpochId::new(40)).is_none());
        let iv = layered
            .base_store_arc()
            .edge_validity(eid)
            .expect("closed row retained");
        assert_eq!(
            iv,
            grafeo_common::types::EpochInterval::closed(EpochId::new(10), EpochId::new(40))
        );

        let mut filled = Vec::new();
        layered.base_store_arc().fill_neighbors_at_epoch(
            a,
            Direction::Outgoing,
            EpochId::new(15),
            &mut filled,
        );
        let mut via_vec =
            layered
                .base_store_arc()
                .neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        filled.sort_unstable();
        via_vec.sort_unstable();
        assert_eq!(filled, via_vec);
        assert!(filled.contains(&b));
    }

    #[test]
    fn packed_zero_width_edge_identity_survives_two_temporal_merges() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let created = EpochId::new(10);
        let closed = EpochId::new(20);
        overlay.set_epoch(created);
        let source = overlay.create_node(&["Source"]);
        let target = overlay.create_node(&["Target"]);
        let survivor_target = overlay.create_node(&["Target"]);
        let survivor = overlay.create_edge(source, survivor_target, "KNOWS");
        assert!(survivor.is_valid());
        overlay.set_epoch(closed);
        let zero = overlay.create_edge(source, target, "KNOWS");
        assert!(zero.is_valid());
        assert!(overlay.delete_edge_at_epoch(zero, closed));
        // A closed-only relationship type cannot use the live KNOWS table.
        // Its property-bearing sidecar exercises the other identity source.
        let sidecar = overlay.create_edge(target, source, "CLOSED_ONLY");
        assert!(sidecar.is_valid());
        overlay.set_edge_property_at_epoch(sidecar, "weight", Value::Int64(7), closed);
        assert!(overlay.delete_edge_at_epoch(sidecar, closed));
        drop(overlay);

        for merge in 1..=2 {
            layered.merge_overlay_temporal().unwrap();
            let base = layered.base_store_arc();
            assert!(
                base.closed_edge_row(zero).is_none(),
                "merge {merge}: propertyless zero-width edge must not use a sidecar"
            );
            assert_eq!(
                base.packed_closed_validity(zero),
                Some(grafeo_common::types::EpochInterval::closed(closed, closed)),
                "the surviving KNOWS edge must keep the zero-width edge packed"
            );
            assert!(base.closed_edge_row(sidecar).is_some());
            assert!(base.packed_closed_validity(sidecar).is_none());
            for node in [source, target, survivor_target] {
                assert!(base.get_node(node).is_some());
            }
            for (id, src, dst, edge_type) in [
                (zero, source, target, "KNOWS"),
                (sidecar, target, source, "CLOSED_ONLY"),
            ] {
                let history = layered.complete_edge_history(id);
                assert_eq!(history.len(), 1, "merge {merge}: retained edge {id}");
                let (birth, death, edge) = &history[0];
                assert_eq!((*birth, *death), (closed, Some(closed)));
                assert_eq!(edge.id, id);
                assert_eq!((edge.src, edge.dst), (src, dst));
                assert_eq!(edge.edge_type.as_str(), edge_type);
                assert!(edge.properties.is_empty());
                for cut in [created, closed, EpochId::new(30)] {
                    assert!(base.get_edge_at_epoch(id, cut).is_none());
                    assert!(layered.get_edge_at_epoch(id, cut).is_none());
                }
                assert!(layered.get_edge(id).is_none());
            }
            assert!(layered.complete_edge_property_history(zero).is_empty());
            assert_eq!(
                layered.complete_edge_property_history(sidecar),
                vec![(
                    PropertyKey::new("weight"),
                    vec![(closed, Value::Int64(7)), (closed, Value::Null)]
                )]
            );
            assert!(layered.get_edge(survivor).is_some());
            assert_eq!(layered.node_count(), 3);
            assert_eq!(layered.edge_count(), 1);
            assert_eq!(
                layered.neighbors(source, Direction::Outgoing),
                vec![survivor_target]
            );
            assert!(layered.neighbors(target, Direction::Outgoing).is_empty());
            layered.overlay_store().set_epoch(EpochId::new(30));
        }
    }

    #[test]
    fn packed_placed_closed_row_is_not_in_hop_index() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        let ab =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        let _ac =
            overlay.create_edge_versioned(a, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(40));
        assert!(overlay.delete_edge(ab));
        overlay.set_epoch(EpochId::new(40));
        layered.merge_overlay_temporal().unwrap();

        assert!(
            layered.base_store_arc().closed_hop_index_len() == 0,
            "closed AB is on packed KNOWS (AC still open); hop sidecar must be empty"
        );
        assert_eq!(
            layered.base_store_arc().closed_sidecar_len(),
            0,
            "structure-only packed-placed AB must not duplicate in closed_edges"
        );
        assert!(
            layered.get_edge_at_epoch(ab, EpochId::new(15)).is_some(),
            "as-of reconstructs AB from packed fat ids"
        );
        assert!(layered.get_edge(ab).is_none());
        let at15 =
            layered
                .base_store_arc()
                .neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        assert!(at15.contains(&b) && at15.contains(&c));
        let now = layered.base_store_arc().neighbors(a, Direction::Outgoing);
        assert!(!now.contains(&b) && now.contains(&c));
    }

    #[test]
    fn second_merge_after_structure_only_packed_close() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        let d = overlay.create_node(&["Person"]);
        let ab =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        let _ac =
            overlay.create_edge_versioned(a, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(40));
        assert!(overlay.delete_edge(ab));
        overlay.set_epoch(EpochId::new(40));
        layered.merge_overlay_temporal().unwrap();
        assert_eq!(layered.base_store_arc().closed_sidecar_len(), 0);

        layered.overlay_store().set_epoch(EpochId::new(50));
        let ad =
            layered.create_edge_versioned(a, d, "KNOWS", EpochId::new(50), TransactionId::SYSTEM);
        assert!(
            ad.is_valid(),
            "Layered write must promote both cold endpoints"
        );
        layered.merge_overlay_temporal().unwrap();

        let at15 = layered.neighbors_at_epoch(a, Direction::Outgoing, EpochId::new(15));
        assert!(at15.contains(&b) && at15.contains(&c));
        assert!(!at15.contains(&d));
        let now = layered.neighbors(a, Direction::Outgoing);
        assert!(
            !now.contains(&b) && now.contains(&c) && now.contains(&d),
            "current neighbors after second merge: {now:?} b={b:?} c={c:?} d={d:?}"
        );
        assert!(layered.get_edge_at_epoch(ab, EpochId::new(15)).is_some());
    }

    #[test]
    fn merge_overlay_temporal_property_change_then_delete() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let eid =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(eid, "w", Value::Int64(1), EpochId::new(10));
        overlay.set_edge_property_at_epoch(eid, "w", Value::Int64(2), EpochId::new(20));
        overlay.set_epoch(EpochId::new(30));
        assert!(overlay.delete_edge_at_epoch(eid, EpochId::new(30)));
        overlay.set_epoch(EpochId::new(30));
        layered.merge_overlay_temporal().unwrap();

        let key = PropertyKey::new("w");
        let at = |ep: u64| {
            layered
                .get_edge_at_epoch(eid, EpochId::new(ep))
                .and_then(|e| e.properties.get(&key).cloned())
        };
        assert_eq!(at(15), Some(Value::Int64(1)));
        assert_eq!(at(25), Some(Value::Int64(2)));
        assert_eq!(at(35), None);
        assert!(layered.get_edge(eid).is_none());
        assert!(
            layered.base_store_arc().closed_sidecar_len() >= 1,
            "property history must stay on the closed_edges sidecar"
        );
    }

    #[test]
    fn merge_overlay_temporal_excludes_uncommitted_edges() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let committed =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        let tx = TransactionId::new(7);
        let pending = overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), tx);
        overlay.set_epoch(EpochId::new(10));
        layered.merge_overlay_temporal().unwrap();

        assert!(layered.get_edge(committed).is_some());
        assert!(layered.get_edge(pending).is_none());
        assert!(
            layered.base_store_arc().edge_validity(pending).is_none(),
            "uncommitted overlay edge must not be folded"
        );
        assert!(
            !layered
                .base_store_arc()
                .closed_edge_ids()
                .contains(&pending)
        );
    }

    /// Property-write promotion restamps the overlay create at `current_epoch()`.
    /// The second temporal merge must keep the original create `from`, or as-of
    /// between create and promote disappears.
    #[test]
    fn merge_overlay_temporal_second_merge_preserves_create_epoch_after_promote() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let eid =
            overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_edge_property_at_epoch(eid, "w", Value::Int64(1), EpochId::new(10));
        overlay.set_epoch(EpochId::new(10));
        layered.merge_overlay_temporal().unwrap();

        let after_first = layered
            .base_store_arc()
            .edge_validity(eid)
            .expect("first merge folds create");
        assert!(after_first.is_open());
        assert_eq!(after_first.from(), EpochId::new(10));
        assert!(layered.get_edge_at_epoch(eid, EpochId::new(15)).is_some());

        // Promote at a later epoch (property write copies the base edge into the
        // overlay at `current_epoch()`, not the original create).
        layered.overlay_store().set_epoch(EpochId::new(20));
        layered.set_edge_property(eid, "w", Value::Int64(2));
        // Historical readers use this window: as-of must work *before* rematch/second merge.
        assert!(layered.get_edge_at_epoch(eid, EpochId::new(15)).is_some());
        layered.merge_overlay_temporal().unwrap();

        assert!(
            layered.get_edge_at_epoch(eid, EpochId::new(15)).is_some(),
            "as-of before the promote epoch must still see the edge"
        );
        let iv = layered
            .base_store_arc()
            .edge_validity(eid)
            .expect("promoted edge remains in the base");
        assert!(
            iv.contains(EpochId::new(15)),
            "validity {iv:?} must contain as-of 15"
        );
        assert_eq!(
            iv.from(),
            EpochId::new(10),
            "second merge must not restamp create to the promote epoch"
        );
        assert!(iv.is_open());
        assert!(layered.get_edge(eid).is_some());
        assert_eq!(
            layered.get_edge_property(eid, &PropertyKey::new("w")),
            Some(Value::Int64(2))
        );
    }

    /// SP2 slice 5 (incremental): nodes committed after the retention boundary
    /// stay whole and hot in the overlay; older nodes compact to the cold base.
    /// Both serve correct as-of reads through the existing per-node routing.
    #[test]
    fn merge_overlay_temporal_retaining_keeps_recent_nodes_hot() {
        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(base, 1000, 1000).unwrap();
        let overlay = layered.overlay_store();
        let key = PropertyKey::new("score");
        // A: committed at 10, 20 (all <= 25 -> cold). B: 10, 30 (30 > 25 -> hot).
        let a = overlay.create_node(&["Item"]);
        overlay.set_node_property_at_epoch(a, "score", Value::Int64(1), EpochId::new(10));
        overlay.set_node_property_at_epoch(a, "score", Value::Int64(2), EpochId::new(20));
        let b = overlay.create_node(&["Item"]);
        overlay.set_node_property_at_epoch(b, "score", Value::Int64(10), EpochId::new(10));
        overlay.set_node_property_at_epoch(b, "score", Value::Int64(30), EpochId::new(30));
        overlay.set_epoch(EpochId::new(30));

        layered
            .merge_overlay_temporal_retaining(EpochId::new(25))
            .unwrap();

        let at = |id, e: u64| {
            layered
                .get_node_at_epoch(id, EpochId::new(e))
                .and_then(|n| n.properties.get(&key).cloned())
        };
        // A is cold (compacted to the base): as-of preserved, served by the base.
        assert_eq!(at(a, 15), Some(Value::Int64(1)));
        assert_eq!(at(a, 25), Some(Value::Int64(2)));
        assert!(layered.base_store_arc().get_node(a).is_some());
        // B stayed hot (whole history in the overlay): the recent (>25) version reads back.
        assert_eq!(at(b, 15), Some(Value::Int64(10)));
        assert_eq!(at(b, 35), Some(Value::Int64(30)));
        // Current reads and counts intact.
        assert_eq!(layered.get_node_property(a, &key), Some(Value::Int64(2)));
        assert_eq!(layered.get_node_property(b, &key), Some(Value::Int64(30)));
        assert_eq!(layered.node_count(), 2);
    }

    /// The layered columnar scrub combines the cold base, the hot overlay
    /// (override), and nodes created since the last merge (append).
    #[test]
    fn layered_scrub_at_epoch_combines_base_overlay_and_new_nodes() {
        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(base, 1000, 1000).unwrap();
        let overlay = layered.overlay_store();
        let key = PropertyKey::new("score");
        let a = overlay.create_node(&["Item"]); // cold (history <= 20)
        overlay.set_node_property_at_epoch(a, "score", Value::Int64(1), EpochId::new(10));
        overlay.set_node_property_at_epoch(a, "score", Value::Int64(2), EpochId::new(20));
        let b = overlay.create_node(&["Item"]); // hot (version at 30 > 25)
        overlay.set_node_property_at_epoch(b, "score", Value::Int64(10), EpochId::new(10));
        overlay.set_node_property_at_epoch(b, "score", Value::Int64(30), EpochId::new(30));
        overlay.set_epoch(EpochId::new(30));
        layered
            .merge_overlay_temporal_retaining(EpochId::new(25))
            .unwrap();

        // A node created AFTER the merge: overlay-only, absent from the base.
        let overlay2 = layered.overlay_store();
        let c = overlay2.create_node(&["Item"]);
        overlay2.set_node_property_at_epoch(c, "score", Value::Int64(100), EpochId::new(40));
        overlay2.set_epoch(EpochId::new(40));

        let val_at = |frames: &[NodeTableScrub], id: NodeId| -> Option<Value> {
            for f in frames {
                if let Some(i) = f.node_ids.iter().position(|n| *n == id) {
                    return f.columns.get(&key).and_then(|col| col[i].clone());
                }
            }
            None
        };

        // At 15: a from the cold base, b from the hot overlay, c not yet created.
        let f15 = layered.scrub_at_epoch(EpochId::new(15));
        assert_eq!(val_at(&f15, a), Some(Value::Int64(1)));
        assert_eq!(val_at(&f15, b), Some(Value::Int64(10)));
        assert_eq!(val_at(&f15, c), None);

        // At 45: b's recent (>25) value comes from the overlay (the base only
        // holds <=25), and c is appended with its committed value.
        let f45 = layered.scrub_at_epoch(EpochId::new(45));
        assert_eq!(val_at(&f45, a), Some(Value::Int64(2)));
        assert_eq!(val_at(&f45, b), Some(Value::Int64(30)));
        assert_eq!(val_at(&f45, c), Some(Value::Int64(100)));
    }

    /// Regression: `scrub_at_epoch` must include a property the overlay added to a
    /// base node *after* its last compaction. The override loop previously only
    /// visited columns already present in the cold base, silently dropping keys
    /// new to the table — so the columnar scrub disagreed with `get_node_at_epoch`.
    #[test]
    fn layered_scrub_at_epoch_includes_overlay_added_property_on_base_node() {
        let base = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(base, 1000, 1000).unwrap();
        let overlay = layered.overlay_store();
        let a = overlay.create_node(&["Item"]);
        overlay.set_node_property_at_epoch(a, "score", Value::Int64(1), EpochId::new(10));
        overlay.set_epoch(EpochId::new(10));
        layered.merge_overlay_temporal().unwrap(); // A -> cold base, not dirty

        // Add a property new to the table (`tag`) to base node A; this promotes A
        // into the overlay (now dirty) with a key the base frame has no column for.
        layered.overlay_store().set_epoch(EpochId::new(20));
        layered.set_node_property(a, "tag", Value::from("x"));

        let frames = layered.scrub_at_epoch(EpochId::new(20));
        let find = |key: &PropertyKey| -> Option<Value> {
            frames.iter().find_map(|f| {
                f.node_ids
                    .iter()
                    .position(|n| *n == a)
                    .and_then(|i| f.columns.get(key).and_then(|c| c[i].clone()))
            })
        };
        assert_eq!(find(&PropertyKey::new("score")), Some(Value::Int64(1)));
        assert_eq!(find(&PropertyKey::new("tag")), Some(Value::from("x")));
    }

    #[test]
    fn test_create_node_in_overlay() {
        let layered = build_test_layered();
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        assert_eq!(layered.node_count(), 4);
        let node = layered.get_node(vincent).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );
    }

    #[test]
    fn test_delete_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let deleted = layered.delete_node(persons[0]);
        assert!(deleted);
        assert!(layered.get_node(persons[0]).is_none());

        let remaining_persons = layered.nodes_by_label("Person");
        assert_eq!(remaining_persons.len(), 1);
        assert_eq!(layered.node_count(), 2);
    }

    #[test]
    fn test_modify_base_node_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Original value.
        let original_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert!(matches!(original_age, Value::Int64(_)));

        // Modify: this should promote the node to the overlay.
        layered.set_node_property(first, "age", Value::Int64(99));

        let new_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert_eq!(new_age, Value::Int64(99));
    }

    #[test]
    fn test_create_edge_between_base_and_overlay() {
        let layered = build_test_layered();
        let paris = layered.create_node(&["City"]);
        layered.set_node_property(paris, "name", Value::from("Paris"));

        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // Create cross-layer edge.
        let eid = layered.create_edge(first_person, paris, "VISITS");
        assert!(layered.get_edge(eid).is_some());

        let edge = layered.get_edge(eid).unwrap();
        assert_eq!(edge.src, first_person);
        assert_eq!(edge.dst, paris);
    }

    #[test]
    fn test_traversal_merges_layers() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        // Base has 2 incoming LIVES_IN edges.
        let incoming = layered.edges_from(amsterdam, Direction::Incoming);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_node_ids_combines_layers() {
        let layered = build_test_layered();
        let initial = layered.node_ids();
        assert_eq!(initial.len(), 3);

        layered.create_node(&["New"]);
        let after = layered.node_ids();
        assert_eq!(after.len(), 4);
    }

    #[test]
    fn test_delete_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let (_, eid) = edges[0];
        let deleted = layered.delete_edge(eid);
        assert!(deleted);

        let after = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_all_labels_combines() {
        let layered = build_test_layered();
        layered.create_node(&["NewLabel"]);

        let labels = layered.all_labels();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"City".to_string()));
        assert!(labels.contains(&"NewLabel".to_string()));
    }

    // ── A. Read-through operations ────────────────────────────────

    #[test]
    fn test_get_edge_from_base() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let (_, eid) = edges[0];
        let edge = layered.get_edge(eid);
        assert!(edge.is_some(), "edge should be readable from base");
        let edge = edge.unwrap();
        assert_eq!(edge.edge_type.as_str(), "LIVES_IN");

        // edge_type() accessor should agree
        assert_eq!(layered.edge_type(eid).as_deref(), Some("LIVES_IN"));

        // Edge property from base should be readable
        let since = layered.get_edge_property(eid, &PropertyKey::new("since"));
        assert!(
            since.is_some(),
            "edge property should be readable from base"
        );
    }

    #[test]
    fn test_get_node_property_batch_across_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");

        // Create an overlay-only node.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        let all_ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();
        let names = layered.get_node_property_batch(&all_ids, &PropertyKey::new("name"));

        // All should have a name.
        for name in &names {
            assert!(name.is_some(), "every node should have a name property");
        }
        // The overlay node should return "Vincent".
        assert_eq!(
            names.last().unwrap().as_ref().unwrap(),
            &Value::String(ArcStr::from("Vincent"))
        );
    }

    #[test]
    fn test_out_degree_both_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // Base has 1 outgoing edge (LIVES_IN) for an unmodified node.
        assert_eq!(layered.out_degree(first_person), 1);

        // Add an edge purely in the overlay (between overlay-only nodes).
        let vincent = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        layered.create_edge(vincent, berlin, "VISITS");

        // Overlay-only node should have 1 outgoing edge.
        assert_eq!(layered.out_degree(vincent), 1);

        // Base node remains unmodified, still sees its base edge.
        assert_eq!(layered.out_degree(first_person), 1);
    }

    #[test]
    fn test_in_degree_both_layers() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        // Base has 2 incoming LIVES_IN edges.
        assert_eq!(layered.in_degree(amsterdam), 2);

        // Create overlay-only edges between overlay-only nodes.
        let jules = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        layered.create_edge(jules, berlin, "LIVES_IN");

        // Berlin (overlay-only) should have 1 incoming edge.
        assert_eq!(layered.in_degree(berlin), 1);

        // Amsterdam (base, not dirty) should still have 2 incoming edges.
        assert_eq!(layered.in_degree(amsterdam), 2);
    }

    // ── B. Mutation operations ────────────────────────────────────

    #[test]
    fn test_set_node_property_promotes_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        assert_eq!(layered.overlay_mutation_count(), 0);

        // Setting a property on a base node should promote it.
        layered.set_node_property(first, "city", Value::from("Amsterdam"));

        // Node should now be dirty (in overlay).
        assert!(layered.overlay_mutation_count() > 0);

        // Property should be readable.
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Amsterdam")));
    }

    #[test]
    fn test_set_edge_property_promotes_base_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        assert_eq!(layered.overlay_mutation_count(), 0);

        // Setting a property on a base edge should promote it and its endpoints.
        layered.set_edge_property(eid, "weight", Value::Float64(1.5));

        assert!(layered.overlay_mutation_count() > 0);

        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(1.5));
    }

    #[test]
    fn test_remove_node_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Node has "age" property in the base.
        assert!(
            layered
                .get_node_property(first, &PropertyKey::new("age"))
                .is_some()
        );

        // Remove it (promotes to overlay first).
        let removed = layered.remove_node_property(first, "age");
        assert!(removed.is_some());

        // Should be gone now.
        assert!(
            layered
                .get_node_property(first, &PropertyKey::new("age"))
                .is_none()
        );
    }

    #[test]
    fn test_remove_edge_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Remove edge property (promotes edge and endpoints).
        let removed = layered.remove_edge_property(eid, "since");
        assert!(removed.is_some());

        // Should be gone now.
        assert!(
            layered
                .get_edge_property(eid, &PropertyKey::new("since"))
                .is_none()
        );
    }

    #[test]
    fn test_add_label_to_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Add a new label (promotes the node).
        let added = layered.add_label(first, "Employee");
        assert!(added);

        // Node should now have both labels.
        let node = layered.get_node(first).unwrap();
        let label_strs: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        assert!(label_strs.contains(&"Person"));
        assert!(label_strs.contains(&"Employee"));
    }

    #[test]
    fn test_remove_label_from_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Remove the "Person" label (promotes first).
        let removed = layered.remove_label(first, "Person");
        assert!(removed);

        // Should no longer appear in nodes_by_label("Person").
        let after_persons = layered.nodes_by_label("Person");
        assert!(!after_persons.contains(&first));

        // But node should still exist.
        assert!(layered.get_node(first).is_some());
    }

    #[test]
    fn test_delete_node_edges_cascade() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // First person has 1 outgoing edge.
        let edges_before = layered.edges_from(first_person, Direction::Outgoing);
        assert_eq!(edges_before.len(), 1);

        // Delete all edges connected to this node.
        layered.delete_node_edges(first_person);

        // The base edges should now be marked as deleted.
        let edges_after = layered.edges_from(first_person, Direction::Outgoing);
        assert_eq!(edges_after.len(), 0);
    }

    #[test]
    fn test_batch_create_edges_cross_layer() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_person = persons[0];

        // Create overlay-only cities.
        let berlin = layered.create_node(&["City"]);
        let paris = layered.create_node(&["City"]);

        // Batch create edges with a mix of base and overlay endpoints.
        let edge_specs: Vec<(NodeId, NodeId, &str)> = vec![
            (base_person, berlin, "VISITS"),
            (base_person, paris, "VISITS"),
        ];
        let eids = layered.batch_create_edges(&edge_specs);
        assert_eq!(eids.len(), 2);

        for eid in &eids {
            let edge = layered.get_edge(*eid);
            assert!(edge.is_some());
            assert_eq!(edge.unwrap().edge_type.as_str(), "VISITS");
        }
    }

    // ── C. Promotion logic ────────────────────────────────────────

    #[test]
    fn test_promotion_copies_all_properties() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Before promotion, read the base node properties.
        let original_name = layered
            .get_node_property(first, &PropertyKey::new("name"))
            .unwrap();
        let original_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();

        // Promote by setting a new property.
        layered.set_node_property(first, "city", Value::from("Berlin"));

        // All original properties should still be present.
        let after_name = layered
            .get_node_property(first, &PropertyKey::new("name"))
            .unwrap();
        let after_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();

        assert_eq!(original_name, after_name);
        assert_eq!(original_age, after_age);

        // New property also present.
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Berlin")));
    }

    #[test]
    fn test_promotion_is_idempotent() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // First promotion.
        layered.set_node_property(first, "x", Value::Int64(1));
        let count_after_first = layered.node_count();

        // Second promotion attempt (node already in overlay).
        layered.set_node_property(first, "y", Value::Int64(2));
        let count_after_second = layered.node_count();

        // Node count should not change.
        assert_eq!(count_after_first, count_after_second);

        // Both properties should exist.
        assert_eq!(
            layered
                .get_node_property(first, &PropertyKey::new("x"))
                .unwrap(),
            Value::Int64(1)
        );
        assert_eq!(
            layered
                .get_node_property(first, &PropertyKey::new("y"))
                .unwrap(),
            Value::Int64(2)
        );
    }

    #[test]
    fn test_edge_promotion_promotes_endpoints() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Promote the edge by setting a property on it.
        layered.set_edge_property(eid, "weight", Value::Float64(0.5));

        // The edge's source and destination nodes should now be in the overlay.
        let edge = layered.get_edge(eid).unwrap();
        let src_node = layered.get_node(edge.src);
        let dst_node = layered.get_node(edge.dst);
        assert!(
            src_node.is_some(),
            "source node should be accessible after edge promotion"
        );
        assert!(
            dst_node.is_some(),
            "destination node should be accessible after edge promotion"
        );
    }

    // ── D. Deleted entity tracking ────────────────────────────────

    #[test]
    fn test_deleted_base_node_invisible() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        let deleted = layered.delete_node(target);
        assert!(deleted);

        // get_node should return None.
        assert!(layered.get_node(target).is_none());

        // get_node_property should also return None.
        assert!(
            layered
                .get_node_property(target, &PropertyKey::new("name"))
                .is_none()
        );
    }

    #[test]
    fn test_deleted_node_excluded_from_nodes_by_label() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let target = persons[0];
        layered.delete_node(target);

        let after = layered.nodes_by_label("Person");
        assert_eq!(after.len(), 1);
        assert!(!after.contains(&target));
    }

    #[test]
    fn test_deleted_node_excluded_from_node_ids() {
        let layered = build_test_layered();
        let all_before = layered.node_ids();
        assert_eq!(all_before.len(), 3);

        let persons = layered.nodes_by_label("Person");
        let target = persons[0];
        layered.delete_node(target);

        let all_after = layered.node_ids();
        assert_eq!(all_after.len(), 2);
        assert!(!all_after.contains(&target));
    }

    #[test]
    fn test_deleted_edge_excluded_from_edges_from() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        let edges = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);

        let after = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_deleted_edge_excluded_from_neighbors() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Deleting the target NODE should remove it from neighbors.
        let neighbors_before = layered.neighbors(first, Direction::Outgoing);
        assert_eq!(neighbors_before.len(), 1);

        let target_node = neighbors_before[0];
        layered.delete_node(target_node);

        let neighbors_after = layered.neighbors(first, Direction::Outgoing);
        assert_eq!(
            neighbors_after.len(),
            0,
            "deleted node should not appear in neighbors"
        );
    }

    #[test]
    fn test_node_count_reflects_deletions() {
        let layered = build_test_layered();
        assert_eq!(layered.node_count(), 3);

        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);
        assert_eq!(layered.node_count(), 2);

        layered.delete_node(persons[1]);
        assert_eq!(layered.node_count(), 1);
    }

    #[test]
    fn test_edge_count_reflects_deletions() {
        let layered = build_test_layered();
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);
        assert_eq!(layered.edge_count(), 1);
    }

    // ── E. Search & statistics ────────────────────────────────────

    #[test]
    fn test_find_nodes_by_property_across_layers() {
        let layered = build_test_layered();

        // Base has Alix (age=30) and Gus (age=25).
        let age_30 = layered.find_nodes_by_property("age", &Value::Int64(30));
        assert_eq!(age_30.len(), 1);

        // Add an overlay node with the same property value.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "age", Value::Int64(30));

        let age_30_after = layered.find_nodes_by_property("age", &Value::Int64(30));
        assert_eq!(age_30_after.len(), 2);
        assert!(age_30_after.contains(&vincent));
    }

    #[test]
    fn test_find_nodes_in_range_across_layers() {
        let layered = build_test_layered();

        // Add an overlay node with age=35.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "age", Value::Int64(35));

        // Range query: age in [25, 35].
        let in_range = layered.find_nodes_in_range(
            "age",
            Some(&Value::Int64(25)),
            Some(&Value::Int64(35)),
            true,
            true,
        );

        // Should find Gus (25), Alix (30), and Mia (35).
        assert!(
            in_range.len() >= 3,
            "expected at least 3 nodes in range, got {}",
            in_range.len()
        );
        assert!(in_range.contains(&mia));
    }

    #[test]
    fn test_statistics_reflects_overlay() {
        let layered = build_test_layered();
        let stats_before = layered.statistics();
        let nodes_before = stats_before.total_nodes;

        // Add overlay nodes.
        layered.create_node(&["Person"]);
        layered.create_node(&["City"]);

        let stats_after = layered.statistics();
        assert_eq!(stats_after.total_nodes, nodes_before + 2);
    }

    #[test]
    fn test_all_edge_types_combines_layers() {
        let layered = build_test_layered();
        let types_before = layered.all_edge_types();
        assert!(types_before.contains(&"LIVES_IN".to_string()));

        // Add a new edge type in the overlay.
        let persons = layered.nodes_by_label("Person");
        let butch = layered.create_node(&["Person"]);
        layered.create_edge(persons[0], butch, "KNOWS");

        let types_after = layered.all_edge_types();
        assert!(types_after.contains(&"LIVES_IN".to_string()));
        assert!(types_after.contains(&"KNOWS".to_string()));
    }

    #[test]
    fn test_all_property_keys_combines_layers() {
        let layered = build_test_layered();
        let keys_before = layered.all_property_keys();
        assert!(keys_before.contains(&"name".to_string()));
        assert!(keys_before.contains(&"age".to_string()));

        // Add a new property key in the overlay.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "email", Value::from("mia@example.com"));

        let keys_after = layered.all_property_keys();
        assert!(keys_after.contains(&"email".to_string()));
        assert!(keys_after.contains(&"name".to_string()));
    }

    // ── F. Visibility ─────────────────────────────────────────────

    #[test]
    fn test_overlay_mutation_count() {
        let layered = build_test_layered();
        assert_eq!(layered.overlay_mutation_count(), 0);

        // Create a node: 1 dirty node.
        layered.create_node(&["Person"]);
        assert_eq!(layered.overlay_mutation_count(), 1);

        // Delete a base node: 1 dirty node + 1 deleted base node.
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);
        assert_eq!(layered.overlay_mutation_count(), 2);
    }

    #[test]
    fn test_memory_bytes_nonzero() {
        let layered = build_test_layered();
        assert!(
            layered.memory_bytes() > 0,
            "memory_bytes should be positive for a non-empty store"
        );
    }

    // ── G. Versioned mutation methods ────────────────────────────────

    // ── G. Versioned read methods ────────────────────────────────────
    // Note: versioned mutation tests are omitted because the layered store's
    // epoch ordering (base at MAX, overlay at 0) prevents versioned writes
    // from appending to the version log. The non-versioned mutation tests
    // above already exercise the ensure_in_overlay promotion logic.

    #[test]
    fn test_versioned_node_reads() {
        let layered = build_test_layered();
        let epoch = layered.current_epoch();
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Base node falls through
        assert!(
            layered.get_node_versioned(first, epoch, txn_id).is_some(),
            "versioned read should fall through to base"
        );
        assert!(
            layered.get_node_at_epoch(first, epoch).is_some(),
            "base node should be visible at epoch 0"
        );

        // Overlay node is readable
        // Keep this fixture committed: this test qualifies versioned reads,
        // while transaction-local mutation visibility is covered separately.
        let hans = layered.create_node(&["Person"]);
        layered.set_node_property(hans, "name", Value::from("Hans"));
        let node = layered.get_node_versioned(hans, epoch, txn_id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Hans")))
        );
        assert!(layered.get_node_at_epoch(hans, epoch).is_some());

        // Deleted base node returns None
        layered.delete_node(first);
        assert!(
            layered.get_node_versioned(first, epoch, txn_id).is_none(),
            "versioned read should return None for deleted base node"
        );
        assert!(
            layered.get_node_at_epoch(first, epoch).is_none(),
            "deleted base node should not be visible at epoch"
        );
    }

    #[test]
    fn test_versioned_edge_reads() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let base_edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = base_edges[0];

        // Base edge falls through
        assert!(
            layered
                .get_edge_versioned(base_eid, epoch, txn_id)
                .is_some(),
            "versioned read should fall through to base edge"
        );
        assert!(
            layered.get_edge_at_epoch(base_eid, epoch).is_some(),
            "base edge should be visible at epoch 0"
        );

        // Overlay edge is readable
        let barcelona = layered.create_node(&["City"]);
        let overlay_eid =
            layered.create_edge_versioned(persons[0], barcelona, "VISITS", epoch, txn_id);
        let edge = layered
            .get_edge_versioned(overlay_eid, epoch, txn_id)
            .unwrap();
        assert_eq!(edge.edge_type.as_str(), "VISITS");

        // Deleted base edge returns None
        layered.delete_edge(base_eid);
        assert!(
            layered
                .get_edge_versioned(base_eid, epoch, txn_id)
                .is_none(),
            "versioned read should return None for deleted base edge"
        );
        assert!(
            layered.get_edge_at_epoch(base_eid, epoch).is_none(),
            "deleted base edge should not be visible at epoch"
        );
    }

    // ── I. Visibility methods ────────────────────────────────────────

    #[test]
    fn test_node_visibility() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        // Base node visible (epoch and versioned)
        assert!(layered.is_node_visible_at_epoch(target, epoch));
        assert!(layered.is_node_visible_versioned(target, epoch, txn_id));

        // Overlay node visible
        let beatrix = layered.create_node(&["Person"]);
        assert!(layered.is_node_visible_at_epoch(beatrix, epoch));

        // Versioned overlay node visible
        let butch = layered.create_node_versioned(&["Person"], epoch, txn_id);
        assert!(layered.is_node_visible_versioned(butch, epoch, txn_id));

        // Deleted base node invisible
        layered.delete_node(target);
        assert!(!layered.is_node_visible_at_epoch(target, epoch));
        assert!(!layered.is_node_visible_versioned(target, epoch, txn_id));
    }

    #[test]
    fn test_edge_visibility() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Base edge visible (epoch and versioned)
        assert!(layered.is_edge_visible_at_epoch(eid, epoch));
        assert!(layered.is_edge_visible_versioned(eid, epoch, txn_id));

        // Deleted base edge invisible
        layered.delete_edge(eid);
        assert!(!layered.is_edge_visible_at_epoch(eid, epoch));
        assert!(!layered.is_edge_visible_versioned(eid, epoch, txn_id));
    }

    #[test]
    fn test_filter_visible_node_ids() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);

        let all_ids = layered.node_ids();
        assert_eq!(all_ids.len(), 3);

        // All nodes visible (epoch and versioned)
        assert_eq!(layered.filter_visible_node_ids(&all_ids, epoch).len(), 3);
        assert_eq!(
            layered
                .filter_visible_node_ids_versioned(&all_ids, epoch, txn_id)
                .len(),
            3
        );

        // Delete one, both filters should exclude it
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);

        let visible_epoch = layered.filter_visible_node_ids(&all_ids, epoch);
        assert_eq!(visible_epoch.len(), 2);
        assert!(!visible_epoch.contains(&persons[0]));

        let visible_versioned = layered.filter_visible_node_ids_versioned(&all_ids, epoch, txn_id);
        assert_eq!(visible_versioned.len(), 2);
        assert!(!visible_versioned.contains(&persons[0]));
    }

    // ── J. History methods ───────────────────────────────────────────

    #[test]
    fn test_history_base_and_promoted_entities() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];
        let edges = layered.edges_from(first, Direction::Outgoing);
        let (_, eid) = edges[0];

        // Pristine compact rows still have an exact synthesized structural
        // lifetime. This is the public audit trail, not overlay-only history.
        let node_history = layered.get_node_history(first);
        assert_eq!(node_history.len(), 1);
        assert_eq!(node_history[0].0, EpochId::INITIAL);
        assert_eq!(node_history[0].1, None);
        assert_eq!(node_history[0].2.id, first);
        let edge_history = layered.get_edge_history(eid);
        assert_eq!(edge_history.len(), 1);
        assert_eq!(edge_history[0].0, EpochId::INITIAL);
        assert_eq!(edge_history[0].1, None);
        assert_eq!(edge_history[0].2.id, eid);

        // Promote both to overlay by modifying them
        layered.set_node_property(first, "age", Value::Int64(42));
        layered.set_edge_property(eid, "weight", Value::Float64(2.0));

        // Promotion extends the exact property audit log without splitting the
        // original structural lifetime.
        let node_history = layered.get_node_history(first);
        assert_eq!(node_history.len(), 1);
        assert_eq!(
            (node_history[0].0, node_history[0].1),
            (EpochId::INITIAL, None)
        );
        assert_eq!(
            layered
                .complete_node_property_history_for_key(first, "age")
                .last()
                .map(|(_, value)| value),
            Some(&Value::Int64(42))
        );
        let edge_history = layered.get_edge_history(eid);
        assert_eq!(edge_history.len(), 1);
        assert_eq!(
            (edge_history[0].0, edge_history[0].1),
            (EpochId::INITIAL, None)
        );
        assert_eq!(
            layered
                .edge_full_history(eid)
                .properties
                .get(&PropertyKey::new("weight"))
                .and_then(|history| history.last())
                .map(|(_, value)| value),
            Some(&Value::Float64(2.0))
        );
    }

    // ── K. Multi-condition search and batch reads ────────────────────

    #[test]
    fn test_find_nodes_by_properties_across_layers() {
        let layered = build_test_layered();

        // Base has Alix (name="Alix", age=30) and Gus (name="Gus", age=25).
        let results = layered
            .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::Int64(30))]);
        assert_eq!(results.len(), 1);

        // Add overlay node matching the same conditions.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "name", Value::from("Alix"));
        layered.set_node_property(mia, "age", Value::Int64(30));

        let results_after = layered
            .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::Int64(30))]);
        assert_eq!(results_after.len(), 2);
        assert!(results_after.contains(&mia));
    }

    #[test]
    fn test_find_nodes_by_properties_empty_conditions() {
        let layered = build_test_layered();

        // Empty conditions should return all node IDs.
        let results = layered.find_nodes_by_properties(&[]);
        assert_eq!(results.len(), layered.node_ids().len());
    }

    #[test]
    fn test_get_nodes_properties_selective_batch() {
        let layered = build_test_layered();

        let persons = layered.nodes_by_label("Person");
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        layered.set_node_property(vincent, "age", Value::Int64(38));

        let all_ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();

        let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
        let batch = layered.get_nodes_properties_selective_batch(&all_ids, &keys);

        assert_eq!(batch.len(), all_ids.len());
        // Each map should contain only requested keys.
        for map in &batch {
            for key in map.keys() {
                assert!(
                    keys.contains(key),
                    "unexpected key {:?} in selective batch",
                    key
                );
            }
        }

        // Vincent's map should have both keys.
        let vincent_map = &batch[batch.len() - 1];
        assert_eq!(
            vincent_map.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );
        assert_eq!(
            vincent_map.get(&PropertyKey::new("age")),
            Some(&Value::Int64(38))
        );
    }

    #[test]
    fn test_get_edges_properties_selective_batch() {
        let layered = build_test_layered();

        let persons = layered.nodes_by_label("Person");
        let edges_a = layered.edges_from(persons[0], Direction::Outgoing);
        let edges_b = layered.edges_from(persons[1], Direction::Outgoing);

        let edge_ids: Vec<EdgeId> = edges_a
            .iter()
            .chain(edges_b.iter())
            .map(|(_, eid)| *eid)
            .collect();

        let keys = vec![PropertyKey::new("since")];
        let batch = layered.get_edges_properties_selective_batch(&edge_ids, &keys);

        assert_eq!(batch.len(), edge_ids.len());
        for map in &batch {
            assert!(
                map.contains_key(&PropertyKey::new("since")),
                "each edge should have the 'since' property"
            );
        }
    }

    // ── L. Other uncovered methods ───────────────────────────────────

    #[test]
    fn test_estimate_label_cardinality() {
        let layered = build_test_layered();

        let person_card = layered.estimate_label_cardinality("Person");
        assert!(
            person_card >= 2.0,
            "should estimate at least 2 Person nodes, got {}",
            person_card
        );

        let city_card = layered.estimate_label_cardinality("City");
        assert!(
            city_card >= 1.0,
            "should estimate at least 1 City node, got {}",
            city_card
        );

        // Non-existent labels: the overlay's statistics may return a non-zero
        // default estimate, so we only check the call does not panic.
        let missing_card = layered.estimate_label_cardinality("NonExistent");
        assert!(
            missing_card >= 0.0,
            "cardinality for unknown label should be non-negative"
        );
    }

    #[test]
    fn test_estimate_avg_degree() {
        let layered = build_test_layered();

        let avg_out = layered.estimate_avg_degree("LIVES_IN", true);
        assert!(
            avg_out > 0.0,
            "average out-degree for LIVES_IN should be positive"
        );

        let avg_in = layered.estimate_avg_degree("LIVES_IN", false);
        assert!(
            avg_in > 0.0,
            "average in-degree for LIVES_IN should be positive"
        );
    }

    #[test]
    fn test_estimate_avg_degree_empty() {
        let store = LpgStore::new().unwrap();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let layered = LayeredStore::new(compact, 0, 0).unwrap();

        let avg = layered.estimate_avg_degree("NONEXISTENT", true);
        assert_eq!(avg, 0.0, "empty store should have avg degree 0");
    }

    #[test]
    fn test_node_property_might_match() {
        let layered = build_test_layered();

        // "age" exists in the base, so might_match for Eq with an Int64 should be true
        // (zone maps allow Int64 values).
        let might = layered.node_property_might_match(
            &PropertyKey::new("age"),
            CompareOp::Eq,
            &Value::Int64(30),
        );
        assert!(might, "zone map should indicate age might match 30");
    }

    #[test]
    fn test_edge_property_might_match() {
        let layered = build_test_layered();

        let might = layered.edge_property_might_match(
            &PropertyKey::new("since"),
            CompareOp::Eq,
            &Value::Int64(2020),
        );
        assert!(might, "zone map should indicate since might match 2020");
    }

    // ── M. Focused coverage for promotion and delete-then-recreate ────

    /// Deleting a base node and then creating a fresh overlay node with the
    /// same label must not double-count. The overlay allocator seeded by
    /// `max_node_id + 1` guarantees the new node gets a distinct ID, and
    /// the tombstone on the deleted base ID prevents it from reappearing.
    /// Covers the deletion bookkeeping in `neighbors`, `node_ids`, and
    /// `node_count`.
    #[test]
    fn test_layered_delete_and_recreate_node() {
        let layered = build_test_layered();

        let persons_before = layered.nodes_by_label("Person");
        assert_eq!(persons_before.len(), 2);
        let target = persons_before[0];

        // Record neighbors of the other person (baseline). Alix -> Amsterdam,
        // Gus -> Amsterdam both exist in the base; choose the non-target.
        let other = persons_before[1];
        let other_neighbors_before = layered.neighbors(other, Direction::Outgoing);

        // Delete target (base node), then create a fresh Person in the overlay.
        assert!(layered.delete_node(target));
        let replacement = layered.create_node(&["Person"]);
        layered.set_node_property(replacement, "name", Value::from("Shosanna"));

        // Counts should reflect: 3 base - 1 deleted + 1 overlay node = 3.
        assert_eq!(layered.node_count(), 3);

        // nodes_by_label should see exactly 2 Persons again: the non-deleted
        // base person and the new overlay person.
        let persons_after = layered.nodes_by_label("Person");
        assert_eq!(persons_after.len(), 2);
        assert!(persons_after.contains(&other));
        assert!(persons_after.contains(&replacement));
        assert!(
            !persons_after.contains(&target),
            "deleted base node must not reappear"
        );

        // Neighbors of the non-target person should be unchanged.
        let other_neighbors_after = layered.neighbors(other, Direction::Outgoing);
        assert_eq!(other_neighbors_before, other_neighbors_after);

        // node_ids must not contain the tombstoned id.
        let all_ids = layered.node_ids();
        assert!(!all_ids.contains(&target));
        assert!(all_ids.contains(&replacement));
    }

    /// Mutating a base-only node promotes it into the overlay with all its
    /// labels and properties copied over, and the overlay's node ID counter
    /// is NOT modified by promotion (create_node_with_id never touches it),
    /// so subsequent `create_node` calls still get fresh IDs.
    /// Exercises `ensure_in_overlay` end to end.
    #[test]
    fn test_layered_promote_node_on_mutation() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        // Record the overlay's next-id allocator before promotion so we can
        // verify it is untouched afterwards (create_node_with_id never
        // modifies next_node_id — no save/restore required).
        let next_id_before = layered.overlay.load().next_node_id();

        // Snapshot the base node to compare after promotion.
        let base_node = layered.base.load().get_node(target).unwrap();
        let base_labels: Vec<String> = base_node
            .labels
            .iter()
            .map(|l| l.as_str().to_string())
            .collect();
        let base_name = layered
            .base
            .load()
            .get_node_property(target, &PropertyKey::new("name"))
            .unwrap();

        // Mutate: set a new property to trigger promotion.
        layered.set_node_property(target, "city", Value::from("Amsterdam"));

        // Overlay now owns the node; labels survived.
        let promoted = layered.overlay.load().get_node(target).unwrap();
        let promoted_labels: Vec<String> = promoted
            .labels
            .iter()
            .map(|l| l.as_str().to_string())
            .collect();
        assert_eq!(promoted_labels, base_labels);

        // Existing properties survived (read through the layered store).
        let name_after = layered
            .get_node_property(target, &PropertyKey::new("name"))
            .unwrap();
        assert_eq!(name_after, base_name);

        // New property is set.
        assert_eq!(
            layered.get_node_property(target, &PropertyKey::new("city")),
            Some(Value::String(ArcStr::from("Amsterdam")))
        );

        // ID counter is untouched by promotion: allocating a new node must
        // not collide with the promoted id or any existing base id.
        let next_id_after = layered.overlay.load().next_node_id();
        assert_eq!(
            next_id_before, next_id_after,
            "overlay next_node_id must not be modified by promotion"
        );
        let fresh = layered.create_node(&["Person"]);
        assert_ne!(fresh, target);
        for &p in &persons {
            assert_ne!(fresh, p);
        }
    }

    /// Mutating a base-only edge promotes the edge into the overlay together
    /// with both its endpoints, and its properties are preserved. Covers
    /// `ensure_edge_in_overlay` including its cascade into
    /// `ensure_in_overlay` for src and dst.
    #[test]
    fn test_layered_promote_edge_on_mutation() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let (target_dst, target_eid) = edges[0];

        // Capture the base edge's metadata for later comparison.
        let base_edge = layered.base.load().get_edge(target_eid).unwrap();
        let base_since = base_edge
            .properties
            .get(&PropertyKey::new("since"))
            .cloned()
            .unwrap();

        // Mutate: this promotes the edge and its endpoints.
        layered.set_edge_property(target_eid, "weight", Value::Float64(0.75));

        // The edge is now owned by the overlay.
        assert!(layered.overlay.load().get_edge(target_eid).is_some());
        // Original property still readable via the layered store.
        let since_after = layered
            .get_edge_property(target_eid, &PropertyKey::new("since"))
            .unwrap();
        assert_eq!(since_after, base_since);
        // New property is readable.
        assert_eq!(
            layered.get_edge_property(target_eid, &PropertyKey::new("weight")),
            Some(Value::Float64(0.75))
        );

        // Both endpoints are promoted and reachable from the overlay.
        assert!(
            layered.overlay.load().get_node(persons[0]).is_some(),
            "edge source must be in the overlay after promotion"
        );
        assert!(
            layered.overlay.load().get_node(target_dst).is_some(),
            "edge destination must be in the overlay after promotion"
        );

        // Endpoints' existing properties are intact through the layered view.
        assert!(
            layered
                .get_node_property(persons[0], &PropertyKey::new("name"))
                .is_some()
        );
        assert!(
            layered
                .get_node_property(target_dst, &PropertyKey::new("name"))
                .is_some()
        );
    }

    /// Setting a property on a base-only node marks the node dirty. Directly
    /// exercises the private `is_node_dirty` accessor used by the promotion
    /// machinery.
    #[test]
    fn test_layered_is_node_dirty_after_mutation() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        assert!(
            !layered.is_node_dirty(target),
            "base-only node should start clean"
        );

        layered.set_node_property(target, "city", Value::from("Berlin"));

        assert!(
            layered.is_node_dirty(target),
            "node must be dirty after a mutating set_node_property"
        );
    }

    // ── N. Accessor / debug coverage ─────────────────────────────────

    #[test]
    fn test_base_store_and_overlay_store_accessors() {
        let layered = build_test_layered();

        // base_store() returns a reference whose counts match the original.
        assert_eq!(layered.base_store_arc().node_count(), 3);
        assert_eq!(layered.base_store_arc().edge_count(), 2);

        // base_store_arc() returns an owned Arc that aliases the base.
        let arc = layered.base_store_arc();
        assert_eq!(arc.node_count(), 3);

        // overlay_store() returns the Arc<LpgStore> reference.
        assert_eq!(layered.overlay_store().node_count(), 0);
    }

    #[test]
    fn test_has_backward_adjacency() {
        let layered = build_test_layered();
        // Base is built via from_graph_store_preserving_ids which enables
        // backward CSR for every rel table.
        assert!(layered.has_backward_adjacency());
    }

    #[test]
    fn test_debug_format_does_not_panic() {
        let layered = build_test_layered();
        let s = format!("{layered:?}");
        assert!(s.contains("LayeredStore"));
    }

    #[test]
    fn test_current_epoch_delegates_to_overlay() {
        let layered = build_test_layered();
        let epoch = layered.current_epoch();
        // Just verify that the delegation does not panic, and that overlay
        // agrees.
        assert_eq!(epoch, layered.overlay_store().current_epoch());
    }

    // ── N. Overlay-only mutation and delete scenarios ────────────────

    #[test]
    fn test_delete_overlay_only_node() {
        let layered = build_test_layered();

        // Create a fresh overlay node, then delete it. This hits the
        // `is_node_dirty` branch of delete_node.
        let beatrix = layered.create_node(&["Person"]);
        assert!(layered.get_node(beatrix).is_some());

        let deleted = layered.delete_node(beatrix);
        assert!(deleted);
        assert!(
            layered.get_node(beatrix).is_none(),
            "overlay-only node should be unreadable after delete"
        );

        // delete on a non-existent ID returns false.
        let missing = NodeId::from(9_999_999u64);
        assert!(!layered.delete_node(missing));
    }

    #[test]
    fn test_delete_overlay_only_edge() {
        let layered = build_test_layered();

        // Overlay-only edge between two overlay-only nodes.
        let django = layered.create_node(&["Person"]);
        let shosanna = layered.create_node(&["Person"]);
        let eid = layered.create_edge(django, shosanna, "KNOWS");
        assert!(layered.get_edge(eid).is_some());

        let deleted = layered.delete_edge(eid);
        assert!(deleted);
        assert!(
            layered.get_edge(eid).is_none(),
            "overlay-only edge should be unreadable after delete"
        );

        // delete on an unknown edge id returns false.
        let missing = EdgeId::from(9_999_999u64);
        assert!(!layered.delete_edge(missing));
    }

    #[test]
    fn test_delete_then_recreate_node_with_same_label() {
        let layered = build_test_layered();
        let persons_before = layered.nodes_by_label("Person");
        assert_eq!(persons_before.len(), 2);

        // Delete one base Person, then add a new overlay Person.
        layered.delete_node(persons_before[0]);
        let hans = layered.create_node(&["Person"]);
        layered.set_node_property(hans, "name", Value::from("Hans"));

        let persons_after = layered.nodes_by_label("Person");
        // 1 remaining base Person + 1 new overlay Person = 2.
        assert_eq!(persons_after.len(), 2);
        assert!(persons_after.contains(&hans));
        assert!(!persons_after.contains(&persons_before[0]));
    }

    #[test]
    fn test_neighbors_from_promoted_node_with_new_overlay_edges() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Promote the base node, then add a new outgoing edge on it.
        layered.set_node_property(first, "touched", Value::Bool(true));
        let paris = layered.create_node(&["City"]);
        layered.set_node_property(paris, "name", Value::from("Paris"));
        let _ = layered.create_edge(first, paris, "VISITS");

        // Neighbors should include BOTH the original Amsterdam (a base edge
        // that pre-dates promotion) and the new Paris (an overlay edge added
        // after promotion). `ensure_in_overlay` only copies the node's labels
        // and properties — never its adjacency — so the base layer remains
        // authoritative for pre-promotion edges and must be merged with the
        // overlay's post-promotion edges.
        let outgoing = layered.neighbors(first, Direction::Outgoing);
        assert!(
            outgoing.contains(&paris),
            "overlay-created edge target should appear in neighbors"
        );
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities
            .iter()
            .copied()
            .find(|&c| c != paris)
            .expect("base City Amsterdam should still be present");
        assert!(
            outgoing.contains(&amsterdam),
            "base edge target should still appear in neighbors after promotion"
        );
    }

    #[test]
    fn test_edges_from_promoted_node_has_overlay_edges() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Promote and add overlay-only edges.
        let berlin = layered.create_node(&["City"]);
        layered.set_node_property(first, "touched", Value::Bool(true));
        let new_eid = layered.create_edge(first, berlin, "VISITS");

        let edges = layered.edges_from(first, Direction::Outgoing);
        let found_ids: Vec<EdgeId> = edges.iter().map(|(_, e)| *e).collect();
        assert!(
            found_ids.contains(&new_eid),
            "new overlay edge should be reachable via edges_from after promotion"
        );
    }

    #[test]
    fn test_edge_count_with_overlay_adds() {
        let layered = build_test_layered();
        // Base: 2 edges.
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        let vincent = layered.create_node(&["Person"]);
        let _ = layered.create_edge(persons[0], vincent, "KNOWS");
        let _ = layered.create_edge(persons[1], vincent, "KNOWS");

        // Base (2) - deleted (0) - promoted (0) + overlay (2 new) = 4.
        assert_eq!(layered.edge_count(), 4);
    }

    #[test]
    fn test_edge_count_with_base_edge_promoted_is_not_double_counted() {
        let layered = build_test_layered();
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        let base_edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = base_edges[0];

        // Promote base edge to overlay (by setting a new property on it).
        layered.set_edge_property(base_eid, "weight", Value::Float64(1.0));

        // Total must remain 2 (promoted, not duplicated).
        assert_eq!(layered.edge_count(), 2);
    }

    #[test]
    fn counts_ignore_tombstones_absent_from_the_pinned_base_generation() {
        let layered = build_test_layered();
        let node_count = layered.node_count();
        let edge_count = layered.edge_count();

        layered.seed_deleted_from_base([NodeId::new(9_900_001)], [EdgeId::new(9_900_002)]);

        assert_eq!(
            layered.node_count(),
            node_count,
            "a historical/foreign node tombstone cannot subtract a row this base never counted"
        );
        assert_eq!(
            layered.edge_count(),
            edge_count,
            "a historical/foreign edge tombstone cannot subtract a row this base never counted"
        );
    }

    #[test]
    fn edge_count_deduplicates_promoted_open_reincarnation_with_closed_sidecar() {
        let layered = empty_layered();
        let overlay = layered.overlay_store();
        let src = overlay.create_node(&["Source"]);
        let dst = overlay.create_node(&["Target"]);
        let edge = EdgeId::new(8_750);
        overlay
            .restore_edge_history_exact(
                edge,
                src,
                dst,
                "LINKS",
                &[
                    (EpochId::new(10), Some(EpochId::new(20))),
                    (EpochId::new(30), None),
                ],
            )
            .expect("restore closed then open edge lifetimes");
        overlay.set_epoch(EpochId::new(30));
        layered
            .merge_overlay_temporal()
            .expect("compact reincarnated edge");

        let base = layered.base_store_arc();
        assert!(base.closed_edge_ids().contains(&edge));
        assert!(base.get_edge(edge).is_some());
        assert_eq!(layered.edge_count(), 1);
        drop(base);
        let old_snapshot = EpochId::new(15);
        let gap_snapshot = EpochId::new(25);
        let current_snapshot = EpochId::new(35);
        let tx = TransactionId::new(8_751);
        assert!(layered.get_edge_versioned(edge, old_snapshot, tx).is_some());
        assert!(layered.get_edge_versioned(edge, gap_snapshot, tx).is_none());
        assert!(
            layered
                .get_edge_versioned(edge, current_snapshot, tx)
                .is_some()
        );

        layered.set_edge_property(edge, "promoted", Value::Bool(true));
        assert_eq!(
            layered.edge_count(),
            1,
            "the current CSR incarnation must be subtracted when its promoted overlay copy publishes"
        );
        assert!(
            layered.get_edge_versioned(edge, old_snapshot, tx).is_some(),
            "promotion of the open reincarnation must retain the older compact lifetime"
        );
        assert!(layered.get_edge_versioned(edge, gap_snapshot, tx).is_none());
        assert!(
            layered
                .get_edge_versioned(edge, current_snapshot, tx)
                .is_some()
        );
        assert!(
            layered
                .edges_from_versioned(src, Direction::Outgoing, old_snapshot, tx)
                .iter()
                .any(|(_, candidate)| *candidate == edge),
            "versioned traversal must source the older reincarnation from compact history"
        );
    }

    #[test]
    fn test_delete_edge_then_recreate() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];
        let edges = layered.edges_from(first, Direction::Outgoing);
        let (target, base_eid) = edges[0];

        // Delete the base edge.
        assert!(layered.delete_edge(base_eid));
        assert_eq!(layered.edges_from(first, Direction::Outgoing).len(), 0);

        // Recreate a fresh overlay edge between the same endpoints.
        let new_eid = layered.create_edge(first, target, "LIVES_IN");
        assert_ne!(new_eid, base_eid);

        let edges_after = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(edges_after.len(), 1);
        assert_eq!(edges_after[0].1, new_eid);
    }

    #[test]
    fn test_overlay_mutation_count_tracks_all_four_kinds() {
        let layered = build_test_layered();
        assert_eq!(layered.overlay_mutation_count(), 0);

        // Kind 1: dirty node (new overlay node).
        layered.create_node(&["Person"]);
        assert_eq!(
            layered.overlay_mutation_count(),
            1,
            "kind 1 (dirty node) must increment by exactly 1"
        );

        // Kind 2: dirty edge (new overlay edge between new overlay nodes).
        // Two more dirty nodes + one dirty edge = +3. Running total 1 + 3 = 4.
        let a = layered.create_node(&["Person"]);
        let b = layered.create_node(&["Person"]);
        let _ = layered.create_edge(a, b, "KNOWS");
        assert_eq!(
            layered.overlay_mutation_count(),
            4,
            "kind 2 (2 nodes + 1 edge) must add exactly 3, for total 4"
        );

        // Kind 3: deleted base node.
        let persons = layered.nodes_by_label("Person");
        let base_person = *persons
            .iter()
            .find(|id| layered.base.load().get_node(**id).is_some())
            .expect("fixture must have at least one base node");
        let before_delete_node = layered.overlay_mutation_count();
        layered.delete_node(base_person);
        assert_eq!(
            layered.overlay_mutation_count(),
            before_delete_node + 1,
            "kind 3 (deleted base node) must increment by exactly 1"
        );

        // Kind 4: deleted base edge. The fixture is required to have one so
        // this branch always executes; a conditional would let the kind-4
        // tracker silently regress.
        let persons2 = layered.nodes_by_label("Person");
        let (other_base, base_eid) = persons2
            .iter()
            .find_map(|id| {
                layered.base.load().get_node(*id)?;
                let edges = layered.edges_from(*id, Direction::Outgoing);
                edges.first().map(|(_, eid)| (*id, *eid))
            })
            .expect("fixture must have at least one base edge to delete");
        let _ = other_base;
        let before_delete_edge = layered.overlay_mutation_count();
        layered.delete_edge(base_eid);
        assert_eq!(
            layered.overlay_mutation_count(),
            before_delete_edge + 1,
            "kind 4 (deleted base edge) must increment by exactly 1"
        );
    }

    #[test]
    fn test_get_entity_history_includes_pristine_base_rows() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = edges[0];

        let edge_history = layered.get_edge_history(base_eid);
        assert_eq!(edge_history.len(), 1);
        assert_eq!(edge_history[0].0, EpochId::INITIAL);
        assert_eq!(edge_history[0].1, None);
        assert_eq!(edge_history[0].2.id, base_eid);

        let node_history = layered.get_node_history(persons[0]);
        assert_eq!(node_history.len(), 1);
        assert_eq!(node_history[0].0, EpochId::INITIAL);
        assert_eq!(node_history[0].1, None);
        assert_eq!(node_history[0].2.id, persons[0]);
    }

    // ── M. Accessors and Debug ───────────────────────────────────────

    #[test]
    fn test_base_store_accessors() {
        let layered = build_test_layered();

        // base_store returns a reference with 3 base nodes
        assert_eq!(layered.base_store_arc().node_count(), 3);

        // base_store_arc returns a cloned Arc that sees the same data
        let arc_clone = layered.base_store_arc();
        assert_eq!(arc_clone.node_count(), 3);
        assert!(Arc::strong_count(&arc_clone) >= 2);
    }

    #[test]
    fn test_overlay_store_accessor() {
        let layered = build_test_layered();
        assert_eq!(layered.overlay_store().node_count(), 0);

        layered.create_node(&["Person"]);
        assert_eq!(layered.overlay_store().node_count(), 1);
    }

    #[test]
    fn test_debug_impl_renders() {
        let layered = build_test_layered();
        layered.create_node(&["Person"]);
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);

        let rendered = format!("{layered:?}");
        assert!(rendered.contains("LayeredStore"));
        assert!(rendered.contains("base_node_count"));
        assert!(rendered.contains("overlay_node_count"));
        assert!(rendered.contains("dirty_nodes"));
        assert!(rendered.contains("deleted_base_nodes"));
    }

    // ── N. Miscellaneous read paths ──────────────────────────────────

    #[test]
    fn test_get_nodes_properties_batch_full() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        let ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();
        let batch = layered.get_nodes_properties_batch(&ids);
        assert_eq!(batch.len(), ids.len());

        // Base nodes have name and age.
        for map in batch.iter().take(persons.len()) {
            assert!(map.contains_key(&PropertyKey::new("name")));
            assert!(map.contains_key(&PropertyKey::new("age")));
        }

        // Overlay node has name.
        let vincent_map = &batch[batch.len() - 1];
        assert_eq!(
            vincent_map.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );

        // Missing node returns an empty map rather than panicking.
        let missing = NodeId::new(999_999);
        let batch_missing = layered.get_nodes_properties_batch(&[missing]);
        assert_eq!(batch_missing.len(), 1);
        assert!(batch_missing[0].is_empty());
    }

    #[test]
    fn test_get_node_deleted_returns_none() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        layered.delete_node(target);
        assert!(layered.get_node(target).is_none());

        // Property batch for a deleted node should also see empty entries.
        let batch = layered.get_nodes_properties_batch(&[target]);
        assert!(batch[0].is_empty());
    }

    #[test]
    fn test_get_edge_dirty_path_via_promotion() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Promote edge to overlay by setting a property on it.
        layered.set_edge_property(eid, "weight", Value::Float64(1.25));

        // get_edge should now return via overlay.
        let edge = layered.get_edge(eid).unwrap();
        assert_eq!(edge.edge_type.as_str(), "LIVES_IN");

        // edge_type should also route through overlay.
        assert_eq!(layered.edge_type(eid).as_deref(), Some("LIVES_IN"));

        // get_edge_property for a dirty edge reads from overlay.
        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(1.25));
    }

    #[test]
    fn test_get_edge_property_deleted() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);
        assert!(
            layered
                .get_edge_property(eid, &PropertyKey::new("since"))
                .is_none(),
            "deleted edge should not expose properties"
        );
        assert!(layered.edge_type(eid).is_none());
    }

    #[test]
    fn test_get_node_at_epoch_and_versioned_dirty() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);

        // Create an overlay (dirty) node.
        let jules = layered.create_node(&["Person"]);
        layered.set_node_property(jules, "name", Value::from("Jules"));

        // Dirty branch for get_node_at_epoch and get_node_versioned.
        assert!(layered.get_node_at_epoch(jules, epoch).is_some());
        assert!(layered.get_node_versioned(jules, epoch, txn_id).is_some());
    }

    #[test]
    fn test_get_edge_at_epoch_and_versioned_dirty() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);

        // Overlay edge between overlay nodes.
        let django = layered.create_node(&["Person"]);
        let prague = layered.create_node(&["City"]);
        let eid = layered.create_edge(django, prague, "VISITS");

        assert!(layered.get_edge_at_epoch(eid, epoch).is_some());
        assert!(layered.get_edge_versioned(eid, epoch, txn_id).is_some());
    }

    // ── O. Delete branches ───────────────────────────────────────────

    #[test]
    fn test_delete_nonexistent_node_returns_false() {
        let layered = build_test_layered();
        let missing = NodeId::new(999_999);
        assert!(!layered.delete_node(missing));

        let txn_id = TransactionId::from(1);
        let epoch = EpochId::from(u64::MAX);
        assert!(!layered.delete_node_versioned(missing, epoch, txn_id));
    }

    #[test]
    fn test_delete_nonexistent_edge_returns_false() {
        let layered = build_test_layered();
        let missing = EdgeId::new(999_999);
        assert!(!layered.delete_edge(missing));

        let txn_id = TransactionId::from(1);
        let epoch = EpochId::from(u64::MAX);
        assert!(!layered.delete_edge_versioned(missing, epoch, txn_id));
    }

    #[test]
    fn test_delete_dirty_node_via_overlay() {
        let layered = build_test_layered();
        // Create an overlay-only node then delete it through the dirty branch.
        let shosanna = layered.create_node(&["Person"]);
        assert!(layered.get_node(shosanna).is_some());
        assert!(layered.delete_node(shosanna));
        assert!(layered.get_node(shosanna).is_none());
    }

    #[test]
    fn test_delete_dirty_edge_via_overlay() {
        let layered = build_test_layered();
        let hans = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        let eid = layered.create_edge(hans, berlin, "LIVES_IN");

        assert!(layered.delete_edge(eid));
        assert!(layered.get_edge(eid).is_none());
    }

    #[test]
    fn test_delete_base_node_versioned() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");

        // Base-path deletion via versioned delete.
        assert!(layered.delete_node_versioned(persons[0], epoch, txn_id));
        assert!(layered.get_node(persons[0]).is_none());
    }

    #[test]
    fn base_only_node_delete_fans_out_named_and_unqualified_predicates() {
        let layered = build_test_layered();
        let tx = TransactionId::new(41);
        let spy = Arc::new(PredicateWriteSpy::default());
        layered.register_write_tracker(tx, spy.clone());
        let node = layered.nodes_by_label("Person")[0];

        assert!(layered.delete_node_versioned(node, EpochId::new(1), tx));
        assert_eq!(spy.full_node_writes.load(Ordering::Relaxed), 0);
        assert_eq!(spy.dataset_writes.load(Ordering::Relaxed), 1);
        assert_eq!(&*spy.label_name_writes.lock(), &["Person"]);
        assert!(spy.rel_type_name_writes.lock().is_empty());
    }

    #[test]
    fn test_delete_base_edge_versioned() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        assert!(layered.delete_edge_versioned(eid, epoch, txn_id));
        assert!(layered.get_edge(eid).is_none());
    }

    #[test]
    fn base_only_edge_delete_fans_out_named_and_unqualified_predicates() {
        let layered = build_test_layered();
        let tx = TransactionId::new(42);
        let spy = Arc::new(PredicateWriteSpy::default());
        layered.register_write_tracker(tx, spy.clone());
        let person = layered.nodes_by_label("Person")[0];
        let edge = layered.edges_from(person, Direction::Outgoing)[0].1;

        assert!(layered.delete_edge_versioned(edge, EpochId::new(1), tx));
        assert_eq!(spy.full_edge_writes.load(Ordering::Relaxed), 0);
        assert_eq!(spy.dataset_writes.load(Ordering::Relaxed), 1);
        assert_eq!(&*spy.rel_type_name_writes.lock(), &["LIVES_IN"]);
        assert!(spy.label_name_writes.lock().is_empty());
    }

    #[test]
    fn overlay_delete_does_not_repeat_name_only_fallback_fanout() {
        let layered = build_test_layered();
        let node = layered.create_node(&["OverlayOnly"]);
        let other = layered.create_node(&["Other"]);
        let edge = layered.create_edge(node, other, "OVERLAY_REL");
        let tx = TransactionId::new(43);
        let spy = Arc::new(PredicateWriteSpy::default());
        layered.register_write_tracker(tx, spy.clone());

        assert!(layered.delete_edge_versioned(edge, EpochId::new(1), tx));
        assert!(layered.delete_node_versioned(node, EpochId::new(1), tx));

        assert_eq!(spy.full_node_writes.load(Ordering::Relaxed), 1);
        assert_eq!(spy.full_edge_writes.load(Ordering::Relaxed), 1);
        assert_eq!(
            spy.dataset_writes.load(Ordering::Relaxed),
            0,
            "the normal overlay fan-out already carries the dataset guard"
        );
        assert_eq!(&*spy.label_name_writes.lock(), &["OverlayOnly"]);
        assert_eq!(&*spy.rel_type_name_writes.lock(), &["OVERLAY_REL"]);
    }

    #[test]
    fn test_delete_node_edges_on_dirty_source() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Promote the source node into the overlay.
        layered.set_node_property(first, "city", Value::from("Berlin"));
        assert!(layered.overlay.load().get_node(first).is_some());

        // delete_node_edges should now cascade through both overlay and base edges.
        layered.delete_node_edges(first);
        let remaining = layered.edges_from(first, Direction::Outgoing);
        assert!(
            remaining.is_empty(),
            "edges from a dirty source should be fully removed"
        );
    }

    // ── P. Versioned property/label mutations ────────────────────────

    #[test]
    fn test_set_node_property_versioned_promotes_base() {
        let layered = build_test_layered();
        let txn_id = TransactionId::from(42);
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Versioned set promotes the base node into the overlay.
        layered.set_node_property_versioned(first, "city", Value::from("Paris"), txn_id);
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Paris")));
    }

    #[test]
    fn test_set_edge_property_versioned_promotes_base() {
        let layered = build_test_layered();
        let txn_id = TransactionId::from(7);
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Versioned edge property set promotes the edge and its endpoints.
        layered.set_edge_property_versioned(eid, "weight", Value::Float64(3.5), txn_id);
        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(3.5));
    }

    #[test]
    fn test_remove_node_property_versioned_on_overlay_node() {
        // Use an overlay-only node to avoid the epoch-ordering restriction
        // that exists when promoting base nodes and then doing versioned removes.
        let layered = build_test_layered();
        let txn_id = TransactionId::from(101);

        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "email", Value::from("mia@example.com"));

        let removed = layered.remove_node_property_versioned(mia, "email", txn_id);
        assert_eq!(
            removed,
            Some(Value::String(ArcStr::from("mia@example.com")))
        );
        assert!(
            layered
                .get_node_property(mia, &PropertyKey::new("email"))
                .is_none()
        );
    }

    #[test]
    fn test_remove_edge_property_versioned_on_overlay_edge() {
        // Use an overlay-only edge to avoid epoch-ordering restrictions.
        let layered = build_test_layered();
        let txn_id = TransactionId::from(202);

        let django = layered.create_node(&["Person"]);
        let paris = layered.create_node(&["City"]);
        let eid = layered.create_edge(django, paris, "VISITS");
        layered.set_edge_property(eid, "year", Value::Int64(2024));

        let removed = layered.remove_edge_property_versioned(eid, "year", txn_id);
        assert_eq!(removed, Some(Value::Int64(2024)));
        assert!(
            layered
                .get_edge_property(eid, &PropertyKey::new("year"))
                .is_none()
        );
    }

    #[test]
    fn test_add_and_remove_label_versioned_on_overlay_node() {
        // Use an overlay-only node to avoid epoch-ordering issues that can occur
        // when promoting a base node and then writing versioned labels on top of
        // the epoch-0 promotion entry.
        let layered = build_test_layered();
        let txn_id = TransactionId::from(11);

        let butch = layered.create_node(&["Person"]);
        assert!(layered.add_label_versioned(butch, "Employee", txn_id));

        let node = layered.get_node(butch).unwrap();
        let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        assert!(labels.contains(&"Employee"));
        assert!(labels.contains(&"Person"));

        assert!(layered.remove_label_versioned(butch, "Employee", txn_id));
        let node = layered.get_node(butch).unwrap();
        let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        assert!(!labels.contains(&"Employee"));
    }

    // ── Q. ensure_in_overlay / ensure_edge_in_overlay edge cases ─────

    #[test]
    fn test_ensure_in_overlay_noop_for_nonexistent_node() {
        let layered = build_test_layered();
        let missing = NodeId::new(999_999);

        // set_node_property on a non-existent node should not crash; ensure_in_overlay
        // takes the "not in base either" early return.
        layered.set_node_property(missing, "name", Value::from("Ghost"));

        // The phantom property lands in the overlay even though the node does not
        // exist in either layer, so verify the path did not panic and no base node
        // appeared.
        assert!(layered.base_store_arc().get_node(missing).is_none());
    }

    #[test]
    fn test_ensure_edge_in_overlay_noop_for_nonexistent_edge() {
        let layered = build_test_layered();
        let missing = EdgeId::new(999_999);

        // set_edge_property on a missing edge should take the "not in base" branch.
        layered.set_edge_property(missing, "weight", Value::Float64(1.0));
        assert!(layered.base_store_arc().get_edge(missing).is_none());
    }

    #[test]
    fn failed_edge_creation_endpoint_hydration_is_representation_only() {
        fn assert_failed_attempt(
            layered: &LayeredStore,
            attempt: impl FnOnce(&LayeredStore, NodeId, NodeId) -> Vec<EdgeId>,
        ) {
            let source = layered.nodes_by_label("Person")[0];
            let missing = NodeId::new(9_999_999);
            let before = layered.node_structural_history(source);
            let node_count = layered.node_count();
            let edge_count = layered.edge_count();
            let overlay = layered.overlay_store();
            let next_node_id = overlay.next_node_id();
            let next_edge_id = overlay.next_edge_id();

            assert!(attempt(layered, source, missing).is_empty());
            assert_eq!(layered.node_count(), node_count);
            assert_eq!(layered.edge_count(), edge_count);
            assert_eq!(overlay.next_node_id(), next_node_id);
            assert_eq!(overlay.next_edge_id(), next_edge_id);
            assert!(layered.get_node(missing).is_none());
            let after = layered.node_structural_history(source);
            assert_eq!(after.lifetimes, before.lifetimes);
            assert_eq!(after.label_versions, before.label_versions);
            assert_eq!(after.labels, before.labels);
            assert_eq!(after.properties, before.properties);
        }

        let single = build_test_layered();
        assert_failed_attempt(&single, |store, source, missing| {
            let id = store.create_edge(source, missing, "MISSING_ENDPOINT");
            id.is_valid().then_some(id).into_iter().collect()
        });

        let batch = build_test_layered();
        assert_failed_attempt(&batch, |store, source, missing| {
            store.batch_create_edges(&[(source, missing, "MISSING_ENDPOINT")])
        });
    }

    #[test]
    fn test_ensure_edge_in_overlay_idempotent() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // First call promotes the edge; second call should take the early return.
        layered.set_edge_property(eid, "weight", Value::Float64(1.0));
        layered.set_edge_property(eid, "weight", Value::Float64(2.0));

        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(2.0));
    }

    // ── R. Traversal edge-cases for deleted neighbors ────────────────

    #[test]
    fn test_neighbors_incoming_with_deleted_source() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        let persons = layered.nodes_by_label("Person");
        // Delete one of the LIVES_IN source nodes; amsterdam's incoming neighbors
        // should drop that deleted node.
        layered.delete_node(persons[0]);

        let incoming = layered.neighbors(amsterdam, Direction::Incoming);
        assert!(!incoming.contains(&persons[0]));
        assert_eq!(incoming.len(), 1);
    }

    #[test]
    fn neighbors_excludes_target_of_deleted_base_edge() {
        let layered = build_test_layered();
        let person = layered.nodes_by_label("Person")[0];
        let (target, eid) = layered.edges_from(person, Direction::Outgoing)[0];

        // Delete the (un-promoted) base edge. edges_from already drops it...
        assert!(layered.delete_edge(eid));
        assert!(layered.edges_from(person, Direction::Outgoing).is_empty());

        // ...but neighbors() must agree: no target reachable only via a deleted edge.
        assert!(
            !layered
                .neighbors(person, Direction::Outgoing)
                .contains(&target),
            "neighbors() reported a target whose only edge was deleted"
        );
    }

    #[test]
    fn delete_promoted_edge_tombstones_base() {
        let layered = build_test_layered();
        let person = layered.nodes_by_label("Person")[0];
        let (target, eid) = layered.edges_from(person, Direction::Outgoing)[0];

        // Promote the edge into the overlay (copies it; base copy remains).
        layered.set_edge_property(eid, "weight", Value::Int64(5));
        assert!(layered.is_edge_dirty(eid));

        // Delete it. Every read path must agree it is gone.
        assert!(layered.delete_edge(eid));
        assert!(
            layered.get_edge(eid).is_none(),
            "deleted promoted edge still resolves"
        );
        assert!(layered.edges_from(person, Direction::Outgoing).is_empty());
        assert!(
            !layered
                .neighbors(person, Direction::Outgoing)
                .contains(&target)
        );
        // Idempotent: nothing left to delete.
        assert!(!layered.delete_edge(eid));
    }

    #[test]
    fn delete_promoted_node_tombstones_base() {
        let layered = build_test_layered();
        let person = layered.nodes_by_label("Person")[0];

        // Promote the node (labels/properties copied to overlay; base adjacency stays).
        layered.set_node_property(person, "nick", Value::from("x"));
        assert!(layered.is_node_dirty(person));

        assert!(layered.delete_node(person));
        assert!(
            layered.get_node(person).is_none(),
            "deleted promoted node still resolves"
        );
        // Its base edges must not resurface through the deleted node.
        assert!(layered.edges_from(person, Direction::Outgoing).is_empty());
    }

    #[test]
    fn test_edges_from_dirty_source_merges_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Capture the base-tier outgoing edges before any promotion.
        let base_outgoing: Vec<(NodeId, EdgeId)> = layered.edges_from(first, Direction::Outgoing);
        assert!(
            !base_outgoing.is_empty(),
            "test fixture should give the first Person a base edge"
        );

        // Promote `first` into the overlay (ensure_in_overlay copies labels and
        // properties only — never adjacency) and add a fresh overlay-only edge.
        layered.set_node_property(first, "city", Value::from("Berlin"));
        let prague = layered.create_node(&["City"]);
        layered.create_edge(first, prague, "VISITS");

        let outgoing = layered.edges_from(first, Direction::Outgoing);

        // The new overlay edge must be visible …
        assert!(
            outgoing.iter().any(|(target, _)| *target == prague),
            "new overlay edge should appear in edges_from"
        );

        // … and every pre-promotion base edge must remain visible. ensure_in_overlay
        // only promotes the node; base adjacency is still authoritative for edges
        // that pre-date the promotion, and must not silently disappear once the
        // source node becomes dirty.
        for (target, eid) in &base_outgoing {
            assert!(
                outgoing.iter().any(|(t, e)| t == target && e == eid),
                "base edge {eid:?} (→ {target:?}) must remain visible after promotion"
            );
        }
    }

    /// Regression for property-anchored edge lookups across the snapshot
    /// boundary: when a base node is promoted into the overlay (e.g. by
    /// `set_node_property` or by becoming the endpoint of a new overlay
    /// edge), its pre-existing base-tier edges must remain reachable via
    /// `neighbors` and `edges_from` in BOTH directions. Otherwise GQL
    /// patterns like `MATCH (a {id: $x})-[:T]->(b {id: $y})` start
    /// returning zero rows once any overlay write has touched either
    /// endpoint, which the planner uses to walk the edge from.
    #[test]
    fn test_base_edge_visible_from_promoted_endpoint() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let cities = layered.nodes_by_label("City");
        let alix = persons[0];
        let amsterdam = cities[0];

        // Sanity: the base edge alix -LIVES_IN-> amsterdam exists pre-promotion.
        let pre_out = layered.edges_from(alix, Direction::Outgoing);
        let pre_in = layered.edges_from(amsterdam, Direction::Incoming);
        let pre_neigh_out = layered.neighbors(alix, Direction::Outgoing);
        let pre_neigh_in = layered.neighbors(amsterdam, Direction::Incoming);
        assert!(pre_out.iter().any(|(t, _)| *t == amsterdam));
        assert!(pre_in.iter().any(|(t, _)| *t == alix));
        assert!(pre_neigh_out.contains(&amsterdam));
        assert!(pre_neigh_in.contains(&alix));

        // Promote BOTH endpoints into the overlay by way of an unrelated
        // overlay write. The new edge intentionally points at a brand-new
        // overlay node so the LIVES_IN edge between alix and amsterdam is
        // not touched in any way.
        let oslo = layered.create_node(&["City"]);
        layered.create_edge(alix, oslo, "VISITS"); // promotes alix
        layered.set_node_property(amsterdam, "touched", Value::Bool(true)); // promotes amsterdam

        // Both endpoints are now dirty.
        assert!(layered.is_node_dirty(alix));
        assert!(layered.is_node_dirty(amsterdam));

        // The pre-existing base edge must still be reachable from either side,
        // both as an edge (with its original EdgeId) and as a neighbor.
        let post_out = layered.edges_from(alix, Direction::Outgoing);
        let post_in = layered.edges_from(amsterdam, Direction::Incoming);
        let post_neigh_out = layered.neighbors(alix, Direction::Outgoing);
        let post_neigh_in = layered.neighbors(amsterdam, Direction::Incoming);

        let base_eid = pre_out
            .iter()
            .find(|(t, _)| *t == amsterdam)
            .map(|(_, e)| *e)
            .expect("pre-promotion fixture has a base LIVES_IN edge");

        assert!(
            post_out
                .iter()
                .any(|(t, e)| *t == amsterdam && *e == base_eid),
            "base LIVES_IN edge must remain visible via edges_from(src, Outgoing) after src is promoted"
        );
        assert!(
            post_in.iter().any(|(t, e)| *t == alix && *e == base_eid),
            "base LIVES_IN edge must remain visible via edges_from(dst, Incoming) after dst is promoted"
        );
        assert!(
            post_neigh_out.contains(&amsterdam),
            "base neighbor must remain visible via neighbors(src, Outgoing) after src is promoted"
        );
        assert!(
            post_neigh_in.contains(&alix),
            "base neighbor must remain visible via neighbors(dst, Incoming) after dst is promoted"
        );
    }

    // ── Phase 5c: overlay reset + in-place merge ──────────────────────

    /// `reset_overlay` swaps in a fresh empty `LpgStore` and clears
    /// dirty/deleted bookkeeping. Base reads keep working. Overlay-only
    /// nodes disappear (they were never persisted to base).
    #[test]
    fn alix_reset_overlay_clears_mutations_preserves_base() {
        let layered = build_test_layered();
        let base_persons_before = layered.nodes_by_label("Person").len();

        // Add an overlay node + delete a base node + dirty a base property.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        let base_persons = layered.nodes_by_label("Person");
        let to_delete = base_persons[0];
        layered.delete_node(to_delete);
        let still_alive = base_persons[1];
        layered.set_node_property(still_alive, "tagged", Value::from("hot"));

        assert!(layered.overlay_mutation_count() > 0);

        // Reset.
        layered.reset_overlay();

        // After reset: overlay is empty, base reads intact.
        assert_eq!(
            layered.overlay_mutation_count(),
            0,
            "overlay must be empty after reset"
        );
        assert_eq!(
            layered.nodes_by_label("Person").len(),
            base_persons_before,
            "base nodes restored (delete was overlay-only)"
        );
        assert!(
            layered.get_node(vincent).is_none(),
            "overlay-only node disappears after reset"
        );
        let base_node = layered.get_node(still_alive).unwrap();
        assert!(
            !base_node
                .properties
                .contains_key(&PropertyKey::new("tagged")),
            "base property dirty was reset"
        );
    }

    /// Reset discards the complete overlay representation. Its fresh successor
    /// must not inherit derived registries containing identities whose graph
    /// rows were intentionally thrown away. A retained old Arc remains a
    /// coherent independent snapshot, including its indexes and named graphs,
    /// and keeps the pre-existing raw-snapshot mutation semantics.
    #[test]
    fn reset_overlay_does_not_attach_discarded_runtime_registries_to_fresh_successor() {
        let layered = empty_layered();
        let snapshot = layered.overlay_store();
        snapshot.create_property_index("kind");
        let discarded = snapshot.create_node(&["Transient"]);
        snapshot.set_node_property(discarded, "kind", Value::from("before-reset"));
        let named = snapshot
            .graph_or_create("urn:grafeo:discarded")
            .expect("named overlay graph");
        let named_node = named.create_node(&["NamedTransient"]);
        #[cfg(feature = "text-index")]
        let detached_text_index = {
            use crate::index::text::{BM25Config, InvertedIndex};
            let index = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
            snapshot.add_text_index("Transient", "body", Arc::clone(&index));
            assert!(snapshot.get_text_index("Transient", "body").is_some());
            index
        };
        #[cfg(feature = "vector-index")]
        let detached_vector_index = {
            use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind};
            let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Euclidean,
            ))));
            snapshot.add_vector_index("Transient", "embedding", Arc::clone(&index));
            assert!(
                snapshot
                    .get_vector_index("Transient", "embedding")
                    .is_some()
            );
            index
        };

        layered.reset_overlay();
        let successor = layered.overlay_store();
        assert!(!Arc::ptr_eq(&snapshot, &successor));
        assert!(layered.get_node(discarded).is_none());
        assert!(successor.get_node(discarded).is_none());
        assert!(
            !successor.has_property_index("kind"),
            "a fresh reset successor must not inherit the discarded row registry"
        );
        assert!(successor.graph("urn:grafeo:discarded").is_none());
        #[cfg(feature = "text-index")]
        {
            successor.add_text_index("Transient", "body", Arc::clone(&detached_text_index));
            assert!(
                successor.get_text_index("Transient", "body").is_none(),
                "a detached text handle cannot rebind into the reset successor"
            );
        }
        #[cfg(feature = "vector-index")]
        {
            successor.add_vector_index(
                "Transient",
                "embedding",
                Arc::clone(&detached_vector_index),
            );
            assert!(
                successor
                    .get_vector_index("Transient", "embedding")
                    .is_none(),
                "a detached vector handle cannot rebind into the reset successor"
            );
        }

        assert_eq!(
            snapshot.find_nodes_by_property("kind", &Value::from("before-reset")),
            vec![discarded],
            "the retained snapshot keeps its exact physical property index"
        );
        assert!(Arc::ptr_eq(
            &snapshot
                .graph("urn:grafeo:discarded")
                .expect("retained snapshot keeps its named topology"),
            &named
        ));
        assert!(named.get_node(named_node).is_some());

        snapshot.set_node_property(discarded, "kind", Value::from("after-reset"));
        assert_eq!(
            snapshot.find_nodes_by_property("kind", &Value::from("after-reset")),
            vec![discarded],
            "detached reset snapshots remain independently mutable"
        );
        assert!(
            layered
                .find_nodes_by_property("kind", &Value::from("after-reset"))
                .is_empty(),
            "mutating a detached reset snapshot cannot contaminate the live generation"
        );
    }

    #[test]
    fn generation_reader_cannot_observe_pointer_swap_before_routing_install() {
        let layered = Arc::new(build_test_layered());
        let overlay_only = layered.create_node(&["Transient"]);
        assert_eq!(layered.node_count(), 4);

        let publication = Arc::new(std::sync::Barrier::new(2));
        *layered.generation_publication_barrier.write() = Some(Arc::clone(&publication));
        let resetting = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || layered.reset_overlay())
        };
        publication.wait();

        let (count_tx, count_rx) = mpsc::channel();
        let reader = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                count_tx.send(layered.node_count()).unwrap();
            })
        };
        assert!(
            count_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "generation-sensitive count must wait while pointers and routing maps are between swaps"
        );

        publication.wait();
        resetting.join().expect("reset publisher");
        assert_eq!(count_rx.recv().unwrap(), 3);
        reader.join().expect("generation reader");
        assert!(layered.get_node(overlay_only).is_none());
        *layered.generation_publication_barrier.write() = None;
    }

    #[test]
    fn counts_never_apply_prior_base_tombstones_to_the_successor_base() {
        let source = LpgStore::new().expect("count-cut source store");
        let retained_left = source.create_node(&["Retained"]);
        let retained_right = source.create_node(&["Retained"]);
        let deleted_node = source.create_node(&["Deleted"]);
        let deleted_edge = source.create_edge(retained_left, retained_right, "DELETED");
        let base = Arc::new(
            from_graph_store_preserving_ids(&source).expect("count-cut compact generation"),
        );
        let layered = Arc::new(
            LayeredStore::with_overlay(base, Arc::new(LpgStore::new().expect("count-cut overlay")))
                .expect("adopt count-cut overlay"),
        );
        assert!(layered.delete_node(deleted_node));
        assert!(layered.delete_edge(deleted_edge));
        assert_eq!(layered.node_count(), 2);
        assert_eq!(layered.edge_count(), 0);

        let publication = Arc::new(std::sync::Barrier::new(2));
        *layered.generation_publication_barrier.write() = Some(Arc::clone(&publication));
        let merging = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || layered.merge_overlay_in_place())
        };
        publication.wait();

        let (counts_tx, counts_rx) = mpsc::channel();
        let reader = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                counts_tx
                    .send((layered.node_count(), layered.edge_count()))
                    .unwrap();
            })
        };
        assert!(
            counts_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "counts must not inspect the successor base while prior-generation tombstones remain installed"
        );

        publication.wait();
        merging
            .join()
            .expect("count-cut merge thread")
            .expect("count-cut generation merge");
        assert_eq!(counts_rx.recv().unwrap(), (2, 0));
        reader.join().expect("count-cut reader");
        *layered.generation_publication_barrier.write() = None;
    }

    #[test]
    fn reset_successor_preserves_scope_epoch_adjacency_highwaters_and_incarnation() {
        use crate::graph::lpg::LpgStoreConfig;

        let empty_source = LpgStore::new().expect("empty compact source");
        let base = Arc::new(
            from_graph_store_preserving_ids(&empty_source).expect("empty compact generation"),
        );
        let overlay = Arc::new(
            LpgStore::with_config(LpgStoreConfig {
                backward_edges: false,
                initial_node_capacity: 17,
                initial_edge_capacity: 31,
            })
            .expect("configured overlay"),
        );
        let transport_source = overlay.create_node(&["Source"]);
        let transport_destination = overlay.create_node(&["Destination"]);
        let receipt = overlay
            .create_transport_edge_with_id(
                EdgeId::new(9_000),
                transport_source,
                transport_destination,
                "CARRIED",
            )
            .expect("transport edge allocation")
            .expect("fresh transport identity");
        overlay.sync_epoch(EpochId::new(91));
        overlay.set_next_node_id(9_101);
        overlay.set_next_edge_id(9_102);
        let owner = WriteAuthority::new();
        assert!(overlay.seal_unframed_writes(&owner));
        let layered = with_authority(&owner, || LayeredStore::with_overlay(base, overlay))
            .expect("adopt authorized overlay");

        with_authority(&owner, || layered.reset_overlay());
        let successor = layered.overlay_store();
        assert!(!successor.has_backward_adjacency());
        assert_eq!(successor.current_epoch(), EpochId::new(91));
        assert_eq!(successor.next_node_id(), 9_101);
        assert_eq!(successor.next_edge_id(), 9_102);
        assert!(successor.transport_receipt_belongs_to_current_incarnation(&receipt));
        assert!(!successor.create_node(&["unauthorized"]).is_valid());
        assert!(with_authority(&owner, || successor
            .create_node(&["authorized"])
            .is_valid()));
    }

    /// `merge_overlay_in_place` rebuilds the base from the combined view,
    /// swaps the base, and clears the overlay. After the call: all
    /// previously-visible data is in the base, overlay is empty.
    #[test]
    fn gus_merge_overlay_in_place_promotes_mutations_into_base() {
        let layered = build_test_layered();
        let count_before = layered.node_count();

        // Add an overlay node.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        layered.set_node_property(vincent, "age", Value::Int64(33));

        let count_after_mutation = layered.node_count();
        assert_eq!(count_after_mutation, count_before + 1);

        // Merge.
        layered
            .merge_overlay_in_place()
            .expect("merge_overlay_in_place");

        // Overlay empty; total node count preserved.
        assert_eq!(
            layered.overlay_mutation_count(),
            0,
            "overlay must be empty after merge"
        );
        assert_eq!(
            layered.node_count(),
            count_after_mutation,
            "total node count preserved across merge"
        );

        // Vincent now lives in the new base — verify by checking that
        // resetting the overlay would NOT make him disappear (post-merge,
        // he's part of the base).
        layered.reset_overlay();
        let still_there = layered.get_node(vincent);
        assert!(
            still_there.is_some(),
            "merged node persists after a subsequent reset_overlay (it's in base)"
        );
    }

    /// Round-trip property: merge then reset is a no-op on visible
    /// state.  Both operations leave the overlay empty.
    #[test]
    fn vincent_merge_then_reset_is_no_op_on_visible_state() {
        let layered = build_test_layered();
        let visible_before: Vec<NodeId> = layered.node_ids();

        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "name", Value::from("Mia"));

        layered.merge_overlay_in_place().unwrap();
        layered.reset_overlay();

        let visible_after: Vec<NodeId> = layered.node_ids();
        let mut a = visible_before;
        a.push(mia);
        a.sort_unstable();
        let mut b = visible_after;
        b.sort_unstable();
        assert_eq!(a, b);
    }

    // ── Phase 5d: concurrent base-swap + merge correctness ───────────

    #[test]
    fn sealed_swap_base_requires_current_overlay_authority() {
        let layered = build_test_layered();
        let original = layered.base_store_arc();
        let candidate = Arc::new(
            from_graph_store_preserving_ids(&layered).expect("candidate compact generation"),
        );
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(layered.overlay_store().seal_unframed_writes(&owner));

        let raw_candidate = Arc::clone(&candidate);
        let raw = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = layered.swap_base(raw_candidate);
        }));
        assert!(raw.is_err(), "raw base publication must fail closed");
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));

        let foreign_candidate = Arc::clone(&candidate);
        let foreign_attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&foreign, || {
                let _ = layered.swap_base(foreign_candidate);
            });
        }));
        assert!(
            foreign_attempt.is_err(),
            "foreign authority must not publish a compact base"
        );
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));

        let replaced = with_authority(&owner, || layered.swap_base(Arc::clone(&candidate)));
        assert!(Arc::ptr_eq(&replaced, &original));
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &candidate));
    }

    #[test]
    fn sealed_base_transition_checks_authority_before_callbacks_and_unwinds() {
        let layered = build_test_layered();
        let original = layered.base_store_arc();
        let candidate = Arc::new(
            from_graph_store_preserving_ids(&layered).expect("candidate compact generation"),
        );
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(layered.overlay_store().seal_unframed_writes(&owner));

        let prepare_calls = AtomicUsize::new(0);
        let finalize_calls = AtomicUsize::new(0);
        let raw = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<Arc<CompactStore>, ()> = layered.transition_base_generation(
                |_| {
                    prepare_calls.fetch_add(1, Ordering::SeqCst);
                    Ok((Arc::clone(&candidate), ()))
                },
                |()| {
                    finalize_calls.fetch_add(1, Ordering::SeqCst);
                },
            );
        }));
        assert!(raw.is_err(), "raw generation transition must fail closed");

        let foreign_attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&foreign, || {
                let _: Result<Arc<CompactStore>, ()> = layered.transition_base_generation(
                    |_| {
                        prepare_calls.fetch_add(1, Ordering::SeqCst);
                        Ok((Arc::clone(&candidate), ()))
                    },
                    |()| {
                        finalize_calls.fetch_add(1, Ordering::SeqCst);
                    },
                );
            });
        }));
        assert!(
            foreign_attempt.is_err(),
            "foreign authority must not enter generation callbacks"
        );
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
        assert_eq!(finalize_calls.load(Ordering::SeqCst), 0);
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));

        let authorized_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || {
                let _: Result<Arc<CompactStore>, ()> = layered.transition_base_generation(
                    |_| {
                        prepare_calls.fetch_add(1, Ordering::SeqCst);
                        Ok((Arc::clone(&candidate), ()))
                    },
                    |()| {
                        finalize_calls.fetch_add(1, Ordering::SeqCst);
                        panic!("hostile authorized finalizer");
                    },
                );
            });
        }));
        assert!(authorized_panic.is_err());
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
        assert_eq!(finalize_calls.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));

        let post_unwind_prepare_calls = AtomicUsize::new(0);
        let post_unwind_raw = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<Arc<CompactStore>, ()> = layered.transition_base_generation(
                |_| {
                    post_unwind_prepare_calls.fetch_add(1, Ordering::SeqCst);
                    Ok((Arc::clone(&candidate), ()))
                },
                |()| {},
            );
        }));
        assert!(
            post_unwind_raw.is_err(),
            "caught callback panic must not leak write authority"
        );
        assert_eq!(post_unwind_prepare_calls.load(Ordering::SeqCst), 0);

        let replaced = with_authority(&owner, || {
            layered
                .transition_base_generation(|_| Ok::<_, ()>((Arc::clone(&candidate), ())), |()| {})
                .expect("exact authority retries after callback unwind")
        });
        assert!(Arc::ptr_eq(&replaced, &original));
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &candidate));
    }

    #[test]
    fn sealed_retiring_base_transition_requires_authority_and_unwinds() {
        let layered = build_test_layered();
        let original = layered.base_store_arc();
        let candidate = Arc::new(
            from_graph_store_preserving_ids(&layered).expect("candidate compact generation"),
        );
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(layered.overlay_store().seal_unframed_writes(&owner));

        let prepare_calls = AtomicUsize::new(0);
        let publish_calls = AtomicUsize::new(0);
        let raw = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(Arc<CompactStore>, ()), ()> = layered
                .transition_base_generation_with_retirement(
                    |_| {
                        prepare_calls.fetch_add(1, Ordering::SeqCst);
                        Ok((Arc::clone(&candidate), ()))
                    },
                    |()| {
                        publish_calls.fetch_add(1, Ordering::SeqCst);
                    },
                );
        }));
        assert!(raw.is_err(), "raw retiring transition must fail closed");

        let foreign_attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&foreign, || {
                let _: Result<(Arc<CompactStore>, ()), ()> = layered
                    .transition_base_generation_with_retirement(
                        |_| {
                            prepare_calls.fetch_add(1, Ordering::SeqCst);
                            Ok((Arc::clone(&candidate), ()))
                        },
                        |()| {
                            publish_calls.fetch_add(1, Ordering::SeqCst);
                        },
                    );
            });
        }));
        assert!(
            foreign_attempt.is_err(),
            "foreign authority must not enter retiring callbacks"
        );
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
        assert_eq!(publish_calls.load(Ordering::SeqCst), 0);
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));

        let authorized_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || {
                let _: Result<(Arc<CompactStore>, ()), ()> = layered
                    .transition_base_generation_with_retirement(
                        |_| {
                            prepare_calls.fetch_add(1, Ordering::SeqCst);
                            Ok((Arc::clone(&candidate), ()))
                        },
                        |()| {
                            publish_calls.fetch_add(1, Ordering::SeqCst);
                            panic!("hostile authorized metadata publisher");
                        },
                    );
            });
        }));
        assert!(authorized_panic.is_err());
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
        assert_eq!(publish_calls.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));

        let (replaced, retirement) = with_authority(&owner, || {
            layered
                .transition_base_generation_with_retirement(
                    |_| Ok::<_, ()>((Arc::clone(&candidate), ())),
                    |()| "retired metadata",
                )
                .expect("exact authority retries after publisher unwind")
        });
        assert!(Arc::ptr_eq(&replaced, &original));
        assert_eq!(retirement, "retired metadata");
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &candidate));
    }

    #[test]
    fn compact_generation_finalize_panic_never_publishes_candidate_and_allows_retry() {
        let layered = build_test_layered();
        let original = layered.base_store_arc();
        let candidate = Arc::new(
            from_graph_store_preserving_ids(&layered).expect("candidate compact generation"),
        );
        let candidate_for_prepare = Arc::clone(&candidate);
        let finalize_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&finalize_calls);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<Arc<CompactStore>, ()> = layered.transition_base_generation(
                move |_| Ok((candidate_for_prepare, ())),
                move |()| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    panic!("hostile compact metadata finalizer");
                },
            );
        }));
        assert!(panic.is_err());
        assert_eq!(finalize_calls.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));
        assert!(!Arc::ptr_eq(&layered.base_store_arc(), &candidate));

        let retry_candidate =
            Arc::new(from_graph_store_preserving_ids(&layered).expect("retry compact generation"));
        let retry_witness = Arc::clone(&retry_candidate);
        let replaced: Arc<CompactStore> = layered
            .transition_base_generation(move |_| Ok::<_, ()>((retry_candidate, ())), |()| {})
            .expect("generation retry after caught panic");
        assert!(Arc::ptr_eq(&replaced, &original));
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &retry_witness));
    }

    #[test]
    fn generation_retirement_token_drops_only_after_both_layered_guards_drain() {
        struct GuardDrainProbe {
            layered: Arc<LayeredStore>,
            observed_drained: Arc<AtomicBool>,
        }

        impl Drop for GuardDrainProbe {
            fn drop(&mut self) {
                let publication_drained = self.layered.publication_guard.try_write().is_some();
                let mutation_drained = self.layered.merge_guard.try_write().is_some();
                self.observed_drained
                    .store(publication_drained && mutation_drained, Ordering::SeqCst);
            }
        }

        let layered = Arc::new(build_test_layered());
        let candidate = Arc::new(
            from_graph_store_preserving_ids(layered.as_ref())
                .expect("candidate compact generation"),
        );
        let observed_drained = Arc::new(AtomicBool::new(false));
        let probe_layered = Arc::clone(&layered);
        let probe_observed = Arc::clone(&observed_drained);
        let (_previous, retirement) = layered
            .transition_base_generation_with_retirement(
                move |_| Ok::<_, ()>((candidate, ())),
                move |()| GuardDrainProbe {
                    layered: probe_layered,
                    observed_drained: probe_observed,
                },
            )
            .expect("publish reversible generation");

        assert!(!observed_drained.load(Ordering::SeqCst));
        drop(retirement);
        assert!(
            observed_drained.load(Ordering::SeqCst),
            "retirement must run only after publication and mutation guards drain"
        );
    }

    #[test]
    fn compact_generation_panicking_finalizer_blocks_reader_then_restores_prior_base() {
        let layered = Arc::new(build_test_layered());
        let original = layered.base_store_arc();
        let candidate = Arc::new(
            from_graph_store_preserving_ids(layered.as_ref())
                .expect("candidate compact generation"),
        );
        let candidate_for_prepare = Arc::clone(&candidate);
        let finalizer_release = Arc::new(std::sync::Barrier::new(2));
        let (entered_tx, entered_rx) = mpsc::channel();
        let publisher = {
            let layered = Arc::clone(&layered);
            let finalizer_release = Arc::clone(&finalizer_release);
            std::thread::spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _: Result<Arc<CompactStore>, ()> = layered.transition_base_generation(
                        move |_| Ok((candidate_for_prepare, ())),
                        |()| {
                            entered_tx.send(()).unwrap();
                            finalizer_release.wait();
                            panic!("hostile compact metadata finalizer before base publication");
                        },
                    );
                }))
            })
        };
        entered_rx
            .recv()
            .expect("finalizer entered publication cut");

        let (captured_tx, captured_rx) = mpsc::channel();
        let reader = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || captured_tx.send(layered.base_store_arc()).unwrap())
        };
        assert!(
            captured_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "reader must remain behind the finalizer's coherent publication cut"
        );
        finalizer_release.wait();
        assert!(publisher.join().expect("publisher thread").is_err());
        let captured = captured_rx.recv().expect("reader resumes after unwind");
        reader.join().expect("publication reader");
        assert!(Arc::ptr_eq(&captured, &original));
        assert!(!Arc::ptr_eq(&captured, &candidate));
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));
    }

    #[test]
    fn compact_generation_callback_reentry_panics_without_deadlock_and_can_retry() {
        let layered = build_test_layered();
        let original = layered.base_store_arc();
        let candidate = Arc::new(
            from_graph_store_preserving_ids(&layered).expect("candidate compact generation"),
        );
        let candidate_for_prepare = Arc::clone(&candidate);
        let nested =
            Arc::new(from_graph_store_preserving_ids(&layered).expect("nested compact generation"));

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<Arc<CompactStore>, ()> = layered.transition_base_generation(
                move |_| Ok((candidate_for_prepare, ())),
                |()| {
                    let _ = layered.swap_base(nested);
                },
            );
        }));
        assert!(panic.is_err(), "same-store re-entry must fail promptly");
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &original));
        assert!(!Arc::ptr_eq(&layered.base_store_arc(), &candidate));

        let retry_candidate =
            Arc::new(from_graph_store_preserving_ids(&layered).expect("retry compact generation"));
        let retry_witness = Arc::clone(&retry_candidate);
        layered
            .transition_base_generation(move |_| Ok::<_, ()>((retry_candidate, ())), |()| {})
            .expect("generation retry after rejected callback re-entry");
        assert!(Arc::ptr_eq(&layered.base_store_arc(), &retry_witness));
    }

    /// Many readers + one swapper: swap_base should never produce a
    /// torn read or a panic.  Runs for a fixed iteration budget so
    /// the test stays bounded.
    #[test]
    fn jules_concurrent_readers_survive_repeated_base_swaps() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let layered = Arc::new(build_test_layered());
        let stop = Arc::new(AtomicBool::new(false));

        let mut readers = Vec::new();
        for _ in 0..4 {
            let l = Arc::clone(&layered);
            let s = Arc::clone(&stop);
            readers.push(thread::spawn(move || {
                let mut total = 0u64;
                while !s.load(Ordering::Relaxed) {
                    let people = l.nodes_by_label("Person");
                    // Person count is base(2) + overlay(0..many); never less than base.
                    assert!(people.len() >= 2, "lost a base node mid-swap");
                    total += people.len() as u64;
                }
                total
            }));
        }

        // Swapper builds a fresh base with one extra Person each round.
        let l = Arc::clone(&layered);
        let s = Arc::clone(&stop);
        let swapper = thread::spawn(move || {
            for _ in 0..200 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                // Read the current combined view, build a new compact base.
                let new_base = from_graph_store_preserving_ids(&*l).unwrap();
                l.swap_base(Arc::new(new_base));
            }
        });

        swapper.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        let totals: Vec<u64> = readers.into_iter().map(|h| h.join().unwrap()).collect();
        // Sanity: every reader observed at least one snapshot.
        for t in totals {
            assert!(t > 0, "reader saw zero snapshots");
        }
    }

    #[test]
    fn generation_reader_finishes_while_overlay_source_cut_is_retained()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let layered = Arc::new(build_test_layered());
        let unrelated = LpgStore::new()?;
        let added = layered.create_node(&["Person"]);
        let overlay = layered.overlay_store();
        let mutation = MutationWriteScope::enter(&layered);
        let transition = overlay
            .pin_exclusive_unframed_transition()
            .ok_or("unsealed overlay transition unavailable")?;
        let reading = Arc::clone(&layered);
        let (finished_tx, finished_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            {
                let _generation = GenerationReadScope::enter(&reading);
                assert!(has_retained_overlay_read(&reading.overlay_store()));
                assert!(!has_retained_overlay_read(&unrelated));
                let people = reading.nodes_by_label("Person");
                let node = reading.get_node(added);
                let _ = finished_tx.send((people, node.is_some()));
            }
            assert!(!has_retained_overlay_read(&reading.overlay_store()));
        });
        // The merger must be able to build its successor while readers finish
        // on the retained source. Release before checking the timeout so a
        // regression fails without leaving a blocked worker behind.
        let observed = finished_rx.recv_timeout(Duration::from_secs(2));
        drop(transition);
        drop(mutation);
        reader.join().map_err(|_| "generation reader panicked")?;
        let (people, saw_node) = observed?;
        assert_eq!(people.len(), 3);
        assert!(saw_node);
        Ok(())
    }

    /// Concurrent readers + writer + periodic merge_overlay_in_place.
    /// The merger thread races with both reads and writes; correctness
    /// requirement is that all writes that completed before a join
    /// remain visible after the test.
    #[test]
    fn shosanna_concurrent_writes_survive_periodic_merge() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let layered = Arc::new(build_test_layered());
        let stop = Arc::new(AtomicBool::new(false));
        let writer_count = Arc::new(AtomicUsize::new(0));

        // Readers: spin reading.
        let mut readers = Vec::new();
        for _ in 0..2 {
            let l = Arc::clone(&layered);
            let s = Arc::clone(&stop);
            readers.push(thread::spawn(move || {
                let mut iters = 0u64;
                while !s.load(Ordering::Relaxed) {
                    let _ = l.nodes_by_label("Person");
                    iters += 1;
                }
                iters
            }));
        }

        // Writer: insert a flurry of nodes.
        let l = Arc::clone(&layered);
        let s = Arc::clone(&stop);
        let wc = Arc::clone(&writer_count);
        let writer = thread::spawn(move || {
            for i in 0..500 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                let id = l.create_node(&["Person"]);
                l.set_node_property(id, "tag", Value::Int64(i));
                wc.fetch_add(1, Ordering::Relaxed);
            }
        });

        // Merger: periodically merge while writer/readers are active.
        let l = Arc::clone(&layered);
        let s = Arc::clone(&stop);
        let merger = thread::spawn(move || {
            for _ in 0..20 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                let _ = l.merge_overlay_in_place();
                std::thread::yield_now();
            }
        });

        writer.join().unwrap();
        merger.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        for h in readers {
            h.join().unwrap();
        }

        // Final invariant: total Person count = 2 (base) + writer_count.
        let expected = 2 + writer_count.load(Ordering::Relaxed);
        let actual = layered.nodes_by_label("Person").len();
        assert_eq!(
            actual, expected,
            "lost writes during concurrent merge; expected {expected}, got {actual}"
        );
    }

    /// reset_overlay under concurrent readers: snapshots taken before
    /// the reset must remain valid.  This exercises ArcSwap snapshot
    /// semantics: a reader holding `overlay_store()` keeps the old
    /// LpgStore alive even after the overlay is replaced.
    #[test]
    fn beatrix_reset_overlay_does_not_invalidate_held_snapshots() {
        use std::sync::Arc;
        use std::thread;

        let layered = Arc::new(build_test_layered());

        // Add a node so the overlay is non-empty.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        // Take a snapshot of the overlay; if reset_overlay swaps, this
        // Arc should keep the old LpgStore alive and queryable.
        let snapshot = layered.overlay_store();
        assert!(snapshot.get_node(vincent).is_some());

        // Reset on a thread to maximise the race window.
        let l = Arc::clone(&layered);
        let resetter = thread::spawn(move || {
            l.reset_overlay();
        });
        resetter.join().unwrap();

        // Snapshot still has Vincent; live overlay does not.
        assert!(
            snapshot.get_node(vincent).is_some(),
            "snapshot must remain valid after concurrent reset"
        );
        assert!(
            layered.overlay_store().get_node(vincent).is_none(),
            "live overlay is empty post-reset"
        );
    }

    #[test]
    fn test_has_property_index_false_when_no_index() {
        // Use an empty compact base: every compact property column is itself a
        // valid equality seek, so the populated fixture intentionally reports
        // its `name` and `age` columns as indexed.
        let layered = empty_layered();
        assert!(!layered.has_property_index("name"));
    }

    #[test]
    fn test_has_property_index_true_when_overlay_has_index() {
        // Regression test: LayeredStore::has_property_index must call
        // self.overlay.load().has_property_index(), not
        // self.overlay.has_property_index() (ArcSwap does not impl GraphStore).
        let layered = empty_layered();
        layered.overlay_store().create_property_index("name");
        assert!(layered.has_property_index("name"));
        assert!(!layered.has_property_index("age"));
    }

    // ── Task 6: snapshot-aware property isolation at LayeredStore level ──
    //
    // These tests verify that full MVCC delegation works end-to-end through
    // the LayeredStore wrapper — using the store API directly with explicit
    // TransactionId / EpochId (no full session required).

    #[test]
    fn layered_store_buffered_set_is_invisible_to_committed_reads() {
        use crate::graph::traits::{GraphStore, GraphStoreMut};
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        // Committed age should be 30 (from the base fixture).
        let key = PropertyKey::new("age");
        let epoch = EpochId::new(0);
        let tx = TransactionId::new(42);

        // Buffer an uncommitted write via the trait: writer sees 99.
        layered.set_node_property_buffered(target, "age", Value::Int64(99), tx);
        assert_eq!(
            layered.read_node_property_visible(target, &key, epoch, Some(tx)),
            Some(Value::Int64(99)),
            "writer must see its own buffered write"
        );

        // Committed read (tx = None) must still see the original value.
        assert_eq!(
            layered.read_node_property_visible(target, &key, epoch, None),
            Some(Value::Int64(30)),
            "uncommitted buffered SET must not be visible as a committed read"
        );

        // Apply the overlay: committed read now sees 99.
        layered.apply_tx_overlay(tx);
        assert_eq!(
            layered.read_node_property_visible(target, &key, epoch, None),
            Some(Value::Int64(99)),
            "committed read must see 99 after apply_tx_overlay"
        );
    }

    #[test]
    fn layered_store_buffered_set_is_dropped_on_rollback() {
        use crate::graph::traits::{GraphStore, GraphStoreMut};
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        let key = PropertyKey::new("age");
        let epoch = EpochId::new(0);
        let tx = TransactionId::new(7);

        layered.set_node_property_buffered(target, "age", Value::Int64(999), tx);
        // Drop the overlay (rollback).
        layered.drop_tx_overlay(tx);

        // Committed read must still see the original value (30), not 999.
        assert_eq!(
            layered.read_node_property_visible(target, &key, epoch, None),
            Some(Value::Int64(30)),
            "drop_tx_overlay must discard the buffered write"
        );
    }

    #[test]
    fn layered_store_whole_entity_properties_visible_merges_delta() {
        use crate::graph::traits::{GraphStore, GraphStoreMut};
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        let epoch = EpochId::new(0);
        let tx = TransactionId::new(5);

        // Buffer a Set and a Remove.
        layered.set_node_property_buffered(target, "age", Value::Int64(55), tx);
        layered.remove_node_property_buffered(target, "name", tx);

        // Writer's whole-entity view must reflect both operations.
        let writer_view = layered.read_node_properties_visible(target, epoch, Some(tx));
        assert_eq!(
            writer_view.get(&PropertyKey::new("age")),
            Some(&Value::Int64(55)),
            "writer's whole-entity view must contain the buffered age"
        );
        assert!(
            !writer_view.contains_key(&PropertyKey::new("name")),
            "writer's whole-entity view must NOT contain the removed name"
        );

        // Committed view must remain unchanged.
        let committed_view = layered.read_node_properties_visible(target, epoch, None);
        assert_eq!(
            committed_view.get(&PropertyKey::new("age")),
            Some(&Value::Int64(30)),
            "committed whole-entity view must retain original age"
        );
        assert!(
            committed_view.contains_key(&PropertyKey::new("name")),
            "committed whole-entity view must still have the name"
        );
    }

    // --- Task 5 (label delegation) tests ---

    #[test]
    fn layered_store_buffered_label_add_is_isolated() {
        use crate::graph::traits::{GraphStore, GraphStoreMut};
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        let epoch = EpochId::new(0);
        let tx = TransactionId::new(42);

        // Buffer an uncommitted label add: writer sees it, others do not.
        layered.add_label_buffered(target, "Secret", tx);

        let writer_labels = layered.read_node_labels_visible(target, epoch, Some(tx));
        assert!(
            writer_labels.iter().any(|l| l.as_str() == "Secret"),
            "writer must see its own buffered label add"
        );

        let committed_labels = layered.read_node_labels_visible(target, epoch, None);
        assert!(
            !committed_labels.iter().any(|l| l.as_str() == "Secret"),
            "uncommitted label add must not be visible as a committed read"
        );

        // Label scan: writer sees the node, committed scan does not.
        let writer_scan = layered.nodes_by_label_visible("Secret", Some(tx));
        assert!(
            writer_scan.contains(&target),
            "writer's label scan must include the buffered-add node"
        );

        let committed_scan = layered.nodes_by_label_visible("Secret", None);
        assert!(
            !committed_scan.contains(&target),
            "committed label scan must not include the uncommitted node"
        );

        // Apply the overlay: committed read now sees the label.
        layered.apply_tx_overlay(tx);
        let after_labels = layered.read_node_labels_visible(target, epoch, None);
        assert!(
            after_labels.iter().any(|l| l.as_str() == "Secret"),
            "committed read must see the label after apply_tx_overlay"
        );
    }

    #[test]
    fn layered_store_buffered_label_add_is_dropped_on_rollback() {
        use crate::graph::traits::{GraphStore, GraphStoreMut};
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        let epoch = EpochId::new(0);
        let tx = TransactionId::new(7);

        layered.add_label_buffered(target, "Secret", tx);
        // Drop the overlay (rollback).
        layered.drop_tx_overlay(tx);

        // Committed read must not see the rolled-back label.
        let after = layered.read_node_labels_visible(target, epoch, None);
        assert!(
            !after.iter().any(|l| l.as_str() == "Secret"),
            "drop_tx_overlay must discard the buffered label add"
        );
    }

    // ── N. SSI read-set: base-resident reads are recorded ────────────────────
    //
    // Regression tests for the C1 completeness gap: a Serializable tx reading a
    // base-resident entity (the steady state after `compact()`) must see its
    // read recorded into the tracker registered on the overlay.  Before the fix
    // the base path silently skipped `record_read_node` / `record_read_edge`.

    /// SpyTracker shared across SSI read-set tests.
    mod ssi_spy {
        use crate::execution::operators::{ReadTracker, SharedReadTracker};
        use grafeo_common::types::{EdgeId, NodeId, TransactionId};
        use parking_lot::Mutex;
        use std::sync::Arc;

        pub struct SpyTracker {
            pub nodes: Mutex<Vec<NodeId>>,
            pub edges: Mutex<Vec<EdgeId>>,
        }

        impl SpyTracker {
            pub fn new() -> Arc<Self> {
                Arc::new(Self {
                    nodes: Mutex::new(Vec::new()),
                    edges: Mutex::new(Vec::new()),
                })
            }
            pub fn saw_node(&self, id: NodeId) -> bool {
                self.nodes.lock().contains(&id)
            }
            pub fn saw_edge(&self, id: EdgeId) -> bool {
                self.edges.lock().contains(&id)
            }
            pub fn edge_read_count(&self, id: EdgeId) -> usize {
                self.edges
                    .lock()
                    .iter()
                    .filter(|recorded| **recorded == id)
                    .count()
            }
        }

        impl ReadTracker for SpyTracker {
            fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
                self.nodes.lock().push(id);
            }
            fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
                self.edges.lock().push(id);
            }
        }

        /// Register a SpyTracker for `tx` on `layered` and return the Arc.
        pub fn register(layered: &super::LayeredStore, tx: TransactionId) -> Arc<SpyTracker> {
            use crate::graph::traits::GraphStore;
            let spy = SpyTracker::new();
            let tracker: SharedReadTracker = Arc::clone(&spy) as SharedReadTracker;
            layered.register_read_tracker(tx, tracker);
            spy
        }
    }

    #[test]
    fn ssi_get_node_versioned_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        // After build_test_layered(), all nodes are in the compact base.
        let persons = layered.nodes_by_label("Person");
        let base_node = persons[0];

        let tx = TransactionId::new(101);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        // Read a base-resident node via the versioned accessor.
        let result = layered.get_node_versioned(base_node, epoch, tx);
        assert!(result.is_some(), "base-resident node must be visible");
        assert!(
            spy.saw_node(base_node),
            "get_node_versioned must record base-resident node into the SSI read-set"
        );
    }

    #[test]
    fn ssi_get_edge_versioned_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = edges[0];

        let tx = TransactionId::new(102);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let result = layered.get_edge_versioned(base_eid, epoch, tx);
        assert!(result.is_some(), "base-resident edge must be visible");
        assert!(
            spy.saw_edge(base_eid),
            "get_edge_versioned must record base-resident edge into the SSI read-set"
        );
    }

    #[test]
    fn ssi_is_node_visible_versioned_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_node = persons[0];

        let tx = TransactionId::new(103);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let visible = layered.is_node_visible_versioned(base_node, epoch, tx);
        assert!(visible, "base-resident node must be visible");
        assert!(
            spy.saw_node(base_node),
            "is_node_visible_versioned must record base-resident node when visible"
        );
    }

    #[test]
    fn ssi_is_edge_visible_versioned_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = edges[0];

        let tx = TransactionId::new(104);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let visible = layered.is_edge_visible_versioned(base_eid, epoch, tx);
        assert!(visible, "base-resident edge must be visible");
        assert!(
            spy.saw_edge(base_eid),
            "is_edge_visible_versioned must record base-resident edge when visible"
        );
    }

    #[test]
    fn ssi_read_node_property_visible_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_node = persons[0];

        let tx = TransactionId::new(105);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let result = layered.read_node_property_visible(
            base_node,
            &PropertyKey::new("name"),
            epoch,
            Some(tx),
        );
        assert!(result.is_some(), "base-resident property must be readable");
        assert!(
            spy.saw_node(base_node),
            "read_node_property_visible must record base-resident node into SSI read-set"
        );
    }

    #[test]
    fn ssi_read_edge_property_visible_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = edges[0];

        let tx = TransactionId::new(106);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let result = layered.read_edge_property_visible(
            base_eid,
            &PropertyKey::new("since"),
            epoch,
            Some(tx),
        );
        assert!(
            result.is_some(),
            "base-resident edge property must be readable"
        );
        assert!(
            spy.saw_edge(base_eid),
            "read_edge_property_visible must record base-resident edge into SSI read-set"
        );
    }

    #[test]
    fn ssi_read_node_properties_visible_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_node = persons[0];

        let tx = TransactionId::new(107);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let props = layered.read_node_properties_visible(base_node, epoch, Some(tx));
        assert!(!props.is_empty(), "base-resident node must have properties");
        assert!(
            spy.saw_node(base_node),
            "read_node_properties_visible must record base-resident node into SSI read-set"
        );
    }

    #[test]
    fn ssi_read_edge_properties_visible_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = edges[0];

        let tx = TransactionId::new(108);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let props = layered.read_edge_properties_visible(base_eid, epoch, Some(tx));
        assert!(!props.is_empty(), "base-resident edge must have properties");
        assert!(
            spy.saw_edge(base_eid),
            "read_edge_properties_visible must record base-resident edge into SSI read-set"
        );
    }

    #[test]
    fn ssi_read_node_labels_visible_records_base_resident_read() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_node = persons[0];

        let tx = TransactionId::new(109);
        let spy = ssi_spy::register(&layered, tx);
        let epoch = EpochId::from(u64::MAX);

        let labels = layered.read_node_labels_visible(base_node, epoch, Some(tx));
        assert!(!labels.is_empty(), "base-resident node must have labels");
        assert!(
            spy.saw_node(base_node),
            "read_node_labels_visible must record base-resident node into SSI read-set"
        );
    }

    #[test]
    fn ssi_nodes_by_label_visible_records_base_resident_reads() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::TransactionId;

        let layered = build_test_layered();
        // Snapshot which Person IDs are in the base.
        let base_persons: Vec<NodeId> = layered.nodes_by_label("Person");
        assert_eq!(base_persons.len(), 2, "fixture must have 2 base Persons");

        let tx = TransactionId::new(110);
        let spy = ssi_spy::register(&layered, tx);

        let visible = layered.nodes_by_label_visible("Person", Some(tx));
        assert_eq!(visible.len(), 2);
        for &id in &base_persons {
            assert!(
                spy.saw_node(id),
                "nodes_by_label_visible must record each base-resident Person ({:?})",
                id
            );
        }
    }

    /// Ensure that an unregistered tx does NOT cause a spurious recording
    /// (no-op path, mirrors the LpgStore behaviour).
    #[test]
    fn ssi_unregistered_tx_does_not_record() {
        use crate::graph::traits::GraphStore;
        use grafeo_common::types::{EpochId, TransactionId};

        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_node = persons[0];

        // Register a *different* tx so the tracker map is non-empty, then read
        // with an unregistered one; nothing must be recorded for the latter.
        let registered_tx = TransactionId::new(200);
        let spy = ssi_spy::register(&layered, registered_tx);

        let unregistered_tx = TransactionId::new(201);
        let epoch = EpochId::from(u64::MAX);

        let _ = layered.get_node_versioned(base_node, epoch, unregistered_tx);
        assert!(
            !spy.saw_node(base_node),
            "unregistered tx must not produce a recording on the registered tracker"
        );
    }

    // ── Snapshot-isolated base-edge deletes (epoch-versioned tombstone) ──
    //
    // Regression coverage for the bug where a base edge deleted by a
    // concurrent transaction AFTER a reader's snapshot start was hidden from
    // EVERY snapshot (the base-edge tombstone was epoch-blind). The fix gives
    // each base-edge tombstone an (epoch, deleter) stamp so versioned readers
    // apply the same snapshot-isolation boundary as the overlay version chain,
    // while the latest (non-versioned) view keeps hiding any tombstone.

    /// HEADLINE: a base edge G live at tx A's snapshot E0; tx B (a different
    /// tx) deletes G and the delete commits at E1 > E0. A (snapshot E0) must
    /// STILL see G — both via `is_edge_visible_versioned` and via
    /// `edges_from_versioned`. This is the probe that previously FAILED.
    #[test]
    fn base_edge_delete_committed_after_snapshot_is_invisible_only_to_later_snapshots() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (dst, g) = edges[0];

        let tx_a = TransactionId::from(1); // the reader (snapshot E0)
        let tx_b = TransactionId::from(2); // the concurrent deleter
        let e0 = EpochId::new(0); // A's snapshot
        let e1 = EpochId::new(1); // B's commit epoch (> E0)

        // Precondition: G is visible to A at E0.
        assert!(
            layered.is_edge_visible_versioned(g, e0, tx_a),
            "base edge must be visible to A's snapshot before any delete"
        );

        // B deletes G (PENDING) then commits at E1 (drive the finalize path
        // that the session would run: take pending overlay tuples, then
        // finalize by id + commit epoch — the LayeredStore base tombstone is
        // finalized inside its own override).
        assert!(layered.delete_edge_versioned(g, e1, tx_b));
        let pending = layered.take_pending_edge_deletes(tx_b);
        layered.finalize_edge_deletes_by_id(tx_b, e1, &pending);

        // A's snapshot started at E0 (before the delete's commit at E1), so A
        // must STILL see G.
        assert!(
            layered.is_edge_visible_versioned(g, e0, tx_a),
            "snapshot-isolation violated: A (E0) lost a base edge a concurrent tx deleted at E1>E0"
        );
        let a_out = layered.edges_from_versioned(src, Direction::Outgoing, e0, tx_a);
        assert!(
            a_out.iter().any(|&(t, e)| t == dst && e == g),
            "edges_from_versioned for A (E0) must still include the concurrently-deleted base edge"
        );

        // A LATER snapshot (E1, the commit epoch) must NOT see G — boundary is
        // `deleted_epoch <= viewing_epoch` (mirrors VersionInfo::is_visible_at).
        let tx_c = TransactionId::from(3);
        assert!(
            !layered.is_edge_visible_versioned(g, e1, tx_c),
            "a snapshot AT the delete's commit epoch must not see the edge"
        );
        let c_out = layered.edges_from_versioned(src, Direction::Outgoing, e1, tx_c);
        assert!(
            !c_out.iter().any(|&(_, e)| e == g),
            "edges_from_versioned at the commit epoch must exclude the deleted base edge"
        );
    }

    /// An UNCOMMITTED base-edge delete by B is NOT visible to B's own reads
    /// (read-your-writes), but IS still visible to a concurrent A.
    #[test]
    fn uncommitted_base_edge_delete_hidden_from_deleter_visible_to_others() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (_, g) = edges[0];

        let tx_a = TransactionId::from(1);
        let tx_b = TransactionId::from(2);
        let e0 = EpochId::new(0);

        // B deletes G but does NOT commit (still PENDING).
        assert!(layered.delete_edge_versioned(g, e0, tx_b));

        // B's own read must not see G (read-your-writes).
        assert!(
            !layered.is_edge_visible_versioned(g, e0, tx_b),
            "deleter must not see its own uncommitted base-edge delete"
        );
        // Concurrent A must STILL see G (the delete is uncommitted).
        assert!(
            layered.is_edge_visible_versioned(g, e0, tx_a),
            "an uncommitted base-edge delete must remain invisible to other snapshots"
        );
        let a_out = layered.edges_from_versioned(src, Direction::Outgoing, e0, tx_a);
        assert!(
            a_out.iter().any(|&(_, e)| e == g),
            "A's traversal must still include an edge B has only PENDING-deleted"
        );
    }

    #[test]
    fn base_tombstones_are_first_writer_owned_and_never_restamped() {
        let layered = build_test_layered();
        let node = layered.nodes_by_label("Person")[0];
        let edge = layered.edges_from(node, Direction::Outgoing)[0].1;
        let owner = TransactionId::from(701);
        let competitor = TransactionId::from(702);
        let owner_snapshot = EpochId::new(3);
        let committed = EpochId::new(11);

        assert!(layered.delete_node_versioned(node, owner_snapshot, owner));
        assert!(layered.delete_edge_versioned(edge, owner_snapshot, owner));
        assert!(
            !layered.delete_node_versioned(node, owner_snapshot, competitor),
            "a competing transaction must not acquire an existing node tombstone"
        );
        assert!(
            !layered.delete_edge_versioned(edge, owner_snapshot, competitor),
            "a competing transaction must not acquire an existing edge tombstone"
        );
        assert!(
            !layered.delete_node(node),
            "auto-commit repeat must not overwrite a transaction-owned node tombstone"
        );
        assert!(
            !layered.delete_edge(edge),
            "auto-commit repeat must not overwrite a transaction-owned edge tombstone"
        );

        // The bulk base-adjacency path must obey the same first-writer rule;
        // in particular it cannot turn another transaction's PENDING edge into
        // an auto-commit tombstone.
        layered.delete_node_edges(node);
        let node_delete = layered.deleted_from_base_nodes.read()[&node];
        let edge_delete = layered.deleted_from_base_edges.read()[&edge];
        assert_eq!(node_delete.epoch, EpochId::PENDING);
        assert_eq!(node_delete.deleter, Some(owner));
        assert_eq!(edge_delete.epoch, EpochId::PENDING);
        assert_eq!(edge_delete.deleter, Some(owner));
        assert_eq!(
            layered.pending_base_node_deletes.read().get(&owner),
            Some(&vec![node])
        );
        assert_eq!(
            layered.pending_base_edge_deletes.read().get(&owner),
            Some(&vec![edge])
        );
        assert!(
            !layered
                .pending_base_node_deletes
                .read()
                .contains_key(&competitor)
        );
        assert!(
            !layered
                .pending_base_edge_deletes
                .read()
                .contains_key(&competitor)
        );
        assert!(
            layered.is_node_visible_versioned(node, owner_snapshot, competitor),
            "a failed competing delete must still see the owner's uncommitted node"
        );
        assert!(
            layered.is_edge_visible_versioned(edge, owner_snapshot, competitor),
            "a failed competing delete must still see the owner's uncommitted edge"
        );

        layered.finalize_deletes_by_id(owner, committed, &[]);
        layered.finalize_edge_deletes_by_id(owner, committed, &[]);
        let committed_node_delete = layered.deleted_from_base_nodes.read()[&node];
        let committed_edge_delete = layered.deleted_from_base_edges.read()[&edge];
        assert_eq!(committed_node_delete.epoch, committed);
        assert_eq!(committed_node_delete.deleter, Some(owner));
        assert_eq!(committed_edge_delete.epoch, committed);
        assert_eq!(committed_edge_delete.deleter, Some(owner));

        assert!(!layered.delete_node(node));
        assert!(!layered.delete_edge(edge));
        assert!(!layered.delete_node_versioned(node, EpochId::new(30), competitor));
        assert!(!layered.delete_edge_versioned(edge, EpochId::new(30), competitor));
        let repeated_node_delete = layered.deleted_from_base_nodes.read()[&node];
        let repeated_edge_delete = layered.deleted_from_base_edges.read()[&edge];
        assert_eq!(repeated_node_delete.epoch, committed);
        assert_eq!(repeated_node_delete.deleter, Some(owner));
        assert_eq!(repeated_edge_delete.epoch, committed);
        assert_eq!(repeated_edge_delete.deleter, Some(owner));
    }

    /// A transaction's OWN base-edge delete is hidden from itself immediately
    /// (even while PENDING), at its own snapshot epoch.
    #[test]
    fn own_base_edge_delete_hidden_immediately() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (_, g) = edges[0];

        let tx_b = TransactionId::from(2);
        let e0 = EpochId::new(0);

        assert!(layered.is_edge_visible_versioned(g, e0, tx_b));
        assert!(layered.delete_edge_versioned(g, e0, tx_b));
        assert!(
            !layered.is_edge_visible_versioned(g, e0, tx_b),
            "a tx must not see a base edge it just deleted"
        );
        let out = layered.edges_from_versioned(src, Direction::Outgoing, e0, tx_b);
        assert!(
            !out.iter().any(|&(_, e)| e == g),
            "deleter's own traversal must exclude its just-deleted base edge"
        );
    }

    /// Rolling back B's uncommitted base-edge delete (via `drop_tx_overlay`)
    /// restores visibility for everyone.
    #[test]
    fn rolled_back_base_edge_delete_is_restored() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (dst, g) = edges[0];

        let tx_b = TransactionId::from(2);
        let e0 = EpochId::new(0);

        assert!(layered.delete_edge_versioned(g, e0, tx_b));
        assert!(!layered.is_edge_visible_versioned(g, e0, tx_b));

        // Roll back B (the session calls drop_tx_overlay on abort, then
        // rolls back pending edge deletes).
        layered.drop_tx_overlay(tx_b);
        let pending = layered.take_pending_edge_deletes(tx_b);
        layered
            .overlay_store()
            .rollback_pending_edge_deletes(tx_b, &pending);

        assert!(
            layered.is_edge_visible_versioned(g, e0, tx_b),
            "rolled-back base-edge delete must restore visibility to the deleter"
        );
        // Latest (non-versioned) view must also see it again.
        assert!(
            layered.get_edge(g).is_some(),
            "rolled-back base-edge delete must restore the latest-view edge"
        );
        let out = layered.edges_from_versioned(src, Direction::Outgoing, e0, tx_b);
        assert!(out.iter().any(|&(t, e)| t == dst && e == g));
    }

    /// A persisted/seeded base-edge delete (a prior session's committed delete)
    /// is hidden from ALL current snapshots, because every new snapshot starts
    /// after it.
    #[test]
    fn seeded_base_edge_delete_hidden_from_all_snapshots() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (_, g) = edges[0];

        // Seed as a prior-session committed delete (keys-only on-disk format).
        layered.seed_deleted_from_base(std::iter::empty(), std::iter::once(g));

        // Even the earliest possible snapshot (E0) must not see it.
        let tx = TransactionId::from(7);
        let e0 = EpochId::new(0);
        assert!(
            !layered.is_edge_visible_versioned(g, e0, tx),
            "a seeded (prior-session committed) base-edge delete must be hidden from every snapshot"
        );
        assert!(
            layered.get_edge(g).is_none(),
            "seeded base-edge delete must also be hidden from the latest view"
        );
        // Persistence round-trips the key unchanged.
        let snap = layered.snapshot_deleted_edge_ids();
        assert!(
            snap.contains(&g),
            "snapshot_deleted_edge_ids must still return the seeded edge id (keys-only format)"
        );
    }

    #[test]
    fn epoch_seeded_node_and_edge_deletes_preserve_half_open_boundary() {
        let layered = build_test_layered();
        let node = layered.nodes_by_label("Person")[0];
        let edge = layered.edges_from(node, Direction::Outgoing)[0].1;
        let deleted_at = EpochId::new(20);

        layered.seed_deleted_from_base_at_epochs(
            std::iter::once((node, deleted_at)),
            std::iter::once((edge, deleted_at)),
        );

        assert!(layered.get_node_at_epoch(node, EpochId::new(19)).is_some());
        assert!(layered.get_edge_at_epoch(edge, EpochId::new(19)).is_some());
        assert!(layered.get_node_at_epoch(node, deleted_at).is_none());
        assert!(layered.get_edge_at_epoch(edge, deleted_at).is_none());
        assert_eq!(layered.snapshot_deleted_nodes(), vec![(node, deleted_at)]);
        assert_eq!(layered.snapshot_deleted_edges(), vec![(edge, deleted_at)]);
    }

    /// LATEST-view regression guard: a SYSTEM (auto-commit, non-versioned)
    /// base-edge delete must NOT be resurrected by the latest-view
    /// `edges_from` / `neighbors` (the audit fix must stay fixed), and must be
    /// hidden from versioned reads at the current epoch too.
    #[test]
    fn system_base_edge_delete_not_resurrected_in_latest_view() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (dst, g) = edges[0];

        // SYSTEM delete (auto-commit, no tx snapshot).
        assert!(layered.delete_edge(g));

        // Latest view must not resurrect it.
        assert!(layered.get_edge(g).is_none());
        let out = layered.edges_from(src, Direction::Outgoing);
        assert!(
            !out.iter().any(|&(_, e)| e == g),
            "latest-view edges_from must not resurrect a SYSTEM-deleted base edge"
        );
        assert!(
            !layered.neighbors(src, Direction::Outgoing).contains(&dst)
                || layered
                    .edges_from(src, Direction::Outgoing)
                    .iter()
                    .any(|&(t, _)| t == dst),
            "neighbors must agree with edges_from after a SYSTEM base-edge delete"
        );
        // Versioned read at the current epoch must also be hidden.
        let tx = TransactionId::from(9);
        let now = layered.current_epoch();
        assert!(!layered.is_edge_visible_versioned(g, now, tx));
    }

    // ── Fix A: base-EDGE *property* accessors must honor the snapshot-aware
    //          delete predicate (not the epoch-blind latest one) ──
    //
    // An old snapshot can still traverse a base edge a concurrent tx deleted
    // AFTER the snapshot started; its property reads must agree with that
    // topology and STILL return the value, not NULL/empty.

    /// `read_edge_property_visible` (singular): A (E0) reads a base edge's
    /// property after B committed a delete at E1 > E0. A must still get the
    /// value. Previously returned NULL (the accessor used the epoch-blind
    /// `is_edge_deleted_from_base`).
    #[test]
    fn read_edge_property_visible_old_snapshot_sees_concurrently_deleted_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (_, g) = edges[0];
        let since = PropertyKey::new("since");

        let tx_a = TransactionId::from(1); // reader, snapshot E0
        let tx_b = TransactionId::from(2); // concurrent deleter
        let e0 = EpochId::new(0);
        let e1 = EpochId::new(1);

        // Precondition: A sees the value at E0.
        let before = layered.read_edge_property_visible(g, &since, e0, Some(tx_a));
        assert!(
            before.is_some(),
            "base edge property must be visible to A before any delete"
        );

        // B deletes G and commits at E1 > E0.
        assert!(layered.delete_edge_versioned(g, e1, tx_b));
        let pending = layered.take_pending_edge_deletes(tx_b);
        layered.finalize_edge_deletes_by_id(tx_b, e1, &pending);

        // A's snapshot (E0) precedes the commit; A must STILL read the value.
        let after = layered.read_edge_property_visible(g, &since, e0, Some(tx_a));
        assert_eq!(
            after, before,
            "read_edge_property_visible must agree with the snapshot-aware gate: \
             A (E0) keeps the value of a base edge deleted at E1>E0"
        );

        // And a LATER snapshot (E1) must NOT see it (boundary check).
        let tx_c = TransactionId::from(3);
        assert!(
            layered
                .read_edge_property_visible(g, &since, e1, Some(tx_c))
                .is_none(),
            "a snapshot AT the delete's commit epoch must not read the deleted edge's property"
        );
    }

    /// `read_edge_properties_visible` (plural): same scenario; the map must be
    /// non-empty for A (E0). Previously the snapshot-aware gate passed but the
    /// fall-through `self.get_edge(id)` re-applied the epoch-blind latest
    /// predicate and returned an empty map.
    #[test]
    fn read_edge_properties_visible_old_snapshot_sees_concurrently_deleted_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let src = persons[0];
        let edges = layered.edges_from(src, Direction::Outgoing);
        let (_, g) = edges[0];

        let tx_a = TransactionId::from(1);
        let tx_b = TransactionId::from(2);
        let e0 = EpochId::new(0);
        let e1 = EpochId::new(1);

        let before = layered.read_edge_properties_visible(g, e0, Some(tx_a));
        assert!(
            !before.is_empty(),
            "base edge property map must be visible to A before any delete"
        );

        assert!(layered.delete_edge_versioned(g, e1, tx_b));
        let pending = layered.take_pending_edge_deletes(tx_b);
        layered.finalize_edge_deletes_by_id(tx_b, e1, &pending);

        let after = layered.read_edge_properties_visible(g, e0, Some(tx_a));
        assert_eq!(
            after, before,
            "read_edge_properties_visible must agree with the snapshot-aware gate: \
             A (E0) keeps the full property map of a base edge deleted at E1>E0"
        );

        let tx_c = TransactionId::from(3);
        assert!(
            layered
                .read_edge_properties_visible(g, e1, Some(tx_c))
                .is_empty(),
            "a snapshot AT the delete's commit epoch must read an empty property map"
        );
    }

    // ── Fix B: snapshot-isolated base-NODE deletes (epoch-versioned tombstone) ──
    //
    // Mirror of the base-edge tombstone fix. A base node deleted by a concurrent
    // transaction AFTER a reader's snapshot start must stay visible to that
    // reader (topology AND properties/labels), while the latest (non-versioned)
    // view keeps hiding any tombstone (audit base-tier-resurrection guard).

    /// HEADLINE: a base node N live at tx A's snapshot E0; tx B deletes N and
    /// commits at E1 > E0. A (E0) must STILL see N via `is_node_visible_versioned`
    /// AND read its properties/labels; a LATER snapshot (E1) must not.
    #[test]
    fn base_node_delete_committed_after_snapshot_is_invisible_only_to_later_snapshots() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let n = persons[0];
        let name = PropertyKey::new("name");

        let tx_a = TransactionId::from(1); // reader (snapshot E0)
        let tx_b = TransactionId::from(2); // concurrent deleter
        let e0 = EpochId::new(0);
        let e1 = EpochId::new(1);

        // Precondition: N visible + readable to A at E0.
        assert!(
            layered.is_node_visible_versioned(n, e0, tx_a),
            "base node must be visible to A's snapshot before any delete"
        );
        let name_before = layered.read_node_property_visible(n, &name, e0, Some(tx_a));
        let props_before = layered.read_node_properties_visible(n, e0, Some(tx_a));
        let labels_before = layered.read_node_labels_visible(n, e0, Some(tx_a));
        assert!(name_before.is_some());
        assert!(!props_before.is_empty());
        assert!(!labels_before.is_empty());

        // B deletes N (PENDING) and commits at E1 (drive the node finalize path).
        assert!(layered.delete_node_versioned(n, e1, tx_b));
        let pending = layered.take_pending_deletes(tx_b);
        layered.finalize_deletes_by_id(tx_b, e1, &pending);

        // A's snapshot started at E0 (< E1), so A must STILL see + read N.
        assert!(
            layered.is_node_visible_versioned(n, e0, tx_a),
            "snapshot-isolation violated: A (E0) lost a base node a concurrent tx deleted at E1>E0"
        );
        assert_eq!(
            layered.read_node_property_visible(n, &name, e0, Some(tx_a)),
            name_before,
            "A (E0) must keep the property of a base node deleted at E1>E0"
        );
        assert_eq!(
            layered.read_node_properties_visible(n, e0, Some(tx_a)),
            props_before,
            "A (E0) must keep the property map of a base node deleted at E1>E0"
        );
        assert_eq!(
            layered.read_node_labels_visible(n, e0, Some(tx_a)),
            labels_before,
            "A (E0) must keep the labels of a base node deleted at E1>E0"
        );

        // A LATER snapshot (E1) must NOT see N — boundary `deleted_epoch <= viewing_epoch`.
        let tx_c = TransactionId::from(3);
        assert!(
            !layered.is_node_visible_versioned(n, e1, tx_c),
            "a snapshot AT the delete's commit epoch must not see the node"
        );
        assert!(
            layered
                .read_node_property_visible(n, &name, e1, Some(tx_c))
                .is_none(),
            "a snapshot AT the commit epoch must read no property"
        );
        assert!(
            layered
                .read_node_properties_visible(n, e1, Some(tx_c))
                .is_empty(),
            "a snapshot AT the commit epoch must read an empty property map"
        );
        assert!(
            layered
                .read_node_labels_visible(n, e1, Some(tx_c))
                .is_empty(),
            "a snapshot AT the commit epoch must read no labels"
        );
    }

    /// An UNCOMMITTED base-node delete by B is NOT visible to B's own reads
    /// (read-your-writes) but IS still visible to a concurrent A.
    #[test]
    fn uncommitted_base_node_delete_hidden_from_deleter_visible_to_others() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let n = persons[0];

        let tx_a = TransactionId::from(1);
        let tx_b = TransactionId::from(2);
        let e0 = EpochId::new(0);

        // B deletes N but does NOT commit (still PENDING).
        assert!(layered.delete_node_versioned(n, e0, tx_b));

        // B's own read must not see N.
        assert!(
            !layered.is_node_visible_versioned(n, e0, tx_b),
            "deleter must not see its own uncommitted base-node delete"
        );
        // Concurrent A must STILL see N (the delete is uncommitted).
        assert!(
            layered.is_node_visible_versioned(n, e0, tx_a),
            "an uncommitted base-node delete must remain invisible to other snapshots"
        );
        assert!(
            layered
                .read_node_property_visible(n, &PropertyKey::new("name"), e0, Some(tx_a))
                .is_some(),
            "A must still read the property of a node B has only PENDING-deleted"
        );
    }

    /// Rolling back B's uncommitted base-node delete (via `drop_tx_overlay`)
    /// restores visibility for everyone, including the latest view.
    #[test]
    fn rolled_back_base_node_delete_is_restored() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let n = persons[0];

        let tx_b = TransactionId::from(2);
        let e0 = EpochId::new(0);

        assert!(layered.delete_node_versioned(n, e0, tx_b));
        assert!(!layered.is_node_visible_versioned(n, e0, tx_b));

        // Roll back B (abort path: drop_tx_overlay, then rollback pending node deletes).
        layered.drop_tx_overlay(tx_b);
        let pending = layered.take_pending_deletes(tx_b);
        layered
            .overlay_store()
            .rollback_pending_deletes(tx_b, &pending);

        assert!(
            layered.is_node_visible_versioned(n, e0, tx_b),
            "rolled-back base-node delete must restore visibility to the deleter"
        );
        assert!(
            layered.get_node(n).is_some(),
            "rolled-back base-node delete must restore the latest-view node"
        );
    }

    /// A persisted/seeded base-node delete (a prior session's committed delete)
    /// is hidden from ALL current snapshots and the latest view.
    #[test]
    fn seeded_base_node_delete_hidden_from_all_snapshots() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let n = persons[0];

        // Seed as a prior-session committed delete (keys-only on-disk format).
        layered.seed_deleted_from_base(std::iter::once(n), std::iter::empty());

        let tx = TransactionId::from(7);
        let e0 = EpochId::new(0);
        assert!(
            !layered.is_node_visible_versioned(n, e0, tx),
            "a seeded (prior-session committed) base-node delete must be hidden from every snapshot"
        );
        assert!(
            layered.get_node(n).is_none(),
            "seeded base-node delete must also be hidden from the latest view"
        );
        let snap = layered.snapshot_deleted_node_ids();
        assert!(
            snap.contains(&n),
            "snapshot_deleted_node_ids must still return the seeded node id (keys-only format)"
        );
    }

    /// LATEST-view regression guard: a SYSTEM (auto-commit, non-versioned)
    /// base-node delete must NOT be resurrected by the latest view, and must be
    /// hidden from versioned reads at the current epoch too.
    #[test]
    fn system_base_node_delete_not_resurrected_in_latest_view() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let n = persons[0];

        // SYSTEM delete (auto-commit, no tx snapshot).
        assert!(layered.delete_node(n));

        // Latest view must not resurrect it.
        assert!(layered.get_node(n).is_none());
        assert!(
            layered
                .get_node_property(n, &PropertyKey::new("name"))
                .is_none()
        );
        assert!(
            !layered.nodes_by_label("Person").contains(&n),
            "latest-view nodes_by_label must not resurrect a SYSTEM-deleted base node"
        );
        // Versioned read at the current epoch must also be hidden.
        let tx = TransactionId::from(9);
        let now = layered.current_epoch();
        assert!(!layered.is_node_visible_versioned(n, now, tx));
    }

    // ── Concurrent-promotion correctness ────────────────────────────────────

    /// Single-threaded: `create_node_with_id` inserts at the exact base id and
    /// a subsequent normal `create_node` gets a non-colliding fresh id.
    ///
    /// Validates the core invariant that `create_node_with_id` never touches
    /// the counter, so the counter still returns the next-fresh id afterwards.
    #[test]
    fn test_create_node_with_id_does_not_disturb_counter() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_id = persons[0];

        let counter_before = layered.overlay.load().next_node_id();

        // Trigger promotion (internally calls create_node_with_id).
        layered.set_node_property(base_id, "promoted", Value::Bool(true));

        // Counter must be unchanged.
        let counter_after = layered.overlay.load().next_node_id();
        assert_eq!(
            counter_before, counter_after,
            "create_node_with_id must not modify next_node_id"
        );

        // The promoted node must be visible at the exact base id.
        let promoted = layered.overlay.load().get_node(base_id);
        assert!(
            promoted.is_some(),
            "promoted node must be present in overlay at base id"
        );
        assert_eq!(promoted.unwrap().id, base_id);

        // A fresh allocation must not collide with any base id.
        let fresh = layered.create_node(&["Person"]);
        for &p in &persons {
            assert_ne!(fresh, p, "fresh id must not alias a base id");
        }
        assert_ne!(fresh, base_id, "fresh id must not alias the promoted id");
    }

    /// Single-threaded: `create_edge_with_id` inserts at the exact base edge
    /// id and the counter is unchanged.
    #[test]
    fn test_create_edge_with_id_does_not_disturb_counter() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert!(!edges.is_empty(), "test requires at least one base edge");
        let (_, base_eid) = edges[0];

        let counter_before = layered.overlay.load().next_edge_id();

        // Trigger edge promotion (internally calls create_edge_with_id).
        layered.set_edge_property(base_eid, "promoted", Value::Bool(true));

        // Counter must be unchanged.
        let counter_after = layered.overlay.load().next_edge_id();
        assert_eq!(
            counter_before, counter_after,
            "create_edge_with_id must not modify next_edge_id"
        );

        // The promoted edge must be visible at the exact base id.
        let promoted = layered.overlay.load().get_edge(base_eid);
        assert!(
            promoted.is_some(),
            "promoted edge must be present in overlay at base id"
        );
        assert_eq!(promoted.unwrap().id, base_eid);
    }

    /// Concurrent: N threads each trigger ensure_in_overlay on a distinct base
    /// node. After joining all threads every promoted node must be visible at
    /// its correct id, and no id aliasing must have occurred.
    ///
    /// This test is racy and fails non-deterministically on the old
    /// save/lower/restore implementation; it is guaranteed to pass with
    /// `create_node_with_id` because that path never mutates the counter.
    #[test]
    #[allow(clippy::cast_possible_wrap)] // loop index i is 0..8
    fn test_concurrent_ensure_in_overlay_no_aliasing() {
        use std::sync::Arc;

        // Build a layered store with a few base nodes to promote concurrently.
        let store = LpgStore::new().unwrap();
        let mut base_ids = Vec::new();
        for i in 0..8u64 {
            let n = store.create_node(&["Worker"]);
            store.set_node_property(n, "idx", Value::Int64(i as i64));
            base_ids.push(n);
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let max_nid = base_ids.iter().map(|id| id.as_u64()).max().unwrap_or(0);
        let layered = Arc::new(LayeredStore::new(compact, max_nid, 0).unwrap());

        // Spawn one thread per base node, each promoting its own node by
        // setting a property (triggers ensure_in_overlay).
        let handles: Vec<_> = base_ids
            .iter()
            .copied()
            .map(|nid| {
                let layered = Arc::clone(&layered);
                std::thread::spawn(move || {
                    layered.set_node_property(nid, "promoted", Value::Bool(true));
                })
            })
            .collect();

        for h in handles {
            h.join()
                .expect("thread panicked during concurrent promotion");
        }

        // Every base node must now be correctly promoted.
        for nid in &base_ids {
            let node = layered
                .overlay
                .load()
                .get_node(*nid)
                .unwrap_or_else(|| panic!("node {nid:?} missing from overlay after promotion"));
            assert_eq!(
                node.id, *nid,
                "promoted node id mismatch: expected {nid:?}, got {:?}",
                node.id
            );
        }

        // Collect all overlay node ids and verify no duplicates (aliasing).
        let mut all_overlay_ids = layered.overlay.load().all_node_ids();
        all_overlay_ids.sort_unstable();
        all_overlay_ids.dedup();
        // All 8 base nodes must appear exactly once.
        for nid in &base_ids {
            assert!(
                all_overlay_ids.contains(nid),
                "base node {nid:?} missing from overlay id list after concurrent promotion"
            );
        }
        // No extra entries beyond what we promoted.
        assert_eq!(
            all_overlay_ids.len(),
            base_ids.len(),
            "overlay must contain exactly the promoted nodes, found aliases or extra entries"
        );

        // A fresh node allocation must not collide with any base id.
        let fresh = layered.create_node(&["Worker"]);
        for nid in &base_ids {
            assert_ne!(fresh, *nid, "fresh id must not alias a base id");
        }
    }

    /// Concurrent writers targeting one cold node must share one promotion.
    /// Re-running hydration would duplicate the authentic cold event log even
    /// if every writer's independent mutation happened to remain visible.
    #[test]
    fn concurrent_same_node_promotion_hydrates_cold_history_once() {
        use std::sync::{Arc, Barrier};

        const WRITERS: usize = 16;
        const HISTORY_EVENTS: usize = 4;

        let layered = empty_layered();
        let cold_source = layered.overlay_store();
        let created = EpochId::new(10);
        cold_source.set_epoch(created);
        let node = cold_source.create_node(&["Contended"]);
        for _ in 0..HISTORY_EVENTS {
            cold_source.set_node_property_at_epoch(node, "stable", Value::Bool(true), created);
        }
        layered
            .merge_overlay_temporal()
            .expect("build exact cold promotion fixture");
        assert_eq!(
            layered
                .base_store_arc()
                .node_property_history(node)
                .into_iter()
                .find(|(key, _)| key.as_str() == "stable")
                .map_or(0, |(_, versions)| versions.len()),
            HISTORY_EVENTS,
            "fixture must expose every authentic cold event"
        );
        let layered = Arc::new(layered);
        layered.overlay_store().set_epoch(EpochId::new(20));
        let start = Arc::new(Barrier::new(WRITERS));
        *layered.node_promotion_barrier.write() = Some(Arc::clone(&start));

        let handles: Vec<_> = (0..WRITERS)
            .map(|writer| {
                let layered = Arc::clone(&layered);
                std::thread::spawn(move || {
                    layered.set_node_property(
                        node,
                        &format!("writer_{writer}"),
                        Value::Int64(i64::try_from(writer).expect("writer index fits i64")),
                    );
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("same-node writer must not panic");
        }

        let overlay = layered.overlay_store();
        assert_eq!(
            overlay.node_property_history_for_key(node, "stable").len(),
            HISTORY_EVENTS,
            "cold history must be replayed by exactly one promoter"
        );
        assert_eq!(overlay.node_count(), 1, "promotion must create one hot row");
        assert_eq!(overlay.node_ids(), vec![node]);
        for writer in 0..WRITERS {
            assert_eq!(
                overlay.get_node_property(node, &PropertyKey::new(format!("writer_{writer}"))),
                Some(Value::Int64(
                    i64::try_from(writer).expect("writer index fits i64"),
                )),
                "writer {writer} must survive the shared promotion"
            );
        }
    }

    /// Concurrent writers targeting one cold edge must share one promotion,
    /// including the endpoint hydration that precedes edge publication.
    #[test]
    fn concurrent_same_edge_promotion_hydrates_cold_history_once() {
        use std::sync::{Arc, Barrier};

        const WRITERS: usize = 16;
        const HISTORY_EVENTS: usize = 4;

        let layered = empty_layered();
        let cold_source = layered.overlay_store();
        let created = EpochId::new(10);
        cold_source.set_epoch(created);
        let src = cold_source.create_node(&["Source"]);
        let dst = cold_source.create_node(&["Target"]);
        let edge = cold_source.create_edge(src, dst, "CONTENDED");
        for _ in 0..HISTORY_EVENTS {
            cold_source.set_edge_property_at_epoch(edge, "stable", Value::Bool(true), created);
        }
        layered
            .merge_overlay_temporal()
            .expect("build exact cold edge-promotion fixture");
        assert_eq!(
            layered
                .edge_full_history(edge)
                .properties
                .get(&PropertyKey::new("stable"))
                .map_or(0, Vec::len),
            HISTORY_EVENTS,
            "fixture must expose every authentic cold edge event"
        );

        let layered = Arc::new(layered);
        layered.overlay_store().set_epoch(EpochId::new(20));
        let start = Arc::new(Barrier::new(WRITERS));
        *layered.edge_promotion_barrier.write() = Some(Arc::clone(&start));
        let handles: Vec<_> = (0..WRITERS)
            .map(|writer| {
                let layered = Arc::clone(&layered);
                std::thread::spawn(move || {
                    layered.set_edge_property(
                        edge,
                        &format!("writer_{writer}"),
                        Value::Int64(i64::try_from(writer).expect("writer index fits i64")),
                    );
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("same-edge writer must not panic");
        }

        let overlay = layered.overlay_store();
        let stable_history = overlay
            .edge_property_history(edge)
            .into_iter()
            .find(|(key, _)| key.as_str() == "stable")
            .map_or_else(Vec::new, |(_, versions)| versions);
        assert_eq!(
            stable_history.len(),
            HISTORY_EVENTS,
            "cold edge history must be replayed by exactly one promoter"
        );
        assert_eq!(
            overlay.edge_count(),
            1,
            "promotion must create one hot edge"
        );
        assert_eq!(
            overlay.node_count(),
            2,
            "edge promotion must hydrate each endpoint once"
        );
        let mut endpoint_ids = overlay.node_ids();
        endpoint_ids.sort_unstable();
        let mut expected_endpoint_ids = vec![src, dst];
        expected_endpoint_ids.sort_unstable();
        assert_eq!(endpoint_ids, expected_endpoint_ids);
        let promoted = overlay.get_edge(edge).expect("promoted edge must exist");
        assert_eq!((promoted.src, promoted.dst), (src, dst));
        for writer in 0..WRITERS {
            assert_eq!(
                overlay.get_edge_property(edge, &PropertyKey::new(format!("writer_{writer}"))),
                Some(Value::Int64(
                    i64::try_from(writer).expect("writer index fits i64"),
                )),
                "writer {writer} must survive the shared edge promotion"
            );
        }
    }

    /// A fully hydrated overlay row is still private promotion state until its
    /// dirty marker is published. Aggregate readers must therefore observe the
    /// same single logical node on both sides of that publication point.
    #[test]
    fn node_count_hides_unpublished_promotion_row() {
        let layered = empty_layered();
        let node = layered.overlay_store().create_node(&["Cold"]);
        layered
            .merge_overlay_temporal()
            .expect("build cold node fixture");

        let layered = Arc::new(layered);
        let (hook, mut pause) = promotion_pause();
        *layered.node_publication_hook.write() = Some(hook);

        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(node, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let count_before_publication = layered.node_count();
        pause.release();
        writer.join().expect("node promoter must not panic");

        assert_eq!(
            count_before_publication, 1,
            "a hydrated but unpublished base-overlap row must not be counted twice"
        );
        assert_eq!(layered.node_count(), 1);
        assert_eq!(
            layered.get_node_property(node, &PropertyKey::new("writer")),
            Some(Value::Bool(true))
        );
    }

    /// Promotion reconciles equality indexes and label metadata before the
    /// dirty marker becomes visible. Those derived structures must not expose
    /// or double-count the private overlay copy.
    #[test]
    fn node_search_and_statistics_hide_unpublished_promotion_row() {
        let layered = empty_layered();
        let node = layered.overlay_store().create_node(&["Cold"]);
        layered
            .overlay_store()
            .set_node_property(node, "code", Value::Int64(7));
        layered
            .merge_overlay_temporal()
            .expect("build indexed cold node fixture");

        let layered = Arc::new(layered);
        let (hook, mut pause) = promotion_pause();
        *layered.node_publication_hook.write() = Some(hook);

        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(node, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let search_before_publication = layered.find_nodes_by_property("code", &Value::Int64(7));
        let stats_before_publication = layered.statistics();
        let estimate_before_publication = layered.estimate_label_cardinality("Cold");
        pause.release();
        writer.join().expect("node promoter must not panic");

        assert_eq!(search_before_publication, vec![node]);
        assert_eq!(stats_before_publication.total_nodes, 1);
        assert_eq!(
            stats_before_publication
                .get_label("Cold")
                .expect("cold label statistics must exist")
                .node_count,
            1
        );
        assert_eq!(estimate_before_publication, 1.0);
        assert_eq!(
            layered.find_nodes_by_property("code", &Value::Int64(7)),
            vec![node]
        );
        assert_eq!(
            layered
                .statistics()
                .get_label("Cold")
                .expect("published label statistics must exist")
                .node_count,
            1
        );
        assert_eq!(layered.estimate_label_cardinality("Cold"), 1.0);
    }

    /// A reader racing exact history hydration must see either the complete
    /// cold history or the complete published overlay history, never a replay
    /// prefix that happens to own a same-epoch group.
    #[test]
    fn node_history_hides_partial_promotion_replay() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let layered = empty_layered();
        let source = layered.overlay_store();
        let event_epoch = EpochId::new(10);
        source.set_epoch(event_epoch);
        let node = source.create_node(&["Temporal"]);
        source.set_node_property_at_epoch(node, "state", Value::Int64(1), event_epoch);
        source.set_node_property_at_epoch(node, "state", Value::Int64(2), event_epoch);
        source.set_node_property_at_epoch(node, "state", Value::Int64(3), event_epoch);
        layered
            .merge_overlay_temporal()
            .expect("build exact cold history fixture");
        let expected = layered.complete_node_property_history_for_key(node, "state");
        assert_eq!(expected.len(), 3, "fixture must retain all exact events");

        let layered = Arc::new(layered);
        let (pause_hook, mut pause) = promotion_pause();
        let calls = Arc::new(AtomicUsize::new(0));
        *layered.node_replay_event_hook.write() = Some(Arc::new({
            let calls = Arc::clone(&calls);
            move || {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    pause_hook();
                }
            }
        }));

        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(node, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let history_during_replay = layered.complete_node_property_history_for_key(node, "state");
        pause.release();
        writer.join().expect("node promoter must not panic");

        assert_eq!(history_during_replay, expected);
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "state"),
            expected
        );
    }

    /// Edge history has the same exact-event publication contract as node
    /// history: a same-epoch group cannot be replaced by a replay prefix.
    #[test]
    fn edge_history_hides_partial_promotion_replay() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let layered = empty_layered();
        let source = layered.overlay_store();
        let event_epoch = EpochId::new(10);
        source.set_epoch(event_epoch);
        let from = source.create_node(&["Source"]);
        let to = source.create_node(&["Target"]);
        let edge = source.create_edge(from, to, "TEMPORAL");
        source.set_edge_property_at_epoch(edge, "state", Value::Int64(1), event_epoch);
        source.set_edge_property_at_epoch(edge, "state", Value::Int64(2), event_epoch);
        source.set_edge_property_at_epoch(edge, "state", Value::Int64(3), event_epoch);
        layered
            .merge_overlay_temporal()
            .expect("build exact cold edge-history fixture");
        let key = PropertyKey::new("state");
        let expected = layered
            .edge_full_history(edge)
            .properties
            .get(&key)
            .cloned()
            .expect("fixture edge history must exist");
        assert_eq!(expected.len(), 3, "fixture must retain all exact events");

        let layered = Arc::new(layered);
        let (pause_hook, mut pause) = promotion_pause();
        let calls = Arc::new(AtomicUsize::new(0));
        *layered.edge_replay_event_hook.write() = Some(Arc::new({
            let calls = Arc::clone(&calls);
            move || {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    pause_hook();
                }
            }
        }));
        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_edge_property(edge, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let history_during_replay = layered
            .edge_full_history(edge)
            .properties
            .get(&key)
            .cloned()
            .expect("cold edge history must remain visible during replay");
        pause.release();
        writer.join().expect("edge promoter must not panic");

        assert_eq!(history_during_replay, expected);
        assert_eq!(
            layered
                .edge_full_history(edge)
                .properties
                .get(&key)
                .cloned(),
            Some(expected)
        );
    }

    #[test]
    fn caught_node_replay_panic_rolls_back_private_hydration_and_retry_succeeds() {
        let layered = empty_layered();
        let source = layered.overlay_store();
        let created = EpochId::new(10);
        let changed = EpochId::new(15);
        source.set_epoch(created);
        let node = source.create_node(&["Cold"]);
        source.set_node_property_at_epoch(node, "state", Value::Int64(1), created);
        source.set_node_property_at_epoch(node, "state", Value::Int64(2), changed);
        source.set_epoch(changed);
        layered
            .merge_overlay_temporal()
            .expect("build replay-unwind node fixture");
        let expected = layered.node_structural_history(node);
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(20));
        let hostile_attempts = 3;
        let replay_calls = Arc::new(AtomicUsize::new(0));
        *layered.node_replay_event_hook.write() = Some(Arc::new({
            let replay_calls = Arc::clone(&replay_calls);
            move || {
                assert!(
                    replay_calls.fetch_add(1, Ordering::SeqCst) >= hostile_attempts,
                    "hostile node replay unwind"
                );
            }
        }));

        let next_node_id = overlay.next_node_id();
        let label_count = overlay.label_count();
        for attempt in 0..hostile_attempts {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                layered.set_node_property(node, "writer", Value::Bool(true));
            }));
            assert!(panic.is_err(), "hostile node attempt {attempt} must unwind");
            assert!(!layered.dirty_node_ids.read().contains(&node));
            assert!(overlay.get_node(node).is_none());
            assert!(overlay.node_property_history(node).is_empty());
            assert_eq!(
                overlay.next_node_id(),
                next_node_id,
                "rolled-back promotion must not consume a node identity"
            );
            assert_eq!(
                overlay.label_count(),
                label_count,
                "rollback must restore the exact pre-promotion label catalog"
            );
            let after_unwind = layered.node_structural_history(node);
            assert_eq!(after_unwind.lifetimes, expected.lifetimes);
            assert_eq!(after_unwind.label_versions, expected.label_versions);
            assert_eq!(after_unwind.properties, expected.properties);
        }
        layered.set_node_property(node, "writer", Value::Bool(true));
        assert!(layered.dirty_node_ids.read().contains(&node));
        assert_eq!(overlay.label_count(), label_count + 1);
        assert_eq!(
            layered.get_node_property(node, &PropertyKey::new("writer")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            layered.complete_node_property_history_for_key(node, "state"),
            vec![(created, Value::Int64(1)), (changed, Value::Int64(2))],
            "retry must hydrate the exact cold history once"
        );
        assert_eq!(
            layered
                .get_node_at_epoch(node, created)
                .expect("node remains visible at its original creation")
                .labels
                .into_iter()
                .collect::<Vec<_>>(),
            expected.labels
        );
    }

    fn historical_property_miss_work(layered: &LayeredStore, epoch: EpochId) -> u64 {
        use crate::graph::{GraphStoreSearch, PropertyIndexPredicate, PropertyIndexRequest};
        let overlay = layered.overlay_store();
        let before = overlay.work_snapshot();
        assert_eq!(
            layered
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "state",
                    predicate: PropertyIndexPredicate::Equal(&Value::Int64(1)),
                    epoch,
                    transaction_id: None,
                })
                .unwrap(),
            Some(Vec::new())
        );
        overlay
            .work_snapshot()
            .since(before)
            .property_index_posting_intervals
    }

    #[test]
    fn property_index_hydration_unwind_and_retry_preserve_full_postings() {
        // One retained interval: one interval-directory node inspection.
        let layered = empty_layered();
        let source = layered.overlay_store();
        source.set_epoch(EpochId::new(10));
        let node = source.create_node(&["Cold"]);
        source.set_node_property(node, "state", Value::Int64(1));
        source.set_epoch(EpochId::new(15));
        source.set_node_property(node, "state", Value::Int64(2));
        source.create_property_index("state");
        layered.merge_overlay_temporal().unwrap();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(20));
        assert_eq!(historical_property_miss_work(&layered, EpochId::new(20)), 1);
        *layered.node_replay_event_hook.write() =
            Some(Arc::new(|| panic!("abort private property replay")));
        for _ in 0..3 {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    layered.set_node_property(node, "trigger", Value::Bool(true));
                }))
                .is_err()
            );
            assert!(overlay.get_node(node).is_none());
            assert_eq!(
                overlay.find_nodes_by_property("state", &Value::Int64(2)),
                vec![node],
                "hydration rollback must retain valid cold current membership"
            );
            assert!(
                overlay
                    .find_nodes_by_property("state", &Value::Int64(1))
                    .is_empty()
            );
            assert_eq!(
                historical_property_miss_work(&layered, EpochId::new(20)),
                1,
                "hydration rollback must not append/reopen historical intervals"
            );
        }
        *layered.node_replay_event_hook.write() = None;
        layered.set_node_property(node, "trigger", Value::Bool(true));
        assert_eq!(
            overlay.find_nodes_by_property("state", &Value::Int64(2)),
            vec![node]
        );
        assert_eq!(historical_property_miss_work(&layered, EpochId::new(20)), 1);
    }

    #[test]
    fn property_index_retained_hot_reseed_does_not_duplicate_history() {
        // One retained interval: one interval-directory node inspection.
        let layered = empty_layered();
        let source = layered.overlay_store();
        source.set_epoch(EpochId::new(10));
        let node = source.create_node(&["Hot"]);
        source.set_node_property(node, "state", Value::Int64(1));
        source.set_epoch(EpochId::new(15));
        source.set_node_property(node, "state", Value::Int64(2));
        source.create_property_index("state");
        assert_eq!(historical_property_miss_work(&layered, EpochId::new(15)), 1);
        for _ in 0..3 {
            layered
                .merge_overlay_temporal_retaining(EpochId::new(10))
                .unwrap();
            assert!(
                layered.overlay_store().get_node(node).is_some(),
                "fixture must reseed a hot node"
            );
            assert_eq!(
                historical_property_miss_work(&layered, EpochId::new(15)),
                1,
                "replaying already indexed history must not duplicate interval work"
            );
            assert_eq!(
                layered
                    .overlay_store()
                    .find_nodes_by_property("state", &Value::Int64(2)),
                vec![node]
            );
        }
    }

    #[test]
    fn indexed_cold_hit_survives_partially_replayed_private_hydration() {
        use crate::graph::{GraphStoreSearch, PropertyIndexPredicate, PropertyIndexRequest};
        use std::sync::mpsc;
        use std::time::Duration;

        let layered = Arc::new(empty_layered());
        let source = layered.overlay_store();
        source.set_epoch(EpochId::new(10));
        let node = source.create_node(&["Cold"]);
        source.set_node_property(node, "state", Value::Int64(1));
        source.set_epoch(EpochId::new(15));
        source.set_node_property(node, "state", Value::Int64(2));
        source.create_property_index("state");
        layered.merge_overlay_temporal().unwrap();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(20));

        let (paused_tx, paused_rx) = mpsc::sync_channel(1);
        let (resume_tx, resume_rx) = mpsc::sync_channel(1);
        let resume_rx = parking_lot::Mutex::new(resume_rx);
        let first = AtomicBool::new(true);
        *layered.node_replay_event_hook.write() = Some(Arc::new(move || {
            if first.swap(false, Ordering::SeqCst) {
                paused_tx.send(()).unwrap();
                resume_rx
                    .lock()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
        }));
        let writer = std::thread::spawn({
            let layered = Arc::clone(&layered);
            move || layered.set_node_property(node, "trigger", Value::Bool(true))
        });
        paused_rx.recv_timeout(Duration::from_secs(10)).unwrap();

        // A exists in the private physical row, but B remains authoritative in
        // the logical cold generation until the dirty marker publishes.
        let partial = overlay.get_node_property(node, &PropertyKey::new("state"));
        let bound = Value::Int64(2);
        let equal = layered.lookup_nodes_indexed(PropertyIndexRequest {
            property: "state",
            predicate: PropertyIndexPredicate::Equal(&bound),
            epoch: EpochId::new(20),
            transaction_id: None,
        });
        let range = layered.lookup_nodes_indexed(PropertyIndexRequest {
            property: "state",
            predicate: PropertyIndexPredicate::Range {
                min: Some(&bound),
                max: Some(&bound),
                min_inclusive: true,
                max_inclusive: true,
            },
            epoch: EpochId::new(20),
            transaction_id: None,
        });
        // Release the writer before assertions so a regression cannot strand it.
        resume_tx.send(()).unwrap();
        writer.join().unwrap();
        *layered.node_replay_event_hook.write() = None;
        assert_eq!(
            partial,
            Some(Value::Int64(1)),
            "fixture must expose a half-replayed row"
        );
        assert_eq!(equal.unwrap(), Some(vec![node]));
        assert_eq!(range.unwrap(), Some(vec![node]));
    }

    #[test]
    fn caught_edge_replay_panic_rolls_back_private_hydration_and_retry_succeeds() {
        let layered = empty_layered();
        let source = layered.overlay_store();
        let created = EpochId::new(10);
        let changed = EpochId::new(15);
        source.set_epoch(created);
        let from = source.create_node(&["Source"]);
        let to = source.create_node(&["Target"]);
        let edge = source.create_edge(from, to, "TEMPORAL");
        source.set_edge_property_at_epoch(edge, "state", Value::Int64(1), created);
        source.set_edge_property_at_epoch(edge, "state", Value::Int64(2), changed);
        source.set_epoch(changed);
        layered
            .merge_overlay_temporal()
            .expect("build replay-unwind edge fixture");
        let expected = layered.edge_full_history(edge);
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(20));
        let hostile_attempts = 3;
        let replay_calls = Arc::new(AtomicUsize::new(0));
        *layered.edge_replay_event_hook.write() = Some(Arc::new({
            let replay_calls = Arc::clone(&replay_calls);
            move || {
                assert!(
                    replay_calls.fetch_add(1, Ordering::SeqCst) >= hostile_attempts,
                    "hostile edge replay unwind"
                );
            }
        }));

        let next_edge_id = overlay.next_edge_id();
        let edge_type_count = overlay.edge_type_count();
        for attempt in 0..hostile_attempts {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                layered.set_edge_property(edge, "writer", Value::Bool(true));
            }));
            assert!(panic.is_err(), "hostile edge attempt {attempt} must unwind");
            assert!(!layered.dirty_edge_ids.read().contains(&edge));
            assert!(overlay.get_edge(edge).is_none());
            assert!(overlay.edge_property_history(edge).is_empty());
            assert_eq!(
                overlay.next_edge_id(),
                next_edge_id,
                "rolled-back promotion must not consume an edge identity"
            );
            assert_eq!(
                overlay.edge_type_count(),
                edge_type_count,
                "rollback must restore the exact pre-promotion edge-type catalog"
            );
            let after_unwind = layered.edge_full_history(edge);
            assert_eq!(after_unwind.lifetimes, expected.lifetimes);
            assert_eq!(after_unwind.properties, expected.properties);
        }
        layered.set_edge_property(edge, "writer", Value::Bool(true));
        assert!(layered.dirty_edge_ids.read().contains(&edge));
        assert_eq!(overlay.edge_type_count(), edge_type_count + 1);
        assert_eq!(
            layered.get_edge_property(edge, &PropertyKey::new("writer")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            layered
                .edge_full_history(edge)
                .properties
                .get(&PropertyKey::new("state")),
            Some(&vec![
                (created, Value::Int64(1)),
                (changed, Value::Int64(2)),
            ]),
            "retry must hydrate the exact cold edge history once"
        );
        assert!(layered.get_edge_at_epoch(edge, created).is_some());
    }

    /// A base-owned property miss is authoritative until promotion publishes.
    /// Falling through to a partial replay would transiently resurrect a value
    /// that the cold history has already tombstoned.
    #[test]
    fn node_property_reads_hide_partial_promotion_replay() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let layered = empty_layered();
        let source = layered.overlay_store();
        let node = source.create_node(&["Temporal"]);
        source.set_node_property_at_epoch(node, "retired", Value::Int64(1), EpochId::new(10));
        source.set_node_property_at_epoch(node, "retired", Value::Null, EpochId::new(20));
        source.set_epoch(EpochId::new(20));
        layered
            .merge_overlay_temporal()
            .expect("build tombstoned node-property fixture");
        let key = PropertyKey::new("retired");
        assert_eq!(
            layered
                .complete_node_property_history_for_key(node, "retired")
                .len(),
            2,
            "fixture must retain the value and its tombstone"
        );
        assert_eq!(layered.get_node_property(node, &key), None);

        let layered = Arc::new(layered);
        let (pause_hook, mut pause) = promotion_pause();
        let calls = Arc::new(AtomicUsize::new(0));
        *layered.node_replay_event_hook.write() = Some(Arc::new({
            let calls = Arc::clone(&calls);
            move || {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    pause_hook();
                }
            }
        }));

        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(node, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let current_during_replay = layered.get_node_property(node, &key);
        let as_of_during_replay = layered.get_node_property_at_epoch(node, &key, EpochId::new(20));
        pause.release();
        writer.join().expect("node promoter must not panic");

        assert_eq!(current_during_replay, None);
        assert_eq!(as_of_during_replay, None);
        assert_eq!(layered.get_node_property(node, &key), None);
    }

    /// Edge property routing obeys the same base-authority rule as nodes; a
    /// partial exact replay cannot resurrect an already-removed edge value.
    #[test]
    fn edge_property_reads_hide_partial_promotion_replay() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let layered = empty_layered();
        let source = layered.overlay_store();
        let from = source.create_node(&["Source"]);
        let to = source.create_node(&["Target"]);
        let edge = source.create_edge(from, to, "TEMPORAL");
        source.set_edge_property_at_epoch(edge, "retired", Value::Int64(1), EpochId::new(10));
        source.set_edge_property_at_epoch(edge, "retired", Value::Null, EpochId::new(20));
        source.set_epoch(EpochId::new(20));
        layered
            .merge_overlay_temporal()
            .expect("build tombstoned edge-property fixture");
        let key = PropertyKey::new("retired");
        assert_eq!(
            layered
                .edge_full_history(edge)
                .properties
                .get(&key)
                .map_or(0, Vec::len),
            2,
            "fixture must retain the value and its tombstone"
        );
        assert_eq!(layered.get_edge_property(edge, &key), None);

        let layered = Arc::new(layered);
        let (pause_hook, mut pause) = promotion_pause();
        let calls = Arc::new(AtomicUsize::new(0));
        *layered.edge_replay_event_hook.write() = Some(Arc::new({
            let calls = Arc::clone(&calls);
            move || {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    pause_hook();
                }
            }
        }));

        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_edge_property(edge, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let value_during_replay = layered.get_edge_property(edge, &key);
        pause.release();
        writer.join().expect("edge promoter must not panic");

        assert_eq!(value_during_replay, None);
        assert_eq!(layered.get_edge_property(edge, &key), None);
    }

    /// Whole-graph Text state is unchanged during hydration. The valid cold
    /// hit must remain visible before and after its overlay routing publishes.
    #[cfg(feature = "text-index")]
    #[test]
    fn text_search_preserves_cold_hit_during_hydration() -> Result<(), Box<dyn std::error::Error>> {
        use crate::index::text::{BM25Config, InvertedIndex};

        let layered = empty_layered();
        let cold = layered.overlay_store();
        cold.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );
        let promoted = cold.create_node(&["Doc"]);
        cold.set_node_property(promoted, "body", Value::from("needle needle needle"));
        layered
            .merge_overlay_temporal()
            .expect("build cold text fixture");

        let published = layered.create_node(&["Doc"]);
        layered.set_node_property(published, "body", Value::from("needle"));
        assert_eq!(
            layered
                .text_search("Doc", "body", "needle", 1)
                .first()
                .map(|(id, _)| *id),
            Some(promoted),
            "fixture must start with the best cold hit indexed"
        );

        let layered = Arc::new(layered);
        let (hook, mut pause) = promotion_pause();
        *layered.node_publication_hook.write() = Some(hook);
        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(promoted, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let score_during_promotion = layered.score_text(promoted, "Doc", "body", "needle");
        let visible_score_during_promotion = layered.score_text_visible(
            promoted,
            "Doc",
            "body",
            "needle",
            EpochId::new(u64::MAX - 1),
            TransactionId::INVALID,
        )?;
        let top_during_promotion = layered.text_search("Doc", "body", "needle", 1);
        let threshold_during_promotion =
            layered.text_search_with_threshold("Doc", "body", "needle", 0.0);
        let visible_top_during_promotion = layered.text_search_visible(
            "Doc",
            "body",
            "needle",
            1,
            EpochId::new(u64::MAX - 1),
            TransactionId::INVALID,
        )?;
        let visible_threshold_during_promotion = layered.text_search_with_threshold_visible(
            "Doc",
            "body",
            "needle",
            0.0,
            EpochId::new(u64::MAX - 1),
            TransactionId::INVALID,
        )?;
        pause.release();
        writer.join().expect("text node promoter must not panic");

        assert!(score_during_promotion.is_some());
        assert!(visible_score_during_promotion.is_some());
        assert_eq!(
            top_during_promotion.first().map(|(id, _)| *id),
            Some(promoted),
            "representation-only hydration must keep the cold top-k hit visible"
        );
        assert!(
            threshold_during_promotion
                .iter()
                .any(|(id, _)| *id == published)
        );
        assert!(
            threshold_during_promotion
                .iter()
                .any(|(id, _)| *id == promoted)
        );
        assert_eq!(
            visible_top_during_promotion.first().map(|(id, _)| *id),
            Some(promoted)
        );
        assert!(
            visible_threshold_during_promotion
                .iter()
                .any(|(id, _)| *id == published)
        );
        assert!(
            visible_threshold_during_promotion
                .iter()
                .any(|(id, _)| *id == promoted)
        );
        assert_eq!(
            layered
                .text_search("Doc", "body", "needle", 1)
                .first()
                .map(|(id, _)| *id),
            Some(promoted),
            "the same cold hit remains visible after dirty publication"
        );
        Ok(())
    }

    /// Vector ranking also retains the unchanged cold hit during hydration.
    #[cfg(feature = "vector-index")]
    #[test]
    fn vector_search_preserves_cold_hit_during_hydration() {
        use crate::index::vector::{HnswConfig, HnswIndex, VectorIndexKind};

        let layered = empty_layered();
        let cold = layered.overlay_store();
        cold.add_vector_index(
            "Doc",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Euclidean,
            )))),
        );
        let promoted = cold.create_node(&["Doc"]);
        cold.set_node_property(
            promoted,
            "embedding",
            Value::Vector(vec![0.0_f32, 0.0].into()),
        );
        layered
            .merge_overlay_temporal()
            .expect("build cold vector fixture");

        let published = layered.create_node(&["Doc"]);
        layered.set_node_property(
            published,
            "embedding",
            Value::Vector(vec![10.0_f32, 10.0].into()),
        );
        assert_eq!(
            layered
                .vector_search(
                    Some("Doc"),
                    "embedding",
                    &[0.0, 0.0],
                    1,
                    DistanceMetric::Euclidean,
                )
                .first()
                .map(|(id, _)| *id),
            Some(promoted),
            "fixture must start with the best cold hit indexed"
        );

        let layered = Arc::new(layered);
        let (hook, mut pause) = promotion_pause();
        *layered.node_publication_hook.write() = Some(hook);
        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(promoted, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let top_during_promotion = layered.vector_search(
            Some("Doc"),
            "embedding",
            &[0.0, 0.0],
            1,
            DistanceMetric::Euclidean,
        );
        let threshold_during_promotion = layered.vector_search_with_threshold(
            Some("Doc"),
            "embedding",
            &[0.0, 0.0],
            100.0,
            DistanceMetric::Euclidean,
        );
        let visible_top_during_promotion = layered.vector_search_visible(
            "Doc",
            "embedding",
            &[0.0, 0.0],
            1,
            EpochId::from(u64::MAX),
            TransactionId::INVALID,
        );
        pause.release();
        writer.join().expect("vector node promoter must not panic");

        assert_eq!(
            top_during_promotion.first().map(|(id, _)| *id),
            Some(promoted),
            "representation-only hydration must keep the cold top-k hit visible"
        );
        assert!(
            threshold_during_promotion
                .iter()
                .any(|(id, _)| *id == published)
        );
        assert!(
            threshold_during_promotion
                .iter()
                .any(|(id, _)| *id == promoted)
        );
        assert_eq!(
            visible_top_during_promotion.first().map(|(id, _)| *id),
            Some(promoted)
        );
        assert_eq!(
            layered
                .vector_search(
                    Some("Doc"),
                    "embedding",
                    &[0.0, 0.0],
                    1,
                    DistanceMetric::Euclidean,
                )
                .first()
                .map(|(id, _)| *id),
            Some(promoted),
            "the same cold hit remains visible after dirty publication"
        );
    }

    /// A private hydration row is not an authoritative hot mutation and must
    /// not disable cold-CSR-only algorithms before publication.
    #[test]
    fn triangle_fast_path_ignores_unpublished_promotion_row() {
        let layered = empty_layered();
        let source = layered.overlay_store();
        let a = source.create_node(&["V"]);
        let b = source.create_node(&["V"]);
        let c = source.create_node(&["V"]);
        source.create_edge(a, b, "R");
        source.create_edge(b, c, "R");
        source.create_edge(c, a, "R");
        layered
            .merge_overlay_temporal()
            .expect("build cold triangle fixture");
        let expected = layered.try_count_all_directed_triangles(None);
        assert_eq!(expected, Some(3), "fixture must use the cold CSR kernel");

        let layered = Arc::new(layered);
        let (hook, mut pause) = promotion_pause();
        *layered.node_publication_hook.write() = Some(hook);
        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(a, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let count_during_promotion = layered.try_count_all_directed_triangles(None);
        pause.release();
        writer
            .join()
            .expect("triangle node promoter must not panic");

        assert_eq!(count_during_promotion, expected);
        assert_eq!(
            layered.try_count_all_directed_triangles(None),
            None,
            "published hot state must still disable the cold-only kernel"
        );
    }

    /// Serializable traversal must not visit an unpublished physical overlay
    /// duplicate: result deduplication is too late because visibility checks
    /// have already recorded the edge into the SSI read-set. Promotion retains
    /// the affected adjacency/catalog gates until routing publication, so a
    /// reader that needs that cut must serialize rather than escape a partial
    /// result.
    #[test]
    fn versioned_traversal_records_unpublished_promoted_edge_once() {
        let layered = empty_layered();
        let source = layered.overlay_store();
        let from = source.create_node(&["Source"]);
        let to = source.create_node(&["Target"]);
        let edge = source.create_edge(from, to, "CONNECTS");
        layered
            .merge_overlay_temporal()
            .expect("build cold SSI traversal fixture");

        let layered = Arc::new(layered);
        let outgoing_tx = TransactionId::new(711);
        let incoming_tx = TransactionId::new(712);
        let both_tx = TransactionId::new(713);
        let outgoing_spy = ssi_spy::register(&layered, outgoing_tx);
        let incoming_spy = ssi_spy::register(&layered, incoming_tx);
        let both_spy = ssi_spy::register(&layered, both_tx);
        let (hook, mut pause) = promotion_pause();
        *layered.edge_publication_hook.write() = Some(hook);
        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_edge_property(edge, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let (reader_started_tx, reader_started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let reader = {
            let layered = Arc::clone(&layered);
            let outgoing_spy = Arc::clone(&outgoing_spy);
            let incoming_spy = Arc::clone(&incoming_spy);
            let both_spy = Arc::clone(&both_spy);
            std::thread::spawn(move || {
                reader_started_tx
                    .send(())
                    .expect("SSI reader coordinator must remain alive");
                let outgoing = layered.edges_from_versioned(
                    from,
                    Direction::Outgoing,
                    EpochId::from(u64::MAX),
                    outgoing_tx,
                );
                let incoming = layered.edges_from_versioned(
                    to,
                    Direction::Incoming,
                    EpochId::from(u64::MAX),
                    incoming_tx,
                );
                let both = layered.edges_from_versioned(
                    from,
                    Direction::Both,
                    EpochId::from(u64::MAX),
                    both_tx,
                );
                result_tx
                    .send((
                        outgoing,
                        incoming,
                        both,
                        outgoing_spy.edge_read_count(edge),
                        incoming_spy.edge_read_count(edge),
                        both_spy.edge_read_count(edge),
                    ))
                    .expect("SSI reader result must be received");
            })
        };
        reader_started_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("SSI reader must start");
        assert!(
            matches!(
                result_rx.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "an SSI traversal must not complete through a private promotion cut"
        );
        pause.release();
        writer.join().expect("SSI edge promoter must not panic");
        let (outgoing, incoming, both, outgoing_reads, incoming_reads, both_reads) = result_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("SSI reader must complete after routing publication");
        reader.join().expect("SSI traversal reader must not panic");

        assert_eq!(outgoing, vec![(to, edge)]);
        assert_eq!(incoming, vec![(from, edge)]);
        assert_eq!(both, vec![(to, edge)]);
        assert_eq!(outgoing_reads, 1);
        assert_eq!(incoming_reads, 1);
        assert_eq!(
            both_reads, 1,
            "one logical base edge must produce exactly one Serializable read per traversal"
        );
    }

    /// A compacted edge whose lifetime closes after a reader's snapshot still
    /// belongs to that snapshot. Direct reads and every traversal direction
    /// must use the temporal cold relation and record one logical SSI read.
    #[test]
    fn versioned_reads_include_compacted_edge_closed_after_snapshot() {
        let layered = empty_layered();
        let source = layered.overlay_store();
        let created = EpochId::new(10);
        let deleted = EpochId::new(20);
        source.set_epoch(created);
        let from = source.create_node(&["Source"]);
        let to = source.create_node(&["Target"]);
        let edge =
            source.create_edge_versioned(from, to, "CONNECTS", created, TransactionId::SYSTEM);
        source.set_edge_property_at_epoch(edge, "weight", Value::Int64(7), created);
        source.set_epoch(deleted);
        assert!(source.delete_edge_at_epoch(edge, deleted));
        layered
            .merge_overlay_temporal()
            .expect("compact the closed edge fixture");

        let snapshot = EpochId::new(15);
        let base = layered.base_store_arc();
        assert!(
            base.get_edge_versioned(edge, snapshot, TransactionId::SYSTEM)
                .is_some(),
            "CompactStore versioned fetch must honor the requested epoch"
        );

        let layered = Arc::new(layered);
        assert!(
            layered
                .get_edge_versioned(edge, snapshot, TransactionId::new(720))
                .is_some()
        );
        assert!(layered.is_edge_visible_versioned(edge, snapshot, TransactionId::new(721)));
        assert_eq!(
            layered.read_edge_property_visible(
                edge,
                &PropertyKey::new("weight"),
                snapshot,
                Some(TransactionId::new(722)),
            ),
            Some(Value::Int64(7))
        );

        let outgoing_tx = TransactionId::new(723);
        let incoming_tx = TransactionId::new(724);
        let both_tx = TransactionId::new(725);
        let outgoing_spy = ssi_spy::register(&layered, outgoing_tx);
        let incoming_spy = ssi_spy::register(&layered, incoming_tx);
        let both_spy = ssi_spy::register(&layered, both_tx);
        assert_eq!(
            layered.edges_from_versioned(from, Direction::Outgoing, snapshot, outgoing_tx),
            vec![(to, edge)]
        );
        assert_eq!(
            layered.edges_from_versioned(to, Direction::Incoming, snapshot, incoming_tx),
            vec![(from, edge)]
        );
        assert_eq!(
            layered.edges_from_versioned(from, Direction::Both, snapshot, both_tx),
            vec![(to, edge)]
        );
        assert_eq!(outgoing_spy.edge_read_count(edge), 1);
        assert_eq!(incoming_spy.edge_read_count(edge), 1);
        assert_eq!(both_spy.edge_read_count(edge), 1);

        assert!(
            layered
                .get_edge_versioned(edge, deleted, TransactionId::new(726))
                .is_none()
        );
        assert!(
            layered
                .edges_from_versioned(from, Direction::Outgoing, deleted, TransactionId::new(727))
                .is_empty()
        );
    }

    /// Planner pruning, schema enumeration, and as-of scrubs must stay on the
    /// cold logical state while a historical value is only partially replayed.
    #[test]
    fn planner_schema_and_scrub_hide_partial_promotion_replay() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let layered = empty_layered();
        let source = layered.overlay_store();
        source.set_epoch(EpochId::new(10));
        let node = source.create_node(&["Metric"]);
        source.set_node_property_at_epoch(node, "score", Value::Int64(1), EpochId::new(10));
        source.set_node_property_at_epoch(node, "score", Value::Int64(100), EpochId::new(20));
        source.set_epoch(EpochId::new(20));
        layered
            .merge_overlay_temporal()
            .expect("build planner replay fixture");
        let key = PropertyKey::new("score");
        assert!(
            !layered.node_property_might_match(&key, CompareOp::Lt, &Value::Int64(10)),
            "cold current value 100 must be prunable for score < 10"
        );
        let mut expected_keys = layered.all_property_keys();
        expected_keys.sort_unstable();

        let layered = Arc::new(layered);
        let (pause_hook, mut pause) = promotion_pause();
        let calls = Arc::new(AtomicUsize::new(0));
        *layered.node_replay_event_hook.write() = Some(Arc::new({
            let calls = Arc::clone(&calls);
            move || {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    pause_hook();
                }
            }
        }));
        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_node_property(node, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let might_match_during_replay =
            layered.node_property_might_match(&key, CompareOp::Lt, &Value::Int64(10));
        let mut keys_during_replay = layered.all_property_keys();
        keys_during_replay.sort_unstable();
        let scrub_value_during_replay = layered
            .scrub_at_epoch(EpochId::new(20))
            .into_iter()
            .find_map(|frame| {
                frame
                    .node_ids
                    .iter()
                    .position(|id| *id == node)
                    .and_then(|offset| {
                        frame
                            .columns
                            .get(&key)
                            .and_then(|column| column[offset].clone())
                    })
            });
        pause.release();
        writer.join().expect("planner node promoter must not panic");

        assert!(!might_match_during_replay);
        assert_eq!(keys_during_replay, expected_keys);
        assert_eq!(scrub_value_during_replay, Some(Value::Int64(100)));
    }

    /// Degree estimates are computed from the logical merged relation, not a
    /// weighted sum of duplicate physical tier rows. A reader that needs an
    /// affected LPG catalog/adjacency cut serializes until routing publication;
    /// it never returns a half-hydrated physical duplicate.
    #[test]
    fn average_degree_hides_unpublished_edge_promotion() {
        let layered = empty_layered();
        let source = layered.overlay_store();
        let from = source.create_node(&["Source"]);
        source.create_node(&["Source"]);
        source.create_node(&["Source"]);
        let to = source.create_node(&["Target"]);
        let edge = source.create_edge(from, to, "CONNECTS");
        layered
            .merge_overlay_temporal()
            .expect("build degree-estimate fixture");
        let expected_out = layered.estimate_avg_degree("CONNECTS", true);
        let expected_in = layered.estimate_avg_degree("CONNECTS", false);
        assert!((expected_out - (1.0 / 3.0)).abs() < f64::EPSILON);
        assert!((expected_in - 1.0).abs() < f64::EPSILON);

        let layered = Arc::new(layered);
        let (hook, mut pause) = promotion_pause();
        *layered.edge_publication_hook.write() = Some(hook);
        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_edge_property(edge, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let (reader_started_tx, reader_started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let reader = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                reader_started_tx
                    .send(())
                    .expect("degree reader coordinator must remain alive");
                result_tx
                    .send((
                        layered.estimate_avg_degree("CONNECTS", true),
                        layered.estimate_avg_degree("CONNECTS", false),
                    ))
                    .expect("degree reader result must be received");
            })
        };
        reader_started_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("degree reader must start");
        assert!(
            matches!(
                result_rx.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "degree estimation must not complete through a private promotion cut"
        );
        pause.release();
        writer.join().expect("degree edge promoter must not panic");
        let (out_during_promotion, in_during_promotion) = result_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("degree reader must complete after routing publication");
        reader.join().expect("degree reader must not panic");

        assert!((out_during_promotion - expected_out).abs() < f64::EPSILON);
        assert!((in_during_promotion - expected_in).abs() < f64::EPSILON);
        assert!(
            (layered.estimate_avg_degree("CONNECTS", true) - expected_out).abs() < f64::EPSILON
        );
        assert!(
            (layered.estimate_avg_degree("CONNECTS", false) - expected_in).abs() < f64::EPSILON
        );
    }

    /// Schema keys remain known after the last owning overlay entity is
    /// deleted; filtering private promotion rows must not weaken that contract.
    #[test]
    fn property_keys_retain_deleted_overlay_schema() {
        let layered = empty_layered();
        let node = layered.create_node(&["Ephemeral"]);
        layered.set_node_property(node, "retained_schema", Value::Bool(true));
        assert!(
            layered
                .all_property_keys()
                .contains(&"retained_schema".to_owned())
        );
        assert!(layered.delete_node(node));
        assert!(
            layered
                .all_property_keys()
                .contains(&"retained_schema".to_owned()),
            "deleting the last entity must not forget its registered property key"
        );
    }

    /// Edge hydration installs the overlay edge and its adjacency entries
    /// before publishing the dirty marker. The structural count must keep that
    /// private copy invisible until publication.
    #[test]
    fn edge_count_hides_unpublished_promotion_row() {
        let layered = empty_layered();
        let source = layered.overlay_store().create_node(&["Source"]);
        let target = layered.overlay_store().create_node(&["Target"]);
        let edge = layered
            .overlay_store()
            .create_edge(source, target, "CONNECTS");
        layered
            .merge_overlay_temporal()
            .expect("build cold edge fixture");

        let layered = Arc::new(layered);
        let (hook, mut pause) = promotion_pause();
        *layered.edge_publication_hook.write() = Some(hook);

        let writer = {
            let layered = Arc::clone(&layered);
            std::thread::spawn(move || {
                layered.set_edge_property(edge, "writer", Value::Bool(true));
            })
        };

        pause.wait_until_reached();
        let count_before_publication = layered.edge_count();
        pause.release();
        writer.join().expect("edge promoter must not panic");

        assert_eq!(
            count_before_publication, 1,
            "a hydrated but unpublished base-overlap edge must not be counted twice"
        );
        assert_eq!(layered.edge_count(), 1);
        assert_eq!(
            layered.get_edge_property(edge, &PropertyKey::new("writer")),
            Some(Value::Bool(true))
        );
    }
}
