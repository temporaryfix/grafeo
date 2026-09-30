//! The in-memory LPG graph store.
//!
//! This is where your nodes and edges actually live. Most users interact
//! through [`GrafeoDB`](grafeo_engine::GrafeoDB), but algorithm implementers
//! sometimes need the raw [`LpgStore`] for direct adjacency traversal.
//!
//! Key features:
//! - MVCC versioning - concurrent readers don't block each other
//! - Columnar properties with zone maps for fast filtering
//! - Forward and backward adjacency indexes

mod capture;
mod data_labels;
pub use data_labels::PreparedNodeLabelImages;
mod data_publication;
pub(crate) use data_publication::DataCommitScope;
mod edge_ops;
#[cfg(any(test, feature = "compact-store"))]
pub(crate) use edge_ops::PreparedPurgeOutcome;
mod graph_identity;
mod graph_store_impl;
mod index;
mod index_batch;
pub use index_batch::{
    IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey, IndexRegistryMaintenance,
    IndexRegistryWorkspace, InstalledIndexRegistryFence, InstalledLpgCommit, LpgCommitWorkspace,
    PreparedIndexRegistryBatch, PreparedLpgCommit, ReleasedLpgCommit, StoreCommitInput,
    StoreIndexEdits, prepare_index_registry_batch, with_prepared_lpg_commit,
    with_prepared_lpg_replacement,
};
#[cfg(feature = "vector-index")]
pub use index_batch::{VectorCommitChanges, VectorCommitInput};
mod index_registration;
pub use index_registration::IndexRegistrationObservation;
use index_registration::{PhysicalStoreIdentity, PropertyIndexRows, RegisteredIndex};
mod memory;
mod node_ops;
mod property_index;
mod property_ops;
pub use property_index::PropertyIndexImage;
mod recovery_labels;
mod restore;
pub use restore::LpgReplacementWorkspace;
mod schema;
mod search;
mod statistics;
mod traversal;
#[cfg(feature = "vector-index")]
pub(crate) mod vector_accessor;
mod versioning;

#[cfg(all(feature = "vector-index", feature = "compact-store"))]
pub(crate) use index::VisibleVectorReadContext;
#[cfg(feature = "compact-store")]
pub(crate) use index::{PreparedRepresentationTransfer, PublishedRepresentationTransfer};

#[cfg(test)]
mod tests;

use super::PropertyStorage;
#[cfg(not(feature = "tiered-storage"))]
use super::{EdgeRecord, NodeRecord};
use crate::execution::operators::{SharedReadTracker, SharedWriteTracker};
use crate::graph::lpg::{Edge, Node};
use crate::index::adjacency::ChunkedAdjacency;
use crate::statistics::Statistics;
use arcstr::ArcStr;
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
use grafeo_common::types::{
    EdgeId, EdgeTypeId, EpochId, LabelId, NodeId, PropertyKey, TransactionId, Value,
};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::{Condvar, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cell::RefCell;
use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

#[cfg(feature = "text-index")]
use crate::index::text::{InvertedIndex, RegisteredTextIndex};
#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;

#[cfg(feature = "tiered-storage")]
use crate::codec::EpochStore;
use grafeo_common::memory::arena::AllocError;
#[cfg(feature = "tiered-storage")]
use grafeo_common::memory::arena::ArenaAllocator;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::VersionIndex;
use grafeo_common::temporal::VersionLog;

thread_local! {
    /// Exact exclusive transitions held by this thread; reads grant no mutation.
    static EXCLUSIVE_BULK_RESTORE_STACK: RefCell<Vec<(usize, ExclusiveMode)>> = const { RefCell::new(Vec::new()) };
    // Read-only reentrancy for callers already holding this thread's exact
    // exclusive representation barrier. This never authorizes a mutation.
    static EXCLUSIVE_READ_SCOPES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

struct ExclusiveReadScope {
    identity: usize,
}

impl ExclusiveReadScope {
    fn enter(store: &LpgStore) -> Self {
        let identity = std::ptr::from_ref(store).addr();
        EXCLUSIVE_READ_SCOPES.with_borrow_mut(|scopes| scopes.push(identity));
        Self { identity }
    }

    fn is_held(store: &LpgStore) -> bool {
        let identity = std::ptr::from_ref(store).addr();
        EXCLUSIVE_READ_SCOPES.with_borrow(|scopes| scopes.contains(&identity))
    }
}

impl Drop for ExclusiveReadScope {
    fn drop(&mut self) {
        EXCLUSIVE_READ_SCOPES.with_borrow_mut(|scopes| {
            if let Some(position) = scopes
                .iter()
                .rposition(|identity| *identity == self.identity)
            {
                scopes.swap_remove(position);
            }
        });
    }
}

struct PinnedExclusiveScope<'a> {
    _reads: ExclusiveReadScope,
    _guard: RwLockWriteGuard<'a, ()>,
}

/// Serializes LPG named-graph topology changes and recursive sealing across
/// stores. Named-graph DDL is cold-path work; one process-wide gate makes
/// cycle rejection and seal preflight deterministic without penalizing entity
/// mutation concurrency.
static NAMED_GRAPH_TOPOLOGY_GATE: Mutex<()> = Mutex::new(());

/// Checked, process-local identities used to bind a retained derived-index Arc
/// to one logical store and one exact encoded label/property slot. Zero is
/// reserved for an unbound index; exhaustion fails closed without reuse.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
static NEXT_INDEX_OWNER_ID: AtomicU64 = AtomicU64::new(1);
#[cfg(any(feature = "vector-index", feature = "text-index"))]
static NEXT_INDEX_SLOT_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn allocate_index_binding_id(source: &AtomicU64) -> Result<u64, AllocError> {
    source
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(1)
        })
        .map_err(|_| AllocError::InsufficientSpace)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExclusiveMode {
    ReadOnly,
    Restore,
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    RecordedIndexRecovery {
        #[cfg(feature = "text-index")]
        text: bool,
        #[cfg(feature = "vector-index")]
        vector: bool,
    },
}

/// Unwind-safe carrier for an already-held exclusive restore proof.
struct ExclusiveBulkRestoreContext {
    previous_depth: usize,
}

impl ExclusiveBulkRestoreContext {
    fn has_read_capture() -> bool {
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow(|stack| {
            stack
                .iter()
                .any(|(_, mode)| *mode == ExclusiveMode::ReadOnly)
        })
    }

    fn enter(store: &LpgStore) -> Self {
        Self::enter_mode(store, ExclusiveMode::Restore)
    }

    fn enter_mode(store: &LpgStore, mode: ExclusiveMode) -> Self {
        let store = std::ptr::from_ref(store).addr();
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow_mut(|stack| {
            let previous_depth = stack.len();
            stack.push((store, mode));
            Self { previous_depth }
        })
    }

    #[cfg(feature = "text-index")]
    fn records_text(store: &LpgStore) -> bool {
        let store = std::ptr::from_ref(store).addr();
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow(|stack| {
            stack.iter().any(|(known, mode)| {
                *known == store
                    && matches!(
                        mode,
                        ExclusiveMode::RecordedIndexRecovery { text: true, .. }
                    )
            })
        })
    }

    #[cfg(feature = "vector-index")]
    fn records_vector(store: &LpgStore) -> bool {
        let store = std::ptr::from_ref(store).addr();
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow(|stack| {
            stack.iter().any(|(known, mode)| {
                *known == store
                    && matches!(
                        mode,
                        ExclusiveMode::RecordedIndexRecovery { vector: true, .. }
                    )
            })
        })
    }

    fn is_active(store: &LpgStore) -> bool {
        let store = std::ptr::from_ref(store).addr();
        EXCLUSIVE_BULK_RESTORE_STACK
            .with_borrow(|stack| stack.iter().any(|(known, _)| *known == store))
    }

    fn is_read_only(store: &LpgStore) -> bool {
        let store = std::ptr::from_ref(store).addr();
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow(|stack| {
            stack
                .iter()
                .any(|(known, mode)| *known == store && *mode == ExclusiveMode::ReadOnly)
        })
    }
}

impl Drop for ExclusiveBulkRestoreContext {
    fn drop(&mut self) {
        EXCLUSIVE_BULK_RESTORE_STACK.with_borrow_mut(|stack| stack.truncate(self.previous_depth));
    }
}

/// Store-local authority embedded in an extract transport receipt.
///
/// The type and its fields are crate-private. Receipt validation uses
/// `Arc::ptr_eq`, so neither an edge id nor copied endpoint values can confer
/// authority over an ordinary or foreign-store identity.
#[derive(Debug)]
pub(crate) struct TransportExtractAuthority {
    next_nonce: AtomicU64,
    edge_nonces: RwLock<FxHashMap<EdgeId, u64>>,
}

impl TransportExtractAuthority {
    fn new() -> Self {
        Self {
            next_nonce: AtomicU64::new(1),
            edge_nonces: RwLock::new(FxHashMap::default()),
        }
    }

    fn reserve(&self, id: EdgeId) -> Option<u64> {
        let mut nonces = self.edge_nonces.write();
        if nonces.contains_key(&id) {
            return None;
        }
        let nonce = self
            .next_nonce
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if current == 0 {
                    None
                } else {
                    current.checked_add(1)
                }
            })
            .ok()?;
        nonces.insert(id, nonce);
        Some(nonce)
    }

    fn contains(&self, id: EdgeId, nonce: u64) -> bool {
        nonce != 0 && self.edge_nonces.read().get(&id) == Some(&nonce)
    }

    fn revoke(&self, id: EdgeId, nonce: u64) {
        let mut nonces = self.edge_nonces.write();
        if nonces.get(&id) == Some(&nonce) {
            nonces.remove(&id);
        }
    }

    /// Revokes any provenance attached to a structurally removed or ordinarily
    /// republished identity. Only the receipt-bearing transport creator may
    /// install a fresh nonce for a vacant reservation.
    fn revoke_id(&self, id: EdgeId) {
        self.edge_nonces.write().remove(&id);
    }

    /// Validates a complete receipt batch under one nonce-map cut.
    fn contains_all<'a>(
        authority: &Arc<Self>,
        receipts: impl IntoIterator<Item = &'a TransportEdgeReceipt>,
    ) -> bool {
        let nonces = authority.edge_nonces.read();
        receipts.into_iter().all(|receipt| {
            receipt.nonce != 0
                && Arc::ptr_eq(&receipt.authority, authority)
                && nonces.get(&receipt.id) == Some(&receipt.nonce)
        })
    }

    /// Revalidates and revokes a duplicate-free receipt batch in one
    /// allocation-free nonce-map write cut. A failed commit-time recheck leaves
    /// every nonce unchanged.
    fn revoke_all_if_present<'a, I>(authority: &Arc<Self>, receipts: I) -> bool
    where
        I: IntoIterator<Item = &'a TransportEdgeReceipt>,
        I::IntoIter: Clone,
    {
        let receipts = receipts.into_iter();
        let mut nonces = authority.edge_nonces.write();
        if receipts.clone().any(|receipt| {
            receipt.nonce == 0
                || !Arc::ptr_eq(&receipt.authority, authority)
                || nonces.get(&receipt.id) != Some(&receipt.nonce)
        }) {
            return false;
        }
        for receipt in receipts {
            nonces.remove(&receipt.id);
        }
        true
    }
}

/// Panic-safe ownership of a newly reserved transport nonce.
///
/// Exact structural creation can allocate and can run test publication hooks.
/// Until the move-only public receipt has been fully constructed, unwinding
/// must revoke the nonce rather than leave unreachable provenance installed in
/// the store incarnation.
struct TransportNonceReservation {
    authority: Arc<TransportExtractAuthority>,
    id: EdgeId,
    nonce: u64,
    armed: bool,
}

/// Allocation-complete reservation for one edge-type catalog identity.
///
/// The lightweight publication mutex serializes candidate IDs without
/// retaining catalog guards across arena allocation. New map/vector capacity
/// is reserved before this value is returned; [`PreparedEdgeType::commit`]
/// therefore installs the row without further allocation.
struct PreparedEdgeType<'a> {
    _publication: parking_lot::MutexGuard<'a, ()>,
    name: ArcStr,
    id: u32,
    is_new: bool,
}

/// Reversible publication of one edge-type catalog suffix for an unpublished
/// compact-base edge promotion.
///
/// Property replay needs the type row to be read-visible, so the publication
/// cannot simply be deferred. The retained gate prevents another type creator
/// from consuming the suffix. Drop removes a newly appended zero-count row
/// without allocation; `commit` makes it permanent.
#[cfg(feature = "compact-store")]
pub(super) struct PreparedPromotionEdgeType<'a> {
    store: &'a LpgStore,
    _publication: parking_lot::MutexGuard<'a, ()>,
    name: ArcStr,
    id: u32,
    is_new: bool,
    armed: bool,
}

/// Allocation-complete reservation for a duplicate-tolerant edge-type batch.
struct PreparedEdgeTypes<'a> {
    _publication: parking_lot::MutexGuard<'a, ()>,
    ids: FxHashMap<ArcStr, u32>,
    new_types: Vec<(ArcStr, u32)>,
}

/// Temporarily published label-catalog suffix for one unpublished compact-base
/// node promotion.
///
/// Label names must be visible while property/vector/text replay routes the
/// promoted identity through derived indexes. The retained publication guard
/// prevents another label creator from extending the suffix. Dropping an
/// uncommitted preparation removes exactly that suffix and restores the prior
/// label-index length without allocation; `commit` makes it permanent.
pub(super) struct PreparedNodeLabels<'a> {
    store: &'a LpgStore,
    _publication: parking_lot::MutexGuard<'a, ()>,
    ids: FxHashMap<ArcStr, u32>,
    original_registry_len: usize,
    original_label_index_len: usize,
    armed: bool,
}

/// Allocation-complete reservation for one exact vector/text index slot.
///
/// The permanent registry survives DDL removal, preventing a retained Arc
/// from being rebound under a different encoded label/property key.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
pub(super) struct PreparedIndexSlot<'a> {
    slots: parking_lot::MutexGuard<'a, FxHashMap<String, u64>>,
    key: String,
    id: u64,
    is_new: bool,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl PreparedIndexSlot<'_> {
    pub(super) const fn id(&self) -> u64 {
        self.id
    }

    pub(super) fn commit(mut self) {
        if self.is_new {
            let previous = self.slots.insert(self.key, self.id);
            debug_assert!(previous.is_none(), "prepared index slot is still vacant");
        }
    }
}

impl PreparedEdgeType<'_> {
    const fn id(&self) -> u32 {
        self.id
    }

    fn commit(self, store: &LpgStore) -> u32 {
        if self.is_new {
            let mut type_to_id = store.edge_type_to_id.write();
            let mut id_to_type = store.id_to_edge_type.write();
            let mut counts = store.edge_type_live_counts.write();
            debug_assert_eq!(id_to_type.len(), self.id as usize);
            debug_assert!(!type_to_id.contains_key(&self.name));
            type_to_id.insert(self.name.clone(), self.id);
            id_to_type.push(self.name);
            counts.push(0);
        }
        self.id
    }
}

#[cfg(feature = "compact-store")]
impl PreparedPromotionEdgeType<'_> {
    pub(super) const fn id(&self) -> u32 {
        self.id
    }

    pub(super) fn commit(mut self) {
        self.armed = false;
    }
}

#[cfg(feature = "compact-store")]
impl Drop for PreparedPromotionEdgeType<'_> {
    fn drop(&mut self) {
        if !self.armed || !self.is_new {
            return;
        }
        let Ok(index) = usize::try_from(self.id) else {
            return;
        };
        let mut type_to_id = self.store.edge_type_to_id.write();
        let mut id_to_type = self.store.id_to_edge_type.write();
        let mut counts = self.store.edge_type_live_counts.write();
        let owns_exact_suffix = id_to_type.len() == index.saturating_add(1)
            && id_to_type.last() == Some(&self.name)
            && type_to_id.get(&self.name) == Some(&self.id)
            && counts.len() == id_to_type.len()
            && counts.get(index) == Some(&0);
        if owns_exact_suffix {
            type_to_id.remove(&self.name);
            id_to_type.pop();
            counts.pop();
        }
    }
}

impl PreparedEdgeTypes<'_> {
    fn id(&self, name: &str) -> u32 {
        self.ids[name]
    }

    fn commit(self, store: &LpgStore) {
        if self.new_types.is_empty() {
            return;
        }
        let mut type_to_id = store.edge_type_to_id.write();
        let mut id_to_type = store.id_to_edge_type.write();
        let mut counts = store.edge_type_live_counts.write();
        for (name, id) in self.new_types {
            debug_assert_eq!(id_to_type.len(), id as usize);
            debug_assert!(!type_to_id.contains_key(&name));
            type_to_id.insert(name.clone(), id);
            id_to_type.push(name);
            counts.push(0);
        }
    }
}

impl PreparedNodeLabels<'_> {
    pub(super) fn id(&self, label: &str) -> Option<u32> {
        self.ids.get(label).copied()
    }

    pub(super) fn commit(mut self) {
        self.armed = false;
    }
}

impl Drop for PreparedNodeLabels<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut registry = self.store.label_registry.write();
        registry.truncate(self.original_registry_len);
        drop(registry);
        self.store
            .label_index
            .write()
            .truncate(self.original_label_index_len);
    }
}

impl TransportNonceReservation {
    fn new(authority: &Arc<TransportExtractAuthority>, id: EdgeId) -> Option<Self> {
        let nonce = authority.reserve(id)?;
        Some(Self {
            authority: Arc::clone(authority),
            id,
            nonce,
            armed: true,
        })
    }

    fn commit(mut self) -> u64 {
        self.armed = false;
        self.nonce
    }
}

impl Drop for TransportNonceReservation {
    fn drop(&mut self) {
        if self.armed {
            self.authority.revoke(self.id, self.nonce);
        }
    }
}

/// Store-local edge identities reserved by an in-flight publication or purge.
///
/// Reservations live outside the version maps, so readers, GC, and snapshot
/// enumeration never observe an empty structural sentinel. `clear` holds the
/// registry at an empty boundary, while GC/discard retain every reserved ID.
struct EdgeIdentityReservationRegistry {
    identities: Mutex<FxHashSet<EdgeId>>,
    drained: Condvar,
}

/// Out-of-map ownership for in-flight generated, exact, and representation-
/// promotion node publication/removal.
struct NodeIdentityReservationRegistry {
    identities: Mutex<FxHashSet<NodeId>>,
    drained: Condvar,
}

impl NodeIdentityReservationRegistry {
    fn new() -> Self {
        Self {
            identities: Mutex::new(FxHashSet::default()),
            drained: Condvar::new(),
        }
    }

    fn lock_when_empty(&self) -> parking_lot::MutexGuard<'_, FxHashSet<NodeId>> {
        let mut identities = self.identities.lock();
        while !identities.is_empty() {
            self.drained.wait(&mut identities);
        }
        identities
    }

    fn release(&self, id: NodeId) {
        let mut identities = self.identities.lock();
        identities.remove(&id);
        if identities.is_empty() {
            self.drained.notify_all();
        }
    }
}

/// Panic-safe ownership of one node identity outside the structural map.
struct NodeIdentityReservation<'a> {
    registry: &'a NodeIdentityReservationRegistry,
    id: NodeId,
    armed: bool,
}

impl<'a> NodeIdentityReservation<'a> {
    fn vacant(store: &'a LpgStore, id: NodeId) -> Option<Self> {
        if !id.is_valid() {
            return None;
        }
        let mut identities = store.node_identity_reservations.identities.lock();
        if identities.contains(&id) {
            return None;
        }
        #[cfg(not(feature = "tiered-storage"))]
        let occupied = store.nodes.read().contains_key(&id);
        #[cfg(feature = "tiered-storage")]
        let occupied = store.node_versions.read().contains_key(&id);
        if occupied {
            return None;
        }
        identities.insert(id);
        drop(identities);
        Some(Self {
            registry: &store.node_identity_reservations,
            id,
            armed: true,
        })
    }

    fn generated(store: &'a LpgStore) -> Option<Self> {
        let mut identities = store.node_identity_reservations.identities.lock();
        #[cfg(not(feature = "tiered-storage"))]
        let published = store.nodes.read();
        #[cfg(feature = "tiered-storage")]
        let published = store.node_versions.read();
        loop {
            let raw = store
                .next_node_id
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current.checked_add(1)
                })
                .ok()?;
            let id = NodeId::new(raw);
            if id.is_valid() && !identities.contains(&id) && !published.contains_key(&id) {
                identities.insert(id);
                drop(published);
                drop(identities);
                return Some(Self {
                    registry: &store.node_identity_reservations,
                    id,
                    armed: true,
                });
            }
        }
    }

    fn resident(store: &'a LpgStore, id: NodeId) -> Option<Self> {
        if !id.is_valid() {
            return None;
        }
        let mut identities = store.node_identity_reservations.identities.lock();
        if identities.contains(&id) {
            return None;
        }
        #[cfg(not(feature = "tiered-storage"))]
        let occupied = store.nodes.read().contains_key(&id);
        #[cfg(feature = "tiered-storage")]
        let occupied = store.node_versions.read().contains_key(&id);
        if !occupied {
            return None;
        }
        identities.insert(id);
        drop(identities);
        Some(Self {
            registry: &store.node_identity_reservations,
            id,
            armed: true,
        })
    }

    const fn id(&self) -> NodeId {
        self.id
    }

    fn commit(mut self) {
        self.registry.release(self.id);
        self.armed = false;
    }
}

impl Drop for NodeIdentityReservation<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.registry.release(self.id);
        }
    }
}

impl EdgeIdentityReservationRegistry {
    fn new() -> Self {
        Self {
            identities: Mutex::new(FxHashSet::default()),
            drained: Condvar::new(),
        }
    }

    fn lock_when_empty(&self) -> parking_lot::MutexGuard<'_, FxHashSet<EdgeId>> {
        let mut identities = self.identities.lock();
        while !identities.is_empty() {
            self.drained.wait(&mut identities);
        }
        identities
    }

    fn release(&self, ids: &[EdgeId]) {
        let mut identities = self.identities.lock();
        for id in ids {
            identities.remove(id);
        }
        if identities.is_empty() {
            self.drained.notify_all();
        }
    }
}

/// Panic-safe ownership of one or more exact structural edge reservations.
///
/// All edge creators reserve before catalog work, arena allocation, or map
/// publication. Physical transport purge reserves the still-present identity
/// through structural, adjacency, and property removal. Dropping this guard on
/// any error or unwind releases only the out-of-map reservation.
struct EdgeIdentityReservation<'a> {
    registry: &'a EdgeIdentityReservationRegistry,
    ids: Vec<EdgeId>,
    armed: bool,
}

/// Shared proof that one ordinary mutation was authorized at a stable store
/// scope and cannot overlap an exclusive representation transition.
///
/// `read_recursive` is required because public composite mutators call other
/// mutators. A queued writer must not make a same-thread nested read deadlock.
/// The outermost public operation retains its proof for the whole composite;
/// inner helpers may either accept that proof explicitly or take another
/// recursive shared proof. This proof is never upgraded to exclusive.
pub(crate) struct PinnedMutation<'a> {
    _scope_transition: Option<RwLockReadGuard<'a, ()>>,
}

/// Shared barrier-only proof for non-authoritative derived-cache maintenance.
///
/// Statistics and zone-map refresh remain available to sealed read/query paths,
/// but cannot overlap clear or whole-generation replacement. This proof confers
/// no permission to change authoritative graph state.
pub(crate) struct PinnedMaintenance<'a> {
    _scope_transition: Option<RwLockReadGuard<'a, ()>>,
}

/// Exclusive barrier-only proof for a derived-cache rebuild that rewrites
/// counters computed from the complete authoritative store.
///
/// Unlike [`PinnedLpgTransition`], this proof does not inspect or confer write
/// authority and does not pin transport provenance. It only excludes ordinary
/// mutations and destructive representation transitions while a sealed query
/// path publishes a self-consistent derived snapshot. Callers must not acquire
/// it while retaining [`PinnedMaintenance`] or [`PinnedMutation`].
pub(crate) struct PinnedExclusiveMaintenance<'a> {
    _scope_transition: Option<PinnedExclusiveScope<'a>>,
}

/// Exclusive proof that one LPG store's mutation scope and representation are
/// pinned for a whole-generation transition.
///
/// The transport publication seams pass this proof to their preparation
/// closure and retain it through external publication and physical removal.
/// Ordinary mutators take the shared side of the same transition barrier, so
/// they cannot overlap this proof.
/// It is intentionally crate-private and cannot be forged by downstream code.
pub(crate) struct PinnedLpgTransition<'a> {
    store: &'a LpgStore,
    _scope_transition: PinnedExclusiveScope<'a>,
    transport_authority: RwLockReadGuard<'a, Arc<TransportExtractAuthority>>,
}

/// Allocation-free rejection while final data writers are held. Materialize a
/// public error only after the aggregate acquisition scope releases its guards.
#[derive(Debug)]
pub enum DataRebindError {
    /// The captured input or prequalified state is invalid.
    Invalid(&'static str),
    /// Another reader/writer or changed identity prevents final binding.
    Conflict(&'static str),
    /// Candidate preparation could not reserve required memory.
    Allocation(grafeo_common::memory::AllocError),
}

impl DataRebindError {
    pub(crate) fn new(reason: &'static str) -> Self {
        Self::Invalid(reason)
    }

    /// Materializes an owned public error. Call only after all final companion
    /// guards have drained: formatting may allocate.
    #[must_use]
    pub fn into_error(self) -> grafeo_common::utils::error::Error {
        match self {
            Self::Invalid(reason) => grafeo_common::utils::error::TransactionError::InvalidState(
                format!("prepared data rebind: {reason}"),
            )
            .into(),
            Self::Conflict(reason) => {
                grafeo_common::utils::error::TransactionError::WriteConflict(reason.to_owned())
                    .into()
            }
            Self::Allocation(error) => error.into(),
        }
    }
}

impl From<grafeo_common::memory::AllocError> for DataRebindError {
    fn from(error: grafeo_common::memory::AllocError) -> Self {
        Self::Allocation(error)
    }
}

/// Proof that the process-wide named-graph topology is stable while a
/// same-incarnation representation successor is prepared and published.
///
/// Named-graph DDL takes this gate before a store mutation pin. Compact
/// replacement follows the same order, then takes the source's exclusive LPG
/// transition, preventing both topology races and lock inversion.
#[cfg(feature = "compact-store")]
pub(crate) struct PinnedNamedGraphTopology {
    _guard: parking_lot::MutexGuard<'static, ()>,
}

impl PinnedLpgTransition<'_> {
    /// Qualifies a cooperating representation against this exact mutation cut.
    pub(crate) fn pins_store(&self, store: &LpgStore) -> bool {
        std::ptr::eq(self.store, store)
    }

    fn transport_authority(&self) -> &Arc<TransportExtractAuthority> {
        &self.transport_authority
    }

    /// Qualifies releasing and later rebinding this store's property writer
    /// while the same exclusive transition remains continuously borrowed.
    /// A layout-compatible storage or another store is not this target.
    pub(crate) fn pins_property_storage<Id: super::property::EntityId>(
        &self,
        storage: &PropertyStorage<Id>,
    ) -> bool {
        let storage: &dyn std::any::Any = storage;
        if let Some(storage) = storage.downcast_ref::<PropertyStorage<NodeId>>() {
            std::ptr::eq(storage, &raw const self.store.node_properties)
        } else if let Some(storage) = storage.downcast_ref::<PropertyStorage<EdgeId>>() {
            std::ptr::eq(storage, &raw const self.store.edge_properties)
        } else {
            false
        }
    }

    /// Qualifies only this exact store's forward or backward adjacency.
    pub(crate) fn pins_adjacency(&self, adjacency: &ChunkedAdjacency) -> bool {
        std::ptr::eq(adjacency, &raw const self.store.forward_adj)
            || self
                .store
                .backward_adj
                .as_ref()
                .is_some_and(|backward| std::ptr::eq(adjacency, backward))
    }

    #[cfg(test)]
    pub(crate) fn adjacency_for_test(&self) -> &ChunkedAdjacency {
        &self.store.forward_adj
    }

    /// Builds an empty representation successor under this exact scope cut.
    ///
    /// The successor preserves the source representation configuration,
    /// current epoch, allocator high-water marks, and exact mutation seal.
    /// `preserve_transport_authority` may be `true` only when replacing the
    /// representation of the same logical store: it shares the current nonce
    /// authority so outstanding receipts keep their meaning. Passing `false`
    /// rotates to a fresh authority and deliberately invalidates every old
    /// receipt, as required for a new logical store incarnation.
    /// `preserve_index_authority` is independent: exact merge handoff preserves
    /// owner/slot identity, while destructive reset rotates it so the still-
    /// mutable detached snapshot cannot lend an old index handle to the live
    /// row-empty successor.
    ///
    /// This proof excludes sealing, clearing, physical purge, generation
    /// replacement, and every ordinary graph mutation for its whole lifetime.
    /// The caller must additionally hold its representation publication barrier
    /// (the layered store's generation write guard) while swapping the overlay
    /// and compact generations as one externally visible cut.
    #[cfg(feature = "compact-store")]
    fn prepare_empty_successor(
        &self,
        preserve_transport_authority: bool,
        preserve_index_authority: bool,
    ) -> Result<LpgStore, AllocError> {
        #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
        let _ = preserve_index_authority;
        let successor = LpgStore::with_config_inner(
            self.store.representation_config.clone(),
            #[cfg(any(feature = "vector-index", feature = "text-index"))]
            preserve_index_authority.then(|| {
                (
                    self.store.index_owner_id,
                    self.store.index_slots.lock().clone(),
                )
            }),
        )?;
        *successor.graph_identity.write() = self.store.graph_identity.read().clone();
        successor.current_epoch.store(
            self.store.current_epoch.load(Ordering::Acquire),
            Ordering::Release,
        );
        successor.retained_history_floor.store(
            self.store.retained_history_floor.load(Ordering::Acquire),
            Ordering::Release,
        );
        successor.next_node_id.store(
            self.store.next_node_id.load(Ordering::Acquire),
            Ordering::Release,
        );
        successor.next_edge_id.store(
            self.store.next_edge_id.load(Ordering::Acquire),
            Ordering::Release,
        );
        successor.mutation_scope.store(
            self.store.mutation_scope.load(Ordering::Acquire),
            Ordering::Release,
        );
        if preserve_transport_authority {
            *successor.transport_extract_authority.write() = Arc::clone(self.transport_authority());
            successor.transport_edges_may_be_unresolved.store(
                self.store
                    .transport_edges_may_be_unresolved
                    .load(Ordering::Acquire),
                Ordering::Release,
            );
        }
        Ok(successor)
    }

    /// Builds an empty successor for a new representation of the same logical
    /// store. Outstanding transport receipts retain their exact nonce meaning.
    #[cfg(feature = "compact-store")]
    pub(crate) fn prepare_same_incarnation_empty_successor(&self) -> Result<LpgStore, AllocError> {
        self.prepare_empty_successor(true, true)
    }

    /// Builds the fresh row/index representation used by destructive reset.
    ///
    /// Reset preserves the logical transport nonce and write seal, but rotates
    /// derived-index ownership. Its detached source remains independently
    /// mutable, so sharing owner/slot identity would let a retained caller
    /// handle rebind the discarded physical index into the live successor.
    #[cfg(feature = "compact-store")]
    pub(crate) fn prepare_reset_empty_successor(&self) -> Result<LpgStore, AllocError> {
        self.prepare_empty_successor(true, false)
    }

    /// Runs one empty-generation preparation/publication/rollback protocol
    /// while this already-acquired source transition remains exclusive.
    ///
    /// Layered temporal compaction acquires this proof before reading the
    /// source graph, then retains it through exact registry preparation and
    /// coherent publication. This variant avoids releasing and reacquiring the
    /// source cut between the temporal snapshot and the representation handoff.
    #[cfg(feature = "compact-store")]
    pub(crate) fn publish_empty_generation_after_prepare_and_publish_with_rollback<E, P, R>(
        &self,
        prepare: impl FnOnce(&PinnedLpgTransition<'_>) -> Result<P, E>,
        publish: impl FnOnce(P) -> R,
        rollback: impl FnOnce(R),
    ) -> Result<R, E> {
        #[cfg(test)]
        if let Some(barrier) = self.store.transport_purge_barrier.read().clone() {
            barrier.wait();
            barrier.wait();
        }
        let prepared = prepare(self)?;
        let published = publish(prepared);
        let boundary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(test)]
            if self
                .store
                .transport_post_publish_panic
                .swap(false, Ordering::SeqCst)
            {
                panic!("hostile post-publication LPG generation failpoint");
            }
        }));
        if let Err(payload) = boundary {
            rollback(published);
            std::panic::resume_unwind(payload);
        }
        Ok(published)
    }

    /// Builds an empty successor for a fresh logical store incarnation. Every
    /// receipt issued by the source incarnation is deliberately invalidated.
    #[cfg(all(feature = "compact-store", test))]
    pub(crate) fn prepare_fresh_incarnation_empty_successor(&self) -> Result<LpgStore, AllocError> {
        self.prepare_empty_successor(false, false)
    }
}

impl<'a> EdgeIdentityReservation<'a> {
    fn vacant(store: &'a LpgStore, id: EdgeId) -> Option<Self> {
        if !id.is_valid() {
            return None;
        }
        let mut reservation = Self {
            registry: &store.edge_identity_reservations,
            ids: Vec::with_capacity(1),
            armed: true,
        };
        let mut identities = store.edge_identity_reservations.identities.lock();
        if identities.contains(&id) {
            return None;
        }
        #[cfg(not(feature = "tiered-storage"))]
        let occupied = store.edges.read().contains_key(&id);
        #[cfg(feature = "tiered-storage")]
        let occupied = store.edge_versions.read().contains_key(&id);
        if occupied {
            return None;
        }
        identities.insert(id);
        reservation.ids.push(id);
        drop(identities);
        Some(reservation)
    }

    fn generated(store: &'a LpgStore, count: usize) -> Option<Self> {
        let mut reservation = Self {
            registry: &store.edge_identity_reservations,
            ids: Vec::with_capacity(count),
            armed: true,
        };
        let mut identities = store.edge_identity_reservations.identities.lock();
        #[cfg(not(feature = "tiered-storage"))]
        let published = store.edges.read();
        #[cfg(feature = "tiered-storage")]
        let published = store.edge_versions.read();
        while reservation.ids.len() < count {
            let Ok(raw) =
                store
                    .next_edge_id
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                        current.checked_add(1)
                    })
            else {
                return None;
            };
            let id = EdgeId::new(raw);
            if id.is_valid() && !identities.contains(&id) && !published.contains_key(&id) {
                identities.insert(id);
                reservation.ids.push(id);
            }
        }
        drop(published);
        drop(identities);
        Some(reservation)
    }

    /// Reserves one duplicate-free transition batch with explicit structural
    /// occupancy. Resident IDs must exist; authority-only IDs must be absent.
    fn matching_occupancy(
        store: &'a LpgStore,
        resident_ids: &[EdgeId],
        authority_only_ids: &[EdgeId],
    ) -> Option<Self> {
        let count = resident_ids.len().checked_add(authority_only_ids.len())?;
        let mut unique = FxHashSet::default();
        unique.reserve(count);
        if resident_ids
            .iter()
            .chain(authority_only_ids)
            .any(|id| !id.is_valid() || !unique.insert(*id))
        {
            return None;
        }
        let mut reservation = Self {
            registry: &store.edge_identity_reservations,
            ids: Vec::with_capacity(count),
            armed: true,
        };
        let mut identities = store.edge_identity_reservations.identities.lock();
        if unique.iter().any(|id| identities.contains(id)) {
            return None;
        }
        #[cfg(not(feature = "tiered-storage"))]
        let occupancy_matches = {
            let published = store.edges.read();
            resident_ids.iter().all(|id| published.contains_key(id))
                && authority_only_ids
                    .iter()
                    .all(|id| !published.contains_key(id))
        };
        #[cfg(feature = "tiered-storage")]
        let occupancy_matches = {
            let published = store.edge_versions.read();
            resident_ids.iter().all(|id| published.contains_key(id))
                && authority_only_ids
                    .iter()
                    .all(|id| !published.contains_key(id))
        };
        if !occupancy_matches {
            return None;
        }
        for id in resident_ids.iter().chain(authority_only_ids) {
            identities.insert(*id);
            reservation.ids.push(*id);
        }
        drop(identities);
        Some(reservation)
    }

    fn id(&self) -> EdgeId {
        self.ids[0]
    }

    fn ids(&self) -> &[EdgeId] {
        &self.ids
    }

    fn commit(mut self) {
        self.registry.release(&self.ids);
        self.armed = false;
    }
}

impl Drop for EdgeIdentityReservation<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.registry.release(&self.ids);
        }
    }
}

/// Move-only proof that one exact edge identity was created for subgraph
/// transport.
///
/// A receipt can only be obtained from atomic transport creation or validated
/// exact transport restore/recovery. Ordinary creation and ordinary recovery
/// never mint this provenance. Both receipt-bearing routes reserve the edge ID
/// and return no receipt for an existing identity. The private store authority
/// prevents a receipt from authorizing a same-numbered edge in another store.
/// The sealed structural identity includes its exact endpoints and immutable
/// edge type; property-only changes do not reincarnate it. This type
/// intentionally implements neither `Clone` nor `Copy`.
#[derive(Debug)]
pub struct TransportEdgeReceipt {
    authority: Arc<TransportExtractAuthority>,
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: ArcStr,
    destination_labels: Arc<[ArcStr]>,
    nonce: u64,
    lifetimes: Arc<[(EpochId, Option<EpochId>)]>,
}

/// Transaction-local proof that one exact open transport identity may retain
/// its carried provenance while an ordinary framed mutation changes only its
/// properties.
///
/// The grant is minted from a live [`TransportEdgeReceipt`] while the exact
/// database write authority is held. It deliberately carries no deletion or
/// purge capability. Commit validation rechecks the store incarnation,
/// structural lifetime, endpoints, and relationship type before accepting a
/// missing endpoint. Fields are private so downstream code cannot manufacture
/// a broad endpoint-validation bypass.
#[doc(hidden)]
#[derive(Debug)]
pub struct TransportEdgeMutationGrant {
    authority: Arc<TransportExtractAuthority>,
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: ArcStr,
    destination_labels: Arc<[ArcStr]>,
    nonce: u64,
    lifetimes: Arc<[(EpochId, Option<EpochId>)]>,
}

/// Exact structural phase of a receipt in one pinned store generation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportEdgeState {
    /// The original history remains intact and its final lifetime is open.
    Open,
    /// The original history ends committed closed and has exact adjacency debt.
    Closed,
    /// Authority, identity, lifetime, type, or adjacency no longer qualifies.
    Invalid,
}

/// Match newest-first structural history against the receipt's exact original
/// ascending history. Only closing its final open lifetime is authorized.
fn transport_lifetime_state(
    expected: &[(EpochId, Option<EpochId>)],
    mut actual: impl Iterator<Item = (EpochId, Option<EpochId>)>,
    open_since: EpochId,
    closed_since: EpochId,
) -> TransportEdgeState {
    let mut state = TransportEdgeState::Invalid;
    for (position, &(created, deleted)) in expected.iter().rev().enumerate() {
        let Some((actual_created, actual_deleted)) = actual.next() else {
            return TransportEdgeState::Invalid;
        };
        if actual_created != created
            || created == EpochId::PENDING
            || (actual_deleted != deleted && !(position == 0 && deleted.is_none()))
            || actual_deleted.is_some_and(|end| end == EpochId::PENDING || end < created)
        {
            return TransportEdgeState::Invalid;
        }
        if position == 0 {
            state = match actual_deleted {
                None if created <= open_since => TransportEdgeState::Open,
                Some(end) if created <= closed_since && end <= closed_since => {
                    TransportEdgeState::Closed
                }
                _ => TransportEdgeState::Invalid,
            };
        }
    }
    if actual.next().is_some() {
        TransportEdgeState::Invalid
    } else {
        state
    }
}

impl TransportEdgeReceipt {
    pub(crate) fn history_state(
        &self,
        actual: impl Iterator<Item = (EpochId, Option<EpochId>)>,
        open_since: EpochId,
        closed_since: EpochId,
    ) -> TransportEdgeState {
        transport_lifetime_state(&self.lifetimes, actual, open_since, closed_since)
    }
}

impl TransportEdgeMutationGrant {
    #[cfg(feature = "compact-store")]
    pub(crate) fn history_is_open(
        &self,
        actual: impl Iterator<Item = (EpochId, Option<EpochId>)>,
        since: EpochId,
    ) -> bool {
        transport_lifetime_state(&self.lifetimes, actual, since, since) == TransportEdgeState::Open
    }

    pub(crate) fn from_receipt(receipt: &TransportEdgeReceipt) -> Self {
        Self {
            authority: Arc::clone(&receipt.authority),
            id: receipt.id,
            src: receipt.src,
            dst: receipt.dst,
            edge_type: receipt.edge_type.clone(),
            destination_labels: Arc::clone(&receipt.destination_labels),
            nonce: receipt.nonce,
            lifetimes: Arc::clone(&receipt.lifetimes),
        }
    }

    pub(crate) fn belongs_to(&self, authority: &Arc<TransportExtractAuthority>) -> bool {
        Arc::ptr_eq(&self.authority, authority) && authority.contains(self.id, self.nonce)
    }

    #[cfg(feature = "compact-store")]
    pub(crate) const fn edge_id(&self) -> EdgeId {
        self.id
    }

    #[cfg(feature = "compact-store")]
    pub(crate) const fn source(&self) -> NodeId {
        self.src
    }

    #[cfg(feature = "compact-store")]
    pub(crate) const fn destination(&self) -> NodeId {
        self.dst
    }

    #[cfg(feature = "compact-store")]
    pub(crate) fn edge_type(&self) -> &str {
        &self.edge_type
    }

    /// Complete destination label set attested at the extract cut.
    ///
    /// Commit still revalidates the exact receipt/store identity before this
    /// witness can satisfy target endpoint schema constraints.
    #[doc(hidden)]
    #[must_use]
    pub fn destination_labels(&self) -> &[ArcStr] {
        &self.destination_labels
    }
}

impl TransportEdgeReceipt {
    /// Exact carried edge identity covered by this receipt.
    #[must_use]
    pub const fn edge_id(&self) -> EdgeId {
        self.id
    }

    /// Source endpoint captured when the carried identity was created.
    #[must_use]
    pub const fn source(&self) -> NodeId {
        self.src
    }

    /// Destination endpoint captured when the carried identity was created.
    #[must_use]
    pub const fn destination(&self) -> NodeId {
        self.dst
    }

    pub(crate) fn edge_type(&self) -> &str {
        &self.edge_type
    }

    /// Complete destination label set captured at the extract cut.
    #[doc(hidden)]
    #[must_use]
    pub fn destination_labels(&self) -> &[ArcStr] {
        &self.destination_labels
    }

    pub(crate) fn belongs_to(&self, authority: &Arc<TransportExtractAuthority>) -> bool {
        Arc::ptr_eq(&self.authority, authority) && authority.contains(self.id, self.nonce)
    }
}

impl Drop for TransportEdgeReceipt {
    fn drop(&mut self) {
        self.authority.revoke(self.id, self.nonce);
    }
}

/// Undo entry for a property mutation within a transaction.
///
/// Captures the previous state of a property so it can be restored on rollback.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PropertyUndoEntry {
    /// A node property was changed or added.
    NodeProperty {
        /// The node that was modified.
        node_id: NodeId,
        /// The property key that was set or removed.
        key: PropertyKey,
        /// The previous value, or `None` if the property did not exist before.
        old_value: Option<Value>,
    },
    /// An edge property was changed or added.
    EdgeProperty {
        /// The edge that was modified.
        edge_id: EdgeId,
        /// The property key that was set or removed.
        key: PropertyKey,
        /// The previous value, or `None` if the property did not exist before.
        old_value: Option<Value>,
    },
    /// A label was added to a node.
    LabelAdded {
        /// The node that had a label added.
        node_id: NodeId,
        /// The label string that was added.
        label: String,
    },
    /// A label was removed from a node.
    LabelRemoved {
        /// The node that had a label removed.
        node_id: NodeId,
        /// The label string that was removed.
        label: String,
    },
    /// A node was deleted (for rollback restoration).
    NodeDeleted {
        /// The node that was deleted.
        node_id: NodeId,
        /// The labels the node had before deletion.
        labels: Vec<String>,
        /// The properties the node had before deletion.
        properties: Vec<(PropertyKey, Value)>,
    },
    /// An edge was deleted (for rollback restoration).
    EdgeDeleted {
        /// The edge that was deleted.
        edge_id: EdgeId,
        /// The source node.
        src: NodeId,
        /// The destination node.
        dst: NodeId,
        /// The edge type name.
        edge_type: String,
        /// The properties the edge had before deletion.
        properties: Vec<(PropertyKey, Value)>,
    },
}

/// A single buffered (uncommitted) property mutation in a transaction's overlay.
///
/// Part of the unified-MVCC delta: uncommitted property writes are recorded
/// here (the "hot delta") instead of write-through to the committed column,
/// and merged over it by [`LpgStore::read_node_property_visible`].
#[derive(Debug, Clone)]
pub(super) enum PropOp {
    /// Set the property to a value.
    Set(Value),
    /// Remove the property (tombstone — hides the committed value for own-reads).
    Remove,
}

/// A buffered label change for a transaction's delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LabelOp {
    /// Add this label to the node.
    Add,
    /// Remove this label from the node.
    Remove,
}

type EntityPropertyEntries<I> = FxHashMap<(I, PropertyKey), PropOp>;

/// One authoritative property delta, partitioned by entity for own-write reads.
/// The flat iteration interface is retained for publication/index consumers;
/// each operation is stored only in its entity's bucket, including tombstones.
#[derive(Debug, Clone)]
pub(super) struct EntityPropertyDelta<I> {
    entities: FxHashMap<I, EntityPropertyEntries<I>>,
}

impl<I> Default for EntityPropertyDelta<I> {
    fn default() -> Self {
        Self {
            entities: FxHashMap::default(),
        }
    }
}

impl<I: Copy + Eq + std::hash::Hash> EntityPropertyDelta<I> {
    fn insert(&mut self, key: (I, PropertyKey), operation: PropOp) -> Option<PropOp> {
        self.entities
            .entry(key.0)
            .or_default()
            .insert(key, operation)
    }

    fn get(&self, key: &(I, PropertyKey)) -> Option<&PropOp> {
        self.entities.get(&key.0)?.get(key)
    }

    fn for_entity(&self, id: I) -> impl Iterator<Item = (&(I, PropertyKey), &PropOp)> {
        self.entities.get(&id).into_iter().flatten()
    }

    fn iter(&self) -> impl Iterator<Item = (&(I, PropertyKey), &PropOp)> {
        self.entities.values().flatten()
    }

    fn keys(&self) -> impl Iterator<Item = &(I, PropertyKey)> {
        self.iter().map(|(key, _)| key)
    }

    fn len(&self) -> usize {
        self.entities.values().map(FxHashMap::len).sum()
    }

    fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

impl<I> IntoIterator for EntityPropertyDelta<I> {
    type Item = ((I, PropertyKey), PropOp);
    type IntoIter =
        std::iter::Flatten<hashbrown::hash_map::IntoValues<I, EntityPropertyEntries<I>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entities.into_values().flatten()
    }
}

impl<'a, I> IntoIterator for &'a EntityPropertyDelta<I> {
    type Item = (&'a (I, PropertyKey), &'a PropOp);
    type IntoIter =
        std::iter::Flatten<hashbrown::hash_map::Values<'a, I, EntityPropertyEntries<I>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entities.values().flatten()
    }
}

/// A transaction's uncommitted writes — the per-transaction MVCC delta.
/// Carries node property ops, edge property ops, and node label ops.
/// Read-merged over the committed store for the writing transaction's own reads;
/// applied to the committed store on commit; dropped on rollback.
/// Edge-delete tracking is a later increment.
///
/// Also used as the savepoint snapshot type returned by
/// [`LpgStore::tx_overlay_snapshot`] and consumed by
/// [`LpgStore::tx_overlay_restore`].
#[doc(hidden)]
#[derive(Debug, Default, Clone)]
pub struct TxDelta {
    /// Uncommitted node property writes, grouped by node.
    pub(super) node_props: EntityPropertyDelta<NodeId>,
    /// Uncommitted edge property writes, grouped by edge.
    pub(super) edge_props: EntityPropertyDelta<EdgeId>,
    /// Uncommitted node label changes, keyed by (node, label_id).
    pub(super) node_labels: FxHashMap<(NodeId, u32), LabelOp>,
}

/// Compares two values for ordering (used for range checks).
pub(super) fn compare_values_for_range(a: &Value, b: &Value) -> Option<CmpOrdering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Checks if a value is within a range.
pub(super) fn value_in_range(
    value: &Value,
    min: Option<&Value>,
    max: Option<&Value>,
    min_inclusive: bool,
    max_inclusive: bool,
) -> bool {
    // Check lower bound
    if let Some(min_val) = min {
        match compare_values_for_range(value, min_val) {
            Some(CmpOrdering::Less) => return false,
            Some(CmpOrdering::Equal) if !min_inclusive => return false,
            None => return false, // Can't compare
            _ => {}
        }
    }

    // Check upper bound
    if let Some(max_val) = max {
        match compare_values_for_range(value, max_val) {
            Some(CmpOrdering::Greater) => return false,
            Some(CmpOrdering::Equal) if !max_inclusive => return false,
            None => return false,
            _ => {}
        }
    }

    true
}

/// Configuration for the LPG store.
///
/// The defaults work well for most cases. Tune `backward_edges` if you only
/// traverse in one direction (saves memory), or adjust capacities if you know
/// your graph size upfront (avoids reallocations).
#[derive(Debug, Clone)]
pub struct LpgStoreConfig {
    /// Maintain backward adjacency for incoming edge queries. Turn off if
    /// you only traverse outgoing edges - saves ~50% adjacency memory.
    pub backward_edges: bool,
    /// Initial capacity for nodes (avoids early reallocations).
    pub initial_node_capacity: usize,
    /// Initial capacity for edges (avoids early reallocations).
    pub initial_edge_capacity: usize,
}

impl Default for LpgStoreConfig {
    fn default() -> Self {
        Self {
            backward_edges: true,
            initial_node_capacity: 1024,
            initial_edge_capacity: 4096,
        }
    }
}

/// Bidirectional label name/ID registry.
///
/// Combines the name-to-ID and ID-to-name mappings behind a single lock,
/// reducing lock acquisitions on both the read path (`build_node`) and the
/// write path (`get_or_create_label_id`).
pub(super) struct LabelRegistry {
    /// Label name to numeric ID.
    name_to_id: FxHashMap<ArcStr, u32>,
    /// Numeric ID to label name (index = ID).
    id_to_name: Vec<ArcStr>,
}

impl LabelRegistry {
    fn new() -> Self {
        Self {
            name_to_id: FxHashMap::default(),
            id_to_name: Vec::new(),
        }
    }

    /// Looks up an existing label ID by name.
    pub(super) fn get_id(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    /// Returns the label name for a given ID.
    pub(super) fn get_name(&self, id: u32) -> Option<&ArcStr> {
        self.id_to_name.get(id as usize)
    }

    /// Returns or creates a label ID for the given name.
    pub(super) fn get_or_create(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        // reason: label registry size bounded by practical limits, fits u32
        #[allow(clippy::cast_possible_truncation)]
        let id = self.id_to_name.len() as u32;
        let label: ArcStr = name.into();
        self.name_to_id.insert(label.clone(), id);
        self.id_to_name.push(label);
        id
    }

    /// Returns the total number of distinct labels.
    pub(super) fn len(&self) -> usize {
        self.id_to_name.len()
    }

    /// Returns the ID-to-name slice for iteration.
    pub(super) fn names(&self) -> &[ArcStr] {
        &self.id_to_name
    }

    /// Clears all label mappings.
    pub(super) fn clear(&mut self) {
        self.name_to_id.clear();
        self.id_to_name.clear();
    }

    /// Removes an unpublished append-only suffix without allocating.
    fn truncate(&mut self, len: usize) {
        while self.id_to_name.len() > len {
            let Some(name) = self.id_to_name.pop() else {
                break;
            };
            self.name_to_id.remove(name.as_str());
        }
    }

    /// Estimates heap memory usage in bytes.
    pub(super) fn heap_bytes(&self) -> usize {
        let map_bytes = self.name_to_id.capacity()
            * (std::mem::size_of::<ArcStr>() + std::mem::size_of::<u32>());
        let vec_bytes = self.id_to_name.capacity() * std::mem::size_of::<ArcStr>();
        let string_bytes: usize = self.id_to_name.iter().map(|s| s.len()).sum();
        map_bytes + vec_bytes + string_bytes
    }
}

/// The core in-memory graph storage.
///
/// Everything lives here: nodes, edges, properties, adjacency indexes, and
/// version chains for MVCC. Concurrent reads never block each other.
///
/// Most users should go through `GrafeoDB` (from the `grafeo_engine` crate) which
/// adds transaction management and query execution. Use `LpgStore` directly
/// when you need raw performance for algorithm implementations.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_core::graph::Direction;
///
/// let store = LpgStore::new().expect("arena allocation");
///
/// // Create a small social network
/// let alix = store.create_node(&["Person"]);
/// let gus = store.create_node(&["Person"]);
/// store.create_edge(alix, gus, "KNOWS");
///
/// // Traverse outgoing edges
/// for neighbor in store.neighbors(alix, Direction::Outgoing) {
///     println!("Alix knows node {:?}", neighbor);
/// }
/// ```
///
/// # Lock Ordering
///
/// `LpgStore` contains multiple `RwLock` fields that must be acquired in a
/// consistent order to prevent deadlocks. The process-wide transition order is:
///
/// 1. named-graph topology gate;
/// 2. vector-index scope-transition proof;
/// 3. text-index scope-transition proof;
/// 4. parent store transition proof, then child store transition proofs.
///
/// Within one store, retain guards in this order when they overlap:
///
/// 1. store transition proof;
/// 2. transport-incarnation authority;
/// 3. node/edge identity reservation;
/// 4. structural entity storage;
/// 5. catalogs, adjacency, properties, indexes, counters, and transaction
///    bookkeeping.
///
/// Fallible arena/catalog preparation may run between an initial entity
/// preflight and a final entity recheck, but must not retain an entity-version
/// guard while allocating. Prepared catalog publication gates are reservations,
/// not published catalog locks.
///
/// ## Level 1: Entity Storage (mutually exclusive via feature flag)
/// 1. `nodes` / `node_versions`
/// 2. `edges` / `edge_versions`
///
/// ## Level 2: Catalogs
/// 3. `label_registry`
/// 4. `edge_type_to_id` + `id_to_edge_type`
///
/// ## Level 3: Indexes
/// 5. `label_index`
/// 6. `node_labels`
/// 7. `property_indexes`
///
/// ## Level 4: Statistics
/// 8. `statistics`
///
/// ## Level 5: Nested Locks (internal to other structs)
/// 9. `PropertyStorage::columns` (via `node_properties`/`edge_properties`)
/// 10. `ChunkedAdjacency::lists` (via `forward_adj`/`backward_adj`)
///
/// ## Rules
/// - Ordinary authoritative mutators retain the recursive shared store
///   transition proof for their complete composite operation. Clear, recursive
///   seal, physical purge, and generation replacement retain the exclusive side.
/// - Never hold entity locks while acquiring catalog locks in a different scope.
/// - Statistics lock is always last.
/// - Read locks are generally safe, but avoid read-to-write upgrades.
/// - Invoke external trackers/callbacks only after cloning their handles and
///   releasing registry guards.
pub struct LpgStore {
    /// Physical representation identity: unlike its address, this survives moves.
    index_physical_identity: Arc<PhysicalStoreIdentity>,
    /// Node records indexed by NodeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 1
    #[cfg(not(feature = "tiered-storage"))]
    pub(super) nodes: RwLock<FxHashMap<NodeId, VersionChain<NodeRecord>>>,

    /// Edge records indexed by EdgeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 2
    #[cfg(not(feature = "tiered-storage"))]
    pub(super) edges: RwLock<FxHashMap<EdgeId, VersionChain<EdgeRecord>>>,

    // === Tiered Storage Fields (feature-gated) ===
    //
    // Lock ordering for arena access:
    //   version_lock (read/write) → arena read lock (via arena_allocator.arena())
    //
    // Rules:
    // - Acquire arena read lock *after* version locks, never before.
    // - Multiple threads may call arena.read_at() concurrently (shared refs only).
    // - Never acquire arena write lock (alloc_new_chunk) while holding version locks.
    // - freeze_epoch order: node_versions.read() → arena.read_at(),
    //   then edge_versions.read() → arena.read_at().
    /// Arena allocator for hot data storage.
    /// Data is stored in per-epoch arenas for fast allocation and bulk deallocation.
    #[cfg(feature = "tiered-storage")]
    pub(super) arena_allocator: Arc<ArenaAllocator>,

    /// Node version indexes - store metadata and arena offsets.
    /// The actual NodeRecord data is stored in the arena.
    /// Lock order: 1
    #[cfg(feature = "tiered-storage")]
    pub(super) node_versions: RwLock<FxHashMap<NodeId, VersionIndex>>,

    /// Edge version indexes - store metadata and arena offsets.
    /// The actual EdgeRecord data is stored in the arena.
    /// Lock order: 2
    #[cfg(feature = "tiered-storage")]
    pub(super) edge_versions: RwLock<FxHashMap<EdgeId, VersionIndex>>,

    /// Out-of-map ownership for edge publication and physical removal.
    /// Readers and GC never observe a placeholder structural version.
    edge_identity_reservations: EdgeIdentityReservationRegistry,

    /// Out-of-map ownership for node publication and physical removal.
    node_identity_reservations: NodeIdentityReservationRegistry,

    /// Construction-time representation choices retained so an in-place
    /// compact/base transition can build an equivalent empty successor.
    representation_config: LpgStoreConfig,

    /// Whether this physical representation still owns mutation for its
    /// logical store incarnation.
    ///
    /// A compact generation handoff leaves the displaced `Arc<LpgStore>`
    /// fully readable, including its derived-index registries, but retires its
    /// mutation surface before releasing the source transition. Original index
    /// objects move to the exact successor while the retired source receives
    /// frozen forks, so neither graph nor fresh index reads can drift forward.
    #[cfg(feature = "compact-store")]
    representation_active: AtomicBool,

    /// Stable binding owner for retained vector/text index aliases. The slot
    /// registry permanently distinguishes exact encoded label/property keys,
    /// so one index Arc cannot silently combine two logical indexes.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    pub(super) index_owner_id: u64,
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    pub(super) index_slots: Mutex<FxHashMap<String, u64>>,

    /// Cold storage for frozen epochs.
    /// Contains compressed epoch blocks for historical data.
    #[cfg(feature = "tiered-storage")]
    pub(super) epoch_store: Arc<EpochStore>,

    /// Property storage for nodes.
    pub(super) node_properties: PropertyStorage<NodeId>,

    /// Property storage for edges.
    pub(super) edge_properties: PropertyStorage<EdgeId>,

    /// Bidirectional label name/ID registry.
    /// Lock order: 3
    pub(super) label_registry: RwLock<LabelRegistry>,

    /// Serializes append-only label-catalog preparation/publication. Compact
    /// promotion may retain this gate after publishing a reversible suffix so
    /// replay can resolve label names without another writer consuming it.
    label_publication_gate: Mutex<()>,

    /// Edge type name to ID mapping.
    /// Lock order: 4 (acquire with id_to_edge_type)
    pub(super) edge_type_to_id: RwLock<FxHashMap<ArcStr, u32>>,

    /// Serializes allocation-complete edge-type preparation with catalog
    /// publication, without retaining catalog locks across fallible arena work.
    edge_type_publication_gate: Mutex<()>,

    /// Edge type ID to name mapping.
    /// Lock order: 4 (acquire with edge_type_to_id)
    pub(super) id_to_edge_type: RwLock<Vec<ArcStr>>,

    /// Forward adjacency lists (outgoing edges).
    pub(super) forward_adj: ChunkedAdjacency,

    /// Backward adjacency lists (incoming edges).
    /// Only populated if config.backward_edges is true.
    pub(super) backward_adj: Option<ChunkedAdjacency>,

    /// Unforgeable, store-incarnation-local authority embedded in move-only
    /// transport receipts returned for atomically-created extract edges.
    ///
    /// Transport creation/purge hold the read side across their full structural
    /// operation. Full replacement holds the write side while clearing and
    /// rotates the authority before publication, so a receipt can never cross
    /// an identity-reusing store incarnation.
    pub(crate) transport_extract_authority: RwLock<Arc<TransportExtractAuthority>>,

    /// Sticky fast-path discriminator for query traversal. It is set before
    /// the first transport edge is structurally visible and cleared only with
    /// the whole store incarnation, avoiding receipt-revocation races without
    /// taxing ordinary stores with endpoint point-lookups.
    transport_edges_may_be_unresolved: AtomicBool,

    /// Deterministic test pause after transport identity publication but
    /// before receipt minting.
    #[cfg(test)]
    pub(super) transport_create_barrier: RwLock<Option<Arc<std::sync::Barrier>>>,

    /// Deterministic test pause after purge qualification but before physical
    /// removal.
    #[cfg(test)]
    pub(super) transport_purge_barrier: RwLock<Option<Arc<std::sync::Barrier>>>,

    /// One-shot hostile unwind after external publication but before the
    /// allocation-free internal purge commit.
    #[cfg(test)]
    pub(super) transport_post_publish_panic: AtomicBool,

    /// One-shot late resident qualification action: 1 rejects, 2 unwinds.
    #[cfg(test)]
    pub(super) generation_purge_late_action: std::sync::atomic::AtomicU8,
    #[cfg(test)]
    pub(super) generation_purge_late_hits: std::sync::atomic::AtomicUsize,

    /// Deterministic pause with an out-of-map edge reservation held.
    #[cfg(test)]
    pub(super) edge_publication_barrier: RwLock<Option<Arc<std::sync::Barrier>>>,

    /// Counted work for the bulk scan primitives, so tests can assert that a
    /// lookup's cost does not track the database size. Lock-free; see
    /// [`crate::graph::work_counters`].
    pub(super) work_counters: crate::graph::work_counters::WorkCounters,

    /// Label index: label_id -> set of node IDs.
    /// Lock order: 5
    pub(super) label_index: RwLock<Vec<FxHashMap<NodeId, ()>>>,

    /// Versioned node labels: node_id -> version log of label sets.
    /// Lock order: 6
    pub(super) node_labels: RwLock<FxHashMap<NodeId, VersionLog<FxHashSet<u32>>>>,

    /// Property indexes: property_key -> (value -> set of node IDs).
    ///
    /// When a property is indexed, lookups by value are O(1) instead of O(n).
    /// Use [`create_property_index`] to enable indexing for a property.
    /// Lock order: 7
    property_indexes: RwLock<FxHashMap<PropertyKey, RegisteredIndex<Arc<PropertyIndexRows>>>>,

    /// Vector indexes: "label:property" -> HNSW index.
    ///
    /// Created via [`GrafeoDB::create_vector_index`](grafeo_engine::GrafeoDB::create_vector_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "vector-index")]
    vector_indexes: RwLock<FxHashMap<String, RegisteredIndex<Arc<VectorIndexKind>>>>,

    /// One-shot compact predecessor binding for this physical representation.
    /// Topology-only vector indexes also consult its immutable property tier.
    ///
    /// The compact base does not point back to this store, so retaining it
    /// cannot form an ownership cycle. `OnceLock` makes the binding part of the
    /// unpublished representation construction: it cannot drift after the
    /// generation is published.
    #[cfg(all(feature = "compact-store", feature = "vector-index"))]
    pub(super) compact_base: std::sync::OnceLock<Arc<crate::graph::compact::CompactStore>>,
    // Without physical vector readers, adoption needs only the one-shot
    // identity. Do not keep the old in-memory base alive after an mmap swap.
    #[cfg(all(feature = "compact-store", not(feature = "vector-index")))]
    pub(super) compact_base:
        std::sync::OnceLock<std::sync::Weak<crate::graph::compact::CompactStore>>,

    /// Text indexes: "label:property" -> inverted index with BM25 scoring.
    ///
    /// Created via [`GrafeoDB::create_text_index`](grafeo_engine::GrafeoDB::create_text_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "text-index")]
    text_indexes: RwLock<FxHashMap<String, RegisteredIndex<RegisteredTextIndex>>>,

    /// Next node ID.
    pub(super) next_node_id: AtomicU64,

    /// Next edge ID.
    pub(super) next_edge_id: AtomicU64,

    /// Current epoch.
    pub(super) current_epoch: AtomicU64,

    /// Earliest complete historical view, advanced by actual version GC.
    /// A value's first write is not a retention boundary: older absence is known.
    pub(super) retained_history_floor: AtomicU64,

    /// Live (non-deleted) node count, maintained incrementally.
    /// Avoids O(n) full scan in `compute_statistics()`.
    pub(super) live_node_count: AtomicI64,

    /// Live (non-deleted) edge count, maintained incrementally.
    /// Avoids O(m) full scan in `compute_statistics()`.
    pub(super) live_edge_count: AtomicI64,

    /// Per-edge-type live counts, indexed by edge_type_id.
    /// Avoids O(m) edge scan in `compute_statistics()`.
    /// Lock order: 8 (same level as statistics)
    pub(super) edge_type_live_counts: RwLock<Vec<i64>>,

    /// Statistics for cost-based optimization.
    /// Lock order: 8 (always last)
    pub(super) statistics: RwLock<Arc<Statistics>>,

    /// Whether statistics need full recomputation (e.g., after rollback).
    pub(super) needs_stats_recompute: AtomicBool,

    /// Serializes the one-way unsealed -> sealed scope transition against
    /// operations that require an exact authority cut across callbacks.
    mutation_scope_gate: RwLock<()>,

    /// Named graphs, each an independent `LpgStore` partition.
    /// Zero overhead for single-graph databases (empty HashMap).
    /// Lock order: 9 (after statistics)
    named_graphs: RwLock<FxHashMap<String, Arc<LpgStore>>>,
    graph_identity: RwLock<graph_identity::GraphIdentity>,

    /// Undo log for property mutations within transactions.
    ///
    /// Maps transaction IDs to a list of undo entries that capture the
    /// previous property values. On rollback, entries are replayed in
    /// reverse order to restore properties. On commit, the entries are
    /// simply discarded.
    /// Lock order: 10 (after named_graphs, independent of other locks)
    property_undo_log: RwLock<FxHashMap<TransactionId, Vec<PropertyUndoEntry>>>,

    /// Per-transaction lists of entities created with PENDING versions, recorded
    /// at `create_*_versioned` — the single chokepoint every PENDING chain passes
    /// through (query operators, MERGE, LOAD DATA, and session-direct APIs alike).
    /// Used for write-set-scoped commit/rollback: complete by construction, unlike
    /// operator-level write tracking which MERGE/LOAD DATA bypass. Cleared when the
    /// transaction commits or rolls back.
    pending_tx_creates: RwLock<FxHashMap<TransactionId, (Vec<NodeId>, Vec<EdgeId>)>>,

    /// Per-transaction uncommitted property delta (the "hot tier" of the unified
    /// MVCC model, first increment). Uncommitted property writes land here and are
    /// merged over the committed column for the writing transaction's own reads;
    /// applied to the committed column on commit, dropped on rollback. Other
    /// sessions never see it. Lock order: after `pending_tx_creates`.
    tx_property_overlay: RwLock<FxHashMap<TransactionId, TxDelta>>,

    /// Per-transaction uncommitted text-index delta (mirrors `tx_property_overlay`).
    ///
    /// When a transactional write touches a property covered by a text index the
    /// change is buffered here instead of mutating the committed `InvertedIndex`.
    /// Applied to the committed index at commit (TI5 — not yet) and dropped on
    /// rollback.  The committed index is never mutated on the transactional path,
    /// keeping it as the snapshot-consistent base for all other readers.
    /// Lock order: same level as `tx_property_overlay`, independent key space.
    #[cfg(feature = "text-index")]
    pub(crate) text_index_overlay:
        RwLock<FxHashMap<TransactionId, crate::index::text::TextIndexDelta>>,

    /// Per-transaction lists of node IDs deleted with a PENDING `deleted_epoch`,
    /// recorded at `delete_node_transactional` — the chokepoint that defers
    /// label-index/adjacency removal until commit. Finalized by
    /// `finalize_deletes_by_id`, dropped on rollback.
    /// Lock order: after `tx_property_overlay`.
    pub(crate) pending_tx_deletes: RwLock<FxHashMap<TransactionId, Vec<NodeId>>>,

    /// Per-transaction lists of edges deleted with a PENDING `deleted_epoch`,
    /// recorded at `delete_edge_transactional` — the chokepoint that defers the
    /// adjacency tombstone, edge-property removal, and live/edge-type count
    /// decrements until commit. Each tuple is `(src, edge, dst)`. Finalized by
    /// `finalize_edge_deletes_by_id`, dropped (via `unmark_deleted_by`) on rollback.
    /// Lock order: after `pending_tx_deletes`.
    pub(crate) pending_tx_edge_deletes:
        RwLock<FxHashMap<TransactionId, Vec<(NodeId, EdgeId, NodeId)>>>,

    /// Per-transaction read trackers (Serializable only). Set by the engine at
    /// Serializable tx begin, dropped at commit/rollback. When present for a tx, the
    /// visible-read accessors record observed entities into it (the SSI read-set,
    /// complete by construction). Empty for SI/ReadCommitted (zero cost).
    read_trackers: RwLock<FxHashMap<TransactionId, SharedReadTracker>>,

    /// Per-transaction write trackers (Serializable only). Set by the engine at
    /// Serializable tx begin, dropped at commit/rollback. When present, the
    /// buffered indexed-write path records index writes via
    /// [`crate::execution::operators::WriteTracker::record_index_write`] for anti-phantom SSI.
    /// Parallel to `read_trackers`; empty for SI/ReadCommitted (zero cost).
    write_trackers: RwLock<FxHashMap<TransactionId, SharedWriteTracker>>,

    /// Database authority scope required by public mutators (`0` = unsealed).
    mutation_scope: AtomicU64,
}

impl LpgStore {
    /// Retained weak concrete identities for nonblocking retirement tests.
    /// Unlike a view, these probes cannot keep a dropped payload alive.
    #[cfg(all(test, feature = "text-index", feature = "compact-store"))]
    pub(crate) fn generation_text_targets_for_test(
        &self,
    ) -> Vec<std::sync::Weak<RwLock<crate::index::text::InvertedIndex>>> {
        self.text_indexes
            .read()
            .values()
            .map(|index| index.target_identity())
            .collect()
    }

    /// Creates a new LPG store with default configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if a required allocator or checked store/index
    /// binding identity cannot be allocated.
    pub fn new() -> Result<Self, AllocError> {
        Self::with_config(LpgStoreConfig::default())
    }

    /// Creates a new LPG store with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if a required allocator or checked store/index
    /// binding identity cannot be allocated.
    pub fn with_config(config: LpgStoreConfig) -> Result<Self, AllocError> {
        Self::with_config_inner(
            config,
            #[cfg(any(feature = "vector-index", feature = "text-index"))]
            None,
        )
    }

    /// Internal constructor that can preserve the exact derived-index binding
    /// identity of a same-incarnation representation successor without
    /// consuming a fresh global owner ID.
    fn with_config_inner(
        config: LpgStoreConfig,
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        preserved_index_binding: Option<(u64, FxHashMap<String, u64>)>,
    ) -> Result<Self, AllocError> {
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        let (index_owner_id, index_slots) = if let Some((owner, slots)) = preserved_index_binding {
            if owner == 0 || slots.values().any(|slot| *slot == 0) {
                return Err(AllocError::InsufficientSpace);
            }
            (owner, slots)
        } else {
            (
                allocate_index_binding_id(&NEXT_INDEX_OWNER_ID)?,
                FxHashMap::default(),
            )
        };
        let backward_adj = if config.backward_edges {
            Some(ChunkedAdjacency::new())
        } else {
            None
        };

        Ok(Self {
            index_physical_identity: Arc::new(PhysicalStoreIdentity),
            #[cfg(not(feature = "tiered-storage"))]
            nodes: RwLock::new(FxHashMap::default()),
            #[cfg(not(feature = "tiered-storage"))]
            edges: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            arena_allocator: Arc::new(ArenaAllocator::new()?),
            #[cfg(feature = "tiered-storage")]
            node_versions: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            edge_versions: RwLock::new(FxHashMap::default()),
            edge_identity_reservations: EdgeIdentityReservationRegistry::new(),
            node_identity_reservations: NodeIdentityReservationRegistry::new(),
            representation_config: config.clone(),
            #[cfg(feature = "compact-store")]
            representation_active: AtomicBool::new(true),
            #[cfg(any(feature = "vector-index", feature = "text-index"))]
            index_owner_id,
            #[cfg(any(feature = "vector-index", feature = "text-index"))]
            index_slots: Mutex::new(index_slots),
            #[cfg(feature = "tiered-storage")]
            epoch_store: Arc::new(EpochStore::new()),
            node_properties: PropertyStorage::new(),
            edge_properties: PropertyStorage::new(),
            label_registry: RwLock::new(LabelRegistry::new()),
            label_publication_gate: Mutex::new(()),
            edge_type_to_id: RwLock::new(FxHashMap::default()),
            edge_type_publication_gate: Mutex::new(()),
            id_to_edge_type: RwLock::new(Vec::new()),
            forward_adj: ChunkedAdjacency::new(),
            backward_adj,
            transport_extract_authority: RwLock::new(Arc::new(TransportExtractAuthority::new())),
            transport_edges_may_be_unresolved: AtomicBool::new(false),
            #[cfg(test)]
            transport_create_barrier: RwLock::new(None),
            #[cfg(test)]
            transport_purge_barrier: RwLock::new(None),
            #[cfg(test)]
            transport_post_publish_panic: AtomicBool::new(false),
            #[cfg(test)]
            generation_purge_late_action: std::sync::atomic::AtomicU8::new(0),
            #[cfg(test)]
            generation_purge_late_hits: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            edge_publication_barrier: RwLock::new(None),
            work_counters: crate::graph::work_counters::WorkCounters::default(),
            label_index: RwLock::new(Vec::with_capacity(16)),
            node_labels: RwLock::new(FxHashMap::default()),
            property_indexes: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "vector-index")]
            vector_indexes: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "compact-store")]
            compact_base: std::sync::OnceLock::new(),
            #[cfg(feature = "text-index")]
            text_indexes: RwLock::new(FxHashMap::default()),
            next_node_id: AtomicU64::new(0),
            next_edge_id: AtomicU64::new(0),
            current_epoch: AtomicU64::new(0),
            retained_history_floor: AtomicU64::new(0),
            live_node_count: AtomicI64::new(0),
            live_edge_count: AtomicI64::new(0),
            edge_type_live_counts: RwLock::new(Vec::new()),
            statistics: RwLock::new(Arc::new(Statistics::new())),
            needs_stats_recompute: AtomicBool::new(false),
            mutation_scope_gate: RwLock::new(()),
            named_graphs: RwLock::new(FxHashMap::default()),
            graph_identity: RwLock::new(graph_identity::GraphIdentity::default()),
            property_undo_log: RwLock::new(FxHashMap::default()),
            pending_tx_creates: RwLock::new(FxHashMap::default()),
            tx_property_overlay: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "text-index")]
            text_index_overlay: RwLock::new(FxHashMap::default()),
            pending_tx_deletes: RwLock::new(FxHashMap::default()),
            pending_tx_edge_deletes: RwLock::new(FxHashMap::default()),
            read_trackers: RwLock::new(FxHashMap::default()),
            write_trackers: RwLock::new(FxHashMap::default()),
            mutation_scope: AtomicU64::new(0),
        })
    }

    /// Seals this store to one database-scoped mutation authority.
    ///
    /// Returns `false` if the store was already sealed by a different
    /// authority. Named substores inherit the same scope recursively.
    pub fn seal_unframed_writes(
        &self,
        authority: &crate::graph::write_permit::WriteAuthority,
    ) -> bool {
        crate::graph::write_permit::with_authority(authority, || {
            self.seal_with_scope(authority.scope().get())
        })
    }

    fn seal_with_scope(&self, scope: u64) -> bool {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return false;
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        #[cfg(feature = "vector-index")]
        let vector_transition = VectorIndexKind::pin_scope_transition();
        #[cfg(feature = "text-index")]
        let text_transition = InvertedIndex::pin_scope_transition();
        let mut visiting = FxHashSet::default();
        let mut checked = FxHashSet::default();
        if !self.can_seal_named_graph_tree(
            scope,
            &mut visiting,
            &mut checked,
            #[cfg(feature = "vector-index")]
            &|index, owner, slot| {
                index.is_bound_to(owner, slot, &vector_transition)
                    && index.scope_is_compatible(scope, &vector_transition)
            },
            #[cfg(feature = "text-index")]
            &|index, owner, slot| {
                index.is_bound_to(owner, slot, &text_transition)
                    && index.scope_is_compatible(scope, &text_transition)
            },
        ) {
            return false;
        }
        let mut sealed = FxHashSet::default();
        self.seal_named_graph_tree_preflighted(
            scope,
            &mut sealed,
            #[cfg(feature = "vector-index")]
            &|index| index.seal_with_scope_under_transition(scope, &vector_transition),
            #[cfg(feature = "text-index")]
            &|index| index.seal_with_scope_under_transition(scope, &text_transition),
        )
    }

    /// Read-only recursive preflight for one failure-atomic named-graph seal.
    /// The topology gate prevents installs/drops while this walk and the apply
    /// phase run. A recursion-stack hit is an alias cycle and fails closed.
    fn can_seal_named_graph_tree(
        &self,
        scope: u64,
        visiting: &mut FxHashSet<usize>,
        checked: &mut FxHashSet<usize>,
        #[cfg(feature = "vector-index")] vector_compatible: &dyn Fn(
            &VectorIndexKind,
            u64,
            u64,
        ) -> bool,
        #[cfg(feature = "text-index")] text_compatible: &dyn Fn(&InvertedIndex, u64, u64) -> bool,
    ) -> bool {
        #[cfg(feature = "compact-store")]
        if !self.representation_is_active() {
            return false;
        }
        let identity = std::ptr::from_ref(self).addr();
        if checked.contains(&identity) {
            return true;
        }
        if !visiting.insert(identity) {
            return false;
        }
        let current = self.mutation_scope.load(Ordering::Acquire);
        if current != 0 && current != scope {
            visiting.remove(&identity);
            return false;
        }
        #[cfg(feature = "vector-index")]
        {
            let slots = self.index_slots.lock();
            if self.vector_indexes.read().iter().any(|(key, index)| {
                slots
                    .get(key)
                    .is_none_or(|&slot| !vector_compatible(index, self.index_owner_id, slot))
            }) {
                visiting.remove(&identity);
                return false;
            }
        }
        #[cfg(feature = "text-index")]
        {
            let slots = self.index_slots.lock();
            if self.text_indexes.read().iter().any(|(key, index)| {
                let index = index.read();
                slots
                    .get(key)
                    .is_none_or(|&slot| !text_compatible(&index, self.index_owner_id, slot))
            }) {
                visiting.remove(&identity);
                return false;
            }
        }
        let children: Vec<_> = self.named_graphs.read().values().cloned().collect();
        if children.iter().any(|child| {
            !child.can_seal_named_graph_tree(
                scope,
                visiting,
                checked,
                #[cfg(feature = "vector-index")]
                vector_compatible,
                #[cfg(feature = "text-index")]
                text_compatible,
            )
        }) {
            visiting.remove(&identity);
            return false;
        }
        visiting.remove(&identity);
        checked.insert(identity);
        true
    }

    /// Applies a fully preflighted recursive seal. Parent transition guards are
    /// retained while child scopes are published, establishing the documented
    /// parent-map-child lock order. Shared aliases are sealed once.
    fn seal_named_graph_tree_preflighted(
        &self,
        scope: u64,
        sealed: &mut FxHashSet<usize>,
        #[cfg(feature = "vector-index")] seal_vector: &dyn Fn(&VectorIndexKind) -> bool,
        #[cfg(feature = "text-index")] seal_text: &dyn Fn(&InvertedIndex) -> bool,
    ) -> bool {
        let identity = std::ptr::from_ref(self).addr();
        if !sealed.insert(identity) {
            return true;
        }
        let _scope_transition = self.mutation_scope_gate.write();
        #[cfg(feature = "compact-store")]
        if !self.representation_is_active() {
            return false;
        }
        let children: Vec<_> = self.named_graphs.read().values().cloned().collect();
        if children.iter().any(|child| {
            !child.seal_named_graph_tree_preflighted(
                scope,
                sealed,
                #[cfg(feature = "vector-index")]
                seal_vector,
                #[cfg(feature = "text-index")]
                seal_text,
            )
        }) {
            return false;
        }
        // Pair with the under-lock authority re-check in property-index DDL.
        // Once this guard is acquired, an insertion either completed before
        // sealing or will observe the published scope and fail closed.
        let _property_indexes = self.property_indexes.read();
        #[cfg(feature = "vector-index")]
        for index in self.vector_indexes.read().values() {
            if !seal_vector(index) {
                return false;
            }
        }
        #[cfg(feature = "text-index")]
        for index in self.text_indexes.read().values() {
            let index = index.read();
            if !seal_text(&index) {
                return false;
            }
        }
        match self
            .mutation_scope
            .compare_exchange(0, scope, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(existing) => existing == scope,
        }
    }

    /// Collects one topology into `seen`, rejecting both cycles and aliases.
    /// Callers retain [`NAMED_GRAPH_TOPOLOGY_GATE`] for the complete walk.
    fn collect_unique_named_graph_tree(&self, seen: &mut FxHashSet<usize>) -> bool {
        let identity = std::ptr::from_ref(self).addr();
        if !seen.insert(identity) {
            return false;
        }
        let children: Vec<_> = self.named_graphs.read().values().cloned().collect();
        children
            .iter()
            .all(|child| child.collect_unique_named_graph_tree(seen))
    }

    /// Seals one already topology-validated child while the caller retains the
    /// process topology gate, the global vector/text transition proofs, and
    /// this parent store's shared mutation proof. The caller acquires those
    /// global proofs before the store proof to preserve the process-wide lock
    /// order.
    fn seal_child_under_topology_gate(
        &self,
        child: &LpgStore,
        scope: u64,
        #[cfg(feature = "vector-index")] vector_compatible: &dyn Fn(
            &VectorIndexKind,
            u64,
            u64,
        ) -> bool,
        #[cfg(feature = "vector-index")] seal_vector: &dyn Fn(&VectorIndexKind) -> bool,
        #[cfg(feature = "text-index")] text_compatible: &dyn Fn(&InvertedIndex, u64, u64) -> bool,
        #[cfg(feature = "text-index")] seal_text: &dyn Fn(&InvertedIndex) -> bool,
    ) -> bool {
        let mut visiting = FxHashSet::default();
        let mut checked = FxHashSet::default();
        child.can_seal_named_graph_tree(
            scope,
            &mut visiting,
            &mut checked,
            #[cfg(feature = "vector-index")]
            vector_compatible,
            #[cfg(feature = "text-index")]
            text_compatible,
        ) && child.seal_named_graph_tree_preflighted(
            scope,
            &mut FxHashSet::default(),
            #[cfg(feature = "vector-index")]
            seal_vector,
            #[cfg(feature = "text-index")]
            seal_text,
        )
    }

    /// Exclusively pins the current mutation scope and representation after
    /// checking the caller's exact unwind-safe write authority.
    ///
    /// Sealing, clearing, ordinary mutation, and another whole-generation
    /// transition cannot race the returned proof. Callers must never try to
    /// upgrade a shared mutation pin into this exclusive proof.
    pub(crate) fn pin_exclusive_unframed_transition(&self) -> Option<PinnedLpgTransition<'_>> {
        let scope_transition = self.pin_exclusive_authorized_scope()?;
        // `clear` retains the opposing write side across reset and authority
        // rotation, so the proof pins one clear/seal incarnation.
        let transport_authority = self.transport_extract_authority.read();
        Some(PinnedLpgTransition {
            store: self,
            _scope_transition: scope_transition,
            transport_authority,
        })
    }

    /// Returns whether this store currently carries any transport-only edge
    /// authority. Ordinary stores keep this false, allowing query expansion to
    /// avoid endpoint point-lookups on its hot path.
    pub(crate) fn may_have_unresolved_transport_edges(&self) -> bool {
        self.transport_edges_may_be_unresolved
            .load(Ordering::Acquire)
    }

    /// Pins the exclusive side of the store transition barrier after checking
    /// current mutation authority, without retaining receipt authority.
    ///
    /// This narrower proof is used by `clear`, which must rotate receipt
    /// authority while the transition remains exclusive. It cannot use
    /// [`Self::pin_exclusive_unframed_transition`] because that proof
    /// deliberately retains the opposing receipt-authority read guard.
    fn pin_exclusive_authorized_scope(&self) -> Option<PinnedExclusiveScope<'_>> {
        if ExclusiveBulkRestoreContext::is_read_only(self) {
            return None;
        }
        let scope_transition = self.mutation_scope_gate.write();
        #[cfg(feature = "compact-store")]
        if !self.representation_is_active() {
            return None;
        }
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0
            && !std::num::NonZeroU64::new(scope).is_some_and(crate::graph::write_permit::is_held)
        {
            return None;
        }
        Some(PinnedExclusiveScope {
            _reads: ExclusiveReadScope::enter(self),
            _guard: scope_transition,
        })
    }

    /// Pins an ordinary mutation to one authorized scope and representation.
    /// Nested public mutators use a recursive shared read; callers must retain
    /// the outer proof until every structural and derived write is complete.
    pub(crate) fn pin_mutation(&self) -> Option<PinnedMutation<'_>> {
        if ExclusiveBulkRestoreContext::is_active(self) {
            if ExclusiveBulkRestoreContext::is_read_only(self) {
                return None;
            }
            #[cfg(feature = "compact-store")]
            if !self.representation_is_active() {
                return None;
            }
            return Some(PinnedMutation {
                _scope_transition: None,
            });
        }
        let scope_transition = self.mutation_scope_gate.read_recursive();
        #[cfg(feature = "compact-store")]
        if !self.representation_is_active() {
            return None;
        }
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0
            && !std::num::NonZeroU64::new(scope).is_some_and(crate::graph::write_permit::is_held)
        {
            return None;
        }
        Some(PinnedMutation {
            _scope_transition: Some(scope_transition),
        })
    }

    /// Pins a composite raw read against whole-store replacement. Layered
    /// generation readers already retain their exact overlay and publication
    /// barrier: taking the overlay scope beneath that barrier would invert
    /// the merger's overlay-scope -> publication order.
    fn pin_read(&self) -> PinnedMaintenance<'_> {
        #[cfg(feature = "compact-store")]
        if crate::graph::compact::layered::has_retained_overlay_read(self) {
            return PinnedMaintenance {
                _scope_transition: None,
            };
        }
        self.pin_maintenance()
    }

    /// Pins non-authoritative derived-cache maintenance against destructive
    /// representation transitions without requiring write authority.
    pub(crate) fn pin_maintenance(&self) -> PinnedMaintenance<'_> {
        if ExclusiveBulkRestoreContext::is_active(self) || ExclusiveReadScope::is_held(self) {
            return PinnedMaintenance {
                _scope_transition: None,
            };
        }
        PinnedMaintenance {
            _scope_transition: Some(self.mutation_scope_gate.read_recursive()),
        }
    }

    /// Exclusively pins the representation barrier for a non-authoritative
    /// whole-store derived rebuild, without requiring mutation authority.
    ///
    /// The bulk-restore context already owns this exact store's exclusive side;
    /// nested restore helpers reuse that proof instead of self-deadlocking.
    pub(crate) fn pin_exclusive_maintenance(&self) -> PinnedExclusiveMaintenance<'_> {
        if ExclusiveBulkRestoreContext::is_active(self) {
            return PinnedExclusiveMaintenance {
                _scope_transition: None,
            };
        }
        PinnedExclusiveMaintenance {
            _scope_transition: Some(PinnedExclusiveScope {
                _guard: self.mutation_scope_gate.write(),
                _reads: ExclusiveReadScope::enter(self),
            }),
        }
    }

    /// Runs a pristine-target bulk restore under one exclusive, authority-
    /// checked transition. Nested ordinary mutator boundaries dynamically reuse
    /// this exact proof and therefore never acquire shared beneath exclusive.
    ///
    /// `None` means the current thread lacks this sealed store's write authority.
    /// The scoped context and exclusive proof are both unwind-safe.
    pub(crate) fn with_exclusive_bulk_restore<T>(
        &self,
        restore: impl FnOnce(&PinnedLpgTransition<'_>) -> T,
    ) -> Option<T> {
        let transition = self.pin_exclusive_unframed_transition()?;
        let _context = ExclusiveBulkRestoreContext::enter(self);
        Some(restore(&transition))
    }

    /// Replays authoritative data without deriving recorded index changes twice.
    ///
    /// Only unsealed, unpublished startup stores are admitted. The caller must
    /// install the requested families' recorded postimages before exposing the
    /// recovered store. Property maintenance and authoritative histories remain
    /// unchanged, as do families not selected for suppression. Suppression
    /// applies only to this exact store and stack scope, not image installation.
    ///
    /// # Errors
    /// Rejects unsupported requested families and sealed, retired, or recursively
    /// scoped stores before calling `operation`; propagates its error.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    #[doc(hidden)]
    pub fn with_recorded_index_recovery<T>(
        &self,
        record_text: bool,
        record_vector: bool,
        operation: impl FnOnce() -> grafeo_common::utils::error::Result<T>,
    ) -> grafeo_common::utils::error::Result<T> {
        if (record_text && !cfg!(feature = "text-index"))
            || (record_vector && !cfg!(feature = "vector-index"))
        {
            return Err(grafeo_common::Error::InvalidValue(
                "recorded index recovery requests an unavailable index family".into(),
            ));
        }
        let rejected = || {
            grafeo_common::Error::InvalidValue(
                "recorded index recovery requires an unsealed, unscoped startup store".into(),
            )
        };
        if self.mutation_scope.load(Ordering::Acquire) != 0
            || ExclusiveBulkRestoreContext::is_active(self)
        {
            return Err(rejected());
        }
        self.with_exclusive_bulk_restore(|_| {
            // A concurrent seal may have won before the exclusive acquisition.
            if self.mutation_scope.load(Ordering::Acquire) != 0 {
                return Err(rejected());
            }
            let _indexes = ExclusiveBulkRestoreContext::enter_mode(
                self,
                ExclusiveMode::RecordedIndexRecovery {
                    #[cfg(feature = "text-index")]
                    text: record_text,
                    #[cfg(feature = "vector-index")]
                    vector: record_vector,
                },
            );
            operation()
        })
        .ok_or_else(rejected)?
    }

    #[cfg(feature = "text-index")]
    fn recorded_text_recovery_active(&self) -> bool {
        ExclusiveBulkRestoreContext::records_text(self)
    }

    #[cfg(feature = "vector-index")]
    fn recorded_vector_recovery_active(&self) -> bool {
        ExclusiveBulkRestoreContext::records_vector(self)
    }

    /// Returns whether this store is unsealed or is sealed to `authority`, and
    /// the caller currently holds that exact unwind-safe authority scope.
    pub(crate) fn accepts_held_write_authority(
        &self,
        authority: &crate::graph::write_permit::WriteAuthority,
    ) -> bool {
        #[cfg(feature = "compact-store")]
        if !self.representation_is_active() {
            return false;
        }
        if !crate::graph::write_permit::is_held(authority.scope()) {
            return false;
        }
        let scope = self.mutation_scope.load(Ordering::Acquire);
        scope == 0 || scope == authority.scope().get()
    }

    /// Pins the process-wide named-graph registry while a compact replacement
    /// prepares and publishes an exact same-incarnation topology.
    #[cfg(feature = "compact-store")]
    pub(crate) fn pin_named_graph_topology(&self) -> PinnedNamedGraphTopology {
        PinnedNamedGraphTopology {
            _guard: NAMED_GRAPH_TOPOLOGY_GATE.lock(),
        }
    }

    /// Whether this physical representation still owns mutation for its
    /// logical store incarnation. Retired representations remain read-only
    /// snapshots until their final `Arc` is released.
    #[cfg(feature = "compact-store")]
    pub(super) fn representation_is_active(&self) -> bool {
        self.representation_active.load(Ordering::Acquire)
    }

    /// Retires one physical representation while its exclusive transition is
    /// held by the compact generation publisher.
    #[cfg(feature = "compact-store")]
    pub(super) fn retire_representation(&self) {
        let was_active = self.representation_active.swap(false, Ordering::AcqRel);
        assert!(was_active, "LPG representation retired exactly once");
    }

    /// Reactivates the exact source representation during publication
    /// rollback. No ordinary mutator can overlap because the original
    /// exclusive transition remains held.
    #[cfg(feature = "compact-store")]
    pub(super) fn reactivate_representation(&self) {
        let was_active = self.representation_active.swap(true, Ordering::AcqRel);
        assert!(
            !was_active,
            "only a retired LPG representation can be restored"
        );
    }

    #[cfg(test)]
    fn pause_edge_publication_for_test(&self) {
        // End the configuration read guard before waiting. Tests deliberately
        // disable the hook while a publisher is paused so a competing creator
        // can exercise the same reservation registry without joining this
        // two-party barrier.
        let barrier = { self.edge_publication_barrier.read().clone() };
        if let Some(barrier) = barrier {
            barrier.wait();
            barrier.wait();
        }
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire))
    }

    /// Earliest epoch whose structural and property history remains complete.
    #[must_use]
    pub fn retained_history_floor(&self) -> EpochId {
        EpochId::new(self.retained_history_floor.load(Ordering::Acquire))
    }

    /// Advances coverage after collection or an explicitly qualified restore.
    /// The caller must retain the store's mutation/publication authority.
    #[doc(hidden)]
    pub fn advance_retained_history_floor(&self, floor: EpochId) {
        if floor != EpochId::PENDING {
            self.retained_history_floor
                .fetch_max(floor.as_u64(), Ordering::AcqRel);
        }
    }

    /// Reads the counted bulk-scan work this store has performed.
    ///
    /// Take one snapshot before a query and one after, then assert on
    /// [`crate::graph::work_counters::WorkSnapshot::since`]: a point lookup must scan nothing, and its cost
    /// must not change when the database grows. Every wrapping store delegates
    /// its scans here, so a snapshot of this store sees work performed through
    /// `WalStore`, `CdcStore` or a snapshot view as well.
    #[must_use]
    pub fn work_snapshot(&self) -> crate::graph::work_counters::WorkSnapshot {
        self.work_counters.snapshot()
    }

    /// Creates a new epoch.
    #[doc(hidden)]
    pub fn new_epoch(&self) -> EpochId {
        let Some(_mutation) = self.pin_mutation() else {
            return self.current_epoch();
        };
        let id = self
            .current_epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < EpochId::PENDING.as_u64() - 1).then_some(current + 1)
            })
            .expect("epoch identity space exhausted")
            + 1;
        EpochId::new(id)
    }

    /// Syncs the store epoch to match an external epoch counter.
    ///
    /// Used by the transaction manager to keep the store's epoch in step
    /// after a transaction commit advances the global epoch.
    #[doc(hidden)]
    pub fn sync_epoch(&self, epoch: EpochId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        if epoch == EpochId::PENDING {
            return;
        }
        self.current_epoch
            .fetch_max(epoch.as_u64(), Ordering::AcqRel);
    }

    /// Returns the current next node ID counter value.
    #[doc(hidden)]
    #[must_use]
    pub fn next_node_id(&self) -> u64 {
        self.next_node_id.load(Ordering::Acquire)
    }

    /// Returns the current next edge ID counter value.
    #[doc(hidden)]
    #[must_use]
    pub fn next_edge_id(&self) -> u64 {
        self.next_edge_id.load(Ordering::Acquire)
    }

    /// Sets the next node ID counter.
    ///
    /// Used by [`LayeredStore`](crate::graph::compact::layered::LayeredStore)
    /// to seed the overlay's ID allocator above the compact base's maximum ID.
    #[doc(hidden)]
    pub fn set_next_node_id(&self, id: u64) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.next_node_id.fetch_max(id, Ordering::AcqRel);
    }

    /// Sets the next edge ID counter.
    ///
    /// See [`set_next_node_id`](Self::set_next_node_id).
    #[doc(hidden)]
    pub fn set_next_edge_id(&self, id: u64) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.next_edge_id.fetch_max(id, Ordering::AcqRel);
    }

    /// Restores both entity allocator high-water marks after exact history recovery.
    ///
    /// The requested counters must not regress the counters already derived
    /// from restored identities, and each must remain strictly above every
    /// corresponding known identity. Validation of both counters completes
    /// before either is installed. This is a recovery-only seam for an
    /// unpublished, quiescent store.
    ///
    /// # Errors
    ///
    /// Returns an error when mutation authority is absent or either counter
    /// would make a known identity reusable.
    #[doc(hidden)]
    pub fn restore_allocator_high_water_exact(
        &self,
        next_node_id: u64,
        next_edge_id: u64,
    ) -> Result<(), String> {
        if ExclusiveBulkRestoreContext::is_read_only(self) {
            return Err("exact allocator restore requires write authority".to_owned());
        }
        // Section restore already retains this same store's exclusive proof.
        // Standalone callers must acquire it here: a shared proof would allow a
        // generated allocation between validation and the final stores, which
        // could regress a high-water and make a published identity reusable.
        let _standalone_transition = if ExclusiveBulkRestoreContext::is_active(self) {
            None
        } else {
            Some(
                self.pin_exclusive_authorized_scope()
                    .ok_or_else(|| "exact allocator restore requires write authority".to_owned())?,
            )
        };

        let derived_node_id = self.next_node_id.load(Ordering::Acquire);
        if next_node_id < derived_node_id {
            return Err(format!(
                "next node id {next_node_id} regresses derived high-water {derived_node_id}"
            ));
        }
        let derived_edge_id = self.next_edge_id.load(Ordering::Acquire);
        if next_edge_id < derived_edge_id {
            return Err(format!(
                "next edge id {next_edge_id} regresses derived high-water {derived_edge_id}"
            ));
        }

        let maximum_node_id = self.all_node_ids().into_iter().map(|id| id.as_u64()).max();
        if maximum_node_id.is_some_and(|maximum| next_node_id <= maximum) {
            return Err(format!(
                "next node id {next_node_id} must exceed maximum identity {}",
                maximum_node_id.unwrap_or_default()
            ));
        }
        let maximum_edge_id = self
            .all_known_edge_ids()
            .into_iter()
            .map(|id| id.as_u64())
            .max();
        if maximum_edge_id.is_some_and(|maximum| next_edge_id <= maximum) {
            return Err(format!(
                "next edge id {next_edge_id} must exceed maximum identity {}",
                maximum_edge_id.unwrap_or_default()
            ));
        }

        self.next_node_id.store(next_node_id, Ordering::Release);
        self.next_edge_id.store(next_edge_id, Ordering::Release);
        Ok(())
    }

    /// Removes all data from the store, resetting it to an empty state.
    ///
    /// Acquires locks in the documented ordering to prevent deadlocks.
    /// After clearing, the store behaves as if freshly constructed.
    pub fn clear(&self) {
        let Some(_scope_transition) = self.pin_exclusive_authorized_scope() else {
            return;
        };
        // Exclude receipt mint/validation for the complete replacement. The
        // fresh authority is installed before releasing this guard, so no
        // pre-clear receipt can authorize a reincarnated same-numbered edge.
        let mut transport_authority = self.transport_extract_authority.write();
        // No publication may enter after this empty boundary, and every
        // in-flight publication/removal completes before the replacement.
        let _node_identity_cut = self.node_identity_reservations.lock_when_empty();
        let _edge_identity_cut = self.edge_identity_reservations.lock_when_empty();
        // Level 1: Entity storage
        #[cfg(not(feature = "tiered-storage"))]
        {
            self.nodes.write().clear();
            self.edges.write().clear();
        }
        #[cfg(feature = "tiered-storage")]
        {
            self.node_versions.write().clear();
            self.edge_versions.write().clear();
            // Arena allocator chunks are leaked; epochs are cleared via epoch_store.
        }

        // Level 2: Catalogs
        self.label_registry.write().clear();
        {
            self.edge_type_to_id.write().clear();
            self.id_to_edge_type.write().clear();
        }

        // Level 3: Indexes
        self.label_index.write().clear();
        self.node_labels.write().clear();
        self.property_indexes.write().clear();
        #[cfg(feature = "vector-index")]
        self.vector_indexes.write().clear();
        #[cfg(feature = "text-index")]
        self.text_indexes.write().clear();

        // Nested: Properties and adjacency
        self.node_properties.clear();
        self.edge_properties.clear();
        self.forward_adj.clear();
        if let Some(ref backward) = self.backward_adj {
            backward.clear();
        }

        // Atomics: ID counters
        self.next_node_id.store(0, Ordering::Release);
        self.next_edge_id.store(0, Ordering::Release);
        self.current_epoch.store(0, Ordering::Release);
        self.retained_history_floor.store(0, Ordering::Release);

        // Level 4: Statistics
        self.live_node_count.store(0, Ordering::Release);
        self.live_edge_count.store(0, Ordering::Release);
        self.edge_type_live_counts.write().clear();
        *self.statistics.write() = Arc::new(Statistics::new());
        self.needs_stats_recompute.store(false, Ordering::Release);

        // Level 5: Undo log
        self.property_undo_log.write().clear();
        self.pending_tx_creates.write().clear();
        self.tx_property_overlay.write().clear();
        self.pending_tx_deletes.write().clear();
        self.pending_tx_edge_deletes.write().clear();
        #[cfg(feature = "text-index")]
        self.text_index_overlay.write().clear();
        self.read_trackers.write().clear();
        self.write_trackers.write().clear();
        *transport_authority = Arc::new(TransportExtractAuthority::new());
        self.transport_edges_may_be_unresolved
            .store(false, Ordering::Release);
    }

    /// Returns whether backward adjacency (incoming edge index) is available.
    ///
    /// When backward adjacency is enabled (the default), bidirectional search
    /// algorithms can traverse from the target toward the source.
    #[must_use]
    pub fn has_backward_adjacency(&self) -> bool {
        self.backward_adj.is_some()
    }

    // === Named Graph Management ===

    /// Returns a named graph by name, or `None` if it does not exist.
    #[must_use]
    pub fn graph(&self, name: &str) -> Option<Arc<LpgStore>> {
        let _maintenance = self.pin_maintenance();
        let g = self.named_graphs.read().get(name).cloned()?;
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0 && g.mutation_scope.load(Ordering::Acquire) != scope {
            return None;
        }
        Some(g)
    }

    /// Returns a named graph, creating it if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if a new store cannot be allocated or the caller
    /// lacks this sealed store's mutation authority.
    pub fn graph_or_create(&self, name: &str) -> Result<Arc<LpgStore>, AllocError> {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return Err(AllocError::InsufficientSpace);
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        #[cfg(feature = "vector-index")]
        let vector_transition = VectorIndexKind::pin_scope_transition();
        #[cfg(feature = "text-index")]
        let text_transition = InvertedIndex::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            // The historical return type predates mutation sealing and cannot
            // represent authority denial. Fail closed through its generic
            // capacity failure rather than returning a writable detached graph.
            return Err(AllocError::InsufficientSpace);
        };
        let mut graphs = self.named_graphs.write();
        if let Some(g) = graphs.get(name) {
            let scope = self.mutation_scope.load(Ordering::Acquire);
            if scope != 0 && g.mutation_scope.load(Ordering::Acquire) != scope {
                return Err(AllocError::InsufficientSpace);
            }
            return Ok(Arc::clone(g));
        }
        let child = self.allocate_named_graph()?;
        let scope = self.mutation_scope.load(Ordering::Acquire);
        let store = Arc::new(child);
        if scope != 0
            && !self.seal_child_under_topology_gate(
                &store,
                scope,
                #[cfg(feature = "vector-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &vector_transition)
                        && index.scope_is_compatible(scope, &vector_transition)
                },
                #[cfg(feature = "vector-index")]
                &|index| index.seal_with_scope_under_transition(scope, &vector_transition),
                #[cfg(feature = "text-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &text_transition)
                        && index.scope_is_compatible(scope, &text_transition)
                },
                #[cfg(feature = "text-index")]
                &|index| index.seal_with_scope_under_transition(scope, &text_transition),
            )
        {
            return Err(AllocError::InsufficientSpace);
        }
        graphs.insert(name.to_string(), Arc::clone(&store));
        Ok(store)
    }

    /// Creates a named graph. Returns `true` on success, `false` if it already exists.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the new store cannot be allocated.
    pub fn create_graph(&self, name: &str) -> Result<bool, AllocError> {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return Ok(false);
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        #[cfg(feature = "vector-index")]
        let vector_transition = VectorIndexKind::pin_scope_transition();
        #[cfg(feature = "text-index")]
        let text_transition = InvertedIndex::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return Ok(false);
        };
        let mut graphs = self.named_graphs.write();
        if graphs.contains_key(name) {
            return Ok(false);
        }
        let child = self.allocate_named_graph()?;
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0
            && !self.seal_child_under_topology_gate(
                &child,
                scope,
                #[cfg(feature = "vector-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &vector_transition)
                        && index.scope_is_compatible(scope, &vector_transition)
                },
                #[cfg(feature = "vector-index")]
                &|index| index.seal_with_scope_under_transition(scope, &vector_transition),
                #[cfg(feature = "text-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &text_transition)
                        && index.scope_is_compatible(scope, &text_transition)
                },
                #[cfg(feature = "text-index")]
                &|index| index.seal_with_scope_under_transition(scope, &text_transition),
            )
        {
            return Ok(false);
        }
        graphs.insert(name.to_string(), Arc::new(child));
        Ok(true)
    }

    /// Publishes a fully prepared named graph if the name is still absent.
    ///
    /// Transactional callers build named graphs off-map so uncommitted graph
    /// existence and contents cannot leak to other sessions. The engine calls
    /// this only while holding its publication write lock, after the commit
    /// record is durable. Returning `false` means the catalog changed since the
    /// transaction prepared and must be treated as an invariant violation.
    #[doc(hidden)]
    pub fn install_graph_if_absent(&self, name: &str, graph: Arc<LpgStore>) -> bool {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return false;
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        #[cfg(feature = "vector-index")]
        let vector_transition = VectorIndexKind::pin_scope_transition();
        #[cfg(feature = "text-index")]
        let text_transition = InvertedIndex::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        if !self.shares_graph_allocator(&graph) {
            return false;
        }
        let mut topology = FxHashSet::default();
        if !self.collect_unique_named_graph_tree(&mut topology)
            || !graph.collect_unique_named_graph_tree(&mut topology)
        {
            return false;
        }
        let mut graphs = self.named_graphs.write();
        if graphs.contains_key(name) {
            return false;
        }
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0
            && !self.seal_child_under_topology_gate(
                &graph,
                scope,
                #[cfg(feature = "vector-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &vector_transition)
                        && index.scope_is_compatible(scope, &vector_transition)
                },
                #[cfg(feature = "vector-index")]
                &|index| index.seal_with_scope_under_transition(scope, &vector_transition),
                #[cfg(feature = "text-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &text_transition)
                        && index.scope_is_compatible(scope, &text_transition)
                },
                #[cfg(feature = "text-index")]
                &|index| index.seal_with_scope_under_transition(scope, &text_transition),
            )
        {
            return false;
        }
        graphs.insert(name.to_string(), graph);
        true
    }

    /// Removes `name` only if it still names the exact expected graph.
    ///
    /// Pointer identity prevents a transaction that observed an older graph
    /// incarnation from deleting a concurrently replaced one.
    #[doc(hidden)]
    pub fn drop_graph_if_same(&self, name: &str, expected: &Arc<LpgStore>) -> bool {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return false;
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut graphs = self.named_graphs.write();
        let Some(current) = graphs.get(name) else {
            return false;
        };
        if !Arc::ptr_eq(current, expected) {
            return false;
        }
        graphs.remove(name);
        true
    }

    /// Replaces `name` only if it still names the exact expected graph.
    ///
    /// Used for transactional `DROP GRAPH; CREATE GRAPH` replacement. The
    /// comparison and replacement happen under one map write lock.
    #[doc(hidden)]
    pub fn replace_graph_if_same(
        &self,
        name: &str,
        expected: &Arc<LpgStore>,
        replacement: Arc<LpgStore>,
    ) -> bool {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return false;
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        #[cfg(feature = "vector-index")]
        let vector_transition = VectorIndexKind::pin_scope_transition();
        #[cfg(feature = "text-index")]
        let text_transition = InvertedIndex::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        if !self.shares_graph_allocator(&replacement) {
            return false;
        }
        if Arc::ptr_eq(expected, &replacement) {
            return self
                .named_graphs
                .read()
                .get(name)
                .is_some_and(|current| Arc::ptr_eq(current, expected));
        }
        let mut topology = FxHashSet::default();
        if !self.collect_unique_named_graph_tree(&mut topology)
            || !replacement.collect_unique_named_graph_tree(&mut topology)
        {
            return false;
        }
        let mut graphs = self.named_graphs.write();
        let Some(current) = graphs.get(name) else {
            return false;
        };
        if !Arc::ptr_eq(current, expected) {
            return false;
        }
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0
            && !self.seal_child_under_topology_gate(
                &replacement,
                scope,
                #[cfg(feature = "vector-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &vector_transition)
                        && index.scope_is_compatible(scope, &vector_transition)
                },
                #[cfg(feature = "vector-index")]
                &|index| index.seal_with_scope_under_transition(scope, &vector_transition),
                #[cfg(feature = "text-index")]
                &|index, owner, slot| {
                    index.is_bound_to(owner, slot, &text_transition)
                        && index.scope_is_compatible(scope, &text_transition)
                },
                #[cfg(feature = "text-index")]
                &|index| index.seal_with_scope_under_transition(scope, &text_transition),
            )
        {
            return false;
        }
        graphs.insert(name.to_string(), replacement);
        true
    }

    /// Drops a named graph. Returns `false` if it did not exist.
    pub fn drop_graph(&self, name: &str) -> bool {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return false;
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let mut graphs = self.named_graphs.write();
        graphs.remove(name).is_some()
    }

    /// Visits borrowed graph names while holding one registry read cut.
    /// Callers can admit output capacity before copying any names.
    pub fn with_graph_names<R>(
        &self,
        visit: impl FnOnce(hashbrown::hash_map::Keys<'_, String, Arc<LpgStore>>) -> R,
    ) -> R {
        let graphs = self.named_graphs.read();
        visit(graphs.keys())
    }

    /// Returns all named graph names.
    #[must_use]
    pub fn graph_names(&self) -> Vec<String> {
        self.named_graphs.read().keys().cloned().collect()
    }

    /// Takes a non-destructive snapshot of the named-graph registry.
    ///
    /// Compaction prepares a replacement default-graph topology through
    /// fallible work. Cloning the registry's `Arc`s keeps the original store
    /// untouched on error; the snapshot is installed into the fresh overlay
    /// only after the temporal merge succeeds.
    #[doc(hidden)]
    #[must_use]
    pub fn named_graph_entries(&self) -> FxHashMap<String, Arc<LpgStore>> {
        self.named_graphs.read().clone()
    }

    /// Drains the named-graph map, leaving it empty.
    ///
    /// Used by the engine's `compact()` / `recompact()` to carry named graphs
    /// across a store rebuild. Named graphs are LPG-specific and outside the
    /// `GraphStore` trait, so the columnar base cannot preserve them; the
    /// engine moves them across the pre- and post-compact overlays with this.
    #[must_use]
    pub fn take_named_graphs(&self) -> FxHashMap<String, Arc<LpgStore>> {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return FxHashMap::default();
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        let Some(_mutation) = self.pin_mutation() else {
            return FxHashMap::default();
        };
        let mut graphs = self.named_graphs.write();
        std::mem::take(&mut *graphs)
    }

    /// Replaces the named-graph map, overwriting any existing entries.
    ///
    /// Paired with [`take_named_graphs`](Self::take_named_graphs) to transfer
    /// named graphs across a compact rebuild.
    pub fn install_named_graphs(&self, graphs: FxHashMap<String, Arc<LpgStore>>) {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return;
        }
        let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
        #[cfg(feature = "vector-index")]
        let vector_transition = VectorIndexKind::pin_scope_transition();
        #[cfg(feature = "text-index")]
        let text_transition = InvertedIndex::pin_scope_transition();
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        let mut topology = FxHashSet::default();
        topology.insert(std::ptr::from_ref(self).addr());
        if graphs.values().any(|graph| {
            !self.shares_graph_allocator(graph)
                || !graph.collect_unique_named_graph_tree(&mut topology)
        }) {
            return;
        }
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0 {
            // Preflight the complete replacement forest before publishing any
            // child scope. The topology and index-transition guards retained
            // above make the successful preflight stable through the apply
            // phase, including across detached aliases held by callers.
            let mut visiting = FxHashSet::default();
            let mut checked = FxHashSet::default();
            if graphs.values().any(|graph| {
                !graph.can_seal_named_graph_tree(
                    scope,
                    &mut visiting,
                    &mut checked,
                    #[cfg(feature = "vector-index")]
                    &|index, owner, slot| {
                        index.is_bound_to(owner, slot, &vector_transition)
                            && index.scope_is_compatible(scope, &vector_transition)
                    },
                    #[cfg(feature = "text-index")]
                    &|index, owner, slot| {
                        index.is_bound_to(owner, slot, &text_transition)
                            && index.scope_is_compatible(scope, &text_transition)
                    },
                )
            }) {
                return;
            }
            let mut sealed = FxHashSet::default();
            if graphs.values().any(|graph| {
                !graph.seal_named_graph_tree_preflighted(
                    scope,
                    &mut sealed,
                    #[cfg(feature = "vector-index")]
                    &|index| index.seal_with_scope_under_transition(scope, &vector_transition),
                    #[cfg(feature = "text-index")]
                    &|index| index.seal_with_scope_under_transition(scope, &text_transition),
                )
            }) {
                debug_assert!(
                    false,
                    "stable named-graph seal preflight changed during apply"
                );
                return;
            }
        }
        let mut current = self.named_graphs.write();
        *current = graphs;
    }

    /// Returns the number of named graphs.
    #[must_use]
    pub fn graph_count(&self) -> usize {
        self.named_graphs.read().len()
    }

    /// Clears a specific graph, or the default graph if `name` is `None`.
    pub fn clear_graph(&self, name: Option<&str>) {
        if ExclusiveBulkRestoreContext::has_read_capture() {
            return;
        }
        match name {
            Some(n) => {
                let _topology = NAMED_GRAPH_TOPOLOGY_GATE.lock();
                let child = {
                    let _parent = self.pin_maintenance();
                    self.named_graphs.read().get(n).cloned()
                };
                if let Some(g) = child {
                    g.clear();
                }
            }
            None => self.clear(),
        }
    }

    /// Copies all data from the source graph to the destination graph.
    /// Creates the destination graph if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the destination store cannot be allocated.
    pub fn copy_graph(&self, source: Option<&str>, dest: Option<&str>) -> Result<(), AllocError> {
        // Self-copy guard: copying a store onto itself would iterate the
        // destination while mutating it. Both-None (default->default) or equal
        // names resolve to the same store.
        let same = match (source, dest) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        if same {
            return Ok(());
        }

        // Resolve the source store (None = this default store). A missing named
        // source has nothing to copy.
        let src_arc;
        let src: &LpgStore = match source {
            Some(name) => match self.graph(name) {
                Some(g) => {
                    src_arc = g;
                    &src_arc
                }
                None => return Ok(()),
            },
            None => self,
        };

        // Snapshot source data into owned vectors so the source is not borrowed
        // while the destination is written.
        let nodes: Vec<Node> = src.all_nodes().collect();
        let edges: Vec<Edge> = src.all_edges().collect();
        let index_keys = src.property_index_keys();

        // Resolve-or-create the destination store (None = this default store).
        let dst_arc;
        let dst: &LpgStore = match dest {
            Some(name) => {
                dst_arc = self.graph_or_create(name)?;
                &dst_arc
            }
            None => self,
        };

        let Some(_destination_mutation) = dst.pin_mutation() else {
            return Ok(());
        };

        // Copy nodes, recording an old -> new id mapping for edge endpoints.
        let mut id_map: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        for node in nodes {
            let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
            let new_id = dst.create_node_with_props(&labels, node.properties);
            if !new_id.is_valid() {
                return Err(AllocError::InsufficientSpace);
            }
            id_map.insert(node.id, new_id);
        }

        // Copy edges with remapped endpoints.
        for edge in edges {
            let (Some(&new_src), Some(&new_dst)) = (id_map.get(&edge.src), id_map.get(&edge.dst))
            else {
                continue; // endpoint not copied (should not happen for live edges)
            };
            let new_edge = dst.create_edge_with_props(
                new_src,
                new_dst,
                edge.edge_type.as_str(),
                edge.properties,
            );
            if !new_edge.is_valid() {
                return Err(AllocError::InsufficientSpace);
            }
        }

        // Re-create property indexes on the destination. Vector/text indexes are
        // not carried by the copy (they would require re-embedding / re-tokenizing
        // every value); recreate them on the destination if needed.
        for key in index_keys {
            dst.create_property_index(&key);
        }

        Ok(())
    }

    // === Internal Helpers ===

    /// Reserves the permanent binding ID for one exact encoded derived-index
    /// key. Capacity and the checked global ID are prepared before the caller
    /// binds an index; dropping an uncommitted reservation leaves no registry
    /// side effect.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    pub(super) fn prepare_index_slot(
        &self,
        key: String,
    ) -> Result<PreparedIndexSlot<'_>, AllocError> {
        let mut slots = self.index_slots.lock();
        if let Some(&id) = slots.get(&key) {
            return Ok(PreparedIndexSlot {
                slots,
                key,
                id,
                is_new: false,
            });
        }
        let id = allocate_index_binding_id(&NEXT_INDEX_SLOT_ID)?;
        slots.reserve(1);
        Ok(PreparedIndexSlot {
            slots,
            key,
            id,
            is_new: true,
        })
    }

    /// Publishes every label name needed by one unpublished representation
    /// promotion and returns their stable numeric identities.
    ///
    /// The catalog suffix remains read-visible for derived-index replay. Error
    /// or unwind drops the returned preparation and restores the exact prior
    /// catalog cardinality; the caller commits it only after routing publication.
    pub(super) fn prepare_node_labels(
        &self,
        label_versions: &[(EpochId, Vec<ArcStr>)],
    ) -> Result<PreparedNodeLabels<'_>, AllocError> {
        let publication = self.label_publication_gate.lock();
        let requested = label_versions
            .iter()
            .try_fold(0usize, |count, (_, labels)| count.checked_add(labels.len()));
        let requested = requested.ok_or(AllocError::InsufficientSpace)?;

        let mut ids = FxHashMap::default();
        ids.try_reserve(requested)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let mut missing = Vec::new();
        missing
            .try_reserve(requested)
            .map_err(|_| AllocError::InsufficientSpace)?;

        let mut registry = self.label_registry.write();
        let original_registry_len = registry.len();
        for (_, labels) in label_versions {
            for label in labels {
                if ids.contains_key(label.as_str()) {
                    continue;
                }
                if let Some(id) = registry.get_id(label.as_str()) {
                    ids.insert(label.clone(), id);
                } else {
                    // `u32::MAX` is outside the checked publishable label-ID
                    // range and deduplicates missing names across versions.
                    ids.insert(label.clone(), u32::MAX);
                    missing.push(label.clone());
                }
            }
        }

        registry
            .name_to_id
            .try_reserve(missing.len())
            .map_err(|_| AllocError::InsufficientSpace)?;
        registry
            .id_to_name
            .try_reserve(missing.len())
            .map_err(|_| AllocError::InsufficientSpace)?;
        let published_registry_len = original_registry_len
            .checked_add(missing.len())
            .ok_or(AllocError::InsufficientSpace)?;
        if u32::try_from(published_registry_len).is_err() {
            return Err(AllocError::InsufficientSpace);
        }

        let mut label_index = self.label_index.write();
        let original_label_index_len = label_index.len();
        let additional_slots = published_registry_len.saturating_sub(original_label_index_len);
        label_index
            .try_reserve(additional_slots)
            .map_err(|_| AllocError::InsufficientSpace)?;

        for label in missing {
            let id = u32::try_from(registry.id_to_name.len())
                .map_err(|_| AllocError::InsufficientSpace)?;
            let prior = registry.name_to_id.insert(label.clone(), id);
            debug_assert!(prior.is_none(), "prepared label suffix remains vacant");
            registry.id_to_name.push(label.clone());
            ids.insert(label, id);
        }
        label_index.resize_with(published_registry_len, FxHashMap::default);
        drop(label_index);
        drop(registry);

        Ok(PreparedNodeLabels {
            store: self,
            _publication: publication,
            ids,
            original_registry_len,
            original_label_index_len,
            armed: true,
        })
    }

    pub(super) fn get_or_create_label_id(&self, label: &str) -> u32 {
        let _publication = self.label_publication_gate.lock();
        if let Some(id) = self.label_registry.read().get_id(label) {
            return id;
        }
        self.label_registry.write().get_or_create(label)
    }

    /// Reserves a stable edge-type ID and all capacity needed to publish it,
    /// but does not make a new catalog row visible. The returned reservation
    /// may be retained across fallible arena preparation; every catalog writer
    /// uses this same gate.
    fn prepare_edge_type(&self, edge_type: &str) -> Result<PreparedEdgeType<'_>, AllocError> {
        let publication = self.edge_type_publication_gate.lock();
        let name: ArcStr = edge_type.into();
        if let Some(&id) = self.edge_type_to_id.read().get(&name) {
            return Ok(PreparedEdgeType {
                _publication: publication,
                name,
                id,
                is_new: false,
            });
        }

        let mut type_to_id = self.edge_type_to_id.write();
        let mut id_to_type = self.id_to_edge_type.write();
        let mut counts = self.edge_type_live_counts.write();
        // No other type publisher can consume this capacity or candidate ID
        // before the reservation commits.
        type_to_id
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        id_to_type
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        counts
            .try_reserve(1)
            .map_err(|_| AllocError::InsufficientSpace)?;
        let id = u32::try_from(id_to_type.len()).map_err(|_| AllocError::InsufficientSpace)?;
        drop(counts);
        drop(id_to_type);
        drop(type_to_id);
        Ok(PreparedEdgeType {
            _publication: publication,
            name,
            id,
            is_new: true,
        })
    }

    /// Publishes a reversible edge-type suffix for an unpublished compact
    /// representation promotion. Derived replay may read the row immediately;
    /// dropping the returned token restores a newly appended zero-count suffix.
    #[cfg(feature = "compact-store")]
    pub(super) fn prepare_edge_type_for_promotion(
        &self,
        edge_type: &str,
    ) -> Result<PreparedPromotionEdgeType<'_>, AllocError> {
        let PreparedEdgeType {
            _publication,
            name,
            id,
            is_new,
        } = self.prepare_edge_type(edge_type)?;
        if is_new {
            let mut type_to_id = self.edge_type_to_id.write();
            let mut id_to_type = self.id_to_edge_type.write();
            let mut counts = self.edge_type_live_counts.write();
            debug_assert_eq!(id_to_type.len(), id as usize);
            debug_assert!(!type_to_id.contains_key(&name));
            type_to_id.insert(name.clone(), id);
            id_to_type.push(name.clone());
            counts.push(0);
        }
        Ok(PreparedPromotionEdgeType {
            store: self,
            _publication,
            name,
            id,
            is_new,
            armed: true,
        })
    }

    /// Batch form of [`Self::prepare_edge_type`]. Duplicate names share one
    /// candidate ID and all new catalog capacity is reserved before return.
    fn prepare_edge_types(&self, edge_types: &[&str]) -> Result<PreparedEdgeTypes<'_>, AllocError> {
        let publication = self.edge_type_publication_gate.lock();
        let type_to_id = self.edge_type_to_id.read();
        let id_to_type = self.id_to_edge_type.read();
        let mut ids = FxHashMap::default();
        ids.try_reserve(edge_types.len())
            .map_err(|_| AllocError::InsufficientSpace)?;
        let mut new_types = Vec::new();
        new_types
            .try_reserve(edge_types.len())
            .map_err(|_| AllocError::InsufficientSpace)?;
        for &edge_type in edge_types {
            if ids.contains_key(edge_type) {
                continue;
            }
            let name: ArcStr = edge_type.into();
            let id = if let Some(&id) = type_to_id.get(&name) {
                id
            } else {
                let offset =
                    u32::try_from(new_types.len()).map_err(|_| AllocError::InsufficientSpace)?;
                let base =
                    u32::try_from(id_to_type.len()).map_err(|_| AllocError::InsufficientSpace)?;
                let id = base
                    .checked_add(offset)
                    .ok_or(AllocError::InsufficientSpace)?;
                new_types.push((name.clone(), id));
                id
            };
            ids.insert(name, id);
        }
        drop(id_to_type);
        drop(type_to_id);
        if !new_types.is_empty() {
            self.edge_type_to_id
                .write()
                .try_reserve(new_types.len())
                .map_err(|_| AllocError::InsufficientSpace)?;
            self.id_to_edge_type
                .write()
                .try_reserve(new_types.len())
                .map_err(|_| AllocError::InsufficientSpace)?;
            self.edge_type_live_counts
                .write()
                .try_reserve(new_types.len())
                .map_err(|_| AllocError::InsufficientSpace)?;
        }
        Ok(PreparedEdgeTypes {
            _publication: publication,
            ids,
            new_types,
        })
    }

    /// Increments the live edge count for a given edge type.
    pub(super) fn increment_edge_type_count(&self, type_id: u32) {
        let mut counts = self.edge_type_live_counts.write();
        if counts.len() <= type_id as usize {
            counts.resize(type_id as usize + 1, 0);
        }
        counts[type_id as usize] += 1;
    }

    /// Decrements the live edge count for a given edge type.
    pub(super) fn decrement_edge_type_count(&self, type_id: u32) {
        let mut counts = self.edge_type_live_counts.write();
        // reason: counts.len() is bounded by edge type registry size, fits u32
        #[allow(clippy::cast_possible_truncation)]
        if type_id < counts.len() as u32 {
            counts[type_id as usize] -= 1;
        }
    }

    /// Records a node created with a PENDING version under a transaction, for
    /// write-set-scoped commit/rollback. No-op for the system transaction
    /// (non-transactional creates are immediately visible, never PENDING).
    pub(super) fn record_pending_node(&self, transaction_id: TransactionId, id: NodeId) {
        if transaction_id != TransactionId::SYSTEM {
            self.pending_tx_creates
                .write()
                .entry(transaction_id)
                .or_default()
                .0
                .push(id);
        }
    }

    /// Records an edge created with a PENDING version under a transaction. See
    /// [`record_pending_node`](Self::record_pending_node).
    pub(super) fn record_pending_edge(&self, transaction_id: TransactionId, id: EdgeId) {
        if transaction_id != TransactionId::SYSTEM {
            self.pending_tx_creates
                .write()
                .entry(transaction_id)
                .or_default()
                .1
                .push(id);
        }
    }

    /// Takes (removes and returns) the pending-create lists for a transaction.
    /// Called by write-set-scoped commit/rollback, which then finalizes or
    /// discards exactly these entities and leaves the map empty for the tx.
    #[doc(hidden)]
    pub fn take_pending_creates(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<NodeId>, Vec<EdgeId>) {
        let Some(_mutation) = self.pin_mutation() else {
            return (Vec::new(), Vec::new());
        };
        self.pending_tx_creates
            .write()
            .remove(&transaction_id)
            .unwrap_or_default()
    }

    /// Non-draining snapshot of the node ids this transaction has created with a
    /// PENDING version (from `pending_tx_creates`). Used by MERGE to treat
    /// same-tx-created nodes as match candidates (read-your-writes). Returns empty
    /// for the system transaction (its creates are immediately visible / committed).
    pub fn pending_node_creates(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.pending_tx_creates
            .read()
            .get(&transaction_id)
            .map(|(nodes, _edges)| nodes.clone())
            .unwrap_or_default()
    }

    /// Non-draining snapshot of the edge ids this transaction has created with a
    /// PENDING version (from `pending_tx_creates`). Mirrors [`pending_node_creates`](Self::pending_node_creates)
    /// for edges. Returns empty for the system transaction.
    pub fn pending_edge_creates(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        self.pending_tx_creates
            .read()
            .get(&transaction_id)
            .map(|(_nodes, edges)| edges.clone())
            .unwrap_or_default()
    }

    /// Non-draining snapshot of node ids this transaction has queued for deletion
    /// (from `pending_tx_deletes`). Does NOT consume the list — the existing
    /// [`take_pending_deletes`](Self::take_pending_deletes) still needs it.
    /// Returns empty if the transaction has no pending node deletes.
    pub fn pending_node_deletes_peek(&self, transaction_id: TransactionId) -> Vec<NodeId> {
        self.pending_tx_deletes
            .read()
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Non-draining snapshot of edge ids this transaction has queued for deletion
    /// (from `pending_tx_edge_deletes`). Does NOT consume the list — the existing
    /// [`take_pending_edge_deletes`](Self::take_pending_edge_deletes) still needs it.
    /// Returns only the `EdgeId` component of each `(src, edge, dst)` triple.
    /// Returns empty if the transaction has no pending edge deletes.
    pub fn pending_edge_deletes_peek(&self, transaction_id: TransactionId) -> Vec<EdgeId> {
        self.pending_tx_edge_deletes
            .read()
            .get(&transaction_id)
            .map(|v| v.iter().map(|(_src, eid, _dst)| *eid).collect())
            .unwrap_or_default()
    }

    /// Non-draining snapshot of all entities this transaction has touched via
    /// the property/label overlay (`tx_property_overlay`). Returns the unique
    /// node ids from `node_props` keys ∪ `node_labels` keys, and the unique
    /// edge ids from `edge_props` keys. Both lists are deduplicated.
    /// Returns `(vec![], vec![])` if the transaction has no overlay delta.
    pub fn overlay_touched_entities(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<NodeId>, Vec<EdgeId>) {
        let overlay = self.tx_property_overlay.read();
        match overlay.get(&transaction_id) {
            None => (Vec::new(), Vec::new()),
            Some(delta) => {
                // Collect unique node ids from node_props and node_labels keys.
                let mut node_set: FxHashSet<NodeId> = FxHashSet::default();
                for (node_id, _key) in delta.node_props.keys() {
                    node_set.insert(*node_id);
                }
                for (node_id, _label_id) in delta.node_labels.keys() {
                    node_set.insert(*node_id);
                }
                // Collect unique edge ids from edge_props keys.
                let mut edge_set: FxHashSet<EdgeId> = FxHashSet::default();
                for (edge_id, _key) in delta.edge_props.keys() {
                    edge_set.insert(*edge_id);
                }
                (
                    node_set.into_iter().collect(),
                    edge_set.into_iter().collect(),
                )
            }
        }
    }

    /// Non-draining snapshot of property-level writes from the overlay, for use
    /// under property-granularity conflict detection.
    ///
    /// Returns:
    /// - `Vec<(NodeId, Option<String>)>`: node-property writes as `(node, Some(key))`;
    ///   label changes (structural) as `(node, None)`.
    /// - `Vec<(EdgeId, Option<String>)>`: edge-property writes as `(edge, Some(key))`.
    ///
    /// Unlike [`overlay_touched_entities`](Self::overlay_touched_entities) (which
    /// deduplicates to unique entity ids), this method returns one entry *per
    /// `(entity, property)`* write so the caller can build tagged write-set entries.
    /// Duplicates within the same `(entity, key)` are naturally collapsed by the
    /// `HashSet<(EntityId, PropTag)>` write-set.
    pub fn overlay_touched_properties(
        &self,
        transaction_id: TransactionId,
    ) -> (Vec<(NodeId, Option<String>)>, Vec<(EdgeId, Option<String>)>) {
        let overlay = self.tx_property_overlay.read();
        match overlay.get(&transaction_id) {
            None => (Vec::new(), Vec::new()),
            Some(delta) => {
                // Each (node, property_key) write → (node, Some(key)).
                let mut node_props: Vec<(NodeId, Option<String>)> = delta
                    .node_props
                    .keys()
                    .map(|(node_id, key)| (*node_id, Some(key.to_string())))
                    .collect();
                // Label changes are structural: record as (node, None).
                for (node_id, _label_id) in delta.node_labels.keys() {
                    node_props.push((*node_id, None));
                }
                // Edge-property writes → (edge, Some(key)).
                let edge_props: Vec<(EdgeId, Option<String>)> = delta
                    .edge_props
                    .keys()
                    .map(|(edge_id, key)| (*edge_id, Some(key.to_string())))
                    .collect();
                (node_props, edge_props)
            }
        }
    }

    /// Attaches `tracker` to `tx` so that subsequent visible-read accessors
    /// record every observed node/edge into it (the SSI read-set). Call at
    /// Serializable tx begin; the tracker is held until
    /// [`unregister_read_tracker`](Self::unregister_read_tracker) is called at
    /// commit/rollback. No-op for non-Serializable transactions (just don't call it).
    pub fn register_read_tracker(&self, tx: TransactionId, tracker: SharedReadTracker) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.read_trackers.write().insert(tx, tracker);
    }

    /// Removes the read tracker for `tx`, dropping the Arc. Call at
    /// commit/rollback for Serializable transactions. Silent no-op if none was
    /// registered.
    pub fn unregister_read_tracker(&self, tx: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.read_trackers.write().remove(&tx);
    }

    /// Records that `tx` observed `id` as a node read. Called by the store's
    /// visible-read accessors at the point visibility is confirmed.
    /// Silent no-op when no tracker is registered for `tx`
    /// (SI/ReadCommitted, or no Serializable tx active).
    #[inline]
    pub(crate) fn record_read_node(&self, tx: TransactionId, id: NodeId) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_node_read(tx, id);
        }
    }

    /// Records that `tx` observed `id` as an edge read. Mirrors
    /// [`record_read_node`](Self::record_read_node) for edges.
    #[inline]
    pub(crate) fn record_read_edge(&self, tx: TransactionId, id: EdgeId) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_edge_read(tx, id);
        }
    }

    /// Records the complete label predicate before a label scan enumerates
    /// rows. The logical name key protects absent and not-yet-interned labels;
    /// an existing numeric key is retained for fine-read escalation.
    #[inline]
    pub(crate) fn record_label_predicate_read(&self, tx: TransactionId, label: &str) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_label_name_predicate_read(tx, label);
            if let Some(label_id) = self.label_id(label) {
                t.record_label_predicate_read(tx, LabelId::new(label_id));
            }
        }
    }

    /// Records the complete relationship-type predicate before a typed
    /// traversal enumerates edges. The logical name key protects absent types;
    /// an existing numeric key is retained for fine-read escalation.
    #[inline]
    pub(crate) fn record_rel_type_predicate_read(&self, tx: TransactionId, rel_type: &str) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_rel_type_name_predicate_read(tx, rel_type);
            if let Some(&type_id) = self.edge_type_to_id.read().get(rel_type) {
                t.record_rel_type_predicate_read(tx, EdgeTypeId::new(type_id));
            }
        }
    }

    /// Records an unqualified structural scan of the LPG dataset.
    #[inline]
    pub(crate) fn record_lpg_dataset_read(&self, tx: TransactionId) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_lpg_dataset_read(tx);
        }
    }

    /// Records that `tx` read property `key` of node `id`.
    ///
    /// Delegates to [`ReadTracker::record_node_property_read`]; the default
    /// implementation drops the key and falls back to entity-level recording,
    /// so behaviour is identical to [`record_read_node`](Self::record_read_node)
    /// until an override is registered.
    #[inline]
    #[allow(dead_code)] // SSI property-granularity hook; not reached in native profile
    pub(crate) fn record_read_node_property(&self, tx: TransactionId, id: NodeId, key: &str) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_node_property_read(tx, id, key);
        }
    }

    /// Records that `tx` read property `key` of edge `id`.
    ///
    /// Mirrors [`record_read_node_property`](Self::record_read_node_property) for edges.
    #[inline]
    #[allow(dead_code)] // SSI property-granularity hook; not reached in native profile
    pub(crate) fn record_read_edge_property(&self, tx: TransactionId, id: EdgeId, key: &str) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_edge_property_read(tx, id, key);
        }
    }

    /// Records that `tx` read `id` while scanning label `label_id` (e.g. from
    /// `MATCH (n:L)`). Delegates to
    /// [`ReadTracker::record_read_node_in_label`]; the default implementation
    /// falls back to entity-level recording, so behaviour is identical to
    /// [`record_read_node`](Self::record_read_node) for non-engine trackers.
    ///
    /// The engine override routes through the manager's `record_read_in_label`
    /// so fine `Node` reads under a single label predicate escalate to the
    /// coarse `EntityId::Label(L)` key once the threshold is exceeded (GE3).
    #[inline]
    pub(crate) fn record_read_node_in_label(
        &self,
        tx: TransactionId,
        id: NodeId,
        label_id: LabelId,
    ) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_read_node_in_label(tx, id, label_id);
        }
    }

    /// Records that `tx` read `id` while scanning relationship type `rel_type`
    /// (e.g. from `MATCH ()-[:T]->()`). Symmetric with
    /// [`record_read_node_in_label`](Self::record_read_node_in_label).
    ///
    /// Delegates to [`ReadTracker::record_read_edge_in_rel_type`]; the engine
    /// override escalates fine `Edge` reads to `EntityId::RelType(T)` at
    /// threshold (GE3).
    ///
    /// Called from `is_edge_visible_versioned` and `get_edge_versioned` (both
    /// cfg variants) so every edge-read path escalates by intrinsic type.
    #[inline]
    pub(crate) fn record_read_edge_in_rel_type(
        &self,
        tx: TransactionId,
        id: EdgeId,
        rel_type: EdgeTypeId,
    ) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_read_edge_in_rel_type(tx, id, rel_type);
        }
    }

    /// Records that `tx` materialized node `id` (e.g. `RETURN n`), routing
    /// through the label-carrying tracker method so an already-escalated label
    /// short-circuits instead of re-adding a fine `(Node(n), None)` entry.
    ///
    /// Skips the `committed_node_label_ids` lookup when no tracker is registered
    /// for `tx` (SI/ReadCommitted pay nothing — mirrors the pattern of
    /// [`record_read_node_property_escalating`](Self::record_read_node_property_escalating)).
    #[inline]
    pub(crate) fn record_read_node_materialized(&self, tx: TransactionId, id: NodeId) {
        let Some(t) = self.read_trackers.read().get(&tx).cloned() else {
            return;
        };
        let labels = self.committed_node_label_ids(id);
        t.record_node_read_in_labels(tx, id, &labels);
    }

    /// Records that `tx` read property `key` of node `id`, routing through the
    /// label-set-carrying tracker method for read-set escalation.
    ///
    /// Avoids the `committed_node_label_ids` lookup when no tracker is registered
    /// for `tx` (SI/ReadCommitted pay nothing).  When a tracker is present the
    /// node's committed labels are fetched without read-recording and forwarded
    /// to [`ReadTracker::record_node_property_read_in_labels`], which the engine
    /// bridge uses to route the read under each label the transaction has already
    /// scanned (intersection).
    #[inline]
    pub(crate) fn record_read_node_property_escalating(
        &self,
        tx: TransactionId,
        id: NodeId,
        key: &str,
    ) {
        let Some(t) = self.read_trackers.read().get(&tx).cloned() else {
            return;
        };
        let labels = self.committed_node_label_ids(id);
        t.record_node_property_read_in_labels(tx, id, key, &labels);
    }

    /// Records that `tx` read property `key` of edge `id`, routing through the
    /// rel-type-carrying tracker method for read-set escalation.
    ///
    /// Mirrors [`record_read_node_property_escalating`](Self::record_read_node_property_escalating)
    /// for edges: fetches the committed edge type (without read-recording) and
    /// forwards to [`ReadTracker::record_edge_property_read_in_rel`].
    #[inline]
    pub(crate) fn record_read_edge_property_escalating(
        &self,
        tx: TransactionId,
        id: EdgeId,
        key: &str,
    ) {
        let Some(t) = self.read_trackers.read().get(&tx).cloned() else {
            return;
        };
        let rel = self.committed_edge_type_id(id);
        t.record_edge_property_read_in_rel(tx, id, key, rel);
    }

    // ── Write-tracker: per-transaction registration for indexed-SET anti-phantom ──

    /// Attaches `tracker` to `tx` so that subsequent buffered indexed-property
    /// writes record the index write via [`crate::execution::operators::WriteTracker::record_index_write`].
    /// Call at Serializable tx begin alongside
    /// [`register_read_tracker`](Self::register_read_tracker).
    /// No-op for SI/ReadCommitted (just don't call it).
    pub fn register_write_tracker(&self, tx: TransactionId, tracker: SharedWriteTracker) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.write_trackers.write().insert(tx, tracker);
    }

    /// Removes the write tracker for `tx`. Call at commit/rollback for
    /// Serializable transactions. Silent no-op if none was registered.
    pub fn unregister_write_tracker(&self, tx: TransactionId) {
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.write_trackers.write().remove(&tx);
    }

    /// Records that `tx` executed an index search on `index_key` (`"label:property"`).
    ///
    /// Forwards to the registered [`ReadTracker::record_index_read`] if one is
    /// present for `tx`. Silent no-op for SI/ReadCommitted (no tracker registered).
    /// Used for both text and vector indexes.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    #[inline]
    pub(crate) fn record_read_index(&self, tx: TransactionId, index_key: &str) {
        if let Some(t) = self.read_trackers.read().get(&tx).cloned() {
            t.record_index_read(tx, index_key);
        }
    }

    /// Records that `tx` wrote to the index identified by `index_key`
    /// (`"label:property"` format). Forwards to the registered
    /// [`crate::execution::operators::WriteTracker::record_index_write`] if present.
    /// Silent no-op for SI/ReadCommitted. Used for both text and vector indexes.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    #[inline]
    pub(crate) fn record_write_index(&self, tx: TransactionId, index_key: &str) {
        if let Some(t) = self.write_trackers.read().get(&tx).cloned() {
            t.record_index_write(tx, index_key);
        }
    }

    /// Records a node write with coarse `Label(L)` fan-out for each label.
    ///
    /// Forwards to the registered write tracker's
    /// [`record_node_write_with_labels`](crate::execution::operators::WriteTracker::record_node_write_with_labels).
    /// Silent no-op when no write tracker is registered for `tx`
    /// (SI/ReadCommitted, or the system transaction).
    ///
    /// This is the phantom-write chokepoint for `create_node_versioned` and
    /// `add_label_buffered`: every node create or label-add under a Serializable
    /// transaction must record the coarse `Label(L)` write so a concurrent
    /// escalated `Label(L)` reader forms an rw-antidependency.
    ///
    /// Errors are silently ignored: if the write tracker returns a conflict
    /// the conflict will also be caught at the node-entity level by the
    /// operator-level `record_node_write` call that precedes this one.
    /// The phantom Label/RelType write is a supplementary coarse record —
    /// it must not double-abort an already-doomed transaction.
    #[inline]
    pub(crate) fn record_coarse_node_write(
        &self,
        tx: TransactionId,
        node_id: NodeId,
        labels: &[LabelId],
    ) {
        if tx == TransactionId::SYSTEM {
            return;
        }
        if let Some(t) = self.write_trackers.read().get(&tx).cloned() {
            // Ignore conflict result: phantom coarse write; entity-level W-W
            // is checked by the operator before mutating the store.
            let _ = t.record_node_write_with_labels(tx, node_id, labels);
            let names: Vec<_> = {
                let registry = self.label_registry.read();
                labels
                    .iter()
                    .filter_map(|label| registry.get_name(label.as_u32()).cloned())
                    .collect()
            };
            for name in &names {
                let _ = t.record_label_name_predicate_write(tx, name);
            }
        }
    }

    /// Records an edge write with coarse `RelType(T)` fan-out.
    ///
    /// Forwards to the registered write tracker's
    /// [`record_edge_write_with_type`](crate::execution::operators::WriteTracker::record_edge_write_with_type).
    /// Silent no-op when no write tracker is registered for `tx`.
    ///
    /// Errors are silently ignored (same rationale as
    /// [`record_coarse_node_write`](Self::record_coarse_node_write)).
    #[inline]
    pub(crate) fn record_coarse_edge_write(
        &self,
        tx: TransactionId,
        edge_id: EdgeId,
        rel_type: EdgeTypeId,
    ) {
        if tx == TransactionId::SYSTEM {
            return;
        }
        if let Some(t) = self.write_trackers.read().get(&tx).cloned() {
            let _ = t.record_edge_write_with_type(tx, edge_id, rel_type);
            let name = self
                .id_to_edge_type
                .read()
                .get(rel_type.as_u32() as usize)
                .cloned();
            if let Some(name) = name {
                let _ = t.record_rel_type_name_predicate_write(tx, &name);
            }
        }
    }

    /// Records dataset and logical label-name predicate writes without a fine
    /// node write or fabricated overlay-local numeric label IDs.
    ///
    /// The compact layered store uses this only when it tombstones a base-only
    /// node that the overlay did not contain. Normal overlay mutations must use
    /// [`record_coarse_node_write`](Self::record_coarse_node_write), which also
    /// carries the fine entity and numeric label fan-out.
    #[cfg(feature = "compact-store")]
    pub(crate) fn record_coarse_node_predicates_by_name(&self, tx: TransactionId, labels: &[&str]) {
        if tx == TransactionId::SYSTEM {
            return;
        }
        if let Some(t) = self.write_trackers.read().get(&tx).cloned() {
            let _ = t.record_lpg_dataset_write(tx);
            for label in labels {
                let _ = t.record_label_name_predicate_write(tx, label);
            }
        }
    }

    /// Records dataset and logical relationship-type predicate writes without
    /// a fine edge write or fabricated overlay-local numeric type ID.
    ///
    /// Compact base-only edge tombstones use this when the overlay deletion
    /// path did not run. Normal overlay mutations use
    /// [`record_coarse_edge_write`](Self::record_coarse_edge_write).
    #[cfg(feature = "compact-store")]
    pub(crate) fn record_coarse_edge_predicates_by_name(&self, tx: TransactionId, rel_type: &str) {
        if tx == TransactionId::SYSTEM {
            return;
        }
        if let Some(t) = self.write_trackers.read().get(&tx).cloned() {
            let _ = t.record_lpg_dataset_write(tx);
            let _ = t.record_rel_type_name_predicate_write(tx, rel_type);
        }
    }

    /// Fans out **only** the coarse `Label(L)` phantom write(s) for a node — NO
    /// fine `Node` entity write.
    ///
    /// Used by the transactional **property-write** paths
    /// (`set_node_property_buffered` / `remove_node_property_buffered`): the fine
    /// `Node(n)` write is recorded elsewhere (operator under its property tag, or
    /// commit-time write-set completion), so this records only the coarse guard.
    /// Re-recording the fine `Node` write here (as
    /// [`record_coarse_node_write`](Self::record_coarse_node_write) does for
    /// structural ops) would inject a `None`-tagged wildcard `Node` write that
    /// breaks Property-granularity disjointness — see
    /// [`record_node_labels_write`](crate::execution::operators::WriteTracker::record_node_labels_write).
    ///
    /// `key` is the written property name; forwarded to the tracker so it can
    /// compute `Some(prop_tag(key))` for the coarse write's tag (Part-G knob).
    ///
    /// Silent no-op for SYSTEM and when no write tracker is registered (SI/RC).
    #[inline]
    pub(crate) fn record_coarse_node_labels_only(
        &self,
        tx: TransactionId,
        labels: &[LabelId],
        key: &str,
    ) {
        if tx == TransactionId::SYSTEM {
            return;
        }
        if let Some(t) = self.write_trackers.read().get(&tx).cloned() {
            let _ = t.record_node_labels_write(tx, labels, key);
        }
    }

    /// Fans out **only** the coarse `RelType(T)` phantom write for an edge — NO
    /// fine `Edge` entity write. Edge mirror of
    /// [`record_coarse_node_labels_only`](Self::record_coarse_node_labels_only),
    /// used by the edge property-write paths. `key` is forwarded for the
    /// Part-G disjoint-property knob.
    ///
    /// Silent no-op for SYSTEM and when no write tracker is registered (SI/RC).
    #[inline]
    pub(crate) fn record_coarse_edge_type_only(
        &self,
        tx: TransactionId,
        rel_type: EdgeTypeId,
        key: &str,
    ) {
        if tx == TransactionId::SYSTEM {
            return;
        }
        if let Some(t) = self.write_trackers.read().get(&tx).cloned() {
            let _ = t.record_edge_type_write(tx, rel_type, key);
        }
    }
}
