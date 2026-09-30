//! Prepared, allocation-complete publication of physical index registries.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::index_registration::{ObservedIndex, PropertyIndexRows, RegisteredIndex};
use super::{DataRebindError, IndexRegistrationObservation, LpgStore, PinnedLpgTransition};
mod authority;
use authority::{BindingFences, RegistryAuthority, RegistryAuthorityWorkspace};
mod commit;
mod replacement;
pub use commit::{
    InstalledLpgCommit, LpgCommitWorkspace, PreparedLpgCommit, ReleasedLpgCommit, StoreCommitInput,
    with_prepared_lpg_commit,
};
#[cfg(feature = "vector-index")]
pub use commit::{VectorCommitChanges, VectorCommitInput};
pub use replacement::with_prepared_lpg_replacement;
mod property_maintenance;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use crate::graph::lpg::encode_index_key;
#[cfg(feature = "text-index")]
use crate::index::text::{
    InvertedIndex, RegisteredTextIndex, TextCommitScope, TextCommitWorkspace,
    TextRegistryBatchFence, TextRegistryFenceWorkspace,
};
#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;
use grafeo_common::memory::arena::AllocError;
use grafeo_common::types::EpochId;
#[cfg(feature = "text-index")]
use grafeo_common::types::TransactionId;
use grafeo_common::types::{HashableValue, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use parking_lot::MutexGuard;
use parking_lot::RwLockWriteGuard;
use property_maintenance::PropertyMaintenanceWorkspace;
use std::sync::Arc;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use std::sync::atomic::Ordering;

/// Store-local physical index identity; this is not a durable key grammar.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IndexRegistryKey {
    /// Property/BTree membership spans all labels.
    Property(PropertyKey),
    /// Full-text membership for one label and property.
    #[cfg(feature = "text-index")]
    Text {
        /// Exact label name.
        label: String,
        /// Exact property name.
        property: String,
    },
    /// Vector membership for one label and property.
    #[cfg(feature = "vector-index")]
    Vector {
        /// Exact label name.
        label: String,
        /// Exact property name.
        property: String,
    },
}

/// Owned unpublished contents. The caller supplies the correct row/commit view.
pub enum IndexRegistryContents {
    /// Rows from which private Property memberships are prepared.
    Property(Vec<(NodeId, Value)>),
    /// Complete retained property history and its current compatibility view.
    PropertyHistory(super::PropertyIndexImage),
    /// A concrete populated Text index, never a moved forwarding shell.
    #[cfg(feature = "text-index")]
    Text(InvertedIndex),
    /// A populated Vector index with its exact configuration.
    #[cfg(feature = "vector-index")]
    Vector(VectorIndexKind),
}

/// Final-row changes for an existing registration, never a replacement handle.
pub enum IndexRegistryMaintenance {
    /// Unique node identities with committed-old and final-new property values.
    /// `None` and `Value::Null` mean absence; unchanged pairs perform no work.
    /// The caller supplies the qualified commit view, not an operation log.
    Property(Vec<(NodeId, Option<Value>, Option<Value>)>),
    /// Property changes tied to the actual reserved publication epoch.
    PropertyAt {
        /// Normalized committed-before and final-after values.
        rows: Vec<(NodeId, Option<Value>, Option<Value>)>,
        /// Reserved epoch installed with the paired data postimage.
        commit_epoch: EpochId,
    },
    /// Unique node identities and final text (`None` removes the document).
    /// Existing tokenizer/configuration and histories survive unchanged; new
    /// entries become visible at `commit_epoch`, not at the initial epoch.
    #[cfg(feature = "text-index")]
    Text {
        /// Qualified final rows, not intermediate operation-log steps.
        rows: Vec<(NodeId, Option<String>)>,
        /// Committed preparation frontier P.
        frontier: EpochId,
        /// Reserved publication epoch C, strictly after P.
        commit_epoch: EpochId,
        /// Exact transaction whose normalized rows are supplied.
        transaction_id: TransactionId,
    },
    /// Exact recorded sparse changes for unpublished WAL recovery. Validation
    /// and installation reuse the ordinary Text commit workspace.
    #[cfg(feature = "text-index")]
    TextRecorded {
        /// Bounded current Text postimage bytes.
        payload: Vec<u8>,
        /// Recorded committed preparation frontier.
        frontier: EpochId,
        /// Owning transaction's committed epoch.
        commit_epoch: EpochId,
        /// Owning transaction's identity.
        transaction_id: TransactionId,
    },
    /// Complete final Text population. Missing current documents are removed
    /// at C; tokenizer, configuration, registration and history remain.
    #[cfg(feature = "text-index")]
    TextRebuild {
        /// Canonical owner's exact BM25 configuration.
        config: crate::index::text::BM25Config,
        /// Canonical owner's exact Simple tokenizer minimum length.
        min_token_length: usize,
        /// Every final document, unique by NodeId.
        rows: Vec<(NodeId, Option<String>)>,
        /// Committed preparation frontier P.
        frontier: EpochId,
        /// Reserved publication epoch C.
        commit_epoch: EpochId,
        /// Exact preparing transaction.
        transaction_id: TransactionId,
    },
}

/// One normalized change to a store's physical index registry.
pub enum IndexRegistryEdit {
    /// Create a vacant physical key.
    Create {
        /// The required vacant physical key.
        key: IndexRegistryKey,
        /// Private populated contents matching the key's family.
        contents: IndexRegistryContents,
    },
    /// Remove exactly an observed registration.
    Drop {
        /// Exact previously observed registration, not write permission.
        expected: IndexRegistrationObservation,
    },
    /// Replace exactly an observed registration with a new concrete target.
    Replace {
        /// Exact previously observed registration, not write permission.
        expected: IndexRegistrationObservation,
        /// Private replacement contents matching the existing family.
        contents: IndexRegistryContents,
    },
    /// Update exact surviving memberships while preserving registration, Arc
    /// identity, configuration and binding slot.
    Maintain {
        /// Exact previously observed registration, not write permission.
        expected: IndexRegistrationObservation,
        /// Final-row membership changes matching the observed family.
        changes: IndexRegistryMaintenance,
    },
}

/// A borrowed physical store and its normalized registry changes.
pub struct StoreIndexEdits<'store> {
    /// Exact target store; observations must belong to this representation.
    pub store: &'store LpgStore,
    /// At most one change per physical key.
    pub edits: Vec<IndexRegistryEdit>,
}

/// Owns every private and displaced payload outside publication guard scopes.
///
/// Construct this workspace before acquiring enclosing lifecycle, publication
/// or catalog guards. Preparation borrows it; releasing a prepared or installed
/// proof releases only guards. Drop the workspace after all enclosing guards.
pub struct IndexRegistryWorkspace<'store> {
    // Release registry writers before authority even if a borrowed batch was
    // explicitly forgotten. Both owners retain their allocated buffers.
    registry: RegistryWorkspace<'store>,
    authority: RegistryAuthorityWorkspace<'store>,
}

struct RegistryWorkspace<'store> {
    // Guard buffers, like candidate/retired maps, outlive each fence loan.
    active: Vec<StorePublication<'store>>,
    stores: Vec<PendingStore<'store>>,
    attempted: bool,
    postimages_attempted: bool,
    #[cfg(feature = "text-index")]
    text_anchors: Vec<RegisteredTextIndex>,
    #[cfg(feature = "text-index")]
    text_fence: TextRegistryFenceWorkspace,
}

impl<'store> IndexRegistryWorkspace<'store> {
    /// Captures inputs without validating or publishing any contents.
    ///
    /// Concrete index values are wrapped, not copied or rebuilt. Every later
    /// fallible conversion operates on this already-owned storage.
    #[must_use]
    pub fn new(edits: Vec<StoreIndexEdits<'store>>) -> Self {
        Self {
            registry: RegistryWorkspace::new(edits),
            authority: RegistryAuthorityWorkspace::new(),
        }
    }

    fn prepare_inputs(&mut self) -> Result<()> {
        self.registry.prepare_inputs()
    }
}

impl<'store> RegistryWorkspace<'store> {
    fn new(edits: Vec<StoreIndexEdits<'store>>) -> Self {
        Self {
            active: Vec::new(),
            stores: edits
                .into_iter()
                .map(|StoreIndexEdits { store, edits }| PendingStore {
                    store,
                    edits: edits.into_iter().map(PendingEdit::new).collect(),
                    postimages: Postimages::default(),
                })
                .collect(),
            attempted: false,
            postimages_attempted: false,
            #[cfg(feature = "text-index")]
            text_anchors: Vec::new(),
            #[cfg(feature = "text-index")]
            text_fence: TextRegistryFenceWorkspace::new(),
        }
    }

    fn prepare_inputs(&mut self) -> Result<()> {
        if self.attempted {
            return Err(conflict("workspace preparation has already been attempted"));
        }
        self.attempted = true;
        prepare_inputs(&mut self.stores)?;
        reserve_vec(&mut self.active, self.stores.len())
    }
}

impl RegistryWorkspace<'_> {
    fn release_guards(&mut self) {
        // Also handle an explicitly forgotten fence: release every inner
        // writer before automatic field destruction can retire any payload.
        #[cfg(feature = "text-index")]
        {
            release_maintenance_scopes(&mut self.stores);
            self.text_fence.release_guards();
        }
        release_publications(&mut self.stores, &mut self.active);
    }
}

impl Drop for RegistryWorkspace<'_> {
    fn drop(&mut self) {
        self.release_guards();
    }
}

impl Drop for IndexRegistryWorkspace<'_> {
    fn drop(&mut self) {
        self.registry.release_guards();
        self.authority.release_guards();
    }
}

/// Prepared registry postimages retaining publication exclusion.
#[must_use]
pub struct PreparedIndexRegistryBatch<'store, 'workspace> {
    fences: RegistryFences<'store, 'workspace>,
    authority: RegistryAuthority<'store, 'workspace>,
}

/// Installed publication fence, held across the caller's companion publication.
#[must_use]
pub struct InstalledIndexRegistryFence<'store, 'workspace> {
    _fences: RegistryFences<'store, 'workspace>,
    _authority: RegistryAuthority<'store, 'workspace>,
}

/// Prepares every change before making any registry visible.
///
/// # Errors
/// Rejects stale/foreign observations, denied authority and invalid batches.
pub fn prepare_index_registry_batch<'store, 'workspace>(
    workspace: &'workspace mut IndexRegistryWorkspace<'store>,
) -> Result<PreparedIndexRegistryBatch<'store, 'workspace>> {
    workspace.prepare_inputs()?;
    let authority =
        RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)?;
    let released = prepare_under_authority(&mut workspace.registry, authority.workspace)?;
    let prepared = released.rebind().map_err(DataRebindError::into_error)?;
    // The public registry-only owner keeps the same proofs alive without a
    // self-reference. The private aggregate path instead borrows its driver's
    // proofs, so data and registry preparation never reacquire a store gate.
    let BorrowedPreparedRegistry { fences, .. } = prepared;
    Ok(PreparedIndexRegistryBatch { fences, authority })
}

fn prepare_under_authority<'store, 'workspace, 'authority>(
    workspace: &'workspace mut RegistryWorkspace<'store>,
    authority: &'authority RegistryAuthorityWorkspace<'store>,
) -> Result<ReleasedRegistryBatch<'store, 'workspace, 'authority>> {
    if !workspace.attempted || workspace.postimages_attempted {
        return Err(conflict("registry postimage preparation is not fresh"));
    }
    workspace.postimages_attempted = true;
    for pending in &workspace.stores {
        authority
            .transition(pending.store)
            .map_err(DataRebindError::into_error)?;
    }
    let fences = RegistryFences::acquire(&mut workspace.stores, &mut workspace.active)
        .map_err(DataRebindError::into_error)?;
    for publication in fences.active.iter_mut() {
        let transition = authority
            .transition(publication.store)
            .map_err(DataRebindError::into_error)?;
        publication.prepare(
            &mut fences.stores[publication.workspace_index].edits,
            transition,
            authority.bindings()?,
            #[cfg(feature = "text-index")]
            &mut workspace.text_anchors,
        )?;
    }
    // Keep one cleanup owner across released and rebound stages. Only inner
    // registry writers drain here; survivor mutation scopes remain pinned.
    release_publications(fences.stores, fences.active);
    #[cfg(feature = "text-index")]
    workspace.text_fence.prepare(&workspace.text_anchors)?;
    Ok(ReleasedRegistryBatch {
        fences,
        #[cfg(feature = "text-index")]
        text_workspace: &mut workspace.text_fence,
        authority,
    })
}

struct ReleasedRegistryBatch<'store, 'workspace, 'authority> {
    fences: RegistryFences<'store, 'workspace>,
    #[cfg(feature = "text-index")]
    text_workspace: &'workspace mut TextRegistryFenceWorkspace,
    authority: &'authority RegistryAuthorityWorkspace<'store>,
}

struct BorrowedPreparedRegistry<'store, 'workspace, 'authority> {
    fences: RegistryFences<'store, 'workspace>,
    _authority: &'authority RegistryAuthorityWorkspace<'store>,
}

impl<'store, 'workspace, 'authority> ReleasedRegistryBatch<'store, 'workspace, 'authority> {
    fn rebind(
        mut self,
    ) -> std::result::Result<
        BorrowedPreparedRegistry<'store, 'workspace, 'authority>,
        DataRebindError,
    > {
        self.fences
            .acquire_publications(RegistryAcquisition::FinalRebind)?;
        #[cfg(not(feature = "text-index"))]
        let fences = self.fences;
        #[cfg(feature = "text-index")]
        let mut fences = self.fences;
        #[cfg(feature = "text-index")]
        {
            fences.text_fence = Some(TextRegistryBatchFence::try_acquire(self.text_workspace)?);
        }
        for publication in fences.active.iter() {
            self.authority.transition(publication.store)?;
            for edit in &fences.stores[publication.workspace_index].edits {
                if !publication.matches_current(edit) {
                    return Err(DataRebindError::Conflict(
                        "prepared index registration changed",
                    ));
                }
                if let Some(maintenance) = &edit.maintenance {
                    maintenance.validate(
                        &edit.key,
                        &publication.guards,
                        #[cfg(feature = "text-index")]
                        fences.text_fence.as_mut(),
                    )?;
                }
            }
        }
        Ok(BorrowedPreparedRegistry {
            fences,
            _authority: self.authority,
        })
    }
}

impl<'store, 'workspace> PreparedIndexRegistryBatch<'store, 'workspace> {
    /// Installs the prepared postimages without releasing publication exclusion.
    pub fn install(mut self) -> InstalledIndexRegistryFence<'store, 'workspace> {
        // Each guard and postimage are structurally paired, not zipped from
        // independently sized vectors. Retired maps stay in the outer workspace.
        self.fences.install();
        InstalledIndexRegistryFence {
            _fences: self.fences,
            _authority: self.authority,
        }
    }
}

impl RegistryFences<'_, '_> {
    fn install(&mut self) {
        for publication in self.active.iter_mut() {
            for edit in &mut self.stores[publication.workspace_index].edits {
                if let Some(maintenance) = &mut edit.maintenance {
                    maintenance.install(
                        &edit.key,
                        &publication.guards,
                        #[cfg(feature = "text-index")]
                        self.text_fence.as_mut(),
                    );
                }
            }
            std::mem::swap(
                &mut *publication.guards.property,
                &mut publication.postimages.property,
            );
            #[cfg(feature = "vector-index")]
            std::mem::swap(
                &mut *publication.guards.vector,
                &mut publication.postimages.vector,
            );
            #[cfg(feature = "text-index")]
            std::mem::swap(
                &mut *publication.guards.text,
                &mut publication.postimages.text,
            );
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            std::mem::swap(
                &mut *publication.guards.slots,
                &mut publication.postimages.slots,
            );
        }
    }
}

fn conflict(reason: &str) -> Error {
    Error::Transaction(TransactionError::WriteConflict(format!(
        "index registry batch: {reason}"
    )))
}

fn reservation() -> Result<()> {
    #[cfg(test)]
    {
        let denied = RESERVATION_FAILURE.with(|remaining| match remaining.get() {
            Some(0) => true,
            Some(n) => {
                remaining.set(Some(n - 1));
                false
            }
            None => false,
        });
        if denied {
            return Err(AllocError::OutOfMemory.into());
        }
    }
    Ok(())
}

fn reserve_vec<T>(values: &mut Vec<T>, additional: usize) -> Result<()> {
    reservation()?;
    values
        .try_reserve(additional)
        .map_err(|_| AllocError::OutOfMemory.into())
}

fn reserve_map<K: std::hash::Hash + Eq, V>(
    values: &mut FxHashMap<K, V>,
    additional: usize,
) -> Result<()> {
    reservation()?;
    values
        .try_reserve(additional)
        .map_err(|_| AllocError::OutOfMemory.into())
}

#[cfg(test)]
thread_local! {
    static RESERVATION_FAILURE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    static SLOT_SOURCE: std::cell::RefCell<Option<Arc<std::sync::atomic::AtomicU64>>> = const { std::cell::RefCell::new(None) };
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum LocalKey {
    Property(PropertyKey),
    #[cfg(feature = "text-index")]
    Text(String),
    #[cfg(feature = "vector-index")]
    Vector(String),
}

impl LocalKey {
    fn from_request(key: IndexRegistryKey) -> Self {
        match key {
            IndexRegistryKey::Property(key) => Self::Property(key),
            #[cfg(feature = "text-index")]
            IndexRegistryKey::Text { label, property } => {
                Self::Text(encode_index_key(&label, &property))
            }
            #[cfg(feature = "vector-index")]
            IndexRegistryKey::Vector { label, property } => {
                Self::Vector(encode_index_key(&label, &property))
            }
        }
    }

    fn observed(observation: &IndexRegistrationObservation) -> Self {
        match observation.target() {
            ObservedIndex::Property(key, _) => Self::Property(key.clone()),
            #[cfg(feature = "text-index")]
            ObservedIndex::Text(key, _) => Self::Text(key.clone()),
            #[cfg(feature = "vector-index")]
            ObservedIndex::Vector(key, _) => Self::Vector(key.clone()),
        }
    }
}

enum PrivateContents {
    Property {
        rows: Vec<(NodeId, Value)>,
        history: Option<super::PropertyIndexImage>,
        prepared: RegisteredIndex<Arc<PropertyIndexRows>>,
    },
    #[cfg(feature = "text-index")]
    Text {
        input: Arc<parking_lot::RwLock<InvertedIndex>>,
        prepared: Option<RegisteredIndex<RegisteredTextIndex>>,
    },
    #[cfg(feature = "vector-index")]
    Vector(RegisteredIndex<Arc<VectorIndexKind>>),
}

impl PrivateContents {
    fn new(contents: IndexRegistryContents) -> Self {
        match contents {
            IndexRegistryContents::Property(rows) => Self::Property {
                rows,
                history: None,
                prepared: RegisteredIndex::new(Arc::new(PropertyIndexRows::new())),
            },
            IndexRegistryContents::PropertyHistory(history) => Self::Property {
                rows: Vec::new(),
                history: Some(history),
                prepared: RegisteredIndex::new(Arc::new(PropertyIndexRows::new())),
            },
            #[cfg(feature = "text-index")]
            IndexRegistryContents::Text(index) => Self::Text {
                input: Arc::new(parking_lot::RwLock::new(index)),
                prepared: None,
            },
            #[cfg(feature = "vector-index")]
            IndexRegistryContents::Vector(index) => {
                Self::Vector(RegisteredIndex::new(Arc::new(index)))
            }
        }
    }

    fn prepare(&mut self, key: &LocalKey, epoch: EpochId) -> Result<()> {
        match (key, self) {
            (
                LocalKey::Property(_),
                Self::Property {
                    rows,
                    history,
                    prepared,
                },
            ) => {
                if let Some(image) = history.take() {
                    prepared.payload = Arc::new(PropertyIndexRows::from_image(image)?);
                    return Ok(());
                }
                let map = Arc::get_mut(&mut prepared.payload).ok_or_else(|| {
                    conflict("private Property contents have already been shared")
                })?;
                reservation()?;
                map.try_reserve(rows.len())
                    .map_err(|_| AllocError::OutOfMemory)?;
                for (id, value) in rows {
                    let mut members = map.entry(HashableValue::new(value.clone())).or_default();
                    reservation()?;
                    members
                        .try_reserve(1)
                        .map_err(|_| AllocError::OutOfMemory)?;
                    members.insert(*id);
                }
                // Legacy registry-only inputs have current rows but do not
                // assert older coverage. Their first admitted epoch is explicit.
                map.seed_current_history(epoch)?;
                Ok(())
            }
            #[cfg(feature = "text-index")]
            (LocalKey::Text(_), Self::Text { input, prepared }) => {
                let mut index = input.write();
                if !index.is_concrete_registry_candidate() {
                    return Err(conflict("owned Text contents are a forwarding shell"));
                }
                let target = index.pin_registry_target();
                *prepared = Some(RegisteredIndex::new(RegisteredTextIndex::new(
                    Arc::clone(input),
                    target,
                )));
                Ok(())
            }
            #[cfg(feature = "vector-index")]
            (LocalKey::Vector(_), Self::Vector(_)) => Ok(()),
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            _ => Err(conflict("contents do not match the physical index family")),
        }
    }
}

struct PendingEdit {
    key: LocalKey,
    expected: Option<IndexRegistrationObservation>,
    contents: Option<PrivateContents>,
    maintenance: Option<PrivateMaintenance>,
    #[cfg(feature = "text-index")]
    capture_text_wal: bool,
    #[cfg(feature = "text-index")]
    text_wal: Option<(bool, Vec<u8>)>,
    #[cfg(feature = "vector-index")]
    capture_vector_wal: bool,
    #[cfg(feature = "vector-index")]
    vector_wal: Option<Vec<u8>>,
}

impl PendingEdit {
    fn new(edit: IndexRegistryEdit) -> Self {
        let (key, expected, contents, maintenance) = match edit {
            IndexRegistryEdit::Create { key, contents } => {
                (LocalKey::from_request(key), None, Some(contents), None)
            }
            IndexRegistryEdit::Drop { expected } => {
                (LocalKey::observed(&expected), Some(expected), None, None)
            }
            IndexRegistryEdit::Replace { expected, contents } => (
                LocalKey::observed(&expected),
                Some(expected),
                Some(contents),
                None,
            ),
            IndexRegistryEdit::Maintain { expected, changes } => (
                LocalKey::observed(&expected),
                Some(expected),
                None,
                Some(PrivateMaintenance::new(changes)),
            ),
        };
        Self {
            key,
            expected,
            contents: contents.map(PrivateContents::new),
            maintenance,
            #[cfg(feature = "text-index")]
            capture_text_wal: false,
            #[cfg(feature = "text-index")]
            text_wal: None,
            #[cfg(feature = "vector-index")]
            capture_vector_wal: false,
            #[cfg(feature = "vector-index")]
            vector_wal: None,
        }
    }
}

enum PrivateMaintenance {
    Property(PropertyMaintenanceWorkspace),
    #[cfg(feature = "text-index")]
    Text {
        // Scope guard drains before outer payloads, including on abandonment.
        scope: Option<TextCommitScope>,
        workspace: TextCommitWorkspace,
    },
}

impl PrivateMaintenance {
    fn new(changes: IndexRegistryMaintenance) -> Self {
        match changes {
            IndexRegistryMaintenance::Property(changes) => {
                Self::Property(PropertyMaintenanceWorkspace::new(changes))
            }
            IndexRegistryMaintenance::PropertyAt { rows, commit_epoch } => {
                Self::Property(PropertyMaintenanceWorkspace::at_epoch(rows, commit_epoch))
            }
            #[cfg(feature = "text-index")]
            IndexRegistryMaintenance::Text {
                rows,
                frontier,
                commit_epoch,
                transaction_id,
            } => Self::Text {
                scope: None,
                workspace: TextCommitWorkspace::new(rows, frontier, commit_epoch, transaction_id),
            },
            #[cfg(feature = "text-index")]
            IndexRegistryMaintenance::TextRecorded {
                payload,
                frontier,
                commit_epoch,
                transaction_id,
            } => Self::Text {
                scope: None,
                workspace: TextCommitWorkspace::from_recorded(
                    payload,
                    frontier,
                    commit_epoch,
                    transaction_id,
                ),
            },
            #[cfg(feature = "text-index")]
            IndexRegistryMaintenance::TextRebuild {
                config,
                min_token_length,
                rows,
                frontier,
                commit_epoch,
                transaction_id,
            } => Self::Text {
                scope: None,
                workspace: TextCommitWorkspace::for_rebuild(
                    rows,
                    frontier,
                    commit_epoch,
                    transaction_id,
                    (config, min_token_length),
                ),
            },
        }
    }

    fn prepare(&mut self, key: &LocalKey, guards: &StoreGuards<'_>, epoch: EpochId) -> Result<()> {
        match (self, key) {
            (Self::Property(workspace), LocalKey::Property(key)) => {
                let current = guards
                    .property
                    .get(key)
                    .ok_or_else(|| conflict("Property maintenance registration is absent"))?;
                workspace.prepare_at(&current.payload, epoch)
            }
            #[cfg(feature = "text-index")]
            (Self::Text { workspace, scope }, LocalKey::Text(key)) => {
                let current = guards
                    .text
                    .get(key)
                    .ok_or_else(|| conflict("Text maintenance registration is absent"))?;
                let mut index = current
                    .try_write()
                    .ok_or_else(|| conflict("Text maintenance target is in use"))?;
                *scope = Some(index.pin_commit_scope(workspace)?);
                let scope = scope
                    .as_ref()
                    .ok_or_else(|| conflict("Text maintenance scope was not retained"))?;
                index.prepare_commit_fragments(workspace, scope)?;
                Ok(())
            }
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            _ => Err(conflict(
                "maintenance does not match the physical index family",
            )),
        }
    }

    fn validate(
        &self,
        key: &LocalKey,
        guards: &StoreGuards<'_>,
        #[cfg(feature = "text-index")] text_fence: Option<&mut TextRegistryBatchFence<'_>>,
    ) -> std::result::Result<(), DataRebindError> {
        match (self, key) {
            (Self::Property(workspace), LocalKey::Property(key)) => {
                let current = guards.property.get(key).ok_or(DataRebindError::Conflict(
                    "Property maintenance registration is absent",
                ))?;
                workspace.validate(&current.payload)
            }
            #[cfg(feature = "text-index")]
            (
                Self::Text {
                    workspace,
                    scope: Some(scope),
                },
                LocalKey::Text(key),
            ) => {
                let current = guards.text.get(key).ok_or(DataRebindError::Conflict(
                    "Text maintenance registration is absent",
                ))?;
                let fence =
                    text_fence.ok_or(DataRebindError::new("Text collective fence is absent"))?;
                let index = fence.maintenance_target(&current.payload, scope).ok_or(
                    DataRebindError::new("Text maintenance scope or target changed"),
                )?;
                workspace.validate_prepared(index, scope)
            }
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            _ => Err(DataRebindError::new(
                "maintenance family differs from prepared registration",
            )),
        }
    }

    fn install(
        &mut self,
        key: &LocalKey,
        guards: &StoreGuards<'_>,
        #[cfg(feature = "text-index")] text_fence: Option<&mut TextRegistryBatchFence<'_>>,
    ) {
        // Registry rebind proved the paired family, target and capacities;
        // every registry writer remains retained until companion publication.
        match (self, key) {
            (Self::Property(workspace), LocalKey::Property(key)) => {
                if let Some(current) = guards.property.get(key) {
                    workspace.install(&current.payload);
                }
            }
            #[cfg(feature = "text-index")]
            (
                Self::Text {
                    workspace,
                    scope: Some(scope),
                },
                LocalKey::Text(key),
            ) => {
                if let (Some(current), Some(fence)) = (guards.text.get(key), text_fence) {
                    fence.install_maintenance(&current.payload, workspace, scope);
                }
            }
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            _ => {}
        }
    }
}

struct PendingStore<'store> {
    store: &'store LpgStore,
    edits: Vec<PendingEdit>,
    postimages: Postimages,
}

#[cfg(feature = "text-index")]
fn release_maintenance_scopes(stores: &mut [PendingStore<'_>]) {
    for store in stores.iter_mut().rev() {
        for edit in store.edits.iter_mut().rev() {
            if let Some(PrivateMaintenance::Text { scope, .. }) = &mut edit.maintenance {
                drop(scope.take());
            }
        }
    }
}

fn prepare_inputs(pending: &mut [PendingStore<'_>]) -> Result<()> {
    let mut seen = FxHashSet::default();
    reservation()?;
    seen.try_reserve(pending.len())
        .map_err(|_| AllocError::OutOfMemory)?;
    for store in pending {
        if !seen.insert(physical_order(store.store)) {
            return Err(conflict("duplicate physical store"));
        }
        let mut keys = FxHashSet::default();
        reservation()?;
        keys.try_reserve(store.edits.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for edit in &mut store.edits {
            if !keys.insert(edit.key.clone()) {
                return Err(conflict("repeated operation on one physical key"));
            }
            if let Some(contents) = &mut edit.contents {
                #[cfg(feature = "text-index")]
                if edit.capture_text_wal
                    && let PrivateContents::Text { input, .. } = contents
                {
                    edit.text_wal = Some((true, input.read().encode_wal_birth()?));
                }
                #[cfg(feature = "vector-index")]
                if edit.capture_vector_wal
                    && let PrivateContents::Vector(input) = contents
                {
                    edit.vector_wal = Some(input.payload.encode_wal_birth()?);
                }
                contents.prepare(&edit.key, store.store.current_epoch())?;
            }
        }
    }
    Ok(())
}

fn physical_order(store: &LpgStore) -> usize {
    // Ordering only: retained opaque identity remains the equality authority.
    Arc::as_ptr(&store.index_physical_identity).addr()
}

type PropertyRegistry = FxHashMap<PropertyKey, RegisteredIndex<Arc<PropertyIndexRows>>>;
#[cfg(feature = "text-index")]
type TextRegistry = FxHashMap<String, RegisteredIndex<RegisteredTextIndex>>;
#[cfg(feature = "vector-index")]
type VectorRegistry = FxHashMap<String, RegisteredIndex<Arc<VectorIndexKind>>>;

struct StoreGuards<'store> {
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    slots: MutexGuard<'store, FxHashMap<String, u64>>,
    property: RwLockWriteGuard<'store, PropertyRegistry>,
    #[cfg(feature = "vector-index")]
    vector: RwLockWriteGuard<'store, VectorRegistry>,
    #[cfg(feature = "text-index")]
    text: RwLockWriteGuard<'store, TextRegistry>,
}

#[derive(Default)]
struct Postimages {
    property: PropertyRegistry,
    #[cfg(feature = "vector-index")]
    vector: VectorRegistry,
    #[cfg(feature = "text-index")]
    text: TextRegistry,
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    slots: FxHashMap<String, u64>,
}

struct StorePublication<'store> {
    store: &'store LpgStore,
    guards: StoreGuards<'store>,
    workspace_index: usize,
    postimages: Postimages,
}

impl StorePublication<'_> {
    fn prepare(
        &mut self,
        edits: &mut [PendingEdit],
        transition: &PinnedLpgTransition<'_>,
        bindings: &BindingFences,
        #[cfg(feature = "text-index")] text_anchors: &mut Vec<RegisteredTextIndex>,
    ) -> Result<()> {
        reserve_map(
            &mut self.postimages.property,
            self.guards
                .property
                .len()
                .checked_add(edits.len())
                .ok_or(AllocError::InsufficientSpace)?,
        )?;
        self.postimages.property.extend(
            self.guards
                .property
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        #[cfg(feature = "vector-index")]
        {
            reserve_map(
                &mut self.postimages.vector,
                self.guards
                    .vector
                    .len()
                    .checked_add(edits.len())
                    .ok_or(AllocError::InsufficientSpace)?,
            )?;
            self.postimages.vector.extend(
                self.guards
                    .vector
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }
        #[cfg(feature = "text-index")]
        {
            reserve_map(
                &mut self.postimages.text,
                self.guards
                    .text
                    .len()
                    .checked_add(edits.len())
                    .ok_or(AllocError::InsufficientSpace)?,
            )?;
            self.postimages.text.extend(
                self.guards
                    .text
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
            reserve_vec(text_anchors, edits.len())?;
        }
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        {
            reserve_map(
                &mut self.postimages.slots,
                self.guards
                    .slots
                    .len()
                    .checked_add(edits.len())
                    .ok_or(AllocError::InsufficientSpace)?,
            )?;
            self.postimages.slots.extend(
                self.guards
                    .slots
                    .iter()
                    .map(|(key, value)| (key.clone(), *value)),
            );
        }
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        let _ = bindings;
        for edit in edits {
            if let Some(expected) = &edit.expected {
                self.store
                    .validate_registration_store(expected, &transition.transport_authority)?;
            }
            if !self.matches_current(edit) {
                return Err(conflict("registration is absent, occupied, or changed"));
            }
            #[cfg(feature = "text-index")]
            if let (LocalKey::Text(key), Some(_)) = (&edit.key, &edit.expected) {
                let current = self
                    .guards
                    .text
                    .get(key)
                    .ok_or_else(|| conflict("validated Text registration is absent"))?;
                text_anchors.push(current.payload.clone());
            }
            if let Some(maintenance) = &mut edit.maintenance {
                maintenance.prepare(&edit.key, &self.guards, self.store.current_epoch())?;
                #[cfg(feature = "text-index")]
                if edit.capture_text_wal
                    && let PrivateMaintenance::Text { workspace, .. } = maintenance
                {
                    edit.text_wal = Some((false, workspace.encode_wal_postimage()?));
                }
                continue;
            }
            match (&edit.key, &edit.contents) {
                (
                    LocalKey::Property(key),
                    Some(PrivateContents::Property {
                        prepared: contents, ..
                    }),
                ) => {
                    self.postimages
                        .property
                        .insert(key.clone(), contents.clone());
                }
                (LocalKey::Property(key), None) => {
                    self.postimages.property.remove(key);
                }
                #[cfg(feature = "vector-index")]
                (LocalKey::Vector(key), Some(PrivateContents::Vector(contents))) => {
                    let slot = self.postimages.slot(key)?;
                    bindings.bind_vector(
                        &contents.payload,
                        self.store.index_owner_id,
                        slot,
                        self.store.mutation_scope.load(Ordering::Acquire),
                    )?;
                    self.postimages.vector.insert(key.clone(), contents.clone());
                }
                #[cfg(feature = "vector-index")]
                (LocalKey::Vector(key), None) => {
                    self.postimages.vector.remove(key);
                }
                #[cfg(feature = "text-index")]
                (
                    LocalKey::Text(key),
                    Some(PrivateContents::Text {
                        prepared: Some(contents),
                        ..
                    }),
                ) => {
                    let slot = self.postimages.slot(key)?;
                    bindings.bind_text(
                        &contents.read(),
                        self.store.index_owner_id,
                        slot,
                        self.store.mutation_scope.load(Ordering::Acquire),
                    )?;
                    self.postimages.text.insert(key.clone(), contents.clone());
                }
                #[cfg(feature = "text-index")]
                (LocalKey::Text(key), None) => {
                    self.postimages.text.remove(key);
                }
                #[cfg(any(feature = "text-index", feature = "vector-index"))]
                _ => return Err(conflict("private contents do not match normalized key")),
            }
        }
        Ok(())
    }

    fn matches_current(&self, edit: &PendingEdit) -> bool {
        match &edit.key {
            LocalKey::Property(key) => match (&edit.expected, self.guards.property.get(key)) {
                (None, None) => true,
                (Some(expected), Some(current)) => expected.matches_property(current),
                _ => false,
            },
            #[cfg(feature = "vector-index")]
            LocalKey::Vector(key) => match (&edit.expected, self.guards.vector.get(key)) {
                (None, None) => true,
                (Some(expected), Some(current)) => expected.matches_vector(current),
                _ => false,
            },
            #[cfg(feature = "text-index")]
            LocalKey::Text(key) => match (&edit.expected, self.guards.text.get(key)) {
                (None, None) => true,
                (Some(expected), Some(current)) => expected.matches_text(current),
                _ => false,
            },
        }
    }
}

impl Postimages {
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    fn slot(&mut self, key: &str) -> Result<u64> {
        if let Some(&slot) = self.slots.get(key) {
            return Ok(slot);
        }
        let slot = allocate_slot()?;
        // Full postimage capacity was reserved before binding any candidate.
        self.slots.insert(key.to_owned(), slot);
        Ok(slot)
    }
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn allocate_slot() -> Result<u64> {
    #[cfg(test)]
    if let Some(source) = SLOT_SOURCE.with_borrow(Clone::clone) {
        return super::allocate_index_binding_id(&source).map_err(Into::into);
    }
    super::allocate_index_binding_id(&super::NEXT_INDEX_SLOT_ID).map_err(Into::into)
}

struct RegistryFences<'store, 'workspace> {
    #[cfg(feature = "text-index")]
    text_fence: Option<TextRegistryBatchFence<'workspace>>,
    stores: &'workspace mut [PendingStore<'store>],
    active: &'workspace mut Vec<StorePublication<'store>>,
}

#[derive(Clone, Copy)]
enum RegistryAcquisition {
    Preparation,
    FinalRebind,
}

impl<'store, 'workspace> RegistryFences<'store, 'workspace> {
    fn acquire(
        stores: &'workspace mut [PendingStore<'store>],
        active: &'workspace mut Vec<StorePublication<'store>>,
    ) -> std::result::Result<Self, DataRebindError> {
        // RAII exists before the first writer. On rejection/unwind it returns
        // every paired postimage to its original workspace slot, releasing only
        // guards. Neither payloads nor guard-buffer allocations are destroyed.
        let mut fence = Self {
            #[cfg(feature = "text-index")]
            text_fence: None,
            stores,
            active,
        };
        fence.acquire_publications(RegistryAcquisition::Preparation)?;
        Ok(fence)
    }

    fn acquire_publications(
        &mut self,
        mode: RegistryAcquisition,
    ) -> std::result::Result<(), DataRebindError> {
        if !self.active.is_empty() || self.active.capacity() < self.stores.len() {
            return Err(DataRebindError::new(
                "registry guard workspace is not ready",
            ));
        }
        for (workspace_index, pending) in self.stores.iter_mut().enumerate() {
            let store = pending.store;
            // Parking can allocate its thread/hash-table bookkeeping. Final
            // rebind must reject contention before durability without parking,
            // including when an earlier store already retains final writers.
            let guards = StoreGuards {
                #[cfg(any(feature = "text-index", feature = "vector-index"))]
                slots: match mode {
                    RegistryAcquisition::Preparation => store.index_slots.lock(),
                    RegistryAcquisition::FinalRebind => store
                        .index_slots
                        .try_lock()
                        .ok_or(DataRebindError::Conflict("index binding slots are in use"))?,
                },
                property: match mode {
                    RegistryAcquisition::Preparation => store.property_indexes.write(),
                    RegistryAcquisition::FinalRebind => store
                        .property_indexes
                        .try_write()
                        .ok_or(DataRebindError::Conflict("Property registry is in use"))?,
                },
                #[cfg(feature = "vector-index")]
                vector: match mode {
                    RegistryAcquisition::Preparation => store.vector_indexes.write(),
                    RegistryAcquisition::FinalRebind => store
                        .vector_indexes
                        .try_write()
                        .ok_or(DataRebindError::Conflict("Vector registry is in use"))?,
                },
                #[cfg(feature = "text-index")]
                text: match mode {
                    RegistryAcquisition::Preparation => store.text_indexes.write(),
                    RegistryAcquisition::FinalRebind => store
                        .text_indexes
                        .try_write()
                        .ok_or(DataRebindError::Conflict("Text registry is in use"))?,
                },
            };
            self.active.push(StorePublication {
                store,
                guards,
                workspace_index,
                postimages: std::mem::take(&mut pending.postimages),
            });
        }
        Ok(())
    }
}

impl Drop for RegistryFences<'_, '_> {
    fn drop(&mut self) {
        #[cfg(feature = "text-index")]
        {
            release_maintenance_scopes(self.stores);
            drop(self.text_fence.take());
        }
        release_publications(self.stores, self.active);
    }
}

fn release_publications(stores: &mut [PendingStore<'_>], active: &mut Vec<StorePublication<'_>>) {
    for mut publication in active.drain(..).rev() {
        // Private acquisition pairs each guard with this exact slot. The
        // borrowed slice cannot resize/reorder during the fence loan.
        std::mem::swap(
            &mut stores[publication.workspace_index].postimages,
            &mut publication.postimages,
        );
    }
}

#[cfg(test)]
#[path = "index_batch_tests.rs"]
mod tests;

#[cfg(all(test, feature = "text-index"))]
mod text_maintenance_tests;

#[cfg(all(test, feature = "vector-index"))]
mod aggregate_slot_tests;
