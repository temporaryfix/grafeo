//! Sparse label histories and current memberships for prepared data publication.
//!
//! Untouched foreign pending state is left intact. A touched existing node with
//! pending label history is rejected: current membership has no owner metadata,
//! so a union would leak pending labels through committed membership readers.

use super::data_publication::{DataCommitScope, PendingNodeCreationProof};
use super::{LabelOp, LpgStore, PinnedLpgTransition};
use crate::graph::lpg::DataRebindError;
use grafeo_common::memory::arena::AllocError;
use grafeo_common::temporal::VersionLog;
use grafeo_common::types::{EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLockWriteGuard;

type LabelHistory = VersionLog<FxHashSet<u32>>;
type Membership = FxHashMap<NodeId, ()>;

/// Exact label images prepared for one node at the enclosing commit epoch.
///
/// These immutable facts describe publication, not intermediate mutation intent.
/// They are captured only when the enclosing commit requests durable images.
pub struct PreparedNodeLabelImages {
    id: NodeId,
    birth: bool,
    images: Vec<Vec<String>>,
}

impl PreparedNodeLabelImages {
    /// The graph-local node identity qualified by the enclosing store target.
    #[must_use]
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Whether the first image replaces this transaction's creation intent.
    #[must_use]
    pub fn birth(&self) -> bool {
        self.birth
    }

    /// Ordered complete label sets at the enclosing commit epoch.
    #[must_use]
    pub fn images(&self) -> &[Vec<String>] {
        &self.images
    }
}

/// Must outlive every enclosing publication and transition guard.
pub(super) struct LabelCommitWorkspace {
    raw_ops: Vec<(NodeId, u32, LabelOp)>,
    deleted_ids: Vec<NodeId>,
    created_ids: Vec<NodeId>,
    deleted: FxHashSet<NodeId>,
    created: FxHashSet<NodeId>,
    ops: FxHashMap<NodeId, FxHashMap<u32, LabelOp>>,
    touched: Vec<NodeId>,
    final_labels: FxHashMap<NodeId, FxHashSet<u32>>,
    histories: FxHashMap<NodeId, LabelHistory>,
    memberships: FxHashMap<usize, MembershipChange>,
    extension: Vec<Membership>,
    retired_histories: Vec<LabelHistory>,
    label_counts: Vec<(NodeId, u16)>,
    capture_images: bool,
    committed_images: Vec<PreparedNodeLabelImages>,
    original_slots: usize,
    attempted: bool,
}

impl LabelCommitWorkspace {
    pub(super) fn new(
        ops: Vec<(NodeId, u32, LabelOp)>,
        deleted_ids: Vec<NodeId>,
        created_ids: Vec<NodeId>,
    ) -> Self {
        Self {
            raw_ops: ops,
            deleted_ids,
            created_ids,
            deleted: FxHashSet::default(),
            created: FxHashSet::default(),
            ops: FxHashMap::default(),
            touched: Vec::new(),
            final_labels: FxHashMap::default(),
            histories: FxHashMap::default(),
            memberships: FxHashMap::default(),
            extension: Vec::new(),
            retired_histories: Vec::new(),
            label_counts: Vec::new(),
            capture_images: false,
            committed_images: Vec::new(),
            original_slots: 0,
            attempted: false,
        }
    }
}

#[derive(Default)]
struct MembershipChange {
    adds: Vec<NodeId>,
    removes: Vec<NodeId>,
}

#[must_use]
#[cfg(test)]
pub(super) struct PreparedLabelData<'store, 'workspace, 'transition> {
    store: &'store LpgStore,
    guards: LabelDataGuards<'store>,
    workspace: &'workspace mut LabelCommitWorkspace,
    transition: &'transition PinnedLpgTransition<'store>,
}

#[must_use]
#[cfg(test)]
pub(super) struct ReleasedLabelData<'store, 'workspace, 'transition> {
    store: &'store LpgStore,
    workspace: &'workspace mut LabelCommitWorkspace,
    transition: &'transition PinnedLpgTransition<'store>,
}

#[must_use]
#[cfg(test)]
pub(super) struct InstalledLabelDataFence<'store, 'workspace, 'transition> {
    _guards: LabelDataGuards<'store>,
    _workspace: &'workspace mut LabelCommitWorkspace,
    _transition: &'transition PinnedLpgTransition<'store>,
}

/// Store-only writer loans; paired with one workspace by the coordinator.
pub(super) struct LabelDataGuards<'store> {
    // Field order releases in reverse acquisition order.
    labels: RwLockWriteGuard<'store, FxHashMap<NodeId, LabelHistory>>,
    index: RwLockWriteGuard<'store, Vec<Membership>>,
}

impl LpgStore {
    #[cfg(test)]
    pub(super) fn prepare_commit_labels<'store, 'workspace, 'transition>(
        &'store self,
        publication_epoch: EpochId,
        commit_epoch: EpochId,
        transaction_id: TransactionId,
        workspace: &'workspace mut LabelCommitWorkspace,
        transition: &'transition PinnedLpgTransition<'store>,
        creation: &PendingNodeCreationProof<'_, '_, '_>,
    ) -> Result<PreparedLabelData<'store, 'workspace, 'transition>> {
        let guards = self.prepare_commit_label_fragments(
            publication_epoch,
            commit_epoch,
            transaction_id,
            workspace,
            transition,
            creation,
        )?;
        Ok(PreparedLabelData {
            store: self,
            guards,
            workspace,
            transition,
        })
    }

    pub(super) fn prepare_commit_label_fragments<'store>(
        &'store self,
        publication_epoch: EpochId,
        commit_epoch: EpochId,
        transaction_id: TransactionId,
        workspace: &mut LabelCommitWorkspace,
        transition: &PinnedLpgTransition<'store>,
        creation: &PendingNodeCreationProof<'_, '_, '_>,
    ) -> Result<LabelDataGuards<'store>> {
        if workspace.attempted {
            return Err(invalid("workspace preparation has already been attempted"));
        }
        workspace.attempted = true;
        if publication_epoch == EpochId::PENDING
            || commit_epoch == EpochId::PENDING
            || commit_epoch <= publication_epoch
            || transaction_id == TransactionId::INVALID
            || transaction_id == TransactionId::SYSTEM
            || !std::ptr::eq(transition.store, self)
        {
            return Err(invalid(
                "invalid commit epochs, transaction or store transition",
            ));
        }
        for id in &workspace.created_ids {
            if !creation.qualifies(self, transition, transaction_id, *id) {
                return Err(invalid(
                    "created node lacks exact structural ownership proof",
                ));
            }
        }
        workspace
            .deleted
            .try_reserve(workspace.deleted_ids.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .created
            .try_reserve(workspace.created_ids.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .deleted
            .extend(workspace.deleted_ids.iter().copied());
        workspace
            .created
            .extend(workspace.created_ids.iter().copied());
        let registry = self.label_registry.read();
        for &(node, label, op) in &workspace.raw_ops {
            if registry.get_name(label).is_none() {
                return Err(invalid("buffered label ID is absent from the registry"));
            }
            workspace.ops.entry(node).or_default().insert(label, op);
        }
        workspace.touched.extend(workspace.ops.keys().copied());
        workspace.touched.extend(workspace.deleted.iter().copied());
        workspace.touched.extend(workspace.created.iter().copied());
        workspace.touched.sort_unstable();
        workspace.touched.dedup();
        workspace
            .final_labels
            .try_reserve(workspace.touched.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .histories
            .try_reserve(workspace.touched.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .label_counts
            .try_reserve(workspace.touched.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        if workspace.capture_images {
            workspace
                .committed_images
                .try_reserve(workspace.touched.len())
                .map_err(|_| AllocError::OutOfMemory)?;
        }

        let mut index = self.label_index.write();
        let labels = self.node_labels.write();
        workspace.original_slots = index.len();
        for &node in &workspace.touched {
            let old = labels
                .get(&node)
                .ok_or_else(|| invalid("touched node has no label history"))?;
            let created = workspace.created.contains(&node);
            let deleted = workspace.deleted.contains(&node);
            let baseline = if created {
                if old.len() != 1 || old.latest_epoch() != Some(EpochId::PENDING) {
                    return Err(invalid(
                        "creation label history is not one qualified PENDING entry",
                    ));
                }
                old.latest()
                    .ok_or_else(|| invalid("creation label baseline is absent"))?
            } else {
                validate_committed_history(old, publication_epoch)?;
                old.at(publication_epoch)
                    .ok_or_else(|| invalid("committed label baseline is absent"))?
            };
            for label in baseline {
                if registry.get_name(*label).is_none() {
                    return Err(invalid("label baseline references an absent registry ID"));
                }
                let slot = usize::try_from(*label).map_err(|_| AllocError::InsufficientSpace)?;
                if !index
                    .get(slot)
                    .is_some_and(|members| members.contains_key(&node))
                {
                    return Err(invalid("label baseline has no current membership"));
                }
            }
            // Keep final-label scratch in the outer owner before any later
            // fallible count/slot check, on both successful and failed prepare.
            let final_labels = workspace
                .final_labels
                .entry(node)
                .or_insert_with(|| baseline.clone());
            if let Some(ops) = workspace.ops.get(&node) {
                for (&label, op) in ops {
                    match op {
                        LabelOp::Add => {
                            final_labels.insert(label);
                        }
                        LabelOp::Remove => {
                            final_labels.remove(&label);
                        }
                    }
                }
            }
            // Deletion hides the node, not its final committed label image.
            // Retained history must fit even though no live count is published.
            let count = u16::try_from(final_labels.len())
                .map_err(|_| invalid("node label count exceeds record capacity"))?;
            workspace
                .label_counts
                .push((node, if deleted { 0 } else { count }));
            if created {
                let mut history = VersionLog::with_value(
                    commit_epoch,
                    if deleted {
                        baseline.clone()
                    } else {
                        final_labels.clone()
                    },
                );
                if deleted && &*final_labels != baseline {
                    history.append(commit_epoch, final_labels.clone());
                }
                workspace.histories.insert(node, history);
            } else if &*final_labels != baseline {
                let mut history = old.clone();
                history.append(commit_epoch, final_labels.clone());
                workspace.histories.insert(node, history);
            }
            if workspace.capture_images
                && let Some(history) = workspace.histories.get(&node)
            {
                // The completed history is the authority, including the two
                // images retained for a changed zero-width creation. Keep all
                // name storage in the outer workspace before any fallible work.
                workspace.committed_images.push(PreparedNodeLabelImages {
                    id: node,
                    birth: created,
                    images: Vec::new(),
                });
                let captured = workspace
                    .committed_images
                    .last_mut()
                    .ok_or_else(|| invalid("prepared label image slot is absent"))?;
                let history = history.history();
                let start = history.partition_point(|(epoch, _)| *epoch < commit_epoch);
                captured
                    .images
                    .try_reserve(history.len() - start)
                    .map_err(|_| AllocError::OutOfMemory)?;
                for (_, labels) in &history[start..] {
                    captured.images.push(Vec::new());
                    let names = captured
                        .images
                        .last_mut()
                        .ok_or_else(|| invalid("prepared label name slot is absent"))?;
                    names
                        .try_reserve(labels.len())
                        .map_err(|_| AllocError::OutOfMemory)?;
                    for label in labels {
                        let name = registry.get_name(*label).ok_or_else(|| {
                            invalid("prepared label image references an absent registry ID")
                        })?;
                        names.push(String::new());
                        let owned = names
                            .last_mut()
                            .ok_or_else(|| invalid("prepared label name is absent"))?;
                        owned
                            .try_reserve(name.len())
                            .map_err(|_| AllocError::OutOfMemory)?;
                        owned.push_str(name.as_str());
                    }
                    names.sort_unstable();
                }
            }
            for &label in baseline {
                if deleted || !final_labels.contains(&label) {
                    let slot = usize::try_from(label).map_err(|_| AllocError::InsufficientSpace)?;
                    workspace
                        .memberships
                        .entry(slot)
                        .or_default()
                        .removes
                        .push(node);
                }
            }
            if !deleted {
                for &label in final_labels.iter() {
                    if !baseline.contains(&label) {
                        let slot =
                            usize::try_from(label).map_err(|_| AllocError::InsufficientSpace)?;
                        workspace
                            .memberships
                            .entry(slot)
                            .or_default()
                            .adds
                            .push(node);
                    }
                }
            }
        }
        drop(registry);

        #[cfg(test)]
        if FAIL_AFTER_HISTORIES.with(std::cell::Cell::get) {
            return Err(AllocError::OutOfMemory.into());
        }

        let mut required_slots = index.len();
        for &slot in workspace.memberships.keys() {
            required_slots =
                required_slots.max(slot.checked_add(1).ok_or(AllocError::InsufficientSpace)?);
        }
        let additional = required_slots - index.len();
        index
            .try_reserve(additional)
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .extension
            .try_reserve(additional)
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .extension
            .resize_with(additional, Membership::default);
        for (&slot, change) in &workspace.memberships {
            let members = if slot < index.len() {
                &mut index[slot]
            } else {
                &mut workspace.extension[slot - index.len()]
            };
            members
                .try_reserve(change.adds.len())
                .map_err(|_| AllocError::OutOfMemory)?;
        }
        workspace
            .retired_histories
            .try_reserve(workspace.histories.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        Ok(LabelDataGuards { labels, index })
    }
}

impl LabelCommitWorkspace {
    pub(super) fn capture_images(&mut self) {
        self.capture_images = true;
    }

    pub(super) fn committed_images(&self) -> &[PreparedNodeLabelImages] {
        &self.committed_images
    }

    pub(super) fn label_counts(&self) -> &[(NodeId, u16)] {
        &self.label_counts
    }
}

#[cfg(test)]
impl<'store, 'workspace, 'transition> PreparedLabelData<'store, 'workspace, 'transition> {
    pub(super) fn label_counts(&self) -> &[(NodeId, u16)] {
        self.workspace.label_counts()
    }

    pub(super) fn release(
        self,
        transition: &'transition PinnedLpgTransition<'store>,
    ) -> std::result::Result<ReleasedLabelData<'store, 'workspace, 'transition>, DataRebindError>
    {
        if !std::ptr::eq(self.transition, transition) {
            return Err(DataRebindError::new(
                "release requires the original label transition proof",
            ));
        }
        let Self {
            store,
            guards,
            workspace,
            transition,
        } = self;
        drop(guards);
        Ok(ReleasedLabelData {
            store,
            workspace,
            transition,
        })
    }

    pub(super) fn install(self) -> InstalledLabelDataFence<'store, 'workspace, 'transition> {
        let Self {
            store: _,
            mut guards,
            workspace,
            transition,
        } = self;
        guards.install(workspace);
        InstalledLabelDataFence {
            _guards: guards,
            _workspace: workspace,
            _transition: transition,
        }
    }
}

impl LabelDataGuards<'_> {
    fn install(&mut self, workspace: &mut LabelCommitWorkspace) {
        let index = &mut self.index;
        let labels = &mut self.labels;
        // Both Vec capacity and every destination membership map were reserved.
        // Draining retains the extension and history-map allocations outside.
        index.extend(workspace.extension.drain(..));
        for (&slot, change) in &workspace.memberships {
            // Preparation/rebinding qualified this slot under the retained
            // transition and writer; no target discovery occurs here.
            let members = &mut index[slot];
            for node in &change.removes {
                members.remove(node);
            }
            for &node in &change.adds {
                members.insert(node, ());
            }
        }
        for (node, history) in workspace.histories.drain() {
            // All history keys were qualified as existing under these writers.
            // HashMap::insert may reserve before lookup even for replacement.
            if let Some(live) = labels.get_mut(&node) {
                workspace
                    .retired_histories
                    .push(std::mem::replace(live, history));
            }
        }
    }

    pub(super) fn install_in_scope(
        &mut self,
        workspace: &mut LabelCommitWorkspace,
        _scope: &DataCommitScope<'_, '_>,
    ) {
        self.install(workspace);
    }
}

#[cfg(test)]
impl<'store, 'workspace, 'transition> ReleasedLabelData<'store, 'workspace, 'transition> {
    pub(super) fn rebind(
        self,
    ) -> std::result::Result<PreparedLabelData<'store, 'workspace, 'transition>, DataRebindError>
    {
        if !std::ptr::eq(self.transition.store, self.store) {
            return Err(DataRebindError::new("label store transition changed"));
        }
        let guards = LabelDataGuards::rebind(self.store, self.workspace)?;
        Ok(PreparedLabelData {
            store: self.store,
            guards,
            workspace: self.workspace,
            transition: self.transition,
        })
    }
}

impl<'store> LabelDataGuards<'store> {
    pub(super) fn rebind_in_scope(
        workspace: &LabelCommitWorkspace,
        scope: &DataCommitScope<'store, '_>,
    ) -> std::result::Result<Self, DataRebindError> {
        Self::rebind(scope.transition().store, workspace)
    }

    fn rebind(
        store: &'store LpgStore,
        workspace: &LabelCommitWorkspace,
    ) -> std::result::Result<Self, DataRebindError> {
        let index = store
            .label_index
            .try_write()
            .ok_or(DataRebindError::Conflict(
                "label commit membership directory is in use",
            ))?;
        let labels = store
            .node_labels
            .try_write()
            .ok_or(DataRebindError::Conflict(
                "label commit histories are in use",
            ))?;
        let required = workspace
            .original_slots
            .checked_add(workspace.extension.len())
            .ok_or(AllocError::InsufficientSpace)?;
        if index.len() != workspace.original_slots || index.capacity() < required {
            return Err(DataRebindError::new(
                "label membership directory changed or lost capacity",
            ));
        }
        for (&slot, change) in &workspace.memberships {
            let members = if let Some(members) = index.get(slot) {
                members
            } else {
                workspace.extension.get(slot - index.len()).ok_or_else(|| {
                    DataRebindError::new("prepared label membership slot is absent")
                })?
            };
            if members.capacity()
                < members
                    .len()
                    .checked_add(change.adds.len())
                    .ok_or(AllocError::InsufficientSpace)?
            {
                return Err(DataRebindError::new(
                    "reserved label membership capacity was lost",
                ));
            }
        }
        if workspace
            .histories
            .keys()
            .any(|node| !labels.contains_key(node))
        {
            return Err(DataRebindError::new(
                "qualified node label history disappeared",
            ));
        }
        Ok(Self { labels, index })
    }
}

fn validate_committed_history(history: &LabelHistory, publication_epoch: EpochId) -> Result<()> {
    let mut previous = None;
    for (epoch, _) in history.history() {
        if *epoch == EpochId::PENDING
            || *epoch > publication_epoch
            || previous.is_some_and(|previous| previous > *epoch)
        {
            return Err(invalid(
                "existing-node label history is pending, unordered or newer than publication",
            ));
        }
        previous = Some(*epoch);
    }
    Ok(())
}

fn invalid(reason: &str) -> Error {
    TransactionError::InvalidState(format!("label commit preparation: {reason}")).into()
}

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_HISTORIES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
mod tests;
