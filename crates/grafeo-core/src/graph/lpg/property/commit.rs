//! Sparse, allocation-complete property publication for buffered transactions.
//!
//! Workspaces must enclose the caller's publication/transition guards. A column
//! fragment contains only changed entity histories, never unrelated column rows.
//! Existing PENDING entries are preserved, not attributed to a transaction by
//! inference. Write-through pending conversion needs separate ownership proof.

use super::{EntityId, PropertyColumn, PropertyStorage};
#[cfg(test)]
use crate::graph::lpg::store::PinnedLpgTransition;
use crate::graph::lpg::{DataCommitScope, DataRebindError};
use crate::index::zone_map::ZoneMapEntry;
use grafeo_common::memory::arena::AllocError;
use grafeo_common::temporal::VersionLog;
use grafeo_common::types::{EpochId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use hashbrown::hash_map::Entry;
use parking_lot::RwLockWriteGuard;

type Columns<Id> = FxHashMap<PropertyKey, PropertyColumn<Id>>;
type FinalOps<Id> = FxHashMap<PropertyKey, FxHashMap<Id, Option<Value>>>;

/// Outer owner of inputs, sparse candidates and all displaced allocations.
///
/// `None` removes a property; repeated input cells are last-write-wins. Deletion
/// dominates supplied operations except structurally qualified creations closed
/// at the same commit: their final non-null values precede a same-epoch tombstone.
/// Inputs describe buffered final writes, not a request to finalize unidentified
/// PENDING property entries.
pub(crate) struct PropertyCommitWorkspace<Id: EntityId> {
    zero_width_ids: Vec<Id>,
    raw_ops: Vec<(Id, PropertyKey, Option<Value>)>,
    deleted_ids: Vec<Id>,
    deleted: FxHashSet<Id>,
    normalized: FinalOps<Id>,
    keys: Vec<PropertyKey>,
    fragments: Vec<ColumnFragment<Id>>,
    retired_logs: Vec<VersionLog<Value>>,
    retired_metadata: Vec<Vec<ZoneMapEntry>>,
    attempted: bool,
}

impl<Id: EntityId> PropertyCommitWorkspace<Id> {
    /// The coordinator qualifies these sorted creation/deletion intersections
    /// against exact structural ownership and absent property history while
    /// retaining the same excluding store transition.
    pub(crate) fn new(
        zero_width_ids: Vec<Id>,
        ops: Vec<(Id, PropertyKey, Option<Value>)>,
        deleted_ids: Vec<Id>,
    ) -> Self {
        Self {
            zero_width_ids,
            raw_ops: ops,
            deleted_ids,
            deleted: FxHashSet::default(),
            normalized: FxHashMap::default(),
            keys: Vec::new(),
            fragments: Vec::new(),
            retired_logs: Vec::new(),
            retired_metadata: Vec::new(),
            attempted: false,
        }
    }

    fn is_zero_width(ids: &[Id], id: Id) -> bool {
        ids.binary_search_by_key(&id.as_u64(), |candidate| candidate.as_u64())
            .ok()
            .and_then(|index| ids.get(index))
            .is_some_and(|candidate| *candidate == id)
    }
}

struct ColumnFragment<Id: EntityId> {
    key: PropertyKey,
    was_occupied: bool,
    // Occupied-column installation leaves the emptied container here. New
    // columns move their complete candidate into the reserved live directory.
    candidate: Option<PropertyColumn<Id>>,
    missing_entities: usize,
}

/// Writer and sparse postimages bound to one exact property storage.
#[must_use]
#[cfg(test)]
pub(crate) struct PreparedPropertyData<'store, 'workspace, 'transition, Id: EntityId> {
    storage: &'store PropertyStorage<Id>,
    guards: PropertyDataGuards<'store, Id>,
    workspace: &'workspace mut PropertyCommitWorkspace<Id>,
    transition: Option<&'transition PinnedLpgTransition<'store>>,
}

/// Released columns writer, still borrowing the exact excluding LPG transition.
#[must_use]
#[cfg(test)]
pub(crate) struct ReleasedPropertyData<'store, 'workspace, 'transition, Id: EntityId> {
    storage: &'store PropertyStorage<Id>,
    workspace: &'workspace mut PropertyCommitWorkspace<Id>,
    transition: &'transition PinnedLpgTransition<'store>,
}

/// Retains property exclusion and any transition loan after the prepared merge.
/// No candidate or retired payload belongs to this fence.
#[must_use]
#[cfg(test)]
pub(crate) struct InstalledPropertyDataFence<'store, 'workspace, 'transition, Id: EntityId> {
    _guards: PropertyDataGuards<'store, Id>,
    _workspace: &'workspace mut PropertyCommitWorkspace<Id>,
    _transition: Option<&'transition PinnedLpgTransition<'store>>,
}

/// Only target-store loans, never a workspace or local authority loan. The
/// coordinator pairs this bundle with its workspace in one private outer slot.
pub(crate) struct PropertyDataGuards<'store, Id: EntityId> {
    columns: RwLockWriteGuard<'store, Columns<Id>>,
}

impl<Id: EntityId> PropertyStorage<Id> {
    /// Counts committed non-null cells for one touched entity without cloning.
    pub(crate) fn commit_property_count(
        &self,
        id: Id,
        publication_epoch: EpochId,
    ) -> Result<usize> {
        if publication_epoch == EpochId::PENDING {
            return Err(invalid("property count requires a real publication epoch"));
        }
        let columns = self.columns.read();
        let mut count = 0usize;
        for column in columns.values() {
            if let Some(log) = column.values.get(&id) {
                validate_history(log, publication_epoch)?;
                if log
                    .at(publication_epoch)
                    .is_some_and(|value| !value.is_null())
                {
                    count = checked_add(count, 1)?;
                }
            }
        }
        Ok(count)
    }

    /// Buffered creations have no write-through column history to attribute.
    /// Reject compound/other write-through creation histories before durability.
    pub(crate) fn validate_buffered_creation(&self, id: Id) -> Result<()> {
        if self
            .columns
            .read()
            .values()
            .any(|column| column.values.contains_key(&id))
        {
            return Err(invalid(
                "buffered creation already has unqualified property history",
            ));
        }
        Ok(())
    }

    /// Prepares completed cell histories and every installation capacity.
    ///
    /// This writer must not be acquired beneath retained entity-version guards
    /// while allocation is still possible. Use `release`/`rebind` under a
    /// continuously retained exact LPG transition to establish aggregate order.
    #[cfg(test)]
    pub(crate) fn prepare_commit_data<'store, 'workspace, 'transition>(
        &'store self,
        publication_epoch: EpochId,
        commit_epoch: EpochId,
        workspace: &'workspace mut PropertyCommitWorkspace<Id>,
    ) -> Result<PreparedPropertyData<'store, 'workspace, 'transition, Id>> {
        let guards = self.prepare_commit_fragments(publication_epoch, commit_epoch, workspace)?;
        Ok(PreparedPropertyData {
            storage: self,
            guards,
            workspace,
            transition: None,
        })
    }

    /// Completes sparse candidates and reservations; the aggregate retains the
    /// exact excluding transition while releasing these initial writers.
    pub(crate) fn prepare_commit_fragments<'store>(
        &'store self,
        publication_epoch: EpochId,
        commit_epoch: EpochId,
        workspace: &mut PropertyCommitWorkspace<Id>,
    ) -> Result<PropertyDataGuards<'store, Id>> {
        if workspace.attempted {
            return Err(invalid("workspace preparation has already been attempted"));
        }
        workspace.attempted = true;
        if publication_epoch == EpochId::PENDING
            || commit_epoch == EpochId::PENDING
            || commit_epoch <= publication_epoch
        {
            return Err(invalid("commit epoch must be real and after publication"));
        }

        reservation()?;
        workspace
            .deleted
            .try_reserve(workspace.deleted_ids.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .deleted
            .extend(workspace.deleted_ids.iter().copied());
        if workspace
            .zero_width_ids
            .iter()
            .zip(workspace.zero_width_ids.iter().skip(1))
            .any(|(previous, next)| previous.as_u64() >= next.as_u64())
            || workspace
                .zero_width_ids
                .iter()
                .any(|id| !workspace.deleted.contains(id))
        {
            return Err(invalid(
                "zero-width creations must be sorted unique deletion members",
            ));
        }
        for (id, key, value) in &workspace.raw_ops {
            let value = if workspace.deleted.contains(id) {
                if PropertyCommitWorkspace::is_zero_width(&workspace.zero_width_ids, *id) {
                    value.clone().filter(|value| !value.is_null())
                } else {
                    None
                }
            } else {
                value.clone()
            };
            normalize(&mut workspace.normalized, key, *id, value)?;
        }

        let mut columns = self.columns.write();
        // There is no entity -> property-key directory: probe deleted IDs in
        // each column, without visiting unrelated entity histories.
        if !workspace.deleted.is_empty() {
            for (key, column) in &*columns {
                for id in &workspace.deleted {
                    if let Some(log) = column.values.get(id) {
                        validate_history(log, publication_epoch)?;
                        if log
                            .at(publication_epoch)
                            .is_some_and(|value| !value.is_null())
                        {
                            normalize(&mut workspace.normalized, key, *id, None)?;
                        }
                    }
                }
            }
        }

        reserve_vec(&mut workspace.keys, workspace.normalized.len())?;
        workspace.keys.extend(workspace.normalized.keys().cloned());
        workspace.keys.sort_unstable();
        reserve_vec(&mut workspace.fragments, workspace.keys.len())?;
        let mut new_columns = 0usize;
        let mut displaced_logs = 0usize;
        let mut displaced_metadata = 0usize;
        for key in &workspace.keys {
            let source = columns.get(key);
            let mode = source.map_or(self.default_compression, |column| column.compression_mode);
            workspace.fragments.push(ColumnFragment {
                key: key.clone(),
                was_occupied: source.is_some(),
                candidate: Some(PropertyColumn::with_compression(mode)),
                missing_entities: 0,
            });
            let fragment = workspace
                .fragments
                .last_mut()
                .ok_or_else(|| invalid("new column fragment is absent"))?;
            let candidate = fragment
                .candidate
                .as_mut()
                .ok_or_else(|| invalid("new column candidate is absent"))?;
            let ops = workspace
                .normalized
                .get(key)
                .ok_or_else(|| invalid("normalized column is absent"))?;
            reserve_map(&mut candidate.values, ops.len())?;
            for (id, value) in ops {
                let old = source.and_then(|column| column.values.get(id));
                let zero_width =
                    PropertyCommitWorkspace::is_zero_width(&workspace.zero_width_ids, *id);
                if zero_width && old.is_some() {
                    return Err(invalid("zero-width creation already has property history"));
                }
                if let Some(log) = old {
                    validate_history(log, publication_epoch)?;
                }
                // Removal observes committed state, not a foreign pending tail.
                // Missing/already-tombstoned cells do not acquire empty histories.
                if value.is_none()
                    && old
                        .and_then(|log| log.at(publication_epoch))
                        .is_none_or(|value| value.is_null())
                {
                    continue;
                }
                // Finish fallible accounting before constructing the owned
                // postimage, which then moves directly into its outer owner.
                if old.is_some() {
                    displaced_logs = checked_add(displaced_logs, 1)?;
                } else {
                    fragment.missing_entities = checked_add(fragment.missing_entities, 1)?;
                }
                let final_value = match value {
                    Some(value) => value.clone(),
                    None => Value::Null,
                };
                let mut postimage = VersionLog::new();
                if let Some(log) = old {
                    for (epoch, value) in log.history() {
                        if *epoch != EpochId::PENDING {
                            postimage.append(*epoch, value.clone());
                        }
                    }
                }
                // Inputs are normalized: one own final value at C. A qualified
                // zero-width creation closes that value at the same C without
                // making it current or restoring an index membership.
                postimage.append(commit_epoch, final_value);
                if zero_width {
                    postimage.append(commit_epoch, Value::Null);
                }
                if let Some(log) = old {
                    for (epoch, value) in log.history() {
                        if *epoch == EpochId::PENDING {
                            postimage.append(*epoch, value.clone());
                        }
                    }
                }
                candidate.values.insert(*id, postimage);
            }
            if candidate.values.is_empty() {
                continue;
            }
            candidate.zone_map_dirty = true;
            if fragment.was_occupied {
                displaced_metadata = checked_add(displaced_metadata, 1)?;
            } else {
                new_columns = checked_add(new_columns, 1)?;
            }
        }

        reserve_map(&mut columns, new_columns)?;
        for fragment in &workspace.fragments {
            if fragment.was_occupied {
                let column = columns
                    .get_mut(&fragment.key)
                    .ok_or_else(|| invalid("qualified property column disappeared"))?;
                reserve_map(&mut column.values, fragment.missing_entities)?;
            }
        }
        reserve_vec(&mut workspace.retired_logs, displaced_logs)?;
        reserve_vec(&mut workspace.retired_metadata, displaced_metadata)?;
        Ok(PropertyDataGuards { columns })
    }
}

#[cfg(test)]
impl<'store, 'workspace, 'transition, Id: EntityId>
    PreparedPropertyData<'store, 'workspace, 'transition, Id>
{
    /// Releases only the columns writer, retaining exact mutation exclusion.
    pub(crate) fn release(
        self,
        transition: &'transition PinnedLpgTransition<'store>,
    ) -> std::result::Result<
        ReleasedPropertyData<'store, 'workspace, 'transition, Id>,
        DataRebindError,
    > {
        if !transition.pins_property_storage(self.storage)
            || self
                .transition
                .is_some_and(|previous| !std::ptr::eq(previous, transition))
        {
            return Err(DataRebindError::new(
                "release requires the exact property-store transition",
            ));
        }
        let Self {
            storage,
            guards,
            workspace,
            ..
        } = self;
        drop(guards);
        Ok(ReleasedPropertyData {
            storage,
            workspace,
            transition,
        })
    }

    /// Installs only completed histories through capacity-proven keyed merges.
    /// Empty fragment containers and displaced payloads remain workspace-owned.
    pub(crate) fn install(self) -> InstalledPropertyDataFence<'store, 'workspace, 'transition, Id> {
        let Self {
            storage: _,
            mut guards,
            workspace,
            transition,
        } = self;
        guards.install(workspace);
        InstalledPropertyDataFence {
            _guards: guards,
            _workspace: workspace,
            _transition: transition,
        }
    }
}

impl<Id: EntityId> PropertyDataGuards<'_, Id> {
    fn install(&mut self, workspace: &mut PropertyCommitWorkspace<Id>) {
        let columns = &mut self.columns;
        for fragment in &mut workspace.fragments {
            if let Some(mut candidate) = fragment.candidate.take() {
                if candidate.values.is_empty() {
                    fragment.candidate = Some(candidate);
                    continue;
                }
                match columns.entry(fragment.key.clone()) {
                    Entry::Occupied(mut entry) => {
                        let column = entry.get_mut();
                        for (id, history) in candidate.values.drain() {
                            // HashMap::insert may reserve before finding an
                            // existing key. Entry replacement does not grow a
                            // full map; vacant cells use the reserved capacity.
                            match column.values.entry(id) {
                                Entry::Occupied(mut entry) => {
                                    workspace.retired_logs.push(entry.insert(history));
                                }
                                Entry::Vacant(entry) => {
                                    entry.insert(history);
                                }
                            }
                        }
                        column.zone_map_dirty = true;
                        workspace
                            .retired_metadata
                            .push(std::mem::take(&mut column.block_zone_maps));
                        fragment.candidate = Some(candidate);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(candidate);
                    }
                }
            }
        }
    }

    /// Only the coordinator can construct this scope; its private slot retains
    /// the qualified workspace continuously from bind through installation.
    pub(crate) fn install_in_scope(
        &mut self,
        workspace: &mut PropertyCommitWorkspace<Id>,
        _scope: &DataCommitScope<'_, '_>,
    ) {
        self.install(workspace);
    }
}

#[cfg(test)]
impl<'store, 'workspace, 'transition, Id: EntityId>
    ReleasedPropertyData<'store, 'workspace, 'transition, Id>
{
    /// Reacquires the writer before durability, under the continuously borrowed
    /// transition. This is not a general optimistic property rebind operation.
    pub(crate) fn rebind(
        self,
    ) -> std::result::Result<
        PreparedPropertyData<'store, 'workspace, 'transition, Id>,
        DataRebindError,
    > {
        if !self.transition.pins_property_storage(self.storage) {
            return Err(DataRebindError::new(
                "property transition no longer matches storage",
            ));
        }
        let guards = PropertyDataGuards::rebind(self.storage, self.workspace)?;
        Ok(PreparedPropertyData {
            storage: self.storage,
            guards,
            workspace: self.workspace,
            transition: Some(self.transition),
        })
    }
}

impl<'store, Id: EntityId> PropertyDataGuards<'store, Id> {
    pub(crate) fn rebind_in_scope(
        storage: &'store PropertyStorage<Id>,
        workspace: &PropertyCommitWorkspace<Id>,
        scope: &DataCommitScope<'store, '_>,
    ) -> std::result::Result<Self, DataRebindError> {
        if !scope.transition().pins_property_storage(storage) {
            return Err(DataRebindError::new(
                "property slot target differs from its scope",
            ));
        }
        Self::rebind(storage, workspace)
    }

    fn rebind(
        storage: &'store PropertyStorage<Id>,
        workspace: &PropertyCommitWorkspace<Id>,
    ) -> std::result::Result<Self, DataRebindError> {
        let columns = storage
            .columns
            .try_write()
            .ok_or(DataRebindError::Conflict(
                "property commit columns are in use",
            ))?;
        let mut missing_columns = 0usize;
        for fragment in &workspace.fragments {
            let candidate = fragment.candidate.as_ref().ok_or_else(|| {
                DataRebindError::new("property candidate has already been consumed")
            })?;
            if let Some(column) = columns.get(&fragment.key) {
                if !fragment.was_occupied {
                    return Err(DataRebindError::new(
                        "property column appeared while writer was released",
                    ));
                }
                if column.values.capacity()
                    < column
                        .values
                        .len()
                        .checked_add(fragment.missing_entities)
                        .ok_or(AllocError::InsufficientSpace)?
                {
                    return Err(DataRebindError::new(
                        "reserved property entity capacity was lost",
                    ));
                }
            } else {
                if fragment.was_occupied {
                    return Err(DataRebindError::new(
                        "property column disappeared while writer was released",
                    ));
                }
                if !candidate.values.is_empty() {
                    missing_columns = missing_columns
                        .checked_add(1)
                        .ok_or(AllocError::InsufficientSpace)?;
                }
            }
        }
        if columns.capacity()
            < columns
                .len()
                .checked_add(missing_columns)
                .ok_or(AllocError::InsufficientSpace)?
        {
            return Err(DataRebindError::new(
                "reserved property column capacity was lost",
            ));
        }
        Ok(Self { columns })
    }
}

fn normalize<Id: EntityId>(
    ops: &mut FinalOps<Id>,
    key: &PropertyKey,
    id: Id,
    value: Option<Value>,
) -> Result<()> {
    if !ops.contains_key(key) {
        reserve_map(ops, 1)?;
    }
    let column = ops.entry(key.clone()).or_default();
    if !column.contains_key(&id) {
        reserve_map(column, 1)?;
    }
    column.insert(id, value);
    Ok(())
}

fn validate_history(log: &VersionLog<Value>, publication_epoch: EpochId) -> Result<()> {
    let mut previous = None;
    for (epoch, _) in log.history() {
        if previous.is_some_and(|previous| previous > *epoch)
            || (*epoch != EpochId::PENDING && *epoch > publication_epoch)
        {
            return Err(invalid("history is unordered or newer than publication"));
        }
        previous = Some(*epoch);
    }
    Ok(())
}

fn invalid(reason: &str) -> Error {
    TransactionError::InvalidState(format!("property commit preparation: {reason}")).into()
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| AllocError::InsufficientSpace.into())
}

fn reservation() -> Result<()> {
    #[cfg(test)]
    if RESERVATION_FAILURE.with(|remaining| match remaining.get() {
        Some(0) => true,
        Some(n) => {
            remaining.set(Some(n - 1));
            false
        }
        None => false,
    }) {
        return Err(AllocError::OutOfMemory.into());
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
}

#[cfg(test)]
mod tests;
