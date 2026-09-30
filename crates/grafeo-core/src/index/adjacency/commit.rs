//! Sparse, prevalidated commit tombstones that retain every physical row.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::{AdjacencyList, ChunkedAdjacency};
#[cfg(test)]
use crate::graph::lpg::PinnedLpgTransition;
use crate::graph::lpg::{DataCommitScope, DataRebindError};
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLockWriteGuard;
use std::ops::Range;
use std::sync::atomic::Ordering;

/// Outer ownership for requested identities and all preparation scratch.
///
/// Declare before publication/transition guards. Neither preparation proof
/// release nor installation destroys any of these allocations.
pub(crate) struct AdjacencyCommitWorkspace {
    requested: Vec<(NodeId, EdgeId)>,
    targets: Vec<DeleteTarget>,
    groups: Vec<DeleteGroup>,
    lookup: FxHashMap<EdgeId, usize>,
    counts: DeleteCounts,
    attempted: bool,
    #[cfg(test)]
    scanned_lists: usize,
}

struct DeleteTarget {
    source: NodeId,
    edge: EdgeId,
    seen: bool,
    missing: bool,
}

struct DeleteGroup {
    source: NodeId,
    targets: Range<usize>,
}

impl AdjacencyCommitWorkspace {
    pub(crate) fn new(requested: Vec<(NodeId, EdgeId)>) -> Self {
        Self {
            requested,
            targets: Vec::new(),
            groups: Vec::new(),
            lookup: FxHashMap::default(),
            counts: DeleteCounts {
                physical: 0,
                deleted_before: 0,
                deleted_after: 0,
            },
            attempted: false,
            #[cfg(test)]
            scanned_lists: 0,
        }
    }

    fn normalize(&mut self) -> Result<()> {
        if self.attempted {
            return Err(conflict("workspace preparation was already attempted"));
        }
        self.attempted = true;
        reserve_vec(&mut self.targets, self.requested.len())?;
        for &(source, edge) in &self.requested {
            if !source.is_valid() || !edge.is_valid() {
                return Err(conflict("invalid source or edge identity"));
            }
            self.targets.push(DeleteTarget {
                source,
                edge,
                seen: false,
                missing: false,
            });
        }
        self.targets
            .sort_unstable_by_key(|target| (target.source, target.edge));
        self.targets
            .dedup_by_key(|target| (target.source, target.edge));
        let group_count = self
            .targets
            .chunk_by(|left, right| left.source == right.source)
            .count();
        reserve_vec(&mut self.groups, group_count)?;
        let mut start = 0usize;
        let mut largest_group = 0;
        for targets in self
            .targets
            .chunk_by(|left, right| left.source == right.source)
        {
            let Some(first) = targets.first() else {
                return Err(conflict("normalized group is empty"));
            };
            let end = start
                .checked_add(targets.len())
                .ok_or(AllocError::InsufficientSpace)?;
            self.groups.push(DeleteGroup {
                source: first.source,
                targets: start..end,
            });
            largest_group = largest_group.max(targets.len());
            start = end;
        }
        reservation()?;
        self.lookup
            .try_reserve(largest_group)
            .map_err(|_| AllocError::OutOfMemory)?;
        Ok(())
    }
}

/// Exact identities and deleted-count successor under the actual list writer.
#[must_use]
#[cfg(test)]
pub(crate) struct PreparedAdjacencyDeletes<'adjacency, 'workspace, 'transition> {
    owner: &'adjacency ChunkedAdjacency,
    guards: AdjacencyDataGuards<'adjacency>,
    workspace: &'workspace mut AdjacencyCommitWorkspace,
    transition: Option<&'transition PinnedLpgTransition<'adjacency>>,
}

/// Temporary list-writer release qualified by this exact store transition.
///
/// The loan prevents dropping the exclusive transition during the unlocked
/// interval. There is no constructor or standalone unqualified rebind.
#[must_use]
#[cfg(test)]
pub(crate) struct ReleasedAdjacencyDeletes<'adjacency, 'workspace, 'transition> {
    owner: &'adjacency ChunkedAdjacency,
    workspace: &'workspace mut AdjacencyCommitWorkspace,
    transition: &'transition PinnedLpgTransition<'adjacency>,
}

/// Installation retaining exclusion and the outer scratch ownership loan.
#[must_use]
#[cfg(test)]
pub(crate) struct InstalledAdjacencyDeletes<'adjacency, 'workspace, 'transition> {
    _guards: AdjacencyDataGuards<'adjacency>,
    _workspace: &'workspace mut AdjacencyCommitWorkspace,
    _transition: Option<&'transition PinnedLpgTransition<'adjacency>>,
}

/// Borrowed only from the exact adjacency target, not local authority or payload.
pub(crate) struct AdjacencyDataGuards<'adjacency> {
    owner: &'adjacency ChunkedAdjacency,
    lists: RwLockWriteGuard<'adjacency, FxHashMap<NodeId, AdjacencyList>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct DeleteCounts {
    physical: usize,
    deleted_before: usize,
    deleted_after: usize,
}

#[derive(Clone, Copy)]
enum CapacityMode {
    Reserve,
    Retained,
}

impl ChunkedAdjacency {
    /// Prepares sparse soft deletes; no physical row is removed or rebuilt.
    #[cfg(test)]
    pub(crate) fn prepare_commit_deletes<'adjacency, 'workspace, 'transition>(
        &'adjacency self,
        workspace: &'workspace mut AdjacencyCommitWorkspace,
    ) -> Result<PreparedAdjacencyDeletes<'adjacency, 'workspace, 'transition>> {
        let guards = self.prepare_commit_fragments(workspace)?;
        Ok(PreparedAdjacencyDeletes {
            owner: self,
            guards,
            workspace,
            transition: None,
        })
    }

    /// Prepares the aggregate's sparse candidate and reserved destinations.
    pub(crate) fn prepare_commit_fragments<'adjacency>(
        &'adjacency self,
        workspace: &mut AdjacencyCommitWorkspace,
    ) -> Result<AdjacencyDataGuards<'adjacency>> {
        workspace.normalize()?;
        // The inner binding scope drops its writer before an error reaches
        // this boundary. Unlike final rebind, no structural writer is retained
        // by this preparation caller while diagnostics are materialized.
        self.bind_commit_deletes(workspace, CapacityMode::Reserve)
            .map_err(DataRebindError::into_error)
    }

    fn bind_commit_deletes<'adjacency>(
        &'adjacency self,
        workspace: &mut AdjacencyCommitWorkspace,
        capacity: CapacityMode,
    ) -> std::result::Result<AdjacencyDataGuards<'adjacency>, DataRebindError> {
        let mut lists = match capacity {
            CapacityMode::Reserve => self.lists.write(),
            CapacityMode::Retained => self.lists.try_write().ok_or(DataRebindError::Conflict(
                "adjacency commit lists are in use",
            ))?,
        };
        let mut missing_total = 0usize;
        for group in &workspace.groups {
            let list = lists
                .get_mut(&group.source)
                .ok_or_else(|| DataRebindError::new("source has no physical adjacency list"))?;
            workspace.lookup.clear();
            let targets = workspace
                .targets
                .get_mut(group.targets.clone())
                .ok_or_else(|| DataRebindError::new("normalized target range is invalid"))?;
            if workspace.lookup.capacity() < targets.len() {
                return Err(DataRebindError::new(
                    "reserved membership scratch capacity was lost",
                ));
            }
            for (index, target) in targets.iter_mut().enumerate() {
                target.seen = false;
                workspace.lookup.insert(target.edge, index);
            }
            #[cfg(test)]
            {
                workspace.scanned_lists += 1;
            }
            // Membership needs only edge IDs. Inspect packed cold IDs directly
            // instead of allocating decoded destination/edge vectors.
            for chunk in &list.cold_chunks {
                if chunk.edge_ids.len() != chunk.count {
                    return Err(DataRebindError::new(
                        "cold adjacency edge count is inconsistent",
                    ));
                }
                for index in 0..chunk.count {
                    let edge = chunk.edge_ids.get(index).ok_or_else(|| {
                        DataRebindError::new("cold adjacency edge encoding is incomplete")
                    })?;
                    record_membership(&workspace.lookup, targets, EdgeId::new(edge))?;
                }
            }
            for chunk in &list.hot_chunks {
                if chunk.destinations.len() != chunk.edge_ids.len() {
                    return Err(DataRebindError::new(
                        "hot adjacency columns have different lengths",
                    ));
                }
                for &edge in &chunk.edge_ids {
                    record_membership(&workspace.lookup, targets, edge)?;
                }
            }
            for &(_, edge) in &list.delta_inserts {
                record_membership(&workspace.lookup, targets, edge)?;
            }
            let mut missing = 0usize;
            for target in targets {
                if !target.seen {
                    return Err(DataRebindError::new(
                        "requested edge has no physical membership",
                    ));
                }
                let missing_now = !list.deleted.contains(&target.edge);
                match capacity {
                    CapacityMode::Reserve => target.missing = missing_now,
                    CapacityMode::Retained if target.missing != missing_now => {
                        return Err(DataRebindError::new(
                            "tombstone membership changed while writer was released",
                        ));
                    }
                    CapacityMode::Retained => {}
                }
                if target.missing {
                    missing = missing
                        .checked_add(1)
                        .ok_or(AllocError::InsufficientSpace)?;
                }
            }
            if missing != 0 {
                match capacity {
                    CapacityMode::Reserve => {
                        reservation()?;
                        list.deleted
                            .try_reserve(missing)
                            .map_err(|_| AllocError::OutOfMemory)?;
                    }
                    CapacityMode::Retained => {
                        let required = list
                            .deleted
                            .len()
                            .checked_add(missing)
                            .ok_or(AllocError::InsufficientSpace)?;
                        if list.deleted.capacity() < required {
                            return Err(DataRebindError::new(
                                "reserved tombstone capacity was lost",
                            ));
                        }
                    }
                }
            }
            missing_total = missing_total
                .checked_add(missing)
                .ok_or(AllocError::InsufficientSpace)?;
        }
        let physical = self.edge_count.load(Ordering::Relaxed);
        let deleted_before = self.deleted_count.load(Ordering::Relaxed);
        let deleted_after = deleted_before
            .checked_add(missing_total)
            .filter(|deleted| *deleted <= physical)
            .ok_or_else(|| {
                DataRebindError::new("deleted-edge counter would exceed physical edge count")
            })?;
        let counts = DeleteCounts {
            physical,
            deleted_before,
            deleted_after,
        };
        match capacity {
            CapacityMode::Reserve => workspace.counts = counts,
            CapacityMode::Retained => {
                if workspace.counts != counts {
                    return Err(DataRebindError::new(
                        "adjacency counters changed while writer was released",
                    ));
                }
            }
        }
        Ok(AdjacencyDataGuards { owner: self, lists })
    }
}

fn record_membership(
    lookup: &FxHashMap<EdgeId, usize>,
    targets: &mut [DeleteTarget],
    edge: EdgeId,
) -> std::result::Result<(), DataRebindError> {
    if let Some(&index) = lookup.get(&edge) {
        let target = targets
            .get_mut(index)
            .ok_or_else(|| DataRebindError::new("membership scratch index is invalid"))?;
        if target.seen {
            return Err(DataRebindError::new(
                "requested edge has multiple physical memberships",
            ));
        }
        target.seen = true;
    }
    Ok(())
}

#[cfg(test)]
impl<'adjacency, 'workspace, 'transition>
    PreparedAdjacencyDeletes<'adjacency, 'workspace, 'transition>
{
    /// Releases only this writer. Rejection remains allocation-free because
    /// this ready proof may have been rebound beneath other retained writers.
    pub(crate) fn release(
        self,
        transition: &'transition PinnedLpgTransition<'adjacency>,
    ) -> std::result::Result<
        ReleasedAdjacencyDeletes<'adjacency, 'workspace, 'transition>,
        DataRebindError,
    > {
        if !transition.pins_adjacency(self.owner)
            || self
                .transition
                .is_some_and(|previous| !std::ptr::eq(previous, transition))
        {
            return Err(DataRebindError::new(
                "transition does not pin this adjacency",
            ));
        }
        let Self {
            owner,
            guards,
            workspace,
            ..
        } = self;
        drop(guards);
        Ok(ReleasedAdjacencyDeletes {
            owner,
            workspace,
            transition,
        })
    }

    /// Inserts only missing, pre-reserved tombstones and publishes one count.
    /// All physical chunks and scratch allocations survive this operation.
    pub(crate) fn install(
        mut self,
    ) -> InstalledAdjacencyDeletes<'adjacency, 'workspace, 'transition> {
        self.guards.install(self.workspace);
        InstalledAdjacencyDeletes {
            _guards: self.guards,
            _workspace: self.workspace,
            _transition: self.transition,
        }
    }
}

impl<'adjacency> AdjacencyDataGuards<'adjacency> {
    pub(crate) fn rebind_in_scope(
        owner: &'adjacency ChunkedAdjacency,
        workspace: &mut AdjacencyCommitWorkspace,
        scope: &DataCommitScope<'adjacency, '_>,
    ) -> std::result::Result<Self, DataRebindError> {
        if !scope.transition().pins_adjacency(owner) {
            return Err(DataRebindError::new(
                "adjacency slot target differs from its scope",
            ));
        }
        owner.bind_commit_deletes(workspace, CapacityMode::Retained)
    }

    pub(crate) fn install_in_scope(
        &mut self,
        workspace: &mut AdjacencyCommitWorkspace,
        _scope: &DataCommitScope<'_, '_>,
    ) {
        self.install(workspace);
    }

    fn install(&mut self, workspace: &mut AdjacencyCommitWorkspace) {
        for group in &workspace.groups {
            // Preparation qualified this exact key under the continuously
            // retained writer. Scoped rebind qualifies it again before this
            // proof can be reconstructed; no topology mutation occurs here.
            if let Some(list) = self.lists.get_mut(&group.source) {
                for target in &workspace.targets[group.targets.clone()] {
                    if target.missing {
                        list.deleted.insert(target.edge);
                    }
                }
            }
        }
        self.owner
            .deleted_count
            .store(workspace.counts.deleted_after, Ordering::Relaxed);
    }
}

#[cfg(test)]
impl<'adjacency, 'workspace, 'transition>
    ReleasedAdjacencyDeletes<'adjacency, 'workspace, 'transition>
{
    /// Reacquires and revalidates before the durable marker, while the exact
    /// exclusive store transition remains continuously borrowed.
    /// Rejection carries no allocated diagnostics: the enclosing coordinator
    /// may still retain structural writers and must materialize errors later.
    pub(crate) fn rebind(
        self,
    ) -> std::result::Result<
        PreparedAdjacencyDeletes<'adjacency, 'workspace, 'transition>,
        DataRebindError,
    > {
        if !self.transition.pins_adjacency(self.owner) {
            return Err(DataRebindError::new(
                "released transition no longer pins this adjacency",
            ));
        }
        let guards = self
            .owner
            .bind_commit_deletes(self.workspace, CapacityMode::Retained)?;
        Ok(PreparedAdjacencyDeletes {
            owner: self.owner,
            guards,
            workspace: self.workspace,
            transition: Some(self.transition),
        })
    }
}

fn conflict(reason: &str) -> Error {
    Error::Transaction(TransactionError::WriteConflict(format!(
        "adjacency commit preparation: {reason}"
    )))
}

fn reservation() -> std::result::Result<(), AllocError> {
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
            return Err(AllocError::OutOfMemory);
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

#[cfg(test)]
thread_local! {
    static RESERVATION_FAILURE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    struct ResetReservations;

    impl Drop for ResetReservations {
        fn drop(&mut self) {
            RESERVATION_FAILURE.with(|remaining| remaining.set(None));
        }
    }

    #[test]
    fn adjacency_commit_deduplicates_and_retains_historical_physical_rows() -> TestResult {
        let adjacency = ChunkedAdjacency::with_chunk_capacity(4);
        let source = NodeId::new(1);
        for edge in 0..320 {
            adjacency.add_edge(source, NodeId::new(edge + 2), EdgeId::new(edge));
        }
        adjacency.freeze_all();
        // Also retain a recent delta entry alongside the compressed chunks.
        adjacency.add_edge(source, NodeId::new(999), EdgeId::new(500));
        adjacency.mark_deleted(source, EdgeId::new(10));
        let history = adjacency.edges_from_including_deleted(source);
        let mut workspace = AdjacencyCommitWorkspace::new(vec![
            (source, EdgeId::new(500)),
            (source, EdgeId::new(10)),
            (source, EdgeId::new(20)),
            (source, EdgeId::new(20)),
        ]);
        let prepared = adjacency.prepare_commit_deletes(&mut workspace)?;
        assert!(!prepared.guards.lists[&source].cold_chunks.is_empty());
        let capacity = prepared.guards.lists[&source].deleted.capacity();
        let cold_chunks = prepared.guards.lists[&source].cold_chunks.len();
        let hot_chunks = prepared.guards.lists[&source].hot_chunks.len();
        let delta_capacity = prepared.guards.lists[&source].delta_inserts.capacity();
        let installed = prepared.install();
        assert!(adjacency.lists.try_read().is_none());
        assert_eq!(
            installed._guards.lists[&source].deleted.capacity(),
            capacity
        );
        assert_eq!(
            installed._guards.lists[&source].cold_chunks.len(),
            cold_chunks
        );
        assert_eq!(
            installed._guards.lists[&source].hot_chunks.len(),
            hot_chunks
        );
        assert_eq!(
            installed._guards.lists[&source].delta_inserts.capacity(),
            delta_capacity
        );
        drop(installed);
        assert_eq!(adjacency.total_edge_count(), 321);
        assert_eq!(adjacency.active_edge_count(), 318);
        assert_eq!(adjacency.edges_from_including_deleted(source), history);
        let active = adjacency.edges_from(source);
        for edge in [10, 20, 500] {
            assert!(
                !active
                    .iter()
                    .any(|(_, candidate)| *candidate == EdgeId::new(edge))
            );
        }
        assert_eq!(workspace.requested.len(), 4);
        assert_eq!(workspace.targets.len(), 3);
        assert_eq!(workspace.scanned_lists, 1);

        let mut repeated = AdjacencyCommitWorkspace::new(vec![
            (source, EdgeId::new(20)),
            (source, EdgeId::new(10)),
            (source, EdgeId::new(20)),
        ]);
        let prepared = adjacency.prepare_commit_deletes(&mut repeated)?;
        assert_eq!(prepared.guards.lists[&source].deleted.capacity(), capacity);
        assert_eq!(prepared.workspace.counts.deleted_after, 3);
        assert!(
            prepared
                .workspace
                .targets
                .iter()
                .all(|target| !target.missing)
        );
        drop(prepared.install());
        assert_eq!(adjacency.active_edge_count(), 318);
        assert_eq!(adjacency.edges_from_including_deleted(source), history);
        Ok(())
    }

    #[test]
    fn adjacency_commit_late_membership_failure_changes_no_logical_state() {
        let adjacency = ChunkedAdjacency::new();
        adjacency.add_edge(NodeId::new(1), NodeId::new(3), EdgeId::new(10));
        adjacency.add_edge(NodeId::new(2), NodeId::new(3), EdgeId::new(20));
        let mut workspace = AdjacencyCommitWorkspace::new(vec![
            (NodeId::new(1), EdgeId::new(10)),
            (NodeId::new(2), EdgeId::new(99)),
        ]);
        assert!(adjacency.prepare_commit_deletes(&mut workspace).is_err());
        assert_eq!(workspace.scanned_lists, 2);
        assert!(workspace.targets[0].missing);
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(adjacency.total_edge_count(), 2);
        assert_eq!(adjacency.active_edge_count(), 2);
        assert_eq!(adjacency.out_degree(NodeId::new(1)), 1);
        assert_eq!(adjacency.out_degree(NodeId::new(2)), 1);
    }

    #[test]
    fn adjacency_binding_keeps_membership_rejection_static_until_outer_conversion() -> TestResult {
        let adjacency = ChunkedAdjacency::new();
        let source = NodeId::new(1);
        adjacency.add_edge(source, NodeId::new(2), EdgeId::new(10));
        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, EdgeId::new(99))]);
        workspace.normalize()?;
        assert!(matches!(
            adjacency.bind_commit_deletes(&mut workspace, CapacityMode::Reserve),
            Err(DataRebindError::Invalid(
                "requested edge has no physical membership"
            ))
        ));
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(workspace.requested.len(), 1);
        assert_eq!(workspace.targets.len(), 1);
        assert_eq!(adjacency.active_edge_count(), 1);
        Ok(())
    }

    #[test]
    fn adjacency_binding_preserves_reservation_failure_without_public_error_conversion()
    -> TestResult {
        let adjacency = ChunkedAdjacency::new();
        let source = NodeId::new(1);
        let edge = EdgeId::new(10);
        adjacency.add_edge(source, NodeId::new(2), edge);
        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, edge)]);
        workspace.normalize()?;
        {
            let _reset = ResetReservations;
            RESERVATION_FAILURE.with(|remaining| remaining.set(Some(0)));
            assert!(matches!(
                adjacency.bind_commit_deletes(&mut workspace, CapacityMode::Reserve),
                Err(DataRebindError::Allocation(AllocError::OutOfMemory))
            ));
        }
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(workspace.targets.len(), 1);
        assert!(adjacency.lists.read()[&source].deleted.is_empty());
        assert_eq!(adjacency.active_edge_count(), 1);
        Ok(())
    }

    #[test]
    fn adjacency_commit_existing_tombstone_reserves_no_deleted_entries() -> TestResult {
        let adjacency = ChunkedAdjacency::new();
        let source = NodeId::new(1);
        let edge = EdgeId::new(10);
        adjacency.add_edge(source, NodeId::new(2), edge);
        adjacency.mark_deleted(source, edge);
        let capacity = adjacency.lists.read()[&source].deleted.capacity();
        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, edge), (source, edge)]);
        {
            let _reset = ResetReservations;
            // Permit only target/group/lookup scratch reservations. A fourth
            // reservation for this already-present tombstone would fail.
            RESERVATION_FAILURE.with(|remaining| remaining.set(Some(3)));
            let prepared = adjacency.prepare_commit_deletes(&mut workspace)?;
            assert_eq!(prepared.guards.lists[&source].deleted.capacity(), capacity);
            assert_eq!(RESERVATION_FAILURE.with(std::cell::Cell::get), Some(0));
            drop(prepared.install());
        }
        assert_eq!(adjacency.total_edge_count(), 1);
        assert_eq!(adjacency.active_edge_count(), 0);
        assert_eq!(adjacency.edges_from_including_deleted(source).len(), 1);
        Ok(())
    }

    #[test]
    fn adjacency_commit_late_reservation_failure_keeps_partial_scratch_owned() {
        let adjacency = ChunkedAdjacency::new();
        adjacency.add_edge(NodeId::new(1), NodeId::new(3), EdgeId::new(10));
        adjacency.add_edge(NodeId::new(2), NodeId::new(3), EdgeId::new(20));
        let mut workspace = AdjacencyCommitWorkspace::new(vec![
            (NodeId::new(1), EdgeId::new(10)),
            (NodeId::new(2), EdgeId::new(20)),
        ]);
        {
            let _reset = ResetReservations;
            // Targets, groups, lookup and the first deleted set reserve;
            // the second deleted set is the late failing reservation.
            RESERVATION_FAILURE.with(|remaining| remaining.set(Some(4)));
            assert!(matches!(
                adjacency.prepare_commit_deletes(&mut workspace),
                Err(Error::Storage(
                    grafeo_common::utils::error::StorageError::Full
                ))
            ));
        }
        assert_eq!(workspace.requested.len(), 2);
        assert_eq!(workspace.targets.len(), 2);
        assert_eq!(workspace.groups.len(), 2);
        assert_eq!(workspace.scanned_lists, 2);
        assert_eq!(adjacency.total_edge_count(), 2);
        assert_eq!(adjacency.active_edge_count(), 2);
        assert!(
            adjacency
                .lists
                .read()
                .values()
                .all(|list| list.deleted.is_empty())
        );
    }

    #[test]
    fn adjacency_commit_rejects_duplicate_physical_membership() {
        let adjacency = ChunkedAdjacency::new();
        let source = NodeId::new(1);
        let edge = EdgeId::new(10);
        adjacency.add_edge(source, NodeId::new(2), edge);
        adjacency.add_edge(source, NodeId::new(3), edge);
        let history = adjacency.edges_from_including_deleted(source);
        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, edge), (source, edge)]);
        assert!(adjacency.prepare_commit_deletes(&mut workspace).is_err());
        assert_eq!(adjacency.edges_from_including_deleted(source), history);
        assert_eq!(adjacency.active_edge_count(), 2);
        assert!(adjacency.lists.read()[&source].deleted.is_empty());
    }

    #[test]
    fn adjacency_commit_rejects_missing_and_invalid_targets() {
        let adjacency = ChunkedAdjacency::new();
        adjacency.add_edge(NodeId::new(1), NodeId::new(2), EdgeId::new(10));
        for request in [
            (NodeId::INVALID, EdgeId::new(10)),
            (NodeId::new(1), EdgeId::INVALID),
            (NodeId::new(9), EdgeId::new(10)),
        ] {
            let mut workspace = AdjacencyCommitWorkspace::new(vec![request]);
            assert!(adjacency.prepare_commit_deletes(&mut workspace).is_err());
            assert!(adjacency.lists.try_write().is_some());
            assert_eq!(adjacency.active_edge_count(), 1);
        }
    }

    #[test]
    fn adjacency_commit_abandon_and_install_release_only_the_writer() -> TestResult {
        let adjacency = ChunkedAdjacency::new();
        let source = NodeId::new(1);
        let edge = EdgeId::new(10);
        adjacency.add_edge(source, NodeId::new(2), edge);
        let mut abandoned = AdjacencyCommitWorkspace::new(vec![(source, edge)]);
        let ready = adjacency.prepare_commit_deletes(&mut abandoned)?;
        assert!(adjacency.lists.try_read().is_none());
        drop(ready);
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(abandoned.targets.len(), 1);
        assert_eq!(adjacency.active_edge_count(), 1);
        assert!(adjacency.prepare_commit_deletes(&mut abandoned).is_err());

        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, edge)]);
        let installed = adjacency.prepare_commit_deletes(&mut workspace)?.install();
        assert!(adjacency.lists.try_read().is_none());
        drop(installed);
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(workspace.targets.len(), 1);
        assert_eq!(adjacency.active_edge_count(), 0);
        assert_eq!(adjacency.total_edge_count(), 1);
        Ok(())
    }

    #[test]
    fn adjacency_commit_foreign_transition_cannot_release_for_rebind() -> TestResult {
        let adjacency = ChunkedAdjacency::new();
        let source = NodeId::new(1);
        let edge = EdgeId::new(10);
        adjacency.add_edge(source, NodeId::new(2), edge);
        let store = crate::graph::lpg::LpgStore::new()?;
        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, edge)]);
        let transition = store
            .pin_exclusive_unframed_transition()
            .ok_or("standalone store transition is denied")?;
        let ready = adjacency.prepare_commit_deletes(&mut workspace)?;
        assert!(matches!(
            ready.release(&transition),
            Err(DataRebindError::Invalid(
                "transition does not pin this adjacency"
            ))
        ));
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(adjacency.active_edge_count(), 1);
        assert_eq!(workspace.targets.len(), 1);
        Ok(())
    }

    #[test]
    fn adjacency_final_list_contention_rejects_without_allocator_traffic() {
        let store = crate::graph::lpg::LpgStore::new().unwrap();
        let source = store.create_node(&[]);
        let destination = store.create_node(&[]);
        let edge = store.create_edge(source, destination, "LINK");
        let transition = store.pin_exclusive_unframed_transition().unwrap();
        let adjacency = transition.adjacency_for_test();
        let physical = adjacency.edges_from_including_deleted(source);
        let active = adjacency.edges_from(source);
        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, edge)]);
        let released = adjacency
            .prepare_commit_deletes(&mut workspace)
            .unwrap()
            .release(&transition)
            .unwrap();
        let blocker = adjacency.lists.read();
        crate::allocation_test::start();
        let result = released.rebind();
        let conflict = matches!(result, Err(DataRebindError::Conflict(_)));
        drop(result);
        let traffic = crate::allocation_test::stop();
        assert!(conflict);
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert!(blocker[&source].deleted.is_empty());
        drop(blocker);
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(adjacency.total_edge_count(), 1);
        assert_eq!(adjacency.active_edge_count(), 1);
        assert_eq!(adjacency.edges_from(source), active);
        assert_eq!(adjacency.edges_from_including_deleted(source), physical);
        let mut retry = AdjacencyCommitWorkspace::new(vec![(source, edge)]);
        let released = adjacency
            .prepare_commit_deletes(&mut retry)
            .unwrap()
            .release(&transition)
            .unwrap();
        crate::allocation_test::start();
        drop(released.rebind().unwrap());
        assert_eq!(
            crate::allocation_test::stop(),
            crate::allocation_test::Counts::default()
        );
        assert!(adjacency.lists.try_write().is_some());
        assert_eq!(adjacency.active_edge_count(), 1);
        assert_eq!(adjacency.edges_from_including_deleted(source), physical);
    }

    #[test]
    fn adjacency_commit_counts_reject_overflow_before_tombstone_install() {
        let adjacency = ChunkedAdjacency::new();
        let source = NodeId::new(1);
        let edge = EdgeId::new(10);
        adjacency.add_edge(source, NodeId::new(2), edge);
        adjacency.deleted_count.store(usize::MAX, Ordering::Relaxed);
        let mut workspace = AdjacencyCommitWorkspace::new(vec![(source, edge)]);
        assert!(adjacency.prepare_commit_deletes(&mut workspace).is_err());
        assert_eq!(adjacency.deleted_count.load(Ordering::Relaxed), usize::MAX);
        assert!(adjacency.lists.read()[&source].deleted.is_empty());
    }

    #[test]
    fn adjacency_commit_scans_only_affected_lists_once() -> TestResult {
        let adjacency = ChunkedAdjacency::new();
        for source in 0..100 {
            for edge in 0..8 {
                adjacency.add_edge(
                    NodeId::new(source),
                    NodeId::new(999),
                    EdgeId::new(source * 10 + edge),
                );
            }
        }
        let mut workspace = AdjacencyCommitWorkspace::new(vec![
            (NodeId::new(3), EdgeId::new(31)),
            (NodeId::new(3), EdgeId::new(32)),
            (NodeId::new(3), EdgeId::new(31)),
            (NodeId::new(7), EdgeId::new(73)),
        ]);
        drop(adjacency.prepare_commit_deletes(&mut workspace)?.install());
        assert_eq!(workspace.scanned_lists, 2);
        assert_eq!(adjacency.total_edge_count(), 800);
        assert_eq!(adjacency.active_edge_count(), 797);
        assert_eq!(adjacency.out_degree(NodeId::new(4)), 8);
        Ok(())
    }
}
