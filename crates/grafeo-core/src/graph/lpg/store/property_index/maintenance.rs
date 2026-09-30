//! Sparse, prepared publication of retained property memberships.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::hash::BuildHasher;

use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, HashableValue, NodeId, Value};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use hashbrown::hash_map::{Entry, RawEntryMut};

use super::interval::PreparedInterval;
use super::key::{PropertyOrderKey, keys_for_value_fixed};
use super::ordered::PreparedKey;
use super::versioned::PreparedVersion;
use super::{HistoryBucket, Interval, PropertyHistory};

type Change = (NodeId, Option<Value>, Option<Value>);

#[derive(Clone, Copy)]
enum Operation {
    Close,
    Open,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MemberState {
    len: usize,
    last: Option<Interval>,
}

impl MemberState {
    fn capture(intervals: &[Interval]) -> Self {
        Self {
            len: intervals.len(),
            last: intervals.last().copied(),
        }
    }
}

struct MemberPlan {
    id: NodeId,
    operation: Operation,
    expected: Option<MemberState>,
    candidate: Option<Vec<Interval>>,
    interval_node: Option<PreparedInterval>,
}

struct BucketPlan {
    key: HashableValue,
    hash: u64,
    was_present: bool,
    open_count: usize,
    members: Vec<MemberPlan>,
    new_members: usize,
    candidate: Option<(HashableValue, HistoryBucket)>,
    // Insert returns unused duplicates to the same slots. Every allocation
    // therefore remains owned until the aggregate's final guards have drained.
    ordered: Vec<Option<PreparedKey<PropertyOrderKey>>>,
}

/// Owns all detached buckets, member vectors and routing nodes. Preparation
/// changes live capacities only; dropping this workspace before install aborts
/// without changing membership. After install, keep the workspace alive until
/// the parent's registry, history and publication guards have all drained.
pub(in crate::graph::lpg::store) struct PreparedHistory {
    floor: EpochId,
    epoch: EpochId,
    hasher_fingerprint: u64,
    new_buckets: usize,
    buckets: Vec<BucketPlan>,
    roots: PreparedVersion<PropertyOrderKey>,
    installed: bool,
}

fn invalid(reason: &str) -> Error {
    Error::Transaction(TransactionError::WriteConflict(reason.into()))
}

/// Validation failures cannot allocate an explanatory String under publication.
fn conflict() -> Error {
    Error::Transaction(TransactionError::Conflict)
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<()> {
    values
        .try_reserve(additional)
        .map_err(|_| AllocError::OutOfMemory.into())
}

fn key(value: &Option<Value>) -> Option<HashableValue> {
    value
        .as_ref()
        .filter(|value| !value.is_null())
        .map(|value| HashableValue::new(value.clone()))
}

fn new_intervals(epoch: EpochId) -> Result<Vec<Interval>> {
    let mut intervals = Vec::new();
    intervals
        .try_reserve_exact(1)
        .map_err(|_| AllocError::OutOfMemory)?;
    intervals.push(Interval {
        from: epoch,
        to: None,
    });
    Ok(intervals)
}

fn valid_predecessor(state: Option<MemberState>, operation: Operation, epoch: EpochId) -> bool {
    match operation {
        Operation::Close => state
            .and_then(|state| state.last)
            .is_some_and(|last| last.to.is_none() && last.from <= epoch),
        Operation::Open => state
            .and_then(|state| state.last)
            .is_none_or(|last| last.to.is_some_and(|end| last.from <= end && end <= epoch)),
    }
}

impl PreparedHistory {
    /// Reserves only affected maps/vectors and allocates every absent owner.
    /// The parent supplies the committed epoch and retains the same payload
    /// identity through final validation/publication; this type has no global
    /// commit frontier and cannot independently authorize an epoch or generation.
    pub(in crate::graph::lpg::store) fn prepare(
        history: &mut PropertyHistory,
        changes: &[Change],
        epoch: EpochId,
    ) -> Result<Self> {
        if !history.roots_valid || epoch == EpochId::PENDING || epoch < history.floor {
            return Err(invalid(
                "property history requires a committed epoch at or after its floor",
            ));
        }
        let bound = changes
            .len()
            .checked_mul(2)
            .ok_or(AllocError::InsufficientSpace)?;
        let mut buckets: Vec<BucketPlan> = Vec::new();
        reserve(&mut buckets, bound)?;
        let mut lookup = FxHashMap::default();
        lookup
            .try_reserve(bound)
            .map_err(|_| AllocError::OutOfMemory)?;
        let mut seen = FxHashSet::default();
        seen.try_reserve(changes.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for (id, before, after) in changes {
            if !id.is_valid() || !seen.insert(*id) {
                return Err(invalid(
                    "property history requires unique valid node identities",
                ));
            }
            let before = key(before);
            let after = key(after);
            if before == after {
                continue;
            }
            for (key, operation) in [(before, Operation::Close), (after, Operation::Open)] {
                let Some(key) = key else {
                    continue;
                };
                let offset = if let Some(offset) = lookup.get(&key) {
                    *offset
                } else {
                    let offset = buckets.len();
                    let hash = history.rows.hasher().hash_one(&key);
                    lookup.insert(key.clone(), offset);
                    buckets.push(BucketPlan {
                        key,
                        hash,
                        was_present: false,
                        open_count: 0,
                        members: Vec::new(),
                        new_members: 0,
                        candidate: None,
                        ordered: Vec::new(),
                    });
                    offset
                };
                let members = &mut buckets[offset].members;
                reserve(members, 1)?;
                members.push(MemberPlan {
                    id: *id,
                    operation,
                    expected: None,
                    candidate: None,
                    interval_node: None,
                });
            }
        }

        let mut new_buckets = 0;
        for plan in &mut buckets {
            if let Some(bucket) = history.rows.get_mut(&plan.key) {
                plan.was_present = true;
                plan.open_count = bucket.directory.open_count();
                for member in &mut plan.members {
                    member.expected = bucket.get(&member.id).map(|v| MemberState::capture(v));
                    if !valid_predecessor(member.expected, member.operation, epoch) {
                        return Err(invalid(
                            "property history does not match the expected old membership",
                        ));
                    }
                    if let Some(last) = member.expected.and_then(|state| state.last)
                        && bucket.directory.get(&(last.from, member.id)) != Some(last)
                    {
                        return Err(invalid(
                            "property interval directory differs from member history",
                        ));
                    }
                    if let Operation::Open = member.operation {
                        if member
                            .expected
                            .and_then(|state| state.last)
                            .is_none_or(|last| last.to != Some(epoch))
                        {
                            if bucket.directory.get(&(epoch, member.id)).is_some() {
                                return Err(invalid(
                                    "property interval directory has an orphaned opening",
                                ));
                            }
                            member.interval_node = Some(PreparedInterval::new(
                                member.id,
                                Interval {
                                    from: epoch,
                                    to: None,
                                },
                            )?);
                        }
                        if let Some(intervals) = bucket.get_mut(&member.id) {
                            reserve(intervals, 1)?;
                        } else {
                            member.candidate = Some(new_intervals(epoch)?);
                            plan.new_members += 1;
                        }
                    }
                }
                bucket
                    .try_reserve(plan.new_members)
                    .map_err(|_| AllocError::OutOfMemory)?;
            } else {
                let mut bucket = HistoryBucket::default();
                bucket
                    .try_reserve(plan.members.len())
                    .map_err(|_| AllocError::OutOfMemory)?;
                for member in &plan.members {
                    if matches!(member.operation, Operation::Close) {
                        return Err(invalid("property history lacks the expected old bucket"));
                    }
                    bucket.insert(member.id, new_intervals(epoch)?);
                    drop(bucket.directory.insert_prepared(PreparedInterval::new(
                        member.id,
                        Interval {
                            from: epoch,
                            to: None,
                        },
                    )?));
                }
                let routing = keys_for_value_fixed(plan.key.inner());
                reserve(&mut plan.ordered, routing.iter().flatten().count())?;
                for key in routing.into_iter().flatten() {
                    if history.ordered.contains(&key) {
                        return Err(invalid("property history has an orphaned ordered key"));
                    }
                    plan.ordered.push(Some(PreparedKey::new(key)?));
                }
                plan.candidate = Some((plan.key.clone(), bucket));
                new_buckets += 1;
            }
        }
        history
            .rows
            .try_reserve(new_buckets)
            .map_err(|_| AllocError::OutOfMemory)?;
        let mut updates = Vec::new();
        reserve(
            &mut updates,
            bound.checked_mul(3).ok_or(AllocError::InsufficientSpace)?,
        )?;
        for plan in &buckets {
            let closed = plan
                .members
                .iter()
                .filter(|member| matches!(member.operation, Operation::Close))
                .count();
            let opened = plan.members.len() - closed;
            let final_count = plan
                .open_count
                .checked_sub(closed)
                .and_then(|count| count.checked_add(opened))
                .ok_or_else(|| {
                    invalid("property live-value count differs from prepared changes")
                })?;
            if (plan.open_count == 0) != (final_count == 0) {
                let source = history
                    .rows
                    .get_key_value(&plan.key)
                    .map_or(plan.key.inner(), |(key, _)| key.inner());
                updates.extend(
                    keys_for_value_fixed(source)
                        .into_iter()
                        .flatten()
                        .map(|key| (key, final_count != 0)),
                );
            }
        }
        let roots = history.versions.prepare(epoch, &updates)?;
        Ok(Self {
            floor: history.floor,
            epoch,
            hasher_fingerprint: history.rows.hasher().hash_one(0x70726f7068697374_u64),
            new_buckets,
            buckets,
            roots,
            installed: false,
        })
    }

    /// Checks only captured affected members plus the capacities installation
    /// will consume. This success and rejection path performs no allocations.
    pub(in crate::graph::lpg::store) fn validate(&self, history: &PropertyHistory) -> Result<()> {
        if self.installed
            || !history.roots_valid
            || !self.roots.validate(&history.versions)
            || history.floor != self.floor
            || history.rows.hasher().hash_one(0x70726f7068697374_u64) != self.hasher_fingerprint
            || history.rows.capacity().saturating_sub(history.rows.len()) < self.new_buckets
        {
            return Err(conflict());
        }
        for plan in &self.buckets {
            let bucket = history
                .rows
                .raw_entry()
                .from_hash(plan.hash, |key| key == &plan.key)
                .map(|(_, bucket)| bucket);
            if bucket.is_some() != plan.was_present {
                return Err(conflict());
            }
            if let Some(bucket) = bucket {
                if bucket.directory.open_count() != plan.open_count {
                    return Err(conflict());
                }
                if bucket.capacity().saturating_sub(bucket.len()) < plan.new_members {
                    return Err(conflict());
                }
                for member in &plan.members {
                    let intervals = bucket.get(&member.id);
                    if intervals.map(|v| MemberState::capture(v)) != member.expected {
                        return Err(conflict());
                    }
                    if let Some(last) = member.expected.and_then(|state| state.last)
                        && bucket.directory.get(&(last.from, member.id)) != Some(last)
                    {
                        return Err(conflict());
                    }
                    if let Some(node) = &member.interval_node
                        && bucket.directory.get(node.key()).is_some()
                    {
                        return Err(conflict());
                    }
                    if matches!(member.operation, Operation::Open)
                        && intervals.is_some_and(|v| v.capacity() == v.len())
                    {
                        return Err(conflict());
                    }
                }
            } else {
                if plan.candidate.is_none() {
                    return Err(conflict());
                }
                for key in plan.ordered.iter().flatten() {
                    if history.ordered.contains(key.key()) {
                        return Err(conflict());
                    }
                }
            }
        }
        Ok(())
    }

    /// Publishes a previously validated workspace under the parent's retained
    /// proof. No hashing of Value, allocation, cloning or retirement occurs.
    /// Defensive occupied-entry branches retain unused owners in this workspace.
    pub(in crate::graph::lpg::store) fn install(&mut self, history: &mut PropertyHistory) {
        if self.installed {
            return;
        }
        for plan in &mut self.buckets {
            if !plan.was_present {
                let Some((key, bucket)) = plan.candidate.take() else {
                    continue;
                };
                match history
                    .rows
                    .raw_entry_mut()
                    .from_hash(plan.hash, |key| key == &plan.key)
                {
                    RawEntryMut::Vacant(entry) => {
                        entry.insert_hashed_nocheck(plan.hash, key, bucket);
                    }
                    RawEntryMut::Occupied(_) => {
                        plan.candidate = Some((key, bucket));
                        continue;
                    }
                }
                for slot in &mut plan.ordered {
                    if let Some(key) = slot.take() {
                        *slot = history.ordered.insert_prepared(key);
                    }
                }
            } else if let RawEntryMut::Occupied(mut entry) = history
                .rows
                .raw_entry_mut()
                .from_hash(plan.hash, |key| key == &plan.key)
            {
                let bucket = entry.get_mut();
                for member in &mut plan.members {
                    if let Some(candidate) = member.candidate.take() {
                        match bucket.members.entry(member.id) {
                            Entry::Vacant(entry) => {
                                entry.insert(candidate);
                            }
                            Entry::Occupied(_) => {
                                member.candidate = Some(candidate);
                            }
                        }
                    } else if let Some(intervals) = bucket.members.get_mut(&member.id) {
                        match member.operation {
                            Operation::Close => {
                                if let Some(last) = intervals.last_mut() {
                                    last.to = Some(self.epoch);
                                }
                            }
                            Operation::Open => {
                                if let Some(last) = intervals.last_mut()
                                    && last.to == Some(self.epoch)
                                {
                                    last.to = None;
                                } else {
                                    intervals.push(Interval {
                                        from: self.epoch,
                                        to: None,
                                    });
                                }
                            }
                        }
                    }
                    match member.operation {
                        Operation::Close => {
                            if let Some(last) = member.expected.and_then(|state| state.last) {
                                bucket
                                    .directory
                                    .set_end(&(last.from, member.id), Some(self.epoch));
                            }
                        }
                        Operation::Open => {
                            if let Some(node) = member.interval_node.take() {
                                member.interval_node = bucket.directory.insert_prepared(node);
                            } else if let Some(last) = member.expected.and_then(|state| state.last)
                            {
                                bucket.directory.set_end(&(last.from, member.id), None);
                            }
                        }
                    }
                }
            }
        }
        self.roots.install(&mut history.versions);
        self.installed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PropertyIndexPredicate;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn epoch(value: u64) -> EpochId {
        EpochId::new(value)
    }

    fn add(history: &mut PropertyHistory, id: u64, value: &Value) {
        history
            .add_interval(
                NodeId::new(id),
                value,
                Interval {
                    from: epoch(1),
                    to: None,
                },
            )
            .unwrap();
        history.rebuild_value_roots().unwrap();
    }

    fn apply(
        history: &mut PropertyHistory,
        id: u64,
        before: Option<Value>,
        after: Option<Value>,
        at: u64,
    ) {
        let mut prepared =
            PreparedHistory::prepare(history, &[(NodeId::new(id), before, after)], epoch(at))
                .unwrap();
        prepared.validate(history).unwrap();
        prepared.install(history);
    }

    #[test]
    fn history_publication_and_rejection_have_no_allocator_traffic() {
        let mut history = PropertyHistory::new(epoch(0));
        let a = Value::String("alpha".into());
        let b = Value::String("bravo".into());
        let c = Value::GCounter(Arc::new(HashMap::from([("replica".to_owned(), 3)])));
        add(&mut history, 1, &a);
        add(&mut history, 2, &a);
        add(&mut history, 3, &b);
        add(&mut history, 4, &c);
        let changes = [
            (NodeId::new(1), Some(a.clone()), Some(b)),
            (NodeId::new(2), Some(a), Some(Value::String("new".into()))),
            (NodeId::new(4), Some(c.clone()), None),
            (NodeId::new(5), None, Some(c)),
        ];
        let mut prepared = PreparedHistory::prepare(&mut history, &changes, epoch(2)).unwrap();
        crate::allocation_test::start();
        let validation = prepared.validate(&history);
        prepared.install(&mut history);
        let repeated = prepared.validate(&history);
        prepared.install(&mut history); // A consumed workspace is inert.
        let traffic = crate::allocation_test::stop();
        assert!(validation.is_ok());
        assert!(repeated.is_err());
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        let query = Value::String("new".into());
        assert_eq!(
            history
                .candidates(PropertyIndexPredicate::Equal(&query), epoch(2))
                .unwrap()
                .0,
            vec![NodeId::new(2)]
        );
    }

    #[test]
    fn history_abandoned_preparation_preserves_all_logical_contents() {
        let mut history = PropertyHistory::new(epoch(0));
        add(&mut history, 1, &Value::Int64(10));
        let before = history.rows.clone();
        let ordered_len = history.ordered.len();
        let prepared = PreparedHistory::prepare(
            &mut history,
            &[(
                NodeId::new(1),
                Some(Value::Int64(10)),
                Some(Value::Int64(20)),
            )],
            epoch(2),
        )
        .unwrap();
        assert!(prepared.validate(&history).is_ok());
        drop(prepared);
        assert_eq!(history.rows, before);
        assert_eq!(history.ordered.len(), ordered_len);
    }

    #[test]
    fn history_interval_allocation_failure_preserves_logical_contents() {
        let mut history = PropertyHistory::new(epoch(0));
        add(&mut history, 1, &Value::Int64(10));
        let before = history.rows.clone();
        let ordered_len = history.ordered.len();
        let changes = [(
            NodeId::new(1),
            Some(Value::Int64(10)),
            Some(Value::Int64(20)),
        )];
        let (result, fired) = crate::allocation_test::with_failure(
            std::mem::size_of::<Interval>(),
            std::mem::align_of::<Interval>(),
            0,
            || PreparedHistory::prepare(&mut history, &changes, epoch(2)),
        );
        assert!(
            fired,
            "intercept the absent member's fallible interval allocation"
        );
        assert!(result.is_err());
        assert_eq!(history.rows, before);
        assert_eq!(history.ordered.len(), ordered_len);
    }

    #[test]
    fn history_unused_routing_nodes_remain_owned_after_defensive_duplicate() {
        let mut history = PropertyHistory::new(epoch(0));
        let value = Value::Int64(10);
        let mut prepared = PreparedHistory::prepare(
            &mut history,
            &[(NodeId::new(1), None, Some(value.clone()))],
            epoch(1),
        )
        .unwrap();
        // Deliberately invalidate only the routing proof to exercise the
        // defensive ownership branch. Real callers must stop at failed validate.
        for key in keys_for_value_fixed(&value).into_iter().flatten() {
            let _ = history
                .ordered
                .insert_prepared(PreparedKey::new(key).unwrap());
        }
        crate::allocation_test::start();
        let rejected = prepared.validate(&history);
        prepared.install(&mut history);
        let traffic = crate::allocation_test::stop();
        assert!(rejected.is_err());
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert_eq!(prepared.buckets[0].ordered.iter().flatten().count(), 1);
        crate::allocation_test::start();
        drop(prepared);
        let retirement = crate::allocation_test::stop();
        assert!(retirement.dealloc > 0);
    }

    #[test]
    fn history_a_b_a_and_adjacent_memberships_preserve_epoch_visibility() {
        let mut history = PropertyHistory::new(epoch(0));
        add(&mut history, 1, &Value::Int64(10));
        apply(
            &mut history,
            1,
            Some(Value::Int64(10)),
            Some(Value::Int64(20)),
            2,
        );
        apply(
            &mut history,
            1,
            Some(Value::Int64(20)),
            Some(Value::Int64(10)),
            3,
        );
        let key = HashableValue::new(Value::Int64(10));
        assert_eq!(
            history.rows[&key][&NodeId::new(1)],
            vec![
                Interval {
                    from: epoch(1),
                    to: Some(epoch(2))
                },
                Interval {
                    from: epoch(3),
                    to: None
                }
            ]
        );
        for (at, expected) in [(0, false), (1, true), (2, false), (3, true)] {
            assert_eq!(
                history
                    .candidates(PropertyIndexPredicate::Equal(key.inner()), epoch(at))
                    .unwrap()
                    .0
                    .contains(&NodeId::new(1)),
                expected
            );
        }
        apply(&mut history, 1, Some(Value::Int64(10)), None, 4);
        apply(&mut history, 1, None, Some(Value::Int64(10)), 4);
        assert_eq!(history.rows[&key][&NodeId::new(1)].len(), 2);
        assert_eq!(history.rows[&key][&NodeId::new(1)][1].to, None);
    }

    #[test]
    fn history_rejects_duplicates_pending_stale_inputs_and_changed_proofs() {
        let mut history = PropertyHistory::new(epoch(0));
        add(&mut history, 1, &Value::Int64(10));
        let change = (NodeId::new(1), Some(Value::Int64(10)), None);
        assert!(
            PreparedHistory::prepare(&mut history, &[change.clone(), change.clone()], epoch(2))
                .is_err()
        );
        assert!(
            PreparedHistory::prepare(
                &mut history,
                std::slice::from_ref(&change),
                EpochId::PENDING
            )
            .is_err()
        );
        assert!(
            PreparedHistory::prepare(&mut history, std::slice::from_ref(&change), epoch(0))
                .is_err()
        );
        let prepared =
            PreparedHistory::prepare(&mut history, std::slice::from_ref(&change), epoch(2))
                .unwrap();
        history.floor = epoch(1);
        assert!(prepared.validate(&history).is_err());
        history.floor = epoch(0);
        history
            .rows
            .get_mut(&HashableValue::new(Value::Int64(10)))
            .unwrap()
            .get_mut(&NodeId::new(1))
            .unwrap()[0]
            .to = Some(epoch(2));
        assert!(prepared.validate(&history).is_err());
    }

    #[test]
    fn history_rejects_changed_open_count_without_a_value_root_change() {
        let mut history = PropertyHistory::new(epoch(0));
        let value = Value::Int64(10);
        add(&mut history, 1, &value);
        let prepared = PreparedHistory::prepare(
            &mut history,
            &[(NodeId::new(1), Some(value.clone()), None)],
            epoch(3),
        )
        .unwrap();
        // Another member enters an already-live bucket: no ordered root or
        // captured member tail changes, but removing member 1 is no longer
        // allowed to remove this value from the live directory.
        apply(&mut history, 2, None, Some(value.clone()), 2);
        assert!(prepared.roots.validate(&history.versions));
        crate::allocation_test::start();
        let validation = prepared.validate(&history);
        let traffic = crate::allocation_test::stop();
        assert!(validation.is_err());
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        apply(&mut history, 1, Some(value.clone()), None, 3);
        let (ids, _) = history
            .candidates(
                PropertyIndexPredicate::Range {
                    min: Some(&value),
                    max: Some(&value),
                    min_inclusive: true,
                    max_inclusive: true,
                },
                epoch(3),
            )
            .unwrap();
        assert_eq!(ids, vec![NodeId::new(2)]);
    }

    #[test]
    fn history_null_and_identical_hash_changes_are_noops() {
        let mut history = PropertyHistory::new(epoch(0));
        add(&mut history, 1, &Value::Float64(-0.0));
        let before = history.rows.clone();
        let mut prepared = PreparedHistory::prepare(
            &mut history,
            &[
                (
                    NodeId::new(1),
                    Some(Value::Float64(-0.0)),
                    Some(Value::Float64(0.0)),
                ),
                (NodeId::new(2), Some(Value::Null), None),
            ],
            epoch(2),
        )
        .unwrap();
        prepared.validate(&history).unwrap();
        prepared.install(&mut history);
        assert_eq!(history.rows, before);
    }

    #[test]
    fn history_rebind_checks_interval_directory_and_installs_endpoints_without_allocation() {
        let mut history = PropertyHistory::new(epoch(0));
        let a = Value::Int64(10);
        let b = Value::Int64(20);
        let id = NodeId::new(1);
        add(&mut history, 1, &a);
        let mut prepared = PreparedHistory::prepare(
            &mut history,
            &[(id, Some(a.clone()), Some(b.clone()))],
            epoch(2),
        )
        .unwrap();
        let a_key = HashableValue::new(a.clone());
        history
            .rows
            .get_mut(&a_key)
            .unwrap()
            .directory
            .set_end(&(epoch(1), id), Some(epoch(9)));
        crate::allocation_test::start();
        let rejected = prepared.validate(&history);
        let traffic = crate::allocation_test::stop();
        assert!(
            rejected.is_err(),
            "unchanged member vectors cannot conceal a changed interval directory"
        );
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        history
            .rows
            .get_mut(&a_key)
            .unwrap()
            .directory
            .set_end(&(epoch(1), id), None);
        crate::allocation_test::start();
        let validated = prepared.validate(&history);
        prepared.install(&mut history);
        let traffic = crate::allocation_test::stop();
        assert!(validated.is_ok());
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert_eq!(
            history
                .candidates(PropertyIndexPredicate::Equal(&a), epoch(1))
                .unwrap()
                .0,
            vec![id]
        );
        assert!(
            history
                .candidates(PropertyIndexPredicate::Equal(&a), epoch(2))
                .unwrap()
                .0
                .is_empty()
        );
        assert_eq!(
            history
                .candidates(PropertyIndexPredicate::Equal(&b), epoch(2))
                .unwrap()
                .0,
            vec![id]
        );
    }
}
