//! Sparse membership maintenance beneath the paired registry/transition proof.
//!
//! Property rows are private to the store: public queries retain the registry
//! reader, and mutations retain the store mutation pin. The aggregate's exact
//! transition and final registry writer therefore exclude both. Do not expose
//! these helpers as an independent row-mutation capability.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::super::property_index::PreparedHistory;
use super::{DataRebindError, PropertyIndexRows, conflict, reservation, reserve_vec};
#[cfg(feature = "compact-store")]
use crate::graph::compact::layered::commit::LayeredCommitPin;
#[cfg(feature = "compact-store")]
use crate::graph::lpg::PinnedLpgTransition;
use dashmap::SharedValue;
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, HashableValue, NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::hash::BuildHasher;
use std::sync::Arc;

type Change = (NodeId, Option<Value>, Option<Value>);
type Membership = FxHashSet<NodeId>;
type Row = (HashableValue, SharedValue<Membership>);

/// Owns all input, delta, vacant-bucket and retired-bucket allocations.
pub(super) struct PropertyMaintenanceWorkspace {
    changes: Vec<Change>,
    history: Option<PreparedHistory>,
    commit_epoch: Option<EpochId>,
    anchor: Option<Arc<PropertyIndexRows>>,
    buckets: Vec<BucketDelta>,
    // Normalization storage remains outside every final writer, too.
    lookup: FxHashMap<HashableValue, usize>,
    seen: FxHashSet<NodeId>,
    shards: Vec<(usize, usize)>,
    // Logical old values omitted from a genuinely unhydrated cold physical
    // index remain owned here until the complete aggregate scope has drained.
    #[cfg(feature = "compact-store")]
    unhydrated_before: Vec<HashableValue>,
    #[cfg(feature = "compact-store")]
    cold_normalization_attempted: bool,
    attempted: bool,
    prepared: bool,
}

struct BucketDelta {
    key: HashableValue,
    hash: u64,
    shard: usize,
    adds: Vec<NodeId>,
    removes: Vec<NodeId>,
    was_present: bool,
    candidate: Option<Row>,
    retired: Option<Row>,
}

impl PropertyMaintenanceWorkspace {
    pub(super) fn new(changes: Vec<Change>) -> Self {
        Self {
            changes,
            history: None,
            commit_epoch: None,
            anchor: None,
            buckets: Vec::new(),
            lookup: FxHashMap::default(),
            seen: FxHashSet::default(),
            shards: Vec::new(),
            #[cfg(feature = "compact-store")]
            unhydrated_before: Vec::new(),
            #[cfg(feature = "compact-store")]
            cold_normalization_attempted: false,
            attempted: false,
            prepared: false,
        }
    }

    pub(super) fn at_epoch(changes: Vec<Change>, epoch: EpochId) -> Self {
        let mut workspace = Self::new(changes);
        workspace.commit_epoch = Some(epoch);
        workspace
    }

    pub(super) fn prepare_at(
        &mut self,
        rows: &Arc<PropertyIndexRows>,
        epoch: EpochId,
    ) -> Result<()> {
        if self.commit_epoch.is_none() {
            self.commit_epoch = Some(epoch);
        }
        self.prepare(rows)
    }

    /// Converts only proven cold/unhydrated logical predecessors to their
    /// exact physical predecessor. Existing memberships, native/hot rows and
    /// ordinary validation remain unchanged; no live index row is modified.
    #[cfg(feature = "compact-store")]
    pub(super) fn normalize_unhydrated_cold_before(
        &mut self,
        rows: &Arc<PropertyIndexRows>,
        pin: &LayeredCommitPin<'_>,
        transition: &PinnedLpgTransition<'_>,
    ) -> Result<()> {
        if self.attempted || self.cold_normalization_attempted {
            return Err(conflict("Property cold-row normalization is one-shot"));
        }
        self.cold_normalization_attempted = true;
        reserve_vec(&mut self.unhydrated_before, self.changes.len())?;
        for (id, before, after) in &mut self.changes {
            if before.as_ref().is_none_or(Value::is_null)
                // Match prepare's index-key equality, including canonical
                // floating-point values. A logical no-op must remain a no-op
                // even when a transferred cold index has no physical row.
                || before.as_ref().zip(after.as_ref()).is_some_and(|(old, new)| {
                    HashableValue::new(old.clone()) == HashableValue::new(new.clone())
                })
                || !pin
                    .is_unhydrated_cold_node(transition, *id)
                    .map_err(DataRebindError::into_error)?
            {
                continue;
            }
            if let Some(value) = before.take() {
                // Move, rather than clone/drop, even compound hashable values.
                // The reserved outer Vec retains an omitted predecessor on
                // every later preparation failure and through installation.
                self.unhydrated_before.push(HashableValue::new(value));
                if self
                    .unhydrated_before
                    .last()
                    .is_some_and(|key| rows.get(key).is_some_and(|members| members.contains(id)))
                {
                    *before = self.unhydrated_before.pop().map(Value::from);
                }
            }
        }
        Ok(())
    }

    pub(super) fn prepare(&mut self, rows: &Arc<PropertyIndexRows>) -> Result<()> {
        if self.attempted {
            return Err(conflict(
                "Property maintenance preparation was already attempted",
            ));
        }
        self.attempted = true;
        self.anchor = Some(Arc::clone(rows));
        let bound = self
            .changes
            .len()
            .checked_mul(2)
            .ok_or(AllocError::InsufficientSpace)?;
        reserve_vec(&mut self.buckets, bound)?;
        reservation()?;
        self.lookup
            .try_reserve(bound)
            .map_err(|_| AllocError::OutOfMemory)?;
        reservation()?;
        self.seen
            .try_reserve(self.changes.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        reserve_vec(&mut self.shards, bound)?;
        for offset in 0..self.changes.len() {
            let (id, old, new) = &self.changes[offset];
            let id = *id;
            if !id.is_valid() || !self.seen.insert(id) {
                return Err(conflict(
                    "Property maintenance requires unique valid node identities",
                ));
            }
            let old = old
                .as_ref()
                .filter(|v| !matches!(v, Value::Null))
                .map(|v| HashableValue::new(v.clone()));
            let new = new
                .as_ref()
                .filter(|v| !matches!(v, Value::Null))
                .map(|v| HashableValue::new(v.clone()));
            if old == new {
                continue;
            }
            if let Some(key) = old {
                let bucket = self.bucket(key)?;
                reserve_vec(&mut bucket.removes, 1)?;
                bucket.removes.push(id);
            }
            if let Some(key) = new {
                let bucket = self.bucket(key)?;
                reserve_vec(&mut bucket.adds, 1)?;
                bucket.adds.push(id);
            }
        }

        for bucket in &mut self.buckets {
            // Some Value hashes (including nested counter values) allocate.
            // Resolve the exact table hash and shard before final writers.
            bucket.hash = rows.hasher().hash_one(&bucket.key);
            bucket.shard = rows.determine_map(&bucket.key);
            if let Some(mut members) = rows.get_mut(&bucket.key) {
                bucket.was_present = true;
                validate_members(&members, bucket).map_err(DataRebindError::into_error)?;
                // Reserve additions without relying on removals to free slots:
                // occupied HashSet inserts must never trigger a rehash at C.
                reservation()?;
                members
                    .try_reserve(bucket.adds.len())
                    .map_err(|_| AllocError::OutOfMemory)?;
            } else {
                if !bucket.removes.is_empty() {
                    return Err(conflict("Property index lacks expected old membership"));
                }
                bucket.candidate =
                    Some((bucket.key.clone(), SharedValue::new(Membership::default())));
                if let Some((_, members)) = &mut bucket.candidate {
                    reservation()?;
                    members
                        .get_mut()
                        .try_reserve(bucket.adds.len())
                        .map_err(|_| AllocError::OutOfMemory)?;
                    members.get_mut().extend(bucket.adds.iter().copied());
                }
                self.shards.push((bucket.shard, 1));
            }
        }
        self.shards.sort_unstable_by_key(|&(shard, _)| shard);
        let mut count = 0;
        for read in 0..self.shards.len() {
            let (shard, missing) = self.shards[read];
            if count > 0 && self.shards[count - 1].0 == shard {
                self.shards[count - 1].1 = self.shards[count - 1]
                    .1
                    .checked_add(missing)
                    .ok_or(AllocError::InsufficientSpace)?;
            } else {
                self.shards[count] = (shard, missing);
                count += 1;
            }
        }
        self.shards.truncate(count);
        for &(shard, missing) in &self.shards {
            // DashMap::try_reserve requires unique &mut ownership and reserves
            // the same amount in EVERY shard. Its safe raw API preserves the
            // existing Arc and sharding while reserving only affected shards.
            reservation()?;
            rows.shards()[shard]
                .write()
                .try_reserve(missing, |(key, _)| rows.hasher().hash_one(key))
                .map_err(|_| AllocError::OutOfMemory)?;
        }
        self.history = Some(PreparedHistory::prepare(
            &mut rows.history.write(),
            &self.changes,
            self.commit_epoch.unwrap_or(EpochId::INITIAL),
        )?);
        self.prepared = true;
        Ok(())
    }

    fn bucket(&mut self, key: HashableValue) -> Result<&mut BucketDelta> {
        let offset = match self.lookup.get(&key) {
            Some(&offset) => offset,
            None => {
                let offset = self.buckets.len();
                self.buckets.push(BucketDelta {
                    key: key.clone(),
                    hash: 0,
                    shard: 0,
                    adds: Vec::new(),
                    removes: Vec::new(),
                    was_present: false,
                    candidate: None,
                    retired: None,
                });
                self.lookup.insert(key, offset);
                offset
            }
        };
        self.buckets
            .get_mut(offset)
            .ok_or_else(|| conflict("Property maintenance bucket identity was lost"))
    }

    pub(super) fn validate(
        &self,
        rows: &Arc<PropertyIndexRows>,
    ) -> std::result::Result<(), DataRebindError> {
        if !self.prepared
            || !self
                .anchor
                .as_ref()
                .is_some_and(|anchor| Arc::ptr_eq(anchor, rows))
        {
            return Err(DataRebindError::new(
                "Property maintenance target is not its prepared registration",
            ));
        }
        for bucket in &self.buckets {
            let shard = rows.shards()[bucket.shard].read();
            match (
                bucket.was_present,
                shard.get(bucket.hash, |(key, _)| key == &bucket.key),
            ) {
                (true, Some((_, members))) => {
                    let members = members.get();
                    validate_members(members, bucket)?;
                    if members.capacity().saturating_sub(members.len()) < bucket.adds.len() {
                        return Err(DataRebindError::new(
                            "Property membership capacity changed after preparation",
                        ));
                    }
                }
                (false, None) if bucket.candidate.is_some() => {}
                _ => {
                    return Err(DataRebindError::Conflict(
                        "Property membership bucket changed after preparation",
                    ));
                }
            }
        }
        for &(shard, missing) in &self.shards {
            let rows = rows.shards()[shard].read();
            if rows.capacity().saturating_sub(rows.len()) < missing {
                return Err(DataRebindError::new(
                    "Property index shard capacity changed after preparation",
                ));
            }
        }
        let history = self.history.as_ref().ok_or(DataRebindError::new(
            "Property history maintenance was not prepared",
        ))?;
        history.validate(&rows.history.read()).map_err(|_| {
            DataRebindError::Conflict("Property historical membership changed after preparation")
        })?;
        Ok(())
    }

    /// Called only after validate under the continuously retained registry
    /// writer and exact store transition. No ordinary index callbacks run.
    pub(super) fn install(&mut self, rows: &Arc<PropertyIndexRows>) {
        if let Some(history) = &mut self.history {
            history.install(&mut rows.history.write());
        }
        for bucket in &mut self.buckets {
            let hash = bucket.hash;
            let mut shard = rows.shards()[bucket.shard].write();
            if bucket.was_present {
                let empty = if let Some((_, members)) =
                    shard.get_mut(hash, |(key, _)| key == &bucket.key)
                {
                    let members = members.get_mut();
                    for id in &bucket.removes {
                        members.remove(id);
                    }
                    for id in &bucket.adds {
                        members.insert(*id);
                    }
                    members.is_empty()
                } else {
                    false
                };
                if empty {
                    bucket.retired = shard.remove_entry(hash, |(key, _)| key == &bucket.key);
                }
            } else if let Some(candidate) = bucket.candidate.take() {
                // In hashbrown 0.14, capacity - len is growth_left. The
                // retained registry/transition proof and reserved missing-key
                // count keep it positive before every insert. Removals cannot
                // consume growth_left, so insert cannot invoke this rehash
                // callback (which may allocate for counter-valued keys).
                shard.insert(hash, candidate, |(key, _)| rows.hasher().hash_one(key));
            }
        }
    }
}

fn validate_members(
    members: &Membership,
    bucket: &BucketDelta,
) -> std::result::Result<(), DataRebindError> {
    if bucket.removes.iter().any(|id| !members.contains(id))
        || bucket.adds.iter().any(|id| members.contains(id))
    {
        return Err(DataRebindError::Conflict(
            "Property index differs from the qualified old/final membership",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
