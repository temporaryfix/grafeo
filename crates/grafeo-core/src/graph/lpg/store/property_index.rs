//! One registered property payload: current postings and retained ordered history.
//!
//! The current map remains the compatibility view for existing low-level exact
//! lookups. Query lookups use the retained postings. Both belong to the same
//! registration and are prepared and published by the existing index aggregate.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

mod interval;
mod key;
pub(super) mod maintenance;
mod ordered;
mod source;
mod versioned;
pub(super) use maintenance::PreparedHistory;

use std::ops::{Deref, DerefMut};

use dashmap::DashMap;
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, HashableValue, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;

use super::LpgStore;
use crate::execution::operators::ExpressionPredicate;
use crate::graph::{PropertyIndexPredicate, PropertyIndexRequest};
use interval::{IntervalDirectory, PreparedInterval};
use key::{PropertyOrderKey, keys_for_value, routes};
use ordered::{OrderedDirectory, PreparedKey};
use versioned::VersionedDirectory;

/// Detached rebuild input. Null history entries close a preceding membership.
/// Current rows include only structurally visible identities; history also
/// includes deleted identities whose versions remain retained.
pub struct PropertyIndexImage {
    /// Current compatibility postings.
    pub current: Vec<(NodeId, Value)>,
    /// Committed property events in ascending epoch order per identity.
    pub history: Vec<(NodeId, Vec<(EpochId, Value)>)>,
    /// Earliest admitted historical view.
    pub floor: EpochId,
}

impl PropertyIndexImage {
    /// Applies a complete final structural population to detached rebuild input.
    /// Existing retained events survive; removed values receive a closing event.
    ///
    /// # Errors
    /// Returns an error for an invalid publication epoch, regressive history,
    /// or a failed allocation while preparing detached contents.
    pub fn project_final_rows(
        &mut self,
        rows: &[crate::graph::lpg::Node],
        property: &str,
        epoch: EpochId,
    ) -> Result<()> {
        if epoch == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "property index rebuild needs a committed epoch".into(),
            ));
        }
        let mut final_values = FxHashMap::default();
        final_values
            .try_reserve(rows.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for node in rows {
            if let Some(value) = node.get_property(property).filter(|value| !value.is_null()) {
                final_values.insert(node.id, value.clone());
            }
        }
        self.current = final_values
            .iter()
            .map(|(id, value)| (*id, value.clone()))
            .collect();
        for (id, events) in &mut self.history {
            let value = final_values.remove(id).unwrap_or(Value::Null);
            if events.last().is_some_and(|(last, _)| *last > epoch) {
                return Err(Error::InvalidValue(
                    "property index rebuild precedes source history".into(),
                ));
            }
            if events.last().is_none_or(|(_, old)| *old != value) {
                events.try_reserve(1).map_err(|_| AllocError::OutOfMemory)?;
                events.push((epoch, value));
            }
        }
        self.history
            .try_reserve(final_values.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for (id, value) in final_values {
            self.history.push((id, vec![(epoch, value)]));
        }
        self.history.sort_unstable_by_key(|(id, _)| *id);
        Ok(())
    }
}

pub(super) struct PropertyIndexRows {
    current: DashMap<HashableValue, FxHashSet<NodeId>>,
    pub(super) history: RwLock<PropertyHistory>,
}

impl PropertyIndexRows {
    pub(super) fn new() -> Self {
        Self {
            current: DashMap::new(),
            history: RwLock::new(PropertyHistory::new(EpochId::INITIAL)),
        }
    }

    /// Builds detached contents before acquiring a publication writer.
    pub(super) fn from_image(image: PropertyIndexImage) -> Result<Self> {
        let mut result = Self::new();
        result
            .current
            .try_reserve(image.current.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for (id, value) in image.current {
            if !value.is_null() {
                let mut bucket = result.current.entry(HashableValue::new(value)).or_default();
                bucket.try_reserve(1).map_err(|_| AllocError::OutOfMemory)?;
                bucket.insert(id);
            }
        }
        let history = result.history.get_mut();
        history.floor = image.floor;
        for (id, events) in image.history {
            if events.windows(2).any(|pair| pair[0].0 > pair[1].0)
                || events.iter().any(|(epoch, _)| *epoch == EpochId::PENDING)
            {
                return Err(Error::InvalidValue(
                    "property index history is not committed and ordered".into(),
                ));
            }
            for (offset, (from, value)) in events.iter().enumerate() {
                let to = events.get(offset + 1).map(|(epoch, _)| *epoch);
                // Multiple writes at the same epoch have only their final image.
                if value.is_null() || to == Some(*from) || to.is_some_and(|end| end <= image.floor)
                {
                    continue;
                }
                history.add_interval(id, value, Interval { from: *from, to })?;
            }
        }
        history.rebuild_value_roots()?;
        Ok(result)
    }

    pub(super) fn seed_current_history(&mut self, epoch: EpochId) -> Result<()> {
        let history = self.history.get_mut();
        history.floor = epoch;
        for bucket in &self.current {
            for id in bucket.value() {
                history.add_interval(
                    *id,
                    bucket.key().inner(),
                    Interval {
                        from: epoch,
                        to: None,
                    },
                )?;
            }
        }
        history.rebuild_value_roots()
    }
}

impl Deref for PropertyIndexRows {
    type Target = DashMap<HashableValue, FxHashSet<NodeId>>;

    fn deref(&self) -> &Self::Target {
        &self.current
    }
}

impl DerefMut for PropertyIndexRows {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.current
    }
}

impl LpgStore {
    /// Finalizes only the legacy undo-log path. Buffered engine commits have
    /// no entries here and publish history through PreparedHistory instead.
    pub(super) fn finalize_property_index_history(&self, tx: TransactionId, epoch: EpochId) {
        let undo = self.property_undo_log.read();
        let Some(entries) = undo.get(&tx) else {
            return;
        };
        let mut seen = FxHashSet::default();
        let indexes = self.property_indexes.read();
        for entry in entries {
            if let super::PropertyUndoEntry::NodeProperty {
                node_id,
                key,
                old_value,
            } = entry
                && seen.insert((*node_id, key.clone()))
                && let Some(index) = indexes.get(key)
            {
                let after = self.node_properties.get(*node_id, key);
                index.history.write().record_direct(
                    *node_id,
                    old_value.as_ref(),
                    after.as_ref(),
                    epoch,
                );
            }
        }
    }

    /// Acquires candidates and reconciles the transaction overlay under one
    /// physical generation pin. Registered history is queried at the supplied
    /// epoch; a concurrent newer posting cannot erase an older membership.
    ///
    /// # Errors
    /// Returns an error when the requested epoch predates retained coverage
    /// or candidate preparation cannot allocate its result.
    pub fn lookup_nodes_indexed(
        &self,
        request: PropertyIndexRequest<'_>,
    ) -> Result<Option<Vec<NodeId>>> {
        let _read = self.pin_read();
        let Some(mut candidates) = self.lookup_nodes_indexed_candidates(request)? else {
            return Ok(None);
        };
        let property = PropertyKey::new(request.property);
        candidates.retain(|id| {
            let local = self.contains_node_identity(*id);
            let visible = if local {
                match request.transaction_id {
                    Some(tx) => self.get_node_versioned(*id, request.epoch, tx).is_some(),
                    None => self.get_node_at_epoch(*id, request.epoch).is_some(),
                }
            } else {
                #[cfg(feature = "compact-store")]
                {
                    use crate::graph::GraphStore;
                    self.property_index_compact_base().is_some_and(|base| {
                        match request.transaction_id {
                            Some(tx) => base.get_node_versioned(*id, request.epoch, tx).is_some(),
                            None => base.get_node_at_epoch(*id, request.epoch).is_some(),
                        }
                    })
                }
                #[cfg(not(feature = "compact-store"))]
                {
                    false
                }
            };
            if !visible {
                return false;
            }
            let value = self.read_node_property_visible(
                *id,
                &property,
                request.epoch,
                request.transaction_id,
            );
            #[cfg(feature = "compact-store")]
            let value = value.or_else(|| {
                use crate::graph::GraphStore;
                if local {
                    return None;
                }
                self.property_index_compact_base().and_then(|base| {
                    base.read_node_property_visible(
                        *id,
                        &property,
                        request.epoch,
                        request.transaction_id,
                    )
                })
            });
            value.is_some_and(|value| {
                ExpressionPredicate::matches_property_index_predicate(&value, request.predicate)
            })
        });
        Ok(Some(candidates))
    }

    /// Retrieves property candidates only. A Layered caller must apply its
    /// generation-aware visibility and residual checks while retaining its
    /// publication pin; private hydration cannot determine logical visibility.
    pub(crate) fn lookup_nodes_indexed_candidates(
        &self,
        request: PropertyIndexRequest<'_>,
    ) -> Result<Option<Vec<NodeId>>> {
        let _read = self.pin_read();
        let tracker = request.transaction_id.and_then(|tx| {
            self.read_trackers
                .read()
                .get(&tx)
                .cloned()
                .map(|tracker| (tx, tracker))
        });
        if let Some((tx, tracker)) = tracker {
            // Record before empty results and unsupported-index fallbacks.
            // The property predicate outlives a concurrent registry drop.
            tracker.record_property_index_read(tx, request.property);
        }
        let scalar =
            |value: &Value| !matches!(value, Value::List(_) | Value::Map(_) | Value::Path { .. });
        match request.predicate {
            PropertyIndexPredicate::Equal(value) if !scalar(value) => return Ok(None),
            PropertyIndexPredicate::In(values) if values.iter().any(|value| !scalar(value)) => {
                return Ok(None);
            }
            PropertyIndexPredicate::Range {
                min: None,
                max: None,
                ..
            } => return Ok(None),
            _ => {}
        }
        if request.epoch < self.retained_history_floor() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Unsupported,
                "property index view predates retained store history",
            )));
        }
        let property = PropertyKey::new(request.property);
        let indexes = self.property_indexes.read();
        let Some(index) = indexes.get(&property) else {
            return Ok(None);
        };
        let (mut candidates, (visited_keys, posting_ids, posting_intervals)) = index
            .history
            .read()
            .candidates(request.predicate, request.epoch)?;
        self.work_counters
            .record_property_index_route_keys(visited_keys);
        self.work_counters
            .record_property_index_postings(posting_ids, posting_intervals);
        if let Some(tx) = request.transaction_id {
            candidates.extend(self.nodes_with_buffered_property(tx, &property));
        }
        candidates.sort_unstable();
        candidates.dedup();
        Ok(Some(candidates))
    }

    /// Point membership at the retained epoch, including this writer's delta.
    pub fn node_has_label_at_epoch(
        &self,
        id: NodeId,
        label: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.contains_node_identity(id) {
            return self
                .read_node_labels_visible(id, epoch, Some(transaction_id))
                .contains(label);
        }
        #[cfg(feature = "compact-store")]
        {
            use crate::graph::GraphStore;
            self.property_index_compact_base()
                .is_some_and(|base| base.node_has_label_at_epoch(id, label, epoch, transaction_id))
        }
        #[cfg(not(feature = "compact-store"))]
        {
            false
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Interval {
    pub(super) from: EpochId,
    pub(super) to: Option<EpochId>,
}

impl Interval {
    #[cfg(test)]
    fn contains(self, epoch: EpochId) -> bool {
        self.from <= epoch && self.to.is_none_or(|end| epoch < end)
    }
}

#[derive(Default)]
#[cfg_attr(test, derive(Clone, Debug, PartialEq, Eq))]
pub(super) struct HistoryBucket {
    members: FxHashMap<NodeId, Vec<Interval>>,
    directory: IntervalDirectory,
}

impl Deref for HistoryBucket {
    type Target = FxHashMap<NodeId, Vec<Interval>>;
    fn deref(&self) -> &Self::Target {
        &self.members
    }
}

impl DerefMut for HistoryBucket {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.members
    }
}

pub(super) struct PropertyHistory {
    pub(super) floor: EpochId,
    pub(super) rows: FxHashMap<HashableValue, HistoryBucket>,
    ordered: OrderedDirectory<PropertyOrderKey>,
    versions: VersionedDirectory<PropertyOrderKey>,
    roots_valid: bool,
}

impl PropertyHistory {
    fn new(floor: EpochId) -> Self {
        Self {
            floor,
            rows: FxHashMap::default(),
            ordered: OrderedDirectory::new(),
            versions: VersionedDirectory::new(),
            roots_valid: true,
        }
    }

    /// Detached rebuild/recovery only. Sweep endpoints chronologically so
    /// roots describe exact value-key membership, including re-entry gaps.
    fn rebuild_value_roots(&mut self) -> Result<()> {
        let mut events = Vec::new();
        for (key, bucket) in &self.rows {
            if key::keys_for_value_fixed(key.inner())
                .iter()
                .all(Option::is_none)
            {
                continue;
            }
            for intervals in bucket.values() {
                events
                    .try_reserve(
                        intervals
                            .len()
                            .checked_mul(2)
                            .ok_or(AllocError::InsufficientSpace)?,
                    )
                    .map_err(|_| AllocError::OutOfMemory)?;
                for interval in intervals {
                    if interval.to == Some(interval.from) {
                        continue;
                    }
                    if interval.from == EpochId::PENDING
                        || interval
                            .to
                            .is_some_and(|end| end < interval.from || end == EpochId::PENDING)
                    {
                        return Err(Error::InvalidValue(
                            "property root source interval is invalid".into(),
                        ));
                    }
                    events.push((interval.from, key.clone(), true));
                    if let Some(end) = interval.to {
                        events.push((end, key.clone(), false));
                    }
                }
            }
        }
        events.sort_unstable_by_key(|(epoch, _, _)| *epoch);
        let mut active = FxHashMap::<HashableValue, usize>::default();
        active
            .try_reserve(self.rows.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        let mut rebuilt = VersionedDirectory::new();
        let mut updates = Vec::new();
        let mut current_epoch = None;
        for (epoch, key, opening) in events {
            if let Some(previous) = current_epoch
                && previous != epoch
                && !updates.is_empty()
            {
                let mut prepared = rebuilt.prepare(previous, &updates)?;
                prepared.install(&mut rebuilt);
                updates.clear();
            }
            current_epoch = Some(epoch);
            let count = active.entry(key.clone()).or_default();
            let was_live = *count != 0;
            *count = if opening {
                count.checked_add(1).ok_or(AllocError::InsufficientSpace)?
            } else {
                count.checked_sub(1).ok_or_else(|| {
                    Error::InvalidValue("property root source closes absent membership".into())
                })?
            };
            let is_live = *count != 0;
            if was_live != is_live {
                updates
                    .try_reserve(3)
                    .map_err(|_| AllocError::OutOfMemory)?;
                updates.extend(
                    key::keys_for_value_fixed(key.inner())
                        .into_iter()
                        .flatten()
                        .map(|key| (key, is_live)),
                );
            }
        }
        if let Some(epoch) = current_epoch
            && !updates.is_empty()
        {
            let mut prepared = rebuilt.prepare(epoch, &updates)?;
            prepared.install(&mut rebuilt);
        }
        rebuilt.gc(self.floor)?;
        rebuilt.qualify_replacement(&self.versions)?;
        self.versions = rebuilt;
        self.roots_valid = true;
        Ok(())
    }

    /// Legacy void mutation APIs use normal infallible allocation semantics.
    /// Invalid replay histories remain an explicit indexed-read error instead
    /// of being silently served from an obsolete root or treated as OOM.
    fn finish_direct_roots(&mut self, result: Result<()>) {
        match result {
            Ok(()) => {}
            Err(Error::Storage(grafeo_common::utils::error::StorageError::Full)) => {
                std::alloc::handle_alloc_error(std::alloc::Layout::new::<PropertyOrderKey>());
            }
            Err(_) => self.roots_valid = false,
        }
    }

    /// Detached rebuild only. Ordinary commits use sparse prepared maintenance.
    fn add_interval(&mut self, id: NodeId, value: &Value, interval: Interval) -> Result<()> {
        let key = HashableValue::new(value.clone());
        if !self.rows.contains_key(&key) {
            self.rows
                .try_reserve(1)
                .map_err(|_| AllocError::OutOfMemory)?;
            for key in keys_for_value(value) {
                // Duplicate routing nodes are retired here, outside publication.
                drop(self.ordered.insert_prepared(PreparedKey::new(key)?));
            }
        }
        let bucket = self.rows.entry(key).or_default();
        bucket.try_reserve(1).map_err(|_| AllocError::OutOfMemory)?;
        let intervals = bucket.entry(id).or_default();
        intervals
            .try_reserve(1)
            .map_err(|_| AllocError::OutOfMemory)?;
        if let Some(previous) = intervals.last_mut()
            && previous.to == Some(interval.from)
        {
            previous.to = interval.to;
            let key = (previous.from, id);
            bucket.directory.set_end(&key, interval.to);
        } else {
            let prepared = PreparedInterval::new(id, interval)?;
            intervals.push(interval);
            drop(bucket.directory.insert_prepared(prepared));
        }
        Ok(())
    }

    /// Legacy direct mutations are infallible allocation APIs. Transactional
    /// publication instead uses the separately prepared sparse workspace.
    pub(super) fn record_direct(
        &mut self,
        id: NodeId,
        before: Option<&Value>,
        after: Option<&Value>,
        epoch: EpochId,
    ) {
        if epoch == EpochId::PENDING {
            return;
        }
        let before = before.filter(|value| !value.is_null());
        let after = after.filter(|value| !value.is_null());
        if before == after {
            return;
        }
        let previous = [before, after].map(|value| {
            value.map(|value| {
                let requested = HashableValue::new(value.clone());
                self.rows
                    .get_key_value(&requested)
                    .map_or((requested.clone(), false), |(key, bucket)| {
                        (key.clone(), bucket.directory.open_count() != 0)
                    })
            })
        });
        if let Some(value) = before
            && let Some(bucket) = self.rows.get_mut(&HashableValue::new(value.clone()))
            && let Some(last) = bucket
                .get_mut(&id)
                .and_then(|intervals| intervals.last_mut())
            && last.to.is_none()
            && last.from <= epoch
        {
            last.to = Some(epoch);
            let key = (last.from, id);
            bucket.directory.set_end(&key, Some(epoch));
        }
        if let Some(value) = after {
            let key = HashableValue::new(value.clone());
            if !self.rows.contains_key(&key) {
                for route in key::keys_for_value_fixed(value).into_iter().flatten() {
                    drop(
                        self.ordered
                            .insert_prepared(PreparedKey::for_direct_mutation(route)),
                    );
                }
            }
            let bucket = self.rows.entry(key).or_default();
            let intervals = bucket.entry(id).or_default();
            if let Some(last) = intervals.last_mut()
                && (last.to == Some(epoch) || last.to.is_none())
            {
                last.to = None;
                let key = (last.from, id);
                bucket.directory.set_end(&key, None);
            } else {
                let interval = Interval {
                    from: epoch,
                    to: None,
                };
                intervals.push(interval);
                drop(
                    bucket
                        .directory
                        .insert_prepared(PreparedInterval::for_direct_mutation(id, interval)),
                );
            }
        }
        if !self.roots_valid
            || self
                .versions
                .latest_epoch()
                .is_some_and(|latest| epoch < latest)
        {
            // Explicit backdated recovery can replay whole node histories in
            // node order. Rebuild its detached timeline, not a latest-root edit.
            let result = self.rebuild_value_roots();
            self.finish_direct_roots(result);
        } else {
            let mut updates = Vec::new();
            for (key, was_live) in previous.into_iter().flatten() {
                let is_live = self
                    .rows
                    .get(&key)
                    .is_some_and(|bucket| bucket.directory.open_count() != 0);
                if was_live != is_live {
                    updates.extend(
                        key::keys_for_value_fixed(key.inner())
                            .into_iter()
                            .flatten()
                            .map(|key| (key, is_live)),
                    );
                }
            }
            if !updates.is_empty() {
                let result = self.versions.prepare(epoch, &updates).map(|mut prepared| {
                    prepared.install(&mut self.versions);
                });
                self.finish_direct_roots(result);
            }
        }
    }

    /// Returns owned candidates while the caller pins this physical generation.
    /// Residual value/label/visibility tests belong to the same read view.
    pub(super) fn candidates(
        &self,
        predicate: PropertyIndexPredicate<'_>,
        epoch: EpochId,
    ) -> Result<(Vec<NodeId>, (usize, usize, usize))> {
        if !self.roots_valid {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Unsupported,
                "property index root history requires rebuild",
            )));
        }
        if epoch == EpochId::PENDING {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Unsupported,
                "property index view requires a committed epoch",
            )));
        }
        if epoch < self.floor {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Unsupported,
                "property index view predates retained history",
            )));
        }
        let mut found = FxHashSet::default();
        let mut keys = FxHashSet::default();
        let mut add = |value: &Value| {
            keys.insert(HashableValue::new(value.clone()));
        };
        match predicate {
            PropertyIndexPredicate::Equal(value) => add(value),
            PropertyIndexPredicate::In(values) => values.iter().for_each(add),
            PropertyIndexPredicate::Range { .. } => {}
        }
        let mut visited = 0;
        for (lower, upper) in routes(predicate) {
            visited += self
                .versions
                .visit_at(epoch, lower.as_ref(), upper.as_ref(), |key| {
                    if let Some(source) = key.source_key() {
                        keys.insert(source.clone());
                    }
                })
                .1;
        }
        let mut posting_ids = 0;
        let mut posting_intervals = 0;
        for key in keys {
            if let Some(bucket) = self.rows.get(&key) {
                let (matched, inspected) = bucket.directory.visit_at(epoch, |id| {
                    found.insert(id);
                });
                posting_ids += matched;
                posting_intervals += inspected;
            }
        }
        Ok((
            found.into_iter().collect(),
            (visited, posting_ids, posting_intervals),
        ))
    }
}

impl PropertyHistory {
    /// Drops only intervals ending at or before the retained boundary; an
    /// interval covering the boundary remains queryable. This is explicit GC,
    /// outside allocation-free final commit installation.
    pub(super) fn gc(&mut self, floor: EpochId) {
        if floor == EpochId::PENDING || floor <= self.floor {
            return;
        }
        let ordered = &mut self.ordered;
        self.rows.retain(|key, bucket| {
            let HistoryBucket { members, directory } = bucket;
            members.retain(|id, intervals| {
                intervals.retain(|interval| {
                    if interval.to.is_some_and(|end| end <= floor) {
                        drop(directory.remove(&(interval.from, *id)));
                        false
                    } else {
                        true
                    }
                });
                !intervals.is_empty()
            });
            if bucket.is_empty() {
                for key in key::keys_for_value_fixed(key.inner()).into_iter().flatten() {
                    drop(ordered.remove(&key));
                }
                false
            } else {
                true
            }
        });
        self.floor = floor;
        let _ = self.versions.gc(floor);
    }

    /// An aborted fresh identity has no committed incarnation whose history
    /// may survive ID reuse. Hydration rollback must never call this helper.
    pub(super) fn purge_identity(&mut self, id: NodeId) {
        let ordered = &mut self.ordered;
        let mut changed = false;
        self.rows.retain(|key, bucket| {
            if let Some(intervals) = bucket.remove(&id) {
                changed = true;
                for interval in intervals {
                    drop(bucket.directory.remove(&(interval.from, id)));
                }
            }
            if bucket.is_empty() {
                for key in key::keys_for_value_fixed(key.inner()).into_iter().flatten() {
                    drop(ordered.remove(&key));
                }
                false
            } else {
                true
            }
        });
        if changed {
            let result = self.rebuild_value_roots();
            self.finish_direct_roots(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distinct_value_history_store() -> (LpgStore, NodeId) {
        let store = LpgStore::new().unwrap();
        store.set_epoch(EpochId::new(1));
        let mut ids = Vec::new();
        for value in 1..=128 {
            let id = store.create_node(&["Item"]);
            store.set_node_property(id, "k", Value::Int64(value));
            ids.push(id);
        }
        store.create_property_index("k");
        store.set_epoch(EpochId::new(2));
        for (offset, &id) in ids.iter().enumerate() {
            store.set_node_property(id, "k", Value::Int64(1000 + i64::try_from(offset).unwrap()));
        }
        let target = ids[66];
        store.set_epoch(EpochId::new(3));
        store.set_node_property(target, "k", Value::Int64(67));
        store.set_epoch(EpochId::new(4));
        store.set_node_property(target, "k", Value::Int64(1066));
        (store, target)
    }

    #[test]
    fn retained_distinct_value_ranges_skip_expired_keys_and_reentry_gaps() {
        let (store, target) = distinct_value_history_store();
        for (epoch, lower, upper, expected) in [
            (1, 67, 67, vec![target]),
            (2, 1, 128, vec![]),
            (3, 1, 128, vec![target]),
            (4, 1, 128, vec![]),
        ] {
            let min = Value::Int64(lower);
            let max = Value::Int64(upper);
            let before = store.work_snapshot();
            let rows = store
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "k",
                    predicate: PropertyIndexPredicate::Range {
                        min: Some(&min),
                        max: Some(&max),
                        min_inclusive: true,
                        max_inclusive: true,
                    },
                    epoch: EpochId::new(epoch),
                    transaction_id: None,
                })
                .unwrap()
                .unwrap();
            let work = store.work_snapshot().since(before);
            assert_eq!(rows, expected, "epoch {epoch}");
            assert!(!work.scanned_any());
            assert!(
                work.property_index_route_keys <= 64,
                "epoch {epoch} enumerated expired distinct keys: {work:?}"
            );
            assert!(
                work.property_index_posting_intervals <= 64,
                "epoch {epoch} enumerated nonmatching histories: {work:?}"
            );
        }
    }

    #[test]
    fn rebuilt_distinct_value_ranges_preserve_exact_epoch_key_membership() {
        let (store, target) = distinct_value_history_store();
        let rebuilt =
            PropertyIndexRows::from_image(store.property_index_image("k").unwrap()).unwrap();
        for (epoch, expected) in [(2, vec![]), (3, vec![target]), (4, vec![])] {
            let min = Value::Int64(1);
            let max = Value::Int64(128);
            let (rows, (keys, _, intervals)) = rebuilt
                .history
                .read()
                .candidates(
                    PropertyIndexPredicate::Range {
                        min: Some(&min),
                        max: Some(&max),
                        min_inclusive: true,
                        max_inclusive: true,
                    },
                    EpochId::new(epoch),
                )
                .unwrap();
            assert_eq!(rows, expected, "rebuilt epoch {epoch}");
            assert!(
                keys <= 64,
                "rebuilt epoch {epoch} walked {keys} expired value keys"
            );
            assert!(
                intervals <= 64,
                "rebuilt epoch {epoch} walked {intervals} nonmatching intervals"
            );
        }
    }

    #[test]
    fn indexed_recovery_replays_per_node_epoch_histories() {
        let store = LpgStore::new().unwrap();
        store.set_epoch(EpochId::new(1));
        let first = store.create_node(&["Item"]);
        let second = store.create_node(&["Item"]);
        store.create_property_index("k");
        for (id, old, new) in [(first, 10, 100), (second, 20, 200)] {
            store.set_node_property_at_epoch(id, "k", Value::Int64(old), EpochId::new(1));
            store.set_node_property_at_epoch(id, "k", Value::Int64(new), EpochId::new(2));
        }
        store.set_epoch(EpochId::new(2));
        for (epoch, lower, upper, mut expected) in [
            (1, 10, 20, vec![first, second]),
            (2, 10, 20, vec![]),
            (1, 100, 200, vec![]),
            (2, 100, 200, vec![first, second]),
        ] {
            let min = Value::Int64(lower);
            let max = Value::Int64(upper);
            let before = store.work_snapshot();
            let mut ids = store
                .lookup_nodes_indexed(PropertyIndexRequest {
                    property: "k",
                    predicate: PropertyIndexPredicate::Range {
                        min: Some(&min),
                        max: Some(&max),
                        min_inclusive: true,
                        max_inclusive: true,
                    },
                    epoch: EpochId::new(epoch),
                    transaction_id: None,
                })
                .unwrap()
                .unwrap();
            ids.sort_unstable();
            expected.sort_unstable();
            assert_eq!(ids, expected, "epoch {epoch}, range {lower}..={upper}");
            assert!(!store.work_snapshot().since(before).scanned_any());
        }
    }

    #[test]
    fn rebuilt_postings_preserve_remove_and_a_b_a_intervals() {
        let id = NodeId::new(3);
        let rows = PropertyIndexRows::from_image(PropertyIndexImage {
            current: vec![(id, Value::Int64(10))],
            history: vec![(
                id,
                vec![
                    (EpochId::new(1), Value::Int64(10)),
                    (EpochId::new(2), Value::Int64(20)),
                    (EpochId::new(3), Value::Null),
                    (EpochId::new(4), Value::Int64(10)),
                ],
            )],
            floor: EpochId::INITIAL,
        })
        .unwrap();
        let history = rows.history.read();
        for (epoch, expected) in [(0, false), (1, true), (2, false), (3, false), (4, true)] {
            let (ids, _) = history
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(10)),
                    EpochId::new(epoch),
                )
                .unwrap();
            assert_eq!(ids.contains(&id), expected, "epoch {epoch}");
        }
    }

    #[test]
    fn rebuilt_postings_do_not_expose_same_epoch_intermediate_values() {
        let id = NodeId::new(3);
        let rows = PropertyIndexRows::from_image(PropertyIndexImage {
            current: vec![(id, Value::Int64(20))],
            history: vec![(
                id,
                vec![
                    (EpochId::new(1), Value::Int64(10)),
                    (EpochId::new(1), Value::Int64(20)),
                ],
            )],
            floor: EpochId::new(1),
        })
        .unwrap();
        let history = rows.history.read();
        assert!(
            history
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(10)),
                    EpochId::new(1)
                )
                .unwrap()
                .0
                .is_empty()
        );
        assert!(
            history
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(20)),
                    EpochId::INITIAL
                )
                .is_err()
        );
        assert_eq!(
            history
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(20)),
                    EpochId::new(1)
                )
                .unwrap()
                .0,
            vec![id]
        );
    }
}
