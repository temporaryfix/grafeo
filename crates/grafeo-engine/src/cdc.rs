//! Change Data Capture (CDC) for tracking entity mutations.
//!
//! When the `cdc` feature is enabled, the database records every mutation
//! (create, update, delete) with before/after property snapshots. This
//! enables audit trails, temporal queries, and downstream sync.
//!
//! # Event model
//!
//! Every write emits a [`ChangeEvent`] keyed by [`EntityId`]. Three id
//! variants cover the supported entity kinds: `Node(NodeId)` and
//! `Edge(EdgeId)` for LPG, `Triple(u64)` for RDF (hashed via a
//! content-stable triple hash so the log can key events by triple
//! content without maintaining a separate id registry).
//! [`ChangeKind`] distinguishes
//! `Create` / `Update` / `Delete`; the `before` and `after` property maps
//! are populated only for the variants that can carry them (`Update` has
//! both; `Create` has only `after`; `Delete` has only `before`).
//! LPG IDs are graph-local and may collide. Entity-only history aggregates
//! those collisions; `history_in_graph` filters exact component-qualified LPG
//! paths. RDF-specific history uses `history_in_rdf_graph`. Neither coordinate
//! filters DROP/CREATE incarnations; Session events carry their native
//! `graph_incarnation` so consumers can distinguish those lifetimes.
//!
//! LPG events carry `lpg_graph` (including an empty array for root), while RDF
//! events carry `triple_graph` only for named RDF graphs.
//!
//! # Thread safety and ordering
//!
//! [`CdcLog`] owns one `RwLock<VecDeque<ChangeEvent>>` retained window. Native
//! publication appends commit-ordered batches whose vector and destination
//! capacity were reserved before the marker. Sequence equals the retained floor
//! plus window offset, so bounded readers seek directly without collecting or
//! sorting the whole feed. Retention removes an epoch-aligned prefix.
//!
//! Each event carries the MVCC epoch and an HLC timestamp minted once at
//! accumulator staging time. Within each committed epoch, timestamps preserve
//! staging order. The HLC clock remains monotonic across reopen through its
//! persisted high-water. Complete packed-clock exhaustion is fail-stop.
//! Standalone log writers sort supplied epochs before exposing their window;
//! those writers do not provide database durability or public resume authority.
//!
//! # Integration with the commit path
//!
//! Every CDC-enabled Session mutation path stages into one transaction-owned
//! `TransactionChangeAccumulator`. After durable commit acknowledgement and
//! every fallible state-publication step (with durable acknowledgement applying
//! only when WAL/durable storage is configured), the session inserts a batch
//! whose final epoch, surviving LPG lifetimes and destination capacity were
//! prepared before the durable marker. Insertion into [`CdcLog`] occurs while still
//! holding the transaction publication write barrier. GrafeoDB and Session CDC
//! readers therefore observe state/event co-visibility at this healthy
//! live-process publication cut. This is subject to runtime CDC enablement and
//! retention: disabled writes and retention-pruned history legitimately leave
//! state without an event. Standalone public [`CdcLog`] readers use only the
//! log's own lock and do not acquire the database publication barrier. Rollback
//! and savepoint rollback discard or truncate the same accumulator. One
//! per-Session operation gate prevents a concurrent mutation from crossing a
//! savepoint's state/CDC position, and database/session readers recheck sticky
//! durability poison after acquiring their publication cut. There is no
//! asynchronous flush queue.
//!
//! # CDC epoch vs. MVCC epoch
//!
//! The `epoch` field on [`ChangeEvent`] is the MVCC [`EpochId`] produced
//! by the [`crate::transaction::TransactionManager`]
//! at commit time. It is the same epoch that tags the resulting
//! [`LpgStore`] version, so
//! bounded `history_after` pages and `MATCH ... AT EPOCH` use the same
//! commit coordinates. The CDC log has no epoch of its own: retention is
//! driven by external epoch advances through
//! [`apply_retention()`](CdcLog::apply_retention), called from
//! [`GrafeoDB::gc()`](crate::GrafeoDB::gc) under the same epoch used to
//! GC MVCC version chains.
//!
//! # Persistence boundary
//!
//! With WAL enabled, Session commits encode bounded canonical model-local
//! batches before the authenticated marker. Directory-WAL recovery restores
//! their native graph lifetimes, original timestamps and before/after images;
//! it never reconstructs events from current graph state. Builds without WAL
//! and standalone log writers provide in-memory history only.
//!
//! Container checkpoints and Snapshot12's current CDC1 envelope preserve the
//! retained native images, generation, sequence floor/next and clock high-water.
//! Checkpoint plus authenticated WAL-tail recovery installs each event once.
//! A retained checkpoint is bounded to64MiB; larger windows fail before file
//! publication. Explicit snapshot copies also preserve in-memory database feeds.
//! Directory pruning syncs an epoch-aligned sequence-floor record before
//! removing events. Replay checks that transition against its native preimage;
//! graph-recovery WAL remains intact. The database's
//! bounded `changes_after` and indexed `history_after` readers expose durable
//! cursors through database, authorized Session and all seven bindings.
//! Physical directory WAL retirement and final integration remain P5 work.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_engine::GrafeoDB;
//! use grafeo_common::types::Value;
//!
//! let db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_cdc())?;
//! let id = db.create_node(&["Person"]);
//! db.set_node_property(id, "name", Value::from("Alix"))?;
//! db.set_node_property(id, "name", Value::from("Gus"))?;
//!
//! let query = grafeo_engine::cdc::EntityHistoryQuery::new(id);
//! let history = db.history_after(&query, None, 3, 4096)?;
//! assert_eq!(history.events.len(), 3); // create + 2 updates
//! # Ok(())
//! # }
//! ```

pub(crate) mod checkpoint;
mod codec;
mod cursor;
pub use cursor::{ChangePage, EntityHistoryQuery, HistoryGraph};
#[cfg(feature = "wal")]
pub(crate) mod wal;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, priorities};
use grafeo_common::types::{
    EdgeId, EpochId, GraphIncarnationId, GraphPath, HlcClock, HlcTimestamp, NodeId, Value,
};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
use hashbrown::HashMap as HbHashMap;
use parking_lot::{Mutex, RwLock};

/// The kind of mutation that occurred.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ChangeKind {
    /// A new entity was created.
    Create,
    /// An existing entity was updated (property set or removed).
    Update,
    /// An entity was deleted.
    Delete,
}

/// A unique identifier for a graph entity (node, edge, or RDF triple).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EntityId {
    /// A node identifier.
    Node(NodeId),
    /// An edge identifier.
    Edge(EdgeId),
    /// An RDF triple, identified by a content hash of its terms.
    Triple(u64),
}

impl From<NodeId> for EntityId {
    fn from(id: NodeId) -> Self {
        Self::Node(id)
    }
}

impl From<EdgeId> for EntityId {
    fn from(id: EdgeId) -> Self {
        Self::Edge(id)
    }
}

impl EntityId {
    /// Returns the raw u64 value for binding layers.
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        match self {
            Self::Node(id) => id.as_u64(),
            Self::Edge(id) => id.as_u64(),
            Self::Triple(h) => *h,
        }
    }

    /// Returns `true` if this is a node identifier.
    #[must_use]
    pub fn is_node(&self) -> bool {
        matches!(self, Self::Node(_))
    }

    /// Returns `true` if this is an RDF triple identifier.
    #[must_use]
    pub fn is_triple(&self) -> bool {
        matches!(self, Self::Triple(_))
    }
}

/// A recorded change event with before/after property snapshots, or an RDF
/// triple insert/delete.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChangeEvent {
    /// Native graph lifetime captured by a Session at mutation time.
    /// Standalone process-local builders have no native owner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_incarnation: Option<GraphIncarnationId>,
    /// The entity that was changed.
    pub entity_id: EntityId,
    /// The kind of change.
    pub kind: ChangeKind,
    /// MVCC epoch when the change occurred.
    pub epoch: EpochId,
    /// Hybrid Logical Clock timestamp for causal ordering.
    ///
    /// Encodes physical milliseconds (upper 48 bits) and a logical counter
    /// (lower 16 bits) into a `u64`. Backward-compatible: plain wall-clock
    /// values have logical counter = 0.
    pub timestamp: HlcTimestamp,
    /// Properties before the change (None for Create and for triple events).
    pub before: Option<HashMap<String, Value>>,
    /// Properties after the change (None for Delete and for triple events).
    pub after: Option<HashMap<String, Value>>,
    /// Node label image: creation and label-add post-image, or label-remove
    /// and versioned deletion pre-image. Absent on property-only updates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Edge relationship type. Present only on edge Create events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    /// Edge source node ID. Present only on edge Create events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub src_id: Option<u64>,
    /// Edge destination node ID. Present only on edge Create events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dst_id: Option<u64>,
    /// RDF triple subject (N-Triples encoded). Present only on triple events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_subject: Option<String>,
    /// RDF triple predicate (N-Triples encoded). Present only on triple events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_predicate: Option<String>,
    /// RDF triple object (N-Triples encoded). Present only on triple events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_object: Option<String>,
    /// Exact LPG graph coordinate. Root is `Some(GraphPath::root())`;
    /// RDF events have no LPG coordinate. Serialized as a component array.
    #[serde(
        default,
        with = "optional_graph_path",
        skip_serializing_if = "Option::is_none"
    )]
    pub lpg_graph: Option<GraphPath>,
    /// RDF named graph only; `None` denotes the RDF default graph for triples.
    /// LPG events never put their coordinate in this field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triple_graph: Option<String>,
}

impl ChangeEvent {
    /// Returns the exact LPG coordinate, or `None` for an RDF event.
    #[must_use]
    pub fn graph_path(&self) -> Option<&GraphPath> {
        self.lpg_graph.as_ref()
    }
}

/// Process-local JSON shape; never bypass GraphPath's bounded constructor.
mod optional_graph_path {
    use grafeo_common::types::{GraphPath, MAX_GRAPH_PATH_COMPONENTS, MAX_WORLD_GRAPH_NAME_BYTES};
    use serde::de::{DeserializeSeed, Error, SeqAccess, Visitor};
    use serde::{Deserializer, Serialize, Serializer};
    use std::fmt;

    pub(super) fn serialize<S: Serializer>(
        path: &Option<GraphPath>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        path.as_ref()
            .map(GraphPath::components)
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<GraphPath>, D::Error> {
        struct OptionalPath;
        impl<'de> Visitor<'de> for OptionalPath {
            type Value = Option<GraphPath>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("null or a bounded graph component array")
            }
            fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_some<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                deserializer.deserialize_seq(PathComponents).map(Some)
            }
        }
        deserializer.deserialize_option(OptionalPath)
    }

    struct Component;
    impl<'de> DeserializeSeed<'de> for Component {
        type Value = String;
        fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<String, D::Error> {
            struct Name;
            impl Visitor<'_> for Name {
                type Value = String;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a bounded UTF-8 graph name")
                }
                fn visit_str<E: Error>(self, value: &str) -> Result<String, E> {
                    if value.len() > MAX_WORLD_GRAPH_NAME_BYTES {
                        return Err(E::custom("graph component exceeds byte limit"));
                    }
                    let mut name = String::new();
                    name.try_reserve_exact(value.len()).map_err(E::custom)?;
                    name.push_str(value);
                    Ok(name)
                }
                fn visit_string<E: Error>(self, value: String) -> Result<String, E> {
                    if value.len() > MAX_WORLD_GRAPH_NAME_BYTES {
                        return Err(E::custom("graph component exceeds byte limit"));
                    }
                    Ok(value)
                }
            }
            deserializer.deserialize_string(Name)
        }
    }

    struct PathComponents;
    impl<'de> Visitor<'de> for PathComponents {
        type Value = GraphPath;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a bounded graph component array")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<GraphPath, A::Error> {
            if sequence
                .size_hint()
                .is_some_and(|size| size > MAX_GRAPH_PATH_COMPONENTS)
            {
                return Err(A::Error::custom("graph path exceeds depth limit"));
            }
            let mut names = Vec::new();
            while let Some(name) = sequence.next_element_seed(Component)? {
                if names.len() == MAX_GRAPH_PATH_COMPONENTS {
                    return Err(A::Error::custom("graph path exceeds depth limit"));
                }
                names.try_reserve(1).map_err(A::Error::custom)?;
                names.push(name);
            }
            let components: Vec<_> = names.iter().map(String::as_str).collect();
            GraphPath::from_components(&components).map_err(A::Error::custom)
        }
    }
}

/// Transaction-owned staging boundary for live and WAL-persisted CDC events.
///
/// Mutation paths append events with [`EpochId::PENDING`]. The session that
/// owns this accumulator either truncates/discards them during rollback or
/// prepares final events and destination capacity before the durable marker,
/// then publishes the reserved batch after native state publication.
/// Persistence uses the prepared native batch inside the shared WAL group.
pub(crate) struct TransactionChangeAccumulator {
    events: Mutex<Vec<StagedChange>>,
    cdc_log: Arc<CdcLog>,
}

struct StagedChange {
    event: ChangeEvent,
    /// Exact LPG store that accepted this event. RDF events have no LPG token.
    #[cfg(feature = "lpg")]
    lpg_incarnation: Option<Arc<LpgStore>>,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct AccumulatorEventSnapshot(Vec<ChangeEvent>);

#[cfg(test)]
impl std::ops::Deref for AccumulatorEventSnapshot {
    type Target = [ChangeEvent];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TransactionChangeAccumulator {
    pub(crate) fn new(cdc_log: &Arc<CdcLog>) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cdc_log: Arc::clone(cdc_log),
        }
    }

    /// Stages an event while preserving its already-qualified graph name.
    ///
    /// Any preliminary timestamp supplied by an event builder is overwritten:
    /// the final timestamp is minted while holding the same mutex that fixes
    /// vector order, so concurrent use of one Session remains strictly ordered.
    #[cfg(feature = "triple-store")]
    pub(crate) fn stage(&self, mut event: ChangeEvent) {
        let mut events = self.events.lock();
        event.epoch = EpochId::PENDING;
        event.timestamp = self.cdc_log.next_timestamp();
        events.push(StagedChange {
            event,
            #[cfg(feature = "lpg")]
            lpg_incarnation: None,
        });
    }

    /// Stages an LPG event with the exact store that accepted the mutation.
    #[cfg(feature = "lpg")]
    pub(crate) fn stage_lpg(
        &self,
        mut event: ChangeEvent,
        graph: &GraphPath,
        graph_incarnation: Arc<LpgStore>,
    ) {
        let mut events = self.events.lock();
        event.lpg_graph = Some(graph.clone());
        event.graph_incarnation = Some(graph_incarnation.graph_incarnation_id());
        event.triple_graph = None;
        event.epoch = EpochId::PENDING;
        event.timestamp = self.cdc_log.next_timestamp();
        events.push(StagedChange {
            event,
            lpg_incarnation: Some(graph_incarnation),
        });
    }

    /// Publishes one immediately-visible mutation to this
    /// accumulator's constructor-bound log.
    ///
    /// Keeping both the clock and publication target behind the accumulator
    /// makes it impossible for a graph-store wrapper to mint against one log
    /// and publish into another.
    #[cfg(feature = "lpg")]
    pub(crate) fn record_direct(&self, mut event: ChangeEvent, graph: &GraphPath) {
        event.lpg_graph = Some(graph.clone());
        event.triple_graph = None;
        event.timestamp = self.cdc_log.next_timestamp();
        self.cdc_log.record(event);
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    pub(crate) fn position(&self) -> usize {
        self.events.lock().len()
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    pub(crate) fn truncate(&self, position: usize) {
        self.events.lock().truncate(position);
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    pub(crate) fn clear(&self) {
        self.events.lock().clear();
    }

    /// Detaches final events while preparation can still abort the transaction.
    #[cfg(feature = "lpg")]
    pub(crate) fn prepare_committed_lpg(
        &self,
        commit_epoch: EpochId,
        mut incarnation_survives: impl FnMut(&GraphPath, &Arc<LpgStore>) -> bool,
    ) -> grafeo_common::utils::error::Result<PreparedCdcBatch<'_>> {
        self.prepare_committed_filtered(commit_epoch, |staged| {
            staged.lpg_incarnation.as_ref().is_none_or(|incarnation| {
                staged
                    .event
                    .graph_path()
                    .is_some_and(|graph| incarnation_survives(graph, incarnation))
            })
        })
    }

    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
    pub(crate) fn prepare_committed(
        &self,
        commit_epoch: EpochId,
    ) -> grafeo_common::utils::error::Result<PreparedCdcBatch<'_>> {
        self.prepare_committed_filtered(commit_epoch, |_| true)
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    fn prepare_committed_filtered(
        &self,
        commit_epoch: EpochId,
        mut survives: impl FnMut(&StagedChange) -> bool,
    ) -> grafeo_common::utils::error::Result<PreparedCdcBatch<'_>> {
        use grafeo_common::utils::error::Error;
        if commit_epoch == EpochId::PENDING {
            return Err(Error::InvalidValue("CDC commit epoch is pending".into()));
        }
        let staged = std::mem::take(&mut *self.events.lock());
        let mut groups: HbHashMap<EntityId, Vec<ChangeEvent>> = HbHashMap::new();
        for mut staged in staged {
            if !survives(&staged) {
                continue;
            }
            if staged.event.epoch != EpochId::PENDING {
                return Err(Error::InvalidValue(
                    "CDC staged epoch is already resolved".into(),
                ));
            }
            staged.event.epoch = commit_epoch;
            if !groups.contains_key(&staged.event.entity_id) {
                groups.try_reserve(1).map_err(|_| cdc_capacity_error())?;
            }
            let events = groups.entry(staged.event.entity_id).or_default();
            events.try_reserve(1).map_err(|_| cdc_capacity_error())?;
            events.push(staged.event);
        }
        Ok(PreparedCdcBatch {
            log: &self.cdc_log,
            groups,
        })
    }

    #[cfg(test)]
    pub(crate) fn lock(&self) -> AccumulatorEventSnapshot {
        AccumulatorEventSnapshot(
            self.events
                .lock()
                .iter()
                .map(|staged| staged.event.clone())
                .collect(),
        )
    }
}

/// Detached transaction events: no source-store access remains after this cut.
#[cfg(any(feature = "lpg", feature = "triple-store", feature = "wal"))]
pub(crate) struct PreparedCdcBatch<'a> {
    log: &'a CdcLog,
    groups: HbHashMap<EntityId, Vec<ChangeEvent>>,
}

fn cdc_capacity_error() -> grafeo_common::utils::error::Error {
    grafeo_common::utils::error::Error::Storage(grafeo_common::utils::error::StorageError::Full)
}

#[cfg(any(feature = "lpg", feature = "triple-store", feature = "wal"))]
impl<'a> PreparedCdcBatch<'a> {
    /// Last abortable reservation. The writer pins every reserved destination
    /// until insertion, including against standalone log writers and retention.
    pub(crate) fn prepare_publication(
        self,
    ) -> grafeo_common::utils::error::Result<PreparedCdcPublication<'a>> {
        let mut destination = self.log.events.write();
        let count = self.groups.values().map(Vec::len).sum::<usize>();
        let next_sequence = destination
            .next_sequence
            .checked_add(u64::try_from(count).map_err(|_| cdc_capacity_error())?)
            .filter(|next| *next < u64::MAX)
            .ok_or_else(cdc_capacity_error)?;
        let high_timestamp = self
            .groups
            .values()
            .flatten()
            .map(|event| event.timestamp)
            .max()
            .unwrap_or(destination.high_timestamp)
            .max(destination.high_timestamp);
        let mut indices = HbHashMap::new();
        indices
            .try_reserve(self.groups.len())
            .map_err(|_| cdc_capacity_error())?;
        let new_entities = self
            .groups
            .keys()
            .filter(|entity| !destination.by_entity.contains_key(*entity))
            .count();
        destination
            .by_entity
            .try_reserve(new_entities)
            .map_err(|_| cdc_capacity_error())?;
        for (entity, group) in &self.groups {
            let mut positions = VecDeque::new();
            positions
                .try_reserve(group.len())
                .map_err(|_| cdc_capacity_error())?;
            indices.insert(*entity, positions);
            if let Some(existing) = destination.by_entity.get_mut(entity) {
                existing
                    .try_reserve(group.len())
                    .map_err(|_| cdc_capacity_error())?;
            }
        }
        // Both the detached ordered batch and the destination are reserved
        // before the durable marker; publication only moves within capacity.
        let mut events = Vec::new();
        events
            .try_reserve(count)
            .map_err(|_| cdc_capacity_error())?;
        events.extend(self.groups.into_values().flatten());
        events.sort_unstable_by_key(|event| (event.epoch, event.timestamp));
        for (offset, event) in events.iter().enumerate() {
            indices
                .get_mut(&event.entity_id)
                .ok_or_else(cdc_capacity_error)?
                .push_back(destination.next_sequence + offset as u64);
        }
        if destination
            .back()
            .zip(events.first())
            .is_some_and(|(last, first)| {
                (last.epoch, last.timestamp) >= (first.epoch, first.timestamp)
            })
        {
            return Err(grafeo_common::utils::error::Error::Serialization(
                "CDC publication would regress retained sequence order".into(),
            ));
        }
        destination
            .try_reserve(count)
            .map_err(|_| cdc_capacity_error())?;
        #[cfg(any(test, feature = "testing-statement-injection"))]
        if self
            .log
            .fail_next_preparation
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return Err(cdc_capacity_error());
        }
        Ok(PreparedCdcPublication {
            destination,
            next_sequence,
            high_timestamp,
            events,
            indices,
            #[cfg(feature = "testing-statement-injection")]
            log: self.log,
        })
    }
}

/// Allocation-complete feed publication. Dropping it leaves events unchanged.
#[cfg(any(feature = "lpg", feature = "triple-store", feature = "wal"))]
pub(crate) struct PreparedCdcPublication<'a> {
    destination: parking_lot::RwLockWriteGuard<'a, FeedState>,
    next_sequence: u64,
    high_timestamp: HlcTimestamp,
    events: Vec<ChangeEvent>,
    indices: HbHashMap<EntityId, VecDeque<u64>>,
    #[cfg(feature = "testing-statement-injection")]
    log: &'a CdcLog,
}

#[cfg(any(feature = "lpg", feature = "triple-store", feature = "wal"))]
impl PreparedCdcPublication<'_> {
    /// Only moves prepared vectors or appends within their reserved capacities.
    pub(crate) fn publish(mut self) {
        #[cfg(feature = "testing-statement-injection")]
        self.log.pause_before_transaction_publication();
        self.destination.next_sequence = self.next_sequence;
        self.destination.high_timestamp = self.high_timestamp;
        self.destination.extend(self.events);
        for (entity, positions) in self.indices {
            match self.destination.by_entity.entry(entity) {
                hashbrown::hash_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().extend(positions);
                }
                hashbrown::hash_map::Entry::Vacant(entry) => {
                    entry.insert(positions);
                }
            }
        }
    }
}

/// Configuration for CDC event retention.
///
/// Controls how many events the CDC log keeps in memory. When limits are
/// exceeded, the oldest events (by epoch) are pruned automatically.
#[derive(Debug, Clone)]
pub struct CdcRetentionConfig {
    /// Maximum number of epochs to retain. Events older than
    /// `current_epoch - max_epochs` are pruned during GC.
    /// `None` disables epoch-based pruning.
    pub max_epochs: Option<u64>,
    /// Maximum total event count across all entities. Oldest events
    /// (by epoch) are pruned when this limit is exceeded.
    /// `None` disables count-based pruning.
    pub max_events: Option<usize>,
}

impl Default for CdcRetentionConfig {
    fn default() -> Self {
        Self {
            max_epochs: Some(1000),
            max_events: Some(100_000),
        }
    }
}

/// State for one database-scoped test pause immediately before transaction
/// CDC insertion. This exists only in statement-injection builds.
#[cfg(feature = "testing-statement-injection")]
#[derive(Debug, Default)]
struct CdcPublicationPauseState {
    reached: bool,
    released: bool,
}

#[cfg(feature = "testing-statement-injection")]
#[derive(Debug, Default)]
struct CdcPublicationPauseInner {
    state: std::sync::Mutex<CdcPublicationPauseState>,
    changed: std::sync::Condvar,
}

/// Scoped test authority that pauses one exact database immediately before a
/// transaction inserts its CDC batch.
///
/// Dropping the authority always releases a waiting writer and removes the
/// hook from that database. The hook is per-`CdcLog`, so parallel databases
/// and tests cannot intercept one another.
#[cfg(feature = "testing-statement-injection")]
#[doc(hidden)]
#[derive(Debug)]
pub struct CdcPublicationPause {
    owner: std::sync::Weak<CdcLog>,
    inner: Arc<CdcPublicationPauseInner>,
}

#[cfg(feature = "testing-statement-injection")]
impl CdcPublicationPause {
    /// Waits until the committing writer reaches the exact pre-insertion cut.
    #[must_use]
    pub fn wait_until_reached(&self, timeout: std::time::Duration) -> bool {
        let state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (state, _) = self
            .inner
            .changed
            .wait_timeout_while(state, timeout, |state| !state.reached)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.reached
    }

    /// Releases the paused writer. Dropping this authority has the same effect.
    pub fn release(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.released = true;
        self.inner.changed.notify_all();
    }
}

#[cfg(feature = "testing-statement-injection")]
impl Drop for CdcPublicationPause {
    fn drop(&mut self) {
        self.release();
        let Some(owner) = self.owner.upgrade() else {
            return;
        };
        let mut installed = owner.publication_pause.lock();
        if installed
            .as_ref()
            .is_some_and(|inner| Arc::ptr_eq(inner, &self.inner))
        {
            installed.take();
        }
    }
}

/// One retained, contiguous commit-order window. The event map is only an
/// entity lookup; sequence identity is floor plus ordinal in (epoch,HLC) order.
#[derive(Debug)]
struct FeedState {
    events: VecDeque<ChangeEvent>,
    by_entity: HbHashMap<EntityId, VecDeque<u64>>,
    generation: u64,
    floor: u64,
    next_sequence: u64,
    high_timestamp: HlcTimestamp,
}
impl Default for FeedState {
    fn default() -> Self {
        Self {
            events: VecDeque::new(),
            by_entity: HbHashMap::new(),
            generation: 1,
            floor: 1,
            next_sequence: 1,
            high_timestamp: HlcTimestamp::zero(),
        }
    }
}
impl FeedState {
    fn trim_entity_index(&mut self) {
        let floor = self.floor;
        self.by_entity.retain(|_, positions| {
            while positions.front().is_some_and(|sequence| *sequence < floor) {
                positions.pop_front();
            }
            !positions.is_empty()
        });
        if self.events.is_empty() {
            // Drop cached event slots as well as payloads after full pruning.
            self.events = VecDeque::new();
            self.by_entity = HbHashMap::new();
        }
    }
}
impl std::ops::Deref for FeedState {
    type Target = VecDeque<ChangeEvent>;
    fn deref(&self) -> &Self::Target {
        &self.events
    }
}
impl std::ops::DerefMut for FeedState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.events
    }
}

/// The CDC log that records entity mutations.
///
/// Thread-safe: uses one locked commit-ordered retained window. Timestamps
/// are assigned by the embedded [`HlcClock`] to guarantee monotonicity.
///
/// Event retention is controlled by [`CdcRetentionConfig`]. Without retention
/// limits, the log grows unbounded (see [#250]).
///
/// [#250]: https://github.com/GrafeoDB/grafeo/issues/250
#[derive(Debug)]
pub struct CdcLog {
    events: RwLock<FeedState>,
    #[cfg(any(test, feature = "testing-statement-injection"))]
    fail_next_preparation: std::sync::atomic::AtomicBool,
    clock: Arc<HlcClock>,
    retention: CdcRetentionConfig,
    #[cfg(feature = "testing-statement-injection")]
    publication_pause: Mutex<Option<Arc<CdcPublicationPauseInner>>>,
}

impl CdcLog {
    pub(crate) fn has_checkpoint_state(&self) -> bool {
        let state = self.events.read();
        state.generation != 1
            || state.floor != 1
            || state.next_sequence != 1
            || state.high_timestamp != HlcTimestamp::zero()
    }

    /// Creates a new empty CDC log with a fresh HLC clock and default retention.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: RwLock::new(FeedState::default()),
            #[cfg(any(test, feature = "testing-statement-injection"))]
            fail_next_preparation: std::sync::atomic::AtomicBool::new(false),
            clock: Arc::new(HlcClock::new()),
            retention: CdcRetentionConfig::default(),
            #[cfg(feature = "testing-statement-injection")]
            publication_pause: Mutex::new(None),
        }
    }

    /// Creates a new CDC log with the given retention config.
    #[must_use]
    pub fn with_retention(retention: CdcRetentionConfig) -> Self {
        Self {
            events: RwLock::new(FeedState::default()),
            #[cfg(any(test, feature = "testing-statement-injection"))]
            fail_next_preparation: std::sync::atomic::AtomicBool::new(false),
            clock: Arc::new(HlcClock::new()),
            retention,
            #[cfg(feature = "testing-statement-injection")]
            publication_pause: Mutex::new(None),
        }
    }

    /// Installs one database-local scoped test pause at the final state/CDC
    /// publication boundary.
    #[cfg(feature = "testing-statement-injection")]
    pub(crate) fn install_publication_pause(
        self: &Arc<Self>,
    ) -> std::result::Result<CdcPublicationPause, &'static str> {
        let mut installed = self.publication_pause.lock();
        if installed.is_some() {
            return Err("a CDC publication pause is already installed for this database");
        }
        let inner = Arc::new(CdcPublicationPauseInner::default());
        *installed = Some(Arc::clone(&inner));
        Ok(CdcPublicationPause {
            owner: Arc::downgrade(self),
            inner,
        })
    }

    /// Stops at the installed test cut, if any. The registry lock is released
    /// before waiting so the scoped authority can always release/reset it.
    #[cfg(feature = "testing-statement-injection")]
    fn pause_before_transaction_publication(&self) {
        let pause = self.publication_pause.lock().clone();
        let Some(pause) = pause else {
            return;
        };
        let mut state = pause
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.released {
            return;
        }
        state.reached = true;
        pause.changed.notify_all();
        while !state.released {
            state = pause
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    #[cfg(feature = "testing-statement-injection")]
    pub(crate) fn fail_next_preparation(&self) {
        self.fail_next_preparation
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Returns the next HLC timestamp from this log's clock.
    ///
    /// Transaction accumulators call this only while fixing final staging
    /// order; direct paths call it immediately before record.
    pub fn next_timestamp(&self) -> HlcTimestamp {
        self.clock.now()
    }

    /// Returns a reference to the HLC clock for remote timestamp merging.
    pub fn clock(&self) -> &Arc<HlcClock> {
        &self.clock
    }

    /// Records a change event.
    pub fn record(&self, event: ChangeEvent) {
        self.record_batch(std::iter::once(event));
    }

    /// Records a direct event with a newly minted timestamp.
    fn record_with_fresh_timestamp(&self, mut event: ChangeEvent) {
        event.timestamp = self.clock.now();
        self.record(event);
    }

    /// Records a batch of change events with a single write-lock acquisition.
    pub fn record_batch(&self, events: impl IntoIterator<Item = ChangeEvent>) {
        let mut guard = self.events.write();
        let mut needs_sort = false;
        for event in events {
            needs_sort |= guard
                .back()
                .is_some_and(|last| (last.epoch, last.timestamp) > (event.epoch, event.timestamp));
            let sequence = guard.next_sequence;
            guard
                .by_entity
                .entry(event.entity_id)
                .or_default()
                .push_back(sequence);
            guard.next_sequence = guard.next_sequence.saturating_add(1);
            guard.high_timestamp = guard.high_timestamp.max(event.timestamp);
            guard.push_back(event);
        }
        // Standalone compatibility writers may supply epochs out of order.
        // Native transactional publication uses the checked append-only path.
        if needs_sort {
            guard
                .make_contiguous()
                .sort_unstable_by_key(|event| (event.epoch, event.timestamp));
            let mut indices: HbHashMap<EntityId, VecDeque<u64>> = HbHashMap::new();
            for (offset, event) in guard.iter().enumerate() {
                indices
                    .entry(event.entity_id)
                    .or_default()
                    .push_back(guard.floor + offset as u64);
            }
            guard.by_entity = indices;
        }
    }

    /// Records a node creation.
    pub fn record_create_node(
        &self,
        id: NodeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
        labels: Option<Vec<String>>,
    ) {
        let event = self.node_create_event(id, epoch, props, labels, &GraphPath::root());
        self.record_with_fresh_timestamp(event);
    }

    pub(crate) fn node_create_event(
        &self,
        id: NodeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
        labels: Option<Vec<String>>,
        graph: &GraphPath,
    ) -> ChangeEvent {
        ChangeEvent {
            graph_incarnation: None,
            entity_id: EntityId::Node(id),
            kind: ChangeKind::Create,
            epoch,
            // Transactional builders use a sentinel. The owning accumulator
            // mints the one final timestamp while it fixes staging order.
            timestamp: HlcTimestamp::zero(),
            before: None,
            after: props,
            labels,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            lpg_graph: Some(graph.clone()),
            triple_graph: None,
        }
    }

    /// Records an edge creation.
    pub fn record_create_edge(
        &self,
        id: EdgeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
        src_id: u64,
        dst_id: u64,
        edge_type: String,
    ) {
        let event = self.edge_create_event(
            id,
            epoch,
            props,
            (src_id, dst_id),
            edge_type,
            &GraphPath::root(),
        );
        self.record_with_fresh_timestamp(event);
    }

    pub(crate) fn edge_create_event(
        &self,
        id: EdgeId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
        endpoints: (u64, u64),
        edge_type: String,
        graph: &GraphPath,
    ) -> ChangeEvent {
        ChangeEvent {
            graph_incarnation: None,
            entity_id: EntityId::Edge(id),
            kind: ChangeKind::Create,
            epoch,
            timestamp: HlcTimestamp::zero(),
            before: None,
            after: props,
            labels: None,
            edge_type: Some(edge_type),
            src_id: Some(endpoints.0),
            dst_id: Some(endpoints.1),
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            lpg_graph: Some(graph.clone()),
            triple_graph: None,
        }
    }

    /// Records an RDF triple insertion.
    ///
    /// The terms must be N-Triples encoded (e.g. `<http://example.org/s>`,
    /// `"hello"`, `"42"^^<http://www.w3.org/2001/XMLSchema#integer>`).
    pub fn record_triple_insert(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        epoch: EpochId,
    ) {
        let event = self.triple_event(ChangeKind::Create, subject, predicate, object, graph, epoch);
        self.record_with_fresh_timestamp(event);
    }

    /// Records an RDF triple deletion.
    ///
    /// The terms must be N-Triples encoded.
    pub fn record_triple_delete(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        epoch: EpochId,
    ) {
        let event = self.triple_event(ChangeKind::Delete, subject, predicate, object, graph, epoch);
        self.record_with_fresh_timestamp(event);
    }

    /// Builds an RDF CDC event without publishing it.
    ///
    /// The RDF planner uses this to append an event to the owning session's
    /// transaction buffer. Publication then happens atomically with the data
    /// commit, and rollback can discard the buffered event.
    pub(crate) fn triple_event(
        &self,
        kind: ChangeKind,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        epoch: EpochId,
    ) -> ChangeEvent {
        let id = triple_hash(subject, predicate, object, graph);
        ChangeEvent {
            graph_incarnation: None,
            entity_id: EntityId::Triple(id),
            kind,
            epoch,
            timestamp: HlcTimestamp::zero(),
            before: None,
            after: None,
            labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: Some(subject.to_string()),
            triple_predicate: Some(predicate.to_string()),
            triple_object: Some(object.to_string()),
            lpg_graph: None,
            triple_graph: graph.map(ToString::to_string),
        }
    }

    /// Records a property update.
    pub fn record_update(
        &self,
        entity_id: EntityId,
        epoch: EpochId,
        key: &str,
        old_value: Option<Value>,
        new_value: Value,
    ) {
        let before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after_map = HashMap::new();
        after_map.insert(key.to_string(), new_value);

        self.record(ChangeEvent {
            graph_incarnation: None,
            entity_id,
            kind: ChangeKind::Update,
            epoch,
            timestamp: self.clock.now(),
            before,
            after: Some(after_map),
            labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            lpg_graph: (!entity_id.is_triple()).then(GraphPath::root),
            triple_graph: None,
        });
    }

    /// Records an entity deletion.
    pub fn record_delete(
        &self,
        entity_id: EntityId,
        epoch: EpochId,
        props: Option<HashMap<String, Value>>,
    ) {
        self.record(ChangeEvent {
            graph_incarnation: None,
            entity_id,
            kind: ChangeKind::Delete,
            epoch,
            timestamp: self.clock.now(),
            before: props,
            after: None,
            labels: None,
            edge_type: None,
            src_id: None,
            dst_id: None,
            triple_subject: None,
            triple_predicate: None,
            triple_object: None,
            lpg_graph: (!entity_id.is_triple()).then(GraphPath::root),
            triple_graph: None,
        });
    }

    /// Returns entity-only history, aggregating graph-local ID collisions.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn history(&self, entity_id: EntityId) -> Vec<ChangeEvent> {
        self.history_since(entity_id, EpochId::INITIAL)
    }

    /// Returns the entity's retained LPG changes at an exact component path.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn history_in_graph(
        &self,
        entity_id: EntityId,
        graph: &GraphPath,
    ) -> Vec<ChangeEvent> {
        self.history_since_in_graph(entity_id, graph, EpochId::INITIAL)
    }

    /// Returns entity-only changes at or after the supplied epoch.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn history_since(
        &self,
        entity_id: EntityId,
        since_epoch: EpochId,
    ) -> Vec<ChangeEvent> {
        self.collect_entity_history(entity_id, |event| event.epoch >= since_epoch)
    }

    /// Returns exact-path LPG changes at or after the supplied epoch.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn history_since_in_graph(
        &self,
        entity_id: EntityId,
        graph: &GraphPath,
        since_epoch: EpochId,
    ) -> Vec<ChangeEvent> {
        self.collect_entity_history(entity_id, |event| {
            event.graph_path() == Some(graph) && event.epoch >= since_epoch
        })
    }

    /// Returns RDF entity changes in an exact named or default graph.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn history_in_rdf_graph(
        &self,
        entity_id: EntityId,
        graph: Option<&str>,
    ) -> Vec<ChangeEvent> {
        self.history_since_in_rdf_graph(entity_id, graph, EpochId::INITIAL)
    }

    /// Returns exact RDF graph changes at or after the supplied epoch.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn history_since_in_rdf_graph(
        &self,
        entity_id: EntityId,
        graph: Option<&str>,
        since_epoch: EpochId,
    ) -> Vec<ChangeEvent> {
        self.collect_entity_history(entity_id, |event| {
            event.entity_id.is_triple()
                && event.triple_graph.as_deref() == graph
                && event.epoch >= since_epoch
        })
    }

    #[cfg(test)]
    fn collect_entity_history(
        &self,
        entity: EntityId,
        predicate: impl Fn(&ChangeEvent) -> bool,
    ) -> Vec<ChangeEvent> {
        let state = self.events.read();
        let Some(positions) = state.by_entity.get(&entity) else {
            return Vec::new();
        };
        let mut matching = positions
            .iter()
            .filter_map(|sequence| {
                sequence
                    .checked_sub(state.floor)
                    .and_then(|offset| usize::try_from(offset).ok())
                    .and_then(|offset| state.get(offset))
            })
            .filter(|event| predicate(event));
        let Some(first) = matching.next() else {
            return Vec::new();
        };
        // One-event entity histories are common. Allocate their actual size,
        // rather than the iterator collector's minimum of four large slots.
        // Filtering still allocates nothing when no retained event matches.
        let mut events = vec![first.clone()];
        events.extend(matching.cloned());
        events
    }

    #[cfg(test)]
    fn collect_history(&self, predicate: impl Fn(&ChangeEvent) -> bool) -> Vec<ChangeEvent> {
        self.events
            .read()
            .iter()
            .filter(|event| predicate(event))
            .cloned()
            .collect()
    }

    /// Returns retained events in an inclusive epoch range.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn changes_between(
        &self,
        start_epoch: EpochId,
        end_epoch: EpochId,
    ) -> Vec<ChangeEvent> {
        self.collect_history(|event| event.epoch >= start_epoch && event.epoch <= end_epoch)
    }

    /// Returns the total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.read().len()
    }

    /// Estimates the heap footprint of the log in bytes.
    ///
    /// Counts the retained deque's allocated event slots and distinct entities.
    /// Owned property payloads can be larger than these slots; this is an
    /// approximate structural accounting, not the page's serialized-byte bound.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> (usize, usize, usize) {
        let guard = self.events.read();
        let index_bytes = guard.by_entity.capacity()
            * (std::mem::size_of::<(EntityId, VecDeque<u64>)>() + 1)
            + guard
                .by_entity
                .values()
                .map(|positions| positions.capacity() * std::mem::size_of::<u64>())
                .sum::<usize>();
        (
            guard.capacity() * std::mem::size_of::<ChangeEvent>() + index_bytes,
            guard.by_entity.len(),
            guard.len(),
        )
    }

    /// Removes the retained prefix whose epochs precede `min_epoch`.
    pub fn prune_before(&self, min_epoch: EpochId) {
        let mut guard = self.events.write();
        let count = guard
            .iter()
            .take_while(|event| event.epoch < min_epoch)
            .count();
        guard.drain(..count);
        guard.floor = guard.next_sequence - guard.len() as u64;
        guard.trim_entity_index();
    }

    /// Prunes complete oldest epochs to satisfy the configured row limit.
    pub fn prune_to_limit(&self) {
        let Some(max) = self.retention.max_events else {
            return;
        };
        let mut guard = self.events.write();
        if guard.len() <= max {
            return;
        }
        let excess = guard.len() - max;
        let cutoff = guard[excess - 1].epoch;
        let count = guard
            .iter()
            .take_while(|event| event.epoch <= cutoff)
            .count();
        guard.drain(..count);
        guard.floor = guard.next_sequence - guard.len() as u64;
        guard.trim_entity_index();
    }

    #[cfg(feature = "wal")]
    pub(crate) fn retain_durably(
        &self,
        current_epoch: EpochId,
        wal: &grafeo_storage::wal::LpgWal,
    ) -> grafeo_common::utils::error::Result<()> {
        let mut state = self.events.write();
        let epoch_count = self.retention.max_epochs.map_or(0, |max| {
            let floor = current_epoch.as_u64().saturating_sub(max);
            state
                .iter()
                .take_while(|event| event.epoch.as_u64() < floor)
                .count()
        });
        let row_count = self.retention.max_events.map_or(0, |max| {
            let excess = state.len().saturating_sub(max);
            excess.checked_sub(1).map_or(0, |index| {
                let cutoff = state[index].epoch;
                state
                    .iter()
                    .take_while(|event| event.epoch <= cutoff)
                    .count()
            })
        });
        let count = epoch_count.max(row_count);
        if count == 0 {
            return Ok(());
        }
        let floor = state.floor + count as u64;
        let record = grafeo_storage::wal::WalRecord::CdcRetention {
            epoch: current_epoch,
            generation: state.generation,
            previous_floor: state.floor,
            floor,
            next_sequence: state.next_sequence,
        };
        grafeo_common::testing::crash::maybe_crash("cdc_retention:before_record");
        wal.log(&record)?;
        // Retention acknowledgement must survive reopen even in Batch/Adaptive
        // modes. No pruning or fallible allocation precedes this durable cut.
        wal.sync()?;
        grafeo_common::testing::crash::maybe_crash("cdc_retention:after_sync_before_prune");
        state.drain(..count);
        state.floor = floor;
        state.trim_entity_index();
        Ok(())
    }

    /// Applies configured retention under the database publication authority.
    pub fn apply_retention(&self, current_epoch: EpochId) {
        if let Some(max_epochs) = self.retention.max_epochs {
            self.prune_before(EpochId::new(
                current_epoch.as_u64().saturating_sub(max_epochs),
            ));
        }
        self.prune_to_limit();
    }

    /// Approximate allocated structural bytes, including retained window and
    /// entity-index capacities. Owned property payloads are additional.
    #[must_use]
    pub fn approximate_memory_bytes(&self) -> usize {
        self.heap_memory_bytes().0
    }
}

impl Default for CdcLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory accounting for CDC; only database-owned retention may prune events.
impl MemoryConsumer for CdcLog {
    fn name(&self) -> &str {
        "cdc_log"
    }

    fn memory_usage(&self) -> usize {
        self.approximate_memory_bytes()
    }

    fn eviction_priority(&self) -> u8 {
        // Accounting remains in the existing execution-buffer region. The
        // consumer cannot independently advance durable retention authority.
        priorities::QUERY_CACHE
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::ExecutionBuffers
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        // A resumable feed floor is durable authority. Buffer pressure cannot
        // retire it outside database publication and checkpoint ownership.
        0
    }

    fn can_spill(&self) -> bool {
        // Feed persistence uses its authoritative checkpoint and WAL, not
        // temporary query spill files.
        false
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        // CDC log is never spilled to disk.
        if self.memory_usage() == 0 {
            grafeo_common::memory::StorageTier::Uninitialized
        } else {
            grafeo_common::memory::StorageTier::InMemory
        }
    }
}

/// Computes a stable-within-process content hash for an RDF triple.
///
/// Used as the raw `u64` in `EntityId::Triple` so the CDC log can key events
/// by triple content without storing a separate ID registry.
fn triple_hash(subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    subject.hash(&mut h);
    predicate.hash(&mut h);
    object.hash(&mut h);
    graph.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    #[test]
    fn prepared_cdc_reserves_existing_and_new_entities_without_publishing() {
        let log = Arc::new(CdcLog::new());
        log.record_create_node(NodeId::new(1), EpochId::new(1), None, None);
        let pending = TransactionChangeAccumulator::new(&log);
        let stage = || {
            let mut events = pending.events.lock();
            for id in [1, 2, 1, 2, 1] {
                let mut event = log.node_create_event(
                    NodeId::new(id),
                    EpochId::PENDING,
                    None,
                    None,
                    &GraphPath::root(),
                );
                event.timestamp = log.next_timestamp();
                events.push(StagedChange {
                    event,
                    #[cfg(feature = "lpg")]
                    lpg_incarnation: None,
                });
            }
        };
        stage();
        let batch = pending
            .prepare_committed_filtered(EpochId::new(2), |_| true)
            .unwrap();
        assert_eq!(log.event_count(), 1);
        let ready = batch.prepare_publication().unwrap();
        assert!(
            log.events.try_write().is_none(),
            "reservation pins the destination"
        );
        assert_eq!(ready.destination.len(), 1, "new events remain detached");
        drop(ready);
        assert_eq!(log.event_count(), 1, "discard is invisible");

        stage();
        log.fail_next_preparation
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(
            pending
                .prepare_committed_filtered(EpochId::new(3), |_| true)
                .unwrap()
                .prepare_publication()
                .is_err()
        );
        assert_eq!(
            log.event_count(),
            1,
            "late preparation failure is invisible"
        );
        assert!(pending.lock().is_empty());

        stage();
        let ready = pending
            .prepare_committed_filtered(EpochId::new(4), |_| true)
            .unwrap()
            .prepare_publication()
            .unwrap();
        let capacity = ready.destination.capacity();
        let map_capacity = ready.destination.by_entity.capacity();
        let old_index_capacity =
            ready.destination.by_entity[&EntityId::Node(NodeId::new(1))].capacity();
        let new_index_capacity = ready.indices[&EntityId::Node(NodeId::new(2))].capacity();
        ready.publish();
        let events = log.events.read();
        assert_eq!(events.capacity(), capacity, "publication must not allocate");
        assert_eq!(events.len(), 6);
        assert_eq!(events.by_entity.capacity(), map_capacity);
        assert_eq!(
            events.by_entity[&EntityId::Node(NodeId::new(1))].capacity(),
            old_index_capacity
        );
        assert_eq!(
            events.by_entity[&EntityId::Node(NodeId::new(2))].capacity(),
            new_index_capacity
        );
        for (entity, positions) in &events.by_entity {
            for sequence in positions {
                assert_eq!(
                    events[usize::try_from(sequence - events.floor).unwrap()].entity_id,
                    *entity
                );
            }
        }
        assert!(
            events
                .iter()
                .zip(events.iter().skip(1))
                .all(|(a, b)| (a.epoch, a.timestamp) < (b.epoch, b.timestamp))
        );
        assert!(
            events
                .iter()
                .skip(1)
                .all(|event| event.epoch == EpochId::new(4))
        );
    }

    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    #[test]
    fn rdf_session_cdc_preparation_failure_rolls_back_state_and_feed() {
        let db = crate::GrafeoDB::with_config(
            crate::Config::in_memory()
                .with_graph_model(crate::GraphModel::Rdf)
                .with_cdc(),
        )
        .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql("INSERT DATA { <urn:cdc:abort> <urn:cdc:p> 1 }")
            .unwrap();
        db.cdc_log
            .fail_next_preparation
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(session.commit().is_err());
        assert_eq!(db.cdc_log.event_count(), 0);
        assert_eq!(
            session
                .execute_sparql("SELECT ?s WHERE { ?s <urn:cdc:p> ?o }")
                .unwrap()
                .rows
                .len(),
            0
        );
        session
            .execute_sparql("INSERT DATA { <urn:cdc:keep> <urn:cdc:p> 2 }")
            .unwrap();
        assert_eq!(
            db.cdc_log.event_count(),
            1,
            "failure is one-shot and the database is healthy"
        );
    }

    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    #[test]
    fn rdf_only_savepoint_retains_prefix_discards_suffix_and_publishes_once()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::{Config, GrafeoDB, GraphModel, Quad, Session, Term, Triple};

        fn all_changes(db: &GrafeoDB) -> Vec<ChangeEvent> {
            db.changes_between(EpochId::INITIAL, EpochId::new(u64::MAX))
                .expect("CDC history")
        }

        fn visible_rows(session: &Session) -> Vec<Vec<Value>> {
            session
                .execute_sparql("SELECT ?s ?o WHERE { ?s ?p ?o } ORDER BY ?s")
                .expect("transaction-visible RDF rows")
                .rows()
                .to_vec()
        }

        fn quad(subject: &str, value: &str) -> Quad {
            Quad::new(Triple::new(
                Term::iri(subject),
                Term::iri("urn:rdf-cdc:p"),
                Term::literal(value),
            ))
        }

        let db = GrafeoDB::with_config(
            Config::in_memory()
                .with_graph_model(GraphModel::Rdf)
                .with_cdc(),
        )
        .expect("RDF CDC database");
        let mut session = db.session();
        let retained = quad("urn:rdf-cdc:retained", "prefix");
        let discarded = quad("urn:rdf-cdc:discarded", "suffix");
        let accepted = quad("urn:rdf-cdc:accepted", "later");
        let epoch_before = db.rdf_store_commit_epoch();
        session.begin_transaction().expect("explicit transaction");
        session
            .execute_sparql(r#"INSERT DATA { <urn:rdf-cdc:retained> <urn:rdf-cdc:p> "prefix" }"#)
            .expect("retained prefix");
        assert_eq!(
            visible_rows(&session),
            vec![vec![
                Value::from("urn:rdf-cdc:retained"),
                Value::from("prefix")
            ]]
        );
        assert!(
            session
                .try_contains_rdf_quad(&retained)
                .expect("prefix exists")
        );
        session.savepoint("retained").expect("savepoint");
        session
            .execute_sparql(r#"DELETE DATA { <urn:rdf-cdc:retained> <urn:rdf-cdc:p> "prefix" }"#)
            .expect("delete retained prefix after savepoint");
        session
            .execute_sparql(r#"INSERT DATA { <urn:rdf-cdc:discarded> <urn:rdf-cdc:p> "suffix" }"#)
            .expect("discarded suffix");
        assert_eq!(
            visible_rows(&session),
            vec![vec![
                Value::from("urn:rdf-cdc:discarded"),
                Value::from("suffix")
            ]],
            "the real prefix deletion and suffix insertion must be owner-visible"
        );
        assert!(
            !session
                .try_contains_rdf_quad(&retained)
                .expect("prefix deleted")
        );
        assert!(
            session
                .try_contains_rdf_quad(&discarded)
                .expect("suffix exists")
        );
        assert!(visible_rows(&db.session()).is_empty());
        assert!(all_changes(&db).is_empty());
        assert_eq!(db.cdc_log.event_count(), 0);

        session
            .rollback_to_savepoint("retained")
            .expect("rewind only the suffix");
        assert!(session.in_transaction());
        assert_eq!(
            visible_rows(&session),
            vec![vec![
                Value::from("urn:rdf-cdc:retained"),
                Value::from("prefix")
            ]]
        );
        assert!(
            session
                .try_contains_rdf_quad(&retained)
                .expect("prefix restored")
        );
        assert!(
            !session
                .try_contains_rdf_quad(&discarded)
                .expect("suffix removed")
        );
        assert!(all_changes(&db).is_empty());
        assert_eq!(db.cdc_log.event_count(), 0);
        session
            .execute_sparql(r#"INSERT DATA { <urn:rdf-cdc:accepted> <urn:rdf-cdc:p> "later" }"#)
            .expect("accepted continuation");
        let expected_rows = vec![
            vec![Value::from("urn:rdf-cdc:accepted"), Value::from("later")],
            vec![Value::from("urn:rdf-cdc:retained"), Value::from("prefix")],
        ];
        assert_eq!(visible_rows(&session), expected_rows);
        assert!(visible_rows(&db.session()).is_empty());
        assert!(all_changes(&db).is_empty(), "no precommit CDC publication");
        assert_eq!(db.cdc_log.event_count(), 0);

        let commit_epoch = session.commit().expect("commit accepted state");
        assert!(commit_epoch > epoch_before);
        assert_ne!(commit_epoch, EpochId::PENDING);
        assert!(!session.in_transaction());
        assert_eq!(visible_rows(&db.session()), expected_rows);
        let reader = db.session();
        assert!(
            reader
                .try_contains_rdf_quad(&retained)
                .expect("committed prefix")
        );
        assert!(
            reader
                .try_contains_rdf_quad(&accepted)
                .expect("committed continuation")
        );
        assert!(
            !reader
                .try_contains_rdf_quad(&discarded)
                .expect("discarded suffix")
        );
        let events = all_changes(&db);
        assert_eq!(events.len(), 2, "only retained and later writes publish");
        assert_eq!(db.cdc_log.event_count(), 2);
        assert_eq!(
            events
                .iter()
                .map(|event| (
                    event.triple_subject.as_deref(),
                    event.triple_object.as_deref(),
                    event.kind.clone(),
                    event.epoch,
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    Some("<urn:rdf-cdc:retained>"),
                    Some("\"prefix\""),
                    ChangeKind::Create,
                    commit_epoch,
                ),
                (
                    Some("<urn:rdf-cdc:accepted>"),
                    Some("\"later\""),
                    ChangeKind::Create,
                    commit_epoch,
                ),
            ]
        );
        for event in events {
            assert!(event.entity_id.is_triple());
            assert_eq!(event.triple_predicate.as_deref(), Some("<urn:rdf-cdc:p>"));
            assert_eq!(event.graph_path(), None);
            assert!(event.before.is_none() && event.after.is_none());
        }
        Ok(())
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn concurrent_accumulator_staging_mints_timestamps_in_final_vector_order() {
        let log = Arc::new(CdcLog::new());
        let accumulator = Arc::new(TransactionChangeAccumulator::new(&log));
        let mut workers = Vec::new();
        for worker in 0..4_u64 {
            let accumulator = Arc::clone(&accumulator);
            workers.push(std::thread::spawn(move || {
                for offset in 0..128_u64 {
                    accumulator.stage(ChangeEvent {
                        graph_incarnation: None,
                        entity_id: EntityId::Node(NodeId::new(worker * 128 + offset)),
                        kind: ChangeKind::Create,
                        epoch: EpochId::PENDING,
                        timestamp: HlcTimestamp::zero(),
                        before: None,
                        after: None,
                        labels: None,
                        edge_type: None,
                        src_id: None,
                        dst_id: None,
                        triple_subject: None,
                        triple_predicate: None,
                        triple_object: None,
                        lpg_graph: None,
                        triple_graph: None,
                    });
                }
            }));
        }
        for worker in workers {
            worker.join().expect("CDC staging worker");
        }

        let staged = accumulator.lock();
        assert_eq!(staged.len(), 512);
        assert!(
            staged
                .windows(2)
                .all(|pair| pair[0].timestamp < pair[1].timestamp),
            "final timestamps must be minted under the same lock that orders staging"
        );
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn accumulator_publishes_only_to_its_constructor_log() {
        let target = Arc::new(CdcLog::new());
        let unrelated = Arc::new(CdcLog::new());
        let accumulator = TransactionChangeAccumulator::new(&target);
        let node = NodeId::new(7);
        accumulator.stage(target.node_create_event(
            node,
            EpochId::PENDING,
            None,
            Some(vec!["TargetAffinity".to_string()]),
            &GraphPath::root(),
        ));

        #[cfg(feature = "lpg")]
        accumulator
            .prepare_committed_lpg(EpochId::new(9), |_, _| true)
            .unwrap()
            .prepare_publication()
            .unwrap()
            .publish();
        #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
        accumulator
            .prepare_committed(EpochId::new(9))
            .unwrap()
            .prepare_publication()
            .unwrap()
            .publish();

        let published = target.history(EntityId::Node(node));
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].epoch, EpochId::new(9));
        assert!(unrelated.history(EntityId::Node(node)).is_empty());
    }

    #[test]
    fn transactional_event_builders_leave_the_final_timestamp_unminted() {
        let log = CdcLog::new();
        assert_eq!(
            log.node_create_event(
                NodeId::new(1),
                EpochId::PENDING,
                None,
                None,
                &GraphPath::root()
            )
            .timestamp,
            HlcTimestamp::zero()
        );
        assert_eq!(
            log.edge_create_event(
                EdgeId::new(1),
                EpochId::PENDING,
                None,
                (1, 2),
                "REL".to_string(),
                &GraphPath::root()
            )
            .timestamp,
            HlcTimestamp::zero()
        );
        assert_eq!(
            log.triple_event(
                ChangeKind::Create,
                "<urn:s>",
                "<urn:p>",
                "<urn:o>",
                None,
                EpochId::PENDING,
            )
            .timestamp,
            HlcTimestamp::zero()
        );
    }

    #[test]
    fn test_record_and_history() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        log.record_create_node(node_id, EpochId(1), None, None);
        log.record_update(
            EntityId::Node(node_id),
            EpochId(2),
            "name",
            None,
            Value::from("Alix"),
        );
        log.record_update(
            EntityId::Node(node_id),
            EpochId(3),
            "name",
            Some(Value::from("Alix")),
            Value::from("Gus"),
        );

        let history = log.history(EntityId::Node(node_id));
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].kind, ChangeKind::Create);
        assert_eq!(history[1].kind, ChangeKind::Update);
        assert_eq!(history[2].kind, ChangeKind::Update);
    }

    #[test]
    fn test_history_since() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        log.record_create_node(node_id, EpochId(1), None, None);
        log.record_update(
            EntityId::Node(node_id),
            EpochId(5),
            "name",
            None,
            Value::from("Alix"),
        );
        log.record_update(
            EntityId::Node(node_id),
            EpochId(10),
            "name",
            Some(Value::from("Alix")),
            Value::from("Gus"),
        );

        let since_5 = log.history_since(EntityId::Node(node_id), EpochId(5));
        assert_eq!(since_5.len(), 2);
        assert_eq!(since_5[0].epoch, EpochId(5));
    }

    #[test]
    fn graph_qualified_history_distinguishes_colliding_entity_ids()
    -> Result<(), Box<dyn std::error::Error>> {
        let log = CdcLog::new();
        let node = NodeId::new(0);
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["default"])?,
            GraphPath::from_components(&["a/b"])?,
            GraphPath::from_components(&["a", "b"])?,
            GraphPath::from_components(&["λ\0graph"])?,
        ];
        for (ordinal, path) in paths.iter().enumerate() {
            let epoch = EpochId::new(u64::try_from(ordinal)? + 1);
            log.record(log.node_create_event(node, epoch, None, None, path));
        }
        assert_eq!(log.history(EntityId::Node(node)).len(), paths.len());
        for (ordinal, path) in paths.iter().enumerate() {
            let events = log.history_in_graph(EntityId::Node(node), path);
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].graph_path(), Some(path));
            assert!(events[0].triple_graph.is_none());
            assert_eq!(events[0].epoch, EpochId::new(u64::try_from(ordinal)? + 1));
            assert!(
                log.history_since_in_graph(EntityId::Node(node), path, EpochId::new(99))
                    .is_empty()
            );
        }
        let rdf = log.triple_event(
            ChangeKind::Create,
            "<s>",
            "<p>",
            "<o>",
            Some("a/b"),
            EpochId::new(7),
        );
        let rdf_id = rdf.entity_id;
        assert!(rdf.graph_path().is_none());
        log.record(rdf);
        assert_eq!(log.history_in_rdf_graph(rdf_id, Some("a/b")).len(), 1);
        assert!(log.history_in_rdf_graph(rdf_id, None).is_empty());
        assert!(log.history_in_graph(rdf_id, &paths[3]).is_empty());
        Ok(())
    }

    #[test]
    fn cdc_path_json_roundtrips_exact_components_and_rejects_unbounded_input()
    -> Result<(), Box<dyn std::error::Error>> {
        let log = CdcLog::new();
        for components in [
            &[][..],
            &[""][..],
            &["default"][..],
            &["a/b"][..],
            &["a", "b"][..],
        ] {
            let path = GraphPath::from_components(components)?;
            let event = log.node_create_event(NodeId::new(1), EpochId::INITIAL, None, None, &path);
            let json = serde_json::to_value(&event)?;
            assert_eq!(json["lpg_graph"], serde_json::json!(components));
            assert!(json.get("triple_graph").is_none());
            let restored: ChangeEvent = serde_json::from_value(json)?;
            assert_eq!(restored.graph_path(), Some(&path));
        }
        let event = log.node_create_event(
            NodeId::new(1),
            EpochId::INITIAL,
            None,
            None,
            &GraphPath::root(),
        );
        for invalid in [
            serde_json::json!("a/b"),
            serde_json::json!({"components": ["a"]}),
            serde_json::json!(vec![
                "";
                grafeo_common::types::MAX_GRAPH_PATH_COMPONENTS + 1
            ]),
            serde_json::json!(["x".repeat(grafeo_common::types::MAX_WORLD_GRAPH_NAME_BYTES + 1)]),
        ] {
            let mut json = serde_json::to_value(&event)?;
            json["lpg_graph"] = invalid;
            assert!(serde_json::from_value::<ChangeEvent>(json).is_err());
        }
        Ok(())
    }

    #[test]
    fn history_snapshots_sort_reversed_insertion_by_epoch_then_timestamp()
    -> Result<(), Box<dyn std::error::Error>> {
        let log = CdcLog::new();
        let node = NodeId::new(7);
        let event = |epoch, physical_ms, logical, graph_name: &GraphPath| {
            let mut event =
                log.node_create_event(node, EpochId::new(epoch), None, None, graph_name);
            event.timestamp = HlcTimestamp::new(physical_ms, logical);
            event
        };

        let late = event(3, 1, 0, &GraphPath::from_components(&["analytics"])?);
        let same_epoch_late = event(2, 9, 0, &GraphPath::root());
        let early = event(1, 5, 0, &GraphPath::from_components(&["analytics"])?);
        let same_epoch_early = event(2, 3, 0, &GraphPath::from_components(&["analytics"])?);
        log.record_batch([late, same_epoch_late, early, same_epoch_early]);

        let keys = |events: Vec<ChangeEvent>| {
            events
                .into_iter()
                .map(|event| (event.epoch, event.timestamp, event.lpg_graph))
                .collect::<Vec<_>>()
        };

        assert_eq!(
            keys(log.history(EntityId::Node(node))),
            vec![
                (
                    EpochId::new(1),
                    HlcTimestamp::new(5, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
                (
                    EpochId::new(2),
                    HlcTimestamp::new(3, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
                (
                    EpochId::new(2),
                    HlcTimestamp::new(9, 0),
                    Some(GraphPath::root())
                ),
                (
                    EpochId::new(3),
                    HlcTimestamp::new(1, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
            ]
        );
        assert_eq!(
            keys(log.history_in_graph(
                EntityId::Node(node),
                &GraphPath::from_components(&["analytics"])?
            )),
            vec![
                (
                    EpochId::new(1),
                    HlcTimestamp::new(5, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
                (
                    EpochId::new(2),
                    HlcTimestamp::new(3, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
                (
                    EpochId::new(3),
                    HlcTimestamp::new(1, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
            ]
        );
        assert_eq!(
            keys(log.history_since(EntityId::Node(node), EpochId::new(2))),
            vec![
                (
                    EpochId::new(2),
                    HlcTimestamp::new(3, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
                (
                    EpochId::new(2),
                    HlcTimestamp::new(9, 0),
                    Some(GraphPath::root())
                ),
                (
                    EpochId::new(3),
                    HlcTimestamp::new(1, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
            ]
        );
        assert_eq!(
            keys(log.history_since_in_graph(
                EntityId::Node(node),
                &GraphPath::from_components(&["analytics"])?,
                EpochId::new(2),
            )),
            vec![
                (
                    EpochId::new(2),
                    HlcTimestamp::new(3, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
                (
                    EpochId::new(3),
                    HlcTimestamp::new(1, 0),
                    Some(GraphPath::from_components(&["analytics"])?)
                ),
            ]
        );
        Ok(())
    }

    #[test]
    fn test_changes_between() {
        let log = CdcLog::new();

        log.record_create_node(NodeId::new(1), EpochId(1), None, None);
        log.record_create_node(NodeId::new(2), EpochId(3), None, None);
        log.record_update(
            EntityId::Node(NodeId::new(1)),
            EpochId(5),
            "x",
            None,
            Value::from(42),
        );

        let changes = log.changes_between(EpochId(2), EpochId(5));
        assert_eq!(changes.len(), 2); // epoch 3 and 5
    }

    #[test]
    fn test_delete_event() {
        let log = CdcLog::new();
        let node_id = NodeId::new(1);

        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::from("Alix"));

        log.record_create_node(node_id, EpochId(1), Some(props.clone()), None);
        log.record_delete(EntityId::Node(node_id), EpochId(2), Some(props));

        let history = log.history(EntityId::Node(node_id));
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].kind, ChangeKind::Delete);
        assert!(history[1].after.is_none());
        assert!(history[1].before.is_some());
    }

    #[test]
    fn test_empty_history() {
        let log = CdcLog::new();
        let history = log.history(EntityId::Node(NodeId::new(999)));
        assert!(history.is_empty());
    }

    #[test]
    fn test_event_count() {
        let log = CdcLog::new();
        assert_eq!(log.event_count(), 0);

        log.record_create_node(NodeId::new(1), EpochId(1), None, None);
        log.record_create_node(NodeId::new(2), EpochId(2), None, None);
        assert_eq!(log.event_count(), 2);
    }

    #[test]
    fn test_entity_id_conversions() {
        let node_id = NodeId::new(42);
        let entity: EntityId = node_id.into();
        assert!(entity.is_node());
        assert_eq!(entity.as_u64(), 42);

        let edge_id = EdgeId::new(7);
        let entity: EntityId = edge_id.into();
        assert!(!entity.is_node());
        assert_eq!(entity.as_u64(), 7);
    }

    #[test]
    fn test_prune_before() {
        let log = CdcLog::new();

        // Record events across 10 epochs
        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // Prune everything before epoch 6
        log.prune_before(EpochId(6));
        assert_eq!(log.event_count(), 5);

        // Verify only epochs 6-10 remain
        let remaining = log.changes_between(EpochId(0), EpochId(100));
        assert!(remaining.iter().all(|e| e.epoch >= EpochId(6)));
    }

    #[test]
    fn test_prune_to_limit() {
        let retention = CdcRetentionConfig {
            max_epochs: None,
            max_events: Some(5),
        };
        let log = CdcLog::with_retention(retention);

        // Record 10 events
        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // Prune to limit of 5
        log.prune_to_limit();
        assert!(log.event_count() <= 5);
    }

    #[test]
    fn test_apply_retention_epoch_based() {
        let retention = CdcRetentionConfig {
            max_epochs: Some(3),
            max_events: None,
        };
        let log = CdcLog::with_retention(retention);

        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // Apply retention at current epoch 10 with max_epochs=3
        // Should prune events before epoch 7
        log.apply_retention(EpochId(10));

        let remaining = log.changes_between(EpochId(0), EpochId(100));
        assert!(remaining.iter().all(|e| e.epoch >= EpochId(7)));
        assert_eq!(remaining.len(), 4); // epochs 7, 8, 9, 10
    }

    #[test]
    fn test_memory_consumer_evict() {
        let log = CdcLog::new();

        // Record 100 events
        for epoch in 1..=100 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }

        let before = log.approximate_memory_bytes();
        assert!(before > 0);

        // Buffer pressure cannot retire a resumable position independently.
        let freed = log.evict(before / 2);
        assert_eq!(freed, 0);
        assert_eq!(log.event_count(), 100);
    }

    #[test]
    fn test_retention_config_default() {
        let config = CdcRetentionConfig::default();
        assert_eq!(config.max_epochs, Some(1000));
        assert_eq!(config.max_events, Some(100_000));
    }

    #[test]
    fn test_apply_retention_count_only() {
        let retention = CdcRetentionConfig {
            max_epochs: None,
            max_events: Some(4),
        };
        let log = CdcLog::with_retention(retention);

        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // apply_retention with no epoch limit should still prune by count
        log.apply_retention(EpochId(10));
        assert!(
            log.event_count() <= 4,
            "count-based retention should prune to at most 4 events, got {}",
            log.event_count()
        );
    }

    #[test]
    fn test_apply_retention_combined_epoch_and_count() {
        // epoch limit keeps last 5 (epochs 6..=10), count limit keeps 3.
        // The stricter (count) should win after both passes.
        let retention = CdcRetentionConfig {
            max_epochs: Some(5),
            max_events: Some(3),
        };
        let log = CdcLog::with_retention(retention);

        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        log.apply_retention(EpochId(10));

        // Epoch pass prunes epochs < 5 (keeps 6..=10 = 5 events).
        // Count pass then prunes to at most 3.
        assert!(
            log.event_count() <= 3,
            "combined retention should honour the stricter limit, got {}",
            log.event_count()
        );
        // All remaining events should be recent
        let remaining = log.changes_between(EpochId(0), EpochId(100));
        assert!(remaining.iter().all(|e| e.epoch >= EpochId(6)));
    }

    #[test]
    fn test_prune_before_epoch_zero() {
        let log = CdcLog::new();

        for epoch in 1..=5 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 5);

        // Pruning before epoch 0 should be a no-op: all events have epoch >= 1
        log.prune_before(EpochId(0));
        assert_eq!(
            log.event_count(),
            5,
            "prune_before(0) should not remove anything"
        );
    }

    #[test]
    fn test_prune_to_limit_same_epoch() {
        let retention = CdcRetentionConfig {
            max_epochs: None,
            max_events: Some(3),
        };
        let log = CdcLog::with_retention(retention);

        // All 10 events share the same epoch: the cutoff epoch equals the
        // only epoch present, so prune_before removes everything at or below
        // that epoch. This is by design: epoch-granularity pruning cannot
        // split events within the same epoch.
        for i in 1..=10 {
            log.record_create_node(NodeId::new(i), EpochId(5), None, None);
        }
        assert_eq!(log.event_count(), 10);

        log.prune_to_limit();

        // After pruning, the log should have fewer events than before.
        // With all events at the same epoch, the cutoff removes them all
        // because prune_before uses a strict < comparison on epoch+1.
        assert!(
            log.event_count() < 10,
            "prune_to_limit should have removed events, got {}",
            log.event_count()
        );
    }

    #[test]
    fn test_evict_tiny_target_is_noop() {
        let log = CdcLog::new();
        for epoch in 1..=10 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        assert_eq!(log.event_count(), 10);

        // target_bytes < 256 means events_to_remove rounds to 0, so nothing freed
        let freed = log.evict(100);
        assert_eq!(freed, 0, "evict with target < 256 bytes should be a no-op");
        assert_eq!(log.event_count(), 10);
    }

    #[test]
    fn test_heap_memory_bytes_scales_with_events() {
        let log = CdcLog::new();
        let (empty_bytes, empty_entities, empty_events) = log.heap_memory_bytes();
        assert_eq!(empty_entities, 0);
        assert_eq!(empty_events, 0);

        for epoch in 1..=50 {
            log.record_create_node(NodeId::new(epoch), EpochId(epoch), None, None);
        }
        let (populated_bytes, populated_entities, populated_events) = log.heap_memory_bytes();
        assert_eq!(populated_entities, 50);
        assert_eq!(populated_events, 50);
        assert!(
            populated_bytes > empty_bytes,
            "heap estimate must grow when events are recorded"
        );
    }
}

#[cfg(all(test, feature = "lpg", feature = "gql"))]
#[test]
fn cursor_sequence_exhaustion_aborts_the_real_commit_before_publication() {
    use crate::{Config, GrafeoDB};
    use grafeo_common::utils::error::ErrorCode;
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.session().execute("INSERT (:Baseline)").unwrap();
    {
        let mut state = db.cdc_log.events.write();
        assert_eq!(state.len(), 1);
        state.floor = u64::MAX - 2;
        state.next_sequence = u64::MAX - 1;
        for positions in state.by_entity.values_mut() {
            positions[0] = u64::MAX - 2;
        }
    }
    let before = db.changes_after(None, 1, 4096).unwrap();
    assert_eq!(before.next.sequence, u64::MAX - 2);
    assert_eq!(
        db.session()
            .execute("INSERT (:Baseline)")
            .unwrap_err()
            .error_code(),
        ErrorCode::StorageFull
    );
    assert_eq!(db.node_count(), 1);
    let after = db.changes_after(None, 1, 4096).unwrap();
    assert_eq!(after.next, before.next);
    assert_eq!(
        serde_json::to_value(after.events).unwrap(),
        serde_json::to_value(before.events).unwrap()
    );
}
