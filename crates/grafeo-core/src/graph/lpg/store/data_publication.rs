//! Prepared data-only publication for one buffered LPG transaction.
//!
//! This is an inner component of aggregate data/index publication, not a public
//! index-maintenance bypass. Its caller must retain every companion index and
//! catalog fence. In particular, the store transition alone is NOT a raw-reader
//! fence: final entity, label, property and adjacency writers are retained here.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use super::data_labels::{LabelCommitWorkspace, LabelDataGuards, PreparedNodeLabelImages};
use super::{DataRebindError, LpgStore, PinnedLpgTransition, PropOp, PropertyUndoEntry, TxDelta};
use crate::graph::lpg::EdgeRecord;
#[cfg(not(feature = "tiered-storage"))]
use crate::graph::lpg::NodeRecord;
use crate::graph::lpg::property::commit::{PropertyCommitWorkspace, PropertyDataGuards};
use crate::index::adjacency::{AdjacencyCommitWorkspace, AdjacencyDataGuards};
use grafeo_common::memory::AllocError;
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
use grafeo_common::mvcc::VersionInfo;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{VersionIndex, VersionRef};
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLockWriteGuard;
use std::sync::atomic::Ordering;

pub(super) mod slots;

type Creates = (Vec<NodeId>, Vec<EdgeId>);
type EdgeDeletes = Vec<(NodeId, EdgeId, NodeId)>;
#[cfg(not(feature = "tiered-storage"))]
type NodeVersions = FxHashMap<NodeId, VersionChain<NodeRecord>>;
#[cfg(not(feature = "tiered-storage"))]
type EdgeVersions = FxHashMap<EdgeId, VersionChain<EdgeRecord>>;
#[cfg(feature = "tiered-storage")]
type NodeVersions = FxHashMap<NodeId, VersionIndex>;
#[cfg(feature = "tiered-storage")]
type EdgeVersions = FxHashMap<EdgeId, VersionIndex>;

/// Sealed access to raw child helpers. Only this coordinator constructs it,
/// after pairing one exact target, workspace and continuously retained loan.
/// This capability never escapes as an independently usable ready proof.
pub(crate) struct DataCommitScope<'store, 'transition> {
    transition: &'transition PinnedLpgTransition<'store>,
}

impl<'store, 'transition> DataCommitScope<'store, 'transition> {
    pub(crate) fn transition(&self) -> &'transition PinnedLpgTransition<'store> {
        self.transition
    }
}

/// Only this coordinator constructs creation provenance, after structural
/// qualification under the exact continuously retained store transition.
pub(super) struct PendingNodeCreationProof<'store, 'transition, 'ids> {
    store: &'store LpgStore,
    transition: &'transition PinnedLpgTransition<'store>,
    transaction_id: TransactionId,
    ids: &'ids [NodeId],
}

impl PendingNodeCreationProof<'_, '_, '_> {
    pub(super) fn qualifies(
        &self,
        store: &LpgStore,
        transition: &PinnedLpgTransition<'_>,
        transaction_id: TransactionId,
        id: NodeId,
    ) -> bool {
        std::ptr::eq(self.store, store)
            && std::ptr::eq(self.transition, transition)
            && self.transaction_id == transaction_id
            && self.ids.binary_search(&id).is_ok()
    }
}

/// Raw inputs, sparse completed histories and retired transaction payloads.
/// Construct outside all enclosing publication/transition guards.
pub(super) struct StoreDataWorkspace {
    captured: CapturedData,
    node_properties: PropertyCommitWorkspace<NodeId>,
    edge_properties: PropertyCommitWorkspace<EdgeId>,
    labels: LabelCommitWorkspace,
    forward: AdjacencyCommitWorkspace,
    backward: AdjacencyCommitWorkspace,
    retired: RetiredData,
    capture_label_images: bool,
    attempted: bool,
}

impl StoreDataWorkspace {
    pub(super) fn new(
        transaction_id: TransactionId,
        publication: EpochId,
        commit: EpochId,
    ) -> Self {
        Self {
            captured: CapturedData {
                transaction_id,
                publication,
                commit,
                creates: (Vec::new(), Vec::new()),
                deleted_nodes: Vec::new(),
                deleted_edges: Vec::new(),
                delta: TxDelta::default(),
                nodes: Vec::new(),
                edges: Vec::new(),
                touched_nodes: Vec::new(),
                touched_edges: Vec::new(),
                property_nodes: Vec::new(),
                type_counts: Vec::new(),
                deleted_types: Vec::new(),
                node_count: 0,
                edge_count: 0,
            },
            node_properties: PropertyCommitWorkspace::new(Vec::new(), Vec::new(), Vec::new()),
            edge_properties: PropertyCommitWorkspace::new(Vec::new(), Vec::new(), Vec::new()),
            labels: LabelCommitWorkspace::new(Vec::new(), Vec::new(), Vec::new()),
            forward: AdjacencyCommitWorkspace::new(Vec::new()),
            backward: AdjacencyCommitWorkspace::new(Vec::new()),
            retired: RetiredData::default(),
            capture_label_images: false,
            attempted: false,
        }
    }

    fn label_images(&self) -> &[PreparedNodeLabelImages] {
        self.labels.committed_images()
    }
}

struct CapturedData {
    transaction_id: TransactionId,
    publication: EpochId,
    commit: EpochId,
    creates: Creates,
    deleted_nodes: Vec<NodeId>,
    deleted_edges: EdgeDeletes,
    delta: TxDelta,
    nodes: Vec<QualifiedNode>,
    edges: Vec<QualifiedEdge>,
    touched_nodes: Vec<NodeId>,
    touched_edges: Vec<EdgeId>,
    property_nodes: Vec<NodeId>,
    type_counts: Vec<(usize, i64)>,
    deleted_types: Vec<usize>,
    node_count: i64,
    edge_count: i64,
}

struct QualifiedNode {
    id: NodeId,
    stamp: StructuralStamp,
    properties: Option<usize>,
    #[cfg(not(feature = "tiered-storage"))]
    record_properties: Option<u16>,
    labels: Option<u16>,
}

struct QualifiedEdge {
    id: EdgeId,
    stamp: StructuralStamp,
    record: EdgeRecord,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StructuralStamp {
    created: EpochId,
    creator: TransactionId,
    deleted: Option<EpochId>,
    deleter: Option<TransactionId>,
    versions: usize,
    creation_is_hot: bool,
}

impl StructuralStamp {
    fn owns_creation(self, tx: TransactionId) -> bool {
        self.created == EpochId::PENDING
            && self.creator == tx
            && self.versions == 1
            && self.creation_is_hot
    }

    fn from_info(info: VersionInfo, versions: usize) -> Self {
        Self {
            created: info.created_epoch,
            creator: info.created_by,
            deleted: info.deleted_epoch,
            deleter: info.deleted_by,
            versions,
            creation_is_hot: true,
        }
    }

    fn qualify(self, captured: &CapturedData, created: bool, deleted: bool) -> Result<()> {
        if created {
            if !self.owns_creation(captured.transaction_id) {
                return Err(invalid(
                    "pending creation does not own one fresh structural lifetime",
                ));
            }
        } else if self.created == EpochId::PENDING || self.created > captured.publication {
            return Err(invalid(
                "touched entity is not committed at the publication frontier",
            ));
        }
        if deleted {
            if self.deleted != Some(EpochId::PENDING)
                || self.deleter != Some(captured.transaction_id)
            {
                return Err(invalid(
                    "pending deletion does not own the structural delete marker",
                ));
            }
        } else if self.deleted.is_some() {
            // Foreign pending deletes conflict with a write to this entity;
            // finalization cannot decide that conflict by overwriting metadata.
            return Err(invalid("touched entity has an unqualified deletion"));
        }
        Ok(())
    }
}

#[derive(Default)]
struct RetiredData {
    creates: Option<Creates>,
    node_deletes: Option<Vec<NodeId>>,
    edge_deletes: Option<EdgeDeletes>,
    delta: Option<TxDelta>,
    undo: Option<Vec<PropertyUndoEntry>>,
    #[cfg(feature = "text-index")]
    text: Option<crate::index::text::TextIndexDelta>,
}

/// All inner preparation writers have been released, but exact transition
/// authority remains borrowed. No other transaction's bookkeeping was drained.
#[cfg(test)]
pub(super) struct ReleasedStoreData<'store, 'workspace, 'transition> {
    store: &'store LpgStore,
    transition: &'transition PinnedLpgTransition<'store>,
    workspace: &'workspace mut StoreDataWorkspace,
}

struct StructuralWriters<'store> {
    edges: RwLockWriteGuard<'store, EdgeVersions>,
    nodes: RwLockWriteGuard<'store, NodeVersions>,
}

struct BookkeepingWriters<'store> {
    edge_deletes: RwLockWriteGuard<'store, FxHashMap<TransactionId, EdgeDeletes>>,
    node_deletes: RwLockWriteGuard<'store, FxHashMap<TransactionId, Vec<NodeId>>>,
    #[cfg(feature = "text-index")]
    text: RwLockWriteGuard<'store, FxHashMap<TransactionId, crate::index::text::TextIndexDelta>>,
    delta: RwLockWriteGuard<'store, FxHashMap<TransactionId, TxDelta>>,
    creates: RwLockWriteGuard<'store, FxHashMap<TransactionId, Creates>>,
    undo: RwLockWriteGuard<'store, FxHashMap<TransactionId, Vec<PropertyUndoEntry>>>,
}

#[cfg(test)]
pub(super) struct PreparedStoreData<'store, 'workspace, 'transition> {
    transition: &'transition PinnedLpgTransition<'store>,
    workspace: &'workspace mut StoreDataWorkspace,
    guards: StoreDataGuards<'store>,
}

/// Retains all data readers' immediate writers through companion installation.
/// It owns no candidate or retired payload. Drop before the outer workspace.
#[cfg(test)]
pub(super) struct InstalledStoreDataFence<'store, 'workspace, 'transition> {
    _guards: StoreDataGuards<'store>,
    _workspace: &'workspace mut StoreDataWorkspace,
    _transition: &'transition PinnedLpgTransition<'store>,
}

/// Only store loans. Reverse declaration order mirrors final acquisition;
/// dropping this bundle cannot retire any workspace-owned payload.
struct StoreDataGuards<'store> {
    bookkeeping: BookkeepingWriters<'store>,
    backward: Option<AdjacencyDataGuards<'store>>,
    forward: AdjacencyDataGuards<'store>,
    edge_properties: PropertyDataGuards<'store, EdgeId>,
    node_properties: PropertyDataGuards<'store, NodeId>,
    type_counts: RwLockWriteGuard<'store, Vec<i64>>,
    labels: LabelDataGuards<'store>,
    structural: StructuralWriters<'store>,
    store: &'store LpgStore,
}

impl LpgStore {
    /// Every store's fallible preparation must finish before the aggregate takes
    /// any final entity writers. Only this phase may materialize diagnostics.
    #[cfg(test)]
    pub(super) fn prepare_buffered_commit_data<'store, 'workspace, 'transition>(
        &'store self,
        transition: &'transition PinnedLpgTransition<'store>,
        workspace: &'workspace mut StoreDataWorkspace,
    ) -> Result<ReleasedStoreData<'store, 'workspace, 'transition>> {
        self.prepare_buffered_data(transition, workspace)?;
        Ok(ReleasedStoreData {
            store: self,
            transition,
            workspace,
        })
    }

    fn prepare_buffered_data<'store>(
        &'store self,
        transition: &PinnedLpgTransition<'store>,
        workspace: &mut StoreDataWorkspace,
    ) -> Result<()> {
        if workspace.attempted {
            return Err(invalid("workspace preparation was already attempted"));
        }
        workspace.attempted = true;
        if !std::ptr::eq(self, transition.store) {
            return Err(invalid("transition belongs to a different store"));
        }
        let captured = &mut workspace.captured;
        if captured.transaction_id == TransactionId::SYSTEM
            || captured.transaction_id == TransactionId::INVALID
            || captured.publication == EpochId::PENDING
            || captured.commit == EpochId::PENDING
            || captured.commit <= captured.publication
            || self.current_epoch() > captured.publication
        {
            return Err(invalid("invalid transaction or publication/commit epoch"));
        }
        if self
            .property_undo_log
            .read()
            .get(&captured.transaction_id)
            .is_some_and(|undo| !undo.is_empty())
        {
            return Err(invalid(
                "write-through undo has no buffered pending-cell ownership proof",
            ));
        }
        captured.creates = self
            .pending_tx_creates
            .read()
            .get(&captured.transaction_id)
            .cloned()
            .unwrap_or_default();
        captured.deleted_nodes = self
            .pending_tx_deletes
            .read()
            .get(&captured.transaction_id)
            .cloned()
            .unwrap_or_default();
        captured.deleted_edges = self
            .pending_tx_edge_deletes
            .read()
            .get(&captured.transaction_id)
            .cloned()
            .unwrap_or_default();
        captured.delta = self
            .tx_property_overlay
            .read()
            .get(&captured.transaction_id)
            .cloned()
            .unwrap_or_default();
        captured.creates.0.sort_unstable();
        captured.creates.0.dedup();
        captured.creates.1.sort_unstable();
        captured.creates.1.dedup();
        captured.deleted_nodes.sort_unstable();
        captured.deleted_nodes.dedup();
        captured
            .deleted_edges
            .sort_unstable_by_key(|(_, id, _)| *id);
        for pair in captured.deleted_edges.windows(2) {
            if pair[0].1 == pair[1].1 && pair[0] != pair[1] {
                return Err(invalid("one pending edge delete has different endpoints"));
            }
        }
        captured.deleted_edges.dedup();
        self.capture_data_structure(captured)?;
        for &id in &captured.creates.0 {
            self.node_properties.validate_buffered_creation(id)?;
        }
        for &id in &captured.creates.1 {
            self.edge_properties.validate_buffered_creation(id)?;
        }

        workspace.node_properties = PropertyCommitWorkspace::new(
            if captured.creates.0.is_empty() || captured.deleted_nodes.is_empty() {
                Vec::new()
            } else {
                captured
                    .deleted_nodes
                    .iter()
                    .copied()
                    .filter(|id| captured.creates.0.binary_search(id).is_ok())
                    .collect()
            },
            captured
                .delta
                .node_props
                .iter()
                .map(|((id, key), op)| (*id, key.clone(), property_value(op)))
                .collect(),
            captured.deleted_nodes.clone(),
        );
        workspace.edge_properties = PropertyCommitWorkspace::new(
            if captured.creates.1.is_empty() || captured.deleted_edges.is_empty() {
                Vec::new()
            } else {
                captured
                    .deleted_edges
                    .iter()
                    .map(|(_, id, _)| *id)
                    .filter(|id| captured.creates.1.binary_search(id).is_ok())
                    .collect()
            },
            captured
                .delta
                .edge_props
                .iter()
                .map(|((id, key), op)| (*id, key.clone(), property_value(op)))
                .collect(),
            captured
                .deleted_edges
                .iter()
                .map(|(_, id, _)| *id)
                .collect(),
        );
        workspace.labels = LabelCommitWorkspace::new(
            captured
                .delta
                .node_labels
                .iter()
                .map(|((id, label), op)| (*id, *label, *op))
                .collect(),
            captured.deleted_nodes.clone(),
            captured.creates.0.clone(),
        );
        if workspace.capture_label_images {
            workspace.labels.capture_images();
        }
        workspace.forward = AdjacencyCommitWorkspace::new(
            captured
                .deleted_edges
                .iter()
                .map(|(src, id, _)| (*src, *id))
                .collect(),
        );
        workspace.backward = AdjacencyCommitWorkspace::new(
            captured
                .deleted_edges
                .iter()
                .map(|(_, id, dst)| (*dst, *id))
                .collect(),
        );
        let creation_proof = PendingNodeCreationProof {
            store: self,
            transition,
            transaction_id: captured.transaction_id,
            ids: &captured.creates.0,
        };
        let labels = self.prepare_commit_label_fragments(
            captured.publication,
            captured.commit,
            captured.transaction_id,
            &mut workspace.labels,
            transition,
            &creation_proof,
        )?;
        for &(id, count) in workspace.labels.label_counts() {
            let index = captured
                .nodes
                .binary_search_by_key(&id, |node| node.id)
                .map_err(|_| invalid("prepared labels lack a qualified node"))?;
            captured.nodes[index].labels = Some(count);
        }
        drop(labels);
        drop(self.node_properties.prepare_commit_fragments(
            captured.publication,
            captured.commit,
            &mut workspace.node_properties,
        )?);
        drop(self.edge_properties.prepare_commit_fragments(
            captured.publication,
            captured.commit,
            &mut workspace.edge_properties,
        )?);
        drop(
            self.forward_adj
                .prepare_commit_fragments(&mut workspace.forward)?,
        );
        if let Some(adjacency) = &self.backward_adj {
            drop(adjacency.prepare_commit_fragments(&mut workspace.backward)?);
        }
        Ok(())
    }

    fn capture_data_structure(&self, captured: &mut CapturedData) -> Result<()> {
        captured
            .property_nodes
            .extend(captured.delta.node_props.keys().map(|(id, _)| *id));
        captured.property_nodes.sort_unstable();
        captured.property_nodes.dedup();
        captured
            .touched_nodes
            .extend(captured.creates.0.iter().copied());
        captured
            .touched_nodes
            .extend(captured.deleted_nodes.iter().copied());
        captured
            .touched_nodes
            .extend(captured.delta.node_props.keys().map(|(id, _)| *id));
        captured
            .touched_nodes
            .extend(captured.delta.node_labels.keys().map(|(id, _)| *id));
        captured.touched_nodes.sort_unstable();
        captured.touched_nodes.dedup();
        captured
            .nodes
            .try_reserve(captured.touched_nodes.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for &id in &captured.touched_nodes {
            let stamp = self
                .node_data_stamp(id)
                .ok_or_else(|| invalid("touched node has no structural lifetime"))?;
            let created = captured.creates.0.binary_search(&id).is_ok();
            let deleted = captured.deleted_nodes.binary_search(&id).is_ok();
            stamp.qualify(captured, created, deleted)?;
            let count = if deleted || created {
                Some(0)
            } else if captured.property_nodes.binary_search(&id).is_ok() {
                Some(
                    self.node_properties
                        .commit_property_count(id, captured.publication)?,
                )
            } else {
                // Label-only writes do not inspect or clone property columns.
                None
            };
            captured.nodes.push(QualifiedNode {
                id,
                stamp,
                properties: count,
                #[cfg(not(feature = "tiered-storage"))]
                record_properties: None,
                labels: None,
            });
        }
        // Each normalized cell is considered once, not once per affected node.
        for ((id, key), op) in &captured.delta.node_props {
            if captured.deleted_nodes.binary_search(id).is_ok() {
                continue;
            }
            let index = captured
                .nodes
                .binary_search_by_key(id, |node| node.id)
                .map_err(|_| invalid("property operation lacks a qualified node"))?;
            let count = captured.nodes[index]
                .properties
                .ok_or_else(|| invalid("property operation lacks a prepared count"))?;
            let before = self
                .node_properties
                .get_at(*id, key, captured.publication)
                .is_some();
            let after = matches!(op, PropOp::Set(value) if !value.is_null());
            captured.nodes[index].properties = Some(
                match (before, after) {
                    (false, true) => count.checked_add(1),
                    (true, false) => count.checked_sub(1),
                    _ => Some(count),
                }
                .ok_or_else(|| invalid("property count arithmetic is invalid"))?,
            );
        }
        for node in &mut captured.nodes {
            let Some(properties) = node.properties else {
                continue;
            };
            if properties > usize::from(u16::MAX) {
                return Err(invalid("final property count exceeds record capacity"));
            }
            #[cfg(not(feature = "tiered-storage"))]
            {
                node.record_properties = Some(
                    u16::try_from(properties)
                        .map_err(|_| invalid("final property count cannot be represented"))?,
                );
            }
        }
        captured
            .touched_edges
            .extend(captured.creates.1.iter().copied());
        captured
            .touched_edges
            .extend(captured.deleted_edges.iter().map(|(_, id, _)| *id));
        captured
            .touched_edges
            .extend(captured.delta.edge_props.keys().map(|(id, _)| *id));
        captured.touched_edges.sort_unstable();
        captured.touched_edges.dedup();
        captured
            .edges
            .try_reserve(captured.touched_edges.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for &id in &captured.touched_edges {
            let (stamp, record) = self
                .edge_data_stamp(id)
                .ok_or_else(|| invalid("touched edge has no structural lifetime"))?;
            let deleted = captured
                .deleted_edges
                .binary_search_by_key(&id, |(_, edge, _)| *edge)
                .ok();
            stamp.qualify(
                captured,
                captured.creates.1.binary_search(&id).is_ok(),
                deleted.is_some(),
            )?;
            if let Some(index) = deleted {
                let (src, _, dst) = captured.deleted_edges[index];
                if record.src != src || record.dst != dst {
                    return Err(invalid(
                        "pending edge endpoints differ from its structural lifetime",
                    ));
                }
            }
            captured.edges.push(QualifiedEdge { id, stamp, record });
        }
        captured.node_count = subtract_count(
            self.live_node_count.load(Ordering::Relaxed),
            captured.deleted_nodes.len(),
        )?;
        captured.edge_count = subtract_count(
            self.live_edge_count.load(Ordering::Relaxed),
            captured.deleted_edges.len(),
        )?;
        captured
            .type_counts
            .try_reserve(captured.deleted_edges.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        captured
            .deleted_types
            .try_reserve(captured.deleted_edges.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for (_, id, _) in &captured.deleted_edges {
            let index = captured
                .edges
                .binary_search_by_key(id, |edge| edge.id)
                .map_err(|_| invalid("edge deletion lacks structural qualification"))?;
            let type_id = usize::try_from(captured.edges[index].record.type_id)
                .map_err(|_| invalid("edge type cannot be represented"))?;
            captured.deleted_types.push(type_id);
        }
        captured.deleted_types.sort_unstable();
        for group in captured.deleted_types.chunk_by(|left, right| left == right) {
            let Some(&type_id) = group.first() else {
                continue;
            };
            let before = self
                .edge_type_live_counts
                .read()
                .get(type_id)
                .copied()
                .ok_or_else(|| invalid("edge type has no live counter"))?;
            captured
                .type_counts
                .push((type_id, subtract_count(before, group.len())?));
        }
        Ok(())
    }

    #[cfg(not(feature = "tiered-storage"))]
    fn node_data_stamp(&self, id: NodeId) -> Option<StructuralStamp> {
        let nodes = self.nodes.read();
        node_stamp(nodes.get(&id)?)
    }

    #[cfg(feature = "tiered-storage")]
    fn node_data_stamp(&self, id: NodeId) -> Option<StructuralStamp> {
        let (stamp, reference) = {
            let nodes = self.node_versions.read();
            let index = nodes.get(&id)?;
            (node_stamp(index)?, index.latest()?)
        };
        (!self.read_node_record(&reference)?.is_deleted()).then_some(stamp)
    }

    #[cfg(not(feature = "tiered-storage"))]
    fn edge_data_stamp(&self, id: EdgeId) -> Option<(StructuralStamp, EdgeRecord)> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        Some((edge_stamp(chain)?, *chain.latest()?))
    }

    #[cfg(feature = "tiered-storage")]
    fn edge_data_stamp(&self, id: EdgeId) -> Option<(StructuralStamp, EdgeRecord)> {
        let (stamp, reference) = {
            let edges = self.edge_versions.read();
            let index = edges.get(&id)?;
            (edge_stamp(index)?, index.latest()?)
        };
        let record = self.read_edge_record(&reference)?;
        (!record.is_deleted()).then_some((stamp, record))
    }

    #[cfg(test)]
    pub(super) fn pending_creation_proof_for_test<'store, 'transition, 'ids>(
        &'store self,
        transition: &'transition PinnedLpgTransition<'store>,
        transaction_id: TransactionId,
        ids: &'ids [NodeId],
    ) -> Result<PendingNodeCreationProof<'store, 'transition, 'ids>> {
        if !std::ptr::eq(self, transition.store)
            || transaction_id == TransactionId::SYSTEM
            || transaction_id == TransactionId::INVALID
            || ids.windows(2).any(|pair| pair[0] >= pair[1])
            || ids.iter().any(|id| {
                !self
                    .node_data_stamp(*id)
                    .is_some_and(|stamp| stamp.owns_creation(transaction_id))
            })
        {
            return Err(invalid(
                "test creation proof failed the structural creation qualifier",
            ));
        }
        Ok(PendingNodeCreationProof {
            store: self,
            transition,
            transaction_id,
            ids,
        })
    }
}

#[cfg(test)]
impl<'store, 'workspace, 'transition> ReleasedStoreData<'store, 'workspace, 'transition> {
    /// Final structural/derived lock acquisition. All capacity was completed
    /// before these entity writers. Success and rejection allocate nothing.
    /// Materialize an error only after dropping ALL stores' final writers.
    pub(super) fn rebind(
        self,
    ) -> std::result::Result<PreparedStoreData<'store, 'workspace, 'transition>, DataRebindError>
    {
        if !std::ptr::eq(self.store, self.transition.store) {
            return Err(DataRebindError::new("data transition target changed"));
        }
        let scope = DataCommitScope {
            transition: self.transition,
        };
        let guards = StoreDataGuards::rebind(self.workspace, &scope)?;
        Ok(PreparedStoreData {
            transition: self.transition,
            workspace: self.workspace,
            guards,
        })
    }
}

impl<'store> StoreDataGuards<'store> {
    fn rebind(
        workspace: &mut StoreDataWorkspace,
        scope: &DataCommitScope<'store, '_>,
    ) -> std::result::Result<Self, DataRebindError> {
        let store = scope.transition.store;
        let captured = &workspace.captured;
        #[cfg(not(feature = "tiered-storage"))]
        let structural = StructuralWriters {
            nodes: store
                .nodes
                .try_write()
                .ok_or(DataRebindError::Conflict("commit nodes are in use"))?,
            edges: store
                .edges
                .try_write()
                .ok_or(DataRebindError::Conflict("commit edges are in use"))?,
        };
        #[cfg(feature = "tiered-storage")]
        let structural = StructuralWriters {
            nodes: store
                .node_versions
                .try_write()
                .ok_or(DataRebindError::Conflict("commit node versions are in use"))?,
            edges: store
                .edge_versions
                .try_write()
                .ok_or(DataRebindError::Conflict("commit edge versions are in use"))?,
        };
        for node in &captured.nodes {
            if structural.nodes.get(&node.id).and_then(node_stamp) != Some(node.stamp) {
                return Err(DataRebindError::new(
                    "qualified node lifetime changed during preparation",
                ));
            }
        }
        for edge in &captured.edges {
            if structural.edges.get(&edge.id).and_then(edge_stamp) != Some(edge.stamp) {
                return Err(DataRebindError::new(
                    "qualified edge lifetime changed during preparation",
                ));
            }
        }
        let labels = LabelDataGuards::rebind_in_scope(&workspace.labels, scope)?;
        let type_counts =
            store
                .edge_type_live_counts
                .try_write()
                .ok_or(DataRebindError::Conflict(
                    "commit edge type counts are in use",
                ))?;
        let node_properties = PropertyDataGuards::rebind_in_scope(
            &store.node_properties,
            &workspace.node_properties,
            scope,
        )?;
        let edge_properties = PropertyDataGuards::rebind_in_scope(
            &store.edge_properties,
            &workspace.edge_properties,
            scope,
        )?;
        let forward = AdjacencyDataGuards::rebind_in_scope(
            &store.forward_adj,
            &mut workspace.forward,
            scope,
        )?;
        let backward = match &store.backward_adj {
            Some(backward) => Some(AdjacencyDataGuards::rebind_in_scope(
                backward,
                &mut workspace.backward,
                scope,
            )?),
            None => None,
        };
        let bookkeeping =
            BookkeepingWriters {
                undo: store
                    .property_undo_log
                    .try_write()
                    .ok_or(DataRebindError::Conflict("commit property undo is in use"))?,
                creates: store
                    .pending_tx_creates
                    .try_write()
                    .ok_or(DataRebindError::Conflict(
                        "commit pending creates are in use",
                    ))?,
                delta: store
                    .tx_property_overlay
                    .try_write()
                    .ok_or(DataRebindError::Conflict(
                        "commit property overlay is in use",
                    ))?,
                #[cfg(feature = "text-index")]
                text: store
                    .text_index_overlay
                    .try_write()
                    .ok_or(DataRebindError::Conflict("commit Text overlay is in use"))?,
                node_deletes: store.pending_tx_deletes.try_write().ok_or(
                    DataRebindError::Conflict("commit pending node deletes are in use"),
                )?,
                edge_deletes: store.pending_tx_edge_deletes.try_write().ok_or(
                    DataRebindError::Conflict("commit pending edge deletes are in use"),
                )?,
            };
        Ok(Self {
            bookkeeping,
            backward,
            forward,
            edge_properties,
            node_properties,
            type_counts,
            labels,
            structural,
            store,
        })
    }
}

#[cfg(test)]
impl<'store, 'workspace, 'transition> PreparedStoreData<'store, 'workspace, 'transition> {
    /// Installs qualified structural epochs, sparse histories, adjacency and
    /// bookkeeping exactly once. Never calls normal property/index mutators.
    pub(super) fn install(mut self) -> InstalledStoreDataFence<'store, 'workspace, 'transition> {
        let scope = DataCommitScope {
            transition: self.transition,
        };
        self.guards.install(self.workspace, &scope);
        InstalledStoreDataFence {
            _guards: self.guards,
            _workspace: self.workspace,
            _transition: self.transition,
        }
    }
}

impl StoreDataGuards<'_> {
    fn install(&mut self, workspace: &mut StoreDataWorkspace, scope: &DataCommitScope<'_, '_>) {
        let captured = &workspace.captured;
        let retired = &mut workspace.retired;
        let tx = captured.transaction_id;
        let commit = captured.commit;
        for node in &captured.nodes {
            // Keys were qualified before durability and remain under the same
            // writer; this is a capacity-proven keyed merge, not target discovery.
            if let Some(chain) = self.structural.nodes.get_mut(&node.id) {
                chain.finalize_epochs(tx, commit);
                chain.finalize_deleted_epochs(tx, commit);
                #[cfg(not(feature = "tiered-storage"))]
                if let Some(record) = chain.latest_mut() {
                    if let Some(properties) = node.record_properties {
                        record.props_count = properties;
                    }
                    if let Some(labels) = node.labels {
                        record.set_label_count(labels);
                    }
                }
            }
        }
        for edge in &captured.edges {
            if let Some(chain) = self.structural.edges.get_mut(&edge.id) {
                chain.finalize_epochs(tx, commit);
                chain.finalize_deleted_epochs(tx, commit);
            }
        }
        self.labels.install_in_scope(&mut workspace.labels, scope);
        self.node_properties
            .install_in_scope(&mut workspace.node_properties, scope);
        self.edge_properties
            .install_in_scope(&mut workspace.edge_properties, scope);
        self.forward.install_in_scope(&mut workspace.forward, scope);
        if let Some(backward) = &mut self.backward {
            backward.install_in_scope(&mut workspace.backward, scope);
        }
        for &(id, count) in &captured.type_counts {
            if let Some(live) = self.type_counts.get_mut(id) {
                *live = count;
            }
        }
        self.store
            .live_node_count
            .store(captured.node_count, Ordering::Relaxed);
        self.store
            .live_edge_count
            .store(captured.edge_count, Ordering::Relaxed);
        self.store
            .current_epoch
            .fetch_max(commit.as_u64(), Ordering::AcqRel);
        retired.creates = self.bookkeeping.creates.remove(&tx);
        retired.delta = self.bookkeeping.delta.remove(&tx);
        retired.undo = self.bookkeeping.undo.remove(&tx);
        retired.node_deletes = self.bookkeeping.node_deletes.remove(&tx);
        retired.edge_deletes = self.bookkeeping.edge_deletes.remove(&tx);
        #[cfg(feature = "text-index")]
        {
            retired.text = self.bookkeeping.text.remove(&tx);
        }
    }
}

#[cfg(not(feature = "tiered-storage"))]
fn node_stamp(chain: &VersionChain<NodeRecord>) -> Option<StructuralStamp> {
    let (info, record) = chain.history().next()?;
    (!record.is_deleted()).then(|| StructuralStamp::from_info(*info, chain.version_count()))
}

#[cfg(not(feature = "tiered-storage"))]
fn edge_stamp(chain: &VersionChain<EdgeRecord>) -> Option<StructuralStamp> {
    let (info, record) = chain.history().next()?;
    (!record.is_deleted()).then(|| StructuralStamp::from_info(*info, chain.version_count()))
}

#[cfg(feature = "tiered-storage")]
fn node_stamp(index: &VersionIndex) -> Option<StructuralStamp> {
    tiered_stamp(index)
}
#[cfg(feature = "tiered-storage")]
fn edge_stamp(index: &VersionIndex) -> Option<StructuralStamp> {
    tiered_stamp(index)
}

#[cfg(feature = "tiered-storage")]
fn tiered_stamp(index: &VersionIndex) -> Option<StructuralStamp> {
    let reference = index.latest()?;
    let deleter = match reference {
        VersionRef::Hot(hot) => hot.deleted_by,
        VersionRef::Cold(cold) => cold.deleted_by,
        _ => return None,
    };
    let mut stamp = StructuralStamp::from_info(
        VersionInfo {
            created_epoch: reference.epoch(),
            created_by: reference.created_by(),
            deleted_epoch: reference.deleted_epoch(),
            deleted_by: deleter,
        },
        index.hot_count().checked_add(index.cold_count())?,
    );
    stamp.creation_is_hot = reference.is_hot();
    Some(stamp)
}

fn property_value(op: &PropOp) -> Option<grafeo_common::types::Value> {
    match op {
        PropOp::Set(value) => Some(value.clone()),
        PropOp::Remove => None,
    }
}

fn subtract_count(before: i64, deleted: usize) -> Result<i64> {
    let deleted =
        i64::try_from(deleted).map_err(|_| invalid("delete count cannot be represented"))?;
    before
        .checked_sub(deleted)
        .filter(|count| *count >= 0)
        .ok_or_else(|| invalid("live counter underflow"))
}

fn invalid(message: &str) -> Error {
    TransactionError::InvalidState(format!("buffered data publication: {message}")).into()
}

#[cfg(test)]
mod tests;
