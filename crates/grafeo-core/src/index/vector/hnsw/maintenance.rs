//! Sparse private HNSW postimages for aggregate publication.
//!
//! This is not a standalone commit API. The coordinator owns registration,
//! graph final-row qualification and the surrounding store transition. It must
//! exclude graph-accessor readers BEFORE reacquiring final graph writers.

use super::{
    Arc, HashMap, HashSet, HnswIndex, HnswMutationGuard, HnswNeighborsIter, HnswNode, HnswRng,
    NodeId, Ordering, TopologyBackend, TopologyRead, VectorAccessor,
};
#[cfg(test)]
use super::{HnswConfig, HnswExactState, MmapTopology};
use crate::graph::lpg::DataRebindError;
use grafeo_common::memory::AllocError;
use grafeo_common::utils::error::{Error, Result, TransactionError};
use parking_lot::RwLockWriteGuard;

mod wal;

enum MaintenanceOutput {
    State,
    StateAndWal,
}

/// Continuous exclusion against aliases that mutate this exact index.
pub(crate) struct HnswMaintenancePin<'index> {
    index: &'index HnswIndex,
    _mutation: HnswMutationGuard<'index>,
    preparation_claimed: std::cell::Cell<bool>,
}

impl Drop for HnswMaintenancePin<'_> {
    fn drop(&mut self) {
        // The alias guard still exists while this destructor runs.
        self.index
            .maintenance_active
            .store(false, Ordering::Release);
    }
}

/// Exclusive parallel-reader admission, acquired before final graph writers.
#[cfg(test)]
pub(crate) struct HnswReaderFence<'index, 'pin> {
    _readers: RwLockWriteGuard<'index, ()>,
    pin: &'pin HnswMaintenancePin<'index>,
}

impl HnswIndex {
    pub(crate) fn pin_maintenance(&self) -> Result<HnswMaintenancePin<'_>> {
        let mutation = self
            .pin_mutation()
            .ok_or_else(|| invalid("write authority denied"))?;
        self.maintenance_active.store(true, Ordering::Release);
        Ok(HnswMaintenancePin {
            index: self,
            _mutation: mutation,
            preparation_claimed: std::cell::Cell::new(false),
        })
    }
}

impl<'index> HnswMaintenancePin<'index> {
    #[cfg(test)]
    pub(crate) fn exclude_readers(&self) -> HnswReaderFence<'index, '_> {
        HnswReaderFence {
            _readers: self.index.reader_admission.write(),
            pin: self,
        }
    }

    #[cfg(test)]
    pub(crate) fn try_exclude_readers(&self) -> Option<HnswReaderFence<'index, '_>> {
        self.index
            .reader_admission
            .try_write()
            .map(|readers| HnswReaderFence {
                _readers: readers,
                pin: self,
            })
    }

    #[cfg(test)]
    pub(crate) fn state_guards_available_for_test(&self) -> bool {
        StateGuards::try_acquire(self.index).is_ok()
    }

    /// `vectors` must be the coordinator's immutable, non-recording final-row
    /// accessor, never a callback that mutates this index or resolves live
    /// transaction state. Every returned vector is dimension-checked here.
    #[cfg(test)]
    pub(crate) fn prepare<'workspace, 'pin>(
        &'pin self,
        workspace: &'workspace mut HnswMaintenanceWorkspace,
        vectors: &impl VectorAccessor,
    ) -> Result<PreparedHnswMaintenance<'index, 'workspace, 'pin>> {
        let guards = self.prepare_guards(workspace, vectors)?;
        Ok(PreparedHnswMaintenance {
            guards,
            workspace,
            pin: self,
        })
    }

    /// Completes the sparse postimage and releases initial state writers while
    /// the exact pin remains borrowed by the outer slot coordinator.
    pub(in crate::index::vector) fn prepare_workspace(
        &self,
        workspace: &mut HnswMaintenanceWorkspace,
        vectors: &impl VectorAccessor,
    ) -> Result<()> {
        drop(self.prepare_guards(workspace, vectors)?);
        Ok(())
    }

    fn prepare_guards(
        &self,
        workspace: &mut HnswMaintenanceWorkspace,
        vectors: &impl VectorAccessor,
    ) -> Result<StateGuards<'index>> {
        // One shared pin must not admit independently installable postimages:
        // installing one would invalidate the other's exact baseline/capacity.
        // Reject before claiming the workspace, so a fresh pin can retry it.
        if self.preparation_claimed.replace(true) {
            return Err(invalid("maintenance pin preparation is one-shot"));
        }
        if workspace.attempted {
            return Err(invalid("workspace preparation is one-shot"));
        }
        workspace.attempted = true;
        if let Some(payload) = workspace.recorded.take() {
            let result = wal::prepare_recorded(self.index, workspace, &payload);
            workspace.recorded = Some(payload);
            result?;
        } else {
            workspace.normalize(self.index.config.dimensions)?;
            let baseline = self.index.nodes.read();
            workspace.entry_point = *self.index.entry_point.read();
            workspace.max_level = *self.index.max_level.read();
            workspace.rng = *self.index.rng.read();
            workspace.base_len = baseline.len();
            workspace.mmap_additional = match &*baseline {
                TopologyBackend::Heap(_) => 0,
                TopologyBackend::Mmap {
                    additional_nodes, ..
                } => *additional_nodes,
            };
            let accessor = CheckedAccessor {
                source: vectors,
                dimensions: self.index.config.dimensions,
                invalid: std::sync::atomic::AtomicBool::new(false),
            };
            for position in 0..workspace.operations.len() {
                let (id, vector) = workspace
                    .operations
                    .get(position)
                    .ok_or_else(|| invalid("normalized operation is absent"))?;
                let id = *id;
                let vector = vector.clone();
                match vector {
                    Some(vector) => {
                        stage_insert(self.index, &baseline, workspace, id, &vector, &accessor)?;
                    }
                    None => {
                        if baseline.contains(id) || workspace.nodes.contains_key(&id) {
                            reserve_map(&mut workspace.deleted, 1)?;
                            workspace.deleted.insert(id, true);
                        }
                    }
                }
                if accessor.invalid.load(Ordering::Relaxed) {
                    return Err(invalid("final-row routing vector has wrong dimensions"));
                }
            }
            if matches!(workspace.output, MaintenanceOutput::StateAndWal) {
                wal::capture(self.index, &baseline, workspace)?;
            }
            drop(baseline);
        }
        // Alias exclusion remains continuous while upgrading only the inner
        // state locks for capacity reservation. Staging itself admits readers.
        let mut guards = StateGuards::acquire(self.index);
        let missing_nodes = match &*guards.nodes {
            TopologyBackend::Heap(nodes) => workspace
                .nodes
                .keys()
                .filter(|id| !nodes.contains_key(*id))
                .count(),
            TopologyBackend::Mmap {
                base, overrides, ..
            } => {
                let new_identities = workspace
                    .nodes
                    .keys()
                    .filter(|id| !base.contains(**id) && !overrides.contains_key(*id))
                    .count();
                workspace.mmap_additional = workspace
                    .mmap_additional
                    .checked_add(new_identities)
                    .ok_or(AllocError::InsufficientSpace)?;
                workspace
                    .nodes
                    .keys()
                    .filter(|id| !overrides.contains_key(*id))
                    .count()
            }
        };
        let target_nodes = match &mut *guards.nodes {
            TopologyBackend::Heap(nodes)
            | TopologyBackend::Mmap {
                overrides: nodes, ..
            } => nodes,
        };
        reserve_map(target_nodes, missing_nodes)?;
        let missing_deletes = workspace
            .deleted
            .iter()
            .filter(|(id, deleted)| **deleted && !guards.deleted.contains(*id))
            .count();
        guards
            .deleted
            .try_reserve(missing_deletes)
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace
            .retired_nodes
            .try_reserve(workspace.nodes.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace.prepared = true;
        Ok(guards)
    }
}

/// Outer-owned operations, sparse candidate nodes and displaced topology.
/// Construct before all enclosing publication guards; retire after they drain.
pub(crate) struct HnswMaintenanceWorkspace {
    raw: Vec<(NodeId, Option<Arc<[f32]>>)>,
    normalized: HashMap<NodeId, Option<Arc<[f32]>>>,
    operations: Vec<(NodeId, Option<Arc<[f32]>>)>,
    nodes: HashMap<NodeId, HnswNode>,
    deleted: HashMap<NodeId, bool>,
    entry_point: Option<NodeId>,
    max_level: usize,
    rng: HnswRng,
    base_len: usize,
    added_nodes: usize,
    mmap_additional: usize,
    retired_nodes: Vec<HnswNode>,
    retired_candidate_layers: Vec<Vec<NodeId>>,
    recorded: Option<Vec<u8>>,
    output: MaintenanceOutput,
    wal_baseline: Option<wal::Captured>,
    final_presence: Vec<(NodeId, bool)>,
    normalized_ready: bool,
    attempted: bool,
    prepared: bool,
}

impl HnswMaintenanceWorkspace {
    /// Repeated upserts are last-write-wins; explicit final absence dominates.
    pub(crate) fn new(operations: Vec<(NodeId, Option<Arc<[f32]>>)>) -> Self {
        Self {
            raw: operations,
            normalized: HashMap::new(),
            operations: Vec::new(),
            nodes: HashMap::new(),
            deleted: HashMap::new(),
            entry_point: None,
            max_level: 0,
            rng: HnswRng::from_state(0),
            base_len: 0,
            added_nodes: 0,
            mmap_additional: 0,
            retired_nodes: Vec::new(),
            retired_candidate_layers: Vec::new(),
            recorded: None,
            output: MaintenanceOutput::State,
            wal_baseline: None,
            final_presence: Vec::new(),
            normalized_ready: false,
            attempted: false,
            prepared: false,
        }
    }

    fn normalize(&mut self, dimensions: usize) -> Result<()> {
        if self.recorded.is_some() {
            return Err(invalid("recorded topology has no vector input rows"));
        }
        if self.normalized_ready {
            return self.validate_dimensions(dimensions);
        }
        reserve_map(&mut self.normalized, self.raw.len())?;
        for (id, vector) in &self.raw {
            if !id.is_valid() {
                return Err(invalid("invalid final-row node identity"));
            }
            let entry = self.normalized.entry(*id).or_insert_with(|| vector.clone());
            if entry.is_some() {
                entry.clone_from(vector);
            }
        }
        self.operations
            .try_reserve(self.normalized.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        self.operations.extend(
            self.normalized
                .iter()
                .map(|(id, vector)| (*id, vector.clone())),
        );
        self.operations.sort_unstable_by_key(|(id, _)| *id);
        self.validate_dimensions(dimensions)?;
        self.normalized_ready = true;
        Ok(())
    }

    fn validate_dimensions(&self, dimensions: usize) -> Result<()> {
        if self.operations.iter().any(|(_, vector)| {
            vector
                .as_ref()
                .is_some_and(|vector| vector.len() != dimensions)
        }) {
            return Err(invalid("final-row vector has wrong dimensions"));
        }
        Ok(())
    }

    /// Auxiliary quantized state and topology consume this exact final order.
    /// This only prepares outer-owned input; the pin still admits one complete
    /// topology preparation, including when normalization already succeeded.
    pub(crate) fn normalized_operations(
        &mut self,
        dimensions: usize,
    ) -> Result<&[(NodeId, Option<Arc<[f32]>>)]> {
        self.normalize(dimensions)?;
        Ok(&self.operations)
    }

    fn touch<'a>(&'a mut self, base: &TopologyBackend, id: NodeId) -> Result<&'a mut HnswNode> {
        if !self.nodes.contains_key(&id) {
            reserve_map(&mut self.nodes, 1)?;
            let mut layers = Vec::new();
            let mut layer = 0;
            while let Some(neighbors) = base.neighbors_at(id, layer) {
                layers.push(neighbors.collect());
                layer += 1;
            }
            if layers.is_empty() {
                return Err(invalid("staged neighbor is absent from baseline topology"));
            }
            self.nodes.insert(id, HnswNode { neighbors: layers });
        }
        self.nodes
            .get_mut(&id)
            .ok_or_else(|| invalid("staged topology node is absent"))
    }

    fn replace_layer(
        &mut self,
        base: &TopologyBackend,
        id: NodeId,
        layer: usize,
        replacement: Vec<NodeId>,
    ) -> Result<()> {
        // Both the completed replacement and the displaced candidate survive
        // later errors in the outer workspace.
        // Caller reserves this retirement slot before constructing replacement.
        self.retired_candidate_layers.push(replacement);
        let replacement_position = self.retired_candidate_layers.len() - 1;
        self.touch(base, id)?;
        let node = self
            .nodes
            .get_mut(&id)
            .ok_or_else(|| invalid("staged topology node is absent"))?;
        let target = node
            .neighbors
            .get_mut(layer)
            .ok_or_else(|| invalid("staged layer is absent"))?;
        let replacement = self
            .retired_candidate_layers
            .get_mut(replacement_position)
            .ok_or_else(|| invalid("staged layer replacement is absent"))?;
        std::mem::swap(target, replacement);
        Ok(())
    }
}

struct Shadow<'a> {
    base: &'a TopologyBackend,
    workspace: &'a HnswMaintenanceWorkspace,
}

impl TopologyRead for Shadow<'_> {
    fn len(&self) -> usize {
        self.workspace.base_len + self.workspace.added_nodes
    }

    fn neighbors_at(&self, id: NodeId, layer: usize) -> Option<HnswNeighborsIter<'_>> {
        if let Some(node) = self.workspace.nodes.get(&id) {
            node.neighbors
                .get(layer)
                .map(|neighbors| HnswNeighborsIter::Heap(neighbors.iter()))
        } else {
            self.base.neighbors_at(id, layer)
        }
    }
}

struct CheckedAccessor<'a, T> {
    source: &'a T,
    dimensions: usize,
    invalid: std::sync::atomic::AtomicBool,
}

impl<T: VectorAccessor> VectorAccessor for CheckedAccessor<'_, T> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        let vector = self.source.get_vector(id)?;
        if vector.len() != self.dimensions {
            self.invalid.store(true, Ordering::Relaxed);
            None
        } else {
            Some(vector)
        }
    }
}

fn stage_insert(
    index: &HnswIndex,
    base: &TopologyBackend,
    workspace: &mut HnswMaintenanceWorkspace,
    id: NodeId,
    vector: &[f32],
    accessor: &impl VectorAccessor,
) -> Result<()> {
    let existing = base.contains(id) || workspace.nodes.contains_key(&id);
    if !existing {
        let next_count = workspace
            .base_len
            .checked_add(workspace.added_nodes)
            .and_then(|count| count.checked_add(1))
            .ok_or(AllocError::InsufficientSpace)?;
        if index
            .config
            .max_elements
            .is_some_and(|max| next_count > max)
        {
            return Err(invalid("index maximum element count would be exceeded"));
        }
        reserve_map(&mut workspace.nodes, 1)?;
    }
    let sampled = HnswIndex::sample_level(&mut workspace.rng, index.config.ml);
    let level = if existing {
        workspace.touch(base, id)?.neighbors.len().saturating_sub(1)
    } else {
        workspace.nodes.insert(
            id,
            HnswNode {
                neighbors: vec![Vec::new(); sampled + 1],
            },
        );
        workspace.added_nodes += 1;
        sampled
    };
    reserve_map(&mut workspace.deleted, 1)?;
    workspace.deleted.insert(id, false);
    let Some(entry) = workspace.entry_point else {
        workspace.entry_point = Some(id);
        workspace.max_level = level;
        return Ok(());
    };
    let current_max = workspace.max_level;
    let mut current_entry = entry;
    for layer in (level + 1..=current_max).rev() {
        current_entry = index.search_layer_single(
            &Shadow { base, workspace },
            accessor,
            vector,
            current_entry,
            layer,
        );
    }
    for layer in (0..=level.min(current_max)).rev() {
        let max_neighbors = if layer == 0 {
            index.config.m_max
        } else {
            index.config.m
        };
        let mut candidates = index.search_layer(
            &Shadow { base, workspace },
            accessor,
            vector,
            current_entry,
            index.config.ef_construction,
            layer,
        );
        candidates.retain(|neighbor| neighbor.id != id);
        let selected = index.select_neighbors_heuristic(accessor, &candidates, max_neighbors);
        workspace
            .retired_candidate_layers
            .try_reserve(1)
            .map_err(|_| AllocError::OutOfMemory)?;
        workspace.replace_layer(base, id, layer, selected.clone())?;
        let mut pruning = Vec::new();
        for &neighbor_id in &selected {
            if (Shadow { base, workspace })
                .neighbors_at(neighbor_id, layer)
                .is_none()
            {
                continue;
            }
            let neighbor = workspace.touch(base, neighbor_id)?;
            if let Some(neighbors) = neighbor.neighbors.get_mut(layer)
                && !neighbors.contains(&id)
            {
                neighbors.push(id);
                if neighbors.len() > max_neighbors {
                    pruning.push(neighbor_id);
                }
            }
        }
        for neighbor_id in pruning {
            let Some(base_vector) = accessor.get_vector(neighbor_id) else {
                continue;
            };
            let distances: Vec<_> = Shadow { base, workspace }
                .neighbors_at(neighbor_id, layer)
                .into_iter()
                .flatten()
                .map(|other| {
                    let distance = accessor.get_vector(other).map_or(f32::MAX, |value| {
                        index.vector_distance(&base_vector, &value)
                    });
                    (other, distance)
                })
                .collect();
            workspace
                .retired_candidate_layers
                .try_reserve(1)
                .map_err(|_| AllocError::OutOfMemory)?;
            let pruned = HnswIndex::pruned_neighbors_with_distances(&distances, max_neighbors);
            workspace.replace_layer(base, neighbor_id, layer, pruned)?;
        }
        if let Some(first) = selected.first() {
            current_entry = *first;
        }
    }
    if level > current_max {
        workspace.entry_point = Some(id);
        workspace.max_level = level;
    }
    Ok(())
}

struct StateGuards<'index> {
    deleted: RwLockWriteGuard<'index, HashSet<NodeId>>,
    rng: RwLockWriteGuard<'index, HnswRng>,
    max_level: RwLockWriteGuard<'index, usize>,
    entry_point: RwLockWriteGuard<'index, Option<NodeId>>,
    nodes: RwLockWriteGuard<'index, TopologyBackend>,
}

impl<'index> StateGuards<'index> {
    fn acquire(index: &'index HnswIndex) -> Self {
        let nodes = index.nodes.write();
        let entry_point = index.entry_point.write();
        let max_level = index.max_level.write();
        let rng = index.rng.write();
        let deleted = index.deleted.write();
        Self {
            deleted,
            rng,
            max_level,
            entry_point,
            nodes,
        }
    }

    /// Final rebind must not park while companion final guards are retained.
    /// Pure index readers need not join graph-accessor admission, so contention
    /// remains possible; static conflicts retire partial guards without parking
    /// bookkeeping or public-error allocation.
    fn try_acquire(index: &'index HnswIndex) -> std::result::Result<Self, DataRebindError> {
        let nodes = index
            .nodes
            .try_write()
            .ok_or(DataRebindError::Conflict("HNSW topology is busy"))?;
        let entry_point = index
            .entry_point
            .try_write()
            .ok_or(DataRebindError::Conflict("HNSW entry point is busy"))?;
        let max_level = index
            .max_level
            .try_write()
            .ok_or(DataRebindError::Conflict("HNSW maximum level is busy"))?;
        let rng = index
            .rng
            .try_write()
            .ok_or(DataRebindError::Conflict("HNSW RNG is busy"))?;
        let deleted = index
            .deleted
            .try_write()
            .ok_or(DataRebindError::Conflict("HNSW deletion state is busy"))?;
        Ok(Self {
            deleted,
            rng,
            max_level,
            entry_point,
            nodes,
        })
    }
}

#[cfg(test)]
pub(crate) struct PreparedHnswMaintenance<'index, 'workspace, 'pin> {
    guards: StateGuards<'index>,
    workspace: &'workspace mut HnswMaintenanceWorkspace,
    pin: &'pin HnswMaintenancePin<'index>,
}

#[cfg(test)]
pub(crate) struct ReleasedHnswMaintenance<'index, 'workspace, 'pin> {
    // The quantized test adapter reconstructs this loan only after invoking
    // the same shared preparation used by its production outer slots.
    pub(in crate::index::vector) workspace: &'workspace mut HnswMaintenanceWorkspace,
    pub(in crate::index::vector) pin: &'pin HnswMaintenancePin<'index>,
}

#[cfg(test)]
impl<'index, 'workspace, 'pin> PreparedHnswMaintenance<'index, 'workspace, 'pin> {
    pub(crate) fn release(self) -> ReleasedHnswMaintenance<'index, 'workspace, 'pin> {
        drop(self.guards);
        ReleasedHnswMaintenance {
            workspace: self.workspace,
            pin: self.pin,
        }
    }
}

#[cfg(test)]
pub(crate) struct ReadyHnswMaintenance<'index, 'workspace, 'pin, 'fence> {
    prepared: PreparedHnswMaintenance<'index, 'workspace, 'pin>,
    readers: &'fence HnswReaderFence<'index, 'pin>,
}

#[cfg(test)]
pub(crate) struct InstalledHnswMaintenance<'index, 'workspace, 'pin, 'fence> {
    _ready: ReadyHnswMaintenance<'index, 'workspace, 'pin, 'fence>,
}

#[cfg(test)]
impl<'index, 'workspace, 'pin> ReleasedHnswMaintenance<'index, 'workspace, 'pin> {
    pub(crate) fn rebind<'fence>(
        self,
        readers: &'fence HnswReaderFence<'index, 'pin>,
    ) -> std::result::Result<ReadyHnswMaintenance<'index, 'workspace, 'pin, 'fence>, DataRebindError>
    {
        if !std::ptr::eq(self.pin, readers.pin) || !self.workspace.prepared {
            return Err(DataRebindError::new(
                "HNSW maintenance reader fence does not match prepared target",
            ));
        }
        let guards = StateGuards::try_acquire(self.pin.index)?;
        Ok(ReadyHnswMaintenance {
            prepared: PreparedHnswMaintenance {
                guards,
                workspace: self.workspace,
                pin: self.pin,
            },
            readers,
        })
    }
}

#[cfg(test)]
impl<'index, 'workspace, 'pin, 'fence> ReadyHnswMaintenance<'index, 'workspace, 'pin, 'fence> {
    pub(crate) fn install(mut self) -> InstalledHnswMaintenance<'index, 'workspace, 'pin, 'fence> {
        install_postimage(&mut *self.prepared.workspace, &mut self.prepared.guards);
        InstalledHnswMaintenance { _ready: self }
    }

    pub(crate) fn release(self) -> ReleasedHnswMaintenance<'index, 'workspace, 'pin> {
        let _readers = self.readers;
        self.prepared.release()
    }
}

/// Guard storage for an independently anchored target. It contains no loan of
/// a local pin or candidate workspace. Only the vector batch's private, paired
/// slots use this type; the batch retains the exact pin loan throughout.
pub(in crate::index::vector) struct HnswGuardSlot<'index> {
    state: Option<StateGuards<'index>>,
    readers: Option<RwLockWriteGuard<'index, ()>>,
    index: &'index HnswIndex,
}

impl<'index> HnswGuardSlot<'index> {
    pub(in crate::index::vector) fn new(index: &'index HnswIndex) -> Self {
        Self {
            state: None,
            readers: None,
            index,
        }
    }

    pub(in crate::index::vector) fn matches(&self, pin: &HnswMaintenancePin<'_>) -> bool {
        std::ptr::eq(self.index, pin.index)
    }

    pub(in crate::index::vector) fn exclude_readers(
        &mut self,
        pin: &HnswMaintenancePin<'index>,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'index, '_>,
    ) -> std::result::Result<(), DataRebindError> {
        if !scope.pins_hnsw(pin)
            || !self.matches(pin)
            || self.readers.is_some()
            || self.state.is_some()
        {
            return Err(DataRebindError::new(
                "HNSW reader slot does not match an unbound target",
            ));
        }
        // Every batch reader exclusion precedes every final graph writer.
        self.readers = Some(self.index.reader_admission.write());
        Ok(())
    }

    pub(in crate::index::vector) fn rebind(
        &mut self,
        pin: &HnswMaintenancePin<'index>,
        workspace: &HnswMaintenanceWorkspace,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'index, '_>,
    ) -> std::result::Result<(), DataRebindError> {
        if !scope.pins_hnsw(pin)
            || !self.matches(pin)
            || !workspace.prepared
            || self.readers.is_none()
            || self.state.is_some()
        {
            return Err(DataRebindError::new(
                "HNSW state slot lacks its prepared target and reader exclusion",
            ));
        }
        self.state = Some(StateGuards::try_acquire(self.index)?);
        Ok(())
    }

    pub(in crate::index::vector) fn install(
        &mut self,
        workspace: &mut HnswMaintenanceWorkspace,
        _scope: &crate::index::vector::maintenance::VectorCommitScope<'_, '_>,
    ) {
        // The private ready batch qualified this exact slot and retains its
        // exclusive mutable loan. No method can clear it between bind/install.
        if let Some(state) = &mut self.state {
            install_postimage(workspace, state);
        }
    }

    pub(in crate::index::vector) fn release_state(&mut self) {
        drop(self.state.take());
    }

    pub(in crate::index::vector) fn release_readers(&mut self) {
        drop(self.readers.take());
    }
}

impl Drop for HnswGuardSlot<'_> {
    fn drop(&mut self) {
        self.release_state();
        self.release_readers();
    }
}

/// Candidate and raw guard slots stay together in the caller's outer buffer.
/// The index is borrowed from an external anchor, never from a sibling Arc.
pub(in crate::index::vector) struct HnswMaintenanceSlot<'index> {
    guards: HnswGuardSlot<'index>,
    workspace: HnswMaintenanceWorkspace,
}

impl<'index> HnswMaintenanceSlot<'index> {
    pub(in crate::index::vector) fn new(
        index: &'index HnswIndex,
        rows: Vec<(NodeId, Option<Arc<[f32]>>)>,
    ) -> Self {
        Self {
            guards: HnswGuardSlot::new(index),
            workspace: HnswMaintenanceWorkspace::new(rows),
        }
    }

    pub(in crate::index::vector) fn matches(&self, pin: &HnswMaintenancePin<'_>) -> bool {
        self.guards.matches(pin)
    }

    pub(in crate::index::vector) fn prepare(
        &mut self,
        pin: &HnswMaintenancePin<'index>,
        vectors: &impl VectorAccessor,
    ) -> Result<()> {
        if !self.matches(pin) {
            return Err(invalid("maintenance slot belongs to a different target"));
        }
        pin.prepare_workspace(&mut self.workspace, vectors)
    }

    pub(in crate::index::vector) fn exclude_readers(
        &mut self,
        pin: &HnswMaintenancePin<'index>,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'index, '_>,
    ) -> std::result::Result<(), DataRebindError> {
        self.guards.exclude_readers(pin, scope)
    }

    pub(in crate::index::vector) fn rebind(
        &mut self,
        pin: &HnswMaintenancePin<'index>,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'index, '_>,
    ) -> std::result::Result<(), DataRebindError> {
        self.guards.rebind(pin, &self.workspace, scope)
    }

    pub(in crate::index::vector) fn install(
        &mut self,
        scope: &crate::index::vector::maintenance::VectorCommitScope<'_, '_>,
    ) {
        self.guards.install(&mut self.workspace, scope);
    }

    pub(in crate::index::vector) fn release_state(&mut self) {
        self.guards.release_state();
    }

    pub(in crate::index::vector) fn release_readers(&mut self) {
        self.guards.release_readers();
    }
}

fn install_postimage(workspace: &mut HnswMaintenanceWorkspace, guards: &mut StateGuards<'_>) {
    let nodes = match &mut *guards.nodes {
        TopologyBackend::Heap(nodes) => nodes,
        TopologyBackend::Mmap {
            overrides,
            additional_nodes,
            ..
        } => {
            *additional_nodes = workspace.mmap_additional;
            overrides
        }
    };
    for (id, node) in workspace.nodes.drain() {
        if let Some(target) = nodes.get_mut(&id) {
            workspace
                .retired_nodes
                .push(std::mem::replace(target, node));
        } else {
            nodes.entry(id).or_insert(node);
        }
    }
    for (id, deleted) in &workspace.deleted {
        if *deleted {
            if !guards.deleted.contains(id) {
                guards.deleted.insert(*id);
            }
        } else {
            guards.deleted.remove(id);
        }
    }
    *guards.entry_point = workspace.entry_point;
    *guards.max_level = workspace.max_level;
    *guards.rng = workspace.rng;
    workspace.prepared = false;
}

fn reserve_map<K: Eq + std::hash::Hash, V>(
    map: &mut HashMap<K, V>,
    additional: usize,
) -> Result<()> {
    map.try_reserve(additional)
        .map_err(|_| AllocError::OutOfMemory.into())
}

fn invalid(reason: &str) -> Error {
    TransactionError::InvalidState(format!("HNSW maintenance preparation: {reason}")).into()
}

#[cfg(test)]
mod tests;
