//! Bounded recorded topology fragments for the existing maintenance installer.

use super::{
    HnswGuardSlot, HnswIndex, HnswMaintenanceSlot, HnswMaintenanceWorkspace, HnswNode, HnswRng,
    TopologyBackend, invalid, reserve_map,
};
use crate::index::vector::{DistanceMetric, HnswConfig};
use grafeo_common::memory::AllocError;
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;

const MAGIC: [u8; 4] = *b"HNM1";
const LIMIT: usize = 16 * 1024 * 1024;
type Layers = Vec<Vec<NodeId>>;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Configuration {
    dimensions: usize,
    metric: u8,
    m: usize,
    m_max: usize,
    ef_construction: usize,
    ef: usize,
    ml_bits: u64,
    alpha_bits: u32,
    max_elements: Option<usize>,
}

impl Configuration {
    fn capture(config: &HnswConfig) -> Self {
        Self {
            dimensions: config.dimensions,
            metric: match config.metric {
                DistanceMetric::Cosine => 0,
                DistanceMetric::Euclidean => 1,
                DistanceMetric::DotProduct => 2,
                DistanceMetric::Manhattan => 3,
            },
            m: config.m,
            m_max: config.m_max,
            ef_construction: config.ef_construction,
            ef: config.ef,
            ml_bits: config.ml.to_bits(),
            alpha_bits: config.alpha.to_bits(),
            max_elements: config.max_elements,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct State {
    entry_point: Option<NodeId>,
    max_level: usize,
    rng: u64,
    nodes: usize,
    deleted: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct Deleted {
    id: NodeId,
    before: bool,
    after: bool,
}

#[derive(Clone, Serialize, Deserialize)]
struct NodeChange {
    id: NodeId,
    before: Option<Layers>,
    after: Layers,
}

#[derive(Serialize)]
struct NodeRef<'a> {
    id: NodeId,
    before: Option<&'a Layers>,
    after: &'a Layers,
}

#[derive(Serialize, Deserialize)]
struct Wire {
    configuration: Configuration,
    before: State,
    after: State,
    operations: Vec<(NodeId, bool)>,
    nodes: Vec<NodeChange>,
    deleted: Vec<Deleted>,
}

#[derive(Serialize)]
struct WireRef<'a> {
    configuration: &'a Configuration,
    before: &'a State,
    after: State,
    operations: &'a [(NodeId, bool)],
    nodes: Vec<NodeRef<'a>>,
    deleted: &'a [Deleted],
}

/// Only touched immutable baseline rows, retained by the existing outer owner.
pub(super) struct Captured {
    configuration: Configuration,
    state: State,
    nodes: Vec<(NodeId, Option<Layers>)>,
    deleted: Vec<Deleted>,
}

fn state(index: &HnswIndex, topology: &TopologyBackend, deleted: usize) -> State {
    State {
        entry_point: *index.entry_point.read(),
        max_level: *index.max_level.read(),
        rng: index.rng.read().state,
        nodes: topology.len(),
        deleted,
    }
}

fn copy_layers(topology: &TopologyBackend, id: NodeId) -> Result<Option<Layers>> {
    if !topology.contains(id) {
        return Ok(None);
    }
    let mut layers = Vec::new();
    let mut layer = 0;
    while let Some(neighbors) = topology.neighbors_at(id, layer) {
        layers.try_reserve(1).map_err(|_| AllocError::OutOfMemory)?;
        let mut copied = Vec::new();
        for neighbor in neighbors {
            copied.try_reserve(1).map_err(|_| AllocError::OutOfMemory)?;
            copied.push(neighbor);
        }
        layers.push(copied);
        layer = layer
            .checked_add(1)
            .ok_or_else(|| invalid("baseline layer count overflows"))?;
    }
    Ok(Some(layers))
}

pub(super) fn capture(
    index: &HnswIndex,
    topology: &TopologyBackend,
    workspace: &mut HnswMaintenanceWorkspace,
) -> Result<()> {
    let deleted = index.deleted.read();
    let mut captured = Captured {
        configuration: Configuration::capture(&index.config),
        state: state(index, topology, deleted.len()),
        nodes: Vec::new(),
        deleted: Vec::new(),
    };
    captured
        .nodes
        .try_reserve(workspace.nodes.len())
        .map_err(|_| AllocError::OutOfMemory)?;
    for id in workspace.nodes.keys() {
        captured.nodes.push((*id, copy_layers(topology, *id)?));
    }
    captured.nodes.sort_unstable_by_key(|(id, _)| *id);
    captured
        .deleted
        .try_reserve(workspace.deleted.len())
        .map_err(|_| AllocError::OutOfMemory)?;
    for (id, after) in &workspace.deleted {
        captured.deleted.push(Deleted {
            id: *id,
            before: deleted.contains(id),
            after: *after,
        });
    }
    captured.deleted.sort_unstable_by_key(|change| change.id);
    workspace
        .final_presence
        .try_reserve(workspace.operations.len())
        .map_err(|_| AllocError::OutOfMemory)?;
    workspace.final_presence.extend(
        workspace
            .operations
            .iter()
            .map(|(id, vector)| (*id, vector.is_some())),
    );
    workspace.wal_baseline = Some(captured);
    Ok(())
}

impl HnswMaintenanceWorkspace {
    /// Opts in before preparation; memory-only commits retain no WAL baseline.
    pub(in crate::index::vector) fn capture_wal(&mut self) -> Result<()> {
        if self.attempted {
            return Err(invalid("WAL capture must be requested before preparation"));
        }
        self.output = super::MaintenanceOutput::StateAndWal;
        Ok(())
    }

    pub(in crate::index::vector) fn from_recorded(payload: Vec<u8>) -> Self {
        let mut workspace = Self::new(Vec::new());
        workspace.recorded = Some(payload);
        workspace
    }

    /// Final presence without duplicated vectors or fabricated replay input.
    pub(in crate::index::vector) fn prepared_final_presence(
        &self,
    ) -> impl Iterator<Item = (NodeId, bool)> + '_ {
        self.final_presence.iter().copied()
    }

    /// Shares the exact nested decode allocation accounting with quantized WAL.
    pub(in crate::index::vector) fn recorded_allocation_bytes(payload: &[u8]) -> Result<usize> {
        preflight(payload)
    }

    pub(in crate::index::vector) fn encode_wal_postimage(&self) -> Result<Vec<u8>> {
        if !self.prepared {
            return Err(invalid("WAL postimage is not prepared"));
        }
        let captured = self
            .wal_baseline
            .as_ref()
            .ok_or_else(|| invalid("WAL baseline is absent"))?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve(captured.nodes.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for (id, before) in &captured.nodes {
            let after = self
                .nodes
                .get(id)
                .ok_or_else(|| invalid("WAL candidate topology is absent"))?;
            nodes.push(NodeRef {
                id: *id,
                before: before.as_ref(),
                after: &after.neighbors,
            });
        }
        let after = State {
            entry_point: self.entry_point,
            max_level: self.max_level,
            rng: self.rng.state,
            nodes: self
                .base_len
                .checked_add(self.added_nodes)
                .ok_or_else(|| invalid("WAL node count overflows"))?,
            deleted: changed_deleted_count(captured.state.deleted, &captured.deleted)?,
        };
        let wire = WireRef {
            configuration: &captured.configuration,
            before: &captured.state,
            after,
            operations: &self.final_presence,
            nodes,
            deleted: &captured.deleted,
        };
        let mut bytes = Bounded(Vec::new());
        bytes
            .write_all(&MAGIC)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        bincode::serde::encode_into_std_write(wire, &mut bytes, bincode::config::standard())
            .map_err(|error| Error::Serialization(error.to_string()))?;
        // Reject byte-fit records that exceed the actual recovery claim budget.
        decode(&bytes.0)?;
        Ok(bytes.0)
    }
}

impl<'index> HnswMaintenanceSlot<'index> {
    pub(in crate::index::vector) fn capture_wal(&mut self) -> Result<()> {
        self.workspace.capture_wal()
    }

    pub(in crate::index::vector) fn from_recorded(
        index: &'index HnswIndex,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            guards: HnswGuardSlot::new(index),
            workspace: HnswMaintenanceWorkspace::from_recorded(payload),
        }
    }

    pub(in crate::index::vector) fn encode_wal_postimage(&self) -> Result<Vec<u8>> {
        self.workspace.encode_wal_postimage()
    }
}

fn changed_deleted_count(before: usize, changes: &[Deleted]) -> Result<usize> {
    changes.iter().try_fold(before, |count, change| {
        let next = match (change.before, change.after) {
            (false, true) => count.checked_add(1),
            (true, false) => count.checked_sub(1),
            _ => Some(count),
        };
        next.ok_or_else(|| invalid("recorded deletion count overflows"))
    })
}

fn node_at(wire: &Wire, id: NodeId) -> Option<&NodeChange> {
    wire.nodes
        .binary_search_by_key(&id, |change| change.id)
        .ok()
        .and_then(|position| wire.nodes.get(position))
}

fn operation_at(wire: &Wire, id: NodeId) -> Option<bool> {
    wire.operations
        .binary_search_by_key(&id, |(id, _)| *id)
        .ok()
        .and_then(|position| wire.operations.get(position))
        .map(|(_, present)| *present)
}

fn validate(index: &HnswIndex, topology: &TopologyBackend, wire: &Wire) -> Result<()> {
    let deleted = index.deleted.read();
    if wire.configuration != Configuration::capture(&index.config)
        || wire.before != state(index, topology, deleted.len())
        || wire
            .operations
            .windows(2)
            .any(|pair| matches!(pair, [left, right] if left.0 >= right.0))
        || wire
            .nodes
            .windows(2)
            .any(|pair| matches!(pair, [left, right] if left.id >= right.id))
        || wire
            .deleted
            .windows(2)
            .any(|pair| matches!(pair, [left, right] if left.id >= right.id))
    {
        return Err(invalid(
            "recorded configuration, baseline or canonical identities differ",
        ));
    }
    let mut added = 0usize;
    let mut unique = std::collections::HashSet::new();
    for change in &wire.nodes {
        if !change.id.is_valid() || change.after.is_empty() || change.after.len() > 64 {
            return Err(invalid("recorded node identity or layers are invalid"));
        }
        match &change.before {
            Some(layers) => {
                if layers.len() != change.after.len()
                    || layers.is_empty()
                    || topology.neighbors_at(change.id, layers.len()).is_some()
                {
                    return Err(invalid("recorded topology preimage layer count differs"));
                }
                for (layer, expected) in layers.iter().enumerate() {
                    if !topology
                        .neighbors_at(change.id, layer)
                        .is_some_and(|actual| actual.eq(expected.iter().copied()))
                    {
                        return Err(invalid("recorded topology preimage differs"));
                    }
                }
            }
            None => {
                if topology.contains(change.id) || operation_at(wire, change.id) != Some(true) {
                    return Err(invalid(
                        "recorded new identity already exists or has no upsert",
                    ));
                }
                added = added
                    .checked_add(1)
                    .ok_or_else(|| invalid("recorded node count overflows"))?;
            }
        }
        for (layer, neighbors) in change.after.iter().enumerate() {
            let max_neighbors = if layer == 0 {
                index.config.m_max
            } else {
                index.config.m
            };
            if neighbors.len() > max_neighbors {
                return Err(invalid("recorded neighbor degree exceeds configuration"));
            }
            unique.clear();
            unique
                .try_reserve(neighbors.len())
                .map_err(|_| AllocError::OutOfMemory)?;
            for neighbor in neighbors {
                let exists = node_at(wire, *neighbor).map_or_else(
                    || topology.neighbors_at(*neighbor, layer).is_some(),
                    |node| node.after.get(layer).is_some(),
                );
                if !neighbor.is_valid()
                    || *neighbor == change.id
                    || !exists
                    || !unique.insert(*neighbor)
                {
                    return Err(invalid(
                        "recorded neighbor is duplicate, missing, self or above its layer",
                    ));
                }
            }
        }
    }
    let mut entry = wire.before.entry_point;
    let mut level = wire.before.max_level;
    let mut draws = 0u64;
    let mut required_deletes = 0usize;
    for (id, present) in &wire.operations {
        if !id.is_valid() {
            return Err(invalid("recorded operation has an invalid identity"));
        }
        let candidate = node_at(wire, *id);
        if *present {
            let candidate =
                candidate.ok_or_else(|| invalid("recorded upsert has no topology postimage"))?;
            let node_level = candidate
                .after
                .len()
                .checked_sub(1)
                .ok_or_else(|| invalid("recorded node has no layer"))?;
            if entry.is_none() || node_level > level {
                entry = Some(*id);
                level = node_level;
            }
            draws = draws
                .checked_add(1)
                .ok_or_else(|| invalid("recorded RNG advance count overflows"))?;
        }
        if *present || topology.contains(*id) {
            required_deletes = required_deletes
                .checked_add(1)
                .ok_or_else(|| invalid("recorded delete count overflows"))?;
            let change = wire
                .deleted
                .binary_search_by_key(id, |change| change.id)
                .ok()
                .and_then(|position| wire.deleted.get(position))
                .ok_or_else(|| invalid("recorded operation has no deletion postimage"))?;
            if change.after == *present {
                return Err(invalid("recorded final presence disagrees with deletion"));
            }
        } else if candidate.is_some() {
            return Err(invalid("recorded absent operation manufactures topology"));
        }
    }
    for change in &wire.deleted {
        if change.before != deleted.contains(&change.id)
            || operation_at(wire, change.id).is_none()
            || !(topology.contains(change.id) || node_at(wire, change.id).is_some())
        {
            return Err(invalid("recorded deletion preimage differs"));
        }
    }
    let count = wire
        .before
        .nodes
        .checked_add(added)
        .ok_or_else(|| invalid("recorded node count overflows"))?;
    if wire.deleted.len() != required_deletes
        || wire.after.nodes != count
        || wire.after.deleted != changed_deleted_count(wire.before.deleted, &wire.deleted)?
        || wire.after.entry_point != entry
        || wire.after.max_level != level
        || wire.after.rng
            != wire
                .before
                .rng
                .wrapping_add(HnswRng::GAMMA.wrapping_mul(draws))
        || (draws == 0 && !wire.nodes.is_empty())
        || index.config.max_elements.is_some_and(|max| count > max)
    {
        return Err(invalid("recorded topology header or RNG postimage differs"));
    }
    Ok(())
}

pub(super) fn prepare_recorded(
    index: &HnswIndex,
    workspace: &mut HnswMaintenanceWorkspace,
    payload: &[u8],
) -> Result<()> {
    let wire = decode(payload)?;
    let topology = index.nodes.read();
    validate(index, &topology, &wire)?;
    reserve_map(&mut workspace.nodes, wire.nodes.len())?;
    reserve_map(&mut workspace.deleted, wire.deleted.len())?;
    let mut captured = Captured {
        configuration: wire.configuration,
        state: wire.before,
        nodes: Vec::new(),
        deleted: wire.deleted,
    };
    captured
        .nodes
        .try_reserve(wire.nodes.len())
        .map_err(|_| AllocError::OutOfMemory)?;
    for change in wire.nodes {
        captured.nodes.push((change.id, change.before));
        workspace.nodes.insert(
            change.id,
            HnswNode {
                neighbors: change.after,
            },
        );
    }
    for change in &captured.deleted {
        workspace.deleted.insert(change.id, change.after);
    }
    workspace.entry_point = wire.after.entry_point;
    workspace.max_level = wire.after.max_level;
    workspace.rng = HnswRng::from_state(wire.after.rng);
    workspace.base_len = captured.state.nodes;
    workspace.added_nodes = wire
        .after
        .nodes
        .checked_sub(captured.state.nodes)
        .ok_or_else(|| invalid("recorded topology cannot remove identities"))?;
    workspace.mmap_additional = match &*topology {
        TopologyBackend::Heap(_) => 0,
        TopologyBackend::Mmap {
            additional_nodes, ..
        } => *additional_nodes,
    };
    workspace.final_presence = wire.operations;
    workspace.wal_baseline = Some(captured);
    Ok(())
}

struct Bounded(Vec<u8>);
impl Write for Bounded {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > LIMIT.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("HNSW WAL exceeds 16 MiB"));
        }
        self.0
            .try_reserve(bytes.len())
            .map_err(std::io::Error::other)?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn decode(payload: &[u8]) -> Result<Wire> {
    preflight(payload)?;
    let body = payload
        .get(MAGIC.len()..)
        .ok_or_else(|| invalid("truncated recorded topology"))?;
    let (wire, consumed) =
        bincode::serde::decode_from_slice(body, bincode::config::standard().with_limit::<LIMIT>())
            .map_err(|error| Error::Serialization(error.to_string()))?;
    if consumed != body.len() {
        return Err(invalid("recorded topology has trailing bytes"));
    }
    Ok(wire)
}

struct Preflight<'a> {
    bytes: &'a [u8],
    position: usize,
    heap: usize,
}
impl Preflight<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| invalid("recorded range overflows"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated recorded topology"))?;
        self.position = end;
        Ok(bytes)
    }
    fn byte(&mut self) -> Result<u8> {
        self.take(1)?
            .first()
            .copied()
            .ok_or_else(|| invalid("missing recorded byte"))
    }
    fn int(&mut self) -> Result<u64> {
        let (value, minimum) = match self.byte()? {
            value @ 0..=250 => return Ok(u64::from(value)),
            251 => (
                u64::from(u16::from_le_bytes(
                    self.take(2)?
                        .try_into()
                        .map_err(|_| invalid("truncated integer"))?,
                )),
                251,
            ),
            252 => (
                u64::from(u32::from_le_bytes(
                    self.take(4)?
                        .try_into()
                        .map_err(|_| invalid("truncated integer"))?,
                )),
                65_536,
            ),
            253 => (
                u64::from_le_bytes(
                    self.take(8)?
                        .try_into()
                        .map_err(|_| invalid("truncated integer"))?,
                ),
                4_294_967_296,
            ),
            _ => return Err(invalid("unsupported recorded integer")),
        };
        if value < minimum {
            return Err(invalid("noncanonical recorded integer"));
        }
        Ok(value)
    }
    fn boolean(&mut self) -> Result<bool> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid recorded boolean")),
        }
    }
    fn option(&mut self, value: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        if self.boolean()? {
            value(self)?;
        }
        Ok(())
    }
    fn sequence<T>(&mut self) -> Result<usize> {
        let count =
            usize::try_from(self.int()?).map_err(|_| invalid("recorded count exceeds host"))?;
        self.heap = count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| self.heap.checked_add(bytes))
            .ok_or_else(|| invalid("recorded allocation overflows"))?;
        if self.heap > LIMIT
            || count
                > self
                    .bytes
                    .len()
                    .checked_sub(self.position)
                    .ok_or_else(|| invalid("invalid recorded cursor"))?
        {
            return Err(invalid(
                "recorded allocation exceeds 16 MiB or remaining bytes",
            ));
        }
        Ok(count)
    }
    fn layers(&mut self) -> Result<()> {
        let layers = self.sequence::<Vec<NodeId>>()?;
        if !(1..=64).contains(&layers) {
            return Err(invalid("recorded layer count exceeds current bounds"));
        }
        for _ in 0..layers {
            for _ in 0..self.sequence::<NodeId>()? {
                self.int()?;
            }
        }
        Ok(())
    }
    fn state(&mut self) -> Result<()> {
        self.option(|wire| {
            wire.int()?;
            Ok(())
        })?;
        for _ in 0..4 {
            self.int()?;
        }
        Ok(())
    }
}

fn preflight(payload: &[u8]) -> Result<usize> {
    if payload.len() > LIMIT || !payload.starts_with(&MAGIC) {
        return Err(invalid("unsupported or oversized recorded topology"));
    }
    let bytes = payload
        .get(MAGIC.len()..)
        .ok_or_else(|| invalid("missing recorded topology"))?;
    let mut wire = Preflight {
        bytes,
        position: 0,
        heap: 0,
    };
    wire.int()?;
    if wire.byte()? > 3 {
        return Err(invalid("unknown recorded distance metric"));
    }
    for _ in 0..6 {
        wire.int()?;
    }
    wire.option(|wire| {
        wire.int()?;
        Ok(())
    })?;
    wire.state()?;
    wire.state()?;
    for _ in 0..wire.sequence::<(NodeId, bool)>()? {
        wire.int()?;
        wire.boolean()?;
    }
    for _ in 0..wire.sequence::<NodeChange>()? {
        wire.int()?;
        wire.option(Preflight::layers)?;
        wire.layers()?;
    }
    for _ in 0..wire.sequence::<Deleted>()? {
        wire.int()?;
        wire.boolean()?;
        wire.boolean()?;
    }
    if wire.position != bytes.len() {
        return Err(invalid("recorded topology has trailing bytes"));
    }
    Ok(wire.heap)
}

#[cfg(test)]
mod tests;
