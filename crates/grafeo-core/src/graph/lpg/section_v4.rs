//! Canonical, exact LPG section wire format used by `.grafeo` section v4.
//!
//! The outer container protects section bytes, but this envelope is also
//! independently self-identifying and checksummed so the core decoder never
//! has to trust a directory's version label. The bincode payload contains only
//! ordered vectors and primitive fields. Property values use a separate,
//! exhaustive codec: that makes CRDT maps canonical and bounds recursive
//! decoding before constructing [`Value`] trees.

use arcstr::ArcStr;
use grafeo_common::types::{
    EdgeId, EpochId, GraphIncarnationId, GraphPath, NodeId, PropertyKey, Value,
};
use grafeo_common::utils::error::{Error, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::graph::lpg::LpgStore;
use crate::graph::lpg::exact_history::{
    epoch_is_inside_lifetime, lifetime_is_covered_by, validate_lifetimes,
};
use crate::graph::lpg::value_codec::{decode_value_exact, encode_value};

pub(super) const MAGIC: [u8; 4] = *b"LPG4";

const WIRE_VERSION: u8 = 3;
const HEADER_LEN: usize = 24;

// The payload retains the existing u32-size ceiling. Bincode's limit accounts for container
// allocations before allocating them, rejecting tiny inputs that advertise
// hostile collection lengths as well as genuinely oversized sections.
const MAX_PAYLOAD_BYTES: usize = u32::MAX as usize;

use grafeo_common::types::graph_path_bytes as path_bytes;

#[derive(Debug, Serialize, Deserialize)]
struct WireSnapshotV4 {
    next_graph_incarnation_id: u64,
    graphs: Vec<WirePathGraphV4>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WirePathGraphV4 {
    #[serde(with = "path_bytes")]
    path: GraphPath,
    graph: WireGraphV4,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireGraphV4 {
    incarnation: GraphIncarnationId,
    committed_epoch: u64,
    retained_history_floor: u64,
    next_node_id: u64,
    next_edge_id: u64,
    nodes: Vec<WireNodeV4>,
    edges: Vec<WireEdgeV4>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireNodeV4 {
    id: u64,
    lifetimes: Vec<WireLifetimeV4>,
    label_history: Vec<WireLabelVersionV4>,
    properties: Vec<WirePropertyV4>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireEdgeV4 {
    id: u64,
    src: u64,
    dst: u64,
    edge_type: String,
    lifetimes: Vec<WireLifetimeV4>,
    properties: Vec<WirePropertyV4>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct WireLifetimeV4 {
    created: u64,
    deleted: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireLabelVersionV4 {
    epoch: u64,
    labels: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WirePropertyV4 {
    key: String,
    history: Vec<WirePropertyVersionV4>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WirePropertyVersionV4 {
    epoch: u64,
    value: Vec<u8>,
}

struct SnapshotV4 {
    next_graph_incarnation_id: u64,
    graphs: Vec<PathGraphV4>,
}

struct PathGraphV4 {
    path: GraphPath,
    graph: GraphV4,
}

struct GraphV4 {
    incarnation: GraphIncarnationId,
    committed_epoch: EpochId,
    retained_history_floor: EpochId,
    next_node_id: u64,
    next_edge_id: u64,
    nodes: Vec<NodeV4>,
    edges: Vec<EdgeV4>,
}

struct NodeV4 {
    id: NodeId,
    lifetimes: Vec<(EpochId, Option<EpochId>)>,
    label_history: Vec<(EpochId, Vec<ArcStr>)>,
    properties: Vec<PropertyV4>,
}

struct EdgeV4 {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: ArcStr,
    lifetimes: Vec<(EpochId, Option<EpochId>)>,
    properties: Vec<PropertyV4>,
}

struct PropertyV4 {
    key: String,
    history: Vec<(EpochId, Value)>,
}

pub(super) fn serialize(store: &LpgStore) -> Result<Vec<u8>> {
    let snapshot = SnapshotV4::collect(store)?;
    snapshot.validate()?;
    let wire = WireSnapshotV4::from_snapshot(&snapshot)?;
    encode_wire(&wire)
}

/// Only the Section facade's private pinned callback supplies this graph set.
pub(super) fn serialize_pinned_graphs(graphs: &[(GraphPath, Arc<LpgStore>)]) -> Result<Vec<u8>> {
    let mut captured = Vec::new();
    captured
        .try_reserve(graphs.len())
        .map_err(|error| Error::Io(std::io::Error::other(error)))?;
    for (path, store) in graphs {
        captured.push(PathGraphV4 {
            path: path.clone(),
            graph: GraphV4::collect(store)?,
        });
    }
    let snapshot = SnapshotV4 {
        next_graph_incarnation_id: graphs
            .first()
            .ok_or_else(|| serialization("missing root"))?
            .1
            .next_graph_incarnation_id(),
        graphs: captured,
    };
    snapshot.validate()?;
    encode_wire(&WireSnapshotV4::from_snapshot(&snapshot)?)
}

/// The replacement workspace supplies its private root and exactly pinned
/// descendant owners; no topology or representation gate is reacquired here.
pub(super) fn serialize_replacement_image(
    root: &LpgStore,
    graphs: &[(GraphPath, Arc<LpgStore>)],
) -> Result<Vec<u8>> {
    let mut captured = Vec::new();
    captured
        .try_reserve(graphs.len())
        .map_err(|error| Error::Io(std::io::Error::other(error)))?;
    for (path, store) in graphs {
        captured.push(PathGraphV4 {
            path: path.clone(),
            graph: GraphV4::collect(if path.components().is_empty() {
                root
            } else {
                store
            })?,
        });
    }
    let snapshot = SnapshotV4 {
        next_graph_incarnation_id: root.next_graph_incarnation_id(),
        graphs: captured,
    };
    snapshot.validate()?;
    encode_wire(&WireSnapshotV4::from_snapshot(&snapshot)?)
}

fn encode_wire(wire: &WireSnapshotV4) -> Result<Vec<u8>> {
    let payload = bincode::serde::encode_to_vec(
        wire,
        bincode::config::standard().with_limit::<MAX_PAYLOAD_BYTES>(),
    )
    .map_err(|error| serialization(format!("encode LPG v4 payload: {error}")))?;
    if u32::try_from(payload.len()).is_err() {
        return Err(serialization(format!(
            "LPG v4 payload is {} bytes; maximum is {MAX_PAYLOAD_BYTES}",
            payload.len()
        )));
    }

    let payload_len = u64::try_from(payload.len())
        .map_err(|_| serialization("LPG v4 payload length exceeds u64"))?;
    let checksum = crc32fast::hash(&payload);
    let encoded_len = HEADER_LEN
        .checked_add(payload.len())
        .ok_or_else(|| serialization("LPG v4 envelope length overflow"))?;
    let mut encoded = Vec::with_capacity(encoded_len);
    encoded.extend_from_slice(&MAGIC);
    encoded.push(WIRE_VERSION);
    encoded.push(0); // flags
    encoded.extend_from_slice(&0_u16.to_le_bytes()); // reserved
    encoded.extend_from_slice(&payload_len.to_le_bytes());
    encoded.extend_from_slice(&checksum.to_le_bytes());
    encoded.extend_from_slice(&0_u32.to_le_bytes()); // reserved
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

pub(super) fn deserialize_into(store: &LpgStore, data: &[u8]) -> Result<()> {
    let snapshot = decode(data)?;
    let mut candidate = store.new_restore_candidate()?;
    for entry in &snapshot.graphs {
        if entry.path.components().is_empty() {
            restore_graph(&candidate, &entry.graph)?;
            continue;
        }
        let (name, parents) = entry
            .path
            .components()
            .split_last()
            .ok_or_else(|| serialization("non-root LPG graph has no name"))?;
        let mut parent: Option<Arc<LpgStore>> = None;
        for component in parents {
            let target = match parent.as_deref() {
                Some(parent) => parent,
                None => &candidate,
            };
            parent = Some(target.graph(component).ok_or_else(|| {
                serialization(format!("missing parent while restoring {:?}", entry.path))
            })?);
        }
        let target = match parent.as_deref() {
            Some(parent) => parent,
            None => &candidate,
        };
        if !target.create_graph(name)? {
            return Err(serialization(format!(
                "duplicate graph while restoring {:?}",
                entry.path
            )));
        }
        let child = target.graph(name).ok_or_else(|| {
            serialization(format!("missing child while restoring {:?}", entry.path))
        })?;
        restore_graph(&child, &entry.graph)?;
    }
    candidate
        .restore_graph_incarnations(&snapshot.incarnations(), snapshot.next_graph_incarnation_id)?;
    store.install_pristine_image(candidate)
}

fn decode(data: &[u8]) -> Result<SnapshotV4> {
    if !data.starts_with(&MAGIC) {
        return Err(serialization(
            "unsupported LPG section generation; expected LPG4",
        ));
    }
    if data.len() < HEADER_LEN {
        return Err(serialization("truncated LPG v4 envelope header"));
    }
    if data[4] != WIRE_VERSION {
        return Err(serialization(format!(
            "unsupported LPG v4 wire version {}",
            data[4]
        )));
    }
    if data[5] != 0 || data[6..8] != [0, 0] || data[20..24] != [0, 0, 0, 0] {
        return Err(serialization(
            "unsupported LPG v4 envelope flags or reserved fields",
        ));
    }

    let declared_len = u64::from_le_bytes(
        data[8..16]
            .try_into()
            .map_err(|_| serialization("invalid LPG v4 payload length"))?,
    );
    let payload_len = usize::try_from(declared_len)
        .map_err(|_| serialization("LPG v4 payload length exceeds this platform"))?;
    if u32::try_from(payload_len).is_err() {
        return Err(serialization(format!(
            "LPG v4 payload is {payload_len} bytes; maximum is {MAX_PAYLOAD_BYTES}"
        )));
    }
    let expected_total = HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| serialization("LPG v4 envelope length overflow"))?;
    if data.len() < expected_total {
        return Err(serialization(format!(
            "truncated LPG v4 payload: declared {payload_len} bytes, have {}",
            data.len() - HEADER_LEN
        )));
    }
    if data.len() > expected_total {
        return Err(serialization(format!(
            "trailing LPG v4 bytes: {}",
            data.len() - expected_total
        )));
    }

    let expected_checksum = u32::from_le_bytes(
        data[16..20]
            .try_into()
            .map_err(|_| serialization("invalid LPG v4 checksum"))?,
    );
    let payload = &data[HEADER_LEN..];
    let actual_checksum = crc32fast::hash(payload);
    if actual_checksum != expected_checksum {
        return Err(serialization(format!(
            "LPG v4 checksum mismatch: expected {expected_checksum:#010x}, got {actual_checksum:#010x}"
        )));
    }

    let wire = decode_payload::<WireSnapshotV4>(payload)?;
    let snapshot = wire.into_snapshot()?;
    snapshot.validate()?;
    Ok(snapshot)
}

impl SnapshotV4 {
    fn incarnations(&self) -> Vec<(GraphPath, GraphIncarnationId)> {
        self.graphs
            .iter()
            .map(|entry| (entry.path.clone(), entry.graph.incarnation))
            .collect()
    }
    fn collect(store: &LpgStore) -> Result<Self> {
        store.with_pinned_recursive_capture(|root, descendants| {
            let mut graphs = Vec::new();
            graphs
                .try_reserve(
                    descendants
                        .len()
                        .checked_add(1)
                        .ok_or_else(|| serialization("LPG graph count overflow"))?,
                )
                .map_err(|error| Error::Io(std::io::Error::other(error)))?;
            graphs.push(PathGraphV4 {
                path: GraphPath::root(),
                graph: GraphV4::collect(root)?,
            });
            for (path, graph) in descendants {
                graphs.push(PathGraphV4 {
                    path: path.clone(),
                    graph: GraphV4::collect(graph)?,
                });
            }
            Ok(Self {
                next_graph_incarnation_id: root.next_graph_incarnation_id(),
                graphs,
            })
        })
    }

    fn validate(&self) -> Result<()> {
        LpgStore::validate_graph_incarnations(
            &self.incarnations(),
            self.next_graph_incarnation_id,
        )?;
        if self
            .graphs
            .first()
            .is_none_or(|entry| !entry.path.components().is_empty())
        {
            return Err(serialization(
                "LPG v4 requires exactly one root graph first",
            ));
        }
        for (index, entry) in self.graphs.iter().enumerate() {
            if index > 0 && self.graphs[index - 1].path >= entry.path {
                return Err(serialization(
                    "LPG v4 graph paths must be strictly sorted and unique",
                ));
            }
            if let Some(parent) = entry
                .path
                .parent()
                .map_err(|error| serialization(format!("invalid LPG parent path: {error}")))?
                && self.graphs[..index]
                    .binary_search_by(|prior| prior.path.cmp(&parent))
                    .is_err()
            {
                return Err(serialization(format!(
                    "LPG v4 graph {:?} has no parent",
                    entry.path
                )));
            }
            entry.graph.validate(&format!("graph {:?}", entry.path))?;
        }
        Ok(())
    }
}

impl GraphV4 {
    fn collect(store: &LpgStore) -> Result<Self> {
        let mut node_ids = store.all_node_ids();
        node_ids.sort_unstable();
        node_ids.dedup();
        let mut nodes = Vec::with_capacity(node_ids.len());
        for id in node_ids {
            let mut history = store.get_node_history(id);
            if history.is_empty() {
                return Err(serialization(format!(
                    "node {id} has an identity but no structural history"
                )));
            }
            history.reverse();
            let lifetimes = history
                .into_iter()
                .map(|(created, deleted, _)| (created, deleted))
                .collect();
            let label_history = store.node_label_history(id);
            let properties = collect_properties(store.node_property_history(id));
            nodes.push(NodeV4 {
                id,
                lifetimes,
                label_history,
                properties,
            });
        }

        let mut edge_ids = store.all_known_edge_ids();
        edge_ids.sort_unstable();
        edge_ids.dedup();
        let mut edges = Vec::with_capacity(edge_ids.len());
        for id in edge_ids {
            let mut history = store.get_edge_history(id);
            let (_, _, identity) = history.first().ok_or_else(|| {
                serialization(format!(
                    "edge {id} has an identity but no structural history"
                ))
            })?;
            let src = identity.src;
            let dst = identity.dst;
            let edge_type = identity.edge_type.clone();
            if history.iter().any(|(_, _, edge)| {
                edge.src != src || edge.dst != dst || edge.edge_type != edge_type
            }) {
                return Err(serialization(format!(
                    "edge {id} changes immutable endpoints or type across lifetimes"
                )));
            }
            history.reverse();
            let lifetimes = history
                .into_iter()
                .map(|(created, deleted, _)| (created, deleted))
                .collect();
            let properties = collect_properties(store.edge_property_history(id));
            edges.push(EdgeV4 {
                id,
                src,
                dst,
                edge_type,
                lifetimes,
                properties,
            });
        }

        Ok(Self {
            incarnation: store.graph_incarnation_id(),
            committed_epoch: store.current_epoch(),
            retained_history_floor: store.retained_history_floor(),
            next_node_id: store.next_node_id(),
            next_edge_id: store.next_edge_id(),
            nodes,
            edges,
        })
    }

    fn validate(&self, context: &str) -> Result<()> {
        if self.committed_epoch == EpochId::PENDING {
            return Err(serialization(format!(
                "{context} committed epoch cannot be PENDING"
            )));
        }
        if self.retained_history_floor == EpochId::PENDING
            || self.retained_history_floor > self.committed_epoch
        {
            return Err(serialization(format!(
                "{context} retained history floor is outside committed epoch"
            )));
        }
        validate_strictly_sorted_by(&self.nodes, |node| node.id, "node identities")?;
        validate_strictly_sorted_by(&self.edges, |edge| edge.id, "edge identities")?;

        if let Some(maximum) = self.nodes.last().map(|node| node.id.as_u64())
            && self.next_node_id <= maximum
        {
            return Err(serialization(format!(
                "{context} next node id {} must exceed maximum identity {}",
                self.next_node_id, maximum
            )));
        }
        if let Some(maximum) = self.edges.last().map(|edge| edge.id.as_u64())
            && self.next_edge_id <= maximum
        {
            return Err(serialization(format!(
                "{context} next edge id {} must exceed maximum identity {}",
                self.next_edge_id, maximum
            )));
        }

        for node in &self.nodes {
            node.validate(self.committed_epoch, context)?;
        }
        for edge in &self.edges {
            edge.validate(self.committed_epoch, context)?;
            for (kind, id) in [("source", edge.src), ("destination", edge.dst)] {
                let endpoint = self
                    .nodes
                    .binary_search_by_key(&id, |node| node.id)
                    .ok()
                    .and_then(|index| self.nodes.get(index))
                    .ok_or_else(|| {
                        serialization(format!(
                            "{context} edge {} references an unknown endpoint {id} ({kind})",
                            edge.id
                        ))
                    })?;
                for &life in &edge.lifetimes {
                    if !lifetime_is_covered_by(life, &endpoint.lifetimes) {
                        return Err(serialization(format!(
                            "{context} edge {} outlives {kind} node {id}",
                            edge.id
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

impl NodeV4 {
    fn validate(&self, committed_epoch: EpochId, context: &str) -> Result<()> {
        if !self.id.is_valid() {
            return Err(serialization(format!("{context} contains invalid NodeId")));
        }
        validate_lifetimes("node", &self.lifetimes, committed_epoch)
            .map_err(|error| serialization(format!("node {}: {error}", self.id)))?;
        if self.label_history.is_empty() {
            return Err(serialization(format!(
                "{context} node {} has no complete label history",
                self.id
            )));
        }
        let mut prior_epoch = None;
        for (index, (epoch, labels)) in self.label_history.iter().enumerate() {
            validate_committed_epoch(*epoch, committed_epoch, "node label", self.id.as_u64())?;
            if prior_epoch.is_some_and(|prior| *epoch < prior) {
                return Err(serialization(format!(
                    "{context} node {} label history is not epoch-ascending at entry {index}",
                    self.id
                )));
            }
            prior_epoch = Some(*epoch);
            if labels.len() > usize::from(u16::MAX) {
                return Err(serialization(format!(
                    "{context} node {} label version {index} exceeds u16::MAX labels",
                    self.id
                )));
            }
            validate_strictly_sorted_by(labels, Clone::clone, "node labels")?;
            if !epoch_is_inside_lifetime(*epoch, &self.lifetimes, true) {
                return Err(serialization(format!(
                    "{context} node {} label epoch {} is outside its structural history",
                    self.id,
                    epoch.as_u64()
                )));
            }
        }
        for (index, (created, _)) in self.lifetimes.iter().enumerate() {
            if !self.label_history.iter().any(|(epoch, _)| epoch == created) {
                return Err(serialization(format!(
                    "{context} node {} lifetime {index} has no complete label set at creation",
                    self.id
                )));
            }
        }
        validate_properties(
            &self.properties,
            &self.lifetimes,
            committed_epoch,
            "node",
            self.id.as_u64(),
        )
    }
}

impl EdgeV4 {
    fn validate(&self, committed_epoch: EpochId, context: &str) -> Result<()> {
        if !self.id.is_valid() || !self.src.is_valid() || !self.dst.is_valid() {
            return Err(serialization(format!(
                "{context} contains invalid edge IDs"
            )));
        }
        validate_lifetimes("edge", &self.lifetimes, committed_epoch)
            .map_err(|error| serialization(format!("edge {}: {error}", self.id)))?;
        validate_properties(
            &self.properties,
            &self.lifetimes,
            committed_epoch,
            "edge",
            self.id.as_u64(),
        )
    }
}

fn collect_properties(histories: Vec<(PropertyKey, Vec<(EpochId, Value)>)>) -> Vec<PropertyV4> {
    let mut properties: Vec<_> = histories
        .into_iter()
        .map(|(key, history)| PropertyV4 {
            key: key.to_string(),
            history,
        })
        .collect();
    properties.sort_by(|left, right| left.key.cmp(&right.key));
    properties
}

fn validate_properties(
    properties: &[PropertyV4],
    lifetimes: &[(EpochId, Option<EpochId>)],
    committed_epoch: EpochId,
    entity: &str,
    id: u64,
) -> Result<()> {
    validate_strictly_sorted_by(properties, |property| property.key.clone(), "property keys")?;
    for property in properties {
        if property.history.is_empty() {
            return Err(serialization(format!(
                "{entity} {id} property {:?} has empty history",
                property.key
            )));
        }
        let mut prior_epoch = None;
        for (epoch, _) in &property.history {
            validate_committed_epoch(*epoch, committed_epoch, "property", id)?;
            if prior_epoch.is_some_and(|prior| *epoch < prior) {
                return Err(serialization(format!(
                    "{entity} {id} property {:?} history is not epoch-ascending",
                    property.key
                )));
            }
            prior_epoch = Some(*epoch);
            if !epoch_is_inside_lifetime(*epoch, lifetimes, true) {
                return Err(serialization(format!(
                    "{entity} {id} property {:?} epoch {} is outside its structural history",
                    property.key,
                    epoch.as_u64()
                )));
            }
        }
    }
    Ok(())
}

fn validate_committed_epoch(
    epoch: EpochId,
    committed_epoch: EpochId,
    field: &str,
    id: u64,
) -> Result<()> {
    if epoch == EpochId::PENDING {
        return Err(serialization(format!(
            "{field} history for identity {id} contains PENDING"
        )));
    }
    if epoch > committed_epoch {
        return Err(serialization(format!(
            "{field} history for identity {id} reaches epoch {}, beyond committed epoch {}",
            epoch.as_u64(),
            committed_epoch.as_u64()
        )));
    }
    Ok(())
}

fn validate_strictly_sorted_by<T, K: Ord>(
    values: &[T],
    key: impl Fn(&T) -> K,
    description: &str,
) -> Result<()> {
    if values.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
        return Err(serialization(format!(
            "LPG v4 {description} must be strictly sorted and unique"
        )));
    }
    Ok(())
}

impl WireSnapshotV4 {
    fn from_snapshot(snapshot: &SnapshotV4) -> Result<Self> {
        Ok(Self {
            next_graph_incarnation_id: snapshot.next_graph_incarnation_id,
            graphs: snapshot
                .graphs
                .iter()
                .map(|entry| {
                    Ok(WirePathGraphV4 {
                        path: entry.path.clone(),
                        graph: WireGraphV4::from_graph(&entry.graph)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn into_snapshot(self) -> Result<SnapshotV4> {
        Ok(SnapshotV4 {
            next_graph_incarnation_id: self.next_graph_incarnation_id,
            graphs: self
                .graphs
                .into_iter()
                .map(|entry| {
                    Ok(PathGraphV4 {
                        path: entry.path,
                        graph: entry.graph.into_graph()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl WireGraphV4 {
    fn from_graph(graph: &GraphV4) -> Result<Self> {
        Ok(Self {
            incarnation: graph.incarnation,
            committed_epoch: graph.committed_epoch.as_u64(),
            retained_history_floor: graph.retained_history_floor.as_u64(),
            next_node_id: graph.next_node_id,
            next_edge_id: graph.next_edge_id,
            nodes: graph
                .nodes
                .iter()
                .map(WireNodeV4::from_node)
                .collect::<Result<Vec<_>>>()?,
            edges: graph
                .edges
                .iter()
                .map(WireEdgeV4::from_edge)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn into_graph(self) -> Result<GraphV4> {
        Ok(GraphV4 {
            incarnation: self.incarnation,
            committed_epoch: EpochId::new(self.committed_epoch),
            retained_history_floor: EpochId::new(self.retained_history_floor),
            next_node_id: self.next_node_id,
            next_edge_id: self.next_edge_id,
            nodes: self
                .nodes
                .into_iter()
                .map(WireNodeV4::into_node)
                .collect::<Result<Vec<_>>>()?,
            edges: self
                .edges
                .into_iter()
                .map(WireEdgeV4::into_edge)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl WireNodeV4 {
    fn from_node(node: &NodeV4) -> Result<Self> {
        Ok(Self {
            id: node.id.as_u64(),
            lifetimes: node
                .lifetimes
                .iter()
                .map(|(created, deleted)| WireLifetimeV4 {
                    created: created.as_u64(),
                    deleted: deleted.map(|epoch| epoch.as_u64()),
                })
                .collect(),
            label_history: node
                .label_history
                .iter()
                .map(|(epoch, labels)| WireLabelVersionV4 {
                    epoch: epoch.as_u64(),
                    labels: labels.iter().map(ToString::to_string).collect(),
                })
                .collect(),
            properties: node
                .properties
                .iter()
                .map(WirePropertyV4::from_property)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn into_node(self) -> Result<NodeV4> {
        Ok(NodeV4 {
            id: NodeId::new(self.id),
            lifetimes: self
                .lifetimes
                .into_iter()
                .map(|lifetime| {
                    (
                        EpochId::new(lifetime.created),
                        lifetime.deleted.map(EpochId::new),
                    )
                })
                .collect(),
            label_history: self
                .label_history
                .into_iter()
                .map(|version| {
                    (
                        EpochId::new(version.epoch),
                        version.labels.into_iter().map(ArcStr::from).collect(),
                    )
                })
                .collect(),
            properties: self
                .properties
                .into_iter()
                .map(WirePropertyV4::into_property)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl WireEdgeV4 {
    fn from_edge(edge: &EdgeV4) -> Result<Self> {
        Ok(Self {
            id: edge.id.as_u64(),
            src: edge.src.as_u64(),
            dst: edge.dst.as_u64(),
            edge_type: edge.edge_type.to_string(),
            lifetimes: edge
                .lifetimes
                .iter()
                .map(|(created, deleted)| WireLifetimeV4 {
                    created: created.as_u64(),
                    deleted: deleted.map(|epoch| epoch.as_u64()),
                })
                .collect(),
            properties: edge
                .properties
                .iter()
                .map(WirePropertyV4::from_property)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn into_edge(self) -> Result<EdgeV4> {
        Ok(EdgeV4 {
            id: EdgeId::new(self.id),
            src: NodeId::new(self.src),
            dst: NodeId::new(self.dst),
            edge_type: ArcStr::from(self.edge_type),
            lifetimes: self
                .lifetimes
                .into_iter()
                .map(|lifetime| {
                    (
                        EpochId::new(lifetime.created),
                        lifetime.deleted.map(EpochId::new),
                    )
                })
                .collect(),
            properties: self
                .properties
                .into_iter()
                .map(WirePropertyV4::into_property)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl WirePropertyV4 {
    fn from_property(property: &PropertyV4) -> Result<Self> {
        Ok(Self {
            key: property.key.clone(),
            history: property
                .history
                .iter()
                .map(|(epoch, value)| {
                    Ok(WirePropertyVersionV4 {
                        epoch: epoch.as_u64(),
                        value: encode_value(value)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn into_property(self) -> Result<PropertyV4> {
        Ok(PropertyV4 {
            key: self.key,
            history: self
                .history
                .into_iter()
                .map(|version| {
                    Ok((
                        EpochId::new(version.epoch),
                        decode_value_exact(&version.value)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

fn restore_graph(store: &LpgStore, graph: &GraphV4) -> Result<()> {
    for node in &graph.nodes {
        store
            .restore_node_history_exact(node.id, &node.lifetimes, &node.label_history)
            .map_err(|error| serialization(format!("restore node {}: {error}", node.id)))?;
        restore_node_properties(store, node);
    }
    for edge in &graph.edges {
        store
            .restore_edge_history_exact(
                edge.id,
                edge.src,
                edge.dst,
                &edge.edge_type,
                &edge.lifetimes,
            )
            .map_err(|error| serialization(format!("restore edge {}: {error}", edge.id)))?;
        restore_edge_properties(store, edge);
    }
    store
        .restore_allocator_high_water_exact(graph.next_node_id, graph.next_edge_id)
        .map_err(|error| serialization(format!("restore LPG allocator high-water: {error}")))?;
    store.sync_epoch(graph.committed_epoch);
    store.advance_retained_history_floor(graph.retained_history_floor);
    Ok(())
}

fn decode_payload<T: DeserializeOwned>(payload: &[u8]) -> Result<T> {
    let (value, consumed): (T, usize) = bincode::serde::decode_from_slice(
        payload,
        bincode::config::standard().with_limit::<MAX_PAYLOAD_BYTES>(),
    )
    .map_err(|error| serialization(format!("decode LPG v4 payload: {error}")))?;
    if consumed != payload.len() {
        return Err(serialization(format!(
            "trailing LPG v4 payload bytes: {}",
            payload.len() - consumed
        )));
    }
    Ok(value)
}

fn restore_node_properties(store: &LpgStore, node: &NodeV4) {
    for property in &node.properties {
        for (epoch, value) in &property.history {
            store.set_node_property_at_epoch(node.id, &property.key, value.clone(), *epoch);
        }
    }
}

fn restore_edge_properties(store: &LpgStore, edge: &EdgeV4) {
    for property in &edge.properties {
        for (epoch, value) in &property.history {
            store.set_edge_property_at_epoch(edge.id, &property.key, value.clone(), *epoch);
        }
    }
}

fn serialization(message: impl Into<String>) -> Error {
    Error::Serialization(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::storage::section::Section;
    use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};
    use std::collections::{BTreeMap, HashMap};

    fn wire_root(graph: WireGraphV4) -> WireSnapshotV4 {
        WireSnapshotV4 {
            next_graph_incarnation_id: 10,
            graphs: vec![WirePathGraphV4 {
                path: GraphPath::root(),
                graph,
            }],
        }
    }

    fn empty_graph(epoch: u64) -> WireGraphV4 {
        WireGraphV4 {
            incarnation: GraphIncarnationId::DEFAULT_GRAPH,
            committed_epoch: epoch,
            retained_history_floor: epoch,
            next_node_id: 0,
            next_edge_id: 0,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    #[test]
    fn predecessor_wire_rejected_before_restore()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::graph::lpg::section::LpgStoreSection;
        use grafeo_common::storage::section::Section;

        // Exact predecessor root: epoch7, zero allocator counters and rows.
        let payload = [1_u8, 4, 0, 0, 0, 0, 7, 0, 0, 0, 0];
        for version in 0..WIRE_VERSION {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&MAGIC);
            bytes.extend_from_slice(&[version, 0, 0, 0]);
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
            bytes.extend_from_slice(&[0; 4]);
            bytes.extend_from_slice(&payload);
            for populated in [false, true] {
                let store = Arc::new(LpgStore::new()?);
                if populated {
                    store.create_node(&["Preserved"]);
                    store.graph_or_create("nested")?.create_node(&["Child"]);
                }
                let mut section = LpgStoreSection::new(store);
                let before = section.serialize()?;
                let error = section
                    .deserialize(&bytes)
                    .expect_err("predecessor LPG wire must reject before restore");
                assert!(
                    error
                        .to_string()
                        .contains("unsupported LPG v4 wire version")
                );
                assert_eq!(section.serialize()?, before);
            }
            bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(
                decode(&bytes)
                    .err()
                    .ok_or("hostile predecessor admitted")?
                    .to_string()
                    .contains("unsupported LPG v4 wire version")
            );
        }
        Ok(())
    }

    fn wire_node(id: u64, labels: Vec<&str>) -> WireNodeV4 {
        WireNodeV4 {
            id,
            lifetimes: vec![WireLifetimeV4 {
                created: 1,
                deleted: None,
            }],
            label_history: vec![WireLabelVersionV4 {
                epoch: 1,
                labels: labels.into_iter().map(str::to_owned).collect(),
            }],
            properties: Vec::new(),
        }
    }

    #[test]
    fn current_envelope_empty_image_and_reserved_fields_are_exact()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let wire = wire_root(empty_graph(0));
        let bytes = encode_wire(&wire)?;
        assert_eq!(&bytes[..8], b"LPG4\x03\0\0\0");
        assert_eq!(&bytes[8..16], &14_u64.to_le_bytes());
        assert_eq!(&bytes[20..24], &[0; 4]);
        assert_eq!(
            &bytes[HEADER_LEN..],
            &[10, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert!(decode(&bytes).is_ok());
        for offset in [4, 5, 6, 7, 20, 21, 22, 23] {
            let mut invalid = bytes.clone();
            invalid[offset] = if offset == 4 { 4 } else { 2 };
            assert!(
                decode(&invalid).is_err(),
                "accepted reserved/version byte {offset}"
            );
        }
        let mut oversized = bytes.clone();
        oversized[8..16].copy_from_slice(&(u64::from(u32::MAX) + 1).to_le_bytes());
        assert!(decode(&oversized).is_err());
        Ok(())
    }

    #[test]
    fn graph_path_wire_checks_lengths_depth_utf8_and_exact_consumption()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        #[derive(Debug, Serialize, Deserialize)]
        struct PathOnly(#[serde(with = "path_bytes")] GraphPath);

        let maximum = GraphPath::from_components(&[""; 256])?;
        let bytes =
            bincode::serde::encode_to_vec(PathOnly(maximum.clone()), bincode::config::standard())?;
        let (decoded, consumed): (PathOnly, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
        assert_eq!(decoded.0, maximum);
        assert_eq!(consumed, bytes.len());
        let malformed = [
            257_u32.to_le_bytes().to_vec(),
            [1_u32.to_le_bytes().as_slice(), &65_537_u32.to_le_bytes()].concat(),
            vec![1, 0, 0, 0, 1, 0, 0, 0, 0xff],
            vec![1, 0, 0, 0, 2, 0, 0, 0, b'a'],
            vec![0, 0, 0, 0, 1],
        ];
        for path in malformed {
            let bytes = bincode::serde::encode_to_vec(path, bincode::config::standard())?;
            assert!(
                bincode::serde::decode_from_slice::<PathOnly, _>(
                    &bytes,
                    bincode::config::standard()
                )
                .is_err()
            );
        }
        // This advertises only an excessive outer byte-vector length: no body
        // exists. The path visitor must reject the count before reserving it.
        let huge = bincode::serde::encode_to_vec(u64::from(u32::MAX), bincode::config::standard())?;
        let error =
            bincode::serde::decode_from_slice::<PathOnly, _>(&huge, bincode::config::standard())
                .expect_err("oversized path count must fail before reading its body");
        assert!(
            error
                .to_string()
                .contains("graph path byte length exceeds its bounds"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn recursive_empty_graphs_preserve_literal_paths_and_per_graph_cuts()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Arc::new(LpgStore::new()?);
        for name in ["", "a/b", "a"] {
            assert!(source.create_graph(name)?);
        }
        let parent = source.graph("a").ok_or("missing parent")?;
        assert!(parent.create_graph("b")?);
        assert!(parent.create_graph("")?);
        let nested = parent.graph("b").ok_or("missing nested graph")?;
        let literal = source.graph("a/b").ok_or("missing literal graph")?;
        let nested_node = nested.create_node(&["Nested"]);
        let literal_node = literal.create_node(&["Literal"]);
        assert_eq!(nested_node, literal_node);
        nested.set_node_property(nested_node, "value", Value::from("nested"));
        literal.set_node_property(literal_node, "value", Value::from("literal"));
        nested.sync_epoch(EpochId::new(9));
        nested.restore_allocator_high_water_exact(50, 70)?;
        literal.sync_epoch(EpochId::new(12));
        source
            .graph("")
            .ok_or("missing empty child")?
            .sync_epoch(EpochId::new(3));
        source.sync_epoch(EpochId::new(20));

        let bytes = serialize(&source)?;
        let target = Arc::new(LpgStore::new()?);
        let retained_target = Arc::clone(&target);
        super::super::LpgStoreSection::new(Arc::clone(&target)).deserialize(&bytes)?;
        assert!(Arc::ptr_eq(&target, &retained_target));
        let restored_parent = target.graph("a").ok_or("restored parent missing")?;
        let restored_nested = restored_parent
            .graph("b")
            .ok_or("restored nested missing")?;
        let restored_literal = target.graph("a/b").ok_or("restored literal missing")?;
        assert!(restored_parent.graph("").is_some());
        assert_eq!(
            target
                .graph("")
                .ok_or("restored empty child missing")?
                .current_epoch(),
            EpochId::new(3)
        );
        assert_eq!(
            restored_nested.get_node_property(nested_node, &PropertyKey::new("value")),
            Some(Value::from("nested"))
        );
        assert_eq!(
            restored_literal.get_node_property(literal_node, &PropertyKey::new("value")),
            Some(Value::from("literal"))
        );
        assert_eq!(restored_nested.current_epoch(), EpochId::new(9));
        assert_eq!(restored_literal.current_epoch(), EpochId::new(12));
        assert_eq!(
            (
                restored_nested.next_node_id(),
                restored_nested.next_edge_id()
            ),
            (50, 70)
        );
        assert_eq!(serialize(&target)?, bytes);
        let snapshot = decode(&bytes)?;
        let paths: Vec<_> = snapshot
            .graphs
            .iter()
            .map(|entry| entry.path.components().to_vec())
            .collect();
        assert_eq!(
            paths,
            vec![
                Vec::<String>::new(),
                vec![String::new()],
                vec!["a".into()],
                vec!["a".into(), String::new()],
                vec!["a".into(), "b".into()],
                vec!["a/b".into()]
            ]
        );
        Ok(())
    }

    #[test]
    fn section_round_trips_256_levels_and_refuses_a_257th_capture()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Arc::new(LpgStore::new()?);
        let mut leaf = Arc::clone(&source);
        for _ in 0..grafeo_common::types::MAX_GRAPH_PATH_COMPONENTS {
            leaf = leaf.graph_or_create("level")?;
        }
        leaf.sync_epoch(EpochId::new(5));
        let node = leaf.create_node(&["Deep"]);
        assert!(node.is_valid());
        leaf.set_node_property(node, "depth", Value::Int64(256));
        leaf.restore_allocator_high_water_exact(99, 77)?;
        source.sync_epoch(EpochId::new(11));
        let source_section = super::super::LpgStoreSection::new(Arc::clone(&source));
        let bytes = source_section.serialize()?;

        let target = Arc::new(LpgStore::new()?);
        let mut target_section = super::super::LpgStoreSection::new(Arc::clone(&target));
        target_section.deserialize(&bytes)?;
        assert!(Arc::ptr_eq(target_section.store(), &target));
        let mut restored = Arc::clone(&target);
        for _ in 0..grafeo_common::types::MAX_GRAPH_PATH_COMPONENTS {
            restored = restored.graph("level").ok_or("missing restored level")?;
        }
        assert!(restored.graph_names().is_empty());
        assert_eq!(restored.current_epoch(), EpochId::new(5));
        assert_eq!((restored.next_node_id(), restored.next_edge_id()), (99, 77));
        assert_eq!(
            restored.get_node_property(node, &PropertyKey::new("depth")),
            Some(Value::Int64(256))
        );
        assert_eq!(target.current_epoch(), EpochId::new(11));
        assert_eq!(target_section.serialize()?, bytes);

        assert!(leaf.create_graph("level")?);
        let error = source_section
            .serialize()
            .err()
            .ok_or("257-level capture was accepted")?;
        assert!(
            error.to_string().contains("exceeds 256 components"),
            "{error}"
        );
        assert!(leaf.graph("level").is_some());
        assert_eq!(
            leaf.get_node_property(node, &PropertyKey::new("depth")),
            Some(Value::Int64(256))
        );
        assert_eq!(target_section.serialize()?, bytes);
        Ok(())
    }

    #[test]
    fn section_refuses_overlong_literal_name_at_capture()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Arc::new(LpgStore::new()?);
        let name = "x".repeat(grafeo_common::types::MAX_WORLD_GRAPH_NAME_BYTES + 1);
        let child = source.graph_or_create(&name)?;
        let node = child.create_node(&["Retained"]);
        assert!(node.is_valid());
        let section = super::super::LpgStoreSection::new(Arc::clone(&source));
        let error = section
            .serialize()
            .err()
            .ok_or("overlong capture was accepted")?;
        assert!(
            error
                .to_string()
                .contains("component 0 exceeds 65536 bytes"),
            "{error}"
        );
        assert!(Arc::ptr_eq(section.store(), &source));
        assert!(Arc::ptr_eq(
            &source.graph(&name).ok_or("source child disappeared")?,
            &child
        ));
        assert!(child.get_node(node).is_some());
        Ok(())
    }

    #[test]
    fn late_child_allocation_failure_preserves_target_and_allows_retry()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::graph::write_permit::{WriteAuthority, with_authority};

        // The final path's exact Vec<u8> reservation has this unique layout:
        // count + "z" component + long component = 4 + 5 + 4 + 4086.
        const PATH_BYTES: usize = 4099;
        let late_name = "x".repeat(PATH_BYTES - 13);
        let source = Arc::new(LpgStore::new()?);
        source.sync_epoch(EpochId::new(5));
        assert!(source.create_node(&["Root"]).is_valid());
        assert!(
            source
                .graph_or_create("a")?
                .create_node(&["Early"])
                .is_valid()
        );
        let late = source.graph_or_create("z")?.graph_or_create(&late_name)?;
        assert!(late.create_node(&["Late"]).is_valid());
        late.restore_allocator_high_water_exact(90, 70)?;
        let bytes = serialize(&source)?;
        let wire = decode(&bytes)?;
        let last = wire.graphs.last().ok_or("missing final child")?;
        assert_eq!(last.path.to_bytes(PATH_BYTES)?.len(), PATH_BYTES);

        let faults = [
            (PATH_BYTES, 1, 0),
            // Candidate construction allocates initial root, root epoch 5,
            // early child, parent z, then the late child's 1 MiB/16 arena.
            // Refuse that fifth allocation, after earlier postimages exist.
            #[cfg(feature = "tiered-storage")]
            (1024 * 1024, 16, 4),
        ];
        for (size, align, skip) in faults {
            for populated in [false, true] {
                let target = Arc::new(LpgStore::new()?);
                let retained_child = if populated {
                    target.sync_epoch(EpochId::new(9));
                    assert!(target.create_node(&["KeepRoot"]).is_valid());
                    let child = target.graph_or_create("keep")?;
                    assert!(child.create_node(&["KeepChild"]).is_valid());
                    Some(child)
                } else {
                    None
                };
                let owner = WriteAuthority::new();
                assert!(target.seal_unframed_writes(&owner));
                let mut section = super::super::LpgStoreSection::new(Arc::clone(&target));
                let before = section.serialize()?;
                let epoch = target.current_epoch();
                let (failed, fired) = with_authority(&owner, || {
                    crate::allocation_test::with_failure(size, align, skip, || {
                        section.deserialize(&bytes)
                    })
                });
                assert!(
                    fired,
                    "allocation {size}/{align} after {skip} matches must actually fail"
                );
                let error = failed.err().ok_or("allocation failure was ignored")?;
                if size == PATH_BYTES {
                    assert!(
                        error.to_string().contains("memory allocation failed"),
                        "{error}"
                    );
                } else {
                    assert!(
                        matches!(
                            error,
                            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                        ),
                        "{error}"
                    );
                }
                assert!(Arc::ptr_eq(section.store(), &target));
                assert_eq!(target.current_epoch(), epoch);
                assert_eq!(section.serialize()?, before);
                assert!(!target.create_node(&["Denied"]).is_valid());
                if let Some(child) = retained_child {
                    assert!(Arc::ptr_eq(
                        &child,
                        &target.graph("keep").ok_or("lost child")?
                    ));
                    assert!(!child.create_node(&["Denied"]).is_valid());
                    assert!(with_authority(&owner, || section.deserialize(&bytes)).is_err());
                    assert_eq!(section.serialize()?, before);
                } else {
                    with_authority(&owner, || section.deserialize(&bytes))?;
                    assert!(Arc::ptr_eq(section.store(), &target));
                    assert_eq!(section.serialize()?, bytes);
                    let restored = target
                        .graph("z")
                        .ok_or("lost z")?
                        .graph(&late_name)
                        .ok_or("lost late child")?;
                    assert_eq!(restored.next_node_id(), 90);
                    assert_eq!(restored.next_edge_id(), 70);
                    assert!(!restored.create_node(&["Denied"]).is_valid());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn corrupt_late_child_semantics_preserve_section_target_bytes_and_arc()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut wire = WireSnapshotV4 {
            next_graph_incarnation_id: 10,
            graphs: vec![
                WirePathGraphV4 {
                    path: GraphPath::root(),
                    graph: empty_graph(5),
                },
                WirePathGraphV4 {
                    path: GraphPath::from_components(&["a"])?,
                    graph: empty_graph(5),
                },
                WirePathGraphV4 {
                    path: GraphPath::from_components(&["z"])?,
                    graph: empty_graph(5),
                },
                WirePathGraphV4 {
                    path: GraphPath::from_components(&["z", "late"])?,
                    graph: empty_graph(5),
                },
            ],
        };
        for (id, entry) in wire.graphs.iter_mut().enumerate() {
            entry.graph.incarnation = GraphIncarnationId::new(id as u64);
        }
        wire.graphs[0].graph.nodes.push(wire_node(1, vec!["Root"]));
        wire.graphs[0].graph.next_node_id = 2;
        wire.graphs[1].graph.nodes.push(wire_node(2, vec!["Early"]));
        wire.graphs[1].graph.next_node_id = 3;
        wire.graphs[3].graph.nodes.push(wire_node(3, vec!["Late"]));
        wire.graphs[3].graph.next_node_id = 4;

        let target = Arc::new(LpgStore::new()?);
        let mut section = super::super::LpgStoreSection::new(Arc::clone(&target));
        let before = section.serialize()?;
        let valid = encode_wire(&wire)?;
        assert!(decode(&valid).is_ok());
        wire.graphs[3].graph.nodes[0].lifetimes[0].deleted = Some(0);
        let error = section
            .deserialize(&encode_wire(&wire)?)
            .err()
            .ok_or("late child inverted lifetime was accepted")?;
        assert!(
            error
                .to_string()
                .contains("node lifetime 0 has invalid delete epoch"),
            "{error}"
        );
        assert!(Arc::ptr_eq(section.store(), &target));
        assert_eq!(section.serialize()?, before);
        assert!(target.graph_names().is_empty());
        assert_eq!(target.node_count(), 0);
        // The same untouched target must remain usable for the healthy image.
        section.deserialize(&valid)?;
        assert!(Arc::ptr_eq(section.store(), &target));
        assert_eq!(section.serialize()?, valid);
        let late = target
            .graph("z")
            .ok_or("missing z")?
            .graph("late")
            .ok_or("missing late child")?;
        assert!(late.get_node(NodeId::new(3)).is_some());
        Ok(())
    }

    #[test]
    fn malformed_graph_path_order_and_parentage_leave_target_unchanged()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cases: &[&[&[&str]]] = &[
            &[],
            &[&["a"]],
            &[&[], &[]],
            &[&[], &["z"], &["a"]],
            &[&[], &["a", "missing-parent", "child"]],
            &[&[], &["a/b"], &["a", "b"]],
        ];
        for paths in cases {
            let wire = WireSnapshotV4 {
                next_graph_incarnation_id: 10,
                graphs: paths
                    .iter()
                    .map(|parts| {
                        Ok(WirePathGraphV4 {
                            path: GraphPath::from_components(parts)?,
                            graph: empty_graph(0),
                        })
                    })
                    .collect::<std::result::Result<Vec<_>, grafeo_common::types::GraphPathError>>(
                    )?,
            };
            let target = LpgStore::new()?;
            let before = serialize(&target)?;
            assert!(deserialize_into(&target, &encode_wire(&wire)?).is_err());
            assert_eq!(serialize(&target)?, before);
        }
        Ok(())
    }

    #[test]
    fn endpoint_lifetime_escape_is_rejected_before_any_restore_mutation() {
        for endpoint_id in [1, 2] {
            for endpoint_lives in [
                vec![(1, Some(3))],
                vec![(1, Some(3)), (4, None)],
                vec![(1, Some(3)), (3, None)],
            ] {
                let mut nodes = vec![wire_node(1, vec!["Source"]), wire_node(2, vec!["Target"])];
                let endpoint = &mut nodes[endpoint_id - 1];
                endpoint.lifetimes = endpoint_lives
                    .iter()
                    .map(|&(created, deleted)| WireLifetimeV4 { created, deleted })
                    .collect();
                endpoint.label_history = endpoint_lives
                    .iter()
                    .map(|&(epoch, _)| WireLabelVersionV4 {
                        epoch,
                        labels: vec!["Endpoint".to_owned()],
                    })
                    .collect();
                let wire = wire_root(WireGraphV4 {
                    incarnation: GraphIncarnationId::DEFAULT_GRAPH,
                    committed_epoch: 5,
                    retained_history_floor: 5,
                    next_node_id: 3,
                    next_edge_id: 3,
                    nodes,
                    edges: vec![
                        WireEdgeV4 {
                            id: 1,
                            src: 1,
                            dst: 2,
                            edge_type: "VALID".to_owned(),
                            lifetimes: vec![WireLifetimeV4 {
                                created: 1,
                                deleted: Some(2),
                            }],
                            properties: Vec::new(),
                        },
                        WireEdgeV4 {
                            id: 2,
                            src: 1,
                            dst: 2,
                            edge_type: "ESCAPES".to_owned(),
                            lifetimes: vec![WireLifetimeV4 {
                                created: 2,
                                deleted: Some(5),
                            }],
                            properties: Vec::new(),
                        },
                    ],
                });
                let bytes = encode_wire(&wire).expect("well-framed hostile history");
                let target = LpgStore::new().expect("target");
                let before = serialize(&target).expect("pristine image");
                let error = deserialize_into(&target, &bytes)
                    .expect_err("edge must fit within one lifetime of each endpoint");
                assert!(error.to_string().contains("outlives"), "{error}");
                assert_eq!(serialize(&target).expect("unchanged target"), before);
                assert!(target.all_node_ids().is_empty());
                assert!(target.all_known_edge_ids().is_empty());
                assert_eq!(target.current_epoch(), EpochId::INITIAL);
                assert_eq!((target.next_node_id(), target.next_edge_id()), (0, 0));
            }
        }
    }

    #[test]
    fn current_lpg_value_variants_round_trip_exactly() {
        let source = LpgStore::new().expect("source");
        let node = source.create_node(&["Values"]);
        let mut counter = HashMap::new();
        counter.insert("z".to_owned(), 7);
        counter.insert("a".to_owned(), 3);
        let values = vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(-42),
            Value::Float64(f64::from_bits(0x7ff8_0000_0000_0042)),
            Value::from("é"),
            Value::Bytes(vec![0, 255].into()),
            Value::Timestamp(Timestamp::from_micros(-1234567)),
            Value::Date(Date::from_days(-42)),
            Value::Time(Time::from_nanos(123456789).expect("time").with_offset(3600)),
            Value::Duration(Duration::new(-2, 3, -4)),
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1234567),
                -3600,
            )),
            Value::Map(Arc::new(BTreeMap::from([(
                PropertyKey::new("nested"),
                Value::Int64(9),
            )]))),
            Value::Vector(vec![f32::from_bits(0x7fc0_0042), -0.0].into()),
            Value::Path {
                nodes: vec![Value::Int64(1)].into(),
                edges: Vec::new().into(),
            },
            Value::GCounter(Arc::new(counter.clone())),
            Value::OnCounter {
                pos: Arc::new(counter),
                neg: Arc::new(HashMap::from([("b".to_owned(), 2)])),
            },
            Value::RdfLiteral {
                lexical: "bonjour".into(),
                language: Some("fr".into()),
                datatype: None,
            },
        ];
        source.set_node_property(node, "values", Value::List(values.into()));
        let bytes = serialize(&source).expect("canonical current writer");
        let target = LpgStore::new().expect("target");
        deserialize_into(&target, &bytes).expect("restore all Value variants");
        assert_eq!(serialize(&target).expect("canonical restored bytes"), bytes);
    }

    #[test]
    fn semantic_error_in_later_entity_is_rejected_before_any_restore_mutation() {
        let mut invalid_node = wire_node(2, vec!["Missing"]);
        invalid_node.label_history.clear();
        let wire = wire_root(WireGraphV4 {
            incarnation: GraphIncarnationId::DEFAULT_GRAPH,
            committed_epoch: 3,
            retained_history_floor: 3,
            next_node_id: 3,
            next_edge_id: 0,
            nodes: vec![
                wire_node(1, vec!["Valid"]),
                // Missing the mandatory complete label set at create.
                invalid_node,
            ],
            edges: Vec::new(),
        });
        let bytes = encode_wire(&wire).expect("well-framed hostile fixture");
        let target = LpgStore::new().expect("target store");

        let error = deserialize_into(&target, &bytes)
            .expect_err("semantic corruption must fail before restore starts");

        assert!(error.to_string().contains("no complete label history"));
        assert!(target.all_node_ids().is_empty());
        assert!(target.all_known_edge_ids().is_empty());
        assert_eq!(target.current_epoch(), EpochId::INITIAL);
        assert_eq!(target.next_node_id(), 0);
        assert_eq!(target.next_edge_id(), 0);
    }

    #[test]
    fn zero_width_node_and_edge_histories_round_trip_canonically_but_never_appear() {
        let source = LpgStore::new().expect("source store");
        let committed = EpochId::new(70);
        let before = EpochId::new(69);
        let after = EpochId::new(71);
        let ephemeral = NodeId::new(70);
        let anchor = NodeId::new(71);
        let edge = EdgeId::new(170);

        source
            .restore_node_history_exact(
                ephemeral,
                &[(committed, Some(committed))],
                &[
                    (committed, vec![ArcStr::from("Draft")]),
                    (
                        committed,
                        vec![ArcStr::from("Draft"), ArcStr::from("Reviewed")],
                    ),
                ],
            )
            .expect("restore zero-width source node");
        source
            .restore_node_history_exact(
                anchor,
                &[(committed, None)],
                &[(committed, vec![ArcStr::from("Anchor")])],
            )
            .expect("restore live endpoint");
        source.set_node_property_at_epoch(ephemeral, "title", Value::from("ephemeral"), committed);
        source.set_node_property_at_epoch(ephemeral, "title", Value::Null, committed);
        source
            .restore_edge_history_exact(
                edge,
                ephemeral,
                anchor,
                "TEMPORARY",
                &[(committed, Some(committed))],
            )
            .expect("restore zero-width source edge");
        source.set_edge_property_at_epoch(edge, "weight", Value::Int64(7), committed);
        source.set_edge_property_at_epoch(edge, "weight", Value::Null, committed);
        source
            .restore_allocator_high_water_exact(100, 200)
            .expect("restore allocator gaps");

        let bytes = serialize(&source).expect("serialize exact zero-width history");
        assert_eq!(
            serialize(&source).expect("repeat canonical serialization"),
            bytes
        );
        let target = LpgStore::new().expect("target store");
        deserialize_into(&target, &bytes).expect("restore exact zero-width history");
        assert_eq!(
            serialize(&target).expect("re-export restored history"),
            bytes,
            "exact restore must reproduce canonical section bytes"
        );

        for epoch in [before, committed, after] {
            assert!(target.get_node_at_epoch(ephemeral, epoch).is_none());
            assert!(target.get_edge_at_epoch(edge, epoch).is_none());
        }
        assert!(target.get_node(ephemeral).is_none());
        assert!(target.get_edge(edge).is_none());
        assert!(target.all_node_ids().contains(&ephemeral));
        assert!(target.all_known_edge_ids().contains(&edge));
        assert_eq!(
            target
                .get_node_history(ephemeral)
                .into_iter()
                .map(|(created, deleted, _)| (created, deleted))
                .collect::<Vec<_>>(),
            vec![(committed, Some(committed))]
        );
        let edge_history = target.get_edge_history(edge);
        assert_eq!(edge_history.len(), 1);
        assert_eq!(
            (edge_history[0].0, edge_history[0].1),
            (committed, Some(committed))
        );
        assert_eq!(edge_history[0].2.src, ephemeral);
        assert_eq!(edge_history[0].2.dst, anchor);
        assert_eq!(edge_history[0].2.edge_type.as_str(), "TEMPORARY");
        assert_eq!(
            target.node_label_history(ephemeral),
            vec![
                (committed, vec![ArcStr::from("Draft")]),
                (
                    committed,
                    vec![ArcStr::from("Draft"), ArcStr::from("Reviewed")],
                ),
            ]
        );
        assert_eq!(
            target.node_property_history_for_key(ephemeral, "title"),
            vec![
                (committed, Value::from("ephemeral")),
                (committed, Value::Null),
            ]
        );
        assert_eq!(
            target.edge_property_history(edge),
            vec![(
                PropertyKey::new("weight"),
                vec![(committed, Value::Int64(7)), (committed, Value::Null)],
            )]
        );
        assert_eq!(target.current_epoch(), committed);
        assert_eq!(target.next_node_id(), 100);
        assert_eq!(target.next_edge_id(), 200);
    }

    #[test]
    fn inverted_and_pending_lifetimes_are_rejected_before_restore_mutates_target() {
        let assert_rejected_pristine = |wire: WireSnapshotV4, expected: &str| {
            let bytes = encode_wire(&wire).expect("well-framed hostile fixture");
            let target = LpgStore::new().expect("target store");
            let error = deserialize_into(&target, &bytes)
                .expect_err("hostile lifetime must fail before restore");
            assert!(
                error.to_string().contains(expected),
                "unexpected error: {error}"
            );
            assert!(target.all_node_ids().is_empty());
            assert!(target.all_known_edge_ids().is_empty());
            assert!(target.graph_names().is_empty());
            assert_eq!(target.current_epoch(), EpochId::INITIAL);
            assert_eq!(target.next_node_id(), 0);
            assert_eq!(target.next_edge_id(), 0);
        };

        let mut inverted = wire_node(2, vec!["Inverted"]);
        inverted.lifetimes[0] = WireLifetimeV4 {
            created: 3,
            deleted: Some(2),
        };
        inverted.label_history[0].epoch = 3;
        assert_rejected_pristine(
            wire_root(WireGraphV4 {
                incarnation: GraphIncarnationId::DEFAULT_GRAPH,
                committed_epoch: 3,
                retained_history_floor: 3,
                next_node_id: 3,
                next_edge_id: 0,
                nodes: vec![wire_node(1, vec!["Valid"]), inverted],
                edges: Vec::new(),
            }),
            "created <= deleted",
        );

        assert_rejected_pristine(
            wire_root(WireGraphV4 {
                incarnation: GraphIncarnationId::DEFAULT_GRAPH,
                committed_epoch: 3,
                retained_history_floor: 3,
                next_node_id: 3,
                next_edge_id: 10,
                nodes: vec![wire_node(1, vec!["Source"]), wire_node(2, vec!["Target"])],
                edges: vec![WireEdgeV4 {
                    id: 9,
                    src: 1,
                    dst: 2,
                    edge_type: String::from("PENDING"),
                    lifetimes: vec![WireLifetimeV4 {
                        created: 1,
                        deleted: Some(EpochId::PENDING.as_u64()),
                    }],
                    properties: Vec::new(),
                }],
            }),
            "PENDING",
        );
    }
}
