//! Persistence, snapshots, and data export for GrafeoDB.

#[cfg(any(feature = "wal", feature = "grafeo-file"))]
use std::path::Path;

#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::grafeo_warn;
use grafeo_common::{grafeo_debug_span, grafeo_info, grafeo_info_span};
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use hashbrown::{HashMap, HashSet};

use crate::config::Config;

#[cfg(feature = "wal")]
use grafeo_storage::wal::WalRecord;

use crate::catalog::{
    EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition, ProcedureDefinition,
};

/// Current snapshot version.
const SNAPSHOT_VERSION: u8 = 4;

/// How `open_multi` should reconcile schema catalogs across snapshots.
///
/// The default ([`SchemaMergePolicy::UnionWithConflictCheck`]) matches
/// the natural pattern of "shared chunk has shared DDL, niche chunk
/// has niche-specific DDL, no overlap on type names." For callers who
/// genuinely need byte-equal schemas across all chunks (e.g. testing
/// snapshot determinism), set [`SchemaMergePolicy::StrictEquality`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SchemaMergePolicy {
    /// Union types across all snapshots. Same-name types must have
    /// matching definitions (compared via canonical bincode bytes);
    /// differently-named types are accumulated into the merged
    /// catalog. This is the default — required for sibling extracts
    /// of a single source DB where each carries only the relevant
    /// types but every type appears in at least one extract.
    #[default]
    UnionWithConflictCheck,

    /// All snapshots must declare identical schemas (after canonical
    /// ordering of inner Vecs). Stricter than `UnionWithConflictCheck`
    /// — useful for tests that want to detect any schema drift, not
    /// just incompatible drift.
    StrictEquality,
}

/// Configuration for [`GrafeoDB::open_multi_with`]. Cheap to construct
/// via `..Default::default()` field updates.
#[derive(Debug, Clone, Default)]
pub struct OpenMultiOptions {
    /// How to reconcile schema catalogs across the input snapshots.
    /// See [`SchemaMergePolicy`].
    pub schema_policy: SchemaMergePolicy,
}

/// Binary snapshot format (v4: graph data, named graphs, RDF, schema, index metadata,
/// and property version history for temporal support).
#[derive(serde::Serialize, serde::Deserialize)]
struct Snapshot {
    version: u8,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
    named_graphs: Vec<NamedGraphSnapshot>,
    rdf_triples: Vec<SnapshotTriple>,
    rdf_named_graphs: Vec<RdfNamedGraphSnapshot>,
    schema: SnapshotSchema,
    indexes: SnapshotIndexes,
    /// Current store epoch at snapshot time (0 when temporal is disabled).
    epoch: u64,
}

/// Schema metadata within a snapshot.
#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
struct SnapshotSchema {
    node_types: Vec<NodeTypeDefinition>,
    edge_types: Vec<EdgeTypeDefinition>,
    graph_types: Vec<GraphTypeDefinition>,
    procedures: Vec<ProcedureDefinition>,
    schemas: Vec<String>,
    graph_type_bindings: Vec<(String, String)>,
}

/// Index metadata within a snapshot (definitions only, not index data).
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SnapshotIndexes {
    property_indexes: Vec<String>,
    vector_indexes: Vec<SnapshotVectorIndex>,
    text_indexes: Vec<SnapshotTextIndex>,
}

/// Vector index definition for snapshot persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotVectorIndex {
    label: String,
    property: String,
    dimensions: usize,
    metric: grafeo_core::index::vector::DistanceMetric,
    m: usize,
    ef_construction: usize,
}

/// Text index definition for snapshot persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotTextIndex {
    label: String,
    property: String,
}

/// A named graph partition within a v2 snapshot.
#[derive(serde::Serialize, serde::Deserialize)]
struct NamedGraphSnapshot {
    name: String,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
}

/// An RDF triple in snapshot format (N-Triples encoded terms).
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotTriple {
    subject: String,
    predicate: String,
    object: String,
}

/// An RDF named graph in snapshot format.
#[derive(serde::Serialize, serde::Deserialize)]
struct RdfNamedGraphSnapshot {
    name: String,
    triples: Vec<SnapshotTriple>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotNode {
    id: NodeId,
    labels: Vec<String>,
    /// Each property has a list of `(epoch, value)` entries (ascending epoch order).
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    /// Each property has a list of `(epoch, value)` entries (ascending epoch order).
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

/// Collects all nodes from a store into snapshot format.
///
/// With `temporal`: stores full property version history.
/// Without: wraps each current value as a single-entry version list at epoch 0.
fn collect_snapshot_nodes(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<SnapshotNode> {
    let mut nodes: Vec<SnapshotNode> = store
        .all_nodes()
        .map(|n| {
            #[cfg(feature = "temporal")]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
                .node_property_history(n.id)
                .into_iter()
                .map(|(k, entries)| (k.to_string(), entries))
                .collect();

            #[cfg(not(feature = "temporal"))]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = n
                .properties
                .into_iter()
                .map(|(k, v)| (k.to_string(), vec![(EpochId::new(0), v)]))
                .collect();

            properties.sort_by(|(a, _), (b, _)| a.cmp(b));

            let mut labels: Vec<String> = n.labels.iter().map(|l| l.to_string()).collect();
            labels.sort();

            SnapshotNode {
                id: n.id,
                labels,
                properties,
            }
        })
        .collect();
    nodes.sort_by_key(|n| n.id);
    nodes
}

/// Collects all edges from a store into snapshot format.
///
/// With `temporal`: stores full property version history.
/// Without: wraps each current value as a single-entry version list at epoch 0.
fn collect_snapshot_edges(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<SnapshotEdge> {
    let mut edges: Vec<SnapshotEdge> = store
        .all_edges()
        .map(|e| {
            #[cfg(feature = "temporal")]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
                .edge_property_history(e.id)
                .into_iter()
                .map(|(k, entries)| (k.to_string(), entries))
                .collect();

            #[cfg(not(feature = "temporal"))]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = e
                .properties
                .into_iter()
                .map(|(k, v)| (k.to_string(), vec![(EpochId::new(0), v)]))
                .collect();

            properties.sort_by(|(a, _), (b, _)| a.cmp(b));

            SnapshotEdge {
                id: e.id,
                src: e.src,
                dst: e.dst,
                edge_type: e.edge_type.to_string(),
                properties,
            }
        })
        .collect();
    edges.sort_by_key(|e| e.id);
    edges
}

/// Populates a store from snapshot node/edge data.
///
/// With `temporal`: replays all `(epoch, value)` entries into version logs.
/// Without: reads the latest value from each property's version list.
fn populate_store_from_snapshot(
    store: &grafeo_core::graph::lpg::LpgStore,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
) -> Result<()> {
    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.create_node_with_id(node.id, &label_refs)?;
        for (key, entries) in node.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_node_property_at_epoch(node.id, &key, value, epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.into_iter().last() {
                store.set_node_property(node.id, &key, value);
            }
        }
    }
    for edge in edges {
        store.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        for (key, entries) in edge.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_edge_property_at_epoch(edge.id, &key, value, epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.into_iter().last() {
                store.set_edge_property(edge.id, &key, value);
            }
        }
    }
    Ok(())
}

/// Validates snapshot nodes/edges for duplicates within a single
/// snapshot. Does NOT check edge endpoint resolution — that's done
/// separately so multi-snapshot callers can validate endpoints across
/// the union.
fn validate_snapshot_ids(nodes: &[SnapshotNode], edges: &[SnapshotEdge]) -> Result<()> {
    let mut node_ids = HashSet::with_capacity(nodes.len());
    for node in nodes {
        if !node_ids.insert(node.id) {
            return Err(Error::Internal(format!(
                "snapshot contains duplicate node ID {}",
                node.id
            )));
        }
    }
    let mut edge_ids = HashSet::with_capacity(edges.len());
    for edge in edges {
        if !edge_ids.insert(edge.id) {
            return Err(Error::Internal(format!(
                "snapshot contains duplicate edge ID {}",
                edge.id
            )));
        }
    }
    Ok(())
}

/// Validates snapshot nodes/edges for duplicates AND that every edge
/// endpoint resolves inside the same snapshot. Used by the single-
/// snapshot path (`import_snapshot`, `restore_snapshot`) where edges
/// must be self-contained.
fn validate_snapshot_data(nodes: &[SnapshotNode], edges: &[SnapshotEdge]) -> Result<()> {
    validate_snapshot_ids(nodes, edges)?;
    let node_ids: HashSet<NodeId> = nodes.iter().map(|n| n.id).collect();
    for edge in edges {
        if !node_ids.contains(&edge.src) {
            return Err(Error::Internal(format!(
                "snapshot edge {} references non-existent source node {}",
                edge.id, edge.src
            )));
        }
        if !node_ids.contains(&edge.dst) {
            return Err(Error::Internal(format!(
                "snapshot edge {} references non-existent destination node {}",
                edge.id, edge.dst
            )));
        }
    }
    Ok(())
}

/// Produces a canonical byte representation of a `SnapshotSchema` for
/// cross-snapshot equality comparison. `collect_schema()` reads from
/// `HashMap::values()` so the *outer* Vec ordering varies across runs;
/// sort each outer Vec by its primary name field before encoding.
///
/// **Inner ordering assumption.** Within each type definition, nested
/// `properties` / `constraints` Vecs are NOT re-sorted — we treat their
/// DDL-insertion order as part of the schema identity. Two snapshots
/// of the same logical schema produced by the same DDL batch will
/// share inner ordering and round-trip cleanly. If a future code path
/// builds the same logical schema via different DDL insertion orders,
/// extend the sort here to normalise the nested Vecs too (will require
/// `Ord` on `TypeConstraint`).
fn normalize_schema_bytes(schema: &SnapshotSchema) -> Vec<u8> {
    let mut node_types = schema.node_types.clone();
    node_types.sort_by(|a, b| a.name.cmp(&b.name));
    let mut edge_types = schema.edge_types.clone();
    edge_types.sort_by(|a, b| a.name.cmp(&b.name));
    let mut graph_types = schema.graph_types.clone();
    graph_types.sort_by(|a, b| a.name.cmp(&b.name));
    let mut procedures = schema.procedures.clone();
    procedures.sort_by(|a, b| a.name.cmp(&b.name));
    let mut schemas = schema.schemas.clone();
    schemas.sort();
    let mut bindings = schema.graph_type_bindings.clone();
    bindings.sort();

    let normalized = SnapshotSchema {
        node_types,
        edge_types,
        graph_types,
        procedures,
        schemas,
        graph_type_bindings: bindings,
    };
    bincode::serde::encode_to_vec(&normalized, bincode::config::standard())
        .expect("encoding a SnapshotSchema cannot fail")
}

/// Merges the schemas from multiple decoded snapshots into one
/// `SnapshotSchema`, per the chosen policy. Returns the merged schema
/// ready to hand to `restore_schema_from_snapshot`.
fn merge_snapshot_schemas(
    snapshots: &[Snapshot],
    policy: SchemaMergePolicy,
) -> Result<SnapshotSchema> {
    match policy {
        SchemaMergePolicy::StrictEquality => {
            let canonical = normalize_schema_bytes(&snapshots[0].schema);
            for (idx, snap) in snapshots.iter().enumerate().skip(1) {
                if normalize_schema_bytes(&snap.schema) != canonical {
                    return Err(Error::Internal(format!(
                        "snapshot[{idx}] schema does not match snapshot[0] schema \
                         (SchemaMergePolicy::StrictEquality); all snapshots must \
                         declare byte-identical schemas after canonical ordering"
                    )));
                }
            }
            Ok(snapshots[0].schema.clone())
        }
        SchemaMergePolicy::UnionWithConflictCheck => {
            // Each typed catalog field is HashMap<name, def>. Inserting
            // again with the same name requires the definitions to be
            // byte-equal (post-normalization within each definition).
            // Same-name+different-shape rejects with a diagnostic.

            let mut node_types: HashMap<String, NodeTypeDefinition> = HashMap::new();
            let mut edge_types: HashMap<String, EdgeTypeDefinition> = HashMap::new();
            let mut graph_types: HashMap<String, GraphTypeDefinition> = HashMap::new();
            let mut procedures: HashMap<String, ProcedureDefinition> = HashMap::new();
            let mut schemas: HashSet<String> = HashSet::new();
            let mut bindings: HashMap<String, String> = HashMap::new();

            let cfg = bincode::config::standard();

            for (idx, snap) in snapshots.iter().enumerate() {
                for def in &snap.schema.node_types {
                    match node_types.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .expect("schema entry encode");
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).expect("schema entry encode");
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines NodeType {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for def in &snap.schema.edge_types {
                    match edge_types.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .expect("schema entry encode");
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).expect("schema entry encode");
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines EdgeType {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for def in &snap.schema.graph_types {
                    match graph_types.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .expect("schema entry encode");
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).expect("schema entry encode");
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines GraphType {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for def in &snap.schema.procedures {
                    match procedures.entry(def.name.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            let existing_bytes = bincode::serde::encode_to_vec(existing.get(), cfg)
                                .expect("schema entry encode");
                            let def_bytes =
                                bincode::serde::encode_to_vec(def, cfg).expect("schema entry encode");
                            if existing_bytes != def_bytes {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] redefines Procedure {:?} with \
                                     a shape that differs from an earlier snapshot",
                                    def.name
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(def.clone());
                        }
                    }
                }
                for s in &snap.schema.schemas {
                    schemas.insert(s.clone());
                }
                for (gname, gtype) in &snap.schema.graph_type_bindings {
                    match bindings.entry(gname.clone()) {
                        hashbrown::hash_map::Entry::Occupied(existing) => {
                            if existing.get() != gtype {
                                return Err(Error::Internal(format!(
                                    "snapshot[{idx}] binds graph {gname:?} to \
                                     {gtype:?} but an earlier snapshot bound it \
                                     to {:?}",
                                    existing.get()
                                )));
                            }
                        }
                        hashbrown::hash_map::Entry::Vacant(v) => {
                            v.insert(gtype.clone());
                        }
                    }
                }
            }

            let mut merged = SnapshotSchema {
                node_types: node_types.into_values().collect(),
                edge_types: edge_types.into_values().collect(),
                graph_types: graph_types.into_values().collect(),
                procedures: procedures.into_values().collect(),
                schemas: schemas.into_iter().collect(),
                graph_type_bindings: bindings.into_iter().collect(),
            };
            // Sort the merged vecs so the result is deterministic and
            // the eventual round-trip through export_snapshot is
            // reproducible.
            merged.node_types.sort_by(|a, b| a.name.cmp(&b.name));
            merged.edge_types.sort_by(|a, b| a.name.cmp(&b.name));
            merged.graph_types.sort_by(|a, b| a.name.cmp(&b.name));
            merged.procedures.sort_by(|a, b| a.name.cmp(&b.name));
            merged.schemas.sort();
            merged.graph_type_bindings.sort();
            Ok(merged)
        }
    }
}

/// Origin information for a node observed during cross-snapshot
/// validation. Captured the first time a NodeId is seen so collision
/// errors can name BOTH conflicting sides.
#[derive(Debug, Clone)]
struct NodeOrigin {
    snapshot_idx: usize,
    /// Sorted to match `collect_snapshot_nodes`' canonical ordering;
    /// safe to compare across runs.
    labels: Vec<String>,
    /// The external `id` property value if present — gives the operator
    /// a domain handle when debugging a collision (e.g. `concept:bitter`).
    id_prop: Option<String>,
}

impl NodeOrigin {
    fn from_node(snapshot_idx: usize, node: &SnapshotNode) -> Self {
        let id_prop = node.properties.iter().find_map(|(key, history)| {
            if key != "id" {
                return None;
            }
            history.last().and_then(|(_, value)| match value {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
        });
        Self {
            snapshot_idx,
            labels: node.labels.clone(),
            id_prop,
        }
    }

    fn describe(&self) -> String {
        let labels = if self.labels.is_empty() {
            String::from("[]")
        } else {
            format!("[{}]", self.labels.join(","))
        };
        match &self.id_prop {
            Some(id) => format!("snapshot[{}] (labels={labels}, id={id:?})", self.snapshot_idx),
            None => format!("snapshot[{}] (labels={labels})", self.snapshot_idx),
        }
    }
}

#[derive(Debug, Clone)]
struct EdgeOrigin {
    snapshot_idx: usize,
    edge_type: String,
    src: NodeId,
    dst: NodeId,
}

impl EdgeOrigin {
    fn from_edge(snapshot_idx: usize, edge: &SnapshotEdge) -> Self {
        Self {
            snapshot_idx,
            edge_type: edge.edge_type.clone(),
            src: edge.src,
            dst: edge.dst,
        }
    }

    fn describe(&self) -> String {
        format!(
            "snapshot[{}] ({}→{} :{})",
            self.snapshot_idx, self.src, self.dst, self.edge_type
        )
    }
}

/// Validates that node IDs and edge IDs are disjoint across all
/// snapshots, and that every edge endpoint resolves somewhere in the
/// union of all snapshots' nodes.
///
/// Per-snapshot duplicate-ID validation is done separately by
/// `validate_snapshot_ids` (multi-snapshot path) or
/// `validate_snapshot_data` (single-snapshot path) and runs first.
fn validate_snapshot_set(snapshots: &[Snapshot]) -> Result<()> {
    let mut node_origins: HashMap<NodeId, NodeOrigin> = HashMap::with_capacity(
        snapshots.iter().map(|s| s.nodes.len()).sum(),
    );

    for (idx, snap) in snapshots.iter().enumerate() {
        for node in &snap.nodes {
            match node_origins.entry(node.id) {
                hashbrown::hash_map::Entry::Occupied(existing) => {
                    return Err(Error::Internal(format!(
                        "duplicate NodeId {id} is claimed by {prev} and by {curr}; \
                         open_multi requires NodeIds to be disjoint across \
                         snapshots — extract sibling chunks from a single \
                         source DB to preserve a shared ID namespace",
                        id = node.id,
                        prev = existing.get().describe(),
                        curr = NodeOrigin::from_node(idx, node).describe(),
                    )));
                }
                hashbrown::hash_map::Entry::Vacant(vacant) => {
                    vacant.insert(NodeOrigin::from_node(idx, node));
                }
            }
        }
    }

    let mut edge_origins: HashMap<EdgeId, EdgeOrigin> = HashMap::with_capacity(
        snapshots.iter().map(|s| s.edges.len()).sum(),
    );

    for (idx, snap) in snapshots.iter().enumerate() {
        for edge in &snap.edges {
            match edge_origins.entry(edge.id) {
                hashbrown::hash_map::Entry::Occupied(existing) => {
                    return Err(Error::Internal(format!(
                        "duplicate EdgeId {id} is claimed by {prev} and by {curr}; \
                         open_multi requires EdgeIds to be disjoint across \
                         snapshots",
                        id = edge.id,
                        prev = existing.get().describe(),
                        curr = EdgeOrigin::from_edge(idx, edge).describe(),
                    )));
                }
                hashbrown::hash_map::Entry::Vacant(vacant) => {
                    vacant.insert(EdgeOrigin::from_edge(idx, edge));
                }
            }
            if !node_origins.contains_key(&edge.src) {
                return Err(Error::Internal(format!(
                    "snapshot[{idx}] edge {} references non-existent source \
                     node {} (not present in any snapshot)",
                    edge.id, edge.src
                )));
            }
            if !node_origins.contains_key(&edge.dst) {
                return Err(Error::Internal(format!(
                    "snapshot[{idx}] edge {} references non-existent \
                     destination node {} (not present in any snapshot)",
                    edge.id, edge.dst
                )));
            }
        }
    }

    Ok(())
}

/// Collects all triples from an RDF store into snapshot format.
#[cfg(feature = "triple-store")]
fn collect_rdf_triples(store: &grafeo_core::graph::rdf::RdfStore) -> Vec<SnapshotTriple> {
    store
        .triples()
        .into_iter()
        .map(|t| SnapshotTriple {
            subject: t.subject().to_string(),
            predicate: t.predicate().to_string(),
            object: t.object().to_string(),
        })
        .collect()
}

/// Populates an RDF store from snapshot triples.
#[cfg(feature = "triple-store")]
fn populate_rdf_store(store: &grafeo_core::graph::rdf::RdfStore, triples: &[SnapshotTriple]) {
    use grafeo_core::graph::rdf::{Term, Triple};
    for triple in triples {
        if let (Some(s), Some(p), Some(o)) = (
            Term::from_ntriples(&triple.subject),
            Term::from_ntriples(&triple.predicate),
            Term::from_ntriples(&triple.object),
        ) {
            store.insert(Triple::new(s, p, o));
        }
    }
}

// =========================================================================
// Snapshot deserialization helpers (used by single-file format)
// =========================================================================

/// Decodes snapshot bytes and populates a store and catalog.
#[cfg(feature = "grafeo-file")]
pub(super) fn load_snapshot_into_store(
    store: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &std::sync::Arc<grafeo_core::graph::rdf::RdfStore>,
    data: &[u8],
) -> grafeo_common::utils::error::Result<()> {
    use grafeo_common::utils::error::Error;

    let config = bincode::config::standard();
    let (snapshot, _) =
        bincode::serde::decode_from_slice::<Snapshot, _>(data, config).map_err(|e| {
            Error::Serialization(format!("failed to decode snapshot from .grafeo file: {e}"))
        })?;

    populate_store_from_snapshot_ref(store, &snapshot.nodes, &snapshot.edges)?;

    // Restore epoch from snapshot (store-level only; TransactionManager
    // sync is handled in with_config() after all recovery completes).
    #[cfg(feature = "temporal")]
    store.sync_epoch(EpochId::new(snapshot.epoch));

    for graph in &snapshot.named_graphs {
        store
            .create_graph(&graph.name)
            .map_err(|e| Error::Internal(e.to_string()))?;
        if let Some(graph_store) = store.graph(&graph.name) {
            populate_store_from_snapshot_ref(&graph_store, &graph.nodes, &graph.edges)?;
            #[cfg(feature = "temporal")]
            graph_store.sync_epoch(EpochId::new(snapshot.epoch));
        }
    }
    restore_schema_from_snapshot(store, catalog, &snapshot.schema);

    // Restore RDF triples
    #[cfg(feature = "triple-store")]
    {
        populate_rdf_store(rdf_store, &snapshot.rdf_triples);
        for rdf_graph in &snapshot.rdf_named_graphs {
            rdf_store.create_graph(&rdf_graph.name);
            if let Some(graph_store) = rdf_store.graph(&rdf_graph.name) {
                populate_rdf_store(&graph_store, &rdf_graph.triples);
            }
        }
    }

    Ok(())
}

/// Populates a store from snapshot refs (borrowed). Used by `open_multi`
/// and by the single-file loader.
fn populate_store_from_snapshot_ref(
    store: &grafeo_core::graph::lpg::LpgStore,
    nodes: &[SnapshotNode],
    edges: &[SnapshotEdge],
) -> grafeo_common::utils::error::Result<()> {
    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.create_node_with_id(node.id, &label_refs)?;
        for (key, entries) in &node.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_node_property_at_epoch(node.id, key, value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_node_property(node.id, key, value.clone());
            }
        }
    }
    for edge in edges {
        store.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        for (key, entries) in &edge.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_edge_property_at_epoch(edge.id, key, value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_edge_property(edge.id, key, value.clone());
            }
        }
    }
    Ok(())
}

/// Restores schema definitions from a snapshot into the catalog.
///
/// Also ensures each schema has its `__default__` graph partition, which
/// may be missing in snapshots created before the schema hierarchy feature.
fn restore_schema_from_snapshot(
    store: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    schema: &SnapshotSchema,
) {
    for def in &schema.node_types {
        catalog.register_or_replace_node_type(def.clone());
    }
    for def in &schema.edge_types {
        catalog.register_or_replace_edge_type_def(def.clone());
    }
    for def in &schema.graph_types {
        let _ = catalog.register_graph_type(def.clone());
    }
    for def in &schema.procedures {
        catalog.replace_procedure(def.clone()).ok();
    }
    for name in &schema.schemas {
        let _ = catalog.register_schema_namespace(name.clone());
        // Ensure the schema's default graph partition exists
        let default_key = format!("{name}/__default__");
        let _ = store.create_graph(&default_key);
    }
    for (graph_name, type_name) in &schema.graph_type_bindings {
        let _ = catalog.bind_graph_type(graph_name, type_name.clone());
    }
}

/// Collects schema definitions from the catalog into snapshot format.
fn collect_schema(catalog: &std::sync::Arc<crate::catalog::Catalog>) -> SnapshotSchema {
    SnapshotSchema {
        node_types: catalog.all_node_type_defs(),
        edge_types: catalog.all_edge_type_defs(),
        graph_types: catalog.all_graph_type_defs(),
        procedures: catalog.all_procedure_defs(),
        schemas: catalog.schema_names(),
        graph_type_bindings: catalog.all_graph_type_bindings(),
    }
}

/// Restores indexes from snapshot metadata by rebuilding them from existing data.
///
/// Must be called after all nodes/edges have been populated, since index
/// creation scans existing data.
fn restore_indexes_from_snapshot(db: &super::GrafeoDB, indexes: &SnapshotIndexes) {
    for name in &indexes.property_indexes {
        db.lpg_store().create_property_index(name);
    }

    #[cfg(feature = "vector-index")]
    for vi in &indexes.vector_indexes {
        if let Err(err) = db.create_vector_index(
            &vi.label,
            &vi.property,
            Some(vi.dimensions),
            Some(vi.metric.name()),
            Some(vi.m),
            Some(vi.ef_construction),
            None,
        ) {
            grafeo_warn!(
                "Failed to restore vector index :{label}({property}): {err}",
                label = vi.label,
                property = vi.property,
            );
        }
    }

    #[cfg(feature = "text-index")]
    for ti in &indexes.text_indexes {
        if let Err(err) = db.create_text_index(&ti.label, &ti.property) {
            grafeo_warn!(
                "Failed to restore text index :{label}({property}): {err}",
                label = ti.label,
                property = ti.property,
            );
        }
    }
}

/// Decode a snapshot blob into the in-memory `Snapshot` struct after
/// validating the leading version byte. Shared by `import_snapshot` and
/// `open_multi` so the two paths can't drift.
fn decode_snapshot_bytes(data: &[u8]) -> Result<Snapshot> {
    if data.is_empty() {
        return Err(Error::Internal("empty snapshot data".to_string()));
    }

    let version = data[0];
    if version != SNAPSHOT_VERSION {
        return Err(Error::Internal(format!(
            "unsupported snapshot version: {version} (expected {SNAPSHOT_VERSION})"
        )));
    }

    let config = bincode::config::standard();
    let (snapshot, _): (Snapshot, _) = bincode::serde::decode_from_slice(data, config)
        .map_err(|e| Error::Internal(format!("snapshot decode failed: {e}")))?;

    Ok(snapshot)
}

/// Collects index metadata from a store into snapshot format.
fn collect_index_metadata(store: &grafeo_core::graph::lpg::LpgStore) -> SnapshotIndexes {
    let property_indexes = store.property_index_keys();

    #[cfg(feature = "vector-index")]
    let vector_indexes: Vec<SnapshotVectorIndex> = store
        .vector_index_entries()
        .into_iter()
        .filter_map(|(key, index)| {
            let (label, property) = key.split_once(':')?;
            let config = index.config();
            Some(SnapshotVectorIndex {
                label: label.to_string(),
                property: property.to_string(),
                dimensions: config.dimensions,
                metric: config.metric,
                m: config.m,
                ef_construction: config.ef_construction,
            })
        })
        .collect();
    #[cfg(not(feature = "vector-index"))]
    let vector_indexes = Vec::new();

    #[cfg(feature = "text-index")]
    let text_indexes: Vec<SnapshotTextIndex> = store
        .text_index_entries()
        .into_iter()
        .filter_map(|(key, _)| {
            let (label, property) = key.split_once(':')?;
            Some(SnapshotTextIndex {
                label: label.to_string(),
                property: property.to_string(),
            })
        })
        .collect();
    #[cfg(not(feature = "text-index"))]
    let text_indexes = Vec::new();

    SnapshotIndexes {
        property_indexes,
        vector_indexes,
        text_indexes,
    }
}

/// Unions index metadata across multiple snapshots. Deduplicates by
/// the natural key (property name for property indexes, label+property
/// for vector and text indexes). The first occurrence of a given key
/// wins on configuration values (dimensions, metric, m, ef_construction
/// for vector indexes) so the lead snapshot drives index parameters —
/// callers needing identical configs across chunks should produce them
/// from the same source.
fn union_index_metadata(snapshots: &[Snapshot]) -> SnapshotIndexes {
    let mut property_index_set: HashSet<String> = HashSet::new();
    let mut property_indexes = Vec::new();
    for snap in snapshots {
        for name in &snap.indexes.property_indexes {
            if property_index_set.insert(name.clone()) {
                property_indexes.push(name.clone());
            }
        }
    }

    #[cfg(feature = "vector-index")]
    let mut vector_keys: HashSet<(String, String)> = HashSet::new();
    #[cfg(feature = "vector-index")]
    let mut vector_indexes = Vec::new();
    #[cfg(feature = "vector-index")]
    for snap in snapshots {
        for vi in &snap.indexes.vector_indexes {
            if vector_keys.insert((vi.label.clone(), vi.property.clone())) {
                vector_indexes.push(SnapshotVectorIndex {
                    label: vi.label.clone(),
                    property: vi.property.clone(),
                    dimensions: vi.dimensions,
                    metric: vi.metric,
                    m: vi.m,
                    ef_construction: vi.ef_construction,
                });
            }
        }
    }
    #[cfg(not(feature = "vector-index"))]
    let vector_indexes = Vec::new();

    #[cfg(feature = "text-index")]
    let mut text_keys: HashSet<(String, String)> = HashSet::new();
    #[cfg(feature = "text-index")]
    let mut text_indexes = Vec::new();
    #[cfg(feature = "text-index")]
    for snap in snapshots {
        for ti in &snap.indexes.text_indexes {
            if text_keys.insert((ti.label.clone(), ti.property.clone())) {
                text_indexes.push(SnapshotTextIndex {
                    label: ti.label.clone(),
                    property: ti.property.clone(),
                });
            }
        }
    }
    #[cfg(not(feature = "text-index"))]
    let text_indexes = Vec::new();

    SnapshotIndexes {
        property_indexes,
        vector_indexes,
        text_indexes,
    }
}

impl super::GrafeoDB {
    // =========================================================================
    // ADMIN API: Persistence Control
    // =========================================================================

    /// Saves the database to a file path.
    ///
    /// - If the path ends in `.grafeo`: creates a single-file database
    /// - Otherwise: creates a WAL directory-backed database at the path
    /// - If in-memory: creates a new persistent database at path
    /// - If file-backed: creates a copy at the new path
    ///
    /// The original database remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the save operation fails.
    ///
    /// Requires the `wal` feature for persistence support.
    #[cfg(feature = "wal")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Single-file format: export snapshot directly to a .grafeo file
        #[cfg(feature = "grafeo-file")]
        if path.extension().is_some_and(|ext| ext == "grafeo") {
            return self.save_as_grafeo_file(path);
        }

        // Create target database with WAL enabled
        let target_config = Config::persistent(path);
        let target = Self::with_config(target_config)?;

        // Copy all nodes using WAL-enabled methods
        for node in self.lpg_store().all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target
                .lpg_store()
                .create_node_with_id(node.id, &label_refs)?;

            // Log to WAL
            target.log_wal(&WalRecord::CreateNode {
                id: node.id,
                labels: node.labels.iter().map(|s| s.to_string()).collect(),
            })?;

            // Copy properties
            for (key, value) in node.properties {
                target
                    .lpg_store()
                    .set_node_property(node.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetNodeProperty {
                    id: node.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Copy all edges using WAL-enabled methods
        for edge in self.lpg_store().all_edges() {
            target
                .lpg_store()
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;

            // Log to WAL
            target.log_wal(&WalRecord::CreateEdge {
                id: edge.id,
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type.to_string(),
            })?;

            // Copy properties
            for (key, value) in edge.properties {
                target
                    .lpg_store()
                    .set_edge_property(edge.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetEdgeProperty {
                    id: edge.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Copy named graphs
        for graph_name in self.lpg_store().graph_names() {
            if let Some(src_graph) = self.lpg_store().graph(&graph_name) {
                target.log_wal(&WalRecord::CreateNamedGraph {
                    name: graph_name.clone(),
                })?;
                target
                    .lpg_store()
                    .create_graph(&graph_name)
                    .map_err(|e| Error::Internal(e.to_string()))?;

                if let Some(dst_graph) = target.lpg_store().graph(&graph_name) {
                    // Switch WAL context to this named graph
                    target.log_wal(&WalRecord::SwitchGraph {
                        name: Some(graph_name.clone()),
                    })?;

                    for node in src_graph.all_nodes() {
                        let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
                        dst_graph.create_node_with_id(node.id, &label_refs)?;
                        target.log_wal(&WalRecord::CreateNode {
                            id: node.id,
                            labels: node.labels.iter().map(|s| s.to_string()).collect(),
                        })?;
                        for (key, value) in node.properties {
                            dst_graph.set_node_property(node.id, key.as_str(), value.clone());
                            target.log_wal(&WalRecord::SetNodeProperty {
                                id: node.id,
                                key: key.to_string(),
                                value,
                            })?;
                        }
                    }
                    for edge in src_graph.all_edges() {
                        dst_graph.create_edge_with_id(
                            edge.id,
                            edge.src,
                            edge.dst,
                            &edge.edge_type,
                        )?;
                        target.log_wal(&WalRecord::CreateEdge {
                            id: edge.id,
                            src: edge.src,
                            dst: edge.dst,
                            edge_type: edge.edge_type.to_string(),
                        })?;
                        for (key, value) in edge.properties {
                            dst_graph.set_edge_property(edge.id, key.as_str(), value.clone());
                            target.log_wal(&WalRecord::SetEdgeProperty {
                                id: edge.id,
                                key: key.to_string(),
                                value,
                            })?;
                        }
                    }
                }
            }
        }

        // Switch WAL context back to default graph
        if !self.lpg_store().graph_names().is_empty() {
            target.log_wal(&WalRecord::SwitchGraph { name: None })?;
        }

        // Copy RDF data with WAL logging
        #[cfg(feature = "triple-store")]
        {
            for triple in self.rdf_store.triples() {
                let record = WalRecord::InsertRdfTriple {
                    subject: triple.subject().to_string(),
                    predicate: triple.predicate().to_string(),
                    object: triple.object().to_string(),
                    graph: None,
                };
                target.rdf_store.insert((*triple).clone());
                target.log_wal(&record)?;
            }
            for name in self.rdf_store.graph_names() {
                target.log_wal(&WalRecord::CreateRdfGraph { name: name.clone() })?;
                if let Some(src_graph) = self.rdf_store.graph(&name) {
                    let dst_graph = target.rdf_store.graph_or_create(&name);
                    for triple in src_graph.triples() {
                        let record = WalRecord::InsertRdfTriple {
                            subject: triple.subject().to_string(),
                            predicate: triple.predicate().to_string(),
                            object: triple.object().to_string(),
                            graph: Some(name.clone()),
                        };
                        dst_graph.insert((*triple).clone());
                        target.log_wal(&record)?;
                    }
                }
            }
        }

        // Checkpoint and close the target database
        target.close()?;

        Ok(())
    }

    /// Creates an in-memory copy of this database.
    ///
    /// Returns a new database that is completely independent, including
    /// all named graph data.
    /// Useful for:
    /// Saves the database to a single `.grafeo` file.
    #[cfg(feature = "grafeo-file")]
    fn save_as_grafeo_file(&self, path: &Path) -> Result<()> {
        use grafeo_storage::file::GrafeoFileManager;

        let snapshot_data = self.export_snapshot()?;
        let epoch = self.lpg_store().current_epoch();
        let transaction_id = self
            .transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0);
        let node_count = self.lpg_store().node_count() as u64;
        let edge_count = self.lpg_store().edge_count() as u64;

        let fm = GrafeoFileManager::create(path)?;
        fm.write_snapshot(
            &snapshot_data,
            epoch.0,
            transaction_id,
            node_count,
            edge_count,
        )?;
        Ok(())
    }

    /// - Testing modifications without affecting the original
    /// - Faster operations when persistence isn't needed
    ///
    /// # Errors
    ///
    /// Returns an error if the copy operation fails.
    pub fn to_memory(&self) -> Result<Self> {
        let config = Config::in_memory();
        let target = Self::with_config(config)?;

        // Copy default graph nodes
        for node in self.lpg_store().all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target
                .lpg_store()
                .create_node_with_id(node.id, &label_refs)?;
            for (key, value) in node.properties {
                target
                    .lpg_store()
                    .set_node_property(node.id, key.as_str(), value);
            }
        }

        // Copy default graph edges
        for edge in self.lpg_store().all_edges() {
            target
                .lpg_store()
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
            for (key, value) in edge.properties {
                target
                    .lpg_store()
                    .set_edge_property(edge.id, key.as_str(), value);
            }
        }

        // Copy named graphs
        for graph_name in self.lpg_store().graph_names() {
            if let Some(src_graph) = self.lpg_store().graph(&graph_name) {
                target
                    .lpg_store()
                    .create_graph(&graph_name)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                if let Some(dst_graph) = target.lpg_store().graph(&graph_name) {
                    for node in src_graph.all_nodes() {
                        let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
                        dst_graph.create_node_with_id(node.id, &label_refs)?;
                        for (key, value) in node.properties {
                            dst_graph.set_node_property(node.id, key.as_str(), value);
                        }
                    }
                    for edge in src_graph.all_edges() {
                        dst_graph.create_edge_with_id(
                            edge.id,
                            edge.src,
                            edge.dst,
                            &edge.edge_type,
                        )?;
                        for (key, value) in edge.properties {
                            dst_graph.set_edge_property(edge.id, key.as_str(), value);
                        }
                    }
                }
            }
        }

        // Copy RDF data
        #[cfg(feature = "triple-store")]
        {
            for triple in self.rdf_store.triples() {
                target.rdf_store.insert((*triple).clone());
            }
            for name in self.rdf_store.graph_names() {
                if let Some(src_graph) = self.rdf_store.graph(&name) {
                    let dst_graph = target.rdf_store.graph_or_create(&name);
                    for triple in src_graph.triples() {
                        dst_graph.insert((*triple).clone());
                    }
                }
            }
        }

        Ok(target)
    }

    /// Opens a database file and loads it entirely into memory.
    ///
    /// The returned database has no connection to the original file.
    /// Changes will NOT be written back to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be opened or loaded.
    #[cfg(feature = "wal")]
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self> {
        // Open the source database (triggers WAL recovery)
        let source = Self::open(path)?;

        // Create in-memory copy
        let target = source.to_memory()?;

        // Close the source (releases file handles)
        source.close()?;

        Ok(target)
    }

    // =========================================================================
    // ADMIN API: Subgraph Extraction
    // =========================================================================

    /// Produces a new in-memory database containing exactly the
    /// requested nodes plus every edge whose SOURCE is in `node_ids`,
    /// regardless of where the destination lives. NodeIds and EdgeIds
    /// are preserved verbatim so multiple sibling extracts can later
    /// be merged via [`open_multi`](Self::open_multi) without ID
    /// collisions.
    ///
    /// Edges: every edge whose SOURCE is in `node_ids` is carried,
    /// regardless of where its destination lives. The result is that
    /// each edge in the source is "owned" by exactly one extract — the
    /// one containing its source node. Merging sibling extracts of a
    /// partition via [`open_multi`](Self::open_multi) restores the
    /// source's edge set exactly. An extract may carry edges with
    /// dangling dst NodeIds in isolation; this is intentional and only
    /// valid when consumed via `open_multi` (which validates endpoints
    /// across the merged union), not via `import_snapshot` (which
    /// demands per-snapshot endpoint resolution).
    ///
    /// Properties (full temporal history when the `temporal` feature
    /// is on), the full schema catalog, and index metadata are copied
    /// from the source. Index *data* is rebuilt over the extracted
    /// nodes/edges; index parameters (vector dimensions, metric, m,
    /// ef_construction) carry verbatim.
    ///
    /// # Errors
    ///
    /// Returns an error if any `node_ids` entry does not exist in
    /// `self`, or if copy operations fail.
    pub fn extract_subgraph(&self, node_ids: &[NodeId]) -> Result<Self> {
        let store = self.lpg_store();

        // Dedup the request set — caller convenience; duplicates here
        // would otherwise cause `create_node_with_id` to error on the
        // second insert.
        let requested: HashSet<NodeId> = node_ids.iter().copied().collect();

        // Validate up front so a missing ID surfaces before any work.
        for &id in &requested {
            if store.get_node(id).is_none() {
                return Err(Error::Internal(format!(
                    "extract_subgraph: NodeId {id} does not exist in source database"
                )));
            }
        }

        let target = Self::new_in_memory();
        let target_store = target.lpg_store();

        // Copy nodes preserving IDs + labels + properties. Property
        // history is restored under `temporal`; otherwise just the
        // current value.
        for &id in &requested {
            let Some(node) = store.get_node(id) else { continue };
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target_store
                .create_node_with_id(id, &label_refs)
                .map_err(|e| Error::Internal(format!("extract_subgraph: create node {id}: {e}")))?;

            #[cfg(feature = "temporal")]
            {
                for (key, entries) in store.node_property_history(id) {
                    for (epoch, value) in entries {
                        target_store.set_node_property_at_epoch(id, key.as_str(), value, epoch);
                    }
                }
            }
            #[cfg(not(feature = "temporal"))]
            {
                for (key, value) in node.properties {
                    target_store.set_node_property(id, key.as_str(), value);
                }
            }
        }

        // Copy outgoing edges. For each requested node, walk every
        // outgoing edge and carry it into the target — including
        // edges whose destination is OUTSIDE the request set. This
        // is the "source-side ownership" semantic: each edge is owned
        // by the extract that contains its source node. When sibling
        // extracts of a partition are merged via `open_multi`, every
        // edge surfaces in exactly one extract and the union restores
        // the source edge set verbatim. Cross-snapshot endpoint
        // validation in `open_multi` confirms every dst exists
        // somewhere in the union.
        //
        // An individual extract may therefore contain edges with
        // dangling dst NodeIds in isolation. This is intentional;
        // extracts are transport artifacts for `open_multi`, not
        // standalone databases. `import_snapshot` would reject such
        // an extract (its per-snapshot validator demands endpoint
        // resolution); `open_multi` does not.
        //
        // Each edge surfaces exactly once because the outer loop
        // iterates the requested set as a `HashSet<NodeId>` and
        // `Direction::Outgoing` consults only the forward adjacency
        // list.
        for &src_id in &requested {
            for (_dst_id, edge_id) in
                store.edges_from(src_id, grafeo_core::graph::Direction::Outgoing)
            {
                let Some(edge) = store.get_edge(edge_id) else { continue };
                target_store
                    .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)
                    .map_err(|e| {
                        Error::Internal(format!(
                            "extract_subgraph: create edge {}: {e}",
                            edge.id
                        ))
                    })?;

                #[cfg(feature = "temporal")]
                {
                    for (key, entries) in store.edge_property_history(edge.id) {
                        for (epoch, value) in entries {
                            target_store.set_edge_property_at_epoch(
                                edge.id,
                                key.as_str(),
                                value,
                                epoch,
                            );
                        }
                    }
                }
                #[cfg(not(feature = "temporal"))]
                {
                    for (key, value) in edge.properties {
                        target_store.set_edge_property(edge.id, key.as_str(), value);
                    }
                }
            }
        }

        // Sync epoch so subsequent writes against the extract use the
        // same logical clock as the source.
        #[cfg(feature = "temporal")]
        {
            let epoch = self.transaction_manager.current_epoch();
            target_store.sync_epoch(epoch);
            target.transaction_manager.sync_epoch(epoch);
        }

        // Carry the full source schema catalog so the extract
        // round-trips with the source's type system intact. We copy
        // the full schema verbatim (not a subset filtered to the
        // requested labels) because the merge-time schema policy
        // already handles unionable schemas, and stripping types here
        // would prevent the natural "extract many, merge into one"
        // workflow.
        let schema = collect_schema(&self.catalog);
        restore_schema_from_snapshot(&target_store, &target.catalog, &schema);

        // Carry index metadata. Index *data* is rebuilt over the
        // populated nodes/edges by `restore_indexes_from_snapshot`.
        let indexes = collect_index_metadata(&store);
        restore_indexes_from_snapshot(&target, &indexes);

        Ok(target)
    }

    // =========================================================================
    // ADMIN API: Snapshot Export/Import
    // =========================================================================

    /// Exports the entire database to a binary snapshot.
    ///
    /// The returned bytes can be stored (e.g. in IndexedDB) and later
    /// restored with [`import_snapshot()`](Self::import_snapshot).
    /// Includes all named graph data.
    ///
    /// Properties are stored as version-history lists. When `temporal` is
    /// enabled, the full history is captured. Otherwise, each property is
    /// wrapped as a single-entry list at epoch 0.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export_snapshot(&self) -> Result<Vec<u8>> {
        let nodes = collect_snapshot_nodes(self.lpg_store());
        let edges = collect_snapshot_edges(self.lpg_store());

        // Collect named graphs
        let named_graphs: Vec<NamedGraphSnapshot> = self
            .lpg_store()
            .graph_names()
            .into_iter()
            .filter_map(|name| {
                self.lpg_store()
                    .graph(&name)
                    .map(|graph_store| NamedGraphSnapshot {
                        name,
                        nodes: collect_snapshot_nodes(&graph_store),
                        edges: collect_snapshot_edges(&graph_store),
                    })
            })
            .collect();

        // Collect RDF triples
        #[cfg(feature = "triple-store")]
        let rdf_triples = collect_rdf_triples(&self.rdf_store);
        #[cfg(not(feature = "triple-store"))]
        let rdf_triples = Vec::new();

        #[cfg(feature = "triple-store")]
        let rdf_named_graphs: Vec<RdfNamedGraphSnapshot> = self
            .rdf_store
            .graph_names()
            .into_iter()
            .filter_map(|name| {
                self.rdf_store
                    .graph(&name)
                    .map(|graph| RdfNamedGraphSnapshot {
                        name,
                        triples: collect_rdf_triples(&graph),
                    })
            })
            .collect();
        #[cfg(not(feature = "triple-store"))]
        let rdf_named_graphs = Vec::new();

        let schema = collect_schema(&self.catalog);
        let indexes = collect_index_metadata(self.lpg_store());

        let snapshot = Snapshot {
            version: SNAPSHOT_VERSION,
            nodes,
            edges,
            named_graphs,
            rdf_triples,
            rdf_named_graphs,
            schema,
            indexes,
            #[cfg(feature = "temporal")]
            epoch: self.transaction_manager.current_epoch().as_u64(),
            #[cfg(not(feature = "temporal"))]
            epoch: 0,
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("snapshot export failed: {e}")))
    }

    /// Creates a new in-memory database from a binary snapshot.
    ///
    /// The `data` must have been produced by [`export_snapshot()`](Self::export_snapshot).
    ///
    /// All edge references are validated before any data is inserted: every
    /// edge's source and destination must reference a node present in the
    /// snapshot, and duplicate node/edge IDs are rejected. If validation
    /// fails, no database is created.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, or deserialization fails.
    pub fn import_snapshot(data: &[u8]) -> Result<Self> {
        let snapshot = decode_snapshot_bytes(data)?;

        // Validate default graph data
        validate_snapshot_data(&snapshot.nodes, &snapshot.edges)?;

        // Validate each named graph
        for ng in &snapshot.named_graphs {
            validate_snapshot_data(&ng.nodes, &ng.edges)?;
        }

        let db = Self::new_in_memory();
        populate_store_from_snapshot(db.lpg_store(), snapshot.nodes, snapshot.edges)?;

        // Restore epoch from snapshot
        #[cfg(feature = "temporal")]
        {
            let epoch = EpochId::new(snapshot.epoch);
            db.lpg_store().sync_epoch(epoch);
            db.transaction_manager.sync_epoch(epoch);
        }

        // Capture epoch before moving snapshot fields
        #[cfg(feature = "temporal")]
        let snapshot_epoch = EpochId::new(snapshot.epoch);

        // Restore named graphs
        for ng in snapshot.named_graphs {
            db.lpg_store()
                .create_graph(&ng.name)
                .map_err(|e| Error::Internal(e.to_string()))?;
            if let Some(graph_store) = db.lpg_store().graph(&ng.name) {
                populate_store_from_snapshot(&graph_store, ng.nodes, ng.edges)?;
                // Named graph stores need the same epoch so temporal property
                // lookups via current_epoch() return the correct values.
                #[cfg(feature = "temporal")]
                graph_store.sync_epoch(snapshot_epoch);
            }
        }

        // Restore RDF triples
        #[cfg(feature = "triple-store")]
        {
            populate_rdf_store(&db.rdf_store, &snapshot.rdf_triples);
            for rng in &snapshot.rdf_named_graphs {
                let graph = db.rdf_store.graph_or_create(&rng.name);
                populate_rdf_store(&graph, &rng.triples);
            }
        }

        // Restore schema
        restore_schema_from_snapshot(db.lpg_store(), &db.catalog, &snapshot.schema);

        // Restore indexes (must come after data population)
        restore_indexes_from_snapshot(&db, &snapshot.indexes);

        Ok(db)
    }

    /// Creates a new in-memory database by merging multiple binary snapshots.
    ///
    /// Each blob in `snapshots` must have been produced by
    /// [`export_snapshot()`](Self::export_snapshot). Snapshots are unioned
    /// into one database:
    /// - Nodes and edges are preserved with their producer-allocated IDs.
    /// - A NodeId (or EdgeId) appearing in two snapshots is rejected as a
    ///   producer bug; the caller is responsible for emitting disjoint
    ///   subsets. (Cross-snapshot dedup validation is not yet enforced;
    ///   see Task 5 of the open_multi plan.)
    /// - Every edge endpoint must exist somewhere in the union, so a chunk
    ///   may carry edges whose endpoints belong to a different chunk.
    ///   (Endpoint resolution across the union is not yet enforced; see
    ///   Task 5–7 of the open_multi plan.)
    /// - All snapshots must declare the same schema (after canonical
    ///   ordering). Differing schemas are rejected. (Schema-equality
    ///   check is not yet enforced; see Task 10 of the open_multi plan.)
    /// - Indexes are unioned; the epoch is the maximum across inputs.
    ///   (Currently restored from the first snapshot only; index union
    ///   lands in Task 12 of the open_multi plan.)
    /// - At most one snapshot may carry named graphs or RDF triples.
    ///   (Currently rejected outright; full single-owner policy lands in
    ///   Task 11 of the open_multi plan.)
    ///
    /// All validation runs before any data is inserted; a rejection
    /// leaves no partial database behind (the function never publishes
    /// `self`).
    ///
    /// # Errors
    ///
    /// Returns an error if `snapshots` is empty, any blob fails to
    /// decode, any cross-snapshot validation fails, or population fails.
    ///
    /// # Panics
    ///
    /// Does not panic in practice. The internal `.expect()` that resolves
    /// the maximum epoch is guarded by the non-empty check at the top of
    /// the function; it is unreachable when the function is called correctly.
    pub fn open_multi(snapshots: &[&[u8]]) -> Result<Self> {
        Self::open_multi_with(snapshots, OpenMultiOptions::default())
    }

    /// Variant of [`open_multi`](Self::open_multi) that takes an
    /// explicit [`OpenMultiOptions`] for callers who need a non-default
    /// schema-merge policy.
    ///
    /// # Errors
    ///
    /// Same conditions as [`open_multi`](Self::open_multi).
    pub fn open_multi_with(snapshots: &[&[u8]], options: OpenMultiOptions) -> Result<Self> {
        if snapshots.is_empty() {
            return Err(Error::Internal(
                "open_multi requires at least one snapshot blob".to_string(),
            ));
        }

        let _span = grafeo_info_span!("open_multi", n_snapshots = snapshots.len());

        // Decode every blob first so any version / bincode failure
        // surfaces before we touch a target database.
        let decoded: Vec<Snapshot> = {
            let _decode = grafeo_debug_span!("decode");
            snapshots
                .iter()
                .enumerate()
                .map(|(idx, bytes)| {
                    decode_snapshot_bytes(bytes)
                        .map_err(|e| Error::Internal(format!("snapshot[{idx}]: {e}")))
                })
                .collect::<Result<_>>()?
        };

        // Per-snapshot duplicate-ID validation only (not endpoint
        // resolution — niche chunks can reference nodes from other
        // snapshots). Cross-snapshot endpoint validation runs next via
        // validate_snapshot_set.
        {
            let _validate = grafeo_debug_span!("validate");
            for (idx, snap) in decoded.iter().enumerate() {
                validate_snapshot_ids(&snap.nodes, &snap.edges)
                    .map_err(|e| Error::Internal(format!("snapshot[{idx}]: {e}")))?;
            }
            validate_snapshot_set(&decoded)?;
        }

        // Reconcile schemas per the chosen policy. The merged schema
        // is what gets restored to the target catalog below.
        let merged_schema = {
            let _schema = grafeo_debug_span!("schema");
            merge_snapshot_schemas(&decoded, options.schema_policy)?
        };

        // Build the merged database from the decoded snapshots.
        let db = Self::new_in_memory();

        // Default graph: union of all snapshots' (nodes, edges).
        {
            let _populate = grafeo_debug_span!("populate");
            for snap in &decoded {
                populate_store_from_snapshot_ref(db.lpg_store(), &snap.nodes, &snap.edges)?;
            }
        }

        // Named graphs: each name may live in exactly one snapshot.
        // A name appearing in two snapshots is rejected — the caller
        // must produce disjoint chunk subsets.
        {
            let _named_graphs = grafeo_debug_span!("named_graphs");
            let mut seen_graphs: HashSet<String> = HashSet::new();
            for (idx, snap) in decoded.iter().enumerate() {
                for graph in &snap.named_graphs {
                    if !seen_graphs.insert(graph.name.clone()) {
                        return Err(Error::Internal(format!(
                            "named graph '{}' appears in snapshot[{idx}] and at \
                             least one earlier snapshot; open_multi requires named \
                             graphs to be disjoint across snapshots",
                            graph.name
                        )));
                    }
                    db.lpg_store()
                        .create_graph(&graph.name)
                        .map_err(|e| Error::Internal(e.to_string()))?;
                    if let Some(graph_store) = db.lpg_store().graph(&graph.name) {
                        populate_store_from_snapshot_ref(
                            &graph_store,
                            &graph.nodes,
                            &graph.edges,
                        )?;
                    }
                }
            }

            // RDF: same single-owner policy.
            #[cfg(feature = "triple-store")]
            {
                let owners = decoded
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| !s.rdf_triples.is_empty() || !s.rdf_named_graphs.is_empty())
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>();
                if owners.len() > 1 {
                    return Err(Error::Internal(format!(
                        "RDF data is present in snapshots {owners:?}; open_multi \
                         requires at most one snapshot to carry RDF triples or \
                         named graphs"
                    )));
                }
                if let Some(&idx) = owners.first() {
                    let snap = &decoded[idx];
                    populate_rdf_store(&db.rdf_store, &snap.rdf_triples);
                    for rng in &snap.rdf_named_graphs {
                        let graph = db.rdf_store.graph_or_create(&rng.name);
                        populate_rdf_store(&graph, &rng.triples);
                    }
                }
            }
        }

        // Restore epoch, schema, and indexes.
        {
            let _indexes = grafeo_debug_span!("indexes");

            // Restore epoch as max across snapshots.
            #[cfg(feature = "temporal")]
            {
                let max_epoch = decoded
                    .iter()
                    .map(|s| s.epoch)
                    .max()
                    .expect("decoded is non-empty after empty-input guard");
                let epoch = EpochId::new(max_epoch);
                db.lpg_store().sync_epoch(epoch);
                db.transaction_manager.sync_epoch(epoch);
            }

            // Restore the merged schema (union or strict-equality, per policy).
            restore_schema_from_snapshot(db.lpg_store(), &db.catalog, &merged_schema);

            // Restore indexes from the union of all snapshots' metadata.
            restore_indexes_from_snapshot(&db, &union_index_metadata(&decoded));
        }

        grafeo_info!(
            "open_multi complete: nodes={nodes} edges={edges}",
            nodes = db.node_count(),
            edges = db.edge_count()
        );

        Ok(db)
    }

    /// Replaces the current database contents with data from a binary snapshot.
    ///
    /// The `data` must have been produced by
    /// [`export_snapshot()`](Self::export_snapshot).
    ///
    /// All validation (duplicate IDs, dangling edge references) is performed
    /// before any data is modified. If validation fails, the current database
    /// is left unchanged. If validation passes, the store is cleared and
    /// rebuilt from the snapshot atomically (from the perspective of
    /// subsequent queries).
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, or deserialization fails.
    pub fn restore_snapshot(&self, data: &[u8]) -> Result<()> {
        let snapshot = decode_snapshot_bytes(data)?;

        // Validate all data before making any changes
        validate_snapshot_data(&snapshot.nodes, &snapshot.edges)?;
        for ng in &snapshot.named_graphs {
            validate_snapshot_data(&ng.nodes, &ng.edges)?;
        }

        // Drop all existing named graphs, then clear default store
        for name in self.lpg_store().graph_names() {
            self.lpg_store().drop_graph(&name);
        }
        self.lpg_store().clear();

        populate_store_from_snapshot(self.lpg_store(), snapshot.nodes, snapshot.edges)?;

        // Restore epoch from temporal snapshot
        #[cfg(feature = "temporal")]
        let snapshot_epoch = {
            let epoch = EpochId::new(snapshot.epoch);
            self.lpg_store().sync_epoch(epoch);
            self.transaction_manager.sync_epoch(epoch);
            epoch
        };

        // Restore named graphs
        for ng in snapshot.named_graphs {
            self.lpg_store()
                .create_graph(&ng.name)
                .map_err(|e| Error::Internal(e.to_string()))?;
            if let Some(graph_store) = self.lpg_store().graph(&ng.name) {
                populate_store_from_snapshot(&graph_store, ng.nodes, ng.edges)?;
                #[cfg(feature = "temporal")]
                graph_store.sync_epoch(snapshot_epoch);
            }
        }

        // Restore RDF data
        #[cfg(feature = "triple-store")]
        {
            // Clear existing RDF data
            self.rdf_store.clear();
            for name in self.rdf_store.graph_names() {
                self.rdf_store.drop_graph(&name);
            }
            populate_rdf_store(&self.rdf_store, &snapshot.rdf_triples);
            for rng in &snapshot.rdf_named_graphs {
                let graph = self.rdf_store.graph_or_create(&rng.name);
                populate_rdf_store(&graph, &rng.triples);
            }
        }

        // Restore schema
        restore_schema_from_snapshot(self.lpg_store(), &self.catalog, &snapshot.schema);

        // Restore indexes (must come after data population)
        restore_indexes_from_snapshot(self, &snapshot.indexes);

        Ok(())
    }

    // =========================================================================
    // ADMIN API: Iteration
    // =========================================================================

    /// Returns an iterator over all nodes in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_nodes(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Node> + '_ {
        self.lpg_store().all_nodes()
    }

    /// Returns an iterator over all edges in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_edges(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Edge> + '_ {
        self.lpg_store().all_edges()
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId, Value};

    use super::super::GrafeoDB;
    use super::{
        SNAPSHOT_VERSION, Snapshot, SnapshotEdge, SnapshotIndexes, SnapshotNode, SnapshotSchema,
    };

    #[test]
    fn test_restore_snapshot_basic() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Populate
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        let snapshot = db.export_snapshot().unwrap();

        // Modify
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        assert_eq!(db.lpg_store().node_count(), 3);

        // Restore original
        db.restore_snapshot(&snapshot).unwrap();

        assert_eq!(db.lpg_store().node_count(), 2);
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_restore_snapshot_validation_failure() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Corrupt snapshot: just garbage bytes
        let result = db.restore_snapshot(b"garbage");
        assert!(result.is_err());

        // DB should be unchanged
        assert_eq!(db.lpg_store().node_count(), 1);
    }

    #[test]
    fn test_restore_snapshot_empty_db() {
        let db = GrafeoDB::new_in_memory();

        // Export empty snapshot, then populate, then restore to empty
        let empty_snapshot = db.export_snapshot().unwrap();

        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        assert_eq!(db.lpg_store().node_count(), 1);

        db.restore_snapshot(&empty_snapshot).unwrap();
        assert_eq!(db.lpg_store().node_count(), 0);
    }

    #[test]
    fn test_restore_snapshot_with_edges() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        assert_eq!(db.lpg_store().edge_count(), 1);

        // Modify: add more data
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(db.lpg_store().node_count(), 2);
        assert_eq!(db.lpg_store().edge_count(), 1);
    }

    #[test]
    fn test_restore_snapshot_preserves_sessions() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let snapshot = db.export_snapshot().unwrap();

        // Modify
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();

        // Session should still work and see restored data
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();
        let session2 = db2.session();

        let result = session2.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    // --- to_memory() ---

    #[test]
    fn test_to_memory_empty() {
        let db = GrafeoDB::new_in_memory();
        let copy = db.to_memory().unwrap();
        assert_eq!(copy.lpg_store().node_count(), 0);
        assert_eq!(copy.lpg_store().edge_count(), 0);
    }

    #[test]
    fn test_to_memory_copies_nodes_and_properties() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.lpg_store().node_count(), 2);

        let s2 = copy.session();
        let result = s2
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn test_to_memory_copies_edges_and_properties() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["Person"]);
        db.set_node_property(a, "name", "Alix".into());
        let b = db.create_node(&["Person"]);
        db.set_node_property(b, "name", "Gus".into());
        let edge = db.create_edge(a, b, "KNOWS");
        db.set_edge_property(edge, "since", Value::Int64(2020));

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.lpg_store().node_count(), 2);
        assert_eq!(copy.lpg_store().edge_count(), 1);

        let s2 = copy.session();
        let result = s2.execute("MATCH ()-[e:KNOWS]->() RETURN e.since").unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2020));
    }

    #[test]
    fn test_to_memory_is_independent() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let copy = db.to_memory().unwrap();

        // Mutating original should not affect copy
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        assert_eq!(db.lpg_store().node_count(), 2);
        assert_eq!(copy.lpg_store().node_count(), 1);
    }

    // --- iter_nodes() / iter_edges() ---

    #[test]
    fn test_iter_nodes_empty() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.iter_nodes().count(), 0);
    }

    #[test]
    fn test_iter_nodes_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let id1 = db.create_node(&["Person"]);
        db.set_node_property(id1, "name", "Alix".into());
        let id2 = db.create_node(&["Animal"]);
        db.set_node_property(id2, "name", "Fido".into());

        let nodes: Vec<_> = db.iter_nodes().collect();
        assert_eq!(nodes.len(), 2);

        let names: Vec<_> = nodes
            .iter()
            .filter_map(|n| n.properties.iter().find(|(k, _)| k.as_str() == "name"))
            .map(|(_, v)| v.clone())
            .collect();
        assert!(names.contains(&Value::String("Alix".into())));
        assert!(names.contains(&Value::String("Fido".into())));
    }

    #[test]
    fn test_iter_edges_empty() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.iter_edges().count(), 0);
    }

    #[test]
    fn test_iter_edges_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        let c = db.create_node(&["C"]);
        db.create_edge(a, b, "R1");
        db.create_edge(b, c, "R2");

        let edges: Vec<_> = db.iter_edges().collect();
        assert_eq!(edges.len(), 2);

        let types: Vec<_> = edges.iter().map(|e| e.edge_type.as_ref()).collect();
        assert!(types.contains(&"R1"));
        assert!(types.contains(&"R2"));
    }

    // --- restore_snapshot() validation ---

    fn make_snapshot(version: u8, nodes: Vec<SnapshotNode>, edges: Vec<SnapshotEdge>) -> Vec<u8> {
        let snap = Snapshot {
            version,
            nodes,
            edges,
            named_graphs: vec![],
            rdf_triples: vec![],
            rdf_named_graphs: vec![],
            schema: SnapshotSchema::default(),
            indexes: SnapshotIndexes::default(),
            epoch: 0,
        };
        bincode::serde::encode_to_vec(&snap, bincode::config::standard()).unwrap()
    }

    #[test]
    fn test_restore_rejects_unsupported_version() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let bytes = make_snapshot(99, vec![], vec![]);

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unsupported snapshot version"), "got: {err}");

        // DB unchanged
        assert_eq!(db.lpg_store().node_count(), 1);
    }

    #[test]
    fn test_restore_rejects_duplicate_node_ids() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec!["A".into()],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec!["B".into()],
                    properties: vec![],
                },
            ],
            vec![],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate node ID"), "got: {err}");
        assert_eq!(db.lpg_store().node_count(), 1);
    }

    #[test]
    fn test_restore_rejects_duplicate_edge_ids() {
        let db = GrafeoDB::new_in_memory();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec![],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(1),
                    labels: vec![],
                    properties: vec![],
                },
            ],
            vec![
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    properties: vec![],
                },
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    properties: vec![],
                },
            ],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate edge ID"), "got: {err}");
    }

    #[test]
    fn test_restore_rejects_dangling_source() {
        let db = GrafeoDB::new_in_memory();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![SnapshotNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            }],
            vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(999),
                dst: NodeId::new(0),
                edge_type: "REL".into(),
                properties: vec![],
            }],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("non-existent source node"), "got: {err}");
    }

    #[test]
    fn test_restore_rejects_dangling_destination() {
        let db = GrafeoDB::new_in_memory();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![SnapshotNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            }],
            vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(0),
                dst: NodeId::new(999),
                edge_type: "REL".into(),
                properties: vec![],
            }],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("non-existent destination node"), "got: {err}");
    }

    // --- index metadata roundtrip ---

    #[test]
    fn test_snapshot_roundtrip_property_index() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', email: 'alix@example.com'})")
            .unwrap();
        db.create_property_index("email");
        assert!(db.has_property_index("email"));

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();

        assert!(db2.has_property_index("email"));

        // Verify the index actually works for O(1) lookups
        let found = db2.find_nodes_by_property("email", &Value::String("alix@example.com".into()));
        assert_eq!(found.len(), 1);
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn test_snapshot_roundtrip_vector_index() {
        use std::sync::Arc;

        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]);
        db.set_node_property(
            n1,
            "embedding",
            Value::Vector(Arc::from([1.0_f32, 0.0, 0.0])),
        );
        let n2 = db.create_node(&["Doc"]);
        db.set_node_property(
            n2,
            "embedding",
            Value::Vector(Arc::from([0.0_f32, 1.0, 0.0])),
        );

        db.create_vector_index(
            "Doc",
            "embedding",
            None,
            Some("cosine"),
            Some(4),
            Some(32),
            None,
        )
        .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();

        // Vector search should work on the restored database
        let results = db2
            .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 2, None, None)
            .unwrap();
        assert_eq!(results.len(), 2);
        // Closest to [1,0,0] should be n1
        assert_eq!(results[0].0, n1);
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn test_snapshot_roundtrip_text_index() {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Article"]);
        db.set_node_property(n1, "body", Value::String("rust graph database".into()));
        let n2 = db.create_node(&["Article"]);
        db.set_node_property(n2, "body", Value::String("python web framework".into()));

        db.create_text_index("Article", "body").unwrap();

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();

        // Text search should work on the restored database
        let results = db2
            .text_search("Article", "body", "graph database", 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, n1);
    }

    #[test]
    fn test_snapshot_roundtrip_property_index_via_restore() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', email: 'alix@example.com'})")
            .unwrap();
        db.create_property_index("email");

        let snapshot = db.export_snapshot().unwrap();

        // Mutate the database
        session
            .execute("INSERT (:Person {name: 'Gus', email: 'gus@example.com'})")
            .unwrap();
        db.drop_property_index("email");
        assert!(!db.has_property_index("email"));

        // Restore should bring back the index
        db.restore_snapshot(&snapshot).unwrap();
        assert!(db.has_property_index("email"));
    }
}
