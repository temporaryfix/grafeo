//! Persistence, snapshots, and data export for GrafeoDB.

#[cfg(feature = "wal")]
use std::path::Path;

#[cfg(feature = "lpg")]
use grafeo_common::types::HistoryCompleteness;
use grafeo_common::types::{
    EdgeId, EpochId, GraphPath, NodeId, SnapshotArtifact, StoreId, Value, WorldIdentityMetadataV1,
};
use grafeo_common::utils::error::{Error, Result};
#[cfg(feature = "lpg")]
use grafeo_common::{grafeo_debug_span, grafeo_info, grafeo_info_span};
use grafeo_core::graph::lpg::exact_history::{
    epoch_is_inside_lifetime, lifetime_is_covered_by,
    validate_lifetimes as validate_structural_lifetimes,
};
use hashbrown::{HashMap, HashSet};

use crate::config::{Config, GraphModel};

use crate::catalog::{CatalogRead, CatalogWorkspace};
#[cfg(feature = "lpg")]
use crate::catalog::{
    EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition, ProcedureDefinition,
};

/// Current-only recursive portable snapshot with exact graph and index state.
const SNAPSHOT_VERSION: u8 = 12;

#[cfg(feature = "lpg")]
#[path = "live_restore_indexes.rs"]
mod live_restore_indexes;
#[cfg(all(test, feature = "lpg"))]
#[path = "live_restore_tests.rs"]
mod live_restore_tests;
#[cfg(all(test, feature = "lpg"))]
#[path = "save_exactness_tests.rs"]
mod save_exactness_tests;
#[path = "persistence_indexes.rs"]
mod snapshot_indexes;

#[cfg(feature = "lpg")]
enum LiveRestoreError {
    Input(Error),
    Rebind(grafeo_core::graph::lpg::DataRebindError),
}

#[cfg(feature = "lpg")]
impl From<Error> for LiveRestoreError {
    fn from(error: Error) -> Self {
        Self::Input(error)
    }
}

#[cfg(feature = "lpg")]
impl From<grafeo_core::graph::lpg::DataRebindError> for LiveRestoreError {
    fn from(error: grafeo_core::graph::lpg::DataRebindError) -> Self {
        Self::Rebind(error)
    }
}

#[cfg(feature = "lpg")]
impl LiveRestoreError {
    fn into_error(self) -> Error {
        match self {
            Self::Input(error) => error,
            Self::Rebind(error) => error.into_error(),
        }
    }
}
#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
use snapshot_indexes::install_snapshot_catalog_and_indexes_into;
#[cfg(feature = "lpg")]
use snapshot_indexes::{
    capture_snapshot_indexes, install_rebuilt_snapshot_catalog,
    install_snapshot_catalog_and_indexes, merge_snapshot_catalogs, merge_snapshot_indexes,
    merge_snapshot_schemas, stage_snapshot_indexes, validate_snapshot_indexes_for_rebuild,
};
use snapshot_indexes::{snapshot_world_descriptor, stage_snapshot_catalog};

/// Maximum encoded size accepted by the in-memory portable snapshot API.
///
/// Portable snapshots are decoded as one owned object graph, so an explicit
/// envelope budget is part of the trust boundary. The one-GiB ceiling leaves
/// room for the separately bounded 512-MiB RDF history payload while rejecting
/// inputs that cannot be a practical in-memory interchange artifact. Larger
/// databases must use the section-based container save path.
const MAX_PORTABLE_SNAPSHOT_BYTES: usize = 1024 * 1024 * 1024;

fn validate_snapshot_size(size: usize) -> Result<()> {
    if size > MAX_PORTABLE_SNAPSHOT_BYTES {
        return Err(Error::Serialization(format!(
            "portable snapshot has {size} bytes; maximum is {MAX_PORTABLE_SNAPSHOT_BYTES}"
        )));
    }
    Ok(())
}

/// How `open_multi` should reconcile schema catalogs across snapshots.
///
/// The default ([`SchemaMergePolicy::UnionWithConflictCheck`]) matches
/// the natural pattern of "shared chunk has shared DDL, niche chunk
/// has niche-specific DDL, no overlap on type names." For callers who
/// genuinely need byte-equal schemas across all chunks (e.g. testing
/// snapshot determinism), set [`SchemaMergePolicy::StrictEquality`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg(feature = "lpg")]
pub enum SchemaMergePolicy {
    /// Union types across all snapshots. Same-name types must have
    /// matching definitions (compared via canonical bincode bytes);
    /// differently-named types are accumulated into the merged
    /// catalog. Sibling extracts retain their source's full schema;
    /// independently defined compatible schemas can also be united.
    #[default]
    UnionWithConflictCheck,

    /// All snapshots must declare identical schemas (after canonical
    /// ordering of inner Vecs). Stricter than `UnionWithConflictCheck`
    /// — useful for tests that want to detect any schema drift, not
    /// just incompatible drift.
    StrictEquality,
}

/// How a snapshot union constructs its physical search indexes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg(feature = "lpg")]
pub enum IndexMergePolicy {
    /// Preserve physical images; shared owners require identical images.
    #[default]
    ExactState,
    /// Validate every input image, then rebuild from the union's retained data.
    /// Useful for independently scored subsets; this does not assert provenance.
    RebuildFromUnion,
}

/// Configuration for [`crate::GrafeoDB::open_multi_with`]. Cheap to construct
/// via `..Default::default()` field updates.
#[derive(Debug, Clone, Default)]
#[cfg(feature = "lpg")]
pub struct OpenMultiOptions {
    /// How to reconcile schema catalogs across the input snapshots.
    /// See [`SchemaMergePolicy`].
    pub schema_policy: SchemaMergePolicy,
    /// Whether to retain exact physical images or explicitly rebuild union indexes.
    pub index_policy: IndexMergePolicy,
}

/// Summary metadata about a snapshot blob, as produced by
/// [`snapshot_info`]. Useful for pre-flight checks (e.g. rejecting a
/// snapshot whose `version` is older than the runtime expects, or
/// alerting on an unexpected `node_count`) without loading the
/// snapshot into a full database.
#[derive(Debug, Clone)]
#[cfg(feature = "lpg")]
pub struct SnapshotInfo {
    /// Current snapshot format version (12, with the mandatory CDC1 envelope).
    pub version: u8,
    /// Store epoch at snapshot time (0 when temporal is disabled).
    pub epoch: u64,
    /// Number of nodes serialized in the blob.
    pub node_count: usize,
    /// Number of edges serialized in the blob.
    pub edge_count: usize,
    /// Number of named graphs (outer Vec length, not inner node/edge counts).
    pub named_graph_count: usize,
    /// Number of default-graph RDF triple history rows serialized in the blob.
    pub rdf_triple_count: usize,
    /// Number of property index keys recorded in the blob.
    pub property_index_count: usize,
    /// Number of vector index descriptors recorded in the blob.
    pub vector_index_count: usize,
    /// Number of text index descriptors recorded in the blob.
    pub text_index_count: usize,
}

/// Parses a snapshot blob and returns its summary metadata without
/// constructing a [`crate::GrafeoDB`]. The blob is fully decoded
/// (bincode requires it), so this isn't free — it's just much cheaper
/// than [`crate::GrafeoDB::open_multi`] because no graph store is built and
/// no indexes are rebuilt.
///
/// # Errors
///
/// Returns an error if the blob is empty, the version byte is
/// unsupported, or the bincode payload fails to decode.
#[cfg(feature = "lpg")]
pub fn snapshot_info(bytes: &[u8]) -> Result<SnapshotInfo> {
    let snap = decode_snapshot_bytes(bytes)?;
    let catalog = stage_snapshot_catalog(&snap)?;
    let _images = stage_snapshot_indexes(&snap, &catalog)?;
    let (mut property_index_count, mut vector_index_count, mut text_index_count) = (0, 0, 0);
    for owner in catalog.all_indexes() {
        match owner.key.family() {
            grafeo_core::graph::lpg::PhysicalIndexFamily::Property => property_index_count += 1,
            grafeo_core::graph::lpg::PhysicalIndexFamily::Vector => vector_index_count += 1,
            grafeo_core::graph::lpg::PhysicalIndexFamily::Text => text_index_count += 1,
        }
    }
    #[cfg(feature = "triple-store")]
    let rdf_triple_count = if matches!(snap.graph_model, 1 | 2) {
        decode_rdf_dataset_history(&snap)?.quad_versions().len()
    } else {
        0
    };
    #[cfg(not(feature = "triple-store"))]
    let rdf_triple_count = 0;
    Ok(SnapshotInfo {
        version: snap.version,
        epoch: snap.epoch,
        node_count: snap.nodes()?.len(),
        edge_count: snap.edges()?.len(),
        named_graph_count: snap.named_graphs().len(),
        rdf_triple_count,
        property_index_count,
        vector_index_count,
        text_index_count,
    })
}

/// Current portable snapshot: explicit identity, exact LPG histories and
/// canonical graph-qualified RDF temporal history.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(super) struct Snapshot {
    version: u8,
    epoch: u64,
    graph_model: u8,
    world_identity: WorldIdentityMetadataV1,
    next_graph_incarnation_id: u64,
    graphs: Vec<SnapshotLpgGraph>,
    /// Authoritative current Catalog7 image; no descriptor/schema mirror.
    catalog_state: Vec<u8>,
    text_indexes: Vec<u8>,
    vector_indexes: Vec<u8>,
    rdf_lpg_projections: Vec<u8>,
    rdf_dataset_history: Vec<u8>,
    cdc_checkpoint: Vec<u8>,
}

impl Snapshot {
    #[cfg(any(feature = "lpg", feature = "cdc"))]
    fn incarnations(&self) -> Vec<(GraphPath, grafeo_common::types::GraphIncarnationId)> {
        self.graphs
            .iter()
            .map(|graph| (graph.path.clone(), graph.incarnation))
            .collect()
    }
    fn root_rows(&self) -> Result<(&[SnapshotNode], &[SnapshotEdge])> {
        match (resolved_snapshot_graph_model(self)?, self.graphs.first()) {
            (GraphModel::Rdf, None) => Ok((&[], &[])),
            (GraphModel::Lpg | GraphModel::Both, Some(graph))
                if graph.path == GraphPath::root() =>
            {
                Ok((&graph.nodes, &graph.edges))
            }
            _ => Err(Error::Serialization(
                "snapshot has no model-appropriate LPG root".into(),
            )),
        }
    }
    fn nodes(&self) -> Result<&[SnapshotNode]> {
        self.root_rows().map(|rows| rows.0)
    }
    fn edges(&self) -> Result<&[SnapshotEdge]> {
        self.root_rows().map(|rows| rows.1)
    }
    fn named_graphs(&self) -> &[SnapshotLpgGraph] {
        match self.graphs.split_first() {
            Some((_, descendants)) => descendants,
            None => &[],
        }
    }
}

/// One exact, component-qualified LPG graph. Counters cannot be detached from
/// their owner, including when no entity exists for reserved allocator gaps.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct SnapshotLpgGraph {
    #[serde(with = "super::portable_wire::graph_path")]
    path: GraphPath,
    incarnation: grafeo_common::types::GraphIncarnationId,
    next_node_id: u64,
    next_edge_id: u64,
    /// Earliest epoch whose LPG history is complete for this graph.
    retained_history_floor: u64,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
}

/// Transient schema merge view, derived from the authoritative Catalog7 image.
#[derive(Default, Clone)]
#[cfg(feature = "lpg")]
struct SnapshotSchema {
    node_types: Vec<NodeTypeDefinition>,
    edge_types: Vec<EdgeTypeDefinition>,
    graph_types: Vec<GraphTypeDefinition>,
    procedures: Vec<ProcedureDefinition>,
    schemas: Vec<String>,
    graph_type_bindings: Vec<(GraphPath, String)>,
}

#[cfg(feature = "triple-store")]
const SNAPSHOT_RDF_HISTORY_PAYLOAD_VERSION: u8 = 1;
#[cfg(feature = "triple-store")]
const MAX_SNAPSHOT_RDF_HISTORY_BYTES: usize = 512 * 1024 * 1024;

#[cfg(feature = "triple-store")]
#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotRdfDatasetHistoryV1 {
    version: u8,
    next_graph_incarnation: grafeo_common::types::GraphIncarnationId,
    graph_lives: Vec<grafeo_core::graph::rdf::RdfGraphLife>,
    quad_versions: Vec<grafeo_core::graph::rdf::RdfQuadVersion>,
}

#[cfg(feature = "triple-store")]
fn encode_rdf_dataset_history(
    history: &grafeo_core::graph::rdf::RdfDatasetHistory,
) -> Result<Vec<u8>> {
    let wire = SnapshotRdfDatasetHistoryV1 {
        version: SNAPSHOT_RDF_HISTORY_PAYLOAD_VERSION,
        next_graph_incarnation: history.next_graph_incarnation(),
        graph_lives: history.graph_lives().to_vec(),
        quad_versions: history.quad_versions().to_vec(),
    };
    let bytes = bincode::serde::encode_to_vec(
        &wire,
        bincode::config::standard().with_limit::<MAX_SNAPSHOT_RDF_HISTORY_BYTES>(),
    )
    .map_err(|error| {
        Error::Serialization(format!("snapshot RDF history encode failed: {error}"))
    })?;
    if bytes.len() > MAX_SNAPSHOT_RDF_HISTORY_BYTES {
        return Err(Error::Serialization(format!(
            "snapshot RDF history has {} bytes; maximum is {MAX_SNAPSHOT_RDF_HISTORY_BYTES}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

#[cfg(feature = "triple-store")]
fn decode_rdf_dataset_history(
    snapshot: &Snapshot,
) -> Result<grafeo_core::graph::rdf::RdfDatasetHistory> {
    if snapshot.rdf_dataset_history.is_empty() {
        return Err(Error::Serialization(format!(
            "snapshot v{} RDF graph model requires canonical dataset history",
            snapshot.version
        )));
    }
    if snapshot.rdf_dataset_history.len() > MAX_SNAPSHOT_RDF_HISTORY_BYTES {
        return Err(Error::Serialization(format!(
            "snapshot RDF history has {} bytes; maximum is {MAX_SNAPSHOT_RDF_HISTORY_BYTES}",
            snapshot.rdf_dataset_history.len()
        )));
    }
    let config = bincode::config::standard().with_limit::<MAX_SNAPSHOT_RDF_HISTORY_BYTES>();
    let (wire, consumed): (SnapshotRdfDatasetHistoryV1, usize) =
        bincode::serde::decode_from_slice(&snapshot.rdf_dataset_history, config).map_err(
            |error| Error::Serialization(format!("snapshot RDF history decode failed: {error}")),
        )?;
    if consumed != snapshot.rdf_dataset_history.len() {
        return Err(Error::Serialization(format!(
            "snapshot RDF history has {} trailing bytes",
            snapshot.rdf_dataset_history.len() - consumed
        )));
    }
    if wire.version != SNAPSHOT_RDF_HISTORY_PAYLOAD_VERSION {
        return Err(Error::Serialization(format!(
            "unsupported snapshot RDF history payload version {}",
            wire.version
        )));
    }
    let history = grafeo_core::graph::rdf::RdfDatasetHistory::new_with_high_water(
        snapshot.world_identity.store_id(),
        snapshot.world_identity.history(),
        wire.next_graph_incarnation,
        wire.graph_lives,
        wire.quad_versions,
    )
    .map_err(|error| Error::Serialization(format!("invalid snapshot RDF history: {error}")))?;
    let snapshot_epoch = EpochId::new(snapshot.epoch);
    for tx in history
        .graph_lives()
        .iter()
        .map(grafeo_core::graph::rdf::RdfGraphLife::tx)
        .chain(
            history
                .quad_versions()
                .iter()
                .map(grafeo_core::graph::rdf::RdfQuadVersion::tx),
        )
    {
        if tx.from() > snapshot_epoch || (!tx.is_open() && tx.to() > snapshot_epoch) {
            return Err(Error::Serialization(
                "snapshot RDF history extends beyond the snapshot epoch".to_string(),
            ));
        }
    }
    Ok(history)
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct SnapshotNode {
    id: NodeId,
    /// Committed structural lifetimes, oldest first, as half-open
    /// `(created, deleted)` intervals. `None` is the current open lifetime.
    lifetimes: Vec<(EpochId, Option<EpochId>)>,
    /// Complete label-set versions in stable ascending epoch order.
    label_versions: Vec<(EpochId, Vec<String>)>,
    /// Each property has a list of `(epoch, value)` entries (ascending epoch order).
    #[serde(with = "super::portable_wire::properties")]
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

#[cfg(feature = "lpg")]
impl SnapshotNode {
    fn labels(&self) -> Result<&[String]> {
        self.label_versions
            .last()
            .map(|(_, labels)| labels.as_slice())
            .ok_or_else(|| {
                Error::Serialization(format!("snapshot node {} has no label history", self.id))
            })
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct SnapshotEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    /// Committed structural lifetimes, oldest first.
    lifetimes: Vec<(EpochId, Option<EpochId>)>,
    /// Each property has a list of `(epoch, value)` entries (ascending epoch order).
    #[serde(with = "super::portable_wire::properties")]
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

fn next_id_after_known_max(
    graph: &GraphPath,
    entity: &str,
    ids: impl Iterator<Item = u64>,
) -> Result<u64> {
    ids.max().map_or(Ok(0), |maximum| {
        maximum.checked_add(1).ok_or_else(|| {
            Error::Serialization(format!(
                "snapshot LPG graph {graph:?} contains {entity} ID {maximum}, leaving no representable allocator high-water"
            ))
        })
    })
}

/// Collects all nodes from a store into snapshot format.
///
/// With `temporal`: stores full property version history.
/// Without: wraps each current value as a single-entry version list at epoch 0.
#[cfg(feature = "lpg")]
fn collect_snapshot_nodes(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<SnapshotNode> {
    collect_snapshot_nodes_selected(store, store.all_node_ids())
}

#[cfg(feature = "lpg")]
fn collect_snapshot_nodes_selected(
    store: &grafeo_core::graph::lpg::LpgStore,
    ids: Vec<NodeId>,
) -> Vec<SnapshotNode> {
    let boundary = store.current_epoch();
    let mut nodes: Vec<SnapshotNode> = ids
        .into_iter()
        .filter_map(|id| {
            let history = store.get_node_history(id);
            let mut lifetimes: Vec<_> = history
                .iter()
                .filter_map(|(created, deleted, _)| {
                    (*created != EpochId::PENDING && *created <= boundary).then_some((
                        *created,
                        deleted.filter(|epoch| *epoch != EpochId::PENDING && *epoch <= boundary),
                    ))
                })
                .collect();
            // The store yields newest first; equal-birth zero-width lives
            // must precede their open successor in the portable history.
            lifetimes.reverse();
            lifetimes.sort_by_key(|(created, _)| *created);
            if lifetimes.is_empty() {
                return None;
            }

            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
                .node_property_history(id)
                .into_iter()
                .filter_map(|(key, entries)| {
                    let entries: Vec<_> = entries
                        .into_iter()
                        .filter(|(epoch, _)| *epoch != EpochId::PENDING && *epoch <= boundary)
                        .collect();
                    (!entries.is_empty()).then(|| (key.to_string(), entries))
                })
                .collect();
            properties.sort_by(|(a, _), (b, _)| a.cmp(b));

            let mut label_versions: Vec<(EpochId, Vec<String>)> = store
                .node_label_history(id)
                .into_iter()
                .filter_map(|(epoch, labels)| {
                    let visible = epoch != EpochId::PENDING
                        && epoch <= boundary
                        && lifetimes.iter().any(|(created, deleted)| {
                            *created <= epoch && deleted.is_none_or(|deleted| epoch <= deleted)
                        });
                    visible.then(|| {
                        let mut labels: Vec<String> =
                            labels.into_iter().map(|label| label.to_string()).collect();
                        labels.sort();
                        labels.dedup();
                        (epoch, labels)
                    })
                })
                .collect();
            label_versions.sort_by_key(|(epoch, _)| *epoch);

            // A complete set at every create boundary makes each lifetime
            // independently replayable. Older stores normally already have
            // this entry; synthesize it from the structural record if needed.
            for (created, _) in &lifetimes {
                if !label_versions.iter().any(|(epoch, _)| epoch == created) {
                    let mut labels: Vec<String> = history
                        .iter()
                        .find(|(epoch, _, _)| epoch == created)
                        .map(|(_, _, node)| node.labels.iter().map(ToString::to_string).collect())
                        .unwrap_or_default();
                    labels.sort();
                    labels.dedup();
                    label_versions.push((*created, labels));
                }
            }
            label_versions.sort_by_key(|(epoch, _)| *epoch);

            Some(SnapshotNode {
                id,
                lifetimes,
                label_versions,
                properties,
            })
        })
        .collect();
    nodes.sort_by_key(|n| n.id);
    nodes
}

/// Collects all edges from a store into snapshot format.
///
/// With `temporal`: stores full property version history.
/// Without: wraps each current value as a single-entry version list at epoch 0.
#[cfg(feature = "lpg")]
fn collect_snapshot_edges(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<SnapshotEdge> {
    collect_snapshot_edges_selected(store, store.all_known_edge_ids())
}

#[cfg(feature = "lpg")]
fn collect_snapshot_edges_selected(
    store: &grafeo_core::graph::lpg::LpgStore,
    ids: Vec<EdgeId>,
) -> Vec<SnapshotEdge> {
    let boundary = store.current_epoch();
    let mut edges: Vec<SnapshotEdge> = ids
        .into_iter()
        .filter_map(|id| {
            let history = store.get_edge_history(id);
            let mut committed: Vec<_> = history
                .iter()
                .filter(|(created, _, _)| *created != EpochId::PENDING && *created <= boundary)
                .collect();
            committed.reverse();
            committed.sort_by_key(|(created, _, _)| *created);
            let (_, _, identity) = committed.first()?;
            let lifetimes = committed
                .iter()
                .map(|(created, deleted, _)| {
                    (
                        *created,
                        deleted.filter(|epoch| *epoch != EpochId::PENDING && *epoch <= boundary),
                    )
                })
                .collect();
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
                .edge_property_history(id)
                .into_iter()
                .filter_map(|(key, entries)| {
                    let entries: Vec<_> = entries
                        .into_iter()
                        .filter(|(epoch, _)| *epoch != EpochId::PENDING && *epoch <= boundary)
                        .collect();
                    (!entries.is_empty()).then(|| (key.to_string(), entries))
                })
                .collect();
            properties.sort_by(|(a, _), (b, _)| a.cmp(b));

            Some(SnapshotEdge {
                id,
                src: identity.src,
                dst: identity.dst,
                edge_type: identity.edge_type.to_string(),
                lifetimes,
                properties,
            })
        })
        .collect();
    edges.sort_by_key(|e| e.id);
    edges
}

fn validate_property_histories(
    entity: &str,
    properties: &[(String, Vec<(EpochId, Value)>)],
    lifetimes: &[(EpochId, Option<EpochId>)],
    boundary: EpochId,
) -> Result<()> {
    let mut keys = HashSet::with_capacity(properties.len());
    for (key, versions) in properties {
        if !keys.insert(key.as_str()) {
            return Err(Error::Serialization(format!(
                "{entity} contains duplicate property history {key:?}"
            )));
        }
        if versions.is_empty() {
            return Err(Error::Serialization(format!(
                "{entity} property {key:?} has an empty history"
            )));
        }
        let mut previous = None;
        for (index, (epoch, _)) in versions.iter().enumerate() {
            if *epoch == EpochId::PENDING || *epoch > boundary {
                return Err(Error::Serialization(format!(
                    "{entity} property {key:?} version {index} is outside the snapshot cut"
                )));
            }
            if previous.is_some_and(|previous| *epoch < previous) {
                return Err(Error::Serialization(format!(
                    "{entity} property {key:?} versions are not epoch-ordered"
                )));
            }
            if !epoch_is_inside_lifetime(*epoch, lifetimes, true) {
                return Err(Error::Serialization(format!(
                    "{entity} property {key:?} version {index} is outside every structural lifetime"
                )));
            }
            previous = Some(*epoch);
        }
    }
    Ok(())
}

fn validate_snapshot_histories(
    nodes: &[SnapshotNode],
    edges: &[SnapshotEdge],
    snapshot_epoch: u64,
) -> Result<()> {
    let boundary = EpochId::new(snapshot_epoch);
    for node in nodes {
        let entity = format!("snapshot node {}", node.id);
        validate_structural_lifetimes(&entity, &node.lifetimes, boundary)?;
        validate_property_histories(&entity, &node.properties, &node.lifetimes, boundary)?;

        if node.label_versions.is_empty() {
            return Err(Error::Serialization(format!(
                "{entity} has no label history"
            )));
        }
        let mut previous = None;
        for (index, (epoch, labels)) in node.label_versions.iter().enumerate() {
            if *epoch == EpochId::PENDING
                || *epoch > boundary
                || !epoch_is_inside_lifetime(*epoch, &node.lifetimes, true)
            {
                return Err(Error::Serialization(format!(
                    "{entity} label version {index} is outside the committed structural history"
                )));
            }
            if previous.is_some_and(|previous| *epoch < previous) {
                return Err(Error::Serialization(format!(
                    "{entity} label versions are not epoch-ordered"
                )));
            }
            let mut unique = HashSet::with_capacity(labels.len());
            if labels.iter().any(|label| !unique.insert(label.as_str())) {
                return Err(Error::Serialization(format!(
                    "{entity} label version {index} contains duplicate labels"
                )));
            }
            previous = Some(*epoch);
        }
        for (index, (created, _)) in node.lifetimes.iter().enumerate() {
            if !node
                .label_versions
                .iter()
                .any(|(epoch, _)| epoch == created)
            {
                return Err(Error::Serialization(format!(
                    "{entity} lifetime {index} has no complete label set at creation"
                )));
            }
        }
    }

    for edge in edges {
        let entity = format!("snapshot edge {}", edge.id);
        validate_structural_lifetimes(&entity, &edge.lifetimes, boundary)?;
        validate_property_histories(&entity, &edge.properties, &edge.lifetimes, boundary)?;
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
            return Err(Error::Serialization(format!(
                "snapshot contains duplicate node ID {}",
                node.id
            )));
        }
    }
    let mut edge_ids = HashSet::with_capacity(edges.len());
    for edge in edges {
        if !edge_ids.insert(edge.id) {
            return Err(Error::Serialization(format!(
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
    let node_by_id: HashMap<NodeId, &SnapshotNode> =
        nodes.iter().map(|node| (node.id, node)).collect();
    for edge in edges {
        let Some(source) = node_by_id.get(&edge.src) else {
            return Err(Error::Serialization(format!(
                "snapshot edge {} references non-existent source node {}",
                edge.id, edge.src
            )));
        };
        let Some(destination) = node_by_id.get(&edge.dst) else {
            return Err(Error::Serialization(format!(
                "snapshot edge {} references non-existent destination node {}",
                edge.id, edge.dst
            )));
        };
        for &lifetime in &edge.lifetimes {
            if !lifetime_is_covered_by(lifetime, &source.lifetimes) {
                return Err(Error::Serialization(format!(
                    "snapshot edge {} outlives source node {}",
                    edge.id, edge.src
                )));
            }
            if !lifetime_is_covered_by(lifetime, &destination.lifetimes) {
                return Err(Error::Serialization(format!(
                    "snapshot edge {} outlives destination node {}",
                    edge.id, edge.dst
                )));
            }
        }
    }
    Ok(())
}

/// Origin information for a node observed during cross-snapshot
/// validation. Captured the first time a NodeId is seen so collision
/// errors can name BOTH conflicting sides.
#[derive(Debug, Clone)]
#[cfg(feature = "lpg")]
struct NodeOrigin {
    snapshot_idx: usize,
    /// Sorted to match `collect_snapshot_nodes`' canonical ordering;
    /// safe to compare across runs.
    labels: Vec<String>,
    /// The external `id` property value if present — gives the operator
    /// a domain handle when debugging a collision (e.g. `concept:bitter`).
    id_prop: Option<String>,
}

#[cfg(feature = "lpg")]
impl NodeOrigin {
    fn from_node(snapshot_idx: usize, node: &SnapshotNode) -> Result<Self> {
        let id_prop = node.properties.iter().find_map(|(key, history)| {
            if key != "id" {
                return None;
            }
            history.last().and_then(|(_, value)| match value {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
        });
        Ok(Self {
            snapshot_idx,
            labels: node.labels()?.to_vec(),
            id_prop,
        })
    }

    fn describe(&self) -> String {
        let labels = if self.labels.is_empty() {
            String::from("[]")
        } else {
            format!("[{}]", self.labels.join(","))
        };
        match &self.id_prop {
            Some(id) => format!(
                "snapshot[{}] (labels={labels}, id={id:?})",
                self.snapshot_idx
            ),
            None => format!("snapshot[{}] (labels={labels})", self.snapshot_idx),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "lpg")]
struct EdgeOrigin {
    snapshot_idx: usize,
    edge_type: String,
    src: NodeId,
    dst: NodeId,
}

#[cfg(feature = "lpg")]
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
#[cfg(feature = "lpg")]
fn validate_snapshot_set(snapshots: &[Snapshot]) -> Result<()> {
    let mut node_origins: HashMap<NodeId, NodeOrigin> =
        HashMap::with_capacity(snapshots.iter().try_fold(0_usize, |total, snapshot| {
            total
                .checked_add(snapshot.nodes()?.len())
                .ok_or_else(|| Error::Serialization("snapshot union node count overflow".into()))
        })?);
    let mut node_histories: HashMap<NodeId, &SnapshotNode> =
        HashMap::with_capacity(node_origins.capacity());

    for (idx, snap) in snapshots.iter().enumerate() {
        for node in snap.nodes()? {
            match node_origins.entry(node.id) {
                hashbrown::hash_map::Entry::Occupied(existing) => {
                    return Err(Error::Internal(format!(
                        "duplicate NodeId {id} is claimed by {prev} and by {curr}; \
                         open_multi requires NodeIds to be disjoint across \
                         snapshots — extract sibling chunks from a single \
                         source DB to preserve a shared ID namespace",
                        id = node.id,
                        prev = existing.get().describe(),
                        curr = NodeOrigin::from_node(idx, node)?.describe(),
                    )));
                }
                hashbrown::hash_map::Entry::Vacant(vacant) => {
                    vacant.insert(NodeOrigin::from_node(idx, node)?);
                    node_histories.insert(node.id, node);
                }
            }
        }
    }

    let mut edge_origins: HashMap<EdgeId, EdgeOrigin> =
        HashMap::with_capacity(snapshots.iter().try_fold(0_usize, |total, snapshot| {
            total
                .checked_add(snapshot.edges()?.len())
                .ok_or_else(|| Error::Serialization("snapshot union edge count overflow".into()))
        })?);

    for (idx, snap) in snapshots.iter().enumerate() {
        for edge in snap.edges()? {
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
            let source = node_histories
                .get(&edge.src)
                .expect("source existence checked above");
            let destination = node_histories
                .get(&edge.dst)
                .expect("destination existence checked above");
            for &lifetime in &edge.lifetimes {
                if !lifetime_is_covered_by(lifetime, &source.lifetimes) {
                    return Err(Error::Serialization(format!(
                        "snapshot[{idx}] edge {} outlives source node {}",
                        edge.id, edge.src
                    )));
                }
                if !lifetime_is_covered_by(lifetime, &destination.lifetimes) {
                    return Err(Error::Serialization(format!(
                        "snapshot[{idx}] edge {} outlives destination node {}",
                        edge.id, edge.dst
                    )));
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "triple-store")]
fn stage_snapshot_rdf_history(
    snapshot: &Snapshot,
) -> Result<grafeo_core::graph::rdf::RdfDatasetHistory> {
    match resolved_snapshot_graph_model(snapshot)? {
        GraphModel::Rdf | GraphModel::Both => decode_rdf_dataset_history(snapshot),
        GraphModel::Lpg => grafeo_core::graph::rdf::RdfDatasetHistory::new_with_high_water(
            snapshot.world_identity.store_id(),
            snapshot.world_identity.history(),
            grafeo_common::types::GraphIncarnationId::FIRST_NAMED,
            Vec::new(),
            Vec::new(),
        )
        .map_err(|error| {
            Error::Serialization(format!(
                "stage exact empty RDF facade for LPG snapshot: {error}"
            ))
        }),
    }
}

#[cfg(feature = "triple-store")]
fn install_snapshot_rdf_history(
    store: &grafeo_core::graph::rdf::RdfStore,
    history: grafeo_core::graph::rdf::RdfDatasetHistory,
    snapshot_epoch: EpochId,
    commit_guard: Option<&grafeo_core::graph::rdf::RdfCommitGuard<'_>>,
) -> Result<()> {
    match commit_guard {
        Some(guard) => {
            store.replace_dataset_history_exact_under_commit_gate(guard, history, snapshot_epoch)
        }
        None => store.replace_dataset_history_exact(history, snapshot_epoch),
    }
    .map_err(|error| Error::Serialization(format!("install RDF dataset history: {error}")))
}

#[cfg(all(feature = "triple-store", feature = "lpg"))]
fn encode_rdf_lpg_projections(
    registry: &grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
) -> Result<Vec<u8>> {
    if registry.snapshot().is_empty() {
        return Ok(Vec::new());
    }
    registry
        .encode_persistence_v3()
        .map_err(|error| Error::Serialization(format!("encode RDF→LPG projections: {error}")))
}

#[cfg(all(feature = "triple-store", feature = "lpg"))]
fn decode_rdf_lpg_projections(
    data: &[u8],
    expected_store_id: StoreId,
) -> Result<Vec<grafeo_core::graph::rdf::RdfLpgProjectionDefinition>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    grafeo_core::graph::rdf::RdfLpgProjectionRegistry::decode_persistence(expected_store_id, data)
        .map_err(|error| Error::Serialization(format!("decode RDF→LPG projections: {error}")))
}

#[cfg(all(feature = "triple-store", feature = "lpg"))]
fn restore_rdf_lpg_projections(
    registry: &grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
    expected_store_id: StoreId,
    definitions: Vec<grafeo_core::graph::rdf::RdfLpgProjectionDefinition>,
) -> Result<()> {
    registry
        .restore_for_store(expected_store_id, definitions)
        .map_err(|error| Error::Serialization(format!("restore RDF→LPG projections: {error}")))
}

fn verify_snapshot_artifact(artifact: &SnapshotArtifact) -> Result<Snapshot> {
    artifact
        .verify()
        .map_err(|error| Error::Serialization(format!("verify snapshot artifact: {error}")))?;
    let snapshot = decode_snapshot_bytes(artifact.bytes())?;
    let actual = snapshot_world_descriptor(&snapshot)?;
    if artifact.cut().descriptor() != &actual {
        return Err(Error::Serialization(
            "snapshot artifact descriptor does not match decoded snapshot semantics".to_string(),
        ));
    }
    Ok(snapshot)
}

#[cfg(all(feature = "triple-store", feature = "lpg"))]
struct ProjectionRowInventory {
    historical_node_ids: HashSet<NodeId>,
}

/// Proves that every persisted projection-owned row has canonical provenance.
///
/// The ownership property is an engine-reserved capability marker, not merely
/// user data. A current-shape check is insufficient: grafting a valid marker
/// onto an older ordinary lifetime would let reconciliation appropriate that
/// row after reopen. This validator therefore binds both marker histories and
/// the label history to the row's one structural lifetime, rejects edges and
/// named-graph ownership, and checks the published receipt's current row count.
#[cfg(all(feature = "triple-store", feature = "lpg"))]
fn validate_rdf_lpg_projection_rows(
    nodes: &[SnapshotNode],
    edges: &[SnapshotEdge],
    named_graphs: &[SnapshotLpgGraph],
    definitions: &[grafeo_core::graph::rdf::RdfLpgProjectionDefinition],
) -> Result<ProjectionRowInventory> {
    use grafeo_core::graph::rdf::{
        RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY,
        RdfLpgProjectionDefinition,
    };

    fn canonical_marker(
        node: &SnapshotNode,
        property: &str,
        created: EpochId,
        deleted: Option<EpochId>,
    ) -> Result<String> {
        let history = node
            .properties
            .iter()
            .find(|(key, _)| key == property)
            .map(|(_, history)| history)
            .ok_or_else(|| {
                Error::Serialization(format!(
                    "non-canonical projection provenance for node {}: missing reserved property {property:?}",
                    node.id
                ))
            })?;
        let Some((first_epoch, Value::String(first_value))) = history.first() else {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: reserved property {property:?} must begin with a string",
                node.id
            )));
        };
        if *first_epoch != created {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: reserved property {property:?} begins at {first_epoch}, not structural creation {created}",
                node.id
            )));
        }
        let expected_len = if deleted.is_some() { 2 } else { 1 };
        if history.len() != expected_len
            || deleted.is_some_and(|deleted| {
                !matches!(history.get(1), Some((epoch, Value::Null)) if *epoch == deleted)
            })
        {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: reserved property {property:?} must be immutable for exactly its structural lifetime",
                node.id
            )));
        }
        Ok(first_value.to_string())
    }

    let mut by_owner: HashMap<String, &RdfLpgProjectionDefinition> = HashMap::new();
    for definition in definitions {
        let marker = definition.owner_marker();
        if let Some(existing) = by_owner.insert(marker.clone(), definition)
            && existing.id() != definition.id()
        {
            return Err(Error::Serialization(format!(
                "projection owner marker {marker:?} is ambiguous between mappings {} and {}",
                existing.id(),
                definition.id()
            )));
        }
    }
    let mut historical_node_ids = HashSet::new();
    let mut rows_by_owner: HashMap<String, HashSet<String>> = HashMap::new();

    for node in nodes {
        let has_owner = node
            .properties
            .iter()
            .any(|(key, _)| key == RDF_LPG_PROJECTION_OWNER_PROPERTY);
        let has_iri = node
            .properties
            .iter()
            .any(|(key, _)| key == RDF_LPG_PROJECTION_IRI_PROPERTY);
        if !has_owner && !has_iri {
            continue;
        }
        if !has_owner {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: reserved source IRI has no ownership marker",
                node.id
            )));
        }
        let [(created, deleted)] = node.lifetimes.as_slice() else {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: expected exactly one structural lifetime",
                node.id
            )));
        };
        if node.properties.len() != 2
            || node.properties.iter().any(|(key, _)| {
                key != RDF_LPG_PROJECTION_OWNER_PROPERTY && key != RDF_LPG_PROJECTION_IRI_PROPERTY
            })
        {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: properties exist outside the projection metadata plane",
                node.id
            )));
        }
        let owner = canonical_marker(node, RDF_LPG_PROJECTION_OWNER_PROPERTY, *created, *deleted)?;
        let definition = by_owner.get(owner.as_str()).ok_or_else(|| {
            Error::Serialization(format!(
                "projection-owned node {} references unknown owner {owner:?}",
                node.id
            ))
        })?;
        if definition.last_source_epoch().is_none() {
            return Err(Error::Serialization(format!(
                "projection-owned node {} references unpublished projection {}",
                node.id,
                definition.id()
            )));
        }
        let iri = canonical_marker(node, RDF_LPG_PROJECTION_IRI_PROPERTY, *created, *deleted)?;
        if iri.is_empty() {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: source IRI is empty",
                node.id
            )));
        }
        if node.label_versions.len() != 1
            || node.label_versions[0].0 != *created
            || node.label_versions[0].1.len() != 1
            || node.label_versions[0].1[0] != definition.node_label()
            || node.labels()?.len() != 1
            || node
                .labels()?
                .first()
                .ok_or_else(|| Error::Serialization("snapshot node has no label".into()))?
                != definition.node_label()
        {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: label history is not exactly {:?} from structural creation",
                node.id,
                definition.node_label()
            )));
        }
        historical_node_ids.insert(node.id);

        if deleted.is_some() {
            continue;
        }
        if let Some(target_epoch) = definition.last_target_epoch()
            && *created > target_epoch
        {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance for node {}: creation epoch {created} is after projection target epoch {target_epoch}",
                node.id
            )));
        }
        if !rows_by_owner
            .entry(definition.owner_marker())
            .or_default()
            .insert(iri.clone())
        {
            return Err(Error::Serialization(format!(
                "projection owner {owner:?} has duplicate row IRI {iri:?}"
            )));
        }
    }

    for graph in named_graphs {
        for node in &graph.nodes {
            if node.properties.iter().any(|(key, _)| {
                key == RDF_LPG_PROJECTION_OWNER_PROPERTY || key == RDF_LPG_PROJECTION_IRI_PROPERTY
            }) {
                return Err(Error::Serialization(format!(
                    "non-canonical projection provenance for named graph {:?}, node {}: projection ownership is confined to the default graph",
                    graph.path, node.id
                )));
            }
        }
    }
    for edge in edges {
        if historical_node_ids.contains(&edge.src) || historical_node_ids.contains(&edge.dst) {
            return Err(Error::Serialization(format!(
                "non-canonical projection provenance: edge {} is incident to projection-owned node history",
                edge.id
            )));
        }
    }
    for definition in definitions
        .iter()
        .filter(|definition| definition.last_source_epoch().is_some())
    {
        let owner = definition.owner_marker();
        let actual = rows_by_owner.get(owner.as_str()).map_or(0, HashSet::len);
        let expected = usize::try_from(definition.row_count()).map_err(|_| {
            Error::Serialization(format!(
                "projection {} row count does not fit this platform",
                definition.id()
            ))
        })?;
        if actual != expected {
            return Err(Error::Serialization(format!(
                "projection {} status records {expected} rows but the captured target has {actual}",
                definition.id()
            )));
        }
    }

    Ok(ProjectionRowInventory {
        historical_node_ids,
    })
}

/// Converts an exact restore artifact into a distinct logical store while
/// preserving ordinary temporal state. Store-bound RDF handles and projection
/// receipts cannot cross the fork boundary.
fn reidentify_snapshot_for_fork_to(
    mut snapshot: Snapshot,
    target_store_id: StoreId,
) -> Result<Snapshot> {
    let source_store_id = snapshot.world_identity.store_id();
    if source_store_id == target_store_id {
        return Err(Error::Serialization(
            "snapshot fork target identity must differ from its source".to_string(),
        ));
    }

    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    {
        let definitions =
            decode_rdf_lpg_projections(&snapshot.rdf_lpg_projections, source_store_id)?;
        let inventory = validate_rdf_lpg_projection_rows(
            snapshot.nodes()?,
            snapshot.edges()?,
            snapshot.named_graphs(),
            &definitions,
        )?;
        if !inventory.historical_node_ids.is_empty() {
            snapshot_indexes::validate_snapshot_fork_indexes(
                &snapshot,
                &inventory.historical_node_ids,
            )?;
            let root = snapshot
                .graphs
                .first_mut()
                .ok_or_else(|| Error::Serialization("projection rows have no LPG root".into()))?;
            root.nodes
                .retain(|node| !inventory.historical_node_ids.contains(&node.id));
            root.edges.retain(|edge| {
                !inventory.historical_node_ids.contains(&edge.src)
                    && !inventory.historical_node_ids.contains(&edge.dst)
            });
        }

        let registry = grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new();
        registry
            .restore_mappings_for_fork(definitions)
            .map_err(|error| {
                Error::Serialization(format!(
                    "reset RDF→LPG projection mappings for snapshot fork: {error}"
                ))
            })?;
        snapshot.rdf_lpg_projections = encode_rdf_lpg_projections(&registry)?;
    }

    #[cfg(feature = "triple-store")]
    if snapshot.version == SNAPSHOT_VERSION
        && matches!(
            resolved_snapshot_graph_model(&snapshot)?,
            GraphModel::Rdf | GraphModel::Both
        )
    {
        let history = decode_rdf_dataset_history(&snapshot)?;
        let forked = history
            .reidentify_for_fork(target_store_id)
            .map_err(|error| {
                Error::Serialization(format!("reidentify RDF history for snapshot fork: {error}"))
            })?;
        snapshot.rdf_dataset_history = encode_rdf_dataset_history(&forked)?;
    }

    snapshot.cdc_checkpoint = super::cdc_checkpoint::reidentify(
        &snapshot.cdc_checkpoint,
        source_store_id,
        target_store_id,
        EpochId::new(snapshot.epoch),
    )?;
    snapshot.world_identity =
        WorldIdentityMetadataV1::new(target_store_id, snapshot.world_identity.history()).map_err(
            |error| Error::Serialization(format!("install snapshot-fork identity: {error}")),
        )?;
    Ok(snapshot)
}

fn generate_snapshot_fork_store_id(snapshots: &[Snapshot]) -> Result<StoreId> {
    loop {
        let candidate = StoreId::generate().map_err(|error| {
            Error::Serialization(format!("generate snapshot-fork identity: {error}"))
        })?;
        if snapshots
            .iter()
            .all(|snapshot| snapshot.world_identity.store_id() != candidate)
        {
            return Ok(candidate);
        }
    }
}

fn reidentify_snapshot_for_fork(snapshot: Snapshot) -> Result<Snapshot> {
    let target_store_id = generate_snapshot_fork_store_id(std::slice::from_ref(&snapshot))?;
    validate_decoded_snapshot(reidentify_snapshot_for_fork_to(snapshot, target_store_id)?)
}

// =========================================================================
// Snapshot deserialization helpers (used by single-file format)
// =========================================================================

/// Decodes snapshot bytes and populates a store and catalog.
#[cfg(feature = "grafeo-file")]
#[cfg(feature = "lpg")]
pub(super) fn load_snapshot_into_store(
    #[cfg(feature = "cdc")] cdc_log: &crate::cdc::CdcLog,
    store: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    catalog: &std::sync::Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &std::sync::Arc<grafeo_core::graph::rdf::RdfStore>,
    #[cfg(feature = "triple-store")] rdf_projections: &std::sync::Arc<
        grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
    >,
    data: &[u8],
) -> grafeo_common::utils::error::Result<(u8, WorldIdentityMetadataV1)> {
    use grafeo_common::utils::error::Error;

    super::catalog_section::CatalogSection::validate_catalog_index_persistence(
        catalog.read().view(),
    )?;
    let snapshot = decode_snapshot_bytes(data).map_err(|e| {
        Error::Serialization(format!("failed to decode snapshot from .grafeo file: {e}"))
    })?;
    let staged_catalog = stage_snapshot_catalog(&snapshot)?;
    let staged_indexes = stage_snapshot_indexes(&snapshot, &staged_catalog)?;

    // Complete structural validation is deliberately ahead of every live
    // mutation. `decode_snapshot_bytes` validates each entity's own history;
    // this second pass proves endpoint existence and lifetime containment.
    validate_snapshot_data(snapshot.nodes()?, snapshot.edges()?)?;
    for graph in snapshot.named_graphs() {
        validate_snapshot_data(&graph.nodes, &graph.edges)?;
    }

    // Nested RDF/projection decoding is likewise staged before LPG restore so
    // malformed opaque payloads cannot leave a half-loaded startup store.
    #[cfg(feature = "triple-store")]
    let exact_rdf_history = stage_snapshot_rdf_history(&snapshot)?;
    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    let projection_definitions = decode_rdf_lpg_projections(
        &snapshot.rdf_lpg_projections,
        snapshot.world_identity.store_id(),
    )?;

    #[cfg(feature = "cdc")]
    super::cdc_checkpoint::prepare(
        &snapshot.cdc_checkpoint,
        snapshot.world_identity.store_id(),
        EpochId::new(snapshot.epoch),
    )?
    .install(cdc_log);
    populate_snapshot_graphs(store, &snapshot)?;
    store.sync_epoch(EpochId::new(snapshot.epoch));
    // Restore RDF triples
    #[cfg(feature = "triple-store")]
    {
        install_snapshot_rdf_history(
            rdf_store,
            exact_rdf_history,
            EpochId::new(snapshot.epoch),
            None,
        )?;
        #[cfg(feature = "lpg")]
        restore_rdf_lpg_projections(
            rdf_projections,
            rdf_store.store_id(),
            projection_definitions,
        )?;
    }

    // Install authoritative catalog/index payloads only after detached
    // validation and complete graph population.
    install_snapshot_catalog_and_indexes_into(store, catalog, staged_catalog, &staged_indexes)?;

    Ok((snapshot.graph_model, snapshot.world_identity))
}

/// Populates a store from snapshot refs (borrowed). Used by `open_multi`
/// and by the single-file loader.
#[cfg(feature = "lpg")]
fn populate_store_from_snapshot_ref(
    store: &grafeo_core::graph::lpg::LpgStore,
    nodes: &[SnapshotNode],
    edges: &[SnapshotEdge],
) -> grafeo_common::utils::error::Result<()> {
    for node in nodes {
        let label_versions: Vec<(EpochId, Vec<arcstr::ArcStr>)> = node
            .label_versions
            .iter()
            .map(|(epoch, labels)| {
                (
                    *epoch,
                    labels
                        .iter()
                        .map(|label| arcstr::ArcStr::from(label.as_str()))
                        .collect(),
                )
            })
            .collect();
        store
            .restore_node_history_exact(node.id, &node.lifetimes, &label_versions)
            .map_err(|error| {
                Error::Serialization(format!(
                    "restore structural history for node {}: {error}",
                    node.id
                ))
            })?;
        for (key, entries) in &node.properties {
            for (epoch, value) in entries {
                store.set_node_property_at_epoch(node.id, key, value.clone(), *epoch);
            }
        }
    }
    for edge in edges {
        store
            .restore_edge_history_exact(
                edge.id,
                edge.src,
                edge.dst,
                &edge.edge_type,
                &edge.lifetimes,
            )
            .map_err(|error| {
                Error::Serialization(format!(
                    "restore structural history for edge {}: {error}",
                    edge.id
                ))
            })?;
        for (key, entries) in &edge.properties {
            for (epoch, value) in entries {
                store.set_edge_property_at_epoch(edge.id, key, value.clone(), *epoch);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "lpg")]
fn snapshot_allocator_record<'a>(
    snapshot: &'a Snapshot,
    path: &GraphPath,
) -> Result<&'a SnapshotLpgGraph> {
    snapshot
        .graphs
        .binary_search_by(|graph| graph.path.cmp(path))
        .ok()
        .and_then(|index| snapshot.graphs.get(index))
        .ok_or_else(|| {
            Error::Serialization(format!("snapshot is missing validated LPG graph {path:?}"))
        })
}

#[cfg(feature = "lpg")]
fn create_snapshot_graph(
    root: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    path: &GraphPath,
) -> Result<std::sync::Arc<grafeo_core::graph::lpg::LpgStore>> {
    let Some((name, parents)) = path.components().split_last() else {
        return Ok(std::sync::Arc::clone(root));
    };
    let mut parent = std::sync::Arc::clone(root);
    for name in parents {
        parent = parent.graph(name).ok_or_else(|| {
            Error::Serialization(format!("snapshot target has no parent for {path:?}"))
        })?;
    }
    parent.create_graph(name)?;
    parent
        .graph(name)
        .ok_or_else(|| Error::Internal(format!("snapshot graph creation did not publish {path:?}")))
}

#[cfg(feature = "lpg")]
fn apply_lpg_allocator_record(
    store: &grafeo_core::graph::lpg::LpgStore,
    path: &GraphPath,
    next_node_id: u64,
    next_edge_id: u64,
) -> Result<()> {
    store.set_next_node_id(next_node_id);
    store.set_next_edge_id(next_edge_id);
    if store.peek_next_node_id() != next_node_id || store.peek_next_edge_id() != next_edge_id {
        return Err(Error::Internal(format!(
            "LPG mutation authority rejected allocator restore for {path:?}"
        )));
    }
    Ok(())
}

#[cfg(feature = "lpg")]
fn apply_lpg_history_floor(
    store: &grafeo_core::graph::lpg::LpgStore,
    path: &GraphPath,
    floor: u64,
    snapshot_epoch: u64,
) -> Result<()> {
    if floor == EpochId::PENDING.as_u64() || floor > snapshot_epoch {
        return Err(Error::Serialization(format!(
            "snapshot LPG history floor for {path:?} is outside its snapshot epoch"
        )));
    }
    store.advance_retained_history_floor(EpochId::new(floor));
    Ok(())
}

#[cfg(feature = "lpg")]
fn build_snapshot_graphs(
    target: &grafeo_core::graph::lpg::LpgStore,
    snapshot: &Snapshot,
) -> Result<grafeo_core::graph::lpg::LpgStore> {
    let root = std::sync::Arc::new(target.new_live_replacement_candidate()?);
    for record in &snapshot.graphs {
        let graph = create_snapshot_graph(&root, &record.path)?;
        populate_store_from_snapshot_ref(&graph, &record.nodes, &record.edges)?;
        graph.sync_epoch(EpochId::new(snapshot.epoch));
        apply_lpg_allocator_record(
            &graph,
            &record.path,
            record.next_node_id,
            record.next_edge_id,
        )?;
        apply_lpg_history_floor(
            &graph,
            &record.path,
            record.retained_history_floor,
            snapshot.epoch,
        )?;
    }
    root.sync_epoch(EpochId::new(snapshot.epoch));
    let mut root = std::sync::Arc::try_unwrap(root).map_err(|_| {
        Error::Internal("detached snapshot retained an unexpected root alias".into())
    })?;
    if !snapshot.graphs.is_empty() {
        root.restore_graph_incarnations(
            &snapshot.incarnations(),
            snapshot.next_graph_incarnation_id,
        )?;
    }
    Ok(root)
}

#[cfg(feature = "lpg")]
fn populate_snapshot_graphs(
    root: &std::sync::Arc<grafeo_core::graph::lpg::LpgStore>,
    snapshot: &Snapshot,
) -> Result<()> {
    root.install_pristine_image(build_snapshot_graphs(root, snapshot)?)
}

/// Validates every path, history and native-model plane before any target mutation.
fn validate_decoded_snapshot(snapshot: Snapshot) -> Result<Snapshot> {
    let graph_model = resolved_snapshot_graph_model(&snapshot)?;
    validate_snapshot_topology(&snapshot, graph_model)?;
    if snapshot.epoch == EpochId::PENDING.as_u64() {
        return Err(Error::Serialization(
            "snapshot epoch is the reserved PENDING sentinel".into(),
        ));
    }
    super::cdc_checkpoint::validate(
        &snapshot.cdc_checkpoint,
        snapshot.world_identity.store_id(),
        EpochId::new(snapshot.epoch),
    )?;
    if snapshot
        .world_identity
        .history()
        .authoritative_from()
        .is_some_and(|epoch| epoch.as_u64() > snapshot.epoch)
    {
        return Err(Error::Serialization(
            "snapshot history-completeness boundary exceeds its committed epoch".into(),
        ));
    }
    match graph_model {
        GraphModel::Rdf | GraphModel::Both if snapshot.rdf_dataset_history.is_empty() => {
            return Err(Error::Serialization(
                "snapshot RDF model has no canonical dataset history".into(),
            ));
        }
        GraphModel::Lpg if !snapshot.rdf_dataset_history.is_empty() => {
            return Err(Error::Serialization(
                "snapshot LPG model unexpectedly carries RDF history".into(),
            ));
        }
        _ => {}
    }
    validate_snapshot_feature_support(&snapshot, graph_model)?;
    validate_snapshot_lpg_allocator_state(&snapshot, graph_model)?;
    if graph_model == GraphModel::Rdf {
        if snapshot.next_graph_incarnation_id != 1 {
            return Err(Error::Serialization(
                "RDF-only snapshot has LPG allocator state".into(),
            ));
        }
    } else {
        #[cfg(feature = "lpg")]
        grafeo_core::graph::lpg::LpgStore::validate_graph_incarnations(
            &snapshot.incarnations(),
            snapshot.next_graph_incarnation_id,
        )?;
        if !snapshot.graphs[0].incarnation.is_default_graph() {
            return Err(Error::Serialization(
                "snapshot default LPG graph has a named incarnation".into(),
            ));
        }
    }

    for graph in &snapshot.graphs {
        if graph.retained_history_floor == EpochId::PENDING.as_u64()
            || graph.retained_history_floor > snapshot.epoch
        {
            return Err(Error::Serialization(format!(
                "snapshot LPG history floor for {:?} is outside its snapshot epoch",
                graph.path
            )));
        }
        validate_snapshot_histories(&graph.nodes, &graph.edges, snapshot.epoch)?;
    }
    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    {
        let definitions = decode_rdf_lpg_projections(
            &snapshot.rdf_lpg_projections,
            snapshot.world_identity.store_id(),
        )?;
        validate_rdf_lpg_projection_rows(
            snapshot.nodes()?,
            snapshot.edges()?,
            snapshot.named_graphs(),
            &definitions,
        )?;
    }
    #[cfg(feature = "triple-store")]
    let rdf_next = if matches!(graph_model, GraphModel::Rdf | GraphModel::Both) {
        decode_rdf_dataset_history(&snapshot)?
            .next_graph_incarnation()
            .as_u64()
    } else {
        0
    };
    #[cfg(not(feature = "triple-store"))]
    let rdf_next = 0;
    #[cfg(feature = "cdc")]
    super::cdc_checkpoint::prepare(
        &snapshot.cdc_checkpoint,
        snapshot.world_identity.store_id(),
        EpochId::new(snapshot.epoch),
    )?
    .validate_native(crate::cdc::checkpoint::NativeCut {
        model: snapshot.graph_model,
        lpg_next: snapshot.next_graph_incarnation_id,
        lpg_owners: &snapshot.incarnations(),
        rdf_next,
    })?;
    #[cfg(not(feature = "cdc"))]
    let _ = rdf_next;
    Ok(snapshot)
}

/// Requires the explicit graph model carried by the current snapshot.
fn resolved_snapshot_graph_model(snapshot: &Snapshot) -> Result<GraphModel> {
    GraphModel::from_u8(snapshot.graph_model).ok_or_else(|| {
        Error::Serialization(format!(
            "snapshot v{} has invalid graph model tag {}",
            snapshot.version, snapshot.graph_model
        ))
    })
}

/// Proves that payload planes agree with the declared graph topology before
/// any feature-gated decoder can silently skip them.
fn validate_snapshot_topology(snapshot: &Snapshot, graph_model: GraphModel) -> Result<()> {
    if graph_model == GraphModel::Rdf && !snapshot.graphs.is_empty() {
        return Err(Error::Serialization(
            "RDF snapshot unexpectedly carries LPG graph rows".into(),
        ));
    }
    if graph_model == GraphModel::Lpg && !snapshot.rdf_dataset_history.is_empty() {
        return Err(Error::Serialization(
            "LPG snapshot unexpectedly carries RDF state".into(),
        ));
    }
    if graph_model != GraphModel::Both && !snapshot.rdf_lpg_projections.is_empty() {
        return Err(Error::Serialization(
            "RDF→LPG projection metadata requires GraphModel::Both".into(),
        ));
    }
    Ok(())
}

pub(super) fn encode_snapshot_bytes(snapshot: &Snapshot) -> Result<Vec<u8>> {
    let body = bincode::serde::encode_to_vec(snapshot, bincode::config::standard())
        .map_err(|error| Error::Serialization(format!("snapshot encode failed: {error}")))?;
    validate_snapshot_size(body.len().saturating_add(13))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve(13 + body.len())
        .map_err(|error| Error::Serialization(error.to_string()))?;
    bytes.push(snapshot.version);
    bytes.extend_from_slice(b"CDC1");
    bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

fn decode_snapshot_bytes(data: &[u8]) -> Result<Snapshot> {
    validate_snapshot_size(data.len())?;
    let version = data
        .first()
        .copied()
        .ok_or_else(|| Error::Serialization("empty snapshot data".into()))?;
    if version != SNAPSHOT_VERSION {
        return Err(Error::Serialization(format!(
            "unsupported snapshot version: {version} (expected {SNAPSHOT_VERSION})"
        )));
    }
    if data.len() < 13 || &data[1..5] != b"CDC1" {
        return Err(Error::Serialization(
            "unsupported Snapshot12 outer schema (expected CDC1)".into(),
        ));
    }
    let mut length = [0; 8];
    length.copy_from_slice(&data[5..13]);
    if u64::from_le_bytes(length) != (data.len() - 13) as u64 {
        return Err(Error::Serialization(
            "Snapshot12 bounded body length mismatch".into(),
        ));
    }
    let (snapshot, consumed): (Snapshot, usize) = bincode::serde::decode_from_slice(
        &data[13..],
        bincode::config::standard().with_limit::<MAX_PORTABLE_SNAPSHOT_BYTES>(),
    )
    .map_err(|error| Error::Serialization(format!("snapshot v12 decode failed: {error}")))?;
    if consumed != data.len() - 13 || snapshot.version != SNAPSHOT_VERSION {
        return Err(Error::Serialization(
            "Snapshot12 trailing bytes or inner version mismatch".into(),
        ));
    }

    if snapshot.catalog_state.is_empty() {
        return Err(Error::Serialization(format!(
            "snapshot v{} requires its authoritative Catalog7 image",
            snapshot.version
        )));
    }
    validate_decoded_snapshot(snapshot)
}

fn validate_snapshot_feature_support(snapshot: &Snapshot, graph_model: GraphModel) -> Result<()> {
    let _ = (snapshot, graph_model);
    #[cfg(not(feature = "triple-store"))]
    if !snapshot.rdf_dataset_history.is_empty()
        || !snapshot.rdf_lpg_projections.is_empty()
        || matches!(graph_model, GraphModel::Rdf | GraphModel::Both)
    {
        return Err(Error::Serialization(
            "snapshot contains RDF state but triple-store support is disabled".into(),
        ));
    }
    #[cfg(not(feature = "lpg"))]
    if !snapshot.graphs.is_empty()
        || !snapshot.text_indexes.is_empty()
        || !snapshot.vector_indexes.is_empty()
        || !snapshot.rdf_lpg_projections.is_empty()
        || matches!(graph_model, GraphModel::Lpg | GraphModel::Both)
    {
        return Err(Error::Serialization(
            "snapshot contains LPG state but lpg support is disabled".into(),
        ));
    }
    #[cfg(not(feature = "text-index"))]
    if !snapshot.text_indexes.is_empty() {
        return Err(Error::Serialization(
            "snapshot contains Text state but text-index support is disabled".into(),
        ));
    }
    #[cfg(not(feature = "vector-index"))]
    if !snapshot.vector_indexes.is_empty() {
        return Err(Error::Serialization(
            "snapshot contains Vector state but vector-index support is disabled".into(),
        ));
    }
    Ok(())
}

fn validate_snapshot_lpg_allocator_state(
    snapshot: &Snapshot,
    graph_model: GraphModel,
) -> Result<()> {
    if graph_model == GraphModel::Rdf {
        if !snapshot.graphs.is_empty() {
            return Err(Error::Serialization(
                "RDF snapshot unexpectedly carries LPG allocator state".into(),
            ));
        }
        return Ok(());
    }
    if snapshot
        .graphs
        .first()
        .is_none_or(|graph| graph.path != GraphPath::root())
    {
        return Err(Error::Serialization(
            "snapshot is missing its LPG root graph".into(),
        ));
    }
    let mut seen = HashSet::<&[String]>::with_capacity(snapshot.graphs.len());
    let mut previous = None;
    for graph in &snapshot.graphs {
        if previous.is_some_and(|path| path >= &graph.path) {
            return Err(Error::Serialization(
                "snapshot LPG graph paths must be canonical, sorted and unique".into(),
            ));
        }
        if let Some((_, parent)) = graph.path.components().split_last()
            && !seen.contains(parent)
        {
            return Err(Error::Serialization(format!(
                "snapshot LPG graph {:?} has no parent",
                graph.path
            )));
        }
        seen.insert(graph.path.components());
        previous = Some(&graph.path);
        let node_floor = next_id_after_known_max(
            &graph.path,
            "node",
            graph.nodes.iter().map(|node| node.id.as_u64()),
        )?;
        let edge_floor = next_id_after_known_max(
            &graph.path,
            "edge",
            graph.edges.iter().map(|edge| edge.id.as_u64()),
        )?;
        if graph.next_node_id < node_floor || graph.next_edge_id < edge_floor {
            return Err(Error::Serialization(format!(
                "snapshot LPG allocator for {:?} is below required node/edge high-water ({node_floor}, {edge_floor})",
                graph.path
            )));
        }
    }
    Ok(())
}

impl super::GrafeoDB {
    /// Validates complete, tier-merged LPG history against the decoded
    /// RDF→LPG projection registry.
    ///
    /// Current LPG4 containers and all newly captured state retain exact
    /// structural, label, property, and incident-edge history. Validate that
    /// history after compact/layered wiring and before the database becomes
    /// observable or a new artifact is published.
    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    pub(super) fn validate_live_rdf_lpg_projection_rows(&self) -> Result<()> {
        let (nodes, edges) = self.collect_default_graph_snapshot();
        let graph_cut =
            grafeo_core::graph::lpg::LpgStoreSection::new(std::sync::Arc::clone(self.store_arc()))
                .capture_graphs()?;
        let named_graphs = graph_cut
            .into_iter()
            .skip(1)
            .map(|(path, store)| SnapshotLpgGraph {
                incarnation: store.graph_incarnation_id(),
                path,
                next_node_id: store.peek_next_node_id(),
                next_edge_id: store.peek_next_edge_id(),
                retained_history_floor: store.retained_history_floor().as_u64(),
                nodes: collect_snapshot_nodes(&store),
                edges: collect_snapshot_edges(&store),
            })
            .collect::<Vec<_>>();
        validate_rdf_lpg_projection_rows(
            &nodes,
            &edges,
            &named_graphs,
            &self.rdf_projections.snapshot(),
        )?;
        Ok(())
    }

    /// Collects a complete immutable snapshot while the caller holds the RDF
    /// commit gate followed by the publication barrier. This intentionally
    /// acquires neither lock itself: save and export have different lifecycle
    /// requirements, and recursive acquisition can deadlock behind a writer.
    fn collect_snapshot_at_publication(&self) -> Result<Snapshot> {
        let catalog = self.catalog.read();
        self.collect_snapshot_with_catalog(catalog.view())
    }

    fn collect_snapshot_with_catalog(&self, catalog: CatalogRead<'_>) -> Result<Snapshot> {
        let epoch = self.transaction_manager.current_epoch().as_u64();
        #[cfg(feature = "lpg")]
        let (graphs, graph_cut) = {
            let graph_cut = grafeo_core::graph::lpg::LpgStoreSection::new(std::sync::Arc::clone(
                self.store_arc(),
            ))
            .capture_graphs()?;
            let mut graphs = Vec::new();
            graphs.try_reserve_exact(graph_cut.len()).map_err(|error| {
                Error::Serialization(format!("allocate portable graph cut: {error}"))
            })?;
            for (path, store) in &graph_cut {
                let (nodes, edges) = if *path == GraphPath::root() {
                    self.collect_default_graph_snapshot()
                } else {
                    (collect_snapshot_nodes(store), collect_snapshot_edges(store))
                };
                if self.config.graph_model == GraphModel::Rdf {
                    if *path != GraphPath::root() || !nodes.is_empty() || !edges.is_empty() {
                        return Err(Error::Serialization(
                            "RDF database carries unexpected LPG history".into(),
                        ));
                    }
                    continue;
                }
                graphs.push(SnapshotLpgGraph {
                    incarnation: store.graph_incarnation_id(),
                    path: path.clone(),
                    nodes,
                    edges,
                    next_node_id: store.peek_next_node_id(),
                    next_edge_id: store.peek_next_edge_id(),
                    retained_history_floor: if *path == GraphPath::root() {
                        #[cfg(feature = "compact-store")]
                        {
                            self.layered_store.as_ref().map_or_else(
                                || store.retained_history_floor().as_u64(),
                                |layered| {
                                    layered
                                        .overlay_store()
                                        .retained_history_floor()
                                        .as_u64()
                                        .max(
                                            layered
                                                .base_store_arc()
                                                .property_history_floor()
                                                .unwrap_or(EpochId::new(epoch))
                                                .as_u64(),
                                        )
                                },
                            )
                        }
                        #[cfg(not(feature = "compact-store"))]
                        {
                            store.retained_history_floor().as_u64()
                        }
                    } else {
                        store.retained_history_floor().as_u64()
                    },
                });
            }

            (graphs, graph_cut)
        };
        #[cfg(not(feature = "lpg"))]
        let graphs = Vec::new();
        #[cfg(feature = "triple-store")]
        let (world_identity, rdf_dataset_history) =
            if matches!(self.config.graph_model, GraphModel::Rdf | GraphModel::Both) {
                let history =
                    self.rdf_store
                        .dataset_history_under_commit_gate()
                        .map_err(|error| {
                            Error::Serialization(format!("capture RDF dataset history: {error}"))
                        })?;
                let identity =
                    WorldIdentityMetadataV1::new(history.store_id(), history.completeness())
                        .map_err(|error| {
                            Error::Serialization(format!("capture RDF identity metadata: {error}"))
                        })?;
                if identity != self.world_identity() {
                    return Err(Error::Internal(
                        "database and RDF store identity metadata diverged".to_string(),
                    ));
                }
                let payload = encode_rdf_dataset_history(&history)?;
                (identity, payload)
            } else {
                (self.world_identity(), Vec::new())
            };
        #[cfg(not(feature = "triple-store"))]
        let (world_identity, rdf_dataset_history) = (self.world_identity(), Vec::new());

        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let rdf_lpg_projections = if self.config.graph_model == GraphModel::Both {
            encode_rdf_lpg_projections(&self.rdf_projections)?
        } else if self.rdf_projections.snapshot().is_empty() {
            Vec::new()
        } else {
            return Err(Error::Serialization(
                "RDF→LPG projection metadata requires GraphModel::Both".to_string(),
            ));
        };
        #[cfg(not(all(feature = "triple-store", feature = "lpg")))]
        let rdf_lpg_projections = Vec::new();

        #[cfg(feature = "lpg")]
        let (catalog_state, text_indexes, vector_indexes) =
            capture_snapshot_indexes(&self.catalog, catalog, graph_cut, epoch)?;
        #[cfg(not(feature = "lpg"))]
        let (catalog_state, text_indexes, vector_indexes) = (
            super::catalog_wire::encode_catalog_read(catalog, epoch)?,
            Vec::new(),
            Vec::new(),
        );
        #[cfg(feature = "lpg")]
        let next_graph_incarnation_id = self.store_arc().next_graph_incarnation_id();
        #[cfg(not(feature = "lpg"))]
        let next_graph_incarnation_id = 1;
        let snapshot = Snapshot {
            version: SNAPSHOT_VERSION,
            epoch,
            graph_model: self.config.graph_model.as_u8(),
            world_identity,
            next_graph_incarnation_id,
            graphs,
            catalog_state,
            text_indexes,
            vector_indexes,
            rdf_lpg_projections,
            rdf_dataset_history,
            cdc_checkpoint: super::cdc_checkpoint::capture(self, EpochId::new(epoch))?,
        };
        validate_decoded_snapshot(snapshot)
    }

    // =========================================================================
    // ADMIN API: Persistence Control
    // =========================================================================

    /// Creates an independent in-memory fork of this database.
    ///
    /// The fork includes default and named LPG/RDF data, exact temporal
    /// history, exact indexes, canonical owners and allocator high-water. It receives a new logical
    /// `StoreId`; StoreId-derived RDF statement handles are therefore re-keyed.
    /// Projection-owned LPG rows are removed and their mappings become pending
    /// so receipts from the source cannot be misattributed to the fork.
    ///
    /// This is useful for testing or exploratory modifications without changing
    /// the original. Exporting and importing the portable snapshot is O(database)
    /// and requires a quiescent capture; it is not a cheap live view.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy operation fails.
    pub fn to_memory(&self) -> Result<Self> {
        Self::import_snapshot_as_fork(&self.export_snapshot()?)
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
    /// requested currently visible nodes plus every currently visible edge
    /// whose SOURCE is in `node_ids`,
    /// regardless of where the destination lives. NodeIds and EdgeIds
    /// are preserved verbatim so multiple sibling extracts can later
    /// be merged via [`open_multi`](Self::open_multi) without ID
    /// collisions.
    ///
    /// Edges: every currently visible edge whose SOURCE is in `node_ids` is carried,
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
    /// Properties and the full schema catalog are copied from the source.
    /// Canonical index owners and configuration are preserved; physical indexes
    /// are rebuilt against the selected retained corpus. Text scores belong to
    /// this subset. Use [`IndexMergePolicy::RebuildFromUnion`] to reunite subsets.
    ///
    /// # Errors
    ///
    /// Returns an error if any `node_ids` entry does not exist in
    /// `self`, or if copy operations fail.
    #[cfg(feature = "lpg")]
    pub fn extract_subgraph(&self, node_ids: &[NodeId]) -> Result<Self> {
        let _capture = self.acquire_quiescent_capture("extract_subgraph")?;
        // Tier-merged read view: after compact() the base-tier nodes/edges are
        // invisible to lpg_store() (overlay only), which made base nodes error
        // as "does not exist" and dropped promoted nodes' base edges.
        let store = self.read_graph_view();

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

        let epoch = self.transaction_manager.current_epoch();
        let graph_cut =
            grafeo_core::graph::lpg::LpgStoreSection::new(std::sync::Arc::clone(self.store_arc()))
                .capture_graphs()?;
        let catalog = self.catalog.read();
        let catalog_bytes = super::catalog_section::CatalogSection::new_with_graphs(
            std::sync::Arc::clone(&self.catalog),
            graph_cut.clone(),
            || 0,
        )?
        .serialize_from_read(catalog.view(), epoch.as_u64())?;
        let staged_catalog =
            crate::database::catalog_wire::decode_catalog(&catalog_bytes, epoch.as_u64())?;
        drop(catalog);

        // Preserve the existing selection: live root nodes and their live outgoing
        // edges. Capture every retained version of those identities, without
        // cloning unrelated source rows or any source physical search image.
        let mut selected_nodes: Vec<_> = requested.iter().copied().collect();
        selected_nodes.sort_unstable();
        let mut selected_edges = Vec::new();
        for id in &selected_nodes {
            selected_edges.extend(
                store
                    .edges_from(*id, grafeo_core::graph::Direction::Outgoing)
                    .into_iter()
                    .filter_map(|(_, edge_id)| store.get_edge(edge_id).map(|_| edge_id)),
            );
        }
        selected_edges.sort_unstable();
        selected_edges.dedup();
        let (nodes, edges) =
            self.collect_selected_default_graph_snapshot(&selected_nodes, &selected_edges);
        let target = Self::new_in_memory();
        let target_store = target.store_arc();
        populate_store_from_snapshot_ref(target_store, &nodes, &[])?;
        for edge in &edges {
            if requested.contains(&edge.dst) {
                populate_store_from_snapshot_ref(target_store, &[], std::slice::from_ref(edge))?;
            } else {
                let labels = store
                    .get_node(edge.dst)
                    .map(|node| node.labels.into_iter().collect::<Vec<_>>())
                    .unwrap_or_default();
                let receipt = target_store
                    .restore_transport_edge_history_exact_with_destination_labels(
                        edge.id,
                        edge.src,
                        edge.dst,
                        &edge.edge_type,
                        &edge.lifetimes,
                        &labels,
                    )
                    .map_err(|error| {
                        Error::Serialization(format!(
                            "extract structural history for edge {}: {error}",
                            edge.id,
                        ))
                    })?;
                for (key, versions) in &edge.properties {
                    for (version_epoch, value) in versions {
                        target_store.set_edge_property_at_epoch(
                            edge.id,
                            key,
                            value.clone(),
                            *version_epoch,
                        );
                    }
                }
                target
                    .transport_extract_receipts
                    .lock()
                    .insert(edge.id, receipt);
            }
        }
        for (path, source_graph) in &graph_cut {
            let graph = create_snapshot_graph(target_store, path)?;
            graph.sync_epoch(epoch);
            apply_lpg_allocator_record(
                &graph,
                path,
                source_graph.peek_next_node_id(),
                source_graph.peek_next_edge_id(),
            )?;
            let floor = source_graph.retained_history_floor();
            #[cfg(feature = "compact-store")]
            let floor = if *path == GraphPath::root() {
                self.layered_store.as_ref().map_or(floor, |layered| {
                    floor.max(
                        layered
                            .base_store_arc()
                            .property_history_floor()
                            .unwrap_or(epoch),
                    )
                })
            } else {
                floor
            };
            apply_lpg_history_floor(&graph, path, floor.as_u64(), epoch.as_u64())?;
        }
        target.transaction_manager.try_sync_epoch(epoch)?;
        install_rebuilt_snapshot_catalog(&target, staged_catalog)?;

        Ok(target)
    }

    /// Delete every edge whose destination node does not exist in this
    /// database. Returns the number of edges deleted.
    ///
    /// Intended use: after [`extract_subgraph`](Self::extract_subgraph),
    /// the resulting database carries every outgoing edge from in-set
    /// source nodes (source-side ownership), including edges to dst
    /// nodes outside the subgraph. Those dangling-dst edges fail
    /// [`import_snapshot`](Self::import_snapshot) validation at reopen.
    /// Call `remove_orphan_edges()` before
    /// [`export_snapshot`](Self::export_snapshot) to make the subgraph
    /// self-consistent.
    ///
    /// Symmetric semantic: only DST is checked. Source nodes of any
    /// carried edge are always in-set by `extract_subgraph`'s
    /// source-side ownership contract.
    #[cfg(feature = "lpg")]
    pub fn remove_orphan_edges(&self) -> usize {
        // Tier-merged read view so a base-tier dst node post-compact is not
        // mistaken for a missing node (which would delete a valid edge).
        let store = self.read_graph_view();
        let mut receipt_map = self.transport_extract_receipts.lock();
        let orphans: Vec<EdgeId> = receipt_map
            .values()
            .filter(|receipt| store.get_node(receipt.destination()).is_none())
            .map(|receipt| receipt.edge_id())
            .collect();
        let mut receipts = Vec::with_capacity(orphans.len());
        for edge_id in &orphans {
            if let Some(receipt) = receipt_map.remove(edge_id) {
                receipts.push(receipt);
            }
        }

        if receipts.is_empty() {
            return 0;
        }

        let target_store = self.store_arc();
        for edge_id in &orphans {
            // Transport purge deliberately accepts only an exact committed-
            // closed lifetime. A prior interrupted attempt may already have
            // closed it, so `false` here is not itself a purge failure.
            target_store.delete_edge(*edge_id);
        }
        let receipt_refs: Vec<_> = receipts.iter().collect();
        if target_store.purge_transport_extract_edges(&receipt_refs) {
            receipts.len()
        } else {
            for receipt in receipts {
                receipt_map.insert(receipt.edge_id(), receipt);
            }
            0
        }
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
    /// Default-graph nodes/edges for a snapshot, **tier-merged**.
    ///
    /// Without compaction this is the existing full-history collection over the
    /// built-in store. After `compact()` complete structural, label, and
    /// property histories are read from the pinned layered generation, including
    /// identities retained only in temporal sidecars.
    #[cfg(feature = "compact-store")]
    #[cfg(feature = "lpg")]
    fn collect_default_graph_snapshot(&self) -> (Vec<SnapshotNode>, Vec<SnapshotEdge>) {
        self.collect_default_graph_snapshot_selection(None)
    }

    #[cfg(feature = "compact-store")]
    #[cfg(feature = "lpg")]
    fn collect_default_graph_snapshot_selection(
        &self,
        selected: Option<(&[NodeId], &[EdgeId])>,
    ) -> (Vec<SnapshotNode>, Vec<SnapshotEdge>) {
        if let Some((nodes, edges)) = selected
            && self.layered_store.is_none()
        {
            return (
                collect_snapshot_nodes_selected(self.store_arc(), nodes.to_vec()),
                collect_snapshot_edges_selected(self.store_arc(), edges.to_vec()),
            );
        }
        if self.layered_store.is_none() {
            return (
                collect_snapshot_nodes(self.store_arc()),
                collect_snapshot_edges(self.store_arc()),
            );
        }
        let boundary = self.transaction_manager.current_epoch();
        let layered = self
            .layered_store
            .as_ref()
            .expect("layered store checked above");
        let node_histories = selected.map_or_else(
            || layered.complete_node_histories(),
            |(ids, _)| layered.selected_node_histories(ids),
        );
        let nodes: Vec<_> = node_histories
            .into_iter()
            .filter_map(|(id, history)| {
                let lifetimes: Vec<_> = history
                    .lifetimes
                    .into_iter()
                    .filter(|lifetime| {
                        lifetime.created != EpochId::PENDING && lifetime.created <= boundary
                    })
                    .map(|lifetime| {
                        (
                            lifetime.created,
                            lifetime.deleted.filter(|deleted| {
                                *deleted != EpochId::PENDING && *deleted <= boundary
                            }),
                        )
                    })
                    .collect();
                if lifetimes.is_empty() {
                    return None;
                }
                let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = history
                    .properties
                    .into_iter()
                    .filter_map(|(key, versions)| {
                        let versions: Vec<_> = versions
                            .into_iter()
                            .filter(|(epoch, _)| *epoch != EpochId::PENDING && *epoch <= boundary)
                            .collect();
                        (!versions.is_empty()).then(|| (key.to_string(), versions))
                    })
                    .collect();
                properties.sort_by(|(a, _), (b, _)| a.cmp(b));

                let mut label_versions: Vec<(EpochId, Vec<String>)> = history
                    .label_versions
                    .into_iter()
                    .filter_map(|(epoch, labels)| {
                        let visible = epoch != EpochId::PENDING
                            && epoch <= boundary
                            && lifetimes.iter().any(|(created, deleted)| {
                                *created <= epoch && deleted.is_none_or(|deleted| epoch <= deleted)
                            });
                        visible.then(|| {
                            let mut labels: Vec<_> =
                                labels.into_iter().map(|label| label.to_string()).collect();
                            labels.sort();
                            labels.dedup();
                            (epoch, labels)
                        })
                    })
                    .collect();
                for (created, _) in &lifetimes {
                    if !label_versions.iter().any(|(epoch, _)| epoch == created) {
                        let mut labels: Vec<_> =
                            history.labels.iter().map(ToString::to_string).collect();
                        labels.sort();
                        labels.dedup();
                        label_versions.push((*created, labels));
                    }
                }
                label_versions.sort_by_key(|(epoch, _)| *epoch);
                Some(SnapshotNode {
                    id,
                    lifetimes,
                    label_versions,
                    properties,
                })
            })
            .collect();
        let edge_histories = selected.map_or_else(
            || layered.complete_edge_histories(),
            |(_, ids)| layered.selected_edge_histories(ids),
        );
        let edges: Vec<_> = edge_histories
            .into_iter()
            .filter_map(|(id, history)| {
                let lifetimes: Vec<_> = history
                    .lifetimes
                    .into_iter()
                    .filter(|lifetime| {
                        lifetime.created != EpochId::PENDING && lifetime.created <= boundary
                    })
                    .map(|lifetime| {
                        (
                            lifetime.created,
                            lifetime.deleted.filter(|deleted| {
                                *deleted != EpochId::PENDING && *deleted <= boundary
                            }),
                        )
                    })
                    .collect();
                if lifetimes.is_empty() {
                    return None;
                }
                let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = history
                    .properties
                    .into_iter()
                    .filter_map(|(key, versions)| {
                        let versions: Vec<_> = versions
                            .into_iter()
                            .filter(|(epoch, _)| *epoch != EpochId::PENDING && *epoch <= boundary)
                            .collect();
                        (!versions.is_empty()).then(|| (key.to_string(), versions))
                    })
                    .collect();
                properties.sort_by(|(a, _), (b, _)| a.cmp(b));
                Some(SnapshotEdge {
                    id,
                    src: history.src,
                    dst: history.dst,
                    edge_type: history.edge_type.to_string(),
                    lifetimes,
                    properties,
                })
            })
            .collect();
        (nodes, edges)
    }

    #[cfg(feature = "lpg")]
    fn collect_selected_default_graph_snapshot(
        &self,
        nodes: &[NodeId],
        edges: &[EdgeId],
    ) -> (Vec<SnapshotNode>, Vec<SnapshotEdge>) {
        #[cfg(feature = "compact-store")]
        {
            self.collect_default_graph_snapshot_selection(Some((nodes, edges)))
        }
        #[cfg(not(feature = "compact-store"))]
        {
            (
                collect_snapshot_nodes_selected(self.store_arc(), nodes.to_vec()),
                collect_snapshot_edges_selected(self.store_arc(), edges.to_vec()),
            )
        }
    }

    /// Non-compact builds: always the built-in store with full history.
    #[cfg(not(feature = "compact-store"))]
    #[cfg(feature = "lpg")]
    fn collect_default_graph_snapshot(&self) -> (Vec<SnapshotNode>, Vec<SnapshotEdge>) {
        (
            collect_snapshot_nodes(self.store_arc()),
            collect_snapshot_edges(self.store_arc()),
        )
    }

    /// Captures an integrity-sealed portable snapshot and its truthful world-cut
    /// descriptor. Tier-merged databases are captured from the complete
    /// columnar base plus overlay history, never just the live overlay.
    ///
    /// # Errors
    ///
    /// Returns an error if the database is not quiescent, serialization fails,
    /// or captured metadata disagrees with the snapshot bytes.
    /// Snapshot12 preserves canonical index owners, allocator high-water and
    /// exact Text/Vector images; inconsistent ownership or configuration is
    /// rejected before an artifact is returned.
    pub fn export_snapshot_artifact(&self) -> Result<SnapshotArtifact> {
        let _capture = self.acquire_quiescent_capture("export_snapshot")?;
        let snapshot = self.collect_snapshot_at_publication()?;
        let descriptor = snapshot_world_descriptor(&snapshot)?;
        let bytes = encode_snapshot_bytes(&snapshot)?;
        SnapshotArtifact::new(bytes, descriptor)
            .map_err(|error| Error::Serialization(format!("seal snapshot artifact: {error}")))
    }

    /// Serializes the complete database into portable snapshot v12 bytes.
    /// Exact import preserves the logical StoreId; use
    /// [`import_snapshot_as_fork`](Self::import_snapshot_as_fork) for a new
    /// independently writable logical store.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as
    /// [`export_snapshot_artifact`](Self::export_snapshot_artifact).
    pub fn export_snapshot(&self) -> Result<Vec<u8>> {
        Ok(self.export_snapshot_artifact()?.bytes().to_vec())
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
        Self::import_decoded_snapshot(snapshot)
    }

    /// Imports a manifest-verified snapshot after proving that its descriptor
    /// matches the decoded StoreId, epoch, model, schema and projection state.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bytes, a digest mismatch, or any false
    /// descriptor field.
    pub fn import_snapshot_artifact(artifact: &SnapshotArtifact) -> Result<Self> {
        Self::import_decoded_snapshot(verify_snapshot_artifact(artifact)?)
    }

    /// Imports snapshot bytes as a distinct logical store.
    ///
    /// Ordinary LPG/RDF temporal history is preserved, while StoreId-bound RDF
    /// statement handles are re-keyed and materialized projection rows and
    /// receipts are reset to unpublished mappings for an explicit rebuild.
    ///
    /// # Errors
    ///
    /// Returns an error if the source snapshot is invalid or cannot be safely
    /// transformed into a fork.
    pub fn import_snapshot_as_fork(data: &[u8]) -> Result<Self> {
        let snapshot = reidentify_snapshot_for_fork(decode_snapshot_bytes(data)?)?;
        Self::import_decoded_snapshot(snapshot)
    }

    fn import_decoded_snapshot(snapshot: Snapshot) -> Result<Self> {
        let graph_model = resolved_snapshot_graph_model(&snapshot)?;
        let staged_catalog = stage_snapshot_catalog(&snapshot)?;
        #[cfg(feature = "lpg")]
        let staged_indexes = stage_snapshot_indexes(&snapshot, &staged_catalog)?;
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let projection_definitions = decode_rdf_lpg_projections(
            &snapshot.rdf_lpg_projections,
            snapshot.world_identity.store_id(),
        )?;

        // Validate default graph data
        validate_snapshot_data(snapshot.nodes()?, snapshot.edges()?)?;

        // Validate each named graph
        for ng in snapshot.named_graphs() {
            validate_snapshot_data(&ng.nodes, &ng.edges)?;
        }

        // Stage and validate every RDF term, transaction interval, and valid
        // interval before constructing or mutating the detached destination.
        #[cfg(feature = "triple-store")]
        let staged_rdf_history = stage_snapshot_rdf_history(&snapshot)?;

        let db = Self::with_config(
            Config::in_memory()
                .with_graph_model(graph_model)
                .with_world_identity(snapshot.world_identity.clone()),
        )?;
        #[cfg(feature = "lpg")]
        populate_snapshot_graphs(db.store_arc(), &snapshot)?;
        let snapshot_epoch = EpochId::new(snapshot.epoch);
        db.transaction_manager
            .try_sync_epoch(snapshot_epoch)
            .map_err(|error| Error::Serialization(format!("invalid snapshot epoch: {error}")))?;
        #[cfg(feature = "lpg")]
        db.lpg_store().sync_epoch(snapshot_epoch);

        // Restore RDF triples
        #[cfg(feature = "triple-store")]
        {
            install_snapshot_rdf_history(&db.rdf_store, staged_rdf_history, snapshot_epoch, None)?;
            db.sync_world_identity_from_rdf()?;
        }

        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        restore_rdf_lpg_projections(
            &db.rdf_projections,
            db.rdf_store.store_id(),
            projection_definitions,
        )?;

        // Catalog and index restoration is authoritative and fail-closed. It
        // runs after data population so physical indexes can be rebuilt, but
        // this database is still detached and is discarded on any failure.
        #[cfg(feature = "lpg")]
        install_snapshot_catalog_and_indexes(&db, staged_catalog, &staged_indexes)?;
        #[cfg(not(feature = "lpg"))]
        {
            let mut workspace = CatalogWorkspace::replacement(staged_catalog);
            db.catalog
                .prepare_replacement(&mut workspace)
                .map_err(|error| Error::Serialization(error.to_string()))?
                .install()
                .finish();
        }

        #[cfg(feature = "cdc")]
        super::cdc_checkpoint::prepare(
            &snapshot.cdc_checkpoint,
            snapshot.world_identity.store_id(),
            snapshot_epoch,
        )?
        .install(&db.cdc_log);

        Ok(db)
    }

    /// Creates a new in-memory database by merging multiple binary snapshots.
    ///
    /// Each blob in `snapshots` must have been produced by
    /// [`export_snapshot()`](Self::export_snapshot). Snapshots are unioned
    /// into one database:
    /// - Nodes and edges are preserved with their producer-allocated IDs.
    /// - A NodeId (or EdgeId) appearing in two snapshots is rejected as a
    ///   producer bug; the caller must emit disjoint subsets.
    /// - Every edge endpoint must exist somewhere in the union; a chunk may
    ///   carry edges whose endpoints belong to a different chunk.
    /// - Schema catalogs are reconciled per
    ///   [`OpenMultiOptions::schema_policy`] (default
    ///   [`SchemaMergePolicy::UnionWithConflictCheck`]): same-named types
    ///   must match; incompatible definitions are rejected.
    /// - Catalog index IDs, names, typed keys and resolved configurations are
    ///   preserved. Independent owners must have disjoint IDs; an ID/name/key
    ///   conflict is rejected rather than renumbered. Shared Text/Vector owner
    ///   images must be byte-identical; exact HNSW/history images are not rebuilt
    ///   or concatenated. The epoch is the maximum across inputs.
    /// - The result is always a logical fork with a fresh [`StoreId`]. Exact
    ///   replicas use [`import_snapshot`](Self::import_snapshot) instead.
    /// - At most one input may declare an RDF-bearing model (`Rdf` or `Both`).
    ///   Its complete graph-lifecycle and quad history is preserved, while
    ///   stable statement handles are re-keyed to the fresh result identity.
    /// - The result model is the union of the declared input models. For
    ///   example, an LPG input plus one RDF input produces `Both`.
    /// - RDF→LPG projection mappings survive only as unpublished `Pending`
    ///   definitions. Store-bound receipts and materialized projection rows
    ///   are removed and must be rebuilt explicitly in the result.
    /// - LPG allocator high-water marks are merged exactly, including gaps
    ///   left by reserved, aborted, or removed IDs.
    ///
    /// All validation runs before any data is inserted; a rejection
    /// leaves no partial database behind (the function never publishes
    /// `self`).
    ///
    /// # Errors
    ///
    /// Returns an error if `snapshots` is empty, any blob fails to decode,
    /// more than one input declares RDF ownership, any cross-snapshot
    /// validation fails, or detached population fails.
    ///
    #[cfg(feature = "lpg")]
    pub fn open_multi<I, B>(snapshots: I) -> Result<Self>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        Self::open_multi_with(snapshots, OpenMultiOptions::default())
    }

    /// Variant of [`open_multi`](Self::open_multi) that takes an
    /// explicit [`OpenMultiOptions`] for callers who need a non-default
    /// schema-merge policy.
    ///
    /// # Errors
    ///
    /// Returns an error if `snapshots` is empty, in addition to the
    /// conditions listed on [`open_multi`](Self::open_multi).
    ///
    #[cfg(feature = "lpg")]
    pub fn open_multi_with<I, B>(snapshots: I, options: OpenMultiOptions) -> Result<Self>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let owned: Vec<B> = snapshots.into_iter().collect();
        if owned.is_empty() {
            return Err(Error::Internal(
                "open_multi requires at least one snapshot blob".to_string(),
            ));
        }

        let _span = grafeo_info_span!("open_multi", n_snapshots = owned.len());

        // Decode every blob first so any version / bincode failure
        // surfaces before we touch a target database.
        let decoded: Vec<Snapshot> = {
            let _decode = grafeo_debug_span!("decode");
            owned
                .iter()
                .enumerate()
                .map(|(idx, bytes)| {
                    decode_snapshot_bytes(bytes.as_ref())
                        .map_err(|e| Error::Internal(format!("snapshot[{idx}]: {e}")))
                })
                .collect::<Result<_>>()?
        };

        let cdc_owners = decoded
            .iter()
            .enumerate()
            .filter_map(|(index, snapshot)| {
                super::cdc_checkpoint::is_enabled(&snapshot.cdc_checkpoint).then_some(index)
            })
            .collect::<Vec<_>>();
        if cdc_owners.len() > 1 {
            return Err(Error::Serialization(
                "open_multi requires at most one authoritative retained CDC feed".into(),
            ));
        }
        let source_models = decoded
            .iter()
            .map(resolved_snapshot_graph_model)
            .collect::<Result<Vec<_>>>()?;
        let rdf_owners = source_models
            .iter()
            .enumerate()
            .filter_map(|(index, model)| {
                matches!(model, GraphModel::Rdf | GraphModel::Both).then_some(index)
            })
            .collect::<Vec<_>>();
        if rdf_owners.len() > 1 {
            return Err(Error::Serialization(format!(
                "snapshots {rdf_owners:?} declare RDF/Both ownership; open_multi requires at most one authoritative RDF owner"
            )));
        }
        let rdf_owner = rdf_owners.first().copied();
        let has_lpg = source_models
            .iter()
            .any(|model| matches!(model, GraphModel::Lpg | GraphModel::Both));
        let has_rdf = rdf_owner.is_some();
        let result_graph_model = match (has_lpg, has_rdf) {
            (true, true) => GraphModel::Both,
            (true, false) => GraphModel::Lpg,
            (false, true) => GraphModel::Rdf,
            (false, false) => {
                return Err(Error::Serialization(
                    "open_multi inputs declare no supported graph model".to_string(),
                ));
            }
        };

        // `open_multi` is a fork/union constructor, never an exact replica.
        // Every input is normalized under one fresh identity before cross-input
        // validation so projection-owned rows are removed before ID collision
        // checks, while their allocator gaps remain authoritative.
        let target_store_id = generate_snapshot_fork_store_id(&decoded)?;
        let target_history = rdf_owner.map_or(HistoryCompleteness::Complete, |index| {
            decoded[index].world_identity.history()
        });
        let target_world_identity = WorldIdentityMetadataV1::new(target_store_id, target_history)
            .map_err(|error| {
            Error::Serialization(format!("create open_multi fork identity: {error}"))
        })?;
        let decoded = decoded
            .into_iter()
            .enumerate()
            .map(|(index, snapshot)| {
                reidentify_snapshot_for_fork_to(snapshot, target_store_id).map_err(|error| {
                    Error::Serialization(format!(
                        "snapshot[{index}] fork normalization failed: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let decoded_projections = decoded
            .iter()
            .map(|snapshot| {
                decode_rdf_lpg_projections(
                    &snapshot.rdf_lpg_projections,
                    snapshot.world_identity.store_id(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "triple-store")]
        let staged_rdf_history = rdf_owner
            .map(|index| stage_snapshot_rdf_history(&decoded[index]))
            .transpose()?;

        // Per-snapshot duplicate-ID validation only (not endpoint
        // resolution — niche chunks can reference nodes from other
        // snapshots). Cross-snapshot endpoint validation runs next via
        // validate_snapshot_set.
        {
            let _validate = grafeo_debug_span!("validate");
            for (idx, snap) in decoded.iter().enumerate() {
                validate_snapshot_ids(snap.nodes()?, snap.edges()?)
                    .map_err(|e| Error::Internal(format!("snapshot[{idx}]: {e}")))?;
                for graph in snap.named_graphs() {
                    validate_snapshot_data(&graph.nodes, &graph.edges).map_err(|error| {
                        Error::Internal(format!(
                            "snapshot[{idx}] named graph {:?}: {error}",
                            graph.path
                        ))
                    })?;
                }
            }
            validate_snapshot_set(&decoded)?;

            let mut named_graph_owners = HashMap::<&GraphPath, usize>::new();
            for (index, snapshot) in decoded.iter().enumerate() {
                for graph in snapshot.named_graphs() {
                    if let Some(previous) = named_graph_owners.insert(&graph.path, index) {
                        let prior = snapshot_allocator_record(&decoded[previous], &graph.path)?;
                        if options.index_policy == IndexMergePolicy::RebuildFromUnion
                            && graph.nodes.is_empty()
                            && graph.edges.is_empty()
                            && prior.nodes.is_empty()
                            && prior.edges.is_empty()
                        {
                            continue;
                        }
                        return Err(Error::Internal(format!(
                            "named graph {:?} appears in snapshot[{previous}] and snapshot[{index}]; open_multi requires named graphs to be disjoint across snapshots",
                            graph.path
                        )));
                    }
                }
            }
        }

        // Reconcile schemas per the chosen policy. The merged schema
        // is what gets restored to the target catalog below.
        let merged_schema = {
            let _schema = grafeo_debug_span!("schema");
            merge_snapshot_schemas(&decoded, options.schema_policy)?
        };
        let merged_catalog = merge_snapshot_catalogs(&decoded, &merged_schema)?;
        let merged_indexes = match options.index_policy {
            IndexMergePolicy::ExactState => {
                Some(merge_snapshot_indexes(&decoded, &merged_catalog)?)
            }
            IndexMergePolicy::RebuildFromUnion => {
                for snapshot in &decoded {
                    validate_snapshot_indexes_for_rebuild(snapshot)?;
                }
                None
            }
        };

        // Build the merged database from the decoded snapshots under the one
        // preflighted fork identity. It remains detached until this function
        // returns successfully.
        let db = Self::with_config(
            Config::in_memory()
                .with_graph_model(result_graph_model)
                .with_world_identity(target_world_identity),
        )?;

        // Reserve the feed owner's original native lifetimes before creating
        // any union-only graph. Retired IDs below its allocator floor must not
        // be reused for a different path merely because this is a new StoreId.
        #[cfg(feature = "cdc")]
        if let Some(&index) = cdc_owners.first() {
            let owner = &decoded[index];
            if !owner.graphs.is_empty() {
                let root = std::sync::Arc::new(db.store_arc().new_live_replacement_candidate()?);
                for graph in &owner.graphs {
                    create_snapshot_graph(&root, &graph.path)?;
                }
                let mut root = std::sync::Arc::try_unwrap(root).map_err(|_| {
                    Error::Internal("detached feed topology retained an unexpected alias".into())
                })?;
                root.restore_graph_incarnations(
                    &owner.incarnations(),
                    owner.next_graph_incarnation_id,
                )?;
                db.store_arc().install_pristine_image(root)?;
            }
        }

        // Default graph: publish the complete node union before any edge. A
        // source-owned transport edge may resolve its destination in a sibling
        // snapshot, so per-snapshot node+edge population is order-dependent.
        {
            let _populate = grafeo_debug_span!("populate");
            for snap in &decoded {
                populate_store_from_snapshot_ref(db.store_arc(), snap.nodes()?, &[])?;
            }
            for snap in &decoded {
                populate_store_from_snapshot_ref(db.store_arc(), &[], snap.edges()?)?;
            }
        }

        // Populated named graphs remain disjoint. Explicit index rebuilding
        // also permits the validated repeated empty topology of extracts.
        {
            let _named_graphs = grafeo_debug_span!("named_graphs");
            let mut populated_paths = HashSet::new();
            for snap in &decoded {
                for graph in snap.named_graphs() {
                    let first = populated_paths.insert(graph.path.clone());
                    let graph_store = if first {
                        create_snapshot_graph(db.store_arc(), &graph.path)?
                    } else {
                        let mut store = std::sync::Arc::clone(db.store_arc());
                        for component in graph.path.components() {
                            store = store.graph(component).ok_or_else(|| {
                                Error::Internal("validated empty union topology is missing".into())
                            })?;
                        }
                        store
                    };
                    populate_store_from_snapshot_ref(&graph_store, &graph.nodes, &graph.edges)?;
                    apply_lpg_allocator_record(
                        &graph_store,
                        &graph.path,
                        graph.next_node_id.max(graph_store.peek_next_node_id()),
                        graph.next_edge_id.max(graph_store.peek_next_edge_id()),
                    )?;
                    apply_lpg_history_floor(
                        &graph_store,
                        &graph.path,
                        graph.retained_history_floor,
                        snap.epoch,
                    )?;
                }
            }

            // RDF history comes from the one declared owner selected before
            // any detached population. Empty RDF datasets still own lineage.
            #[cfg(feature = "triple-store")]
            if let Some(staged) = staged_rdf_history {
                let index = rdf_owner.ok_or_else(|| {
                    Error::Internal(
                        "staged RDF history has no declared open_multi owner".to_string(),
                    )
                })?;
                let snapshot = &decoded[index];
                install_snapshot_rdf_history(
                    &db.rdf_store,
                    staged,
                    EpochId::new(snapshot.epoch),
                    None,
                )?;
                db.sync_world_identity_from_rdf()?;
                restore_rdf_lpg_projections(
                    &db.rdf_projections,
                    db.rdf_store.store_id(),
                    decoded_projections[index].clone(),
                )?;
            }
        }

        // The default LPG allocator dominates every LPG-bearing shard, while
        // each disjoint named graph retains its sole owner's exact counter.
        // RDF-only snapshots intentionally carry no LPG allocator record.
        if has_lpg {
            let lpg_snapshots = decoded
                .iter()
                .zip(&source_models)
                .filter_map(|(snapshot, model)| {
                    matches!(model, GraphModel::Lpg | GraphModel::Both).then_some(snapshot)
                })
                .collect::<Vec<_>>();
            let mut allocator_states = lpg_snapshots
                .iter()
                .map(|snapshot| snapshot_allocator_record(snapshot, &GraphPath::root()));
            let first = allocator_states
                .next()
                .ok_or_else(|| Error::Internal("LPG union has no allocator owner".to_string()))??;
            let (merged_next_node, merged_next_edge) = allocator_states.try_fold(
                (first.next_node_id, first.next_edge_id),
                |(next_node, next_edge), state| {
                    let state = state?;
                    Ok::<_, Error>((
                        next_node.max(state.next_node_id),
                        next_edge.max(state.next_edge_id),
                    ))
                },
            )?;
            apply_lpg_allocator_record(
                db.store_arc(),
                &GraphPath::root(),
                merged_next_node,
                merged_next_edge,
            )?;
            let merged_history_floor = lpg_snapshots
                .iter()
                .map(|snapshot| {
                    snapshot_allocator_record(snapshot, &GraphPath::root())
                        .map(|graph| graph.retained_history_floor)
                })
                .try_fold(0_u64, |floor, value| Ok::<_, Error>(floor.max(value?)))?;
            let merged_epoch = lpg_snapshots
                .iter()
                .map(|snapshot| snapshot.epoch)
                .max()
                .ok_or_else(|| Error::Internal("LPG union has no committed epoch".to_string()))?;
            apply_lpg_history_floor(
                db.store_arc(),
                &GraphPath::root(),
                merged_history_floor,
                merged_epoch,
            )?;
        }

        // Restore epoch, schema, and indexes.
        {
            let _indexes = grafeo_debug_span!("indexes");

            // Restore epoch as max across snapshots.
            {
                let max_epoch = decoded.iter().map(|s| s.epoch).max().ok_or_else(|| {
                    Error::Internal("snapshot union has no committed epoch".to_string())
                })?;
                let epoch = EpochId::new(max_epoch);
                db.transaction_manager
                    .try_sync_epoch(epoch)
                    .map_err(|error| {
                        Error::Serialization(format!("invalid open_multi snapshot epoch: {error}"))
                    })?;
                db.lpg_store().sync_epoch(epoch);
                for (_, graph) in grafeo_core::graph::lpg::LpgStoreSection::new(
                    std::sync::Arc::clone(db.store_arc()),
                )
                .capture_graphs()?
                {
                    graph.sync_epoch(epoch);
                }
                #[cfg(feature = "triple-store")]
                db.rdf_store.try_set_commit_epoch(epoch).map_err(|error| {
                    Error::Serialization(format!("invalid open_multi RDF snapshot epoch: {error}"))
                })?;
            }

            match merged_indexes {
                Some(indexes) => {
                    install_snapshot_catalog_and_indexes(&db, merged_catalog, &indexes)?;
                }
                None => install_rebuilt_snapshot_catalog(&db, merged_catalog)?,
            }
        }

        grafeo_info!(
            "open_multi complete: nodes={nodes} edges={edges}",
            nodes = db.node_count(),
            edges = db.edge_count()
        );

        #[cfg(feature = "cdc")]
        if let Some(&index) = cdc_owners.first() {
            let snapshot = &decoded[index];
            let prepared = super::cdc_checkpoint::prepare(
                &snapshot.cdc_checkpoint,
                snapshot.world_identity.store_id(),
                EpochId::new(snapshot.epoch),
            )?;
            let graph_cut = grafeo_core::graph::lpg::LpgStoreSection::new(std::sync::Arc::clone(
                db.store_arc(),
            ))
            .capture_graphs()?;
            let owners = graph_cut
                .iter()
                .map(|(path, graph)| (path.clone(), graph.graph_incarnation_id()))
                .collect::<Vec<_>>();
            #[cfg(feature = "triple-store")]
            let rdf_next = db.rdf_store.next_graph_incarnation().as_u64();
            #[cfg(not(feature = "triple-store"))]
            let rdf_next = 0;
            prepared.validate_native(crate::cdc::checkpoint::NativeCut {
                model: result_graph_model.as_u8(),
                lpg_next: db.store_arc().next_graph_incarnation_id(),
                lpg_owners: &owners,
                rdf_next,
            })?;
            prepared.install(&db.cdc_log);
        }
        Ok(db)
    }

    /// Replaces the current database contents with data from a binary snapshot.
    ///
    /// The `data` must have been produced by
    /// [`export_snapshot()`](Self::export_snapshot).
    ///
    /// Data, exact physical indexes, catalog, RDF history and projection state
    /// are prepared before any live mutation. Installation swaps prepared
    /// state under retained publication/backing guards; a preparation failure
    /// leaves the target unchanged. The root store Arc is preserved.
    ///
    /// Exact replacement is deliberately limited to an unlayered, non-WAL
    /// target at a quiescent committed cut. It refuses a WAL-backed target, a
    /// compacted layered target, any target with active transactions, and any snapshot whose
    /// declared graph model differs from the target. Every such
    /// refusal happens before target data is mutated.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, fails deserialization, or the target is
    /// WAL-backed, compacted, non-quiescent, or model-incompatible.
    pub fn restore_snapshot(&self, data: &[u8]) -> Result<()> {
        #[cfg(feature = "wal")]
        if self.wal.is_some() {
            return Err(Error::Internal(
                "restore_snapshot is not a framed persist path on a WAL-backed database"
                    .to_string(),
            ));
        }

        #[cfg(feature = "compact-store")]
        if self.layered_store.is_some() {
            return Err(Error::Internal(
                "restore_snapshot cannot exactly replace a compacted layered database; reopen an in-memory target first"
                    .to_string(),
            ));
        }

        let snapshot = decode_snapshot_bytes(data)?;
        for graph in &snapshot.graphs {
            validate_snapshot_data(&graph.nodes, &graph.edges)?;
        }
        let staged_catalog = stage_snapshot_catalog(&snapshot)?;
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let projection_definitions = decode_rdf_lpg_projections(
            &snapshot.rdf_lpg_projections,
            snapshot.world_identity.store_id(),
        )?;
        #[cfg(feature = "triple-store")]
        let staged_rdf_history = stage_snapshot_rdf_history(&snapshot)?;

        if GraphModel::from_u8(snapshot.graph_model) != Some(self.config.graph_model) {
            return Err(Error::Serialization(format!(
                "snapshot graph model {} does not match target model {}",
                snapshot.graph_model,
                self.config.graph_model.as_u8()
            )));
        }

        // All candidates and displaced payloads are owned outside the live
        // fences. Decode/build once; do not construct a second database merely
        // to validate and then rebuild it against the live target.
        #[cfg(feature = "cdc")]
        let prepared_cdc = super::cdc_checkpoint::prepare(
            &snapshot.cdc_checkpoint,
            snapshot.world_identity.store_id(),
            EpochId::new(snapshot.epoch),
        )?;
        #[cfg(feature = "lpg")]
        let candidate = build_snapshot_graphs(self.store_arc(), &snapshot)?;
        let snapshot_epoch = EpochId::new(snapshot.epoch);
        #[cfg(feature = "lpg")]
        let mut data_workspace = grafeo_core::graph::lpg::LpgReplacementWorkspace::new(
            std::sync::Arc::clone(self.store_arc()),
            candidate,
        )?;
        #[cfg(feature = "lpg")]
        let anchors = data_workspace.graphs().to_vec();
        #[cfg(feature = "lpg")]
        let mut indexes;
        let mut catalog_workspace;
        let mut clock_workspace = crate::transaction::SnapshotClockWorkspace::default();
        let mut incoming_world = snapshot.world_identity.clone();
        #[cfg(feature = "triple-store")]
        let mut rdf_workspace = self
            .transaction_manager
            .with_write_authority(|| {
                self.rdf_store
                    .prepare_dataset_history_replacement(staged_rdf_history, snapshot_epoch)
            })
            .map_err(|error| {
                Error::Serialization(format!("prepare RDF snapshot replacement: {error}"))
            })?;
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let mut projection_workspace = self
            .rdf_projections
            .prepare_restore_for_store(incoming_world.store_id(), projection_definitions)
            .map_err(|error| {
                Error::Serialization(format!("prepare RDF projection replacement: {error}"))
            })?;

        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let _projection_rebuild = self.rdf_projections.lock_rebuild();
        let database_open = self.is_open.write();
        if !*database_open || self.read_only {
            return Err(Error::Internal(
                "restore_snapshot requires an open writable database".into(),
            ));
        }
        self.require_quiescent("restore_snapshot")?;
        #[cfg(feature = "triple-store")]
        let _rdf_gate = self.rdf_store.lock_commit_scoped();
        let _publication = self.transaction_manager.publication().write();
        self.require_quiescent("restore_snapshot")?;
        // Observe the complete existing registry only after query/DDL writers
        // are excluded: a new target-only index must not escape the removal diff.
        #[cfg(feature = "lpg")]
        {
            indexes = live_restore_indexes::stage_live_restore_indexes(
                &snapshot,
                &staged_catalog,
                data_workspace.candidate(),
                &anchors,
            )?;
        }
        catalog_workspace = CatalogWorkspace::replacement(staged_catalog);
        let ready_clock = self.transaction_manager.prepare_snapshot_clock(
            &mut clock_workspace,
            snapshot_epoch,
            &_publication,
        )?;
        let ready_catalog = self
            .catalog
            .prepare_replacement(&mut catalog_workspace)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        #[cfg(feature = "triple-store")]
        let ready_rdf = self
            .transaction_manager
            .with_write_authority(|| rdf_workspace.ready(&_rdf_gate))
            .map_err(|error| {
                Error::Serialization(format!("qualify RDF snapshot replacement: {error}"))
            })?;
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let ready_projections = projection_workspace.ready().map_err(|error| {
            Error::Serialization(format!("qualify RDF projection replacement: {error}"))
        })?;
        let mut world = self.world_identity.write();
        self.transaction_manager.with_write_authority(|| {
            let install_shared = || {
                let _catalog = ready_catalog.install();
                let _clock = ready_clock.install();
                #[cfg(feature = "triple-store")]
                let _rdf = ready_rdf.install();
                #[cfg(all(feature = "triple-store", feature = "lpg"))]
                let _projections = ready_projections.install();
                #[cfg(feature = "cdc")]
                prepared_cdc.install(&self.cdc_log);
                std::mem::swap(&mut *world, &mut incoming_world);
            };
            #[cfg(feature = "lpg")]
            grafeo_core::graph::lpg::with_prepared_lpg_replacement(
                &mut data_workspace,
                &mut indexes,
                || Ok::<(), LiveRestoreError>(()),
                |()| install_shared(),
            )
            .map_err(LiveRestoreError::into_error)?;
            #[cfg(not(feature = "lpg"))]
            {
                install_shared();
            }
            Ok::<(), Error>(())
        })?;
        // Plans are derived state. Retire them after backing/registry guards
        // drain, but before query publication admission is reopened.
        self.clear_plan_cache();

        Ok(())
    }

    // =========================================================================
    // ADMIN API: Iteration
    // =========================================================================

    /// Returns an iterator over all nodes in the database.
    ///
    /// Useful for dump/export operations.
    #[cfg(feature = "lpg")]
    pub fn iter_nodes(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Node> + '_ {
        self.read_all_nodes().into_iter()
    }

    /// Returns an iterator over all edges in the database.
    ///
    /// Useful for dump/export operations.
    #[cfg(feature = "lpg")]
    pub fn iter_edges(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Edge> + '_ {
        self.read_all_edges().into_iter()
    }
}

// =========================================================================
// ADMIN API: Content-addressed cold-base save/load (opt-in, cold base only)
// =========================================================================
//
// This is a NEW opt-in persistence path, separate from `save()`/`.grafeo`.
// It persists ONLY the temporal cold base (`LayeredStore::base_store_arc()`)
// to a directory in a content-addressed layout that deduplicates unchanged
// column value blocks across successive saves. The mutable overlay, indexes,
// catalog, named graphs, and the full-DB snapshot are NOT part of this path —
// for a complete database image keep using the existing `save()`.
//
// On-disk layout (rooted at `dir`):
//   dir/
//     cold_base.section          -- content-addressed section bytes for the
//                                   cold base. Each node value block is replaced
//                                   inline by its 32-byte ContentId; the real
//                                   block bytes live in `blocks/`.
//     blocks/
//       <hex-content-id>          -- one file per distinct column value block,
//                                   named by the lowercase hex of its 32-byte
//                                   ContentId (the file name *is* the content
//                                   address). The directory is the cross-save
//                                   dedup store.
//
// Incremental dedup: a re-save to the SAME `dir` re-serializes the (possibly
// changed) cold base into a fresh pool, then `BlockPool::persist_to` writes
// only the blocks whose ContentId is not already a file in `blocks/` —
// unchanged columns hash identically across base generations and are skipped.
// `save_cold_base_content_addressed` returns the count of blocks newly written.
//
// Load reconstructs the pool by reading every file in `blocks/` and interning
// its bytes (intern recomputes the ContentId), giving a pool that holds every
// block the section references; the section then deserializes against it.

#[cfg(all(feature = "compact-store", feature = "lpg"))]
impl super::GrafeoDB {
    /// Saves ONLY the temporal cold base to `dir` in a content-addressed
    /// layout that deduplicates unchanged column blocks across saves.
    ///
    /// Requires a prior [`compact()`](Self::compact) (the cold base lives in the
    /// [`LayeredStore`](grafeo_core::graph::compact::layered::LayeredStore)).
    /// The mutable overlay and the rest of the database are NOT persisted here —
    /// use [`save()`](Self::save) for a full image.
    ///
    /// Writes `dir/cold_base.section` (section bytes carrying inline 32-byte
    /// content ids) and persists the column value blocks into `dir/blocks/`
    /// (one content-addressed file per block). Re-saving to the SAME `dir`
    /// writes only the blocks whose content id is not already present — unchanged
    /// columns are skipped (cross-save block dedup). Returns the number of blocks
    /// newly written by this save (`0` when nothing changed).
    ///
    /// # Errors
    ///
    /// Returns an error if the database has not been compacted (no layered
    /// store), if the cold base could not be serialized, or if writing the
    /// section bytes / persisting the block pool fails.
    // Native only: backed by the filesystem `FsBlockBackend` (gated
    // `not(target_arch = "wasm32")`). Browser builds persist the cold base via
    // the async OPFS backend in `grafeo-wasm` instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_cold_base_content_addressed(&self, dir: &std::path::Path) -> Result<usize> {
        use grafeo_core::graph::compact::content_dedup::{BlockPool, FsBlockBackend};
        use grafeo_core::graph::compact::section::CompactStoreSection;

        let layered = self.layered_store.as_ref().ok_or_else(|| {
            Error::Internal(
                "save_cold_base_content_addressed() requires a prior compact() (no layered store)"
                    .into(),
            )
        })?;

        // Serialize the cold base content-addressed: value blocks go into the
        // pool, only their content ids stay inline in the section bytes.
        let section = CompactStoreSection::new(layered.base_store_arc());
        let mut pool = BlockPool::new();
        let section_bytes = section.serialize_content_addressed(&mut pool)?;

        std::fs::create_dir_all(dir)
            .map_err(|e| Error::Internal(format!("create_dir_all {}: {e}", dir.display())))?;
        // Crash-atomic section write: write to a temp file, fsync, then rename
        // (same discipline as the WAL checkpoint metadata). A crash mid-write
        // otherwise leaves a truncated `cold_base.section` that fails to parse on
        // load; with temp+rename the previous section stays intact. The content
        // blocks are addressed by hash and written separately/idempotently, so
        // they are never the corruption point.
        let section_path = dir.join("cold_base.section");
        let tmp_path = dir.join("cold_base.section.tmp");
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&tmp_path)
                .map_err(|e| Error::Internal(format!("create {}: {e}", tmp_path.display())))?;
            f.write_all(&section_bytes)
                .map_err(|e| Error::Internal(format!("write {}: {e}", tmp_path.display())))?;
            f.sync_all()
                .map_err(|e| Error::Internal(format!("fsync {}: {e}", tmp_path.display())))?;
        }
        std::fs::rename(&tmp_path, &section_path)
            .map_err(|e| Error::Internal(format!("rename {}: {e}", section_path.display())))?;

        // Persist the pool incrementally: the `blocks/` dir IS the dedup store
        // across saves. `persist_to` writes only blocks the backend doesn't
        // already hold, returning the count newly written.
        let backend = FsBlockBackend::open(dir.join("blocks")).map_err(Error::Internal)?;
        let written = pool.persist_to(&backend).map_err(Error::Internal)?;
        Ok(written)
    }

    /// Loads a temporal cold base previously written by
    /// [`save_cold_base_content_addressed`](Self::save_cold_base_content_addressed)
    /// from `dir`.
    ///
    /// Reconstructs the in-memory block pool by reading every file under
    /// `dir/blocks/` and interning its bytes (interning recomputes the content
    /// id), then deserializes `dir/cold_base.section` against that pool. Because
    /// every block the section references is present in `blocks/`, the pool
    /// always holds every referenced block.
    ///
    /// Returns the reconstructed
    /// [`CompactStore`](grafeo_core::graph::compact::CompactStore) (the cold base
    /// only — the overlay and the rest of the database are not part of this
    /// opt-in path). It is a standalone associated function: loading the cold
    /// base does not require an existing database.
    ///
    /// # Errors
    ///
    /// Returns an error if `dir/cold_base.section` or `dir/blocks/` cannot be
    /// read, or if the section bytes are malformed / reference a block missing
    /// from the pool.
    // Native only: the save side ([`save_cold_base_content_addressed`]) is
    // gated `not(target_arch = "wasm32")`, so its loader is too. Browser builds
    // read the cold base via the async OPFS backend in `grafeo-wasm`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_cold_base_content_addressed(
        dir: &std::path::Path,
    ) -> Result<grafeo_core::graph::compact::CompactStore> {
        use grafeo_core::graph::compact::content_dedup::BlockPool;
        use grafeo_core::graph::compact::section::deserialize_content_addressed;

        let section_path = dir.join("cold_base.section");
        let section_bytes = std::fs::read(&section_path)
            .map_err(|e| Error::Internal(format!("read {}: {e}", section_path.display())))?;

        // Rebuild the pool by interning every block file. Interning recomputes
        // the content id from the bytes, so the pool ends up keyed exactly as
        // the section's inline ids expect; the hex file name need not be parsed.
        let blocks_dir = dir.join("blocks");
        let mut pool = BlockPool::new();
        let entries = std::fs::read_dir(&blocks_dir)
            .map_err(|e| Error::Internal(format!("read_dir {}: {e}", blocks_dir.display())))?;
        for entry in entries {
            let entry = entry
                .map_err(|e| Error::Internal(format!("read_dir {}: {e}", blocks_dir.display())))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&path)
                .map_err(|e| Error::Internal(format!("read {}: {e}", path.display())))?;
            pool.intern(bytes);
        }

        deserialize_content_addressed(&bytes::Bytes::from(section_bytes), &pool)
            .map_err(|e| Error::Internal(format!("deserialize cold base: {e}")))
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    #[test]
    fn hostile_native_graph_coordinates_reject_before_import_and_live_restore()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::GraphIncarnationId;
        let source = GrafeoDB::new_in_memory();
        source.create_graph("a")?;
        source.create_graph("b")?;
        let snapshot = decode_snapshot_bytes(&source.export_snapshot()?)?;
        for case in 0..6 {
            let mut invalid = snapshot.clone();
            match case {
                0 => invalid.graphs[2].incarnation = invalid.graphs[1].incarnation,
                1 => invalid.graphs[1].incarnation = GraphIncarnationId::DEFAULT_GRAPH,
                2 => invalid.graphs[0].incarnation = GraphIncarnationId::new(9),
                3 => invalid.next_graph_incarnation_id = 2,
                4 => invalid.next_graph_incarnation_id = 0,
                _ => invalid.graphs[2].incarnation = GraphIncarnationId::new(u64::MAX),
            }
            let bytes = encode_snapshot_bytes(&invalid)?;
            assert!(snapshot_info(&bytes).is_err());
            assert!(GrafeoDB::import_snapshot(&bytes).is_err());
            for populated in [false, true] {
                let target = GrafeoDB::new_in_memory();
                if populated {
                    target.create_graph("keep")?;
                }
                let before = target.export_snapshot()?;
                assert!(target.restore_snapshot(&bytes).is_err());
                assert_eq!(target.export_snapshot()?, before);
            }
        }
        let mut exhausted = snapshot;
        exhausted.next_graph_incarnation_id = u64::MAX;
        let bytes = encode_snapshot_bytes(&exhausted)?;
        let restored = GrafeoDB::import_snapshot(&bytes)?;
        let before = restored.export_snapshot()?;
        assert!(restored.create_graph("exhausted").is_err());
        assert_eq!(restored.export_snapshot()?, before);
        Ok(())
    }

    #[test]
    fn dropped_index_floor_survives_portable_and_saved_copies()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::catalog::IndexConfiguration;

        let db = GrafeoDB::new_in_memory();
        let node = db.create_node(&["Doc"]);
        let label = db.catalog.get_or_create_label("Doc")?;
        let property = db.catalog.get_or_create_property_key("key")?;
        let owner = db.catalog.create_index(
            None,
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        assert!(db.catalog.drop_index(owner));
        let blank = GrafeoDB::new_in_memory().export_snapshot()?;
        let bytes = db.export_snapshot()?;
        for copied in [GrafeoDB::import_snapshot(&bytes)?, db.to_memory()?] {
            assert!(copied.get_node(node).is_some());
            assert_eq!(copied.catalog.index_count(), 0);
            assert_eq!(copied.catalog.index_allocator_high_water(), 1);
        }
        assert!(db.get_node(node).is_some());
        assert_eq!(db.catalog.index_count(), 0);
        assert_eq!(db.catalog.index_allocator_high_water(), 1);

        #[cfg(feature = "wal")]
        {
            let directory = tempfile::tempdir()?;
            for name in ["exact-floor", "exact-floor.grafeo"] {
                let container = directory.path().join(name);
                db.save(&container)?;
                assert!(container.is_file());
                let restored = GrafeoDB::open(&container)?;
                assert_eq!(restored.catalog.index_count(), 0);
                assert_eq!(restored.catalog.index_allocator_high_water(), 1);
                assert!(restored.get_node(node).is_some());
                assert_eq!(restored.export_snapshot()?, bytes);
                restored.close()?;
            }
        }
        // Extraction preserves canonical owners and the allocator floor, so a
        // dropped owner's id is not reissued in the subset either.
        let extracted = db.extract_subgraph(&[node])?;
        assert!(extracted.get_node(node).is_some());
        assert_eq!(extracted.catalog.index_count(), 0);
        assert_eq!(extracted.catalog.index_allocator_high_water(), 1);
        assert_eq!(db.export_snapshot()?, bytes);
        let retained_root = std::sync::Arc::clone(db.store_arc());
        db.restore_snapshot(&blank)?;
        assert!(std::sync::Arc::ptr_eq(&retained_root, db.store_arc()));
        assert!(db.get_node(node).is_none());
        assert_eq!(db.catalog.index_count(), 0);
        assert_eq!(db.catalog.index_allocator_high_water(), 0);
        assert_eq!(db.export_snapshot()?, blank);
        db.restore_snapshot(&bytes)?;
        assert!(std::sync::Arc::ptr_eq(&retained_root, db.store_arc()));
        assert!(db.get_node(node).is_some());
        assert_eq!(db.catalog.index_count(), 0);
        assert_eq!(db.catalog.index_allocator_high_water(), 1);
        assert_eq!(db.export_snapshot()?, bytes);
        let fresh = GrafeoDB::new_in_memory();
        fresh.restore_snapshot(&bytes)?;
        assert!(fresh.get_node(node).is_some());
        assert_eq!(fresh.catalog.index_count(), 0);
        assert_eq!(fresh.catalog.index_allocator_high_water(), 1);
        Ok(())
    }

    #[test]
    fn ownerless_index_transfer_and_invalid_exact_catalog_are_refused()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = GrafeoDB::new_in_memory();
        let node = source.create_node(&["Doc"]);
        source.store_arc().create_property_index("key");
        assert_eq!(source.catalog.index_allocator_high_water(), 0);
        assert!(source.extract_subgraph(&[node]).is_err());
        assert!(source.export_snapshot().is_err());
        assert!(source.get_node(node).is_some());
        assert!(source.store_arc().has_property_index("key"));

        let bytes = GrafeoDB::new_in_memory().export_snapshot()?;
        let mut snapshot = decode_snapshot_bytes(&bytes)?;
        snapshot.catalog_state.push(0xff);
        let invalid = encode_snapshot_bytes(&snapshot)?;
        assert!(GrafeoDB::import_snapshot(&invalid).is_err());
        Ok(())
    }

    #[cfg(feature = "gql")]
    #[test]
    fn catalog_foundation_actual_live_restore_preparation_fails_before_clear()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = GrafeoDB::new_in_memory();
        source
            .session()
            .execute("INSERT (:Fresh {email:'incoming'})")?;
        source
            .session()
            .execute("CREATE INDEX fresh_owner FOR (n:Fresh) ON (n.email)")?;
        source.session().execute("CREATE GRAPH kept")?;
        let incoming = source.export_snapshot()?;
        let target = GrafeoDB::new_in_memory();
        target
            .session()
            .execute("INSERT (:Old {slug:'preserved'})")?;
        target
            .session()
            .execute("CREATE INDEX old_owner FOR (n:Old) ON (n.slug)")?;
        target.session().execute("CREATE GRAPH old_graph")?;
        let old_graph = target
            .lpg_store()
            .graph("old_graph")
            .ok_or("missing old graph")?;
        let before = target.export_snapshot()?;
        // The detached candidate prepares root successfully, then fails before
        // preparing `kept`. No registry/catalog publication or live clear occurs.
        let injection =
            super::super::catalog_section::CurrentIndexPreparationFailure::after_successes(1);
        let error = target
            .restore_snapshot(&incoming)
            .err()
            .ok_or("current index preparation unexpectedly succeeded")?;
        assert!(
            error
                .to_string()
                .contains("injected current index preparation failure")
        );
        drop(injection);
        assert_eq!(target.export_snapshot()?, before);
        let current_graph = target
            .lpg_store()
            .graph("old_graph")
            .ok_or("old graph was removed")?;
        assert!(std::sync::Arc::ptr_eq(&old_graph, &current_graph));
        assert!(target.lpg_store().has_property_index("slug"));
        // Healthy retry exercises the same aggregate, including exact removal
        // of the target-only physical owner and named topology.
        target.restore_snapshot(&incoming)?;
        assert_eq!(target.export_snapshot()?, incoming);
        assert_eq!(target.catalog.find_index_by_name("old_owner"), None);
        assert!(target.catalog.find_index_by_name("fresh_owner").is_some());
        Ok(())
    }

    #[cfg(feature = "triple-store")]
    use grafeo_common::types::ValidTimeInterval;
    use grafeo_common::types::{
        EdgeId, EpochId, GraphPath, HistoryCompleteness, NodeId, StoreId, Value,
        WorldIdentityMetadataV1,
    };
    use grafeo_common::utils::error::Error;

    use super::super::GrafeoDB;
    use super::{
        MAX_PORTABLE_SNAPSHOT_BYTES, SNAPSHOT_VERSION, Snapshot, SnapshotEdge, SnapshotLpgGraph,
        SnapshotNode, decode_snapshot_bytes, encode_snapshot_bytes, snapshot_info,
        validate_snapshot_size,
    };
    #[cfg(any(feature = "triple-store", feature = "wal"))]
    use crate::config::Config;
    use crate::config::GraphModel;

    #[test]
    fn extraction_and_snapshot_keep_repeated_same_epoch_lifetime_order() {
        let epoch = EpochId::new(1);
        let lives = vec![(epoch, Some(epoch)), (epoch, Some(epoch)), (epoch, None)];
        let nodes = [1, 2]
            .into_iter()
            .map(|id| SnapshotNode {
                id: NodeId::new(id),
                lifetimes: lives.clone(),
                label_versions: vec![(epoch, vec!["Item".into()])],
                properties: Vec::new(),
            })
            .collect();
        let bytes = make_snapshot_at_epoch(
            SNAPSHOT_VERSION,
            nodes,
            vec![SnapshotEdge {
                id: EdgeId::new(7),
                src: NodeId::new(1),
                dst: NodeId::new(2),
                edge_type: "MULTI".into(),
                lifetimes: lives.clone(),
                properties: Vec::new(),
            }],
            4,
        );
        let source = GrafeoDB::import_snapshot(&bytes).expect("accepted same-epoch source");
        let source_image = decode_snapshot_bytes(&source.export_snapshot().unwrap()).unwrap();
        assert_eq!(source_image.nodes().unwrap()[0].lifetimes, lives);
        assert_eq!(source_image.edges().unwrap()[0].lifetimes, lives);
        let left = source
            .extract_subgraph(&[NodeId::new(1)])
            .expect("same-epoch extract");
        let right = source
            .extract_subgraph(&[NodeId::new(2)])
            .expect("same-epoch destination");
        let union = GrafeoDB::open_multi([
            left.export_snapshot().unwrap(),
            right.export_snapshot().unwrap(),
        ])
        .unwrap();
        let image = decode_snapshot_bytes(&union.export_snapshot().unwrap()).unwrap();
        assert_eq!(image.nodes().unwrap()[0].lifetimes, lives);
        assert_eq!(image.edges().unwrap()[0].lifetimes, lives);
        assert_eq!(left.remove_orphan_edges(), 1);
    }

    #[test]
    fn extraction_preserves_multiple_edge_lifetimes_through_union_and_cleanup() {
        let nodes = [1, 2]
            .into_iter()
            .map(|id| SnapshotNode {
                id: NodeId::new(id),
                lifetimes: vec![(EpochId::new(0), None)],
                label_versions: vec![(EpochId::new(0), vec!["Item".into()])],
                properties: Vec::new(),
            })
            .collect();
        let lives = vec![
            (EpochId::new(1), Some(EpochId::new(2))),
            (EpochId::new(3), None),
        ];
        let bytes = make_snapshot_at_epoch(
            SNAPSHOT_VERSION,
            nodes,
            vec![SnapshotEdge {
                id: EdgeId::new(7),
                src: NodeId::new(1),
                dst: NodeId::new(2),
                edge_type: "MULTI".into(),
                lifetimes: lives.clone(),
                properties: Vec::new(),
            }],
            4,
        );
        let source = GrafeoDB::import_snapshot(&bytes).expect("valid multi-lifetime source");
        let left = source
            .extract_subgraph(&[NodeId::new(1)])
            .expect("multi-lifetime extract");
        let right = source
            .extract_subgraph(&[NodeId::new(2)])
            .expect("destination extract");
        let left_bytes = left.export_snapshot().expect("left export");
        let right_bytes = right.export_snapshot().expect("right export");
        let union = GrafeoDB::open_multi([left_bytes, right_bytes]).expect("multi-lifetime union");
        let restored = decode_snapshot_bytes(&union.export_snapshot().unwrap()).unwrap();
        assert_eq!(restored.edges().unwrap()[0].lifetimes, lives);
        for (epoch, expected) in [(1, true), (2, false), (3, true), (4, true)] {
            assert_eq!(
                union
                    .graph_store()
                    .get_edge_at_epoch(EdgeId::new(7), EpochId::new(epoch))
                    .is_some(),
                expected
            );
        }
        assert_eq!(left.remove_orphan_edges(), 1);
        let reopened = GrafeoDB::import_snapshot(&left.export_snapshot().unwrap()).unwrap();
        assert_eq!(reopened.node_count(), 1);
        assert_eq!(reopened.edge_count(), 0);
    }

    #[test]
    fn transported_sibling_endpoint_policy_retains_lifetime_validation() {
        let node = |id, label: &str| SnapshotNode {
            id: NodeId::new(id),
            lifetimes: vec![(EpochId::new(0), None)],
            label_versions: vec![(EpochId::new(0), vec![label.to_owned()])],
            properties: Vec::new(),
        };
        let source = node(1, "Source");
        let destination = node(2, "Destination");
        let edge = SnapshotEdge {
            id: EdgeId::new(1),
            src: source.id,
            dst: destination.id,
            edge_type: "TRANSPORTED".to_owned(),
            lifetimes: vec![(EpochId::new(1), Some(EpochId::new(3)))],
            properties: Vec::new(),
        };
        assert!(
            super::validate_snapshot_data(
                std::slice::from_ref(&source),
                std::slice::from_ref(&edge)
            )
            .is_err()
        );
        let mut siblings = vec![
            decode_snapshot_bytes(&make_snapshot_at_epoch(
                SNAPSHOT_VERSION,
                vec![source],
                vec![edge],
                3,
            ))
            .expect("source chunk"),
            decode_snapshot_bytes(&make_snapshot_at_epoch(
                SNAPSHOT_VERSION,
                vec![destination],
                vec![],
                3,
            ))
            .expect("destination chunk"),
        ];
        super::validate_snapshot_set(&siblings).expect("endpoint resolves in sibling union");
        siblings[1].graphs[0].nodes[0].lifetimes[0].1 = Some(EpochId::new(2));
        let error = super::validate_snapshot_set(&siblings)
            .expect_err("transport does not permit an outliving edge");
        assert!(
            error.to_string().contains("outlives destination node"),
            "{error}"
        );
    }

    #[test]
    fn current_portable_bytes_survive_history_extraction() {
        use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};
        let identity = WorldIdentityMetadataV1::new(
            StoreId::from_bytes([0x42; StoreId::LEN]).expect("fixed identity"),
            HistoryCompleteness::Complete,
        )
        .expect("world identity");
        let db =
            GrafeoDB::with_config(crate::config::Config::in_memory().with_world_identity(identity))
                .expect("database");
        let node = db.create_node(&["Values"]);
        let values = vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(-42),
            Value::Float64(-0.0),
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
            Value::Vector(vec![1.25, -0.0].into()),
            Value::RdfLiteral {
                lexical: "bonjour".into(),
                language: Some("fr".into()),
                datatype: None,
            },
        ];
        db.set_node_property(node, "values", Value::List(values.into()))
            .expect("set node property");
        let bytes = db.export_snapshot().expect("current portable writer");
        // Snapshot12 uses the shared exact LPG value codec; the old v10
        // golden hash is not a current-format compatibility contract.
        assert_eq!(bytes.first(), Some(&SNAPSHOT_VERSION));
        assert_eq!(
            snapshot_info(&bytes)
                .expect("current snapshot info")
                .version,
            SNAPSHOT_VERSION
        );
        let restored = GrafeoDB::import_snapshot(&bytes).expect("portable restore");
        assert_eq!(
            restored.export_snapshot().expect("portable re-export"),
            bytes
        );
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
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
        assert_eq!(db.node_count(), 3);

        // Restore original
        db.restore_snapshot(&snapshot).unwrap();

        assert_eq!(db.node_count(), 2);
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn test_restore_snapshot_validation_failure() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Corrupt snapshot: just garbage bytes
        let result = db.restore_snapshot(b"garbage");
        assert!(result.is_err());

        // DB should be unchanged
        assert_eq!(db.node_count(), 1);
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn restore_snapshot_refuses_compacted_layer_without_mutation() {
        let incoming = GrafeoDB::new_in_memory();
        incoming.create_node(&["Incoming"]);
        let snapshot = incoming.export_snapshot().unwrap();

        let mut target = GrafeoDB::new_in_memory();
        let existing = target.create_node(&["Existing"]);
        target.compact().unwrap();

        let error = target
            .restore_snapshot(&snapshot)
            .expect_err("exact replacement of a compacted layered store must fail closed")
            .to_string();
        assert!(error.contains("compacted layered"), "{error}");
        assert_eq!(target.node_count(), 1);
        assert_eq!(
            target.get_node_labels(existing),
            Some(vec!["Existing".to_string()]),
            "refusal must leave the cold base and overlay unchanged"
        );
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn test_restore_snapshot_empty_db() {
        let db = GrafeoDB::new_in_memory();

        // Export empty snapshot, then populate, then restore to empty
        let empty_snapshot = db.export_snapshot().unwrap();

        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        assert_eq!(db.node_count(), 1);

        db.restore_snapshot(&empty_snapshot).unwrap();
        assert_eq!(db.node_count(), 0);
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
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
        assert_eq!(db.edge_count(), 1);

        // Modify: add more data
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
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
    #[cfg(all(feature = "lpg", feature = "gql"))]
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

    // Recursive Snapshot12 export/copy and late-path restore rejection are
    // covered together below; nested data no longer requires a refusal guard.

    // --- to_memory() ---

    #[test]
    fn test_to_memory_empty() {
        let db = GrafeoDB::new_in_memory();
        let copy = db.to_memory().unwrap();
        assert_eq!(copy.node_count(), 0);
        assert_eq!(copy.edge_count(), 0);
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
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
        assert_eq!(copy.node_count(), 2);

        let s2 = copy.session();
        let result = s2
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows[1][0], Value::String("Gus".into()));
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn test_to_memory_copies_edges_and_properties() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["Person"]);
        db.set_node_property(a, "name", "Alix".into())
            .expect("set node property");
        let b = db.create_node(&["Person"]);
        db.set_node_property(b, "name", "Gus".into())
            .expect("set node property");
        let edge = db.create_edge(a, b, "KNOWS");
        db.set_edge_property(edge, "since", Value::Int64(2020))
            .expect("set edge property");

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.node_count(), 2);
        assert_eq!(copy.edge_count(), 1);

        let s2 = copy.session();
        let result = s2.execute("MATCH ()-[e:KNOWS]->() RETURN e.since").unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2020));
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn test_to_memory_is_independent() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let copy = db.to_memory().unwrap();

        // Mutating original should not affect copy
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(copy.node_count(), 1);
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
        db.set_node_property(id1, "name", "Alix".into())
            .expect("set node property");
        let id2 = db.create_node(&["Animal"]);
        db.set_node_property(id2, "name", "Fido".into())
            .expect("set node property");

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

    // --- remove_orphan_edges() ---

    #[test]
    fn remove_orphan_edges_drops_dangling_dst_edges() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        db.create_edge(a, b, "MAPS_TO");
        assert_eq!(db.edge_count(), 1);

        // extract_subgraph carries every outgoing edge from `a`
        // (source-side ownership), including a->b even though `b` is
        // not in the request set. Result: 1 node, 1 dangling-dst edge.
        let sub = db.extract_subgraph(&[a]).expect("extract");
        assert_eq!(sub.node_count(), 1);
        assert_eq!(
            sub.edge_count(),
            1,
            "extract_subgraph carries the dangling a->b edge"
        );

        let deleted = sub.remove_orphan_edges();
        assert_eq!(deleted, 1, "the dangling a->b edge should be deleted");
        assert_eq!(sub.edge_count(), 0);

        // The subgraph is now self-consistent; export/import round-trips
        // through the per-snapshot validator that previously rejected
        // dangling-dst edges.
        let bytes = sub.export_snapshot().expect("export");
        let _reopened = GrafeoDB::import_snapshot(&bytes).expect("self-consistent");
    }

    #[test]
    fn remove_orphan_edges_returns_zero_when_no_orphans() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        db.create_edge(a, b, "MAPS_TO");
        assert_eq!(db.remove_orphan_edges(), 0);
        assert_eq!(db.edge_count(), 1, "closed graph unchanged");
    }

    // --- restore_snapshot() validation ---

    fn make_snapshot(version: u8, nodes: Vec<SnapshotNode>, edges: Vec<SnapshotEdge>) -> Vec<u8> {
        make_snapshot_at_epoch(version, nodes, edges, 0)
    }

    fn encode_snapshot_for_version(snapshot: &Snapshot) -> Vec<u8> {
        encode_snapshot_bytes(snapshot).unwrap()
    }

    #[test]
    fn portable_v11_round_trip_preserves_retained_history_floor() {
        let mut snapshot = decode_snapshot_bytes(&make_snapshot_at_epoch(
            SNAPSHOT_VERSION,
            Vec::new(),
            Vec::new(),
            9,
        ))
        .unwrap();
        snapshot.graphs[0].retained_history_floor = 7;
        let reopened = GrafeoDB::import_snapshot(&encode_snapshot_for_version(&snapshot)).unwrap();
        assert_eq!(
            reopened.store_arc().retained_history_floor(),
            EpochId::new(7)
        );
    }

    #[test]
    #[cfg(feature = "lpg")]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn portable_v11_restores_index_history_after_gc_at_retained_floor()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_core::graph::{PropertyIndexPredicate, PropertyIndexRequest};

        let db = GrafeoDB::new_in_memory();
        let mut writer = db.session();
        writer.begin_transaction()?;
        writer.execute("INSERT (:Indexed {score: 1})")?;
        writer.commit()?;
        let node = crate::database::testing::root_lpg_store(&db)
            .node_ids()
            .into_iter()
            .next()
            .ok_or("initial indexed node was not published")?;

        writer.begin_transaction()?;
        writer.execute("CREATE INDEX indexed_score FOR (n:Indexed) ON (n.score)")?;
        writer.commit()?;
        let retained_epoch = db.current_epoch();

        // Pin the old snapshot while two later A→B→A property versions commit.
        let mut reader = db.session();
        reader.begin_transaction()?;
        writer.begin_transaction()?;
        writer.execute("MATCH (n:Indexed) SET n.score = 2")?;
        writer.commit()?;
        writer.begin_transaction()?;
        writer.execute("MATCH (n:Indexed) SET n.score = 1")?;
        writer.commit()?;

        db.gc()?;
        assert_eq!(
            crate::database::testing::root_lpg_store(&db).retained_history_floor(),
            retained_epoch
        );
        reader.commit()?;

        let property = grafeo_common::types::PropertyKey::new("score");
        let oracle: Vec<_> = crate::database::testing::root_lpg_store(&db)
            .node_ids()
            .into_iter()
            .filter(|id| {
                crate::database::testing::root_lpg_store(&db)
                    .get_node_at_epoch(*id, retained_epoch)
                    .is_some()
                    && crate::database::testing::root_lpg_store(&db)
                        .get_node_property_at_epoch(*id, &property, retained_epoch)
                        .is_some_and(|value| value == Value::Int64(1))
            })
            .collect();
        assert_eq!(oracle, vec![node]);

        let bytes = db.export_snapshot()?;
        let restored = GrafeoDB::import_snapshot(&bytes)?;
        assert_eq!(
            crate::database::testing::root_lpg_store(&restored).retained_history_floor(),
            retained_epoch
        );
        let one = Value::Int64(1);
        let two = Value::Int64(2);
        let in_values = [one.clone(), two];
        let indexed_eq = crate::database::testing::root_lpg_store(&restored)
            .lookup_nodes_indexed(PropertyIndexRequest {
                property: "score",
                predicate: PropertyIndexPredicate::Equal(&one),
                epoch: retained_epoch,
                transaction_id: None,
            })?
            .ok_or("restored equality index was not registered")?;
        assert_eq!(indexed_eq, oracle);
        let indexed_in = crate::database::testing::root_lpg_store(&restored)
            .lookup_nodes_indexed(PropertyIndexRequest {
                property: "score",
                predicate: PropertyIndexPredicate::In(&in_values),
                epoch: retained_epoch,
                transaction_id: None,
            })?
            .ok_or("restored IN index was not registered")?;
        assert_eq!(indexed_in, oracle);
        let indexed_range = crate::database::testing::root_lpg_store(&restored)
            .lookup_nodes_indexed(PropertyIndexRequest {
                property: "score",
                predicate: PropertyIndexPredicate::Range {
                    min: Some(&one),
                    max: Some(&one),
                    min_inclusive: true,
                    max_inclusive: true,
                },
                epoch: retained_epoch,
                transaction_id: None,
            })?
            .ok_or("restored range index was not registered")?;
        assert_eq!(indexed_range, oracle);

        let below_floor = crate::database::testing::root_lpg_store(&restored)
            .lookup_nodes_indexed(PropertyIndexRequest {
                property: "score",
                predicate: PropertyIndexPredicate::Equal(&one),
                epoch: EpochId::new(retained_epoch.as_u64().saturating_sub(1)),
                transaction_id: None,
            })
            .expect_err("index lookup below retained history floor must fail");
        assert!(
            below_floor
                .to_string()
                .contains("predates retained store history")
        );
        Ok(())
    }

    fn make_snapshot_at_epoch(
        version: u8,
        nodes: Vec<SnapshotNode>,
        edges: Vec<SnapshotEdge>,
        epoch: u64,
    ) -> Vec<u8> {
        (|| -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
            let identity = WorldIdentityMetadataV1::new(
                StoreId::from_bytes([0x42; StoreId::LEN])?,
                HistoryCompleteness::Complete,
            )?;
            let source = GrafeoDB::with_config(
                crate::config::Config::in_memory().with_world_identity(identity),
            )?;
            let mut snapshot = decode_snapshot_bytes(&source.export_snapshot()?)?;
            snapshot.version = version;
            snapshot.epoch = epoch;
            snapshot.cdc_checkpoint = crate::database::cdc_checkpoint::disabled(
                snapshot.world_identity.store_id(),
                EpochId::new(epoch),
            )?;
            let next_node_id =
                nodes
                    .iter()
                    .map(|node| node.id.as_u64())
                    .max()
                    .map_or(Ok(0), |id| {
                        id.checked_add(1)
                            .ok_or("fixture node IDs exhaust allocator")
                    })?;
            let next_edge_id =
                edges
                    .iter()
                    .map(|edge| edge.id.as_u64())
                    .max()
                    .map_or(Ok(0), |id| {
                        id.checked_add(1)
                            .ok_or("fixture edge IDs exhaust allocator")
                    })?;
            snapshot.graphs = vec![SnapshotLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                path: GraphPath::root(),
                next_node_id,
                next_edge_id,
                retained_history_floor: 0,
                nodes,
                edges,
            }];
            // Pending-epoch negatives must reach outer admission; valid epochs
            // agree with the nested current catalog image.
            let catalog_epoch = if epoch == u64::MAX { 0 } else { epoch };
            snapshot.catalog_state = crate::database::catalog_wire::encode_catalog_read(
                source.catalog.read().view(),
                catalog_epoch,
            )?;
            Ok(encode_snapshot_for_version(&snapshot))
        })()
        .unwrap()
    }

    #[test]
    fn current_snapshot_rejects_unknown_graph_model_tag() {
        let bytes = make_snapshot_at_epoch(SNAPSHOT_VERSION, Vec::new(), Vec::new(), 0);
        let mut snapshot = decode_snapshot_bytes(&bytes).unwrap();
        snapshot.graph_model = 3;
        let forged = encode_snapshot_bytes(&snapshot).unwrap();

        assert!(GrafeoDB::import_snapshot(&forged).is_err());
    }

    #[test]
    fn portable_snapshot_budget_rejects_oversized_input_before_decode() {
        let error = validate_snapshot_size(MAX_PORTABLE_SNAPSHOT_BYTES + 1)
            .expect_err("oversized snapshot must fail before bincode allocation")
            .to_string();
        assert!(error.contains("maximum"), "{error}");
    }

    #[test]
    fn portable_v10_zero_width_lifetimes_round_trip_exactly_and_are_never_visible() {
        let committed = EpochId::new(7);
        let before = EpochId::new(6);
        let after = EpochId::new(8);
        let subject = NodeId::new(40);
        let object = NodeId::new(41);
        let relation = EdgeId::new(90);
        let subject_labels = vec!["Ephemeral".to_string(), "Transient".to_string()];
        let subject_label_history = vec![
            (committed, vec!["Ephemeral".to_string()]),
            (committed, subject_labels.clone()),
        ];
        let subject_properties = vec![(
            "name".to_string(),
            vec![(committed, Value::from("draft")), (committed, Value::Null)],
        )];
        let edge_properties = vec![(
            "weight".to_string(),
            vec![(committed, Value::Int64(1)), (committed, Value::Null)],
        )];
        let bytes = make_snapshot_at_epoch(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: subject,
                    lifetimes: vec![(committed, Some(committed))],
                    label_versions: subject_label_history.clone(),
                    properties: subject_properties.clone(),
                },
                SnapshotNode {
                    id: object,
                    lifetimes: vec![(committed, Some(committed))],
                    label_versions: vec![(committed, vec!["Object".to_string()])],
                    properties: Vec::new(),
                },
            ],
            vec![SnapshotEdge {
                id: relation,
                src: subject,
                dst: object,
                edge_type: "TOUCHES".to_string(),
                lifetimes: vec![(committed, Some(committed))],
                properties: edge_properties.clone(),
            }],
            committed.as_u64(),
        );

        let restored = GrafeoDB::import_snapshot(&bytes)
            .expect("a same-commit create/delete is exact retained history");
        assert_eq!(restored.node_count(), 0);
        assert_eq!(restored.edge_count(), 0);
        for epoch in [before, committed, after] {
            assert!(
                restored.get_node_at_epoch(subject, epoch).is_none(),
                "zero-width node unexpectedly visible at {epoch}"
            );
            assert!(
                restored.get_edge_at_epoch(relation, epoch).is_none(),
                "zero-width edge unexpectedly visible at {epoch}"
            );
        }

        let store = restored.store_arc();
        let node_lifetimes = store
            .get_node_history(subject)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect::<Vec<_>>();
        assert_eq!(node_lifetimes, vec![(committed, Some(committed))]);
        let edge_lifetimes = store
            .get_edge_history(relation)
            .into_iter()
            .map(|(created, deleted, _)| (created, deleted))
            .collect::<Vec<_>>();
        assert_eq!(edge_lifetimes, vec![(committed, Some(committed))]);
        let restored_label_history = store
            .node_label_history(subject)
            .into_iter()
            .map(|(epoch, labels)| {
                (
                    epoch,
                    labels.into_iter().map(|label| label.to_string()).collect(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            restored_label_history, subject_label_history,
            "ordered same-epoch label states are exact history"
        );
        assert_eq!(
            store.node_property_history_for_key(subject, "name"),
            subject_properties[0].1,
        );
        assert_eq!(
            store
                .edge_property_history(relation)
                .into_iter()
                .find(|(key, _)| key.as_str() == "weight")
                .expect("zero-width edge property history")
                .1,
            edge_properties[0].1,
        );
        assert_eq!(store.peek_next_node_id(), 42);
        assert_eq!(store.peek_next_edge_id(), 91);

        let round_trip = decode_snapshot_bytes(&restored.export_snapshot().unwrap()).unwrap();
        let round_subject = round_trip.graphs[0]
            .nodes
            .iter()
            .find(|node| node.id == subject)
            .expect("zero-width node survives v10 re-export");
        assert_eq!(round_subject.lifetimes, vec![(committed, Some(committed))]);
        assert_eq!(round_subject.label_versions, subject_label_history);
        assert_eq!(round_subject.properties, subject_properties);
        let round_edge = round_trip.graphs[0]
            .edges
            .iter()
            .find(|edge| edge.id == relation)
            .expect("zero-width edge survives v10 re-export");
        assert_eq!(round_edge.lifetimes, vec![(committed, Some(committed))]);
        assert_eq!(round_edge.properties, edge_properties);
        assert_eq!(round_trip.graphs.len(), 1);
        assert_eq!(round_trip.graphs[0].path, GraphPath::root());
        assert_eq!(
            (
                round_trip.graphs[0].next_node_id,
                round_trip.graphs[0].next_edge_id
            ),
            (42, 91)
        );
    }

    #[test]
    fn portable_v10_rejects_negative_width_lifetimes_before_restore_mutation() {
        let created = EpochId::new(7);
        let deleted = EpochId::new(6);
        let invalid_node = make_snapshot_at_epoch(
            SNAPSHOT_VERSION,
            vec![SnapshotNode {
                id: NodeId::new(40),
                lifetimes: vec![(created, Some(deleted))],
                label_versions: vec![(created, vec!["Invalid".to_string()])],
                properties: Vec::new(),
            }],
            Vec::new(),
            created.as_u64(),
        );
        let invalid_edge = make_snapshot_at_epoch(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: NodeId::new(40),
                    lifetimes: vec![(EpochId::INITIAL, None)],
                    label_versions: vec![(EpochId::INITIAL, vec!["Source".to_string()])],
                    properties: Vec::new(),
                },
                SnapshotNode {
                    id: NodeId::new(41),
                    lifetimes: vec![(EpochId::INITIAL, None)],
                    label_versions: vec![(EpochId::INITIAL, vec!["Destination".to_string()])],
                    properties: Vec::new(),
                },
            ],
            vec![SnapshotEdge {
                id: EdgeId::new(90),
                src: NodeId::new(40),
                dst: NodeId::new(41),
                edge_type: "INVALID".to_string(),
                lifetimes: vec![(created, Some(deleted))],
                properties: Vec::new(),
            }],
            created.as_u64(),
        );

        for (entity, bytes) in [
            ("snapshot node", invalid_node),
            ("snapshot edge", invalid_edge),
        ] {
            let target = GrafeoDB::new_in_memory();
            let existing = target.create_node(&["Existing"]);
            let before_allocator = (
                target.store_arc().peek_next_node_id(),
                target.store_arc().peek_next_edge_id(),
            );

            let error = target
                .restore_snapshot(&bytes)
                .expect_err("negative-width history must fail closed")
                .to_string();
            assert!(error.contains(entity), "{error}");
            assert!(error.contains("invalid delete epoch"), "{error}");
            assert_eq!(target.node_count(), 1);
            assert_eq!(target.edge_count(), 0);
            assert_eq!(
                target.get_node_labels(existing),
                Some(vec!["Existing".to_string()]),
            );
            assert_eq!(
                (
                    target.store_arc().peek_next_node_id(),
                    target.store_arc().peek_next_edge_id(),
                ),
                before_allocator,
                "rejected restore must not consume allocator IDs"
            );
        }
    }

    fn encode_current_snapshot(snapshot: &Snapshot) -> Vec<u8> {
        encode_snapshot_bytes(snapshot).unwrap()
    }

    #[test]
    fn recursive_snapshot_v10_import_and_memory_copy_preserve_paths_owners_and_reserved_gaps()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_core::graph::lpg::LpgStoreSection;
        use std::sync::Arc;

        let source = GrafeoDB::with_config(crate::config::Config::in_memory().with_gc_interval(0))?;
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["literal/slash"])?,
            GraphPath::from_components(&["literal", "slash"])?,
        ];
        // Private authority is confined to fixture topology. Every data write,
        // rollback reservation and index owner uses its public managed API.
        source.transaction_manager.with_write_authority(
            || -> grafeo_common::utils::error::Result<()> {
                for path in &paths {
                    let mut graph = Arc::clone(source.store_arc());
                    for component in path.components() {
                        graph = graph.graph_or_create(component)?;
                    }
                }
                Ok(())
            },
        )?;
        let mut nodes = Vec::new();
        for (index, path) in paths.iter().enumerate() {
            let mut session = source.session();
            session.use_graph_path(path)?;
            let value = i64::try_from(index)?;
            let node =
                session.create_node_with_props(&["Recorded"], [("value", Value::Int64(value))])?;
            session.set_node_property(node, "value", Value::Int64(value + 10))?;
            session.begin_transaction()?;
            let discarded_a = session.create_node(&["Discarded"]);
            let discarded_b = session.create_node(&["Discarded"]);
            assert!(discarded_a.is_valid() && discarded_b.is_valid());
            assert!(
                session
                    .create_edge(discarded_a, discarded_b, "DISCARDED")
                    .is_valid()
            );
            session.rollback()?;
            drop(session);
            source.create_index(crate::CreateIndexRequest {
                graph: path.clone(),
                name: Some(format!("path-{index}")),
                label: None,
                property: "value".into(),
                kind: crate::IndexCreateKind::Property,
            })?;
            nodes.push(node);
        }
        assert!(
            nodes.windows(2).all(|pair| pair[0] == pair[1]),
            "graph-local IDs deliberately collide"
        );
        let bytes = source.export_snapshot()?;
        let decoded = decode_snapshot_bytes(&bytes)?;
        assert_eq!(
            decoded.graphs.len(),
            5,
            "the intermediate literal parent is explicit"
        );
        let graph_image =
            bincode::serde::encode_to_vec(&decoded.graphs, bincode::config::standard())?;
        let mut owners = source.catalog.all_indexes();
        owners.sort_unstable_by_key(|owner| owner.id);
        for restored in [GrafeoDB::import_snapshot(&bytes)?, source.to_memory()?] {
            let copied = decode_snapshot_bytes(&restored.export_snapshot()?)?;
            assert_eq!(
                bincode::serde::encode_to_vec(&copied.graphs, bincode::config::standard())?,
                graph_image
            );
            assert_eq!(copied.catalog_state, decoded.catalog_state);
            let mut copied_owners = restored.catalog.all_indexes();
            copied_owners.sort_unstable_by_key(|owner| owner.id);
            assert_eq!(copied_owners, owners);
            let graphs = LpgStoreSection::new(Arc::clone(restored.store_arc())).capture_graphs()?;
            for (index, path) in paths.iter().enumerate() {
                let graph = &graphs
                    .iter()
                    .find(|(candidate, _)| candidate == path)
                    .ok_or("restored path missing")?
                    .1;
                assert_eq!(
                    (graph.peek_next_node_id(), graph.peek_next_edge_id()),
                    (3, 1)
                );
                let value = Value::Int64(i64::try_from(index)? + 10);
                assert_eq!(
                    graph.find_nodes_by_property("value", &value),
                    vec![nodes[index]]
                );
                assert_eq!(
                    graph
                        .node_property_history_for_key(nodes[index], "value")
                        .len(),
                    2
                );
                assert_eq!(graph.node_count(), 1);
                assert_eq!(graph.edge_count(), 0);
            }
        }
        // Live restore with active owners remains guarded. An ownerless target
        // reaches the late path validator instead of that earlier admission.
        let target = GrafeoDB::new_in_memory();
        assert!(target.create_node(&["RetainedTarget"]).is_valid());
        let before = target.export_snapshot()?;
        let retained_root = Arc::clone(target.store_arc());
        for invalid_path in [
            GraphPath::from_components(&["zz-missing-parent", "late"])?,
            GraphPath::root(),
        ] {
            let mut invalid = decoded.clone();
            invalid.graphs.last_mut().ok_or("missing last graph")?.path = invalid_path;
            let invalid_bytes = encode_snapshot_bytes(&invalid)?;
            let error = target
                .restore_snapshot(&invalid_bytes)
                .err()
                .ok_or("invalid late path was accepted")?;
            assert!(matches!(error, Error::Serialization(_)), "{error}");
            assert!(
                error.to_string().contains("no parent") || error.to_string().contains("canonical"),
                "{error}"
            );
            assert_eq!(target.export_snapshot()?, before);
            assert!(Arc::ptr_eq(&retained_root, target.store_arc()));
        }
        Ok(())
    }

    #[test]
    fn portable_v10_large_graph_allocator_cut_preserves_gaps_and_rejects_invalid_records()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = GrafeoDB::new_in_memory();
        let first = source.create_node(&["Known"]);
        let second = source.create_node(&["Known"]);
        assert!(first.is_valid() && second.is_valid());
        assert!(source.create_edge(first, second, "LINK").is_valid());
        let mut snapshot = decode_snapshot_bytes(&source.export_snapshot()?)?;
        let mut names: Vec<String> = (0..2_048).map(|id| format!("graph-{id:04}")).collect();
        names.extend([String::new(), "default".into(), "literal/slash".into()]);
        names.sort_unstable();
        assert_eq!(names.len(), 2_051);
        snapshot.graphs[0].next_node_id = 90_000;
        snapshot.graphs[0].next_edge_id = 90_001;
        let root = snapshot.graphs[0].clone();
        snapshot.next_graph_incarnation_id = u64::try_from(names.len())? + 1;
        for (index, name) in names.iter().enumerate() {
            let offset = u64::try_from(index)?;
            snapshot.graphs.push(SnapshotLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(offset + 1),
                path: GraphPath::from_components(&[name])?,
                next_node_id: 1_000 + offset,
                next_edge_id: 2_000 + offset,
                retained_history_floor: 0,
                nodes: root.nodes.clone(),
                edges: root.edges.clone(),
            });
        }
        let encoded = encode_snapshot_bytes(&snapshot)?;
        let decoded = decode_snapshot_bytes(&encoded)?;
        assert_eq!(decoded.graphs.len(), 2_052);
        for expected in &snapshot.graphs {
            let actual = super::snapshot_allocator_record(&decoded, &expected.path)?;
            assert_eq!(actual.path, expected.path);
            assert_eq!(
                (actual.next_node_id, actual.next_edge_id),
                (expected.next_node_id, expected.next_edge_id)
            );
        }
        assert_eq!(
            super::snapshot_allocator_record(&decoded, &GraphPath::root())?.next_node_id,
            90_000
        );
        assert_eq!(
            super::snapshot_allocator_record(&decoded, &GraphPath::from_components(&[""])?)?
                .next_node_id,
            1_000
        );
        assert!(
            super::snapshot_allocator_record(
                &decoded,
                &GraphPath::from_components(&["literal/slash"])?
            )
            .is_ok()
        );
        assert!(
            super::snapshot_allocator_record(
                &decoded,
                &GraphPath::from_components(&["literal", "slash"])?
            )
            .is_err()
        );

        // Counters and rows now share one record. An extra well-formed empty
        // graph is valid; missing-root/parent and noncanonical records are not.
        let mut extended = snapshot.clone();
        extended.next_graph_incarnation_id += 1;
        extended.graphs.push(SnapshotLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(
                extended.next_graph_incarnation_id - 1,
            ),
            path: GraphPath::from_components(&["zz-extra"])?,
            next_node_id: 23,
            next_edge_id: 29,
            retained_history_floor: 0,
            nodes: Vec::new(),
            edges: Vec::new(),
        });
        super::validate_decoded_snapshot(extended)?;
        for mutation in 0..10 {
            let mut invalid = snapshot.clone();
            let diagnostic = match mutation {
                0 => {
                    invalid.graphs.insert(1, invalid.graphs[0].clone());
                    "unique"
                }
                1 => {
                    invalid.graphs.remove(0);
                    "root"
                }
                2 => {
                    invalid.graphs[1].path = GraphPath::from_components(&["", "orphan"])?;
                    "no parent"
                }
                3 => {
                    invalid.graphs[1..].reverse();
                    "canonical"
                }
                4 => {
                    invalid.graphs.swap(1, 2_050);
                    "canonical"
                }
                5 => {
                    invalid.graphs[0].next_node_id = 0;
                    "below required"
                }
                6 => {
                    invalid.graphs[1].next_node_id = 0;
                    "below required"
                }
                7 => {
                    invalid.graphs[1].next_edge_id = 0;
                    "below required"
                }
                8 => {
                    invalid.graphs[0].nodes[0].id = NodeId::new(u64::MAX);
                    "no representable allocator high-water"
                }
                _ => {
                    invalid.graphs[1].edges[0].id = EdgeId::new(u64::MAX);
                    "no representable allocator high-water"
                }
            };
            let bytes = encode_snapshot_bytes(&invalid)?;
            let error = decode_snapshot_bytes(&bytes)
                .err()
                .ok_or("invalid graph allocator cut accepted")?;
            assert!(
                matches!(error, Error::Serialization(_)),
                "case {mutation}: {error}"
            );
            assert!(
                error.to_string().contains(diagnostic),
                "case {mutation}: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn portable_v10_rejects_noncanonical_duplicate_and_parentless_graph_records()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = GrafeoDB::new_in_memory();
        source.transaction_manager.with_write_authority(
            || -> grafeo_common::utils::error::Result<()> {
                source.lpg_store().create_graph("zeta")?;
                source.lpg_store().create_graph("alpha")?;
                Ok(())
            },
        )?;
        let canonical = decode_snapshot_bytes(&source.export_snapshot()?)?;
        assert_eq!(
            canonical
                .graphs
                .iter()
                .map(|graph| graph.path.clone())
                .collect::<Vec<_>>(),
            vec![
                GraphPath::root(),
                GraphPath::from_components(&["alpha"])?,
                GraphPath::from_components(&["zeta"])?
            ]
        );
        for mutation in 0..3 {
            let mut invalid = canonical.clone();
            let diagnostic = match mutation {
                0 => {
                    invalid.graphs.swap(1, 2);
                    "canonical"
                }
                1 => {
                    invalid.graphs.insert(1, invalid.graphs[0].clone());
                    "unique"
                }
                _ => {
                    invalid.graphs[2].path = GraphPath::from_components(&["zz-absent", "child"])?;
                    "no parent"
                }
            };
            let bytes = encode_snapshot_bytes(&invalid)?;
            let error = decode_snapshot_bytes(&bytes)
                .err()
                .ok_or("invalid graph records accepted")?;
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
        Ok(())
    }

    #[test]
    fn portable_v10_rejects_missing_root_or_regressive_allocator_high_water()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = GrafeoDB::new_in_memory();
        assert!(source.create_node(&["Known"]).is_valid());
        let canonical = decode_snapshot_bytes(&source.export_snapshot()?)?;
        let mut missing = canonical.clone();
        missing.graphs.clear();
        let error = decode_snapshot_bytes(&encode_current_snapshot(&missing))
            .err()
            .ok_or("missing root accepted")?;
        assert!(error.to_string().contains("root"), "{error}");
        let mut regressive = canonical;
        regressive.graphs[0].next_node_id = 0;
        let error = decode_snapshot_bytes(&encode_current_snapshot(&regressive))
            .err()
            .ok_or("regressive counter accepted")?;
        assert!(error.to_string().contains("below required"), "{error}");
        Ok(())
    }

    #[test]
    fn portable_v10_truncation_is_rejected() {
        let source = GrafeoDB::new_in_memory();
        let mut bytes = source.export_snapshot().unwrap();
        bytes.pop();
        let error = decode_snapshot_bytes(&bytes)
            .err()
            .expect("truncated snapshot must be rejected")
            .to_string();
        assert!(
            error.contains("Snapshot12 bounded body length mismatch"),
            "{error}"
        );
    }

    fn reserve_ids_then_rollback(db: &GrafeoDB, graph: Option<&str>) -> (u64, u64) {
        let mut session = db.session();
        if let Some(graph) = graph {
            session
                .use_graph_path(
                    &grafeo_common::types::GraphPath::from_components(&[graph])
                        .expect("literal graph path"),
                )
                .expect("select existing graph");
        }
        session.begin_transaction().unwrap();
        let source = session.create_node(&["Reserved"]);
        let destination = session.create_node(&["Reserved"]);
        session.create_edge(source, destination, "RESERVED");
        session.rollback().unwrap();

        let store = graph.map_or_else(
            || std::sync::Arc::clone(db.store_arc()),
            |name| db.store_arc().graph(name).unwrap(),
        );
        assert_eq!(store.all_nodes().count(), 0);
        assert_eq!(store.all_edges().count(), 0);
        (store.peek_next_node_id(), store.peek_next_edge_id())
    }

    #[test]
    fn portable_v10_exact_import_and_fork_preserve_rolled_back_allocator_gaps() {
        let source = GrafeoDB::new_in_memory();
        source.lpg_store().create_graph("reserved").unwrap();
        let default_high_water = reserve_ids_then_rollback(&source, None);
        let named_high_water = reserve_ids_then_rollback(&source, Some("reserved"));
        assert_eq!(default_high_water, (2, 1));
        assert_eq!(named_high_water, (2, 1));

        let bytes = source.export_snapshot().unwrap();
        let exact = GrafeoDB::import_snapshot(&bytes).unwrap();
        let fork = GrafeoDB::import_snapshot_as_fork(&bytes).unwrap();

        for restored in [&exact, &fork] {
            assert_eq!(
                (
                    restored.store_arc().peek_next_node_id(),
                    restored.store_arc().peek_next_edge_id(),
                ),
                default_high_water
            );
            let named = restored.store_arc().graph("reserved").unwrap();
            assert_eq!(
                (named.peek_next_node_id(), named.peek_next_edge_id()),
                named_high_water
            );
        }
    }

    #[test]
    fn portable_v10_restore_replaces_target_allocator_high_water_exactly() {
        let source = GrafeoDB::new_in_memory();
        source.lpg_store().create_graph("reserved").unwrap();
        let default_high_water = reserve_ids_then_rollback(&source, None);
        let named_high_water = reserve_ids_then_rollback(&source, Some("reserved"));
        let bytes = source.export_snapshot().unwrap();

        let target = GrafeoDB::new_in_memory();
        target.store_arc().set_next_node_id(900);
        target.store_arc().set_next_edge_id(901);
        target.restore_snapshot(&bytes).unwrap();

        assert_eq!(
            (
                target.store_arc().peek_next_node_id(),
                target.store_arc().peek_next_edge_id(),
            ),
            default_high_water
        );
        let named = target.store_arc().graph("reserved").unwrap();
        assert_eq!(
            (named.peek_next_node_id(), named.peek_next_edge_id()),
            named_high_water
        );
    }

    #[test]
    fn open_multi_uses_max_default_and_exact_named_allocator_high_water() {
        let left = GrafeoDB::new_in_memory();
        left.store_arc().set_next_node_id(30);
        left.store_arc().set_next_edge_id(40);

        let right = GrafeoDB::new_in_memory();
        right.store_arc().set_next_node_id(50);
        right.store_arc().set_next_edge_id(20);
        right.lpg_store().create_graph("right_only").unwrap();
        let right_named = right.store_arc().graph("right_only").unwrap();
        right_named.set_next_node_id(70);
        right_named.set_next_edge_id(80);

        let left = left.export_snapshot().unwrap();
        let right = right.export_snapshot().unwrap();
        let merged = GrafeoDB::open_multi([left.as_slice(), right.as_slice()]).unwrap();

        assert_eq!(merged.store_arc().peek_next_node_id(), 50);
        assert_eq!(merged.store_arc().peek_next_edge_id(), 40);
        let named = merged.store_arc().graph("right_only").unwrap();
        assert_eq!(named.peek_next_node_id(), 70);
        assert_eq!(named.peek_next_edge_id(), 80);
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn open_multi_with_both_owner_preserves_rolled_back_allocator_gaps() {
        let both =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
        both.lpg_store().create_graph("both_reserved").unwrap();
        let both_default = reserve_ids_then_rollback(&both, None);
        let both_named = reserve_ids_then_rollback(&both, Some("both_reserved"));
        let source_history = both.rdf_dataset_history().unwrap();

        let lpg = GrafeoDB::new_in_memory();
        lpg.lpg_store().create_graph("lpg_reserved").unwrap();
        reserve_ids_then_rollback(&lpg, None);
        let lpg_default = reserve_ids_then_rollback(&lpg, None);
        let lpg_named = reserve_ids_then_rollback(&lpg, Some("lpg_reserved"));

        let both_bytes = both.export_snapshot().unwrap();
        let lpg_bytes = lpg.export_snapshot().unwrap();
        let merged = GrafeoDB::open_multi([both_bytes.as_slice(), lpg_bytes.as_slice()]).unwrap();

        assert_eq!(merged.graph_model(), GraphModel::Both);
        assert_eq!(both_default, (2, 1));
        assert_eq!(lpg_default, (4, 2));
        assert_eq!(
            (
                merged.store_arc().peek_next_node_id(),
                merged.store_arc().peek_next_edge_id(),
            ),
            lpg_default
        );
        let restored_both_named = merged.store_arc().graph("both_reserved").unwrap();
        assert_eq!(
            (
                restored_both_named.peek_next_node_id(),
                restored_both_named.peek_next_edge_id(),
            ),
            both_named
        );
        let restored_lpg_named = merged.store_arc().graph("lpg_reserved").unwrap();
        assert_eq!(
            (
                restored_lpg_named.peek_next_node_id(),
                restored_lpg_named.peek_next_edge_id(),
            ),
            lpg_named
        );
        let merged_history = merged.rdf_dataset_history().unwrap();
        assert_eq!(merged_history.store_id(), merged.store_id());
        assert_eq!(merged_history.completeness(), source_history.completeness());
        assert_eq!(
            merged_history.next_graph_incarnation(),
            source_history.next_graph_incarnation()
        );
        assert_eq!(merged_history.graph_lives(), source_history.graph_lives());
        assert_eq!(
            merged_history.quad_versions(),
            source_history.quad_versions()
        );
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn current_snapshot_cannot_hide_rdf_history_behind_unknown_model_tag() {
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
        let mut snapshot = decode_snapshot_bytes(&db.export_snapshot().unwrap()).unwrap();
        assert!(!snapshot.rdf_dataset_history.is_empty());
        snapshot.graph_model = 3;
        let forged = encode_snapshot_bytes(&snapshot).unwrap();

        assert!(GrafeoDB::import_snapshot(&forged).is_err());
    }

    #[test]
    #[cfg(feature = "triple-store")]
    fn graph_model_v6_both_roundtrips_on_import() {
        let db =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
        db.create_node(&["Person"]);
        let bytes = db.export_snapshot().unwrap();
        let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
        assert_eq!(restored.graph_model(), GraphModel::Both);
        assert_eq!(restored.node_count(), 1);
    }

    #[test]
    fn snapshot_rejects_reserved_pending_epoch() {
        let bytes = make_snapshot_at_epoch(SNAPSHOT_VERSION, Vec::new(), Vec::new(), u64::MAX);
        let error = decode_snapshot_bytes(&bytes)
            .err()
            .expect("PENDING snapshot epoch must be rejected")
            .to_string();
        assert!(error.contains("reserved PENDING sentinel"), "{error}");
    }

    #[cfg(feature = "grafeo-file")]
    #[test]
    fn monolithic_loader_restores_current_complete_identity_and_model_histories()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::Arc;

        fn assert_model_history(
            model: GraphModel,
        ) -> std::result::Result<(), Box<dyn std::error::Error>> {
            let source = GrafeoDB::with_config(
                crate::config::Config::in_memory()
                    .with_graph_model(model)
                    .with_gc_interval(0),
            )?;
            let lpg_ids = if model != GraphModel::Rdf {
                let node =
                    source.create_node_with_props(&["Recorded"], [("value", Value::Int64(1))]);
                let other = source.create_node(&["Endpoint"]);
                assert!(node.is_valid() && other.is_valid());
                source.set_node_property(node, "value", Value::Int64(2))?;
                let edge = source.create_edge(node, other, "RECORDED");
                assert!(edge.is_valid());
                source.set_edge_property(edge, "revision", Value::Int64(1))?;
                source.set_edge_property(edge, "revision", Value::Int64(2))?;
                Some((node, edge))
            } else {
                None
            };
            #[cfg(feature = "triple-store")]
            if model != GraphModel::Lpg {
                use grafeo_core::graph::rdf::{Term, Triple};
                assert_eq!(
                    source.insert_rdf_valid_tai_ns(
                        [Triple::new(
                            Term::iri("http://ex.org/s"),
                            Term::iri("http://ex.org/p"),
                            Term::literal("recorded")
                        )],
                        -1_001,
                        2_003,
                    )?,
                    1
                );
            }
            let bytes = source.export_snapshot()?;
            assert_eq!(snapshot_info(&bytes)?.version, SNAPSHOT_VERSION);
            assert_eq!(
                source.world_identity().history(),
                HistoryCompleteness::Complete
            );
            let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new()?);
            let catalog = Arc::new(crate::catalog::Catalog::new());
            #[cfg(feature = "triple-store")]
            let rdf = Arc::new(grafeo_core::graph::rdf::RdfStore::new());
            #[cfg(feature = "triple-store")]
            let projections = Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());
            let (loaded_model, identity) = super::load_snapshot_into_store(
                #[cfg(feature = "cdc")]
                &crate::cdc::CdcLog::new(),
                &store,
                &catalog,
                #[cfg(feature = "triple-store")]
                &rdf,
                #[cfg(feature = "triple-store")]
                &projections,
                &bytes,
            )?;
            assert_eq!(loaded_model, model.as_u8());
            assert_eq!(identity, source.world_identity());
            assert_eq!(identity.history(), HistoryCompleteness::Complete);
            assert_eq!(store.current_epoch(), source.current_epoch());
            assert_eq!(
                (store.node_count(), store.edge_count()),
                (source.node_count(), source.edge_count())
            );
            assert_eq!(
                catalog.encode_wal_state_v1()?,
                source.catalog.encode_wal_state_v1()?
            );
            if let Some((node, edge)) = lpg_ids {
                assert_eq!(
                    store.node_property_history(node),
                    source.store_arc().node_property_history(node)
                );
                assert_eq!(
                    store.node_label_history(node),
                    source.store_arc().node_label_history(node)
                );
                assert_eq!(
                    store.edge_property_history(edge),
                    source.store_arc().edge_property_history(edge)
                );
                let restored_node = store.get_node(node).ok_or("loaded node missing")?;
                assert!(restored_node.has_label("Recorded"));
                assert_eq!(restored_node.get_property("value"), Some(&Value::Int64(2)));
                let restored_edge = store.get_edge(edge).ok_or("loaded edge missing")?;
                let source_edge = source.get_edge(edge).ok_or("source edge missing")?;
                assert_eq!(
                    (restored_edge.src, restored_edge.dst),
                    (source_edge.src, source_edge.dst)
                );
                assert_eq!(restored_edge.edge_type.as_str(), "RECORDED");
                assert_eq!(
                    restored_edge.get_property("revision"),
                    Some(&Value::Int64(2))
                );
            }
            #[cfg(feature = "triple-store")]
            {
                assert_eq!(rdf.store_id(), identity.store_id());
                assert_eq!(rdf.history_completeness(), HistoryCompleteness::Complete);
                assert_eq!(rdf.commit_epoch(), source.current_epoch());
                let history = rdf.dataset_history()?;
                let source_history = source.rdf_dataset_history()?;
                assert_eq!(history.store_id(), source_history.store_id());
                assert_eq!(history.completeness(), source_history.completeness());
                assert_eq!(
                    history.next_graph_incarnation(),
                    source_history.next_graph_incarnation()
                );
                assert_eq!(history.graph_lives(), source_history.graph_lives());
                assert_eq!(history.quad_versions(), source_history.quad_versions());
                assert!(projections.snapshot().is_empty());
                if model == GraphModel::Lpg {
                    assert_eq!(rdf.len(), 0);
                    assert!(history.graph_lives().is_empty());
                    assert!(history.quad_versions().is_empty());
                }
            }
            Ok(())
        }
        assert_model_history(GraphModel::Lpg)?;
        #[cfg(feature = "triple-store")]
        for model in [GraphModel::Rdf, GraphModel::Both] {
            assert_model_history(model)?;
        }
        Ok(())
    }

    #[cfg(feature = "grafeo-file")]
    #[test]
    fn monolithic_loader_rejects_endpoint_lifetime_escape_before_mutation() {
        let source = SnapshotNode {
            id: NodeId::new(1),
            lifetimes: vec![(EpochId::new(0), Some(EpochId::new(2)))],
            label_versions: vec![(EpochId::new(0), vec!["Source".to_string()])],
            properties: Vec::new(),
        };
        let destination = SnapshotNode {
            id: NodeId::new(2),
            lifetimes: vec![(EpochId::new(0), None)],
            label_versions: vec![(EpochId::new(0), vec!["Destination".to_string()])],
            properties: Vec::new(),
        };
        let edge = SnapshotEdge {
            id: EdgeId::new(1),
            src: source.id,
            dst: destination.id,
            edge_type: "ESCAPES".to_string(),
            lifetimes: vec![(EpochId::new(1), None)],
            properties: Vec::new(),
        };
        let bytes =
            make_snapshot_at_epoch(SNAPSHOT_VERSION, vec![source, destination], vec![edge], 3);

        let store = std::sync::Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let catalog = std::sync::Arc::new(crate::catalog::Catalog::new());
        #[cfg(feature = "triple-store")]
        let rdf_store = std::sync::Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        #[cfg(feature = "triple-store")]
        let projections =
            std::sync::Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());

        let error = super::load_snapshot_into_store(
            #[cfg(feature = "cdc")]
            &crate::cdc::CdcLog::new(),
            &store,
            &catalog,
            #[cfg(feature = "triple-store")]
            &rdf_store,
            #[cfg(feature = "triple-store")]
            &projections,
            &bytes,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outlives source node"), "{error}");
        assert_eq!(store.node_count(), 0, "validation must precede LPG restore");
        assert_eq!(store.edge_count(), 0, "validation must precede LPG restore");
        assert!(catalog.all_labels().is_empty());
        #[cfg(feature = "triple-store")]
        assert_eq!(rdf_store.len(), 0, "validation must precede RDF restore");
    }

    #[test]
    fn snapshots_v4_through_v9_reject_headers_before_payload_or_target_mutation()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let target = GrafeoDB::new_in_memory();
        let retained =
            target.create_node_with_props(&["Retained"], [("value", Value::from("untouched"))]);
        assert!(retained.is_valid());
        let before = target.export_snapshot()?;
        let epoch = target.current_epoch();
        let allocator = target.store_arc().peek_next_node_id();
        let current = GrafeoDB::new_in_memory().export_snapshot()?;
        assert_eq!(snapshot_info(&current)?.version, SNAPSHOT_VERSION);
        assert_eq!(
            GrafeoDB::import_snapshot(&current)?.export_snapshot()?,
            current
        );

        for version in 4..=9 {
            let mut current_body = current.clone();
            current_body[0] = version;
            // Header-only and malformed collection bytes prove rejection
            // precedes any predecessor decoder or allocation.
            let candidates = [
                vec![version],
                vec![version, 255, 255, 255, 255, 255],
                current_body,
            ];
            for bytes in candidates {
                for error in [
                    decode_snapshot_bytes(&bytes).err(),
                    snapshot_info(&bytes).err(),
                    GrafeoDB::import_snapshot(&bytes).err(),
                    target.restore_snapshot(&bytes).err(),
                ] {
                    let error = error.ok_or("predecessor snapshot was accepted")?;
                    assert!(matches!(error, Error::Serialization(_)), "{error}");
                    let message = error.to_string();
                    assert!(
                        message.contains("unsupported snapshot version"),
                        "{message}"
                    );
                    assert!(message.contains(&version.to_string()), "{message}");
                }
                assert_eq!(target.export_snapshot()?, before);
                assert_eq!(target.current_epoch(), epoch);
                assert_eq!(target.store_arc().peek_next_node_id(), allocator);
                assert!(target.get_node(retained).is_some());
            }
        }
        Ok(())
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn current_snapshot_rdf_valid_time_roundtrips_and_rejects_corrupt_bounds()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::types::TaiNanoseconds;
        use grafeo_core::graph::rdf::{Term, Triple};

        let source = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))?;
        let valid = ValidTimeInterval::from_tai_nanoseconds(-1_001, 2_003)?;
        assert_eq!(
            source.insert_rdf_valid_tai_ns(
                [Triple::new(
                    Term::iri("http://ex.org/current"),
                    Term::iri("http://ex.org/p"),
                    Term::literal("value"),
                )],
                valid.from().as_i128(),
                valid.to().as_i128(),
            )?,
            1
        );
        let bytes = source.export_snapshot()?;
        assert_eq!(snapshot_info(&bytes)?.version, SNAPSHOT_VERSION);
        let target = GrafeoDB::import_snapshot(&bytes)?;
        let history = target.rdf_dataset_history()?;
        assert_eq!(history.quad_versions().len(), 1);
        assert_eq!(history.quad_versions()[0].valid(), Some(valid));
        assert_eq!(target.export_snapshot()?, bytes);

        // The sole quad's final field is its optional canonical interval.
        // Preserve the real current writer's complete preceding history bytes.
        let suffix = bincode::serde::encode_to_vec(Some(valid), bincode::config::standard())?;
        for (invalid_suffix, diagnostic) in [
            (
                bincode::serde::encode_to_vec(
                    Some(TaiNanoseconds::new(10)),
                    bincode::config::standard(),
                )?,
                "snapshot RDF history decode failed",
            ),
            (
                bincode::serde::encode_to_vec(
                    Some((TaiNanoseconds::new(10), TaiNanoseconds::new(10))),
                    bincode::config::standard(),
                )?,
                "from < to",
            ),
            (
                bincode::serde::encode_to_vec(
                    Some((TaiNanoseconds::new(11), TaiNanoseconds::new(10))),
                    bincode::config::standard(),
                )?,
                "from < to",
            ),
        ] {
            let mut snapshot = decode_snapshot_bytes(&bytes)?;
            assert!(snapshot.rdf_dataset_history.ends_with(&suffix));
            snapshot
                .rdf_dataset_history
                .truncate(snapshot.rdf_dataset_history.len() - suffix.len());
            snapshot.rdf_dataset_history.extend(invalid_suffix);
            let invalid = encode_snapshot_bytes(&snapshot)?;
            for error in [
                snapshot_info(&invalid).err(),
                GrafeoDB::import_snapshot(&invalid).err(),
                target.restore_snapshot(&invalid).err(),
            ] {
                let error = error.ok_or("invalid current RDF interval was accepted")?;
                assert!(error.to_string().contains(diagnostic), "{error}");
            }
            assert_eq!(target.export_snapshot()?, bytes);
        }
        Ok(())
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn current_snapshot_rejects_canonical_alias_overlap_before_replacement()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_core::graph::rdf::{Quad, RdfQuadVersion, Term, Triple};
        for model in [GraphModel::Rdf, GraphModel::Both]
            .into_iter()
            .filter(|model| cfg!(feature = "lpg") || *model == GraphModel::Rdf)
        {
            let source = GrafeoDB::with_config(Config::in_memory().with_graph_model(model))?;
            source.batch_insert_rdf([Triple::new(
                Term::iri("urn:s"),
                Term::iri("urn:p"),
                Term::lang_literal("hello", "EN"),
            )])?;
            let original = source.export_snapshot()?;
            let target = GrafeoDB::with_config(Config::in_memory().with_graph_model(model))?;
            target.batch_insert_rdf([Triple::new(
                Term::iri("urn:sentinel"),
                Term::iri("urn:p"),
                Term::literal("retained"),
            )])?;
            let before = target.export_snapshot()?;
            let mut snapshot = decode_snapshot_bytes(&original)?;
            let (mut wire, _): (super::SnapshotRdfDatasetHistoryV1, usize) =
                bincode::serde::decode_from_slice(
                    &snapshot.rdf_dataset_history,
                    bincode::config::standard(),
                )?;
            let first = &wire.quad_versions[0];
            let duplicate = RdfQuadVersion::new(
                snapshot.world_identity.store_id(),
                Quad::new(Triple::new(
                    Term::iri("urn:s"),
                    Term::iri("urn:p"),
                    Term::lang_literal("hello", "en"),
                )),
                first.graph_incarnation(),
                first.tx(),
                first.valid(),
            )?;
            wire.quad_versions.push(duplicate);
            snapshot.rdf_dataset_history =
                bincode::serde::encode_to_vec(&wire, bincode::config::standard())?;
            let invalid = encode_snapshot_bytes(&snapshot)?;
            for error in [
                snapshot_info(&invalid).err(),
                GrafeoDB::import_snapshot(&invalid).err(),
                target.restore_snapshot(&invalid).err(),
            ] {
                let error = error.ok_or("canonical duplicate RDF snapshot accepted")?;
                assert!(error.to_string().contains("overlapping"), "{error}");
            }
            assert_eq!(target.export_snapshot()?, before);
        }
        Ok(())
    }

    #[cfg(not(feature = "triple-store"))]
    #[test]
    fn current_snapshots_with_rdf_state_fail_closed_without_triple_store()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let target = GrafeoDB::new_in_memory();
        let before = target.export_snapshot()?;
        for graph_model in [GraphModel::Rdf, GraphModel::Both] {
            let mut snapshot = decode_snapshot_bytes(&before)?;
            snapshot.graph_model = graph_model.as_u8();
            if graph_model == GraphModel::Rdf {
                snapshot.graphs.clear();
            }
            // Feature admission must reject without decoding this RDF body.
            snapshot.rdf_dataset_history = vec![255];
            let bytes = encode_snapshot_bytes(&snapshot)?;
            for error in [
                snapshot_info(&bytes).err(),
                GrafeoDB::import_snapshot(&bytes).err(),
                target.restore_snapshot(&bytes).err(),
            ] {
                let error = error.ok_or("RDF snapshot accepted without triple-store")?;
                assert!(
                    error
                        .to_string()
                        .contains("triple-store support is disabled"),
                    "{error}"
                );
            }
            assert_eq!(target.export_snapshot()?, before);
        }
        Ok(())
    }

    fn snapshot_with_label_history(
        lifetimes: Vec<(EpochId, Option<EpochId>)>,
        label_versions: Vec<(EpochId, Vec<String>)>,
        epoch: u64,
    ) -> Vec<u8> {
        make_snapshot_at_epoch(
            SNAPSHOT_VERSION,
            vec![SnapshotNode {
                id: NodeId::new(81),
                lifetimes,
                label_versions,
                properties: Vec::new(),
            }],
            Vec::new(),
            epoch,
        )
    }

    fn raw_label_history(db: &GrafeoDB) -> Vec<(EpochId, Vec<String>)> {
        db.store_arc()
            .node_label_history(NodeId::new(81))
            .into_iter()
            .map(|(epoch, labels)| {
                (
                    epoch,
                    labels.into_iter().map(|label| label.to_string()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn portable_v10_preserves_label_change_at_structural_delete_boundary() {
        let expected = vec![
            (EpochId::new(1), vec!["Alive".to_string()]),
            (
                EpochId::new(3),
                vec!["Alive".to_string(), "AtDelete".to_string()],
            ),
        ];
        let bytes = snapshot_with_label_history(
            vec![(EpochId::new(1), Some(EpochId::new(3)))],
            expected.clone(),
            3,
        );
        let source = GrafeoDB::import_snapshot(&bytes).unwrap();
        assert_eq!(raw_label_history(&source), expected);

        let restored = GrafeoDB::import_snapshot(&source.export_snapshot().unwrap()).unwrap();
        assert_eq!(raw_label_history(&restored), expected);
        assert!(restored.get_node(NodeId::new(81)).is_none());
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn compact_recompact_preserves_label_change_at_delete_boundary() {
        let expected = vec![
            (EpochId::new(1), vec!["Alive".to_string()]),
            (
                EpochId::new(3),
                vec!["Alive".to_string(), "AtDelete".to_string()],
            ),
        ];
        let bytes = snapshot_with_label_history(
            vec![(EpochId::new(1), Some(EpochId::new(3)))],
            expected.clone(),
            3,
        );
        let mut source = GrafeoDB::import_snapshot(&bytes).unwrap();
        source.compact().unwrap();
        source.compact().unwrap();

        let restored = GrafeoDB::import_snapshot(&source.export_snapshot().unwrap()).unwrap();
        assert_eq!(raw_label_history(&restored), expected);
        assert!(restored.get_node(NodeId::new(81)).is_none());
    }

    #[cfg(feature = "wal")]
    #[test]
    fn save_preserves_all_same_epoch_label_transitions() {
        let expected = vec![
            (EpochId::new(1), vec!["A".to_string()]),
            (EpochId::new(1), vec!["A".to_string(), "B".to_string()]),
            (EpochId::new(1), vec!["B".to_string()]),
        ];
        let bytes = snapshot_with_label_history(vec![(EpochId::new(1), None)], expected.clone(), 1);
        let source = GrafeoDB::import_snapshot(&bytes).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("same-epoch-labels");
        source.save(&destination).unwrap();

        let restored = GrafeoDB::open(&destination).unwrap();
        assert_eq!(raw_label_history(&restored), expected);
        restored.close().unwrap();
    }

    #[cfg(feature = "wal")]
    #[test]
    fn save_preserves_complete_repeated_and_empty_label_images() {
        let expected = vec![
            (EpochId::new(1), vec!["A".to_string(), "B".to_string()]),
            (EpochId::new(2), vec!["C".to_string(), "D".to_string()]),
            (EpochId::new(2), vec!["C".to_string(), "D".to_string()]),
            (EpochId::new(2), Vec::new()),
        ];
        let bytes = snapshot_with_label_history(
            vec![(EpochId::new(1), Some(EpochId::new(2)))],
            expected.clone(),
            2,
        );
        let source = GrafeoDB::import_snapshot(&bytes).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("complete-label-images");
        source.save(&destination).unwrap();
        let restored = GrafeoDB::open(&destination).unwrap();
        assert_eq!(raw_label_history(&restored), expected);
        assert!(restored.get_node(NodeId::new(81)).is_none());
        restored.close().unwrap();
    }

    #[cfg(feature = "wal")]
    #[test]
    fn save_preserves_plural_structural_lifetimes()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let bytes = snapshot_with_label_history(
            vec![
                (EpochId::new(1), Some(EpochId::new(2))),
                (EpochId::new(3), None),
            ],
            vec![
                (EpochId::new(1), vec!["First".to_string()]),
                (EpochId::new(3), vec!["Second".to_string()]),
            ],
            3,
        );
        let source = GrafeoDB::import_snapshot(&bytes)?;
        let before = source.export_snapshot()?;
        let temp = tempfile::tempdir()?;
        for name in ["plural-lifetimes", "plural-lifetimes.grafeo"] {
            let destination = temp.path().join(name);
            source.save(&destination)?;
            let restored = GrafeoDB::open(&destination)?;
            assert_eq!(restored.export_snapshot()?, before);
            assert_eq!(
                restored.store_arc().get_node_history(NodeId::new(81)).len(),
                2
            );
            restored.close()?;
        }
        assert_eq!(source.export_snapshot()?, before);
        Ok(())
    }

    #[cfg(feature = "wal")]
    #[test]
    fn save_preserves_last_real_epoch_without_administrative_publications()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = GrafeoDB::new_in_memory();
        let final_real = EpochId::PENDING.as_u64() - 1;
        source
            .transaction_manager
            .try_sync_epoch(EpochId::new(final_real))?;
        source.lpg_store().sync_epoch(EpochId::new(final_real));
        let before = source.export_snapshot()?;
        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("last-real-epoch");
        source.save(&destination)?;
        let restored = GrafeoDB::open(&destination)?;
        assert_eq!(
            restored.transaction_manager.current_epoch().as_u64(),
            final_real
        );
        assert_eq!(restored.export_snapshot()?, before);
        assert_eq!(source.export_snapshot()?, before);
        restored.close()?;
        Ok(())
    }

    fn first_column_strings(db: &GrafeoDB, query: &str) -> Vec<String> {
        let mut values: Vec<_> = db
            .session()
            .execute(query)
            .unwrap()
            .rows()
            .iter()
            .filter_map(|row| row[0].as_str().map(ToString::to_string))
            .collect();
        values.sort();
        values
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn current_snapshot_named_constraints_roundtrip_enforce_and_drop() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for statement in [
            "CREATE CONSTRAINT c_unique FOR (p:Person) ON (p.email) UNIQUE",
            "CREATE CONSTRAINT c_key FOR (p:Person) ON (p.tenant, p.id) NODE KEY",
            "CREATE CONSTRAINT c_required FOR (p:Person) ON (p.name) NOT NULL",
            "CREATE CONSTRAINT c_exists FOR (p:Person) ON (p.handle) EXISTS",
        ] {
            session.execute(statement).unwrap();
        }
        let snapshot = db.export_snapshot().unwrap();
        assert_eq!(snapshot_info(&snapshot).unwrap().version, SNAPSHOT_VERSION);
        let restored = GrafeoDB::import_snapshot(&snapshot).unwrap();
        assert_eq!(
            first_column_strings(&restored, "SHOW CONSTRAINTS"),
            ["c_exists", "c_key", "c_required", "c_unique"]
                .map(ToString::to_string)
                .to_vec()
        );

        let restored_session = restored.session();
        restored_session
            .execute("INSERT (:Person {tenant:'t', id:'1', name:'A', handle:'a', email:'a@x'})")
            .unwrap();
        assert!(
            restored_session
                .execute("INSERT (:Person {tenant:'u', id:'2', name:'B', handle:'b', email:'a@x'})")
                .is_err(),
            "restored UNIQUE constraint must be enforced"
        );
        for name in ["c_exists", "c_key", "c_required", "c_unique"] {
            restored_session
                .execute(&format!("DROP CONSTRAINT {name}"))
                .unwrap();
        }
        assert!(first_column_strings(&restored, "SHOW CONSTRAINTS").is_empty());
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn restore_v10_nested_decode_failure_leaves_target_unchanged() {
        let source = GrafeoDB::new_in_memory();
        source
            .session()
            .execute("INSERT (:Fresh {name:'source'})")
            .unwrap();
        let incoming = source.export_snapshot().unwrap();

        for corrupt_catalog in [true, false] {
            let target = GrafeoDB::new_in_memory();
            target
                .session()
                .execute("INSERT (:Stale {name:'target'})")
                .unwrap();
            let before = target.export_snapshot().unwrap();
            let mut decoded = decode_snapshot_bytes(&incoming).unwrap();
            if corrupt_catalog {
                decoded.catalog_state.push(0xff);
            } else {
                decoded.text_indexes.push(0xff);
            }
            let malformed = encode_snapshot_bytes(&decoded).unwrap();
            assert!(target.restore_snapshot(&malformed).is_err());
            assert_eq!(target.export_snapshot().unwrap(), before);
        }
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn restore_v10_exactly_removes_target_only_catalog_and_indexes() {
        let source = GrafeoDB::new_in_memory();
        let source_session = source.session();
        source_session
            .execute("INSERT (:Fresh {email:'fresh@example.com'})")
            .unwrap();
        source_session
            .execute("CREATE CONSTRAINT fresh_constraint FOR (n:Fresh) ON (n.email) UNIQUE")
            .unwrap();
        source_session
            .execute("CREATE INDEX fresh_index FOR (n:Fresh) ON (n.email)")
            .unwrap();
        source_session.execute("CREATE GRAPH kept").unwrap();
        let incoming = source.export_snapshot().unwrap();

        let target = GrafeoDB::new_in_memory();
        let target_session = target.session();
        target_session
            .execute("INSERT (:Stale {slug:'stale'})")
            .unwrap();
        target_session
            .execute("CREATE CONSTRAINT stale_constraint FOR (n:Stale) ON (n.slug) UNIQUE")
            .unwrap();
        target_session
            .execute("CREATE INDEX stale_index FOR (n:Stale) ON (n.slug)")
            .unwrap();
        target_session.execute("CREATE GRAPH stale_graph").unwrap();
        let _ = target_session
            .execute("MATCH (n:Stale) RETURN n.slug")
            .unwrap();

        target.restore_snapshot(&incoming).unwrap();
        assert_eq!(target.export_snapshot().unwrap(), incoming);
        assert_eq!(
            first_column_strings(&target, "SHOW CONSTRAINTS"),
            vec!["fresh_constraint".to_string()]
        );
        assert_eq!(
            first_column_strings(&target, "SHOW INDEXES"),
            vec!["fresh_index".to_string()]
        );
        assert_eq!(target.node_count(), 1);
        assert_eq!(target.lpg_store().graph_names(), vec!["kept".to_string()]);
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn restore_v10_exactly_removes_target_only_rdf_history() {
        use grafeo_common::types::EpochId;
        use grafeo_core::graph::rdf::{Term, Triple};

        let source =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
        let incoming = source.export_snapshot().unwrap();

        let target =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
        let stale = Triple::new(
            Term::iri("http://example.org/stale"),
            Term::iri("http://example.org/p"),
            Term::literal("gone"),
        );
        target
            .rdf_store
            .try_set_commit_epoch(EpochId::new(3))
            .unwrap();
        assert!(target.rdf_store.insert(stale.clone()));
        assert!(
            target
                .rdf_store
                .try_remove_at_epoch(&stale, EpochId::new(4))
                .unwrap()
        );
        assert!(target.rdf_store.has_history());

        target.restore_snapshot(&incoming).unwrap();
        assert!(target.rdf_store.quad_history().is_empty());
        assert!(target.rdf_store.graph_names().is_empty());
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn open_multi_v10_deduplicates_identical_owners_and_rejects_conflicts()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        fn snapshot_with_names(
            constraint_name: &str,
            constraint_label: &str,
            constraint_property: &str,
            index_name: &str,
            reserve_first: bool,
        ) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
            let db = GrafeoDB::new_in_memory();
            if reserve_first {
                let retired = db.create_index(crate::CreateIndexRequest {
                    graph: GraphPath::root(),
                    name: None,
                    label: None,
                    property: "reserved-gap".into(),
                    kind: crate::IndexCreateKind::Property,
                })?;
                assert_eq!(retired.as_u32(), 0);
                assert!(db.drop_index(retired)?);
            }
            let session = db.session();
            session
                .execute(&format!(
                    "CREATE CONSTRAINT {constraint_name} FOR (n:{constraint_label}) ON (n.{constraint_property}) UNIQUE"
                ))
                ?;
            session.execute(&format!(
                "CREATE INDEX {index_name} FOR (n:{constraint_label}) ON (n.{constraint_property})"
            ))?;
            let owner = db
                .catalog
                .find_index_by_name(index_name)
                .ok_or("live fixture owner missing")?;
            assert_eq!(owner.as_u32(), u32::from(reserve_first));
            assert_eq!(
                db.catalog.index_allocator_high_water(),
                u32::from(reserve_first) + 1
            );
            Ok(db.export_snapshot()?)
        }

        let identical = snapshot_with_names("c_shared", "Person", "email", "idx_shared", false)?;
        let merged = GrafeoDB::open_multi([&identical, &identical]).unwrap();
        assert_eq!(
            first_column_strings(&merged, "SHOW CONSTRAINTS"),
            vec!["c_shared".to_string()]
        );
        assert_eq!(
            first_column_strings(&merged, "SHOW INDEXES"),
            vec!["idx_shared".to_string()]
        );

        assert_eq!(
            merged
                .catalog
                .find_index_by_name("idx_shared")
                .ok_or("merged owner missing")?
                .as_u32(),
            0
        );
        assert_eq!(merged.catalog.index_allocator_high_water(), 1);
        let colliding = snapshot_with_names("c_other", "Account", "handle", "idx_other", false)?;
        let error = GrafeoDB::open_multi([&identical, &colliding])
            .err()
            .ok_or("independent ID0 owners were silently renumbered")?;
        assert!(error.to_string().contains("owner"), "{error}");

        let disjoint = snapshot_with_names("c_other", "Account", "handle", "idx_other", true)?;
        let union = GrafeoDB::open_multi([&identical, &disjoint]).unwrap();
        assert_eq!(
            first_column_strings(&union, "SHOW CONSTRAINTS"),
            ["c_other", "c_shared"].map(ToString::to_string).to_vec()
        );
        assert_eq!(
            first_column_strings(&union, "SHOW INDEXES"),
            ["idx_other", "idx_shared"]
                .map(ToString::to_string)
                .to_vec()
        );

        assert_eq!(
            union
                .catalog
                .find_index_by_name("idx_shared")
                .ok_or("first union owner missing")?
                .as_u32(),
            0
        );
        assert_eq!(
            union
                .catalog
                .find_index_by_name("idx_other")
                .ok_or("second union owner missing")?
                .as_u32(),
            1
        );
        assert_eq!(union.catalog.index_allocator_high_water(), 2);

        let conflicting_constraint =
            snapshot_with_names("c_shared", "Account", "handle", "idx_other", true)?;
        let error = GrafeoDB::open_multi([&identical, &conflicting_constraint])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("c_shared")
                && error.contains("snapshot[0]")
                && error.contains("snapshot[1]"),
            "{error}"
        );

        let equivalent_target =
            snapshot_with_names("c_alias", "Person", "email", "idx_shared", false)?;
        let error = GrafeoDB::open_multi([&identical, &equivalent_target])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("c_shared") && error.contains("c_alias"),
            "{error}"
        );

        fn index_snapshot(label: &str, property: &str, name: &str) -> Vec<u8> {
            let db = GrafeoDB::new_in_memory();
            db.session()
                .execute(&format!(
                    "CREATE INDEX {name} FOR (n:{label}) ON (n.{property})"
                ))
                .unwrap();
            db.export_snapshot().unwrap()
        }
        let alias_a = index_snapshot("Person", "email", "idx_a");
        let alias_b = index_snapshot("Account", "email", "idx_b");
        let error = GrafeoDB::open_multi([&alias_a, &alias_b])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("index owner ID 0 conflicts"), "{error}");

        let name_a = index_snapshot("Person", "email", "idx_name");
        let name_b = index_snapshot("Account", "handle", "idx_name");
        let error = GrafeoDB::open_multi([&name_a, &name_b])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("index owner ID 0 conflicts"), "{error}");
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
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
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn test_restore_rejects_duplicate_node_ids() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    lifetimes: vec![(EpochId::INITIAL, None)],
                    label_versions: vec![(EpochId::INITIAL, vec!["A".into()])],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(0),
                    lifetimes: vec![(EpochId::INITIAL, None)],
                    label_versions: vec![(EpochId::INITIAL, vec!["B".into()])],
                    properties: vec![],
                },
            ],
            vec![],
        );

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate node ID"), "got: {err}");
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn test_restore_rejects_duplicate_edge_ids() {
        let db = GrafeoDB::new_in_memory();

        let bytes = make_snapshot(
            SNAPSHOT_VERSION,
            vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    lifetimes: vec![(EpochId::INITIAL, None)],
                    label_versions: vec![(EpochId::INITIAL, vec![])],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(1),
                    lifetimes: vec![(EpochId::INITIAL, None)],
                    label_versions: vec![(EpochId::INITIAL, vec![])],
                    properties: vec![],
                },
            ],
            vec![
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    lifetimes: vec![(EpochId::INITIAL, None)],
                    properties: vec![],
                },
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    lifetimes: vec![(EpochId::INITIAL, None)],
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
                lifetimes: vec![(EpochId::INITIAL, None)],
                label_versions: vec![(EpochId::INITIAL, vec![])],
                properties: vec![],
            }],
            vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(999),
                dst: NodeId::new(0),
                edge_type: "REL".into(),
                lifetimes: vec![(EpochId::INITIAL, None)],
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
                lifetimes: vec![(EpochId::INITIAL, None)],
                label_versions: vec![(EpochId::INITIAL, vec![])],
                properties: vec![],
            }],
            vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(0),
                dst: NodeId::new(999),
                edge_type: "REL".into(),
                lifetimes: vec![(EpochId::INITIAL, None)],
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
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn test_snapshot_roundtrip_property_index() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', email: 'alix@example.com'})")
            .unwrap();
        db.create_index(crate::CreateIndexRequest {
            graph: grafeo_common::types::GraphPath::root(),
            name: None,
            label: None,
            property: "email".into(),
            kind: crate::IndexCreateKind::Property,
        })
        .unwrap();
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
        )
        .expect("set node property");
        let n2 = db.create_node(&["Doc"]);
        db.set_node_property(
            n2,
            "embedding",
            Value::Vector(Arc::from([0.0_f32, 1.0, 0.0])),
        )
        .expect("set node property");

        db.create_index(crate::CreateIndexRequest {
            graph: grafeo_common::types::GraphPath::root(),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: crate::IndexCreateKind::Vector {
                dimensions: None,
                metric: Some("cosine".into()),
                m: Some(4),
                ef_construction: Some(32),
                ef: None,
                quantization: None,
            },
        })
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
    fn portable_snapshot_preserves_commit_born_text_and_exact_historical_scores()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use grafeo_common::storage::Section;
        use grafeo_common::types::TransactionId;
        use grafeo_core::graph::lpg::PhysicalIndexKey;
        use grafeo_core::index::text::TextIndexSection;

        let db = GrafeoDB::with_config(crate::config::Config::in_memory().with_gc_interval(0))?;
        let n1 = db.create_node(&["Article"]);
        db.set_node_property(n1, "body", Value::from("rust graph database"))?;
        let n2 = db.create_node(&["Article"]);
        db.set_node_property(n2, "body", Value::from("python web framework"))?;
        let owner = db.create_index(crate::CreateIndexRequest {
            graph: GraphPath::root(),
            name: None,
            label: Some("Article".into()),
            property: "body".into(),
            kind: crate::IndexCreateKind::Text {
                min_token_length: None,
            },
        })?;
        let born = db.current_epoch();
        let source_text = db
            .store_arc()
            .get_text_index("Article", "body")
            .ok_or("Text owner missing")?;
        let score = source_text
            .read()
            .score_document_visible(n1, "rust", born, TransactionId::INVALID, None, false)?
            .ok_or("birth score missing")?;
        assert!(score > 0.0);
        db.set_node_property(n1, "body", Value::from("current revision"))?;
        let exact = TextIndexSection::from_views(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Article", "body"),
            source_text,
        )])
        .serialize()?;
        let bytes = db.export_snapshot()?;
        assert_eq!(decode_snapshot_bytes(&bytes)?.text_indexes, exact);
        let artifact = db.export_snapshot_artifact()?;
        for restored in [
            GrafeoDB::import_snapshot(&bytes)?,
            GrafeoDB::import_snapshot_artifact(&artifact)?,
            db.to_memory()?,
        ] {
            assert_eq!(
                restored.catalog.get_index(owner),
                db.catalog.get_index(owner)
            );
            let text = restored
                .store_arc()
                .get_text_index("Article", "body")
                .ok_or("restored Text missing")?;
            assert_eq!(
                text.read()
                    .score_document_visible(n1, "rust", born, TransactionId::INVALID, None, false,)?
                    .map(f64::to_bits),
                Some(score.to_bits())
            );
            assert_eq!(text.read().score_document(n1, "rust"), 0.0);
            assert_eq!(
                TextIndexSection::from_views(vec![(
                    PhysicalIndexKey::text(GraphPath::root(), "Article", "body"),
                    text,
                )])
                .serialize()?,
                exact
            );
            // Live replacement removes a newer target revision while retaining
            // the root Arc and restoring the source's exact historical image.
            let retained_root = std::sync::Arc::clone(restored.store_arc());
            restored.set_node_property(n1, "body", Value::from("live target revision"))?;
            restored.restore_snapshot(&bytes)?;
            assert!(std::sync::Arc::ptr_eq(&retained_root, restored.store_arc()));
            assert_eq!(restored.export_snapshot()?, bytes);
            assert_eq!(
                restored.catalog.get_index(owner),
                db.catalog.get_index(owner)
            );
            assert_eq!(
                restored.catalog.index_allocator_high_water(),
                db.catalog.index_allocator_high_water()
            );
            let text = restored
                .store_arc()
                .get_text_index("Article", "body")
                .ok_or("live-restored Text missing")?;
            assert_eq!(
                text.read()
                    .score_document_visible(n1, "rust", born, TransactionId::INVALID, None, false)?
                    .map(f64::to_bits),
                Some(score.to_bits())
            );
            assert_eq!(text.read().score_document(n1, "rust"), 0.0);
            assert_eq!(
                TextIndexSection::from_views(vec![(
                    PhysicalIndexKey::text(GraphPath::root(), "Article", "body"),
                    text,
                )])
                .serialize()?,
                exact
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    #[test]
    fn snapshot_v10_preserves_named_graph_index_ownership_and_full_configs()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::catalog::IndexConfiguration;
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
        };
        use parking_lot::RwLock;
        use std::sync::Arc;

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH analytics").unwrap();
        session.execute("USE GRAPH analytics").unwrap();
        session
            .execute("INSERT (:Doc {embedding: vector([1.0, 0.0, 0.0]), body:'graph text'})")
            .unwrap();
        session.execute("USE GRAPH DEFAULT").unwrap();

        let graph = db.store_arc().graph("analytics").unwrap();
        let config = HnswConfig {
            dimensions: 3,
            metric: DistanceMetric::Cosine,
            m: 12,
            m_max: 29,
            ef_construction: 91,
            ef: 37,
            ml: 0.61,
            alpha: 1.15,
            max_elements: Some(64),
        };
        let bm25 = BM25Config { k1: 1.7, b: 0.42 };
        let path = GraphPath::from_components(&["analytics"])?;
        let node = graph
            .nodes_by_label("Doc")
            .into_iter()
            .next()
            .ok_or("fixture Doc missing")?;
        let label = db.catalog.get_or_create_label("Doc")?;
        let embedding = db.catalog.get_or_create_property_key("embedding")?;
        let body = db.catalog.get_or_create_property_key("body")?;
        // Full options are not all exposed by CreateIndexRequest. This owning
        // fixture publishes matching resolved catalog and physical state;
        // there is no detached descriptor or mismatched existing owner.
        let (vector_owner, text_owner) = db.transaction_manager.with_write_authority(
            || -> std::result::Result<_, Box<dyn std::error::Error>> {
                let vector_owner = db.catalog.create_index(
                    Some("graph_vec"),
                    label,
                    embedding,
                    path.clone(),
                    IndexConfiguration::Vector {
                        config: config.clone(),
                        quantization: QuantizationType::Binary,
                    },
                )?;
                let text_owner = db.catalog.create_index(
                    Some("graph_text"),
                    label,
                    body,
                    path.clone(),
                    IndexConfiguration::Text {
                        config: bm25.clone(),
                        min_token_length: 2,
                    },
                )?;
                let vector = QuantizedHnswIndex::new(config.clone(), QuantizationType::Binary);
                vector.insert(node, &[1.0, 0.0, 0.0]);
                graph.add_vector_index(
                    "Doc",
                    "embedding",
                    Arc::new(VectorIndexKind::Quantized(vector)),
                );
                let mut text = InvertedIndex::with_simple_tokenizer(bm25, 2);
                text.insert_versioned(node, "graph text", db.current_epoch(), None);
                graph.add_text_index("Doc", "body", Arc::new(RwLock::new(text)));
                Ok((vector_owner, text_owner))
            },
        )?;
        let floor = db.catalog.index_allocator_high_water();
        let bytes = db.export_snapshot()?;
        let restored = GrafeoDB::import_snapshot(&bytes)?;
        assert_eq!(restored.export_snapshot()?, bytes);
        assert_eq!(
            restored.catalog.get_index(vector_owner),
            db.catalog.get_index(vector_owner)
        );
        assert_eq!(
            restored.catalog.get_index(text_owner),
            db.catalog.get_index(text_owner)
        );
        assert_eq!(restored.catalog.index_allocator_high_water(), floor);
        assert_eq!(
            first_column_strings(&restored, "SHOW INDEXES"),
            ["graph_text", "graph_vec"]
                .map(ToString::to_string)
                .to_vec()
        );
        let graph = restored.store_arc().graph("analytics").unwrap();
        let vector = graph.get_vector_index("Doc", "embedding").unwrap();
        let restored_config = vector.config();
        assert_eq!(restored_config.dimensions, 3);
        assert_eq!(restored_config.m, 12);
        assert_eq!(restored_config.m_max, 29);
        assert_eq!(restored_config.ef_construction, 91);
        assert_eq!(restored_config.ef, 37);
        assert_eq!(restored_config.ml.to_bits(), 0.61_f64.to_bits());
        assert_eq!(restored_config.alpha.to_bits(), 1.15_f32.to_bits());
        assert_eq!(restored_config.max_elements, Some(64));
        assert_eq!(vector.quantization_type(), Some(QuantizationType::Binary));
        let text = graph.get_text_index("Doc", "body").unwrap();
        let text = text.read();
        assert_eq!(text.config().k1.to_bits(), 1.7_f64.to_bits());
        assert_eq!(text.config().b.to_bits(), 0.42_f64.to_bits());
        drop(text);

        restored.session().execute("DROP INDEX graph_vec").unwrap();
        assert!(graph.get_vector_index("Doc", "embedding").is_none());
        Ok(())
    }

    #[cfg(feature = "wal")]
    #[test]
    fn save_round_trips_empty_named_graph_distinct_from_default() {
        use grafeo_common::types::PropertyKey;
        let db = GrafeoDB::new_in_memory();
        let default = db.create_node(&["DefaultOnly"]);
        db.set_node_property(default, "scope", Value::from("default"))
            .expect("set node property");
        let empty_named = db
            .store_arc()
            .graph_or_create("")
            .expect("create empty-name graph");
        let sentinel = empty_named.create_node(&["EmptyNamedOnly"]);
        empty_named.set_node_property(sentinel, "scope", Value::from("named-empty"));

        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("empty-named-graph");
        db.save(&destination)
            .expect("the exact save preserves an empty named graph");

        let restored = GrafeoDB::open(&destination).expect("open exact save");
        assert_eq!(
            crate::database::testing::root_lpg_store(&restored)
                .get_node_property(default, &PropertyKey::new("scope")),
            Some(Value::from("default")),
            "default-graph data must remain in the default graph"
        );
        let restored_empty = restored
            .store_arc()
            .graph("")
            .expect("restore the legal empty named graph");
        assert_eq!(
            restored_empty.get_node_property(sentinel, &PropertyKey::new("scope")),
            Some(Value::from("named-empty")),
            "the empty named graph must not alias the default graph"
        );
        restored.close().expect("close restored container");

        assert!(db.get_node(default).is_some());
        assert_eq!(
            empty_named.get_node_property(sentinel, &PropertyKey::new("scope")),
            Some(Value::from("named-empty")),
            "save must leave both source graph scopes untouched"
        );
    }

    #[cfg(feature = "wal")]
    #[test]
    fn save_preserves_recursive_paths_and_exact_graph_histories()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = GrafeoDB::new_in_memory();
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["literal"])?,
            GraphPath::from_components(&["literal", "slash"])?,
            GraphPath::from_components(&["literal/slash"])?,
        ];
        source.transaction_manager.with_write_authority(
            || -> grafeo_common::utils::error::Result<()> {
                for graph in &paths {
                    super::create_snapshot_graph(source.store_arc(), graph)?;
                }
                Ok(())
            },
        )?;
        for (ordinal, graph) in paths.iter().enumerate() {
            let session = source.session();
            session.use_graph_path(graph)?;
            let first = session.create_node(&["First"]);
            let second = session.create_node(&["Second"]);
            assert!(first.is_valid() && second.is_valid());
            let edge = session.create_edge(first, second, "LINK");
            assert!(edge.is_valid());
            session.set_node_property(first, "value", Value::Int64(i64::try_from(ordinal)?))?;
            session.set_node_property(first, "value", Value::Int64(99))?;
        }
        let source_bytes = source.export_snapshot()?;
        let captured = decode_snapshot_bytes(&source_bytes)?;
        let expected =
            bincode::serde::encode_to_vec(&captured.graphs, bincode::config::standard())?;
        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("recursive");
        source.save(&destination)?;
        let reopened = GrafeoDB::open(&destination)?;
        let restored = decode_snapshot_bytes(&reopened.export_snapshot()?)?;
        assert_eq!(
            bincode::serde::encode_to_vec(&restored.graphs, bincode::config::standard())?,
            expected
        );
        assert_eq!(reopened.export_snapshot()?, source_bytes);
        assert_eq!(source.export_snapshot()?, source_bytes);
        reopened.close()?;
        Ok(())
    }

    #[cfg(all(feature = "wal", feature = "vector-index"))]
    #[test]
    fn save_preserves_nondefault_hnsw_and_owner_floor()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::catalog::IndexConfiguration;
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, HnswIndex, QuantizationType, VectorIndexKind,
        };
        use std::sync::Arc;

        let db = GrafeoDB::new_in_memory();
        let config = HnswConfig::new(3, DistanceMetric::Cosine).with_ef(77);
        let label = db.catalog.get_or_create_label("Doc")?;
        let property = db.catalog.get_or_create_property_key("embedding")?;
        let owner = db.transaction_manager.with_write_authority(|| {
            let owner = db.catalog.create_index(
                Some("full_hnsw"),
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Vector {
                    config: config.clone(),
                    quantization: QuantizationType::None,
                },
            )?;
            db.store_arc().add_vector_index(
                "Doc",
                "embedding",
                Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config))),
            );
            Ok::<_, crate::catalog::CatalogError>(owner)
        })?;
        let before = db.export_snapshot()?;
        assert_eq!(
            db.store_arc()
                .get_vector_index("Doc", "embedding")
                .ok_or("physical vector missing")?
                .config()
                .ef,
            77
        );
        assert_eq!(db.catalog.index_allocator_high_water(), owner.as_u32() + 1);

        let temp = tempfile::tempdir()?;
        for name in ["exact-vector-copy", "exact-vector-copy.grafeo"] {
            let destination = temp.path().join(name);
            db.save(&destination)?;
            assert!(destination.is_file());
            let restored = GrafeoDB::open(&destination)?;
            assert_eq!(restored.export_snapshot()?, before);
            assert_eq!(
                restored.catalog.index_allocator_high_water(),
                owner.as_u32() + 1
            );
            assert_eq!(
                restored
                    .store_arc()
                    .get_vector_index("Doc", "embedding")
                    .ok_or("saved Vector missing")?
                    .config()
                    .ef,
                77
            );
            restored.close()?;
        }
        assert_eq!(db.export_snapshot()?, before);
        Ok(())
    }

    #[cfg(all(feature = "wal", feature = "text-index"))]
    #[test]
    fn save_preserves_nondefault_named_graph_bm25_and_owner_floor()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::catalog::IndexConfiguration;
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        use parking_lot::RwLock;
        use std::sync::Arc;

        let db = GrafeoDB::new_in_memory();
        db.create_graph("analytics")?;
        let graph = db
            .store_arc()
            .graph("analytics")
            .ok_or("analytics graph missing")?;
        let path = GraphPath::from_components(&["analytics"])?;
        let config = BM25Config { k1: 1.7, b: 0.42 };
        let label = db.catalog.get_or_create_label("Doc")?;
        let property = db.catalog.get_or_create_property_key("body")?;
        let owner = db.transaction_manager.with_write_authority(|| {
            let owner = db.catalog.create_index(
                Some("named_bm25"),
                label,
                property,
                path,
                IndexConfiguration::Text {
                    config: config.clone(),
                    min_token_length: 2,
                },
            )?;
            graph.add_text_index(
                "Doc",
                "body",
                Arc::new(RwLock::new(InvertedIndex::with_simple_tokenizer(config, 2))),
            );
            Ok::<_, crate::catalog::CatalogError>(owner)
        })?;
        let before = db.export_snapshot()?;
        let text = graph
            .get_text_index("Doc", "body")
            .ok_or("physical text missing")?;
        assert_eq!(text.read().config().k1.to_bits(), 1.7_f64.to_bits());
        assert_eq!(text.read().config().b.to_bits(), 0.42_f64.to_bits());
        assert!(text.read().has_simple_tokenizer(2));
        assert_eq!(db.catalog.index_allocator_high_water(), owner.as_u32() + 1);

        let temp = tempfile::tempdir()?;
        for name in ["exact-text-copy", "exact-text-copy.grafeo"] {
            let destination = temp.path().join(name);
            db.save(&destination)?;
            assert!(destination.is_file());
            let restored = GrafeoDB::open(&destination)?;
            assert_eq!(restored.export_snapshot()?, before);
            assert_eq!(
                restored.catalog.index_allocator_high_water(),
                owner.as_u32() + 1
            );
            let restored_text = restored
                .store_arc()
                .graph("analytics")
                .ok_or("saved named graph missing")?
                .get_text_index("Doc", "body")
                .ok_or("saved Text missing")?;
            assert!(restored_text.read().has_simple_tokenizer(2));
            restored.close()?;
        }
        assert_eq!(db.export_snapshot()?, before);
        Ok(())
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn open_multi_v10_rejects_quantization_conflict()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        fn snapshot(
            quantization: &str,
        ) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
            let db = GrafeoDB::new_in_memory();
            let owner = db.create_index(crate::CreateIndexRequest {
                graph: GraphPath::root(),
                name: Some("vec_idx".into()),
                label: Some("Doc".into()),
                property: "embedding".into(),
                kind: crate::IndexCreateKind::Vector {
                    dimensions: Some(3),
                    metric: Some("cosine".into()),
                    m: None,
                    ef_construction: None,
                    ef: None,
                    quantization: Some(quantization.into()),
                },
            })?;
            assert_eq!(db.catalog.index_allocator_high_water(), owner.as_u32() + 1);
            Ok(db.export_snapshot()?)
        }
        let scalar = snapshot("scalar")?;
        let binary = snapshot("binary")?;
        let error = GrafeoDB::open_multi([&scalar, &binary])
            .err()
            .ok_or("conflicting quantization owners were merged")?;
        assert!(
            matches!(error, Error::Serialization(ref message)
            if message.contains("index owner ID 0 conflicts")),
            "{error}"
        );
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn test_snapshot_roundtrip_property_index_via_restore() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', email: 'alix@example.com'})")
            .unwrap();
        let owner = db
            .create_index(crate::CreateIndexRequest {
                graph: grafeo_common::types::GraphPath::root(),
                name: None,
                label: None,
                property: "email".into(),
                kind: crate::IndexCreateKind::Property,
            })
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();

        // Mutate the database
        session
            .execute("INSERT (:Person {name: 'Gus', email: 'gus@example.com'})")
            .unwrap();
        assert!(db.drop_index(owner).unwrap());
        assert!(!db.has_property_index("email"));

        // Restore should bring back the index
        db.restore_snapshot(&snapshot).unwrap();
        assert!(db.has_property_index("email"));
    }

    // ── Content-addressed cold-base save/load ────────────────────────────

    #[cfg(all(feature = "compact-store", not(target_arch = "wasm32")))]
    fn assert_cold_base_gcst_rejected(bytes: &[u8], expected: &str) {
        let tmp = tempfile::tempdir().unwrap();
        let section_path = tmp.path().join("cold_base.section");
        let blocks = tmp.path().join("blocks");
        std::fs::create_dir(&blocks).unwrap();
        let retained_block = blocks.join("retained-block");
        let retained_bytes = b"retained content pool bytes";
        std::fs::write(&retained_block, retained_bytes).unwrap();
        std::fs::write(&section_path, bytes).unwrap();

        let Err(error) = GrafeoDB::load_cold_base_content_addressed(tmp.path()) else {
            panic!("unsupported compact wire must not produce a cold base");
        };
        assert!(
            matches!(&error, Error::Internal(message) if message.contains(expected)),
            "{error}"
        );
        assert_eq!(std::fs::read(&section_path).unwrap(), bytes);
        assert_eq!(std::fs::read(&retained_block).unwrap(), retained_bytes);
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 2);
        assert_eq!(std::fs::read_dir(&blocks).unwrap().count(), 1);
    }

    #[cfg(all(feature = "compact-store", not(target_arch = "wasm32")))]
    #[test]
    fn cold_base_content_addressed_rejects_authentic_gcst_predecessors() {
        let fixtures: [&[u8]; 8] = [
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v1.bin"),
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v2.bin"),
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v3.bin"),
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v4.bin"),
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v5.bin"),
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v6.bin"),
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v7.bin"),
            include_bytes!("../../tests/fixtures/gcst/rejected_gcst_v8.bin"),
        ];
        for (index, bytes) in fixtures.iter().enumerate() {
            let version = index + 1;
            assert_eq!(usize::from(bytes[4]), version);
            assert_cold_base_gcst_rejected(
                bytes,
                &format!("unsupported CompactStore section version {version}"),
            );
        }
    }

    #[cfg(all(feature = "compact-store", not(target_arch = "wasm32")))]
    #[test]
    fn cold_base_content_addressed_rejects_unknown_gcst_flags() {
        let current = include_bytes!("../../tests/fixtures/gcst/current_gcst_v9.bin");
        assert_eq!(current[4], 9);
        for flag in [0x04, 0x80] {
            let mut bytes = current.to_vec();
            bytes[5] |= flag;
            let crc_offset = bytes.len() - 4;
            let crc = crc32fast::hash(&bytes[..crc_offset]);
            bytes[crc_offset..].copy_from_slice(&crc.to_le_bytes());
            assert_cold_base_gcst_rejected(&bytes, "unsupported CompactStore flags");
        }
    }

    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    #[test]
    fn cold_base_content_addressed_round_trip() {
        use grafeo_common::types::{EpochId, PropertyKey};
        use grafeo_core::graph::traits::GraphStore;

        let tmp = tempfile::TempDir::new().unwrap();

        let mut db = GrafeoDB::new_in_memory();
        // Several multi-column nodes across two label tables.
        let alix = db.create_node(&["Person"]);
        db.set_node_property(alix, "name", Value::String("Alix".into()))
            .expect("set node property");
        db.set_node_property(alix, "age", Value::Int64(30))
            .expect("set node property");
        let gus = db.create_node(&["Person"]);
        db.set_node_property(gus, "name", Value::String("Gus".into()))
            .expect("set node property");
        db.set_node_property(gus, "age", Value::Int64(25))
            .expect("set node property");
        let ams = db.create_node(&["City"]);
        db.set_node_property(ams, "name", Value::String("Amsterdam".into()))
            .expect("set node property");
        db.set_node_property(ams, "population", Value::Int64(900_000))
            .expect("set node property");

        db.compact().unwrap();

        let written = db.save_cold_base_content_addressed(tmp.path()).unwrap();
        assert!(written > 0, "first save must write at least one block");

        // Reload the cold base from (section + blocks) only.
        let cold = GrafeoDB::load_cold_base_content_addressed(tmp.path()).unwrap();

        // Node count and several property values match the original cold base.
        let base = db.layered_store.as_ref().unwrap().base_store_arc();
        assert_eq!(cold.node_count(), base.node_count());
        assert_eq!(cold.node_count(), 3);

        let name = PropertyKey::new("name");
        let age = PropertyKey::new("age");
        let pop = PropertyKey::new("population");
        // get_node round-trips full property maps.
        let alix_node = cold.get_node(alix).expect("Alix in reloaded cold base");
        assert_eq!(
            alix_node.properties.get(&name),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(alix_node.properties.get(&age), Some(&Value::Int64(30)));
        let ams_node = cold.get_node(ams).expect("Amsterdam in reloaded cold base");
        assert_eq!(
            ams_node.properties.get(&name),
            Some(&Value::String("Amsterdam".into()))
        );
        assert_eq!(ams_node.properties.get(&pop), Some(&Value::Int64(900_000)));

        // As-of read agrees with the original base at the current epoch.
        let epoch = EpochId::new(u64::MAX);
        assert_eq!(
            cold.get_node_property_at_epoch(gus, &age, epoch),
            base.get_node_property_at_epoch(gus, &age, epoch),
        );
    }

    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    #[test]
    fn cold_base_content_addressed_cross_save_dedup() {
        let tmp = tempfile::TempDir::new().unwrap();

        let mut db = GrafeoDB::new_in_memory();
        // Two label tables so that changing a Person property leaves the City
        // table's column blocks byte-identical across generations.
        let alix = db.create_node(&["Person"]);
        db.set_node_property(alix, "name", Value::String("Alix".into()))
            .expect("set node property");
        db.set_node_property(alix, "age", Value::Int64(30))
            .expect("set node property");
        let gus = db.create_node(&["Person"]);
        db.set_node_property(gus, "name", Value::String("Gus".into()))
            .expect("set node property");
        db.set_node_property(gus, "age", Value::Int64(25))
            .expect("set node property");
        let ams = db.create_node(&["City"]);
        db.set_node_property(ams, "name", Value::String("Amsterdam".into()))
            .expect("set node property");
        db.set_node_property(ams, "population", Value::Int64(900_000))
            .expect("set node property");

        db.compact().unwrap();

        // First save: every referenced block is new.
        let n1 = db.save_cold_base_content_addressed(tmp.path()).unwrap();
        assert!(n1 > 0, "first save writes all blocks");

        // Change exactly ONE node's property, then fold it into a fresh cold
        // base via recompact(). The write must go through the LayeredStore (not
        // the raw overlay LpgStore): a node resident only in the cold base is
        // promoted into the overlay on write (`ensure_in_overlay`), which a
        // direct overlay write would skip. Advancing the overlay clock first
        // records the new value at a strictly newer committed epoch so it
        // supersedes the folded one, matching the real flow where a committing
        // transaction advances the epoch.
        use grafeo_core::graph::traits::GraphStoreMut;
        let layered = db.layered_store.as_ref().unwrap();
        let next_epoch = grafeo_common::types::EpochId::new(
            layered.overlay_store().current_epoch().as_u64() + 1,
        );
        layered.overlay_store().sync_epoch(next_epoch);
        layered.set_node_property(alix, "age", Value::Int64(31));
        db.compact().unwrap();

        // Second save to the SAME dir: unchanged column blocks (notably the
        // entire City table and the Person `name` column) are already on disk
        // and skipped, so strictly fewer blocks are written than the first save.
        let n2 = db.save_cold_base_content_addressed(tmp.path()).unwrap();
        assert!(
            n2 < n1,
            "cross-save dedup: changed re-save ({n2}) must write fewer blocks than the first ({n1})"
        );
        // Only one column (Person.age) changed, so exactly one new block lands.
        assert_eq!(
            n2, 1,
            "only the single changed column block is newly written"
        );

        // A no-op re-save of the unchanged generation writes nothing.
        let n3 = db.save_cold_base_content_addressed(tmp.path()).unwrap();
        assert_eq!(
            n3, 0,
            "re-saving an unchanged cold base writes no new blocks"
        );

        // And the reloaded base reflects the change.
        use grafeo_common::types::PropertyKey;
        let cold = GrafeoDB::load_cold_base_content_addressed(tmp.path()).unwrap();
        let alix_node =
            grafeo_core::graph::traits::GraphStore::get_node(&cold, alix).expect("Alix reloaded");
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("age")),
            Some(&Value::Int64(31))
        );
    }
}

#[cfg(all(test, feature = "lpg", feature = "cdc", feature = "gql"))]
mod retained_cdc_tests {
    use super::*;
    use crate::GrafeoDB;
    #[test]
    fn retained_feed_cannot_invent_an_unallocated_native_graph_lifetime() {
        let source = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
        source.session().execute("INSERT (:Source)").unwrap();
        let mut snapshot = decode_snapshot_bytes(&source.export_snapshot().unwrap()).unwrap();
        let mut forged = source
            .changes_between(EpochId::INITIAL, EpochId::PENDING)
            .unwrap()
            .remove(0);
        forged.timestamp = source.cdc_log.next_timestamp();
        forged.lpg_graph = Some(GraphPath::from_components(&["forged"]).unwrap());
        forged.graph_incarnation = Some(grafeo_common::types::GraphIncarnationId::new(999));
        source.cdc_log.record(forged);
        snapshot.cdc_checkpoint =
            crate::database::cdc_checkpoint::capture(&source, source.current_epoch()).unwrap();
        let target = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
        target.session().execute("INSERT (:Keep)").unwrap();
        let before = target.export_snapshot().unwrap();
        assert!(
            target
                .restore_snapshot(&encode_snapshot_bytes(&snapshot).unwrap())
                .is_err()
        );
        assert_eq!(target.export_snapshot().unwrap(), before);
    }

    #[test]
    fn late_corrupt_feed_and_pre_cdc_schema_leave_populated_restore_unchanged() {
        let source = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
        source
            .session()
            .execute("INSERT (:Incoming {value: 1})")
            .unwrap();
        let bytes = source.export_snapshot().unwrap();
        let target = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
        target
            .session()
            .execute("INSERT (:Original {value: 2})")
            .unwrap();
        let before = target.export_snapshot().unwrap();
        let snapshot = decode_snapshot_bytes(&bytes).unwrap();
        let old_schema =
            bincode::serde::encode_to_vec(&snapshot, bincode::config::standard()).unwrap();
        assert!(target.restore_snapshot(&old_schema).is_err());
        assert_eq!(target.export_snapshot().unwrap(), before);
        for mutation in 0..4 {
            let mut forged = snapshot.clone();
            match mutation {
                0 => forged.cdc_checkpoint[9] ^= 1,  // foreign StoreId
                1 => forged.cdc_checkpoint[41] ^= 1, // foreign publication cut
                2 => forged.cdc_checkpoint[57 + 16..57 + 24].copy_from_slice(&1u64.to_le_bytes()), // regressive sequence
                _ => {
                    forged.cdc_checkpoint.push(0); // late body corruption under a valid outer length
                    let length = (forged.cdc_checkpoint.len() - 57) as u64;
                    forged.cdc_checkpoint[49..57].copy_from_slice(&length.to_le_bytes());
                }
            }
            assert!(
                target
                    .restore_snapshot(&encode_snapshot_bytes(&forged).unwrap())
                    .is_err()
            );
            assert_eq!(
                target.export_snapshot().unwrap(),
                before,
                "mutation {mutation}"
            );
        }
    }
}

#[cfg(all(test, feature = "triple-store"))]
mod rdf_portable_feature_tests {
    use super::{decode_snapshot_bytes, encode_snapshot_bytes};
    use crate::{Config, GrafeoDB, GraphModel};

    #[test]
    fn rdf_only_snapshot_refuses_unowned_auxiliary_images_even_with_index_codecs() {
        let source =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
        let bytes = source.export_snapshot().unwrap();
        let target =
            GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
        let before = target.export_snapshot().unwrap();
        for family in ["text", "vector", "projection"] {
            let mut snapshot = decode_snapshot_bytes(&bytes).unwrap();
            match family {
                "text" => snapshot.text_indexes = vec![1],
                "vector" => snapshot.vector_indexes = vec![1],
                _ => snapshot.rdf_lpg_projections = vec![1],
            }
            let malformed = encode_snapshot_bytes(&snapshot).unwrap();
            if let Some(directory) = std::env::var_os("GRAFEO_RDF_PORTABLE_BAD_IMAGES") {
                use std::io::Write;
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(std::path::Path::new(&directory).join(format!("{family}.bin")))
                    .unwrap()
                    .write_all(&malformed)
                    .unwrap();
            }
            assert!(GrafeoDB::import_snapshot(&malformed).is_err(), "{family}");
            assert!(target.restore_snapshot(&malformed).is_err(), "{family}");
            assert_eq!(target.export_snapshot().unwrap(), before);
        }
    }
}
