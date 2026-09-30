//! Durable RDF→LPG projection definitions and rebuild status.
//!
//! A projection is an explicit mapping, never an implicit RDF/LPG identity.
//! The complete mapping is identified by a 256-bit digest. The historical
//! 64-bit id remains only as an API compatibility shorthand; the registry
//! always compares the full digest before accepting an id. Materialized LPG
//! rows carry the complete digest below so rebuilds can reconcile their owned
//! generation without touching user-authored rows.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::types::{
    Digest256, EpochId, GraphIncarnationId, MAX_WORLD_GRAPH_NAME_BYTES, ProjectionCut,
    ProjectionReconciliationState, ProjectionSourceGraph, StoreId,
};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockWriteGuard};
use serde::{Deserialize, Serialize};

/// Reserved LPG property identifying the projection that owns a materialized
/// row. User-facing projection code must never infer RDF/LPG identity without
/// this explicit marker.
pub const RDF_LPG_PROJECTION_OWNER_PROPERTY: &str = "__grafeo_rdf_projection";

/// LPG property carrying the source RDF IRI for a materialized row.
pub const RDF_LPG_PROJECTION_IRI_PROPERTY: &str = "iri";

/// Current canonical RDF→LPG mapping grammar.
pub const RDF_LPG_PROJECTION_MAPPING_VERSION: u16 = 2;

/// Current canonical RDF→LPG publication-receipt grammar.
pub const RDF_LPG_PROJECTION_RECEIPT_VERSION: u16 = 3;

const MAPPING_DIGEST_DOMAIN: &[u8] = b"org.grafeo.rdf-lpg.projection-mapping.v2\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"org.grafeo.rdf-lpg.projection-receipt.v3\0";
const RECEIPT_MAGIC: &[u8; 4] = b"GRPR";
const PERSISTENCE_MAGIC: &[u8; 4] = b"GRPS";
const PROJECTION_PERSISTENCE_VERSION: u16 = 3;
const MAX_MAPPING_IRI_BYTES: usize = 64 * 1024;
const MAX_MAPPING_LABEL_BYTES: usize = 16 * 1024;
const MAX_RECEIPT_BYTES: usize = MAX_WORLD_GRAPH_NAME_BYTES + 256;
const MAX_PROJECTION_PERSISTENCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_PROJECTION_DEFINITIONS: usize = 1_000_000;

/// A declared RDF→LPG mapping and the last successfully published generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfLpgProjectionDefinition {
    id: u64,
    mapping_digest: Digest256,
    mapping_format_version: u16,
    source_graph: Option<String>,
    type_iri: String,
    node_label: String,
    generation: u64,
    last_source_epoch: Option<EpochId>,
    last_target_epoch: Option<EpochId>,
    row_count: u64,
    reconciliation: ProjectionReconciliationState,
    receipt: Option<RdfLpgProjectionReceipt>,
}

impl RdfLpgProjectionDefinition {
    /// Builds a newly declared projection.
    #[must_use]
    pub fn new(type_iri: impl Into<String>, node_label: impl Into<String>) -> Self {
        let type_iri = type_iri.into();
        let node_label = node_label.into();
        Self::from_mapping_parts(None, type_iri, node_label)
    }

    /// Builds a validated mapping over the default graph or one logical named
    /// graph. The graph incarnation is captured by each publication receipt,
    /// not by the declaration, so dropping and recreating the same graph name
    /// produces a new receipt without changing the mapping identity.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or oversized graph/mapping fields.
    pub fn new_for_graph(
        source_graph: Option<&str>,
        type_iri: impl Into<String>,
        node_label: impl Into<String>,
    ) -> Result<Self, String> {
        let type_iri = type_iri.into();
        let node_label = node_label.into();
        RdfLpgProjectionRegistry::validate_mapping_for_graph(source_graph, &type_iri, &node_label)?;
        Ok(Self::from_mapping_parts(
            source_graph.map(str::to_owned),
            type_iri,
            node_label,
        ))
    }

    fn from_mapping_parts(
        source_graph: Option<String>,
        type_iri: String,
        node_label: String,
    ) -> Self {
        let mapping_digest =
            Self::full_mapping_digest(source_graph.as_deref(), &type_iri, &node_label);
        // Preserve the historical default-graph id exactly. Named-graph
        // mappings did not exist in that format and use the digest prefix.
        let id = source_graph.as_deref().map_or_else(
            || Self::content_id(&type_iri, &node_label),
            |_| compatibility_id(mapping_digest),
        );
        Self {
            id,
            mapping_digest,
            mapping_format_version: RDF_LPG_PROJECTION_MAPPING_VERSION,
            source_graph,
            type_iri,
            node_label,
            generation: 0,
            last_source_epoch: None,
            last_target_epoch: None,
            row_count: 0,
            reconciliation: ProjectionReconciliationState::Pending,
            receipt: None,
        }
    }

    /// Stable, domain-separated content hash of the complete mapping.
    #[must_use]
    pub fn content_id(type_iri: &str, node_label: &str) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"grafeo:rdf-lpg-projection:v1\0");
        hasher.update(&(type_iri.len() as u64).to_le_bytes());
        hasher.update(type_iri.as_bytes());
        hasher.update(&(node_label.len() as u64).to_le_bytes());
        hasher.update(node_label.as_bytes());
        let digest = hasher.finalize();
        let mut prefix = [0_u8; 8];
        prefix.copy_from_slice(&digest.as_bytes()[..8]);
        u64::from_le_bytes(prefix)
    }

    /// Complete digest of the versioned logical mapping.
    #[must_use]
    pub fn full_mapping_digest(
        source_graph: Option<&str>,
        type_iri: &str,
        node_label: &str,
    ) -> Digest256 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(MAPPING_DIGEST_DOMAIN);
        hasher.update(&RDF_LPG_PROJECTION_MAPPING_VERSION.to_le_bytes());
        hash_optional_string(&mut hasher, source_graph);
        hash_string(&mut hasher, type_iri);
        hash_string(&mut hasher, node_label);
        Digest256::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Public projection id.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Complete, collision-resistant identity of this mapping.
    #[must_use]
    pub const fn mapping_digest(&self) -> Digest256 {
        self.mapping_digest
    }

    /// Canonical mapping grammar version.
    #[must_use]
    pub const fn mapping_format_version(&self) -> u16 {
        self.mapping_format_version
    }

    /// Logical RDF source graph (`None` is the default graph).
    #[must_use]
    pub fn source_graph(&self) -> Option<&str> {
        self.source_graph.as_deref()
    }

    /// RDF class IRI selected by this mapping.
    #[must_use]
    pub fn type_iri(&self) -> &str {
        &self.type_iri
    }

    /// LPG label assigned to materialized nodes.
    #[must_use]
    pub fn node_label(&self) -> &str {
        &self.node_label
    }

    /// Monotonic count of successfully published rebuild generations.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// RDF source epoch captured by the last successful rebuild.
    #[must_use]
    pub const fn last_source_epoch(&self) -> Option<EpochId> {
        self.last_source_epoch
    }

    /// LPG transaction commit epoch of the last successful generation.
    ///
    /// Published definitions always carry the target commit epoch.
    #[must_use]
    pub const fn last_target_epoch(&self) -> Option<EpochId> {
        self.last_target_epoch
    }

    /// Number of source rows in the last successful generation.
    #[must_use]
    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Durable confidence in the published generation.
    #[must_use]
    pub const fn reconciliation(&self) -> ProjectionReconciliationState {
        self.reconciliation
    }

    /// Complete verified receipt for the current generation, if one exists.
    #[must_use]
    pub const fn receipt(&self) -> Option<&RdfLpgProjectionReceipt> {
        self.receipt.as_ref()
    }

    /// Canonical string written into the LPG ownership marker.
    #[must_use]
    pub fn owner_marker(&self) -> String {
        self.mapping_digest.to_string()
    }

    /// Whether an LPG owner marker belongs to this mapping.
    #[must_use]
    pub fn owner_marker_matches(&self, marker: &str) -> bool {
        marker == self.owner_marker()
    }

    /// Preserves only this logical mapping for installation into a new store.
    ///
    /// Publication receipts bind the source store id and exact source
    /// and target coordinates. A logical fork must therefore discard every
    /// publication field and rebuild in the destination.
    #[must_use]
    pub fn unpublished_for_fork(&self) -> Self {
        Self {
            id: self.id,
            mapping_digest: self.mapping_digest,
            mapping_format_version: self.mapping_format_version,
            source_graph: self.source_graph.clone(),
            type_iri: self.type_iri.clone(),
            node_label: self.node_label.clone(),
            generation: 0,
            last_source_epoch: None,
            last_target_epoch: None,
            row_count: 0,
            reconciliation: ProjectionReconciliationState::Pending,
            receipt: None,
        }
    }
}

/// Durable, self-verifying publication receipt for one RDF→LPG generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfLpgProjectionReceipt {
    format_version: u16,
    store_id: StoreId,
    mapping_digest: Digest256,
    projection_id: u64,
    source_graph: ProjectionSourceGraph,
    source_epoch: EpochId,
    target_epoch: EpochId,
    generation: u64,
    row_count: u64,
    reconciliation: ProjectionReconciliationState,
    receipt_digest: Digest256,
}

impl RdfLpgProjectionReceipt {
    /// Constructs and integrity-seals one V3 publication receipt.
    ///
    /// # Errors
    ///
    /// Rejects pending/impossible epochs, generation zero, or reconciliation
    /// states that cannot truthfully describe a new V3 publication.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store_id: StoreId,
        mapping_digest: Digest256,
        projection_id: u64,
        source_graph: ProjectionSourceGraph,
        source_epoch: EpochId,
        target_epoch: EpochId,
        generation: u64,
        row_count: u64,
        reconciliation: ProjectionReconciliationState,
    ) -> Result<Self, String> {
        let mut receipt = Self {
            format_version: RDF_LPG_PROJECTION_RECEIPT_VERSION,
            store_id,
            mapping_digest,
            projection_id,
            source_graph,
            source_epoch,
            target_epoch,
            generation,
            row_count,
            reconciliation,
            receipt_digest: Digest256::from_bytes([0; 32]),
        };
        receipt.validate_fields()?;
        receipt.receipt_digest = receipt.compute_digest();
        Ok(receipt)
    }

    /// Receipt wire-format version.
    #[must_use]
    pub const fn format_version(&self) -> u16 {
        self.format_version
    }

    /// Logical store bound by this receipt.
    #[must_use]
    pub const fn store_id(&self) -> StoreId {
        self.store_id
    }

    /// Full mapping digest.
    #[must_use]
    pub const fn mapping_digest(&self) -> Digest256 {
        self.mapping_digest
    }

    /// Compatibility projection id.
    #[must_use]
    pub const fn projection_id(&self) -> u64 {
        self.projection_id
    }

    /// Exact source graph lifetime.
    #[must_use]
    pub const fn source_graph(&self) -> &ProjectionSourceGraph {
        &self.source_graph
    }

    /// RDF source cut materialized by this generation.
    #[must_use]
    pub const fn source_epoch(&self) -> EpochId {
        self.source_epoch
    }

    /// LPG transaction epoch that atomically owns rows and receipt.
    #[must_use]
    pub const fn target_epoch(&self) -> EpochId {
        self.target_epoch
    }

    /// Monotonic generation number.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of materialized rows.
    #[must_use]
    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Durable reconciliation state.
    #[must_use]
    pub const fn reconciliation(&self) -> ProjectionReconciliationState {
        self.reconciliation
    }

    /// Full canonical receipt digest.
    #[must_use]
    pub const fn receipt_digest(&self) -> Digest256 {
        self.receipt_digest
    }

    /// Verifies receipt integrity and routing coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error for tampering, a foreign store, or a different mapping.
    pub fn verify_for(
        &self,
        store_id: StoreId,
        mapping_digest: Digest256,
        projection_id: u64,
    ) -> Result<(), String> {
        self.validate_fields()?;
        if self.receipt_digest != self.compute_digest() {
            return Err("RDF→LPG projection receipt digest does not match its fields".into());
        }
        if self.store_id != store_id {
            return Err(format!(
                "RDF→LPG projection receipt belongs to store {}, not {}",
                self.store_id, store_id
            ));
        }
        if self.mapping_digest != mapping_digest || self.projection_id != projection_id {
            return Err("RDF→LPG projection receipt does not match its mapping identity".into());
        }
        Ok(())
    }

    /// Converts a verified receipt into portable WorldCut projection metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the receipt fails its own integrity checks or the
    /// WorldCut layer rejects its coordinates.
    pub fn to_world_cut(&self) -> Result<ProjectionCut, String> {
        self.verify_for(self.store_id, self.mapping_digest, self.projection_id)?;
        ProjectionCut::from_verified_receipt(
            self.store_id,
            self.mapping_digest,
            self.format_version,
            self.generation,
            self.source_graph.clone(),
            self.source_epoch,
            Some(self.target_epoch),
            self.row_count,
            self.reconciliation,
            self.receipt_digest,
        )
        .map_err(|error| error.to_string())
    }

    /// Exact bounded wire encoding carried by the transaction-owned WAL V3
    /// record and portable projection metadata.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192 + self.source_graph.name().map_or(0, str::len));
        bytes.extend_from_slice(RECEIPT_MAGIC);
        bytes.extend_from_slice(&self.format_version.to_le_bytes());
        bytes.extend_from_slice(self.store_id.as_bytes());
        bytes.extend_from_slice(self.mapping_digest.as_bytes());
        bytes.extend_from_slice(&self.projection_id.to_le_bytes());
        encode_source_graph(&mut bytes, &self.source_graph);
        bytes.extend_from_slice(&self.source_epoch.as_u64().to_le_bytes());
        bytes.extend_from_slice(&self.target_epoch.as_u64().to_le_bytes());
        bytes.extend_from_slice(&self.generation.to_le_bytes());
        bytes.extend_from_slice(&self.row_count.to_le_bytes());
        bytes.push(reconciliation_tag(self.reconciliation));
        bytes.extend_from_slice(self.receipt_digest.as_bytes());
        bytes
    }

    /// Decodes one exact, bounded V3 receipt and verifies its digest.
    ///
    /// # Errors
    ///
    /// Rejects oversized, truncated, trailing, unknown-version/state,
    /// malformed graph, invalid epoch, and tampered payloads without panicking.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_RECEIPT_BYTES {
            return Err(format!(
                "RDF→LPG projection receipt has {} bytes; maximum is {MAX_RECEIPT_BYTES}",
                bytes.len()
            ));
        }
        let mut cursor = ReceiptCursor::new(bytes);
        if cursor.take(4)? != RECEIPT_MAGIC {
            return Err("RDF→LPG projection receipt has invalid magic".into());
        }
        let format_version = cursor.u16()?;
        if format_version != RDF_LPG_PROJECTION_RECEIPT_VERSION {
            return Err(format!(
                "unsupported RDF→LPG projection receipt version {format_version}"
            ));
        }
        let store_id = StoreId::from_bytes(cursor.array::<32>()?)
            .map_err(|error| format!("invalid RDF→LPG projection receipt store: {error}"))?;
        let mapping_digest = Digest256::from_bytes(cursor.array::<32>()?);
        let projection_id = cursor.u64()?;
        let source_graph = decode_source_graph(&mut cursor)?;
        let source_epoch = EpochId::new(cursor.u64()?);
        let target_epoch = EpochId::new(cursor.u64()?);
        let generation = cursor.u64()?;
        let row_count = cursor.u64()?;
        let reconciliation = decode_reconciliation(cursor.u8()?)?;
        let receipt_digest = Digest256::from_bytes(cursor.array::<32>()?);
        if !cursor.is_empty() {
            return Err(format!(
                "RDF→LPG projection receipt contains {} trailing bytes",
                cursor.remaining()
            ));
        }
        let receipt = Self {
            format_version,
            store_id,
            mapping_digest,
            projection_id,
            source_graph,
            source_epoch,
            target_epoch,
            generation,
            row_count,
            reconciliation,
            receipt_digest,
        };
        receipt.verify_for(store_id, mapping_digest, projection_id)?;
        Ok(receipt)
    }

    fn validate_fields(&self) -> Result<(), String> {
        if self.format_version != RDF_LPG_PROJECTION_RECEIPT_VERSION {
            return Err(format!(
                "unsupported RDF→LPG projection receipt version {}",
                self.format_version
            ));
        }
        validate_source_graph(&self.source_graph)?;
        if self.source_epoch == EpochId::PENDING || self.target_epoch == EpochId::PENDING {
            return Err("RDF→LPG projection receipt cannot contain a pending epoch".into());
        }
        if self.target_epoch.as_u64() == 0 {
            return Err("RDF→LPG projection receipt target epoch must be non-zero".into());
        }
        if self.source_epoch > self.target_epoch {
            return Err(format!(
                "RDF→LPG projection receipt source epoch {} follows target epoch {}",
                self.source_epoch.as_u64(),
                self.target_epoch.as_u64()
            ));
        }
        if self.generation == 0 {
            return Err("RDF→LPG projection receipt generation must be non-zero".into());
        }
        if !matches!(
            self.reconciliation,
            ProjectionReconciliationState::Reconciled
                | ProjectionReconciliationState::NeedsReconciliation
        ) {
            return Err(
                "new RDF→LPG projection receipt must be reconciled or need reconciliation".into(),
            );
        }
        Ok(())
    }

    fn compute_digest(&self) -> Digest256 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(&self.format_version.to_le_bytes());
        hasher.update(self.store_id.as_bytes());
        hasher.update(self.mapping_digest.as_bytes());
        hasher.update(&self.projection_id.to_le_bytes());
        hash_source_graph(&mut hasher, &self.source_graph);
        hasher.update(&self.source_epoch.as_u64().to_le_bytes());
        hasher.update(&self.target_epoch.as_u64().to_le_bytes());
        hasher.update(&self.generation.to_le_bytes());
        hasher.update(&self.row_count.to_le_bytes());
        hasher.update(&[reconciliation_tag(self.reconciliation)]);
        Digest256::from_bytes(*hasher.finalize().as_bytes())
    }
}

fn compatibility_id(digest: Digest256) -> u64 {
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(prefix)
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_optional_string(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(value) => {
            hasher.update(&[1]);
            hash_string(hasher, value);
        }
    }
}

fn hash_source_graph(hasher: &mut blake3::Hasher, graph: &ProjectionSourceGraph) {
    hash_optional_string(hasher, graph.name());
    hasher.update(&graph.incarnation().as_u64().to_le_bytes());
}

fn encode_source_graph(bytes: &mut Vec<u8>, graph: &ProjectionSourceGraph) {
    match graph.name() {
        None => bytes.push(0),
        Some(name) => {
            bytes.push(1);
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
        }
    }
    bytes.extend_from_slice(&graph.incarnation().as_u64().to_le_bytes());
}

fn decode_source_graph(cursor: &mut ReceiptCursor<'_>) -> Result<ProjectionSourceGraph, String> {
    let name = match cursor.u8()? {
        0 => None,
        1 => {
            let length = cursor.length(MAX_WORLD_GRAPH_NAME_BYTES, "source graph name")?;
            let bytes = cursor.take(length)?;
            Some(
                std::str::from_utf8(bytes)
                    .map_err(|_| "RDF→LPG projection source graph is not UTF-8".to_string())?
                    .to_owned(),
            )
        }
        tag => {
            return Err(format!(
                "RDF→LPG projection receipt has unknown source graph tag {tag}"
            ));
        }
    };
    let incarnation = GraphIncarnationId::new(cursor.u64()?);
    match name {
        None if incarnation.is_default_graph() => Ok(ProjectionSourceGraph::default_graph()),
        Some(name) => ProjectionSourceGraph::named(name, incarnation).map_err(|e| e.to_string()),
        _ => Err("RDF→LPG projection source graph name and incarnation disagree".into()),
    }
}

fn validate_source_graph(graph: &ProjectionSourceGraph) -> Result<(), String> {
    match graph.name() {
        None if graph.incarnation().is_default_graph() => Ok(()),
        Some(name)
            if !name.is_empty()
                && name.len() <= MAX_WORLD_GRAPH_NAME_BYTES
                && !name.chars().any(char::is_control)
                && !graph.incarnation().is_default_graph() =>
        {
            Ok(())
        }
        Some(name) if name.len() > MAX_WORLD_GRAPH_NAME_BYTES => Err(format!(
            "RDF→LPG projection source graph has {} bytes; maximum is {MAX_WORLD_GRAPH_NAME_BYTES}",
            name.len()
        )),
        Some(name) if name.chars().any(char::is_control) => {
            Err("RDF→LPG projection source graph contains a control character".into())
        }
        _ => Err("RDF→LPG projection source graph name and incarnation disagree".into()),
    }
}

fn reconciliation_tag(state: ProjectionReconciliationState) -> u8 {
    match state {
        ProjectionReconciliationState::Pending => 0,
        ProjectionReconciliationState::Reconciled => 1,
        ProjectionReconciliationState::NeedsReconciliation => 2,
    }
}

fn decode_reconciliation(tag: u8) -> Result<ProjectionReconciliationState, String> {
    match tag {
        1 => Ok(ProjectionReconciliationState::Reconciled),
        2 => Ok(ProjectionReconciliationState::NeedsReconciliation),
        0 | 3 => Err(format!(
            "RDF→LPG projection V3 receipt cannot use reconciliation state {tag}"
        )),
        _ => Err(format!(
            "RDF→LPG projection receipt has unknown reconciliation state {tag}"
        )),
    }
}

struct ReceiptCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ReceiptCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| "RDF→LPG projection receipt length overflow".to_string())?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| "truncated RDF→LPG projection receipt".to_string())?;
        self.position = end;
        Ok(value)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        self.take(N)?
            .try_into()
            .map_err(|_| "truncated RDF→LPG projection receipt".to_string())
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn length(&mut self, maximum: usize, field: &str) -> Result<usize, String> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| format!("RDF→LPG projection {field} length does not fit usize"))?;
        if length > maximum {
            return Err(format!(
                "RDF→LPG projection {field} has {length} bytes; maximum is {maximum}"
            ));
        }
        Ok(length)
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }
}

fn encode_persisted_definition(
    bytes: &mut Vec<u8>,
    definition: &RdfLpgProjectionDefinition,
) -> Result<(), String> {
    bytes.extend_from_slice(&definition.id.to_le_bytes());
    bytes.extend_from_slice(definition.mapping_digest.as_bytes());
    bytes.extend_from_slice(&definition.mapping_format_version.to_le_bytes());
    encode_optional_bounded_string(
        bytes,
        definition.source_graph.as_deref(),
        MAX_WORLD_GRAPH_NAME_BYTES,
        "source graph",
    )?;
    encode_bounded_string(
        bytes,
        &definition.type_iri,
        MAX_MAPPING_IRI_BYTES,
        "type IRI",
    )?;
    encode_bounded_string(
        bytes,
        &definition.node_label,
        MAX_MAPPING_LABEL_BYTES,
        "node label",
    )?;
    bytes.extend_from_slice(&definition.generation.to_le_bytes());
    encode_optional_epoch(bytes, definition.last_source_epoch);
    encode_optional_epoch(bytes, definition.last_target_epoch);
    bytes.extend_from_slice(&definition.row_count.to_le_bytes());
    bytes.push(reconciliation_tag(definition.reconciliation));
    let receipt = definition
        .receipt
        .as_ref()
        .map_or_else(Vec::new, RdfLpgProjectionReceipt::encode);
    let receipt_len = u32::try_from(receipt.len())
        .map_err(|_| "RDF→LPG projection receipt exceeds u32".to_string())?;
    bytes.extend_from_slice(&receipt_len.to_le_bytes());
    bytes.extend_from_slice(&receipt);
    Ok(())
}

fn encode_optional_bounded_string(
    bytes: &mut Vec<u8>,
    value: Option<&str>,
    maximum: usize,
    field: &str,
) -> Result<(), String> {
    match value {
        None => bytes.push(0),
        Some(value) => {
            bytes.push(1);
            encode_bounded_string(bytes, value, maximum, field)?;
        }
    }
    Ok(())
}

fn encode_bounded_string(
    bytes: &mut Vec<u8>,
    value: &str,
    maximum: usize,
    field: &str,
) -> Result<(), String> {
    if value.len() > maximum {
        return Err(format!(
            "RDF→LPG projection {field} has {} bytes; maximum is {maximum}",
            value.len()
        ));
    }
    let length = u32::try_from(value.len())
        .map_err(|_| format!("RDF→LPG projection {field} exceeds u32"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn encode_optional_epoch(bytes: &mut Vec<u8>, epoch: Option<EpochId>) {
    match epoch {
        None => bytes.push(0),
        Some(epoch) => {
            bytes.push(1);
            bytes.extend_from_slice(&epoch.as_u64().to_le_bytes());
        }
    }
}

fn decode_persistence_v3(bytes: &[u8]) -> Result<Vec<RdfLpgProjectionDefinition>, String> {
    if !bytes.starts_with(PERSISTENCE_MAGIC) {
        return Err("invalid RDF→LPG projection persistence magic".into());
    }
    let mut cursor = ReceiptCursor::new(bytes);
    let _magic = cursor.take(4)?;
    let version = cursor.u16()?;
    if version != PROJECTION_PERSISTENCE_VERSION {
        return Err(format!(
            "unsupported RDF→LPG projection persistence version {version}"
        ));
    }
    let count = usize::try_from(cursor.u32()?)
        .map_err(|_| "RDF→LPG projection count does not fit usize".to_string())?;
    if count > MAX_PROJECTION_DEFINITIONS {
        return Err(format!(
            "RDF→LPG projection registry has {count} definitions; maximum is {MAX_PROJECTION_DEFINITIONS}"
        ));
    }
    let mut definitions = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let id = cursor.u64()?;
        let mapping_digest = Digest256::from_bytes(cursor.array::<32>()?);
        let mapping_format_version = cursor.u16()?;
        let source_graph = decode_optional_persisted_string(
            &mut cursor,
            MAX_WORLD_GRAPH_NAME_BYTES,
            "source graph",
        )?;
        let type_iri = decode_persisted_string(&mut cursor, MAX_MAPPING_IRI_BYTES, "type IRI")?;
        let node_label =
            decode_persisted_string(&mut cursor, MAX_MAPPING_LABEL_BYTES, "node label")?;
        let generation = cursor.u64()?;
        let last_source_epoch = decode_optional_epoch(&mut cursor, "source epoch")?;
        let last_target_epoch = decode_optional_epoch(&mut cursor, "target epoch")?;
        let row_count = cursor.u64()?;
        let reconciliation = decode_persisted_reconciliation(cursor.u8()?)?;
        let receipt_length = usize::try_from(cursor.u32()?)
            .map_err(|_| "RDF→LPG receipt length does not fit usize".to_string())?;
        if receipt_length > MAX_RECEIPT_BYTES {
            return Err(format!(
                "RDF→LPG projection receipt has {receipt_length} bytes; maximum is {MAX_RECEIPT_BYTES}"
            ));
        }
        let receipt = if receipt_length == 0 {
            None
        } else {
            Some(RdfLpgProjectionReceipt::decode(
                cursor.take(receipt_length)?,
            )?)
        };
        definitions.push(RdfLpgProjectionDefinition {
            id,
            mapping_digest,
            mapping_format_version,
            source_graph,
            type_iri,
            node_label,
            generation,
            last_source_epoch,
            last_target_epoch,
            row_count,
            reconciliation,
            receipt,
        });
    }
    if !cursor.is_empty() {
        return Err(format!(
            "RDF→LPG projection metadata contains {} trailing bytes",
            cursor.remaining()
        ));
    }
    Ok(definitions)
}

fn decode_optional_persisted_string(
    cursor: &mut ReceiptCursor<'_>,
    maximum: usize,
    field: &str,
) -> Result<Option<String>, String> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => decode_persisted_string(cursor, maximum, field).map(Some),
        tag => Err(format!(
            "RDF→LPG projection {field} has unknown option tag {tag}"
        )),
    }
}

fn decode_persisted_string(
    cursor: &mut ReceiptCursor<'_>,
    maximum: usize,
    field: &str,
) -> Result<String, String> {
    let length = usize::try_from(cursor.u32()?)
        .map_err(|_| format!("RDF→LPG projection {field} length does not fit usize"))?;
    if length > maximum {
        return Err(format!(
            "RDF→LPG projection {field} has {length} bytes; maximum is {maximum}"
        ));
    }
    std::str::from_utf8(cursor.take(length)?)
        .map(str::to_owned)
        .map_err(|_| format!("RDF→LPG projection {field} is not UTF-8"))
}

fn decode_optional_epoch(
    cursor: &mut ReceiptCursor<'_>,
    field: &str,
) -> Result<Option<EpochId>, String> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => Ok(Some(EpochId::new(cursor.u64()?))),
        tag => Err(format!(
            "RDF→LPG projection {field} has unknown option tag {tag}"
        )),
    }
}

fn decode_persisted_reconciliation(tag: u8) -> Result<ProjectionReconciliationState, String> {
    match tag {
        0 => Ok(ProjectionReconciliationState::Pending),
        1 => Ok(ProjectionReconciliationState::Reconciled),
        2 => Ok(ProjectionReconciliationState::NeedsReconciliation),
        _ => Err(format!(
            "RDF→LPG projection metadata has unknown reconciliation state {tag}"
        )),
    }
}

/// Shared registry for declared projections and rebuild serialization.
#[derive(Debug, Default)]
pub struct RdfLpgProjectionRegistry {
    definitions: RwLock<FxHashMap<u64, RdfLpgProjectionDefinition>>,
    rebuild_lock: Mutex<()>,
    dirty: AtomicBool,
}

/// Fully validated receipt publication prepared before a durable commit marker.
///
/// Fields are private and the token owns the exact registry it was validated
/// against. Publishing performs only an in-memory post-image replacement.
#[must_use = "a prepared projection receipt must be published after its durable commit"]
pub struct PreparedRdfLpgProjectionReceipt {
    registry: Arc<RdfLpgProjectionRegistry>,
    previous_generation: u64,
    receipt: RdfLpgProjectionReceipt,
}

/// Fully validated registry replacement prepared before durable publication.
///
/// The token owns the complete post-image and the registry state against which
/// it was prepared. Publication performs no parsing, allocation, or fallible
/// validation after the durable boundary.
#[must_use = "a prepared projection registry must be published after its durable records"]
pub struct PreparedRdfLpgProjectionRegistryRestore {
    registry: Arc<RdfLpgProjectionRegistry>,
    previous: FxHashMap<u64, RdfLpgProjectionDefinition>,
    restored: FxHashMap<u64, RdfLpgProjectionDefinition>,
    installed: bool,
}

/// Validated registry replacement retaining its writer before publication.
#[must_use = "install only after all companion replacements are ready"]
pub struct ReadyRdfLpgProjectionRegistryRestore<'ready> {
    definitions: RwLockWriteGuard<'ready, FxHashMap<u64, RdfLpgProjectionDefinition>>,
    restored: &'ready mut FxHashMap<u64, RdfLpgProjectionDefinition>,
    installed: &'ready mut bool,
    dirty: &'ready AtomicBool,
    dirty_value: bool,
}

/// Installed registry retaining its writer while companion state is published.
#[must_use = "release after all companion replacements are installed"]
pub struct InstalledRdfLpgProjectionRegistryRestore<'ready> {
    ready: ReadyRdfLpgProjectionRegistryRestore<'ready>,
}

impl PreparedRdfLpgProjectionReceipt {
    /// Receipt encoded into the owning transaction's WAL record.
    #[must_use]
    pub const fn receipt(&self) -> &RdfLpgProjectionReceipt {
        &self.receipt
    }

    /// Publishes the already-validated registry post-image.
    ///
    /// This is intentionally not fallible: the caller writes the durable
    /// commit marker before invoking it. Rebuild serialization must remain
    /// held from preparation through publication, so no registry generation
    /// can change between those points.
    ///
    /// # Panics
    ///
    /// Panics fail-stop if the engine violates that serialization invariant;
    /// reopening replays the committed receipt and rows from WAL.
    pub fn publish(self) {
        let mut definitions = self.registry.definitions.write();
        let definition = definitions
            .get_mut(&self.receipt.projection_id())
            .expect("prepared RDF→LPG receipt lost its declaration");
        assert_eq!(
            definition.generation, self.previous_generation,
            "prepared RDF→LPG receipt generation changed before publication"
        );
        apply_receipt(definition, self.receipt);
        self.registry.dirty.store(true, Ordering::Release);
    }
}

impl PreparedRdfLpgProjectionRegistryRestore {
    /// Validates the live preimage and retains its writer before publication.
    ///
    /// # Errors
    ///
    /// Rejects concurrent registry changes, a busy writer or repeated install.
    pub fn ready(&mut self) -> Result<ReadyRdfLpgProjectionRegistryRestore<'_>, String> {
        if self.installed {
            return Err("RDF→LPG registry replacement has already been installed".into());
        }
        let definitions =
            self.registry.definitions.try_write().ok_or_else(|| {
                "RDF→LPG registry is busy during replacement preparation".to_string()
            })?;
        if *definitions != self.previous {
            return Err("prepared RDF→LPG registry changed before publication".into());
        }
        Ok(ReadyRdfLpgProjectionRegistryRestore {
            definitions,
            restored: &mut self.restored,
            installed: &mut self.installed,
            dirty: &self.registry.dirty,
            dirty_value: true,
        })
    }

    /// Publishes the replacement through the same guarded install path.
    ///
    /// Aggregate callers must obtain [`Self::ready`] before their durable or
    /// companion publication boundary and retain its installed fence instead.
    ///
    /// # Errors
    ///
    /// Returns the prepublication errors from [`Self::ready`].
    pub fn publish(mut self) -> Result<(), String> {
        self.ready()?.install().release();
        Ok(())
    }
}

impl<'ready> ReadyRdfLpgProjectionRegistryRestore<'ready> {
    /// Swaps the complete registry without allocation, locking or validation.
    /// The outer workspace retains displaced definitions until it is dropped.
    pub fn install(mut self) -> InstalledRdfLpgProjectionRegistryRestore<'ready> {
        std::mem::swap(&mut *self.definitions, self.restored);
        *self.installed = true;
        self.dirty.store(self.dirty_value, Ordering::Release);
        InstalledRdfLpgProjectionRegistryRestore { ready: self }
    }
}

impl InstalledRdfLpgProjectionRegistryRestore<'_> {
    /// Releases the registry writer, retaining displaced data in its workspace.
    pub fn release(self) {
        drop(self.ready);
    }
}

impl RdfLpgProjectionRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validates the user-controlled fields of a projection mapping.
    ///
    /// RDF IRIs cannot contain raw whitespace/control characters or the ASCII
    /// delimiters forbidden by RFC 3987. LPG labels may contain ordinary
    /// whitespace when quoted by a query language, but an empty/whitespace-only
    /// or control-bearing label is not a usable projection target.
    ///
    /// # Errors
    ///
    /// Returns an error when either mapping field is empty or contains
    /// characters forbidden by the corresponding RDF/LPG representation.
    pub fn validate_mapping(type_iri: &str, node_label: &str) -> Result<(), String> {
        Self::validate_mapping_for_graph(None, type_iri, node_label)
    }

    /// Validates a complete logical mapping, including its source graph.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, oversized, or syntactically unsafe fields.
    pub fn validate_mapping_for_graph(
        source_graph: Option<&str>,
        type_iri: &str,
        node_label: &str,
    ) -> Result<(), String> {
        if let Some(source_graph) = source_graph {
            if source_graph.is_empty() {
                return Err("RDF→LPG projection source graph must not be empty".into());
            }
            if source_graph.len() > MAX_WORLD_GRAPH_NAME_BYTES {
                return Err(format!(
                    "RDF→LPG projection source graph has {} bytes; maximum is {MAX_WORLD_GRAPH_NAME_BYTES}",
                    source_graph.len()
                ));
            }
            if source_graph.chars().any(char::is_control) {
                return Err("RDF→LPG projection source graph contains a control character".into());
            }
        }
        if type_iri.is_empty() {
            return Err("RDF→LPG projection type IRI must not be empty".into());
        }
        if type_iri.len() > MAX_MAPPING_IRI_BYTES {
            return Err(format!(
                "RDF→LPG projection type IRI has {} bytes; maximum is {MAX_MAPPING_IRI_BYTES}",
                type_iri.len()
            ));
        }
        if type_iri
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
            || type_iri.chars().any(|character| {
                matches!(
                    character,
                    '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`'
                )
            })
        {
            return Err(format!(
                "RDF→LPG projection type IRI {type_iri:?} contains forbidden characters"
            ));
        }
        if node_label.trim().is_empty() {
            return Err("RDF→LPG projection node label must not be empty".into());
        }
        if node_label.len() > MAX_MAPPING_LABEL_BYTES {
            return Err(format!(
                "RDF→LPG projection node label has {} bytes; maximum is {MAX_MAPPING_LABEL_BYTES}",
                node_label.len()
            ));
        }
        if node_label.chars().any(char::is_control) {
            return Err(format!(
                "RDF→LPG projection node label {node_label:?} contains control characters"
            ));
        }
        Ok(())
    }

    /// Declares a mapping idempotently and returns its content-derived id.
    ///
    /// A collision between distinct mappings is fail-stop: the public API is
    /// constrained to a 64-bit id, so silently aliasing two definitions would
    /// be materially worse than refusing to continue.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed mapping fields or a content-id collision.
    pub fn declare(&self, type_iri: &str, node_label: &str) -> Result<u64, String> {
        self.declare_for_graph(None, type_iri, node_label)
    }

    /// Declares a mapping over the default graph or one logical named graph.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed fields or an id/full-digest collision.
    pub fn declare_for_graph(
        &self,
        source_graph: Option<&str>,
        type_iri: &str,
        node_label: &str,
    ) -> Result<u64, String> {
        let definition =
            RdfLpgProjectionDefinition::new_for_graph(source_graph, type_iri, node_label)?;
        let id = definition.id();
        self.install_definition_v3(
            id,
            definition.mapping_digest(),
            definition.mapping_format_version(),
            source_graph,
            type_iri,
            node_label,
        )?;
        Ok(id)
    }

    /// Installs one complete V3 declaration idempotently.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed fields, a mismatched digest/version/id,
    /// or a collision with an existing shorthand id.
    #[allow(clippy::too_many_arguments)]
    pub fn install_definition_v3(
        &self,
        id: u64,
        mapping_digest: Digest256,
        mapping_format_version: u16,
        source_graph: Option<&str>,
        type_iri: &str,
        node_label: &str,
    ) -> Result<bool, String> {
        Self::validate_mapping_for_graph(source_graph, type_iri, node_label)?;
        if mapping_format_version != RDF_LPG_PROJECTION_MAPPING_VERSION {
            return Err(format!(
                "unsupported RDF→LPG projection mapping version {mapping_format_version}"
            ));
        }
        let definition =
            RdfLpgProjectionDefinition::new_for_graph(source_graph, type_iri, node_label)?;
        if definition.mapping_digest() != mapping_digest {
            return Err(format!(
                "RDF→LPG projection id {id} does not match its full mapping digest"
            ));
        }
        if definition.id() != id {
            return Err(format!(
                "RDF→LPG projection id {id} does not match its mapping content hash"
            ));
        }
        let mut definitions = self.definitions.write();
        if let Some(existing) = definitions
            .values()
            .find(|existing| existing.mapping_digest() == mapping_digest && existing.id() != id)
        {
            return Err(format!(
                "RDF→LPG projection mapping digest {} is already installed as id {}, not {id}",
                mapping_digest,
                existing.id()
            ));
        }
        match definitions.entry(id) {
            hashbrown::hash_map::Entry::Occupied(existing) => {
                if existing.get().mapping_digest() != mapping_digest
                    || existing.get().mapping_format_version() != mapping_format_version
                    || existing.get().source_graph() != source_graph
                    || existing.get().type_iri() != type_iri
                    || existing.get().node_label() != node_label
                {
                    return Err(format!(
                        "RDF→LPG projection shorthand id {id} collides across full mapping definitions"
                    ));
                }
                Ok(false)
            }
            hashbrown::hash_map::Entry::Vacant(entry) => {
                entry.insert(definition);
                self.dirty.store(true, Ordering::Release);
                Ok(true)
            }
        }
    }

    /// Returns one declared projection.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<RdfLpgProjectionDefinition> {
        self.definitions.read().get(&id).cloned()
    }

    /// Whether any projection has been declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.definitions.read().is_empty()
    }

    /// Serializes rebuilds so two generations cannot both reconcile from the
    /// same target cut and create duplicate owned rows.
    pub fn lock_rebuild(&self) -> MutexGuard<'_, ()> {
        self.rebuild_lock.lock()
    }

    /// Installs one verified V3 receipt idempotently.
    ///
    /// Older generations are ignored when a checkpoint already contains a
    /// newer one. An exact same-generation replay is a no-op; any differing
    /// receipt for that generation is corruption. Advancing generations must
    /// move both source and target coordinates monotonically.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign/tampered receipt, an undeclared mapping,
    /// a logical-source mismatch, or conflicting/non-monotonic replay.
    pub fn prepare_receipt(
        self: &Arc<Self>,
        expected_store_id: StoreId,
        receipt: RdfLpgProjectionReceipt,
    ) -> Result<PreparedRdfLpgProjectionReceipt, String> {
        let definitions = self.definitions.read();
        let definition = definitions.get(&receipt.projection_id()).ok_or_else(|| {
            format!(
                "prepared receipt for undeclared RDF→LPG projection {}",
                receipt.projection_id()
            )
        })?;
        validate_receipt_for_definition(definition, expected_store_id, &receipt)?;
        let expected_generation = definition.generation.checked_add(1).ok_or_else(|| {
            format!(
                "RDF→LPG projection {} exhausted its generation counter",
                definition.id()
            )
        })?;
        if receipt.generation() != expected_generation {
            return Err(format!(
                "RDF→LPG projection {} must publish generation {expected_generation}, not {}",
                definition.id(),
                receipt.generation()
            ));
        }
        validate_receipt_monotonicity(definition, &receipt)?;
        Ok(PreparedRdfLpgProjectionReceipt {
            registry: Arc::clone(self),
            previous_generation: definition.generation,
            receipt,
        })
    }

    /// Installs one verified V3 receipt idempotently during WAL recovery.
    ///
    /// Runtime commits use [`prepare_receipt`](Self::prepare_receipt) and the
    /// resulting infallible publication token. This method additionally allows
    /// a forward generation jump because a checkpoint may already have retired
    /// intermediate receipts.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign/tampered receipt, an undeclared mapping,
    /// a logical-source mismatch, or conflicting/non-monotonic replay.
    pub fn install_receipt(
        &self,
        expected_store_id: StoreId,
        receipt: RdfLpgProjectionReceipt,
    ) -> Result<bool, String> {
        let mut definitions = self.definitions.write();
        let definition = definitions
            .get_mut(&receipt.projection_id())
            .ok_or_else(|| {
                format!(
                    "published receipt for undeclared RDF→LPG projection {}",
                    receipt.projection_id()
                )
            })?;
        validate_receipt_for_definition(definition, expected_store_id, &receipt)?;
        if receipt.generation() < definition.generation {
            return Ok(false);
        }
        if receipt.generation() == definition.generation {
            return match definition.receipt.as_ref() {
                Some(known) if known == &receipt => Ok(false),
                _ => Err(format!(
                    "conflicting RDF→LPG projection receipt for id {}, generation {}",
                    definition.id(),
                    receipt.generation()
                )),
            };
        }
        validate_receipt_monotonicity(definition, &receipt)?;
        apply_receipt(definition, receipt);
        self.dirty.store(true, Ordering::Release);
        Ok(true)
    }

    /// Deterministic registry snapshot for persistence.
    #[must_use]
    pub fn snapshot(&self) -> Vec<RdfLpgProjectionDefinition> {
        let mut definitions: Vec<_> = self.definitions.read().values().cloned().collect();
        definitions
            .sort_unstable_by_key(|definition| (definition.mapping_digest(), definition.id()));
        definitions
    }

    /// Encodes the complete registry in the canonical, bounded V3 persistence
    /// grammar. This format is shared by the RDF container addendum and engine
    /// snapshots; it does not rely on the in-memory struct's serde layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the registry exceeds the durable payload bound.
    pub fn encode_persistence_v3(&self) -> Result<Vec<u8>, String> {
        let definitions = self.snapshot();
        if definitions.len() > MAX_PROJECTION_DEFINITIONS {
            return Err(format!(
                "RDF→LPG projection registry has {} definitions; maximum is {MAX_PROJECTION_DEFINITIONS}",
                definitions.len()
            ));
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PERSISTENCE_MAGIC);
        bytes.extend_from_slice(&PROJECTION_PERSISTENCE_VERSION.to_le_bytes());
        bytes.extend_from_slice(
            &u32::try_from(definitions.len())
                .map_err(|_| "RDF→LPG projection definition count exceeds u32".to_string())?
                .to_le_bytes(),
        );
        for definition in definitions {
            encode_persisted_definition(&mut bytes, &definition)?;
            if bytes.len() > MAX_PROJECTION_PERSISTENCE_BYTES {
                return Err(format!(
                    "RDF→LPG projection metadata has {} bytes; maximum is {MAX_PROJECTION_PERSISTENCE_BYTES}",
                    bytes.len()
                ));
            }
        }
        Ok(bytes)
    }

    /// Decodes current canonical V3 metadata.
    ///
    /// Current receipts are verified against `expected_store_id`.
    ///
    /// # Errors
    ///
    /// Rejects oversized, malformed, non-canonical, foreign-store, colliding,
    /// or trailing metadata without publishing any registry state.
    pub fn decode_persistence(
        expected_store_id: StoreId,
        bytes: &[u8],
    ) -> Result<Vec<RdfLpgProjectionDefinition>, String> {
        if bytes.len() > MAX_PROJECTION_PERSISTENCE_BYTES {
            return Err(format!(
                "RDF→LPG projection metadata has {} bytes; maximum is {MAX_PROJECTION_PERSISTENCE_BYTES}",
                bytes.len()
            ));
        }
        let definitions = decode_persistence_v3(bytes)?;
        let validation = Self::new();
        validation.restore_for_store(expected_store_id, definitions.clone())?;
        Ok(definitions)
    }

    /// Deterministic mapping-only snapshot safe to install in a logical fork.
    ///
    /// Store-bound generations and receipts are deliberately reset to
    /// unpublished `Pending`; the destination must rebuild them.
    #[must_use]
    pub fn snapshot_mappings_for_fork(&self) -> Vec<RdfLpgProjectionDefinition> {
        self.snapshot()
            .into_iter()
            .map(|definition| definition.unpublished_for_fork())
            .collect()
    }

    /// Restores only logical mappings from a foreign/fork source.
    ///
    /// # Errors
    ///
    /// Returns an error if any mapping identity or field is invalid. Durable
    /// generation evidence is never imported.
    pub fn restore_mappings_for_fork(
        &self,
        definitions: Vec<RdfLpgProjectionDefinition>,
    ) -> Result<(), String> {
        self.restore(
            definitions
                .into_iter()
                .map(|definition| definition.unpublished_for_fork())
                .collect(),
        )
    }

    /// Replaces the registry from a validated persistence snapshot.
    ///
    /// Duplicate ids and content/id mismatches are rejected rather than
    /// allowing an ambiguous mapping to reach a rebuild.
    ///
    /// # Errors
    ///
    /// Returns an error if any definition, published status, content id, or
    /// registry identity is invalid or duplicated.
    pub fn restore(&self, definitions: Vec<RdfLpgProjectionDefinition>) -> Result<(), String> {
        self.restore_inner(None, definitions)
    }

    /// Replaces the registry and proves every V3 receipt belongs to the
    /// expected logical store.
    ///
    /// # Errors
    ///
    /// Returns an error for all [`restore`](Self::restore) failures or a
    /// receipt from a foreign store.
    pub fn restore_for_store(
        &self,
        expected_store_id: StoreId,
        definitions: Vec<RdfLpgProjectionDefinition>,
    ) -> Result<(), String> {
        self.restore_inner(Some(expected_store_id), definitions)
    }

    /// Validates and materializes a complete StoreId-bound registry post-image
    /// for infallible publication after a durable boundary.
    ///
    /// # Errors
    ///
    /// Returns every error from [`restore_for_store`](Self::restore_for_store)
    /// without changing this registry.
    pub fn prepare_restore_for_store(
        self: &Arc<Self>,
        expected_store_id: StoreId,
        definitions: Vec<RdfLpgProjectionDefinition>,
    ) -> Result<PreparedRdfLpgProjectionRegistryRestore, String> {
        let staged = Self::new();
        staged.restore_for_store(expected_store_id, definitions)?;
        let restored = std::mem::take(&mut *staged.definitions.write());
        let previous = self.definitions.read().clone();
        Ok(PreparedRdfLpgProjectionRegistryRestore {
            registry: Arc::clone(self),
            previous,
            restored,
            installed: false,
        })
    }

    fn restore_inner(
        &self,
        expected_store_id: Option<StoreId>,
        definitions: Vec<RdfLpgProjectionDefinition>,
    ) -> Result<(), String> {
        let mut restored = FxHashMap::default();
        let mut digests = std::collections::BTreeMap::new();
        for definition in definitions {
            Self::validate_mapping_for_graph(
                definition.source_graph(),
                definition.type_iri(),
                definition.node_label(),
            )?;
            if definition.mapping_format_version != RDF_LPG_PROJECTION_MAPPING_VERSION {
                return Err(format!(
                    "unsupported RDF→LPG projection mapping version {}",
                    definition.mapping_format_version
                ));
            }
            let expected = RdfLpgProjectionDefinition::new_for_graph(
                definition.source_graph(),
                definition.type_iri(),
                definition.node_label(),
            )?;
            if definition.mapping_digest != expected.mapping_digest() {
                return Err(format!(
                    "RDF→LPG projection id {} does not match its full mapping digest",
                    definition.id()
                ));
            }
            if definition.id() != expected.id() {
                return Err(format!(
                    "RDF→LPG projection id {} does not match its mapping content hash",
                    definition.id()
                ));
            }
            if definition.generation == 0 {
                if definition.last_source_epoch.is_some()
                    || definition.last_target_epoch.is_some()
                    || definition.row_count != 0
                    || definition.reconciliation != ProjectionReconciliationState::Pending
                    || definition.receipt.is_some()
                {
                    return Err(format!(
                        "RDF→LPG projection {} has unpublished generation 0 with published status fields",
                        definition.id()
                    ));
                }
            } else {
                let source_epoch = definition.last_source_epoch.ok_or_else(|| {
                    format!(
                        "RDF→LPG projection {}, generation {} has no source epoch",
                        definition.id(),
                        definition.generation
                    )
                })?;
                if source_epoch == EpochId::PENDING {
                    return Err(format!(
                        "RDF→LPG projection {}, generation {} has pending source epoch",
                        definition.id(),
                        definition.generation
                    ));
                }
                let target_epoch = definition.last_target_epoch.ok_or_else(|| {
                    format!(
                        "RDF→LPG projection {}, generation {} has no target epoch",
                        definition.id(),
                        definition.generation
                    )
                })?;
                if target_epoch == EpochId::PENDING || target_epoch.as_u64() == 0 {
                    return Err(format!(
                        "RDF→LPG projection {}, generation {} has invalid target epoch 0",
                        definition.id(),
                        definition.generation
                    ));
                }
                if source_epoch > target_epoch {
                    return Err(format!(
                        "RDF→LPG projection {}, generation {} has source epoch {} after target epoch {}",
                        definition.id(),
                        definition.generation,
                        source_epoch.as_u64(),
                        target_epoch.as_u64()
                    ));
                }
                if definition.reconciliation == ProjectionReconciliationState::Pending {
                    return Err(format!(
                        "RDF→LPG projection {}, generation {} remains pending",
                        definition.id(),
                        definition.generation
                    ));
                }
                let receipt = definition.receipt.as_ref().ok_or_else(|| {
                    format!(
                        "RDF→LPG projection {}, generation {} has no V3 receipt",
                        definition.id(),
                        definition.generation
                    )
                })?;
                let receipt_store = expected_store_id.unwrap_or(receipt.store_id());
                receipt.verify_for(receipt_store, definition.mapping_digest(), definition.id())?;
                if receipt.source_graph().name() != definition.source_graph()
                    || receipt.source_epoch() != source_epoch
                    || receipt.target_epoch() != target_epoch
                    || receipt.generation() != definition.generation
                    || receipt.row_count() != definition.row_count
                    || receipt.reconciliation() != definition.reconciliation
                {
                    return Err(format!(
                        "RDF→LPG projection {} status disagrees with its V3 receipt",
                        definition.id()
                    ));
                }
            }
            let id = definition.id();
            if let Some(other_id) = digests.insert(definition.mapping_digest(), id)
                && other_id != id
            {
                return Err(format!(
                    "duplicate RDF→LPG projection mapping digest {} under ids {other_id} and {id}",
                    definition.mapping_digest()
                ));
            }
            if restored.insert(id, definition).is_some() {
                return Err(format!("duplicate RDF→LPG projection id {id}"));
            }
        }
        let mut installed = false;
        ReadyRdfLpgProjectionRegistryRestore {
            definitions: self.definitions.write(),
            restored: &mut restored,
            installed: &mut installed,
            dirty: &self.dirty,
            dirty_value: false,
        }
        .install()
        .release();
        Ok(())
    }

    /// Whether registry metadata changed since its last successful flush.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Marks persisted registry metadata clean.
    pub fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }
}

fn validate_receipt_for_definition(
    definition: &RdfLpgProjectionDefinition,
    expected_store_id: StoreId,
    receipt: &RdfLpgProjectionReceipt,
) -> Result<(), String> {
    receipt.verify_for(
        expected_store_id,
        definition.mapping_digest(),
        definition.id(),
    )?;
    if definition.source_graph() != receipt.source_graph().name() {
        return Err(format!(
            "RDF→LPG projection {} receipt source graph does not match its declaration",
            definition.id()
        ));
    }
    Ok(())
}

fn validate_receipt_monotonicity(
    definition: &RdfLpgProjectionDefinition,
    receipt: &RdfLpgProjectionReceipt,
) -> Result<(), String> {
    if let Some(previous_source) = definition.last_source_epoch
        && receipt.source_epoch() < previous_source
    {
        return Err(format!(
            "RDF→LPG projection {}, generation {} regresses source epoch from {} to {}",
            definition.id(),
            receipt.generation(),
            previous_source.as_u64(),
            receipt.source_epoch().as_u64()
        ));
    }
    if let Some(previous_target) = definition.last_target_epoch
        && receipt.target_epoch() <= previous_target
    {
        return Err(format!(
            "RDF→LPG projection {}, generation {} does not advance target epoch beyond {}",
            definition.id(),
            receipt.generation(),
            previous_target.as_u64()
        ));
    }
    Ok(())
}

fn apply_receipt(definition: &mut RdfLpgProjectionDefinition, receipt: RdfLpgProjectionReceipt) {
    definition.generation = receipt.generation();
    definition.last_source_epoch = Some(receipt.source_epoch());
    definition.last_target_epoch = Some(receipt.target_epoch());
    definition.row_count = receipt.row_count();
    definition.reconciliation = receipt.reconciliation();
    definition.receipt = Some(receipt);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_id(byte: u8) -> grafeo_common::types::StoreId {
        grafeo_common::types::StoreId::from_bytes([byte; 32]).unwrap()
    }

    #[test]
    fn v3_mapping_digest_binds_every_logical_mapping_coordinate() {
        let default = RdfLpgProjectionDefinition::new("http://example.org/Person", "Person");
        let named = RdfLpgProjectionDefinition::new_for_graph(
            Some("http://example.org/claims"),
            "http://example.org/Person",
            "Person",
        )
        .unwrap();
        let other_type =
            RdfLpgProjectionDefinition::new_for_graph(None, "http://example.org/Agent", "Person")
                .unwrap();
        let other_label =
            RdfLpgProjectionDefinition::new_for_graph(None, "http://example.org/Person", "Agent")
                .unwrap();

        assert_ne!(default.mapping_digest(), named.mapping_digest());
        assert_ne!(default.mapping_digest(), other_type.mapping_digest());
        assert_ne!(default.mapping_digest(), other_label.mapping_digest());
        assert_eq!(named.source_graph(), Some("http://example.org/claims"));
        assert_eq!(named.mapping_format_version(), 2);
        assert_eq!(named.owner_marker().len(), 64);
        assert!(named.owner_marker_matches(&named.owner_marker()));
        assert!(!named.owner_marker_matches(&format!("{:016x}", named.id())));
    }

    #[test]
    fn v3_receipt_wire_is_exact_bounded_and_tamper_evident() {
        let definition = RdfLpgProjectionDefinition::new_for_graph(
            Some("http://example.org/claims"),
            "http://example.org/Person",
            "Person",
        )
        .unwrap();
        let source_graph = grafeo_common::types::ProjectionSourceGraph::named(
            "http://example.org/claims",
            grafeo_common::types::GraphIncarnationId::new(7),
        )
        .unwrap();
        let receipt = RdfLpgProjectionReceipt::new(
            store_id(1),
            definition.mapping_digest(),
            definition.id(),
            source_graph,
            EpochId::new(11),
            EpochId::new(12),
            3,
            19,
            grafeo_common::types::ProjectionReconciliationState::Reconciled,
        )
        .unwrap();

        let encoded = receipt.encode();
        assert_eq!(RdfLpgProjectionReceipt::decode(&encoded).unwrap(), receipt);
        receipt
            .verify_for(store_id(1), definition.mapping_digest(), definition.id())
            .unwrap();
        assert!(
            receipt
                .verify_for(store_id(2), definition.mapping_digest(), definition.id())
                .is_err(),
            "a receipt from another logical store must not verify"
        );

        for index in 0..encoded.len() {
            let mut tampered = encoded.clone();
            tampered[index] ^= 0x01;
            assert!(
                RdfLpgProjectionReceipt::decode(&tampered).is_err(),
                "tampering byte {index} must be detected"
            );
        }

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(RdfLpgProjectionReceipt::decode(&trailing).is_err());

        let mut unknown_version = encoded.clone();
        unknown_version[4..6].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(RdfLpgProjectionReceipt::decode(&unknown_version).is_err());

        let graph_tag_offset = 4 + 2 + 32 + 32 + 8;
        let mut unknown_graph_tag = encoded.clone();
        unknown_graph_tag[graph_tag_offset] = u8::MAX;
        assert!(RdfLpgProjectionReceipt::decode(&unknown_graph_tag).is_err());

        let graph_length_offset = graph_tag_offset + 1;
        let mut overlong_graph = encoded.clone();
        overlong_graph[graph_length_offset..graph_length_offset + 8]
            .copy_from_slice(&((MAX_WORLD_GRAPH_NAME_BYTES as u64) + 1).to_le_bytes());
        assert!(RdfLpgProjectionReceipt::decode(&overlong_graph).is_err());

        let graph_name_offset = graph_length_offset + 8;
        let mut control_graph = encoded.clone();
        control_graph[graph_name_offset] = b'\n';
        assert!(RdfLpgProjectionReceipt::decode(&control_graph).is_err());

        let state_offset = graph_name_offset
            + receipt.source_graph().name().unwrap().len()
            + 8 // incarnation
            + 8 // source epoch
            + 8 // target epoch
            + 8 // generation
            + 8; // row count
        let mut unknown_state = encoded.clone();
        unknown_state[state_offset] = u8::MAX;
        assert!(RdfLpgProjectionReceipt::decode(&unknown_state).is_err());

        assert!(
            RdfLpgProjectionReceipt::decode(&vec![0; MAX_RECEIPT_BYTES + 1]).is_err(),
            "oversized hostile input must be rejected before allocation or field decoding"
        );
    }

    #[test]
    fn v3_receipt_rejects_impossible_publication_coordinates() {
        let definition = RdfLpgProjectionDefinition::new("http://example.org/Person", "Person");
        let make = |source, target, generation, state| {
            RdfLpgProjectionReceipt::new(
                store_id(1),
                definition.mapping_digest(),
                definition.id(),
                grafeo_common::types::ProjectionSourceGraph::default_graph(),
                source,
                target,
                generation,
                1,
                state,
            )
        };
        use grafeo_common::types::ProjectionReconciliationState as State;
        assert!(make(EpochId::PENDING, EpochId::new(2), 1, State::Reconciled).is_err());
        assert!(make(EpochId::new(1), EpochId::PENDING, 1, State::Reconciled).is_err());
        assert!(make(EpochId::new(2), EpochId::new(1), 1, State::Reconciled).is_err());
        assert!(make(EpochId::new(1), EpochId::new(2), 0, State::Reconciled).is_err());
        assert!(make(EpochId::new(1), EpochId::new(2), 1, State::Pending).is_err());
        assert!(decode_persisted_reconciliation(3).is_err());
    }

    #[test]
    fn prepared_v3_receipt_publishes_infallibly_and_replays_exactly() {
        let registry = Arc::new(RdfLpgProjectionRegistry::new());
        let id = registry
            .declare("http://example.org/Person", "Person")
            .unwrap();
        let definition = registry.get(id).unwrap();
        let receipt = RdfLpgProjectionReceipt::new(
            store_id(1),
            definition.mapping_digest(),
            id,
            grafeo_common::types::ProjectionSourceGraph::default_graph(),
            EpochId::new(2),
            EpochId::new(3),
            1,
            7,
            grafeo_common::types::ProjectionReconciliationState::Reconciled,
        )
        .unwrap();

        let prepared = registry
            .prepare_receipt(store_id(1), receipt.clone())
            .unwrap();
        assert_eq!(registry.get(id).unwrap().generation(), 0);
        prepared.publish();
        assert_eq!(registry.get(id).unwrap().receipt(), Some(&receipt));
        assert!(
            !registry
                .install_receipt(store_id(1), receipt.clone())
                .unwrap()
        );

        let conflicting = RdfLpgProjectionReceipt::new(
            store_id(1),
            definition.mapping_digest(),
            id,
            grafeo_common::types::ProjectionSourceGraph::default_graph(),
            EpochId::new(2),
            EpochId::new(3),
            1,
            8,
            grafeo_common::types::ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        assert!(registry.install_receipt(store_id(1), conflicting).is_err());
        assert!(registry.install_receipt(store_id(2), receipt).is_err());
    }

    #[test]
    fn prepared_registry_restore_keeps_forward_generation_hidden_until_publish() {
        let staged = RdfLpgProjectionRegistry::new();
        let id = staged
            .declare("http://example.org/Person", "Person")
            .unwrap();
        let definition = staged.get(id).unwrap();
        let receipt = RdfLpgProjectionReceipt::new(
            store_id(1),
            definition.mapping_digest(),
            id,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(7),
            EpochId::new(9),
            5,
            2,
            ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        staged
            .install_receipt(store_id(1), receipt.clone())
            .unwrap();

        let live = Arc::new(RdfLpgProjectionRegistry::new());
        let prepared = live
            .prepare_restore_for_store(store_id(1), staged.snapshot())
            .unwrap();
        assert!(live.get(id).is_none());
        prepared.publish().unwrap();
        assert_eq!(live.get(id).unwrap().receipt(), Some(&receipt));
    }

    #[test]
    fn prepared_registry_restore_rejects_changed_preimage_and_busy_writer() {
        let live = Arc::new(RdfLpgProjectionRegistry::new());
        let mut prepared = live
            .prepare_restore_for_store(store_id(1), Vec::new())
            .unwrap();
        let reader = live.definitions.read();
        assert!(prepared.ready().is_err());
        drop(reader);
        let added = live.declare("http://example.org/Added", "Added").unwrap();
        assert!(prepared.ready().is_err());
        assert!(live.get(added).is_some());
        assert!(live.definitions.try_write().is_some());
    }

    #[test]
    fn prepared_registry_restore_holds_writer_and_retires_maps_after_release() {
        let live = Arc::new(RdfLpgProjectionRegistry::new());
        let old = live.declare("http://example.org/Old", "Old").unwrap();
        let source = RdfLpgProjectionRegistry::new();
        let replacement = source.declare("http://example.org/New", "New").unwrap();
        let mut prepared = live
            .prepare_restore_for_store(store_id(1), source.snapshot())
            .unwrap();
        let ready = prepared.ready().unwrap();
        #[cfg(feature = "lpg")]
        crate::allocation_test::start();
        let installed = ready.install();
        #[cfg(feature = "lpg")]
        assert_eq!(
            crate::allocation_test::stop(),
            crate::allocation_test::Counts::default()
        );
        assert!(live.definitions.try_read().is_none());
        #[cfg(feature = "lpg")]
        crate::allocation_test::start();
        installed.release();
        #[cfg(feature = "lpg")]
        assert_eq!(
            crate::allocation_test::stop(),
            crate::allocation_test::Counts::default()
        );
        assert!(live.get(old).is_none());
        assert!(live.get(replacement).is_some());
        assert!(prepared.restored.contains_key(&old));
        assert!(prepared.ready().is_err());
    }

    #[test]
    fn logical_fork_preserves_mapping_but_discards_store_bound_evidence() {
        let source = Arc::new(RdfLpgProjectionRegistry::new());
        let id = source
            .declare_for_graph(
                Some("http://example.org/claims"),
                "http://example.org/Person",
                "Person",
            )
            .unwrap();
        let definition = source.get(id).unwrap();
        let receipt = RdfLpgProjectionReceipt::new(
            store_id(1),
            definition.mapping_digest(),
            id,
            ProjectionSourceGraph::named("http://example.org/claims", GraphIncarnationId::new(8))
                .unwrap(),
            EpochId::new(4),
            EpochId::new(5),
            1,
            2,
            ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        source.install_receipt(store_id(1), receipt).unwrap();

        let fork = RdfLpgProjectionRegistry::new();
        fork.restore_mappings_for_fork(source.snapshot()).unwrap();
        let forked = fork.get(id).unwrap();
        assert_eq!(forked.mapping_digest(), definition.mapping_digest());
        assert_eq!(forked.source_graph(), definition.source_graph());
        assert_eq!(forked.generation(), 0);
        assert_eq!(forked.last_source_epoch(), None);
        assert_eq!(forked.last_target_epoch(), None);
        assert_eq!(forked.row_count(), 0);
        assert_eq!(
            forked.reconciliation(),
            ProjectionReconciliationState::Pending
        );
        assert_eq!(forked.receipt(), None);
    }

    #[test]
    fn persistence_v3_is_exact_store_bound_and_predecessors_are_rejected() {
        let registry = Arc::new(RdfLpgProjectionRegistry::new());
        let id = registry
            .declare("http://example.org/Person", "Person")
            .unwrap();
        let definition = registry.get(id).unwrap();
        let receipt = RdfLpgProjectionReceipt::new(
            store_id(1),
            definition.mapping_digest(),
            id,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(7),
            EpochId::new(8),
            1,
            3,
            ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        registry.install_receipt(store_id(1), receipt).unwrap();

        let encoded = registry.encode_persistence_v3().unwrap();
        let decoded = RdfLpgProjectionRegistry::decode_persistence(store_id(1), &encoded).unwrap();
        assert_eq!(decoded, registry.snapshot());
        assert!(RdfLpgProjectionRegistry::decode_persistence(store_id(2), &encoded).is_err());

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(RdfLpgProjectionRegistry::decode_persistence(store_id(1), &trailing).is_err());
        let mut unknown_version = encoded.clone();
        unknown_version[4..6].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(
            RdfLpgProjectionRegistry::decode_persistence(store_id(1), &unknown_version).is_err()
        );

        let pending = RdfLpgProjectionRegistry::new();
        pending
            .declare("http://example.org/Pending", "Pending")
            .unwrap();
        let pending = pending.encode_persistence_v3().unwrap();
        let mut excessive_count = pending.clone();
        let excessive_definition_count = u32::try_from(MAX_PROJECTION_DEFINITIONS)
            .unwrap()
            .checked_add(1)
            .unwrap();
        excessive_count[6..10].copy_from_slice(&excessive_definition_count.to_le_bytes());
        assert!(
            RdfLpgProjectionRegistry::decode_persistence(store_id(1), &excessive_count).is_err()
        );
        let type_iri_length_offset = 4 + 2 + 4 + 8 + 32 + 2 + 1;
        let mut excessive_type_iri = pending.clone();
        let excessive_mapping_bytes = u32::try_from(MAX_MAPPING_IRI_BYTES)
            .unwrap()
            .checked_add(1)
            .unwrap();
        excessive_type_iri[type_iri_length_offset..type_iri_length_offset + 4]
            .copy_from_slice(&excessive_mapping_bytes.to_le_bytes());
        assert!(
            RdfLpgProjectionRegistry::decode_persistence(store_id(1), &excessive_type_iri).is_err()
        );
        let state_offset = type_iri_length_offset
            + 4
            + "http://example.org/Pending".len()
            + 4
            + "Pending".len()
            + 8
            + 1
            + 1
            + 8;
        let mut unknown_state = pending.clone();
        unknown_state[state_offset] = u8::MAX;
        assert!(RdfLpgProjectionRegistry::decode_persistence(store_id(1), &unknown_state).is_err());
        for end in 0..pending.len() {
            let result = std::panic::catch_unwind(|| {
                RdfLpgProjectionRegistry::decode_persistence(store_id(1), &pending[..end])
            });
            assert!(result.is_ok(), "decoder panicked on prefix length {end}");
            assert!(result.unwrap().is_err(), "truncated prefix {end} decoded");
        }

        // The standard bincode encoding of an empty predecessor vector is one
        // zero byte. It used to be accepted by the pre-v3 fallback.
        let predecessor_empty_registry = [0];
        let error =
            RdfLpgProjectionRegistry::decode_persistence(store_id(1), &predecessor_empty_registry)
                .expect_err("pre-v3 projection metadata must be rejected");
        assert!(error.contains("persistence magic"), "{error}");
    }

    #[test]
    fn declaration_is_content_addressed_and_idempotent() {
        let registry = RdfLpgProjectionRegistry::new();
        let first = registry
            .declare("http://example.org/Person", "Person")
            .unwrap();
        let second = registry
            .declare("http://example.org/Person", "Person")
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(registry.snapshot().len(), 1);
    }

    #[test]
    fn shorthand_collision_is_rejected_even_when_full_digests_are_valid() {
        let registry = RdfLpgProjectionRegistry::new();
        let first = RdfLpgProjectionDefinition::new("http://example.org/Person", "Person");
        let second = RdfLpgProjectionDefinition::new("http://example.org/Agent", "Agent");
        registry
            .install_definition_v3(
                first.id(),
                first.mapping_digest(),
                first.mapping_format_version(),
                first.source_graph(),
                first.type_iri(),
                first.node_label(),
            )
            .unwrap();
        assert!(
            registry
                .install_definition_v3(
                    first.id(),
                    second.mapping_digest(),
                    second.mapping_format_version(),
                    second.source_graph(),
                    second.type_iri(),
                    second.node_label(),
                )
                .is_err(),
            "a valid foreign full digest must never alias an occupied u64 shorthand"
        );
    }

    #[test]
    fn malformed_mapping_is_rejected_by_install_and_restore() {
        let registry = RdfLpgProjectionRegistry::new();
        let empty_iri = RdfLpgProjectionDefinition::new("", "Person");
        assert!(
            registry
                .install_definition_v3(
                    empty_iri.id(),
                    empty_iri.mapping_digest(),
                    empty_iri.mapping_format_version(),
                    empty_iri.source_graph(),
                    "",
                    "Person",
                )
                .is_err()
        );
        assert!(registry.restore(vec![empty_iri]).is_err());

        let empty_label = RdfLpgProjectionDefinition::new("http://example.org/Person", "  ");
        assert!(
            registry
                .install_definition_v3(
                    empty_label.id(),
                    empty_label.mapping_digest(),
                    empty_label.mapping_format_version(),
                    empty_label.source_graph(),
                    "http://example.org/Person",
                    "  ",
                )
                .is_err()
        );
        assert!(registry.restore(vec![empty_label]).is_err());
        assert!(registry.is_empty());

        let mut unpublished_with_status =
            RdfLpgProjectionDefinition::new("http://example.org/Person", "Person");
        unpublished_with_status.last_source_epoch = Some(EpochId::new(1));
        assert!(registry.restore(vec![unpublished_with_status]).is_err());

        let mut published_without_source =
            RdfLpgProjectionDefinition::new("http://example.org/Person", "Person");
        published_without_source.generation = 1;
        published_without_source.row_count = 1;
        assert!(registry.restore(vec![published_without_source]).is_err());

        let mut source_after_target =
            RdfLpgProjectionDefinition::new("http://example.org/Person", "Person");
        source_after_target.generation = 1;
        source_after_target.last_source_epoch = Some(EpochId::new(3));
        source_after_target.last_target_epoch = Some(EpochId::new(2));
        assert!(registry.restore(vec![source_after_target]).is_err());
    }
}
