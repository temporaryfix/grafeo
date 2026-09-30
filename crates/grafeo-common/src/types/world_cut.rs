//! Portable, cryptographically verifiable world-cut metadata.
//!
//! A [`WorldCut`] identifies one committed database state without relying
//! on process-local handles or serialization-map ordering. Its manifest digest
//! is BLAKE3 over an explicit, domain-separated byte grammar. A
//! [`SnapshotArtifact`] additionally binds that metadata to the exact bytes of
//! one portable snapshot.

use core::{fmt, mem::size_of};
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{EpochId, GraphIncarnationId, HistoryCompleteness, StoreId};

const SCHEMA_DIGEST_DOMAIN: &[u8] = b"org.grafeo.world.schema.v1\0";
const PROJECTION_MAPPING_DIGEST_DOMAIN: &[u8] = b"org.grafeo.world.projection-mapping.v1\0";
const SNAPSHOT_STATE_DIGEST_DOMAIN: &[u8] = b"org.grafeo.world.snapshot-bytes.v1\0";
const COMPONENT_STATE_DIGEST_DOMAIN: &[u8] = b"org.grafeo.world.authoritative-components.v1\0";
const WORLD_MANIFEST_DIGEST_DOMAIN: &[u8] = b"org.grafeo.world.manifest.v1\0";
const RECOVERY_IMAGE_DIGEST_DOMAIN: &[u8] = b"org.grafeo.world.recovery-image.v1\0";
const RECOVERY_IMAGE_COORDINATES_V1_DOMAIN: &[u8] = b"grafeo:recovery-coordinates:v1\0";
const RECOVERY_IMAGE_COORDINATES_V1_ENCODED_LEN: usize =
    RECOVERY_IMAGE_COORDINATES_V1_DOMAIN.len() + 1 + 4 * size_of::<u64>();
const WORLD_METADATA_SECTION_TYPE_ID: u32 = 6;

/// Maximum UTF-8 byte length of a graph name embedded in portable history or
/// world-cut metadata.
pub const MAX_WORLD_GRAPH_NAME_BYTES: usize = 64 * 1024;
/// Maximum number of projection descriptors accepted in one portable cut.
pub const MAX_WORLD_PROJECTIONS: usize = 65_536;
/// Defensive decode limit for one standalone world-cut value.
pub const MAX_WORLD_CUT_BYTES: usize = 64 * 1024 * 1024;
/// Defensive decode limit for compact store-identity metadata.
pub const MAX_WORLD_IDENTITY_METADATA_BYTES: usize = 4 * 1024;
/// Defensive decode limit for a container WorldMetadata section.
pub const MAX_WORLD_METADATA_BYTES: usize = MAX_WORLD_CUT_BYTES + 4 * 1024;
/// Maximum number of physical components accepted in one recovery-image digest.
///
/// The current `.grafeo` directory is substantially smaller; this larger
/// bound leaves room for future directory formats without allowing an
/// unbounded canonicalization allocation.
pub const MAX_RECOVERY_IMAGE_COMPONENTS: usize = 65_536;

/// A complete, non-truncated BLAKE3 digest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Digest256([u8; Self::LEN]);

impl Digest256 {
    /// Digest length in bytes.
    pub const LEN: usize = 32;

    /// Constructs a digest from all 256 bits.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; Self::LEN]) -> Self {
        Self(bytes)
    }

    /// Returns all 256 digest bits.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Self::LEN] {
        &self.0
    }

    /// Consumes the digest into all 256 bits.
    #[must_use]
    pub const fn into_bytes(self) -> [u8; Self::LEN] {
        self.0
    }

    /// Hashes a canonical schema payload using the schema-specific domain.
    #[must_use]
    pub fn schema(canonical_schema: &[u8]) -> Self {
        hash_one(SCHEMA_DIGEST_DOMAIN, canonical_schema)
    }

    /// Hashes a complete canonical projection mapping.
    #[must_use]
    pub fn projection_mapping(canonical_mapping: &[u8]) -> Self {
        hash_one(PROJECTION_MAPPING_DIGEST_DOMAIN, canonical_mapping)
    }
}

impl fmt::Debug for Digest256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest256({self})")
    }
}

impl fmt::Display for Digest256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Stable model tag used by portable manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum GraphModelTag {
    /// Labeled property graph only.
    Lpg = 0,
    /// RDF dataset only.
    Rdf = 1,
    /// Native LPG and RDF stores in one database.
    Both = 2,
}

impl GraphModelTag {
    /// Stable byte tag used by the canonical manifest grammar.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Parses the stable byte tag.
    ///
    /// # Errors
    ///
    /// Returns [`WorldCutError::InvalidGraphModel`] for an unknown tag.
    pub const fn from_u8(value: u8) -> Result<Self, WorldCutError> {
        match value {
            0 => Ok(Self::Lpg),
            1 => Ok(Self::Rdf),
            2 => Ok(Self::Both),
            _ => Err(WorldCutError::InvalidGraphModel(value)),
        }
    }

    const fn has_lpg(self) -> bool {
        matches!(self, Self::Lpg | Self::Both)
    }

    const fn has_rdf(self) -> bool {
        matches!(self, Self::Rdf | Self::Both)
    }
}

/// Authoritative representation whose exact version contributed to a cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AuthoritativeFormat {
    /// Schema, catalog, and logical index metadata.
    Catalog = 1,
    /// Mutable labeled-property-graph store.
    Lpg = 2,
    /// Native RDF dataset.
    Rdf = 3,
    /// Columnar compact LPG base.
    Compact = 4,
    /// Compact-base deletion overlay.
    OverlayDeletions = 5,
    /// Graph-qualified RDF temporal history.
    RdfHistory = 6,
    /// Portable whole-database snapshot envelope.
    PortableSnapshot = 7,
    /// Retained native change feed and sequence authority.
    Cdc = 8,
}

impl AuthoritativeFormat {
    /// Stable byte tag used by the canonical manifest grammar.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Exact version of one authoritative representation in a cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelFormatVersion {
    format: AuthoritativeFormat,
    version: u16,
}

impl ModelFormatVersion {
    /// Constructs a non-zero format version.
    ///
    /// # Errors
    ///
    /// Returns [`WorldCutError::InvalidFormatVersion`] for version zero.
    pub const fn new(format: AuthoritativeFormat, version: u16) -> Result<Self, WorldCutError> {
        if version == 0 {
            return Err(WorldCutError::InvalidFormatVersion { format });
        }
        Ok(Self { format, version })
    }

    /// Representation described by this entry.
    #[must_use]
    pub const fn format(self) -> AuthoritativeFormat {
        self.format
    }

    /// Exact representation version.
    #[must_use]
    pub const fn version(self) -> u16 {
        self.version
    }
}

/// Canonical schema representation present at a world cut.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaCut {
    format_version: u16,
    digest: Digest256,
}

impl SchemaCut {
    /// Constructs schema provenance from its format version and full digest.
    ///
    /// # Errors
    ///
    /// Returns [`WorldCutError::InvalidSchemaVersion`] for version zero.
    pub const fn new(format_version: u16, digest: Digest256) -> Result<Self, WorldCutError> {
        if format_version == 0 {
            return Err(WorldCutError::InvalidSchemaVersion);
        }
        Ok(Self {
            format_version,
            digest,
        })
    }

    /// Digests a deterministic canonical catalog post-image.
    ///
    /// The caller is responsible for using the catalog's canonical encoder,
    /// which must sort registry/map entries before producing `post_image`.
    ///
    /// # Errors
    ///
    /// Returns [`WorldCutError::InvalidSchemaVersion`] for version zero.
    pub fn from_canonical_post_image(
        format_version: u16,
        post_image: &[u8],
    ) -> Result<Self, WorldCutError> {
        Self::new(format_version, Digest256::schema(post_image))
    }

    /// Schema serialization version.
    #[must_use]
    pub const fn format_version(&self) -> u16 {
        self.format_version
    }

    /// Digest of the canonical schema post-image.
    #[must_use]
    pub const fn digest(&self) -> Digest256 {
        self.digest
    }

    const fn validate(&self) -> Result<(), WorldCutError> {
        if self.format_version == 0 {
            return Err(WorldCutError::InvalidSchemaVersion);
        }
        Ok(())
    }
}

/// Exact RDF graph lifetime used as a projection source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionSourceGraph {
    name: Option<String>,
    incarnation: GraphIncarnationId,
}

impl ProjectionSourceGraph {
    /// The permanent RDF default graph.
    #[must_use]
    pub const fn default_graph() -> Self {
        Self {
            name: None,
            incarnation: GraphIncarnationId::DEFAULT_GRAPH,
        }
    }

    /// Constructs one named-graph lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, oversized, control-bearing name or a
    /// reserved default-graph incarnation.
    pub fn named(
        name: impl Into<String>,
        incarnation: GraphIncarnationId,
    ) -> Result<Self, WorldCutError> {
        let source = Self {
            name: Some(name.into()),
            incarnation,
        };
        source.validate()?;
        Ok(source)
    }

    /// Named graph IRI, or `None` for the default graph.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Exact named-graph incarnation.
    #[must_use]
    pub const fn incarnation(&self) -> GraphIncarnationId {
        self.incarnation
    }

    fn validate(&self) -> Result<(), WorldCutError> {
        match &self.name {
            None if self.incarnation.is_default_graph() => Ok(()),
            Some(name)
                if !name.is_empty()
                    && name.len() <= MAX_WORLD_GRAPH_NAME_BYTES
                    && !name.chars().any(char::is_control)
                    && !self.incarnation.is_default_graph() =>
            {
                Ok(())
            }
            Some(name) if name.len() > MAX_WORLD_GRAPH_NAME_BYTES => {
                Err(WorldCutError::InvalidProjection(format!(
                    "projection source graph name has {} bytes; maximum is {MAX_WORLD_GRAPH_NAME_BYTES}",
                    name.len()
                )))
            }
            Some(name) if name.chars().any(char::is_control) => {
                Err(WorldCutError::InvalidProjection(
                    "projection source graph name contains a control character".into(),
                ))
            }
            _ => Err(WorldCutError::InvalidProjection(
                "projection source graph name and incarnation disagree".into(),
            )),
        }
    }
}

/// Durable reconciliation state of a projection generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ProjectionReconciliationState {
    /// Declaration has not published a generation.
    Pending = 0,
    /// Materialized rows and the receipt agree.
    Reconciled = 1,
    /// The generation is durable but requires reconciliation.
    NeedsReconciliation = 2,
}

impl ProjectionReconciliationState {
    const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Projection generation represented in a world cut.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionCut {
    mapping_digest: Digest256,
    format_version: u16,
    generation: u64,
    source_graph: ProjectionSourceGraph,
    source_epoch: Option<EpochId>,
    target_epoch: Option<EpochId>,
    row_count: u64,
    reconciliation: ProjectionReconciliationState,
    receipt_store_id: Option<StoreId>,
    receipt_digest: Option<Digest256>,
}

impl ProjectionCut {
    /// Constructs an unpublished projection declaration.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero format version or invalid source graph.
    pub fn unpublished(
        mapping_digest: Digest256,
        format_version: u16,
        source_graph: ProjectionSourceGraph,
    ) -> Result<Self, WorldCutError> {
        let cut = Self {
            mapping_digest,
            format_version,
            generation: 0,
            source_graph,
            source_epoch: None,
            target_epoch: None,
            row_count: 0,
            reconciliation: ProjectionReconciliationState::Pending,
            receipt_store_id: None,
            receipt_digest: None,
        };
        cut.validate(None, None)?;
        Ok(cut)
    }

    /// Constructs a cut from a projection receipt already verified by the
    /// projection subsystem's canonical receipt decoder.
    ///
    /// The receipt's logical store is retained explicitly so installing this
    /// cut into a foreign [`WorldCutDescriptor`] fails closed. This layer does
    /// not duplicate the receipt byte grammar: it binds the full
    /// receipt digest and every routing/status field in the world manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if publication provenance, epochs, receipt identity,
    /// or reconciliation state is incomplete or contradictory.
    #[allow(clippy::too_many_arguments)]
    pub fn from_verified_receipt(
        receipt_store_id: StoreId,
        mapping_digest: Digest256,
        format_version: u16,
        generation: u64,
        source_graph: ProjectionSourceGraph,
        source_epoch: EpochId,
        target_epoch: Option<EpochId>,
        row_count: u64,
        reconciliation: ProjectionReconciliationState,
        receipt_digest: Digest256,
    ) -> Result<Self, WorldCutError> {
        let cut = Self {
            mapping_digest,
            format_version,
            generation,
            source_graph,
            source_epoch: Some(source_epoch),
            target_epoch,
            row_count,
            reconciliation,
            receipt_store_id: Some(receipt_store_id),
            receipt_digest: Some(receipt_digest),
        };
        cut.validate(None, Some(receipt_store_id))?;
        Ok(cut)
    }

    /// Full digest of the complete mapping.
    #[must_use]
    pub const fn mapping_digest(&self) -> Digest256 {
        self.mapping_digest
    }

    /// Projection receipt/definition format version.
    #[must_use]
    pub const fn format_version(&self) -> u16 {
        self.format_version
    }

    /// Last successfully published generation, or zero if unpublished.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Exact RDF graph lifetime used as the source.
    #[must_use]
    pub const fn source_graph(&self) -> &ProjectionSourceGraph {
        &self.source_graph
    }

    /// Source RDF epoch, if published.
    #[must_use]
    pub const fn source_epoch(&self) -> Option<EpochId> {
        self.source_epoch
    }

    /// Target LPG commit epoch, if published.
    #[must_use]
    pub const fn target_epoch(&self) -> Option<EpochId> {
        self.target_epoch
    }

    /// Number of materialized rows in this generation.
    #[must_use]
    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Durable reconciliation state.
    #[must_use]
    pub const fn reconciliation(&self) -> ProjectionReconciliationState {
        self.reconciliation
    }

    /// Full digest of the projection receipt, absent while unpublished.
    #[must_use]
    pub const fn receipt_digest(&self) -> Option<Digest256> {
        self.receipt_digest
    }

    /// Logical store bound by the durable receipt, if one exists.
    #[must_use]
    pub const fn receipt_store_id(&self) -> Option<StoreId> {
        self.receipt_store_id
    }

    fn validate(
        &self,
        cut_epoch: Option<EpochId>,
        expected_store_id: Option<StoreId>,
    ) -> Result<(), WorldCutError> {
        self.validate_fields(cut_epoch)?;
        match (self.receipt_store_id, self.receipt_digest) {
            (None, None) if self.generation == 0 => Ok(()),
            (Some(receipt_store_id), Some(_)) if self.generation > 0 => {
                if let Some(expected) = expected_store_id
                    && receipt_store_id != expected
                {
                    return Err(WorldCutError::InvalidProjection(format!(
                        "projection receipt store {receipt_store_id} does not match world store {expected}"
                    )));
                }
                Ok(())
            }
            (None, _) | (_, None) if self.generation > 0 => Err(WorldCutError::InvalidProjection(
                "projection generation has no complete receipt identity/digest".into(),
            )),
            _ => Err(WorldCutError::InvalidProjection(
                "unpublished projection declaration cannot carry receipt metadata".into(),
            )),
        }
    }

    fn validate_fields(&self, cut_epoch: Option<EpochId>) -> Result<(), WorldCutError> {
        if self.format_version == 0 {
            return Err(WorldCutError::InvalidProjection(
                "projection format version must be non-zero".into(),
            ));
        }
        self.source_graph.validate()?;
        if self.generation == 0 {
            if self.source_epoch.is_some()
                || self.target_epoch.is_some()
                || self.row_count != 0
                || self.reconciliation != ProjectionReconciliationState::Pending
                || self.receipt_store_id.is_some()
                || self.receipt_digest.is_some()
            {
                return Err(WorldCutError::InvalidProjection(
                    "unpublished generation 0 cannot carry publication status".into(),
                ));
            }
            return Ok(());
        }
        if self.reconciliation == ProjectionReconciliationState::Pending {
            return Err(WorldCutError::InvalidProjection(
                "published projection generation cannot remain pending".into(),
            ));
        }
        let source = self.source_epoch.ok_or_else(|| {
            WorldCutError::InvalidProjection(
                "a published projection generation requires a source epoch".into(),
            )
        })?;
        if source == EpochId::PENDING {
            return Err(WorldCutError::InvalidProjection(
                "projection source epoch cannot be pending/uncommitted".into(),
            ));
        }
        let target = self.target_epoch.ok_or_else(|| {
            WorldCutError::InvalidProjection(
                "a published projection generation requires a target epoch".into(),
            )
        })?;
        if target == EpochId::PENDING {
            return Err(WorldCutError::InvalidProjection(
                "projection target epoch cannot be pending/uncommitted".into(),
            ));
        }
        if target.as_u64() == 0 {
            return Err(WorldCutError::InvalidProjection(
                "a published projection target epoch must be non-zero".into(),
            ));
        }
        if source > target {
            return Err(WorldCutError::InvalidProjection(format!(
                "projection source epoch {} follows target epoch {}",
                source.as_u64(),
                target.as_u64()
            )));
        }
        if let Some(epoch) = cut_epoch
            && (source > epoch || target > epoch)
        {
            return Err(WorldCutError::InvalidProjection(format!(
                "projection provenance lies after world-cut epoch {}",
                epoch.as_u64()
            )));
        }
        Ok(())
    }
}

/// Metadata needed to seal a world cut against a concrete state digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldCutDescriptor {
    store_id: StoreId,
    epoch: EpochId,
    graph_model: GraphModelTag,
    formats: Vec<ModelFormatVersion>,
    schema: SchemaCut,
    projections: Vec<ProjectionCut>,
    history: HistoryCompleteness,
}

/// Canonical logical identity embedded in snapshots, WAL metadata, and
/// container world metadata.
///
/// Version 1 deliberately contains only values that must survive every
/// restore. Artifact-specific epoch/model/format metadata belongs to
/// [`WorldCutDescriptor`], while this compact record gives all persistence
/// paths one identity wire contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldIdentityMetadataV1 {
    store_id: StoreId,
    history: HistoryCompleteness,
}

impl WorldIdentityMetadataV1 {
    /// Current identity metadata wire version.
    pub const FORMAT_VERSION: u8 = 1;

    /// Constructs validated logical identity metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if legacy provenance uses reserved source version zero.
    pub fn new(store_id: StoreId, history: HistoryCompleteness) -> Result<Self, WorldCutError> {
        let metadata = Self { store_id, history };
        metadata.validate()?;
        Ok(metadata)
    }

    /// Logical store identity preserved by restore.
    #[must_use]
    pub const fn store_id(&self) -> StoreId {
        self.store_id
    }

    /// Truthful RDF history boundary.
    #[must_use]
    pub const fn history(&self) -> HistoryCompleteness {
        self.history
    }

    /// Encodes the exact versioned identity wire representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is invalid or serialization fails.
    pub fn encode(&self) -> Result<Vec<u8>, WorldCutError> {
        self.validate()?;
        bincode::serde::encode_to_vec(
            self,
            bincode::config::standard().with_limit::<MAX_WORLD_IDENTITY_METADATA_BYTES>(),
        )
        .map_err(|error| WorldCutError::IdentityMetadataSerialization(error.to_string()))
    }

    /// Decodes and validates identity metadata with no trailing bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, malformed, unsupported, invalid, or
    /// trailing input.
    pub fn decode(bytes: &[u8]) -> Result<Self, WorldCutError> {
        if bytes.is_empty() {
            return Err(WorldCutError::EmptyIdentityMetadata);
        }
        if bytes.len() > MAX_WORLD_IDENTITY_METADATA_BYTES {
            return Err(WorldCutError::IdentityMetadataTooLarge {
                bytes: bytes.len(),
                maximum: MAX_WORLD_IDENTITY_METADATA_BYTES,
            });
        }
        let (metadata, consumed): (Self, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_WORLD_IDENTITY_METADATA_BYTES>(),
        )
        .map_err(|error| WorldCutError::IdentityMetadataSerialization(error.to_string()))?;
        if consumed != bytes.len() {
            return Err(WorldCutError::TrailingIdentityMetadata {
                trailing: bytes.len() - consumed,
            });
        }
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<(), WorldCutError> {
        if let HistoryCompleteness::LegacyCurrentState {
            observed_at,
            source_version,
        } = self.history
        {
            if source_version == 0 {
                return Err(WorldCutError::InvalidLegacySourceVersion);
            }
            if observed_at == EpochId::PENDING {
                return Err(WorldCutError::PendingLegacyBoundary);
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct WorldIdentityMetadataWire {
    version: u8,
    store_id: StoreId,
    history: HistoryCompleteness,
}

impl Serialize for WorldIdentityMetadataV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WorldIdentityMetadataWire {
            version: Self::FORMAT_VERSION,
            store_id: self.store_id,
            history: self.history,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WorldIdentityMetadataV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WorldIdentityMetadataWire::deserialize(deserializer)?;
        if wire.version != Self::FORMAT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported world identity metadata version {}",
                wire.version
            )));
        }
        Self::new(wire.store_id, wire.history).map_err(serde::de::Error::custom)
    }
}

impl WorldCutDescriptor {
    /// Constructs, canonicalizes, and validates cut metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if model formats, projection provenance, history
    /// boundaries, or the committed epoch are inconsistent.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store_id: StoreId,
        epoch: EpochId,
        graph_model: GraphModelTag,
        mut formats: Vec<ModelFormatVersion>,
        schema: SchemaCut,
        mut projections: Vec<ProjectionCut>,
        history: HistoryCompleteness,
    ) -> Result<Self, WorldCutError> {
        formats.sort_unstable_by_key(|entry| entry.format);
        projections.sort_unstable_by_key(|entry| entry.mapping_digest);
        let descriptor = Self {
            store_id,
            epoch,
            graph_model,
            formats,
            schema,
            projections,
            history,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Logical store identity.
    #[must_use]
    pub const fn store_id(&self) -> StoreId {
        self.store_id
    }

    /// Committed transaction-time cut.
    #[must_use]
    pub const fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Native graph model at the cut.
    #[must_use]
    pub const fn graph_model(&self) -> GraphModelTag {
        self.graph_model
    }

    /// Canonically ordered authoritative format versions.
    #[must_use]
    pub fn formats(&self) -> &[ModelFormatVersion] {
        &self.formats
    }

    /// Canonical schema provenance.
    #[must_use]
    pub const fn schema(&self) -> &SchemaCut {
        &self.schema
    }

    /// Canonically ordered projection generations.
    #[must_use]
    pub fn projections(&self) -> &[ProjectionCut] {
        &self.projections
    }

    /// RDF history provenance boundary.
    #[must_use]
    pub const fn history(&self) -> HistoryCompleteness {
        self.history
    }

    fn validate(&self) -> Result<(), WorldCutError> {
        if self.epoch == EpochId::PENDING {
            return Err(WorldCutError::PendingEpoch);
        }
        self.schema.validate()?;
        if !strictly_sorted_unique_by(&self.formats, |entry| entry.format) {
            return Err(WorldCutError::NonCanonicalFormats);
        }
        for entry in &self.formats {
            if entry.version == 0 {
                return Err(WorldCutError::InvalidFormatVersion {
                    format: entry.format,
                });
            }
        }
        self.validate_model_formats()?;
        if !self.projections.is_empty() && self.graph_model != GraphModelTag::Both {
            return Err(WorldCutError::ProjectionRequiresBothModels);
        }
        if !strictly_sorted_unique_by(&self.projections, |entry| entry.mapping_digest) {
            return Err(WorldCutError::NonCanonicalProjections);
        }
        if self.projections.len() > MAX_WORLD_PROJECTIONS {
            return Err(WorldCutError::TooManyProjections {
                count: self.projections.len(),
                maximum: MAX_WORLD_PROJECTIONS,
            });
        }
        for projection in &self.projections {
            projection.validate(Some(self.epoch), Some(self.store_id))?;
        }
        match self.history {
            HistoryCompleteness::Complete => {}
            HistoryCompleteness::LegacyCurrentState {
                observed_at,
                source_version,
            } => {
                if source_version == 0 {
                    return Err(WorldCutError::InvalidLegacySourceVersion);
                }
                if observed_at == EpochId::PENDING {
                    return Err(WorldCutError::PendingLegacyBoundary);
                }
                if observed_at > self.epoch {
                    return Err(WorldCutError::LegacyBoundaryAfterCut {
                        observed_at,
                        cut: self.epoch,
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_model_formats(&self) -> Result<(), WorldCutError> {
        let has = |format| self.formats.iter().any(|entry| entry.format == format);
        if !has(AuthoritativeFormat::Catalog) {
            return Err(WorldCutError::MissingFormat(AuthoritativeFormat::Catalog));
        }
        if has(AuthoritativeFormat::Lpg) != self.graph_model.has_lpg() {
            return Err(if self.graph_model.has_lpg() {
                WorldCutError::MissingFormat(AuthoritativeFormat::Lpg)
            } else {
                WorldCutError::UnexpectedFormat(AuthoritativeFormat::Lpg)
            });
        }
        if has(AuthoritativeFormat::Rdf) != self.graph_model.has_rdf() {
            return Err(if self.graph_model.has_rdf() {
                WorldCutError::MissingFormat(AuthoritativeFormat::Rdf)
            } else {
                WorldCutError::UnexpectedFormat(AuthoritativeFormat::Rdf)
            });
        }
        if has(AuthoritativeFormat::Compact) && !self.graph_model.has_lpg() {
            return Err(WorldCutError::UnexpectedFormat(
                AuthoritativeFormat::Compact,
            ));
        }
        if has(AuthoritativeFormat::OverlayDeletions) && !has(AuthoritativeFormat::Compact) {
            return Err(WorldCutError::UnexpectedFormat(
                AuthoritativeFormat::OverlayDeletions,
            ));
        }
        if has(AuthoritativeFormat::RdfHistory) != self.graph_model.has_rdf() {
            return Err(if self.graph_model.has_rdf() {
                WorldCutError::MissingFormat(AuthoritativeFormat::RdfHistory)
            } else {
                WorldCutError::UnexpectedFormat(AuthoritativeFormat::RdfHistory)
            });
        }
        Ok(())
    }
}

/// The byte grammar used to derive a cut's state digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum StateDigestKind {
    /// Exact bytes returned by portable snapshot export.
    SnapshotBytes = 1,
    /// Canonical ordered sequence of authoritative container components.
    AuthoritativeComponents = 2,
}

/// Full digest of the authoritative data represented by a cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateDigest {
    kind: StateDigestKind,
    digest: Digest256,
}

impl StateDigest {
    /// Digests exact portable snapshot bytes.
    #[must_use]
    pub fn snapshot_bytes(bytes: &[u8]) -> Self {
        Self {
            kind: StateDigestKind::SnapshotBytes,
            digest: hash_one(SNAPSHOT_STATE_DIGEST_DOMAIN, bytes),
        }
    }

    /// Digests authoritative components independent of caller iteration order.
    ///
    /// Each component contributes its stable format tag, exact version, byte
    /// length, and payload. Duplicate component kinds are rejected.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate kinds or reserved format version zero.
    pub fn authoritative_components(
        components: &[(ModelFormatVersion, &[u8])],
    ) -> Result<Self, WorldCutError> {
        let mut components = components.to_vec();
        if let Some((entry, _)) = components.iter().find(|(entry, _)| entry.version == 0) {
            return Err(WorldCutError::InvalidFormatVersion {
                format: entry.format,
            });
        }
        components.sort_unstable_by_key(|(entry, _)| entry.format);
        if !strictly_sorted_unique_by(&components, |(entry, _)| entry.format) {
            return Err(WorldCutError::DuplicateStateComponent);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMPONENT_STATE_DIGEST_DOMAIN);
        hash_u64(&mut hasher, components.len() as u64);
        for (entry, bytes) in components {
            hasher.update(&[entry.format.as_u8()]);
            hasher.update(&entry.version.to_le_bytes());
            hash_bytes(&mut hasher, bytes);
        }
        Ok(Self {
            kind: StateDigestKind::AuthoritativeComponents,
            digest: Digest256::from_bytes(*hasher.finalize().as_bytes()),
        })
    }

    /// State-digest byte grammar.
    #[must_use]
    pub const fn kind(self) -> StateDigestKind {
        self.kind
    }

    /// Full state digest.
    #[must_use]
    pub const fn digest(self) -> Digest256 {
        self.digest
    }
}

/// A committed world cut sealed by a canonical manifest digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldCut {
    descriptor: WorldCutDescriptor,
    state_digest: StateDigest,
    manifest_digest: Digest256,
}

impl WorldCut {
    /// Current portable wire version for serialized world cuts.
    pub const FORMAT_VERSION: u8 = 1;

    /// Seals validated metadata against an authoritative state digest.
    ///
    /// # Errors
    ///
    /// Returns an error when the descriptor is structurally invalid.
    pub fn seal(
        descriptor: WorldCutDescriptor,
        state_digest: StateDigest,
    ) -> Result<Self, WorldCutError> {
        descriptor.validate()?;
        let manifest_digest = compute_manifest_digest(&descriptor, state_digest);
        Ok(Self {
            descriptor,
            state_digest,
            manifest_digest,
        })
    }

    /// Seals exact authoritative component bytes after proving that their
    /// format/version set exactly matches the descriptor.
    ///
    /// This is the safe constructor for section-based container manifests. It
    /// rejects omitted, extra, duplicated, and zero-version components before
    /// hashing any state.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid metadata or a component format/version set
    /// that does not exactly match the descriptor.
    pub fn seal_components(
        descriptor: WorldCutDescriptor,
        components: &[(ModelFormatVersion, &[u8])],
    ) -> Result<Self, WorldCutError> {
        descriptor.validate()?;
        require_exact_component_formats(&descriptor, components)?;
        Self::seal(
            descriptor,
            StateDigest::authoritative_components(components)?,
        )
    }

    /// Encodes this verified cut with the bounded portable v1 wire grammar.
    ///
    /// # Errors
    ///
    /// Returns an error if verification or bounded serialization fails.
    pub fn encode(&self) -> Result<Vec<u8>, WorldCutError> {
        self.verify()?;
        bincode::serde::encode_to_vec(
            self,
            bincode::config::standard().with_limit::<MAX_WORLD_CUT_BYTES>(),
        )
        .map_err(|error| WorldCutError::WorldCutSerialization(error.to_string()))
    }

    /// Decodes exactly one bounded portable v1 world cut.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, oversized, malformed, unsupported,
    /// trailing, or cryptographically invalid input.
    pub fn decode(bytes: &[u8]) -> Result<Self, WorldCutError> {
        if bytes.is_empty() {
            return Err(WorldCutError::EmptyWorldCut);
        }
        if bytes.len() > MAX_WORLD_CUT_BYTES {
            return Err(WorldCutError::WorldCutTooLarge {
                bytes: bytes.len(),
                maximum: MAX_WORLD_CUT_BYTES,
            });
        }
        let (cut, consumed): (Self, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_WORLD_CUT_BYTES>(),
        )
        .map_err(|error| WorldCutError::WorldCutSerialization(error.to_string()))?;
        if consumed != bytes.len() {
            return Err(WorldCutError::TrailingWorldCut {
                trailing: bytes.len() - consumed,
            });
        }
        cut.verify()?;
        Ok(cut)
    }

    /// Verifies structural invariants and the canonical manifest digest.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid metadata or a manifest-digest mismatch.
    pub fn verify(&self) -> Result<(), WorldCutError> {
        self.descriptor.validate()?;
        let actual = compute_manifest_digest(&self.descriptor, self.state_digest);
        if actual != self.manifest_digest {
            return Err(WorldCutError::ManifestDigestMismatch {
                expected: self.manifest_digest,
                actual,
            });
        }
        Ok(())
    }

    /// Verifies the cut and requires one logical store identity.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest verification fails or the store differs.
    pub fn verify_for_store(&self, expected: StoreId) -> Result<(), WorldCutError> {
        self.verify()?;
        if self.store_id() != expected {
            return Err(WorldCutError::StoreIdentityMismatch {
                expected,
                actual: self.store_id(),
            });
        }
        Ok(())
    }

    /// Verifies the manifest against an exact authoritative component set.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong digest grammar, component mismatch, byte
    /// mismatch, or invalid manifest.
    pub fn verify_components(
        &self,
        components: &[(ModelFormatVersion, &[u8])],
    ) -> Result<(), WorldCutError> {
        self.verify()?;
        if self.state_digest.kind != StateDigestKind::AuthoritativeComponents {
            return Err(WorldCutError::WrongStateDigestKind {
                expected: StateDigestKind::AuthoritativeComponents,
                actual: self.state_digest.kind,
            });
        }
        require_exact_component_formats(&self.descriptor, components)?;
        let actual = StateDigest::authoritative_components(components)?;
        if actual != self.state_digest {
            return Err(WorldCutError::StateDigestMismatch {
                expected: self.state_digest.digest,
                actual: actual.digest,
            });
        }
        Ok(())
    }

    /// Unsealed metadata carried by this cut.
    #[must_use]
    pub const fn descriptor(&self) -> &WorldCutDescriptor {
        &self.descriptor
    }

    /// Logical store identity.
    #[must_use]
    pub const fn store_id(&self) -> StoreId {
        self.descriptor.store_id
    }

    /// Committed epoch.
    #[must_use]
    pub const fn epoch(&self) -> EpochId {
        self.descriptor.epoch
    }

    /// Digest of the represented authoritative data.
    #[must_use]
    pub const fn state_digest(&self) -> StateDigest {
        self.state_digest
    }

    /// Full canonical manifest digest.
    #[must_use]
    pub const fn manifest_digest(&self) -> Digest256 {
        self.manifest_digest
    }
}

#[derive(Serialize, Deserialize)]
struct WorldCutWire {
    version: u8,
    descriptor: WorldCutDescriptor,
    state_digest: StateDigest,
    manifest_digest: Digest256,
}

impl Serialize for WorldCut {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WorldCutWire {
            version: Self::FORMAT_VERSION,
            descriptor: self.descriptor.clone(),
            state_digest: self.state_digest,
            manifest_digest: self.manifest_digest,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WorldCut {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WorldCutWire::deserialize(deserializer)?;
        if wire.version != Self::FORMAT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported world-cut wire version {}",
                wire.version
            )));
        }
        let cut = Self {
            descriptor: wire.descriptor,
            state_digest: wire.state_digest,
            manifest_digest: wire.manifest_digest,
        };
        cut.verify().map_err(serde::de::Error::custom)?;
        Ok(cut)
    }
}

/// Immutable portable snapshot bytes and the cut that integrity-binds them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotArtifact {
    bytes: Arc<[u8]>,
    cut: WorldCut,
}

impl SnapshotArtifact {
    /// Seals exact snapshot bytes with validated world metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when the descriptor is invalid or omits an explicit
    /// portable-snapshot format version.
    pub fn new(
        bytes: impl Into<Arc<[u8]>>,
        descriptor: WorldCutDescriptor,
    ) -> Result<Self, WorldCutError> {
        let bytes = bytes.into();
        require_portable_snapshot_format(&descriptor)?;
        let cut = WorldCut::seal(descriptor, StateDigest::snapshot_bytes(&bytes))?;
        Ok(Self { bytes, cut })
    }

    /// Restores an artifact while verifying that the cut binds the exact bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the cut is invalid, is not a snapshot cut, or does
    /// not match the supplied bytes.
    pub fn from_parts(bytes: impl Into<Arc<[u8]>>, cut: WorldCut) -> Result<Self, WorldCutError> {
        let artifact = Self {
            bytes: bytes.into(),
            cut,
        };
        require_portable_snapshot_format(artifact.cut.descriptor())?;
        artifact.verify()?;
        Ok(artifact)
    }

    /// Verifies both the cut manifest and every snapshot byte.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid metadata, the wrong digest grammar, or a
    /// byte-digest mismatch.
    pub fn verify(&self) -> Result<(), WorldCutError> {
        self.cut.verify()?;
        require_portable_snapshot_format(self.cut.descriptor())?;
        if self.cut.state_digest.kind != StateDigestKind::SnapshotBytes {
            return Err(WorldCutError::WrongStateDigestKind {
                expected: StateDigestKind::SnapshotBytes,
                actual: self.cut.state_digest.kind,
            });
        }
        let actual = StateDigest::snapshot_bytes(&self.bytes);
        if actual != self.cut.state_digest {
            return Err(WorldCutError::StateDigestMismatch {
                expected: self.cut.state_digest.digest,
                actual: actual.digest,
            });
        }
        Ok(())
    }

    /// Verifies the artifact and requires one logical store identity.
    ///
    /// # Errors
    ///
    /// Returns an error when artifact verification fails or the store differs.
    pub fn verify_for_store(&self, expected: StoreId) -> Result<(), WorldCutError> {
        self.verify()?;
        self.cut.verify_for_store(expected)
    }

    /// Exact immutable snapshot bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Integrity-sealed world cut.
    #[must_use]
    pub const fn cut(&self) -> &WorldCut {
        &self.cut
    }

    /// Separates the immutable bytes and integrity-sealed cut.
    #[must_use]
    pub fn into_parts(self) -> (Arc<[u8]>, WorldCut) {
        (self.bytes, self.cut)
    }
}

/// Required authoritative payload stored in a `.grafeo` WorldMetadata
/// section with directory section-version 1.
///
/// The container CRC detects accidental section damage; this payload provides
/// the independent cryptographic binding from the logical store/cut metadata
/// to every authoritative component byte and serializer version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldMetadataSectionV1 {
    cut: WorldCut,
}

impl WorldMetadataSectionV1 {
    /// Exact section-directory version for this payload.
    pub const SECTION_VERSION: u8 = 1;
    const MAGIC: [u8; 8] = *b"WRLDCUT1";

    /// Wraps an already sealed authoritative-component cut.
    ///
    /// Call [`Self::seal`] when component bytes are available so their exact
    /// format correspondence is checked during construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the cut is invalid or does not use authoritative
    /// component hashing.
    pub fn new(cut: WorldCut) -> Result<Self, WorldCutError> {
        cut.verify()?;
        if cut.state_digest.kind != StateDigestKind::AuthoritativeComponents {
            return Err(WorldCutError::WrongStateDigestKind {
                expected: StateDigestKind::AuthoritativeComponents,
                actual: cut.state_digest.kind,
            });
        }
        Ok(Self { cut })
    }

    /// Seals a descriptor and exact authoritative section payloads.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid metadata or mismatched components.
    pub fn seal(
        descriptor: WorldCutDescriptor,
        components: &[(ModelFormatVersion, &[u8])],
    ) -> Result<Self, WorldCutError> {
        Self::new(WorldCut::seal_components(descriptor, components)?)
    }

    /// Encodes the exact versioned WorldMetadata section payload.
    ///
    /// # Errors
    ///
    /// Returns an error if manifest verification or serialization fails.
    pub fn encode(&self) -> Result<Vec<u8>, WorldCutError> {
        self.cut.verify()?;
        bincode::serde::encode_to_vec(
            self,
            bincode::config::standard().with_limit::<MAX_WORLD_METADATA_BYTES>(),
        )
        .map_err(|error| WorldCutError::WorldMetadataSerialization(error.to_string()))
    }

    /// Decodes a complete WorldMetadata section without accepting trailing
    /// bytes or an unverified manifest.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, malformed, unsupported, trailing, or
    /// cryptographically invalid input.
    pub fn decode(bytes: &[u8]) -> Result<Self, WorldCutError> {
        if bytes.is_empty() {
            return Err(WorldCutError::EmptyWorldMetadata);
        }
        if bytes.len() > MAX_WORLD_METADATA_BYTES {
            return Err(WorldCutError::WorldMetadataTooLarge {
                bytes: bytes.len(),
                maximum: MAX_WORLD_METADATA_BYTES,
            });
        }
        let (metadata, consumed): (Self, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_WORLD_METADATA_BYTES>(),
        )
        .map_err(|error| WorldCutError::WorldMetadataSerialization(error.to_string()))?;
        if consumed != bytes.len() {
            return Err(WorldCutError::TrailingWorldMetadata {
                trailing: bytes.len() - consumed,
            });
        }
        metadata.cut.verify()?;
        Ok(metadata)
    }

    /// Verifies the cut and exact authoritative component bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata, component formats, or bytes differ.
    pub fn verify_components(
        &self,
        components: &[(ModelFormatVersion, &[u8])],
    ) -> Result<(), WorldCutError> {
        self.cut.verify_components(components)
    }

    /// Verifies exact components and rejects a foreign logical-store namespace.
    ///
    /// # Errors
    ///
    /// Returns an error when component verification fails or the store differs.
    pub fn verify_for_store(
        &self,
        expected: StoreId,
        components: &[(ModelFormatVersion, &[u8])],
    ) -> Result<(), WorldCutError> {
        self.cut.verify_for_store(expected)?;
        self.cut.verify_components(components)
    }

    /// Integrity-sealed world cut carried by this section.
    #[must_use]
    pub const fn cut(&self) -> &WorldCut {
        &self.cut
    }
}

#[derive(Serialize, Deserialize)]
struct WorldMetadataSectionWireV1 {
    magic: [u8; 8],
    version: u8,
    cut: WorldCut,
}

impl Serialize for WorldMetadataSectionV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WorldMetadataSectionWireV1 {
            magic: Self::MAGIC,
            version: Self::SECTION_VERSION,
            cut: self.cut.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WorldMetadataSectionV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WorldMetadataSectionWireV1::deserialize(deserializer)?;
        if wire.magic != Self::MAGIC {
            return Err(serde::de::Error::custom("invalid WorldMetadata magic"));
        }
        if wire.version != Self::SECTION_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported WorldMetadata section version {}",
                wire.version
            )));
        }
        Self::new(wire.cut).map_err(serde::de::Error::custom)
    }
}

/// One exact component in a physical recovery image.
///
/// Most components are non-WorldMetadata sections. Reserved synthetic
/// component identifiers may additionally bind recovery-critical state which
/// lives outside the section directory, such as [`RecoveryImageCoordinatesV1`].
/// Numeric identifiers rather than [`crate::storage::SectionType`] values keep
/// the integrity grammar usable by storage implementations and
/// forward-compatible readers that do not understand every component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryImageComponent<'a> {
    section_type: u32,
    directory_version: u16,
    bytes: &'a [u8],
}

impl<'a> RecoveryImageComponent<'a> {
    /// Describes one exact physical recovery component.
    ///
    /// # Errors
    ///
    /// Returns an error for the reserved type zero, the recursively excluded
    /// WorldMetadata type, or reserved directory version zero.
    pub const fn new(
        section_type: u32,
        directory_version: u16,
        bytes: &'a [u8],
    ) -> Result<Self, WorldCutError> {
        if section_type == 0 || section_type == WORLD_METADATA_SECTION_TYPE_ID {
            return Err(WorldCutError::InvalidRecoveryImageSectionType { section_type });
        }
        if directory_version == 0 {
            return Err(WorldCutError::InvalidRecoveryImageSectionVersion { section_type });
        }
        Ok(Self {
            section_type,
            directory_version,
            bytes,
        })
    }

    /// Stable numeric recovery-component identifier.
    #[must_use]
    pub const fn section_type(self) -> u32 {
        self.section_type
    }

    /// Exact section serializer or synthetic-component version.
    #[must_use]
    pub const fn directory_version(self) -> u16 {
        self.directory_version
    }

    /// Exact component payload bytes.
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// Canonical recovery-critical coordinates carried outside the section directory.
///
/// WorldMetadata v2 deliberately leaves the logical [`WorldCut`] unchanged,
/// but a storage engine can include this reserved synthetic component in its
/// recovery-image digest. Doing so binds the WAL replay floor and the other
/// active-header coordinates to the exact section inventory, preventing a
/// separately valid header from being spliced onto a foreign checkpoint.
///
/// The frozen byte grammar is:
///
/// `domain || graph_model:u8 || epoch:u64le || transaction_id:u64le || node_count:u64le || edge_count:u64le`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryImageCoordinatesV1 {
    epoch: u64,
    transaction_id: u64,
    graph_model: GraphModelTag,
    node_count: u64,
    edge_count: u64,
    encoded: [u8; RECOVERY_IMAGE_COORDINATES_V1_ENCODED_LEN],
}

impl RecoveryImageCoordinatesV1 {
    /// Reserved synthetic recovery-image component identifier.
    ///
    /// This value is outside the current `.grafeo` section namespace and must
    /// never be assigned to a real section.
    pub const COMPONENT_TYPE: u32 = u32::MAX;
    /// Version of the synthetic component's canonical byte grammar.
    pub const COMPONENT_VERSION: u16 = 1;
    /// Exact length of the canonical coordinate bytes.
    pub const ENCODED_LEN: usize = RECOVERY_IMAGE_COORDINATES_V1_ENCODED_LEN;

    /// Encodes one set of recovery coordinates using the frozen v1 grammar.
    #[must_use]
    pub fn new(
        epoch: u64,
        transaction_id: u64,
        graph_model: GraphModelTag,
        node_count: u64,
        edge_count: u64,
    ) -> Self {
        let mut encoded = [0; RECOVERY_IMAGE_COORDINATES_V1_ENCODED_LEN];
        let mut offset = 0;

        encoded[..RECOVERY_IMAGE_COORDINATES_V1_DOMAIN.len()]
            .copy_from_slice(RECOVERY_IMAGE_COORDINATES_V1_DOMAIN);
        offset += RECOVERY_IMAGE_COORDINATES_V1_DOMAIN.len();
        encoded[offset] = graph_model.as_u8();
        offset += 1;

        for value in [epoch, transaction_id, node_count, edge_count] {
            let end = offset + size_of::<u64>();
            encoded[offset..end].copy_from_slice(&value.to_le_bytes());
            offset = end;
        }
        debug_assert_eq!(offset, encoded.len());

        Self {
            epoch,
            transaction_id,
            graph_model,
            node_count,
            edge_count,
            encoded,
        }
    }

    /// Checkpoint epoch stored in the active database header.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Transaction-id high-water mark used as this checkpoint's WAL replay floor.
    #[must_use]
    pub const fn transaction_id(&self) -> u64 {
        self.transaction_id
    }

    /// Graph-model tag stored in the immutable file header.
    #[must_use]
    pub const fn graph_model(&self) -> GraphModelTag {
        self.graph_model
    }

    /// Cached node cardinality stored in the active database header.
    #[must_use]
    pub const fn node_count(&self) -> u64 {
        self.node_count
    }

    /// Cached edge cardinality stored in the active database header.
    #[must_use]
    pub const fn edge_count(&self) -> u64 {
        self.edge_count
    }

    /// Exact canonical bytes committed by the synthetic component.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; RECOVERY_IMAGE_COORDINATES_V1_ENCODED_LEN] {
        &self.encoded
    }

    /// Borrows these canonical bytes as the reserved digest component.
    #[must_use]
    pub const fn component(&self) -> RecoveryImageComponent<'_> {
        RecoveryImageComponent {
            section_type: Self::COMPONENT_TYPE,
            directory_version: Self::COMPONENT_VERSION,
            bytes: &self.encoded,
        }
    }
}

/// Domain-separated digest of a complete physical recovery-component image.
///
/// The v1 digest grammar is:
///
/// `domain || count:u64le || (type:u32le || version:u16le || length:u64le || bytes)*`
///
/// Components are ordered by numeric component type before hashing. A type may
/// occur only once, matching both the `.grafeo` directory invariant and the
/// reserved synthetic-component namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RecoveryImageDigest(Digest256);

impl fmt::Display for RecoveryImageDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl RecoveryImageDigest {
    /// Hashes an exact canonical recovery-component inventory.
    ///
    /// # Errors
    ///
    /// Returns an error for too many components, duplicate component types, a
    /// reserved component type, or component version zero.
    pub fn from_components(
        components: &[RecoveryImageComponent<'_>],
    ) -> Result<Self, WorldCutError> {
        if components.len() > MAX_RECOVERY_IMAGE_COMPONENTS {
            return Err(WorldCutError::TooManyRecoveryImageComponents {
                count: components.len(),
                maximum: MAX_RECOVERY_IMAGE_COMPONENTS,
            });
        }

        let mut ordered = components.to_vec();
        ordered.sort_unstable_by_key(|component| component.section_type);
        for component in &ordered {
            // Revalidate here rather than relying on callers to preserve the
            // constructor invariant across future representation changes.
            Self::validate_component(component)?;
        }
        if let Some(pair) = ordered
            .windows(2)
            .find(|pair| pair[0].section_type == pair[1].section_type)
        {
            return Err(WorldCutError::DuplicateRecoveryImageSection {
                section_type: pair[0].section_type,
            });
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(RECOVERY_IMAGE_DIGEST_DOMAIN);
        hash_u64(&mut hasher, ordered.len() as u64);
        for component in ordered {
            hasher.update(&component.section_type.to_le_bytes());
            hasher.update(&component.directory_version.to_le_bytes());
            hash_bytes(&mut hasher, component.bytes);
        }
        Ok(Self(Digest256::from_bytes(*hasher.finalize().as_bytes())))
    }

    fn validate_component(component: &RecoveryImageComponent<'_>) -> Result<(), WorldCutError> {
        if component.section_type == 0 || component.section_type == WORLD_METADATA_SECTION_TYPE_ID {
            return Err(WorldCutError::InvalidRecoveryImageSectionType {
                section_type: component.section_type,
            });
        }
        if component.directory_version == 0 {
            return Err(WorldCutError::InvalidRecoveryImageSectionVersion {
                section_type: component.section_type,
            });
        }
        Ok(())
    }

    /// Returns all 256 digest bits.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Digest256::LEN] {
        self.0.as_bytes()
    }
}

/// WorldMetadata v2: an unchanged logical [`WorldCut`] plus an exact physical
/// recovery-image digest.
///
/// The logical cut retains the frozen v1 world-manifest semantics. The
/// separate recovery digest binds every component supplied by the storage
/// protocol. These normally include every non-WorldMetadata section and may
/// include reserved synthetic state such as [`RecoveryImageCoordinatesV1`],
/// without reclassifying any of it as authoritative graph state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldMetadataSectionV2 {
    cut: WorldCut,
    recovery_image_digest: RecoveryImageDigest,
}

impl WorldMetadataSectionV2 {
    /// Exact section-directory version for this payload.
    pub const SECTION_VERSION: u8 = 2;
    const MAGIC: [u8; 8] = *b"WRLDCUT2";

    /// Seals an existing authoritative-component cut and exact physical image.
    ///
    /// The caller supplies the complete inventory defined by its recovery
    /// protocol; numeric physical component types are intentionally not
    /// interpreted here.
    ///
    /// # Errors
    ///
    /// Returns an error if the cut is invalid, uses another state-digest
    /// grammar, or the recovery inventory is invalid.
    pub fn seal(
        cut: WorldCut,
        components: &[RecoveryImageComponent<'_>],
    ) -> Result<Self, WorldCutError> {
        validate_authoritative_component_cut(&cut)?;
        Ok(Self {
            cut,
            recovery_image_digest: RecoveryImageDigest::from_components(components)?,
        })
    }

    /// Encodes the exact bounded WorldMetadata v2 payload.
    ///
    /// # Errors
    ///
    /// Returns an error if logical-cut verification or serialization fails.
    pub fn encode(&self) -> Result<Vec<u8>, WorldCutError> {
        validate_authoritative_component_cut(&self.cut)?;
        bincode::serde::encode_to_vec(
            self,
            bincode::config::standard().with_limit::<MAX_WORLD_METADATA_BYTES>(),
        )
        .map_err(|error| WorldCutError::WorldMetadataSerialization(error.to_string()))
    }

    /// Decodes exactly one bounded WorldMetadata v2 payload.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, oversized, malformed, unsupported,
    /// trailing, or logically invalid input.
    pub fn decode(bytes: &[u8]) -> Result<Self, WorldCutError> {
        if bytes.is_empty() {
            return Err(WorldCutError::EmptyWorldMetadata);
        }
        if bytes.len() > MAX_WORLD_METADATA_BYTES {
            return Err(WorldCutError::WorldMetadataTooLarge {
                bytes: bytes.len(),
                maximum: MAX_WORLD_METADATA_BYTES,
            });
        }
        let (metadata, consumed): (Self, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_WORLD_METADATA_BYTES>(),
        )
        .map_err(|error| WorldCutError::WorldMetadataSerialization(error.to_string()))?;
        if consumed != bytes.len() {
            return Err(WorldCutError::TrailingWorldMetadata {
                trailing: bytes.len() - consumed,
            });
        }
        validate_authoritative_component_cut(&metadata.cut)?;
        Ok(metadata)
    }

    /// Verifies the logical cut and every supplied physical recovery byte.
    ///
    /// The caller must provide the complete current component inventory defined
    /// by its recovery protocol. Omitted, added, relabeled, re-versioned, or
    /// changed components produce a digest mismatch. This verifies the cut's
    /// own manifest but cannot infer which numeric physical sections implement
    /// its logical formats; recovery must also call
    /// [`WorldCut::verify_components`] with the authoritative-format mapping.
    ///
    /// # Errors
    ///
    /// Returns an error when the cut, inventory, or recovery digest is invalid.
    pub fn verify_recovery_components(
        &self,
        components: &[RecoveryImageComponent<'_>],
    ) -> Result<(), WorldCutError> {
        validate_authoritative_component_cut(&self.cut)?;
        let actual = RecoveryImageDigest::from_components(components)?;
        if actual != self.recovery_image_digest {
            return Err(WorldCutError::RecoveryImageDigestMismatch {
                expected: self.recovery_image_digest,
                actual,
            });
        }
        Ok(())
    }

    /// Integrity-sealed logical world cut, unchanged from v1 semantics.
    #[must_use]
    pub const fn cut(&self) -> &WorldCut {
        &self.cut
    }

    /// Digest of the exact physical recovery-component image.
    #[must_use]
    pub const fn recovery_image_digest(&self) -> RecoveryImageDigest {
        self.recovery_image_digest
    }
}

#[derive(Serialize, Deserialize)]
struct WorldMetadataSectionWireV2 {
    magic: [u8; 8],
    version: u8,
    cut: WorldCut,
    recovery_image_digest: RecoveryImageDigest,
}

impl Serialize for WorldMetadataSectionV2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WorldMetadataSectionWireV2 {
            magic: Self::MAGIC,
            version: Self::SECTION_VERSION,
            cut: self.cut.clone(),
            recovery_image_digest: self.recovery_image_digest,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WorldMetadataSectionV2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WorldMetadataSectionWireV2::deserialize(deserializer)?;
        if wire.magic != Self::MAGIC {
            return Err(serde::de::Error::custom("invalid WorldMetadata v2 magic"));
        }
        if wire.version != Self::SECTION_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported WorldMetadata section version {}",
                wire.version
            )));
        }
        validate_authoritative_component_cut(&wire.cut).map_err(serde::de::Error::custom)?;
        Ok(Self {
            cut: wire.cut,
            recovery_image_digest: wire.recovery_image_digest,
        })
    }
}

fn validate_authoritative_component_cut(cut: &WorldCut) -> Result<(), WorldCutError> {
    cut.verify()?;
    if cut.state_digest.kind != StateDigestKind::AuthoritativeComponents {
        return Err(WorldCutError::WrongStateDigestKind {
            expected: StateDigestKind::AuthoritativeComponents,
            actual: cut.state_digest.kind,
        });
    }
    Ok(())
}

/// A world cut, manifest, or snapshot artifact failed validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WorldCutError {
    /// A stable graph-model tag was unknown.
    #[error("invalid graph model tag {0}")]
    InvalidGraphModel(u8),
    /// A model format used reserved version zero.
    #[error("{format:?} format version must be non-zero")]
    InvalidFormatVersion {
        /// Invalid model format.
        format: AuthoritativeFormat,
    },
    /// Schema format used reserved version zero.
    #[error("schema format version must be non-zero")]
    InvalidSchemaVersion,
    /// A world cut tried to name the uncommitted sentinel epoch.
    #[error("world cut cannot use the pending/uncommitted epoch")]
    PendingEpoch,
    /// A required authoritative model format was absent.
    #[error("world cut is missing required {0:?} format")]
    MissingFormat(AuthoritativeFormat),
    /// A format contradicted the declared graph model or storage layout.
    #[error("world cut contains unexpected {0:?} format")]
    UnexpectedFormat(AuthoritativeFormat),
    /// Format entries were duplicated or not canonically ordered.
    #[error("world-cut formats are duplicated or not in canonical order")]
    NonCanonicalFormats,
    /// Projection entries were duplicated or not canonically ordered.
    #[error("world-cut projections are duplicated or not in canonical order")]
    NonCanonicalProjections,
    /// A cut exceeded the defensive projection-count bound.
    #[error("world cut contains {count} projections; maximum is {maximum}")]
    TooManyProjections {
        /// Supplied projection count.
        count: usize,
        /// Maximum accepted projection count.
        maximum: usize,
    },
    /// RDF→LPG projection metadata was attached to a single-model store.
    #[error("RDF-to-LPG projection metadata requires GraphModel::Both")]
    ProjectionRequiresBothModels,
    /// Projection provenance was structurally impossible.
    #[error("invalid projection cut: {0}")]
    InvalidProjection(String),
    /// Legacy provenance used the reserved source version zero.
    #[error("legacy history source version must be non-zero")]
    InvalidLegacySourceVersion,
    /// Legacy provenance used the uncommitted epoch sentinel.
    #[error("legacy history boundary cannot be pending/uncommitted")]
    PendingLegacyBoundary,
    /// A legacy history boundary lay after the cut it described.
    #[error("legacy history boundary {observed_at} lies after world-cut epoch {cut}")]
    LegacyBoundaryAfterCut {
        /// Claimed history boundary.
        observed_at: EpochId,
        /// World-cut epoch.
        cut: EpochId,
    },
    /// Authoritative state supplied the same component kind more than once.
    #[error("authoritative state contains a duplicate component kind")]
    DuplicateStateComponent,
    /// Component format/version entries did not exactly match the descriptor.
    #[error("authoritative component formats do not exactly match the world-cut descriptor")]
    ComponentFormatMismatch,
    /// A recovery component used a reserved or recursively included type.
    #[error("recovery image contains invalid section type {section_type}")]
    InvalidRecoveryImageSectionType {
        /// Invalid stable section type.
        section_type: u32,
    },
    /// A recovery component used reserved directory version zero.
    #[error("recovery-image section {section_type} has reserved version zero")]
    InvalidRecoveryImageSectionVersion {
        /// Stable section type whose version was invalid.
        section_type: u32,
    },
    /// The same physical section type appeared more than once.
    #[error("recovery image contains duplicate section type {section_type}")]
    DuplicateRecoveryImageSection {
        /// Duplicated stable section type.
        section_type: u32,
    },
    /// A recovery image exceeded its canonicalization allocation bound.
    #[error("recovery image contains {count} sections; maximum is {maximum}")]
    TooManyRecoveryImageComponents {
        /// Supplied section count.
        count: usize,
        /// Maximum accepted section count.
        maximum: usize,
    },
    /// Exact physical recovery bytes did not match their stored digest.
    #[error("recovery image digest mismatch: expected {expected}, calculated {actual}")]
    RecoveryImageDigestMismatch {
        /// Digest stored in WorldMetadata v2.
        expected: RecoveryImageDigest,
        /// Digest calculated from the supplied physical image.
        actual: RecoveryImageDigest,
    },
    /// Stored manifest digest did not match canonical metadata.
    #[error("manifest digest mismatch: expected {expected}, calculated {actual}")]
    ManifestDigestMismatch {
        /// Digest stored in the cut.
        expected: Digest256,
        /// Digest calculated from canonical fields.
        actual: Digest256,
    },
    /// State bytes did not match the digest bound by the cut.
    #[error("state digest mismatch: expected {expected}, calculated {actual}")]
    StateDigestMismatch {
        /// Digest stored in the cut.
        expected: Digest256,
        /// Digest calculated from supplied state bytes.
        actual: Digest256,
    },
    /// An artifact used the wrong state-digest byte grammar.
    #[error("wrong state digest kind: expected {expected:?}, got {actual:?}")]
    WrongStateDigestKind {
        /// Required byte grammar.
        expected: StateDigestKind,
        /// Supplied byte grammar.
        actual: StateDigestKind,
    },
    /// Verification was attempted in a different store namespace.
    #[error("store identity mismatch: expected {expected}, got {actual}")]
    StoreIdentityMismatch {
        /// Store required by the caller.
        expected: StoreId,
        /// Store bound by the artifact.
        actual: StoreId,
    },
    /// Identity metadata payload was empty.
    #[error("world identity metadata is empty")]
    EmptyIdentityMetadata,
    /// Identity metadata exceeded its defensive byte bound.
    #[error("world identity metadata has {bytes} bytes; maximum is {maximum}")]
    IdentityMetadataTooLarge {
        /// Supplied payload length.
        bytes: usize,
        /// Maximum accepted payload length.
        maximum: usize,
    },
    /// Identity metadata could not be encoded or decoded.
    #[error("world identity metadata serialization failed: {0}")]
    IdentityMetadataSerialization(String),
    /// Identity metadata contained bytes after its exact wire value.
    #[error("world identity metadata has {trailing} trailing bytes")]
    TrailingIdentityMetadata {
        /// Number of unconsumed bytes.
        trailing: usize,
    },
    /// Standalone world-cut payload was empty.
    #[error("world cut is empty")]
    EmptyWorldCut,
    /// Standalone world-cut payload exceeded its defensive byte bound.
    #[error("world cut has {bytes} bytes; maximum is {maximum}")]
    WorldCutTooLarge {
        /// Supplied payload length.
        bytes: usize,
        /// Maximum accepted payload length.
        maximum: usize,
    },
    /// A standalone world cut could not be encoded or decoded.
    #[error("world-cut serialization failed: {0}")]
    WorldCutSerialization(String),
    /// Standalone world-cut bytes contained an unframed suffix.
    #[error("world cut has {trailing} trailing bytes")]
    TrailingWorldCut {
        /// Number of unconsumed bytes.
        trailing: usize,
    },
    /// WorldMetadata section payload was empty.
    #[error("WorldMetadata section is empty")]
    EmptyWorldMetadata,
    /// WorldMetadata exceeded its defensive byte bound.
    #[error("WorldMetadata section has {bytes} bytes; maximum is {maximum}")]
    WorldMetadataTooLarge {
        /// Supplied payload length.
        bytes: usize,
        /// Maximum accepted payload length.
        maximum: usize,
    },
    /// WorldMetadata could not be encoded or decoded.
    #[error("WorldMetadata section serialization failed: {0}")]
    WorldMetadataSerialization(String),
    /// WorldMetadata contained bytes after its exact wire payload.
    #[error("WorldMetadata section has {trailing} trailing bytes")]
    TrailingWorldMetadata {
        /// Number of unconsumed bytes.
        trailing: usize,
    },
}

fn compute_manifest_digest(
    descriptor: &WorldCutDescriptor,
    state_digest: StateDigest,
) -> Digest256 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(WORLD_MANIFEST_DIGEST_DOMAIN);
    hasher.update(descriptor.store_id.as_bytes());
    hash_u64(&mut hasher, descriptor.epoch.as_u64());
    hasher.update(&[descriptor.graph_model.as_u8()]);
    hash_u64(&mut hasher, descriptor.formats.len() as u64);
    for entry in &descriptor.formats {
        hasher.update(&[entry.format.as_u8()]);
        hasher.update(&entry.version.to_le_bytes());
    }
    hasher.update(&descriptor.schema.format_version.to_le_bytes());
    hasher.update(descriptor.schema.digest.as_bytes());
    hash_u64(&mut hasher, descriptor.projections.len() as u64);
    for projection in &descriptor.projections {
        hasher.update(projection.mapping_digest.as_bytes());
        hasher.update(&projection.format_version.to_le_bytes());
        hash_u64(&mut hasher, projection.generation);
        hash_projection_source_graph(&mut hasher, &projection.source_graph);
        hash_optional_epoch(&mut hasher, projection.source_epoch);
        hash_optional_epoch(&mut hasher, projection.target_epoch);
        hash_u64(&mut hasher, projection.row_count);
        hasher.update(&[projection.reconciliation.as_u8()]);
        hash_optional_store_id(&mut hasher, projection.receipt_store_id);
        hash_optional_digest(&mut hasher, projection.receipt_digest);
    }
    match descriptor.history {
        HistoryCompleteness::Complete => {
            hasher.update(&[0]);
        }
        HistoryCompleteness::LegacyCurrentState {
            observed_at,
            source_version,
        } => {
            hasher.update(&[1]);
            hash_u64(&mut hasher, observed_at.as_u64());
            hasher.update(&source_version.to_le_bytes());
        }
    }
    hasher.update(&[state_digest.kind as u8]);
    hasher.update(state_digest.digest.as_bytes());
    Digest256::from_bytes(*hasher.finalize().as_bytes())
}

fn hash_one(domain: &[u8], bytes: &[u8]) -> Digest256 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hash_bytes(&mut hasher, bytes);
    Digest256::from_bytes(*hasher.finalize().as_bytes())
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hash_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn hash_optional_epoch(hasher: &mut blake3::Hasher, epoch: Option<EpochId>) {
    match epoch {
        None => {
            hasher.update(&[0]);
        }
        Some(epoch) => {
            hasher.update(&[1]);
            hash_u64(hasher, epoch.as_u64());
        }
    }
}

fn hash_optional_digest(hasher: &mut blake3::Hasher, digest: Option<Digest256>) {
    match digest {
        None => {
            hasher.update(&[0]);
        }
        Some(digest) => {
            hasher.update(&[1]);
            hasher.update(digest.as_bytes());
        }
    }
}

fn hash_optional_store_id(hasher: &mut blake3::Hasher, store_id: Option<StoreId>) {
    match store_id {
        None => {
            hasher.update(&[0]);
        }
        Some(store_id) => {
            hasher.update(&[1]);
            hasher.update(store_id.as_bytes());
        }
    }
}

fn hash_projection_source_graph(hasher: &mut blake3::Hasher, graph: &ProjectionSourceGraph) {
    match graph.name() {
        None => {
            hasher.update(&[0]);
        }
        Some(name) => {
            hasher.update(&[1]);
            hash_bytes(hasher, name.as_bytes());
        }
    }
    hash_u64(hasher, graph.incarnation.as_u64());
}

fn require_portable_snapshot_format(descriptor: &WorldCutDescriptor) -> Result<(), WorldCutError> {
    if descriptor
        .formats
        .iter()
        .any(|entry| entry.format == AuthoritativeFormat::PortableSnapshot)
    {
        Ok(())
    } else {
        Err(WorldCutError::MissingFormat(
            AuthoritativeFormat::PortableSnapshot,
        ))
    }
}

fn require_exact_component_formats(
    descriptor: &WorldCutDescriptor,
    components: &[(ModelFormatVersion, &[u8])],
) -> Result<(), WorldCutError> {
    let mut actual: Vec<_> = components.iter().map(|(format, _)| *format).collect();
    if let Some(entry) = actual.iter().find(|entry| entry.version == 0) {
        return Err(WorldCutError::InvalidFormatVersion {
            format: entry.format,
        });
    }
    actual.sort_unstable_by_key(|entry| entry.format);
    if !strictly_sorted_unique_by(&actual, |entry| entry.format) {
        return Err(WorldCutError::DuplicateStateComponent);
    }
    if actual != descriptor.formats {
        return Err(WorldCutError::ComponentFormatMismatch);
    }
    Ok(())
}

fn strictly_sorted_unique_by<T, K: Ord>(values: &[T], key: impl Fn(&T) -> K) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}
