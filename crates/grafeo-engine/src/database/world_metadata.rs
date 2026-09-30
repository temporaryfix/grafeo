//! Integrity-sealed metadata for one complete `.grafeo` container image.
//!
//! The storage layer provides atomic installation and CRCs. This module adds
//! the logical invariant above it: every authoritative plaintext payload and
//! its independent serializer version belong to one store-scoped world cut.

use std::collections::HashSet;

use grafeo_common::storage::{Section, SectionType};
use grafeo_common::types::{
    AuthoritativeFormat, EpochId, GraphModelTag, ModelFormatVersion, ProjectionCut,
    RecoveryImageComponent, RecoveryImageCoordinatesV1, SchemaCut, WorldCut, WorldCutDescriptor,
    WorldCutError, WorldIdentityMetadataV1, WorldMetadataSectionV1, WorldMetadataSectionV2,
};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_storage::container::SectionDirectory;
use grafeo_storage::file::GrafeoFileManager;

pub(super) type RecoveryCoordinates = RecoveryImageCoordinatesV1;

/// One section serialized exactly once for atomic container publication.
pub(super) struct EncodedSection {
    section_type: SectionType,
    version: u8,
    bytes: Vec<u8>,
}

/// Verified metadata generations accepted by container recovery.
///
/// V1 binds only logical model components and is retained for frozen legacy
/// acceleration generations. Both V2 profiles seal every physical section;
/// the current profile additionally seals recovery-critical header
/// coordinates without changing the logical [`WorldCut`].
#[derive(Debug)]
pub(super) enum VerifiedWorldMetadata {
    LegacyV1(WorldMetadataSectionV1),
    /// Frozen V2 profile emitted before header coordinates joined the physical
    /// recovery inventory. Exact section bytes are authenticated, but header
    /// transaction/cardinality coordinates are not.
    SectionSealedV2(WorldMetadataSectionV2),
    /// Current V2 profile: sections and recovery-header coordinates are one
    /// authenticated physical image.
    RecoverySealedV2(WorldMetadataSectionV2),
}

impl VerifiedWorldMetadata {
    pub(super) const fn cut(&self) -> &WorldCut {
        match self {
            Self::LegacyV1(metadata) => metadata.cut(),
            Self::SectionSealedV2(metadata) | Self::RecoverySealedV2(metadata) => metadata.cut(),
        }
    }

    /// Whether every non-metadata section is bound to this image.
    ///
    /// Both V2 recovery profiles satisfy this contract, so exact auxiliary
    /// generations remain admissible when reading the frozen section-only V2
    /// profile.
    pub(super) const fn recovery_image_is_sealed(&self) -> bool {
        matches!(self, Self::SectionSealedV2(_) | Self::RecoverySealedV2(_))
    }

    /// Whether the active header's recovery-critical coordinates are bound to
    /// the exact section image.
    ///
    /// Only this stronger profile may supply an authenticated WAL transaction
    /// replay floor. A section-only V2 image remains readable, but its raw
    /// header transaction ID is never trusted for WAL prefix discard.
    pub(super) const fn recovery_coordinates_are_sealed(&self) -> bool {
        matches!(self, Self::RecoverySealedV2(_))
    }
}

impl EncodedSection {
    pub(super) fn new(section_type: SectionType, version: u8, bytes: Vec<u8>) -> Result<Self> {
        if version == 0 {
            return Err(Error::Serialization(format!(
                "section {section_type:?} uses reserved wire version zero"
            )));
        }
        Ok(Self {
            section_type,
            version,
            bytes,
        })
    }

    pub(super) const fn section_type(&self) -> SectionType {
        self.section_type
    }

    pub(super) const fn version(&self) -> u8 {
        self.version
    }

    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Reads and CRC/decryption-validates every section before the engine mutates
/// any live store object.
pub(super) fn read_container_image(
    file_manager: &GrafeoFileManager,
    directory: &SectionDirectory,
) -> Result<Vec<EncodedSection>> {
    let mut seen = HashSet::with_capacity(directory.len());
    let mut sections = Vec::with_capacity(directory.len());
    for entry in directory.entries() {
        if !seen.insert(entry.section_type) {
            return Err(corruption(format!(
                "duplicate {:?} section in container directory",
                entry.section_type
            )));
        }
        if entry.version == 0 {
            return Err(corruption(format!(
                "section {:?} uses reserved wire version zero",
                entry.section_type
            )));
        }
        sections.push(EncodedSection::new(
            entry.section_type,
            entry.version,
            file_manager.read_section_data(entry).map_err(|error| {
                corruption(format!(
                    "failed to read checksummed {:?} section: {error}",
                    entry.section_type
                ))
            })?,
        )?);
    }
    Ok(sections)
}

/// Reads one active container image, validates every section checksum, and
/// verifies a present world manifest against the header coordinates. The
/// returned bytes are the exact bytes callers must deserialize; reopening
/// individual entries would weaken the single-image staging guarantee.
pub(super) fn read_verified_container_image(
    file_manager: &GrafeoFileManager,
    directory: &SectionDirectory,
) -> Result<(
    Vec<EncodedSection>,
    Option<VerifiedWorldMetadata>,
    GraphModelTag,
)> {
    let graph_model = GraphModelTag::from_u8(file_manager.graph_model_tag())
        .map_err(|error| corruption(format!("invalid container graph model: {error}")))?;
    let header = file_manager.active_header();
    let recovery_coordinates = RecoveryCoordinates::new(
        header.epoch,
        header.transaction_id,
        graph_model,
        header.node_count,
        header.edge_count,
    );
    let sections = read_container_image(file_manager, directory)?;
    validate_authoritative_section_set(&sections, graph_model)?;
    validate_section_contracts(&sections, graph_model)?;
    let metadata = verify_world_metadata(&sections, recovery_coordinates)?;
    let cut = metadata
        .as_ref()
        .ok_or_else(|| corruption("CDC section requires world authority"))?
        .cut();
    let cdc = find_section(&sections, SectionType::Cdc)
        .ok_or_else(|| corruption("missing CDC section"))?;
    super::cdc_checkpoint::validate(cdc.bytes(), cut.store_id(), cut.epoch())?;
    Ok((sections, metadata, graph_model))
}

/// Enforces the model-level section cardinality before any model decoder sees
/// staged bytes. Legacy containers may omit `WORLD_METADATA`; they never had
/// permission to omit the authoritative model planes themselves.
fn validate_authoritative_section_set(
    sections: &[EncodedSection],
    graph_model: GraphModelTag,
) -> Result<()> {
    let has_lpg = matches!(graph_model, GraphModelTag::Lpg | GraphModelTag::Both);
    let has_rdf = matches!(graph_model, GraphModelTag::Rdf | GraphModelTag::Both);

    for (section_type, expected) in [
        (SectionType::Catalog, 1),
        (SectionType::Cdc, 1),
        (SectionType::LpgStore, usize::from(has_lpg)),
        (SectionType::RdfStore, usize::from(has_rdf)),
    ] {
        let actual = sections
            .iter()
            .filter(|section| section.section_type == section_type)
            .count();
        if actual != expected {
            return Err(corruption(format!(
                "container graph model {graph_model:?} requires exactly {expected} {section_type:?} section(s), found {actual}"
            )));
        }
    }

    let has_compact_store = sections
        .iter()
        .any(|section| section.section_type == SectionType::CompactStore);
    let has_overlay_deletions = sections
        .iter()
        .any(|section| section.section_type == SectionType::OverlayDeletions);
    if has_overlay_deletions && !has_compact_store {
        return Err(corruption(
            "OverlayDeletions requires the CompactStore generation whose identities it masks",
        ));
    }
    Ok(())
}

/// Rejects an image whose directory claims a serializer this binary cannot
/// decode, a model plane outside the declared graph topology, or a supported
/// directory version that disagrees with the payload's own framing.
///
/// World manifests intentionally hash the directory version independently of
/// the bytes. Decoders must therefore enforce both halves of that contract;
/// accepting bytes labelled as some other version would let a self-consistent
/// manifest make a false exact-format claim.
fn validate_section_contracts(
    sections: &[EncodedSection],
    graph_model: GraphModelTag,
) -> Result<()> {
    let has_lpg = matches!(graph_model, GraphModelTag::Lpg | GraphModelTag::Both);
    let has_rdf = matches!(graph_model, GraphModelTag::Rdf | GraphModelTag::Both);

    for section in sections {
        let section_type = section.section_type;
        let version = section.version;
        let model_allowed = match section_type {
            SectionType::Catalog | SectionType::WorldMetadata | SectionType::Cdc => true,
            SectionType::LpgStore
            | SectionType::CompactStore
            | SectionType::OverlayDeletions
            | SectionType::VectorStore
            | SectionType::TextIndex
            | SectionType::PropertyIndex => has_lpg,
            SectionType::RdfStore | SectionType::RdfRing => has_rdf,
            _ => false,
        };
        if !model_allowed {
            return Err(corruption(format!(
                "section {section_type:?} is incompatible with container graph model {graph_model:?}"
            )));
        }

        let feature_available = match section_type {
            SectionType::LpgStore => cfg!(feature = "lpg"),
            SectionType::RdfStore => cfg!(feature = "triple-store"),
            SectionType::CompactStore | SectionType::OverlayDeletions => {
                cfg!(all(feature = "compact-store", feature = "lpg"))
            }
            SectionType::VectorStore => cfg!(all(feature = "vector-index", feature = "lpg")),
            SectionType::TextIndex => cfg!(all(feature = "text-index", feature = "lpg")),
            SectionType::RdfRing => cfg!(all(feature = "ring-index", feature = "triple-store")),
            // No persistent PropertyIndex section decoder exists. Logical
            // property indexes are rebuilt from the authoritative catalog.
            SectionType::PropertyIndex => false,
            SectionType::Catalog | SectionType::WorldMetadata | SectionType::Cdc => true,
            _ => false,
        };
        if !feature_available {
            return Err(corruption(format!(
                "container section {section_type:?} requires unavailable engine feature support"
            )));
        }

        let version_supported = match section_type {
            SectionType::Catalog if graph_model == GraphModelTag::Rdf => version == 2,
            SectionType::Catalog => version == 7,
            // LPG4 is the only current recursive exact LPG generation.
            SectionType::LpgStore => version == 4,
            SectionType::RdfStore => (1..=6).contains(&version),
            // GCST v9 carries the retained property-history floor.
            SectionType::CompactStore => version == 9,
            SectionType::OverlayDeletions => version == 2,
            SectionType::VectorStore => version == 4,
            SectionType::TextIndex => version == 5,
            SectionType::RdfRing => matches!(version, 1 | 2),
            SectionType::WorldMetadata => matches!(version, 1 | 2),
            SectionType::Cdc => version == 1,
            SectionType::PropertyIndex => false,
            _ => false,
        };
        if !version_supported {
            return Err(corruption(format!(
                "unsupported {section_type:?} section directory version {version}"
            )));
        }

        let payload_matches = match section_type {
            SectionType::Cdc => section.bytes.starts_with(super::cdc_checkpoint::MAGIC),
            SectionType::Catalog | SectionType::TextIndex => {
                section.bytes.first().copied() == Some(version)
            }
            SectionType::LpgStore if version == 4 => section.bytes.starts_with(b"LPG4"),
            SectionType::RdfStore | SectionType::CompactStore | SectionType::OverlayDeletions => {
                section.bytes.get(4).copied() == Some(version)
            }
            // Exact Vector v4 is the sole payload; its independent version
            // coordinate must agree with the directory before decoding.
            SectionType::VectorStore => {
                section.bytes.starts_with(b"GVST") && section.bytes.get(4).copied() == Some(version)
            }
            SectionType::RdfRing if version == 2 => section.bytes.starts_with(b"GRFR"),
            SectionType::RdfRing => !section.bytes.starts_with(b"GRFR"),
            SectionType::WorldMetadata if version == 1 => section.bytes.starts_with(b"WRLDCUT1"),
            SectionType::WorldMetadata if version == 2 => section.bytes.starts_with(b"WRLDCUT2"),
            SectionType::PropertyIndex => false,
            _ => false,
        };
        if !payload_matches {
            return Err(corruption(format!(
                "{section_type:?} directory version {version} disagrees with its payload framing"
            )));
        }
    }
    Ok(())
}

pub(super) fn find_section(
    sections: &[EncodedSection],
    section_type: SectionType,
) -> Option<&EncodedSection> {
    sections
        .iter()
        .find(|section| section.section_type == section_type)
}

/// Logical metadata captured under the same quiescent publication gate as the
/// section payloads.
#[derive(Clone)]
pub(super) struct WorldCutInputs {
    pub(super) identity: WorldIdentityMetadataV1,
    pub(super) epoch: EpochId,
    pub(super) graph_model: GraphModelTag,
    pub(super) schema: SchemaCut,
    pub(super) projections: Vec<ProjectionCut>,
}

/// Proves that the database-level identity and the authoritative RDF dataset
/// still describe the same logical store before sealing an image.
///
/// Component digests alone cannot catch a publisher that pairs internally
/// valid RDF bytes with a different `StoreId` in the manifest. Reopen would
/// reject that image, but the publisher must fail before writing it in the
/// first place.
#[cfg(feature = "triple-store")]
pub(super) fn validate_live_world_identity(
    declared: WorldIdentityMetadataV1,
    graph_model: GraphModelTag,
    rdf_store: &grafeo_core::graph::rdf::RdfStore,
) -> Result<WorldIdentityMetadataV1> {
    if !matches!(graph_model, GraphModelTag::Rdf | GraphModelTag::Both) {
        return Ok(declared);
    }

    let rdf_identity =
        WorldIdentityMetadataV1::new(rdf_store.store_id(), rdf_store.history_completeness())
            .map_err(|error| corruption(format!("invalid live RDF world identity: {error}")))?;
    if rdf_identity != declared {
        return Err(corruption(format!(
            "database world identity {} ({:?}) does not match authoritative RDF identity {} ({:?})",
            declared.store_id(),
            declared.history(),
            rdf_identity.store_id(),
            rdf_identity.history()
        )));
    }
    Ok(declared)
}

/// Captures the logical coordinates that accompany one physical container
/// image. Callers hold the publication barrier (and the RDF commit gate when
/// enabled), so the catalog, projection receipts, epoch, and serialized
/// section payloads all describe the same committed cut.
pub(super) fn capture_world_cut_inputs(
    identity: WorldIdentityMetadataV1,
    epoch: EpochId,
    graph_model: GraphModelTag,
    catalog: &crate::catalog::Catalog,
    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    projections: &grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
) -> Result<WorldCutInputs> {
    let catalog_state = catalog.encode_current_state_v2().map_err(|error| {
        Error::Serialization(format!(
            "capture canonical catalog state for world cut: {error}"
        ))
    })?;
    let schema =
        SchemaCut::from_canonical_post_image(2, &catalog_state).map_err(world_cut_error)?;

    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    let projections = projections
        .snapshot()
        .into_iter()
        // Only published V3 receipts identify a source graph incarnation.
        // Pending declarations remain in the RDF section and are omitted from
        // the verified world-cut generations.
        .filter_map(|definition| definition.receipt().cloned())
        .map(|receipt| receipt.to_world_cut().map_err(world_cut_error))
        .collect::<Result<Vec<_>>>()?;
    #[cfg(not(all(feature = "triple-store", feature = "lpg")))]
    let projections = Vec::new();

    Ok(WorldCutInputs {
        identity,
        epoch,
        graph_model,
        schema,
        projections,
    })
}

/// Serializes a complete set of section wrappers exactly once.
pub(super) fn encode_sections(
    sections: &[&dyn Section],
    captured_lpg: Option<EncodedSection>,
) -> Result<Vec<EncodedSection>> {
    let mut seen = HashSet::with_capacity(sections.len());
    let mut encoded = Vec::with_capacity(sections.len() + 1);
    if let Some(lpg) = captured_lpg {
        if lpg.section_type != SectionType::LpgStore || lpg.version != 4 {
            return Err(Error::Serialization(
                "invalid preencoded LPG capture".into(),
            ));
        }
        seen.insert(lpg.section_type);
        encoded.push(lpg);
    }
    for section in sections {
        let section_type = section.section_type();
        if section_type == SectionType::WorldMetadata {
            return Err(Error::Serialization(
                "WorldMetadata is generated by the container publisher".to_string(),
            ));
        }
        if !seen.insert(section_type) {
            return Err(Error::Serialization(format!(
                "duplicate {section_type:?} section in one container image"
            )));
        }
        encoded.push(EncodedSection::new(
            section_type,
            section.version(),
            section.serialize()?,
        )?);
    }
    Ok(encoded)
}

/// Seals the logical model cut and complete recovery image, then appends the
/// current `WorldMetadata` section.
///
/// The logical [`WorldCut`] remains independent of acceleration layout. A
/// separate v2 digest binds every non-metadata section byte and version plus
/// the recovery-critical header coordinates, so neither a section nor a valid
/// foreign header can be spliced across checkpoints.
pub(super) fn append_world_metadata(
    sections: &mut Vec<EncodedSection>,
    inputs: WorldCutInputs,
    recovery_coordinates: RecoveryCoordinates,
) -> Result<grafeo_common::types::WorldCut> {
    if inputs.epoch.as_u64() != recovery_coordinates.epoch() {
        return Err(Error::Serialization(format!(
            "WorldMetadata epoch {} does not match publication header epoch {}",
            inputs.epoch.as_u64(),
            recovery_coordinates.epoch()
        )));
    }
    if inputs.graph_model != recovery_coordinates.graph_model() {
        return Err(Error::Serialization(format!(
            "WorldMetadata graph model {:?} does not match publication graph model {:?}",
            inputs.graph_model,
            recovery_coordinates.graph_model()
        )));
    }
    if sections
        .iter()
        .any(|section| section.section_type == SectionType::WorldMetadata)
    {
        return Err(Error::Serialization(
            "container image already contains WorldMetadata".to_string(),
        ));
    }

    let cut = seal_logical_world_cut(sections, inputs)?;
    let recovery_components = recovery_image_components(sections, &recovery_coordinates)?;
    let metadata =
        WorldMetadataSectionV2::seal(cut, &recovery_components).map_err(world_cut_error)?;
    let cut = metadata.cut().clone();
    let metadata = metadata.encode().map_err(world_cut_error)?;
    sections.push(EncodedSection::new(
        SectionType::WorldMetadata,
        WorldMetadataSectionV2::SECTION_VERSION,
        metadata,
    )?);
    Ok(cut)
}

/// Seals only the portable logical cut represented by the supplied sections.
///
/// This helper is intentionally independent of a checkpoint header. Callers
/// that publish a container must use [`append_world_metadata`] with the real
/// final recovery coordinates; APIs that only inspect logical identity should
/// not fabricate physical coordinates merely to obtain a [`WorldCut`].
pub(super) fn seal_logical_world_cut(
    sections: &[EncodedSection],
    inputs: WorldCutInputs,
) -> Result<WorldCut> {
    let components = authoritative_components(sections)?;
    let formats = components.iter().map(|(format, _)| *format).collect();
    let descriptor = WorldCutDescriptor::new(
        inputs.identity.store_id(),
        inputs.epoch,
        inputs.graph_model,
        formats,
        inputs.schema,
        inputs.projections,
        inputs.identity.history(),
    )
    .map_err(world_cut_error)?;
    WorldCut::seal_components(descriptor, &components).map_err(world_cut_error)
}

/// Verifies a present metadata section before any component is installed into
/// live engine state. Absence is returned as an explicit legacy-container case.
pub(super) fn verify_world_metadata(
    sections: &[EncodedSection],
    recovery_coordinates: RecoveryCoordinates,
) -> Result<Option<VerifiedWorldMetadata>> {
    let mut metadata = sections
        .iter()
        .filter(|section| section.section_type == SectionType::WorldMetadata);
    let Some(section) = metadata.next() else {
        if sections.iter().any(|section| {
            matches!(
                (section.section_type, section.version),
                (SectionType::Catalog, 7)
                    | (SectionType::Cdc, 1)
                    | (SectionType::LpgStore, 4)
                    | (SectionType::RdfStore, 6)
                    | (SectionType::TextIndex, 5)
                    | (SectionType::VectorStore, 4)
            )
        }) {
            return Err(corruption(
                "container uses an exact model generation that requires WorldMetadata",
            ));
        }
        return Ok(None);
    };
    if metadata.next().is_some() {
        return Err(corruption(
            "container contains duplicate WorldMetadata sections",
        ));
    }
    let components = authoritative_components(sections)?;
    let metadata = match section.version {
        WorldMetadataSectionV1::SECTION_VERSION => {
            let has_graph_exact_catalog = sections.iter().any(|section| {
                section.section_type == SectionType::Catalog && section.version == 7
            });
            let has_vector_v4 = sections.iter().any(|section| {
                section.section_type == SectionType::VectorStore && section.version == 4
            });
            let has_text_v5 = sections.iter().any(|section| {
                section.section_type == SectionType::TextIndex && section.version == 5
            });
            if has_text_v5 || has_graph_exact_catalog || has_vector_v4 {
                return Err(corruption(
                    "graph-exact auxiliary recovery requires WorldMetadata v2",
                ));
            }
            let metadata =
                WorldMetadataSectionV1::decode(&section.bytes).map_err(world_cut_error)?;
            metadata
                .verify_components(&components)
                .map_err(world_cut_error)?;
            VerifiedWorldMetadata::LegacyV1(metadata)
        }
        WorldMetadataSectionV2::SECTION_VERSION => {
            let metadata =
                WorldMetadataSectionV2::decode(&section.bytes).map_err(world_cut_error)?;
            metadata
                .cut()
                .verify_components(&components)
                .map_err(world_cut_error)?;
            let recovery_components = recovery_image_components(sections, &recovery_coordinates)?;
            match metadata.verify_recovery_components(&recovery_components) {
                Ok(()) => VerifiedWorldMetadata::RecoverySealedV2(metadata),
                Err(
                    coordinate_profile_error @ WorldCutError::RecoveryImageDigestMismatch { .. },
                ) => {
                    // V2 originally sealed exactly the non-metadata section
                    // inventory. Header coordinates were later added as a
                    // reserved synthetic component without changing the V2
                    // wire payload. Recognize that frozen profile by its exact
                    // digest rather than making already-published containers
                    // unreadable. It retains full section/auxiliary integrity,
                    // but callers must not trust header-only replay metadata.
                    let section_components = recovery_section_components(sections)?;
                    if metadata
                        .verify_recovery_components(&section_components)
                        .is_ok()
                    {
                        VerifiedWorldMetadata::SectionSealedV2(metadata)
                    } else {
                        return Err(world_cut_error(coordinate_profile_error));
                    }
                }
                Err(error) => return Err(world_cut_error(error)),
            }
        }
        version => {
            return Err(corruption(format!(
                "unsupported WorldMetadata section version {version}"
            )));
        }
    };
    if metadata.cut().epoch().as_u64() != recovery_coordinates.epoch() {
        return Err(corruption(format!(
            "WorldMetadata epoch {} does not match container header epoch {}",
            metadata.cut().epoch().as_u64(),
            recovery_coordinates.epoch()
        )));
    }
    if metadata.cut().descriptor().graph_model() != recovery_coordinates.graph_model() {
        return Err(corruption(format!(
            "WorldMetadata graph model {:?} does not match container graph model {:?}",
            metadata.cut().descriptor().graph_model(),
            recovery_coordinates.graph_model()
        )));
    }
    Ok(Some(metadata))
}

/// Cross-checks the logical metadata against the state reconstructed by the
/// model decoders. Component integrity verification alone proves the bytes are the
/// bytes named by the manifest; this second check proves the manifest did not
/// merely notarize a false StoreId, schema digest, or projection receipt set.
pub(super) fn verify_loaded_state(
    metadata: &VerifiedWorldMetadata,
    identity: WorldIdentityMetadataV1,
    catalog: &crate::catalog::Catalog,
    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    projections: &grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
) -> Result<()> {
    let descriptor = metadata.cut().descriptor();
    if descriptor.store_id() != identity.store_id() {
        return Err(corruption(format!(
            "WorldMetadata store {} does not match decoded store {}",
            descriptor.store_id(),
            identity.store_id()
        )));
    }
    if descriptor.history() != identity.history() {
        return Err(corruption(
            "WorldMetadata RDF-history provenance does not match decoded state",
        ));
    }

    let actual = capture_world_cut_inputs(
        identity,
        descriptor.epoch(),
        descriptor.graph_model(),
        catalog,
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        projections,
    )?;
    if descriptor.schema() != &actual.schema {
        return Err(corruption(
            "WorldMetadata schema digest does not match decoded catalog state",
        ));
    }
    if descriptor.projections() != actual.projections.as_slice() {
        return Err(corruption(
            "WorldMetadata projection receipts do not match decoded registry state",
        ));
    }
    Ok(())
}

fn authoritative_components(
    sections: &[EncodedSection],
) -> Result<Vec<(ModelFormatVersion, &[u8])>> {
    let mut components = Vec::with_capacity(sections.len() + 1);
    for section in sections {
        let version = u16::from(section.version);
        let push = |components: &mut Vec<_>, format| -> Result<()> {
            components.push((
                ModelFormatVersion::new(format, version).map_err(world_cut_error)?,
                section.bytes.as_slice(),
            ));
            Ok(())
        };
        match section.section_type {
            SectionType::Catalog => push(&mut components, AuthoritativeFormat::Catalog)?,
            SectionType::Cdc => push(&mut components, AuthoritativeFormat::Cdc)?,
            SectionType::LpgStore => push(&mut components, AuthoritativeFormat::Lpg)?,
            SectionType::RdfStore => {
                #[cfg(feature = "triple-store")]
                {
                    push(&mut components, AuthoritativeFormat::Rdf)?;
                    // The dataset history has an independent canonical grammar
                    // nested inside the outer RDF section. Projection addenda and
                    // outer framing remain covered by `Rdf`, while `RdfHistory`
                    // names only the exact payload and its own format version.
                    let (history_version, history_bytes) =
                        grafeo_core::graph::rdf::section::canonical_history_component(
                            section.bytes.as_slice(),
                        )
                        .map_err(|error| {
                            corruption(format!(
                                "extract canonical RDF history component for world manifest: {error}"
                            ))
                        })?;
                    components.push((
                        ModelFormatVersion::new(AuthoritativeFormat::RdfHistory, history_version)
                            .map_err(world_cut_error)?,
                        history_bytes,
                    ));
                }
                #[cfg(not(feature = "triple-store"))]
                return Err(corruption(
                    "RDF section cannot enter a world manifest without RDF decoder support",
                ));
            }
            SectionType::CompactStore => push(&mut components, AuthoritativeFormat::Compact)?,
            SectionType::OverlayDeletions => {
                push(&mut components, AuthoritativeFormat::OverlayDeletions)?;
            }
            SectionType::WorldMetadata
            | SectionType::VectorStore
            | SectionType::TextIndex
            | SectionType::RdfRing
            | SectionType::PropertyIndex => {}
            other => {
                return Err(Error::Serialization(format!(
                    "unsupported section type {other:?} in world manifest"
                )));
            }
        }
    }
    Ok(components)
}

/// Canonical complete physical image sealed by WorldMetadata v2.
///
/// Metadata excludes itself to avoid a recursive digest. Every other section
/// participates, including acceleration generations that remain deliberately
/// absent from the logical WorldCut. The reserved synthetic coordinate
/// component also binds the active header values used by recovery.
fn recovery_section_components(
    sections: &[EncodedSection],
) -> Result<Vec<RecoveryImageComponent<'_>>> {
    sections
        .iter()
        .filter(|section| section.section_type != SectionType::WorldMetadata)
        .map(|section| {
            RecoveryImageComponent::new(
                section.section_type as u32,
                u16::from(section.version),
                section.bytes.as_slice(),
            )
            .map_err(world_cut_error)
        })
        .collect()
}

fn recovery_image_components<'a>(
    sections: &'a [EncodedSection],
    recovery_coordinates: &'a RecoveryCoordinates,
) -> Result<Vec<RecoveryImageComponent<'a>>> {
    let mut components = recovery_section_components(sections)?;
    components.push(recovery_coordinates.component());
    Ok(components)
}

fn world_cut_error(error: impl std::fmt::Display) -> Error {
    corruption(format!("invalid world metadata: {error}"))
}

fn corruption(message: impl Into<String>) -> Error {
    Error::Storage(StorageError::Corruption(message.into()))
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{Digest256, HistoryCompleteness, StoreId};

    use super::*;

    fn store_id(byte: u8) -> StoreId {
        StoreId::from_bytes([byte; StoreId::LEN]).unwrap()
    }

    fn schema() -> SchemaCut {
        SchemaCut::new(2, Digest256::schema(b"catalog-state")).unwrap()
    }

    fn inputs(model: GraphModelTag) -> WorldCutInputs {
        WorldCutInputs {
            identity: WorldIdentityMetadataV1::new(store_id(7), HistoryCompleteness::Complete)
                .unwrap(),
            epoch: EpochId::new(11),
            graph_model: model,
            schema: schema(),
            projections: Vec::new(),
        }
    }

    fn coordinates(model: GraphModelTag) -> RecoveryCoordinates {
        RecoveryCoordinates::new(11, 13, model, 17, 19)
    }

    #[test]
    fn recovery_coordinates_use_the_frozen_domain_separated_grammar() {
        let coordinates = RecoveryCoordinates::new(
            0x0102_0304_0506_0708,
            0x1112_1314_1516_1718,
            GraphModelTag::Both,
            0x2122_2324_2526_2728,
            0x3132_3334_3536_3738,
        );
        let mut expected = b"grafeo:recovery-coordinates:v1\0".to_vec();
        expected.extend_from_slice(&[
            2, // GraphModelTag::Both
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13,
            0x12, 0x11, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x38, 0x37, 0x36, 0x35,
            0x34, 0x33, 0x32, 0x31,
        ]);

        assert_eq!(coordinates.as_bytes().as_slice(), expected);
        assert_eq!(coordinates.component().section_type(), u32::MAX);
        assert_eq!(coordinates.component().directory_version(), 1);
        assert_eq!(coordinates.component().bytes(), expected);
    }

    fn capture_inputs(
        identity: WorldIdentityMetadataV1,
        epoch: EpochId,
        model: GraphModelTag,
        catalog: &crate::catalog::Catalog,
    ) -> WorldCutInputs {
        capture_world_cut_inputs(
            identity,
            epoch,
            model,
            catalog,
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            &grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new(),
        )
        .unwrap()
    }

    fn verify_decoded_state(
        metadata: &VerifiedWorldMetadata,
        identity: WorldIdentityMetadataV1,
        catalog: &crate::catalog::Catalog,
    ) -> Result<()> {
        verify_loaded_state(
            metadata,
            identity,
            catalog,
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            &grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new(),
        )
    }

    fn section(section_type: SectionType, version: u8, bytes: &[u8]) -> EncodedSection {
        EncodedSection::new(section_type, version, bytes.to_vec()).unwrap()
    }

    fn append_legacy_world_metadata_v1(
        sections: &mut Vec<EncodedSection>,
        inputs: WorldCutInputs,
    ) -> WorldCut {
        let components = authoritative_components(sections).unwrap();
        let formats = components.iter().map(|(format, _)| *format).collect();
        let descriptor = WorldCutDescriptor::new(
            inputs.identity.store_id(),
            inputs.epoch,
            inputs.graph_model,
            formats,
            inputs.schema,
            inputs.projections,
            inputs.identity.history(),
        )
        .unwrap();
        let metadata = WorldMetadataSectionV1::seal(descriptor, &components).unwrap();
        let cut = metadata.cut().clone();
        sections.push(
            EncodedSection::new(
                SectionType::WorldMetadata,
                WorldMetadataSectionV1::SECTION_VERSION,
                metadata.encode().unwrap(),
            )
            .unwrap(),
        );
        cut
    }

    /// Reproduces the frozen first V2 recovery profile, whose digest covered
    /// every physical section but no synthetic header-coordinate component.
    fn append_section_sealed_world_metadata_v2(
        sections: &mut Vec<EncodedSection>,
        inputs: WorldCutInputs,
    ) -> WorldCut {
        let cut = seal_logical_world_cut(sections, inputs).unwrap();
        let recovery_components = recovery_section_components(sections).unwrap();
        let metadata = WorldMetadataSectionV2::seal(cut, &recovery_components).unwrap();
        let cut = metadata.cut().clone();
        sections.push(
            EncodedSection::new(
                SectionType::WorldMetadata,
                WorldMetadataSectionV2::SECTION_VERSION,
                metadata.encode().unwrap(),
            )
            .unwrap(),
        );
        cut
    }

    fn authoritative_set(model: GraphModelTag) -> Vec<EncodedSection> {
        let catalog_version = if model == GraphModelTag::Rdf { 2 } else { 7 };
        let mut sections = vec![section(
            SectionType::Catalog,
            catalog_version,
            &[catalog_version],
        )];
        sections.push(section(
            SectionType::Cdc,
            1,
            super::super::cdc_checkpoint::MAGIC,
        ));
        if matches!(model, GraphModelTag::Lpg | GraphModelTag::Both) {
            sections.push(section(SectionType::LpgStore, 4, b"LPG4"));
        }
        if matches!(model, GraphModelTag::Rdf | GraphModelTag::Both) {
            sections.push(section(SectionType::RdfStore, 5, b"GRDF\x05"));
        }
        sections
    }

    #[test]
    fn model_section_cardinality_rejects_every_missing_or_foreign_authoritative_plane() {
        for model in [GraphModelTag::Lpg, GraphModelTag::Rdf, GraphModelTag::Both] {
            let mut required = match model {
                GraphModelTag::Lpg => vec![SectionType::Catalog, SectionType::LpgStore],
                GraphModelTag::Rdf => vec![SectionType::Catalog, SectionType::RdfStore],
                GraphModelTag::Both => vec![
                    SectionType::Catalog,
                    SectionType::LpgStore,
                    SectionType::RdfStore,
                ],
            };
            required.push(SectionType::Cdc);
            validate_authoritative_section_set(&authoritative_set(model), model)
                .expect("complete authoritative model set");

            for missing in required {
                let mut incomplete = authoritative_set(model);
                incomplete.retain(|section| section.section_type != missing);
                let error = validate_authoritative_section_set(&incomplete, model)
                    .expect_err("an authoritative plane must never be optional");
                assert!(error.to_string().contains("requires exactly 1"), "{error}");
                assert!(
                    error.to_string().contains(&format!("{missing:?}")),
                    "{error}"
                );
            }
        }

        let mut lpg_with_foreign_rdf = authoritative_set(GraphModelTag::Lpg);
        lpg_with_foreign_rdf.push(section(SectionType::RdfStore, 5, b"GRDF\x05"));
        let error = validate_authoritative_section_set(&lpg_with_foreign_rdf, GraphModelTag::Lpg)
            .expect_err("an undeclared authoritative model plane must be rejected");
        assert!(error.to_string().contains("requires exactly 0"), "{error}");
        assert!(error.to_string().contains("RdfStore"), "{error}");
    }

    #[test]
    fn overlay_deletions_cannot_exist_without_their_compact_generation() {
        let mut orphaned = authoritative_set(GraphModelTag::Lpg);
        orphaned.push(section(SectionType::OverlayDeletions, 2, b"GDEL\x02"));

        let error = validate_authoritative_section_set(&orphaned, GraphModelTag::Lpg)
            .expect_err("a deletion mask without its compact identity space is corruption");
        assert!(error.to_string().contains("OverlayDeletions"), "{error}");
        assert!(error.to_string().contains("CompactStore"), "{error}");

        orphaned.push(section(SectionType::CompactStore, 9, b"GCST\x09"));
        validate_authoritative_section_set(&orphaned, GraphModelTag::Lpg)
            .expect("the deletion mask is meaningful with its compact generation");
    }

    #[test]
    fn exact_model_generations_cannot_drop_world_metadata_into_legacy_mode() {
        for (model, exact) in [
            (
                GraphModelTag::Lpg,
                vec![
                    section(SectionType::Catalog, 7, &[7]),
                    section(SectionType::LpgStore, 4, b"LPG4"),
                ],
            ),
            (
                GraphModelTag::Rdf,
                vec![
                    section(SectionType::Catalog, 2, b"\x02catalog"),
                    section(SectionType::RdfStore, 6, b"GRDF\x06"),
                ],
            ),
            (
                GraphModelTag::Lpg,
                vec![
                    section(SectionType::Catalog, 7, &[7]),
                    section(SectionType::LpgStore, 4, b"LPG4"),
                    section(SectionType::TextIndex, 5, &[5]),
                ],
            ),
        ] {
            let error = verify_world_metadata(&exact, RecoveryCoordinates::new(0, 0, model, 0, 0))
                .expect_err("an exact generation must carry its world identity manifest");
            assert!(
                error.to_string().contains("requires WorldMetadata"),
                "{error}"
            );
        }

        // Isolate LPG4's own manifest requirement from Catalog/auxiliary checks.
        let current_lpg = vec![section(SectionType::LpgStore, 4, b"LPG4")];
        let error = verify_world_metadata(
            &current_lpg,
            RecoveryCoordinates::new(0, 0, GraphModelTag::Lpg, 0, 0),
        );
        assert!(matches!(
            error,
            Err(Error::Storage(StorageError::Corruption(message)))
                if message.contains("requires WorldMetadata")
        ));
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn publisher_rejects_a_live_rdf_identity_divergence_before_sealing() {
        use grafeo_core::graph::rdf::{RdfStore, RdfStoreConfig};

        let rdf = RdfStore::with_config_and_store_id(RdfStoreConfig::default(), store_id(7));
        let matching =
            WorldIdentityMetadataV1::new(store_id(7), HistoryCompleteness::Complete).unwrap();
        assert_eq!(
            validate_live_world_identity(matching.clone(), GraphModelTag::Rdf, &rdf).unwrap(),
            matching
        );

        let foreign =
            WorldIdentityMetadataV1::new(store_id(8), HistoryCompleteness::Complete).unwrap();
        let error = validate_live_world_identity(foreign, GraphModelTag::Both, &rdf)
            .expect_err("a publisher must not notarize a foreign StoreId");
        assert!(error.to_string().contains("does not match"), "{error}");
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn section_contract_rejects_supported_directory_labels_with_wrong_framing() {
        for malformed in [
            section(SectionType::Catalog, 7, &[5]),
            section(SectionType::Catalog, 7, &[6]),
            section(SectionType::LpgStore, 4, b"LPG3"),
            section(SectionType::LpgStore, 4, b"LPGB"),
        ] {
            let error = validate_section_contracts(&[malformed], GraphModelTag::Lpg)
                .expect_err("a supported directory label must not override payload framing");
            assert!(
                error.to_string().contains("disagrees"),
                "unexpected structured error: {error}"
            );
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn lpg_section_contract_rejects_every_predecessor_before_payload_inspection() {
        assert!(
            validate_section_contracts(
                &[section(SectionType::LpgStore, 4, b"LPG4")],
                GraphModelTag::Lpg,
            )
            .is_ok()
        );
        for version in [1, 2, 3, 5, 255] {
            // Even current magic cannot rescue an unsupported outer version.
            let result = validate_section_contracts(
                &[section(SectionType::LpgStore, version, b"LPG4")],
                GraphModelTag::Lpg,
            );
            assert!(matches!(
                result,
                Err(Error::Storage(StorageError::Corruption(message)))
                    if message.contains("unsupported LpgStore section directory version")
            ));
        }
    }

    #[cfg(all(feature = "lpg", feature = "vector-index", feature = "text-index"))]
    #[test]
    fn section_contract_admits_graph_exact_index_generations() {
        let exact = [
            section(SectionType::Catalog, 7, &[7]),
            section(SectionType::LpgStore, 4, b"LPG4"),
            section(SectionType::VectorStore, 4, b"GVST\x04"),
            section(SectionType::TextIndex, 5, &[5]),
        ];

        validate_section_contracts(&exact, GraphModelTag::Lpg)
            .expect("current graph-exact directory generations must be admitted");
    }

    #[cfg(all(feature = "lpg", feature = "text-index"))]
    #[test]
    fn section_contract_rejects_predecessor_catalog_and_text_generations() {
        for (kind, current) in [(SectionType::Catalog, 7), (SectionType::TextIndex, 5)] {
            assert!(
                validate_section_contracts(
                    &[section(kind, current, &[current])],
                    GraphModelTag::Lpg,
                )
                .is_ok()
            );
            assert!(matches!(
                validate_section_contracts(
                    &[section(kind, current, &[current - 1])], GraphModelTag::Lpg,
                ),
                Err(Error::Storage(StorageError::Corruption(message)))
                    if message.contains("disagrees")
            ));
            for version in 1..current {
                let result = validate_section_contracts(
                    &[section(kind, version, &[version])],
                    GraphModelTag::Lpg,
                );
                assert!(matches!(
                    result,
                    Err(Error::Storage(StorageError::Corruption(message)))
                        if message.contains("unsupported")
                ));
            }
        }
    }

    #[cfg(all(feature = "lpg", feature = "vector-index"))]
    #[test]
    fn vector_section_contract_requires_current_directory_and_payload() {
        validate_section_contracts(
            &[section(SectionType::VectorStore, 4, b"GVST\x04")],
            GraphModelTag::Lpg,
        )
        .expect("current vector directory and payload framing");
        assert!(
            EncodedSection::new(SectionType::VectorStore, 0, b"hostile payload".to_vec()).is_err()
        );
        for version in [1, 2, 3, 5, 255] {
            let error = validate_section_contracts(
                &[section(
                    SectionType::VectorStore,
                    version,
                    b"hostile payload",
                )],
                GraphModelTag::Lpg,
            )
            .expect_err("unsupported directory version precedes payload inspection");
            assert!(error.to_string().contains("unsupported"), "{error}");
        }

        for mismatched in [
            section(SectionType::VectorStore, 4, b"legacy-bincode"),
            section(SectionType::VectorStore, 4, b"GVST\x01"),
            section(SectionType::VectorStore, 4, b"GVST\x02"),
            section(SectionType::VectorStore, 4, b"GVST\x03"),
        ] {
            let error = validate_section_contracts(&[mismatched], GraphModelTag::Lpg)
                .expect_err("a directory label must not override vector payload framing");
            assert!(error.to_string().contains("disagrees"), "{error}");
        }
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn rdf_section_contract_rejects_supported_directory_label_with_wrong_framing() {
        let malformed = section(SectionType::RdfStore, 5, &[b'G', b'R', b'D', b'F', 6]);
        let error = validate_section_contracts(&[malformed], GraphModelTag::Rdf)
            .expect_err("RDF directory and payload versions must agree");
        assert!(error.to_string().contains("disagrees"), "{error}");
    }

    #[test]
    fn rdf_catalog_state_requires_the_current_directory_generation() {
        assert!(
            validate_section_contracts(
                &[section(SectionType::Catalog, 2, b"\x02state")],
                GraphModelTag::Rdf,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_section_contracts(
                &[section(SectionType::Catalog, 1, b"state")], GraphModelTag::Rdf,
            ),
            Err(Error::Storage(StorageError::Corruption(message)))
                if message.contains("unsupported Catalog section directory version")
        ));
    }

    #[test]
    fn section_contract_rejects_data_from_a_model_plane_outside_the_header() {
        let rdf = section(SectionType::RdfStore, 6, &[b'G', b'R', b'D', b'F', 6]);
        let error = validate_section_contracts(&[rdf], GraphModelTag::Lpg)
            .expect_err("an LPG header must not carry authoritative RDF state");
        assert!(error.to_string().contains("incompatible"), "{error}");
    }

    #[cfg(not(feature = "lpg"))]
    #[test]
    fn section_contract_rejects_known_model_section_without_decoder_feature() {
        let lpg = section(SectionType::LpgStore, 4, b"LPG4");
        let error = validate_section_contracts(&[lpg], GraphModelTag::Lpg)
            .expect_err("a known section must not be silently dropped without its decoder");
        assert!(error.to_string().contains("feature support"), "{error}");
    }

    #[test]
    fn seals_and_verifies_exact_authoritative_versions_and_bytes() {
        let mut sections = vec![
            section(SectionType::Catalog, 7, b"\x07catalog"),
            section(SectionType::LpgStore, 4, b"LPG4"),
            section(SectionType::VectorStore, 4, b"exact-vector-image"),
        ];
        append_world_metadata(
            &mut sections,
            inputs(GraphModelTag::Lpg),
            coordinates(GraphModelTag::Lpg),
        )
        .unwrap();

        let metadata = verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg))
            .unwrap()
            .unwrap();
        assert_eq!(metadata.cut().store_id(), store_id(7));

        sections
            .iter_mut()
            .find(|section| section.section_type == SectionType::LpgStore)
            .unwrap()
            .bytes[0] ^= 0x80;
        assert!(verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg)).is_err());
    }

    #[test]
    fn world_metadata_v1_rejects_graph_exact_auxiliary_images() {
        for exact in [
            section(SectionType::VectorStore, 4, b"GVST\x04exact"),
            section(SectionType::TextIndex, 5, b"\x05exact"),
        ] {
            let mut sections = vec![
                section(SectionType::Catalog, 7, &[7]),
                section(SectionType::LpgStore, 4, b"LPG4"),
                exact,
            ];
            append_legacy_world_metadata_v1(&mut sections, inputs(GraphModelTag::Lpg));
            let error = verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg))
                .expect_err("an exact auxiliary image must never install under the v1 seal");
            assert!(
                error.to_string().contains("requires WorldMetadata v2"),
                "{error}"
            );
        }
    }

    #[test]
    fn world_metadata_v2_admits_and_binds_graph_exact_aux_images() {
        let mut sections = vec![
            section(SectionType::Catalog, 7, &[7]),
            section(SectionType::LpgStore, 4, b"LPG4"),
            section(SectionType::VectorStore, 4, b"GVST\x04exact"),
            section(SectionType::TextIndex, 5, b"\x05exact"),
        ];
        append_world_metadata(
            &mut sections,
            inputs(GraphModelTag::Lpg),
            coordinates(GraphModelTag::Lpg),
        )
        .unwrap();
        assert!(matches!(
            verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg)).unwrap(),
            Some(VerifiedWorldMetadata::RecoverySealedV2(_))
        ));
    }

    #[test]
    fn section_only_v2_remains_readable_without_authenticating_header_coordinates() {
        let mut sections = vec![
            section(SectionType::Catalog, 7, &[7]),
            section(SectionType::LpgStore, 4, b"LPG4"),
            section(SectionType::VectorStore, 4, b"GVST\x04exact"),
            section(SectionType::TextIndex, 5, b"\x05exact"),
        ];
        append_section_sealed_world_metadata_v2(&mut sections, inputs(GraphModelTag::Lpg));

        for header_coordinates in [
            coordinates(GraphModelTag::Lpg),
            RecoveryCoordinates::new(11, 9_999_999, GraphModelTag::Lpg, 123, 456),
        ] {
            let metadata = verify_world_metadata(&sections, header_coordinates)
                .unwrap()
                .expect("section-only v2 metadata");
            assert!(matches!(
                &metadata,
                VerifiedWorldMetadata::SectionSealedV2(_)
            ));
            assert!(metadata.recovery_image_is_sealed());
            assert!(!metadata.recovery_coordinates_are_sealed());
        }

        sections
            .iter_mut()
            .find(|section| section.section_type == SectionType::VectorStore)
            .unwrap()
            .bytes[5] ^= 0x80;
        assert!(verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg)).is_err());
    }

    #[test]
    fn recovery_coordinates_change_only_the_v2_physical_seal() {
        let seal = |coordinates| {
            let mut sections = vec![
                section(SectionType::Catalog, 7, b"\x07catalog"),
                section(SectionType::LpgStore, 4, b"LPG4"),
            ];
            let logical_cut =
                append_world_metadata(&mut sections, inputs(GraphModelTag::Lpg), coordinates)
                    .unwrap();
            let recovery_digest = match verify_world_metadata(&sections, coordinates)
                .unwrap()
                .unwrap()
            {
                VerifiedWorldMetadata::RecoverySealedV2(metadata) => {
                    metadata.recovery_image_digest()
                }
                VerifiedWorldMetadata::SectionSealedV2(_) => {
                    panic!("new publisher emitted the frozen section-only v2 profile")
                }
                VerifiedWorldMetadata::LegacyV1(_) => panic!("new publisher emitted v1"),
            };
            (sections, logical_cut, recovery_digest)
        };

        let original = coordinates(GraphModelTag::Lpg);
        let changed_transaction = RecoveryCoordinates::new(11, 14, GraphModelTag::Lpg, 17, 19);
        let (sections, original_cut, original_digest) = seal(original);
        let (_, changed_cut, changed_digest) = seal(changed_transaction);

        assert_eq!(
            original_cut, changed_cut,
            "physical checkpoint coordinates must not redefine portable WorldCut identity"
        );
        assert_ne!(original_digest, changed_digest);

        for foreign in [
            changed_transaction,
            RecoveryCoordinates::new(11, 13, GraphModelTag::Lpg, 18, 19),
            RecoveryCoordinates::new(11, 13, GraphModelTag::Lpg, 17, 20),
        ] {
            let error = verify_world_metadata(&sections, foreign)
                .expect_err("every recovery-critical header coordinate must be sealed");
            assert!(
                error.to_string().contains("recovery image digest"),
                "{error}"
            );
        }
    }

    #[test]
    fn current_catalog_rejects_v1_instead_of_inventing_header_authority() {
        let mut sections = vec![
            section(SectionType::Catalog, 7, b"\x07catalog"),
            section(SectionType::LpgStore, 4, b"LPG4"),
        ];
        append_legacy_world_metadata_v1(&mut sections, inputs(GraphModelTag::Lpg));

        let foreign_physical_coordinates =
            RecoveryCoordinates::new(11, u64::MAX, GraphModelTag::Lpg, 123, 456);
        assert!(matches!(
            verify_world_metadata(&sections, foreign_physical_coordinates),
            Err(Error::Storage(StorageError::Corruption(message)))
                if message.contains("requires WorldMetadata v2")
        ));
    }

    #[test]
    fn publisher_rejects_incoherent_logical_and_header_coordinates_before_mutation() {
        for incoherent in [
            RecoveryCoordinates::new(12, 13, GraphModelTag::Lpg, 17, 19),
            RecoveryCoordinates::new(11, 13, GraphModelTag::Both, 17, 19),
        ] {
            let mut sections = vec![
                section(SectionType::Catalog, 7, b"\x07catalog"),
                section(SectionType::LpgStore, 4, b"LPG4"),
            ];
            let error =
                append_world_metadata(&mut sections, inputs(GraphModelTag::Lpg), incoherent)
                    .expect_err("publisher must not create a self-inconsistent image");
            assert!(matches!(error, Error::Serialization(_)), "{error:?}");
            assert_eq!(sections.len(), 2, "failed sealing must not append metadata");
        }
    }

    #[test]
    fn recovery_seal_covers_every_acceleration_without_changing_the_logical_cut() {
        for (section_type, version) in [
            (SectionType::VectorStore, 4),
            (SectionType::TextIndex, 5),
            (SectionType::RdfRing, 2),
            (SectionType::PropertyIndex, 1),
        ] {
            let make_world = |image: &[u8]| {
                let mut sections = vec![
                    section(SectionType::Catalog, 7, b"\x07same-catalog"),
                    section(SectionType::LpgStore, 4, b"LPG4same-lpg"),
                    section(section_type, version, image),
                ];
                let cut = append_world_metadata(
                    &mut sections,
                    inputs(GraphModelTag::Lpg),
                    coordinates(GraphModelTag::Lpg),
                )
                .expect("seal recovery image");
                let digest = match verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg))
                    .unwrap()
                    .unwrap()
                {
                    VerifiedWorldMetadata::RecoverySealedV2(metadata) => {
                        metadata.recovery_image_digest()
                    }
                    VerifiedWorldMetadata::SectionSealedV2(_) => {
                        panic!("new publisher emitted the frozen section-only v2 profile")
                    }
                    VerifiedWorldMetadata::LegacyV1(_) => panic!("new publisher emitted v1"),
                };
                (sections, cut, digest)
            };

            let (mut world_a, cut_a, digest_a) = make_world(b"world-a-image");
            let (_, cut_b, digest_b) = make_world(b"world-b-image");
            assert_eq!(
                cut_a, cut_b,
                "physical acceleration layout must not redefine the logical world"
            );
            assert_ne!(digest_a, digest_b, "recovery images must remain distinct");

            world_a
                .iter_mut()
                .find(|section| section.section_type == section_type)
                .unwrap()
                .bytes = b"world-b-image".to_vec();
            let error = verify_world_metadata(&world_a, coordinates(GraphModelTag::Lpg))
                .expect_err("a foreign acceleration image must fail before model recovery");
            assert!(
                error.to_string().contains("recovery image digest"),
                "{error}"
            );
        }
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    #[test]
    fn world_metadata_directory_and_payload_generations_cannot_be_relabelled() {
        fn assert_model(model: GraphModelTag, sections: impl Fn() -> Vec<EncodedSection>) {
            let mut v2 = sections();
            append_world_metadata(&mut v2, inputs(model), coordinates(model)).unwrap();
            v2.iter_mut()
                .find(|section| section.section_type == SectionType::WorldMetadata)
                .unwrap()
                .version = 1;
            let error = validate_section_contracts(&v2, model)
                .expect_err("v2 payload cannot wear a v1 directory label");
            assert!(error.to_string().contains("disagrees"), "{error}");
            assert!(verify_world_metadata(&v2, coordinates(model)).is_err());

            let mut v1 = sections();
            append_legacy_world_metadata_v1(&mut v1, inputs(model));
            v1.iter_mut()
                .find(|section| section.section_type == SectionType::WorldMetadata)
                .unwrap()
                .version = 2;
            let error = validate_section_contracts(&v1, model)
                .expect_err("v1 payload cannot wear a v2 directory label");
            assert!(error.to_string().contains("disagrees"), "{error}");
            assert!(verify_world_metadata(&v1, coordinates(model)).is_err());
        }
        #[cfg(feature = "lpg")]
        assert_model(GraphModelTag::Lpg, || {
            vec![
                section(SectionType::Catalog, 7, b"\x07catalog"),
                section(SectionType::LpgStore, 4, b"LPG4"),
            ]
        });
        #[cfg(feature = "triple-store")]
        {
            use grafeo_core::graph::rdf::section::RdfStoreSection;
            use grafeo_core::graph::rdf::{RdfStore, RdfStoreConfig};
            let rdf = RdfStoreSection::new(std::sync::Arc::new(
                RdfStore::with_config_and_store_id(RdfStoreConfig::default(), store_id(7)),
            ));
            assert_model(GraphModelTag::Rdf, || {
                vec![
                    section(SectionType::Catalog, 2, b"\x02catalog"),
                    section(
                        SectionType::RdfStore,
                        rdf.version(),
                        &rdf.serialize().unwrap(),
                    ),
                ]
            });
        }
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn rdf_section_binds_dataset_and_history_under_distinct_tags() {
        use std::sync::Arc;

        use grafeo_core::graph::rdf::section::RdfStoreSection;
        use grafeo_core::graph::rdf::{RdfStore, RdfStoreConfig};

        let rdf = Arc::new(RdfStore::with_config_and_store_id(
            RdfStoreConfig::default(),
            store_id(7),
        ));
        let rdf_section = RdfStoreSection::new(rdf);
        let rdf_bytes = rdf_section.serialize().unwrap();
        let mut sections = vec![
            section(SectionType::Catalog, 2, b"\x02catalog"),
            section(SectionType::RdfStore, rdf_section.version(), &rdf_bytes),
        ];
        append_world_metadata(
            &mut sections,
            inputs(GraphModelTag::Rdf),
            coordinates(GraphModelTag::Rdf),
        )
        .unwrap();
        let metadata = verify_world_metadata(&sections, coordinates(GraphModelTag::Rdf))
            .unwrap()
            .unwrap();
        let formats = metadata.cut().descriptor().formats();
        assert!(
            formats.iter().any(|entry| {
                entry.format() == AuthoritativeFormat::Rdf && entry.version() == 6
            })
        );
        assert!(formats.iter().any(|entry| {
            entry.format() == AuthoritativeFormat::RdfHistory && entry.version() == 1
        }));
    }

    #[test]
    fn metadata_rejects_wrong_header_coordinates_and_versions() {
        let mut sections = vec![
            section(SectionType::Catalog, 7, b"\x07catalog"),
            section(SectionType::LpgStore, 4, b"LPG4"),
        ];
        append_world_metadata(
            &mut sections,
            inputs(GraphModelTag::Lpg),
            coordinates(GraphModelTag::Lpg),
        )
        .unwrap();
        assert!(
            verify_world_metadata(
                &sections,
                RecoveryCoordinates::new(12, 13, GraphModelTag::Lpg, 17, 19),
            )
            .is_err()
        );
        assert!(
            verify_world_metadata(
                &sections,
                RecoveryCoordinates::new(11, 13, GraphModelTag::Both, 17, 19),
            )
            .is_err()
        );

        sections
            .iter_mut()
            .find(|section| section.section_type == SectionType::Catalog)
            .unwrap()
            .version = 4;
        assert!(verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg)).is_err());
    }

    #[test]
    fn current_lpg_requires_metadata_with_or_without_vector_sections() {
        for with_vector in [false, true] {
            let mut sections = vec![
                section(SectionType::Catalog, 7, &[7]),
                section(SectionType::LpgStore, 4, b"LPG4"),
            ];
            if with_vector {
                sections.push(section(SectionType::VectorStore, 4, b"GVST\x04current"));
            }
            let result = verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg));
            assert!(matches!(
                result,
                Err(Error::Storage(StorageError::Corruption(message)))
                    if message.contains("requires WorldMetadata")
            ));
        }
    }

    #[test]
    fn decoded_state_must_match_manifest_identity_and_schema()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let catalog = crate::catalog::Catalog::new();
        let identity =
            WorldIdentityMetadataV1::new(store_id(7), HistoryCompleteness::Complete).unwrap();
        let actual = capture_inputs(
            identity.clone(),
            EpochId::new(11),
            GraphModelTag::Lpg,
            &catalog,
        );
        let mut sections = vec![
            section(SectionType::Catalog, 7, b"\x07catalog"),
            section(SectionType::LpgStore, 4, b"LPG4"),
        ];
        append_world_metadata(&mut sections, actual, coordinates(GraphModelTag::Lpg)).unwrap();
        let metadata = verify_world_metadata(&sections, coordinates(GraphModelTag::Lpg))
            .unwrap()
            .unwrap();

        verify_decoded_state(&metadata, identity, &catalog).unwrap();
        let foreign =
            WorldIdentityMetadataV1::new(store_id(8), HistoryCompleteness::Complete).unwrap();
        assert!(verify_decoded_state(&metadata, foreign, &catalog).is_err());

        let changed_catalog = crate::catalog::Catalog::new();
        changed_catalog.get_or_create_label("Changed")?;
        let matching_identity =
            WorldIdentityMetadataV1::new(store_id(7), HistoryCompleteness::Complete).unwrap();
        assert!(verify_decoded_state(&metadata, matching_identity, &changed_catalog).is_err());
        Ok(())
    }
}
