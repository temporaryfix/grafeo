//! RDF section serializer for the `.grafeo` container format.
//!
//! Implements the [`Section`] trait for RDF triple data (triples, named graphs).
//! Version 6 persists the validated, canonical [`RdfDatasetHistory`] model so
//! dropped graph incarnations and their statement histories survive checkpoint
//! and compaction. The 0.0.1 fork accepts only this current grammar.
//!
//! # Layout
//!
//! ```text
//! v6:
//! [Header 32B: magic "RDFB", version, quad-version count, graph-life count]
//! [Canonical history payload length]
//! [World identity + incarnation high-water + graph lives + typed-Quad versions]
//! [History payload CRC]
//! [Optional projection metadata: magic + length + versioned payload + CRC]
//!
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EpochId, StoreId, WorldIdentityMetadataV1};
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::graph::rdf::{
    RdfDatasetHistory, RdfGraphLife, RdfLpgProjectionDefinition, RdfLpgProjectionRegistry,
    RdfQuadVersion, RdfStore,
};

/// Current RDF section format version. Version 6 stores one canonical dataset
/// history payload. Version 5 stores validated signed `i128` TAI-nanosecond
/// bounds but only current named-graph partitions.
const RDF_SECTION_VERSION: u8 = 6;

/// Magic bytes for the RDF block format.
const RDF_BLOCK_MAGIC: [u8; 4] = *b"RDFB";
/// Header: magic(4) + version(1) + flags(1) + triple_count(4) + graph_count(4) + pad(18) = 32
const HEADER_SIZE: usize = 32;
/// Header flag: a projection metadata addendum follows the RDF block.
const FLAG_PROJECTION_METADATA: u8 = 0x01;
/// Projection addendum magic.
const PROJECTION_MAGIC: [u8; 4] = *b"RPJ1";
/// Version of the canonical history payload nested in an RDF section.
const RDF_HISTORY_PAYLOAD_VERSION: u8 = 1;
/// Defensive decode/allocation ceiling for one history payload.
const MAX_RDF_HISTORY_PAYLOAD_BYTES: usize = 512 * 1024 * 1024;
const MAX_PROJECTION_METADATA_BYTES: usize = 64 * 1024 * 1024;

/// Returns the independently versioned canonical RDF-history component nested
/// in a current RDF section.
///
/// The outer RDF section also carries framing and, optionally, RDF→LPG
/// projection metadata. World manifests must therefore identify the outer
/// section and the dataset-history grammar as two distinct authoritative
/// components rather than assigning the outer section version and bytes to
/// both.
///
/// # Errors
///
/// Returns an error unless `data` contains valid current-format RDF history
/// framing and a canonical history payload that passes the history decoder's
/// checks. The outer section digest and the normal section decoder separately
/// cover and validate any projection addendum.
pub fn canonical_history_component(data: &[u8]) -> Result<(u16, &[u8])> {
    let (_, _, history_end) = read_rdf_history_v6(data)?;
    let payload_len_bytes: [u8; 4] = data
        .get(HEADER_SIZE..HEADER_SIZE + 4)
        .ok_or_else(|| {
            Error::Serialization("RDF section lacks a history payload length".to_string())
        })?
        .try_into()
        .map_err(|_| Error::Serialization("invalid RDF history payload length".to_string()))?;
    let payload_len = u32::from_le_bytes(payload_len_bytes) as usize;
    let payload_start = HEADER_SIZE + 4;
    let payload_end = payload_start.checked_add(payload_len).ok_or_else(|| {
        Error::Serialization("RDF history payload range overflows usize".to_string())
    })?;
    if payload_end.checked_add(4) != Some(history_end) {
        return Err(Error::Serialization(
            "RDF history component boundary disagrees with validated section framing".to_string(),
        ));
    }
    let payload = data.get(payload_start..payload_end).ok_or_else(|| {
        Error::Serialization("RDF section is truncated at canonical history payload".to_string())
    })?;
    Ok((u16::from(RDF_HISTORY_PAYLOAD_VERSION), payload))
}

#[derive(Debug, Serialize, Deserialize)]
struct RdfHistorySectionV1 {
    version: u8,
    identity: WorldIdentityMetadataV1,
    next_graph_incarnation: grafeo_common::types::GraphIncarnationId,
    graph_lives: Vec<RdfGraphLife>,
    quad_versions: Vec<RdfQuadVersion>,
}

// ── Serialization ──────────────────────────────────────────────────

fn write_rdf_history_v6(history: &RdfDatasetHistory, commit_epoch: EpochId) -> Result<Vec<u8>> {
    if commit_epoch == EpochId::PENDING {
        return Err(Error::Serialization(
            "RDF section cannot persist the pending epoch sentinel".to_string(),
        ));
    }
    let graph_count = u32::try_from(history.graph_lives().len()).map_err(|_| {
        Error::Serialization("RDF graph-lifecycle count exceeds u32 section limit".to_string())
    })?;
    let quad_count = u32::try_from(history.quad_versions().len()).map_err(|_| {
        Error::Serialization("RDF quad-version count exceeds u32 section limit".to_string())
    })?;
    let identity = WorldIdentityMetadataV1::new(history.store_id(), history.completeness())
        .map_err(|error| Error::Serialization(format!("invalid RDF identity metadata: {error}")))?;
    let payload = RdfHistorySectionV1 {
        version: RDF_HISTORY_PAYLOAD_VERSION,
        identity,
        next_graph_incarnation: history.next_graph_incarnation(),
        graph_lives: history.graph_lives().to_vec(),
        quad_versions: history.quad_versions().to_vec(),
    };
    let payload = bincode::serde::encode_to_vec(
        &payload,
        bincode::config::standard().with_limit::<MAX_RDF_HISTORY_PAYLOAD_BYTES>(),
    )
    .map_err(|error| Error::Serialization(format!("RDF history serialization failed: {error}")))?;
    if payload.len() > MAX_RDF_HISTORY_PAYLOAD_BYTES {
        return Err(Error::Serialization(format!(
            "RDF history payload has {} bytes; maximum is {MAX_RDF_HISTORY_PAYLOAD_BYTES}",
            payload.len()
        )));
    }
    let payload_len = u32::try_from(payload.len()).map_err(|_| {
        Error::Serialization("RDF history payload exceeds u32 section limit".to_string())
    })?;

    let capacity = HEADER_SIZE
        .checked_add(4)
        .and_then(|size| size.checked_add(payload.len()))
        .and_then(|size| size.checked_add(4))
        .ok_or_else(|| Error::Serialization("RDF section size overflow".to_string()))?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&RDF_BLOCK_MAGIC);
    bytes.push(RDF_SECTION_VERSION);
    bytes.push(0);
    bytes.extend_from_slice(&quad_count.to_le_bytes());
    bytes.extend_from_slice(&graph_count.to_le_bytes());
    bytes.extend_from_slice(&commit_epoch.as_u64().to_le_bytes());
    bytes.extend_from_slice(&[0; 10]);
    debug_assert_eq!(bytes.len(), HEADER_SIZE);
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    Ok(bytes)
}

struct CheckedCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CheckedCursor<'a> {
    const fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn take(&mut self, len: usize, field: &str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Error::Serialization(format!("RDF section {field} length overflow")))?;
        let value = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| Error::Serialization(format!("RDF section truncated at {field}")))?;
        self.pos = end;
        Ok(value)
    }

    fn read_u32(&mut self, field: &str) -> Result<u32> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .map_err(|_| Error::Serialization(format!("invalid RDF section {field}")))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self, field: &str) -> Result<u64> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .map_err(|_| Error::Serialization(format!("invalid RDF section {field}")))?;
        Ok(u64::from_le_bytes(bytes))
    }

    const fn position(&self) -> usize {
        self.pos
    }
}

fn read_rdf_history_v6(data: &[u8]) -> Result<(RdfDatasetHistory, EpochId, usize)> {
    if data.len() < HEADER_SIZE {
        return Err(Error::Serialization(
            "RDF history section too short for header".to_string(),
        ));
    }
    let mut header = CheckedCursor::new(data, 0);
    if header.take(4, "magic")? != RDF_BLOCK_MAGIC {
        return Err(Error::Serialization(
            "invalid RDF block magic bytes".to_string(),
        ));
    }
    let version = header.take(1, "version")?[0];
    if version != RDF_SECTION_VERSION {
        return Err(Error::Serialization(format!(
            "expected RDF section version {RDF_SECTION_VERSION}, got {version}"
        )));
    }
    let flags = header.take(1, "flags")?[0];
    if flags & !FLAG_PROJECTION_METADATA != 0 {
        return Err(Error::Serialization(format!(
            "unsupported RDF section flags 0x{flags:02x}"
        )));
    }
    let quad_count = header.read_u32("quad-version count")? as usize;
    let graph_count = header.read_u32("graph-lifecycle count")? as usize;
    let commit_epoch = EpochId::new(header.read_u64("commit epoch")?);
    if commit_epoch == EpochId::PENDING {
        return Err(Error::Serialization(
            "RDF history section uses the pending epoch sentinel".to_string(),
        ));
    }
    if header.take(10, "reserved header bytes")? != [0; 10] {
        return Err(Error::Serialization(
            "RDF history section has non-zero reserved header bytes".to_string(),
        ));
    }

    let mut cursor = CheckedCursor::new(data, HEADER_SIZE);
    let payload_len = cursor.read_u32("history payload length")? as usize;
    if payload_len > MAX_RDF_HISTORY_PAYLOAD_BYTES {
        return Err(Error::Serialization(format!(
            "RDF history payload has {payload_len} bytes; maximum is {MAX_RDF_HISTORY_PAYLOAD_BYTES}"
        )));
    }
    let payload = cursor.take(payload_len, "history payload")?;
    let expected_crc = cursor.read_u32("history payload CRC")?;
    let actual_crc = crc32fast::hash(payload);
    if expected_crc != actual_crc {
        return Err(Error::Serialization(format!(
            "RDF history payload CRC mismatch: expected {expected_crc:08x}, got {actual_crc:08x}"
        )));
    }

    let config = bincode::config::standard().with_limit::<MAX_RDF_HISTORY_PAYLOAD_BYTES>();
    let (wire, consumed): (RdfHistorySectionV1, usize) =
        bincode::serde::decode_from_slice(payload, config).map_err(|error| {
            Error::Serialization(format!("RDF history deserialization failed: {error}"))
        })?;
    if consumed != payload.len() {
        return Err(Error::Serialization(format!(
            "RDF history payload contains {} trailing bytes",
            payload.len() - consumed
        )));
    }
    if wire.version != RDF_HISTORY_PAYLOAD_VERSION {
        return Err(Error::Serialization(format!(
            "unsupported RDF history payload version {}",
            wire.version
        )));
    }
    if wire.graph_lives.len() != graph_count || wire.quad_versions.len() != quad_count {
        return Err(Error::Serialization(
            "RDF history header counts do not match the canonical payload".to_string(),
        ));
    }
    for tx in wire
        .graph_lives
        .iter()
        .map(RdfGraphLife::tx)
        .chain(wire.quad_versions.iter().map(RdfQuadVersion::tx))
    {
        if tx.from() > commit_epoch || (!tx.is_open() && tx.to() > commit_epoch) {
            return Err(Error::Serialization(
                "RDF history contains a transaction interval beyond the section commit epoch"
                    .to_string(),
            ));
        }
    }
    if wire
        .identity
        .history()
        .authoritative_from()
        .is_some_and(|epoch| epoch > commit_epoch)
    {
        return Err(Error::Serialization(
            "RDF history completeness boundary exceeds the section commit epoch".to_string(),
        ));
    }
    let history = RdfDatasetHistory::new_with_high_water(
        wire.identity.store_id(),
        wire.identity.history(),
        wire.next_graph_incarnation,
        wire.graph_lives,
        wire.quad_versions,
    )
    .map_err(|error| Error::Serialization(format!("invalid RDF dataset history: {error}")))?;
    Ok((history, commit_epoch, cursor.position()))
}

fn append_projection_metadata(
    data: &mut Vec<u8>,
    projections: &RdfLpgProjectionRegistry,
) -> Result<()> {
    let payload = projections.encode_persistence_v3().map_err(|error| {
        Error::Serialization(format!(
            "RDF projection metadata serialization failed: {error}"
        ))
    })?;
    if payload.len() > MAX_PROJECTION_METADATA_BYTES {
        return Err(Error::Serialization(format!(
            "RDF projection metadata has {} bytes; maximum is {MAX_PROJECTION_METADATA_BYTES}",
            payload.len()
        )));
    }
    let len = u32::try_from(payload.len()).map_err(|_| {
        Error::Serialization("RDF projection metadata exceeds u32 section limit".to_string())
    })?;
    data[5] |= FLAG_PROJECTION_METADATA;
    data.extend_from_slice(&PROJECTION_MAGIC);
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(&payload);
    data.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    Ok(())
}

fn read_projection_metadata(
    data: &[u8],
    pos: usize,
    expected_store_id: StoreId,
) -> Result<Vec<RdfLpgProjectionDefinition>> {
    let mut cursor = CheckedCursor::new(data, pos);
    if cursor.take(4, "projection metadata magic")? != PROJECTION_MAGIC {
        return Err(Error::Serialization(
            "invalid RDF projection metadata magic".to_string(),
        ));
    }
    let len = cursor.read_u32("projection metadata length")? as usize;
    if len > MAX_PROJECTION_METADATA_BYTES {
        return Err(Error::Serialization(format!(
            "RDF projection metadata has {len} bytes; maximum is {MAX_PROJECTION_METADATA_BYTES}"
        )));
    }
    let payload = cursor.take(len, "projection metadata payload")?;
    let expected_crc = cursor.read_u32("projection metadata CRC")?;
    if cursor.position() != data.len() {
        return Err(Error::Serialization(format!(
            "RDF projection metadata contains {} trailing bytes",
            data.len() - cursor.position()
        )));
    }
    let actual_crc = crc32fast::hash(payload);
    if expected_crc != actual_crc {
        return Err(Error::Serialization(format!(
            "RDF projection metadata CRC mismatch: expected {expected_crc:08x}, got {actual_crc:08x}"
        )));
    }
    RdfLpgProjectionRegistry::decode_persistence(expected_store_id, payload).map_err(|error| {
        Error::Serialization(format!(
            "RDF projection metadata deserialization failed: {error}"
        ))
    })
}

// ── Section implementation ──────────────────────────────────────────

/// RDF store section for the `.grafeo` container.
pub struct RdfStoreSection {
    store: Arc<RdfStore>,
    projections: Option<Arc<RdfLpgProjectionRegistry>>,
    captured_history: Option<(RdfDatasetHistory, EpochId)>,
    dirty: AtomicBool,
}

impl RdfStoreSection {
    /// Create a new RDF section wrapping the given store.
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            projections: None,
            captured_history: None,
            dirty: AtomicBool::new(false),
        }
    }

    /// Captures a coherent section while the caller already holds the store's
    /// commit gate.
    ///
    /// This constructor is for checkpoint paths whose lock order is
    /// `RDF commit gate -> publication barrier`. Ordinary callers use [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns an error if the captured dataset history violates an
    /// authoritative persistence invariant.
    pub fn new_under_commit_gate(store: Arc<RdfStore>) -> Result<Self> {
        let captured_history = Some(Self::capture_under_commit_gate(&store)?);
        Ok(Self {
            store,
            projections: None,
            captured_history,
            dirty: AtomicBool::new(false),
        })
    }

    /// Creates an RDF section that also persists RDF→LPG projection metadata.
    pub fn with_projections(
        store: Arc<RdfStore>,
        projections: Arc<RdfLpgProjectionRegistry>,
    ) -> Self {
        Self {
            store,
            projections: Some(projections),
            captured_history: None,
            dirty: AtomicBool::new(false),
        }
    }

    /// Projection-aware variant of [`Self::new_under_commit_gate`].
    ///
    /// # Errors
    ///
    /// Returns an error if the captured dataset history violates an
    /// authoritative persistence invariant.
    pub fn with_projections_under_commit_gate(
        store: Arc<RdfStore>,
        projections: Arc<RdfLpgProjectionRegistry>,
    ) -> Result<Self> {
        let captured_history = Some(Self::capture_under_commit_gate(&store)?);
        Ok(Self {
            store,
            projections: Some(projections),
            captured_history,
            dirty: AtomicBool::new(false),
        })
    }

    fn capture_under_commit_gate(store: &RdfStore) -> Result<(RdfDatasetHistory, EpochId)> {
        let history = store.dataset_history_under_commit_gate().map_err(|error| {
            Error::Serialization(format!("invalid RDF dataset history: {error}"))
        })?;
        Ok((history, store.commit_epoch()))
    }

    fn capture(&self) -> Result<(RdfDatasetHistory, EpochId)> {
        if let Some(captured) = &self.captured_history {
            return Ok(captured.clone());
        }
        let _commit = self.store.lock_commit();
        Self::capture_under_commit_gate(&self.store)
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Access the underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<RdfStore> {
        &self.store
    }
}

impl Section for RdfStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::RdfStore
    }

    fn version(&self) -> u8 {
        RDF_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let (history, commit_epoch) = self.capture()?;
        let mut data = write_rdf_history_v6(&history, commit_epoch)?;
        if let Some(projections) = &self.projections
            && !projections.snapshot().is_empty()
        {
            append_projection_metadata(&mut data, projections)?;
        }
        Ok(data)
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let (history, commit_epoch, rdf_len) = read_rdf_history_v6(data)?;
        let definitions = if data[5] & FLAG_PROJECTION_METADATA != 0 {
            read_projection_metadata(data, rdf_len, history.store_id())?
        } else {
            if rdf_len != data.len() {
                return Err(Error::Serialization(format!(
                    "RDF history section contains {} unframed trailing bytes",
                    data.len() - rdf_len
                )));
            }
            Vec::new()
        };

        // Validate every addendum before publishing either component.
        let validation = RdfLpgProjectionRegistry::new();
        validation
            .restore_for_store(history.store_id(), definitions.clone())
            .map_err(|error| {
                Error::Serialization(format!("invalid RDF projection registry metadata: {error}"))
            })?;
        self.store
            .replace_dataset_history_exact(history, commit_epoch)
            .map_err(|error| {
                Error::Serialization(format!("RDF history install failed: {error}"))
            })?;
        if let Some(projections) = &self.projections {
            projections
                .restore_for_store(self.store.store_id(), definitions)
                .map_err(|error| {
                    Error::Serialization(format!(
                        "invalid RDF projection registry metadata: {error}"
                    ))
                })?;
        }
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
            || self
                .projections
                .as_ref()
                .is_some_and(|projections| projections.is_dirty())
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
        if let Some(projections) = &self.projections {
            projections.mark_clean();
        }
    }

    fn memory_usage(&self) -> usize {
        self.store.len() * 200
            + self
                .projections
                .as_ref()
                .map_or(0, |projections| projections.snapshot().len() * 160)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{Term, Triple};
    use grafeo_common::types::{
        HistoryCompleteness, ProjectionReconciliationState, ProjectionSourceGraph,
        ValidTimeInterval,
    };

    #[test]
    fn canonical_duplicate_section_rejects_before_any_target_publication() {
        use crate::graph::rdf::Quad;
        use grafeo_common::types::{EpochInterval, GraphIncarnationId};
        let source = Arc::new(RdfStore::new());
        source.try_set_commit_epoch(EpochId::new(3)).unwrap();
        assert!(source.insert(Triple::new(
            Term::iri("urn:s"),
            Term::iri("urn:p"),
            Term::lang_literal("value", "EN"),
        )));
        let mut bytes = RdfStoreSection::new(Arc::clone(&source))
            .serialize()
            .unwrap();
        let payload_len =
            u32::from_le_bytes(bytes[HEADER_SIZE..HEADER_SIZE + 4].try_into().unwrap()) as usize;
        let (mut wire, _): (RdfHistorySectionV1, usize) = bincode::serde::decode_from_slice(
            &bytes[HEADER_SIZE + 4..HEADER_SIZE + 4 + payload_len],
            bincode::config::standard(),
        )
        .unwrap();
        wire.quad_versions.push(
            RdfQuadVersion::new(
                source.store_id(),
                Quad::new(Triple::new(
                    Term::iri("urn:s"),
                    Term::iri("urn:p"),
                    Term::lang_literal("value", "en"),
                )),
                GraphIncarnationId::DEFAULT_GRAPH,
                EpochInterval::open(EpochId::new(3)),
                None,
            )
            .unwrap(),
        );
        let payload = bincode::serde::encode_to_vec(&wire, bincode::config::standard()).unwrap();
        bytes[6..10].copy_from_slice(&2_u32.to_le_bytes());
        bytes.truncate(HEADER_SIZE);
        bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        let target = Arc::new(RdfStore::new());
        target.try_set_commit_epoch(EpochId::new(9)).unwrap();
        assert!(target.insert(Triple::new(
            Term::iri("urn:keep"),
            Term::iri("urn:p"),
            Term::literal("old")
        )));
        let projections = Arc::new(RdfLpgProjectionRegistry::new());
        projections.declare("urn:Sentinel", "Sentinel").unwrap();
        let before = target.dataset_history().unwrap();
        let before_projections = projections.snapshot();
        let error =
            RdfStoreSection::with_projections(Arc::clone(&target), Arc::clone(&projections))
                .deserialize(&bytes)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("overlapping transaction-time versions"),
            "{error}"
        );
        assert!(canonical_history_component(&bytes).is_err());
        let after = target.dataset_history().unwrap();
        assert_eq!(after.store_id(), before.store_id());
        assert_eq!(after.completeness(), before.completeness());
        assert_eq!(after.graph_lives(), before.graph_lives());
        assert_eq!(after.quad_versions(), before.quad_versions());
        assert_eq!(target.commit_epoch(), EpochId::new(9));
        assert_eq!(projections.snapshot(), before_projections);
    }

    #[test]
    fn canonical_alias_nonoverlapping_section_retains_lossless_lifetimes() {
        use crate::graph::rdf::Quad;
        use grafeo_common::types::{EpochInterval, GraphIncarnationId};
        let sid = StoreId::from_bytes([31; StoreId::LEN]).unwrap();
        let versions: Vec<_> = [
            (
                "EN",
                EpochInterval::closed(EpochId::new(1), EpochId::new(3)),
            ),
            ("en", EpochInterval::open(EpochId::new(3))),
        ]
        .into_iter()
        .map(|(language, tx)| {
            RdfQuadVersion::new(
                sid,
                Quad::new(Triple::new(
                    Term::iri("urn:s"),
                    Term::iri("urn:p"),
                    Term::lang_literal("value", language),
                )),
                GraphIncarnationId::DEFAULT_GRAPH,
                tx,
                None,
            )
            .unwrap()
        })
        .collect();
        let history =
            RdfDatasetHistory::new(sid, HistoryCompleteness::Complete, vec![], versions).unwrap();
        let bytes = write_rdf_history_v6(&history, EpochId::new(3)).unwrap();
        let target = Arc::new(RdfStore::new());
        RdfStoreSection::new(Arc::clone(&target))
            .deserialize(&bytes)
            .unwrap();
        let restored = target.dataset_history().unwrap();
        assert_eq!(restored.quad_versions(), history.quad_versions());
        assert_eq!(
            restored.cut(EpochId::new(2)).unwrap(),
            history.cut(EpochId::new(2)).unwrap()
        );
        assert_eq!(
            restored.cut(EpochId::new(3)).unwrap(),
            history.cut(EpochId::new(3)).unwrap()
        );
        assert_eq!(target.len(), 1);
    }

    #[test]
    fn rdf_section_round_trip() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));

        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"RDFB");

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(section2.store().len(), 2);
    }

    #[test]
    fn rdf_section_round_trips_i128_tai_nanosecond_valid_time() {
        let store = Arc::new(RdfStore::new());
        let triple = Triple::new(
            Term::iri("http://example.org/high-precision"),
            Term::iri("http://example.org/p"),
            Term::literal("value"),
        );
        let from = i128::from(i64::MAX) * 1_000 + 1;
        let valid = ValidTimeInterval::from_tai_nanoseconds(from, from + 3).unwrap();
        assert!(
            store
                .try_insert_at_epoch_with_valid(triple.clone(), EpochId::new(7), Some(valid),)
                .unwrap()
        );
        store.try_set_commit_epoch(EpochId::new(7)).unwrap();

        let section = RdfStoreSection::new(store);
        let bytes = section.serialize().expect("serialize TAI-ns interval");
        assert_eq!(bytes[4], RDF_SECTION_VERSION);
        let restored = Arc::new(RdfStore::new());
        let mut section = RdfStoreSection::new(Arc::clone(&restored));
        section
            .deserialize(&bytes)
            .expect("deserialize TAI-ns interval");

        let (_, lives) = restored
            .quad_history()
            .into_iter()
            .find(|(candidate, _)| candidate.as_ref() == &triple)
            .expect("restored quad history");
        assert_eq!(lives.len(), 1);
        assert_eq!(lives[0].valid, Some(valid));
    }

    #[test]
    fn rdf_section_type() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(store);
        assert_eq!(section.section_type(), SectionType::RdfStore);
    }

    #[test]
    fn rdf_section_version() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(store);
        assert_eq!(section.version(), RDF_SECTION_VERSION);
    }

    #[test]
    fn rdf_section_dirty_tracking() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(store);

        assert!(!section.is_dirty(), "new section should be clean");

        section.mark_dirty();
        assert!(
            section.is_dirty(),
            "section should be dirty after mark_dirty"
        );

        section.mark_clean();
        assert!(
            !section.is_dirty(),
            "section should be clean after mark_clean"
        );
    }

    #[test]
    fn rdf_section_memory_usage() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/vincent"),
            Term::iri("http://xmlns.com/foaf/0.1/knows"),
            Term::iri("http://example.org/jules"),
        ));
        let section = RdfStoreSection::new(store);
        let usage = section.memory_usage();
        assert_eq!(usage, 200);
    }

    #[test]
    fn rdf_section_named_graph_round_trip() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/mia"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Mia"),
        ));

        store.create_graph("http://example.org/graph/butch");
        if let Some(named) = store.graph("http://example.org/graph/butch") {
            named.insert(Triple::new(
                Term::iri("http://example.org/butch"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Butch"),
            ));
            named.insert(Triple::new(
                Term::iri("http://example.org/butch"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/mia"),
            ));
        }

        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize named graphs");

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize named graphs");

        assert_eq!(section2.store().len(), 1);

        let names = section2.store().graph_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "http://example.org/graph/butch");

        let named = section2
            .store()
            .graph("http://example.org/graph/butch")
            .expect("named graph should exist");
        assert_eq!(named.len(), 2);
    }

    #[test]
    fn rdf_projection_metadata_round_trip() {
        let store = Arc::new(RdfStore::new());
        store.try_set_commit_epoch(EpochId::new(12)).unwrap();
        let plain_bytes = RdfStoreSection::new(Arc::clone(&store))
            .serialize()
            .expect("serialize canonical RDF history without projection metadata");
        let projections = Arc::new(RdfLpgProjectionRegistry::new());
        let id = projections
            .declare("http://example.org/Person", "Person")
            .unwrap();
        let declared = projections.get(id).unwrap();
        let receipt = crate::graph::rdf::RdfLpgProjectionReceipt::new(
            store.store_id(),
            declared.mapping_digest(),
            id,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(11),
            EpochId::new(12),
            1,
            0,
            ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        assert!(
            projections
                .install_receipt(store.store_id(), receipt.clone())
                .unwrap()
        );

        let section = RdfStoreSection::with_projections(store, projections);
        let bytes = section.serialize().expect("serialize projection metadata");

        let (plain_history_version, plain_history) =
            canonical_history_component(&plain_bytes).expect("extract plain RDF history");
        let (projected_history_version, projected_history) =
            canonical_history_component(&bytes).expect("extract projected RDF history");
        assert_eq!(plain_history_version, 1);
        assert_eq!(projected_history_version, 1);
        assert_eq!(plain_history, projected_history);
        assert_ne!(
            plain_bytes, bytes,
            "projection metadata changes the outer RDF section"
        );

        let restored = Arc::new(RdfLpgProjectionRegistry::new());
        let mut section =
            RdfStoreSection::with_projections(Arc::new(RdfStore::new()), Arc::clone(&restored));
        section
            .deserialize(&bytes)
            .expect("deserialize projection metadata");

        let definition = restored.get(id).expect("projection restored");
        assert_eq!(definition.last_source_epoch(), Some(EpochId::new(11)));
        assert_eq!(definition.last_target_epoch(), Some(EpochId::new(12)));
        assert_eq!(definition.generation(), 1);
        assert_eq!(definition.row_count(), 0);
        assert_eq!(definition.receipt(), Some(&receipt));
        assert_eq!(section.store().store_id(), receipt.store_id());
    }

    #[test]
    fn rdf_section_deserialize_invalid_data() {
        let store = Arc::new(RdfStore::new());
        let mut section = RdfStoreSection::new(store);
        let bad_bytes = &[0xFF, 0xFE, 0xFD, 0x00, 0x01];
        let result = section.deserialize(bad_bytes);
        assert!(
            result.is_err(),
            "corrupted data should fail deserialization"
        );
        assert!(
            canonical_history_component(bad_bytes).is_err(),
            "corrupted data must not yield a canonical history manifest component"
        );
    }

    #[test]
    fn rdf_section_empty_store_round_trip() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize empty store");

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize empty store");
        assert_eq!(section2.store().len(), 0);
        assert_eq!(section2.memory_usage(), 0);
    }

    #[test]
    fn rdf_section_crc_corruption_detected() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/test"),
            Term::iri("http://example.org/pred"),
            Term::literal("value"),
        ));

        let section = RdfStoreSection::new(Arc::clone(&store));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte near the end (triple data area)
        let last = bytes.len() - 5;
        bytes[last] ^= 0xFF;

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        assert!(section2.deserialize(&bytes).is_err());
    }

    #[test]
    fn rdf_section_string_deduplication() {
        let store = Arc::new(RdfStore::new());
        let pred = Term::iri("http://xmlns.com/foaf/0.1/name");
        // Same predicate used in multiple triples
        for i in 0..100 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/node{i}")),
                pred.clone(),
                Term::literal(format!("Name{i}")),
            ));
        }

        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        // Verify round-trip
        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2.deserialize(&bytes).unwrap();
        assert_eq!(section2.store().len(), 100);
    }

    #[test]
    fn rdf_section_v6_retains_dropped_graph_history_identity_and_cdc() {
        let store = Arc::new(RdfStore::new());
        store.try_set_commit_epoch(EpochId::new(1)).unwrap();
        assert!(store.create_graph("urn:g"));
        let first = store.graph("urn:g").unwrap();
        store.try_set_commit_epoch(EpochId::new(2)).unwrap();
        assert!(first.insert(Triple::new(
            Term::iri("urn:old"),
            Term::iri("urn:p"),
            Term::literal("old"),
        )));
        store.try_set_commit_epoch(EpochId::new(3)).unwrap();
        assert!(store.drop_graph("urn:g"));
        store.try_set_commit_epoch(EpochId::new(4)).unwrap();
        assert!(store.create_graph("urn:g"));
        assert!(store.graph("urn:g").unwrap().insert(Triple::new(
            Term::iri("urn:new"),
            Term::iri("urn:p"),
            Term::literal("new"),
        )));

        let expected = store.dataset_history().unwrap();
        let bytes = RdfStoreSection::new(Arc::clone(&store))
            .serialize()
            .unwrap();
        assert_eq!(bytes[4], RDF_SECTION_VERSION);

        let restored = Arc::new(RdfStore::new());
        RdfStoreSection::new(Arc::clone(&restored))
            .deserialize(&bytes)
            .unwrap();
        let actual = restored.dataset_history().unwrap();
        assert_eq!(actual.store_id(), expected.store_id());
        assert_eq!(actual.completeness(), HistoryCompleteness::Complete);
        assert_eq!(actual.graph_lives(), expected.graph_lives());
        assert_eq!(actual.quad_versions(), expected.quad_versions());
        assert_eq!(
            actual
                .ordered_diff(EpochId::new(0), EpochId::new(4))
                .unwrap(),
            expected
                .ordered_diff(EpochId::new(0), EpochId::new(4))
                .unwrap()
        );
        assert_eq!(actual.cut(EpochId::new(2)).unwrap().quads.len(), 1);
        assert_eq!(actual.cut(EpochId::new(3)).unwrap().quads.len(), 0);
        assert_eq!(actual.cut(EpochId::new(4)).unwrap().quads.len(), 1);
    }

    #[test]
    fn rdf_section_rejects_pre_0_0_1_versions_before_mutation() {
        let source = Arc::new(RdfStore::new());
        source.try_set_commit_epoch(EpochId::new(10)).unwrap();
        assert!(source.insert(Triple::new(
            Term::iri("urn:old"),
            Term::iri("urn:p"),
            Term::literal("old"),
        )));
        let mut bytes = RdfStoreSection::new(source).serialize().unwrap();
        bytes[4] = 5;

        let target = Arc::new(RdfStore::new());
        target.try_set_commit_epoch(EpochId::new(12)).unwrap();
        assert!(target.insert(Triple::new(
            Term::iri("urn:sentinel"),
            Term::iri("urn:p"),
            Term::literal("keep"),
        )));
        let before = target.dataset_history().unwrap();

        let error = RdfStoreSection::new(Arc::clone(&target))
            .deserialize(&bytes)
            .expect_err("pre-0.0.1 RDF sections must be rejected");
        assert!(
            error
                .to_string()
                .contains("expected RDF section version 6, got 5"),
            "{error}"
        );
        let after = target.dataset_history().unwrap();
        assert_eq!(after.store_id(), before.store_id());
        assert_eq!(after.completeness(), before.completeness());
        assert_eq!(after.graph_lives(), before.graph_lives());
        assert_eq!(after.quad_versions(), before.quad_versions());
        assert_eq!(target.commit_epoch(), EpochId::new(12));
    }

    #[test]
    fn rdf_section_v6_rejects_hostile_length_without_panic_or_publication() {
        let source = Arc::new(RdfStore::new());
        let mut bytes = RdfStoreSection::new(source).serialize().unwrap();
        bytes[HEADER_SIZE..HEADER_SIZE + 4].copy_from_slice(&u32::MAX.to_le_bytes());

        let target = Arc::new(RdfStore::new());
        assert!(target.insert(Triple::new(
            Term::iri("urn:sentinel"),
            Term::iri("urn:p"),
            Term::literal("keep"),
        )));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RdfStoreSection::new(Arc::clone(&target)).deserialize(&bytes)
        }));
        assert!(result.is_ok(), "hostile section length must not panic");
        assert!(result.unwrap().is_err());
        assert_eq!(target.len(), 1, "decode failure must not publish a prefix");
    }

    #[test]
    fn rdf_section_late_addendum_failure_preserves_target_exactly() {
        fn section_with_projection_payload(payload: &[u8]) -> Vec<u8> {
            let source = Arc::new(RdfStore::new());
            source.try_set_commit_epoch(EpochId::new(5)).unwrap();
            assert!(source.insert(Triple::new(
                Term::iri("urn:incoming"),
                Term::iri("urn:p"),
                Term::literal("incoming"),
            )));
            let mut bytes = RdfStoreSection::new(source).serialize().unwrap();
            bytes[5] |= FLAG_PROJECTION_METADATA;
            bytes.extend_from_slice(&PROJECTION_MAGIC);
            bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
            bytes.extend_from_slice(payload);
            bytes.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
            bytes
        }

        let mut altered_v3 = b"GRPS".to_vec();
        altered_v3.extend_from_slice(&u16::MAX.to_le_bytes());
        for (case, payload) in [
            ("pre-v3", vec![0]),
            ("altered-v3", altered_v3),
            ("malformed", vec![0xff]),
        ] {
            let bytes = section_with_projection_payload(&payload);
            let target = Arc::new(RdfStore::new());
            target.try_set_commit_epoch(EpochId::new(12)).unwrap();
            assert!(target.insert(Triple::new(
                Term::iri("urn:existing"),
                Term::iri("urn:p"),
                Term::literal("existing"),
            )));
            let projections = Arc::new(RdfLpgProjectionRegistry::new());
            projections
                .declare("http://example.org/Sentinel", "Sentinel")
                .unwrap();
            let before_history = target.dataset_history().unwrap();
            let before_projections = projections.snapshot();

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                RdfStoreSection::with_projections(Arc::clone(&target), Arc::clone(&projections))
                    .deserialize(&bytes)
            }));
            assert!(result.is_ok(), "{case} addendum must not panic");
            assert!(result.unwrap().is_err(), "{case} addendum must reject");
            let after = target.dataset_history().unwrap();
            assert_eq!(after.store_id(), before_history.store_id(), "{case}");
            assert_eq!(
                after.completeness(),
                before_history.completeness(),
                "{case}"
            );
            assert_eq!(after.graph_lives(), before_history.graph_lives(), "{case}");
            assert_eq!(
                after.quad_versions(),
                before_history.quad_versions(),
                "{case}"
            );
            assert_eq!(target.commit_epoch(), EpochId::new(12), "{case}");
            assert_eq!(projections.snapshot(), before_projections, "{case}");
        }
    }
}
