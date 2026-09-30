//! Canonical, bounded metadata for the current backup chain format.
//!
//! This module deliberately contains no filesystem or restore policy.  It
//! owns the outer envelope and the small canonical primitives shared by the
//! backup manifest and its callers.

use blake3::Hasher;
use std::collections::HashSet;

use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};

pub(crate) const MAGIC: [u8; 4] = *b"GBK2";
pub(crate) const VERSION: u32 = 2;
pub(crate) const KIND_MANIFEST: u8 = 1;
pub(crate) const KIND_CURSOR: u8 = 2;
pub(crate) const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
const HEADER_BYTES: usize = 4 + 4 + 1 + 8;

pub(crate) fn envelope(kind: u8, payload: &[u8]) -> Result<Vec<u8>> {
    validate_kind(kind)?;
    if payload.len() > MAX_PAYLOAD_BYTES {
        return Err(Error::Serialization(format!(
            "backup metadata payload exceeds {} bytes",
            MAX_PAYLOAD_BYTES
        )));
    }
    let length = u64::try_from(payload.len())
        .map_err(|_| Error::Serialization("backup metadata length overflow".into()))?;
    let mut output = Vec::with_capacity(HEADER_BYTES.saturating_add(payload.len()));
    output.extend_from_slice(&MAGIC);
    output.extend_from_slice(&VERSION.to_le_bytes());
    output.push(kind);
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(payload);
    Ok(output)
}

pub(crate) fn payload(kind: u8, bytes: &[u8]) -> Result<&[u8]> {
    validate_kind(kind)?;
    if bytes.len() < 8 {
        return Err(Error::Serialization("backup metadata is truncated".into()));
    }
    if bytes[..4] != MAGIC {
        return Err(Error::Serialization(
            "unsupported backup metadata magic".into(),
        ));
    }
    let version = u32::from_le_bytes(
        bytes[4..8]
            .try_into()
            .map_err(|_| Error::Serialization("backup metadata version is truncated".into()))?,
    );
    if version != VERSION {
        return Err(Error::Serialization(format!(
            "unsupported backup metadata version {version}"
        )));
    }
    if bytes.len() < HEADER_BYTES {
        return Err(Error::Serialization("backup metadata is truncated".into()));
    }
    if bytes[8] != kind {
        return Err(Error::Serialization("backup metadata kind mismatch".into()));
    }
    let declared = u64::from_le_bytes(
        bytes[9..17]
            .try_into()
            .map_err(|_| Error::Serialization("backup metadata length is truncated".into()))?,
    );
    let declared = usize::try_from(declared)
        .map_err(|_| Error::Serialization("backup metadata length overflows usize".into()))?;
    if declared > MAX_PAYLOAD_BYTES {
        return Err(Error::Serialization(
            "backup metadata payload is too large".into(),
        ));
    }
    let expected = HEADER_BYTES
        .checked_add(declared)
        .ok_or_else(|| Error::Serialization("backup metadata length overflow".into()))?;
    if expected != bytes.len() {
        return Err(Error::Serialization(
            "backup metadata has trailing or missing bytes".into(),
        ));
    }
    Ok(&bytes[HEADER_BYTES..])
}

pub(crate) fn digest(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

const MANIFEST_DOMAIN: &[u8] = b"grafeo/backup-v2/manifest";
const CURSOR_DOMAIN: &[u8] = b"grafeo/backup-v2/cursor";
const SEGMENT_METADATA_DOMAIN: &[u8] = b"grafeo/backup-v2/segment-metadata";
const MAX_SEGMENTS: u64 = 1_000_000;
const MAX_FILENAME_BYTES: usize = 4096;

/// The committed identity shared by a manifest and its cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BackupGeneration {
    pub(super) chain_id: [u8; 32],
    pub(super) generation: u64,
    pub(super) end_sequence: u64,
    pub(super) end_epoch: EpochId,
    pub(super) manifest_digest: [u8; 32],
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn tag(&mut self, tag: u8) {
        self.bytes.push(tag);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn fixed(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn bytes(&mut self, value: &[u8]) -> Result<()> {
        let length = u64::try_from(value.len())
            .map_err(|_| Error::Serialization("backup field length overflow".into()))?;
        let required = self
            .bytes
            .len()
            .checked_add(8)
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| Error::Serialization("backup metadata payload size overflow".into()))?;
        if required > MAX_PAYLOAD_BYTES.saturating_sub(32) {
            return Err(Error::Serialization(
                "backup metadata payload is too large".into(),
            ));
        }
        self.u64(length);
        self.fixed(value);
        Ok(())
    }

    fn finish(self) -> Result<Vec<u8>> {
        if self.bytes.len() > MAX_PAYLOAD_BYTES.saturating_sub(32) {
            return Err(Error::Serialization(
                "backup metadata payload is too large".into(),
            ));
        }
        Ok(self.bytes)
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(Error::Serialization(
                "backup metadata payload is too large".into(),
            ));
        }
        Ok(Self { bytes, offset: 0 })
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| Error::Serialization("backup metadata length overflow".into()))?;
        if end > self.bytes.len() {
            return Err(Error::Serialization("backup metadata is truncated".into()));
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn tag(&mut self, expected: u8) -> Result<()> {
        let actual = self.u8()?;
        if actual != expected {
            return Err(Error::Serialization(format!(
                "unexpected backup metadata tag {actual}, expected {expected}"
            )));
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(*self
            .take(1)?
            .first()
            .ok_or_else(|| Error::Serialization("backup metadata is truncated".into()))?)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(
            |_| Error::Serialization("backup metadata integer is truncated".into()),
        )?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(
            |_| Error::Serialization("backup metadata integer is truncated".into()),
        )?))
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N)?
            .try_into()
            .map_err(|_| Error::Serialization("backup metadata fixed field is truncated".into()))
    }

    fn bytes(&mut self, maximum: usize) -> Result<Vec<u8>> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| Error::Serialization("backup metadata length overflows usize".into()))?;
        if length > maximum {
            return Err(Error::Serialization(
                "backup metadata field is too large".into(),
            ));
        }
        Ok(self.take(length)?.to_vec())
    }

    fn done(&self) -> Result<()> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::Serialization(
                "backup metadata has trailing bytes".into(),
            ))
        }
    }
}

fn with_integrity(kind: u8, domain: &[u8], body: Vec<u8>) -> Result<Vec<u8>> {
    let checksum = digest(domain, &body);
    let mut payload = body;
    payload.extend_from_slice(&checksum);
    envelope(kind, &payload)
}

fn without_integrity<'a>(kind: u8, domain: &[u8], bytes: &'a [u8]) -> Result<Decoder<'a>> {
    let payload = payload(kind, bytes)?;
    if payload.len() < 32 {
        return Err(Error::Serialization(
            "backup metadata integrity suffix is missing".into(),
        ));
    }
    let split = payload.len() - 32;
    let (body, actual) = payload.split_at(split);
    if digest(domain, body) != actual {
        return Err(Error::Serialization(
            "backup metadata integrity mismatch".into(),
        ));
    }
    Decoder::new(body)
}

fn encode_segment(segment: &super::BackupSegment) -> Result<Vec<u8>> {
    validate_filename(&segment.filename)?;
    let mut out = Encoder::new();
    out.tag(1);
    out.u8(match segment.kind {
        super::BackupKind::Full => 0,
        super::BackupKind::Incremental => 1,
    });
    out.tag(2);
    out.bytes(segment.filename.as_bytes())?;
    out.tag(3);
    out.u64(segment.start_epoch.as_u64());
    out.tag(4);
    out.u64(segment.end_epoch.as_u64());
    out.tag(5);
    out.u32(segment.checksum);
    out.tag(6);
    out.fixed(&segment.content_digest);
    out.tag(7);
    match segment.world_cut.as_deref() {
        Some(value) => {
            out.u8(1);
            out.bytes(value)?;
        }
        None => out.u8(0),
    }
    out.tag(8);
    out.u64(segment.size_bytes);
    out.tag(9);
    out.u64(segment.created_at_ms);
    out.tag(10);
    out.fixed(&segment.chain_id);
    out.tag(11);
    out.fixed(&segment.store_id);
    out.tag(12);
    if segment.model > 2 {
        return Err(Error::Serialization("unknown backup graph model".into()));
    }
    out.u8(segment.model);
    out.tag(13);
    out.u64(segment.sequence);
    out.tag(14);
    out.u64(segment.wal_start_sequence);
    out.tag(15);
    out.u64(segment.wal_end_sequence);
    out.tag(16);
    out.fixed(&segment.predecessor_digest);
    out.tag(17);
    out.u64(segment.record_count);
    out.finish()
}

fn decode_segment(bytes: &[u8]) -> Result<super::BackupSegment> {
    let mut input = Decoder::new(bytes)?;
    input.tag(1)?;
    let kind = match input.u8()? {
        0 => super::BackupKind::Full,
        1 => super::BackupKind::Incremental,
        value => {
            return Err(Error::Serialization(format!(
                "unknown backup segment kind {value}"
            )));
        }
    };
    input.tag(2)?;
    let filename = String::from_utf8(input.bytes(MAX_FILENAME_BYTES)?)
        .map_err(|_| Error::Serialization("backup segment filename is not UTF-8".into()))?;
    validate_filename(&filename)?;
    input.tag(3)?;
    let start_epoch = EpochId::new(input.u64()?);
    input.tag(4)?;
    let end_epoch = EpochId::new(input.u64()?);
    input.tag(5)?;
    let checksum = input.u32()?;
    input.tag(6)?;
    let content_digest = input.fixed()?;
    input.tag(7)?;
    let world_cut = match input.u8()? {
        0 => None,
        1 => Some(input.bytes(MAX_PAYLOAD_BYTES)?),
        value => {
            return Err(Error::Serialization(format!(
                "unknown WorldCut presence tag {value}"
            )));
        }
    };
    input.tag(8)?;
    let size_bytes = input.u64()?;
    input.tag(9)?;
    let created_at_ms = input.u64()?;
    input.tag(10)?;
    let chain_id = input.fixed()?;
    input.tag(11)?;
    let store_id = input.fixed()?;
    input.tag(12)?;
    let model = input.u8()?;
    if model > 2 {
        return Err(Error::Serialization("unknown backup graph model".into()));
    }
    input.tag(13)?;
    let sequence = input.u64()?;
    input.tag(14)?;
    let wal_start_sequence = input.u64()?;
    input.tag(15)?;
    let wal_end_sequence = input.u64()?;
    input.tag(16)?;
    let predecessor_digest = input.fixed()?;
    input.tag(17)?;
    let record_count = input.u64()?;
    input.done()?;
    Ok(super::BackupSegment {
        chain_id,
        store_id,
        model,
        sequence,
        wal_start_sequence,
        wal_end_sequence,
        predecessor_digest,
        record_count,
        kind,
        filename,
        start_epoch,
        end_epoch,
        checksum,
        content_digest,
        world_cut,
        size_bytes,
        created_at_ms,
    })
}

pub(crate) fn encode_manifest(manifest: &super::BackupManifest) -> Result<Vec<u8>> {
    if manifest.version != VERSION {
        return Err(Error::Serialization(
            "unsupported backup manifest version".into(),
        ));
    }
    if manifest.model > 2 {
        return Err(Error::Serialization("unknown backup graph model".into()));
    }
    validate_manifest_segments(&manifest.segments)?;
    encode_manifest_body(manifest)
}

// Hostile-input fixtures need valid wire checksums around invalid topology.
#[cfg(all(test, feature = "lpg", feature = "gql"))]
pub(super) fn encode_unchecked_test_manifest(manifest: &super::BackupManifest) -> Result<Vec<u8>> {
    encode_manifest_body(manifest)
}

fn encode_manifest_body(manifest: &super::BackupManifest) -> Result<Vec<u8>> {
    let mut out = Encoder::new();
    out.tag(1);
    out.u32(manifest.version);
    out.tag(2);
    out.fixed(&manifest.chain_id);
    out.tag(3);
    out.fixed(&manifest.store_id);
    out.tag(4);
    out.u8(manifest.model);
    out.tag(5);
    out.u64(manifest.generation);
    out.tag(6);
    match manifest.world_cut.as_deref() {
        Some(value) => {
            out.u8(1);
            out.bytes(value)?;
        }
        None => out.u8(0),
    }
    out.tag(7);
    let count = u64::try_from(manifest.segments.len())
        .map_err(|_| Error::Serialization("backup segment count overflow".into()))?;
    if count > MAX_SEGMENTS {
        return Err(Error::Serialization(
            "backup segment count is too large".into(),
        ));
    }
    out.u64(count);
    for segment in &manifest.segments {
        let encoded = encode_segment(segment)?;
        out.bytes(&encoded)?;
    }
    let body = out.finish()?;
    with_integrity(KIND_MANIFEST, MANIFEST_DOMAIN, body)
}

pub(crate) fn decode_manifest(bytes: &[u8]) -> Result<super::BackupManifest> {
    let mut input = without_integrity(KIND_MANIFEST, MANIFEST_DOMAIN, bytes)?;
    input.tag(1)?;
    let version = input.u32()?;
    if version != VERSION {
        return Err(Error::Serialization(format!(
            "unsupported backup manifest version {version}"
        )));
    }
    input.tag(2)?;
    let chain_id = input.fixed()?;
    input.tag(3)?;
    let store_id = input.fixed()?;
    input.tag(4)?;
    let model = input.u8()?;
    if model > 2 {
        return Err(Error::Serialization("unknown backup graph model".into()));
    }
    input.tag(5)?;
    let generation = input.u64()?;
    input.tag(6)?;
    let world_cut = match input.u8()? {
        0 => None,
        1 => Some(input.bytes(MAX_PAYLOAD_BYTES)?),
        value => {
            return Err(Error::Serialization(format!(
                "unknown WorldCut presence tag {value}"
            )));
        }
    };
    input.tag(7)?;
    let count = input.u64()?;
    if count > MAX_SEGMENTS {
        return Err(Error::Serialization(
            "backup segment count is too large".into(),
        ));
    }
    let mut segments = Vec::new();
    for _ in 0..count {
        segments.push(decode_segment(&input.bytes(MAX_PAYLOAD_BYTES)?)?);
    }
    input.done()?;
    let manifest = super::BackupManifest {
        version,
        chain_id,
        store_id,
        model,
        generation,
        world_cut,
        segments,
    };
    validate_manifest_segments(&manifest.segments)?;
    Ok(manifest)
}

pub(crate) fn encode_cursor(cursor: &super::BackupCursor) -> Result<Vec<u8>> {
    let mut out = Encoder::new();
    out.tag(1);
    out.fixed(&cursor.chain_id);
    out.tag(2);
    out.u64(cursor.generation);
    out.tag(3);
    out.fixed(&cursor.manifest_digest);
    out.tag(4);
    out.u64(cursor.backed_up_epoch.as_u64());
    out.tag(5);
    out.u64(cursor.log_sequence);
    out.tag(6);
    out.u64(cursor.timestamp_ms);
    with_integrity(KIND_CURSOR, CURSOR_DOMAIN, out.finish()?)
}

pub(crate) fn decode_cursor(bytes: &[u8]) -> Result<super::BackupCursor> {
    let mut input = without_integrity(KIND_CURSOR, CURSOR_DOMAIN, bytes)?;
    input.tag(1)?;
    let chain_id = input.fixed()?;
    input.tag(2)?;
    let generation = input.u64()?;
    input.tag(3)?;
    let manifest_digest = input.fixed()?;
    input.tag(4)?;
    let backed_up_epoch = EpochId::new(input.u64()?);
    input.tag(5)?;
    let log_sequence = input.u64()?;
    input.tag(6)?;
    let timestamp_ms = input.u64()?;
    input.done()?;
    Ok(super::BackupCursor {
        chain_id,
        generation,
        manifest_digest,
        backed_up_epoch,
        log_sequence,
        timestamp_ms,
    })
}

pub(crate) fn segment_digest(segment: &super::BackupSegment) -> Result<[u8; 32]> {
    Ok(digest(SEGMENT_METADATA_DOMAIN, &encode_segment(segment)?))
}

pub(super) fn generation_from_manifest(
    manifest: &super::BackupManifest,
) -> Result<BackupGeneration> {
    let last = manifest
        .segments
        .last()
        .ok_or_else(|| Error::Serialization("backup manifest has no committed segments".into()))?;
    Ok(BackupGeneration {
        chain_id: manifest.chain_id,
        generation: manifest.generation,
        end_sequence: last.wal_end_sequence,
        end_epoch: last.end_epoch,
        manifest_digest: digest(MANIFEST_DOMAIN, &encode_manifest(manifest)?),
    })
}

pub(super) fn generation_from_cursor(cursor: &super::BackupCursor) -> BackupGeneration {
    BackupGeneration {
        chain_id: cursor.chain_id,
        generation: cursor.generation,
        end_sequence: cursor.log_sequence,
        end_epoch: cursor.backed_up_epoch,
        manifest_digest: cursor.manifest_digest,
    }
}

pub(super) fn matching_generation(manifest: &BackupGeneration, cursor: &BackupGeneration) -> bool {
    manifest == cursor
}

#[cfg(feature = "lpg")]
pub(crate) fn chain_id() -> Result<[u8; 32]> {
    grafeo_common::types::StoreId::generate()
        .map(|id| id.into_bytes())
        .map_err(|error| {
            Error::Internal(format!("failed to generate backup chain identity: {error}"))
        })
}

pub(crate) fn validate_digest(value: &[u8; 32]) -> Result<()> {
    if value.iter().all(|byte| *byte == 0) {
        return Err(Error::Serialization(
            "backup identity must not be all zero".into(),
        ));
    }
    Ok(())
}

fn validate_kind(kind: u8) -> Result<()> {
    if !matches!(kind, KIND_MANIFEST | KIND_CURSOR) {
        return Err(Error::Serialization(format!(
            "unsupported backup metadata kind {kind}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_filename(filename: &str) -> Result<()> {
    let path = std::path::Path::new(filename);
    if filename.is_empty()
        || filename.len() > MAX_FILENAME_BYTES
        || filename
            .bytes()
            .any(|byte| matches!(byte, b'\\' | b'/' | b':' | 0))
        || path.is_absolute()
        || path.components().count() != 1
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(Error::Serialization(format!(
            "backup segment filename is not a relative single component: {filename:?}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_manifest_segments(segments: &[super::BackupSegment]) -> Result<()> {
    if let Some(first) = segments.first()
        && first.kind != super::BackupKind::Full
    {
        return Err(Error::Serialization(
            "backup chain must begin with a full segment".into(),
        ));
    }
    let mut filenames = HashSet::new();
    let mut previous_end = None;
    for (index, segment) in segments.iter().enumerate() {
        validate_filename(&segment.filename)?;
        if !filenames.insert(segment.filename.as_str()) {
            return Err(Error::Serialization(
                "backup chain contains duplicate segment filenames".into(),
            ));
        }
        if segment.start_epoch == grafeo_common::types::EpochId::PENDING
            || segment.end_epoch == grafeo_common::types::EpochId::PENDING
        {
            return Err(Error::Serialization(format!(
                "backup segment {} uses the reserved PENDING epoch",
                index
            )));
        }
        if segment.start_epoch > segment.end_epoch {
            return Err(Error::Serialization(format!(
                "backup segment {} has inverted epoch bounds",
                index
            )));
        }
        match segment.kind {
            super::BackupKind::Full => {
                if segment.start_epoch != grafeo_common::types::EpochId::INITIAL {
                    return Err(Error::Serialization(
                        "full backup segments must start at the initial epoch".into(),
                    ));
                }
                if let Some(end) = previous_end
                    && segment.end_epoch < end
                {
                    return Err(Error::Serialization(
                        "full backup segment epochs cannot move backwards".into(),
                    ));
                }
            }
            super::BackupKind::Incremental => {
                if let Some(end) = previous_end
                    && segment.start_epoch <= end
                {
                    return Err(Error::Serialization(
                        "backup incremental segments overlap or are out of order".into(),
                    ));
                }
            }
        }
        previous_end = Some(segment.end_epoch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(
        kind: super::super::BackupKind,
        name: &str,
        start: u64,
        end: u64,
    ) -> super::super::BackupSegment {
        super::super::BackupSegment {
            kind,
            filename: name.into(),
            start_epoch: grafeo_common::types::EpochId::new(start),
            end_epoch: grafeo_common::types::EpochId::new(end),
            checksum: 0,
            content_digest: [0; 32],
            world_cut: None,
            size_bytes: 0,
            created_at_ms: 0,
            chain_id: [1; 32],
            store_id: [2; 32],
            model: 0,
            sequence: 0,
            wal_start_sequence: 0,
            wal_end_sequence: 0,
            predecessor_digest: [0; 32],
            record_count: 0,
        }
    }

    fn reseal_manifest(mut encoded: Vec<u8>, body_offset: usize, replacement: &[u8]) -> Vec<u8> {
        let body_end = encoded.len() - 32;
        encoded[body_offset..body_offset + replacement.len()].copy_from_slice(replacement);
        let checksum = digest(MANIFEST_DOMAIN, &encoded[HEADER_BYTES..body_end]);
        encoded[body_end..].copy_from_slice(&checksum);
        encoded
    }

    #[test]
    fn canonical_manifest_and_cursor_round_trip_with_integrity_suffixes() {
        let manifest = super::super::BackupManifest {
            version: VERSION,
            chain_id: [3; 32],
            store_id: [4; 32],
            model: 2,
            generation: 7,
            world_cut: Some(vec![9, 8, 7]),
            segments: vec![segment(super::super::BackupKind::Full, "full", 0, 4)],
        };
        let encoded = encode_manifest(&manifest).unwrap();
        assert_eq!(decode_manifest(&encoded).unwrap().segments.len(), 1);
        let cursor = super::super::BackupCursor {
            chain_id: manifest.chain_id,
            generation: manifest.generation,
            manifest_digest: [5; 32],
            backed_up_epoch: EpochId::new(4),
            log_sequence: 11,
            timestamp_ms: 12,
        };
        let cursor_bytes = encode_cursor(&cursor).unwrap();
        assert_eq!(decode_cursor(&cursor_bytes).unwrap().log_sequence, 11);
        let mut corrupt = cursor_bytes;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode_cursor(&corrupt).is_err());
    }

    #[test]
    fn canonical_manifest_rejects_trailing_outer_bytes() {
        let manifest = super::super::BackupManifest {
            version: VERSION,
            chain_id: [3; 32],
            store_id: [4; 32],
            model: 0,
            generation: 1,
            world_cut: None,
            segments: vec![segment(super::super::BackupKind::Full, "full", 0, 0)],
        };
        let mut encoded = encode_manifest(&manifest).unwrap();
        encoded.push(0);
        assert!(decode_manifest(&encoded).is_err());
    }

    #[test]
    fn canonical_manifest_rejects_resealed_unknown_tag_and_count() {
        let manifest = super::super::BackupManifest {
            version: VERSION,
            chain_id: [3; 32],
            store_id: [4; 32],
            model: 0,
            generation: 1,
            world_cut: None,
            segments: vec![segment(super::super::BackupKind::Full, "full", 0, 0)],
        };
        let encoded = encode_manifest(&manifest).unwrap();
        let unknown_tag = reseal_manifest(encoded.clone(), HEADER_BYTES, &[99]);
        assert!(decode_manifest(&unknown_tag).is_err());

        // Fixed fields before the segment count occupy 85 bytes in the body.
        let count = reseal_manifest(encoded, HEADER_BYTES + 85, &u64::MAX.to_le_bytes());
        assert!(decode_manifest(&count).is_err());
    }

    #[test]
    fn backup_generation_matches_only_when_every_field_matches() {
        let manifest = super::super::BackupManifest {
            version: VERSION,
            chain_id: [3; 32],
            store_id: [4; 32],
            model: 0,
            generation: 7,
            world_cut: None,
            segments: vec![segment(super::super::BackupKind::Full, "full", 0, 4)],
        };
        let generation = generation_from_manifest(&manifest).unwrap();
        let cursor = super::super::BackupCursor {
            chain_id: generation.chain_id,
            generation: generation.generation,
            manifest_digest: generation.manifest_digest,
            backed_up_epoch: generation.end_epoch,
            log_sequence: generation.end_sequence,
            timestamp_ms: 1,
        };
        let from_cursor = generation_from_cursor(&cursor);
        assert!(matching_generation(&generation, &from_cursor));

        let mismatches = [
            BackupGeneration {
                chain_id: [9; 32],
                ..generation.clone()
            },
            BackupGeneration {
                generation: generation.generation + 1,
                ..generation.clone()
            },
            BackupGeneration {
                end_sequence: generation.end_sequence + 1,
                ..generation.clone()
            },
            BackupGeneration {
                end_epoch: EpochId::new(generation.end_epoch.as_u64() + 1),
                ..generation.clone()
            },
            BackupGeneration {
                manifest_digest: [8; 32],
                ..generation
            },
        ];
        for mismatch in mismatches {
            assert!(!matching_generation(&mismatch, &from_cursor));
        }
    }

    #[test]
    fn version_is_rejected_once_the_outer_prefix_is_available() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(VERSION + 1).to_le_bytes());
        assert!(payload(KIND_MANIFEST, &bytes).is_err());
    }

    #[test]
    fn hostile_outer_codec_inputs_are_rejected() {
        assert!(envelope(99, b"x").is_err());
        let mut bytes = envelope(KIND_MANIFEST, b"x").unwrap();
        bytes.push(0);
        assert!(payload(KIND_MANIFEST, &bytes).is_err());
        let mut oversized = MAGIC.to_vec();
        oversized.extend_from_slice(&VERSION.to_le_bytes());
        oversized.push(KIND_MANIFEST);
        oversized.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(payload(KIND_MANIFEST, &oversized).is_err());
    }

    #[test]
    fn filenames_are_portable_single_components() {
        for name in ["a/b", r"a\\b", "a:b", "a\0b", ".", ".."] {
            assert!(validate_filename(name).is_err(), "{name:?}");
        }
        assert!(validate_filename("backup_full_0000.grafeo").is_ok());
    }

    #[test]
    fn manifest_chain_semantics_are_fail_closed() {
        use super::super::BackupKind::{Full, Incremental};
        assert!(validate_manifest_segments(&[]).is_ok());
        assert!(
            validate_manifest_segments(&[
                segment(Full, "f0", 0, 10),
                segment(Incremental, "i", 11, 20),
                segment(Full, "f1", 0, 15),
            ])
            .is_err()
        );
        assert!(
            validate_manifest_segments(&[
                segment(Full, "f0", 0, 10),
                segment(Incremental, "i", 11, 20),
                segment(Full, "f1", 0, 20),
            ])
            .is_ok()
        );
        assert!(validate_manifest_segments(&[segment(Incremental, "i", 1, 2)]).is_err());
        assert!(
            validate_manifest_segments(&[segment(Full, "f0", 0, 10), segment(Full, "f1", 0, 9),])
                .is_err()
        );
        assert!(
            validate_manifest_segments(&[
                segment(Full, "f", 0, 10),
                segment(Incremental, "i", 11, 12),
                segment(Incremental, "i", 13, 14),
            ])
            .is_err()
        );
        assert!(
            validate_manifest_segments(&[
                segment(Full, "f", 0, 10),
                segment(Incremental, "i", 11, 12),
            ])
            .is_ok()
        );
    }

    #[test]
    fn pending_epochs_are_not_committed_chain_bounds() {
        use super::super::BackupKind::Full;
        assert!(validate_manifest_segments(&[segment(Full, "f", 0, u64::MAX),]).is_err());
    }
}
