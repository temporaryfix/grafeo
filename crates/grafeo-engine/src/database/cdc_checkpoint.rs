//! Current CDC section envelope, including an explicit disabled image in builds
//! without CDC. The native payload is prepared before any restore publication.
use super::GrafeoDB;
use grafeo_common::types::{EpochId, StoreId};
use grafeo_common::utils::error::{Error, Result};

pub(super) const MAGIC: &[u8; 8] = b"GCDCCHK1";
const HEADER: usize = 57;
const MAX_BYTES: usize = 64 * 1024 * 1024;

fn invalid(message: &str) -> Error {
    Error::Serialization(format!("CDC section: {message}"))
}

fn envelope(store: StoreId, epoch: EpochId, body: &[u8]) -> Result<Vec<u8>> {
    if body.len() > MAX_BYTES {
        return Err(invalid("image exceeds 64 MiB"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve(HEADER + body.len())
        .map_err(|_| invalid("allocation failed"))?;
    bytes.extend_from_slice(MAGIC);
    bytes.push(u8::from(!body.is_empty()));
    bytes.extend_from_slice(store.as_bytes());
    bytes.extend_from_slice(&epoch.as_u64().to_le_bytes());
    bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
    bytes.extend_from_slice(body);
    Ok(bytes)
}

fn body(data: &[u8], store: StoreId, epoch: EpochId) -> Result<&[u8]> {
    if data.len() < HEADER || data.len() > HEADER + MAX_BYTES || !data.starts_with(MAGIC) {
        return Err(invalid("unsupported schema or bounded length"));
    }
    let mut epoch_bytes = [0; 8];
    epoch_bytes.copy_from_slice(&data[41..49]);
    let mut length_bytes = [0; 8];
    length_bytes.copy_from_slice(&data[49..57]);
    let length = u64::from_le_bytes(length_bytes);
    if &data[9..41] != store.as_bytes()
        || u64::from_le_bytes(epoch_bytes) != epoch.as_u64()
        || epoch == EpochId::PENDING
        || length != (data.len() - HEADER) as u64
        || !matches!((data[8], length), (0, 0) | (1, 1..))
    {
        return Err(invalid("foreign identity/cut, invalid flags or length"));
    }
    Ok(&data[HEADER..])
}

pub(super) fn capture(db: &GrafeoDB, epoch: EpochId) -> Result<Vec<u8>> {
    capture_state(
        db.world_identity().store_id(),
        epoch,
        #[cfg(feature = "cdc")]
        &db.cdc_log,
    )
}

pub(super) fn capture_state(
    store: StoreId,
    epoch: EpochId,
    #[cfg(feature = "cdc")] log: &crate::cdc::CdcLog,
) -> Result<Vec<u8>> {
    #[cfg(feature = "cdc")]
    let bytes = if log.has_checkpoint_state() {
        crate::cdc::checkpoint::encode(log)?
    } else {
        Vec::new()
    };
    #[cfg(not(feature = "cdc"))]
    let bytes = Vec::new();
    envelope(store, epoch, &bytes)
}

pub(super) fn validate(data: &[u8], store: StoreId, epoch: EpochId) -> Result<()> {
    let bytes = body(data, store, epoch)?;
    if !bytes.is_empty() {
        #[cfg(feature = "cdc")]
        let _ = crate::cdc::checkpoint::prepare(bytes, epoch)?;
        #[cfg(not(feature = "cdc"))]
        return Err(invalid("retained feed requires CDC support"));
    }
    Ok(())
}

#[cfg(feature = "cdc")]
pub(super) fn prepare(
    data: &[u8],
    store: StoreId,
    epoch: EpochId,
) -> Result<crate::cdc::checkpoint::PreparedRestore> {
    let bytes = body(data, store, epoch)?;
    if bytes.is_empty() {
        Ok(crate::cdc::checkpoint::empty())
    } else {
        crate::cdc::checkpoint::prepare(bytes, epoch)
    }
}

#[cfg(any(feature = "lpg", feature = "triple-store"))]
pub(super) fn reidentify(
    data: &[u8],
    source: StoreId,
    target: StoreId,
    epoch: EpochId,
) -> Result<Vec<u8>> {
    validate(data, source, epoch)?;
    envelope(target, epoch, body(data, source, epoch)?)
}

#[cfg(feature = "lpg")]
pub(super) fn is_enabled(data: &[u8]) -> bool {
    data.get(8) == Some(&1)
}

#[cfg(feature = "grafeo-file")]
pub(super) struct CapturedSection(pub(super) Vec<u8>);
#[cfg(feature = "grafeo-file")]
impl grafeo_common::storage::Section for CapturedSection {
    fn section_type(&self) -> grafeo_common::storage::SectionType {
        grafeo_common::storage::SectionType::Cdc
    }
    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.0.clone())
    }
    fn deserialize(&mut self, _: &[u8]) -> Result<()> {
        Err(invalid("capture wrapper cannot restore"))
    }
    fn is_dirty(&self) -> bool {
        true
    }
    fn mark_clean(&self) {}
    fn memory_usage(&self) -> usize {
        self.0.len()
    }
}

#[cfg(all(feature = "grafeo-file", feature = "cdc"))]
pub(super) fn restore_section(
    image: &[super::world_metadata::EncodedSection],
    metadata: Option<&super::world_metadata::VerifiedWorldMetadata>,
    log: &crate::cdc::CdcLog,
) -> Result<()> {
    let cut = metadata
        .ok_or_else(|| invalid("missing world authority"))?
        .cut();
    let section =
        super::world_metadata::find_section(image, grafeo_common::storage::SectionType::Cdc)
            .ok_or_else(|| invalid("missing required section"))?;
    prepare(section.bytes(), cut.store_id(), cut.epoch())?.install(log);
    Ok(())
}

#[cfg(all(test, feature = "lpg"))]
pub(super) fn disabled(store: StoreId, epoch: EpochId) -> Result<Vec<u8>> {
    envelope(store, epoch, &[])
}

#[cfg(all(test, not(feature = "cdc")))]
#[test]
fn retained_feed_is_rejected_without_cdc_support() {
    let store = StoreId::from_bytes([0x42; StoreId::LEN]).unwrap();
    let epoch = EpochId::INITIAL;
    validate(&envelope(store, epoch, &[]).unwrap(), store, epoch).unwrap();
    let bytes = envelope(store, epoch, &[1]).unwrap();
    assert!(
        validate(&bytes, store, epoch)
            .unwrap_err()
            .to_string()
            .contains("requires CDC support")
    );
}

#[cfg(all(test, feature = "cdc", feature = "lpg"))]
#[test]
fn empty_feed_generation_and_clock_authority_survive_exact_recapture() {
    use grafeo_common::types::HlcTimestamp;
    for (generation, high) in [
        (2u64, 0u64),
        (
            1,
            HlcTimestamp::new(HlcTimestamp::now().physical_ms() + 3_600_000, 7).as_u64(),
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let epoch = db.current_epoch();
        let mut payload = Vec::new();
        for value in [generation, 1, 1, high] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        payload.extend_from_slice(&0u32.to_le_bytes());
        let bytes = envelope(db.store_id(), epoch, &payload).unwrap();
        prepare(&bytes, db.store_id(), epoch)
            .unwrap()
            .install(&db.cdc_log);
        assert_eq!(capture(&db, epoch).unwrap(), bytes);
    }
}
