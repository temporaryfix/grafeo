//! Shared current Catalog7 envelope; physical index installation is separate.

use crate::catalog::{CURRENT_CATALOG_STATE_VERSION, Catalog, CatalogRead, CatalogStateV2};
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Serialize};

pub(super) const CATALOG_SECTION_VERSION: u8 = 7;
const MAX_CATALOG_BYTES: usize = u32::MAX as usize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CatalogPayloadVersion {
    GraphExactV7,
}
#[derive(Serialize, Deserialize)]
pub(super) struct CatalogSnapshotV7 {
    pub(super) version: u8,
    pub(super) state_version: u8,
    pub(super) state: CatalogStateV2,
    pub(super) epoch: u64,
}
pub(super) fn decode_graph_exact_catalog_payload(data: &[u8]) -> Result<CatalogSnapshotV7> {
    classify_catalog_payload(data)?;
    let (snapshot, consumed): (CatalogSnapshotV7, usize) =
        crate::catalog::decode_current_catalog_bounded(data).map_err(Error::Serialization)?;
    if snapshot.version != CATALOG_SECTION_VERSION
        || snapshot.state_version != CURRENT_CATALOG_STATE_VERSION
        || consumed != data.len()
        || snapshot.epoch == grafeo_common::types::EpochId::PENDING.as_u64()
    {
        return Err(Error::Serialization(
            "Catalog7 version, length or committed epoch mismatch".into(),
        ));
    }
    Ok(snapshot)
}
pub(crate) fn classify_catalog_payload(data: &[u8]) -> Result<CatalogPayloadVersion> {
    if data.first() != Some(&CATALOG_SECTION_VERSION) {
        return Err(Error::Serialization(
            "unsupported catalog section generation; expected 7".into(),
        ));
    }
    if data.get(1) != Some(&CURRENT_CATALOG_STATE_VERSION) || u32::try_from(data.len()).is_err() {
        return Err(Error::Serialization(
            "Catalog7 state version or payload budget mismatch".into(),
        ));
    }
    Ok(CatalogPayloadVersion::GraphExactV7)
}
#[cfg(all(test, feature = "lpg"))]
pub(crate) fn graph_exact_catalog_epoch(data: &[u8]) -> Result<u64> {
    let snapshot = decode_graph_exact_catalog_payload(data)?;
    Catalog::from_current_state_v2(snapshot.state).map_err(Error::Serialization)?;
    Ok(snapshot.epoch)
}
/// Decodes the authoritative catalog once, before any recovery mutation.
pub(super) fn decode_catalog(data: &[u8], expected_epoch: u64) -> Result<Catalog> {
    let snapshot = decode_graph_exact_catalog_payload(data)?;
    if snapshot.epoch != expected_epoch {
        return Err(Error::Serialization(
            "Catalog7 epoch does not match enclosing snapshot".into(),
        ));
    }
    Catalog::from_current_state_v2(snapshot.state).map_err(Error::Serialization)
}

/// Encodes the catalog envelope independently of physical publication.
/// Callers validating artifacts must separately qualify the graph targets.
pub(super) fn encode_catalog_read(catalog: CatalogRead<'_>, epoch: u64) -> Result<Vec<u8>> {
    if epoch == grafeo_common::types::EpochId::PENDING.as_u64() {
        return Err(Error::Serialization(
            "Catalog7 epoch cannot be pending".into(),
        ));
    }
    let snapshot = CatalogSnapshotV7 {
        version: CATALOG_SECTION_VERSION,
        state_version: CURRENT_CATALOG_STATE_VERSION,
        state: catalog.current_state_v2().map_err(Error::Serialization)?,
        epoch,
    };
    bincode::serde::encode_to_vec(
        &snapshot,
        bincode::config::standard().with_limit::<MAX_CATALOG_BYTES>(),
    )
    .map_err(|error| Error::Serialization(format!("cannot encode Catalog7: {error}")))
}
