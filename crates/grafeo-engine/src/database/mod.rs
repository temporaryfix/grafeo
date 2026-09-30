//! The main database struct and operations.
//!
//! Start here with [`GrafeoDB`] - it's your handle to everything.
//!
//! Operations are split across focused submodules:
//! - `query` - Query execution (execute, execute_cypher, etc.)
//! - `crud` - Node/edge CRUD operations
//! - `index` - Property, vector, and text index management
//! - `search` - Vector, text, and hybrid search
//! - `embed` - Embedding model management
//! - `persistence` - Save, load, snapshots, iteration
//! - `admin` - Stats, introspection, diagnostics, CDC

#[cfg(feature = "lpg")]
mod admin;
#[cfg(feature = "arrow-export")]
pub mod arrow;
#[cfg(all(feature = "async-storage", feature = "lpg"))]
mod async_ops;
#[cfg(any(feature = "wal", all(test, feature = "lpg", feature = "grafeo-file")))]
mod atomic_install;
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
pub mod backup;
#[cfg(feature = "lpg")]
pub(crate) mod catalog_section;
#[cfg(feature = "grafeo-file")]
mod catalog_state_section;
#[cfg(any(feature = "lpg", feature = "triple-store"))]
mod catalog_wire;
#[cfg(feature = "cdc")]
mod cdc_api;
#[cfg(all(feature = "cdc", feature = "lpg"))]
pub(crate) mod cdc_store;
#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
mod checkpoint_timer;
#[cfg(all(
    test,
    feature = "compact-store",
    feature = "lpg",
    feature = "grafeo-file"
))]
mod compact_maintenance_tests;
#[cfg(all(feature = "compact-store", feature = "mmap"))]
pub mod compact_tiered;
#[cfg(feature = "lpg")]
mod crud;
#[cfg(feature = "embed")]
mod embed;
#[cfg(feature = "grafeo-file")]
pub(crate) mod flush;
#[cfg(feature = "lpg")]
mod import;
#[cfg(feature = "lpg")]
mod index;
mod info;
#[cfg(feature = "lpg")]
pub use index::{CreateIndexRequest, IndexCreateKind};
mod cdc_checkpoint;
#[cfg(all(
    test,
    feature = "lpg",
    feature = "wal",
    feature = "grafeo-file",
    feature = "text-index",
    feature = "vector-index"
))]
mod current_index_persistence_tests;
#[cfg(all(feature = "wal", feature = "lpg"))]
mod graph_replay;
#[cfg(all(feature = "wal", feature = "lpg"))]
pub(crate) mod index_commit_wire;
#[cfg(feature = "lpg")]
mod index_sections;
#[cfg(all(feature = "wal", feature = "lpg"))]
mod label_replay;
#[cfg(all(feature = "wal", feature = "lpg"))]
mod owner_replay;
#[cfg(any(feature = "lpg", feature = "triple-store"))]
mod persistence;
#[cfg(any(test, feature = "lpg", feature = "triple-store"))]
mod portable_wire;
#[cfg(all(
    test,
    feature = "lpg",
    feature = "wal",
    feature = "grafeo-file",
    feature = "text-index",
    feature = "vector-index"
))]
mod recursive_capture_tests;
#[cfg(feature = "wal")]
mod save;
#[cfg(feature = "lpg")]
pub use persistence::{
    IndexMergePolicy, OpenMultiOptions, SchemaMergePolicy, SnapshotInfo, snapshot_info,
};
mod query;
#[cfg(feature = "triple-store")]
mod rdf_ops;
mod read_views;
#[cfg(feature = "lpg")]
mod search;
pub(crate) mod section_consumer;
#[cfg(feature = "statement-table")]
mod statements;
#[cfg(feature = "lpg")]
#[doc(hidden)]
pub mod testing;
#[cfg(all(feature = "wal", feature = "lpg"))]
pub(crate) mod wal_store;
#[cfg(feature = "grafeo-file")]
mod world_metadata;

#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
pub use read_views::CompactStoreTieredView;
#[cfg(feature = "grafeo-file")]
pub use read_views::DatabaseFileView;
#[cfg(all(feature = "compact-store", feature = "lpg"))]
pub use read_views::LayeredStoreView;

use grafeo_common::grafeo_error;
#[cfg(any(feature = "wal", feature = "grafeo-file"))]
use grafeo_common::grafeo_warn;
#[cfg(feature = "wal")]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use parking_lot::RwLock;

use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig, MemoryGrant};
#[cfg(feature = "wal")]
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::types::{HistoryCompleteness, StoreId, WorldIdentityMetadataV1};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result, TransactionError};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::{LpgStore, TransportEdgeReceipt};
#[cfg(feature = "triple-store")]
use grafeo_core::graph::rdf::RdfStore;
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;
#[cfg(all(feature = "lpg", feature = "wal"))]
use grafeo_storage::wal::LpgMutationOp;
#[cfg(feature = "wal")]
use grafeo_storage::wal::WalRecovery;
#[cfg(feature = "wal")]
use grafeo_storage::wal::{
    DurabilityMode as WalDurabilityMode, LpgWal, WalConfig, WalEntry, WalRecord,
};

fn fresh_world_identity() -> Result<WorldIdentityMetadataV1> {
    let store_id = StoreId::generate().map_err(|error| {
        Error::Internal(format!(
            "failed to generate database identity from system entropy: {error}"
        ))
    })?;
    WorldIdentityMetadataV1::new(store_id, HistoryCompleteness::Complete)
        .map_err(|error| Error::Internal(format!("invalid fresh database identity: {error}")))
}

/// Validates a persisted transaction high-water mark before it can influence
/// the allocator. This is intentionally separate from WAL replay coverage:
/// even an unsealed legacy header is useful as a conservative allocation hint,
/// but it never authorizes discarding data records.
#[cfg(any(feature = "wal", feature = "grafeo-file"))]
fn validated_transaction_allocation_floor(
    raw: u64,
    source: &str,
) -> Result<Option<grafeo_common::types::TransactionId>> {
    if raw == 0 {
        return Ok(None);
    }
    if raw >= u64::MAX - 1 {
        return Err(Error::Storage(
            grafeo_common::utils::error::StorageError::Corruption(format!(
                "{source} transaction high-water {raw} is at or beyond the safe allocator limit"
            )),
        ));
    }
    Ok(Some(grafeo_common::types::TransactionId::new(raw)))
}

/// Checks the cardinality coordinates authenticated by the recovery-image
/// seal. Legacy section-only/v1/absent images deliberately bypass this gate:
/// their raw header counts are compatibility hints, not integrity claims.
#[cfg(feature = "grafeo-file")]
fn validate_sealed_recovery_cardinality(
    header: &grafeo_storage::file::DbHeader,
    recovery_coordinates_are_sealed: bool,
    actual_nodes: usize,
    actual_edges: usize,
    model: &str,
) -> Result<()> {
    if !recovery_coordinates_are_sealed {
        return Ok(());
    }
    let actual_nodes = u64::try_from(actual_nodes).map_err(|_| {
        Error::Serialization(format!(
            "{model} node count does not fit the recovery header"
        ))
    })?;
    let actual_edges = u64::try_from(actual_edges).map_err(|_| {
        Error::Serialization(format!(
            "{model} edge count does not fit the recovery header"
        ))
    })?;
    if header.node_count != actual_nodes || header.edge_count != actual_edges {
        return Err(Error::Storage(
            grafeo_common::utils::error::StorageError::Corruption(format!(
                "coordinate-sealed {model} cardinality {actual_nodes} nodes/{actual_edges} edges disagrees with recovery header {}/{}",
                header.node_count, header.edge_count
            )),
        ));
    }
    Ok(())
}

#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
fn validate_default_lpg_recovery_cardinality(
    header: &grafeo_storage::file::DbHeader,
    recovery_image_is_sealed: bool,
    recovery_coordinates_are_sealed: bool,
    store: &LpgStore,
    #[cfg(feature = "compact-store")] layered: Option<
        &Arc<grafeo_core::graph::compact::layered::LayeredStore>,
    >,
) -> Result<()> {
    if recovery_coordinates_are_sealed && !recovery_image_is_sealed {
        return Err(Error::Serialization(
            "recovery header coordinates cannot be sealed by an unsealed LPG section image"
                .to_string(),
        ));
    }
    if !recovery_coordinates_are_sealed {
        return Ok(());
    }
    #[cfg(feature = "compact-store")]
    let (actual_nodes, actual_edges) = layered.map_or_else(
        || (store.node_count(), store.edge_count()),
        |layered| {
            (
                grafeo_core::graph::GraphStore::node_count(layered.as_ref()),
                grafeo_core::graph::GraphStore::edge_count(layered.as_ref()),
            )
        },
    );
    #[cfg(not(feature = "compact-store"))]
    let (actual_nodes, actual_edges) = (store.node_count(), store.edge_count());

    validate_sealed_recovery_cardinality(
        header,
        true,
        actual_nodes,
        actual_edges,
        "default LPG view",
    )
}

#[cfg(feature = "wal")]
#[derive(Clone, Debug)]
struct AuthenticatedContainerIdentity {
    identity: WorldIdentityMetadataV1,
    graph_model: crate::config::GraphModel,
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
impl AuthenticatedContainerIdentity {
    /// Called only at the end of successful detached section decoding. A
    /// current verified cut, not the populated store or replay flags, supplies
    /// the identity and model. Predecessor images cannot authorize WAL tails.
    fn from_decoded_image(
        metadata: Option<&world_metadata::VerifiedWorldMetadata>,
    ) -> Result<Option<Self>> {
        let Some(world_metadata::VerifiedWorldMetadata::RecoverySealedV2(metadata)) = metadata
        else {
            return Ok(None);
        };
        let cut = metadata.cut();
        let identity = WorldIdentityMetadataV1::new(cut.store_id(), cut.descriptor().history())
            .map_err(|error| {
                Error::Serialization(format!("invalid authenticated container identity: {error}"))
            })?;
        let graph_model =
            crate::config::GraphModel::from_u8(cut.descriptor().graph_model().as_u8()).ok_or_else(
                || Error::Serialization("invalid authenticated container graph model".to_string()),
            )?;
        Ok(Some(Self {
            identity,
            graph_model,
        }))
    }
}

#[cfg(feature = "wal")]
struct WalAuthority {
    identity: Option<WorldIdentityMetadataV1>,
    graph_model: crate::config::GraphModel,
}

#[cfg(feature = "wal")]
struct RecoveredWal {
    records: Vec<WalRecord>,
    authority: WalAuthority,
}

/// Resolve authority without changing the model stores or configuration.
#[cfg(feature = "wal")]
fn resolve_wal_identity(
    records: &[WalRecord],
    has_prior_state: bool,
    checkpoint: Option<&AuthenticatedContainerIdentity>,
) -> Result<Option<WorldIdentityMetadataV1>> {
    let mut declared = None;
    for record in records {
        if let WalRecord::StoreIdentityMeta { metadata } = record {
            if declared.is_some_and(|known| known != metadata) {
                return Err(Error::Serialization(
                    "WAL contains conflicting logical-store identity metadata".to_string(),
                ));
            }
            declared = Some(metadata);
        }
    }
    match (declared, checkpoint) {
        (Some(identity), Some(checkpoint)) if identity != &checkpoint.identity => {
            Err(Error::Serialization(
                "WAL identity disagrees with authenticated container identity".to_string(),
            ))
        }
        (Some(identity), _) => Ok(Some(identity.clone())),
        (None, Some(checkpoint)) => Ok(Some(checkpoint.identity.clone())),
        (None, None) if !has_prior_state && records.is_empty() => Ok(None),
        (None, None) => Err(Error::Serialization(
            "nonempty WAL has no authenticated store identity".to_string(),
        )),
    }
}

#[cfg(feature = "wal")]
fn resolve_wal_authority(
    config: &Config,
    records: &[WalRecord],
    has_prior_state: bool,
    checkpoint: Option<&AuthenticatedContainerIdentity>,
) -> Result<WalAuthority> {
    let identity = resolve_wal_identity(records, has_prior_state, checkpoint)?;
    if checkpoint.is_none()
        && (has_prior_state || !records.is_empty())
        && !records
            .iter()
            .any(|record| matches!(record, WalRecord::GraphModelMeta { .. }))
    {
        return Err(Error::Serialization(
            "nonempty WAL has no durable graph model declaration".to_string(),
        ));
    }
    let mut resolved = config.clone();
    GrafeoDB::adopt_stored_graph_model(&mut resolved, records)?;
    if let Some(checkpoint) = checkpoint {
        if records.iter().any(|record| matches!(record, WalRecord::GraphModelMeta { model } if *model != checkpoint.graph_model.as_u8())) {
            return Err(Error::Serialization("WAL graph model disagrees with authenticated container".to_string()));
        }
        if config.graph_model_pinned && config.graph_model != checkpoint.graph_model {
            return Err(Error::InvalidValue(
                "requested graph model disagrees with authenticated container".to_string(),
            ));
        }
        resolved.graph_model = checkpoint.graph_model;
    }
    resolved.validate_resolved_graph_model().map_err(|error| {
        Error::Serialization(format!("unsupported recovered graph model: {error}"))
    })?;
    Ok(WalAuthority {
        identity,
        graph_model: resolved.graph_model,
    })
}

#[cfg(feature = "wal")]
fn recover_committed_wal(
    recovery: &mut WalRecovery,
    tm: &TransactionManager,
    config: &Config,
    #[cfg(feature = "cdc")] cdc_source: crate::cdc::wal::RecoverySource<'_>,
    checkpoint: Option<&AuthenticatedContainerIdentity>,
    #[cfg(feature = "grafeo-file")] snapshot: Option<SnapshotWalRecovery>,
) -> Result<RecoveredWal> {
    let (recovered, floor) = recovery.recover_validated(|report, has_prior_state, prefix| {
        #[cfg(feature = "grafeo-file")]
        if has_prior_state
            && snapshot
                .as_ref()
                .is_some_and(|snapshot| !snapshot.header.is_empty())
            && checkpoint.is_none()
        {
            return Err(Error::Serialization(
                "cannot bind WAL to an unauthenticated container image".to_string(),
            ));
        }
        let authority =
            resolve_wal_authority(config, &report.committed, has_prior_state, checkpoint)?;
        // Prefix declarations constrain already-resolved suffix/container
        // authority; they may never create that authority or enter replay.
        for record in prefix {
            let agrees = match record {
                WalRecord::StoreIdentityMeta { metadata } => {
                    authority.identity.as_ref() == Some(metadata)
                }
                WalRecord::GraphModelMeta { model } => *model == authority.graph_model.as_u8(),
                _ => false,
            };
            if !agrees {
                return Err(Error::Serialization(
                    "retained pre-checkpoint WAL identity/model disagrees with recovery authority"
                        .to_string(),
                ));
            }
        }
        let floor = report
            .max_transaction_id
            .map(|tid| validated_transaction_allocation_floor(tid.as_u64(), "WAL recovery"))
            .transpose()?
            .flatten();
        let records = report.committed;
        #[cfg(feature = "grafeo-file")]
        let records = match snapshot {
            Some(snapshot) if !records.is_empty() && !snapshot.header.is_empty() => {
                let evidence = snapshot.evidence.ok_or_else(|| {
                    Error::Serialization("container has no decoded WAL replay evidence".to_string())
                })?;
                let floor = evidence.resolve(&snapshot.header)?;
                wal_records_after_snapshot(records, floor.epoch, floor.transaction_id)?
            }
            _ => records,
        };
        #[cfg(feature = "cdc")]
        crate::cdc::wal::recover(cdc_source, &records)?;
        #[cfg(not(feature = "cdc"))]
        if records.iter().any(|record| {
            matches!(
                record,
                WalRecord::CdcBatch { .. }
                    | WalRecord::CommittedWithCdc { .. }
                    | WalRecord::CdcRetention { .. }
            )
        }) {
            return Err(Error::Serialization("WAL feed requires CDC support".into()));
        }
        Ok((RecoveredWal { records, authority }, floor))
    })?;
    if let Some(floor) = floor {
        tm.advance_next_transaction_id(floor);
    }
    Ok(recovered)
}

/// Authenticated boundary between an installed single-file image and its
/// sidecar WAL.
///
/// The epoch comes from the decoded, checksummed model payload. The
/// transaction coordinate is present only for a coordinate-sealed metadata
/// generation, which binds that header field into the recovery-image digest.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotReplayFloor {
    epoch: EpochId,
    transaction_id: TransactionId,
}

/// Evidence retained after a snapshot decoder succeeds but before the engine
/// knows whether a sidecar WAL actually needs a replay boundary.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotReplayEvidence {
    authenticated_epoch: Option<EpochId>,
    recovery_coordinates_are_sealed: bool,
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
struct SnapshotWalRecovery {
    evidence: Option<SnapshotReplayEvidence>,
    header: grafeo_storage::file::DbHeader,
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
impl SnapshotReplayEvidence {
    fn resolve(self, header: &grafeo_storage::file::DbHeader) -> Result<SnapshotReplayFloor> {
        let decoded_epoch = self.authenticated_epoch.ok_or_else(|| {
            Error::Serialization(
                "cannot establish a sidecar WAL replay floor: this legacy snapshot generation carries no authenticated epoch"
                    .to_string(),
            )
        })?;
        snapshot_replay_floor_from_decoded_image(
            header,
            decoded_epoch,
            self.recovery_coordinates_are_sealed,
        )
    }
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
fn snapshot_replay_floor_from_decoded_image(
    header: &grafeo_storage::file::DbHeader,
    decoded_epoch: EpochId,
    recovery_coordinates_are_sealed: bool,
) -> Result<SnapshotReplayFloor> {
    if header.is_empty() {
        return Err(Error::Serialization(
            "cannot derive a replay floor from an unpublished container image".to_string(),
        ));
    }
    if decoded_epoch == EpochId::PENDING {
        return Err(Error::Serialization(
            "container model payload carries the reserved pending epoch".to_string(),
        ));
    }
    if decoded_epoch.as_u64() != header.epoch {
        return Err(Error::Storage(
            grafeo_common::utils::error::StorageError::Corruption(format!(
                "decoded snapshot epoch {} disagrees with container header epoch {}",
                decoded_epoch.as_u64(),
                header.epoch
            )),
        ));
    }

    let transaction_id = if recovery_coordinates_are_sealed {
        validated_transaction_allocation_floor(
            header.transaction_id,
            "coordinate-sealed container header",
        )?
        // Zero is the authenticated pre-user-transaction frontier. Retaining
        // it as a valid floor lets a later legacy TID-bearing group be kept
        // without pretending that the image had already committed a user TID.
        .unwrap_or_else(|| TransactionId::new(0))
    } else {
        // WorldMetadata v1, legacy section-only v2, and metadata-absent images
        // authenticate no header transaction coordinate. Never turn that
        // untrusted value into a skip boundary, even when its numeric value
        // looks plausible.
        TransactionId::INVALID
    };
    Ok(SnapshotReplayFloor {
        epoch: decoded_epoch,
        transaction_id,
    })
}

#[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
fn decoded_snapshot_epoch(
    graph_model: crate::config::GraphModel,
    lpg_store: &LpgStore,
    #[cfg(feature = "triple-store")] rdf_store: &RdfStore,
) -> Result<EpochId> {
    match graph_model {
        crate::config::GraphModel::Lpg => Ok(lpg_store.current_epoch()),
        #[cfg(feature = "triple-store")]
        crate::config::GraphModel::Rdf => Ok(rdf_store.commit_epoch()),
        #[cfg(feature = "triple-store")]
        crate::config::GraphModel::Both => {
            let lpg_epoch = lpg_store.current_epoch();
            let rdf_epoch = rdf_store.commit_epoch();
            if lpg_epoch != rdf_epoch {
                return Err(Error::Storage(
                    grafeo_common::utils::error::StorageError::Corruption(format!(
                        "mixed snapshot has divergent LPG epoch {} and RDF epoch {}",
                        lpg_epoch.as_u64(),
                        rdf_epoch.as_u64()
                    )),
                ));
            }
            Ok(lpg_epoch)
        }
        #[cfg(not(feature = "triple-store"))]
        crate::config::GraphModel::Rdf | crate::config::GraphModel::Both => Err(
            Error::Serialization("RDF snapshot requires triple-store support".to_string()),
        ),
    }
}

#[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
fn authenticated_section_replay_epoch(
    metadata_epoch: Option<EpochId>,
    graph_model: grafeo_common::types::GraphModelTag,
    lpg_store: &LpgStore,
    #[cfg(feature = "triple-store")] rdf_store: &RdfStore,
    #[cfg(feature = "triple-store")] rdf_section_version: Option<u8>,
) -> Result<Option<EpochId>> {
    let has_lpg = matches!(
        graph_model,
        grafeo_common::types::GraphModelTag::Lpg | grafeo_common::types::GraphModelTag::Both
    );
    let has_rdf = matches!(
        graph_model,
        grafeo_common::types::GraphModelTag::Rdf | grafeo_common::types::GraphModelTag::Both
    );

    #[cfg(not(feature = "triple-store"))]
    if has_rdf {
        return Err(Error::Serialization(
            "RDF snapshot requires triple-store support".to_string(),
        ));
    }

    let lpg_epoch = has_lpg.then(|| lpg_store.current_epoch());
    #[cfg(feature = "triple-store")]
    let rdf_epoch = (has_rdf && rdf_section_version.is_some_and(|version| version >= 3))
        .then(|| rdf_store.commit_epoch());

    if let Some(epoch) = metadata_epoch {
        if let Some(decoded) = lpg_epoch
            && decoded != epoch
        {
            return Err(Error::Storage(
                grafeo_common::utils::error::StorageError::Corruption(format!(
                    "decoded LPG snapshot epoch {} disagrees with authenticated world epoch {}",
                    decoded.as_u64(),
                    epoch.as_u64()
                )),
            ));
        }
        #[cfg(feature = "triple-store")]
        if let Some(decoded) = rdf_epoch
            && decoded != epoch
        {
            return Err(Error::Storage(
                grafeo_common::utils::error::StorageError::Corruption(format!(
                    "decoded RDF snapshot epoch {} disagrees with authenticated world epoch {}",
                    decoded.as_u64(),
                    epoch.as_u64()
                )),
            ));
        }
        return Ok(Some(epoch));
    }

    match graph_model {
        grafeo_common::types::GraphModelTag::Lpg => Ok(lpg_epoch),
        #[cfg(feature = "triple-store")]
        grafeo_common::types::GraphModelTag::Rdf => Ok(rdf_epoch),
        #[cfg(feature = "triple-store")]
        grafeo_common::types::GraphModelTag::Both => {
            let lpg_epoch = lpg_epoch.expect("Both model has an LPG section");
            if let Some(decoded) = rdf_epoch
                && decoded != lpg_epoch
            {
                return Err(Error::Storage(
                    grafeo_common::utils::error::StorageError::Corruption(format!(
                        "metadata-absent mixed snapshot has divergent LPG epoch {} and RDF epoch {}",
                        lpg_epoch.as_u64(),
                        decoded.as_u64()
                    )),
                ));
            }
            // RDF v1/v2 omitted its clock, but the checksummed LPG v2/v3 plane
            // still supplies the shared cut for a metadata-absent Both image.
            Ok(Some(lpg_epoch))
        }
        #[cfg(not(feature = "triple-store"))]
        grafeo_common::types::GraphModelTag::Rdf | grafeo_common::types::GraphModelTag::Both => {
            unreachable!("rejected above")
        }
    }
}

/// Removes committed WAL groups already represented by an installed
/// single-file snapshot.
///
/// Snapshot installation and checkpoint-metadata publication are two durable
/// steps. If the process dies between them, the sidecar still starts before the
/// snapshot boundary. Reapplying that prefix is not generally idempotent (it
/// duplicates counts and rewrites MVCC history), so the caller supplies an
/// epoch derived from the verified snapshot payload. A transaction-id floor is
/// used only when the recovery-image seal authenticates it. Current records
/// carry a crash-stable commit epoch; the reverse group walk also covers legacy
/// untagged mutations that belong to the following commit marker.
///
/// # Errors
///
/// Returns an error when the snapshot has no authenticated transaction-id
/// floor and the WAL contains a committed legacy group with no epoch. Such a
/// group cannot safely be classified as covered or newer than the snapshot.
#[cfg(feature = "wal")]
fn wal_records_after_snapshot(
    records: Vec<WalRecord>,
    snapshot_epoch: EpochId,
    snapshot_transaction: TransactionId,
) -> Result<Vec<WalRecord>> {
    use std::collections::{HashMap, HashSet};

    let mut commit_epochs = HashMap::<TransactionId, EpochId>::new();
    let mut legacy_commit = None;
    for record in &records {
        match record {
            WalRecord::Committed {
                transaction_id,
                epoch,
            }
            | WalRecord::CommittedWithCdc {
                transaction_id,
                epoch,
                ..
            } => {
                commit_epochs.insert(*transaction_id, *epoch);
                legacy_commit = None;
            }
            WalRecord::TransactionCommit { transaction_id } => {
                legacy_commit = Some(*transaction_id);
            }
            WalRecord::EpochAdvance { epoch } => {
                if let Some(transaction_id) = legacy_commit.take() {
                    commit_epochs.insert(transaction_id, *epoch);
                }
            }
            _ => {}
        }
    }

    // Epoch zero precedes every committed transaction, so an epoch-less
    // legacy group is necessarily newer and can be retained without a TID
    // coordinate. At any later snapshot epoch the same group is ambiguous:
    // it may already be represented by the installed image.
    if snapshot_epoch.as_u64() > 0
        && !snapshot_transaction.is_valid()
        && let Some(transaction_id) = records
            .iter()
            .filter(|record| matches!(record, WalRecord::TransactionCommit { .. }))
            .filter_map(WalEntry::transaction_id)
            .find(|transaction_id| {
                transaction_id.is_valid() && !commit_epochs.contains_key(transaction_id)
            })
    {
        return Err(Error::Serialization(format!(
            "cannot establish the snapshot replay floor for epoch-less committed WAL transaction {transaction_id}: the recovery image has no authenticated transaction-id coordinate"
        )));
    }

    let covered: HashSet<TransactionId> = records
        .iter()
        .filter_map(WalEntry::transaction_id)
        .filter(|transaction_id| {
            transaction_id.is_valid()
                && commit_epochs.get(transaction_id).map_or_else(
                    || snapshot_transaction.is_valid() && *transaction_id <= snapshot_transaction,
                    |epoch| *epoch <= snapshot_epoch,
                )
        })
        .collect();

    let mut discard = vec![false; records.len()];
    let mut following_group_is_covered = false;
    for (index, record) in records.iter().enumerate().rev() {
        match record {
            WalRecord::CatalogBatchV3 { epoch, .. }
            | WalRecord::RdfLpgProjectionDeclaredV3 { epoch, .. } => {
                discard[index] = *epoch <= snapshot_epoch;
                // An epoch-bearing standalone publication is a group boundary
                // in its own right. Legacy untagged data before it must not
                // inherit the coverage decision of a later transaction group.
                following_group_is_covered = false;
                continue;
            }
            _ => {}
        }
        if record.is_metadata() {
            // Standalone format/catalog metadata has no transaction boundary.
            // Its replay implementations are replacement/idempotent operations;
            // retain it for compatibility with older WALs.
            continue;
        }
        if record.is_checkpoint() {
            discard[index] = true;
            continue;
        }
        if record.is_commit() {
            following_group_is_covered = record
                .transaction_id()
                .is_some_and(|transaction_id| covered.contains(&transaction_id));
            discard[index] = following_group_is_covered;
            continue;
        }
        if let Some(transaction_id) = record.transaction_id()
            && transaction_id.is_valid()
        {
            discard[index] = covered.contains(&transaction_id);
        } else {
            // Legacy untagged mutations are emitted as a contiguous logical
            // prefix immediately before the commit that authenticated them.
            discard[index] = following_group_is_covered;
        }
    }

    Ok(records
        .into_iter()
        .zip(discard)
        .filter_map(|(record, discard)| (!discard).then_some(record))
        .collect())
}

/// Safe administrative view of a database's write-ahead log.
///
/// This deliberately exposes durability controls and read-only status only.
/// Transaction records are an engine trust boundary: allowing callers to
/// append arbitrary mutations or `Committed` markers could make recovery
/// authenticate data that was never published by the transaction manager.
#[cfg(feature = "wal")]
#[derive(Clone, Copy)]
pub struct WalControl<'a> {
    wal: &'a LpgWal,
}

#[cfg(feature = "wal")]
impl WalControl<'_> {
    /// Flushes buffered bytes and fsyncs the active WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if buffered WAL data cannot be written or synchronized.
    pub fn sync(self) -> Result<()> {
        self.wal.sync()
    }

    /// Flushes buffered bytes to the operating system.
    ///
    /// # Errors
    ///
    /// Returns an error if buffered WAL data cannot be written.
    pub fn flush(self) -> Result<()> {
        self.wal.flush()
    }

    /// Force-rotates the active WAL segment.
    ///
    /// # Errors
    ///
    /// Returns an error if the current segment cannot be finalized or a new
    /// segment cannot be created.
    pub fn rotate(self) -> Result<()> {
        self.wal.rotate()
    }

    /// Returns the number of frames appended through this handle.
    #[must_use]
    pub fn record_count(self) -> u64 {
        self.wal.record_count()
    }

    /// Returns whether the WAL has failed closed after an I/O error.
    #[must_use]
    pub fn is_poisoned(self) -> bool {
        self.wal.is_poisoned()
    }

    /// Returns the active WAL segment sequence.
    #[must_use]
    pub fn current_sequence(self) -> u64 {
        self.wal.current_sequence()
    }

    /// Returns the total size of WAL files and checkpoint metadata.
    ///
    /// # Errors
    /// Returns terminal-state, filesystem, or size-overflow errors.
    pub fn size_bytes(self) -> Result<usize> {
        self.wal.size_bytes()
    }
}

use crate::catalog::Catalog;
use crate::config::Config;
use crate::query::cache::QueryCache;
use crate::session::Session;
use crate::transaction::TransactionManager;

#[cfg(test)]
thread_local! {
    static PUBLICATION_CUT_RENDEZVOUS: std::cell::RefCell<Option<std::sync::mpsc::Sender<()>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn publication_cut_test_point() {
    PUBLICATION_CUT_RENDEZVOUS.with(|point| {
        if let Some(arrived) = point.borrow_mut().take() {
            arrived.send(()).unwrap();
        }
    });
}

/// Your handle to a Grafeo database.
///
/// Start here. Create one with [`new_in_memory()`](Self::new_in_memory) for
/// quick experiments, or [`open()`](Self::open) for persistent storage.
/// Then grab a [`session()`](Self::session) to start querying.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// // Quick in-memory database
/// let db = GrafeoDB::new_in_memory();
///
/// // Add some data
/// db.create_node(&["Person"]);
///
/// // Query it
/// let session = db.session();
/// let result = session.execute("MATCH (p:Person) RETURN p")?;
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
pub struct GrafeoDB {
    /// Database configuration.
    pub(super) config: Config,
    /// Portable logical identity shared by every authoritative model.
    pub(super) world_identity: Arc<RwLock<WorldIdentityMetadataV1>>,
    /// The underlying graph store (None when using an external store).
    #[cfg(feature = "lpg")]
    pub(super) store: Option<Arc<LpgStore>>,
    /// Schema and metadata catalog shared across sessions.
    pub(super) catalog: Arc<Catalog>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "triple-store")]
    pub(super) rdf_store: Arc<RdfStore>,
    /// Durable RDF→LPG projection definitions and last successful generations.
    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    pub(super) rdf_projections: Arc<grafeo_core::graph::rdf::RdfLpgProjectionRegistry>,
    /// Transaction manager.
    pub(super) transaction_manager: Arc<TransactionManager>,
    /// Unified buffer manager.
    pub(super) buffer_manager: Arc<BufferManager>,
    /// Shared authenticated spill-root owner, opened only by configured queries.
    #[cfg(feature = "spill")]
    pub(super) spill_root: Arc<crate::spill_crypto::DatabaseSpillRoot>,
    /// Write-ahead log manager (if durability is enabled).
    #[cfg(feature = "wal")]
    pub(super) wal: Option<Arc<LpgWal>>,
    /// Query cache for parsed and optimized plans.
    pub(super) query_cache: Arc<QueryCache>,
    /// Physical operator trees shared by every session on this database.
    pub(super) physical_cache: Arc<parking_lot::Mutex<crate::query::cache::PhysicalPlanCache>>,
    /// Shared commit counter for auto-GC across sessions.
    #[cfg(feature = "lpg")]
    pub(super) commit_counter: Arc<AtomicUsize>,
    /// Set when abort/log/fsync fails; further mutations must fail closed.
    durability_poisoned: Arc<AtomicBool>,
    /// Whether the database is open.
    pub(super) is_open: Arc<RwLock<bool>>,
    /// Actual sealed W retained through C mutation; failed C close keeps it
    /// until owned database destruction. This carries no writable authority.
    #[cfg(feature = "wal")]
    retained_wal_seal: parking_lot::Mutex<Option<grafeo_storage::wal::SealedWal>>,
    /// Number of live Session handles. Store-replacing maintenance operations
    /// require this to be zero so no handle can retain a stale overlay.
    active_sessions: Arc<AtomicUsize>,
    /// Move-only authority for dangling edges carried by a transport extract.
    /// Only `extract_subgraph` populates this map; orphan cleanup consumes the
    /// matching receipts to purge the exact transport histories physically.
    #[cfg(feature = "lpg")]
    transport_extract_receipts: parking_lot::Mutex<
        grafeo_common::utils::hash::FxHashMap<grafeo_common::types::EdgeId, TransportEdgeReceipt>,
    >,
    /// Change data capture log for tracking mutations.
    #[cfg(feature = "cdc")]
    pub(super) cdc_log: Arc<crate::cdc::CdcLog>,
    /// Whether CDC is active for new sessions and direct CRUD (runtime-mutable).
    #[cfg(feature = "cdc")]
    cdc_enabled: std::sync::atomic::AtomicBool,
    /// Registered embedding models for text-to-vector conversion.
    #[cfg(feature = "embed")]
    pub(super) embedding_models:
        RwLock<hashbrown::HashMap<String, Arc<dyn crate::embedding::EmbeddingModel>>>,
    /// Single-file database manager (when using `.grafeo` format).
    #[cfg(feature = "grafeo-file")]
    pub(super) file_manager: Option<Arc<GrafeoFileManager>>,
    /// Periodic checkpoint timer (when `checkpoint_interval` is configured).
    /// Wrapped in Mutex because `close()` takes `&self` but needs to stop the timer.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    checkpoint_timer: parking_lot::Mutex<Option<checkpoint_timer::CheckpointTimer>>,
    /// External read-only graph store (when using with_store() or with_read_store()).
    /// When set, sessions route queries through this store instead of the built-in LpgStore.
    pub(super) external_read_store: Option<Arc<dyn GraphStoreSearch>>,
    /// External writable graph store (when using with_store()).
    /// None for read-only databases created via with_read_store().
    pub(super) external_write_store: Option<Arc<dyn GraphStoreMut>>,
    /// Metrics registry shared across all sessions.
    #[cfg(feature = "metrics")]
    pub(crate) metrics: Option<Arc<crate::metrics::MetricsRegistry>>,
    /// One coherent resolved graph path and its independent language selectors.
    /// Sessions inherit this validated context without rerunning selection.
    current_context: RwLock<crate::session::SessionGraphContext>,
    /// Whether this database is open in read-only mode.
    /// When true, sessions automatically enforce read-only transactions.
    read_only: bool,
    /// Named graph projections (virtual subgraphs), shared with sessions.
    projections: crate::session::VirtualProjectionRegistry,
    /// Layered store (compact base + mutable overlay), set after `compact()`.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    layered_store: Option<Arc<grafeo_core::graph::compact::layered::LayeredStore>>,
    /// Claim / statement table (opaque seven-column claim layout).
    #[cfg(feature = "statement-table")]
    statement_table: RwLock<grafeo_core::graph::compact::statement_table::StatementTable>,
    /// Disk-backed tier wrapper for the compact base, set after `compact()`.
    ///
    /// Provides the spill path for [`CompactStoreConsumer`]: when the buffer
    /// manager signals memory pressure, the consumer calls
    /// `persist_to_mmap()` here and then publishes the fresh base to
    /// `layered_store` via `swap_base()`.
    #[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
    compact_tiered: Option<Arc<compact_tiered::CompactStoreTiered>>,
}

/// Locks retained for the full duration of one committed-state capture.
///
/// The fields are deliberately ordered by the global lock hierarchy:
/// projection rebuild, database lifecycle, RDF commit, then cross-model
/// publication. The lifecycle write guard excludes transaction registration,
/// rather than merely sampling `active_count` around serialization.
pub(super) struct QuiescentCapture<'a> {
    #[cfg(all(feature = "triple-store", feature = "lpg"))]
    _projection_rebuild: parking_lot::MutexGuard<'a, ()>,
    _database_open: parking_lot::RwLockWriteGuard<'a, bool>,
    #[cfg(feature = "triple-store")]
    _rdf_commit: parking_lot::MutexGuard<'a, ()>,
    _publication: parking_lot::RwLockReadGuard<'a, ()>,
}

/// Exact mutable destination for LPG WAL recovery.
///
/// A compact container has one columnar base only for the default graph, so
/// default-graph records must traverse the already-reconstructed layered view.
/// Named graphs remain independent flat `LpgStore` incarnations, including the
/// legal empty-string name.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[derive(Clone)]
enum LpgRecoveryTarget {
    Flat(Arc<LpgStore>),
    #[cfg(feature = "compact-store")]
    Layered(Arc<grafeo_core::graph::compact::layered::LayeredStore>),
}

#[cfg(all(feature = "wal", feature = "lpg"))]
impl LpgRecoveryTarget {
    /// Replay recorded index state separately from ordinary derived hooks.
    fn with_index_replay<T>(
        &self,
        recorded: bool,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        {
            if recorded {
                return match self {
                    Self::Flat(store) => store.with_recorded_index_recovery(
                        cfg!(feature = "text-index"),
                        cfg!(feature = "vector-index"),
                        operation,
                    ),
                    #[cfg(feature = "compact-store")]
                    Self::Layered(store) => store.with_recorded_index_recovery(
                        cfg!(feature = "text-index"),
                        cfg!(feature = "vector-index"),
                        operation,
                    ),
                };
            }
            let has_indexes = |store: &LpgStore| {
                let mut present = false;
                #[cfg(feature = "text-index")]
                {
                    present |= !store.text_index_entries().is_empty();
                }
                #[cfg(feature = "vector-index")]
                {
                    present |= !store.vector_index_entries().is_empty();
                }
                present
            };
            let present = match self {
                Self::Flat(store) => has_indexes(store),
                #[cfg(feature = "compact-store")]
                Self::Layered(store) => has_indexes(&store.overlay_store()),
            };
            if present {
                return Err(Error::Serialization(
                    "index mutation lacks a current owner-qualified WAL witness".into(),
                ));
            }
        }
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        let _ = recorded;
        operation()
    }

    fn flat_node_creation_matches(
        store: &LpgStore,
        id: grafeo_common::types::NodeId,
        labels: &[&str],
    ) -> bool {
        if store.get_node(id).is_none() {
            return false;
        }
        let label_history = store.node_label_history(id);
        let Some((_, creation_labels)) = label_history.first() else {
            return false;
        };
        let mut actual: Vec<&str> = creation_labels.iter().map(arcstr::ArcStr::as_str).collect();
        actual.sort_unstable();
        actual.dedup();
        let mut expected = labels.to_vec();
        expected.sort_unstable();
        expected.dedup();
        actual == expected
    }

    fn flat_edge_matches(
        store: &LpgStore,
        id: grafeo_common::types::EdgeId,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
    ) -> bool {
        store.get_edge(id).is_some_and(|edge| {
            edge.src == src && edge.dst == dst && edge.edge_type.as_str() == edge_type
        })
    }

    fn mutation_store(&self) -> &dyn GraphStoreMut {
        match self {
            Self::Flat(store) => store.as_ref(),
            #[cfg(feature = "compact-store")]
            Self::Layered(store) => store.as_ref(),
        }
    }

    fn sync_epoch(&self, epoch: grafeo_common::types::EpochId) {
        match self {
            Self::Flat(store) => store.sync_epoch(epoch),
            #[cfg(feature = "compact-store")]
            Self::Layered(store) => store.overlay_store().sync_epoch(epoch),
        }
    }

    fn recover_create_node_with_id(
        &self,
        id: grafeo_common::types::NodeId,
        labels: &[&str],
    ) -> Result<()> {
        match self {
            Self::Flat(store) => {
                if store.get_node(id).is_some() {
                    return if Self::flat_node_creation_matches(store, id, labels) {
                        Ok(())
                    } else {
                        Err(Error::Storage(
                            grafeo_common::utils::error::StorageError::Corruption(format!(
                                "cannot replay node {id}: the identity already has a different structural create"
                            )),
                        ))
                    };
                }
                store.create_node_with_id(id, labels)?;
                if !Self::flat_node_creation_matches(store, id, labels) {
                    return Err(Error::Storage(
                        grafeo_common::utils::error::StorageError::Corruption(format!(
                            "cannot replay node {id}: invalid, closed, or conflicting identity"
                        )),
                    ));
                }
            }
            #[cfg(feature = "compact-store")]
            Self::Layered(store) => store.recover_create_node_with_id(id, labels)?,
        }
        Ok(())
    }

    fn recover_create_edge_with_id(
        &self,
        id: grafeo_common::types::EdgeId,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
    ) -> Result<()> {
        match self {
            Self::Flat(store) => {
                if store.get_node(src).is_none() || store.get_node(dst).is_none() {
                    return Err(Error::Storage(
                        grafeo_common::utils::error::StorageError::Corruption(format!(
                            "cannot replay edge {id}: endpoint {src} or {dst} is missing"
                        )),
                    ));
                }
                if store.get_edge(id).is_some() {
                    return if Self::flat_edge_matches(store, id, src, dst, edge_type) {
                        Ok(())
                    } else {
                        Err(Error::Storage(
                            grafeo_common::utils::error::StorageError::Corruption(format!(
                                "cannot replay edge {id}: the identity already has different endpoints or type"
                            )),
                        ))
                    };
                }
                store.create_edge_with_id(id, src, dst, edge_type)?;
                if !Self::flat_edge_matches(store, id, src, dst, edge_type) {
                    return Err(Error::Storage(
                        grafeo_common::utils::error::StorageError::Corruption(format!(
                            "cannot replay edge {id}: invalid, closed, or conflicting identity"
                        )),
                    ));
                }
            }
            #[cfg(feature = "compact-store")]
            Self::Layered(store) => {
                store.recover_create_edge_with_id(id, src, dst, edge_type)?;
            }
        }
        Ok(())
    }
}

/// Recovery decisions that must survive beyond detached section decoding.
///
/// Exact catalog and auxiliary state require an authenticated recovery image.
/// Compact base/overlay wiring is completed before the database is exposed.
#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
#[allow(
    clippy::struct_excessive_bools,
    reason = "detached recovery records independent authenticated facts before publication"
)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SectionLoadOutcome {
    recovery_image_is_sealed: bool,
    recovery_coordinates_are_sealed: bool,
    #[cfg(feature = "wal")]
    authenticated_replay_epoch: Option<EpochId>,
    #[cfg(all(feature = "triple-store", feature = "ring-index"))]
    ensure_rdf_ring_after_recovery: bool,
}

/// Detached state decoded from one already-authenticated container image.
///
/// The layered view is built from the exact `EncodedSection` bytes whose
/// inventory and digest were verified. Keeping it beside the scalar outcome
/// prevents a second directory/data read from crossing the recovery trust
/// boundary after authentication.
#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
struct LoadedSectionState {
    outcome: SectionLoadOutcome,
    #[cfg(feature = "wal")]
    authenticated_identity: Option<AuthenticatedContainerIdentity>,
    #[cfg(feature = "compact-store")]
    layered: Option<Arc<grafeo_core::graph::compact::layered::LayeredStore>>,
}

#[cfg(all(
    feature = "grafeo-file",
    feature = "triple-store",
    not(feature = "lpg")
))]
struct LoadedRdfSectionState {
    ring_declared: bool,
    recovery_image_is_sealed: bool,
    recovery_coordinates_are_sealed: bool,
    authenticated_replay_epoch: Option<grafeo_common::types::EpochId>,
    #[cfg(feature = "wal")]
    authenticated_identity: Option<AuthenticatedContainerIdentity>,
}

#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
impl SectionLoadOutcome {
    #[allow(
        clippy::fn_params_excessive_bools,
        reason = "construction derives independent recovery facts from one decoded image"
    )]
    fn for_image(
        catalog_version: Option<catalog_wire::CatalogPayloadVersion>,
        recovery_image_is_sealed: bool,
        recovery_coordinates_are_sealed: bool,
        #[cfg(all(feature = "triple-store", feature = "ring-index"))] has_ring_section: bool,
    ) -> Result<Self> {
        if !recovery_image_is_sealed
            && catalog_version == Some(catalog_wire::CatalogPayloadVersion::GraphExactV7)
        {
            return Err(Error::Serialization(
                "Catalog v7 exact auxiliary recovery requires WorldMetadata v2".to_string(),
            ));
        }
        if recovery_coordinates_are_sealed && !recovery_image_is_sealed {
            return Err(Error::Serialization(
                "recovery header coordinates cannot be sealed by an unsealed section image"
                    .to_string(),
            ));
        }

        Ok(Self {
            recovery_image_is_sealed,
            recovery_coordinates_are_sealed,
            #[cfg(feature = "wal")]
            authenticated_replay_epoch: None,
            #[cfg(all(feature = "triple-store", feature = "ring-index"))]
            ensure_rdf_ring_after_recovery: has_ring_section,
        })
    }
}

/// Requires the current exact index section and its authenticated recovery seal.
#[cfg(all(
    feature = "grafeo-file",
    feature = "lpg",
    any(feature = "text-index", feature = "vector-index")
))]
fn validate_exact_index_recovery_section(
    recovery_image_is_sealed: bool,
    section_version: Option<u8>,
    kind: &str,
    current_version: u8,
) -> Result<()> {
    let Some(version) = section_version else {
        return Ok(());
    };
    if version != current_version {
        return Err(Error::Serialization(format!(
            "unsupported {kind} section version {version}; expected {current_version}"
        )));
    }
    if !recovery_image_is_sealed {
        return Err(Error::Serialization(format!(
            "{kind} v{current_version} requires an authenticated recovery image seal"
        )));
    }
    Ok(())
}

/// A private handle to the built-in overlay `LpgStore`.
///
/// Only explicit overlay-local operations used by database internals are
/// exposed. Whole-graph reads must go through the tier-merged database view.
#[cfg(feature = "lpg")]
pub(crate) struct OverlayStore<'a>(&'a Arc<LpgStore>);

#[cfg(feature = "lpg")]
impl OverlayStore<'_> {
    #[cfg(test)]
    fn graph(&self, name: &str) -> Option<Arc<LpgStore>> {
        self.0.graph(name)
    }
    #[cfg(test)]
    fn create_graph(
        &self,
        name: &str,
    ) -> std::result::Result<bool, grafeo_common::memory::arena::AllocError> {
        self.0.create_graph(name)
    }
    fn graph_names(&self) -> Vec<String> {
        self.0.graph_names()
    }
    fn current_epoch(&self) -> grafeo_common::types::EpochId {
        self.0.current_epoch()
    }
    fn sync_epoch(&self, epoch: grafeo_common::types::EpochId) {
        self.0.sync_epoch(epoch);
    }
    fn get_node_at_epoch(
        &self,
        id: grafeo_common::types::NodeId,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_core::graph::lpg::Node> {
        self.0.get_node_at_epoch(id, epoch)
    }
    fn get_edge_at_epoch(
        &self,
        id: grafeo_common::types::EdgeId,
        epoch: grafeo_common::types::EpochId,
    ) -> Option<grafeo_core::graph::lpg::Edge> {
        self.0.get_edge_at_epoch(id, epoch)
    }
    fn neighbors_versioned(
        &self,
        node: grafeo_common::types::NodeId,
        direction: grafeo_core::graph::Direction,
        epoch: grafeo_common::types::EpochId,
        tx: grafeo_common::types::TransactionId,
    ) -> Vec<grafeo_common::types::NodeId> {
        self.0.neighbors_versioned(node, direction, epoch, tx)
    }
    fn fill_neighbors_of_types_at_epoch(
        &self,
        node: grafeo_common::types::NodeId,
        direction: grafeo_core::graph::Direction,
        epoch: grafeo_common::types::EpochId,
        types: &[String],
        out: &mut Vec<grafeo_common::types::NodeId>,
    ) {
        grafeo_core::graph::GraphStore::fill_neighbors_of_types_at_epoch(
            self.0.as_ref(),
            node,
            direction,
            epoch,
            types,
            out,
        );
    }
    fn all_node_ids(&self) -> Vec<grafeo_common::types::NodeId> {
        self.0.all_node_ids()
    }
    fn has_property_index(&self, property: &str) -> bool {
        self.0.has_property_index(property)
    }
    #[cfg(feature = "vector-index")]
    fn get_vector_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<grafeo_core::index::vector::VectorIndexView> {
        self.0.get_vector_index(label, property)
    }
    #[cfg(feature = "text-index")]
    fn get_text_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<grafeo_core::index::text::TextIndexView> {
        self.0.get_text_index(label, property)
    }
    #[cfg(all(feature = "wal", feature = "text-index"))]
    fn text_index_entries(&self) -> Vec<(String, grafeo_core::index::text::TextIndexView)> {
        self.0.text_index_entries()
    }
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    fn vector_index_entries(&self) -> Vec<(String, grafeo_core::index::vector::VectorIndexView)> {
        self.0.vector_index_entries()
    }
    fn ensure_statistics_fresh(&self) {
        self.0.ensure_statistics_fresh();
    }
    fn gc_versions(&self, epoch: grafeo_common::types::EpochId) {
        self.0.gc_versions(epoch);
    }
    #[cfg(feature = "text-index")]
    fn gc_text_indexes(&self, epoch: grafeo_common::types::EpochId) -> Result<()> {
        self.0.gc_text_indexes(epoch)
    }
    #[cfg(feature = "vector-index")]
    fn gc_vector_indexes(&self, epoch: grafeo_common::types::EpochId) -> Result<()> {
        self.0.gc_vector_indexes(epoch)
    }
    fn memory_breakdown(
        &self,
    ) -> (
        grafeo_common::memory::StoreMemory,
        grafeo_common::memory::IndexMemory,
        grafeo_common::memory::MvccMemory,
        grafeo_common::memory::StringPoolMemory,
    ) {
        self.0.memory_breakdown()
    }
}

impl GrafeoDB {
    /// Acquires a race-free, quiescent committed-state capture boundary.
    pub(super) fn acquire_quiescent_capture(
        &self,
        operation: &'static str,
    ) -> Result<QuiescentCapture<'_>> {
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let projection_rebuild = self.rdf_projections.lock_rebuild();

        let database_open = self.is_open.write();
        if !*database_open {
            return Err(Error::Transaction(TransactionError::InvalidState(format!(
                "cannot {operation} a closed database"
            ))));
        }
        #[cfg(feature = "wal")]
        if self.is_durability_poisoned() {
            return Err(Error::Transaction(TransactionError::DurabilityFailure(
                format!(
                    "cannot {operation} a durability-poisoned database; reopen and recover the WAL first"
                ),
            )));
        }
        self.require_quiescent(operation)?;

        #[cfg(test)]
        publication_cut_test_point();
        #[cfg(feature = "triple-store")]
        let rdf_commit = self.rdf_store.lock_commit();
        let publication = self.transaction_manager.publication().read();
        self.require_quiescent(operation)?;
        #[cfg(feature = "wal")]
        if self.is_durability_poisoned() {
            return Err(Error::Transaction(TransactionError::DurabilityFailure(
                format!(
                    "cannot {operation} a durability-poisoned database; reopen and recover the WAL first"
                ),
            )));
        }

        Ok(QuiescentCapture {
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            _projection_rebuild: projection_rebuild,
            _database_open: database_open,
            #[cfg(feature = "triple-store")]
            _rdf_commit: rdf_commit,
            _publication: publication,
        })
    }

    /// Portable identity of this logical database.
    #[must_use]
    pub fn store_id(&self) -> StoreId {
        self.world_identity.read().store_id()
    }

    /// Portable database identity and truthful RDF-history provenance.
    #[must_use]
    pub fn world_identity(&self) -> WorldIdentityMetadataV1 {
        self.world_identity.read().clone()
    }

    #[cfg(feature = "triple-store")]
    pub(super) fn sync_world_identity_from_rdf(&self) -> Result<()> {
        let identity = WorldIdentityMetadataV1::new(
            self.rdf_store.store_id(),
            self.rdf_store.history_completeness(),
        )
        .map_err(|error| Error::Internal(format!("invalid restored RDF identity: {error}")))?;
        *self.world_identity.write() = identity;
        Ok(())
    }

    /// The built-in overlay `LpgStore`, as a raw `Arc` — for the store-wiring
    /// machinery only (session construction, the read/write graph views, base
    /// swaps). Everything else should use [`lpg_store`](Self::lpg_store) (guarded)
    /// or the tier-merged [`read_graph_view`](Self::read_graph_view).
    ///
    /// # Panics
    ///
    /// Panics if the database was created with an external store.
    #[cfg(feature = "lpg")]
    fn store_arc(&self) -> &Arc<LpgStore> {
        self.store.as_ref().expect(
            "no built-in LpgStore: this GrafeoDB was created with an external store \
             (with_store / with_read_store). Use session() or graph_store() instead.",
        )
    }

    /// A guarded handle to the built-in overlay `LpgStore` for overlay-LOCAL
    /// operations. See [`OverlayStore`] — whole-graph reads are a build error;
    /// use [`read_graph_view`](Self::read_graph_view) for those.
    ///
    /// # Panics
    ///
    /// Panics if the database was created with [`with_store()`](Self::with_store) or
    /// [`with_read_store()`](Self::with_read_store), which use an external store
    /// instead of the built-in LPG store.
    #[cfg(feature = "lpg")]
    fn lpg_store(&self) -> OverlayStore<'_> {
        OverlayStore(self.store_arc())
    }

    /// Returns the active **read** graph view, tier-merged.
    ///
    /// After [`compact()`](Self::compact) this is the `LayeredStore`
    /// (columnar base + overlay); otherwise the built-in `LpgStore`. Use this
    /// for whole-graph reads that must see both tiers (`extract_subgraph`,
    /// `remove_orphan_edges`) — `lpg_store()` alone is overlay-only
    /// post-compact, so base-tier nodes/edges are invisible to it.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    fn read_graph_view(&self) -> &dyn grafeo_core::graph::GraphStore {
        if let Some(ref layered) = self.layered_store {
            &**layered
        } else {
            &**self.store_arc()
        }
    }

    /// Non-compact builds: the read view is always the built-in store.
    #[cfg(all(not(feature = "compact-store"), feature = "lpg"))]
    fn read_graph_view(&self) -> &dyn grafeo_core::graph::GraphStore {
        &**self.store_arc()
    }

    /// Every node in the database, **tier-merged** (base + overlay).
    ///
    /// The blessed whole-graph node enumeration for save / snapshot / export.
    /// After `compact()` the data lives in the columnar base, so iterating
    /// `lpg_store().all_nodes()` (the raw overlay) returns nothing — always use
    /// this for whole-graph reads.
    #[cfg(feature = "lpg")]
    fn read_all_nodes(&self) -> Vec<grafeo_core::graph::lpg::Node> {
        let view = self.read_graph_view();
        view.node_ids()
            .into_iter()
            .filter_map(|id| view.get_node(id))
            .collect()
    }

    /// Every edge in the database, **tier-merged** (base + overlay).
    /// See [`read_all_nodes`](Self::read_all_nodes). Each edge is visited once
    /// (via its source node's outgoing adjacency).
    #[cfg(feature = "lpg")]
    fn read_all_edges(&self) -> Vec<grafeo_core::graph::lpg::Edge> {
        use grafeo_core::graph::Direction;
        let view = self.read_graph_view();
        let mut edges = Vec::new();
        for nid in view.node_ids() {
            for (_, eid) in view.edges_from(nid, Direction::Outgoing) {
                if let Some(edge) = view.get_edge(eid) {
                    edges.push(edge);
                }
            }
        }
        edges
    }

    /// The **live** overlay `LpgStore`, for index maintenance and the concrete
    /// LPG-only helpers (vector/text index accessors) that are not part of the
    /// graph-store traits.
    ///
    /// After [`compact()`](Self::compact) this
    /// is the overlay's *current* store, re-fetched each call — so it is never
    /// the stale handle captured when the overlay was last swapped by a merge.
    /// An entity just written through [`write_graph_view`](Self::write_graph_view)
    /// has been promoted into this same overlay, so reading it back here to
    /// update indexes sees the new value. Without a layered store it is the
    /// built-in `LpgStore`.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    fn live_overlay(&self) -> Arc<LpgStore> {
        if let Some(ref layered) = self.layered_store {
            layered.overlay_store()
        } else {
            Arc::clone(self.store_arc())
        }
    }

    /// Non-compact builds: the live overlay is the built-in store.
    #[cfg(all(not(feature = "compact-store"), feature = "lpg"))]
    fn live_overlay(&self) -> Arc<LpgStore> {
        Arc::clone(self.store_arc())
    }

    /// Returns a borrowed reference to the active graph store.
    ///
    /// In layered mode (after [`compact()`](Self::compact)), returns the
    /// `LayeredStore` which merges the columnar base with the overlay.
    /// Otherwise, returns the built-in `LpgStore`.
    ///
    /// Unlike [`graph_store()`](Self::graph_store) (which clones an `Arc`),
    /// this borrows from `self` — suitable for constructing accessors that
    /// need `&'a dyn GraphStore` tied to the database lifetime.
    #[cfg(any(
        feature = "vector-index",
        feature = "text-index",
        feature = "hybrid-search",
        feature = "embed",
    ))]
    #[cfg(feature = "lpg")]
    fn graph_store_ref(&self) -> &dyn grafeo_core::graph::GraphStore {
        if let Some(ref ext_read) = self.external_read_store {
            ext_read.as_ref()
        } else {
            // The merged tier view filters compacted/deleted entries through
            // the same database read view used by public LPG search.
            self.read_graph_view()
        }
    }

    /// Returns whether CDC is active (runtime check).
    #[cfg(feature = "cdc")]
    #[inline]
    pub(super) fn cdc_active(&self) -> bool {
        self.cdc_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Creates an in-memory database, fast to create, gone when dropped.
    ///
    /// Use this for tests, experiments, or when you don't need persistence.
    /// For data that survives restarts, use [`open()`](Self::open) instead.
    ///
    /// # Panics
    ///
    /// Panics if in-memory database initialization fails, including allocator,
    /// identity-entropy, or configuration initialization. Use
    /// [`with_config()`](Self::with_config) for a fallible alternative.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_config(Config::in_memory()).expect("In-memory database creation should not fail")
    }

    /// Opens a database at the given path, creating it if it doesn't exist.
    ///
    /// If you've used this path before, Grafeo recovers your data from the
    /// write-ahead log automatically. First open on a new path creates an
    /// empty database.
    ///
    /// # Errors
    ///
    /// Returns an error if the path isn't writable or recovery fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::open("./my_social_network")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "wal")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(Config::persistent(path.as_ref()))
    }

    /// Opens an existing database in read-only mode.
    ///
    /// Uses a shared file lock, so multiple processes can read the same
    /// `.grafeo` file concurrently. The database loads the last checkpoint
    /// snapshot but does **not** replay the WAL or allow mutations.
    ///
    /// Currently only supports the single-file (`.grafeo`) format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file doesn't exist or can't be read.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::open_read_only("./my_graph.grafeo")?;
    /// let session = db.session();
    /// let result = session.execute("MATCH (n) RETURN n LIMIT 10")?;
    /// // Mutations will return an error:
    /// // session.execute("INSERT (:Person)") => Err(ReadOnly)
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "grafeo-file")]
    pub fn open_read_only(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::with_config(Config::read_only(path.as_ref()))
    }

    /// Creates a database with custom configuration.
    ///
    /// Use this when you need fine-grained control over memory limits,
    /// thread counts, or persistence settings. For most cases,
    /// [`new_in_memory()`](Self::new_in_memory) or [`open()`](Self::open)
    /// are simpler.
    ///
    /// # Errors
    ///
    /// Returns an error if the database can't be created or recovery fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, Config};
    ///
    /// // In-memory with a 512MB limit
    /// let config = Config::in_memory()
    ///     .with_memory_limit(512 * 1024 * 1024);
    ///
    /// let db = GrafeoDB::with_config(config)?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn with_config(config: Config) -> Result<Self> {
        Self::with_owned_config(
            config,
            #[cfg(feature = "grafeo-file")]
            None,
            #[cfg(feature = "wal")]
            None,
        )
    }

    // Restore transfers already admitted private capabilities into the same
    // loader and authenticated recovery path used by ordinary opens.
    fn with_owned_config(
        config: Config,
        #[cfg(feature = "grafeo-file")] mut owned_file: Option<GrafeoFileManager>,
        #[cfg(feature = "wal")] mut owned_recovery: Option<WalRecovery>,
    ) -> Result<Self> {
        #[cfg(feature = "grafeo-file")]
        if let Some(file) = &owned_file {
            if config.path.as_deref() != Some(file.path())
                || file.is_read_only()
                    != (config.access_mode == crate::config::AccessMode::ReadOnly)
            {
                return Err(Error::InvalidValue(
                    "restore container capability/config mismatch".into(),
                ));
            }
            #[cfg(feature = "wal")]
            if let Some(recovery) = &owned_recovery
                && recovery.path() != file.sidecar_wal_path()
            {
                return Err(Error::InvalidValue(
                    "restore WAL is not the owned container sidecar".into(),
                ));
            }
        }
        let config = config.resolve_unpinned_graph_model();
        #[cfg(any(feature = "wal", feature = "grafeo-file"))]
        let mut config = config;
        #[cfg(not(any(feature = "wal", feature = "grafeo-file")))]
        let config = config;
        // Validate configuration before proceeding
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        // Resolve the storage route before allocating stores or mutating persistent state.
        // A configured path must never silently become an in-memory database.
        #[cfg(not(feature = "grafeo-file"))]
        if config.path.is_some() {
            return Err(Error::InvalidValue(
                "persistent storage requires the grafeo-file or wal feature".into(),
            ));
        }
        #[cfg(feature = "grafeo-file")]
        let use_single_file = config.access_mode == crate::config::AccessMode::ReadOnly
            || config
                .path
                .as_deref()
                .is_some_and(|path| Self::should_use_single_file(path, config.storage_format));
        #[cfg(feature = "grafeo-file")]
        if config.path.is_some() {
            #[cfg(feature = "wal")]
            let has_persistence = use_single_file || config.wal_enabled;
            #[cfg(not(feature = "wal"))]
            let has_persistence = use_single_file;
            if !has_persistence {
                return Err(Error::InvalidValue(
                    "directory persistence requires the wal feature and enabled WAL".into(),
                ));
            }
        }

        #[cfg(feature = "cdc")]
        let cdc_log = Arc::new(crate::cdc::CdcLog::with_retention(
            config.cdc_retention.clone(),
        ));
        let initial_world_identity = match config.world_identity_override.clone() {
            Some(identity) => identity,
            None => fresh_world_identity()?,
        };
        let world_identity = Arc::new(RwLock::new(initial_world_identity));
        #[cfg(feature = "lpg")]
        let store = Arc::new(LpgStore::new()?);
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::with_config_and_store_id(
            grafeo_core::graph::rdf::RdfStoreConfig::default(),
            world_identity.read().store_id(),
        ));
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        let rdf_projections = Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());
        let transaction_manager = Arc::new(TransactionManager::new());

        // Create buffer manager with configured limits
        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                // reason: product of system RAM and 0.75 is always a valid positive usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b = (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize;
                b
            }),
            spill_path: config.spill_path.clone().or_else(|| {
                config.path.as_ref().and_then(|p| {
                    let parent = p.parent()?;
                    let name = p.file_name()?.to_str()?;
                    Some(parent.join(format!("{name}.spill")))
                })
            }),
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        // Create catalog early so WAL replay can restore schema definitions
        let catalog = Arc::new(Catalog::new());

        let is_read_only = config.access_mode == crate::config::AccessMode::ReadOnly;
        #[cfg(feature = "wal")]
        let mut wal_recovery: Option<WalRecovery> = None;

        // Rebuild a compact container's default-graph routing before sidecar
        // WAL replay. Recovery must mutate base identities through this exact
        // LayeredStore; constructing it only after replay silently loses base
        // property/label changes, tombstones, and base-endpoint edge creates.
        // The same instance is installed after `GrafeoDB` construction so no
        // replay routing state or one-shot cold-tier binding is discarded.
        #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "compact-store"))]
        let mut loaded_layered_store: Option<
            Arc<grafeo_core::graph::compact::layered::LayeredStore>,
        > = None;

        // Retain section-load decisions that depend on the fully wired
        // database. Projection provenance needs the LPG generation, and an
        // unsealed compact legacy image must rebuild its default auxiliary
        // indexes only after its base and overlay become one layered view.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let mut section_load_outcome = SectionLoadOutcome::default();

        // RDF-only section loading has no LPG outcome object. Retain whether
        // an on-disk Ring existed so recovery can preserve that capability
        // after any sidecar WAL replay has invalidated the decoded cache.
        #[cfg(all(
            feature = "grafeo-file",
            feature = "triple-store",
            feature = "ring-index",
            not(feature = "lpg")
        ))]
        let mut ensure_rdf_ring_after_recovery = false;

        // --- Single-file format (.grafeo) ---
        #[cfg(feature = "grafeo-file")]
        let file_manager: Option<Arc<GrafeoFileManager>> = if is_read_only {
            // Read-only mode: open with shared lock, load snapshot, skip WAL
            if let Some(ref db_path) = config.path {
                if db_path.exists() && db_path.is_file() {
                    let fm = match owned_file.take() {
                        Some(file) => file,
                        None => GrafeoFileManager::open_read_only(db_path)?,
                    };
                    if let Some(floor) = validated_transaction_allocation_floor(
                        fm.active_header().transaction_id,
                        "container header",
                    )? {
                        transaction_manager.advance_next_transaction_id(floor);
                    }
                    Self::apply_graph_model_tag(&mut config, fm.graph_model_tag())?;
                    // Try v2 section-based format first
                    #[cfg(feature = "lpg")]
                    if fm.read_section_directory()?.is_some() {
                        let loaded_sections = Self::load_from_sections(
                            #[cfg(feature = "cdc")]
                            &cdc_log,
                            &fm,
                            &store,
                            &catalog,
                            &world_identity,
                            #[cfg(feature = "triple-store")]
                            &rdf_store,
                            #[cfg(feature = "triple-store")]
                            &rdf_projections,
                        )?;
                        section_load_outcome = loaded_sections.outcome;
                        #[cfg(feature = "compact-store")]
                        {
                            loaded_layered_store = loaded_sections.layered;
                        }
                        validate_default_lpg_recovery_cardinality(
                            &fm.active_header(),
                            section_load_outcome.recovery_image_is_sealed,
                            section_load_outcome.recovery_coordinates_are_sealed,
                            &store,
                            #[cfg(feature = "compact-store")]
                            loaded_layered_store.as_ref(),
                        )?;
                    } else {
                        // Fall back to v1 blob format
                        let snapshot_data = fm.read_snapshot()?;
                        if !snapshot_data.is_empty() {
                            let (_, identity) = Self::apply_snapshot_data(
                                #[cfg(feature = "cdc")]
                                &cdc_log,
                                &store,
                                &catalog,
                                #[cfg(feature = "triple-store")]
                                &rdf_store,
                                #[cfg(feature = "triple-store")]
                                &rdf_projections,
                                &snapshot_data,
                            )?;
                            *world_identity.write() = identity;
                        }
                    }
                    #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
                    if fm.read_section_directory()?.is_some() {
                        let LoadedRdfSectionState {
                            ring_declared,
                            recovery_image_is_sealed,
                            recovery_coordinates_are_sealed,
                            authenticated_replay_epoch,
                            ..
                        } = Self::load_rdf_sections(
                            #[cfg(feature = "cdc")]
                            &cdc_log,
                            &fm,
                            &catalog,
                            &world_identity,
                            &rdf_store,
                        )?;
                        if recovery_coordinates_are_sealed && !recovery_image_is_sealed {
                            return Err(Error::Serialization(
                                "recovery header coordinates cannot be sealed by an unsealed RDF section image"
                                    .to_string(),
                            ));
                        }
                        validate_sealed_recovery_cardinality(
                            &fm.active_header(),
                            recovery_coordinates_are_sealed,
                            0,
                            0,
                            "RDF-only view",
                        )?;
                        #[cfg(feature = "ring-index")]
                        {
                            ensure_rdf_ring_after_recovery = ring_declared;
                        }
                        #[cfg(not(feature = "ring-index"))]
                        let _ = ring_declared;
                        let _ = authenticated_replay_epoch;
                    }
                    Some(Arc::new(fm))
                } else {
                    return Err(grafeo_common::utils::error::Error::Internal(format!(
                        "read-only open requires an existing .grafeo file: {}",
                        db_path.display()
                    )));
                }
            } else {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "read-only mode requires a database path".to_string(),
                ));
            }
        } else if let Some(ref db_path) = config.path {
            // Initialize the file manager whenever single-file format is selected,
            // regardless of whether WAL is enabled. Without this, a database opened
            // with wal_enabled:false + StorageFormat::SingleFile would produce no
            // output at all (the file manager was previously gated behind wal_enabled).
            if use_single_file {
                let fm = if let Some(file) = owned_file.take() {
                    file
                } else if db_path.exists() && db_path.is_file() {
                    GrafeoFileManager::open(db_path)?
                } else if !db_path.exists() {
                    GrafeoFileManager::create_with_graph_model(db_path, config.graph_model.as_u8())?
                } else {
                    // Path exists but is not a file (directory, etc.)
                    return Err(grafeo_common::utils::error::Error::Internal(format!(
                        "path exists but is not a file: {}",
                        db_path.display()
                    )));
                };

                // Validate and adopt the persisted allocator high-water before
                // section/WAL recovery or WAL-manager creation can mutate disk.
                // Legacy headers remain useful for non-reuse, but never become
                // an unauthenticated replay-discard boundary.
                if let Some(floor) = validated_transaction_allocation_floor(
                    fm.active_header().transaction_id,
                    "container header",
                )? {
                    transaction_manager.advance_next_transaction_id(floor);
                }

                #[cfg(feature = "wal")]
                if db_path.exists() && db_path.is_file() {
                    let rec = [WalRecord::GraphModelMeta {
                        model: fm.graph_model_tag(),
                    }];
                    Self::adopt_stored_graph_model(&mut config, &rec)?;
                }
                Self::apply_graph_model_tag(&mut config, fm.graph_model_tag())?;

                // Set only after the active snapshot loads successfully. The
                // sidecar may start after a retired checkpoint prefix, so a
                // malformed active container is never eligible for best-effort
                // WAL reconstruction.
                #[cfg(all(feature = "wal", feature = "lpg"))]
                let snapshot_replay_evidence: Option<SnapshotReplayEvidence>;
                #[cfg(all(feature = "wal", not(feature = "lpg"), feature = "triple-store"))]
                let mut snapshot_replay_evidence = None;
                #[cfg(all(feature = "wal", any(feature = "lpg", feature = "triple-store")))]
                let mut authenticated_container_identity = None;

                // Load data: try v2 section-based format, fall back to v1 blob.
                #[cfg(feature = "lpg")]
                match fm.read_section_directory()? {
                    Some(_) => {
                        let loaded_sections = Self::load_from_sections(
                            #[cfg(feature = "cdc")]
                            &cdc_log,
                            &fm,
                            &store,
                            &catalog,
                            &world_identity,
                            #[cfg(feature = "triple-store")]
                            &rdf_store,
                            #[cfg(feature = "triple-store")]
                            &rdf_projections,
                        )?;
                        section_load_outcome = loaded_sections.outcome;
                        #[cfg(feature = "wal")]
                        {
                            authenticated_container_identity =
                                loaded_sections.authenticated_identity;
                            snapshot_replay_evidence = Some(SnapshotReplayEvidence {
                                authenticated_epoch: section_load_outcome
                                    .authenticated_replay_epoch,
                                recovery_coordinates_are_sealed: section_load_outcome
                                    .recovery_coordinates_are_sealed,
                            });
                        }
                        #[cfg(feature = "compact-store")]
                        {
                            loaded_layered_store = loaded_sections.layered;
                        }
                    }
                    None => {
                        let snapshot_data = fm.read_snapshot()?;
                        if !snapshot_data.is_empty() {
                            let (tag, identity) = Self::apply_snapshot_data(
                                #[cfg(feature = "cdc")]
                                &cdc_log,
                                &store,
                                &catalog,
                                #[cfg(feature = "triple-store")]
                                &rdf_store,
                                #[cfg(feature = "triple-store")]
                                &rdf_projections,
                                &snapshot_data,
                            )?;
                            *world_identity.write() = identity;
                            #[cfg(feature = "wal")]
                            if tag != 255 {
                                let rec = [WalRecord::GraphModelMeta { model: tag }];
                                Self::adopt_stored_graph_model(&mut config, &rec)?;
                            }
                            #[cfg(not(feature = "wal"))]
                            let _ = tag;
                        }
                        #[cfg(feature = "wal")]
                        {
                            snapshot_replay_evidence = if fm.active_header().is_empty() {
                                None
                            } else {
                                let decoded_epoch = decoded_snapshot_epoch(
                                    config.graph_model,
                                    &store,
                                    #[cfg(feature = "triple-store")]
                                    &rdf_store,
                                )?;
                                Some(SnapshotReplayEvidence {
                                    authenticated_epoch: Some(decoded_epoch),
                                    recovery_coordinates_are_sealed: false,
                                })
                            };
                        }
                    }
                }

                #[cfg(feature = "lpg")]
                validate_default_lpg_recovery_cardinality(
                    &fm.active_header(),
                    section_load_outcome.recovery_image_is_sealed,
                    section_load_outcome.recovery_coordinates_are_sealed,
                    &store,
                    #[cfg(feature = "compact-store")]
                    loaded_layered_store.as_ref(),
                )?;

                // Recover sidecar WAL if WAL is enabled and a sidecar exists
                #[cfg(all(feature = "wal", feature = "lpg"))]
                if config.wal_enabled {
                    let mut recovery = match owned_recovery.take() {
                        Some(recovery) => recovery,
                        None => WalRecovery::new(fm.sidecar_wal_path())?,
                    };
                    let RecoveredWal { records, authority } = recover_committed_wal(
                        &mut recovery,
                        &transaction_manager,
                        &config,
                        #[cfg(feature = "cdc")]
                        crate::cdc::wal::RecoverySource {
                            log: &cdc_log,
                            #[cfg(feature = "lpg")]
                            root: &store,
                        },
                        authenticated_container_identity.as_ref(),
                        Some(SnapshotWalRecovery {
                            evidence: snapshot_replay_evidence,
                            header: fm.active_header(),
                        }),
                    )?;
                    Self::adopt_wal_authority(
                        &mut config,
                        &world_identity,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                        authority,
                    )?;
                    Self::apply_wal_records_with_default(
                        &store,
                        #[cfg(feature = "compact-store")]
                        loaded_layered_store.as_ref(),
                        &catalog,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                        #[cfg(feature = "triple-store")]
                        &rdf_projections,
                        &records,
                    )?;
                    wal_recovery = Some(recovery);
                }

                #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
                match fm.read_section_directory()? {
                    Some(_) => {
                        let LoadedRdfSectionState {
                            ring_declared,
                            recovery_image_is_sealed,
                            recovery_coordinates_are_sealed,
                            authenticated_replay_epoch,
                            #[cfg(feature = "wal")]
                            authenticated_identity,
                        } = Self::load_rdf_sections(
                            #[cfg(feature = "cdc")]
                            &cdc_log,
                            &fm,
                            &catalog,
                            &world_identity,
                            &rdf_store,
                        )?;
                        if recovery_coordinates_are_sealed && !recovery_image_is_sealed {
                            return Err(Error::Serialization(
                                "recovery header coordinates cannot be sealed by an unsealed RDF section image"
                                    .to_string(),
                            ));
                        }
                        validate_sealed_recovery_cardinality(
                            &fm.active_header(),
                            recovery_coordinates_are_sealed,
                            0,
                            0,
                            "RDF-only view",
                        )?;
                        #[cfg(feature = "ring-index")]
                        {
                            ensure_rdf_ring_after_recovery = ring_declared;
                        }
                        #[cfg(not(feature = "ring-index"))]
                        let _ = ring_declared;
                        #[cfg(feature = "wal")]
                        {
                            authenticated_container_identity = authenticated_identity;
                            snapshot_replay_evidence = Some(SnapshotReplayEvidence {
                                authenticated_epoch: authenticated_replay_epoch,
                                recovery_coordinates_are_sealed,
                            });
                        }
                        #[cfg(not(feature = "wal"))]
                        let _ = authenticated_replay_epoch;
                    }
                    None => {}
                }

                #[cfg(all(feature = "wal", feature = "triple-store", not(feature = "lpg")))]
                if config.wal_enabled {
                    let mut recovery = match owned_recovery.take() {
                        Some(recovery) => recovery,
                        None => WalRecovery::new(fm.sidecar_wal_path())?,
                    };
                    let RecoveredWal { records, authority } = recover_committed_wal(
                        &mut recovery,
                        &transaction_manager,
                        &config,
                        #[cfg(feature = "cdc")]
                        crate::cdc::wal::RecoverySource {
                            log: &cdc_log,
                            #[cfg(feature = "lpg")]
                            root: &store,
                        },
                        authenticated_container_identity.as_ref(),
                        Some(SnapshotWalRecovery {
                            evidence: snapshot_replay_evidence,
                            header: fm.active_header(),
                        }),
                    )?;
                    Self::adopt_wal_authority(&mut config, &world_identity, &rdf_store, authority)?;
                    rdf_ops::replay_rdf_wal_records(&rdf_store, &records)?;
                    wal_recovery = Some(recovery);
                }

                Some(Arc::new(fm))
            } else {
                None
            }
        } else {
            None
        };

        // Snapshot/section loading may replace the initially generated RDF
        // identity. Publish that exact value to the database before a new WAL
        // is opened, so its first identity record cannot describe a different
        // handle namespace.
        #[cfg(feature = "triple-store")]
        if matches!(
            config.graph_model,
            crate::config::GraphModel::Rdf | crate::config::GraphModel::Both
        ) {
            *world_identity.write() = WorldIdentityMetadataV1::new(
                rdf_store.store_id(),
                rdf_store.history_completeness(),
            )
            .map_err(|error| {
                Error::Internal(format!("invalid recovered database identity: {error}"))
            })?;
        }

        // Determine whether to use the WAL directory path or sidecar.
        // Read-only mode skips WAL entirely (no recovery, no creation).
        #[cfg(feature = "wal")]
        let wal = if is_read_only {
            None
        } else if config.wal_enabled {
            if let Some(ref db_path) = config.path {
                // When using single-file format, the WAL is a sidecar directory
                #[cfg(feature = "grafeo-file")]
                let wal_path = if let Some(ref fm) = file_manager {
                    fm.sidecar_wal_path()
                } else {
                    // Validated WAL ownership provisions missing parents.
                    db_path.join("wal")
                };

                #[cfg(not(feature = "grafeo-file"))]
                let wal_path = db_path.join("wal");

                let mut recovery = match wal_recovery.take() {
                    Some(recovery) => recovery,
                    None => WalRecovery::new(&wal_path)?,
                };

                // Recover the directory while retaining its startup authority.
                #[cfg(feature = "grafeo-file")]
                let is_single_file = file_manager.is_some();
                #[cfg(not(feature = "grafeo-file"))]
                let is_single_file = false;

                #[cfg(feature = "lpg")]
                if !is_single_file {
                    let RecoveredWal { records, authority } = recover_committed_wal(
                        &mut recovery,
                        &transaction_manager,
                        &config,
                        #[cfg(feature = "cdc")]
                        crate::cdc::wal::RecoverySource {
                            log: &cdc_log,
                            #[cfg(feature = "lpg")]
                            root: &store,
                        },
                        None,
                        #[cfg(feature = "grafeo-file")]
                        None,
                    )?;
                    Self::adopt_wal_authority(
                        &mut config,
                        &world_identity,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                        authority,
                    )?;
                    Self::apply_wal_records(
                        &store,
                        &catalog,
                        #[cfg(feature = "triple-store")]
                        &rdf_store,
                        #[cfg(feature = "triple-store")]
                        &rdf_projections,
                        &records,
                    )?;
                }

                #[cfg(all(feature = "wal", feature = "triple-store", not(feature = "lpg")))]
                if !is_single_file {
                    let RecoveredWal { records, authority } = recover_committed_wal(
                        &mut recovery,
                        &transaction_manager,
                        &config,
                        #[cfg(feature = "cdc")]
                        crate::cdc::wal::RecoverySource {
                            log: &cdc_log,
                            #[cfg(feature = "lpg")]
                            root: &store,
                        },
                        None,
                        #[cfg(feature = "grafeo-file")]
                        None,
                    )?;
                    Self::adopt_wal_authority(&mut config, &world_identity, &rdf_store, authority)?;
                    rdf_ops::replay_rdf_wal_records(&rdf_store, &records)?;
                }

                // Open/create WAL manager with configured durability
                let wal_durability = match config.wal_durability {
                    crate::config::DurabilityMode::Sync => WalDurabilityMode::Sync,
                    crate::config::DurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    } => WalDurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    },
                    crate::config::DurabilityMode::Adaptive { target_interval_ms } => {
                        WalDurabilityMode::Adaptive { target_interval_ms }
                    }
                    crate::config::DurabilityMode::NoSync => WalDurabilityMode::NoSync,
                };
                let wal_config = WalConfig {
                    durability: wal_durability,
                    ..WalConfig::default()
                };
                #[cfg(not(any(feature = "lpg", feature = "triple-store")))]
                recovery.recover()?;
                let wal_manager = LpgWal::from_manager(recovery.into_wal(wal_config)?);
                wal_manager.log(&WalRecord::StoreIdentityMeta {
                    metadata: world_identity.read().clone(),
                })?;
                wal_manager.log(&WalRecord::GraphModelMeta {
                    model: config.graph_model.as_u8(),
                })?;
                Some(Arc::new(wal_manager))
            } else {
                None
            }
        } else {
            None
        };

        // Preserve a persisted Ring across recovery without trusting an
        // unsealed cache image. A sealed image remains byte-exact when no RDF
        // WAL mutation followed it; WAL replay invalidates that cache, while
        // an unsealed image is deliberately never installed. Rebuild only
        // when the image declared a Ring and no current Ring survived, after
        // every sidecar or legacy-directory WAL record has been applied.
        #[cfg(all(
            feature = "grafeo-file",
            feature = "triple-store",
            feature = "ring-index",
            feature = "lpg"
        ))]
        if section_load_outcome.ensure_rdf_ring_after_recovery && rdf_store.ring().is_none() {
            rdf_store.rebuild_ring();
        }
        #[cfg(all(
            feature = "grafeo-file",
            feature = "triple-store",
            feature = "ring-index",
            not(feature = "lpg")
        ))]
        if ensure_rdf_ring_after_recovery && rdf_store.ring().is_none() {
            rdf_store.rebuild_ring();
        }

        // Create query cache with default capacity (1000 queries)
        let query_cache = Arc::new(QueryCache::default());
        let physical_cache = crate::query::cache::PhysicalPlanCache::shared(32);

        // After all snapshot/WAL recovery, sync TransactionManager epoch
        // with every store so reopen never reissues an epoch.
        #[cfg(feature = "lpg")]
        transaction_manager
            .try_sync_epoch(store.current_epoch())
            .map_err(|error| {
                Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                    format!("invalid recovered LPG epoch: {error}"),
                ))
            })?;
        #[cfg(feature = "triple-store")]
        transaction_manager
            .try_sync_epoch(rdf_store.commit_epoch())
            .map_err(|error| {
                Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                    format!("invalid recovered RDF epoch: {error}"),
                ))
            })?;

        #[cfg(feature = "triple-store")]
        if matches!(
            config.graph_model,
            crate::config::GraphModel::Rdf | crate::config::GraphModel::Both
        ) {
            *world_identity.write() = WorldIdentityMetadataV1::new(
                rdf_store.store_id(),
                rdf_store.history_completeness(),
            )
            .map_err(|error| {
                Error::Internal(format!("invalid recovered database identity: {error}"))
            })?;
        }

        #[cfg(feature = "cdc")]
        let cdc_enabled_val = config.cdc_enabled;

        // Clone Arcs for the checkpoint timer before moving originals into the struct.
        // The timer captures its own references and runs in a background thread.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let checkpoint_interval = config.checkpoint_interval;
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let timer_store = Arc::clone(&store);
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let timer_catalog = Arc::clone(&catalog);
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let timer_tm = Arc::clone(&transaction_manager);
        #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "triple-store"))]
        let timer_rdf = Arc::clone(&rdf_store);
        #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "triple-store"))]
        let timer_rdf_projections = Arc::clone(&rdf_projections);
        #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "wal"))]
        let timer_wal = wal.clone();

        #[cfg(feature = "spill")]
        let spill_root = Arc::new(crate::spill_crypto::DatabaseSpillRoot::new(&config));

        let mut db = Self {
            config,
            world_identity,
            #[cfg(feature = "lpg")]
            store: Some(store),
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_store,
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            rdf_projections,
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "spill")]
            spill_root,
            #[cfg(feature = "wal")]
            wal,
            query_cache,
            physical_cache,
            #[cfg(feature = "lpg")]
            commit_counter: Arc::new(AtomicUsize::new(0)),
            durability_poisoned: Arc::new(AtomicBool::new(false)),
            is_open: Arc::new(RwLock::new(true)),
            #[cfg(feature = "wal")]
            retained_wal_seal: parking_lot::Mutex::new(None),
            active_sessions: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "lpg")]
            transport_extract_receipts: parking_lot::Mutex::new(
                grafeo_common::utils::hash::FxHashMap::default(),
            ),
            #[cfg(feature = "cdc")]
            cdc_log,
            #[cfg(feature = "cdc")]
            cdc_enabled: std::sync::atomic::AtomicBool::new(cdc_enabled_val),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager,
            #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
            checkpoint_timer: parking_lot::Mutex::new(None),
            external_read_store: None,
            external_write_store: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_context: RwLock::new(crate::session::SessionGraphContext::default()),
            read_only: is_read_only,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            #[cfg(all(feature = "compact-store", feature = "lpg"))]
            layered_store: None,
            #[cfg(feature = "statement-table")]
            statement_table: RwLock::new(
                grafeo_core::graph::compact::statement_table::StatementTable::new(),
            ),
            #[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
            compact_tiered: None,
        };

        // Register storage sections as memory consumers for pressure tracking
        db.register_section_consumers();

        // Phase 5e: if the loaded file has a CompactStore section, the
        // database was previously compacted. Reconstruct the LayeredStore
        // wiring (base + overlay + tier wrapper + consumers) so the
        // engine sees the full picture and the read/write paths route
        // through the layered store.
        #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "compact-store"))]
        if let Some(layered) = loaded_layered_store {
            db.install_layered_after_load(layered)?;
        }

        // Cross-model publication receipts are meaningful only when their
        // materialized LPG rows carry canonical ownership provenance in the
        // fully wired tier-merged store. Validate after section and WAL replay
        // plus compact-base installation, but before any recovered DDL,
        // background timer, sealing, or publication to the caller.
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        db.validate_live_rdf_lpg_projection_rows()?;

        // Start periodic checkpoint timer if configured
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        #[cfg(feature = "compact-store")]
        let periodic_checkpoint_topology_supported = db.layered_store.is_none();
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        #[cfg(not(feature = "compact-store"))]
        let periodic_checkpoint_topology_supported = true;
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        if let (Some(interval), Some(fm)) = (checkpoint_interval, &db.file_manager)
            && !is_read_only
            && periodic_checkpoint_topology_supported
        {
            *db.checkpoint_timer.lock() = Some(checkpoint_timer::CheckpointTimer::start(
                #[cfg(feature = "cdc")]
                Arc::clone(&db.cdc_log),
                interval,
                Arc::clone(fm),
                timer_store,
                timer_catalog,
                timer_tm,
                Arc::clone(&db.durability_poisoned),
                Arc::clone(&db.world_identity),
                db.config.graph_model,
                #[cfg(feature = "triple-store")]
                timer_rdf,
                #[cfg(feature = "triple-store")]
                timer_rdf_projections,
                #[cfg(feature = "wal")]
                timer_wal,
            ));
        }

        // Phase 8a: apply per-section ForceDisk overrides. Each section
        // type configured as ForceDisk triggers a targeted spill of its
        // matching consumer; sections with Auto/ForceRam are left alone.
        // Must happen after register_section_consumers() which creates
        // the consumers we're about to spill.
        db.apply_force_disk_overrides();

        #[cfg(all(feature = "lpg", any(feature = "wal", feature = "grafeo-file")))]
        if db.requires_store_authority()
            && !db
                .store_arc()
                .seal_unframed_writes(db.transaction_manager.write_authority())
        {
            return Err(Error::Internal(
                "LPG store was sealed by a different database authority".into(),
            ));
        }
        #[cfg(all(
            feature = "triple-store",
            any(feature = "wal", feature = "grafeo-file")
        ))]
        if db.requires_store_authority()
            && !db
                .rdf_store
                .seal_unframed_writes(db.transaction_manager.write_authority())
        {
            return Err(Error::Internal(
                "RDF store was sealed by a different database authority".into(),
            ));
        }

        Ok(db)
    }

    /// Creates a database backed by a custom [`GraphStoreMut`] implementation.
    ///
    /// The external store handles all data persistence. WAL, CDC, and index
    /// management are the responsibility of the store implementation.
    ///
    /// Query execution (all 6 languages, optimizer, planner) works through the
    /// provided store. Admin operations (schema introspection, persistence,
    /// vector/text indexes) are not available on external stores.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use grafeo_engine::{GrafeoDB, Config};
    /// use grafeo_core::graph::GraphStoreMut;
    ///
    /// fn example(store: Arc<dyn GraphStoreMut>) -> grafeo_common::utils::error::Result<()> {
    ///     let db = GrafeoDB::with_store(store, Config::in_memory())?;
    ///     let result = db.execute("MATCH (n) RETURN count(n)")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if config validation fails.
    ///
    /// [`GraphStoreMut`]: grafeo_core::graph::GraphStoreMut
    pub fn with_store(store: Arc<dyn GraphStoreMut>, config: Config) -> Result<Self> {
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        let transaction_manager = Arc::new(TransactionManager::new());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                // reason: product of system RAM and 0.75 is always a valid positive usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b = (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize;
                b
            }),
            spill_path: None,
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        let query_cache = Arc::new(QueryCache::default());
        let physical_cache = crate::query::cache::PhysicalPlanCache::shared(32);
        let world_identity = Arc::new(RwLock::new(fresh_world_identity()?));

        #[cfg(feature = "cdc")]
        let cdc_enabled_val = config.cdc_enabled;

        #[cfg(feature = "spill")]
        let spill_root = Arc::new(crate::spill_crypto::DatabaseSpillRoot::new(&config));

        Ok(Self {
            config,
            world_identity: Arc::clone(&world_identity),
            #[cfg(feature = "lpg")]
            store: None,
            catalog: Arc::new(Catalog::new()),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::with_config_and_store_id(
                grafeo_core::graph::rdf::RdfStoreConfig::default(),
                world_identity.read().store_id(),
            )),
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            rdf_projections: Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new()),
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "spill")]
            spill_root,
            #[cfg(feature = "wal")]
            wal: None,
            query_cache,
            physical_cache,
            #[cfg(feature = "lpg")]
            commit_counter: Arc::new(AtomicUsize::new(0)),
            durability_poisoned: Arc::new(AtomicBool::new(false)),
            is_open: Arc::new(RwLock::new(true)),
            #[cfg(feature = "wal")]
            retained_wal_seal: parking_lot::Mutex::new(None),
            active_sessions: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "lpg")]
            transport_extract_receipts: parking_lot::Mutex::new(
                grafeo_common::utils::hash::FxHashMap::default(),
            ),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_enabled: std::sync::atomic::AtomicBool::new(cdc_enabled_val),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager: None,
            #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
            checkpoint_timer: parking_lot::Mutex::new(None),
            external_read_store: Some(Arc::clone(&store) as Arc<dyn GraphStoreSearch>),
            external_write_store: Some(store),
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_context: RwLock::new(crate::session::SessionGraphContext::default()),
            read_only: false,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            #[cfg(all(feature = "compact-store", feature = "lpg"))]
            layered_store: None,
            #[cfg(feature = "statement-table")]
            statement_table: RwLock::new(
                grafeo_core::graph::compact::statement_table::StatementTable::new(),
            ),
            #[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
            compact_tiered: None,
        })
    }

    /// Creates a database backed by a read-only [`GraphStore`].
    ///
    /// The database is set to read-only mode. Write queries (CREATE, SET,
    /// DELETE) will return `TransactionError::ReadOnly`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use grafeo_engine::{GrafeoDB, Config};
    /// use grafeo_core::graph::GraphStoreSearch;
    ///
    /// fn example(store: Arc<dyn GraphStoreSearch>) -> grafeo_common::utils::error::Result<()> {
    ///     let db = GrafeoDB::with_read_store(store, Config::in_memory())?;
    ///     let result = db.execute("MATCH (n) RETURN count(n)")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if config validation fails.
    ///
    /// [`GraphStore`]: grafeo_core::graph::GraphStore
    pub fn with_read_store(store: Arc<dyn GraphStoreSearch>, config: Config) -> Result<Self> {
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        let transaction_manager = Arc::new(TransactionManager::new());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                // reason: product of system RAM and 0.75 is always a valid positive usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b = (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize;
                b
            }),
            spill_path: None,
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        let query_cache = Arc::new(QueryCache::default());
        let physical_cache = crate::query::cache::PhysicalPlanCache::shared(32);
        let world_identity = Arc::new(RwLock::new(fresh_world_identity()?));

        #[cfg(feature = "cdc")]
        let cdc_enabled_val = config.cdc_enabled;

        #[cfg(feature = "spill")]
        let spill_root = Arc::new(crate::spill_crypto::DatabaseSpillRoot::new(&config));

        Ok(Self {
            config,
            world_identity: Arc::clone(&world_identity),
            #[cfg(feature = "lpg")]
            store: None,
            catalog: Arc::new(Catalog::new()),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::with_config_and_store_id(
                grafeo_core::graph::rdf::RdfStoreConfig::default(),
                world_identity.read().store_id(),
            )),
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            rdf_projections: Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new()),
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "spill")]
            spill_root,
            #[cfg(feature = "wal")]
            wal: None,
            query_cache,
            physical_cache,
            #[cfg(feature = "lpg")]
            commit_counter: Arc::new(AtomicUsize::new(0)),
            durability_poisoned: Arc::new(AtomicBool::new(false)),
            is_open: Arc::new(RwLock::new(true)),
            #[cfg(feature = "wal")]
            retained_wal_seal: parking_lot::Mutex::new(None),
            active_sessions: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "lpg")]
            transport_extract_receipts: parking_lot::Mutex::new(
                grafeo_common::utils::hash::FxHashMap::default(),
            ),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_enabled: std::sync::atomic::AtomicBool::new(cdc_enabled_val),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager: None,
            #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
            checkpoint_timer: parking_lot::Mutex::new(None),
            external_read_store: Some(store),
            external_write_store: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_context: RwLock::new(crate::session::SessionGraphContext::default()),
            read_only: true,
            projections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            #[cfg(all(feature = "compact-store", feature = "lpg"))]
            layered_store: None,
            #[cfg(feature = "statement-table")]
            statement_table: RwLock::new(
                grafeo_core::graph::compact::statement_table::StatementTable::new(),
            ),
            #[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
            compact_tiered: None,
        })
    }

    /// Folds committed LPG history into a columnar base with a writable overlay.
    ///
    /// Call this between transactions after dropping every live Session. It is
    /// repeatable: the first call converts the native store; later calls fold
    /// both the existing cold base and new overlay history through the same
    /// temporal generation transfer. Retained Layered views keep their owner.
    /// Node/edge identities, named graphs and exact indexes are preserved.
    ///
    /// A failed preparation leaves the native or layered representation usable
    /// for retry. An external read store supplies only its current snapshot,
    /// not historical versions. Successful conversion remains writable.
    ///
    /// Compaction is not a durability checkpoint. The flat-only periodic timer
    /// is stopped after successful conversion; use `wal_checkpoint()` or
    /// `close()` to persist the complete layered state.
    ///
    /// # Errors
    ///
    /// Rejects closed, durability-poisoned or non-quiescent databases and live
    /// Sessions. Conversion, allocation and temporal preparation errors are
    /// returned without publishing a partial representation.
    ///
    /// The former first/repeat method split is not part of this API:
    ///
    /// ```compile_fail
    /// # use grafeo_engine::GrafeoDB;
    /// let mut db = GrafeoDB::new_in_memory();
    /// db.recompact();
    /// ```
    ///
    /// ```compile_fail
    /// # use grafeo_engine::GrafeoDB;
    /// let mut db = GrafeoDB::new_in_memory();
    /// db.maybe_recompact();
    /// ```
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    pub fn compact(&mut self) -> Result<()> {
        use grafeo_core::graph::compact::from_graph_store_preserving_ids;
        use grafeo_core::graph::compact::layered::LayeredStore;

        // Declare retirement before the gates: even on unwind the timer joins
        // only after every gate its worker may need has drained.
        #[cfg(feature = "grafeo-file")]
        let retired_checkpoint_timer;
        let lifecycle = Arc::clone(&self.is_open);
        let open = lifecycle.read();
        if !*open {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "cannot compact a closed database".into(),
            )));
        }
        // An external read-store constructor explicitly permits conversion to
        // a new writable in-memory representation. A read-only database open
        // does not grant that authority or permission to replace its stores.
        if self.read_only && self.store.is_some() {
            return Err(TransactionError::ReadOnly.into());
        }
        if self.is_durability_poisoned() {
            return Err(Error::Transaction(TransactionError::DurabilityFailure(
                "cannot compact a durability-poisoned database; reopen and recover first".into(),
            )));
        }
        self.require_no_sessions("compact")?;
        #[cfg(test)]
        publication_cut_test_point();
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::clone(&self.rdf_store);
        #[cfg(feature = "triple-store")]
        let rdf_gate = rdf_store.lock_commit();
        let transaction_manager = Arc::clone(&self.transaction_manager);
        let publication = transaction_manager.publication().write();
        self.require_no_sessions("compact")?;
        self.require_quiescent("compact")?;
        if self.is_durability_poisoned() {
            return Err(Error::Transaction(TransactionError::DurabilityFailure(
                "cannot compact a durability-poisoned database; reopen and recover first".into(),
            )));
        }

        let current_epoch = transaction_manager.current_epoch();
        let first_compaction = self.layered_store.is_none();
        #[cfg(feature = "testing-crash-injection")]
        grafeo_common::testing::crash::maybe_crash("compact:before_representation_transfer");
        let layered = if let Some(layered) = &self.layered_store {
            let layered = Arc::clone(layered);
            transaction_manager
                .with_write_authority(|| {
                    // Tier metadata and the Layered base share one reversible
                    // publication. Keep the existing wrapper and consumer so
                    // long-lived views observe every later generation.
                    #[cfg(feature = "mmap")]
                    if let Some(tiered) = &self.compact_tiered {
                        let retired = layered.merge_overlay_temporal_with_publication(
                            |base| {
                                let prepared =
                                    tiered.prepare_generation_in_memory(Arc::clone(&base));
                                Ok((base, prepared))
                            },
                            |prepared| tiered.publish_prepared_reversibly(prepared),
                            |published| tiered.restore_published_reversibly(published),
                        )?;
                        retired.commit();
                        return Ok(());
                    }
                    layered.merge_overlay_temporal()
                })
                .map_err(Error::Internal)?;
            layered
        } else if let Some(overlay) = self.store.clone() {
            // The source stays native and unbound until the existing exact
            // representation transfer has prepared and published its successor.
            Arc::new(
                transaction_manager
                    .with_write_authority(|| LayeredStore::from_native_temporal(overlay))?,
            )
        } else {
            let current_store = self.graph_store();
            let max_node_id = current_store.node_ids().last().map_or(0, |id| id.as_u64());
            let mut max_edge_id = 0u64;
            for nid in current_store.node_ids() {
                for (_, eid) in
                    current_store.edges_from(nid, grafeo_core::graph::Direction::Outgoing)
                {
                    max_edge_id = max_edge_id.max(eid.as_u64());
                }
            }
            let compact = from_graph_store_preserving_ids(current_store.as_ref())
                .map_err(|error| Error::Internal(error.to_string()))?;
            Arc::new(LayeredStore::new(compact, max_node_id, max_edge_id)?)
        };

        #[cfg(feature = "testing-crash-injection")]
        grafeo_common::testing::crash::maybe_crash("compact:after_representation_transfer");
        transaction_manager
            .with_write_authority(|| layered.overlay_store().sync_epoch(current_epoch));
        self.external_read_store = Some(Arc::clone(&layered) as Arc<dyn GraphStoreSearch>);
        self.external_write_store = Some(Arc::clone(&layered) as Arc<dyn GraphStoreMut>);
        self.store = Some(layered.overlay_store());

        #[cfg(feature = "mmap")]
        if self.compact_tiered.is_none() {
            let tiered = Arc::new(compact_tiered::CompactStoreTiered::new_in_memory(
                layered.base_store_arc(),
            ));
            let spill_path = self.buffer_manager.config().spill_path.clone();
            let consumer = Arc::new(section_consumer::CompactStoreConsumer::new(
                &tiered,
                &layered,
                &transaction_manager,
                spill_path,
            ));
            self.buffer_manager.register_consumer(consumer);
            self.compact_tiered = Some(tiered);
        }

        self.layered_store = Some(layered);
        self.read_only = false;
        self.query_cache.clear();
        self.physical_cache.lock().clear();
        if first_compaction {
            self.projections.write().clear();
        }

        // A timer already waiting on publication must observe this signal
        // under that same gate before it can checkpoint the retired native store.
        #[cfg(feature = "grafeo-file")]
        {
            retired_checkpoint_timer = self.checkpoint_timer.lock().take();
            if let Some(timer) = &retired_checkpoint_timer {
                timer.request_stop();
            }
        }
        drop(publication);
        #[cfg(feature = "triple-store")]
        drop(rdf_gate);
        drop(open);
        #[cfg(feature = "grafeo-file")]
        drop(retired_checkpoint_timer);
        Ok(())
    }

    /// Runs explicit compaction when the configured overlay threshold is exceeded.
    ///
    /// Before the first compaction the native store is the input; later calls
    /// count the live nodes and edges in the overlay. Returns `false` when
    /// [`Config::compaction_overlay_threshold`](crate::Config) is disabled,
    /// the count does not exceed it, a transaction is active, or the database
    /// wraps an external read store. This is an application-controlled checkpoint,
    /// not an automatic per-commit hook.
    ///
    /// # Errors
    ///
    /// When the threshold is exceeded, propagates all admission and preparation
    /// errors from [`compact()`](Self::compact), including live Sessions.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    pub fn compact_if_needed(&mut self) -> Result<bool> {
        let Some(threshold) = self.config.compaction_overlay_threshold else {
            return Ok(false);
        };
        if self.transaction_manager.active_count() > 0 {
            return Ok(false);
        }
        let Some(overlay) = self.store.as_ref() else {
            return Ok(false);
        };
        if overlay.node_count().saturating_add(overlay.edge_count()) <= threshold {
            return Ok(false);
        }
        self.compact()?;
        Ok(true)
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn apply_lpg_mutation_op(target: &LpgRecoveryTarget, op: &LpgMutationOp) -> Result<()> {
        match op {
            LpgMutationOp::CreateNode { id, labels } => {
                let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                target.recover_create_node_with_id(*id, &label_refs)?;
            }
            LpgMutationOp::DeleteNode { id } => {
                target.mutation_store().delete_node(*id);
            }
            LpgMutationOp::CreateEdge {
                id,
                src,
                dst,
                edge_type,
            } => {
                target.recover_create_edge_with_id(*id, *src, *dst, edge_type)?;
            }
            LpgMutationOp::DeleteEdge { id } => {
                target.mutation_store().delete_edge(*id);
            }
            LpgMutationOp::SetNodeProperty { id, key, value } => {
                target
                    .mutation_store()
                    .set_node_property(*id, key, value.clone());
            }
            LpgMutationOp::SetEdgeProperty { id, key, value } => {
                target
                    .mutation_store()
                    .set_edge_property(*id, key, value.clone());
            }
            LpgMutationOp::AddNodeLabel { id, label } => {
                target.mutation_store().add_label(*id, label);
            }
            LpgMutationOp::RemoveNodeLabel { id, label } => {
                target.mutation_store().remove_label(*id, label);
            }
            LpgMutationOp::RemoveNodeProperty { id, key } => {
                target.mutation_store().remove_node_property(*id, key);
            }
            LpgMutationOp::RemoveEdgeProperty { id, key } => {
                target.mutation_store().remove_edge_property(*id, key);
            }
            LpgMutationOp::NodeLabelImages { .. } | LpgMutationOp::PublishGraph => {
                return Err(Error::Serialization(
                    "committed label/graph publication requires qualified replay scheduling".into(),
                ));
            }
        }
        Ok(())
    }

    #[cfg(feature = "grafeo-file")]
    fn apply_graph_model_tag(config: &mut Config, tag: u8) -> Result<()> {
        let stored = crate::config::GraphModel::from_u8(tag).ok_or_else(|| {
            Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                format!("invalid container graph model tag {tag}"),
            ))
        })?;
        if config.graph_model_pinned && config.graph_model != stored {
            return Err(Error::InvalidValue(format!(
                "graph model mismatch: file is {}, config requested {}",
                stored.as_name(),
                config.graph_model.as_name()
            )));
        }
        if !config.graph_model_pinned {
            config.graph_model = stored;
        }
        config.validate_resolved_graph_model().map_err(|error| {
            Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                format!(
                    "container declares unsupported graph model {}: {error}",
                    stored.as_name()
                ),
            ))
        })?;
        Ok(())
    }

    #[cfg(feature = "wal")]
    fn adopt_stored_graph_model(config: &mut Config, records: &[WalRecord]) -> Result<()> {
        let mut stored = None;
        for record in records {
            if let WalRecord::GraphModelMeta { model } = record {
                if stored.is_some_and(|known| known != *model) {
                    return Err(Error::Serialization(
                        "WAL contains conflicting graph model metadata".to_string(),
                    ));
                }
                stored = Some(*model);
            }
        }
        let Some(tag) = stored else {
            return Ok(());
        };
        let stored = crate::config::GraphModel::from_u8(tag).ok_or_else(|| {
            Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                format!("invalid stored WAL graph model tag {tag}"),
            ))
        })?;
        if config.graph_model_pinned && config.graph_model != stored {
            return Err(Error::InvalidValue(format!(
                "graph model mismatch: file is {}, config requested {}",
                stored.as_name(),
                config.graph_model.as_name()
            )));
        }
        let mut resolved = config.clone();
        resolved.graph_model = stored;
        resolved.validate_resolved_graph_model().map_err(|error| {
            Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                format!(
                    "WAL declares unsupported graph model {}: {error}",
                    stored.as_name()
                ),
            ))
        })?;
        config.graph_model = stored;
        Ok(())
    }

    /// Install previously resolved authority into detached startup state.
    #[cfg(feature = "wal")]
    fn adopt_wal_authority(
        config: &mut Config,
        world_identity: &Arc<RwLock<WorldIdentityMetadataV1>>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        authority: WalAuthority,
    ) -> Result<()> {
        if let Some(metadata) = authority.identity {
            #[cfg(feature = "triple-store")]
            rdf_store
                .adopt_recovery_identity(&metadata)
                .map_err(|error| {
                    Error::Serialization(format!("adopt WAL store identity: {error}"))
                })?;
            *world_identity.write() = metadata;
        }
        config.graph_model = authority.graph_model;
        Ok(())
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn missing_wal_graph(graph: &grafeo_common::types::GraphPath) -> Error {
        Error::Storage(grafeo_common::utils::error::StorageError::InvalidWalEntry(
            format!("WAL targets absent named LPG graph {graph:?}"),
        ))
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn resolve_wal_graph(
        root: &Arc<LpgStore>,
        path: &grafeo_common::types::GraphPath,
    ) -> Result<Arc<LpgStore>> {
        let mut target = Arc::clone(root);
        for component in path.components() {
            target = target
                .graph(component)
                .ok_or_else(|| Self::missing_wal_graph(path))?;
        }
        Ok(target)
    }

    /// Applies WAL records to restore the database state.
    ///
    /// Every LPG record carries its exact component-qualified graph path.
    /// Data records resolve existing topology only; explicit lifecycle records
    /// create children after their parents. Root and the empty child differ.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn apply_wal_records(
        store: &Arc<LpgStore>,
        catalog: &Catalog,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: &Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
        records: &[WalRecord],
    ) -> Result<()> {
        Self::apply_wal_records_with_default(
            store,
            #[cfg(feature = "compact-store")]
            None,
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_store,
            #[cfg(feature = "triple-store")]
            rdf_projections,
            records,
        )
    }

    /// Applies WAL records with an optional preconstructed compact destination
    /// for the default LPG graph. Named graphs always remain flat stores.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    fn apply_wal_records_with_default(
        store: &Arc<LpgStore>,
        #[cfg(feature = "compact-store")] default_layered: Option<
            &Arc<grafeo_core::graph::compact::layered::LayeredStore>,
        >,
        catalog: &Catalog,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: &Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
        records: &[WalRecord],
    ) -> Result<()> {
        use crate::catalog::{
            EdgeTypeDefinition, NamedConstraintDefinition, NamedConstraintKind, NodeTypeDefinition,
            PropertyDataType, TypeConstraint, TypedProperty,
        };
        use grafeo_common::utils::error::Error;

        // Recovery runs before the newly constructed stores are sealed, so it
        // does not need a runtime mutation authority.
        #[cfg(feature = "compact-store")]
        let default_target = match default_layered {
            Some(layered) => LpgRecoveryTarget::Layered(Arc::clone(layered)),
            None => LpgRecoveryTarget::Flat(Arc::clone(store)),
        };
        #[cfg(not(feature = "compact-store"))]
        let default_target = LpgRecoveryTarget::Flat(Arc::clone(store));

        let mut owner_replay = owner_replay::OwnerReplay::new(store, &default_target, catalog)?;

        // Resolve every tagged transaction to the exact epoch carried by its
        // durable commit marker before applying any LPG mutation. Recovery used
        // to call `new_epoch()` when it encountered a marker, which collapsed
        // deliberately reserved epoch gaps and stamped the transaction's data
        // at the *preceding* store epoch. Named-graph mutations were worse: an
        // identity-carrying record does not update the legacy graph cursor, so
        // the marker could advance an unrelated graph.
        //
        // The legacy TransactionCommit + EpochAdvance pair is retained for old
        // WALs. `WalRecovery` emits each transaction as a contiguous logical
        // group, so the following EpochAdvance belongs to the last legacy
        // TransactionCommit in that group.
        let mut commit_epochs = std::collections::HashMap::<
            grafeo_common::types::TransactionId,
            grafeo_common::types::EpochId,
        >::new();
        let mut legacy_commit = None;
        for record in records {
            match record {
                WalRecord::Committed {
                    transaction_id,
                    epoch,
                }
                | WalRecord::CommittedWithCdc {
                    transaction_id,
                    epoch,
                    ..
                } => {
                    commit_epochs.insert(*transaction_id, *epoch);
                    legacy_commit = None;
                }
                WalRecord::TransactionCommit { transaction_id } => {
                    legacy_commit = Some(*transaction_id);
                }
                WalRecord::EpochAdvance { epoch } => {
                    if let Some(transaction_id) = legacy_commit.take() {
                        commit_epochs.insert(transaction_id, *epoch);
                    }
                }
                _ => {}
            }
        }

        // A single current envelope witnesses even a normalized Text no-op.
        // Resolve membership without retaining a second decoded recovery tail.
        let mut index_transactions = std::collections::HashSet::new();
        let mut catalog_transactions = std::collections::HashSet::new();
        for record in records {
            if let WalRecord::CatalogPostimage { transaction_id, .. } = record
                && (!commit_epochs.contains_key(transaction_id)
                    || !catalog_transactions.insert(*transaction_id))
            {
                return Err(Error::Serialization(
                    "duplicate or uncommitted catalog postimage".into(),
                ));
            }
            if let WalRecord::IndexOwnerBatch { transaction_id, .. } = record
                && (!commit_epochs.contains_key(transaction_id)
                    || !index_transactions.insert(*transaction_id))
            {
                return Err(Error::Serialization(
                    "duplicate or uncommitted index publication batch".into(),
                ));
            }
        }

        let initial =
            grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store)).capture_graphs()?;
        let initial_graphs: Vec<_> = initial.iter().map(|(path, _)| path.clone()).collect();
        let label_schedule =
            label_replay::LabelReplaySchedule::prepare(records, &initial_graphs, &commit_epochs)?;
        graph_replay::validate(&initial, store.next_graph_incarnation_id(), records)?;
        let mutation_epoch = |transaction_id: &grafeo_common::types::TransactionId| {
            if *transaction_id == grafeo_common::types::TransactionId::SYSTEM {
                return Ok(None);
            }
            commit_epochs
                .get(transaction_id)
                .copied()
                .map(Some)
                .ok_or_else(|| Error::Serialization("LPG mutation lacks a committed epoch".into()))
        };

        let sync_flat_replay_epoch =
            |target: &LpgStore, epoch: Option<grafeo_common::types::EpochId>| {
                if let Some(epoch) = epoch {
                    store.sync_epoch(epoch);
                    target.sync_epoch(epoch);
                }
            };
        let sync_replay_epoch =
            |target: &LpgRecoveryTarget, epoch: Option<grafeo_common::types::EpochId>| {
                if let Some(epoch) = epoch {
                    store.sync_epoch(epoch);
                    target.sync_epoch(epoch);
                }
            };

        #[cfg(feature = "triple-store")]
        let sync_projection_epoch = |epoch: grafeo_common::types::EpochId| -> Result<()> {
            store.sync_epoch(epoch);
            for graph in store.named_graph_entries().values() {
                graph.sync_epoch(epoch);
            }
            rdf_store.try_set_commit_epoch(epoch).map_err(|error| {
                Error::Storage(grafeo_common::utils::error::StorageError::InvalidWalEntry(
                    error.to_string(),
                ))
            })?;
            Ok(())
        };

        // RDF transaction-time is carried by the commit record that follows
        // its mutations, so resolve and replay the RDF batch before the
        // record-at-a-time LPG pass.
        #[cfg(feature = "triple-store")]
        rdf_ops::replay_rdf_wal_records(rdf_store, records)?;

        for (record_index, record) in records.iter().enumerate() {
            match record {
                WalRecord::CatalogPostimage { payload, .. } => {
                    let candidate =
                        Catalog::decode_current_state_v2(payload).map_err(Error::Serialization)?;
                    let mut workspace = crate::catalog::CatalogWorkspace::replacement(candidate);
                    catalog
                        .prepare_transaction_metadata_replacement(&mut workspace)
                        .map_err(|error| Error::Serialization(error.to_string()))?
                        .install()
                        .finish();
                }
                WalRecord::IndexOwnerBatch {
                    transaction_id,
                    payload,
                } => {
                    let epoch = commit_epochs.get(transaction_id).copied().ok_or_else(|| {
                        Error::Serialization("owner batch lacks committed transaction".into())
                    })?;
                    owner_replay.apply(
                        store,
                        &default_target,
                        catalog,
                        payload,
                        *transaction_id,
                        epoch,
                    )?;
                }
                WalRecord::CreateLpgGraph {
                    graph,
                    incarnation,
                    transaction_id,
                }
                | WalRecord::DropLpgGraph {
                    graph,
                    incarnation,
                    transaction_id,
                } => {
                    let epoch = mutation_epoch(transaction_id)?;
                    let parent_path = graph
                        .parent()
                        .map_err(|error| Error::Serialization(error.to_string()))?
                        .ok_or_else(|| {
                            Error::Serialization("root LPG graph has no lifecycle parent".into())
                        })?;
                    let parent = Self::resolve_wal_graph(store, &parent_path)?;
                    let name = graph.components().last().ok_or_else(|| {
                        Error::Serialization("root LPG graph has no lifecycle name".into())
                    })?;
                    sync_flat_replay_epoch(&parent, epoch);
                    if matches!(record, WalRecord::CreateLpgGraph { .. }) {
                        let created = Arc::new(parent.new_replayed_graph_candidate(*incarnation)?);
                        if !parent.install_graph_if_absent(name, Arc::clone(&created)) {
                            return Err(Error::Serialization(
                                "validated WAL graph could not be installed".into(),
                            ));
                        }
                        sync_flat_replay_epoch(&created, epoch);
                    } else {
                        if let Some(child) = parent.graph(name) {
                            sync_flat_replay_epoch(&child, epoch);
                        }
                        parent.drop_graph(name);
                        for (bound_path, _) in catalog.all_graph_type_bindings() {
                            if bound_path.components().starts_with(graph.components()) {
                                catalog
                                    .replay_graph_type_binding(&bound_path, None)
                                    .map_err(|error| {
                                        Error::Serialization(format!(
                                            "cannot retire graph type binding for {bound_path:?}: {error}"
                                        ))
                                    })?;
                            }
                        }
                    }
                }
                WalRecord::SetGraphTypeBinding {
                    graph,
                    graph_type,
                    transaction_id,
                } => {
                    // This transaction's checked complete metadata image owns
                    // its final bindings. Statement deltas can refer to types
                    // created privately earlier in the same transaction, so
                    // replaying them before that image is neither necessary
                    // nor valid. Graph/data lifecycle records still replay.
                    if catalog_transactions.contains(transaction_id) {
                        continue;
                    }
                    let _ = mutation_epoch(transaction_id)?;
                    if graph_type.is_some() {
                        Self::resolve_wal_graph(store, graph)?;
                    }
                    catalog
                        .replay_graph_type_binding(graph, graph_type.clone())
                        .map_err(|error| {
                            Error::Internal(format!(
                                "cannot replay graph type binding for {graph:?}: {error}"
                            ))
                        })?;
                }
                WalRecord::Reserved15
                | WalRecord::Reserved16
                | WalRecord::Reserved17
                | WalRecord::Reserved18
                | WalRecord::Reserved19
                | WalRecord::Reserved26
                | WalRecord::Reserved28
                | WalRecord::Reserved31
                | WalRecord::Reserved41
                | WalRecord::Reserved36
                | WalRecord::Reserved37
                | WalRecord::Reserved39
                | WalRecord::Reserved40 => {
                    return Err(Error::Serialization("reserved WAL record tag".into()));
                }
                #[cfg(feature = "triple-store")]
                WalRecord::RdfLpgProjectionDeclaredV3 {
                    projection_id,
                    mapping_digest,
                    mapping_format_version,
                    source_graph,
                    type_iri,
                    node_label,
                    epoch,
                } => {
                    if epoch.as_u64() == 0 || *epoch == grafeo_common::types::EpochId::PENDING {
                        return Err(Error::Serialization(
                            "RDF→LPG projection V3 declaration has an invalid publication epoch"
                                .into(),
                        ));
                    }
                    rdf_projections
                        .install_definition_v3(
                            *projection_id,
                            *mapping_digest,
                            *mapping_format_version,
                            source_graph.as_deref(),
                            type_iri,
                            node_label,
                        )
                        .map_err(|error| {
                            Error::Serialization(format!(
                                "cannot replay RDF→LPG projection V3 declaration: {error}"
                            ))
                        })?;
                    sync_projection_epoch(*epoch)?;
                }
                #[cfg(feature = "triple-store")]
                WalRecord::RdfLpgProjectionPublishedV3 {
                    transaction_id,
                    receipt,
                } => {
                    let commit_epoch = commit_epochs.get(transaction_id).copied().ok_or_else(|| {
                        Error::Serialization(format!(
                            "RDF→LPG projection V3 receipt belongs to transaction {transaction_id} without a durable commit marker"
                        ))
                    })?;
                    let receipt = grafeo_core::graph::rdf::RdfLpgProjectionReceipt::decode(receipt)
                        .map_err(|error| {
                            Error::Serialization(format!(
                                "cannot decode RDF→LPG projection V3 receipt: {error}"
                            ))
                        })?;
                    if receipt.target_epoch() != commit_epoch {
                        return Err(Error::Serialization(format!(
                            "RDF→LPG projection V3 receipt target epoch {} disagrees with owning transaction {transaction_id} commit epoch {}",
                            receipt.target_epoch().as_u64(),
                            commit_epoch.as_u64()
                        )));
                    }
                    rdf_projections
                        .install_receipt(rdf_store.store_id(), receipt)
                        .map_err(|error| {
                            Error::Serialization(format!(
                                "cannot replay RDF→LPG projection V3 receipt: {error}"
                            ))
                        })?;
                    sync_projection_epoch(commit_epoch)?;
                }
                #[cfg(not(feature = "triple-store"))]
                WalRecord::RdfLpgProjectionDeclaredV3 { .. }
                | WalRecord::RdfLpgProjectionPublishedV3 { .. } => {}
                WalRecord::LpgMutation {
                    transaction_id,
                    graph,
                    op,
                } => {
                    let epoch = mutation_epoch(transaction_id)?;
                    let dest = if graph.components().is_empty() {
                        default_target.clone()
                    } else {
                        LpgRecoveryTarget::Flat(Self::resolve_wal_graph(store, graph)?)
                    };
                    sync_replay_epoch(&dest, epoch);
                    dest.with_index_replay(index_transactions.contains(transaction_id), || {
                        label_schedule.apply(record_index, &dest, op)
                    })?;
                }

                WalRecord::CatalogBatchV2 { version, records } => {
                    if *version != 1
                        || records.iter().any(|record| {
                            !matches!(
                                record,
                                WalRecord::CreateNodeType { .. }
                                    | WalRecord::DropNodeType { .. }
                                    | WalRecord::CreateEdgeType { .. }
                                    | WalRecord::DropEdgeType { .. }
                                    | WalRecord::CreateConstraint { .. }
                                    | WalRecord::DropConstraint { .. }
                                    | WalRecord::CreateGraphType { .. }
                                    | WalRecord::DropGraphType { .. }
                                    | WalRecord::CreateSchema { .. }
                                    | WalRecord::DropSchema { .. }
                                    | WalRecord::AlterNodeType { .. }
                                    | WalRecord::AlterEdgeType { .. }
                                    | WalRecord::AlterGraphType { .. }
                                    | WalRecord::CreateProcedure { .. }
                                    | WalRecord::DropProcedure { .. }
                            )
                        })
                    {
                        return Err(Error::Serialization(
                            "invalid or unsupported CatalogBatchV2 payload".to_string(),
                        ));
                    }
                    let mut workspace = crate::catalog::CatalogWorkspace::new();
                    let edit = catalog
                        .prepare_edit(&mut workspace)
                        .map_err(|error| Error::Serialization(error.to_string()))?;
                    let applied = Self::apply_wal_records_with_default(
                        store,
                        #[cfg(feature = "compact-store")]
                        default_layered,
                        edit.candidate(),
                        #[cfg(feature = "triple-store")]
                        rdf_store,
                        #[cfg(feature = "triple-store")]
                        rdf_projections,
                        records,
                    );
                    applied?;
                    edit.finish().install().finish();
                }
                WalRecord::CatalogBatchV3 {
                    version,
                    epoch,
                    catalog_state,
                    created_graphs,
                    dropped_graphs,
                    created_graph_incarnations,
                    dropped_graph_incarnations: _,
                } => {
                    if *version != 2 {
                        return Err(Error::Serialization(format!(
                            "unsupported CatalogBatchV3 payload version {version}"
                        )));
                    }

                    let mut lifecycle_names = std::collections::HashSet::with_capacity(
                        created_graphs.len() + dropped_graphs.len(),
                    );
                    for name in created_graphs {
                        if name.components().is_empty() || !lifecycle_names.insert(name) {
                            return Err(Error::Serialization(format!(
                                "invalid or duplicate created graph {name:?} in CatalogBatchV3"
                            )));
                        }
                    }
                    for name in dropped_graphs {
                        if name.components().is_empty() || !lifecycle_names.insert(name) {
                            return Err(Error::Serialization(format!(
                                "invalid, duplicate, or conflicting dropped graph {name:?} in CatalogBatchV3"
                            )));
                        }
                    }

                    let candidate = Catalog::decode_wal_state_v1(catalog_state)
                        .map_err(Error::Serialization)?;
                    let mut workspace = crate::catalog::CatalogWorkspace::replacement(candidate);
                    // Retain every existing affected parent, not just root:
                    // nested topology edits must also roll back on failure.
                    let mut graphs_before = Vec::new();
                    for path in dropped_graphs.iter().chain(created_graphs) {
                        let parent = path
                            .parent()
                            .map_err(|error| Error::Serialization(error.to_string()))?
                            .ok_or_else(|| {
                                Error::Serialization("root graph lifecycle is invalid".into())
                            })?;
                        if let Ok(parent) = Self::resolve_wal_graph(store, &parent)
                            && !graphs_before
                                .iter()
                                .any(|(known, _)| Arc::ptr_eq(known, &parent))
                        {
                            let children = parent.named_graph_entries();
                            graphs_before.push((parent, children));
                        }
                    }
                    let ready = catalog
                        .prepare_wal_replacement(&mut workspace)
                        .map_err(|error| Error::Serialization(error.to_string()))?;
                    let applied = (|| -> Result<()> {
                        for path in dropped_graphs {
                            let parent = path
                                .parent()
                                .map_err(|error| Error::Serialization(error.to_string()))?
                                .ok_or_else(|| {
                                    Error::Serialization("root graph lifecycle is invalid".into())
                                })?;
                            let parent = Self::resolve_wal_graph(store, &parent)?;
                            let name = path.components().last().ok_or_else(|| {
                                Error::Serialization("root graph lifecycle is invalid".into())
                            })?;
                            parent.drop_graph(name);
                        }
                        for (path, incarnation) in
                            created_graphs.iter().zip(created_graph_incarnations)
                        {
                            let parent = path
                                .parent()
                                .map_err(|error| Error::Serialization(error.to_string()))?
                                .ok_or_else(|| {
                                    Error::Serialization("root graph lifecycle is invalid".into())
                                })?;
                            let parent = Self::resolve_wal_graph(store, &parent)?;
                            let name = path.components().last().ok_or_else(|| {
                                Error::Serialization("root graph lifecycle is invalid".into())
                            })?;
                            let created =
                                Arc::new(parent.new_replayed_graph_candidate(*incarnation)?);
                            if !parent.install_graph_if_absent(name, created) {
                                return Err(Error::Serialization(
                                    "validated catalog graph could not be installed".into(),
                                ));
                            }
                        }

                        store.sync_epoch(*epoch);
                        for (_, graph) in
                            grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store))
                                .capture_graphs()?
                        {
                            graph.sync_epoch(*epoch);
                        }
                        #[cfg(feature = "triple-store")]
                        rdf_store.try_set_commit_epoch(*epoch).map_err(|error| {
                            Error::Storage(
                                grafeo_common::utils::error::StorageError::InvalidWalEntry(
                                    error.to_string(),
                                ),
                            )
                        })?;
                        Ok(())
                    })();
                    if let Err(error) = applied {
                        for (parent, children) in graphs_before.into_iter().rev() {
                            parent.install_named_graphs(children);
                        }
                        return Err(error);
                    }
                    ready.install().finish();
                }

                // --- Schema DDL replay (always on root catalog) ---
                WalRecord::CreateNodeType {
                    name,
                    properties,
                    constraints,
                } => {
                    let def = NodeTypeDefinition {
                        name: name.clone(),
                        properties: properties
                            .iter()
                            .map(|(n, t, nullable)| TypedProperty {
                                name: n.clone(),
                                data_type: PropertyDataType::from_type_name(t),
                                nullable: *nullable,
                                default_value: None,
                            })
                            .collect(),
                        constraints: constraints
                            .iter()
                            .map(|(kind, props)| match kind.as_str() {
                                "unique" => TypeConstraint::Unique(props.clone()),
                                "primary_key" => TypeConstraint::PrimaryKey(props.clone()),
                                "not_null" if !props.is_empty() => {
                                    TypeConstraint::NotNull(props[0].clone())
                                }
                                _ => TypeConstraint::Unique(props.clone()),
                            })
                            .collect(),
                        parent_types: Vec::new(),
                    };
                    // `register_or_replace` (not `register`): a `CREATE OR REPLACE
                    // NODE TYPE` emits a second CreateNodeType record without a
                    // preceding drop, so a plain `register` would hit
                    // TypeAlreadyExists and keep the OLD definition after replay.
                    catalog.register_or_replace_node_type(def);
                }
                WalRecord::DropNodeType { name } => {
                    let _ = catalog.drop_node_type(name);
                }
                WalRecord::CreateEdgeType {
                    name,
                    properties,
                    constraints,
                } => {
                    let def = EdgeTypeDefinition {
                        name: name.clone(),
                        properties: properties
                            .iter()
                            .map(|(n, t, nullable)| TypedProperty {
                                name: n.clone(),
                                data_type: PropertyDataType::from_type_name(t),
                                nullable: *nullable,
                                default_value: None,
                            })
                            .collect(),
                        constraints: constraints
                            .iter()
                            .map(|(kind, props)| match kind.as_str() {
                                "unique" => TypeConstraint::Unique(props.clone()),
                                "primary_key" => TypeConstraint::PrimaryKey(props.clone()),
                                "not_null" if !props.is_empty() => {
                                    TypeConstraint::NotNull(props[0].clone())
                                }
                                _ => TypeConstraint::Unique(props.clone()),
                            })
                            .collect(),
                        source_node_types: Vec::new(),
                        target_node_types: Vec::new(),
                    };
                    // See CreateNodeType above: use `or_replace` so a CREATE OR
                    // REPLACE EDGE TYPE survives WAL replay.
                    catalog.register_or_replace_edge_type_def(def);
                }
                WalRecord::DropEdgeType { name } => {
                    let _ = catalog.drop_edge_type_def(name);
                }
                WalRecord::CreateConstraint {
                    name,
                    label,
                    properties,
                    kind,
                } => {
                    let kind = NamedConstraintKind::from_wal_str(kind).ok_or_else(|| {
                        Error::Serialization(format!(
                            "constraint '{name}' has unknown kind '{kind}'"
                        ))
                    })?;
                    let definition = NamedConstraintDefinition {
                        name: name.clone(),
                        label: label.clone(),
                        properties: properties.clone(),
                        kind,
                    };
                    catalog
                        .restore_named_constraint_from_wal(definition)
                        .map_err(|error| Error::Serialization(error.to_string()))?;
                }
                WalRecord::DropConstraint { name } => match catalog.drop_named_constraint(name) {
                    Ok(_) => {}
                    Err(crate::catalog::CatalogError::ConstraintNotFound(_))
                        if catalog.has_legacy_constraint_owners() =>
                    {
                        return Err(Error::Serialization(format!(
                            "cannot safely replay DROP CONSTRAINT '{name}': the checkpoint contains legacy enforcement owners but not the original constraint-name mapping"
                        )));
                    }
                    Err(crate::catalog::CatalogError::ConstraintNotFound(_)) => {}
                    Err(error) => return Err(Error::Serialization(error.to_string())),
                },
                WalRecord::CreateGraphType {
                    name,
                    node_types,
                    edge_types,
                    open,
                } => {
                    use crate::catalog::GraphTypeDefinition;
                    let def = GraphTypeDefinition {
                        name: name.clone(),
                        allowed_node_types: node_types.clone(),
                        allowed_edge_types: edge_types.clone(),
                        open: *open,
                    };
                    catalog.register_or_replace_graph_type(def);
                }
                WalRecord::DropGraphType { name } => {
                    let _ = catalog.drop_graph_type(name);
                }
                WalRecord::CreateSchema { name } => {
                    let _ = catalog.register_schema_namespace(name.clone());
                }
                WalRecord::DropSchema { name } => {
                    let _ = catalog.drop_schema_namespace(name);
                }

                WalRecord::AlterNodeType { name, alterations } => {
                    for (action, prop_name, type_name, nullable) in alterations {
                        match action.as_str() {
                            "add" => {
                                let prop = TypedProperty {
                                    name: prop_name.clone(),
                                    data_type: PropertyDataType::from_type_name(type_name),
                                    nullable: *nullable,
                                    default_value: None,
                                };
                                let _ = catalog.alter_node_type_add_property(name, prop);
                            }
                            "drop" => {
                                let _ = catalog.alter_node_type_drop_property(name, prop_name);
                            }
                            _ => {}
                        }
                    }
                }
                WalRecord::AlterEdgeType { name, alterations } => {
                    for (action, prop_name, type_name, nullable) in alterations {
                        match action.as_str() {
                            "add" => {
                                let prop = TypedProperty {
                                    name: prop_name.clone(),
                                    data_type: PropertyDataType::from_type_name(type_name),
                                    nullable: *nullable,
                                    default_value: None,
                                };
                                let _ = catalog.alter_edge_type_add_property(name, prop);
                            }
                            "drop" => {
                                let _ = catalog.alter_edge_type_drop_property(name, prop_name);
                            }
                            _ => {}
                        }
                    }
                }
                WalRecord::AlterGraphType { name, alterations } => {
                    for (action, type_name) in alterations {
                        let current = catalog.get_graph_type_def(name).ok_or_else(|| {
                            Error::Serialization(format!(
                                "cannot replay ALTER GRAPH TYPE for missing type '{name}'"
                            ))
                        })?;
                        match action.as_str() {
                            "add_node_type" | "add_node" => {
                                if !current.allowed_node_types.contains(type_name) {
                                    catalog
                                        .alter_graph_type_add_node_type(name, type_name.clone())
                                        .map_err(|error| Error::Serialization(error.to_string()))?;
                                }
                            }
                            "drop_node_type" | "drop_node" => {
                                if current.allowed_node_types.contains(type_name) {
                                    catalog
                                        .alter_graph_type_drop_node_type(name, type_name)
                                        .map_err(|error| Error::Serialization(error.to_string()))?;
                                }
                            }
                            "add_edge_type" | "add_edge" => {
                                if !current.allowed_edge_types.contains(type_name) {
                                    catalog
                                        .alter_graph_type_add_edge_type(name, type_name.clone())
                                        .map_err(|error| Error::Serialization(error.to_string()))?;
                                }
                            }
                            "drop_edge_type" | "drop_edge" => {
                                if current.allowed_edge_types.contains(type_name) {
                                    catalog
                                        .alter_graph_type_drop_edge_type(name, type_name)
                                        .map_err(|error| Error::Serialization(error.to_string()))?;
                                }
                            }
                            unknown => {
                                return Err(Error::Serialization(format!(
                                    "unknown ALTER GRAPH TYPE WAL action '{unknown}'"
                                )));
                            }
                        }
                    }
                }

                WalRecord::CreateProcedure {
                    name,
                    params,
                    returns,
                    body,
                } => {
                    use crate::catalog::ProcedureDefinition;
                    let def = ProcedureDefinition {
                        name: name.clone(),
                        params: params.clone(),
                        returns: returns.clone(),
                        body: body.clone(),
                    };
                    catalog
                        .replace_procedure(def)
                        .map_err(|error| Error::Serialization(error.to_string()))?;
                }
                WalRecord::DropProcedure { name } => {
                    let _ = catalog.drop_procedure(name);
                }

                // --- RDF triple replay ---
                #[cfg(feature = "triple-store")]
                WalRecord::InsertRdfQuadV3 { .. }
                | WalRecord::DeleteRdfQuadV3 { .. }
                | WalRecord::DropNamedRdfGraphV2 { .. }
                | WalRecord::CreateNamedRdfGraphV2 { .. } => {}
                #[cfg(not(feature = "triple-store"))]
                WalRecord::InsertRdfQuadV3 { .. }
                | WalRecord::DeleteRdfQuadV3 { .. }
                | WalRecord::DropNamedRdfGraphV2 { .. }
                | WalRecord::CreateNamedRdfGraphV2 { .. } => {}

                WalRecord::TransactionCommit { transaction_id } => {
                    // Legacy commit markers acquire their exact epoch from the
                    // following EpochAdvance when present. A marker without one
                    // predates crash-stable epoch recording; retain the old
                    // monotonic fallback for that compatibility case.
                    if let Some(epoch) = commit_epochs.get(transaction_id).copied() {
                        store.sync_epoch(epoch);
                    } else {
                        store.new_epoch();
                    }
                }
                WalRecord::Committed { epoch, .. } | WalRecord::CommittedWithCdc { epoch, .. } => {
                    // Idempotent for duplicate markers (including old close-time
                    // markers) and preserves intentional gaps after a failed
                    // durable commit reservation.
                    store.sync_epoch(*epoch);
                    #[cfg(feature = "triple-store")]
                    rdf_store.try_set_commit_epoch(*epoch).map_err(|error| {
                        Error::Storage(grafeo_common::utils::error::StorageError::InvalidWalEntry(
                            error.to_string(),
                        ))
                    })?;
                    #[cfg(not(feature = "triple-store"))]
                    let _ = epoch;
                }
                WalRecord::TransactionAbort { .. }
                | WalRecord::TransactionSavepoint { .. }
                | WalRecord::TransactionRollbackToSavepoint { .. }
                | WalRecord::Checkpoint { .. } => {
                    // Transaction control records don't need replay action
                    // (recovery already filtered to only committed transactions)
                }
                WalRecord::EpochAdvance { epoch } => {
                    store.sync_epoch(*epoch);
                    #[cfg(feature = "triple-store")]
                    rdf_store.try_set_commit_epoch(*epoch).map_err(|error| {
                        Error::Storage(grafeo_common::utils::error::StorageError::InvalidWalEntry(
                            error.to_string(),
                        ))
                    })?;
                    #[cfg(not(feature = "triple-store"))]
                    let _ = epoch;
                }
                WalRecord::CdcBatch { .. }
                | WalRecord::CdcRetention { .. }
                | WalRecord::GraphModelMeta { .. }
                | WalRecord::StoreIdentityMeta { .. }
                | WalRecord::RdfGraphIncarnationHighWaterMeta { .. } => {}
            }
        }
        owner_replay.validate_live(store, &default_target, catalog)?;
        Ok(())
    }

    // =========================================================================
    // Single-file format helpers
    // =========================================================================

    /// Returns `true` if the given path should use single-file format.
    #[cfg(feature = "grafeo-file")]
    fn should_use_single_file(
        path: &std::path::Path,
        configured: crate::config::StorageFormat,
    ) -> bool {
        use crate::config::StorageFormat;
        match configured {
            StorageFormat::SingleFile => true,
            StorageFormat::WalDirectory => false,
            StorageFormat::Auto => {
                // Existing file: check magic bytes
                if path.is_file() {
                    if let Ok(mut f) = std::fs::File::open(path) {
                        use std::io::Read;
                        let mut magic = [0u8; 4];
                        if f.read_exact(&mut magic).is_ok() && magic == grafeo_storage::file::MAGIC
                        {
                            return true;
                        }
                    }
                    return false;
                }
                // Existing directory: legacy format
                if path.is_dir() {
                    return false;
                }
                // New path: check extension
                path.extension().is_some_and(|ext| ext == "grafeo")
            }
        }
    }

    /// Applies snapshot data (from a `.grafeo` file) to restore the store and catalog.
    ///
    /// Supports both v1 (monolithic blob) and v2 (section-based) formats.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn apply_snapshot_data(
        #[cfg(feature = "cdc")] cdc_log: &crate::cdc::CdcLog,
        store: &Arc<LpgStore>,
        catalog: &Arc<crate::catalog::Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: &Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
        data: &[u8],
    ) -> Result<(u8, WorldIdentityMetadataV1)> {
        // v1 blob format: pass through to legacy loader
        persistence::load_snapshot_into_store(
            #[cfg(feature = "cdc")]
            cdc_log,
            store,
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_store,
            #[cfg(feature = "triple-store")]
            rdf_projections,
            data,
        )
    }

    /// Builds the recovered compact view before sidecar WAL replay.
    ///
    /// Default-graph WAL records must see the compact base while they mutate
    /// the loaded overlay: deletes and promotions are layered operations, and
    /// an exact edge may reference endpoints that exist only in the base. The
    /// returned instance is therefore the one replay targets and later installs
    /// into the database; it must never be reconstructed after replay.
    #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "compact-store"))]
    fn prepare_layered_after_load(
        overlay_store: &Arc<LpgStore>,
        compact_base: Option<Arc<grafeo_core::graph::compact::CompactStore>>,
        deletion_log: Option<(
            Vec<(grafeo_common::types::NodeId, grafeo_common::types::EpochId)>,
            Vec<(grafeo_common::types::EdgeId, grafeo_common::types::EpochId)>,
        )>,
    ) -> Result<Option<Arc<grafeo_core::graph::compact::layered::LayeredStore>>> {
        use grafeo_core::graph::compact::layered::LayeredStore;

        let Some(compact_base) = compact_base else {
            if deletion_log.is_some() {
                return Err(Error::Serialization(
                    "OverlayDeletions section requires a CompactStore section".to_string(),
                ));
            }
            return Ok(None);
        };

        let layered = Arc::new(
            LayeredStore::with_overlay(compact_base, Arc::clone(overlay_store)).map_err(
                |error| {
                    Error::Serialization(format!(
                        "cannot reconstruct compact recovery layer: {error}"
                    ))
                },
            )?,
        );

        if let Some((nodes, edges)) = deletion_log {
            layered.seed_deleted_from_base_at_epochs(nodes, edges);
        }

        Ok(Some(layered))
    }

    /// Publishes the exact compact view used during WAL recovery.
    ///
    /// This is deliberately installation-only: reconstructing a new layered
    /// store here would discard replayed tombstones, promotions, and routing
    /// state, and could also violate the overlay's one-generation binding.
    #[cfg(all(feature = "grafeo-file", feature = "lpg", feature = "compact-store"))]
    fn install_layered_after_load(
        &mut self,
        layered: Arc<grafeo_core::graph::compact::layered::LayeredStore>,
    ) -> Result<()> {
        let overlay_store = self.store.as_ref().ok_or_else(|| {
            Error::Internal("install_layered_after_load: no LpgStore".to_string())
        })?;
        let recovered_overlay = layered.overlay_store();
        if !Arc::ptr_eq(&recovered_overlay, overlay_store) {
            return Err(Error::Internal(
                "install_layered_after_load: recovered layer belongs to a different LPG overlay"
                    .to_string(),
            ));
        }

        // Replay has already applied every durable epoch to this overlay. The
        // transaction manager may additionally carry an RDF-only frontier.
        let current_epoch = self.transaction_manager.current_epoch();
        self.transaction_manager
            .with_write_authority(|| layered.overlay_store().sync_epoch(current_epoch));

        self.external_read_store = Some(Arc::clone(&layered) as Arc<dyn GraphStoreSearch>);
        self.external_write_store = Some(Arc::clone(&layered) as Arc<dyn GraphStoreMut>);

        // Install the tier wrapper + consumers (mirror of compact()'s flow).
        #[cfg(feature = "mmap")]
        {
            let tiered = Arc::new(compact_tiered::CompactStoreTiered::new_in_memory(
                layered.base_store_arc(),
            ));
            let spill_path = self.buffer_manager.config().spill_path.clone();
            let consumer = Arc::new(section_consumer::CompactStoreConsumer::new(
                &tiered,
                &layered,
                &self.transaction_manager,
                spill_path,
            ));
            self.buffer_manager.register_consumer(consumer);
            self.compact_tiered = Some(tiered);
        }

        self.layered_store = Some(layered);

        Ok(())
    }

    /// RDF-only open: load the RdfStore section without an LPG store.
    #[cfg(all(
        feature = "grafeo-file",
        feature = "triple-store",
        not(feature = "lpg")
    ))]
    fn load_rdf_sections(
        #[cfg(feature = "cdc")] cdc_log: &crate::cdc::CdcLog,
        fm: &GrafeoFileManager,
        catalog: &Arc<crate::catalog::Catalog>,
        world_identity: &Arc<RwLock<WorldIdentityMetadataV1>>,
        rdf_store: &Arc<RdfStore>,
    ) -> Result<LoadedRdfSectionState> {
        use grafeo_common::storage::{Section, SectionType};
        use world_metadata::VerifiedWorldMetadata;

        let dir = fm.read_section_directory()?.ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "expected v2 section directory but found none".to_string(),
            )
        })?;
        let (image, metadata, _graph_model) =
            world_metadata::read_verified_container_image(fm, &dir)?;
        #[cfg(feature = "cdc")]
        cdc_checkpoint::restore_section(&image, metadata.as_ref(), cdc_log)?;
        let recovery_image_is_sealed = metadata
            .as_ref()
            .is_some_and(VerifiedWorldMetadata::recovery_image_is_sealed);
        let recovery_coordinates_are_sealed = metadata
            .as_ref()
            .is_some_and(VerifiedWorldMetadata::recovery_coordinates_are_sealed);
        let rdf_data = world_metadata::find_section(&image, SectionType::RdfStore);
        #[cfg(feature = "ring-index")]
        let ring_data = world_metadata::find_section(&image, SectionType::RdfRing);

        if let Some(section_data) = world_metadata::find_section(&image, SectionType::Catalog) {
            let mut section = catalog_state_section::CatalogStateSection::new(Arc::clone(catalog));
            section.deserialize(section_data.bytes())?;
        }
        if let Some(section_data) = rdf_data {
            let mut section = grafeo_core::graph::rdf::RdfStoreSection::new(Arc::clone(rdf_store));
            section.deserialize(section_data.bytes())?;
        }

        #[cfg(feature = "ring-index")]
        if let Some(section_data) = ring_data
            && recovery_image_is_sealed
        {
            let mut section = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(rdf_store));
            section.deserialize(section_data.bytes())?;
        }

        let identity =
            WorldIdentityMetadataV1::new(rdf_store.store_id(), rdf_store.history_completeness())
                .map_err(|error| {
                    Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                        format!("invalid decoded RDF world identity: {error}"),
                    ))
                })?;
        if let Some(metadata) = metadata.as_ref() {
            world_metadata::verify_loaded_state(metadata, identity.clone(), catalog)?;
        }
        let metadata_epoch = metadata.as_ref().map(|metadata| metadata.cut().epoch());
        let decoded_epoch = rdf_data
            .is_some_and(|section| section.version() >= 3)
            .then(|| rdf_store.commit_epoch());
        if let (Some(authenticated), Some(decoded)) = (metadata_epoch, decoded_epoch)
            && authenticated != decoded
        {
            return Err(Error::Storage(
                grafeo_common::utils::error::StorageError::Corruption(format!(
                    "decoded RDF snapshot epoch {} disagrees with authenticated world epoch {}",
                    decoded.as_u64(),
                    authenticated.as_u64()
                )),
            ));
        }
        let authenticated_replay_epoch = metadata_epoch.or(decoded_epoch);
        *world_identity.write() = identity;
        #[cfg(feature = "cdc")]
        cdc_log.validate_checkpoint_native(crate::cdc::checkpoint::NativeCut {
            model: 1,
            lpg_next: 0,
            lpg_owners: &[],
            rdf_next: rdf_store.next_graph_incarnation().as_u64(),
        })?;
        Ok(LoadedRdfSectionState {
            #[cfg(feature = "ring-index")]
            ring_declared: ring_data.is_some(),
            #[cfg(not(feature = "ring-index"))]
            ring_declared: false,
            recovery_image_is_sealed,
            recovery_coordinates_are_sealed,
            authenticated_replay_epoch,
            #[cfg(feature = "wal")]
            authenticated_identity: AuthenticatedContainerIdentity::from_decoded_image(
                metadata.as_ref(),
            )?,
        })
    }

    /// Loads from a section-based `.grafeo` file (v2 format).
    ///
    /// Reads the section directory, then deserializes each section independently.
    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    fn load_from_sections(
        #[cfg(feature = "cdc")] cdc_log: &crate::cdc::CdcLog,
        fm: &GrafeoFileManager,
        store: &Arc<LpgStore>,
        catalog: &Arc<crate::catalog::Catalog>,
        world_identity: &Arc<RwLock<WorldIdentityMetadataV1>>,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: &Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
    ) -> Result<LoadedSectionState> {
        use grafeo_common::storage::{Section, SectionType};
        use world_metadata::VerifiedWorldMetadata;

        let dir = fm.read_section_directory()?.ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "expected v2 section directory but found none".to_string(),
            )
        })?;

        // Authenticate the exact complete image before deserializing any
        // model. The constructor owns detached stores, so a later decoder
        // error discards the whole staged state without exposing a partial DB.
        let (image, metadata, graph_model) =
            world_metadata::read_verified_container_image(fm, &dir)?;
        #[cfg(feature = "cdc")]
        cdc_checkpoint::restore_section(&image, metadata.as_ref(), cdc_log)?;
        let recovery_image_is_sealed = metadata
            .as_ref()
            .is_some_and(VerifiedWorldMetadata::recovery_image_is_sealed);
        let recovery_coordinates_are_sealed = metadata
            .as_ref()
            .is_some_and(VerifiedWorldMetadata::recovery_coordinates_are_sealed);
        let catalog_data = world_metadata::find_section(&image, SectionType::Catalog);
        let lpg_data = world_metadata::find_section(&image, SectionType::LpgStore);
        #[cfg(feature = "triple-store")]
        let rdf_data = world_metadata::find_section(&image, SectionType::RdfStore);
        #[cfg(feature = "ring-index")]
        let ring_data = world_metadata::find_section(&image, SectionType::RdfRing);
        let vector_data = world_metadata::find_section(&image, SectionType::VectorStore);
        let text_data = world_metadata::find_section(&image, SectionType::TextIndex);
        let compact_data = world_metadata::find_section(&image, SectionType::CompactStore);
        let overlay_deletions_data =
            world_metadata::find_section(&image, SectionType::OverlayDeletions);
        let has_compact_store = compact_data.is_some();
        let has_overlay_deletions = overlay_deletions_data.is_some();
        if has_overlay_deletions && !has_compact_store {
            return Err(Error::Serialization(
                "OverlayDeletions section requires a CompactStore section".to_string(),
            ));
        }

        #[cfg(not(feature = "vector-index"))]
        if vector_data.is_some() {
            return Err(Error::Serialization(
                "Container has a Vector Store section but vector-index support is disabled"
                    .to_string(),
            ));
        }
        #[cfg(not(feature = "text-index"))]
        if text_data.is_some() {
            return Err(Error::Serialization(
                "Container has a Text Index section but text-index support is disabled".to_string(),
            ));
        }
        #[cfg(feature = "text-index")]
        validate_exact_index_recovery_section(
            recovery_image_is_sealed,
            text_data.map(world_metadata::EncodedSection::version),
            "Text Index",
            5,
        )?;
        #[cfg(feature = "vector-index")]
        validate_exact_index_recovery_section(
            recovery_image_is_sealed,
            vector_data.map(world_metadata::EncodedSection::version),
            "Vector Store",
            4,
        )?;

        // Catalog v7 and the graph-qualified exact index sections form one
        // admission contract. Classify and reject incompatible pairs before
        // even the detached recovery store is populated; a corrupt payload
        // must never select a weaker descriptor-rebuild path.
        let lpg_catalog_version = if matches!(
            graph_model,
            grafeo_common::types::GraphModelTag::Lpg | grafeo_common::types::GraphModelTag::Both
        ) {
            catalog_data
                .map(|data| catalog_wire::classify_catalog_payload(data.bytes()))
                .transpose()?
        } else {
            None
        };

        let load_outcome = SectionLoadOutcome::for_image(
            lpg_catalog_version,
            recovery_image_is_sealed,
            recovery_coordinates_are_sealed,
            #[cfg(all(feature = "triple-store", feature = "ring-index"))]
            ring_data.is_some(),
        )?;
        #[cfg(feature = "wal")]
        let mut load_outcome = load_outcome;

        // When the file has a CompactStore section, this LpgStore data is the
        // overlay; this loader later wires it into a LayeredStore from the
        // same authenticated section image. Load data before creating its
        // catalog-owned private index targets. Each exact auxiliary payload is
        // decoded once below; no database handle escapes unless all succeed.
        if let Some(data) = lpg_data {
            let mut section = grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store));
            section.deserialize(data.bytes())?;
            if !store.graph_incarnation_id().is_default_graph() {
                return Err(Error::Serialization(
                    "container default LPG graph has a named incarnation".into(),
                ));
            }
        }

        // Bind the cold predecessor before Catalog7 rebuilds property
        // indexes.  Otherwise its image contains overlay rows only and a
        // reopened indexed query cannot discover cold-only identities.  This
        // layer is still detached until the database is fully constructed.
        #[cfg(feature = "compact-store")]
        let (layered, property_index_deletions) = {
            let compact_base = match compact_data {
                Some(data) => {
                    let mut section =
                        grafeo_core::graph::compact::section::CompactStoreSection::empty();
                    section.deserialize(data.bytes())?;
                    Some(section.store().ok_or_else(|| {
                        Error::Serialization(
                            "decoded CompactStore section did not produce a store".to_string(),
                        )
                    })?)
                }
                None => None,
            };
            let deletion_log = match overlay_deletions_data {
                Some(data) => {
                    let mut section = grafeo_core::graph::compact::deletions_section::
                        OverlayDeletionsSection::empty();
                    section.deserialize(data.bytes())?;
                    Some(section.take())
                }
                None => None,
            };
            let property_index_deletions = deletion_log
                .as_ref()
                .map_or_else(Vec::new, |(nodes, _)| nodes.clone());
            let layered = Self::prepare_layered_after_load(store, compact_base, deletion_log)?;
            (layered, property_index_deletions)
        };

        let index_graphs = if matches!(
            graph_model,
            grafeo_common::types::GraphModelTag::Lpg | grafeo_common::types::GraphModelTag::Both
        ) {
            Some(index_sections::LpgIndexGraphCut::capture(Arc::clone(
                store,
            ))?)
        } else {
            None
        };

        if let Some(data) = catalog_data {
            if graph_model == grafeo_common::types::GraphModelTag::Rdf {
                let mut section =
                    catalog_state_section::CatalogStateSection::new(Arc::clone(catalog));
                section.deserialize(data.bytes())?;
            } else {
                let tm = Arc::new(crate::transaction::TransactionManager::new());
                let index_graphs = index_graphs.as_ref().ok_or_else(|| {
                    Error::Serialization("LPG catalog has no captured graph topology".into())
                })?;
                let expected_epoch = metadata
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Serialization(
                            "Catalog v7 requires verified WorldMetadata before recovery".into(),
                        )
                    })?
                    .cut()
                    .epoch()
                    .as_u64();
                let mut section = catalog_section::CatalogSection::new_with_graphs(
                    Arc::clone(catalog),
                    index_graphs.graphs(),
                    move || tm.current_epoch().as_u64(),
                )?
                .with_expected_graph_exact_epoch(expected_epoch);
                #[cfg(feature = "compact-store")]
                {
                    section =
                        section.with_property_index_deletions(property_index_deletions.clone());
                }
                #[cfg(feature = "vector-index")]
                {
                    section = section.with_vector_payloads(vector_data.map(|data| data.bytes()))?;
                }
                section.deserialize(data.bytes())?;
            }
        }

        #[cfg(feature = "triple-store")]
        if let Some(data) = rdf_data {
            let mut section = grafeo_core::graph::rdf::RdfStoreSection::with_projections(
                Arc::clone(rdf_store),
                Arc::clone(rdf_projections),
            );
            section.deserialize(data.bytes())?;
        }

        #[cfg(feature = "ring-index")]
        if let Some(data) = ring_data
            && recovery_image_is_sealed
        {
            let mut section = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(rdf_store));
            section.deserialize(data.bytes())?;
        }

        #[cfg(feature = "vector-index")]
        if recovery_image_is_sealed {
            let indexes = match index_graphs.as_ref() {
                Some(index_graphs) => index_graphs.vector_views()?,
                None if vector_data.is_some() => {
                    return Err(Error::Serialization(
                        "Vector Store section is present outside an LPG graph model".to_string(),
                    ));
                }
                None => Vec::new(),
            };
            match vector_data {
                Some(data) => {
                    // Always invoke the decoder, including with an empty target
                    // set: an orphan exact section is corruption, not optional
                    // metadata to ignore.
                    let mut section = grafeo_core::index::vector::VectorStoreSection::
                        for_unpublished_recovery_views(indexes);
                    section.deserialize(data.bytes())?;
                }
                None if !indexes.is_empty() => {
                    return Err(Error::Serialization(
                        "Catalog declares vector indexes but the authoritative Vector Store section is missing"
                            .to_string(),
                    ));
                }
                None => {}
            }
        }

        #[cfg(feature = "text-index")]
        if recovery_image_is_sealed {
            let indexes = match index_graphs.as_ref() {
                Some(index_graphs) => index_graphs.text_views()?,
                None if text_data.is_some() => {
                    return Err(Error::Serialization(
                        "Text Index section is present outside an LPG graph model".to_string(),
                    ));
                }
                None => Vec::new(),
            };
            match text_data {
                Some(data) => {
                    // The exact decoder enforces a one-to-one scoped key set
                    // and acquires every destination proof before the first
                    // move. These detached targets remain unpublished until
                    // the complete database constructor succeeds.
                    let mut section =
                        grafeo_core::index::text::TextIndexSection::for_unpublished_recovery_views(
                            indexes,
                        );
                    section.deserialize(data.bytes())?;
                }
                None if !indexes.is_empty() => {
                    return Err(Error::Serialization(
                        "Catalog declares text indexes but the authoritative Text Index section is missing"
                            .to_string(),
                    ));
                }
                None => {}
            }
        }

        #[cfg(feature = "triple-store")]
        let decoded_identity = if matches!(
            graph_model,
            grafeo_common::types::GraphModelTag::Rdf | grafeo_common::types::GraphModelTag::Both
        ) {
            WorldIdentityMetadataV1::new(rdf_store.store_id(), rdf_store.history_completeness())
                .map_err(|error| {
                    Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                        format!("invalid decoded RDF world identity: {error}"),
                    ))
                })?
        } else {
            metadata
                .as_ref()
                .map(|metadata| {
                    WorldIdentityMetadataV1::new(
                        metadata.cut().store_id(),
                        metadata.cut().descriptor().history(),
                    )
                    .map_err(|error| {
                        Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                            format!("invalid container world identity: {error}"),
                        ))
                    })
                })
                .transpose()?
                .unwrap_or_else(|| world_identity.read().clone())
        };
        #[cfg(not(feature = "triple-store"))]
        let decoded_identity = metadata
            .as_ref()
            .map(|metadata| {
                WorldIdentityMetadataV1::new(
                    metadata.cut().store_id(),
                    metadata.cut().descriptor().history(),
                )
                .map_err(|error| {
                    Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                        format!("invalid container world identity: {error}"),
                    ))
                })
            })
            .transpose()?
            .unwrap_or_else(|| world_identity.read().clone());

        // A triple-enabled binary still constructs an unused RDF store for an
        // LPG-only database. Bind that empty provisional store to the LPG
        // world's recovered identity so later snapshot/export code cannot
        // diverge merely because RDF support was compiled in.
        #[cfg(feature = "triple-store")]
        if graph_model == grafeo_common::types::GraphModelTag::Lpg {
            rdf_store
                .adopt_recovery_identity(&decoded_identity)
                .map_err(|error| {
                    Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                        format!("bind unused RDF store to LPG world identity: {error}"),
                    ))
                })?;
        }

        if let Some(metadata) = metadata.as_ref() {
            world_metadata::verify_loaded_state(
                metadata,
                decoded_identity.clone(),
                catalog,
                #[cfg(feature = "triple-store")]
                rdf_projections,
            )?;
        }
        #[cfg(feature = "wal")]
        {
            load_outcome.authenticated_replay_epoch = authenticated_section_replay_epoch(
                metadata.as_ref().map(|metadata| metadata.cut().epoch()),
                graph_model,
                store,
                #[cfg(feature = "triple-store")]
                rdf_store,
                #[cfg(feature = "triple-store")]
                rdf_data.map(world_metadata::EncodedSection::version),
            )?;
        }
        *world_identity.write() = decoded_identity;

        #[cfg(feature = "cdc")]
        {
            let owners = grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store))
                .capture_graphs()?
                .into_iter()
                .map(|(path, graph)| (path, graph.graph_incarnation_id()))
                .collect::<Vec<_>>();
            #[cfg(feature = "triple-store")]
            let rdf_next = rdf_store.next_graph_incarnation().as_u64();
            #[cfg(not(feature = "triple-store"))]
            let rdf_next = 0;
            cdc_log.validate_checkpoint_native(crate::cdc::checkpoint::NativeCut {
                model: graph_model.as_u8(),
                lpg_next: store.next_graph_incarnation_id(),
                lpg_owners: &owners,
                rdf_next,
            })?;
        }
        Ok(LoadedSectionState {
            outcome: load_outcome,
            #[cfg(feature = "wal")]
            authenticated_identity: AuthenticatedContainerIdentity::from_decoded_image(
                metadata.as_ref(),
            )?,
            #[cfg(feature = "compact-store")]
            layered,
        })
    }

    // =========================================================================
    // Session & Configuration
    // =========================================================================

    /// Opens a new session for running queries.
    ///
    /// Sessions are cheap to create: spin up as many as you need. Each
    /// gets its own transaction context, so concurrent sessions won't
    /// block each other on reads.
    ///
    /// # Panics
    ///
    /// Panics if the database was configured with an external graph store and
    /// the internal arena allocator cannot be initialized (out of memory).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Run queries through the session
    /// let result = session.execute("MATCH (n) RETURN count(n)")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn session(&self) -> Session {
        self.create_session_inner(None)
    }

    /// Creates a session scoped to the given identity.
    ///
    /// The identity determines what operations the session is allowed to
    /// perform. A [`Role::ReadOnly`](crate::auth::Role::ReadOnly) identity
    /// creates a read-only session; a [`Role::ReadWrite`](crate::auth::Role::ReadWrite)
    /// identity allows data mutations but not schema DDL; a
    /// [`Role::Admin`](crate::auth::Role::Admin) identity has full access.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, auth::{Identity, Role}};
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let identity = Identity::new("app-service", [Role::ReadWrite]);
    /// let session = db.session_with_identity(identity);
    /// ```
    #[must_use]
    pub fn session_with_identity(&self, identity: crate::auth::Identity) -> Session {
        let force_read_only = !identity.can_write();
        self.create_session_inner_full(None, force_read_only, identity)
    }

    /// Creates a session scoped to a single role.
    ///
    /// Convenience shorthand for
    /// `session_with_identity(Identity::new("anonymous", [role]))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, auth::Role};
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let reader = db.session_with_role(Role::ReadOnly);
    /// ```
    #[must_use]
    pub fn session_with_role(&self, role: crate::auth::Role) -> Session {
        self.session_with_identity(crate::auth::Identity::new("anonymous", [role]))
    }

    /// Creates a session with an explicit CDC override.
    ///
    /// When `cdc_enabled` is `true`, mutations in this session are tracked
    /// regardless of the database default. When `false`, mutations are not
    /// tracked regardless of the database default.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    ///
    /// // Opt in to CDC for just this session
    /// let tracked = db.session_with_cdc(true);
    /// tracked.execute("INSERT (:Person {name: 'Alix'})")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn session_with_cdc(&self, cdc_enabled: bool) -> Session {
        self.create_session_inner(Some(cdc_enabled))
    }

    /// Shared session creation logic.
    ///
    /// `cdc_override` overrides the database-wide `cdc_enabled` default when
    /// `Some`. `None` falls back to the database default.
    #[allow(unused_variables)] // cdc_override unused when cdc feature is off
    fn create_session_inner(&self, cdc_override: Option<bool>) -> Session {
        self.create_session_inner_full(cdc_override, false, crate::auth::Identity::anonymous())
    }

    /// Shared session creation with all overrides.
    #[allow(unused_variables)]
    fn create_session_inner_full(
        &self,
        cdc_override: Option<bool>,
        force_read_only: bool,
        identity: crate::auth::Identity,
    ) -> Session {
        let session_cfg = || crate::session::SessionConfig {
            transaction_manager: Arc::clone(&self.transaction_manager),
            query_cache: Arc::clone(&self.query_cache),
            physical_cache: Arc::clone(&self.physical_cache),
            #[cfg(any(
                feature = "lpg",
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            catalog: Arc::clone(&self.catalog),
            adaptive_config: self.config.adaptive.clone(),
            #[cfg(any(
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            factorized_execution: self.config.factorized_execution,
            graph_model: self.config.graph_model,
            query_timeout: self.config.query_timeout,
            result_limits: self.config.result_limits,
            #[cfg(any(
                feature = "lpg",
                feature = "gql",
                feature = "cypher",
                feature = "gremlin",
                feature = "graphql",
                feature = "sql-pgq"
            ))]
            max_property_size: self.config.max_property_size,
            buffer_manager: Some(Arc::clone(&self.buffer_manager)),
            #[cfg(feature = "spill")]
            spill_root: Arc::clone(&self.spill_root),
            #[cfg(any(feature = "spill", feature = "cdc"))]
            world_identity: Arc::clone(&self.world_identity),
            #[cfg(feature = "lpg")]
            commit_counter: Arc::clone(&self.commit_counter),
            durability_poisoned: Arc::clone(&self.durability_poisoned),
            database_open: Arc::clone(&self.is_open),
            active_sessions: Arc::clone(&self.active_sessions),
            #[cfg(feature = "lpg")]
            gc_interval: self.config.gc_interval,
            read_only: self.read_only || force_read_only,
            identity: identity.clone(),
            #[cfg(feature = "lpg")]
            projections: Arc::clone(&self.projections),
        };

        // Capture one coherent store topology. Compaction takes the matching
        // publication write lock and rechecks `active_sessions` before swapping
        // any pointer, so a Session is either fully pre- or post-compaction.
        let _lifecycle = self.is_open.read();
        let _publication = self.transaction_manager.publication().read();

        let regular_session = || {
            if let Some(ref ext_read) = self.external_read_store {
                return Session::with_external_store(
                    Arc::clone(ext_read),
                    self.external_write_store.as_ref().map(Arc::clone),
                    session_cfg(),
                )
                .expect("arena allocation for external store session");
            }

            #[cfg(all(feature = "lpg", feature = "triple-store"))]
            {
                Session::with_rdf_store_and_adaptive(
                    Arc::clone(self.store_arc()),
                    Arc::clone(&self.rdf_store),
                    session_cfg(),
                )
            }
            #[cfg(all(feature = "lpg", not(feature = "triple-store")))]
            {
                Session::with_adaptive(Arc::clone(self.store_arc()), session_cfg())
            }
            #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
            {
                Session::with_rdf_only(Arc::clone(&self.rdf_store), session_cfg())
            }
            #[cfg(all(not(feature = "lpg"), not(feature = "triple-store")))]
            {
                Session::with_external_store(
                    self.graph_store(),
                    self.graph_store_mut(),
                    session_cfg(),
                )
                .expect("session creation for non-lpg build")
            }
        };

        // Layered store: use the overlay LpgStore as the session's internal
        // store so MVCC operations work, while retaining the database's
        // canonical RDF store for GraphModel::Both. The session then flows
        // through the common WAL/CDC/metrics/context setup below.
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        let mut session = if let Some(ref layered) = self.layered_store {
            let overlay = layered.overlay_store();
            let layered_arc = Arc::clone(layered);
            #[cfg(feature = "triple-store")]
            let mut session = Session::with_rdf_store_and_adaptive(
                overlay,
                Arc::clone(&self.rdf_store),
                session_cfg(),
            );
            #[cfg(not(feature = "triple-store"))]
            let mut session = Session::with_adaptive(overlay, session_cfg());
            // Override graph_store/graph_store_mut to use the LayeredStore
            // (which merges base + overlay), not just the overlay alone.
            session.override_stores(
                Arc::clone(&layered_arc) as Arc<dyn GraphStoreSearch>,
                Some(layered_arc as Arc<dyn GraphStoreMut>),
            );
            session
        } else {
            regular_session()
        };
        #[cfg(not(all(feature = "compact-store", feature = "lpg")))]
        let mut session = regular_session();

        #[cfg(all(feature = "wal", feature = "lpg"))]
        if let Some(ref wal) = self.wal {
            session.set_wal(Arc::clone(wal));
        }
        #[cfg(all(feature = "wal", not(feature = "lpg")))]
        if let Some(ref wal) = self.wal {
            session.set_wal_handle(Arc::clone(wal));
        }

        #[cfg(feature = "cdc")]
        {
            let should_enable = cdc_override.unwrap_or_else(|| self.cdc_active());
            session.set_cdc_log(Arc::clone(&self.cdc_log), should_enable);
        }

        #[cfg(feature = "metrics")]
        {
            if let Some(ref m) = self.metrics {
                session.set_metrics(Arc::clone(m));
                m.session_created
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                m.session_active
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // Copy the already-validated coordinate and metadata atomically. No
        // fallible selection or schema re-resolution belongs in this factory.
        session.inherit_graph_context(self.current_context.read().clone());

        // Suppress unused_mut when cdc/wal are disabled
        let _ = &mut session;

        self.active_sessions
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);

        session
    }

    /// Returns the current graph name, if any.
    ///
    /// This is the persistent graph context used by one-shot `execute()` calls.
    /// It is updated whenever `execute()` encounters `USE GRAPH`, `SESSION SET GRAPH`,
    /// or `SESSION RESET`.
    #[must_use]
    pub fn current_graph(&self) -> Option<String> {
        self.current_context.read().graph.clone()
    }

    /// Resolves one persistent graph selector against an already-canonical
    /// schema context while the caller holds the publication barrier.
    #[cfg(feature = "lpg")]
    fn resolve_persistent_graph_selector(
        &self,
        requested: &str,
        current_schema: Option<&str>,
    ) -> Result<(String, grafeo_common::types::GraphPath)> {
        use crate::session::{SCHEMA_DEFAULT_GRAPH, Session};

        let (context_name, storage_key) = if requested.eq_ignore_ascii_case("default") {
            (
                "default".to_string(),
                Session::storage_key_for_context(current_schema, Some("default")),
            )
        } else if let Some((prefix, suffix)) = requested.split_once('/') {
            if let Some(owner) = self
                .catalog
                .schema_names()
                .into_iter()
                .find(|registered| registered.eq_ignore_ascii_case(prefix))
            {
                let suffix = if suffix.eq_ignore_ascii_case(SCHEMA_DEFAULT_GRAPH) {
                    SCHEMA_DEFAULT_GRAPH
                } else {
                    suffix
                };
                let context_name = current_schema.map_or_else(
                    || format!("{owner}/{suffix}"),
                    |active| {
                        if active.eq_ignore_ascii_case(&owner) {
                            suffix.to_string()
                        } else {
                            format!("{owner}/{suffix}")
                        }
                    },
                );
                (context_name, Some(format!("{owner}/{suffix}")))
            } else {
                // An unregistered slash prefix is a parser-free absolute/root
                // graph name, including URI-shaped names. Preserve that
                // public capability even when a schema context is active.
                (requested.to_string(), Some(requested.to_string()))
            }
        } else {
            (
                requested.to_string(),
                Session::storage_key_for_context(current_schema, Some(requested)),
            )
        };

        if let (Some(store), Some(key)) = (self.store.as_ref(), storage_key.as_deref())
            && store.graph(key).is_none()
        {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("Graph '{requested}' does not exist"),
            )));
        }
        let path = match storage_key {
            None => grafeo_common::types::GraphPath::root(),
            Some(name) => grafeo_common::types::GraphPath::from_components(&[&name])
                .map_err(|error| Error::InvalidValue(error.to_string()))?,
        };
        Ok((context_name, path))
    }

    /// Sets the current graph context for subsequent one-shot `execute()` calls.
    ///
    /// This is equivalent to running `USE GRAPH <name>` but without creating a session.
    /// Pass `None` to reset to the default graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the named graph does not exist.
    pub fn set_current_graph(&self, name: Option<&str>) -> Result<()> {
        let _publication = crate::session::acquire_publication_read(&self.transaction_manager);
        #[cfg(feature = "lpg")]
        {
            let mut context = self.current_context.write();
            let (graph, path) = match name {
                Some(requested) => {
                    let (graph, path) = self
                        .resolve_persistent_graph_selector(requested, context.schema.as_deref())?;
                    (Some(graph), path)
                }
                None => (
                    None,
                    Session::context_graph_path(context.schema.as_deref(), None)?,
                ),
            };
            context.graph = graph;
            context.storage_key = path;
            context.native = false;
            Ok(())
        }
        #[cfg(not(feature = "lpg"))]
        {
            if let Some(name) = name
                && !name.eq_ignore_ascii_case("default")
            {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Unsupported,
                    format!(
                        "cannot select LPG graph '{name}': this build does not include LPG support"
                    ),
                )));
            }
            let mut context = self.current_context.write();
            let path = Session::context_graph_path(context.schema.as_deref(), name)?;
            context.graph = name.map(str::to_owned);
            context.storage_key = path;
            context.native = false;
            Ok(())
        }
    }

    /// Returns the current schema name, if any.
    ///
    /// This is the persistent schema context used by one-shot `execute()` calls.
    /// It is updated whenever `execute()` encounters `SESSION SET SCHEMA` or `SESSION RESET`.
    #[must_use]
    pub fn current_schema(&self) -> Option<String> {
        self.current_context.read().schema.clone()
    }

    /// Sets the current schema context for subsequent one-shot `execute()` calls.
    ///
    /// This is equivalent to running `SESSION SET SCHEMA <name>` but without creating
    /// a session. Pass `None` to clear the schema context.
    ///
    /// # Errors
    ///
    /// Returns an error if the named schema does not exist.
    pub fn set_current_schema(&self, name: Option<&str>) -> Result<()> {
        let _publication = crate::session::acquire_publication_read(&self.transaction_manager);
        let canonical = name
            .map(|name| {
                self.catalog
                    .schema_names()
                    .into_iter()
                    .find(|registered| registered.eq_ignore_ascii_case(name))
                    .ok_or_else(|| {
                        Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!("Schema '{name}' does not exist"),
                        ))
                    })
            })
            .transpose()?;
        let mut context = self.current_context.write();
        let path = Session::context_graph_path(canonical.as_deref(), context.graph.as_deref())?;
        context.schema = canonical;
        if !context.native {
            context.storage_key = path;
        }
        Ok(())
    }

    /// Returns the adaptive execution configuration.
    #[must_use]
    pub fn adaptive_config(&self) -> &crate::config::AdaptiveConfig {
        &self.config.adaptive
    }

    /// Returns `true` if this database was opened in read-only mode.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Returns `true` after abort/log/fsync failed closed (C1 poison).
    #[must_use]
    pub fn is_durability_poisoned(&self) -> bool {
        let shared = self
            .durability_poisoned
            .load(std::sync::atomic::Ordering::SeqCst);
        #[cfg(feature = "wal")]
        {
            shared || self.wal.as_ref().is_some_and(|wal| wal.is_poisoned())
        }
        #[cfg(not(feature = "wal"))]
        {
            shared
        }
    }

    /// Returns the current committed epoch of the database for either graph model.
    #[must_use]
    pub fn current_epoch(&self) -> grafeo_common::types::EpochId {
        let _publication = self.transaction_manager.publication().read();
        self.transaction_manager.current_epoch()
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the graph data model of this database.
    #[must_use]
    pub fn graph_model(&self) -> crate::config::GraphModel {
        self.config.graph_model
    }

    /// Returns the configured memory limit in bytes, if any.
    #[must_use]
    pub fn memory_limit(&self) -> Option<usize> {
        self.config.memory_limit
    }

    /// Returns a point-in-time snapshot of all metrics.
    ///
    /// If the `metrics` feature is disabled or the registry is not
    /// initialized, returns a default (all-zero) snapshot.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn metrics(&self) -> crate::metrics::MetricsSnapshot {
        let mut snapshot = self
            .metrics
            .as_ref()
            .map_or_else(crate::metrics::MetricsSnapshot::default, |m| m.snapshot());

        // Augment with cache stats from the query cache (not tracked in the registry)
        let cache_stats = self.query_cache.stats();
        snapshot.cache_hits = cache_stats.parsed_hits + cache_stats.optimized_hits;
        snapshot.cache_misses = cache_stats.parsed_misses + cache_stats.optimized_misses;
        snapshot.cache_size = cache_stats.parsed_size + cache_stats.optimized_size;
        snapshot.cache_invalidations = cache_stats.invalidations;

        snapshot
    }

    /// Returns all metrics in Prometheus text exposition format.
    ///
    /// The output is ready to serve from an HTTP `/metrics` endpoint.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn metrics_prometheus(&self) -> String {
        self.metrics
            .as_ref()
            .map_or_else(String::new, |m| m.to_prometheus())
    }

    /// Resets all metrics counters and histograms to zero.
    #[cfg(feature = "metrics")]
    pub fn reset_metrics(&self) {
        if let Some(ref m) = self.metrics {
            m.reset();
        }
        self.query_cache.reset_stats();
    }

    // === Named Graph Management ===

    /// Creates one literal LPG graph path in an automatically committed transaction.
    ///
    /// The parent must exist. Empty components and embedded separators are
    /// literal names; returns `false` when the path already exists.
    ///
    /// # Errors
    /// Returns parent, namespace, permission, allocation or durability errors.
    #[cfg(feature = "lpg")]
    pub fn create_graph_path(&self, path: &grafeo_common::types::GraphPath) -> Result<bool> {
        self.session().create_graph_path(path)
    }

    /// Drops one literal LPG subtree and its index/projection owners atomically.
    ///
    /// Returns `false` when absent. Explicit native selectors are not retargeted;
    /// selecting a removed path again requires creating it or choosing another.
    ///
    /// # Errors
    /// Rejects root/schema-default partitions and admission or durability failures.
    #[cfg(feature = "lpg")]
    pub fn drop_graph_path(&self, path: &grafeo_common::types::GraphPath) -> Result<bool> {
        self.session().drop_graph_path(path)
    }

    /// Creates a named graph. Returns `true` if created, `false` if it already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if arena allocation fails.
    #[cfg(feature = "lpg")]
    pub fn create_graph(&self, name: &str) -> Result<bool> {
        if self.lpg_mutation_closed() {
            return Ok(false);
        }
        self.session().create_named_graph(name)
    }

    /// Drops a named graph. Returns `true` if dropped, `false` if it did not exist.
    ///
    /// If the dropped graph was the active graph context, the context is reset
    /// to the default graph.
    ///
    /// # Errors
    /// Returns graph admission, durability, or context-preparation errors.
    #[cfg(feature = "lpg")]
    pub fn drop_graph(&self, name: &str) -> Result<bool> {
        let session = self.session();
        let captured = session.graph_context_snapshot();
        // Prepare the reset before DROP can publish. The post-publication
        // context update only moves this already-qualified path into place.
        let reset_path = Session::context_graph_path(captured.schema.as_deref(), None)?;
        let (dropped, dropped_key) = session.drop_named_graph_resolved(name)?;
        if dropped {
            // Re-enter the publication barrier before inspecting the final
            // namespace. If a replacement graph won between DROP publication
            // and this facade update, its freshly selected context must live.
            let _publication = crate::session::acquire_publication_read(&self.transaction_manager);
            let target_is_still_absent = self
                .store
                .as_ref()
                .is_none_or(|store| store.graph(&dropped_key).is_none());
            if target_is_still_absent {
                let mut context = self.current_context.write();
                if context.schema == captured.schema
                    && matches!(context.storage_key.components(), [key] if key == &dropped_key)
                {
                    context.graph = None;
                    context.native = false;
                    context.storage_key = reset_path;
                }
            }
        }
        Ok(dropped)
    }

    /// Returns all named graph names.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn list_graphs(&self) -> Vec<String> {
        let _publication = crate::session::acquire_publication_read(&self.transaction_manager);
        self.lpg_store().graph_names()
    }

    // === Graph Projections ===

    /// Creates a named graph projection (virtual subgraph).
    ///
    /// The projection filters the graph store to only include nodes with the
    /// specified labels and edges with the specified types. Returns `true` if
    /// created, `false` if a projection with that name already exists.
    /// Registry entries are process-local read views, not the durable,
    /// materialized RDF→LPG projections exposed by the RDF projection APIs.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    /// use grafeo_core::graph::ProjectionSpec;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let spec = ProjectionSpec::new()
    ///     .with_node_labels(["Person", "City"])
    ///     .with_edge_types(["LIVES_IN"]);
    /// assert!(db.create_projection("social", spec));
    /// ```
    pub fn create_projection(
        &self,
        name: impl Into<String>,
        spec: grafeo_core::graph::ProjectionSpec,
    ) -> bool {
        use grafeo_core::graph::GraphProjection;
        use std::collections::hash_map::Entry;

        // These legacy bool APIs register a process-local read view rather
        // than mutating authoritative graph data. Keep them usable for
        // read-only external stores while serializing their registry cut with
        // transactional projection DDL and graph lifecycle publication.
        let _lifecycle = self.is_open.read();
        let _publication = self.transaction_manager.publication().write();
        let store = self.graph_store();
        let projection = Arc::new(GraphProjection::new(store, spec));
        let registered = Arc::new(crate::session::RegisteredGraphProjection::root_or_external(
            projection,
        ));
        let mut projections = self.projections.write();
        match projections.entry(name.into()) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(registered);
                true
            }
        }
    }

    /// Drops a named graph projection. Returns `true` if it existed.
    pub fn drop_projection(&self, name: &str) -> bool {
        let _lifecycle = self.is_open.read();
        let _publication = self.transaction_manager.publication().write();
        self.projections.write().remove(name).is_some()
    }

    /// Returns the names of all graph projections.
    #[must_use]
    pub fn list_projections(&self) -> Vec<String> {
        let _publication = crate::session::acquire_publication_read(&self.transaction_manager);
        self.projections.read().keys().cloned().collect()
    }

    /// Returns a named projection as a [`GraphStoreSearch`] trait object.
    ///
    /// The returned `Arc` pins the exact view and source incarnation. Dropping
    /// the registry name or its source graph prevents future lookup but does
    /// not revoke an already acquired, read-only handle.
    #[must_use]
    pub fn projection(&self, name: &str) -> Option<Arc<dyn GraphStoreSearch>> {
        let _publication = crate::session::acquire_publication_read(&self.transaction_manager);
        self.projections
            .read()
            .get(name)
            .map(|entry| entry.view() as Arc<dyn GraphStoreSearch>)
    }

    /// Returns the graph store as a trait object.
    ///
    /// Returns a read-only trait object for the active graph store.
    ///
    /// This provides the [`GraphStoreSearch`] interface (graph-structure reads
    /// plus text/vector search capabilities) for code that only needs read
    /// operations. Database-owned writes go through [`Session`].
    ///
    /// After [`compact()`](Self::compact), this returns the retained Layered
    /// owner and reads both the cold base and its mutable overlay, including
    /// base-deletion tombstones. An external read store is likewise returned
    /// directly; otherwise reads use the built-in native store.
    ///
    /// The concrete root store is not part of the application API:
    ///
    /// ```compile_fail,E0599
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let _raw = db.store();
    /// ```
    #[must_use]
    pub fn graph_store(&self) -> Arc<dyn GraphStoreSearch> {
        if let Some(ref ext_read) = self.external_read_store {
            Arc::clone(ext_read)
        } else {
            #[cfg(feature = "lpg")]
            {
                Arc::clone(self.store_arc()) as Arc<dyn GraphStoreSearch>
            }
            #[cfg(not(feature = "lpg"))]
            {
                Arc::new(grafeo_core::graph::NullGraphStore) as Arc<dyn GraphStoreSearch>
            }
        }
    }

    /// Returns a caller-owned external writable graph store, if configured.
    ///
    /// The built-in LPG store is never exposed through `GraphStoreMut`; its
    /// authoritative writes are owned by the Session/publication protocol.
    /// Read-only external stores and ordinary built-in databases return `None`.
    #[must_use]
    pub fn graph_store_mut(&self) -> Option<Arc<dyn GraphStoreMut>> {
        #[cfg(any(feature = "wal", feature = "grafeo-file"))]
        if self.requires_store_authority() {
            return None;
        }
        #[cfg(feature = "lpg")]
        if self.store.is_some() {
            return None;
        }
        self.external_write_store.as_ref().map(Arc::clone)
    }

    /// Whether the built-in stores participate in an engine-owned durable
    /// publication boundary and therefore must reject unscoped mutation.
    #[cfg(any(feature = "wal", feature = "grafeo-file"))]
    fn requires_store_authority(&self) -> bool {
        #[cfg(all(feature = "wal", feature = "grafeo-file"))]
        {
            self.wal.is_some() || self.file_manager.is_some()
        }
        #[cfg(all(feature = "wal", not(feature = "grafeo-file")))]
        {
            self.wal.is_some()
        }
        #[cfg(all(not(feature = "wal"), feature = "grafeo-file"))]
        {
            self.file_manager.is_some()
        }
    }

    /// Garbage collects old MVCC versions that are no longer visible.
    ///
    /// Determines the minimum epoch required by active transactions and prunes
    /// version chains older than that threshold. Also cleans up completed
    /// transaction metadata in the transaction manager, and prunes the CDC
    /// event log according to its retention policy.
    ///
    /// # Errors
    ///
    /// Returns an error if Text or Vector history cannot be pruned under the
    /// current mutation authority or retention horizon, or if retained vector
    /// backing is invalid. WAL-backed Text/Vector collection is refused before any
    /// pruning until an owner-qualified durable GC transition is available.
    /// Collection is not atomic
    /// across the database's independently maintained structures.
    pub fn gc(&self) -> Result<()> {
        let lifecycle = self.is_open.read();
        if !*lifecycle {
            return Err(TransactionError::InvalidState("database is closed".into()).into());
        }
        if self.read_only {
            return Err(TransactionError::ReadOnly.into());
        }
        #[cfg(any(
            feature = "cdc",
            all(
                feature = "lpg",
                feature = "wal",
                any(feature = "text-index", feature = "vector-index")
            )
        ))]
        let _publication = self.transaction_manager.publication().write();
        if self.is_durability_poisoned() {
            return Err(TransactionError::DurabilityFailure(
                "cannot collect a durability-poisoned database; reopen and recover the WAL".into(),
            )
            .into());
        }
        #[cfg(all(
            feature = "lpg",
            feature = "wal",
            any(feature = "text-index", feature = "vector-index")
        ))]
        if self.wal.is_some() {
            let mut has_index = false;
            #[cfg(feature = "text-index")]
            {
                has_index |= !self.lpg_store().text_index_entries().is_empty();
            }
            #[cfg(feature = "vector-index")]
            {
                has_index |= !self.lpg_store().vector_index_entries().is_empty();
            }
            has_index |= self
                .catalog
                .all_indexes()
                .iter()
                .any(|owner| match owner.configuration {
                    #[cfg(feature = "text-index")]
                    crate::catalog::IndexConfiguration::Text { .. } => true,
                    #[cfg(feature = "vector-index")]
                    crate::catalog::IndexConfiguration::Vector { .. } => true,
                    _ => false,
                });
            if has_index {
                return Err(Error::Serialization("WAL-backed Text/Vector GC requires an exact owner-qualified durable GC transition".into()));
            }
        }
        #[cfg(feature = "lpg")]
        self.transaction_manager
            .with_write_authority(|| -> Result<()> {
                let min_epoch = self.transaction_manager.min_active_epoch();
                self.lpg_store().gc_versions(min_epoch);
                #[cfg(feature = "text-index")]
                self.lpg_store().gc_text_indexes(min_epoch)?;
                #[cfg(feature = "vector-index")]
                {
                    #[cfg(feature = "compact-store")]
                    if let Some(layered) = &self.layered_store {
                        layered.gc_vector_indexes(min_epoch)?;
                    } else {
                        self.lpg_store().gc_vector_indexes(min_epoch)?;
                    }
                    #[cfg(not(feature = "compact-store"))]
                    self.lpg_store().gc_vector_indexes(min_epoch)?;
                }
                Ok(())
            })?;
        #[cfg(feature = "cdc")]
        let current_epoch = self.transaction_manager.current_epoch();
        self.transaction_manager.gc();

        // Prune CDC events based on retention config (epoch + count limits)
        #[cfg(all(feature = "cdc", feature = "wal"))]
        {
            #[cfg(feature = "grafeo-file")]
            let directory = self.file_manager.is_none();
            #[cfg(not(feature = "grafeo-file"))]
            let directory = true;
            if directory && let Some(wal) = &self.wal {
                // The directory WAL still owns graph recovery. Persist only
                // the feed floor; this is not a graph checkpoint or log lease.
                self.cdc_log.retain_durably(current_epoch, wal)?;
                return Ok(());
            }
        }
        #[cfg(feature = "cdc")]
        self.cdc_log.apply_retention(current_epoch);
        Ok(())
    }

    /// Returns the buffer manager for memory-aware operations.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Returns a read-only view of the layered store, if
    /// [`compact()`](Self::compact) has been called.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    #[must_use]
    pub fn layered_store(&self) -> Option<LayeredStoreView> {
        self.layered_store
            .as_ref()
            .map(|store| LayeredStoreView::new(Arc::clone(store)))
    }

    /// Returns a read-only view of the compact-base tier, if [`Self::compact`]
    /// has been called and `mmap` is enabled.
    #[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
    #[must_use]
    pub fn compact_tiered(&self) -> Option<CompactStoreTieredView> {
        self.compact_tiered
            .as_ref()
            .zip(self.layered_store.as_ref())
            .map(|(tiered, layered)| {
                CompactStoreTieredView::new(Arc::clone(tiered), Arc::clone(layered))
            })
    }

    /// Returns the query cache.
    #[must_use]
    pub fn query_cache(&self) -> &Arc<QueryCache> {
        &self.query_cache
    }

    /// Number of cached physical operator trees (shared by every session).
    #[cfg(all(test, feature = "lpg", feature = "gql"))]
    #[must_use]
    pub(crate) fn physical_plan_cache_len(&self) -> usize {
        self.physical_cache.lock().len()
    }

    /// Clears all cached query plans.
    ///
    /// This is called automatically after DDL operations, but can also be
    /// invoked manually after external schema changes (e.g., WAL replay,
    /// import) or when you want to force re-optimization of all queries.
    ///
    /// Parsed/optimized logical plans and physical operator trees share the
    /// same `Arc`s as every session, so this invalidates one-shot
    /// [`execute`](Self::execute) as well as reused sessions.
    pub fn clear_plan_cache(&self) {
        self.query_cache.clear();
        self.physical_cache.lock().clear();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Closes the database, flushing all pending writes.
    ///
    /// For persistent databases, this ensures everything is safely on disk.
    /// Called automatically when the database is dropped, but you can call
    /// it explicitly if you need to guarantee durability at a specific point.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL can't be flushed (check disk space/permissions),
    /// or if an explicit Session transaction is still active (`close` will not
    /// snapshot uncommitted work).
    pub fn close(&self) -> Result<()> {
        self.close_with_lifecycle_guard(self.is_open.write(), false)
    }

    /// Closes the database without waiting for an active lifecycle reader.
    ///
    /// Use this from an event-loop callback that must remain available to close
    /// an outstanding result stream. Once admitted, flushing and cleanup use
    /// the same ordering and durability guarantees as [`close`](Self::close).
    ///
    /// # Errors
    ///
    /// Returns `TransactionInvalidState` while a query or stream retains the
    /// lifecycle, RDF commit, or publication lock. Otherwise returns the same errors as `close`.
    pub fn try_close(&self) -> Result<()> {
        let is_open = self.is_open.try_write().ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "database is busy; close active result streams before closing the database".into(),
            ))
        })?;
        self.close_with_lifecycle_guard(is_open, true)
    }

    fn close_with_lifecycle_guard(
        &self,
        mut is_open: parking_lot::RwLockWriteGuard<'_, bool>,
        nonblocking: bool,
    ) -> Result<()> {
        // Blocking close: lifecycle -> timer join -> RDF/publication -> C -> W.
        // Nonblocking close admits all gates before changing timer state.
        if !*is_open {
            return if self.is_durability_poisoned() {
                Err(Error::Transaction(
                    grafeo_common::utils::error::TransactionError::DurabilityFailure(
                        "database close failed; drop its owner and recover before further I/O"
                            .into(),
                    ),
                ))
            } else {
                Ok(())
            };
        }
        // An active transaction rejection is pre-admission and stays retryable.
        if !self.read_only {
            self.require_quiescent("close")?;
        }
        let poison = self.is_durability_poisoned().then(|| Error::Transaction(
            grafeo_common::utils::error::TransactionError::DurabilityFailure(
                "cannot close/checkpoint a durability-poisoned database; reopen and recover the WAL to resolve the commit outcome".into(),
            ),
        ));
        // Declare before the gates: unwind must release both gates before
        // this timer's Drop joins a worker that may be waiting on them.
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        let mut deferred_timer = None;
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        if !nonblocking && let Some(mut timer) = self.checkpoint_timer.lock().take() {
            timer.stop();
        }
        #[cfg(test)]
        publication_cut_test_point();
        let busy = || {
            Error::Transaction(TransactionError::InvalidState(
            "database is busy; finish active queries or close result streams before closing the database".into(),
        ))
        };
        #[cfg(feature = "triple-store")]
        let rdf_gate = if nonblocking {
            self.rdf_store.try_lock_commit().ok_or_else(busy)?
        } else {
            self.rdf_store.lock_commit()
        };
        let publication = if nonblocking {
            self.transaction_manager
                .publication()
                .try_write()
                .ok_or_else(busy)?
        } else {
            self.transaction_manager.publication().write()
        };
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        if nonblocking {
            deferred_timer = self.checkpoint_timer.lock().take();
            if let Some(timer) = &deferred_timer {
                timer.request_stop();
            }
        }
        let outcome = self.close_at_publication_cut(&mut is_open, poison);
        drop(publication);
        #[cfg(feature = "triple-store")]
        drop(rdf_gate);
        #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
        drop(deferred_timer);
        outcome
    }

    /// The lifecycle and publication owners remain held by the caller.
    fn close_at_publication_cut(&self, is_open: &mut bool, poison: Option<Error>) -> Result<()> {
        // Every error after admission converges on the same drain below.
        let outcome = (|| -> Result<()> {
            if let Some(error) = poison {
                return Err(error);
            }
            // A durable Session commit can decrement the active count before
            // its final fallible publication. Recheck at the owned cut: raw W
            // may still be healthy while shared engine publication is not.
            if self.is_durability_poisoned() {
                return Err(Error::Transaction(TransactionError::DurabilityFailure(
                    "cannot close/checkpoint a durability-poisoned database; reopen and recover the WAL to resolve the commit outcome".into(),
                )));
            }
            if !self.read_only {
                self.require_quiescent("close")?;
            }
            if self.read_only {
                #[cfg(feature = "grafeo-file")]
                if let Some(fm) = &self.file_manager {
                    fm.close()?;
                }
                return Ok(());
            }
            #[cfg(feature = "grafeo-file")]
            if let Some(fm) = &self.file_manager {
                #[cfg(feature = "wal")]
                if let Some(wal) = &self.wal {
                    wal.sync()?;
                }
                let flush_result = self.checkpoint_to_file(fm, flush::FlushReason::Explicit)?;
                #[cfg(feature = "wal")]
                let flush_result = if flush_result.sections_written == 0
                    && self.wal.as_ref().is_some_and(|wal| wal.record_count() > 0)
                {
                    self.checkpoint_to_file(fm, flush::FlushReason::Explicit)?
                } else {
                    flush_result
                };
                // Committed backup cursor generations share this owned WAL
                // namespace. Preserve its coordinates across close/reopen;
                // deleting it would reset sequences and orphan durable chains.
                #[cfg(feature = "wal")]
                let preserve_backup_state = match &self.wal {
                    Some(wal) => wal.capture()?.has_backup_generations()?,
                    None => false,
                };
                #[cfg(not(feature = "wal"))]
                let preserve_backup_state = false;
                let mut retirement = fm.sidecar_retirement()?;
                #[cfg(feature = "wal")]
                if let Some(wal) = &self.wal {
                    let seal = wal.seal()?;
                    *self.retained_wal_seal.lock() = Some(seal);
                }
                #[cfg(feature = "wal")]
                let has_wal_records = self.wal.as_ref().is_some_and(|wal| wal.record_count() > 0);
                #[cfg(not(feature = "wal"))]
                let has_wal_records = false;
                if !preserve_backup_state && (flush_result.sections_written > 0 || !has_wal_records)
                {
                    grafeo_common::testing::crash::maybe_crash("close:before_remove_sidecar_wal");
                    #[cfg(feature = "wal")]
                    match self.retained_wal_seal.lock().as_mut() {
                        Some(seal) => retirement.retire_sidecar(seal)?,
                        None => retirement.remove_unopened_sidecar()?,
                    }
                    #[cfg(not(feature = "wal"))]
                    retirement.remove_unopened_sidecar()?;
                } else if !preserve_backup_state {
                    grafeo_warn!(
                        "keeping sidecar WAL for recovery: checkpoint wrote 0 sections but WAL has records"
                    );
                }
                retirement.close()?;
                // Only successful C close releases the real seal.
                #[cfg(feature = "wal")]
                self.retained_wal_seal.lock().take();
                return Ok(());
            }
            #[cfg(feature = "wal")]
            if let Some(wal) = &self.wal {
                wal.close()?;
            }
            Ok(())
        })();
        *is_open = false;
        if let Err(error) = outcome {
            self.durability_poisoned
                .store(true, std::sync::atomic::Ordering::SeqCst);
            #[cfg(feature = "wal")]
            if self.retained_wal_seal.lock().is_none()
                && let Some(wal) = &self.wal
            {
                // The original C/checkpoint/engine error wins. The raw drain
                // closes FDs and workers but retains non-writable W until Drop.
                if let Err(drain_error) = wal.fail_and_drain() {
                    grafeo_warn!(
                        "secondary WAL drain error after failed database close: {drain_error}"
                    );
                }
            }
            return Err(error);
        }
        outcome
    }

    /// Returns a safe administrative WAL view, if durability is enabled.
    ///
    /// Raw record append is intentionally not exposed: transaction framing
    /// and commit markers are owned exclusively by the engine.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn wal(&self) -> Option<WalControl<'_>> {
        self.wal.as_deref().map(|wal| WalControl { wal })
    }

    /// Logs a WAL record if WAL is enabled. Fail-closed: log errors poison
    /// the database so later mutations cannot look successful.
    #[cfg(all(feature = "wal", feature = "lpg", feature = "triple-store"))]
    pub(super) fn log_wal(&self, record: &WalRecord) -> Result<()> {
        if self.is_durability_poisoned() {
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(
                    "database poisoned after WAL abort/log/fsync failure".into(),
                ),
            ));
        }
        if let Some(ref wal) = self.wal
            && let Err(e) = wal.log(record)
        {
            self.durability_poisoned
                .store(true, std::sync::atomic::Ordering::SeqCst);
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(format!(
                    "WAL log failed: {e}"
                )),
            ));
        }
        Ok(())
    }

    /// Registers storage sections as [`MemoryConsumer`]s with the BufferManager.
    ///
    /// Each section reports its memory usage to the buffer manager, enabling
    /// accurate pressure tracking. Called once after database construction.
    fn register_section_consumers(&mut self) {
        // LPG store section
        #[cfg(feature = "lpg")]
        let store_ref = self.store.as_ref();
        #[cfg(feature = "lpg")]
        if let Some(store) = store_ref {
            let lpg = grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store));
            self.buffer_manager.register_consumer(Arc::new(
                section_consumer::SectionConsumer::new(Arc::new(lpg)),
            ));
        }

        // RDF store: only when data exists
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        if !self.rdf_store.is_empty()
            || self.rdf_store.graph_count() > 0
            || self.rdf_store.has_history()
            || !self.rdf_projections.is_empty()
        {
            let rdf = grafeo_core::graph::rdf::RdfStoreSection::with_projections(
                Arc::clone(&self.rdf_store),
                Arc::clone(&self.rdf_projections),
            );
            self.buffer_manager.register_consumer(Arc::new(
                section_consumer::SectionConsumer::new(Arc::new(rdf)),
            ));
        }
        #[cfg(all(feature = "triple-store", not(feature = "lpg")))]
        if !self.rdf_store.is_empty()
            || self.rdf_store.graph_count() > 0
            || self.rdf_store.has_history()
        {
            let rdf = grafeo_core::graph::rdf::RdfStoreSection::new(Arc::clone(&self.rdf_store));
            self.buffer_manager.register_consumer(Arc::new(
                section_consumer::SectionConsumer::new(Arc::new(rdf)),
            ));
        }

        // Ring Index: only when Ring has been built
        #[cfg(feature = "ring-index")]
        if self.rdf_store.ring().is_some() {
            let ring = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(&self.rdf_store));
            // Ring is the first index section to opt into spill wiring.
            // The Ring's `swap_to_mmap` still returns `NotSupported` until
            // its packed disk format lands, so a spill attempt today fails
            // cleanly without leaking files; once the format is in place,
            // no engine-side change is required to enable real eviction.
            let consumer = match self.buffer_manager.config().spill_path.clone() {
                Some(path) => section_consumer::SectionConsumer::with_spill(Arc::new(ring), path),
                None => section_consumer::SectionConsumer::new(Arc::new(ring)),
            };
            self.buffer_manager.register_consumer(Arc::new(consumer));
        }

        // Text indexes: same dynamic approach as vector indexes.
        #[cfg(all(feature = "lpg", feature = "text-index"))]
        if let Some(store) = store_ref {
            self.buffer_manager
                .register_consumer(Arc::new(section_consumer::TextIndexConsumer::new(store)));
        }

        // CDC log: register as memory consumer so the buffer manager can
        // prune events under memory pressure.
        #[cfg(feature = "cdc")]
        self.buffer_manager.register_consumer(
            Arc::clone(&self.cdc_log) as Arc<dyn grafeo_common::memory::MemoryConsumer>
        );
    }

    /// Applies `TierOverride::ForceDisk` and `TierOverride::ForceRam`
    /// overrides at database open time.
    ///
    /// For each section type in [`Config::section_configs`]:
    ///
    /// - `ForceDisk`: spills the matching registered consumer (named
    ///   `"section:<TypeName>"`) once.
    /// - `ForceRam` (Phase 8g): pins the matching consumer in the buffer
    ///   manager so subsequent spill loops (pressure-driven, explicit, or
    ///   targeted) skip it.
    /// - `Auto`: no action; the BufferManager applies its default policy.
    ///
    /// Must be called after [`Self::register_section_consumers`].
    fn apply_force_disk_overrides(&self) {
        use grafeo_common::storage::TierOverride;

        for (section_type, mem_config) in &self.config.section_configs {
            let consumer_name = format!("section:{section_type:?}");
            match mem_config.tier {
                TierOverride::ForceDisk => {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "grafeo::tier",
                        section = ?section_type,
                        tier = "ForceDisk",
                        "applying tier override at db open"
                    );
                    self.buffer_manager.spill_consumer_by_name(&consumer_name);
                }
                TierOverride::ForceRam => {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "grafeo::tier",
                        section = ?section_type,
                        tier = "ForceRam",
                        "pinning consumer to RAM"
                    );
                    self.buffer_manager.mark_force_ram(&consumer_name);
                }
                TierOverride::Auto => {}
                _ => {}
            }
        }
    }

    /// Reloads spilled consumers back into RAM up to a target memory fraction.
    ///
    /// Phase 9a: closes the spill / reload loop. After memory pressure drops
    /// (e.g. a workload finishes, or a checkpoint freed mutation overlay
    /// state), call this to bring spilled section data back into RAM for
    /// faster subsequent reads.
    ///
    /// Walks consumers currently reporting `StorageTier::OnDisk`, in priority
    /// order (highest first), reloading each as long as projected memory
    /// usage stays below `target_fraction * memory_limit`.
    ///
    /// Returns the number of consumers successfully reloaded.
    ///
    /// `target_fraction` is clamped to `[0.0, 1.0]`. A typical value is `0.7`
    /// (matching the default `soft_limit_fraction`).
    pub fn reload_eligible(&self, target_fraction: f64) -> usize {
        self.buffer_manager.reload_eligible(target_fraction)
    }

    /// Returns the current [`StorageTier`] of every registered section consumer.
    ///
    /// The map keys are the [`SectionType`]s parsed from each consumer's name
    /// (consumers whose names don't follow the `"section:<TypeName>"` convention
    /// are skipped). Tier classification is best-effort: a consumer reporting
    /// zero `memory_usage()` and `can_spill() == true` is reported as `OnDisk`,
    /// otherwise `InMemory` (or `Uninitialized` if both are zero).
    ///
    /// Useful for tests, observability, and binding-side introspection.
    ///
    /// [`StorageTier`]: grafeo_common::memory::buffer::StorageTier
    /// [`SectionType`]: grafeo_common::storage::SectionType
    #[must_use]
    pub fn storage_tiers(
        &self,
    ) -> hashbrown::HashMap<
        grafeo_common::storage::SectionType,
        grafeo_common::memory::buffer::StorageTier,
    > {
        use grafeo_common::storage::SectionType;
        let snapshot = self.buffer_manager.snapshot_consumer_tiers();
        let mut out = hashbrown::HashMap::new();
        for (name, tier) in snapshot {
            let Some(suffix) = name.strip_prefix("section:") else {
                continue;
            };
            let section_type = match suffix {
                "LpgStore" => SectionType::LpgStore,
                "RdfStore" => SectionType::RdfStore,
                "CompactStore" => SectionType::CompactStore,
                "VectorStore" => SectionType::VectorStore,
                "TextIndex" => SectionType::TextIndex,
                "RdfRing" => SectionType::RdfRing,
                "PropertyIndex" => SectionType::PropertyIndex,
                "Catalog" => SectionType::Catalog,
                _ => continue,
            };
            out.insert(section_type, tier);
        }
        out
    }

    /// Discovers and re-opens spill files from a previous session.
    ///
    /// Builds section objects for the current database state.
    #[cfg(feature = "grafeo-file")]
    fn build_sections(
        &self,
        mark_omitted_deletions_clean: bool,
    ) -> Result<(
        Vec<Box<dyn grafeo_common::storage::Section>>,
        Option<world_metadata::EncodedSection>,
    )> {
        #[cfg(not(all(feature = "compact-store", feature = "lpg")))]
        let _ = mark_omitted_deletions_clean;
        #[cfg(feature = "lpg")]
        let mut captured_lpg = None;
        #[cfg(not(feature = "lpg"))]
        let captured_lpg = None;
        let mut sections: Vec<Box<dyn grafeo_common::storage::Section>> = Vec::new();

        // Layered store: serialize the compact base, its mutable LPG overlay,
        // and any base tombstones. These sections replace only the ordinary
        // LPG store section; catalog, index, RDF, and Ring sections below are
        // orthogonal database state and must still be included in the same
        // whole-container snapshot.
        #[cfg(all(feature = "compact-store", feature = "lpg"))]
        if matches!(
            self.config.graph_model,
            crate::config::GraphModel::Lpg | crate::config::GraphModel::Both
        ) && let Some(ref layered) = self.layered_store
        {
            // Compact base section.
            let compact_section = grafeo_core::graph::compact::section::CompactStoreSection::new(
                layered.base_store_arc(),
            );
            sections.push(Box::new(compact_section));

            // Overlay deletion log: persists base-node/edge tombstones
            // that have not yet been merged into the base. Without this,
            // close+reopen silently un-deletes those entities. Only push
            // when there is actually something to record so we don't
            // emit an empty section on every checkpoint.
            let deletions = grafeo_core::graph::compact::deletions_section::OverlayDeletionsSection::from_layered(
                Arc::clone(layered),
            );
            if !deletions.is_empty() {
                sections.push(Box::new(deletions));
            } else if mark_omitted_deletions_clean {
                // The set may have transitioned from non-empty to empty
                // (e.g. a compact merged the deletes); make sure the
                // dirty flag is cleared so subsequent checkpoints don't
                // think they need to keep flushing.
                layered.mark_deletions_clean();
            }
        }

        // RDF-only containers still carry the canonical catalog component,
        // but must not serialize an artificial empty LPG store merely because
        // this binary was compiled with LPG support.
        #[cfg(feature = "lpg")]
        if matches!(self.config.graph_model, crate::config::GraphModel::Rdf) {
            sections.push(Box::new(catalog_state_section::CatalogStateSection::new(
                Arc::clone(&self.catalog),
            )));
        }
        #[cfg(not(feature = "lpg"))]
        sections.push(Box::new(catalog_state_section::CatalogStateSection::new(
            Arc::clone(&self.catalog),
        )));

        // LPG sections: store, catalog, vector indexes, text indexes
        #[cfg(feature = "lpg")]
        if matches!(
            self.config.graph_model,
            crate::config::GraphModel::Lpg | crate::config::GraphModel::Both
        ) && let Some(store) = self.store.as_ref()
        {
            #[cfg(feature = "compact-store")]
            if let Some(layered) = &self.layered_store
                && !Arc::ptr_eq(store, &layered.overlay_store())
            {
                return Err(Error::Serialization(
                    "compact overlay differs from the captured LPG root".into(),
                ));
            }
            let captured =
                grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store)).capture()?;
            #[cfg(all(
                test,
                feature = "wal",
                feature = "text-index",
                feature = "vector-index"
            ))]
            recursive_capture_tests::after_lpg_capture();
            captured_lpg = Some(world_metadata::EncodedSection::new(
                grafeo_common::storage::SectionType::LpgStore,
                4,
                captured.bytes,
            )?);
            let index_graphs = index_sections::LpgIndexGraphCut::from_graphs(captured.graphs);

            let catalog = catalog_section::CatalogSection::new_with_graphs(
                Arc::clone(&self.catalog),
                index_graphs.graphs(),
                {
                    let tm = Arc::clone(&self.transaction_manager);
                    move || tm.current_epoch().as_u64()
                },
            )?;

            sections.push(Box::new(catalog));

            // Vector indexes: persist HNSW topology to avoid rebuild on load
            #[cfg(feature = "vector-index")]
            {
                let indexes = index_graphs.vector_views()?;
                if !indexes.is_empty() {
                    let vector =
                        grafeo_core::index::vector::VectorStoreSection::from_views(indexes);
                    sections.push(Box::new(vector));
                }
            }

            // Text indexes: persist BM25 postings to avoid rebuild on load
            #[cfg(feature = "text-index")]
            {
                let indexes = index_graphs.text_views()?;
                if !indexes.is_empty() {
                    let text = grafeo_core::index::text::TextIndexSection::from_views(indexes);
                    sections.push(Box::new(text));
                }
            }
        }

        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        if matches!(
            self.config.graph_model,
            crate::config::GraphModel::Rdf | crate::config::GraphModel::Both
        ) {
            let rdf = grafeo_core::graph::rdf::RdfStoreSection::with_projections_under_commit_gate(
                Arc::clone(&self.rdf_store),
                Arc::clone(&self.rdf_projections),
            )?;
            sections.push(Box::new(rdf));
        }

        #[cfg(all(feature = "triple-store", not(feature = "lpg")))]
        if matches!(
            self.config.graph_model,
            crate::config::GraphModel::Rdf | crate::config::GraphModel::Both
        ) {
            let rdf = grafeo_core::graph::rdf::RdfStoreSection::new_under_commit_gate(Arc::clone(
                &self.rdf_store,
            ))?;
            sections.push(Box::new(rdf));
        }

        #[cfg(feature = "ring-index")]
        if matches!(
            self.config.graph_model,
            crate::config::GraphModel::Rdf | crate::config::GraphModel::Both
        ) && self.rdf_store.ring().is_some()
        {
            let ring = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(&self.rdf_store));
            sections.push(Box::new(ring));
        }

        sections.push(Box::new(cdc_checkpoint::CapturedSection(
            cdc_checkpoint::capture(self, self.transaction_manager.current_epoch())?,
        )));
        Ok((sections, captured_lpg))
    }

    /// Captures an integrity-sealed world cut for the current committed state.
    ///
    /// The cut seals exact authoritative serializer versions and bytes, plus
    /// StoreId, epoch, graph model, canonical schema and verified projection
    /// receipts. Capturing is O(database size) because the state components are
    /// serialized in memory; it performs no file write and does not mark source
    /// sections clean.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is active, durability is poisoned,
    /// projection provenance is inconsistent, or serialization fails.
    #[cfg(feature = "grafeo-file")]
    pub fn world_cut(&self) -> Result<grafeo_common::types::WorldCut> {
        let _capture = self.acquire_quiescent_capture("capture a world cut from")?;
        self.world_cut_under_capture()
    }

    /// Captures a logical WorldCut while the caller owns the quiescent
    /// publication boundary. Backup uses this to bind metadata to the exact
    /// container image copied under that same guard.
    #[cfg(feature = "grafeo-file")]
    fn world_cut_under_capture(&self) -> Result<grafeo_common::types::WorldCut> {
        #[cfg(all(feature = "triple-store", feature = "lpg"))]
        self.validate_live_rdf_lpg_projection_rows()?;

        let (sections, captured_lpg) = self.build_sections(false)?;
        let section_refs: Vec<&dyn grafeo_common::storage::Section> =
            sections.iter().map(|section| section.as_ref()).collect();
        let encoded = world_metadata::encode_sections(&section_refs, captured_lpg)?;
        let graph_model = grafeo_common::types::GraphModelTag::from_u8(
            self.config.graph_model.as_u8(),
        )
        .map_err(|error| Error::Serialization(format!("capture world-cut graph model: {error}")))?;
        let world_identity = self.world_identity();
        #[cfg(feature = "triple-store")]
        let world_identity = world_metadata::validate_live_world_identity(
            world_identity,
            graph_model,
            &self.rdf_store,
        )?;
        let inputs = world_metadata::capture_world_cut_inputs(
            world_identity,
            self.transaction_manager.current_epoch(),
            graph_model,
            &self.catalog,
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            &self.rdf_projections,
        )?;
        world_metadata::seal_logical_world_cut(&encoded, inputs)
    }

    // =========================================================================
    // Backup API
    // =========================================================================

    /// Creates a full backup of the database in the given directory.
    ///
    /// Checkpoints the database, copies the `.grafeo` file, and creates a
    /// backup manifest. Subsequent incremental backups will use this as the
    /// base.
    ///
    /// # Errors
    ///
    /// Returns an error if the database has no file manager, I/O fails, or an
    /// explicit Session transaction is still active.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    pub fn backup_full(&self, backup_dir: &std::path::Path) -> Result<backup::BackupSegment> {
        let _capture = self.acquire_quiescent_capture("create a full backup from")?;
        let current_epoch = self.transaction_manager.current_epoch();
        backup::validate_committed_backup_epoch(current_epoch, "full backup")?;
        let fm = self
            .file_manager
            .as_ref()
            .ok_or_else(|| Error::Internal("backup requires a persistent database".to_string()))?;

        // Checkpoint to ensure the container has the latest data.
        // Skip for read-only databases: the on-disk file is already a valid
        // snapshot and the file manager rejects writes.
        if !self.read_only {
            let _ = self.checkpoint_to_file(fm, flush::FlushReason::Explicit)?;
        }

        let world_cut = self
            .world_cut_under_capture()?
            .encode()
            .map_err(|error| Error::Serialization(format!("encode backup WorldCut: {error}")))?;
        backup::do_backup_full(
            backup_dir,
            fm,
            self.wal.as_deref(),
            current_epoch,
            self.world_identity().store_id().into_bytes(),
            self.graph_model().as_u8(),
            Some(world_cut),
        )
    }

    /// Creates an incremental backup containing WAL records since the last backup.
    ///
    /// Requires a prior full backup in the backup directory.
    ///
    /// # Errors
    ///
    /// Returns an error if no full backup exists, or if the WAL has no new records.
    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    pub fn backup_incremental(
        &self,
        backup_dir: &std::path::Path,
    ) -> Result<backup::BackupSegment> {
        let _capture = self.acquire_quiescent_capture("create an incremental backup from")?;
        let wal = self
            .wal
            .as_ref()
            .ok_or_else(|| Error::Internal("incremental backup requires WAL".to_string()))?;

        // The lifecycle/RDF/publication capture guard prevents transaction
        // registration and commit publication until the WAL cut, manifest and
        // cursor have all been published as one ordered backup operation.
        let current_epoch = self.transaction_manager.current_epoch();
        let world_cut = self.world_cut_under_capture()?.encode().map_err(|error| {
            Error::Serialization(format!("encode incremental WorldCut: {error}"))
        })?;
        backup::do_backup_incremental(backup_dir, wal, current_epoch, world_cut)
    }

    /// Returns the backup manifest for a backup directory, if one exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest file exists but cannot be parsed.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    pub fn read_backup_manifest(
        backup_dir: &std::path::Path,
    ) -> Result<Option<backup::BackupManifest>> {
        backup::read_manifest(backup_dir)
    }

    /// Returns the current backup cursor (last backed-up position), if any.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    ///
    /// # Errors
    /// Returns terminal-state, checked filesystem or cursor decoding errors.
    pub fn backup_cursor(&self) -> Result<Option<backup::BackupCursor>> {
        match self.wal.as_ref() {
            Some(wal) => backup::read_backup_cursor(&mut wal.capture()?),
            None => Ok(None),
        }
    }

    /// Restores a validated v2 chain to a closed, complete container image.
    ///
    /// Replays and verifies the selected chain in detached staging, reopens
    /// the target cut, then atomically replaces `output_path`. An existing
    /// destination WAL sidecar is rejected without changing the destination.
    ///
    /// # Errors
    ///
    /// Returns an error for targets outside coverage, invalid source metadata
    /// or content, destination contention, failed replay/verification or I/O.
    /// An error after replacement reports an uncertain durability acknowledgement.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    pub fn restore_to_epoch(
        backup_dir: &std::path::Path,
        target_epoch: grafeo_common::types::EpochId,
        output_path: &std::path::Path,
    ) -> Result<()> {
        backup::do_restore_to_epoch(backup_dir, target_epoch, output_path)
    }

    /// Writes the current database state to the `.grafeo` file using the unified flush.
    ///
    /// Does NOT remove the sidecar WAL: callers that want to clean up
    /// the sidecar (e.g. `close()`) should call `fm.remove_sidecar_wal()`
    /// separately after this returns.
    #[cfg(feature = "grafeo-file")]
    fn checkpoint_to_file(
        &self,
        fm: &GrafeoFileManager,
        reason: flush::FlushReason,
    ) -> Result<flush::FlushResult> {
        let (sections, captured_lpg) = self.build_sections(true)?;
        let section_refs: Vec<&dyn grafeo_common::storage::Section> =
            sections.iter().map(|s| s.as_ref()).collect();
        let epoch = self.transaction_manager.current_epoch();
        let graph_model =
            grafeo_common::types::GraphModelTag::from_u8(self.config.graph_model.as_u8()).map_err(
                |error| Error::Serialization(format!("capture checkpoint graph model: {error}")),
            )?;
        let world_identity = self.world_identity();
        #[cfg(feature = "triple-store")]
        let world_identity = world_metadata::validate_live_world_identity(
            world_identity,
            graph_model,
            &self.rdf_store,
        )?;
        let world_cut = world_metadata::capture_world_cut_inputs(
            world_identity,
            epoch,
            graph_model,
            &self.catalog,
            #[cfg(all(feature = "triple-store", feature = "lpg"))]
            &self.rdf_projections,
        )?;
        #[cfg(feature = "lpg")]
        let context =
            flush::build_context(self.read_graph_view(), &self.transaction_manager, world_cut);
        #[cfg(not(feature = "lpg"))]
        let context =
            flush::build_context_minimal(&self.transaction_manager, epoch.as_u64(), world_cut);

        flush::flush(
            fm,
            &section_refs,
            captured_lpg,
            &context,
            reason,
            #[cfg(feature = "wal")]
            self.wal.as_deref(),
        )
    }

    /// Returns a read-only view of the database file, if using single-file
    /// format.
    #[cfg(feature = "grafeo-file")]
    #[must_use]
    pub fn file_manager(&self) -> Option<DatabaseFileView> {
        self.file_manager
            .as_ref()
            .map(|manager| DatabaseFileView::new(Arc::clone(manager)))
    }

    fn require_quiescent(&self, op: &str) -> Result<()> {
        let n = self.transaction_manager.active_count();
        if n > 0 {
            return Err(Error::Transaction(TransactionError::InvalidState(format!(
                "{op} requires a quiescent committed cut; {n} transaction(s) still active"
            ))));
        }
        Ok(())
    }

    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    fn require_no_sessions(&self, op: &str) -> Result<()> {
        let count = self
            .active_sessions
            .load(std::sync::atomic::Ordering::Acquire);
        if count > 0 {
            return Err(Error::Transaction(TransactionError::InvalidState(format!(
                "{op} replaces the active store topology; drop {count} live Session handle(s) and retry"
            ))));
        }
        Ok(())
    }

    /// Forces a durable snapshot, then retires WAL that is now in that snapshot.
    ///
    /// Snapshot/container write happens first. Checkpoint metadata and old-log
    /// truncation happen only after that write returns. A crash between the two
    /// leaves the WAL intact so recovery can replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot or WAL checkpoint fails, or if an
    /// explicit Session transaction is still active.
    pub fn wal_checkpoint(&self) -> Result<()> {
        #[cfg(feature = "triple-store")]
        let _rdf_gate = self.rdf_store.lock_commit();
        let _publication = self.transaction_manager.publication().write();
        if self.read_only {
            return Ok(());
        }
        self.require_quiescent("wal_checkpoint")?;

        #[cfg(all(feature = "wal", feature = "grafeo-file"))]
        if let Some(ref wal) = self.wal {
            wal.sync()?;
        }

        #[cfg(feature = "grafeo-file")]
        if let Some(ref fm) = self.file_manager {
            let _ = self.checkpoint_to_file(fm, flush::FlushReason::Explicit)?;
            {
                use grafeo_common::testing::crash::maybe_crash;
                maybe_crash("checkpoint:after_snapshot_before_wal_retire");
            }
        }

        // Directory WAL is the source of truth: never write checkpoint.meta
        // (that would skip older logs). Single-file format may retire WAL
        // now that the container snapshot is durable.
        #[cfg(all(feature = "wal", feature = "grafeo-file"))]
        let retire = self.file_manager.is_some();
        #[cfg(all(feature = "wal", not(feature = "grafeo-file")))]
        let retire = false;

        #[cfg(feature = "wal")]
        if retire && let Some(ref wal) = self.wal {
            let epoch = {
                #[cfg(all(feature = "lpg", feature = "triple-store"))]
                {
                    let lpg = self.lpg_store().current_epoch();
                    let rdf = self.rdf_store.commit_epoch();
                    if rdf > lpg { rdf } else { lpg }
                }
                #[cfg(all(feature = "lpg", not(feature = "triple-store")))]
                {
                    self.lpg_store().current_epoch()
                }
                #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
                {
                    self.rdf_store.commit_epoch()
                }
                #[cfg(all(not(feature = "lpg"), not(feature = "triple-store")))]
                {
                    self.transaction_manager.current_epoch()
                }
            };
            // A checkpoint is a system boundary, not a synthetic statement in
            // the most recently assigned user transaction. Reusing that id can
            // make aborted/rejected transaction metadata look authenticated.
            wal.checkpoint(grafeo_common::types::TransactionId::SYSTEM, epoch)?;
        }
        #[cfg(feature = "wal")]
        if !retire && let Some(ref wal) = self.wal {
            wal.sync()?;
        }

        Ok(())
    }
}

impl Drop for GrafeoDB {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            grafeo_error!("Error closing database: {}", e);
        }
    }
}

#[cfg(feature = "lpg")]
impl crate::admin::AdminService for GrafeoDB {
    fn info(&self) -> crate::admin::DatabaseInfo {
        self.info()
    }

    fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        self.detailed_stats()
    }

    fn schema(&self) -> crate::admin::SchemaInfo {
        self.schema()
    }

    fn validate(&self) -> crate::admin::ValidationResult {
        self.validate()
    }

    fn wal_status(&self) -> Result<crate::admin::WalStatus> {
        self.wal_status()
    }

    fn wal_checkpoint(&self) -> Result<()> {
        self.wal_checkpoint()
    }
}

// =========================================================================
// Query Result Types
// =========================================================================

/// A retained accounting token for resident query-result storage.
///
/// The token is shared with owned row extraction so moving rows out of a
/// `QueryResult` cannot release their reservation prematurely.
#[derive(Debug)]
pub(crate) struct ResultReservation {
    grant: parking_lot::Mutex<MemoryGrant>,
}

impl ResultReservation {
    pub(crate) fn new(grant: MemoryGrant) -> Self {
        Self {
            grant: parking_lot::Mutex::new(grant),
        }
    }

    pub(crate) fn try_resize(&self, target: usize) -> Result<()> {
        self.grant.lock().try_resize(target).map_err(|error| {
            Error::Storage(grafeo_common::utils::error::StorageError::Full)
                .with_context(error.to_string())
        })
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[must_use]
    pub(crate) fn size(&self) -> usize {
        self.grant.lock().size()
    }
}

/// Owned rows extracted from a bounded query result.
#[derive(Debug)]
pub struct OwnedRows {
    rows: Vec<Vec<grafeo_common::types::Value>>,
    _reservation: Option<Arc<ResultReservation>>,
}

impl OwnedRows {
    /// Creates an empty row owner without allocating or reserving memory.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            rows: Vec::new(),
            _reservation: None,
        }
    }

    pub(crate) fn new(
        rows: Vec<Vec<grafeo_common::types::Value>>,
        reservation: Option<Arc<ResultReservation>>,
    ) -> Self {
        Self {
            rows,
            _reservation: reservation,
        }
    }
}

impl std::ops::Deref for OwnedRows {
    type Target = [Vec<grafeo_common::types::Value>];

    fn deref(&self) -> &Self::Target {
        &self.rows
    }
}

/// One owned row that retains the result reservation until it is dropped.
#[derive(Debug)]
pub struct OwnedRow {
    row: Vec<grafeo_common::types::Value>,
    _reservation: Option<Arc<ResultReservation>>,
}

impl std::ops::Deref for OwnedRow {
    type Target = [grafeo_common::types::Value];

    fn deref(&self) -> &Self::Target {
        &self.row
    }
}

impl<'a> IntoIterator for &'a OwnedRows {
    type Item = &'a Vec<grafeo_common::types::Value>;
    type IntoIter = std::slice::Iter<'a, Vec<grafeo_common::types::Value>>;
    fn into_iter(self) -> Self::IntoIter {
        self.rows.iter()
    }
}

impl IntoIterator for OwnedRows {
    type Item = OwnedRow;
    type IntoIter = OwnedRowsIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        OwnedRowsIntoIter {
            rows: self.rows.into_iter(),
            _reservation: self._reservation,
        }
    }
}

/// Iterator retaining the result reservation while owned rows are consumed.
pub struct OwnedRowsIntoIter {
    rows: std::vec::IntoIter<Vec<grafeo_common::types::Value>>,
    _reservation: Option<Arc<ResultReservation>>,
}

impl Iterator for OwnedRowsIntoIter {
    type Item = OwnedRow;

    fn next(&mut self) -> Option<Self::Item> {
        self.rows.next().map(|row| OwnedRow {
            row,
            _reservation: self._reservation.clone(),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.rows.size_hint()
    }
}

impl ExactSizeIterator for OwnedRowsIntoIter {}

/// Owned dense integer columns extracted from a bounded query result.
#[derive(Debug)]
pub struct OwnedInt64Columns {
    columns: Vec<Vec<i64>>,
    _reservation: Option<Arc<ResultReservation>>,
}

impl OwnedInt64Columns {
    pub(crate) fn new(columns: Vec<Vec<i64>>, reservation: Option<Arc<ResultReservation>>) -> Self {
        Self {
            columns,
            _reservation: reservation,
        }
    }
}

impl std::ops::Deref for OwnedInt64Columns {
    type Target = [Vec<i64>];

    fn deref(&self) -> &Self::Target {
        &self.columns
    }
}

/// One owned dense column retaining the result reservation until dropped.
#[derive(Debug)]
pub struct OwnedInt64Column {
    column: Vec<i64>,
    _reservation: Option<Arc<ResultReservation>>,
}

impl std::ops::Deref for OwnedInt64Column {
    type Target = [i64];

    fn deref(&self) -> &Self::Target {
        &self.column
    }
}

impl<'a> IntoIterator for &'a OwnedInt64Columns {
    type Item = &'a Vec<i64>;
    type IntoIter = std::slice::Iter<'a, Vec<i64>>;
    fn into_iter(self) -> Self::IntoIter {
        self.columns.iter()
    }
}

impl IntoIterator for OwnedInt64Columns {
    type Item = OwnedInt64Column;
    type IntoIter = OwnedInt64ColumnsIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        OwnedInt64ColumnsIntoIter {
            columns: self.columns.into_iter(),
            _reservation: self._reservation,
        }
    }
}

/// Iterator retaining the result reservation while dense columns are consumed.
pub struct OwnedInt64ColumnsIntoIter {
    columns: std::vec::IntoIter<Vec<i64>>,
    _reservation: Option<Arc<ResultReservation>>,
}

impl Iterator for OwnedInt64ColumnsIntoIter {
    type Item = OwnedInt64Column;

    fn next(&mut self) -> Option<Self::Item> {
        self.columns.next().map(|column| OwnedInt64Column {
            column,
            _reservation: self._reservation.clone(),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.columns.size_hint()
    }
}

impl ExactSizeIterator for OwnedInt64ColumnsIntoIter {}

/// The result of running a query.
///
/// Contains rows and columns, like a table. Use [`iter()`](Self::iter) to
/// loop through rows, or [`scalar()`](Self::scalar) if you expect a single value.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// let db = GrafeoDB::new_in_memory();
/// db.create_node(&["Person"]);
///
/// let result = db.execute("MATCH (p:Person) RETURN count(p) AS total")?;
///
/// // Check what we got
/// println!("Columns: {:?}", result.columns);
/// println!("Rows: {}", result.row_count());
///
/// // Iterate through results
/// for row in result.iter() {
///     println!("{:?}", row);
/// }
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
#[derive(Debug)]
pub struct QueryResult {
    /// Column names from the RETURN clause.
    pub columns: Vec<String>,
    /// Column types - useful for distinguishing NodeId/EdgeId from plain integers.
    pub column_types: Vec<grafeo_common::types::LogicalType>,
    /// The actual result rows.
    ///
    /// Use [`rows()`](Self::rows) for borrowed access or
    /// [`into_rows()`](Self::into_rows) to take ownership.
    pub(crate) rows: Vec<Vec<grafeo_common::types::Value>>,
    /// Dense Int64 columns collected without per-row `Vec`s (`RETURN id(c)`).
    /// When `Some`, [`row_count`](Self::row_count) is `int64_cols[0].len()`.
    pub(crate) int64_cols: Option<Vec<Vec<i64>>>,
    /// Materialized row view of [`int64_cols`](Self::int64_cols) (lazy).
    pub(crate) cached_int64_rows: std::sync::OnceLock<Vec<Vec<grafeo_common::types::Value>>>,
    /// Query execution time in milliseconds (if timing was enabled).
    pub execution_time_ms: Option<f64>,
    /// Number of rows scanned during query execution (estimate).
    pub rows_scanned: Option<u64>,
    /// Status message for DDL and session commands (e.g., "Created node type 'Person'").
    pub status_message: Option<String>,
    /// GQLSTATUS code per ISO/IEC 39075:2024, sec 23.
    pub gql_status: grafeo_common::utils::GqlStatus,
    pub(crate) result_reservation: Option<Arc<ResultReservation>>,
}

impl QueryResult {
    /// Creates a fully empty query result (no columns, no rows).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            int64_cols: None,
            cached_int64_rows: std::sync::OnceLock::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
            result_reservation: None,
        }
    }

    /// Creates a query result with only a status message (for DDL commands).
    #[must_use]
    pub fn status(msg: impl Into<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            int64_cols: None,
            cached_int64_rows: std::sync::OnceLock::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: Some(msg.into()),
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
            result_reservation: None,
        }
    }

    /// Creates a new empty query result.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows: Vec::new(),
            int64_cols: None,
            cached_int64_rows: std::sync::OnceLock::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
            result_reservation: None,
        }
    }

    /// Creates a new empty query result with column types.
    #[must_use]
    pub fn with_types(
        columns: Vec<String>,
        column_types: Vec<grafeo_common::types::LogicalType>,
    ) -> Self {
        Self {
            columns,
            column_types,
            rows: Vec::new(),
            int64_cols: None,
            cached_int64_rows: std::sync::OnceLock::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
            result_reservation: None,
        }
    }

    /// Creates a query result with pre-populated rows.
    #[must_use]
    pub fn from_rows(columns: Vec<String>, rows: Vec<Vec<grafeo_common::types::Value>>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows,
            int64_cols: None,
            cached_int64_rows: std::sync::OnceLock::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
            result_reservation: None,
        }
    }

    /// Appends a row to an untracked, caller-owned result.
    ///
    /// # Errors
    /// Rejects growth of an engine-produced result outside its bounded collector.
    pub fn push_row(&mut self, row: Vec<grafeo_common::types::Value>) -> Result<()> {
        if self.result_reservation.is_some() {
            return Err(Error::Internal(
                "engine-produced results cannot be extended outside the bounded collector"
                    .to_string(),
            ));
        }
        self.push_row_unchecked(row);
        Ok(())
    }

    fn push_row_unchecked(&mut self, row: Vec<grafeo_common::types::Value>) {
        if let Some(cols) = self.int64_cols.take() {
            self.rows = self
                .cached_int64_rows
                .take()
                .unwrap_or_else(|| materialize_int64_rows(&cols));
        }
        self.rows.push(row);
    }

    /// Attaches the reservation acquired by a query collector.
    pub(crate) fn with_result_reservation(mut self, grant: MemoryGrant) -> Self {
        self.result_reservation = Some(Arc::new(ResultReservation::new(grant)));
        self
    }

    /// Sets the execution metrics on this result.
    pub fn with_metrics(mut self, execution_time_ms: f64, rows_scanned: u64) -> Self {
        self.execution_time_ms = Some(execution_time_ms);
        self.rows_scanned = Some(rows_scanned);
        self
    }

    /// Returns the execution time in milliseconds, if available.
    #[must_use]
    pub fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Returns the number of rows scanned, if available.
    #[must_use]
    pub fn rows_scanned(&self) -> Option<u64> {
        self.rows_scanned
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        if let Some(cols) = &self.int64_cols {
            return cols.first().map_or(0, Vec::len);
        }
        self.rows.len()
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.row_count() == 0
    }

    /// Dense Int64 column, if results were collected without per-row `Vec`s.
    #[must_use]
    pub fn int64_column(&self, idx: usize) -> Option<&[i64]> {
        self.int64_cols.as_ref()?.get(idx).map(Vec::as_slice)
    }

    /// Takes columnar Int64 storage (leaves [`rows`](Self::rows) empty).
    pub fn take_int64_cols(&mut self) -> Option<OwnedInt64Columns> {
        let cols = self.int64_cols.take();
        if cols.is_some() {
            self.cached_int64_rows.take();
        }
        cols.map(|columns| OwnedInt64Columns::new(columns, self.result_reservation.take()))
    }

    /// True when every column is a dense Int64 vector (no entity maps).
    #[must_use]
    pub fn is_int64_columnar(&self) -> bool {
        self.int64_cols.is_some()
    }

    /// Extracts a single value from the result.
    ///
    /// Use this when your query returns exactly one row with one column,
    /// like `RETURN count(n)` or `RETURN sum(p.amount)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the result has multiple rows or columns.
    pub fn scalar<T: FromValue>(&self) -> Result<T> {
        if self.row_count() != 1 || self.columns.len() != 1 {
            return Err(grafeo_common::utils::error::Error::InvalidValue(
                "Expected single value".to_string(),
            ));
        }
        T::from_value(&self.rows()[0][0])
    }

    /// Returns a slice of all result rows.
    #[must_use]
    pub fn rows(&self) -> &[Vec<grafeo_common::types::Value>] {
        if let Some(cols) = &self.int64_cols {
            return self
                .cached_int64_rows
                .get_or_init(|| materialize_int64_rows(cols));
        }
        &self.rows
    }

    /// Takes ownership of all result rows, retaining their memory reservation.
    ///
    /// # Errors
    /// Returns resource exhaustion if the admitted dense row view cannot be allocated.
    pub fn into_rows(self) -> Result<OwnedRows> {
        let reservation = self.result_reservation.clone();
        if let Some(cached) = self.cached_int64_rows.into_inner() {
            return Ok(OwnedRows::new(cached, reservation));
        }
        if let Some(cols) = self.int64_cols {
            return Ok(OwnedRows::new(
                try_materialize_int64_rows(&cols)?,
                reservation,
            ));
        }
        Ok(OwnedRows::new(self.rows, reservation))
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<grafeo_common::types::Value>> {
        self.rows().iter()
    }

    /// Converts this query result to an Arrow [`RecordBatch`](arrow_array::RecordBatch).
    ///
    /// Each column in the result becomes an Arrow array. Type mapping:
    /// - `Int64` / `Float64` / `Bool` / `String` / `Bytes`: direct Arrow equivalents
    /// - `Timestamp` / `ZonedDatetime`: `Timestamp(Microsecond, UTC)`
    /// - `Date`: `Date32`, `Time`: `Time64(Nanosecond)`
    /// - `Vector`: `FixedSizeList(Float32, dim)`
    /// - `Duration` / `List` / `Map` / `Path`: serialized as `Utf8`
    ///
    /// Heterogeneous columns (mixed types) fall back to `Utf8`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`](arrow::ArrowExportError) if Arrow array construction fails.
    #[cfg(feature = "arrow-export")]
    pub fn to_record_batch(
        &self,
    ) -> std::result::Result<arrow_array::RecordBatch, arrow::ArrowExportError> {
        arrow::query_result_to_record_batch(&self.columns, &self.column_types, self.rows())
    }

    /// Serializes this query result as Arrow IPC stream bytes.
    ///
    /// The returned bytes can be read by any Arrow implementation:
    /// - Python: `pyarrow.ipc.open_stream(buf).read_all()`
    /// - Polars: `pl.read_ipc(buf)`
    /// - Node.js: `apache-arrow` `RecordBatchStreamReader`
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`](arrow::ArrowExportError) on conversion or serialization failure.
    #[cfg(feature = "arrow-export")]
    pub fn to_arrow_ipc(&self) -> std::result::Result<Vec<u8>, arrow::ArrowExportError> {
        let batch = self.to_record_batch()?;
        arrow::record_batch_to_ipc_stream(&batch)
    }
}

fn try_materialize_int64_rows(
    columns: &[Vec<i64>],
) -> Result<Vec<Vec<grafeo_common::types::Value>>> {
    let exhausted = |error: std::collections::TryReserveError| {
        Error::Storage(grafeo_common::utils::error::StorageError::Full)
            .with_context(error.to_string())
    };
    let count = columns.first().map_or(0, Vec::len);
    let mut rows = Vec::new();
    rows.try_reserve_exact(count).map_err(exhausted)?;
    for index in 0..count {
        let mut row = Vec::new();
        row.try_reserve_exact(columns.len()).map_err(exhausted)?;
        row.extend(
            columns
                .iter()
                .map(|column| grafeo_common::types::Value::Int64(column[index])),
        );
        rows.push(row);
    }
    Ok(rows)
}

fn materialize_int64_rows(cols: &[Vec<i64>]) -> Vec<Vec<grafeo_common::types::Value>> {
    let n = cols.first().map_or(0, Vec::len);
    let width = cols.len();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(width);
        for col in cols {
            row.push(grafeo_common::types::Value::Int64(col[i]));
        }
        rows.push(row);
    }
    rows
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let table = grafeo_common::fmt::format_result_table(
            &self.columns,
            self.rows(),
            self.execution_time_ms,
            self.status_message.as_deref(),
        );
        f.write_str(&table)
    }
}

/// Converts a [`grafeo_common::types::Value`] to a concrete Rust type.
///
/// Implemented for common types like `i64`, `f64`, `String`, and `bool`.
/// Used by [`QueryResult::scalar()`] to extract typed values.
pub trait FromValue: Sized {
    /// Attempts the conversion, returning an error on type mismatch.
    ///
    /// # Errors
    ///
    /// Returns `Error::TypeMismatch` if the value is not the expected type.
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_int64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "INT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for f64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_float64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "FLOAT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for String {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value.as_str().map(String::from).ok_or_else(|| {
            grafeo_common::utils::error::Error::TypeMismatch {
                expected: "STRING".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for bool {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "BOOL".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn flat_exact_edge_replay_remains_idempotent() -> Result<()> {
        use grafeo_common::types::EdgeId;

        let store = Arc::new(LpgStore::new()?);
        let source = store.create_node(&["Source"]);
        let destination = store.create_node(&["Destination"]);
        let target = LpgRecoveryTarget::Flat(Arc::clone(&store));
        let edge = EdgeId::new(77);
        target.recover_create_edge_with_id(edge, source, destination, "EXACT")?;
        store.set_edge_property(edge, "preserved", grafeo_common::types::Value::Int64(9));
        let before_next = store.next_edge_id();
        let history = || {
            store
                .get_edge_history(edge)
                .into_iter()
                .map(|(created, deleted, row)| (created, deleted, row.src, row.dst, row.edge_type))
                .collect::<Vec<_>>()
        };
        let before_history = history();
        let before_properties = store.edge_property_history(edge);
        let before_types = store.all_edge_types();

        target.recover_create_edge_with_id(edge, source, destination, "EXACT")?;
        assert!(
            target
                .recover_create_edge_with_id(edge, destination, source, "EXACT")
                .is_err()
        );
        assert!(
            target
                .recover_create_edge_with_id(edge, source, destination, "CONFLICT")
                .is_err()
        );
        assert_eq!(store.node_count(), 2);
        assert_eq!(store.edge_count(), 1);
        assert_eq!(store.next_edge_id(), before_next);
        assert_eq!(history(), before_history);
        assert_eq!(store.edge_property_history(edge), before_properties);
        assert_eq!(store.all_edge_types(), before_types);
        assert!(LpgRecoveryTarget::flat_edge_matches(
            &store,
            edge,
            source,
            destination,
            "EXACT"
        ));
        Ok(())
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "grafeo-file"))]
    fn poison_at_publication_cut(operation: &str) {
        use grafeo_storage::wal::WalRecovery;
        use std::sync::mpsc;
        use std::time::Duration;

        fn wal_image(path: &std::path::Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
            let mut image: Vec<_> = std::fs::read_dir(path)
                .unwrap()
                .map(|entry| {
                    let entry = entry.unwrap();
                    (
                        entry.file_name().into(),
                        std::fs::read(entry.path()).unwrap(),
                    )
                })
                .collect();
            image.sort();
            image
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("publication.grafeo");
        let destination = dir.path().join("destination.grafeo");
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        let mut session = db.session();
        assert!(
            db.create_node(&["DurableBeforeFailedPublication"])
                .is_valid()
        );
        db.wal().unwrap().sync().unwrap();
        let wal_path = db.wal.as_ref().unwrap().dir().to_owned();
        let before_c = std::fs::read(&path).unwrap();
        let before_w = wal_image(&wal_path);

        // This is the real Session publication authority. Active count is
        // already zero, but its last fallible publication can still poison.
        let publication = db.transaction_manager.publication().write();
        assert!(!db.is_durability_poisoned());
        let (arrived, arrival) = mpsc::channel();
        let worker_db = Arc::clone(&db);
        let worker_destination = destination.clone();
        let operation = operation.to_owned();
        let worker_operation = operation.clone();
        let worker = std::thread::spawn(move || {
            PUBLICATION_CUT_RENDEZVOUS.with(|point| *point.borrow_mut() = Some(arrived));
            match worker_operation.as_str() {
                "close" => worker_db.close(),
                "save" => worker_db.save(&worker_destination),
                "backup" => worker_db.backup_full(&worker_destination).map(|_| ()),
                _ => unreachable!(),
            }
        });
        arrival.recv_timeout(Duration::from_secs(10)).unwrap();
        // Arrival is after the initial health check with lifecycle held, and
        // this guard excludes both actual publication readers and writers.
        assert!(db.is_open.try_read().is_none());
        assert!(db.transaction_manager.publication().try_read().is_none());
        assert!(db.transaction_manager.publication().try_write().is_none());
        assert!(!db.wal.as_ref().unwrap().is_poisoned());
        db.durability_poisoned
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(!db.wal.as_ref().unwrap().is_poisoned());
        drop(publication);
        let result = worker.join().unwrap();
        assert!(
            matches!(
                result,
                Err(Error::Transaction(TransactionError::DurabilityFailure(_)))
            ),
            "{operation} missed engine-only poison published during its wait: {result:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), before_c);
        assert_eq!(wal_image(&wal_path), before_w);
        assert!(!destination.exists());
        if operation == "close" {
            assert!(!*db.is_open.read());
            assert!(db.checkpoint_timer.lock().is_none());
            assert!(db.wal().unwrap().sync().is_err());
            assert!(db.backup_cursor().is_err());
            assert!(crate::admin::AdminService::wal_status(db.as_ref()).is_err());
            assert!(session.begin_transaction().is_err());
            assert!(db.close().is_err());
        } else {
            assert!(*db.is_open.read());
            assert!(!db.wal.as_ref().unwrap().is_poisoned());
            assert!(db.close().is_err());
        }
        assert!(
            matches!(WalRecovery::new(&wal_path), Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock)
        );
        drop(db);
        assert!(
            matches!(WalRecovery::new(&wal_path), Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock)
        );
        assert_eq!(wal_image(&wal_path), before_w);
        drop(session);
        let _released = WalRecovery::new(&wal_path).unwrap();
        assert_eq!(wal_image(&wal_path), before_w);
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "grafeo-file"))]
    #[test]
    fn close_rechecks_poison_at_publication_cut() {
        poison_at_publication_cut("close");
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "grafeo-file"))]
    #[test]
    fn save_rechecks_poison_at_publication_cut() {
        poison_at_publication_cut("save");
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "grafeo-file"))]
    #[test]
    fn backup_rechecks_poison_at_publication_cut() {
        poison_at_publication_cut("backup");
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "grafeo-file"))]
    #[test]
    fn poisoned_close_drains_without_checkpoint_and_keeps_retained_session_w() {
        use grafeo_storage::wal::WalRecovery;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("poisoned.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        db.create_node(&["Uncheckpointed"]);
        let container_before = std::fs::read(&path).unwrap();
        let wal = db.wal.as_ref().unwrap();
        let wal_path = wal.dir().to_owned();
        std::fs::create_dir(wal_path.join("backup_cursor.meta.tmp")).unwrap();
        let mut capture = wal.capture().unwrap();
        assert!(capture.write_backup_cursor_bytes(b"failure").is_err());
        assert!(capture.write_backup_cursor_bytes(b"cannot retry").is_err());
        drop(capture);
        assert!(db.is_durability_poisoned());
        let error = db.close().unwrap_err();
        assert!(matches!(
            error,
            Error::Transaction(TransactionError::DurabilityFailure(_))
        ));
        assert_eq!(std::fs::read(&path).unwrap(), container_before);
        assert!(db.checkpoint_timer.lock().is_none());
        assert!(wal.sync().is_err());
        assert!(WalRecovery::new(&wal_path).is_err());
        drop(db);
        assert!(WalRecovery::new(&wal_path).is_err());
        drop(session);
        let _released = WalRecovery::new(&wal_path).unwrap();
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn successful_directory_close_retires_retained_session_and_admin_handles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("directory");
        let db = GrafeoDB::open(&path).unwrap();
        let mut session = db.session();
        db.create_node(&["BeforeClose"]);
        let control = db.wal().unwrap();
        db.close().unwrap();
        db.close().unwrap();
        assert!(control.sync().is_err());
        assert!(control.rotate().is_err());
        assert!(control.size_bytes().is_err());
        assert!(crate::admin::AdminService::wal_status(&db).is_err());
        assert!(session.begin_transaction().is_err());
        let reopened = GrafeoDB::open(&path).unwrap();
        assert_eq!(reopened.node_count(), 1);
        reopened.close().unwrap();
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn configured_directory_startup_preserves_exclusive_recovery_owner() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("directory");
        let wal_path = path.join("wal");
        let recovery = WalRecovery::new(&wal_path)?;
        let before = std::fs::read_dir(&path)?.count();
        let config = || {
            Config::persistent(&path)
                .with_storage_format(crate::config::StorageFormat::WalDirectory)
        };
        assert!(matches!(
            GrafeoDB::with_config(config()),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::WouldBlock
        ));
        assert_eq!(std::fs::read_dir(&path)?.count(), before);
        assert!(WalRecovery::new(&wal_path).is_err());
        drop(recovery);
        let db = GrafeoDB::with_config(config())?;
        db.create_node(&["AfterRecoveryRelease"]);
        db.close()?;
        let reopened = GrafeoDB::with_config(config())?;
        assert_eq!(reopened.node_count(), 1);
        reopened.close()?;
        Ok(())
    }

    #[cfg(feature = "wal")]
    #[test]
    fn wal_identity_rejects_unattributed_nonempty_wal_without_mutation() {
        let initial = fresh_world_identity().unwrap();
        let world_identity = Arc::new(RwLock::new(initial.clone()));
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::with_config_and_store_id(
            grafeo_core::graph::rdf::RdfStoreConfig::default(),
            initial.store_id(),
        ));
        let mut config = supported_test_config(Config::in_memory());
        let result = resolve_wal_authority(
            &config,
            &[WalRecord::GraphModelMeta {
                model: config.graph_model.as_u8(),
            }],
            true,
            None,
        )
        .and_then(|authority| {
            GrafeoDB::adopt_wal_authority(
                &mut config,
                &world_identity,
                #[cfg(feature = "triple-store")]
                &rdf_store,
                authority,
            )
        });
        assert!(
            result.is_err(),
            "nonempty WAL needs an authenticated identity"
        );
        assert_eq!(*world_identity.read(), initial);
        #[cfg(feature = "triple-store")]
        assert_eq!(rdf_store.history_completeness(), initial.history());
    }

    #[cfg(feature = "wal")]
    #[test]
    fn wal_identity_rejects_identity_without_durable_model_without_mutation() {
        let initial = fresh_world_identity().unwrap();
        let world_identity = Arc::new(RwLock::new(initial.clone()));
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::with_config_and_store_id(
            grafeo_core::graph::rdf::RdfStoreConfig::default(),
            initial.store_id(),
        ));
        let mut config = supported_test_config(Config::in_memory());
        let original_model = config.graph_model;
        let result = resolve_wal_authority(
            &config,
            &[WalRecord::StoreIdentityMeta {
                metadata: fresh_world_identity().unwrap(),
            }],
            true,
            None,
        )
        .and_then(|authority| {
            GrafeoDB::adopt_wal_authority(
                &mut config,
                &world_identity,
                #[cfg(feature = "triple-store")]
                &rdf_store,
                authority,
            )
        });
        assert!(
            result.is_err(),
            "prior-state WAL needs its own durable model declaration"
        );
        assert_eq!(*world_identity.read(), initial);
        assert_eq!(config.graph_model, original_model);
        #[cfg(feature = "triple-store")]
        assert_eq!(rdf_store.store_id(), initial.store_id());
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "triple-store"))]
    #[test]
    fn wal_identity_rejects_conflicting_model_declarations_without_mutation() {
        let mut config = Config::in_memory();
        let original = config.graph_model;
        let result = GrafeoDB::adopt_stored_graph_model(
            &mut config,
            &[
                WalRecord::GraphModelMeta { model: 0 },
                WalRecord::GraphModelMeta { model: 1 },
            ],
        );
        assert!(result.is_err(), "all WAL model declarations must agree");
        assert_eq!(config.graph_model, original);
    }

    #[cfg(feature = "wal")]
    #[test]
    fn wal_identity_resolution_requires_prior_authority_but_allows_fresh_empty() {
        assert!(resolve_wal_identity(&[], false, None).unwrap().is_none());
        assert!(resolve_wal_identity(&[], true, None).is_err());
        assert!(
            resolve_wal_identity(&[WalRecord::GraphModelMeta { model: 0 }], false, None).is_err()
        );
    }

    #[cfg(feature = "wal")]
    #[test]
    fn wal_identity_resolution_accepts_identical_duplicates_and_authenticated_tail() {
        let identity = fresh_world_identity().unwrap();
        let proof = AuthenticatedContainerIdentity {
            identity: identity.clone(),
            graph_model: crate::config::GraphModel::Rdf,
        };
        let declarations = [
            WalRecord::StoreIdentityMeta {
                metadata: identity.clone(),
            },
            WalRecord::StoreIdentityMeta {
                metadata: identity.clone(),
            },
        ];
        assert_eq!(
            resolve_wal_identity(&declarations, true, None).unwrap(),
            Some(identity.clone())
        );
        assert_eq!(
            resolve_wal_identity(&declarations, true, Some(&proof)).unwrap(),
            Some(identity.clone())
        );
        assert_eq!(
            resolve_wal_identity(&[], true, Some(&proof)).unwrap(),
            Some(identity)
        );
    }

    #[cfg(feature = "wal")]
    #[test]
    fn wal_identity_resolution_rejects_checkpoint_and_history_disagreement() {
        let identity = fresh_world_identity().unwrap();
        let proof = AuthenticatedContainerIdentity {
            identity: identity.clone(),
            graph_model: crate::config::GraphModel::Rdf,
        };
        let foreign = fresh_world_identity().unwrap();
        let altered_history = WorldIdentityMetadataV1::new(
            identity.store_id(),
            HistoryCompleteness::LegacyCurrentState {
                observed_at: EpochId::new(2),
                source_version: 1,
            },
        )
        .unwrap();
        for conflicting in [foreign, altered_history] {
            let declaration = WalRecord::StoreIdentityMeta {
                metadata: conflicting,
            };
            assert!(
                resolve_wal_identity(std::slice::from_ref(&declaration), true, Some(&proof))
                    .is_err()
            );
            assert!(
                resolve_wal_identity(
                    &[
                        WalRecord::StoreIdentityMeta {
                            metadata: identity.clone()
                        },
                        declaration
                    ],
                    true,
                    None
                )
                .is_err()
            );
        }
    }

    #[cfg(feature = "wal")]
    #[test]
    fn wal_identity_resolution_preserves_identical_model_duplicates() {
        let config = supported_test_config(Config::in_memory());
        let identity = fresh_world_identity().unwrap();
        let proof = AuthenticatedContainerIdentity {
            identity: identity.clone(),
            graph_model: config.graph_model,
        };
        let records = [
            WalRecord::GraphModelMeta {
                model: config.graph_model.as_u8(),
            },
            WalRecord::GraphModelMeta {
                model: config.graph_model.as_u8(),
            },
        ];
        let resolved = resolve_wal_authority(&config, &records, true, Some(&proof)).unwrap();
        assert_eq!(resolved.identity, Some(identity));
        assert_eq!(resolved.graph_model, config.graph_model);
    }

    fn supported_test_config(config: Config) -> Config {
        #[cfg(all(not(feature = "lpg"), feature = "triple-store"))]
        {
            config.with_graph_model(crate::config::GraphModel::Rdf)
        }
        #[cfg(not(all(not(feature = "lpg"), feature = "triple-store")))]
        {
            config
        }
    }

    #[cfg(all(feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn section_load_outcome_requires_current_catalog_seal() {
        for sealed in [false, true] {
            let result = SectionLoadOutcome::for_image(
                Some(catalog_wire::CatalogPayloadVersion::GraphExactV7),
                sealed,
                sealed,
                #[cfg(all(feature = "triple-store", feature = "ring-index"))]
                false,
            );
            assert_eq!(result.is_ok(), sealed);
        }
    }

    #[cfg(all(
        feature = "grafeo-file",
        feature = "lpg",
        any(feature = "text-index", feature = "vector-index")
    ))]
    #[test]
    fn exact_index_recovery_requires_current_version_and_image_seal() {
        for (kind, current) in [("Text Index", 5), ("Vector Store", 4)] {
            assert!(validate_exact_index_recovery_section(false, None, kind, current).is_ok());
            assert!(
                validate_exact_index_recovery_section(true, Some(current), kind, current).is_ok()
            );
            for version in (0..=u8::MAX).filter(|version| *version != current) {
                let error =
                    validate_exact_index_recovery_section(true, Some(version), kind, current)
                        .expect_err("a predecessor or unknown section must fail closed");
                assert!(error.to_string().contains("unsupported"), "{error}");
                assert!(error.to_string().contains(kind), "{error}");
            }
            let error = validate_exact_index_recovery_section(false, Some(current), kind, current)
                .expect_err("exact indexes require an authenticated image seal");
            assert!(error.to_string().contains("recovery image seal"), "{error}");
        }
    }

    #[cfg(any(feature = "lpg", feature = "triple-store"))]
    #[test]
    fn quiescent_capture_blocks_transaction_begin_until_release() {
        use std::sync::{Barrier, mpsc};
        use std::time::Duration;

        let db = GrafeoDB::with_config(supported_test_config(Config::in_memory())).unwrap();
        let mut session = db.session();
        let capture = db
            .acquire_quiescent_capture("test quiescent capture")
            .unwrap();
        assert!(
            db.is_open.try_read().is_none(),
            "capture must retain the lifecycle write lock for its full lifetime"
        );
        let start = Arc::new(Barrier::new(2));
        let worker_start = Arc::clone(&start);
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(0);
        let (completed_tx, completed_rx) = mpsc::sync_channel(0);

        let worker = std::thread::spawn(move || {
            worker_start.wait();
            attempting_tx.send(()).unwrap();
            let result = session.begin_transaction();
            completed_tx.send(result).unwrap();
        });

        start.wait();
        attempting_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("transaction-begin worker did not reach the lifecycle boundary");
        assert!(
            matches!(
                completed_rx.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "transaction begin crossed a held lifecycle write capture"
        );
        assert_eq!(db.transaction_manager.active_count(), 0);

        drop(capture);
        completed_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("transaction begin remained blocked after capture release")
            .unwrap();
        worker.join().unwrap();
    }

    #[cfg(all(feature = "wal", not(feature = "lpg"), feature = "triple-store"))]
    #[test]
    fn unsupported_unpinned_wal_model_is_reported_as_storage_corruption() {
        use grafeo_common::utils::error::{ErrorCode, StorageError};

        let mut config = Config::in_memory();
        config.graph_model = crate::config::GraphModel::Rdf;
        let error = GrafeoDB::adopt_stored_graph_model(
            &mut config,
            &[WalRecord::GraphModelMeta {
                model: crate::config::GraphModel::Lpg.as_u8(),
            }],
        )
        .unwrap_err();

        assert_eq!(error.error_code(), ErrorCode::StorageCorrupted);
        assert!(matches!(error, Error::Storage(StorageError::Corruption(_))));
    }

    #[cfg(all(feature = "wal", not(feature = "lpg"), feature = "triple-store"))]
    #[test]
    fn user_pinned_wal_model_mismatch_is_invalid_input() {
        use grafeo_common::utils::error::ErrorCode;

        let mut config = Config::in_memory().with_graph_model(crate::config::GraphModel::Rdf);
        let error = GrafeoDB::adopt_stored_graph_model(
            &mut config,
            &[WalRecord::GraphModelMeta {
                model: crate::config::GraphModel::Lpg.as_u8(),
            }],
        )
        .unwrap_err();

        assert_eq!(error.error_code(), ErrorCode::InvalidInput);
        assert!(matches!(error, Error::InvalidValue(_)));
    }

    #[cfg(feature = "wal")]
    #[test]
    fn snapshot_replay_floor_discards_only_covered_transaction_groups() {
        let covered = TransactionId::new(11);
        let after = TransactionId::new(12);
        let records = vec![
            WalRecord::InsertRdfQuadV3 {
                subject: "<s1>".into(),
                predicate: "<p>".into(),
                object: "<o1>".into(),
                graph: None,
                graph_incarnation: grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id: covered,
            },
            WalRecord::Committed {
                transaction_id: covered,
                epoch: EpochId::new(5),
            },
            WalRecord::GraphModelMeta { model: 2 },
            WalRecord::InsertRdfQuadV3 {
                subject: "<s2>".into(),
                predicate: "<p>".into(),
                object: "<o2>".into(),
                graph: None,
                graph_incarnation: grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id: after,
            },
            WalRecord::Committed {
                transaction_id: after,
                epoch: EpochId::new(6),
            },
        ];

        let filtered =
            wal_records_after_snapshot(records, EpochId::new(5), covered).expect("valid floor");
        assert_eq!(filtered.len(), 3);
        assert!(matches!(
            filtered[0],
            WalRecord::GraphModelMeta { model: 2 }
        ));
        assert!(matches!(
            filtered[1],
            WalRecord::InsertRdfQuadV3 { transaction_id, .. } if transaction_id == after
        ));
        assert!(matches!(
            filtered[2],
            WalRecord::Committed { transaction_id, epoch }
                if transaction_id == after && epoch == EpochId::new(6)
        ));
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn snapshot_replay_floor_handles_tagged_lpg_group_with_legacy_commit() {
        let transaction_id = TransactionId::new(9);
        let records = vec![
            WalRecord::lpg(
                transaction_id,
                grafeo_common::types::GraphPath::root(),
                LpgMutationOp::CreateNode {
                    id: grafeo_common::types::NodeId::new(1),
                    labels: vec!["Legacy".into()],
                },
            ),
            WalRecord::TransactionCommit { transaction_id },
            WalRecord::EpochAdvance {
                epoch: EpochId::new(4),
            },
        ];

        let filtered = wal_records_after_snapshot(records, EpochId::new(4), transaction_id)
            .expect("authenticated legacy transaction floor");
        assert!(
            filtered
                .iter()
                .all(|record| !matches!(record, WalRecord::LpgMutation { .. })),
            "the snapshot-covered legacy mutation must not be applied twice"
        );
    }

    #[cfg(feature = "wal")]
    #[test]
    fn unsealed_snapshot_rejects_epoch_less_committed_wal_group() {
        let transaction_id = TransactionId::new(9);
        let records = vec![
            WalRecord::InsertRdfQuadV3 {
                subject: "<s>".into(),
                predicate: "<p>".into(),
                object: "<o>".into(),
                graph: None,
                graph_incarnation: grafeo_common::types::GraphIncarnationId::DEFAULT_GRAPH,
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id,
            },
            WalRecord::TransactionCommit { transaction_id },
        ];

        let error = wal_records_after_snapshot(records, EpochId::new(4), TransactionId::INVALID)
            .expect_err("an unsealed snapshot cannot classify an epoch-less commit");
        assert!(
            error
                .to_string()
                .contains("no authenticated transaction-id coordinate"),
            "{error}"
        );
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn epoch_zero_snapshot_retains_epoch_less_committed_wal_group() {
        let transaction_id = TransactionId::new(9);
        let records = vec![
            WalRecord::lpg(
                transaction_id,
                grafeo_common::types::GraphPath::root(),
                LpgMutationOp::CreateNode {
                    id: grafeo_common::types::NodeId::new(1),
                    labels: vec!["Legacy".into()],
                },
            ),
            WalRecord::TransactionCommit { transaction_id },
        ];

        let filtered =
            wal_records_after_snapshot(records.clone(), EpochId::INITIAL, TransactionId::INVALID)
                .expect("epoch zero precedes every committed WAL transaction");
        assert_eq!(filtered.len(), records.len());
        assert!(matches!(
            &filtered[0],
            WalRecord::LpgMutation { op: LpgMutationOp::CreateNode { id, labels }, .. }
                if *id == grafeo_common::types::NodeId::new(1)
                    && labels.iter().map(String::as_str).eq(["Legacy"])
        ));
        assert!(matches!(
            filtered[1],
            WalRecord::TransactionCommit { transaction_id: retained }
                if retained == transaction_id
        ));
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn legacy_snapshot_floor_uses_payload_epoch_but_never_raw_header_transaction() {
        let mut header = grafeo_storage::file::DbHeader::EMPTY;
        header.iteration = 1;
        header.epoch = 7;
        header.transaction_id = 41;

        let legacy = snapshot_replay_floor_from_decoded_image(&header, EpochId::new(7), false)
            .expect("decoded legacy epoch agrees with its container");
        assert_eq!(legacy.epoch, EpochId::new(7));
        assert_eq!(legacy.transaction_id, TransactionId::INVALID);

        let sealed = snapshot_replay_floor_from_decoded_image(&header, EpochId::new(7), true)
            .expect("sealed coordinates agree");
        assert_eq!(sealed.transaction_id, TransactionId::new(41));

        let allocation_floor = validated_transaction_allocation_floor(
            header.transaction_id,
            "legacy container header",
        )
        .expect("legacy header remains a bounded allocator hint");
        assert_eq!(allocation_floor, Some(TransactionId::new(41)));
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn unknown_snapshot_boundary_is_not_flattened_into_no_snapshot() {
        let evidence = SnapshotReplayEvidence {
            authenticated_epoch: None,
            recovery_coordinates_are_sealed: false,
        };
        let mut header = grafeo_storage::file::DbHeader::EMPTY;
        header.iteration = 1;

        let error = evidence
            .resolve(&header)
            .expect_err("an existing legacy image without an epoch must fail closed");
        assert!(
            error.to_string().contains("no authenticated epoch"),
            "{error}"
        );
    }

    #[cfg(any(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn transaction_allocation_floor_rejects_invalid_exhaustion_boundary() {
        for raw in [u64::MAX - 1, u64::MAX] {
            let error = validated_transaction_allocation_floor(raw, "test header")
                .expect_err("allocator must never advance to INVALID");
            assert!(
                error.to_string().contains("safe allocator limit"),
                "{error}"
            );
        }
    }

    #[cfg(feature = "grafeo-file")]
    #[test]
    fn header_cardinality_is_authoritative_only_with_coordinate_seal() {
        let mut header = grafeo_storage::file::DbHeader::EMPTY;
        header.iteration = 1;
        header.node_count = 3;
        header.edge_count = 2;

        validate_sealed_recovery_cardinality(&header, false, 99, 98, "legacy LPG")
            .expect("unsealed legacy header counts are not an admission gate");
        validate_sealed_recovery_cardinality(&header, true, 3, 2, "default LPG view")
            .expect("sealed counts agree");

        let error = validate_sealed_recovery_cardinality(&header, true, 0, 0, "RDF-only view")
            .expect_err("RDF-only sealed coordinates must be zero");
        assert!(matches!(
            error,
            Error::Storage(grafeo_common::utils::error::StorageError::Corruption(_))
        ));
    }

    #[cfg(feature = "wal")]
    #[test]
    fn snapshot_replay_floor_filters_catalog_batch_v3_by_its_epoch() {
        let batch = |epoch| WalRecord::CatalogBatchV3 {
            created_graph_incarnations: vec![],
            dropped_graph_incarnations: vec![],
            version: 2,
            epoch: EpochId::new(epoch),
            catalog_state: vec![u8::try_from(epoch).expect("test epoch must fit in u8")],
            created_graphs: Vec::new(),
            dropped_graphs: Vec::new(),
        };
        let records = vec![batch(4), batch(5), batch(6)];

        let filtered = wal_records_after_snapshot(records, EpochId::new(5), TransactionId::INVALID)
            .expect("standalone records carry their own epochs");
        assert!(matches!(
            filtered.as_slice(),
            [WalRecord::CatalogBatchV3 { epoch, .. }] if *epoch == EpochId::new(6)
        ));
    }

    #[cfg(feature = "wal")]
    #[test]
    fn snapshot_replay_floor_keeps_projection_v3_receipt_with_its_transaction() {
        let declaration = |epoch| {
            let epoch_byte = u8::try_from(epoch).expect("fixture epoch must fit in one byte");
            WalRecord::RdfLpgProjectionDeclaredV3 {
                projection_id: epoch,
                mapping_digest: grafeo_common::types::Digest256::from_bytes([epoch_byte; 32]),
                mapping_format_version: 2,
                source_graph: None,
                type_iri: format!("http://example.org/Type{epoch}"),
                node_label: format!("Type{epoch}"),
                epoch: EpochId::new(epoch),
            }
        };
        let mut records = Vec::new();
        for epoch in [4, 5, 6] {
            let epoch_byte = u8::try_from(epoch).expect("fixture epoch must fit in one byte");
            let transaction_id = TransactionId::new(epoch);
            records.push(declaration(epoch));
            records.push(WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id,
                receipt: vec![epoch_byte],
            });
            records.push(WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(epoch),
            });
        }

        let filtered = wal_records_after_snapshot(records, EpochId::new(5), TransactionId::new(5))
            .expect("authenticated transaction floor");
        assert_eq!(filtered.len(), 3);
        assert!(matches!(
            filtered[0],
            WalRecord::RdfLpgProjectionDeclaredV3 { epoch, .. }
                if epoch == EpochId::new(6)
        ));
        assert!(matches!(
            filtered[1],
            WalRecord::RdfLpgProjectionPublishedV3 { transaction_id, .. }
                if transaction_id == TransactionId::new(6)
        ));
        assert!(matches!(
            filtered[2],
            WalRecord::Committed { transaction_id, epoch }
                if transaction_id == TransactionId::new(6) && epoch == EpochId::new(6)
        ));
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn legacy_catalog_replay_handles_exact_graph_actions_and_procedure_replace() {
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Catalog::new();
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::new());
        #[cfg(feature = "triple-store")]
        let rdf_projections = Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());
        let records = vec![WalRecord::CatalogBatchV2 {
            version: 1,
            records: vec![
                WalRecord::CreateGraphType {
                    name: "Network".to_string(),
                    node_types: vec!["Device".to_string()],
                    edge_types: Vec::new(),
                    open: false,
                },
                WalRecord::AlterGraphType {
                    name: "Network".to_string(),
                    alterations: vec![("add_edge_type".to_string(), "CONNECTS".to_string())],
                },
                WalRecord::CreateProcedure {
                    name: "answer".to_string(),
                    params: Vec::new(),
                    returns: vec![("value".to_string(), "INTEGER".to_string())],
                    body: "RETURN 1 AS value".to_string(),
                },
                WalRecord::CreateProcedure {
                    name: "answer".to_string(),
                    params: Vec::new(),
                    returns: vec![("value".to_string(), "INTEGER".to_string())],
                    body: "RETURN 2 AS value".to_string(),
                },
            ],
        }];

        GrafeoDB::apply_wal_records(
            &store,
            &catalog,
            #[cfg(feature = "triple-store")]
            &rdf_store,
            #[cfg(feature = "triple-store")]
            &rdf_projections,
            &records,
        )
        .unwrap();
        // Legacy metadata is deliberately retained across checkpoint floors;
        // replaying it again must remain a replacement/idempotent operation.
        GrafeoDB::apply_wal_records(
            &store,
            &catalog,
            #[cfg(feature = "triple-store")]
            &rdf_store,
            #[cfg(feature = "triple-store")]
            &rdf_projections,
            &records,
        )
        .unwrap();

        assert_eq!(
            catalog
                .get_graph_type_def("Network")
                .unwrap()
                .allowed_edge_types,
            ["CONNECTS"]
        );
        assert_eq!(
            catalog.get_procedure("answer").unwrap().body,
            "RETURN 2 AS value"
        );
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn legacy_constraint_drop_without_name_mapping_fails_closed() {
        use crate::catalog::{NamedConstraintDefinition, NamedConstraintKind};

        fn hex(value: &str) -> String {
            use std::fmt::Write as _;

            let mut encoded = String::with_capacity(value.len() * 2);
            for byte in value.as_bytes() {
                write!(&mut encoded, "{byte:02x}").unwrap();
            }
            encoded
        }

        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Catalog::new();
        #[cfg(feature = "triple-store")]
        let rdf_store = Arc::new(RdfStore::new());
        #[cfg(feature = "triple-store")]
        let rdf_projections = Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());

        let label = "Person";
        let property = "email";
        let legacy_name = format!(
            "__grafeo_legacy_{}_{}_1_{}",
            hex(NamedConstraintKind::Unique.as_str()),
            hex(label),
            hex(property)
        );
        catalog
            .create_named_constraint(NamedConstraintDefinition {
                name: legacy_name,
                label: label.to_string(),
                properties: vec![property.to_string()],
                kind: NamedConstraintKind::Unique,
            })
            .unwrap();
        assert!(catalog.has_legacy_constraint_owners());

        let error = GrafeoDB::apply_wal_records(
            &store,
            &catalog,
            #[cfg(feature = "triple-store")]
            &rdf_store,
            #[cfg(feature = "triple-store")]
            &rdf_projections,
            &[WalRecord::DropConstraint {
                name: "original_user_name".to_string(),
            }],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot safely replay DROP CONSTRAINT")
        );
        assert!(
            catalog.is_property_unique(
                catalog.get_label_id(label).unwrap(),
                catalog.get_property_key_id(property).unwrap()
            ),
            "ambiguous recovery must retain enforcement rather than silently claiming the drop"
        );
    }

    #[cfg(all(feature = "wal", feature = "lpg", feature = "triple-store"))]
    #[test]
    fn projection_v3_replay_is_store_bound_committed_and_conflict_detecting() {
        use grafeo_common::types::{ProjectionReconciliationState, ProjectionSourceGraph, StoreId};
        use grafeo_core::graph::rdf::{
            RDF_LPG_PROJECTION_MAPPING_VERSION, RdfLpgProjectionDefinition,
            RdfLpgProjectionReceipt, RdfLpgProjectionRegistry,
        };

        let store = Arc::new(LpgStore::new().unwrap());
        store.create_graph("named").unwrap();
        let catalog = Catalog::new();
        let rdf_store = Arc::new(RdfStore::new());
        let rdf_projections = Arc::new(RdfLpgProjectionRegistry::new());
        let definition = RdfLpgProjectionDefinition::new("http://example.org/Person", "Person");
        let projection_id = definition.id();
        let transaction_id = TransactionId::new(17);
        let receipt = RdfLpgProjectionReceipt::new(
            rdf_store.store_id(),
            definition.mapping_digest(),
            projection_id,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(3),
            EpochId::new(4),
            1,
            0,
            ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        let records = vec![
            WalRecord::RdfLpgProjectionDeclaredV3 {
                projection_id,
                mapping_digest: definition.mapping_digest(),
                mapping_format_version: RDF_LPG_PROJECTION_MAPPING_VERSION,
                source_graph: None,
                type_iri: definition.type_iri().into(),
                node_label: definition.node_label().into(),
                epoch: EpochId::new(2),
            },
            WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id,
                receipt: receipt.encode(),
            },
            WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(4),
            },
        ];

        for _ in 0..2 {
            GrafeoDB::apply_wal_records(&store, &catalog, &rdf_store, &rdf_projections, &records)
                .unwrap();
        }
        let status = rdf_projections.get(projection_id).unwrap();
        assert_eq!(status.receipt(), Some(&receipt));
        assert_eq!(store.current_epoch(), EpochId::new(4));
        assert_eq!(
            store.graph("named").unwrap().current_epoch(),
            EpochId::new(4)
        );
        assert_eq!(rdf_store.commit_epoch(), EpochId::new(4));

        let uncommitted = [WalRecord::RdfLpgProjectionPublishedV3 {
            transaction_id: TransactionId::new(18),
            receipt: receipt.encode(),
        }];
        let error = GrafeoDB::apply_wal_records(
            &store,
            &catalog,
            &rdf_store,
            &rdf_projections,
            &uncommitted,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without a durable commit marker")
        );

        let conflicting = RdfLpgProjectionReceipt::new(
            rdf_store.store_id(),
            definition.mapping_digest(),
            projection_id,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(3),
            EpochId::new(4),
            1,
            1,
            ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        let conflict_transaction = TransactionId::new(19);
        let conflict_records = [
            WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id: conflict_transaction,
                receipt: conflicting.encode(),
            },
            WalRecord::Committed {
                transaction_id: conflict_transaction,
                epoch: EpochId::new(4),
            },
        ];
        let error = GrafeoDB::apply_wal_records(
            &store,
            &catalog,
            &rdf_store,
            &rdf_projections,
            &conflict_records,
        )
        .unwrap_err();
        assert!(error.to_string().contains("conflicting"), "{error}");

        let foreign_store = StoreId::from_bytes([0x5a; StoreId::LEN]).unwrap();
        let foreign = RdfLpgProjectionReceipt::new(
            foreign_store,
            definition.mapping_digest(),
            projection_id,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(4),
            EpochId::new(5),
            2,
            0,
            ProjectionReconciliationState::Reconciled,
        )
        .unwrap();
        let foreign_transaction = TransactionId::new(20);
        let foreign_records = [
            WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id: foreign_transaction,
                receipt: foreign.encode(),
            },
            WalRecord::Committed {
                transaction_id: foreign_transaction,
                epoch: EpochId::new(5),
            },
        ];
        let error = GrafeoDB::apply_wal_records(
            &store,
            &catalog,
            &rdf_store,
            &rdf_projections,
            &foreign_records,
        )
        .unwrap_err();
        assert!(error.to_string().contains("belongs to store"), "{error}");
    }

    #[cfg(all(feature = "lpg", feature = "triple-store"))]
    #[test]
    fn projection_declaration_rejects_malformed_mapping_before_install() {
        let db = GrafeoDB::with_config(
            Config::in_memory().with_graph_model(crate::config::GraphModel::Both),
        )
        .unwrap();

        for (type_iri, node_label) in [("", "Person"), ("http://example.org/Person", "  ")] {
            let error = db
                .declare_rdf_lpg_projection(type_iri, node_label)
                .unwrap_err();
            assert!(matches!(error, Error::InvalidValue(_)));
        }
        assert!(db.rdf_projections.is_empty());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_create_in_memory_database() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_database_config() {
        let config =
            supported_test_config(Config::in_memory().with_threads(4).with_query_logging());

        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.config().threads, 4);
        assert!(db.config().query_logging);
    }

    #[test]
    fn test_database_session() {
        let db = GrafeoDB::new_in_memory();
        let _session = db.session();
        // Session should be created successfully
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn test_persistent_database_recovery() {
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Create database and add some data
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"))
                .expect("set node property");

            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"))
                .expect("set node property");

            let _edge = db.create_edge(alix, gus, "KNOWS");

            // Explicitly close to flush WAL
            db.close().unwrap();
        }

        // Reopen and verify data was recovered
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            assert_eq!(db.node_count(), 2);
            assert_eq!(db.edge_count(), 1);

            // Verify nodes exist
            let node0 = db.get_node(grafeo_common::types::NodeId::new(0));
            assert!(node0.is_some());

            let node1 = db.get_node(grafeo_common::types::NodeId::new(1));
            assert!(node1.is_some());
        }
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn test_wal_logging() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("wal_test_db");

        let db = GrafeoDB::open(&db_path).unwrap();

        // Create some data
        let node = db.create_node(&["Test"]);
        db.delete_node(node);

        // WAL should have records
        if let Some(wal) = db.wal() {
            assert!(wal.record_count() > 0);
        }

        db.close().unwrap();
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn incremental_backup_rejects_an_active_transaction_without_advancing_its_cut() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("active_incremental.grafeo");
        let backup_dir = dir.path().join("backups");
        let db = GrafeoDB::open(&db_path).unwrap();
        assert!(db.create_node(&["Seed"]).is_valid());
        db.backup_full(&backup_dir).unwrap();
        assert!(db.create_node(&["Delta"]).is_valid());

        let cursor_before = db.backup_cursor().unwrap().unwrap();
        let sequence_before = db.wal().unwrap().current_sequence();
        let manifest_len_before = GrafeoDB::read_backup_manifest(&backup_dir)
            .unwrap()
            .unwrap()
            .segments
            .len();
        let mut session = db.session();
        session.begin_transaction().unwrap();

        let error = db.backup_incremental(&backup_dir).unwrap_err();

        assert!(matches!(
            error,
            Error::Transaction(TransactionError::InvalidState(_))
        ));
        assert_eq!(db.wal().unwrap().current_sequence(), sequence_before);
        let cursor_after = db.backup_cursor().unwrap().unwrap();
        assert_eq!(cursor_after.backed_up_epoch, cursor_before.backed_up_epoch);
        assert_eq!(cursor_after.log_sequence, cursor_before.log_sequence);
        assert_eq!(
            GrafeoDB::read_backup_manifest(&backup_dir)
                .unwrap()
                .unwrap()
                .segments
                .len(),
            manifest_len_before
        );
        session.rollback().unwrap();
    }

    #[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
    #[test]
    fn incremental_backup_serializes_a_commit_behind_its_exact_wal_cut() {
        use std::sync::{Arc, mpsc};
        use std::time::{Duration, Instant};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("serialized_incremental.grafeo");
        let backup_dir = dir.path().join("backups");
        let db = Arc::new(GrafeoDB::open(&db_path).unwrap());
        assert!(db.create_node(&["Seed"]).is_valid());
        db.backup_full(&backup_dir).unwrap();
        assert!(db.create_node(&["BeforeCut"]).is_valid());

        // Hold the commit-publication barrier long enough for backup to take
        // the lifecycle write guard and queue at publication. A new writer is
        // then forced behind that lifecycle guard, establishing deterministic
        // backup-before-commit ordering rather than relying on thread timing.
        let publication = db.transaction_manager.publication().write();
        let (backup_tx, backup_rx) = mpsc::channel();
        let backup_db = Arc::clone(&db);
        let backup_path = backup_dir.clone();
        let backup_worker = std::thread::spawn(move || {
            let _ = backup_tx.send(backup_db.backup_incremental(&backup_path));
        });

        let deadline = Instant::now() + Duration::from_secs(10);
        while db.is_open.try_read().is_some() {
            assert!(
                Instant::now() < deadline,
                "incremental backup did not acquire its lifecycle cut"
            );
            std::thread::yield_now();
        }

        let (writer_attempt_tx, writer_attempt_rx) = mpsc::sync_channel(0);
        let (writer_tx, writer_rx) = mpsc::channel();
        let writer_db = Arc::clone(&db);
        let writer = std::thread::spawn(move || {
            writer_attempt_tx.send(()).unwrap();
            let _ = writer_tx.send(writer_db.create_node(&["AfterCut"]));
        });
        writer_attempt_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("post-cut writer did not begin its attempt");
        assert!(matches!(
            backup_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(matches!(
            writer_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));

        drop(publication);
        let first = backup_rx
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        backup_worker.join().unwrap();
        assert!(
            writer_rx
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .is_valid()
        );
        writer.join().unwrap();

        let second = db.backup_incremental(&backup_dir).unwrap();
        assert!(second.start_epoch > first.end_epoch);

        let first_restore = dir.path().join("first_cut.grafeo");
        GrafeoDB::restore_to_epoch(&backup_dir, first.end_epoch, &first_restore).unwrap();
        let first_db = GrafeoDB::open(&first_restore).unwrap();
        assert_eq!(first_db.node_count(), 2, "post-cut commit leaked backward");
        first_db.close().unwrap();

        let second_restore = dir.path().join("second_cut.grafeo");
        GrafeoDB::restore_to_epoch(&backup_dir, second.end_epoch, &second_restore).unwrap();
        let second_db = GrafeoDB::open(&second_restore).unwrap();
        assert_eq!(second_db.node_count(), 3, "post-cut commit was skipped");
        second_db.close().unwrap();
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn test_wal_recovery_multiple_sessions() {
        // Tests that WAL recovery works correctly across multiple open/close cycles
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("multi_session_db");

        // Session 1: Create initial data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"))
                .expect("set node property");
            db.close().unwrap();
        }

        // Session 2: Add more data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 1); // Previous data recovered
            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"))
                .expect("set node property");
            db.close().unwrap();
        }

        // Session 3: Verify all data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 2);

            // Verify properties were recovered correctly
            let node0 = db.get_node(grafeo_common::types::NodeId::new(0)).unwrap();
            assert!(node0.labels.iter().any(|l| l.as_str() == "Person"));

            let node1 = db.get_node(grafeo_common::types::NodeId::new(1)).unwrap();
            assert!(node1.labels.iter().any(|l| l.as_str() == "Person"));
        }
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn test_database_consistency_after_mutations() {
        // Tests that database remains consistent after a series of create/delete operations
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("consistency_db");

        {
            let db = GrafeoDB::open(&db_path).unwrap();

            // Create nodes
            let a = db.create_node(&["Node"]);
            let b = db.create_node(&["Node"]);
            let c = db.create_node(&["Node"]);

            // Create edges
            let e1 = db.create_edge(a, b, "LINKS");
            let _e2 = db.create_edge(b, c, "LINKS");

            // Delete middle node and its edge
            db.delete_edge(e1);
            db.delete_node(b);

            // Set properties on remaining nodes
            db.set_node_property(a, "value", Value::Int64(1))
                .expect("set node property");
            db.set_node_property(c, "value", Value::Int64(3))
                .expect("set node property");

            db.close().unwrap();
        }

        // Reopen and verify consistency
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            // Should have 2 nodes (a and c), b was deleted
            // Note: node_count includes deleted nodes in some implementations
            // What matters is that the non-deleted nodes are accessible
            let node_a = db.get_node(grafeo_common::types::NodeId::new(0));
            assert!(node_a.is_some());

            let node_c = db.get_node(grafeo_common::types::NodeId::new(2));
            assert!(node_c.is_some());

            // Middle node should be deleted
            let node_b = db.get_node(grafeo_common::types::NodeId::new(1));
            assert!(node_b.is_none());
        }
    }

    #[cfg(all(feature = "wal", feature = "lpg"))]
    #[test]
    fn test_close_is_idempotent() {
        // Calling close() multiple times should not cause errors
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("close_test_db");

        let db = GrafeoDB::open(&db_path).unwrap();
        db.create_node(&["Test"]);

        // First close should succeed
        assert!(db.close().is_ok());

        // Second close should also succeed (idempotent)
        assert!(db.close().is_ok());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_with_store_external_backend() {
        use grafeo_core::graph::lpg::LpgStore;

        let external = Arc::new(LpgStore::new().unwrap());

        // Seed data on the external store directly
        let n1 = external.create_node(&["Person"]);
        external.set_node_property(n1, "name", grafeo_common::types::Value::from("Alix"));

        let db = GrafeoDB::with_store(
            Arc::clone(&external) as Arc<dyn GraphStoreMut>,
            Config::in_memory(),
        )
        .unwrap();

        let session = db.session();

        // Session should see data from the external store via execute
        #[cfg(feature = "gql")]
        {
            let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
            assert_eq!(result.rows.len(), 1);
        }
    }

    #[test]
    fn test_with_config_custom_memory_limit() {
        let config = supported_test_config(Config::in_memory().with_memory_limit(64 * 1024 * 1024)); // 64 MB

        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.config().memory_limit, Some(64 * 1024 * 1024));
    }

    #[cfg(all(feature = "metrics", feature = "lpg"))]
    #[test]
    fn test_database_metrics_registry() {
        let db = GrafeoDB::new_in_memory();

        // Perform some operations
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);

        // Check that metrics snapshot returns data
        let snap = db.metrics();
        // Session created counter should reflect at least 0 (metrics is initialized)
        assert_eq!(snap.query_count, 0); // No queries executed yet
    }

    #[cfg(all(feature = "metrics", feature = "lpg", feature = "gql"))]
    #[test]
    fn test_columnar_query_metrics_use_logical_row_count() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);
        db.reset_metrics();

        let result = db
            .execute("MATCH (n:Person) RETURN id(n)")
            .expect("columnar query succeeds");

        assert!(result.is_int64_columnar());
        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows_scanned(), Some(3));

        let snapshot = db.metrics();
        assert_eq!(snapshot.rows_returned, 3);
        assert_eq!(snapshot.rows_scanned, 3);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_query_result_has_metrics() {
        // Verifies that query results include execution metrics
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);

        #[cfg(feature = "gql")]
        {
            let result = db.execute("MATCH (n:Person) RETURN n").unwrap();

            // Metrics should be populated
            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert!(result.execution_time_ms.unwrap() >= 0.0);
            assert_eq!(result.rows_scanned.unwrap(), 2);
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_empty_query_result_metrics() {
        // Verifies metrics are correct for queries returning no results
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);

        #[cfg(feature = "gql")]
        {
            // Query that matches nothing
            let result = db.execute("MATCH (n:NonExistent) RETURN n").unwrap();

            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert_eq!(result.rows_scanned.unwrap(), 0);
        }
    }

    #[cfg(all(feature = "gql", feature = "lpg"))]
    #[test]
    fn oneshot_execute_shares_physical_cache() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        let q = "MATCH (n:Person) RETURN count(n) AS n";
        assert_eq!(db.physical_plan_cache_len(), 0);
        let a = db.execute(q).unwrap();
        assert!(
            db.physical_plan_cache_len() >= 1,
            "first one-shot execute must insert a physical plan"
        );
        let b = db.execute(q).unwrap();
        assert_eq!(a.rows(), b.rows());
        assert_eq!(
            db.physical_plan_cache_len(),
            1,
            "repeat execute reuses the same cache slot"
        );
        db.clear_plan_cache();
        assert_eq!(db.physical_plan_cache_len(), 0);
    }

    #[cfg(all(feature = "gql", feature = "compact-store", feature = "lpg"))]
    #[test]
    fn compact_clears_shared_physical_cache() {
        let mut db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        db.execute("MATCH (n:Person) RETURN count(n) AS n").unwrap();
        assert!(db.physical_plan_cache_len() >= 1);
        db.compact().expect("compact");
        assert_eq!(
            db.physical_plan_cache_len(),
            0,
            "compact must clear the shared physical cache in place"
        );
    }

    #[cfg(all(feature = "cdc", feature = "lpg"))]
    mod cdc_integration {
        use super::*;

        /// Helper: creates an in-memory database with CDC enabled.
        fn cdc_db() -> GrafeoDB {
            GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
        }

        #[test]
        fn test_node_lifecycle_history() {
            let db = cdc_db();

            // Create
            let id = db.create_node(&["Person"]);
            // Update
            db.set_node_property(id, "name", "Alix".into())
                .expect("set node property");
            db.set_node_property(id, "name", "Gus".into())
                .expect("set node property");
            // Delete
            db.delete_node(id);

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 4); // create + 2 updates + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert!(history[1].before.is_none()); // first set_node_property has no prior value
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Update);
            assert!(history[2].before.is_some()); // second update has prior "Alix"
            assert_eq!(history[3].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_edge_lifecycle_history() {
            let db = cdc_db();

            let alix = db.create_node(&["Person"]);
            let gus = db.create_node(&["Person"]);
            let edge = db.create_edge(alix, gus, "KNOWS");
            db.set_edge_property(edge, "since", 2024i64.into())
                .expect("set edge property");
            db.delete_edge(edge);

            let history = db.history(edge).unwrap();
            assert_eq!(history.len(), 3); // create + update + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_create_node_with_props_cdc() {
            let db = cdc_db();

            let id = db.create_node_with_props(
                &["Person"],
                vec![
                    ("name", grafeo_common::types::Value::from("Alix")),
                    ("age", grafeo_common::types::Value::from(30i64)),
                ],
            );

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 1);
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            // Props should be captured
            let after = history[0].after.as_ref().unwrap();
            assert_eq!(after.len(), 2);
        }

        #[test]
        fn test_changes_between() {
            let db = cdc_db();

            let id1 = db.create_node(&["A"]);
            let _id2 = db.create_node(&["B"]);
            db.set_node_property(id1, "x", 1i64.into())
                .expect("set node property");

            // All events should be at the same epoch (in-memory, epoch doesn't advance without tx)
            let changes = db
                .changes_between(
                    grafeo_common::types::EpochId(0),
                    grafeo_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert_eq!(changes.len(), 3); // 2 creates + 1 update
        }

        #[test]
        fn test_cdc_disabled_by_default() {
            let db = GrafeoDB::new_in_memory();
            assert!(!db.is_cdc_enabled());

            let id = db.create_node(&["Person"]);
            db.set_node_property(id, "name", "Alix".into())
                .expect("set node property");

            let history = db.history(id).unwrap();
            assert!(history.is_empty(), "CDC off by default: no events recorded");
        }

        #[test]
        fn test_session_with_cdc_override_on() {
            // Database default is off, but session opts in
            let db = GrafeoDB::new_in_memory();
            let session = db.session_with_cdc(true);
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            // The CDC log should have events from the opted-in session
            let changes = db
                .changes_between(
                    grafeo_common::types::EpochId(0),
                    grafeo_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert!(
                !changes.is_empty(),
                "session_with_cdc(true) should record events"
            );
        }

        #[test]
        fn test_session_with_cdc_override_off() {
            // Database default is on, but session opts out
            let db = cdc_db();
            let session = db.session_with_cdc(false);
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            let changes = db
                .changes_between(
                    grafeo_common::types::EpochId(0),
                    grafeo_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert!(
                changes.is_empty(),
                "session_with_cdc(false) should not record events"
            );
        }

        #[test]
        fn test_set_cdc_enabled_runtime() {
            let db = GrafeoDB::new_in_memory();
            assert!(!db.is_cdc_enabled());

            // Enable at runtime
            db.set_cdc_enabled(true);
            assert!(db.is_cdc_enabled());

            let id = db.create_node(&["Person"]);
            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 1, "CDC enabled at runtime records events");

            // Disable again
            db.set_cdc_enabled(false);
            let id2 = db.create_node(&["Person"]);
            let history2 = db.history(id2).unwrap();
            assert!(
                history2.is_empty(),
                "CDC disabled at runtime stops recording"
            );
        }
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn test_with_store_basic() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Person"]);
        store.set_node_property(n1, "name", "Alix".into());

        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let result = db.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn test_with_store_session() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let session = db.session();
        let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn test_with_store_mutations() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let mut session = db.session();

        // Use an explicit transaction so INSERT and MATCH share the same
        // transaction context. With PENDING epochs, uncommitted versions are
        // only visible to the owning transaction.
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);

        session.commit().unwrap();
    }

    // =========================================================================
    // QueryResult tests
    // =========================================================================

    #[test]
    fn test_query_result_empty() {
        let result = QueryResult::empty();
        assert!(result.is_empty());
        assert_eq!(result.row_count(), 0);
        assert_eq!(result.column_count(), 0);
        assert!(result.execution_time_ms().is_none());
        assert!(result.rows_scanned().is_none());
        assert!(result.status_message.is_none());
    }

    #[test]
    fn test_query_result_status() {
        let result = QueryResult::status("Created node type 'Person'");
        assert!(result.is_empty());
        assert_eq!(result.column_count(), 0);
        assert_eq!(
            result.status_message.as_deref(),
            Some("Created node type 'Person'")
        );
    }

    #[test]
    fn test_query_result_new_with_columns() {
        let result = QueryResult::new(vec!["name".into(), "age".into()]);
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 0);
        assert!(result.is_empty());
        // Column types should default to Any
        assert_eq!(
            result.column_types,
            vec![
                grafeo_common::types::LogicalType::Any,
                grafeo_common::types::LogicalType::Any
            ]
        );
    }

    #[test]
    fn test_query_result_with_types() {
        use grafeo_common::types::LogicalType;
        let result = QueryResult::with_types(
            vec!["name".into(), "age".into()],
            vec![LogicalType::String, LogicalType::Int64],
        );
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column_types[0], LogicalType::String);
        assert_eq!(result.column_types[1], LogicalType::Int64);
    }

    #[test]
    fn test_query_result_with_metrics() {
        let result = QueryResult::new(vec!["x".into()]).with_metrics(42.5, 100);
        assert_eq!(result.execution_time_ms(), Some(42.5));
        assert_eq!(result.rows_scanned(), Some(100));
    }

    #[test]
    fn test_query_result_scalar_success() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["count".into()]);
        result.rows.push(vec![Value::Int64(42)]);

        let val: i64 = result.scalar().unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_query_result_scalar_wrong_shape() {
        use grafeo_common::types::Value;
        // Multiple rows
        let mut result = QueryResult::new(vec!["x".into()]);
        result.rows.push(vec![Value::Int64(1)]);
        result.rows.push(vec![Value::Int64(2)]);
        assert!(result.scalar::<i64>().is_err());

        // Multiple columns
        let mut result2 = QueryResult::new(vec!["a".into(), "b".into()]);
        result2.rows.push(vec![Value::Int64(1), Value::Int64(2)]);
        assert!(result2.scalar::<i64>().is_err());

        // Empty
        let result3 = QueryResult::new(vec!["x".into()]);
        assert!(result3.scalar::<i64>().is_err());
    }

    #[test]
    fn test_query_result_int64_columnar_row_count() {
        let mut result = QueryResult::new(vec!["id(c)".into()]);
        result.int64_cols = Some(vec![vec![1, 2, 3]]);
        assert_eq!(result.row_count(), 3);
        assert!(!result.is_empty());
        assert!(result.is_int64_columnar());
        assert_eq!(result.int64_column(0), Some(&[1, 2, 3][..]));
        assert_eq!(result.rows().len(), 3);
        assert_eq!(result.rows()[1][0], grafeo_common::types::Value::Int64(2));
        assert_eq!(
            result.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![
                grafeo_common::types::Value::Int64(1),
                grafeo_common::types::Value::Int64(2),
                grafeo_common::types::Value::Int64(3),
            ]
        );
    }

    #[test]
    fn test_query_result_push_row_materializes_columnar_storage() {
        use grafeo_common::types::Value;

        let mut result = QueryResult::new(vec!["value".into()]);
        result.int64_cols = Some(vec![vec![1, 2]]);
        assert_eq!(result.rows()[0][0], Value::Int64(1));

        result.push_row(vec![Value::Int64(3)]).unwrap();

        assert!(!result.is_int64_columnar());
        assert_eq!(result.row_count(), 3);
        assert_eq!(
            result.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]
        );
        assert_eq!(result.into_rows().expect("owned rows").len(), 3);
    }

    #[test]
    fn test_query_result_take_columnar_storage_clears_cached_rows() {
        let mut result = QueryResult::new(vec!["value".into()]);
        result.int64_cols = Some(vec![vec![1, 2]]);
        assert_eq!(result.rows().len(), 2);

        let columns = result.take_int64_cols().expect("columnar storage");
        assert_eq!(&*columns, &vec![vec![1, 2]]);
        assert!(result.is_empty());
        assert!(result.rows().is_empty());
        assert!(result.into_rows().expect("empty rows").is_empty());
    }

    #[test]
    fn owned_row_retains_result_grant_after_outer_values_drop() -> Result<()> {
        use grafeo_common::types::Value;
        use grafeo_core::execution::QueryResourceContext;

        let db = GrafeoDB::new_in_memory();
        let before = db.buffer_manager().allocated();
        let resources = QueryResourceContext::new(Arc::clone(db.buffer_manager())).unwrap();
        let grant = resources.try_allocate(4096).unwrap();
        let mut result = QueryResult::new(vec!["value".into()]).with_result_reservation(grant);
        result.rows.push(vec![Value::Int64(7)]);

        let owned = result.into_rows()?;
        let mut rows = owned.into_iter();
        let row = rows.next().expect("owned row");
        drop(rows);
        assert!(db.buffer_manager().allocated() >= before + 4096);
        assert_eq!(&*row, &[Value::Int64(7)]);
        drop(row);
        assert_eq!(db.buffer_manager().allocated(), before);
        Ok(())
    }

    #[test]
    fn owned_dense_column_retains_result_grant_after_outer_values_drop() -> Result<()> {
        use grafeo_core::execution::QueryResourceContext;

        let db = GrafeoDB::new_in_memory();
        let before = db.buffer_manager().allocated();
        let resources = QueryResourceContext::new(Arc::clone(db.buffer_manager())).unwrap();
        let grant = resources.try_allocate(4096).unwrap();
        let mut result = QueryResult::new(vec!["value".into()]).with_result_reservation(grant);
        result.int64_cols = Some(vec![vec![1, 2, 3]]);

        let columns = result.take_int64_cols().expect("dense column");
        let mut columns_iter = columns.into_iter();
        let column = columns_iter.next().expect("owned column");
        drop(columns_iter);
        assert!(db.buffer_manager().allocated() >= before + 4096);
        assert_eq!(&*column, &[1, 2, 3]);
        drop(column);
        assert_eq!(db.buffer_manager().allocated(), before);
        Ok(())
    }

    #[test]
    fn test_query_result_iter() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["x".into()]);
        result.rows.push(vec![Value::Int64(1)]);
        result.rows.push(vec![Value::Int64(2)]);

        let collected: Vec<_> = result.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0][0], Value::Int64(1));
        assert_eq!(collected[1][0], Value::Int64(2));
    }

    #[test]
    fn test_query_result_display() {
        use grafeo_common::types::Value;
        let mut result = QueryResult::new(vec!["name".into()]);
        result.rows.push(vec![Value::from("Alix")]);
        let display = result.to_string();
        assert!(display.contains("name"));
        assert!(display.contains("Alix"));
    }

    // =========================================================================
    // FromValue error paths
    // =========================================================================

    #[test]
    fn test_from_value_i64_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::from("not a number");
        assert!(i64::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_f64_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::from("not a float");
        assert!(f64::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_string_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::Int64(42);
        assert!(String::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_bool_type_mismatch() {
        use grafeo_common::types::Value;
        let val = Value::Int64(1);
        assert!(bool::from_value(&val).is_err());
    }

    #[test]
    fn test_from_value_all_success() {
        use grafeo_common::types::Value;
        assert_eq!(i64::from_value(&Value::Int64(99)).unwrap(), 99);
        assert!((f64::from_value(&Value::Float64(2.72)).unwrap() - 2.72).abs() < f64::EPSILON);
        assert_eq!(String::from_value(&Value::from("hello")).unwrap(), "hello");
        assert!(bool::from_value(&Value::Bool(true)).unwrap());
    }

    // =========================================================================
    // GrafeoDB accessor tests
    // =========================================================================

    #[test]
    fn test_database_is_read_only_false_by_default() {
        let db = GrafeoDB::new_in_memory();
        assert!(!db.is_read_only());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_database_graph_model() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.graph_model(), crate::config::GraphModel::Lpg);
    }

    #[test]
    fn test_database_memory_limit_none_by_default() {
        let db = GrafeoDB::new_in_memory();
        assert!(db.memory_limit().is_none());
    }

    #[test]
    fn test_database_memory_limit_custom() {
        let config =
            supported_test_config(Config::in_memory().with_memory_limit(128 * 1024 * 1024));
        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.memory_limit(), Some(128 * 1024 * 1024));
    }

    #[test]
    fn test_database_adaptive_config() {
        let db = GrafeoDB::new_in_memory();
        let adaptive = db.adaptive_config();
        assert!(adaptive.enabled);
        assert!((adaptive.threshold - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_database_buffer_manager() {
        let db = GrafeoDB::new_in_memory();
        let _bm = db.buffer_manager();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_database_query_cache() {
        let db = GrafeoDB::new_in_memory();
        let _qc = db.query_cache();
    }

    #[test]
    fn test_database_clear_plan_cache() {
        let db = GrafeoDB::new_in_memory();
        // Execute a query to populate the cache
        #[cfg(feature = "gql")]
        {
            let _ = db.execute("MATCH (n) RETURN count(n)");
        }
        db.clear_plan_cache();
        // No panic means success
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_database_gc() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        db.gc().expect("collect retained history");
        // Verify no panic, node still accessible
        assert_eq!(db.node_count(), 1);
    }

    // =========================================================================
    // Named graph management
    // =========================================================================

    #[cfg(feature = "lpg")]
    #[test]
    fn test_create_and_list_graphs() {
        let db = GrafeoDB::new_in_memory();
        let created = db.create_graph("social").unwrap();
        assert!(created);

        // Creating same graph again returns false
        let created_again = db.create_graph("social").unwrap();
        assert!(!created_again);

        let names = db.list_graphs();
        assert!(names.contains(&"social".to_string()));
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_drop_graph() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("temp").unwrap();
        assert!(db.drop_graph("temp").expect("drop graph"));
        assert!(!db.drop_graph("temp").expect("drop graph")); // Already dropped
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_drop_graph_resets_current_graph() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("active").unwrap();
        db.set_current_graph(Some("active")).unwrap();
        assert_eq!(db.current_graph(), Some("active".to_string()));

        db.drop_graph("active").expect("drop graph");
        assert_eq!(db.current_graph(), None);
    }

    // =========================================================================
    // Current graph / schema context
    // =========================================================================

    #[test]
    fn test_current_graph_default_none() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.current_graph(), None);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_set_current_graph_valid() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("social").unwrap();
        db.set_current_graph(Some("social")).unwrap();
        assert_eq!(db.current_graph(), Some("social".to_string()));
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn persistent_graph_context_resolves_and_resets_by_canonical_storage_key() {
        let db = GrafeoDB::new_in_memory();
        db.session().execute("CREATE SCHEMA Canonical").unwrap();
        db.create_graph("canonical/owned").unwrap();
        db.set_current_schema(Some("CANONICAL")).unwrap();

        db.set_current_graph(Some("owned")).unwrap();
        assert_eq!(db.current_schema().as_deref(), Some("Canonical"));
        assert_eq!(db.current_graph().as_deref(), Some("owned"));
        assert!(db.drop_graph("cAnOnIcAl/owned").expect("drop graph"));
        assert_eq!(db.current_graph(), None);

        db.create_graph("canonical/owned").unwrap();
        db.set_current_graph(Some("CANONICAL/owned")).unwrap();
        assert_eq!(
            db.current_graph().as_deref(),
            Some("owned"),
            "an absolute selector in the active schema must not be double-qualified"
        );
        assert!(db.drop_graph("Canonical/owned").expect("drop graph"));
        assert_eq!(db.current_graph(), None);

        db.create_graph("urn:example/root").unwrap();
        db.set_current_graph(Some("urn:example/root")).unwrap();
        assert_eq!(db.current_schema().as_deref(), Some("Canonical"));
        assert_eq!(db.current_graph().as_deref(), Some("urn:example/root"));
        assert!(db.drop_graph("urn:example/root").expect("drop graph"));
        assert_eq!(db.current_graph(), None);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn set_current_graph_validates_and_assigns_under_publication_barrier() {
        use std::sync::mpsc;
        use std::time::Duration;

        let db = Arc::new(GrafeoDB::new_in_memory());
        db.create_graph("vanishing").unwrap();
        let publication = db.transaction_manager.publication().write();
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(0);
        let (completed_tx, completed_rx) = mpsc::sync_channel(0);
        let worker_db = Arc::clone(&db);
        let worker = std::thread::spawn(move || {
            attempting_tx.send(()).unwrap();
            let result = worker_db.set_current_graph(Some("vanishing"));
            completed_tx
                .send((result, worker_db.current_graph()))
                .unwrap();
        });

        attempting_rx.recv().unwrap();
        assert!(matches!(
            completed_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(db.transaction_manager.with_write_authority(|| {
            crate::database::testing::root_lpg_store(&db).drop_graph("vanishing")
        }));
        drop(publication);

        let (result, current) = completed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert!(result.is_err());
        assert_eq!(current, None);
    }

    #[test]
    fn test_set_current_graph_nonexistent() {
        let db = GrafeoDB::new_in_memory();
        let result = db.set_current_graph(Some("nonexistent"));
        assert!(result.is_err());
        assert_eq!(db.current_graph(), None);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_set_current_graph_none_resets() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("social").unwrap();
        db.set_current_graph(Some("social")).unwrap();
        db.set_current_graph(None).unwrap();
        assert_eq!(db.current_graph(), None);
    }

    #[test]
    fn test_set_current_graph_default_keyword() {
        let db = GrafeoDB::new_in_memory();
        // "default" is a special case that always succeeds
        db.set_current_graph(Some("default")).unwrap();
        assert_eq!(db.current_graph(), Some("default".to_string()));
    }

    #[test]
    fn test_current_schema_default_none() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.current_schema(), None);
    }

    #[test]
    fn test_set_current_schema_nonexistent() {
        let db = GrafeoDB::new_in_memory();
        let result = db.set_current_schema(Some("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_set_current_schema_none_resets() {
        let db = GrafeoDB::new_in_memory();
        db.set_current_schema(None).unwrap();
        assert_eq!(db.current_schema(), None);
    }

    #[test]
    fn set_current_schema_canonicalizes_registered_case() {
        let db = GrafeoDB::new_in_memory();
        db.catalog
            .register_schema_namespace("CamelCase".to_string())
            .unwrap();

        db.set_current_schema(Some("camelcase")).unwrap();

        assert_eq!(db.current_schema().as_deref(), Some("CamelCase"));
    }

    #[test]
    fn set_current_schema_validates_and_assigns_under_publication_barrier() {
        use std::sync::mpsc;
        use std::time::Duration;

        let db = Arc::new(GrafeoDB::new_in_memory());
        db.catalog
            .register_schema_namespace("Vanishing".to_string())
            .unwrap();
        let publication = db.transaction_manager.publication().write();
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(0);
        let (completed_tx, completed_rx) = mpsc::sync_channel(0);
        let worker_db = Arc::clone(&db);
        let worker = std::thread::spawn(move || {
            attempting_tx.send(()).unwrap();
            let result = worker_db.set_current_schema(Some("vanishing"));
            completed_tx
                .send((result, worker_db.current_schema()))
                .unwrap();
        });

        attempting_rx.recv().unwrap();
        assert!(matches!(
            completed_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        db.catalog.drop_schema_namespace("Vanishing").unwrap();
        drop(publication);

        let (result, current) = completed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert!(result.is_err());
        assert_eq!(current, None);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn list_graphs_reads_under_publication_barrier() {
        use std::sync::mpsc;
        use std::time::Duration;

        let db = Arc::new(GrafeoDB::new_in_memory());
        db.create_graph("vanishing").unwrap();
        let publication = db.transaction_manager.publication().write();
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(0);
        let (completed_tx, completed_rx) = mpsc::sync_channel(0);
        let worker_db = Arc::clone(&db);
        let worker = std::thread::spawn(move || {
            attempting_tx.send(()).unwrap();
            completed_tx.send(worker_db.list_graphs()).unwrap();
        });

        attempting_rx.recv().unwrap();
        assert!(matches!(
            completed_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(db.transaction_manager.with_write_authority(|| {
            crate::database::testing::root_lpg_store(&db).drop_graph("vanishing")
        }));
        drop(publication);

        let graphs = completed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert!(!graphs.iter().any(|name| name == "vanishing"));
    }

    // =========================================================================
    // graph_store / graph_store_mut
    // =========================================================================

    #[cfg(feature = "lpg")]
    #[test]
    fn test_graph_store_returns_lpg_by_default() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        let store = db.graph_store();
        assert_eq!(store.node_count(), 1);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_graph_store_mut_hides_the_builtin_store() {
        let db = GrafeoDB::new_in_memory();
        assert!(db.graph_store_mut().is_none());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_with_read_store() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Person"]);

        let read_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let db = GrafeoDB::with_read_store(read_store, Config::in_memory()).unwrap();

        assert!(db.is_read_only());
        assert!(db.graph_store_mut().is_none());

        // Read queries should work
        let gs = db.graph_store();
        assert_eq!(gs.node_count(), 1);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn set_current_graph_preserves_legacy_selector_on_external_read_store() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let read_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let db = GrafeoDB::with_read_store(read_store, Config::in_memory()).unwrap();

        db.set_current_graph(Some("opaque/external"))
            .expect("external stores historically accept opaque graph selectors");
        assert_eq!(db.current_graph().as_deref(), Some("opaque/external"));
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn test_with_store_graph_store_methods() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Person"]);

        let db = GrafeoDB::with_store(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Config::in_memory(),
        )
        .unwrap();

        assert!(!db.is_read_only());
        assert!(db.graph_store_mut().is_some());
        assert_eq!(db.graph_store().node_count(), 1);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn set_current_graph_preserves_legacy_selector_on_external_write_store() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let db = GrafeoDB::with_store(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Config::in_memory(),
        )
        .unwrap();

        db.set_current_graph(Some("opaque/external"))
            .expect("external stores historically accept opaque graph selectors");
        assert_eq!(db.current_graph().as_deref(), Some("opaque/external"));
    }

    // =========================================================================
    // Read-only session role
    // =========================================================================

    #[cfg(feature = "lpg")]
    #[test]
    fn test_session_with_readonly_role() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);

        let session = db.session_with_role(crate::auth::Role::ReadOnly);
        assert!(session.identity().can_read());
        assert!(!session.identity().can_write());
        // Read queries should work
        #[cfg(feature = "gql")]
        {
            let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
            assert_eq!(result.row_count(), 1);
        }
    }

    // =========================================================================
    // close on in-memory database
    // =========================================================================

    #[cfg(feature = "lpg")]
    #[test]
    fn test_close_in_memory_database() {
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        assert!(db.close().is_ok());
        // Second close should also be fine (idempotent)
        assert!(db.close().is_ok());
    }

    #[cfg(all(feature = "lpg", feature = "gql"))]
    #[test]
    fn test_try_close_rejects_active_stream_without_waiting_and_retries() {
        let db = Arc::new(GrafeoDB::new_in_memory());
        db.create_node(&["CloseLease"]);
        let mut stream = db
            .execute_streaming("MATCH (n:CloseLease) RETURN id(n) AS id")
            .unwrap();
        assert_eq!(stream.columns(), &["id".to_owned()]);

        let worker_db = Arc::clone(&db);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            sender.send(worker_db.try_close()).unwrap();
        });
        let attempt = receiver.recv_timeout(std::time::Duration::from_secs(1));
        // Always release the stream before joining, so a regression to blocking
        // close reports a failure instead of leaving a hung test process.
        stream.close().unwrap();
        worker.join().unwrap();
        let error = attempt
            .expect("try_close must return while the stream still owns its lifecycle pin")
            .unwrap_err();
        assert_eq!(
            error.error_code(),
            grafeo_common::utils::error::ErrorCode::TransactionInvalidState,
        );
        assert!(!db.is_durability_poisoned());
        db.try_close().unwrap();
        db.try_close().unwrap();
        db.close().unwrap();
    }

    #[cfg(all(
        target_os = "linux",
        feature = "lpg",
        feature = "gql",
        feature = "grafeo-file"
    ))]
    #[cfg(feature = "wal")]
    #[test]
    fn test_try_close_preserves_persistent_timer_while_stream_is_active() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("try-close-stream.grafeo");
        let db = Arc::new(
            GrafeoDB::with_config(
                Config::persistent(&path)
                    .with_checkpoint_interval(std::time::Duration::from_millis(10)),
            )
            .unwrap(),
        );
        db.execute("INSERT (:CloseTimer {id: 1})").unwrap();
        let mut stream = db
            .execute_streaming("MATCH (n:CloseTimer) RETURN n.id")
            .unwrap();
        let timer_thread = {
            let timer = db.checkpoint_timer.lock();
            super::checkpoint_timer::tests::running_timer_thread(timer.as_ref().unwrap())
        };
        // Give the periodic worker an opportunity to queue behind the stream.
        // Correctness does not depend on scheduling: busy must never remove or
        // stop the timer, whether it is asleep or already waiting for a gate.
        std::thread::sleep(std::time::Duration::from_millis(150));
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker_db = Arc::clone(&db);
        let worker = std::thread::spawn(move || sender.send(worker_db.try_close()).unwrap());
        let attempt = receiver.recv_timeout(std::time::Duration::from_secs(1));
        // Unblock even a broken implementation before joining its worker.
        stream.close().unwrap();
        worker.join().unwrap();
        assert_eq!(
            attempt
                .expect("close waited on the stream/timer")
                .unwrap_err()
                .error_code(),
            grafeo_common::utils::error::ErrorCode::TransactionInvalidState
        );
        {
            let timer = db.checkpoint_timer.lock();
            assert_eq!(
                super::checkpoint_timer::tests::running_timer_thread(timer.as_ref().unwrap()),
                timer_thread
            );
        }
        db.close().unwrap();
        assert!(db.checkpoint_timer.lock().is_none());
        drop(stream);
        drop(db);
        let reopened = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            reopened
                .execute("MATCH (n:CloseTimer) RETURN n.id")
                .unwrap()
                .row_count(),
            1
        );
        reopened.close().unwrap();
    }

    // =========================================================================
    // with_config validation failure
    // =========================================================================

    #[test]
    fn test_with_config_invalid_config_zero_threads() {
        let config = Config::in_memory().with_threads(0);
        let result = GrafeoDB::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_config_invalid_config_zero_memory_limit() {
        let config = Config::in_memory().with_memory_limit(0);
        let result = GrafeoDB::with_config(config);
        assert!(result.is_err());
    }

    // =========================================================================
    // StorageFormat display (for config.rs coverage)
    // =========================================================================

    #[test]
    fn test_storage_format_display() {
        use crate::config::StorageFormat;
        assert_eq!(StorageFormat::Auto.to_string(), "auto");
        assert_eq!(StorageFormat::WalDirectory.to_string(), "wal-directory");
        assert_eq!(StorageFormat::SingleFile.to_string(), "single-file");
    }

    #[test]
    fn test_storage_format_default() {
        use crate::config::StorageFormat;
        assert_eq!(StorageFormat::default(), StorageFormat::Auto);
    }

    #[test]
    fn test_config_with_storage_format() {
        use crate::config::StorageFormat;
        let config = Config::in_memory().with_storage_format(StorageFormat::SingleFile);
        assert_eq!(config.storage_format, StorageFormat::SingleFile);
    }

    // =========================================================================
    // Config CDC
    // =========================================================================

    #[test]
    fn test_config_with_cdc() {
        let config = Config::in_memory().with_cdc();
        assert!(config.cdc_enabled);
    }

    #[test]
    fn test_config_cdc_default_false() {
        let config = Config::in_memory();
        assert!(!config.cdc_enabled);
    }

    // =========================================================================
    // ConfigError as std::error::Error
    // =========================================================================

    #[test]
    fn test_config_error_is_error_trait() {
        use crate::config::ConfigError;
        let err: Box<dyn std::error::Error> = Box::new(ConfigError::ZeroMemoryLimit);
        assert!(err.source().is_none());
    }

    // =========================================================================
    // Metrics tests
    // =========================================================================

    #[cfg(feature = "metrics")]
    #[test]
    fn test_metrics_prometheus_output() {
        let db = GrafeoDB::new_in_memory();
        let prom = db.metrics_prometheus();
        // Should contain at least some metric names
        assert!(!prom.is_empty());
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_reset_metrics() {
        let db = GrafeoDB::new_in_memory();
        // Execute something to generate metrics
        let _session = db.session();
        db.reset_metrics();
        let snap = db.metrics();
        assert_eq!(snap.query_count, 0);
    }

    // =========================================================================
    // drop_graph on external store
    // =========================================================================

    #[cfg(feature = "lpg")]
    #[test]
    fn test_drop_graph_on_external_store() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let read_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let db = GrafeoDB::with_read_store(read_store, Config::in_memory()).unwrap();

        // External read stores cannot admit graph lifecycle mutations.
        assert!(db.drop_graph("anything").is_err());
    }
}
