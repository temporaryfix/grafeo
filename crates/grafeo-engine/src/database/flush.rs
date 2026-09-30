//! Unified flush: one code path for checkpoint, eviction, and explicit CHECKPOINT.
//!
//! The [`FlushManager`] replaces separate checkpoint/snapshot/close paths with
//! a single `flush()` method. Three triggers, one implementation:
//!
//! | Trigger | What gets written | RAM after flush |
//! |---------|-------------------|-----------------|
//! | Periodic checkpoint | Complete image if any section is dirty | Kept |
//! | Memory pressure | Lowest-priority section | Mmap back, release RAM |
//! | Explicit CHECKPOINT | All sections | Kept |
//!
//! Future phases will add memory pressure integration with BufferManager.

use grafeo_common::storage::Section;
#[cfg(test)]
use grafeo_common::storage::SectionType;
#[cfg(feature = "grafeo-file")]
use grafeo_common::types::GraphModelTag;
#[cfg(feature = "grafeo-file")]
use grafeo_common::utils::error::Error;
use grafeo_common::utils::error::Result;

#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

/// Reason for triggering a flush.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlushReason {
    /// Periodic checkpoint (timer-driven) or database close.
    #[allow(dead_code)] // Used by async_ops (async-storage feature)
    Checkpoint,
    /// User-initiated `CHECKPOINT` command or `wal_checkpoint()` API.
    Explicit,
}

/// Context needed by each section during serialization.
pub(super) struct FlushContext {
    pub epoch: u64,
    pub transaction_id: u64,
    pub node_count: u64,
    pub edge_count: u64,
    pub world_cut: super::world_metadata::WorldCutInputs,
}

/// Result of a flush operation.
#[derive(Debug)]
pub(super) struct FlushResult {
    /// Number of sections written to the container.
    pub sections_written: usize,
}

/// Executes the unified flush: serialize dirty sections, write to container, truncate WAL.
///
/// This is the single write path for all persistence operations.
///
/// # Errors
///
/// Returns an error if serialization or I/O fails.
#[cfg(feature = "grafeo-file")]
pub(super) fn flush(
    fm: &GrafeoFileManager,
    sections: &[&dyn Section],
    captured_lpg: Option<super::world_metadata::EncodedSection>,
    context: &FlushContext,
    reason: FlushReason,
    #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
) -> Result<FlushResult> {
    use grafeo_common::testing::crash::maybe_crash;

    maybe_crash("flush:before_serialize");

    // The file manager installs a whole new container image atomically. A
    // periodic poll may avoid that work when the state is already clean, but
    // once publication is needed every section must be present in the image;
    // publishing only the dirty subset would silently drop every clean section.
    let should_publish =
        reason == FlushReason::Explicit || sections.iter().any(|section| section.is_dirty());
    if !should_publish {
        return Ok(FlushResult {
            sections_written: 0,
        });
    }

    let file_graph_model = GraphModelTag::from_u8(fm.graph_model_tag()).map_err(|error| {
        Error::Serialization(format!(
            "cannot publish checkpoint through an invalid file graph-model tag: {error}"
        ))
    })?;
    if file_graph_model != context.world_cut.graph_model {
        return Err(Error::Serialization(format!(
            "checkpoint graph model {:?} does not match immutable file graph model {:?}",
            context.world_cut.graph_model, file_graph_model
        )));
    }

    let mut targets = super::world_metadata::encode_sections(sections, captured_lpg)?;
    let recovery_coordinates = super::world_metadata::RecoveryCoordinates::new(
        context.epoch,
        context.transaction_id,
        context.world_cut.graph_model,
        context.node_count,
        context.edge_count,
    );
    super::world_metadata::append_world_metadata(
        &mut targets,
        context.world_cut.clone(),
        recovery_coordinates,
    )?;

    let sections_written = targets.len();

    maybe_crash("flush:after_serialize");

    // Write sections to container
    let section_refs: Vec<SectionWrite<'_>> = targets
        .iter()
        .map(|section| {
            SectionWrite::new(section.section_type(), section.version(), section.bytes())
        })
        .collect();

    fm.write_versioned_sections(
        &section_refs,
        context.epoch,
        context.transaction_id,
        context.node_count,
        context.edge_count,
    )?;

    // The complete cut is durable: every wrapper now represents clean state.
    for section in sections {
        section.mark_clean();
    }

    maybe_crash("flush:after_write");

    // Sync WAL to disk (all data is now in the container)
    #[cfg(feature = "wal")]
    if let Some(wal) = wal {
        wal.sync()?;
    }

    Ok(FlushResult { sections_written })
}

/// Builds the flush context from the current database state.
#[cfg(feature = "lpg")]
pub(super) fn build_context(
    store: &dyn grafeo_core::graph::GraphStore,
    transaction_manager: &crate::transaction::TransactionManager,
    world_cut: super::world_metadata::WorldCutInputs,
) -> FlushContext {
    FlushContext {
        // The transaction manager owns the cross-model commit frontier. An
        // RDF-only mutation in GraphModel::Both must advance the installed cut
        // even when the LPG overlay did not change in that transaction.
        epoch: transaction_manager.current_epoch().as_u64(),
        transaction_id: transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0),
        node_count: store.node_count() as u64,
        edge_count: store.edge_count() as u64,
        world_cut,
    }
}

/// Builds a flush context when no LPG store is available (RDF-only).
#[cfg(not(feature = "lpg"))]
pub(super) fn build_context_minimal(
    transaction_manager: &crate::transaction::TransactionManager,
    epoch: u64,
    world_cut: super::world_metadata::WorldCutInputs,
) -> FlushContext {
    FlushContext {
        epoch,
        transaction_id: transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0),
        node_count: 0,
        edge_count: 0,
        world_cut,
    }
}

#[cfg(all(test, feature = "grafeo-file"))]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use grafeo_common::storage::Section;
    use grafeo_common::types::{
        Digest256, EpochId, GraphModelTag, HistoryCompleteness, SchemaCut, StoreId,
        WorldIdentityMetadataV1,
    };

    use super::*;

    struct TestSection {
        section_type: SectionType,
        version: u8,
        bytes: &'static [u8],
        dirty: AtomicBool,
    }

    impl TestSection {
        fn new(section_type: SectionType, version: u8, bytes: &'static [u8], dirty: bool) -> Self {
            Self {
                section_type,
                version,
                bytes,
                dirty: AtomicBool::new(dirty),
            }
        }
    }

    impl Section for TestSection {
        fn section_type(&self) -> SectionType {
            self.section_type
        }

        fn version(&self) -> u8 {
            self.version
        }

        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(self.bytes.to_vec())
        }

        fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
            Ok(())
        }

        fn is_dirty(&self) -> bool {
            self.dirty.load(Ordering::Acquire)
        }

        fn mark_clean(&self) {
            self.dirty.store(false, Ordering::Release);
        }

        fn memory_usage(&self) -> usize {
            self.bytes.len()
        }
    }

    fn context() -> FlushContext {
        FlushContext {
            epoch: 7,
            transaction_id: 11,
            node_count: 13,
            edge_count: 17,
            world_cut: super::super::world_metadata::WorldCutInputs {
                identity: WorldIdentityMetadataV1::new(
                    StoreId::from_bytes([7; StoreId::LEN]).unwrap(),
                    HistoryCompleteness::Complete,
                )
                .unwrap(),
                epoch: EpochId::new(7),
                graph_model: GraphModelTag::Lpg,
                schema: SchemaCut::new(1, Digest256::schema(b"catalog")).unwrap(),
                projections: Vec::new(),
            },
        }
    }

    #[test]
    fn periodic_publication_installs_a_complete_versioned_container() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("complete-cut.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        let clean = TestSection::new(SectionType::Catalog, 5, b"catalog", false);
        let dirty = TestSection::new(SectionType::LpgStore, 9, b"lpg", true);
        let sections: [&dyn Section; 2] = [&clean, &dirty];

        let result = flush(
            &manager,
            &sections,
            None,
            &context(),
            FlushReason::Checkpoint,
            #[cfg(feature = "wal")]
            None,
        )
        .unwrap();

        assert_eq!(result.sections_written, 3);
        let directory = manager.read_section_directory().unwrap().unwrap();
        let catalog = directory.find(SectionType::Catalog).unwrap();
        let lpg = directory.find(SectionType::LpgStore).unwrap();
        let world = directory.find(SectionType::WorldMetadata).unwrap();
        assert_eq!(catalog.version, 5);
        assert_eq!(lpg.version, 9);
        assert_eq!(world.version, 2);
        assert_eq!(manager.read_section_data(catalog).unwrap(), b"catalog");
        assert_eq!(manager.read_section_data(lpg).unwrap(), b"lpg");
        assert!(!clean.is_dirty());
        assert!(!dirty.is_dirty());
    }

    #[test]
    fn clean_periodic_poll_leaves_the_installed_cut_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean-poll.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        let section = TestSection::new(SectionType::Catalog, 5, b"catalog", false);
        let before = manager.active_header();

        let result = flush(
            &manager,
            &[&section],
            None,
            &context(),
            FlushReason::Checkpoint,
            #[cfg(feature = "wal")]
            None,
        )
        .unwrap();

        assert_eq!(result.sections_written, 0);
        assert_eq!(manager.active_header(), before);
        assert!(manager.read_section_directory().unwrap().is_none());
    }

    #[test]
    fn publication_rejects_a_context_for_another_file_graph_model_before_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("foreign-model.grafeo");
        let manager =
            GrafeoFileManager::create_with_graph_model(&path, GraphModelTag::Rdf.as_u8()).unwrap();
        let dirty = TestSection::new(SectionType::LpgStore, 9, b"lpg", true);
        let before = manager.active_header();

        let error = flush(
            &manager,
            &[&dirty],
            None,
            &context(),
            FlushReason::Explicit,
            #[cfg(feature = "wal")]
            None,
        )
        .expect_err("an LPG cut must not publish through an RDF file manager");

        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert!(
            dirty.is_dirty(),
            "rejected publication must remain retryable"
        );
        assert_eq!(manager.active_header(), before);
        assert!(manager.read_section_directory().unwrap().is_none());
    }
}
