//! Periodic checkpoint timer for automatic durability.
//!
//! When [`Config::checkpoint_interval`](crate::Config::checkpoint_interval) is set,
//! the engine spawns a background thread that periodically flushes dirty sections
//! to the `.grafeo` container. This bounds WAL size and recovery work. Commit
//! loss is governed by the configured WAL durability mode; the default `Sync`
//! mode does not trade acknowledged commits for checkpoint cadence.
//!
//! The timer polls a shutdown flag in short intervals (100 ms) so `close()`
//! completes promptly without blocking for the full checkpoint interval.

#[cfg(feature = "grafeo-file")]
use std::sync::Arc;
#[cfg(feature = "grafeo-file")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "grafeo-file")]
use std::time::Duration;

#[cfg(feature = "grafeo-file")]
use grafeo_common::types::{GraphModelTag, WorldIdentityMetadataV1};
#[cfg(feature = "grafeo-file")]
use grafeo_common::utils::error::{Error, Result, TransactionError};
#[cfg(feature = "grafeo-file")]
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;

#[cfg(feature = "grafeo-file")]
use crate::catalog::Catalog;
#[cfg(feature = "grafeo-file")]
use crate::config::GraphModel;
#[cfg(feature = "grafeo-file")]
use crate::transaction::TransactionManager;

/// How often the timer thread checks the shutdown flag.
#[cfg(feature = "grafeo-file")]
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Background checkpoint timer.
///
/// Spawns a thread that periodically triggers a unified flush. The thread
/// exits cleanly when [`stop`](Self::stop) is called (from `GrafeoDB::close()`).
#[cfg(feature = "grafeo-file")]
pub(super) struct CheckpointTimer {
    /// Shutdown signal: set to true to stop the timer thread.
    shutdown: Arc<AtomicBool>,
    /// Thread handle (taken on stop).
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "grafeo-file")]
impl CheckpointTimer {
    /// Starts the checkpoint timer.
    ///
    /// The background thread wakes every `interval`, checks if any sections
    /// are dirty, and flushes them to the container. If no mutations happened
    /// since the last checkpoint, the flush is skipped (no I/O).
    #[allow(
        clippy::too_many_arguments,
        reason = "checkpoint ownership is passed explicitly into the background thread"
    )]
    pub(super) fn start(
        #[cfg(feature = "cdc")] cdc_log: Arc<crate::cdc::CdcLog>,
        interval: Duration,
        file_manager: Arc<GrafeoFileManager>,
        store: Arc<LpgStore>,
        catalog: Arc<Catalog>,
        transaction_manager: Arc<TransactionManager>,
        durability_poisoned: Arc<AtomicBool>,
        world_identity: Arc<parking_lot::RwLock<WorldIdentityMetadataV1>>,
        graph_model: GraphModel,
        #[cfg(feature = "triple-store")] rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
        #[cfg(feature = "wal")] wal: Option<Arc<grafeo_storage::wal::LpgWal>>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = std::thread::Builder::new()
            .name("grafeo-checkpoint".to_string())
            .spawn(move || {
                Self::run(
                    #[cfg(feature = "cdc")]
                    &cdc_log,
                    &shutdown_clone,
                    interval,
                    &file_manager,
                    &store,
                    &catalog,
                    &transaction_manager,
                    &durability_poisoned,
                    &world_identity,
                    graph_model,
                    #[cfg(feature = "triple-store")]
                    &rdf_store,
                    #[cfg(feature = "triple-store")]
                    &rdf_projections,
                    #[cfg(feature = "wal")]
                    wal.as_deref(),
                );
            })
            .expect("failed to spawn checkpoint timer thread");

        Self {
            shutdown,
            handle: Some(handle),
        }
    }

    /// Signals the timer thread to stop and waits for it to exit.
    ///
    /// Returns within ~100 ms regardless of the checkpoint interval.
    pub(super) fn stop(&mut self) {
        self.request_stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Requests shutdown without joining a worker that may be waiting on the
    /// caller's publication gate. Join only after releasing that gate.
    pub(super) fn request_stop(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Timer loop: sleep in short increments, checkpoint when the interval
    /// elapses, exit when shutdown is signaled.
    #[allow(clippy::too_many_arguments)]
    fn run(
        #[cfg(feature = "cdc")] cdc_log: &crate::cdc::CdcLog,
        shutdown: &AtomicBool,
        interval: Duration,
        file_manager: &GrafeoFileManager,
        store: &Arc<LpgStore>,
        catalog: &Arc<Catalog>,
        transaction_manager: &TransactionManager,
        durability_poisoned: &AtomicBool,
        world_identity: &Arc<parking_lot::RwLock<WorldIdentityMetadataV1>>,
        graph_model: GraphModel,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: &Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
        #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    ) {
        let mut elapsed = Duration::ZERO;

        loop {
            std::thread::sleep(POLL_INTERVAL);

            if shutdown.load(Ordering::Acquire) {
                break;
            }

            elapsed += POLL_INTERVAL;
            if elapsed < interval {
                continue;
            }
            elapsed = Duration::ZERO;

            // Attempt checkpoint (errors are logged, not propagated)
            if let Err(e) = Self::try_checkpoint(
                #[cfg(feature = "cdc")]
                cdc_log,
                shutdown,
                file_manager,
                store,
                catalog,
                transaction_manager,
                durability_poisoned,
                world_identity,
                graph_model,
                #[cfg(feature = "triple-store")]
                rdf_store,
                #[cfg(feature = "triple-store")]
                rdf_projections,
                #[cfg(feature = "wal")]
                wal,
            ) {
                eprintln!("periodic checkpoint failed: {e}");
            }
        }
    }

    /// Runs a single checkpoint cycle.
    #[allow(
        clippy::too_many_arguments,
        reason = "one checkpoint cut must receive every authoritative model and durability owner"
    )]
    fn try_checkpoint(
        #[cfg(feature = "cdc")] cdc_log: &crate::cdc::CdcLog,
        shutdown: &AtomicBool,
        file_manager: &GrafeoFileManager,
        store: &Arc<LpgStore>,
        catalog: &Arc<Catalog>,
        transaction_manager: &TransactionManager,
        durability_poisoned: &AtomicBool,
        world_identity: &Arc<parking_lot::RwLock<WorldIdentityMetadataV1>>,
        graph_model: GraphModel,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: &Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
        #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    ) -> Result<()> {
        use super::flush;

        Self::require_durability_healthy(
            durability_poisoned,
            #[cfg(feature = "wal")]
            wal,
        )?;
        #[cfg(test)]
        tests::checkpoint_admission_rendezvous();
        #[cfg(feature = "triple-store")]
        let _rdf_gate = rdf_store.lock_commit();
        let _publication = transaction_manager.publication().write();
        // Compaction signals shutdown while holding this exact publication
        // cut. A worker that was already queued must never serialize the
        // retired native representation after the Layered successor is live.
        if shutdown.load(Ordering::Acquire) {
            return Ok(());
        }
        // This is the authoritative check. Catalog and transaction
        // publication hold the same exclusive barrier while appending WAL and
        // setting the sticky poison flag. Therefore a timer that raced a lost
        // acknowledgement either checkpointed the preceding committed cut or
        // observes poison here before it captures the newer epoch.
        Self::require_durability_healthy(
            durability_poisoned,
            #[cfg(feature = "wal")]
            wal,
        )?;
        if transaction_manager.active_count() > 0 {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "periodic checkpoint requires a quiescent committed cut".to_string(),
                ),
            ));
        }
        // The catalog and world metadata must describe the same publication
        // cut. Capture the epoch once while publication remains excluded and
        // reuse it for both encodings.
        let epoch = transaction_manager.current_epoch();
        let (mut sections, captured_lpg) = Self::build_sections(
            store,
            catalog,
            graph_model,
            epoch.as_u64(),
            #[cfg(feature = "triple-store")]
            rdf_store,
            #[cfg(feature = "triple-store")]
            rdf_projections,
        )?;
        sections.push(Box::new(super::cdc_checkpoint::CapturedSection(
            super::cdc_checkpoint::capture_state(
                world_identity.read().store_id(),
                epoch,
                #[cfg(feature = "cdc")]
                cdc_log,
            )?,
        )));
        let section_refs: Vec<&dyn grafeo_common::storage::Section> =
            sections.iter().map(|s| s.as_ref()).collect();
        let graph_model_tag = GraphModelTag::from_u8(graph_model.as_u8()).map_err(|error| {
            grafeo_common::utils::error::Error::Serialization(format!(
                "capture periodic checkpoint graph model: {error}"
            ))
        })?;
        let world_identity = world_identity.read().clone();
        #[cfg(feature = "triple-store")]
        let world_identity = super::world_metadata::validate_live_world_identity(
            world_identity,
            graph_model_tag,
            rdf_store,
        )?;
        let world_cut = super::world_metadata::capture_world_cut_inputs(
            world_identity,
            epoch,
            graph_model_tag,
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_projections,
        )?;
        let context = flush::build_context(store.as_ref(), transaction_manager, world_cut);

        flush::flush(
            file_manager,
            &section_refs,
            captured_lpg,
            &context,
            flush::FlushReason::Explicit,
            #[cfg(feature = "wal")]
            wal,
        )
        .map(|_| ())
    }

    fn require_durability_healthy(
        durability_poisoned: &AtomicBool,
        #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    ) -> Result<()> {
        #[cfg(feature = "wal")]
        if wal.is_some_and(grafeo_storage::wal::LpgWal::is_poisoned) {
            durability_poisoned.store(true, Ordering::SeqCst);
        }
        if durability_poisoned.load(Ordering::SeqCst) {
            return Err(Error::Transaction(TransactionError::DurabilityFailure(
                "periodic checkpoint refused a durability-poisoned cut; reopen and recover the WAL first"
                    .to_string(),
            )));
        }
        Ok(())
    }

    /// Builds section objects from the captured components.
    fn build_sections(
        store: &Arc<LpgStore>,
        catalog: &Arc<Catalog>,
        graph_model: GraphModel,
        epoch: u64,
        #[cfg(feature = "triple-store")] rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
        #[cfg(feature = "triple-store")] rdf_projections: &Arc<
            grafeo_core::graph::rdf::RdfLpgProjectionRegistry,
        >,
    ) -> Result<(
        Vec<Box<dyn grafeo_common::storage::Section>>,
        Option<super::world_metadata::EncodedSection>,
    )> {
        #[allow(
            unused_mut,
            reason = "mut needed when triple-store/ring-index/vector-index/text-index features push sections"
        )]
        let mut sections: Vec<Box<dyn grafeo_common::storage::Section>> = Vec::new();

        let captured = matches!(graph_model, GraphModel::Lpg | GraphModel::Both)
            .then(|| grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store)).capture())
            .transpose()?;
        let (index_graphs, captured_lpg) = match captured {
            Some(captured) => (
                Some(super::index_sections::LpgIndexGraphCut::from_graphs(
                    captured.graphs,
                )),
                Some(super::world_metadata::EncodedSection::new(
                    grafeo_common::storage::SectionType::LpgStore,
                    4,
                    captured.bytes,
                )?),
            ),
            None => (None, None),
        };

        if let Some(index_graphs) = &index_graphs {
            sections.push(Box::new(
                super::catalog_section::CatalogSection::new_with_graphs(
                    Arc::clone(catalog),
                    index_graphs.graphs(),
                    move || epoch,
                )?,
            ));
        } else {
            sections.push(Box::new(
                super::catalog_state_section::CatalogStateSection::new(Arc::clone(catalog)),
            ));
        }

        #[cfg(feature = "triple-store")]
        if matches!(graph_model, GraphModel::Rdf | GraphModel::Both) {
            let rdf = grafeo_core::graph::rdf::RdfStoreSection::with_projections_under_commit_gate(
                Arc::clone(rdf_store),
                Arc::clone(rdf_projections),
            )?;
            sections.push(Box::new(rdf));
        }

        #[cfg(feature = "ring-index")]
        if matches!(graph_model, GraphModel::Rdf | GraphModel::Both) && rdf_store.ring().is_some() {
            let ring = grafeo_core::index::ring::RdfRingSection::new(Arc::clone(rdf_store));
            sections.push(Box::new(ring));
        }

        #[cfg(feature = "vector-index")]
        if let Some(index_graphs) = &index_graphs {
            let indexes = index_graphs.vector_views()?;
            if !indexes.is_empty() {
                let vector = grafeo_core::index::vector::VectorStoreSection::from_views(indexes);
                sections.push(Box::new(vector));
            }
        }

        #[cfg(feature = "text-index")]
        if let Some(index_graphs) = &index_graphs {
            let indexes = index_graphs.text_views()?;
            if !indexes.is_empty() {
                let text = grafeo_core::index::text::TextIndexSection::from_views(indexes);
                sections.push(Box::new(text));
            }
        }

        Ok((sections, captured_lpg))
    }
}

#[cfg(feature = "grafeo-file")]
impl Drop for CheckpointTimer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
#[cfg(feature = "grafeo-file")]
pub(super) mod tests {
    use super::*;
    #[cfg(all(feature = "lpg", feature = "vector-index", feature = "text-index"))]
    use grafeo_common::types::EpochId;
    use grafeo_common::types::{HistoryCompleteness, StoreId};
    use std::time::Instant;

    std::thread_local! {
        static CHECKPOINT_ADMISSION: std::cell::RefCell<Option<(
            std::sync::mpsc::Sender<()>,
            std::sync::mpsc::Receiver<()>,
        )>> = const { std::cell::RefCell::new(None) };
    }

    pub(super) fn checkpoint_admission_rendezvous() {
        CHECKPOINT_ADMISSION.with(|slot| {
            if let Some((arrived, resume)) = slot.borrow_mut().take() {
                arrived.send(()).expect("checkpoint reached admission");
                resume.recv().expect("resume checkpoint admission");
            }
        });
    }

    #[cfg(any(target_os = "linux", all(feature = "compact-store", feature = "lpg")))]
    pub(in crate::database) fn running_timer_thread(
        timer: &CheckpointTimer,
    ) -> std::thread::ThreadId {
        assert!(!timer.shutdown.load(Ordering::Acquire));
        let handle = timer.handle.as_ref().expect("live checkpoint thread");
        assert!(!handle.is_finished());
        handle.thread().id()
    }

    fn stopped_checkpoint_waiter_does_not_publish(block_rdf: bool) {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["OldNativeSource"]);
        let catalog = Arc::new(Catalog::new());
        let tm = Arc::new(TransactionManager::new());
        let shutdown = AtomicBool::new(false);
        let poison = AtomicBool::new(false);
        let identity = lpg_identity();
        #[cfg(feature = "triple-store")]
        let rdf = Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        #[cfg(feature = "triple-store")]
        let projections = Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stopped-native-checkpoint.grafeo");
        let fm = GrafeoFileManager::create(&path).unwrap();
        let before = std::fs::read(&path).unwrap();
        let iteration = fm.active_header().iteration;

        std::thread::scope(|scope| {
            #[cfg(feature = "triple-store")]
            let rdf_gate = block_rdf.then(|| rdf.lock_commit());
            let publication = (!block_rdf).then(|| tm.publication().write());
            let (arrived_tx, arrived_rx) = std::sync::mpsc::channel();
            let (resume_tx, resume_rx) = std::sync::mpsc::channel();
            let worker = scope.spawn(|| {
                CHECKPOINT_ADMISSION.with(|slot| {
                    *slot.borrow_mut() = Some((arrived_tx, resume_rx));
                });
                CheckpointTimer::try_checkpoint(
                    #[cfg(feature = "cdc")]
                    &crate::cdc::CdcLog::new(),
                    &shutdown,
                    &fm,
                    &store,
                    &catalog,
                    &tm,
                    &poison,
                    &identity,
                    GraphModel::Lpg,
                    #[cfg(feature = "triple-store")]
                    &rdf,
                    #[cfg(feature = "triple-store")]
                    &projections,
                    #[cfg(feature = "wal")]
                    None,
                )
            });
            arrived_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            resume_tx.send(()).unwrap();
            // The caller owns the gate through shutdown publication. The
            // worker has passed its initial checks, but cannot capture a cut.
            shutdown.store(true, Ordering::Release);
            drop(publication);
            #[cfg(feature = "triple-store")]
            drop(rdf_gate);
            worker.join().unwrap().unwrap();
        });
        assert_eq!(fm.active_header().iteration, iteration);
        assert!(fm.read_section_directory().unwrap().is_none());
        assert_eq!(std::fs::read(&path).unwrap(), before);

        // Positive control: the same concrete source is checkpointable when
        // the stop flag is absent; the negative cannot pass on invalid input.
        shutdown.store(false, Ordering::Release);
        CheckpointTimer::try_checkpoint(
            #[cfg(feature = "cdc")]
            &crate::cdc::CdcLog::new(),
            &shutdown,
            &fm,
            &store,
            &catalog,
            &tm,
            &poison,
            &identity,
            GraphModel::Lpg,
            #[cfg(feature = "triple-store")]
            &rdf,
            #[cfg(feature = "triple-store")]
            &projections,
            #[cfg(feature = "wal")]
            None,
        )
        .unwrap();
        assert!(fm.active_header().iteration > iteration);
        assert!(fm.read_section_directory().unwrap().is_some());
    }

    #[test]
    fn stopped_timer_waiting_for_publication_cannot_checkpoint_retired_native_source() {
        stopped_checkpoint_waiter_does_not_publish(false);
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn stopped_timer_waiting_for_rdf_cannot_checkpoint_retired_native_source() {
        stopped_checkpoint_waiter_does_not_publish(true);
    }

    fn identity(store_id: StoreId) -> Arc<parking_lot::RwLock<WorldIdentityMetadataV1>> {
        Arc::new(parking_lot::RwLock::new(
            WorldIdentityMetadataV1::new(store_id, HistoryCompleteness::Complete).unwrap(),
        ))
    }

    fn lpg_identity() -> Arc<parking_lot::RwLock<WorldIdentityMetadataV1>> {
        identity(StoreId::from_bytes([7; StoreId::LEN]).unwrap())
    }

    #[test]
    fn timer_stops_promptly() {
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Arc::new(Catalog::new());
        let tm = Arc::new(TransactionManager::new());

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("timer_test.grafeo");
        let fm = Arc::new(GrafeoFileManager::create(&path).unwrap());

        let mut timer = CheckpointTimer::start(
            #[cfg(feature = "cdc")]
            Arc::new(crate::cdc::CdcLog::new()),
            Duration::from_mins(1), // Long interval
            fm,
            store,
            catalog,
            tm,
            Arc::new(AtomicBool::new(false)),
            lpg_identity(),
            GraphModel::Lpg,
            #[cfg(feature = "triple-store")]
            Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
            #[cfg(feature = "triple-store")]
            Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new()),
            #[cfg(feature = "wal")]
            None,
        );

        let start = Instant::now();
        timer.stop();
        let elapsed = start.elapsed();

        // Should stop within a few poll cycles, not 60 seconds
        assert!(
            elapsed < Duration::from_secs(2),
            "stop() took {elapsed:?}, expected < 2s"
        );
    }

    #[test]
    fn timer_checkpoints_on_interval() {
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Arc::new(Catalog::new());
        let tm = Arc::new(TransactionManager::new());

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("interval_test.grafeo");
        let fm = Arc::new(GrafeoFileManager::create(&path).unwrap());

        // Add some data so sections have content
        store.create_node(&["Test"]);

        let mut timer = CheckpointTimer::start(
            #[cfg(feature = "cdc")]
            Arc::new(crate::cdc::CdcLog::new()),
            Duration::from_millis(200), // Short interval for testing
            Arc::clone(&fm),
            Arc::clone(&store),
            Arc::clone(&catalog),
            Arc::clone(&tm),
            Arc::new(AtomicBool::new(false)),
            lpg_identity(),
            GraphModel::Lpg,
            #[cfg(feature = "triple-store")]
            Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
            #[cfg(feature = "triple-store")]
            Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new()),
            #[cfg(feature = "wal")]
            None,
        );

        // Wait for at least one checkpoint cycle (200ms interval + margin)
        std::thread::sleep(Duration::from_millis(500));
        timer.stop();

        // Verify that a checkpoint happened (iteration > 0)
        let header = fm.active_header();
        assert!(
            header.iteration > 0,
            "expected at least one checkpoint, got iteration={}",
            header.iteration
        );
        assert_eq!(header.node_count, 1);
    }

    #[test]
    fn timer_skips_when_clean() {
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Arc::new(Catalog::new());
        let tm = Arc::new(TransactionManager::new());

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("clean_test.grafeo");
        let fm = Arc::new(GrafeoFileManager::create(&path).unwrap());

        let mut timer = CheckpointTimer::start(
            #[cfg(feature = "cdc")]
            Arc::new(crate::cdc::CdcLog::new()),
            Duration::from_millis(200),
            Arc::clone(&fm),
            Arc::clone(&store),
            Arc::clone(&catalog),
            Arc::clone(&tm),
            Arc::new(AtomicBool::new(false)),
            lpg_identity(),
            GraphModel::Lpg,
            #[cfg(feature = "triple-store")]
            Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
            #[cfg(feature = "triple-store")]
            Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new()),
            #[cfg(feature = "wal")]
            None,
        );

        std::thread::sleep(Duration::from_millis(500));
        timer.stop();

        // Just verify no crash occurred
        let header = fm.active_header();
        assert!(header.iteration <= 5);
    }

    #[cfg(all(feature = "lpg", feature = "vector-index", feature = "text-index"))]
    #[test]
    fn graph_scoped_indexes_survive_periodic_checkpoint_reopen_exactly() {
        use grafeo_common::storage::{Section, SectionType};
        use grafeo_common::types::{NodeId, PropertyKey, Value};
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, HnswIndex, PropertyVectorAccessor, VectorIndexKind,
        };
        use parking_lot::RwLock;

        fn install_indexes(
            store: &Arc<LpgStore>,
            catalog: &Catalog,
            path: grafeo_common::types::GraphPath,
            body: &str,
            embedding: [f32; 3],
            scope: &str,
        ) -> NodeId {
            let node = store.create_node(&["Doc"]);
            store.set_node_property(node, "body", Value::from(body));
            store.set_node_property(node, "embedding", Value::Vector(embedding.to_vec().into()));
            store.set_node_property(node, "scope", Value::from(scope));

            let vector = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                3,
                DistanceMetric::Cosine,
            ))));
            let accessor = PropertyVectorAccessor::new(store.as_ref(), "embedding");
            vector.insert(node, &embedding, &accessor);
            store.add_vector_index("Doc", "embedding", vector);

            let mut text = InvertedIndex::new(BM25Config::default());
            text.insert(node, body);
            store.add_text_index("Doc", "body", Arc::new(RwLock::new(text)));
            let label = catalog.get_or_create_label("Doc").expect("owner label");
            let embedding = catalog
                .get_or_create_property_key("embedding")
                .expect("owner property");
            let body_key = catalog
                .get_or_create_property_key("body")
                .expect("owner property");
            catalog
                .create_index(
                    Some(&format!("{scope}-vector")),
                    label,
                    embedding,
                    path.clone(),
                    crate::catalog::IndexConfiguration::Vector {
                        config: HnswConfig::new(3, DistanceMetric::Cosine),
                        quantization: grafeo_core::index::vector::QuantizationType::None,
                    },
                )
                .expect("real vector owner");
            catalog
                .create_index(
                    Some(&format!("{scope}-text")),
                    label,
                    body_key,
                    path,
                    crate::catalog::IndexConfiguration::Text {
                        config: BM25Config::default(),
                        min_token_length: 2,
                    },
                )
                .expect("real text owner");
            node
        }

        fn exact_index_images(store: Arc<LpgStore>) -> (Vec<u8>, Vec<u8>) {
            let cut = super::super::index_sections::LpgIndexGraphCut::capture(store)
                .expect("capture graph-qualified index cut");
            let vector = grafeo_core::index::vector::VectorStoreSection::from_views(
                cut.vector_views().expect("capture vector views"),
            )
            .serialize()
            .expect("serialize exact vector image");
            let text = grafeo_core::index::text::TextIndexSection::from_views(
                cut.text_views().expect("capture text views"),
            )
            .serialize()
            .expect("serialize exact text image");
            (vector, text)
        }

        let store = Arc::new(LpgStore::new().expect("default graph store"));
        let catalog = Arc::new(Catalog::new());
        let tm = Arc::new(TransactionManager::new());

        // Exercise a real, nonzero publication coordinate. A zero-only
        // fixture would not catch the historical periodic-checkpoint bug that
        // serialized Catalog at a hard-coded epoch while the header and
        // WorldCut used the transaction manager's actual frontier.
        tm.try_sync_epoch(EpochId::new(512)).unwrap();
        let epoch_transaction = tm.begin();
        let checkpoint_epoch = tm
            .commit(epoch_transaction)
            .expect("commit the periodic-checkpoint cut");
        assert_eq!(checkpoint_epoch, EpochId::new(513));
        store.sync_epoch(checkpoint_epoch);

        let default_node = install_indexes(
            &store,
            &catalog,
            grafeo_common::types::GraphPath::root(),
            "defaultneedle",
            [0.0, 1.0, 0.0],
            "default-graph",
        );

        assert!(store.create_graph("").expect("create empty-name graph"));
        let empty_named = store.graph("").expect("empty-name graph store");
        empty_named.sync_epoch(checkpoint_epoch);
        empty_named.create_node(&["Padding"]);
        let named_node = install_indexes(
            &empty_named,
            &catalog,
            grafeo_common::types::GraphPath::root()
                .child("")
                .expect("empty child path"),
            "namedneedle",
            [1.0, 0.0, 0.0],
            "empty-named-graph",
        );
        assert_ne!(default_node, named_node, "scope fixture IDs must differ");

        let expected_images = exact_index_images(Arc::clone(&store));
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("graph-scoped-periodic.grafeo");
        let fm = GrafeoFileManager::create(&path).unwrap();
        CheckpointTimer::try_checkpoint(
            #[cfg(feature = "cdc")]
            &crate::cdc::CdcLog::new(),
            &AtomicBool::new(false),
            &fm,
            &store,
            &catalog,
            &tm,
            &AtomicBool::new(false),
            &lpg_identity(),
            GraphModel::Lpg,
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new()),
            #[cfg(feature = "wal")]
            None,
        )
        .expect("publish periodic graph-qualified checkpoint");
        drop(fm);

        let persisted = GrafeoFileManager::open_read_only(&path).unwrap();
        let header = persisted.active_header();
        assert_eq!(
            header.epoch,
            checkpoint_epoch.as_u64(),
            "periodic checkpoint header must publish the captured transaction frontier"
        );
        let directory = persisted
            .read_section_directory()
            .unwrap()
            .expect("periodic checkpoint section directory");
        let catalog_entry = directory
            .find(SectionType::Catalog)
            .expect("Catalog section");
        let vector_entry = directory
            .find(SectionType::VectorStore)
            .expect("Vector Store section");
        let text_entry = directory
            .find(SectionType::TextIndex)
            .expect("Text Index section");
        assert_eq!(catalog_entry.version, 7);
        assert_eq!(vector_entry.version, 4);
        assert_eq!(text_entry.version, 5);

        let (verified_image, metadata, graph_model) =
            super::super::world_metadata::read_verified_container_image(&persisted, &directory)
                .expect("verify the complete periodic checkpoint image");
        assert_eq!(graph_model, GraphModelTag::Lpg);
        let metadata = metadata.expect("Catalog v7 periodic checkpoint requires WorldMetadata");
        assert_eq!(
            metadata.cut().epoch(),
            checkpoint_epoch,
            "WorldCut epoch must match the periodic checkpoint header"
        );
        let catalog_epoch = super::super::catalog_wire::graph_exact_catalog_epoch(
            super::super::world_metadata::find_section(&verified_image, SectionType::Catalog)
                .expect("verified Catalog section")
                .bytes(),
        )
        .expect("decode the exact Catalog v6 epoch");
        assert_eq!(
            catalog_epoch,
            checkpoint_epoch.as_u64(),
            "Catalog v6 epoch must match the periodic checkpoint header and WorldCut"
        );
        assert_eq!(
            (
                persisted.read_section_data(vector_entry).unwrap(),
                persisted.read_section_data(text_entry).unwrap(),
            ),
            expected_images,
            "periodic publication must use the exact graph-qualified cut"
        );
        drop(persisted);

        let reopened = crate::GrafeoDB::with_config(
            crate::Config::read_only(&path).with_graph_model(GraphModel::Lpg),
        )
        .expect("reopen periodic checkpoint");
        assert_eq!(
            reopened.current_epoch(),
            checkpoint_epoch,
            "reopen must install the persisted periodic-checkpoint frontier"
        );
        assert_eq!(
            reopened
                .world_cut()
                .expect("capture reopened world cut")
                .epoch(),
            checkpoint_epoch,
            "reopened WorldCut must retain the persisted checkpoint coordinate"
        );
        let restored_root = Arc::clone(crate::database::testing::root_lpg_store(&reopened));
        let restored_named = restored_root
            .graph("")
            .expect("restore the empty-name graph as named");
        assert_eq!(
            exact_index_images(Arc::clone(&restored_root)),
            expected_images,
            "reopen must preserve exact vector and text section state"
        );
        assert_eq!(
            restored_root.get_node_property(default_node, &PropertyKey::new("scope")),
            Some(Value::from("default-graph"))
        );
        assert_eq!(
            restored_named.get_node_property(named_node, &PropertyKey::new("scope")),
            Some(Value::from("empty-named-graph"))
        );

        let default_vector = restored_root
            .get_vector_index("Doc", "embedding")
            .expect("default vector index");
        let default_accessor = PropertyVectorAccessor::new(restored_root.as_ref(), "embedding");
        assert_eq!(
            default_vector.search(&[0.0, 1.0, 0.0], 1, &default_accessor)[0].0,
            default_node
        );
        let named_vector = restored_named
            .get_vector_index("Doc", "embedding")
            .expect("empty-name graph vector index");
        let named_accessor = PropertyVectorAccessor::new(restored_named.as_ref(), "embedding");
        assert_eq!(
            named_vector.search(&[1.0, 0.0, 0.0], 1, &named_accessor)[0].0,
            named_node
        );

        let default_text = restored_root
            .get_text_index("Doc", "body")
            .expect("default text index");
        assert_eq!(
            default_text.read().search("defaultneedle", 10)[0].0,
            default_node
        );
        assert!(default_text.read().search("namedneedle", 10).is_empty());
        let named_text = restored_named
            .get_text_index("Doc", "body")
            .expect("empty-name graph text index");
        assert_eq!(named_text.read().search("namedneedle", 10)[0].0, named_node);
        assert!(named_text.read().search("defaultneedle", 10).is_empty());
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn projection_only_registry_survives_periodic_checkpoint_reopen() {
        use grafeo_common::storage::{Section, SectionType};

        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Arc::new(Catalog::new());
        let tm = Arc::new(TransactionManager::new());
        let rdf_store = Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        let projections = Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());
        let projection_id = projections
            .declare("http://example.org/Person", "Person")
            .unwrap();
        assert!(rdf_store.is_empty());
        assert_eq!(rdf_store.graph_count(), 0);

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("projection-only.grafeo");
        let fm =
            GrafeoFileManager::create_with_graph_model(&path, GraphModel::Both.as_u8()).unwrap();
        CheckpointTimer::try_checkpoint(
            #[cfg(feature = "cdc")]
            &crate::cdc::CdcLog::new(),
            &AtomicBool::new(false),
            &fm,
            &store,
            &catalog,
            &tm,
            &AtomicBool::new(false),
            &identity(rdf_store.store_id()),
            GraphModel::Both,
            &rdf_store,
            &projections,
            #[cfg(feature = "wal")]
            None,
        )
        .unwrap();
        drop(fm);

        let reopened = GrafeoFileManager::open_read_only(&path).unwrap();
        let directory = reopened
            .read_section_directory()
            .unwrap()
            .expect("periodic checkpoint section directory");
        let entry = directory
            .find(SectionType::RdfStore)
            .expect("projection-only checkpoint must include RDF section");
        let data = reopened.read_section_data(entry).unwrap();

        let restored_store = Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        let restored_projections =
            Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new());
        let mut section = grafeo_core::graph::rdf::RdfStoreSection::with_projections(
            restored_store,
            Arc::clone(&restored_projections),
        );
        section.deserialize(&data).unwrap();
        let restored = restored_projections
            .get(projection_id)
            .expect("projection declaration restored from periodic checkpoint");
        assert_eq!(restored.type_iri(), "http://example.org/Person");
        assert_eq!(restored.node_label(), "Person");
    }

    #[test]
    fn durability_poison_prevents_periodic_container_publication() {
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Arc::new(Catalog::new());
        let tm = Arc::new(TransactionManager::new());
        store.create_node(&["MustRemainUncheckpointed"]);

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("poisoned-periodic-checkpoint.grafeo");
        let fm = GrafeoFileManager::create(&path).unwrap();
        let iteration_before = fm.active_header().iteration;
        let durability_poisoned = AtomicBool::new(true);

        let error = CheckpointTimer::try_checkpoint(
            #[cfg(feature = "cdc")]
            &crate::cdc::CdcLog::new(),
            &AtomicBool::new(false),
            &fm,
            &store,
            &catalog,
            &tm,
            &durability_poisoned,
            &lpg_identity(),
            GraphModel::Lpg,
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfLpgProjectionRegistry::new()),
            #[cfg(feature = "wal")]
            None,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Transaction(TransactionError::DurabilityFailure(_))
        ));
        assert_eq!(fm.active_header().iteration, iteration_before);
        assert!(fm.read_section_directory().unwrap().is_none());
    }
}

#[cfg(all(test, feature = "cdc", feature = "gql", feature = "wal"))]
#[test]
fn periodic_checkpoint_captures_the_actual_retained_feed() {
    use super::GrafeoDB;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retained.grafeo");
    let cfg = crate::Config::persistent(&path).with_cdc();
    let db = GrafeoDB::with_config(cfg.clone()).unwrap();
    db.session().execute("INSERT (:PeriodicFeed)").unwrap();
    let expected = db
        .changes_between(
            grafeo_common::types::EpochId::INITIAL,
            grafeo_common::types::EpochId::PENDING,
        )
        .unwrap();
    assert_eq!(expected.len(), 1);
    CheckpointTimer::try_checkpoint(
        &db.cdc_log,
        &AtomicBool::new(false),
        db.file_manager.as_ref().unwrap(),
        db.store_arc(),
        &db.catalog,
        &db.transaction_manager,
        &db.durability_poisoned,
        &db.world_identity,
        db.config.graph_model,
        #[cfg(feature = "triple-store")]
        &db.rdf_store,
        #[cfg(feature = "triple-store")]
        &db.rdf_projections,
        db.wal.as_deref(),
    )
    .unwrap();
    let fm = db.file_manager.as_ref().unwrap();
    let directory = fm.read_section_directory().unwrap().unwrap();
    let (image, _, _) =
        super::world_metadata::read_verified_container_image(fm, &directory).unwrap();
    let section =
        super::world_metadata::find_section(&image, grafeo_common::storage::SectionType::Cdc)
            .unwrap();
    assert_eq!(section.bytes()[8], 1);
    let restored = crate::cdc::CdcLog::new();
    super::cdc_checkpoint::prepare(section.bytes(), db.store_id(), db.current_epoch())
        .unwrap()
        .install(&restored);
    assert_eq!(
        serde_json::to_value(restored.changes_between(
            grafeo_common::types::EpochId::INITIAL,
            grafeo_common::types::EpochId::PENDING
        ))
        .unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    db.close().unwrap();
}
