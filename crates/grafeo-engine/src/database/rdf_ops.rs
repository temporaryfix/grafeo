//! RDF-specific operations for GrafeoDB.
//!
//! This module consolidates all RDF functionality that was previously scattered
//! across `query.rs`, `crud.rs`, `admin.rs`, and `mod.rs`. The entire module
//! is gated behind `#[cfg(feature = "triple-store")]` in the parent.

use std::sync::Arc;

use grafeo_common::utils::error::Result;
use grafeo_core::graph::rdf::RdfStore;

use super::GrafeoDB;

/// Latest epoch at which the default RDF graph's materialized state changed.
///
/// The RDF store commit clock follows the shared database clock, so an LPG-only
/// commit also advances it. Projection lag must instead compare RDF data
/// revisions; otherwise every successful projection transaction would make
/// itself immediately appear one epoch stale.
#[cfg(feature = "lpg")]
fn rdf_data_revision(store: &RdfStore) -> grafeo_common::types::EpochId {
    let mut revision = grafeo_common::types::EpochId::new(0);
    for (_, lives) in store.quad_history() {
        for life in lives {
            revision = revision.max(life.tx.from());
            if !life.tx.is_open() {
                revision = revision.max(life.tx.to());
            }
        }
    }
    revision
}

// =========================================================================
// Query operations
// =========================================================================

impl GrafeoDB {
    /// Executes a SPARQL query and returns the result.
    ///
    /// SPARQL queries operate on the RDF triple store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::{Config, GrafeoDB, GraphModel};
    ///
    /// let db = GrafeoDB::with_config(
    ///     Config::in_memory().with_graph_model(GraphModel::Rdf),
    /// )?;
    /// let result = db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "sparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<super::QueryResult> {
        // Session is the single boundary for GraphModel checks, durability
        // poison, isolation policy, EXPLAIN/PROFILE, and one-statement commit.
        self.session().execute_sparql(query)
    }

    /// Returns the underlying RDF store.
    ///
    /// # Consistency and durability
    ///
    /// This is a low-level, uncoordinated read handle intended for inspection
    /// and engine integrations. Reads do not acquire the database publication
    /// barrier. On a sealed database every authoritative mutator requires the
    /// owning Session's private store scope; hostile raw calls and foreign
    /// authorities fail closed. Application code should write through
    /// [`GrafeoDB::session`](super::GrafeoDB::session),
    /// [`execute_sparql`](Self::execute_sparql), or the RDF CRUD methods below.
    #[must_use]
    pub fn rdf_store(&self) -> &Arc<RdfStore> {
        &self.rdf_store
    }
}

// =========================================================================
// CRUD operations
// =========================================================================

impl GrafeoDB {
    /// Batch-inserts RDF triples through the Session/WAL mutation chokepoint.
    ///
    /// Autocommit one transaction. Duplicates are skipped. On a persistent
    /// database this logs WAL and only returns `Ok` after commit.
    ///
    /// Returns the number of triples that were newly inserted.
    ///
    /// # Errors
    ///
    /// Returns an error if the session is poisoned, RDF is not enabled for this
    /// database, or WAL/commit fails.
    pub fn batch_insert_rdf(
        &self,
        triples: impl IntoIterator<Item = grafeo_core::graph::rdf::Triple>,
    ) -> Result<usize> {
        self.session().insert_rdf_batch(triples)
    }

    /// Autocommit batch of named-graph quads. Returns `(inserted, commit epoch)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the session is poisoned, RDF is not enabled, or commit fails.
    pub fn insert_rdf_quads(
        &self,
        quads: impl IntoIterator<Item = grafeo_core::graph::rdf::Quad>,
    ) -> Result<(usize, grafeo_common::types::EpochId)> {
        let mut session = self.session();
        session.begin_transaction()?;
        let n = session.insert_rdf_quads(quads)?;
        let epoch = session.commit()?;
        Ok((n, epoch))
    }

    /// Exact typed-quad membership against committed state (no session tx).
    ///
    /// This infallible compatibility method returns `false` on a model,
    /// lifecycle, poison, or SSI-tracking error. New callers should use
    /// [`try_contains_rdf_quad`](Self::try_contains_rdf_quad).
    #[must_use]
    pub fn contains_rdf_quad(&self, quad: &grafeo_core::graph::rdf::Quad) -> bool {
        self.session().contains_rdf_quad(quad)
    }

    /// Fallible exact typed-quad membership against committed state.
    ///
    /// # Errors
    ///
    /// Returns an error when RDF is not enabled or the session cannot establish
    /// the lifecycle, poison, publication, or Serializable read guarantees.
    pub fn try_contains_rdf_quad(&self, quad: &grafeo_core::graph::rdf::Quad) -> Result<bool> {
        self.session().try_contains_rdf_quad(quad)
    }

    /// RDF store commit epoch used for transaction-time stamps.
    #[must_use]
    pub fn rdf_store_commit_epoch(&self) -> grafeo_common::types::EpochId {
        let _publication = self.transaction_manager.publication().read();
        self.rdf_store.commit_epoch()
    }

    /// Compatibility insert using a legacy microsecond valid-time coordinate.
    ///
    /// The bounds are promoted exactly by `×1000`; new code should use
    /// [`insert_rdf_valid_tai_ns`](Self::insert_rdf_valid_tai_ns).
    ///
    /// # Errors
    ///
    /// Returns an error if insert or WAL commit fails.
    pub fn insert_rdf_valid(
        &self,
        triples: impl IntoIterator<Item = grafeo_core::graph::rdf::Triple>,
        valid_from: i64,
        valid_to: i64,
    ) -> Result<usize> {
        let valid =
            grafeo_common::types::ValidTimeInterval::from_legacy_micros(valid_from, valid_to)
                .map_err(|error| {
                    grafeo_common::utils::error::Error::InvalidValue(error.to_string())
                })?;
        self.session()
            .insert_rdf_batch_with_valid(triples, Some(valid))
    }

    /// Inserts RDF triples with a canonical signed TAI-nanosecond valid-time
    /// interval `[valid_from_tai_ns, valid_to_tai_ns)`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an empty/inverted interval, or the
    /// underlying Session/WAL error if publication fails.
    pub fn insert_rdf_valid_tai_ns(
        &self,
        triples: impl IntoIterator<Item = grafeo_core::graph::rdf::Triple>,
        valid_from_tai_ns: i128,
        valid_to_tai_ns: i128,
    ) -> Result<usize> {
        let valid = grafeo_common::types::ValidTimeInterval::from_tai_nanoseconds(
            valid_from_tai_ns,
            valid_to_tai_ns,
        )
        .map_err(|error| grafeo_common::utils::error::Error::InvalidValue(error.to_string()))?;
        self.session()
            .insert_rdf_batch_with_valid(triples, Some(valid))
    }

    /// Snapshot RDF literal search (same graph SPARQL would see).
    ///
    /// Scores subjects whose literal objects contain all query tokens (BM25-shaped
    /// bag-of-words; not a second index kernel).
    #[must_use]
    pub fn search_rdf(&self, query: &str) -> Vec<grafeo_core::graph::rdf::Term> {
        let _publication = self.transaction_manager.publication().read();
        let tokens: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut hits = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for t in self.rdf_store.triples() {
            let grafeo_core::graph::rdf::Term::Literal(lit) = t.object() else {
                continue;
            };
            let hay = lit.value().to_lowercase();
            if tokens.iter().all(|tok| hay.contains(tok.as_str()))
                && seen.insert(t.subject().clone())
            {
                hits.push(t.subject().clone());
            }
        }
        hits
    }

    /// Declares a durable, versioned RDF→LPG projection. IRIs are not identified
    /// with LPG nodes until
    /// [`rebuild_rdf_lpg_projection`](Self::rebuild_rdf_lpg_projection).
    /// Repeating an identical declaration returns the same content-derived id.
    #[cfg(feature = "lpg")]
    ///
    /// # Errors
    ///
    /// Returns an error without installing the declaration if its WAL metadata
    /// cannot be written, if the database is closed/read-only or durability
    /// poisoned, or if a content-id collision is detected. The record requests
    /// the configured durability barrier; an immediate fsync is guaranteed only
    /// when WAL durability is `Sync`.
    pub fn declare_rdf_lpg_projection(&self, type_iri: &str, node_label: &str) -> Result<u64> {
        self.declare_rdf_lpg_projection_for_graph(None, type_iri, node_label)
    }

    /// Declares a durable RDF→LPG projection over one logical RDF graph.
    ///
    /// `source_graph = None` selects the default graph. A named declaration is
    /// bound to the logical graph name; every publication receipt additionally
    /// pins the exact incarnation that was materialized, so drop/recreate never
    /// aliases two graph lifetimes.
    ///
    /// # Errors
    ///
    /// Returns an error without installing the declaration if its fields are
    /// invalid, its full mapping digest collides with an existing compatibility
    /// id, or durable metadata publication fails.
    #[cfg(feature = "lpg")]
    pub fn declare_rdf_lpg_projection_for_graph(
        &self,
        source_graph: Option<&str>,
        type_iri: &str,
        node_label: &str,
    ) -> Result<u64> {
        use grafeo_common::utils::error::{Error, TransactionError};
        use grafeo_core::graph::rdf::RdfLpgProjectionDefinition;

        // Global lock order: projection rebuild mutex -> lifecycle ->
        // publication. Holding the lifecycle read through the short metadata
        // publication prevents close from snapshotting an acknowledged WAL
        // declaration without its in-memory registry post-image.
        let _registry = self.rdf_projections.lock_rebuild();
        let lifecycle = Arc::clone(&self.is_open);
        let database_open = lifecycle.read();
        if !*database_open {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "cannot declare an RDF→LPG projection on a closed database".into(),
            )));
        }
        if self.read_only {
            return Err(Error::Transaction(TransactionError::ReadOnly));
        }
        if self.is_durability_poisoned() {
            return Err(Error::Transaction(TransactionError::DurabilityFailure(
                "cannot declare an RDF→LPG projection on a durability-poisoned database; reopen and recover first"
                    .into(),
            )));
        }

        let definition =
            RdfLpgProjectionDefinition::new_for_graph(source_graph, type_iri, node_label)
                .map_err(Error::InvalidValue)?;
        let id = definition.id();
        let _publication = self.transaction_manager.publication().write();
        // A concurrent WAL failure can poison the shared database while this
        // caller waits for publication authority.
        if self.is_durability_poisoned() {
            return Err(Error::Transaction(TransactionError::DurabilityFailure(
                "cannot declare an RDF→LPG projection on a durability-poisoned database; reopen and recover first"
                    .into(),
            )));
        }
        if let Some(existing) = self.rdf_projections.get(id) {
            if existing.mapping_digest() == definition.mapping_digest() {
                return Ok(id);
            }
            return Err(Error::Transaction(TransactionError::InvalidState(format!(
                "RDF→LPG projection compatibility id {id} collides between distinct full mapping digests"
            ))));
        }

        self.transaction_manager.with_write_authority(|| {
            let epoch = self.transaction_manager.reserve_publication_epoch()?;
            #[cfg(feature = "wal")]
            self.log_wal(
                &grafeo_storage::wal::WalRecord::RdfLpgProjectionDeclaredV3 {
                    projection_id: id,
                    mapping_digest: definition.mapping_digest(),
                    mapping_format_version: definition.mapping_format_version(),
                    source_graph: source_graph.map(str::to_owned),
                    type_iri: type_iri.to_owned(),
                    node_label: node_label.to_owned(),
                    epoch,
                },
            )?;

            self.rdf_projections
                .install_definition_v3(
                    id,
                    definition.mapping_digest(),
                    definition.mapping_format_version(),
                    source_graph,
                    type_iri,
                    node_label,
                )
                .map_err(|message| Error::Transaction(TransactionError::InvalidState(message)))?;

            // A standalone metadata epoch participates in the same global cut
            // as both native models. This keeps subsequent transaction stamps,
            // snapshot headers, and point-in-time selection totally ordered.
            if let Some(store) = &self.store {
                store.sync_epoch(epoch);
                for graph in store.named_graph_entries().values() {
                    graph.sync_epoch(epoch);
                }
            }
            if let Err(error) = self.rdf_store.try_set_commit_epoch(epoch) {
                self.durability_poisoned
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(Error::Transaction(TransactionError::DurabilityFailure(
                    format!(
                        "durable RDF→LPG projection declaration could not advance the RDF clock: {error}; reopen and recover before continuing"
                    ),
                )));
            }
            self.transaction_manager.publish_reserved_epoch(epoch);
            Ok::<(), Error>(())
        })?;
        Ok(id)
    }

    /// Convenience wrapper for a projection sourced from one named RDF graph.
    ///
    /// # Errors
    ///
    /// See [`declare_rdf_lpg_projection_for_graph`](Self::declare_rdf_lpg_projection_for_graph).
    #[cfg(feature = "lpg")]
    pub fn declare_named_rdf_lpg_projection(
        &self,
        source_graph: &str,
        type_iri: &str,
        node_label: &str,
    ) -> Result<u64> {
        self.declare_rdf_lpg_projection_for_graph(Some(source_graph), type_iri, node_label)
    }

    /// Returns the durable definition and last successful rebuild status.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn rdf_lpg_projection(
        &self,
        projection_id: u64,
    ) -> Option<grafeo_core::graph::rdf::RdfLpgProjectionDefinition> {
        self.rdf_projections.get(projection_id)
    }

    /// Reconciles LPG nodes with one pinned RDF `rdf:type` source cut.
    ///
    /// Projection-owned nodes are identified by a reserved ownership property.
    /// Existing rows are reused, duplicates and rows absent from the source cut
    /// are deleted, and missing rows are created. Every target change is applied
    /// in one explicit LPG transaction. Rebuild status advances only after that
    /// transaction commits successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if `projection_id` is unknown or the target transaction
    /// cannot be committed.
    #[cfg(feature = "lpg")]
    pub fn rebuild_rdf_lpg_projection(&self, projection_id: u64) -> Result<usize> {
        use std::collections::{BTreeMap, BTreeSet};

        use grafeo_common::types::{PropertyKey, Value};
        use grafeo_common::utils::error::{Error, TransactionError};
        use grafeo_core::graph::rdf::{
            RDF_LPG_PROJECTION_IRI_PROPERTY, RDF_LPG_PROJECTION_OWNER_PROPERTY, Term,
        };

        // Rebuild serialization is the outermost lock. Lifecycle checks are
        // deliberately short and never span Session begin/commit.
        let _rebuild = self.rdf_projections.lock_rebuild();
        {
            let lifecycle = Arc::clone(&self.is_open);
            let database_open = lifecycle.read();
            if !*database_open {
                return Err(Error::Transaction(TransactionError::InvalidState(
                    "cannot rebuild an RDF→LPG projection on a closed database".into(),
                )));
            }
            if self.read_only {
                return Err(Error::Transaction(TransactionError::ReadOnly));
            }
            if self.is_durability_poisoned() {
                return Err(Error::Transaction(TransactionError::DurabilityFailure(
                    "cannot rebuild an RDF→LPG projection on a durability-poisoned database; reopen and recover first"
                        .into(),
                )));
            }
        }
        let definition = self.rdf_projections.get(projection_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(format!(
                "unknown RDF→LPG projection {projection_id}"
            )))
        })?;
        let generation = definition.generation().checked_add(1).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(format!(
                "RDF→LPG projection {projection_id} exhausted its generation counter"
            )))
        })?;

        // Pin the RDF commit epoch and derive the complete desired key set while
        // holding the cross-model publication barrier. `triples_at_epoch`
        // remains a stable source cut after the guard is released.
        let (source_graph, source_epoch, desired): (_, _, BTreeSet<String>) = {
            let _publication = self.transaction_manager.publication().read();
            let source_cut = self.rdf_store.commit_epoch();
            let (source_graph, source_store) = match definition.source_graph() {
                None => (
                    grafeo_common::types::ProjectionSourceGraph::default_graph(),
                    Arc::clone(&self.rdf_store),
                ),
                Some(name) => {
                    let store = self.rdf_store.graph(name).ok_or_else(|| {
                        Error::Transaction(TransactionError::InvalidState(format!(
                            "RDF→LPG projection {projection_id} source graph {name:?} does not exist"
                        )))
                    })?;
                    let graph = grafeo_common::types::ProjectionSourceGraph::named(
                        name.to_owned(),
                        store.graph_incarnation(),
                    )
                    .map_err(|error| Error::InvalidValue(error.to_string()))?;
                    (graph, store)
                }
            };
            let type_predicate = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
            let type_object = Term::iri(definition.type_iri());
            let desired = source_store
                .triples_at_epoch(source_cut)
                .into_iter()
                .filter(|triple| {
                    triple.predicate() == &type_predicate && triple.object() == &type_object
                })
                .filter_map(|triple| triple.subject().as_iri().map(|iri| iri.as_str().to_owned()))
                .collect();
            (source_graph, source_cut, desired)
        };
        let row_count = u64::try_from(desired.len()).map_err(|_| {
            Error::Transaction(TransactionError::InvalidState(format!(
                "RDF→LPG projection {projection_id} source row count exceeds u64"
            )))
        })?;

        let owner_marker = definition.owner_marker();
        let owner_key = PropertyKey::new(RDF_LPG_PROJECTION_OWNER_PROPERTY);
        let iri_key = PropertyKey::new(RDF_LPG_PROJECTION_IRI_PROPERTY);
        let mut retained = BTreeMap::new();
        let mut stale = Vec::new();

        // Only rows carrying this projection's reserved marker are in scope.
        // User-authored nodes, including nodes with a coincidentally equal IRI,
        // are deliberately untouched.
        {
            let _publication = self.transaction_manager.publication().read();
            let view = self.read_graph_view();
            for node_id in view.node_ids() {
                let Some(node) = view.get_node(node_id) else {
                    continue;
                };
                let Some(found_marker) = node.properties.get(&owner_key).and_then(Value::as_str)
                else {
                    continue;
                };
                if !definition.owner_marker_matches(found_marker) {
                    continue;
                }
                let Some(iri) = node.properties.get(&iri_key).and_then(Value::as_str) else {
                    stale.push(node_id);
                    continue;
                };
                let canonical = found_marker == owner_marker
                    && node.labels.len() == 1
                    && node.labels[0].as_str() == definition.node_label()
                    && node.properties.len() == 2;
                if !canonical || !desired.contains(iri) || retained.contains_key(iri) {
                    stale.push(node_id);
                    continue;
                }
                retained.insert(iri.to_owned(), node_id);
            }
        }

        let mut session = self.session();
        session.authorize_rdf_projection_rebuild_v3(
            projection_id,
            owner_marker.clone(),
            definition.node_label().to_owned(),
            desired.clone(),
            Arc::clone(&self.rdf_projections),
            self.rdf_store.store_id(),
            definition.mapping_digest(),
            source_graph,
            source_epoch,
            generation,
            row_count,
        )?;
        session.begin_transaction()?;
        let apply = (|| -> Result<grafeo_common::types::EpochId> {
            for node_id in stale {
                if !session.delete_node(node_id) {
                    return Err(Error::Transaction(TransactionError::InvalidState(format!(
                        "projection-owned node {node_id} disappeared during rebuild"
                    ))));
                }
            }
            for iri in desired.iter().filter(|iri| !retained.contains_key(*iri)) {
                session.create_node_with_props(
                    &[definition.node_label()],
                    [
                        (RDF_LPG_PROJECTION_IRI_PROPERTY, Value::from(iri.clone())),
                        (
                            RDF_LPG_PROJECTION_OWNER_PROPERTY,
                            Value::from(owner_marker.clone()),
                        ),
                    ],
                )?;
            }
            session.commit()
        })();

        match apply {
            Ok(_) => {}
            Err(error) => {
                if session.in_transaction()
                    && let Err(rollback_error) = session.rollback()
                {
                    return Err(Error::Transaction(TransactionError::DurabilityFailure(
                        format!(
                            "RDF→LPG projection rebuild failed ({error}); rollback also failed \
                         ({rollback_error}); reopen and recover before further mutation"
                        ),
                    )));
                }
                return Err(error);
            }
        }
        Ok(desired.len())
    }

    /// Epoch lag between RDF commit clock and last rebuild (0 if current).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn rdf_projection_lag(&self, projection_id: u64) -> Option<u64> {
        let _publication = self.transaction_manager.publication().read();
        let definition = self.rdf_projections.get(projection_id)?;
        let rebuilt = definition.last_source_epoch()?;
        let (source_store, incarnation_changed) = match definition.source_graph() {
            None => (Arc::clone(&self.rdf_store), false),
            Some(name) => match self.rdf_store.graph(name) {
                Some(store) => {
                    let changed = definition.receipt().is_none_or(|receipt| {
                        receipt.source_graph().incarnation() != store.graph_incarnation()
                    });
                    (store, changed)
                }
                None => return Some(1),
            },
        };
        let epoch_lag = rdf_data_revision(&source_store)
            .as_u64()
            .saturating_sub(rebuilt.as_u64());
        Some(if incarnation_changed {
            epoch_lag.max(1)
        } else {
            epoch_lag
        })
    }
}

// =========================================================================
// Authoritative RDF dataset history
// =========================================================================

impl GrafeoDB {
    fn capture_rdf_dataset_history(&self) -> Result<grafeo_core::graph::rdf::RdfDatasetHistory> {
        // Global order matches commit/checkpoint: RDF commit gate, then the
        // cross-model publication barrier. The under-gate capture cannot mix
        // partitions from two commits and does not recursively lock.
        let _rdf_gate = self.rdf_store.lock_commit();
        let _publication = self.transaction_manager.publication().read();
        self.rdf_store
            .dataset_history_under_commit_gate()
            .map_err(|error| {
                grafeo_common::utils::error::Error::Storage(
                    grafeo_common::utils::error::StorageError::Corruption(format!(
                        "invalid persisted RDF dataset history: {error}"
                    )),
                )
            })
    }

    /// Captures the authoritative graph-qualified RDF interval history.
    ///
    /// The result includes store identity, named-graph create/drop/incarnation
    /// lifetimes, lossless typed quads, full 256-bit statement handles,
    /// transaction and valid time, and an explicit legacy completeness bound.
    /// It is one coherent committed publication cut.
    ///
    /// # Errors
    ///
    /// Returns a storage-corruption error if persisted history invariants do
    /// not validate. Legacy incompleteness is represented in the returned value.
    pub fn rdf_dataset_history(&self) -> Result<grafeo_core::graph::rdf::RdfDatasetHistory> {
        self.capture_rdf_dataset_history()
    }

    /// Returns the graph-qualified typed RDF dataset visible at `epoch`.
    ///
    /// The history readers return typed cuts and ordered transitions:
    ///
    /// ```
    /// use grafeo_common::types::{EpochId, TaiNanoseconds};
    /// use grafeo_common::utils::error::Result;
    /// use grafeo_core::graph::rdf::{RdfHistoryCut, RdfHistoryDiff};
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let _: fn(&GrafeoDB, EpochId) -> Result<RdfHistoryCut> = GrafeoDB::rdf_history_cut;
    /// let _: fn(&GrafeoDB, EpochId, Option<TaiNanoseconds>) -> Result<RdfHistoryCut> =
    ///     GrafeoDB::rdf_history_cut_at;
    /// let _: fn(&GrafeoDB, EpochId, EpochId) -> Result<RdfHistoryDiff> =
    ///     GrafeoDB::rdf_history_diff;
    /// ```
    ///
    /// The former default-graph-only read method is unavailable:
    ///
    /// ```compile_fail,E0599
    /// use grafeo_engine::GrafeoDB;
    /// let _ = GrafeoDB::rdf_triples_at_epoch;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an explicit error when `epoch` predates a legacy history's
    /// truthful completeness boundary.
    pub fn rdf_history_cut(
        &self,
        epoch: grafeo_common::types::EpochId,
    ) -> Result<grafeo_core::graph::rdf::RdfHistoryCut> {
        self.capture_rdf_dataset_history()?
            .cut(epoch)
            .map_err(rdf_history_query_error)
    }

    /// Returns an RDF dataset cut composed across transaction and valid time.
    ///
    /// `valid_at = None` applies no valid-time filter. Versions without an
    /// application valid-time interval are always valid. A supplied instant is
    /// a signed, lossless TAI-nanosecond coordinate and applies to named as well
    /// as default graph quads at the historical transaction cut.
    ///
    /// The former current-transaction and microsecond read methods are unavailable:
    ///
    /// ```compile_fail,E0599
    /// use grafeo_engine::GrafeoDB;
    /// let _ = GrafeoDB::rdf_triples_at_valid_time;
    /// ```
    ///
    /// ```compile_fail,E0599
    /// use grafeo_engine::GrafeoDB;
    /// let _ = GrafeoDB::rdf_triples_at_valid;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an explicit error for a cut outside truthful history coverage.
    pub fn rdf_history_cut_at(
        &self,
        epoch: grafeo_common::types::EpochId,
        valid_at: Option<grafeo_common::types::TaiNanoseconds>,
    ) -> Result<grafeo_core::graph::rdf::RdfHistoryCut> {
        self.capture_rdf_dataset_history()?
            .cut_at(epoch, valid_at)
            .map_err(rdf_history_query_error)
    }

    /// Returns canonical lifecycle and statement transitions in `(from, through]`.
    ///
    /// Ordering is stable within an atomic commit: statement retract, graph
    /// drop, graph create, statement assert; ties are ordered by graph
    /// incarnation and full statement handle. This makes a same-epoch graph
    /// replacement consumable without exposing overlapping incarnations.
    ///
    /// The former unordered default-graph-only diff method is unavailable:
    ///
    /// ```compile_fail,E0599
    /// use grafeo_engine::GrafeoDB;
    /// let _ = GrafeoDB::rdf_diff;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an explicit error for inverted or incomplete history ranges.
    pub fn rdf_history_diff(
        &self,
        from: grafeo_common::types::EpochId,
        through: grafeo_common::types::EpochId,
    ) -> Result<grafeo_core::graph::rdf::RdfHistoryDiff> {
        self.capture_rdf_dataset_history()?
            .ordered_diff(from, through)
            .map_err(rdf_history_query_error)
    }

    /// Returns a stable page of durable RDF CDC derived from persisted history.
    ///
    /// Keep `from` fixed while paging transitions through a fixed `through`.
    /// Preserve the returned cursor even after `has_more` becomes false: the
    /// same cursor can resume later with a larger upper bound. Cursors are
    /// store-bound and forged or cross-store cursors fail closed.
    ///
    /// # Errors
    ///
    /// Returns an explicit error for invalid bounds, zero page size, legacy
    /// incompleteness, or an invalid cursor.
    pub fn rdf_cdc_page(
        &self,
        from: grafeo_common::types::EpochId,
        through: grafeo_common::types::EpochId,
        after: Option<&grafeo_core::graph::rdf::RdfHistoryCursor>,
        limit: usize,
    ) -> Result<grafeo_core::graph::rdf::RdfCdcPage> {
        self.capture_rdf_dataset_history()?
            .cdc_page(from, through, after, limit)
            .map_err(rdf_history_query_error)
    }

    /// Resolves the stable full-width handle for a typed quad and graph lifetime.
    ///
    /// The v1 domain-separated BLAKE3 identity includes this store's portable
    /// identity, graph IRI, graph incarnation, and lossless typed RDF terms.
    /// A canonical alias of a live statement resolves to its retained spelling.
    /// Snapshot transfer that preserves store identity preserves handles;
    /// explicit store forks and DROP/CREATE graph reincarnations change them.
    ///
    /// # Errors
    ///
    /// Returns an error if graph kind and incarnation disagree.
    pub fn rdf_statement_handle(
        &self,
        quad: &grafeo_core::graph::rdf::Quad,
        graph_incarnation: grafeo_common::types::GraphIncarnationId,
    ) -> Result<grafeo_common::types::StatementHandle> {
        let _rdf_gate = self.rdf_store.lock_commit();
        let _publication = self.transaction_manager.publication().read();
        // A live canonical alias names the retained lossless representative.
        // An absent fact or unmatched graph incarnation uses the supplied spelling.
        // Historical callers retain the handle recorded on RdfHistoricalQuad.
        let target = match quad.graph() {
            Some(name) => self.rdf_store.graph(name),
            None => Some(Arc::clone(&self.rdf_store)),
        };
        let representative = target
            .filter(|target| target.graph_incarnation() == graph_incarnation)
            .and_then(|target| {
                let triple = quad.triple();
                target
                    .find(&grafeo_core::graph::rdf::TriplePattern {
                        subject: Some(triple.subject().clone()),
                        predicate: Some(triple.predicate().clone()),
                        object: Some(triple.object().clone()),
                    })
                    .into_iter()
                    .next()
            })
            .map(|triple| match quad.graph() {
                Some(name) => grafeo_core::graph::rdf::Quad::named(triple.as_ref().clone(), name),
                None => grafeo_core::graph::rdf::Quad::new(triple.as_ref().clone()),
            });
        grafeo_core::graph::rdf::statement_handle(
            self.rdf_store.store_id(),
            representative.as_ref().unwrap_or(quad),
            graph_incarnation,
        )
        .map_err(rdf_history_query_error)
    }
}

fn rdf_history_query_error(
    error: grafeo_core::graph::rdf::RdfHistoryError,
) -> grafeo_common::utils::error::Error {
    grafeo_common::utils::error::Error::InvalidValue(error.to_string())
}

// =========================================================================
// Admin operations
// =========================================================================

impl GrafeoDB {
    /// Returns RDF schema information.
    ///
    /// Only available when the RDF feature is enabled.
    #[must_use]
    pub fn rdf_schema(&self) -> crate::admin::SchemaInfo {
        let _publication = self.transaction_manager.publication().read();
        let stats = self.rdf_store.stats();

        let predicates = self
            .rdf_store
            .predicates()
            .into_iter()
            .map(|predicate| {
                let count = self.rdf_store.triples_with_predicate(&predicate).len();
                crate::admin::PredicateInfo {
                    iri: predicate.to_string(),
                    count,
                }
            })
            .collect();

        crate::admin::SchemaInfo::Rdf(crate::admin::RdfSchemaInfo {
            predicates,
            named_graphs: Vec::new(),
            subject_count: stats.subject_count,
            object_count: stats.object_count,
        })
    }
}

// =========================================================================
// SHACL validation
// =========================================================================

#[cfg(feature = "shacl")]
impl GrafeoDB {
    /// Validates the default graph against SHACL shapes in a named graph.
    ///
    /// # Errors
    ///
    /// Returns an error if shape parsing fails or the shapes graph doesn't exist.
    pub fn validate_shacl(
        &self,
        shapes_graph: &str,
    ) -> grafeo_common::utils::error::Result<grafeo_core::graph::rdf::shacl::ValidationReport> {
        let session = self.session();
        session.validate_shacl(shapes_graph)
    }
}

// =========================================================================
// WAL replay helper
// =========================================================================

/// Replays committed RDF WAL records into the RDF store.
///
/// RDF mutations precede their commit marker in the WAL. Recovery therefore
/// resolves every transaction's durable commit epoch before applying any
/// mutation, so the reconstructed transaction-time intervals are identical to
/// the live commit rather than depending on replay order or a mutable epoch.
#[cfg(feature = "wal")]
pub(super) fn replay_rdf_wal_records(
    rdf_store: &Arc<RdfStore>,
    records: &[grafeo_storage::wal::WalRecord],
) -> Result<()> {
    use std::collections::{HashMap, VecDeque};

    use grafeo_common::types::{EpochId, GraphIncarnationId, TransactionId};
    use grafeo_common::utils::error::StorageError;
    use grafeo_core::graph::rdf::{Quad, RdfGraphIdentity, Term};
    use grafeo_storage::wal::{WalEntry, WalRecord};

    // This helper is also used against an already-live store. Validate every
    // decoded record before deriving commit metadata or applying standalone
    // high-water records, so a corrupt tail cannot publish an observable
    // prefix. In particular, Committed/EpochAdvance must never carry PENDING.
    for record in records {
        record
            .validate_recovery()
            .map_err(StorageError::InvalidWalEntry)?;
    }

    // Current writers put the epoch directly on `Committed`. Pre-G3 writers
    // used `TransactionCommit` followed by an untagged `EpochAdvance`; retain
    // that on-disk compatibility by pairing those records in commit order.
    let mut commit_epochs = HashMap::<TransactionId, EpochId>::new();
    let mut legacy_commits = VecDeque::<TransactionId>::new();
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
            }
            WalRecord::TransactionCommit { transaction_id } => {
                legacy_commits.push_back(*transaction_id);
            }
            WalRecord::EpochAdvance { epoch } => {
                if let Some(transaction_id) = legacy_commits.pop_front() {
                    commit_epochs.entry(transaction_id).or_insert(*epoch);
                }
            }
            _ => {}
        }
    }

    let parse_triple = |subject: &str, predicate: &str, object: &str| {
        let invalid = || {
            StorageError::InvalidWalEntry(format!(
                "invalid RDF N-Triples terms: subject={subject:?}, predicate={predicate:?}, object={object:?}"
            ))
        };
        let subject = Term::from_ntriples(subject).ok_or_else(invalid)?;
        let predicate = Term::from_ntriples(predicate).ok_or_else(invalid)?;
        let object = Term::from_ntriples(object).ok_or_else(invalid)?;
        Ok::<_, StorageError>(grafeo_core::graph::rdf::Triple::new(
            subject, predicate, object,
        ))
    };
    let commit_epoch = |transaction_id: TransactionId| {
        commit_epochs.get(&transaction_id).copied().ok_or_else(|| {
            StorageError::InvalidWalEntry(format!(
                "committed RDF mutation for transaction {} has no durable commit epoch",
                transaction_id.as_u64()
            ))
        })
    };
    let tai_valid_time = |from: i128, to: i128| {
        grafeo_common::types::ValidTimeInterval::from_tai_nanoseconds(from, to).map_err(|error| {
            StorageError::InvalidWalEntry(format!(
                "invalid RDF TAI-nanosecond valid-time interval: {error}"
            ))
        })
    };
    let v3_valid_time = |from: Option<i128>, to: Option<i128>| match (from, to) {
        (None, None) => Ok(None),
        (Some(from), Some(to)) => tai_valid_time(from, to).map(Some),
        _ => Err(StorageError::InvalidWalEntry(
            "RDF quad V3 has only one valid-time bound".to_string(),
        )),
    };

    // Preflight every term, interval, epoch, graph identity, and allocator
    // declaration before applying any record. A corrupt tail must not leave an
    // observable prefix installed when this helper is used outside startup.
    for record in records {
        match record {
            WalRecord::InsertRdfQuadV3 {
                subject,
                predicate,
                object,
                graph,
                graph_incarnation,
                transaction_id,
                ..
            }
            | WalRecord::DeleteRdfQuadV3 {
                subject,
                predicate,
                object,
                graph,
                graph_incarnation,
                transaction_id,
                ..
            } => {
                let triple = parse_triple(subject, predicate, object)?;
                let quad = match graph {
                    Some(graph) => Quad::named(triple, graph.clone()),
                    None => Quad::new(triple),
                };
                RdfGraphIdentity::for_quad(&quad, *graph_incarnation).map_err(|error| {
                    StorageError::InvalidWalEntry(format!(
                        "invalid RDF quad V3 graph identity: {error}"
                    ))
                })?;
                if let WalRecord::InsertRdfQuadV3 {
                    valid_from_tai_ns,
                    valid_to_tai_ns,
                    ..
                } = record
                {
                    v3_valid_time(*valid_from_tai_ns, *valid_to_tai_ns)?;
                }
                commit_epoch(*transaction_id)?;
            }
            WalRecord::CreateNamedRdfGraphV2 {
                name,
                incarnation,
                transaction_id,
            }
            | WalRecord::DropNamedRdfGraphV2 {
                name,
                incarnation,
                transaction_id,
            } => {
                RdfGraphIdentity::named(name.clone(), *incarnation).map_err(|error| {
                    StorageError::InvalidWalEntry(format!(
                        "invalid RDF graph lifecycle V2 identity: {error}"
                    ))
                })?;
                commit_epoch(*transaction_id)?;
            }
            WalRecord::RdfGraphIncarnationHighWaterMeta {
                store_id,
                next_incarnation,
            } if *store_id != rdf_store.store_id() || next_incarnation.is_default_graph() => {
                return Err(StorageError::InvalidWalEntry(format!(
                    "invalid RDF graph-incarnation high-water metadata for store {store_id}"
                ))
                .into());
            }
            WalRecord::RdfGraphIncarnationHighWaterMeta { .. } => {}
            _ => {}
        }
    }

    // High-water metadata is standalone and survives aborted transactions.
    // Apply its monotonic post-image before replay allocates any graph.
    for record in records {
        if let WalRecord::RdfGraphIncarnationHighWaterMeta {
            store_id,
            next_incarnation,
        } = record
        {
            rdf_store
                .adopt_graph_incarnation_high_water(*store_id, *next_incarnation)
                .map_err(StorageError::InvalidWalEntry)?;
        }
    }

    let exact_target = |graph: &Option<String>,
                        incarnation: GraphIncarnationId,
                        epoch: EpochId,
                        create_if_missing: bool|
     -> std::result::Result<Arc<RdfStore>, StorageError> {
        let Some(name) = graph else {
            if incarnation != GraphIncarnationId::DEFAULT_GRAPH {
                return Err(StorageError::InvalidWalEntry(format!(
                    "default RDF graph cannot use incarnation {incarnation}"
                )));
            }
            return Ok(Arc::clone(rdf_store));
        };
        if let Some(target) = rdf_store.graph_with_incarnation(name, incarnation) {
            return Ok(target);
        }
        if rdf_store.graph(name).is_some() {
            return Err(StorageError::InvalidWalEntry(format!(
                "RDF graph <{name}> is not incarnation {incarnation}"
            )));
        }
        if !create_if_missing {
            return Err(StorageError::InvalidWalEntry(format!(
                "RDF graph <{name}> incarnation {incarnation} is absent"
            )));
        }
        rdf_store
            .create_graph_with_incarnation_at(name, incarnation, epoch)
            .map_err(StorageError::InvalidWalEntry)?;
        rdf_store
            .graph_with_incarnation(name, incarnation)
            .ok_or_else(|| {
                StorageError::InvalidWalEntry(format!(
                    "created RDF graph <{name}> incarnation {incarnation} disappeared"
                ))
            })
    };

    for record in records {
        match record {
            WalRecord::InsertRdfQuadV3 {
                subject,
                predicate,
                object,
                graph,
                graph_incarnation,
                valid_from_tai_ns,
                valid_to_tai_ns,
                transaction_id,
            } => {
                let epoch = commit_epoch(*transaction_id)?;
                let triple = parse_triple(subject, predicate, object)?;
                let valid = v3_valid_time(*valid_from_tai_ns, *valid_to_tai_ns)?;
                exact_target(graph, *graph_incarnation, epoch, true)?
                    .try_insert_at_epoch_with_valid(triple, epoch, valid)?;
            }
            WalRecord::DeleteRdfQuadV3 {
                subject,
                predicate,
                object,
                graph,
                graph_incarnation,
                transaction_id,
            } => {
                let epoch = commit_epoch(*transaction_id)?;
                let triple = parse_triple(subject, predicate, object)?;
                exact_target(graph, *graph_incarnation, epoch, false)?
                    .try_remove_at_epoch(&triple, epoch)?;
            }
            WalRecord::CreateNamedRdfGraphV2 {
                name,
                incarnation,
                transaction_id,
            } => {
                rdf_store
                    .create_graph_with_incarnation_at(
                        name,
                        *incarnation,
                        commit_epoch(*transaction_id)?,
                    )
                    .map_err(StorageError::InvalidWalEntry)?;
            }
            WalRecord::DropNamedRdfGraphV2 {
                name,
                incarnation,
                transaction_id,
            } => {
                rdf_store
                    .drop_graph_with_incarnation_at(
                        name,
                        *incarnation,
                        commit_epoch(*transaction_id)?,
                    )
                    .map_err(StorageError::InvalidWalEntry)?;
            }
            WalRecord::Committed { epoch, .. }
            | WalRecord::CommittedWithCdc { epoch, .. }
            | WalRecord::EpochAdvance { epoch } => {
                rdf_store
                    .try_set_commit_epoch(*epoch)
                    .map_err(|error| StorageError::InvalidWalEntry(error.to_string()))?;
            }
            WalRecord::RdfGraphIncarnationHighWaterMeta { .. } => {}
            _ => {}
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "wal"))]
mod wal_replay_tests {
    use super::*;
    use grafeo_common::types::{EpochId, GraphIncarnationId, StoreId, TransactionId};
    use grafeo_storage::wal::WalRecord;

    fn plain_insert(transaction_id: TransactionId) -> WalRecord {
        WalRecord::InsertRdfQuadV3 {
            subject: "<http://ex.org/first>".to_string(),
            predicate: "<http://ex.org/p>".to_string(),
            object: "\"value\"".to_string(),
            graph: None,
            graph_incarnation: GraphIncarnationId::DEFAULT_GRAPH,
            valid_from_tai_ns: None,
            valid_to_tai_ns: None,
            transaction_id,
        }
    }

    #[test]
    fn invalid_tai_interval_rejects_entire_replay_batch() {
        let store = Arc::new(RdfStore::new());
        let transaction_id = TransactionId::new(41);
        let records = vec![
            plain_insert(transaction_id),
            WalRecord::InsertRdfQuadV3 {
                subject: "<http://ex.org/invalid>".to_string(),
                predicate: "<http://ex.org/p>".to_string(),
                object: "\"value\"".to_string(),
                graph: Some("http://ex.org/invalid-graph".to_string()),
                graph_incarnation: GraphIncarnationId::new(7),
                valid_from_tai_ns: Some(7),
                valid_to_tai_ns: Some(7),
                transaction_id,
            },
            WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(3),
            },
        ];

        let error = replay_rdf_wal_records(&store, &records).unwrap_err();
        assert!(error.to_string().contains("from < to"), "{error}");
        assert!(store.is_empty());
        assert!(store.graph("http://ex.org/invalid-graph").is_none());
    }

    #[test]
    fn pending_commit_marker_rejects_entire_replay_batch_without_state_change() {
        let store = Arc::new(RdfStore::new());
        let existing = grafeo_core::graph::rdf::Triple::new(
            grafeo_core::graph::rdf::Term::iri("http://ex.org/existing"),
            grafeo_core::graph::rdf::Term::iri("http://ex.org/p"),
            grafeo_core::graph::rdf::Term::literal("preserved"),
        );
        assert!(
            store
                .try_insert_at_epoch_with_valid(existing, EpochId::new(2), None,)
                .unwrap()
        );
        store.try_set_commit_epoch(EpochId::new(2)).unwrap();

        let before = store.dataset_history().unwrap();
        let before_epoch = store.commit_epoch();
        let transaction_id = TransactionId::new(49);
        let records = vec![
            plain_insert(transaction_id),
            WalRecord::Committed {
                transaction_id,
                epoch: EpochId::PENDING,
            },
        ];

        let error = replay_rdf_wal_records(&store, &records).unwrap_err();
        assert!(error.to_string().contains("reserved PENDING"), "{error}");

        let after = store.dataset_history().unwrap();
        assert_eq!(store.commit_epoch(), before_epoch);
        assert_eq!(after.store_id(), before.store_id());
        assert_eq!(after.completeness(), before.completeness());
        assert_eq!(
            after.next_graph_incarnation(),
            before.next_graph_incarnation()
        );
        assert_eq!(after.graph_lives(), before.graph_lives());
        assert_eq!(after.quad_versions(), before.quad_versions());
    }

    #[test]
    fn malformed_v3_interval_rejects_entire_replay_batch() {
        for (from, to, diagnostic) in [
            (Some(9), Some(8), "from < to"),
            (Some(9), None, "only one valid-time bound"),
            (None, Some(8), "only one valid-time bound"),
        ] {
            let store = Arc::new(RdfStore::new());
            let before_epoch = store.commit_epoch();
            let transaction_id = TransactionId::new(42);
            let records = vec![
                plain_insert(transaction_id),
                WalRecord::InsertRdfQuadV3 {
                    subject: "<http://ex.org/invalid>".to_string(),
                    predicate: "<http://ex.org/p>".to_string(),
                    object: "\"value\"".to_string(),
                    graph: None,
                    graph_incarnation: GraphIncarnationId::DEFAULT_GRAPH,
                    valid_from_tai_ns: from,
                    valid_to_tai_ns: to,
                    transaction_id,
                },
                WalRecord::Committed {
                    transaction_id,
                    epoch: EpochId::new(3),
                },
            ];

            let error = replay_rdf_wal_records(&store, &records).unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
            assert!(store.is_empty());
            assert_eq!(store.commit_epoch(), before_epoch);
        }
    }

    #[test]
    fn exact_named_delete_of_absent_incarnation_does_not_manufacture_a_graph() {
        let store = Arc::new(RdfStore::new());
        let before = store.dataset_history().unwrap();
        let before_epoch = store.commit_epoch();
        let transaction_id = TransactionId::new(43);
        let records = vec![
            WalRecord::DeleteRdfQuadV3 {
                subject: "<http://ex.org/absent>".to_string(),
                predicate: "<http://ex.org/p>".to_string(),
                object: "\"value\"".to_string(),
                graph: Some("http://ex.org/absent-graph".to_string()),
                graph_incarnation: GraphIncarnationId::new(7),
                transaction_id,
            },
            WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(3),
            },
        ];

        let error = replay_rdf_wal_records(&store, &records).unwrap_err();
        assert!(
            error.to_string().contains("incarnation 7 is absent"),
            "{error}"
        );

        assert!(store.graph("http://ex.org/absent-graph").is_none());
        let after = store.dataset_history().unwrap();
        assert_eq!(store.commit_epoch(), before_epoch);
        assert_eq!(after.graph_lives(), before.graph_lives());
        assert_eq!(after.quad_versions(), before.quad_versions());
        assert_eq!(
            after.next_graph_incarnation(),
            before.next_graph_incarnation()
        );
    }

    #[test]
    fn stale_exact_delete_cannot_alias_a_recreated_graph() {
        let store = Arc::new(RdfStore::new());
        let transaction_id = TransactionId::new(44);
        let records = vec![
            WalRecord::CreateNamedRdfGraphV2 {
                name: "urn:g".to_string(),
                incarnation: GraphIncarnationId::new(7),
                transaction_id,
            },
            WalRecord::InsertRdfQuadV3 {
                subject: "<urn:new>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"new\"".to_string(),
                graph: Some("urn:g".to_string()),
                graph_incarnation: GraphIncarnationId::new(7),
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id,
            },
            WalRecord::DeleteRdfQuadV3 {
                subject: "<urn:new>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"new\"".to_string(),
                graph: Some("urn:g".to_string()),
                graph_incarnation: GraphIncarnationId::new(4),
                transaction_id,
            },
            WalRecord::Committed {
                transaction_id,
                epoch: EpochId::new(4),
            },
        ];

        let error = replay_rdf_wal_records(&store, &records).unwrap_err();

        assert!(error.to_string().contains("not incarnation 4"), "{error}");
        let replacement = store
            .graph("urn:g")
            .expect("replacement graph remains active");
        assert_eq!(replacement.graph_incarnation(), GraphIncarnationId::new(7));
        assert_eq!(
            replacement.len(),
            1,
            "stale delete must not touch replacement"
        );
    }

    #[test]
    fn stale_exact_delete_after_drop_cannot_alias_a_recreated_graph() {
        let store = Arc::new(RdfStore::new());
        let old = TransactionId::new(45);
        let drop_old = TransactionId::new(46);
        let replacement = TransactionId::new(47);
        let stale_delete = TransactionId::new(48);
        let records = vec![
            WalRecord::CreateNamedRdfGraphV2 {
                name: "urn:g".to_string(),
                incarnation: GraphIncarnationId::new(1),
                transaction_id: old,
            },
            WalRecord::InsertRdfQuadV3 {
                subject: "<urn:old>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"old\"".to_string(),
                graph: Some("urn:g".to_string()),
                graph_incarnation: GraphIncarnationId::new(1),
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id: old,
            },
            WalRecord::DropNamedRdfGraphV2 {
                name: "urn:g".to_string(),
                incarnation: GraphIncarnationId::new(1),
                transaction_id: drop_old,
            },
            WalRecord::CreateNamedRdfGraphV2 {
                name: "urn:g".to_string(),
                incarnation: GraphIncarnationId::new(2),
                transaction_id: replacement,
            },
            WalRecord::InsertRdfQuadV3 {
                subject: "<urn:new>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"new\"".to_string(),
                graph: Some("urn:g".to_string()),
                graph_incarnation: GraphIncarnationId::new(2),
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id: replacement,
            },
            WalRecord::DeleteRdfQuadV3 {
                subject: "<urn:new>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"new\"".to_string(),
                graph: Some("urn:g".to_string()),
                graph_incarnation: GraphIncarnationId::new(1),
                transaction_id: stale_delete,
            },
            WalRecord::Committed {
                transaction_id: old,
                epoch: EpochId::new(1),
            },
            WalRecord::Committed {
                transaction_id: drop_old,
                epoch: EpochId::new(2),
            },
            WalRecord::Committed {
                transaction_id: replacement,
                epoch: EpochId::new(3),
            },
            WalRecord::Committed {
                transaction_id: stale_delete,
                epoch: EpochId::new(1),
            },
        ];

        let error = replay_rdf_wal_records(&store, &records).unwrap_err();

        assert!(error.to_string().contains("not incarnation 1"), "{error}");
        let active = store.graph("urn:g").expect("replacement remains active");
        assert_eq!(active.graph_incarnation(), GraphIncarnationId::new(2));
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn exact_incarnations_drop_recreate_and_aborted_gaps_survive_replay() {
        let store_id = StoreId::from_bytes([0x37; StoreId::LEN]).unwrap();
        let store = Arc::new(RdfStore::with_config_and_store_id(
            grafeo_core::graph::rdf::RdfStoreConfig::default(),
            store_id,
        ));
        let first = TransactionId::new(51);
        let replacement = TransactionId::new(52);
        let graph = Some("urn:g".to_string());
        let records = vec![
            WalRecord::RdfGraphIncarnationHighWaterMeta {
                store_id,
                next_incarnation: GraphIncarnationId::new(10),
            },
            WalRecord::CreateNamedRdfGraphV2 {
                name: "urn:g".to_string(),
                incarnation: GraphIncarnationId::new(4),
                transaction_id: first,
            },
            WalRecord::InsertRdfQuadV3 {
                subject: "<urn:old>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"old\"".to_string(),
                graph: graph.clone(),
                graph_incarnation: GraphIncarnationId::new(4),
                valid_from_tai_ns: None,
                valid_to_tai_ns: None,
                transaction_id: first,
            },
            WalRecord::Committed {
                transaction_id: first,
                epoch: EpochId::new(2),
            },
            WalRecord::DeleteRdfQuadV3 {
                subject: "<urn:old>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"old\"".to_string(),
                graph: graph.clone(),
                graph_incarnation: GraphIncarnationId::new(4),
                transaction_id: replacement,
            },
            WalRecord::DropNamedRdfGraphV2 {
                name: "urn:g".to_string(),
                incarnation: GraphIncarnationId::new(4),
                transaction_id: replacement,
            },
            WalRecord::CreateNamedRdfGraphV2 {
                name: "urn:g".to_string(),
                incarnation: GraphIncarnationId::new(7),
                transaction_id: replacement,
            },
            WalRecord::InsertRdfQuadV3 {
                subject: "<urn:new>".to_string(),
                predicate: "<urn:p>".to_string(),
                object: "\"new\"".to_string(),
                graph,
                graph_incarnation: GraphIncarnationId::new(7),
                valid_from_tai_ns: Some(1_001),
                valid_to_tai_ns: Some(1_003),
                transaction_id: replacement,
            },
            WalRecord::Committed {
                transaction_id: replacement,
                epoch: EpochId::new(3),
            },
        ];

        replay_rdf_wal_records(&store, &records).unwrap();
        let history = store.dataset_history().unwrap();
        assert_eq!(
            history.next_graph_incarnation(),
            GraphIncarnationId::new(10)
        );
        assert_eq!(history.graph_lives().len(), 2);
        assert_eq!(
            history.graph_lives()[0].graph().incarnation(),
            GraphIncarnationId::new(4)
        );
        assert_eq!(
            history.graph_lives()[1].graph().incarnation(),
            GraphIncarnationId::new(7)
        );
        let old = history.cut(EpochId::new(2)).unwrap();
        let new = history.cut(EpochId::new(3)).unwrap();
        assert_eq!(old.quads.len(), 1);
        assert_eq!(new.quads.len(), 1);
        assert_ne!(old.quads[0].statement, new.quads[0].statement);

        assert!(store.create_graph("urn:after-reopen"));
        assert_eq!(
            store.graph("urn:after-reopen").unwrap().graph_incarnation(),
            GraphIncarnationId::new(10)
        );
    }
}
