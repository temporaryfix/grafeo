//! Catalog participation in the existing Session transaction and publication.

use super::{Arc, Catalog, Result, Session, TransactionId};
use grafeo_common::utils::error::{Error, TransactionError};

pub(super) struct TransactionCatalog {
    pub(super) base: Arc<Catalog>,
    pub(super) current: Arc<Catalog>,
}

#[cfg(all(test, feature = "gql"))]
mod tests {
    use super::*;

    #[test]
    fn savepoint_catalog_payload_retires_after_commit_publication() -> Result<()> {
        for action in [
            "COMMIT",
            "ROLLBACK",
            "ROLLBACK TO SAVEPOINT retained",
            "RELEASE SAVEPOINT retained",
            "conflict",
        ] {
            let db = crate::GrafeoDB::new_in_memory();
            let session = db.session();
            session.execute("START TRANSACTION")?;
            session.execute("CREATE NODE TYPE First (value INTEGER)")?;
            let manager = Arc::clone(&session.transaction_manager);
            let releases = Arc::new(parking_lot::Mutex::new(Vec::new()));
            session.catalog_view().observe_retirement(
                Arc::new(move || manager.publication().try_read().is_some()),
                Arc::clone(&releases),
            );
            session.savepoint("retained")?;
            session.execute("CREATE NODE TYPE Second (value INTEGER)")?;
            releases.lock().clear();
            if action == "conflict" {
                db.session()
                    .execute("CREATE NODE TYPE Concurrent (value INTEGER)")?;
                assert!(session.execute("COMMIT").is_err());
            } else {
                session.execute(action)?;
            }
            assert!(!releases.lock().is_empty());
            assert!(releases.lock().iter().all(|released| *released));
        }
        Ok(())
    }
}

impl TransactionCatalog {
    pub(super) fn begin(catalog: &Catalog) -> Self {
        let base = Arc::new(catalog.snapshot());
        let current = Arc::new(base.snapshot());
        Self { base, current }
    }

    pub(super) fn changed(&self) -> bool {
        !self.base.same_cut(&self.current)
    }
}

impl Clone for TransactionCatalog {
    fn clone(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            current: Arc::new(self.current.snapshot()),
        }
    }
}

#[cfg(feature = "gql")]
impl Session {
    /// Reuse detached DDL validation and the existing graph lifecycle staging.
    /// The enclosing statement savepoint owns every admission failure.
    pub(super) fn execute_transaction_catalog_command(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SchemaStatement,
        workspace: &mut super::CatalogWorkspace,
    ) -> Result<crate::database::QueryResult> {
        use super::{CatalogDdlTarget, PendingGraphNamespace};

        let _publication = self.publication_read_guard();
        let catalog = self.catalog_view();
        let candidate = workspace.transaction_candidate(&catalog);
        let before = catalog
            .ddl_comparison_state()
            .map_err(Error::Serialization)?;
        // A type DROP must see preceding graph unbindings. Keep these in the
        // private metadata cut; the pending entries retain their original live
        // expectations for commit validation.
        for (path, binding) in self.pending_graph_type_bindings.lock().iter() {
            if let Some(graph_type) = &binding.replacement {
                candidate
                    .validate_graph_type_binding_target(graph_type)
                    .map_err(|error| Self::index_ddl_error(error.to_string()))?;
            }
            let expected = candidate.get_graph_type_binding(path);
            if !candidate.publish_graph_type_binding_if_same(
                path,
                expected.as_deref(),
                binding.replacement.clone(),
            ) {
                return Err(Self::index_ddl_error("private graph type binding changed"));
            }
        }
        let mut names: std::collections::HashSet<String> =
            self.store.graph_names().into_iter().collect();
        names.extend(
            self.pending_created_graphs
                .lock()
                .keys()
                .filter_map(|path| match path.components() {
                    [name] => Some(name.clone()),
                    _ => None,
                }),
        );
        let mut graphs_before = grafeo_common::utils::hash::FxHashMap::default();
        for name in names {
            let path = Self::graph_path_for_storage_key(Some(&name))?;
            if let Some(graph) = self.session_graph_path(&path) {
                graphs_before.insert(name, graph);
            }
        }
        let staged_store = self.store.new_graph_topology_candidate()?;
        staged_store.install_named_graphs(graphs_before.clone());
        let mut context = self.current_context.lock().clone();
        if Self::schema_command_uses_current_schema(&cmd) {
            if let Some(requested) = context.schema.as_deref() {
                context.schema = Some(
                    candidate
                        .schema_names()
                        .into_iter()
                        .find(|name| name.eq_ignore_ascii_case(requested))
                        .ok_or_else(|| {
                            Self::index_ddl_error(format!("Schema '{requested}' does not exist"))
                        })?,
                );
            }
            self.require_schema_graph_write_scope(
                context.schema.as_deref(),
                candidate.read().view(),
            )?;
        }
        let schema = parking_lot::Mutex::new(context.schema.clone());
        #[cfg(feature = "wal")]
        let batch = super::CatalogWalBatchGuard::begin(&self.catalog_wal_batch)?;
        let result = self.execute_schema_command_against(
            cmd,
            CatalogDdlTarget {
                catalog: candidate,
                store: &staged_store,
                current_schema: &schema,
                projections: &self.projections,
            },
            false,
        );
        #[cfg(feature = "wal")]
        let _discarded_statement_records = batch.finish();
        let result = result?;
        let graphs_after = staged_store.named_graph_entries();
        if candidate
            .ddl_comparison_state()
            .map_err(Error::Serialization)?
            == before
            && graphs_after
                .keys()
                .all(|name| graphs_before.contains_key(name))
            && graphs_before
                .keys()
                .all(|name| graphs_after.contains_key(name))
        {
            return Ok(result);
        }
        for name in graphs_before
            .keys()
            .filter(|name| !graphs_after.contains_key(*name))
        {
            let path = Self::graph_path_for_storage_key(Some(name))?;
            self.stage_drop_graph_path(&path, &PendingGraphNamespace::Root)?;
        }
        for name in graphs_after
            .keys()
            .filter(|name| !graphs_before.contains_key(*name))
        {
            let path = Self::graph_path_for_storage_key(Some(name))?;
            self.stage_create_graph_path(&path, PendingGraphNamespace::Root)?;
        }
        context.schema = schema.into_inner();
        if !context.native {
            context.storage_key =
                Self::context_graph_path(context.schema.as_deref(), context.graph.as_deref())?;
        }
        let mut transaction = self.transaction_catalog.lock();
        let transaction = transaction.as_mut().ok_or_else(|| {
            Self::index_ddl_error("catalog statement lost its transaction snapshot")
        })?;
        transaction.current = Arc::new(candidate.snapshot());
        *self.current_context.lock() = context;
        self.query_cache.clear();
        self.physical_cache.lock().clear();
        Ok(result)
    }
}

impl Session {
    /// Retain immutable payloads across outer authority scopes. Capturing a cut
    /// only bumps its Arc; private catalog wrappers may subsequently change.
    pub(super) fn pin_catalog_cuts(&self) -> Vec<Catalog> {
        let mut cuts = Vec::new();
        if let Some(catalog) = self.transaction_catalog.lock().as_ref() {
            cuts.push(catalog.base.snapshot());
            cuts.push(catalog.current.snapshot());
        }
        for savepoint in self.savepoints.lock().iter() {
            if let Some(catalog) = &savepoint.catalog_snapshot {
                cuts.push(catalog.current.snapshot());
            }
        }
        cuts
    }

    /// Exact transaction-visible topology, including detached creates and
    /// excluding retired incarnations. This runs only for catalog DDL scans.
    #[cfg(feature = "gql")]
    pub(super) fn catalog_graphs(&self) -> Result<Vec<(super::GraphPath, Arc<super::LpgStore>)>> {
        use super::GraphPath;
        let created: Vec<_> = self.pending_created_graphs.lock().keys().cloned().collect();
        let mut work = vec![GraphPath::root()];
        let mut seen = std::collections::HashSet::new();
        let mut graphs = Vec::new();
        while let Some(path) = work.pop() {
            let Some(graph) = self.session_graph_path(&path) else {
                continue;
            };
            if !seen.insert(Arc::as_ptr(&graph).addr()) {
                return Err(Self::index_ddl_error(
                    "catalog graph scan contains an alias or cycle",
                ));
            }
            let mut children: std::collections::HashSet<_> =
                graph.graph_names().into_iter().collect();
            for candidate in &created {
                if candidate.components().len() == path.components().len() + 1
                    && candidate.components().starts_with(path.components())
                    && let Some(name) = candidate.components().last()
                {
                    children.insert(name.clone());
                }
            }
            for name in children {
                work.push(
                    path.child(&name)
                        .map_err(|error| Self::index_ddl_error(error.to_string()))?,
                );
            }
            graphs.push((path, graph));
        }
        Ok(graphs)
    }

    #[cfg(feature = "gql")]
    pub(super) fn catalog_graph_has_data(
        graph: &super::LpgStore,
        epoch: super::EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Result<bool> {
        use super::GraphStore;
        if !graph
            .prepare_index_node_rows(epoch, transaction_id)?
            .is_empty()
        {
            return Ok(true);
        }
        let pending_edges =
            transaction_id.map_or_else(Vec::new, |tid| graph.pending_edge_creates(tid));
        Ok(graph
            .all_edges()
            .map(|edge| edge.id)
            .chain(pending_edges)
            .any(|id| match transaction_id {
                Some(tid) => graph.is_edge_visible_versioned(id, epoch, tid),
                None => graph.is_edge_visible_at_epoch(id, epoch),
            }))
    }

    pub(super) fn catalog_changed(&self) -> bool {
        self.transaction_catalog
            .lock()
            .as_ref()
            .is_some_and(TransactionCatalog::changed)
    }

    /// The publication gate excludes concurrent commits throughout validation.
    pub(super) fn validate_transaction_catalog(&self, transaction_id: TransactionId) -> Result<()> {
        let (base, current, changed) = {
            let catalog = self.transaction_catalog.lock();
            let catalog = catalog.as_ref().ok_or_else(|| {
                Error::Transaction(TransactionError::InvalidState(
                    "transaction lacks its catalog snapshot".into(),
                ))
            })?;
            (
                Arc::clone(&catalog.base),
                Arc::clone(&catalog.current),
                catalog.changed(),
            )
        };
        let metadata_changed = if changed {
            !self.catalog.same_cut(&base)
        } else {
            !self.catalog.same_schema(&base)
        };
        if metadata_changed {
            let writes = changed
                || !self.pending_index_ddl.lock().is_empty()
                || !self.pending_created_graphs.lock().is_empty()
                || !self.pending_dropped_graphs.lock().is_empty()
                || !self.pending_graph_type_bindings.lock().is_empty()
                || !self
                    .transaction_manager
                    .get_write_set(transaction_id)?
                    .is_empty();
            if writes {
                let removed: Vec<_> = base
                    .schema_names()
                    .into_iter()
                    .filter(|name| !self.catalog.schema_exists(name))
                    .collect();
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!(
                        "catalog schema definitions changed since this transaction's metadata snapshot; removed schemas: {removed:?}"
                    ),
                )));
            }
        }
        #[cfg(feature = "gql")]
        if changed {
            self.validate_catalog_ddl_postimage(&base, &current, transaction_id)?;
        }
        #[cfg(not(feature = "gql"))]
        let _ = current;
        Ok(())
    }

    #[cfg(feature = "gql")]
    fn validate_catalog_ddl_postimage(
        &self,
        base: &Catalog,
        current: &Catalog,
        transaction_id: TransactionId,
    ) -> Result<()> {
        let frontier = self.transaction_manager.current_epoch();
        let old_schemas = base.schema_names();
        for schema in current.schema_names() {
            if !old_schemas.contains(&schema)
                && self
                    .store
                    .graph_names()
                    .iter()
                    .any(|name| super::CatalogDdlTarget::key_belongs_to_schema(name, &schema))
            {
                return Err(Error::Transaction(TransactionError::WriteConflict(
                    format!("schema '{schema}' namespace was claimed during the transaction"),
                )));
            }
        }
        for schema in old_schemas {
            let default_name = format!("{schema}/{}", super::SCHEMA_DEFAULT_GRAPH);
            let path = Self::graph_path_for_storage_key(Some(&default_name))?;
            let retired = self.pending_dropped_graphs.lock().get(&path).cloned();
            if let Some(graph) = retired {
                let surviving_member = self.store.graph_names().iter().any(|name| {
                    if name == &default_name
                        || !super::CatalogDdlTarget::key_belongs_to_schema(name, &schema)
                    {
                        return false;
                    }
                    Self::graph_path_for_storage_key(Some(name)).map_or(true, |path| {
                        !self.pending_dropped_graphs.lock().contains_key(&path)
                    })
                });
                let surviving_child = graph.graph_names().iter().any(|name| {
                    path.child(name).map_or(true, |child| {
                        !self.pending_dropped_graphs.lock().contains_key(&child)
                    })
                });
                if surviving_member
                    || surviving_child
                    || Self::catalog_graph_has_data(&graph, frontier, Some(transaction_id))?
                {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        format!("schema '{schema}' became nonempty before commit"),
                    )));
                }
            }
        }
        let current = current.read();
        for definition in current.all_named_constraints() {
            if base.get_named_constraint(&definition.name).as_ref() != Some(&definition) {
                self.validate_named_constraint_existing_data_at(
                    &definition,
                    current.view(),
                    frontier,
                    Some(transaction_id),
                )?;
            }
        }
        Ok(())
    }
}
