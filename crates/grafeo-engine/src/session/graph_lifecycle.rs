//! Literal native graph lifecycle staged through the ordinary Session commit.

use super::{
    Arc, GraphPath, PendingCreatedGraph, PendingGraphNamespace, Result, SCHEMA_DEFAULT_GRAPH,
    Session,
};

impl Session {
    /// Captures the namespace of a literal root-child name without resolving
    /// aliases or changing any component. The caller holds publication read.
    fn native_graph_namespace(&self, path: &GraphPath) -> Result<PendingGraphNamespace> {
        let Some((prefix, _)) = path
            .components()
            .first()
            .and_then(|name| name.split_once('/'))
        else {
            return Ok(PendingGraphNamespace::Root);
        };
        let canonical = self
            .catalog_view()
            .schema_names()
            .into_iter()
            .find(|name| name.eq_ignore_ascii_case(prefix));
        let Some(canonical) = canonical else {
            return Ok(PendingGraphNamespace::UnregisteredPrefix(prefix.to_owned()));
        };
        if canonical != prefix {
            return Err(Self::index_ddl_error(format!(
                "Native graph path {path:?} must use the exact schema prefix '{canonical}'"
            )));
        }
        let owner = self
            .schema_incarnation_for_name(Some(&canonical))?
            .ok_or_else(|| Self::index_ddl_error("native graph schema incarnation is missing"))?;
        Ok(PendingGraphNamespace::Schema(owner))
    }

    /// Creates one literal LPG path in the current or an automatic transaction.
    ///
    /// The parent must already exist in the transaction's graph view. Empty
    /// components and embedded slashes are literal names; no ancestor is
    /// created implicitly. Returns `false` for an existing path, including root.
    ///
    /// # Errors
    /// Rejects missing parents, insufficient permissions, namespace conflicts,
    /// historical/read-only views and transaction or durability failures.
    pub fn create_graph_path(&self, path: &GraphPath) -> Result<bool> {
        self.with_lpg_graph_lifecycle(|| {
            let _publication = self.publication_read_guard();
            let namespace = self.native_graph_namespace(path)?;
            self.stage_create_graph_path(path, namespace)
        })
    }

    /// Shared resolved-path CREATE seam. The caller retains publication read
    /// from its one namespace resolution through this staging operation.
    pub(super) fn stage_create_graph_path(
        &self,
        path: &GraphPath,
        namespace: PendingGraphNamespace,
    ) -> Result<bool> {
        self.require_graph_path_grant(path, crate::auth::Role::ReadWrite)?;
        let Some(parent_path) = path
            .parent()
            .map_err(|error| Self::index_ddl_error(error.to_string()))?
        else {
            return Ok(false);
        };
        let transaction_id = self.current_transaction_id().ok_or_else(|| {
            Self::index_ddl_error("graph creation requires an active transaction")
        })?;
        let parent = self.session_graph_path(&parent_path);
        self.track_lpg_graph_coordinate(transaction_id, parent_path.clone(), parent.clone())?;
        let parent = parent.ok_or_else(|| {
            Self::index_ddl_error(format!("Graph parent {parent_path:?} does not exist"))
        })?;
        let existing = self.session_graph_path(path);
        self.track_lpg_graph_coordinate(transaction_id, path.clone(), existing.clone())?;
        if existing.is_some() {
            return Ok(false);
        }

        let graph = self.new_detached_lpg_graph()?;
        self.pending_created_graphs.lock().insert(
            path.clone(),
            PendingCreatedGraph {
                store: Arc::clone(&graph),
                parent,
                namespace,
                copy_source_indexes: None,
            },
        );
        self.track_lpg_graph_coordinate(transaction_id, path.clone(), Some(Arc::clone(&graph)))?;
        #[cfg(feature = "wal")]
        self.log_schema_wal(&grafeo_storage::wal::WalRecord::CreateLpgGraph {
            graph: path.clone(),
            incarnation: graph.graph_incarnation_id(),
            transaction_id,
        })?;
        Ok(true)
    }

    /// Drops a literal LPG path and its transaction-visible descendants.
    ///
    /// Returns `false` when the path is absent. Cascades index owners, graph
    /// type bindings and virtual projections while preserving transaction and
    /// savepoint ownership of every retired graph incarnation.
    ///
    /// # Errors
    /// Rejects root and schema-default partitions, insufficient permissions,
    /// invalid ownership, and transaction or durability failures.
    pub fn drop_graph_path(&self, path: &GraphPath) -> Result<bool> {
        self.with_lpg_graph_lifecycle(|| {
            let _publication = self.publication_read_guard();
            let namespace = self.native_graph_namespace(path)?;
            self.stage_drop_graph_path(path, &namespace)
        })
    }

    /// Shared resolved-path DROP seam for native and language/schema callers.
    pub(super) fn stage_drop_graph_path(
        &self,
        path: &GraphPath,
        namespace: &PendingGraphNamespace,
    ) -> Result<bool> {
        self.require_graph_path_grant(path, crate::auth::Role::ReadWrite)?;
        let Some(parent_path) = path
            .parent()
            .map_err(|error| Self::index_ddl_error(error.to_string()))?
        else {
            return Err(Self::index_ddl_error("The root graph cannot be dropped"));
        };
        let owner_schema = match namespace {
            PendingGraphNamespace::Schema(owner) => {
                if matches!(path.components(), [name] if name == &format!("{}/{SCHEMA_DEFAULT_GRAPH}", owner.name))
                {
                    return Err(Self::index_ddl_error(format!(
                        "Graph {path:?} is the owning schema's default partition and cannot be dropped directly"
                    )));
                }
                Some(owner)
            }
            PendingGraphNamespace::Root | PendingGraphNamespace::UnregisteredPrefix(_) => None,
        };
        let transaction_id = self.current_transaction_id().ok_or_else(|| {
            Self::index_ddl_error("graph lifecycle requires an active transaction")
        })?;
        let _publication = self.publication_read_guard();
        let parent = self.session_graph_path(&parent_path);
        self.track_lpg_graph_coordinate(transaction_id, parent_path, parent)?;
        let target = self.session_graph_path(path);
        self.track_lpg_graph_coordinate(transaction_id, path.clone(), target.clone())?;
        let Some(target) = target else {
            return Ok(false);
        };

        // Detached creates are not installed in their parent's child map.
        // Include their subtrees, then resolve every coordinate through the
        // transaction view to exclude replaced/cancelled old incarnations.
        let mut candidates = Self::index_subtree(path.clone(), target)?;
        let staged: Vec<_> = self
            .pending_created_graphs
            .lock()
            .iter()
            .filter(|(candidate, _)| candidate.components().starts_with(path.components()))
            .map(|(candidate, pending)| (candidate.clone(), Arc::clone(&pending.store)))
            .collect();
        for (candidate, store) in staged {
            candidates.extend(Self::index_subtree(candidate, store)?);
        }
        let mut retiring: Vec<_> = candidates
            .into_keys()
            .filter_map(|candidate| {
                self.session_graph_path(&candidate)
                    .map(|store| (candidate, store))
            })
            .collect();
        retiring.sort_by(|(left, _), (right, _)| {
            right
                .components()
                .len()
                .cmp(&left.components().len())
                .then_with(|| left.cmp(right))
        });
        for (candidate, store) in &retiring {
            self.require_graph_path_grant(candidate, crate::auth::Role::ReadWrite)?;
            self.track_lpg_graph_coordinate(
                transaction_id,
                candidate.clone(),
                Some(Arc::clone(store)),
            )?;
        }
        self.stage_graph_type_binding_cascade(path)?;
        for (candidate, store) in &retiring {
            self.stage_graph_index_cascade(candidate, store, owner_schema)?;
            self.stage_virtual_projection_cascade(candidate, store);
        }
        for (candidate, store) in retiring {
            #[cfg(feature = "wal")]
            let incarnation = store.graph_incarnation_id();
            let (created, privately_owned) = {
                let mut pending = self.pending_created_graphs.lock();
                let privately_owned = pending
                    .keys()
                    .any(|created| candidate.components().starts_with(created.components()));
                let created = if pending
                    .get(&candidate)
                    .is_some_and(|created| Arc::ptr_eq(&created.store, &store))
                {
                    pending.remove(&candidate)
                } else {
                    None
                };
                (created, privately_owned)
            };
            if privately_owned {
                // A detached COPY may already contain physical descendants
                // without a separate pending-create entry. They belong to the
                // cancelled incarnation too, never to the published DROP CAS.
                let retired = created.map_or_else(|| Arc::clone(&store), |created| created.store);
                self.cancelled_created_graphs
                    .lock()
                    .entry(candidate.clone())
                    .or_default()
                    .push(retired);
                self.pending_index_ddl
                    .lock()
                    .retain(|ddl| ddl.graph != candidate || !Arc::ptr_eq(&ddl.target, &store));
            } else {
                self.pending_dropped_graphs
                    .lock()
                    .insert(candidate.clone(), store);
            }
            #[cfg(feature = "wal")]
            {
                self.log_schema_wal(&grafeo_storage::wal::WalRecord::DropLpgGraph {
                    graph: candidate.clone(),
                    incarnation,
                    transaction_id,
                })?;
                self.log_schema_wal(&grafeo_storage::wal::WalRecord::SetGraphTypeBinding {
                    transaction_id,
                    graph: candidate,
                    graph_type: None,
                })?;
            }
        }
        Ok(true)
    }
}
