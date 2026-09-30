//! Canonical index requests and committed owner results. This module stages
//! into the existing Session commit driver; it owns no second registry.

use super::{
    Arc, CatalogRead, GraphPath, IndexId, LpgStore, PendingIndexDdl, PendingIndexKind, Result,
    Session,
};
use crate::database::{CreateIndexRequest, IndexCreateKind};

impl Session {
    pub(super) fn index_target_survives(&self, path: &GraphPath, target: &Arc<LpgStore>) -> bool {
        self.lpg_incarnation_survives_pending_lifecycle(path, target)
    }

    #[cfg(feature = "gql")]
    pub(super) fn index_path_matches_flat(path: &GraphPath, key: Option<&str>) -> bool {
        match (path.components(), key) {
            ([], None) => true,
            ([component], Some(name)) => component == name,
            _ => false,
        }
    }

    pub(super) fn validate_index_rebuild_owner(
        catalog: CatalogRead<'_>,
        ddl: &PendingIndexDdl,
    ) -> Result<()> {
        let owner = ddl
            .expected_owner
            .and_then(|id| catalog.get_index(id))
            .ok_or_else(|| Self::index_ddl_error("rebuild owner disappeared during preparation"))?;
        let label = catalog.get_label_name(owner.label);
        let property = catalog.get_property_key_name(owner.property_key);
        if ddl.name.as_deref() != Some(owner.name.as_str())
            || &ddl.graph != owner.key.graph()
            || label.as_deref() != Some(ddl.label.as_str())
            || property.as_deref() != Some(ddl.property.as_str())
            || ddl.configuration.as_ref() != Some(&owner.configuration)
        {
            return Err(Self::index_ddl_error(
                "rebuild owner changed during preparation",
            ));
        }
        Ok(())
    }

    pub(super) fn graph_path_for_storage_key(key: Option<&str>) -> Result<GraphPath> {
        match key {
            None => Ok(GraphPath::root()),
            Some(name) => GraphPath::from_components(&[name])
                .map_err(|error| Self::index_ddl_error(error.to_string())),
        }
    }

    pub(super) fn require_index_path_write_grant(&self, path: &GraphPath) -> Result<()> {
        self.require_graph_path_grant(path, crate::auth::Role::ReadWrite)
    }

    pub(super) fn resolve_index_graph_path(&self, path: &GraphPath) -> Result<Arc<LpgStore>> {
        self.session_graph_path(path)
            .ok_or_else(|| Self::index_ddl_error(format!("index graph {path:?} does not exist")))
    }

    pub(crate) fn create_index_durable(&self, request: CreateIndexRequest) -> Result<IndexId> {
        // A database convenience call always creates its own auto-commit
        // Session. Do not expose a provisional ID from a transaction Session.
        if self.in_transaction() || !self.auto_commit {
            return Err(Self::index_ddl_error(
                "create_index requires an auto-commit Session",
            ));
        }
        let result = self.with_lpg_auto_commit(|| {
            let property_wide = matches!(request.kind, IndexCreateKind::Property | IndexCreateKind::BTree);
            if request.property.is_empty()
                || (property_wide && request.label.is_some())
                || (!property_wide && request.label.as_ref().is_none_or(String::is_empty))
            {
                return Err(Self::index_ddl_error(
                    "index property must be nonempty; Property/BTree forbid a label and Text/Vector require one",
                ));
            }
            self.require_index_path_write_grant(&request.graph)?;
            let kind = match request.kind {
                IndexCreateKind::Property => PendingIndexKind::Property,
                IndexCreateKind::BTree => PendingIndexKind::BTree,
                IndexCreateKind::Text { min_token_length } => PendingIndexKind::Text { min_token_length },
                IndexCreateKind::Vector { dimensions, metric, m, ef_construction, ef, quantization } => {
                    PendingIndexKind::Vector { dimensions, metric, m, ef_construction, ef, quantization }
                }
            };
            let _publication = self.publication_read_guard();
            let target = self.resolve_index_graph_path(&request.graph)?;
            let result = Arc::new(std::sync::OnceLock::new());
            self.stage_create_index_on(PendingIndexDdl {
                create: true, rebuild: false, expected_owner: None, configuration: None,
                owner_result: Arc::clone(&result), owner_schema: None, graph: request.graph,
                name: request.name, label: request.label.unwrap_or_default(),
                property: request.property, kind, target,
            }, false)?;
            Ok(result)
        })?;
        result.get().copied().ok_or_else(|| {
            Self::index_ddl_error("committed index creation did not publish its owner result")
        })
    }

    pub(crate) fn drop_index_durable(&self, owner: IndexId) -> Result<bool> {
        self.with_lpg_auto_commit(|| self.stage_index_owner_operation(owner, false))
    }

    pub(crate) fn rebuild_index_durable(&self, owner: IndexId) -> Result<()> {
        self.with_lpg_auto_commit(|| {
            if !self.stage_index_owner_operation(owner, true)? {
                return Err(Self::index_ddl_error(format!(
                    "index owner {} does not exist",
                    owner.as_u32()
                )));
            }
            Ok(())
        })
    }

    pub(super) fn stage_index_owner_operation(
        &self,
        owner: IndexId,
        rebuild: bool,
    ) -> Result<bool> {
        let _publication = self.publication_read_guard();
        let catalog_owner = self.catalog_view();
        let catalog = catalog_owner.read();
        let Some(definition) = catalog.get_index(owner) else {
            return Ok(false);
        };
        self.require_index_path_write_grant(definition.key.graph())?;
        let label = catalog
            .get_label_name(definition.label)
            .ok_or_else(|| Self::index_ddl_error("index owner label is missing"))?
            .to_string();
        let property = catalog
            .get_property_key_name(definition.property_key)
            .ok_or_else(|| Self::index_ddl_error("index owner property is missing"))?
            .to_string();
        let target = self.resolve_index_graph_path(definition.key.graph())?;
        let kind = Self::pending_kind_from_catalog(definition.index_type);
        if !Self::physical_index_exists(&target, &definition.key) {
            return Err(Self::index_ddl_error("index owner has no physical index"));
        }
        self.pending_index_ddl.lock().push(PendingIndexDdl {
            create: rebuild,
            rebuild,
            graph: definition.key.graph().clone(),
            owner_schema: None,
            name: Some(definition.name),
            expected_owner: Some(owner),
            configuration: Some(definition.configuration),
            owner_result: Arc::new(std::sync::OnceLock::new()),
            label,
            property,
            kind,
            target,
        });
        Ok(true)
    }
}
