//! Model-independent catalog persistence for RDF-only containers.
//!
//! LPG containers use `CatalogSection`, whose payload also owns physical LPG
//! index configuration. An RDF-only build has no `LpgStore`, but still needs a
//! canonical authoritative catalog component for schema provenance and world
//! manifests.

use std::sync::Arc;

use grafeo_common::storage::{Section, SectionType};
use grafeo_common::utils::error::{Error, Result};

use crate::catalog::{CURRENT_CATALOG_STATE_VERSION, Catalog, CatalogWorkspace};

pub(super) struct CatalogStateSection {
    catalog: Arc<Catalog>,
}

impl CatalogStateSection {
    pub(super) const FORMAT_VERSION: u8 = CURRENT_CATALOG_STATE_VERSION;

    pub(super) const fn new(catalog: Arc<Catalog>) -> Self {
        Self { catalog }
    }
}

impl Section for CatalogStateSection {
    fn section_type(&self) -> SectionType {
        SectionType::Catalog
    }

    fn version(&self) -> u8 {
        Self::FORMAT_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        validate_model_independent_state(&self.catalog)?;
        self.catalog.encode_current_state_v2().map_err(|error| {
            Error::Serialization(format!("RDF-only catalog serialization failed: {error}"))
        })
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let candidate = Catalog::decode_current_state_v2(data).map_err(|error| {
            Error::Serialization(format!("RDF-only catalog deserialization failed: {error}"))
        })?;
        validate_model_independent_state(&candidate)?;
        let mut workspace = CatalogWorkspace::replacement(candidate);
        self.catalog
            .prepare_replacement(&mut workspace)
            .map_err(|error| Error::Serialization(error.to_string()))?
            .install()
            .finish();
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        false
    }

    fn mark_clean(&self) {}

    fn memory_usage(&self) -> usize {
        0
    }
}

fn validate_model_independent_state(catalog: &Catalog) -> Result<()> {
    if catalog.index_allocator_high_water() != 0 || !catalog.all_graph_type_bindings().is_empty() {
        return Err(Error::Serialization(
            "RDF-only catalog cannot own LPG indexes or graph-type bindings without LPG topology"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state2_without_lpg_topology_rejects_owners_and_graph_bindings_before_publication()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let target = Arc::new(Catalog::new());
        let mut section = CatalogStateSection::new(Arc::clone(&target));
        let before = section.serialize()?;
        let owner_source = Catalog::new();
        let label = owner_source.get_or_create_label("Item")?;
        let property = owner_source.get_or_create_property_key("value")?;
        owner_source.create_index(
            None,
            label,
            property,
            grafeo_common::types::GraphPath::root(),
            crate::catalog::IndexConfiguration::Property,
        )?;
        assert!(
            section
                .deserialize(&owner_source.encode_current_state_v2()?)
                .is_err()
        );
        assert_eq!(section.serialize()?, before);
        let binding_source = Catalog::new();
        binding_source.register_graph_type(crate::catalog::GraphTypeDefinition {
            name: "Open".into(),
            allowed_node_types: Vec::new(),
            allowed_edge_types: Vec::new(),
            open: true,
        })?;
        binding_source.bind_graph_type(&grafeo_common::types::GraphPath::root(), "Open".into())?;
        assert!(
            section
                .deserialize(&binding_source.encode_current_state_v2()?)
                .is_err()
        );
        assert_eq!(section.serialize()?, before);
        assert!(
            CatalogStateSection::new(Arc::new(binding_source))
                .serialize()
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn exact_catalog_state_round_trips_without_an_lpg_store()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = Arc::new(Catalog::new());
        let label = source.get_or_create_label("Observation")?;
        let property = source.get_or_create_property_key("observed_at")?;
        let section = CatalogStateSection::new(source);
        let bytes = section.serialize().unwrap();

        let target = Arc::new(Catalog::new());
        let mut restored = CatalogStateSection::new(Arc::clone(&target));
        restored.deserialize(&bytes).unwrap();

        assert_eq!(target.get_label_name(label).as_deref(), Some("Observation"));
        assert_eq!(
            target.get_property_key_name(property).as_deref(),
            Some("observed_at")
        );
        Ok(())
    }
}
