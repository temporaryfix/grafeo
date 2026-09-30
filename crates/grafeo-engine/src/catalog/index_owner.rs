//! Resolved creation contracts for canonical logical index owners.

use super::{CatalogError, IndexType};
use grafeo_common::types::GraphPath;
use grafeo_core::graph::lpg::PhysicalIndexKey;

/// Namespace reserved for names allocated by the catalog itself.
pub const ANONYMOUS_INDEX_PREFIX: &str = "@grafeo-index:";

/// Complete resolved index configuration, retained by its logical owner.
#[derive(Debug, Clone)]
pub enum IndexConfiguration {
    /// Property-wide equality index.
    Property,
    /// Property-wide ordered index, sharing the physical Property family.
    BTree,
    /// Text index using the explicitly described Simple tokenizer.
    #[cfg(feature = "text-index")]
    Text {
        /// Exact BM25 scoring parameters.
        config: grafeo_core::index::text::BM25Config,
        /// Simple tokenizer minimum token length; zero is valid.
        min_token_length: usize,
    },
    /// Vector index with every HNSW option and quantization choice resolved.
    #[cfg(feature = "vector-index")]
    Vector {
        /// Exact construction and search parameters.
        config: grafeo_core::index::vector::HnswConfig,
        /// Exact quantization kind, including Product subdivision count.
        quantization: grafeo_core::index::vector::QuantizationType,
    },
}

impl IndexConfiguration {
    /// Read-only kind projection; configuration remains authoritative.
    #[must_use]
    pub const fn index_type(&self) -> IndexType {
        match self {
            Self::Property => IndexType::Hash,
            Self::BTree => IndexType::BTree,
            #[cfg(feature = "text-index")]
            Self::Text { .. } => IndexType::FullText,
            #[cfg(feature = "vector-index")]
            Self::Vector { .. } => IndexType::Vector,
        }
    }

    /// Qualifies resolved options before any owner identity is allocated.
    ///
    /// # Errors
    /// Rejects unsupported or invalid physical configuration without defaults.
    pub fn validate(&self) -> Result<(), CatalogError> {
        match self {
            Self::Property | Self::BTree => Ok(()),
            #[cfg(feature = "text-index")]
            Self::Text { config, .. } => {
                if !config.k1.is_finite()
                    || config.k1 < 0.0
                    || !config.b.is_finite()
                    || !(0.0..=1.0).contains(&config.b)
                {
                    return Err(CatalogError::InvalidIndex(
                        "invalid resolved BM25 parameters".into(),
                    ));
                }
                Ok(())
            }
            #[cfg(feature = "vector-index")]
            Self::Vector {
                config,
                quantization,
            } => {
                use grafeo_core::index::vector::QuantizationType;
                if config.dimensions == 0
                    || config.m == 0
                    || config.m_max < config.m
                    || config.ef_construction == 0
                    || config.ef == 0
                    || !config.ml.is_finite()
                    || config.ml <= 0.0
                    || !config.alpha.is_finite()
                    || config.alpha <= 0.0
                    || config.max_elements == Some(0)
                {
                    return Err(CatalogError::InvalidIndex(
                        "invalid resolved HNSW parameters".into(),
                    ));
                }
                match quantization {
                    QuantizationType::None
                    | QuantizationType::Scalar
                    | QuantizationType::Binary => Ok(()),
                    QuantizationType::Product { num_subvectors } => {
                        if *num_subvectors == 0
                            || !config.dimensions.is_multiple_of(*num_subvectors)
                        {
                            return Err(CatalogError::InvalidIndex(
                                "invalid resolved Product quantization dimensions".into(),
                            ));
                        }
                        Ok(())
                    }
                    _ => Err(CatalogError::InvalidIndex(
                        "unsupported resolved quantization kind".into(),
                    )),
                }
            }
        }
    }

    pub(super) fn physical_key(
        &self,
        graph: GraphPath,
        label: &str,
        property: &str,
    ) -> PhysicalIndexKey {
        match self.index_type() {
            IndexType::Hash | IndexType::BTree => PhysicalIndexKey::property(graph, property),
            IndexType::FullText => PhysicalIndexKey::text(graph, label, property),
            IndexType::Vector => PhysicalIndexKey::vector(graph, label, property),
        }
    }
}

// Floating-point options are creation-contract bits, not approximate scores.
impl PartialEq for IndexConfiguration {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Property, Self::Property) | (Self::BTree, Self::BTree) => true,
            #[cfg(feature = "text-index")]
            (
                Self::Text {
                    config: a,
                    min_token_length: al,
                },
                Self::Text {
                    config: b,
                    min_token_length: bl,
                },
            ) => a.k1.to_bits() == b.k1.to_bits() && a.b.to_bits() == b.b.to_bits() && al == bl,
            #[cfg(feature = "vector-index")]
            (
                Self::Vector {
                    config: a,
                    quantization: aq,
                },
                Self::Vector {
                    config: b,
                    quantization: bq,
                },
            ) => {
                a.dimensions == b.dimensions
                    && a.metric == b.metric
                    && a.m == b.m
                    && a.m_max == b.m_max
                    && a.ef_construction == b.ef_construction
                    && a.ef == b.ef
                    && a.ml.to_bits() == b.ml.to_bits()
                    && a.alpha.to_bits() == b.alpha.to_bits()
                    && a.max_elements == b.max_elements
                    && aq == bq
            }
            _ => false,
        }
    }
}

impl Eq for IndexConfiguration {}

#[cfg(test)]
mod tests {
    use super::{ANONYMOUS_INDEX_PREFIX, IndexConfiguration, PhysicalIndexKey};
    use crate::catalog::{Catalog, CatalogError};
    use grafeo_common::types::{GraphPath, IndexId, LabelId, PropertyKeyId};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn fixture() -> Result<(Catalog, LabelId, PropertyKeyId), CatalogError> {
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("Person")?;
        let property = catalog.get_or_create_property_key("value")?;
        Ok((catalog, label, property))
    }

    #[test]
    fn anonymous_owner_names_and_high_water_survive_all_drops() -> TestResult {
        let (catalog, label, property) = fixture()?;
        for expected in 0..3 {
            let owner = catalog.create_index(
                None,
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Property,
            )?;
            assert_eq!(owner.as_u32(), expected);
            assert_eq!(
                catalog.get_index(owner).ok_or("owner missing")?.name,
                format!("{ANONYMOUS_INDEX_PREFIX}{expected}")
            );
            assert!(catalog.drop_index(owner));
            assert_eq!(catalog.index_count(), 0);
            assert_eq!(catalog.index_allocator_high_water(), expected + 1);
            assert_eq!(catalog.index_graph(owner), None);
        }
        for name in ["@grafeo-index:", "@grafeo-index:0", "@grafeo-index:future"] {
            assert!(matches!(
                catalog.create_index(
                    Some(name),
                    label,
                    property,
                    GraphPath::root(),
                    IndexConfiguration::Property
                ),
                Err(CatalogError::InvalidIndex(_))
            ));
        }
        assert_eq!(catalog.index_allocator_high_water(), 3);
        assert_eq!(catalog.index_count(), 0);
        let owner = catalog.create_index(
            Some("named"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        assert_eq!(owner.as_u32(), 3);
        assert_eq!(catalog.find_index_by_name("named"), Some(owner));
        Ok(())
    }

    #[test]
    fn exact_paths_do_not_alias_root_empty_slash_and_nested_graphs() -> TestResult {
        let (catalog, label, property) = fixture()?;
        for graph in [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["schema/graph"])?,
            GraphPath::from_components(&["schema", "graph"])?,
        ] {
            let owner = catalog.create_index(
                None,
                label,
                property,
                graph.clone(),
                IndexConfiguration::Property,
            )?;
            assert_eq!(catalog.index_graph(owner), Some(graph.clone()));
            assert_eq!(
                catalog.get_index(owner).ok_or("owner missing")?.key,
                PhysicalIndexKey::property(graph, "value")
            );
        }
        assert_eq!(catalog.index_count(), 4);
        assert_eq!(catalog.index_graph(IndexId::new(99)), None);
        assert_eq!(
            catalog.index_graph(IndexId::new(0)),
            Some(GraphPath::root())
        );
        Ok(())
    }

    #[test]
    fn physical_property_family_has_one_owner_across_declaration_labels() -> TestResult {
        let (catalog, label, property) = fixture()?;
        let other_label = catalog.get_or_create_label("Other")?;
        let property_owner = catalog.create_index(
            Some("property"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        assert_eq!(
            catalog.create_index(
                Some("alias"),
                other_label,
                property,
                GraphPath::root(),
                IndexConfiguration::BTree
            ),
            Err(CatalogError::IndexPhysicalAlreadyOwned(property_owner))
        );
        assert_eq!(catalog.index_allocator_high_water(), 1);
        assert!(catalog.drop_index(property_owner));
        let ordered = catalog.create_index(
            Some("ordered"),
            other_label,
            property,
            GraphPath::root(),
            IndexConfiguration::BTree,
        )?;
        assert_eq!(
            catalog.create_index(
                None,
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Property
            ),
            Err(CatalogError::IndexPhysicalAlreadyOwned(ordered))
        );
        let named = GraphPath::from_components(&["named"])?;
        assert_eq!(
            catalog.create_index(
                Some("ordered"),
                label,
                property,
                named.clone(),
                IndexConfiguration::Property
            ),
            Err(CatalogError::IndexAlreadyExists("ordered".into()))
        );
        catalog.create_index(None, label, property, named, IndexConfiguration::Property)?;
        assert_eq!(catalog.index_count(), 2);
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn natural_owner_lookup_tracks_exact_keys_and_detached_prepared_recreation() -> TestResult {
        use crate::catalog::CatalogWorkspace;

        let (catalog, label, property) = fixture()?;
        let key = PhysicalIndexKey::property(GraphPath::root(), "value");
        let original = catalog.create_index(
            None,
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        {
            let cut = catalog.read();
            let definition = cut
                .view()
                .physical_index_owner(&key)
                .ok_or("owner missing")?;
            assert_eq!(definition.id, original);
            assert_eq!(definition.key, key);
            for different in [
                PhysicalIndexKey::property(GraphPath::from_components(&[""])?, "value"),
                PhysicalIndexKey::property(GraphPath::root(), "value\0suffix"),
                PhysicalIndexKey::text(GraphPath::root(), "Person", "value"),
            ] {
                assert!(cut.view().physical_index_owner(&different).is_none());
            }
        }
        let mut workspace = CatalogWorkspace::new();
        let edit = catalog.prepare_edit(&mut workspace)?;
        let candidate = edit.candidate();
        assert!(candidate.drop_index(original));
        assert!(candidate.read().view().physical_index_owner(&key).is_none());
        let replacement = candidate.create_index(
            None,
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::BTree,
        )?;
        assert_eq!(replacement.as_u32(), original.as_u32() + 1);
        let ready = edit.finish();
        let before = ready
            .preimage()
            .physical_index_owner(&key)
            .ok_or("live owner missing")?;
        let after = ready
            .view()
            .physical_index_owner(&key)
            .ok_or("candidate owner missing")?;
        assert_eq!(before.id, original);
        assert_eq!(after.id, replacement);
        assert_eq!(after.configuration, IndexConfiguration::BTree);
        assert_eq!(after.key, key);
        ready.install().finish();
        assert_eq!(
            catalog
                .read()
                .view()
                .physical_index_owner(&key)
                .map(|definition| definition.id),
            Some(replacement),
        );
        assert!(catalog.drop_index(replacement));
        assert!(catalog.read().view().physical_index_owner(&key).is_none());
        assert_eq!(
            catalog.index_allocator_high_water(),
            replacement.as_u32() + 1
        );
        Ok(())
    }

    #[test]
    fn canonical_owner_exhaustion_and_invalid_referents_leave_state_unchanged() -> TestResult {
        let (catalog, label, property) = fixture()?;
        assert!(matches!(
            catalog.create_index(
                None,
                LabelId::new(99),
                property,
                GraphPath::root(),
                IndexConfiguration::Property
            ),
            Err(CatalogError::LabelNotFound(_))
        ));
        assert!(matches!(
            catalog.create_index(
                None,
                label,
                PropertyKeyId::new(99),
                GraphPath::root(),
                IndexConfiguration::Property
            ),
            Err(CatalogError::PropertyKeyNotFound(_))
        ));
        assert_eq!(catalog.index_allocator_high_water(), 0);
        catalog.state.write().indexes.next_id = u32::MAX - 1;
        let last = catalog.create_index(
            None,
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        assert_eq!(last.as_u32(), u32::MAX - 1);
        assert_eq!(catalog.index_allocator_high_water(), u32::MAX);
        assert!(catalog.drop_index(last));
        assert_eq!(
            catalog.create_index(
                None,
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Property
            ),
            Err(CatalogError::IdExhausted("index"))
        );
        assert_eq!(catalog.index_allocator_high_water(), u32::MAX);
        assert_eq!(catalog.index_count(), 0);
        Ok(())
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn text_owner_retains_exact_config_and_zero_minimum_length() -> TestResult {
        use grafeo_core::index::text::BM25Config;
        let (catalog, label, property) = fixture()?;
        let config = IndexConfiguration::Text {
            config: BM25Config { k1: 0.0, b: 1.0 },
            min_token_length: 0,
        };
        let owner =
            catalog.create_index(None, label, property, GraphPath::root(), config.clone())?;
        assert_eq!(
            catalog.get_index(owner).ok_or("owner missing")?.key,
            PhysicalIndexKey::text(GraphPath::root(), "Person", "value"),
        );
        assert_eq!(
            catalog
                .get_index(owner)
                .ok_or("owner missing")?
                .configuration,
            config
        );
        let changed = IndexConfiguration::Text {
            config: BM25Config { k1: 0.0, b: 1.0 },
            min_token_length: 7,
        };
        assert_ne!(config, changed);
        assert_eq!(
            catalog.create_index(None, label, property, GraphPath::root(), changed),
            Err(CatalogError::IndexPhysicalAlreadyOwned(owner))
        );
        let other = catalog.get_or_create_label("Other")?;
        catalog.create_index(None, other, property, GraphPath::root(), config)?;
        catalog.create_index(
            None,
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        for scoring in [
            BM25Config {
                k1: f64::NAN,
                b: 0.5,
            },
            BM25Config {
                k1: 1.0,
                b: f64::INFINITY,
            },
            BM25Config { k1: -1.0, b: 0.5 },
        ] {
            assert!(matches!(
                catalog.create_index(
                    None,
                    label,
                    property,
                    GraphPath::root(),
                    IndexConfiguration::Text {
                        config: scoring,
                        min_token_length: 2,
                    }
                ),
                Err(CatalogError::InvalidIndex(_))
            ));
        }
        assert_eq!(catalog.index_allocator_high_water(), 3);
        Ok(())
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn vector_owner_retains_every_resolved_option_and_quantization() -> TestResult {
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, QuantizationType};
        let (catalog, label, property) = fixture()?;
        let hnsw = HnswConfig {
            dimensions: 12,
            metric: DistanceMetric::Manhattan,
            m: 3,
            m_max: 9,
            ef_construction: 17,
            ef: 11,
            ml: 0.6,
            alpha: 1.25,
            max_elements: Some(29),
        };
        let config = IndexConfiguration::Vector {
            config: hnsw.clone(),
            quantization: QuantizationType::Product { num_subvectors: 3 },
        };
        let owner =
            catalog.create_index(None, label, property, GraphPath::root(), config.clone())?;
        let expected = catalog.get_index(owner).ok_or("owner missing")?;
        assert_eq!(
            expected.key,
            PhysicalIndexKey::vector(GraphPath::root(), "Person", "value"),
        );
        assert_eq!(expected.configuration, config);
        let mut changed = hnsw.clone();
        changed.ef += 1;
        let changed = IndexConfiguration::Vector {
            config: changed,
            quantization: QuantizationType::Product { num_subvectors: 3 },
        };
        assert_ne!(expected.configuration, changed);
        assert_eq!(
            catalog.create_index(
                Some("replacement"),
                label,
                property,
                GraphPath::root(),
                changed
            ),
            Err(CatalogError::IndexPhysicalAlreadyOwned(owner))
        );
        assert_eq!(catalog.get_index(owner), Some(expected));
        for quantization in [
            QuantizationType::Product { num_subvectors: 0 },
            QuantizationType::Product { num_subvectors: 5 },
        ] {
            assert!(matches!(
                catalog.create_index(
                    None,
                    label,
                    property,
                    GraphPath::root(),
                    IndexConfiguration::Vector {
                        config: hnsw.clone(),
                        quantization
                    }
                ),
                Err(CatalogError::InvalidIndex(_))
            ));
        }
        let mut invalid = hnsw;
        invalid.alpha = f32::NAN;
        assert!(matches!(
            catalog.create_index(
                None,
                label,
                property,
                GraphPath::root(),
                IndexConfiguration::Vector {
                    config: invalid,
                    quantization: QuantizationType::None
                }
            ),
            Err(CatalogError::InvalidIndex(_))
        ));
        assert_eq!(catalog.index_allocator_high_water(), 1);
        Ok(())
    }
}
