//! Bounded, exact logical owner changes carried inside the current WAL record.

use std::io::Write;

use grafeo_common::types::{GraphPath, IndexId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::current_wire::ConfigurationV2;
use super::{
    ANONYMOUS_INDEX_PREFIX, Catalog, CatalogError, CatalogRead, IndexConfiguration,
    IndexDefinition, LabelCatalog, PropertyCatalog,
};

type Result<T> = std::result::Result<T, CatalogError>;
const VERSION: u8 = 1;
const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexOwnerImage {
    pub(crate) id: IndexId,
    pub(crate) name: String,
    pub(crate) graph: GraphPath,
    pub(crate) label: String,
    pub(crate) property: String,
    pub(crate) configuration: IndexConfiguration,
}

/// Declaration order is the feature-independent current wire discriminant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum IndexOwnerChange {
    Create(IndexOwnerImage),
    Drop(IndexOwnerImage),
    Rebuild(IndexOwnerImage),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexOwnerBatch {
    pub(crate) expected_floor: u32,
    pub(crate) next_floor: u32,
    pub(crate) changes: Vec<IndexOwnerChange>,
}

#[derive(Serialize)]
struct WireOwnerRef<'a> {
    id: u32,
    name: &'a str,
    #[serde(with = "grafeo_common::types::graph_path_bytes")]
    graph: &'a GraphPath,
    label: &'a str,
    property: &'a str,
    configuration: ConfigurationV2,
}

#[derive(Deserialize)]
struct WireOwner {
    id: u32,
    name: String,
    #[serde(with = "grafeo_common::types::graph_path_bytes")]
    graph: GraphPath,
    label: String,
    property: String,
    configuration: ConfigurationV2,
}

impl Serialize for IndexOwnerImage {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        WireOwnerRef {
            id: self.id.as_u32(),
            name: &self.name,
            graph: &self.graph,
            label: &self.label,
            property: &self.property,
            configuration: ConfigurationV2::from_live(&self.configuration)
                .map_err(serde::ser::Error::custom)?,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IndexOwnerImage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let wire = WireOwner::deserialize(deserializer)?;
        let image = Self {
            id: IndexId::new(wire.id),
            name: wire.name,
            graph: wire.graph,
            label: wire.label,
            property: wire.property,
            configuration: wire
                .configuration
                .into_live()
                .map_err(serde::de::Error::custom)?,
        };
        image.validate().map_err(serde::de::Error::custom)?;
        Ok(image)
    }
}

#[derive(Serialize)]
struct WireBatchRef<'a> {
    version: u8,
    expected_floor: u32,
    next_floor: u32,
    changes: &'a [IndexOwnerChange],
}

#[derive(Deserialize)]
struct WireBatch {
    version: u8,
    expected_floor: u32,
    next_floor: u32,
    changes: Vec<IndexOwnerChange>,
}

fn invalid(message: impl Into<String>) -> CatalogError {
    CatalogError::InvalidIndex(message.into())
}

impl IndexOwnerImage {
    pub(crate) fn capture(catalog: CatalogRead<'_>, id: IndexId) -> Result<Self> {
        let owner = catalog
            .get_index(id)
            .ok_or(CatalogError::IndexNotFound(id))?;
        let image = Self {
            id,
            name: owner.name,
            graph: owner.key.graph().clone(),
            label: catalog
                .get_label_name(owner.label)
                .ok_or_else(|| invalid("recorded owner label is missing"))?
                .to_string(),
            property: catalog
                .get_property_key_name(owner.property_key)
                .ok_or_else(|| invalid("recorded owner property is missing"))?
                .to_string(),
            configuration: owner.configuration,
        };
        image.validate()?;
        Ok(image)
    }

    fn validate(&self) -> Result<()> {
        if self.name.is_empty() || self.property.is_empty() {
            return Err(invalid("recorded owner name/property must be nonempty"));
        }
        if self.id.as_u32() == u32::MAX {
            return Err(invalid("recorded owner ID has no allocator successor"));
        }
        if self.name.starts_with(ANONYMOUS_INDEX_PREFIX)
            && self.name != format!("{ANONYMOUS_INDEX_PREFIX}{}", self.id.as_u32())
        {
            return Err(invalid(
                "recorded anonymous owner name disagrees with its ID",
            ));
        }
        self.configuration.validate()
    }

    fn require_exact(
        &self,
        owner: Option<IndexDefinition>,
        labels: &LabelCatalog,
        properties: &PropertyCatalog,
    ) -> Result<()> {
        let owner = owner.ok_or(CatalogError::IndexNotFound(self.id))?;
        if owner.id != self.id
            || owner.name != self.name
            || owner.key
                != self
                    .configuration
                    .physical_key(self.graph.clone(), &self.label, &self.property)
            || owner.configuration != self.configuration
            || owner.index_type != self.configuration.index_type()
            || labels.get_name(owner.label).as_deref() != Some(self.label.as_str())
            || properties.get_name(owner.property_key).as_deref() != Some(self.property.as_str())
        {
            return Err(invalid(
                "recorded owner preimage does not match the exact live definition",
            ));
        }
        Ok(())
    }
}

impl IndexOwnerBatch {
    fn validate(&self) -> Result<()> {
        let mut floor = self.expected_floor;
        for change in &self.changes {
            let image = match change {
                IndexOwnerChange::Create(image) => {
                    if image.id.as_u32() != floor {
                        return Err(invalid(
                            "recorded Create IDs must follow the exact allocator floor",
                        ));
                    }
                    floor = floor
                        .checked_add(1)
                        .ok_or(CatalogError::IdExhausted("index"))?;
                    image
                }
                IndexOwnerChange::Drop(image) | IndexOwnerChange::Rebuild(image) => image,
            };
            image.validate()?;
        }
        if floor != self.next_floor {
            return Err(invalid(
                "recorded owner allocator floor does not match its Create sequence",
            ));
        }
        Ok(())
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        struct Bounded(Vec<u8>);
        impl Write for Bounded {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > MAX_BYTES.saturating_sub(self.0.len()) {
                    return Err(std::io::Error::other("owner WAL batch exceeds 16 MiB"));
                }
                self.0
                    .try_reserve(bytes.len())
                    .map_err(std::io::Error::other)?;
                self.0.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut output = Bounded(Vec::new());
        bincode::serde::encode_into_std_write(
            WireBatchRef {
                version: VERSION,
                expected_floor: self.expected_floor,
                next_floor: self.next_floor,
                changes: &self.changes,
            },
            &mut output,
            bincode::config::standard(),
        )
        .map_err(|error| invalid(format!("cannot encode owner WAL batch: {error}")))?;
        // Bincode's decode budget includes allocation/primitive claims, not
        // merely encoded byte length. Never acknowledge bytes our own bounded
        // decoder would refuse during reopen.
        Self::decode(&output.0)?;
        Ok(output.0)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_BYTES || bytes.first() != Some(&VERSION) {
            return Err(invalid("unsupported or oversized owner WAL batch"));
        }
        let (wire, consumed): (WireBatch, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_BYTES>(),
        )
        .map_err(|error| invalid(format!("cannot decode bounded owner WAL batch: {error}")))?;
        if wire.version != VERSION || consumed != bytes.len() {
            return Err(invalid("owner WAL batch version or exact length mismatch"));
        }
        let batch = Self {
            expected_floor: wire.expected_floor,
            next_floor: wire.next_floor,
            changes: wire.changes,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Mutates only an unpublished candidate. Stage the affected dictionaries
    /// and owner maps so malformed late changes leave even that candidate intact.
    pub(crate) fn apply(&self, candidate: &Catalog) -> Result<()> {
        self.validate()?;
        let mut state = candidate.state.write();
        if state.indexes.next_id != self.expected_floor {
            return Err(invalid(
                "owner WAL expected allocator floor does not match the catalog",
            ));
        }
        if self
            .changes
            .iter()
            .all(|change| matches!(change, IndexOwnerChange::Rebuild(_)))
        {
            for change in &self.changes {
                if let IndexOwnerChange::Rebuild(image) = change {
                    image.require_exact(
                        state.indexes.get(image.id),
                        &state.labels,
                        &state.property_keys,
                    )?;
                }
            }
            return Ok(());
        }
        let mut indexes = state.indexes.clone();
        let mut labels = state.labels.clone();
        let mut properties = state.property_keys.clone();
        for change in &self.changes {
            match change {
                IndexOwnerChange::Create(image) => {
                    let label = labels.get_or_create(&image.label)?;
                    let property_key = properties.get_or_create(&image.property)?;
                    indexes.create_exact(IndexDefinition {
                        id: image.id,
                        name: image.name.clone(),
                        label,
                        property_key,
                        key: image.configuration.physical_key(
                            image.graph.clone(),
                            &image.label,
                            &image.property,
                        ),
                        configuration: image.configuration.clone(),
                        index_type: image.configuration.index_type(),
                    })?;
                }
                IndexOwnerChange::Drop(image) => {
                    image.require_exact(indexes.get(image.id), &labels, &properties)?;
                    if !indexes.drop(image.id) {
                        return Err(CatalogError::IndexNotFound(image.id));
                    }
                }
                IndexOwnerChange::Rebuild(image) => {
                    image.require_exact(indexes.get(image.id), &labels, &properties)?;
                }
            }
        }
        if indexes.next_id != self.next_floor {
            return Err(invalid(
                "owner WAL install did not retain the recorded allocator floor",
            ));
        }
        state.indexes = indexes;
        state.labels = labels;
        state.property_keys = properties;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(id: u32, name: &str, graph: GraphPath, property: &str) -> IndexOwnerImage {
        IndexOwnerImage {
            id: IndexId::new(id),
            name: name.into(),
            graph,
            label: "Declared".into(),
            property: property.into(),
            configuration: IndexConfiguration::Property,
        }
    }

    #[test]
    fn exact_ids_paths_rebuild_and_retired_floor_roundtrip() -> Result<()> {
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""]).map_err(|e| invalid(e.to_string()))?,
            GraphPath::from_components(&["a/b"]).map_err(|e| invalid(e.to_string()))?,
            GraphPath::from_components(&["a", "b"]).map_err(|e| invalid(e.to_string()))?,
        ];
        let changes: Vec<_> = paths
            .into_iter()
            .enumerate()
            .map(|(id, path)| {
                IndexOwnerChange::Create(image(
                    u32::try_from(id).unwrap(),
                    &format!("{ANONYMOUS_INDEX_PREFIX}{id}"),
                    path,
                    "value",
                ))
            })
            .collect();
        let batch = IndexOwnerBatch {
            expected_floor: 0,
            next_floor: 4,
            changes,
        };
        assert_eq!(IndexOwnerBatch::decode(&batch.encode()?)?, batch);
        let catalog = Catalog::new();
        batch.apply(&catalog)?;
        for change in &batch.changes {
            let IndexOwnerChange::Create(expected) = change else {
                return Err(invalid("fixture expected Create"));
            };
            assert_eq!(
                IndexOwnerImage::capture(catalog.read().view(), expected.id)?,
                *expected
            );
        }
        let retained = IndexOwnerImage::capture(catalog.read().view(), IndexId::new(2))?;
        let maintenance = IndexOwnerBatch {
            expected_floor: 4,
            next_floor: 4,
            changes: vec![
                IndexOwnerChange::Rebuild(retained.clone()),
                IndexOwnerChange::Drop(retained.clone()),
            ],
        };
        IndexOwnerBatch::decode(&maintenance.encode()?)?.apply(&catalog)?;
        assert_eq!(catalog.index_allocator_high_water(), 4);
        assert!(catalog.get_index(retained.id).is_none());
        assert!(maintenance.apply(&catalog).is_err());
        Ok(())
    }

    #[test]
    fn malformed_batch_and_late_duplicate_leave_candidate_unchanged() -> Result<()> {
        let catalog = Catalog::new();
        let first = image(0, "first", GraphPath::root(), "value");
        let batch = IndexOwnerBatch {
            expected_floor: 0,
            next_floor: 1,
            changes: vec![IndexOwnerChange::Create(first.clone())],
        };
        let mut bytes = batch.encode()?;
        bytes.push(0);
        assert!(IndexOwnerBatch::decode(&bytes).is_err());
        assert!(IndexOwnerBatch::decode(&[0, 255]).is_err());
        assert!(IndexOwnerBatch::decode(&vec![VERSION; MAX_BYTES + 1]).is_err());
        let before = catalog.encode_current_state_v2().map_err(invalid)?;
        for malformed in [
            IndexOwnerBatch {
                expected_floor: 1,
                next_floor: 1,
                changes: vec![],
            },
            IndexOwnerBatch {
                expected_floor: 0,
                next_floor: 1,
                changes: vec![IndexOwnerChange::Create(image(
                    0,
                    "",
                    GraphPath::root(),
                    "value",
                ))],
            },
            IndexOwnerBatch {
                expected_floor: 0,
                next_floor: 1,
                changes: vec![IndexOwnerChange::Create(image(
                    0,
                    "empty",
                    GraphPath::root(),
                    "",
                ))],
            },
            IndexOwnerBatch {
                next_floor: 2,
                ..batch.clone()
            },
            IndexOwnerBatch {
                expected_floor: 0,
                next_floor: 2,
                changes: vec![
                    IndexOwnerChange::Create(first.clone()),
                    IndexOwnerChange::Create(image(1, "first", GraphPath::root(), "other")),
                ],
            },
            IndexOwnerBatch {
                expected_floor: 0,
                next_floor: 2,
                changes: vec![
                    IndexOwnerChange::Create(first.clone()),
                    IndexOwnerChange::Create(image(1, "second", GraphPath::root(), "value")),
                ],
            },
        ] {
            assert!(malformed.apply(&catalog).is_err());
            assert_eq!(catalog.encode_current_state_v2().map_err(invalid)?, before);
        }
        batch.apply(&catalog)?;
        let before = catalog.encode_current_state_v2().map_err(invalid)?;
        let mut wrong = first;
        wrong.configuration = IndexConfiguration::BTree;
        for change in [
            IndexOwnerChange::Drop(wrong.clone()),
            IndexOwnerChange::Rebuild(wrong),
        ] {
            assert!(
                IndexOwnerBatch {
                    expected_floor: 1,
                    next_floor: 1,
                    changes: vec![change]
                }
                .apply(&catalog)
                .is_err()
            );
            assert_eq!(catalog.encode_current_state_v2().map_err(invalid)?, before);
        }
        Ok(())
    }

    #[test]
    fn encoding_refuses_bytes_exceeding_its_recovery_budget() -> Result<()> {
        let mut owner = image(0, "x", GraphPath::root(), "value");
        owner.property = "p".repeat(MAX_BYTES - 48);
        let batch = IndexOwnerBatch {
            expected_floor: 0,
            next_floor: 1,
            changes: vec![IndexOwnerChange::Create(owner)],
        };
        let raw = bincode::serde::encode_to_vec(
            WireBatchRef {
                version: VERSION,
                expected_floor: 0,
                next_floor: 1,
                changes: &batch.changes,
            },
            bincode::config::standard(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        assert!(
            raw.len() < MAX_BYTES,
            "byte budget alone accepts this batch"
        );
        assert!(
            IndexOwnerBatch::decode(&raw).is_err(),
            "decode claims include memory and primitive widths"
        );
        assert!(
            batch.encode().is_err(),
            "refuse before any WAL frame/commit marker"
        );
        Ok(())
    }

    #[cfg(all(feature = "text-index", feature = "vector-index"))]
    #[test]
    fn resolved_text_vector_configuration_bits_survive_codec_and_rebuild() -> Result<()> {
        use grafeo_core::index::{
            text::BM25Config,
            vector::{DistanceMetric, HnswConfig, QuantizationType},
        };
        let mut text = image(0, "text", GraphPath::root(), "body");
        text.configuration = IndexConfiguration::Text {
            config: BM25Config { k1: 1.7, b: 0.42 },
            min_token_length: 4,
        };
        let mut vector = image(1, "vector", GraphPath::root(), "embedding");
        let mut config = HnswConfig::new(8, DistanceMetric::Manhattan);
        config.m = 7;
        config.m_max = 19;
        config.ef_construction = 31;
        config.ef = 23;
        config.ml = 0.31;
        config.alpha = 1.3;
        config.max_elements = Some(1234);
        vector.configuration = IndexConfiguration::Vector {
            config,
            quantization: QuantizationType::Product { num_subvectors: 2 },
        };
        let batch = IndexOwnerBatch {
            expected_floor: 0,
            next_floor: 2,
            changes: vec![
                IndexOwnerChange::Create(text.clone()),
                IndexOwnerChange::Create(vector.clone()),
            ],
        };
        let decoded = IndexOwnerBatch::decode(&batch.encode()?)?;
        assert_eq!(decoded, batch);
        let catalog = Catalog::new();
        decoded.apply(&catalog)?;
        IndexOwnerBatch {
            expected_floor: 2,
            next_floor: 2,
            changes: vec![
                IndexOwnerChange::Rebuild(text.clone()),
                IndexOwnerChange::Rebuild(vector.clone()),
            ],
        }
        .apply(&catalog)?;
        assert_eq!(
            IndexOwnerImage::capture(catalog.read().view(), text.id)?,
            text
        );
        assert_eq!(
            IndexOwnerImage::capture(catalog.read().view(), vector.id)?,
            vector
        );
        Ok(())
    }
}
