//! Recursive owner-bearing catalog sections and current persistence admission.
use super::catalog_wire::{CATALOG_SECTION_VERSION, decode_graph_exact_catalog_payload};
#[cfg(test)]
use super::catalog_wire::{
    CatalogPayloadVersion, CatalogSnapshotV7, classify_catalog_payload, graph_exact_catalog_epoch,
};
use crate::catalog::{Catalog, CatalogRead, CatalogWorkspace, IndexConfiguration};
use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EpochId, GraphPath, NodeId, PropertyKey};
use grafeo_common::utils::error::{Error, Result};
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_core::graph::lpg::decode_index_key;
use grafeo_core::graph::lpg::{
    IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey, IndexRegistryWorkspace, LpgStore,
    PhysicalIndexKey, StoreIndexEdits, prepare_index_registry_batch,
};
use std::sync::Arc;
#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexPopulation {
    ExactImages,
    Rebuild,
}
#[cfg(test)]
pub(super) fn current_index_preparation_test_point() -> Result<()> {
    CURRENT_INDEX_PREPARATION_FAILURE.with(|remaining| match remaining.get() {
        Some(0) => {
            remaining.set(None);
            Err(Error::Serialization(
                "injected current index preparation failure".into(),
            ))
        }
        Some(count) => {
            remaining.set(Some(count - 1));
            Ok(())
        }
        None => Ok(()),
    })
}
pub struct CatalogSection {
    catalog: Arc<Catalog>,
    store: Arc<LpgStore>,
    named_graphs: Vec<(GraphPath, Arc<LpgStore>)>,
    epoch_fn: Box<dyn Fn() -> u64 + Send + Sync>,
    expected_graph_exact_epoch: Option<u64>,
    /// Recovery-only Layered base-node tombstones. Legacy compact property
    /// columns do not carry these delete epochs, so Catalog7 feeds them into
    /// detached property-index image normalization before publication.
    property_index_deletions: Vec<(NodeId, EpochId)>,
    #[cfg(feature = "vector-index")]
    quantized_vector_keys: std::collections::BTreeSet<PhysicalIndexKey>,
}
impl CatalogSection {
    pub(crate) fn new_with_graphs(
        catalog: Arc<Catalog>,
        graphs: Vec<(GraphPath, Arc<LpgStore>)>,
        epoch_fn: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Result<Self> {
        let (path, store) = graphs
            .first()
            .ok_or_else(|| Error::Serialization("catalog graph cut is empty".into()))?;
        if !path.components().is_empty() {
            return Err(Error::Serialization(
                "catalog graph cut must start with root".into(),
            ));
        }
        let section = Self {
            catalog,
            store: Arc::clone(store),
            named_graphs: graphs.into_iter().skip(1).collect(),
            epoch_fn: Box::new(epoch_fn),
            expected_graph_exact_epoch: None,
            property_index_deletions: Vec::new(),
            #[cfg(feature = "vector-index")]
            quantized_vector_keys: std::collections::BTreeSet::new(),
        };
        section.validate_named_graph_snapshot()?;
        Ok(section)
    }
    #[cfg(any(feature = "grafeo-file", test))]
    pub(crate) fn with_expected_graph_exact_epoch(mut self, epoch: u64) -> Self {
        self.expected_graph_exact_epoch = Some(epoch);
        self
    }

    /// Supplies persisted Layered tombstones for root property-index rebuilds.
    /// The list is borrowed only during detached Catalog7 preparation and is
    /// never retained in graph storage.
    #[cfg(all(feature = "compact-store", feature = "grafeo-file"))]
    pub(crate) fn with_property_index_deletions(
        mut self,
        deletions: Vec<(NodeId, EpochId)>,
    ) -> Self {
        self.property_index_deletions = deletions;
        self
    }

    /// Selects concrete private Vector targets from validated exact images.
    /// Logical catalog configuration remains authoritative and is checked by
    /// the exact decoder before the detached database is published.
    #[cfg(feature = "vector-index")]
    pub(super) fn with_vector_payloads<'data>(
        mut self,
        payloads: impl IntoIterator<Item = &'data [u8]>,
    ) -> Result<Self> {
        for payload in payloads {
            for key in
                grafeo_core::index::vector::VectorStoreSection::payload_quantized_keys(payload)?
            {
                if !self.quantized_vector_keys.insert(key) {
                    return Err(Error::Serialization(
                        "exact Vector images repeat a quantized target".into(),
                    ));
                }
            }
        }
        Ok(self)
    }

    /// Encodes an already-retained catalog read without recursively locking it.
    pub(super) fn serialize_from_read(
        &self,
        catalog: CatalogRead<'_>,
        epoch: u64,
    ) -> Result<Vec<u8>> {
        self.validate_named_graph_snapshot()?;
        self.validate_owner_bijection(catalog)?;
        super::catalog_wire::encode_catalog_read(catalog, epoch)
    }

    /// Installs catalog-owned targets into an unpublished, empty-registry graph set.
    /// Exact Text/Vector sections must finish before the database is exposed.
    pub(super) fn install_unpublished_catalog(&self, candidate: Catalog) -> Result<()> {
        self.install_unpublished_catalog_with_population(candidate, IndexPopulation::ExactImages)
    }

    /// Rebuilds physical indexes from the detached target's retained corpus,
    /// preserving the supplied canonical owners and allocator state.
    pub(super) fn install_unpublished_catalog_rebuilt(&self, candidate: Catalog) -> Result<()> {
        self.install_unpublished_catalog_with_population(candidate, IndexPopulation::Rebuild)
    }

    fn install_unpublished_catalog_with_population(
        &self,
        candidate: Catalog,
        population: IndexPopulation,
    ) -> Result<()> {
        self.validate_named_graph_snapshot()?;
        self.validate_current_targets(candidate.read().view())?;
        Self::validate_catalog_restore_target(self.catalog.read().view())?;
        let edits = self.prepare_current_indexes(&candidate, population)?;
        let mut registry_workspace = IndexRegistryWorkspace::new(edits);
        let mut catalog_workspace = CatalogWorkspace::replacement(candidate);
        let registry = prepare_index_registry_batch(&mut registry_workspace)?;
        let catalog = self
            .catalog
            .prepare_replacement(&mut catalog_workspace)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        Self::validate_catalog_restore_target(catalog.preimage())?;
        let registry_fence = registry.install();
        let catalog_fence = catalog.install();
        catalog_fence.finish();
        drop(registry_fence);
        Ok(())
    }
    fn graph_store(&self, path: &GraphPath) -> Option<&Arc<LpgStore>> {
        if path.components().is_empty() {
            return Some(&self.store);
        }
        self.named_graphs
            .binary_search_by(|(candidate, _)| candidate.cmp(path))
            .ok()
            .map(|index| &self.named_graphs[index].1)
    }
    fn validate_named_graph_snapshot(&self) -> Result<()> {
        let mut previous = GraphPath::root();
        for (path, _) in &self.named_graphs {
            if path <= &previous {
                return Err(Error::Serialization(
                    "catalog graph cut must be strictly sorted and unique".into(),
                ));
            }
            previous = path.clone();
        }
        for (path, store) in std::iter::once((GraphPath::root(), &self.store)).chain(
            self.named_graphs
                .iter()
                .map(|(path, store)| (path.clone(), store)),
        ) {
            if let Some(parent) = path
                .parent()
                .map_err(|error| Error::Serialization(error.to_string()))?
            {
                let parent_store = self.graph_store(&parent).ok_or_else(|| {
                    Error::Serialization("catalog graph cut has no parent".into())
                })?;
                let name = path
                    .components()
                    .last()
                    .ok_or_else(|| Error::Serialization("catalog child has no name".into()))?;
                if parent_store
                    .graph(name)
                    .is_none_or(|actual| !Arc::ptr_eq(&actual, store))
                {
                    return Err(Error::Serialization(format!(
                        "catalog graph cut identity mismatch at {path:?}"
                    )));
                }
            }
            for (name, child) in store.named_graph_entries() {
                let child_path = path
                    .child(&name)
                    .map_err(|error| Error::Serialization(error.to_string()))?;
                if self
                    .graph_store(&child_path)
                    .is_none_or(|actual| !Arc::ptr_eq(actual, &child))
                {
                    return Err(Error::Serialization(format!(
                        "catalog graph cut omits or mismatches {child_path:?}"
                    )));
                }
            }
        }
        Ok(())
    }
    fn validate_current_targets(&self, catalog: CatalogRead<'_>) -> Result<()> {
        for (path, _) in catalog.all_graph_type_bindings() {
            if self.graph_store(&path).is_none() {
                return Err(Error::Serialization(format!(
                    "catalog binding targets missing graph {path:?}"
                )));
            }
        }
        for namespace in catalog.schema_names() {
            let name = format!("{namespace}/__default__");
            let path = GraphPath::from_components(&[&name])
                .map_err(|error| Error::Serialization(error.to_string()))?;
            if self.graph_store(&path).is_none() {
                return Err(Error::Serialization(format!(
                    "catalog namespace requires graph {path:?}"
                )));
            }
        }
        for owner in catalog.all_indexes() {
            if self.graph_store(owner.key.graph()).is_none() {
                return Err(Error::Serialization(format!(
                    "catalog owner {} targets a missing graph",
                    owner.id
                )));
            }
        }
        Ok(())
    }
    fn validate_owner_bijection(&self, catalog: CatalogRead<'_>) -> Result<()> {
        self.validate_current_targets(catalog)?;
        let mut owners: std::collections::HashMap<_, _> = catalog
            .all_indexes()
            .into_iter()
            .map(|owner| (owner.key.clone(), owner.configuration))
            .collect();
        for (path, store) in std::iter::once((GraphPath::root(), &self.store)).chain(
            self.named_graphs
                .iter()
                .map(|(path, store)| (path.clone(), store)),
        ) {
            for property in store.property_index_keys() {
                let key = PhysicalIndexKey::property(path.clone(), &property);
                if !matches!(
                    owners.remove(&key),
                    Some(IndexConfiguration::Property | IndexConfiguration::BTree)
                ) {
                    return Err(Error::Serialization(format!(
                        "physical Property index has no matching owner: {key:?}"
                    )));
                }
            }
            #[cfg(feature = "vector-index")]
            for (key, index) in store.vector_index_entries() {
                let (label, property) = decode_index_key(&key)
                    .ok_or_else(|| Error::Serialization("invalid physical Vector key".into()))?;
                let key = PhysicalIndexKey::vector(path.clone(), label, property);
                let quantization = match index.quantization_type() {
                    Some(kind) => kind,
                    None => grafeo_core::index::vector::QuantizationType::None,
                };
                let configuration = IndexConfiguration::Vector {
                    config: index.config().clone(),
                    quantization,
                };
                if owners.remove(&key).as_ref() != Some(&configuration) {
                    return Err(Error::Serialization(format!(
                        "physical Vector owner/config mismatch: {key:?}"
                    )));
                }
            }
            #[cfg(feature = "text-index")]
            for (key, index) in store.text_index_entries() {
                let (label, property) = decode_index_key(&key)
                    .ok_or_else(|| Error::Serialization("invalid physical Text key".into()))?;
                let key = PhysicalIndexKey::text(path.clone(), label, property);
                let index = index.read();
                let Some(IndexConfiguration::Text {
                    config,
                    min_token_length,
                }) = owners.remove(&key)
                else {
                    return Err(Error::Serialization(format!(
                        "physical Text index has no owner: {key:?}"
                    )));
                };
                let actual = index.config();
                if actual.k1.to_bits() != config.k1.to_bits()
                    || actual.b.to_bits() != config.b.to_bits()
                    || !index.has_simple_tokenizer(min_token_length)
                {
                    return Err(Error::Serialization(format!(
                        "physical Text owner/config mismatch: {key:?}"
                    )));
                }
            }
        }
        if !owners.is_empty() {
            return Err(Error::Serialization(
                "catalog owner has no physical index".into(),
            ));
        }
        Ok(())
    }
    fn prepare_current_indexes(
        &self,
        catalog: &Catalog,
        population: IndexPopulation,
    ) -> Result<Vec<StoreIndexEdits<'_>>> {
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        let _ = population;
        let mut owners = catalog.all_indexes();
        owners.sort_unstable_by(|left, right| {
            (left.key.graph(), left.id).cmp(&(right.key.graph(), right.id))
        });
        let mut first_owner = 0;
        let mut stores = Vec::new();
        for (path, store) in std::iter::once((GraphPath::root(), &self.store)).chain(
            self.named_graphs
                .iter()
                .map(|(path, store)| (path.clone(), store)),
        ) {
            #[cfg(test)]
            current_index_preparation_test_point()?;
            let occupied = !store.property_index_keys().is_empty();
            #[cfg(feature = "vector-index")]
            let occupied = occupied || !store.vector_index_entries().is_empty();
            #[cfg(feature = "text-index")]
            let occupied = occupied || !store.text_index_entries().is_empty();
            if occupied {
                return Err(Error::Serialization(format!(
                    "Catalog7 registry restore target is not empty: {path:?}"
                )));
            }
            let mut edits = Vec::new();
            let mut end_owner = first_owner;
            while owners
                .get(end_owner)
                .is_some_and(|owner| owner.key.graph() == &path)
            {
                end_owner += 1;
            }
            for owner in &owners[first_owner..end_owner] {
                if population == IndexPopulation::Rebuild {
                    owner
                        .configuration
                        .validate()
                        .map_err(|error| Error::Serialization(error.to_string()))?;
                }
                #[cfg(any(feature = "text-index", feature = "vector-index"))]
                let label = catalog
                    .get_label_name(owner.label)
                    .ok_or_else(|| Error::Serialization("owner label missing".into()))?;
                let property = catalog
                    .get_property_key_name(owner.property_key)
                    .ok_or_else(|| Error::Serialization("owner property missing".into()))?;
                let (key, contents) = match &owner.configuration {
                    IndexConfiguration::Property | IndexConfiguration::BTree => {
                        let key = PropertyKey::new(property.as_ref());
                        let mut image = store.property_index_image(property.as_ref())?;
                        if path.components().is_empty() {
                            image.close_deleted_nodes(&self.property_index_deletions)?;
                        }
                        (
                            IndexRegistryKey::Property(key),
                            IndexRegistryContents::PropertyHistory(image),
                        )
                    }
                    #[cfg(feature = "text-index")]
                    IndexConfiguration::Text {
                        config,
                        min_token_length,
                    } => {
                        let mut index =
                            grafeo_core::index::text::InvertedIndex::with_simple_tokenizer(
                                config.clone(),
                                *min_token_length,
                            );
                        if population == IndexPopulation::Rebuild {
                            populate_rebuilt_text(&mut index, store, &label, &property)?;
                        }
                        (
                            IndexRegistryKey::Text {
                                label: label.to_string(),
                                property: property.to_string(),
                            },
                            IndexRegistryContents::Text(index),
                        )
                    }
                    #[cfg(feature = "vector-index")]
                    IndexConfiguration::Vector {
                        config,
                        quantization,
                    } => {
                        use grafeo_core::index::vector::{
                            HnswIndex, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
                        };
                        let index = match quantization {
                            QuantizationType::None
                                if !self.quantized_vector_keys.contains(&owner.key) =>
                            {
                                VectorIndexKind::Hnsw(HnswIndex::new(config.clone()))
                            }
                            kind => VectorIndexKind::Quantized(QuantizedHnswIndex::new(
                                config.clone(),
                                *kind,
                            )),
                        };
                        if population == IndexPopulation::Rebuild {
                            populate_rebuilt_vector(&index, store, &label, &property)?;
                        }
                        (
                            IndexRegistryKey::Vector {
                                label: label.to_string(),
                                property: property.to_string(),
                            },
                            IndexRegistryContents::Vector(index),
                        )
                    }
                };
                edits.push(IndexRegistryEdit::Create { key, contents });
            }
            first_owner = end_owner;
            stores.push(StoreIndexEdits { store, edits });
        }
        Ok(stores)
    }
}

/// Visits each retained transition in entity existence, label membership or
/// property value. The target is an unpublished, fully restored hot store;
/// callers must not pass an overlay without its compact predecessor restored.
#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn visit_rebuilt_property_states(
    store: &LpgStore,
    label: &str,
    property: &str,
    mut visit: impl FnMut(NodeId, EpochId, Option<&grafeo_common::types::Value>) -> Result<()>,
) -> Result<()> {
    use grafeo_common::memory::AllocError;
    let floor = store.retained_history_floor();
    let current = store.current_epoch();
    if floor > current || current == EpochId::PENDING {
        return Err(Error::Serialization(
            "invalid index rebuild history boundary".into(),
        ));
    }
    let mut ids = store.all_node_ids();
    ids.sort_unstable();
    for id in ids {
        let mut lives = store.get_node_history(id);
        let mut labels = store.node_label_history(id);
        let mut values = store.node_property_history_for_key(id, property);
        lives.reverse();
        lives.sort_by_key(|(created, _, _)| *created);
        labels.sort_by_key(|(epoch, _)| *epoch);
        values.sort_by_key(|(epoch, _)| *epoch);
        let capacity = lives
            .len()
            .checked_mul(2)
            .and_then(|count| count.checked_add(labels.len()))
            .and_then(|count| count.checked_add(values.len()))
            .and_then(|count| count.checked_add(2))
            .ok_or(AllocError::OutOfMemory)?;
        let mut epochs = Vec::new();
        epochs
            .try_reserve_exact(capacity)
            .map_err(|_| AllocError::OutOfMemory)?;
        epochs.extend([floor, current]);
        for (created, deleted, _) in &lives {
            epochs.push(*created);
            epochs.extend(*deleted);
        }
        epochs.extend(labels.iter().map(|(epoch, _)| *epoch));
        epochs.extend(values.iter().map(|(epoch, _)| *epoch));
        if epochs.contains(&EpochId::PENDING) {
            return Err(Error::Serialization(
                "uncommitted index rebuild history".into(),
            ));
        }
        epochs.retain(|epoch| *epoch >= floor && *epoch <= current);
        epochs.sort_unstable();
        epochs.dedup();
        for epoch in epochs {
            let life = lives
                .partition_point(|(created, _, _)| *created <= epoch)
                .checked_sub(1)
                .and_then(|index| lives.get(index));
            let alive = life.is_some_and(|(_, deleted, _)| deleted.is_none_or(|end| epoch < end));
            let member = label.is_empty()
                || labels
                    .partition_point(|(at, _)| *at <= epoch)
                    .checked_sub(1)
                    .and_then(|index| labels.get(index))
                    .is_some_and(|(_, names)| names.iter().any(|name| name.as_str() == label));
            let value = if alive && member {
                values
                    .partition_point(|(at, _)| *at <= epoch)
                    .checked_sub(1)
                    .and_then(|index| values.get(index))
                    .map(|(_, value)| value)
            } else {
                None
            };
            visit(id, epoch, value)?;
        }
    }
    Ok(())
}

#[cfg(feature = "text-index")]
fn populate_rebuilt_text(
    index: &mut grafeo_core::index::text::InvertedIndex,
    store: &LpgStore,
    label: &str,
    property: &str,
) -> Result<()> {
    use grafeo_common::memory::AllocError;
    use grafeo_common::types::{ArcStr, Value};
    let mut events: Vec<(EpochId, NodeId, Option<ArcStr>)> = Vec::new();
    let mut previous: Option<(NodeId, Option<ArcStr>)> = None;
    visit_rebuilt_property_states(store, label, property, |id, epoch, value| {
        let text = match value {
            Some(Value::String(text)) => Some(text),
            _ => None,
        };
        let old = previous
            .as_ref()
            .filter(|(old_id, _)| *old_id == id)
            .and_then(|(_, text)| text.as_ref());
        if old != text {
            events.try_reserve(1).map_err(|_| AllocError::OutOfMemory)?;
            events.push((epoch, id, text.cloned()));
        }
        previous = Some((id, text.cloned()));
        Ok(())
    })?;
    // Aggregate deltas must remain globally ordered, including transitions
    // from distinct documents at the same epoch.
    events.sort_unstable_by_key(|(epoch, id, _)| (*epoch, *id));
    for (epoch, id, text) in events {
        if let Some(text) = text {
            index.insert_versioned(id, text.as_str(), epoch, None);
        } else {
            index.remove_versioned(id, epoch, None);
        }
    }
    index.gc(store.retained_history_floor())?;
    Ok(())
}

#[cfg(feature = "vector-index")]
fn populate_rebuilt_vector(
    index: &grafeo_core::index::vector::VectorIndexKind,
    store: &LpgStore,
    label: &str,
    property: &str,
) -> Result<()> {
    use grafeo_common::memory::AllocError;
    use grafeo_core::index::vector::value_to_vector;
    // Retain one routing payload for every historically eligible identity.
    // Snapshot searches use the graph's exact historical accessor, including
    // quantized rescoring; current-ineligible vertices become tombstones only
    // after the entire routing graph has been built.
    let mut rows: Vec<(NodeId, Arc<[f32]>, bool)> = Vec::new();
    visit_rebuilt_property_states(store, label, property, |id, _, value| {
        let vector = value.and_then(value_to_vector);
        if let Some(vector) = &vector {
            if vector.len() != index.config().dimensions {
                return Err(Error::Serialization(format!(
                    "Vector dimension mismatch rebuilding node {id}: expected {}, found {}",
                    index.config().dimensions,
                    vector.len()
                )));
            }
            if let Some((last_id, retained, live)) = rows.last_mut()
                && *last_id == id
            {
                *retained = Arc::clone(vector);
                *live = true;
            } else {
                rows.try_reserve(1).map_err(|_| AllocError::OutOfMemory)?;
                rows.push((id, Arc::clone(vector), true));
            }
        } else if let Some((last_id, _, live)) = rows.last_mut()
            && *last_id == id
        {
            *live = false;
        }
        Ok(())
    })?;
    if index
        .config()
        .max_elements
        .is_some_and(|max| rows.len() > max)
    {
        return Err(Error::InvalidValue(
            "retained vector rebuild corpus exceeds configured max_elements".into(),
        ));
    }
    let accessor = |id| {
        rows.binary_search_by_key(&id, |(id, _, _)| *id)
            .ok()
            .map(|position| Arc::clone(&rows[position].1))
    };
    for (id, vector, _) in &rows {
        index.insert(*id, vector, &accessor);
    }
    for (id, _, live) in rows {
        if !live {
            index.remove(id);
        }
    }
    Ok(())
}

impl CatalogSection {
    #[cfg(feature = "grafeo-file")]
    pub(super) fn validate_catalog_index_persistence(catalog: CatalogRead<'_>) -> Result<()> {
        if catalog.index_allocator_high_water() != 0 {
            return Err(Error::Serialization(
                "current persistence formats cannot preserve canonical index owners, resolved configuration, or index allocator high-water".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_catalog_restore_target(catalog: CatalogRead<'_>) -> Result<()> {
        if catalog.label_count() != 0
            || catalog.property_key_count() != 0
            || catalog.edge_type_count() != 0
            || catalog.index_allocator_high_water() != 0
            || !catalog.all_node_type_defs().is_empty()
            || !catalog.all_edge_type_defs().is_empty()
            || !catalog.all_graph_type_defs().is_empty()
            || !catalog.all_procedure_defs().is_empty()
            || !catalog.schema_names().is_empty()
            || !catalog.all_graph_type_bindings().is_empty()
            || !catalog.all_named_constraints().is_empty()
        {
            return Err(Error::Serialization(
                "Catalog restore requires a pristine detached catalog target".to_string(),
            ));
        }
        Ok(())
    }
}
impl Section for CatalogSection {
    fn section_type(&self) -> SectionType {
        SectionType::Catalog
    }
    fn version(&self) -> u8 {
        CATALOG_SECTION_VERSION
    }
    fn serialize(&self) -> Result<Vec<u8>> {
        let catalog = self.catalog.read();
        self.serialize_from_read(catalog.view(), (self.epoch_fn)())
    }
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let snapshot = decode_graph_exact_catalog_payload(data)?;
        if self
            .expected_graph_exact_epoch
            .is_some_and(|expected| expected != snapshot.epoch)
        {
            return Err(Error::Serialization(
                "Catalog7 epoch does not match enclosing WorldCut".into(),
            ));
        }
        let candidate =
            Catalog::from_current_state_v2(snapshot.state).map_err(Error::Serialization)?;
        self.install_unpublished_catalog(candidate)
    }
    fn is_dirty(&self) -> bool {
        false
    }
    fn mark_clean(&self) {}
    fn memory_usage(&self) -> usize {
        4096
    }
}
#[cfg(test)]
thread_local! {
    static CURRENT_INDEX_PREPARATION_FAILURE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) struct CurrentIndexPreparationFailure(Option<usize>);

#[cfg(test)]
impl CurrentIndexPreparationFailure {
    pub(super) fn after_successes(count: usize) -> Self {
        Self(CURRENT_INDEX_PREPARATION_FAILURE.with(|remaining| remaining.replace(Some(count))))
    }
}

#[cfg(test)]
impl Drop for CurrentIndexPreparationFailure {
    fn drop(&mut self) {
        CURRENT_INDEX_PREPARATION_FAILURE.with(|remaining| remaining.set(self.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{GraphTypeDefinition, NamedConstraintDefinition, NamedConstraintKind};
    use grafeo_common::types::Value;
    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn section(catalog: Arc<Catalog>, root: Arc<LpgStore>) -> Result<CatalogSection> {
        let graphs = grafeo_core::graph::lpg::LpgStoreSection::new(root).capture_graphs()?;
        CatalogSection::new_with_graphs(catalog, graphs, || 9)
    }
    fn fixture() -> Result<Arc<LpgStore>> {
        let root = Arc::new(LpgStore::new()?);
        root.create_graph("")?;
        root.create_graph("a/b")?;
        root.graph_or_create("a")?.create_graph("b")?;
        for store in [
            Arc::clone(&root),
            root.graph_or_create("")?,
            root.graph_or_create("a/b")?,
            root.graph_or_create("a")?.graph_or_create("b")?,
        ] {
            let id = store.create_node(&["Item"]);
            store.set_node_property(id, "value", Value::Int64(7));
        }
        Ok(root)
    }
    fn own_property(catalog: &Catalog, path: GraphPath) -> Result<grafeo_common::types::IndexId> {
        let label = catalog
            .get_or_create_label("Item")
            .map_err(|error| Error::Serialization(error.to_string()))?;
        let property = catalog
            .get_or_create_property_key("value")
            .map_err(|error| Error::Serialization(error.to_string()))?;
        catalog
            .create_index(None, label, property, path, IndexConfiguration::Property)
            .map_err(|error| Error::Serialization(error.to_string()))
    }

    #[test]
    fn catalog7_snapshot_capture_reuses_read_and_checks_exact_epoch() -> TestResult {
        let catalog = Arc::new(Catalog::new());
        let root = fixture()?;
        own_property(&catalog, GraphPath::root())?;
        root.create_property_index("value");
        let source = section(Arc::clone(&catalog), root)?;
        let held = catalog.read();
        let bytes = source.serialize_from_read(held.view(), 9)?;
        drop(held);
        assert_eq!(bytes, source.serialize()?);
        let candidate = crate::database::catalog_wire::decode_catalog(&bytes, 9)?;
        assert_eq!(candidate.all_indexes(), catalog.all_indexes());
        assert_eq!(
            candidate.index_allocator_high_water(),
            catalog.index_allocator_high_water()
        );
        assert!(crate::database::catalog_wire::decode_catalog(&bytes, 8).is_err());
        Ok(())
    }

    #[test]
    fn catalog7_recursive_owners_bindings_and_floor_round_trip() -> TestResult {
        let catalog = Arc::new(Catalog::new());
        catalog.register_graph_type(GraphTypeDefinition {
            name: "Open".into(),
            allowed_node_types: Vec::new(),
            allowed_edge_types: Vec::new(),
            open: true,
        })?;
        let root = fixture()?;
        let source = section(Arc::clone(&catalog), root)?;
        let gap = own_property(&catalog, GraphPath::root())?;
        assert!(catalog.drop_index(gap));
        for path in [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["a/b"])?,
            GraphPath::from_components(&["a", "b"])?,
        ] {
            own_property(&catalog, path.clone())?;
            catalog.bind_graph_type(&path, "Open".into())?;
            source
                .graph_store(&path)
                .ok_or("missing source graph")?
                .create_property_index("value");
        }
        let bytes = source.serialize()?;
        assert_eq!(source.version(), 7);
        assert_eq!(
            classify_catalog_payload(&bytes)?,
            CatalogPayloadVersion::GraphExactV7
        );
        assert_eq!(graph_exact_catalog_epoch(&bytes)?, 9);
        let restored_catalog = Arc::new(Catalog::new());
        let mut target = section(Arc::clone(&restored_catalog), fixture()?)?;
        target.deserialize(&bytes)?;
        assert_eq!(target.serialize()?, bytes);
        assert_eq!(restored_catalog.index_allocator_high_water(), 5);
        for owner in catalog.all_indexes() {
            assert_eq!(restored_catalog.get_index(owner.id), Some(owner.clone()));
            let store = target
                .graph_store(owner.key.graph())
                .ok_or("missing restored graph")?;
            assert_eq!(
                store
                    .find_nodes_by_property("value", &Value::Int64(7))
                    .len(),
                1
            );
        }
        for owner in catalog.all_indexes() {
            assert!(
                source
                    .graph_store(owner.key.graph())
                    .ok_or("missing source graph")?
                    .drop_property_index("value")
            );
            assert!(catalog.drop_index(owner.id));
        }
        let empty_bytes = source.serialize()?;
        let floor_catalog = Arc::new(Catalog::new());
        let mut floor_target = section(Arc::clone(&floor_catalog), fixture()?)?;
        floor_target.deserialize(&empty_bytes)?;
        assert_eq!(floor_catalog.index_allocator_high_water(), 5);
        assert_eq!(floor_catalog.index_count(), 0);
        Ok(())
    }

    #[test]
    fn catalog7_rejects_predecessors_corrupt_bounds_and_late_missing_owner_target() -> TestResult {
        let source = section(Arc::new(Catalog::new()), fixture()?)?;
        let bytes = source.serialize()?;
        let catalog = Arc::new(Catalog::new());
        let mut target = section(Arc::clone(&catalog), fixture()?)?;
        let before = target.serialize()?;
        for version in 0..7 {
            assert!(target.deserialize(&[version]).is_err());
            assert_eq!(target.serialize()?, before);
        }
        for end in 0..bytes.len() {
            assert!(target.deserialize(&bytes[..end]).is_err());
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(target.deserialize(&trailing).is_err());
        // One label with an enormous declared string length: bounded decoding
        // rejects this tiny input without reserving its advertised payload.
        let mut oversized = vec![7, 2, 1];
        oversized.extend(bincode::serde::encode_to_vec(
            u64::MAX,
            bincode::config::standard(),
        )?);
        let error = target
            .deserialize(&oversized)
            .err()
            .ok_or("oversized string accepted")?;
        assert!(error.to_string().contains("LimitExceeded"), "{error}");
        let incoming = Catalog::new();
        own_property(&incoming, GraphPath::root())?;
        own_property(&incoming, GraphPath::from_components(&["z", "missing"])?)?;
        let invalid = bincode::serde::encode_to_vec(
            CatalogSnapshotV7 {
                version: 7,
                state_version: 2,
                state: incoming.current_state_v2()?,
                epoch: 9,
            },
            bincode::config::standard(),
        )?;
        assert!(target.deserialize(&invalid).is_err());
        assert_eq!(target.serialize()?, before);
        assert!(!target.store.has_property_index("value"));
        assert_eq!(catalog.index_allocator_high_water(), 0);
        let wrong_epoch = target.with_expected_graph_exact_epoch(10);
        target = wrong_epoch;
        assert!(target.deserialize(&source.serialize()?).is_err());
        assert_eq!(target.serialize()?, before);
        Ok(())
    }

    #[test]
    fn catalog7_rejects_physical_orphans_missing_indexes_and_changed_incarnations() -> TestResult {
        let catalog = Arc::new(Catalog::new());
        let root = fixture()?;
        let source = section(Arc::clone(&catalog), Arc::clone(&root))?;
        root.create_property_index("value");
        assert!(source.serialize().is_err());
        let owner = own_property(&catalog, GraphPath::root())?;
        assert!(source.serialize().is_ok());
        assert!(root.drop_property_index("value"));
        assert!(source.serialize().is_err());
        assert!(catalog.drop_index(owner));
        assert!(root.drop_graph("a/b"));
        root.create_graph("a/b")?;
        assert!(source.serialize().is_err());
        Ok(())
    }

    #[test]
    fn catalog7_preserves_named_constraints_and_rejects_late_scope_before_publication() -> TestResult
    {
        let catalog = Arc::new(Catalog::new());
        let source = section(Arc::clone(&catalog), fixture()?)?;
        let constraint = NamedConstraintDefinition {
            name: "item_value".into(),
            label: "Item".into(),
            properties: vec!["value".into()],
            kind: NamedConstraintKind::Unique,
        };
        catalog.create_named_constraint(constraint.clone())?;
        for path in [GraphPath::root(), GraphPath::from_components(&["a", "b"])?] {
            own_property(&catalog, path.clone())?;
            source
                .graph_store(&path)
                .ok_or("missing graph")?
                .create_property_index("value");
        }
        let bytes = source.serialize()?;
        let target_catalog = Arc::new(Catalog::new());
        let mut target = section(Arc::clone(&target_catalog), fixture()?)?;
        let authority = grafeo_core::graph::write_permit::WriteAuthority::new();
        let late = target
            .graph_store(&GraphPath::from_components(&["a", "b"])?)
            .ok_or("missing target")?;
        assert!(late.seal_unframed_writes(&authority));
        assert!(target.deserialize(&bytes).is_err());
        assert!(!target.store.has_property_index("value"));
        assert_eq!(target_catalog.index_allocator_high_water(), 0);
        grafeo_core::graph::write_permit::with_authority(&authority, || {
            target.deserialize(&bytes)
        })?;
        assert_eq!(
            target_catalog.get_named_constraint("item_value"),
            Some(constraint)
        );
        target_catalog.drop_named_constraint("item_value")?;
        assert!(target_catalog.all_named_constraints().is_empty());
        Ok(())
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn catalog7_preserves_text_configuration_and_imports_only_empty_target() -> TestResult {
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        let catalog = Arc::new(Catalog::new());
        let root = fixture()?;
        let label = catalog.get_or_create_label("Item")?;
        let property = catalog.get_or_create_property_key("value")?;
        let config = BM25Config { k1: 1.7, b: 0.4 };
        let owner = catalog.create_index(
            Some("text"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Text {
                config: config.clone(),
                min_token_length: 0,
            },
        )?;
        let mut index = InvertedIndex::with_simple_tokenizer(config.clone(), 0);
        index.insert(grafeo_common::types::NodeId::new(0), "x y z");
        root.add_text_index("Item", "value", Arc::new(parking_lot::RwLock::new(index)));
        let source = section(Arc::clone(&catalog), Arc::clone(&root))?;
        let bytes = source.serialize()?;
        let mut target = section(Arc::new(Catalog::new()), fixture()?)?;
        target.deserialize(&bytes)?;
        assert_eq!(target.catalog.get_index(owner), catalog.get_index(owner));
        let text = target
            .store
            .get_text_index("Item", "value")
            .ok_or("target missing")?;
        assert!(text.read().has_simple_tokenizer(0));
        assert_eq!(
            text.read().len(),
            0,
            "exact Text payload belongs to its physical section"
        );
        assert!(root.remove_text_index("Item", "value"));
        root.add_text_index(
            "Item",
            "value",
            Arc::new(parking_lot::RwLock::new(
                InvertedIndex::with_simple_tokenizer(config, 2),
            )),
        );
        assert!(
            source.serialize().is_err(),
            "tokenizer disagreement must reject"
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "text-index")]
    fn rebuilt_text_replays_lifetimes_labels_properties_and_subset_aggregates() -> TestResult {
        use grafeo_common::types::TransactionId;
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        let store = LpgStore::new()?;
        let first = NodeId::new(1);
        let second = NodeId::new(2);
        let epoch = EpochId::new;
        store.restore_node_history_exact(
            first,
            &[(epoch(2), None)],
            &[
                (epoch(2), vec!["Item".into()]),
                (epoch(4), vec!["Other".into()]),
                (epoch(6), vec!["Item".into()]),
            ],
        )?;
        store.restore_node_history_exact(
            second,
            &[(epoch(2), Some(epoch(5)))],
            &[(epoch(2), vec!["Item".into()])],
        )?;
        store.set_node_property_at_epoch(first, "text", Value::from("xx aa"), epoch(2));
        store.set_node_property_at_epoch(first, "text", Value::from("yy aa"), epoch(3));
        store.set_node_property_at_epoch(first, "text", Value::Null, epoch(7));
        store.set_node_property_at_epoch(second, "text", Value::from("xx"), epoch(2));
        store.sync_epoch(epoch(8));
        let mut index = InvertedIndex::with_simple_tokenizer(BM25Config { k1: 1.7, b: 0.4 }, 2);
        populate_rebuilt_text(&mut index, &store, "Item", "text")?;
        for (at, count) in [(1, 0), (2, 2), (3, 2), (4, 1), (5, 0), (6, 1), (7, 0)] {
            assert_eq!(
                index.doc_count_at(epoch(at), TransactionId::INVALID)?,
                count
            );
        }
        assert_eq!(index.avgdl_at(epoch(2), TransactionId::INVALID)?, 1.5);
        assert_eq!(index.avgdl_at(epoch(4), TransactionId::INVALID)?, 1.0);
        assert!(index.has_simple_tokenizer(2));
        Ok(())
    }

    #[test]
    #[cfg(feature = "vector-index")]
    fn rebuilt_vector_keeps_historical_candidates_after_label_and_property_removal() -> TestResult {
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex,
            VectorIndexKind, value_to_vector,
        };
        let store = LpgStore::new()?;
        let id = NodeId::new(1);
        store.restore_node_history_exact(
            id,
            &[(EpochId::new(2), None)],
            &[
                (EpochId::new(2), vec!["Item".into()]),
                (EpochId::new(4), vec!["Other".into()]),
            ],
        )?;
        store.set_node_property_at_epoch(
            id,
            "vector",
            Value::Vector(Arc::from([1.0_f32, 0.0])),
            EpochId::new(2),
        );
        store.set_node_property_at_epoch(id, "vector", Value::Null, EpochId::new(5));
        store.sync_epoch(EpochId::new(6));
        let config = HnswConfig::new(2, DistanceMetric::Euclidean);
        for index in [
            VectorIndexKind::Hnsw(HnswIndex::new(config.clone())),
            VectorIndexKind::Quantized(QuantizedHnswIndex::new(config, QuantizationType::Scalar)),
        ] {
            populate_rebuilt_vector(&index, &store, "Item", "vector")?;
            assert!(
                !index.contains(id),
                "current label/property removal remains visible"
            );
            let accessor = |candidate| {
                store
                    .get_node_property_at_epoch(
                        candidate,
                        &PropertyKey::new("vector"),
                        EpochId::new(3),
                    )
                    .as_ref()
                    .and_then(value_to_vector)
            };
            let result =
                index.search_visible(&[1.0, 0.0], 1, 8, &|candidate| candidate == id, &accessor);
            assert_eq!(
                result,
                vec![(id, 0.0)],
                "historical payload is supplied by the snapshot"
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    fn rebuilt_indexes_keep_last_same_epoch_state_and_open_successor() -> TestResult {
        let store = LpgStore::new()?;
        let id = NodeId::new(7);
        let epoch = EpochId::new(2);
        // Portable recovery accepts zero-length lives and ordered equal-epoch
        // versions. The last structural life and label/property versions win.
        let labels = (0..32)
            .map(|position| {
                (
                    epoch,
                    vec![if position % 2 == 0 {
                        "Other".into()
                    } else {
                        "Item".into()
                    }],
                )
            })
            .collect::<Vec<_>>();
        store.restore_node_history_exact(id, &[(epoch, Some(epoch)), (epoch, None)], &labels)?;
        for _ in 0..31 {
            store.set_node_property_at_epoch(id, "text", Value::from("discarded"), epoch);
            store.set_node_property_at_epoch(
                id,
                "vector",
                Value::Vector(Arc::from([9.0_f32, 0.0])),
                epoch,
            );
        }
        store.set_node_property_at_epoch(id, "text", Value::from("winner"), epoch);
        store.set_node_property_at_epoch(
            id,
            "vector",
            Value::Vector(Arc::from([1.0_f32, 0.0])),
            epoch,
        );
        store.sync_epoch(EpochId::new(3));
        assert!(store.get_node_at_epoch(id, epoch).is_some());

        #[cfg(feature = "text-index")]
        {
            use grafeo_common::types::TransactionId;
            use grafeo_core::index::text::{BM25Config, InvertedIndex};
            let mut index = InvertedIndex::new(BM25Config::default());
            populate_rebuilt_text(&mut index, &store, "Item", "text")?;
            assert_eq!(index.doc_count_at(epoch, TransactionId::INVALID)?, 1);
            assert_eq!(
                index
                    .search("winner", 10)
                    .iter()
                    .map(|(id, _)| *id)
                    .collect::<Vec<_>>(),
                vec![id]
            );
            assert!(index.search("discarded", 10).is_empty());
        }
        #[cfg(feature = "vector-index")]
        {
            use grafeo_core::index::vector::{
                DistanceMetric, HnswConfig, QuantizationType, QuantizedHnswIndex, VectorIndexKind,
            };
            let index = VectorIndexKind::Quantized(QuantizedHnswIndex::new(
                HnswConfig::new(2, DistanceMetric::Euclidean),
                QuantizationType::Scalar,
            ));
            populate_rebuilt_vector(&index, &store, "Item", "vector")?;
            assert!(index.contains(id));
            let quantized = index.as_quantized().ok_or("expected quantized fixture")?;
            assert_eq!(
                quantized.get(id).as_deref(),
                Some([1.0_f32, 0.0].as_slice())
            );
        }
        Ok(())
    }
}
