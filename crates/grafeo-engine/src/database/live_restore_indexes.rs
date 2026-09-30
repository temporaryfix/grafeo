//! Detached exact index contents for the existing-root live restore driver.

use super::Snapshot;
use super::snapshot_indexes::stage_snapshot_indexes;
use crate::catalog::{Catalog, IndexConfiguration};
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use grafeo_common::storage::Section;
use grafeo_common::types::{GraphPath, PropertyKey};
use grafeo_common::utils::error::{Error, Result};
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use grafeo_core::graph::lpg::decode_index_key;
use grafeo_core::graph::lpg::{
    IndexRegistrationObservation, IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey,
    IndexRegistryWorkspace, LpgStore, PhysicalIndexKey, StoreIndexEdits,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The anchors must outlive the registry workspace independently of the data
/// workspace: its candidate is mutably borrowed during the final backing swap.
/// Root rows come from `candidate`; descendant targets are the incoming stores.
pub(super) fn stage_live_restore_indexes<'store>(
    snapshot: &Snapshot,
    catalog: &Catalog,
    candidate: &LpgStore,
    targets: &'store [(GraphPath, Arc<LpgStore>)],
) -> Result<IndexRegistryWorkspace<'store>> {
    stage_snapshot_indexes(snapshot, catalog)?;
    if targets.first().map(|(path, _)| path) != Some(&GraphPath::root())
        || targets.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(invalid(
            "replacement anchors must be canonical and root-inclusive",
        ));
    }
    let target_map: BTreeMap<_, _> = targets
        .iter()
        .map(|(path, store)| (path, store.as_ref()))
        .collect();
    let mut incoming = BTreeMap::new();
    #[cfg(feature = "text-index")]
    let mut text = Vec::new();
    #[cfg(feature = "vector-index")]
    let mut vector = Vec::new();
    #[cfg(feature = "vector-index")]
    let quantized_vector_keys = if snapshot.vector_indexes.is_empty() {
        Vec::new()
    } else {
        grafeo_core::index::vector::VectorStoreSection::payload_quantized_keys(
            &snapshot.vector_indexes,
        )?
    };
    for owner in catalog.all_indexes() {
        let target = target_map
            .get(owner.key.graph())
            .copied()
            .ok_or_else(|| invalid("index owner has no replacement graph"))?;
        match owner.configuration {
            IndexConfiguration::Property | IndexConfiguration::BTree => {
                let source = if owner.key.graph() == &GraphPath::root() {
                    candidate
                } else {
                    target
                };
                let image = source.property_index_image(owner.key.property_name())?;
                incoming.insert(owner.key, IndexRegistryContents::PropertyHistory(image));
            }
            #[cfg(feature = "text-index")]
            IndexConfiguration::Text {
                config,
                min_token_length,
            } => text.push((
                owner.key,
                Arc::new(parking_lot::RwLock::new(
                    grafeo_core::index::text::InvertedIndex::with_simple_tokenizer(
                        config,
                        min_token_length,
                    ),
                )),
            )),
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
                        if quantized_vector_keys.binary_search(&owner.key).is_err() =>
                    {
                        VectorIndexKind::Hnsw(HnswIndex::new(config))
                    }
                    kind => VectorIndexKind::Quantized(QuantizedHnswIndex::new(config, kind)),
                };
                vector.push((owner.key, Arc::new(index)));
            }
        }
    }
    // Decode into fresh standalone objects, never registered candidate handles.
    // The current decoders retain complete history, retention floors and RNG.
    #[cfg(feature = "text-index")]
    if !text.is_empty() {
        text.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        {
            let mut section =
                grafeo_core::index::text::TextIndexSection::for_unpublished_recovery(text.clone());
            section.deserialize(&snapshot.text_indexes)?;
        }
        for (key, index) in text {
            let index = Arc::try_unwrap(index)
                .map_err(|_| invalid("detached Text restore retained an unexpected alias"))?
                .into_inner();
            incoming.insert(key, IndexRegistryContents::Text(index));
        }
    }
    #[cfg(feature = "vector-index")]
    if !vector.is_empty() {
        vector.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        {
            let mut section =
                grafeo_core::index::vector::VectorStoreSection::for_unpublished_recovery(
                    vector.clone(),
                );
            section.deserialize(&snapshot.vector_indexes)?;
        }
        for (key, index) in vector {
            let index = Arc::try_unwrap(index)
                .map_err(|_| invalid("detached Vector restore retained an unexpected alias"))?;
            incoming.insert(key, IndexRegistryContents::Vector(index));
        }
    }

    let mut stores = Vec::with_capacity(targets.len());
    for (path, store) in targets {
        #[cfg(test)]
        crate::database::catalog_section::current_index_preparation_test_point()?;
        let mut current = observe_current_indexes(path, store)?;
        let keys: Vec<_> = incoming
            .keys()
            .filter(|key| key.graph() == path)
            .cloned()
            .collect();
        let mut edits = Vec::with_capacity(keys.len() + current.len());
        for key in keys {
            let contents = incoming
                .remove(&key)
                .ok_or_else(|| invalid("prepared replacement index is absent"))?;
            edits.push(match current.remove(&key) {
                Some(expected) => IndexRegistryEdit::Replace { expected, contents },
                None => IndexRegistryEdit::Create {
                    key: registry_key(&key)?,
                    contents,
                },
            });
        }
        edits.extend(
            current
                .into_values()
                .map(|expected| IndexRegistryEdit::Drop { expected }),
        );
        // Empty roots still enroll authority and hold every registry guard.
        stores.push(StoreIndexEdits { store, edits });
    }
    if !incoming.is_empty() {
        return Err(invalid("prepared indexes have no replacement target"));
    }
    Ok(IndexRegistryWorkspace::new(stores))
}

fn observe_current_indexes(
    path: &GraphPath,
    store: &LpgStore,
) -> Result<BTreeMap<PhysicalIndexKey, IndexRegistrationObservation>> {
    let mut current = BTreeMap::new();
    for property in store.property_index_keys() {
        let expected = store
            .observe_property_index(&property)
            .ok_or_else(|| invalid("replacement Property registration changed"))?;
        current.insert(PhysicalIndexKey::property(path.clone(), property), expected);
    }
    #[cfg(feature = "text-index")]
    for (key, _) in store.text_index_entries() {
        let (label, property) = decode_index_key(&key)
            .ok_or_else(|| invalid("replacement Text registry key is malformed"))?;
        let expected = store
            .observe_text_index(label, property)
            .ok_or_else(|| invalid("replacement Text registration changed"))?;
        current.insert(
            PhysicalIndexKey::text(path.clone(), label, property),
            expected,
        );
    }
    #[cfg(feature = "vector-index")]
    for (key, _) in store.vector_index_entries() {
        let (label, property) = decode_index_key(&key)
            .ok_or_else(|| invalid("replacement Vector registry key is malformed"))?;
        let expected = store
            .observe_vector_index(label, property)
            .ok_or_else(|| invalid("replacement Vector registration changed"))?;
        current.insert(
            PhysicalIndexKey::vector(path.clone(), label, property),
            expected,
        );
    }
    Ok(current)
}

fn registry_key(key: &PhysicalIndexKey) -> Result<IndexRegistryKey> {
    use grafeo_core::graph::lpg::PhysicalIndexFamily;
    match key.family() {
        PhysicalIndexFamily::Property => Ok(IndexRegistryKey::Property(PropertyKey::new(
            key.property_name(),
        ))),
        #[cfg(feature = "text-index")]
        PhysicalIndexFamily::Text => Ok(IndexRegistryKey::Text {
            label: key
                .label()
                .ok_or_else(|| invalid("Text owner has no label"))?
                .into(),
            property: key.property_name().into(),
        }),
        #[cfg(feature = "vector-index")]
        PhysicalIndexFamily::Vector => Ok(IndexRegistryKey::Vector {
            label: key
                .label()
                .ok_or_else(|| invalid("Vector owner has no label"))?
                .into(),
            property: key.property_name().into(),
        }),
        #[cfg(not(all(feature = "text-index", feature = "vector-index")))]
        _ => Err(invalid("replacement index family is not enabled")),
    }
}

fn invalid(reason: &str) -> Error {
    Error::Serialization(format!("live index replacement: {reason}"))
}
