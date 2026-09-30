//! Exact owner import into startup's unpublished catalog and physical stores.
//! Retained Arcs distinguish a dropped graph from a replacement with its name.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{EpochId, GraphPath, IndexId, PropertyKey, TransactionId};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::{
    IndexRegistrationObservation, IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey,
    LpgCommitWorkspace, LpgStore, StoreCommitInput, StoreIndexEdits, with_prepared_lpg_commit,
};

use super::LpgRecoveryTarget;
use crate::catalog::{
    Catalog, CatalogWorkspace, IndexConfiguration, IndexOwnerChange, IndexOwnerImage,
};

pub(super) struct OwnerReplay {
    targets: HashMap<IndexId, Arc<LpgStore>>,
}

#[cfg(all(test, feature = "vector-index"))]
mod vector_tests;

fn invalid(reason: impl std::fmt::Display) -> Error {
    Error::Serialization(format!("invalid WAL index owners: {reason}"))
}

fn resolve(
    root: &Arc<LpgStore>,
    default: &LpgRecoveryTarget,
    path: &GraphPath,
) -> Result<Arc<LpgStore>> {
    if path.components().is_empty() {
        return Ok(match default {
            LpgRecoveryTarget::Flat(store) => Arc::clone(store),
            #[cfg(feature = "compact-store")]
            LpgRecoveryTarget::Layered(store) => store.overlay_store(),
        });
    }
    let mut target = Arc::clone(root);
    for component in path.components() {
        target = target
            .graph(component)
            .ok_or_else(|| invalid(format!("missing graph {path:?}")))?;
    }
    Ok(target)
}

fn registry_key(image: &IndexOwnerImage) -> Result<IndexRegistryKey> {
    match &image.configuration {
        IndexConfiguration::Property | IndexConfiguration::BTree => Ok(IndexRegistryKey::Property(
            PropertyKey::from(image.property.as_str()),
        )),
        #[cfg(feature = "text-index")]
        IndexConfiguration::Text { .. } => Ok(IndexRegistryKey::Text {
            label: image.label.clone(),
            property: image.property.clone(),
        }),
        #[cfg(feature = "vector-index")]
        IndexConfiguration::Vector { .. } => Ok(IndexRegistryKey::Vector {
            label: image.label.clone(),
            property: image.property.clone(),
        }),
    }
}

fn observe(target: &LpgStore, key: &IndexRegistryKey) -> Option<IndexRegistrationObservation> {
    match key {
        IndexRegistryKey::Property(property) => target.observe_property_index(property.as_str()),
        #[cfg(feature = "text-index")]
        IndexRegistryKey::Text { label, property } => target.observe_text_index(label, property),
        #[cfg(feature = "vector-index")]
        IndexRegistryKey::Vector { label, property } => {
            target.observe_vector_index(label, property)
        }
    }
}

impl OwnerReplay {
    pub(super) fn new(
        root: &Arc<LpgStore>,
        default: &LpgRecoveryTarget,
        catalog: &Catalog,
    ) -> Result<Self> {
        let mut targets = HashMap::new();
        for owner in catalog.all_indexes() {
            targets.insert(owner.id, resolve(root, default, owner.key.graph())?);
        }
        Ok(Self { targets })
    }

    pub(super) fn apply(
        &mut self,
        root: &Arc<LpgStore>,
        default: &LpgRecoveryTarget,
        catalog: &Catalog,
        payload: &[u8],
        transaction_id: TransactionId,
        commit_epoch: EpochId,
    ) -> Result<()> {
        use super::index_commit_wire::IndexCommitBatch;
        let batch = IndexCommitBatch::decode(payload)?;
        if batch.commit != commit_epoch {
            return Err(invalid(
                "postimage epoch differs from its committed transaction",
            ));
        }
        struct Final {
            target: Arc<LpgStore>,
            owner: Option<IndexOwnerImage>,
        }
        let mut final_keys: HashMap<(usize, IndexRegistryKey), Final> = HashMap::new();
        for change in &batch.owners.changes {
            let (image, create, dropping) = match change {
                IndexOwnerChange::Create(image) => (image, true, false),
                IndexOwnerChange::Drop(image) => (image, false, true),
                IndexOwnerChange::Rebuild(image) => (image, false, false),
            };
            let local_key = registry_key(image)?;
            let target = if create {
                let target = resolve(root, default, &image.graph)?;
                if self.targets.insert(image.id, Arc::clone(&target)).is_some() {
                    return Err(invalid("duplicate owner identity"));
                }
                target
            } else {
                let target = self
                    .targets
                    .get(&image.id)
                    .cloned()
                    .ok_or_else(|| invalid("missing exact owner target"))?;
                if dropping {
                    self.targets.remove(&image.id);
                } else if !Arc::ptr_eq(&target, &resolve(root, default, &image.graph)?) {
                    return Err(invalid("rebuild targets a retired graph incarnation"));
                }
                target
            };
            let key = (Arc::as_ptr(&target).addr(), local_key);
            let present = final_keys.get(&key).map_or_else(
                || observe(&target, &key.1).is_some(),
                |entry| entry.owner.is_some(),
            );
            if present == create {
                return Err(invalid("owner change disagrees with physical registration"));
            }
            final_keys.insert(
                key,
                Final {
                    target,
                    owner: (!dropping).then(|| image.clone()),
                },
            );
        }
        let mut text_images = HashMap::new();
        for image in batch.text {
            let created = batch.owners.changes.iter().find_map(|change| match change {
                IndexOwnerChange::Create(owner) if owner.id == image.owner.id => Some(owner),
                _ => None,
            });
            if batch.owners.changes.iter().any(|change| matches!(change, IndexOwnerChange::Drop(owner) if owner.id == image.owner.id)) {
                return Err(invalid("postimage for a dropped owner"));
            }
            let expected = if let Some(created) = created {
                created.clone()
            } else {
                IndexOwnerImage::capture(catalog.read().view(), image.owner.id).map_err(invalid)?
            };
            if expected != image.owner || created.is_some() != image.birth {
                return Err(invalid("Text postimage disagrees with exact owner/birth"));
            }
            let target = self
                .targets
                .get(&image.owner.id)
                .cloned()
                .ok_or_else(|| invalid("Text postimage lacks exact owner target"))?;
            if !Arc::ptr_eq(&target, &resolve(root, default, &image.owner.graph)?) {
                return Err(invalid("Text postimage targets a retired graph"));
            }
            let key = (Arc::as_ptr(&target).addr(), registry_key(&image.owner)?);
            final_keys.entry(key).or_insert_with(|| Final {
                target,
                owner: Some(image.owner.clone()),
            });
            text_images.insert(image.owner.id, image);
        }
        let mut vector_images = HashMap::new();
        for image in batch.vectors {
            let created = batch.owners.changes.iter().find_map(|change| match change {
                IndexOwnerChange::Create(owner) if owner.id == image.owner.id => Some(owner),
                _ => None,
            });
            let rebuilding = batch.owners.changes.iter().any(|change| matches!(change, IndexOwnerChange::Rebuild(owner) if owner.id == image.owner.id));
            if batch.owners.changes.iter().any(|change| matches!(change, IndexOwnerChange::Drop(owner) if owner.id == image.owner.id)) {
                return Err(invalid("Vector postimage for dropped owner"));
            }
            let expected = if let Some(created) = created {
                created.clone()
            } else {
                IndexOwnerImage::capture(catalog.read().view(), image.owner.id).map_err(invalid)?
            };
            if expected != image.owner || image.complete != (created.is_some() || rebuilding) {
                return Err(invalid(
                    "Vector postimage disagrees with exact owner/replacement",
                ));
            }
            let target = self
                .targets
                .get(&image.owner.id)
                .cloned()
                .ok_or_else(|| invalid("Vector postimage lacks exact owner target"))?;
            if !Arc::ptr_eq(&target, &resolve(root, default, &image.owner.graph)?) {
                return Err(invalid("Vector postimage targets retired graph"));
            }
            let key = (Arc::as_ptr(&target).addr(), registry_key(&image.owner)?);
            final_keys.entry(key).or_insert_with(|| Final {
                target,
                owner: Some(image.owner.clone()),
            });
            vector_images.insert(image.owner.id, image);
        }
        #[cfg(feature = "vector-index")]
        let mut vector_changes = Vec::new();
        let targets: Vec<_> = final_keys.into_iter().collect();
        let mut edits_by_store: HashMap<usize, StoreIndexEdits<'_>> = HashMap::new();
        for ((identity, key), entry) in &targets {
            let target = &entry.target;
            let expected = observe(target, key);
            edits_by_store
                .entry(*identity)
                .or_insert_with(|| StoreIndexEdits {
                    store: target.as_ref(),
                    edits: Vec::new(),
                });
            let edit = if let Some(owner) = &entry.owner {
                #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
                let _ = owner;
                match key {
                    IndexRegistryKey::Property(property) => {
                        let contents = IndexRegistryContents::PropertyHistory(
                            target.property_index_image(property.as_str())?,
                        );
                        match expected {
                            Some(expected) => IndexRegistryEdit::Replace { expected, contents },
                            None => IndexRegistryEdit::Create {
                                key: key.clone(),
                                contents,
                            },
                        }
                    }
                    #[cfg(feature = "text-index")]
                    IndexRegistryKey::Text { .. } => {
                        use grafeo_core::graph::lpg::IndexRegistryMaintenance;
                        let image = text_images
                            .remove(&owner.id)
                            .ok_or_else(|| invalid("Text create/rebuild lacks exact postimage"))?;
                        if image.birth {
                            let index = grafeo_core::index::text::InvertedIndex::decode_wal_birth(
                                &image.payload,
                                batch.commit,
                            )?;
                            let IndexConfiguration::Text {
                                config,
                                min_token_length,
                            } = &owner.configuration
                            else {
                                return Err(invalid("Text birth has non-Text configuration"));
                            };
                            let actual = index.config();
                            if actual.k1.to_bits() != config.k1.to_bits()
                                || actual.b.to_bits() != config.b.to_bits()
                                || !index.has_simple_tokenizer(*min_token_length)
                            {
                                return Err(invalid("Text birth configuration differs from owner"));
                            }
                            let contents = IndexRegistryContents::Text(index);
                            match expected {
                                Some(expected) => IndexRegistryEdit::Replace { expected, contents },
                                None => IndexRegistryEdit::Create {
                                    key: key.clone(),
                                    contents,
                                },
                            }
                        } else {
                            IndexRegistryEdit::Maintain {
                                expected: expected.ok_or_else(|| {
                                    invalid("Text survivor registration is absent")
                                })?,
                                changes: IndexRegistryMaintenance::TextRecorded {
                                    payload: image.payload,
                                    frontier: batch.frontier,
                                    commit_epoch: batch.commit,
                                    transaction_id,
                                },
                            }
                        }
                    }
                    #[cfg(feature = "vector-index")]
                    IndexRegistryKey::Vector { label, property } => {
                        let image = vector_images.remove(&owner.id).ok_or_else(|| {
                            invalid("Vector create/rebuild lacks exact postimage")
                        })?;
                        if image.complete {
                            let IndexConfiguration::Vector {
                                config,
                                quantization,
                            } = &owner.configuration
                            else {
                                return Err(invalid("Vector image has non-Vector owner"));
                            };
                            let source: &dyn grafeo_core::graph::GraphStore = match default {
                                LpgRecoveryTarget::Flat(_) => target.as_ref(),
                                #[cfg(feature = "compact-store")]
                                LpgRecoveryTarget::Layered(layered)
                                    if Arc::ptr_eq(target, &layered.overlay_store()) =>
                                {
                                    layered.as_ref()
                                }
                                #[cfg(feature = "compact-store")]
                                LpgRecoveryTarget::Layered(_) => target.as_ref(),
                            };
                            let rows = source.prepare_index_node_rows(batch.commit, None)?;
                            let population = rows
                                .iter()
                                .filter(|node| node.has_label(label))
                                .filter_map(|node| {
                                    node.get_property(property)
                                        .and_then(grafeo_core::index::vector::value_to_vector)
                                        .map(|vector| (node.id, vector))
                                });
                            let index =
                                grafeo_core::index::vector::VectorIndexKind::decode_wal_birth(
                                    &image.payload,
                                    config,
                                    *quantization,
                                    population,
                                )?;
                            let contents = IndexRegistryContents::Vector(index);
                            match expected {
                                Some(expected) => IndexRegistryEdit::Replace { expected, contents },
                                None => IndexRegistryEdit::Create {
                                    key: key.clone(),
                                    contents,
                                },
                            }
                        } else {
                            let expected = expected
                                .ok_or_else(|| invalid("Vector survivor registration is absent"))?;
                            let view = target
                                .get_vector_index(label, property)
                                .ok_or_else(|| invalid("Vector survivor view is absent"))?;
                            vector_changes.push((
                                target.as_ref(),
                                view,
                                expected,
                                PropertyKey::from(property.as_str()),
                                image.payload,
                            ));
                            continue;
                        }
                    }
                }
            } else {
                let Some(expected) = expected else {
                    continue;
                };
                IndexRegistryEdit::Drop { expected }
            };
            edits_by_store
                .entry(*identity)
                .or_insert_with(|| StoreIndexEdits {
                    store: target.as_ref(),
                    edits: Vec::new(),
                })
                .edits
                .push(edit);
        }
        #[cfg(not(feature = "text-index"))]
        let _ = transaction_id;
        if !vector_images.is_empty() {
            return Err(invalid("unconsumed Vector postimages"));
        }
        if !text_images.is_empty() {
            return Err(invalid("unconsumed Text postimages"));
        }
        // Outer payloads predate every borrowed publication guard.
        let stores = edits_by_store
            .into_values()
            .map(|entry| {
                let source: &dyn grafeo_core::graph::GraphStoreMut = match default {
                    LpgRecoveryTarget::Flat(_) => entry.store,
                    #[cfg(feature = "compact-store")]
                    LpgRecoveryTarget::Layered(layered)
                        if std::ptr::eq(entry.store, layered.overlay_store().as_ref()) =>
                    {
                        layered.as_ref()
                    }
                    #[cfg(feature = "compact-store")]
                    LpgRecoveryTarget::Layered(_) => entry.store,
                };
                StoreCommitInput {
                    store: entry.store,
                    source,
                    graph: None,
                    publish_data: false,
                    edits: entry.edits,
                }
            })
            .collect();
        #[cfg(feature = "vector-index")]
        let vectors = vector_changes
            .iter_mut()
            .map(|(store, view, expected, property, payload)| {
                grafeo_core::graph::lpg::VectorCommitInput {
                    store,
                    view,
                    expected,
                    property: property.clone(),
                    changes: grafeo_core::graph::lpg::VectorCommitChanges::Recorded(
                        std::mem::take(payload),
                    ),
                    routing: Default::default(),
                }
            })
            .collect();
        let mut registry = LpgCommitWorkspace::new(
            stores,
            #[cfg(feature = "vector-index")]
            vectors,
            transaction_id,
            batch.frontier,
            batch.commit,
        );
        let authority = grafeo_core::graph::write_permit::WriteAuthority::new();
        let mut logical = CatalogWorkspace::new();
        let edit = if batch.owners.changes.is_empty() {
            if catalog.index_allocator_high_water() != batch.owners.expected_floor {
                return Err(invalid("survivor allocator floor mismatch"));
            }
            None
        } else {
            let edit = catalog.prepare_edit(&mut logical).map_err(invalid)?;
            batch.owners.apply(edit.candidate()).map_err(invalid)?;
            Some(edit)
        };
        let ready = edit.map(|edit| edit.finish());
        grafeo_core::graph::write_permit::with_authority(&authority, || {
            with_prepared_lpg_commit(&mut registry, &authority, |released| {
                let prepared = released.rebind().map_err(|error| error.into_error())?;
                let physical = prepared.install();
                if let Some(ready) = ready {
                    ready.install().finish();
                }
                drop(physical);
                Ok(())
            })
        })
    }

    pub(super) fn validate_live(
        &self,
        root: &Arc<LpgStore>,
        default: &LpgRecoveryTarget,
        catalog: &Catalog,
    ) -> Result<()> {
        for owner in catalog.all_indexes() {
            let target = self
                .targets
                .get(&owner.id)
                .ok_or_else(|| invalid("owner lacks replay witness"))?;
            if !Arc::ptr_eq(target, &resolve(root, default, owner.key.graph())?) {
                return Err(invalid(
                    "surviving owner refers to a retired graph incarnation",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::IndexOwnerBatch;

    fn image(id: u32, graph: GraphPath) -> IndexOwnerImage {
        IndexOwnerImage {
            id: IndexId::new(id),
            name: format!("owner{id}"),
            graph,
            label: "Person".into(),
            property: "code".into(),
            configuration: IndexConfiguration::Property,
        }
    }

    fn encode(owners: IndexOwnerBatch) -> Vec<u8> {
        super::super::index_commit_wire::IndexCommitBatch {
            frontier: EpochId::INITIAL,
            commit: EpochId::new(1),
            owners,
            text: Vec::new(),
            vectors: Vec::new(),
        }
        .encode()
        .unwrap()
    }

    fn create(image: IndexOwnerImage) -> Vec<u8> {
        encode(IndexOwnerBatch {
            expected_floor: image.id.as_u32(),
            next_floor: image.id.as_u32() + 1,
            changes: vec![IndexOwnerChange::Create(image)],
        })
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn owner_replay_rejects_birth_from_another_epoch_before_publication() -> Result<()> {
        use super::super::index_commit_wire::{IndexCommitBatch, TextPostimage};
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        for epoch in [EpochId::INITIAL, EpochId::new(2)] {
            let root = Arc::new(LpgStore::new()?);
            let default = LpgRecoveryTarget::Flat(Arc::clone(&root));
            let catalog = Catalog::new();
            let owner = IndexOwnerImage {
                configuration: IndexConfiguration::Text {
                    config: BM25Config::default(),
                    min_token_length: 2,
                },
                ..image(0, GraphPath::root())
            };
            let mut index = InvertedIndex::with_simple_tokenizer(BM25Config::default(), 2);
            index.insert_versioned(
                grafeo_common::types::NodeId::new(1),
                "wrong epoch",
                epoch,
                None,
            );
            let bytes = IndexCommitBatch {
                frontier: EpochId::INITIAL,
                commit: EpochId::new(1),
                owners: IndexOwnerBatch {
                    expected_floor: 0,
                    next_floor: 1,
                    changes: vec![IndexOwnerChange::Create(owner.clone())],
                },
                text: vec![TextPostimage {
                    owner,
                    birth: true,
                    payload: index.encode_wal_birth()?,
                }],
                vectors: Vec::new(),
            }
            .encode()?;
            let mut replay = OwnerReplay::new(&root, &default, &catalog)?;
            assert!(
                replay
                    .apply(
                        &root,
                        &default,
                        &catalog,
                        &bytes,
                        TransactionId::new(2),
                        EpochId::new(1)
                    )
                    .is_err()
            );
            assert!(catalog.all_indexes().is_empty());
            assert_eq!(catalog.index_allocator_high_water(), 0);
            assert!(root.text_index_entries().is_empty());
        }
        Ok(())
    }

    #[test]
    fn owner_replay_rejects_stale_floor_without_publishing_physical_index() {
        let root = Arc::new(LpgStore::new().unwrap());
        let default = LpgRecoveryTarget::Flat(Arc::clone(&root));
        let catalog = Catalog::new();
        let mut replay = OwnerReplay::new(&root, &default, &catalog).unwrap();
        assert!(
            replay
                .apply(
                    &root,
                    &default,
                    &catalog,
                    &create(image(5, GraphPath::root())),
                    TransactionId::new(2),
                    EpochId::new(1)
                )
                .is_err()
        );
        assert!(!root.has_property_index("code"));
        assert_eq!(catalog.index_allocator_high_water(), 0);
    }

    #[test]
    fn owner_replay_rejects_missing_target_and_unowned_physical_registration() {
        let root = Arc::new(LpgStore::new().unwrap());
        let default = LpgRecoveryTarget::Flat(Arc::clone(&root));
        let catalog = Catalog::new();
        let mut replay = OwnerReplay::new(&root, &default, &catalog).unwrap();
        assert!(
            replay
                .apply(
                    &root,
                    &default,
                    &catalog,
                    &create(image(0, GraphPath::from_components(&["absent"]).unwrap())),
                    TransactionId::new(2),
                    EpochId::new(1)
                )
                .is_err()
        );
        assert!(root.graph("absent").is_none());
        root.create_property_index("code");
        assert!(
            replay
                .apply(
                    &root,
                    &default,
                    &catalog,
                    &create(image(0, GraphPath::root())),
                    TransactionId::new(2),
                    EpochId::new(1)
                )
                .is_err()
        );
        assert!(catalog.all_indexes().is_empty());
    }

    #[test]
    fn owner_replay_retains_exact_graph_for_drop_and_rejects_retired_rebuild() {
        let root = Arc::new(LpgStore::new().unwrap());
        root.create_graph("same").unwrap();
        let default = LpgRecoveryTarget::Flat(Arc::clone(&root));
        let catalog = Catalog::new();
        let owner = image(0, GraphPath::from_components(&["same"]).unwrap());
        let mut replay = OwnerReplay::new(&root, &default, &catalog).unwrap();
        replay
            .apply(
                &root,
                &default,
                &catalog,
                &create(owner.clone()),
                TransactionId::new(2),
                EpochId::new(1),
            )
            .unwrap();
        let retired = root.graph("same").unwrap();
        root.drop_graph("same");
        root.create_graph("same").unwrap();
        let replacement = root.graph("same").unwrap();
        replacement.create_property_index("code");
        let mut batch = IndexOwnerBatch {
            expected_floor: 1,
            next_floor: 1,
            changes: vec![IndexOwnerChange::Rebuild(owner.clone())],
        };
        assert!(
            replay
                .apply(
                    &root,
                    &default,
                    &catalog,
                    &encode(batch.clone()),
                    TransactionId::new(2),
                    EpochId::new(1)
                )
                .is_err()
        );
        assert!(replay.validate_live(&root, &default, &catalog).is_err());
        batch.changes = vec![IndexOwnerChange::Drop(owner)];
        replay
            .apply(
                &root,
                &default,
                &catalog,
                &encode(batch.clone()),
                TransactionId::new(2),
                EpochId::new(1),
            )
            .unwrap();
        assert!(!retired.has_property_index("code"));
        assert!(replacement.has_property_index("code"));
        replay.validate_live(&root, &default, &catalog).unwrap();
    }
}
