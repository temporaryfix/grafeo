//! Detached property-index rebuild input.

#[cfg(feature = "compact-store")]
use std::sync::Arc;

use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;

#[cfg(feature = "compact-store")]
use crate::graph::traits::GraphStore;

use super::super::LpgStore;
use super::PropertyIndexImage;

impl LpgStore {
    /// Exports the committed current rows and complete property history used
    /// to build a property index.  The read pin keeps the local representation
    /// and its compact predecessor stable for the duration of the export.
    ///
    /// # Errors
    /// Returns an error if a source property log is not in committed epoch order.
    pub fn property_index_image(&self, property: &str) -> Result<PropertyIndexImage> {
        let _read = self.pin_read();
        let key = PropertyKey::new(property);
        let epoch = self.current_epoch();

        let local_histories = self.node_properties.column_history(&key);
        self.work_counters
            .record_property_index_rebuild_rows(local_histories.len());
        let mut histories: FxHashMap<NodeId, Vec<(EpochId, Value)>> = FxHashMap::default();
        for (id, history) in local_histories {
            histories.insert(id, committed_history(id, history, epoch)?);
        }

        #[cfg(feature = "compact-store")]
        if let Some(base) = self.property_index_compact_base() {
            let mut base_ids = base.node_ids();
            base_ids.extend(base.temporal_node_ids());
            base_ids.sort_unstable();
            base_ids.dedup();
            self.work_counters
                .record_property_index_rebuild_rows(base_ids.len());
            for id in base_ids {
                let base_history = base
                    .node_property_history(id)
                    .into_iter()
                    .find_map(|(base_key, history)| (base_key == key).then_some(history))
                    .unwrap_or_default();
                if base_history.is_empty() {
                    continue;
                }
                let base_history = committed_history(id, base_history, epoch)?;
                if let Some(local) = histories.get_mut(&id) {
                    merge_histories(local, base_history);
                } else {
                    histories.insert(id, base_history);
                }
            }
        }

        let mut history: Vec<_> = histories.into_iter().collect();
        history.sort_unstable_by_key(|(id, _)| *id);

        let mut current = Vec::new();
        for (id, events) in &history {
            let visible = if self.contains_node_identity(*id) {
                self.get_node_at_epoch(*id, epoch).is_some()
            } else {
                #[cfg(feature = "compact-store")]
                {
                    self.property_index_compact_base()
                        .and_then(|base| base.get_node_at_epoch(*id, epoch))
                        .is_some()
                }
                #[cfg(not(feature = "compact-store"))]
                {
                    false
                }
            };
            if !visible {
                continue;
            }
            if let Some((_, value)) = events.last()
                && !value.is_null()
            {
                current.push((*id, value.clone()));
            }
        }

        let floor = self.retained_history_floor();
        #[cfg(feature = "compact-store")]
        let floor = self.property_index_compact_base().map_or(floor, |base| {
            floor.max(base.property_history_floor().unwrap_or(epoch))
        });

        Ok(PropertyIndexImage {
            current,
            history,
            floor,
        })
    }

    #[cfg(feature = "compact-store")]
    pub(super) fn property_index_compact_base(
        &self,
    ) -> Option<Arc<crate::graph::compact::CompactStore>> {
        #[cfg(feature = "vector-index")]
        {
            self.compact_base.get().cloned()
        }
        #[cfg(not(feature = "vector-index"))]
        {
            self.compact_base.get().and_then(std::sync::Weak::upgrade)
        }
    }
}

impl PropertyIndexImage {
    /// Applies committed Layered base-node tombstones to this detached rebuild
    /// image. Legacy compact columns contain the last property value but not
    /// the overlay's delete epoch, so the persisted tombstone list must close
    /// that value before the image is turned into ordered index intervals.
    ///
    /// # Errors
    /// Returns an error for a pending or conflicting tombstone epoch, or for
    /// retained property history that is unordered or extends past its delete.
    pub fn close_deleted_nodes(&mut self, deletions: &[(NodeId, EpochId)]) -> Result<()> {
        let mut delete_epochs: FxHashMap<NodeId, EpochId> = FxHashMap::default();
        delete_epochs
            .try_reserve(deletions.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for &(id, delete_epoch) in deletions {
            if delete_epoch == EpochId::PENDING {
                return Err(Error::InvalidValue(
                    "property index delete normalization contains PENDING epoch".into(),
                ));
            }
            if let Some(previous) = delete_epochs.insert(id, delete_epoch)
                && previous != delete_epoch
            {
                return Err(Error::InvalidValue(
                    "property index delete normalization has conflicting epochs".into(),
                ));
            }
        }
        for (id, events) in &mut self.history {
            let Some(&delete_epoch) = delete_epochs.get(id) else {
                continue;
            };
            if events.windows(2).any(|pair| pair[0].0 > pair[1].0) {
                return Err(Error::InvalidValue(
                    "property index history is not ordered before delete normalization".into(),
                ));
            }
            if let Some((last_epoch, last_value)) = events.last() {
                if *last_epoch > delete_epoch {
                    return Err(Error::InvalidValue(
                        "property index delete precedes a retained property event".into(),
                    ));
                }
                if *last_epoch < delete_epoch || !last_value.is_null() {
                    events.push((delete_epoch, Value::Null));
                }
            }
        }
        self.current
            .retain(|(id, _)| !delete_epochs.contains_key(id));
        Ok(())
    }
}

fn committed_history(
    id: NodeId,
    history: Vec<(EpochId, Value)>,
    frontier: EpochId,
) -> Result<Vec<(EpochId, Value)>> {
    let mut previous = None;
    let mut pending = false;
    for (epoch, _) in &history {
        if *epoch == EpochId::PENDING {
            pending = true;
            continue;
        }
        if pending || previous.is_some_and(|prior| *epoch < prior) {
            return Err(Error::InvalidValue(format!(
                "property index history for node {id:?} is not ordered"
            )));
        }
        previous = Some(*epoch);
    }
    Ok(history
        .into_iter()
        .filter(|(epoch, _)| *epoch != EpochId::PENDING && *epoch <= frontier)
        .collect())
}

#[cfg(feature = "compact-store")]
fn merge_histories(overlay: &mut Vec<(EpochId, Value)>, mut base: Vec<(EpochId, Value)>) {
    base.retain(|(epoch, _)| {
        !overlay
            .iter()
            .any(|(overlay_epoch, _)| overlay_epoch == epoch)
    });
    base.append(overlay);
    base.sort_by_key(|(epoch, _)| *epoch);
    *overlay = base;
}

#[cfg(test)]
mod tests {
    use super::super::PropertyIndexRows;
    use super::*;
    use crate::graph::{PropertyIndexPredicate, PropertyIndexRequest};

    fn history_store() -> (LpgStore, NodeId) {
        let store = LpgStore::new().unwrap();
        store.set_epoch(EpochId::new(1));
        let id = store.create_node(&["Item"]);
        for (epoch, value) in [(1, 10), (2, 20), (3, 10)] {
            store.set_node_property_at_epoch(id, "score", Value::Int64(value), EpochId::new(epoch));
        }
        store.set_epoch(EpochId::new(3));
        (store, id)
    }

    fn sparse_history_store(node_count: usize, leave_at_epoch_two: usize) -> LpgStore {
        let store = LpgStore::new().unwrap();
        let epoch_one = EpochId::new(1);
        let epoch_two = EpochId::new(2);
        store.set_epoch(epoch_one);
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let node = store.create_node(&["Item"]);
            store.set_node_property_at_epoch(node, "score", Value::Int64(1), epoch_one);
            nodes.push(node);
        }
        store.create_property_index("score");

        store.set_epoch(epoch_two);
        let moved_at_epoch_two = node_count.saturating_sub(leave_at_epoch_two);
        for &node in nodes.iter().take(moved_at_epoch_two) {
            store.set_node_property_at_epoch(node, "score", Value::Int64(2), epoch_two);
        }
        if leave_at_epoch_two != 0 {
            let epoch_three = EpochId::new(3);
            store.set_epoch(epoch_three);
            for &node in nodes.iter().skip(moved_at_epoch_two) {
                store.set_node_property_at_epoch(node, "score", Value::Int64(2), epoch_three);
            }
        }
        // Preserve the old snapshot while removing no history needed by it.
        store.gc_versions(epoch_one);
        store
    }

    fn indexed_work(store: &LpgStore, epoch: EpochId) -> (Vec<NodeId>, crate::graph::WorkSnapshot) {
        let before = store.work_snapshot();
        let mut nodes = store
            .lookup_nodes_indexed(PropertyIndexRequest {
                property: "score",
                predicate: PropertyIndexPredicate::Equal(&Value::Int64(1)),
                epoch,
                transaction_id: None,
            })
            .unwrap()
            .unwrap();
        nodes.sort_unstable();
        (nodes, store.work_snapshot().since(before))
    }

    fn total_index_work(work: crate::graph::WorkSnapshot) -> u64 {
        work.property_index_route_keys
            .saturating_add(work.property_index_posting_ids)
            .saturating_add(work.property_index_posting_intervals)
    }

    #[test]
    fn property_image_knows_absence_before_first_write() {
        let store = LpgStore::new().unwrap();
        let id = store.create_node(&["Item"]);
        store.set_node_property_at_epoch(id, "late", Value::Int64(7), EpochId::new(5));
        store.set_epoch(EpochId::new(5));
        let image = store.property_index_image("late").unwrap();
        assert_eq!(image.floor, EpochId::INITIAL);
        let index = PropertyIndexRows::from_image(image).unwrap();
        assert!(
            index
                .history
                .read()
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(7)),
                    EpochId::new(4)
                )
                .unwrap()
                .0
                .is_empty()
        );
        assert_eq!(
            store.property_index_image("absent").unwrap().floor,
            EpochId::INITIAL
        );
    }

    #[test]
    fn property_gc_preserves_inclusive_floor_and_retires_closed_ordered_keys() {
        let (store, id) = history_store();
        store.create_property_index("score");
        store.gc_versions(EpochId::new(2));
        assert_eq!(store.retained_history_floor(), EpochId::new(2));
        assert_eq!(
            store.property_index_image("score").unwrap().floor,
            EpochId::new(2)
        );
        {
            let indexes = store.property_indexes.read();
            let history = indexes[&PropertyKey::new("score")].history.read();
            assert!(
                history
                    .candidates(
                        PropertyIndexPredicate::Equal(&Value::Int64(10)),
                        EpochId::new(1)
                    )
                    .is_err()
            );
            assert_eq!(
                history
                    .candidates(
                        PropertyIndexPredicate::Equal(&Value::Int64(20)),
                        EpochId::new(2)
                    )
                    .unwrap()
                    .0,
                vec![id]
            );
            assert!(
                history
                    .candidates(
                        PropertyIndexPredicate::Equal(&Value::Int64(10)),
                        EpochId::new(2)
                    )
                    .unwrap()
                    .0
                    .is_empty()
            );
            assert_eq!(history.ordered.len(), 2);
        }
        store.gc_versions(EpochId::new(3));
        let indexes = store.property_indexes.read();
        let history = indexes[&PropertyKey::new("score")].history.read();
        assert_eq!(history.floor, EpochId::new(3));
        assert_eq!(history.ordered.len(), 1);
        assert!(
            !history
                .rows
                .contains_key(&grafeo_common::types::HashableValue::new(Value::Int64(20)))
        );
        assert_eq!(
            history
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(10)),
                    EpochId::new(3)
                )
                .unwrap()
                .0,
            vec![id]
        );
    }

    #[test]
    fn property_gc_floor_is_monotonic_and_pending_gc_is_inert() {
        let (store, _) = history_store();
        store.gc_versions(EpochId::new(2));
        store.gc_versions(EpochId::new(1));
        store.gc_versions(EpochId::PENDING);
        assert_eq!(
            store.property_index_image("score").unwrap().floor,
            EpochId::new(2)
        );
        assert_eq!(
            store.property_index_image("absent").unwrap().floor,
            EpochId::new(2)
        );
        store.clear();
        assert_eq!(store.retained_history_floor(), EpochId::INITIAL);
    }

    #[test]
    fn indexed_lookup_empty_expired_bucket_has_size_independent_work() {
        let small = sparse_history_store(128, 0);
        let large = sparse_history_store(512, 0);
        let (small_nodes, small_work) = indexed_work(&small, EpochId::new(2));
        let (large_nodes, large_work) = indexed_work(&large, EpochId::new(2));

        assert!(small_nodes.is_empty());
        assert!(large_nodes.is_empty());
        assert_eq!(small_work.property_index_posting_ids, 0);
        assert_eq!(large_work.property_index_posting_ids, 0);
        let small_total = total_index_work(small_work);
        let large_total = total_index_work(large_work);
        assert!(small_total <= 64);
        assert!(large_total <= 64);
        assert!(large_total <= small_total.saturating_add(16));
    }

    #[test]
    fn indexed_lookup_retained_snapshot_visits_only_sparse_hits() {
        let small = sparse_history_store(128, 1);
        let large = sparse_history_store(512, 1);
        let (small_nodes, small_work) = indexed_work(&small, EpochId::new(2));
        let (large_nodes, large_work) = indexed_work(&large, EpochId::new(2));

        assert_eq!(small_nodes.len(), 1);
        assert_eq!(large_nodes.len(), 1);
        assert_eq!(small_work.property_index_posting_ids, 1);
        assert_eq!(large_work.property_index_posting_ids, 1);
        let small_total = total_index_work(small_work);
        let large_total = total_index_work(large_work);
        assert!(small_total <= 64);
        assert!(large_total <= 64);
        assert!(large_total <= small_total.saturating_add(16));
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn temporal_compaction_preserves_property_coverage_after_gc() {
        let (store, id) = history_store();
        store.gc_versions(EpochId::new(2));
        let layered =
            crate::graph::compact::layered::LayeredStore::from_native_temporal(Arc::new(store))
                .unwrap();
        assert_eq!(
            layered.base_store_arc().property_history_floor(),
            Some(EpochId::new(2))
        );
        let overlay = layered.overlay_store();
        assert_eq!(overlay.retained_history_floor(), EpochId::new(2));
        let image = overlay.property_index_image("score").unwrap();
        assert_eq!(image.floor, EpochId::new(2));
        let index = PropertyIndexRows::from_image(image).unwrap();
        assert_eq!(
            index
                .history
                .read()
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(20)),
                    EpochId::new(2)
                )
                .unwrap()
                .0,
            vec![id]
        );
        assert!(
            index
                .history
                .read()
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(10)),
                    EpochId::new(1)
                )
                .is_err()
        );
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn property_image_closes_legacy_compact_node_deleted_in_overlay() {
        let native = LpgStore::new().unwrap();
        native.set_epoch(EpochId::new(1));
        let deleted = native.create_node(&["Item"]);
        native.set_node_property_at_epoch(deleted, "score", Value::Int64(7), EpochId::new(1));
        let base = crate::graph::compact::from_graph_store_preserving_ids(&native).unwrap();
        let layered =
            crate::graph::compact::layered::LayeredStore::new(base, deleted.as_u64(), 0).unwrap();
        layered.seed_deleted_from_base_at_epochs([(deleted, EpochId::new(2))], std::iter::empty());
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(2));

        let mut image = overlay.property_index_image("score").unwrap();
        assert!(
            image.current.iter().any(|(id, _)| *id == deleted),
            "raw compact columns cannot observe the Layered tombstone"
        );
        image
            .close_deleted_nodes(&[(deleted, EpochId::new(2))])
            .unwrap();
        assert!(
            image.current.is_empty(),
            "a base node deleted by the overlay must not be a current posting"
        );
        let rebuilt = PropertyIndexRows::from_image(image).unwrap();
        assert_eq!(
            rebuilt
                .history
                .read()
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(7)),
                    EpochId::new(1),
                )
                .unwrap()
                .0,
            vec![deleted],
            "the retained pre-delete snapshot must keep the old value"
        );
        assert!(
            rebuilt
                .history
                .read()
                .candidates(
                    PropertyIndexPredicate::Equal(&Value::Int64(7)),
                    EpochId::new(2),
                )
                .unwrap()
                .0
                .is_empty(),
            "the delete epoch must close the historical property interval"
        );
    }

    #[test]
    fn close_deleted_nodes_preserves_nulls_and_rejects_conflicting_epochs() {
        let deleted = NodeId::new(10);
        let untouched = NodeId::new(11);
        let mut image = PropertyIndexImage {
            current: vec![(deleted, Value::Int64(7)), (untouched, Value::Int64(8))],
            history: vec![
                (deleted, vec![(EpochId::new(1), Value::Int64(7))]),
                (untouched, vec![(EpochId::new(1), Value::Int64(8))]),
            ],
            floor: EpochId::INITIAL,
        };
        image
            .close_deleted_nodes(&[(deleted, EpochId::new(2)), (deleted, EpochId::new(2))])
            .unwrap();
        let deleted_history = &image.history[0].1;
        assert_eq!(
            deleted_history,
            &[
                (EpochId::new(1), Value::Int64(7)),
                (EpochId::new(2), Value::Null)
            ]
        );
        assert_eq!(image.current, vec![(untouched, Value::Int64(8))]);
        let before = image.history.clone();
        assert!(
            image
                .close_deleted_nodes(&[(deleted, EpochId::new(3)), (deleted, EpochId::new(4))])
                .is_err()
        );
        assert_eq!(
            image.history, before,
            "invalid input must not alter history"
        );
    }
}
