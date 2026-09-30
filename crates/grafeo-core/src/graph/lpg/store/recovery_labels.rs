//! Exact complete-label-image replay on an existing, live recovery identity.

use super::{ExclusiveBulkRestoreContext, LpgStore};
use arcstr::ArcStr;
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, NodeId};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_common::utils::hash::FxHashSet;
use std::sync::atomic::Ordering;

#[derive(Default)]
struct ReplayLabels {
    // Completed scratch retires after the exclusive transition. Historical
    // images stay in their existing log; replay reserves one append in place.
    baseline: FxHashSet<u32>,
    image: FxHashSet<u32>,
    additions: Vec<usize>,
    removals: Vec<usize>,
    changed_names: Vec<ArcStr>,
}

fn corruption(message: &str) -> Error {
    Error::Storage(StorageError::Corruption(format!(
        "exact label-image replay: {message}"
    )))
}

impl LpgStore {
    pub(crate) fn validate_node_label_replay_image(
        id: NodeId,
        epoch: EpochId,
        labels: &[ArcStr],
    ) -> Result<u16> {
        if !id.is_valid() || epoch == EpochId::PENDING {
            return Err(corruption("invalid identity or PENDING epoch"));
        }
        let count = u16::try_from(labels.len())
            .map_err(|_| corruption("complete label image exceeds u16 capacity"))?;
        let mut unique = FxHashSet::default();
        unique
            .try_reserve(labels.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for label in labels {
            if !unique.insert(label.as_str()) {
                return Err(corruption("complete label image contains duplicate names"));
            }
        }
        Ok(count)
    }

    /// Appends one exact complete label image at the current recovery frontier.
    ///
    /// The node must already have a live committed structural lifetime. Replay
    /// images before its deletion, including same-epoch create/delete. Every
    /// explicit image is retained, even if equal to its predecessor. Replay
    /// checkpoint/group fencing, not this method, owns duplicate application.
    ///
    /// Call only on an unpublished or otherwise quiescent recovery store and
    /// with its exact write authority. The method owns exclusive LPG admission;
    /// callers must not retain a shared LPG mutation pin. Current derived
    /// indexes are reconciled after structural and label guards have drained.
    ///
    /// # Errors
    ///
    /// Rejects invalid, closed, pending or missing identities; a non-current or
    /// out-of-order epoch; invalid label history/membership; duplicate or
    /// oversized images; denied authority; and fallible capacity reservations.
    #[doc(hidden)]
    pub fn replay_node_labels_at_epoch(
        &self,
        id: NodeId,
        epoch: EpochId,
        labels: &[ArcStr],
    ) -> Result<()> {
        let count = Self::validate_node_label_replay_image(id, epoch, labels)?;
        let requested = [(epoch, labels.to_vec())];
        let mut replay = ReplayLabels::default();
        let transition = if ExclusiveBulkRestoreContext::is_active(self) {
            if self.pin_mutation().is_none() {
                return Err(corruption("current store write authority is not held"));
            }
            None
        } else {
            Some(
                self.pin_exclusive_unframed_transition()
                    .ok_or_else(|| corruption("current store write authority is not held"))?,
            )
        };
        if self.current_epoch() != epoch {
            return Err(corruption(
                "image epoch differs from the current recovery frontier",
            ));
        }
        let created = self.validate_label_replay_lifetime(id, epoch)?;
        {
            let registry = self.label_registry.read();
            let index = self.label_index.read();
            let histories = self.node_labels.read();
            let history = histories
                .get(&id)
                .ok_or_else(|| corruption("live node has no complete label history"))?;
            let baseline = history
                .latest()
                .ok_or_else(|| corruption("live node has an empty label history"))?;
            // Ordinary and exact-restore producers maintain sorted, valid
            // logs. Qualify the frontier and current image without rescanning
            // or cloning every earlier image on each WAL record.
            if history
                .latest_epoch()
                .is_some_and(|at| at == EpochId::PENDING || at > epoch)
            {
                return Err(corruption(
                    "existing label history is pending or out of order",
                ));
            }
            if baseline.len() > usize::from(u16::MAX)
                || baseline
                    .iter()
                    .any(|label| registry.get_name(*label).is_none())
            {
                return Err(corruption(
                    "existing label image has invalid names or capacity",
                ));
            }
            let birth = history.history().partition_point(|(at, _)| *at < created);
            if history
                .history()
                .get(birth)
                .is_none_or(|(at, _)| *at != created)
            {
                return Err(corruption(
                    "latest structural lifetime has no creation label image",
                ));
            }
            for label in baseline {
                let slot = usize::try_from(*label).map_err(|_| AllocError::InsufficientSpace)?;
                if !index
                    .get(slot)
                    .is_some_and(|members| members.contains_key(&id))
                {
                    return Err(corruption("current label image has missing membership"));
                }
            }
            replay.baseline.clone_from(baseline);
        }
        // This token rolls back any newly interned suffix on error. Its scope
        // encloses the final map guards, but remains inside the source transition.
        let prepared_labels = self.prepare_node_labels(&requested)?;
        replay
            .image
            .try_reserve(labels.len())
            .map_err(|_| AllocError::OutOfMemory)?;
        for label in labels {
            let label_id = prepared_labels
                .id(label.as_str())
                .ok_or_else(|| corruption("prepared label name has no identity"))?;
            replay.image.insert(label_id);
        }
        {
            let registry = self.label_registry.read();
            for label in replay.baseline.symmetric_difference(&replay.image) {
                let name = registry
                    .get_name(*label)
                    .ok_or_else(|| corruption("changed label identity is absent"))?;
                replay.changed_names.push(name.clone());
                let slot = usize::try_from(*label).map_err(|_| AllocError::InsufficientSpace)?;
                if replay.image.contains(label) {
                    replay.additions.push(slot);
                } else {
                    replay.removals.push(slot);
                }
            }
        }
        // The transition excludes log replacement between this fallible
        // reservation and the final append under structural writers.
        self.node_labels
            .write()
            .get_mut(&id)
            .ok_or_else(|| corruption("qualified label history disappeared"))?
            .try_reserve(1)?;
        // Reserve membership capacity without any structural entity guard.
        {
            let mut index = self.label_index.write();
            for &slot in &replay.additions {
                let members = index
                    .get_mut(slot)
                    .ok_or_else(|| corruption("prepared membership slot is absent"))?;
                if members.contains_key(&id) {
                    return Err(corruption("new label already has inconsistent membership"));
                }
                members
                    .try_reserve(1)
                    .map_err(|_| AllocError::OutOfMemory)?;
            }
        }
        #[cfg(test)]
        if FAIL_BEFORE_INSTALL.with(std::cell::Cell::get) {
            return Err(AllocError::OutOfMemory.into());
        }
        {
            #[cfg(not(feature = "tiered-storage"))]
            let mut nodes = self.nodes.write();
            #[cfg(feature = "tiered-storage")]
            let nodes = self.node_versions.write();
            let mut index = self.label_index.write();
            let mut histories = self.node_labels.write();
            #[cfg(not(feature = "tiered-storage"))]
            let record = nodes
                .get_mut(&id)
                .and_then(|chain| chain.latest_mut())
                .ok_or_else(|| corruption("qualified structural identity disappeared"))?;
            #[cfg(feature = "tiered-storage")]
            if !nodes.contains_key(&id) {
                return Err(corruption("qualified structural identity disappeared"));
            }
            let history = histories
                .get_mut(&id)
                .ok_or_else(|| corruption("qualified label history disappeared"))?;
            // The continuously retained transition qualifies these keyed
            // operations; there is no allocation or fallible work after here.
            for &slot in &replay.removals {
                if let Some(members) = index.get_mut(slot) {
                    members.remove(&id);
                }
            }
            for &slot in &replay.additions {
                if let Some(members) = index.get_mut(slot) {
                    members.insert(id, ());
                }
            }
            history.append(epoch, std::mem::take(&mut replay.image));
            #[cfg(not(feature = "tiered-storage"))]
            record.set_label_count(count);
            // Tiered records remain immutable, as in prepared data publication;
            // their label log and membership are the authoritative sources.
            #[cfg(feature = "tiered-storage")]
            let _ = count;
        }
        prepared_labels.commit();
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        drop(transition);
        for label in &replay.changed_names {
            #[cfg(feature = "vector-index")]
            self.refresh_vector_indexes_for_label(id, label.as_str());
            #[cfg(feature = "text-index")]
            self.refresh_text_indexes_for_label(id, label.as_str());
            #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
            let _ = label;
        }
        Ok(())
    }

    #[cfg(not(feature = "tiered-storage"))]
    fn validate_label_replay_lifetime(&self, id: NodeId, epoch: EpochId) -> Result<EpochId> {
        let nodes = self.nodes.read();
        let (info, record) = nodes
            .get(&id)
            .and_then(|chain| chain.history().next())
            .ok_or_else(|| corruption("node identity is absent"))?;
        if info.created_epoch == EpochId::PENDING
            || info.created_epoch > epoch
            || info.deleted_epoch.is_some()
            || record.is_deleted()
        {
            return Err(corruption(
                "latest structural lifetime is not live and committed",
            ));
        }
        Ok(info.created_epoch)
    }

    #[cfg(feature = "tiered-storage")]
    fn validate_label_replay_lifetime(&self, id: NodeId, epoch: EpochId) -> Result<EpochId> {
        let version = self
            .node_versions
            .read()
            .get(&id)
            .and_then(|index| index.latest())
            .ok_or_else(|| corruption("node identity is absent"))?;
        if version.epoch() == EpochId::PENDING
            || version.epoch() > epoch
            || version.deleted_epoch().is_some()
        {
            return Err(corruption(
                "latest structural lifetime is not live and committed",
            ));
        }
        let record = self
            .read_node_record(&version)
            .ok_or_else(|| corruption("latest structural record is absent"))?;
        if record.is_deleted() {
            return Err(corruption("latest structural record is deleted"));
        }
        Ok(version.epoch())
    }
}

#[cfg(test)]
std::thread_local! {
    static FAIL_BEFORE_INSTALL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
mod tests {
    use super::{FAIL_BEFORE_INSTALL, LpgStore};
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use arcstr::ArcStr;
    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    use grafeo_common::types::Value;
    use grafeo_common::types::{EpochId, NodeId, TransactionId};

    fn labels(names: &[&str]) -> Vec<ArcStr> {
        names.iter().map(|name| ArcStr::from(*name)).collect()
    }

    #[test]
    fn appending_to_long_label_history_does_not_clone_prior_images() {
        let measure = |prior_images: usize| {
            let store = LpgStore::new().unwrap();
            let epoch = EpochId::new(3);
            store.sync_epoch(epoch);
            let id = store.create_node(&["Stable"]);
            {
                let mut histories = store.node_labels.write();
                let history = histories.get_mut(&id).unwrap();
                let baseline = history.latest().unwrap().clone();
                history.try_reserve(prior_images + 1).unwrap();
                for _ in 1..prior_images {
                    history.append(epoch, baseline.clone());
                }
            }
            let image = labels(&["Stable"]);
            // Exclude the thread's first exclusive-read TLS stack allocation
            // from both samples. This pins exactly the same admission scope
            // as replay without touching any label image or history capacity.
            let warmup = store.pin_exclusive_unframed_transition();
            assert!(warmup.is_some());
            drop(warmup);
            crate::allocation_test::start();
            let result = store.replay_node_labels_at_epoch(id, epoch, &image);
            let counts = crate::allocation_test::stop();
            result.unwrap();
            assert_eq!(
                store.node_labels.read().get(&id).unwrap().len(),
                prior_images + 1
            );
            counts
        };
        let short = measure(2);
        let long = measure(257);
        assert!(
            short.alloc > 0,
            "positive control must observe image preparation"
        );
        assert_eq!(
            long, short,
            "replay must not allocate or retire prior images"
        );
    }

    #[test]
    fn exact_images_preserve_equal_entries_empty_sets_and_zero_width_deletion() {
        let store = LpgStore::new().unwrap();
        let epoch = EpochId::new(3);
        store.sync_epoch(epoch);
        let id = store.create_node(&["Initial"]);
        let image = labels(&["Final", "Shared"]);
        store
            .replay_node_labels_at_epoch(id, epoch, &image)
            .unwrap();
        store
            .replay_node_labels_at_epoch(id, epoch, &image)
            .unwrap();
        store.replay_node_labels_at_epoch(id, epoch, &[]).unwrap();
        assert_eq!(
            store.node_label_history(id),
            vec![
                (epoch, labels(&["Initial"])),
                (epoch, image.clone()),
                (epoch, image),
                (epoch, vec![]),
            ]
        );
        assert!(store.nodes_by_label("Initial").is_empty());
        assert!(store.nodes_by_label("Final").is_empty());
        assert!(store.nodes_by_label("Shared").is_empty());
        assert_eq!(store.node_count(), 1);
        #[cfg(not(feature = "tiered-storage"))]
        assert_eq!(
            store
                .nodes
                .read()
                .get(&id)
                .unwrap()
                .latest()
                .unwrap()
                .label_count(),
            0
        );
        let history = store.node_label_history(id);
        assert!(store.delete_node(id));
        assert_eq!(store.node_label_history(id), history);
        assert_eq!(
            store
                .get_node_history(id)
                .iter()
                .map(|(created, deleted, _)| (*created, *deleted))
                .collect::<Vec<_>>(),
            vec![(epoch, Some(epoch))]
        );
        assert!(store.get_node(id).is_none());
        assert_eq!(store.node_count(), 0);
        assert!(
            store
                .replay_node_labels_at_epoch(id, epoch, &labels(&["Resurrect"]))
                .is_err()
        );
        assert!(store.label_id("Resurrect").is_none());
    }

    #[test]
    fn invalid_replay_and_failed_preparation_leave_labels_and_catalog_unchanged() {
        let store = LpgStore::new().unwrap();
        let epoch = EpochId::new(5);
        store.sync_epoch(epoch);
        let id = store.create_node(&["Before"]);
        let before = store.node_label_history(id);
        let catalog = store.all_labels();
        let high_water = store.next_node_id();
        for (target, at, image) in [
            (NodeId::INVALID, epoch, labels(&["Denied"])),
            (NodeId::new(999), epoch, labels(&["Denied"])),
            (id, EpochId::PENDING, labels(&["Denied"])),
            (id, EpochId::new(4), labels(&["Denied"])),
            (id, EpochId::new(6), labels(&["Denied"])),
            (id, epoch, labels(&["Duplicate", "Duplicate"])),
        ] {
            assert!(
                store
                    .replay_node_labels_at_epoch(target, at, &image)
                    .is_err()
            );
        }
        let oversized: Vec<_> = (0..=u16::MAX)
            .map(|index| ArcStr::from(format!("Large{index}")))
            .collect();
        assert!(
            store
                .replay_node_labels_at_epoch(id, epoch, &oversized)
                .is_err()
        );
        FAIL_BEFORE_INSTALL.with(|flag| flag.set(true));
        let failed = store.replay_node_labels_at_epoch(id, epoch, &labels(&["Unpublished"]));
        FAIL_BEFORE_INSTALL.with(|flag| flag.set(false));
        assert!(failed.is_err());
        assert_eq!(store.node_label_history(id), before);
        assert_eq!(store.all_labels(), catalog);
        assert_eq!(store.next_node_id(), high_water);
        assert_eq!(store.nodes_by_label("Before"), vec![id]);
        assert!(store.label_id("Unpublished").is_none());
        store
            .replay_node_labels_at_epoch(id, epoch, &labels(&["After"]))
            .unwrap();
        assert_eq!(store.nodes_by_label("After"), vec![id]);
        assert!(store.nodes_by_label("Before").is_empty());
    }

    #[test]
    fn exact_replay_rejects_pending_target_history_and_missing_membership() {
        let store = LpgStore::new().unwrap();
        let epoch = EpochId::new(7);
        store.sync_epoch(epoch);
        let pending = store.create_node_versioned(&["Pending"], epoch, TransactionId::new(91));
        assert!(
            store
                .replay_node_labels_at_epoch(pending, epoch, &labels(&["Denied"]))
                .is_err()
        );
        let foreign = store.create_node(&["Foreign"]);
        store.add_label_versioned(foreign, "PendingLabel", TransactionId::new(92));
        let foreign_history = store.node_label_history(foreign);
        assert!(
            store
                .replay_node_labels_at_epoch(foreign, epoch, &labels(&["Denied"]))
                .is_err()
        );
        let valid = store.create_node(&["Valid"]);
        store
            .replay_node_labels_at_epoch(valid, epoch, &labels(&["Updated"]))
            .unwrap();
        assert_eq!(store.node_label_history(foreign), foreign_history);
        assert_eq!(
            store.node_label_history(pending),
            vec![(EpochId::PENDING, labels(&["Pending"]))]
        );
        let missing = store.label_id("Updated").unwrap() as usize;
        store.label_index.write()[missing].remove(&valid);
        let valid_history = store.node_label_history(valid);
        assert!(
            store
                .replay_node_labels_at_epoch(valid, epoch, &labels(&["Denied"]))
                .is_err()
        );
        assert_eq!(store.node_label_history(valid), valid_history);
        assert!(store.label_id("Denied").is_none());
    }

    #[test]
    fn exact_replay_requires_the_owning_write_authority() {
        let store = LpgStore::new().unwrap();
        let id = store.create_node(&["Before"]);
        let epoch = store.current_epoch();
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(store.seal_unframed_writes(&owner));
        assert!(
            store
                .replay_node_labels_at_epoch(id, epoch, &labels(&["Denied"]))
                .is_err()
        );
        assert!(
            with_authority(&foreign, || store.replay_node_labels_at_epoch(
                id,
                epoch,
                &labels(&["Denied"])
            ))
            .is_err()
        );
        assert!(store.label_id("Denied").is_none());
        with_authority(&owner, || {
            store.replay_node_labels_at_epoch(id, epoch, &labels(&["Accepted"]))
        })
        .unwrap();
        assert_eq!(store.nodes_by_label("Accepted"), vec![id]);
    }

    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    #[test]
    fn exact_replay_reconciles_existing_vector_and_text_handles() {
        use crate::index::text::{BM25Config, InvertedIndex};
        use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
        use parking_lot::RwLock;
        use std::sync::Arc;

        let store = LpgStore::new().unwrap();
        let vector = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            2,
            DistanceMetric::Euclidean,
        ))));
        let text = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        store.add_vector_index("Doc", "vector", Arc::clone(&vector));
        store.add_text_index("Doc", "body", Arc::clone(&text));
        let id = store.create_node_with_props(
            &["Other"],
            [
                ("vector", Value::Vector(vec![1.0_f32, 2.0].into())),
                ("body", Value::from("exact image reconciliation")),
            ],
        );
        let properties = store.node_property_history(id);
        let epoch = store.current_epoch();
        store
            .replay_node_labels_at_epoch(id, epoch, &labels(&["Doc"]))
            .unwrap();
        assert!(vector.contains(id));
        assert_eq!(
            text.read()
                .search("reconciliation", 10)
                .iter()
                .map(|(node, _)| *node)
                .collect::<Vec<_>>(),
            vec![id]
        );
        store.replay_node_labels_at_epoch(id, epoch, &[]).unwrap();
        assert!(!vector.contains(id));
        assert!(text.read().search("reconciliation", 10).is_empty());
        assert_eq!(store.node_property_history(id), properties);
    }
}
