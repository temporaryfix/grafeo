use super::*;
use grafeo_common::types::PropertyKey;

fn value(text: &str) -> Option<Value> {
    Some(Value::from(text))
}
fn key(text: &str) -> HashableValue {
    HashableValue::new(Value::from(text))
}
fn node(id: u64) -> NodeId {
    NodeId::new(id)
}
fn rows_with(values: &[(&str, &[u64])]) -> Arc<PropertyIndexRows> {
    let rows = Arc::new(PropertyIndexRows::new());
    for (value, ids) in values {
        rows.insert(key(value), ids.iter().copied().map(node).collect());
    }
    seed_history(rows)
}

fn seed_history(mut rows: Arc<PropertyIndexRows>) -> Arc<PropertyIndexRows> {
    Arc::get_mut(&mut rows)
        .unwrap()
        .seed_current_history(EpochId::INITIAL)
        .unwrap();
    rows
}

#[cfg(feature = "compact-store")]
#[test]
fn cold_predecessor_normalization_retains_omitted_values_and_requires_exact_proof() {
    use crate::graph::compact::{from_graph_store_preserving_ids, layered::LayeredStore};
    use crate::graph::lpg::LpgStore;

    let source = LpgStore::new().unwrap();
    let id = source.create_node(&["Cold"]);
    let base = from_graph_store_preserving_ids(&source).unwrap();
    let layered = LayeredStore::new(base, id.as_u64(), 0).unwrap();
    let overlay = layered.overlay_store();
    let foreign = LpgStore::new().unwrap();
    let rows = rows_with(&[]);
    let before = Value::List(Arc::from([Value::from("retained old value")]));
    let mut rejected = PropertyMaintenanceWorkspace::new(vec![(id, Some(before.clone()), None)]);
    let mut workspace = PropertyMaintenanceWorkspace::new(vec![(id, Some(before.clone()), None)]);
    let mut unchanged: Vec<_> = [
        (before.clone(), before.clone()),
        (Value::Float64(0.0), Value::Float64(-0.0)),
        (Value::Float64(f64::NAN), Value::Float64(f64::NAN)),
    ]
    .into_iter()
    .map(|(old, new)| PropertyMaintenanceWorkspace::new(vec![(id, Some(old), Some(new))]))
    .collect();
    {
        let pin = layered.pin_commit(&overlay).unwrap();
        let foreign_transition = foreign.pin_exclusive_unframed_transition().unwrap();
        assert!(
            rejected
                .normalize_unhydrated_cold_before(&rows, &pin, &foreign_transition)
                .is_err()
        );
        assert_eq!(rejected.changes[0].1, Some(before.clone()));
        assert!(rejected.unhydrated_before.is_empty());
        drop(foreign_transition);
        let transition = overlay.pin_exclusive_unframed_transition().unwrap();
        for no_op in &mut unchanged {
            no_op
                .normalize_unhydrated_cold_before(&rows, &pin, &transition)
                .unwrap();
            assert!(no_op.changes[0].1.is_some());
            assert!(no_op.unhydrated_before.is_empty());
            no_op.prepare(&rows).unwrap();
            assert!(no_op.buckets.is_empty());
            crate::allocation_test::start();
            no_op.validate(&rows).unwrap();
            no_op.install(&rows);
            let traffic = crate::allocation_test::stop();
            assert_eq!(traffic, crate::allocation_test::Counts::default());
            assert!(rows.is_empty());
        }
        workspace
            .normalize_unhydrated_cold_before(&rows, &pin, &transition)
            .unwrap();
        assert!(workspace.changes[0].1.is_none());
        assert_eq!(workspace.unhydrated_before.len(), 1);
        assert_eq!(workspace.unhydrated_before[0].inner(), &before);
        assert!(
            workspace
                .normalize_unhydrated_cold_before(&rows, &pin, &transition)
                .is_err()
        );
        workspace.prepare(&rows).unwrap();
        crate::allocation_test::start();
        workspace.validate(&rows).unwrap();
        workspace.install(&rows);
        let traffic = crate::allocation_test::stop();
        assert_eq!(traffic, crate::allocation_test::Counts::default());
        assert_eq!(workspace.unhydrated_before[0].inner(), &before);
    }
    assert!(rows.is_empty());
    assert!(!overlay.contains_node_identity(id));
}

#[test]
fn counter_and_nested_counter_keys_do_not_hash_under_final_writers() {
    use std::collections::{BTreeMap, HashMap};
    let counts = Arc::new(HashMap::from([("replica-a".to_owned(), 7)]));
    let negative = Arc::new(HashMap::from([("replica-b".to_owned(), 3)]));
    let grow = Value::GCounter(Arc::clone(&counts));
    let signed = Value::OnCounter {
        pos: counts,
        neg: negative,
    };
    let values = [
        grow.clone(),
        signed.clone(),
        Value::List(Arc::from([grow.clone(), signed.clone()])),
        Value::Map(Arc::new(BTreeMap::from([(
            PropertyKey::new("counter"),
            signed.clone(),
        )]))),
        Value::Path {
            nodes: Arc::from([grow.clone(), signed]),
            edges: Arc::from([grow]),
        },
    ];
    for (offset, value) in values.into_iter().enumerate() {
        let rows = rows_with(&[("old", &[1])]);
        let counter_key = HashableValue::new(value.clone());
        // Positive control: ordinary hashing really allocates and retires
        // scratch, so the final-phase witness can detect the original bug.
        crate::allocation_test::start();
        std::hint::black_box(rows.hasher().hash_one(&counter_key));
        let positive = crate::allocation_test::stop();
        assert!(positive.alloc > 0 && positive.dealloc > 0);
        let mut insert = PropertyMaintenanceWorkspace::new(vec![
            (node(1), value_string("old"), Some(value.clone())),
            (node(2), None, Some(value.clone())),
        ]);
        insert.prepare(&rows).unwrap();
        crate::allocation_test::start();
        insert.validate(&rows).unwrap();
        insert.install(&rows);
        let traffic = crate::allocation_test::stop();
        assert_eq!(
            traffic,
            crate::allocation_test::Counts::default(),
            "{offset}"
        );
        assert_eq!(rows.get(&counter_key).unwrap().len(), 2);

        let mut retire = PropertyMaintenanceWorkspace::new(vec![
            (node(1), Some(value.clone()), None),
            (node(2), Some(value), value_string("final")),
        ]);
        retire.prepare(&rows).unwrap();
        crate::allocation_test::start();
        retire.validate(&rows).unwrap();
        retire.install(&rows);
        let traffic = crate::allocation_test::stop();
        assert_eq!(
            traffic,
            crate::allocation_test::Counts::default(),
            "{offset}"
        );
        assert!(!rows.contains_key(&counter_key));
        assert_eq!(
            *rows.get(&key("final")).unwrap(),
            Membership::from_iter([node(2)])
        );
        assert!(retire.buckets.iter().any(|bucket| bucket.retired.is_some()));
    }
}

fn value_string(text: &str) -> Option<Value> {
    Some(Value::from(text))
}

#[test]
fn property_reservation_rejection_preserves_live_memberships_and_outer_fragments() {
    let mut admitted = false;
    for fail_at in 0..64 {
        let rows = rows_with(&[("old", &[1, 2]), ("retire", &[3])]);
        let mut workspace = PropertyMaintenanceWorkspace::new(vec![
            (node(1), value("old"), value("new")),
            (node(3), value("retire"), None),
            (node(4), None, value("old")),
        ]);
        super::super::RESERVATION_FAILURE.with(|point| point.set(Some(fail_at)));
        let result = workspace.prepare(&rows);
        super::super::RESERVATION_FAILURE.with(|point| point.set(None));
        assert_eq!(rows.len(), 2);
        assert_eq!(
            *rows.get(&key("old")).unwrap(),
            Membership::from_iter([node(1), node(2)])
        );
        assert_eq!(
            *rows.get(&key("retire")).unwrap(),
            Membership::from_iter([node(3)])
        );
        assert!(!rows.contains_key(&key("new")));
        assert!(
            workspace
                .anchor
                .as_ref()
                .is_some_and(|anchor| Arc::ptr_eq(anchor, &rows))
        );
        if result.is_ok() {
            admitted = true;
            break;
        }
        assert!(!workspace.prepared);
    }
    assert!(admitted, "the sweep must reach a successful preparation");
}

#[test]
fn tombstone_heavy_property_shard_reserves_every_missing_bucket_before_install() {
    let rows = Arc::new(PropertyIndexRows::new());
    let shard_id = rows.determine_map(&key("anchor"));
    let candidates: Vec<_> = (0..100_000)
        .map(|id| {
            (
                id,
                HashableValue::new(Value::Int64(i64::try_from(id).unwrap())),
            )
        })
        .filter(|(_, key)| rows.determine_map(key) == shard_id)
        .take(160)
        .collect();
    assert_eq!(candidates.len(), 160);
    for &(id, ref key) in &candidates[..96] {
        rows.insert(key.clone(), Membership::from_iter([node(id)]));
    }
    for (_, key) in candidates[..96].iter().step_by(2) {
        rows.remove(key);
    }
    let rows = seed_history(rows);
    let mut changes = Vec::new();
    for (id, key) in candidates[1..96].iter().step_by(2) {
        changes.push((node(*id), Some(key.inner().clone()), None));
    }
    for (id, key) in &candidates[96..] {
        changes.push((node(*id), None, Some(key.inner().clone())));
    }
    let mut workspace = PropertyMaintenanceWorkspace::new(changes);
    workspace.prepare(&rows).unwrap();
    assert_eq!(workspace.shards, vec![(shard_id, 64)]);
    crate::allocation_test::start();
    workspace.validate(&rows).unwrap();
    workspace.install(&rows);
    let traffic = crate::allocation_test::stop();
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    assert_eq!(rows.len(), 64);
    for (_, key) in &candidates[96..] {
        assert!(rows.contains_key(key));
    }
}

#[test]
fn sparse_property_memberships_prepare_without_changes_then_install_without_allocator_traffic() {
    let rows = rows_with(&[("old", &[1, 2]), ("remove", &[3]), ("untouched", &[4])]);
    let mut workspace = PropertyMaintenanceWorkspace::new(vec![
        (node(1), value("old"), value("new")),
        (node(3), value("remove"), None),
        (node(5), None, value("old")),
        (node(6), Some(Value::Null), value("new")),
        (node(4), value("untouched"), value("untouched")),
    ]);
    workspace.prepare(&rows).unwrap();
    assert_eq!(rows.get(&key("old")).unwrap().len(), 2);
    assert!(rows.contains_key(&key("remove")));
    assert!(!rows.contains_key(&key("new")));
    crate::allocation_test::start();
    workspace.validate(&rows).unwrap();
    workspace.install(&rows);
    let traffic = crate::allocation_test::stop();
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    assert_eq!(
        *rows.get(&key("old")).unwrap(),
        Membership::from_iter([node(2), node(5)])
    );
    assert_eq!(
        *rows.get(&key("new")).unwrap(),
        Membership::from_iter([node(1), node(6)])
    );
    assert!(!rows.contains_key(&key("remove")));
    assert_eq!(
        *rows.get(&key("untouched")).unwrap(),
        Membership::from_iter([node(4)])
    );
    assert!(
        workspace
            .buckets
            .iter()
            .any(|bucket| bucket.retired.is_some())
    );
    crate::allocation_test::start();
    drop(workspace);
    let retirement = crate::allocation_test::stop();
    assert!(retirement.dealloc > 0);
}

#[test]
fn sparse_property_maintenance_keeps_large_existing_membership_allocations_in_place() {
    let rows = Arc::new(PropertyIndexRows::new());
    let mut large = Membership::with_capacity_and_hasher(20_000, Default::default());
    large.extend((1..=16_384).map(node));
    rows.insert(key("shared"), large);
    rows.insert(key("untouched"), (30_000..40_000).map(node).collect());
    let rows = seed_history(rows);
    let allocation_identity = |name: &str| {
        let values = rows.get(&key(name)).unwrap();
        (
            values.capacity(),
            std::ptr::from_ref::<NodeId>(values.iter().next().unwrap()),
        )
    };
    let before_shared = allocation_identity("shared");
    let before_untouched = allocation_identity("untouched");
    let mut workspace = PropertyMaintenanceWorkspace::new(vec![
        (node(1), value("shared"), None),
        (node(20_000), None, value("shared")),
    ]);
    workspace.prepare(&rows).unwrap();
    assert_eq!(allocation_identity("shared"), before_shared);
    assert_eq!(allocation_identity("untouched"), before_untouched);
    assert_eq!(workspace.buckets.len(), 1);
    assert!(workspace.buckets[0].candidate.is_none());
    assert_eq!(workspace.buckets[0].adds, vec![node(20_000)]);
    assert_eq!(workspace.buckets[0].removes, vec![node(1)]);
    crate::allocation_test::start();
    workspace.validate(&rows).unwrap();
    workspace.install(&rows);
    let traffic = crate::allocation_test::stop();
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    assert_eq!(allocation_identity("untouched"), before_untouched);
    let shared = rows.get(&key("shared")).unwrap();
    // Hashbrown can mark the removed bucket as a tombstone. Reported capacity
    // is len + growth_left, not the allocation size, so removing one member
    // can lower it by one even though no reallocation occurred. The all-four-
    // zero allocator witness above is the actual allocation proof.
    assert!((before_shared.0 - 1..=before_shared.0).contains(&shared.capacity()));
    assert_eq!(shared.len(), 16_384);
    assert!(!shared.contains(&node(1)));
    assert!(shared.contains(&node(20_000)));
}

#[test]
fn sparse_property_maintenance_reserves_only_shards_with_missing_keys() {
    let rows = Arc::new(PropertyIndexRows::new());
    let before: Vec<_> = rows
        .shards()
        .iter()
        .map(|shard| shard.read().capacity())
        .collect();
    let mut workspace =
        PropertyMaintenanceWorkspace::new(vec![(node(1), None, value("one-new-key"))]);
    workspace.prepare(&rows).unwrap();
    let target = rows.determine_map(&key("one-new-key"));
    for (id, shard) in rows.shards().iter().enumerate() {
        if id == target {
            assert!(shard.read().capacity() > 0);
        } else {
            assert_eq!(shard.read().capacity(), before[id]);
        }
    }
    crate::allocation_test::start();
    workspace.validate(&rows).unwrap();
    workspace.install(&rows);
    let traffic = crate::allocation_test::stop();
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    assert_eq!(
        *rows.get(&key("one-new-key")).unwrap(),
        Membership::from_iter([node(1)])
    );
}

#[test]
fn sparse_property_maintenance_rejects_duplicate_or_inconsistent_input_without_publication() {
    for changes in [
        vec![
            (node(1), value("old"), value("new")),
            (node(1), value("old"), None),
        ],
        vec![(node(2), value("old"), value("new"))],
        vec![(node(1), value("missing"), value("new"))],
        vec![(node(1), None, value("old"))],
    ] {
        let rows = rows_with(&[("old", &[1])]);
        let mut workspace = PropertyMaintenanceWorkspace::new(changes);
        assert!(workspace.prepare(&rows).is_err());
        assert!(workspace.prepare(&rows).is_err());
        assert_eq!(rows.len(), 1);
        assert_eq!(
            *rows.get(&key("old")).unwrap(),
            Membership::from_iter([node(1)])
        );
        assert!(!rows.contains_key(&key("new")));
    }
}

#[test]
fn sparse_property_maintenance_rebind_faults_are_allocation_free() {
    let rows = rows_with(&[("old", &[1])]);
    let other = rows_with(&[("old", &[1])]);
    let mut workspace =
        PropertyMaintenanceWorkspace::new(vec![(node(1), value("old"), value("new"))]);
    workspace.prepare(&rows).unwrap();
    crate::allocation_test::start();
    let foreign = workspace.validate(&other);
    let traffic = crate::allocation_test::stop();
    assert!(foreign.is_err());
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    // Private test fault: normal writers cannot pass the retained transition.
    rows.get_mut(&key("old")).unwrap().remove(&node(1));
    crate::allocation_test::start();
    let stale = workspace.validate(&rows);
    let traffic = crate::allocation_test::stop();
    assert!(matches!(stale, Err(DataRebindError::Conflict(_))));
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    assert!(!rows.contains_key(&key("new")));
}

#[test]
fn sparse_property_maintenance_rejects_lost_shard_capacity_without_allocator_traffic() {
    let rows = Arc::new(PropertyIndexRows::new());
    let mut workspace = PropertyMaintenanceWorkspace::new(vec![(node(1), None, value("new"))]);
    workspace.prepare(&rows).unwrap();
    let shard = rows.determine_map(&key("new"));
    rows.shards()[shard]
        .write()
        .shrink_to(0, |(key, _)| rows.hasher().hash_one(key));
    crate::allocation_test::start();
    let rejected = workspace.validate(&rows);
    let traffic = crate::allocation_test::stop();
    assert!(rejected.is_err());
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    assert!(rows.is_empty());
}
