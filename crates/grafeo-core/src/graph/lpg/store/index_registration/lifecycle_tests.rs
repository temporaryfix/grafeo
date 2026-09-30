use super::*;
use crate::graph::write_permit::{WriteAuthority, with_authority};
use grafeo_common::types::Value;

#[test]
fn property_observation_does_not_retain_dropped_rows() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("code");
    let rows = Arc::downgrade(&store.property_indexes.read()[&PropertyKey::new("code")].payload);
    let observed = store.observe_property_index("code").unwrap();
    assert!(store.drop_property_index("code"));
    assert!(
        rows.upgrade().is_none(),
        "stale equality evidence must not retain index rows"
    );
    assert!(store.validate_index_registration(&observed).is_err());
}

#[cfg(feature = "vector-index")]
#[test]
fn vector_observation_does_not_retain_dropped_payload() {
    let store = LpgStore::new().unwrap();
    let index = super::tests::vector();
    let weak = Arc::downgrade(&index);
    store.add_vector_index("Doc", "embedding", index);
    let observed = store.observe_vector_index("Doc", "embedding").unwrap();
    assert!(store.remove_vector_index("Doc", "embedding"));
    assert!(
        weak.upgrade().is_none(),
        "stale equality evidence must not retain vector payload"
    );
    assert!(store.validate_index_registration(&observed).is_err());
}

#[cfg(feature = "text-index")]
#[test]
fn text_observation_does_not_retain_dropped_caller_or_payload() {
    let store = LpgStore::new().unwrap();
    let caller = super::tests::text();
    let weak = Arc::downgrade(&caller);
    store.add_text_index("Doc", "body", caller);
    let target = store.text_indexes.read()[&encode_index_key("Doc", "body")].target_identity();
    let retained_target_control = target.upgrade().unwrap();
    let observed = store.observe_text_index("Doc", "body").unwrap();
    assert!(store.remove_text_index("Doc", "body"));
    assert!(
        weak.upgrade().is_none(),
        "stale equality evidence must not retain the text caller shell"
    );
    assert!(
        target.upgrade().is_some(),
        "the held control must retain the actual concrete target"
    );
    drop(retained_target_control);
    assert!(
        target.upgrade().is_none(),
        "stale equality evidence must not retain the separate concrete text target"
    );
    assert!(store.validate_index_registration(&observed).is_err());
}

#[test]
fn observation_does_not_retain_its_store() {
    let store = Arc::new(LpgStore::new().unwrap());
    store.create_property_index("code");
    let weak = Arc::downgrade(&store);
    let observed = store.observe_property_index("code").unwrap();
    drop(store);
    assert!(weak.upgrade().is_none());
    drop(observed);
}

#[test]
fn moving_store_value_keeps_its_physical_identity() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("code");
    let observed = store.observe_property_index("code").unwrap();
    let moved = Box::new(store);
    moved.validate_index_registration(&observed).unwrap();
    moved.drop_index_if_unchanged(observed).unwrap();
    assert!(!moved.has_property_index("code"));
}

#[test]
fn foreign_store_observation_cannot_remove_an_equal_key() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    first.create_property_index("code");
    second.create_property_index("code");
    let foreign = first.observe_property_index("code").unwrap();
    let own = second.observe_property_index("code").unwrap();
    assert!(second.validate_index_registration(&foreign).is_err());
    assert!(second.drop_index_if_unchanged(foreign).is_err());
    second.validate_index_registration(&own).unwrap();
    assert!(first.has_property_index("code"));
    assert!(second.has_property_index("code"));
}

#[test]
fn same_arc_clear_recreate_does_not_resurrect_observation() {
    let store = Arc::new(LpgStore::new().unwrap());
    store.create_property_index("code");
    let old = store.observe_property_index("code").unwrap();
    store.clear();
    let node = store.create_node(&["Item"]);
    store.set_node_property(node, "code", Value::Int64(42));
    store.create_property_index("code");
    let current = store.observe_property_index("code").unwrap();
    assert!(store.validate_index_registration(&old).is_err());
    assert!(store.drop_index_if_unchanged(old).is_err());
    store.validate_index_registration(&current).unwrap();
    assert_eq!(
        store.find_nodes_by_property("code", &Value::Int64(42)),
        vec![node]
    );
}

#[test]
fn observation_is_not_sealed_store_write_authority() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("code");
    let retained = store.observe_property_index("code").unwrap();
    let denied = store.observe_property_index("code").unwrap();
    let authority = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&authority));
    assert!(store.validate_index_registration(&retained).is_err());
    assert!(store.drop_index_if_unchanged(denied).is_err());
    assert!(store.has_property_index("code"));
    with_authority(&authority, || {
        store.validate_index_registration(&retained).unwrap();
        store.drop_index_if_unchanged(retained).unwrap();
    });
    assert!(!store.has_property_index("code"));
}

#[test]
fn already_removed_registration_returns_error() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("code");
    let first = store.observe_property_index("code").unwrap();
    let second = store.observe_property_index("code").unwrap();
    store.drop_index_if_unchanged(first).unwrap();
    assert!(store.validate_index_registration(&second).is_err());
    assert!(store.drop_index_if_unchanged(second).is_err());
}

#[test]
fn property_updates_and_duplicate_create_preserve_registration() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Item"]);
    store.set_node_property(node, "code", Value::Int64(7));
    store.create_property_index("code");
    let observed = store.observe_property_index("code").unwrap();
    store.create_property_index("code");
    store.set_node_property(node, "code", Value::Int64(9));
    store.add_label(node, "Other");
    store.validate_index_registration(&observed).unwrap();
    assert_eq!(
        store.find_nodes_by_property("code", &Value::Int64(9)),
        vec![node]
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn failed_foreign_vector_registration_preserves_current_registration() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    let foreign = super::tests::vector();
    first.add_vector_index("Item", "embedding", Arc::clone(&foreign));
    second.add_vector_index("Item", "embedding", super::tests::vector());
    let observed = second.observe_vector_index("Item", "embedding").unwrap();
    second.add_vector_index("Item", "embedding", foreign);
    second.validate_index_registration(&observed).unwrap();
    second.drop_index_if_unchanged(observed).unwrap();
    assert!(first.get_vector_index("Item", "embedding").is_some());
}

#[cfg(feature = "text-index")]
#[test]
fn failed_foreign_text_registration_preserves_current_registration() {
    let first = LpgStore::new().unwrap();
    let second = LpgStore::new().unwrap();
    let foreign = super::tests::text();
    first.add_text_index("Doc", "body", Arc::clone(&foreign));
    second.add_text_index("Doc", "body", super::tests::text());
    let observed = second.observe_text_index("Doc", "body").unwrap();
    second.add_text_index("Doc", "body", foreign);
    second.validate_index_registration(&observed).unwrap();
    second.drop_index_if_unchanged(observed).unwrap();
    assert!(first.get_text_index("Doc", "body").is_some());
}

#[cfg(feature = "text-index")]
#[test]
fn replacing_text_caller_shell_cannot_redirect_observed_drop() {
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::NodeId;
    let store = LpgStore::new().unwrap();
    let caller = super::tests::text();
    store.add_text_index("Doc", "body", Arc::clone(&caller));
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "body", Value::from("original document"));
    let view = store.get_text_index("Doc", "body").unwrap();
    let observed = store.observe_text_index("Doc", "body").unwrap();
    let mut unrelated = InvertedIndex::new(BM25Config::default());
    unrelated.insert(NodeId::new(99), "unrelated shell");
    *caller.write() = unrelated;
    store.validate_index_registration(&observed).unwrap();
    store.drop_index_if_unchanged(observed).unwrap();
    assert!(store.get_text_index("Doc", "body").is_none());
    assert_eq!(view.read().search("original", 10)[0].0, node);
    assert_eq!(caller.read().search("unrelated", 10)[0].0, NodeId::new(99));
}

#[cfg(feature = "text-index")]
#[test]
fn text_updates_labels_and_gc_preserve_registration()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use grafeo_common::types::EpochId;
    let store = LpgStore::new().unwrap();
    store.add_text_index("Doc", "body", super::tests::text());
    let observed = store.observe_text_index("Doc", "body").unwrap();
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "body", Value::from("original document"));
    store.set_node_property(node, "body", Value::from("replacement document"));
    store.remove_label(node, "Doc");
    store.add_label(node, "Doc");
    store.gc_text_indexes(EpochId::new(1))?;
    store.validate_index_registration(&observed).unwrap();
    assert_eq!(
        store
            .get_text_index("Doc", "body")
            .unwrap()
            .read()
            .search("replacement", 10)[0]
            .0,
        node
    );
    Ok(())
}

#[cfg(feature = "vector-index")]
#[test]
fn vector_updates_labels_and_gc_preserve_registration() -> grafeo_common::utils::error::Result<()> {
    use grafeo_common::types::EpochId;
    let store = LpgStore::new().unwrap();
    store.add_vector_index("Doc", "embedding", super::tests::vector());
    let observed = store.observe_vector_index("Doc", "embedding").unwrap();
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()));
    store.set_node_property(node, "embedding", Value::Vector(vec![0.0, 1.0, 0.0].into()));
    store.remove_label(node, "Doc");
    store.add_label(node, "Doc");
    store.gc_vector_indexes(EpochId::new(1))?;
    store.validate_index_registration(&observed).unwrap();
    assert_eq!(store.get_vector_index("Doc", "embedding").unwrap().len(), 1);
    Ok(())
}

#[cfg(all(feature = "text-index", feature = "vector-index"))]
#[test]
fn conditional_drop_never_crosses_index_families_at_the_same_local_key() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("body");
    store.add_text_index("Doc", "body", super::tests::text());
    store.add_vector_index("Doc", "body", super::tests::vector());
    let property = store.observe_property_index("body").unwrap();
    let text = store.observe_text_index("Doc", "body").unwrap();
    let vector = store.observe_vector_index("Doc", "body").unwrap();
    store.drop_index_if_unchanged(text).unwrap();
    store.validate_index_registration(&property).unwrap();
    store.validate_index_registration(&vector).unwrap();
    store.drop_index_if_unchanged(vector).unwrap();
    store.validate_index_registration(&property).unwrap();
    store.drop_index_if_unchanged(property).unwrap();
    assert!(!store.has_property_index("body"));
}

#[cfg(feature = "text-index")]
#[test]
fn unpublished_text_hydration_preserves_its_new_registration() {
    use crate::index::text::TextIndexSection;
    use grafeo_common::storage::section::Section;
    let source = LpgStore::new().unwrap();
    source.add_text_index("Doc", "body", super::tests::text());
    let node = source.create_node(&["Doc"]);
    source.set_node_property(node, "body", Value::from("restored document"));
    let bytes = TextIndexSection::from_views(
        source
            .text_index_entries()
            .into_iter()
            .map(|(_, view)| {
                (
                    crate::graph::lpg::PhysicalIndexKey::text(
                        grafeo_common::types::GraphPath::root(),
                        "Doc",
                        "body",
                    ),
                    view,
                )
            })
            .collect(),
    )
    .serialize()
    .unwrap();
    let target = LpgStore::new().unwrap();
    target.add_text_index("Doc", "body", super::tests::text());
    let observed = target.observe_text_index("Doc", "body").unwrap();
    let mut section = TextIndexSection::for_unpublished_recovery_views(
        target
            .text_index_entries()
            .into_iter()
            .map(|(_, view)| {
                (
                    crate::graph::lpg::PhysicalIndexKey::text(
                        grafeo_common::types::GraphPath::root(),
                        "Doc",
                        "body",
                    ),
                    view,
                )
            })
            .collect(),
    );
    section.deserialize(&bytes).unwrap();
    target.validate_index_registration(&observed).unwrap();
    assert_eq!(section.serialize().unwrap(), bytes);
    assert_eq!(
        target
            .get_text_index("Doc", "body")
            .unwrap()
            .read()
            .search("restored", 10)[0]
            .0,
        node
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn unpublished_vector_hydration_preserves_its_new_registration() {
    use crate::index::vector::VectorStoreSection;
    use grafeo_common::storage::section::Section;
    let source = LpgStore::new().unwrap();
    source.add_vector_index("Doc", "embedding", super::tests::vector());
    let node = source.create_node(&["Doc"]);
    source.set_node_property(node, "embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()));
    let bytes = VectorStoreSection::from_views(
        source
            .vector_index_entries()
            .into_iter()
            .map(|(_, view)| {
                (
                    crate::graph::lpg::PhysicalIndexKey::vector(
                        grafeo_common::types::GraphPath::root(),
                        "Doc",
                        "embedding",
                    ),
                    view,
                )
            })
            .collect(),
    )
    .serialize()
    .unwrap();
    let target = LpgStore::new().unwrap();
    target.add_vector_index("Doc", "embedding", super::tests::vector());
    let observed = target.observe_vector_index("Doc", "embedding").unwrap();
    let mut section = VectorStoreSection::for_unpublished_recovery_views(
        target
            .vector_index_entries()
            .into_iter()
            .map(|(_, view)| {
                (
                    crate::graph::lpg::PhysicalIndexKey::vector(
                        grafeo_common::types::GraphPath::root(),
                        "Doc",
                        "embedding",
                    ),
                    view,
                )
            })
            .collect(),
    );
    section.deserialize(&bytes).unwrap();
    target.validate_index_registration(&observed).unwrap();
    assert_eq!(section.serialize().unwrap(), bytes);
    assert_eq!(
        target.get_vector_index("Doc", "embedding").unwrap().len(),
        1
    );
}
