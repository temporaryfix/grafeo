use super::*;
use grafeo_common::types::Value;

#[test]
fn property_drop_recreate_rejects_old_observation_without_removing_replacement_rows() {
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Item"]);
    store.set_node_property(node, "code", Value::Int64(7));
    store.create_property_index("code");
    let observed = store.observe_property_index("code").unwrap();
    let validation = store.observe_property_index("code").unwrap();
    assert!(store.drop_property_index("code"));
    store.set_node_property(node, "code", Value::Int64(9));
    store.create_property_index("code");
    let replacement = store.observe_property_index("code").unwrap();

    assert!(
        store.drop_index_if_unchanged(observed).is_err(),
        "stale conditional drop must not remove the replacement"
    );
    assert!(store.validate_index_registration(&validation).is_err());
    assert!(store.validate_index_registration(&replacement).is_ok());
    assert!(store.has_property_index("code"));
    assert_eq!(
        store.find_nodes_by_property("code", &Value::Int64(9)),
        vec![node]
    );
    assert!(
        store
            .find_nodes_by_property("code", &Value::Int64(7))
            .is_empty()
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn same_vector_arc_reregistration_is_not_the_observed_registration() {
    use crate::index::vector::VectorStoreSection;
    use grafeo_common::storage::section::Section;
    let store = LpgStore::new().unwrap();
    let index = super::tests::vector();
    let node = store.create_node(&["Item"]);
    store.set_node_property(node, "embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()));
    index.insert(node, &[1.0, 0.0, 0.0], &|_| None);
    store.add_vector_index("Item", "embedding", Arc::clone(&index));
    let observed = store.observe_vector_index("Item", "embedding").unwrap();
    let validation = store.observe_vector_index("Item", "embedding").unwrap();
    let section = VectorStoreSection::new(vec![(
        crate::graph::lpg::PhysicalIndexKey::vector(
            grafeo_common::types::GraphPath::root(),
            "Item",
            "embedding",
        ),
        Arc::clone(&index),
    )]);
    let before = section.serialize().unwrap();
    assert!(store.remove_vector_index("Item", "embedding"));
    store.add_vector_index("Item", "embedding", Arc::clone(&index));
    let replacement = store.observe_vector_index("Item", "embedding").unwrap();

    assert!(
        store.drop_index_if_unchanged(observed).is_err(),
        "re-adding the same payload must not resurrect a stale observation"
    );
    assert!(store.validate_index_registration(&validation).is_err());
    assert!(store.validate_index_registration(&replacement).is_ok());
    assert_eq!(section.serialize().unwrap(), before);
    assert_eq!(
        store.get_vector_index("Item", "embedding").unwrap().len(),
        1
    );
}

#[cfg(feature = "text-index")]
#[test]
fn same_text_forwarding_handle_reregistration_is_not_the_observed_registration() {
    let store = LpgStore::new().unwrap();
    let retained = super::tests::text();
    store.add_text_index("Doc", "body", Arc::clone(&retained));
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "body", Value::from("original searchable document"));
    let observed = store.observe_text_index("Doc", "body").unwrap();
    let validation = store.observe_text_index("Doc", "body").unwrap();
    assert!(store.remove_text_index("Doc", "body"));
    store.add_text_index("Doc", "body", Arc::clone(&retained));
    let replacement = store.observe_text_index("Doc", "body").unwrap();

    assert!(
        store.drop_index_if_unchanged(observed).is_err(),
        "the same forwarded concrete target must receive a fresh registration"
    );
    assert!(store.validate_index_registration(&validation).is_err());
    assert!(store.validate_index_registration(&replacement).is_ok());
    assert_eq!(
        store
            .get_text_index("Doc", "body")
            .unwrap()
            .read()
            .search("original", 10)[0]
            .0,
        node
    );
}
