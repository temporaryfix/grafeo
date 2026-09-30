use super::*;
use grafeo_common::types::Value;

fn indexed_store() -> Arc<LpgStore> {
    let store = Arc::new(LpgStore::new().unwrap());
    let node = store.create_node(&["Doc"]);
    store.set_node_property(node, "code", Value::Int64(7));
    store.create_property_index("code");
    #[cfg(feature = "vector-index")]
    store.add_vector_index("Doc", "embedding", super::tests::vector());
    #[cfg(feature = "text-index")]
    {
        store.add_text_index("Doc", "body", super::tests::text());
        store.set_node_property(node, "body", Value::from("original document"));
    }
    store
}

fn observations(store: &LpgStore) -> Vec<IndexRegistrationObservation> {
    Vec::from([
        store.observe_property_index("code").unwrap(),
        #[cfg(feature = "vector-index")]
        store.observe_vector_index("Doc", "embedding").unwrap(),
        #[cfg(feature = "text-index")]
        store.observe_text_index("Doc", "body").unwrap(),
    ])
}

#[test]
fn same_incarnation_transfer_preserves_tags_but_requires_new_physical_observations() {
    let source = indexed_store();
    let old = observations(&source);
    let topology = source.pin_named_graph_topology();
    let transition = source.pin_exclusive_unframed_transition().unwrap();
    let target = Arc::new(
        transition
            .prepare_same_incarnation_empty_successor()
            .unwrap(),
    );
    let prepared = transition
        .prepare_same_incarnation_representation_transfer(
            &topology,
            Arc::clone(&source),
            Arc::clone(&target),
        )
        .unwrap();
    prepared.validate_unpublished_target().unwrap();
    prepared.publish().commit();
    drop(transition);
    drop(topology);

    let fresh = observations(&target);
    let frozen = observations(&source);
    assert_eq!(old.len(), fresh.len());
    for ((old, fresh), frozen) in old.into_iter().zip(fresh).zip(frozen) {
        assert!(Arc::ptr_eq(&old.registration, &fresh.registration));
        assert!(Arc::ptr_eq(&old.registration, &frozen.registration));
        assert!(Arc::ptr_eq(&old.incarnation, &fresh.incarnation));
        assert!(!Arc::ptr_eq(&old.physical, &fresh.physical));
        assert!(source.validate_index_registration(&old).is_err());
        assert!(source.drop_index_if_unchanged(frozen).is_err());
        assert!(target.validate_index_registration(&old).is_err());
        assert!(target.drop_index_if_unchanged(old).is_err());
        target.validate_index_registration(&fresh).unwrap();
        target.drop_index_if_unchanged(fresh).unwrap();
    }
    assert!(source.has_property_index("code"));
    assert!(!target.has_property_index("code"));
}

#[test]
fn rollback_restores_original_registration_and_retires_successor_observations() {
    let source = indexed_store();
    let old = observations(&source);
    let topology = source.pin_named_graph_topology();
    let transition = source.pin_exclusive_unframed_transition().unwrap();
    let target = Arc::new(
        transition
            .prepare_same_incarnation_empty_successor()
            .unwrap(),
    );
    let prepared = transition
        .prepare_same_incarnation_representation_transfer(
            &topology,
            Arc::clone(&source),
            Arc::clone(&target),
        )
        .unwrap();
    prepared.validate_unpublished_target().unwrap();
    let published = prepared.publish();
    let successor = observations(&target);
    published.rollback();
    drop(transition);
    drop(topology);

    for (old, successor) in old.into_iter().zip(successor) {
        assert!(Arc::ptr_eq(&old.registration, &successor.registration));
        source.validate_index_registration(&old).unwrap();
        assert!(source.validate_index_registration(&successor).is_err());
        assert!(source.drop_index_if_unchanged(successor).is_err());
        source.validate_index_registration(&old).unwrap();
        source.drop_index_if_unchanged(old).unwrap();
    }
    let retired = observations(&target);
    for observed in retired {
        assert!(target.validate_index_registration(&observed).is_err());
        assert!(target.drop_index_if_unchanged(observed).is_err());
    }
    assert_eq!(
        source
            .find_nodes_by_property("code", &Value::Int64(7))
            .len(),
        1
    );
}
