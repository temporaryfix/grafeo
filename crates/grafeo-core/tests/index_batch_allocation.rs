//! Counts the actual allocator traffic of the public install seam only.
#![cfg(feature = "lpg")]

#[test]
fn index_batch_install_and_guard_release_have_zero_allocator_traffic() {
    for install in [true, false] {
        let parent = LpgStore::new().unwrap();
        let child = parent.graph_or_create("child").unwrap();
        let mut workspace = IndexRegistryWorkspace::new(vec![
            StoreIndexEdits {
                store: &child,
                edits: vec![IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Property(PropertyKey::new("child")),
                    contents: IndexRegistryContents::Property(vec![(
                        NodeId::new(1),
                        Value::Int64(1),
                    )]),
                }],
            },
            StoreIndexEdits {
                store: &parent,
                edits: vec![IndexRegistryEdit::Create {
                    key: IndexRegistryKey::Property(PropertyKey::new("parent")),
                    contents: IndexRegistryContents::Property(vec![(
                        NodeId::new(2),
                        Value::Int64(2),
                    )]),
                }],
            },
        ]);
        let prepared = prepare_index_registry_batch(&mut workspace).unwrap();
        allocation::start();
        if install {
            drop(std::hint::black_box(prepared).install());
        } else {
            drop(std::hint::black_box(prepared));
        }
        let observed = allocation::stop();
        assert_eq!(observed, allocation::Counts::default(), "install={install}");
        assert_eq!(parent.has_property_index("parent"), install);
        assert_eq!(child.has_property_index("child"), install);
    }
}

#[path = "support/allocation.rs"]
mod allocation;

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::{
    IndexRegistryContents, IndexRegistryEdit, IndexRegistryKey, IndexRegistryMaintenance,
    IndexRegistryWorkspace, LpgStore, StoreIndexEdits, prepare_index_registry_batch,
};

#[test]
fn index_batch_install_has_zero_allocator_traffic() {
    allocation::start();
    let mut control = Vec::<u8>::with_capacity(17);
    control.extend_from_slice(&[1; 17]);
    control.reserve(1024);
    std::hint::black_box(&control);
    let zeroed = vec![0_u8; std::hint::black_box(8192)];
    std::hint::black_box(&zeroed);
    drop(control);
    drop(zeroed);
    let positive = allocation::stop();
    assert!(positive.alloc > 0);
    assert!(positive.zeroed > 0);
    assert!(positive.realloc > 0);
    assert!(positive.dealloc > 0);

    let a = LpgStore::new().unwrap();
    let b = LpgStore::new().unwrap();
    a.create_property_index("retired");
    let removed = a.observe_property_index("retired").unwrap();
    #[allow(unused_mut)]
    let mut edits = vec![
        IndexRegistryEdit::Drop { expected: removed },
        IndexRegistryEdit::Create {
            key: IndexRegistryKey::Property(PropertyKey::new("first")),
            contents: IndexRegistryContents::Property(vec![(NodeId::new(1), Value::Int64(42))]),
        },
    ];
    #[cfg(feature = "text-index")]
    {
        let mut text = grafeo_core::index::text::InvertedIndex::new(
            grafeo_core::index::text::BM25Config::default(),
        );
        text.insert(NodeId::new(2), "prebuilt text");
        edits.push(IndexRegistryEdit::Create {
            key: IndexRegistryKey::Text {
                label: "Doc".into(),
                property: "body".into(),
            },
            contents: IndexRegistryContents::Text(text),
        });
    }
    #[cfg(feature = "vector-index")]
    {
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};
        let vector = VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            2,
            DistanceMetric::Euclidean,
        )));
        vector.insert(NodeId::new(3), &[0.0, 1.0], &|_| None);
        edits.push(IndexRegistryEdit::Create {
            key: IndexRegistryKey::Vector {
                label: "Doc".into(),
                property: "embedding".into(),
            },
            contents: IndexRegistryContents::Vector(vector),
        });
    }
    let mut workspace = IndexRegistryWorkspace::new(vec![
        StoreIndexEdits { store: &a, edits },
        StoreIndexEdits {
            store: &b,
            edits: vec![IndexRegistryEdit::Create {
                key: IndexRegistryKey::Property(PropertyKey::new("second")),
                contents: IndexRegistryContents::Property(vec![(NodeId::new(4), Value::Int64(43))]),
            }],
        },
    ]);
    let prepared = prepare_index_registry_batch(&mut workspace).unwrap();
    allocation::start();
    let installed = std::hint::black_box(prepared).install();
    let observed = allocation::stop();
    drop(installed);
    assert_eq!(observed, allocation::Counts::default());
    assert!(!a.has_property_index("retired"));
    assert_eq!(
        a.find_nodes_by_property("first", &Value::Int64(42)),
        vec![NodeId::new(1)]
    );
    assert_eq!(
        b.find_nodes_by_property("second", &Value::Int64(43)),
        vec![NodeId::new(4)]
    );
}

#[test]
fn surviving_property_and_text_install_has_zero_allocator_traffic() {
    use std::{collections::HashMap, sync::Arc};
    let store = LpgStore::new().unwrap();
    let node = store.create_node(&["Doc"]);
    let old = Value::GCounter(Arc::new(HashMap::from([("replica-a".to_owned(), 1)])));
    let new = Value::GCounter(Arc::new(HashMap::from([("replica-a".to_owned(), 2)])));
    store.set_node_property(node, "counter", old.clone());
    store.create_property_index("counter");
    let before = store.observe_property_index("counter").unwrap();
    let edits = vec![IndexRegistryEdit::Maintain {
        expected: store.observe_property_index("counter").unwrap(),
        changes: IndexRegistryMaintenance::Property(vec![(
            node,
            Some(old.clone()),
            Some(new.clone()),
        )]),
    }];
    #[cfg(feature = "text-index")]
    let (edits, text_before) = {
        let mut edits = edits;
        use grafeo_common::types::{EpochId, TransactionId};
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        let mut text = InvertedIndex::new(BM25Config::default());
        text.insert(node, "oldtoken");
        store.add_text_index("Doc", "body", Arc::new(parking_lot::RwLock::new(text)));
        let before = store.observe_text_index("Doc", "body").unwrap();
        edits.push(IndexRegistryEdit::Maintain {
            expected: store.observe_text_index("Doc", "body").unwrap(),
            changes: IndexRegistryMaintenance::Text {
                rows: vec![(node, Some("finaltoken".into()))],
                frontier: EpochId::INITIAL,
                commit_epoch: EpochId::new(1),
                transaction_id: TransactionId::new(930),
            },
        });
        (edits, before)
    };
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &store,
        edits,
    }]);
    let prepared = prepare_index_registry_batch(&mut workspace).unwrap();
    allocation::start();
    let installed = prepared.install();
    let observed = allocation::stop();
    drop(installed);
    assert_eq!(observed, allocation::Counts::default());
    store.validate_index_registration(&before).unwrap();
    assert_eq!(store.find_nodes_by_property("counter", &new), vec![node]);
    assert!(store.find_nodes_by_property("counter", &old).is_empty());
    #[cfg(feature = "text-index")]
    {
        store.validate_index_registration(&text_before).unwrap();
        let text = store.get_text_index("Doc", "body").unwrap();
        assert_eq!(text.read().search("finaltoken", 10)[0].0, node);
        assert!(text.read().search("oldtoken", 10).is_empty());
    }
}
