use super::*;
use crate::graph::Direction;
use crate::graph::lpg::property::CompareOp;
use grafeo_common::types::TransactionId;

#[test]
fn test_create_node() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person"]);
    assert!(id.is_valid());

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(!node.has_label("Animal"));
}

#[test]
fn test_create_node_with_props() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("age", Value::from(30i64))],
    );

    let node = store.get_node(id).unwrap();
    assert_eq!(
        node.get_property("name").and_then(|v| v.as_str()),
        Some("Alix")
    );
    assert_eq!(
        node.get_property("age").and_then(|v| v.as_int64()),
        Some(30)
    );
}

#[test]
fn test_delete_node() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person"]);
    assert_eq!(store.node_count(), 1);

    assert!(store.delete_node(id));
    assert_eq!(store.node_count(), 0);
    assert!(store.get_node(id).is_none());

    // Double delete should return false
    assert!(!store.delete_node(id));
}

#[test]
fn test_create_edge() {
    let store = LpgStore::new().unwrap();

    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);

    let edge_id = store.create_edge(alix, gus, "KNOWS");
    assert!(edge_id.is_valid());

    let edge = store.get_edge(edge_id).unwrap();
    assert_eq!(edge.src, alix);
    assert_eq!(edge.dst, gus);
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
}

#[test]
fn test_neighbors() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");

    let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
    assert_eq!(outgoing.len(), 2);
    assert!(outgoing.contains(&b));
    assert!(outgoing.contains(&c));

    let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 1);
    assert!(incoming.contains(&a));
}

#[test]
fn test_nodes_by_label() {
    let store = LpgStore::new().unwrap();

    let p1 = store.create_node(&["Person"]);
    let p2 = store.create_node(&["Person"]);
    let _a = store.create_node(&["Animal"]);

    let persons = store.nodes_by_label("Person");
    assert_eq!(persons.len(), 2);
    assert!(persons.contains(&p1));
    assert!(persons.contains(&p2));

    let animals = store.nodes_by_label("Animal");
    assert_eq!(animals.len(), 1);
}

#[test]
fn test_delete_edge() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    assert_eq!(store.edge_count(), 1);

    assert!(store.delete_edge(edge_id));
    assert_eq!(store.edge_count(), 0);
    assert!(store.get_edge(edge_id).is_none());
}

// === New tests for improved coverage ===

#[test]
fn test_lpg_store_config() {
    // Test with_config
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 100,
        initial_edge_capacity: 200,
    };
    let store = LpgStore::with_config(config).unwrap();

    // Store should work but without backward adjacency
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    // Outgoing should work
    let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
    assert_eq!(outgoing.len(), 1);

    // Incoming should be empty (no backward adjacency)
    let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 0);
}

#[test]
fn test_epoch_management() {
    let store = LpgStore::new().unwrap();

    let epoch0 = store.current_epoch();
    assert_eq!(epoch0.as_u64(), 0);

    let epoch1 = store.new_epoch();
    assert_eq!(epoch1.as_u64(), 1);

    let current = store.current_epoch();
    assert_eq!(current.as_u64(), 1);
}

#[test]
fn test_node_properties() {
    let store = LpgStore::new().unwrap();
    let id = store.create_node(&["Person"]);

    // Set and get property
    store.set_node_property(id, "name", Value::from("Alix"));
    let name = store.get_node_property(id, &"name".into());
    assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Alix"));

    // Update property
    store.set_node_property(id, "name", Value::from("Gus"));
    let name = store.get_node_property(id, &"name".into());
    assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Gus"));

    // Remove property
    let old = store.remove_node_property(id, "name");
    assert!(matches!(old, Some(Value::String(s)) if s.as_str() == "Gus"));

    // Property should be gone
    let name = store.get_node_property(id, &"name".into());
    assert!(name.is_none());

    // Remove non-existent property
    let none = store.remove_node_property(id, "nonexistent");
    assert!(none.is_none());
}

#[test]
fn test_edge_properties() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    // Set and get property
    store.set_edge_property(edge_id, "since", Value::from(2020i64));
    let since = store.get_edge_property(edge_id, &"since".into());
    assert_eq!(since.and_then(|v| v.as_int64()), Some(2020));

    // Remove property
    let old = store.remove_edge_property(edge_id, "since");
    assert_eq!(old.and_then(|v| v.as_int64()), Some(2020));

    let since = store.get_edge_property(edge_id, &"since".into());
    assert!(since.is_none());
}

#[test]
fn test_add_remove_label() {
    let store = LpgStore::new().unwrap();
    let id = store.create_node(&["Person"]);

    // Add new label
    assert!(store.add_label(id, "Employee"));

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));

    // Adding same label again should fail
    assert!(!store.add_label(id, "Employee"));

    // Remove label
    assert!(store.remove_label(id, "Employee"));

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(!node.has_label("Employee"));

    // Removing non-existent label should fail
    assert!(!store.remove_label(id, "Employee"));
    assert!(!store.remove_label(id, "NonExistent"));
}

#[test]
fn test_add_label_to_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.add_label(fake_id, "Label"));
}

#[test]
fn test_remove_label_from_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.remove_label(fake_id, "Label"));
}

#[test]
fn test_node_ids() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    let ids = store.node_ids();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&n1));
    assert!(ids.contains(&n2));
    assert!(ids.contains(&n3));

    // Delete one
    store.delete_node(n2);
    let ids = store.node_ids();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&n2));
}

#[test]
fn test_delete_node_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.delete_node(fake_id));
}

#[test]
fn test_delete_edge_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = EdgeId::new(999);
    assert!(!store.delete_edge(fake_id));
}

#[test]
fn test_delete_edge_double() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    assert!(store.delete_edge(edge_id));
    assert!(!store.delete_edge(edge_id)); // Double delete
}

#[test]
fn test_create_edge_with_props() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);

    let edge_id = store.create_edge_with_props(
        a,
        b,
        "KNOWS",
        [
            ("since", Value::from(2020i64)),
            ("weight", Value::from(1.0)),
        ],
    );

    let edge = store.get_edge(edge_id).unwrap();
    assert_eq!(
        edge.get_property("since").and_then(|v| v.as_int64()),
        Some(2020)
    );
    assert_eq!(
        edge.get_property("weight").and_then(|v| v.as_float64()),
        Some(1.0)
    );
}

#[test]
fn test_delete_node_edges() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS"); // a -> b
    store.create_edge(c, a, "KNOWS"); // c -> a

    assert_eq!(store.edge_count(), 2);

    // Delete all edges connected to a
    store.delete_node_edges(a);

    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_delete_node_edges_self_loop() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let _e = store.create_edge(a, a, "SELF"); // self-loop

    assert_eq!(store.edge_count(), 1);

    // Self-loop appears in both outgoing and incoming scans.
    // The fix deduplicates via HashSet, so only one delete happens.
    store.delete_node_edges(a);

    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_delete_node_edges_self_loop_plus_others() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, a, "SELF"); // self-loop on a
    store.create_edge(a, b, "KNOWS"); // outgoing from a
    store.create_edge(c, a, "KNOWS"); // incoming to a
    store.create_edge(b, c, "KNOWS"); // unrelated

    assert_eq!(store.edge_count(), 4);

    store.delete_node_edges(a);

    // Only the b->c edge should remain
    assert_eq!(store.edge_count(), 1);
}

#[test]
fn test_delete_node_edges_atomic_batch() {
    use std::sync::Arc;

    let store = Arc::new(LpgStore::new().unwrap());

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);
    let d = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(d, a, "KNOWS");

    assert_eq!(store.edge_count(), 3);

    // Spawn a reader thread that checks edge count.
    // With batch locking, the reader should never see 1 or 2
    // (partially deleted): it should see either 3 (before) or 0 (after).
    //
    // A barrier ensures both threads start at the same time, and an
    // AtomicBool keeps the reader spinning until deletion finishes,
    // so the two threads are guaranteed to overlap.
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};

    let barrier = Arc::new(Barrier::new(2));
    let done = Arc::new(AtomicBool::new(false));

    let reader = Arc::clone(&store);
    let reader_barrier = Arc::clone(&barrier);
    let reader_done = Arc::clone(&done);

    let handle = std::thread::spawn(move || {
        let mut saw_partial = false;
        reader_barrier.wait();
        while !reader_done.load(Ordering::Acquire) {
            let count = reader.edge_count();
            if count != 0 && count != 3 {
                saw_partial = true;
                break;
            }
        }
        saw_partial
    });

    barrier.wait();
    store.delete_node_edges(a);
    done.store(true, Ordering::Release);

    let saw_partial = handle.join().unwrap();
    assert!(
        !saw_partial,
        "concurrent reader observed partially deleted edges"
    );
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_neighbors_both_directions() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS"); // a -> b
    store.create_edge(c, a, "KNOWS"); // c -> a

    // Direction::Both for node a
    let neighbors: Vec<_> = store.neighbors(a, Direction::Both).collect();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&b)); // outgoing
    assert!(neighbors.contains(&c)); // incoming
}

#[test]
fn test_edges_from() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    let e1 = store.create_edge(a, b, "KNOWS");
    let e2 = store.create_edge(a, c, "KNOWS");

    let edges: Vec<_> = store.edges_from(a, Direction::Outgoing).collect();
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().any(|(_, e)| *e == e1));
    assert!(edges.iter().any(|(_, e)| *e == e2));

    // Incoming edges to b
    let incoming: Vec<_> = store.edges_from(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].1, e1);
}

#[test]
fn test_edges_to() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    let e1 = store.create_edge(a, b, "KNOWS");
    let e2 = store.create_edge(c, b, "KNOWS");

    // Edges pointing TO b
    let to_b = store.edges_to(b);
    assert_eq!(to_b.len(), 2);
    assert!(to_b.iter().any(|(src, e)| *src == a && *e == e1));
    assert!(to_b.iter().any(|(src, e)| *src == c && *e == e2));
}

#[test]
fn test_out_degree_in_degree() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(c, b, "KNOWS");

    assert_eq!(store.out_degree(a), 2);
    assert_eq!(store.out_degree(b), 0);
    assert_eq!(store.out_degree(c), 1);

    assert_eq!(store.in_degree(a), 0);
    assert_eq!(store.in_degree(b), 2);
    assert_eq!(store.in_degree(c), 1);
}

#[test]
fn test_edge_type() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    let edge_type = store.edge_type(edge_id);
    assert_eq!(edge_type.as_deref(), Some("KNOWS"));

    // Non-existent edge
    let fake_id = EdgeId::new(999);
    assert!(store.edge_type(fake_id).is_none());
}

#[test]
fn test_count_methods() {
    let store = LpgStore::new().unwrap();

    assert_eq!(store.label_count(), 0);
    assert_eq!(store.edge_type_count(), 0);
    assert_eq!(store.property_key_count(), 0);

    let a = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let b = store.create_node(&["Company"]);
    store.create_edge_with_props(a, b, "WORKS_AT", [("since", Value::from(2020i64))]);

    assert_eq!(store.label_count(), 2); // Person, Company
    assert_eq!(store.edge_type_count(), 1); // WORKS_AT
    assert_eq!(store.property_key_count(), 2); // age, since
}

#[test]
fn test_all_nodes_and_edges() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    let nodes: Vec<_> = store.all_nodes().collect();
    assert_eq!(nodes.len(), 2);

    let edges: Vec<_> = store.all_edges().collect();
    assert_eq!(edges.len(), 1);
}

#[test]
fn test_all_labels_and_edge_types() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Company"]);
    let a = store.create_node(&["Animal"]);
    let b = store.create_node(&["Animal"]);
    store.create_edge(a, b, "EATS");

    let labels = store.all_labels();
    assert_eq!(labels.len(), 3);
    assert!(labels.contains(&"Person".to_string()));
    assert!(labels.contains(&"Company".to_string()));
    assert!(labels.contains(&"Animal".to_string()));

    let edge_types = store.all_edge_types();
    assert_eq!(edge_types.len(), 1);
    assert!(edge_types.contains(&"EATS".to_string()));
}

#[test]
fn test_all_property_keys() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);
    let b = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    store.create_edge_with_props(a, b, "KNOWS", [("since", Value::from(2020i64))]);

    let keys = store.all_property_keys();
    assert!(keys.contains(&"name".to_string()));
    assert!(keys.contains(&"age".to_string()));
    assert!(keys.contains(&"since".to_string()));
}

#[test]
fn test_nodes_with_label() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Company"]);

    let persons: Vec<_> = store.nodes_with_label("Person").collect();
    assert_eq!(persons.len(), 2);

    let companies: Vec<_> = store.nodes_with_label("Company").collect();
    assert_eq!(companies.len(), 1);

    let none: Vec<_> = store.nodes_with_label("NonExistent").collect();
    assert_eq!(none.len(), 0);
}

#[test]
fn test_edges_with_type() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Company"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "WORKS_AT");

    let knows: Vec<_> = store.edges_with_type("KNOWS").collect();
    assert_eq!(knows.len(), 1);

    let works_at: Vec<_> = store.edges_with_type("WORKS_AT").collect();
    assert_eq!(works_at.len(), 1);

    let none: Vec<_> = store.edges_with_type("NonExistent").collect();
    assert_eq!(none.len(), 0);
}

#[test]
fn test_nodes_by_label_nonexistent() {
    let store = LpgStore::new().unwrap();
    store.create_node(&["Person"]);

    let empty = store.nodes_by_label("NonExistent");
    assert!(empty.is_empty());
}

#[test]
fn test_nodes_by_label_count_matches_vec_len() {
    // nodes_by_label_count is the O(1) fast path used by the planner to
    // bound unbounded VectorScan k; it must agree with the Vec-returning
    // variant for every label, including ones that don't exist.
    let store = LpgStore::new().unwrap();
    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Animal"]);

    for label in ["Person", "Animal", "NonExistent"] {
        assert_eq!(
            store.nodes_by_label_count(label),
            store.nodes_by_label(label).len(),
            "count mismatch for label {label:?}"
        );
    }
}

#[test]
fn test_statistics() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Company"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "WORKS_AT");

    store.compute_statistics();
    let stats = store.statistics();

    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.total_edges, 2);

    // Estimates
    let person_card = store.estimate_label_cardinality("Person");
    assert!(person_card > 0.0);

    let avg_degree = store.estimate_avg_degree("KNOWS", true);
    assert!(avg_degree >= 0.0);
}

#[test]
fn test_zone_maps() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(35i64))]);

    // Zone map should indicate possible matches (30 is within [25, 35] range)
    let might_match =
        store.node_property_might_match(&"age".into(), CompareOp::Eq, &Value::from(30i64));
    // Zone maps return true conservatively when value is within min/max range
    assert!(might_match);

    let zone = store.node_property_zone_map(&"age".into());
    assert!(zone.is_some());

    // Non-existent property
    let no_zone = store.node_property_zone_map(&"nonexistent".into());
    assert!(no_zone.is_none());

    // Edge zone maps
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    store.create_edge_with_props(a, b, "REL", [("weight", Value::from(1.0))]);

    let edge_zone = store.edge_property_zone_map(&"weight".into());
    assert!(edge_zone.is_some());
}

#[test]
fn test_rebuild_zone_maps() {
    let store = LpgStore::new().unwrap();
    store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);

    // Should not panic
    store.rebuild_zone_maps();
}

#[test]
fn test_create_node_with_id() {
    let store = LpgStore::new().unwrap();

    let specific_id = NodeId::new(100);
    store
        .create_node_with_id(specific_id, &["Person", "Employee"])
        .unwrap();

    let node = store.get_node(specific_id).unwrap();
    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));

    // Next auto-generated ID should be > 100
    let next = store.create_node(&["Other"]);
    assert!(next.as_u64() > 100);
}

#[test]
fn test_create_edge_with_id() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);

    let specific_id = EdgeId::new(500);
    store.create_edge_with_id(specific_id, a, b, "REL").unwrap();

    let edge = store.get_edge(specific_id).unwrap();
    assert_eq!(edge.src, a);
    assert_eq!(edge.dst, b);
    assert_eq!(edge.edge_type.as_str(), "REL");

    // Next auto-generated ID should be > 500
    let next = store.create_edge(a, b, "OTHER");
    assert!(next.as_u64() > 500);
}

#[test]
fn test_set_epoch() {
    let store = LpgStore::new().unwrap();

    assert_eq!(store.current_epoch().as_u64(), 0);

    store.set_epoch(EpochId::new(42));
    assert_eq!(store.current_epoch().as_u64(), 42);
}

#[test]
fn test_get_node_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(store.get_node(fake_id).is_none());
}

#[test]
fn test_get_edge_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = EdgeId::new(999);
    assert!(store.get_edge(fake_id).is_none());
}

#[test]
fn test_multiple_labels() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person", "Employee", "Manager"]);
    let node = store.get_node(id).unwrap();

    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));
    assert!(node.has_label("Manager"));
    assert!(!node.has_label("Other"));
}

#[test]
fn test_new_store_is_empty() {
    let store = LpgStore::new().unwrap();
    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_edges_from_both_directions() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let c = store.create_node(&["C"]);

    let e1 = store.create_edge(a, b, "R1"); // a -> b
    let e2 = store.create_edge(c, a, "R2"); // c -> a

    // Both directions from a
    let edges: Vec<_> = store.edges_from(a, Direction::Both).collect();
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().any(|(_, e)| *e == e1)); // outgoing
    assert!(edges.iter().any(|(_, e)| *e == e2)); // incoming
}

#[test]
fn test_no_backward_adj_in_degree() {
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 10,
        initial_edge_capacity: 10,
    };
    let store = LpgStore::with_config(config).unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    store.create_edge(a, b, "R");

    // in_degree should still work (falls back to scanning)
    let degree = store.in_degree(b);
    assert_eq!(degree, 1);
}

#[test]
fn test_no_backward_adj_edges_to() {
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 10,
        initial_edge_capacity: 10,
    };
    let store = LpgStore::with_config(config).unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let e = store.create_edge(a, b, "R");

    // edges_to should still work (falls back to scanning)
    let edges = store.edges_to(b);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].1, e);
}

#[test]
fn test_node_versioned_creation() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let id = store.create_node_versioned(&["Person"], epoch, transaction_id);
    assert!(store.get_node(id).is_some());
}

#[test]
fn test_edge_versioned_creation() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let edge_id = store.create_edge_versioned(a, b, "REL", epoch, transaction_id);
    assert!(store.get_edge(edge_id).is_some());
}

#[test]
fn test_node_with_props_versioned() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let id = store.create_node_with_props_versioned(
        &["Person"],
        [("name", Value::from("Alix"))],
        epoch,
        transaction_id,
    );

    let node = store.get_node(id).unwrap();
    assert_eq!(
        node.get_property("name").and_then(|v| v.as_str()),
        Some("Alix")
    );
}

#[test]
fn test_discard_uncommitted_versions() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(42);

    // Create node with specific tx (uses PENDING epoch, invisible to get_node)
    let node_id = store.create_node_versioned(&["Person"], epoch, transaction_id);
    // Verify the node exists via versioned lookup (own tx can see its PENDING writes)
    assert!(
        store
            .get_node_versioned(node_id, epoch, transaction_id)
            .is_some(),
        "Node should be visible to its own transaction"
    );

    // Discard uncommitted versions for this tx
    store.discard_uncommitted_versions(transaction_id);

    // Node should be gone (version chain was removed)
    assert!(
        store
            .get_node_versioned(node_id, epoch, transaction_id)
            .is_none(),
        "Node should be gone after discard"
    );
}

// === Property Index Tests ===

#[test]
fn test_property_index_create_and_lookup() {
    let store = LpgStore::new().unwrap();

    // Create nodes with properties
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);

    store.set_node_property(alix, "city", Value::from("NYC"));
    store.set_node_property(gus, "city", Value::from("NYC"));
    store.set_node_property(vincent, "city", Value::from("LA"));

    // Before indexing, lookup still works (via scan)
    let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    assert_eq!(nyc_people.len(), 2);

    // Create index
    store.create_property_index("city");
    assert!(store.has_property_index("city"));

    // Indexed lookup should return same results
    let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    assert_eq!(nyc_people.len(), 2);
    assert!(nyc_people.contains(&alix));
    assert!(nyc_people.contains(&gus));

    let la_people = store.find_nodes_by_property("city", &Value::from("LA"));
    assert_eq!(la_people.len(), 1);
    assert!(la_people.contains(&vincent));
}

#[test]
fn test_property_index_maintained_on_update() {
    let store = LpgStore::new().unwrap();

    // Create index first
    store.create_property_index("status");

    let node = store.create_node(&["Task"]);
    store.set_node_property(node, "status", Value::from("pending"));

    // Should find by initial value
    let pending = store.find_nodes_by_property("status", &Value::from("pending"));
    assert_eq!(pending.len(), 1);
    assert!(pending.contains(&node));

    // Update the property
    store.set_node_property(node, "status", Value::from("done"));

    // Old value should not find it
    let pending = store.find_nodes_by_property("status", &Value::from("pending"));
    assert!(pending.is_empty());

    // New value should find it
    let done = store.find_nodes_by_property("status", &Value::from("done"));
    assert_eq!(done.len(), 1);
    assert!(done.contains(&node));
}

#[test]
fn test_property_index_maintained_on_remove() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("tag");

    let node = store.create_node(&["Item"]);
    store.set_node_property(node, "tag", Value::from("important"));

    // Should find it
    let found = store.find_nodes_by_property("tag", &Value::from("important"));
    assert_eq!(found.len(), 1);

    // Remove the property
    store.remove_node_property(node, "tag");

    // Should no longer find it
    let found = store.find_nodes_by_property("tag", &Value::from("important"));
    assert!(found.is_empty());
}

#[test]
fn test_property_index_drop() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("key");
    assert!(store.has_property_index("key"));

    assert!(store.drop_property_index("key"));
    assert!(!store.has_property_index("key"));

    // Dropping non-existent index returns false
    assert!(!store.drop_property_index("key"));
}

#[test]
fn test_property_index_multiple_values() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("age");

    // Create multiple nodes with same and different ages
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);
    let n4 = store.create_node(&["Person"]);

    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "age", Value::from(25i64));
    store.set_node_property(n3, "age", Value::from(30i64));
    store.set_node_property(n4, "age", Value::from(25i64));

    let age_25 = store.find_nodes_by_property("age", &Value::from(25i64));
    assert_eq!(age_25.len(), 3);

    let age_30 = store.find_nodes_by_property("age", &Value::from(30i64));
    assert_eq!(age_30.len(), 1);

    let age_40 = store.find_nodes_by_property("age", &Value::from(40i64));
    assert!(age_40.is_empty());
}

#[test]
fn test_property_index_builds_from_existing_data() {
    let store = LpgStore::new().unwrap();

    // Create nodes first
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.set_node_property(n1, "email", Value::from("alix@example.com"));
    store.set_node_property(n2, "email", Value::from("gus@example.com"));

    // Create index after data exists
    store.create_property_index("email");

    // Index should include existing data
    let alix = store.find_nodes_by_property("email", &Value::from("alix@example.com"));
    assert_eq!(alix.len(), 1);
    assert!(alix.contains(&n1));

    let gus = store.find_nodes_by_property("email", &Value::from("gus@example.com"));
    assert_eq!(gus.len(), 1);
    assert!(gus.contains(&n2));
}

#[test]
fn test_get_node_property_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "age", Value::from(30i64));
    // n3 has no age property

    let age_key = PropertyKey::new("age");
    let values = store.get_node_property_batch(&[n1, n2, n3], &age_key);

    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some(Value::from(25i64)));
    assert_eq!(values[1], Some(Value::from(30i64)));
    assert_eq!(values[2], None);
}

#[test]
fn test_get_node_property_batch_empty() {
    let store = LpgStore::new().unwrap();
    let key = PropertyKey::new("any");

    let values = store.get_node_property_batch(&[], &key);
    assert!(values.is_empty());
}

#[test]
fn test_get_nodes_properties_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    store.set_node_property(n1, "name", Value::from("Alix"));
    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "name", Value::from("Gus"));
    // n3 has no properties

    let all_props = store.get_nodes_properties_batch(&[n1, n2, n3]);

    assert_eq!(all_props.len(), 3);
    assert_eq!(all_props[0].len(), 2); // name and age
    assert_eq!(all_props[1].len(), 1); // name only
    assert_eq!(all_props[2].len(), 0); // no properties

    assert_eq!(
        all_props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
    assert_eq!(
        all_props[1].get(&PropertyKey::new("name")),
        Some(&Value::from("Gus"))
    );
}

#[test]
fn test_get_nodes_properties_batch_empty() {
    let store = LpgStore::new().unwrap();

    let all_props = store.get_nodes_properties_batch(&[]);
    assert!(all_props.is_empty());
}

#[test]
fn test_get_nodes_properties_selective_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);

    // Set multiple properties
    store.set_node_property(n1, "name", Value::from("Alix"));
    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n1, "email", Value::from("alix@example.com"));
    store.set_node_property(n2, "name", Value::from("Gus"));
    store.set_node_property(n2, "age", Value::from(30i64));
    store.set_node_property(n2, "city", Value::from("NYC"));

    // Request only name and age (not email or city)
    let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    let props = store.get_nodes_properties_selective_batch(&[n1, n2], &keys);

    assert_eq!(props.len(), 2);

    // n1: should have name and age, but NOT email
    assert_eq!(props[0].len(), 2);
    assert_eq!(
        props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
    assert_eq!(
        props[0].get(&PropertyKey::new("age")),
        Some(&Value::from(25i64))
    );
    assert_eq!(props[0].get(&PropertyKey::new("email")), None);

    // n2: should have name and age, but NOT city
    assert_eq!(props[1].len(), 2);
    assert_eq!(
        props[1].get(&PropertyKey::new("name")),
        Some(&Value::from("Gus"))
    );
    assert_eq!(
        props[1].get(&PropertyKey::new("age")),
        Some(&Value::from(30i64))
    );
    assert_eq!(props[1].get(&PropertyKey::new("city")), None);
}

#[test]
fn test_get_nodes_properties_selective_batch_empty_keys() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", Value::from("Alix"));

    // Request no properties
    let props = store.get_nodes_properties_selective_batch(&[n1], &[]);

    assert_eq!(props.len(), 1);
    assert!(props[0].is_empty()); // Empty map when no keys requested
}

#[test]
fn test_get_nodes_properties_selective_batch_missing_keys() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", Value::from("Alix"));

    // Request a property that doesn't exist
    let keys = vec![PropertyKey::new("nonexistent"), PropertyKey::new("name")];
    let props = store.get_nodes_properties_selective_batch(&[n1], &keys);

    assert_eq!(props.len(), 1);
    assert_eq!(props[0].len(), 1); // Only name exists
    assert_eq!(
        props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
}

// === Range Query Tests ===

#[test]
fn test_find_nodes_in_range_inclusive() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
    let _n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

    // age >= 20 AND age <= 40 (inclusive both sides)
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(20i64)),
        Some(&Value::from(40i64)),
        true,
        true,
    );
    assert_eq!(result.len(), 3);
    assert!(result.contains(&n1));
    assert!(result.contains(&n2));
    assert!(result.contains(&n3));
}

#[test]
fn test_find_nodes_in_range_exclusive() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);

    // age > 20 AND age < 40 (exclusive both sides)
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(20i64)),
        Some(&Value::from(40i64)),
        false,
        false,
    );
    assert_eq!(result.len(), 1);
    assert!(result.contains(&n2));
}

#[test]
fn test_find_nodes_in_range_open_ended() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
    let n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

    // age >= 35 (no upper bound)
    let result = store.find_nodes_in_range("age", Some(&Value::from(35i64)), None, true, true);
    assert_eq!(result.len(), 2);
    assert!(result.contains(&n3));
    assert!(result.contains(&n4));

    // age <= 25 (no lower bound)
    let result = store.find_nodes_in_range("age", None, Some(&Value::from(25i64)), true, true);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_nodes_in_range_empty_result() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

    // Range that doesn't match anything
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(100i64)),
        Some(&Value::from(200i64)),
        true,
        true,
    );
    assert!(result.is_empty());
}

#[test]
fn test_find_nodes_in_range_nonexistent_property() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

    let result = store.find_nodes_in_range(
        "weight",
        Some(&Value::from(50i64)),
        Some(&Value::from(100i64)),
        true,
        true,
    );
    assert!(result.is_empty());
}

// === Multi-Property Query Tests ===

#[test]
fn test_find_nodes_by_properties_multiple_conditions() {
    let store = LpgStore::new().unwrap();

    let alix = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("city", Value::from("NYC"))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Gus")), ("city", Value::from("NYC"))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("city", Value::from("LA"))],
    );

    // Match name="Alix" AND city="NYC"
    let result = store
        .find_nodes_by_properties(&[("name", Value::from("Alix")), ("city", Value::from("NYC"))]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&alix));
}

#[test]
fn test_find_nodes_by_properties_empty_conditions() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);

    // Empty conditions should return all nodes
    let result = store.find_nodes_by_properties(&[]);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_find_nodes_by_properties_no_match() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);

    let result = store.find_nodes_by_properties(&[("name", Value::from("Nobody"))]);
    assert!(result.is_empty());
}

#[test]
fn test_find_nodes_by_properties_with_index() {
    let store = LpgStore::new().unwrap();

    // Create index on name
    store.create_property_index("name");

    let alix = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("age", Value::from(30i64))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Gus")), ("age", Value::from(30i64))],
    );

    // Index should accelerate the lookup
    let result = store
        .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::from(30i64))]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&alix));
}

// === Cardinality Estimation Tests ===

#[test]
fn test_estimate_label_cardinality() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Animal"]);

    store.ensure_statistics_fresh();

    let person_est = store.estimate_label_cardinality("Person");
    let animal_est = store.estimate_label_cardinality("Animal");
    let unknown_est = store.estimate_label_cardinality("Unknown");

    assert!(
        person_est >= 1.0,
        "Person should have cardinality >= 1, got {person_est}"
    );
    assert!(
        animal_est >= 1.0,
        "Animal should have cardinality >= 1, got {animal_est}"
    );
    // Unknown label should return some default (not panic)
    assert!(unknown_est >= 0.0);
}

#[test]
fn test_estimate_avg_degree() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(b, c, "KNOWS");

    store.ensure_statistics_fresh();

    let outgoing = store.estimate_avg_degree("KNOWS", true);
    let incoming = store.estimate_avg_degree("KNOWS", false);

    assert!(
        outgoing > 0.0,
        "Outgoing degree should be > 0, got {outgoing}"
    );
    assert!(
        incoming > 0.0,
        "Incoming degree should be > 0, got {incoming}"
    );
}

// === Delete operations ===

#[test]
fn test_delete_node_does_not_cascade() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let e = store.create_edge(a, b, "KNOWS");

    assert!(store.delete_node(a));
    assert!(store.get_node(a).is_none());

    // Edges are NOT automatically deleted (non-detach delete)
    assert!(
        store.get_edge(e).is_some(),
        "Edge should survive non-detach node delete"
    );
}

#[test]
fn test_delete_already_deleted_node() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);

    assert!(store.delete_node(a));
    // Second delete should return false (already deleted)
    assert!(!store.delete_node(a));
}

#[test]
fn test_delete_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    assert!(!store.delete_node(NodeId::new(999)));
}

// === GraphStore / GraphStoreMut Trait Compliance ===

/// Verifies that LpgStore's trait implementations are object-safe and
/// produce identical results to the concrete methods.
mod graph_store_traits {
    use super::*;
    use crate::graph::Direction;
    use crate::graph::traits::{GraphStore, GraphStoreMut};

    #[test]
    fn trait_object_safety() {
        // Must compile: Arc<dyn GraphStoreMut> proves object safety
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let _read: &dyn GraphStore = &*store;
    }

    #[test]
    fn trait_round_trip() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        // Create nodes via trait
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person", "Developer"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::from(30i64));
        store.set_node_property(gus, "name", Value::from("Gus"));

        // Create edge via trait
        let edge = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(edge, "since", Value::from(2020i64));

        // Read back via GraphStore trait
        let read: &dyn GraphStore = store;

        // Point lookups
        let alice_node = read.get_node(alix).expect("alix should exist");
        assert!(alice_node.labels.contains(&arcstr::literal!("Person")));

        let edge_data = read.get_edge(edge).expect("edge should exist");
        assert_eq!(edge_data.src, alix);
        assert_eq!(edge_data.dst, gus);

        // Properties
        assert_eq!(
            read.get_node_property(alix, &PropertyKey::new("name")),
            Some(Value::from("Alix"))
        );
        assert_eq!(
            read.get_edge_property(edge, &PropertyKey::new("since")),
            Some(Value::from(2020i64))
        );

        // Traversal
        let neighbors = read.neighbors(alix, Direction::Outgoing);
        assert_eq!(neighbors, vec![gus]);

        let edges = read.edges_from(alix, Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (gus, edge));

        assert_eq!(read.out_degree(alix), 1);
        assert_eq!(read.in_degree(gus), 1);

        // Scans
        assert_eq!(read.node_count(), 2);
        assert_eq!(read.edge_count(), 1);
        assert_eq!(read.nodes_by_label("Person").len(), 2);
        assert_eq!(read.node_ids().len(), 2);

        // Edge type
        assert_eq!(read.edge_type(edge), Some(arcstr::literal!("KNOWS")));

        // Search
        let found = read.find_nodes_by_property("name", &Value::from("Alix"));
        assert_eq!(found, vec![alix]);
    }

    #[test]
    fn trait_mutation_operations() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        let node = store.create_node(&["A"]);

        // Label mutation
        assert!(store.add_label(node, "B"));
        assert!(store.remove_label(node, "B"));

        // Property mutation
        store.set_node_property(node, "key", Value::from("val"));
        let removed = store.remove_node_property(node, "key");
        assert_eq!(removed, Some(Value::from("val")));

        // Deletion
        assert!(store.delete_node(node));
        assert!(store.get_node(node).is_none());
    }

    #[test]
    fn trait_batch_edges() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);

        let ids = store.batch_create_edges(&[(a, b, "E"), (a, c, "E"), (b, c, "E")]);
        assert_eq!(ids.len(), 3);
        assert_eq!(store.edge_count(), 3);
    }
}

#[test]
fn test_clear() {
    let store = LpgStore::new().unwrap();
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", "Alix".into());
    let _e = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(_e, "since", 2024.into());

    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);

    store.clear();

    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);

    // Should be able to add new data after clear
    let n3 = store.create_node(&["Animal"]);
    assert_eq!(store.node_count(), 1);
    assert!(store.get_node(n3).is_some());
}

#[test]
fn copy_graph_deep_copies_nodes_edges_props_labels_and_index() {
    let store = LpgStore::new().expect("arena");
    let src = store.graph_or_create("src").expect("src graph");
    let a = src.create_node_with_props(
        &["Person"],
        [("name", Value::from("Ann")), ("age", Value::from(30i64))],
    );
    let b = src.create_node_with_props(&["Person"], [("name", Value::from("Bob"))]);
    src.create_edge_with_props(a, b, "KNOWS", [("since", Value::from(2020i64))]);
    src.create_property_index("name");

    store.copy_graph(Some("src"), Some("dst")).expect("copy");
    let dst = store.graph("dst").expect("dst graph created");

    // Counts, labels, props.
    assert_eq!(dst.node_count(), 2);
    assert_eq!(dst.edge_count(), 1);
    let ann_ids = dst.find_nodes_by_property("name", &Value::from("Ann"));
    assert_eq!(ann_ids.len(), 1, "property index must work on the copy");
    let ann = dst.get_node(ann_ids[0]).unwrap();
    assert!(ann.labels.iter().any(|l| l.as_str() == "Person"));
    assert_eq!(
        ann.properties.get(&PropertyKey::new("age")),
        Some(&Value::Int64(30))
    );

    // Edge: type, remapped endpoints, and property carried.
    let edges: Vec<_> = dst.all_edges().collect();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_type.as_str(), "KNOWS");
    assert_eq!(
        edges[0].properties.get(&PropertyKey::new("since")),
        Some(&Value::Int64(2020))
    );

    // Deep copy: mutating the copy does not affect the source.
    dst.set_node_property(ann_ids[0], "age", Value::from(99i64));
    let src_ann = src.find_nodes_by_property("name", &Value::from("Ann"));
    assert_eq!(
        src.get_node(src_ann[0])
            .unwrap()
            .properties
            .get(&PropertyKey::new("age")),
        Some(&Value::Int64(30)),
        "source must be unchanged by mutations to the copy"
    );
}

#[test]
fn copy_graph_self_copy_is_a_noop() {
    let store = LpgStore::new().expect("arena");
    let g = store.graph_or_create("g").expect("g");
    g.create_node(&["X"]);
    store
        .copy_graph(Some("g"), Some("g"))
        .expect("self-copy ok");
    assert_eq!(
        store.graph("g").unwrap().node_count(),
        1,
        "self-copy must not duplicate"
    );
}

#[test]
fn finalize_entities_by_id_scopes_to_named_entities() {
    use grafeo_common::types::EpochId;
    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(2);
    // Two nodes created by the same transaction at PENDING (invisible until finalized).
    let n1 = store.create_node_versioned(&["A"], EpochId::new(0), tx);
    let n2 = store.create_node_versioned(&["A"], EpochId::new(0), tx);
    let commit_epoch = EpochId::new(5);

    // Finalize only n1.
    store.finalize_entities_by_id(tx, commit_epoch, &[n1], &[]);

    assert!(
        store.is_node_visible_at_epoch(n1, commit_epoch),
        "finalized node must be visible at the commit epoch"
    );
    assert!(
        !store.is_node_visible_at_epoch(n2, commit_epoch),
        "an un-finalized node must remain PENDING (invisible)"
    );
}

#[test]
fn tx_property_overlay_isolates_uncommitted_writes() {
    use grafeo_common::types::PropertyKey;
    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["P"]);
    store.set_node_property(n, "age", Value::from(30i64)); // committed
    let tx = TransactionId::new(2);
    let key = PropertyKey::new("age");
    let epoch = store.current_epoch();

    // Uncommitted write goes to the transaction's delta, not the committed column.
    store.set_node_property_buffered(n, "age", Value::from(99i64), tx);

    // The writing transaction sees its own uncommitted value (read-your-writes).
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, Some(tx)),
        Some(Value::Int64(99))
    );
    // Any other reader (no transaction) sees only the committed value (no dirty read).
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(30))
    );
    // The committed column is untouched while the write is buffered.
    assert_eq!(store.get_node_property(n, &key), Some(Value::Int64(30)));

    // Apply (commit) promotes the value to the committed column.
    store.apply_tx_overlay(tx);
    assert_eq!(store.get_node_property(n, &key), Some(Value::Int64(99)));

    // A buffered remove tombstones for own-reads; drop (rollback) discards it.
    let tx2 = TransactionId::new(3);
    store.remove_node_property_buffered(n, "age", tx2);
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, Some(tx2)),
        None
    );
    assert_eq!(
        store.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(99))
    );
    store.drop_tx_overlay(tx2);
    assert_eq!(store.get_node_property(n, &key), Some(Value::Int64(99)));
}

#[test]
fn trait_accessor_isolates_buffered_writes() {
    use crate::graph::traits::GraphStoreMut;
    use grafeo_common::types::{PropertyKey, TransactionId};

    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    store.set_node_property(n, "age", Value::Int64(30));

    let tx = TransactionId::new(7);
    let key = PropertyKey::new("age");

    // Buffer an uncommitted write via the trait object.
    let s: &dyn GraphStoreMut = &store;
    s.set_node_property_buffered(n, "age", Value::Int64(99), tx);

    // Writer (Some(tx)) sees its own write; everyone else (None) sees committed.
    let epoch = store.current_epoch();
    assert_eq!(
        s.read_node_property_visible(n, &key, epoch, Some(tx)),
        Some(Value::Int64(99))
    );
    assert_eq!(
        s.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(30))
    );

    // Apply promotes the delta to the committed store.
    s.apply_tx_overlay(tx);
    assert_eq!(
        s.read_node_property_visible(n, &key, epoch, None),
        Some(Value::Int64(99))
    );
}

#[test]
fn whole_entity_accessor_merges_delta_for_writer() {
    use grafeo_common::types::{PropertyKey, TransactionId};

    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Thing"]);
    store.set_node_property(n, "keep", Value::Int64(1));
    store.set_node_property(n, "change", Value::Int64(2));
    store.set_node_property(n, "remove_me", Value::Int64(3));

    let tx = TransactionId::new(42);
    let epoch = store.current_epoch();

    // Buffer: change one key, remove another.
    store.set_node_property_buffered(n, "change", Value::Int64(99), tx);
    store.remove_node_property_buffered(n, "remove_me", tx);

    // Writer sees merged map: keep=1, change=99; remove_me absent.
    let writer_map = store.read_node_properties_visible(n, epoch, Some(tx));
    assert_eq!(
        writer_map.get(&PropertyKey::new("keep")),
        Some(&Value::Int64(1))
    );
    assert_eq!(
        writer_map.get(&PropertyKey::new("change")),
        Some(&Value::Int64(99))
    );
    assert!(
        !writer_map.contains_key(&PropertyKey::new("remove_me")),
        "remove_me must be absent for writer"
    );

    // Reader (tx=None) sees committed map: keep=1, change=2, remove_me=3.
    let reader_map = store.read_node_properties_visible(n, epoch, None);
    assert_eq!(
        reader_map.get(&PropertyKey::new("keep")),
        Some(&Value::Int64(1))
    );
    assert_eq!(
        reader_map.get(&PropertyKey::new("change")),
        Some(&Value::Int64(2))
    );
    assert_eq!(
        reader_map.get(&PropertyKey::new("remove_me")),
        Some(&Value::Int64(3))
    );
}

#[test]
fn label_delta_isolates_buffered_label_ops() {
    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    let tx = TransactionId::new(7);

    // Buffer add :Secret and remove :Person for tx.
    store.add_label_buffered(n, "Secret", tx);
    store.remove_label_buffered(n, "Person", tx);

    let writer_view =
        store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), Some(tx));
    assert!(
        writer_view.contains(&arcstr::ArcStr::from("Secret")),
        "writer sees buffered add"
    );
    assert!(
        !writer_view.contains(&arcstr::ArcStr::from("Person")),
        "writer sees buffered remove"
    );

    // Other readers (None) see committed labels unchanged.
    let committed_view =
        store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), None);
    assert!(
        committed_view.contains(&arcstr::ArcStr::from("Person")),
        "others see committed :Person"
    );
    assert!(
        !committed_view.contains(&arcstr::ArcStr::from("Secret")),
        "others do NOT see uncommitted :Secret"
    );

    // Apply promotes.
    store.apply_tx_overlay(tx);
    let after = store.read_node_labels_visible(n, grafeo_common::types::EpochId::new(0), None);
    assert!(
        after.contains(&arcstr::ArcStr::from("Secret"))
            && !after.contains(&arcstr::ArcStr::from("Person")),
        "commit applied label ops"
    );
}

#[test]
fn label_delta_rollback_and_savepoint() {
    let store = LpgStore::new().unwrap();
    let n = store.create_node(&["Person"]);
    let epoch = grafeo_common::types::EpochId::new(0);

    // --- Rollback path ---
    // tx A buffers :Secret but is then dropped (rolled back).
    let tx_a = TransactionId::new(10);
    store.add_label_buffered(n, "Secret", tx_a);
    store.drop_tx_overlay(tx_a);

    // After rollback the committed base must be unchanged — no :Secret visible.
    let after_rollback = store.read_node_labels_visible(n, epoch, None);
    assert!(
        after_rollback.contains(&arcstr::ArcStr::from("Person")),
        "committed :Person must survive rollback of tx A"
    );
    // :Secret must not have leaked into committed view.
    assert!(
        !after_rollback.contains(&arcstr::ArcStr::from("Secret")),
        "rolled-back :Secret must NOT be visible in committed view"
    );

    // --- Savepoint path ---
    let tx_b = TransactionId::new(11);

    // Buffer :Vip, take a savepoint, then buffer :Temp.
    store.add_label_buffered(n, "Vip", tx_b);
    let snap = store.tx_overlay_snapshot(tx_b);
    store.add_label_buffered(n, "Temp", tx_b);

    // Before restore: writer sees both Vip and Temp.
    let before_restore = store.read_node_labels_visible(n, epoch, Some(tx_b));
    assert!(
        before_restore.contains(&arcstr::ArcStr::from("Vip")),
        "writer must see buffered :Vip before restore"
    );
    assert!(
        before_restore.contains(&arcstr::ArcStr::from("Temp")),
        "writer must see buffered :Temp before restore"
    );

    // Restore to snapshot (Vip only, no Temp).
    store.tx_overlay_restore(tx_b, snap);

    let after_restore = store.read_node_labels_visible(n, epoch, Some(tx_b));
    assert!(
        after_restore.contains(&arcstr::ArcStr::from("Vip")),
        "writer must still see :Vip after savepoint restore"
    );
    assert!(
        !after_restore.contains(&arcstr::ArcStr::from("Temp")),
        "savepoint restore must discard :Temp — if this fails it is a real bug"
    );
    // The committed :Person must also be visible to the writer.
    assert!(
        after_restore.contains(&arcstr::ArcStr::from("Person")),
        "committed :Person must be visible to writer after restore"
    );
}

/// Mirror of `label_delta_isolates_buffered_label_ops` but exercised through
/// `&dyn GraphStoreMut` so the trait surface is tested at the trait level.
#[test]
fn trait_label_accessor_isolates() {
    use crate::graph::traits::GraphStoreMut;
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    // Exercise through a trait-object pointer.
    let dyn_store: &dyn GraphStoreMut = &store;

    let n = dyn_store.create_node(&["Person"]);
    let tx = TransactionId::new(99);

    // Buffer add :Secret and remove :Person for tx — via the trait object.
    dyn_store.add_label_buffered(n, "Secret", tx);
    dyn_store.remove_label_buffered(n, "Person", tx);

    // Writer view (Some(tx)) through the trait object.
    let writer_view = dyn_store.read_node_labels_visible(n, EpochId::new(0), Some(tx));
    assert!(
        writer_view.contains(&arcstr::ArcStr::from("Secret")),
        "writer sees buffered add via trait"
    );
    assert!(
        !writer_view.contains(&arcstr::ArcStr::from("Person")),
        "writer sees buffered remove via trait"
    );

    // Other reader (None) sees only committed labels.
    let committed_view = dyn_store.read_node_labels_visible(n, EpochId::new(0), None);
    assert!(
        committed_view.contains(&arcstr::ArcStr::from("Person")),
        "others see committed :Person via trait"
    );
    assert!(
        !committed_view.contains(&arcstr::ArcStr::from("Secret")),
        "others do NOT see uncommitted :Secret via trait"
    );

    // Apply promotes delta: committed view now reflects the writes.
    dyn_store.apply_tx_overlay(tx);
    let after = dyn_store.read_node_labels_visible(n, EpochId::new(0), None);
    assert!(
        after.contains(&arcstr::ArcStr::from("Secret"))
            && !after.contains(&arcstr::ArcStr::from("Person")),
        "commit promoted label ops via trait"
    );
}

/// Regression: a node inline-created **within a transaction** (e.g.
/// `MERGE (:Item)` / `CREATE (:Item)`) registers its labels directly into
/// `node_labels` at `EpochId::PENDING`. The writing transaction must see that
/// label through `read_node_labels_visible` so a later UNWIND row's MERGE can
/// dedupe against it. Under the `temporal` feature the committed-base read used
/// `VersionLog::at(real_epoch)`, which skips the PENDING entry and returned an
/// empty set — dropping the inline-create label and breaking MERGE-in-UNWIND
/// dedup (regression_external::unwind_merge_*).
#[test]
fn writer_sees_inline_create_label_for_own_pending_node() {
    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(42);
    let epoch = store.current_epoch();

    // Transactional inline create: labels land in `node_labels` at PENDING
    // (version_epoch = PENDING for a non-SYSTEM transaction).
    let n = store.create_node_versioned(&["Item"], epoch, tx);

    // The writing transaction must see its own just-created label.
    let writer_view = store.read_node_labels_visible(n, epoch, Some(tx));
    assert!(
        writer_view.contains(&arcstr::ArcStr::from("Item")),
        "writer must see its own inline-created :Item (got {writer_view:?})"
    );
}

/// Edge-delete isolation (MVCC increment 2b, Task 1).
///
/// A `delete_edge_transactional` must isolate the delete to the writing
/// transaction until commit, mirroring the node-delete deferral model:
///  - the writer sees the edge gone (read-your-writes via the version chain),
///  - every other session still sees it (PENDING `deleted_epoch` > any real epoch),
///  - the candidate adjacency index stays populated (it is NON-MVCC by design;
///    visibility is post-filtered one layer up in the expand operators), and
///  - edge properties and the live/edge-type counts are NOT touched at delete —
///    they are deferred to `finalize_edge_deletes_by_id` at commit, so an
///    uncommitted/rolled-back delete neither drops another session's property
///    read nor under-counts.
#[test]
fn edge_delete_pending_isolates() {
    use grafeo_common::types::{EpochId, PropertyKey};

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    // Edge with a property so we can assert property removal is deferred.
    let eid = store.create_edge_with_props(a, b, "R", [("prop", Value::from(7i64))]);

    let key = PropertyKey::new("prop");
    let epoch = store.current_epoch();
    let tx = TransactionId::new(2);
    let other_tx = TransactionId::new(3);

    // Pre-conditions: edge live and readable by everyone.
    assert!(store.is_edge_visible_versioned(eid, epoch, tx));
    assert_eq!(store.edge_properties.get(eid, &key), Some(Value::Int64(7)));
    let live_before = store.live_edge_count.load(Ordering::Relaxed);

    // Transactional delete: stamps PENDING deleted_epoch by `tx`.
    assert!(store.delete_edge_transactional(eid, epoch, tx));

    // Writer sees it gone (via the chain).
    assert!(
        store.get_edge_versioned(eid, epoch, tx).is_none(),
        "writer must not see its own deleted edge"
    );
    assert!(
        !store.is_edge_visible_versioned(eid, epoch, tx),
        "writer visibility check must report the edge gone"
    );

    // Other transactions still see it (via the chain) — no dirty write.
    assert!(
        store.get_edge_versioned(eid, epoch, other_tx).is_some(),
        "other session must still see the not-yet-committed edge"
    );
    assert!(
        store.is_edge_visible_versioned(eid, epoch, other_tx),
        "other session visibility check must still report the edge present"
    );

    // Candidate adjacency is NOT tombstoned (non-MVCC index, by design — true for
    // everyone; visibility is enforced by the expand operators' post-filter).
    assert!(
        store
            .forward_adj
            .edges_from(a)
            .iter()
            .any(|(_, e)| *e == eid),
        "forward adjacency must still contain the candidate edge before commit"
    );
    assert!(
        store
            .backward_adj
            .as_ref()
            .expect("default config enables backward adjacency")
            .edges_from(b)
            .iter()
            .any(|(_, e)| *e == eid),
        "backward adjacency must still contain the candidate edge before commit"
    );

    // Property removal is deferred: the stored property is still readable, and an
    // other-session read of the edge still carries it.
    assert_eq!(
        store.edge_properties.get(eid, &key),
        Some(Value::Int64(7)),
        "edge property must not be removed at delete time"
    );
    let other_view = store
        .get_edge_versioned(eid, epoch, other_tx)
        .expect("edge still visible to other session");
    assert_eq!(
        other_view.get_property("prop").and_then(|v| v.as_int64()),
        Some(7),
        "other session must still read the edge's property"
    );

    // Count decrement is deferred too.
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before,
        "live-edge count must not change at delete time"
    );

    // Commit: finalize the deferred delete at a real commit epoch.
    let commit_epoch = EpochId::new(5);
    store.finalize_edge_deletes_by_id(tx, commit_epoch, &[(a, eid, b)]);

    // Gone for everyone at the commit epoch.
    assert!(
        !store.is_edge_visible_versioned(eid, commit_epoch, other_tx),
        "after finalize the edge must be invisible to all sessions"
    );
    // Adjacency tombstone is now applied.
    assert!(
        !store
            .forward_adj
            .edges_from(a)
            .iter()
            .any(|(_, e)| *e == eid),
        "forward adjacency tombstone must be applied at finalize"
    );
    assert!(
        !store
            .backward_adj
            .as_ref()
            .expect("default config enables backward adjacency")
            .edges_from(b)
            .iter()
            .any(|(_, e)| *e == eid),
        "backward adjacency tombstone must be applied at finalize"
    );
    // Property removal moved to finalize too.
    assert_eq!(
        store.edge_properties.get(eid, &key),
        None,
        "edge property must be removed at finalize"
    );
    // The live-edge count decrement moved to finalize.
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before - 1,
        "live-edge count must decrease by one at finalize"
    );
}

/// Edge-delete rollback cleanliness (MVCC increment 2b, Task 3).
///
/// After a `delete_edge_transactional`, draining the pending set with
/// `take_pending_edge_deletes` and replaying it through
/// `rollback_pending_edge_deletes` must FULLY restore the edge — the chain is
/// unmarked (not left PENDING), so the edge is visible again to the writer.
/// Because the deferred model never touched adjacency, properties, or counts at
/// delete time, those are untouched throughout. Finally the pending set must be
/// drained (the `take` already emptied it). This is the only probe that
/// distinguishes a real restore from leaked-PENDING state.
#[test]
fn edge_delete_rollback_restores() {
    use grafeo_common::types::PropertyKey;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    // Edge with a property so we can assert it survives the rollback.
    let eid = store.create_edge_with_props(a, b, "R", [("prop", Value::from(7i64))]);

    let key = PropertyKey::new("prop");
    let epoch = store.current_epoch();
    let tx = TransactionId::new(2);

    let live_before = store.live_edge_count.load(Ordering::Relaxed);

    // Transactional delete: stamps PENDING deleted_epoch by `tx`.
    assert!(store.delete_edge_transactional(eid, epoch, tx));
    assert!(
        !store.is_edge_visible_versioned(eid, epoch, tx),
        "writer must not see its own deleted edge before rollback"
    );

    // Rollback: drain the pending set and unmark the chain.
    let ed = store.take_pending_edge_deletes(tx);
    assert_eq!(
        ed,
        vec![(a, eid, b)],
        "pending edge-delete set must carry the (src, edge, dst) tuple"
    );
    store.rollback_pending_edge_deletes(tx, &ed);

    // FULLY restored: chain unmarked, edge visible to the writer again (NOT a
    // leaked-PENDING state — a leak would leave it invisible to `tx`).
    assert!(
        store.is_edge_visible_versioned(eid, epoch, tx),
        "edge must be visible to the writer again after rollback (chain unmarked)"
    );
    assert!(
        store.get_edge_versioned(eid, epoch, tx).is_some(),
        "writer must read the restored edge after rollback"
    );

    // Adjacency was never touched (deferred path) — still present both directions.
    assert!(
        store
            .forward_adj
            .edges_from(a)
            .iter()
            .any(|(_, e)| *e == eid),
        "forward adjacency must still contain the edge after rollback"
    );
    assert!(
        store
            .backward_adj
            .as_ref()
            .expect("default config enables backward adjacency")
            .edges_from(b)
            .iter()
            .any(|(_, e)| *e == eid),
        "backward adjacency must still contain the edge after rollback"
    );

    // Live-edge count unchanged (decrement was deferred, never applied).
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before,
        "live-edge count must be unchanged after rollback"
    );

    // Property survives the rollback (removal was deferred, never applied).
    assert_eq!(
        store.edge_properties.get(eid, &key),
        Some(Value::Int64(7)),
        "edge property must survive the rollback"
    );

    // The pending set is drained — `take` already emptied it.
    assert!(
        store.take_pending_edge_deletes(tx).is_empty(),
        "pending edge-delete set must be drained after take"
    );
}

/// Re-deleting an edge already deleted by the SAME transaction must be an
/// idempotent no-op (MVCC increment 2b). The PENDING `deleted_epoch` is
/// `u64::MAX`, so a naive `visible_at(epoch)` re-delete guard still sees the
/// record and re-stamps it, pushing a DUPLICATE `(src, edge, dst)` into the
/// pending set; `finalize_edge_deletes_by_id` would then decrement counts twice.
/// The tx-aware guard (`visible_to`, which hides an edge from the tx that
/// deleted it) makes the second call return `false` with no second push.
#[test]
fn edge_delete_transactional_is_idempotent_per_tx() {
    use grafeo_common::types::EpochId;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let eid = store.create_edge(a, b, "R");

    let epoch = store.current_epoch();
    let tx = TransactionId::new(2);
    let live_before = store.live_edge_count.load(Ordering::Relaxed);

    // First delete succeeds and records the edge once.
    assert!(store.delete_edge_transactional(eid, epoch, tx));
    // Second delete by the SAME tx is a no-op: the edge is already gone for `tx`.
    assert!(
        !store.delete_edge_transactional(eid, epoch, tx),
        "re-delete by the same tx must be an idempotent no-op (return false)"
    );

    // The pending set must carry the edge exactly ONCE (not twice).
    {
        let pending = store.pending_tx_edge_deletes.read();
        let entries = pending.get(&tx).map_or(0, |v| v.len());
        assert_eq!(
            entries, 1,
            "pending edge-delete set must carry the edge exactly once after a re-delete"
        );
    }

    // Finalize must decrement the live-edge count by exactly ONE (not two).
    let commit_epoch = EpochId::new(5);
    let pending = store.take_pending_edge_deletes(tx);
    store.finalize_edge_deletes_by_id(tx, commit_epoch, &pending);
    assert_eq!(
        store.live_edge_count.load(Ordering::Relaxed),
        live_before - 1,
        "live-edge count must decrease by exactly one despite the duplicate delete attempt"
    );
}

// ── Read-tracker registry ────────────────────────────────────────────────────

#[test]
fn test_read_tracker_registered_records_node() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::{EdgeId, NodeId};
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct SpyTracker {
        nodes: Mutex<Vec<NodeId>>,
        edges: Mutex<Vec<EdgeId>>,
    }
    impl ReadTracker for SpyTracker {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.nodes.lock().push(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().push(id);
        }
    }

    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(42);
    let spy = Arc::new(SpyTracker {
        nodes: Mutex::new(Vec::new()),
        edges: Mutex::new(Vec::new()),
    });
    let tracker: SharedReadTracker = spy.clone();

    // Before registration: no-op.
    store.record_read_node(tx, NodeId::new(1));
    store.record_read_edge(tx, EdgeId::new(1));
    assert!(
        spy.nodes.lock().is_empty(),
        "no recording before registration"
    );
    assert!(
        spy.edges.lock().is_empty(),
        "no recording before registration"
    );

    // After registration: reads are recorded.
    store.register_read_tracker(tx, tracker);
    store.record_read_node(tx, NodeId::new(5));
    store.record_read_edge(tx, EdgeId::new(7));
    assert_eq!(
        *spy.nodes.lock(),
        vec![NodeId::new(5)],
        "node read recorded"
    );
    assert_eq!(
        *spy.edges.lock(),
        vec![EdgeId::new(7)],
        "edge read recorded"
    );

    // After unregistration: no further recording.
    store.unregister_read_tracker(tx);
    store.record_read_node(tx, NodeId::new(99));
    store.record_read_edge(tx, EdgeId::new(99));
    assert_eq!(
        spy.nodes.lock().len(),
        1,
        "no extra node read after unregistration"
    );
    assert_eq!(
        spy.edges.lock().len(),
        1,
        "no extra edge read after unregistration"
    );
}

#[test]
fn test_read_tracker_unregistered_tx_is_noop() {
    // record_read_* for a tx that was never registered must be a silent no-op
    // (no panic, nothing collected anywhere).
    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(999);
    // These must not panic.
    store.record_read_node(tx, NodeId::new(1));
    store.record_read_edge(tx, EdgeId::new(1));
}

#[test]
fn test_read_tracker_cleared_by_store_clear() {
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::{EdgeId, NodeId};
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct SpyTracker {
        nodes: Mutex<Vec<NodeId>>,
        edges: Mutex<Vec<EdgeId>>,
    }
    impl ReadTracker for SpyTracker {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.nodes.lock().push(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().push(id);
        }
    }

    let store = LpgStore::new().unwrap();
    let tx = TransactionId::new(1);
    let spy = Arc::new(SpyTracker {
        nodes: Mutex::new(Vec::new()),
        edges: Mutex::new(Vec::new()),
    });
    let tracker: SharedReadTracker = spy.clone();

    store.register_read_tracker(tx, tracker);
    // Sanity: records before clear.
    store.record_read_node(tx, NodeId::new(3));
    assert_eq!(spy.nodes.lock().len(), 1);

    // clear() must drop the tracker entry.
    store.clear();
    store.record_read_node(tx, NodeId::new(9));
    assert_eq!(
        spy.nodes.lock().len(),
        1,
        "read tracker must be cleared by store.clear()"
    );
}

// ── Store-level visible-read chokepoints record into the tracker ─────────────

/// Builds a reusable spy tracker + helper to avoid duplication across tests.
#[cfg(test)]
mod visible_read_recording {
    use super::*;
    use crate::execution::operators::{ReadTracker, SharedReadTracker};
    use grafeo_common::types::{EdgeId, NodeId};
    use parking_lot::Mutex;
    use std::sync::Arc;

    pub struct SpyTracker {
        pub nodes: Mutex<Vec<NodeId>>,
        pub edges: Mutex<Vec<EdgeId>>,
    }

    impl SpyTracker {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                nodes: Mutex::new(Vec::new()),
                edges: Mutex::new(Vec::new()),
            })
        }
    }

    impl ReadTracker for SpyTracker {
        fn record_node_read(&self, _tx: TransactionId, id: NodeId) {
            self.nodes.lock().push(id);
        }
        fn record_edge_read(&self, _tx: TransactionId, id: EdgeId) {
            self.edges.lock().push(id);
        }
    }

    /// Creates a store with:
    ///   - two committed nodes (n_visible, n_deleted) with properties
    ///   - one committed edge between them (e_visible)
    ///   - n_deleted is then deleted
    ///   - a Serializable-tx read tracker registered for `tx`
    pub fn fixture() -> (
        LpgStore,
        TransactionId,
        NodeId,
        NodeId,
        EdgeId,
        Arc<SpyTracker>,
    ) {
        let store = LpgStore::new().unwrap();
        let epoch = store.current_epoch();

        let n_visible = store.create_node_versioned(&["Person"], epoch, TransactionId::SYSTEM);
        let n_deleted = store.create_node_versioned(&["Person"], epoch, TransactionId::SYSTEM);
        let e_visible = store.create_edge_versioned(
            n_visible,
            n_deleted,
            "KNOWS",
            epoch,
            TransactionId::SYSTEM,
        );
        store.set_node_property(n_visible, "name", Value::from("Alix"));
        store.set_node_property(n_deleted, "name", Value::from("ghost"));
        store.set_edge_property(e_visible, "since", Value::from(2020i64));
        // Now delete n_deleted so visibility checks return false for it
        store.delete_node(n_deleted);
        // n_deleted is gone — but e_visible still exists (endpoints: n_visible→n_deleted)
        // (edge endpoints survive node deletes unless DETACH DELETE is used)

        let tx = TransactionId::new(77);
        let spy = SpyTracker::new();
        let tracker: SharedReadTracker = Arc::clone(&spy) as SharedReadTracker;
        store.register_read_tracker(tx, tracker);

        (store, tx, n_visible, n_deleted, e_visible, spy)
    }
}

#[test]
fn test_get_node_versioned_records_visible_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, n_deleted, _e, spy) = fixture();
    let epoch = store.current_epoch();

    // Visible node → recorded
    let node = store.get_node_versioned(n_visible, epoch, tx);
    assert!(node.is_some(), "expected node to be visible");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "visible node must be recorded"
    );

    // Deleted node → not recorded
    let deleted = store.get_node_versioned(n_deleted, epoch, tx);
    assert!(deleted.is_none(), "deleted node should not be visible");
    assert!(
        !spy.nodes.lock().contains(&n_deleted),
        "non-visible (deleted) node must NOT be recorded"
    );

    // tx=None → no recording
    spy.nodes.lock().clear();
    let _ = store.get_node_versioned(n_visible, epoch, TransactionId::new(999));
    assert!(
        spy.nodes.lock().is_empty(),
        "unregistered tx must not record anything"
    );
}

#[test]
fn test_get_edge_versioned_records_visible_edge() {
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();

    // Visible edge → recorded
    let edge = store.get_edge_versioned(e_visible, epoch, tx);
    assert!(edge.is_some(), "expected edge to be visible");
    assert!(
        spy.edges.lock().contains(&e_visible),
        "visible edge must be recorded"
    );
}

#[test]
fn test_is_node_visible_versioned_records_on_true() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, n_deleted, _e, spy) = fixture();
    let epoch = store.current_epoch();

    // True → recorded
    assert!(store.is_node_visible_versioned(n_visible, epoch, tx));
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "visible node must be recorded on true return"
    );

    // False (deleted) → not recorded
    spy.nodes.lock().clear();
    assert!(!store.is_node_visible_versioned(n_deleted, epoch, tx));
    assert!(
        spy.nodes.lock().is_empty(),
        "non-visible (deleted) node must NOT be recorded"
    );
}

#[test]
fn test_is_edge_visible_versioned_records_on_true() {
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();

    assert!(store.is_edge_visible_versioned(e_visible, epoch, tx));
    assert!(
        spy.edges.lock().contains(&e_visible),
        "visible edge must be recorded"
    );
}

#[test]
fn test_filter_visible_node_ids_versioned_records_each_visible_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, n_deleted, _e, spy) = fixture();
    let epoch = store.current_epoch();

    let visible = store.filter_visible_node_ids_versioned(&[n_visible, n_deleted], epoch, tx);
    assert_eq!(
        visible,
        vec![n_visible],
        "only n_visible should pass filter"
    );
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "visible node must be in recorded set"
    );
    assert!(
        !spy.nodes.lock().contains(&n_deleted),
        "non-visible node must NOT be recorded"
    );
}

#[test]
fn test_read_node_property_visible_records_node() {
    use grafeo_common::types::PropertyKey;
    use visible_read_recording::fixture;
    let (store, tx, n_visible, _nd, _e, spy) = fixture();
    let epoch = store.current_epoch();
    let key = PropertyKey::new("name");

    // With Some(tx) → records the node
    let val = store.read_node_property_visible(n_visible, &key, epoch, Some(tx));
    assert!(val.is_some(), "property should exist");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "node must be recorded on property read"
    );

    // With None (no tx) → no recording
    spy.nodes.lock().clear();
    let _ = store.read_node_property_visible(n_visible, &key, epoch, None);
    assert!(spy.nodes.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_edge_property_visible_records_edge() {
    use grafeo_common::types::PropertyKey;
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();
    let key = PropertyKey::new("since");

    let val = store.read_edge_property_visible(e_visible, &key, epoch, Some(tx));
    assert!(val.is_some(), "property should exist");
    assert!(
        spy.edges.lock().contains(&e_visible),
        "edge must be recorded on property read"
    );

    // With None → no recording
    spy.edges.lock().clear();
    let _ = store.read_edge_property_visible(e_visible, &key, epoch, None);
    assert!(spy.edges.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_node_properties_visible_records_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, _nd, _e, spy) = fixture();
    let epoch = store.current_epoch();

    let props = store.read_node_properties_visible(n_visible, epoch, Some(tx));
    assert!(!props.is_empty(), "properties should exist");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "node must be recorded on whole-entity property read"
    );

    spy.nodes.lock().clear();
    let _ = store.read_node_properties_visible(n_visible, epoch, None);
    assert!(spy.nodes.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_edge_properties_visible_records_edge() {
    use visible_read_recording::fixture;
    let (store, tx, _n, _nd, e_visible, spy) = fixture();
    let epoch = store.current_epoch();

    let props = store.read_edge_properties_visible(e_visible, epoch, Some(tx));
    assert!(!props.is_empty(), "properties should exist");
    assert!(
        spy.edges.lock().contains(&e_visible),
        "edge must be recorded on whole-entity property read"
    );

    spy.edges.lock().clear();
    let _ = store.read_edge_properties_visible(e_visible, epoch, None);
    assert!(spy.edges.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_read_node_labels_visible_records_node() {
    use visible_read_recording::fixture;
    let (store, tx, n_visible, _nd, _e, spy) = fixture();
    let epoch = store.current_epoch();

    let labels = store.read_node_labels_visible(n_visible, epoch, Some(tx));
    assert!(!labels.is_empty(), "labels should exist");
    assert!(
        spy.nodes.lock().contains(&n_visible),
        "node must be recorded on label read"
    );

    spy.nodes.lock().clear();
    let _ = store.read_node_labels_visible(n_visible, epoch, None);
    assert!(spy.nodes.lock().is_empty(), "tx=None must not record");
}

#[test]
fn test_nodes_by_label_visible_records_each_returned_node() {
    use visible_read_recording::fixture;
    let (store, tx, _n_visible, n_deleted, _e, spy) = fixture();

    // n_deleted was deleted so nodes_by_label returns only n_visible
    let ids = store.nodes_by_label_visible("Person", Some(tx));
    // The label index doesn't filter by tx-visibility; it returns n_visible
    // (n_deleted was removed from the label_index by delete_node)
    for &id in &ids {
        assert!(
            spy.nodes.lock().contains(&id),
            "each returned node must be recorded"
        );
    }
    assert!(
        !spy.nodes.lock().contains(&n_deleted),
        "deleted node removed from label_index must not be recorded"
    );

    // tx=None → no recording
    spy.nodes.lock().clear();
    let _ = store.nodes_by_label_visible("Person", None);
    assert!(spy.nodes.lock().is_empty(), "tx=None must not record");
}
