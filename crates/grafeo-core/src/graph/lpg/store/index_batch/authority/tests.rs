use super::*;
use crate::graph::lpg::store::index_batch::{IndexRegistryWorkspace, StoreIndexEdits};

#[test]
fn nested_targets_share_one_linear_topology_capture() {
    let mut stores = vec![Arc::new(LpgStore::new().unwrap())];
    for index in 0..23 {
        let child = stores[index].graph_or_create("next").unwrap();
        stores.push(child);
    }
    let mut workspace = IndexRegistryWorkspace::new(
        stores
            .iter()
            .rev()
            .map(|store| StoreIndexEdits {
                store: store.as_ref(),
                edits: Vec::new(),
            })
            .collect(),
    );
    workspace.prepare_inputs().unwrap();
    let authority =
        RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
            .unwrap();
    assert_eq!(authority.workspace.nodes.len(), stores.len());
    assert_eq!(authority.workspace.identities.len(), stores.len());
    assert_eq!(authority.workspace.ordered.len(), stores.len());
    for (actual, expected) in authority.workspace.ordered.iter().zip(&stores) {
        assert!(std::ptr::eq(*actual, expected.as_ref()));
    }
    assert!(
        authority
            .workspace
            .nodes
            .iter()
            .all(|node| matches!(node.state, VisitState::Complete))
    );
}

#[test]
fn descendant_cycle_is_rejected_even_when_no_cycle_node_is_a_target() {
    let root = LpgStore::new().unwrap();
    let child = root.graph_or_create("child").unwrap();
    let leaf = child.graph_or_create("leaf").unwrap();
    // Deliberately malformed internal topology; public DDL rejects this shape.
    leaf.named_graphs
        .write()
        .insert("cycle".to_owned(), Arc::clone(&child));
    let mut workspace = IndexRegistryWorkspace::new(vec![StoreIndexEdits {
        store: &root,
        edits: Vec::new(),
    }]);
    workspace.prepare_inputs().unwrap();
    let result =
        RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority);
    assert!(result.is_err());
    drop(result);
    assert!(workspace.authority.transitions.is_empty());
    assert!(workspace.authority.bindings.is_none());
    assert_eq!(workspace.authority.nodes.len(), 3);
    // Break the intentionally created Arc cycle before retiring the fixture.
    leaf.named_graphs.write().remove("cycle");
    assert!(root.mutation_scope_gate.try_write().is_some());
    assert!(child.mutation_scope_gate.try_write().is_some());
    assert!(leaf.mutation_scope_gate.try_write().is_some());
}

#[test]
fn authority_release_retains_all_guard_and_topology_buffers() {
    let root = LpgStore::new().unwrap();
    let child = root.graph_or_create("child").unwrap();
    let leaf = child.graph_or_create("leaf").unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![
        StoreIndexEdits {
            store: &leaf,
            edits: Vec::new(),
        },
        StoreIndexEdits {
            store: &child,
            edits: Vec::new(),
        },
        StoreIndexEdits {
            store: &root,
            edits: Vec::new(),
        },
    ]);
    workspace.prepare_inputs().unwrap();
    let authority =
        RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
            .unwrap();
    assert!(std::ptr::eq(
        authority.workspace.ordered[0],
        &raw const root
    ));
    assert!(std::ptr::eq(authority.workspace.ordered[1], child.as_ref()));
    assert!(std::ptr::eq(authority.workspace.ordered[2], leaf.as_ref()));
    let transition_buffer = authority.workspace.transitions.as_ptr();
    let topology_buffer = authority.workspace.nodes.as_ptr();
    let ordered_buffer = authority.workspace.ordered.as_ptr();
    let frontier_buffer = authority.workspace.frontier.as_ptr();
    let capacities = (
        authority.workspace.transitions.capacity(),
        authority.workspace.nodes.capacity(),
        authority.workspace.ordered.capacity(),
        authority.workspace.frontier.capacity(),
    );
    let identities: Vec<_> = authority
        .workspace
        .nodes
        .iter()
        .map(|node| physical_order(node.anchor.store()))
        .collect();
    assert_eq!(identities.len(), 3, "one captured node per reachable store");
    let identity_capacity = authority.workspace.identities.capacity();
    crate::allocation_test::start();
    drop(authority);
    let traffic = crate::allocation_test::stop();
    assert_eq!(traffic, crate::allocation_test::Counts::default());
    let retained = &workspace.authority;
    assert!(retained.transitions.is_empty());
    assert!(retained.bindings.is_none());
    assert_eq!(retained.transitions.as_ptr(), transition_buffer);
    assert_eq!(retained.nodes.as_ptr(), topology_buffer);
    assert_eq!(retained.ordered.as_ptr(), ordered_buffer);
    assert_eq!(retained.frontier.as_ptr(), frontier_buffer);
    assert_eq!(
        (
            retained.transitions.capacity(),
            retained.nodes.capacity(),
            retained.ordered.capacity(),
            retained.frontier.capacity()
        ),
        capacities
    );
    assert_eq!(
        retained
            .nodes
            .iter()
            .map(|node| physical_order(node.anchor.store()))
            .collect::<Vec<_>>(),
        identities
    );
    assert_eq!(retained.identities.capacity(), identity_capacity);
    assert!(root.mutation_scope_gate.try_write().is_some());
    assert!(child.mutation_scope_gate.try_write().is_some());
    assert!(leaf.mutation_scope_gate.try_write().is_some());
}

#[test]
fn forgotten_authority_drains_transitions_before_topology_anchors_retire() {
    let root = LpgStore::new().unwrap();
    let child = root.graph_or_create("child").unwrap();
    let leaf = child.graph_or_create("leaf").unwrap();
    let initial_anchors = Arc::strong_count(&leaf);
    let mut workspace = IndexRegistryWorkspace::new(vec![
        StoreIndexEdits {
            store: &child,
            edits: Vec::new(),
        },
        StoreIndexEdits {
            store: &root,
            edits: Vec::new(),
        },
    ]);
    workspace.prepare_inputs().unwrap();
    let authority =
        RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
            .unwrap();
    std::mem::forget(authority);
    assert!(root.mutation_scope_gate.try_write().is_none());
    assert!(child.mutation_scope_gate.try_write().is_none());
    assert!(Arc::strong_count(&leaf) > initial_anchors);
    drop(workspace);
    assert!(root.mutation_scope_gate.try_write().is_some());
    assert!(child.mutation_scope_gate.try_write().is_some());
    assert_eq!(Arc::strong_count(&leaf), initial_anchors);
}

#[test]
fn failed_topology_reservation_keeps_scratch_and_releases_all_authority() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            super::super::RESERVATION_FAILURE.with(|counter| counter.set(None));
        }
    }
    let root = LpgStore::new().unwrap();
    let child = root.graph_or_create("child").unwrap();
    let mut workspace = IndexRegistryWorkspace::new(vec![
        StoreIndexEdits {
            store: &child,
            edits: Vec::new(),
        },
        StoreIndexEdits {
            store: &root,
            edits: Vec::new(),
        },
    ]);
    workspace.prepare_inputs().unwrap();
    let result = {
        let _reset = Reset;
        // Five outer reservations then fail during the first topology walk.
        super::super::RESERVATION_FAILURE.with(|counter| counter.set(Some(5)));
        RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority)
    };
    assert!(result.is_err());
    drop(result);
    assert!(workspace.authority.transitions.is_empty());
    assert!(workspace.authority.bindings.is_none());
    assert_eq!(workspace.authority.nodes.len(), 2);
    assert!(workspace.authority.nodes.capacity() >= 2);
    assert!(root.mutation_scope_gate.try_write().is_some());
    assert!(child.mutation_scope_gate.try_write().is_some());
    assert!(
        RegistryAuthority::acquire(&mut workspace.registry.stores, &mut workspace.authority,)
            .is_err()
    );
}
