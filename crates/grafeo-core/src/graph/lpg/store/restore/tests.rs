use super::*;
use crate::allocation_test::{self, Counts};
use crate::graph::Direction;
use crate::graph::lpg::store::LpgStoreConfig;
use crate::graph::write_permit::{WriteAuthority, with_authority};
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, TransactionId, Value};
use std::cell::Cell;

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

thread_local! {
    static REJECT_LATE: Cell<bool> = const { Cell::new(false) };
    static MONITOR: Cell<bool> = const { Cell::new(false) };
    static STARTED: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<Counts> = const { Cell::new(Counts { alloc: 0, zeroed: 0, realloc: 0, dealloc: 0 }) };
}

pub(super) fn before_install() -> std::result::Result<(), &'static str> {
    if REJECT_LATE.with(Cell::get) {
        return Err("injected late pristine qualification failure");
    }
    if MONITOR.with(Cell::get) {
        STARTED.with(|value| value.set(true));
        allocation_test::start();
    }
    Ok(())
}

pub(super) fn finish_probe() {
    if STARTED.with(|value| value.replace(false)) {
        let counts = allocation_test::stop();
        COUNTS.with(|value| value.set(counts));
    }
}

fn image(target: &LpgStore) -> TestResult<LpgStore> {
    let source = target.new_restore_candidate()?;
    let birth = EpochId::new(3);
    let closed = EpochId::new(7);
    let n = NodeId::new(4);
    let m = NodeId::new(8);
    source.restore_node_history_exact(
        n,
        &[(birth, None)],
        &[(birth, vec![ArcStr::from("Person")])],
    )?;
    source.restore_node_history_exact(m, &[(birth, None)], &[(birth, Vec::new())])?;
    source.restore_edge_history_exact(EdgeId::new(12), n, m, "KNOWS", &[(birth, Some(closed))])?;
    source.set_node_property_at_epoch(n, "name", Value::from("before"), birth);
    source.set_node_property_at_epoch(n, "name", Value::from("after"), closed);
    source.set_edge_property_at_epoch(EdgeId::new(12), "weight", Value::Int64(9), birth);
    source.restore_allocator_high_water_exact(50, 60)?;
    source.sync_epoch(closed);
    source.create_graph("a/b")?;
    {
        let parent = source.graph_or_create("a")?;
        parent.create_graph("")?;
        let leaf = parent.graph_or_create("b")?;
        assert!(leaf.create_node(&["Child"]).is_valid());
    }
    Ok(source)
}

fn assert_pristine(store: &LpgStore) {
    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);
    assert_eq!(store.next_node_id(), 0);
    assert_eq!(store.next_edge_id(), 0);
    assert_eq!(store.current_epoch(), EpochId::INITIAL);
    assert!(store.graph_names().is_empty());
    assert!(store.node_labels.read().is_empty());
    assert!(store.label_registry.read().id_to_name.is_empty());
    assert!(store.id_to_edge_type.read().is_empty());
}

#[test]
fn pristine_restore_preserves_retained_arc_configuration_and_exact_authority() -> TestResult {
    let target = Arc::new(LpgStore::with_config(LpgStoreConfig {
        backward_edges: false,
        ..LpgStoreConfig::default()
    })?);
    let retained = Arc::clone(&target);
    let physical = Arc::clone(&target.index_physical_identity);
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    let binding = target.index_owner_id;
    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(target.seal_unframed_writes(&owner));
    assert!(!target.new_restore_candidate()?.has_backward_adjacency());
    assert!(target.install_pristine_image(image(&target)?).is_err());
    assert_pristine(&target);
    let candidate = image(&target)?;
    assert!(with_authority(&foreign, || target.install_pristine_image(candidate)).is_err());
    assert_pristine(&target);
    let candidate = image(&target)?;
    with_authority(&owner, || target.install_pristine_image(candidate))?;
    assert!(Arc::ptr_eq(&target, &retained));
    assert!(Arc::ptr_eq(&target.index_physical_identity, &physical));
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    assert_eq!(target.index_owner_id, binding);
    assert!(!retained.has_backward_adjacency());
    assert_eq!(retained.node_count(), 2);
    assert_eq!(retained.next_node_id(), 50);
    assert_eq!(retained.next_edge_id(), 60);
    assert!(retained.graph("a/b").is_some());
    let parent = retained.graph("a").ok_or("missing imported parent")?;
    assert!(parent.graph("").is_some());
    let leaf = parent.graph("b").ok_or("missing imported child")?;
    assert_eq!(leaf.node_count(), 1);
    assert!(!leaf.create_node(&["Denied"]).is_valid());
    assert!(with_authority(&owner, || leaf.create_node(&["Allowed"])).is_valid());
    Ok(())
}

#[test]
fn pristine_restore_rejects_late_failure_and_aliased_child_without_publication() -> TestResult {
    let target = LpgStore::new()?;
    let candidate = image(&target)?;
    REJECT_LATE.with(|value| value.set(true));
    let rejected = target.install_pristine_image(candidate);
    REJECT_LATE.with(|value| value.set(false));
    assert!(rejected.is_err());
    assert_pristine(&target);
    let candidate = image(&target)?;
    let parent = candidate.graph("a").ok_or("missing candidate parent")?;
    let child = parent.graph("b").ok_or("missing candidate child")?;
    drop(parent);
    assert!(target.install_pristine_image(candidate).is_err());
    assert_pristine(&target);
    assert!(child.create_node(&["StillUnsealed"]).is_valid());
    let candidate = image(&target)?;
    let late_reader = target.write_trackers.read();
    assert!(target.install_pristine_image(candidate).is_err());
    drop(late_reader);
    assert_pristine(&target);
    target.install_pristine_image(image(&target)?)?;
    Ok(())
}

#[test]
fn pristine_restore_final_install_and_guard_release_have_zero_allocator_traffic() -> TestResult {
    let target = LpgStore::new()?;
    let candidate = image(&target)?;
    MONITOR.with(|value| value.set(true));
    let result = target.install_pristine_image(candidate);
    MONITOR.with(|value| value.set(false));
    result?;
    assert_eq!(COUNTS.with(Cell::get), Counts::default());
    let n = NodeId::new(4);
    let edge = EdgeId::new(12);
    assert_eq!(
        target.get_node_property_at_epoch(n, &PropertyKey::new("name"), EpochId::new(3)),
        Some(Value::from("before"))
    );
    assert_eq!(
        target.get_node_property_at_epoch(n, &PropertyKey::new("name"), EpochId::new(7)),
        Some(Value::from("after"))
    );
    assert_eq!(
        target
            .get_edge_history(edge)
            .into_iter()
            .map(|(birth, death, _)| (birth, death))
            .collect::<Vec<_>>(),
        vec![(EpochId::new(3), Some(EpochId::new(7)))]
    );
    assert!(target.get_edge_at_epoch(edge, EpochId::new(3)).is_some());
    assert!(target.get_edge_at_epoch(edge, EpochId::new(7)).is_none());
    assert_eq!(target.edge_count(), 0);
    assert!(target.neighbors(n, Direction::Outgoing).next().is_none());
    Ok(())
}

#[test]
fn pristine_restore_rejects_pending_and_nonpristine_state() -> TestResult {
    let target = LpgStore::new()?;
    let candidate = target.new_restore_candidate()?;
    assert!(
        candidate
            .create_node_versioned(&["Pending"], EpochId::PENDING, TransactionId::new(1))
            .is_valid()
    );
    assert!(target.install_pristine_image(candidate).is_err());
    assert_pristine(&target);
    let candidate = image(&target)?;
    let existing = target.create_node(&["Keep"]);
    assert!(target.install_pristine_image(candidate).is_err());
    assert_eq!(target.node_count(), 1);
    assert!(target.get_node(existing).is_some());
    assert!(target.graph_names().is_empty());
    Ok(())
}

#[cfg(feature = "tiered-storage")]
#[test]
fn pristine_restore_moves_hot_backing_and_refuses_cold_candidates() -> TestResult {
    let target = LpgStore::new()?;
    let candidate = image(&target)?;
    assert!(candidate.freeze_epoch(EpochId::new(3)) > 0);
    assert!(target.install_pristine_image(candidate).is_err());
    assert_pristine(&target);
    let arena_identity = Arc::clone(&target.arena_allocator);
    target.install_pristine_image(image(&target)?)?;
    assert!(Arc::ptr_eq(&arena_identity, &target.arena_allocator));
    assert!(
        target
            .get_node_at_epoch(NodeId::new(4), EpochId::new(3))
            .is_some()
    );
    assert!(
        target
            .get_edge_at_epoch(EdgeId::new(12), EpochId::new(3))
            .is_some()
    );
    assert!(target.freeze_epoch(EpochId::new(3)) > 0);
    assert!(
        target
            .get_node_at_epoch(NodeId::new(4), EpochId::new(3))
            .is_some()
    );
    assert!(
        target
            .get_edge_at_epoch(EdgeId::new(12), EpochId::new(3))
            .is_some()
    );
    Ok(())
}

#[derive(Debug)]
struct ReplacementTestError(Error);

impl From<Error> for ReplacementTestError {
    fn from(error: Error) -> Self {
        Self(error)
    }
}

impl From<crate::graph::lpg::DataRebindError> for ReplacementTestError {
    fn from(error: crate::graph::lpg::DataRebindError) -> Self {
        Self(error.into_error())
    }
}

fn publish_replacement(data: &mut LpgReplacementWorkspace, fail: bool) -> Result<()> {
    use crate::graph::lpg::{
        IndexRegistryWorkspace, StoreIndexEdits, with_prepared_lpg_replacement,
    };
    let anchors = data.graphs().to_vec();
    let mut indexes = IndexRegistryWorkspace::new(
        anchors
            .iter()
            .map(|(_, store)| StoreIndexEdits {
                store,
                edits: Vec::new(),
            })
            .collect(),
    );
    with_prepared_lpg_replacement(
        data,
        &mut indexes,
        || {
            if fail {
                Err(ReplacementTestError(Error::Serialization(
                    "companion preparation failed".into(),
                )))
            } else {
                Ok(())
            }
        },
        |()| (),
    )
    .map_err(|error| error.0)
}

#[test]
fn replacement_preserves_populated_root_and_old_child_seals() -> TestResult {
    let target = Arc::new(LpgStore::new()?);
    let old = target.create_node(&["Old"]);
    target.set_node_property(old, "value", Value::from("old"));
    let child = target.graph_or_create("old")?;
    let child_node = child.create_node(&["OldChild"]);
    let authority = WriteAuthority::new();
    assert!(target.seal_unframed_writes(&authority));
    let retained = Arc::clone(&target);
    let mut rejected = LpgReplacementWorkspace::new(Arc::clone(&target), image(&target)?)?;
    assert!(with_authority(&authority, || publish_replacement(&mut rejected, true)).is_err());
    assert_eq!(target.node_count(), 1);
    assert!(Arc::ptr_eq(
        &child,
        &target.graph("old").ok_or("old child missing")?
    ));
    assert_eq!(
        target.get_node_property(old, &PropertyKey::new("value")),
        Some(Value::from("old"))
    );

    let mut accepted = LpgReplacementWorkspace::new(Arc::clone(&target), image(&target)?)?;
    with_authority(&authority, || publish_replacement(&mut accepted, false))?;
    assert!(Arc::ptr_eq(&target, &retained));
    assert_eq!(retained.node_count(), 2);
    assert_eq!(retained.next_node_id(), 50);
    assert_eq!(retained.next_edge_id(), 60);
    assert!(retained.graph("old").is_none());
    assert!(child.get_node(child_node).is_some());
    assert!(!child.create_node(&["Denied"]).is_valid());
    let incoming = retained
        .graph("a")
        .and_then(|parent| parent.graph("b"))
        .ok_or("incoming child missing")?;
    assert!(!incoming.create_node(&["Denied"]).is_valid());
    Ok(())
}

#[test]
fn unsealed_replacement_rejects_changed_child_bytes_and_equal_byte_topology() -> TestResult {
    for change_topology in [false, true] {
        let target = Arc::new(LpgStore::new()?);
        let old = target.create_node(&["Retained"]);
        let candidate = target.new_live_replacement_candidate()?;
        {
            let parent = candidate.graph_or_create("parent")?;
            parent.create_graph("child")?;
        }
        let mut workspace = LpgReplacementWorkspace::new(Arc::clone(&target), candidate)?;
        let parent = workspace
            .candidate()
            .graph("parent")
            .ok_or("parent missing")?;
        if change_topology {
            assert!(parent.drop_graph("child"));
            assert!(parent.create_graph("child")?);
        } else {
            assert!(parent.create_node(&["ChangedAfterStaging"]).is_valid());
        }
        assert!(publish_replacement(&mut workspace, false).is_err());
        assert!(target.get_node(old).is_some());
        assert!(target.graph_names().is_empty());
    }
    let target = Arc::new(LpgStore::new()?);
    target.create_node(&["Old"]);
    let mut workspace = LpgReplacementWorkspace::new(Arc::clone(&target), image(&target)?)?;
    publish_replacement(&mut workspace, false)?;
    assert_eq!(target.node_count(), 2);
    assert!(target.create_node(&["StillUnsealed"]).is_valid());
    Ok(())
}

#[cfg(feature = "tiered-storage")]
#[test]
fn replacement_retires_populated_cold_and_arena_backing_after_guards() -> TestResult {
    let target = Arc::new(LpgStore::new()?);
    let old = target.create_node(&["Frozen"]);
    target.set_node_property(old, "value", Value::from("cold"));
    target.sync_epoch(EpochId::new(2));
    assert!(target.freeze_epoch(EpochId::INITIAL) > 0);
    let arena = Arc::clone(&target.arena_allocator);
    let cold = Arc::clone(&target.epoch_store);
    let mut workspace = LpgReplacementWorkspace::new(Arc::clone(&target), image(&target)?)?;
    publish_replacement(&mut workspace, false)?;
    assert!(Arc::ptr_eq(&arena, &target.arena_allocator));
    assert!(Arc::ptr_eq(&cold, &target.epoch_store));
    assert!(target.get_node(old).is_none());
    assert!(
        target
            .get_node_at_epoch(NodeId::new(4), EpochId::new(3))
            .is_some()
    );
    assert!(target.freeze_epoch(EpochId::new(3)) > 0);
    assert!(
        target
            .get_node_at_epoch(NodeId::new(4), EpochId::new(3))
            .is_some()
    );
    Ok(())
}
