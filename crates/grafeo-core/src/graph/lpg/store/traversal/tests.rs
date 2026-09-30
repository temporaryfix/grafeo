use super::*;
use std::sync::{Arc, mpsc};
use std::time::Duration;

fn assert_edge_buffer_transfer(width: usize, direction: Direction) {
    use crate::graph::GraphStore;
    use crate::index::adjacency::take_edge_buffer_capture;

    let store = LpgStore::new().unwrap();
    let hub = store.create_node(&[]);
    let peers: Vec<_> = (0..width / 2).map(|_| store.create_node(&[])).collect();
    let expected: Vec<_> = (0..width)
        .map(|index| {
            let peer = peers[index / 2];
            let edge = match direction {
                Direction::Outgoing => store.create_edge(hub, peer, "LINK"),
                Direction::Incoming => store.create_edge(peer, hub, "LINK"),
                Direction::Both => unreachable!("transfer control selects one adjacency"),
            };
            (peer, edge)
        })
        .collect();
    let adjacency = match direction {
        Direction::Outgoing => &store.forward_adj,
        Direction::Incoming => store.backward_adj.as_ref().unwrap(),
        Direction::Both => unreachable!("transfer control selects one adjacency"),
    };

    // Positive control observes the real adjacency result, with no allocator
    // replacement or synthetic work count. The hook must capture a live buffer.
    take_edge_buffer_capture();
    let control = adjacency.edges_from(hub);
    assert_eq!(control, expected);
    assert_eq!(
        take_edge_buffer_capture(),
        Some((control.as_ptr() as usize, width, control.capacity()))
    );
    drop(control);

    let result = GraphStore::edges_from(&store, hub, direction);
    let (buffer, length, capacity) = take_edge_buffer_capture().expect("actual adjacency call");
    assert_eq!(result, expected);
    assert_eq!(length, width);
    assert!(capacity >= width);
    // The old boxed-chain collection allocates its destination while the
    // adjacency buffer is still live. A direct transfer keeps this exact owner.
    assert_eq!(result.as_ptr() as usize, buffer, "{direction:?}, N={width}");
    assert_eq!(result.capacity(), capacity);
}

macro_rules! edge_buffer_transfer_test {
    ($name:ident, $width:expr, $direction:expr) => {
        #[test]
        fn $name() {
            assert_edge_buffer_transfer($width, $direction);
        }
    };
}

edge_buffer_transfer_test!(outgoing_edge_buffer_transfer_32, 32, Direction::Outgoing);
edge_buffer_transfer_test!(outgoing_edge_buffer_transfer_64, 64, Direction::Outgoing);
edge_buffer_transfer_test!(outgoing_edge_buffer_transfer_128, 128, Direction::Outgoing);
edge_buffer_transfer_test!(incoming_edge_buffer_transfer_32, 32, Direction::Incoming);
edge_buffer_transfer_test!(incoming_edge_buffer_transfer_64, 64, Direction::Incoming);
edge_buffer_transfer_test!(incoming_edge_buffer_transfer_128, 128, Direction::Incoming);

#[test]
fn edge_buffers_preserve_direction_order_parallel_self_loop_deleted_and_missing_pairs() {
    use crate::graph::GraphStore;

    let store = LpgStore::new().unwrap();
    let hub = store.create_node(&[]);
    let dst = store.create_node(&[]);
    let src = store.create_node(&[]);
    let first = store.create_edge(hub, dst, "LINK");
    let second = store.create_edge(hub, dst, "LINK");
    let incoming = store.create_edge(src, hub, "LINK");
    let self_loop = store.create_edge(hub, hub, "LINK");
    let deleted_out = store.create_edge(hub, dst, "LINK");
    let deleted_in = store.create_edge(src, hub, "LINK");
    assert!(store.delete_edge(deleted_out));
    assert!(store.delete_edge(deleted_in));
    let outgoing = vec![(dst, first), (dst, second), (hub, self_loop)];
    let incoming = vec![(src, incoming), (hub, self_loop)];
    let both: Vec<_> = outgoing.iter().chain(&incoming).copied().collect();
    for (direction, expected) in [
        (Direction::Outgoing, outgoing),
        (Direction::Incoming, incoming),
        (Direction::Both, both),
    ] {
        assert_eq!(GraphStore::edges_from(&store, hub, direction), expected);
        assert_eq!(
            store.edges_from(hub, direction).collect::<Vec<_>>(),
            expected
        );
        assert!(GraphStore::edges_from(&store, NodeId::new(u64::MAX), direction).is_empty());
    }
}

#[test]
fn edge_buffers_preserve_absent_backward_index() {
    use crate::graph::GraphStore;
    use crate::graph::lpg::LpgStoreConfig;

    let store = LpgStore::with_config(LpgStoreConfig {
        backward_edges: false,
        ..LpgStoreConfig::default()
    })
    .unwrap();
    let hub = store.create_node(&[]);
    let peer = store.create_node(&[]);
    let edge = store.create_edge(hub, peer, "LINK");
    assert!(GraphStore::edges_from(&store, hub, Direction::Incoming).is_empty());
    assert_eq!(
        GraphStore::edges_from(&store, hub, Direction::Both),
        vec![(peer, edge)]
    );
}

fn structural_writer_is_blocked(store: &LpgStore) -> bool {
    #[cfg(not(feature = "tiered-storage"))]
    {
        store.edges.try_write().is_none()
    }
    #[cfg(feature = "tiered-storage")]
    {
        store.edge_versions.try_write().is_none()
    }
}

#[test]
fn edge_type_scan_acquires_structure_before_the_directory() {
    let store = Arc::new(LpgStore::new().unwrap());
    let src = store.create_node(&[]);
    let dst = store.create_node(&[]);
    let edge = store.create_edge(src, dst, "MATCH");
    let directory = store.edge_type_to_id.write();
    let (pinned_tx, pinned_rx) = mpsc::channel();
    let reader_store = Arc::clone(&store);
    let reader = std::thread::spawn(move || {
        TYPE_SCAN_STRUCTURE_PINNED.with_borrow_mut(|hook| {
            *hook = Some(Box::new(move || {
                pinned_tx.send(()).unwrap();
            }));
        });
        reader_store
            .edges_with_type("MATCH")
            .map(|edge| edge.id)
            .collect::<Vec<_>>()
    });
    // A directory-first reader cannot reach the structural-pinned event while
    // this directory writer is held. Always release it before joining, even
    // after timeout, so the regression fails without stranding a deadlock.
    let pinned = pinned_rx.recv_timeout(Duration::from_secs(3)).is_ok();
    let structural_reader_held = structural_writer_is_blocked(&store);
    drop(directory);
    let result = reader.join().unwrap();
    assert!(
        pinned,
        "reader must pin structure before waiting for the directory"
    );
    assert!(
        structural_reader_held,
        "type lookup must retain its structural cut"
    );
    assert_eq!(result, vec![edge]);
}

#[test]
fn edge_type_scan_requalifies_type_after_clear_during_capacity_retry() {
    let store = Arc::new(LpgStore::new().unwrap());
    let src = store.create_node(&[]);
    let dst = store.create_node(&[]);
    store.create_edge(src, dst, "WANTED");
    let original_type = store.edge_type_to_id.read()["WANTED"];
    let (sized_tx, sized_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let reader_store = Arc::clone(&store);
    let reader = std::thread::spawn(move || {
        TYPE_SCAN_AFTER_RESERVE.with_borrow_mut(|hook| {
            *hook = Some(Box::new(move |_| {
                sized_tx.send(()).unwrap();
                resume_rx.recv_timeout(Duration::from_secs(3)).unwrap();
            }));
        });
        reader_store
            .edges_with_type("WANTED")
            .map(|edge| (edge.id, edge.edge_type))
            .collect::<Vec<_>>()
    });
    let sized = sized_rx.recv_timeout(Duration::from_secs(3)).is_ok();
    store.clear();
    let src = store.create_node(&[]);
    let dst = store.create_node(&[]);
    let other = store.create_edge(src, dst, "OTHER");
    let wanted = store.create_edge(src, dst, "WANTED");
    let reused = store.edge_type_to_id.read()["OTHER"];
    let wanted_type = store.edge_type_to_id.read()["WANTED"];
    resume_tx.send(()).unwrap();
    let result = reader.join().unwrap();
    assert!(
        sized,
        "fixture must cross the actual unlocked reservation seam"
    );
    assert_eq!(original_type, reused);
    assert_ne!(wanted_type, original_type);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, wanted);
    assert_eq!(result[0].1.as_str(), "WANTED");
    assert_ne!(result[0].0, other);
}

#[test]
fn edge_type_scan_scratch_tracks_matches_not_retained_lifetimes() {
    use std::cell::Cell;
    use std::rc::Rc;

    let store = LpgStore::new().unwrap();
    let src = store.create_node(&[]);
    let dst = store.create_node(&[]);
    let visible = store.create_edge(src, dst, "SPARSE");
    for _ in 0..128 {
        store.create_edge(src, dst, "OTHER");
        let deleted = store.create_edge(src, dst, "SPARSE");
        assert!(store.delete_edge(deleted));
        store.create_edge_versioned(
            src,
            dst,
            "SPARSE",
            EpochId::INITIAL,
            TransactionId::new(902),
        );
    }
    let reserved = Rc::new(Cell::new(0));
    let observed = Rc::clone(&reserved);
    TYPE_SCAN_AFTER_RESERVE.with_borrow_mut(|hook| {
        *hook = Some(Box::new(move |capacity| observed.set(capacity)));
    });
    let result = store
        .edges_with_type("SPARSE")
        .map(|edge| edge.id)
        .collect::<Vec<_>>();
    assert_eq!(result, vec![visible]);
    // Permit Vec's small-allocation rounding, but not space for hundreds of
    // unrelated, deleted, or foreign-pending retained edge identities.
    assert!((1..128).contains(&reserved.get()));

    // A known type with no qualifying edges also needs no scratch allocation.
    assert!(store.delete_edge(visible));
    crate::allocation_test::start();
    let empty = store.edges_with_type("SPARSE");
    let allocation = crate::allocation_test::stop();
    assert_eq!(allocation, crate::allocation_test::Counts::default());
    assert_eq!(empty.count(), 0);
}

#[test]
fn edge_type_iterator_keeps_visible_id_selection_and_lazy_materialization() {
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&[]);
    let dst = store.create_node(&[]);
    let visible = store.create_edge(src, dst, "MATCH");
    let deleted = store.create_edge(src, dst, "MATCH");
    assert!(store.delete_edge(deleted));
    store.create_edge_versioned(src, dst, "MATCH", EpochId::INITIAL, TransactionId::new(901));
    store.create_edge(src, dst, "OTHER");
    assert_eq!(
        store
            .edges_with_type("MATCH")
            .map(|edge| edge.id)
            .collect::<Vec<_>>(),
        vec![visible]
    );
    let iterator = store.edges_with_type("MATCH");
    assert!(store.edge_type_to_id.try_write().is_some());
    assert!(!structural_writer_is_blocked(&store));
    // Preserve the existing lazy API: values are materialized on iteration,
    // not frozen together with the initially selected IDs.
    store.set_edge_property(visible, "late", 7i64.into());
    let rows = iterator.collect::<Vec<_>>();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0]
            .properties
            .get(&grafeo_common::types::PropertyKey::new("late")),
        Some(&7i64.into())
    );
    crate::allocation_test::start();
    let missing = store.edges_with_type("ABSENT");
    let allocation = crate::allocation_test::stop();
    assert_eq!(
        allocation,
        crate::allocation_test::Counts::default(),
        "unknown type needs no ID scratch allocation"
    );
    assert_eq!(missing.count(), 0);
}

#[cfg(feature = "tiered-storage")]
#[test]
fn edge_type_scan_reads_valid_cold_records_without_decode_allocation() {
    use grafeo_common::mvcc::{ColdVersionRef, OptionalEpochId, VersionRef};
    let store = LpgStore::new().unwrap();
    let src = store.create_node(&[]);
    let dst = store.create_node(&[]);
    let edge = store.create_edge(src, dst, "COLD");
    let record = {
        let versions = store.edge_versions.read();
        let reference = versions[&edge].visible_at(EpochId::INITIAL).unwrap();
        store.read_edge_record(&reference).unwrap()
    };
    let (_, entries) =
        store
            .epoch_store
            .freeze_epoch(EpochId::INITIAL, vec![], vec![(edge.as_u64(), record)]);
    let entry = entries.first().unwrap();
    let cold = ColdVersionRef {
        epoch: EpochId::INITIAL,
        block_offset: entry.offset,
        length: entry.length,
        created_by: TransactionId::SYSTEM,
        deleted_epoch: OptionalEpochId::NONE,
        deleted_by: None,
    };
    store
        .edge_versions
        .write()
        .get_mut(&edge)
        .unwrap()
        .freeze_epoch(EpochId::INITIAL, std::iter::once(cold));
    crate::allocation_test::start();
    let decoded = store.read_edge_record(&VersionRef::Cold(cold));
    let allocation = crate::allocation_test::stop();
    assert_eq!(allocation, crate::allocation_test::Counts::default());
    assert_eq!(decoded.unwrap().id, edge);
    assert_eq!(
        store
            .edges_with_type("COLD")
            .map(|edge| edge.id)
            .collect::<Vec<_>>(),
        vec![edge]
    );
}
