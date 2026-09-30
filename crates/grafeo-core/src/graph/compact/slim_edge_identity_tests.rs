//! Original identities must survive the no-property current-CSR slim.

use super::{CompactStoreBuilder, id, layered::LayeredStore};
use crate::graph::lpg::LpgStore;
use crate::graph::traits::GraphStore;
use grafeo_common::types::{EdgeId, EpochId, EpochInterval, NodeId, PropertyKey, Value};
use std::sync::Arc;

fn epoch(value: u64) -> EpochId {
    EpochId::new(value)
}

fn compact_history_fixture(property_on_sparse: bool, mixed: bool) -> LayeredStore {
    let overlay = Arc::new(LpgStore::new().unwrap());
    for (id, label) in [(11, "A"), (29, "B")] {
        overlay
            .restore_node_history_exact(
                NodeId::new(id),
                &[(epoch(1), None)],
                &[(epoch(1), vec![label.into()])],
            )
            .unwrap();
    }
    for (id, edge_type, lifetimes) in [
        (41, "T", vec![(epoch(3), Some(epoch(5)))]),
        (
            509,
            "T",
            vec![(epoch(7), Some(epoch(11))), (epoch(15), None)],
        ),
    ] {
        overlay
            .restore_edge_history_exact(
                EdgeId::new(id),
                NodeId::new(11),
                NodeId::new(29),
                edge_type,
                &lifetimes,
            )
            .unwrap();
    }
    if property_on_sparse {
        overlay.set_edge_property_at_epoch(EdgeId::new(509), "weight", Value::Int64(8), epoch(15));
    }
    if mixed {
        overlay
            .restore_edge_history_exact(
                EdgeId::new(701),
                NodeId::new(11),
                NodeId::new(29),
                "WEIGHTED",
                &[(epoch(9), None)],
            )
            .unwrap();
        overlay.set_edge_property_at_epoch(EdgeId::new(701), "weight", Value::Int64(12), epoch(9));
    }
    overlay.sync_epoch(epoch(20));
    let empty = CompactStoreBuilder::new().build().unwrap();
    let layered = LayeredStore::with_overlay(Arc::new(empty), overlay).unwrap();
    layered.merge_overlay_temporal().unwrap();
    layered
}

#[test]
fn slim_live_inventory_preserves_sparse_ids_without_duplicate_reverse_map() {
    let layered = compact_history_fixture(false, true);
    let base = layered.base_store_arc();
    let table = base.rel_table("T").unwrap();
    assert_eq!(table.open_edge_ids(), vec![EdgeId::new(509)]);
    assert!(!table.current_from_packed());
    let table_index = base
        .rel_tables_by_id
        .iter()
        .position(|table| table.edge_type().as_str() == "T")
        .unwrap();
    assert!(base.edge_offset_to_id.as_ref().unwrap()[table_index].is_empty());
    let mut live = base.live_original_edge_ids();
    live.sort_unstable();
    assert_eq!(live, vec![EdgeId::new(509), EdgeId::new(701)]);
}

#[test]
fn slim_structural_rows_preserve_sparse_open_and_closed_lifetimes() {
    let layered = compact_history_fixture(false, true);
    assert_eq!(
        layered.base_store_arc().structural_edge_rows(),
        vec![
            (EdgeId::new(41), EpochInterval::closed(epoch(3), epoch(5))),
            (EdgeId::new(509), EpochInterval::closed(epoch(7), epoch(11))),
            (EdgeId::new(509), EpochInterval::open(epoch(15))),
            (EdgeId::new(701), EpochInterval::open(epoch(9))),
        ]
    );
}

fn assert_sparse_history(layered: &LayeredStore, weighted: bool) {
    let history = layered.complete_edge_history(EdgeId::new(509));
    assert_eq!(history.len(), 2);
    assert_eq!((history[0].0, history[0].1), (epoch(15), None));
    assert_eq!((history[1].0, history[1].1), (epoch(7), Some(epoch(11))));
    for (_, _, edge) in &history {
        assert_eq!(edge.id, EdgeId::new(509));
        assert_eq!((edge.src, edge.dst), (NodeId::new(11), NodeId::new(29)));
        assert_eq!(edge.edge_type.as_str(), "T");
    }
    assert!(
        layered
            .get_edge_at_epoch(EdgeId::new(509), epoch(6))
            .is_none()
    );
    assert!(
        layered
            .get_edge_at_epoch(EdgeId::new(509), epoch(8))
            .is_some()
    );
    assert!(
        layered
            .get_edge_at_epoch(EdgeId::new(509), epoch(12))
            .is_none()
    );
    assert!(
        layered
            .get_edge_at_epoch(EdgeId::new(509), epoch(20))
            .is_some()
    );
    if weighted {
        assert_eq!(
            history[0].2.properties.get(&PropertyKey::new("weight")),
            Some(&Value::Int64(8))
        );
    }
}

#[test]
fn mixed_slim_complete_histories_survive_recompact() {
    let layered = compact_history_fixture(false, true);
    for _ in 0..2 {
        assert_sparse_history(&layered, false);
        let histories = layered.complete_edge_histories();
        assert_eq!(
            histories.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![EdgeId::new(41), EdgeId::new(509), EdgeId::new(701)]
        );
        assert_eq!(histories[0].1.lifetimes.len(), 1);
        assert_eq!(histories[0].1.lifetimes[0].created, epoch(3));
        assert_eq!(histories[0].1.lifetimes[0].deleted, Some(epoch(5)));
        assert_eq!(layered.edge_count(), 2);
        layered.merge_overlay_temporal().unwrap();
    }
}

#[test]
fn property_bearing_reverse_map_keeps_original_identities() {
    let layered = compact_history_fixture(true, false);
    for _ in 0..2 {
        let base = layered.base_store_arc();
        assert!(base.rel_table("T").unwrap().open_edge_ids().is_empty());
        assert_eq!(base.live_original_edge_ids(), vec![EdgeId::new(509)]);
        assert_sparse_history(&layered, true);
        assert_eq!(base.structural_edge_rows().len(), 3);
        layered.merge_overlay_temporal().unwrap();
    }
}

#[test]
fn non_preserving_current_csr_keeps_encoded_identities() {
    let base = CompactStoreBuilder::new()
        .node_table("A", |table| table.column_dict("name", &["a", "b"]))
        .rel_table("T", "A", "A", |table| table.edges([(0, 1), (1, 0)]))
        .build()
        .unwrap();
    assert!(base.edge_id_map.is_none());
    let ids = vec![id::encode_edge_id(0, 0), id::encode_edge_id(0, 1)];
    assert_eq!(base.live_original_edge_ids(), ids);
    assert_eq!(
        base.structural_edge_rows(),
        vec![
            (ids[0], EpochInterval::open(EpochId::INITIAL)),
            (ids[1], EpochInterval::open(EpochId::INITIAL)),
        ]
    );
    assert_eq!(base.get_edge(ids[0]).unwrap().src, id::encode_node_id(0, 0));
    assert_eq!(base.get_edge(ids[1]).unwrap().src, id::encode_node_id(0, 1));
}
