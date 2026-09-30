//! Final index-build rows are maintenance reads, not transactional query reads.

#![cfg(feature = "lpg")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::{EdgeId, EpochId, LabelId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_core::execution::operators::ReadTracker;
use grafeo_core::graph::lpg::{LpgStore, Node};
use grafeo_core::graph::{GraphStore, GraphStoreMut};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type RowImage = (NodeId, Vec<String>, BTreeMap<PropertyKey, Value>);

#[derive(Default)]
struct ReadSpy(AtomicUsize);

impl ReadSpy {
    fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

impl ReadTracker for ReadSpy {
    fn record_node_read(&self, _: TransactionId, _: NodeId) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn record_edge_read(&self, _: TransactionId, _: EdgeId) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn record_label_name_predicate_read(&self, _: TransactionId, _: &str) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn record_label_predicate_read(&self, _: TransactionId, _: LabelId) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn record_lpg_dataset_read(&self, _: TransactionId) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn image(rows: &[Node]) -> Vec<RowImage> {
    assert!(rows.windows(2).all(|pair| pair[0].id < pair[1].id));
    rows.iter()
        .map(|node| {
            let mut labels = node
                .labels
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            labels.sort_unstable();
            (node.id, labels, node.properties_as_btree())
        })
        .collect()
}

fn seed(store: &LpgStore, name: &str) -> NodeId {
    let id = store.create_node(&["Doc"]);
    assert!(id.is_valid());
    store.set_node_property(id, "name", Value::from(name));
    id
}

#[test]
fn lpg_index_rows_use_final_changes_without_read_tracking_or_publication() -> TestResult {
    let store = LpgStore::new()?;
    let retained = seed(&store, "retained");
    let own_deleted = seed(&store, "own-delete");
    let foreign_deleted = seed(&store, "foreign-delete");
    store.set_node_property(retained, "remove", Value::Int64(1));
    store.set_node_property(retained, "value", Value::Int64(10));

    // A transaction that started at INITIAL must build at the publication
    // frontier, including this later committed row and property value.
    let frontier = EpochId::new(5);
    store.sync_epoch(frontier);
    let later = seed(&store, "later-committed");
    store.set_node_property(retained, "newer", Value::Int64(50));
    let committed = store.prepare_index_node_rows(frontier, None)?;
    assert_eq!(
        committed.iter().map(|node| node.id).collect::<Vec<_>>(),
        vec![retained, own_deleted, foreign_deleted, later]
    );

    let own = TransactionId::new(100);
    let foreign = TransactionId::new(101);
    store.set_node_property_buffered(retained, "value", Value::Int64(20), own);
    store.remove_node_property_buffered(retained, "value", own);
    store.set_node_property_buffered(retained, "value", Value::Int64(30), own);
    store.remove_node_property_buffered(retained, "remove", own);
    store.add_label_buffered(retained, "Final", own);
    store.remove_label_buffered(retained, "Final", own);
    store.add_label_buffered(retained, "Final", own);
    store.remove_label_buffered(retained, "Doc", own);
    store.set_node_property_buffered(retained, "newer", Value::Int64(999), foreign);
    store.add_label_buffered(retained, "Foreign", foreign);
    assert!(store.delete_node_versioned(own_deleted, frontier, own));
    assert!(store.delete_node_versioned(foreign_deleted, frontier, foreign));

    let own_created = store.create_node_versioned(&["Doc"], EpochId::INITIAL, own);
    assert!(own_created.is_valid());
    store.set_node_property_buffered(own_created, "name", Value::from("own-create"), own);
    let vanished = store.create_node_versioned(&["Doc"], EpochId::INITIAL, own);
    assert!(vanished.is_valid());
    assert!(store.delete_node_versioned(vanished, frontier, own));
    let foreign_created = store.create_node_versioned(&["Doc"], frontier, foreign);
    assert!(foreign_created.is_valid());
    store.set_node_property_buffered(foreign_created, "name", Value::from("foreign"), foreign);

    let own_spy = Arc::new(ReadSpy::default());
    let foreign_spy = Arc::new(ReadSpy::default());
    store.register_read_tracker(own, own_spy.clone());
    store.register_read_tracker(foreign, foreign_spy.clone());
    let pending_before = store.pending_node_creates(own);
    let deletes_before = store.pending_node_deletes_peek(own);

    let rows = store.prepare_index_node_rows(frontier, Some(own))?;
    let mut expected = committed.clone();
    expected.retain(|node| node.id != own_deleted);
    let changed = expected
        .iter_mut()
        .find(|node| node.id == retained)
        .expect("retained row");
    changed.set_property("value", Value::Int64(30));
    changed.remove_property("remove");
    changed.remove_label("Doc");
    changed.add_label("Final");
    let mut created = Node::with_labels(own_created, ["Doc"]);
    created.set_property("name", Value::from("own-create"));
    expected.push(created);
    assert_eq!(image(&rows), image(&expected));
    assert_eq!(
        image(&store.prepare_index_node_rows(frontier, None)?),
        image(&committed)
    );
    assert_eq!(
        image(&store.prepare_index_node_rows(frontier, Some(own))?),
        image(&expected)
    );
    assert_eq!(store.pending_node_creates(own), pending_before);
    assert_eq!(store.pending_node_deletes_peek(own), deletes_before);
    assert_eq!(own_spy.count(), 0);
    assert_eq!(foreign_spy.count(), 0);

    // The existing query accessor still records its three materialization
    // hooks; obtaining a maintenance image did not detach the tracker.
    let queried = store
        .get_node_versioned(retained, frontier, own)
        .expect("query row");
    assert_eq!(queried.get_property("value"), Some(&Value::Int64(30)));
    assert_eq!(own_spy.count(), 3);
    Ok(())
}

#[test]
fn lpg_index_rows_treat_null_as_committed_property_absence() -> TestResult {
    let store = LpgStore::new()?;
    let id = seed(&store, "retained");
    store.set_node_property(id, "cleared", Value::Int64(1));
    let tx = TransactionId::new(150);
    store.set_node_property_buffered(id, "cleared", Value::Null, tx);
    store.set_node_property_buffered(id, "never-present", Value::Null, tx);
    let prepared = store.prepare_index_node_rows(EpochId::INITIAL, Some(tx))?;
    assert_eq!(prepared.len(), 1);
    assert!(prepared[0].get_property("cleared").is_none());
    assert!(prepared[0].get_property("never-present").is_none());
    // Preparation leaves the committed column and overlay untouched.
    assert_eq!(
        store.get_node_property(id, &PropertyKey::new("cleared")),
        Some(Value::Int64(1))
    );
    store.finalize_entities_by_id(tx, EpochId::new(1), &[], &[]);
    store.apply_tx_overlay(tx);
    assert_eq!(
        image(&prepared),
        image(&store.prepare_index_node_rows(EpochId::new(1), None)?)
    );
    Ok(())
}

#[test]
fn lpg_index_rows_keep_named_graph_identity_separate() -> TestResult {
    let store = LpgStore::new()?;
    let root_id = seed(&store, "root");
    let graph = store.graph_or_create("named")?;
    let named_id = seed(&graph, "named");
    assert_eq!(
        root_id, named_id,
        "fixture deliberately reuses graph-local IDs"
    );
    let tx = TransactionId::new(200);
    graph.set_node_property_buffered(named_id, "name", Value::from("named-final"), tx);

    let root_rows = store.prepare_index_node_rows(EpochId::INITIAL, Some(tx))?;
    let named_rows = graph.prepare_index_node_rows(EpochId::INITIAL, Some(tx))?;
    assert_eq!(root_rows.len(), 1);
    assert_eq!(named_rows.len(), 1);
    assert_eq!(
        root_rows[0].get_property("name"),
        Some(&Value::from("root"))
    );
    assert_eq!(
        named_rows[0].get_property("name"),
        Some(&Value::from("named-final"))
    );
    Ok(())
}

#[test]
fn lpg_index_rows_reject_non_publication_context() -> TestResult {
    let store = LpgStore::new()?;
    for (epoch, tx) in [
        (EpochId::PENDING, None),
        (EpochId::INITIAL, Some(TransactionId::INVALID)),
        (EpochId::INITIAL, Some(TransactionId::SYSTEM)),
    ] {
        assert!(matches!(
            store.prepare_index_node_rows(epoch, tx),
            Err(Error::Transaction(TransactionError::InvalidState(_)))
        ));
    }
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn layered_index_rows_include_cold_and_overlay_only_final_state_without_tracking() -> TestResult {
    use grafeo_core::graph::compact::builder::from_graph_store_preserving_ids;
    use grafeo_core::graph::compact::layered::LayeredStore;

    let source = LpgStore::new()?;
    let cold = seed(&source, "cold");
    let promoted = seed(&source, "promoted");
    let own_deleted = seed(&source, "own-delete");
    let foreign_deleted = seed(&source, "foreign-delete");
    let base = from_graph_store_preserving_ids(&source)?;
    let layered = LayeredStore::new(base, foreign_deleted.as_u64(), 0)?;
    let overlay = layered.overlay_store();
    let frontier = EpochId::new(5);
    overlay.sync_epoch(frontier);
    // Direct Session creates use the overlay without a Layered dirty marker.
    let overlay_only = seed(&overlay, "overlay-only");
    overlay.set_node_property(overlay_only, "remove", Value::Int64(1));
    let committed = layered.prepare_index_node_rows(frontier, None)?;
    assert_eq!(
        committed.iter().map(|node| node.id).collect::<Vec<_>>(),
        vec![cold, promoted, own_deleted, foreign_deleted, overlay_only]
    );

    let own = TransactionId::new(300);
    let foreign = TransactionId::new(301);
    layered.set_node_property_buffered(promoted, "name", Value::from("promoted-final"), own);
    layered.add_label_buffered(promoted, "Final", own);
    layered.remove_label_buffered(promoted, "Doc", own);
    layered.set_node_property_buffered(overlay_only, "name", Value::from("transient"), own);
    layered.remove_node_property_buffered(overlay_only, "name", own);
    layered.set_node_property_buffered(overlay_only, "name", Value::from("overlay-final"), own);
    layered.remove_node_property_buffered(overlay_only, "remove", own);
    layered.add_label_buffered(overlay_only, "Final", own);
    assert!(layered.delete_node_versioned(own_deleted, frontier, own));
    assert!(layered.delete_node_versioned(foreign_deleted, frontier, foreign));
    let own_created = overlay.create_node_versioned(&["Doc"], frontier, own);
    assert!(own_created.is_valid());
    layered.set_node_property_buffered(own_created, "name", Value::from("own-create"), own);
    let vanished = overlay.create_node_versioned(&["Doc"], frontier, own);
    assert!(vanished.is_valid());
    assert!(layered.delete_node_versioned(vanished, frontier, own));
    let foreign_created = overlay.create_node_versioned(&["Doc"], frontier, foreign);
    assert!(foreign_created.is_valid());
    layered.set_node_property_buffered(foreign_created, "name", Value::from("foreign"), foreign);

    let own_spy = Arc::new(ReadSpy::default());
    let foreign_spy = Arc::new(ReadSpy::default());
    layered.register_read_tracker(own, own_spy.clone());
    layered.register_read_tracker(foreign, foreign_spy.clone());
    let pending_before = layered.pending_node_creates(own);
    let deletes_before = layered.pending_node_deletes_peek(own);
    let rows = layered.prepare_index_node_rows(frontier, Some(own))?;
    let mut expected = committed.clone();
    expected.retain(|node| node.id != own_deleted);
    let changed = expected
        .iter_mut()
        .find(|node| node.id == promoted)
        .expect("promoted row");
    changed.set_property("name", Value::from("promoted-final"));
    changed.remove_label("Doc");
    changed.add_label("Final");
    let changed = expected
        .iter_mut()
        .find(|node| node.id == overlay_only)
        .expect("overlay row");
    changed.set_property("name", Value::from("overlay-final"));
    changed.remove_property("remove");
    changed.add_label("Final");
    let mut created = Node::with_labels(own_created, ["Doc"]);
    created.set_property("name", Value::from("own-create"));
    expected.push(created);
    assert_eq!(image(&rows), image(&expected));
    assert_eq!(
        image(&layered.prepare_index_node_rows(frontier, None)?),
        image(&committed)
    );
    assert_eq!(
        image(&layered.prepare_index_node_rows(frontier, Some(own))?),
        image(&expected)
    );
    assert_eq!(layered.pending_node_creates(own), pending_before);
    assert_eq!(layered.pending_node_deletes_peek(own), deletes_before);
    assert_eq!(own_spy.count(), 0);
    assert_eq!(foreign_spy.count(), 0);

    let cold_query = layered
        .get_node_versioned(cold, frontier, own)
        .expect("cold query row");
    assert_eq!(cold_query.get_property("name"), Some(&Value::from("cold")));
    assert_eq!(own_spy.count(), 1);
    let overlay_query = layered
        .get_node_versioned(overlay_only, frontier, own)
        .expect("overlay query row");
    assert_eq!(
        overlay_query.get_property("name"),
        Some(&Value::from("overlay-final"))
    );
    assert_eq!(own_spy.count(), 4);
    Ok(())
}
