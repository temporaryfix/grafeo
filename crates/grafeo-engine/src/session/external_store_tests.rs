//! External transaction admission must never substitute the internal sentinel.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::lpg::{CompareOp, Edge, LpgStore, Node};
use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut, GraphStoreSearch};
use grafeo_core::statistics::Statistics;

use crate::{Config, GrafeoDB};

/// Implements the ordinary storage contract without opting into native commit.
struct ExternalOnly(Arc<LpgStore>);

macro_rules! forward {
    ($source:ident; $(fn $method:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)+) => {
        $(fn $method(&self, $($arg: $ty),*) $(-> $ret)? {
            $source::$method(self.0.as_ref(), $($arg),*)
        })+
    };
}

impl GraphStore for ExternalOnly {
    forward! { GraphStore;
        fn get_node(id: NodeId) -> Option<Node>;
        fn get_edge(id: EdgeId) -> Option<Edge>;
        fn get_node_versioned(id: NodeId, epoch: EpochId, tx: TransactionId) -> Option<Node>;
        fn get_edge_versioned(id: EdgeId, epoch: EpochId, tx: TransactionId) -> Option<Edge>;
        fn get_node_at_epoch(id: NodeId, epoch: EpochId) -> Option<Node>;
        fn get_edge_at_epoch(id: EdgeId, epoch: EpochId) -> Option<Edge>;
        fn get_node_property(id: NodeId, key: &PropertyKey) -> Option<Value>;
        fn get_edge_property(id: EdgeId, key: &PropertyKey) -> Option<Value>;
        fn get_node_property_batch(ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>>;
        fn get_nodes_properties_batch(ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>>;
        fn get_nodes_properties_selective_batch(ids: &[NodeId], keys: &[PropertyKey]) -> Vec<FxHashMap<PropertyKey, Value>>;
        fn get_edges_properties_selective_batch(ids: &[EdgeId], keys: &[PropertyKey]) -> Vec<FxHashMap<PropertyKey, Value>>;
        fn neighbors(id: NodeId, direction: Direction) -> Vec<NodeId>;
        fn edges_from(id: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)>;
        fn out_degree(id: NodeId) -> usize;
        fn in_degree(id: NodeId) -> usize;
        fn has_backward_adjacency() -> bool;
        fn node_ids() -> Vec<NodeId>;
        fn nodes_by_label(label: &str) -> Vec<NodeId>;
        fn node_count() -> usize;
        fn edge_count() -> usize;
        fn edge_type(id: EdgeId) -> Option<ArcStr>;
        fn find_nodes_by_property(key: &str, value: &Value) -> Vec<NodeId>;
        fn find_nodes_by_properties(conditions: &[(&str, Value)]) -> Vec<NodeId>;
        fn find_nodes_in_range(key: &str, min: Option<&Value>, max: Option<&Value>, min_inclusive: bool, max_inclusive: bool) -> Vec<NodeId>;
        fn node_property_might_match(key: &PropertyKey, op: CompareOp, value: &Value) -> bool;
        fn edge_property_might_match(key: &PropertyKey, op: CompareOp, value: &Value) -> bool;
        fn statistics() -> Arc<Statistics>;
        fn estimate_label_cardinality(label: &str) -> f64;
        fn estimate_avg_degree(edge_type: &str, outgoing: bool) -> f64;
        fn current_epoch() -> EpochId;
    }
}

impl GraphStoreSearch for ExternalOnly {}

impl GraphStoreMut for ExternalOnly {
    forward! { GraphStoreMut;
        fn create_node(labels: &[&str]) -> NodeId;
        fn create_node_versioned(labels: &[&str], epoch: EpochId, tx: TransactionId) -> NodeId;
        fn create_edge(src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId;
        fn create_edge_versioned(src: NodeId, dst: NodeId, edge_type: &str, epoch: EpochId, tx: TransactionId) -> EdgeId;
        fn batch_create_edges(edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId>;
        fn delete_node(id: NodeId) -> bool;
        fn delete_node_versioned(id: NodeId, epoch: EpochId, tx: TransactionId) -> bool;
        fn delete_node_edges(id: NodeId);
        fn delete_edge(id: EdgeId) -> bool;
        fn delete_edge_versioned(id: EdgeId, epoch: EpochId, tx: TransactionId) -> bool;
        fn set_node_property(id: NodeId, key: &str, value: Value);
        fn set_edge_property(id: EdgeId, key: &str, value: Value);
        fn remove_node_property(id: NodeId, key: &str) -> Option<Value>;
        fn remove_edge_property(id: EdgeId, key: &str) -> Option<Value>;
        fn add_label(id: NodeId, label: &str) -> bool;
        fn remove_label(id: NodeId, label: &str) -> bool;
    }
}

#[test]
fn unsupported_external_transactions_reject_before_mutation_and_keep_reads_available() {
    let inner = Arc::new(LpgStore::new().unwrap());
    let old = inner.create_node(&["External"]);
    inner.set_node_property(old, "name", Value::from("existing"));
    let source: Arc<dyn GraphStoreMut> = Arc::new(ExternalOnly(Arc::clone(&inner)));
    assert!(Arc::clone(&source).lpg_commit_store().is_none());
    let db = GrafeoDB::with_store(source, Config::in_memory()).unwrap();
    let mut session = db.session();
    assert!(!Arc::ptr_eq(&session.store, &inner));
    assert_eq!(
        session
            .execute("MATCH (n:External) RETURN n.name")
            .unwrap()
            .row_count(),
        1
    );

    let error = session.begin_transaction().unwrap_err();
    assert!(
        matches!(error, Error::Transaction(TransactionError::InvalidState(message))
        if message == "external store does not support native LPG transactions")
    );
    assert!(session.current_transaction.lock().is_none());
    let error = session
        .execute("INSERT (:External {name: 'rejected'})")
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Transaction(TransactionError::InvalidState(_))
    ));
    assert!(session.current_transaction.lock().is_none());
    assert_eq!(inner.node_count(), 1);
    assert_eq!(
        db.session()
            .execute("MATCH (n:External) RETURN n.name")
            .unwrap()
            .rows,
        vec![vec![Value::from("existing")]]
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn external_layered_transactions_reject_before_mutation_across_overlay_replacement() {
    use grafeo_core::graph::compact::from_graph_store_preserving_ids;
    use grafeo_core::graph::compact::layered::LayeredStore;

    let source = LpgStore::new().unwrap();
    let first = source.create_node(&["Cold"]);
    let second = source.create_node(&["Cold"]);
    let edge = source.create_edge(first, second, "LINK");
    source.set_node_property(first, "name", Value::from("first"));
    source.set_node_property(second, "name", Value::from("second"));
    let base = from_graph_store_preserving_ids(&source).unwrap();
    let layered = Arc::new(LayeredStore::new(base, second.as_u64(), edge.as_u64()).unwrap());
    let original = layered.overlay_store();
    assert!(Arc::clone(&layered).lpg_commit_store().is_none());
    let db = GrafeoDB::with_store(
        Arc::clone(&layered) as Arc<dyn GraphStoreMut>,
        Config::in_memory(),
    )
    .unwrap();
    let session = db.session();
    assert!(!Arc::ptr_eq(&session.store, &original));

    // This Session already exists when an unrelated external caller replaces
    // O0. Retaining O0 alone is not an exclusion against this handoff.
    layered.merge_overlay_in_place().unwrap();
    let current = layered.overlay_store();
    assert!(!Arc::ptr_eq(&original, &current));
    assert_eq!(original.create_node(&["Retired"]), NodeId::INVALID);

    for mut reader in [session, db.session()] {
        // The manager reserves SYSTEM before any user transaction. Compare
        // exact state around each rejection, before unrelated reads can run.
        macro_rules! without_transaction_admission {
            ($operation:expr) => {{
                let before = (
                    db.transaction_manager.last_assigned_transaction_id(),
                    db.transaction_manager.active_count(),
                );
                let result = $operation;
                assert_eq!(
                    (
                        db.transaction_manager.last_assigned_transaction_id(),
                        db.transaction_manager.active_count(),
                    ),
                    before,
                    "rejected operation must not allocate or retain a transaction: {}",
                    stringify!($operation)
                );
                assert!(reader.current_transaction.lock().is_none());
                result
            }};
        }

        let error = without_transaction_admission!(reader.begin_transaction()).unwrap_err();
        assert!(
            matches!(error, Error::Transaction(TransactionError::InvalidState(message))
            if message == "external store does not support native LPG transactions")
        );
        let error =
            without_transaction_admission!(reader.execute("INSERT (:Cold {name: 'rejected'})"))
                .unwrap_err();
        assert!(matches!(
            error,
            Error::Transaction(TransactionError::InvalidState(_))
        ));
        assert!(
            without_transaction_admission!(reader.set_node_property(
                first,
                "name",
                Value::from("rejected")
            ))
            .is_err()
        );
        assert!(!without_transaction_admission!(reader.delete_node(first)));
        assert!(!without_transaction_admission!(reader.delete_edge(edge)));
        assert_eq!(
            reader.get_node_property(first, "name"),
            Some(Value::from("first"))
        );
        assert!(reader.get_node(second).is_some());
        assert_eq!(
            reader
                .execute("MATCH (n:Cold) RETURN n.name")
                .unwrap()
                .row_count(),
            2
        );
    }
    assert_eq!(db.transaction_manager.active_count(), 0);
    assert_eq!(current.node_count(), 0);
    assert_eq!(current.edge_count(), 0);
    // Every rejection preserved the exact allocator state. A fresh probe's
    // structural queues must therefore remain absent in both overlays.
    let prospective = db.transaction_manager.begin();
    for target in [&original, &current] {
        assert!(target.pending_node_creates(prospective).is_empty());
        assert!(target.pending_edge_creates(prospective).is_empty());
        assert!(target.pending_node_deletes_peek(prospective).is_empty());
        assert!(target.pending_edge_deletes_peek(prospective).is_empty());
    }
    assert!(layered.pending_node_deletes_peek(prospective).is_empty());
    assert!(layered.pending_edge_deletes_peek(prospective).is_empty());
    db.transaction_manager.abort(prospective).unwrap();
}

#[cfg(feature = "cdc")]
#[test]
fn external_cdc_decorator_retains_the_same_native_commit_target() {
    let inner = Arc::new(LpgStore::new().unwrap());
    let source: Arc<dyn GraphStoreMut> = Arc::new(crate::database::cdc_store::CdcGraphStore::new(
        Arc::clone(&inner) as Arc<dyn GraphStoreMut>,
        Arc::new(crate::cdc::CdcLog::new()),
        Arc::clone(&inner),
        grafeo_common::types::GraphPath::root(),
    ));
    assert!(Arc::ptr_eq(
        &Arc::clone(&source).lpg_commit_store().unwrap(),
        &inner
    ));
    let db = GrafeoDB::with_store(source, Config::in_memory()).unwrap();
    db.session()
        .execute("INSERT (:External {name: 'committed'})")
        .unwrap();
    assert_eq!(
        db.session()
            .execute("MATCH (n:External) RETURN n.name")
            .unwrap()
            .rows,
        vec![vec![Value::from("committed")]]
    );
}
