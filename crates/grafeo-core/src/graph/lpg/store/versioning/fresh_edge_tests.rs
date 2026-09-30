use super::LpgStore;
use crate::graph::Direction;
use crate::graph::lpg::LpgStoreSection;
use crate::graph::write_permit::{WriteAuthority, with_authority};
use arcstr::ArcStr;
use grafeo_common::storage::Section;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::ErrorCode;
use std::sync::{Arc, Barrier};

#[derive(Debug, PartialEq)]
struct EdgeState {
    id: EdgeId,
    lifetimes: Vec<(EpochId, Option<EpochId>, NodeId, NodeId, ArcStr)>,
    properties: Vec<(PropertyKey, Vec<(EpochId, Value)>)>,
}

#[derive(Debug, PartialEq)]
struct GraphState {
    encoded: Option<Result<Vec<u8>, String>>,
    nodes: usize,
    edges: usize,
    floors: (u64, u64),
    types: Vec<String>,
    type_counts: Vec<i64>,
    identities: Vec<EdgeState>,
    outgoing: Vec<Vec<(NodeId, EdgeId)>>,
    incoming: Vec<Vec<(NodeId, EdgeId)>>,
}

fn state(store: &Arc<LpgStore>, nodes: &[NodeId]) -> GraphState {
    let mut state = state_fields(store, nodes);
    // Pending/future-born fixtures retain encoder refusal as well as explicit
    // fields; an encoder error alone is not proof that state stayed unchanged.
    state.encoded = Some(
        LpgStoreSection::new(Arc::clone(store))
            .serialize()
            .map_err(|error| error.to_string()),
    );
    state
}

fn state_fields(store: &Arc<LpgStore>, nodes: &[NodeId]) -> GraphState {
    let mut ids = store.all_known_edge_ids();
    ids.sort_unstable();
    let identities = ids
        .into_iter()
        .map(|id| EdgeState {
            id,
            lifetimes: store
                .get_edge_history(id)
                .into_iter()
                .map(|(created, deleted, edge)| {
                    (created, deleted, edge.src, edge.dst, edge.edge_type)
                })
                .collect(),
            properties: store.edge_property_history(id),
        })
        .collect();
    let adjacency = |direction| {
        nodes
            .iter()
            .map(|&node| {
                let mut edges: Vec<_> = store.edges_from(node, direction).collect();
                edges.sort_unstable();
                edges
            })
            .collect()
    };
    GraphState {
        encoded: None,
        nodes: store.node_count(),
        edges: store.edge_count(),
        floors: (store.next_node_id(), store.next_edge_id()),
        types: store.all_edge_types(),
        type_counts: store.edge_type_live_counts.read().clone(),
        identities,
        outgoing: adjacency(Direction::Outgoing),
        incoming: adjacency(Direction::Incoming),
    }
}

fn fixture() -> (Arc<LpgStore>, NodeId, NodeId) {
    let store = Arc::new(LpgStore::new().unwrap());
    store.sync_epoch(EpochId::new(5));
    let src = store.create_node(&["Source"]);
    let dst = store.create_node(&["Destination"]);
    let retained = store.create_edge(src, dst, "RETAINED");
    store.set_edge_property(retained, "keep", Value::Int64(7));
    assert!(LpgStoreSection::new(Arc::clone(&store)).serialize().is_ok());
    (store, src, dst)
}

#[test]
fn fresh_edge_rejects_invalid_identity_and_missing_endpoints_without_changes() {
    let (store, src, dst) = fixture();
    let missing = NodeId::new(99);
    let vacant = EdgeId::new(50);
    let before = state(&store, &[src, dst, missing, NodeId::INVALID]);
    for (id, from, to) in [
        (EdgeId::INVALID, src, dst),
        (vacant, missing, dst),
        (vacant, src, missing),
        (vacant, NodeId::INVALID, dst),
        (vacant, src, NodeId::INVALID),
    ] {
        let result = store.create_edge_with_id(id, from, to, "NEVER");
        assert_eq!(state(&store, &[src, dst, missing, NodeId::INVALID]), before);
        assert!(
            result.is_err(),
            "rejected fresh edge must not report success: {id:?}/{from:?}/{to:?}"
        );
        let expected = if !id.is_valid() || !from.is_valid() || !to.is_valid() {
            ErrorCode::InvalidInput
        } else {
            ErrorCode::NodeNotFound
        };
        assert_eq!(result.unwrap_err().error_code(), expected);
    }
}

#[test]
fn fresh_edge_rejects_occupied_live_closed_and_pending_identities() {
    let (store, src, dst) = fixture();
    let live = store.create_edge(src, dst, "LIVE");
    let closed = store.create_edge(src, dst, "CLOSED");
    assert!(store.delete_edge(closed));
    let pending = store.create_edge_versioned(
        src,
        dst,
        "PENDING",
        store.current_epoch(),
        TransactionId::new(7),
    );
    let before = state(&store, &[src, dst]);
    for id in [live, closed, pending] {
        let result = store.create_edge_with_id(id, src, dst, "NEVER");
        assert_eq!(state(&store, &[src, dst]), before);
        assert!(
            result.is_err(),
            "occupied identity must not report insertion: {id:?}"
        );
        assert_eq!(result.unwrap_err().error_code(), ErrorCode::InvalidInput);
    }
}

#[test]
fn fresh_edge_rejects_non_committed_live_endpoints_in_both_directions() {
    let (store, src, dst) = fixture();
    let closed = store.create_node(&["Closed"]);
    assert!(store.delete_node(closed));
    let pending =
        store.create_node_versioned(&["Pending"], store.current_epoch(), TransactionId::new(8));
    let future = store.create_node_versioned(&["Future"], EpochId::new(6), TransactionId::SYSTEM);
    let deleting = store.create_node(&["Deleting"]);
    #[cfg(not(feature = "tiered-storage"))]
    assert!(
        store
            .nodes
            .write()
            .get_mut(&deleting)
            .unwrap()
            .mark_deleted(EpochId::PENDING, TransactionId::new(9))
    );
    #[cfg(feature = "tiered-storage")]
    assert!(
        store
            .node_versions
            .write()
            .get_mut(&deleting)
            .unwrap()
            .mark_deleted(EpochId::PENDING, TransactionId::new(9))
    );
    let nodes = [src, dst, closed, pending, future, deleting];
    let before = state(&store, &nodes);
    for endpoint in [closed, pending, future, deleting] {
        for (from, to) in [(endpoint, dst), (src, endpoint)] {
            let result = store.create_edge_with_id(EdgeId::new(50), from, to, "NEVER");
            assert_eq!(state(&store, &nodes), before);
            assert!(
                result.is_err(),
                "non-committed-live endpoint must be rejected: {endpoint:?}"
            );
            let expected = if endpoint == pending || endpoint == deleting {
                ErrorCode::TransactionConflict
            } else {
                ErrorCode::NodeNotFound
            };
            assert_eq!(result.unwrap_err().error_code(), expected);
        }
    }
}

#[test]
fn fresh_edge_requires_exact_write_authority() {
    let (store, src, dst) = fixture();
    let owner = WriteAuthority::new();
    let foreign = WriteAuthority::new();
    assert!(store.seal_unframed_writes(&owner));
    let before = state(&store, &[src, dst]);
    let denied = store.create_edge_with_id(EdgeId::new(50), src, dst, "NEVER");
    let wrong = with_authority(&foreign, || {
        store.create_edge_with_id(EdgeId::new(50), src, dst, "NEVER")
    });
    assert_eq!(state(&store, &[src, dst]), before);
    assert!(denied.is_err());
    assert!(wrong.is_err());
    assert_eq!(
        denied.unwrap_err().error_code(),
        ErrorCode::TransactionInvalidState
    );
    assert_eq!(
        wrong.unwrap_err().error_code(),
        ErrorCode::TransactionInvalidState
    );
    with_authority(&owner, || {
        store.create_edge_with_id(EdgeId::new(50), src, dst, "OWNED")
    })
    .unwrap();
    assert!(store.get_edge(EdgeId::new(50)).is_some());
}

#[test]
fn rejected_fresh_edge_preserves_existing_transport_receipt() {
    let (store, src, dst) = fixture();
    let id = EdgeId::new(50);
    let receipt = store
        .create_transport_edge_with_id(id, src, dst, "TRANSPORT")
        .unwrap()
        .unwrap();
    assert!(store.is_transport_extract_edge(&receipt));
    let before = state(&store, &[src, dst]);
    let result = store.create_edge_with_id(id, dst, src, "NEVER");
    assert_eq!(state(&store, &[src, dst]), before);
    assert!(store.is_transport_extract_edge(&receipt));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().error_code(), ErrorCode::InvalidInput);
}

#[test]
fn fresh_edge_accepts_vacant_low_id_and_last_valid_successor() {
    let (store, src, dst) = fixture();
    store
        .create_edge_with_id(EdgeId::new(50), src, dst, "RECOVERED")
        .unwrap();
    assert_eq!(store.next_edge_id(), 51);
    store
        .create_edge_with_id(EdgeId::new(7), src, dst, "RECOVERED")
        .unwrap();
    assert_eq!(store.next_edge_id(), 51);
    let last = EdgeId::new(u64::MAX - 1);
    store
        .create_edge_with_id(last, src, dst, "RECOVERED")
        .unwrap();
    assert_eq!(store.next_edge_id(), u64::MAX);
    store
        .create_edge_with_id(EdgeId::new(8), src, dst, "RECOVERED")
        .unwrap();
    assert_eq!(store.next_edge_id(), u64::MAX);
    for id in [EdgeId::new(50), EdgeId::new(7), last, EdgeId::new(8)] {
        let edge = store.get_edge(id).unwrap();
        assert_eq!(
            (edge.src, edge.dst, edge.edge_type.as_str()),
            (src, dst, "RECOVERED")
        );
        assert_eq!(
            store
                .edges_from(src, Direction::Outgoing)
                .filter(|(_, found)| *found == id)
                .count(),
            1
        );
        assert_eq!(
            store
                .edges_from(dst, Direction::Incoming)
                .filter(|(_, found)| *found == id)
                .count(),
            1
        );
    }
    assert_eq!(store.edge_count(), 5);
}

#[test]
fn fresh_edge_rechecks_endpoint_after_preparation_pause() {
    let (store, src, dst) = fixture();
    let endpoint = store.create_node(&["Disappearing"]);
    let barrier = Arc::new(Barrier::new(2));
    *store.edge_publication_barrier.write() = Some(Arc::clone(&barrier));
    let creator = Arc::clone(&store);
    let worker = std::thread::spawn(move || {
        creator.create_edge_with_id(EdgeId::new(50), src, endpoint, "NEVER")
    });
    barrier.wait();
    assert!(store.delete_node(endpoint));
    // A coherent section capture must wait for the deliberately paused writer.
    // Compare all explicit fields here; qualify the durable image after join.
    let before = state_fields(&store, &[src, dst, endpoint]);
    barrier.wait();
    let result = worker.join().unwrap();
    assert_eq!(state_fields(&store, &[src, dst, endpoint]), before);
    assert!(LpgStoreSection::new(Arc::clone(&store)).serialize().is_ok());
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().error_code(), ErrorCode::NodeNotFound);
}
