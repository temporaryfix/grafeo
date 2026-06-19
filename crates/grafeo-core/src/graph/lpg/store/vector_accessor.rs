//! Snapshot-aware vector accessor for the HNSW vector index.
//!
//! [`SnapshotVectorAccessor`] reads each node's vector property **as-of a
//! snapshot** (epoch + optional transaction) via the property MVCC version
//! chain, so HNSW distance/rescore computations use the snapshot-consistent
//! value — correct under re-embedding.
//!
//! The soundness guarantee: if a node's embedding is updated between the
//! snapshot epoch and query time, the accessor returns the value that was
//! committed *at or before* `epoch`, not the newer value.  Callers that own
//! the writing transaction may also pass `tx` to obtain read-your-writes
//! behaviour before committing.

use std::sync::Arc;

use grafeo_common::types::{EpochId, NodeId, PropertyKey, TransactionId, Value};

use crate::index::vector::VectorAccessor;

use super::LpgStore;

/// Converts a [`Value`] to `Arc<[f32]>` if it is a `Value::Vector`.
///
/// This is the canonical conversion used by both the index-build path
/// (see `database/index.rs` line 142) and this accessor.  It returns
/// `None` for every other variant so callers can filter non-vector
/// properties cleanly with `?`.
// reason: production callers arrive in a later task (snapshot-search path);
// the function is exercised by the test module below.
#[allow(dead_code)]
#[inline]
pub(super) fn value_to_vector(value: &Value) -> Option<Arc<[f32]>> {
    match value {
        Value::Vector(v) => Some(Arc::clone(v)),
        _ => None,
    }
}

/// A [`VectorAccessor`] that reads each node's vector property
/// **as-of a snapshot** (epoch + tx) via the property MVCC version chain.
///
/// Construct one per query/search operation, tying it to the snapshot epoch
/// of the transaction or read that issued the search.
///
/// ```ignore
/// // SnapshotVectorAccessor is pub(crate); doc-compile example shown for
/// // illustration only.
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_common::types::{NodeId, PropertyKey};
///
/// let store = LpgStore::new().unwrap();
/// let epoch = store.current_epoch();
/// // Construct via the store's own crate only.
/// ```
// reason: production callers arrive in a later task (snapshot-search path);
// the struct is exercised by the test module below.
#[allow(dead_code)]
pub(crate) struct SnapshotVectorAccessor<'a> {
    /// The store to read from.
    pub(crate) store: &'a LpgStore,
    /// The property key that holds the vector data.
    pub(crate) property: PropertyKey,
    /// The snapshot epoch — reads return the value committed at or before
    /// this epoch (requires `temporal` feature; ignored otherwise).
    pub(crate) epoch: EpochId,
    /// Optional owning transaction for read-your-writes.  Pass `Some(tx)`
    /// when the caller holds an active transaction and wants to see its own
    /// uncommitted vector writes; pass `None` for pure snapshot reads.
    pub(crate) tx: Option<TransactionId>,
}

impl VectorAccessor for SnapshotVectorAccessor<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        let value =
            self.store
                .read_node_property_visible(id, &self.property, self.epoch, self.tx)?;
        value_to_vector(&value)
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;
    use grafeo_common::types::{EpochId, PropertyKey, TransactionId, Value};

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Commits `value` for `(node, key)` at commit epoch `c` (u64), using the
    /// same ordering the engine uses: advance the store epoch first (finalize),
    /// then promote the overlay (apply_tx_overlay stamps at current_epoch).
    fn commit_vector(
        store: &LpgStore,
        node: NodeId,
        key: &str,
        value: Value,
        tx: TransactionId,
        c: u64,
    ) {
        store.set_node_property_buffered(node, key, value, tx);
        store.sync_epoch(EpochId::new(c));
        store.apply_tx_overlay(tx);
    }

    // ── basic value_to_vector conversion ─────────────────────────────────────

    #[test]
    fn value_to_vector_extracts_vector_variant() {
        let v: Arc<[f32]> = vec![1.0_f32, 0.0, 0.0].into();
        let val = Value::Vector(Arc::clone(&v));
        let result = value_to_vector(&val);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), v.as_ref());
    }

    #[test]
    fn value_to_vector_returns_none_for_non_vector() {
        assert!(value_to_vector(&Value::Int64(42)).is_none());
        assert!(value_to_vector(&Value::String("hello".into())).is_none());
        assert!(value_to_vector(&Value::Bool(true)).is_none());
        assert!(value_to_vector(&Value::Null).is_none());
    }

    // ── missing node / wrong property type ──────────────────────────────────

    #[test]
    fn get_vector_returns_none_for_missing_node() {
        let store = LpgStore::new().unwrap();
        let epoch = store.current_epoch();
        let accessor = SnapshotVectorAccessor {
            store: &store,
            property: PropertyKey::new("embedding"),
            epoch,
            tx: None,
        };
        assert!(accessor.get_vector(NodeId::new(999)).is_none());
    }

    #[test]
    fn get_vector_returns_none_when_property_is_not_a_vector() {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Thing"]);
        store.set_node_property(node, "name", Value::String("hello".into()));
        let epoch = store.current_epoch();
        let accessor = SnapshotVectorAccessor {
            store: &store,
            property: PropertyKey::new("name"),
            epoch,
            tx: None,
        };
        assert!(accessor.get_vector(node).is_none());
    }

    // ── read-your-writes (tx overlay) ───────────────────────────────────────

    #[test]
    fn get_vector_honours_tx_overlay_read_your_writes() {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Entity"]);

        // Commit a base vector.
        let base: Arc<[f32]> = vec![1.0_f32, 0.0, 0.0].into();
        store.set_node_property(node, "embedding", Value::Vector(Arc::clone(&base)));

        let epoch = store.current_epoch();
        let tx = TransactionId::new(77);
        // Buffer an update in the tx overlay — not yet committed.
        let updated: Arc<[f32]> = vec![0.0_f32, 1.0, 0.0].into();
        store.set_node_property_buffered(
            node,
            "embedding",
            Value::Vector(Arc::clone(&updated)),
            tx,
        );

        // The writing transaction sees its own (uncommitted) value.
        let accessor_with_tx = SnapshotVectorAccessor {
            store: &store,
            property: PropertyKey::new("embedding"),
            epoch,
            tx: Some(tx),
        };
        assert_eq!(
            accessor_with_tx.get_vector(node).unwrap().as_ref(),
            updated.as_ref(),
            "writing tx must see its own buffered vector (read-your-writes)"
        );

        // A reader without a tx sees the committed base value.
        let accessor_no_tx = SnapshotVectorAccessor {
            store: &store,
            property: PropertyKey::new("embedding"),
            epoch,
            tx: None,
        };
        assert_eq!(
            accessor_no_tx.get_vector(node).unwrap().as_ref(),
            base.as_ref(),
            "other readers must not see uncommitted writes (no dirty reads)"
        );
    }

    // ── as-of-epoch isolation (requires `temporal` feature) ─────────────────
    //
    // Without `temporal`, `read_node_property_visible` ignores the epoch and
    // always returns the latest committed value, so epoch-differentiation
    // tests only make sense when temporal versioning is active.

    #[cfg(feature = "temporal")]
    #[test]
    fn snapshot_accessor_returns_as_of_epoch_vector() {
        let store = LpgStore::new().unwrap();
        let node = store.create_node(&["Entity"]);
        let key = "embedding";

        let v1: Arc<[f32]> = vec![1.0_f32, 0.0, 0.0].into();
        let v2: Arc<[f32]> = vec![0.0_f32, 1.0, 0.0].into();

        // C1 = epoch 1: commit v1
        commit_vector(
            &store,
            node,
            key,
            Value::Vector(Arc::clone(&v1)),
            TransactionId::new(1),
            1,
        );

        // C2 = epoch 2: commit v2 (re-embedding)
        commit_vector(
            &store,
            node,
            key,
            Value::Vector(Arc::clone(&v2)),
            TransactionId::new(2),
            2,
        );

        let prop_key = PropertyKey::new(key);

        // Snapshot at C1 — must see v1.
        let a1 = SnapshotVectorAccessor {
            store: &store,
            property: prop_key.clone(),
            epoch: EpochId::new(1),
            tx: None,
        };
        assert_eq!(
            a1.get_vector(node).unwrap().as_ref(),
            v1.as_ref(),
            "snapshot at epoch 1 must return v1"
        );

        // Snapshot at C2 — must see v2.
        let a2 = SnapshotVectorAccessor {
            store: &store,
            property: prop_key.clone(),
            epoch: EpochId::new(2),
            tx: None,
        };
        assert_eq!(
            a2.get_vector(node).unwrap().as_ref(),
            v2.as_ref(),
            "snapshot at epoch 2 must return v2"
        );

        // Snapshot at epoch 0 (before any commit) — None.
        let a0 = SnapshotVectorAccessor {
            store: &store,
            property: prop_key.clone(),
            epoch: EpochId::new(0),
            tx: None,
        };
        assert!(
            a0.get_vector(node).is_none(),
            "snapshot before any commit must return None"
        );
    }
}
