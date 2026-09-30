//! Store-scoped authority for framed mutations.
//!
//! WAL-backed stores are sealed before they are exposed publicly. Engine-owned
//! mutation paths enter an authority scope while applying a framed transaction;
//! raw callers do not possess that database's authority and therefore cannot
//! mutate its stores outside the WAL/transaction boundary.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SCOPE: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static HELD_SCOPES: RefCell<Vec<NonZeroU64>> = const { RefCell::new(Vec::new()) };
}

/// Authority for one database's sealed stores.
///
/// The scope identifier is intentionally private. Creating another authority
/// cannot authorize writes to an already-sealed store because every authority
/// receives a distinct process-local scope.
#[derive(Debug)]
pub struct WriteAuthority {
    scope: NonZeroU64,
}

impl WriteAuthority {
    /// Creates a fresh database-scoped mutation authority.
    ///
    /// # Panics
    ///
    /// Panics only if the process exhausts all non-zero 64-bit authority scope
    /// identifiers.
    #[must_use]
    pub fn new() -> Self {
        let id = NEXT_SCOPE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .unwrap_or_else(|_| panic!("Grafeo write-authority scope space exhausted"));
        let scope = NonZeroU64::new(id).expect("write-authority counter starts non-zero");
        Self { scope }
    }

    #[cfg_attr(
        not(any(feature = "lpg", feature = "triple-store")),
        allow(
            dead_code,
            reason = "minimal core profiles keep authority construction and unwind tests but compile no sealed graph store that reads the private scope"
        )
    )]
    pub(crate) const fn scope(&self) -> NonZeroU64 {
        self.scope
    }
}

impl Default for WriteAuthority {
    fn default() -> Self {
        Self::new()
    }
}

struct ScopeGuard {
    scope: NonZeroU64,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        HELD_SCOPES.with(|held| {
            let popped = held.borrow_mut().pop();
            debug_assert_eq!(popped, Some(self.scope));
        });
    }
}

/// Runs `f` while this thread holds `authority`.
///
/// The RAII guard makes nested scopes unwind-safe: a panic cannot leave later
/// raw writes authorized on the thread.
pub fn with_authority<T>(authority: &WriteAuthority, f: impl FnOnce() -> T) -> T {
    HELD_SCOPES.with(|held| held.borrow_mut().push(authority.scope));
    let _guard = ScopeGuard {
        scope: authority.scope,
    };
    f()
}

#[cfg_attr(
    not(any(
        feature = "lpg",
        feature = "triple-store",
        feature = "vector-index",
        feature = "text-index"
    )),
    allow(
        dead_code,
        reason = "minimal core profiles retain unwind-safe authority scopes without compiling an authority-gated store or index"
    )
)]
pub(crate) fn is_held(scope: NonZeroU64) -> bool {
    HELD_SCOPES.with(|held| held.borrow().contains(&scope))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_is_scope_specific_and_nested() {
        let a = WriteAuthority::new();
        let b = WriteAuthority::new();
        assert!(!is_held(a.scope()));
        with_authority(&a, || {
            assert!(is_held(a.scope()));
            assert!(!is_held(b.scope()));
            with_authority(&b, || {
                assert!(is_held(a.scope()));
                assert!(is_held(b.scope()));
            });
            assert!(is_held(a.scope()));
            assert!(!is_held(b.scope()));
        });
        assert!(!is_held(a.scope()));
    }

    #[test]
    fn authority_is_released_during_unwind() {
        let authority = WriteAuthority::new();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&authority, || panic!("injected panic"));
        }));
        assert!(!is_held(authority.scope()));
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn sealed_stores_accept_only_their_own_authority() {
        use crate::execution::operators::{ReadTracker, SharedReadTracker};
        use crate::graph::lpg::LpgStore;
        use grafeo_common::types::{EdgeId, NodeId, TransactionId};
        use parking_lot::Mutex;
        use std::sync::Arc;

        struct ReadSpy(Mutex<Vec<NodeId>>);

        impl ReadTracker for ReadSpy {
            fn record_node_read(&self, _transaction_id: TransactionId, node_id: NodeId) {
                self.0.lock().push(node_id);
            }

            fn record_edge_read(&self, _transaction_id: TransactionId, _edge_id: EdgeId) {}
        }

        let authority_a = WriteAuthority::new();
        let authority_b = WriteAuthority::new();
        let store_a = LpgStore::new().unwrap();
        let store_b = LpgStore::new().unwrap();
        assert!(store_a.seal_unframed_writes(&authority_a));
        assert!(store_b.seal_unframed_writes(&authority_b));

        assert!(!store_a.create_node(&["raw"]).is_valid());
        assert!(
            !store_a
                .create_node_with_props(&["raw"], [("secret", 1_i64)])
                .is_valid(),
            "compound node creation must fail before touching properties or indexes"
        );
        let (src, dst) = with_authority(&authority_a, || {
            (store_a.create_node(&["src"]), store_a.create_node(&["dst"]))
        });
        assert!(
            !store_a
                .create_edge_with_props(src, dst, "RAW", [("secret", 1_i64)])
                .is_valid(),
            "compound edge creation must fail before touching properties"
        );
        with_authority(&authority_a, || {
            assert!(store_a.create_node(&["owned"]).is_valid());
            assert!(
                !store_b.create_node(&["wrong-scope"]).is_valid(),
                "authority A must never authorize store B"
            );
        });
        with_authority(&authority_b, || {
            assert!(store_b.create_node(&["owned"]).is_valid());
        });

        let transaction_id = TransactionId::new(7);
        let owner_spy = Arc::new(ReadSpy(Mutex::new(Vec::new())));
        let foreign_spy = Arc::new(ReadSpy(Mutex::new(Vec::new())));

        store_a.register_read_tracker(
            transaction_id,
            Arc::clone(&foreign_spy) as SharedReadTracker,
        );
        store_a.record_read_node(transaction_id, NodeId::new(1));
        assert!(foreign_spy.0.lock().is_empty());

        with_authority(&authority_a, || {
            store_a
                .register_read_tracker(transaction_id, Arc::clone(&owner_spy) as SharedReadTracker);
        });
        with_authority(&authority_b, || {
            store_a.register_read_tracker(
                transaction_id,
                Arc::clone(&foreign_spy) as SharedReadTracker,
            );
            store_a.unregister_read_tracker(transaction_id);
        });
        store_a.record_read_node(transaction_id, NodeId::new(2));
        assert_eq!(*owner_spy.0.lock(), vec![NodeId::new(2)]);
        assert!(foreign_spy.0.lock().is_empty());
    }
}
