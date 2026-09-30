//! Regression: a transaction's commit must become visible atomically — a
//! concurrent reader must never observe a half-applied commit (torn read).
//!
//! A writer repeatedly commits `x = i` and `y = i` together, so any consistent
//! observation has `x == y`. Before the fix, `apply_tx_overlay` applied the two
//! property writes under separate lock acquisitions, so a concurrent reader's
//! `get_node` (a single `get_all` under one read lock) could slip in between and
//! observe `x = i, y = i-1`. `PropertyStorage::apply_ops` now applies a commit's
//! property changes under a single write lock, closing the window.
//!
//! Probabilistic but extremely reliable: pre-fix this failed on the very first
//! iteration (`x=1, y=0`).

#![cfg(feature = "lpg")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::thread;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

#[test]
fn commit_is_atomic_no_torn_read() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let n = db.create_node(&["N"]);
    db.set_node_property(n, "x", Value::Int64(0))
        .expect("set node property");
    db.set_node_property(n, "y", Value::Int64(0))
        .expect("set node property");

    let stop = Arc::new(AtomicBool::new(false));
    // (x, y) of the first torn read observed; (-1, -1) means none.
    let torn = Arc::new((AtomicI64::new(-1), AtomicI64::new(-1)));

    let writer = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut s = db.session();
            for i in 1..=20_000i64 {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                s.begin_transaction().unwrap();
                s.set_node_property(n, "x", Value::Int64(i)).unwrap();
                s.set_node_property(n, "y", Value::Int64(i)).unwrap();
                s.commit().unwrap();
            }
            stop.store(true, Ordering::Relaxed);
        })
    };

    let reader = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let torn = Arc::clone(&torn);
        thread::spawn(move || {
            let s = db.session();
            let kx = PropertyKey::new("x");
            let ky = PropertyKey::new("y");
            while !stop.load(Ordering::Relaxed) {
                if let Some(node) = s.get_node(n)
                    && let (Some(Value::Int64(xv)), Some(Value::Int64(yv))) = (
                        node.properties.get(&kx).cloned(),
                        node.properties.get(&ky).cloned(),
                    )
                    && xv != yv
                {
                    torn.0.store(xv, Ordering::Relaxed);
                    torn.1.store(yv, Ordering::Relaxed);
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
            }
        })
    };

    writer.join().unwrap();
    reader.join().unwrap();
    let (x, y) = (
        torn.0.load(Ordering::Relaxed),
        torn.1.load(Ordering::Relaxed),
    );
    assert_eq!(
        (x, y),
        (-1, -1),
        "TORN READ: observed x={x} y={y} (commit not atomic)"
    );
}
