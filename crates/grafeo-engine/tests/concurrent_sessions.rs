//! Concurrent Sessions Integration Tests
//!
//! Tests for multi-session concurrent access patterns:
//! - Multiple sessions executing queries simultaneously
//! - Thread-safe shared database access
//! - Transaction isolation across sessions
//! - Race condition handling

#![cfg(feature = "lpg")]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_engine::{Config, GrafeoDB, Session};

fn is_classified_write_conflict(error: &impl std::fmt::Display) -> bool {
    let text = error.to_string();
    text.contains("GRAFEO-T001") || text.contains("Write-write conflict")
}

fn execute_stress_write(db: &GrafeoDB, query: &str) {
    const MAX_ATTEMPTS: usize = 128;
    for attempt in 0..MAX_ATTEMPTS {
        let session = db.session();
        let result = session.execute(query);
        assert!(
            !session.in_transaction(),
            "autocommit must close its transaction"
        );
        match result {
            Ok(_) => return,
            Err(Error::Transaction(TransactionError::WriteConflict(_))) => {
                // Publication can reject a busy rebind. A fresh attempt must
                // either complete this logical write or fail the bounded test.
                thread::sleep(std::time::Duration::from_micros(1_u64 << attempt.min(10)));
            }
            Err(error) => panic!("unexpected stress write failure: {error}"),
        }
    }
    panic!("stress write exhausted {MAX_ATTEMPTS} attempts: {query}");
}

fn unwind_rows(rows: Vec<Value>) -> HashMap<String, Value> {
    HashMap::from([("rows".into(), Value::List(rows.into()))])
}

fn row_map(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (PropertyKey::new(key), value))
            .collect::<BTreeMap<_, _>>()
            .into(),
    )
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

// ============================================================================
// Concurrent Session Access Tests
// ============================================================================

#[test]
fn test_concurrent_read_sessions() {
    // Multiple sessions reading simultaneously should not block.
    // 2 threads to stay within nextest timeout on 2-core CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    {
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session.execute("INSERT (:Person {name: 'Harm'})").unwrap();
    }

    let num_threads = 2;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();

                let session = db.session();
                let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();

                if result.row_count() == 3 {
                    success_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All concurrent reads should succeed"
    );
}

#[test]
fn test_concurrent_write_sessions() {
    // Multiple sessions writing to different entities should succeed.
    // Kept at 2 threads to stay within the nextest 60s timeout on
    // resource-constrained 2-core CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 2;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();

                let session = db.session();
                let query = format!("INSERT (:Thread{} {{id: {}}})", i, i);
                if session.execute(&query).is_ok() {
                    success_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All concurrent writes to different entities should succeed"
    );

    let session = db.session();
    for i in 0..num_threads {
        let query = format!("MATCH (n:Thread{}) RETURN n", i);
        let result = session.execute(&query).unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Node for thread {} should exist exactly once",
            i
        );
    }
}

#[test]
fn test_session_isolation_between_threads() {
    // Changes in one session's transaction should not be visible to other sessions
    // until committed
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Writer thread creates data in a transaction
    let db_clone = Arc::clone(&db);
    let writer_started = Arc::new(Barrier::new(2));
    let reader_check = Arc::new(Barrier::new(2));
    let writer_done = Arc::new(Barrier::new(2));

    let writer_started_clone = Arc::clone(&writer_started);
    let reader_check_clone = Arc::clone(&reader_check);
    let writer_done_clone = Arc::clone(&writer_done);

    let writer_handle = thread::spawn(move || {
        let mut session = db_clone.session();
        session.begin_transaction().unwrap();

        // Create a node within the transaction
        session
            .execute("INSERT (:IsolatedNode {secret: 'hidden'})")
            .unwrap();

        // Signal that writer has created the node
        writer_started_clone.wait();

        // Wait for reader to check
        reader_check_clone.wait();

        // Now commit
        session.commit().unwrap();

        // Signal done
        writer_done_clone.wait();
    });

    // Reader thread checks visibility
    let reader_handle = thread::spawn(move || {
        let session = db.session();

        // Wait for writer to create node (but not commit)
        writer_started.wait();

        // Check if we can see the node (we shouldn't - transaction not committed)
        // Note: This test checks the expected behavior when MVCC is fully integrated
        let result = session.execute("MATCH (n:IsolatedNode) RETURN n").unwrap();
        let before_commit_count = result.row_count();

        // Signal that reader has checked
        reader_check.wait();

        // Wait for writer to commit
        writer_done.wait();

        // Now we should see the node (after commit)
        let result = session.execute("MATCH (n:IsolatedNode) RETURN n").unwrap();
        let after_commit_count = result.row_count();

        (before_commit_count, after_commit_count)
    });

    writer_handle.join().expect("Writer thread panicked");
    let (before, after) = reader_handle.join().expect("Reader thread panicked");

    // After commit, the node should be visible
    assert_eq!(after, 1, "Node should be visible after commit");

    // Uncommitted data uses PENDING epoch, invisible to other sessions.
    assert_eq!(
        before, 0,
        "Dirty read prevented: uncommitted data is invisible to other sessions"
    );
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_many_sessions_rapid_creation() {
    // Creating many sessions rapidly should not cause issues.
    // 2 threads to stay within nextest timeout on 2-core CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 2;
    let sessions_per_thread = 20;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..sessions_per_thread {
                    let session = db.session();
                    // Just creating and dropping sessions
                    drop(session);
                }
                success_count.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All threads should complete without panic"
    );
}

#[test]
fn test_interleaved_transactions() {
    // Multiple sessions with interleaved transaction operations.
    // Kept lightweight (2 threads, 3 iterations) to avoid lock-contention
    // slowdowns on resource-constrained CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..2)
        .map(|thread_id| {
            let db = Arc::clone(&db);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                for i in 0..3 {
                    let mut session = db.session();

                    session.begin_transaction().unwrap();

                    let query =
                        format!("INSERT (:Work {{thread: {}, iteration: {}}})", thread_id, i);
                    let _ = session.execute(&query);

                    if i % 3 == 0 {
                        let _ = session.rollback();
                    } else {
                        let _ = session.commit();
                    }
                }

                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        2,
        "All threads should complete"
    );
}

// ============================================================================
// Session State Tests
// ============================================================================

#[test]
fn test_session_transaction_state_independence() {
    // Each session should maintain independent transaction state
    let db = GrafeoDB::new_in_memory();

    let mut session1 = db.session();
    let mut session2 = db.session();

    // Session 1 starts transaction
    session1.begin_transaction().unwrap();
    assert!(session1.in_transaction());
    assert!(!session2.in_transaction());

    // Session 2 starts its own transaction
    session2.begin_transaction().unwrap();
    assert!(session1.in_transaction());
    assert!(session2.in_transaction());

    // Session 1 commits
    session1.commit().unwrap();
    assert!(!session1.in_transaction());
    assert!(session2.in_transaction());

    // Session 2 rolls back
    session2.rollback().unwrap();
    assert!(!session1.in_transaction());
    assert!(!session2.in_transaction());
}

#[test]
fn test_session_auto_commit_independence() {
    // Auto-commit setting should be independent per session
    let db = GrafeoDB::new_in_memory();

    let mut session1 = db.session();
    let session2 = db.session();

    assert!(session1.auto_commit());
    assert!(session2.auto_commit());

    session1.set_auto_commit(false);

    assert!(!session1.auto_commit());
    assert!(session2.auto_commit());
}

// ============================================================================
// Database Shared State Tests
// ============================================================================

#[test]
fn test_sessions_share_committed_data() {
    // Data committed by one session should be visible to others
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    let session2 = db.session();

    // Session 1 creates and commits data
    session1.execute("INSERT (:Shared {key: 'value'})").unwrap();

    // Session 2 should see the data
    let result = session2.execute("MATCH (n:Shared) RETURN n.key").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Session 2 should see committed data from Session 1"
    );
}

#[test]
fn test_node_count_consistency() {
    // Node count should be consistent across sessions
    let db = GrafeoDB::new_in_memory();

    // Create nodes from multiple sessions
    for i in 0..10 {
        let session = db.session();
        let query = format!("INSERT (:CountTest{{id: {}}})", i);
        session.execute(&query).unwrap();
    }

    // Check count from a new session
    let session = db.session();
    let result = session.execute("MATCH (n:CountTest) RETURN n").unwrap();
    assert_eq!(result.row_count(), 10, "Should see all 10 nodes");
}

// ============================================================================
// Async Session Tests (using tokio)
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_concurrent_sessions() {
    // Kept lightweight (3 tasks) to avoid lock-contention slowdowns
    // on resource-constrained CI runners (2-core GitHub Actions).
    use tokio::task;

    let db = Arc::new(GrafeoDB::new_in_memory());
    let num_tasks = 3;

    // Spawn async tasks
    let mut handles = Vec::new();

    for i in 0..num_tasks {
        let db: Arc<GrafeoDB> = Arc::clone(&db);
        handles.push(task::spawn_blocking(move || {
            let session = db.session();
            let query = format!("INSERT (:AsyncNode {{id: {}}})", i);
            session.execute(&query).unwrap();
        }));
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.expect("Task panicked");
    }

    // Verify results
    let session = db.session();
    let result = session.execute("MATCH (n:AsyncNode) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_tasks,
        "All async nodes should exist"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_transaction_isolation() {
    use std::sync::atomic::AtomicBool;
    use tokio::task;

    let db = Arc::new(GrafeoDB::new_in_memory());
    let writer_committed = Arc::new(AtomicBool::new(false));

    // Writer task
    let db_writer: Arc<GrafeoDB> = Arc::clone(&db);
    let committed_flag = Arc::clone(&writer_committed);

    let writer = task::spawn_blocking(move || {
        let mut session = db_writer.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:AsyncIsolated {data: 'test'})")
            .unwrap();
        session.commit().unwrap();
        committed_flag.store(true, Ordering::Release);
    });

    // Reader task: waits for writer commit via atomic flag, no sleep
    let db_reader: Arc<GrafeoDB> = Arc::clone(&db);
    let reader_flag = Arc::clone(&writer_committed);

    let reader = task::spawn_blocking(move || {
        // Spin until writer has committed
        while !reader_flag.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }

        let session = db_reader.session();
        let result = session.execute("MATCH (n:AsyncIsolated) RETURN n").unwrap();
        result.row_count()
    });

    writer.await.expect("Writer task panicked");
    let count = reader.await.expect("Reader task panicked");

    assert_eq!(count, 1, "Should see committed data after writer completes");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_session_after_transaction_error() {
    // Session should be usable after a transaction error
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Try to commit without transaction (should error)
    let result = session.commit();
    assert!(result.is_err());

    // Session should still work
    session.begin_transaction().unwrap();
    session.execute("INSERT (:AfterError)").unwrap();
    session.commit().unwrap();

    let result = session.execute("MATCH (n:AfterError) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_multiple_sequential_transactions() {
    // Same session should handle multiple sequential transactions
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    for i in 0..5 {
        session.begin_transaction().unwrap();
        let query = format!("INSERT (:Sequential{{iteration: {}}})", i);
        session.execute(&query).unwrap();
        session.commit().unwrap();
    }

    let result = session.execute("MATCH (n:Sequential) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        5,
        "All 5 sequential transactions should have committed"
    );
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_writers() {
    // 8 threads each inserting 50 nodes simultaneously
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 8;
    let writes_per_thread = 50;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..writes_per_thread {
                    let query = format!("INSERT (:Stress {{thread: {tid}, seq: {i}}})");
                    execute_stress_write(&db, &query);
                }
                success_count.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(success_count.load(Ordering::Relaxed), num_threads);

    // Verify total node count
    let session = db.session();
    let result = session
        .execute("MATCH (n:Stress) RETURN n.thread, n.seq")
        .unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * writes_per_thread,
        "All nodes should be created"
    );
    let actual: BTreeSet<_> = result
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Int64(tid), Value::Int64(seq)) => (*tid, *seq),
            other => panic!("invalid stress write identity: {other:?}"),
        })
        .collect();
    let expected: BTreeSet<_> = (0..num_threads)
        .flat_map(|tid| {
            (0..writes_per_thread)
                .map(move |seq| (i64::try_from(tid).unwrap(), i64::try_from(seq).unwrap()))
        })
        .collect();
    assert_eq!(
        actual, expected,
        "every logical write must commit exactly once"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_reads_during_writes() {
    // Mixed workload: 4 writers + 8 readers operating simultaneously
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed initial data
    {
        let session = db.session();
        for i in 0..100 {
            session
                .execute(&format!("INSERT (:Item {{id: {i}}})"))
                .unwrap();
        }
    }

    let num_writers = 4;
    let num_readers = 8;
    let barrier = Arc::new(Barrier::new(num_writers + num_readers));
    let read_errors = Arc::new(AtomicUsize::new(0));
    let write_errors = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Writer threads
    for tid in 0..num_writers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let errors = Arc::clone(&write_errors);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..20 {
                let session = db.session();
                let id = 1000 + tid * 100 + i;
                if session
                    .execute(&format!("INSERT (:Written {{id: {id}}})"))
                    .is_err()
                {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // Reader threads
    for _ in 0..num_readers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let errors = Arc::clone(&read_errors);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..20 {
                let session = db.session();
                if session.execute("MATCH (n:Item) RETURN n.id").is_err() {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        read_errors.load(Ordering::Relaxed),
        0,
        "No read errors expected"
    );
    assert_eq!(
        write_errors.load(Ordering::Relaxed),
        0,
        "No write errors expected"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_transaction_conflicts() {
    // 4 threads with interleaved commit/rollback patterns
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 4;
    let iterations = 6;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..iterations {
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    let query = format!("INSERT (:TxNode {{thread: {tid}, iter: {i}}})");
                    let _ = session.execute(&query);

                    // Commit even iterations, rollback odd
                    if i % 2 == 0 {
                        let _ = session.commit();
                    } else {
                        let _ = session.rollback();
                    }
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // Only committed nodes (even iterations) should exist
    let session = db.session();
    let result = session.execute("MATCH (n:TxNode) RETURN n").unwrap();
    // Each thread commits 5 of 10 iterations (0, 2, 4, 6, 8)
    let expected = num_threads * (iterations / 2);
    assert_eq!(
        result.row_count(),
        expected,
        "Only committed transactions should be visible"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_epoch_pressure() {
    const MAX_ATTEMPTS: usize = 128;
    // 4 threads each running 8 sequential transactions, creates many epochs
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 4;
    let txns_per_thread = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..txns_per_thread {
                    let query = format!("INSERT (:Epoch {{thread: {tid}, txn: {i}}})");
                    let mut committed = false;
                    for attempt in 0..MAX_ATTEMPTS {
                        let mut session = db.session();
                        let result = session
                            .begin_transaction()
                            .and_then(|()| session.execute(&query).map(drop))
                            .and_then(|()| session.commit());
                        match result {
                            Ok(_) => {
                                assert!(!session.in_transaction());
                                committed = true;
                                break;
                            }
                            Err(Error::Transaction(TransactionError::WriteConflict(_))) => {
                                // A failed commit may already have aborted the transaction.
                                if session.in_transaction() {
                                    session
                                        .rollback()
                                        .expect("abort conflicted epoch transaction");
                                }
                                assert!(!session.in_transaction());
                                drop(session);
                                thread::sleep(std::time::Duration::from_micros(
                                    1_u64 << attempt.min(10),
                                ));
                            }
                            Err(error) => panic!("unexpected epoch transaction failure: {error}"),
                        }
                    }
                    assert!(
                        committed,
                        "epoch transaction exhausted {MAX_ATTEMPTS} attempts: {query}"
                    );
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // Every committed identity must appear exactly once, including after retries.
    let session = db.session();
    let result = session
        .execute("MATCH (n:Epoch) RETURN n.thread, n.txn")
        .unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * txns_per_thread,
        "All epoch nodes should exist"
    );
    let actual: BTreeSet<_> = result
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Int64(tid), Value::Int64(txn)) => (*tid, *txn),
            other => panic!("invalid epoch transaction identity: {other:?}"),
        })
        .collect();
    let expected: BTreeSet<_> = (0..num_threads)
        .flat_map(|tid| {
            (0..txns_per_thread)
                .map(move |txn| (i64::try_from(tid).unwrap(), i64::try_from(txn).unwrap()))
        })
        .collect();
    assert_eq!(
        actual, expected,
        "Each epoch transaction must commit exactly once"
    );
}

#[test]
#[ignore = "stress test"]
fn concurrent_merge_same_node() {
    // 8 threads each run 10 rounds of MERGE with their own key.
    // The contention comes from all threads issuing MERGE on the same
    // label, forcing concurrent read-then-write cycles on the index.
    // Each thread's MERGE should be idempotent: no matter how many
    // rounds, it should produce exactly one node per key.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 8;
    let rounds: usize = 10;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for round in 0..rounds {
                    let query = format!(
                        "MERGE (n:Shared {{key: 'thread_{tid}'}}) \
                         ON CREATE SET n.thread_id = {tid} \
                         ON MATCH SET n.round = {round}"
                    );
                    execute_stress_write(&db, &query);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Each thread should have exactly one node (MERGE is idempotent per key)
    let session = db.session();
    let result = session.execute("MATCH (n:Shared) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads,
        "Each thread should produce exactly 1 node via MERGE"
    );

    // Verify each thread's node exists
    for tid in 0..num_threads {
        let query = format!("MATCH (n:Shared {{key: 'thread_{tid}'}}) RETURN n.thread_id, n.round");
        let result = session.execute(&query).unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Thread {tid} should have exactly 1 node"
        );
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(i64::try_from(tid).unwrap())
        );
        assert_eq!(
            result.rows()[0][1],
            Value::Int64(i64::try_from(rounds - 1).unwrap()),
            "every MERGE round must complete"
        );
    }
}

#[test]
#[ignore = "stress test"]
fn concurrent_mixed_read_write_high_contention() {
    // 12 threads: 6 writers inserting unique nodes, 6 readers counting
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_writers = 6;
    let num_readers = 6;
    let total_threads = num_writers + num_readers;
    let barrier = Arc::new(Barrier::new(total_threads));
    let write_success = Arc::new(AtomicUsize::new(0));
    let read_errors = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Writer threads: each inserts a uniquely-labeled node
    static NAMES: [&str; 6] = ["Alix", "Gus", "Vincent", "Jules", "Mia", "Butch"];
    for wid in 0..num_writers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let write_success = Arc::clone(&write_success);
        let name = NAMES[wid];

        handles.push(thread::spawn(move || {
            barrier.wait();

            let session = db.session();
            let query = format!("INSERT (:Contention {{name: '{name}', writer: {wid}}})");
            if session.execute(&query).is_ok() {
                write_success.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    // Reader threads: each runs a MATCH + count query multiple times
    for _ in 0..num_readers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let read_errors = Arc::clone(&read_errors);

        handles.push(thread::spawn(move || {
            barrier.wait();

            for _ in 0..10 {
                let session = db.session();
                if session
                    .execute("MATCH (n:Contention) RETURN count(n)")
                    .is_err()
                {
                    read_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        read_errors.load(Ordering::Relaxed),
        0,
        "No read errors expected during high contention"
    );

    let writes = write_success.load(Ordering::Relaxed);
    assert_eq!(writes, num_writers, "All writers should succeed");

    // Verify total node count matches the number of successful writes
    let session = db.session();
    let result = session.execute("MATCH (n:Contention) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        writes,
        "Total nodes should equal number of successful writes"
    );
}

#[test]
#[ignore = "stress test"]
fn concurrent_schema_mutation_with_queries() {
    // 4 threads running queries while 2 threads create/drop node types.
    // Verifies no panics or crashes under concurrent schema changes.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_query_threads = 4;
    let num_schema_threads = 2;
    let total_threads = num_query_threads + num_schema_threads;
    let barrier = Arc::new(Barrier::new(total_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Schema mutation threads: create and drop node types in a loop
    for sid in 0..num_schema_threads {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let completed = Arc::clone(&completed);

        handles.push(thread::spawn(move || {
            barrier.wait();

            for i in 0..5 {
                let session = db.session();
                let type_name = format!("Temp{sid}_{i}");
                // Create a node type
                let _ = session.execute(&format!("CREATE NODE TYPE {type_name} (val INTEGER)"));
                // Insert a node of that type
                let _ = session.execute(&format!("INSERT (:{type_name} {{val: {i}}})"));
                // Drop the node type
                let _ = session.execute(&format!("DROP NODE TYPE {type_name}"));
            }
            completed.fetch_add(1, Ordering::Relaxed);
        }));
    }

    // Query threads: run read queries concurrently with schema changes
    static QUERY_NAMES: [&str; 4] = ["Django", "Shosanna", "Hans", "Beatrix"];
    for qid in 0..num_query_threads {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let completed = Arc::clone(&completed);
        let name = QUERY_NAMES[qid];

        handles.push(thread::spawn(move || {
            barrier.wait();

            let session = db.session();
            // Insert a node unrelated to the schema mutations
            let _ = session.execute(&format!(
                "INSERT (:QueryNode {{name: '{name}', qid: {qid}}})"
            ));
            // Run several read queries
            for _ in 0..10 {
                let _ = session.execute("MATCH (n:QueryNode) RETURN n.name");
            }
            completed.fetch_add(1, Ordering::Relaxed);
        }));
    }

    for handle in handles {
        handle
            .join()
            .expect("Thread panicked during concurrent schema mutation");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        total_threads,
        "All threads should complete without panic"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_rapid_session_lifecycle() {
    // 16 threads rapidly creating, using, and dropping sessions
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 16;
    let cycles = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for _ in 0..cycles {
                    let session = db.session();
                    let _ = session.execute("MATCH (n) RETURN n LIMIT 1");
                    drop(session);
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        num_threads,
        "All threads should complete"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_edges_and_nodes() {
    // Create nodes and edges simultaneously from multiple threads
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed some nodes first (needed for edge creation)
    let session = db.session();
    for i in 0..20 {
        session
            .execute(&format!("INSERT (:Hub {{id: {i}}})"))
            .unwrap();
    }
    drop(session);

    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                let session = db.session();
                for i in 0..10 {
                    // Create new nodes
                    session
                        .execute(&format!("INSERT (:Spoke {{thread: {tid}, id: {i}}})"))
                        .unwrap();
                    // Create edges between existing hub nodes
                    let src = (tid * 5 + i) % 20;
                    let dst = (tid * 5 + i + 1) % 20;
                    let _ = session.execute(&format!(
                        "MATCH (a:Hub {{id: {src}}}), (b:Hub {{id: {dst}}}) \
                         INSERT (a)-[:LINK {{thread: {tid}}}]->(b)"
                    ));
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // Verify spoke nodes were created
    let session = db.session();
    let result = session.execute("MATCH (n:Spoke) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * 10,
        "All spoke nodes should exist"
    );
}

// ============================================================================
// Isolation Anomaly Tests (T1-03)
// ============================================================================

/// Documents current dirty-read behavior: uncommitted writes in one session's
/// Dirty read prevention: uncommitted data from one transaction is invisible
/// to other sessions. Versions use PENDING epoch until committed.
#[test]
fn test_dirty_read_prevented() {
    let db = GrafeoDB::new_in_memory();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:DirtyRead {val: 'uncommitted'})")
        .unwrap();

    // Reader (auto-commit, no explicit transaction) must NOT see uncommitted data.
    // Uncommitted versions use PENDING epoch, invisible to epoch-based reads.
    let reader = db.session();
    let result = reader.execute("MATCH (n:DirtyRead) RETURN n").unwrap();

    assert_eq!(
        result.row_count(),
        0,
        "Dirty read prevented: uncommitted data is invisible to other sessions"
    );

    // After commit, the data becomes visible
    writer.commit().unwrap();

    let result2 = reader.execute("MATCH (n:DirtyRead) RETURN n").unwrap();
    assert_eq!(
        result2.row_count(),
        1,
        "Committed data should be visible to other sessions"
    );
}

/// Documents that after rollback, the rolled-back data is no longer visible.
#[test]
fn test_rollback_hides_data_from_other_sessions() {
    let db = GrafeoDB::new_in_memory();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:RollbackTest {val: 'temp'})")
        .unwrap();
    writer.rollback().unwrap();

    let reader = db.session();
    let result = reader.execute("MATCH (n:RollbackTest) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Rolled-back data should not be visible"
    );
}

/// Non-repeatable read: reader sees different results for the same query
/// when another session commits between reads.
#[test]
fn test_non_repeatable_read() {
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    session1.execute("INSERT (:NRR {val: 'original'})").unwrap();

    // Reader sees val='original'
    let reader = db.session();
    let r1 = reader.execute("MATCH (n:NRR) RETURN n.val AS val").unwrap();
    assert_eq!(r1.rows().len(), 1);

    // Writer updates
    session1
        .execute("MATCH (n:NRR) SET n.val = 'updated'")
        .unwrap();

    // Reader sees val='updated' (non-repeatable read)
    let r2 = reader.execute("MATCH (n:NRR) RETURN n.val AS val").unwrap();
    assert_eq!(r2.rows().len(), 1);
    // Without snapshot isolation, the reader sees the updated value
    assert_eq!(
        r2.rows()[0][0],
        grafeo_common::types::Value::String("updated".into()),
        "Non-repeatable read: reader sees committed update"
    );
}

/// Phantom read: reader sees new rows that didn't exist during its first read,
/// because another session inserted and committed between reads.
#[test]
fn test_phantom_read() {
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    session1.execute("INSERT (:Phantom {id: 1})").unwrap();

    // Reader sees 1 row
    let reader = db.session();
    let r1 = reader.execute("MATCH (n:Phantom) RETURN n").unwrap();
    assert_eq!(r1.row_count(), 1);

    // Writer inserts another row
    session1.execute("INSERT (:Phantom {id: 2})").unwrap();

    // Reader sees 2 rows (phantom read)
    let r2 = reader.execute("MATCH (n:Phantom) RETURN n").unwrap();
    assert_eq!(
        r2.row_count(),
        2,
        "Phantom read: new rows from other sessions are visible"
    );
}

/// Session drop mid-transaction: verifies that rolled-back data is not visible.
/// (Related to T1-04: Session Drop should auto-rollback.)
#[test]
fn test_drop_session_mid_transaction() {
    let db = GrafeoDB::new_in_memory();

    {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:DropTest {val: 'should_vanish'})")
            .unwrap();
        // Session drops here without commit or rollback
    }

    let reader = db.session();
    let result = reader.execute("MATCH (n:DropTest) RETURN n").unwrap();
    // Drop impl auto-rollbacks the active transaction, so uncommitted data is discarded.
    assert_eq!(
        result.row_count(),
        0,
        "Drop impl should auto-rollback, discarding uncommitted data"
    );
}

// ============================================================================
// Write-Write Conflict Tests (T1-07)
// ============================================================================

/// Tests first-writer-wins conflict detection: two concurrent sessions
/// modifying the same node property through `Session.execute()`.
///
/// The second session's SET fails immediately (not at commit time) because
/// the first session already recorded a write to the same entity.
#[test]
fn test_write_write_conflict_through_execute() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Account {name: 'shared', balance: 100})")
        .unwrap();

    // Session 1: begin tx, read and update
    let mut s1 = db.session();
    s1.begin_transaction().unwrap();
    s1.execute("MATCH (a:Account {name: 'shared'}) SET a.balance = 200")
        .unwrap();

    // Session 2: begin tx, attempt to update the same node
    let mut s2 = db.session();
    s2.begin_transaction().unwrap();
    let set_result = s2.execute("MATCH (a:Account {name: 'shared'}) SET a.balance = 300");
    assert!(
        set_result.is_err(),
        "Second SET should fail with write-write conflict: {set_result:?}"
    );

    // First commit succeeds
    let commit1 = s1.commit();
    assert!(commit1.is_ok(), "First commit should succeed: {commit1:?}");

    // s2 rollback (transaction is still active even though SET failed)
    let _ = s2.rollback();

    // Verify final state: first writer's value persists
    let result = session
        .execute("MATCH (a:Account {name: 'shared'}) RETURN a.balance AS b")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], grafeo_common::types::Value::Int64(200));
}

/// Tests that a rollback in one session doesn't affect another session's committed writes.
#[test]
fn test_concurrent_write_one_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Counter {name: 'hits', val: 0})")
        .unwrap();

    // Session 1: update and commit
    let mut s1 = db.session();
    s1.begin_transaction().unwrap();
    s1.execute("MATCH (c:Counter {name: 'hits'}) SET c.val = 10")
        .unwrap();
    s1.commit().unwrap();

    // Session 2: update and rollback
    let mut s2 = db.session();
    s2.begin_transaction().unwrap();
    s2.execute("MATCH (c:Counter {name: 'hits'}) SET c.val = 999")
        .unwrap();
    s2.rollback().unwrap();

    // Session 1's value (10) should be the final state
    let result = session
        .execute("MATCH (c:Counter {name: 'hits'}) RETURN c.val AS v")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::Int64(10),
        "Rolled-back write should not affect committed value"
    );
}

// ============================================================================
// T2-05: Edge creation/deletion rollback
// ============================================================================

/// Create an edge inside a transaction, rollback, verify edge is absent.
#[test]
fn test_edge_create_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    // Edge exists inside transaction
    let mid = session.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(mid.row_count(), 1, "Edge should exist inside transaction");

    session.rollback().unwrap();

    // After rollback, edge should be gone
    let reader = db.session();
    let result = reader.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Edge should not exist after rollback"
    );
}

/// Documents that DELETE edge followed by rollback does NOT restore the edge.
///
/// The `discard_uncommitted_versions` method correctly removes versions created
/// DELETE edge followed by rollback restores the edge.
///
/// The transactional delete captures undo information (edge type, endpoints, properties)
/// and marks the version with `deleted_by`. Rollback replays the undo log to restore.
#[test]
fn test_edge_delete_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    // Verify edge exists
    let before = session.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(before.row_count(), 1);

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) DELETE r")
        .unwrap();
    session.rollback().unwrap();

    let reader = db.session();
    let result = reader.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Edge should be restored after rollback"
    );
}

/// DELETE node followed by rollback restores the node with its labels and properties.
#[test]
fn test_node_delete_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Temp {name: 'ephemeral'})")
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (t:Temp {name: 'ephemeral'}) DELETE t")
        .unwrap();
    session.rollback().unwrap();

    let reader = db.session();
    let result = reader
        .execute("MATCH (t:Temp) RETURN t.name AS name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Node should be restored after rollback"
    );
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::String("ephemeral".into())
    );
}

/// DETACH DELETE followed by rollback restores the node and its edges.
#[test]
fn test_detach_delete_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:Person {name: 'Alix'}) DETACH DELETE a")
        .unwrap();
    session.rollback().unwrap();

    let reader = db.session();
    let nodes = reader
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(
        nodes.row_count(),
        2,
        "Both nodes should be restored after rollback"
    );
    assert_eq!(
        nodes.rows()[0][0],
        grafeo_common::types::Value::String("Alix".into())
    );
    assert_eq!(
        nodes.rows()[1][0],
        grafeo_common::types::Value::String("Gus".into())
    );

    // Edge should also be restored
    let edges = reader.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(
        edges.row_count(),
        1,
        "Edge should be restored after DETACH DELETE rollback"
    );
}

// ============================================================================
// T2-06: Cross-session commit visibility
// ============================================================================

/// After INSERT+COMMIT in an explicit transaction, a new session sees the data.
#[test]
fn test_cross_session_visibility_after_explicit_commit() {
    let db = GrafeoDB::new_in_memory();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Visible {key: 'committed'})")
        .unwrap();
    writer.commit().unwrap();

    // New session should see committed data
    let reader = db.session();
    let result = reader
        .execute("MATCH (v:Visible) RETURN v.key AS key")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "New session should see committed data"
    );
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::String("committed".into())
    );
}

/// Multiple mutations across transactions, verified by a fresh session.
#[test]
fn test_cross_session_visibility_multiple_mutations() {
    let db = GrafeoDB::new_in_memory();

    // First session: insert + set property
    let session1 = db.session();
    session1
        .execute("INSERT (:Item {name: 'widget', price: 10})")
        .unwrap();
    session1
        .execute("MATCH (i:Item {name: 'widget'}) SET i.price = 25")
        .unwrap();

    // Second session sees the updated value
    let session2 = db.session();
    let result = session2
        .execute("MATCH (i:Item {name: 'widget'}) RETURN i.price AS price")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::Int64(25),
        "New session should see the updated price"
    );
}

/// Public Session shape of SF3 `lost_update`: workers use explicit transactions
/// to increment one `:Node:Person` row, while the reset uses the public
/// auto-commit `MATCH ... SET ... RETURN n` path. This is a correctness control;
/// passing it does not establish closure of the retained crash.
#[test]
fn test_concurrent_auto_commit_set_return_preserves_person_labels() {
    // graph-bench lost_update. Default stays CI-sized. SF3 is
    // GRAFEO_LU_PEOPLE=27000 GRAFEO_LU_EDGES=540000 GRAFEO_LU_ROUNDS=10
    let people = env_usize("GRAFEO_LU_PEOPLE", 2048);
    let edges = env_usize("GRAFEO_LU_EDGES", 0);
    let rounds = env_usize("GRAFEO_LU_ROUNDS", 1).max(1);
    assert!(people > 0, "lost_update requires at least one person");
    const BATCH: usize = 1000;
    eprintln!("lost_update load people={people} edges={edges} rounds={rounds}");
    let db = Arc::new(GrafeoDB::with_config(Config::in_memory()).unwrap());
    {
        let session = db.session();
        for start in (0..people).step_by(BATCH) {
            let rows = (start..(start + BATCH).min(people))
                .map(|i| {
                    row_map([
                        ("id", Value::String(format!("person-{i}").into())),
                        ("viewCount", Value::Int64(0)),
                        ("lastUpdate", Value::String(String::new().into())),
                        ("lastAccess", Value::String(String::new().into())),
                    ])
                })
                .collect();
            session
                .execute_with_params(
                    "UNWIND $rows AS r INSERT (:Person:Node {id: r.id, viewCount: r.viewCount, lastUpdate: r.lastUpdate, lastAccess: r.lastAccess})",
                    unwind_rows(rows),
                )
                .unwrap();
            session
                .execute("CREATE INDEX IF NOT EXISTS gb_id FOR (n:Node) ON (n.id)")
                .unwrap();
        }
        for start in (0..edges).step_by(BATCH) {
            let rows = (start..(start + BATCH).min(edges))
                .map(|edge| {
                    row_map([
                        (
                            "s",
                            Value::String(format!("person-{}", edge % people).into()),
                        ),
                        (
                            "t",
                            Value::String(format!("person-{}", (edge + 1) % people).into()),
                        ),
                    ])
                })
                .collect();
            session
                .execute_with_params(
                    "UNWIND $rows AS e MATCH (s:Node {id:e.s}), (t:Node {id:e.t}) CREATE (s)-[:KNOWS]->(t)",
                    unwind_rows(rows),
                )
                .unwrap();
        }
        let setup_updates = people.min(100);
        for i in 0..setup_updates {
            let reset = session
                .execute(&format!(
                    "MATCH (n:Node {{id: 'person-{i}'}}) SET n.viewCount = 0, n.lastUpdate = '', n.lastAccess = '' RETURN n"
                ))
                .expect("setup auto-commit SET RETURN must not crash");
            assert_eq!(reset.row_count(), 1, "setup person-{i}");
        }
    }

    for round in 0..rounds {
        eprintln!("lost_update round {round}");
        let session = db.session();
        let reset = session
            .execute("MATCH (n:Node {id: 'person-0'}) SET n.viewCount = 0 RETURN n")
            .expect("round reset SET RETURN must not crash");
        assert_eq!(
            reset.row_count(),
            1,
            "round {round} reset should return one node"
        );
        let Value::Map(node) = &reset.rows()[0][0] else {
            panic!("round {round} reset should return a node map");
        };
        assert_eq!(
            node.get(&PropertyKey::new("id")),
            Some(&Value::String("person-0".into()))
        );
        assert_eq!(
            node.get(&PropertyKey::new("viewCount")),
            Some(&Value::Int64(0))
        );
        let Value::List(labels) = node
            .get(&PropertyKey::new("_labels"))
            .expect("round reset should include typed labels")
        else {
            panic!("round {round} reset labels should be a list");
        };
        let mut label_names = Vec::with_capacity(labels.len());
        for label in labels.iter() {
            let Value::String(label) = label else {
                panic!("round {round} reset labels must be strings");
            };
            label_names.push(label.as_str());
        }
        label_names.sort_unstable();
        assert_eq!(label_names, ["Node", "Person"]);
        drop(session);

        let workers = 4;
        let barrier = Arc::new(Barrier::new(workers));
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..25 {
                        let mut last_error = None;
                        let mut succeeded = false;
                        for _ in 0..64 {
                            let mut session = db.session();
                            session.begin_transaction().unwrap();
                            let result = (|| {
                                let rows = session.execute(
                                    "MATCH (p:Person {id: 'person-0'}) RETURN p.viewCount AS viewCount",
                                )?;
                                assert_eq!(rows.row_count(), 1);
                                let Value::Int64(current) = &rows.rows()[0][0] else {
                                    panic!("viewCount must be an integer: {:?}", rows.rows()[0][0]);
                                };
                                let next = current + 1;
                                session.execute(&format!(
                                    "MATCH (p:Person {{id: 'person-0'}}) SET p.viewCount = {next}"
                                ))?;
                                session.commit()
                            })();
                            match result {
                                Ok(_) => {
                                    succeeded = true;
                                    break;
                                }
                                Err(error) if is_classified_write_conflict(&error) => {
                                    let _ = session.rollback();
                                    last_error = Some(error);
                                }
                                Err(error) => panic!("lost_update SET must not crash: {error}"),
                            }
                        }
                        assert!(
                            succeeded,
                            "worker increment exhausted retries: {last_error:?}"
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("worker thread panicked");
        }

        let node_rows = db
            .session()
            .execute("MATCH (n:Node {id: 'person-0'}) RETURN n.viewCount")
            .expect("Node post-worker read must succeed");
        assert_eq!(
            node_rows.rows(),
            &[vec![Value::Int64(100)]],
            "round {round} Node total"
        );
        let person_rows = db
            .session()
            .execute("MATCH (n:Person {id: 'person-0'}) RETURN n.viewCount")
            .expect("Person post-worker read must succeed");
        assert_eq!(
            person_rows.rows(),
            &[vec![Value::Int64(100)]],
            "round {round} Person total"
        );
    }

    let session = db.session();
    let reset = session
        .execute("MATCH (n:Node {id: 'person-0'}) SET n.viewCount = 0 RETURN n")
        .expect("reset SET RETURN must not crash");
    assert_eq!(reset.row_count(), 1);
    let Value::Map(node) = &reset.rows()[0][0] else {
        panic!("reset must return a node map");
    };
    assert_eq!(
        node.get(&PropertyKey::new("id")),
        Some(&Value::String("person-0".into()))
    );
    assert_eq!(
        node.get(&PropertyKey::new("viewCount")),
        Some(&Value::Int64(0))
    );
    let Value::List(labels) = node
        .get(&PropertyKey::new("_labels"))
        .expect("final reset should include typed labels")
    else {
        panic!("final reset labels should be a list");
    };
    let mut label_names = Vec::with_capacity(labels.len());
    for label in labels.iter() {
        let Value::String(label) = label else {
            panic!("final reset labels must be strings");
        };
        label_names.push(label.as_str());
    }
    label_names.sort_unstable();
    assert_eq!(label_names, ["Node", "Person"]);
    assert_eq!(
        session
            .execute("MATCH (n:Person {id: 'person-0'}) RETURN n.viewCount")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(0)]]
    );
}

/// Repeated auto-commit updates must preserve a labeled node and its identity.
#[test]
fn test_repeated_auto_commit_labeled_node_updates_preserve_rows_and_labels() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let node_id = session
        .create_node_with_props(
            &["Node", "Person"],
            [
                ("id", Value::String("person".into())),
                ("viewCount", Value::Int64(0)),
            ],
        )
        .unwrap();
    let expected_labels = ["Node", "Person"];
    let assert_labels = |check: &Session, expected: &[&str]| {
        let node = check
            .get_node(node_id)
            .expect("seeded node should remain visible");
        let mut labels: Vec<&str> = node.labels.iter().map(|label| label.as_str()).collect();
        labels.sort_unstable();
        assert_eq!(labels, expected);
        assert_eq!(
            check.get_node_property(node_id, "id"),
            Some(Value::String("person".into()))
        );
    };

    for cycle in 0..32 {
        let nonzero = i64::from(cycle + 1);
        session
            .set_node_property(node_id, "viewCount", Value::Int64(nonzero))
            .unwrap();
        assert!(!session.add_node_label(node_id, "Person"));
        assert!(!session.remove_node_label(node_id, "Absent"));
        assert!(session.add_node_label(node_id, "CycleLabel"));
        assert_labels(&session, &["CycleLabel", "Node", "Person"]);
        assert!(session.remove_node_label(node_id, "CycleLabel"));
        assert_labels(&session, &expected_labels);
        assert_eq!(
            session.get_node_property(node_id, "viewCount"),
            Some(Value::Int64(nonzero))
        );

        let result = session
            .execute("MATCH (n:Node {id: 'person'}) SET n.viewCount = 0 RETURN n")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "cycle {cycle} should return one node"
        );
        let Value::Map(node) = &result.rows()[0][0] else {
            panic!("cycle {cycle} should return a node map");
        };
        assert_eq!(
            node.get(&PropertyKey::new("_id")),
            Some(&Value::Int64(i64::try_from(node_id.as_u64()).unwrap()))
        );
        assert_eq!(
            node.get(&PropertyKey::new("id")),
            Some(&Value::String("person".into()))
        );
        assert_eq!(
            node.get(&PropertyKey::new("viewCount")),
            Some(&Value::Int64(0))
        );

        let node_rows = session
            .execute("MATCH (n:Node) RETURN n.id, n.viewCount")
            .unwrap();
        assert_eq!(node_rows.row_count(), 1, "cycle {cycle} Node row count");
        assert_eq!(
            node_rows.rows()[0],
            [Value::String("person".into()), Value::Int64(0)]
        );
        let person_rows = session
            .execute("MATCH (n:Person) RETURN n.id, n.viewCount")
            .unwrap();
        assert_eq!(person_rows.row_count(), 1, "cycle {cycle} Person row count");
        assert_eq!(
            person_rows.rows()[0],
            [Value::String("person".into()), Value::Int64(0)]
        );
        assert_labels(&session, &expected_labels);
        assert!(
            !session.in_transaction(),
            "cycle {cycle} left a transaction active"
        );

        let fresh = db.session();
        assert_labels(&fresh, &expected_labels);
        assert_eq!(
            fresh.get_node_property(node_id, "viewCount"),
            Some(Value::Int64(0))
        );
        assert!(
            !fresh.in_transaction(),
            "fresh session entered a transaction"
        );
    }
}

/// Races the auto-commit `SET ... RETURN n` label read against every writer
/// that mutates `node_labels` concurrently: explicit increments, label
/// add/remove on the same node, and open transactions that inline-create
/// labelled nodes (PENDING label logs) and then commit or roll back. Readers
/// must never see a rolled-back node's labels. This is a regression control
/// for concurrent label access; it does not establish general crash safety.
#[test]
fn test_concurrent_label_reads_race_label_writers_and_pending_creates() {
    // GRAFEO_LABEL_RACE_ITERS scales every writer; default stays CI-sized.
    let iterations = env_usize("GRAFEO_LABEL_RACE_ITERS", 50);
    let db = Arc::new(GrafeoDB::with_config(Config::in_memory()).unwrap());
    {
        let session = db.session();
        session
            .execute("INSERT (:Person:Node {id: 'person-0', viewCount: 0})")
            .unwrap();
        session
            .execute("CREATE INDEX IF NOT EXISTS gb_id FOR (n:Node) ON (n.id)")
            .unwrap();
    }

    fn sorted_returned_labels(node: &Value) -> Vec<String> {
        let Value::Map(node) = node else {
            panic!("RETURN n must yield a node map: {node:?}");
        };
        let Some(Value::List(labels)) = node.get(&PropertyKey::new("_labels")) else {
            panic!("returned node must carry a _labels list: {node:?}");
        };
        let mut names: Vec<String> = labels
            .iter()
            .map(|label| match label {
                Value::String(label) => label.as_str().to_owned(),
                other => panic!("label must be a string: {other:?}"),
            })
            .collect();
        names.sort_unstable();
        names
    }

    /// Runs `body` in a fresh explicit transaction, retrying classified conflicts.
    fn with_retry(
        db: &GrafeoDB,
        what: &str,
        body: impl Fn(&Session) -> grafeo_common::utils::error::Result<()>,
    ) {
        let mut last_error = None;
        for attempt in 0..1024u32 {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            match body(&session).and_then(|()| session.commit().map(|_| ())) {
                Ok(()) => return,
                Err(error) if is_classified_write_conflict(&error) => {
                    let _ = session.rollback();
                    last_error = Some(error.to_string());
                    // Jittered backoff, bounded at ~5ms, like the benchmark harness.
                    let micros = (100u64 << attempt.min(5)) + u64::from(attempt * 37 % 97);
                    thread::sleep(std::time::Duration::from_micros(micros));
                }
                Err(error) => panic!("{what} must not fail: {error}"),
            }
        }
        panic!("{what} exhausted retries: {last_error:?}");
    }

    let writers_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let committed_pending = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(5));
    let mut writers = Vec::new();

    // Increment workers: lost_update shape.
    for _ in 0..2 {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        writers.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..iterations {
                with_retry(&db, "increment", |session| {
                    session.execute(
                        "MATCH (p:Person {id: 'person-0'}) SET p.viewCount = p.viewCount + 1",
                    )?;
                    Ok(())
                });
            }
        }));
    }

    // Label churn on the node every reader materializes.
    {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        writers.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..iterations {
                with_retry(&db, "add Churn label", |session| {
                    session.execute("MATCH (n:Node {id: 'person-0'}) SET n:Churn")?;
                    Ok(())
                });
                with_retry(&db, "remove Churn label", |session| {
                    session.execute("MATCH (n:Node {id: 'person-0'}) REMOVE n:Churn")?;
                    Ok(())
                });
            }
        }));
    }

    // Inline creates hold PENDING label logs open across other commits.
    {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let committed_pending = Arc::clone(&committed_pending);
        writers.push(thread::spawn(move || {
            barrier.wait();
            for k in 0..iterations {
                let insert =
                    format!("INSERT (:Person:Node:Pending {{id: 'pending-{k}', viewCount: 0}})");
                if k % 2 == 0 {
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    session.execute(&insert).unwrap();
                    for _ in 0..8 {
                        thread::yield_now();
                    }
                    session.rollback().unwrap();
                } else {
                    // Indexed publication rejects registry contention as T001.
                    with_retry(&db, "pending create", |session| {
                        session.execute(&insert)?;
                        for _ in 0..8 {
                            thread::yield_now();
                        }
                        Ok(())
                    });
                    committed_pending.fetch_add(1, Ordering::SeqCst);
                }
            }
        }));
    }

    let mut readers = Vec::new();

    // Auto-commit SET ... RETURN n: the retained crash's label-read site.
    {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let writers_done = Arc::clone(&writers_done);
        readers.push(thread::spawn(move || {
            barrier.wait();
            let mut returned = 0usize;
            while !writers_done.load(Ordering::SeqCst) || returned == 0 {
                let session = db.session();
                match session
                    .execute("MATCH (n:Node {id: 'person-0'}) SET n.lastAccess = 'x' RETURN n")
                {
                    Ok(result) => {
                        assert_eq!(result.row_count(), 1, "reset must return person-0");
                        let labels = sorted_returned_labels(&result.rows()[0][0]);
                        assert!(
                            labels == ["Node", "Person"] || labels == ["Churn", "Node", "Person"],
                            "person-0 labels corrupted: {labels:?}"
                        );
                        returned += 1;
                    }
                    Err(error) if is_classified_write_conflict(&error) => {}
                    Err(error) => panic!("auto-commit SET RETURN must not fail: {error}"),
                }
                // Pace like the benchmark reset so explicit writers are not starved.
                thread::sleep(std::time::Duration::from_micros(500));
            }
            returned
        }));
    }

    let pending_reader = {
        let db = Arc::clone(&db);
        let writers_done = Arc::clone(&writers_done);
        thread::spawn(move || {
            while !writers_done.load(Ordering::SeqCst) {
                let rows = db
                    .session()
                    .execute("MATCH (n:Pending) RETURN n.id")
                    .expect("Pending scan must succeed");
                for row in rows.rows() {
                    let Value::String(id) = &row[0] else {
                        panic!("Pending id must be a string: {:?}", row[0]);
                    };
                    let k: usize = id
                        .as_str()
                        .strip_prefix("pending-")
                        .unwrap()
                        .parse()
                        .unwrap();
                    assert!(k % 2 == 1, "rolled-back {id} became visible");
                }
            }
        })
    };

    for writer in writers {
        writer.join().expect("writer thread panicked");
    }
    writers_done.store(true, Ordering::SeqCst);
    for reader in readers {
        assert!(reader.join().expect("reader thread panicked") > 0);
    }
    pending_reader.join().expect("pending reader panicked");

    let session = db.session();
    let expected = i64::try_from(2 * iterations).unwrap();
    assert_eq!(
        session
            .execute("MATCH (n:Person {id: 'person-0'}) RETURN n.viewCount")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(expected)]],
        "every committed increment must survive"
    );
    let final_reset = session
        .execute("MATCH (n:Node {id: 'person-0'}) SET n.viewCount = 0 RETURN n")
        .unwrap();
    assert_eq!(
        sorted_returned_labels(&final_reset.rows()[0][0]),
        ["Node", "Person"]
    );
    let committed = i64::try_from(committed_pending.load(Ordering::SeqCst)).unwrap();
    assert_eq!(committed, i64::try_from(iterations / 2).unwrap());
    for (label, count) in [
        ("Pending", committed),
        ("Person", committed + 1),
        ("Node", committed + 1),
    ] {
        assert_eq!(
            session
                .execute(&format!("MATCH (n:{label}) RETURN count(n)"))
                .unwrap()
                .rows(),
            &[vec![Value::Int64(count)]],
            "{label} count after rollbacks"
        );
    }
    assert_eq!(
        session
            .execute("MATCH (n:Node {id: 'pending-0'}) RETURN n")
            .unwrap()
            .row_count(),
        0,
        "rolled-back create must stay invisible"
    );
}

/// Edge creation visibility: edges inserted in one session visible in another.
#[test]
fn test_cross_session_edge_visibility() {
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    session1
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    let session2 = db.session();
    let result = session2
        .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS from, b.name AS to")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::String("Alix".into())
    );
    assert_eq!(
        result.rows()[0][1],
        grafeo_common::types::Value::String("Gus".into())
    );
}
