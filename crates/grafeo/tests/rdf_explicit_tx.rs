//! Explicit RDF-only transaction boundary without LPG.
//!
//! An RDF client wraps multiple named-graph SPARQL fragments in one Session
//! transaction. Per-statement autocommit is not enough.
//!
//! ```text
//! cargo test -p grafeo --no-default-features --features rdf \
//!   --test rdf_explicit_tx -- --test-threads=1
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo::{Config, GrafeoDB, GraphModel, IsolationLevel, Quad, Session, Term, Triple};

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).expect("rdf db")
}

fn count_in_graph(session: &Session, graph: &str) -> usize {
    session
        .execute_sparql(&format!(
            "SELECT ?o WHERE {{ GRAPH <{graph}> {{ ?s <http://ex.org/p> ?o }} }}"
        ))
        .expect("select")
        .row_count()
}

fn insert_named(session: &Session, graph: &str, subject: &str, value: &str) {
    session
        .execute_sparql(&format!(
            r#"INSERT DATA {{ GRAPH <{graph}> {{ <{subject}> <http://ex.org/p> "{value}" }} }}"#
        ))
        .unwrap_or_else(|e| panic!("insert into {graph}: {e}"));
}

/// begin → two named-graph inserts → RYW → rollback → both absent
#[test]
fn explicit_tx_two_named_graph_inserts_rollback() {
    let db = rdf_db();
    let mut session = db.session();

    session.begin_transaction().expect("begin");
    insert_named(&session, "http://ex.org/g1", "http://ex.org/s1", "a");
    insert_named(&session, "http://ex.org/g2", "http://ex.org/s2", "b");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        1,
        "read-your-writes in named graph g1"
    );
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g2"),
        1,
        "read-your-writes in named graph g2"
    );

    session.rollback().expect("rollback");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        0,
        "g1 insert must be absent after rollback"
    );
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g2"),
        0,
        "g2 insert must be absent after rollback"
    );
}

/// begin → two named-graph inserts → commit → both visible
#[test]
fn explicit_tx_two_named_graph_inserts_commit() {
    let db = rdf_db();
    let mut session = db.session();

    session.begin_transaction().expect("begin");
    insert_named(&session, "http://ex.org/g1", "http://ex.org/s1", "a");
    insert_named(&session, "http://ex.org/g2", "http://ex.org/s2", "b");
    let epoch = session.commit().expect("commit");
    assert!(
        epoch.as_u64() > 0,
        "commit must return a crash-stable epoch"
    );

    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        1,
        "g1 insert must be visible after commit"
    );
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g2"),
        1,
        "g2 insert must be visible after commit"
    );

    let other = db.session();
    assert_eq!(
        count_in_graph(&other, "http://ex.org/g1"),
        1,
        "committed g1 must be visible to another session"
    );
    assert_eq!(
        count_in_graph(&other, "http://ex.org/g2"),
        1,
        "committed g2 must be visible to another session"
    );
}

fn create_graph_ok(session: &Session, graph: &str) -> bool {
    session
        .execute_sparql(&format!("CREATE GRAPH <{graph}>"))
        .is_ok()
}

/// Dropping a session with an open RDF tx must auto-rollback.
#[test]
fn drop_open_tx_discards_named_graph_insert() {
    let db = rdf_db();
    {
        let mut session = db.session();
        session.begin_transaction().expect("begin");
        insert_named(&session, "http://ex.org/g1", "http://ex.org/s1", "a");
        assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 1);
    }
    let session = db.session();
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        0,
        "dropped session must not leak uncommitted triples"
    );
    assert!(
        create_graph_ok(&session, "http://ex.org/g1"),
        "dropped session must not leave the named graph"
    );
}

/// A repeated begin creates a nested savepoint-backed transaction scope.
#[test]
fn repeated_begin_creates_nested_scope() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session.begin_transaction().expect("begin nested scope");
    session.rollback().expect("rollback nested scope");
    assert!(session.in_transaction(), "outer scope remains active");
    session.rollback().expect("rollback outer scope");
    assert!(!session.in_transaction());
}

/// The facade must expose the isolation selector for RDF-only applications;
/// making callers depend on an engine-internal module would turn the public
/// Serializable promise into an LPG/default-profile accident.
#[test]
fn rdf_facade_exposes_serializable_isolation() {
    let db = rdf_db();
    let mut session = db.session();
    session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect("begin Serializable through the RDF-only facade");
    session
        .rollback()
        .expect("rollback Serializable transaction");
}

/// Docs: rollback errors if no transaction is active.
#[test]
fn rollback_without_transaction_errors() {
    let db = rdf_db();
    let mut session = db.session();
    let err = session.rollback();
    assert!(
        err.is_err(),
        "rollback without a transaction must error, got {err:?}"
    );
}

/// Rollback of INSERT DATA GRAPH must not leave an empty named graph.
#[test]
fn rollback_drops_named_graph_created_by_insert() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    insert_named(&session, "http://ex.org/g-new", "http://ex.org/s", "v");
    session.rollback().expect("rollback");
    assert!(
        create_graph_ok(&session, "http://ex.org/g-new"),
        "rollback must drop the named graph created by INSERT DATA GRAPH"
    );
}

/// Pre-existing named graph survives rollback of inserts into it.
#[test]
fn rollback_keeps_preexisting_named_graph() {
    let db = rdf_db();
    let mut session = db.session();
    session
        .execute_sparql("CREATE GRAPH <http://ex.org/g-keep>")
        .expect("create");
    session.begin_transaction().expect("begin");
    insert_named(&session, "http://ex.org/g-keep", "http://ex.org/s", "v");
    session.rollback().expect("rollback");
    let again = session.execute_sparql("CREATE GRAPH <http://ex.org/g-keep>");
    assert!(
        again.is_err(),
        "pre-existing named graph must still exist after rollback, got {again:?}"
    );
}

/// CREATE GRAPH inside a transaction must disappear on rollback.
#[test]
fn rollback_drops_create_graph() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("CREATE GRAPH <http://ex.org/g-created>")
        .expect("create in tx");
    session.rollback().expect("rollback");
    assert!(
        create_graph_ok(&session, "http://ex.org/g-created"),
        "CREATE GRAPH in a rolled-back tx must not leave the graph"
    );
}

/// DELETE WHERE must buffer until commit; rollback restores the triple.
#[test]
fn delete_where_rolls_back() {
    let db = rdf_db();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "keep" }"#)
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("DELETE WHERE { ?s <http://ex.org/p> ?o }")
        .expect("delete where");
    assert_eq!(
        session
            .execute_sparql("SELECT ?o WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        0,
        "DELETE WHERE must be visible as RYW"
    );
    session.rollback().expect("rollback");
    assert_eq!(
        session
            .execute_sparql("SELECT ?o WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        1,
        "DELETE WHERE must not persist after rollback"
    );
}

/// INSERT { GRAPH } WHERE must buffer; rollback leaves the dest graph absent.
#[test]
fn insert_where_named_graph_rolls_back() {
    let db = rdf_db();
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "src" }"#)
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql(
            r#"INSERT { GRAPH <http://ex.org/g-copy> { ?s <http://ex.org/p> ?o } }
               WHERE { ?s <http://ex.org/p> ?o }"#,
        )
        .expect("insert where");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g-copy"),
        1,
        "INSERT WHERE named GRAPH must RYW"
    );
    session.rollback().expect("rollback");
    assert_eq!(count_in_graph(&session, "http://ex.org/g-copy"), 0);
    assert!(
        create_graph_ok(&session, "http://ex.org/g-copy"),
        "INSERT WHERE must not leave an empty dest graph after rollback"
    );
}

/// Concurrent RDF commits must assign unique epochs and leave current epoch at max.
#[test]
fn concurrent_commits_epochs_monotonic() {
    use grafeo_common::utils::error::TransactionError;
    use std::sync::{Arc, Barrier};
    use std::thread;

    let db = Arc::new(rdf_db());
    let n = 8usize;
    let first_attempts = Arc::new(Barrier::new(n));
    let handles: Vec<_> = (0..n)
        .map(|i| {
            let db = Arc::clone(&db);
            let first_attempts = Arc::clone(&first_attempts);
            thread::spawn(move || {
                let mut conflicts = 0;
                // Each revision conflict requires another writer to have committed.
                // With n writers committing once, at most n - 1 retries are needed.
                for attempt in 0..n {
                    let mut session = db.session();
                    session.begin_transaction().expect("begin");
                    session
                        .execute_sparql(&format!(
                            r#"INSERT DATA {{ <http://ex.org/s{i}> <http://ex.org/p> "{i}" }}"#
                        ))
                        .expect("insert");
                    if attempt == 0 {
                        first_attempts.wait();
                    }
                    match session.commit() {
                        Ok(epoch) => return (epoch, conflicts),
                        Err(grafeo::Error::Transaction(TransactionError::WriteConflict(_))) => {
                            assert!(!session.in_transaction());
                            conflicts += 1;
                            thread::yield_now();
                        }
                        Err(error) => panic!("unexpected commit failure: {error}"),
                    }
                }
                panic!("writer {i} exhausted {n} transaction attempts")
            })
        })
        .collect();
    let commits: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    assert!(
        commits
            .iter()
            .map(|(_, conflicts)| conflicts)
            .sum::<usize>()
            >= n - 1,
        "the aligned first attempts must exercise RDF revision conflicts"
    );
    let mut epochs: Vec<u64> = commits.iter().map(|(epoch, _)| epoch.as_u64()).collect();
    epochs.sort_unstable();
    let mut unique = epochs.clone();
    unique.dedup();
    assert_eq!(
        unique.len(),
        n,
        "commit epochs must be unique, got {epochs:?}"
    );
    assert_eq!(
        db.rdf_store_commit_epoch().as_u64(),
        *epochs.last().unwrap(),
        "store commit epoch must be the max assigned epoch, not a later-applied earlier one"
    );
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }")
            .unwrap()
            .row_count(),
        n,
        "every concurrent commit must be visible"
    );
}

fn seed_named(db: &grafeo::GrafeoDB, graph: &str, subject: &str, value: &str) {
    db.execute_sparql(&format!(
        r#"INSERT DATA {{ GRAPH <{graph}> {{ <{subject}> <http://ex.org/p> "{value}" }} }}"#
    ))
    .unwrap();
}

/// COPY into a new graph must RYW and roll back without leaving the dest graph.
#[test]
fn copy_graph_rolls_back() {
    let db = rdf_db();
    seed_named(&db, "http://ex.org/g1", "http://ex.org/s1", "a");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("COPY <http://ex.org/g1> TO <http://ex.org/g2>")
        .expect("copy");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        1,
        "COPY keeps source"
    );
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g2"),
        1,
        "COPY RYW dest"
    );
    session.rollback().expect("rollback");
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 1);
    assert_eq!(count_in_graph(&session, "http://ex.org/g2"), 0);
    assert!(
        create_graph_ok(&session, "http://ex.org/g2"),
        "rolled-back COPY must not leave dest graph"
    );
}

/// ADD must union into dest and roll back to the pre-tx dest.
#[test]
fn add_graph_rolls_back() {
    let db = rdf_db();
    seed_named(&db, "http://ex.org/g1", "http://ex.org/s1", "a");
    seed_named(&db, "http://ex.org/g2", "http://ex.org/s2", "b");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("ADD <http://ex.org/g1> TO <http://ex.org/g2>")
        .expect("add");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g2"),
        2,
        "ADD RYW union"
    );
    session.rollback().expect("rollback");
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 1);
    assert_eq!(count_in_graph(&session, "http://ex.org/g2"), 1);
}

/// MOVE must RYW dest and restore source on rollback.
#[test]
fn move_graph_rolls_back() {
    let db = rdf_db();
    seed_named(&db, "http://ex.org/g1", "http://ex.org/s1", "a");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("MOVE <http://ex.org/g1> TO <http://ex.org/g2>")
        .expect("move");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g2"),
        1,
        "MOVE RYW dest"
    );
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        0,
        "MOVE hides source triples"
    );
    session.rollback().expect("rollback");
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 1);
    assert_eq!(count_in_graph(&session, "http://ex.org/g2"), 0);
    assert!(
        create_graph_ok(&session, "http://ex.org/g2"),
        "rolled-back MOVE must not leave dest graph"
    );
}

/// CLEAR GRAPH must RYW empty and restore triples on rollback.
#[test]
fn clear_graph_rolls_back() {
    let db = rdf_db();
    seed_named(&db, "http://ex.org/g1", "http://ex.org/s1", "a");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("CLEAR GRAPH <http://ex.org/g1>")
        .expect("clear");
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 0, "CLEAR RYW");
    session.rollback().expect("rollback");
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 1);
}

fn ex_triple(subject: &str, value: &str) -> Triple {
    Triple::new(
        Term::iri(subject),
        Term::iri("http://ex.org/p"),
        Term::literal(value),
    )
}

/// Typed named-graph quad batch spans one Session transaction.
#[test]
fn insert_rdf_quads_two_graphs_rollback_and_commit() {
    let db = rdf_db();
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let n = session
        .insert_rdf_quads([
            Quad::named(ex_triple("http://ex.org/s1", "a"), "http://ex.org/g1"),
            Quad::named(ex_triple("http://ex.org/s2", "b"), "http://ex.org/g2"),
        ])
        .expect("insert quads");
    assert_eq!(n, 2);
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 1);
    assert_eq!(count_in_graph(&session, "http://ex.org/g2"), 1);
    session.rollback().expect("rollback");
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 0);
    assert!(create_graph_ok(&session, "http://ex.org/g1"));

    session.begin_transaction().expect("begin 2");
    session
        .insert_rdf_quads([
            Quad::named(ex_triple("http://ex.org/s1", "a"), "http://ex.org/g1"),
            Quad::named(ex_triple("http://ex.org/s2", "b"), "http://ex.org/g2"),
        ])
        .unwrap();
    session.commit().expect("commit");
    let other = db.session();
    assert_eq!(count_in_graph(&other, "http://ex.org/g1"), 1);
    assert_eq!(count_in_graph(&other, "http://ex.org/g2"), 1);
}

/// One-shot GrafeoDB quad batch returns the crash-stable epoch.
#[test]
fn insert_rdf_quads_oneshot_returns_epoch() {
    let db = rdf_db();
    let (n, epoch) = db
        .insert_rdf_quads([
            Quad::named(ex_triple("http://ex.org/s1", "a"), "http://ex.org/g1"),
            Quad::named(ex_triple("http://ex.org/s2", "b"), "http://ex.org/g2"),
        ])
        .expect("oneshot quads");
    assert_eq!(n, 2);
    assert!(epoch.as_u64() > 0);
    assert_eq!(epoch, db.rdf_store_commit_epoch());
    let session = db.session();
    assert_eq!(count_in_graph(&session, "http://ex.org/g1"), 1);
    assert_eq!(count_in_graph(&session, "http://ex.org/g2"), 1);
}

/// Membership is a read barrier: uncommitted quads are visible to the writer only.
#[test]
fn contains_rdf_quad_session_ryw_read_barrier() {
    let db = rdf_db();
    let quad = Quad::named(ex_triple("http://ex.org/s1", "a"), "http://ex.org/g1");
    let mut writer = db.session();
    writer.begin_transaction().expect("begin");
    writer.insert_rdf_quads([quad.clone()]).expect("insert");
    assert!(
        writer.contains_rdf_quad(&quad),
        "writer must see its own uncommitted quad"
    );
    let reader = db.session();
    assert!(
        !reader.contains_rdf_quad(&quad),
        "other session must not see uncommitted quad"
    );
    writer.commit().expect("commit");
    assert!(reader.contains_rdf_quad(&quad));
    assert!(db.contains_rdf_quad(&quad));
}

/// Intra-batch duplicates are skipped in O(n), not via repeated pending scans.
#[test]
fn insert_rdf_quads_batch_local_dedup_10k() {
    use std::time::Instant;

    let db = rdf_db();
    let dup = Quad::named(ex_triple("http://ex.org/s", "x"), "http://ex.org/g");
    let dups = vec![dup.clone(); 10_000];
    let t0 = Instant::now();
    let (n, _) = db.insert_rdf_quads(dups).expect("dup batch");
    let dup_elapsed = t0.elapsed();
    assert_eq!(n, 1, "10k identical quads insert once");
    assert!(
        db.contains_rdf_quad(&dup),
        "surviving quad must be exactly the object inserted"
    );
    assert!(
        dup_elapsed.as_secs_f64() < 1.0,
        "10k intra-batch dedup must be linear, took {dup_elapsed:?}"
    );

    let unique: Vec<Quad> = (0..10_000)
        .map(|i| {
            Quad::named(
                ex_triple(&format!("http://ex.org/s{i}"), "v"),
                "http://ex.org/g",
            )
        })
        .collect();
    let t1 = Instant::now();
    let (n, _) = db.insert_rdf_quads(unique).expect("unique batch");
    let unique_elapsed = t1.elapsed();
    assert_eq!(n, 10_000);
    assert!(
        unique_elapsed.as_secs_f64() < 5.0,
        "10k unique quads must stay linear, took {unique_elapsed:?}"
    );
}

/// DROP GRAPH hides the graph in the tx and restores it on rollback.
#[test]
fn drop_graph_ryw_and_rollback() {
    let db = rdf_db();
    seed_named(&db, "http://ex.org/g1", "http://ex.org/s1", "a");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("DROP GRAPH <http://ex.org/g1>")
        .expect("drop");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        0,
        "DROP GRAPH must RYW-hide triples"
    );
    session.rollback().expect("rollback");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        1,
        "rollback must restore the dropped graph"
    );
}

/// DROP GRAPH commit removes the graph; another session cannot see it.
#[test]
fn drop_graph_commit_removes() {
    let db = rdf_db();
    seed_named(&db, "http://ex.org/g1", "http://ex.org/s1", "a");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("DROP GRAPH <http://ex.org/g1>")
        .expect("drop");
    session.commit().expect("commit");
    let other = db.session();
    assert_eq!(count_in_graph(&other, "http://ex.org/g1"), 0);
    assert!(
        create_graph_ok(&other, "http://ex.org/g1"),
        "committed DROP must remove the named graph"
    );
}

/// DROP GRAPH hides graph existence in the tx (CREATE succeeds); rollback restores it.
#[test]
fn drop_graph_hides_existence_until_commit() {
    let db = rdf_db();
    seed_named(&db, "http://ex.org/g1", "http://ex.org/s1", "a");
    let mut session = db.session();
    session.begin_transaction().expect("begin");
    session
        .execute_sparql("DROP GRAPH <http://ex.org/g1>")
        .expect("drop");
    assert!(
        create_graph_ok(&session, "http://ex.org/g1"),
        "DROP must hide graph existence so CREATE GRAPH succeeds in the same tx"
    );
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        0,
        "CREATE after DROP must see an empty graph"
    );
    session.rollback().expect("rollback");
    assert_eq!(
        count_in_graph(&session, "http://ex.org/g1"),
        1,
        "rollback must restore dropped graph triples"
    );
    let err = session.execute_sparql("CREATE GRAPH <http://ex.org/g1>");
    assert!(
        err.is_err(),
        "restored graph must exist after rollback, got {err:?}"
    );
}
