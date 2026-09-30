//! Mixed snapshot is per-database; open transactions cannot take a snapshot.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,triple-store,sparql" --test mixed_snapshot_bound -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "sparql",
    feature = "triple-store"
))]

use grafeo_engine::{Config, GrafeoDB, GraphModel};

fn both() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap()
}

#[test]
fn snapshot_on_a_does_not_skip_b_publication() {
    let db_a = both();
    let db_b = both();
    let session_a = db_a.session();
    let snap_a = session_a.snapshot().expect("snapshot A");
    let mut session_b = db_b.session();
    session_b.begin_transaction().unwrap();
    session_b.execute("INSERT (:Person {name: 'B'})").unwrap();
    session_b
        .execute_sparql(r#"INSERT DATA { <http://ex.org/b> <http://ex.org/p> "v" }"#)
        .unwrap();
    session_b.commit().unwrap();
    drop(snap_a);
    let n = db_b
        .session()
        .execute("MATCH (n:Person) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    assert_eq!(
        n, 1,
        "commit on B must publish even while A holds a mixed snapshot"
    );
}

#[test]
fn snapshot_rejects_open_transaction() {
    let db = both();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'x'})").unwrap();
    assert!(
        session.snapshot().is_err(),
        "snapshot with an explicit open transaction must be rejected"
    );
    session.rollback().unwrap();
}
