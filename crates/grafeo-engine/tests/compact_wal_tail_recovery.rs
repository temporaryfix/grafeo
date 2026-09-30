//! Hostile recovery qualification for LPG WAL records after a compact checkpoint.
//!
//! The child opens a sealed compact database, commits a broad mutation tail in
//! `DurabilityMode::Sync`, signals only after the commit acknowledgement, and is
//! then killed without `close()`. Reopen must apply the tail through the
//! LayeredStore rather than the otherwise-empty LPG overlay.

#![cfg(all(
    feature = "compact-store",
    feature = "grafeo-file",
    feature = "lpg",
    feature = "wal"
))]

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};

const CHILD_MODE: &str = "GRAFEO_COMPACT_WAL_TAIL_CHILD";
const DATABASE_PATH: &str = "GRAFEO_COMPACT_WAL_TAIL_PATH";
const READY_PATH: &str = "GRAFEO_COMPACT_WAL_TAIL_READY";
const BASE_A: &str = "GRAFEO_COMPACT_WAL_TAIL_BASE_A";
const BASE_B: &str = "GRAFEO_COMPACT_WAL_TAIL_BASE_B";
const VICTIM: &str = "GRAFEO_COMPACT_WAL_TAIL_VICTIM";
const MUTABLE_EDGE: &str = "GRAFEO_COMPACT_WAL_TAIL_MUTABLE_EDGE";
const DELETED_EDGE: &str = "GRAFEO_COMPACT_WAL_TAIL_DELETED_EDGE";

#[derive(Clone, Copy)]
struct Fixture {
    base_a: NodeId,
    base_b: NodeId,
    victim: NodeId,
    mutable_edge: EdgeId,
    deleted_edge: EdgeId,
    incoming: EdgeId,
    outgoing: EdgeId,
    self_loop: EdgeId,
}

#[derive(Clone, Copy)]
struct TailIdentities {
    node: NodeId,
    base_to_base: EdgeId,
    base_to_new: EdgeId,
}

fn open_sync(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(GraphModel::Lpg)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .expect("open persistent LPG database")
}

fn parse_node_env(name: &str) -> NodeId {
    NodeId::new(
        std::env::var(name)
            .unwrap_or_else(|_| panic!("missing child environment variable {name}"))
            .parse()
            .unwrap_or_else(|_| panic!("invalid node identity in {name}")),
    )
}

fn parse_edge_env(name: &str) -> EdgeId {
    EdgeId::new(
        std::env::var(name)
            .unwrap_or_else(|_| panic!("missing child environment variable {name}"))
            .parse()
            .unwrap_or_else(|_| panic!("invalid edge identity in {name}")),
    )
}

fn write_committed_tail(path: &Path, ready: &Path, fixture: Fixture) {
    let db = open_sync(path);
    assert!(
        grafeo_engine::database::testing::root_lpg_store(&db)
            .get_node(fixture.base_a)
            .is_none(),
        "the hostile fixture must begin in the compact tier, not an LPG snapshot"
    );

    let mut session = db.session();
    session
        .begin_transaction()
        .expect("begin compact WAL-tail transaction");

    session
        .set_node_property(fixture.base_a, "mode", Value::from("after-crash"))
        .expect("set property on compact-base node");
    session
        .set_node_property(fixture.base_a, "added", Value::Bool(true))
        .expect("add property to compact-base node");
    assert!(session.remove_node_property(fixture.base_a, "remove_me"));
    assert!(session.add_node_label(fixture.base_a, "Recovered"));
    assert!(session.remove_node_label(fixture.base_a, "RemoveMe"));

    session
        .set_edge_property(fixture.mutable_edge, "weight", Value::Int64(99))
        .expect("set property on compact-base edge");
    assert!(session.remove_edge_property(fixture.mutable_edge, "remove_me"));
    assert!(session.delete_edge(fixture.deleted_edge));
    assert!(
        session.delete_node(fixture.victim),
        "DETACH DELETE compact-base victim"
    );

    let node = session
        .create_node_with_props(
            &["TailNode"],
            [("origin", Value::from("post-checkpoint WAL"))],
        )
        .expect("create WAL-tail node");
    let base_to_base = session.create_edge(fixture.base_a, fixture.base_b, "TAIL_BASE_BASE");
    let base_to_new = session
        .create_edge_with_props(
            fixture.base_b,
            node,
            "TAIL_BASE_NEW",
            [("durable", Value::Bool(true))],
        )
        .expect("create base-to-new WAL-tail edge");
    assert!(node.is_valid());
    assert!(base_to_base.is_valid());
    assert!(base_to_new.is_valid());
    session
        .commit()
        .expect("Sync commit compact WAL mutation tail");

    let pending_ready = ready.with_extension("pending");
    std::fs::write(
        &pending_ready,
        format!(
            "{} {} {}\n",
            node.as_u64(),
            base_to_base.as_u64(),
            base_to_new.as_u64()
        ),
    )
    .expect("publish child-ready identities");
    std::fs::rename(pending_ready, ready).expect("atomically publish child-ready marker");

    // Retain every database object and wait for the parent to kill this process.
    // Reaching normal scope exit would checkpoint and invalidate the test.
    std::mem::forget(session);
    std::mem::forget(db);
    loop {
        std::thread::park();
    }
}

fn spawn_tail_child(path: &Path, ready: &Path, fixture: Fixture) -> Child {
    Command::new(std::env::current_exe().expect("current compact WAL test executable"))
        .arg("compact_post_checkpoint_wal_tail_survives_kill")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_MODE, "1")
        .env(DATABASE_PATH, path)
        .env(READY_PATH, ready)
        .env(BASE_A, fixture.base_a.as_u64().to_string())
        .env(BASE_B, fixture.base_b.as_u64().to_string())
        .env(VICTIM, fixture.victim.as_u64().to_string())
        .env(MUTABLE_EDGE, fixture.mutable_edge.as_u64().to_string())
        .env(DELETED_EDGE, fixture.deleted_edge.as_u64().to_string())
        .spawn()
        .expect("spawn compact WAL-tail crash child")
}

fn await_tail_and_kill(mut child: Child, ready: &Path) -> TailIdentities {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready.exists() {
        if let Some(status) = child.try_wait().expect("poll compact WAL-tail child") {
            panic!("compact WAL-tail child exited before its Sync acknowledgement: {status}");
        }
        assert!(
            Instant::now() < deadline,
            "compact WAL-tail child did not commit within 30 seconds"
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    let identities = std::fs::read_to_string(ready).expect("read committed tail identities");
    let mut identities = identities.split_whitespace().map(|value| {
        value
            .parse::<u64>()
            .expect("child emitted a valid graph identity")
    });
    let tail = TailIdentities {
        node: NodeId::new(identities.next().expect("tail node identity")),
        base_to_base: EdgeId::new(identities.next().expect("base-to-base edge identity")),
        base_to_new: EdgeId::new(identities.next().expect("base-to-new edge identity")),
    };
    assert!(
        identities.next().is_none(),
        "child emitted exactly three IDs"
    );

    child.kill().expect("kill compact WAL-tail child");
    let _ = child.wait().expect("reap compact WAL-tail child");
    tail
}

fn property(node: &grafeo_core::graph::lpg::Node, key: &str) -> Option<Value> {
    node.properties.get(&PropertyKey::new(key)).cloned()
}

#[test]
fn compact_post_checkpoint_wal_tail_survives_kill() {
    if std::env::var(CHILD_MODE).ok().as_deref() == Some("1") {
        let path = PathBuf::from(std::env::var(DATABASE_PATH).expect("child database path"));
        let ready = PathBuf::from(std::env::var(READY_PATH).expect("child ready path"));
        let fixture = Fixture {
            base_a: parse_node_env(BASE_A),
            base_b: parse_node_env(BASE_B),
            victim: parse_node_env(VICTIM),
            mutable_edge: parse_edge_env(MUTABLE_EDGE),
            deleted_edge: parse_edge_env(DELETED_EDGE),
            // The child needs only the identities directly mutated before its
            // crash; incident detach identities are asserted by the parent.
            incoming: EdgeId::INVALID,
            outgoing: EdgeId::INVALID,
            self_loop: EdgeId::INVALID,
        };
        write_committed_tail(&path, &ready, fixture);
        unreachable!("the parent must kill the parked crash child");
    }

    let directory = tempfile::tempdir().expect("temporary compact WAL directory");
    let path = directory.path().join("compact-wal-tail.grafeo");
    let ready = directory.path().join("tail-committed.ready");

    let fixture = {
        let mut db = open_sync(&path);
        let base_a = db.create_node_with_props(
            &["Base", "RemoveMe"],
            [
                ("mode", Value::from("checkpoint")),
                ("remove_me", Value::from("checkpoint-only")),
            ],
        );
        let base_b = db.create_node(&["Base"]);
        let victim = db.create_node(&["Victim"]);
        let mutable_edge = db.create_edge_with_props(
            base_a,
            base_b,
            "MUTABLE",
            [
                ("weight", Value::Int64(1)),
                ("remove_me", Value::from("checkpoint-only")),
            ],
        );
        let deleted_edge = db.create_edge(base_b, base_a, "DELETE_ME");
        let incoming = db.create_edge(base_a, victim, "INCOMING");
        let outgoing = db.create_edge(victim, base_b, "OUTGOING");
        let self_loop = db.create_edge(victim, victim, "SELF");
        let fixture = Fixture {
            base_a,
            base_b,
            victim,
            mutable_edge,
            deleted_edge,
            incoming,
            outgoing,
            self_loop,
        };

        db.compact()
            .expect("compact authoritative checkpoint prefix");
        let layered = db.layered_store().expect("compact installs LayeredStore");
        assert_eq!(layered.overlay_node_count(), 0);
        assert_eq!(layered.overlay_edge_count(), 0);
        db.wal_checkpoint()
            .expect("install compact checkpoint before hostile tail");
        db.close().expect("close compact checkpoint prefix");
        fixture
    };

    let child = spawn_tail_child(&path, &ready, fixture);
    let tail = await_tail_and_kill(child, &ready);

    let reopened = open_sync(&path);
    let base_a = reopened
        .get_node(fixture.base_a)
        .expect("recover mutated compact-base node");
    assert!(base_a.has_label("Base"));
    assert!(base_a.has_label("Recovered"));
    assert!(!base_a.has_label("RemoveMe"));
    assert_eq!(property(&base_a, "mode"), Some(Value::from("after-crash")));
    assert_eq!(property(&base_a, "added"), Some(Value::Bool(true)));
    assert_eq!(property(&base_a, "remove_me"), None);

    let mutable_edge = reopened
        .get_edge(fixture.mutable_edge)
        .expect("recover compact-base edge property mutations");
    assert_eq!(mutable_edge.get_property("weight"), Some(&Value::Int64(99)));
    assert_eq!(mutable_edge.get_property("remove_me"), None);
    assert!(reopened.get_edge(fixture.deleted_edge).is_none());

    assert!(reopened.get_node(fixture.victim).is_none());
    assert!(reopened.get_edge(fixture.incoming).is_none());
    assert!(reopened.get_edge(fixture.outgoing).is_none());
    assert!(reopened.get_edge(fixture.self_loop).is_none());

    let tail_node = reopened
        .get_node(tail.node)
        .expect("recover post-checkpoint exact node ID");
    assert!(tail_node.has_label("TailNode"));
    assert_eq!(
        property(&tail_node, "origin"),
        Some(Value::from("post-checkpoint WAL"))
    );
    let base_to_base = reopened
        .get_edge(tail.base_to_base)
        .expect("recover post-checkpoint base-to-base edge");
    assert_eq!(
        (base_to_base.src, base_to_base.dst),
        (fixture.base_a, fixture.base_b)
    );
    assert_eq!(base_to_base.edge_type.as_str(), "TAIL_BASE_BASE");
    let base_to_new = reopened
        .get_edge(tail.base_to_new)
        .expect("recover post-checkpoint base-to-new edge");
    assert_eq!(
        (base_to_new.src, base_to_new.dst),
        (fixture.base_b, tail.node)
    );
    assert_eq!(base_to_new.edge_type.as_str(), "TAIL_BASE_NEW");
    assert_eq!(
        base_to_new.get_property("durable"),
        Some(&Value::Bool(true))
    );

    let generated_node = reopened.create_node(&["AfterRecovery"]);
    let generated_edge = reopened.create_edge(fixture.base_a, generated_node, "AFTER_RECOVERY");
    assert!(generated_node.as_u64() > tail.node.as_u64());
    assert!(
        generated_edge.as_u64() > tail.base_to_base.as_u64().max(tail.base_to_new.as_u64()),
        "recovery must raise the edge allocator above every exact WAL identity"
    );
}
