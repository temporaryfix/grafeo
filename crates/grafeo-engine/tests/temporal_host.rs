//! temporal-host host surface: CRUD, compact, dest-only as-of fill, statements.
//!
//! No GQL/Cypher — the application's compile set is `temporal-host` without query languages.
//! Mixed-label OBSERVED_BY (Source → Entity) is the dest-CSR OOB fixture.

#![cfg(all(
    feature = "lpg",
    feature = "compact-store",
    feature = "statement-table"
))]

use grafeo_common::types::{ContentId, EpochId, Value};
use grafeo_core::graph::Direction;
use grafeo_engine::{GrafeoDB, StatementIngest};

fn bump(db: &GrafeoDB) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.commit().unwrap();
}

fn ingest(n: u8) -> StatementIngest {
    let mut target = [0u8; 32];
    target[0] = n;
    StatementIngest {
        target_kind: 1,
        target_ref: target,
        target_row: u64::from(n),
        meta_kind: 0,
        assertion: vec![n],
        timestamp_ns: 1_000 + i64::from(n),
        source_id: [n; 32],
    }
}

#[test]
fn temporal_host_compact_fill_and_statements() {
    let mut db = GrafeoDB::new_in_memory();
    let entity = db.create_node_with_props(&["Entity"], [("id", Value::from("h"))]);
    let radar = db.create_node_with_props(&["Source"], [("id", Value::from("radar"))]);
    let lidar = db.create_node_with_props(&["Source"], [("id", Value::from("lidar"))]);
    db.create_edge(radar, entity, "OBSERVED_BY");
    let dead = db.create_edge(lidar, entity, "OBSERVED_BY");
    let e_open = db.current_epoch();

    bump(&db);
    assert!(db.delete_edge(dead));
    let e_del = db.current_epoch();
    bump(&db);

    db.compact()
        .expect("compact mixed-label Source→Entity OBSERVED_BY");

    let obs = ["OBSERVED_BY".to_string()];
    let mut buf = Vec::new();
    db.fill_neighbors_of_types_at_epoch(entity, Direction::Incoming, e_open, &obs, &mut buf);
    buf.sort_unstable();
    let mut open = vec![radar, lidar];
    open.sort_unstable();
    assert_eq!(
        buf, open,
        "as-of before close must include both observers (dest-only incoming)"
    );

    db.fill_neighbors_of_types_at_epoch(entity, Direction::Incoming, e_del, &obs, &mut buf);
    assert_eq!(
        buf,
        vec![radar],
        "closed OBSERVED_BY absent at/after delete epoch"
    );

    db.fill_neighbors_at_epoch(entity, Direction::Incoming, EpochId::PENDING, &mut buf);
    assert_eq!(buf, vec![radar], "PENDING incoming matches current");

    let ids = db
        .insert_statement_batch(&[ingest(1), ingest(2)], EpochId::new(10))
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(db.statements_at_epoch(EpochId::new(15)).len(), 2);
    assert_eq!(db.statements_at_valid_time(1001).len(), 1);
    let mut target = [0u8; 32];
    target[0] = 1;
    assert_eq!(db.statements_for_target(ContentId::from(target)).len(), 1);
}

/// First `compact()` keeps property history from Session CRUD (no GQL).
/// One-shot `GrafeoDB::set_node_property` writes at `current_epoch` and does
/// not open a transaction; Applications should commit versions on a [`grafeo_engine::Session`].
#[test]
fn first_compact_keeps_property_as_of_without_gql() {
    use grafeo_common::types::PropertyKey;

    let mut db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let n = session
        .create_node_with_props(
            &["Entity"],
            [("id", Value::from("track")), ("lat", Value::Float64(51.5))],
        )
        .unwrap();
    session.commit().unwrap();
    let e1 = db.current_epoch();

    session.begin_transaction().unwrap();
    session
        .set_node_property(n, "lat", Value::Float64(51.6))
        .unwrap();
    session.commit().unwrap();
    drop(session);

    db.compact().expect("first compact");

    let lat = PropertyKey::new("lat");
    let at_e1 = db
        .get_node_at_epoch(n, e1)
        .and_then(|node| node.properties.get(&lat).cloned());
    assert_eq!(
        at_e1,
        Some(Value::Float64(51.5)),
        "first compact must keep lat at e1 after Session commits"
    );
    assert_eq!(
        db.get_node_property_at_epoch(n, "lat", e1),
        Some(Value::Float64(51.5)),
        "single-key as-of must hit the cold base, not the empty overlay"
    );
}

/// Overlay write after `compact()` must not drop pre-compact property history.
/// Promotion copies current values into the overlay; as-of at e1 still lives
/// in the cold base and must remain visible without `recompact()`.
#[test]
fn overlay_after_compact_keeps_property_as_of_without_gql() {
    let mut db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let n = session
        .create_node_with_props(
            &["Entity"],
            [("id", Value::from("track")), ("lat", Value::Float64(51.5))],
        )
        .unwrap();
    session.commit().unwrap();
    let e1 = db.current_epoch();

    session.begin_transaction().unwrap();
    session
        .set_node_property(n, "lat", Value::Float64(51.6))
        .unwrap();
    session.commit().unwrap();
    drop(session);

    db.compact().expect("first compact");

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(n, "lat", Value::Float64(51.7))
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let e3 = db.current_epoch();

    assert_eq!(
        db.get_node_property_at_epoch(n, "lat", e1),
        Some(Value::Float64(51.5)),
        "overlay write after compact must not hide lat at e1"
    );
    assert_eq!(
        db.get_node_property_at_epoch(n, "lat", e3),
        Some(Value::Float64(51.7)),
        "current overlay value after compact"
    );
}

/// compact → close → reopen without recompact: property and dest-only as-of
/// must match RAM. A durable host cannot rely on persistence that drops the overlay
/// fold or rewrites the cold base as all-open.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn persist_as_of_after_compact_save_open_without_recompact() {
    use grafeo_engine::Config;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("host_asof.grafeo");

    let n;
    let entity;
    let radar;
    let lidar;
    let e1;
    let e_open;
    let e_del;
    {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        n = session
            .create_node_with_props(
                &["Entity"],
                [("id", Value::from("track")), ("lat", Value::Float64(51.5))],
            )
            .unwrap();
        session.commit().unwrap();
        e1 = db.current_epoch();

        session.begin_transaction().unwrap();
        session
            .set_node_property(n, "lat", Value::Float64(51.6))
            .unwrap();
        session.commit().unwrap();
        drop(session);

        entity = db.create_node_with_props(&["Entity"], [("id", Value::from("h"))]);
        radar = db.create_node_with_props(&["Source"], [("id", Value::from("radar"))]);
        lidar = db.create_node_with_props(&["Source"], [("id", Value::from("lidar"))]);
        db.create_edge(radar, entity, "OBSERVED_BY");
        let dead = db.create_edge(lidar, entity, "OBSERVED_BY");
        e_open = db.current_epoch();
        bump(&db);
        assert!(db.delete_edge(dead));
        e_del = db.current_epoch();
        bump(&db);

        db.compact().expect("compact before persist");
        assert_eq!(
            db.get_node_property_at_epoch(n, "lat", e1),
            Some(Value::Float64(51.5)),
            "RAM as-of after compact is the persist contract"
        );
        let obs = ["OBSERVED_BY".to_string()];
        let mut buf = Vec::new();
        db.fill_neighbors_of_types_at_epoch(entity, Direction::Incoming, e_open, &obs, &mut buf);
        buf.sort_unstable();
        let mut open = vec![radar, lidar];
        open.sort_unstable();
        assert_eq!(buf, open, "RAM dest-only as-of before persist");
        db.close().unwrap();
    }

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.get_node_property_at_epoch(n, "lat", e1),
        Some(Value::Float64(51.5)),
        "as-of after compact/save/open without recompact must keep lat at e1"
    );
    assert_eq!(
        db.get_node_property_at_epoch(n, "lat", db.current_epoch()),
        Some(Value::Float64(51.6)),
        "current value after reopen"
    );
    let obs = ["OBSERVED_BY".to_string()];
    let mut buf = Vec::new();
    db.fill_neighbors_of_types_at_epoch(entity, Direction::Incoming, e_open, &obs, &mut buf);
    buf.sort_unstable();
    let mut open = vec![radar, lidar];
    open.sort_unstable();
    assert_eq!(
        buf, open,
        "dest-only as-of before close must survive reopen without recompact"
    );
    db.fill_neighbors_of_types_at_epoch(entity, Direction::Incoming, e_del, &obs, &mut buf);
    assert_eq!(
        buf,
        vec![radar],
        "closed OBSERVED_BY absent at/after delete epoch after reopen"
    );
}

/// Overlay write after compact, then persist without recompact: as-of of
/// pre-compact and overlay versions must match RAM on reopen.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn persist_as_of_overlay_after_compact_without_recompact() {
    use grafeo_engine::Config;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("host_overlay_asof.grafeo");

    let n;
    let e1;
    let e3;
    {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        n = session
            .create_node_with_props(
                &["Entity"],
                [("id", Value::from("track")), ("lat", Value::Float64(51.5))],
            )
            .unwrap();
        session.commit().unwrap();
        e1 = db.current_epoch();

        session.begin_transaction().unwrap();
        session
            .set_node_property(n, "lat", Value::Float64(51.6))
            .unwrap();
        session.commit().unwrap();
        drop(session);

        db.compact().expect("compact");

        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .set_node_property(n, "lat", Value::Float64(51.7))
            .unwrap();
        session.commit().unwrap();
        drop(session);
        e3 = db.current_epoch();

        assert_eq!(
            db.get_node_property_at_epoch(n, "lat", e1),
            Some(Value::Float64(51.5)),
            "RAM overlay-after-compact as-of is the persist contract"
        );
        assert_eq!(
            db.get_node_property_at_epoch(n, "lat", e3),
            Some(Value::Float64(51.7))
        );
        db.close().unwrap();
    }

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.get_node_property_at_epoch(n, "lat", e1),
        Some(Value::Float64(51.5)),
        "overlay after compact must persist pre-compact as-of without recompact"
    );
    assert_eq!(
        db.get_node_property_at_epoch(n, "lat", e3),
        Some(Value::Float64(51.7)),
        "overlay write after compact must persist without recompact"
    );
}

/// In-memory compact then `save()` to `.grafeo` (no recompact) must keep as-of.
/// `export_snapshot` after compact currently dumps current values at epoch 0.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn save_grafeo_after_compact_keeps_as_of_without_recompact() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("saved_asof.grafeo");

    let mut db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let n = session
        .create_node_with_props(
            &["Entity"],
            [("id", Value::from("track")), ("lat", Value::Float64(51.5))],
        )
        .unwrap();
    session.commit().unwrap();
    let e1 = db.current_epoch();

    session.begin_transaction().unwrap();
    session
        .set_node_property(n, "lat", Value::Float64(51.6))
        .unwrap();
    session.commit().unwrap();
    drop(session);

    db.compact().expect("compact");
    db.save(&path).expect("save compacted db");

    let restored = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        restored.get_node_property_at_epoch(n, "lat", e1),
        Some(Value::Float64(51.5)),
        "save() after compact without recompact must keep lat at e1"
    );
}
