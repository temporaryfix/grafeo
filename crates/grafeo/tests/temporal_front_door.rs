//! Public-facade witnesses for the temporal user guide.

#![cfg(any(feature = "native", feature = "gql"))]

use grafeo::{Config, GrafeoDB, Result, Value};

#[test]
fn committed_epoch_property_example() -> Result<()> {
    let db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    let mut session = db.session();

    session.begin_transaction()?;
    let id = session.create_node_with_props(&["Asset"], [("status", Value::from("ready"))])?;
    let recorded = session.commit()?;

    session.begin_transaction()?;
    session.set_node_property(id, "status", Value::from("running"))?;
    session.commit()?;

    assert_eq!(
        db.get_node_property_at_epoch(id, "status", recorded),
        Some(Value::from("ready")),
    );
    Ok(())
}

#[cfg(feature = "compact-store")]
#[test]
fn managed_compaction_preserves_the_cut_not_a_property_revision_feed() -> Result<()> {
    let mut db = GrafeoDB::with_config(Config::in_memory().with_gc_interval(0))?;
    let mut session = db.session();
    session.begin_transaction()?;
    let id = session.create_node_with_props(&["Asset"], [("status", Value::from("ready"))])?;
    let recorded = session.commit()?;
    session.begin_transaction()?;
    session.set_node_property(id, "status", Value::from("running"))?;
    session.commit()?;
    drop(session);
    db.compact()?;

    assert_eq!(
        db.get_node_property_at_epoch(id, "status", recorded),
        Some(Value::from("ready")),
    );
    assert_eq!(
        db.get_node_property_at_epoch(id, "status", db.current_epoch()),
        Some(Value::from("running"))
    );
    let lifetimes = db.get_node_history(id);
    assert_eq!(lifetimes.len(), 1);
    assert_eq!(lifetimes[0].0, recorded);
    assert_eq!(lifetimes[0].1, None);
    assert_eq!(
        lifetimes[0].2.get_property("status"),
        Some(&Value::from("ready"))
    );

    #[cfg(feature = "gql")]
    assert_eq!(
        db.session()
            .execute_at_epoch("MATCH (a:Asset) RETURN a.status", recorded)?
            .rows(),
        &[vec![Value::from("ready")]],
    );
    Ok(())
}
