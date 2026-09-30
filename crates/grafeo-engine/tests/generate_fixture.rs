//! One-shot helper to regenerate the current golden snapshot fixture.
//!
//! Run with: cargo test --all-features -p grafeo-engine --test golden_format -- regenerate_snapshot_fixture --ignored
//! By default this writes `tests/fixtures/snapshot_v12.bin`; set
//! `GRAFEO_SNAPSHOT_FIXTURE_PATH` to write to an explicit path instead.
//! This test is ignored by default.
//!
//! **When to regenerate:** only after an intentional snapshot format change
//! (i.e. bumping `SNAPSHOT_VERSION`). If the golden test in `golden_format.rs`
//! fails without a version bump, fix the regression instead of regenerating.

#![cfg(feature = "lpg")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const CURRENT_SNAPSHOT_VERSION: u8 = 12;

/// Builds the canonical fixture graph used by both the generator and the
/// golden format tests. Keep this in sync with the assertions in
/// `golden_format.rs`.
///
/// # Errors
///
/// Returns an error if creating the fixture graph or its property values fails.
pub fn build_fixture_db() -> TestResult<GrafeoDB> {
    let db = GrafeoDB::new_in_memory();

    // 3 nodes with different labels and properties
    let alix = db.create_node(&["Person"]);
    db.set_node_property(alix, "name", Value::String("Alix".into()))?;
    db.set_node_property(alix, "age", Value::Int64(30))?;

    let gus = db.create_node(&["Person", "Employee"]);
    db.set_node_property(gus, "name", Value::String("Gus".into()))?;
    db.set_node_property(gus, "age", Value::Int64(25))?;

    let acme = db.create_node(&["Company"]);
    db.set_node_property(acme, "name", Value::String("Acme Corp".into()))?;

    // 2 edges with properties
    let knows = db.create_edge(alix, gus, "KNOWS");
    db.set_edge_property(knows, "since", Value::Int64(2020))?;

    let works = db.create_edge(gus, acme, "WORKS_AT");
    db.set_edge_property(works, "role", Value::String("Engineer".into()))?;

    Ok(db)
}

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn regenerate_snapshot_fixture() -> TestResult {
    let db = build_fixture_db()?;
    let bytes = db.export_snapshot()?;
    assert_eq!(bytes.first(), Some(&CURRENT_SNAPSHOT_VERSION));

    let default_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v12.bin"
    );
    let path = std::env::var_os("GRAFEO_SNAPSHOT_FIXTURE_PATH").map_or_else(
        || std::path::PathBuf::from(default_path),
        std::path::PathBuf::from,
    );
    std::fs::write(&path, &bytes)?;
    println!("Wrote {} bytes to {}", bytes.len(), path.display());
    println!("Snapshot version: {CURRENT_SNAPSHOT_VERSION}");
    Ok(())
}
