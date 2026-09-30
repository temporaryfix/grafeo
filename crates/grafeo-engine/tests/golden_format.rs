//! Golden fixture tests for binary format stability.
//!
//! These tests deserialize a **committed** current binary fixture
//! (`snapshot_v12.bin`) and verify the current code can still read it correctly.
//! A separate v10 fixture verifies predecessor rejection. If a code change
//! silently alters the binary layout (bincode bump, serde derive change, enum
//! variant reorder), these tests fail immediately.
//! Predecessor headers and hostile bodies pin rejection before payload decoding.
//!
//! Three layers of protection:
//! 1. **Current-read**: load committed current bytes, assert correct data
//! 2. **Round-trip**: import golden -> export -> re-import, verify no data loss
//! 3. **Byte-equality**: export from identical graph, assert exact byte match
//!
//! ## When these tests fail
//!
//! - **Accidental breakage** (no `SNAPSHOT_VERSION` bump): fix the regression.
//! - **Intentional format change** (version bumped):
//!   1. Retain predecessor fixtures only as rejection controls.
//!   2. Regenerate:
//!      ```
//!      cargo test --all-features -p grafeo-engine --test golden_format -- regenerate_snapshot_fixture --ignored
//!      ```
//!   3. Name the new fixture for the new version (e.g. `snapshot_v12.bin`).
//!   4. Update `EXPECTED_SNAPSHOT_VERSION` and `golden_bytes()` below
//!   5. Commit the new current fixture and predecessor rejection tests.

#![cfg(feature = "lpg")]

mod generate_fixture;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Must match `SNAPSHOT_VERSION` in `persistence.rs`.
const EXPECTED_SNAPSHOT_VERSION: u8 = 12;

/// Load the committed golden fixture bytes.
fn golden_bytes() -> TestResult<Vec<u8>> {
    // Runtime loading lets the explicitly invoked generator compile before a
    // new fixture exists. Missing bytes are a test error, never a skipped gate.
    Ok(std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v12.bin"
    ))?)
}

fn predecessor_golden_bytes() -> TestResult<Vec<u8>> {
    Ok(std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v10.bin"
    ))?)
}

#[test]
fn predecessor_snapshot_rejected_before_import_or_restore() -> TestResult {
    let legacy = predecessor_golden_bytes()?;
    let current = golden_bytes()?;
    let previous = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v11.bin"
    ))?;
    for version in 0..EXPECTED_SNAPSHOT_VERSION {
        for payload in [&legacy, &previous, &current] {
            let mut bytes = payload.clone();
            bytes[0] = version;
            assert!(
                grafeo_engine::snapshot_info(&bytes)
                    .expect_err("predecessor snapshot metadata must reject")
                    .to_string()
                    .contains("unsupported snapshot version")
            );
            let error = GrafeoDB::import_snapshot(&bytes)
                .err()
                .ok_or("predecessor snapshot was imported")?;
            assert!(error.to_string().contains("unsupported snapshot version"));
            for populated in [false, true] {
                let target = GrafeoDB::new_in_memory();
                if populated {
                    target
                        .session()
                        .execute("INSERT (:Preserved {value: 42})")?;
                }
                let before = target.export_snapshot()?;
                let error = target
                    .restore_snapshot(&bytes)
                    .expect_err("predecessor snapshot must not restore");
                assert!(error.to_string().contains("unsupported snapshot version"));
                assert_eq!(target.export_snapshot()?, before);
            }
        }
        let hostile = [version, 255, 255, 255, 255, 255, 255, 255, 255];
        let error = GrafeoDB::import_snapshot(&hostile)
            .err()
            .ok_or("hostile predecessor snapshot was imported")?;
        assert!(error.to_string().contains("unsupported snapshot version"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Current-read tests: can today's code deserialize the committed current bytes?
// ---------------------------------------------------------------------------

#[test]
fn golden_snapshot_version_byte() -> TestResult {
    let bytes = golden_bytes()?;
    assert_eq!(
        bytes[0], EXPECTED_SNAPSHOT_VERSION,
        "fixture version byte ({}) does not match EXPECTED_SNAPSHOT_VERSION ({}), regenerate the fixture",
        bytes[0], EXPECTED_SNAPSHOT_VERSION,
    );
    Ok(())
}

#[test]
fn golden_import_succeeds() -> TestResult {
    GrafeoDB::import_snapshot(&golden_bytes()?)?;
    Ok(())
}

#[cfg(feature = "triple-store")]
#[test]
fn golden_lpg_import_binds_empty_rdf_facade_to_world_identity() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let history = db.rdf_dataset_history()?;

    assert_eq!(history.store_id(), db.store_id());
    assert_eq!(history.completeness(), db.world_identity().history());
    assert!(history.graph_lives().is_empty());
    assert!(history.quad_versions().is_empty());
    Ok(())
}

#[test]
fn golden_node_count() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    assert_eq!(db.node_count(), 3);
    Ok(())
}

#[test]
fn golden_edge_count() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    assert_eq!(db.edge_count(), 2);
    Ok(())
}

#[test]
fn golden_node_labels() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let session = db.session();

    let persons = session.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")?;
    assert_eq!(persons.rows().len(), 2);
    assert_eq!(persons.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(persons.rows()[1][0], Value::String("Gus".into()));

    let companies = session.execute("MATCH (c:Company) RETURN c.name")?;
    assert_eq!(companies.rows().len(), 1);
    assert_eq!(companies.rows()[0][0], Value::String("Acme Corp".into()));
    Ok(())
}

#[test]
fn golden_node_properties() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let session = db.session();

    let result = session.execute("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.age")?;
    assert_eq!(result.rows()[0][0], Value::Int64(30));

    let result = session.execute("MATCH (p:Person) WHERE p.name = 'Gus' RETURN p.age")?;
    assert_eq!(result.rows()[0][0], Value::Int64(25));
    Ok(())
}

#[test]
fn golden_multi_labels() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let session = db.session();

    let employees = session.execute("MATCH (e:Employee) RETURN e.name")?;
    assert_eq!(employees.rows().len(), 1);
    assert_eq!(employees.rows()[0][0], Value::String("Gus".into()));
    Ok(())
}

#[test]
fn golden_edge_types_and_properties() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let session = db.session();

    let knows = session.execute("MATCH ()-[e:KNOWS]->() RETURN e.since")?;
    assert_eq!(knows.rows().len(), 1);
    assert_eq!(knows.rows()[0][0], Value::Int64(2020));

    let works = session.execute("MATCH ()-[e:WORKS_AT]->() RETURN e.role")?;
    assert_eq!(works.rows().len(), 1);
    assert_eq!(works.rows()[0][0], Value::String("Engineer".into()));
    Ok(())
}

#[test]
fn golden_edge_connectivity() -> TestResult {
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let session = db.session();

    // Alix -[:KNOWS]-> Gus
    let result = session.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")?;
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("Gus".into()));

    // Gus -[:WORKS_AT]-> Acme Corp
    let result =
        session.execute("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")?;
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[0][1], Value::String("Acme Corp".into()));
    Ok(())
}

// ---------------------------------------------------------------------------
// Round-trip through golden fixture: import -> re-export -> re-import
// ---------------------------------------------------------------------------

#[test]
fn golden_round_trip_preserves_data() -> TestResult {
    // Import the golden fixture, re-export, re-import, verify data survives.
    // This catches format drift that is not backward-compatible (e.g. a field
    // type change that serializes but produces different logical data).
    let db1 = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let re_exported = db1.export_snapshot()?;
    let db2 = GrafeoDB::import_snapshot(&re_exported)?;

    assert_eq!(db2.node_count(), 3);
    assert_eq!(db2.edge_count(), 2);

    let session = db2.session();

    let result = session.execute("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.age")?;
    assert_eq!(result.rows()[0][0], Value::Int64(30));

    let result = session.execute("MATCH ()-[e:KNOWS]->() RETURN e.since")?;
    assert_eq!(result.rows()[0][0], Value::Int64(2020));

    let result =
        session.execute("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")?;
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[0][1], Value::String("Acme Corp".into()));
    Ok(())
}

// ---------------------------------------------------------------------------
// Byte-equality stability: deterministic export enables exact comparison
// ---------------------------------------------------------------------------

#[test]
fn golden_byte_equality() -> TestResult {
    // Re-export the same logical store and check byte-for-byte identity.
    // StoreId intentionally differs across separately-created databases, so
    // deterministic serialization is defined for one preserved identity.
    let db = GrafeoDB::import_snapshot(&golden_bytes()?)?;
    let fresh_bytes = db.export_snapshot()?;
    let golden = golden_bytes()?;

    assert_eq!(
        fresh_bytes.len(),
        golden.len(),
        "snapshot byte length changed: golden={}, fresh={}, format may have drifted",
        golden.len(),
        fresh_bytes.len(),
    );

    assert_eq!(
        fresh_bytes.as_slice(),
        golden.as_slice(),
        "snapshot bytes differ despite same graph data, export may not be deterministic",
    );
    Ok(())
}

fn assert_predecessor_rejected(version: u8) -> TestResult {
    let mut current_body = golden_bytes()?;
    *current_body.first_mut().ok_or("current golden is empty")? = version;
    for bytes in [
        vec![version],
        vec![version, 255, 255, 255, 255, 255],
        current_body,
    ] {
        assert_unsupported_version(&bytes, version)?;
    }
    Ok(())
}

fn assert_unsupported_version(bytes: &[u8], version: u8) -> TestResult {
    use grafeo_common::utils::error::Error;

    assert_eq!(bytes.first(), Some(&version));
    let Err(error) = GrafeoDB::import_snapshot(bytes) else {
        return Err(format!("snapshot v{version} unexpectedly imported").into());
    };
    assert!(
        matches!(&error, Error::Serialization(message)
        if message.contains(&format!("unsupported snapshot version: {version}"))),
        "predecessor must fail at version admission: {error}"
    );
    let info = grafeo_engine::snapshot_info(bytes);
    assert!(
        matches!(&info, Err(Error::Serialization(message))
        if message.contains(&format!("unsupported snapshot version: {version}"))),
        "metadata inspection must use the same version admission: {info:?}"
    );
    Ok(())
}

#[test]
fn golden_v4_header_is_rejected() -> TestResult {
    assert_predecessor_rejected(4)
}

#[test]
fn golden_v5_header_is_rejected() -> TestResult {
    assert_predecessor_rejected(5)
}

#[test]
fn golden_v7_header_is_rejected() -> TestResult {
    assert_predecessor_rejected(7)
}

#[test]
fn golden_v8_header_is_rejected() -> TestResult {
    assert_predecessor_rejected(8)
}

#[test]
fn golden_v6_header_is_rejected() -> TestResult {
    assert_predecessor_rejected(6)
}

#[test]
fn golden_v9_header_is_rejected() -> TestResult {
    assert_predecessor_rejected(9)
}

#[test]
fn pre_cdc_snapshot12_outer_schema_is_rejected_unchanged() -> TestResult {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v12_pre_cdc.bin"
    ))?;
    assert_eq!(bytes[0], EXPECTED_SNAPSHOT_VERSION);
    assert!(
        grafeo_engine::snapshot_info(&bytes)
            .expect_err("unprefixed Snapshot12")
            .to_string()
            .contains("outer schema")
    );
    assert!(GrafeoDB::import_snapshot(&bytes).is_err());
    let target = GrafeoDB::new_in_memory();
    target.create_node(&["Preserved"]);
    let before = target.export_snapshot()?;
    assert!(target.restore_snapshot(&bytes).is_err());
    assert_eq!(target.export_snapshot()?, before);
    Ok(())
}
