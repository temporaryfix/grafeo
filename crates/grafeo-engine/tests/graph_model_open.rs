//! Stored GraphModel is enforced on writable and read-only open.
//!
//! ```text
//! cargo test -p grafeo-engine --features "lpg,gql,triple-store,sparql,wal,grafeo-file" --test graph_model_open -- --test-threads=1
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "triple-store",
    feature = "wal",
    feature = "grafeo-file"
))]

use grafeo_common::storage::SectionType;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB, GraphModel};
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const MODEL_FIXTURES: [(&str, GraphModel, &[u8]); 3] = [
    (
        "graph_model_lpg.grafeo",
        GraphModel::Lpg,
        include_bytes!("fixtures/graph_model_lpg.grafeo"),
    ),
    (
        "graph_model_rdf.grafeo",
        GraphModel::Rdf,
        include_bytes!("fixtures/graph_model_rdf.grafeo"),
    ),
    (
        "graph_model_both.grafeo",
        GraphModel::Both,
        include_bytes!("fixtures/graph_model_both.grafeo"),
    ),
];

fn check_current_catalog(path: &std::path::Path, model: GraphModel) -> TestResult {
    let manager = GrafeoFileManager::open_read_only(path)?;
    let directory = manager
        .read_section_directory()?
        .ok_or("missing directory")?;
    let catalog = directory
        .entries()
        .iter()
        .find(|entry| entry.section_type == SectionType::Catalog)
        .ok_or("missing Catalog")?;
    let expected = if model == GraphModel::Rdf { 2 } else { 7 };
    assert_eq!(
        catalog.version, expected,
        "current Catalog directory {model:?}"
    );
    assert_eq!(manager.read_section_data(catalog)?.first(), Some(&expected));
    if model != GraphModel::Rdf {
        let lpg = directory
            .entries()
            .iter()
            .find(|entry| entry.section_type == SectionType::LpgStore)
            .ok_or("missing LPG section")?;
        assert_eq!(lpg.version, 4);
        assert!(manager.read_section_data(lpg)?.starts_with(b"LPG4\x03"));
    }
    let cdc = directory
        .entries()
        .iter()
        .find(|entry| entry.section_type == SectionType::Cdc)
        .ok_or("missing CDC")?;
    assert_eq!(cdc.version, 1);
    manager.close()?;
    Ok(())
}

/// Regenerate only after an intentional current-format cut, then commit the
/// three outputs. Run this ignored test explicitly; ordinary tests never write
/// repository fixtures. `GRAFEO_GRAPH_MODEL_FIXTURE_DIR` redirects generated
/// bytes to a qualification output directory. File identities/timestamps are
/// fresh, not deterministic.
#[test]
#[ignore = "one-shot current GraphModel fixture generator"]
fn regenerate_graph_model_fixtures() -> TestResult {
    let output = std::env::var_os("GRAFEO_GRAPH_MODEL_FIXTURE_DIR").map_or_else(
        || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        std::path::PathBuf::from,
    );
    std::fs::create_dir_all(&output)?;
    for (name, model, _) in MODEL_FIXTURES {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(name);
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(model))?;
        db.save(&path)?;
        db.close()?;
        check_current_catalog(&path, model)?;
        let reopened = GrafeoDB::open_read_only(&path)?;
        assert_eq!(reopened.graph_model(), model);
        drop(reopened);
        let bytes = std::fs::read(&path)?;
        let target = output.join(name);
        std::fs::write(target, &bytes)?;
        println!("Wrote {name}: {} bytes, model {model:?}", bytes.len());
    }
    Ok(())
}

fn write_model(path: &std::path::Path, model: GraphModel) {
    let db = GrafeoDB::with_config(
        Config::persistent(path)
            .with_graph_model(model)
            .with_wal_durability(DurabilityMode::Sync),
    )
    .unwrap();
    db.close().unwrap();
}

#[test]
fn writable_and_read_only_open_adopt_stored_model() {
    for model in [GraphModel::Lpg, GraphModel::Both] {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("m.grafeo");
        write_model(&path, model);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.graph_model(), model, "writable open {model:?}");
        db.close().unwrap();
        let ro = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(ro.graph_model(), model, "read-only open {model:?}");
    }
}

#[cfg(feature = "triple-store")]
#[test]
fn rdf_file_round_trips_model() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rdf.grafeo");
    write_model(&path, GraphModel::Rdf);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.graph_model(), GraphModel::Rdf);
    db.close().unwrap();
    let ro = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(ro.graph_model(), GraphModel::Rdf);
}

#[test]
fn pinned_mismatch_is_refused_writable_and_read_only() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("pin.grafeo");
    write_model(&path, GraphModel::Lpg);
    let err = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_graph_model(GraphModel::Rdf)
            .with_wal_durability(DurabilityMode::Sync),
    );
    assert!(err.is_err(), "pinned RDF open of LPG file must fail");
    let err = GrafeoDB::with_config(Config::read_only(&path).with_graph_model(GraphModel::Rdf));
    assert!(err.is_err(), "pinned RDF read-only of LPG file must fail");
}

#[test]
fn committed_graph_model_fixtures_round_trip() -> TestResult {
    for (name, model, bytes) in MODEL_FIXTURES {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(name);
        std::fs::write(&path, bytes)?;
        check_current_catalog(&path, model)?;
        // Read-only must accept the committed bytes before any writer can
        // checkpoint them into a different image.
        let ro = GrafeoDB::open_read_only(&path)?;
        assert_eq!(ro.graph_model(), model, "read-only fixture {model:?}");
        drop(ro);
        assert_eq!(std::fs::read(&path)?, bytes);
        let db = GrafeoDB::open(&path)?;
        assert_eq!(db.graph_model(), model, "writable fixture {model:?}");
        db.close()?;
    }
    Ok(())
}

#[test]
fn predecessor_without_cdc_is_refused_before_writable_or_read_only_open() -> TestResult {
    // Authentic predecessors lack the now-required durable CDC section. They
    // must fail that outer admission before any inner LPG version is decoded.
    let predecessors: [&[u8]; 4] = [
        include_bytes!("fixtures/rejected_graph_model_lpg_wire1.grafeo"),
        include_bytes!("fixtures/rejected_graph_model_both_wire1.grafeo"),
        include_bytes!("fixtures/rejected_graph_model_lpg_wire2.grafeo"),
        include_bytes!("fixtures/rejected_graph_model_both_wire2.grafeo"),
    ];
    for bytes in predecessors {
        for read_only in [true, false] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("predecessor.grafeo");
            std::fs::write(&path, bytes)?;
            let config = if read_only {
                Config::read_only(&path)
            } else {
                Config::persistent(&path)
            };
            let error = GrafeoDB::with_config(config)
                .err()
                .ok_or("predecessor LPG wire was opened")?;
            assert!(
                error
                    .to_string()
                    .contains("requires exactly 1 Cdc section(s), found 0"),
                "{error}"
            );
            assert_eq!(std::fs::read(&path)?, bytes);
        }
    }
    Ok(())
}

// Rebuild both logical and physical seals so hostile payloads reach the
// intended admission boundary instead of failing an unrelated checksum.
fn write_sealed_lpg_sections(
    file: &GrafeoFileManager,
    sections: &mut [(SectionType, u8, Vec<u8>)],
) -> TestResult {
    use grafeo_common::types::{
        AuthoritativeFormat, GraphModelTag, RecoveryImageComponent, RecoveryImageCoordinatesV1,
        WorldCut, WorldMetadataSectionV2,
    };
    let header = file.active_header();
    let metadata_index = sections
        .iter()
        .position(|entry| entry.0 == SectionType::WorldMetadata)
        .ok_or("missing metadata")?;
    let metadata = WorldMetadataSectionV2::decode(&sections[metadata_index].2)?;
    let descriptor = metadata.cut().descriptor().clone();
    let mut logical = Vec::new();
    for format in descriptor.formats() {
        let kind = match format.format() {
            AuthoritativeFormat::Lpg => SectionType::LpgStore,
            AuthoritativeFormat::Catalog => SectionType::Catalog,
            AuthoritativeFormat::Cdc => SectionType::Cdc,
            _ => return Err("unexpected LPG format".into()),
        };
        let entry = sections
            .iter()
            .find(|entry| entry.0 == kind)
            .ok_or("missing component")?;
        logical.push((*format, entry.2.as_slice()));
    }
    let cut = WorldCut::seal_components(descriptor.clone(), &logical)?;
    let coordinates = RecoveryImageCoordinatesV1::new(
        header.epoch,
        header.transaction_id,
        GraphModelTag::Lpg,
        header.node_count,
        header.edge_count,
    );
    let mut physical = sections
        .iter()
        .filter(|entry| entry.0 != SectionType::WorldMetadata)
        .map(|entry| RecoveryImageComponent::new(entry.0 as u32, u16::from(entry.1), &entry.2))
        .collect::<Result<Vec<_>, _>>()?;
    physical.push(coordinates.component());
    sections[metadata_index].2 = WorldMetadataSectionV2::seal(cut, &physical)?.encode()?;
    let writes: Vec<_> = sections
        .iter()
        .map(|entry| SectionWrite::new(entry.0, entry.1, &entry.2))
        .collect();
    file.write_versioned_sections(
        &writes,
        header.epoch,
        header.transaction_id,
        header.node_count,
        header.edge_count,
    )?;
    Ok(())
}

#[test]
fn predecessor_lpg_wire_versions_are_rejected_in_current_container() -> TestResult {
    for version in [1, 2] {
        for read_only in [true, false] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("old-lpg-wire.grafeo");
            std::fs::write(&path, MODEL_FIXTURES[0].2)?;
            let file = GrafeoFileManager::open(&path)?;
            let directory = file.read_section_directory()?.ok_or("missing directory")?;
            let mut sections = directory
                .entries()
                .iter()
                .map(|entry| {
                    Ok((
                        entry.section_type,
                        entry.version,
                        file.read_section_data(entry)?,
                    ))
                })
                .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
            let lpg = &mut sections
                .iter_mut()
                .find(|entry| entry.0 == SectionType::LpgStore)
                .ok_or("missing LPG")?
                .2;
            assert!(lpg.starts_with(b"LPG4\x03"));
            lpg[4] = version;
            write_sealed_lpg_sections(&file, &mut sections)?;
            file.close()?;
            drop(file);
            let before = std::fs::read(&path)?;
            let config = if read_only {
                Config::read_only(&path)
            } else {
                Config::persistent(&path)
            };
            let error = GrafeoDB::with_config(config)
                .err()
                .ok_or("predecessor LPG wire opened")?;
            assert!(
                error
                    .to_string()
                    .contains(&format!("unsupported LPG v4 wire version {version}")),
                "{error}"
            );
            assert_eq!(std::fs::read(&path)?, before);
        }
    }
    Ok(())
}

#[test]
fn sealed_container_cannot_promote_a_named_incarnation_to_default() -> TestResult {
    use grafeo_common::storage::Section;
    use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection};
    use std::sync::Arc;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("named-as-default.grafeo");
    std::fs::write(&path, MODEL_FIXTURES[0].2)?;
    let file = GrafeoFileManager::open(&path)?;
    let directory = file.read_section_directory()?.ok_or("missing directory")?;
    let mut sections = directory
        .entries()
        .iter()
        .map(|entry| {
            Ok((
                entry.section_type,
                entry.version,
                file.read_section_data(entry)?,
            ))
        })
        .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
    let owner = LpgStore::new()?;
    let named = owner.graph_or_create("named")?;
    let named_bytes = LpgStoreSection::new(named).serialize()?;
    // Relative named sections remain valid core images. Whole databases must
    // additionally require their model's default identity, even with valid seals.
    let detached = Arc::new(LpgStore::new()?);
    LpgStoreSection::new(Arc::clone(&detached)).deserialize(&named_bytes)?;
    assert!(!detached.graph_incarnation_id().is_default_graph());
    sections
        .iter_mut()
        .find(|entry| entry.0 == SectionType::LpgStore)
        .ok_or("missing LPG")?
        .2 = named_bytes;
    write_sealed_lpg_sections(&file, &mut sections)?;
    file.close()?;
    drop(file);
    let before = std::fs::read(&path)?;
    for read_only in [false, true] {
        let config = if read_only {
            Config::read_only(&path)
        } else {
            Config::persistent(&path)
        };
        let error = GrafeoDB::with_config(config)
            .err()
            .ok_or("named default admitted")?;
        assert!(
            error
                .to_string()
                .contains("container default LPG graph has a named incarnation"),
            "{error}"
        );
        assert_eq!(std::fs::read(&path)?, before);
    }
    Ok(())
}

#[test]
fn obsolete_rdf_catalog_directory_is_rejected_before_open() -> TestResult {
    for read_only in [true, false] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("obsolete.grafeo");
        std::fs::write(&path, include_bytes!("fixtures/graph_model_rdf.grafeo"))?;
        let manager = GrafeoFileManager::open(&path)?;
        let header = manager.active_header();
        let directory = manager
            .read_section_directory()?
            .ok_or("missing directory")?;
        let sections = directory
            .entries()
            .iter()
            .map(|entry| {
                let version = if entry.section_type == SectionType::Catalog {
                    1
                } else {
                    entry.version
                };
                Ok((
                    entry.section_type,
                    version,
                    manager.read_section_data(entry)?,
                ))
            })
            .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
        let writes: Vec<_> = sections
            .iter()
            .map(|(kind, version, bytes)| SectionWrite::new(*kind, *version, bytes))
            .collect();
        manager.write_versioned_sections(
            &writes,
            header.epoch,
            header.transaction_id,
            header.node_count,
            header.edge_count,
        )?;
        manager.close()?;
        let config = if read_only {
            Config::read_only(&path)
        } else {
            Config::persistent(&path)
        };
        let error = GrafeoDB::with_config(config)
            .err()
            .ok_or("obsolete Catalog accepted")?;
        assert!(
            error
                .to_string()
                .contains("unsupported Catalog section directory version 1"),
            "expected version admission failure, got {error}"
        );
    }
    Ok(())
}
