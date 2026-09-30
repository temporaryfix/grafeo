//! Unpublished current-format history input, then only real product callers.

use super::{TestResult, data_paths, persistent, view};
use grafeo_common::storage::{Section, SectionType};
use grafeo_common::types::{
    AuthoritativeFormat, EdgeId, EpochId, GraphModelTag, NodeId, PropertyKey,
    RecoveryImageComponent, RecoveryImageCoordinatesV1, Value, WorldCut, WorldMetadataSectionV2,
};
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection};
use grafeo_engine::GrafeoDB;
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};
use std::sync::Arc;

fn epochs(baseline: EpochId) -> TestResult<(EpochId, EpochId, EpochId)> {
    let birth = baseline
        .as_u64()
        .checked_sub(5)
        .ok_or("fixture epoch too early")?;
    Ok((
        EpochId::new(birth),
        EpochId::new(birth + 1),
        EpochId::new(birth + 3),
    ))
}

pub(super) fn install(db: &GrafeoDB) -> TestResult {
    let auxiliary_before = super::auxiliary(db)?;
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("same-id.grafeo");
    db.save(&path)?;
    let file = GrafeoFileManager::open(&path)?;
    let header = file.active_header();
    let directory = file
        .read_section_directory()?
        .ok_or("section directory missing")?;
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
        .collect::<TestResult<Vec<_>>>()?;
    let root = Arc::new(LpgStore::new()?);
    let mut lpg = LpgStoreSection::new(Arc::clone(&root));
    let lpg_position = sections
        .iter()
        .position(|entry| entry.0 == SectionType::LpgStore)
        .ok_or("LPG section missing")?;
    lpg.deserialize(&sections[lpg_position].2)?;
    let (birth, death, rebirth) = epochs(db.current_epoch())?;
    for path in data_paths()? {
        let mut store = Arc::clone(&root);
        for name in path.components() {
            store = store.graph(name).ok_or("restored graph missing")?;
        }
        let lives = [(birth, Some(death)), (rebirth, None)];
        store.restore_node_history_exact(
            NodeId::new(100),
            &lives,
            &[
                (birth, vec!["Draft".into(), "First".into()]),
                (birth, vec!["First".into()]),
                (death, vec!["ClosedBoundary".into()]),
                (rebirth, vec!["Second".into()]),
            ],
        )?;
        store.restore_node_history_exact(
            NodeId::new(101),
            &[(birth, None)],
            &[(birth, vec!["Anchor".into()])],
        )?;
        store.restore_edge_history_exact(
            EdgeId::new(100),
            NodeId::new(100),
            NodeId::new(101),
            "LIFETIME",
            &lives,
        )?;
        for (epoch, value) in [
            (birth, Value::from("first")),
            (death, Value::Null),
            (rebirth, Value::from("draft-second")),
            (rebirth, Value::from("second")),
        ] {
            store.set_node_property_at_epoch(NodeId::new(100), "life", value.clone(), epoch);
            store.set_edge_property_at_epoch(EdgeId::new(100), "life", value, epoch);
        }
        assert_eq!(store.peek_next_node_id(), 102);
        assert_eq!(store.peek_next_edge_id(), 101);
    }
    sections[lpg_position].2 = lpg.serialize()?;

    // Use the public current seal constructors: no fabricated digest or legacy
    // metadata exception. All non-LPG payload bytes and versions stay exact.
    let metadata_position = sections
        .iter()
        .position(|entry| entry.0 == SectionType::WorldMetadata)
        .ok_or("WorldMetadata missing")?;
    let metadata = WorldMetadataSectionV2::decode(&sections[metadata_position].2)?;
    let descriptor = metadata.cut().descriptor().clone();
    let mut logical = Vec::new();
    for format in descriptor.formats() {
        let section_type = match format.format() {
            AuthoritativeFormat::Lpg => SectionType::LpgStore,
            AuthoritativeFormat::Catalog => SectionType::Catalog,
            AuthoritativeFormat::Cdc => SectionType::Cdc,
            AuthoritativeFormat::Rdf | AuthoritativeFormat::RdfHistory => SectionType::RdfStore,
            _ => return Err("unexpected format in native fixture".into()),
        };
        let section = sections
            .iter()
            .find(|entry| entry.0 == section_type)
            .ok_or("authoritative section missing")?;
        let bytes = if format.format() == AuthoritativeFormat::RdfHistory {
            let (version, bytes) =
                grafeo_core::graph::rdf::section::canonical_history_component(&section.2)?;
            assert_eq!(format.version(), version);
            bytes
        } else {
            assert_eq!(format.version(), u16::from(section.1));
            section.2.as_slice()
        };
        logical.push((*format, bytes));
    }
    let cut = WorldCut::seal_components(descriptor.clone(), &logical)?;
    let node_count = u64::try_from(root.node_count())?;
    let edge_count = u64::try_from(root.edge_count())?;
    let coordinates = RecoveryImageCoordinatesV1::new(
        header.epoch,
        header.transaction_id,
        GraphModelTag::Both,
        node_count,
        edge_count,
    );
    let mut physical = sections
        .iter()
        .filter(|entry| entry.0 != SectionType::WorldMetadata)
        .map(|entry| RecoveryImageComponent::new(entry.0 as u32, u16::from(entry.1), &entry.2))
        .collect::<Result<Vec<_>, _>>()?;
    physical.push(coordinates.component());
    let metadata = WorldMetadataSectionV2::seal(cut, &physical)?.encode()?;
    sections[metadata_position].2 = metadata;
    let writes: Vec<_> = sections
        .iter()
        .map(|entry| SectionWrite::new(entry.0, entry.1, &entry.2))
        .collect();
    file.write_versioned_sections(
        &writes,
        header.epoch,
        header.transaction_id,
        node_count,
        edge_count,
    )?;
    file.close()?;
    drop(file);
    let restored = persistent(&path)?;
    assert_eq!(
        LpgStoreSection::new(Arc::clone(
            grafeo_engine::database::testing::root_lpg_store(&restored)
        ))
        .serialize()?,
        sections[lpg_position].2,
        "initial current-format restore must preserve every exact history entry"
    );
    let snapshot = restored.export_snapshot()?;
    db.restore_snapshot(&snapshot)?;
    assert_eq!(db.export_snapshot()?, snapshot);
    assert_eq!(super::auxiliary(db)?, auxiliary_before);
    restored.close()?;
    Ok(())
}

pub(super) fn mutate_for_recompact(db: &GrafeoDB) -> TestResult {
    let mut session = db.session();
    session.begin_transaction()?;
    session.set_node_property(NodeId::new(100), "recompact_probe", Value::Int64(7))?;
    session.set_edge_property(EdgeId::new(100), "recompact_probe", Value::Int64(7))?;
    session.commit()?;
    let graph = db.graph_store();
    assert_eq!(
        graph
            .get_node(NodeId::new(100))
            .ok_or("revived node lost")?
            .get_property("recompact_probe"),
        Some(&Value::Int64(7))
    );
    assert_eq!(
        graph
            .get_edge(EdgeId::new(100))
            .ok_or("revived edge lost")?
            .properties
            .get(&PropertyKey::new("recompact_probe")),
        Some(&Value::Int64(7))
    );
    Ok(())
}

pub(super) fn assert_lives(db: &GrafeoDB, baseline: EpochId, has_tail: bool) -> TestResult {
    let (birth, death, rebirth) = epochs(baseline)?;
    for path in data_paths()? {
        let graph = view(db, &path)?;
        for epoch in [
            EpochId::new(birth.as_u64() - 1),
            death,
            EpochId::new(death.as_u64() + 1),
        ] {
            assert!(
                graph.get_node_at_epoch(NodeId::new(100), epoch).is_none(),
                "node gap at {path:?}/{epoch}"
            );
            assert!(
                graph.get_edge_at_epoch(EdgeId::new(100), epoch).is_none(),
                "edge gap at {path:?}/{epoch}"
            );
        }
        for (epoch, label, value) in [
            (birth, "First", "first"),
            (rebirth, "Second", "second"),
            (baseline, "Second", "second"),
        ] {
            let node = graph
                .get_node_at_epoch(NodeId::new(100), epoch)
                .ok_or("same-ID life missing")?;
            assert!(node.has_label(label));
            assert!(!node.has_label("Draft") && !node.has_label("ClosedBoundary"));
            assert_eq!(node.get_property("life"), Some(&Value::from(value)));
            let edge = graph
                .get_edge_at_epoch(EdgeId::new(100), epoch)
                .ok_or("same-ID edge life missing")?;
            assert_eq!((edge.src, edge.dst), (NodeId::new(100), NodeId::new(101)));
            assert_eq!(
                edge.properties.get(&PropertyKey::new("life")),
                Some(&Value::from(value))
            );
        }
        assert!(
            graph
                .get_node(NodeId::new(100))
                .ok_or("current second life missing")?
                .has_label("Second")
        );
        let value = Value::from(if has_tail { "tail-second" } else { "second" });
        assert_eq!(
            graph
                .get_node(NodeId::new(100))
                .ok_or("current second life missing")?
                .get_property("life"),
            Some(&value)
        );
        assert_eq!(
            graph
                .get_edge(EdgeId::new(100))
                .ok_or("current edge life missing")?
                .properties
                .get(&PropertyKey::new("life")),
            Some(&value)
        );
    }
    Ok(())
}
