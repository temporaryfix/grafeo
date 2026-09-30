//! Hostile whole-image recovery-seal qualification.
//!
//! Per-section CRCs prove only that bytes were written intact. These tests
//! splice a valid recovery section from a different logical world into an
//! otherwise untouched container, then publish a fresh directory so every
//! storage-layer checksum is valid. `WORLD_METADATA` v2 must bind the complete
//! recovery image and reject the mix before any section decoder can publish it.

#![cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    any(
        all(feature = "lpg", any(feature = "vector-index", feature = "text-index")),
        all(feature = "triple-store", feature = "ring-index")
    )
))]

use std::path::Path;

use grafeo_common::storage::SectionType;
use grafeo_common::utils::error::{Error, StorageError};
use grafeo_engine::GrafeoDB;
use grafeo_storage::file::{GrafeoFileManager, SectionWrite};

#[derive(Clone, Debug)]
struct RawSection {
    section_type: SectionType,
    version: u8,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct RawImage {
    epoch: u64,
    transaction_id: u64,
    node_count: u64,
    edge_count: u64,
    sections: Vec<RawSection>,
}

impl RawImage {
    fn section(&self, section_type: SectionType) -> &RawSection {
        self.sections
            .iter()
            .find(|section| section.section_type == section_type)
            .unwrap_or_else(|| panic!("missing {section_type:?} section"))
    }

    fn section_mut(&mut self, section_type: SectionType) -> &mut RawSection {
        self.sections
            .iter_mut()
            .find(|section| section.section_type == section_type)
            .unwrap_or_else(|| panic!("missing {section_type:?} section"))
    }
}

fn read_image(path: &Path) -> RawImage {
    let manager = GrafeoFileManager::open_read_only(path).expect("open raw container image");
    let header = manager.active_header();
    let directory = manager
        .read_section_directory()
        .expect("read section directory")
        .expect("active section directory");
    let sections = directory
        .entries()
        .iter()
        .map(|entry| RawSection {
            section_type: entry.section_type,
            version: entry.version,
            bytes: manager
                .read_section_data(entry)
                .expect("read CRC-valid section bytes"),
        })
        .collect();
    manager.close().expect("close raw container image");

    RawImage {
        epoch: header.epoch,
        transaction_id: header.transaction_id,
        node_count: header.node_count,
        edge_count: header.edge_count,
        sections,
    }
}

fn publish_image(path: &Path, image: &RawImage) {
    let manager = GrafeoFileManager::open(path).expect("open splice target");
    let sections: Vec<_> = image
        .sections
        .iter()
        .map(|section| SectionWrite::new(section.section_type, section.version, &section.bytes))
        .collect();
    manager
        .write_versioned_sections(
            &sections,
            image.epoch,
            image.transaction_id,
            image.node_count,
            image.edge_count,
        )
        .expect("publish foreign section with fresh directory checksums");
    manager.close().expect("close splice target");
}

fn splice_foreign_section(
    world_a: &Path,
    world_b: &Path,
    target: &Path,
    section_type: SectionType,
    expected_version: u8,
) {
    std::fs::copy(world_a, target).expect("copy world A to hostile target");

    let mut image_a = read_image(world_a);
    let image_b = read_image(world_b);
    let metadata_a = image_a.section(SectionType::WorldMetadata).clone();
    assert_eq!(
        metadata_a.version, 2,
        "the hostile splice qualifies the WorldMetadata v2 recovery-image seal"
    );

    let foreign = image_b.section(section_type);
    assert_eq!(foreign.version, expected_version);
    let displaced = image_a.section_mut(section_type);
    assert_eq!(displaced.version, expected_version);
    assert_ne!(
        displaced.bytes, foreign.bytes,
        "the two valid worlds must have distinct {section_type:?} recovery images"
    );
    displaced.bytes.clone_from(&foreign.bytes);

    // This rewrites the section directory and every per-section CRC. The only
    // seal deliberately left stale is world A's WorldMetadata v2 payload.
    publish_image(target, &image_a);
    let spliced = read_image(target);
    assert_eq!(
        spliced.section(SectionType::WorldMetadata).bytes,
        metadata_a.bytes,
        "the hostile writer must not reseal logical world metadata"
    );
    assert_eq!(
        spliced.section(section_type).bytes,
        foreign.bytes,
        "the active, CRC-valid directory must point at the foreign section"
    );
}

fn assert_open_rejects_without_rewriting(path: &Path) {
    let before = std::fs::read(path).expect("read hostile container before open");
    let error = match GrafeoDB::open(path) {
        Ok(db) => {
            let _ = db.close();
            panic!("mixed-world recovery image unexpectedly opened")
        }
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::Storage(StorageError::Corruption(_))),
        "expected structured container corruption, got {error:?}"
    );
    assert!(
        error.to_string().contains("recovery image digest mismatch"),
        "the V2 recovery-image seal, not a model decoder, must reject the splice: {error}"
    );
    assert_eq!(
        std::fs::read(path).expect("read hostile container after rejected open"),
        before,
        "failed recovery must not rewrite the rejected container"
    );
}

#[cfg(all(feature = "lpg", any(feature = "vector-index", feature = "text-index")))]
mod lpg_indexes {
    use grafeo_common::types::{NodeId, Value};
    use grafeo_engine::{Config, GraphModel};

    use super::*;

    const NAMED_GRAPH: &str = "archive";

    #[derive(Clone, Copy)]
    enum Fixture {
        A,
        B,
    }

    struct Expectations {
        first_default_node: NodeId,
        #[cfg(feature = "text-index")]
        text_needle: &'static str,
        #[cfg(feature = "vector-index")]
        vector_needle: [f32; 3],
    }

    fn node_properties(fixture: Fixture, first: bool, named: bool) -> Vec<(&'static str, Value)> {
        let mut properties = Vec::new();

        #[cfg(feature = "vector-index")]
        {
            let vector = match (fixture, first, named) {
                (Fixture::A, true, false) => [1.0, 0.0, 0.0],
                (Fixture::A, false, false) => [0.0, 1.0, 0.0],
                (Fixture::A, true, true) => [0.0, 0.0, 1.0],
                (Fixture::A, false, true) => [-1.0, 0.0, 0.0],
                (Fixture::B, true, false) => [0.0, 0.0, -1.0],
                (Fixture::B, false, false) => [0.0, -1.0, 0.0],
                (Fixture::B, true, true) => [1.0, 1.0, 0.0],
                (Fixture::B, false, true) => [-1.0, -1.0, 0.0],
            };
            properties.push(("embedding", Value::Vector(vector.to_vec().into())));
        }

        #[cfg(feature = "text-index")]
        {
            let text = match (fixture, first, named) {
                (Fixture::A, true, false) => "amber alpha",
                (Fixture::A, false, false) => "azure atlas",
                (Fixture::A, true, true) => "apricot archive",
                (Fixture::A, false, true) => "agate annex",
                (Fixture::B, true, false) => "bronze bravo",
                (Fixture::B, false, false) => "beige basil",
                (Fixture::B, true, true) => "brick branch",
                (Fixture::B, false, true) => "birch bay",
            };
            properties.push(("body", Value::from(text)));
        }

        properties
    }

    fn create_world(path: &Path, fixture: Fixture) -> Expectations {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg))
            .expect("create LPG fixture world");

        let first_default_node =
            db.create_node_with_props(&["Doc"], node_properties(fixture, true, false));
        db.create_node_with_props(&["Doc"], node_properties(fixture, false, false));

        #[cfg(feature = "vector-index")]
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create default vector descriptor");
        #[cfg(feature = "text-index")]
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: Default::default(),
            name: None,
            label: Some("Doc".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create default BM25 descriptor");

        assert!(
            grafeo_engine::database::testing::root_lpg_store(&db)
                .create_graph(NAMED_GRAPH)
                .expect("create named graph fixture")
        );
        db.set_current_graph(Some(NAMED_GRAPH))
            .expect("select named graph fixture");
        db.create_node_with_props(&["Doc"], node_properties(fixture, true, true));
        db.create_node_with_props(&["Doc"], node_properties(fixture, false, true));

        #[cfg(feature = "vector-index")]
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: grafeo_common::types::GraphPath::from_components(&[NAMED_GRAPH])
                .expect("exact named graph path"),
            name: None,
            label: Some("Doc".into()),
            property: "embedding".into(),
            kind: grafeo_engine::IndexCreateKind::Vector {
                dimensions: Some(3),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(64),
                ef: None,
                quantization: None,
            },
        })
        .expect("create named vector descriptor");
        #[cfg(feature = "text-index")]
        db.create_index(grafeo_engine::CreateIndexRequest {
            graph: grafeo_common::types::GraphPath::from_components(&[NAMED_GRAPH])
                .expect("exact named graph path"),
            name: None,
            label: Some("Doc".into()),
            property: "body".into(),
            kind: grafeo_engine::IndexCreateKind::Text {
                min_token_length: None,
            },
        })
        .expect("create named BM25 descriptor");

        db.set_current_graph(None)
            .expect("restore default graph before capture");
        db.save(path).expect("save exact scoped-index fixture");
        db.close().expect("close in-memory fixture");

        Expectations {
            first_default_node,
            #[cfg(feature = "text-index")]
            text_needle: match fixture {
                Fixture::A => "alpha",
                Fixture::B => "bravo",
            },
            #[cfg(feature = "vector-index")]
            vector_needle: match fixture {
                Fixture::A => [1.0, 0.0, 0.0],
                Fixture::B => [0.0, 0.0, -1.0],
            },
        }
    }

    fn assert_source_is_unchanged_and_usable(
        path: &Path,
        exact_file_bytes: &[u8],
        expected: &Expectations,
    ) {
        assert_eq!(
            std::fs::read(path).expect("read untouched source world"),
            exact_file_bytes,
            "hostile target construction and rejection must not modify either source world"
        );

        let db = GrafeoDB::open_read_only(path).expect("reopen untouched source world");
        assert_eq!(db.node_count(), 2);
        assert_eq!(
            grafeo_engine::database::testing::root_lpg_store(&db)
                .graph(NAMED_GRAPH)
                .expect("restore named graph")
                .node_count(),
            2
        );

        #[cfg(feature = "vector-index")]
        assert_eq!(
            db.vector_search("Doc", "embedding", &expected.vector_needle, 1, None, None,)
                .expect("query intact vector image")[0]
                .0,
            expected.first_default_node
        );
        #[cfg(feature = "text-index")]
        assert_eq!(
            db.text_search("Doc", "body", expected.text_needle, 10)
                .expect("query intact BM25 image")[0]
                .0,
            expected.first_default_node
        );
    }

    fn fixtures(
        dir: &tempfile::TempDir,
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        Expectations,
        Expectations,
    ) {
        let world_a = dir.path().join("world-a.grafeo");
        let world_b = dir.path().join("world-b.grafeo");
        let expected_a = create_world(&world_a, Fixture::A);
        let expected_b = create_world(&world_b, Fixture::B);
        (world_a, world_b, expected_a, expected_b)
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn world_metadata_v2_rejects_crc_valid_foreign_vector_v4_before_publication() {
        let dir = tempfile::tempdir().unwrap();
        let (world_a, world_b, expected_a, expected_b) = fixtures(&dir);
        let pristine_a = std::fs::read(&world_a).unwrap();
        let pristine_b = std::fs::read(&world_b).unwrap();
        let target = dir.path().join("foreign-vector.grafeo");

        // Vector4 is a bounded exact-state envelope, not the old topology DTO.
        // Splice a complete current payload with the same physical key set so
        // refusal must come from the world seal, not malformed wire or names.
        let keys = |path: &Path| {
            grafeo_core::index::vector::VectorStoreSection::payload_keys(
                &read_image(path).section(SectionType::VectorStore).bytes,
            )
            .expect("fixture has a valid current Vector4 envelope and payload")
        };
        let keys_a = keys(&world_a);
        assert_eq!(
            keys_a.len(),
            2,
            "default and named vector images are present"
        );
        assert_eq!(keys_a, keys(&world_b));
        splice_foreign_section(&world_a, &world_b, &target, SectionType::VectorStore, 4);
        assert_open_rejects_without_rewriting(&target);
        assert_source_is_unchanged_and_usable(&world_a, &pristine_a, &expected_a);
        assert_source_is_unchanged_and_usable(&world_b, &pristine_b, &expected_b);
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn world_metadata_v2_rejects_crc_valid_foreign_text_v5_before_publication() {
        let dir = tempfile::tempdir().unwrap();
        let (world_a, world_b, expected_a, expected_b) = fixtures(&dir);
        let pristine_a = std::fs::read(&world_a).unwrap();
        let pristine_b = std::fs::read(&world_b).unwrap();
        let target = dir.path().join("foreign-text.grafeo");

        // Text5 includes exact posting histories and the retained epoch floor.
        // Preserve that complete grammar while making only its world binding stale.
        let keys = |path: &Path| {
            grafeo_core::index::text::TextIndexSection::payload_keys(
                &read_image(path).section(SectionType::TextIndex).bytes,
            )
            .expect("fixture has a valid current Text5 payload")
        };
        let keys_a = keys(&world_a);
        assert_eq!(keys_a.len(), 2, "default and named text images are present");
        assert_eq!(keys_a, keys(&world_b));
        splice_foreign_section(&world_a, &world_b, &target, SectionType::TextIndex, 5);
        assert_open_rejects_without_rewriting(&target);
        assert_source_is_unchanged_and_usable(&world_a, &pristine_a, &expected_a);
        assert_source_is_unchanged_and_usable(&world_b, &pristine_b, &expected_b);
    }
}

#[cfg(all(feature = "triple-store", feature = "ring-index"))]
mod rdf_ring {
    use grafeo_core::graph::rdf::{Term, Triple};
    use grafeo_engine::{Config, GraphModel};

    use super::*;

    #[derive(Clone, Copy)]
    enum Fixture {
        A,
        B,
    }

    fn triples(fixture: Fixture) -> Vec<Triple> {
        let prefix = match fixture {
            Fixture::A => "a",
            Fixture::B => "b",
        };
        vec![
            Triple::new(
                Term::iri(format!("http://ex.org/{prefix}0")),
                Term::iri("http://ex.org/p"),
                Term::literal(format!("{prefix}a")),
            ),
            Triple::new(
                Term::iri(format!("http://ex.org/{prefix}1")),
                Term::iri("http://ex.org/p"),
                Term::literal(format!("{prefix}b")),
            ),
        ]
    }

    fn create_world(path: &Path, fixture: Fixture) -> Triple {
        let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf))
            .expect("create RDF fixture world");
        let contents = triples(fixture);
        let expected = contents[0].clone();
        db.rdf_store().bulk_load(contents);
        db.rdf_store().rebuild_ring();
        assert!(db.rdf_store().ring().is_some());
        db.save(path).expect("save exact Ring fixture");
        db.close().expect("close in-memory RDF fixture");
        expected
    }

    fn assert_source_is_unchanged_and_usable(path: &Path, exact: &[u8], expected: &Triple) {
        assert_eq!(
            std::fs::read(path).expect("read untouched RDF source world"),
            exact,
            "hostile target construction and rejection must not modify either RDF source world"
        );
        let db = GrafeoDB::open_read_only(path).expect("reopen untouched RDF source world");
        assert!(db.rdf_store().contains(expected));
        assert!(db.rdf_store().ring().is_some());
    }

    #[test]
    fn world_metadata_v2_rejects_crc_valid_foreign_rdf_ring_v2_before_publication() {
        let dir = tempfile::tempdir().unwrap();
        let world_a = dir.path().join("rdf-world-a.grafeo");
        let world_b = dir.path().join("rdf-world-b.grafeo");
        let expected_a = create_world(&world_a, Fixture::A);
        let expected_b = create_world(&world_b, Fixture::B);
        let pristine_a = std::fs::read(&world_a).unwrap();
        let pristine_b = std::fs::read(&world_b).unwrap();
        let target = dir.path().join("foreign-ring.grafeo");

        splice_foreign_section(&world_a, &world_b, &target, SectionType::RdfRing, 2);
        assert_open_rejects_without_rewriting(&target);
        assert_source_is_unchanged_and_usable(&world_a, &pristine_a, &expected_a);
        assert_source_is_unchanged_and_usable(&world_b, &pristine_b, &expected_b);
    }
}
