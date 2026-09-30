//! Portable snapshot artifact and explicit logical-fork qualification.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "triple-store",
    feature = "sparql",
    feature = "grafeo-file"
))]

use grafeo_common::types::Value;
use grafeo_common::types::{
    AuthoritativeFormat, Digest256, EpochId, SchemaCut, SnapshotArtifact, StoreId,
    WorldCutDescriptor,
};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PERSON: &str = "http://ex.org/Person";

fn both_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap()
}

fn assert_format(artifact: &SnapshotArtifact, expected: AuthoritativeFormat) {
    assert!(
        artifact
            .cut()
            .descriptor()
            .formats()
            .iter()
            .any(|format| format.format() == expected),
        "snapshot cut omitted {expected:?}"
    );
}

fn format_version(cut: &grafeo_common::types::WorldCut, expected: AuthoritativeFormat) -> u16 {
    cut.descriptor()
        .formats()
        .iter()
        .find(|format| format.format() == expected)
        .unwrap_or_else(|| panic!("world cut omitted {expected:?}"))
        .version()
}

#[test]
fn artifact_is_deterministic_integrity_sealed_and_restore_preserves_identity() {
    let source = both_db();
    source.session().execute("INSERT (:World {id: 1})").unwrap();
    source
        .execute_sparql(r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/name> "Alix"@en . }"#)
        .unwrap();

    let artifact = source.export_snapshot_artifact().unwrap();
    artifact.verify().unwrap();
    assert_eq!(artifact.cut().store_id(), source.store_id());
    assert_format(&artifact, AuthoritativeFormat::Catalog);
    assert_format(&artifact, AuthoritativeFormat::Lpg);
    assert_format(&artifact, AuthoritativeFormat::Rdf);
    assert_format(&artifact, AuthoritativeFormat::RdfHistory);
    assert_format(&artifact, AuthoritativeFormat::PortableSnapshot);
    let installed = source.world_cut().unwrap();
    installed.verify().unwrap();
    assert_eq!(installed.store_id(), artifact.cut().store_id());
    assert_eq!(installed.epoch(), artifact.cut().epoch());
    assert_eq!(
        installed.descriptor().graph_model(),
        artifact.cut().descriptor().graph_model()
    );
    assert_eq!(
        installed.descriptor().schema(),
        artifact.cut().descriptor().schema()
    );
    assert_eq!(
        installed.descriptor().projections(),
        artifact.cut().descriptor().projections()
    );
    assert_eq!(
        format_version(&installed, AuthoritativeFormat::Rdf),
        6,
        "the container cut must name the outer RDF section grammar"
    );
    assert_eq!(
        format_version(&installed, AuthoritativeFormat::RdfHistory),
        1,
        "the independently named container history component must report its nested grammar"
    );
    assert_eq!(
        format_version(artifact.cut(), AuthoritativeFormat::RdfHistory),
        format_version(&installed, AuthoritativeFormat::RdfHistory),
        "portable and container cuts must agree on the canonical RDF-history grammar"
    );

    let restored = GrafeoDB::import_snapshot_artifact(&artifact).unwrap();
    assert_eq!(restored.store_id(), source.store_id());
    assert_eq!(restored.node_count(), 1);
    assert_eq!(
        restored
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        restored
            .export_snapshot_artifact()
            .unwrap()
            .cut()
            .manifest_digest(),
        artifact.cut().manifest_digest(),
        "an unchanged restore must reproduce the same portable cut"
    );
}

#[test]
fn import_rejects_an_artifact_with_a_self_consistent_false_descriptor() {
    let source = both_db();
    source.session().execute("INSERT (:World {id: 1})").unwrap();
    let artifact = source.export_snapshot_artifact().unwrap();
    let authentic = artifact.cut().descriptor();
    let foreign_store = StoreId::generate().unwrap();
    assert_ne!(foreign_store, authentic.store_id());
    let forged_descriptor = WorldCutDescriptor::new(
        foreign_store,
        authentic.epoch(),
        authentic.graph_model(),
        authentic.formats().to_vec(),
        authentic.schema().clone(),
        authentic.projections().to_vec(),
        authentic.history(),
    )
    .unwrap();
    let forged = SnapshotArtifact::new(artifact.bytes().to_vec(), forged_descriptor).unwrap();
    forged.verify().unwrap();
    assert!(GrafeoDB::import_snapshot_artifact(&forged).is_err());

    let false_schema = SchemaCut::new(
        authentic.schema().format_version(),
        Digest256::schema(b"not the snapshot catalog"),
    )
    .unwrap();
    let forged_descriptor = WorldCutDescriptor::new(
        authentic.store_id(),
        authentic.epoch(),
        authentic.graph_model(),
        authentic.formats().to_vec(),
        false_schema,
        authentic.projections().to_vec(),
        authentic.history(),
    )
    .unwrap();
    let forged = SnapshotArtifact::new(artifact.bytes().to_vec(), forged_descriptor).unwrap();
    forged.verify().unwrap();
    assert!(GrafeoDB::import_snapshot_artifact(&forged).is_err());
}

#[test]
fn explicit_snapshot_fork_rekeys_history_and_resets_projection_provenance() {
    let source = both_db();
    source
        .session()
        .execute("INSERT (:Ordinary {name: 'kept'})")
        .unwrap();
    source
        .execute_sparql(&format!(
            r#"INSERT DATA {{ <http://ex.org/alix> a <{PERSON}> . }}"#
        ))
        .unwrap();
    let projection_id = source.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    assert_eq!(source.rebuild_rdf_lpg_projection(projection_id).unwrap(), 1);
    let source_history = source.rdf_dataset_history().unwrap();
    let source_version = source_history.quad_versions().first().unwrap();
    let source_cursor = source
        .rdf_cdc_page(EpochId::INITIAL, source.current_epoch(), None, 1)
        .unwrap()
        .next_cursor
        .expect("one asserted statement yields a durable cursor");
    let artifact = source.export_snapshot_artifact().unwrap();
    assert_eq!(artifact.cut().descriptor().projections().len(), 1);

    let fork = GrafeoDB::import_snapshot_as_fork(&source.export_snapshot().unwrap()).unwrap();

    assert_ne!(fork.store_id(), source.store_id());
    let fork_history = fork.rdf_dataset_history().unwrap();
    let fork_version = fork_history.quad_versions().first().unwrap();
    assert_eq!(fork_version.quad(), source_version.quad());
    assert_eq!(fork_version.tx(), source_version.tx());
    assert_eq!(fork_version.valid(), source_version.valid());
    assert_ne!(fork_version.statement(), source_version.statement());
    assert!(
        fork.rdf_cdc_page(
            EpochId::INITIAL,
            fork.current_epoch(),
            Some(&source_cursor),
            1,
        )
        .is_err(),
        "source CDC cursors must fail closed in the fork namespace"
    );

    let pending = fork
        .rdf_lpg_projection(projection_id)
        .expect("logical mapping survives the fork");
    assert_eq!(pending.generation(), 0);
    assert_eq!(pending.receipt(), None);
    assert_eq!(pending.row_count(), 0);
    assert_eq!(fork.node_count(), 1, "foreign projection rows are removed");
    assert_eq!(
        fork.session()
            .execute("MATCH (n:Ordinary) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(1)
    );
    assert_eq!(fork.rebuild_rdf_lpg_projection(projection_id).unwrap(), 1);
    assert_eq!(
        fork.rdf_lpg_projection(projection_id)
            .unwrap()
            .receipt()
            .unwrap()
            .store_id(),
        fork.store_id()
    );
}

#[test]
fn open_multi_forks_one_both_owner_and_resets_projection_provenance() {
    let source = both_db();
    source
        .session()
        .execute("INSERT (:Ordinary {name: 'kept'})")
        .unwrap();
    source
        .execute_sparql(&format!(
            r#"INSERT DATA {{ <http://ex.org/alix> a <{PERSON}> . }}"#
        ))
        .unwrap();
    let projection_id = source.declare_rdf_lpg_projection(PERSON, "Person").unwrap();
    assert_eq!(source.rebuild_rdf_lpg_projection(projection_id).unwrap(), 1);

    let source_history = source.rdf_dataset_history().unwrap();
    let source_version = source_history.quad_versions().first().unwrap();
    let bytes = source.export_snapshot().unwrap();

    let merged = GrafeoDB::open_multi([bytes.as_slice()]).unwrap();

    assert_eq!(merged.graph_model(), GraphModel::Both);
    assert_ne!(merged.store_id(), source.store_id());
    let merged_history = merged.rdf_dataset_history().unwrap();
    assert_eq!(merged_history.store_id(), merged.store_id());
    assert_eq!(merged_history.completeness(), source_history.completeness());
    assert_eq!(
        merged_history.next_graph_incarnation(),
        source_history.next_graph_incarnation()
    );
    assert_eq!(merged_history.graph_lives(), source_history.graph_lives());
    let merged_version = merged_history.quad_versions().first().unwrap();
    assert_eq!(merged_version.quad(), source_version.quad());
    assert_eq!(
        merged_version.graph_incarnation(),
        source_version.graph_incarnation()
    );
    assert_eq!(merged_version.tx(), source_version.tx());
    assert_eq!(merged_version.valid(), source_version.valid());
    assert_ne!(merged_version.statement(), source_version.statement());
    assert_eq!(
        merged
            .rdf_statement_handle(merged_version.quad(), merged_version.graph_incarnation())
            .unwrap(),
        merged_version.statement()
    );

    let pending = merged
        .rdf_lpg_projection(projection_id)
        .expect("logical mapping survives the union fork");
    assert_eq!(pending.generation(), 0);
    assert_eq!(pending.receipt(), None);
    assert_eq!(pending.row_count(), 0);
    assert_eq!(
        merged.node_count(),
        1,
        "projection-owned LPG rows must not cross the fork boundary"
    );
    assert_eq!(
        merged
            .session()
            .execute("MATCH (n:Ordinary) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(1)
    );
}

#[test]
fn open_multi_unions_lpg_and_rdf_into_both_under_fresh_lineage() {
    let lpg = GrafeoDB::new_in_memory();
    lpg.session()
        .execute("INSERT (:Local {name: 'lpg'})")
        .unwrap();

    let rdf = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    rdf.execute_sparql(r#"INSERT DATA { <http://ex.org/rdf> <http://ex.org/name> "RDF"@en . }"#)
        .unwrap();
    let rdf_store_id = rdf.store_id();

    let lpg_bytes = lpg.export_snapshot().unwrap();
    let rdf_bytes = rdf.export_snapshot().unwrap();
    let merged = GrafeoDB::open_multi([lpg_bytes.as_slice(), rdf_bytes.as_slice()]).unwrap();

    assert_eq!(merged.graph_model(), GraphModel::Both);
    assert_ne!(merged.store_id(), lpg.store_id());
    assert_ne!(merged.store_id(), rdf_store_id);
    assert_eq!(merged.node_count(), 1);
    assert_eq!(
        merged
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        merged.rdf_dataset_history().unwrap().store_id(),
        merged.store_id()
    );
}

#[test]
fn open_multi_rejects_two_declared_rdf_owners_even_when_empty() {
    let first =
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let second =
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Both)).unwrap();
    let first = first.export_snapshot().unwrap();
    let second = second.export_snapshot().unwrap();

    let Err(error) = GrafeoDB::open_multi([first.as_slice(), second.as_slice()]) else {
        panic!("two authoritative RDF owners unexpectedly merged");
    };
    assert!(
        error.to_string().contains("at most one"),
        "unexpected diagnostic: {error}"
    );
}
