//! Hostile behavioral tests for portable world-cut manifests.

use grafeo_common::types::{
    AuthoritativeFormat, Digest256, EpochId, GraphIncarnationId, GraphModelTag,
    HistoryCompleteness, ModelFormatVersion, ProjectionCut, ProjectionReconciliationState,
    ProjectionSourceGraph, RecoveryImageComponent, RecoveryImageDigest, SchemaCut,
    SnapshotArtifact, StateDigest, StateDigestKind, StoreId, WorldCut, WorldCutDescriptor,
    WorldCutError, WorldIdentityMetadataV1, WorldMetadataSectionV1, WorldMetadataSectionV2,
};

fn store(byte: u8) -> StoreId {
    StoreId::from_bytes([byte; StoreId::LEN]).expect("test store id is non-zero")
}

fn formats() -> Vec<ModelFormatVersion> {
    vec![
        ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).expect("non-zero format version"),
        ModelFormatVersion::new(AuthoritativeFormat::Rdf, 6).expect("non-zero format version"),
        ModelFormatVersion::new(AuthoritativeFormat::RdfHistory, 1)
            .expect("non-zero format version"),
        ModelFormatVersion::new(AuthoritativeFormat::Catalog, 5).expect("non-zero format version"),
        ModelFormatVersion::new(AuthoritativeFormat::PortableSnapshot, 9)
            .expect("non-zero format version"),
    ]
}

fn projection(store_id: StoreId) -> ProjectionCut {
    ProjectionCut::from_verified_receipt(
        store_id,
        Digest256::projection_mapping(b"rdf:type Person -> :Person"),
        2,
        3,
        ProjectionSourceGraph::default_graph(),
        EpochId::new(9),
        Some(EpochId::new(10)),
        42,
        ProjectionReconciliationState::Reconciled,
        Digest256::from_bytes([0x44; Digest256::LEN]),
    )
    .expect("valid projection cut")
}

fn descriptor(store_id: StoreId) -> WorldCutDescriptor {
    WorldCutDescriptor::new(
        store_id,
        EpochId::new(11),
        GraphModelTag::Both,
        formats(),
        SchemaCut::from_canonical_post_image(5, b"canonical schema v5")
            .expect("schema version is non-zero"),
        vec![projection(store_id)],
        HistoryCompleteness::Complete,
    )
    .expect("valid world-cut descriptor")
}

fn logical_lpg_cut() -> WorldCut {
    let catalog = ModelFormatVersion::new(AuthoritativeFormat::Catalog, 6).unwrap();
    let lpg = ModelFormatVersion::new(AuthoritativeFormat::Lpg, 3).unwrap();
    WorldCut::seal_components(
        WorldCutDescriptor::new(
            store(6),
            EpochId::new(17),
            GraphModelTag::Lpg,
            vec![catalog, lpg],
            SchemaCut::from_canonical_post_image(6, b"catalog-v6").unwrap(),
            Vec::new(),
            HistoryCompleteness::Complete,
        )
        .unwrap(),
        &[(catalog, &b"catalog-v6"[..]), (lpg, &b"lpg-v3"[..])],
    )
    .unwrap()
}

fn recovery_image() -> [RecoveryImageComponent<'static>; 4] {
    [
        RecoveryImageComponent::new(1, 6, b"catalog-v6").unwrap(),
        RecoveryImageComponent::new(2, 3, b"lpg-v3").unwrap(),
        RecoveryImageComponent::new(10, 3, b"vector-v3").unwrap(),
        RecoveryImageComponent::new(11, 3, b"text-v3").unwrap(),
    ]
}

#[test]
fn snapshot_artifact_digest_binds_every_data_byte() {
    let first = SnapshotArtifact::new(b"portable-snapshot".to_vec(), descriptor(store(1)))
        .expect("valid artifact");
    let second = SnapshotArtifact::new(b"portable-snapshoU".to_vec(), descriptor(store(1)))
        .expect("valid artifact");

    assert_ne!(
        first.cut().manifest_digest(),
        second.cut().manifest_digest()
    );

    let (bytes, cut) = first.into_parts();
    let mut tampered = bytes.to_vec();
    tampered[0] ^= 0x80;
    let error = SnapshotArtifact::from_parts(tampered, cut)
        .expect_err("a cut cannot authenticate different bytes");
    assert!(
        error.to_string().contains("state digest mismatch"),
        "{error}"
    );
}

#[test]
fn manifest_digest_binds_each_metadata_axis() {
    let base = SnapshotArtifact::new(b"same bytes".to_vec(), descriptor(store(1)))
        .expect("valid artifact");
    let base_digest = base.cut().manifest_digest();

    let cases = [
        WorldCutDescriptor::new(
            store(2),
            EpochId::new(11),
            GraphModelTag::Both,
            formats(),
            SchemaCut::from_canonical_post_image(5, b"canonical schema v5").unwrap(),
            descriptor(store(2)).projections().to_vec(),
            HistoryCompleteness::Complete,
        )
        .unwrap(),
        WorldCutDescriptor::new(
            store(1),
            EpochId::new(12),
            GraphModelTag::Both,
            formats(),
            SchemaCut::from_canonical_post_image(5, b"canonical schema v5").unwrap(),
            descriptor(store(1)).projections().to_vec(),
            HistoryCompleteness::Complete,
        )
        .unwrap(),
        WorldCutDescriptor::new(
            store(1),
            EpochId::new(11),
            GraphModelTag::Both,
            formats(),
            SchemaCut::from_canonical_post_image(5, b"different schema").unwrap(),
            descriptor(store(1)).projections().to_vec(),
            HistoryCompleteness::Complete,
        )
        .unwrap(),
        WorldCutDescriptor::new(
            store(1),
            EpochId::new(11),
            GraphModelTag::Both,
            formats(),
            SchemaCut::from_canonical_post_image(5, b"canonical schema v5").unwrap(),
            descriptor(store(1)).projections().to_vec(),
            HistoryCompleteness::LegacyCurrentState {
                observed_at: EpochId::new(7),
                source_version: 7,
            },
        )
        .unwrap(),
    ];

    for changed in cases {
        let artifact = SnapshotArtifact::new(b"same bytes".to_vec(), changed).unwrap();
        assert_ne!(base_digest, artifact.cut().manifest_digest());
    }
}

#[test]
fn canonical_order_does_not_change_manifest_digest() {
    let forward = descriptor(store(1));
    let mut reversed_formats = formats();
    reversed_formats.reverse();
    let reversed = WorldCutDescriptor::new(
        store(1),
        EpochId::new(11),
        GraphModelTag::Both,
        reversed_formats,
        forward.schema().clone(),
        forward.projections().iter().cloned().rev().collect(),
        HistoryCompleteness::Complete,
    )
    .unwrap();

    let a = SnapshotArtifact::new(b"same".to_vec(), forward).unwrap();
    let b = SnapshotArtifact::new(b"same".to_vec(), reversed).unwrap();
    assert_eq!(a.cut().manifest_digest(), b.cut().manifest_digest());
}

#[test]
fn cross_store_verification_is_rejected() {
    let artifact = SnapshotArtifact::new(b"same".to_vec(), descriptor(store(1))).unwrap();
    artifact.verify_for_store(store(1)).unwrap();

    let error = artifact
        .verify_for_store(store(2))
        .expect_err("foreign store identity must not verify");
    assert!(
        error.to_string().contains("store identity mismatch"),
        "{error}"
    );
}

#[test]
fn invalid_and_duplicate_manifest_entries_fail_closed() {
    let duplicate_formats = vec![
        ModelFormatVersion::new(AuthoritativeFormat::Catalog, 5).unwrap(),
        ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).unwrap(),
        ModelFormatVersion::new(AuthoritativeFormat::Lpg, 8).unwrap(),
    ];
    assert!(
        WorldCutDescriptor::new(
            store(1),
            EpochId::new(1),
            GraphModelTag::Lpg,
            duplicate_formats,
            SchemaCut::from_canonical_post_image(5, b"schema").unwrap(),
            Vec::new(),
            HistoryCompleteness::Complete,
        )
        .is_err()
    );

    let projection = ProjectionCut::from_verified_receipt(
        store(1),
        Digest256::projection_mapping(b"mapping"),
        1,
        1,
        ProjectionSourceGraph::default_graph(),
        EpochId::new(2),
        Some(EpochId::new(1)),
        1,
        ProjectionReconciliationState::Reconciled,
        Digest256::from_bytes([3; Digest256::LEN]),
    );
    assert!(projection.is_err(), "source cannot follow target");
}

#[test]
fn published_projection_receipt_binds_source_rows_and_reconciliation() {
    let mapping = Digest256::projection_mapping(b"mapping");
    let default = ProjectionCut::from_verified_receipt(
        store(1),
        mapping,
        3,
        7,
        ProjectionSourceGraph::default_graph(),
        EpochId::new(5),
        Some(EpochId::new(6)),
        10,
        ProjectionReconciliationState::Reconciled,
        Digest256::from_bytes([0x11; Digest256::LEN]),
    )
    .unwrap();
    let named = ProjectionCut::from_verified_receipt(
        store(1),
        mapping,
        3,
        7,
        ProjectionSourceGraph::named("http://example.org/source", GraphIncarnationId::new(4))
            .unwrap(),
        EpochId::new(5),
        Some(EpochId::new(6)),
        10,
        ProjectionReconciliationState::Reconciled,
        Digest256::from_bytes([0x22; Digest256::LEN]),
    )
    .unwrap();
    let different_rows = ProjectionCut::from_verified_receipt(
        store(1),
        mapping,
        3,
        7,
        ProjectionSourceGraph::default_graph(),
        EpochId::new(5),
        Some(EpochId::new(6)),
        11,
        ProjectionReconciliationState::NeedsReconciliation,
        Digest256::from_bytes([0x33; Digest256::LEN]),
    )
    .unwrap();

    assert_ne!(default.receipt_digest(), named.receipt_digest());
    assert_ne!(default.receipt_digest(), different_rows.receipt_digest());
    assert_eq!(default.row_count(), 10);
    assert_eq!(
        named.source_graph().name(),
        Some("http://example.org/source")
    );
}

#[test]
fn projection_receipt_requires_target_epoch() {
    let args = || {
        (
            store(1),
            Digest256::projection_mapping(b"mapping"),
            3,
            7,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(5),
            None,
            10,
            Digest256::from_bytes([0x55; Digest256::LEN]),
        )
    };
    let (sid, mapping, version, generation, graph, source, target, rows, receipt) = args();
    assert!(
        ProjectionCut::from_verified_receipt(
            sid,
            mapping,
            version,
            generation,
            graph,
            source,
            target,
            rows,
            ProjectionReconciliationState::Reconciled,
            receipt,
        )
        .is_err()
    );
    let (sid, mapping, version, generation, graph, source, target, rows, receipt) = args();
    assert!(
        ProjectionCut::from_verified_receipt(
            sid,
            mapping,
            version,
            generation,
            graph,
            source,
            target,
            rows,
            ProjectionReconciliationState::NeedsReconciliation,
            receipt,
        )
        .is_err()
    );
}

#[test]
fn projection_receipt_namespace_must_match_world_store() {
    let error = WorldCutDescriptor::new(
        store(2),
        EpochId::new(11),
        GraphModelTag::Both,
        formats(),
        SchemaCut::from_canonical_post_image(5, b"schema").unwrap(),
        vec![projection(store(1))],
        HistoryCompleteness::Complete,
    )
    .expect_err("a receipt from another logical store must be rejected");
    assert!(error.to_string().contains("receipt store"), "{error}");
}

#[test]
fn snapshot_artifact_requires_a_portable_snapshot_format_version() {
    let descriptor = WorldCutDescriptor::new(
        store(1),
        EpochId::new(1),
        GraphModelTag::Lpg,
        vec![
            ModelFormatVersion::new(AuthoritativeFormat::Catalog, 5).unwrap(),
            ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).unwrap(),
        ],
        SchemaCut::from_canonical_post_image(5, b"schema").unwrap(),
        Vec::new(),
        HistoryCompleteness::Complete,
    )
    .unwrap();

    let error = SnapshotArtifact::new(b"snapshot".to_vec(), descriptor)
        .expect_err("snapshot bytes require an explicit snapshot wire version");
    assert!(error.to_string().contains("PortableSnapshot"), "{error}");
}

#[test]
fn section_state_digest_is_order_independent_and_version_sensitive() {
    let catalog_v5 = ModelFormatVersion::new(AuthoritativeFormat::Catalog, 5).unwrap();
    let lpg_v7 = ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).unwrap();
    let lpg_v8 = ModelFormatVersion::new(AuthoritativeFormat::Lpg, 8).unwrap();

    let forward = StateDigest::authoritative_components(&[
        (catalog_v5, b"catalog".as_slice()),
        (lpg_v7, b"lpg".as_slice()),
    ])
    .unwrap();
    let reverse = StateDigest::authoritative_components(&[
        (lpg_v7, b"lpg".as_slice()),
        (catalog_v5, b"catalog".as_slice()),
    ])
    .unwrap();
    let changed = StateDigest::authoritative_components(&[
        (catalog_v5, b"catalog".as_slice()),
        (lpg_v8, b"lpg".as_slice()),
    ])
    .unwrap();

    assert_eq!(forward, reverse);
    assert_ne!(forward, changed);
    assert_eq!(forward.kind(), StateDigestKind::AuthoritativeComponents);
}

#[test]
fn world_cut_round_trip_verifies_without_panicking() {
    let artifact = SnapshotArtifact::new(b"snapshot".to_vec(), descriptor(store(3))).unwrap();
    let encoded = artifact.cut().encode().expect("encode cut");
    let decoded = WorldCut::decode(&encoded).expect("decode verified cut");

    assert_eq!(decoded, *artifact.cut());
    decoded.verify().unwrap();

    let mut trailing = encoded;
    trailing.push(0);
    assert!(WorldCut::decode(&trailing).is_err());
}

#[test]
fn generated_store_id_uses_fallible_system_entropy() {
    let first = StoreId::generate().expect("operating-system CSPRNG is available");
    let second = StoreId::generate().expect("operating-system CSPRNG is available");

    assert_ne!(first, second);
    assert!(first.as_bytes().iter().any(|byte| *byte != 0));
    assert!(second.as_bytes().iter().any(|byte| *byte != 0));
}

#[test]
fn world_identity_metadata_round_trips_and_rejects_trailing_bytes() {
    let metadata = WorldIdentityMetadataV1::new(
        store(9),
        HistoryCompleteness::LegacyCurrentState {
            observed_at: EpochId::new(17),
            source_version: 8,
        },
    )
    .unwrap();
    let encoded = metadata.encode().unwrap();
    let decoded = WorldIdentityMetadataV1::decode(&encoded).unwrap();

    assert_eq!(decoded, metadata);
    assert_eq!(decoded.store_id(), store(9));
    assert_eq!(
        decoded.history(),
        HistoryCompleteness::LegacyCurrentState {
            observed_at: EpochId::new(17),
            source_version: 8,
        }
    );

    let mut trailing = encoded;
    trailing.push(0);
    assert!(WorldIdentityMetadataV1::decode(&trailing).is_err());
}

#[test]
fn world_identity_metadata_rejects_an_unversioned_legacy_claim() {
    let invalid = WorldIdentityMetadataV1::new(
        store(9),
        HistoryCompleteness::LegacyCurrentState {
            observed_at: EpochId::new(17),
            source_version: 0,
        },
    );
    assert!(invalid.is_err());

    let pending = WorldIdentityMetadataV1::new(
        store(9),
        HistoryCompleteness::LegacyCurrentState {
            observed_at: EpochId::PENDING,
            source_version: 8,
        },
    );
    assert!(pending.is_err());
}

#[test]
fn world_metadata_section_authenticates_authoritative_container_sections() {
    let catalog = ModelFormatVersion::new(AuthoritativeFormat::Catalog, 5).unwrap();
    let lpg = ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).unwrap();
    let descriptor = WorldCutDescriptor::new(
        store(7),
        EpochId::new(19),
        GraphModelTag::Lpg,
        vec![catalog, lpg],
        SchemaCut::from_canonical_post_image(5, b"catalog post-image").unwrap(),
        Vec::new(),
        HistoryCompleteness::Complete,
    )
    .unwrap();
    let components = [
        (catalog, b"catalog post-image".as_slice()),
        (lpg, b"lpg bytes".as_slice()),
    ];
    let cut = WorldCut::seal(
        descriptor,
        StateDigest::authoritative_components(&components).unwrap(),
    )
    .unwrap();
    let metadata = WorldMetadataSectionV1::new(cut).unwrap();
    let encoded = metadata.encode().unwrap();
    let decoded = WorldMetadataSectionV1::decode(&encoded).unwrap();

    assert_eq!(decoded, metadata);
    decoded.verify_components(&components).unwrap();
    decoded.verify_for_store(store(7), &components).unwrap();

    let changed = [
        (catalog, b"catalog post-image".as_slice()),
        (lpg, b"lpg byteS".as_slice()),
    ];
    assert!(decoded.verify_components(&changed).is_err());
    assert!(decoded.verify_for_store(store(8), &components).is_err());
}

#[test]
fn world_metadata_section_decode_is_exact_and_rejects_snapshot_digest_kind() {
    let artifact = SnapshotArtifact::new(b"snapshot".to_vec(), descriptor(store(3))).unwrap();
    assert!(
        WorldMetadataSectionV1::new(artifact.cut().clone()).is_err(),
        "container metadata cannot substitute a portable-snapshot byte digest"
    );

    let catalog = ModelFormatVersion::new(AuthoritativeFormat::Catalog, 5).unwrap();
    let lpg = ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).unwrap();
    let components = [(catalog, &b"catalog"[..]), (lpg, &b"lpg"[..])];
    let cut = WorldCut::seal(
        WorldCutDescriptor::new(
            store(3),
            EpochId::new(1),
            GraphModelTag::Lpg,
            vec![catalog, lpg],
            SchemaCut::from_canonical_post_image(5, b"catalog").unwrap(),
            Vec::new(),
            HistoryCompleteness::Complete,
        )
        .unwrap(),
        StateDigest::authoritative_components(&components).unwrap(),
    )
    .unwrap();
    let mut encoded = WorldMetadataSectionV1::new(cut).unwrap().encode().unwrap();
    encoded.push(0);
    assert!(WorldMetadataSectionV1::decode(&encoded).is_err());
}

#[test]
fn world_metadata_v2_round_trips_without_changing_the_logical_cut() {
    let cut = logical_lpg_cut();
    let v1 = WorldMetadataSectionV1::new(cut.clone()).unwrap();
    let forward = recovery_image();
    let mut reverse = forward;
    reverse.reverse();

    let metadata = WorldMetadataSectionV2::seal(cut.clone(), &forward).unwrap();
    assert_eq!(metadata.cut(), v1.cut());
    assert_eq!(metadata.cut().manifest_digest(), cut.manifest_digest());
    assert_eq!(
        metadata.recovery_image_digest(),
        RecoveryImageDigest::from_components(&reverse).unwrap(),
        "directory iteration order must not affect the physical-image digest"
    );
    assert_eq!(
        metadata.recovery_image_digest().to_string(),
        "ad0f4ab57131348e112782634ec2bd4cca127e0c30abfd7df300d1d6b1db1656",
        "the published recovery-image v1 digest grammar is frozen"
    );
    metadata.verify_recovery_components(&reverse).unwrap();

    let encoded = metadata.encode().unwrap();
    let decoded = WorldMetadataSectionV2::decode(&encoded).unwrap();
    assert_eq!(decoded, metadata);
    assert_eq!(decoded.cut(), &cut);
    decoded.verify_recovery_components(&forward).unwrap();

    let snapshot = SnapshotArtifact::new(b"snapshot".to_vec(), descriptor(store(9))).unwrap();
    assert!(matches!(
        WorldMetadataSectionV2::seal(snapshot.cut().clone(), &forward),
        Err(WorldCutError::WrongStateDigestKind {
            expected: StateDigestKind::AuthoritativeComponents,
            actual: StateDigestKind::SnapshotBytes,
        })
    ));
}

#[test]
fn world_metadata_v2_rejects_every_recovery_inventory_mismatch_axis() {
    let metadata = WorldMetadataSectionV2::seal(logical_lpg_cut(), &recovery_image()).unwrap();
    let mismatch = |components: &[RecoveryImageComponent<'_>]| {
        assert!(matches!(
            metadata.verify_recovery_components(components),
            Err(WorldCutError::RecoveryImageDigestMismatch { .. })
        ));
    };

    let changed_byte = [
        RecoveryImageComponent::new(1, 6, b"catalog-v6").unwrap(),
        RecoveryImageComponent::new(2, 3, b"lpg-v3").unwrap(),
        RecoveryImageComponent::new(10, 3, b"vector-vX").unwrap(),
        RecoveryImageComponent::new(11, 3, b"text-v3").unwrap(),
    ];
    mismatch(&changed_byte);

    let changed_version = [
        RecoveryImageComponent::new(1, 6, b"catalog-v6").unwrap(),
        RecoveryImageComponent::new(2, 3, b"lpg-v3").unwrap(),
        RecoveryImageComponent::new(10, 4, b"vector-v3").unwrap(),
        RecoveryImageComponent::new(11, 3, b"text-v3").unwrap(),
    ];
    mismatch(&changed_version);

    let changed_type = [
        RecoveryImageComponent::new(1, 6, b"catalog-v6").unwrap(),
        RecoveryImageComponent::new(2, 3, b"lpg-v3").unwrap(),
        RecoveryImageComponent::new(10, 3, b"vector-v3").unwrap(),
        RecoveryImageComponent::new(12, 3, b"text-v3").unwrap(),
    ];
    mismatch(&changed_type);

    let swapped_payloads = [
        RecoveryImageComponent::new(1, 6, b"catalog-v6").unwrap(),
        RecoveryImageComponent::new(2, 3, b"lpg-v3").unwrap(),
        RecoveryImageComponent::new(10, 3, b"text-v3").unwrap(),
        RecoveryImageComponent::new(11, 3, b"vector-v3").unwrap(),
    ];
    mismatch(&swapped_payloads);

    let complete = recovery_image();
    mismatch(&complete[..complete.len() - 1]);

    let mut added = complete.to_vec();
    added.push(RecoveryImageComponent::new(12, 1, b"ring-v1").unwrap());
    mismatch(&added);
}

#[test]
fn recovery_image_components_reject_invalid_and_duplicate_directory_entries() {
    assert!(matches!(
        RecoveryImageComponent::new(0, 1, b"reserved"),
        Err(WorldCutError::InvalidRecoveryImageSectionType { section_type: 0 })
    ));
    assert!(matches!(
        RecoveryImageComponent::new(6, 2, b"recursive metadata"),
        Err(WorldCutError::InvalidRecoveryImageSectionType { section_type: 6 })
    ));
    assert!(matches!(
        RecoveryImageComponent::new(10, 0, b"zero version"),
        Err(WorldCutError::InvalidRecoveryImageSectionVersion { section_type: 10 })
    ));

    let duplicate = [
        RecoveryImageComponent::new(10, 2, b"old vector").unwrap(),
        RecoveryImageComponent::new(10, 3, b"new vector").unwrap(),
    ];
    assert!(matches!(
        RecoveryImageDigest::from_components(&duplicate),
        Err(WorldCutError::DuplicateRecoveryImageSection { section_type: 10 })
    ));

    let excessive = vec![
        RecoveryImageComponent::new(1, 1, b"bounded").unwrap();
        grafeo_common::types::MAX_RECOVERY_IMAGE_COMPONENTS + 1
    ];
    assert!(matches!(
        RecoveryImageDigest::from_components(&excessive),
        Err(WorldCutError::TooManyRecoveryImageComponents { .. })
    ));
}

#[test]
fn world_metadata_v2_decode_is_exact_bounded_and_wire_separated_from_v1() {
    let v2 = WorldMetadataSectionV2::seal(logical_lpg_cut(), &recovery_image()).unwrap();
    let encoded_v2 = v2.encode().unwrap();
    assert!(WorldMetadataSectionV2::decode(&[]).is_err());

    let mut trailing = encoded_v2.clone();
    trailing.push(0);
    assert!(matches!(
        WorldMetadataSectionV2::decode(&trailing),
        Err(WorldCutError::TrailingWorldMetadata { trailing: 1 })
    ));

    let mut malformed_magic = encoded_v2.clone();
    malformed_magic[0] ^= 0x80;
    assert!(WorldMetadataSectionV2::decode(&malformed_magic).is_err());

    let mut unsupported_version = encoded_v2.clone();
    assert_eq!(&unsupported_version[..8], b"WRLDCUT2");
    unsupported_version[8] = 3;
    assert!(WorldMetadataSectionV2::decode(&unsupported_version).is_err());

    let encoded_v1 = WorldMetadataSectionV1::new(logical_lpg_cut())
        .unwrap()
        .encode()
        .unwrap();
    assert!(WorldMetadataSectionV2::decode(&encoded_v1).is_err());
    assert!(WorldMetadataSectionV1::decode(&encoded_v2).is_err());
}

#[test]
fn recovery_image_digest_has_an_independent_domain() {
    let recovery =
        RecoveryImageDigest::from_components(&[
            RecoveryImageComponent::new(1, 1, b"same payload").unwrap()
        ])
        .unwrap();
    let logical = StateDigest::authoritative_components(&[(
        ModelFormatVersion::new(AuthoritativeFormat::Catalog, 1).unwrap(),
        &b"same payload"[..],
    )])
    .unwrap();
    let schema = Digest256::schema(b"same payload");

    assert_ne!(recovery.as_bytes(), logical.digest().as_bytes());
    assert_ne!(recovery.as_bytes(), schema.as_bytes());
}

#[test]
fn published_projection_rejects_pending_epochs_and_hostile_graph_names() {
    let receipt = Digest256::from_bytes([0x77; Digest256::LEN]);
    let mapping = Digest256::projection_mapping(b"mapping");
    assert!(
        ProjectionCut::from_verified_receipt(
            store(1),
            mapping,
            1,
            1,
            ProjectionSourceGraph::default_graph(),
            EpochId::PENDING,
            Some(EpochId::new(2)),
            0,
            ProjectionReconciliationState::Reconciled,
            receipt,
        )
        .is_err()
    );
    assert!(
        ProjectionCut::from_verified_receipt(
            store(1),
            mapping,
            1,
            1,
            ProjectionSourceGraph::default_graph(),
            EpochId::new(1),
            Some(EpochId::PENDING),
            0,
            ProjectionReconciliationState::Reconciled,
            receipt,
        )
        .is_err()
    );
    assert!(
        ProjectionSourceGraph::named("https://example.test/a\n", GraphIncarnationId::new(1))
            .is_err()
    );
    assert!(
        ProjectionSourceGraph::named("x".repeat(64 * 1024 + 1), GraphIncarnationId::new(1),)
            .is_err()
    );
}

#[test]
fn rdf_world_cuts_require_the_history_format_and_projections_require_both_models() {
    let schema = SchemaCut::from_canonical_post_image(1, b"catalog").unwrap();
    let without_history = WorldCutDescriptor::new(
        store(1),
        EpochId::new(1),
        GraphModelTag::Rdf,
        vec![
            ModelFormatVersion::new(AuthoritativeFormat::Catalog, 1).unwrap(),
            ModelFormatVersion::new(AuthoritativeFormat::Rdf, 6).unwrap(),
        ],
        schema.clone(),
        Vec::new(),
        HistoryCompleteness::LegacyCurrentState {
            observed_at: EpochId::new(1),
            source_version: 5,
        },
    );
    assert!(without_history.is_err());

    let lpg_with_projection = WorldCutDescriptor::new(
        store(1),
        EpochId::new(11),
        GraphModelTag::Lpg,
        vec![
            ModelFormatVersion::new(AuthoritativeFormat::Catalog, 1).unwrap(),
            ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).unwrap(),
        ],
        schema,
        vec![projection(store(1))],
        HistoryCompleteness::Complete,
    );
    assert!(lpg_with_projection.is_err());
}

#[test]
fn component_sealing_requires_exact_nonzero_descriptor_formats() {
    let catalog = ModelFormatVersion::new(AuthoritativeFormat::Catalog, 5).unwrap();
    let lpg = ModelFormatVersion::new(AuthoritativeFormat::Lpg, 7).unwrap();
    let descriptor = WorldCutDescriptor::new(
        store(5),
        EpochId::new(1),
        GraphModelTag::Lpg,
        vec![catalog, lpg],
        SchemaCut::from_canonical_post_image(5, b"catalog").unwrap(),
        Vec::new(),
        HistoryCompleteness::Complete,
    )
    .unwrap();
    assert!(
        WorldCut::seal_components(descriptor.clone(), &[(catalog, b"catalog")]).is_err(),
        "omitting an authoritative format must fail"
    );
    assert!(
        WorldCut::seal_components(
            descriptor,
            &[(catalog, &b"catalog"[..]), (lpg, &b"lpg"[..])],
        )
        .is_ok()
    );

    let zero_wire = bincode::serde::encode_to_vec(
        (AuthoritativeFormat::Lpg, 0_u16),
        bincode::config::standard(),
    )
    .unwrap();
    let (zero, _): (ModelFormatVersion, usize) =
        bincode::serde::decode_from_slice(&zero_wire, bincode::config::standard()).unwrap();
    assert!(StateDigest::authoritative_components(&[(zero, b"lpg")]).is_err());
}

#[test]
fn world_cut_wire_rejects_unknown_format_versions() {
    let artifact = SnapshotArtifact::new(b"snapshot".to_vec(), descriptor(store(3))).unwrap();
    let mut encoded = bincode::serde::encode_to_vec(artifact.cut(), bincode::config::standard())
        .expect("encode cut");
    assert_eq!(encoded.first().copied(), Some(1));
    encoded[0] = 2;

    let decoded: Result<(WorldCut, usize), _> =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard());
    assert!(decoded.is_err());
}

#[test]
fn projection_reconciliation_wire_rejects_removed_legacy_tag() {
    let config = bincode::config::standard();
    for (state, expected) in [
        (ProjectionReconciliationState::Pending, vec![0]),
        (ProjectionReconciliationState::Reconciled, vec![1]),
        (ProjectionReconciliationState::NeedsReconciliation, vec![2]),
    ] {
        assert_eq!(
            bincode::serde::encode_to_vec(state, config).expect("encode reconciliation state"),
            expected
        );
    }

    let decoded: Result<(ProjectionReconciliationState, usize), _> =
        bincode::serde::decode_from_slice(&[3], config);
    assert!(decoded.is_err());
}
