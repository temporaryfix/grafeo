//! World identity/cut value types are available from the facade in every profile.

use grafeo::{
    AuthoritativeFormat, Digest256, EpochId, GraphModelTag, HistoryCompleteness,
    MAX_RECOVERY_IMAGE_COMPONENTS, ModelFormatVersion, RecoveryImageComponent, RecoveryImageDigest,
    SchemaCut, StoreId, WorldCut, WorldCutDescriptor, WorldMetadataSectionV2,
};

#[test]
fn facade_exports_portable_world_types_without_rdf_operational_features() {
    let store_id = StoreId::from_bytes([7; StoreId::LEN]).unwrap();
    let descriptor = WorldCutDescriptor::new(
        store_id,
        EpochId::new(1),
        GraphModelTag::Lpg,
        vec![
            ModelFormatVersion::new(AuthoritativeFormat::Catalog, 1).unwrap(),
            ModelFormatVersion::new(AuthoritativeFormat::Lpg, 1).unwrap(),
        ],
        SchemaCut::new(1, Digest256::schema(b"catalog")).unwrap(),
        Vec::new(),
        HistoryCompleteness::Complete,
    )
    .unwrap();

    assert_eq!(descriptor.store_id(), store_id);

    let logical_components = [
        (descriptor.formats()[0], b"catalog".as_slice()),
        (descriptor.formats()[1], b"lpg".as_slice()),
    ];
    let cut = WorldCut::seal_components(descriptor, &logical_components).unwrap();
    let recovery_components = [
        RecoveryImageComponent::new(1, 1, b"catalog").unwrap(),
        RecoveryImageComponent::new(2, 1, b"lpg").unwrap(),
    ];
    let digest = RecoveryImageDigest::from_components(&recovery_components).unwrap();
    let metadata = WorldMetadataSectionV2::seal(cut, &recovery_components).unwrap();
    assert_eq!(metadata.recovery_image_digest(), digest);
    assert!(MAX_RECOVERY_IMAGE_COMPONENTS >= recovery_components.len());
}
