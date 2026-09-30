//! Current owner-bearing catalog image, shared by container catalog sections.
//!
//! This is distinct from the still-guarded schema-only WAL/portable projection.
//! Wire tags are feature-independent; unavailable index kinds fail closed.

use super::{
    ANONYMOUS_INDEX_PREFIX, Catalog, CatalogState, CatalogWalStateV1, EdgeTypeWalV1,
    GraphTypeWalV1, IndexCatalog, IndexConfiguration, IndexDefinition, NamedConstraintWalV1,
    NodeTypeWalV1, ProcedureWalV1, SchemaWalStateV1,
};
use bincode::de::{
    BorrowDecode, Decoder, DecoderImpl,
    read::{BorrowReader, Reader, SliceReader},
};
use grafeo_common::types::{GraphPath, IndexId, LabelId, PropertyKeyId};
use serde::{Deserialize, Serialize};

pub(crate) const CURRENT_CATALOG_STATE_VERSION: u8 = 2;
const MAX_IMAGE_BYTES: usize = u32::MAX as usize;

/// Retains the slice reader's behavior while exposing exact byte consumption.
struct CountedCatalogReader<'a> {
    inner: SliceReader<'a>,
    remaining: usize,
}

impl Reader for CountedCatalogReader<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> Result<(), bincode::error::DecodeError> {
        self.inner.read(bytes)?;
        self.remaining -= bytes.len();
        Ok(())
    }

    fn peek_read(&mut self, length: usize) -> Option<&[u8]> {
        self.inner.peek_read(length)
    }

    fn consume(&mut self, length: usize) {
        self.inner.consume(length);
        self.remaining = self.remaining.saturating_sub(length);
    }
}

impl<'a> BorrowReader<'a> for CountedCatalogReader<'a> {
    fn take_bytes(&mut self, length: usize) -> Result<&'a [u8], bincode::error::DecodeError> {
        let bytes = self.inner.take_bytes(length)?;
        self.remaining -= length;
        Ok(bytes)
    }
}

/// Bound speculative container/string allocation by the actual payload size,
/// retaining the previous geometric catalog preflight limits without decoding
/// a second candidate. Each current decoder consumes this result exactly once.
pub(crate) fn decode_bounded<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
) -> Result<(T, usize), String> {
    if u32::try_from(bytes.len()).is_err() {
        return Err("catalog payload exceeds its byte budget".into());
    }
    let limit = match bytes.len() {
        0..=8 => 4_096,
        9..=128 => 65_536,
        129..=2_048 => 1_048_576,
        2_049..=32_768 => 16_777_216,
        32_769..=524_288 => 268_435_456,
        _ => MAX_IMAGE_BYTES,
    };
    let reader = CountedCatalogReader {
        inner: SliceReader::new(bytes),
        remaining: bytes.len(),
    };
    let mut decoder = DecoderImpl::new(
        reader,
        bincode::config::standard().with_limit::<MAX_IMAGE_BYTES>(),
        (),
    );
    let decode_error = |error| format!("bounded catalog decode failed: {error}");
    // Reserve the unused budget instead of instantiating the complete serde
    // graph for each limit. Every subsequent claim/unclaim has the same room
    // as before. On 32-bit targets, checked-add overflow still means that the
    // selected budget was exceeded and yields the same LimitExceeded error.
    decoder
        .claim_bytes_read(MAX_IMAGE_BYTES - limit)
        .map_err(decode_error)?;
    let bincode::serde::BorrowCompat(value) =
        bincode::serde::BorrowCompat::<T>::borrow_decode(&mut decoder).map_err(decode_error)?;
    Ok((value, bytes.len() - decoder.reader().remaining))
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CatalogStateV2 {
    labels: Vec<String>,
    property_keys: Vec<String>,
    edge_types: Vec<String>,
    schema: SchemaV2,
    next_index_id: u32,
    owners: Vec<OwnerV2>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::GraphTypeDefinition;
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// The former decoder remains an independent oracle for limits and errors.
    fn decode_bounded_before<T: serde::de::DeserializeOwned>(
        bytes: &[u8],
    ) -> Result<(T, usize), String> {
        if u32::try_from(bytes.len()).is_err() {
            return Err("catalog payload exceeds its byte budget".into());
        }
        fn decode<T: serde::de::DeserializeOwned, const LIMIT: usize>(
            bytes: &[u8],
        ) -> Result<(T, usize), String> {
            bincode::serde::decode_from_slice(
                bytes,
                bincode::config::standard().with_limit::<LIMIT>(),
            )
            .map_err(|error| format!("bounded catalog decode failed: {error}"))
        }
        match bytes.len() {
            0..=8 => decode::<T, 4_096>(bytes),
            9..=128 => decode::<T, 65_536>(bytes),
            129..=2_048 => decode::<T, 1_048_576>(bytes),
            2_049..=32_768 => decode::<T, 16_777_216>(bytes),
            32_769..=524_288 => decode::<T, 268_435_456>(bytes),
            _ => decode::<T, MAX_IMAGE_BYTES>(bytes),
        }
    }

    fn assert_decode_parity<T: serde::de::DeserializeOwned + Serialize>(
        bytes: &[u8],
    ) -> Result<(T, usize), String> {
        let actual = decode_bounded::<T>(bytes);
        let before = decode_bounded_before::<T>(bytes);
        let canonical = |result: &Result<(T, usize), String>| {
            result
                .as_ref()
                .map(|(value, consumed)| {
                    (
                        bincode::serde::encode_to_vec(value, bincode::config::standard())
                            .expect("parity fixture must re-encode"),
                        *consumed,
                    )
                })
                .map_err(Clone::clone)
        };
        assert_eq!(
            canonical(&actual),
            canonical(&before),
            "catalog decoder differs for {} input bytes",
            bytes.len()
        );
        actual
    }

    #[test]
    fn bounded_decoder_preserves_bucket_boundaries_and_consumed_bytes() -> TestResult {
        assert!(assert_decode_parity::<u8>(&[]).is_err());
        for length in [
            1, 8, 9, 128, 129, 2_048, 2_049, 32_768, 32_769, 524_288, 524_289,
        ] {
            let mut bytes = vec![42];
            bytes.resize(length, 0);
            assert_eq!(assert_decode_parity::<u8>(&bytes)?, (42, 1));
        }
        let nested = vec![vec![String::new()]];
        let encoded = bincode::serde::encode_to_vec(&nested, bincode::config::standard())?;
        for length in [
            8, 9, 128, 129, 2_048, 2_049, 32_768, 32_769, 524_288, 524_289,
        ] {
            let mut bytes = encoded.clone();
            bytes.resize(length, 0);
            assert_eq!(
                assert_decode_parity::<Vec<Vec<String>>>(&bytes)?,
                (nested.clone(), encoded.len())
            );
        }
        Ok(())
    }

    #[test]
    fn bounded_decoder_preserves_nested_truncation_and_invalid_input() -> TestResult {
        // Large varints exercise peek/consume, floats exercise read, and UTF-8
        // strings exercise the borrowed reader, including within containers.
        type Fixture = (u64, f64, char, Vec<Vec<String>>);
        let fixture: Fixture = (
            u64::MAX,
            -0.25,
            '雪',
            vec![vec!["λ".into(), "nested".into()], vec![], vec!["尾".into()]],
        );
        let encoded = bincode::serde::encode_to_vec(&fixture, bincode::config::standard())?;
        for end in 0..encoded.len() {
            assert!(assert_decode_parity::<Fixture>(&encoded[..end]).is_err());
        }
        for length in [encoded.len(), 128, 129, 2_049, 32_769, 524_289] {
            let mut bytes = encoded.clone();
            bytes.resize(length, 0);
            assert_eq!(
                assert_decode_parity::<Fixture>(&bytes)?,
                (fixture.clone(), encoded.len())
            );
        }
        assert!(assert_decode_parity::<String>(&[1, 0xff]).is_err());
        assert!(assert_decode_parity::<bool>(&[2]).is_err());
        assert!(assert_decode_parity::<char>(&[0xff]).is_err());
        let huge_length = bincode::serde::encode_to_vec(u64::MAX, bincode::config::standard())?;
        assert!(assert_decode_parity::<String>(&huge_length).is_err());
        Ok(())
    }

    #[test]
    fn bounded_decoder_preserves_six_allocation_limits_and_overflow_denial() -> TestResult {
        // A string first claims eight bytes for its length. Two enclosing
        // sequences claim another sixteen. At the exact remaining budget the
        // forged string reaches the reader and fails as truncated; one byte
        // above it must fail admission before the reader or allocation.
        for (length, limit) in [
            (8, 4_096),
            (9, 65_536),
            (128, 65_536),
            (129, 1_048_576),
            (2_048, 1_048_576),
            (2_049, 16_777_216),
            (32_768, 16_777_216),
            (32_769, 268_435_456),
            (524_288, 268_435_456),
            (524_289, MAX_IMAGE_BYTES),
        ] {
            for nested in [false, true] {
                let length_claims = if nested { 24 } else { 8 };
                for excess in [0, 1] {
                    let declared = limit - length_claims + excess;
                    let declared_wire = u64::try_from(declared)?;
                    let mut bytes = if nested {
                        bincode::serde::encode_to_vec(
                            (1u64, 1u64, declared_wire),
                            bincode::config::standard(),
                        )?
                    } else {
                        bincode::serde::encode_to_vec(declared_wire, bincode::config::standard())?
                    };
                    let header_bytes = bytes.len();
                    assert!(header_bytes <= length);
                    bytes.resize(length, 0);
                    let error = if nested {
                        assert_decode_parity::<Vec<Vec<String>>>(&bytes)
                            .expect_err("forged nested length must fail before allocation")
                    } else {
                        assert_decode_parity::<String>(&bytes)
                            .expect_err("forged string length must fail before allocation")
                    };
                    let expected = if excess == 0 {
                        bincode::error::DecodeError::UnexpectedEnd {
                            additional: declared - (length - header_bytes),
                        }
                    } else {
                        // MAX_IMAGE_BYTES + 1 cannot fit usize on WASM32.
                        // The decoder must report the same budget error there.
                        bincode::error::DecodeError::LimitExceeded
                    };
                    assert_eq!(
                        error,
                        format!("bounded catalog decode failed: {expected}"),
                        "length={length}, limit={limit}, nested={nested}, excess={excess}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bounded_decoder_preclaim_preserves_claim_unclaim_at_u32_limit() -> TestResult {
        fn check<const LIMIT: usize>() -> TestResult {
            let mut before = DecoderImpl::new(
                SliceReader::new(&[]),
                bincode::config::standard().with_limit::<LIMIT>(),
                (),
            );
            let mut shared = DecoderImpl::new(
                SliceReader::new(&[]),
                bincode::config::standard().with_limit::<MAX_IMAGE_BYTES>(),
                (),
            );
            shared.claim_bytes_read(MAX_IMAGE_BYTES - LIMIT)?;
            before.claim_bytes_read(LIMIT)?;
            shared.claim_bytes_read(LIMIT)?;
            before.unclaim_bytes_read(LIMIT / 2);
            shared.unclaim_bytes_read(LIMIT / 2);
            before.claim_bytes_read(LIMIT / 2)?;
            shared.claim_bytes_read(LIMIT / 2)?;
            assert!(matches!(
                before.claim_bytes_read(1),
                Err(bincode::error::DecodeError::LimitExceeded)
            ));
            assert!(matches!(
                shared.claim_bytes_read(1),
                Err(bincode::error::DecodeError::LimitExceeded)
            ));
            Ok(())
        }
        check::<4_096>()?;
        check::<65_536>()?;
        check::<1_048_576>()?;
        check::<16_777_216>()?;
        check::<268_435_456>()?;
        check::<MAX_IMAGE_BYTES>()
    }

    #[test]
    fn bounded_decoder_preserves_current_catalog_wire() -> TestResult {
        let encoded = source()?.encode_current_state_v2()?;
        for end in 0..encoded.len() {
            assert!(assert_decode_parity::<(u8, CatalogStateV2)>(&encoded[..end]).is_err());
        }
        for length in [encoded.len(), 2_049, 32_769, 524_289] {
            let mut bytes = encoded.clone();
            bytes.resize(length.max(encoded.len()), 0);
            let (decoded, consumed) = assert_decode_parity::<(u8, CatalogStateV2)>(&bytes)?;
            assert_eq!(consumed, encoded.len());
            assert_eq!(
                bincode::serde::encode_to_vec(&decoded, bincode::config::standard())?,
                encoded
            );
        }
        Ok(())
    }

    fn source() -> Result<Catalog, Box<dyn std::error::Error>> {
        let catalog = Catalog::new();
        catalog.register_graph_type(GraphTypeDefinition {
            name: "Open".into(),
            allowed_node_types: Vec::new(),
            allowed_edge_types: Vec::new(),
            open: true,
        })?;
        let label = catalog.get_or_create_label("Item")?;
        let property = catalog.get_or_create_property_key("value")?;
        let gap = catalog.create_index(
            None,
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Property,
        )?;
        assert!(catalog.drop_index(gap));
        for path in [
            GraphPath::root(),
            GraphPath::from_components(&[""])?,
            GraphPath::from_components(&["a/b"])?,
            GraphPath::from_components(&["a", "b"])?,
        ] {
            catalog.bind_graph_type(&path, "Open".into())?;
            catalog.create_index(None, label, property, path, IndexConfiguration::BTree)?;
        }
        Ok(catalog)
    }

    #[test]
    fn current_state_retains_real_ids_gaps_paths_bindings_and_exhausted_floor() -> TestResult {
        let catalog = source()?;
        let bytes = catalog.encode_current_state_v2()?;
        assert!(
            catalog.encode_wal_state_v1().is_err(),
            "flat WAL remains guarded for these bindings"
        );
        let restored = Catalog::decode_current_state_v2(&bytes)?;
        assert_eq!(restored.all_indexes().len(), 4);
        for owner in catalog.all_indexes() {
            assert_eq!(restored.get_index(owner.id), Some(owner));
        }
        assert_eq!(restored.index_allocator_high_water(), 5);
        assert_eq!(restored.encode_current_state_v2()?, bytes);
        for owner in restored.all_indexes() {
            assert!(restored.drop_index(owner.id));
        }
        let mut state = restored.current_state_v2()?;
        state.next_index_id = u32::MAX;
        let exhausted = Catalog::from_current_state_v2(state)?;
        assert_eq!(exhausted.index_count(), 0);
        assert_eq!(exhausted.index_allocator_high_water(), u32::MAX);
        let roundtrip = Catalog::decode_current_state_v2(&exhausted.encode_current_state_v2()?)?;
        assert_eq!(roundtrip.index_allocator_high_water(), u32::MAX);
        assert!(
            roundtrip
                .create_index(
                    Some("cannot-reuse"),
                    LabelId::new(0),
                    PropertyKeyId::new(0),
                    GraphPath::root(),
                    IndexConfiguration::Property
                )
                .is_err()
        );
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn current_owner_union_preserves_ids_duplicates_and_retired_floor() -> TestResult {
        let source = source()?;
        let expected = source.encode_current_state_v2()?;
        let duplicate = Catalog::decode_current_state_v2(&expected)?;
        let mut schema = source.current_state_v2()?;
        schema.owners.clear();
        schema.next_index_id = 0;
        let merged = Catalog::from_current_state_v2(schema)?
            .merge_current_index_owners(&[source, duplicate])?;
        assert_eq!(merged.encode_current_state_v2()?, expected);

        let mut retired = merged.current_state_v2()?;
        retired.owners.clear();
        retired.next_index_id = u32::MAX;
        let retired = Catalog::from_current_state_v2(retired)?;
        let empty = Catalog::new().merge_current_index_owners(&[retired])?;
        assert_eq!(empty.index_count(), 0);
        assert_eq!(empty.index_allocator_high_water(), u32::MAX);
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn current_owner_union_keeps_disjoint_exact_ids_without_reallocation() -> TestResult {
        let left = source()?;
        let left_ids: Vec<_> = left
            .all_indexes()
            .into_iter()
            .map(|owner| owner.id)
            .collect();
        let mut right = left.current_state_v2()?;
        right.owners.truncate(1);
        let owner = right.owners.first_mut().ok_or("fixture owner missing")?;
        owner.id = right.next_index_id;
        right.next_index_id += 1;
        owner.name = "independent-owner".into();
        owner.graph = GraphPath::from_components(&["independent"])?;
        let expected_id = IndexId::new(owner.id);
        let expected_floor = right.next_index_id;
        let right = Catalog::from_current_state_v2(right)?;
        let expected = right.get_index(expected_id).ok_or("source owner missing")?;
        let mut schema = left.current_state_v2()?;
        schema.owners.clear();
        schema.next_index_id = 0;
        let merged =
            Catalog::from_current_state_v2(schema)?.merge_current_index_owners(&[left, right])?;
        assert_eq!(merged.get_index(expected_id), Some(expected));
        assert_eq!(merged.index_allocator_high_water(), expected_floor);
        for id in left_ids {
            assert!(merged.get_index(id).is_some());
        }
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn current_owner_union_rejects_id_name_and_physical_conflicts() -> TestResult {
        for kind in ["id", "name", "physical"] {
            let left = source()?;
            let mut right = left.current_state_v2()?;
            right.owners.truncate(1);
            let owner = right.owners.first_mut().ok_or("fixture owner missing")?;
            match kind {
                "id" => owner.name = "different-owner".into(),
                "name" => {
                    owner.name = "shared-name".into();
                    let mut left_wire = left.current_state_v2()?;
                    left_wire
                        .owners
                        .first_mut()
                        .ok_or("fixture owner missing")?
                        .name = "shared-name".into();
                    let left = Catalog::from_current_state_v2(left_wire)?;
                    owner.id = right.next_index_id;
                    right.next_index_id += 1;
                    let mut schema = left.current_state_v2()?;
                    schema.owners.clear();
                    schema.next_index_id = 0;
                    assert!(
                        Catalog::from_current_state_v2(schema)?
                            .merge_current_index_owners(&[
                                left,
                                Catalog::from_current_state_v2(right)?
                            ])
                            .is_err()
                    );
                    continue;
                }
                _ => {
                    owner.id = right.next_index_id;
                    right.next_index_id += 1;
                    owner.name = "different-owner".into();
                }
            }
            let mut schema = left.current_state_v2()?;
            schema.owners.clear();
            schema.next_index_id = 0;
            assert!(
                Catalog::from_current_state_v2(schema)?
                    .merge_current_index_owners(&[left, Catalog::from_current_state_v2(right)?])
                    .is_err(),
                "{kind}"
            );
        }
        Ok(())
    }

    #[test]
    fn current_state_rejects_invalid_owner_images_and_predecessors() -> TestResult {
        let source = source()?;
        for mutation in 0..6 {
            let mut wire = source.current_state_v2()?;
            match mutation {
                0 => wire.next_index_id = 0,
                1 => wire.owners.swap(0, 1),
                2 => wire.owners[0].label_id = u32::MAX,
                3 => wire.owners[0].property_id = u32::MAX,
                4 => wire.owners[0].name = "@grafeo-index:999".into(),
                _ => wire.owners[1].graph = wire.owners[0].graph.clone(),
            }
            assert!(
                Catalog::from_current_state_v2(wire).is_err(),
                "accepted corruption {mutation}"
            );
        }
        let bytes = source.encode_current_state_v2()?;
        for version in [0, 1, 3, 255] {
            let mut invalid = bytes.clone();
            invalid[0] = version;
            assert!(Catalog::decode_current_state_v2(&invalid).is_err());
        }
        for end in 0..bytes.len() {
            assert!(Catalog::decode_current_state_v2(&bytes[..end]).is_err());
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(Catalog::decode_current_state_v2(&trailing).is_err());
        Ok(())
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn current_state_retains_bm25_bits_and_zero_minimum_token_length() -> TestResult {
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("Doc")?;
        let property = catalog.get_or_create_property_key("body")?;
        let id = catalog.create_index(
            Some("text"),
            label,
            property,
            GraphPath::root(),
            IndexConfiguration::Text {
                config: grafeo_core::index::text::BM25Config { k1: -0.0, b: 0.625 },
                min_token_length: 0,
            },
        )?;
        let restored = Catalog::decode_current_state_v2(&catalog.encode_current_state_v2()?)?;
        assert_eq!(restored.get_index(id), catalog.get_index(id));
        Ok(())
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn current_state_retains_every_vector_configuration_kind_without_defaults() -> TestResult {
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, QuantizationType};
        let catalog = Catalog::new();
        let label = catalog.get_or_create_label("Embedding")?;
        for (position, quantization) in [
            QuantizationType::None,
            QuantizationType::Scalar,
            QuantizationType::Binary,
            QuantizationType::Product { num_subvectors: 2 },
        ]
        .into_iter()
        .enumerate()
        {
            let property = catalog.get_or_create_property_key(&format!("v{position}"))?;
            let config = HnswConfig {
                dimensions: 8,
                metric: DistanceMetric::Manhattan,
                m: 3,
                m_max: 7,
                ef_construction: 29,
                ef: 17,
                ml: 0.625,
                alpha: 1.25,
                max_elements: Some(91),
            };
            catalog.create_index(
                None,
                label,
                property,
                GraphPath::from_components(&["a", "b"])?,
                IndexConfiguration::Vector {
                    config,
                    quantization,
                },
            )?;
        }
        let restored = Catalog::decode_current_state_v2(&catalog.encode_current_state_v2()?)?;
        for owner in catalog.all_indexes() {
            assert_eq!(restored.get_index(owner.id), Some(owner));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct BindingV2 {
    #[serde(with = "grafeo_common::types::graph_path_bytes")]
    graph: GraphPath,
    graph_type: String,
}

#[derive(Serialize, Deserialize)]
struct SchemaV2 {
    unique_constraints: Vec<(u32, u32)>,
    required_properties: Vec<(u32, u32)>,
    named_constraints: Vec<NamedConstraintWalV1>,
    node_types: Vec<NodeTypeWalV1>,
    edge_types: Vec<EdgeTypeWalV1>,
    graph_types: Vec<GraphTypeWalV1>,
    schemas: Vec<String>,
    graph_type_bindings: Vec<BindingV2>,
    procedures: Vec<ProcedureWalV1>,
}

#[derive(Serialize, Deserialize)]
struct OwnerV2 {
    id: u32,
    name: String,
    #[serde(with = "grafeo_common::types::graph_path_bytes")]
    graph: GraphPath,
    label_id: u32,
    property_id: u32,
    configuration: ConfigurationV2,
}

/// Declaration order is the frozen, feature-independent wire discriminant.
#[derive(Serialize, Deserialize)]
pub(super) enum ConfigurationV2 {
    Property,
    BTree,
    Text {
        k1_bits: u64,
        b_bits: u64,
        min_token_length: u64,
    },
    Vector {
        dimensions: u64,
        metric: MetricV2,
        m: u64,
        m_max: u64,
        ef_construction: u64,
        ef: u64,
        ml_bits: u64,
        alpha_bits: u32,
        max_elements: Option<u64>,
        quantization: QuantizationV2,
    },
}

#[derive(Serialize, Deserialize)]
pub(super) enum MetricV2 {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

#[derive(Serialize, Deserialize)]
pub(super) enum QuantizationV2 {
    None,
    Scalar,
    Binary,
    Product { num_subvectors: u64 },
}

impl Catalog {
    /// Preserves exact owner identities while merging already-validated catalogs.
    /// Dictionary references are resolved by name in the merged schema; owners
    /// are never admitted through the allocator or inferred from physical state.
    #[cfg(feature = "lpg")]
    pub(crate) fn merge_current_index_owners(self, sources: &[Catalog]) -> Result<Self, String> {
        use std::collections::BTreeMap;

        let mut wire = self.current_state_v2()?;
        if !wire.owners.is_empty() || wire.next_index_id != 0 {
            return Err("catalog owner merge requires a schema-only destination".into());
        }
        let mut owners = BTreeMap::new();
        let mut names = BTreeMap::new();
        let mut keys = BTreeMap::new();
        for source in sources {
            wire.next_index_id = wire.next_index_id.max(source.index_allocator_high_water());
            for mut owner in source.all_indexes() {
                let label_name = source
                    .get_label_name(owner.label)
                    .ok_or_else(|| "source owner label is missing".to_string())?;
                let property_name = source
                    .get_property_key_name(owner.property_key)
                    .ok_or_else(|| "source owner property is missing".to_string())?;
                owner.label = self
                    .get_label_id(&label_name)
                    .ok_or_else(|| "merged owner label is missing".to_string())?;
                owner.property_key = self
                    .get_property_key_id(&property_name)
                    .ok_or_else(|| "merged owner property is missing".to_string())?;
                if let Some(existing) = owners.get(&owner.id) {
                    if existing != &owner {
                        return Err(format!("open_multi: index owner ID {} conflicts", owner.id));
                    }
                    continue;
                }
                if names.insert(owner.name.clone(), owner.id).is_some() {
                    return Err(format!(
                        "open_multi: index owner name {:?} conflicts",
                        owner.name
                    ));
                }
                if keys.insert(owner.key.clone(), owner.id).is_some() {
                    return Err(format!(
                        "open_multi: physical index owner {:?} conflicts",
                        owner.key
                    ));
                }
                owners.insert(owner.id, owner);
            }
        }
        wire.owners = owners
            .into_values()
            .map(|owner| {
                Ok(OwnerV2 {
                    id: owner.id.as_u32(),
                    name: owner.name,
                    graph: owner.key.graph().clone(),
                    label_id: owner.label.as_u32(),
                    property_id: owner.property_key.as_u32(),
                    configuration: ConfigurationV2::from_live(&owner.configuration)?,
                })
            })
            .collect::<Result<_, String>>()?;
        Self::from_current_state_v2(wire)
    }

    pub(crate) fn current_state_v2(&self) -> Result<CatalogStateV2, String> {
        self.state.read().current_state_v2()
    }

    pub(crate) fn from_current_state_v2(wire: CatalogStateV2) -> Result<Self, String> {
        wire.into_catalog()
    }

    pub(crate) fn encode_current_state_v2(&self) -> Result<Vec<u8>, String> {
        let state = self.current_state_v2()?;
        bincode::serde::encode_to_vec(
            (CURRENT_CATALOG_STATE_VERSION, state),
            bincode::config::standard().with_limit::<MAX_IMAGE_BYTES>(),
        )
        .map_err(|error| format!("cannot encode catalog state2: {error}"))
    }

    #[cfg(any(feature = "grafeo-file", test))]
    pub(crate) fn decode_current_state_v2(bytes: &[u8]) -> Result<Self, String> {
        if bytes.first() != Some(&CURRENT_CATALOG_STATE_VERSION) {
            return Err("unsupported catalog state generation; expected 2".into());
        }
        let ((version, state), consumed): ((u8, CatalogStateV2), usize) = decode_bounded(bytes)?;
        if version != CURRENT_CATALOG_STATE_VERSION || consumed != bytes.len() {
            return Err("catalog state2 version or exact length mismatch".into());
        }
        Self::from_current_state_v2(state)
    }
}

impl CatalogState {
    pub(crate) fn current_state_v2(&self) -> Result<CatalogStateV2, String> {
        let mut snapshot = self.snapshot_state();
        let schema = snapshot
            .schema
            .as_mut()
            .ok_or_else(|| "current catalog state requires schema support".to_string())?;
        let mut bindings: Vec<_> = std::mem::take(&mut schema.graph_type_bindings)
            .into_iter()
            .map(|(graph, graph_type)| BindingV2 { graph, graph_type })
            .collect();
        bindings.sort_unstable_by(|left, right| left.graph.cmp(&right.graph));
        // Reuse exact dictionary/schema conversions, not their flat wire bytes.
        let base = CatalogWalStateV1::from_snapshot(snapshot)?;
        let SchemaWalStateV1 {
            unique_constraints,
            required_properties,
            named_constraints,
            node_types,
            edge_types,
            graph_types,
            schemas,
            graph_type_bindings: _,
            procedures,
        } = base.schema;
        let mut owners = self.indexes.all();
        owners.sort_unstable_by_key(|owner| owner.id);
        let owners = owners
            .into_iter()
            .map(|owner| {
                Ok(OwnerV2 {
                    id: owner.id.as_u32(),
                    name: owner.name,
                    graph: owner.key.graph().clone(),
                    label_id: owner.label.as_u32(),
                    property_id: owner.property_key.as_u32(),
                    configuration: ConfigurationV2::from_live(&owner.configuration)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(CatalogStateV2 {
            labels: base.labels,
            property_keys: base.property_keys,
            edge_types: base.edge_types,
            schema: SchemaV2 {
                unique_constraints,
                required_properties,
                named_constraints,
                node_types,
                edge_types,
                graph_types,
                schemas,
                graph_type_bindings: bindings,
                procedures,
            },
            next_index_id: self.indexes.next_id,
            owners,
        })
    }
}

impl CatalogStateV2 {
    fn into_catalog(self) -> Result<Catalog, String> {
        let SchemaV2 {
            unique_constraints,
            required_properties,
            named_constraints,
            node_types,
            edge_types,
            graph_types,
            schemas,
            graph_type_bindings,
            procedures,
        } = self.schema;
        let base = CatalogWalStateV1 {
            labels: self.labels,
            property_keys: self.property_keys,
            edge_types: self.edge_types,
            schema: SchemaWalStateV1 {
                unique_constraints,
                required_properties,
                named_constraints,
                node_types,
                edge_types,
                graph_types,
                schemas,
                graph_type_bindings: Vec::new(),
                procedures,
            },
        };
        let mut state = CatalogState::from_snapshot(base.into_snapshot()?)
            .map_err(|error| error.to_string())?;
        let mut previous = None;
        for binding in graph_type_bindings {
            if previous.as_ref().is_some_and(|path| path >= &binding.graph) {
                return Err("catalog state2 bindings must be sorted and unique".into());
            }
            previous = Some(binding.graph.clone());
            state
                .bind_graph_type(&binding.graph, binding.graph_type)
                .map_err(|error| error.to_string())?;
        }
        let mut indexes = IndexCatalog::new();
        let count = self.owners.len();
        indexes
            .indexes
            .try_reserve(count)
            .map_err(|error| error.to_string())?;
        indexes
            .name_index
            .try_reserve(count)
            .map_err(|error| error.to_string())?;
        indexes
            .physical_owners
            .try_reserve(count)
            .map_err(|error| error.to_string())?;
        indexes
            .label_indexes
            .try_reserve(count)
            .map_err(|error| error.to_string())?;
        indexes
            .label_property_indexes
            .try_reserve(count)
            .map_err(|error| error.to_string())?;
        let mut previous_id = None;
        for owner in self.owners {
            if owner.id >= self.next_index_id || previous_id.is_some_and(|id| id >= owner.id) {
                return Err("catalog state2 owner IDs must be sorted, unique and below their allocator floor".into());
            }
            previous_id = Some(owner.id);
            if owner.name.starts_with(ANONYMOUS_INDEX_PREFIX)
                && owner.name != format!("{ANONYMOUS_INDEX_PREFIX}{}", owner.id)
            {
                return Err("catalog state2 anonymous owner name disagrees with its ID".into());
            }
            let label_id = LabelId::new(owner.label_id);
            let property_id = PropertyKeyId::new(owner.property_id);
            let label = state
                .labels
                .get_name(label_id)
                .ok_or_else(|| "catalog state2 owner references missing label".to_string())?;
            let property = state
                .property_keys
                .get_name(property_id)
                .ok_or_else(|| "catalog state2 owner references missing property".to_string())?;
            let configuration = owner.configuration.into_live()?;
            configuration
                .validate()
                .map_err(|error| error.to_string())?;
            let key = configuration.physical_key(owner.graph, &label, &property);
            if indexes.name_index.contains_key(&owner.name)
                || indexes.physical_owners.contains_key(&key)
            {
                return Err(
                    "catalog state2 has duplicate owner name or natural physical identity".into(),
                );
            }
            let id = IndexId::new(owner.id);
            indexes.name_index.insert(owner.name.clone(), id);
            indexes.physical_owners.insert(key.clone(), id);
            indexes.label_indexes.entry(label_id).or_default().push(id);
            indexes
                .label_property_indexes
                .entry((label_id, property_id))
                .or_default()
                .push(id);
            indexes.indexes.insert(
                id,
                IndexDefinition {
                    id,
                    name: owner.name,
                    label: label_id,
                    property_key: property_id,
                    key,
                    index_type: configuration.index_type(),
                    configuration,
                },
            );
        }
        indexes.next_id = self.next_index_id;
        state.indexes = indexes;
        Ok(Catalog {
            state: super::state::CatalogStateLock::new(state),
        })
    }
}

impl ConfigurationV2 {
    pub(super) fn from_live(configuration: &IndexConfiguration) -> Result<Self, String> {
        configuration
            .validate()
            .map_err(|error| error.to_string())?;
        match configuration {
            IndexConfiguration::Property => Ok(Self::Property),
            IndexConfiguration::BTree => Ok(Self::BTree),
            #[cfg(feature = "text-index")]
            IndexConfiguration::Text {
                config,
                min_token_length,
            } => Ok(Self::Text {
                k1_bits: config.k1.to_bits(),
                b_bits: config.b.to_bits(),
                min_token_length: u64::try_from(*min_token_length)
                    .map_err(|error| error.to_string())?,
            }),
            #[cfg(feature = "vector-index")]
            IndexConfiguration::Vector {
                config,
                quantization,
            } => {
                use grafeo_core::index::vector::{DistanceMetric, QuantizationType};
                let metric = match config.metric {
                    DistanceMetric::Cosine => MetricV2::Cosine,
                    DistanceMetric::Euclidean => MetricV2::Euclidean,
                    DistanceMetric::DotProduct => MetricV2::DotProduct,
                    DistanceMetric::Manhattan => MetricV2::Manhattan,
                    _ => return Err("unsupported current vector metric".into()),
                };
                let quantization = match quantization {
                    QuantizationType::None => QuantizationV2::None,
                    QuantizationType::Scalar => QuantizationV2::Scalar,
                    QuantizationType::Binary => QuantizationV2::Binary,
                    QuantizationType::Product { num_subvectors } => QuantizationV2::Product {
                        num_subvectors: u64::try_from(*num_subvectors)
                            .map_err(|error| error.to_string())?,
                    },
                    _ => return Err("unsupported current vector quantization".into()),
                };
                Ok(Self::Vector {
                    dimensions: u64::try_from(config.dimensions)
                        .map_err(|error| error.to_string())?,
                    metric,
                    m: u64::try_from(config.m).map_err(|error| error.to_string())?,
                    m_max: u64::try_from(config.m_max).map_err(|error| error.to_string())?,
                    ef_construction: u64::try_from(config.ef_construction)
                        .map_err(|error| error.to_string())?,
                    ef: u64::try_from(config.ef).map_err(|error| error.to_string())?,
                    ml_bits: config.ml.to_bits(),
                    alpha_bits: config.alpha.to_bits(),
                    max_elements: config
                        .max_elements
                        .map(u64::try_from)
                        .transpose()
                        .map_err(|error| error.to_string())?,
                    quantization,
                })
            }
        }
    }

    pub(super) fn into_live(self) -> Result<IndexConfiguration, String> {
        match self {
            Self::Property => Ok(IndexConfiguration::Property),
            Self::BTree => Ok(IndexConfiguration::BTree),
            #[cfg(feature = "text-index")]
            Self::Text {
                k1_bits,
                b_bits,
                min_token_length,
            } => Ok(IndexConfiguration::Text {
                config: grafeo_core::index::text::BM25Config {
                    k1: f64::from_bits(k1_bits),
                    b: f64::from_bits(b_bits),
                },
                min_token_length: usize::try_from(min_token_length)
                    .map_err(|error| error.to_string())?,
            }),
            #[cfg(not(feature = "text-index"))]
            Self::Text { .. } => {
                Err("catalog state2 contains Text owners but text-index is disabled".into())
            }
            #[cfg(feature = "vector-index")]
            Self::Vector {
                dimensions,
                metric,
                m,
                m_max,
                ef_construction,
                ef,
                ml_bits,
                alpha_bits,
                max_elements,
                quantization,
            } => {
                use grafeo_core::index::vector::{DistanceMetric, HnswConfig, QuantizationType};
                let metric = match metric {
                    MetricV2::Cosine => DistanceMetric::Cosine,
                    MetricV2::Euclidean => DistanceMetric::Euclidean,
                    MetricV2::DotProduct => DistanceMetric::DotProduct,
                    MetricV2::Manhattan => DistanceMetric::Manhattan,
                };
                let quantization = match quantization {
                    QuantizationV2::None => QuantizationType::None,
                    QuantizationV2::Scalar => QuantizationType::Scalar,
                    QuantizationV2::Binary => QuantizationType::Binary,
                    QuantizationV2::Product { num_subvectors } => QuantizationType::Product {
                        num_subvectors: usize::try_from(num_subvectors)
                            .map_err(|error| error.to_string())?,
                    },
                };
                Ok(IndexConfiguration::Vector {
                    config: HnswConfig {
                        dimensions: usize::try_from(dimensions)
                            .map_err(|error| error.to_string())?,
                        metric,
                        m: usize::try_from(m).map_err(|error| error.to_string())?,
                        m_max: usize::try_from(m_max).map_err(|error| error.to_string())?,
                        ef_construction: usize::try_from(ef_construction)
                            .map_err(|error| error.to_string())?,
                        ef: usize::try_from(ef).map_err(|error| error.to_string())?,
                        ml: f64::from_bits(ml_bits),
                        alpha: f32::from_bits(alpha_bits),
                        max_elements: max_elements
                            .map(usize::try_from)
                            .transpose()
                            .map_err(|error| error.to_string())?,
                    },
                    quantization,
                })
            }
            #[cfg(not(feature = "vector-index"))]
            Self::Vector { .. } => {
                Err("catalog state2 contains Vector owners but vector-index is disabled".into())
            }
        }
    }
}
