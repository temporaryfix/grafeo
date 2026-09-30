//! Checked adapters for the current portable snapshot grammar.

pub(super) mod graph_path {
    // This shared adapter bounds the declared byte sequence before reserving
    // it, then calls GraphPath::from_bytes before creating owned components.
    pub use grafeo_common::types::graph_path_bytes::{deserialize, serialize};
}

pub(super) mod properties {
    use grafeo_common::types::{EpochId, Value};
    use serde::de::{Error, SeqAccess, Visitor};
    use serde::ser::SerializeSeq;
    #[cfg(feature = "lpg")]
    use serde::{Deserialize, Serialize};
    use serde::{Deserializer, Serializer};

    type Properties = Vec<(String, Vec<(EpochId, Value)>)>;

    /// Streams borrowed histories, retaining only one encoded value at a time.
    pub fn serialize<S: Serializer>(
        properties: &[(String, Vec<(EpochId, Value)>)],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        #[cfg(not(feature = "lpg"))]
        {
            if !properties.is_empty() {
                return Err(serde::ser::Error::custom(
                    "portable LPG properties require the lpg feature",
                ));
            }
            serializer.serialize_seq(Some(0))?.end()
        }
        #[cfg(feature = "lpg")]
        {
            let mut sequence = serializer.serialize_seq(Some(properties.len()))?;
            for (key, history) in properties {
                sequence.serialize_element(&(key, HistoryRef(history)))?;
            }
            sequence.end()
        }
    }

    #[cfg(feature = "lpg")]
    struct HistoryRef<'a>(&'a [(EpochId, Value)]);

    #[cfg(feature = "lpg")]
    impl Serialize for HistoryRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
            for (epoch, value) in self.0 {
                sequence.serialize_element(&(epoch, ValueRef(value)))?;
            }
            sequence.end()
        }
    }

    #[cfg(feature = "lpg")]
    struct ValueRef<'a>(&'a Value);

    #[cfg(feature = "lpg")]
    impl Serialize for ValueRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let encoded =
                grafeo_core::graph::lpg::encode_value(self.0).map_err(serde::ser::Error::custom)?;
            serializer.serialize_bytes(&encoded)
        }
    }

    /// Decodes each value exactly; history validation remains with the caller.
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Properties, D::Error> {
        struct PropertiesVisitor;
        impl<'de> Visitor<'de> for PropertiesVisitor {
            type Value = Properties;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("property histories containing exact LPG value bytes")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Properties, A::Error> {
                #[cfg(feature = "lpg")]
                {
                    let mut properties = Vec::new();
                    while let Some((key, History(history))) =
                        sequence.next_element::<(String, History)>()?
                    {
                        properties.try_reserve(1).map_err(A::Error::custom)?;
                        properties.push((key, history));
                    }
                    Ok(properties)
                }
                #[cfg(not(feature = "lpg"))]
                {
                    // The seed refuses even a property with an empty history,
                    // before decoding its key or value on a disabled build.
                    sequence.next_element_seed(RejectProperty)?;
                    Ok(Vec::new())
                }
            }
        }
        deserializer.deserialize_seq(PropertiesVisitor)
    }

    #[cfg(not(feature = "lpg"))]
    struct RejectProperty;

    #[cfg(not(feature = "lpg"))]
    impl<'de> serde::de::DeserializeSeed<'de> for RejectProperty {
        type Value = ();

        fn deserialize<D: Deserializer<'de>>(self, _deserializer: D) -> Result<(), D::Error> {
            Err(D::Error::custom(
                "portable LPG properties require the lpg feature",
            ))
        }
    }

    #[cfg(feature = "lpg")]
    struct History(Vec<(EpochId, Value)>);

    #[cfg(feature = "lpg")]
    impl<'de> Deserialize<'de> for History {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct HistoryVisitor;
            impl<'de> Visitor<'de> for HistoryVisitor {
                type Value = History;

                fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    formatter.write_str("epoch and exact LPG value pairs")
                }

                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut sequence: A,
                ) -> Result<History, A::Error> {
                    let mut history = Vec::new();
                    while let Some((epoch, DecodedValue(value))) =
                        sequence.next_element::<(EpochId, DecodedValue)>()?
                    {
                        history.try_reserve(1).map_err(A::Error::custom)?;
                        history.push((epoch, value));
                    }
                    Ok(History(history))
                }
            }
            deserializer.deserialize_seq(HistoryVisitor)
        }
    }

    #[cfg(feature = "lpg")]
    struct DecodedValue(Value);

    #[cfg(feature = "lpg")]
    impl<'de> Deserialize<'de> for DecodedValue {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct ValueVisitor;
            impl<'de> Visitor<'de> for ValueVisitor {
                type Value = DecodedValue;

                fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    formatter.write_str("one exactly encoded LPG property value")
                }

                fn visit_bytes<E: Error>(self, bytes: &[u8]) -> Result<DecodedValue, E> {
                    grafeo_core::graph::lpg::decode_value_exact(bytes)
                        .map(DecodedValue)
                        .map_err(E::custom)
                }

                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut sequence: A,
                ) -> Result<DecodedValue, A::Error> {
                    // Binary slice decoders borrow through visit_bytes. Other
                    // Serde readers may supply a sequence; reserve incrementally
                    // so an untrusted count cannot request a giant allocation.
                    let maximum = u32::MAX as usize;
                    if sequence.size_hint().is_some_and(|length| length > maximum) {
                        return Err(A::Error::custom("LPG property value byte limit exceeded"));
                    }
                    let mut bytes = Vec::new();
                    while let Some(byte) = sequence.next_element::<u8>()? {
                        if bytes.len() == maximum {
                            return Err(A::Error::custom("LPG property value byte limit exceeded"));
                        }
                        bytes.try_reserve(1).map_err(A::Error::custom)?;
                        bytes.push(byte);
                    }
                    self.visit_bytes(&bytes)
                }
            }
            deserializer.deserialize_bytes(ValueVisitor)
        }
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{
        EpochId, GraphPath, MAX_GRAPH_PATH_COMPONENTS, MAX_WORLD_GRAPH_NAME_BYTES, Value,
    };
    use serde::de::{DeserializeSeed, SeqAccess};
    use serde::{Deserialize, Serialize};

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    #[derive(Debug, Serialize, Deserialize)]
    struct PathRecord(#[serde(with = "super::graph_path")] GraphPath);

    #[derive(Debug, Serialize, Deserialize)]
    struct PropertiesRecord(
        #[serde(with = "super::properties")] Vec<(String, Vec<(EpochId, Value)>)>,
    );

    #[test]
    fn portable_paths_use_shared_exact_bytes_and_distinguish_components() -> TestResult {
        let mut encoded_paths = Vec::new();
        for components in [
            &[][..],
            &[""][..],
            &["a/b"][..],
            &["a", "b"][..],
            &["a", "", "b"][..],
        ] {
            let path = GraphPath::from_components(components)?;
            let expected =
                bincode::serde::encode_to_vec(path.to_bytes(1024)?, bincode::config::standard())?;
            let encoded = bincode::serde::encode_to_vec(
                PathRecord(path.clone()),
                bincode::config::standard(),
            )?;
            assert_eq!(encoded, expected);
            let (decoded, consumed): (PathRecord, usize) =
                bincode::serde::decode_from_slice(&encoded, bincode::config::standard())?;
            assert_eq!(decoded.0, path);
            assert_eq!(consumed, encoded.len());
            encoded_paths.push(encoded);
        }
        let unique: std::collections::HashSet<_> = encoded_paths.iter().collect();
        assert_eq!(unique.len(), encoded_paths.len());
        let deepest = PathRecord(GraphPath::from_components(
            &[""; MAX_GRAPH_PATH_COMPONENTS],
        )?);
        let encoded = bincode::serde::encode_to_vec(&deepest, bincode::config::standard())?;
        let (decoded, _): (PathRecord, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard())?;
        assert_eq!(decoded.0, deepest.0);
        Ok(())
    }

    #[test]
    fn portable_paths_reject_malformed_shared_frames() -> TestResult {
        for frame in [
            vec![],
            vec![0, 0, 0],
            vec![0, 0, 0, 0, 1],
            u32::MAX.to_le_bytes().to_vec(),
            257_u32.to_le_bytes().to_vec(),
            vec![1, 0, 0, 0, 1, 0, 0, 0, 0xff],
            vec![1, 0, 0, 0, 1, 0, 1, 0],
        ] {
            let encoded = bincode::serde::encode_to_vec(frame, bincode::config::standard())?;
            assert!(
                bincode::serde::decode_from_slice::<PathRecord, _>(
                    &encoded,
                    bincode::config::standard()
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn portable_path_declared_size_is_rejected_before_element_access() {
        struct DeclaredLength<'a> {
            length: usize,
            reads: &'a std::cell::Cell<usize>,
        }
        impl<'de> SeqAccess<'de> for DeclaredLength<'_> {
            type Error = serde::de::value::Error;
            fn next_element_seed<T: DeserializeSeed<'de>>(
                &mut self,
                _seed: T,
            ) -> Result<Option<T::Value>, Self::Error> {
                self.reads.set(self.reads.get() + 1);
                Err(serde::de::Error::custom(
                    "element access must not be reached",
                ))
            }
            fn size_hint(&self) -> Option<usize> {
                Some(self.length)
            }
        }
        let maximum = 4 + MAX_GRAPH_PATH_COMPONENTS * (4 + MAX_WORLD_GRAPH_NAME_BYTES);
        for length in [0, 3, maximum + 1, usize::MAX] {
            let reads = std::cell::Cell::new(0);
            let deserializer = serde::de::value::SeqAccessDeserializer::new(DeclaredLength {
                length,
                reads: &reads,
            });
            let error = super::graph_path::deserialize(deserializer);
            assert!(error.is_err());
            assert_eq!(reads.get(), 0);
            assert!(
                error.err().is_some_and(|error| error
                    .to_string()
                    .contains("byte length exceeds its bounds"))
            );
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn portable_properties_match_shared_value_bytes_including_nested_temporal_bits() -> TestResult {
        use grafeo_common::types::{Duration, PropertyKey, Time};
        let value = Value::Map(std::sync::Arc::new(
            [
                (
                    PropertyKey::new("nested"),
                    Value::List(
                        vec![
                            Value::Null,
                            Value::Float64(-0.0),
                            Value::Float64(f64::from_bits(0x7ff8_0000_0000_1234)),
                            Value::Time(
                                Time::from_nanos(12_345_678_901).ok_or("invalid test time")?,
                            ),
                            Value::Duration(Duration::new(2, 3, 987_654_321)),
                        ]
                        .into(),
                    ),
                ),
                (
                    PropertyKey::new("literal"),
                    Value::RdfLiteral {
                        lexical: "2026-09-08T12:00:00.123456789Z".into(),
                        language: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#dateTime".into()),
                    },
                ),
                (
                    PropertyKey::new("vector"),
                    Value::Vector(vec![-0.0, f32::from_bits(0x7fc0_1234)].into()),
                ),
            ]
            .into_iter()
            .collect(),
        ));
        let properties = PropertiesRecord(vec![
            (
                "body".into(),
                vec![(EpochId::new(7), value), (EpochId::new(7), Value::Null)],
            ),
            ("empty".into(), Vec::new()),
        ]);
        let expected = properties
            .0
            .iter()
            .map(|(key, history)| {
                let values = history
                    .iter()
                    .map(|(epoch, value)| {
                        Ok((*epoch, grafeo_core::graph::lpg::encode_value(value)?))
                    })
                    .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
                Ok((key, values))
            })
            .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
        let encoded = bincode::serde::encode_to_vec(&properties, bincode::config::standard())?;
        assert_eq!(
            encoded,
            bincode::serde::encode_to_vec(expected, bincode::config::standard())?
        );
        let (decoded, consumed): (PropertiesRecord, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard())?;
        assert_eq!(consumed, encoded.len());
        assert_eq!(
            bincode::serde::encode_to_vec(decoded, bincode::config::standard())?,
            encoded
        );
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn portable_properties_reject_unknown_trailing_and_overdeep_values() -> TestResult {
        for value_bytes in [
            Vec::<u8>::new(),
            vec![255],
            vec![0, 0],
            vec![1, 2],
            vec![11, 255, 255, 255, 255],
        ] {
            let raw = vec![("k", vec![(EpochId::new(1), value_bytes)])];
            let bytes = bincode::serde::encode_to_vec(raw, bincode::config::standard())?;
            assert!(
                bincode::serde::decode_from_slice::<PropertiesRecord, _>(
                    &bytes,
                    bincode::config::standard()
                )
                .is_err()
            );
        }
        let mut nested = Value::Null;
        for _ in 0..130 {
            nested = Value::List(vec![nested].into());
        }
        let properties = PropertiesRecord(vec![("deep".into(), vec![(EpochId::new(1), nested)])]);
        assert!(bincode::serde::encode_to_vec(properties, bincode::config::standard()).is_err());
        Ok(())
    }

    #[cfg(not(feature = "lpg"))]
    #[test]
    fn portable_properties_without_lpg_accept_only_an_empty_property_set() -> TestResult {
        let empty = PropertiesRecord(Vec::new());
        let bytes = bincode::serde::encode_to_vec(&empty, bincode::config::standard())?;
        let (decoded, _): (PropertiesRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
        assert!(decoded.0.is_empty());
        let nonempty = PropertiesRecord(vec![("empty-history".into(), Vec::new())]);
        assert!(bincode::serde::encode_to_vec(nonempty, bincode::config::standard()).is_err());
        let raw = vec![("empty-history", Vec::<(EpochId, Vec<u8>)>::new())];
        let bytes = bincode::serde::encode_to_vec(raw, bincode::config::standard())?;
        let result = bincode::serde::decode_from_slice::<PropertiesRecord, _>(
            &bytes,
            bincode::config::standard(),
        );
        assert!(
            result
                .err()
                .is_some_and(|error| error.to_string().contains("require the lpg feature"))
        );
        Ok(())
    }
}
