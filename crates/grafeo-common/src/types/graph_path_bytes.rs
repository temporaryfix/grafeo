//! Bounded Serde adapter for the canonical checked [`GraphPath`] byte codec.
//!
//! Formats opt into this adapter instead of deriving an unchecked path decoder.

use super::{GraphPath, MAX_GRAPH_PATH_COMPONENTS, MAX_WORLD_GRAPH_NAME_BYTES};
use serde::de::{Error, SeqAccess, Visitor};
use serde::{Deserializer, Serialize, Serializer};

const MAX_BYTES: usize = 4 + MAX_GRAPH_PATH_COMPONENTS * (4 + MAX_WORLD_GRAPH_NAME_BYTES);

/// Serializes a checked path using its canonical length-prefixed byte grammar.
///
/// # Errors
/// Returns path encoding, allocation, or serializer failures.
pub fn serialize<S: Serializer>(path: &GraphPath, serializer: S) -> Result<S::Ok, S::Error> {
    path.to_bytes(MAX_BYTES)
        .map_err(serde::ser::Error::custom)?
        .serialize(serializer)
}

/// Checks any declared length before allocation, then validates the complete path.
/// Streaming formats without a length hint remain bounded by the same byte cap.
///
/// # Errors
/// Rejects invalid lengths, malformed paths, allocation failure and decoder errors.
pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<GraphPath, D::Error> {
    struct PathVisitor;
    impl<'de> Visitor<'de> for PathVisitor {
        type Value = GraphPath;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a bounded, exactly encoded graph path")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<GraphPath, A::Error> {
            let length = sequence.size_hint();
            if length.is_some_and(|length| !(4..=MAX_BYTES).contains(&length)) {
                return Err(A::Error::custom(
                    "graph path byte length exceeds its bounds",
                ));
            }
            let mut bytes = Vec::new();
            if let Some(length) = length {
                bytes.try_reserve_exact(length).map_err(A::Error::custom)?;
            }
            while let Some(byte) = sequence.next_element::<u8>()? {
                if bytes.len() == length.unwrap_or(MAX_BYTES) {
                    return Err(A::Error::custom(
                        "graph path exceeds its declared byte length",
                    ));
                }
                bytes.try_reserve(1).map_err(A::Error::custom)?;
                bytes.push(byte);
            }
            if length.is_some_and(|length| bytes.len() != length) {
                return Err(A::Error::custom("truncated graph path byte sequence"));
            }
            GraphPath::from_bytes(&bytes, MAX_BYTES).map_err(A::Error::custom)
        }
    }
    deserializer.deserialize_seq(PathVisitor)
}
