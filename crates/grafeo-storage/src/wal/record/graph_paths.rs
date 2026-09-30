//! Bounded collections of individually checked native graph coordinates.

use grafeo_common::types::GraphPath;
use serde::de::{Error, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MAX_PATHS: usize = 64 * 1024 * 1024 / std::mem::size_of::<GraphPath>();

#[derive(Serialize)]
struct PathRef<'a>(#[serde(with = "grafeo_common::types::graph_path_bytes")] &'a GraphPath);

#[derive(Deserialize)]
struct CheckedPath(#[serde(with = "grafeo_common::types::graph_path_bytes")] GraphPath);

pub(super) fn serialize<S: Serializer>(
    paths: &[GraphPath],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if paths.len() > MAX_PATHS {
        return Err(serde::ser::Error::custom(
            "WAL graph path collection exceeds its budget",
        ));
    }
    let mut sequence = serializer.serialize_seq(Some(paths.len()))?;
    for path in paths {
        sequence.serialize_element(&PathRef(path))?;
    }
    sequence.end()
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<GraphPath>, D::Error> {
    struct PathsVisitor;
    impl<'de> Visitor<'de> for PathsVisitor {
        type Value = Vec<GraphPath>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a bounded collection of checked graph paths")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let count = sequence.size_hint();
            if count.is_some_and(|count| count > MAX_PATHS) {
                return Err(A::Error::custom(
                    "WAL graph path collection exceeds its budget",
                ));
            }
            let mut paths = Vec::new();
            // Do not allocate the declared vector on a tiny truncated input.
            // Each element first passes the shared path-byte length validator.
            while let Some(CheckedPath(path)) = sequence.next_element()? {
                if paths.len() == MAX_PATHS || count.is_some_and(|count| paths.len() == count) {
                    return Err(A::Error::custom(
                        "graph path collection exceeds its bound or declared count",
                    ));
                }
                paths.try_reserve(1).map_err(A::Error::custom)?;
                paths.push(path);
            }
            if count.is_some_and(|count| paths.len() != count) {
                return Err(A::Error::custom("truncated graph path collection"));
            }
            Ok(paths)
        }
    }
    deserializer.deserialize_seq(PathsVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    struct Paths(#[serde(with = "super")] Vec<GraphPath>);

    #[test]
    fn collections_preserve_literal_paths_in_json_and_bincode() {
        let paths = Paths(vec![
            GraphPath::from_components(&[""]).unwrap(),
            GraphPath::from_components(&["a", "b"]).unwrap(),
            GraphPath::from_components(&["a/b"]).unwrap(),
        ]);
        let json = serde_json::to_string(&paths).unwrap();
        assert_eq!(serde_json::from_str::<Paths>(&json).unwrap().0, paths.0);
        let bytes = bincode::serde::encode_to_vec(&paths, bincode::config::standard()).unwrap();
        let (decoded, consumed): (Paths, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(decoded.0, paths.0);
    }

    #[test]
    fn collection_claims_are_bounded_before_allocation() {
        for count in [u64::MAX, u64::try_from(MAX_PATHS + 1).unwrap(), 1] {
            let bytes = bincode::serde::encode_to_vec(count, bincode::config::standard()).unwrap();
            assert!(
                bincode::serde::decode_from_slice::<Paths, _>(&bytes, bincode::config::standard())
                    .is_err()
            );
        }
        assert!(serde_json::from_str::<Paths>("[[1,1,0,0]]").is_err());
    }
}
