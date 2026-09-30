//! Current transaction-owned logical/physical index publication envelope.

use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;

use crate::catalog::{IndexOwnerBatch, IndexOwnerImage};

const VERSION: u8 = 3;
const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
pub(crate) struct TextPostimage {
    pub(crate) owner: IndexOwnerImage,
    pub(crate) birth: bool,
    pub(crate) payload: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct VectorPostimage {
    pub(crate) owner: IndexOwnerImage,
    pub(crate) complete: bool,
    pub(crate) payload: Vec<u8>,
}

pub(crate) struct IndexCommitBatch {
    pub(crate) frontier: EpochId,
    pub(crate) commit: EpochId,
    pub(crate) owners: IndexOwnerBatch,
    pub(crate) text: Vec<TextPostimage>,
    pub(crate) vectors: Vec<VectorPostimage>,
}

#[derive(Deserialize)]
struct Wire {
    version: u8,
    frontier: EpochId,
    commit: EpochId,
    owners: Vec<u8>,
    text: Vec<TextPostimage>,
    vectors: Vec<VectorPostimage>,
}

#[derive(Serialize)]
struct WireRef<'a> {
    version: u8,
    frontier: EpochId,
    commit: EpochId,
    owners: &'a [u8],
    text: &'a [TextPostimage],
    vectors: &'a [VectorPostimage],
}

fn invalid(reason: impl std::fmt::Display) -> Error {
    Error::Serialization(format!("invalid current index commit batch: {reason}"))
}

impl IndexCommitBatch {
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        struct Bounded(Vec<u8>);
        impl Write for Bounded {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > MAX_BYTES.saturating_sub(self.0.len()) {
                    return Err(std::io::Error::other("index commit batch exceeds 16 MiB"));
                }
                self.0
                    .try_reserve(bytes.len())
                    .map_err(std::io::Error::other)?;
                self.0.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let owners = self.owners.encode().map_err(invalid)?;
        let mut bytes = Bounded(Vec::new());
        bincode::serde::encode_into_std_write(
            WireRef {
                version: VERSION,
                frontier: self.frontier,
                commit: self.commit,
                owners: &owners,
                text: &self.text,
                vectors: &self.vectors,
            },
            &mut bytes,
            bincode::config::standard(),
        )
        .map_err(invalid)?;
        Self::decode(&bytes.0)?;
        Ok(bytes.0)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_BYTES || bytes.first() != Some(&VERSION) {
            return Err(invalid("unsupported generation or oversized payload"));
        }
        let (wire, used): (Wire, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_BYTES>(),
        )
        .map_err(invalid)?;
        if used != bytes.len()
            || wire.version != VERSION
            || wire.commit == EpochId::PENDING
            || wire.frontier >= wire.commit
        {
            return Err(invalid("length, version or publication frontier mismatch"));
        }
        let owners = IndexOwnerBatch::decode(&wire.owners).map_err(invalid)?;
        #[cfg(feature = "text-index")]
        {
            let mut identities = std::collections::HashSet::new();
            for image in &wire.text {
                if !identities.insert(image.owner.id) || image.payload.is_empty() {
                    return Err(invalid("duplicate or empty Text postimage"));
                }
                if !matches!(
                    image.owner.configuration,
                    crate::catalog::IndexConfiguration::Text { .. }
                ) {
                    return Err(invalid("Text postimage has a non-Text owner"));
                }
            }
        }
        #[cfg(not(feature = "text-index"))]
        if !wire.text.is_empty() {
            return Err(invalid("Text postimage requires text-index support"));
        }
        #[cfg(feature = "vector-index")]
        {
            let mut identities = std::collections::HashSet::new();
            for image in &wire.vectors {
                if !identities.insert(image.owner.id)
                    || image.payload.is_empty()
                    || !matches!(
                        image.owner.configuration,
                        crate::catalog::IndexConfiguration::Vector { .. }
                    )
                {
                    return Err(invalid("duplicate, empty or wrong-family Vector postimage"));
                }
            }
        }
        #[cfg(not(feature = "vector-index"))]
        if !wire.vectors.is_empty() {
            return Err(invalid("Vector postimage requires vector-index support"));
        }
        Ok(Self {
            frontier: wire.frontier,
            commit: wire.commit,
            owners,
            text: wire.text,
            vectors: wire.vectors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::IndexConfiguration;
    use grafeo_common::types::{GraphPath, IndexId};

    fn batch() -> IndexCommitBatch {
        IndexCommitBatch {
            frontier: EpochId::new(3),
            commit: EpochId::new(4),
            owners: IndexOwnerBatch {
                expected_floor: 7,
                next_floor: 7,
                changes: Vec::new(),
            },
            text: Vec::new(),
            vectors: Vec::new(),
        }
    }

    fn image(configuration: IndexConfiguration, payload: Vec<u8>) -> TextPostimage {
        TextPostimage {
            owner: IndexOwnerImage {
                id: IndexId::new(1),
                name: "owned".into(),
                graph: GraphPath::root(),
                label: "Doc".into(),
                property: "body".into(),
                configuration,
            },
            birth: false,
            payload,
        }
    }

    fn unchecked(batch: &IndexCommitBatch) -> Result<Vec<u8>> {
        let owners = batch.owners.encode().map_err(invalid)?;
        bincode::serde::encode_to_vec(
            WireRef {
                version: VERSION,
                frontier: batch.frontier,
                commit: batch.commit,
                owners: &owners,
                text: &batch.text,
                vectors: &batch.vectors,
            },
            bincode::config::standard(),
        )
        .map_err(invalid)
    }

    #[test]
    fn current_envelope_roundtrips_and_rejects_invalid_epochs() -> Result<()> {
        let source = batch();
        let decoded = IndexCommitBatch::decode(&source.encode()?)?;
        assert_eq!(decoded.frontier, source.frontier);
        assert_eq!(decoded.commit, source.commit);
        assert_eq!(decoded.owners, source.owners);
        assert!(decoded.text.is_empty());
        for (frontier, commit) in [
            (EpochId::new(3), EpochId::new(3)),
            (EpochId::new(4), EpochId::new(3)),
            (EpochId::PENDING, EpochId::new(4)),
            (EpochId::new(3), EpochId::PENDING),
        ] {
            let source = IndexCommitBatch {
                frontier,
                commit,
                ..batch()
            };
            assert!(IndexCommitBatch::decode(&unchecked(&source)?).is_err());
            assert!(source.encode().is_err());
        }
        Ok(())
    }

    #[test]
    fn envelope_rejects_versions_truncation_trailing_bytes_and_bounds() -> Result<()> {
        let bytes = batch().encode()?;
        for version in [0, 1, 2, 4, u8::MAX] {
            let mut bad = bytes.clone();
            bad[0] = version;
            assert!(IndexCommitBatch::decode(&bad).is_err());
        }
        for end in 0..bytes.len() {
            assert!(IndexCommitBatch::decode(&bytes[..end]).is_err());
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(IndexCommitBatch::decode(&trailing).is_err());
        let mut oversized = vec![0; MAX_BYTES + 1];
        oversized[0] = VERSION;
        assert!(IndexCommitBatch::decode(&oversized).is_err());
        let malformed_owner = bincode::serde::encode_to_vec(
            WireRef {
                version: VERSION,
                frontier: EpochId::new(3),
                commit: EpochId::new(4),
                owners: &[u8::MAX],
                text: &[],
                vectors: &[],
            },
            bincode::config::standard(),
        )
        .map_err(invalid)?;
        assert!(IndexCommitBatch::decode(&malformed_owner).is_err());
        Ok(())
    }

    #[test]
    fn text_envelope_rejects_non_text_descriptors_in_every_profile() -> Result<()> {
        let source = IndexCommitBatch {
            text: vec![image(IndexConfiguration::Property, vec![1])],
            ..batch()
        };
        assert!(IndexCommitBatch::decode(&unchecked(&source)?).is_err());
        assert!(source.encode().is_err());
        Ok(())
    }

    fn vector_image(configuration: IndexConfiguration, payload: Vec<u8>) -> VectorPostimage {
        VectorPostimage {
            owner: image(configuration, Vec::new()).owner,
            complete: true,
            payload,
        }
    }

    #[test]
    fn vector_envelope_rejects_non_vector_descriptors_in_every_profile() -> Result<()> {
        let source = IndexCommitBatch {
            vectors: vec![vector_image(IndexConfiguration::Property, vec![1])],
            ..batch()
        };
        assert!(IndexCommitBatch::decode(&unchecked(&source)?).is_err());
        assert!(source.encode().is_err());
        Ok(())
    }

    #[test]
    #[cfg(feature = "vector-index")]
    fn vector_envelope_preserves_bytes_and_rejects_duplicate_empty_oversized() -> Result<()> {
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, QuantizationType};
        let configuration = IndexConfiguration::Vector {
            config: HnswConfig::new(4, DistanceMetric::Euclidean),
            quantization: QuantizationType::Binary,
        };
        let mut source = IndexCommitBatch {
            vectors: vec![vector_image(configuration.clone(), vec![1, 2, 3])],
            ..batch()
        };
        let decoded = IndexCommitBatch::decode(&source.encode()?)?;
        assert_eq!(decoded.vectors.len(), 1);
        assert_eq!(decoded.vectors[0].owner, source.vectors[0].owner);
        assert_eq!(decoded.vectors[0].payload, source.vectors[0].payload);
        assert!(decoded.vectors[0].complete);
        source.vectors.push(vector_image(configuration, vec![4]));
        assert!(IndexCommitBatch::decode(&unchecked(&source)?).is_err());
        assert!(source.encode().is_err());
        assert!(source.vectors.pop().is_some());
        source.vectors[0].payload.clear();
        assert!(IndexCommitBatch::decode(&unchecked(&source)?).is_err());
        assert!(source.encode().is_err());
        source.vectors[0].payload.resize(MAX_BYTES, 0);
        assert!(source.encode().is_err());
        Ok(())
    }

    #[test]
    #[cfg(feature = "text-index")]
    fn text_envelope_preserves_opaque_bytes_and_rejects_duplicate_empty_oversized() -> Result<()> {
        let configuration = IndexConfiguration::Text {
            config: grafeo_core::index::text::BM25Config { k1: 2.3, b: 0.4 },
            min_token_length: 1,
        };
        let mut source = IndexCommitBatch {
            text: vec![image(configuration.clone(), vec![1, 2, 3])],
            ..batch()
        };
        let decoded = IndexCommitBatch::decode(&source.encode()?)?;
        assert_eq!(decoded.text.len(), 1);
        assert_eq!(decoded.text[0].owner, source.text[0].owner);
        assert_eq!(decoded.text[0].payload, source.text[0].payload);
        assert!(!decoded.text[0].birth);
        source.text.push(image(configuration, vec![4]));
        assert!(IndexCommitBatch::decode(&unchecked(&source)?).is_err());
        assert!(source.encode().is_err());
        assert!(source.text.pop().is_some());
        source.text[0].payload.clear();
        assert!(IndexCommitBatch::decode(&unchecked(&source)?).is_err());
        assert!(source.encode().is_err());
        source.text[0].payload.resize(MAX_BYTES, 0);
        assert!(source.encode().is_err());
        Ok(())
    }
}
