//! Bounded complete Vector images for private CREATE/rebuild inputs only.

use super::super::{
    HnswIndex, QuantizedHnswIndex, ReadyVectorExactRestore, VectorExactRestoreGuard,
};
use super::{
    BincodeWirePreflight, HnswConfig, QuantizationType, VectorExactState, VectorIndexKind,
    VectorStateV4, exact_state_from_v4, exact_state_to_v4, preflight_v4_hnsw,
    preflight_v4_quantized,
};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};
use std::io::Write;
use std::sync::Arc;

const MAGIC: [u8; 4] = *b"VWB1";
const LIMIT: usize = 16 * 1024 * 1024;

fn invalid(reason: impl std::fmt::Display) -> Error {
    Error::Serialization(format!("invalid exact Vector WAL image: {reason}"))
}

fn decode(bytes: &[u8]) -> Result<VectorExactState> {
    if bytes.len() > LIMIT || !bytes.starts_with(&MAGIC) {
        return Err(invalid("generation or payload budget"));
    }
    let body = bytes
        .get(MAGIC.len()..)
        .ok_or_else(|| invalid("truncated header"))?;
    let mut preflight = BincodeWirePreflight::new(body, "WAL birth");
    match preflight.read_u64("concrete kind").map_err(invalid)? {
        0 => preflight_v4_hnsw(&mut preflight, &mut |_| {}).map_err(invalid)?,
        1 => preflight_v4_quantized(&mut preflight, &mut |_| {}).map_err(invalid)?,
        _ => return Err(invalid("unsupported concrete kind")),
    }
    if preflight.predicted_heap > LIMIT as u64 {
        return Err(invalid("declared allocation budget"));
    }
    preflight.finish_exact().map_err(invalid)?;
    let (state, used): (VectorStateV4, usize) =
        bincode::serde::decode_from_slice(body, bincode::config::standard().with_limit::<LIMIT>())
            .map_err(invalid)?;
    if used != body.len() {
        return Err(invalid("trailing bytes"));
    }
    exact_state_from_v4(state).map_err(invalid)
}

impl VectorIndexKind {
    /// Captures a bounded complete private creation/rebuild image.
    /// Surviving writes use sparse maintenance records instead.
    ///
    /// # Errors
    /// Rejects invalid exact state or an image exceeding the byte/decode budget.
    pub fn encode_wal_birth(&self) -> Result<Vec<u8>> {
        struct Bounded(Vec<u8>);
        impl Write for Bounded {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > LIMIT.saturating_sub(self.0.len()) {
                    return Err(std::io::Error::other("Vector WAL image exceeds 16 MiB"));
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
        let image =
            exact_state_to_v4(self.snapshot_exact_state().map_err(invalid)?).map_err(invalid)?;
        let mut bytes = Bounded(Vec::new());
        bytes.write_all(&MAGIC).map_err(invalid)?;
        bincode::serde::encode_into_std_write(image, &mut bytes, bincode::config::standard())
            .map_err(invalid)?;
        decode(&bytes.0)?;
        Ok(bytes.0)
    }

    /// Restores a fresh unpublished image against its owner and final population.
    ///
    /// `population` is the owner's complete, ascending-ID live label/property
    /// population at the commit epoch. CREATE and rebuild construct fresh
    /// indexes: retained tombstones are not valid in these full WAL images.
    /// Quantized full-precision vectors must match the authoritative values bit
    /// for bit; topology, codes, models and RNG are restored, never rebuilt.
    ///
    /// # Errors
    /// Rejects malformed, oversized or inconsistent topology/quantizer state,
    /// and any mismatch with the owner's resolved configuration or population.
    pub fn decode_wal_birth(
        bytes: &[u8],
        config: &HnswConfig,
        quantization: QuantizationType,
        population: impl IntoIterator<Item = (NodeId, Arc<[f32]>)>,
    ) -> Result<Self> {
        let state = decode(bytes)?;
        let topology = match &state {
            VectorExactState::Hnsw(state) => state,
            VectorExactState::Quantized(state) => &state.hnsw,
        };
        if !topology.deleted.is_empty() {
            return Err(invalid("fresh image contains retained tombstones"));
        }
        let mut population = population.into_iter();
        for (position, (id, _)) in topology.nodes.iter().enumerate() {
            let (expected_id, vector) = population
                .next()
                .ok_or_else(|| invalid("image contains a node outside the final population"))?;
            if *id != expected_id || vector.len() != config.dimensions {
                return Err(invalid(
                    "image differs from the final label/vector population",
                ));
            }
            if let VectorExactState::Quantized(state) = &state {
                let (stored_id, stored) = state
                    .vectors
                    .get(position)
                    .ok_or_else(|| invalid("quantized image lacks a final vector"))?;
                if stored_id != id
                    || stored.len() != vector.len()
                    || !stored
                        .iter()
                        .zip(vector.iter())
                        .all(|(left, right)| left.to_bits() == right.to_bits())
                {
                    return Err(invalid(
                        "quantized image vector differs from the final property",
                    ));
                }
            }
        }
        if population.next().is_some() {
            return Err(invalid("image omits a node from the final population"));
        }
        let index = match &state {
            VectorExactState::Hnsw(_) if quantization == QuantizationType::None => {
                Self::Hnsw(HnswIndex::with_seed(config.clone(), 0))
            }
            VectorExactState::Quantized(state) if state.quantization_type == quantization => {
                Self::Quantized(QuantizedHnswIndex::with_seed(
                    config.clone(),
                    quantization,
                    0,
                ))
            }
            _ => return Err(invalid("concrete kind differs from owner")),
        };
        let prepared = index.prepare_exact_state(state).map_err(invalid)?;
        let mutation = index
            .pin_mutation()
            .ok_or_else(|| invalid("private image mutation denied"))?;
        let guard = VectorExactRestoreGuard {
            index: &index,
            _mutation: mutation,
        };
        ReadyVectorExactRestore::bind(&index, prepared, &guard)
            .map_err(invalid)?
            .apply();
        drop(guard);
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::{HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex, VectorIndexKind};
    use grafeo_common::types::NodeId;

    #[test]
    fn vector_wal_full_images_preserve_kind_and_reject_wrong_owner_and_malformed()
    -> grafeo_common::utils::error::Result<()> {
        let config = HnswConfig::new(2, crate::index::vector::DistanceMetric::Euclidean).with_m(4);
        for quantization in [
            QuantizationType::None,
            QuantizationType::Binary,
            QuantizationType::Scalar,
            QuantizationType::Product { num_subvectors: 2 },
        ] {
            let index = if quantization == QuantizationType::None {
                VectorIndexKind::Hnsw(HnswIndex::with_seed(config.clone(), 17))
            } else {
                VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
                    config.clone(),
                    quantization,
                    17,
                ))
            };
            index.insert(NodeId::new(1), &[0.2, 0.4], &|_| None);
            let population = || [(NodeId::new(1), std::sync::Arc::<[f32]>::from([0.2, 0.4]))];
            let bytes = index.encode_wal_birth()?;
            let recovered =
                VectorIndexKind::decode_wal_birth(&bytes, &config, quantization, population())?;
            assert_eq!(bytes, recovered.encode_wal_birth()?);
            for end in 0..bytes.len() {
                assert!(
                    VectorIndexKind::decode_wal_birth(
                        &bytes[..end],
                        &config,
                        quantization,
                        population()
                    )
                    .is_err()
                );
            }
            let mut wrong = config.clone();
            wrong.dimensions = 3;
            assert!(
                VectorIndexKind::decode_wal_birth(&bytes, &wrong, quantization, population())
                    .is_err()
            );
            assert!(VectorIndexKind::decode_wal_birth(&bytes, &config, quantization, []).is_err());
            assert!(
                VectorIndexKind::decode_wal_birth(
                    &bytes,
                    &config,
                    quantization,
                    [(NodeId::new(7), std::sync::Arc::<[f32]>::from([0.2, 0.4]))]
                )
                .is_err()
            );
            assert!(
                VectorIndexKind::decode_wal_birth(
                    &bytes,
                    &config,
                    quantization,
                    [
                        population()[0].clone(),
                        (NodeId::new(2), std::sync::Arc::<[f32]>::from([0.2, 0.4]))
                    ]
                )
                .is_err()
            );
            if quantization != QuantizationType::None {
                assert!(
                    VectorIndexKind::decode_wal_birth(
                        &bytes,
                        &config,
                        quantization,
                        [(NodeId::new(1), std::sync::Arc::<[f32]>::from([0.2, 0.5]))]
                    )
                    .is_err()
                );
            }
            index.remove(NodeId::new(1));
            let tombstone = index.encode_wal_birth()?;
            assert!(
                VectorIndexKind::decode_wal_birth(&tombstone, &config, quantization, []).is_err()
            );
            let mut trailing = bytes;
            trailing.push(0);
            assert!(
                VectorIndexKind::decode_wal_birth(&trailing, &config, quantization, population())
                    .is_err()
            );
        }
        Ok(())
    }
}
