//! WASM bindings for the Plan 2 compression codecs.
//!
//! The TypeScript build pipeline calls these to encode a data component
//! to a compressed blob and to open and query that blob.

use grafeo_common::types::NodeId;
use grafeo_core::index::vector::TwoStageVectorIndex;
use wasm_bindgen::prelude::*;

/// A JS-facing handle to a RaBitQ two-stage vector index.
#[wasm_bindgen]
pub struct RabitqCodec {
    inner: TwoStageVectorIndex,
}

#[wasm_bindgen]
impl RabitqCodec {
    /// Encodes a batch of vectors into a compressed blob.
    ///
    /// `ids` is one node id per vector; `flat` holds `ids.length * dim`
    /// `f32` values, row-major (vector `r` occupies `flat[r*dim .. r*dim+dim]`).
    /// `seed` fixes the RaBitQ rotation. Returns the blob bytes.
    ///
    /// # Errors
    /// Returns a `JsError` if `dim` is zero, `ids` is empty, or `flat`'s
    /// length is not `ids.length * dim`.
    #[wasm_bindgen(js_name = "encode")]
    pub fn encode(ids: &[u32], flat: &[f32], dim: usize, seed: f64) -> Result<Vec<u8>, JsError> {
        if dim == 0 || ids.is_empty() {
            return Err(JsError::new("ids must be non-empty and dim must be > 0"));
        }
        if flat.len() != ids.len() * dim {
            return Err(JsError::new("flat.length must equal ids.length * dim"));
        }
        let vectors: Vec<(NodeId, Vec<f32>)> = ids
            .iter()
            .enumerate()
            .map(|(row, &id)| {
                let start = row * dim;
                (NodeId::new(u64::from(id)), flat[start..start + dim].to_vec())
            })
            .collect();
        // reason: seed arrives as a JS number; truncating the fraction is fine
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let seed_u64 = seed as u64;
        Ok(TwoStageVectorIndex::build(&vectors, dim, seed_u64).to_bytes())
    }

    /// Opens a blob produced by [`RabitqCodec::encode`] for querying.
    ///
    /// # Errors
    /// Returns a `JsError` if the blob is malformed (bad magic, version,
    /// truncation, or CRC mismatch).
    #[wasm_bindgen(js_name = "open")]
    pub fn open(blob: &[u8]) -> Result<RabitqCodec, JsError> {
        let inner = TwoStageVectorIndex::from_bytes(blob)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Searches for the `k` nearest neighbours of `query`. Returns node ids
    /// nearest-first. `rerank_factor` controls the recall/latency trade-off
    /// (8–16 is typical).
    #[wasm_bindgen(js_name = "search")]
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize, rerank_factor: usize) -> Vec<u32> {
        // reason: node ids in a snapshot fit u32 for the JS surface
        #[allow(clippy::cast_possible_truncation)]
        self.inner
            .search(query, k, rerank_factor)
            .into_iter()
            .map(|(id, _)| id.as_u64() as u32)
            .collect()
    }

    /// Number of indexed vectors.
    #[wasm_bindgen(js_name = "len")]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if the index holds no vectors.
    #[wasm_bindgen(js_name = "isEmpty")]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
