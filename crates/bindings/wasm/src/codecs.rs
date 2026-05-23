//! WASM bindings for the Plan 2 compression codecs.
//!
//! The TypeScript build pipeline calls these to encode a data component
//! to a compressed blob and to open and query that blob.

#[cfg(feature = "rabitq-codec")]
use grafeo_common::types::NodeId;
#[cfg(feature = "rabitq-codec")]
use grafeo_core::index::vector::TwoStageVectorIndex;
use wasm_bindgen::prelude::*;

/// A JS-facing handle to a RaBitQ two-stage vector index.
#[cfg(feature = "rabitq-codec")]
#[wasm_bindgen]
pub struct RabitqCodec {
    inner: TwoStageVectorIndex,
}

#[cfg(feature = "rabitq-codec")]
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
                (
                    NodeId::new(u64::from(id)),
                    flat[start..start + dim].to_vec(),
                )
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
        let inner =
            TwoStageVectorIndex::from_bytes(blob).map_err(|e| JsError::new(&e.to_string()))?;
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

/// A JS-facing handle to an FSST string codec.
#[cfg(feature = "fsst-codec")]
#[wasm_bindgen]
pub struct FsstCodec {
    inner: grafeo_core::codec::FsstCodec,
}

#[cfg(feature = "fsst-codec")]
#[wasm_bindgen]
impl FsstCodec {
    /// Encodes a batch of strings into a compressed blob.
    ///
    /// `flat` is the concatenation of all string bodies; `lengths` gives
    /// the length of each string in bytes. Strings are NOT separated by
    /// any delimiter — they are reconstructed from `lengths`. The empty
    /// string is permitted (length 0). Returns the blob bytes.
    ///
    /// # Errors
    /// Returns a `JsError` if `sum(lengths) != flat.length`.
    #[wasm_bindgen(js_name = "encode")]
    pub fn encode(flat: &[u8], lengths: &[u32]) -> Result<Vec<u8>, JsError> {
        let total: u64 = lengths.iter().map(|&l| u64::from(l)).sum();
        if total != flat.len() as u64 {
            return Err(JsError::new(
                "sum of lengths must equal flat.length",
            ));
        }
        let mut strings: Vec<&[u8]> = Vec::with_capacity(lengths.len());
        let mut cursor = 0usize;
        for &len in lengths {
            let len = len as usize;
            strings.push(&flat[cursor..cursor + len]);
            cursor += len;
        }
        Ok(grafeo_core::codec::FsstCodec::build(&strings).to_bytes())
    }

    /// Opens a blob produced by [`FsstCodec::encode`] for querying.
    ///
    /// # Errors
    /// Returns a `JsError` if the blob is malformed.
    #[wasm_bindgen(js_name = "open")]
    pub fn open(blob: &[u8]) -> Result<FsstCodec, JsError> {
        let inner = grafeo_core::codec::FsstCodec::from_bytes(blob)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Decodes string `index`. Returns an empty `Vec` for an empty string
    /// AND for out-of-bounds — JS callers should check `index < codec.len()`
    /// first if they need to distinguish.
    #[wasm_bindgen(js_name = "get")]
    #[must_use]
    pub fn get(&self, index: u32) -> Vec<u8> {
        self.inner
            .get(index as usize)
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    /// Number of stored strings.
    #[wasm_bindgen(js_name = "len")]
    #[must_use]
    pub fn len(&self) -> u32 {
        // reason: count of strings fits u32 for any practical snapshot
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.len() as u32
        }
    }

    /// True if the codec holds no strings.
    #[wasm_bindgen(js_name = "isEmpty")]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// A JS-facing handle to a WebGraph adjacency codec.
#[cfg(feature = "webgraph-codec")]
#[wasm_bindgen]
pub struct WebGraphCodec {
    inner: grafeo_core::codec::WebGraphCodec,
}

#[cfg(feature = "webgraph-codec")]
#[wasm_bindgen]
impl WebGraphCodec {
    /// Encodes an edge list into a compressed adjacency blob.
    ///
    /// `srcs` and `dsts` are parallel arrays; entry `i` is the edge
    /// `srcs[i] -> dsts[i]`. All ids must be `< num_nodes`. Duplicates are
    /// de-duplicated by the underlying codec. Returns the blob bytes.
    ///
    /// # Errors
    /// Returns a `JsError` if `srcs.length != dsts.length` or if any id
    /// is `>= num_nodes`.
    #[wasm_bindgen(js_name = "encode")]
    pub fn encode(num_nodes: u32, srcs: &[u32], dsts: &[u32]) -> Result<Vec<u8>, JsError> {
        if srcs.len() != dsts.len() {
            return Err(JsError::new("srcs.length must equal dsts.length"));
        }
        let mut builder = grafeo_core::codec::WebGraphBuilder::new(u64::from(num_nodes));
        for (&s, &d) in srcs.iter().zip(dsts) {
            builder
                .add_edge(u64::from(s), u64::from(d))
                .map_err(|e| JsError::new(&e.to_string()))?;
        }
        Ok(builder.build().to_bytes())
    }

    /// Opens a blob produced by [`WebGraphCodec::encode`] for querying.
    ///
    /// # Errors
    /// Returns a `JsError` if the blob is malformed (bad magic, version,
    /// truncation, or CRC mismatch).
    #[wasm_bindgen(js_name = "open")]
    pub fn open(blob: &[u8]) -> Result<WebGraphCodec, JsError> {
        let inner = grafeo_core::codec::WebGraphCodec::from_bytes(blob)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the successors of `node` as a `Uint32Array`.
    #[wasm_bindgen(js_name = "successors")]
    #[must_use]
    pub fn successors(&self, node: u32) -> Vec<u32> {
        // reason: snapshot node ids fit u32 for the JS surface
        #[allow(clippy::cast_possible_truncation)]
        self.inner
            .successors(u64::from(node))
            .map(|d| d as u32)
            .collect()
    }

    /// Out-degree of `node`.
    #[wasm_bindgen(js_name = "outDegree")]
    #[must_use]
    pub fn out_degree(&self, node: u32) -> u32 {
        // reason: degree fits u32 for any practical snapshot
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.out_degree(u64::from(node)) as u32
        }
    }

    /// Number of nodes.
    #[wasm_bindgen(js_name = "numNodes")]
    #[must_use]
    pub fn num_nodes(&self) -> u32 {
        // reason: snapshot node count fits u32
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.num_nodes() as u32
        }
    }

    /// Number of edges.
    #[wasm_bindgen(js_name = "numEdges")]
    #[must_use]
    pub fn num_edges(&self) -> u32 {
        // reason: snapshot edge count fits u32
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.num_edges() as u32
        }
    }
}
