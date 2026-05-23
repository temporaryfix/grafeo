//! WebGraph-style static compressed adjacency.
//!
//! Each node's sorted successor list is encoded as gap-coded gammas: the
//! out-degree `k` is emitted in gamma, then the first gap `v1 − u` as
//! zigzag-gamma (signed, since `dst < src` is allowed), then subsequent
//! gaps `vi − v(i-1)` in plain gamma (always positive — successors are
//! sorted and de-duplicated). A bit-offset index `offsets[u]` gives O(1)
//! seek to any node's adjacency without decompressing other lists.
//!
//! Snapshots are immutable after build, so the static-sorted-graph
//! assumption holds. For mutable graphs see
//! [`crate::index::adjacency::ChunkedAdjacency`].

use super::bitstream::{BitReader, BitWriter};

/// Errors returned when opening or building a WebGraph blob.
#[derive(Debug, thiserror::Error)]
pub enum WebGraphError {
    /// The blob is shorter than the bytes a field needs.
    #[error("webgraph blob truncated: need {need} bytes, have {have}")]
    Truncated {
        /// Minimum required buffer length (end offset of the failed read).
        need: usize,
        /// Bytes available.
        have: usize,
    },
    /// The leading magic bytes are not `GWBG`.
    #[error("webgraph blob: bad magic (expected GWBG)")]
    BadMagic,
    /// The version byte is not supported.
    #[error("webgraph blob: unsupported version {0}")]
    BadVersion(u8),
    /// The trailing CRC32 does not match the body.
    #[error("webgraph blob: crc mismatch (stored {stored:#010x}, computed {computed:#010x})")]
    CrcMismatch {
        /// CRC read from the trailer.
        stored: u32,
        /// CRC computed over the body.
        computed: u32,
    },
    /// An edge references a node outside `[0, num_nodes)`.
    #[error("webgraph: edge ({src}, {dst}) out of range — num_nodes is {num_nodes}")]
    EdgeOutOfRange {
        /// Source node id.
        src: u64,
        /// Destination node id.
        dst: u64,
        /// Number of nodes declared at builder construction.
        num_nodes: u64,
    },
    /// The bit-offset for a node is malformed (decreasing or past end).
    #[error("webgraph blob: offset {offset} for node {node} exceeds bit stream length {bit_len}")]
    BadOffset {
        /// Node whose offset is invalid.
        node: u64,
        /// The out-of-range offset (bit position).
        offset: u64,
        /// Total bit-stream length.
        bit_len: u64,
    },
    /// Reached the end of the stream mid-record (malformed gamma).
    #[error("webgraph decode: unexpected end of stream at bit {0}")]
    UnexpectedEnd(u64),
}

/// Builds a [`WebGraphCodec`] from `(src, dst)` edges.
///
/// Edges may be added in any order; `build()` sorts them per source and
/// de-duplicates parallel edges.
#[derive(Debug, Clone)]
pub struct WebGraphBuilder {
    num_nodes: u64,
    edges: Vec<(u64, u64)>,
}

impl WebGraphBuilder {
    /// Creates an empty builder for a graph with `num_nodes` nodes. Node
    /// ids are `0..num_nodes`.
    #[must_use]
    pub fn new(num_nodes: u64) -> Self {
        Self {
            num_nodes,
            edges: Vec::new(),
        }
    }

    /// Adds an edge `src -> dst`.
    ///
    /// # Errors
    /// Returns `EdgeOutOfRange` if `src` or `dst` is `>= num_nodes`.
    pub fn add_edge(&mut self, src: u64, dst: u64) -> Result<(), WebGraphError> {
        if src >= self.num_nodes || dst >= self.num_nodes {
            return Err(WebGraphError::EdgeOutOfRange {
                src,
                dst,
                num_nodes: self.num_nodes,
            });
        }
        self.edges.push((src, dst));
        Ok(())
    }

    /// Number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    /// Number of edges added so far (before de-duplication).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl WebGraphBuilder {
    /// Sorts and de-duplicates the added edges, then encodes each node's
    /// adjacency list with gap+gamma.
    ///
    /// Returns a [`WebGraphCodec`] holding the bit-packed stream and a
    /// per-node bit-offset index.
    #[must_use]
    pub fn build(mut self) -> WebGraphCodec {
        // Sort by (src, dst) then de-duplicate parallel edges.
        self.edges.sort_unstable();
        self.edges.dedup();

        let mut writer = BitWriter::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(self.num_nodes as usize + 1);

        // Walk edges in source order; each node's successors are a
        // contiguous sub-slice.
        let mut idx: usize = 0;
        for u in 0..self.num_nodes {
            offsets.push(writer.bit_len());

            // Find the end of node u's successor block.
            let start = idx;
            while idx < self.edges.len() && self.edges[idx].0 == u {
                idx += 1;
            }
            let successors = &self.edges[start..idx];
            let k = successors.len() as u64;

            // Encode out-degree + 1 in gamma (so degree=0 encodes as gamma(1)).
            writer.write_gamma(k + 1);
            if k == 0 {
                continue;
            }

            // First gap is signed: v1 - u, encoded as zigzag-gamma.
            // reason: edges validated as u64 < num_nodes in add_edge
            #[allow(clippy::cast_possible_wrap)]
            let first_gap = successors[0].1 as i64 - u as i64;
            writer.write_zigzag_gamma(first_gap);

            // Subsequent gaps are strictly positive (sorted, deduped).
            for w in successors.windows(2) {
                let gap = w[1].1 - w[0].1; // > 0
                writer.write_gamma(gap);
            }
        }

        // Terminal offset = total bit length, lets `successors` know the
        // end of the last node's adjacency.
        offsets.push(writer.bit_len());

        let (bytes, bit_len) = writer.into_bytes();
        let num_edges = self.edges.len() as u64;
        WebGraphCodec {
            num_nodes: self.num_nodes,
            num_edges,
            offsets,
            bits: bytes,
            bit_len,
        }
    }
}

/// A static compressed adjacency in WebGraph-style gap+gamma encoding.
///
/// Built via [`WebGraphBuilder::build`]; loaded via [`Self::from_bytes`].
/// Per-node successor iteration is supported in-place without
/// decompressing other nodes via the bit-offset index.
#[derive(Debug, Clone)]
pub struct WebGraphCodec {
    pub(crate) num_nodes: u64,
    pub(crate) num_edges: u64,
    /// Bit offset of each node's adjacency block in `bits`. Length
    /// `num_nodes + 1`; the trailing entry is the total bit length.
    pub(crate) offsets: Vec<u64>,
    /// Bit-packed adjacency stream.
    pub(crate) bits: Vec<u8>,
    /// Number of bits actually written (`<= bits.len() * 8`).
    pub(crate) bit_len: u64,
}

impl WebGraphCodec {
    /// Number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    /// Number of edges (post-deduplication).
    #[must_use]
    pub fn num_edges(&self) -> u64 {
        self.num_edges
    }

    /// Out-degree of `node`. Reads only the first gamma of the node's
    /// adjacency block — O(log degree).
    #[must_use]
    pub fn out_degree(&self, node: u64) -> u64 {
        if node >= self.num_nodes {
            return 0;
        }
        let start = self.offsets[node as usize];
        let mut reader = BitReader::new(&self.bits, self.bit_len);
        reader.seek(start);
        // The stored value is degree + 1 (gamma encodes n >= 1).
        reader.read_gamma().unwrap_or(1) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_empty_graph_has_no_edges() {
        let codec = WebGraphBuilder::new(0).build();
        assert_eq!(codec.num_nodes(), 0);
        assert_eq!(codec.num_edges(), 0);
    }

    #[test]
    fn build_isolated_nodes_have_zero_degree() {
        let codec = WebGraphBuilder::new(5).build();
        assert_eq!(codec.num_nodes(), 5);
        assert_eq!(codec.num_edges(), 0);
        for u in 0u64..5 {
            assert_eq!(codec.out_degree(u), 0);
        }
    }

    #[test]
    fn build_records_edge_count_after_deduplication() {
        let mut b = WebGraphBuilder::new(4);
        b.add_edge(0, 1).unwrap();
        b.add_edge(0, 2).unwrap();
        b.add_edge(0, 1).unwrap(); // duplicate
        b.add_edge(1, 2).unwrap();
        let codec = b.build();
        assert_eq!(codec.num_edges(), 3); // duplicate removed
        assert_eq!(codec.out_degree(0), 2);
        assert_eq!(codec.out_degree(1), 1);
        assert_eq!(codec.out_degree(2), 0);
        assert_eq!(codec.out_degree(3), 0);
    }

    #[test]
    fn builder_records_added_edges() {
        let mut b = WebGraphBuilder::new(10);
        b.add_edge(0, 5).unwrap();
        b.add_edge(0, 3).unwrap();
        b.add_edge(2, 7).unwrap();
        assert_eq!(b.num_nodes(), 10);
        assert_eq!(b.edge_count(), 3);
    }

    #[test]
    fn builder_rejects_out_of_range_edges() {
        let mut b = WebGraphBuilder::new(5);
        assert!(matches!(
            b.add_edge(5, 0),
            Err(WebGraphError::EdgeOutOfRange { src: 5, .. })
        ));
        assert!(matches!(
            b.add_edge(0, 5),
            Err(WebGraphError::EdgeOutOfRange { dst: 5, .. })
        ));
    }
}
