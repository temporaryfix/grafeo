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

#[allow(unused_imports)]
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

#[cfg(test)]
mod tests {
    use super::*;

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
