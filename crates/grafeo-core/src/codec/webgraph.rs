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
    ///
    /// Reserved for sub-plan 2d's borrowing `WebGraphView` reader, which
    /// surfaces malformed compressed streams through this variant. The
    /// current owned `from_bytes` path validates the offsets array up
    /// front so this variant is never constructed here.
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
        // reason: num_nodes is bounded by available memory on both 32-bit and 64-bit targets
        #[allow(clippy::cast_possible_truncation)]
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
        // reason: node < num_nodes, bounded by allocation size
        #[allow(clippy::cast_possible_truncation)]
        let start = self.offsets[node as usize];
        let mut reader = BitReader::new(&self.bits, self.bit_len);
        reader.seek(start);
        // The stored value is degree + 1 (gamma encodes n >= 1).
        reader.read_gamma().unwrap_or(1) - 1
    }

    /// Iterator over the successors of `node`, in ascending dst order.
    ///
    /// Reads one node's adjacency block in-place from the bit stream; no
    /// other node's data is touched.
    pub fn successors(&self, node: u64) -> SuccessorIter<'_> {
        if node >= self.num_nodes {
            return SuccessorIter::empty();
        }
        // reason: node < num_nodes, bounded by allocation size
        #[allow(clippy::cast_possible_truncation)]
        let start = self.offsets[node as usize];
        let mut reader = BitReader::new(&self.bits, self.bit_len);
        reader.seek(start);
        // The stored value is degree + 1 (gamma encodes n >= 1).
        let degree_plus_one = reader.read_gamma().unwrap_or(1);
        let degree = degree_plus_one - 1;
        SuccessorIter {
            reader,
            node,
            remaining: degree,
            last_dst: None,
        }
    }
}

/// Streaming iterator over a single node's successors.
pub struct SuccessorIter<'a> {
    reader: BitReader<'a>,
    node: u64,
    remaining: u64,
    /// `None` before the first successor, `Some(prev)` after.
    last_dst: Option<u64>,
}

impl SuccessorIter<'_> {
    fn empty() -> Self {
        Self {
            reader: BitReader::new(&[], 0),
            node: 0,
            remaining: 0,
            last_dst: None,
        }
    }
}

impl Iterator for SuccessorIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let dst = match self.last_dst {
            None => {
                let first_gap = self.reader.read_zigzag_gamma()?;
                // reason: node ids fit in i64 (validated < num_nodes at build);
                // the result is non-negative by construction.
                #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                {
                    (self.node as i64 + first_gap) as u64
                }
            }
            Some(prev) => prev + self.reader.read_gamma()?,
        };
        self.last_dst = Some(dst);
        Some(dst)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // reason: remaining is bounded by out-degree, which fits in memory
        #[allow(clippy::cast_possible_truncation)]
        let r = self.remaining as usize;
        (r, Some(r))
    }
}

/// Current WebGraph blob format version.
const BLOB_VERSION: u8 = 1;

/// Appends zero bytes until `buf.len()` is a multiple of `align`.
fn pad_to(buf: &mut Vec<u8>, align: usize) {
    while !buf.len().is_multiple_of(align) {
        buf.push(0);
    }
}

/// Reads a little-endian `u32` at `*pos`, advancing `*pos`.
// Available for future use (e.g., sub-plan 2d WebGraphView).
#[allow(dead_code)]
fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, WebGraphError> {
    let end = *pos + 4;
    let slice = buf.get(*pos..end).ok_or(WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

/// Reads a little-endian `u64` at `*pos`, advancing `*pos`.
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, WebGraphError> {
    let end = *pos + 8;
    let slice = buf.get(*pos..end).ok_or(WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

impl WebGraphCodec {
    /// Serializes to a self-describing, position-independent blob.
    ///
    /// Honours the Plan 2 zero-copy contract: a fixed 64-byte header with
    /// blob-relative `u64` section offsets, naturally-aligned arrays, and
    /// a trailing CRC32. See [`BLOB_VERSION`] for the version.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GWBG");
        buf.push(BLOB_VERSION);
        buf.push(0); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // padding
        buf.extend_from_slice(&self.num_nodes.to_le_bytes());
        buf.extend_from_slice(&self.num_edges.to_le_bytes());
        buf.extend_from_slice(&self.bit_len.to_le_bytes());

        // Four u64s: two offsets + two reserved (zero), patched after assembly.
        let offsets_pos = buf.len(); // = 32
        buf.extend_from_slice(&[0u8; 32]);

        // Offsets array (num_nodes + 1 entries).
        // reason: section offsets fit u64
        #[allow(clippy::cast_possible_truncation)]
        let offsets_offset = buf.len() as u64;
        for &o in &self.offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        // The offsets section starts at byte 64 (header end, 8-aligned) and
        // writes (num_nodes+1) × 8 bytes, so position is always 8-aligned
        // here. This pad call is kept as structural symmetry with the bits
        // section pad below; it is a no-op by construction.
        pad_to(&mut buf, 8);

        #[allow(clippy::cast_possible_truncation)]
        let bits_offset = buf.len() as u64;
        buf.extend_from_slice(&self.bits);
        pad_to(&mut buf, 4);

        // Patch the two offsets.
        buf[offsets_pos..offsets_pos + 8]
            .copy_from_slice(&offsets_offset.to_le_bytes());
        buf[offsets_pos + 8..offsets_pos + 16]
            .copy_from_slice(&bits_offset.to_le_bytes());
        // The third and fourth u64s stay zero (reserved).

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Opens a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`WebGraphError`] on bad magic, unsupported version,
    /// truncation, CRC mismatch, or a malformed offsets array.
    ///
    /// # Panics
    /// Panics if the CRC trailer slice is not exactly 4 bytes or an offsets
    /// chunk is not exactly 8 bytes — these are internal invariants upheld by
    /// `to_bytes` and cannot occur on a well-formed blob.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, WebGraphError> {
        if buf.len() < 8 {
            return Err(WebGraphError::Truncated {
                need: 8,
                have: buf.len(),
            });
        }
        if &buf[0..4] != b"GWBG" {
            return Err(WebGraphError::BadMagic);
        }
        if buf[4] != BLOB_VERSION {
            return Err(WebGraphError::BadVersion(buf[4]));
        }

        let body_end = buf.len() - 4;
        let stored = u32::from_le_bytes(buf[body_end..].try_into().expect("4 bytes"));
        let computed = crc32fast::hash(&buf[..body_end]);
        if stored != computed {
            return Err(WebGraphError::CrcMismatch { stored, computed });
        }

        let mut pos = 8;
        let num_nodes = read_u64(buf, &mut pos)?;
        let num_edges = read_u64(buf, &mut pos)?;
        let bit_len = read_u64(buf, &mut pos)?;
        // reason: section offsets are blob-relative byte indices; blobs fit in memory
        #[allow(clippy::cast_possible_truncation)]
        let offsets_offset = read_u64(buf, &mut pos)? as usize;
        #[allow(clippy::cast_possible_truncation)]
        let bits_offset = read_u64(buf, &mut pos)? as usize;
        let _reserved1 = read_u64(buf, &mut pos)?;
        let _reserved2 = read_u64(buf, &mut pos)?;

        debug_assert_eq!(pos, 64, "header end at offset 64");

        // Offsets array.
        // reason: num_nodes is bounded by available memory
        #[allow(clippy::cast_possible_truncation)]
        let n_offsets = num_nodes as usize + 1;
        let offsets_byte_end = offsets_offset + 8 * n_offsets;
        let offsets_bytes = buf
            .get(offsets_offset..offsets_byte_end)
            .ok_or(WebGraphError::Truncated {
                need: offsets_byte_end,
                have: buf.len(),
            })?;
        let mut offsets = Vec::with_capacity(n_offsets);
        for chunk in offsets_bytes.chunks_exact(8) {
            offsets.push(u64::from_le_bytes(chunk.try_into().expect("8 bytes")));
        }

        // Bit stream.
        // reason: bit_len / 8 is the byte count, bounded by available memory
        #[allow(clippy::cast_possible_truncation)]
        let bytes_needed = bit_len.div_ceil(8) as usize;
        let bits_end = bits_offset + bytes_needed;
        let bits = buf
            .get(bits_offset..bits_end)
            .ok_or(WebGraphError::Truncated {
                need: bits_end,
                have: buf.len(),
            })?
            .to_vec();

        // Validate offsets: monotonic, [0] == 0, last == bit_len.
        if let Some(&first) = offsets.first()
            && first != 0
        {
            return Err(WebGraphError::BadOffset {
                node: 0,
                offset: first,
                bit_len,
            });
        }
        for (i, w) in offsets.windows(2).enumerate() {
            if w[1] < w[0] || w[1] > bit_len {
                return Err(WebGraphError::BadOffset {
                    node: i as u64,
                    offset: w[1],
                    bit_len,
                });
            }
        }
        if let Some(&last) = offsets.last()
            && last != bit_len
        {
            return Err(WebGraphError::BadOffset {
                node: num_nodes,
                offset: last,
                bit_len,
            });
        }

        Ok(Self {
            num_nodes,
            num_edges,
            offsets,
            bits,
            bit_len,
        })
    }

    /// Opens a blob shared via [`bytes::Bytes`].
    ///
    /// Provided so callers can pass an mmap-backed `Bytes` region without
    /// copying it through `&[u8]` first. The current implementation parses
    /// into owned `Vec`s (one copy); sub-plan 2d adds a borrowing
    /// `WebGraphView` over the same wire format.
    ///
    /// # Errors
    /// Same as [`Self::from_bytes`].
    pub fn from_bytes_shared(blob: bytes::Bytes) -> Result<Self, WebGraphError> {
        Self::from_bytes(&blob)
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

    #[test]
    fn successors_match_input_for_simple_graph() {
        let mut b = WebGraphBuilder::new(6);
        // Node 0 -> {1, 3, 5}, node 2 -> {0, 4}, node 5 -> {5} (self-loop).
        for (s, d) in [(0u64, 1), (0, 3), (0, 5), (2, 0), (2, 4), (5, 5)] {
            b.add_edge(s, d).unwrap();
        }
        let codec = b.build();
        assert_eq!(
            codec.successors(0).collect::<Vec<_>>(),
            vec![1, 3, 5]
        );
        assert_eq!(codec.successors(1).collect::<Vec<_>>(), Vec::<u64>::new());
        assert_eq!(codec.successors(2).collect::<Vec<_>>(), vec![0, 4]);
        assert_eq!(codec.successors(5).collect::<Vec<_>>(), vec![5]);
    }

    #[test]
    fn successors_handles_dst_less_than_src() {
        // First gap is signed; verify dst < src round-trips correctly.
        let mut b = WebGraphBuilder::new(10);
        b.add_edge(7, 0).unwrap();
        b.add_edge(7, 1).unwrap();
        b.add_edge(7, 9).unwrap();
        let codec = b.build();
        assert_eq!(codec.successors(7).collect::<Vec<_>>(), vec![0, 1, 9]);
    }

    #[test]
    fn successors_for_out_of_range_node_is_empty() {
        let codec = WebGraphBuilder::new(3).build();
        assert_eq!(codec.successors(99).collect::<Vec<_>>(), Vec::<u64>::new());
    }

    #[test]
    fn blob_round_trip_preserves_adjacency() {
        let mut b = WebGraphBuilder::new(20);
        let edges = [
            (0u64, 1), (0, 5), (0, 17),
            (1, 0), (1, 2),
            (3, 3),
            (5, 18), (5, 19),
            (10, 0), (10, 5), (10, 11), (10, 12),
            (19, 0),
        ];
        for &(s, d) in &edges {
            b.add_edge(s, d).unwrap();
        }
        let codec = b.build();
        let blob = codec.to_bytes();

        assert_eq!(&blob[0..4], b"GWBG");
        assert_eq!(blob[4], 1);

        let reopened = WebGraphCodec::from_bytes(&blob).expect("from_bytes");
        assert_eq!(reopened.num_nodes(), codec.num_nodes());
        assert_eq!(reopened.num_edges(), codec.num_edges());
        for u in 0..codec.num_nodes() {
            let original: Vec<u64> = codec.successors(u).collect();
            let reopened_succ: Vec<u64> = reopened.successors(u).collect();
            assert_eq!(reopened_succ, original, "successors of {u} mismatched");
        }
    }

    #[test]
    fn blob_rejects_bad_magic_and_crc() {
        let mut b = WebGraphBuilder::new(3);
        b.add_edge(0, 1).unwrap();
        b.add_edge(1, 2).unwrap();
        let mut blob = b.build().to_bytes();

        let mut bad_magic = blob.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            WebGraphCodec::from_bytes(&bad_magic),
            Err(WebGraphError::BadMagic)
        ));

        let mid = blob.len() / 2;
        blob[mid] ^= 0xFF;
        assert!(matches!(
            WebGraphCodec::from_bytes(&blob),
            Err(WebGraphError::CrcMismatch { .. })
        ));
    }

    #[test]
    fn blob_from_bytes_shared_round_trip() {
        let mut b = WebGraphBuilder::new(3);
        b.add_edge(0, 2).unwrap();
        let blob = bytes::Bytes::from(b.build().to_bytes());
        let reopened = WebGraphCodec::from_bytes_shared(blob).expect("from_bytes_shared");
        assert_eq!(reopened.successors(0).collect::<Vec<_>>(), vec![2]);
    }
}
