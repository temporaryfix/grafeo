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

use super::bitstream::{BitReader, BitWriter, BitWriterError};
use std::sync::OnceLock;

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
    /// A size or offset cannot be represented by the current address space.
    #[error("webgraph blob: {0} overflows the current address space")]
    SizeOverflow(&'static str),
    /// The current grammar requires one canonical header and section layout.
    #[error("webgraph blob: non-canonical {0}")]
    NonCanonical(&'static str),
    /// The validated blob is too large to allocate its owned representation.
    #[error("webgraph blob: cannot allocate {count} entries for {field}")]
    AllocationFailed {
        /// The logical allocation being attempted.
        field: &'static str,
        /// Number of entries requested.
        count: usize,
    },
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
    /// One encoded adjacency violates the current graph grammar.
    #[error("webgraph decode: malformed adjacency for node {node}: {reason}")]
    MalformedAdjacency {
        /// Source node whose adjacency is invalid.
        node: u64,
        /// Stable diagnostic for the failed invariant.
        reason: &'static str,
    },
    /// The header edge count disagrees with the validated adjacency stream.
    #[error("webgraph blob: header declares {declared} edges, decoded {decoded}")]
    EdgeCountMismatch {
        /// Edge count stored in the header.
        declared: u64,
        /// Edge count decoded from every adjacency block.
        decoded: u64,
    },
    /// Current graph data cannot be represented by the WebGraph bit grammar.
    #[error("webgraph encode: {0}")]
    Encoding(&'static str),
}

impl From<BitWriterError> for WebGraphError {
    fn from(error: BitWriterError) -> Self {
        match error {
            BitWriterError::AllocationFailed { bytes } => Self::AllocationFailed {
                field: "bit stream",
                count: bytes,
            },
            BitWriterError::LengthOverflow | BitWriterError::AddressSpaceOverflow => {
                Self::SizeOverflow("bit stream")
            }
            BitWriterError::InvalidBitWidth(_) => Self::Encoding("invalid bit width"),
            BitWriterError::InvalidGamma(_) => Self::Encoding("gamma value outside codec domain"),
            BitWriterError::InvalidZigzagGamma(_) => {
                Self::Encoding("first successor gap outside codec domain")
            }
            BitWriterError::InconsistentState => Self::Encoding("inconsistent bit-stream state"),
        }
    }
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
        if self.edges.len() == self.edges.capacity() {
            let count = self
                .edges
                .len()
                .checked_add(1)
                .ok_or(WebGraphError::SizeOverflow("edge list length"))?;
            self.edges
                .try_reserve(1)
                .map_err(|_| WebGraphError::AllocationFailed {
                    field: "edge list",
                    count,
                })?;
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
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if the graph size or a successor gap cannot
    /// be represented, or if codec allocation fails.
    pub fn build(mut self) -> Result<WebGraphCodec, WebGraphError> {
        // Sort by (src, dst) then de-duplicate parallel edges.
        self.edges.sort_unstable();
        self.edges.dedup();

        let mut writer = BitWriter::new();
        let offset_count = usize::try_from(self.num_nodes)
            .map_err(|_| WebGraphError::SizeOverflow("offset count"))?
            .checked_add(1)
            .ok_or(WebGraphError::SizeOverflow("offset count"))?;
        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(offset_count)
            .map_err(|_| WebGraphError::AllocationFailed {
                field: "offset array",
                count: offset_count,
            })?;

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
            let degree = u64::try_from(successors.len())
                .map_err(|_| WebGraphError::SizeOverflow("out-degree"))?;

            // Encode out-degree + 1 in gamma (so degree=0 encodes as gamma(1)).
            let encoded_degree = degree
                .checked_add(1)
                .ok_or(WebGraphError::SizeOverflow("encoded out-degree"))?;
            writer.write_gamma(encoded_degree)?;
            if degree == 0 {
                continue;
            }

            // First gap is signed: v1 - u, encoded as zigzag-gamma.
            let first_gap = i128::from(successors[0].1) - i128::from(u);
            let first_gap = i64::try_from(first_gap)
                .map_err(|_| WebGraphError::Encoding("first successor gap exceeds i64"))?;
            writer.write_zigzag_gamma(first_gap)?;

            // Subsequent gaps are strictly positive (sorted, deduped).
            for w in successors.windows(2) {
                let gap = w[1].1 - w[0].1; // > 0
                writer.write_gamma(gap)?;
            }
        }

        // Terminal offset = total bit length, lets `successors` know the
        // end of the last node's adjacency.
        offsets.push(writer.bit_len());

        let (bytes, bit_len) = writer.into_bytes();
        let num_edges = u64::try_from(self.edges.len())
            .map_err(|_| WebGraphError::SizeOverflow("edge count"))?;
        Ok(WebGraphCodec {
            num_nodes: self.num_nodes,
            num_edges,
            offsets,
            bits: bytes,
            bit_len,
            validated: OnceLock::from(()),
        })
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
    /// Successful full-stream validation, pre-seeded for builder output.
    validated: OnceLock<()>,
}

impl WebGraphCodec {
    /// Number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    /// Number of edges (post-deduplication), after validating the complete
    /// adjacency stream once.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if a reopened blob contains malformed
    /// adjacency data or its declared edge count is false.
    pub fn num_edges(&self) -> Result<u64, WebGraphError> {
        self.validate()?;
        Ok(self.num_edges)
    }

    /// Validates every compressed adjacency block and the declared edge count.
    /// Successful validation is cached. Builder-produced codecs begin in the
    /// validated state; reopened codecs pay this O(nodes + edges) cost only if
    /// full validation is requested.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] on malformed gamma data, invalid successor
    /// coordinates, trailing block bits, or a false edge count.
    pub fn validate(&self) -> Result<(), WebGraphError> {
        if self.validated.get().is_some() {
            return Ok(());
        }
        validate_adjacency_stream(&self.bits, &self.offsets, self.num_nodes, self.num_edges)?;
        let _ = self.validated.set(());
        Ok(())
    }

    /// Out-degree of `node`. Reads only the first gamma of the node's
    /// adjacency block — O(log degree).
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if lazy full-stream validation fails.
    pub fn out_degree(&self, node: u64) -> Result<u64, WebGraphError> {
        self.validate()?;
        let Some((start, end)) = adjacency_bounds(&self.offsets, self.num_nodes, node)? else {
            return Ok(0);
        };
        let mut reader = BitReader::new(&self.bits, end);
        reader.seek(start);
        // The stored value is degree + 1 (gamma encodes n >= 1).
        let degree = reader
            .read_gamma()
            .ok_or(WebGraphError::UnexpectedEnd(end))?
            - 1;
        if degree > self.num_nodes {
            return Err(WebGraphError::MalformedAdjacency {
                node,
                reason: "degree exceeds node count",
            });
        }
        Ok(degree)
    }

    /// Iterator over the successors of `node`, in ascending dst order.
    ///
    /// Reads one node's adjacency block in-place from the bit stream; no
    /// other node's data is touched.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if lazy full-stream validation fails.
    pub fn successors(&self, node: u64) -> Result<SuccessorIter<'_>, WebGraphError> {
        self.validate()?;
        let Some((start, end)) = adjacency_bounds(&self.offsets, self.num_nodes, node)? else {
            return Ok(SuccessorIter::empty());
        };
        let mut reader = BitReader::new(&self.bits, end);
        reader.seek(start);
        // The stored value is degree + 1 (gamma encodes n >= 1).
        let degree = reader
            .read_gamma()
            .ok_or(WebGraphError::UnexpectedEnd(end))?
            - 1;
        if degree > self.num_nodes {
            return Err(WebGraphError::MalformedAdjacency {
                node,
                reason: "degree exceeds node count",
            });
        }
        Ok(SuccessorIter {
            reader,
            node,
            remaining: degree,
            last_dst: None,
        })
    }
}

fn adjacency_bounds(
    offsets: &[u64],
    num_nodes: u64,
    node: u64,
) -> Result<Option<(u64, u64)>, WebGraphError> {
    if node >= num_nodes {
        return Ok(None);
    }
    let index =
        usize::try_from(node).map_err(|_| WebGraphError::SizeOverflow("adjacency node index"))?;
    let next = index
        .checked_add(1)
        .ok_or(WebGraphError::SizeOverflow("adjacency offset index"))?;
    let start = offsets
        .get(index)
        .copied()
        .ok_or(WebGraphError::NonCanonical("missing adjacency start"))?;
    let end = offsets
        .get(next)
        .copied()
        .ok_or(WebGraphError::NonCanonical("missing adjacency end"))?;
    Ok(Some((start, end)))
}

/// Streaming iterator over a single node's successors.
pub struct SuccessorIter<'a> {
    reader: BitReader<'a>,
    node: u64,
    remaining: u64,
    /// `None` before the first successor, `Some(prev)` after.
    last_dst: Option<u64>,
}

impl<'a> SuccessorIter<'a> {
    fn empty() -> Self {
        Self {
            reader: BitReader::new(&[], 0),
            node: 0,
            remaining: 0,
            last_dst: None,
        }
    }

    pub(crate) fn new(reader: BitReader<'a>, node: u64, remaining: u64) -> Self {
        Self {
            reader,
            node,
            remaining,
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
                if first_gap >= 0 {
                    self.node.checked_add(u64::try_from(first_gap).ok()?)?
                } else {
                    self.node.checked_sub(first_gap.unsigned_abs())?
                }
            }
            Some(prev) => prev.checked_add(self.reader.read_gamma()?)?,
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

/// Reads a little-endian `u64` at `*pos`, advancing `*pos`.
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, WebGraphError> {
    let end = pos
        .checked_add(8)
        .ok_or(WebGraphError::SizeOverflow("u64 field end"))?;
    let slice = buf.get(*pos..end).ok_or(WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    let bytes: [u8; 8] = slice.try_into().map_err(|_| WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, WebGraphError> {
    let end = pos
        .checked_add(4)
        .ok_or(WebGraphError::SizeOverflow("u32 field end"))?;
    let slice = buf.get(*pos..end).ok_or(WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    let bytes: [u8; 4] = slice.try_into().map_err(|_| WebGraphError::Truncated {
        need: end,
        have: buf.len(),
    })?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes))
}

struct ParsedWebGraphBlob {
    num_nodes: u64,
    num_edges: u64,
    bit_len: u64,
    offsets: Vec<u64>,
    bits_offset: usize,
    bits_byte_len: usize,
}

fn checked_align_up(
    value: usize,
    alignment: usize,
    field: &'static str,
) -> Result<usize, WebGraphError> {
    let remainder = value % alignment;
    let padding = if remainder == 0 {
        0
    } else {
        alignment - remainder
    };
    value
        .checked_add(padding)
        .ok_or(WebGraphError::SizeOverflow(field))
}

fn validate_adjacency_stream(
    bits: &[u8],
    offsets: &[u64],
    num_nodes: u64,
    declared_edges: u64,
) -> Result<(), WebGraphError> {
    let mut decoded_edges = 0_u64;
    for (node_index, bounds) in offsets.windows(2).enumerate() {
        let node = u64::try_from(node_index)
            .map_err(|_| WebGraphError::SizeOverflow("adjacency node index"))?;
        let start = bounds[0];
        let end = bounds[1];
        let mut reader = BitReader::new(bits, end);
        reader.seek(start);
        let degree = reader
            .read_gamma()
            .ok_or(WebGraphError::UnexpectedEnd(end))?
            - 1;
        if degree > num_nodes {
            return Err(WebGraphError::MalformedAdjacency {
                node,
                reason: "degree exceeds node count",
            });
        }
        decoded_edges = decoded_edges
            .checked_add(degree)
            .ok_or(WebGraphError::SizeOverflow("decoded edge count"))?;

        let mut previous = None;
        for edge_index in 0..degree {
            let destination = if edge_index == 0 {
                let gap = reader
                    .read_zigzag_gamma()
                    .ok_or(WebGraphError::UnexpectedEnd(end))?;
                let signed = i128::from(node) + i128::from(gap);
                u64::try_from(signed).map_err(|_| WebGraphError::MalformedAdjacency {
                    node,
                    reason: "first successor gap is outside the node-id domain",
                })?
            } else {
                let gap = reader
                    .read_gamma()
                    .ok_or(WebGraphError::UnexpectedEnd(end))?;
                previous
                    .and_then(|value: u64| value.checked_add(gap))
                    .ok_or(WebGraphError::MalformedAdjacency {
                        node,
                        reason: "successor gap overflows the node-id domain",
                    })?
            };
            if destination >= num_nodes {
                return Err(WebGraphError::EdgeOutOfRange {
                    src: node,
                    dst: destination,
                    num_nodes,
                });
            }
            previous = Some(destination);
        }
        if reader.read_bit().is_some() {
            return Err(WebGraphError::MalformedAdjacency {
                node,
                reason: "block contains trailing bits",
            });
        }
    }
    if decoded_edges != declared_edges {
        return Err(WebGraphError::EdgeCountMismatch {
            declared: declared_edges,
            decoded: decoded_edges,
        });
    }
    Ok(())
}

fn parse_blob(buf: &[u8]) -> Result<ParsedWebGraphBlob, WebGraphError> {
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
    if buf[5] != 0 || buf[6..8] != [0, 0] {
        return Err(WebGraphError::NonCanonical("header flags or padding"));
    }

    let body_end = buf
        .len()
        .checked_sub(4)
        .ok_or(WebGraphError::SizeOverflow("CRC trailer start"))?;
    let mut crc_pos = body_end;
    let stored = read_u32(buf, &mut crc_pos)?;
    let computed = crc32fast::hash(&buf[..body_end]);
    if stored != computed {
        return Err(WebGraphError::CrcMismatch { stored, computed });
    }

    let mut pos = 8;
    let num_nodes = read_u64(buf, &mut pos)?;
    let num_edges = read_u64(buf, &mut pos)?;
    let bit_len = read_u64(buf, &mut pos)?;
    let offsets_offset = usize::try_from(read_u64(buf, &mut pos)?)
        .map_err(|_| WebGraphError::SizeOverflow("offset-array offset"))?;
    let bits_offset = usize::try_from(read_u64(buf, &mut pos)?)
        .map_err(|_| WebGraphError::SizeOverflow("bit-stream offset"))?;
    let reserved1 = read_u64(buf, &mut pos)?;
    let reserved2 = read_u64(buf, &mut pos)?;
    if pos != 64 || offsets_offset != 64 || reserved1 != 0 || reserved2 != 0 {
        return Err(WebGraphError::NonCanonical("header or reserved fields"));
    }

    let n_offsets = usize::try_from(num_nodes)
        .map_err(|_| WebGraphError::SizeOverflow("offset count"))?
        .checked_add(1)
        .ok_or(WebGraphError::SizeOverflow("offset count"))?;
    let offsets_byte_len = n_offsets
        .checked_mul(8)
        .ok_or(WebGraphError::SizeOverflow("offset array length"))?;
    let offsets_byte_end = offsets_offset
        .checked_add(offsets_byte_len)
        .ok_or(WebGraphError::SizeOverflow("offset array end"))?;
    let expected_bits_offset = checked_align_up(offsets_byte_end, 8, "bit-stream alignment")?;
    if bits_offset != expected_bits_offset {
        return Err(WebGraphError::NonCanonical("section offsets"));
    }
    let offsets_bytes =
        buf.get(offsets_offset..offsets_byte_end)
            .ok_or(WebGraphError::Truncated {
                need: offsets_byte_end,
                have: buf.len(),
            })?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(n_offsets)
        .map_err(|_| WebGraphError::AllocationFailed {
            field: "offset array",
            count: n_offsets,
        })?;
    for chunk in offsets_bytes.chunks_exact(8) {
        let &[byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7] = chunk else {
            return Err(WebGraphError::NonCanonical("offset entry width"));
        };
        offsets.push(u64::from_le_bytes([
            byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7,
        ]));
    }

    let first = offsets
        .first()
        .copied()
        .ok_or(WebGraphError::NonCanonical("empty offset array"))?;
    if first != 0 {
        return Err(WebGraphError::BadOffset {
            node: 0,
            offset: first,
            bit_len,
        });
    }
    for (i, pair) in offsets.windows(2).enumerate() {
        if pair[1] < pair[0] || pair[1] > bit_len {
            return Err(WebGraphError::BadOffset {
                node: u64::try_from(i)
                    .map_err(|_| WebGraphError::SizeOverflow("offset node index"))?,
                offset: pair[1],
                bit_len,
            });
        }
    }
    let last = offsets
        .last()
        .copied()
        .ok_or(WebGraphError::NonCanonical("empty offset array"))?;
    if last != bit_len {
        return Err(WebGraphError::BadOffset {
            node: num_nodes,
            offset: last,
            bit_len,
        });
    }

    let bits_byte_len = usize::try_from(bit_len.div_ceil(8))
        .map_err(|_| WebGraphError::SizeOverflow("bit-stream length"))?;
    let bits_end = bits_offset
        .checked_add(bits_byte_len)
        .ok_or(WebGraphError::SizeOverflow("bit-stream end"))?;
    let canonical_body_end = checked_align_up(bits_end, 4, "blob padding")?;
    if canonical_body_end != body_end {
        return Err(WebGraphError::NonCanonical("blob length"));
    }
    let bits = buf
        .get(bits_offset..bits_end)
        .ok_or(WebGraphError::Truncated {
            need: bits_end,
            have: buf.len(),
        })?;
    let used_bits_in_last_byte = bit_len % 8;
    if used_bits_in_last_byte != 0 {
        let unused_bits = 8_u32
            - u32::try_from(used_bits_in_last_byte)
                .map_err(|_| WebGraphError::SizeOverflow("last-byte bit count"))?;
        let unused_mask = (1_u8 << unused_bits) - 1;
        if bits
            .last()
            .is_some_and(|last_byte| last_byte & unused_mask != 0)
        {
            return Err(WebGraphError::NonCanonical(
                "non-zero unused bit-stream bits",
            ));
        }
    }
    let padding = buf
        .get(bits_end..body_end)
        .ok_or(WebGraphError::Truncated {
            need: body_end,
            have: buf.len(),
        })?;
    if padding.iter().any(|byte| *byte != 0) {
        return Err(WebGraphError::NonCanonical("non-zero blob padding"));
    }
    Ok(ParsedWebGraphBlob {
        num_nodes,
        num_edges,
        bit_len,
        offsets,
        bits_offset,
        bits_byte_len,
    })
}

impl WebGraphCodec {
    /// Serializes to a self-describing, position-independent blob.
    ///
    /// Honours the Plan 2 zero-copy contract: a fixed 64-byte header with
    /// blob-relative `u64` section offsets, naturally-aligned arrays, and
    /// a trailing CRC32. See `BLOB_VERSION` for the version.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if lazy validation detects malformed data in
    /// a reopened codec.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WebGraphError> {
        self.validate()?;
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
        buf[offsets_pos..offsets_pos + 8].copy_from_slice(&offsets_offset.to_le_bytes());
        buf[offsets_pos + 8..offsets_pos + 16].copy_from_slice(&bits_offset.to_le_bytes());
        // The third and fourth u64s stay zero (reserved).

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        Ok(buf)
    }

    /// Opens a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`WebGraphError`] on bad magic, unsupported version,
    /// truncation, CRC mismatch, allocation failure, or malformed current
    /// framing. Compressed adjacency validation is lazy and cached by
    /// [`Self::validate`], [`Self::num_edges`], [`Self::out_degree`], and
    /// [`Self::successors`].
    pub fn from_bytes(buf: &[u8]) -> Result<Self, WebGraphError> {
        let parsed = parse_blob(buf)?;
        let bits_end = parsed
            .bits_offset
            .checked_add(parsed.bits_byte_len)
            .ok_or(WebGraphError::SizeOverflow("bit-stream end"))?;
        let encoded_bits =
            buf.get(parsed.bits_offset..bits_end)
                .ok_or(WebGraphError::Truncated {
                    need: bits_end,
                    have: buf.len(),
                })?;
        let mut bits = Vec::new();
        bits.try_reserve_exact(parsed.bits_byte_len).map_err(|_| {
            WebGraphError::AllocationFailed {
                field: "owned bit stream",
                count: parsed.bits_byte_len,
            }
        })?;
        bits.extend_from_slice(encoded_bits);

        Ok(Self {
            num_nodes: parsed.num_nodes,
            num_edges: parsed.num_edges,
            offsets: parsed.offsets,
            bits,
            bit_len: parsed.bit_len,
            validated: OnceLock::new(),
        })
    }

    /// Opens a blob shared via [`bytes::Bytes`] into an owned codec.
    ///
    /// This still copies the offsets array and bit stream into owned `Vec`s.
    /// For a true borrowing reader that holds `Bytes` slices and serves
    /// adjacency queries without copying, use [`WebGraphView::open`] instead.
    ///
    /// # Errors
    /// Same as [`Self::from_bytes`].
    pub fn from_bytes_shared(blob: bytes::Bytes) -> Result<Self, WebGraphError> {
        Self::from_bytes(&blob)
    }
}

/// A borrowing reader over a [`WebGraphCodec`] blob.
///
/// Holds a single `bytes::Bytes` and parsed header offsets; the
/// per-node bit-offset index and the bit stream are sliced from the
/// held bytes on demand. The owned `Vec<u64>` of offsets is decoded
/// once at open time (typically 8 bytes per node — small relative to
/// the bit stream).
#[derive(Debug, Clone)]
pub struct WebGraphView {
    blob: bytes::Bytes,
    num_nodes: u64,
    num_edges: u64,
    /// Decoded once at open time.
    offsets: Vec<u64>,
    /// Byte offset of the bit stream within `blob`.
    bits_offset: usize,
    /// Number of bytes in the bit stream (`bit_len.div_ceil(8)`).
    bits_byte_len: usize,
    /// Successful full-stream validation.
    validated: OnceLock<()>,
}

impl WebGraphView {
    /// Opens a blob produced by [`WebGraphCodec::to_bytes`].
    ///
    /// # Errors
    /// Returns [`WebGraphError`] on malformed current framing — the same open
    /// conditions as [`WebGraphCodec::from_bytes`]. Compressed adjacency
    /// validation is lazy and cached by the query and validation methods.
    pub fn open(blob: bytes::Bytes) -> Result<Self, WebGraphError> {
        let parsed = parse_blob(&blob)?;

        Ok(Self {
            blob,
            num_nodes: parsed.num_nodes,
            num_edges: parsed.num_edges,
            offsets: parsed.offsets,
            bits_offset: parsed.bits_offset,
            bits_byte_len: parsed.bits_byte_len,
            validated: OnceLock::new(),
        })
    }

    /// Number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    /// Number of edges, after validating the complete adjacency stream once.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if the blob contains malformed adjacency data
    /// or its declared edge count is false.
    pub fn num_edges(&self) -> Result<u64, WebGraphError> {
        self.validate()?;
        Ok(self.num_edges)
    }

    fn bits(&self) -> Result<&[u8], WebGraphError> {
        let end = self
            .bits_offset
            .checked_add(self.bits_byte_len)
            .ok_or(WebGraphError::SizeOverflow("bit-stream end"))?;
        self.blob
            .get(self.bits_offset..end)
            .ok_or(WebGraphError::Truncated {
                need: end,
                have: self.blob.len(),
            })
    }

    /// Validates every compressed adjacency block and the declared edge count.
    /// Successful validation is cached.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] on malformed gamma data, invalid successor
    /// coordinates, trailing block bits, or a false edge count.
    pub fn validate(&self) -> Result<(), WebGraphError> {
        if self.validated.get().is_some() {
            return Ok(());
        }
        validate_adjacency_stream(self.bits()?, &self.offsets, self.num_nodes, self.num_edges)?;
        let _ = self.validated.set(());
        Ok(())
    }

    /// Out-degree of `node`.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if lazy full-stream validation fails.
    pub fn out_degree(&self, node: u64) -> Result<u64, WebGraphError> {
        self.validate()?;
        let Some((start, end)) = adjacency_bounds(&self.offsets, self.num_nodes, node)? else {
            return Ok(0);
        };
        let mut reader = BitReader::new(self.bits()?, end);
        reader.seek(start);
        let degree = reader
            .read_gamma()
            .ok_or(WebGraphError::UnexpectedEnd(end))?
            - 1;
        if degree > self.num_nodes {
            return Err(WebGraphError::MalformedAdjacency {
                node,
                reason: "degree exceeds node count",
            });
        }
        Ok(degree)
    }

    /// Iterator over the successors of `node`, in ascending dst order.
    ///
    /// # Errors
    /// Returns [`WebGraphError`] if lazy full-stream validation fails.
    pub fn successors(&self, node: u64) -> Result<SuccessorIter<'_>, WebGraphError> {
        self.validate()?;
        let Some((start, end)) = adjacency_bounds(&self.offsets, self.num_nodes, node)? else {
            return Ok(SuccessorIter::empty());
        };
        let mut reader = BitReader::new(self.bits()?, end);
        reader.seek(start);
        let degree = reader
            .read_gamma()
            .ok_or(WebGraphError::UnexpectedEnd(end))?
            - 1;
        if degree > self.num_nodes {
            return Err(WebGraphError::MalformedAdjacency {
                node,
                reason: "degree exceeds node count",
            });
        }
        Ok(SuccessorIter::new(reader, node, degree))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refresh_crc(blob: &mut [u8]) {
        let body_end = blob.len() - 4;
        let crc = crc32fast::hash(&blob[..body_end]);
        blob[body_end..].copy_from_slice(&crc.to_le_bytes());
    }

    fn successors(codec: &WebGraphCodec, node: u64) -> Vec<u64> {
        codec.successors(node).unwrap().collect()
    }

    fn view_successors(view: &WebGraphView, node: u64) -> Vec<u64> {
        view.successors(node).unwrap().collect()
    }

    #[test]
    fn build_empty_graph_has_no_edges() {
        let codec = WebGraphBuilder::new(0).build().unwrap();
        assert_eq!(codec.num_nodes(), 0);
        assert_eq!(codec.num_edges().unwrap(), 0);
    }

    #[test]
    fn build_isolated_nodes_have_zero_degree() {
        let codec = WebGraphBuilder::new(5).build().unwrap();
        assert_eq!(codec.num_nodes(), 5);
        assert_eq!(codec.num_edges().unwrap(), 0);
        for u in 0u64..5 {
            assert_eq!(codec.out_degree(u).unwrap(), 0);
        }
    }

    #[test]
    fn build_records_edge_count_after_deduplication() {
        let mut b = WebGraphBuilder::new(4);
        b.add_edge(0, 1).unwrap();
        b.add_edge(0, 2).unwrap();
        b.add_edge(0, 1).unwrap(); // duplicate
        b.add_edge(1, 2).unwrap();
        let codec = b.build().unwrap();
        assert_eq!(codec.num_edges().unwrap(), 3); // duplicate removed
        assert_eq!(codec.out_degree(0).unwrap(), 2);
        assert_eq!(codec.out_degree(1).unwrap(), 1);
        assert_eq!(codec.out_degree(2).unwrap(), 0);
        assert_eq!(codec.out_degree(3).unwrap(), 0);
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
    fn hostile_builder_size_returns_error_without_panicking() {
        let result = std::panic::catch_unwind(|| WebGraphBuilder::new(u64::MAX).build());
        assert!(result.is_ok(), "builder panicked on hostile node count");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn successors_match_input_for_simple_graph() {
        let mut b = WebGraphBuilder::new(6);
        // Node 0 -> {1, 3, 5}, node 2 -> {0, 4}, node 5 -> {5} (self-loop).
        for (s, d) in [(0u64, 1), (0, 3), (0, 5), (2, 0), (2, 4), (5, 5)] {
            b.add_edge(s, d).unwrap();
        }
        let codec = b.build().unwrap();
        assert_eq!(successors(&codec, 0), vec![1, 3, 5]);
        assert_eq!(successors(&codec, 1), Vec::<u64>::new());
        assert_eq!(successors(&codec, 2), vec![0, 4]);
        assert_eq!(successors(&codec, 5), vec![5]);
    }

    #[test]
    fn successors_handles_dst_less_than_src() {
        // First gap is signed; verify dst < src round-trips correctly.
        let mut b = WebGraphBuilder::new(10);
        b.add_edge(7, 0).unwrap();
        b.add_edge(7, 1).unwrap();
        b.add_edge(7, 9).unwrap();
        let codec = b.build().unwrap();
        assert_eq!(successors(&codec, 7), vec![0, 1, 9]);
    }

    #[test]
    fn successors_for_out_of_range_node_is_empty() {
        let codec = WebGraphBuilder::new(3).build().unwrap();
        assert_eq!(successors(&codec, 99), Vec::<u64>::new());
    }

    #[test]
    fn blob_round_trip_preserves_adjacency() {
        let mut b = WebGraphBuilder::new(20);
        let edges = [
            (0u64, 1),
            (0, 5),
            (0, 17),
            (1, 0),
            (1, 2),
            (3, 3),
            (5, 18),
            (5, 19),
            (10, 0),
            (10, 5),
            (10, 11),
            (10, 12),
            (19, 0),
        ];
        for &(s, d) in &edges {
            b.add_edge(s, d).unwrap();
        }
        let codec = b.build().unwrap();
        let blob = codec.to_bytes().unwrap();

        assert_eq!(&blob[0..4], b"GWBG");
        assert_eq!(blob[4], 1);

        let reopened = WebGraphCodec::from_bytes(&blob).expect("from_bytes");
        assert_eq!(reopened.num_nodes(), codec.num_nodes());
        assert_eq!(reopened.num_edges().unwrap(), codec.num_edges().unwrap());
        for u in 0..codec.num_nodes() {
            let original = successors(&codec, u);
            let reopened_succ = successors(&reopened, u);
            assert_eq!(reopened_succ, original, "successors of {u} mismatched");
        }
    }

    #[test]
    fn blob_rejects_bad_magic_and_crc() {
        let mut b = WebGraphBuilder::new(3);
        b.add_edge(0, 1).unwrap();
        b.add_edge(1, 2).unwrap();
        let mut blob = b.build().unwrap().to_bytes().unwrap();

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
    fn blob_rejects_non_current_version_before_crc() {
        let mut blob = WebGraphBuilder::new(1).build().unwrap().to_bytes().unwrap();
        blob[4] = BLOB_VERSION + 1;

        assert!(matches!(
            WebGraphCodec::from_bytes(&blob),
            Err(WebGraphError::BadVersion(2))
        ));
        assert!(matches!(
            WebGraphView::open(bytes::Bytes::from(blob)),
            Err(WebGraphError::BadVersion(2))
        ));
    }

    #[test]
    fn blob_rejects_false_edge_count() {
        let mut blob = WebGraphBuilder::new(1).build().unwrap().to_bytes().unwrap();
        blob[16..24].copy_from_slice(&1_u64.to_le_bytes());
        refresh_crc(&mut blob);

        let owned = WebGraphCodec::from_bytes(&blob).unwrap();
        assert!(matches!(
            owned.validate(),
            Err(WebGraphError::EdgeCountMismatch {
                declared: 1,
                decoded: 0
            })
        ));
        let view = WebGraphView::open(bytes::Bytes::from(blob)).unwrap();
        assert!(matches!(
            view.validate(),
            Err(WebGraphError::EdgeCountMismatch {
                declared: 1,
                decoded: 0
            })
        ));
    }

    #[test]
    fn hostile_node_count_returns_error_without_panicking() {
        let mut blob = WebGraphBuilder::new(1).build().unwrap().to_bytes().unwrap();
        blob[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        refresh_crc(&mut blob);

        let owned = std::panic::catch_unwind(|| WebGraphCodec::from_bytes(&blob));
        assert!(
            owned.is_ok(),
            "owned decoder panicked on hostile node count"
        );
        assert!(owned.unwrap().is_err());

        let view =
            std::panic::catch_unwind(|| WebGraphView::open(bytes::Bytes::copy_from_slice(&blob)));
        assert!(view.is_ok(), "view decoder panicked on hostile node count");
        assert!(view.unwrap().is_err());
    }

    #[test]
    fn malformed_gamma_stream_surfaces_a_structured_access_error() {
        let mut blob = WebGraphBuilder::new(1).build().unwrap().to_bytes().unwrap();
        let bits_offset = usize::try_from(u64::from_le_bytes(
            blob[40..48].try_into().expect("bits offset field"),
        ))
        .expect("test blob offset fits usize");
        blob[bits_offset] = 0;
        refresh_crc(&mut blob);

        let owned = WebGraphCodec::from_bytes(&blob).unwrap();
        assert!(matches!(
            owned.out_degree(0),
            Err(WebGraphError::UnexpectedEnd(_))
        ));
        assert!(matches!(
            owned.validate(),
            Err(WebGraphError::UnexpectedEnd(_))
        ));

        let view = WebGraphView::open(bytes::Bytes::from(blob)).unwrap();
        assert!(matches!(
            view.out_degree(0),
            Err(WebGraphError::UnexpectedEnd(_))
        ));
        assert!(matches!(
            view.validate(),
            Err(WebGraphError::UnexpectedEnd(_))
        ));
    }

    #[test]
    fn blob_from_bytes_shared_round_trip() {
        let mut b = WebGraphBuilder::new(3);
        b.add_edge(0, 2).unwrap();
        let blob = bytes::Bytes::from(b.build().unwrap().to_bytes().unwrap());
        let reopened = WebGraphCodec::from_bytes_shared(blob).expect("from_bytes_shared");
        assert_eq!(successors(&reopened, 0), vec![2]);
    }

    #[test]
    fn view_successors_matches_owned() {
        let mut b = WebGraphBuilder::new(15);
        let edges = [
            (0u64, 1),
            (0, 5),
            (0, 14),
            (3, 0),
            (3, 3),
            (10, 7),
            (10, 8),
            (10, 9),
            (14, 0),
        ];
        for &(s, d) in &edges {
            b.add_edge(s, d).unwrap();
        }
        let owned = b.build().unwrap();
        let blob = bytes::Bytes::from(owned.to_bytes().unwrap());
        let view = WebGraphView::open(blob).expect("open");

        assert_eq!(view.num_nodes(), owned.num_nodes());
        assert_eq!(view.num_edges().unwrap(), owned.num_edges().unwrap());
        for u in 0..owned.num_nodes() {
            let owned_succ = successors(&owned, u);
            let view_succ = view_successors(&view, u);
            assert_eq!(view_succ, owned_succ, "successors of {u} mismatched");
        }
    }

    #[test]
    fn view_rejects_bad_magic() {
        let owned = WebGraphBuilder::new(3).build().unwrap();
        let mut bad = owned.to_bytes().unwrap();
        bad[0] = b'X';
        assert!(matches!(
            WebGraphView::open(bytes::Bytes::from(bad)),
            Err(WebGraphError::BadMagic)
        ));
    }
}
