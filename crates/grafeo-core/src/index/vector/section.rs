//! Vector Store section serializer for the `.grafeo` container format.
//!
//! Serializes exact HNSW/quantized-index state for all vector indexes.
//!
//! Persisting the topology eliminates the O(N log N) HNSW rebuild on
//! database open.
//!
//! Only the exact v4 `GVST` format is supported. It preserves full HNSW
//! configuration, topology, delete state and deterministic PRNG continuation,
//! plus quantized vectors, codes, quantizers and training state. Predecessor
//! images are rejected at the outer boundary before payload decoding.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};

use crate::graph::lpg::{PhysicalIndexFamily, PhysicalIndexKey};
use grafeo_common::types::{GraphPath, MAX_GRAPH_PATH_COMPONENTS, MAX_WORLD_GRAPH_NAME_BYTES};

use super::hnsw::{HNSW_RNG_SPLITMIX64, HnswExactState};
use super::quantization::{ProductQuantizerExactState, ScalarQuantizerExactState};
use super::quantized_hnsw::QuantizedExactState;
use super::{
    DistanceMetric, HnswConfig, PreparedVectorExactState, QuantizationType, VectorExactState,
    VectorIndexKind, VectorIndexView,
};

/// Current exact vector store section format version.
const VECTOR_SECTION_VERSION: u8 = 4;
const VECTOR_MAGIC: &[u8; 4] = b"GVST";

/// Current header size (magic + version + reserved + payload length).
const V4_HEADER_SIZE: usize = 16;

// Preflight the complete allocation-bearing wire before serde constructs DTOs.
struct BincodeWirePreflight<'data> {
    data: &'data [u8],
    position: usize,
    predicted_heap: u64,
    format: &'static str,
}

impl<'data> BincodeWirePreflight<'data> {
    fn new(data: &'data [u8], format: &'static str) -> Self {
        Self {
            data,
            position: 0,
            predicted_heap: 0,
            format,
        }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    fn read_byte(&mut self, description: &str) -> std::result::Result<u8, String> {
        let byte =
            self.data.get(self.position).copied().ok_or_else(|| {
                format!("Vector Store {} {description} is truncated", self.format)
            })?;
        self.position += 1;
        Ok(byte)
    }

    fn read_fixed<const N: usize>(
        &mut self,
        description: &str,
    ) -> std::result::Result<[u8; N], String> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| format!("Vector Store {} {description} range overflows", self.format))?;
        let bytes = self
            .data
            .get(self.position..end)
            .ok_or_else(|| format!("Vector Store {} {description} is truncated", self.format))?;
        self.position = end;
        bytes.try_into().map_err(|_| {
            format!(
                "Vector Store {} {description} has invalid width",
                self.format
            )
        })
    }

    /// Reads bincode standard's little-endian variable-width u64 encoding.
    fn read_u64(&mut self, description: &str) -> std::result::Result<u64, String> {
        match self.read_byte(description)? {
            value @ 0..=250 => Ok(u64::from(value)),
            251 => Ok(u64::from(u16::from_le_bytes(self.read_fixed(description)?))),
            252 => Ok(u64::from(u32::from_le_bytes(self.read_fixed(description)?))),
            253 => Ok(u64::from_le_bytes(self.read_fixed(description)?)),
            254 => Err(format!(
                "Vector Store {} {description} uses an out-of-range u128 encoding",
                self.format
            )),
            255 => Err(format!(
                "Vector Store {} {description} uses a reserved integer encoding",
                self.format
            )),
        }
    }

    fn read_usize(&mut self, description: &str) -> std::result::Result<usize, String> {
        usize::try_from(self.read_u64(description)?).map_err(|_| {
            format!(
                "Vector Store {} {description} exceeds this host",
                self.format
            )
        })
    }

    fn charge_heap(
        &mut self,
        count: usize,
        element_bytes: usize,
        description: &str,
    ) -> std::result::Result<(), String> {
        let bytes = u64::try_from(count)
            .ok()
            .and_then(|count| {
                u64::try_from(element_bytes)
                    .ok()
                    .and_then(|element| count.checked_mul(element))
            })
            .ok_or_else(|| {
                format!(
                    "Vector Store {} {description} allocation overflows",
                    self.format
                )
            })?;
        self.predicted_heap = self
            .predicted_heap
            .checked_add(bytes)
            .ok_or_else(|| format!("Vector Store {} decode heap overflows", self.format))?;
        Ok(())
    }

    fn read_sequence_len(
        &mut self,
        element_bytes: usize,
        description: &str,
    ) -> std::result::Result<usize, String> {
        let count = self.read_usize(description)?;
        // Every supported sequence element begins with at least one encoded byte. This
        // cheap proof rejects hostile huge counts before either looping or
        // allowing serde to forward the count into Vec::with_capacity.
        if count > self.remaining() {
            return Err(format!(
                "Vector Store {} {description} count {count} exceeds the {} remaining bytes",
                self.format,
                self.remaining()
            ));
        }
        self.charge_heap(count, element_bytes, description)?;
        Ok(count)
    }

    fn read_physical_key(&mut self) -> std::result::Result<PhysicalIndexKey, String> {
        const MAX_PATH_BYTES: usize =
            4 + MAX_GRAPH_PATH_COMPONENTS * (4 + MAX_WORLD_GRAPH_NAME_BYTES);
        let length = self.read_sequence_len(1, "index key graph path")?;
        if !(4..=MAX_PATH_BYTES).contains(&length) {
            return Err("index key graph path length exceeds its bounds".to_string());
        }
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| "index key graph path range overflows".to_string())?;
        let bytes = self
            .data
            .get(self.position..end)
            .ok_or_else(|| "index key graph path is truncated".to_string())?;
        let graph = GraphPath::from_bytes(bytes, MAX_PATH_BYTES)
            .map_err(|error| format!("invalid index key graph path: {error}"))?;
        self.position = end;
        let family = match self.read_byte("index key family")? {
            0 => PhysicalIndexFamily::Property,
            1 => PhysicalIndexFamily::Text,
            2 => PhysicalIndexFamily::Vector,
            _ => return Err("invalid index key family".to_string()),
        };
        let label = match self.read_byte("index key label tag")? {
            0 => None,
            1 => Some(self.read_string("index key label")?.to_owned()),
            _ => return Err("invalid index key label tag".to_string()),
        };
        let property = self.read_string("index key property")?.to_owned();
        PhysicalIndexKey::new(graph, family, label, property)
            .map_err(|error| format!("invalid index key: {error}"))
    }

    fn read_string(&mut self, description: &str) -> std::result::Result<&'data str, String> {
        let length = self.read_usize(description)?;
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| format!("Vector Store {} {description} range overflows", self.format))?;
        let bytes = self
            .data
            .get(self.position..end)
            .ok_or_else(|| format!("Vector Store {} {description} is truncated", self.format))?;
        self.charge_heap(length, 1, description)?;
        self.position = end;
        std::str::from_utf8(bytes)
            .map_err(|_| format!("Vector Store {} {description} is not UTF-8", self.format))
    }

    fn read_node_id(&mut self, description: &str) -> std::result::Result<NodeId, String> {
        let id = self.read_u64(description)?;
        if id == NodeId::INVALID.as_u64() {
            return Err(format!(
                "Vector Store {} {description} contains the invalid NodeId sentinel",
                self.format
            ));
        }
        Ok(NodeId::new(id))
    }

    fn read_bool(&mut self, description: &str) -> std::result::Result<(), String> {
        let value = self.read_byte(description)?;
        if value > 1 {
            return Err(format!(
                "Vector Store {} {description} has invalid boolean tag {value}",
                self.format
            ));
        }
        Ok(())
    }

    fn read_option_tag(&mut self, description: &str) -> std::result::Result<bool, String> {
        match self.read_byte(description)? {
            0 => Ok(false),
            1 => Ok(true),
            tag => Err(format!(
                "Vector Store {} {description} has invalid option tag {tag}",
                self.format
            )),
        }
    }

    fn skip_fixed_sequence(
        &mut self,
        wire_width: usize,
        heap_width: usize,
        description: &str,
    ) -> std::result::Result<(), String> {
        let count = self.read_sequence_len(heap_width, description)?;
        let byte_count = count.checked_mul(wire_width).ok_or_else(|| {
            format!(
                "Vector Store {} {description} byte range overflows",
                self.format
            )
        })?;
        let end = self.position.checked_add(byte_count).ok_or_else(|| {
            format!(
                "Vector Store {} {description} byte range overflows",
                self.format
            )
        })?;
        if end > self.data.len() {
            return Err(format!(
                "Vector Store {} {description} needs {byte_count} bytes but only {} remain",
                self.format,
                self.remaining()
            ));
        }
        self.position = end;
        Ok(())
    }

    fn finish_exact(self) -> std::result::Result<(), String> {
        if self.position != self.data.len() {
            return Err(format!(
                "Vector Store {} payload has {} trailing bytes",
                self.format,
                self.data.len() - self.position
            ));
        }
        Ok(())
    }
}

fn preflight_v4_hnsw_config(
    wire: &mut BincodeWirePreflight<'_>,
) -> std::result::Result<(), String> {
    let _dimensions = wire.read_u64("HNSW dimensions")?;
    let metric = wire.read_u64("HNSW distance metric")?;
    if metric > 3 {
        return Err(format!(
            "Vector Store v4 distance metric variant {metric} is unsupported"
        ));
    }
    let _m = wire.read_u64("HNSW m")?;
    let _m_max = wire.read_u64("HNSW m_max")?;
    let _ef_construction = wire.read_u64("HNSW ef_construction")?;
    let _ef = wire.read_u64("HNSW ef")?;
    let _ml_bits = wire.read_u64("HNSW ml bits")?;
    let _alpha_bits = wire.read_u64("HNSW alpha bits")?;
    if wire.read_option_tag("HNSW max_elements")? {
        let _max_elements = wire.read_u64("HNSW max_elements value")?;
    }
    Ok(())
}

fn preflight_v4_hnsw(
    wire: &mut BincodeWirePreflight<'_>,
    visit_node: &mut impl FnMut(NodeId),
) -> std::result::Result<(), String> {
    preflight_v4_hnsw_config(wire)?;
    if wire.read_option_tag("HNSW entry point")? {
        visit_node(wire.read_node_id("HNSW entry point value")?);
    }
    let _max_level = wire.read_u64("HNSW max level")?;
    let node_count = wire.read_sequence_len(
        std::mem::size_of::<(NodeId, Vec<Vec<NodeId>>)>(),
        "HNSW node topology",
    )?;
    for _ in 0..node_count {
        visit_node(wire.read_node_id("HNSW node id")?);
        let level_count =
            wire.read_sequence_len(std::mem::size_of::<Vec<NodeId>>(), "HNSW node levels")?;
        for _ in 0..level_count {
            let neighbor_count =
                wire.read_sequence_len(std::mem::size_of::<NodeId>(), "HNSW neighbor list")?;
            for _ in 0..neighbor_count {
                visit_node(wire.read_node_id("HNSW neighbor id")?);
            }
        }
    }
    let deleted_count = wire.read_sequence_len(std::mem::size_of::<NodeId>(), "HNSW delete set")?;
    for _ in 0..deleted_count {
        visit_node(wire.read_node_id("HNSW deleted node id")?);
    }
    let _rng_algorithm = wire.read_byte("HNSW RNG algorithm")?;
    let _rng_state = wire.read_u64("HNSW RNG state")?;
    Ok(())
}

fn preflight_v4_quantized(
    wire: &mut BincodeWirePreflight<'_>,
    visit_node: &mut impl FnMut(NodeId),
) -> std::result::Result<(), String> {
    preflight_v4_hnsw(wire, visit_node)?;
    let quantization = wire.read_u64("quantization kind")?;
    match quantization {
        0..=2 => {}
        3 => {
            let _num_subvectors = wire.read_u64("PQ subvector count")?;
        }
        variant => {
            return Err(format!(
                "Vector Store v4 quantization variant {variant} is unsupported"
            ));
        }
    }
    if wire.read_option_tag("scalar quantizer")? {
        wire.skip_fixed_sequence(4, std::mem::size_of::<f32>(), "scalar minimums")?;
        wire.skip_fixed_sequence(4, std::mem::size_of::<f32>(), "scalar scales")?;
        wire.skip_fixed_sequence(4, std::mem::size_of::<f32>(), "scalar inverse scales")?;
        let _dimensions = wire.read_u64("scalar quantizer dimensions")?;
    }
    if wire.read_option_tag("product quantizer")? {
        let _num_subvectors = wire.read_u64("PQ subvector count")?;
        let _num_centroids = wire.read_u64("PQ centroid count")?;
        let _subvector_dim = wire.read_u64("PQ subvector dimensions")?;
        let _dimensions = wire.read_u64("PQ dimensions")?;
        wire.skip_fixed_sequence(4, std::mem::size_of::<f32>(), "PQ centroids")?;
    }

    let vector_count =
        wire.read_sequence_len(std::mem::size_of::<(NodeId, Vec<f32>)>(), "full vector map")?;
    for _ in 0..vector_count {
        visit_node(wire.read_node_id("full vector node id")?);
        wire.skip_fixed_sequence(4, std::mem::size_of::<f32>(), "full vector")?;
    }
    let scalar_count = wire.read_sequence_len(
        std::mem::size_of::<(NodeId, Vec<u8>)>(),
        "scalar vector map",
    )?;
    for _ in 0..scalar_count {
        visit_node(wire.read_node_id("scalar vector node id")?);
        wire.skip_fixed_sequence(1, std::mem::size_of::<u8>(), "scalar code")?;
    }
    let binary_count = wire.read_sequence_len(
        std::mem::size_of::<(NodeId, Vec<u64>)>(),
        "binary vector map",
    )?;
    for _ in 0..binary_count {
        visit_node(wire.read_node_id("binary vector node id")?);
        let word_count = wire.read_sequence_len(std::mem::size_of::<u64>(), "binary code words")?;
        for _ in 0..word_count {
            let _word = wire.read_u64("binary code word")?;
        }
    }
    let product_count =
        wire.read_sequence_len(std::mem::size_of::<(NodeId, Vec<u8>)>(), "product code map")?;
    for _ in 0..product_count {
        visit_node(wire.read_node_id("product code node id")?);
        wire.skip_fixed_sequence(1, std::mem::size_of::<u8>(), "product code")?;
    }
    wire.read_bool("rescore flag")?;
    let _rescore_factor = wire.read_u64("rescore factor")?;
    let _training_threshold = wire.read_u64("training threshold")?;
    let sample_count = wire.read_sequence_len(
        std::mem::size_of::<Vec<f32>>(),
        "quantizer training samples",
    )?;
    for _ in 0..sample_count {
        wire.skip_fixed_sequence(4, std::mem::size_of::<f32>(), "training sample")?;
    }
    wire.read_bool("quantizer trained flag")?;
    Ok(())
}

fn preflight_v4_payload(
    data: &[u8],
    visit: impl FnMut(&PhysicalIndexKey, bool, Range<usize>) -> std::result::Result<(), String>,
) -> std::result::Result<(), String> {
    preflight_v4_payload_with_nodes(data, visit, |_, _| {})
}

fn preflight_v4_payload_with_nodes(
    data: &[u8],
    mut visit: impl FnMut(&PhysicalIndexKey, bool, Range<usize>) -> std::result::Result<(), String>,
    mut visit_node: impl FnMut(&PhysicalIndexKey, NodeId),
) -> std::result::Result<(), String> {
    let mut wire = BincodeWirePreflight::new(data, "v4");
    let index_count =
        wire.read_sequence_len(std::mem::size_of::<IndexSnapshotV4>(), "index image list")?;
    let mut previous = None;
    for _ in 0..index_count {
        let start = wire.position;
        let key = wire.read_physical_key()?;
        if key.family() != PhysicalIndexFamily::Vector {
            return Err(format!(
                "Vector section has invalid index key family {key:?}"
            ));
        }
        if previous.as_ref().is_some_and(|prior| prior >= &key) {
            return Err(format!(
                "duplicate vector index key or non-canonical key order at {key:?}"
            ));
        }
        let mut owner_node = |id| visit_node(&key, id);
        let quantized = match wire.read_u64("concrete index kind")? {
            0 => {
                preflight_v4_hnsw(&mut wire, &mut owner_node)?;
                false
            }
            1 => {
                preflight_v4_quantized(&mut wire, &mut owner_node)?;
                true
            }
            variant => {
                return Err(format!(
                    "Vector Store v4 concrete index variant {variant} is unsupported"
                ));
            }
        };
        visit(&key, quantized, start..wire.position)?;
        previous = Some(key);
    }
    wire.finish_exact()
}

mod wal;

// ── v4 exact DTOs ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorStoreSnapshotV4 {
    indexes: Vec<IndexSnapshotV4>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexSnapshotV4 {
    key: PhysicalIndexKey,
    state: VectorStateV4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum VectorStateV4 {
    Hnsw(HnswStateV4),
    Quantized(QuantizedStateV4),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswConfigV4 {
    dimensions: u64,
    metric: DistanceMetric,
    m: u64,
    m_max: u64,
    ef_construction: u64,
    ef: u64,
    ml_bits: u64,
    alpha_bits: u32,
    max_elements: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswRngV4 {
    algorithm: u8,
    state: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswStateV4 {
    config: HnswConfigV4,
    entry_point: Option<NodeId>,
    max_level: u64,
    nodes: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    deleted: Vec<NodeId>,
    rng: HnswRngV4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum QuantizationV4 {
    None,
    Scalar,
    Binary,
    Product { num_subvectors: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarQuantizerV4 {
    min: Vec<f32>,
    scale: Vec<f32>,
    inv_scale: Vec<f32>,
    dimensions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProductQuantizerV4 {
    num_subvectors: u64,
    num_centroids: u64,
    subvector_dim: u64,
    dimensions: u64,
    centroids: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizedStateV4 {
    hnsw: HnswStateV4,
    quantization_type: QuantizationV4,
    scalar_quantizer: Option<ScalarQuantizerV4>,
    product_quantizer: Option<ProductQuantizerV4>,
    vectors: Vec<(NodeId, Vec<f32>)>,
    scalar_vectors: Vec<(NodeId, Vec<u8>)>,
    binary_vectors: Vec<(NodeId, Vec<u64>)>,
    product_codes: Vec<(NodeId, Vec<u8>)>,
    rescore: bool,
    rescore_factor: u64,
    training_threshold: u64,
    training_samples: Vec<Vec<f32>>,
    quantizer_trained: bool,
}

// ── Section implementation ──────────────────────────────────────────

/// Vector Store section for the `.grafeo` container.
///
/// Wraps a collection of read-only index views and serializes their exact
/// vector state for persistence.
///
/// v4 restore is intentionally unavailable from [`Self::new`] and
/// [`Self::from_views`]. Readers do not participate in a section-wide atomic
/// publication protocol, so authoritative exact state may only be installed
/// through [`Self::for_unpublished_recovery`] or
/// [`Self::for_unpublished_recovery_views`] while every target is private.
pub struct VectorStoreSection {
    /// Vector indexes qualified by the captured graph path and physical family.
    indexes: Vec<(PhysicalIndexKey, VectorIndexView)>,
    dirty: AtomicBool,
    restore_mode: RestoreMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RestoreMode {
    SnapshotOnly,
    UnpublishedRecovery,
}

impl VectorStoreSection {
    /// Create a new Vector Store section from the current indexes.
    pub fn new(indexes: Vec<(PhysicalIndexKey, Arc<VectorIndexKind>)>) -> Self {
        Self::from_views(
            indexes
                .into_iter()
                .map(|(key, index)| (key, VectorIndexView::new(index)))
                .collect(),
        )
    }

    /// Creates a section from the capability-reduced handles returned by an
    /// LPG store.
    pub fn from_views(indexes: Vec<(PhysicalIndexKey, VectorIndexView)>) -> Self {
        Self {
            indexes,
            dirty: AtomicBool::new(false),
            restore_mode: RestoreMode::SnapshotOnly,
        }
    }

    /// Creates a section whose targets are private to database recovery.
    ///
    /// Exact installation is failure-before-mutation for writers, but readers
    /// do not take one section-wide publication lock. The caller must therefore
    /// keep every supplied index unreachable by readers until `deserialize`
    /// succeeds and the surrounding database state publishes them together.
    #[must_use]
    pub fn for_unpublished_recovery(
        indexes: Vec<(PhysicalIndexKey, Arc<VectorIndexKind>)>,
    ) -> Self {
        Self::for_unpublished_recovery_views(
            indexes
                .into_iter()
                .map(|(key, index)| (key, VectorIndexView::new(index)))
                .collect(),
        )
    }

    /// Creates an unpublished-recovery section from capability-reduced views.
    ///
    /// See [`Self::for_unpublished_recovery`] for the publication contract.
    #[must_use]
    pub fn for_unpublished_recovery_views(
        indexes: Vec<(PhysicalIndexKey, VectorIndexView)>,
    ) -> Self {
        Self {
            indexes,
            dirty: AtomicBool::new(false),
            restore_mode: RestoreMode::UnpublishedRecovery,
        }
    }

    /// Validates the current envelope and its complete allocation-bearing wire.
    ///
    /// # Errors
    /// Returns an error for missing bytes, unsupported magic/version, malformed
    /// lengths, flags or payload. Target-dependent configuration and derived-code
    /// validation remains part of unpublished recovery.
    pub fn validate_payload(data: &[u8]) -> Result<()> {
        let payload = preflight_v4_envelope(data)?;
        preflight_v4_payload(payload, |_, _, _| Ok(())).map_err(Error::Serialization)
    }

    /// Returns the exact sorted physical key set from a bounded current image.
    ///
    /// # Errors
    /// Rejects unsupported or malformed wire data and duplicate/wrong-family keys.
    pub fn payload_keys(data: &[u8]) -> Result<Vec<PhysicalIndexKey>> {
        let mut keys = Vec::new();
        preflight_v4_payload(preflight_v4_envelope(data)?, |key, _, _| {
            keys.try_reserve(1)
                .map_err(|_| "cannot allocate Vector Store payload keys".to_string())?;
            keys.push(key.clone());
            Ok(())
        })
        .map_err(Error::Serialization)?;
        Ok(keys)
    }

    /// Returns sorted keys whose exact image uses the concrete quantized index.
    ///
    /// A quantized index with `QuantizationType::None` is distinct from a plain
    /// HNSW index even though both have the same logical catalog quantization.
    /// Recovery uses this metadata to construct private targets; configuration
    /// and exact-state validation still run during unpublished deserialization.
    /// The bounded wire preflight collects only keys, not vector/topology DTOs.
    ///
    /// # Errors
    /// Rejects unsupported or malformed wire data and duplicate/wrong-family keys.
    pub fn payload_quantized_keys(data: &[u8]) -> Result<Vec<PhysicalIndexKey>> {
        let mut keys = Vec::new();
        preflight_v4_payload(preflight_v4_envelope(data)?, |key, quantized, _| {
            if quantized {
                keys.try_reserve(1)
                    .map_err(|_| "cannot allocate Vector Store quantized keys".to_string())?;
                keys.push(key.clone());
            }
            Ok(())
        })
        .map_err(Error::Serialization)?;
        Ok(keys)
    }

    /// Tests every encoded node reference against its exact physical owner key.
    ///
    /// This includes HNSW entry points, topology nodes/neighbors and deleted
    /// nodes, plus all full/scalar/binary/product vector-map keys. A match never
    /// skips the remaining wire validation: `Ok(true)` is returned only after
    /// the complete current envelope and payload have passed bounded preflight.
    /// The scan does not decode topology/vector DTOs or rebuild an index.
    ///
    /// # Errors
    /// Rejects malformed or unsupported current wire, including late corruption
    /// after a match, noncanonical owner keys and trailing bytes. This validates
    /// wire structure, not the semantic consistency required for exact restore.
    pub fn payload_references_nodes(
        data: &[u8],
        mut is_selected: impl FnMut(&PhysicalIndexKey, NodeId) -> bool,
    ) -> Result<bool> {
        let mut matched = false;
        preflight_v4_payload_with_nodes(
            preflight_v4_envelope(data)?,
            |_, _, _| Ok(()),
            |key, id| matched |= is_selected(key, id),
        )
        .map_err(Error::Serialization)?;
        Ok(matched)
    }

    /// Returns canonical owner keys and their exact encoded entry ranges.
    ///
    /// Ranges are relative to the complete input, include the physical key,
    /// and exclude the enclosing header and index count. The complete current
    /// wire is preflighted without decoding vector or topology DTOs. Exact
    /// target configuration and topology validation remain part of recovery.
    ///
    /// # Errors
    /// Rejects unsupported or malformed input, duplicate/wrong-family keys,
    /// noncanonical key order, range overflow, or metadata allocation failure.
    pub fn payload_entry_ranges(data: &[u8]) -> Result<Vec<(PhysicalIndexKey, Range<usize>)>> {
        let mut entries = Vec::new();
        preflight_v4_payload(preflight_v4_envelope(data)?, |key, _, range| {
            let start = V4_HEADER_SIZE
                .checked_add(range.start)
                .ok_or_else(|| "Vector Store entry start overflows".to_string())?;
            let end = V4_HEADER_SIZE
                .checked_add(range.end)
                .ok_or_else(|| "Vector Store entry end overflows".to_string())?;
            entries
                .try_reserve(1)
                .map_err(|_| "cannot allocate Vector Store entry ranges".to_string())?;
            entries.push((key.clone(), start..end));
            Ok(())
        })
        .map_err(Error::Serialization)?;
        Ok(entries)
    }

    /// Copies selected complete owner entries into a current Vector4 envelope.
    ///
    /// The requested keys must be sorted, unique and present. Selection changes
    /// only the envelope and index count; encoded topology, RNG continuation,
    /// quantization/training state and configuration bytes are copied unchanged.
    /// Empty selection produces a valid empty current image. The source is
    /// fully validated even when no entries are selected.
    ///
    /// # Errors
    /// Rejects invalid current input, unsorted/duplicate/unknown selected keys,
    /// checked size overflow, or allocation failure.
    pub fn select_payload_keys(data: &[u8], keys: &[PhysicalIndexKey]) -> Result<Vec<u8>> {
        let entries = Self::payload_entry_ranges(data)?;
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::Serialization(
                "Vector Store selected keys must be sorted and unique".into(),
            ));
        }
        if keys.len() > entries.len() {
            return Err(Error::Serialization(
                "Vector Store selected key count exceeds the source".into(),
            ));
        }
        let mut selected = Vec::new();
        selected.try_reserve_exact(keys.len()).map_err(|_| {
            Error::Serialization("cannot allocate Vector Store selected ranges".into())
        })?;
        let mut entry_bytes = 0usize;
        for key in keys {
            let position = entries
                .binary_search_by(|(known, _)| known.cmp(key))
                .map_err(|_| {
                    Error::Serialization(format!("Vector Store selected key is absent: {key:?}"))
                })?;
            let (_, range) = entries.get(position).ok_or_else(|| {
                Error::Serialization("Vector Store selected entry position is absent".into())
            })?;
            entry_bytes = entry_bytes.checked_add(range.len()).ok_or_else(|| {
                Error::Serialization("Vector Store selected entry size overflows".into())
            })?;
            selected.push(range.clone());
        }
        let count = u64::try_from(keys.len()).map_err(|_| {
            Error::Serialization("Vector Store selected index count overflows".into())
        })?;
        let mut count_buffer = [0u8; 9];
        let count_len = bincode::serde::encode_into_slice(
            count,
            &mut count_buffer,
            bincode::config::standard(),
        )
        .map_err(|error| {
            Error::Serialization(format!(
                "cannot encode Vector Store selected count: {error}"
            ))
        })?;
        let count_bytes = count_buffer.get(..count_len).ok_or_else(|| {
            Error::Serialization("Vector Store selected count width is invalid".into())
        })?;
        let payload_len = count_len.checked_add(entry_bytes).ok_or_else(|| {
            Error::Serialization("Vector Store selected payload size overflows".into())
        })?;
        let capacity = V4_HEADER_SIZE.checked_add(payload_len).ok_or_else(|| {
            Error::Serialization("Vector Store selected envelope size overflows".into())
        })?;
        let wire_len = u64::try_from(payload_len).map_err(|_| {
            Error::Serialization("Vector Store selected payload length overflows".into())
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            Error::Serialization("cannot allocate Vector Store selected envelope".into())
        })?;
        bytes.extend_from_slice(VECTOR_MAGIC);
        bytes.push(VECTOR_SECTION_VERSION);
        bytes.extend_from_slice(&[0; 3]);
        bytes.extend_from_slice(&wire_len.to_le_bytes());
        bytes.extend_from_slice(count_bytes);
        for range in selected {
            let entry = data.get(range).ok_or_else(|| {
                Error::Serialization("Vector Store selected entry range is invalid".into())
            })?;
            bytes.extend_from_slice(entry);
        }
        Ok(bytes)
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Returns a stable, key-qualified fingerprint of every exact v4 state.
    ///
    /// Fingerprints are sorted by key and cover every byte that v4 persists,
    /// including the deterministic HNSW continuation.
    ///
    /// # Errors
    ///
    /// Returns an error if the target registry aliases an index, a key is
    /// malformed or duplicated, or
    /// its exact state cannot be encoded.
    pub fn exact_state_fingerprints(&self) -> Result<Vec<(PhysicalIndexKey, [u8; 32])>> {
        let snapshot = capture_v4(&self.indexes)?;
        snapshot
            .indexes
            .into_iter()
            .map(|index| {
                let bytes = bincode::serde::encode_to_vec(&index, bincode::config::standard())
                    .map_err(|error| {
                        Error::Internal(format!(
                            "Vector Store v4 fingerprint serialization failed: {error}"
                        ))
                    })?;
                let mut hasher = blake3::Hasher::new_derive_key(
                    "grafeo.vector-store.v4.exact-state-fingerprint",
                );
                hasher.update(&bytes);
                Ok((index.key, *hasher.finalize().as_bytes()))
            })
            .collect()
    }
}

fn usize_to_u64(value: usize, description: &str) -> std::result::Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("{description} does not fit the v4 wire format"))
}

fn u64_to_usize(value: u64, description: &str) -> std::result::Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{description} does not fit this host"))
}

fn config_to_v4(config: HnswConfig) -> std::result::Result<HnswConfigV4, String> {
    Ok(HnswConfigV4 {
        dimensions: usize_to_u64(config.dimensions, "HNSW dimensions")?,
        metric: config.metric,
        m: usize_to_u64(config.m, "HNSW m")?,
        m_max: usize_to_u64(config.m_max, "HNSW m_max")?,
        ef_construction: usize_to_u64(config.ef_construction, "HNSW ef_construction")?,
        ef: usize_to_u64(config.ef, "HNSW ef")?,
        ml_bits: config.ml.to_bits(),
        alpha_bits: config.alpha.to_bits(),
        max_elements: config
            .max_elements
            .map(|value| usize_to_u64(value, "HNSW max_elements"))
            .transpose()?,
    })
}

fn config_from_v4(config: HnswConfigV4) -> std::result::Result<HnswConfig, String> {
    Ok(HnswConfig {
        dimensions: u64_to_usize(config.dimensions, "HNSW dimensions")?,
        metric: config.metric,
        m: u64_to_usize(config.m, "HNSW m")?,
        m_max: u64_to_usize(config.m_max, "HNSW m_max")?,
        ef_construction: u64_to_usize(config.ef_construction, "HNSW ef_construction")?,
        ef: u64_to_usize(config.ef, "HNSW ef")?,
        ml: f64::from_bits(config.ml_bits),
        alpha: f32::from_bits(config.alpha_bits),
        max_elements: config
            .max_elements
            .map(|value| u64_to_usize(value, "HNSW max_elements"))
            .transpose()?,
    })
}

fn hnsw_to_v4(state: HnswExactState) -> std::result::Result<HnswStateV4, String> {
    Ok(HnswStateV4 {
        config: config_to_v4(state.config)?,
        entry_point: state.entry_point,
        max_level: usize_to_u64(state.max_level, "HNSW max level")?,
        nodes: state.nodes,
        deleted: state.deleted,
        rng: HnswRngV4 {
            algorithm: HNSW_RNG_SPLITMIX64,
            state: state.rng_state,
        },
    })
}

fn hnsw_from_v4(state: HnswStateV4) -> std::result::Result<HnswExactState, String> {
    if state.rng.algorithm != HNSW_RNG_SPLITMIX64 {
        return Err(format!(
            "unsupported HNSW RNG algorithm {}",
            state.rng.algorithm
        ));
    }
    Ok(HnswExactState {
        config: config_from_v4(state.config)?,
        entry_point: state.entry_point,
        max_level: u64_to_usize(state.max_level, "HNSW max level")?,
        nodes: state.nodes,
        deleted: state.deleted,
        rng_state: state.rng.state,
    })
}

fn quantization_to_v4(
    quantization: QuantizationType,
) -> std::result::Result<QuantizationV4, String> {
    Ok(match quantization {
        QuantizationType::None => QuantizationV4::None,
        QuantizationType::Scalar => QuantizationV4::Scalar,
        QuantizationType::Binary => QuantizationV4::Binary,
        QuantizationType::Product { num_subvectors } => QuantizationV4::Product {
            num_subvectors: usize_to_u64(num_subvectors, "PQ subvector count")?,
        },
    })
}

fn quantization_from_v4(
    quantization: QuantizationV4,
) -> std::result::Result<QuantizationType, String> {
    Ok(match quantization {
        QuantizationV4::None => QuantizationType::None,
        QuantizationV4::Scalar => QuantizationType::Scalar,
        QuantizationV4::Binary => QuantizationType::Binary,
        QuantizationV4::Product { num_subvectors } => QuantizationType::Product {
            num_subvectors: u64_to_usize(num_subvectors, "PQ subvector count")?,
        },
    })
}

fn scalar_quantizer_to_v4(
    state: ScalarQuantizerExactState,
) -> std::result::Result<ScalarQuantizerV4, String> {
    Ok(ScalarQuantizerV4 {
        min: state.min,
        scale: state.scale,
        inv_scale: state.inv_scale,
        dimensions: usize_to_u64(state.dimensions, "scalar quantizer dimensions")?,
    })
}

fn scalar_quantizer_from_v4(
    state: ScalarQuantizerV4,
) -> std::result::Result<ScalarQuantizerExactState, String> {
    Ok(ScalarQuantizerExactState {
        min: state.min,
        scale: state.scale,
        inv_scale: state.inv_scale,
        dimensions: u64_to_usize(state.dimensions, "scalar quantizer dimensions")?,
    })
}

fn product_quantizer_to_v4(
    state: ProductQuantizerExactState,
) -> std::result::Result<ProductQuantizerV4, String> {
    Ok(ProductQuantizerV4 {
        num_subvectors: usize_to_u64(state.num_subvectors, "PQ subvector count")?,
        num_centroids: usize_to_u64(state.num_centroids, "PQ centroid count")?,
        subvector_dim: usize_to_u64(state.subvector_dim, "PQ subvector dimensions")?,
        dimensions: usize_to_u64(state.dimensions, "PQ dimensions")?,
        centroids: state.centroids,
    })
}

fn product_quantizer_from_v4(
    state: ProductQuantizerV4,
) -> std::result::Result<ProductQuantizerExactState, String> {
    Ok(ProductQuantizerExactState {
        num_subvectors: u64_to_usize(state.num_subvectors, "PQ subvector count")?,
        num_centroids: u64_to_usize(state.num_centroids, "PQ centroid count")?,
        subvector_dim: u64_to_usize(state.subvector_dim, "PQ subvector dimensions")?,
        dimensions: u64_to_usize(state.dimensions, "PQ dimensions")?,
        centroids: state.centroids,
    })
}

fn exact_state_to_v4(state: VectorExactState) -> std::result::Result<VectorStateV4, String> {
    Ok(match state {
        VectorExactState::Hnsw(state) => VectorStateV4::Hnsw(hnsw_to_v4(state)?),
        VectorExactState::Quantized(state) => VectorStateV4::Quantized(QuantizedStateV4 {
            hnsw: hnsw_to_v4(state.hnsw)?,
            quantization_type: quantization_to_v4(state.quantization_type)?,
            scalar_quantizer: state
                .scalar_quantizer
                .map(scalar_quantizer_to_v4)
                .transpose()?,
            product_quantizer: state
                .product_quantizer
                .map(product_quantizer_to_v4)
                .transpose()?,
            vectors: state.vectors,
            scalar_vectors: state.scalar_vectors,
            binary_vectors: state.binary_vectors,
            product_codes: state.product_codes,
            rescore: state.rescore,
            rescore_factor: usize_to_u64(state.rescore_factor, "rescore factor")?,
            training_threshold: usize_to_u64(
                state.training_threshold,
                "quantizer training threshold",
            )?,
            training_samples: state.training_samples,
            quantizer_trained: state.quantizer_trained,
        }),
    })
}

fn exact_state_from_v4(state: VectorStateV4) -> std::result::Result<VectorExactState, String> {
    Ok(match state {
        VectorStateV4::Hnsw(state) => VectorExactState::Hnsw(hnsw_from_v4(state)?),
        VectorStateV4::Quantized(state) => VectorExactState::Quantized(QuantizedExactState {
            hnsw: hnsw_from_v4(state.hnsw)?,
            quantization_type: quantization_from_v4(state.quantization_type)?,
            scalar_quantizer: state
                .scalar_quantizer
                .map(scalar_quantizer_from_v4)
                .transpose()?,
            product_quantizer: state
                .product_quantizer
                .map(product_quantizer_from_v4)
                .transpose()?,
            vectors: state.vectors,
            scalar_vectors: state.scalar_vectors,
            binary_vectors: state.binary_vectors,
            product_codes: state.product_codes,
            rescore: state.rescore,
            rescore_factor: u64_to_usize(state.rescore_factor, "rescore factor")?,
            training_threshold: u64_to_usize(
                state.training_threshold,
                "quantizer training threshold",
            )?,
            training_samples: state.training_samples,
            quantizer_trained: state.quantizer_trained,
        }),
    })
}

fn canonical_key_map<'key>(
    keys: impl IntoIterator<Item = (usize, &'key PhysicalIndexKey)>,
) -> std::result::Result<BTreeMap<PhysicalIndexKey, usize>, String> {
    let mut canonical = BTreeMap::new();
    for (position, key) in keys {
        if key.family() != PhysicalIndexFamily::Vector {
            return Err(format!("invalid vector index key family {key:?}"));
        }
        if canonical.insert(key.clone(), position).is_some() {
            return Err(format!("duplicate vector index key {key:?}"));
        }
    }
    Ok(canonical)
}

fn capture_v4(indexes: &[(PhysicalIndexKey, VectorIndexView)]) -> Result<VectorStoreSnapshotV4> {
    let keys = canonical_key_map(
        indexes
            .iter()
            .enumerate()
            .map(|(position, (key, _))| (position, key)),
    )
    .map_err(Error::Serialization)?;
    let mut identities = HashSet::with_capacity(indexes.len());
    for (key, index) in indexes {
        if !identities.insert(index.exact_restore_identity()) {
            return Err(Error::Serialization(format!(
                "Vector Store v4 key {key:?} aliases another index handle"
            )));
        }
    }
    let mut captured = Vec::with_capacity(indexes.len());
    for (key, position) in keys {
        let index = &indexes
            .get(position)
            .ok_or_else(|| Error::Internal("Vector Store capture position is absent".to_string()))?
            .1;
        let state = index.snapshot_exact().map_err(|error| {
            Error::Internal(format!(
                "Vector Store v4 snapshot rejected key {key:?}: {error}"
            ))
        })?;
        captured.push(IndexSnapshotV4 {
            key: key.clone(),
            state: exact_state_to_v4(state).map_err(Error::Serialization)?,
        });
    }
    Ok(VectorStoreSnapshotV4 { indexes: captured })
}

fn serialize_v4(indexes: &[(PhysicalIndexKey, VectorIndexView)]) -> Result<Vec<u8>> {
    let snapshot = capture_v4(indexes)?;
    let payload =
        bincode::serde::encode_to_vec(&snapshot, bincode::config::standard()).map_err(|error| {
            Error::Internal(format!("Vector Store v4 serialization failed: {error}"))
        })?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| Error::Internal("Vector Store v4 payload is too large".to_string()))?;
    let capacity = V4_HEADER_SIZE.checked_add(payload.len()).ok_or_else(|| {
        Error::Serialization("Vector Store v4 envelope size overflows".to_string())
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        Error::Serialization("cannot allocate Vector Store v4 envelope".to_string())
    })?;
    bytes.extend_from_slice(VECTOR_MAGIC);
    bytes.push(VECTOR_SECTION_VERSION);
    bytes.extend_from_slice(&[0; 3]);
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

struct PreparedIndexRestore<'a> {
    key: PhysicalIndexKey,
    target: &'a VectorIndexView,
    state: PreparedVectorExactState,
}

fn target_registry(
    format: &str,
    indexes: &[(PhysicalIndexKey, VectorIndexView)],
) -> Result<BTreeMap<PhysicalIndexKey, usize>> {
    let keys = canonical_key_map(
        indexes
            .iter()
            .enumerate()
            .map(|(position, (key, _))| (position, key)),
    )
    .map_err(Error::Serialization)?;
    let mut identities = HashSet::with_capacity(indexes.len());
    for (key, index) in indexes {
        if !identities.insert(index.exact_restore_identity()) {
            return Err(Error::Serialization(format!(
                "Vector Store {format} target key {key:?} aliases another index handle"
            )));
        }
    }
    Ok(keys)
}

fn install_prepared(format: &str, mut prepared: Vec<PreparedIndexRestore<'_>>) -> Result<()> {
    // Canonical keys provide one process-independent global lock order even
    // when caller registry order differs from wire order.
    prepared.sort_by(|left, right| left.key.cmp(&right.key));
    let mut guards = Vec::new();
    guards.try_reserve_exact(prepared.len()).map_err(|_| {
        Error::Serialization("cannot allocate Vector Store restore guards".to_string())
    })?;
    let mut ready = Vec::new();
    ready.try_reserve_exact(prepared.len()).map_err(|_| {
        Error::Serialization("cannot allocate Vector Store ready replacements".to_string())
    })?;
    for entry in &prepared {
        let Some(guard) = entry.target.pin_exact_restore() else {
            return Err(Error::Serialization(format!(
                "Vector Store {format} restore for key {:?} lacks mutation authority",
                entry.key
            )));
        };
        guards.push(guard);
    }
    for (entry, guard) in prepared.into_iter().zip(&guards) {
        let replacement = entry
            .target
            .bind_exact_restore(entry.state, guard)
            .map_err(|error| {
                Error::Serialization(format!(
                    "Vector Store {format} restore for key {:?} is not ready: {error}",
                    entry.key
                ))
            })?;
        ready.push(replacement);
    }
    // Every capability borrows its actual guard. Keep the entire guard set
    // alive until all allocation-free, infallible replacements have applied.
    for replacement in ready {
        replacement.apply();
    }
    Ok(())
}

fn preflight_v4_envelope(data: &[u8]) -> Result<&[u8]> {
    if data.is_empty() {
        return Err(Error::Serialization(
            "Vector Store section payload is missing".to_string(),
        ));
    }
    if data.len() < VECTOR_MAGIC.len() && VECTOR_MAGIC.starts_with(data) {
        return Err(Error::Serialization(
            "Vector Store v4 header truncated".to_string(),
        ));
    }
    if data.get(..4) != Some(VECTOR_MAGIC.as_slice()) {
        return Err(Error::Serialization(
            "unsupported Vector Store magic; expected current GVST envelope".to_string(),
        ));
    }
    let version = data
        .get(4)
        .copied()
        .ok_or_else(|| Error::Serialization("Vector Store v4 header truncated".to_string()))?;
    if version != VECTOR_SECTION_VERSION {
        return Err(Error::Serialization(format!(
            "unsupported Vector Store version {version}; expected {VECTOR_SECTION_VERSION}"
        )));
    }
    if data.len() < V4_HEADER_SIZE {
        return Err(Error::Serialization(
            "Vector Store v4 header truncated".to_string(),
        ));
    }
    if data[5..8] != [0; 3] {
        return Err(Error::Serialization(
            "Vector Store v4 reserved header bits are nonzero".to_string(),
        ));
    }
    let payload_len = usize::try_from(u64::from_le_bytes(
        data[8..16]
            .try_into()
            .map_err(|_| Error::Serialization("invalid Vector Store length width".into()))?,
    ))
    .map_err(|_| Error::Serialization("Vector Store v4 payload length overflows".to_string()))?;
    let expected_len = V4_HEADER_SIZE
        .checked_add(payload_len)
        .ok_or_else(|| Error::Serialization("Vector Store v4 payload range overflows".into()))?;
    if data.len() != expected_len {
        return Err(Error::Serialization(format!(
            "Vector Store v4 payload length mismatch: header {payload_len}, bytes {}",
            data.len() - V4_HEADER_SIZE
        )));
    }
    Ok(&data[V4_HEADER_SIZE..])
}

fn decode_v4_payload(payload: &[u8]) -> Result<VectorStoreSnapshotV4> {
    let (snapshot, consumed): (VectorStoreSnapshotV4, usize) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard()).map_err(
            |error| {
                Error::Serialization(format!("Vector Store v4 deserialization failed: {error}"))
            },
        )?;
    if consumed != payload.len() {
        return Err(Error::Serialization(
            "Vector Store v4 payload contains trailing data".to_string(),
        ));
    }
    Ok(snapshot)
}

fn deserialize_v4(data: &[u8], indexes: &[(PhysicalIndexKey, VectorIndexView)]) -> Result<()> {
    let payload = preflight_v4_envelope(data)?;
    // A serde sequence begins with its u64 element count in bincode. Inspect
    // that scalar before decoding any per-index DTO so a hostile or simply
    // wrong section cannot allocate work for a registry it can never match.
    let (wire_count, _): (u64, usize) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard()).map_err(
            |error| {
                Error::Serialization(format!(
                    "Vector Store v4 index-count preflight failed: {error}"
                ))
            },
        )?;
    let wire_count = usize::try_from(wire_count)
        .map_err(|_| Error::Serialization("Vector Store v4 index count overflows".to_string()))?;
    if wire_count != indexes.len() {
        return Err(Error::Serialization(format!(
            "Vector Store v4 index count mismatch: snapshot {wire_count}, target {}",
            indexes.len()
        )));
    }
    let snapshot = decode_v4_payload(payload)?;
    // Keep the decoded check as a defence against assumptions about serde's
    // sequence representation becoming stale in a future bincode migration.
    if snapshot.indexes.len() != indexes.len() {
        return Err(Error::Serialization(format!(
            "Vector Store v4 index count mismatch: snapshot {}, target {}",
            snapshot.indexes.len(),
            indexes.len()
        )));
    }
    let snapshot_keys = canonical_key_map(
        snapshot
            .indexes
            .iter()
            .enumerate()
            .map(|(position, index)| (position, &index.key)),
    )
    .map_err(Error::Serialization)?;
    let targets = target_registry("v4", indexes)?;
    if snapshot_keys.keys().ne(targets.keys()) {
        return Err(Error::Serialization(
            "Vector Store v4 authoritative key set does not match recovery targets".to_string(),
        ));
    }

    let mut prepared = Vec::with_capacity(snapshot.indexes.len());
    let mut snapshots: Vec<Option<IndexSnapshotV4>> =
        snapshot.indexes.into_iter().map(Some).collect();
    for (key, snapshot_position) in snapshot_keys {
        let target_position = targets.get(&key).copied().ok_or_else(|| {
            Error::Serialization("Vector Store recovery target is absent".to_string())
        })?;
        let target = &indexes
            .get(target_position)
            .ok_or_else(|| {
                Error::Serialization("Vector Store recovery target position is absent".to_string())
            })?
            .1;
        let index = snapshots
            .get_mut(snapshot_position)
            .and_then(Option::take)
            .ok_or_else(|| {
                Error::Serialization(
                    "Vector Store snapshot position is absent or repeated".to_string(),
                )
            })?;
        let state = exact_state_from_v4(index.state).map_err(|error| {
            Error::Serialization(format!(
                "Vector Store v4 key {:?} is invalid: {error}",
                index.key
            ))
        })?;
        let state = target.prepare_exact_restore(state).map_err(|error| {
            Error::Serialization(format!(
                "Vector Store v4 key {:?} cannot be restored: {error}",
                index.key
            ))
        })?;
        prepared.push(PreparedIndexRestore { key, target, state });
    }
    install_prepared("v4", prepared)
}

impl Section for VectorStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::VectorStore
    }

    fn version(&self) -> u8 {
        VECTOR_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        serialize_v4(&self.indexes)
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        Self::validate_payload(data)?;
        if self.restore_mode != RestoreMode::UnpublishedRecovery {
            return Err(Error::Serialization(
                "Vector Store v4 exact restore requires explicitly unpublished recovery targets"
                    .to_string(),
            ));
        }
        deserialize_v4(data, &self.indexes)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.indexes
            .iter()
            .map(|(_, idx)| idx.heap_memory_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "lpg")]
    use crate::graph::lpg::LpgStore;
    #[cfg(feature = "lpg")]
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::vector::{HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex};
    use std::collections::HashMap;

    fn make_test_index() -> (PhysicalIndexKey, Arc<VectorIndexKind>) {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));

        // Manually set up a small topology via snapshot/restore
        let nodes = vec![
            (NodeId::new(1), vec![vec![NodeId::new(2), NodeId::new(3)]]),
            (NodeId::new(2), vec![vec![NodeId::new(1), NodeId::new(3)]]),
            (NodeId::new(3), vec![vec![NodeId::new(1), NodeId::new(2)]]),
        ];
        index.restore_topology(Some(NodeId::new(1)), 0, nodes);

        (
            PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
            index,
        )
    }

    fn exact_fingerprint(section: &VectorStoreSection) -> [u8; 32] {
        section
            .exact_state_fingerprints()
            .expect("fingerprint exact vector state")[0]
            .1
    }

    #[test]
    fn payload_metadata_distinguishes_quantized_none_from_plain_hnsw() -> Result<()> {
        let (plain_key, plain) = make_test_index();
        let quantized_key = PhysicalIndexKey::vector(GraphPath::root(), "Quantized", "embedding");
        let quantized = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::new(
            HnswConfig::new(4, DistanceMetric::Cosine),
            QuantizationType::None,
        )));
        let section = VectorStoreSection::new(vec![
            (quantized_key.clone(), quantized),
            (plain_key.clone(), plain),
        ]);
        let mut bytes = section.serialize()?;
        assert_eq!(
            VectorStoreSection::payload_keys(&bytes)?,
            vec![plain_key, quantized_key.clone()]
        );
        assert_eq!(
            VectorStoreSection::payload_quantized_keys(&bytes)?,
            vec![quantized_key]
        );
        // Kind metadata must still preflight the complete image, not merely
        // trust the prefix that contains the concrete-kind discriminant.
        bytes.pop();
        assert!(VectorStoreSection::payload_quantized_keys(&bytes).is_err());
        assert!(VectorStoreSection::payload_keys(&bytes).is_err());
        Ok(())
    }

    #[test]
    fn payload_references_exact_root_and_named_owners()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let root_key = PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding");
        let named_key =
            PhysicalIndexKey::vector(GraphPath::from_components(&["a", "b"])?, "Doc", "embedding");
        let literal_key =
            PhysicalIndexKey::vector(GraphPath::from_components(&["a/b"])?, "Doc", "embedding");
        let source = VectorStoreSection::new(vec![
            (root_key.clone(), topology_hnsw(7, 1)),
            (named_key.clone(), topology_hnsw(7, 2)),
            (literal_key.clone(), topology_hnsw(9, 3)),
        ]);
        let bytes = source.serialize()?;
        for key in [&root_key, &named_key] {
            assert!(VectorStoreSection::payload_references_nodes(
                &bytes,
                |owner, id| { owner == key && id == NodeId::new(7) }
            )?);
        }
        assert!(!VectorStoreSection::payload_references_nodes(
            &bytes,
            |owner, id| { owner == &root_key && id == NodeId::new(9) }
        )?);
        assert!(!VectorStoreSection::payload_references_nodes(
            &bytes,
            |owner, id| { owner == &named_key && id == NodeId::new(9) }
        )?);
        assert!(VectorStoreSection::payload_references_nodes(
            &bytes,
            |owner, id| { owner == &literal_key && id == NodeId::new(9) }
        )?);
        assert!(!VectorStoreSection::payload_references_nodes(
            &bytes,
            |_, id| { id == NodeId::new(99) }
        )?);
        assert!(!VectorStoreSection::payload_references_nodes(
            &VectorStoreSection::new(Vec::new()).serialize()?,
            |_, _| true,
        )?);
        assert_eq!(source.serialize()?, bytes);
        Ok(())
    }

    #[test]
    fn payload_references_isolated_hnsw_roles() -> Result<()> {
        let key = PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding");
        let source = VectorStoreSection::new(vec![(key.clone(), empty_hnsw(1))]);
        let template = capture_v4(&source.indexes)?
            .indexes
            .into_iter()
            .next()
            .ok_or_else(|| Error::Internal("missing reference fixture owner".into()))?;
        let VectorStateV4::Hnsw(template) = template.state else {
            return Err(Error::Internal(
                "reference fixture is not plain HNSW".into(),
            ));
        };
        let selected = NodeId::new(77);
        // Deliberately isolate wire roles, including references with no live
        // topology node. Semantic restore validation is a separate contract.
        for role in 0..4 {
            let mut state = template.clone();
            match role {
                0 => state.entry_point = Some(selected),
                1 => state.nodes.push((selected, vec![vec![]])),
                2 => state.nodes.push((NodeId::new(1), vec![vec![selected]])),
                _ => state.deleted.push(selected),
            }
            let bytes = encode_v4_fixture(&VectorStoreSnapshotV4 {
                indexes: vec![IndexSnapshotV4 {
                    key: key.clone(),
                    state: VectorStateV4::Hnsw(state),
                }],
            });
            assert!(
                VectorStoreSection::payload_references_nodes(&bytes, |owner, id| {
                    owner == &key && id == selected
                })?,
                "missing HNSW reference role {role}"
            );
            assert!(!VectorStoreSection::payload_references_nodes(
                &bytes,
                |_, id| { id == NodeId::new(99) }
            )?);
        }
        Ok(())
    }

    #[test]
    fn payload_references_retained_deleted_history() -> Result<()> {
        let (key, index) = make_test_index();
        assert!(index.remove(NodeId::new(2)));
        let source = VectorStoreSection::new(vec![(key.clone(), index)]);
        let bytes = source.serialize()?;
        assert!(VectorStoreSection::payload_references_nodes(
            &bytes,
            |owner, id| { owner == &key && id == NodeId::new(2) }
        )?);
        assert_eq!(source.serialize()?, bytes);
        Ok(())
    }

    #[test]
    fn payload_references_all_quantized_map_and_topology_roles() -> Result<()> {
        let key = PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding");
        let source = VectorStoreSection::new(vec![(
            key.clone(),
            Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
                HnswConfig::new(4, DistanceMetric::Cosine),
                QuantizationType::None,
                3,
            ))),
        )]);
        let template = capture_v4(&source.indexes)?
            .indexes
            .into_iter()
            .next()
            .ok_or_else(|| Error::Internal("missing quantized reference fixture".into()))?;
        let VectorStateV4::Quantized(template) = template.state else {
            return Err(Error::Internal("reference fixture is not quantized".into()));
        };
        for kind in [
            QuantizationV4::None,
            QuantizationV4::Scalar,
            QuantizationV4::Binary,
            QuantizationV4::Product { num_subvectors: 2 },
        ] {
            let mut state = template.clone();
            state.quantization_type = kind;
            state.hnsw.entry_point = Some(NodeId::new(11));
            state.hnsw.nodes = vec![(NodeId::new(12), vec![vec![NodeId::new(13)]])];
            state.hnsw.deleted = vec![NodeId::new(14)];
            // Unique IDs prove each encoded role is visited, independently of
            // whether that vector map is active for the configured kind.
            state.vectors = vec![(NodeId::new(21), vec![1.0, 2.0, 3.0, 4.0])];
            state.scalar_vectors = vec![(NodeId::new(22), vec![1, 2, 3, 4])];
            state.binary_vectors = vec![(NodeId::new(23), vec![17])];
            state.product_codes = vec![(NodeId::new(24), vec![1, 2])];
            let bytes = encode_v4_fixture(&VectorStoreSnapshotV4 {
                indexes: vec![IndexSnapshotV4 {
                    key: key.clone(),
                    state: VectorStateV4::Quantized(state),
                }],
            });
            for selected in [11, 12, 13, 14, 21, 22, 23, 24] {
                assert!(
                    VectorStoreSection::payload_references_nodes(&bytes, |owner, id| {
                        owner == &key && id == NodeId::new(selected)
                    })?,
                    "missing quantized reference {selected}"
                );
            }
            assert!(!VectorStoreSection::payload_references_nodes(
                &bytes,
                |_, id| { id == NodeId::new(99) }
            )?);
        }
        Ok(())
    }

    #[test]
    fn payload_references_rejects_malformed_tail_after_match() -> Result<()> {
        let first = PhysicalIndexKey::vector(GraphPath::root(), "A", "embedding");
        let source = VectorStoreSection::new(vec![
            (first.clone(), topology_hnsw(1, 1)),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Z", "embedding"),
                Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
                    HnswConfig::new(4, DistanceMetric::Cosine),
                    QuantizationType::None,
                    2,
                ))),
            ),
        ]);
        let bytes = source.serialize()?;
        let payload = preflight_v4_envelope(&bytes)?;
        let mut truncated = payload.to_vec();
        truncated.pop();
        let mut trailing = payload.to_vec();
        trailing.push(0);
        let mut invalid_bool = payload.to_vec();
        *invalid_bool
            .last_mut()
            .ok_or_else(|| Error::Internal("empty reference fixture payload".into()))? = 2;
        for malformed in [truncated, trailing, invalid_bool] {
            let mut observed_match = false;
            let result = VectorStoreSection::payload_references_nodes(
                &wrap_v4_payload(&malformed),
                |owner, id| {
                    let selected = owner == &first && id == NodeId::new(1);
                    observed_match |= selected;
                    selected
                },
            );
            assert!(
                observed_match,
                "malformed fixture must reach an early match"
            );
            assert!(
                result.is_err(),
                "early match must not hide a malformed tail"
            );
        }
        assert_eq!(source.serialize()?, bytes);
        Ok(())
    }

    #[test]
    fn payload_selection_preserves_exact_entries_and_quantized_continuation() -> Result<()> {
        for quantization in [QuantizationType::None, QuantizationType::Binary] {
            let config = HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4);
            let quantized = Arc::new(VectorIndexKind::Quantized(
                QuantizedHnswIndex::with_seed(config.clone(), quantization, 71)
                    .without_rescore()
                    .with_rescore_factor(7)
                    .with_training_threshold(10),
            ));
            let accessor = |_id: NodeId| -> Option<Arc<[f32]>> { None };
            quantized.insert(NodeId::new(10), &[1.0, 0.0, -1.0, 0.5], &accessor);
            quantized.insert(NodeId::new(11), &[0.0, 1.0, 0.5, -1.0], &accessor);
            let plain_key = PhysicalIndexKey::vector(GraphPath::root(), "Plain", "embedding");
            let quantized_key =
                PhysicalIndexKey::vector(GraphPath::root(), "Quantized", "embedding");
            let source = VectorStoreSection::new(vec![
                (quantized_key.clone(), Arc::clone(&quantized)),
                (plain_key, topology_hnsw(1, 93)),
            ]);
            let bytes = source.serialize()?;
            let keys = VectorStoreSection::payload_keys(&bytes)?;
            assert_eq!(
                VectorStoreSection::select_payload_keys(&bytes, &keys)?,
                bytes
            );
            assert_eq!(
                VectorStoreSection::select_payload_keys(&bytes, &[])?,
                VectorStoreSection::new(Vec::new()).serialize()?
            );
            let subset = VectorStoreSection::select_payload_keys(
                &bytes,
                std::slice::from_ref(&quantized_key),
            )?;
            let source_single =
                VectorStoreSection::new(vec![(quantized_key.clone(), Arc::clone(&quantized))]);
            assert_eq!(subset, source_single.serialize()?);
            let source_ranges = VectorStoreSection::payload_entry_ranges(&bytes)?;
            let subset_ranges = VectorStoreSection::payload_entry_ranges(&subset)?;
            for (key, range) in subset_ranges {
                let (_, original) = source_ranges
                    .iter()
                    .find(|(known, _)| known == &key)
                    .ok_or_else(|| Error::Internal("selected fixture key is absent".into()))?;
                assert_eq!(bytes.get(original.clone()), subset.get(range));
            }
            assert_eq!(
                source.serialize()?,
                bytes,
                "selection must leave its source intact"
            );
            let restored = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
                config,
                quantization,
                999,
            )));
            let mut target = VectorStoreSection::for_unpublished_recovery(vec![(
                quantized_key,
                Arc::clone(&restored),
            )]);
            target.deserialize(&subset)?;
            assert_eq!(target.serialize()?, subset);
            quantized.insert(NodeId::new(12), &[0.25, -0.5, 0.75, 1.0], &accessor);
            restored.insert(NodeId::new(12), &[0.25, -0.5, 0.75, 1.0], &accessor);
            assert_eq!(target.serialize()?, source_single.serialize()?);
        }
        Ok(())
    }

    #[test]
    fn payload_selection_rejects_invalid_requests_and_unselected_corruption() -> Result<()> {
        let source = VectorStoreSection::new(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "A", "embedding"),
                empty_hnsw(1),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "B", "embedding"),
                empty_hnsw(2),
            ),
        ]);
        let bytes = source.serialize()?;
        let keys = VectorStoreSection::payload_keys(&bytes)?;
        let first = keys
            .first()
            .ok_or_else(|| Error::Internal("empty fixture keys".into()))?;
        let mut unsorted = keys.clone();
        unsorted.reverse();
        for requested in [
            unsorted,
            vec![first.clone(), first.clone()],
            vec![PhysicalIndexKey::vector(
                GraphPath::root(),
                "Absent",
                "embedding",
            )],
        ] {
            assert!(VectorStoreSection::select_payload_keys(&bytes, &requested).is_err());
        }
        // Keep the envelope length valid while truncating the second owner's
        // payload: selecting only the first owner must not hide corruption.
        let mut payload = preflight_v4_envelope(&bytes)?.to_vec();
        payload.pop();
        let truncated = wrap_v4_payload(&payload);
        assert!(VectorStoreSection::payload_entry_ranges(&truncated).is_err());
        assert!(
            VectorStoreSection::select_payload_keys(&truncated, std::slice::from_ref(first))
                .is_err()
        );
        assert!(VectorStoreSection::select_payload_keys(&truncated, &[]).is_err());
        let mut duplicate = capture_v4(&source.indexes)?;
        let repeated = duplicate
            .indexes
            .first()
            .cloned()
            .ok_or_else(|| Error::Internal("empty fixture owner image".into()))?;
        duplicate.indexes.push(repeated);
        let malformed = encode_v4_fixture(&duplicate);
        assert!(VectorStoreSection::payload_entry_ranges(&malformed).is_err());
        assert!(VectorStoreSection::select_payload_keys(&malformed, &[]).is_err());
        assert_eq!(source.serialize()?, bytes);
        Ok(())
    }

    #[test]
    fn payload_selection_rewrites_multibyte_owner_count() -> Result<()> {
        let source = VectorStoreSection::new(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Template", "embedding"),
            empty_hnsw(7),
        )]);
        let template = capture_v4(&source.indexes)?
            .indexes
            .into_iter()
            .next()
            .ok_or_else(|| Error::Internal("empty fixture owner image".into()))?;
        let snapshot = VectorStoreSnapshotV4 {
            indexes: (0..252)
                .map(|offset| IndexSnapshotV4 {
                    key: PhysicalIndexKey::vector(
                        GraphPath::root(),
                        format!("Owner{offset:03}"),
                        "embedding",
                    ),
                    state: template.state.clone(),
                })
                .collect(),
        };
        let bytes = encode_v4_fixture(&snapshot);
        let keys = VectorStoreSection::payload_keys(&bytes)?;
        assert_eq!(
            VectorStoreSection::select_payload_keys(&bytes, &keys)?,
            bytes
        );
        for count in [250, 251] {
            let requested = keys
                .get(..count)
                .ok_or_else(|| Error::Internal("fixture selected key range is absent".into()))?;
            let selected = VectorStoreSection::select_payload_keys(&bytes, requested)?;
            VectorStoreSection::validate_payload(&selected)?;
            assert_eq!(VectorStoreSection::payload_keys(&selected)?, requested);
            assert_eq!(
                VectorStoreSection::payload_entry_ranges(&selected)?.len(),
                count
            );
        }
        Ok(())
    }

    #[test]
    fn typed_paths_round_trip_and_wrong_family_is_atomic()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let paths: &[&[&str]] = &[&[], &[""], &["default"], &["a/b"], &["a", "b"]];
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for components in paths {
            let key = PhysicalIndexKey::vector(
                GraphPath::from_components(components)?,
                "Doc",
                "embedding",
            );
            sources.push((key.clone(), topology_hnsw(1, 71)));
            targets.push((key, empty_hnsw(93)));
        }
        let source = VectorStoreSection::new(sources);
        let bytes = source.serialize()?;
        let mut target = VectorStoreSection::for_unpublished_recovery(targets);
        target.deserialize(&bytes)?;
        assert_eq!(target.serialize()?, bytes);
        let mut forged = capture_v4(&source.indexes)?;
        forged.indexes[0].key = PhysicalIndexKey::text(GraphPath::root(), "Doc", "embedding");
        assert!(target.deserialize(&encode_v4_fixture(&forged)).is_err());
        assert_eq!(target.serialize()?, bytes);
        let mut shuffled = capture_v4(&source.indexes)?;
        shuffled.indexes.swap(0, 1);
        assert!(target.deserialize(&encode_v4_fixture(&shuffled)).is_err());
        assert_eq!(target.serialize()?, bytes);
        Ok(())
    }

    fn encode_v4_fixture(snapshot: &VectorStoreSnapshotV4) -> Vec<u8> {
        let payload = bincode::serde::encode_to_vec(snapshot, bincode::config::standard())
            .expect("encode v4 fixture");
        wrap_v4_payload(&payload)
    }

    fn wrap_v4_payload(payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(V4_HEADER_SIZE + payload.len());
        bytes.extend_from_slice(VECTOR_MAGIC);
        bytes.push(VECTOR_SECTION_VERSION);
        bytes.extend_from_slice(&[0; 3]);
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn empty_hnsw(seed: u64) -> Arc<VectorIndexKind> {
        Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            seed,
        )))
    }

    fn topology_hnsw(id: u64, seed: u64) -> Arc<VectorIndexKind> {
        let index = empty_hnsw(seed);
        index.restore_topology(
            Some(NodeId::new(id)),
            0,
            vec![(NodeId::new(id), vec![vec![]])],
        );
        index
    }

    #[test]
    fn vector_current_only_rejects_predecessor_envelopes() {
        // Literal empty v1 bincode and v2 packed images. No predecessor
        // writer or DTO is needed to prove these bytes must be rejected.
        for bytes in [
            &[1, 0][..],
            &b"GVST\x02\0\0\0\0\0\0\0\0\0\0\0"[..],
            &b"GVST\x03\0\0\0\0\0\0\0\0\0\0\0"[..],
        ] {
            let mut section = VectorStoreSection::for_unpublished_recovery(vec![]);
            let before = section.serialize().unwrap();
            let error = section
                .deserialize(bytes)
                .expect_err("predecessor images have no reader");
            assert!(error.to_string().contains("unsupported"), "{error}");
            assert_eq!(section.serialize().unwrap(), before);
        }
    }

    #[test]
    fn vector_current_only_rejects_old_header_before_payload_or_mutation() {
        let mut section = VectorStoreSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
            topology_hnsw(19, 314),
        )]);
        let before = section.serialize().unwrap();
        for version in [0, 1, 2, 3, 5, 255] {
            let mut hostile = before.clone();
            hostile[4] = version;
            hostile[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
            let error = section
                .deserialize(&hostile)
                .expect_err("unsupported version");
            assert!(error.to_string().contains("unsupported"), "{error}");
            assert_eq!(section.serialize().unwrap(), before);
        }
    }

    #[test]
    fn vector_current_writer_has_exact_typed_key_envelope() {
        let section = VectorStoreSection::new(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
            topology_hnsw(19, 314),
        )]);
        let bytes = section.serialize().unwrap();
        // Independent current header and key bytes: one Vector key, root path,
        // explicitly present label Item, and property embedding.
        assert_eq!(bytes.len(), 73);
        assert_eq!(&bytes[..16], b"GVST\x04\0\0\0\x39\0\0\0\0\0\0\0");
        assert_eq!(
            &bytes[16..39],
            b"\x01\x04\0\0\0\0\x02\x01\x04Item\x09embedding"
        );
    }

    #[test]
    fn vector_section_round_trip() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&index))]);

        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty());

        // Create a fresh index with same config to restore into
        let config = index.config().clone();
        let fresh_index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
        let mut section2 =
            VectorStoreSection::for_unpublished_recovery(vec![(key, fresh_index.clone())]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(fresh_index.len(), 3);
        let (ep, ml, nodes) = fresh_index.snapshot_topology();
        assert_eq!(ep, Some(NodeId::new(1)));
        assert_eq!(ml, 0);
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn vector_v4_restores_delete_set_config_and_future_rng_continuation() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean)
            .with_m(4)
            .with_m_max(7)
            .with_ef_construction(23)
            .with_ef(19)
            .with_alpha(1.25)
            .with_max_elements(100);
        let source = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            config.clone(),
            0xfeed_beef,
        )));
        let vectors: HashMap<NodeId, Arc<[f32]>> = (1_u64..=18)
            .map(|raw| {
                let base = raw as f32 / 19.0;
                (
                    NodeId::new(raw),
                    Arc::<[f32]>::from([base, base * 0.5, 1.0 - base, base * base]),
                )
            })
            .collect();
        let accessor = |id: NodeId| vectors.get(&id).cloned();
        for raw in 1_u64..=16 {
            let id = NodeId::new(raw);
            source.insert(id, vectors.get(&id).expect("fixture vector"), &accessor);
        }
        assert!(source.remove(NodeId::new(4)));

        let key = PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding");
        let source_section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&source))]);
        let bytes = source_section.serialize().expect("serialize exact v4");
        let restored = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(config, 9)));
        let mut restored_section =
            VectorStoreSection::for_unpublished_recovery(vec![(key, Arc::clone(&restored))]);
        restored_section
            .deserialize(&bytes)
            .expect("restore exact v4");

        assert_eq!(
            exact_fingerprint(&restored_section),
            exact_fingerprint(&source_section)
        );
        assert!(!restored.contains(NodeId::new(4)));
        assert!(
            restored
                .as_hnsw()
                .expect("plain HNSW")
                .contains_including_deleted(NodeId::new(4))
        );

        let next = NodeId::new(17);
        source.insert(next, vectors.get(&next).expect("next vector"), &accessor);
        restored.insert(next, vectors.get(&next).expect("next vector"), &accessor);
        assert_eq!(
            exact_fingerprint(&restored_section),
            exact_fingerprint(&source_section),
            "restored RNG continuation must make the next insertion identical"
        );
    }

    fn assert_quantized_v4_round_trip(quantization: QuantizationType) {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean)
            .with_m(4)
            .with_ef_construction(24)
            .with_ef(17);
        let source = Arc::new(VectorIndexKind::Quantized(
            QuantizedHnswIndex::with_seed(config.clone(), quantization, 71)
                .without_rescore()
                .with_rescore_factor(7)
                .with_training_threshold(10),
        ));
        let vectors: Vec<Vec<f32>> = (1_u64..=13)
            .map(|raw| {
                let base = raw as f32 / 14.0;
                vec![base, 1.0 - base, base * 2.0 - 1.0, base * base]
            })
            .collect();
        let unused_accessor = |_id: NodeId| -> Option<Arc<[f32]>> { None };
        for raw in 1_u64..=12 {
            let position = usize::try_from(raw - 1).expect("small fixture index fits usize");
            source.insert(NodeId::new(raw), &vectors[position], &unused_accessor);
        }
        assert!(source.remove(NodeId::new(3)));

        let key = PhysicalIndexKey::vector(GraphPath::root(), "Image", "embedding");
        let source_section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&source))]);
        let bytes = source_section.serialize().expect("serialize quantized v4");
        let restored = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
            config,
            quantization,
            999,
        )));
        let mut restored_section =
            VectorStoreSection::for_unpublished_recovery(vec![(key, Arc::clone(&restored))]);
        restored_section
            .deserialize(&bytes)
            .expect("restore quantized v4");

        let restored_quantized = restored.as_quantized().expect("quantized index");
        assert!(!restored_quantized.rescoring_enabled());
        assert_eq!(restored_quantized.rescore_factor(), 7);
        assert_eq!(restored_quantized.training_threshold(), 10);
        assert_eq!(
            exact_fingerprint(&restored_section),
            exact_fingerprint(&source_section)
        );

        source.insert(NodeId::new(13), &vectors[12], &unused_accessor);
        restored.insert(NodeId::new(13), &vectors[12], &unused_accessor);
        assert_eq!(
            exact_fingerprint(&restored_section),
            exact_fingerprint(&source_section),
            "quantized state and future topology must remain exact"
        );
    }

    #[test]
    fn vector_v4_scalar_quantized_round_trip_is_exact() {
        assert_quantized_v4_round_trip(QuantizationType::Scalar);
    }

    #[test]
    fn vector_v4_binary_quantized_round_trip_is_exact() {
        assert_quantized_v4_round_trip(QuantizationType::Binary);
    }

    #[test]
    fn vector_v4_product_quantized_round_trip_is_exact() {
        assert_quantized_v4_round_trip(QuantizationType::Product { num_subvectors: 2 });
    }

    fn quantized_corruption_fixture(
        quantization: QuantizationType,
        dimensions: usize,
    ) -> (HnswConfig, VectorStoreSnapshotV4) {
        let config = HnswConfig::new(dimensions, DistanceMetric::Euclidean).with_m(4);
        let source = Arc::new(VectorIndexKind::Quantized(
            QuantizedHnswIndex::with_seed(config.clone(), quantization, 0x55aa)
                .with_training_threshold(10),
        ));
        let unused_accessor = |_id: NodeId| -> Option<Arc<[f32]>> { None };
        for raw in 1_u64..=10 {
            let vector: Vec<f32> = (0..dimensions)
                .map(|dimension| raw as f32 / 11.0 + dimension as f32 / 17.0 - 0.5)
                .collect();
            source.insert(NodeId::new(raw), &vector, &unused_accessor);
        }
        let section = VectorStoreSection::new(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Image", "embedding"),
            source,
        )]);
        let snapshot = capture_v4(&section.indexes).expect("capture quantized fixture");
        (config, snapshot)
    }

    fn assert_quantized_corruption_rejected(
        snapshot: &VectorStoreSnapshotV4,
        config: HnswConfig,
        quantization: QuantizationType,
        expected: &str,
    ) {
        let target = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
            config,
            quantization,
            17,
        )));
        let mut section = VectorStoreSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Image", "embedding"),
            Arc::clone(&target),
        )]);
        let before = exact_fingerprint(&section);
        let error = section
            .deserialize(&encode_v4_fixture(snapshot))
            .expect_err("derived-code corruption must fail closed");
        assert!(
            error.to_string().contains(expected),
            "unexpected corruption error: {error}"
        );
        assert_eq!(exact_fingerprint(&section), before);
    }

    #[test]
    fn vector_v4_rejects_noncanonical_or_non_derived_quantized_codes() {
        let (scalar_config, mut scalar) = quantized_corruption_fixture(QuantizationType::Scalar, 4);
        let VectorStateV4::Quantized(state) = &mut scalar.indexes[0].state else {
            panic!("scalar fixture must be quantized")
        };
        state.scalar_vectors[0].1[0] ^= 1;
        assert_quantized_corruption_rejected(
            &scalar,
            scalar_config,
            QuantizationType::Scalar,
            "does not match",
        );

        let (binary_config, binary) = quantized_corruption_fixture(QuantizationType::Binary, 5);
        let mut padding = binary.clone();
        let VectorStateV4::Quantized(state) = &mut padding.indexes[0].state else {
            panic!("binary fixture must be quantized")
        };
        *state.binary_vectors[0]
            .1
            .last_mut()
            .expect("five dimensions occupy one word") |= 1_u64 << 63;
        assert_quantized_corruption_rejected(
            &padding,
            binary_config.clone(),
            QuantizationType::Binary,
            "padding bits",
        );

        let mut binary_mismatch = binary;
        let VectorStateV4::Quantized(state) = &mut binary_mismatch.indexes[0].state else {
            panic!("binary fixture must be quantized")
        };
        state.binary_vectors[0].1[0] ^= 1;
        assert_quantized_corruption_rejected(
            &binary_mismatch,
            binary_config,
            QuantizationType::Binary,
            "does not match",
        );

        let product_kind = QuantizationType::Product { num_subvectors: 2 };
        let (product_config, product) = quantized_corruption_fixture(product_kind, 4);
        let mut out_of_range = product.clone();
        let VectorStateV4::Quantized(state) = &mut out_of_range.indexes[0].state else {
            panic!("product fixture must be quantized")
        };
        let quantizer = state
            .product_quantizer
            .as_mut()
            .expect("trained product fixture has a quantizer");
        quantizer.num_centroids = 1;
        quantizer.centroids.truncate(
            usize::try_from(quantizer.num_subvectors * quantizer.subvector_dim)
                .expect("small fixture centroid count"),
        );
        state.product_codes[0].1[0] = 1;
        assert_quantized_corruption_rejected(
            &out_of_range,
            product_config.clone(),
            product_kind,
            "outside its centroid table",
        );

        let mut product_mismatch = product;
        let VectorStateV4::Quantized(state) = &mut product_mismatch.indexes[0].state else {
            panic!("product fixture must be quantized")
        };
        state.product_codes[0].1[0] ^= 1;
        assert_quantized_corruption_rejected(
            &product_mismatch,
            product_config,
            product_kind,
            "does not match",
        );
    }

    #[test]
    fn vector_v4_preserves_empty_public_product_zero_and_its_future() {
        let kind = QuantizationType::Product { num_subvectors: 0 };
        let config = HnswConfig::new(4, DistanceMetric::Euclidean).with_m(4);
        let source = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
            config.clone(),
            kind,
            901,
        )));
        let key = PhysicalIndexKey::vector(GraphPath::root(), "Odd", "embedding");
        let source_section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&source))]);
        let bytes = source_section
            .serialize()
            .expect("empty public Product(0) remains checkpointable");

        let restored = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
            config, kind, 3,
        )));
        let mut restored_section =
            VectorStoreSection::for_unpublished_recovery(vec![(key, Arc::clone(&restored))]);
        restored_section
            .deserialize(&bytes)
            .expect("empty public Product(0) remains restorable");
        assert_eq!(
            exact_fingerprint(&restored_section),
            exact_fingerprint(&source_section)
        );

        let vector = [0.25, -0.5, 0.75, 1.0];
        let unused_accessor = |_id: NodeId| -> Option<Arc<[f32]>> { None };
        source.insert(NodeId::new(1), &vector, &unused_accessor);
        restored.insert(NodeId::new(1), &vector, &unused_accessor);
        assert_eq!(
            exact_fingerprint(&restored_section),
            exact_fingerprint(&source_section),
            "untrained Product(0) continuation must remain exact"
        );
    }

    #[test]
    fn vector_v4_corruption_is_rejected_before_target_mutation() {
        let (key, source) = make_test_index();
        let source_section = VectorStoreSection::new(vec![(key.clone(), source)]);
        let valid = capture_v4(&source_section.indexes).expect("capture valid fixture");

        let (target_key, target) = make_test_index();
        assert_eq!(target_key, key);
        let mut target_section =
            VectorStoreSection::for_unpublished_recovery(vec![(target_key, Arc::clone(&target))]);
        let before = exact_fingerprint(&target_section);

        let mut corruptions = Vec::new();

        let mut duplicate_node = valid.clone();
        let VectorStateV4::Hnsw(state) = &mut duplicate_node.indexes[0].state else {
            panic!("plain HNSW fixture")
        };
        state.nodes.push(state.nodes[0].clone());
        corruptions.push(duplicate_node);

        let mut missing_neighbor = valid.clone();
        let VectorStateV4::Hnsw(state) = &mut missing_neighbor.indexes[0].state else {
            panic!("plain HNSW fixture")
        };
        state.nodes[0].1[0].push(NodeId::new(99_999));
        corruptions.push(missing_neighbor);

        let mut wrong_rng = valid.clone();
        let VectorStateV4::Hnsw(state) = &mut wrong_rng.indexes[0].state else {
            panic!("plain HNSW fixture")
        };
        state.rng.algorithm = 255;
        corruptions.push(wrong_rng);

        let mut wrong_config = valid;
        let VectorStateV4::Hnsw(state) = &mut wrong_config.indexes[0].state else {
            panic!("plain HNSW fixture")
        };
        state.config.dimensions += 1;
        corruptions.push(wrong_config);

        for corruption in corruptions {
            let error = target_section
                .deserialize(&encode_v4_fixture(&corruption))
                .expect_err("corrupt v4 image must fail closed");
            assert!(matches!(error, Error::Serialization(_)));
            assert_eq!(
                exact_fingerprint(&target_section),
                before,
                "failed restore must not mutate the target"
            );
        }
    }

    #[test]
    fn vector_v4_requires_exact_key_set_and_rejects_aliased_targets() {
        let (first_key, first) = make_test_index();
        let second_key = PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding");
        let second = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            88,
        )));
        second.restore_topology(
            Some(NodeId::new(10)),
            0,
            vec![(NodeId::new(10), vec![vec![]])],
        );
        let source_section = VectorStoreSection::new(vec![
            (first_key.clone(), first),
            (second_key.clone(), second),
        ]);
        let bytes = source_section
            .serialize()
            .expect("serialize two exact indexes");

        let shared_target = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            5,
        )));
        let mut aliased = VectorStoreSection::for_unpublished_recovery(vec![
            (first_key.clone(), Arc::clone(&shared_target)),
            (second_key, Arc::clone(&shared_target)),
        ]);
        let error = aliased
            .deserialize(&bytes)
            .expect_err("two keys must not restore into one aliased index");
        assert!(error.to_string().contains("aliases another index handle"));
        assert_eq!(shared_target.len(), 0);

        let missing_target = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            6,
        )));
        let mut incomplete = VectorStoreSection::for_unpublished_recovery(vec![(
            first_key,
            Arc::clone(&missing_target),
        )]);
        let error = incomplete
            .deserialize(&bytes)
            .expect_err("missing catalog destination must fail closed");
        assert!(error.to_string().contains("index count mismatch"));
        assert_eq!(missing_target.len(), 0);

        let mut duplicate_snapshot = capture_v4(&source_section.indexes).expect("capture fixture");
        duplicate_snapshot.indexes[1].key = duplicate_snapshot.indexes[0].key.clone();
        let target_a = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            7,
        )));
        let target_b = Arc::new(VectorIndexKind::Hnsw(HnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            8,
        )));
        let mut distinct_targets = VectorStoreSection::for_unpublished_recovery(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                Arc::clone(&target_a),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding"),
                Arc::clone(&target_b),
            ),
        ]);
        let error = distinct_targets
            .deserialize(&encode_v4_fixture(&duplicate_snapshot))
            .expect_err("duplicate persisted keys must fail closed");
        assert!(error.to_string().contains("duplicate vector index key"));
        assert_eq!(target_a.len(), 0);
        assert_eq!(target_b.len(), 0);

        let mut unknown_snapshot = capture_v4(&source_section.indexes).expect("capture fixture");
        unknown_snapshot.indexes[1].key =
            PhysicalIndexKey::vector(GraphPath::root(), "Unknown", "embedding");
        let error = distinct_targets
            .deserialize(&encode_v4_fixture(&unknown_snapshot))
            .expect_err("unknown persisted key must fail closed");
        assert!(error.to_string().contains("key set"));
        assert_eq!(target_a.len(), 0);
        assert_eq!(target_b.len(), 0);
    }

    #[test]
    fn vector_current_late_invalid_index_preserves_every_target() {
        let source = VectorStoreSection::new(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                topology_hnsw(1, 11),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding"),
                topology_hnsw(2, 12),
            ),
        ]);
        let mut image = capture_v4(&source.indexes).unwrap();
        let VectorStateV4::Hnsw(second) = &mut image.indexes[1].state else {
            panic!("plain HNSW fixture");
        };
        second.nodes[0].1[0].push(NodeId::new(999));
        let mut target = VectorStoreSection::for_unpublished_recovery(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                topology_hnsw(71, 21),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding"),
                topology_hnsw(72, 22),
            ),
        ]);
        let before = target.serialize().unwrap();
        let error = target.deserialize(&encode_v4_fixture(&image)).unwrap_err();
        assert!(error.to_string().contains("missing neighbor"), "{error}");
        assert_eq!(
            target.serialize().unwrap(),
            before,
            "a valid first post-image must not publish before a bad second image fails"
        );
    }

    #[test]
    fn vector_exact_handoff_rejects_mismatched_kind_without_mutation() {
        let plain = topology_hnsw(71, 21);
        let quantized = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            QuantizationType::Binary,
            22,
        )));
        quantized.insert(NodeId::new(72), &[1.0, 0.0, -1.0, 0.5], &|_| None);

        for (target, source) in [
            (Arc::clone(&plain), Arc::clone(&quantized)),
            (quantized, plain),
        ] {
            let target = VectorIndexView::new(target);
            let section = VectorStoreSection::from_views(vec![(
                PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                target.clone(),
            )]);
            let before = section.serialize().unwrap();
            let state = source
                .prepare_exact_state(source.snapshot_exact_state().unwrap())
                .unwrap();
            let guard = target.pin_exact_restore().unwrap();

            // Exercise the private handoff directly: the prepared state is
            // valid for its own source, but not for this concrete target.
            let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                target
                    .bind_exact_restore(state, &guard)
                    .map(|ready| ready.apply())
            }));
            let result = attempted.expect("a kind mismatch must return an error, not panic");
            assert!(
                result.is_err(),
                "a mismatched handoff must not silently succeed"
            );
            drop(guard);
            assert_eq!(section.serialize().unwrap(), before);
        }
    }

    #[test]
    fn vector_exact_handoff_rejects_foreign_guard_without_mutation() {
        let target = VectorIndexView::new(topology_hnsw(81, 31));
        let foreign = VectorIndexView::new(topology_hnsw(82, 32));
        let source = topology_hnsw(83, 33);
        let section = VectorStoreSection::from_views(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                target.clone(),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding"),
                foreign.clone(),
            ),
        ]);
        let before = section.serialize().unwrap();
        let state = target
            .prepare_exact_restore(source.snapshot_exact_state().unwrap())
            .unwrap();
        let guard = foreign.pin_exact_restore().unwrap();

        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            target
                .bind_exact_restore(state, &guard)
                .map(|ready| ready.apply())
        }));
        let result = attempted.expect("a foreign guard must return an error, not panic");
        assert!(
            result.is_err(),
            "a foreign guard must not authorize a handoff"
        );
        drop(guard);
        assert_eq!(section.serialize().unwrap(), before);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn vector_exact_handoff_late_maintenance_pin_preserves_every_target() {
        let first = VectorIndexView::new(topology_hnsw(91, 41));
        let second = topology_hnsw(92, 42);
        let second_view = VectorIndexView::new(Arc::clone(&second));
        let section = VectorStoreSection::from_views(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                first.clone(),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding"),
                second_view.clone(),
            ),
        ]);
        let before = section.serialize().unwrap();
        let replacement = topology_hnsw(93, 43).snapshot_exact_state().unwrap();
        let prepared = vec![
            PreparedIndexRestore {
                key: PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                target: &first,
                state: first.prepare_exact_restore(replacement.clone()).unwrap(),
            },
            PreparedIndexRestore {
                key: PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding"),
                target: &second_view,
                state: second_view.prepare_exact_restore(replacement).unwrap(),
            },
        ];
        let maintenance = second.pin_maintenance().unwrap();
        let error = install_prepared("v4", prepared).unwrap_err();
        assert!(
            error.to_string().contains("lacks mutation authority"),
            "{error}"
        );
        drop(maintenance);
        assert_eq!(section.serialize().unwrap(), before);
    }

    #[test]
    fn vector_v4_canonicalizes_keys_for_stable_state_and_lock_order() {
        let item = empty_hnsw(31);
        let other = empty_hnsw(32);
        let item_key = PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding");
        let other_key = PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding");
        let section = VectorStoreSection::new(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Other", "embedding"),
                other,
            ),
            (item_key.clone(), item),
        ]);
        let keys: Vec<_> = section
            .exact_state_fingerprints()
            .expect("canonical fingerprints")
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(keys, vec![item_key, other_key]);
    }

    #[test]
    fn vector_v4_round_trips_default_and_named_scopes_with_identical_local_keys()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let keys = [
            PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
            PhysicalIndexKey::vector(
                GraphPath::from_components(&["tenant:西/@idx1:"])?,
                "Doc",
                "embedding",
            ),
            PhysicalIndexKey::vector(GraphPath::from_components(&[""])?, "Doc", "embedding"),
        ];
        let source = [empty_hnsw(501), empty_hnsw(502), empty_hnsw(503)];
        let source_section = VectorStoreSection::new(vec![
            (keys[1].clone(), Arc::clone(&source[1])),
            (keys[0].clone(), Arc::clone(&source[0])),
            (keys[2].clone(), Arc::clone(&source[2])),
        ]);
        let bytes = source_section
            .serialize()
            .expect("serialize graph-qualified exact indexes");

        let restored = [empty_hnsw(901), empty_hnsw(902), empty_hnsw(903)];
        let mut restored_section = VectorStoreSection::for_unpublished_recovery(vec![
            (keys[2].clone(), Arc::clone(&restored[2])),
            (keys[0].clone(), Arc::clone(&restored[0])),
            (keys[1].clone(), Arc::clone(&restored[1])),
        ]);
        restored_section
            .deserialize(&bytes)
            .expect("restore every graph scope by canonical key");
        assert_eq!(
            restored_section.exact_state_fingerprints().unwrap(),
            source_section.exact_state_fingerprints().unwrap()
        );

        for (position, (source, restored)) in source.iter().zip(&restored).enumerate() {
            let id = NodeId::new(700 + position as u64);
            let vector = [position as f32, 0.25, -0.5, 1.0];
            let accessor = |_id: NodeId| -> Option<Arc<[f32]>> { None };
            source.insert(id, &vector, &accessor);
            restored.insert(id, &vector, &accessor);
        }
        assert_eq!(
            restored_section.exact_state_fingerprints().unwrap(),
            source_section.exact_state_fingerprints().unwrap(),
            "each graph-scoped RNG continuation must evolve identically"
        );

        let baseline = restored_section
            .exact_state_fingerprints()
            .expect("capture untouched target state");
        let mut wrong_graph = capture_v4(&source_section.indexes).expect("capture scoped fixture");
        wrong_graph.indexes[1].key = PhysicalIndexKey::vector(
            GraphPath::from_components(&["tenant:wrong"])?,
            "Doc",
            "embedding",
        );
        let error = restored_section
            .deserialize(&encode_v4_fixture(&wrong_graph))
            .expect_err("a graph scope is part of the authoritative key");
        assert!(error.to_string().contains("key set"), "{error}");
        assert_eq!(
            restored_section.exact_state_fingerprints().unwrap(),
            baseline,
            "wrong-graph admission must precede every target mutation"
        );
        Ok(())
    }

    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    #[test]
    fn vector_v4_exact_fork_is_frozen() {
        let quantized = Arc::new(VectorIndexKind::Quantized(QuantizedHnswIndex::with_seed(
            HnswConfig::new(4, DistanceMetric::Cosine),
            QuantizationType::Binary,
            51,
        )));
        quantized.insert(NodeId::new(1), &[1.0, 0.0, -1.0, 0.5], &|_| None);
        for (key, source) in [
            make_test_index(),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
                quantized,
            ),
        ] {
            let section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&source))]);
            let fork = source
                .fork_exact_read_snapshot()
                .expect("fork exact retained-reader state");
            let source_fingerprint = exact_fingerprint(&section);
            let fork_section = VectorStoreSection::new(vec![(key, Arc::clone(&fork))]);
            assert_eq!(exact_fingerprint(&fork_section), source_fingerprint);
            assert!(
                !fork.remove(NodeId::new(1)),
                "frozen clone must reject mutation"
            );
            assert_eq!(exact_fingerprint(&fork_section), source_fingerprint);
        }
    }

    #[test]
    fn vector_section_empty() {
        let section = VectorStoreSection::new(vec![]);
        let bytes = section.serialize().expect("serialize should succeed");

        let mut section2 = VectorStoreSection::for_unpublished_recovery(vec![]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");
    }

    #[test]
    fn vector_v4_restore_requires_explicit_unpublished_recovery_targets() {
        let (key, source) = make_test_index();
        let bytes = VectorStoreSection::new(vec![(key.clone(), source)])
            .serialize()
            .expect("serialize exact fixture");
        let target = empty_hnsw(222);
        let mut snapshot_only = VectorStoreSection::new(vec![(key, Arc::clone(&target))]);
        let error = snapshot_only
            .deserialize(&bytes)
            .expect_err("a snapshot-only section must not mutate a potentially published target");
        assert!(error.to_string().contains("unpublished recovery targets"));
        assert_eq!(target.len(), 0);
    }

    #[test]
    fn vector_current_payload_validation_fails_closed() {
        let section = VectorStoreSection::new(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
            topology_hnsw(19, 314),
        )]);
        let bytes = section.serialize().unwrap();
        VectorStoreSection::validate_payload(&bytes).unwrap();
        for length in 0..bytes.len() {
            assert!(
                VectorStoreSection::validate_payload(&bytes[..length]).is_err(),
                "truncated prefix {length} was accepted"
            );
        }
        for offset in 5..8 {
            let mut reserved = bytes.clone();
            reserved[offset] = 1;
            assert!(VectorStoreSection::validate_payload(&reserved).is_err());
        }
        let mut wrong_length = bytes.clone();
        wrong_length[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(VectorStoreSection::validate_payload(&wrong_length).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(VectorStoreSection::validate_payload(&trailing).is_err());
        for garbage in [b"junk".as_slice(), b"not a vector section", &[1, 253]] {
            assert!(VectorStoreSection::validate_payload(garbage).is_err());
        }
    }

    #[test]
    fn vector_v4_preflights_index_count_before_decoding_images() {
        let mut payload = vec![253];
        payload.extend_from_slice(&u64::MAX.to_le_bytes());
        let bytes = wrap_v4_payload(&payload);

        let target = topology_hnsw(77, 902);
        let before = target.snapshot_topology();
        let mut section = VectorStoreSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
            Arc::clone(&target),
        )]);
        let error = section
            .deserialize(&bytes)
            .expect_err("hostile count must fail before per-index decoding");
        assert!(error.to_string().contains("count"));
        assert!(VectorStoreSection::validate_payload(&bytes).is_err());
        assert_eq!(target.snapshot_topology(), before);
    }

    #[test]
    fn vector_v4_preflights_declared_key_length_before_serde_allocation() {
        let mut payload = vec![1, 253];
        payload.extend_from_slice(&u64::MAX.to_le_bytes());
        let bytes = wrap_v4_payload(&payload);

        let target = topology_hnsw(78, 903);
        let before = target.snapshot_topology();
        let mut section = VectorStoreSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
            Arc::clone(&target),
        )]);
        let error = section
            .deserialize(&bytes)
            .expect_err("hostile key length must fail before serde decoding");
        assert!(error.to_string().contains("index key"));
        assert!(VectorStoreSection::validate_payload(&bytes).is_err());
        assert_eq!(target.snapshot_topology(), before);
    }

    #[test]
    fn vector_section_missing_or_orphan_image_fails_closed() {
        let target = empty_hnsw(44);
        let mut missing = VectorStoreSection::new(vec![(
            PhysicalIndexKey::vector(GraphPath::root(), "Item", "embedding"),
            Arc::clone(&target),
        )]);
        let before = target.snapshot_topology();
        let error = missing
            .deserialize(&[])
            .expect_err("catalog target requires an authoritative image");
        assert!(error.to_string().contains("payload is missing"));
        assert_eq!(target.snapshot_topology(), before);

        let (key, source) = make_test_index();
        let orphan_bytes = VectorStoreSection::new(vec![(key, source)])
            .serialize()
            .expect("encode orphan fixture");
        let mut no_targets = VectorStoreSection::for_unpublished_recovery(vec![]);
        let error = no_targets
            .deserialize(&orphan_bytes)
            .expect_err("orphan authoritative image must not be ignored");
        assert!(error.to_string().contains("index count mismatch"));
    }

    #[test]
    fn vector_section_type() {
        let section = VectorStoreSection::new(vec![]);
        assert_eq!(section.section_type(), SectionType::VectorStore);
        assert_eq!(section.version(), 4);
    }

    #[test]
    fn vector_section_dirty_tracking() {
        let section = VectorStoreSection::new(vec![]);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn sealed_vector_section_restore_requires_the_store_authority() {
        let (key, source) = make_test_index();
        let bytes = VectorStoreSection::new(vec![(key.clone(), source)])
            .serialize()
            .expect("serialize fixture topology");

        let target = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            4,
            DistanceMetric::Cosine,
        ))));
        let store = LpgStore::new().unwrap();
        store.add_vector_index("Item", "embedding", Arc::clone(&target));
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(store.seal_unframed_writes(&owner));

        let restore = || {
            let mut section = VectorStoreSection::for_unpublished_recovery(vec![(
                key.clone(),
                Arc::clone(&target),
            )]);
            section.deserialize(&bytes)
        };
        let error = restore().expect_err("raw restore must fail closed");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.len(), 0);

        let error = with_authority(&foreign, restore)
            .expect_err("a foreign store authority must not restore topology");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.len(), 0);

        with_authority(&owner, || {
            restore().expect("owner may restore during recovery");
        });
        assert_eq!(target.len(), 3);

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || panic!("injected restore panic"));
        }));
        let error = restore().expect_err("caught panic must not retain authority");
        assert!(error.to_string().contains("lacks mutation authority"));
    }

    // ── Current format boundary ───────────────────────────────────

    /// New writes produce a v4 buffer (starts with `GVST` magic).
    #[test]
    fn alix_section_serialize_writes_v4_magic_and_version() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key, Arc::clone(&index))]);
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(bytes.len() > 4);
        assert_eq!(
            &bytes[0..4],
            VECTOR_MAGIC,
            "new writes must use packed magic"
        );
        assert_eq!(bytes[4], VECTOR_SECTION_VERSION);
    }

    /// v4 with multiple indexes round-trips by key, including indexes
    /// with different shapes.
    #[test]
    fn jules_section_v4_multiple_indexes_round_trip() {
        let cfg_a = HnswConfig::new(4, DistanceMetric::Cosine);
        let idx_a = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(cfg_a)));
        idx_a.restore_topology(
            Some(NodeId::new(10)),
            1,
            vec![
                (NodeId::new(10), vec![vec![NodeId::new(20)], vec![]]),
                (NodeId::new(20), vec![vec![NodeId::new(10)]]),
            ],
        );

        let cfg_b = HnswConfig::new(8, DistanceMetric::Euclidean);
        let idx_b = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(cfg_b)));
        idx_b.restore_topology(
            Some(NodeId::new(100)),
            0,
            vec![(NodeId::new(100), vec![vec![]])],
        );

        let section = VectorStoreSection::new(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
                Arc::clone(&idx_a),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "User", "embedding"),
                Arc::clone(&idx_b),
            ),
        ]);
        let bytes = section.serialize().expect("v4 serialize");

        // Restore into fresh indexes and verify topology counts and
        // entry points match.
        let restored_a = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            4,
            DistanceMetric::Cosine,
        ))));
        let restored_b = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            8,
            DistanceMetric::Euclidean,
        ))));
        let mut section2 = VectorStoreSection::for_unpublished_recovery(vec![
            (
                PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
                Arc::clone(&restored_a),
            ),
            (
                PhysicalIndexKey::vector(GraphPath::root(), "User", "embedding"),
                Arc::clone(&restored_b),
            ),
        ]);
        section2.deserialize(&bytes).expect("v4 load");

        assert_eq!(restored_a.len(), 2);
        assert_eq!(restored_b.len(), 1);
        let (ep_a, _, _) = restored_a.snapshot_topology();
        let (ep_b, _, _) = restored_b.snapshot_topology();
        assert_eq!(ep_a, Some(NodeId::new(10)));
        assert_eq!(ep_b, Some(NodeId::new(100)));
    }

    /// Truncated packed envelope is rejected without panicking.
    #[test]
    fn shosanna_section_truncated_v4_rejected() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&index))]);
        let bytes = section.serialize().expect("v4 serialize");

        // Truncate to less than the current header.
        let truncated = &bytes[..8];
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(
            index.config().clone(),
        )));
        let mut section2 = VectorStoreSection::for_unpublished_recovery(vec![(key, fresh)]);
        let err = section2
            .deserialize(truncated)
            .expect_err("must reject truncated v4");
        match err {
            Error::Serialization(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
