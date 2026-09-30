//! Text Index section serializer for the `.grafeo` container format.
//!
//! Serializes BM25 inverted indexes (postings lists, document lengths)
//! for all text indexes. Persisting avoids rebuilding from LPG properties
//! on database open.

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::utils::error::{Error, Result};

use crate::graph::lpg::{PhysicalIndexFamily, PhysicalIndexKey};
use grafeo_common::types::{
    EpochId, GraphPath, MAX_GRAPH_PATH_COMPONENTS, MAX_WORLD_GRAPH_NAME_BYTES, NodeId,
};

use super::inverted_index::{
    ExactAggDelta, ExactDocHistory, ExactDocLength, ExactPosting, ExactPostingList,
    ExactTextIndexImage, ExactTokenizerDescriptor, PreparedTextIndexImage,
};
use super::{InvertedIndex, TextIndexView, TextIndexWriteGuard};

/// Current graph-qualified exact text index section format version.
const TEXT_SECTION_VERSION: u8 = 5;

// ── Snapshot types ──────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
struct TextIndexSnapshot {
    version: u8,
    indexes: Vec<SingleIndexSnapshot>,
}

#[derive(Clone, Serialize, Deserialize)]
struct SingleIndexSnapshot {
    /// Canonical default- or named-graph LPG index key.
    key: PhysicalIndexKey,
    /// Complete committed index state, including MVCC history.
    image: ExactTextIndexImage,
}

struct DecodedIndex {
    key: PhysicalIndexKey,
    required_target_config: Bm25Descriptor,
    prepared: PreparedTextIndexImage,
}

struct PlannedRestore {
    key: PhysicalIndexKey,
    destination: usize,
    required_target_config: Bm25Descriptor,
    prepared: PreparedTextIndexImage,
}

#[derive(Clone)]
struct Bm25Descriptor {
    k1_bits: u64,
    b_bits: u64,
    tokenizer: ExactTokenizerDescriptor,
}

impl Bm25Descriptor {
    fn new(k1: f64, b: f64, tokenizer: ExactTokenizerDescriptor) -> Self {
        Self {
            k1_bits: k1.to_bits(),
            b_bits: b.to_bits(),
            tokenizer,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RestoreMode {
    SnapshotOnly,
    UnpublishedRecovery,
}

struct TextWirePreflight<'data> {
    data: &'data [u8],
    position: usize,
    claimed_heap: u64,
    format: &'static str,
}

impl<'data> TextWirePreflight<'data> {
    fn new(data: &'data [u8], format: &'static str) -> Self {
        Self {
            data,
            position: 0,
            claimed_heap: 0,
            format,
        }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    fn read_byte(&mut self, description: &str) -> std::result::Result<u8, String> {
        let byte = self
            .data
            .get(self.position)
            .copied()
            .ok_or_else(|| format!("Text Index {} {description} is truncated", self.format))?;
        self.position += 1;
        Ok(byte)
    }

    fn read_fixed<const N: usize>(&mut self, description: &str) -> std::result::Result<(), String> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| format!("Text Index {} {description} range overflows", self.format))?;
        if end > self.data.len() {
            return Err(format!(
                "Text Index {} {description} is truncated",
                self.format
            ));
        }
        self.position = end;
        Ok(())
    }

    fn read_array<const N: usize>(
        &mut self,
        description: &str,
    ) -> std::result::Result<[u8; N], String> {
        let start = self.position;
        self.read_fixed::<N>(description)?;
        self.data
            .get(start..self.position)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or_else(|| {
                format!(
                    "Text Index {} {description} has an invalid fixed-width field",
                    self.format
                )
            })
    }

    fn read_u64(&mut self, description: &str) -> std::result::Result<u64, String> {
        match self.read_byte(description)? {
            value @ 0..=250 => Ok(u64::from(value)),
            251 => {
                let value = u16::from_le_bytes(self.read_array(description)?);
                if value <= 250 {
                    return Err(format!(
                        "Text Index {} {description} has a non-canonical integer",
                        self.format
                    ));
                }
                Ok(u64::from(value))
            }
            252 => {
                let value = u32::from_le_bytes(self.read_array(description)?);
                if u16::try_from(value).is_ok() {
                    return Err(format!(
                        "Text Index {} {description} has a non-canonical integer",
                        self.format
                    ));
                }
                Ok(u64::from(value))
            }
            253 => {
                let value = u64::from_le_bytes(self.read_array(description)?);
                if u32::try_from(value).is_ok() {
                    return Err(format!(
                        "Text Index {} {description} has a non-canonical integer",
                        self.format
                    ));
                }
                Ok(value)
            }
            marker => Err(format!(
                "Text Index {} {description} has unsupported integer marker {marker}",
                self.format
            )),
        }
    }

    fn read_usize(&mut self, description: &str) -> std::result::Result<usize, String> {
        usize::try_from(self.read_u64(description)?)
            .map_err(|_| format!("Text Index {} {description} exceeds this host", self.format))
    }

    fn charge_heap(
        &mut self,
        count: usize,
        element_size: usize,
        description: &str,
    ) -> std::result::Result<(), String> {
        let bytes = u64::try_from(count)
            .ok()
            .and_then(|count| {
                u64::try_from(element_size)
                    .ok()
                    .and_then(|size| count.checked_mul(size))
            })
            .ok_or_else(|| {
                format!(
                    "Text Index {} {description} allocation overflows",
                    self.format
                )
            })?;
        self.claimed_heap = self
            .claimed_heap
            .checked_add(bytes)
            .ok_or_else(|| format!("Text Index {} decode heap overflows", self.format))?;
        Ok(())
    }

    fn read_sequence_len(
        &mut self,
        element_size: usize,
        description: &str,
    ) -> std::result::Result<usize, String> {
        let count = self.read_usize(description)?;
        if count > self.remaining() {
            return Err(format!(
                "Text Index {} {description} count {count} exceeds the {} remaining bytes",
                self.format,
                self.remaining()
            ));
        }
        self.charge_heap(count, element_size, description)?;
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
            .ok_or_else(|| format!("Text Index {} {description} range overflows", self.format))?;
        let bytes = self
            .data
            .get(self.position..end)
            .ok_or_else(|| format!("Text Index {} {description} is truncated", self.format))?;
        self.charge_heap(length, 1, description)?;
        self.position = end;
        std::str::from_utf8(bytes)
            .map_err(|_| format!("Text Index {} {description} is not UTF-8", self.format))
    }

    fn read_optional_u64(&mut self, description: &str) -> std::result::Result<(), String> {
        match self.read_byte(description)? {
            0 => Ok(()),
            1 => {
                let _value = self.read_u64(description)?;
                Ok(())
            }
            tag => Err(format!(
                "Text Index {} {description} has invalid option tag {tag}",
                self.format
            )),
        }
    }

    fn finish(self) -> std::result::Result<(), String> {
        if self.position != self.data.len() {
            return Err(format!(
                "Text Index {} payload has {} trailing bytes",
                self.format,
                self.data.len() - self.position
            ));
        }
        Ok(())
    }
}

fn preflight_exact_posting(
    wire: &mut TextWirePreflight<'_>,
) -> std::result::Result<NodeId, String> {
    let node_id = NodeId::new(wire.read_u64("posting node id")?);
    let _term_frequency = wire.read_u64("posting term frequency")?;
    let _created_epoch = wire.read_u64("posting created epoch")?;
    wire.read_optional_u64("posting creator transaction")?;
    wire.read_optional_u64("posting deleted epoch")?;
    wire.read_optional_u64("posting deleter transaction")?;
    Ok(node_id)
}

fn preflight_exact_image(
    wire: &mut TextWirePreflight<'_>,
    mut visit_node: impl FnMut(NodeId),
) -> std::result::Result<(), String> {
    wire.read_fixed::<8>("BM25 k1")?;
    wire.read_fixed::<8>("BM25 b")?;
    let tokenizer = wire.read_u64("tokenizer descriptor")?;
    if tokenizer != 0 {
        return Err(format!(
            "Text Index {} tokenizer variant {tokenizer} is unsupported",
            wire.format
        ));
    }
    let _minimum_length = wire.read_u64("tokenizer minimum length")?;
    let retained_from = wire.read_u64("retained-from epoch")?;
    if retained_from == EpochId::PENDING.as_u64() {
        return Err(format!(
            "Text Index {} retained-from epoch must be committed",
            wire.format
        ));
    }

    let posting_list_count =
        wire.read_sequence_len(std::mem::size_of::<ExactPostingList>(), "posting lists")?;
    for _ in 0..posting_list_count {
        let _term = wire.read_string("posting term")?;
        let posting_count =
            wire.read_sequence_len(std::mem::size_of::<ExactPosting>(), "postings")?;
        for _ in 0..posting_count {
            visit_node(preflight_exact_posting(wire)?);
        }
    }

    let document_count =
        wire.read_sequence_len(std::mem::size_of::<ExactDocHistory>(), "document histories")?;
    for _ in 0..document_count {
        visit_node(NodeId::new(wire.read_u64("document node id")?));
        let history_count = wire.read_sequence_len(
            std::mem::size_of::<ExactDocLength>(),
            "document length history",
        )?;
        for _ in 0..history_count {
            let _length = wire.read_u64("document length")?;
            let _created_epoch = wire.read_u64("document created epoch")?;
            wire.read_optional_u64("document creator transaction")?;
            wire.read_optional_u64("document deleted epoch")?;
            wire.read_optional_u64("document deleter transaction")?;
        }
    }

    let aggregate_count =
        wire.read_sequence_len(std::mem::size_of::<ExactAggDelta>(), "aggregate log")?;
    for _ in 0..aggregate_count {
        let _epoch = wire.read_u64("aggregate epoch")?;
        wire.read_optional_u64("aggregate transaction")?;
        let _total_length_delta = wire.read_u64("aggregate total-length delta")?;
        let _document_count_delta = wire.read_u64("aggregate document-count delta")?;
    }
    Ok(())
}

fn preflight_text_wire(data: &[u8]) -> std::result::Result<(), String> {
    visit_text_wire(data, |_, _| Ok(()))
}

fn visit_text_wire(
    data: &[u8],
    visit: impl FnMut(&PhysicalIndexKey, Range<usize>) -> std::result::Result<(), String>,
) -> std::result::Result<(), String> {
    visit_text_wire_nodes(data, visit, |_, _| {})
}

fn visit_text_wire_nodes(
    data: &[u8],
    mut visit: impl FnMut(&PhysicalIndexKey, Range<usize>) -> std::result::Result<(), String>,
    mut visit_node: impl FnMut(&PhysicalIndexKey, NodeId),
) -> std::result::Result<(), String> {
    let mut wire = TextWirePreflight::new(data, "v5");
    let wire_version = wire.read_byte("version")?;
    if wire_version != TEXT_SECTION_VERSION {
        return Err(format!(
            "unsupported Text Index section payload version {wire_version}"
        ));
    }
    let index_count =
        wire.read_sequence_len(std::mem::size_of::<DecodedIndex>(), "index image list")?;
    let mut previous = None;
    for _ in 0..index_count {
        let start = wire.position;
        let key = wire.read_physical_key()?;
        if key.family() != PhysicalIndexFamily::Text {
            return Err(format!("Text section has invalid index key family {key:?}"));
        }
        if previous.as_ref().is_some_and(|prior| prior >= &key) {
            return Err(format!(
                "duplicate index key or non-canonical key order at {key:?}"
            ));
        }
        preflight_exact_image(&mut wire, |node| visit_node(&key, node))?;
        visit(&key, start..wire.position)?;
        previous = Some(key);
    }
    wire.finish()
}

// ── Section implementation ──────────────────────────────────────────

/// Text Index section for the `.grafeo` container.
///
/// Current graph-qualified installation is failure-before-mutation for writers,
/// but existing readers do not take one section-wide publication lock. It is
/// therefore accepted only through [`Self::for_unpublished_recovery`] or
/// [`Self::for_unpublished_recovery_views`], while every target remains private
/// to recovery. Every non-current payload version is rejected before decode or
/// target mutation.
pub struct TextIndexSection {
    indexes: Vec<(PhysicalIndexKey, TextIndexView)>,
    dirty: AtomicBool,
    restore_mode: RestoreMode,
}

impl TextIndexSection {
    /// Create a new Text Index section from the current indexes.
    pub fn new(indexes: Vec<(PhysicalIndexKey, Arc<RwLock<InvertedIndex>>)>) -> Self {
        Self::from_views(
            indexes
                .into_iter()
                .map(|(key, index)| (key, TextIndexView::new(index)))
                .collect(),
        )
    }

    /// Creates a section from the capability-reduced handles returned by an
    /// LPG store.
    pub fn from_views(indexes: Vec<(PhysicalIndexKey, TextIndexView)>) -> Self {
        Self {
            indexes,
            dirty: AtomicBool::new(false),
            restore_mode: RestoreMode::SnapshotOnly,
        }
    }

    /// Creates a section whose targets remain private until recovery
    /// publishes the surrounding database state.
    ///
    /// Every supplied target must be unreachable by readers until
    /// [`Section::deserialize`] succeeds and the surrounding database state
    /// publishes all restored indexes together.
    #[must_use]
    pub fn for_unpublished_recovery(
        indexes: Vec<(PhysicalIndexKey, Arc<RwLock<InvertedIndex>>)>,
    ) -> Self {
        Self::for_unpublished_recovery_views(
            indexes
                .into_iter()
                .map(|(key, index)| (key, TextIndexView::new(index)))
                .collect(),
        )
    }

    /// Creates an unpublished-recovery section from capability-reduced views.
    #[must_use]
    pub fn for_unpublished_recovery_views(indexes: Vec<(PhysicalIndexKey, TextIndexView)>) -> Self {
        Self {
            indexes,
            dirty: AtomicBool::new(false),
            restore_mode: RestoreMode::UnpublishedRecovery,
        }
    }

    /// Validates the one current Text Index payload.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing or unsupported payload.
    pub fn validate_payload(data: &[u8]) -> Result<()> {
        match data.first().copied() {
            Some(TEXT_SECTION_VERSION) => {
                preflight_text_wire(data).map_err(Error::Serialization)?;
                Ok(())
            }
            Some(version) => Err(Error::Serialization(format!(
                "unsupported Text Index section payload version {version}"
            ))),
            None => Err(Error::Serialization(
                "Text Index section payload is missing".to_string(),
            )),
        }
    }

    /// Returns the exact sorted physical key set from a bounded current image.
    ///
    /// # Errors
    /// Rejects unsupported or malformed wire data and duplicate/wrong-family keys.
    pub fn payload_keys(data: &[u8]) -> Result<Vec<PhysicalIndexKey>> {
        Self::validate_payload(data)?;
        let snapshot = decode_current_wire(data)?;
        let mut keys: Vec<_> = snapshot
            .indexes
            .into_iter()
            .map(|index| index.key)
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Returns canonical owner keys and their complete encoded entry ranges.
    ///
    /// Ranges include each key and exact image, relative to the complete input.
    /// The whole current wire is preflighted without decoding posting DTOs.
    /// Target-dependent exact-image validation remains part of recovery.
    ///
    /// # Errors
    /// Rejects unsupported, malformed, truncated or non-canonical wire, and
    /// allocation failures while collecting bounded entry metadata.
    pub fn payload_entry_ranges(data: &[u8]) -> Result<Vec<(PhysicalIndexKey, Range<usize>)>> {
        let mut entries = Vec::new();
        visit_text_wire(data, |key, range| {
            entries.try_reserve(1).map_err(|error| {
                format!("Text Index v5 entry metadata allocation failed: {error}")
            })?;
            entries.push((key.clone(), range));
            Ok(())
        })
        .map_err(Error::Serialization)?;
        Ok(entries)
    }

    /// Reports whether any serialized node reference satisfies `is_selected`.
    ///
    /// Visits every posting (including historical and deleted postings) and
    /// every document-history owner, including documents with no tokens. The
    /// complete bounded current wire is checked even after a match, without
    /// decoding posting DTOs or building indexes. This is wire inspection, not
    /// the semantic exact-image validation performed during recovery.
    ///
    /// # Errors
    /// Rejects unsupported, malformed, truncated or non-canonical wire even
    /// when a matching reference was encountered earlier in the payload.
    pub fn payload_references_nodes(
        data: &[u8],
        mut is_selected: impl FnMut(&PhysicalIndexKey, NodeId) -> bool,
    ) -> Result<bool> {
        let mut matched = false;
        visit_text_wire_nodes(
            data,
            |_, _| Ok(()),
            |key, node| {
                matched |= is_selected(key, node);
            },
        )
        .map_err(Error::Serialization)?;
        Ok(matched)
    }

    /// Selects exact owner entries without decoding or rebuilding their images.
    ///
    /// `keys` must be sorted, unique and present in the current payload. The
    /// output has a current version/count header and byte-identical entries.
    /// An empty selection produces the valid empty current Text section.
    ///
    /// # Errors
    /// Rejects invalid input wire, unknown/duplicate/unsorted requested keys,
    /// size overflow and output allocation failures before copying any entry.
    pub fn select_payload_keys(data: &[u8], keys: &[PhysicalIndexKey]) -> Result<Vec<u8>> {
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::Serialization(
                "Text Index v5 selection keys must be sorted and unique".into(),
            ));
        }
        let entries = Self::payload_entry_ranges(data)?;
        let entry_range = |key: &PhysicalIndexKey| -> Result<&Range<usize>> {
            let position = entries
                .binary_search_by(|(entry, _)| entry.cmp(key))
                .map_err(|_| {
                    Error::Serialization(format!("Text Index v5 selection key is absent: {key:?}"))
                })?;
            entries
                .get(position)
                .map(|(_, range)| range)
                .ok_or_else(|| {
                    Error::Serialization("Text Index v5 selection entry is absent".into())
                })
        };
        let count = u64::try_from(keys.len())
            .map_err(|_| Error::Serialization("Text Index v5 selection count overflows".into()))?;
        // One version byte and at most nine bytes for the canonical u64 count.
        let mut header = [0_u8; 10];
        let header_length = bincode::serde::encode_into_slice(
            (TEXT_SECTION_VERSION, count),
            &mut header,
            bincode::config::standard(),
        )
        .map_err(|error| {
            Error::Serialization(format!("Text Index v5 selection header failed: {error}"))
        })?;
        let length = keys.iter().try_fold(header_length, |length, key| {
            length.checked_add(entry_range(key)?.len()).ok_or_else(|| {
                Error::Serialization("Text Index v5 selection size overflows".into())
            })
        })?;
        let mut selected = Vec::new();
        selected.try_reserve_exact(length).map_err(|error| {
            Error::Serialization(format!(
                "Text Index v5 selection allocation failed: {error}"
            ))
        })?;
        let header = header.get(..header_length).ok_or_else(|| {
            Error::Serialization("Text Index v5 selection header range is invalid".into())
        })?;
        selected.extend_from_slice(header);
        for key in keys {
            let bytes = data.get(entry_range(key)?.clone()).ok_or_else(|| {
                Error::Serialization("Text Index v5 selection entry range is invalid".into())
            })?;
            selected.extend_from_slice(bytes);
        }
        Ok(selected)
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    fn install_decoded(&mut self, format: &str, decoded: Vec<DecodedIndex>) -> Result<()> {
        if decoded.len() != self.indexes.len() {
            return Err(Error::Serialization(format!(
                "Text Index {format} index count mismatch: snapshot {}, target {}",
                decoded.len(),
                self.indexes.len()
            )));
        }
        let decoded_keys = canonical_key_map(
            decoded
                .iter()
                .enumerate()
                .map(|(position, index)| (position, &index.key)),
        )?;
        let target_keys = canonical_key_map(
            self.indexes
                .iter()
                .enumerate()
                .map(|(position, (key, _))| (position, key)),
        )?;
        if decoded_keys.keys().ne(target_keys.keys()) {
            let orphan = decoded_keys
                .keys()
                .filter(|key| !target_keys.contains_key(*key))
                .cloned()
                .collect::<Vec<_>>();
            let missing = target_keys
                .keys()
                .filter(|key| !decoded_keys.contains_key(*key))
                .cloned()
                .collect::<Vec<_>>();
            return Err(Error::Serialization(format!(
                "Text Index {format} authoritative key set does not match recovery targets (orphan {orphan:?}, missing {missing:?})"
            )));
        }

        validate_target_aliases(format, &self.indexes, &target_keys)?;

        // Iterating the canonical map establishes one stable, process-wide
        // lock order independent of payload and registry insertion order.
        let mut decoded = decoded.into_iter().map(Some).collect::<Vec<_>>();
        let mut planned = Vec::with_capacity(decoded.len());
        for (key, decoded_position) in decoded_keys {
            let decoded = decoded
                .get_mut(decoded_position)
                .and_then(Option::take)
                .ok_or_else(|| {
                    Error::Serialization(format!(
                        "Text Index {format} decoded key map is inconsistent at {key:?}"
                    ))
                })?;
            let destination = target_keys.get(&key).copied().ok_or_else(|| {
                Error::Serialization(format!("Text Index {format} target map lost key {key:?}"))
            })?;
            planned.push(PlannedRestore {
                destination,
                key,
                required_target_config: decoded.required_target_config,
                prepared: decoded.prepared,
            });
        }

        // Validate and allocate every payload before this point, then retain
        // all destination locks and all store-scoped mutation proofs before
        // changing the first index. An error therefore leaves every destination
        // on its pre-restore image.
        let mut guards = planned
            .iter()
            .map(|restore| {
                let view = self
                    .indexes
                    .get(restore.destination)
                    .map(|(_, view)| view)
                    .ok_or_else(|| {
                        Error::Serialization(format!(
                            "Text Index {format} target position {} is out of range",
                            restore.destination
                        ))
                    })?;
                Ok(TextIndexWriteGuard::acquire(
                    Arc::clone(&view.gate),
                    view.target.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let proofs = guards
            .iter()
            .zip(&planned)
            .map(|(index, restore)| {
                index.pin_prepared_restore().ok_or_else(|| {
                    Error::Serialization(format!(
                        "Text Index restore for key {:?} lacks mutation authority",
                        restore.key
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        for (index, restore) in guards.iter().zip(&planned) {
            let required = &restore.required_target_config;
            let actual = index.config();
            if actual.k1.to_bits() != required.k1_bits || actual.b.to_bits() != required.b_bits {
                return Err(Error::Serialization(format!(
                    "Text Index v5 BM25 descriptor/image mismatch for key {:?}: target k1={} b={}, image k1={} b={}",
                    restore.key,
                    actual.k1,
                    actual.b,
                    f64::from_bits(required.k1_bits),
                    f64::from_bits(required.b_bits),
                )));
            }
            if !index.tokenizer_matches(&required.tokenizer) {
                return Err(Error::Serialization(format!(
                    "Text Index v5 tokenizer descriptor/image mismatch for key {:?}",
                    restore.key
                )));
            }
        }

        for ((restore, index), proof) in planned.into_iter().zip(&mut guards).zip(&proofs) {
            index.install_prepared_image(restore.prepared, proof);
        }
        Ok(())
    }
}

impl Section for TextIndexSection {
    fn section_type(&self) -> SectionType {
        SectionType::TextIndex
    }

    fn version(&self) -> u8 {
        TEXT_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let keys = canonical_key_map(
            self.indexes
                .iter()
                .enumerate()
                .map(|(position, (key, _))| (position, key)),
        )?;
        validate_target_aliases("v5 snapshot", &self.indexes, &keys)?;

        let indexes: Vec<SingleIndexSnapshot> = keys
            .into_iter()
            .map(|(key, position)| {
                let index_lock = &self.indexes[position].1;
                let index = index_lock.read();
                let image = index.exact_committed_image().map_err(|reason| {
                    Error::Serialization(format!(
                        "text index {key:?} cannot be serialized exactly: {reason}"
                    ))
                })?;
                Ok(SingleIndexSnapshot { key, image })
            })
            .collect::<Result<_>>()?;

        let snapshot = TextIndexSnapshot {
            version: TEXT_SECTION_VERSION,
            indexes,
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Serialization(format!("Text Index v5 encoding failed: {e}")))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::Serialization(format!(
                "Text Index section is missing for {} catalog target(s)",
                self.indexes.len()
            )));
        }
        Self::validate_payload(data)?;
        if self.restore_mode != RestoreMode::UnpublishedRecovery {
            return Err(Error::Serialization(
                "Text Index v5 exact restore requires explicitly unpublished recovery targets"
                    .to_string(),
            ));
        }
        self.install_decoded("v5", decode_current(data)?)
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
            .map(|(_, idx)| idx.read().heap_memory_bytes())
            .sum()
    }
}

fn canonical_key_map<'key>(
    keys: impl IntoIterator<Item = (usize, &'key PhysicalIndexKey)>,
) -> Result<BTreeMap<PhysicalIndexKey, usize>> {
    let mut canonical = BTreeMap::new();
    for (position, key) in keys {
        if key.family() != PhysicalIndexFamily::Text {
            return Err(Error::Serialization(format!(
                "Text Index section contains invalid index key family {key:?}"
            )));
        }
        if canonical.insert(key.clone(), position).is_some() {
            return Err(Error::Serialization(format!(
                "Text Index section contains duplicate index key {key:?}"
            )));
        }
    }
    Ok(canonical)
}

fn validate_target_aliases(
    format: &str,
    indexes: &[(PhysicalIndexKey, TextIndexView)],
    targets: &BTreeMap<PhysicalIndexKey, usize>,
) -> Result<()> {
    let mut gates = HashSet::with_capacity(indexes.len());
    let mut concrete = HashSet::with_capacity(indexes.len());
    for (key, position) in targets {
        let view = &indexes[*position].1;
        if !gates.insert(Arc::as_ptr(&view.gate) as usize) {
            return Err(Error::Serialization(format!(
                "Text Index {format} target key {key:?} aliases another index gate"
            )));
        }
        if let Some(target) = &view.target
            && !concrete.insert(Arc::as_ptr(target) as usize)
        {
            return Err(Error::Serialization(format!(
                "Text Index {format} target key {key:?} aliases another physical index"
            )));
        }
    }
    Ok(())
}

fn decode_current_wire(data: &[u8]) -> Result<TextIndexSnapshot> {
    let config = bincode::config::standard();
    let (snapshot, consumed): (TextIndexSnapshot, _) =
        bincode::serde::decode_from_slice(data, config).map_err(|error| {
            Error::Serialization(format!("Text Index v5 payload is corrupt: {error}"))
        })?;
    if consumed != data.len() {
        return Err(Error::Serialization(format!(
            "Text Index v5 payload has {} trailing bytes",
            data.len() - consumed
        )));
    }
    if snapshot.version != TEXT_SECTION_VERSION {
        return Err(Error::Serialization(format!(
            "Text Index v5 decoder received payload version {}",
            snapshot.version
        )));
    }
    canonical_key_map(
        snapshot
            .indexes
            .iter()
            .enumerate()
            .map(|(position, index)| (position, &index.key)),
    )?;
    Ok(snapshot)
}

fn decode_current(data: &[u8]) -> Result<Vec<DecodedIndex>> {
    decode_current_wire(data)?
        .indexes
        .into_iter()
        .map(|index| {
            let required_target_config =
                Bm25Descriptor::new(index.image.k1, index.image.b, index.image.tokenizer.clone());
            let prepared = InvertedIndex::prepare_exact_image(index.image).map_err(|reason| {
                Error::Serialization(format!(
                    "Text Index v5 image {:?} is invalid: {reason}",
                    index.key
                ))
            })?;
            Ok(DecodedIndex {
                key: index.key,
                required_target_config,
                prepared,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "lpg")]
    use crate::graph::lpg::LpgStore;
    #[cfg(feature = "lpg")]
    use crate::graph::write_permit::{WriteAuthority, with_authority};
    use crate::index::text::{BM25Config, SimpleTokenizer};
    use grafeo_common::types::{EpochId, NodeId, TransactionId};
    #[cfg(feature = "lpg")]
    use std::{sync::mpsc, thread, time::Duration};

    fn decode_v5_fixture(bytes: &[u8]) -> TextIndexSnapshot {
        let (snapshot, consumed): (TextIndexSnapshot, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());
        snapshot
    }

    fn encode_v5_fixture(snapshot: &TextIndexSnapshot) -> Vec<u8> {
        bincode::serde::encode_to_vec(snapshot, bincode::config::standard()).unwrap()
    }

    fn exact_image_bytes(index: &Arc<RwLock<InvertedIndex>>) -> Vec<u8> {
        let image = index
            .read()
            .exact_committed_image()
            .expect("test index has an exact committed image");
        bincode::serde::encode_to_vec(image, bincode::config::standard()).unwrap()
    }

    fn payload_references_fixture() -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut root = InvertedIndex::with_simple_tokenizer(BM25Config::default(), 3);
        for (id, text, epoch) in [
            (7, "deleted historical document", 1),
            (9, "older version", 3),
            (9, "current version", 5),
        ] {
            root.insert_versioned(NodeId::new(id), text, EpochId::new(epoch), None);
        }
        assert!(root.remove_versioned(NodeId::new(7), EpochId::new(6), None));
        let mut named = InvertedIndex::new(BM25Config::default());
        named.insert_versioned(NodeId::new(7), "same identity", EpochId::new(1), None);
        named.insert_versioned(NodeId::new(99), "named only", EpochId::new(2), None);
        Ok(TextIndexSection::new(vec![
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                Arc::new(RwLock::new(root)),
            ),
            (
                PhysicalIndexKey::text(GraphPath::from_components(&["named"])?, "Doc", "body"),
                Arc::new(RwLock::new(named)),
            ),
        ])
        .serialize()?)
    }

    #[test]
    fn payload_references_visits_every_owner_qualified_reference_without_changing_bytes()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let bytes = payload_references_fixture()?;
        let before = bytes.clone();
        let snapshot = decode_v5_fixture(&bytes);
        let expected: Vec<_> = snapshot
            .indexes
            .iter()
            .flat_map(|index| {
                index
                    .image
                    .postings
                    .iter()
                    .flat_map(|list| list.postings.iter().map(|posting| posting.node_id))
                    .chain(
                        index
                            .image
                            .doc_lengths
                            .iter()
                            .map(|document| document.node_id),
                    )
                    .map(move |node| (index.key.clone(), node))
            })
            .collect();
        let mut visited = Vec::new();
        assert!(TextIndexSection::payload_references_nodes(
            &bytes,
            |key, node| {
                visited.push((key.clone(), node));
                true
            }
        )?);
        assert_eq!(visited, expected, "a match must not skip later references");
        for root_only in [false, true] {
            assert!(TextIndexSection::payload_references_nodes(
                &bytes,
                |key, node| {
                    key.graph().components().is_empty() == root_only && node == NodeId::new(7)
                }
            )?);
        }
        assert!(!TextIndexSection::payload_references_nodes(
            &bytes,
            |key, node| { key.graph().components().is_empty() && node == NodeId::new(99) }
        )?);
        assert!(TextIndexSection::payload_references_nodes(
            &bytes,
            |key, node| { !key.graph().components().is_empty() && node == NodeId::new(99) }
        )?);
        assert!(!TextIndexSection::payload_references_nodes(
            &bytes,
            |_, node| node == NodeId::new(100)
        )?);
        assert!(!TextIndexSection::payload_references_nodes(
            &bytes,
            |_, _| false
        )?);
        let empty = TextIndexSection::new(Vec::new()).serialize()?;
        assert!(!TextIndexSection::payload_references_nodes(
            &empty,
            |_, _| true
        )?);
        assert_eq!(bytes, before);
        Ok(())
    }

    #[test]
    fn payload_references_includes_deleted_historical_and_empty_token_documents()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let bytes = payload_references_fixture()?;
        let mut snapshot = decode_v5_fixture(&bytes);
        let root = snapshot
            .indexes
            .iter_mut()
            .find(|index| index.key.graph().components().is_empty())
            .ok_or("missing root fixture")?;
        assert!(
            root.image
                .postings
                .iter()
                .flat_map(|list| &list.postings)
                .any(|posting| {
                    posting.node_id == NodeId::new(7)
                        && posting.deleted_epoch == Some(EpochId::new(6))
                })
        );
        assert!(
            root.image
                .postings
                .iter()
                .flat_map(|list| &list.postings)
                .any(|posting| {
                    posting.node_id == NodeId::new(9)
                        && posting.deleted_epoch == Some(EpochId::new(5))
                })
        );
        assert!(
            !root
                .image
                .postings
                .iter()
                .flat_map(|list| &list.postings)
                .any(|posting| posting.node_id == NodeId::new(8))
        );
        // Current insertion skips empty-token input and semantic recovery
        // rejects zero-length histories. Wire inspection must nevertheless
        // find every encoded document owner without relying on either rule.
        root.image.doc_lengths.push(ExactDocHistory {
            node_id: NodeId::new(8),
            history: vec![ExactDocLength {
                len: 0,
                created_epoch: EpochId::new(2),
                created_by: None,
                deleted_epoch: Some(EpochId::new(7)),
                deleted_by: None,
            }],
        });
        root.image
            .doc_lengths
            .sort_by_key(|document| document.node_id);
        let empty_document = root
            .image
            .doc_lengths
            .iter()
            .find(|document| document.node_id == NodeId::new(8))
            .ok_or("missing empty-token document")?;
        assert!(
            empty_document
                .history
                .iter()
                .any(|entry| entry.len == 0 && entry.deleted_epoch == Some(EpochId::new(7)))
        );
        let bytes = encode_v5_fixture(&snapshot);
        assert!(
            decode_current(&bytes).is_err(),
            "wire scanning is not a semantic restore proof"
        );
        for id in [7, 8, 9] {
            assert!(TextIndexSection::payload_references_nodes(
                &bytes,
                |key, node| { key.graph().components().is_empty() && node == NodeId::new(id) }
            )?);
        }
        Ok(())
    }

    #[test]
    fn payload_references_rejects_unselected_malformed_tail_after_earlier_match()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let bytes = payload_references_fixture()?;
        let before = bytes.clone();
        let ranges = TextIndexSection::payload_entry_ranges(&bytes)?;
        let first_end = ranges[0].1.end;
        for end in 0..bytes.len() {
            let mut matched = false;
            let result = TextIndexSection::payload_references_nodes(&bytes[..end], |key, _| {
                let selected = key.graph().components().is_empty();
                matched |= selected;
                selected
            });
            assert!(result.is_err(), "truncated payload at {end} must fail");
            if end >= first_end {
                assert!(
                    matched,
                    "the complete earlier root image contains references"
                );
            }
        }
        let (last_key, last_range) = ranges.last().ok_or("missing named fixture")?;
        let key_size = bincode::serde::encode_to_vec(last_key, bincode::config::standard())?.len();
        let mut malformed_tokenizer = bytes.clone();
        malformed_tokenizer[last_range.start + key_size + 16] = 1;
        let mut trailing = bytes.clone();
        trailing.push(0xff);
        for malformed in [malformed_tokenizer, trailing] {
            let mut matched = false;
            assert!(
                TextIndexSection::payload_references_nodes(&malformed, |key, _| {
                    let selected = key.graph().components().is_empty();
                    matched |= selected;
                    selected
                })
                .is_err()
            );
            assert!(matched, "an earlier match must not hide an invalid tail");
        }
        for malformed in [vec![4, 0], vec![5, 251, 0, 0], vec![5, 253, 255]] {
            assert!(TextIndexSection::payload_references_nodes(&malformed, |_, _| false).is_err());
        }
        assert_eq!(bytes, before);
        Ok(())
    }

    #[test]
    fn payload_owner_ranges_select_exact_temporal_images_and_configuration()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut sources = Vec::new();
        for (path, config, minimum) in [
            (GraphPath::root(), BM25Config { k1: 1.3, b: 0.6 }, 3),
            (
                GraphPath::from_components(&["a", "", "b"])?,
                BM25Config { k1: 1.7, b: 0.4 },
                4,
            ),
            (
                GraphPath::from_components(&["a//b"])?,
                BM25Config { k1: 1.9, b: 0.8 },
                5,
            ),
        ] {
            let mut index = InvertedIndex::with_simple_tokenizer(config, minimum);
            index.insert_versioned(
                NodeId::new(11),
                "initial retained needle",
                EpochId::new(1),
                None,
            );
            index.insert_versioned(
                NodeId::new(12),
                "deleted retained document",
                EpochId::new(2),
                None,
            );
            index.insert_versioned(
                NodeId::new(11),
                "current retained needle",
                EpochId::new(5),
                None,
            );
            index.remove_versioned(NodeId::new(12), EpochId::new(7), None);
            index.gc(EpochId::new(3))?;
            sources.push((
                PhysicalIndexKey::text(path, "Doc", "body"),
                Arc::new(RwLock::new(index)),
            ));
        }
        let bytes = TextIndexSection::new(sources.clone()).serialize()?;
        let before = bytes.clone();
        let ranges = TextIndexSection::payload_entry_ranges(&bytes)?;
        let decoded = decode_v5_fixture(&bytes);
        assert_eq!(ranges.len(), sources.len());
        for ((key, range), entry) in ranges.iter().zip(&decoded.indexes) {
            assert_eq!(key, &entry.key);
            assert_eq!(
                bytes
                    .get(range.clone())
                    .ok_or("entry range is out of bounds")?,
                bincode::serde::encode_to_vec(entry, bincode::config::standard())?
            );
        }
        let requested = vec![sources[0].0.clone(), sources[2].0.clone()];
        let selected = TextIndexSection::select_payload_keys(&bytes, &requested)?;
        assert_eq!(
            selected,
            TextIndexSection::new(vec![sources[0].clone(), sources[2].clone()]).serialize()?
        );
        let restored = Arc::new(RwLock::new(InvertedIndex::with_simple_tokenizer(
            BM25Config { k1: 1.9, b: 0.8 },
            5,
        )));
        let single = TextIndexSection::select_payload_keys(&selected, &requested[1..])?;
        TextIndexSection::for_unpublished_recovery(vec![(
            requested[1].clone(),
            Arc::clone(&restored),
        )])
        .deserialize(&single)?;
        assert_eq!(restored.read().retained_from(), EpochId::new(3));
        assert_eq!(
            exact_image_bytes(&restored),
            exact_image_bytes(&sources[2].1)
        );
        for epoch in [3, 5, 7] {
            assert_eq!(
                restored
                    .read()
                    .score_document_visible(
                        NodeId::new(11),
                        "initial",
                        EpochId::new(epoch),
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits),
                sources[2]
                    .1
                    .read()
                    .score_document_visible(
                        NodeId::new(11),
                        "initial",
                        EpochId::new(epoch),
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits)
            );
        }
        assert_eq!(bytes, before);
        assert_eq!(TextIndexSection::new(sources).serialize()?, before);
        Ok(())
    }

    #[test]
    fn payload_owner_selection_rejects_invalid_requests_and_all_truncated_wire()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let first = PhysicalIndexKey::text(GraphPath::root(), "Doc", "first");
        let second = PhysicalIndexKey::text(GraphPath::root(), "Doc", "second");
        let bytes = TextIndexSection::new(vec![
            (
                first.clone(),
                Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
            ),
            (
                second.clone(),
                Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
            ),
        ])
        .serialize()?;
        let before = bytes.clone();
        for keys in [
            vec![first.clone(), first.clone()],
            vec![second, first.clone()],
            vec![PhysicalIndexKey::text(GraphPath::root(), "Doc", "unknown")],
            vec![PhysicalIndexKey::property(GraphPath::root(), "first")],
        ] {
            assert!(TextIndexSection::select_payload_keys(&bytes, &keys).is_err());
        }
        for end in 0..bytes.len() {
            assert!(TextIndexSection::payload_entry_ranges(&bytes[..end]).is_err());
            assert!(
                TextIndexSection::select_payload_keys(&bytes[..end], std::slice::from_ref(&first))
                    .is_err()
            );
        }
        let mut trailing = bytes.clone();
        trailing.push(0xff);
        assert!(TextIndexSection::select_payload_keys(&trailing, &[]).is_err());
        let mut reversed = decode_v5_fixture(&bytes);
        reversed.indexes.reverse();
        let reversed = encode_v5_fixture(&reversed);
        assert!(TextIndexSection::payload_entry_ranges(&reversed).is_err());
        for malformed in [vec![4, 0], vec![5, 251, 0, 0], vec![5, 253, 255]] {
            assert!(TextIndexSection::payload_entry_ranges(&malformed).is_err());
            assert!(TextIndexSection::select_payload_keys(&malformed, &[]).is_err());
        }
        assert_eq!(bytes, before);
        Ok(())
    }

    #[test]
    fn payload_owner_selection_preserves_canonical_count_boundaries()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let sources: Vec<_> = (0..252)
            .map(|index| {
                (
                    PhysicalIndexKey::text(
                        GraphPath::root(),
                        "Doc",
                        format!("property-{index:03}"),
                    ),
                    Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
                )
            })
            .collect();
        let bytes = TextIndexSection::new(sources.clone()).serialize()?;
        let keys: Vec<_> = sources.iter().map(|(key, _)| key.clone()).collect();
        assert_eq!(TextIndexSection::select_payload_keys(&bytes, &keys)?, bytes);
        for count in [0, 250, 251] {
            let selected = TextIndexSection::select_payload_keys(&bytes, &keys[..count])?;
            assert_eq!(
                selected,
                TextIndexSection::new(sources[..count].to_vec()).serialize()?
            );
            assert_eq!(
                TextIndexSection::payload_entry_ranges(&selected)?.len(),
                count
            );
        }
        assert_eq!(
            TextIndexSection::select_payload_keys(&bytes, &[])?,
            vec![5, 0]
        );
        Ok(())
    }

    #[test]
    fn typed_paths_round_trip_without_coordinate_aliases()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let paths: &[&[&str]] = &[&[], &[""], &["default"], &["a/b"], &["a", "b"]];
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for components in paths {
            let key =
                PhysicalIndexKey::text(GraphPath::from_components(components)?, "Doc", "body");
            let mut index = InvertedIndex::new(BM25Config::default());
            index.insert(NodeId::new(1), &format!("document {components:?}"));
            sources.push((key.clone(), Arc::new(RwLock::new(index))));
            targets.push((
                key,
                Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
            ));
        }
        let source = TextIndexSection::new(sources);
        let bytes = source.serialize()?;
        let mut target = TextIndexSection::for_unpublished_recovery(targets);
        target.deserialize(&bytes)?;
        assert_eq!(target.serialize()?, bytes);
        let mut shuffled = decode_v5_fixture(&bytes);
        shuffled.indexes.swap(0, 1);
        assert!(target.deserialize(&encode_v5_fixture(&shuffled)).is_err());
        assert_eq!(target.serialize()?, bytes);
        Ok(())
    }

    #[test]
    fn late_tokenizer_owner_mismatch_preserves_every_target()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let make = |minimum, text: &str| {
            let mut index = InvertedIndex::with_simple_tokenizer(BM25Config::default(), minimum);
            index.insert(NodeId::new(1), text);
            Arc::new(RwLock::new(index))
        };
        let keys = [
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "first"),
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "second"),
        ];
        let source = TextIndexSection::new(vec![
            (keys[0].clone(), make(2, "first replacement")),
            (keys[1].clone(), make(4, "second replacement")),
        ]);
        let bytes = source.serialize()?;
        for second_minimum in [3, 4] {
            let first = make(2, "retained sentinel first");
            let second = make(second_minimum, "retained sentinel second");
            let before = [exact_image_bytes(&first), exact_image_bytes(&second)];
            let mut target = TextIndexSection::for_unpublished_recovery(vec![
                (keys[0].clone(), Arc::clone(&first)),
                (keys[1].clone(), Arc::clone(&second)),
            ]);
            let result = target.deserialize(&bytes);
            if second_minimum == 3 {
                assert!(matches!(result, Err(Error::Serialization(message))
                    if message.contains("tokenizer descriptor/image mismatch")));
                assert_eq!(
                    [exact_image_bytes(&first), exact_image_bytes(&second)],
                    before
                );
            } else {
                result?;
                assert_eq!(target.serialize()?, bytes);
            }
        }
        Ok(())
    }

    #[test]
    fn text_section_round_trip() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "rust graph database");
        index.insert(NodeId::new(2), "python web framework");
        index.insert(NodeId::new(3), "rust systems programming");

        let index_arc = Arc::new(RwLock::new(index));
        let section = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Item", "description"),
            Arc::clone(&index_arc),
        )]);

        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty());

        // Restore into a fresh index
        let fresh = InvertedIndex::new(BM25Config::default());
        let fresh_arc = Arc::new(RwLock::new(fresh));
        let mut section2 = TextIndexSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Item", "description"),
            fresh_arc.clone(),
        )]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(fresh_arc.read().len(), 3);
        // 8 unique terms: rust, graph, database, python, web, framework, systems, programming
        assert!(fresh_arc.read().term_count() > 0);
    }

    #[test]
    fn altered_current_version_is_rejected_without_mutating_target() {
        let mut source = InvertedIndex::new(BM25Config::default());
        source.insert(NodeId::new(1), "replacement source");
        let mut bytes = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::new(RwLock::new(source)),
        )])
        .serialize()
        .expect("serialize current text index");
        bytes[0] = 2;

        let mut target = InvertedIndex::new(BM25Config::default());
        target.insert(NodeId::new(99), "retained sentinel");
        let target = Arc::new(RwLock::new(target));
        let before = exact_image_bytes(&target);
        let error = TextIndexSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&target),
        )])
        .deserialize(&bytes)
        .expect_err("altered current version must fail at the outer boundary");

        assert!(error.to_string().contains("unsupported"));
        assert_eq!(exact_image_bytes(&target), before);
    }

    #[test]
    fn predecessor_versions_are_rejected_before_target_mutation() {
        for bytes in [[1_u8, 0], [2, 0], [3, 0], [4, 0]] {
            let mut target = InvertedIndex::new(BM25Config::default());
            target.insert(NodeId::new(99), "retained sentinel");
            let target = Arc::new(RwLock::new(target));
            let before = exact_image_bytes(&target);
            let error = TextIndexSection::for_unpublished_recovery(vec![(
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                Arc::clone(&target),
            )])
            .deserialize(&bytes)
            .expect_err("predecessor text index must be unsupported");

            assert!(error.to_string().contains("unsupported"));
            assert_eq!(exact_image_bytes(&target), before);
        }
    }

    #[test]
    fn v5_round_trip_preserves_update_delete_recreate_history_scores_and_aggregates()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = BM25Config { k1: 1.65, b: 0.35 };
        let mut source = InvertedIndex::with_simple_tokenizer(config.clone(), 4);
        let first = NodeId::new(11);
        let second = NodeId::new(12);
        source.insert_versioned(first, "cat alpha alpha", EpochId::new(1), None);
        source.insert_versioned(second, "alpha theta", EpochId::new(1), None);
        source.insert_versioned(first, "gamma gamma delta", EpochId::new(2), None);
        source.insert_versioned(first, "omega omega delta", EpochId::new(2), None);
        source.insert_versioned(first, "sigma sigma delta", EpochId::new(2), None);
        assert!(source.remove_versioned(second, EpochId::new(3), None));
        source.insert_versioned(second, "epsilon zeta zeta", EpochId::new(4), None);

        let source = Arc::new(RwLock::new(source));
        let bytes = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&source),
        )])
        .serialize()
        .expect("v5 exact temporal image serializes");
        assert_eq!(bytes.first(), Some(&TEXT_SECTION_VERSION));

        let restored = Arc::new(RwLock::new(InvertedIndex::with_simple_tokenizer(
            config.clone(),
            4,
        )));
        TextIndexSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&restored),
        )])
        .deserialize(&bytes)
        .expect("v5 exact temporal image restores");

        let source = source.read();
        let restored = restored.read();
        assert_eq!(restored.config().k1.to_bits(), config.k1.to_bits());
        assert_eq!(restored.config().b.to_bits(), config.b.to_bits());
        assert!(restored.search("cat", 10).is_empty());
        for epoch in 0..=5 {
            let epoch = EpochId::new(epoch);
            assert_eq!(
                restored.total_length_at(epoch, TransactionId::INVALID)?,
                source.total_length_at(epoch, TransactionId::INVALID)?,
                "total length diverged at epoch {epoch}"
            );
            assert_eq!(
                restored.doc_count_at(epoch, TransactionId::INVALID)?,
                source.doc_count_at(epoch, TransactionId::INVALID)?,
                "document count diverged at epoch {epoch}"
            );
            assert_eq!(
                restored.avgdl_at(epoch, TransactionId::INVALID)?.to_bits(),
                source.avgdl_at(epoch, TransactionId::INVALID)?.to_bits(),
                "average length diverged at epoch {epoch}"
            );
            for (node_id, query) in [
                (first, "alpha gamma"),
                (first, "gamma omega sigma delta"),
                (second, "alpha theta epsilon zeta"),
            ] {
                let source_score = source
                    .score_document_visible(
                        node_id,
                        query,
                        epoch,
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits);
                let restored_score = restored
                    .score_document_visible(
                        node_id,
                        query,
                        epoch,
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits);
                assert_eq!(
                    restored_score, source_score,
                    "BM25 score diverged for node {node_id} at epoch {epoch}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn v5_rejects_fabricated_aggregate_history_before_mutating_any_index() {
        for deltas in [vec![1, -1], vec![i64::MAX, i64::MAX, -i64::MAX, -i64::MAX]] {
            let mut source = InvertedIndex::new(BM25Config::default());
            source.insert(NodeId::new(1), "replacement source");
            let bytes = TextIndexSection::new(vec![
                (
                    PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                    Arc::new(RwLock::new(source)),
                ),
                (
                    PhysicalIndexKey::text(GraphPath::root(), "Other", "body"),
                    Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
                ),
            ])
            .serialize()
            .unwrap();
            let mut image = decode_v5_fixture(&bytes);
            image.indexes[1].image.agg_log = deltas
                .into_iter()
                .enumerate()
                .map(|(position, d_total_len)| ExactAggDelta {
                    epoch: EpochId::new(position as u64 + 1),
                    tx: None,
                    d_total_len,
                    d_doc_count: 0,
                })
                .collect();
            let targets = ["Doc", "Other"].map(|label| {
                let mut index = InvertedIndex::new(BM25Config::default());
                index.insert(NodeId::new(99), "retained sentinel");
                (
                    PhysicalIndexKey::text(GraphPath::root(), label, "body"),
                    Arc::new(RwLock::new(index)),
                )
            });
            let before = targets
                .iter()
                .map(|(_, target)| exact_image_bytes(target))
                .collect::<Vec<_>>();
            let error = TextIndexSection::for_unpublished_recovery(targets.to_vec())
                .deserialize(&encode_v5_fixture(&image))
                .expect_err("fabricated aggregates must fail before any index installation");
            assert!(error.to_string().contains("aggregate"));
            assert_eq!(
                targets
                    .iter()
                    .map(|(_, target)| exact_image_bytes(target))
                    .collect::<Vec<_>>(),
                before
            );
        }
    }

    #[test]
    fn v5_gc_round_trip_preserves_retained_history_scores_and_aggregates()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for horizon in [0, 1, 3, 4, 5, 6, 7, 9] {
            let mut source = InvertedIndex::new(BM25Config::default());
            source.insert_versioned(NodeId::new(1), "alpha alpha", EpochId::new(1), None);
            source.insert_versioned(NodeId::new(2), "alpha beta gamma", EpochId::new(2), None);
            source.remove_versioned(NodeId::new(1), EpochId::new(3), None);
            source.insert_versioned(NodeId::new(2), "delta delta", EpochId::new(5), None);
            source.insert_versioned(NodeId::new(2), "alpha delta delta", EpochId::new(5), None);
            source.insert_versioned(
                NodeId::new(3),
                "alpha alpha delta gamma",
                EpochId::new(6),
                None,
            );
            source.remove_versioned(NodeId::new(2), EpochId::new(7), None);
            source.remove_versioned(NodeId::new(3), EpochId::new(9), None);
            let mut expected = Vec::new();
            for (epoch, total, count) in [
                (0, 0, 0),
                (1, 2, 1),
                (2, 5, 2),
                (3, 3, 1),
                (4, 3, 1),
                (5, 3, 1),
                (6, 7, 2),
                (7, 4, 1),
                (8, 4, 1),
                (9, 0, 0),
            ] {
                let score = source
                    .score_document_visible(
                        NodeId::new(2),
                        "alpha delta",
                        EpochId::new(epoch),
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits);
                expected.push((epoch, total, count, score));
            }
            source.gc(EpochId::new(horizon))?;
            assert_eq!(source.retained_from(), EpochId::new(horizon));
            let source = Arc::new(RwLock::new(source));
            let bytes = TextIndexSection::new(vec![(
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                source,
            )])
            .serialize()
            .expect("a real GC baseline with retained updates and deletes is exact");
            let restored = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
            TextIndexSection::for_unpublished_recovery(vec![(
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                Arc::clone(&restored),
            )])
            .deserialize(&bytes)
            .unwrap();
            let mut restored = restored.write();
            assert_eq!(restored.retained_from(), EpochId::new(horizon));
            restored.gc(EpochId::INITIAL)?;
            assert_eq!(restored.retained_from(), EpochId::new(horizon));
            for (epoch, total, count, score) in expected {
                if epoch < horizon {
                    let epoch = EpochId::new(epoch);
                    assert!(
                        restored
                            .total_length_at(epoch, TransactionId::INVALID)
                            .is_err()
                    );
                    assert!(
                        restored
                            .doc_count_at(epoch, TransactionId::INVALID)
                            .is_err()
                    );
                    assert!(restored.avgdl_at(epoch, TransactionId::INVALID).is_err());
                    assert!(
                        restored
                            .score_document_visible(
                                NodeId::new(2),
                                "",
                                epoch,
                                TransactionId::INVALID,
                                None,
                                false
                            )
                            .is_err()
                    );
                    assert!(
                        restored
                            .search_visible(
                                "",
                                0,
                                epoch,
                                TransactionId::INVALID,
                                &[],
                                &Default::default()
                            )
                            .is_err()
                    );
                    continue;
                }
                let epoch = EpochId::new(epoch);
                assert_eq!(
                    restored.total_length_at(epoch, TransactionId::INVALID)?,
                    total
                );
                assert_eq!(restored.doc_count_at(epoch, TransactionId::INVALID)?, count);
                assert_eq!(
                    restored
                        .score_document_visible(
                            NodeId::new(2),
                            "alpha delta",
                            epoch,
                            TransactionId::INVALID,
                            None,
                            false,
                        )?
                        .map(f64::to_bits),
                    score
                );
            }
        }
        Ok(())
    }

    #[test]
    fn v5_rejects_invalid_retention_floor_before_mutating_any_target()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let keys = [
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "first"),
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "second"),
        ];
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for key in &keys {
            let mut source = InvertedIndex::new(BM25Config::default());
            source.insert_versioned(NodeId::new(1), "retained source", EpochId::new(1), None);
            source.gc(EpochId::new(3))?;
            sources.push((key.clone(), Arc::new(RwLock::new(source))));
            let mut target = InvertedIndex::new(BM25Config::default());
            target.insert(NodeId::new(99), "unchanged sentinel");
            targets.push((key.clone(), Arc::new(RwLock::new(target))));
        }
        let bytes = TextIndexSection::new(sources).serialize()?;
        let before = targets
            .iter()
            .map(|(_, target)| exact_image_bytes(target))
            .collect::<Vec<_>>();
        let mut target_section = TextIndexSection::for_unpublished_recovery(targets.clone());
        for floor in [EpochId::PENDING, EpochId::INITIAL] {
            let mut snapshot = decode_v5_fixture(&bytes);
            snapshot.indexes[1].image.retained_from = floor;
            let malformed = encode_v5_fixture(&snapshot);
            if floor == EpochId::PENDING {
                assert!(matches!(TextIndexSection::validate_payload(&malformed),
                    Err(Error::Serialization(message)) if message.contains("retained-from epoch")));
                assert!(TextIndexSection::payload_keys(&malformed).is_err());
            }
            assert!(matches!(
                target_section.deserialize(&malformed),
                Err(Error::Serialization(_))
            ));
            assert_eq!(
                targets
                    .iter()
                    .map(|(_, target)| exact_image_bytes(target))
                    .collect::<Vec<_>>(),
                before
            );
        }
        target_section.deserialize(&bytes)?;
        assert_eq!(target_section.serialize()?, bytes);
        Ok(())
    }

    #[test]
    fn v5_restore_requires_explicitly_unpublished_recovery_targets() {
        let mut source = InvertedIndex::new(BM25Config::default());
        source.insert(NodeId::new(1), "replacement source");
        let bytes = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::new(RwLock::new(source)),
        )])
        .serialize()
        .unwrap();

        let mut target = InvertedIndex::new(BM25Config::default());
        target.insert(NodeId::new(99), "retained sentinel");
        let target = Arc::new(RwLock::new(target));
        let before = exact_image_bytes(&target);
        let error = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&target),
        )])
        .deserialize(&bytes)
        .expect_err("a possibly published target must reject exact v5 installation");
        assert!(error.to_string().contains("unpublished recovery targets"));
        assert_eq!(exact_image_bytes(&target), before);
    }

    #[test]
    fn v5_rejects_bm25_descriptor_mismatch_before_mutating_default_or_named_targets()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let default_key = PhysicalIndexKey::text(GraphPath::root(), "Doc", "body");
        let named_key =
            PhysicalIndexKey::text(GraphPath::from_components(&["tenant"])?, "Doc", "body");
        let expected = BM25Config { k1: 1.7, b: 0.4 };

        let mut default_source = InvertedIndex::new(expected.clone());
        default_source.insert(NodeId::new(1), "default source");
        let mut named_source = InvertedIndex::new(expected.clone());
        named_source.insert(NodeId::new(2), "named source");
        let bytes = TextIndexSection::new(vec![
            (default_key.clone(), Arc::new(RwLock::new(default_source))),
            (named_key.clone(), Arc::new(RwLock::new(named_source))),
        ])
        .serialize()
        .expect("encode scoped exact fixture");

        for mismatched_target in 0..2 {
            let default_config = BM25Config {
                k1: if mismatched_target == 0 {
                    f64::from_bits(expected.k1.to_bits() + 1)
                } else {
                    expected.k1
                },
                b: expected.b,
            };
            let named_config = BM25Config {
                k1: expected.k1,
                b: if mismatched_target == 1 {
                    f64::from_bits(expected.b.to_bits() + 1)
                } else {
                    expected.b
                },
            };
            let mut default_target = InvertedIndex::new(default_config);
            default_target.insert(NodeId::new(101), "default sentinel");
            let default_target = Arc::new(RwLock::new(default_target));
            let mut named_target = InvertedIndex::new(named_config);
            named_target.insert(NodeId::new(102), "named sentinel");
            let named_target = Arc::new(RwLock::new(named_target));
            let default_before = exact_image_bytes(&default_target);
            let named_before = exact_image_bytes(&named_target);

            let mut section = TextIndexSection::for_unpublished_recovery(vec![
                (default_key.clone(), Arc::clone(&default_target)),
                (named_key.clone(), Arc::clone(&named_target)),
            ]);
            let error = section
                .deserialize(&bytes)
                .expect_err("catalog descriptor must match the authoritative v5 image exactly");
            assert!(error.to_string().contains("BM25 descriptor/image mismatch"));
            assert_eq!(exact_image_bytes(&default_target), default_before);
            assert_eq!(exact_image_bytes(&named_target), named_before);
        }
        Ok(())
    }

    #[test]
    fn v5_fails_closed_for_opaque_tokenizers_and_pending_state() {
        let opaque = Arc::new(RwLock::new(InvertedIndex::with_tokenizer(
            BM25Config::default(),
            Box::new(SimpleTokenizer::with_min_length(4)),
        )));
        let error = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "opaque"),
            opaque,
        )])
        .serialize()
        .expect_err("opaque tokenizer cannot silently become SimpleTokenizer");
        let message = error.to_string();
        assert!(message.contains("opaque custom tokenizer"));
        assert!(message.contains("no fallback"));

        let mut pending = InvertedIndex::new(BM25Config::default());
        pending.insert_versioned(
            NodeId::new(21),
            "pending document",
            EpochId::PENDING,
            Some(TransactionId::new(9)),
        );
        let error = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "pending"),
            Arc::new(RwLock::new(pending)),
        )])
        .serialize()
        .expect_err("pending text state cannot enter a committed snapshot");
        assert!(error.to_string().contains("pending transaction state"));
    }

    #[test]
    fn restore_requires_an_exact_one_to_one_key_set_without_mutating_targets() {
        let mut source = InvertedIndex::new(BM25Config::default());
        source.insert(NodeId::new(1), "replacement source");
        let valid_bytes = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::new(RwLock::new(source)),
        )])
        .serialize()
        .unwrap();

        let make_target = || {
            let mut target = InvertedIndex::new(BM25Config::default());
            target.insert(NodeId::new(99), "retained sentinel");
            Arc::new(RwLock::new(target))
        };
        let assert_untouched = |target: &Arc<RwLock<InvertedIndex>>| {
            assert_eq!(target.read().search("sentinel", 10)[0].0, NodeId::new(99));
            assert!(target.read().search("replacement", 10).is_empty());
        };

        let unknown_target = make_target();
        let error = TextIndexSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "other"),
            Arc::clone(&unknown_target),
        )])
        .deserialize(&valid_bytes)
        .expect_err("unknown image key must not be ignored");
        assert!(error.to_string().contains("authoritative key set"));
        assert_untouched(&unknown_target);

        let mut missing_snapshot = decode_v5_fixture(&valid_bytes);
        missing_snapshot.indexes.clear();
        let missing_target = make_target();
        let error = TextIndexSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&missing_target),
        )])
        .deserialize(&encode_v5_fixture(&missing_snapshot))
        .expect_err("missing authoritative image must not leave a default target");
        assert!(error.to_string().contains("index count mismatch"));
        assert_untouched(&missing_target);

        let mut wrong_family_snapshot = decode_v5_fixture(&valid_bytes);
        wrong_family_snapshot.indexes[0].key =
            PhysicalIndexKey::vector(GraphPath::root(), "Doc", "body");
        let wrong_family_target = make_target();
        let error = TextIndexSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&wrong_family_target),
        )])
        .deserialize(&encode_v5_fixture(&wrong_family_snapshot))
        .expect_err("a Vector key in a Text section must fail closed");
        assert!(error.to_string().contains("invalid index key"));
        assert_untouched(&wrong_family_target);
    }

    #[test]
    fn v5_scopes_identical_local_keys_to_default_and_named_graphs_exactly()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let default_key = PhysicalIndexKey::text(GraphPath::root(), "Doc", "body");
        let west_key = PhysicalIndexKey::text(
            GraphPath::from_components(&["tenant/西:🧠"])?,
            "Doc",
            "body",
        );
        let empty_graph_key =
            PhysicalIndexKey::text(GraphPath::from_components(&[""])?, "Doc", "body");

        let make_source = |node: u64, first: &str, second: &str| {
            let mut index = InvertedIndex::with_simple_tokenizer(
                BM25Config {
                    k1: 1.1 + node as f64 / 100.0,
                    b: 0.3,
                },
                2,
            );
            index.insert_versioned(NodeId::new(node), first, EpochId::new(1), None);
            index.insert_versioned(NodeId::new(node), second, EpochId::new(3), None);
            Arc::new(RwLock::new(index))
        };
        let source_default = make_source(11, "default alpha", "default omega");
        let source_west = make_source(12, "west alpha", "west omega");
        let source_empty = make_source(13, "empty alpha", "empty omega");
        let bytes = TextIndexSection::new(vec![
            (west_key.clone(), Arc::clone(&source_west)),
            (default_key.clone(), Arc::clone(&source_default)),
            (empty_graph_key.clone(), Arc::clone(&source_empty)),
        ])
        .serialize()
        .expect("graph-qualified v5 image serializes");
        let snapshot = decode_v5_fixture(&bytes);
        assert_eq!(snapshot.version, TEXT_SECTION_VERSION);
        assert!(
            snapshot
                .indexes
                .windows(2)
                .all(|pair| pair[0].key < pair[1].key),
            "v5 images use one canonical key order"
        );

        let restored_default = Arc::new(RwLock::new(InvertedIndex::with_simple_tokenizer(
            source_default.read().config(),
            2,
        )));
        let restored_west = Arc::new(RwLock::new(InvertedIndex::with_simple_tokenizer(
            source_west.read().config(),
            2,
        )));
        let restored_empty = Arc::new(RwLock::new(InvertedIndex::with_simple_tokenizer(
            source_empty.read().config(),
            2,
        )));
        TextIndexSection::for_unpublished_recovery(vec![
            (empty_graph_key.clone(), Arc::clone(&restored_empty)),
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                Arc::clone(&restored_default),
            ),
            (west_key.clone(), Arc::clone(&restored_west)),
        ])
        .deserialize(&bytes)
        .expect("target order does not affect scoped v5 restore");
        assert_eq!(
            exact_image_bytes(&restored_default),
            exact_image_bytes(&source_default)
        );
        assert_eq!(
            exact_image_bytes(&restored_west),
            exact_image_bytes(&source_west)
        );
        assert_eq!(
            exact_image_bytes(&restored_empty),
            exact_image_bytes(&source_empty)
        );
        assert_eq!(
            restored_default.read().search("default", 10)[0].0,
            NodeId::new(11)
        );
        assert!(restored_default.read().search("west", 10).is_empty());
        assert_eq!(
            restored_west.read().search("west", 10)[0].0,
            NodeId::new(12)
        );
        assert_eq!(
            restored_empty.read().search("empty", 10)[0].0,
            NodeId::new(13)
        );

        let mut wrong_graph = snapshot;
        let west = wrong_graph
            .indexes
            .iter_mut()
            .find(|index| index.key == west_key)
            .expect("west graph image");
        west.key =
            PhysicalIndexKey::text(GraphPath::from_components(&["tenant/east"])?, "Doc", "body");
        let wrong_graph = encode_v5_fixture(&wrong_graph);
        let targets = [
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        ];
        for (position, target) in targets.iter().enumerate() {
            target
                .write()
                .insert(NodeId::new(90 + position as u64), "retained sentinel");
        }
        let before = targets.iter().map(exact_image_bytes).collect::<Vec<_>>();
        let error = TextIndexSection::for_unpublished_recovery(vec![
            (default_key, Arc::clone(&targets[0])),
            (west_key, Arc::clone(&targets[1])),
            (empty_graph_key, Arc::clone(&targets[2])),
        ])
        .deserialize(&wrong_graph)
        .expect_err("a graph-name mismatch must fail before any installation");
        assert!(error.to_string().contains("authoritative key set"));
        assert_eq!(
            targets.iter().map(exact_image_bytes).collect::<Vec<_>>(),
            before
        );
        Ok(())
    }

    #[test]
    fn v5_rejects_key_and_target_aliases_before_mutating_any_index()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let default_key = PhysicalIndexKey::text(GraphPath::root(), "Doc", "body");
        let named_key =
            PhysicalIndexKey::text(GraphPath::from_components(&["tenant"])?, "Doc", "body");
        let mut first = InvertedIndex::new(BM25Config::default());
        first.insert(NodeId::new(1), "first replacement");
        let mut second = InvertedIndex::new(BM25Config::default());
        second.insert(NodeId::new(2), "second replacement");
        let bytes = TextIndexSection::new(vec![
            (default_key.clone(), Arc::new(RwLock::new(first))),
            (named_key.clone(), Arc::new(RwLock::new(second))),
        ])
        .serialize()
        .unwrap();

        let make_target = || {
            let mut target = InvertedIndex::new(BM25Config::default());
            target.insert(NodeId::new(99), "retained sentinel");
            Arc::new(RwLock::new(target))
        };

        let mut duplicate_image = decode_v5_fixture(&bytes);
        duplicate_image
            .indexes
            .iter_mut()
            .find(|index| index.key == named_key)
            .expect("named image")
            .key = PhysicalIndexKey::text(GraphPath::root(), "Doc", "body");
        let first_target = make_target();
        let second_target = make_target();
        let before = [
            exact_image_bytes(&first_target),
            exact_image_bytes(&second_target),
        ];
        let error = TextIndexSection::for_unpublished_recovery(vec![
            (default_key.clone(), Arc::clone(&first_target)),
            (named_key.clone(), Arc::clone(&second_target)),
        ])
        .deserialize(&encode_v5_fixture(&duplicate_image))
        .expect_err("duplicate persisted aliases must fail closed");
        assert!(error.to_string().contains("duplicate index key"));
        assert_eq!(exact_image_bytes(&first_target), before[0]);
        assert_eq!(exact_image_bytes(&second_target), before[1]);

        let first_target = make_target();
        let second_target = make_target();
        let error = TextIndexSection::for_unpublished_recovery(vec![
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                Arc::clone(&first_target),
            ),
            (default_key.clone(), Arc::clone(&second_target)),
        ])
        .deserialize(&bytes)
        .expect_err("duplicate target aliases must fail before lock acquisition");
        assert!(error.to_string().contains("duplicate index key"));
        assert_eq!(
            first_target.read().search("sentinel", 10)[0].0,
            NodeId::new(99)
        );
        assert_eq!(
            second_target.read().search("sentinel", 10)[0].0,
            NodeId::new(99)
        );

        let aliased_target = make_target();
        let before = exact_image_bytes(&aliased_target);
        let error = TextIndexSection::for_unpublished_recovery(vec![
            (default_key, Arc::clone(&aliased_target)),
            (named_key, Arc::clone(&aliased_target)),
        ])
        .deserialize(&bytes)
        .expect_err("two graph scopes cannot alias one physical index");
        assert!(error.to_string().contains("aliases another index gate"));
        assert_eq!(exact_image_bytes(&aliased_target), before);
        Ok(())
    }

    #[test]
    fn corrupt_second_v5_image_fails_before_mutating_the_first_target() {
        let mut first = InvertedIndex::new(BM25Config::default());
        first.insert(NodeId::new(1), "first replacement");
        let mut second = InvertedIndex::new(BM25Config::default());
        second.insert(NodeId::new(2), "second replacement");
        let bytes = TextIndexSection::new(vec![
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "first"),
                Arc::new(RwLock::new(first)),
            ),
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "second"),
                Arc::new(RwLock::new(second)),
            ),
        ])
        .serialize()
        .unwrap();
        let mut corrupt = decode_v5_fixture(&bytes);
        let duplicate = corrupt.indexes[1].image.doc_lengths[0].clone();
        corrupt.indexes[1].image.doc_lengths.push(duplicate);

        let make_target = |node, text| {
            let mut target = InvertedIndex::new(BM25Config::default());
            target.insert(NodeId::new(node), text);
            Arc::new(RwLock::new(target))
        };
        let first = make_target(101, "first sentinel");
        let second = make_target(102, "second sentinel");
        let before = [exact_image_bytes(&first), exact_image_bytes(&second)];
        let error = TextIndexSection::for_unpublished_recovery(vec![
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "first"),
                Arc::clone(&first),
            ),
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "second"),
                Arc::clone(&second),
            ),
        ])
        .deserialize(&encode_v5_fixture(&corrupt))
        .expect_err("corrupt second image must fail before installing the first");
        assert!(error.to_string().contains("invalid"));
        assert_eq!(exact_image_bytes(&first), before[0]);
        assert_eq!(exact_image_bytes(&second), before[1]);
    }

    #[test]
    fn missing_and_orphan_text_sections_fail_closed() {
        let mut target = InvertedIndex::new(BM25Config::default());
        target.insert(NodeId::new(99), "retained sentinel");
        let target = Arc::new(RwLock::new(target));
        let before = exact_image_bytes(&target);
        let error = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&target),
        )])
        .deserialize(&[])
        .expect_err("a catalog target requires an authoritative section");
        assert!(error.to_string().contains("section is missing"));
        assert_eq!(exact_image_bytes(&target), before);

        let mut source = InvertedIndex::new(BM25Config::default());
        source.insert(NodeId::new(1), "orphan source");
        let bytes = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::new(RwLock::new(source)),
        )])
        .serialize()
        .unwrap();
        let error = TextIndexSection::for_unpublished_recovery(vec![])
            .deserialize(&bytes)
            .expect_err("an orphan section cannot be silently skipped");
        assert!(error.to_string().contains("index count mismatch"));
    }

    #[test]
    fn text_payload_validator_accepts_only_the_current_version() {
        let current = TextIndexSection::new(vec![]).serialize().unwrap();
        assert!(TextIndexSection::validate_payload(&current).is_ok());
        assert!(TextIndexSection::validate_payload(&[]).is_err());
        for version in [0, 1, 2, 3, 4, 6, 255] {
            assert!(TextIndexSection::validate_payload(&[version, 0]).is_err());
        }
        assert!(
            TextIndexSection::for_unpublished_recovery(vec![])
                .deserialize(&[])
                .is_err()
        );
    }

    #[test]
    fn text_payload_rejects_noncanonical_varints() {
        let bytes = [TEXT_SECTION_VERSION, 251, 0, 0];
        let error = TextIndexSection::validate_payload(&bytes)
            .expect_err("an overlong index-count varint must be rejected");
        assert!(error.to_string().contains("non-canonical integer"));
    }

    #[test]
    fn text_payloads_preflight_declared_counts_and_strings_without_mutation() {
        let mut hostile_count = vec![TEXT_SECTION_VERSION, 253];
        hostile_count.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut hostile_key = vec![TEXT_SECTION_VERSION, 1, 253];
        hostile_key.extend_from_slice(&u64::MAX.to_le_bytes());

        for (description, bytes) in [
            ("count", hostile_count.as_slice()),
            ("index key", hostile_key.as_slice()),
        ] {
            let error = TextIndexSection::validate_payload(bytes)
                .expect_err("hostile allocation claim must not validate");
            assert!(
                error.to_string().contains(description),
                "unexpected {description} error: {error}"
            );

            let mut sentinel = InvertedIndex::new(BM25Config::default());
            sentinel.insert(NodeId::new(99), "retained sentinel");
            let sentinel = Arc::new(RwLock::new(sentinel));
            let before = exact_image_bytes(&sentinel);
            let mut section = TextIndexSection::for_unpublished_recovery(vec![(
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                Arc::clone(&sentinel),
            )]);
            section
                .deserialize(bytes)
                .expect_err("hostile allocation claim must fail before target mutation");
            assert_eq!(exact_image_bytes(&sentinel), before);
        }
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn multi_index_restore_preflights_every_authority_before_installing_any_image() {
        let mut source_first = InvertedIndex::new(BM25Config::default());
        source_first.insert(NodeId::new(1), "first replacement");
        let mut source_second = InvertedIndex::new(BM25Config::default());
        source_second.insert(NodeId::new(2), "second replacement");
        let bytes = TextIndexSection::new(vec![
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "first"),
                Arc::new(RwLock::new(source_first)),
            ),
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "second"),
                Arc::new(RwLock::new(source_second)),
            ),
        ])
        .serialize()
        .unwrap();

        let mut target_first = InvertedIndex::new(BM25Config::default());
        target_first.insert(NodeId::new(101), "first sentinel");
        let target_first = Arc::new(RwLock::new(target_first));
        let mut target_second = InvertedIndex::new(BM25Config::default());
        target_second.insert(NodeId::new(102), "second sentinel");
        let target_second = Arc::new(RwLock::new(target_second));
        let store = LpgStore::new().unwrap();
        store.add_text_index("Doc", "second", Arc::clone(&target_second));
        assert!(store.seal_unframed_writes(&WriteAuthority::new()));

        let error = TextIndexSection::for_unpublished_recovery(vec![
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "first"),
                Arc::clone(&target_first),
            ),
            (
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "second"),
                Arc::clone(&target_second),
            ),
        ])
        .deserialize(&bytes)
        .expect_err("one unauthorized destination aborts the complete restore");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(
            target_first.read().search("sentinel", 10)[0].0,
            NodeId::new(101)
        );
        assert_eq!(
            target_second.read().search("sentinel", 10)[0].0,
            NodeId::new(102)
        );
        assert!(target_first.read().search("replacement", 10).is_empty());
        assert!(target_second.read().search("replacement", 10).is_empty());
    }

    #[test]
    fn text_section_empty() {
        let section = TextIndexSection::new(vec![]);
        let bytes = section.serialize().expect("serialize should succeed");

        let mut section2 = TextIndexSection::for_unpublished_recovery(vec![]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");
    }

    #[test]
    fn text_section_type() {
        let section = TextIndexSection::new(vec![]);
        assert_eq!(section.section_type(), SectionType::TextIndex);
        assert_eq!(section.version(), TEXT_SECTION_VERSION);
    }

    #[test]
    fn text_section_dirty_tracking() {
        let section = TextIndexSection::new(vec![]);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn sealed_text_section_restore_requires_the_store_authority() {
        let mut source = InvertedIndex::new(BM25Config::default());
        source.insert(NodeId::new(1), "authority scoped text");
        let source = Arc::new(RwLock::new(source));
        let bytes = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Item", "description"),
            source,
        )])
        .serialize()
        .expect("serialize fixture postings");

        let target = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        let store = LpgStore::new().unwrap();
        store.add_text_index("Item", "description", Arc::clone(&target));

        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        assert!(store.seal_unframed_writes(&owner));

        let restore = || {
            let mut section = TextIndexSection::for_unpublished_recovery(vec![(
                PhysicalIndexKey::text(GraphPath::root(), "Item", "description"),
                Arc::clone(&target),
            )]);
            section.deserialize(&bytes)
        };
        let error = restore().expect_err("raw restore must fail closed");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.read().len(), 0);

        let error = with_authority(&foreign, restore)
            .expect_err("a foreign store authority must not restore postings");
        assert!(error.to_string().contains("lacks mutation authority"));
        assert_eq!(target.read().len(), 0);

        with_authority(&owner, || {
            restore().expect("owner may restore during recovery");
        });
        assert_eq!(target.read().len(), 1);

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || panic!("injected restore panic"));
        }));
        let error = restore().expect_err("caught panic must not retain authority");
        assert!(error.to_string().contains("lacks mutation authority"));
    }

    #[test]
    #[cfg(feature = "lpg")]
    fn registered_section_view_pins_one_coherent_persistence_image() {
        let store = Arc::new(LpgStore::new().unwrap());
        let retained = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        store.add_text_index("Doc", "body", Arc::clone(&retained));
        let document = store.create_node(&["Doc"]);
        store.set_node_property(document, "body", "coherent before".into());
        let added_during_snapshot = store.create_node(&["Doc"]);

        let section = TextIndexSection::new(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&retained),
        )]);
        let image = section.indexes[0].1.read();
        assert!(image.is_current_image_persistence_representable());

        let writer_store = Arc::clone(&store);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let writer = thread::spawn(move || {
            started_tx.send(()).unwrap();
            writer_store.set_node_property(added_during_snapshot, "body", "coherent after".into());
            finished_tx.send(()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            finished_rx
                .recv_timeout(Duration::from_millis(100))
                .is_err(),
            "store mutation crossed a section persistence-image read guard"
        );

        // These are the exact observers used by `serialize`; the one paired
        // view guard must keep them on the same concrete target generation.
        let config = image.config();
        let (postings, doc_lengths, total_length) = image.snapshot();
        assert_eq!(config.k1.to_bits(), BM25Config::default().k1.to_bits());
        assert!(!postings.is_empty());
        assert_eq!(doc_lengths.len(), 1);
        assert!(total_length > 0);
        drop(image);

        finished_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        writer.join().unwrap();

        let bytes = section.serialize().expect("serialize coherent post-image");
        let restored = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        let mut restored_section = TextIndexSection::for_unpublished_recovery(vec![(
            PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
            Arc::clone(&restored),
        )]);
        restored_section
            .deserialize(&bytes)
            .expect("restore coherent post-image");
        assert_eq!(restored.read().search("before", 10)[0].0, document);
        assert_eq!(
            restored.read().search("after", 10)[0].0,
            added_during_snapshot
        );
    }
}
